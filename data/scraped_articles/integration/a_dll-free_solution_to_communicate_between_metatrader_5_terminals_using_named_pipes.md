---
title: A DLL-free solution to communicate between MetaTrader 5 terminals using Named Pipes
url: https://www.mql5.com/en/articles/115
categories: Integration, Indicators
relevance_score: 3
scraped_at: 2026-01-23T21:21:57.048660
---

[![](https://www.mql5.com/ff/sh/6zw0dkux8bqt7m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
MQL5 Channels - Messenger for traders\\
\\
Install the app and receive market analytics and trading tips.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=iuciwacmrxvmiibwyujliagqikizpsoo&s=268cbb13914c54b6c5c875db99b154944f6e0122b3400b54c9ac0d4f69f0f0d6&uid=&ref=https://www.mql5.com/en/articles/115&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5071778205607603779)

MetaTrader 5 / Examples


### Introduction

I wondered for some time on possible ways of communication between MetaTrader 5 terminals. My goal was to use tick indicator and display ticks from different quote providers in one of the terminals.

The natural solution was to use separate files on a hard drive. One terminal would write data to the file and the other one would read it. This method though relevant for sending single messages does not seem to be the most effective one for streaming quotes.

Then I came across a [good article](https://www.mql5.com/en/articles/27) by Alexander on how to export quotes to .NET applications using WCF services and when I was about to finish there appeared another [article](https://www.mql5.com/en/articles/73) by Sergeev.

Both articles were close to what I needed but I looked for a DLL-free solution that could be used by different terminals one serving as a Server and the other as a Client. While searching the Web I found a note suggesting that one could use Named Pipes for communication and I read thoroughly [MSDN specification](https://www.mql5.com/go?link=https://msdn.microsoft.com/en-us/library/aa365780(vs.85).aspx "http://msdn.microsoft.com/en-us/library/aa365780(vs.85).aspx") for Interprocess Communication using pipes.

I discovered that Named Pipes support communication over the same computer or over different computers over intranet I decided to go for this approach.

This article introduces Named Pipes communication and describes a process of designing CNamedPipes class. It also includes testing tick indicator streaming between MetaTrader 5 terminals and overall system throughtput.

### 1\. Interprocess Communication using Named Pipes

When we think of a typical pipe we imagine a sort of a cylinder that is used to convey media. This is also a term used for one of the means of interprocess communication on an operating system. You could simply imagine a pipe that connects two processes, in our case MetaTrader 5 terminals that exchange data.

Pipes can be anonymous or named. The are two main differences between them: first one is that anonymous pipes cannot be used over a network and the second that two processes must be related. That is one process must be a parent and the other one child process. Named pipes do not have this limitation.

In order to communicate using pipes a server process must setup a pipe with a known name. Pipe name is a string and must be in a form of \\\servername\\pipe\\pipename. If pipes are used on the same computer, servername can be ommited and a dot can be put instead:  \\\.\\pipe\ _pipename_.

The client that tries to connect to a pipe must know its name. I am using a name convention of  \\\.\\pipe\\mt\[account\_number\] in order to distinguish terminals, but naming convention can be arbitrairly changed.

### 2\. Implementing CNamedPipes class

I will start with a short description of low level mechanism of creating and connecting to a named pipe. On Windows operating systems all functions that handle pipes are available through kernel32.dll library. Function instantiating a named pipe on the server side is [CreateNamedPipe()](https://www.mql5.com/go?link=http://msdn.microsoft.com/en-us/library/aa365150(v=VS.85).aspx "http://msdn.microsoft.com/en-us/library/aa365150(v=VS.85).aspx") **.**

After the pipe is created, server calls [ConnectNamedPipe()](https://www.mql5.com/go?link=https://msdn.microsoft.com/en-us/library/aa365146(v=VS.85).aspx "http://msdn.microsoft.com/en-us/library/aa365146(v=VS.85).aspx") function to wait for a client to connect. If connection is successfull, [ConnectNamedPipe()](https://www.mql5.com/go?link=https://msdn.microsoft.com/en-us/library/aa365146(v=VS.85).aspx "http://msdn.microsoft.com/en-us/library/aa365146(v=VS.85).aspx") returns non-zero integer. It is possible though, that the client successfully connected after calling [CreateNamedPipe()](https://www.mql5.com/go?link=http://msdn.microsoft.com/en-us/library/aa365150(v=VS.85).aspx "http://msdn.microsoft.com/en-us/library/aa365150(v=VS.85).aspx") and before [ConnectNamedPipe()](https://www.mql5.com/go?link=https://msdn.microsoft.com/en-us/library/aa365146(v=VS.85).aspx "http://msdn.microsoft.com/en-us/library/aa365146(v=VS.85).aspx") was called. In this case [ConnectNamedPipe()](https://www.mql5.com/go?link=https://msdn.microsoft.com/en-us/library/aa365146(v=VS.85).aspx "http://msdn.microsoft.com/en-us/library/aa365146(v=VS.85).aspx") returns zero, and [GetLastError()](https://www.mql5.com/en/docs/check/getlasterror) returns error 535 (0X217) : ERROR\_PIPE\_CONNECTED.

Writing to and reading from a pipe is achieved with the same functions as for file access:

```
BOOL WINAPI ReadFile(
  __in         HANDLE hFile,
  __out        LPVOID lpBuffer,
  __in         DWORD nNumberOfBytesToRead,
  __out_opt    LPDWORD lpNumberOfBytesRead,
  __inout_opt  LPOVERLAPPED lpOverlapped
);
```

```
BOOL WINAPI WriteFile(
  __in         HANDLE hFile,
  __in         LPCVOID lpBuffer,
  __in         DWORD nNumberOfBytesToWrite,
  __out_opt    LPDWORD lpNumberOfBytesWritten,
  __inout_opt  LPOVERLAPPED lpOverlapped
);
```

Having learned about named pipes I designed CNamedPipes class in order to hide the underlying low level instructions.

Now it is enough to put **CNamedPipes.mqh** file in appropriate (/include) folder of the terminal and include it in the source code and declare a CNamedPipe object.

The class I designed exposes a few basic methods to handle named pipes:

Create(), Connect(), Disconnect(), Open(), Close(), WriteUnicode(), ReadUnicode(), WriteANSI(), ReadANSI(), WriteTick(), ReadTick()

The class can be further extended according to additional requirements.

The Create() method tries to create a pipe with a given name. To simplify the connection between terminals, the input parameter 'account' is the account number of a client that will use a pipe.

If account name is not entered the method tries to open a pipe with a current terminal's account number. The Create() function returns true is pipe was successfully created.

```
//+------------------------------------------------------------------+
/// Create() : try to create a new instance of Named Pipe
/// \param account - source terminal account number
/// \return true - if created, false otherwise                                                                |
//+------------------------------------------------------------------+
bool CNamedPipe::Create(int account=0)
  {
   if(account==0)
      pipeNumber=IntegerToString(AccountInfoInteger(ACCOUNT_LOGIN));
   else
      pipeNumber=IntegerToString(account);

   string fullPipeName=pipeNamePrefix+pipeNumber;

   hPipe=CreateNamedPipeW(fullPipeName,
                          (int)GENERIC_READ|GENERIC_WRITE|(ENUM_PIPE_ACCESS)PIPE_ACCESS_DUPLEX,
                          (ENUM_PIPE_MODE)PIPE_TYPE_RW_BYTE,PIPE_UNLIMITED_INSTANCES,
                          BufferSize*sizeof(ushort),BufferSize*sizeof(ushort),0,NULL);

   if(hPipe==INVALID_HANDLE_VALUE) return false;
   else
      return true;

  }
```

The Connect() method waits for a client to connect to a pipe. It returns true if the client successfully connected to a pipe.

```
//+------------------------------------------------------------------+
/// Connect() : wait for a client to connect to a pipe
/// \return true - if connected, false otherwise.
//+------------------------------------------------------------------+
bool CNamedPipe::Connect(void)
  {
   if(ConnectNamedPipe(hPipe,NULL)==false)
      return(kernel32::GetLastError()==ERROR_PIPE_CONNECTED);
   else return true;
  }
```

The Disconnect() method disconnects server from a pipe.

```
//+------------------------------------------------------------------+
/// Disconnect(): disconnect from a pipe
/// \return true - if disconnected, false otherwise
//+------------------------------------------------------------------+
bool CNamedPipe::Disconnect(void)
  {
   return DisconnectNamedPipe(hPipe);
  }
```

The Open() method should be used by a client, it tries to open a previously created pipe. It returns true if pipe opening was successful.  It returns false if for some reason it could not connect to created pipe within 5 seconds timeout or if opening pipe failed.

```
//+------------------------------------------------------------------+
/// Open() : try to open previously created pipe
/// \param account - source terminal account number
/// \return true - if successfull, false otherwise.
//+------------------------------------------------------------------+
bool CNamedPipe::Open(int account=0)
  {
   if(account==0)
      pipeName=IntegerToString(AccountInfoInteger(ACCOUNT_LOGIN));
   else
      pipeName=IntegerToString(account);

   string fullPipeName=pipeNamePrefix+pipeName;

   if(hPipe==INVALID_HANDLE_VALUE)
     {
      if(WaitNamedPipeW(fullPipeName,5000)==0)
        {
         Print("Pipe "+fullPipeName+" not available...");
         return false;
        }

      hPipe=CreateFileW(fullPipeName,GENERIC_READ|GENERIC_WRITE,0,NULL,OPEN_EXISTING,0,NULL);
      if(hPipe==INVALID_HANDLE_VALUE)
        {
         Print("Pipe open failed");
         return false;
        }

     }
   return true;
  }
```

The Close() method closes the pipe handle.

```
//+------------------------------------------------------------------+
/// Close() : close pipe handle
/// \return 0 if successfull, non-zero otherwise
//+------------------------------------------------------------------+
int CNamedPipe::Close(void)
  {
   return CloseHandle(hPipe);
  }
```

The next six methods are used to read and write through pipes. First two pairs handle strings in Unicode and ANSI formats, both can be used to send commands or messages between terminals.

The string variable in MQL5 is stored as an object that contains
Unicode, therefore the natural way was to provide Unicode methods, but since MQL5 provides UnicodeToANSI methods I also implemented ANSI string communication. The last two methods handle sending and receiving [MqlTick](https://www.mql5.com/en/docs/constants/structures/mqltick) object through a named pipe.

The WriteUnicode() method writes the message consisting of Unicode characters. Since every character consists of two bytes, it sends as an array of ushort to a pipe.

```
//+------------------------------------------------------------------+
/// WriteUnicode() : write Unicode string to a pipe
/// \param message - string to send
/// \return number of bytes written to a pipe
//+------------------------------------------------------------------+
int CNamedPipe::WriteUnicode(string message)
  {
   int ushortsToWrite, bytesWritten;
   ushort UNICODEarray[];
   ushortsToWrite = StringToShortArray(message, UNICODEarray);
   WriteFile(hPipe,ushortsToWrite,sizeof(int),bytesWritten,0);
   WriteFile(hPipe,UNICODEarray,ushortsToWrite*sizeof(ushort),bytesWritten,0);
   return bytesWritten;
  }
```

The ReadUnicode() method receives array of ushorts and returns a string object.

```
//+------------------------------------------------------------------+
/// ReadUnicode(): read unicode string from a pipe
/// \return unicode string (MQL5 string)
//+------------------------------------------------------------------+
string CNamedPipe::ReadUnicode(void)
  {
   string ret;
   ushort UNICODEarray[STR_SIZE*sizeof(uint)];
   int bytesRead, ushortsToRead;

   ReadFile(hPipe,ushortsToRead,sizeof(int),bytesRead,0);
   ReadFile(hPipe,UNICODEarray,ushortsToRead*sizeof(ushort),bytesRead,0);
   if(bytesRead!=0)
      ret = ShortArrayToString(UNICODEarray);

   return ret;
  }
```

The WriteANSI() method writes ANSI uchar array into a pipe.

```
//+------------------------------------------------------------------+
/// WriteANSI() : write ANSI string to a pipe
/// \param message - string to send
/// \return number of bytes written to a pipe                                                                  |
//+------------------------------------------------------------------+
int CNamedPipe::WriteANSI(string message)
  {
   int bytesToWrite, bytesWritten;
   uchar ANSIarray[];
   bytesToWrite = StringToCharArray(message, ANSIarray);
   WriteFile(hPipe,bytesToWrite,sizeof(int),bytesWritten,0);
   WriteFile(hPipe,ANSIarray,bytesToWrite,bytesWritten,0);
   return bytesWritten;
  }
```

The ReadANSI() method reads uchar array from a pipe and returns a string object.

```
//+------------------------------------------------------------------+
/// ReadANSI(): read ANSI string from a pipe
/// \return unicode string (MQL5 string)
//+------------------------------------------------------------------+
string CNamedPipe::ReadANSI(void)
  {
   string ret;
   uchar ANSIarray[STR_SIZE];
   int bytesRead, bytesToRead;

   ReadFile(hPipe,bytesToRead,sizeof(int),bytesRead,0);
   ReadFile(hPipe,ANSIarray,bytesToRead,bytesRead,0);
   if(bytesRead!=0)
      ret = CharArrayToString(ANSIarray);

   return ret;
  }
```

WriteTick() method writes a single [MqlTick](https://www.mql5.com/en/docs/constants/structures/mqltick) object to a pipe.

```
//+------------------------------------------------------------------+
/// WriteTick() : write MqlTick to a pipe
/// \param MqlTick to send
/// \return true if tick was written correctly, false otherwise
//+------------------------------------------------------------------+
int CNamedPipe::WriteTick(MqlTick &outgoing)
  {
   int bytesWritten;

   WriteFile(hPipe,outgoing,MQLTICK_SIZE,bytesWritten,0);

   return bytesWritten;
  }
```

ReadTick() method reads a single [MqlTick](https://www.mql5.com/en/docs/constants/structures/mqltick) object from a pipe. If a pipe is empty it returns 0, if not it should return a number of bytes of [MqlTick](https://www.mql5.com/en/docs/constants/structures/mqltick) object.

```
//+------------------------------------------------------------------+
/// ReadTick() : read MqlTick from a pipe
/// \return true if tick was read correctly, false otherwise
//+------------------------------------------------------------------+
int CNamedPipe::ReadTick(MqlTick &incoming)
  {
   int bytesRead;

   ReadFile(hPipe,incoming,MQLTICK_SIZE,bytesRead,NULL);

   return bytesRead;
  }
//+------------------------------------------------------------------+
```

Since the basic methods for handling named pipes are known we can start with two MQL programs: a simple script for receiving quotes and an indicator for sending quotes.

### 3\. Server Script for Receiving Quotes

The example server initiates named pipe and waits for a client to connect. After client disconnect it displays how many ticks were received by that client in total and waits for a new client to connect. If client disconnected and server finds a global variable 'gvar0' it exits. If 'gvar0' variable does not exist one can manually stop the server by right-clicking on a chart and choosing Expert List option.

```
//+------------------------------------------------------------------+
//|                                              NamedPipeServer.mq5 |
//|                                      Copyright 2010, Investeo.pl |
//|                                                http:/Investeo.pl |
//+------------------------------------------------------------------+
#property copyright "Copyright 2010, Investeo.pl"
#property link      "http:/Investeo.pl"
#property version   "1.00"

#include <CNamedPipes.mqh>

CNamedPipe pipe;
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
   bool tickReceived;
   int i=0;

   if(pipe.Create()==true)
      while (GlobalVariableCheck("gvar0")==false)
        {
         Print("Waiting for client to connect.");
         if (pipe.Connect()==true)
            Print("Pipe connected");
         while(true)
           {
            do
              {
               tickReceived=pipe.ReadTick();

               if(tickReceived==false)
                 {
                  if(GetError()==ERROR_BROKEN_PIPE)
                    {
                     Print("Client disconnected from pipe "+pipe.Name());
                     pipe.Disconnect();
                     break;
                    }
                 } else i++;
                  Print(IntegerToString(i) + "ticks received.");
              } while(tickReceived==true);
            if (i>0)
            {
               Print(IntegerToString(i) + "ticks received.");
               i=0;
            };
            if(GlobalVariableCheck("gvar0")==true || (GetError()==ERROR_BROKEN_PIPE)) break;
           }

        }

 pipe.Close();
  }
```

### 4\. Simple Indicator for Sending Quotes

The indicator for sending quotes opens a pipe inside [OnInit()](https://www.mql5.com/en/docs/basis/function/events#oninit) method and sends a single [MqlTick](https://www.mql5.com/en/docs/constants/structures/mqltick) each time [OnCalculate()](https://www.mql5.com/en/docs/basis/function/events#oncalculate) method is triggered:

```
//+------------------------------------------------------------------+
//|                                        SendTickPipeIndicator.mq5 |
//|                                      Copyright 2010, Investeo.pl |
//|                                                http:/Investeo.pl |
//+------------------------------------------------------------------+
#property copyright "Copyright 2010, Investeo.pl"
#property link      "http:/Investeo.pl"
#property version   "1.00"
#property indicator_chart_window

#include <CNamedPipes.mqh>

CNamedPipe pipe;
int ctx;

//+------------------------------------------------------------------+
//| Custom indicator initialization function                         |
//+------------------------------------------------------------------+
int OnInit()
  {

   while (!pipe.Open(AccountInfoInteger(ACCOUNT_LOGIN)))
   {
      Print("Pipe not created, retrying in 5 seconds...");
      if (GlobalVariableCheck("gvar1")==true) break;
   }

   ctx = 0;
   return(0);
  }
//+------------------------------------------------------------------+
//| Custom indicator iteration function                              |
//+------------------------------------------------------------------+
int OnCalculate(const int rates_total,
                const int prev_calculated,
                const datetime& time[],
                const double& open[],
                const double& high[],
                const double& low[],
                const double& close[],
                const long& tick_volume[],
                const long& volume[],
                const int& spread[])
  {
   ctx++;
   MqlTick outgoing;
   SymbolInfoTick(Symbol(), outgoing);
   pipe.WriteTick(outgoing);
   Print(IntegerToString(ctx)+" tick send to server by SendTickPipeClick.");
   return(rates_total);
  }
//+------------------------------------------------------------------+
```

### 5\. Running Tick Indicators from Multiple Providers in a Single Client Terminal

The situation got more complicated as I wanted to display incoming quotes in separate tick indicators. I achieved this by implementing pipe server that broadcasts incoming ticks to tick indicator by triggering [EventChartCustom()](https://www.mql5.com/en/docs/eventfunctions/eventchartcustom) method.

Bid and ask quotes are sent as a single string divided by a semicolon e.g. '1.20223;120225'. The appropriate indicator handles a custom event inside [OnChartEvent()](https://www.mql5.com/en/docs/basis/function/events#onchartevent) and displays a tick chart.

```
//+------------------------------------------------------------------+
//|                                   NamedPipeServerBroadcaster.mq5 |
//|                                      Copyright 2010, Investeo.pl |
//|                                                http:/Investeo.pl |
//+------------------------------------------------------------------+
#property copyright "Copyright 2010, Investeo.pl"
#property link      "http:/Investeo.pl"
#property version   "1.00"
#property script_show_inputs
#include <CNamedPipes.mqh>

input int account = 0;

CNamedPipe pipe;
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
   bool tickReceived;
   int i=0;

   if(pipe.Create(account)==true)
      while(GlobalVariableCheck("gvar0")==false)
        {
         if(pipe.Connect()==true)
            Print("Pipe connected");
            i=0;
         while(true)
           {
            do
              {
               tickReceived=pipe.ReadTick();
               if(tickReceived==false)
                 {
                  if(kernel32::GetLastError()==ERROR_BROKEN_PIPE)
                    {
                     Print("Client disconnected from pipe "+pipe.GetPipeName());
                     pipe.Disconnect();
                     break;
                    }
                  } else  {
                   i++; Print(IntegerToString(i)+" ticks received BY server.");
                  string bidask=DoubleToString(pipe.incoming.bid)+";"+DoubleToString(pipe.incoming.ask);
                  long currChart=ChartFirst(); int chart=0;
                  while(chart<100)
                    {
                     EventChartCustom(currChart,6666,0,(double)account,bidask);
                     currChart=ChartNext(currChart);
                     if(currChart==0) break;         // Reached the end of the charts list
                     chart++;
                    }
                     if(GlobalVariableCheck("gvar0")==true || (kernel32::GetLastError()==ERROR_BROKEN_PIPE)) break;

                 }
              }
            while(tickReceived==true);
            if(i>0)
              {
               Print(IntegerToString(i)+"ticks received.");
               i=0;
              };
            if(GlobalVariableCheck("gvar0")==true || (kernel32::GetLastError()==ERROR_BROKEN_PIPE)) break;
            Sleep(100);
           }

        }

  pipe.Close();
  }
```

In order to display ticks I chose tick indicator placed in [MQLmagazine](https://www.mql5.com/go?link=http://mqlmagazine.com/metatrader5/tick-data-charting-why-not-a-level-ii-metatrader/ "http://mqlmagazine.com/metatrader5/tick-data-charting-why-not-a-level-ii-metatrader/"), but instead of [OnCalculate()](https://www.mql5.com/en/docs/basis/function/events#oncalculate) method I implemented processing inside [OnChartEvent()](https://www.mql5.com/en/docs/basis/function/events#onchartevent) and added conditional instructions. A quote is accepted for processing only if dparam parameter is equal to pipe number and event id equals [CHARTEVENT\_CUSTOM](https://www.mql5.com/en/docs/eventfunctions/eventchartcustom) +6666:

```
void OnChartEvent(const int id,
                  const long &lparam,
                  const double &dparam,
                  const string &sparam)
  {
  if (dparam==(double)incomingPipe)
   if(id>CHARTEVENT_CUSTOM)
     {
      if(id==CHARTEVENT_CUSTOM+6666)
        {
        // Process incoming tick
        }
     } else
        {
         // Handle the user event
        }
  }
```

On the screenshot below there are three tick indicators.

Two of them display ticks received through pipes and a third indicator that does not use pipes was run to check if no ticks were lost.

![Tick indicator with data from different terminals](https://c.mql5.com/2/1/indicatorscut.png)

Fig. 1 Quotes received through a named pipe

Please find attached a screencast with comments on how I run the indicators:

YouTube

Fig. 2 Screencast describing indicators setup

### 6\. Testing System Throughput

Since pipes use shared memory the communication is very fast. I conducted tests of sending 100 000 and 1 000 000 ticks in a row between two MetaTrader 5 terminals. The sending script uses WriteTick() function and measures timespan using [GetTickCount()](https://www.mql5.com/en/docs/common/gettickcount):

```
   Print("Sending...");
   uint start = GetTickCount();
   for (int i=0;i<100000;i++)
      pipe.WriteTick(outgoing);
   uint stop = GetTickCount();
   Print("Sending took" + IntegerToString(stop-start) + " [ms]");
   pipe.Close();
```

The server reads incoming quotes. Timespan is measured from the first incoming quote until client disconnects:

```
//+------------------------------------------------------------------+
//|                                          SpeedTestPipeServer.mq5 |
//|                                      Copyright 2010, Investeo.pl |
//|                                                http:/Investeo.pl |
//+------------------------------------------------------------------+
#property copyright "Copyright 2010, Investeo.pl"
#property link      "http:/Investeo.pl"
#property version   "1.00"

#property script_show_inputs
#include <CNamedPipes.mqh>

input int account=0;
bool tickReceived;
uint start,stop;

CNamedPipe pipe;
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
   int i=0;
   if(pipe.Create(account)==true)
      if(pipe.Connect()==true)
         Print("Pipe connected");

   do
     {
      tickReceived=pipe.ReadTick();
      if(i==0) start=GetTickCount();
      if(tickReceived==false)
        {
         if(kernel32::GetLastError()==ERROR_BROKEN_PIPE)
           {
            Print("Client disconnected from pipe "+pipe.GetPipeName());
            pipe.Disconnect();
            break;
           }
        }
      else i++;
     }
   while(tickReceived==true);
   stop=GetTickCount();

   if(i>0)
     {
      Print(IntegerToString(i)+" ticks received.");
      i=0;
     };

   pipe.Close();
   Print("Server: receiving took "+IntegerToString(stop-start)+" [ms]");

  }
//+------------------------------------------------------------------+
```

The results for 10 sample runs were as follows:

| Run | Quotes | Send time  \[ms\] | Receive time  \[ms\] |
| --- | --- | --- | --- |
| 1 | 100000 | 624 | 624 |
| 2 | 100000 | 702 | 702 |
| 3 | 100000 | 687 | 687 |
| 4 | 100000 | 592 | 608 |
| 5 | 100000 | 624 | 624 |
| 6 | 1000000 | 5616 | 5616 |
| 7 | 1000000 | 5788 | 5788 |
| 8 | 1000000 | 5928 | 5913 |
| 9 | 1000000 | 5772 | 5756 |
| 10 | 1000000 | 5710 | 5710 |

Table 1 Throughput speed measurements

The average speed of sending 1 000 000 quotes was about 170 000 ticks/second on a laptop running Windows Vista with 2.0GHz T4200 CPU and 3GB RAM.

### Conclusion

I presented a method of communication between MetaTrader 5 terminals using Names Pipes. The method turned out to be sufficient enough for sending real-time quotes between terminals.

CNamedPipes class can be further extended according to additonal requirements, for example to make possible hedging on two independent accounts. Please find attached CNamedPipe class source code with documentation in chm format and other source code I implemented for writing the article.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/115.zip "Download all attachments in the single ZIP archive")

[cnamedpipes.mqh](https://www.mql5.com/en/articles/download/115/cnamedpipes.mqh "Download cnamedpipes.mqh")(11.44 KB)

[namedpipeserverbroadcaster.mq5](https://www.mql5.com/en/articles/download/115/namedpipeserverbroadcaster.mq5 "Download namedpipeserverbroadcaster.mq5")(3.57 KB)

[speedtestpipeserver.mq5](https://www.mql5.com/en/articles/download/115/speedtestpipeserver.mq5 "Download speedtestpipeserver.mq5")(1.65 KB)

[speedtestpipeclient.mq5](https://www.mql5.com/en/articles/download/115/speedtestpipeclient.mq5 "Download speedtestpipeclient.mq5")(1.3 KB)

[cnamedpipes-doc.zip](https://www.mql5.com/en/articles/download/115/cnamedpipes-doc.zip "Download cnamedpipes-doc.zip")(69.65 KB)

[sendtickpipeindicator.mq5](https://www.mql5.com/en/articles/download/115/sendtickpipeindicator.mq5 "Download sendtickpipeindicator.mq5")(1.89 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Securing MQL5 code: Password Protection, Key Generators, Time-limits, Remote Licenses and Advanced EA License Key Encryption Techniques](https://www.mql5.com/en/articles/359)
- [MQL5-RPC. Remote Procedure Calls from MQL5: Web Service Access and XML-RPC ATC Analyzer for Fun and Profit](https://www.mql5.com/en/articles/342)
- [Applying The Fisher Transform and Inverse Fisher Transform to Markets Analysis in MetaTrader 5](https://www.mql5.com/en/articles/303)
- [Advanced Adaptive Indicators Theory and Implementation in MQL5](https://www.mql5.com/en/articles/288)
- [Using MetaTrader 5 Indicators with ENCOG Machine Learning Framework for Timeseries Prediction](https://www.mql5.com/en/articles/252)
- [Exposing C# code to MQL5 using unmanaged exports](https://www.mql5.com/en/articles/249)
- [Moving Mini-Max: a New Indicator for Technical Analysis and Its Implementation in MQL5](https://www.mql5.com/en/articles/238)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/1276)**
(25)


![Vladimir Simakov](https://c.mql5.com/avatar/avatar_na2.png)

**[Vladimir Simakov](https://www.mql5.com/en/users/simakovva)**
\|
3 Feb 2019 at 13:28

Taking into account the fact that mql4/5 lacks the possibility of multithreading, making a server on mql4/5 is like watching [a](https://www.mql5.com/en/docs/runtime/running "MQL5 Documentation: Programme Execution") tube TV (it seems to work, but the feeling is not the same. That's why it's better to write a separate server and use it to transfer data between mql programs.


![Flying Dutchman](https://c.mql5.com/avatar/avatar_na2.png)

**[Flying Dutchman](https://www.mql5.com/en/users/rjmjanssen)**
\|
19 Jun 2023 at 20:59

Hi, is this still a working [product](https://www.metatrader5.com/en/terminal/help/fundamental/economic_indicators_usa/usa_productivity "Productivity")? If not, can it be updated to work under the latest MT5 build? Right now it's throwing an 'Access Violation'


![sonaht](https://c.mql5.com/avatar/2019/4/5CAA4346-ABDA.png)

**[sonaht](https://www.mql5.com/en/users/sonaht)**
\|
28 Jul 2023 at 19:24

I also faced access violation errors when calling the methods writeTick, readTick... and so on. **Find the  fixed CNamedPipes file attached**.

I also added the WriteDouble Method to the **CNamedPipe** Class:

```
#import "kernel32.dll"
...
int WriteFile(ulong fileHandle,double &var,int bytes,int &numOfBytes,ulong overlapped);
...
#import

...

bool CNamedPipe::WriteDouble(double data)
  {
   int data_size = sizeof(data);
//--- check parameters
   if(!data || data_size<1) return(false);

//--- send data
   int written;
   if(!WriteFile(hPipe,data,data_size,written,NULL) || written!=data_size)
      return(false);
//--- ok
   return(true);
  }
```

Side Note: To read/write on Client Side (e.g from another Metatrader Terminal ), I just used the standard MQL5 FilePipe Library. Examples how to use them can be downloaded here:

[https://www.mql5.com/en/articles/503](https://www.mql5.com/en/articles/503 "https://www.mql5.com/en/articles/503")

![Emilio Reale](https://c.mql5.com/avatar/2017/10/59D23001-E9F2.jpg)

**[Emilio Reale](https://www.mql5.com/en/users/emiliostefano)**
\|
23 Nov 2023 at 18:14

fantastic work


![Adam John Bradley](https://c.mql5.com/avatar/avatar_na2.png)

**[Adam John Bradley](https://www.mql5.com/en/users/adam_j_bradley)**
\|
18 Sep 2024 at 15:17

It's been a while since there was an update here... I'm seeing "Access Violation" error when I attach the indicator to the chart. Guess there's no way to make this work.


![Genetic Algorithms - It's Easy!](https://c.mql5.com/2/0/genetic_algorithms_MQL5.png)[Genetic Algorithms - It's Easy!](https://www.mql5.com/en/articles/55)

In this article the author talks about evolutionary calculations with the use of a personally developed genetic algorithm. He demonstrates the functioning of the algorithm, using examples, and provides practical recommendations for its usage.

![Guide to writing a DLL for MQL5 in Delphi](https://c.mql5.com/2/0/delphi_DLL_MQL5__1.png)[Guide to writing a DLL for MQL5 in Delphi](https://www.mql5.com/en/articles/96)

The article examines the mechanism of creating a DLL module, using the popular programming language of ObjectPascal, within a Delphi programming environment. The materials, provided in this article, are designed to primarily target beginner programmers, who are working with problems, which breach the boundaries of the embedded programming language of MQL5, by connecting the outside DLL modules.

![Creating Information Boards Using Standard Library Classes and Google Chart API](https://c.mql5.com/2/0/info_panel_MQL5.png)[Creating Information Boards Using Standard Library Classes and Google Chart API](https://www.mql5.com/en/articles/102)

The MQL5 programming language primarily targets the creation of automated trading systems and complex instruments of technical analyses. But aside from this, it allows us to create interesting information systems for tracking market situations, and provides a return connection with the trader. The article describes the MQL5 Standard Library components, and shows examples of their use in practice for reaching these objectives. It also demonstrates an example of using Google Chart API for the creation of charts.

![An Example of a Trading System Based on a Heiken-Ashi Indicator](https://c.mql5.com/2/0/Heikin_Ashi_MQL5.png)[An Example of a Trading System Based on a Heiken-Ashi Indicator](https://www.mql5.com/en/articles/91)

In this article we look into the question of using a Heiken-Ashi indicator in trading. Based on this indicator, a simple trading system is considered and an MQL5 Expert Advisor is written. Trading operations are implemented on the bases of classes of the Standard class library. The testing results of the reviewed trading strategy, are based on the history, and obtained using the built-in MetaTrader 5 strategy tester, are provided in the article.

[![](https://www.mql5.com/ff/si/dwquj7nmuxsb297n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F994%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dhow.use.vps%26utm_content%3Drent.vps%26utm_campaign%3D0622.MQL5.com.Internal&a=enhudadyvnrfwcvutcjazdvrxjyrzhyf&s=8f8a773cbff7e7ca26346dfb885f4f329a8b1f2c99472f858f32c0b06b662998&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=aricrwtzuhvfwtlswaaatxxmlqvasugc&ssn=1769192515360603986&ssn_dr=0&ssn_sr=0&fv_date=1769192515&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F115&back_ref=https%3A%2F%2Fwww.google.com%2F&title=A%20DLL-free%20solution%20to%20communicate%20between%20MetaTrader%205%20terminals%20using%20Named%20Pipes%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176919251571867438&fz_uniq=5071778205607603779&sv=2552)

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