---
title: Communicating With MetaTrader 5 Using Named Pipes Without Using DLLs
url: https://www.mql5.com/en/articles/503
categories: Integration
relevance_score: 3
scraped_at: 2026-01-23T21:20:19.052409
---

[![](https://www.mql5.com/ff/sh/bhdtjfb1zry09943z2/267b575d2182c180804d340af38ce02c.jpg)\\
Trade from your iPhone or Android device\\
\\
You only need an internet connection to use the new powerful MetaTrader 5 Web terminal\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=wtigumvtenarnsocpyfoqnanxrilnbxx&s=ec8c539e52b83881ff2d16eaff6913b25803952eb277cac55f670a102b2edc1f&uid=&ref=https://www.mql5.com/en/articles/503&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=6457849819444489697)

MetaTrader 5 / Examples


### Introduction

Many developers face the same problem - how to get to the trading terminal sandbox without using unsafe DLLs.

One of the easiest and safest method is to use standard Named Pipes that work as normal file operations. They allow you to organize interprocessor client-server communication between programs. Although there is an already published article [A DLL-free solution to communicate between MetaTrader 5 terminals using Named Pipes](https://www.mql5.com/en/articles/115 "A DLL-free solution to communicate between MetaTrader 5 terminals using Named Pipes") on this topic that demonstrates enabling access to DLLs, we will use standard and safe features of client terminal.

You can find more information about named pipes in [MSDN](https://www.mql5.com/go?link=https://msdn.microsoft.com/en-us/library/aa365590.aspx "http://msdn.microsoft.com/en-us/library/aa365590.aspx") library, but we will get down to practical examples in C++ and MQL5. We will implement server, client, data exchange between them and then benchmark performance.

### Server Implementation

Let's code a simple server in C++. A script from the terminal will connect to this server and will exchange data with it. The server core has the following set of WinAPI functions:

- [CreateNamedPipe](https://www.mql5.com/go?link=https://msdn.microsoft.com/en-us/library/aa365150(v=vs.85).aspx "http://msdn.microsoft.com/en-us/library/aa365150(v=vs.85).aspx") \- creates a named pipe.
- [ConnectNamedPipe](https://www.mql5.com/go?link=http://msdn.microsoft.com/en-us/library/aa365146(v=vs.85).aspx "http://msdn.microsoft.com/en-us/library/aa365146(v=vs.85).aspx") \- enables server to wait for client connections.
- [WriteFile](https://www.mql5.com/go?link=https://msdn.microsoft.com/en-us/library/aa365747(v=vs.85).aspx "http://msdn.microsoft.com/en-us/library/aa365747(v=vs.85).aspx") \- writes data to pipe.
- [ReadFile](https://www.mql5.com/go?link=https://msdn.microsoft.com/en-us/library/aa365467(v=vs.85).aspx "http://msdn.microsoft.com/en-us/library/aa365467(v=vs.85).aspx") \- reads data from pipe.
- [FlushFileBuffers](https://www.mql5.com/go?link=https://msdn.microsoft.com/en-us/library/aa364439(v=vs.85).aspx "http://msdn.microsoft.com/en-us/library/aa364439(v=vs.85).aspx") \- flushes accumulated buffers.
- [DisconnectNamedPipe](https://www.mql5.com/go?link=https://msdn.microsoft.com/en-us/library/aa365166(v=vs.85).aspx "http://msdn.microsoft.com/en-us/library/aa365166(v=vs.85).aspx") \- disconnects server.
- [CloseHandle](https://www.mql5.com/go?link=https://msdn.microsoft.com/en-us/library/ms724211(v=vs.85).aspx "http://msdn.microsoft.com/en-us/library/ms724211(v=vs.85).aspx") \- closes handle.

Once a named pipe is opened it returns a file handle that can be used for regular read/write file operations. As a result you get a very simple mechanism that don't require any special knowledge in network operations.

Named pipes have one distinctive feature - they can be both local and network. That is, it's easy to implement a remote server that will accept network connections from client terminals.

Here is a simple example of creating a local server as full-duplex channel that works in the bytes exchange mode:

```
//--- open
CPipeManager manager;

if(!manager.Create(L"\\\\.\\pipe\\MQL5.Pipe.Server"))
   return(-1);

//+------------------------------------------------------------------+
//| Create named pipe                                                |
//+------------------------------------------------------------------+
bool CPipeManager::Create(LPCWSTR pipename)
  {
//--- check parameters
   if(!pipename || *pipename==0) return(false);
//--- close old
   Close();
//--- create named pipe
   m_handle=CreateNamedPipe(pipename,PIPE_ACCESS_DUPLEX,
                            PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT,
                            PIPE_UNLIMITED_INSTANCES,256*1024,256*1024,1000,NULL);

   if(m_handle==INVALID_HANDLE_VALUE)
     {
      wprintf(L"Creating pipe '%s' failed\n",pipename);
      return(false);
     }
//--- ok
   wprintf(L"Pipe '%s' created\n",pipename);
   return(true);
  }
```

To get a client connection you have to use the ConnectNamedPipe function:

```
//+------------------------------------------------------------------+
//| Connect client                                                   |
//+------------------------------------------------------------------+
bool CPipeManager::ConnectClient(void)
  {
//--- pipe exists?
   if(m_handle==INVALID_HANDLE_VALUE) return(false);
//--- connected?
   if(!m_connected)
     {
      //--- connect
      if(ConnectNamedPipe(m_handle,NULL)==0)
        {
         //--- client already connected before ConnectNamedPipe?
         if(GetLastError()!=ERROR_PIPE_CONNECTED)
            return(false);
         //--- ok
        }
      m_connected=true;
     }
//---
   return(true);
  }
```

Data exchange is organized using 4 simple functions:

- CPipeManager::Send(void \*data,size\_t data\_size)
- CPipeManager::Read(void \*data,size\_t data\_size)
- CPipeManager::SendString(LPCSTR command)
- CPipeManager::ReadString(LPSTR answer,size\_t answer\_maxlen)

They allow you to send/receive data as binary data or ANSI text strings in MQL5 compatible mode. Moreover, since CFilePipe in MQL5 opens a file in ANSI mode by default, strings are automatically converted to Unicode on receipt and sending. If your MQL5 program opens a file in Unicode mode (FILE\_UNICODE), then it can exchange Unicode strings (with BOM starting signature).

### Client Implementation

We will write our client in MQL5. It will be able to perform regular file operations using the CFilePipe class from Standard Library. This class is almost identical to the [CFileBin](https://www.mql5.com/en/docs/standardlibrary/fileoperations/cfilebin), but it contains an important verification of data availability in a virtual file before reading this data.

```
//+------------------------------------------------------------------+
//| Wait for incoming data                                           |
//+------------------------------------------------------------------+
bool CFilePipe::WaitForRead(const ulong size)
  {
//--- check handle and stop flag
   while(m_handle!=INVALID_HANDLE && !IsStopped())
     {
      //--- enough data?
      if(FileSize(m_handle)>=size)
         return(true);
      //--- wait a little
      Sleep(1);
     }
//--- failure
   return(false);
  }

//+------------------------------------------------------------------+
//| Read an array of variables of double type                        |
//+------------------------------------------------------------------+
uint CFilePipe::ReadDoubleArray(double &array[],const int start_item,const int items_count)
  {
//--- calculate size
   uint size=ArraySize(array);
   if(items_count!=WHOLE_ARRAY) size=items_count;
//--- check for data
   if(WaitForRead(size*sizeof(double)))
      return FileReadArray(m_handle,array,start_item,items_count);
//--- failure
   return(0);
  }
```

Named pipes have significant differences in implementation of their local and network modes. Without such a verification, network mode operations will always return a read error when sending large amounts of data (over 64K).

Let's connect to the server with two checks: either to remote computer named 'RemoteServerName' or to local machine.

```
void OnStart()
  {
//--- wait for pipe server
   while(!IsStopped())
     {
      if(ExtPipe.Open("\\\\RemoteServerName\\pipe\\MQL5.Pipe.Server",FILE_READ|FILE_WRITE|FILE_BIN)!=INVALID_HANDLE) break;
      if(ExtPipe.Open("\\\\.\\pipe\\MQL5.Pipe.Server",FILE_READ|FILE_WRITE|FILE_BIN)!=INVALID_HANDLE) break;
      Sleep(250);
     }
   Print("Client: pipe opened");
```

### Data Exchange

After successful connection let's send a text string with identification info to the server. Unicode string will be automatically converted into ANSI, since the file is opened in ANSI mode.

```
//--- send welcome message
   if(!ExtPipe.WriteString(__FILE__+" on MQL5 build "+IntegerToString(__MQ5BUILD__)))
     {
      Print("Client: sending welcome message failed");
      return;
     }
```

In response, the server will send its string "Hello from pipe server" and the integer 1234567890. The client once again will send string "Test string" and the integer 1234567890.

```
//--- read data from server
   string        str;
   int           value=0;

   if(!ExtPipe.ReadString(str))
     {
      Print("Client: reading string failed");
      return;
     }
   Print("Server: ",str," received");

   if(!ExtPipe.ReadInteger(value))
     {
      Print("Client: reading integer failed");
      return;
     }
   Print("Server: ",value," received");
//--- send data to server
   if(!ExtPipe.WriteString("Test string"))
     {
      Print("Client: sending string failed");
      return;
     }

   if(!ExtPipe.WriteInteger(value))
     {
      Print("Client: sending integer failed");
      return;
     }
```

OK, we are finished with simple data exchange. Now it's time for performance benchmark.

### Performance Benchmark

As a test, we will send 1 gigabyte of data as an array of the double type numbers in blocks of 8 megabytes from server to client, then check correctness of the blocks and measure the transfer rate.

Here is this code in C++ server:

```
//--- benchmark
   double  volume=0.0;
   double *buffer=new double[1024*1024];   // 8 Mb

   wprintf(L"Server: start benchmark\n");
   if(buffer)
     {
      //--- fill the buffer
      for(size_t j=0;j<1024*1024;j++)
         buffer[j]=j;
      //--- send 8 Mb * 128 = 1024 Mb to client
      DWORD   ticks=GetTickCount();

      for(size_t i=0;i<128;i++)
        {
         //--- setup guard signatures
         buffer[0]=i;
         buffer[1024*1024-1]=i+1024*1024-1;
         //---
         if(!manager.Send(buffer,sizeof(double)*1024*1024))
           {
            wprintf(L"Server: benchmark failed, %d\n",GetLastError());
            break;
           }
         volume+=sizeof(double)*1024*1024;
         wprintf(L".");
        }
      wprintf(L"\n");
      //--- read confirmation
      if(!manager.Read(&value,sizeof(value)) || value!=12345)
         wprintf(L"Server: benchmark confirmation failed\n");
      //--- show statistics
      ticks=GetTickCount()-ticks;
      if(ticks>0)
         wprintf(L"Server: %.0lf Mb sent at %.0lf Mb per second\n",volume/1024/1024,volume/1024/ticks);
      //---
      delete[] buffer;
     }
```

and in MQL5 client:

```
//--- benchmark
   double buffer[];
   double volume=0.0;

   if(ArrayResize(buffer,1024*1024,0)==1024*1024)
     {
      uint  ticks=GetTickCount();
      //--- read 8 Mb * 128 = 1024 Mb from server
      for(int i=0;i<128;i++)
        {
         uint items=ExtPipe.ReadDoubleArray(buffer);
         if(items!=1024*1024)
           {
            Print("Client: benchmark failed after ",volume/1024," Kb, ",items," items received");
            break;
           }
         //--- check the data
         if(buffer[0]!=i || buffer[1024*1024-1]!=i+1024*1024-1)
           {
            Print("Client: benchmark invalid content");
            break;
           }
         //---
         volume+=sizeof(double)*1024*1024;
        }
      //--- send confirmation
      value=12345;
      if(!ExtPipe.WriteInteger(value))
         Print("Client: benchmark confirmation failed ");
      //--- show statistics
      ticks=GetTickCount()-ticks;
      if(ticks>0)
         printf("Client: %.0lf Mb received at %.0lf Mb per second\n",volume/1024/1024,volume/1024/ticks);
      //---
      ArrayFree(buffer);
     }
```

Note, that the first and the last elements of transfered blocks are checked in order to make sure that there were no errors during the transfer. Also, when transfer is complete client sends a confirming signal to the server about successful data receipt. If you won't use final confirmations, you will easily encounter a data loss if one of the parties closes connection too early.

Run the PipeServer.exe server locally and attach the PipeClient.mq5 script to any chart:

| PipeServer.exe | PipeClient.mq5 |
| --- | --- |
| ```<br>MQL5 Pipe Server<br>Copyright 2012, MetaQuotes Software Corp.<br>Pipe '\\.\pipe\MQL5.Pipe.Server' created<br>Client: waiting for connection...<br>Client: connected as 'PipeClient.mq5 on MQL5 build 705'<br>Server: send string<br>Server: send integer<br>Server: read string<br>Server: 'Test string' received<br>Server: read integer<br>Server: 1234567890 received<br>Server: start benchmark<br>......................................................<br>........<br>Server: 1024 Mb sent at 2921 Mb per second<br>``` | ```<br>PipeClient (EURUSD,H1)  Client: pipe opened<br>PipeClient (EURUSD,H1)  Server: Hello from pipe server received<br>PipeClient (EURUSD,H1)  Server: 1234567890 received<br>PipeClient (EURUSD,H1)  Client: 1024 Mb received at 2921 Mb per second<br>``` |

For local exchange, transfer rate is truly amazing - almost 3 gigabytes per second. This means that named pipes can be used to transfer almost any amount of data into MQL5 programs.

Now let's benchmark data transfer performance in an ordinary 1 gigabit LAN:

| PipeServer.exe | PipeClient.mq5 |
| --- | --- |
| ```<br>MQL5 Pipe Server<br>Copyright 2012, MetaQuotes Software Corp.<br>Pipe '\\.\pipe\MQL5.Pipe.Server' created<br>Client: waiting for connection...<br>Client: connected as 'PipeClient.mq5 on MQL5 build 705'<br>Server: send string<br>Server: send integer<br>Server: read string<br>Server: 'Test string' received<br>Server: read integer<br>Server: 1234567890 received<br>Server: start benchmark<br>......................................................<br>........<br>Server: 1024 Mb sent at 63 Mb per second<br>``` | ```<br>PipeClient (EURUSD,H1)  Client: pipe opened<br>PipeClient (EURUSD,H1)  Server: Hello from pipe server received<br>PipeClient (EURUSD,H1)  Server: 1234567890 received<br>PipeClient (EURUSD,H1)  Client: 1024 Mb received at 63 Mb per second<br>``` |

In local network, 1 gigabyte of data has been transfered at rate of 63 megabytes per second, which is very good. In fact it is 63% of the gigabit network maximum bandwidth.

### Conclusion

Protection system of the MetaTrader 5 trading platform does not allow MQL5 programs run outside their sandbox, guarding traders against threats when using untrusted Expert Advisors. Using named pipes you can easy create integrations with third-party software and manage EAs from outside. Safely.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/503](https://www.mql5.com/ru/articles/503)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/503.zip "Download all attachments in the single ZIP archive")

[pipeclient.mq5](https://www.mql5.com/en/articles/download/503/pipeclient.mq5 "Download pipeclient.mq5")(3.15 KB)

[pipeserver.zip](https://www.mql5.com/en/articles/download/503/pipeserver.zip "Download pipeserver.zip")(43.59 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

#### Other articles by this author

- [Getting Started with MQL5 Algo Forge](https://www.mql5.com/en/articles/18518)
- [Installing MetaTrader 5 and Other MetaQuotes Apps on HarmonyOS NEXT](https://www.mql5.com/en/articles/18612)
- [MetaTrader 5 on macOS](https://www.mql5.com/en/articles/619)
- [How to earn money by fulfilling traders' orders in the Freelance service](https://www.mql5.com/en/articles/1019)
- [MetaTrader 4 on macOS](https://www.mql5.com/en/articles/1356)
- [Working with ONNX models in float16 and float8 formats](https://www.mql5.com/en/articles/14330)
- [Regression models of the Scikit-learn Library and their export to ONNX](https://www.mql5.com/en/articles/13538)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/8304)**
(66)


![](https://c.mql5.com/avatar/avatar_na2.png)

**\[Deleted\]**
\|
14 Feb 2018 at 08:30

Or are the pips no longer relevant?


![leonerd](https://c.mql5.com/avatar/2017/5/5919A02E-9AEB.jpg)

**[leonerd](https://www.mql5.com/en/users/leonerd)**
\|
11 Sep 2020 at 19:03

Is the server for one client? I'm trying to connect a second client, the connection doesn't open. 5004 error. The file name is the same as in the other client connected.

If I disconnect the first client, the second one connects. So one [named channel](https://www.mql5.com/en/articles/503 "Article: Communication with MetaTrader 5 via named channels without using DLLs") is only one connection?

![leonerd](https://c.mql5.com/avatar/2017/5/5919A02E-9AEB.jpg)

**[leonerd](https://www.mql5.com/en/users/leonerd)**
\|
11 Sep 2020 at 21:05

How to connect two MT clients via named channels?

Tried the code from here [https://www.mql5.com/en/articles/115.](https://www.mql5.com/en/articles/115) It doesn't work. Connect method hangs.

![Pahlavon Tursunaliyev](https://c.mql5.com/avatar/avatar_na2.png)

**[Pahlavon Tursunaliyev](https://www.mql5.com/en/users/pahlavontursunaliyev5)**
\|
11 Jul 2022 at 00:59

Salom


![Quantum Capital International Group Ltd](https://c.mql5.com/avatar/2020/4/5E96871E-4BA3.png)

**[Yang Chih Chou](https://www.mql5.com/en/users/fxchess)**
\|
30 Aug 2023 at 09:00

Can it possible to use in C#?


![Interview with Francisco García García (ATC 2012)](https://c.mql5.com/2/0/avatar__15.png)[Interview with Francisco García García (ATC 2012)](https://www.mql5.com/en/articles/563)

Today we interview Francisco García García (chuliweb) from Spain. A week ago his Expert Advisor reached the 8th place, but the unfortunate logic error in programming threw it from the first page of the Championship leaders. As confirmed by statistics, such an error is not uncommon for many participants.

![Statistical Carry Trade Strategy](https://c.mql5.com/2/0/ava_Carry_trade.png)[Statistical Carry Trade Strategy](https://www.mql5.com/en/articles/491)

An algorithm of statistical protection of open positive swap positions from unwanted price movements. This article features a variant of the carry trade protection strategy that allows to compensate for potential risk of the price movement in the direction opposite to that of the open position.

![How to Write a Good Description for a Market Product](https://c.mql5.com/2/0/avatar__27.png)[How to Write a Good Description for a Market Product](https://www.mql5.com/en/articles/557)

MQL5 Market has many products for sale but some of their descriptions leave much to be desired. Many texts are obviously in need of improvement, as common traders are not able to comprehend them. This article will help you to put your product in a favorable light. Use our recommendations to write an eye-catching description that will easily show your customers what exactly you are selling.

![How to Subscribe to Trading Signals](https://c.mql5.com/2/0/signals_avatar.png)[How to Subscribe to Trading Signals](https://www.mql5.com/en/articles/523)

The Signals service introduces social trading with MetaTrader 4 and MetaTrader 5. The Service is integrated into the trading platform, and allows anyone to easily copy trades of professional traders. Select any of the thousands of signal providers, subscribe in a few clicks and the provider's trades will be copied on your account.

[![](https://www.mql5.com/ff/sh/592yc11u3j4rs5z9z2/01.png)How AI helps create robots for MetaTrader 5Learn from our book "Neural Networks in Algo Trading with MQL5"Read](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/neurobook%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.neurobook%26utm_content=visit.page%26utm_campaign=neurobook.promo.04.2024&a=ghrobswocqgvhztzjldphupateyllpro&s=9929cb0b8629585b5a42fabc06c525e41f6c0ebdf3045d044a5413b93ea88b47&uid=&ref=https://www.mql5.com/en/articles/503&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=6457849819444489697)

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