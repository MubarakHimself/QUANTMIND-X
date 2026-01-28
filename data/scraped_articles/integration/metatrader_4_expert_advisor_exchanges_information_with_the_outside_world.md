---
title: MetaTrader 4 Expert Advisor exchanges information with the outside world
url: https://www.mql5.com/en/articles/1361
categories: Integration, Expert Advisors
relevance_score: 6
scraped_at: 2026-01-23T17:27:00.176926
---

[![](https://www.mql5.com/ff/sh/wm94j0jmkwd29943z2/ddfa713cb3cdd580c3e81e0e13b5b1b8.jpg)\\
Revised MetaTrader 5 Web Terminal\\
\\
Trade with no restrictions from any mobile device, OS and web browser\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=fkjlpstbxdmrrwpblfatcsdjyrxbizyj&s=f462f051eb7aaec36d6b31792d312d60d3f5a50c83b12d0d66e85d5d61bd941b&uid=&ref=https://www.mql5.com/en/articles/1361&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5068245951488915250)

MetaTrader 4 / Examples


### Introduction

Here is a software tool which provides MetaTrader 4 Expert Advisors with an ability of creating both server and clients. Clients can establish connections both with their own servers and with any other types of servers which provide peer-to-peer protocol connections. The offered software tool consists of two components:

- **NetEventsProc.exe** \- is a Windows process that operates in background mode (without console) and runs (stops) when needed by the second component NetEventsProcDLL.dll. By request from applications, it can create for them both server and clients: both for their own servers and for any other types of servers which provide peer-to-peer protocol connections. For example, you can create clients that exchange information with websites. Of course, if you can support HTTP-protocol.

- **NetEventsProcDLL.dll** \- is the interface between application that require the NetEventsProc.exe process services and the NetEventsProc.exe process itself. Thanks to DLL interface, any programs written in any programming language can use this software tool for the two-way information exchange. MetaTrader 4 Expert Advisor is just an example of possible usage of this software tool.


**This article is structured in the following way:**

- **Quick Start**: the practical usage of the offered software tool is demonstrated at first on simple, then on more complex examples.

- **DLL Interface Specification**: detailed description of DLL functions and services provided by NetEventsProc.exe process when addressing to these functions from applications.

- **Project Implementation**: the details of this project implementation are clarified when possible.


The attached NetServerClient.zip archive contains two Microsoft Visual Studio 2010 Ultimate projects: **NetEventsProc** \- to build NetEventsProc.exe, and **NetEventsProcDLL** \- to build NetEventsProcDLL.dll. The source codes are commented in details. You can look into the details of implementation and customize the projects to your specific needs if you like.

NetEventsProc.exe implements server and clients using asynchronous sockets. To switch sockets to the asynchronous mode, one of the possible methods of operating in asynchronous mode is used: binding sockets to the WSAEventSelect(h\_Socket, h\_Event, FD\_ALL\_EVENTS) network events.

This project is based on the fundamental work of a great Master [Elmue](https://www.mql5.com/go?link=https://www.codeproject.com/Articles/34163/A-Universal-TCP-Socket-Class-for-Non-blocking-Serv "/go?link=https://www.codeproject.com/Articles/34163/A-Universal-TCP-Socket-Class-for-Non-blocking-Serv").

### 1\. Quick Start

**1.1. The exe.zip archive**

Extract this archive. You will find the following components:

- **NetEventsProcDLL.dll** \- place it into the "C:\\Windows\\System32\\" folder.

- **NetEventsProc.exe** \- create "C:\\NetEventsProc\\" folder and place this component into this folder. NetEventsProcDLL.dll will search the NetEventsProc.exe module exactly in this folder!


The following components of this archive provide mechanism of importing DLL functions from NetEventsProcDLL.dll to applications:

- **ImportNetEventsProcDLL.mqh** \- prototypes of functions imported to MetaTrader 4 Expert Advisor from NetEventsProcDLL.dll. Place this file into the terminal data folder "MetaTrader 4\\experts\\include\\".

- **cNetEventsProcDLL.h** \- C++ class definition containing all prototypes of DLL functions from NetEventsProcDLL.dll. Including this class in C++ program allows importing all DLL functions from NetEventsProcDLL.dll. For programs written in other programming languages you should either import each function separately or rewrite definition of this class respectively.

- **NetEventsProcDLL.lib** \- module included in programs that import DLL functions from NetEventsProcDLL.dll in the load-time dynamic linking mode (compile with: /EHsc/link NetEventsProcDLL.lib).


This actually completes the configuration process. Now you can write MetaTrader 4 Expert Advisors and applications in any programming languages using DLL functions for creating server and clients.

In order not to digress, we give the source codes of ImportNetEventsProcDLL.mqh and cNetEventsProcDLL.h right here. The ImportNetEventsProcDLL.mqh header file contains prototypes of imported DLL functions of the NetEventsProcDLL.dll program and two additional service functions:

```
string GetErrMsg(int s32_Error);
string FormatIP(int IP);
```

The **GetErrMsg** function converts the return codes of DLL functions to text. The **FormatIP** function converts the binary representation of IP address to the standard text format like "93.127.110.161". You should place the ImportNetEventsProcDLL.mqh file into the terminal data folder "MetaTrader 4\\experts\\include\\".

Here is the source code of **ImportNetEventsProcDLL.mqh** (only a part of the file is given that corresponds directly to the definition of prototypes of imported DLL functions):

```
// ImportNetEventsProcDLL.mqh

#import "NetEventsProcDLL.dll"
// Only for Clients:
int ConnectTo(string  ps8_ServerIP,             // in - string ps8_ServerIP = "0123456789123456"
              int     s32_Port,                 // in
              int&    h_Client[]);              // out - int h_Client[1]

int ConnectClose(int h_Client);                 // in
//
// Only for Server:
int ServerOpen(int  s32_Port);                  // in

int GetAllConnections(int& ph_Client[],         // out - int ph_Client[62]
                      int& ps32_ClientIP[],     // out - int ps32_ClientIP[62]
                      int& ps32_ClientCount[]); // out - int ps32_ClientCount[1]

int DisconnectClient(int h_Client);             // in

int ServerClose();
//
// For both: Clients and Server
int SendToInt   (int  h_Client,             // in
                 int& ps32_SendBuf[],       // in
                 int  s32_SendBufLen);      // in - SendBuf[] array size in int element

int SendToDouble(int     h_Client,          // in
                 double& pd_SendBuf[],      // in
                 int     s32_SendBufLen);   // in - SendBuf[] array size in double element

int SendToString(int    h_Client,           // in
                 string ps8_SendBuf,        // in
                 int    s32_SendBufLen);    // in - SendBuf string size in char element


int ReadFromInt   (int h_Client,            // in
                   int& ps32_ReadBuf[],     // in
                   int  s32_ReadBufLen,     // in  - ReadBuf[] array size in int element
                   int& ps32_ReadLen[]);    // out - int ps32_ReadLen[1] - count of actually read data in int element

int ReadFromDouble(int     h_Client,        // in
                   double& pd_ReadBuf[],    // in
                   int     s32_ReadBufLen,  // in  - ReadBuf[] array size in double element
                   int&    ps32_ReadLen[]); // out - int ps32_ReadLen[1] - count of actually read data in double element

int ReadFromString(int     h_Client,        // in
                   string  ps8_ReadBuf,     // in
                   int     s32_ReadBufLen,  // in  - ReadBuf   string size in char element
                   int&    ps32_ReadLen[]); // out - int ps32_ReadLen[1] - count of actually read data in char element
//
#import
//***************************************************************************************
...
...
...
// Get a human readable error message for an API error code
string GetErrMsg(int s32_Error)
{
   ...
   ..
}

// Convert DWORD IP to string IP
string FormatIP(int IP)
{
   ...
   ...
   ...
}
```

The **cNetEventsProcDLL.h** file has C++ class definition with all DLL functions imported from NetEventsProcDLL.dll. Here is the source code of this file:

```
//+---------------------------------------------------------------------------+
//|                                            cNetEventsProcDLL.h            |
//|                      Copyright © 2012, https://www.mql4.com/ en/users/more  |

//|                                       tradertobe@gmail.com                |
//+---------------------------------------------------------------------------+

// cNetEventsProcDLL.h

#pragma once

#define EXPFUNC __declspec(dllexport)

class cNetEventsProcDLL
{
public:
    static BOOL MessageDLL_PROCESS_ATTACH(void);
    static BOOL MessageDLL_PROCESS_DETACH(void);
//*******************************************************************************************************************
    static EXPFUNC int __stdcall ConnectTo(char* ps8_ServerIP, //in - ps8_ServerIP = "0123456789123456"
                                           int   s32_Port,     //in
                                           int*  ph_Client);   //out - int ph_Client[1]

    static EXPFUNC int __stdcall ConnectClose(int h_Client);   //in

    static EXPFUNC int __stdcall ServerOpen(int s32_Port);     //in

    static EXPFUNC int __stdcall GetAllConnections(int* ph_Client,         // out - int ph_Client[62]
                                                   int* ps32_ClientIP,     // out - int ps32_ClientIP[62]
                                                   int* ps32_ClientCount); // out - int ps32_ClientCount[1]

    static EXPFUNC int __stdcall DisconnectClient(SOCKET h_Client);        // in

    static EXPFUNC int __stdcall ServerClose();

    static EXPFUNC int __stdcall SendToInt(SOCKET h_Client,             // in
                                           int*   ps32_SendBuf,         // in
                                           int    s32_SendBufLen);      // in - SendBuf[] array size in int element

    static EXPFUNC int __stdcall SendToDouble(SOCKET  h_Client,         // in
                                              double* pd_SendBuf,       // in
                                              int     s32_SendBufLen);  // in - SendBuf[] array size in double element

    static EXPFUNC int __stdcall SendToString(SOCKET h_Client,          // in
                                              char*  ps8_SendBuf,       // in
                                              int    s32_SendBufLen);   // SendBuf string size in char element

    static EXPFUNC int __stdcall ReadFromInt(SOCKET h_Client,           // in
                                             int*   ps32_ReadBuf,       // in
                                             int    s32_ReadBufLen,     // ReadBuf[] array size in int element
                                             int*   ps32_ReadLen);      // out - int ps32_ReadLen[1] - actual count of read data in int element

    static EXPFUNC int __stdcall ReadFromDouble(SOCKET  h_Client,       // in
                                                double* pd_ReadBuf,     // in
                                                int     s32_ReadBufLen, // ReadBuf[] array size in double element
                                                int*    ps32_ReadLen);  // out - int ps32_ReadLen[1] - actual count of read data in double element

    static EXPFUNC int __stdcall ReadFromString(SOCKET h_Client,        // in
                                                char*  ps8_ReadBuf,     // in
                                                int    s32_ReadBufLen,  // ReadBuf[] array size in char element
                                                int*   ps32_ReadLen);   // out - int ps32_ReadLen[1] - actual count of read data in char element
//*******************************************************************************************************************
protected:
    static DWORD SendTo(SOCKET h_Client,   char* ps8_SendBuf, INT s32_SendBufLen);
    static DWORD ReadFrom(SOCKET h_Client, char* ps8_ReadBuf, INT s32_ReadBufLen, INT& s32_ReadLen);
};
```

**1.2. The FastStart.zip archive**

This archive contains source codes of all programs used in demo examples. C++ programs are represented as the Microsoft Visual Studio 2010 Ultimate projects **Client** and **EchoServer**. The source codes of MQL4 programs are also present in this archive along with the ImportNetEventsProcDLL.mqh file used to import DLL functions to MQL4 programs. Place this file in the folder "MetaTrader 4\\experts\\include\\".

On further discussion the source codes of all these programs are listed in the text. We will consider 3 examples that demonstrate the work with all DLL functions in MQL4 and C++ programming languages:

- **Section 1.2.1.** demonstrates information exchange between МetaТrader 4 Expert Advisor-server and C++ Client.

- **Section 1.2.2.** demonstrates information exchange between C++ Server and МetaТrader 4 Expert Advisor-client.

- **Section 1.2.3.** demonstrates information exchange between MetaTrader 4 Expert Advisors. One of these EAs acts as a server providing other МetaТrader 4 Expert Advisors-indicators (the Expert Advisors-clients) with the indicator values. In other words, we have implemented the distribution of values of a "secure" indicator to the clients.


**1.2.1. МetaТrader 4 Expert Advisor-server & C++ Program-client**

Consider this traditional task of information exchange between МetaТrader 4 Expert Advisor and C++ program:

- **EchoServer.mq4** \- Expert Advisor acts as an echo server.

- **Client.cpp** \- C++ program acts as a client for this Expert Advisor-server.


C++ client reads messages entered by user in the console and sends them to the Expert Advisor. Expert Advisor receives these messages, displays them in the terminal window and sends them back to the recipient. The images below illustrates this idea:

![Figure 1. МetaТrader 4 Expert Advisor-server & C++ Program-client](https://c.mql5.com/2/13/en_1_1.png)

Here is the source code of the MetaTrader 4 Expert Advisor **EchoServer.mq4** which acts as an echo server:

```
//+---------------------------------------------------------------------------+
//|                                            EchoServer.mq4                 |
//|                      Copyright © 2012, https://www.mql4.com/ en/users/more  |
//|                                       tradertobe@gmail.com                |
//+---------------------------------------------------------------------------+
#property copyright "Copyright © 2012, https://www.mql4.com/ ru/users/more"
#property link      "https://www.mql4.com/ ru/users/more"
#include <ImportNetEventsProcDLL.mqh>
/*int ServerOpen(int  s32_Port); // in
*/
/*int ServerClose();
*/
/*int GetAllConnections(int& ph_Client[],         // out - int ph_Client[62]
                        int& ps32_ClientIP[],     // out - int ps32_ClientIP[62]
                        int& ps32_ClientCount[]); // out - int ps32_ClientCount[1]
*/
/*int SendToString(int    h_Client,               // in
                   string ps8_SendBuf,            // in
                   int    s32_SendBufLen);        // in - SendBuf string size in char element
*/
/*int ReadFromString(int     h_Client,            // in
                   string    ps8_ReadBuf,         // in
                   int       s32_ReadBufLen,      // in  - ReadBuf   string size in char element
                   int&      ps32_ReadLen[]);     // out - int ps32_ReadLen[1] - count of actually read data in char element
*/
// Globals variables
int s32_Error;
int i;

int s32_Port = 2000;
bool b_ServerOpened = false;

// for GetAllConnections(ph_Client, ps32_ClientIP, ps32_ClientCount)
int ph_Client       [62];
int ps32_ClientIP   [62];
int ps32_ClientCount[1 ];

// for int ReadFromString(h_Client, ps8_Buf, s32_BufLen, ps32_ReadLen)
// for int SendToString  (h_Client, ps8_Buf, s32_BufLen)
string ps8_Buf = "0123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789";
int    s32_BufLen;
int    ps32_ReadLen[1];
//+------------------------------------------------------------------+
//| expert initialization function                                   |
//+------------------------------------------------------------------+
int init()
{
   //----
   s32_BufLen = StringLen(ps8_Buf);

   if (!b_ServerOpened)
   {
      s32_Error = ServerOpen(s32_Port);
      Print("ServerOpen() return is: ",GetErrMsg(s32_Error));

      if (s32_Error == OK)
      {
         b_ServerOpened = true;
         Print("Server is Opened and Waiting fo Client connection requests...");
      }
   }
//----
   return(0);
  }
//+------------------------------------------------------------------+
//| expert deinitialization function                                 |
//+------------------------------------------------------------------+
int deinit()
  {
//----
   if (b_ServerOpened)
   {
      s32_Error = ServerClose();
      Print("ServerClose() return is: ",GetErrMsg(s32_Error));

      if (s32_Error == OK)
         b_ServerOpened = false;
   }
//----
   return(0);
  }

int start()
{
//----
   if (!b_ServerOpened)
      return(0);

   s32_Error = GetAllConnections(ph_Client, ps32_ClientIP, ps32_ClientCount);

   if (s32_Error != 0)
   {
      Print("GetAllConnections(...) return is: ",GetErrMsg(s32_Error));
      return(1);
   }


   Print("ClientCount = ", ps32_ClientCount[0]);


   for (i = 0; i<ps32_ClientCount[0]; i++)
   {
      Print("h_Client = ", ph_Client[i], "      Client IP =  ", FormatIP(ps32_ClientIP[i]));

      s32_Error = ReadFromString(ph_Client[i], ps8_Buf, s32_BufLen, ps32_ReadLen);

      Print("ReadFromString(",ph_Client[i],",...) return is: ", GetErrMsg(s32_Error));

      if (s32_Error == 0)
      {
         Print("ReadFromString(",ph_Client[i],",...) ps32_ReadLen = ",ps32_ReadLen[0]);

         if (ps32_ReadLen[0] > 0)
         {
            Print("ReadFromString(",ph_Client[i],",...) Read data is: ", StringSubstr(ps8_Buf,0,ps32_ReadLen[0]));

            s32_Error = SendToString(ph_Client[i], ps8_Buf, StringLen(StringSubstr(ps8_Buf,0,ps32_ReadLen[0])));

            Print("SendToString(", ph_Client[i],",...) return is: ",GetErrMsg(s32_Error));
         }
      }
   }
//----
   return(0);
  }
//+------------------------------------------------------------------+
```

And here is the source code of **Client.cpp** \- C++ program which acts as a client for the Expert Advisor-server:

```
//+---------------------------------------------------------------------------+
//|                                            Client.cpp                     |
//|                      Copyright © 2012, https://www.mql4.com/ en/users/more  |
//|                                       tradertobe@gmail.com                |
//+---------------------------------------------------------------------------+

// Client.cpp

#include <winsock2.h>
#pragma comment(lib, "ws2_32")
#include <iostream>
#pragma comment(lib, "NetEventsProcDLL") // NetEventsProcDLL.lib placed in ...\FastStart\Client\
#include  "cNetEventsProcDLL.h"

// LocalIp = 0x6401a8c0 -> 192.168.1.100
// Returns a list of all local IP's on this computer (multiple IP's if multiple network adapters)
DWORD GetLocalIPs(char s8_IpList[][20], int &s32_IpCount);

/* This is the simple Client.
   It connects to Echo Server, send your input to Echo Server and read Echo from Server.

   ATTENTION !!! If your want to connect to local Server do not use  "127.0.0.1" !!!ATTENTION !!!
   You may get all your local IP's by means of GetLocalIPs(...) function, here is call example:
   // Local IP's list
        char s8_IpList[10][20];

        int s32_IpCount;
        DWORD u32_Err = GetLocalIPs(s8_IpList, s32_IpCount);

        if (u32_Err)
        {
                printf("Get local IP's error !, no local IP means no network available...");
                return 1;
        }

        printf("\nLocal IP's list:\n");

        for (int i = 0; i<s32_IpCount; i++)
                printf("\n%s\n",s8_IpList[i]);
        //
*/

// This is the Server we want to connect to...
#define SERVER_IP   "192.168.1.5"  //this is mine local IP's get by means of GetLocalIPs(...) function,
                                   // do not use so called loopback  "127.0.0.1" address !!!
#define PORT         2000

int main()
{
        // Local IP's list
        char s8_IpList[10][20];

        int s32_IpCount;
        DWORD u32_Err = GetLocalIPs(s8_IpList, s32_IpCount);

        if (u32_Err)
        {
                printf("Get local IP's error !, no local IP means no network available...");
                return 1;
        }

        printf("\nLocal IP's list:\n");

        for (int i = 0; i<s32_IpCount; i++)
                printf("\n%s\n",s8_IpList[i]);
        //

        char s8_ServerIP[] = SERVER_IP;
        int  s32_Port      = PORT;
        int  ph_Client[1];

        int  h_Client;

        DWORD u32_Error = cNetEventsProcDLL::ConnectTo(SERVER_IP, PORT, ph_Client);

        if (u32_Error)
        {
                printf("\nConnectTo(...) failed with error: %d\n", u32_Error);
                return 1;
        }
        else
                printf("\nConnectTo(...) OK, ph_Client[0] = : %d\n", ph_Client[0]);

        h_Client = ph_Client[0];

        // Connection is established successfully ! Let's  have some SendTo(...) data on this connection...
        char  ps8_SendData[200];
        int   s32_SendDataLen;;

        char   ps8_ReadBuf[1025];
        DWORD  s32_ReadBufLen = 1025;
        int    ps32_ReadLen[1];

        while(true)
        {
                std::cout << "\nEnter something to send to Server or Exit 'q':\n" << std::endl;

                std::cin.getline(ps8_SendData, 200);

                s32_SendDataLen = strlen(ps8_SendData);

                OemToCharBuff(ps8_SendData, ps8_SendData, s32_SendDataLen);

                if (ps8_SendData[0] == 'q')
                {
                        u32_Error = cNetEventsProcDLL::ConnectClose(h_Client);
                        if (u32_Error)
                        {
                                printf("\nConnectClose(...) failed with error: %d\n", u32_Error);
                                break;
                        }
                        else
                        {
                                printf("\nConnectClose(...) OK.\n");
                                break;
                        }
                }

                u32_Error = cNetEventsProcDLL::SendToString(h_Client, ps8_SendData, s32_SendDataLen);

                switch (u32_Error)
                {
                case 0:
                        printf("\nSendTo(...) OK");
                        printf("\nSendTo(%d...) sent %d bytes\n", h_Client, s32_SendDataLen);
                        CharToOemBuff(ps8_SendData, ps8_SendData, s32_SendDataLen);
                        printf("\nSendTo(%d...) Sent Data: %s\n",h_Client, ps8_SendData);
                        printf("Waiting now for Echo....");
                        break;
                case ERROR_INVALID_PARAMETER:
                        printf("\nSendTo(%d...) return is: ERROR_INVALID_PARAMETER(%d)\n",h_Client, u32_Error);
                        printf("\nERROR_INVALID_PARAMETER -> One of this parms or more: h_Client, ps8_SendData, u32_SendDataLen is invalid...\n");
                        break;
                case WSAEWOULDBLOCK:
                        printf("\nSendTo(%d...) return is: WSAEWOULDBLOCK(%d)\n",h_Client, u32_Error);
                        printf("\nWSAEWOULDBLOCK -> The data will be send after the next FD_WRITE event, do nouthing\n");
                        break;

                case WSA_IO_PENDING:
                        printf("\nSendTo(%d...) return is: WSA_IO_PENDING(%d)\n",h_Client, u32_Error);
                        printf("\nWSA_IO_PENDING -> Error: A previous Send operation is still pending. This data will not be sent, try latter\n");
                        break;

                default:
                        printf("\nSendTo(%d...)failed with severe error: %d\n",h_Client, u32_Error);
                        // Severe error -> abort event loop
                        printf("\nConnection was closed !\n");
                        break;
                };

                if (u32_Error == 0 || u32_Error == WSAEWOULDBLOCK)
                {
                        int ReadLen = 0;

                        while(!ReadLen)
                        {
                                u32_Error = cNetEventsProcDLL::ReadFromString(h_Client, ps8_ReadBuf, s32_ReadBufLen, ps32_ReadLen);

                                if (u32_Error)
                                {
                                        printf("\nReadFromString(%d...) failed with error: %d\n", h_Client, u32_Error);
                                        break;
                                }

                                ReadLen = ps32_ReadLen[0];
                        }
                        if (u32_Error)
                        {
                                printf("\nReadFromString(%d...) failed with error: %d\n", h_Client, u32_Error);
                        }
                        else
                        {
                                printf("\nReadFromString(%d...) OK, read %d  bytes\n", h_Client, ReadLen);
                        }

                        if (ReadLen > 0)
                        {
                                CharToOemBuff(ps8_ReadBuf, ps8_ReadBuf, s32_SendDataLen);
                                ps8_ReadBuf[ReadLen] = 0;
                                printf("\nReadFromString(%d...) Read Data: %s\n", h_Client, ps8_ReadBuf);
                        }

                }

        }

        return 0;
}

// LocalIp = 0x6401a8c0 -> 192.168.1.100
// Returns a list of all local IP's on this computer (multiple IP's if multiple network adapters)
DWORD GetLocalIPs(char s8_IpList[][20], int &s32_IpCount)
{
        // Winsock version 2.0 is available on ALL Windows operating systems
        // except Windows 95 which comes with Winsock 1.1
        WSADATA k_Data;
        DWORD u32_Error = WSAStartup(MAKEWORD(2,0), &k_Data);
        if (u32_Error)
                return u32_Error;

        int ps32_IpList[20];

        char s8_Host[500];
        if (gethostname(s8_Host, sizeof(s8_Host)) == SOCKET_ERROR)
                return WSAGetLastError();

        struct hostent* pk_Host = gethostbyname(s8_Host);
        if (!pk_Host)
                return WSAGetLastError();

        s32_IpCount = 0;

        for (DWORD i=0; TRUE; i++)
        {
                if (!pk_Host->h_addr_list[i])
                        break; // The IP list is zero terminated

                ps32_IpList[i] = *((DWORD*)pk_Host->h_addr_list[i]);

                s32_IpCount++;
        }

        if (!s32_IpCount)
                return WSAENETDOWN; // no local IP means no network available

        for (int i = 0; i<s32_IpCount; i++)
        {
                BYTE* pu8_Addr = (BYTE*)&ps32_IpList[i];

                sprintf(s8_IpList[i],"%d.%d.%d.%d",pu8_Addr[0], pu8_Addr[1], pu8_Addr[2], pu8_Addr[3]);
        }
        return 0;
}
```

To run this demo example you need:

1. Place the **EchoServer.mq4** file in the terminal data folder "МetaТrader 4\\experts\\" and compile it.

2. Open the **Client** project in Microsoft Visual Studio 2010 Ultimate and build it using the **Release** configuration. If you want to build the project in another IDE don't forget to specify the NetEventsProcDLL.lib module as an additional entry for the editor (compile with: /EHsc/link NetEventsProcDLL.lib).

3. Run the **Client.exe** executable module. You will get error with code 10057 and the list of local IPs of your computer. Correct the following string in the Client.cpp source code


```
#define SERVER_IP   "192.168.1.5"
```


by replacing "192.168.1.5" with the first (or the only) local IP and compile the project again.

4. Run the **EchoServer** Expert Advisor on any chart in the MetaTrader 4 terminal. If everything is done correctly, the terminal window will display the following messages: "ServerOpen() return is: OK", "Server is Opened and Waiting for Client connection requests...". Then on every tick the Expert Advisor checks connection requests, connects, reads incoming messages from every connection and sends a copy of the message back to the recipient. The Expert Advisor displays all current information in the terminal window.

5. Now you can run the **Client.exe** program built in step 3.

6. You can run several copies of this program and watch how the Expert Advisor exchanges information with all copies.

7. If you set global IP of computer where the Expert Advisor-server is running instead of its local IP in step 3, in this case you can run **Client.exe** and communicate with the Expert Advisor on any other computer that is connected to the Internet. Of course don't forget to disable all current firewall protections.


**1.2.2. C++ Program-server & MetaTrader 4 Expert Advisor-client**

Now let's consider the traditional task of information exchange between МetaТrader 4 Expert Advisor and C++ program:

- **EchoServer.cpp** \- C++ program acts as an echo server.

- **Client.mq4** \- Expert Advisor acts as a client for this C++ server.


The Expert Advisor-client reads quotes of the symbol on which it is running and sends these quotes to the C++ server. C++ server returns quotes back to the recipient which receives and displays them in the terminal window. The images below illustrates this idea:

![Figure 2. C++ Program-server & MetaTrader 4 Expert Advisor-client](https://c.mql5.com/2/13/en_2_1.png)

Here is the source code of the МetaТrader 4 Expert Advisor **Client.mq4** which acts as a client:

```
//+---------------------------------------------------------------------------+
//|                                            Client.mq4                     |
//|                      Copyright © 2012, https://www.mql4.com/ en/users/more  |
//|                                       tradertobe@gmail.com                |
//+---------------------------------------------------------------------------+
#property copyright "Copyright © 2012, https://www.mql4.com/ ru/users/more"
#property link      "https://www.mql4.com/ ru/users/more"
#include <ImportNetEventsProcDLL.mqh>
/*int ConnectTo(string  ps8_ServerIP,         // in - string ps8_ServerIP = "0123456789123456"
                int     s32_Port,             // in
                int&    ph_Client[]);         // out - int ph_Client[1]
*/
/*int SendToDouble(int     h_Client,          // in
                   double& pd_SendBuf[],      // in
                   int     s32_SendBufLen);   // in - SendBuf[] array size in double element
*/
/*int ReadFromDouble(int     h_Client,        // in
                     double& pd_ReadBuf[],    // in
                     int     s32_ReadBufLen,  // in  - ReadBuf[] array size in double element
                     int&    ps32_ReadLen[]); // out - int ps32_ReadLen[1] - count of actually read data in double element
*/
/*int ConnectClose(int h_Client);             // in
*/

// Globals variables
int s32_Error;
int i;
// for int ConnectTo(ps8_ServerIP, s32_Port, ph_Client); // out - int h_Client[1]
string ps8_ServerIP = "192.168.1.5";                     // mine local IP
int    s32_Port = 2000;
int    ph_Client[1];

bool b_ConnectTo = false;

// for int SendToDouble(ph_Client[0], pd_Buf, s32_BufLen);
// for int ReadFromDouble(ph_Client[0], pd_Buf, s32_BufLen, ps32_ReadLen);
double pd_Buf[1];
int    s32_BufLen = 1;
int    ps32_ReadLen[1];
//+------------------------------------------------------------------+
//| expert initialization function                                   |
//+------------------------------------------------------------------+
int init()
{
//----
   if (!b_ConnectTo)
   {
      s32_Error = ConnectTo(ps8_ServerIP, s32_Port, ph_Client);
      Print("ConnectTo(...) return is: ",GetErrMsg(s32_Error));
      Print("ConnectTo(...) handle is: ",ph_Client[0]);

      if (s32_Error == OK)
      {
         b_ConnectTo = true;
         Print("Client now is ConnectTo the Server: ",ps8_ServerIP);
      }
   }
//----
   return(0);
}
//+------------------------------------------------------------------+
//| expert deinitialization function                                 |
//+------------------------------------------------------------------+
int deinit()
{
//----
   if (b_ConnectTo)
   {
      s32_Error = ConnectClose(ph_Client[0]);
      Print("ConnectClose(...) return is: ",GetErrMsg(s32_Error));

      if (s32_Error == OK)
         b_ConnectTo = false;
   }
//----
   return(0);
}

int start()
{
//----
   if (!b_ConnectTo)
      return(0);

   RefreshRates();

   double pd_Value[1];

   pd_Value[0] = NormalizeDouble(Bid,Digits);

   s32_Error = SendToDouble(ph_Client[0], pd_Value, s32_BufLen);


   if (s32_Error != 0)
   {
      Print("SendToDouble(",ph_Client[0],"...) return is: ",GetErrMsg(s32_Error));
      return(1);
   }
   else
      Print("SendToDouble(",ph_Client[0],"...) return is: OK");

   s32_Error = ReadFromDouble(ph_Client[0], pd_Buf, s32_BufLen, ps32_ReadLen);

   if (s32_Error != 0)
   {
      Print("ReadFromDouble(",ph_Client[0],"...) return is: ", GetErrMsg(s32_Error));
      return(1);
   }
   else
      Print("ReadFromDouble(",ph_Client[0],"...) return is: OK");

   pd_Buf[0] = NormalizeDouble(pd_Buf[0],Digits);

   if (ps32_ReadLen[0] > 0)
      Print("Read doble value is: ", pd_Buf[0]);

//----
   return(0);
  }
//+------------------------------------------------------------------+
```

And here is the source code of the **EchoServer.cpp** C++ program which acts as an echo server for the Expert Advisor-client:

```
//+---------------------------------------------------------------------------+
//|                                            EchoServer.cpp                 |
//|                      Copyright © 2012, https://www.mql4.com/ en/users/more  |
//|                                       tradertobe@gmail.com                |
//+---------------------------------------------------------------------------+

// EchoServer.cpp

#include <winsock2.h>
#pragma comment(lib, "NetEventsProcDLL") // NetEventsProcDLL.lib placed in ...\FastStart\EchoServer\
#include <iostream>
#include <conio.h>

#include  "cNetEventsProcDLL.h"

BOOL FormatIP(DWORD u32_IP, char* s8_IP);

int main()
{
        int s32_Port = 2000;

        // Try to create server listening on port 2000
        // You may change port.
        DWORD u32_Error = cNetEventsProcDLL::ServerOpen(s32_Port);

        if (u32_Error)
        {
                printf("\nServerOpen() failed with error: %d\n", u32_Error);
                return 1;
        }
        else
                printf("\nServerOpen() fine, we now are waiting for connections...\n");

        DWORD u32_Count = 0;
        DWORD u32_CountOld = 0;

        double pd_Buf[1025];
        DWORD  u32_BufLen = 1025;
        int    ps32_ReadLen[1];

        pd_Buf[0] = 0;

        int ph_Client[62];
        int ps32_ClientIP[62];
        int ps32_ClientCount[1];

        while(!kbhit())
        {
                u32_Error = cNetEventsProcDLL::GetAllConnections(ph_Client, ps32_ClientIP, ps32_ClientCount);

                if (u32_Error)
                {
                        printf("\nGetAllConnections(...) failed with error: %d\n", u32_Error);
                        break;
                }
                else
                        u32_Count = ps32_ClientCount[0];

                if (u32_Count != u32_CountOld)
                {
                        u32_CountOld = u32_Count;

                        printf("\nNumber of connections now = %d\n", u32_Count);
                        printf("#     h_Connect   (peer IP)\n");

                        for (DWORD i = 0; i<u32_Count; i++)
                        {
                                char  s8_IP[20];
                                sprintf(s8_IP, "%s","123456789012345");

                                FormatIP(ps32_ClientIP[i], s8_IP);

                                printf("%d      %d       (%s)\n", i, ph_Client[i], s8_IP);
                        }
                }

                for (DWORD i = 0; i<u32_Count; i++)
                {
                        u32_Error = cNetEventsProcDLL::ReadFromDouble(ph_Client[i], pd_Buf, u32_BufLen, ps32_ReadLen);

                        if (u32_Error)
                        {
                                printf("ReadFromDouble(%d...) failed with error: %d\n", ph_Client[i], u32_Error);
                        }

                        if (ps32_ReadLen[0])
                        {
                                printf("ReadFromDouble(%d...) read %d double values\n", ph_Client[i], ps32_ReadLen[0]);
                                printf("\nReadFromDouble(%d...) Read Data: %9.5f\n", ph_Client[i], pd_Buf[0]);
                        }

                        if (ps32_ReadLen[0])
                        {
                                u32_Error = cNetEventsProcDLL::SendToDouble(ph_Client[i], pd_Buf, ps32_ReadLen[0]);

                                if (u32_Error)
                                {
                                        printf("SendToDouble(%d...) failed with error: %d\n", ph_Client[i], u32_Error);
                                }
                                else
                                {
                                        printf("SendToDouble(%d...) sent %d double values\n", ph_Client[i], ps32_ReadLen[0]);
                                        printf("SendToDouble(%d...) sent Data: %9.5f\n",ph_Client[i], pd_Buf[0]);
                                }
                        }

                }

        }

        u32_Error = cNetEventsProcDLL::ServerClose();

        if (u32_Error)
        {
                printf("\nServerClose() failed with error: %d\n", u32_Error);
                return 1;
        }
        else
                printf("\nServerClose() fine...\n");

        Sleep(10000);
        return 0;
}

BOOL FormatIP(DWORD u32_IP, char* s8_IP)
{
        DWORD u32_Len = strlen(s8_IP);

        if ( u32_Len < 15)
                return FALSE;

        BYTE* pu8_Addr = (BYTE*)&u32_IP;
        sprintf(s8_IP,"%d.%d.%d.%d",pu8_Addr[0], pu8_Addr[1], pu8_Addr[2], pu8_Addr[3]);

        return TRUE;
}
```

To run this demo example you need:

1. Place the **Client.mq4** file in the terminal data folder "MetaTrader 4\\experts\\", assign local IP (obtained in the previous example 1.2.1.) to the string


```
string ps8_ServerIP = "192.168.1.5";
```


and compile it. If C++ server will be run on another computer, then paste here the global IP of this computer. Don't forget to disable all existing protections in your firewall.

2. Open the **Client** project in Microsoft Visual Studio 2010 Ultimate and build it using the **Release** configuration. If you want to build the project in another IDE don't forget to specify the NetEventsProcDLL.lib module as an additional entry for the editor (compile with: /EHsc/link NetEventsProcDLL.lib).

3. Run the **EchoServer.exe** C++ server. If everything is done correctly, the console will display the following message: "ServerOpen() fine, we now are waiting for connections.....". Pressing any key will close the server and terminate the program.

4. Run the **Client.mq4** client on any chart of the МetaТrader 4 terminal.

5. You can run the client simultaneously on several charts, in one or different terminals, on one or different computers.

6. Watch how the C++ server and МetaТrader 4 clients work. Press any key in the C++ server console, and then C++ program will close the server and terminate.


**1.2.3. МetaТrader 4 Expert Advisor-indicator (the Expert Advisor-server) & МetaТrader 4 Client-indicator**

The Expert Advisor-indicator (the Expert Advisor-server) provides the Clients-indicators with the indicator values. In this case these are the values of the standard iEnvelops(...) indicator. This example can have a practical value for distribution of the values of the "secured" indicator to all subscribers-clients.

The images below illustrates this idea:

![Figure 3. МetaТrader 4 Expert Advisor-indicator (the Expert Advisor-server) & МetaТrader 4 Client-indicator](https://c.mql5.com/2/13/en_3_1.png)

Here is the source code of the МetaТrader 4 Expert Advisor **ServerSendInd.mq4** which acts as the provider of the iEnvelops(...) indicator values:

```
//+---------------------------------------------------------------------------+
//|                                            ServerSendInd.mq4              |
//|                      Copyright © 2012, https://www.mql4.com/ en/users/more  |
//|                                       tradertobe@gmail.com                |
//+---------------------------------------------------------------------------+
#property copyright "Copyright © 2012, https://www.mql4.com/ ru/users/more"
#property link      "https://www.mql4.com/ ru/users/more"
#include <ImportNetEventsProcDLL.mqh>
/*int ServerOpen(int  s32_Port);                  // in
*/
/*int ServerClose();
*/
/*int GetAllConnections(int& ph_Client[],         // out - int ph_Client[62]
                        int& ps32_ClientIP[],     // out - int ps32_ClientIP[62]
                        int& ps32_ClientCount[]); // out - int ps32_ClientCount[1]
*/
/*int ReadFromString(int     h_Client,            // in
                   string    ps8_ReadBuf,         // in
                   int       s32_ReadBufLen,      // in  - ReadBuf   string size in char element
                   int&      ps32_ReadLen[]);     // out - int ps32_ReadLen[1] - count of actually read data in char element
*/
/*int SendToDouble(int     h_Client,              // in
                   double& pd_SendBuf[],          // in
                   int     s32_SendBufLen);       // in - SendBuf[] array size in double element
*/
// Globals variables
int s32_Error;
int i;

int s32_Port = 2000;
bool b_ServerOpened = false;

// for GetAllConnections(ph_Client, ps32_ClientIP, ps32_ClientCount)
int ph_Client       [62];
int ps32_ClientIP   [62];
int ps32_ClientCount[1 ];

// for int ReadFromString(h_Client, ps8_ReadBuf, s32_ReadBufLen, ps32_ReadLen)
string ps8_ReadBuf = "0123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789";
int    s32_ReadBufLen;
int    ps32_ReadLen[1];

// for int SendToDouble(ph_Client[0], pd_SendBuf, s32_SendBufLen);
#define  BARS_COUNT  200
double pd_SendBuf      [BARS_COUNT];  //BARS_COUNT/2 Bars for each of 2 line
int    s32_SendBufLen = BARS_COUNT;

//+------------------------------------------------------------------+
//| expert initialization function                                   |
//+------------------------------------------------------------------+
int init()
{
   //----
   s32_ReadBufLen = StringLen(ps8_ReadBuf);

   if (!b_ServerOpened)
   {
      s32_Error = ServerOpen(s32_Port);
      Print("ServerOpen() return is: ",GetErrMsg(s32_Error));

      if (s32_Error == OK)
      {
         b_ServerOpened = true;
         Print("Server is Opened and Waiting for Clients connection requests...");
      }
   }
//----
   return(0);
}
//+------------------------------------------------------------------+
//| expert deinitialization function                                 |
//+------------------------------------------------------------------+
int deinit()
{
//----
   if (b_ServerOpened)
   {
      s32_Error = ServerClose();
      Print("ServerClose() return is: ",GetErrMsg(s32_Error));

      if (s32_Error == OK)
         b_ServerOpened = false;
   }
//----
   return(0);
}

int start()
{
//----
   if (!b_ServerOpened)
      return(0);

   s32_Error = GetAllConnections(ph_Client, ps32_ClientIP, ps32_ClientCount);

   if (s32_Error != 0)
   {
      Print("GetAllConnections(...) failed with error: ",GetErrMsg(s32_Error));
      return(1);
   }

   Print("ClientCount = ", ps32_ClientCount[0]);

   for (i = 0; i<ps32_ClientCount[0]; i++)
   {
      Print("h_Client = ", ph_Client[i], "      Client IP =  ", FormatIP(ps32_ClientIP[i]));

      s32_Error = ReadFromString(ph_Client[i], ps8_ReadBuf, s32_ReadBufLen, ps32_ReadLen);

      if (s32_Error != 0)
      {
         Print("ReadFromString(",ph_Client[i],") failed with error: ", GetErrMsg(s32_Error));
         continue;
      }

      if (ps32_ReadLen[0] > 0)
      {
         // ps8_ReadBuf = "EURUSDMinuts"   i.e. "Symbol+Timeframe"
         string Sym       = StringSubstr(ps8_ReadBuf,0,6);
         int    TimeFrame = StrToInteger(StringSubstr(ps8_ReadBuf,6,ps32_ReadLen[0]-6));


         int k;
         for (k = 0; k<BARS_COUNT/2; k++)
         {
            while(true)
            {
               double UpperLine_k = iEnvelopes(Sym, TimeFrame, 14, MODE_SMA, 0, PRICE_CLOSE, 0.1, MODE_UPPER, k);
               if (GetLastError() != 0)
                  continue;
               else
                  break;
            }
            while(true)
            {
               double LowerLine_k = iEnvelopes(Sym, TimeFrame, 14, MODE_SMA, 0, PRICE_CLOSE, 0.1, MODE_LOWER, k);
               if (GetLastError() != 0)
                  continue;
               else
                  break;
            }

            pd_SendBuf[k]              = UpperLine_k;
            pd_SendBuf[k+BARS_COUNT/2] = LowerLine_k;
         }

         s32_Error = SendToDouble(ph_Client[i], pd_SendBuf, s32_SendBufLen);
         if (s32_Error != 0)
         {
            Print("SendToDouble(",ph_Client[i],") failed with error: ", GetErrMsg(s32_Error));

            continue;
         }
      }
   }
//----
   return(0);
}
//+------------------------------------------------------------------+
```

And here is the source code of the Client-indicator **ClientIndicator.mq4** which gets the iEnvelops(...) indicator values from the **ServerSendInd.mq4** Expert Advisor:

```
//+---------------------------------------------------------------------------+
//|                                            ClientIndicator.mq4            |
//|                      Copyright © 2012, https://www.mql4.com/ en/users/more  |
//|                                       tradertobe@gmail.com                |
//+---------------------------------------------------------------------------+
#property copyright "Copyright © 2012, https://www.mql4.com/ en/users/more"
#property link      "https://www.mql4.com/ ru/users/more"
#include <ImportNetEventsProcDLL.mqh>

/*int ConnectTo(string  ps8_ServerIP,         // in - string ps8_ServerIP = "0123456789123456"
                int     s32_Port,             // in
                int&    ph_Client[]);         // out - int ph_Client[1]
*/
/*
/*int ConnectClose(int h_Client);             // in
*/
/*int SendToString(int    h_Client,           // in
                   string ps8_SendBuf,        // in
                   int    s32_SendBufLen);    // in - SendBuf string size in char element
*/
/*int ReadFromDouble(int     h_Client,        // in
                     double& pd_ReadBuf[],    // in
                     int     s32_ReadBufLen,  // in  - ReadBuf[] array size in double element
                     int&    ps32_ReadLen[]); // out - int ps32_ReadLen[1] - count of actually read data in double element
*/
// Globals variables
int s32_Error;
int i;
// for int ConnectTo(ps8_ServerIP, s32_Port, ph_Client);  // out - int h_Client[1]
string ps8_ServerIP = "192.168.1.5";                      // mine local IP
int    s32_Port = 2000;
int    ph_Client[1];

bool b_ConnectTo = false;

// for int SendToString  (h_Client, ps8_SendBuf, s32_SendBufLen)
string ps8_SendBuf = "0123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789";
int    s32_SendBufLen;

// for int ReadFromDouble(ph_Client[0], pd_ReadBuf, s32_ReadBufLen, ps32_ReadLen);
#define BARS_COUNT  200
double  pd_ReadBuf      [BARS_COUNT];
int     s32_ReadBufLen = BARS_COUNT;
int     ps32_ReadLen[1];

string Indicator_Name = "Envelopes: ";
//----
#property indicator_chart_window
#property indicator_buffers 2
#property indicator_color1 Blue
#property indicator_color2 Red

double UpperLine[];
double LowerLine[];

//+------------------------------------------------------------------+
//| Custom indicator initialization function                         |
//+------------------------------------------------------------------+
int init()
{
//---- indicators
   SetIndexStyle (0,DRAW_LINE);
   SetIndexBuffer(0,UpperLine);

   SetIndexStyle (1,DRAW_LINE);
   SetIndexBuffer(1,LowerLine);

   s32_SendBufLen = StringLen(ps8_SendBuf);

   if (!b_ConnectTo)
   {
      s32_Error = ConnectTo(ps8_ServerIP, s32_Port, ph_Client);
      Print("ConnectTo(...) return is: ",GetErrMsg(s32_Error));
      Print("ConnectTo(...) handle is: ",ph_Client[0]);

      if (s32_Error == OK)
      {
         b_ConnectTo = true;
         Print("Client now is ConnectTo the Server: ",ps8_ServerIP);
      }
   }
//----
   return(0);
}

//+------------------------------------------------------------------+
//| Custom indicator deinitialization function                       |
//+------------------------------------------------------------------+
int deinit()
{
    //----
   if (b_ConnectTo)
   {
      s32_Error = ConnectClose(ph_Client[0]);
      Print("ConnectClose(...) return is: ",GetErrMsg(s32_Error));

      if (s32_Error == OK)
         b_ConnectTo = false;
   }
//----
   return(0);
}
//+------------------------------------------------------------------+
//| Custom indicator iteration function                              |
//+------------------------------------------------------------------+
int start()
{
//----
   if (!b_ConnectTo)
      return(0);

    string Sym       = Symbol();
    int    TimeFrame = Period();
    ps8_SendBuf = Symbol() + DoubleToStr(Period(),0);

    s32_Error = SendToString(ph_Client[0], ps8_SendBuf, StringLen(ps8_SendBuf));

    if (s32_Error != 0)
    {
      Print("SendToString(", ph_Client[0],",...) failed with error: ",GetErrMsg(s32_Error));
      return (1);
    }

    s32_Error = ReadFromDouble(ph_Client[0], pd_ReadBuf, s32_ReadBufLen, ps32_ReadLen);

   if (s32_Error != 0)
   {
      Print("ReadFromDouble(",ph_Client[0],"...) return is: ", GetErrMsg(s32_Error));
      return(1);
   }

   if (ps32_ReadLen[0] == 0)
      return (0);

//--------------------------------------------------------------------
    int Counted_bars = IndicatorCounted();       // Number of calculated bars
    i = Bars - Counted_bars - 1;                 // Index of first not-calculated
    if (i > BARS_COUNT/2-1)  i = BARS_COUNT/2-1; // Calculate specified count if there are many bars
//-----------------------------------------------------------------------
    for (i = BARS_COUNT/2-1; i >= 0; i--)
    {
         UpperLine  [i] = pd_ReadBuf[i];
         LowerLine  [i] = pd_ReadBuf[i+BARS_COUNT/2];
    }

    return;
} // end of int start()
//--------------------------------------------------------------------
```

To run this demo example you need:

1. Place the **ServerSendInd.mq4** file in the terminal data folder "МetaТrader 4\\experts\\" and compile it.

2. Place the **ClientIndicator.mq4** file in the terminal data folder "МetaТrader 4\\experts\\indicators\\" and assign the local IP (obtained in the example 1.2.1.) to the string:


```
string ps8_ServerIP = "192.168.1.5";
```


If ServerSendInd will be run on another computer, then paste here the global IP of this computer. Don't forget to disable all existing protections in your firewall. Compile it.

3. Run **ServerSendInd**. If everything is done correctly, the console will display the following message: "ServerOpen() fine, we now are waiting for connections.....".

4. On any chart of МetaТrader 4 terminal run the **ClientIndicator.mq4** indicator. Two indicator lines will appear on the chart. If you run the standard Envelops indicator on the same chart, make sure that both lines of our indicator coincide with lines of standard indicator.

5. You can run the **ClientIndicator** indicator simultaneously on several charts, in one or different terminals, on one or different computers.

6. Watch how the **ServerSendInd** server and **ClientIndicator** work. Try to change the timeframe of chart with the ClientIndicator indicator. The ServerSendInd server will be set up immediately to send indicator values for this timeframe.

### 2\. DLL Interface Specification

In this section we will describe in details all the DLL functions and parameters that are necessary to call them. All functions return zero in case of successful execution. Otherwise functions return [winsock2 API error codes](https://www.mql5.com/go?link=https://msdn.microsoft.com/en-us/library/windows/desktop/ms681382(v=vs.85).aspx "/go?link=https://msdn.microsoft.com/en-us/library/windows/desktop/ms681382(v=vs.85).aspx"). All exported DLL functions are given in the C++ class **cNetEventsProcDLL.h** declaration, therefore we will provide the source code of this file:

```
//+---------------------------------------------------------------------------+
//|                                            cNetEventsProcDLL.h            |
//|                      Copyright © 2012, https://www.mql4.com/ en/users/more  |
//|                                       tradertobe@gmail.com                |
//+---------------------------------------------------------------------------+
//--- cNetEventsProcDLL.h
#pragma once
#define EXPFUNC __declspec(dllexport)
//---
class cNetEventsProcDLL
  {
public:
   static BOOL MessageDLL_PROCESS_ATTACH(void);
   static BOOL MessageDLL_PROCESS_DETACH(void);
//---
   static EXPFUNC int __stdcall ConnectTo(char *ps8_ServerIP,             // in - ps8_ServerIP = "0123456789123456"
                                          int   s32_Port,                 // in
                                          int*  ph_Client);               // out - int ph_Client[1]
//---
   static EXPFUNC int __stdcall ConnectClose(int h_Client);               // in
//---
   static EXPFUNC int __stdcall ServerOpen(int s32_Port);                 // in
//---
   static EXPFUNC int __stdcall GetAllConnections(int* ph_Client,         // out - int ph_Client[62]
                                                  int* ps32_ClientIP,     // out - int ps32_ClientIP[62]
                                                  int* ps32_ClientCount); // out - int ps32_ClientCount[1]
//---
   static EXPFUNC int __stdcall DisconnectClient(SOCKET h_Client);        // in
//---
   static EXPFUNC int __stdcall ServerClose();
//---
   static EXPFUNC int __stdcall SendToInt(SOCKET h_Client,             // in
                                          int   *ps32_SendBuf,         // in
                                          int    s32_SendBufLen);      // in -  SendBuf[] array size in int element
//---
   static EXPFUNC int __stdcall SendToDouble(SOCKET  h_Client,         // in
                                             double* pd_SendBuf,       // in
                                             int     s32_SendBufLen);  // in -  SendBuf[] array size in double element
//---
   static EXPFUNC int __stdcall SendToString(SOCKET h_Client,          // in
                                             char*  ps8_SendBuf,       // in
                                             INT   s32_SendBufLen);    // SendBuf string size in char element
//---
   static EXPFUNC int __stdcall ReadFromInt(SOCKET h_Client,           // in
                                            int   *ps32_ReadBuf,       // in
                                            int    s32_ReadBufLen,     // ReadBuf[] array size in int element
                                            int   *ps32_ReadLen);      // out - int ps32_ReadLen[1] - actual count of read data in int element
//---
   static EXPFUNC int __stdcall ReadFromDouble(SOCKET  h_Client,       // in
                                               double *pd_ReadBuf,     // in
                                               int     s32_ReadBufLen, // ReadBuf[] array size in double element
                                               int    *ps32_ReadLen);  // out - int ps32_ReadLen[1] - actual count of read data in double element
//---
   static EXPFUNC int __stdcall ReadFromString(SOCKET h_Client,        // in
                                               char  *ps8_ReadBuf,     // in
                                               int    s32_ReadBufLen,  // ReadBuf[] array size in char element
                                               int*   ps32_ReadLen);   // out - int ps32_ReadLen[1] - actual count of read data in char element
//---
protected:
   static DWORD SendTo(SOCKET h_Client,char *ps8_SendBuf,INT s32_SendBufLen);
   static DWORD ReadFrom(SOCKET h_Client,char *ps8_ReadBuf,INT s32_ReadBufLen,INT &s32_ReadLen);
  };
```

Now let's consider all the DLL functions in the order of their appearance in this file:

01. **ConnectTo** \- request to the server to create connection:


    ```
    static EXPFUNC int __stdcall ConnectTo(char* ps8_ServerIP, // in - ps8_ServerIP = "0123456789123456"
                                           int   s32_Port,     // in
                                           int*  ph_Client);   // out - int ph_Client[1]
    ```


    **Function parameters:**


    - **char\* ps8\_ServerIP** \- IP address of the server you want to connect to (for example, "93.127.110.162"). If the server is local, then specify the IP, not "127.0.0.1", but the obtained IP as it is described in the example 1.2.1.

    - **int s32\_Port** \- port number which the server "listens" to.

    - **int\* ph\_Client** \- connection handle is placed in this variable, if the function terminated successfully. This handle must be used in all subsequent operations for this connection.


If successful, the function returns true, otherwise it returns winsock2 API error code. C++ example:

```
DWORD u32_Error = cNetEventsProcDLL::ConnectTo(SERVER_IP, PORT, ph_Client);

if (u32_Error)
{
        printf("\nConnectTo(...) failed with error: %d\n", u32_Error);
        return 1;
}
else
        printf("\nConnectTo(...) OK, ph_Client[0] = : %d\n", ph_Client[0]);

int h_Client = ph_Client[0];
```

02. **ConnectClose** \- request to the server to close connection.


    ```
    static EXPFUNC int __stdcall ConnectClose(int h_Client); // in
    ```


    **Function parameters:**


    - **int h\_Client** \- connection handle that must be closed. If successful, the function returns true, otherwise it returns winsock2 API error code.


C++ example:

```
int u32_Error = cNetEventsProcDLL::ConnectClose(h_Client);

    if (u32_Error)
            printf("\nConnectClose(...) failed with error: %d\n", u32_Error);
    else
            printf("\nConnectClose(...) OK.\n");
```

03. **ServerOpen** \- request to create the server.


    ```
    static EXPFUNC int __stdcall ServerOpen(int s32_Port); //in
    ```


    **Function parameters:**


    - **int s32\_Port** \- port number which the server will "listen to" waiting for the clients requests. If successful, the function returns true, otherwise it returns winsock2 API error code.


C++ example:

```
int u32_Error = cNetEventsProcDLL::ServerOpen(s32_Port);

if (u32_Error)
{
        printf("\nServerOpen() failed with error: %d\n", u32_Error);
        return 1;
}
else
        printf("\nServerOpen() fine, we now are waiting for connections...\n");
```

04. **GetAllConnections** \- request to the server to get information about all current connections.


    ```
    static EXPFUNC int __stdcall GetAllConnections(int* ph_Client,         // out - int ph_Client[62]
                                                   int* ps32_ClientIP,     // out - int ps32_ClientIP[62]
                                                   int* ps32_ClientCount); // out - int ps32_ClientCount[1]
    ```


    **Function parameters:**


    - **int ph\_Client\[62\]** \- output array to which server places handles of all current connections.

    - **int ps32\_ClientIP\[62\]** \- output array to which server places IP addresses of all current connections. To convert these addresses to standard format like "92.127.110.161", use the **string FormatIP(int IP)** function for МetaТrader 4 Expert Advisor or any similar function given in examples of C++ programs. The number 62 in arrays size is specified not coincidentally: it designates the limit for the number of possible connections (clients) per one server.

    - **int\* ps32\_ClientCount** \- server places the number of the current connections into this variable, i.e. the number of elements in the above mentioned arrays.


If successful, the function returns true, otherwise it returns winsock2 API error code. C++ example:

```
int ph_Client[62];
int ps32_ClientIP[62];
int ps32_ClientCount[1];

int u32_Error = cNetEventsProcDLL::GetAllConnections(ph_Client, ps32_ClientIP, ps32_ClientCount);

if (u32_Error)
{
    printf("\nGetAllConnections(...) failed with error: %d\n", u32_Error);
}
else
   int u32_Count = ps32_ClientCount[0];
```

05. **DisconnectClient** \- request to the server to close connection with one of its clients.


    ```
    static EXPFUNC int __stdcall DisconnectClient(int h_Client); // in
    ```


    **Function parameters:**


    - **int h\_Client** \- connection handle that must be closed. If successful, the function returns true, otherwise it returns winsock2 API error code.


C++ example:

```
int u32_Error = cNetEventsProcDLL::DisconnectClient(h_Client);

if (u32_Error)
        printf("\nDisconnectClient(...) failed with error: %d\n", u32_Error);
else
        printf("\nDisconnectClient(...) OK.\n");
```

06. **ServerClose** \- request to close the server.


    ```
    static EXPFUNC int __stdcall ServerClose();
    ```


    When closing the server, all current connections will be closed, so that every client will get the "no connection" return code as a response for any operation. If successful, the function returns true, otherwise it returns winsock2 API error code. C++ example:


    ```
    int u32_Error = cNetEventsProcDLL::ServerClose();

    if (u32_Error)
            printf("\nServerClose() failed with error: %d\n", u32_Error);
    else
            printf("\nServerClose() OK.\n");
    ```


The next group of functions corresponds directly to data exchange via the current connection. Receiving data for each current connection is performed in the asynchronous mode, i.e. without response from the recipient.

All data are sent and received in the form of independent unit of exchange, i.e. block. All blocks for each recipient are accumulated in FIFO stack. The recipient can retrieve these blocks from the stack at any time. Each exchange function operates with one single block.

For data sending operations it is possible that operation was successful, but the return code differs from zero and can have the following values:

    - **WSAEWOULDBLOCK** \- operation was successful, data are not yet sent to the recipient, but will be sent in a suitable time. No user actions are required.

    - **WSA\_IO\_PENDING** \- previous data delivery is not yet complete, user must try to send data later. This is a regular situation, so it is considered that the function executed successfully.


Any other return code indicates the user error.

08. **SendToInt** \- request to send data block (int type array) via the current connection.


    ```
    static EXPFUNC int __stdcall SendToInt(SOCKET h_Client,        // in
                                           int*   ps32_SendBuf,    // in
                                           int    s32_SendBufLen); // in - SendBuf[] array size in int element
    ```


    **Function parameters:**


    - **SOCKET h\_Client** \- handle of current connection.

    - **int ps32\_SendBuf\[s32\_SendBufLen\]** \- single block (int type array) you need to send to the client.

    - **int s32\_SendBufLen** \- array size.


If successful, the function returns true, otherwise it returns winsock2 API error code. C++ example:

```
int ps32_SendBuf[200];
int s32_SendBufLen=200;

int u32_Error = cNetEventsProcDLL::SendToInt(h_Client, ps32_SendBuf, s32_SendBufLen);

switch (u32_Error)
{
   case 0:
       printf("\nSendTo(...) OK");
       break;
   case WSAEWOULDBLOCK:
       printf("\nSendTo(%d...) return is: WSAEWOULDBLOCK(%d)\n",h_Client, u32_Error);
       printf("\nWSAEWOULDBLOCK -> The data will be send after the next FD_WRITE event, do nouthing\n");
       break;
   case WSA_IO_PENDING:
       printf("\nSendTo(%d...) return is: WSA_IO_PENDING(%d)\n",h_Client, u32_Error);
       printf("\nWSA_IO_PENDING -> Error: A previous Send operation is still pending. This data will not be sent, try latter\n");
       break;

   default:
       printf("\nSendTo(%d...)failed with severe error: %d\n",h_Client, u32_Error);
       break;
};
```

09. **SendToDouble** \- request to send data block (double type array) via the current connection.


    ```
    static EXPFUNC int __stdcall SendToDouble(SOCKET h_Client,          // in
                                              double*  pd_SendBuf,      // in
                                              int      s32_SendBufLen); // in - SendBuf[] array size in int element
    ```


    **Function parameters:**


    - **SOCKET h\_Client** \- handle of current connection.

    - **double pd\_SendBuf\[s32\_SendBufLen\]** \- single block (double type array) you need to send to the client.

    - **int s32\_SendBufLen** \- array size.


If successful, the function returns true, otherwise it returns winsock2 API error code. C++ example:

```
double pd_SendBuf[200];
int    s32_SendBufLen=200;

int u32_Error = cNetEventsProcDLL::SendToDouble(h_Client, pd_SendBuf, s32_SendBufLen);

switch (u32_Error)
{
   case 0:
       printf("\nSendTo(...) OK");
       break;
   case WSAEWOULDBLOCK:
       printf("\nSendTo(%d...) return is: WSAEWOULDBLOCK(%d)\n",h_Client, u32_Error);
       printf("\nWSAEWOULDBLOCK -> The data will be send after the next FD_WRITE event, do nouthing\n");
       break;
   case WSA_IO_PENDING:
       printf("\nSendTo(%d...) return is: WSA_IO_PENDING(%d)\n",h_Client, u32_Error);
       printf("\nWSA_IO_PENDING -> Error: A previous Send operation is still pending. This data will not be sent, try latter\n");
       break;

   default:
       printf("\nSendTo(%d...)failed with severe error: %d\n",h_Client, u32_Error);
       break;
};
```

10. **SendToString** \- request to send data block (char type array) via the current connection.


    ```
    static EXPFUNC int __stdcall SendToString(SOCKET h_Client,        // in
                                              char*  ps8_SendBuf,     // in
                                              int    s32_SendBufLen); // in -  SendBuf[] array size in int element
    ```


    **Function parameters:**


    - **SOCKET h\_Client** \- handle of current connection.

    - **char ps8\_SendBuf\[s32\_SendBufLen\]** \- single block (char type array) you need to send to the client.

    - **int s32\_SendBufLen** \- array size.


If successful, the function returns true, otherwise it returns winsock2 API error code. C++ example:

```
char ps8_SendBuf[200];
int  s32_SendBufLen=200;

int u32_Error = cNetEventsProcDLL::SendToString(h_Client, ps8_SendBuf, s32_SendBufLen);

switch (u32_Error)
{
   case 0:
       printf("\nSendTo(...) OK");
       break;
   case WSAEWOULDBLOCK:
       printf("\nSendTo(%d...) return is: WSAEWOULDBLOCK(%d)\n",h_Client, u32_Error);
       printf("\nWSAEWOULDBLOCK -> The data will be send after the next FD_WRITE event, do nouthing\n");
       break;
   case WSA_IO_PENDING:
       printf("\nSendTo(%d...) return is: WSA_IO_PENDING(%d)\n",h_Client, u32_Error);
       printf("\nWSA_IO_PENDING -> Error: A previous Send operation is still pending. This data will not be sent, try latter\n");
       break;

   default:
       printf("\nSendTo(%d...)failed with severe error: %d\n",h_Client, u32_Error);
       break;
};
```

11. **ReadFromInt** \- request to receive data block (int type array) via the current connection.


    ```
    static EXPFUNC int __stdcall ReadFromInt(SOCKET h_Client,       // in
                                             int*   ps32_ReadBuf,   // in
                                             int    s32_ReadBufLen, // ReadBuf[] array size in int element
                                             int*   ps32_ReadLen);  // out - int ps32_ReadLen[1] - actual count of read data in int element
    ```


    **Function parameters:**


    - **SOCKET h\_Client** \- handle of current connection.

    - **int ps32\_ReadBuf\[s32\_ReadBufLen\]** \- int type array for receiving data block.

    - **int s32\_ReadBufLen** \- size of receiving array.

    - **int\* ps32ReadLen** \- this variable holds the real size of data block that was received and placed into the ps32\_ReadBuf\[\] array. If the size of the receiving array is not sufficient to receive data block, then this variable will hold the size, required to receive data block, with a minus sign. The block stays in the stack, and the return code will be equal to zero. If there is no data in the client stack with specified handle, this variable will be equal to zero and the return code will be also equal to zero.


If successful, the function returns true, otherwise it returns winsock2 API error code. C++ example:

```
int ps32_ReadBuf[2000];
int s32_ReadBufLen=2000;
int ps32_ReadLen[1];
int u32_Error = cNetEventsProcDLL::ReadFromInt(h_Client, ps32_ReadBuf, s32_ReadBufLen, ps32_ReadLen);

if(u32_Error)
    printf("ReadFromInt(%d...) failed with error: %d", h_Client, u32_Error);
else
    if(ps32_ReadLen[0] >= 0)
        printf("ReadFromInt(%d...) fine, %d int number was read", h_Client, ps32_ReadLen[0]);
    else
        printf("ReadFromInt(%d...) fine, but ReadBuf must be at least %d int number size", h_Client, -ps32_ReadLen[0]);
```

12. **ReadFromDouble** \- request to receive data block (double type array) via the current connection.


    ```
    static EXPFUNC int __stdcall ReadFromDouble(SOCKET h_Client,        // in
                                                double* pd_ReadBuf,     // in
                                                int     s32_ReadBufLen, // ReadBuf[] array size in double element
                                                int*    ps32_ReadLen);  // out - int ps32_ReadLen[1] - actual count of read data in double element
    ```


    **Function parameters:**


    - **SOCKET h\_Client** \- handle of current connection.

    - **double pd\_ReadBuf\[s32\_ReadBufLen\]** \- double type array for receiving data block.

    - **int s32\_ReadBufLen** \- size of receiving array.

    - **int\* ps32ReadLen** \- this variable holds the real size of data block received and placed into the ps32\_ReadBuf\[\] array. If the size of the receiving array is not sufficient to receive data block, then this variable will hold the size, required to receive data block, with a minus sign. The block stays in the stack, and the return code will be equal to zero. If there is no data in the client stack with specified handle, this variable will be equal to zero and the return code will be also equal to zero.


If successful, the function returns true, otherwise it returns winsock2 API error code. C++ example:

```
double ps32_ReadBuf[2000];
int    s32_ReadBufLen = 2000;
int    ps32_ReadLen[1];
int    u32_Error = cNetEventsProcDLL::ReadFromDouble(h_Client, pd_ReadBuf, s32_ReadBufLen, ps32_ReadLen);

if(u32_Error)
    printf("ReadFromDouble(%d...) failed with error: %d", h_Client, u32_Error);
else
    if(ps32_ReadLen[0] >= 0)
        printf("ReadFromDouble(%d...) fine, %d double number was read", h_Client, ps32_ReadLen[0]);
    else
        printf("ReadFromDouble(%d...) fine, but ReadBuf must be at least %d double number size", h_Client, -ps32_ReadLen[0]);
```

13. **ReadFromString** \- request to receive data block (char type array) via the current connection.


    ```
    static EXPFUNC int __stdcall ReadFromString(SOCKET h_Client,       // in
                                                char*  ps8_ReadBuf,    // in
                                                int    s32_ReadBufLen, // ReadBuf[] array size in char element
                                                int*   ps32_ReadLen);  // out - int ps32_ReadLen[1] - actual count of read data in char element
    ```


    **Function parameters:**


    - **SOCKET h\_Client** \- handle of current connection.

    - **char ps8\_ReadBuf\[s32\_ReadBufLen\]** \- char type array for receiving data block.

    - **int s32\_ReadBufLen** \- size of receiving array.

    - **int\* ps32ReadLen** \- this variable holds the real size of data block received and placed into the ps32\_ReadBuf\[\] array. If the size of the receiving array is not sufficient to receive data block, then this variable will hold the size, required to receive data block, with a minus sign. The block stays in the stack, and the return code will be equal to zero. If there is no data in the client stack with specified handle, this variable will be equal to zero and the return code will be also equal to zero.


If successful, the function returns true, otherwise it returns winsock2 API error code. C++ example:

```
char ps8_ReadBuf[2000];
int  s32_ReadBufLen = 2000;
int  ps32_ReadLen[1];
int  u32_Error = cNetEventsProcDLL::ReadFromString(h_Client, ps8_ReadBuf, s32_ReadBufLen, ps32_ReadLen);

if(u32_Error)
    printf("ReadFromStrung(%d...) failed with error: %d", h_Client, u32_Error);
else
    if(ps32_ReadLen[0] >= 0)
        printf("ReadFromString(%d...) fine, %d char was read", h_Client, ps32_ReadLen[0]);
    else
        printf("ReadFromString(%d...) fine, but ReadBuf must be at least %d char size", h_Client, -ps32_ReadLen[0]);
```

### 3\. Project Implementation

The attached NetServerClient.zip archive contains two Microsoft Visual Studio 2010 Ultimate projects:

- **NetEventsProc** \- to build NetEventsProc.exe
- **NetEventsProcDLL** \- to build NetEventsProcDLL.dll

The source codes are commented in details. You can look into the details of implementation and customize the projects to your specific needs if you like.

NetEventsProc.exe implements server and clients using asynchronous sockets. To switch sockets to the asynchronous mode, one of the possible methods of operating in asynchronous mode is used: binding sockets to the WSAEventSelect(h\_Socket, h\_Event, FD\_ALL\_EVENTS) network events.

If this article causes interest of the readers, then in the next version of the article we will discuss all the details of implementation. But that's all for now. Note once again that this project is based on the fundamental work of a great Master [Elmue](https://www.mql5.com/go?link=https://www.codeproject.com/Articles/34163/A-Universal-TCP-Socket-Class-for-Non-blocking-Serv "/go?link=https://www.codeproject.com/Articles/34163/A-Universal-TCP-Socket-Class-for-Non-blocking-Serv").

### 4\. Conclusion

I hope that this article will solve the problem of information exchange between МetaТrader 4 terminal and the 3rd party applications regardless of their location: whether on local or remote computer in relation to the installed terminal. I hope that this article does not have a lot of [Spaghetti code](https://en.wikipedia.org/wiki/Spaghetti_code "https://en.wikipedia.org/wiki/Spaghetti_code"), and the process of using DLL functions is quite simple and clear. At least, I made efforts to do so.

**One more important moment should be noted:**

Almost all computers are nodes of a local network ( **LAN**). Even if you have only one computer, so it is most probably the node of LAN consisted of one computer.

Connection to the wide area network ( **WAN**) is performed via additional hardware device that can be called router, modem or some other technical term. For simplicity we will call it router. It is for this router providers allocate global IP address.

Routers solve certain security problems when working with WAN, allow to spare global IP addresses, help to arrange WAN connection from the local network. But at the same time in fact they distort the initial sense of the WAN that implies possibility of a direct Peer-To-Peer connection of any two computers.

Such effect is caused by the fact that practically any router performs the so called Network Address Translation ( **NAT**). The network address is represented by the <protocol, IP, port> tuple. Any element of this tuple can be changed by router, it all depends on particular router model.

In this case the computer that access WAN from LAN does not have all those benefits provided by pure WAN. The vast majority of routers has the **OUTBOUND** feature that allows router to remember network address of LAN client that addressed the global WAN network from LAN with some request.

Thanks to this feature, router can send to client all the information received by router as the response to the client request. Thus LAN client can connect to WAN servers. However this may not always be true, as for security reasons and work discipline some network addresses can be hardware blocked.

Therefore, to organize server on a LAN computer you must set up the so called **port forwarding**. As for the example of this article, in the simplest case of one LAN computer you need to forward port number 2000. You can either do it yourself by connecting to the router mini site in browser or you can appeal to a professional. Most probably this mini site is available at 192.168.1.1.

**This all should be considered, if you want to have possibility to exchange information via WAN.**

In the next version of the article we will consider **Peer-To-Peer (p2p)** type of connection.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/1361](https://www.mql5.com/ru/articles/1361)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/1361.zip "Download all attachments in the single ZIP archive")

[EXE.zip](https://www.mql5.com/en/articles/download/1361/EXE.zip "Download EXE.zip")(21.35 KB)

[FastStart.zip](https://www.mql5.com/en/articles/download/1361/FastStart.zip "Download FastStart.zip")(28.26 KB)

[NetServerClient.zip](https://www.mql5.com/en/articles/download/1361/NetServerClient.zip "Download NetServerClient.zip")(56.32 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/39122)**
(28)


![lake274](https://c.mql5.com/avatar/avatar_na2.png)

**[lake274](https://www.mql5.com/en/users/lake274)**
\|
9 Nov 2015 at 16:06

Hello,

Anyone can show me an example how to I must reference NetEventsProcDLL.dll [functions](https://www.mql5.com/en/docs/constants/namedconstants/compilemacros "MQL5 documentation: Predefined Macrosubstitutions") in VB.Net? Thanks in advance!


![astalabimsala](https://c.mql5.com/avatar/avatar_na2.png)

**[astalabimsala](https://www.mql5.com/en/users/astalabimsala)**
\|
19 Nov 2015 at 14:58

Dear Sergey,

I'm testing your code (EA server and c++ client) for a couple of days, without changing any code. Whatever data I sent from client.exe, EA server looks like reading it, gives the length of [data correctly](https://www.mql5.com/en/docs/matrix/matrix_manipulations "MQL5 Documentation: Matrix and vector manipulations") (let say 7), but prints the read data as "0123456", same anomaly goes on the C++client side... it just prints "o" as received string, not the characters I've sent but the read data length correct again, and this is going on without any errors in both side... The code I wrote on Labview also doing the exact same thing... Can this be about unicode/Ansi change you mention? any clue about how to fix it?

Many thanks in advance..

Suha

![Stefano Cerbioni](https://c.mql5.com/avatar/2016/6/5755B8A8-5894.png)

**[Stefano Cerbioni](https://www.mql5.com/en/users/faustf)**
\|
13 Jun 2016 at 10:56

hi guy i try  to use  this  awesome  script , i  choice  the  3  possibility

**1.2.3. МetaТrader 4 Expert Advisor-indicator (the Expert Advisor-server) & МetaТrader 4 Client-indicator**

th **is is  my step i did do**

**1\.**

- **NetEventsProcDLL.dll** \- place it into the "C:\\Windows\\System32\\" folder.

- **NetEventsProc.exe** \- create
"C:\\NetEventsProc\\" folder and place


- **ImportNetEventsProcDLL.mqh**
\-
"MetaTrader 4\\include".


2.  Place the **ServerSendInd.mq4** file in the [terminal data folder](https://www.mql5.com/en/docs/constants/environment_state/terminalstatus#enum_terminal_info_string "MQL5 documentation: Client Terminal Properties") "МetaТrader 4\\experts\\" and compile it.

3\. Place the **ClientIndicator.mq4** file in the terminal data folder "МetaТrader 4\\indicators\\" and assign the local IP in my case 192.168.1.105 (ip of    my server)

4\. Run **ServerSendInd**.  my console  answer  this "2016.06.13 10:51:47.129    ServerSendInd USDCHF,H1: ClientCount = 0" i think is correct

5\. Now in pc  where  i have  my  clientindicator.mq4  run it but  i have this  error

'ClientIndicator.mq4'    ClientIndicator.mq4    1    1

'ImportNetEventsProcDLL.mqh'    ImportNetEventsProcDLL.mqh    1    1

function must return a value    ClientIndicator.mq4    146    5

declaration of 's32\_Error' hides global declaration in file 'ClientIndicator.mq4' at line 27    ImportNetEventsProcDLL.mqh    130    22

'87' - case value already used    ImportNetEventsProcDLL.mqh    206    8

1 error(s), 2 warning(s)        2    3

someone can help me???   thankz  so  much

the  pc  work with  windows 7 32bit  with out  firewall  in my lan

server work in 192.168.1.105 and  client in 192.168.1.106

![Stefano Cerbioni](https://c.mql5.com/avatar/2016/6/5755B8A8-5894.png)

**[Stefano Cerbioni](https://www.mql5.com/en/users/faustf)**
\|
14 Jun 2016 at 12:17

hi  guy

i  have  the  same  error  of   **michaelt4268**

**i  tryed  to modify  but  now  i have  this  error**

**'ConnectTo' - [function](https://www.mql5.com/en/docs/constants/namedconstants/compilemacros "MQL5 documentation: Predefined Macrosubstitutions") must have a body    ClientIndicator.mq4    80    19**

**and  the  error is  in line  modify**

//s32\_Error = ConnectTo(ps8\_ServerIP, s32\_Port, ph\_Client);

      s32\_Error = ConnectTo(uc\_ServerIP, s32\_Port, ph\_Client);

anyone  can helpp me ??' thankz

![Vlad Sipos](https://c.mql5.com/avatar/avatar_na2.png)

**[Vlad Sipos](https://www.mql5.com/en/users/shypy)**
\|
16 Nov 2016 at 14:58

Hello!

I have a problem with the loading of the dll. I get error Cannot load 'NetEventsProcDLL.dll' \[1114\]. I get this error on both my laptop [and vps](https://www.mql5.com/en/vps "Forex VPS for MetaTrader 4/5"). Can anyone help me with this issue?  I searched for a way to fix this issue but nothing works.

![Calculation of Integral Characteristics of Indicator Emissions](https://c.mql5.com/2/0/avatar__22.png)[Calculation of Integral Characteristics of Indicator Emissions](https://www.mql5.com/en/articles/610)

Indicator emissions are a little-studied area of market research. Primarily, this is due to the difficulty of analysis that is caused by the processing of very large arrays of time-varying data. Existing graphical analysis is too resource intensive and has therefore triggered the development of a parsimonious algorithm that uses time series of emissions. This article demonstrates how visual (intuitive image) analysis can be replaced with the study of integral characteristics of emissions. It can be of interest to both traders and developers of automated trading systems.

![Change Expert Advisor Parameters From the User Panel "On the Fly"](https://c.mql5.com/2/0/avatar__24.png)[Change Expert Advisor Parameters From the User Panel "On the Fly"](https://www.mql5.com/en/articles/572)

This article provides a small example demonstrating the implementation of an Expert Advisor whose parameters can be controlled from the user panel. When changing the parameters "on the fly", the Expert Advisor writes the values obtained from the info panel to a file to further read them from the file and display accordingly on the panel. This article may be relevant to those who trade manually or in semi-automatic mode.

![MQL5 Cookbook: Using Different Print Modes](https://c.mql5.com/2/0/avatar2__2.png)[MQL5 Cookbook: Using Different Print Modes](https://www.mql5.com/en/articles/638)

This is the first article of the MQL5 Cookbook series. I will start with simple examples to allow those who are taking their first steps in programming to gradually become familiar with the new language. I remember my first efforts at designing and programming trading systems which I may say was quite difficult, given the fact that it was the first programming language in my life. However, it turned out to be easier than I thought and it only took me a few months before I could develop a fairly complex program.

![Fast Testing of Trading Ideas on the Chart](https://c.mql5.com/2/0/avatar__23.png)[Fast Testing of Trading Ideas on the Chart](https://www.mql5.com/en/articles/505)

The article describes the method of fast visual testing of trading ideas. The method is based on the combination of a price chart, a signal indicator and a balance calculation indicator. I would like to share my method of searching for trading ideas, as well as the method I use for fast testing of these ideas.

[![](https://www.mql5.com/ff/sh/wm94j0jmkwd29943z2/ddfa713cb3cdd580c3e81e0e13b5b1b8.jpg)\\
Revised MetaTrader 5 Web Terminal\\
\\
Trade with no restrictions from any mobile device, OS and web browser\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=fkjlpstbxdmrrwpblfatcsdjyrxbizyj&s=f462f051eb7aaec36d6b31792d312d60d3f5a50c83b12d0d66e85d5d61bd941b&uid=&ref=https://www.mql5.com/en/articles/1361&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5068245951488915250)

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