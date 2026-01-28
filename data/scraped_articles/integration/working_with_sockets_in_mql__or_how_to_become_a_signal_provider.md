---
title: Working with sockets in MQL, or How to become a signal provider
url: https://www.mql5.com/en/articles/2599
categories: Integration
relevance_score: 1
scraped_at: 2026-01-23T21:43:27.618084
---

[![](https://www.mql5.com/ff/sh/zf7a2k61x98jzh89z2/01.png)Speed up your tradingUse our high-speed VPS for MetaTrader 4 and 5Learn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=qtrrsuiwuicrscmckjynyanztbditglq&s=c617dc80d90cfd3783ec1345eec2b419b281f10fec6eac77b3218984ac337259&uid=&ref=https://www.mql5.com/en/articles/2599&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5072045623156355917)

MetaTrader 5 / Examples


### A bit of pathos

Sockets… What in our IT world could possibly exist without them? Dating back to 1982, and hardly changed up to the present time, they smoothly work for us every second. This is the foundation of network, the nerve endings of the Matrix we all live in.

In the morning, you turn on the MetaTrader terminal, and it immediately creates sockets and connects to the servers. You open a browser, and dozens of socket connections are created and closed in order to deliver the information from the Web to you, or to send e-mails, accurate time signals, gigabytes of distributed computing.

First, a little theory is required. Take a look at [Wiki](https://en.wikipedia.org/wiki/Berkeley_sockets "https://en.wikipedia.org/wiki/Berkeley_sockets") or [MSDN](https://www.mql5.com/go?link=https://msdn.microsoft.com/en-us/library/windows/desktop/ms740506(v=vs.85).aspx "https://msdn.microsoft.com/en-us/library/windows/desktop/ms740506(v=vs.85).aspx"). The corresponding articles describe all the necessary arsenal of structures and functions, as well as provide examples of setting up a client and a server.

In this article, translation of this knowledge into MQL will be considered.

### 1\. Porting structures and functions from WinAPI

It is no secret that the WinAPI was designed for the C language. And the MQL language has practically become its blood brother (both in spirit and working style). Let us create an mqh file for these WinAPI functions, which will be used in the main MQL program. The order of our actions is to port them as necessary.

For the TCP client only a few functions are required:

- initialize the library with **WSAStartup**();
- create a socket with **socket**();
- set to nonblocking mode with **ioctlsocket**(), in order not to freeze while waiting for data;
- connect to the server with **connect**();
- listen with **recv**() or send the data using **send**() until the end of the program or a connection loss;
- close the socket with **closesocket**() after work, and deinitialize the library using **WSACleanup**().

A TCP server requires similar functions, with the exception that it will be bound to a specific port and will put the socket into listening mode. The necessary steps are:

- initialize the library - WSAStartup();
- create a socket - socket();
- set to nonblocking mode - ioctlsocket();
- bind to a port - **bind**();
- put into listening mode - **listen**();
- after successful creation listen for **accept**();
- create client connections and continue to work with them in recv()/send() mode until the end of the program or a connection loss;
- after work, close the listening socket of the server and connected clients using closesocket() and deinitialize the library with WSACleanup().

In the case of a UDP socket, there will be fewer steps (in fact there is no "handshake" between client and server). UDP client:

- initialize the library - WSAStartup();
- create a socket - socket();
- set to nonblocking mode with ioctlsocket(), in order not to freeze while waiting for data;
- send - **sendto**() /receive - **recvfrom**() the data;
- close the socket with closesocket() after work, and deinitialize the library using WSACleanup().

Only a single bind function is added in a UDP server:

- initialize the library - WSAStartup();
- create a socket - socket();
- set to nonblocking mode - ioctlsocket();
- bind to a port - bind();
- receive - recvfrom() / send - sendto();
- after work, close the listening socket of the server and connected clients using closesocket() and deinitialize the library with WSACleanup().


As you can see, the path is not too complicated, but the structures will need to be filled to call each function.

**a) WSAStartup()**

See the full description in [MSDN](https://www.mql5.com/go?link=https://msdn.microsoft.com/en-us/library/windows/desktop/ms742213(v=vs.85).aspx "https://msdn.microsoft.com/en-us/library/windows/desktop/ms742213(v=vs.85).aspx"):

```
WINAPI:
int WSAAPI WSAStartup(_In_ WORD wVersionRequested,  _Out_ LPWSADATA lpWSAData);
```

\_In\_, \_Out\_ are empty defines, which point to the scope of parameter. WSAAPI describes the rule for passing parameters, but for our purposes, it can also be left blank.

As can be seen from the documentation, a [MAKEWORD](https://www.mql5.com/go?link=https://msdn.microsoft.com/en-us/library/windows/desktop/ms632663(v=vs.85).aspx "https://msdn.microsoft.com/en-us/library/windows/desktop/ms632663(v=vs.85).aspx") macro will also be necessary to specify the required version in the first parameter, as well as a pointer to the LPWSADATA structure. The macro is not difficult to create, copy it from the header file:

```
#define MAKEWORD(a, b)      ((WORD)(((BYTE)(((DWORD_PTR)(a)) & 0xff)) | ((WORD)((BYTE)(((DWORD_PTR)(b)) & 0xff))) << 8))
```

Moreover, all data types can also be easily defined in terms of MQL:

```
#define BYTE         uchar
#define WORD         ushort
#define DWORD        int
#define DWORD_PTR    ulong
```

Copy the [WSADATA](https://www.mql5.com/go?link=https://msdn.microsoft.com/en-us/library/windows/desktop/ms741563(v=vs.85).aspx "https://msdn.microsoft.com/en-us/library/windows/desktop/ms741563(v=vs.85).aspx") structure from MSDN. The names of most data types should be left unchanged for ease of reading, especially since they have already been defined above.

```
struct WSAData
{
  WORD wVersion;
  WORD  wHighVersion;
  char szDescription[WSADESCRIPTION_LEN+1];
  char szSystemStatus[WSASYS_STATUS_LEN+1];
  ushort iMaxSockets;
  ushort iMaxUdpDg;
  char  lpVendorInfo[];
}
```

Note that the last parameter _lpVendorInfo_ is defined as an array in MQL (in C it was a pointer to char\*). Move the array size constants to the defines as well. Finally, define the pointer to a structure as:

```
#define LPWSADATA        char&
```

Why so? It's simple. Any structure is nothing more than a limited [chunk of memory](https://www.mql5.com/en/articles/364#2). It can be represented in any way - for example, as another structure with the same size or as an array of the same size. Here, the representation as an array will be used, therefore, in all functions the **char&** type will be the address of the array with size corresponding to the size of the required structure. The resulting declaration of the function in MQL looks as follows:

```
MQL:
int WSAStartup(WORD wVersionRequested, LPWSADATA lpWSAData[]);
```

This is how the function call and passing the obtained result to the WSAData structure looks like:

```
char wsaData[]; // byte array of the future structure
ArrayResize(wsaData, sizeof(WSAData)); // resize it to the size of the structure
WSAStartup(MAKEWORD(2,2), wsaData); // call the function
```

The data will be passed to the _wsaData_ byte array, from which it is easy to collect information using casts.

Hopefully, this part was not too difficult — after all, it is only the first function, and already so much work needs to be done. But now the basic principle is clear, so it will get easier and more interesting.

**b) socket()**

```
WINAPI:
SOCKET WSAAPI socket(_In_ int af, _In_ int type, _In_ int protocol);
```

Do the same - copy data from [MSDN](https://www.mql5.com/go?link=https://msdn.microsoft.com/en-us/library/windows/desktop/ms740506(v=vs.85).aspx "https://msdn.microsoft.com/en-us/library/windows/desktop/ms740506(v=vs.85).aspx").

Since we are using TCP sockets for IPv4, set the constants for the parameters of this function right away:

```
#define SOCKET           uint
#define INVALID_SOCKET   (SOCKET)(~0)
#define SOCKET_ERROR            (-1)
#define NO_ERROR         0
#define AF_INET          2 // internetwork: UDP, TCP, etc.
#define SOCK_STREAM      1
#define IPPROTO_TCP      6
```

**c) ioctlsocket()**

```
MQL:
int ioctlsocket(SOCKET s, int cmd, int &argp);
```

It has its last argument changed from a pointer to an address:

**d) connect()**

```
WINAPI:
int connect(_In_ SOCKET s, _In_ const struct sockaddr *name, _In_ int namelen);
```

There is a small difficulty with passing the _sockaddr_ structure, but the main principle is already known – replace the structures with byte arrays and use them to pass data to the WinAPI functions.

Take the structure from MSDN with no changes:

```
struct sockaddr
{
    ushort sa_family; // Address family.
    char sa_data[14]; // Up to 14 bytes of direct address.
};
```

As agreed, the pointer to it will be implemented using the array address:

```
#define LPSOCKADDR    char&
```

Examples in MSDN use the _sockaddr\_in_ structure. It is similar in size, but the parameters are declared differently:

```
struct sockaddr_in
{
    short   sin_family;
    ushort sin_port;
    struct  in_addr sin_addr;
    char    sin_zero[8];
};
```

The data for the _sin\_addr_ is a 'union', one representation of which is an eight-byte integer:

```
struct in_addr
{
   ulong s_addr;
};
```

This is how the resulting declaration of the function looks in MQL:

```
MQL:
int connect(SOCKET s, LPSOCKADDR name[], int namelen);
```

At this stage, we are fully prepared to create a client socket. Only a little is left to do - the function to receive and send data.

**e) recv() and send()** for TCP

The prototypes look like:

```
WINAPI:
int send(_In_ SOCKET s, _In_ const char* buf, _In_ int len, _In_ int flags);
int recv(_In_  SOCKET s, _Out_     char* buf, _In_ int len, _In_ int flags);
MQL:
int send(SOCKET s, char& buf[], int len, int flags);
int recv(SOCKET s, char& buf[], int len, int flags);
```

as it can be seen, the second parameter was changed from a char\* pointer to a char& \[\] array

**f) recvfrom() and sendto()** for UPD

The prototypes in MQL look like:

```
WINAPI:
int recvfrom(_In_  SOCKET s, _Out_ char* buf, _In_ int len, _In_ int flags, _Out_  struct sockaddr *from,
  _Inout_opt_ int *fromlen);
int sendto(_In_ SOCKET s, _In_ const char* buf, _In_ int len, _In_ int flags,  _In_ const struct sockaddr *to,
  _In_       int tolen);
MQL:
int recvfrom(SOCKET s,char &buf[],int len,int flags,LPSOCKADDR from[],int &fromlen);
int sendto(SOCKET s,const char &buf[],int len,int flags,LPSOCKADDR to[],int tolen);
```

And finally, the two important functions for clearing and closing the handles after work:

**g) closesocket() and WSACleanup()**

```
MQL:
int closesocket(SOCKET s);
int WSACleanup();
```

The resulting file of the ported WinAPI functions:

```
#define BYTE              uchar
#define WORD              ushort
#define DWORD             int
#define DWORD_PTR         ulong
#define SOCKET            uint

#define MAKEWORD(a, b)      ((WORD)(((BYTE)(((DWORD_PTR)(a)) & 0xff)) | ((WORD)((BYTE)(((DWORD_PTR)(b)) & 0xff))) << 8))

#define WSADESCRIPTION_LEN      256
#define WSASYS_STATUS_LEN       128

#define INVALID_SOCKET  (SOCKET)(~0)
#define SOCKET_ERROR    (-1)
#define NO_ERROR        0
#define SOMAXCONN       128

#define AF_INET         2 // internetwork: UDP, TCP, etc.
#define SOCK_STREAM     1
#define IPPROTO_TCP     6

#define SD_RECEIVE      0x00
#define SD_SEND         0x01
#define SD_BOTH         0x02

#define IOCPARM_MASK    0x7f            /* parameters must be < 128 bytes */
#define IOC_IN          0x80000000      /* copy in parameters */
#define _IOW(x,y,t)     (IOC_IN|(((int)sizeof(t)&IOCPARM_MASK)<<16)|((x)<<8)|(y))
#define FIONBIO         _IOW('f', 126, int) /* set/clear non-blocking i/o */
//------------------------------------------------------------------    struct WSAData
struct WSAData
  {
   WORD              wVersion;
   WORD              wHighVersion;
   char              szDescription[WSADESCRIPTION_LEN+1];
   char              szSystemStatus[WSASYS_STATUS_LEN+1];
   ushort            iMaxSockets;
   ushort            iMaxUdpDg;
   char              lpVendorInfo[];
  };

#define LPWSADATA               char&
//------------------------------------------------------------------    struct sockaddr_in
struct sockaddr_in
  {
   ushort            sin_family;
   ushort            sin_port;
   ulong             sin_addr; //struct in_addr { ulong s_addr; };
   char              sin_zero[8];
  };
//------------------------------------------------------------------    struct sockaddr
struct sockaddr
  {
   ushort            sa_family; // Address family.
   char              sa_data[14]; // Up to 14 bytes of direct address.
  };
#define LPSOCKADDR      char&

struct ref_sockaddr { char ref[2+14]; };

//------------------------------------------------------------------    import Ws2_32.dll
#import "Ws2_32.dll"
int WSAStartup(WORD wVersionRequested,LPWSADATA lpWSAData[]);
int WSACleanup();
int WSAGetLastError();

ushort htons(ushort hostshort);
ulong inet_addr(char& cp[]);
string inet_ntop(int Family,ulong &pAddr,char &pStringBuf[],uint StringBufSize);
ushort ntohs(ushort netshort);

SOCKET socket(int af,int type,int protocol);
int ioctlsocket(SOCKET s,int cmd,int &argp);
int shutdown(SOCKET s,int how);
int closesocket(SOCKET s);

// server function
int bind(SOCKET s,LPSOCKADDR name[],int namelen);
int listen(SOCKET s,int backlog);
SOCKET accept(SOCKET s,LPSOCKADDR addr[],int &addrlen);

// client function
int connect(SOCKET s,LPSOCKADDR name[],int namelen);
int send(SOCKET s,char &buf[],int len,int flags);
int recv(SOCKET s,char &buf[],int len,int flags);

#import
```

### 2\. Creating a client and a server

After reflecting for some time on the way to implement the work with sockets for further experiments, the choice fell on the demonstration of working with its functions without classes. First, this will give a better understanding of the fact that only linear non-branching programming is involved here. Second, it will allow to refactor the functions according to any needs and any OOP ideology. Experience shows that programmers tend to thoroughly go through simple classes to understand how everything works.

Important! In all your experiments do not forget that a bound port is not released automatically when the server code is aborted. This would cause the repeated creation of a socket and an attempt of 'bind' call to result in an error – Address already in use. To solve this problem, either use the [SO\_REUSEADDR](https://www.mql5.com/go?link=https://msdn.microsoft.com/en-us/library/windows/desktop/ms740476(v=vs.85).aspx "https://msdn.microsoft.com/en-us/library/windows/desktop/ms740476(v=vs.85).aspx") option on the socket, or simply restart the terminal. Use monitoring utilities, such as TCPViewer, to track the sockets created in your OS.

It is also necessary to understand that client can connect to server, provided that the server is not hidden behind a NAT, or the port for the client/server is not blocked by the OS or router.

Therefore, it is possible to experiment with the server and client locally on a single computer. But to fully operate with multiple clients, the server must be run at least on a VPS with "white" external IP address and using an open outgoing port.

### Example 1. Sending the chart layout to clients

Start with a simple interaction – one-time transfer of a tpl file from the server to the client.

In this case, there is no need to maintain the _send/recv_ loop on the client side, as it is necessary to receive only one data portion once connected and then disconnect. The connection will be closed by the server immediately once the data is sent.

That is, when a client connects to the server, the server makes a Send call and shuts down the socket. At the same time, the client makes a Recv call and similarly shuts down the socket. Of course, in the more interesting cases, it is possible to create a constant broadcast of the chart changes, such as instant synchronization of the client and the server charts. This would be useful for trading guru who can show their charts to young padawans online. But today it is done by broadcasting a video stream from the screen through different webinar software, or Skype. Therefore, this topic is best discussed on the forum.

Who and when would find this code example useful? For instance, you place your indicators or graphic objects on the chart on a daily, hourly or minutely basis. At the same time, you have a server EA running on a separate chart, which listens to client connections and gives them the current tpl of the required symbol and period.

Satisfied customers will now be informed about the targets and trading signals from you. It will be sufficient for them to periodically run a script that downloads tpl from the server and applies it to the chart.

So, let's start with the server. Everything works in the _OnTimer_ event, which serves as the thread of the EA. Every second it checks the key blocks of the server: Listening for client -> Sending the data -> Closing connection. It also checks for activeness of the server socket itself, and in case of a connection loss - creates a server socket again.

Unfortunately, the saved tpl template is not available from the file sandbox. Therefore, in order to retrieve it from the _Profiles\\Templates_ folder, the WinAPI must be used once more. This time it will not be described in details, the full listing can be seen below.

```
//+------------------------------------------------------------------+
//|                                                        TplServer |
//|                       programming & development - Alexey Sergeev |
//+------------------------------------------------------------------+
#property copyright "© 2006-2016 Alexey Sergeev"
#property link      "profy.mql@gmail.com"
#property version   "1.00"

#include "SocketLib.mqh"

input string Host="0.0.0.0";
input ushort Port=8080;

uchar tpl[];
int iCnt=0;
string exname="";
SOCKET server=INVALID_SOCKET;
//------------------------------------------------------------------    OnInit
int OnInit()
  {
   EventSetTimer(1);
   exname=MQLInfoString(MQL_PROGRAM_NAME)+".ex5";
   return 0;
  }
//------------------------------------------------------------------    OnDeinit
void OnDeinit(const int reason)
  {
   EventKillTimer();
   CloseClean();
  }
//------------------------------------------------------------------    OnChartEvent
void OnChartEvent(const int id,const long &lparam,const double &dparam,const string &sparam)
  {
   if(iCnt==0) // limit on creating template file - no more than once per second
     {
      Print("Create TPL");
      uchar buf[];
      CreateTpl(buf);
      uchar smb[]; StringToCharArray(Symbol(),smb); ArrayResize(smb,10);
      uchar tf[]; StringToCharArray(IntegerToString(Period()),tf); ArrayResize(tf,10);

      // create data for sending
      ArrayCopy(tpl,smb, ArraySize(tpl)); // add symbol name
      ArrayCopy(tpl, tf, ArraySize(tpl)); // add period value
      ArrayCopy(tpl,buf, ArraySize(tpl)); // add the template itself
     }
   iCnt++;
  }
//------------------------------------------------------------------    OnTimer
void OnTimer()
  {
   iCnt=0; // reset the template creation counter

   if(server==INVALID_SOCKET)
      StartServer(Host,Port);
   else
     {
      // get all clients in a loop and send the current chart template to each client
      SOCKET client=INVALID_SOCKET;
      do
        {
         client=AcceptClient(); // Accept a client socket
         if(client==INVALID_SOCKET) return;

         int slen=ArraySize(tpl);
         int res=send(client,tpl,slen,0);
         if(res==SOCKET_ERROR) Print("-Send failed error: "+WSAErrorDescript(WSAGetLastError()));
         else printf("Sent %d bytes of %d",res,slen);

         if(shutdown(client,SD_BOTH)==SOCKET_ERROR) Print("-Shutdown failed error: "+WSAErrorDescript(WSAGetLastError()));
         closesocket(client);
        }
      while(client!=INVALID_SOCKET);
     }
  }
//------------------------------------------------------------------    StartServer
void StartServer(string addr,ushort port)
  {
// initialize the library
   char wsaData[]; ArrayResize(wsaData,sizeof(WSAData));
   int res=WSAStartup(MAKEWORD(2,2), wsaData);
   if(res!=0) { Print("-WSAStartup failed error: "+string(res)); return; }

// create a socket
   server=socket(AF_INET,SOCK_STREAM,IPPROTO_TCP);
   if(server==INVALID_SOCKET) { Print("-Create failed error: "+WSAErrorDescript(WSAGetLastError())); CloseClean(); return; }

// bind to address and port
   Print("try bind..."+addr+":"+string(port));

   char ch[]; StringToCharArray(addr,ch);
   sockaddr_in addrin;
   addrin.sin_family=AF_INET;
   addrin.sin_addr=inet_addr(ch);
   addrin.sin_port=htons(port);
   ref_sockaddr ref=(ref_sockaddr)addrin;
   if(bind(server,ref.ref,sizeof(addrin))==SOCKET_ERROR)
     {
      int err=WSAGetLastError();
      if(err!=WSAEISCONN) { Print("-Connect failed error: "+WSAErrorDescript(err)+". Cleanup socket"); CloseClean(); return; }
     }

// set to nonblocking mode
   int non_block=1;
   res=ioctlsocket(server,(int)FIONBIO,non_block);
   if(res!=NO_ERROR) { Print("ioctlsocket failed error: "+string(res)); CloseClean(); return; }

// listen port and accept client connections
   if(listen(server,SOMAXCONN)==SOCKET_ERROR) { Print("Listen failed with error: ",WSAErrorDescript(WSAGetLastError())); CloseClean(); return; }

   Print("start server ok");
  }
//------------------------------------------------------------------    Accept
SOCKET AcceptClient() // Accept a client socket
  {
   if(server==INVALID_SOCKET) return INVALID_SOCKET;
   ref_sockaddr ch;
   int len=sizeof(ref_sockaddr);
   SOCKET new_sock=accept(server,ch.ref,len);
//sockaddr_in aclient=(sockaddr_in)ch; convert into structure, if it is necessary to get additional information about the connection
   if(new_sock==INVALID_SOCKET)
     {
      int err=WSAGetLastError();
      if(err==WSAEWOULDBLOCK) Comment("\nWAITING CLIENT ("+string(TimeCurrent())+")");
      else { Print("Accept failed with error: ",WSAErrorDescript(err)); CloseClean(); return INVALID_SOCKET; }
     }
   return new_sock;
  }
//------------------------------------------------------------------    CloseClean
void CloseClean() // close socket
  {
   if(server!=INVALID_SOCKET) { closesocket(server); server=INVALID_SOCKET; }
   WSACleanup();
   Print("stop server");
  }

//------------------------------------------------------------------
#import "kernel32.dll"
int CreateFileW(string lpFileName,uint dwDesiredAccess,uint dwShareMode,uint lpSecurityAttributes,uint dwCreationDisposition,uint dwFlagsAndAttributes,int hTemplateFile);
bool ReadFile(int h,ushort &lpBuffer[],uint nNumberOfBytesToRead,uint &lpNumberOfBytesRead,int lpOverlapped=0);
uint SetFilePointer(int h,int lDistanceToMove,int,uint dwMoveMethod);
bool CloseHandle(int h);
uint GetFileSize(int h,int);
#import

#define FILE_BEGIN                              0
#define OPEN_EXISTING                   3
#define GENERIC_READ                    0x80000000
#define FILE_ATTRIBUTE_NORMAL           0x00000080
#define FILE_SHARE_READ_         0x00000001
//------------------------------------------------------------------    LoadTpl
bool CreateTpl(uchar &abuf[])
  {
   string path=TerminalInfoString(TERMINAL_PATH);
   string name="tcpsend.tpl";

// create template
   ChartSaveTemplate(0,name);

// read the template to the array
   path+="\\Profiles\\Templates\\"+name;
   int h=CreateFileW(path, GENERIC_READ, FILE_SHARE_READ_, 0, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0);
   if(h==INVALID_HANDLE) return false;
   uint sz=GetFileSize(h,NULL);
   ushort rbuf[];
   ArrayResize(rbuf,sz); ArrayInitialize(rbuf,0);
   SetFilePointer(h,0,NULL,FILE_BEGIN); // move to the top
   int r; ReadFile(h,rbuf,sz,r,NULL);
   CloseHandle(h);

// remove the EA name from the template
   string a=ShortArrayToString(rbuf);
   ArrayResize(rbuf,0);
   StringReplace(a,exname," ");
   StringToShortArray(a,rbuf);

// copy the file to a byte array (keeping Unicode)
   sz=ArraySize(rbuf);
   ArrayResize(abuf,sz*2);
   for(uint i=0; i<sz;++i) { abuf[2*i]=(uchar)rbuf[i]; abuf[2*i+1]=(uchar)(rbuf[i]>>8); }

   return true;
  }
```

The client code a little simpler. Since this has been planned as a one-time receipt of a file, there is no need for a constantly running EA with active socket.

The client is implemented as a script. Everything happens within the _OnStart_ event.

```
//+------------------------------------------------------------------+
//|                                                        TplClient |
//|                       programming & development - Alexey Sergeev |
//+------------------------------------------------------------------+
#property copyright "© 2006-2016 Alexey Sergeev"
#property link      "profy.mql@gmail.com"
#property version   "1.00"

#include "..\Experts\SocketLib.mqh"

input string Host="127.0.0.1";
input ushort Port=8080;
SOCKET client=INVALID_SOCKET;
//------------------------------------------------------------------    OnStart
void OnStart()
  {
// initialize the library
   char wsaData[]; ArrayResize(wsaData,sizeof(WSAData));
   int res=WSAStartup(MAKEWORD(2,2), wsaData);
   if(res!=0) { Print("-WSAStartup failed error: "+string(res)); return; }

// create a socket
   client=socket(AF_INET,SOCK_STREAM,IPPROTO_TCP);
   if(client==INVALID_SOCKET) { Print("-Create failed error: "+WSAErrorDescript(WSAGetLastError())); CloseClean(); return; }

// connect to server
   char ch[]; StringToCharArray(Host,ch);
   sockaddr_in addrin;
   addrin.sin_family=AF_INET;
   addrin.sin_addr=inet_addr(ch);
   addrin.sin_port=htons(Port);

   ref_sockaddr ref=(ref_sockaddr)addrin;
   res=connect(client,ref.ref,sizeof(addrin));
   if(res==SOCKET_ERROR)
     {
      int err=WSAGetLastError();
      if(err!=WSAEISCONN) { Print("-Connect failed error: "+WSAErrorDescript(err)); CloseClean(); return; }
     }

// set to nonblocking mode
   int non_block=1;
   res=ioctlsocket(client,(int)FIONBIO,non_block);
   if(res!=NO_ERROR) { Print("ioctlsocket failed error: "+string(res)); CloseClean(); return; }

   Print("connect OK");

// receive data
   uchar rdata[];
   char rbuf[512]; int rlen=512; int rall=0; bool bNext=false;
   while(true)
     {
      res=recv(client,rbuf,rlen,0);
      if(res<0)
        {
         int err=WSAGetLastError();
         if(err!=WSAEWOULDBLOCK) { Print("-Receive failed error: "+string(err)+" "+WSAErrorDescript(err)); CloseClean(); return; }
        }
      else if(res==0 && rall==0) { Print("-Receive. connection closed"); break; }
      else if(res>0) { rall+=res; ArrayCopy(rdata,rbuf,ArraySize(rdata),0,res); }

      if(res>=0 && res<rlen) break;
     }

// close socket
   CloseClean();

   printf("receive %d bytes",ArraySize(rdata));

// take the symbol and period from the file
   string smb=CharArrayToString(rdata,0,10);
   string tf=CharArrayToString(rdata,10,10);

// save the template file
   int h=FileOpen("tcprecv.tpl", FILE_WRITE|FILE_SHARE_WRITE|FILE_BIN); if(h<=0) return;
   FileWriteArray(h,rdata,20);
   FileClose(h);

// apply to the chart
   ChartSetSymbolPeriod(0,smb,(ENUM_TIMEFRAMES)StringToInteger(tf));
   ChartApplyTemplate(0,"\\Files\\tcprecv.tpl");
  }
//------------------------------------------------------------------    CloseClean
void CloseClean() // close socket
  {
   if(client!=INVALID_SOCKET)
     {
      if(shutdown(client,SD_BOTH)==SOCKET_ERROR) Print("-Shutdown failed error: "+WSAErrorDescript(WSAGetLastError()));
      closesocket(client); client=INVALID_SOCKET;
     }
   WSACleanup();
   Print("connect closed");
  }
```

Demonstration of interoperation of these codes:

![](https://c.mql5.com/2/23/tpl.gif)

The attentive reader would note that the client socket can be replaced with a call to the [WebRequest](https://www.mql5.com/en/docs/network/webrequest) MQL function. To do this, add only a couple of [HTTP header](https://en.wikipedia.org/wiki/List_of_HTTP_header_fields "https://en.wikipedia.org/wiki/List_of_HTTP_header_fields") lines to the server and allow the URL in the client terminal settings. You are free to experiment with this.

Important! In some cases, a specific behavior of the terminal has been observed: when calling the _WSACleanup_ function, the MetaTrader closes its own connections.

If you encounter such problem in your experiments, comment _WSAStartup_ and _WSACleanup_ in the code.

### The example 2. Synchronization of trading by symbol

In this example, the server will not close the connection when sending information. The client connection will be kept stable. Thus, any data about changes in trading on the server will be immediately sent via the client sockets. In its turn, the client that accepted a data packet immediately synchronizes its position with the position incoming from the server.

The code of the server and client from the previous example will serve as a basis. With functions for working with the positions added to it.

Let's start with the server:

```
//+------------------------------------------------------------------+
//|                                                     SignalServer |
//|                       programming & development - Alexey Sergeev |
//+------------------------------------------------------------------+
#property copyright "© 2006-2016 Alexey Sergeev"
#property link      "profy.mql@gmail.com"
#property version   "1.00"

#include "SocketLib.mqh"

input string Host="0.0.0.0";
input ushort Port=8081;

bool bChangeTrades;
uchar data[];
SOCKET server=INVALID_SOCKET;
SOCKET conns[];

//------------------------------------------------------------------    OnInit
int OnInit() { OnTrade(); EventSetTimer(1); return 0; }
//------------------------------------------------------------------    OnDeinit
void OnDeinit(const int reason) { EventKillTimer(); CloseClean(); }
//------------------------------------------------------------------    OnTrade
void OnTrade()
  {
   double lot=GetSymbolLot(Symbol());
   StringToCharArray("<<"+Symbol()+"|"+DoubleToString(lot,2)+">>",data); // convert the string to byte array
   bChangeTrades=true;
  }
//------------------------------------------------------------------    OnTimer
void OnTimer()
  {
   if(server==INVALID_SOCKET) StartServer(Host,Port);
   else
     {
      AcceptClients(); // add pending clients
      if(bChangeTrades)
        {
         Print("send new posinfo to clients");
         Send(); bChangeTrades=false;
        }
     }
  }
//------------------------------------------------------------------    StartServer
void StartServer(string addr,ushort port)
  {
// initialize the library
   char wsaData[]; ArrayResize(wsaData,sizeof(WSAData));
   int res=WSAStartup(MAKEWORD(2,2), wsaData);
   if(res!=0) { Print("-WSAStartup failed error: "+string(res)); return; }

// create a socket
   server=socket(AF_INET,SOCK_STREAM,IPPROTO_TCP);
   if(server==INVALID_SOCKET) { Print("-Create failed error: "+WSAErrorDescript(WSAGetLastError())); CloseClean(); return; }

// bind to address and port
   Print("try bind..."+addr+":"+string(port));

   char ch[]; StringToCharArray(addr,ch);
   sockaddr_in addrin;
   addrin.sin_family=AF_INET;
   addrin.sin_addr=inet_addr(ch);
   addrin.sin_port=htons(port);
   ref_sockaddr ref=(ref_sockaddr)addrin;
   if(bind(server,ref.ref,sizeof(addrin))==SOCKET_ERROR)
     {
      int err=WSAGetLastError();
      if(err!=WSAEISCONN) { Print("-Connect failed error: "+WSAErrorDescript(err)+". Cleanup socket"); CloseClean(); return; }
     }

// set to nonblocking mode
   int non_block=1;
   res=ioctlsocket(server,(int)FIONBIO,non_block);
   if(res!=NO_ERROR) { Print("ioctlsocket failed error: "+string(res)); CloseClean(); return; }

// listen port and accept client connections
   if(listen(server,SOMAXCONN)==SOCKET_ERROR) { Print("Listen failed with error: ",WSAErrorDescript(WSAGetLastError())); CloseClean(); return; }

   Print("start server ok");
  }
//------------------------------------------------------------------    Accept
void AcceptClients() // Accept a client socket
  {
   if(server==INVALID_SOCKET) return;

// add all pending clients
   SOCKET client=INVALID_SOCKET;
   do
     {
      ref_sockaddr ch; int len=sizeof(ref_sockaddr);
      client=accept(server,ch.ref,len);
      if(client==INVALID_SOCKET)
        {
         int err=WSAGetLastError();
         if(err==WSAEWOULDBLOCK) Comment("\nWAITING CLIENT ("+string(TimeCurrent())+")");
         else { Print("Accept failed with error: ",WSAErrorDescript(err)); CloseClean(); }
         return;
        }

      // set to nonblocking mode
      int non_block=1;
      int res=ioctlsocket(client, (int)FIONBIO, non_block);
      if(res!=NO_ERROR) { Print("ioctlsocket failed error: "+string(res)); continue; }

      // add client socket to the array
      int n=ArraySize(conns); ArrayResize(conns,n+1);
      conns[n]=client;
      bChangeTrades=true; // flag to indicate that information about the position must be sent

                          // show client information
      char ipstr[23]={0};
      sockaddr_in aclient=(sockaddr_in)ch; // convert into structure to get additional information about the connection
      inet_ntop(aclient.sin_family,aclient.sin_addr,ipstr,sizeof(ipstr)); // get the address
      printf("Accept new client %s : %d",CharArrayToString(ipstr),ntohs(aclient.sin_port));
     }
   while(client!=INVALID_SOCKET);
  }
//------------------------------------------------------------------    SendClient
void Send()
  {
   int len=ArraySize(data);
   for(int i=ArraySize(conns)-1; i>=0; --i) // send out the information to clients
     {
      if(conns[i]==INVALID_SOCKET) continue; // skip closed
      int res=send(conns[i],data,len,0); // send
      if(res==SOCKET_ERROR) { Print("-Send failed error: "+WSAErrorDescript(WSAGetLastError())+". close socket"); Close(conns[i]); }
     }
  }
//------------------------------------------------------------------    CloseClean
void CloseClean() // close and clear operation
  {
   printf("Shutdown server and %d connections",ArraySize(conns));
   if(server!=INVALID_SOCKET) { closesocket(server); server=INVALID_SOCKET; } // close the server
   for(int i=ArraySize(conns)-1; i>=0; --i) Close(conns[i]); // close the clients
   ArrayResize(conns,0);
   WSACleanup();
  }
//------------------------------------------------------------------    Close
void Close(SOCKET &asock) // close one socket
  {
   if(asock==INVALID_SOCKET) return;
   if(shutdown(asock,SD_BOTH)==SOCKET_ERROR) Print("-Shutdown failed error: "+WSAErrorDescript(WSAGetLastError()));
   closesocket(asock);
   asock=INVALID_SOCKET;
  }
//------------------------------------------------------------------    GetSymbolLot
double GetSymbolLot(string smb)
  {
   double slot=0;
   int n=PositionsTotal();
   for(int i=0; i<n;++i)
     {
      PositionSelectByTicket(PositionGetTicket(i));
      if(PositionGetString(POSITION_SYMBOL)!=smb) continue; // filter the position of the current symbol, where the server is running
      double lot=PositionGetDouble(POSITION_VOLUME); // get the volume
      if(PositionGetInteger(POSITION_TYPE)==POSITION_TYPE_SELL) lot=-lot; // consider the direction
      slot+=lot; // add to the sum
     }
   return slot;
  }
```

Every second it checks the key blocks of the server: connecting client and adding it to the total array -> sending the data. It also checks for activeness of the server socket itself, and in case of a connection loss - creates a server socket.

The name of the symbol the EA is running on and the volume of its position are sent to the clients.

Each trade operation will send the symbol and volume as messages:

<<GBPUSD\|0.25>>

<<GBPUSD\|0.00>>

Messages are sent at each trade event, and also when a new client connects.

This time the code of the client is implemented as an expert, as it is necessary to keep the connection active. The client accepts new portion of data from the server and adds it to the existing data. Then it looks for signs of beginning **<<** and end **>>** of the message, parses it and adjusts its volume according to the one on the server for the specified symbol.

```
//+------------------------------------------------------------------+
//|                                                     SignalClient |
//|                       programming & development - Alexey Sergeev |
//+------------------------------------------------------------------+
#property copyright "© 2006-2016 Alexey Sergeev"
#property link      "profy.mql@gmail.com"
#property version   "1.00"

#include "SocketLib.mqh"
#include <Trade\Trade.mqh>

input string Host="127.0.0.1";
input ushort Port=8081;

SOCKET client=INVALID_SOCKET; // client socket
string msg=""; // queue of received messages
//------------------------------------------------------------------    OnInit
int OnInit()
  {
   if(AccountInfoInteger(ACCOUNT_MARGIN_MODE)==ACCOUNT_MARGIN_MODE_RETAIL_HEDGING)
     {
      Alert("Client work only with Netting accounts"); return INIT_FAILED;
     }

   EventSetTimer(1); return INIT_SUCCEEDED;
  }
//------------------------------------------------------------------    OnInit
void OnDeinit(const int reason) { EventKillTimer(); CloseClean(); }
//------------------------------------------------------------------    OnInit
void OnTimer()
  {
   if(client==INVALID_SOCKET) StartClient(Host,Port);
   else
     {
      uchar data[];
      if(Receive(data)>0) // receive data
        {
         msg+=CharArrayToString(data); // if something was received, add it to the total string
         printf("received msg from server: %s",msg);
        }
      CheckMessage();
     }
  }
//------------------------------------------------------------------    CloseClean
void StartClient(string addr,ushort port)
  {
// initialize the library
   int res=0;
   char wsaData[]; ArrayResize(wsaData, sizeof(WSAData));
   res=WSAStartup(MAKEWORD(2,2), wsaData);
   if (res!=0) { Print("-WSAStartup failed error: "+string(res)); return; }

// create a socket
   client=socket(AF_INET,SOCK_STREAM,IPPROTO_TCP);
   if(client==INVALID_SOCKET) { Print("-Create failed error: "+WSAErrorDescript(WSAGetLastError())); CloseClean(); return; }

// connect to server
   char ch[]; StringToCharArray(addr,ch);
   sockaddr_in addrin;
   addrin.sin_family=AF_INET;
   addrin.sin_addr=inet_addr(ch);
   addrin.sin_port=htons(port);

   ref_sockaddr ref=(ref_sockaddr)addrin;
   res=connect(client,ref.ref,sizeof(addrin));
   if(res==SOCKET_ERROR)
     {
      int err=WSAGetLastError();
      if(err!=WSAEISCONN) { Print("-Connect failed error: "+WSAErrorDescript(err)); CloseClean(); return; }
     }

// set to nonblocking mode
   int non_block=1;
   res=ioctlsocket(client,(int)FIONBIO,non_block);
   if(res!=NO_ERROR) { Print("ioctlsocket failed error: "+string(res)); CloseClean(); return; }

   Print("connect OK");
  }
//------------------------------------------------------------------    Receive
int Receive(uchar &rdata[]) // Receive until the peer closes the connection
  {
   if(client==INVALID_SOCKET) return 0; // if the socket is not open

   char rbuf[512]; int rlen=512; int r=0,res=0;
   do
     {
      res=recv(client,rbuf,rlen,0);
      if(res<0)
        {
         int err=WSAGetLastError();
         if(err!=WSAEWOULDBLOCK) { Print("-Receive failed error: "+string(err)+" "+WSAErrorDescript(err)); CloseClean(); return -1; }
         break;
        }
      if(res==0 && r==0) { Print("-Receive. connection closed"); CloseClean(); return -1; }
      r+=res; ArrayCopy(rdata,rbuf,ArraySize(rdata),0,res);
     }
   while(res>0 && res>=rlen);
   return r;
  }
//------------------------------------------------------------------    CloseClean
void CloseClean() // close socket
  {
   if(client!=INVALID_SOCKET)
     {
      if(shutdown(client,SD_BOTH)==SOCKET_ERROR) Print("-Shutdown failed error: "+WSAErrorDescript(WSAGetLastError()));
      closesocket(client); client=INVALID_SOCKET;
     }

   WSACleanup();
   Print("close socket");
  }
//------------------------------------------------------------------    CheckMessage
void CheckMessage()
  {
   string pos;
   while(FindNextPos(pos)) { printf("server position: %s",pos); };  // get the most recent change from the server
   if(StringLen(pos)<=0) return;
// receive data from the message
   string res[]; if(StringSplit(pos,'|',res)!=2) { printf("-wrong pos info: %s",pos); return; }
   string smb=res[0]; double lot=NormalizeDouble(StringToDouble(res[1]),2);

// synchronize volume
   if(!SyncSymbolLot(smb,lot)) msg="<<"+pos+">>"+msg; // if there is an error, return the message to the beginning of the "thread"
  }
//------------------------------------------------------------------    SyncSymbolLot
bool SyncSymbolLot(string smb,double nlot)
  {
// synchronize the server and client volumes
   CTrade trade;
   double clot=GetSymbolLot(smb); // get the current lot for the symbol
   if(clot==nlot) { Print("nothing change"); return true; } // if the volumes are equal, do nothing

                                                            // first, check the special case of no positions present on the server
   if(nlot==0 && clot!=0) { Print("full close position"); return trade.PositionClose(smb); }

// if the server has a position, change it on the client
   double dif=NormalizeDouble(nlot-clot,2);
// buy or sell depending on the difference
   if(dif>0) { Print("add Buy position"); return trade.Buy(dif,smb); }
   else { Print("add Sell position"); return trade.Sell(-dif,smb); }
  }
//------------------------------------------------------------------    FindNextPos
bool FindNextPos(string &pos)
  {
   int b=StringFind(msg, "<<"); if(b<0) return false; // no beginning of the message
   int e=StringFind(msg, ">>"); if(e<0) return false; // no end of the message

   pos=StringSubstr(msg,b+2,e-b-2); // get the information block
   msg=StringSubstr(msg,e+2); // remove it from the message
   return true;
  }
//------------------------------------------------------------------    GetSymbolLot
double GetSymbolLot(string smb)
  {
   double slot=0;
   int n=PositionsTotal();
   for(int i=0; i<n;++i)
     {
      PositionSelectByTicket(PositionGetTicket(i));
      if(PositionGetString(POSITION_SYMBOL)!=smb) continue; // filter the position of the current symbol, where the server is running
      double lot=PositionGetDouble(POSITION_VOLUME); // get the volume
      if(PositionGetInteger(POSITION_TYPE)==POSITION_TYPE_SELL) lot=-lot; // consider the direction
      slot+=lot; // add to the sum
     }
   return NormalizeDouble(slot,2);
  }
```

The final demonstration of paired operation of the server and client:

![](https://c.mql5.com/2/23/trade.gif)

### Example 3. Tick collector.

This example demonstrates UPD sockets. In it, the server will wait for data on symbol from the client.

The server code is simple, as there is no need to store information on clients and listen for their connections. Checks of socket data can be accelerated using a millisecond timer:

```
input string Host="0.0.0.0";
input ushort Port=8082;

SOCKET server=INVALID_SOCKET;

//------------------------------------------------------------------    OnInit
int OnInit() { EventSetMillisecondTimer(300); return 0; }
//------------------------------------------------------------------    OnDeinit
void OnDeinit(const int reason) { EventKillTimer(); CloseClean(); }
//------------------------------------------------------------------    OnTimer
void OnTimer()
  {
   if(server!=INVALID_SOCKET)
     {
      char buf[1024]={0};
      ref_sockaddr ref={0}; int len=ArraySize(ref.ref);
      int res=recvfrom(server,buf,1024,0,ref.ref,len);
      if (res>=0) // receive and display data
         Print("receive tick from client: ", CharArrayToString(buf));
        else
        {
         int err=WSAGetLastError();
         if(err!=WSAEWOULDBLOCK) { Print("-receive failed error: "+WSAErrorDescript(err)+". Cleanup socket"); CloseClean(); return; }
        }

     }
   else // otherwise start the server
     {
      // initialize the library
      char wsaData[]; ArrayResize(wsaData,sizeof(WSAData));
      int res=WSAStartup(MAKEWORD(2,2), wsaData);
      if(res!=0) { Print("-WSAStartup failed error: "+string(res)); return; }

      // create a socket
      server=socket(AF_INET,SOCK_DGRAM,IPPROTO_UDP);
      if(server==INVALID_SOCKET) { Print("-Create failed error: "+WSAErrorDescript(WSAGetLastError())); CloseClean(); return; }

      // bind to address and port
      Print("try bind..."+Host+":"+string(Port));

      char ch[]; StringToCharArray(Host,ch);
      sockaddr_in addrin;
      addrin.sin_family=AF_INET;
      addrin.sin_addr=inet_addr(ch);
      addrin.sin_port=htons(Port);
      ref_sockaddr ref=(ref_sockaddr)addrin;
      if(bind(server,ref.ref,sizeof(addrin))==SOCKET_ERROR)
        {
         int err=WSAGetLastError();
         if(err!=WSAEISCONN) { Print("-Connect failed error: "+WSAErrorDescript(err)+". Cleanup socket"); CloseClean(); return; }
        }

      // set to nonblocking mode
      int non_block=1;
      res=ioctlsocket(server,(int)FIONBIO,non_block);
      if(res!=NO_ERROR) { Print("ioctlsocket failed error: "+string(res)); CloseClean(); return; }

      Print("start server ok");
     }
  }
//------------------------------------------------------------------    CloseClean
void CloseClean() // close and clear operation
  {
   printf("Shutdown server");
   if(server!=INVALID_SOCKET) { closesocket(server); server=INVALID_SOCKET; } // close the server
   WSACleanup();
  }
```

The client code is also simple. All processing takes place within the tick arrival event:

```
input string Host="127.0.0.1";
input ushort Port=8082;

SOCKET client=INVALID_SOCKET; // client socket
ref_sockaddr srvaddr={0}; // structure for connecting to the server
//------------------------------------------------------------------    OnInit
int OnInit()
  {
// fill the structure for the server
   char ch[]; StringToCharArray(Host,ch);
   sockaddr_in addrin;
   addrin.sin_family=AF_INET;
   addrin.sin_addr=inet_addr(ch);
   addrin.sin_port=htons(Port);
   srvaddr=(ref_sockaddr)addrin;

   OnTick(); // create socket immediately

   return INIT_SUCCEEDED;
  }
//------------------------------------------------------------------    OnDeinit
void OnDeinit(const int reason) { CloseClean(); }
//------------------------------------------------------------------    OnTick
void OnTick()
  {
   if(client!=INVALID_SOCKET) // if the socket is created, send
     {
      uchar data[]; StringToCharArray(Symbol()+" "+DoubleToString(SymbolInfoDouble(Symbol(),SYMBOL_BID),Digits()),data);
      if(sendto(client,data,ArraySize(data),0,srvaddr.ref,ArraySize(srvaddr.ref))==SOCKET_ERROR)
        {
         int err=WSAGetLastError();
         if(err!=WSAEWOULDBLOCK) { Print("-Send failed error: "+WSAErrorDescript(err)); CloseClean(); }
        }
      else
         Print("send "+Symbol()+" tick to server");
     }
   else // create a client socket
     {
      int res=0;
      char wsaData[]; ArrayResize(wsaData,sizeof(WSAData));
      res=WSAStartup(MAKEWORD(2,2),wsaData);
      if(res!=0) { Print("-WSAStartup failed error: "+string(res)); return; }

      // create a socket
      client=socket(AF_INET,SOCK_DGRAM,IPPROTO_UDP);
      if(client==INVALID_SOCKET) { Print("-Create failed error: "+WSAErrorDescript(WSAGetLastError())); CloseClean(); return; }

      // set to nonblocking mode
      int non_block=1;
      res=ioctlsocket(client,(int)FIONBIO,non_block);
      if(res!=NO_ERROR) { Print("ioctlsocket failed error: "+string(res)); CloseClean(); return; }

      Print("create socket OK");
     }
  }
//------------------------------------------------------------------    CloseClean
void CloseClean() // close socket
  {
   if(client!=INVALID_SOCKET) { closesocket(client); client=INVALID_SOCKET; }
   WSACleanup();
   Print("close socket");
  }
```

And here is the demonstration of its final work:

![](https://c.mql5.com/2/23/tick.gif)

### 3\. Further methods to enhance the server

Obviously, these examples of servers sending out information to any client are not optimal. For instance, you may want to restrict access to your information. So, the mandatory requirements have to include at least:

- client authorization (login/password);
- protection against password guessing (ban/login or IP blocking).

Also, all the work of the server is performed only within one thread (in timer of one expert). This is critical for a large number of connections or large amounts of information. Therefore, in order to optimize the server, it is reasonable to add at least a pool of experts (each with its own timer) that will handle the interaction with client connections. This will make the server multi-threaded to some extent.

Whether or not to do this in MQL is up to you. There are other means to do that, perhaps they could be more convenient. However, the fact that MQL gives the advantage of direct access to account trading and quotes cannot be denied, as well as the openness of MQL code that does not use third-party DLL.

### **Conclusion**

How else can sockets be used in MetaTrader? Before the article had been written, there were several ideas to be considered as examples:

- market sentiment indicator (when the connected clients send the volumes of their positions and get a response as the total volume received from all clients);
- or, for example, sending out the indicator calculations from the server to clients (for subscribers);
- or vice versa - clients will help with the heavy calculations (tester agent network);
- it is possible to make the server just a "proxy" for exchanging data between clients.

There can be many options. If you have ideas on the application — share them in the article comments. Perhaps, if they are interesting, we will be able to implement them together.

**I wish you good luck and big profits!**

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/2599](https://www.mql5.com/ru/articles/2599)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/2599.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/2599/mql5.zip "Download MQL5.zip")(24.71 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [SQL and MQL5: Working with SQLite Database](https://www.mql5.com/en/articles/862)
- [Getting Rid of Self-Made DLLs](https://www.mql5.com/en/articles/364)
- [Promote Your Development Projects Using EX5 Libraries](https://www.mql5.com/en/articles/362)
- [Using WinInet in MQL5. Part 2: POST Requests and Files](https://www.mql5.com/en/articles/276)
- [Tracing, Debugging and Structural Analysis of Source Code](https://www.mql5.com/en/articles/272)
- [The Prototype of a Trading Robot](https://www.mql5.com/en/articles/132)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/91815)**
(94)


![Juvenille Emperor Limited](https://c.mql5.com/avatar/2019/4/5CB0FE21-E283.jpg)

**[Eleni Anna Branou](https://www.mql5.com/en/users/eleanna74)**
\|
2 Apr 2022 at 09:51

**Mars Eze [#](https://www.mql5.com/en/forum/91815/page2#comment_28694102):**

I want to learn more about Forex

[https://www.mql5.com/en/forum/381853#comment\_25845157](https://www.mql5.com/en/forum/381853#comment_25845157)

![YestheI](https://c.mql5.com/avatar/2022/11/6368bd57-cc11.png)

**[YestheI](https://www.mql5.com/en/users/yesthei)**
\|
7 Nov 2022 at 07:37

Hi, where can I learn more about **ioctlsocket**() mode?

![279DE7B1](https://c.mql5.com/avatar/avatar_na2.png)

**[279DE7B1](https://www.mql5.com/en/users/279de7b1)**
\|
31 Dec 2022 at 13:41

I use the the signalsever and signalclient, but there always dispaly connection timeout for remote communication , what are problems ?  please help me, thank you.


![Gad Benisty](https://c.mql5.com/avatar/2021/12/61C79323-DF36.jpeg)

**[Gad Benisty](https://www.mql5.com/en/users/gadben)**
\|
7 Mar 2023 at 16:12

It looks great.

I would like to use it to implement client EA, receiving message from WSS socket (instead of Host name + port).

For example from : wss://demo.piesocket.com/v3/channel\_123?api\_key=VCXCEuvhGcBDP7XhiJJUDvR1e1D3eiVjgZ9VRiaV&notify\_self

How to connect to this socket server?

Thanks in advance for your help.

![Quantum Capital International Group Ltd](https://c.mql5.com/avatar/2020/4/5E96871E-4BA3.png)

**[Yang Chih Chou](https://www.mql5.com/en/users/fxchess)**
\|
8 Mar 2024 at 06:19

I find that when I use socket in MT5 it must be [close socket](https://www.mql5.com/en/docs/network/socketclose "MQL5 documentation: SocketClose function") every tick.

If I don't do it, the socket send data will failed (block or freeze).

So, I need to send data to the server end and close socket every tick.

Next send I need to connect socket again.

Can it possible to keep socket connection during I send or receive data without close every tick?

![Universal Expert Advisor: Integration with Standard MetaTrader Modules of Signals (Part 7)](https://c.mql5.com/2/23/zapvwy5wjkj_54w2.png)[Universal Expert Advisor: Integration with Standard MetaTrader Modules of Signals (Part 7)](https://www.mql5.com/en/articles/2540)

This part of the article describes the possibilities of the CStrategy engine integration with the signal modules included into the standard library in MetaTrader. The article describes how to work with signals, as well as how to create custom strategies on their basis.

![Using text files for storing input parameters of Expert Advisors, indicators and scripts](https://c.mql5.com/2/23/avatar__3.png)[Using text files for storing input parameters of Expert Advisors, indicators and scripts](https://www.mql5.com/en/articles/2564)

The article describes the application of text files for storing dynamic objects, arrays and other variables used as properties of Expert Advisors, indicators and scripts. The files serve as a convenient addition to the functionality of standard tools offered by MQL languages.

![LifeHack for Traders: Indicators of Balance, Drawdown, Load and Ticks during Testing](https://c.mql5.com/2/23/avac18.png)[LifeHack for Traders: Indicators of Balance, Drawdown, Load and Ticks during Testing](https://www.mql5.com/en/articles/2501)

How to make the testing process more visual? The answer is simple: you need to use one or more indicators in the Strategy Tester, including a tick indicator, an indicator of balance and equity, an indicator of drawdown and deposit load. This solution will help you visually track the nature of ticks, balance and equity changes, as well as drawdown and deposit load.

![Regular expressions for traders](https://c.mql5.com/2/23/ava.png)[Regular expressions for traders](https://www.mql5.com/en/articles/2432)

A regular expression is a special language for handling texts by applying a specified rule, also called a regex or regexp for short. In this article, we are going to show how to handle a trade report with the RegularExpressions library for MQL5, and will also demonstrate the optimization results after using it.

[![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/01.png)![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/02.png)Powerful analytics for traders of any levelAll the necessary trading reports for beginners and professionals](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/454106&a=muccpajyfystoakuukdobwigjejzmpqn&s=52daad60fa795e635264e6f94898f05493bca3b5124d4cca8eb7e82333c2ef12&uid=&ref=https://www.mql5.com/en/articles/2599&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5072045623156355917)

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