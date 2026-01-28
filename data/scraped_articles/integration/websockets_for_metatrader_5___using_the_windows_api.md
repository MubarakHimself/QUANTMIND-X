---
title: WebSockets for MetaTrader 5 — Using the Windows API
url: https://www.mql5.com/en/articles/10275
categories: Integration
relevance_score: 6
scraped_at: 2026-01-23T17:24:47.729657
---

[Running robots on virtual hosting is easyFollow our step-by-step MetaTrader VPS guide for beginnersRead![](https://www.mql5.com/ff/sh/au4fqg4kms7s9mq1z2/01.png)![](https://www.mql5.com/ff/sh/au4fqg4kms7s9mq1z2/02.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/articles/13586&a=uzpprdshbcrtxvjxpmescehprypbymxc&s=516438f25b531570d9b7d49dcfb29c82fa1021f5ede6571df8026dbfbafcd13f&uid=&ref=https://www.mql5.com/en/articles/10275&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5068193969499731628)

MetaTrader 5 / Examples


### Introduction

In the article [WebSockets for MetaTrader 5](https://www.mql5.com/en/articles/8196 "Websockets for Metatrader 5"), we discussed the basics of the WebSocket protocol and created a client that relied on MQL5 implemented sockets. This time around we will leverage the Windows API to build a WebSocket client for MetaTrader 5 programs. It is the next best option as no extra software is required, everything is provided by the operating system. We will implement the client as a class and conduct tests by consuming the Deriv.com WebSocket API to feed live tick data into MetaTrader 5.

### WebSockets in Windows

When it comes to Windows API and the Internet, MQL5 developers are most familiar with the Windows Internet (WinINeT) library. It implements Internet protocols such as File transfer protocol (FTP) and HTTP among others. Similar to it is the Windows HTTP Services (WinHTTP) library. It is a dedicated library for the HTTP protocol with features that are useful for server side development. Some of the features exposed by WinHTTP are utilities for handling WebSocket connections.

The WebSocket protocol was introduced into Windows operating systems from Windows 8.1 and  Windows Server 2012 R2  onwards. Windows 7 and older operating systems do not have native support for it. The programs described in this article will not work on machines running these older operating systems.

### The Winhttp library

To create a WebSocket client connection using winhttp, we will need the functions listed below:

| [WinHttpOpen](https://www.mql5.com/go?link=https://learn.microsoft.com/en-us/windows/win32/api/winhttp/nf-winhttp-winhttpopen "https://docs.microsoft.com/en-us/windows/win32/api/winhttp/nf-winhttp-winhttpopen") | Initializes the library preparing it for use by an application |
| [WinHttpConnect](https://www.mql5.com/go?link=https://learn.microsoft.com/en-us/windows/win32/api/winhttp/nf-winhttp-winhttpconnect "https://docs.microsoft.com/en-us/windows/win32/api/winhttp/nf-winhttp-winhttpconnect") | Sets the domain name of the server the application wants to communicate with |
| [WinHttpOpenRequest](https://www.mql5.com/go?link=https://learn.microsoft.com/en-us/windows/win32/api/winhttp/nf-winhttp-winhttpopenrequest "https://docs.microsoft.com/en-us/windows/win32/api/winhttp/nf-winhttp-winhttpopenrequest") | Creates an HTTP request handle |
| [WinHttpSetOption](https://www.mql5.com/go?link=https://learn.microsoft.com/en-us/windows/win32/api/winhttp/nf-winhttp-winhttpsetoption "https://docs.microsoft.com/en-us/windows/win32/api/winhttp/nf-winhttp-winhttpsetoption") | Sets various configuration options for an HTTP connection |
| [WinHttpSendRequest](https://www.mql5.com/go?link=https://learn.microsoft.com/en-us/windows/win32/api/winhttp/nf-winhttp-winhttpsendrequest "https://docs.microsoft.com/en-us/windows/win32/api/winhttp/nf-winhttp-winhttpsendrequest") | Sends a request to a server |
| [WinHttpReceiveResponse](https://www.mql5.com/go?link=https://learn.microsoft.com/en-us/windows/win32/api/winhttp/nf-winhttp-winhttpreceiveresponse "https://docs.microsoft.com/en-us/windows/win32/api/winhttp/nf-winhttp-winhttpreceiveresponse") | Receives the response from a server after sending a request |
| [WinHttpWebSocketCompleteUpgrade](https://www.mql5.com/go?link=https://learn.microsoft.com/en-us/windows/win32/api/winhttp/nf-winhttp-winhttpwebsocketcompleteupgrade "https://docs.microsoft.com/en-us/windows/win32/api/winhttp/nf-winhttp-winhttpwebsocketcompleteupgrade") | Confirms the response received from the server satisfies the WebSocket protocol |
| [WinHttpCloseHandle](https://www.mql5.com/go?link=https://learn.microsoft.com/en-us/windows/win32/api/winhttp/nf-winhttp-winhttpclosehandle "https://docs.microsoft.com/en-us/windows/win32/api/winhttp/nf-winhttp-winhttpclosehandle") | Used to discard any resource descriptors previously in use |
| [WinHttpWebSocketSend](https://www.mql5.com/go?link=https://learn.microsoft.com/en-us/windows/win32/api/winhttp/nf-winhttp-winhttpwebsocketsend "https://docs.microsoft.com/en-us/windows/win32/api/winhttp/nf-winhttp-winhttpwebsocketsend") | Used to send data via a WebSocket connection |
| [WinHttpWebSocketReceive](https://www.mql5.com/go?link=https://learn.microsoft.com/en-us/windows/win32/api/winhttp/nf-winhttp-winhttpwebsocketreceive "https://docs.microsoft.com/en-us/windows/win32/api/winhttp/nf-winhttp-winhttpwebsocketreceive") | Receives data using a WebSocket connection |
| [WinHttpWebSocketClose](https://www.mql5.com/go?link=https://docs.microsoft.com/en-us/windows/desktop/api/winhttp/nf-winhttp-winhttpwebsocketclose "https://docs.microsoft.com/en-us/windows/desktop/api/winhttp/nf-winhttp-winhttpwebsocketclose") | Closes a WebSocket connection |
| [WinHttpWebSocketQueryCloseStatus](https://www.mql5.com/go?link=https://docs.microsoft.com/en-us/windows/desktop/api/winhttp/nf-winhttp-winhttpwebsocketqueryclosestatus "https://docs.microsoft.com/en-us/windows/desktop/api/winhttp/nf-winhttp-winhttpwebsocketqueryclosestatus") | Checks the close status message sent from the server |

All the functions available in the library are documented by [Microsoft](https://www.mql5.com/go?link=https://docs.microsoft.com/en-us/windows/win32/winhttp/winhttp-functions "https://docs.microsoft.com/en-us/windows/win32/winhttp/winhttp-functions"). Also, a detailed description of all the functions, their input parameters and return types, can be viewed by following the corresponding links above.

The client we will create for MetaTrader 5 works in synchronous mode. That means function calls will block execution until they return. For example, a call to WinHttpWebSocketReceive() will block the executing thread until data is available for reading. Keep this in mind when creating MetaTrader 5 applications.

The winhttp functions are declared and imported in the winhttp.mqh include file.

```
#include <WinAPI\errhandlingapi.mqh>

#define WORD  ushort
#define DWORD ulong
#define BYTE  uchar
#define INTERNET_PORT WORD
#define HINTERNET long
#define LPVOID uint&

#define WINHTTP_ERROR_BASE                     12000

#define ERROR_WINHTTP_OUT_OF_HANDLES           (WINHTTP_ERROR_BASE + 1)
#define ERROR_WINHTTP_TIMEOUT                  (WINHTTP_ERROR_BASE + 2)
#define ERROR_WINHTTP_INTERNAL_ERROR           (WINHTTP_ERROR_BASE + 4)
#define ERROR_WINHTTP_INVALID_URL              (WINHTTP_ERROR_BASE + 5)
#define ERROR_WINHTTP_UNRECOGNIZED_SCHEME      (WINHTTP_ERROR_BASE + 6)
#define ERROR_WINHTTP_NAME_NOT_RESOLVED        (WINHTTP_ERROR_BASE + 7)
#define ERROR_WINHTTP_INVALID_OPTION           (WINHTTP_ERROR_BASE + 9)
#define ERROR_WINHTTP_OPTION_NOT_SETTABLE      (WINHTTP_ERROR_BASE + 11)
#define ERROR_WINHTTP_SHUTDOWN                 (WINHTTP_ERROR_BASE + 12)

#define ERROR_WINHTTP_LOGIN_FAILURE            (WINHTTP_ERROR_BASE + 15)
#define ERROR_WINHTTP_OPERATION_CANCELLED      (WINHTTP_ERROR_BASE + 17)
#define ERROR_WINHTTP_INCORRECT_HANDLE_TYPE    (WINHTTP_ERROR_BASE + 18)
#define ERROR_WINHTTP_INCORRECT_HANDLE_STATE   (WINHTTP_ERROR_BASE + 19)
#define ERROR_WINHTTP_CANNOT_CONNECT           (WINHTTP_ERROR_BASE + 29)
#define ERROR_WINHTTP_CONNECTION_ERROR         (WINHTTP_ERROR_BASE + 30)
#define ERROR_WINHTTP_RESEND_REQUEST           (WINHTTP_ERROR_BASE + 32)

#define ERROR_WINHTTP_CLIENT_AUTH_CERT_NEEDED  (WINHTTP_ERROR_BASE + 44)

#define ERROR_WINHTTP_CANNOT_CALL_BEFORE_OPEN   (WINHTTP_ERROR_BASE + 100)
#define ERROR_WINHTTP_CANNOT_CALL_BEFORE_SEND   (WINHTTP_ERROR_BASE + 101)
#define ERROR_WINHTTP_CANNOT_CALL_AFTER_SEND (WINHTTP_ERROR_BASE + 102)
#define ERROR_WINHTTP_CANNOT_CALL_AFTER_OPEN (WINHTTP_ERROR_BASE + 103)

#define ERROR_WINHTTP_HEADER_NOT_FOUND             (WINHTTP_ERROR_BASE + 150)
#define ERROR_WINHTTP_INVALID_SERVER_RESPONSE      (WINHTTP_ERROR_BASE + 152)
#define ERROR_WINHTTP_INVALID_HEADER               (WINHTTP_ERROR_BASE + 153)
#define ERROR_WINHTTP_INVALID_QUERY_REQUEST        (WINHTTP_ERROR_BASE + 154)
#define ERROR_WINHTTP_HEADER_ALREADY_EXISTS        (WINHTTP_ERROR_BASE + 155)
#define ERROR_WINHTTP_REDIRECT_FAILED              (WINHTTP_ERROR_BASE + 156)

#define ERROR_WINHTTP_AUTO_PROXY_SERVICE_ERROR  (WINHTTP_ERROR_BASE + 178)
#define ERROR_WINHTTP_BAD_AUTO_PROXY_SCRIPT     (WINHTTP_ERROR_BASE + 166)
#define ERROR_WINHTTP_UNABLE_TO_DOWNLOAD_SCRIPT (WINHTTP_ERROR_BASE + 167)
#define ERROR_WINHTTP_UNHANDLED_SCRIPT_TYPE     (WINHTTP_ERROR_BASE + 176)
#define ERROR_WINHTTP_SCRIPT_EXECUTION_ERROR    (WINHTTP_ERROR_BASE + 177)
#define ERROR_WINHTTP_NOT_INITIALIZED          (WINHTTP_ERROR_BASE + 172)
#define ERROR_WINHTTP_SECURE_FAILURE           (WINHTTP_ERROR_BASE + 175)

#define ERROR_WINHTTP_SECURE_CERT_DATE_INVALID    (WINHTTP_ERROR_BASE + 37)
#define ERROR_WINHTTP_SECURE_CERT_CN_INVALID      (WINHTTP_ERROR_BASE + 38)
#define ERROR_WINHTTP_SECURE_INVALID_CA           (WINHTTP_ERROR_BASE + 45)
#define ERROR_WINHTTP_SECURE_CERT_REV_FAILED      (WINHTTP_ERROR_BASE + 57)
#define ERROR_WINHTTP_SECURE_CHANNEL_ERROR        (WINHTTP_ERROR_BASE + 157)
#define ERROR_WINHTTP_SECURE_INVALID_CERT         (WINHTTP_ERROR_BASE + 169)
#define ERROR_WINHTTP_SECURE_CERT_REVOKED         (WINHTTP_ERROR_BASE + 170)
#define ERROR_WINHTTP_SECURE_CERT_WRONG_USAGE     (WINHTTP_ERROR_BASE + 179)

#define ERROR_WINHTTP_AUTODETECTION_FAILED                  (WINHTTP_ERROR_BASE + 180)
#define ERROR_WINHTTP_HEADER_COUNT_EXCEEDED                 (WINHTTP_ERROR_BASE + 181)
#define ERROR_WINHTTP_HEADER_SIZE_OVERFLOW                  (WINHTTP_ERROR_BASE + 182)
#define ERROR_WINHTTP_CHUNKED_ENCODING_HEADER_SIZE_OVERFLOW (WINHTTP_ERROR_BASE + 183)
#define ERROR_WINHTTP_RESPONSE_DRAIN_OVERFLOW               (WINHTTP_ERROR_BASE + 184)
#define ERROR_WINHTTP_CLIENT_CERT_NO_PRIVATE_KEY            (WINHTTP_ERROR_BASE + 185)
#define ERROR_WINHTTP_CLIENT_CERT_NO_ACCESS_PRIVATE_KEY     (WINHTTP_ERROR_BASE + 186)

#define ERROR_WINHTTP_CLIENT_AUTH_CERT_NEEDED_PROXY         (WINHTTP_ERROR_BASE + 187)
#define ERROR_WINHTTP_SECURE_FAILURE_PROXY                  (WINHTTP_ERROR_BASE + 188)
#define ERROR_WINHTTP_RESERVED_189                          (WINHTTP_ERROR_BASE + 189)
#define ERROR_WINHTTP_HTTP_PROTOCOL_MISMATCH                (WINHTTP_ERROR_BASE + 190)

#define WINHTTP_ERROR_LAST                                  (WINHTTP_ERROR_BASE + 188)

enum WINHTTP_WEB_SOCKET_BUFFER_TYPE
  {
   WINHTTP_WEB_SOCKET_BINARY_MESSAGE_BUFFER_TYPE       = 0,
   WINHTTP_WEB_SOCKET_BINARY_FRAGMENT_BUFFER_TYPE      = 1,
   WINHTTP_WEB_SOCKET_UTF8_MESSAGE_BUFFER_TYPE         = 2,
   WINHTTP_WEB_SOCKET_UTF8_FRAGMENT_BUFFER_TYPE        = 3,
   WINHTTP_WEB_SOCKET_CLOSE_BUFFER_TYPE                = 4
  };

enum _WINHTTP_WEB_SOCKET_CLOSE_STATUS
  {
   WINHTTP_WEB_SOCKET_SUCCESS_CLOSE_STATUS                = 1000,
   WINHTTP_WEB_SOCKET_ENDPOINT_TERMINATED_CLOSE_STATUS    = 1001,
   WINHTTP_WEB_SOCKET_PROTOCOL_ERROR_CLOSE_STATUS         = 1002,
   WINHTTP_WEB_SOCKET_INVALID_DATA_TYPE_CLOSE_STATUS      = 1003,
   WINHTTP_WEB_SOCKET_EMPTY_CLOSE_STATUS                  = 1005,
   WINHTTP_WEB_SOCKET_ABORTED_CLOSE_STATUS                = 1006,
   WINHTTP_WEB_SOCKET_INVALID_PAYLOAD_CLOSE_STATUS        = 1007,
   WINHTTP_WEB_SOCKET_POLICY_VIOLATION_CLOSE_STATUS       = 1008,
   WINHTTP_WEB_SOCKET_MESSAGE_TOO_BIG_CLOSE_STATUS        = 1009,
   WINHTTP_WEB_SOCKET_UNSUPPORTED_EXTENSIONS_CLOSE_STATUS = 1010,
   WINHTTP_WEB_SOCKET_SERVER_ERROR_CLOSE_STATUS           = 1011,
   WINHTTP_WEB_SOCKET_SECURE_HANDSHAKE_ERROR_CLOSE_STATUS = 1015
  };

#define WINHTTP_WEB_SOCKET_MAX_CLOSE_REASON_LENGTH 123
#define WINHTTP_FLAG_SECURE                0x00800000

#define WINHTTP_ACCESS_TYPE_DEFAULT_PROXY               0

#define WINHTTP_OPTION_SECURITY_FLAGS                   31
#define WINHTTP_OPTION_SECURE_PROTOCOLS                 84
#define WINHTTP_OPTION_UPGRADE_TO_WEB_SOCKET            114
#define WINHTTP_OPTION_WEB_SOCKET_CLOSE_TIMEOUT         115
#define WINHTTP_OPTION_WEB_SOCKET_KEEPALIVE_INTERVAL    116
#define WINHTTP_OPTION_WEB_SOCKET_RECEIVE_BUFFER_SIZE   122
#define WINHTTP_OPTION_WEB_SOCKET_SEND_BUFFER_SIZE      123

#define SECURITY_FLAG_IGNORE_UNKNOWN_CA         0x00000100
#define SECURITY_FLAG_IGNORE_CERT_DATE_INVALID  0x00002000
#define SECURITY_FLAG_IGNORE_CERT_CN_INVALID    0x00001000
#define SECURITY_FLAG_IGNORE_CERT_WRONG_USAGE   0x00000200

#define ERROR_INVALID_PARAMETER          87L
#define ERROR_INVALID_OPERATION          4317L

#import "winhttp.dll"
HINTERNET WinHttpOpen(string,DWORD,string,string,DWORD);
HINTERNET WinHttpConnect(HINTERNET,string,INTERNET_PORT,DWORD);
HINTERNET WinHttpOpenRequest(HINTERNET,string,string,string,string,string,DWORD);
bool WinHttpSetOption(HINTERNET,DWORD,LPVOID[],DWORD);
bool WinHttpQueryOption(HINTERNET,DWORD,LPVOID[],DWORD&);
bool WinHttpSetTimeouts(HINTERNET,int,int,int,int);
HINTERNET WinHttpSendRequest(HINTERNET,string,DWORD,LPVOID[],DWORD,DWORD,DWORD);
bool WinHttpReceiveResponse(HINTERNET,LPVOID[]);
HINTERNET WinHttpWebSocketCompleteUpgrade(HINTERNET,DWORD&);
bool WinHttpCloseHandle(HINTERNET);
DWORD WinHttpWebSocketSend(HINTERNET,WINHTTP_WEB_SOCKET_BUFFER_TYPE,BYTE&[],DWORD);
DWORD WinHttpWebSocketReceive(HINTERNET,BYTE&[],DWORD,DWORD&,WINHTTP_WEB_SOCKET_BUFFER_TYPE&);
DWORD WinHttpWebSocketClose(HINTERNET,ushort,BYTE&[],DWORD);
DWORD WinHttpWebSocketQueryCloseStatus(HINTERNET,ushort&,BYTE&[],DWORD,DWORD&);
#import
//+------------------------------------------------------------------+
```

### Using the winhttp functions

To establish a WebSocket client with these functions, first we have have to call WinHttpOpen() to initialize the library. The function returns a session handle to be used in subsequent calls to other winhttp library functions.

```
#include<winhttp.mqh>

HINTERNET sessionhandle,connectionhandle,requesthandle,websockethandle;

//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
//---
   sessionhandle=connectionhandle=requesthandle=websockethandle=NULL;

   sessionhandle=WinHttpOpen("MT5 app",WINHTTP_ACCESS_TYPE_DEFAULT_PROXY,NULL,NULL,0);

   if(sessionhandle==NULL)
     {
      Print("WinHttpOpen error" +string(kernel32::GetLastError()));
      return;
     }
```

The second step is the creation of a connection handle which is done with the help of WinHttpConnect(). It is here where we specify the server address and port number. It is important to note that at this point what is required is the domain name for the server without the scheme nor the path. The public IP address can also be used if it is known. Most errors that occur when using winhttp relate to passing an incorrectly formatted server address. For example, if the full server address is wss://ws.example.com/path, WinHttpConnect() expects only ws.example.com.

```
connectionhandle=WinHttpConnect(sessionhandle,server,Port,0);

   if(connectionhandle==NULL)
     {
      Print("WinHttpConnect error "+string(kernel32::GetLastError()));

      if(sessionhandle!=NULL)
         WinHttpCloseHandle(sessionhandle);

      return;
     }
```

With the connection handle successfully created, we will use it to establish a request handle by calling WinHttpOpenRequest(). Here we specify the path component if any from the server's address and also set the option to make the connection secure or not.

```
requesthandle=WinHttpOpenRequest(connectionhandle,"GET",path,NULL,NULL,NULL,(ExtTLS)?WINHTTP_FLAG_SECURE:0);

   if(requesthandle==NULL)
     {
      Print("WinHttpOpenRequest error "+string(kernel32::GetLastError()));

      if(connectionhandle!=NULL)
         WinHttpCloseHandle(connectionhandle);

      if(sessionhandle!=NULL)
         WinHttpCloseHandle(sessionhandle);

      return;
     }
```

Once that is done and we have a valid request handle, we prepare for the WebSocket handshake process by calling WinHttpSetOption().

```
uint nullpointer[]= {};
   if(!WinHttpSetOption(requesthandle,WINHTTP_OPTION_UPGRADE_TO_WEB_SOCKET,nullpointer,0))
     {
      Print("WinHttpSetOption upgrade error "+string(kernel32::GetLastError()));
      if(requesthandle!=NULL)
         WinHttpCloseHandle(requesthandle);

      if(connectionhandle!=NULL)
         WinHttpCloseHandle(connectionhandle);

      if(sessionhandle!=NULL)
         WinHttpCloseHandle(sessionhandle);

      return;
     }
```

This adds the required headers to an http request as specified by the WebSocket protocol. The WebSocket handshake is initiated by calling WinHttpSendRequest() and then WinHttpReceiveResponse() to confirm receipt of a response to our request.

```
if(!WinHttpSendRequest(requesthandle,NULL,0,nullpointer,0,0,0))
     {
      Print("WinHttpSendRequest error "+string(kernel32::GetLastError()));
      if(requesthandle!=NULL)
         WinHttpCloseHandle(requesthandle);

      if(connectionhandle!=NULL)
         WinHttpCloseHandle(connectionhandle);

      if(sessionhandle!=NULL)
         WinHttpCloseHandle(sessionhandle);

      return;
     }

   if(!WinHttpReceiveResponse(requesthandle,nullpointer))
     {
      Print("WinHttpRecieveResponse no response "+string(kernel32::GetLastError()));
      if(requesthandle!=NULL)
         WinHttpCloseHandle(requesthandle);

      if(connectionhandle!=NULL)
         WinHttpCloseHandle(connectionhandle);

      if(sessionhandle!=NULL)
         WinHttpCloseHandle(sessionhandle);

      return;
     }
```

WinHttpWebSocketCompleteUpgrade() checks the response and ensures its adherence to the WebSocket protocol. If satisfied, the function returns the coveted WebSocket handle.

```
ulong nv=0;
   websockethandle=WinHttpWebSocketCompleteUpgrade(requesthandle,nv);
   if(websockethandle==NULL)
     {
      Print("WinHttpWebSocketCompleteUpgrade error "+string(kernel32::GetLastError()));
      if(requesthandle!=NULL)
         WinHttpCloseHandle(requesthandle);

      if(connectionhandle!=NULL)
         WinHttpCloseHandle(connectionhandle);

      if(sessionhandle!=NULL)
         WinHttpCloseHandle(sessionhandle);

      return;
     }

   WinHttpCloseHandle(requesthandle);
   requesthandle=NULL;
```

From henceforth, our WebSocket client is fully functional and we can use WinHttpWebSocketSend() to send and WinHttpWebSocketReceive() to receive data. Since the WebSocket handle has been created, the request handle is no longer required because our http connection has been upgraded to a WebSocket connection. We then can release any resources associated with the request handle by calling WinHttpCloseHandle().

```
bool WebsocketSend(const string message)
  {
   BYTE msg_array[];

   StringToCharArray(message,msg_array,0,WHOLE_ARRAY);

   ArrayRemove(msg_array,ArraySize(msg_array)-1,1);

   DWORD len=(ArraySize(msg_array));

   ulong send=WinHttpWebSocketSend(websockethandle,WINHTTP_WEB_SOCKET_BINARY_MESSAGE_BUFFER_TYPE,msg_array,len);

   if(send)
      return(false);

   return(true);
  }
//+------------------------------------------------------------------+
bool WebSocketRecv(uchar &rxbuffer[],ulong &bytes_read)
  {
   WINHTTP_WEB_SOCKET_BUFFER_TYPE rbuffertype=-1;

   BYTE rbuffer[65539];

   ulong rbuffersize=ulong(ArraySize(rbuffer));

   ulong done=0;
   ulong transferred=0;
   ZeroMemory(rxbuffer);
   ZeroMemory(rbuffer);
   bytes_read=0;
   int called=0;

   do
     {
      called++;
      ulong get=WinHttpWebSocketReceive(websockethandle,rbuffer,rbuffersize,transferred,rbuffertype);
      if(get)
        {
         return(false);
        }

      ArrayCopy(rxbuffer,rbuffer,(int)done,0,(int)transferred);

      done+=transferred;

      transferred=0;

      ZeroMemory(rbuffer);

     }
   while(rbuffertype==WINHTTP_WEB_SOCKET_UTF8_FRAGMENT_BUFFER_TYPE || rbuffertype==WINHTTP_WEB_SOCKET_BINARY_FRAGMENT_BUFFER_TYPE);

   Print("Buffer type is "+EnumToString(rbuffertype)+" bytes read "+IntegerToString(done)+" looped "+IntegerToString(called));

   bytes_read=done;

   return(true);

  }

//+------------------------------------------------------------------+
```

Calling WinHttpWebSocketClose() closes a WebSocket connection. Once a connection is closed, all the handles associated with it should be deinitilized by calling

WinHttpCloseHandle() for each.

```
BYTE closearray[]= {};

   ulong close=WinHttpWebSocketClose(websockethandle,WINHTTP_WEB_SOCKET_SUCCESS_CLOSE_STATUS,closearray,0);
   if(close)
     {
      Print("websocket close error "+string(kernel32::GetLastError()));
      if(requesthandle!=NULL)
         WinHttpCloseHandle(requesthandle);

      if(websockethandle!=NULL)
         WinHttpCloseHandle(websockethandle);

      if(connectionhandle!=NULL)
         WinHttpCloseHandle(connectionhandle);

      if(sessionhandle!=NULL)
         WinHttpCloseHandle(sessionhandle);

      return;
     }
```

### The CWebsocket class

The websocket.mqh file will contain the CWebsocket class which will be a wrapper for the winhttp library functions needed to enable a WebSocket client.

The file begins with an include directive to admit all the functions and declarations imported from Windows API libraries.

```
#include<winhttp.mqh>

#define WEBSOCKET_ERROR_FIRST              WINHTTP_ERROR_LAST+1000
#define WEBSOCKET_ERROR_NOT_INITIALIZED    WEBSOCKET_ERROR_FIRST+1
#define WEBSOCKET_ERROR_EMPTY_SEND_BUFFER  WEBSOCKET_ERROR_FIRST+2
#define WEBSOCKET_ERROR_NOT_CONNECTED      WEBSOCKET_ERROR_FIRST+3
//+------------------------------------------------------------------+
//| websocket state enumeration                                      |
//+------------------------------------------------------------------+

enum ENUM_WEBSOCKET_STATE
  {
   CLOSED = 0,
   CLOSING,
   CONNECTING,
   CONNECTED
  };
```

To start the process of connecting to a websocket server, Connect() would be the first method to call.

Connect() parameters:

- \_serveraddress — server full address (type:string)

- \_port — server port number (type:ushort)
- \_appname — this is a string parameter that can be set to uniquely identify an application using the WebSocket client. It will be sent as one of the headers in the initial http request (type:string)
- \_secure — a boolean value that sets whether a secure connection should be used or not (type:boolean)

The Connect() method calls the private methods initialize() and upgrade() respectively. The private method initialize() processes the full server address and splits it into domain name and path components. Lastly, createSessionConnection() creates the session and connection handles. The upgrade() method comes into play to create the request and WebSocket handles before setting the new state of the client connection.

```
//+------------------------------------------------------------------------------------------------------+
//|Connect method used to set server parameters and establish client connection                          |
//+------------------------------------------------------------------------------------------------------+
bool CWebsocket::Connect(const string _serveraddress, const INTERNET_PORT _port=443, const string _appname=NULL,bool _secure=true)
  {
   if(clientState==CONNECTED)
    {
     if(StringCompare(_serveraddress,serveraddress,false))
       Abort();
     else
       return(true);
    }

   if(!initialize(_serveraddress,_port,appname,_secure))
     return(false);

   return(upgrade());
  }
//+---------------------------------------------------------------------------------+
//| private method used to set the server parameters.                               |
//+---------------------------------------------------------------------------------+
bool CWebsocket::initialize(const string _serveraddress,const ushort _port,const string _appname,bool _secure)
  {
   if(initialized)
      return(true);

   if(_secure)
      isSecure=true;

   if(_port==0)
     {
      if(isSecure)
         serverPort=443;
      else
         serverPort=80;
     }
   else
     {
      serverPort=_port;
      isSecure=_secure;

      if(serverPort==443 && !isSecure)
        isSecure=true;
     }



   if(_appname!=NULL)
      appname=_appname;
   else
      appname="Mt5 app";

   serveraddress=_serveraddress;

   int dot=StringFind(serveraddress,".");

   int ss=(dot>0)?StringFind(serveraddress,"/",dot):-1;

   serverPath=(ss>0)?StringSubstr(serveraddress,ss+1):"/";

   int sss=StringFind(serveraddress,"://");

   if(sss<0)
      sss=-3;

   serverName=StringSubstr(serveraddress,sss+3,ss-(sss+3));

   initialized=createSessionConnection();

   return(initialized);
  }
//+------------------------------------------------------------------+
//|creates the session and connection handles for the client         |
//+------------------------------------------------------------------+
bool CWebsocket::createSessionConnection(void)
  {
   hSession=WinHttpOpen(appname,WINHTTP_ACCESS_TYPE_DEFAULT_PROXY,NULL,NULL,0);

   if(hSession==NULL)
     {
      setErrorDescription();
      return(false);
     }

   hConnection=WinHttpConnect(hSession,serverName,serverPort,0);

   if(hConnection==NULL)
     {
      setErrorDescription();
      reset();
      return(false);
     }

   return(true);

  }
//+---------------------------------------------------------------------+
//|helper method that sets up the required request and websocket handles|
//+---------------------------------------------------------------------+
bool CWebsocket::upgrade(void)
  {
   clientState=CONNECTING;

   hRequest=WinHttpOpenRequest(hConnection,"GET",serverPath,NULL,NULL,NULL,(isSecure)?WINHTTP_FLAG_SECURE:0);

   if(hRequest==NULL)
     {
      setErrorDescription();
      reset();
      return(false);
     }

   uint nullpointer[]= {};
   if(!WinHttpSetOption(hRequest,WINHTTP_OPTION_UPGRADE_TO_WEB_SOCKET,nullpointer,0))
     {
      setErrorDescription();
      reset();
      return(false);
     }

   if(!WinHttpSendRequest(hRequest,NULL,0,nullpointer,0,0,0))
     {
      setErrorDescription();
      reset();
      return(false);
     }

   if(!WinHttpReceiveResponse(hRequest,nullpointer))
     {
      setErrorDescription();
      reset();
      return(false);
     }

   ulong nv=0;
   hWebSocket=WinHttpWebSocketCompleteUpgrade(hRequest,nv);
   if(hWebSocket==NULL)
     {
      setErrorDescription();
      reset();
      return(false);
     }

   WinHttpCloseHandle(hRequest);
   hRequest=NULL;
   clientState=CONNECTED;

   return(true);

  }
```

If the Connect() method returns 'true', we can begin sending data via the WebSocket client. To facilitate this, there are two methods that can be used.

SendString() method takes a string  as an input and the Send() method takes an unsigned character array as its sole function parameter. Both return 'true' on success and call the private method clientsend() which handles all send operations for the class.

```
//+------------------------------------------------------------------+
//| helper method for sending data to the server                     |
//+------------------------------------------------------------------+
bool CWebsocket::clientsend(BYTE &txbuffer[],WINHTTP_WEB_SOCKET_BUFFER_TYPE buffertype)
  {
   DWORD len=(ArraySize(txbuffer));

   if(len<=0)
     {
      setErrorDescription(WEBSOCKET_ERROR_EMPTY_SEND_BUFFER);
      return(false);
     }

   ulong send=WinHttpWebSocketSend(hWebSocket,WINHTTP_WEB_SOCKET_BINARY_MESSAGE_BUFFER_TYPE,txbuffer,len);

   if(send)
     {
      setErrorDescription();
      return(false);
     }

   return(true);

  }

//+------------------------------------------------------------------+
//|public method for sending raw string messages                     |
//+------------------------------------------------------------------+
bool CWebsocket::SendString(const string msg)
  {
   if(!initialized)
     {
      setErrorDescription(WEBSOCKET_ERROR_NOT_INITIALIZED);
      return(false);
     }

   if(clientState!=CONNECTED)
     {
      setErrorDescription(WEBSOCKET_ERROR_NOT_CONNECTED);
      return(false);
     }

   if(StringLen(msg)<=0)
     {
      setErrorDescription(WEBSOCKET_ERROR_EMPTY_SEND_BUFFER);
      return(false);
     }

   BYTE msg_array[];

   StringToCharArray(msg,msg_array,0,WHOLE_ARRAY);

   ArrayRemove(msg_array,ArraySize(msg_array)-1,1);

   DWORD len=(ArraySize(msg_array));

   return(clientsend(msg_array,WINHTTP_WEB_SOCKET_BINARY_MESSAGE_BUFFER_TYPE));
  }

//+------------------------------------------------------------------+
//|Public method for sending data prepackaged in an array            |
//+------------------------------------------------------------------+
bool CWebsocket::Send(BYTE &buffer[])
  {
   if(!initialized)
     {
      setErrorDescription(WEBSOCKET_ERROR_NOT_INITIALIZED);
      return(false);
     }

   if(clientState!=CONNECTED)
     {
      setErrorDescription(WEBSOCKET_ERROR_NOT_CONNECTED);
      return(false);
     }

   return(clientsend(buffer,WINHTTP_WEB_SOCKET_BINARY_MESSAGE_BUFFER_TYPE));
  }
```

To read data sent from the server, we can use either Read() or ReadString(). The methods return the size of the data received. ReadString() requires a string passed by reference to which data received will be written to, whilst Read() writes to an unsigned character array.

```
//+------------------------------------------------------------------+
//|helper method for reading received messages from the server       |
//+------------------------------------------------------------------+
void CWebsocket::clientread(BYTE &rbuffer[],ulong &bytes)
  {

   WINHTTP_WEB_SOCKET_BUFFER_TYPE rbuffertype=-1;

   ulong done=0;
   ulong transferred=0;
   ZeroMemory(rbuffer);
   ZeroMemory(rxbuffer);
   bytes=0;

   do
     {
      ulong get=WinHttpWebSocketReceive(hWebSocket,rxbuffer,rxsize,transferred,rbuffertype);
      if(get)
        {
         setErrorDescription();
         return;
        }

      ArrayCopy(rbuffer,rxbuffer,(int)done,0,(int)transferred);

      done+=transferred;

      ZeroMemory(rxbuffer);

      transferred=0;

     }
   while(rbuffertype==WINHTTP_WEB_SOCKET_UTF8_FRAGMENT_BUFFER_TYPE || rbuffertype==WINHTTP_WEB_SOCKET_BINARY_FRAGMENT_BUFFER_TYPE);

   bytes=done;

   return;

  }

//+------------------------------------------------------------------+
//|public method for reading data sent from the server               |
//+------------------------------------------------------------------+
ulong CWebsocket::Read(BYTE &buffer[])
  {
   if(!initialized)
     {
      setErrorDescription(WEBSOCKET_ERROR_NOT_INITIALIZED);
      return(false);
     }

   if(clientState!=CONNECTED)
     {
      setErrorDescription(WEBSOCKET_ERROR_NOT_CONNECTED);
      return(false);
     }

   ulong bytes_read_from_socket=0;

   clientread(buffer,bytes_read_from_socket);

   return(bytes_read_from_socket);

  }
//+------------------------------------------------------------------+
//|public method for reading data sent from the server               |
//+------------------------------------------------------------------+
ulong CWebsocket::ReadString(string &_response)
  {
   if(!initialized)
     {
      setErrorDescription(WEBSOCKET_ERROR_NOT_INITIALIZED);
      return(false);
     }

   if(clientState!=CONNECTED)
     {
      setErrorDescription(WEBSOCKET_ERROR_NOT_CONNECTED);
      return(false);
     }

   ulong bytes_read_from_socket=0;
   BYTE buffer[];

   clientread(buffer,bytes_read_from_socket);

   _response=(bytes_read_from_socket)?CharArrayToString(buffer):NULL;

   return(bytes_read_from_socket);

  }
```

When the WebSocket client is no longer needed, the connection to the server can be closed with either Close() or Abort(). The Abort() method differs from the Close() method in that it not only closes a WebSocket connection but also goes further to reset the values of some class properties setting them to their default state.

```
//+------------------------------------------------------------------+
//| Closes a websocket client connection                             |
//+------------------------------------------------------------------+
void CWebsocket::Close(void)
  {
   if(clientState==CLOSED)
      return;

   clientState=CLOSING;

   BYTE nullpointer[]= {};

   ulong result=WinHttpWebSocketClose(hWebSocket,WINHTTP_WEB_SOCKET_SUCCESS_CLOSE_STATUS,nullpointer,0);
   if(result)
      setErrorDescription();

   reset();

   return;
  }

//+--------------------------------------------------------------------------+
//|method for abandoning a client connection. All previous server connection |
//|   parameters are reset to their default state                            |
//+--------------------------------------------------------------------------+
void CWebsocket::Abort(void)
  {
   Close();
//---
   serveraddress=serverName=serverPath=NULL;
   serverPort=0;
   isSecure=false;
   last_error=0;
   StringFill(errormsg,0);
//---
   return;
  }
```

ClientState() queries the current state of the WebSocket client.

DomainName(), Port() and ServerPath() return the domain name, port and path component respectively for the current connection.

LastErrorMessage() can be used to get the last error as a detailed string, whereas a call to LastError() retrieves the error code as an integral value.

```
//public getter methods
   string            LastErrorMessage(void)          {  return(errormsg);    }
   uint              LastError(void)      {  return(last_error);  }
   ENUM_WEBSOCKET_STATE ClientState(void) {  return(clientState); }
   string            DomainName(void)                {  return(serverName);  }
   INTERNET_PORT     Port(void)               {  return(serverPort);  }
   string            ServerPath(void)                {  return(serverPath);  }
```

The whole class is shown below.

```
//+------------------------------------------------------------------+
//|Class CWebsocket                                                  |
//| Purpose: class for websocket client                              |
//+------------------------------------------------------------------+

class CWebsocket
  {
private:
   ENUM_WEBSOCKET_STATE clientState;            //websocket state
   HINTERNET            hSession;               //winhttp session handle
   HINTERNET            hConnection;            //winhttp connection handle
   HINTERNET            hWebSocket;             //winhttp websocket handle
   HINTERNET            hRequest;               //winhtttp request handle
   string               appname;                //optional application name sent as one of the headers in initial http request
   string               serveraddress;          //full server address
   string               serverName;             //server domain name
   INTERNET_PORT        serverPort;             //port number
   string               serverPath;             //server path
   bool                 initialized;            //boolean flag that denotes the state of underlying winhttp infrastruture required for client
   BYTE                 rxbuffer[];             //internal buffer for reading from the socket
   bool                 isSecure;               //secure connection flag
   ulong                rxsize;                 //rxbuffer arraysize
   string               errormsg;               //internal buffer for error messages
   uint                 last_error;             //last winhttp/win32/class specific error
   // private methods
   bool              initialize(const string _serveraddress, const INTERNET_PORT _port, const string _appname,bool _secure);
   bool              createSessionConnection(void);
   bool              upgrade(void);
   void              reset(void);
   bool              clientsend(BYTE &txbuffer[],WINHTTP_WEB_SOCKET_BUFFER_TYPE buffertype);
   void              clientread(BYTE &rxbuffer[],ulong &bytes);
   void              setErrorDescription(uint error=0);

public:
                     CWebsocket(void):clientState(0),
                     hSession(NULL),
                     hConnection(NULL),
                     hWebSocket(NULL),
                     hRequest(NULL),
                     serveraddress(NULL),
                     serverName(NULL),
                     serverPort(0),
                     initialized(false),
                     isSecure(false),
                     rxsize(65539),
                     errormsg(NULL),
                     last_error(0)
     {
      ArrayResize(rxbuffer,(int)rxsize);
      ArrayFill(rxbuffer,0,rxsize,0);
      StringInit(errormsg,1000);
     }

                    ~CWebsocket(void)
     {
      Close();
      ArrayFree(rxbuffer);
     }
   //public methods

   bool              Connect(const string _serveraddress, const INTERNET_PORT _port=443, const string _appname=NULL,bool _secure=true);
   void              Close(void);
   bool              SendString(const string msg);
   bool              Send(BYTE &buffer[]);
   ulong             ReadString(string &response);
   ulong             Read(BYTE &buffer[]);
   void              Abort(void);
   void              ResetLastError(void)
     {
      last_error=0;
      StringFill(errormsg,0);
      ::ResetLastError();
     }
   //public getter methods
   string            LastErrorMessage(void)          {  return(errormsg);    }
   uint              LastError(void)      {  return(last_error);  }
   ENUM_WEBSOCKET_STATE ClientState(void) {  return(clientState); }
   string            DomainName(void)                {  return(serverName);  }
   INTERNET_PORT     Port(void)               {  return(serverPort);  }
   string            ServerPath(void)                {  return(serverPath);  }

  };
```

Now that we have our WebSocket class, we can consider an example of its use.

### Testing the CWebsocket class

For testing, we will create a MetaTrader 5 application that adds a custom symbol from Deriv.com. When loaded onto a chart, it will download history and open up a new chart for the custom symbol that will be updated with live tick data.

There will be two versions. DerivCustomSymboWithTickHistory.ex5 will use tick history, whilst the other DerivCustomSymbolWithBarHistory.ex5 will download [OHLC](https://en.wikipedia.org/wiki/Open-high-low-close_chart "https://en.wikipedia.org/wiki/Open-high-low-close_chart") bar history. Both will have similar code.

Deriv.com provides a well [documented API](https://www.mql5.com/go?link=https://api.deriv.com/ "https://api.deriv.com/") that enables developers to build interfaces that interact with their systems. The API relies on websockets with queries and responses provided in [JSON format](https://en.wikipedia.org/wiki/JSON "https://en.wikipedia.org/wiki/JSON").

![Deriv developer portal](https://c.mql5.com/2/73/DerivApi__1.PNG)

The application will be implemented as an Expert Advisor that enlists the help of three important libraries:

- The first is websocket.mqh to handle WebSocket connections,

- the second is JAson.mqh for working with JSON formatted data authored by Alexey Sergeev and [obtainable from vivazzi's github repository](https://www.mql5.com/go?link=https://github.com/vivazzi/JAson "https://github.com/vivazzi/JAson"),

- the third library we will need is FileTxt.mqh for handling file operations.

The EA will have the following user adjustable inputs:

- DERIV\_appid - this is a string parameter required to grant our application access to the API, an app ID can be acquired by following instructions provided on the [developers' portal](https://www.mql5.com/go?link=https://api.deriv.com/ "https://api.deriv.com/"). Subscribing to a symbol tickstream does not require user authentication on Deriv.com, that is why there is no need to specify an API token.
- DERIV\_symbol - this is an enumeration that allows users to select a symbol they want imported into MetaTrader 5.
- DERIV\_timeframe - this is the timeframe of the chart that will be opened once history data has been downloaded and added to MetaTrader 5.

```
//+------------------------------------------------------------------+
//|                             DerivCustomSymbolWithTickHistory.mq5 |
//|                                  Copyright 2024, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"

#include<websocket.mqh>
#include<JAson.mqh>
#include<Files/FileTxt.mqh>

#define DERIV_URL "wss://ws.derivws.com/websockets/v3?app_id="
#define DERIV_SYMBOL_SETTINGS "derivsymbolset.json"
#define DERIV_SYMBOL_BASE_PATH "Deriv.com\\"

enum ENUM_DERIV_SYMBOL
{
DERIV_1HZ10V=0,//Volatility 10 (1s)
DERIV_1HZ25V,//Volatility 25 (1s)
DERIV_1HZ50V,//Volatility 50 (1s)
DERIV_1HZ75V,//Volatility 75 (1s)
DERIV_1HZ100V,//Volatility 100 (1s)
DERIV_1HZ200V,//Volatility 200 (1s)
DERIV_1HZ300V,//Volatility 300 (1s)
DERIV_BOOM300N,//BOOM 300
DERIV_BOOM500,//BOOM 500
DERIV_BOOM1000,//BOOM 1000
DERIV_CRASH300N,//CRASH 300
DERIV_CRASH500,//CRASH 500
DERIV_CRASH1000,//CRASH 1000
DERIV_cryBTCUSD,//BTCUSD
DERIV_cryETHUSD,//ETHUSD
DERIV_frxAUDCAD,//AUDCAD
DERIV_frxAUDCHF,//AUDCHF
DERIV_frxAUDJPY,//AUDJPY
DERIV_frxAUDNZD,//AUDNZD
DERIV_frxAUDUSD,//AUDUSD
DERIV_frxBROUSD,//BROUSD
DERIV_frxEURAUD,//EURAUD
DERIV_frxEURCAD,//EURCAD
DERIV_frxEURCHF,//EURCHF
DERIV_frxEURGBP,//EURGBP
DERIV_frxEURJPY,//EURJPY
DERIV_frxEURNZD,//EURNZD
DERIV_frxEURUSD,//EURUSD
DERIV_frxGBPAUD,//GBPAUD
DERIV_frxGBPCAD,//GBPCAD
DERIV_frxGBPCHF,//GBPCHF
DERIV_frxGBPJPY,//GBPJPY
DERIV_frxGBPNOK,//GBPNOK
DERIV_frxGBPNZD,//GBPNZD
DERIV_frxGBPUSD,//GBPUSD
DERIV_frxNZDJPY,//NZDJPY
DERIV_frxNZDUSD,//NZDUSD
DERIV_frxUSDCAD,//USDCAD
DERIV_frxUSDCHF,//USDCHF
DERIV_frxUSDJPY,//USDJPY
DERIV_frxUSDMXN,//USDMXN
DERIV_frxUSDNOK,//USDNOK
DERIV_frxUSDPLN,//USDPLN
DERIV_frxUSDSEK,//USDSEK
DERIV_frxXAUUSD,//XAUUSD
DERIV_frxXAGUSD,//XAGUSD
DERIV_frxXPDUSD,//XPDUSD
DERIV_frxXPTUSD,//XPTUSD
DERIV_JD10,//Jump 10 Index
DERIV_JD25,//Jump 25 Index
DERIV_JD50,//Jump 50 Index
DERIV_JD75,//Jump 75 Index
DERIV_JD100,//Jump 100 Index
DERIV_OTC_AEX,//Dutch Index
DERIV_OTC_AS51,//Australian Index
DERIV_OTC_DJI,//Wall Street Index
DERIV_OTC_FCHI,//French Index
DERIV_OTC_FTSE,//UK Index
DERIV_OTC_GDAXI,//German Index
DERIV_OTC_HSI,//Hong Kong Index
DERIV_OTC_N225,//Japanese Index
DERIV_OTC_NDX,//US Tech Index
DERIV_OTC_SPC,//US Index
DERIV_OTC_SSMI,//Swiss Index
DERIV_OTC_SX5E,//Euro 50 Index
DERIV_R_10,//Volatility 10 Index
DERIV_R_25,//Volatility 25 Index
DERIV_R_50,//Volatility 50 Index
DERIV_R_75,//Volatility 75 Index
DERIV_R_100,//Volatility 100 Index
DERIV_RDBEAR,//Bear Market Index
DERIV_RDBULL,//Bull Market Index
DERIV_stpRNG,//Step Index
DERIV_WLDAUD,//AUD Index
DERIV_WLDEUR,//EUR Index
DERIV_WLDGBP,//GBP Index
DERIV_WLDUSD,//USD Index
DERIV_WLDXAU//Gold Index
};


input string DERIV_appid="";//Deriv.com registered application ID
input ENUM_DERIV_SYMBOL DERIV_symbol=DERIV_cryBTCUSD;//Deriv.com symbol
input ENUM_TIMEFRAMES DERIV_timeframe=PERIOD_M1;//Chart period
```


The EA will be made up of two classes – CCustomSymbol and CDerivSymbol.

## The CCustomSymbol class

The CCustomSymbol is a class for working with custom symbols from external sources. It is inspired by [fxsaber's](https://www.mql5.com/en/users/fxsaber) [SYMBOL library](https://www.mql5.com/en/code/18855). It provides methods for manipulating and retrieving the properties of symbols as well as opening and closing their corresponding charts amongst other features. More importantly, it provides three virtual methods that child classes can override to allow for variations in the implementation of a custom symbol.

```
//+------------------------------------------------------------------+
//|General class for creating custom symbols from external source    |
//+------------------------------------------------------------------+
class CCustomSymbol
  {
protected:
   string            m_symbol_name;       //symbol name
   datetime          m_history_start;     //existing tick history start date
   datetime          m_history_end;       //existing tick history end date
   bool              m_new;               //flag specifying whether a symbol has just been created or already exists in the terminal
   ENUM_TIMEFRAMES   m_chart_tf;          //chart timeframe
public:
   //constructor
                     CCustomSymbol(void)
     {
      m_symbol_name=NULL;
      m_chart_tf=PERIOD_M1;
      m_history_start=0;
      m_history_end=0;
      m_new=false;
     }
   //destructor
                    ~CCustomSymbol(void)
     {

     }
   //method for initializing symbol, sets the symbol name and chart timeframe properties
   virtual bool      Initialize(const string sy,string sy_path=NULL, ENUM_TIMEFRAMES chart_tf=PERIOD_M1)
     {
      m_symbol_name=sy;
      m_chart_tf=chart_tf;
      return(InitSymbol(sy_path));
     }
   //gets the symbol name
   string            Name(void) const
     {
      return(m_symbol_name);
     }
   //sets the history start date
   bool              SetHistoryStartDate(const datetime startime)
     {
      if(startime>=TimeLocal())
        {
         Print("Invalid history start time");
         return(false);
        }

      m_history_start=startime;

      return(true);
     }
   //gets the history start date
   datetime          GetHistoryStartDate(void)
     {
      return(m_history_start);
     }
   //general methods for setting the properties of the custom symbol
   bool              SetProperty(const ENUM_SYMBOL_INFO_DOUBLE Property, double Value) const
     {
      return(::CustomSymbolSetDouble(m_symbol_name, Property, Value));
     }

   bool              SetProperty(const ENUM_SYMBOL_INFO_INTEGER Property, long Value) const
     {
      return(::CustomSymbolSetInteger(m_symbol_name, Property, Value));
     }

   bool              SetProperty(const ENUM_SYMBOL_INFO_STRING Property, string Value) const
     {
      return(::CustomSymbolSetString(m_symbol_name, Property, Value));
     }
   //general methods for getting the symbol properties of the custom symbol
   long              GetProperty(const ENUM_SYMBOL_INFO_INTEGER Property) const
     {
      return(::SymbolInfoInteger(m_symbol_name, Property));
     }

   double            GetProperty(const ENUM_SYMBOL_INFO_DOUBLE Property) const
     {
      return(::SymbolInfoDouble(m_symbol_name, Property));
     }

   string            GetProperty(const ENUM_SYMBOL_INFO_STRING Property) const
     {
      return(::SymbolInfoString(m_symbol_name, Property));
     }
   //method for deleting a custom symbol
   bool              Delete(void)
     {
      return((bool)(GetProperty(SYMBOL_CUSTOM)) && DeleteAllCharts()  && ::CustomSymbolDelete(m_symbol_name) && SymbolSelect(m_symbol_name,false));
     }
   //unimplemented virtual method for adding new ticks
   virtual void      AddTick(void)
     {
      return;
     }
   //unimplemented virtual method for aquiring the either ticks or candle history from an external source
   virtual bool      UpdateHistory(void)
     {
      return(false);
     }

protected:
   //checks if the symbol already exists or not
   bool              SymbolExists(void)
     {
      return(SymbolSelect(m_symbol_name,true));
     }
   //method that opens a new chart according to the m_chart_tf property
   void              OpenChart(void)
     {
      long Chart = ::ChartFirst();

      bool opened=false;

      while(Chart != -1)
        {
         if((::ChartSymbol(Chart) == m_symbol_name))
           {
            ChartRedraw(Chart);
            if(ChartPeriod(Chart)==m_chart_tf)
               opened=true;
           }
         Chart = ::ChartNext(Chart);
        }

      if(!opened)
        {
         long id = ChartOpen(m_symbol_name,m_chart_tf);
         if(id == 0)
           {
            Print("Can't open new chart for " + m_symbol_name + ", code: " + (string)GetLastError());
            return;
           }
         else
           {
            Sleep(1000);
            ChartSetSymbolPeriod(id, m_symbol_name, m_chart_tf);
            ChartSetInteger(id, CHART_MODE,CHART_CANDLES);
           }
        }
     }
   //deletes all charts for the specified symbol
   bool              DeleteAllCharts(void)
     {
      long Chart = ::ChartFirst();

      while(Chart != -1)
        {
         if((Chart != ::ChartID()) && (::ChartSymbol(Chart) == m_symbol_name))
            if(!ChartClose(Chart))
              {
               Print("Error closing chart id ", Chart, m_symbol_name, ChartPeriod(Chart));
               return(false);
              }

         Chart = ::ChartNext(Chart);
        }

      return(true);
     }
   //helper method that initializes a custom symbol
   bool              InitSymbol(const string _path=NULL)
     {
      if(!SymbolExists())
        {
         if(!CustomSymbolCreate(m_symbol_name,_path))
           {
            Print("error creating custom symbol ", ::GetLastError());
            return(false);
           }

         if(!SetProperty(SYMBOL_CHART_MODE,SYMBOL_CHART_MODE_BID)   ||
            !SetProperty(SYMBOL_SWAP_MODE,SYMBOL_SWAP_MODE_DISABLED) ||
            !SetProperty(SYMBOL_TRADE_MODE,SYMBOL_TRADE_MODE_DISABLED))
           {
            Print("error setting symbol properties");
            return(false);
           }

         if(!SymbolSelect(m_symbol_name,true))
           {
            Print("error adding symbol to market watch",::GetLastError());
            return(false);
           }

         m_new=true;

         return(true);
        }
      else
        {
         long custom=GetProperty(SYMBOL_CUSTOM);

         if(!custom)
           {
            Print("Error, symbol is not custom ",m_symbol_name,::GetLastError());
            return(false);
           }

         m_history_end=GetLastBarTime();
         m_history_start=GetFirstBarTime();
         m_new=false;

         return(true);
        }
     }

   //gets the last tick time for an existing custom symbol

   datetime          GetLastTickTime(void)
     {
      MqlTick tick;

      ZeroMemory(tick);

      if(!SymbolInfoTick(m_symbol_name,tick))
        {
         Print("symbol info tick failure ", ::GetLastError());
         return(0);
        }
      else
         return(tick.time);
     }

   //gets the last bar time of the one minute timeframe in candle history
   datetime          GetLastBarTime(void)
     {
      MqlRates candle[1];

      ZeroMemory(candle);

      int bars=iBars(m_symbol_name,PERIOD_M1);

      if(bars<=0)
         return(0);

      if(CopyRates(m_symbol_name,PERIOD_M1,0,1,candle)>0)
         return(candle[0].time);
      else
         return(0);
     }
   //gets the first bar time of the one minute timeframe  in candle  history
   datetime          GetFirstBarTime(void)
     {
      MqlRates candle[1];

      ZeroMemory(candle);

      int bars=iBars(m_symbol_name,PERIOD_M1);

      if(bars<=0)
         return(0);

      if(CopyRates(m_symbol_name,PERIOD_M1,bars-1,1,candle)>0)
         return(candle[0].time);
      else
         return(0);

     }

  };
```

The Initialize() method parameters:

-   sy - this string parameter sets the symbol name for a custom symbol
-   sy\_path - string parameter that sets the symbol path property
-   chart\_tf - the parameter sets a period of the chart that will be opened when the symbols history has been loaded


The method calls Initsymbol(), which either creates a new custom symbol if it does not already exist or loads up the history properties if the symbol exists.

The other two virtual methods, UpdateHistory() and AddTick(), are not implemented in CCustomSymbol. Any derived class will have to override these methods.

## The CDerivSymbol class

This is where the CDerivSymbol class comes in. It inherits from CCustomSymbol and provides methods that override all the virtual methods of its parent class. Its here where we will use our WebSocket client to consume the Deriv.com API.

```
//+------------------------------------------------------------------+
//|Class for creating custom Deriv.com specific symbols              |
//+------------------------------------------------------------------+
class CDerivSymbol:public CCustomSymbol
  {
private:
   //private properties
   string            m_appID;      //app id string issued by Deriv.com
   string            m_url;        //final url
   string            m_stream_id;  //stream identifier for a symbol
   int               m_index;      //array index
   int               m_max_ticks;  //max number of ticks downloadable from Deriv.com
   CWebsocket*       websocket;    //websocket client
   CJAVal*           json;         //utility json object
   CJAVal*           symbolSpecs;  //json object storing symbol specification
   //private methods
   bool              CheckDerivError(CJAVal &j);
   bool              GetSymbolSettings(void);
public:
   //Constructor
                     CDerivSymbol(void):m_appID(NULL),
                     m_url(NULL),
                     m_stream_id(NULL),
                     m_index(-1),
                     m_max_ticks(86400)
     {
      json=new CJAVal();
      symbolSpecs=new CJAVal();
      websocket=new CWebsocket();
     }
   //Destructor
                    ~CDerivSymbol(void)
     {
      if(CheckPointer(websocket)==POINTER_DYNAMIC)
        {
         if(m_stream_id!="")
            StopTicksStream();
         delete websocket;
        }

      if(CheckPointer(json)==POINTER_DYNAMIC)
         delete json;

      if(CheckPointer(symbolSpecs)==POINTER_DYNAMIC)
         delete symbolSpecs;

      Comment("");

     }
   //public methods
   virtual void      AddTick(void) override;
   virtual bool      Initialize(const string sy,string sy_path=NULL, ENUM_TIMEFRAMES chart_tf=PERIOD_M1) override;
   virtual bool      UpdateHistory(void) override;
   void              SetMaximumHistoryTicks(const int max) { m_max_ticks=(max>=5000)?max:86400; }
   void              SetAppID(const string id);
   bool              StartTicksStream(void);
   bool              StopTicksStream(void);
  };
```

After an instance of the CDerivSymbol class is created, a valid application identifier app\_id should be set using the SetAppID() method. Only then can we move on to initialize a custom symbol.

```
//+------------------------------------------------------------------+
//|sets the the application id used to consume Deriv.com api         |
//+------------------------------------------------------------------+
void CDerivSymbol::SetAppID(const string id)
  {
   if(m_appID!=NULL && StringCompare(id,m_appID,false))
      websocket.Abort();

   m_appID=id;
   m_url=DERIV_URL+m_appID;

  }
```

The Initialize() method uses the getSymbolSpecs() private method to get the properties of a selected symbol. The relevant information is then used to set the symbol properties for a new custom symbol.

```
//+------------------------------------------------------------------+
//|Begins process of creating custom symbol                          |
//+------------------------------------------------------------------+
bool CDerivSymbol::Initialize(const string sy,string sy_path=NULL, ENUM_TIMEFRAMES chart_tf=PERIOD_M1)
  {

   if(CheckPointer(websocket)==POINTER_INVALID || CheckPointer(json)==POINTER_INVALID || CheckPointer(symbolSpecs)==POINTER_INVALID)
     {
      Print("Invalid pointer found ");
      return(false);
     }


   if(m_appID=="")
     {
      Alert("Application ID has not been set, It is required for the program to work");
      return(false);
     }

   m_symbol_name=(StringFind(sy,"DERIV_")>=0)?StringSubstr(sy,6):sy;
   m_chart_tf=chart_tf;

   Comment("Initializing Symbol "+m_symbol_name+".......");

   if(!GetSymbolSettings())
      return(false);

   string s_path=DERIV_SYMBOL_BASE_PATH+symbolSpecs["active_symbols"][m_index]["market_display_name"].ToStr();
   string symbol_description=symbolSpecs["active_symbols"][m_index]["display_name"].ToStr();
   double s_point=symbolSpecs["active_symbols"][m_index]["pip"].ToDbl();
   int s_digits=(int)MathAbs(MathLog10(s_point));

   if(!InitSymbol(s_path))
      return(false);

   if(m_new)
     {
      if(!SetProperty(SYMBOL_DESCRIPTION,symbol_description) ||
         !SetProperty(SYMBOL_POINT,s_point)                 ||
         !SetProperty(SYMBOL_DIGITS,s_digits))
        {
         Print("error setting symbol properties ", ::GetLastError());
         return(false);
        }
     }

   Comment("Symbol "+m_symbol_name+" initialized.......");

           return(true);
  }
```

When the symbol has been initialized, we will need to acquire either rates or tick data to build the chart. This is done by the UpdateHistory() method. After loading the history to the terminal, a new chart for the custom symbol will be opened if it does not already exist. In the code shown below there are two versions of the UpdateHistory() method, the first one uses bar data to fill the history, whilst the second relies on tick data.

```
//+------------------------------------------------------------------+
//|method for updating the tick history for a particular symbol      |
//+------------------------------------------------------------------+
bool CDerivSymbol::UpdateHistory(void)
  {
   if(websocket.ClientState()!=CONNECTED && !websocket.Connect(m_url))
     {
      Print(websocket.LastErrorMessage()," : ",websocket.LastError());
      return(false);
     }

   Comment("Updating history for "+m_symbol_name+".......");

   MqlTick history_ticks[];
   string history=NULL;
   json.Clear();

   json["ticks_history"]=m_symbol_name;

   if(m_new)
     {
      if(m_history_start>0)
        {
         json["start"]=(int)(m_history_start);
        }
     }
   else
      if(m_history_end!=0)
        {
         json["start"]=(int)(m_history_start);
        }

   json["count"]=m_max_ticks;
   json["end"]="latest";
   json["style"]="ticks";

   if(!websocket.SendString(json.Serialize()))
     {
      Print(websocket.LastErrorMessage());
      return(false);
     }

   if(websocket.ReadString(history))
     {
      json.Deserialize(history);

      if(CheckDerivError(json))
         return(false);

      int i=0;

      int z=i;
      int diff=0;

      while(json["history"]["prices"][i].ToDbl()!=0.0)
        {

         diff=(i>0)?(int)(json["history"]["times"][i].ToInt() - json["history"]["times"][i-1].ToInt()):0;//((m_history_end>0)?(json["history"]["times"][i].ToInt() - (int)(m_history_end)):0);

         if(diff > 1)
           {
            int k=z+diff;
            int p=1;

            if(ArrayResize(history_ticks,k,100)!=k)
              {
               Print("Memory allocation error,  "+IntegerToString(::GetLastError()));
               return(false);
              }

            while(z<(k-1))
              {
               history_ticks[z].bid=json["history"]["prices"][i-1].ToDbl();
               history_ticks[z].ask=0;
               history_ticks[z].time=(datetime)(json["history"]["times"][i-1].ToInt()+p);
               history_ticks[z].time_msc=(long)((json["history"]["times"][i-1].ToInt()+p)*1000);
               history_ticks[z].last=0;
               history_ticks[z].volume=0;
               history_ticks[z].volume_real=0;
               history_ticks[z].flags=TICK_FLAG_BID;
               z++;
               p++;
              }

            history_ticks[z].bid=json["history"]["prices"][i].ToDbl();
            history_ticks[z].ask=0;
            history_ticks[z].time=(datetime)(json["history"]["times"][i].ToInt());
            history_ticks[z].time_msc=(long)((json["history"]["times"][i].ToInt())*1000);
            history_ticks[z].last=0;
            history_ticks[z].volume=0;
            history_ticks[z].volume_real=0;
            history_ticks[z].flags=TICK_FLAG_BID;

            i++;
            z++;
           }
         else
           {
            if(ArrayResize(history_ticks,z+1,100)==(z+1))
              {
               history_ticks[z].bid=json["history"]["prices"][i].ToDbl();
               history_ticks[z].ask=0;
               history_ticks[z].time=(datetime)json["history"]["times"][i].ToInt();
               history_ticks[z].time_msc=(long)(json["history"]["times"][i].ToInt()*1000);
               history_ticks[z].last=0;
               history_ticks[z].volume=0;
               history_ticks[z].volume_real=0;
               history_ticks[z].flags=TICK_FLAG_BID;
              }
            else
              {
               Print("Memory allocation error,  "+IntegerToString(::GetLastError()));
               return(false);
              }

            i++;
            z++;
           }
        }

      //Print("z is ",z,". Arraysize is ",ArraySize(history_ticks));

      if(m_history_end>0 && z>0)
        {
         DeleteAllCharts();

         if(CustomTicksDelete(m_symbol_name,int(m_history_start)*1000,(history_ticks[0].time_msc-1000))<0)
           {
            Print("error deleting ticks ", ::GetLastError());
            return(false);
           }
         else
           {
            m_history_end=history_ticks[z-1].time;
            m_history_start=history_ticks[0].time;
           }
        }

      if(ArraySize(history_ticks)>0)
        {
         //ArrayPrint(history_ticks);
         if(CustomTicksAdd(m_symbol_name,history_ticks)<0)//CustomTicksReplace(m_symbol_name,history_ticks[0].time_msc,history_ticks[z-1].time_msc,history_ticks)
           {
            Print("Error adding history "+IntegerToString(::GetLastError()));
            return(false);
           }
        }
      else
        {
         Print("Received unexpected response from server ",IntegerToString(::GetLastError()), " "+history);
         return(false);
        }
     }
   else
     {
      Print("error reading "," error: ",websocket.LastError(), websocket.LastErrorMessage());
      return(false);
     }

   OpenChart();

   return(true);

  }

//+------------------------------------------------------------------+
//|method for updating the tick history for a particular symbol      |
//+------------------------------------------------------------------+
bool CDerivSymbol::UpdateHistory(void)
  {
   if(websocket.ClientState()!=CONNECTED && !websocket.Connect(m_url))
     {
      Print(websocket.LastErrorMessage()," : ",websocket.LastError());
      return(false);
     }

   Comment("Updating history for "+m_symbol_name+".......");

   MqlRates history_candles[];
   string history=NULL;
   json.Clear();

   json["ticks_history"]=m_symbol_name;

   if(m_new)
     {
      if(m_history_start>0)
        {
         json["start"]=(int)(m_history_start);
        }
     }
   else
      if(m_history_end!=0)
        {
         json["start"]=(int)(m_history_start);
        }

   json["end"]="latest";
   json["style"]="candles";

   if(!websocket.SendString(json.Serialize()))
     {
      Print(websocket.LastErrorMessage());
      return(false);
     }

   if(websocket.ReadString(history))
     {
      json.Deserialize(history);

      if(CheckDerivError(json))
         return(false);

      int i=0;

      if(ArrayResize(history_candles,(json["candles"].Size()),100)<0)
        {
         Print("Last error is "+IntegerToString(::GetLastError()));
         return(false);
        }

      while(json["candles"][i]["open"].ToDbl()!=0.0)
        {
         history_candles[i].close=json["candles"][i]["close"].ToDbl();
         history_candles[i].high=json["candles"][i]["high"].ToDbl();
         history_candles[i].low=json["candles"][i]["low"].ToDbl();
         history_candles[i].open=json["candles"][i]["open"].ToDbl();
         history_candles[i].tick_volume=4;
         history_candles[i].real_volume=0;
         history_candles[i].spread=0;
         history_candles[i].time=(datetime)json["candles"][i]["epoch"].ToInt();
         i++;
        }

      if(ArraySize(history_candles)>0)
        {
         if(CustomRatesUpdate(m_symbol_name,history_candles)<0)
           {
            Print("Error adding history "+IntegerToString(::GetLastError()));
            return(false);
           }
        }
      else
        {
         Print("Received unexpected response from server ",IntegerToString(::GetLastError()), " "+history);
         return(false);
        }
     }
   else
     {
      Print("error reading "," error: ",websocket.LastError(), websocket.LastErrorMessage());
      return(false);
     }

   OpenChart();

   return(true);

  }
```

Since the history has been updated and the chart is open, the next step is to subcribe to a tick data stream from Binary.com. The StartTicksStream() method sends the corresponding query and if successful, the server will begin streaming live quotes that will be processed by the AddTick() method. The StopTicksStream() method on the other hand is used to notify the server to stop sending live quotes.

```
//+---------------------------------------------------------------------+
//|method that enables the reciept of new ticks as they become available|
//+---------------------------------------------------------------------+
bool CDerivSymbol::StartTicksStream(void)
  {
   Comment("Starting live ticks stream for "+m_symbol_name+".......");

   if(m_stream_id!="")
      StopTicksStream();

   json.Clear();
   json["subscribe"]=1;
   json["ticks"]=m_symbol_name;

   return(websocket.SendString(json.Serialize()));
  }

//+------------------------------------------------------------------+
//|Overridden method that handles new ticks streamed from Deriv.com |
//+------------------------------------------------------------------+
void CDerivSymbol::AddTick(void)
  {
   string str_tick;

   MqlTick current_tick[1];

   json.Clear();

   if(websocket.ReadString(str_tick))
     {
      json.Deserialize(str_tick);
      ZeroMemory(current_tick);

      if(CheckDerivError(json))
         return;

      if(!json["tick"]["ask"].ToDbl())
         return;

      current_tick[0].ask=json["tick"]["ask"].ToDbl();
      current_tick[0].bid=json["tick"]["bid"].ToDbl();
      current_tick[0].last=0;
      current_tick[0].time=(datetime)json["tick"]["epoch"].ToInt();
      current_tick[0].time_msc=(long)((json["tick"]["epoch"].ToInt())*1000);
      current_tick[0].volume=0;
      current_tick[0].volume_real=0;

      if(current_tick[0].ask)
         current_tick[0].flags|=TICK_FLAG_ASK;
      if(current_tick[0].bid)
         current_tick[0].flags|=TICK_FLAG_BID;

      if(m_stream_id==NULL)
         m_stream_id=json["tick"]["id"].ToStr();

      if(CustomTicksAdd(m_symbol_name,current_tick)<0)
        {
         Print("failed to add new tick ", ::GetLastError());
         return;
        }
      Comment("New ticks for  "+m_symbol_name+".......");
     }
   else
     {
      Print("read error ",websocket.LastError(), websocket.LastErrorMessage());

      websocket.ResetLastError();

      if(websocket.ClientState()!=CONNECTED && websocket.Connect(m_url))
        {
         if(m_stream_id!=NULL)
            if(StopTicksStream())
              {
               if(InitSymbol())
                  if(UpdateHistory())
                    {
                     StartTicksStream();
                     return;
                    }
              }
        }
     }
//---
  }
```

```
//+------------------------------------------------------------------+
//|Used to cancel all tick streams that may have been initiated      |
//+------------------------------------------------------------------+
bool CDerivSymbol::StopTicksStream(void)
  {

   json.Clear();
   json["forget_all"]="ticks";

   if(websocket.SendString(json.Serialize()))
     {
      m_stream_id=NULL;
      if(websocket.ReadString(m_stream_id)>0)
        {
         m_stream_id=NULL;
         Comment("Stopping live ticks stream for  "+m_symbol_name+".......");
         return(true);
        }
     }

   return(false);
  }
```

The code for the EA is shown below.

```
CDerivSymbol b_symbol;
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   b_symbol.SetAppID(DERIV_appid);
//---
   if(!b_symbol.Initialize(EnumToString(DERIV_symbol)))
      return(INIT_FAILED);
//---
   if(!b_symbol.UpdateHistory())
      return(INIT_FAILED);
//---
   if(!b_symbol.StartTicksStream())
      return(INIT_FAILED);
//--- create timer
   EventSetMillisecondTimer(500);
//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//--- destroy timer
   EventKillTimer();
//--- stop the ticks stream
   b_symbol.StopTicksStream();

  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//---

  }
//+------------------------------------------------------------------+
//| Timer function                                                   |
//+------------------------------------------------------------------+
void OnTimer()
  {
//---
   b_symbol.AddTick();
  }
//+------------------------------------------------------------------+
```

Both EAs have the same code with the exception of the UpdateHistory() method.

Running the EA results in the creation of a new custom symbol as shown here.

![EA demonstration](https://c.mql5.com/2/73/Deriv__1.gif)

### Conclusion

We explored how to use the Win32 API to create a WebSocket client for MetaTrader 5. We created a class that encapsulates this functionality and demonstrated its use in the EA that interacts with the Deriv.com WebSockets API.

| Folder | Contents | Description |
| --- | --- | --- |
| Mql5.zip\\Mql5\\include | JAson.mqh, websocket.mqh, winhttp.mqh | Include files contain the code for the JSON parser (CJAval class), WebSocket client (CWebsocket class), WinHttp imported function and type declarations respectively |
| --- | --- | --- |
| Mt5zip\\Mql5\\experts | DerivCustomSymboWithTickHistory.mq5, DerivCustomSymbolWithBarHistory.mq5 | Sample EAs that use the CWebsocket class to create custom symbols by leveraging the Deriv.com WebSocket API |
| --- | --- | --- |

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/10275.zip "Download all attachments in the single ZIP archive")

[Mql5.zip](https://www.mql5.com/en/articles/download/10275/mql5.zip "Download Mql5.zip")(23.14 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Building Volatility models in MQL5 (Part I): The Initial Implementation](https://www.mql5.com/en/articles/20589)
- [Bivariate Copulae in MQL5 (Part 2): Implementing Archimedean copulae in MQL5](https://www.mql5.com/en/articles/19931)
- [Bivariate Copulae in MQL5 (Part 1): Implementing Gaussian and Student's t-Copulae for Dependency Modeling](https://www.mql5.com/en/articles/18361)
- [Dynamic mode decomposition applied to univariate time series in MQL5](https://www.mql5.com/en/articles/19188)
- [Singular Spectrum Analysis in MQL5](https://www.mql5.com/en/articles/18777)
- [Websockets for MetaTrader 5: Asynchronous client connections with the Windows API](https://www.mql5.com/en/articles/17877)
- [Resampling techniques for prediction and classification assessment in MQL5](https://www.mql5.com/en/articles/17446)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/386894)**
(13)


![Kareem Abdelhakim](https://c.mql5.com/avatar/avatar_na2.png)

**[Kareem Abdelhakim](https://www.mql5.com/en/users/karimabdulhakim988)**
\|
5 Feb 2022 at 12:27

**Soewono Effendi [#](https://www.mql5.com/en/forum/386894#comment_27456856):**

Yeah, nice try :)

I was hoping this would be possible too, but if you search the forum, you'll find that function in MQL is a handle, not a memory address, which is required by "C/C++" callback-API.

Maybe someday MQL would add "real" Function Pointer.

yea i hope soon this will be natively supported

![Faisal Mahmood](https://c.mql5.com/avatar/2021/7/60F7782E-F345.jpg)

**[Faisal Mahmood](https://www.mql5.com/en/users/smartpips)**
\|
7 Feb 2022 at 02:26

[@Francis Dube](https://www.mql5.com/en/users/ufranco) Is it possible to create a MQL5 service that acts as the WebSocket server? Do you have some examples?

![Francis Dube](https://c.mql5.com/avatar/2014/8/53E01838-20C6.JPG)

**[Francis Dube](https://www.mql5.com/en/users/ufranco)**
\|
8 Feb 2022 at 13:22

**Faisal Mahmood [#](https://www.mql5.com/en/forum/386894#comment_27552319):**

[@Francis Dube](https://www.mql5.com/en/users/ufranco) Is it possible to create a MQL5 service that acts as the WebSocket server? Do you have some examples?

its a websocket client not server.

![dark_sam](https://c.mql5.com/avatar/avatar_na2.png)

**[dark\_sam](https://www.mql5.com/en/users/dark_sam)**
\|
21 Feb 2022 at 00:12

Hi , Itry it [on VPS](https://www.mql5.com/en/vps "Forex VPS for MetaTrader 4/5") not work and try it on normal local pc work good . are i need open some port on VPS ?


![Ruben Osvaldo Rodriguez](https://c.mql5.com/avatar/2022/7/62CF1C83-4097.jpg)

**[Ruben Osvaldo Rodriguez](https://www.mql5.com/en/users/rubenosvaldorodriguez)**
\|
7 Jun 2023 at 17:15

Hello Francisco. Super grateful for the valuable code provided. I have found that when requesting the [EURUSD symbol](https://www.mql5.com/en/quotes/currencies/eurusd "EURUSD chart: techical analysis"), that is: "frxEURUSD" to DERIV, it throws the following error:

\\* "Matching symbol name not found, Check the symbol name on Binary.com"

```
Any idea why DERIV doesn't seem to recognize the symbol we requested?

I got stuck in development/testing here.

Thank you so much. :)
```

![Learn Why and How to Design Your Algorithmic Trading System](https://c.mql5.com/2/44/why-and-how__1.png)[Learn Why and How to Design Your Algorithmic Trading System](https://www.mql5.com/en/articles/10293)

This article shows the basics of MQL for beginners to design their Algorithmic trading system (Expert Advisor) through designing a simple algorithmic trading system after mentioning some basics of MQL5

![Manual charting and trading toolkit (Part III). Optimization and new tools](https://c.mql5.com/2/43/MQL5-set_of_tools.png)[Manual charting and trading toolkit (Part III). Optimization and new tools](https://www.mql5.com/en/articles/9914)

In this article, we will further develop the idea of drawing graphical objects on charts using keyboard shortcuts. New tools have been added to the library, including a straight line plotted through arbitrary vertices, and a set of rectangles that enable the evaluation of the reversal time and level. Also, the article shows the possibility to optimize code for improved performance. The implementation example has been rewritten, allowing the use of Shortcuts alongside other trading programs. Required code knowledge level: slightly higher than a beginner.

![Graphics in DoEasy library (Part 89): Programming standard graphical objects. Basic functionality](https://c.mql5.com/2/44/MQL5-avatar-doeasy-library3-2__1.png)[Graphics in DoEasy library (Part 89): Programming standard graphical objects. Basic functionality](https://www.mql5.com/en/articles/10119)

Currently, the library is able to track standard graphical objects on the client terminal chart, including their removal and modification of some of their parameters. At the moment, it lacks the ability to create standard graphical objects from custom programs.

![MQL5 Cookbook – Economic Calendar](https://c.mql5.com/2/43/mql5-recipes_calendar-4.png)[MQL5 Cookbook – Economic Calendar](https://www.mql5.com/en/articles/9874)

The article highlights the programming features of the Economic Calendar and considers creating a class for a simplified access to the calendar properties and receiving event values. Developing an indicator using CFTC non-commercial net positions serves as a practical example.

[![](https://www.mql5.com/ff/sh/x8fwvn495ta7y774z2/01.png)Does your broker offer sponsored hosting for trading?Now it's even easier to get MetaTrader VPS for free – contact your broker for details](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=xscnzeyhifcgygpwvysykhqydcmmbgpp&s=f87b748147e376d34c8f0fdb9737b1766f20cc2174769a0e6b9975b5c2e8ddae&uid=&ref=https://www.mql5.com/en/articles/10275&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5068193969499731628)

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