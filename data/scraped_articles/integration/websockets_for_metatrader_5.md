---
title: Websockets for MetaTrader 5
url: https://www.mql5.com/en/articles/8196
categories: Integration
relevance_score: 3
scraped_at: 2026-01-23T21:13:40.427465
---

[What's wrong with regular VPS?Here are the 8 most common problems that algorithmic traders may encounterRead![](https://www.mql5.com/ff/sh/hzatb686qjqxwtr4z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/450486&a=drhremihlwuaqyvgpzfddbtmgciejpba&s=c37d25bcceb93ed153b814e6ba4d4839461a9b2d68dd82b95b142be06d310f3f&uid=&ref=https://www.mql5.com/en/articles/8196&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5071670908734614607)

MetaTrader 5 / Examples


### Introduction

MetaTrader 5 has  matured extensively over the years and provides wide ranging functionality for traders. A stand out feature , is its ability to integrate with various systems and platforms despite the use of a proprietary programming language. This ability is very important as it provides traders with lots of options when it comes to exploring potentially profitable trading strategies.

Key to this integration will be its ability to take advantage of modern networking protocols that are more efficient and easier to implement. It is in this vein that we will investigate implementation of a websocket client for MetaTrader 5 applications without the use of a dynamic link library.

To get started, a brief introduction to the websocket networking protocol.

### Introduction to Websockets

[Websocket](https://en.wikipedia.org/wiki/WebSocket "https://en.wikipedia.org/wiki/WebSocket") protocol which is a method of communication allowing for bi-directional flow of information between a server and client without the need to make multiple hyper text transfer protocol based requests. Browsers and most web interfacing applications use the Websocket protocol to provide various services like instant messaging, dynamic web content and online multiplayer gaming.

**Why the need for Websockets**

Before the existence of the websocket protocol developers had to employ inefficient and costly techniques to achieve asynchronous communication between a server and a client.

These included :

- [Polling](https://en.wikipedia.org/wiki/Polling_(computer_science) "https://en.wikipedia.org/wiki/Polling_(computer_science)")— this is an inherently synchronous method which involves making requests continuously even though there may be no data in need of transmission thereby leading to a waste of compute resourses.
- [Long polling](https://en.wikipedia.org/wiki/Push_technology#Long_polling "https://en.wikipedia.org/wiki/Push_technology#Long_polling") — similar to polling, the difference with this technique is as its name implies. Instead of making frequent requests, a client makes relatively fewer requests to a server that should respond by opening and keeping a connection active until there is some exchange or a timeout comes into effect.

- [Streaming](https://en.wikipedia.org/wiki/Streaming_media "https://en.wikipedia.org/wiki/Streaming_media")  — This method requires a client to make a request for data and the server would then keep the connection alive indefinitely. The main drawback here is the extensive use of HTTP headers that increase the size of the data being retrieved.

- [Ajax](https://en.wikipedia.org/wiki/Ajax_(programming) "https://en.wikipedia.org/wiki/Ajax_(programming)") \- primarily being a browser technology asynchronous javascript and xml ushered in asychronous web content. You could make a post to a website and have the content appear on the webpage almost instantly without the need to refresh the entire webpage.


All the methods described above allowed varying levels of bidirectional data exchange between client and server but, relative to websockets, suffer, because of three main reasons:

- as previously mentioned , the techniques above  provide varying levels of asynchronous transmission, in fact the communication type can be described as being half duplex at best. That means each participant in an exchange has to wait for the other to finnish before a response can be offered.

- the methods above employ extensive use of http headers. Combine this with the frequency of the http requests some times needed, leads to relatively excessive data usage. Which can be a disadvantage when efficient bandwith use is important.

- also related to efficiency is cost. Keeping server connections alive for long periods when they donot need to be or sending data transmissions to clients that may already have navigated away is a waste for big business as it drives up the cost of running servers.

### Features of websockets

A websocket is a TCP based protocol that can be further expanded to support other application or industry defined subprotocols. Since it is TCP based it can work over standard HTTP ports 80 , 443 and has a similar universal resource locator schema. Websocket server addresses are prefixed with ws or wss as opposed to http, but follow the same url address structure as an Http web address. for example:

**ws(s)://websocketexampleurl.com:80/hello.php**

### Understanding Websockets

In order to understand how a websocket client can be implemented in [Mql5](https://www.mql5.com/en/docs/network), it is necessary to be familiar with the fundamentals of general computer networking. The Websocket protocol is  similar to  the hyper text transfer protocol, , where headers are used in client requests to the server. Just like the hyper text transfer protocol, establishing a websocket defined connection requires the use of headers as well. The main difference with websockets is that such a request is only needed to establish or initialize the websocket. A client makes what looks like a normal  hyper text transfer protocol request, then the protocols used will switch from using the  hyper text transfer protocol, to Websocket protocol.

This process is called the websocket handshake. The switch in protocols is done only if the initial hyper text transfer protocol request to the server contains a specific header or headers. The server must then respond accordingly by affirming the desire to establish a websocket connection. Information about the nature of the special headers and how the server may respond is all documented in [RFC 6455](https://www.mql5.com/go?link=https://www.rfc-editor.org/rfc/rfc6455%23section-4 "https://tools.ietf.org/html/rfc6455#section-4") [https://tools.ietf.org/html/rfc6455#section-4](https://www.mql5.com/go?link=https://www.rfc-editor.org/rfc/rfc6455%23section-4 "https://tools.ietf.org/html/rfc6455#section-4").

Once the websocket is established there is no need to use hyper text transfer protocol like requests anymore, this is where the protocols diverge in terms of their operation. A different  format is adopted when exchanging data using the websocket protocol. This format is more streamlined and uses  much less raw bits relative to a hyper text transfer protocol request. The format used is reffered to as the framing protocol, where data exchanged in one transaction  between  hosts is called a frame.

Each frame is a sequence of bits arranged in a specific manner that conforms to the framing protocol as stipulated in [RFC 6455](https://tools.ietf.org/html/rfc6455#section-5.2). Every websocket frame contains bits that define an opcode , the size of the payload and the actual payload itself. The protocol also defines how those bits are arranged and ultimately packaged within the frame. An opcode is simply a reserved numerical value used to classify a frame.

For websockets the base opcodes are defined as follows:

**0 — continuation frame**: this value denotes payload data that is incomplete, therefore more frames should be  expected. This feature enables frame fragmentation. It allows for data to be split into chunks that are packaged in different frames.

**1 — text frame**: this value indicates that the payload data is textual in nature.

**2 — binary frame** : with this value the payload is in binary form.

**8 — close frame:** this value denotes a special type of frame that is sent when either endpoint intends to close an established websocket connection, it is a frame type called a control frame. Control frames already have  special meaning so they may not always contain any payload data, ie the payload is optional.

**9 — ping frame**: another control frame used to determine whether an endpoint is still connected or not

**10 — pong frame**: the pong frame is used as a response whenever an endpoint recieves a ping frame. In such a situation the recepient must respond as soon as possible with an appropriate pong frame. Usually it is adequate to echo whatever payload was contained in the ping frame.

These are the only  base opcodes that should be supported by any websocket. The protocol allows for websocket based API's or websocket subprotocols to expand on these reserved values.

The last important aspect about frames is masking. RFC 6455 requires that all frames sent from a client to a server be masked. Masking serves as a basic security measure for the websocket protocol. It entails mangling the payload with a randomly generated 4 byte value called a key , using a predefined algorithm. You could think of it as a kind of data obfuscation. The algorithm is documented in the [RFC 6455](https://www.mql5.com/go?link=https://www.rfc-editor.org/rfc/rfc6455%23section-5.3 "https://tools.ietf.org/html/rfc6455#section-5.3") document. Every frame sent from the client must use a freshly generated, random key , even for fragmented frames.

This section provided brief details of the important characteristics of the websocket protocol. For more indepth information, all details can be found in the RFC6455 documentation. Armed with this knowledge i think understanding the code implementation will be a lot easier.

### Mql5 websocket client — library overview

To begin the code will be split into three classes.

**CSocket** \- encapsulates the [networking](https://www.mql5.com/en/docs/network) functions of Mql5 api.

**CFrame**  — the frame class represents a websocket frame and will be used primarily to decode frames received from a server.

**CWebSocketClient** — represents the websocket client itself

### CSocket

```
//+------------------------------------------------------------------+
//| structs                                                          |
//+------------------------------------------------------------------+
struct CERT
  {
   string            cert_subject;
   string            cert_issuer;
   string            cert_serial;
   string            cert_thumbprint;
   datetime          cert_expiry;
  };

//+------------------------------------------------------------------+
//| Class CSocket.                                                   |
//| Purpose: Base class of socket operations.                        |
//|                                                                  |
//+------------------------------------------------------------------+

class CSocket
  {
private:
   static int        m_usedsockets;   // tracks number of sockets in use in single program
   bool              m_log;           // logging state
   bool              m_usetls;        //  tls state
   uint              m_tx_timeout;    //  send system socket timeout in milliseconds
   uint              m_rx_timeout;    //  receive system socket timeout in milliseconds
   int               m_socket;        //  socket handle
   string            m_address;       //  server address
   uint              m_port;          //  port

   CERT              m_cert;          //  Server certificate info

public:
                     CSocket();
                    ~CSocket();
   //--- methods to get private properties
   int               SocketID(void)           const { return(m_socket); }
   string            Address(void)            const { return(m_address);   }
   uint              Port(void)               const { return(m_port);  }
   bool              IsSecure(void)           const { return(m_usetls); }
   uint              RxTimeout(void)          const { return(m_rx_timeout); }
   uint              TxTimeout(void)          const { return(m_tx_timeout); }
   bool              ServerCertificate(CERT& certificate);

   //--- methods to set private properties
   bool              SetTimeouts(uint tx_timeout, uint rx_timeout);
   //--- general methods for working sockets
   void              Log(const string custom_message,const int line,const string func);
   static uint       SocketsInUse(void)        {   return(m_usedsockets);  }
   bool              Open(const string server,uint port,uint timeout,bool use_tls=false,bool enablelog=false);
   bool              Close(void);
   uint              Readable(void);
   bool              Writable(void);
   bool              IsConnected(void);
   int               Read(uchar& out[],uint out_len,uint ms_timeout,bool read_available);
   int               Send(uchar& in[],uint in_len);

  };

int CSocket::m_usedsockets=0;
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CSocket::CSocket():m_socket(INVALID_HANDLE),
   m_address(""),
   m_port(0),
   m_usetls(false),
   m_log(false),
   m_rx_timeout(150),
   m_tx_timeout(150)
  {
   ZeroMemory(m_cert);
  }
//+------------------------------------------------------------------+
//| Destructor                                                       |
//+------------------------------------------------------------------+
CSocket::~CSocket()
  {
//--- check handle
   if(m_socket!=INVALID_HANDLE)
      Close();
  }
//+------------------------------------------------------------------+
//| set system socket timeouts                                       |
//+------------------------------------------------------------------+
bool CSocket::SetTimeouts(uint tx_timeout,uint rx_timeout)
  {
   if(m_socket==INVALID_HANDLE)
     {
      Log("Invalid socket",__LINE__,__FUNCTION__);
      return(false);
     }

   if(SocketTimeouts(m_socket,tx_timeout,rx_timeout))
     {
      m_tx_timeout=tx_timeout;
      m_rx_timeout=rx_timeout;
      Log("Socket Timeouts set",__LINE__,__FUNCTION__);
      return(true);
     }

   return(false);
  }

//+------------------------------------------------------------------+
//| certificate                                                      |
//+------------------------------------------------------------------+
bool CSocket::ServerCertificate(CERT& certificate)
  {

   if(m_socket==INVALID_HANDLE)
     {
      Log("Invalid socket",__LINE__,__FUNCTION__);
      return(false);
     }

   if(SocketTlsCertificate(m_socket,m_cert.cert_subject,m_cert.cert_issuer,m_cert.cert_serial,m_cert.cert_thumbprint,m_cert.cert_expiry))
     {
      certificate=m_cert;
      Log("Server certificate retrieved",__LINE__,__FUNCTION__);
      return(true);
     }

   return(false);

  }
//+------------------------------------------------------------------+
//|connect()                                                         |
//+------------------------------------------------------------------+
bool CSocket::Open(const string server,uint port,uint timeout,bool use_tls=false,bool enablelog=false)
  {
   if(m_socket!=INVALID_HANDLE)
      Close();

   if(m_usedsockets>=128)
     {
      Log("Too many sockets open",__LINE__,__FUNCTION__);
      return(false);
     }

   m_usetls=use_tls;

   m_log=enablelog;

   m_socket=SocketCreate();
   if(m_socket==INVALID_HANDLE)
     {
      Log("Invalid socket",__LINE__,__FUNCTION__);
      return(false);
     }
   ++m_usedsockets;
   m_address=server;

   if(port==0)
     {
      if(m_usetls)
         m_port=443;
      else
         m_port=80;
     }
   else
      m_port=port;
//---
   if(!m_usetls && m_port==443)
      m_usetls=true;
//---
   Log("Connecting to "+m_address,__LINE__,__FUNCTION__);
//---
   if(m_usetls)
     {
      if(m_port!=443)
        {
         if(SocketConnect(m_socket,server,port,timeout))
            return(SocketTlsHandshake(m_socket,server));
        }
      else
        {
         return(SocketConnect(m_socket,server,port,timeout));
        }
     }

   return(SocketConnect(m_socket,server,port,timeout));
  }
//+------------------------------------------------------------------+
//|close()                                                           |
//+------------------------------------------------------------------+
bool CSocket::Close(void)
  {
//---
   if(m_socket==INVALID_HANDLE)
     {
      Log("Socket Disconnected",__LINE__,__FUNCTION__);
      return(true);
     }
//---
   if(SocketClose(m_socket))
     {
      m_socket=INVALID_HANDLE;
      --m_usedsockets;
      Log("Socket Disconnected from "+m_address,__LINE__,__FUNCTION__);
      m_address="";
      ZeroMemory(m_cert);
      return(true);
     }
//---
   Log("",__LINE__,__FUNCTION__);
   return(false);
  }
//+------------------------------------------------------------------+
//|readable()                                                        |
//+------------------------------------------------------------------+
uint CSocket::Readable(void)
  {
   if(m_socket==INVALID_HANDLE)
     {
      Log("Invalid socket",__LINE__,__FUNCTION__);
      return(0);
     }
//---
   Log("Is Socket Readable ",__LINE__,__FUNCTION__);
//---
   return(SocketIsReadable(m_socket));
  }
//+------------------------------------------------------------------+
//|writable()                                                        |
//+------------------------------------------------------------------+
bool CSocket::Writable(void)
  {
   if(m_socket==INVALID_HANDLE)
     {
      Log("Invalid socket",__LINE__,__FUNCTION__);
      return(false);
     }
//---
   Log("Is Socket Writable ",__LINE__,__FUNCTION__);
//---
   return(SocketIsWritable(m_socket));
  }
//+------------------------------------------------------------------+
//|isconnected()                                                     |
//+------------------------------------------------------------------+
bool CSocket::IsConnected(void)
  {
   if(m_socket==INVALID_HANDLE)
     {
      Log("Invalid socket",__LINE__,__FUNCTION__);
      return(false);
     }
//---
   Log("Is Socket Connected ",__LINE__,__FUNCTION__);
//---
   return(SocketIsConnected(m_socket));
  }
//+------------------------------------------------------------------+
//|read()                                                            |
//+------------------------------------------------------------------+
int CSocket::Read(uchar& out[],uint out_len,uint ms_timeout,bool read_available=false)
  {
   if(m_socket==INVALID_HANDLE)
     {
      Log("Invalid socket",__LINE__,__FUNCTION__);
      return(-1);
     }
//---
   Log("Reading from "+m_address,__LINE__,__FUNCTION__);

   if(m_usetls)
     {
      if(read_available)
         return(SocketTlsReadAvailable(m_socket,out,out_len));
      else
         return(SocketTlsRead(m_socket,out,out_len));
     }
   else
      return(SocketRead(m_socket,out,out_len,ms_timeout));

   return(-1);
  }
//+------------------------------------------------------------------+
//|send()                                                            |
//+------------------------------------------------------------------+
int CSocket::Send(uchar& in[],uint in_len)
  {
   if(m_socket==INVALID_HANDLE)
     {
      Log("Invalid socket",__LINE__,__FUNCTION__);
      return(-1);
     }
//---
   Log("Sending to "+m_address,__LINE__,__FUNCTION__);
//---
   if(m_usetls)
      return(SocketTlsSend(m_socket,in,in_len));
   else
      return(SocketSend(m_socket,in,in_len));
//---
   return(-1);
  }
//+------------------------------------------------------------------+
//|log()                                                             |
//+------------------------------------------------------------------+
void CSocket::Log(const string custom_message,const int line,const string func)
  {
   if(m_log)
     {
      //---
      int eid=GetLastError();
      //---
      if(eid!=0)
        {
         PrintFormat("[MQL error ID: %d][%s][Line: %d][Function: %s]",eid,custom_message,line,func);
         ResetLastError();
         return;
        }
      if(custom_message!="")
         PrintFormat("[%s][Line: %d][Function: %s]",custom_message,line,func);
     }
//---
  }
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
```

The socket class defines struct CERT which capsulizes server certificate information.

-  cert\_subject  — Certificate owner name
-  cert\_issuer  — Certificate issuer name
-  cert\_serial  — Certificate serial number
-  cert\_thumbprint  — Certificate SHA-1 hash

-  cert\_expiry  — Certificate expiration date

Methods to get private properties:

> SocketID  — returns the socket handle for a successfully created socket.
>
> Address  — returns the remote address the socket is connected to as a string
>
> Port  — returns the remote port an active socket is connected to
>
> IsSecure  — returns true or false depending on whether the socket has TLS security enabled or not.
>
> RxTimeout  — returns set timeout in milliseconds for reading from a socket.
>
> TxTimeout  — return set timeout in milliseconds for writing to a socket
>
> ServerCertificate  — returns the server certificate information for a server the socket is connected to.
>
> SocketsInUse  — returns the total number of sockets currently in use in a single program.

Methods to set private properties.

> SetTimeouts — set the timeouts in milliseconds for reading and writing to a socket.
>
> general methods for working sockets
>
> Log — utility method for logging the activities of a socket.To output messages to the terminal's journal logging   must be set when initializing a socket with the Open method.
>
> Open — method for establishing a connection to a remote server thereby creating a new socket. The method
>
> Close - method for disconnecting from a remote server and deinitilizing a socket.
>
> Readable — returns the number of bytes that available for reading on a socket
>
> Writable — queries whether a socket is available for any sending operations.
>
> IsConnected — checks if a socket connection is still active.
>
> Read — Reads data from a socket
>
> Send — method for performing send operations on an active socket.

### CFrame

```
//+------------------------------------------------------------------+
//| enums                                                            |
//+------------------------------------------------------------------+
enum ENUM_FRAME_TYPE     // type of websocket frames (ie, message types)
  {
   CONTINUATION_FRAME=0x0,
   TEXT_FRAME=0x1,
   BINARY_FRAME= 0x2,
   CLOSE_FRAME = 8,
   PING_FRAME = 9,
   PONG_FRAME = 0xa,
  };
//+------------------------------------------------------------------+
//| class frame                                                      |
//| represents a websocket message frame                             |
//+------------------------------------------------------------------+

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CFrame
  {
private:
   uchar             m_array[];
   uchar             m_isfinal;
   ENUM_FRAME_TYPE   m_msgtype;

   int               Resize(int size) {return(ArrayResize(m_array,size,size));}

public:
                     CFrame():m_isfinal(0),m_msgtype(0) {   }

                    ~CFrame() {      }
   int               Size(void) {return(ArraySize(m_array));}
   bool              Add(const uchar value);
   bool              Fill(const uchar &array[],const int src_start,const int count);
   void              Reset(void);
   uchar             operator[](int index);
   string            ToString(void);
   ENUM_FRAME_TYPE   MessageType(void) { return(m_msgtype);}
   bool              IsFinal(void) { return(m_isfinal==1);}
   void              SetMessageType(ENUM_FRAME_TYPE mtype) { m_msgtype=mtype;}
   void              SetFinal(void) { m_isfinal=1;}

  };
//+------------------------------------------------------------------+
//| Receiving an element by index                                    |
//+------------------------------------------------------------------+

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
uchar CFrame::operator[](int index)
  {
   static uchar invalid_value;
//---
   int max=ArraySize(m_array)-1;
   if(index<0 || index>=ArraySize(m_array))
     {
      PrintFormat("%s index %d is not in range (0-%d)!",__FUNCTION__,index,max);
      return(invalid_value);
     }
//---
   return(m_array[index]);
  }
//+------------------------------------------------------------------+
//| Adding element                                                   |
//+------------------------------------------------------------------+

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CFrame::Fill(const uchar &array[],const int src_start,const int count)
  {
   int p_size=Size();
//---
   int size=Resize(p_size+count);
//---
   if(size>0)
      return(ArrayCopy(m_array,array,p_size,src_start,count)==count);
   else
      return(false);
//---
  }
//+------------------------------------------------------------------+
//| Assigning for the array                                          |
//+------------------------------------------------------------------+

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CFrame::Add(const uchar value)
  {
   int size=Resize(Size()+1);
//---
   if(size>0)
      m_array[size-1]=value;
   else
      return(false);
//---
   return(true);
//---
  }
//+------------------------------------------------------------------+
//|  Reset                                                           |
//+------------------------------------------------------------------+

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CFrame::Reset(void)
  {
   if(Size())
      ArrayFree(m_array);
//---

   m_isfinal=0;

   m_msgtype=0;

  }
//+------------------------------------------------------------------+
//|converting array to string                                        |
//+------------------------------------------------------------------+
string CFrame::ToString(void)
  {
   if(Size())
      if(m_msgtype==CLOSE_FRAME)
         return(CharArrayToString(m_array,2,WHOLE_ARRAY,CP_UTF8));
   else
      return(CharArrayToString(m_array,0,WHOLE_ARRAY,CP_UTF8));
   else
      return(NULL);
  }
```

The frame class defines the enumeration ENUM\_FRAME\_TYPE which describes the different frame types as documented by the websocket protocol.

Instances of the CFrame class represent a single frame recieved from the server. That means that a complete message could be made up of a collection of frames . The class enables funtionality to query the various characteristics of each frame, including the individual byte values that make up a frame.

The Size method returns the size in bytes of a frame. Since the class uses an array of type unsigned character as a container for  a frame . This method simply returns the size of the underlying array.

MessageType method returns the type of frame as type ENUM\_FRAME\_TYPE

IsFinal method is to check if the frame is the last or the final frame, which means what ever data received should be assumed to be whole, this allows one to differentiate between a fragmented and  therefore incomplete message and one that is complete.

operator\[\] - subscript operator overload allows for individual retrieval of any element in the frame in array format.

The CFrame class will be used in websocket client as it reads from a CSocket object. The methods used to fill a frame are Add and Fill. Which allow filling of a frame either by an individual element or using an appropriate array.

The utility method Reset can be used to flush a frame and reset its properties, whilst the ToString method is a handy tool for converting the frame contents into a familiar string value.

### CWebSocketClient

The class has constants that are implemented as #defines. The HEADER prefixed symbols are associated with the http header fields needed to create the opening handshake . GUID is a globally unique identifier used by the websocket protocol on the server side when  generating part of the response headers . Our class uses it to confirm and also demonstrate the correctness of the handshake process, but in essence, is unnecessary, the client only needs to check for the existence of the \|Sec-WebSocket-Accept\| header field to confirm a successful handshake.

```
#include <Socket.mqh>
#include <Frame.mqh>

//+------------------------------------------------------------------+
//| defines                                                          |
//+------------------------------------------------------------------+
#define SH1                 "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
#define HEADER_EOL          "\r\n"
#define HEADER_GET          "GET /"
#define HEADER_HOST         "Host: "
#define HEADER_UPGRADE      "Upgrade: websocket"+HEADER_EOL
#define HEADER_CONNECTION   "Connection: Upgrade"+HEADER_EOL
#define HEADER_KEY          "Sec-WebSocket-Key: "
#define HEADER_WS_VERSION   "Sec-WebSocket-Version: 13"+HEADER_EOL+HEADER_EOL
#define HEADER_HTTP         " HTTP/1.1"
```

The enum type ENUM\_STATUS\_CLOSE\_CODE , lists the close codes that one can send or receive along with a close frame. Whilst enum  ENUM\_WEBSOCKET\_CLIENT\_STATE, symbolises the  different states that the websocket can take.

Closed is the initial state before any socket is allocated for the client, or after a client has dropped a connection and the underlying socket has be closed.

When an initial  connection is made before sending the opening handshake ( header ) , the client is said to be in a connecting state. Once the opening handshake has been sent and a response is received permiting the use of the websocket protocol, then the client is connected.

The closing state manifests when either the client receives a close frame for the first time since client initialization or the client sends the first close frame to notify the server that it is dropping the connection. In a closing state, the client can only send close frames to the server, any attempt to send any other type of frame will fail. Just remember that in a closing state the server may not respond, as it is not obligated to continue serving once it has either sent or received a close notification.

```
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
enum ENUM_CLOSE_CODE                 // possible reasons for disconnecting sent with a close frame
  {
   NORMAL_CLOSE = 1000,            // normal closure initiated by choice
   GOING_AWAY_CLOSE,               // close code for client navigating away from end point, used in browsers
   PROTOCOL_ERROR_CLOSE,           // close caused by some violation of a protocol, usually application defined
   FRAME_TYPE_ERROR_CLOSE,         // close caused by an endpoint receiving frame type that is not supportted or allowed
   UNDEFINED_CLOSE_1,              // close code is not defined by websocket protocol
   UNUSED_CLOSE_1,                 // unused
   UNUSED_CLOSE_2,                 // values
   ENCODING_TYPE_ERROR_CLOSE,      // close caused data in message is of wrong encoding type, usually referring to strings
   APP_POLICY_ERROR_CLOSE,         // close caused by violation of user policy
   MESSAGE_SIZE_ERROR_CLOSE,       // close caused by endpoint receiving message that is too large
   EXTENSION_ERROR_CLOSE,          // close caused by non compliance to or no support for specified extension of websocket protocol
   SERVER_SIDE_ERROR_CLOSE,        // close caused by some error that occurred on the server
   UNUSED_CLOSE_3 = 1015,          // unused
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
enum ENUM_WEBSOCKET_STATE
  {
   CLOSED=0,
   CLOSING,
   CONNECTING,
   CONNECTED
  };
```

The ClientState method retrieves the property defining the connection state of any given websocket client.

```
//+------------------------------------------------------------------+
//| ClientState()                                                    |
//+------------------------------------------------------------------+
ENUM_WEBSOCKET_STATE CWebSocketClient::ClientState(void)
  {
   if(m_socket.IsConnected())
      return(m_wsclient);
//---
   if(m_wsclient!=CLOSED)
     {
      m_socket.Close();
      m_wsclient=CLOSED;
     }
//---
   return(m_wsclient);
  }
```

SetMaxSendSize () is used to configure the frame fragmentation characteristic of the websocket client. This method sets the maximum size in bytes for a single frame sent from the client to the server. Making the client flexible for use with any API that enforces frame size limits.

```
void              SetMaxSendSize(int maxsend) {if(maxsend>=0) m_maxsendsize=maxsend;  else m_maxsendsize=0; }
```

The Connect method is used to establish a websocket connection. The secure parameter is a boolean for configuring the websocket with TLS or not. The method first calls the  open method of the CSocket class to establish an initial TCP connection. On success the state of websocket changes to connecting, after which the upgrade helper method comes into play. Its responsibility is the creation of the required Http header for switching to the websocket protocol. Finally the state of the websocket is checked on function exit.

```
//+------------------------------------------------------------------+
//| Connect(): Used to establish connection  to websocket server     |
//+------------------------------------------------------------------+
bool CWebSocketClient::Connect(const string url,const uint port,const uint timeout,bool use_tls=false,bool enablelog=false)
  {
   reset();
//---
   m_timeout=timeout;
//---
   if(!m_socket.Open(url,port,m_timeout,use_tls,enablelog))
     {
      m_socket.Log("Connect error",__LINE__,__FUNCTION__);
      return(false);
     }
   else
      m_wsclient=CONNECTING;
//---
   if(!upgrade())
      return(false);
//---
   m_socket.Log("ws client state "+EnumToString(m_wsclient),__LINE__,__FUNCTION__);
//---
   if(m_wsclient!=CONNECTED)
     {
      m_wsclient=CLOSED;
      m_socket.Close();
      reset();
     }
//---
   return(m_wsclient==CONNECTED);
  }
```

For closing or dropping a connection the ClientClose method is used. It has two default parameters, the close code and a message body that will be sent as a close frame to the server. The message body will be truncated if it is larger than the 122 character limit. According to the websocket specification if either endpoint (server or client)  receives a close frame ( for the first time) the recipient should respond and the sender should expect a response as acknowledgement of the close request. As can be seen from the clientClose code, once the close frame is sent the underlying TCP socket is closed without waiting for a response, even if closure was initiated by the client. Waiting for a response at this juncture of  client's life cycle seems like a waste of resources, so it was not implemented.

```
//+------------------------------------------------------------------+
//| Close() inform server client is disconnecting                    |
//+------------------------------------------------------------------+
bool CWebSocketClient::Close(ENUM_CLOSE_CODE close_code=NORMAL_CLOSE,const string close_reason="")
  {
   ClientState();
//---
   if(m_wsclient==0)
     {
      m_socket.Log("Client Disconnected",__LINE__,__FUNCTION__);
      //---
      return(true);
     }
//---
   if(ArraySize(m_txbuf)<=0)
     {
      if(close_reason!="")
        {
         int len=StringToCharArray(close_reason,m_txbuf,2,120,CP_UTF8)-1;
         if(len<=0)
            return(false);
         else
            ArrayRemove(m_txbuf,len,1);
        }
      else
        {
         if(ArrayResize(m_txbuf,2)<=0)
           {
            m_socket.Log("array resize error",__LINE__,__FUNCTION__);
            return(false);
           }
        }
      m_txbuf[0]=(uchar)(close_code>>8) & 0xff;
      m_txbuf[1]=(uchar)(close_code>>0) & 0xff;
      //---
     }
//---
   m_msgsize=ArraySize(m_txbuf);
   m_sent=false;
//---
   send(CLOSE_FRAME);
//---
   m_socket.Close();
//---
   reset();
//---
   return(true);
//---
  }
```

When sending arbitrary data to a server there is a choice of two methods that can be used. SendString takes a string and SendData takes an array as input.

SendPing and SendPong are special methods for sending pings and pongs. Both allow for an optional message body on which the 122 character limit applies.

All the public send methods package thier respective inputs into the m\_txbuff array. The private send method sets the type of frame and uses filltxbuffer() to enable message fragmentation depending on value the of m\_maxsendsize property. FillTxbuffer() prepares a single frame, packaging it into array m\_send. Once m\_send is prepared it is sent to the server. All this is done in a loop until all the contents of m\_txbuffer have been sent.

```
//+------------------------------------------------------------------+
//| Send() sends text data to websocket server                       |
//+------------------------------------------------------------------+
int CWebSocketClient::SendString(const string message)
  {
   ClientState();
//---
   if(m_wsclient==CLOSED || m_wsclient==CLOSING)
     {
      m_socket.Log("invalid ws client handle",__LINE__,__FUNCTION__);
      return(0);
     }
//---
   if(message=="")
     {
      m_socket.Log("no message specified",__LINE__,__FUNCTION__);
      return(0);
     }
//---
   int len=StringToCharArray(message,m_txbuf,0,WHOLE_ARRAY,CP_UTF8)-1;
   if(len<=0)
     {
      m_socket.Log("string char array error",__LINE__,__FUNCTION__);
      return(0);
     }
   else
      ArrayRemove(m_txbuf,len,1);
//---
   m_msgsize=ArraySize(m_txbuf);
   m_sent=false;
//---
   return(send(TEXT_FRAME));
  }
//+------------------------------------------------------------------+
//| Send() sends user supplied array buffer                          |
//+------------------------------------------------------------------+
int CWebSocketClient::SendData(uchar &message_buffer[])
  {
   ClientState();
//---
   if(m_wsclient==CLOSED || m_wsclient==CLOSING)
     {
      m_socket.Log("invalid ws client handle",__LINE__,__FUNCTION__);
      return(0);
     }
//---
   if(ArraySize(message_buffer)==0)
     {
      m_socket.Log("array is empty",__LINE__,__FUNCTION__);
      return(0);
     }
//---
   if(ArrayResize(m_txbuf,ArraySize(message_buffer))<0)
     {
      m_socket.Log("array resize error",__LINE__,__FUNCTION__);
      return(0);
     }
   else
      ArrayCopy(m_txbuf,message_buffer);
//---
   m_msgsize=ArraySize(m_txbuf);
   m_sent=false;
//---
   return(send(BINARY_FRAME));
  }
//+------------------------------------------------------------------+
//| SendPong() sends pong response upon receiving ping               |
//+------------------------------------------------------------------+
int CWebSocketClient::SendPong(const string msg="")
  {
   ClientState();
//---
   if(m_wsclient==CLOSED || m_wsclient==CLOSING)
     {
      m_socket.Log("invalid ws client handle",__LINE__,__FUNCTION__);
      return(0);
     }
//---
   if(ArraySize(m_txbuf)<=0)
     {
      if(msg!="")
        {
         int len=StringToCharArray(msg,m_txbuf,0,122,CP_UTF8)-1;
         if(len<=0)
           {
            m_socket.Log("string to char array error",__LINE__,__FUNCTION__);
            return(0);
           }
         else
            ArrayRemove(m_txbuf,len,1);
        }
     }
//---
   m_msgsize=ArraySize(m_txbuf);
   m_sent=false;
//---
   return(send(PONG_FRAME));
  }
//+------------------------------------------------------------------+
//| SendPing() ping  the server                                      |
//+------------------------------------------------------------------+
int CWebSocketClient::SendPing(const string msg="")
  {
   ClientState();
//---
   if(m_wsclient==CLOSED || m_wsclient==CLOSING)
     {
      m_socket.Log("invalid ws client handle",__LINE__,__FUNCTION__);
      return(0);
     }
//---
   if(ArraySize(m_txbuf)<=0)
     {
      if(msg!="")
        {
         int len=StringToCharArray(msg,m_txbuf,0,122,CP_UTF8)-1;
         if(len<=0)
           {
            m_socket.Log("string to char array error",__LINE__,__FUNCTION__);
            return(0);
           }
         else
            ArrayRemove(m_txbuf,len,1);
        }
     }
//---
   m_msgsize=ArraySize(m_txbuf);
   m_sent=false;
//---
   return(send(PING_FRAME));
  }
```

```
//+------------------------------------------------------------------+
//|prepareSendBuffer()prepares array buffer for socket dispatch      |
//+------------------------------------------------------------------+
bool CWebSocketClient::fillTxBuffer(ENUM_FRAME_TYPE ftype)
  {
   uchar header[];
   static int it;
   static int start;
   uchar masking_key[4]={0};
   int maxsend=(m_maxsendsize<7)?m_msgsize:((m_maxsendsize<126)?m_maxsendsize-6:((m_maxsendsize<65536)?m_maxsendsize-8:m_maxsendsize-14));
//---
   for(int i=0; i<4; i++)
     {
      masking_key[i]=(uchar)(255*MathRand()/32767);
     }
//---
   m_socket.Log("[send]max size - "+IntegerToString(maxsend),__LINE__,__FUNCTION__);
   m_socket.Log("[send]should be max size - "+IntegerToString(m_maxsendsize),__LINE__,__FUNCTION__);
   int message_size=(((start+maxsend)-1)<=(m_msgsize-1))?maxsend:m_msgsize%maxsend;
   bool isfinal=((((start+maxsend)-1)==(m_msgsize-1)) || (message_size<maxsend) ||(message_size<=0))?true:false;
   bool isfirst=(start==0)?true:false;
//---
   m_socket.Log("[send]message size - "+IntegerToString(message_size),__LINE__,__FUNCTION__);
   if(isfirst)
      m_socket.Log("[send]first frame",__LINE__,__FUNCTION__);
   if(isfinal)
      m_socket.Log("[send]final frame",__LINE__,__FUNCTION__);
//---
   if(ArrayResize(header,2+(message_size>=126 ? 2 : 0)+(message_size>=65536 ? 6 : 0)+(4))<0)
     {
      m_socket.Log("array resize error",__LINE__,__FUNCTION__);
      return(false);
     }
//header[0] = (isfinal)? (0x80 | 0x1) :( );
   switch(ftype)
     {
      case CLOSE_FRAME:
         header[0]=uchar(0x80|CLOSE_FRAME);
         m_socket.Log("[building]close frame",__LINE__,__FUNCTION__);
         break;
      case PING_FRAME:
         header[0]=uchar(0x80|PING_FRAME);
         m_socket.Log("[building]ping frame",__LINE__,__FUNCTION__);
         break;
      case PONG_FRAME:
         header[0]=uchar(0x80|PONG_FRAME);
         m_socket.Log("[building]pong frame",__LINE__,__FUNCTION__);
         break;
      default:
         header[0]=(isfinal)? 0x80:0x0;
         m_socket.Log("[building]"+EnumToString(ftype),__LINE__,__FUNCTION__);
         if(isfirst)
            header[0]|=uchar(ftype);
         break;

     }
//---
   if(message_size<126)
     {
      header[1] = (uchar)(message_size & 0xff) |  0x80;
      header[2] = masking_key[0];
      header[3] = masking_key[1];
      header[4] = masking_key[2];
      header[5] = masking_key[3];
     }
   else
   if(message_size<65536)
     {
      header[1] = 126 |  0x80;
      header[2] = (uchar)(message_size >> 8) & 0xff;
      header[3] = (uchar)(message_size >> 0) & 0xff;
      header[4] = masking_key[0];
      header[5] = masking_key[1];
      header[6] = masking_key[2];
      header[7] = masking_key[3];
     }
   else
     {
      header[1] = 127 | 0x80;
      header[2] = (uchar)(message_size >> 56) & 0xff;
      header[3] = (uchar)(message_size >> 48) & 0xff;
      header[4] = (uchar)(message_size >> 40) & 0xff;
      header[5] = (uchar)(message_size >> 32) & 0xff;
      header[6] = (uchar)(message_size >> 24) & 0xff;
      header[7] = (uchar)(message_size >> 16) & 0xff;
      header[8] = (uchar)(message_size >>  8) & 0xff;
      header[9] = (uchar)(message_size >>  0) & 0xff;

      header[10] = masking_key[0];
      header[11] = masking_key[1];
      header[12] = masking_key[2];
      header[13] = masking_key[3];

     }
//---
   if(ArrayResize(m_send,ArraySize(header),message_size)<0)
     {
      m_socket.Log("array resize error",__LINE__,__FUNCTION__);
      return(false);
     }
//---
   ArrayCopy(m_send,header,0,0);
//---
   if(message_size)
     {
      if(ArrayResize(m_send,ArraySize(header)+message_size)<0)
        {
         m_socket.Log("array resize error",__LINE__,__FUNCTION__);
         return(false);
        }
      //---
      ArrayCopy(m_send,m_txbuf,ArraySize(header),start,message_size);
      //---
      int bufsize=ArraySize(m_send);
      //---
      int message_offset=bufsize-message_size;
      //---
      for(int i=0; i<message_size; i++)
        {
         m_send[message_offset+i]^=masking_key[i&0x3];
        }
     }
//---
   if(isfinal)
     {
      it=0;
      start=0;
      m_sent=true;
      ArrayFree(m_txbuf);
     }
   else
     {
      it++;
      start=it*maxsend;
     }
//---
   return(true);

  }
```

```
//+------------------------------------------------------------------+
//|int  sendMessage() helper                                         |
//+------------------------------------------------------------------+
int  CWebSocketClient::send(ENUM_FRAME_TYPE frame_type)
  {
//---
   bool done=false;
   int bytes_sent=0,sum_sent=0;

   while(!m_sent)
     {
      done=fillTxBuffer(frame_type);
      if(done && m_socket.Writable())
        {
         bytes_sent=m_socket.Send(m_send,(uint)ArraySize(m_send));
         //---
         if(bytes_sent<0)
            break;
         else
           {
            sum_sent+=bytes_sent;
            ArrayFree(m_send);
           }
         //---
        }
      else
         break;
     }
//---
   if(ArraySize(m_send)>0)
      ArrayFree(m_send);
//---
   m_socket.Log("",__LINE__,__FUNCTION__);
//---
   return(sum_sent);
  }
```

Any data that is sent to the client is buffered in the m\_rxbuff array by the fillrxbuffer() private method, whenever Readable() public method is called. It returns the size of m\_rxbuff array indicating availability of retrievable data using a call to Read() method.

```
//+------------------------------------------------------------------+
//| receiver()fills rxbuf with raw message                           |
//+------------------------------------------------------------------+
int CWebSocketClient::fillRxBuffer(void)
  {
   uint leng=0;
   int rsp_len=-1;

//---
   uint timeout_check=GetTickCount()+m_timeout;
//---
   do
     {
      leng=m_socket.Readable();
      if(leng)
         rsp_len+=m_socket.Read(m_rxbuf,leng,m_timeout);
      leng=0;
     }
   while(GetTickCount()<timeout_check);
//---
   m_socket.Log("receive size "+IntegerToString(rsp_len),__LINE__,__FUNCTION__);
//---
   int m_rxsize=ArraySize(m_rxbuf);
//---
   if(m_rxsize<3)
      return(0);
//---
   switch((uint)m_rxbuf[1])
     {
      case 126:
         if(m_rxsize<4)
           {
            m_rxsize=0;
           }
         break;
      case 127:
         if(m_rxsize<10)
           {
            m_rxsize=0;
           }
         break;
      default:
         break;
     }
//---
   return(m_rxsize);
  }
```

```
int               Readable(void) {  return(fillRxBuffer());}
```

Read() method takes as input an array of type CFrame where all the frame(s) will be written to. The method uses private function parse() to decode the byte data so it can be correctly organized for readability. The parse() method separates the payload from the header bytes that encode descriptive information about the frames just recieved.

```
//+------------------------------------------------------------------+
//| parse() cleans up raw data buffer discarding unnecessary elements|
//+------------------------------------------------------------------+
bool CWebSocketClient::parse(CFrame &out[])
  {
   uint i,data_len=0,frames=0;
   uint s=0;
   m_total_len=0;
//---
   int shift=0;
   for(i=0; i<(uint)ArraySize(m_rxbuf); i+=(data_len+shift))
     {
      ++frames;
      m_socket.Log("value of frame is "+IntegerToString(frames)+" Value of i is "+IntegerToString(i),__LINE__,__FUNCTION__);
      switch((uint)m_rxbuf[i+1])
        {
         case 126:
            data_len=((uint)m_rxbuf[i+2]<<8)+((uint)m_rxbuf[i+3]);
            shift=4;
            break;
         case 127:
            data_len=((uint)m_rxbuf[i+2]<<56)+((uint)m_rxbuf[i+3]<<48)+((uint)m_rxbuf[i+4]<<40)+
            ((uint)m_rxbuf[i+5]<<32)+((uint)m_rxbuf[i+6]<<24)+((uint)m_rxbuf[i+7]<<16)+
            ((uint)m_rxbuf[i+8]<<8)+((uint)m_rxbuf[i+9]);
            shift=10;
            break;
         default:
            data_len=(uint)m_rxbuf[i+1];
            shift=2;
            break;
        }
      m_total_len+=data_len;
      if(data_len>0)
        {
         if(ArraySize(out)<(int)frames)
           {
            if(ArrayResize(out,frames,1)<=0)
              {
               m_socket.Log("array resize error",__LINE__,__FUNCTION__);
               return(false);
              }
           }
         //---
         if(!out[frames-1].Fill(m_rxbuf,i+shift,data_len))
           {
            m_socket.Log("Error adding new frame",__LINE__,__FUNCTION__);
            return(false);
           }
         //---
         switch((uchar)m_rxbuf[i])
           {
            case 0x1:
               if(out[frames-1].MessageType()==0)
               out[frames-1].SetMessageType(TEXT_FRAME);
               break;
            case 0x2:
               if(out[frames-1].MessageType()==0)
               out[frames-1].SetMessageType(BINARY_FRAME);
               break;
            case 0x80:
            case 0x81:
               if(out[frames-1].MessageType()==0)
               out[frames-1].SetMessageType(TEXT_FRAME);
            case 0x82:
               if(out[frames-1].MessageType()==0)
               out[frames-1].SetMessageType(BINARY_FRAME);
               m_socket.Log("received last frame",__LINE__,__FUNCTION__);
               out[frames-1].SetFinal();
               break;
            case 0x88:
               m_socket.Log("received close frame",__LINE__,__FUNCTION__);
               out[frames-1].SetMessageType(CLOSE_FRAME);
               if(m_wsclient==CONNECTED)
                 {
                  ArrayCopy(m_txbuf,m_rxbuf,0,i+shift,data_len);
                  m_wsclient=CLOSING;
                 }
               break;
            case 0x89:
               m_socket.Log("received ping frame",__LINE__,__FUNCTION__);
               out[frames-1].SetMessageType(PING_FRAME);
               if(m_wsclient==CONNECTED)
                  ArrayCopy(m_txbuf,m_rxbuf,0,i+shift,data_len);
               break;
            case 0x8a:
               m_socket.Log("received pong frame",__LINE__,__FUNCTION__);
               out[frames-1].SetMessageType(PONG_FRAME);
               break;
            default:
               break;
           }
        }
     }
//---
   return(true);
  }
```

```
uint CWebSocketClient::Read(CFrame &out[])
  {
   ClientState();
//---
   if(m_wsclient==0)
     {
      m_socket.Log("invalid ws client handle",__LINE__,__FUNCTION__);
      return(0);
     }
//---
   int rx_size=ArraySize(m_rxbuf);
//---
   if(rx_size<=0)
     {
      m_socket.Log("receive buffer is empty, Make sure to call Readable first",__LINE__,__FUNCTION__);
      return(0);
     }
//---clean up rxbuf
   if(!parse(out))
     {
      ArrayFree(m_rxbuf);
      return(0);
     }
//---
   ArrayFree(m_rxbuf);
//---
   return(m_total_len);
  }
```

### Using the class.

With the our websocket class defined let us go over how one can use it in mt5 programs. Before we begin development of any application that implements this class one must first input the address of the remote server you wish to visit, in the  list of allowed endpoints in the terminal settings.

![](https://c.mql5.com/2/40/setUrl.gif)

Remember to include  WebsocketClient.mqh, then follow the steps below:

```
CWebSocketClient wsc;
```

- declare WebsocketClient instance or instances

If you want to specify a maximum send size for all send operations relating to the connection, this is the opportune time to do so. On instance initialization the m\_maxsendsize is 0 indicating the absence of any frame size limits.

```
wsc.SetMaxSendSize(129); // max size in bytes set
```

-      Call the connect method with the appropriate input parameters and check the result.

```
if(wsc.Connect(Address,Port,Timeout,usetls,true))
{
 ////
}
```

If connection was successful you can begin sending or checking for any received messages. You can send data using either send data method which is used for previously prepared arrays.

```
sent=wsc.SendString("string message");
// or
// prepare and fill arbitrary array[] with data and send
sent=wsc.SendData(array);
```

Or if you just want to send a string message use sendstring method.

```
sent=wsc.SendPing("optional message");
```

It is also possible to send the server a ping that can  optionally be accompanied by some message. When waiting for a response after pinging the server the pong  responce frame should echo back what ever was sent with the ping. The client must also do the same if it recieves a ping from the server.

```
if(wsc.Readable()>0)
 {
  //read message....
  //declare frame object to receive message
  // and pass it to read method.
  CFrame msg_frames[];
  received=wsc.Read(msg_frames);
  Print(msg_frames[0].ToString());
  if(msg_frames[0].IsFinal())
   {
     Print("\n Final frame received");
   }
```

For receiving, check for any data readable from the socket using readable method. If method indicates a readable socket call the client read method with a Frame type object array. The websocket will then write to the object array all the message fragments that came through. Here you can use the frame type methods to query the contents of the frame array. As mentioned before if one of the received frames is a ping frame it is recommended to respond with a pong as soon as possible. To help with this requirement the websocket client will create a pong responce frame on receipt of any ping, all the user has to do is call the send ping method without any arguments.

If one of the recieved frames is a close frame the state of the websocket client will change to closing state, this means the server sent a close request and is preparing to drop the connection to this client. When in closing state send operations are limited. Client can only send the obligatory close response frame, again just as with the receipt of a ping frame , receiving a close frame sees the websocket client create a close frame ready for dispatch.

```
wsc.Close(NORMAL_CLOSE,"good bye");
// can also be called with out any arguments.
// wsc.Close();
```

When done using the websocket client call the connection close method, usually it is adequate to call the method without specifying any arguments unless there is something you wish to notify the  server about. In which case you use one of the close code reasons along with a short parting message. This message will forcibly be limited to 122 characters, any thing beyond that will be discarded.

### A local websocket server

For testing purposes the zip file accompanying this article includes a websocket server that provides an echo service. The server was built using the [libwebsocket](https://www.mql5.com/go?link=https://libwebsockets.org/ "https://libwebsockets.org/") library and the source code is available for download on [github](https://www.mql5.com/go?link=https://github.com/ufransiz/Websocketserver "https://github.com/ufransiz/Websocketserver"). To build it only Visual Studio is required as all other dependencies are available in the github repositry.

### Running the server and testing the library

To run the echo server simply double click the application (exe file). The server should start working. Please be aware that an installed firewall might block the server, so just give it the necessary permissions. It is also important to know that the accompanying .dll files contained in the server application directory are required and the server will not without them.

![Idle server](https://c.mql5.com/2/39/Idle_websocket_server.png)

Let us quickly test our WebSocketClient class. Here is an example program.

```
//+------------------------------------------------------------------+
//|                                         Websocketclient_test.mq5 |
//|                        Copyright 2019, MetaQuotes Software Corp. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2019, MetaQuotes Software Corp."
#property link      "https://www.mql5.com"
#property version   "1.00"
#property strict
#include<WebSocketClient.mqh>

input string Address="127.0.0.1";
input int    Port   =7681;
input bool   ExtTLS =false;
input int    MaxSize=256;
input int Timeout=5000;

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
string _msg="For the mql5-program to operate, it must be compiled (Compile button or F7 key). Compilation should"
            "pass without errors (some warnings are possible; they should be analyzed). At this process, an"
            "executable file with the same name and with EX5 extension must be created in the corresponding"
            "directory, terminal_dir\\MQL5\\Experts, terminal_dir\\MQL5\\indicators or terminal_dir\\MQL5\\scripts."
            "This file can be run."
            "Operating features of MQL5 programs are described in the following sections:"
            "- Program running – order of calling predefined event-handlers."
            "- Testing trading strategies – operating features of MQL5 programs in the Strategy Tester."
            "- Client terminal events – description of events, which can be processed in programs."
            "- Call of imported functions – description order, allowed parameters, search details and call agreement"
            "for imported functions."
            "· Runtime errors – getting information about runtime and critical errors."
            "Expert Advisors, custom indicators and scripts are attached to one of opened charts by Drag'n'Drop"
            "method from the Navigator window."
            "For an expert Advisor to stop operating, it should be removed from a chart. To do it select 'Expert'"
            "'list' in chart context menu, then select an Expert Advisor from list and click 'Remove' button."
            "Operation of Expert Advisors is also affected by the state of the 'AutoTrading' button."
            "In order to stop a custom indicator, it should be removed from a chart."
            "Custom indicators and Expert Advisors work until they are explicitly removed from a chart;"
            "information about attached Expert Advisors and Indicators is saved between client terminal sessions."
            "Scripts are executed once and are deleted automatically upon operation completion or change of the"
            "current chart state, or upon client terminal shutdown. After the restart of the client terminal scripts"
            "are not started, because the information about them is not saved."
            "Maximum one Expert Advisor, one script and unlimited number of indicators can operate in one chart."
            "Services do not require to be bound to a chart to work and are designed to perform auxiliary functions."
            "For example, in a service, you can create a custom symbol, open its chart, receive data for it in an"
            "endless loop using the network functions and constantly update it."
            "Each script, each service and each Expert Advisor runs in its own separate thread. All indicators"
            "calculated on one symbol, even if they are attached to different charts, work in the same thread."
            "Thus, all indicators on one symbol share the resources of one thread."
            "All other actions associated with a symbol, like processing of ticks and history synchronization, are"
            "also consistently performed in the same thread with indicators. This means that if an infinite action is"
            "performed in an indicator, all other events associated with its symbol will never be performed."
            "When running an Expert Advisor, make sure that it has an actual trading environment and can access"
            "the history of the required symbol and period, and synchronize data between the terminal and the"
            "server. For all these procedures, the terminal provides a start delay of no more than 5 seconds, after"
            "which the Expert Advisor will be started with available data. Therefore, in case there is no connection"
            "to the server, this may lead to a delay in the start of an Expert Advisor.";
//---
CWebSocketClient wsc;
//---
int sent=-1;
uint received=-1;
//---
// string subject,issuer,serial,thumbprint;
//---
// datetime expiration;
//---
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//--- create timer
   EventSetTimer(2);
//---
   wsc.SetMaxSendSize(MaxSize);
//---
   if(wsc.Connect(Address,Port,Timeout,ExtTLS,true))
     {
      sent=wsc.SendString(_msg);
      //--
      Print("sent data is "+IntegerToString(sent));
      //---
     }
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
   Print("Deinit call");
   wsc.Close();

  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//---

  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void OnTimer()
  {
   if(wsc.Readable()>0)
     {
      CFrame msg_frames[];
      received=wsc.Read(msg_frames);
      if(received>0)
        {
         int ll=ArraySize(msg_frames);
         Print("number of received frames is "+IntegerToString(ll));
         for(int i=0; i<ll; i++)
           {
            Print(msg_frames[i].ToString());
           }

         if(msg_frames[ll-1].IsFinal())
           {
            Print("\n Final frame received");
            wsc.Close(NORMAL_CLOSE,"good bye");
            ExpertRemove();
           }
        }
     }
   else
     {
      Print("\n Nothing readable in socket");
      if(wsc.ClientState()!=CONNECTED)
        {
         Print("\n Client disconnected");
         ExpertRemove();
        }
     }
  }
//+------------------------------------------------------------------+
```

This expert advisor connects to the locally running echo websocket server and immediately tries to send a fairly large message. The EA inputs allow for enabling , disabling tls and adjusting the send size to see how the message fragmentation mechanism works. In the code i set maximum message size to 256 so, each frame will be that size or less.

In the onTimer function the ea checks for any messages available from the server.The received message is output to the mt5 terminal then the websocket connection is dropped. On the next Ontimer event if the connection is closed the EA will remove itself from the chart.  Here is the output from the Mt5 experts tab.

![Header Output](https://c.mql5.com/2/39/header_output_from_client.png)

![Parsing data ](https://c.mql5.com/2/39/parsing_data.png)

![Receiving frames](https://c.mql5.com/2/39/received_frames_converted_to_text.png)

![Close frame being constructed](https://c.mql5.com/2/39/close_frame_being_built.png)

And output from the websocket server .

![Server screenshot](https://c.mql5.com/2/39/console_app_screenshot.png)

Here is a video of the program running whilst connected to the server.

MetaTrader 5 websocket client Demo - YouTube

[Photo image of MQL5.community](https://www.youtube.com/channel/UC8bwZMk2yh5WDToFos49OjQ?embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F8196)

MQL5.community

1.91K subscribers

[MetaTrader 5 websocket client Demo](https://www.youtube.com/watch?v=fYbiB9LELLg)

MQL5.community

Search

Watch later

Share

Copy link

Info

Shopping

Tap to unmute

If playback doesn't begin shortly, try restarting your device.

Share

Include playlist

An error occurred while retrieving sharing information. Please try again later.

0:00

0:00 / 3:03

•Live

•

### Conclusion

This article began with a brief overview of the websocket protocol. Then came a detailed description of how  a websocket client can be implemented in Metrader 5 using only the mql5 programming language. Next, we built a server which we then used to test our mt5 client. I hope you will find the tools described here useful. All the source code is available below for download.

Contents of attached archive.

| Folder | Contents | Description |
| --- | --- | --- |
| MT5zip\\server | echo\_websocket\_server.exe, websockets.dll,ssleay32.dll,libeay32.dll | Server application along with the required dependencies for it |
| MT5zip\\Mql5\\include | Frame.mqh, Socket.mqh, WebsocketClient.mqh | Include files containing code for the CFrame class, CSocket class and the CWebsocket class repectively |
| MT5zip\\Mql5\\Experts | Websocketclient\_test.mq5 | MetaTrader Expert Advisor demonstrating use of CWebsocket class |

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/8196.zip "Download all attachments in the single ZIP archive")

[MT5zip.zip](https://www.mql5.com/en/articles/download/8196/mt5zip.zip "Download MT5zip.zip")(826.64 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/356467)**
(41)


![Soewono Effendi](https://c.mql5.com/avatar/2016/10/57FE8583-A88B.png)

**[Soewono Effendi](https://www.mql5.com/en/users/seffx)**
\|
30 Dec 2024 at 12:49

**Shane Leigh Kingston [#](https://www.mql5.com/en/forum/356467/page2#comment_55496671):**

I think you might be confusing websockets with regular networking sockets or plain HTTP  WebRequests. Websockets require custom coding on top of regular HTTP requests. I managed to get a solution working based on this article with some project specific modifications.

gave you a hint,

if you do not want to look in the book, it's your loss not mine ;)

![pauldic](https://c.mql5.com/avatar/avatar_na2.png)

**[pauldic](https://www.mql5.com/en/users/pauldic)**
\|
9 Mar 2025 at 19:43

**Shane Leigh Kingston [#](https://www.mql5.com/en/forum/356467#comment_55494845):**

**Note there is a bug in WebsocketClient.fillRxBuffer**, in the scenario where there are multiple calls to m\_socket.Read(..) within the while loop.

m\_socket.Read(..) is not appending data to the end of the array, but rather it writes to the start of the array.

In my case, where I was connecting to a local web socket for testing, I found the first call to m\_socket.Read(..) was fetching a single byte only, then a second loop to m\_socket.Read(..) was fetching the rest. As a result, the buffer was missing the first byte, which caused an error when [parsing](https://www.mql5.com/en/articles/5638 "Article: Parsing MQL Using MQL ") the frame.

You also need to ensure the m\_rxbuf is empty prior to filling the buffer, or it may think more data has been fetched then it actually has. The buffer is cleared out after parsing frames, but just to be sure, I decided to clear it out whenever calling fillRxBuffer.

I also made the m\_socket.Read(..) stop looping once there is no more data to read, so it doesn't keep waiting for the timeout period. Ideally I think it should actually keep reading until there is enough data to parse a frame, but that requires restructuring the code a bit.

Thanks for this article though. So far its the closest solution I have found to what I was looking for. Eventually I may create my own web socket library.

Please @ **Shane Leigh Kingston, I knew this is an old post but need a way to implement websocket with mql5 and this library seems to be the only relatively close to it. But again am face the issue you just described but am not an expert in this area. Please can you guide me on how to make this work. I will be high grateful if you could help.**

**Thanks in advance**

![Stanislav Korotky](https://c.mql5.com/avatar/2010/10/4CA7CFA0-1F0C.jpg)

**[Stanislav Korotky](https://www.mql5.com/en/users/marketeer)**
\|
10 Mar 2025 at 13:25

**pauldic [#](https://www.mql5.com/en/forum/356467/page2#comment_56117881):**

Please @ **Shane Leigh Kingston, I knew this is an old post but need a way to implement websocket with mql5 and this library seems to be the only relatively close to it. But again am face the issue you just described but am not an expert in this area. Please can you guide me on how to make this work. I will be high grateful if you could help.**

There is another implementation of websockets in [the algotrading book](https://www.mql5.com/en/book/advanced/project/project_websocket_mql5).

You may find actual version of source codes in the discussion on the forum:

[Forum on trading, automated trading systems and testing trading strategies](https://www.mql5.com/en/forum)

[Experts: MQL5 Programming for Traders – Source Codes from the Book. Part 7](https://www.mql5.com/en/forum/459080#comment_56126838)

[Stanislav Korotky](https://www.mql5.com/en/users/marketeer), 2025.03.10 13:32

I'm attaching some bugfixes and improvements in the websockets classes.

![pauldic](https://c.mql5.com/avatar/avatar_na2.png)

**[pauldic](https://www.mql5.com/en/users/pauldic)**
\|
11 Mar 2025 at 19:48

**Stanislav Korotky [#](https://www.mql5.com/en/forum/356467/page2#comment_56126808):**

There is another implementation of websockets in [the algotrading book](https://www.mql5.com/en/book/advanced/project/project_websocket_mql5).

You may find actual version of source codes in the discussion on the forum:

Yes @Stanislav I later found it yesterday and it has been useful thus far, thank you


![Vladislav Boyko](https://c.mql5.com/avatar/2025/12/692e1587-6181.png)

**[Vladislav Boyko](https://www.mql5.com/en/users/boyvlad)**
\|
18 Oct 2025 at 10:00

Comments that do not relate to this topic, have been moved to " [Experts: MQL5 Programming for Traders – Source Codes from the Book. Part 6](https://www.mql5.com/en/forum/459067)".

![What is a trend and is the market structure based on trend or flat?](https://c.mql5.com/2/39/unnamed.png)[What is a trend and is the market structure based on trend or flat?](https://www.mql5.com/en/articles/8184)

Traders often talk about trends and flats but very few of them really understand what a trend/flat really is and even fewer are able to clearly explain these concepts. Discussing these basic terms is often beset by a solid set of prejudices and misconceptions. However, if we want to make profit, we need to understand the mathematical and logical meaning of these concepts. In this article, I will take a closer look at the essence of trend and flat, as well as try to define whether the market structure is based on trend, flat or something else. I will also consider the most optimal strategies for making profit on trend and flat markets.

![Price series discretization, random component and noise](https://c.mql5.com/2/39/4qc92l.png)[Price series discretization, random component and noise](https://www.mql5.com/en/articles/8136)

We usually analyze the market using candlesticks or bars that slice the price series into regular intervals. Doesn't such discretization method distort the real structure of market movements? Discretization of an audio signal at regular intervals is an acceptable solution because an audio signal is a function that changes over time. The signal itself is an amplitude which depends on time. This signal property is fundamental.

![Neural networks made easy (Part 2): Network training and testing](https://c.mql5.com/2/48/Neural_networks_made_easy_002.png)[Neural networks made easy (Part 2): Network training and testing](https://www.mql5.com/en/articles/8119)

In this second article, we will continue to study neural networks and will consider an example of using our created CNet class in Expert Advisors. We will work with two neural network models, which show similar results both in terms of training time and prediction accuracy.

![Timeseries in DoEasy library (part 49): Multi-period multi-symbol multi-buffer standard indicators](https://c.mql5.com/2/40/MQL5-avatar-doeasy-library__1.png)[Timeseries in DoEasy library (part 49): Multi-period multi-symbol multi-buffer standard indicators](https://www.mql5.com/en/articles/8292)

In the current article, I will improve the library classes to implement the ability to develop multi-symbol multi-period standard indicators requiring several indicator buffers to display their data.

[What's wrong with regular VPS?Here are the 8 most common problems that algorithmic traders may encounterRead![](https://www.mql5.com/ff/sh/hzatb686qjqxwtr4z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/450486&a=drhremihlwuaqyvgpzfddbtmgciejpba&s=c37d25bcceb93ed153b814e6ba4d4839461a9b2d68dd82b95b142be06d310f3f&uid=&ref=https://www.mql5.com/en/articles/8196&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5071670908734614607)

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