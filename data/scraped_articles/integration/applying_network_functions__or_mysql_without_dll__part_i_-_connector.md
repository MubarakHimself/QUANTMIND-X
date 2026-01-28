---
title: Applying network functions, or MySQL without DLL: Part I - Connector
url: https://www.mql5.com/en/articles/7117
categories: Integration
relevance_score: 0
scraped_at: 2026-01-24T14:05:56.595849
---

[Need a reliable hosting solution for your robots?Contact your broker and find out about available Sponsored MetaTrader VPS offeringsLearn more![](https://www.mql5.com/ff/sh/0pw0dk81s56qy774z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=vljwvezfjkfbvviocwskggexlvgykvob&s=70cf8e354b9a125332533ffb65d7365abe8dde5b5c1ede9caac479a9e9df4f25&uid=&ref=https://www.mql5.com/en/articles/7117&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5083334175894804881)

MetaTrader 5 / Integration


### Contents

- [Introduction](https://www.mql5.com/en/articles/7117#para1)
- [Sockets](https://www.mql5.com/en/articles/7117#para2)
- [Wireshark traffic analyzer](https://www.mql5.com/en/articles/7117#para3)
- [Data exchange](https://www.mql5.com/en/articles/7117#idrxtx)

  - [Receipt](https://www.mql5.com/en/articles/7117#idrx)
  - [Sending](https://www.mql5.com/en/articles/7117#idtx)

- [MySQL transaction class](https://www.mql5.com/en/articles/7117#idtransaction)

- [Application](https://www.mql5.com/en/articles/7117#idusage)

- [Documentation](https://www.mql5.com/en/articles/7117#iddocs)

- [Conclusion](https://www.mql5.com/en/articles/7117#para12)

### Introduction

About a year ago, MQL5 [network functions](https://www.mql5.com/en/docs/network) were replenished with functions
for working with [sockets](https://www.mql5.com/en/docs/network/socketcreate). This opened up great opportunities for
programmers developing products for the Market. Now they can implement things that required dynamic libraries before. We will consider
one of such examples in this series of two articles. In the first article, I am going to consider the MySQL connector principles, while in the
second one, I will develop the simplest applications using the connector, namely the service for collecting properties of signals
available in the terminal and the program for viewing their changes over time (see Fig. 1).

![The program for viewing changes in signal properties within a certain time](https://c.mql5.com/2/37/s_from_db_001.png)

Fig. 1. The program for viewing changes in signal properties over time

### Sockets

A socket is a software interface for exchanging data between processes. The processes can be launched both on a single PC and on different
ones connected into a network.

MQL5 provides client TCP sockets only. This means we are able to initiate a connection but we cannot wait for it from the outside. Therefore, if
we need to provide a connection between MQL5 programs via sockets, we need a server which is to act as an intermediary. The server waits for a
connection to the listened port and performs certain functions at the client's request. To connect to the server, we need to know its ip
address and port.

A port is a number ranging from 0 to 65535. There are three port ranges: system (0 - 1023), user (1024-49151) and dynamic ones (49152-65535).
Some ports are allocated to work with certain functions. The allocation is performed by [IANA](https://en.wikipedia.org/wiki/Internet_Assigned_Numbers_Authority "https://en.wikipedia.org/wiki/Internet_Assigned_Numbers_Authority")
– an organization that manages IP address zones and top-level domains, as well as registers MIME data types.

The [port \\
3306](https://www.mql5.com/go?link=https://www.iana.org/assignments/service-names-port-numbers/service-names-port-numbers.xhtml?search=mysql "https://www.iana.org/assignments/service-names-port-numbers/service-names-port-numbers.xhtml?search=mysql") is allocated to MySQL by default. We will connect to it when accessing the server. Please note that this value can be changed.
Therefore, when developing an EA, the port should be set in the inputs along with the IP address.

The following approach is used when working with sockets:

- Create a socket (get a handle or an error)
- Connect to the server
- Exchange data
- Close the socket

When working with multiple connections, keep in mind the limitation of 128 simultaneously open sockets for a single MQL5 program.

### Wireshark traffic analyzer

The traffic analyzer facilitates debugging the code of a program applying sockets. Without it, the entire process resembles repairing
electronics without an oscilloscope. The analyzer captures data from the selected network interface and displays it in a readable form. It
tracks the size of the packets, the time gap between them, the presence of retransmits and connection drops, as well as many other useful
data. It also decrypts many protocols.

Personally, I use [Wireshark](https://en.wikipedia.org/wiki/Wireshark "https://en.wikipedia.org/wiki/Wireshark")
for these purposes.

![Traffic analyzer](https://c.mql5.com/2/37/ws_002.png)

Fig. 2. Wireshark traffic analyzer

Figure 2 shows the traffic analyzer window with captured packets where:

1. Display filter line. "tcp.port==3306" means that only packets with a local or remote TCP port of 3306 are displayed (default MySQL server
    port).
2. Packets. Here we can see the connection setup process, server greeting, authorization request and subsequent exchange.

3. Selected packet content in hexadecimal form. In this case, we can see the contents of the MySQL server greeting packet.

4. Transport level (TCP). We are located here when using functions for working with sockets.

5. Application level (MySQL). This is what we are to consider in this article.


The display filter does not limit packet capture. This is clearly visible in the status bar stating that 35 captured packets out of 2623 ones
located in the memory are being handled at the moment. To reduce the load on PC, we should set the capture filter when selecting the network
interface, as shown in Fig. 3. This should be done only if all the other packages are really not useful.

![Packet capture filter](https://c.mql5.com/2/39/interface_filter__1.png)

Fig. 3. Packet capture filter

To familiarize ourselves with the traffic analyzer, let's try to establish a connection with the "google.com" server and track the
process. To do this, write a small script.

```
void OnStart()
  {
//--- Get socket handle
   int socket=SocketCreate();
   if(socket==INVALID_HANDLE)
      return;
//--- Establish connection
   if(SocketConnect(socket,"google.com",80,2000)==false)
     {
      return;
     }
   Sleep(5000);
//--- Close connection
   SocketClose(socket);
  }
```

So, first we create a socket and get its handle using the [SocketCreate()](https://www.mql5.com/en/docs/network/socketcreate)
function. The reference says that in this case, you can get an error in two cases which are almost impossible:

1. The ERR\_NETSOCKET\_TOO\_MANY\_OPENED error signals that more than 128 sockets are open.
2. The ERR\_FUNCTION\_NOT\_ALLOWED error appears when trying to call a socket creation from an indicator, in which this feature is disabled.

After receiving the handle, try to establish the connection. In this example, we connect to the "google.com" server (do not forget to add it to
the allowed addresses in the terminal settings), namely, to port 80
with the timeout of 2 000 milliseconds. After establishing the
connection, wait for 5 seconds and close
it. Now let's see how it looks in the traffic analyzer window.

![Establishing and closing a connection](https://c.mql5.com/2/37/ws_003.png)

Fig. 4. Establishing and closing a connection

In Figure 4, we can see the data exchange between our script and the "google.com" server with the "172.217.16.14" ip address. DNS queries are
not displayed here since the filter line features the "tcp.port==80" expression.

The upper three packets represent establishing a connection, while the lower three ones represent its closure. The Time column displays the
time between packets, and we can see 5 seconds of downtime. Please note that the packets are colored green, unlike the ones in Figure 2. This is
because in the previous case, the analyzer detected the MySQL protocol in the exchange. In the current case, no data was passed and the
analyzer highlighted the packets with the default TCP color.

### Data exchange

According to the protocol, the MySQL server should send a greeting after establishing a connection. In response to it, the client sends an
authorization request. This mechanism is described in detail in the [Connection \\
Phase](https://www.mql5.com/go?link=https://dev.mysql.com/doc/dev/mysql-server/8.0.12/page_protocol_connection_phase.html "https://dev.mysql.com/doc/dev/mysql-server/8.0.12/page_protocol_connection_phase.html") section on the [dev.mysql.com](https://www.mql5.com/go?link=https://dev.mysql.com/ "https://dev.mysql.com/")
website. If the greeting is not received, the IP address is invalid or the server is listening to another port. In any case, this means we have connected
to something that is definitely not a MySQL server. In a normal situation, we need to receive data (read it from the socket) and parse it.

### Receipt

In the [CMySQLTransaction](https://www.mql5.com/en/articles/7117#idcmysqltransaction) class (to be [described](https://www.mql5.com/en/articles/7117#idtransaction)
in detail a bit later), data receipt has been implemented as follows:

```
//+------------------------------------------------------------------+
//| Data receipt                                                     |
//+------------------------------------------------------------------+
bool CMySQLTransaction::ReceiveData(ushort error_code=0)
  {
   char buf[];
   uint   timeout_check=GetTickCount()+m_timeout;
   do
     {
      //--- Get the amount of data that can be read from the socket
      uint len=SocketIsReadable(m_socket);
      if(len)
        {
         //--- Read data from the socket to the buffer
         int rsp_len=SocketRead(m_socket,buf,len,m_timeout);
         m_rx_counter+= rsp_len;
         //--- Send the buffer for handling
         ENUM_TRANSACTION_STATE res = Incoming(buf,rsp_len);
         //--- Get the result the following actions will depend on
         if(res==MYSQL_TRANSACTION_COMPLETE) // server response fully accepted
            return true;   // exit (successful)
         else
            if(res==MYSQL_TRANSACTION_ERROR) // error
              {
               if(m_packet.error.code)
                  SetUserError(MYSQL_ERR_SERVER_ERROR);
               else
                  SetUserError(MYSQL_ERR_INTERNAL_ERROR);
               return false;  // exit (error)
              }
         //--- In case of another result, continue waiting for data in the loop
        }
     }
   while(GetTickCount()<timeout_check && !IsStopped());
//--- If waiting for the completion of the server response receipt took longer than m_timeout,
//--- exit with the error
   SetUserError(error_code);
   return false;
  }
```

Here **m\_socket** is a socket handle obtained earlier when creating it, while **m\_timeout** is data reading timeout
used as the [SocketRead()](https://www.mql5.com/en/docs/network/socketread) function argument for accepting a fragment of
data, as well as in the form of the timeout of receiving the entire data. Before entering the loop, set a timestamp. Reaching it is considered
the data receipt timeout:

```
uint   timeout_check=GetTickCount()+m_timeout;
```

Next, read the [SocketIsReadable()](https://www.mql5.com/en/docs/network/socketisreadable) function result in a
loop and wait till it returns a non-zero value. After that, read data to the buffer and pass it for handling.

```
      uint len=SocketIsReadable(m_socket);
      if(len)
        {
         //--- Read data from the socket to the buffer
         int rsp_len=SocketRead(m_socket,buf,len,m_timeout);
         m_rx_counter+= rsp_len;
         //--- Send the buffer for handling
         ENUM_TRANSACTION_STATE res = Incoming(buf,rsp_len);

        ...

        }
```

**We cannot count on** the ability to accept the entire packet if there is data in the socket. There are a number of situations where
data can arrive in small portions. For example, it may be a poor connection via a 4G modem with a large number of retransmits. Therefore, our
handler should be able to collect data into some indivisible groups it is possible to work with. Let's use MySQL packets for that.

The CMySQLTransaction::Incoming() method is used to accumulate and handle data:

```
   //--- Handle received data
   ENUM_TRANSACTION_STATE  Incoming(uchar &data[], uint len);
```

The result it returns lets us know what to do next — continue, complete or interrupt the process of receiving data:

```
enum ENUM_TRANSACTION_STATE
  {
   MYSQL_TRANSACTION_ERROR=-1,         // Error
   MYSQL_TRANSACTION_IN_PROGRESS=0,    // In progress
   MYSQL_TRANSACTION_COMPLETE,         // Fully completed
   MYSQL_TRANSACTION_SUBQUERY_COMPLETE // Partially completed
  };
```

In case of an internal error, as well as when getting a server error or completing data receipt, reading data from the socket should be
stopped. In all other cases, it should be continued. MYSQL\_TRANSACTION\_SUBQUERY\_COMPLETE value indicates that one of the server
responses to a client's multiple query has been accepted. It is equivalent to MYSQL\_TRANSACTION\_IN\_PROGRESS for the reading algorithm.

![MySQL packet](https://c.mql5.com/2/37/packet2.png)

Fig. 5. MySQL packet

MySQL packet format is displayed in Fig. 5. The first three bytes define the size of a useful load in the packet, while the next byte means the
serial number of the packet in the sequence and is followed by data. The serial number is set to zero at the beginning of each exchange. For
example, the greeting packet is 0, client authorization request — 1, server response — 2 (end of the connection phase). Next, when sending a
client query, the value of the sequence number should be set to zero again and is increased in each server response packet. If the number of
packets exceeds 255, the number value passes zero.

The simplest packet (MySQL ping) looks as follows in the traffic analyzer:

![Ping packet in the traffic analyzer](https://c.mql5.com/2/37/ws_005__1.png)

Fig. 6. Ping packet in the traffic analyzer

The Ping packet contains one byte of data with the value of 14 (or 0x0E in hexadecimal form).

Let's consider the CMySQLTransaction::Incoming() method which gathers data to packets and passes them to handlers. Its abridged source code
is provided below.

```
ENUM_TRANSACTION_STATE CMySQLTransaction::Incoming(uchar &data[], uint len)
  {
   int ptr=0; // index of the current byte in the 'data' buffer
   ENUM_TRANSACTION_STATE result=MYSQL_TRANSACTION_IN_PROGRESS; // result of handling accepted data
   while(len>0)
     {
      if(m_packet.total_length==0)
        {
         //--- If the amount of data in the packet is unknown
         while(m_rcv_len<4 && len>0)
           {
            m_hdr[m_rcv_len] = data[ptr];
            m_rcv_len++;
            ptr++;
            len--;
           }
         //--- Received the amount of data in the packet
         if(m_rcv_len==4)
           {
            //--- Reset error codes etc.
            m_packet.Reset();
            m_packet.total_length = reader.TotalLength(m_hdr);
            m_packet.number = m_hdr[3];
            //--- Length received, reset the counter of length bytes
            m_rcv_len = 0;
            //--- Highlight the buffer of a specified size
            if(ArrayResize(m_packet.data,m_packet.total_length)!=m_packet.total_length)
               return MYSQL_TRANSACTION_ERROR;  // internal error
           }
         else // if the amount of data is still not accepted
            return MYSQL_TRANSACTION_IN_PROGRESS;
        }
      //--- Collect packet data
      while(len>0 && m_rcv_len<m_packet.total_length)
        {
         m_packet.data[m_rcv_len] = data[ptr];
         m_rcv_len++;
         ptr++;
         len--;
        }
      //--- Make sure the package has been collected already
      if(m_rcv_len<m_packet.total_length)
         return MYSQL_TRANSACTION_IN_PROGRESS;

      //--- Handle received MySQL packet
      //...
      //---

      m_rcv_len = 0;
      m_packet.total_length = 0;
     }
   return result;
  }
```

The first step is to collect the packet header — the first 4 bytes containing data length and serial number in the sequence. To accumulate the
header, use the **m\_hdr** buffer and **m\_rcv\_len** byte counter. When 4 bytes are collected, get their length and change the **m\_packet.data**
buffer based on it. Received packet data is copied to it. When the packet is ready, pass it to the handler.

If **len** length of received data is still not zero after receiving the packet, this means we have received several packets. We can handle
either **several whole packets, or several partial ones** in a single Incoming() method call.

Packet types are provided below:

```
enum ENUM_PACKET_TYPE
  {
   MYSQL_PACKET_NONE=0,    // None
   MYSQL_PACKET_DATA,      // Data
   MYSQL_PACKET_EOF,       // End of file
   MYSQL_PACKET_OK,        // Ok
   MYSQL_PACKET_GREETING,  // Greeting
   MYSQL_PACKET_ERROR      // Error
  };
```

Each of them has its own handler, which parses their sequence and contents according to the protocol. Values received during the parsing are
assigned to members of the corresponding classes. In the current connector implementation, all data received in packets is parsed. This
may seem somewhat redundant since the properties of the "Table" and "Original table" fields often coincide. Besides, the values of some
flags are rarely needed (see Fig. 7). However, the availability of these properties allows to flexibly build the logic of interacting with
the MySQL server at the application layer of the program.

![Packets in Wireshark analyzer](https://c.mql5.com/2/37/ws_fields.png)

Fig. 7. Field description packet

### Sending

Sending data is a bit easier here.

```
//+------------------------------------------------------------------+
//| Form and send ping                                               |
//+------------------------------------------------------------------+
bool CMySQLTransaction::ping(void)
  {
   if(reset_rbuf()==false)
     {
      SetUserError(MYSQL_ERR_INTERNAL_ERROR);
      return false;
     }
//--- Prepare the output buffer
   m_tx_buf.Reset();
//--- Reserve a place for the packet header
   m_tx_buf.Add(0x00,4);
//--- Place the command code
   m_tx_buf+=uchar(0x0E);
//--- Form a header
   m_tx_buf.AddHeader(0);
   uint len = m_tx_buf.Size();
//--- Send a packet
   if(SocketSend(m_socket,m_tx_buf.Buf,len)!=len)
      return false;
   m_tx_counter+= len;
   return true;
  }
```

The source code of the ping sending method is provided above. Copy data to the prepared buffer. In case of the ping, this is the code of the 0x0E
command. Next, form the header considering the amount of data and the packet serial number. For a ping, the serial number is always equal to zero.
After that, try to send the assembled packet using the [SocketSend()](https://www.mql5.com/en/docs/network/socketsend)
function.

The method of sending a query (Query) is similar to sending a ping:

```
//+------------------------------------------------------------------+
//| Form and send a query                                            |
//+------------------------------------------------------------------+
bool CMySQLTransaction::query(string s)
  {
   if(reset_rbuf()==false)
     {
      SetUserError(MYSQL_ERR_INTERNAL_ERROR);
      return false;
     }
//--- Prepare the output buffer
   m_tx_buf.Reset();
//--- Reserve a place for the packet header
   m_tx_buf.Add(0x00,4);
//--- Place the command code
   m_tx_buf+=uchar(0x03);
//--- Add the query string
   m_tx_buf+=s;
//--- Form a header
   m_tx_buf.AddHeader(0);
   uint len = m_tx_buf.Size();
//--- Send a packet
   if(SocketSend(m_socket,m_tx_buf.Buf,len)!=len)
      return false;
   m_tx_counter+= len;
   return true;
  }
```

The only difference is that the useful load consists of the (0x03)
command code and the query string.

Sending data is always followed by the CMySQLTransaction::ReceiveData() receipt method we have considered before. If it returns no error,
transaction is considered successful.

### MySQL transaction class

It is now time to consider the **CMySQLTransaction** class in more detail.

```
//+------------------------------------------------------------------+
//| MySQL transaction class                                          |
//+------------------------------------------------------------------+
class CMySQLTransaction
  {
private:
   //--- Authorization data
   string            m_host;        // MySQL server IP address
   uint              m_port;        // TCP port
   string            m_user;        // User name
   string            m_password;    // Password
   //--- Timeouts
   uint              m_timeout;        // timeout of waiting for TCP data (ms)
   uint              m_timeout_conn;   // timeout of establishing a server connection
   //--- Keep Alive
   uint              m_keep_alive_tout;      // time(ms), after which the connection is closed; the value of 0 - Keep Alive is not used
   uint              m_ping_period;          // period of sending ping (in ms) in the Keep Alive mode
   bool              m_ping_before_query;    // send 'ping' before 'query' (this is reasonable in case of large ping sending periods)
   //--- Network
   int               m_socket;      // socket handle
   ulong             m_rx_counter;  // counter of bytes received
   ulong             m_tx_counter;  // counter of bytes passed
   //--- Timestamps
   ulong             m_dT;                   // last query time
   uint              m_last_resp_timestamp;  // last response time
   uint              m_last_ping_timestamp;  // last ping time
   //--- Server response
   CMySQLPacket      m_packet;      // accepted packet
   uchar             m_hdr[4];      // packet header
   uint              m_rcv_len;     // counter of packet header bytes
   //--- Transfer buffer
   CData             m_tx_buf;
   //--- Authorization request class
   CMySQLLoginRequest m_auth;
   //--- Server response buffer and its size
   CMySQLResponse    m_rbuf[];
   uint              m_responses;
   //--- Waiting and accepting data from the socket
   bool              ReceiveData(ushort error_code);
   //--- Handle received data
   ENUM_TRANSACTION_STATE  Incoming(uchar &data[], uint len);
   //--- Packet handlers for each type
   ENUM_TRANSACTION_STATE  PacketOkHandler(CMySQLPacket *p);
   ENUM_TRANSACTION_STATE  PacketGreetingHandler(CMySQLPacket *p);
   ENUM_TRANSACTION_STATE  PacketDataHandler(CMySQLPacket *p);
   ENUM_TRANSACTION_STATE  PacketEOFHandler(CMySQLPacket *p);
   ENUM_TRANSACTION_STATE  PacketErrorHandler(CMySQLPacket *p);
   //--- Miscellaneous
   bool              ping(void);                // send ping
   bool              query(string s);           // send a query
   bool              reset_rbuf(void);          // initialize the server response buffer
   uint              tick_diff(uint prev_ts);   // get the timestamp difference
   //--- Parser class
   CMySQLPacketReader   reader;
public:
                     CMySQLTransaction();
                    ~CMySQLTransaction();
   //--- Set connection parameters
   bool              Config(string host,uint port,string user,string password,uint keep_alive_tout);
   //--- Keep Alive mode
   void              KeepAliveTimeout(uint tout);                       // set timeout
   void              PingPeriod(uint period) {m_ping_period=period;}    // set ping period in seconds
   void              PingBeforeQuery(bool st) {m_ping_before_query=st;} // enable/disable ping before a query
   //--- Handle timer events (relevant when using Keep Alive)
   void              OnTimer(void);
   //--- Get the pointer to the class for working with authorization
   CMySQLLoginRequest *Handshake(void) {return &m_auth;}
   //--- Send a request
   bool              Query(string q);
   //--- Get the number of server responses
   uint              Responses(void) {return m_responses;}
   //--- Get the pointer to the server response by index
   CMySQLResponse    *Response(uint idx);
   CMySQLResponse    *Response(void) {return Response(0);}
   //--- Get the server error structure
   MySQLServerError  GetServerError(void) {return m_packet.error;}
   //--- Options
   ulong             RequestDuration(void) {return m_dT;}                     // get the last transaction duration
   ulong             RxBytesTotal(void) {return m_rx_counter;}                // get the number of received bytes
   ulong             TxBytesTotal(void) {return m_tx_counter;}                // get the number of passed bytes
   void              ResetBytesCounters(void) {m_rx_counter=0; m_tx_counter=0;} // reset the counters of received and passed bytes
  };
```

Let's have a closer look at the following private members:

- **m\_packet** of CMySQLPacket type — class of the currently handled MySQL packet (source code with comments in the
MySQLPacket.mqh file)

- **m\_tx\_buf** of CData type — class of the transfer buffer created for the convenience of generating a query (Data.mqh file)
- **m\_auth** of CMySQLLoginRequest type — class for working with authorization (password scrambling, storing obtained server
parameters and specified client parameters, the source code is in MySQLLoginRequest.mqh)
- **m\_rbuf** of CMySQLResponse type — server repsonse buffer; the response here is the "Ok" or "Data" type packet
(MySQLResponse.mqh)
- **reader** of CMySQLPacketReader type — MySQL packet parser class


The public methods are described in detail in the [documentation](https://www.mql5.com/en/articles/7117#iddocs).

For the application layer, the transaction class looks as displayed in Figure 8.

![Classes](https://c.mql5.com/2/37/struct.png)

Fig. 8. CMySQLTransaction class structure

where:

- **CMySQLLoginRequest**— should be configured before establishing a connection when specifying client parameters whose
values are different from the predefined ones (optional);

- **CMySQLResponse**— server response if a transaction is completed without errors

  - **CMySQLField**— field description;

  - **CMySQLRow**— row (buffer of field values in text form);


- **MySQLServerError**— error description structure in case a transaction failed.


There are no public methods responsible for establishing and closing a connection. This is done automatically
when calling the CMySQLTransaction::Query() method. When using the constant connection mode, it is established during the first call of
CMySQLTransaction::Query() and closed after the defined timeout.

**Important:** In the constant connection mode, the [OnTimer](https://www.mql5.com/en/docs/event_handlers/ontimer)
event handler should receive the call of the CMySQLTransaction::OnTimer() method. In this case, the timer period should be less than the ping
and timeout periods.

Parameters of connection, user account, as well as special client parameter values should be set before calling
CMySQLTransaction::Query().

In general, interaction with the transaction class is performed according to the following principle:

![Working with the transaction class](https://c.mql5.com/2/39/usage.png)

Fig. 9. Working with the CMySQLTransaction class

### Application

Let's consider the simplest example of applying the connector. To do this, write a script sending the SELECT query to the [world \\
test database](https://www.mql5.com/go?link=https://dev.mysql.com/doc/index-other.html "https://dev.mysql.com/doc/index-other.html").

```
//--- input parameters
input string   inp_server     = "127.0.0.1";          // MySQL server address
input uint     inp_port       = 3306;                 // TCP port
input string   inp_login      = "admin";              // Login
input string   inp_password   = "12345";              // Password
input string   inp_db         = "world";              // Database name

//--- Connect MySQL transaction class
#include  <MySQL\MySQLTransaction.mqh>
CMySQLTransaction mysqlt;

//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
//--- Configure MySQL transaction class
   mysqlt.Config(inp_server,inp_port,inp_login,inp_password);
//--- Make a query
   string q = "select `Name`,`SurfaceArea` "+
              "from `"+inp_db+"`.`country` "+
              "where `Continent`='Oceania' "+
              "order by `SurfaceArea` desc limit 10";
   if(mysqlt.Query(q)==true)
     {
      if(mysqlt.Responses()!=1)
         return;
      CMySQLResponse *r = mysqlt.Response();
      if(r==NULL)
         return;
      Print("Name: ","Surface Area");
      uint rows = r.Rows();
      for(uint i=0; i<rows; i++)
        {
         double area;
         if(r.Row(i).Double("SurfaceArea",area)==false)
            break;
         PrintFormat("%s: %.2f",r.Row(i)["Name"],area);
        }
     }
   else
      if(GetLastError()==(ERR_USER_ERROR_FIRST+MYSQL_ERR_SERVER_ERROR))
        {
         // in case of a server error
         Print("MySQL Server Error: ",mysqlt.GetServerError().code," (",mysqlt.GetServerError().message,")");
        }
      else
        {
         if(GetLastError()>=ERR_USER_ERROR_FIRST)
            Print("Transaction Error: ",EnumToString(ENUM_TRANSACTION_ERROR(GetLastError()-ERR_USER_ERROR_FIRST)));
         else
            Print("Error: ",GetLastError());
        }
  }
```

Suppose that our task is to get a list of countries with the continent value of "Oceania" sorted by area from largest to smallest with the maximum of
10 items in the list. Let's perform the following actions:

- Declare an instance of the **mysqlt** transaction class

- Set connection parameters
- Make the appropriate query
- If the transaction is successful, make sure the number of responses is
equal to the expected value
- Get the pointer to the server response class
- Get the number of rows in the response
- Display the values of rows

The transaction may fail because of one of three reasons:

- A server error \- get its description using the [CMySQLTransaction::GetServerError](https://www.mql5.com/en/articles/7117#idgetservererror)()
method
- An internal error \- use the [EnumToString](https://www.mql5.com/en/docs/convert/enumtostring)()
function to get a description
- Otherwise, get the error code using [GetLastError](https://www.mql5.com/en/docs/check/getlasterror)()


If inputs are specified correctly, the result of the script operation is as follows:

![Test script operation result](https://c.mql5.com/2/39/test_mysql_script.png)

Fig. 10. Test script operation result

More complex examples applying multiple queries and the constant connection mode will be described in the
second part.

### Documentation

Contents

- [CMySQLTransaction transaction class](https://www.mql5.com/en/articles/7117#idcmysqltransaction)


- [Config](https://www.mql5.com/en/articles/7117#idconfig)
- [KeepAliveTimeout](https://www.mql5.com/en/articles/7117#idkeepalivetimeout)
- [PingPeriod](https://www.mql5.com/en/articles/7117#idpingperiod)
- [PingBeforeQuery](https://www.mql5.com/en/articles/7117#idpingbeforequery)
- [OnTimer](https://www.mql5.com/en/articles/7117#idontimer)
- [Handshake](https://www.mql5.com/en/articles/7117#idhandshake)
- [Query](https://www.mql5.com/en/articles/7117#idquery)
- [Responses](https://www.mql5.com/en/articles/7117#idresponses)
- [Response](https://www.mql5.com/en/articles/7117#idresponse)
- [GetServerError](https://www.mql5.com/en/articles/7117#idgetservererror)
- [RequestDuration](https://www.mql5.com/en/articles/7117#idrequestduration)
- [RxBytesTotal](https://www.mql5.com/en/articles/7117#idrxbytestotal)
- [TxBytesTotal](https://www.mql5.com/en/articles/7117#idtxbytestotal)
- [ResetBytesCounters](https://www.mql5.com/en/articles/7117#idresetbytescounters)

- [CMySQLLoginRequest authorization management class](https://www.mql5.com/en/articles/7117#idcmysqlloginrequest)


- [CMySQLResponse server response class](https://www.mql5.com/en/articles/7117#idcmysqlresponse)

- [MySQLServerError server error structure](https://www.mql5.com/en/articles/7117#idmysqlservererror)
- [CMySQLField field class](https://www.mql5.com/en/articles/7117#idcmysqlfield)
- [CMySQLRow row class](https://www.mql5.com/en/articles/7117#idcmysqlrow)


### CMySQLTransaction transaction class

List of CMySQLTransaction class methods

| Method | Action |
| --- | --- |
| [Config](https://www.mql5.com/en/articles/7117#idconfig) | Setting connection parameters |
| [KeepAliveTimeout](https://www.mql5.com/en/articles/7117#idkeepalivetimeout) | Setting timeout for Keep Alive mode in seconds |
| [PingPeriod](https://www.mql5.com/en/articles/7117#idpingperiod) | Setting a ping period for Keep Alive mode in seconds |
| [PingBeforeQuery](https://www.mql5.com/en/articles/7117#idpingbeforequery) | Enable/disable ping before a query |
| [OnTimer](https://www.mql5.com/en/articles/7117#idontimer) | Handling timer events (relevant when using Keep Alive) |
| [Handshake](https://www.mql5.com/en/articles/7117#idhandshake) | Getting the pointer to the class for working with authorization |
| [Query](https://www.mql5.com/en/articles/7117#idquery) | Sending a query |
| [Responses](https://www.mql5.com/en/articles/7117#idresponses) | Getting the number of server responses |
| [Response](https://www.mql5.com/en/articles/7117#idresponse) | Getting the pointer to the server response class |
| [GetServerError](https://www.mql5.com/en/articles/7117#idservererror) | Getting the server error structure |
| [RequestDuration](https://www.mql5.com/en/articles/7117#idrequestduration) | Transaction duration in microseconds |
| [RxBytesTotal](https://www.mql5.com/en/articles/7117#idrxbytestotal) | The counter of accepted bytes since the program launch |
| [TxBytesTotal](https://www.mql5.com/en/articles/7117#idtxbytestotal) | The counter of bytes sent since the program launch |
| [ResetBytesCounters](https://www.mql5.com/en/articles/7117#idresetbytescounters) | Resetting the counters of accepted and sent bytes |

Below is a brief description of each method.

### Config

Sets connection parameters.

```
bool  Config(
   string host,         // server name
   uint port,           // port
   string user,         // user name
   string password,     // password
   string base,         // database name
   uint keep_alive_tout // constant connection timeout (0 - not used)
   );
```

Return value: true if successful, otherwise - false (invalid symbols in string type arguments).

### KeepAliveTimeout

Enables the constant connection mode and sets its timeout. Timeout value is a time in seconds from the moment of sending the last query, after which
the connection is closed. If queries are repeated more often than the defined timeout value, the connection is not closed.

```
void  KeepAliveTimeout(
   uint tout            // set the constant connection timeout in seconds (0 - disable)
   );
```

### PingPeriod

Sets the period of sending 'ping' packets in the constant connection mode. It prevents the server from closing the connection. The ping is sent
after the specified time upon the last query or the previous ping.

```
void  PingPeriod(
   uint period          // set the ping period in seconds (for the constant connection mode)
   );
```

Return value: none.

### PingBeforeQuery

Enables sending the 'ping' packet before a query. Connection may be closed or terminated for some reason in the constant connection mode in time
intervals between queries. In this case, it is possible to send ping to the MySQL server to make sure the connection is active before sending a
query.

```
void  PingBeforeQuery(
   bool st              // enable (true)/disable (false) ping before a query
   );
```

Return value: none.

### OnTimer

Used in the constant connection mode. The method should be called from the [OnTimer](https://www.mql5.com/en/docs/event_handlers/ontimer)
event handler. The timer period should not exceed the minimum value of the KeepAliveTimeout and PingPeriod periods.

```
void  OnTimer(void);
```

Return value: none.

### Handshake

Gets the pointer to the class for working with authorization. It can be used to set the flags of client capabilities and the maximum packet size
before establishing a connection to the server. After the authorization, it allows receiving the version and the flags of the server
capabilities.

```
CMySQLLoginRequest *Handshake(void);
```

Return value: pointer to the CMySQLLoginRequest class for working with authorization.

### Query

Sends a query.

```
bool  Query(
   string q             // query body
   );
```

Return value: execution result; successful - true, error - false.

### Responses

Gets the number of responses.

```
uint  Responses(void);
```

Return value: number of server responses.

Packets of "Ok" or "Data" types are considered responses. If the query is executed successfully, one or more responses (for multiple queries) are
accepted.

### Response

Gets the pointer to the MySQL server response class

```
CMySQLResponse  *Response(
   uint idx                     // server response index
   );
```

Return value: pointer to the CMySQLResponse server response class. Passing an invalid value as an argument returns NULL.

The overloaded method without specifying an index is equivalent to Response(0).

```
CMySQLResponse  *Response(void);
```

Return value: pointer to the CMySQLResponse server response class. If there are no responses, NULL is returned.

### GetServerError

Gets the structure storing the code and the server error message. It can be called after the transaction class returns the
MYSQL\_ERR\_SERVER\_ERROR error.

```
MySQLServerError  GetServerError(void);
```

Return value: MySQLServerError error structure

### RequestDuration

Gets the request execution duration.

```
ulong  RequestDuration(void);
```

Return value: query duration in microseconds from the moment of sending till the end of handling

### RxBytesTotal

Gets the number of accepted bytes.

```
ulong  RxBytesTotal(void);
```

Return value: number of accepted bytes (TCP level) since the program launch. The ResetBytesCounters method is used for a reset.

### TxBytesTotal

Gets the number of sent bytes.

```
ulong  TxBytesTotal(void);
```

Return value: number of passed bytes (TCP level) since the program launch. The ResetBytesCounters method is used for a reset.

### ResetBytesCounters

Resets the counters of accepted and sent bytes.

```
void  ResetBytesCounters(void);
```

### CMySQLLoginRequest authorization management class

CMySQLLoginRequest class methods

| Method | Action |
| --- | --- |
| SetClientCapabilities | Sets [the \<br> client capabilities flags](https://www.mql5.com/go?link=https://dev.mysql.com/doc/dev/mysql-server/latest/group__group__cs__capabilities__flags.html "https://dev.mysql.com/doc/dev/mysql-server/latest/group__group__cs__capabilities__flags.html"). Predefined value: **0x** **005FA685** |
| SetMaxPacketSize | Sets the maximum allowable packet size in bytes. Predefined value: **16777215** |
| SetCharset | Defines [the \<br> set of used symbols](https://www.mql5.com/go?link=https://dev.mysql.com/doc/internals/en/character-set.html "https://dev.mysql.com/doc/internals/en/character-set.html"). Predefined value: **8** |
| Version | Returns the MySQL server version. For example: "5.7.21-log". |
| ThreadId | Returns the current connection thread ID. It corresponds to the [CONNECTION\_ID](https://www.mql5.com/go?link=https://dev.mysql.com/doc/refman/8.0/en/information-functions.html%23function_connection-id "https://dev.mysql.com/doc/refman/8.0/en/information-functions.html#function_connection-id")<br> value. |
| ServerCapabilities | Gets the flags of the server capabilities |
| ServerLanguage | Returns the encoding and the database representation [ID](https://www.mql5.com/go?link=https://dev.mysql.com/doc/internals/en/character-set.html "https://dev.mysql.com/doc/internals/en/character-set.html") [https://dev.mysql.com/doc/refman/8.0/en/charset.html](https://www.mql5.com/go?link=https://dev.mysql.com/doc/refman/8.0/en/charset.html "https://dev.mysql.com/doc/refman/8.0/en/charset.html") |

### CMySQLResponse server response class

A packet of "Ok" or "Data" type is considered a server response. Given that they differ significantly, the class has a separate set of
methods for working with each type of packets.

General CMySQLResponse class methods:

| Method | Return Value |
| --- | --- |
| Type | Server response type: MYSQL\_RESPONSE\_DATA or MYSQL\_RESPONSE\_OK |

Methods for Data type packets:

| Method | Return Value |
| --- | --- |
| Fields | Number of fields |
| Field | Pointer to the [field class](https://www.mql5.com/en/articles/7117#idcmysqlfield) by index <br> (overloaded method - getting the field index by name) |
| Field | Field index by name |
| Rows | Number of rows in a server response |
| Row | The pointer to a row class by index |
| Value | String value by row and field indices |
| Value | String value by row index and field name |
| ColumnToArray | The result of reading a column to the **string** type array |
| ColumnToArray | The result of reading a column to the **int** type array with type verification |
| ColumnToArray | The result of reading a column to the **long** type array with type verification |
| ColumnToArray | The result of reading a column to the **double** type array with type verification |

Methods for "Ok" type packets:

| Method | Return Value |
| --- | --- |
| AffectedRows | Number of rows affected by the last operation |
| LastId | [LAST\_INSERT\_ID](https://www.mql5.com/go?link=https://dev.mysql.com/doc/refman/8.0/en/information-functions.html%23function_last-insert-id "https://dev.mysql.com/doc/refman/8.0/en/information-functions.html#function_last-insert-id")<br> value |
| ServerStatus | [Server \<br> status](https://www.mql5.com/go?link=https://dev.mysql.com/doc/internals/en/status-flags.html "https://dev.mysql.com/doc/internals/en/status-flags.html") flags |
| Warnings | Number of warnings |
| Message | Server text message |

### MySQLServerError server error structure

MySQLServerError structure elements

| Element | Type | Purpose |
| --- | --- | --- |
| code | ushort | Error code |
| sqlstate | uint | State |
| message | string | Server text message |

### CMySQLField field class

CMySQLField class methods

| Method | Return Value |
| --- | --- |
| Catalog | Name of a directory a table belongs to |
| Database | Name of a database a table belongs to |
| Table | Pseudonym of a table a field belongs to |
| OriginalTable | Original name of a table a field belongs to |
| Name | Field pseudonym |
| OriginalName | Original field name |
| Charset | Applied encoding number |
| Length | Value length |
| Type | [Value \<br> type](https://www.mql5.com/go?link=https://dev.mysql.com/doc/internals/en/com-query-response.html%23column-type "https://dev.mysql.com/doc/internals/en/com-query-response.html#column-type") |
| Flags | Flags defining value attributes |
| Decimals | Allowed decimal places |
| MQLType | Field type in the form of the [ENUM\_DATABASE\_FIELD\_TYPE](https://www.mql5.com/en/docs/database/databasecolumntype#enum_database_field_type)<br> value (except for DATABASE\_FIELD\_TYPE\_NULL) |

### CMySQLRow row class

CMySQLRow class methods

| Method | Action |
| --- | --- |
| Value | Returns the field value by number as a string |
| operator\[\] | Returns the field value by name as a string |
| MQLType | Returns the field type by number as the [ENUM\_DATABASE\_FIELD\_TYPE](https://www.mql5.com/en/docs/database/databasecolumntype#enum_database_field_type)<br> value |
| MQLType | Returns the field type by name as the [ENUM\_DATABASE\_FIELD\_TYPE](https://www.mql5.com/en/docs/database/databasecolumntype#enum_database_field_type)<br> value |
| Text | Gets the field value by number as a string with type verification |
| Text | Gets the field value by name as a string with type verification |
| Integer | Gets the **int** type value by field number with type verification |
| Integer | Gets the **int** type value by field name with type verification |
| Long | Gets the **long** type value by field number with type verification |
| Long | Gets the **long** type value by field name with type verification |
| Double | Gets the **double** type value by field number with type verification |
| Double | Gets the **double** type value by field name with type verification |
| Blob | Gets the value in the form of the **uchar array** by field number with type verification |
| Blob | Gets the value in the form of the **uchar array** by field name with type verification |

**Note**. Type verification means that the readable field of the method working with **int** type should be equal
to DATABASE\_FIELD\_TYPE\_INTEGER. In case of a mismatch, no value is received and the method returns 'false'. Converting [MySQL \\
field type IDs](https://www.mql5.com/go?link=https://dev.mysql.com/doc/internals/en/com-query-response.html%23column-type "https://dev.mysql.com/doc/internals/en/com-query-response.html#column-type") to [ENUM\_DATABASE\_FIELD\_TYPE](https://www.mql5.com/en/docs/database/databasecolumntype#enum_database_field_type)
type value is implemented in the CMySQLField::MQLType() method whose source code is provided below.

```
//+------------------------------------------------------------------+
//| Return the field type as the ENUM_DATABASE_FIELD_TYPE value      |
//+------------------------------------------------------------------+
ENUM_DATABASE_FIELD_TYPE CMySQLField::MQLType(void)
  {
   switch(m_type)
     {
      case 0x00:  // decimal
      case 0x04:  // float
      case 0x05:  // double
      case 0xf6:  // newdecimal
         return DATABASE_FIELD_TYPE_FLOAT;
      case 0x01:  // tiny
      case 0x02:  // short
      case 0x03:  // long
      case 0x08:  // longlong
      case 0x09:  // int24
      case 0x10:  // bit
      case 0x07:  // timestamp
      case 0x0c:  // datetime
         return DATABASE_FIELD_TYPE_INTEGER;
      case 0x0f:  // varchar
      case 0xfd:  // varstring
      case 0xfe:  // string
         return DATABASE_FIELD_TYPE_TEXT;
      case 0xfb:  // blob
         return DATABASE_FIELD_TYPE_BLOB;
      default:
         return DATABASE_FIELD_TYPE_INVALID;
     }
  }
```

### Conclusion

In this article, we examined the use of functions for working with sockets using the implementation of the MySQL connector as an example.
This has been a theory. The second part of the article is to be of more practical nature: we will develop a service for collecting signal
properties and a program for viewing changes in them.

The attached archive contains the following files:

- **Include\\MySQL\** path: connector source codes

- **Scripts\\test\_mysql.mq5** file: the example of using the connector considered in the [Application](https://www.mql5.com/en/articles/7117#idusage)
section.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/7117](https://www.mql5.com/ru/articles/7117)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/7117.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/7117/mql5.zip "Download MQL5.zip")(23.17 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Applying network functions, or MySQL without DLL: Part II - Program for monitoring changes in signal properties](https://www.mql5.com/en/articles/7495)
- [Using OpenCL to test candlestick patterns](https://www.mql5.com/en/articles/4236)
- [Drawing Dial Gauges Using the CCanvas Class](https://www.mql5.com/en/articles/1699)
- [Liquid Chart](https://www.mql5.com/en/articles/1208)
- [Working with GSM Modem from an MQL5 Expert Advisor](https://www.mql5.com/en/articles/797)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/339096)**
(10)


![Phillip Kruger](https://c.mql5.com/avatar/2020/3/5E6AAB1F-344D.jpeg)

**[Phillip Kruger](https://www.mql5.com/en/users/nenalemao)**
\|
4 May 2020 at 17:57

Very Good Article, Thank you!


![Dmitri Custurov](https://c.mql5.com/avatar/2015/12/5679D267-1342.png)

**[Dmitri Custurov](https://www.mql5.com/en/users/dimarik1987)**
\|
7 Oct 2022 at 22:39

[![](https://c.mql5.com/3/394/4090747086273__1.png)](https://c.mql5.com/3/394/4090747086273.png "https://c.mql5.com/3/394/4090747086273.png")

Strange, but in the config method there is no input parameter "base", i.e. database name. It is not possible to connect.

![Zbigniew Mirowski](https://c.mql5.com/avatar/2014/9/542002BF-0B35.jpg)

**[Zbigniew Mirowski](https://www.mql5.com/en/users/ziom)**
\|
3 Feb 2023 at 15:02

Great article but it looks like not all functions are implemented or the buid 3550  already changes some features

```
  m_id       =m_wnd.LastId()+1;
   m_chart_id =m_wnd.ChartId();
   m_subwin   =m_wnd.SubwindowNumber();
   m_corner   =(ENUM_BASE_CORNER)m_wnd.Corner();
   m_anchor   =(ENUM_ANCHOR_POINT)m_wnd.Anchor();

'LastId' - undeclared identifier        Element.mqh     841     22
')' - expression expected       Element.mqh     841     29
'ChartId' - undeclared identifier       Element.mqh     842     22
')' - expression expected       Element.mqh     842     30
'SubwindowNumber' - undeclared identifier       Element.mqh     843     22
')' - expression expected       Element.mqh     843     38
'Corner' - undeclared identifier        Element.mqh     844     40
')' - expression expected       Element.mqh     844     47
'Anchor' - undeclared identifier        Element.mqh     845     41
')' - expression expected       Element.mqh     845     48
```

![Viktor Vasilyuk](https://c.mql5.com/avatar/2017/1/586D3F1F-3D97.jpg)

**[Viktor Vasilyuk](https://www.mql5.com/en/users/progma137)**
\|
9 Mar 2023 at 10:53

**Dmitri Custurov [#](https://www.mql5.com/ru/forum/330915#comment_42541164):**

Strange, but in the config method there is no input parameter "base", i.e. database name. It is not possible to connect.

```
select c1, c2 from `db_name`.`table_name`;
```

![Anton Rakhmanov](https://c.mql5.com/avatar/avatar_na2.png)

**[Anton Rakhmanov](https://www.mql5.com/en/users/holodar)**
\|
2 Sep 2023 at 16:43

Thank you very much to the author. The article is excellent and the code is interesting. However, I encountered a problem with error 4014.

What functionality should I enable in the terminal to avoid the 4014 error? The server and the database itself are on a local machine (localhost).

![Library for easy and quick development of MetaTrader programs (part XXXIV): Pending trading requests - removing and modifying orders and positions under certain conditions](https://c.mql5.com/2/38/MQL5-avatar-doeasy__2.png)[Library for easy and quick development of MetaTrader programs (part XXXIV): Pending trading requests - removing and modifying orders and positions under certain conditions](https://www.mql5.com/en/articles/7569)

In this article, we will complete the description of the pending request trading concept and create the functionality for removing pending orders, as well as modifying orders and positions under certain conditions. Thus, we are going to have the entire functionality enabling us to develop simple custom strategies, or rather EA behavior logic activated upon user-defined conditions.

![How to create 3D graphics using DirectX in MetaTrader 5](https://c.mql5.com/2/39/MQL5-avatar-directx_yellow.png)[How to create 3D graphics using DirectX in MetaTrader 5](https://www.mql5.com/en/articles/7708)

3D graphics provide excellent means for analyzing huge amounts of data as they enable the visualization of hidden patterns. These tasks can be solved directly in MQL5, while DireсtX functions allow creating three-dimensional object. Thus, it is even possible to create programs of any complexity, even 3D games for MetaTrader 5. Start learning 3D graphics by drawing simple three-dimensional shapes.

![Applying network functions, or MySQL without DLL: Part II - Program for monitoring changes in signal properties](https://c.mql5.com/2/37/kisspng-computer-icons-application-programming-interface-c-database-administrator-icon-free-download__1.png)[Applying network functions, or MySQL without DLL: Part II - Program for monitoring changes in signal properties](https://www.mql5.com/en/articles/7495)

In the previous part, we considered the implementation of the MySQL connector. In this article, we will consider its application by implementing the service for collecting signal properties and the program for viewing their changes over time. The implemented example has practical sense if users need to observe changes in properties that are not displayed on the signal's web page.

![Library for easy and quick development of MetaTrader programs (part XXXIII): Pending trading requests - closing positions under certain conditions](https://c.mql5.com/2/38/MQL5-avatar-doeasy__1.png)[Library for easy and quick development of MetaTrader programs (part XXXIII): Pending trading requests - closing positions under certain conditions](https://www.mql5.com/en/articles/7554)

We continue the development of the library functionality featuring trading using pending requests. We have already implemented sending conditional trading requests for opening positions and placing pending orders. In the current article, we will implement conditional position closure – full, partial and closing by an opposite position.

[![](https://www.mql5.com/ff/sh/592yc11u3j4rs5z9z2/01.png)How AI helps create robots for MetaTrader 5Learn from our book "Neural Networks in Algo Trading with MQL5"Read](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/neurobook%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.neurobook%26utm_content=visit.page%26utm_campaign=neurobook.promo.04.2024&a=ghrobswocqgvhztzjldphupateyllpro&s=9929cb0b8629585b5a42fabc06c525e41f6c0ebdf3045d044a5413b93ea88b47&uid=&ref=https://www.mql5.com/en/articles/7117&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5083334175894804881)

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