---
title: Developing an MQTT client for MetaTrader 5: a TDD approach
url: https://www.mql5.com/en/articles/12857
categories: Integration, Indicators, Expert Advisors
relevance_score: 6
scraped_at: 2026-01-23T17:22:48.049014
---

[![](https://www.mql5.com/ff/sh/zf7a2k61x98jzh89z2/01.png)Speed up your tradingUse our high-speed VPS for MetaTrader 4 and 5Learn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=qtrrsuiwuicrscmckjynyanztbditglq&s=c617dc80d90cfd3783ec1345eec2b419b281f10fec6eac77b3218984ac337259&uid=&ref=https://www.mql5.com/en/articles/12857&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5068155662686418485)

MetaTrader 5 / Integration


### Introduction

" _... the strategy is definitely: first make it work, then make it right, and, finally, make it fast._"  Stephen C. Johnson and Brian W. Kernighan's "The C Language and Models for Systems Programming" in Byte magazine (August 1983)

To share real-time data between two or more instances of MetaTrader is a common need among traders and account managers. Arguably, the most demanded data sharing is related to trade transactions by the so-called “trade copiers”. But one can easily find requests for account info sharing, symbol screening, and statistical data for [machine learning](https://www.mql5.com/en/articles/10431), to name a few. This functionality can be obtained by means of using network [sockets](https://www.mql5.com/en/docs/network), inter-process communication with [named pipes](https://www.mql5.com/en/articles/115), web services, local file sharing, and possibly other solutions that have already been tested and/or developed.

As is the norm in software development, each one of these solutions has its advantages and disadvantages in terms of usability, stability, trustability, and resources required for development and maintenance. In short, each one presents a different cost-benefit relation depending on the user requirements and budget.

This article reports the first steps in the implementation of the MQTT protocol client side, which happens to be a technology designed to meet precisely this need – real-time data sharing between machines – with high performance, low bandwidth consumption, low resource requirements, and low cost.

### What is MQTT

"MQTT is a Client Server publish/subscribe messaging transport protocol. It is lightweight, open, simple, and designed to be easy to implement. These characteristics make it ideal for use in many situations, including constrained environments such as for communication in Machine to Machine (M2M) and Internet of Things (IoT) contexts where a small code footprint is required and/or network bandwidth is at a premium."

The above definition is from [OASIS](https://www.mql5.com/go?link=https://docs.oasis-open.org/mqtt/mqtt/v5.0/mqtt-v5.0.html "https://docs.oasis-open.org/mqtt/mqtt/v5.0/mqtt-v5.0.html"), the owner, and developer of the protocol as an open standard since 2013.

"In 2013, IBM submitted MQTT v3.1 to the OASIS specification body with a charter that ensured only minor changes to the specification could be accepted. After taking over maintenance of the standard from IBM, OASIS released version 3.1.1 on October 29, 2014. A more substantial upgrade to MQTT version 5, adding several new features, was released on March 7, 2019." ( [Wikipedia](https://en.wikipedia.org/wiki/MQTT "https://en.wikipedia.org/wiki/MQTT"))

IBM started the protocol development in 1999 to address the industrial need to monitor oil pipelines with sensors and to send the data via satellite to remote control centers. According to Arlen Nipper, co-creator of the protocol with Dr. Andy Stanford-Clark, the goal was to provide a **real-time stream of data** to those control centers.

“And what we tried to do was take at the time what was IBM's middleware, MQ Integrator, and at the time what I was doing in the communications industry talking over 1,200-baud dial-up lines and 300-baud dial-up lines and very bandwidth-constrained VSAT and tie those two together.“

Despite the fact that it was designed to be robust, fast, and cheap due to tech stack limitations and expensive network costs, it was required to provide **quality of service** data delivery with continuous session awareness, which allows it to cope with unreliable or even intermittent internet connections.

As a **binary protocol**, MQTT is very efficient in terms of memory and processing requirements. It is even curious that the smallest MQTT packet has only two bytes!

Since it is based on a **publish/subscribe** model (pub/sub), instead of a request/response, MQTT is bi-directional. What is to say, once a client/server connection is established, data can flow from client to server and from server to client at any time, without the need of a previous request, as is the case with HTTP [WebRequest](https://www.mql5.com/en/docs/network/webrequest). Once data arrives, the server immediately forwards it to the recipients. This feature is a cornerstone of real-time data exchange since it allows for the minimum latency between endpoints. Some vendors advertise latency in the order of milliseconds.

The type, format, codec, or anything else about data does not matter. MQTT is **data-agnostic**. The user can send/receive from raw bytes to text formats (XML, JSON objects), Protocol Buffers, pictures, video fragments, etc.

Most of the interactions between the client and the server can be **asynchronous**, which is to say MQTT is scalable. In the IoT industry it is not uncommon to talk about thousands or even millions of devices connected and exchanging data in real-time.

Its messages can be, and usually are, encryptedbetween endpoints since the protocol is **TLS compatible**, with **authentication** and **authorization** mechanisms built-in.

Not surprisingly, MQTT is not only a set of high-standard specifications, but it is a widely adopted technology in several industries

“MQTT today is used in a wide variety of industries, such as automotive, manufacturing, telecommunications, oil and gas, etc.” ( [mqtt.org](https://www.mql5.com/go?link=https://mqtt.org/ "https://mqtt.org/"))

### **Main Components**

The pub/sub is a very well-known message exchange model. A client connects to the server and **publishes** a **message** on a **topic**. Thereafter, all clients **subscribed** to that topic receive the message(s). This is the basic mechanism of the model.

The server acts as a broker, standing between the clients to receive both subscriptions and publications. TCP/IP is the underlying transport protocol and clients are any device that understands TCP/IP and MQTT. The message is usually a JSON or XML payload but can be anything, including raw bytes sequence.

The topic is a UTF-8 encoded string used to describe a namespace-like hierarchical structure

- office/machine01/account123456

- office/machine02/account789012

- home/machine01/account345678


We can also use a hash (#) as a wildcard character to subscribe to a topic. For example, to subscribe to all accounts in machine01 from home:

- home/machine01/#

Or  to subscribe to all machines from the office:

- office/#

OK, so MQTT was developed for Machine to Machine conversation, it is used extensively in the IoT context, and it is robust, fast, and cheap. But you may be asking: what kind of benefit or improvement can this thing bring to a trading environment? What could be the use cases for MQTT in MetaTrader?

As said above, “trade copiers” are the most obvious use case for MQTT in a trading environment. But one may think about feeding machine learning pipelines with real-time data, changing an EA behavior according to real-time data pulled from a web service, or remotely controlling your MetaTrader application from any device.

For any scenario where a **real-time stream of data** between machines is needed, we may take MQTT  into consideration.

### How to use MQTT in MetaTrader

There are free and open-source MQTT [client libraries](https://en.wikipedia.org/wiki/Comparison_of_MQTT_implementations "https://en.wikipedia.org/wiki/Comparison_of_MQTT_implementations") for the most popular general-purpose languages including the respective variants for mobile and embedded devices. So, in order to use MQTT from MQL5 we can generate and import a respective DLL from C, C++, or C#.

If the data to be shared is limited to trade transactions/account info and a relatively greater latency is acceptable, another option would be to use the Python MQTT client library and the MQL5 [Python module](https://www.mql5.com/en/articles/5691) as a “bridge”.

But as we know, the use of DLLs has some negative implications on the MQL5 ecosystem, the most notable being that the marketplace does not accept DLL-dependent EAs. Also, DLL-dependent EAs are not allowed to run backtest optimizations on the MQL5 cloud. To avoid DLL dependency and the Python bridge, the ideal solution is to develop a native MQTT library, client-side, for MetaTrader.

This is what we will be doing in the next weeks: implementing the MQTT-v5.0 protocol, client-side, for MetaTrader 5.

Implementing an MQTT client can be considered “relatively easy” when compared with other network protocols. But relatively easy is not necessarily easy. So we’ll start with a bottom-up approach, test-driven development (TDD), hopefully with community tests and feedback.

Although TDD can be (and often is) used as a “hype” or “buzzword” for almost anything, it fits very well when we have a set of formalized specifications, which is precisely the case of a standardized network protocol.

By adopting a bottom-up approach we can face huge specs by breaking it in, let’s say, baby steps. The [specs](https://www.mql5.com/go?link=https://docs.oasis-open.org/mqtt/mqtt/v5.0/mqtt-v5.0.html "https://docs.oasis-open.org/mqtt/mqtt/v5.0/mqtt-v5.0.html") for MQTT are not really huge and the client side is the easiest when compared to the broker side. But it has its own complexities, in particular since version 5.0 with the inclusion of some additional features.

Since we do not have paid time and a team to combine skills, baby steps seem to be the best way here: how do I send a message? What kind of thing am I supposed to write? How could I start with something that works, so I can improve it to work well before thinking about making it work fast?

### Huge Specs, Baby Steps: Understanding and breaking-down the MQTT Protocol

As is the norm for many (if not all) network protocols, the MQTT protocol works by means of breaking the data to be transmitted in so-called packets. Thus, if the receiver knows what each kind of packet means, it can adopt the proper operational behavior depending on the kind of packet received. In MQTT parlance, the kind of the packet is called **Control Packet Type** and each of them has up to three parts:

- a **fixed header** present in all packets

- a **variable header** present in some packets

- a **payload** also present only in some packets


There are fifteen Control Packet Types in MQTT-v5.0:

Table 1. MQTT Control Packet types (table from OASIS spec)

| Name | Value | Direction of flow | Description |
| --- | --- | --- | --- |
| Reserved | 0 | Forbidden | Reserved |
| CONNECT | 1 | Client to Server | Connection request |
| CONNACK | 2 | Server to Client | Connect acknowledgment |
| PUBLISH | 3 | Client to Server or Server to Client | Publish message |
| PUBREC | 5 | Client to Server or Server to Client | Publish received (QoS 2 delivery part 1) |
| PUBREL | 6 | Client to Server or Server to Client | Publish release (QoS 2 delivery part 2) |
| PUBCOMP | 7 | Client to Server or Server to Client | Publish complete (QoS 2 delivery part 3) |
| SUBSCRIBE | 8 | Client to Server | Subscribe request |
| SUBACK | 9 | Server to Client | Subscribe acknowledgment |
| UNSUBSCRIBE | 10 | Client to Server | Unsubscribe request |
| UNSUBACK | 11 | Server to Client | Unsubscribe acknowledgment |
| PINGREQ | 12 | Client to Server | PING request |
| PINGRESP | 13 | Server to Client | PING response |
| DISCONNECT | 14 | Client to Server or Server to Client | Disconnect notification |
| AUTH | 15 | Client to Server or Server to Client | Authentication exchange |

The fixed header of all Control Packets has the same format.

Fig.1 MQTT Fixed Header Format

![MQTT Fixed Header Format](https://c.mql5.com/2/55/fixed-header-format.PNG)

Since we can do nothing before we have a connection between our Client and the Server and considering that the standard has a clear statement that reads

“After a Network Connection is established by a Client to a Server, the first packet sent from the Client to the Server MUST be a CONNECT packet”,

Let's see how the fixed header of a CONNECT packet must be formatted.

### Fig.2 MQTT Fixed Header Format for CONNECT Packet

![MQTT Fixed Header Format of CONNECT Packet](https://c.mql5.com/2/55/fixed-header-format-connect.PNG)

So we need to fill it with two bytes: the first byte must have the binary value 00010000, and the second byte must have the value of the so-called “Remaining Length”.

The standard defines the Remaining Length as

“a Variable Byte Integer that represents the number of bytes remaining within the current Control Packet, including data in the Variable Header and the Payload. The Remaining Length does not include the bytes used to encode the Remaining Length. The packet size is the total number of bytes in an MQTT Control Packet, this is equal to the length of the Fixed Header plus the Remaining Length.” (emphasis is ours)

The standard also defines the encoding scheme for this Variable Byte Integer.

“The Variable Byte Integer is encoded using an encoding scheme which uses a single byte for values up to 127. Larger values are handled as follows. The least significant seven bits of each byte encode the data, and the most significant bit is used to indicate whether there are bytes following in the representation. Thus, each byte encodes 128 values and a "continuation bit". The maximum number of bytes in the Variable Byte Integer field is four. The encoded value MUST use the minimum number of bytes necessary to represent the value”

Wow! This seems to be a lot of information to be assimilated at once. And we are just trying to fill the second byte!

Fortunately, the standard provides the “algorithm for encoding a non-negative integer (X) into the Variable Byte Integer encoding scheme”.

```
do
   encodedByte = X MOD 128
   X = X DIV 128
   // if there are more data to encode, set the top bit of this byte
   if (X > 0)
      encodedByte = encodedByte OR 128
   endif
   'output' encodedByte
while (X > 0)
```

“Where MOD is the modulo operator (% in C), DIV is integer division (/ in C), and OR is bitwise or (\| in C).”

OK. Now we have:

- the list of all Control Packet types,

- the format of the fixed header for a CONNECT packet with two bytes,

- the value of the first byte,

- and the algorithm to encode the Variable Byte Integer that will fill the second byte.

We can start writing our first test.

NOTE: since we are adopting a bottom-up, TDD approach, we will be writing the tests before the implementation. We can assume from the start that we’ll be 1) writing tests that fail, then we’ll 2) implement only the code required to pass the test, then we’ll 3) refactor the code if needed. It does not matter if the initial implementation is naive, ugly, or if it seems to have bad performance. We’ll deal with these issues once we have code that works. Performance is at the end of our task list.

Without further ado, let’s open our MetaEditor and create a script named _TestFixedHeader_ with the following content.

```
#include <MQTT\mqtt.mqh>
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
//---
   Print(TestFixedHeader_Connect());
  }
//---
bool TestFixedHeader_Connect()
  {
   uchar content_buffer[]; //empty
//---
   uchar expected[2];
   expected[0] = 1; //pkt type
   expected[1] = 0; //remaining length
//---
   uchar fixed_header[];
//---
   GenFixedHeader(CONNECT, content_buffer, fixed_header);
//---
   if(!ArrayCompare(expected, fixed_header) == 0)
     {
      Print(__FUNCTION__);
      for(uint i = 0; i < expected.Size(); i++)
        {
         Print("expected: ", expected[i], " result: ", fixed_header[i]);
        }
      return false;
     }
   return true;
  }
```

Also, create the _mqtt.mqh_ header where we’ll start developing our functions and fill it with the code below.

```
void GenFixedHeader(uint pkt_type, uchar& buf[], uchar& head[])
  {
   ArrayFree(head);
   ArrayResize(head, 2);
//---
   head[0] = uchar(pkt_type);
//---
//Remaining Length
   uint x;
   x = ArraySize(buf);
   do
     {
      uint encodedByte = x % 128;
      x = (uint)(x / 128);
      if(x > 0)
        {
         encodedByte = encodedByte | 128;
        }
      head[1] = uchar(encodedByte);
     }
   while(x > 0);
  }
//+------------------------------------------------------------------+
enum ENUM_PKT_TYPE
  {
   CONNECT      =  1, // Connection request
   CONNACK      =  2, // Connect acknowledgment
   PUBLISH      =  3, // Publish message
   PUBACK       =  4, // Publish acknowledgment (QoS 1)
   PUBREC       =  5, // Publish received (QoS 2 delivery part 1)
   PUBREL       =  6, // Publish release (QoS 2 delivery part 2)
   PUBCOMP      =  7, // Publish complete (QoS 2 delivery part 3)
   SUBSCRIBE    =  8, // Subscribe request
   SUBACK       =  9, // Subscribe acknowledgment
   UNSUBSCRIBE 	=  10, // Unsubscribe request
   UNSUBACK     =  11, // Unsubscribe acknowledgment
   PINGREQ      =  12, // PING request
   PINGRESP     =  13, // PING response
   DISCONNECT  	=  14, // Disconnect notification
   AUTH         =  15, // Authentication exchange
  };
```

By running the script you should see the following in your experts tab.

Fig.3 Output Test Fixed Header - Test Pass

![Output Test Fixed Header - True](https://c.mql5.com/2/55/test-output-fixed-header-connect-true.PNG)

To be sure our test is working, we need to see it fail. So you are strongly encouraged to modify the input represented by the _content\_buffer_ variable while leaving the expected variable unchanged. You should see something like the following in your experts tab output.

Fig.4 Output Test Fixed Header - Test Fail

![Output Test Fixed Header - Test Fail](https://c.mql5.com/2/55/test-output-fixed-header-connect-false.PNG)

Anyway, we can assume that our tests are fragile at this point, as is our code in the _mqtt.mqh_ header. No problem. We are just starting and as we move forward we’ll have the opportunity to make them better, learn from our mistakes and, as a consequence, improve our skills.

By now we can replicate the _TestFixedHeader\_Connect_ function to other packet types. We’ll be ignoring those that have the direction of flow Server to Client only. They are CONNACK, PUBACK, SUBACK,  UNSUBACK, and PINGRESP. These ACK(S) and ping response packet headers will be generated by the Server and we’ll be dealing with them later.

In order to be sure our tests are working as expected, we need to include tests that are expected to fail. These tests will return true on fail.

```
#include <MQTT\mqtt.mqh>
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
//---
   Print(TestFixedHeader_Connect());
   Print(TestFixedHeader_Connect_RemainingLength1_Fail());
   Print(TestFixedHeader_Publish());
   Print(TestFixedHeader_Publish_RemainingLength1_Fail());
   Print(TestFixedHeader_Puback());
   Print(TestFixedHeader_Puback_RemainingLength1_Fail());
   Print(TestFixedHeader_Pubrec());
   Print(TestFixedHeader_Pubrec_RemainingLength1_Fail());
   Print(TestFixedHeader_Pubrel());
   Print(TestFixedHeader_Pubrel_RemainingLength1_Fail());
   Print(TestFixedHeader_Pubcomp());
   Print(TestFixedHeader_Pubcomp_RemainingLength1_Fail());
   Print(TestFixedHeader_Subscribe());
   Print(TestFixedHeader_Subscribe_RemainingLength1_Fail());
   Print(TestFixedHeader_Puback());
   Print(TestFixedHeader_Puback_RemainingLength1_Fail());
   Print(TestFixedHeader_Unsubscribe());
   Print(TestFixedHeader_Unsubscribe_RemainingLength1_Fail());
   Print(TestFixedHeader_Pingreq());
   Print(TestFixedHeader_Pingreq_RemainingLength1_Fail());
   Print(TestFixedHeader_Disconnect());
   Print(TestFixedHeader_Disconnect_RemainingLength1_Fail());
   Print(TestFixedHeader_Auth());
   Print(TestFixedHeader_Auth_RemainingLength1_Fail());
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool TestFixedHeader_Connect()
  {
   uchar content_buffer[]; //empty
//---
   uchar expected[2];
   expected[0] = 1; //pkt type
   expected[1] = 0; //remaining length
//---
   uchar fixed_header[];
//---
   GenFixedHeader(CONNECT, content_buffer, fixed_header);
//---
   if(!ArrayCompare(expected, fixed_header) == 0)
     {
      Print(__FUNCTION__);
      for(uint i = 0; i < expected.Size(); i++)
        {
         Print("expected: ", expected[i], " result: ", fixed_header[i]);
        }
      return false;
     }
   Print(__FUNCTION__);
   return true;
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool TestFixedHeader_Connect_RemainingLength1_Fail()
  {
   uchar content_buffer[]; //empty
   ArrayResize(content_buffer, 1);
   content_buffer[0] = 1;
//---
   uchar expected[2];
   expected[0] = 1; //pkt type
   expected[1] = 0; //remaining length should be 1
//---
   uchar fixed_header[];
//---
   GenFixedHeader(CONNECT, content_buffer, fixed_header);
//---
   if(!ArrayCompare(expected, fixed_header) == 0)
     {
      Print(__FUNCTION__);
      for(uint i = 0; i < expected.Size(); i++)
        {
         Print("expected: ", expected[i], " result: ", fixed_header[i]);
        }
      return true;
     }
   Print(__FUNCTION__);
   return false;
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool TestFixedHeader_Publish()
  {
   uchar content_buffer[]; //empty
//---
   uchar expected[2];
   expected[0] = 3; //pkt type
   expected[1] = 0; //remaining length
//---
   uchar fixed_header[];
//---
   GenFixedHeader(PUBLISH, content_buffer, fixed_header);
//---
   if(!ArrayCompare(expected, fixed_header) == 0)
     {
      Print(__FUNCTION__);
      for(uint i = 0; i < expected.Size(); i++)
        {
         Print("expected: ", expected[i], " result: ", fixed_header[i]);
        }
      return false;
     }
   Print(__FUNCTION__);
   return true;
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool TestFixedHeader_Publish_RemainingLength1_Fail()
  {
   uchar content_buffer[]; //empty
   ArrayResize(content_buffer, 1);
   content_buffer[0] = 1;
//---
   uchar expected[2];
   expected[0] = 3; //pkt type
   expected[1] = 0; //remaining length should be 1
//---
   uchar fixed_header[];
//---
   GenFixedHeader(PUBLISH, content_buffer, fixed_header);
//---
   if(!ArrayCompare(expected, fixed_header) == 0)
     {
      Print(__FUNCTION__);
      for(uint i = 0; i < expected.Size(); i++)
        {
         Print("expected: ", expected[i], " result: ", fixed_header[i]);
        }
      return true;
     }
   Print(__FUNCTION__);
   return false;
  }
.
.
.
(omitted for brevity)
```

Hey! This is a lot of boilerplate code, dozens of typing, and or copy/paste!

Yes, it is, for sure. But it will have a good pay-off in the long term. With these simple – even simplistic – tests in place, we are building a kind of safety net for our development. They should help us

- to stay focused on the task at hand,

- avoid over-engineering,

- and to spot regression bugs.


NOTE: I strongly encourage you to write them by yourself instead of simply using the attached file. You will see how many tiny, “inoffensive” errors they can catch from the start. As we advance with the **Operational Behavior** of our Client, these tests (and other more specific tests) will prove their value. Besides that, we are avoiding a common technical debt: leaving the tests to be written at the end. Usually, tests will never be written if you leave them for the end.

Fig.5 Output Tests Fixed Header - All Pass

![Output Test Fixed Header - All Pass](https://c.mql5.com/2/55/test-output-fixed-header-all-true.PNG)

OK, let’s see if our two bytes CONNECT header is recognized as a valid header by an MQTT broker.

### How to Install an MQTT Broker (and Client) For Development and Testing

There are many production MQTT brokers available online and most of them offer some kind of “sandbox” URL for development and testing purposes. A simple search for “MQTT broker” on your favorite search engine should be enough to help you find some of them.

However, our Client is embrionary at this point. We are not able to receive and read a response without using a packet analyzer to catch our network traffic. This tool will be useful later, but for now, it is enough to have a _specs-compliant_ MQTT broker installed on our development machine, so we can check its logs to see the result of our interactions. Ideally, it should be installed on a virtual machine, in order to have an IP other than that of our Client. By using a broker with a different IP for development and testing we can address connections and authentication issues earlier.

Again, there are several options for Windows, Linux, and Mac. I’ve installed Mosquitto on Windows Subsystem For Linux (WSL). Besides being free and open-source, Mosquitto is very convenient because it comes with two very useful command-line applications for development: _mosquitto\_pub_ and _mosquitto\_sub_ to publish and subscribe to MQTT topics. I’ve also installed it on the Windows dev machine so I can cross-check some errors.

Remember that MetaTrader requires that you list any external URL in the _Tools > Options menu Expert Advisors_ tab and that you are only allowed to access ports 80 or 443 from MetaTrader. Thus, if you follow this path of installing the broker on WSL, do not forget to include its host IP and also do not forget to redirect the network traffic arriving at port 80 to 1883 which is the default MQTT (and Mosquitto) port. There is a tool called _redir_ that performs this port redirection in a simple and stable way.

Fig.6 MetaTrader 5 Dialog - Allow Web Request URL

![MetaTrader 5 Dialog - Allow Web Request URL](https://c.mql5.com/2/55/mt5-tools-options-allow-webrequest-url.jpg)

To get the WSL IP run the command below.

Fig.7 WSL Get The Hostname Command

![Fig.6 - WSL Get The Hostname Command](https://c.mql5.com/2/55/wsl-hostname.PNG)

Once installed, Mosquitto will be self-configured to start as a “Service” at boot. Thus, just reboot your WSL to start Mosquitto on default port 1883.

To redirect network traffic from port 80 to 1883 using _redir_, run the command below.

Fig.8 Redirect Network Traffic With 'redir'

![Port Redirect Using 'redir' Command-Line](https://c.mql5.com/2/55/wsl-redir.PNG)

And finally we can check if our two bytes CONNECT fixed header is recognized as a valid MQTT header by a specs-compliant MQTT broker. Just create a “scratch” script and paste the following code. (Do not forget to change the IP address in the _broker\_ip_ variable according to the output of your _get hostname -I_ command.)

```
#include <MQTT\mqtt.mqh>

string broker_ip = "172.20.155.236";
int broker_port = 80;
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
   int socket = SocketCreate();
   if(socket != INVALID_HANDLE)
     {
      if(SocketConnect(socket, broker_ip, broker_port, 1000))
        {
         Print("Connected ", broker_ip);
         //---
         uchar fixed_header[];
         uchar content_buffer[]; //empty
         //---
         GenFixedHeader(CONNECT, content_buffer, fixed_header);
         //---
         if(SocketSend(socket, fixed_header, ArraySize(fixed_header)) < 0)
           {
            Print("Failed sending fixed header ", GetLastError());
           }
        }
     }
  }
```

You should see the following in your Experts tab output...

Fig.9 Output Local Broker Connect

![Output Local Broker Connect](https://c.mql5.com/2/55/scratch-fixed-header-connect-output.PNG)

… and the following output in your Mosquitto log.

Fig.10 Output Local Broker Connect - Mosquitto Log

![Output Local Broker Connect - Mosquitto Log](https://c.mql5.com/2/55/wsl-mosquitto-log-tail.PNG)

So, yes, our CONNECT fixed header was recognized by Mosquito, but the <unknown> Client was disconnected immediately “due to protocol error”. The error occurred because we didn’t include the Variable Header yet, with the Protocol Name, Protocol Level, and other associated metadata. We will fix that in the next step.

NOTE: as you can see at the very start of the above command, we are using the _tail -f {pathToLogFile}_ command. We can use it during development to follow the Mosquito log updates without the need to be opening and reloading the file.

In the next step, we will be implementing the CONNECT Variable Header – and others – to maintain a stable connection with our Broker. We’ll also PUBLISH a message and deal with CONNACK packets returned by the Broker and their related Reason Codes. This next step will have some interesting bitwise operations to fill our Connect Flags. This next step will also require that we substantially improve our tests to cope with the complexities that will show up as a result of the Client-Broker conversation.

### Conclusion

In this article we saw a quick overview of the MQTT pub/sub real-time message sharing protocol, its origins, and main components. We also pointed out some possible use cases of MQTT for real-time messaging in a trading context and how to use it for automated operations in MetaTrader 5 either by importing DLLs generated from C, C++, or C#, or using the MQTT Python library via MetaTrader 5 Python module.

Considering the limitations imposed over the use of DLLs on the MetaQuotes marketplace and the MetaQuotes Cloud Tester, we also proposed and described our first steps on the implementation of a native MQL5 MQTT client making use of a Test-Driven Development  (TDD) approach.

**Some References that May Be Useful**

We do not need to reinvent all the wheels. Many of the solutions to the most common challenges developers face while writing MQTT clients for other languages are available as open-source libraries/SDKs.

- List of [software](https://www.mql5.com/go?link=https://mqtt.org/software/ "https://mqtt.org/software/"), including brokers, libraries, and tools.
- List of several [resources](https://www.mql5.com/go?link=https://github.com/mqtt/mqtt.org/wiki "List of several resources related to MQTT on GitHub") related to MQTT on GitHub.

If you are a seasoned MQL5 developer and have suggestions, please, drop a comment below. It will be very much appreciated.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/12857.zip "Download all attachments in the single ZIP archive")

[TestFixedHeader.mq5](https://www.mql5.com/en/articles/download/12857/testfixedheader.mq5 "Download TestFixedHeader.mq5")(19.48 KB)

[mqtt.mqh](https://www.mql5.com/en/articles/download/12857/mqtt.mqh "Download mqtt.mqh")(2.19 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Statistical Arbitrage Through Cointegrated Stocks (Part 10): Detecting Structural Breaks](https://www.mql5.com/en/articles/20946)
- [Statistical Arbitrage Through Cointegrated Stocks (Part 9): Backtesting Portfolio Weights Updates](https://www.mql5.com/en/articles/20657)
- [Statistical Arbitrage Through Cointegrated Stocks (Part 8): Rolling Windows Eigenvector Comparison for Portfolio Rebalancing](https://www.mql5.com/en/articles/20485)
- [Statistical Arbitrage Through Cointegrated Stocks (Part 7): Scoring System 2](https://www.mql5.com/en/articles/20173)
- [Statistical Arbitrage Through Cointegrated Stocks (Part 6): Scoring System](https://www.mql5.com/en/articles/20026)
- [Statistical Arbitrage Through Cointegrated Stocks (Part 5): Screening](https://www.mql5.com/en/articles/19626)
- [Statistical Arbitrage Through Cointegrated Stocks (Part 4): Real-time Model Updating](https://www.mql5.com/en/articles/19428)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/450255)**
(4)


![pperea21](https://c.mql5.com/avatar/avatar_na2.png)

**[pperea21](https://www.mql5.com/en/users/pperea21)**
\|
1 Nov 2023 at 01:03

Good article.

A few months ago I migrated the PubSubClient library to MQL5.

https://github.com/gmag11/MQTT-MQL5-Library

![DrPip83](https://c.mql5.com/avatar/avatar_na2.png)

**[DrPip83](https://www.mql5.com/en/users/drpip83)**
\|
6 Feb 2024 at 23:42

Hi Jocimar,

Thank you for this amazing tutorial. I have a trouble running part 2 and if possible  I would like your assistance . I am running on Windows 10 and without WSL.

I have installed mosquitto and start the service by typing (from the installation folder):

```
net start mosquitto
```

and the service successfully starts in the default port of 1883. Next, in order to find the broker\_ip I run:

```
ipconfig/all
```

and get the respective IP address, which later is used to redirect network traffic from 80 or 443 (I have tried both) to the broker\_ip found with the above command. The redirection is happening by typing the following:

```
netsh interface portproxy add v4tov4 listenport=443 listenaddress="what should be put here?" connectport=1883 connectaddress=" my broker_ip"
```

And then I complete in the script with the port and the broker\_ip and execute it. I get 5272 [error code](https://www.mql5.com/en/articles/70 "Article: OOP in MQL5 by Example: Processing Warning and Error Codes "), which means that it "Failed to connect to remote host". Of course, I have included the broker\_ip to the Expert options tab.

Is something that I am missing in the whole process?

Thanks in advance,

Dr.Pip

![Jocimar Lopes](https://c.mql5.com/avatar/2023/2/63de1090-f297.jpg)

**[Jocimar Lopes](https://www.mql5.com/en/users/jslopes)**
\|
14 Mar 2024 at 15:42

**DrPip83 [#](https://www.mql5.com/en/forum/450255#comment_52199716):**

Hi Jocimar,

Thank you for this amazing tutorial. I have a trouble running part 2 and if possible  I would like your assistance . I am running on Windows 10 and without WSL.

I have installed mosquitto and start the service by typing (from the installation folder):

and the service successfully starts in the default port of 1883. Next, in order to find the broker\_ip I run:

and get the respective IP address, which later is used to redirect network traffic from 80 or 443 (I have tried both) to the broker\_ip found with the above command. The redirection is happening by typing the following:

And then I complete in the script with the port and the broker\_ip and execute it. I get 5272 [error code](https://www.mql5.com/en/articles/70 "Article: OOP in MQL5 by Example: Processing Warning and Error Codes "), which means that it "Failed to connect to remote host". Of course, I have included the broker\_ip to the Expert options tab.

Is something that I am missing in the whole process?

Thanks in advance,

Dr.Pip

Hi, DrPip83

I noted that you have asked this questions fourty days ago. I didn't see it before. Right now I received a message from an admin about your question. I'm sorry for this long delay.

To the point: what script are you using to connect? Could you please share it? In part 2 we have \*\*no functional connection\*\* with the broker. Untill that point we had only packet building stubs, the first classes that we were prototyping. We were experimenting with connections "out-of-band", let's say, using only internal scratch scripts not shared on the attachments. Only now we are implementing real connections in the so-called Operational Behavior of the protocol (or the Actions, as per the Standard).

So, could you please share your script? Maybe I can help with it or share with you the scripts we were using for this purpose.

![Adam John Bradley](https://c.mql5.com/avatar/avatar_na2.png)

**[Adam John Bradley](https://www.mql5.com/en/users/adam_j_bradley)**
\|
11 Sep 2024 at 14:05

Wondering if you’ve published to [GitHub](https://www.mql5.com/go?link=https://forge.mql5.io/help/en/guide "MQL5 Algo Forge: Cloud Workspace for Algorithmic Trading Development") as yet? Keen to have a look and contribute if I can!


![Improve Your Trading Charts With Interactive GUI's in MQL5 (Part II): Movable GUI (II)](https://c.mql5.com/2/56/Revolutionize_Your_Trading_Charts_Part_2-avatar.png)[Improve Your Trading Charts With Interactive GUI's in MQL5 (Part II): Movable GUI (II)](https://www.mql5.com/en/articles/12880)

Unlock the potential of dynamic data representation in your trading strategies and utilities with our in-depth guide to creating movable GUIs in MQL5. Delve into the fundamental principles of object-oriented programming and discover how to design and implement single or multiple movable GUIs on the same chart with ease and efficiency.

![Integrating ML models with the Strategy Tester (Part 3): Managing CSV files (II)](https://c.mql5.com/2/51/Avatar_Perceptron_Multicamadas_e_o-Algoritmo_Backpropagation_Parte_3_02.png)[Integrating ML models with the Strategy Tester (Part 3): Managing CSV files (II)](https://www.mql5.com/en/articles/12069)

This material provides a complete guide to creating a class in MQL5 for efficient management of CSV files. We will see the implementation of methods for opening, writing, reading, and transforming data. We will also consider how to use them to store and access information. In addition, we will discuss the limitations and the most important aspects of using such a class. This article ca be a valuable resource for those who want to learn how to process CSV files in MQL5.

![Can Heiken-Ashi Combined With Moving Averages Provide Good Signals Together?](https://c.mql5.com/2/56/heiken_ashi_combined_moving_averages_avatar.png)[Can Heiken-Ashi Combined With Moving Averages Provide Good Signals Together?](https://www.mql5.com/en/articles/12845)

Combinations of strategies may offer better opportunities. We can combine indicators or patterns together, or even better, indicators with patterns, so that we get an extra confirmation factor. Moving averages help us confirm and ride the trend. They are the most known technical indicators and this is because of their simplicity and their proven track record of adding value to analyses.

![Category Theory in MQL5 (Part 12): Orders](https://c.mql5.com/2/56/Category-Theory-p12-avatar.png)[Category Theory in MQL5 (Part 12): Orders](https://www.mql5.com/en/articles/12873)

This article which is part of a series that follows Category Theory implementation of Graphs in MQL5, delves in Orders. We examine how concepts of Order-Theory can support monoid sets in informing trade decisions by considering two major ordering types.

[![](https://www.mql5.com/ff/si/mbxx5fzr169cx07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F498%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dhow.buy.expert%26utm_content%3Dbuy.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=yiuacrhbffqmmulobpsgnypolteeimpt&s=949562ee5e6aca93c0231542844344e241ce4a26ab488f494b70624c190b74d7&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=phjkkpapmwqyuwdvofawxwbaujcbxcxc&ssn=1769178166339631468&ssn_dr=0&ssn_sr=0&fv_date=1769178166&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F12857&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Developing%20an%20MQTT%20client%20for%20MetaTrader%205%3A%20a%20TDD%20approach%20-%20MQL5%20Articles&scr_res=1920x1080&ac=17691781664596919&fz_uniq=5068155662686418485&sv=2552)

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