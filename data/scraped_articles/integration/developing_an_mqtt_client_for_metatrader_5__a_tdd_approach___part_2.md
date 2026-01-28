---
title: Developing an MQTT client for MetaTrader 5: a TDD approach — Part 2
url: https://www.mql5.com/en/articles/13334
categories: Integration
relevance_score: 6
scraped_at: 2026-01-23T17:22:37.366616
---

[![](https://www.mql5.com/ff/sh/rvgkjnsrvj1mzh89z2/01.png)Best VPS for tradersTwo-click launch from MetaTrader, minimum ping to broker, 15 USD/monthLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=wpjhvzsogglsviotmypjoyhhtuxlrzhi&s=aa6c5782a1658c2f617954d478dea9989a27ae26ecabc09d0ab1204277fdf8e3&uid=&ref=https://www.mql5.com/en/articles/13334&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5068153605397083691)

MetaTrader 5 / Integration


### Introduction

_"Premature optimization is the root of all evil." (Donald Knuth)_

In the [previous article](https://www.mql5.com/en/articles/12857), we introduced MQTT, a highly efficient binary pub/sub messaging protocol. We talked about what is MQTT,  why its development started twenty-five years ago, and what it is used for nowadays in several industries, from automotive to Internet of Things, from aerospatial to simple chat applications. We saw that MQTT can be useful in any context where a content-type agnostic message-sharing protocol is required, including the context of trading applications. We stated the benefits of including in our code base a native MQL5 client for MQTT and we did set up a minimum development environment using Mosquitto open-source MQTT broker running on WSL (Windows Subsystem For Linux).

We started developing our native client with only a hardcoded Fixed Header generator function and reached our development to the point that we were able to connect with a local Mosquitto broker, but the connection was immediately reset by the server due to Protocol Error.

A quote from the bottom of the previous article:

“So, yes, our CONNECT fixed header was recognized by Mosquito, but the <unknown> Client was disconnected immediately “due to protocol error”. The error occurred because we didn’t include the Variable Header yet, with the Protocol Name, Protocol Level, and other associated metadata. We will fix that in the next step.”

![Fig. 01 - Metaeditor Experts Log Connect Response Fail](https://c.mql5.com/2/58/01-connect-response-fail.PNG)

Fig. 01 - Metaeditor experts tab log showing a connect response: Fail

Since we are using a [Test-Driven Development](https://en.wikipedia.org/wiki/Test-driven_development "https://en.wikipedia.org/wiki/Test-driven_development") (TDD) approach, we will start this step by writing a test for a CONNECT packet builder that includes those associated metadata and doesn’t get rejected “due to protocol error”.

By writing this test even before we write the code to be tested may be counter-intuitive for many of us. But it will seem very natural, even the expected step to be done before any enterprise, if we think about the tests as an objective description of the project requirements, if we see them as the most objective definition of a goal a developer can have.

“The unit tests are documents. They describe the lowest-level design of the system. They are unambiguous, accurate, written in a language that the audience understands, and are so formal that they execute. They are the best kind of low-level documentation that can exist. What professional would not provide such documentation?” (Robert Martin, The Clean Coder, 2011)

### Code organization: OOP and Header Files

As said above, we started building our connect packet Fixed Header by hardcoding a byte array with the “right” values. Then we tried to connect our client by sending that hardcoded byte array to our local broker. We failed miserably in our attempt to connect “due to protocol error”. But in the meantime, we learned something about our development environment, about our Mosquito log, wrote our first tests, and above all, we started doing something that worked, at least as baby steps.

As you can see, that failure was on purpose. But we know that this way of developing a complex application is not sustainable in the long term.

Building conformant MQTT packets is only the first step in the process of writing a robust and maintainable client. The easiest part, as to say. When it comes to the Operational behavior specification all the complexities of the protocol will emerge. This complexity will demand more of our job as developers. Besides sending good packets, we will be required to deal with a large number of different server responses and different application states. At this point, hardcoded byte arrays – or anything hardcoded for this matter – will not be enough.

Fortunately, MQL5 is an object-oriented programming language and we are not working in the memory/CPU-constrained environment for which MQTT was originally designed. So we can leverage all the benefits of the object-oriented (OOP) paradigm in order to have:

- Easy reasoning about the protocol by choosing the right abstraction level
- Easy code reading (remember that code is read many more times than it is written)
- Easy code maintenance,
- And easy testing

The reference documentation of MQL5 has extensive support for [Object-Oriented-Programming for MQL5](https://www.mql5.com/en/docs/basis/oop), with an entire section dedicated to the topic.

### **Protocol Definitions**

A message-sharing protocol is a set of rules that establishes a common ground of understanding between two or more entities. In our case, between two or more devices. Many of these rules are concerned with what to do by taking into account what was done before. They are stateful. In order to choose the next action our code must evaluate the current state of the application. In MQTT protocol parlance these are the Operational behavior rules.

Besides the stateful rules – in fact, preceding them – there are the definitions of terms, values, and calculations that are independent of the application state. They are usually constants, enumerations, and evaluation algorithms, as is the case of the MQTT protocol name, control packet types, and the fixed header remaining length byte value respectively.

We will be gathering these two distinct sets of rules in two distinct header files. The first of them is just for the definitions of terms and values shared among our files. Without surprise, we will name it _Defines.mqh_. These terms and values are typically constants and this file should evolve almost nothing.

The other header file will be the host of some shared Enums, Structures, and Functions. It will be named _MQTT.mqh_. These Enums, Structures, and Functions will evolve a lot, and not only while we are developing the first version. This file will change whenever we are doing improvements, optimizations, and bug fixing. Probably, this file will be subdivided into other more specific files.

The practice of using header files for code organization is not tied to Object-Oriented Programming. In fact, we will find a useful note about them already in the now classic book “The C Programming Language”, by Brian Kernighan and Dennis Ritchie.

“(…) the definitions and declarations shared among files. As much as possible, we want to centralize this, so that there is only one copy to get and keep right as the program evolves. (…) Up to some moderate program size, it is probably best to have one header file that contains everything that is to be shared between any two parts of the program (...). For a much larger program, more organization and more headers would be needed.”

But it is in Object-Oriented Programming that the practice of organizing the code in small compilation units really shines. Moreover, since we are building a library almost all of our code will be in header files.

**The Defines header**

At this point, the Protocol Name, and the Protocol Level definitions are used only on CONNECT packets. So, if we want, we could put them in the specific _CPktConnect_ class (see below). But we are leaving them on the Defines header for consistency. Although they are used only on CONNECT packets at this point, they may be used in other files later.

The comments about the protocol are literal quotations from the official standard specification.

```
//+------------------------------------------------------------------+
//|                                                      Defines.mqh |
//|            ********* WORK IN PROGRESS **********                 |
//| **** PART OF ARTICLE https://www.mql5.com/en/articles/13334 **** |
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//|              PROTOCOL NAME AND VERSION                           |
//+------------------------------------------------------------------+
#define MQTT_PROTOCOL_NAME_LENGTH_MSB           0x00
#define MQTT_PROTOCOL_NAME_LENGTH_LSB           0x04
#define MQTT_PROTOCOL_NAME_BYTE_3               'M'
#define MQTT_PROTOCOL_NAME_BYTE_4               'Q'
#define MQTT_PROTOCOL_NAME_BYTE_5               'T'
#define MQTT_PROTOCOL_NAME_BYTE_6               'T'
#define MQTT_PROTOCOL_VERSION                   0x05
//+------------------------------------------------------------------+
//|              PROPERTIES                                          |
//+------------------------------------------------------------------+
/*
The last field in the Variable Header of the CONNECT, CONNACK, PUBLISH, PUBACK, PUBREC,
PUBREL, PUBCOMP, SUBSCRIBE, SUBACK, UNSUBSCRIBE, UNSUBACK, DISCONNECT, and
AUTH packet is a set of Properties. In the CONNECT packet there is also an optional set of Properties in
the Will Properties field with the Payload
*/
#define MQTT_PROPERTY_PAYLOAD_FORMAT_INDICATOR          0x01 // (1) Byte
#define MQTT_PROPERTY_MESSAGE_EXPIRY_INTERVAL           0x02 // (2) Four Byte Integer
#define MQTT_PROPERTY_CONTENT_TYPE                      0x03 // (3) UTF-8 Encoded String
#define MQTT_PROPERTY_RESPONSE_TOPIC                    0x08 // (8) UTF-8 Encoded String
#define MQTT_PROPERTY_CORRELATION_DATA                  0x09 // (9) Binary Data
#define MQTT_PROPERTY_SUBSCRIPTION_IDENTIFIER           0x0B // (11) Variable Byte Integer
#define MQTT_PROPERTY_SESSION_EXPIRY_INTERVAL           0x11 // (17) Four Byte Integer
#define MQTT_PROPERTY_ASSIGNED_CLIENT_IDENTIFIER        0x12 // (18) UTF-8 Encoded String
#define MQTT_PROPERTY_SERVER_KEEP_ALIVE                 0x13 // (19) Two Byte Integer
#define MQTT_PROPERTY_AUTHENTICATION_METHOD             0x15 // (21) UTF-8 Encoded String
#define MQTT_PROPERTY_AUTHENTICATION_DATA               0x16 // (22) Binary Data
#define MQTT_PROPERTY_REQUEST_PROBLEM_INFORMATION       0x17 // (23) Byte
#define MQTT_PROPERTY_WILL_DELAY_INTERVAL               0x18 // (24) Four Byte Integer
#define MQTT_PROPERTY_REQUEST_RESPONSE_INFORMATION      0x19 // (25) Byte
#define MQTT_PROPERTY_RESPONSE_INFORMATION              0x1A // (26) UTF-8 Encoded String
#define MQTT_PROPERTY_SERVER_REFERENCE                  0x1C // (28) UTF-8 Encoded String
#define MQTT_PROPERTY_REASON_STRING                     0x1F // (31) UTF-8 Encoded String
#define MQTT_PROPERTY_RECEIVE_MAXIMUM                   0x21 // (33) Two Byte Integer
#define MQTT_PROPERTY_TOPIC_ALIAS_MAXIMUM               0x22 // (34) Two Byte Integer
#define MQTT_PROPERTY_TOPIC_ALIAS                       0x23 // (35) Two Byte Integer
#define MQTT_PROPERTY_MAXIMUM_QOS                       0x24 // (36) Byte
#define MQTT_PROPERTY_RETAIN_AVAILABLE                  0x25 // (37) Byte
#define MQTT_PROPERTY_USER_PROPERTY                     0x26 // (38) UTF-8 String Pair
#define MQTT_PROPERTY_MAXIMUM_PACKET_SIZE               0x27 // (39) Four Byte Integer
#define MQTT_PROPERTY_WILDCARD_SUBSCRIPTION_AVAILABLE   0x28 // (40) Byte
#define MQTT_PROPERTY_SUBSCRIPTION_IDENTIFIER_AVAILABLE 0x29 // (41) Byte
#define MQTT_PROPERTY_SHARED_SUBSCRIPTION_AVAILABLE     0x2A // (42) Byte
//+------------------------------------------------------------------+
//|              REASON CODES                                        |
//+------------------------------------------------------------------+
/*
A Reason Code is a one byte unsigned value that indicates the result of an operation. Reason Codes less
than 0x80 indicate successful completion of an operation. The normal Reason Code for success is 0.
Reason Code values of 0x80 or greater indicate failure.

The CONNACK, PUBACK, PUBREC, PUBREL, PUBCOMP, DISCONNECT and AUTH Control Packets
have a single Reason Code as part of the Variable Header. The SUBACK and UNSUBACK packets
contain a list of one or more Reason Codes in the Payload.
*/
#define MQTT_REASON_CODE_SUCCESS                                0x00 // (0)
#define MQTT_REASON_CODE_NORMAL_DISCONNECTION                   0x00 // (0)
#define MQTT_REASON_CODE_GRANTED_QOS_0                          0x00 // (0)
#define MQTT_REASON_CODE_GRANTED_QOS_1                          0x01 // (1)
#define MQTT_REASON_CODE_GRANTED_QOS_2                          0x02 // (2)
#define MQTT_REASON_CODE_DISCONNECT_WITH_WILL_MESSAGE           0x04 // (4)
#define MQTT_REASON_CODE_NO_MATCHING_SUBSCRIBERS                0x10 // (16)
#define MQTT_REASON_CODE_NO_SUBSCRIPTION_EXISTED                0x11 // (17)
#define MQTT_REASON_CODE_CONTINUE_AUTHENTICATION                0x18 // (24)
#define MQTT_REASON_CODE_RE_AUTHENTICATE                        0x19 // (25)
#define MQTT_REASON_CODE_UNSPECIFIED_ERROR                      0x80 // (128)
#define MQTT_REASON_CODE_MALFORMED_PACKET                       0x81 // (129)
#define MQTT_REASON_CODE_PROTOCOL_ERROR                         0x82 // (130)
#define MQTT_REASON_CODE_IMPLEMENTATION_SPECIFIC_ERROR          0x83 // (131)
#define MQTT_REASON_CODE_UNSUPPORTED_PROTOCOL_VERSION           0x84 // (132)
#define MQTT_REASON_CODE_CLIENT_IDENTIFIER_NOT_VALID            0x85 // (133)
#define MQTT_REASON_CODE_BAD_USER_NAME_OR_PASSWORD              0x86 // (134)
#define MQTT_REASON_CODE_NOT_AUTHORIZED                         0x87 // (135)
#define MQTT_REASON_CODE_SERVER_UNAVAILABLE                     0x88 // (136)
#define MQTT_REASON_CODE_SERVER_BUSY                            0x89 // (137)
#define MQTT_REASON_CODE_BANNED                                 0x8A // (138)
#define MQTT_REASON_CODE_SERVER_SHUTTING_DOWN                   0x8B // (139)
#define MQTT_REASON_CODE_BAD_AUTHENTICATION_METHOD              0x8C // (140)
#define MQTT_REASON_CODE_KEEP_ALIVE_TIMEOUT                     0x8D // (141)
#define MQTT_REASON_CODE_SESSION_TAKEN_OVER                     0x8E // (142)
#define MQTT_REASON_CODE_TOPIC_FILTER_INVALID                   0x8F // (143)
#define MQTT_REASON_CODE_TOPIC_NAME_INVALID                     0x90 // (144)
#define MQTT_REASON_CODE_PACKET_IDENTIFIER_IN_USE               0x91 // (145)
#define MQTT_REASON_CODE_PACKET_IDENTIFIER_NOT_FOUND            0x92 // (146)
#define MQTT_REASON_CODE_RECEIVE_MAXIMUM_EXCEEDED               0x93 // (147)
#define MQTT_REASON_CODE_TOPIC_ALIAS_INVALID                    0x94 // (148)
#define MQTT_REASON_CODE_PACKET_TOO_LARGE                       0x95 // (149)
#define MQTT_REASON_CODE_MESSAGE_RATE_TOO_HIGH                  0x96 // (150)
#define MQTT_REASON_CODE_QUOTA_EXCEEDED                         0x97 // (151)
#define MQTT_REASON_CODE_ADMINISTRATIVE_ACTION                  0x98 // (152)
#define MQTT_REASON_CODE_PAYLOAD_FORMAT_INVALID                 0x99 // (153)
#define MQTT_REASON_CODE_RETAIN_NOT_SUPPORTED                   0x9A // (154)
#define MQTT_REASON_CODE_QOS_NOT_SUPPORTED                      0x9B // (155)
#define MQTT_REASON_CODE_USE_ANOTHER_SERVER                     0x9C // (156)
#define MQTT_REASON_CODE_SERVER_MOVED                           0x9D // (157)
#define MQTT_REASON_CODE_SHARED_SUBSCRIPTIONS_NOT_SUPPORTED     0x9E // (158)
#define MQTT_REASON_CODE_CONNECTION_RATE_EXCEEDED               0x9F // (159)
#define MQTT_REASON_CODE_MAXIMUM_CONNECT_TIME                   0xA0 // (160)
#define MQTT_REASON_CODE_SUBSCRIPTION_IDENTIFIERS_NOT_SUPPORTED 0xA1 // (161)
#define MQTT_REASON_CODE_WILDCARD_SUBSCRIPTIONS_NOT_SUPPORTED   0xA2 // (162)
```

Note that we are prefixing all protocol specific definitions with MQTT. This is to differentiate them from our own definitions to be included further. Also note in the top of our _Defines.mqh_ file, in the PROTOCOL NAME AND VERSION, that we are trying to be as explicit as possible in our identifiers naming. This is meant to cope with the principles of the so called Clean Code. This practice should help to make our code more reader-friendly, easier to debug, and IDE-friendly, that is, more searcheable and well-suited to leverage the autocompletion feature of modern IDE’s.

**The MQTT header**

```
//+------------------------------------------------------------------+
//|                                                         MQTT.mqh |
//|            ********* WORK IN PROGRESS **********                 |
//| **** PART OF ARTICLE https://www.mql5.com/en/articles/13334 **** |
//+------------------------------------------------------------------+
#include "Defines.mqh"
//+------------------------------------------------------------------+
//|              MQTT - CONTROL PACKET - TYPES                       |
//+------------------------------------------------------------------+
/*
Position: byte 1, bits 7-4.
Represented as a 4-bit unsigned value, the values are shown below.
*/
enum ENUM_PKT_TYPE
  {
   CONNECT     =  0x01, // Connection request
   CONNACK     =  0x02, // Connection Acknowledgment
   PUBLISH     =  0x03, // Publish message
   PUBACK      =  0x04, // Publish acknowledgment (QoS 1)
   PUBREC      =  0x05, // Publish received (QoS 2 delivery part 1)
   PUBREL      =  0x06, // Publish release (QoS 2 delivery part 2)
   PUBCOMP     =  0x07, // Publish complete (QoS 2 delivery part 3)
   SUBSCRIBE   =  0x08, // Subscribe request
   SUBACK      =  0x09, // Subscribe acknowledgment
   UNSUBSCRIBE =  0x0A, // Unsubscribe request
   UNSUBACK    =  0x0B, // Unsubscribe acknowledgment
   PINGREQ     =  0x0C, // PING request
   PINGRESP    =  0x0D, // PING response
   DISCONNECT  =  0x0E, // Disconnect notification
   AUTH        =  0x0F, // Authentication exchange
  };
//+------------------------------------------------------------------+
//|             CONNECT - VARIABLE HEADER - CONNECT FLAGS            |
//+------------------------------------------------------------------+
/*
The Connect Flags byte contains several parameters specifying the behavior of the MQTT connection. It
also indicates the presence or absence of fields in the Payload.
*/
enum ENUM_CONNECT_FLAGS
  {
   RESERVED       = 0x00,
   CLEAN_START    = 0x02,
   WILL_FLAG      = 0x04,
   WILL_QOS_1     = 0x08,
   WILL_QOS_2     = 0x10,
   WILL_RETAIN    = 0x20,
   PASSWORD_FLAG  = 0x40,
   USER_NAME_FLAG = 0x80
  };
//+------------------------------------------------------------------+
//|             CONNECT - VARIABLE HEADER - QoS LEVELS               |
//+------------------------------------------------------------------+
/*
Position: bits 4 and 3 of the Connect Flags.
These two bits specify the QoS level to be used when publishing the Will Message.
If the Will Flag is set to 0, then the Will QoS MUST be set to 0 (0x00) [MQTT-3.1.2-11].
If the Will Flag is set to 1, the value of Will QoS can be 0 (0x00), 1 (0x01), or 2 (0x02) [MQTT-3.1.2-12].
*/
enum ENUM_QOS_LEVEL
  {
   AT_MOST_ONCE   = 0x00,
   AT_LEAST_ONCE  = 0x01,
   EXACTLY_ONCE   = 0x02
  };
//+------------------------------------------------------------------+
//|                   SetProtocolVersion                             |
//+------------------------------------------------------------------+
void SetProtocolVersion(uchar& dest_buf[])
  {
   dest_buf[8] = MQTT_PROTOCOL_VERSION;
  }
//+------------------------------------------------------------------+
//|                     SetProtocolName                              |
//+------------------------------------------------------------------+
void SetProtocolName(uchar& dest_buf[])
  {
   dest_buf[2] = MQTT_PROTOCOL_NAME_LENGTH_MSB;
   dest_buf[3] = MQTT_PROTOCOL_NAME_LENGTH_LSB;
   dest_buf[4] = MQTT_PROTOCOL_NAME_BYTE_3;
   dest_buf[5] = MQTT_PROTOCOL_NAME_BYTE_4;
   dest_buf[6] = MQTT_PROTOCOL_NAME_BYTE_5;
   dest_buf[7] = MQTT_PROTOCOL_NAME_BYTE_6;
  }
//+------------------------------------------------------------------+
//|                     SetFixedHeader                               |
//+------------------------------------------------------------------+
void SetFixedHeader(ENUM_PKT_TYPE pkt_type, uchar& buf[], uchar& dest_buf[])
  {
   dest_buf[0] = (uchar)pkt_type << 4;
   dest_buf[1] = GetRemainingLength(buf);
  }
//+------------------------------------------------------------------+
//|                    GetRemainingLength                            |
//+------------------------------------------------------------------+
/*
Position: starts at byte 2.
The Remaining Length is a Variable Byte Integer that represents the number of bytes remaining within the
current Control Packet, including data in the Variable Header and the Payload. The Remaining Length
does not include the bytes used to encode the Remaining Length. The packet size is the total number of
bytes in an MQTT Control Packet, this is equal to the length of the Fixed Header plus the Remaining
Length.
*/
uchar GetRemainingLength(uchar &buf[])
  {
   uint x;
   x = ArraySize(buf);
   uint rem_len;
   do
     {
      rem_len = x % 128;
      x = (x / 128);
      if(x > 0)
        {
         rem_len = rem_len | 128;
        }
     }
   while(x > 0);
   return (uchar)rem_len;
  };

//+------------------------------------------------------------------+
```

### Classes and Structures

**The MQTT Control Packet Interface**

Here we have a design choice to be made: to start the Control Packets object hierarchy with an abstract class or with an interface. We could start with a generic base class well suited for any Control Packet. This abstract class would be specialized in more specific derived Control Packet classes. Or we could start with a simple interface to be implemented by those Control Packet classes.

We are starting with an interface _IcontrolPacket_. This interface will have a simple method. This choice may change when implementing the **Operational Behavior** part of the protocol. We will probably change this interface to an abstract class with some virtual functions.

```
//+------------------------------------------------------------------+
//|                                               IControlPacket.mqh |
//|            ********* WORK IN PROGRESS **********                 |
//| **** PART OF ARTICLE https://www.mql5.com/en/articles/13334 **** |
//+------------------------------------------------------------------+
#include "MQTT.mqh"
//+------------------------------------------------------------------+
//|       Interface IControlPacket                                   |
//|       The root of object hierarchy                               |
//+------------------------------------------------------------------+
interface IControlPacket
  {

   bool              IsControlPacket();

  };
//+------------------------------------------------------------------+
```

As said, for now the single purpose of this interface is to act as the root of the MQTT packets object hierarchy. At this point, it is no more than a fancy placeholder.

**The MQTT Control Packet Connect Class**

The CONNECT Control Packet is the most demanding packet to write. Besides the fact that we are yet to get acquainted with the protocol, this specific packet has received the most significant improvements in version 5.0 , namely the Connect Properties and User Properties.

```
//+------------------------------------------------------------------+
//|                                                   PktConnect.mqh |
//|            ********* WORK IN PROGRESS **********                 |
//| **** PART OF ARTICLE https://www.mql5.com/en/articles/13334 **** |
//+------------------------------------------------------------------+
#include "MQTT.mqh"
#include "Defines.mqh"
#include "IControlPacket.mqh"
//+------------------------------------------------------------------+
//|        CONNECT VARIABLE HEADER                                   |
//+------------------------------------------------------------------+
/*
The Variable Header for the CONNECT Packet contains the following fields in this order:
Protocol Name,Protocol Level, Connect Flags, Keep Alive, and Properties.
*/
struct MqttClientIdentifierLength
  {
   uchar             msb;
   uchar             lsb;
  } clientIdLen;
//---
struct MqttKeepAlive
  {
   uchar             msb;
   uchar             lsb;
  } keepAlive;
//---
struct MqttConnectProperties
  {
   uint              prop_len;
   uchar             session_expiry_interval_id;
   uint              session_expiry_interval;
   uchar             receive_maximum_id;
   ushort            receive_maximum;
   uchar             maximum_packet_size_id;
   ushort            maximum_packet_size;
   uchar             topic_alias_maximum_id;
   ushort            topic_alias_maximum;
   uchar             request_response_information_id;
   uchar             request_response_information;
   uchar             request_problem_information_id;
   uchar             request_problem_information;
   uchar             user_property_id;
   string            user_property_key;
   string            user_property_value;
   uchar             authentication_method_id;
   string            authentication_method;
   uchar             authentication_data_id;
  } connectProps;
//---
struct MqttConnectPayload
  {
   uchar             client_id_len;
   string            client_id;
   ushort            will_properties_len;
   uchar             will_delay_interval_id;
   uint              will_delay_interval;
   uchar             payload_format_indicator_id;
   uchar             payload_format_indicator;
   uchar             message_expiry_interval_id;
   uint              message_expiry_interval;
   uchar             content_type_id;
   string            content_type;
   uchar             response_topic_id; // for request/response
   string            response_topic;
   uchar             correlation_data_id; // for request/response
   ulong             correlation_data[]; // binary data
   uchar             user_property_id;
   string            user_property_key;
   string            user_property_value;
   uchar             will_topic_len;
   string            will_topic;
   uchar             will_payload_len;
   ulong             will_payload[]; // binary data
   uchar             user_name_len;
   string            user_name;
   uchar             password_len;
   ulong             password; // binary data
  } connectPayload;
//+------------------------------------------------------------------+
//| Class CPktConnect.                                               |
//| Purpose: Class of MQTT Connect Control Packets.                  |
//|          Implements IControlPacket                               |
//+------------------------------------------------------------------+
class CPktConnect : public IControlPacket
  {
private:
   bool              IsControlPacket() {return true;}
protected:
   void              InitConnectFlags() {ByteArray[9] = 0;}
   void              InitKeepAlive() {ByteArray[10] = 0; ByteArray[11] = 0;}
   void              InitPropertiesLength() {ByteArray[12] = 0;}
   uchar             m_connect_flags;

public:
                     CPktConnect();
                     CPktConnect(uchar &buf[]);
                    ~CPktConnect();
   //--- methods for setting Connect Flags
   void              SetCleanStart(const bool cleanStart);
   void              SetWillFlag(const bool willFlag);
   void              SetWillQoS_1(const bool willQoS_1);
   void              SetWillQoS_2(const bool willQoS_2);
   void              SetWillRetain(const bool willRetain);
   void              SetPasswordFlag(const bool passwordFlag);
   void              SetUserNameFlag(const bool userNameFlag);
   void              SetKeepAlive(ushort seconds);
   void              SetClientIdentifierLength(string clientId);
   void              SetClientIdentifier(string clientId);

   //--- member for getting the byte array
   uchar             ByteArray[];
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CPktConnect::CPktConnect(uchar &buf[])
  {
   ArrayFree(ByteArray);
   ArrayResize(ByteArray, buf.Size() + 2, UCHAR_MAX);
   SetFixedHeader(CONNECT, buf, ByteArray);
   SetProtocolName(ByteArray);
   SetProtocolVersion(ByteArray);
   InitConnectFlags();
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CPktConnect::SetClientIdentifier(string clientId)
  {
   SetClientIdentifierLength(clientId);
   StringToCharArray(clientId, ByteArray,
                     ByteArray.Size() - StringLen(clientId), StringLen(clientId));
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CPktConnect::SetClientIdentifierLength(string clientId)
  {
   clientIdLen.msb = (char)StringLen(clientId) >> 8;
   clientIdLen.lsb = (char)StringLen(clientId) % 256;
   ByteArray[12] = clientIdLen.msb;
   ByteArray[13] = clientIdLen.lsb;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CPktConnect::SetKeepAlive(ushort seconds) // MQTT max is 65,535 sec
  {
   keepAlive.msb = (uchar)(seconds >> 8) & 255;
   keepAlive.lsb = (uchar)seconds & 255;
   ByteArray[10] = keepAlive.msb;
   ByteArray[11] = keepAlive.lsb;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CPktConnect::SetPasswordFlag(const bool passwordFlag)
  {
   passwordFlag ? m_connect_flags |= PASSWORD_FLAG : m_connect_flags &= ~PASSWORD_FLAG;
   ArrayFill(ByteArray, sizeof(ByteArray), 1, m_connect_flags);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CPktConnect::SetUserNameFlag(const bool userNameFlag)
  {
   userNameFlag ? m_connect_flags |= USER_NAME_FLAG : m_connect_flags &= (uchar) ~USER_NAME_FLAG;
   ArrayFill(ByteArray, sizeof(ByteArray), 1, m_connect_flags);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CPktConnect::SetWillRetain(const bool willRetain)
  {
   willRetain ? m_connect_flags |= WILL_RETAIN : m_connect_flags &= ~WILL_RETAIN;
   ArrayFill(ByteArray, sizeof(ByteArray), 1, m_connect_flags);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CPktConnect::SetWillQoS_2(const bool willQoS_2)
  {
   willQoS_2 ? m_connect_flags |= WILL_QOS_2 : m_connect_flags &= ~WILL_QOS_2;
   ArrayFill(ByteArray, sizeof(ByteArray), 1, m_connect_flags);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CPktConnect::SetWillQoS_1(const bool willQoS_1)
  {
   willQoS_1 ? m_connect_flags |= WILL_QOS_1 : m_connect_flags &= ~WILL_QOS_1;
   ArrayFill(ByteArray, sizeof(ByteArray), 1, m_connect_flags);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CPktConnect::SetWillFlag(const bool willFlag)
  {
   willFlag ? m_connect_flags |= WILL_FLAG : m_connect_flags &= ~WILL_FLAG;
   ArrayFill(ByteArray, sizeof(ByteArray), 1, m_connect_flags);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CPktConnect::SetCleanStart(const bool cleanStart)
  {
   cleanStart ? m_connect_flags |= CLEAN_START : m_connect_flags &= ~CLEAN_START;
   ArrayFill(ByteArray, 9, 1, m_connect_flags);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CPktConnect::CPktConnect()
  {
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CPktConnect::~CPktConnect()
  {
  }
//+------------------------------------------------------------------+
```

### Testing Our First Class

The single purpose of the _CPktConnect_ class is to build a well-formed MQTT CONNECT packet. So, to test it we need to start by producing a kind of “fixture”, a sample byte array to represent a well-formed CONNECT packet. But how could we be sure that our made-up byte array represents a well-formed CONNECT packet? At the end of the day, this is the test that we will be using for starting our class from scratch. Many, if not all of our hard work may be wasted if our fixture represents a bad packet.

The protocol developer and maintainer, OASIS, came to our rescue. In section 3.1.2.12 we will find a Variable Header non-normative example for the CONNECT packet. Since we already tested our Fixed Header generator ( [see previous article](https://www.mql5.com/en/articles/12857)), this OASIS example will be enough to get started. It will allow us to be sure our class is generating a well-formed packet with some different configurations like the boolean CleanSession and the Keep-Alive requested time span.

This hardcoded manually generated byte array will then be compared with the _CpktConnect_ generated packet.

```
//+------------------------------------------------------------------+
//|                                  TEST_CControlPacket_Connect.mq5 |
//|                                                                  |
//|            ********* WORK IN PROGRESS **********                 |
//| **** PART OF ARTICLE https://www.mql5.com/en/articles/13334 **** |
//+------------------------------------------------------------------+
#include <MQTT\CPktConnect.mqh>

//+------------------------------------------------------------------+
//| Tests for CControlPacketConnect class                            |
//+------------------------------------------------------------------+
void OnStart()
  {
   Print(TEST_SetCleanStart_KeepAlive_ClientIdentifier());
   Print(TEST_SetClientIdentifier());
   Print(TEST_SetClientIdentifierLength());
   Print(TEST_SetCleanStart_and_SetKeepAlive());
   Print(TEST_SetKeepAlive());
   Print(TEST_SetCleanStart());
  }
/* REFERENCE ARRAY (FIXTURE)
{16, 24, 0, 4, 77, 81, 84, 84, 5, 2, 0, 10, 0, 4, 7, 17, 0, 0, 0, 10, 25, 1, 77, 81, 76, 53}
*/
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool TEST_SetCleanStart_KeepAlive_ClientIdentifier()
  {
   Print(__FUNCTION__);
//--- Arrange
   static uchar expected[] =
     {16, 16, 0, 4, 77, 81, 84, 84, 5, 2, 0, 10, 0, 4, 77, 81, 76, 53};
   uchar buf[expected.Size() - 2];
   CPktConnect *cut = new CPktConnect(buf);
//--- Act
   cut.SetCleanStart(true);
   cut.SetKeepAlive(10);//10 sec
   cut.SetClientIdentifier("MQL5");
   uchar result[];
   ArrayCopy(result, cut.ByteArray);
//--- Assert
   bool isTrue = Assert(expected, result);
//--- cleanup
   delete cut;
   ZeroMemory(result);
   return  isTrue ? true : false;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool TEST_SetClientIdentifier()
  {
   Print(__FUNCTION__);
//--- Arrange
   static uchar expected[] =
     {16, 16, 0, 4, 77, 81, 84, 84, 5, 0, 0, 0, 0, 4, 77, 81, 76, 53};
   uchar buf[expected.Size() - 2];
   CPktConnect *cut = new CPktConnect(buf);
//--- Act
   cut.SetClientIdentifier("MQL5");
   uchar result[];
   ArrayCopy(result, cut.ByteArray);
//--- Assert
   bool isTrue = Assert(expected, result);
//--- cleanup
   delete cut;
   ZeroMemory(result);
   return  isTrue ? true : false;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool TEST_SetClientIdentifierLength()
  {
   Print(__FUNCTION__);
//--- Arrange
   static uchar expected[] =
     {16, 12, 0, 4, 77, 81, 84, 84, 5, 0, 0, 0, 0, 4};
   uchar buf[expected.Size() - 2];
   CPktConnect *cut = new CPktConnect(buf);
//--- Act
   cut.SetClientIdentifierLength("MQL5");
   uchar result[];
   ArrayCopy(result, cut.ByteArray);
//--- Assert
   bool isTrue = Assert(expected, result);
//--- cleanup
   delete cut;
   ZeroMemory(result);
   return  isTrue ? true : false;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool TEST_SetCleanStart_and_SetKeepAlive()
  {
   Print(__FUNCTION__);
//--- Arrange
   static uchar expected[] =
     {16, 10, 0, 4, 77, 81, 84, 84, 5, 2, 0, 10};
   uchar buf[expected.Size() - 2];
   CPktConnect *cut = new CPktConnect(buf);
//--- Act
   cut.SetCleanStart(true);
   cut.SetKeepAlive(10); //10 secs
   uchar result[];
   ArrayCopy(result, cut.ByteArray);
//--- Assert
   bool isTrue = Assert(expected, result);
//--- cleanup
   delete cut;
   ZeroMemory(result);
   return  isTrue ? true : false;
  }
//+------------------------------------------------------------------+
bool TEST_SetKeepAlive()
  {
   Print(__FUNCTION__);
//--- Arrange
   static uchar expected[] =
     {16, 10, 0, 4, 77, 81, 84, 84, 5, 0, 0, 10};
   uchar buf[expected.Size() - 2];
   CPktConnect *cut = new CPktConnect(buf);
//--- Act
   cut.SetKeepAlive(10); //10 secs
   uchar result[];
   ArrayCopy(result, cut.ByteArray);
//--- Assert
   bool isTrue = Assert(expected, result);
//--- cleanup
   delete cut;
   ZeroMemory(result);
   return  isTrue ? true : false;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool TEST_SetCleanStart()
  {
   Print(__FUNCTION__);
//--- Arrange
   static uchar expected[] =
     {16, 8, 0, 4, 77, 81, 84, 84, 5, 2};
   uchar buf[expected.Size() - 2];
   CPktConnect *cut = new CPktConnect(buf);
//--- Act
   cut.SetCleanStart(true);
   uchar result[];
   ArrayCopy(result, cut.ByteArray);
//--- Assert
   bool isTrue = Assert(expected, result);
//--- cleanup
   delete cut;
//ZeroMemory(result);
   return  isTrue ? true : false;
  }
//+------------------------------------------------------------------+
bool Assert(uchar& expected[], uchar& result[])
  {
   if(!ArrayCompare(expected, result) == 0)
     {
      for(uint i = 0; i < expected.Size(); i++)
        {
         printf("expected\t%d\t\t%d result", expected[i], result[i]);
        }
      printf("expected size %d <=> %d result size", expected.Size(), result.Size());
      Print("Expected");
      ArrayPrint(expected);
      Print("Result");
      ArrayPrint(result);
      return false;
     }
   return true;
  }
//+------------------------------------------------------------------+
```

**NOTE:** As you know, writing tests for things like "let's see if the header array has the data I just put in it" may seem to be a waste of time. But it is not. This set of "obvious" tests will accompany our code forever. One can think of them as a continuous and automated debugging tool that will prove its value when you find some kind of regression bug or even a dumb mistake like wrong copy-pasting. This is the reason why we are not testing only if the connect action works, in a black box way. We want to be sure our header is well-formed before testing the connect action. It is worth remembering that TDD is a process. Many, if not all of these tests will be rewritten or even deleted before we have a first working version of our code. But those that stay, will probably stay forever.

This test will pass only when the _CPktConnect_’ generated byte array returns 0 (zero) when ArrayCompare(d) with our reference byte array, our fixture.    ​

After we have tested some combinations of the basic Connect Properties, the packet will be sent to the broker and must not be rejected “due to protocol error”.

![Fig. 02 - Metaeditor Experts Log TEST_CPktConnect Success](https://c.mql5.com/2/58/experts-metaeditor-log-TEST_CPktConnect-success.PNG)

Fig. 02 - Metaeditor experts tab log showing _CPktConnect_ class test results

**Checking with our local MQTT broker**

Now we can run our Mosquitto local broker on WSL to check if our MQTT connection was successful.

If you have followed the default installation, Mosquito should be running as a Service on Linux. Thus you only need to ‘redir’ the ports (80 → 1883) and include the hostname on allowed URLs in your Metatrader 5 options.

![Fig. 03 - WSL Mosquitto Log Connected/Disconnected Success](https://c.mql5.com/2/58/wsl-mosquitto-log-connected-success.PNG)

Fig. 03 - Mosquitto log on WSL showing connect/disconnect status: Success.

Yes! Our connection attempt doesn’t return a Protocol Error. Now we can try exchanging messages between client and server.

### Conclusion

In the next step we will deal with CONNACK responses. In this step, we will have a solid base for start publishing our first message. And, of course, we will start writing a test for it!  :)  Stay tuned!

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/13334.zip "Download all attachments in the single ZIP archive")

[Defines.mqh](https://www.mql5.com/en/articles/download/13334/defines.mqh "Download Defines.mqh")(8.03 KB)

[MQTT.mqh](https://www.mql5.com/en/articles/download/13334/mqtt.mqh "Download MQTT.mqh")(5.01 KB)

[IControlPacket.mqh](https://www.mql5.com/en/articles/download/13334/icontrolpacket.mqh "Download IControlPacket.mqh")(0.8 KB)

[CPktConnect.mqh](https://www.mql5.com/en/articles/download/13334/cpktconnect.mqh "Download CPktConnect.mqh")(9.97 KB)

[TEST\_CControlPacket\_Connect.mq5](https://www.mql5.com/en/articles/download/13334/test_ccontrolpacket_connect.mq5 "Download TEST_CControlPacket_Connect.mq5")(5.89 KB)

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

**[Go to discussion](https://www.mql5.com/en/forum/454297)**

![Category Theory in MQL5 (Part 20): A detour to Self-Attention and the Transformer](https://c.mql5.com/2/58/Category-Theory-p20-avatar.png)[Category Theory in MQL5 (Part 20): A detour to Self-Attention and the Transformer](https://www.mql5.com/en/articles/13348)

We digress in our series by pondering at part of the algorithm to chatGPT. Are there any similarities or concepts borrowed from natural transformations? We attempt to answer these and other questions in a fun piece, with our code in a signal class format.

![Data label for timeseries mining (Part 2)：Make datasets with trend markers using Python](https://c.mql5.com/2/58/Make_datasets_with_trend_markers_using_Python_Avatar.png)[Data label for timeseries mining (Part 2)：Make datasets with trend markers using Python](https://www.mql5.com/en/articles/13253)

This series of articles introduces several time series labeling methods, which can create data that meets most artificial intelligence models, and targeted data labeling according to needs can make the trained artificial intelligence model more in line with the expected design, improve the accuracy of our model, and even help the model make a qualitative leap!

![Evaluating ONNX models using regression metrics](https://c.mql5.com/2/55/onnx_regression_metrics_avatar__1.png)[Evaluating ONNX models using regression metrics](https://www.mql5.com/en/articles/12772)

Regression is a task of predicting a real value from an unlabeled example. The so-called regression metrics are used to assess the accuracy of regression model predictions.

![Developing a Replay System — Market simulation (Part 06): First improvements (I)](https://c.mql5.com/2/53/replay-p6-avatar.png)[Developing a Replay System — Market simulation (Part 06): First improvements (I)](https://www.mql5.com/en/articles/10768)

In this article, we will begin to stabilize the entire system, without which we might not be able to proceed to the next steps.

[![](https://www.mql5.com/ff/sh/592yc11u3j4rs5z9z2/01.png)How AI helps create robots for MetaTrader 5Learn from our book "Neural Networks in Algo Trading with MQL5"Read](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/neurobook%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.neurobook%26utm_content=visit.page%26utm_campaign=neurobook.promo.04.2024&a=ghrobswocqgvhztzjldphupateyllpro&s=9929cb0b8629585b5a42fabc06c525e41f6c0ebdf3045d044a5413b93ea88b47&uid=&ref=https://www.mql5.com/en/articles/13334&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5068153605397083691)

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