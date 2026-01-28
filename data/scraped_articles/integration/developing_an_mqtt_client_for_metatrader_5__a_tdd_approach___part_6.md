---
title: Developing an MQTT client for Metatrader 5: a TDD approach — Part 6
url: https://www.mql5.com/en/articles/14391
categories: Integration
relevance_score: 6
scraped_at: 2026-01-23T17:21:39.200321
---

[Need a reliable hosting solution for your robots?Contact your broker and find out about available Sponsored MetaTrader VPS offeringsLearn more![](https://www.mql5.com/ff/sh/0pw0dk81s56qy774z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=vljwvezfjkfbvviocwskggexlvgykvob&s=70cf8e354b9a125332533ffb65d7365abe8dde5b5c1ede9caac479a9e9df4f25&uid=&ref=https://www.mql5.com/en/articles/14391&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5068138615961220592)

MetaTrader 5 / Integration


_“Optimism is an occupational hazard of programming; feedback is the treatment.” (Kent Beck)_

### Introduction

The Test-Driven Development methodology provides many benefits and has one major drawback. Among the benefits, it helps us write well-defined units, and well-named variables, to achieve high test coverage, have a better understanding of the domain, avoid over-engineering, and keep the focus on the task at hand. The major drawback is a direct consequence of this narrow focus on the task at hand, that is, to avoid being frightened by the overall complexity of the project we, as developers, keep solving the smallest possible challenge at a time, and only one challenge at a time. If the genius is the person who removes the complexity by solving it, the TDD developer is the person who deliberately ignores the complexity.

Yes, you got it: much like we were horses wearing blinders, much like that donkey following the carrot.

But the complexity doesn’t disappear because we ignored it. It stays there, waiting for us to face it. By ignoring the forest to look closely at the leaf, we keep leaving a technical debt behind. We keep leaving redundant functions, duplicated members, unuseful tests, unnecessary classes, unreadable and unreachable code, you know. This technical debt that is accumulated during development can be harmful to our productivity. It is the reason why refactoring is an integral part of the TDD practice. The below diagram shows the typical steps of a TDD practice.

![The Typical Steps of a TDD Practice: Red, Green, Refactoring](https://c.mql5.com/2/71/tdd-red-green-refactoring.png)

Fig. 01 - The Typical Steps of a TDD Practice: Red, Green, Refactoring (Source: [IBM Developer](https://www.mql5.com/go?link=https://developer.ibm.com/articles/5-steps-of-test-driven-development/ "https://developer.ibm.com/articles/5-steps-of-test-driven-development/"))

In the following sections we are describing how we refactored our previously written classes and commenting on some improvements. We show how we are building our PUBLISH packets after these improvements and how we arrived at a viable blueprint for our packet-building classes. The first class following the new pattern is the PUBACK class. Since PUBACK packets are the counterpart of PUBLISH packets with QoS 1, we need to start dealing with Session State management. Our client will need to have some kind of persistence layer to preserve and update the state.

The persistence layer is out of the scope of the OASIS Standard. It is application-specific. It could be a simple file in the local filesystem or a fully distributed high-availability database system in the cloud. For our purposes, a database like a [PostgreSQL server running locally on Windows or via WSL](https://www.mql5.com/en/articles/12308) would suffice. However, since we have a native integration between MQL and SQLite, this single-file, no-server RDBMS is the obvious choice here. SQLite is lightweight, scalable, trustworthy, and free of server maintenance. We can even have an on-memory-only database, which is pretty convenient for testing and debugging.

But we will not be implementing the persistence layer at this point, because we chose to have the writing and reading of packets well tested before dealing with Session State management. We need to be sure that we are correctly encoding and decoding the different data types used by the MQTT protocol before advancing to the persistence layer. To meet this goal we are writing extensive unit tests and soon we will start with small functional tests against a real broker running locally ( [the open-source mosquitto broker](https://www.mql5.com/go?link=https://mosquitto.org/ "https://mosquitto.org/"), from Eclipse Foundation).

So, to test our PUBLISH/PUBACK interactions we will use a fake database, a collection of functions to generate the controlled data we need for testing, a kind of a fixture. We will introduce it below when describing the CPuback class.

In the descriptions that follow, we are using the terms MUST and MAY as they are used by the OASIS Standard, which in turn uses them as described in [IETF RFC 2119](https://www.mql5.com/go?link=https://www.rfc-editor.org/info/rfc2119 "https://www.rfc-editor.org/info/rfc2119").

Also, unless otherwise stated, all quotes are from the [OASIS Standard](https://www.mql5.com/go?link=https://docs.oasis-open.org/mqtt/mqtt/v5.0/mqtt-v5.0.html "https://docs.oasis-open.org/mqtt/mqtt/v5.0/mqtt-v5.0.html").

### How We Are Building PUBLISH Packets

In the process of rewriting the CPublish class we removed some class members. We also merged the fixed header/variable header building in a one-step builder. These changes are being replicated in other Control Packet classes.

Currently, our CPublish class has the following members and methods.

```
//+------------------------------------------------------------------+
//|                                                      Publish.mqh |
//|            ********* WORK IN PROGRESS **********                 |
//| **** PART OF ARTICLE https://www.mql5.com/en/articles/14391 **** |
//+------------------------------------------------------------------+
#include "IControlPacket.mqh"
//+------------------------------------------------------------------+
//|        PUBLISH VARIABLE HEADER                                   |
//+------------------------------------------------------------------+
/*
The Variable Header of the PUBLISH Packet contains the following fields in the order: Topic Name,
Packet Identifier, and Properties.
*/
//+------------------------------------------------------------------+
//| Class CPublish.                                                  |
//| Purpose: Class of MQTT Publish Control Packets.                  |
//|          Implements IControlPacket                               |
//+------------------------------------------------------------------+
class CPublish : public IControlPacket
  {
private:
   bool              IsControlPacket() {return true;}
   bool              HasWildcardChar(const string str);
protected:
   uchar             m_pubflags;
   uint              m_remlen;
   uchar             m_topname[];
   uchar             m_props[];
   uint              m_payload[];
public:
                     CPublish();
                    ~CPublish();
   //--- methods for setting Publish flags
   void              SetRetain(const bool retain);
   void              SetQoS_1(const bool QoS_1);
   void              SetQoS_2(const bool QoS_2);
   void              SetDup(const bool dup);
   //--- method for setting Topic Name
   void              SetTopicName(const string topic_name);
   //--- methods for setting Properties
   void              SetPayloadFormatIndicator(PAYLOAD_FORMAT_INDICATOR format);
   void              SetMessageExpiryInterval(uint msg_expiry_interval);
   void              SetTopicAlias(ushort topic_alias);
   void              SetResponseTopic(const string response_topic);
   void              SetCorrelationData(uchar &binary_data[]);
   void              SetUserProperty(const string key, const string val);
   void              SetSubscriptionIdentifier(uint subscript_id);
   void              SetContentType(const string content_type);
   //--- method for setting the payload
   void              SetPayload(const string payload);
   //--- method for building the final packet
   void              Build(uchar &result[]);
  };
```

Besides the simplification, now the process of setting the publish flags, topic names, and properties are all independent, meaning each one of them can be set in any order, provided that the Build() method is the last to be invoked.

This test formalizes that behavior. It tests the class constructor with two flags set, RETAIN and QoS1, and the required Topic Name.

```
bool TEST_Ctor_Retain_QoS1_TopicName1Char()
  {
   Print(__FUNCTION__);
   CPublish *cut = new CPublish();
   uchar expected[] = {51, 6, 0, 1, 'a', 0, 1, 0}; // QoS > 0 require packet ID
   uchar result[];
   cut.SetTopicName("a");
   cut.SetRetain(true);
   cut.SetQoS_1(true);
   cut.Build(result);
   bool isTrue = AssertEqual(expected, result);
   delete(cut);
   ZeroMemory(result);
   return isTrue;
  }
```

Now, the methods SetTopicName(), SetRetain(), and SetQos1() can be called in any order and the resulting packet is still valid. As said, this behavior is being replicated in all control packet classes, and we have a test for every combination of publish flags. Please, see the attached files to get all the tests.

**The PUBLISH packet fixed header**

PUBLISH packets fixed headers are different from all other MQTT 5.0 Control Packets in the current version of the protocol. They have three flags that are NOT reserved for future use: RETAIN, QoS, and DUP flags. In the previous article, part 5 of this series, you can see [a detailed write up about these PUBLISH flags](https://www.mql5.com/en/articles/13998).

![MQTT 5.0 PUBLISH packet Fixed Header RETAIN, QoS Level, and DUP flags](https://c.mql5.com/2/71/MQTT_5.0_PUBLISH_packet_Fixed_Header_RETAINn_QoS_Levelm_and_DUP_flags.PNG)

Fig. 02 - MQTT 5.0 PUBLISH packet Fixed Header RETAIN, QoS Level, and DUP flags

We are using the same pattern for toggling any of the publish flags, but now, after refactoring, we are not calling SetFixedHeader() in each of them anymore. First, we define the toggle as a boolean value that is passed as an argument to the function.

```
void CPktPublish::SetRetain(const bool retain)
  {
   retain ? m_pubflags |= RETAIN_FLAG : m_pubflags &= ~RETAIN_FLAG;
  }
```

Then we check if the boolean value is true or false.

```
void CPktPublish::SetQoS_1(const bool QoS_1)
  {
   QoS_1 ? m_pubflags |= QoS_1_FLAG : m_pubflags &= ~QoS_1_FLAG;
  }
```

If the boolean value is true, we perform a bitwise OR assignment between the flag value and a uchar member (one byte) to set the flag.

```
void CPktPublish::SetQoS_2(const bool QoS_2)
  {
   QoS_2 ? m_pubflags |= QoS_2_FLAG : m_pubflags &= ~QoS_2_FLAG;
  }
```

If the boolean value is false, we perform a bitwise AND assignment between the flag value and the same uchar member to unset the flag.

```
void CPktPublish::SetDup(const bool dup)
  {
   dup ? m_pubflags |= DUP_FLAG : m_pubflags &= ~DUP_FLAG;
  }
```

This way, the m\_pubflags variable holds all the flags set/unset while configuring the packet. Later, when the Build() method is called, we perform a bitwise OR assignment again, this time between the m\_pubflags and the first byte of the packet (byte 0).

```
pkt[0] |= m_pubflags;
```

**The PUBLISH packet variable header**

The Variable Header of the PUBLISH Packet contains the following fields in the order: Topic Name, Packet Identifier, and Properties.

Topic Name

Since all the relationships between publishers and subscribers are tied to the Topic Name of the publication, this field is required in PUBLISH packets and cannot contain wildcard chars. When setting this field we have two guard conditions, for wildcard chars and for a string with zero length, returning immediately and logging the error if any of these conditions are true.

```
void CPktPublish::SetTopicName(const string topic_name)
  {
   if(HasWildcardChar(topic_name) || StringLen(topic_name) == 0)
     {
      ArrayFree(m_topname);
      return;
     }
   EncodeUTF8String(topic_name, m_topname);
  }
```

If none of the guard conditions are met, we encode the string as UTF-8 and store the char array in the m\_topname protected member to be further included in the final packet when Build() is called.

Packet Identifier

The Packet Identifier is NOT set by the user and is not required for QoS 0. Instead, it is set automatically on the Build() method, if the required QoS is > 0.

```
// QoS > 0 requires packet ID
   if((m_pubflags & 0x06) != 0)
     {
      SetPacketID(pkt, pkt.Size());
     }
```

When building the final packet we check the m\_pubflags member through a bitwise AND with the binary value of 0110 (0x06). If the result is not equal to zero we know that the packet has QoS\_1 or QoS\_2 and we set the Packet Identifier.

The SetPacketID function generates a pseudorandom integer using TimeLocal() to generate the initial state. To make our life easier while testing we defined a boolean variable TEST. When this variable is true, the function sets the value of 1 as the packet ID.

```
//+------------------------------------------------------------------+
//|            SetPacketID                                           |
//+------------------------------------------------------------------+
#define TEST true

void SetPacketID(uchar& buf[], int start_idx)
  {
// MathRand - Before the first call of the function, it's necessary to call
// MathSrand to set the generator of pseudorandom numbers to the initial state.
   MathSrand((int)TimeLocal());
   int packet_id = MathRand();
   if(ArrayResize(buf, buf.Size() + 2) < 0)
     {
      printf("ERROR: failed to resize array at %s", __FUNCTION__);
      return;
     }
   buf[start_idx] = (uchar)packet_id >> 8; // MSB
   buf[start_idx + 1] = (uchar)(packet_id % 256) & 0xff; //LSB
//--- if testing, set packet ID to 1
   if(TEST)
     {
      Print("WARN: SetPacketID TEST true fixed ID = 1");
      buf[start_idx] = 0; // MSB
      buf[start_idx + 1] = 1; //LSB
     }
  }
```

As you can see we also have a WARNing in place, just in case.

Properties

In [part 4 of this series of articles](https://www.mql5.com/en/articles/13651), we saw in detail what Properties are and their role as part of the MQTT 5.0 Extensibility Mechanisms. Here we will describe how we are implementing them, with special attention to the different [data types](https://en.wikipedia.org/wiki/Data_type "https://en.wikipedia.org/wiki/Data_type") encoding.

There are six types of **data representation** that are used to encode the Properties values, in a MQTT 5.0 Control Packet:

1. One Byte Integer, which are 8-bit unsigned integers
2. Two Byte Integers, which are 16-bit unsigned integers in [big-endian order, also called network order](https://www.mql5.com/en/book/common/maths/maths_byte_swap)
3. Four Byte Integers, which are 32-bit unsigned integers also in big-endian order
4. Variable Byte Integers, which use the minimum number of up to four bytes to represent a value between 0 and 268,435,455
5. Binary Data between 0 and 65,535 in length
6. UTF-8 encoded strings, that can also be used to encode a key:value pair in the User Properties

The following table shows the available PUBLISH properties and their respective data representation.

| Property | Data Representation |
| --- | --- |
| Payload Format Indicator | One Byte Integer |
| Message Expiry Interval | Four Byte Integer |
| Topic Alias | Two Byte Integer |
| Response Topic | UTF-8 Encoded String |
| Correlation Data | Binary Data |
| User Property | UTF-8 Encoded String Pair |
| Subscription Identifier | Variable Byte Integer |
| Content Type | UTF-8 Encoded String |

Table 01 - PUBLISH Properties and their Respective Data Representation in MQTT 5.0

Our Property identifiers were included in our Defines.mqh header.

```
//+------------------------------------------------------------------+
//|              PROPERTIES                                          |
//+------------------------------------------------------------------+
/*
The last field in the Variable Header of the CONNECT, CONNACK, PUBLISH, PUBACK, PUBREC,
PUBREL, PUBCOMP, SUBSCRIBE, SUBACK, UNSUBSCRIBE, UNSUBACK, DISCONNECT, and
AUTH packet is a set of Properties. In the CONNECT packet there is also an optional set of Properties in
the Will Properties field with the Payload
*/
#define MQTT_PROP_IDENTIFIER_PAYLOAD_FORMAT_INDICATOR          0x01 // (1) Byte
#define MQTT_PROP_IDENTIFIER_MESSAGE_EXPIRY_INTERVAL           0x02 // (2) Four Byte Integer
#define MQTT_PROP_IDENTIFIER_CONTENT_TYPE                      0x03 // (3) UTF-8 Encoded String
#define MQTT_PROP_IDENTIFIER_RESPONSE_TOPIC                    0x08 // (8) UTF-8 Encoded String
#define MQTT_PROP_IDENTIFIER_CORRELATION_DATA                  0x09 // (9) Binary Data
#define MQTT_PROP_IDENTIFIER_SUBSCRIPTION_IDENTIFIER           0x0B // (11) Variable Byte Integer
#define MQTT_PROP_IDENTIFIER_SESSION_EXPIRY_INTERVAL           0x11 // (17) Four Byte Integer
.
.
.
```

Payload Format Indicator

The Payload Format Indicator can be a 0 or 1 value, meaning raw bytes or UTF-8 encoded string respectively. If not present it is assumed to be 0 (raw bytes).

Although this field could be set directly on the m\_props member array, we opted for using an auxiliary local buffer as an intermediary to be consistent with the majority of the properties that require some kind of manipulation before being copied to the final properties array.

```
void CPktPublish::SetPayloadFormatIndicator(PAYLOAD_FORMAT_INDICATOR format)
  {
   uchar aux[2];
   aux[0] = MQTT_PROP_IDENTIFIER_PAYLOAD_FORMAT_INDICATOR;
   aux[1] = (uchar)format;
   ArrayCopy(m_props, aux, m_props.Size());
  }
```

Although there are only two possible values for this property, we chose to assign a symbolic value for them for the sake of readability.

```
enum PAYLOAD_FORMAT_INDICATOR
  {
   RAW_BYTES   = 0x00,
   UTF8        = 0x01
  };
```

The use of this symbolic value makes the method calling explicit for the library end user.

```
cut.SetPayloadFormatIndicator(RAW_BYTES);
cut.SetPayloadFormatIndicator(UTF8);
```

Message Expiry Interval

The Message Expiry Interval is represented as a four-byte integer. It is worth remembering that this representation is different from that of a variable byte integer. While the latter will use the minimum number of bytes required to represent the value, the former will always be represented using the whole four bytes.

```
void CPktPublish::SetMessageExpiryInterval(uint msg_expiry_interval)
  {
   uchar aux[4];
   aux[0] = MQTT_PROP_IDENTIFIER_MESSAGE_EXPIRY_INTERVAL;
   ArrayCopy(m_props, aux, m_props.Size(), 0, 1);
   EncodeFourByteInteger(msg_expiry_interval, aux);
   ArrayCopy(m_props, aux, m_props.Size());
  }
```

Our function to encode the four-byte integer follows a well-known pattern of power of two right-shifts to ensure the required big-endian order (or network order).

```
void EncodeFourByteInteger(uint val, uchar &dest_buf[])
  {
   ArrayResize(dest_buf, 4);
   dest_buf[0] = (uchar)(val >> 24) & 0xff;
   dest_buf[1] = (uchar)(val >> 16) & 0xff;
   dest_buf[2] = (uchar)(val >> 8) & 0xff;
   dest_buf[3] = (uchar)val & 0xff;
  }
```

Topic Alias

The Topic Alias property can be used to reduce the packet size. It is restricted to each network connection and is part of the MQTT session state. So, our function to set the Topic Alias can be considered as a stub as it is now. It must be completed when dealing with the Session State.

```
void CPktPublish::SetTopicAlias(ushort topic_alias)
  {
   uchar aux[2];
   aux[0] = MQTT_PROP_IDENTIFIER_TOPIC_ALIAS;
   ArrayCopy(m_props, aux, m_props.Size(), 0, 1);
   EncodeTwoByteInteger(topic_alias, aux);
   ArrayCopy(m_props, aux, m_props.Size());
  }
```

Our function to encode the two-byte integer follows the same well-known pattern we used to encode four-byte integers, i.e., power of two right-shifts to ensure the required big-endian order.

```
void EncodeTwoByteInteger(uint val, uchar &dest_buf[])
  {
   ArrayResize(dest_buf, 2);
   dest_buf[0] = (uchar)(val >> 8) & 0xff;
   dest_buf[1] = (uchar)val & 0xff;
  }
```

Response Topic

The Response Topic property is not part of the publish/subscribe pattern. Instead, it is part of the request/response interaction over MQTT. As you can see, our function uses two auxiliary buffers, one to host the property identifier and the other buffer to host the encoded UTF-8 string. The same will occur with other UTF-8 encoded strings because our string encoder function doesn’t have a third parameter to address the destination buffer start index. This may be solved with an overload in the next versions.

```
void CPktPublish::SetResponseTopic(const string response_topic)
  {
   uchar aux[1];
   aux[0] = MQTT_PROP_IDENTIFIER_RESPONSE_TOPIC;
   ArrayCopy(m_props, aux, m_props.Size());
   uchar buf[];
   EncodeUTF8String(response_topic, buf);
   ArrayCopy(m_props, buf, m_props.Size());
  }
```

Correlation Data

The Correlation Data property is also part of the request/response interaction over MQTT, not part of the publish/subscribe pattern. Since its value is binary data, our function is simply copying the data passed as an argument to m\_props byte array after setting the property identifier.

```
void CPktPublish::SetCorrelationData(uchar &binary_data[])
  {
   uchar aux[1];
   aux[0] = MQTT_PROP_IDENTIFIER_CORRELATION_DATA;
   ArrayCopy(m_props, aux, m_props.Size());
   ArrayCopy(m_props, binary_data, m_props.Size());
  }
```

User Property

The User Property is the most flexible MQTT 5.0 property because it can be used to transmit UTF-8 encoded key:value pairs with application-specific semantics.

“Non-normative comment

This property is intended to provide a means of transferring application layer name-value tags whose meaning and interpretation are known only by the application programs responsible for sending and receiving them”

Our function is using three auxiliary buffers to encode this property because, currently, our UTF-8 string encoder doesn’t have a third parameter to address the destination buffer start index. This may be solved with an overload in the next versions. (see above Response Topic.)

```
void CPktPublish::SetUserProperty(const string key, const string val)
  {
   uchar aux[1];
   aux[0] = MQTT_PROP_IDENTIFIER_USER_PROPERTY;
   ArrayCopy(m_props, aux, m_props.Size());
   uchar key_buf[];
   EncodeUTF8String(key, key_buf);
   ArrayCopy(m_props, key_buf, m_props.Size());
   uchar val_buf[];
   EncodeUTF8String(val, val_buf);
   ArrayCopy(m_props, val_buf, m_props.Size());
  }
```

Subscription Identifier

Our function to set the Subscription Identifier property starts checking if the argument passed is between 1 and 268,435,455, which are the accepted values for this property. If it is not we print/log an error message and return immediately.

```
void CPktPublish::SetSubscriptionIdentifier(uint subscript_id)
  {
   if(subscript_id < 1 || subscript_id > 0xfffffff)
     {
      printf("Error: " + __FUNCTION__ +  "Subscription Identifier must be between 1 and 268,435,455");
      return;
     }
   uchar aux[1];
   aux[0] = MQTT_PROP_IDENTIFIER_SUBSCRIPTION_IDENTIFIER;
   ArrayCopy(m_props, aux, m_props.Size());
   uchar buf[];
   EncodeVariableByteInteger(subscript_id, buf);
   ArrayCopy(m_props, buf, m_props.Size());
  }
```

Content Type

The value of the Content Type property is defined by the application. “MQTT performs no validation of the string except to ensure it is a valid UTF-8 Encoded String.”

```
void CPktPublish::SetContentType(const string content_type)
  {
   uchar aux[1];
   aux[0] = MQTT_PROP_IDENTIFIER_CONTENT_TYPE;
   ArrayCopy(m_props, aux, m_props.Size());
   uchar buf[];
   EncodeUTF8String(content_type, buf);
   ArrayCopy(m_props, buf, m_props.Size());
  };
```

Payload

The last field in the PUBLISH variable header is the payload properly said. A payload of zero length is valid. Our function is nothing more than a wrapper around our UTF-8 string encoder, following the same pattern of using an auxiliary buffer to be further copied to the m\_payload member.

```
void CPktPublish::SetPayload(const string payload)
  {
   uchar aux[];
   EncodeUTF8String(payload, aux);
   ArrayCopy(m_payload, aux, m_props.Size());
  }
```

The final Build method

The purpose of the Build() method is to merge the Fixed Header, the Topic Name, the Packet Identifier, the Properties, and the Payload in the final packet, while encoding both the Property(ies) Length and the packet Remaining Length as variable byte integer.

We first check for the presence of the mandatory Topic Name. If its length is zero we print/log the error and return immediately.

```
void CPktPublish::Build(uchar &pkt[])
  {
   if(m_topname.Size() == 0)
     {
      printf("Error: " + __FUNCTION__ + " topic name is mandatory");
      return;
     }
   ArrayResize(pkt, 2);
```

Then we set the Fixed Header first byte with the Control Packet type and the respective PUBLISH flags.

```
// pkt type with publish flags
   pkt[0] = (uchar)PUBLISH << 4;
   pkt[0] |= m_pubflags;
```

We then copy the m\_topname array to the final packet and set/copy the Packet Identifier if QoS > 0.

```
// topic name
   ArrayCopy(pkt, m_topname, pkt.Size());
// QoS > 0 require packet ID
   if((m_pubflags & 0x06) != 0)
     {
      SetPacketID(pkt, pkt.Size());
     }
```

Next, we encode the Property(ies) Length as a variable byte integer.

```
// properties length
   uchar buf[];
   EncodeVariableByteInteger(m_props.Size(), buf);
   ArrayCopy(pkt, buf, pkt.Size());
```

We copy the properties and the payload from their class members to the final packet array.

```
// properties
   ArrayCopy(pkt, m_props, pkt.Size());
// payload
   ArrayCopy(pkt, m_payload, pkt.Size());
```

Finally, we set the packet Remaining Length encoded as a variable byte integer.

```
// remaining length
   m_remlen += pkt.Size() - 2;
   uchar aux[];
   EncodeVariableByteInteger(m_remlen, aux);
   ArrayCopy(pkt, aux, 1);
  }
```

**The PUBACK Control Packet**

As we saw above while implementing our CPublish class, any PUBLISH packet with QoS 1 requires a non-zero Packet Identifier. This packet ID will be returned in the corresponding PUBACK packet. It is this ID that allows our client to know if the previously sent PUBLISH packet was delivered, or if there was an error. Be it a successful delivery, or a failure, the PUBACK is the trigger that we will be using to update the Session State. We will update the Session State based on the Reason Code(s).

The PUBACK packet will return one of nine Reason Code(s).

SUCCESS - Everything is fine with the message. It was accepted and the publication is ongoing. “Success” here means that the receiver has accepted the ownership of the message. This is the only Reason Code that can be implicit, that is, it is the only Reason Code that can be omitted. A PUBACK with only a packet ID MUST be interpreted as a successful QoS 1 delivery.

“The Client or Server sending the PUBACK packet MUST use one of the PUBACK Reason Codes \[MQTT-3.4.2-1\]. The Reason Code and Property Length can be omitted if the Reason Code is 0x00 (Success) and there are no Properties.”

NO MATCHING SUBSCRIBERS - Everything is fine with the message. It was accepted and the publication is ongoing, but nobody is subscribed to its topic name. This Reason Code is sent only by the broker and is optional, meaning, the broker MAY send this Reason Code instead of SUCCESS.

UNSPECIFIED ERROR - The message is rejected, but the publisher doesn’t want to reveal the reason or none of the other more specific Reason Codes are suitable to describe the reason.

IMPLEMENTATION SPECIFIC ERROR - Everything is fine with the message, but the publisher doesn’t want to publish it. The Standard doesn’t offer additional details about the semantics of this Reason Code, but we may infer that the reason for not publishing is not in the scope of the protocol, meaning, it is application-specific.

NOT AUTHORIZED - Self-explanatory.

TOPIC NAME INVALID - Everything is fine with the message, including the Topic Name, which is a well-formed, well-encoded UTF-8 string. But the publisher, be it the client or the broker, doesn’t accept this Topic Name. Again, we may infer that the reason for not publishing is application-specific.

PACKET IDENTIFIER IN USE - Everything is fine with the message, but there is a possible mismatch in the Session State between the client and the broker because the packet ID we sent in PUBLISH is already in use.

QUOTA EXCEEDED - Self-explanatory. Once again, the reason for rejection is not in the scope of the protocol. It is application-specific.

PAYLOAD FORMAT INVALID - Everything is fine with the message, but the Payload Format Indicator property we sent in our PUBLISH is different from the actual payload format.

Besides the Reason Code, the PUBACK packet may have a Reason String and a User Property.

Reason String is a human-readable UTF-8 encoded string aimed at helping in diagnostics. It is not intended to be parsed by the receiver. Instead, its purpose is to carry additional information that can be logged, printed, attached to reports, etc. It is worth noting that any compliant server or client will not send the Reason String if its inclusion increases the packet size beyond the Maximum Packet Size specified at the connection time (CONNECT packet).

The PUBACK can also have any number of key:value pairs encoded as User Property(ies). These pairs can be used to provide additional information about the error and are application-specific too. That is, the protocol doesn’t define their semantics.

Our client “MUST treat the PUBLISH packet as “unacknowledged” until it has received the corresponding PUBACK packet from the receiver.”

### The CPuback Class

Our CPuback class follows the same blueprint as the CPublish class. It also implements the IControlPacket interface that is standing as our stub root for the object hierarchy.

A PUBACK packet is sent as a response for PUBLISH packets with QoS 1. Its two-byte fixed header has only the control packet identifier on the first byte and the packet’s remaining length on the second byte. Its bit flags are all set to RESERVED in this version of the protocol.

![Structure of the Fixed Header of an MQTT-5.0 PUBACK packet](https://c.mql5.com/2/71/Structure-of-the-Fixed-Header-of-an-MQTT-5.0-PUBACK-packet.PNG)

Fig. 03 - Structure-of-the-Fixed-Header-of-an-MQTT-5.0-PUBACK-packet

“The Variable Header of the PUBACK Packet contains the following fields in the order: Packet Identifier from the PUBLISH packet that is being acknowledged, PUBACK Reason Code, Property Length, and the Properties.”

![Structure of the Variable Header of an MQTT-5.0 PUBACK packet](https://c.mql5.com/2/71/Structure-of-the-Variable-Header-of-an-MQTT-5.0-PUBACK-packet.PNG)

Fig. 04 - Structure-of-the-Variable-Header-of-an-MQTT-5.0-PUBACK-packet

Until now, we’ve been dealing with our Client only as a sender; from now on, we need to take into account the receiver role too. That is because

“The delivery protocol is symmetric, \[...\] the Client and Server can each take the role of either sender or receiver.“

We need to write a test for a function that gets the identifier of the packet being acknowledged

1. from the returned packet sent by the broker when receiving a PUBACK
2. or from our persistence system when sending a PUBACK

A PUBLISH packet with QoS 1 has no meaning without its corresponding PUBACK, which in turn requires some kind of persistence to store the packet ID of its corresponding PUBLISH packet. But, although we already know that at some point we will need to use a real database as a persistence layer, at this point we don’t need it yet. To test and develop our function all that we need is something that acts like a database, something that when queried returns what would be the identifier of the PUBLISH packets pending of acknowledgement. To avoid surprises, let’s create a single function called GetPendingPublishIDs(ushort &result\[\]) and save it on a file named DB.mqh.

```
void GetPendingPublishIDs(ushort &result[])
  {
   ArrayResize(result, 3);
   result[0] = 1;
   result[1] = 255; // one byte
   result[2] = 65535; // two bytes
  }
```

With our “persistence layer” in place, we can concentrate on the task at hand: to write a function that when passed a PUBACK byte array (packet) sent by the broker gets the identifier of the PUBLISH being acknowledged and checks it against the pending PUBLISH IDs stored on our persistence layer. If there is an ID match it returns ‘True’. Later, when implementing the protocol Operational Behavior, we will release this matching ID from the real store.

Given the above PUBACK variable header structure, all that we need for now is to read the first two bytes to get the ID of the packet being acknowledged.

```
ushort CPuback::GetPacketID(uchar &pkt[])
  {
   return (pkt[0] * 256) + pkt[1];
  }
```

Let’s remember that the packet identifier is encoded as a two-byte integer in big-endian (or network) order with the most significant byte (MSB) presented first. To encode it we used a left-shift bitwise operation (<<). To decode it we are multiplying the value of the most significant byte by 256 and adding the least significant byte.

The above function is enough for now. Later, when testing against a real broker in the open network, we might have to deal with endianness issues, but we will not test for them at this point. Let’s keep moving towards our attractive carrot, the task at hand.

```
bool CPuback::IsPendingPkt(uchar &pkt[])
{
   ushort pending_ids[];
   GetPendingPublishIDs(pending_ids);
   ushort packet_id = GetPacketID(pkt);
   for(uint i = 0; i < pending_ids.Size(); i++)
     {
      if(pending_ids[i] == packet_id)
        {
         return true;
        }
     }
   return false;
}
```

The above function receives a byte array as an argument. This byte array is the variable header of the PUBACK packet. It then stores in a local variable (pending\_ids) an array of packet identifiers from our store/database that were not yet acknowledged. Finally, it reads the packet ID on the byte array sent by the broker and compares it against that array of pending IDs. If the packet is in the array our function returns ‘True’ and we can release the ID.

The same logic will allow us to release PUBREC, PUBREL, and PUBCOMP packet identifiers for PUBLISH with QoS 2. Also, later we will replace our fake one-function-in-a-file “persistence layer” with a real database, but the function’s main logic will stay. At this point, another developer could be working on the persistence layer while we develop our packet classes in a totally independent way.

We also need to be able to read Reason Code(s) from the PUBACK variable header. Since this field has a fixed position and size, all that we need is to read that specific byte.

```
uchar CPuback::GetReasonCode(uchar &pkt[])
  {
   return pkt[2];
  }
```

Because we are working only on the receiver side of our client - i.e., not sending PUBACKs yet - the above functions are enough for our next functional tests. Now, against a real broker.

### Conclusion

Continuous refactoring is part of TDD practice. It aims to achieve not only fully functional but also a clean code: single responsibility units and functions (classes and methods here), readable identifiers (class, methods, and variable names), and avoid redundancy (“don’t repeat yourself”). It is a process, not a one-step task. So, we already know, for sure, that we will be refactoring continuously until we have a fully functional MQTT 5.0 client.

Now we are ready to start writing our first functional test against a real MQTT broker to see if our CONNECT, CONNACK, PUBLISH, and PUBACK packets are working as expected.

PUBACK packets are the counterpart of the PUBLISH packets with QoS 1. PUBLISH packets with QoS 2 will require PUBREC, PUBCOMP, and PUBREL packets as their counterpart. They are the subject of our next article.

If you have a good understanding of MQL5 and can contribute to the development of this open-source MQTT client, please, drop a note in the comments below or in our Community Chat.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/14391.zip "Download all attachments in the single ZIP archive")

[MQTT.zip](https://www.mql5.com/en/articles/download/14391/mqtt.zip "Download MQTT.zip")(20.83 KB)

[Tests.zip](https://www.mql5.com/en/articles/download/14391/tests.zip "Download Tests.zip")(16.88 KB)

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

**[Go to discussion](https://www.mql5.com/en/forum/464031)**

![How to create a simple Multi-Currency Expert Advisor using MQL5 (Part 7): ZigZag with Awesome Oscillator Indicators Signal](https://c.mql5.com/2/73/How_to_create_a_simple_Multi-Currency_Expert_Advisor_using_MQL5__Part_7__LOGO.png)[How to create a simple Multi-Currency Expert Advisor using MQL5 (Part 7): ZigZag with Awesome Oscillator Indicators Signal](https://www.mql5.com/en/articles/14329)

The multi-currency expert advisor in this article is an expert advisor or automated trading that uses ZigZag indicator which are filtered with the Awesome Oscillator or filter each other's signals.

![Neural networks made easy (Part 63): Unsupervised Pretraining for Decision Transformer (PDT)](https://c.mql5.com/2/60/Neural_networks_are_easy_wPart_636_Logo.png)[Neural networks made easy (Part 63): Unsupervised Pretraining for Decision Transformer (PDT)](https://www.mql5.com/en/articles/13712)

We continue to discuss the family of Decision Transformer methods. From previous article, we have already noticed that training the transformer underlying the architecture of these methods is a rather complex task and requires a large labeled dataset for training. In this article we will look at an algorithm for using unlabeled trajectories for preliminary model training.

![MQL5 Wizard Techniques you should know (Part 13): DBSCAN for Expert Signal Class](https://c.mql5.com/2/73/MQL5_Wizard_Techniques_you_should_know_Part_13_DBSCAN_for_Expert_Signal_Class___LOGO.png)[MQL5 Wizard Techniques you should know (Part 13): DBSCAN for Expert Signal Class](https://www.mql5.com/en/articles/14489)

Density Based Spatial Clustering for Applications with Noise is an unsupervised form of grouping data that hardly requires any input parameters, save for just 2, which when compared to other approaches like k-means, is a boon. We delve into how this could be constructive for testing and eventually trading with Wizard assembled Expert Advisers

![Advanced Variables and Data Types in MQL5](https://c.mql5.com/2/73/Advanced_Variables_and_Data_Types_in_MQL5___LOGO.png)[Advanced Variables and Data Types in MQL5](https://www.mql5.com/en/articles/14186)

Variables and data types are very important topics not only in MQL5 programming but also in any programming language. MQL5 variables and data types can be categorized as simple and advanced ones. In this article, we will identify and learn about advanced ones because we already mentioned simple ones in a previous article.

[![](https://www.mql5.com/ff/sh/jup0jccfs9655z9z2/01.png)Learn to create your own robotsRead our book "MQL5 Programming for Traders"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/book%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.algobook%26utm_content=visit.page%26utm_campaign=algobook.promo.04.2024&a=rsxjstxkzbrlgjjrxaglpezpvrjflnvw&s=7224440013c3dbc50ba9cc078cd015fabca36df446b8e75028d6b30234663872&uid=&ref=https://www.mql5.com/en/articles/14391&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5068138615961220592)

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