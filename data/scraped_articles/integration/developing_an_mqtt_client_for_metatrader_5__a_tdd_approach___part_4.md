---
title: Developing an MQTT client for Metatrader 5: a TDD approach — Part 4
url: https://www.mql5.com/en/articles/13651
categories: Integration
relevance_score: 6
scraped_at: 2026-01-23T17:22:18.260210
---

[![](https://www.mql5.com/ff/sh/zf7a2k61x98jzh89z2/01.png)Speed up your tradingUse our high-speed VPS for MetaTrader 4 and 5Learn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=qtrrsuiwuicrscmckjynyanztbditglq&s=c617dc80d90cfd3783ec1345eec2b419b281f10fec6eac77b3218984ac337259&uid=&ref=https://www.mql5.com/en/articles/13651&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5068149091386455576)

MetaTrader 5 / Integration


_“Software entities should be open for extension, but closed for modification.” (Open-Closed Principle for Object-Oriented Programming)_

### Introduction

To be sure we are all on the same page, a quick recap might be of some help. In the [first part of this series](https://www.mql5.com/en/articles/12857), we started writing naive tests for a function that would be able to generate an MQTT Control Packet Fixed Header. We started with the first bit of the first byte. If there is a baby step, that was one, for sure.

In the [second part](https://www.mql5.com/en/articles/13334), we organized some shared functions and definitions in two header files.

In the [third part](https://www.mql5.com/en/articles/13388), we start reading the CONNACK’s Acknowledge Flags and Connect Reason Codes. It was our first contact with the Standard’s Operational Behavior section.

Until this point, everything was static and tied to a connection attempt. If the first byte is wrong, one gets a broker failure response and may try again to connect with a well-formed packet.

Properties are another story. They are dynamic attributes of the MQTT Application Message and can change after the connection. A broker's Maximum QoS may change temporarily due to operational reasons. A Receive Maximum may have been changed due to network bottlenecks. Besides that, some properties are dynamic by design, like Content-Type, all the Will Properties, and the User Property. These properties will be changing all the time.

The OASIS Standard is pretty clear in their specifications, but a lot of room is left for client developers to decide how to read, react, persist, and update the current properties. By the way, the persistence layer for Session state management is one hundred percent client developer responsibility. We will have to implement a persistence layer to properly manage properties between Session(s). Our algorithm choices here will be critical for conformance, robustness, and performance.

For future library maintainers, the comments in the section “How we are reading MQTT v5.0 Properties in MQL5” may be of use as informal documentation. Some library end-users may also benefit from those comments. In that section, we will look at the properties from a library developer perspective. We will deal with their datatypes, identifiers, and location on the byte array to better describe how we are reading them. Below we will look at the properties from the perspective of a user of the library. We will try to describe their semantics in each of the possible use cases.

Note: Unless otherwise stated, all quotes are from the [OASIS Standard](https://www.mql5.com/go?link=https://docs.oasis-open.org/mqtt/mqtt/v5.0/mqtt-v5.0.html "https://docs.oasis-open.org/mqtt/mqtt/v5.0/mqtt-v5.0.html").

### What are Properties in MQTT v5.0

Properties are part of the ‘extensibility mechanisms’ that were added in MQTT v5.0. They did not exist in the previous v3.1.1, which was the latest version before this major upgrade. They are ubiquitous in MQTT v5.0. But, what are MQTT properties? Properties of what, exactly?

Properties of the Application Message is the answer. In OASIS Standard’s terminology, the Application Message is

_“The data carried by the MQTT protocol across the network for the application. When an Application Message is transported by MQTT it contains payload data, a Quality of Service (QoS), a collection of Properties, and a Topic Name.”_ (emphasis is ours)

Please, take a look at the yellow rectangle representing the ‘Payload data’ in Figure 1, below. There is an important terminology distinction that we want to bring to your attention here.

![MQTT 5.0 Application Message Abstract Diagram](https://c.mql5.com/2/59/MQTT-v5-application-message-abstract-diagram.jpg)

Fig.01 - MQTT 5.0 Application Message Abstract Diagram

In the context of a message-sharing protocol, when we see the word ‘message’ we are used to thinking about the user message, frequently a text message. More often than not, we do not think about the message as the application as a whole.

But here, the message sent by the users via MQTT is part of the Payload data and Properties are part of the protocol abstract model named Application Message. Thus, when we send a user message via MQTT we can have not only properties related to that ‘user message’, but we can also have properties related to the Application Message as a whole: properties for the connection, properties for publishing, properties for subscribing and unsubscribing to topics, properties for the authentication, and so on.

Besides that, there are the Will Properties attached to the Will Message.

"The Will Message consists of the Will Properties, Will Topic, and Will Payload fields in the CONNECT Payload. "

This terminology may be a bit confusing when one starts implementing the protocol, but we will do our best to make it as clear as possible.

### What are Properties used for?

Besides carrying Payload metadata, properties can be used for configuring every aspect of the interaction between the Client and the Server (broker) and also the interaction between different Clients. From connection to disconnection, they can be used to set content type format, request information from the broker, define a message expiration timespan, choose an authentication method, and even perform a server redirection, among other use cases. As we can see in the tables below, except for the PINGREQ and PINGRESP packets that are used to refresh the Keep Alive period, all packet types can carry some specific properties according to the packet context.

The User Property is a special case of a property that can be used in all packets, and whose meaning is defined by the application, meaning its semantics are not defined by the protocol. We will do a quick introduction to the User Property in the last section, where we talk about how properties can be used to extend the protocol.

Although properties’ names are explicit in their purpose, we need to know:

- when they can be used
- when they must be used
- and what happens if they are set wrong

With the aim of easing the reading and understanding of the descriptions that follow, in the table below we have grouped them under different colors according to their functionality. Please, note that the grouping is somewhat arbitrary because the use of properties overlaps between different packet types.

In the descriptions that follow, we are using the terms MUST and MAY as they are used by the OASIS Standard, which in turn uses them as described in [IETF RFC 2119](https://www.mql5.com/go?link=https://www.rfc-editor.org/info/rfc2119 "https://www.rfc-editor.org/info/rfc2119").

|  | Connection Properties |
| --- | --- |
| CONNECT | Session Expiry Interval, Receive Maximum, Maximum Packet Size, Topic Alias Maximum, Request Response Information, Request Problem Information, User Property, Authentication Method, Authentication Data |
| CONNECT Payload | Will Delay Interval, Payload Format Indicator, Message Expiry Interval, Content Type, Response Topic, Correlation Data, User Property |
| CONNACK | Session Expiry Interval, Receive Maximum, Maximum QoS, Retain Available, Maximum Packet Size, Assigned Client Identifier, Topic Alias Maximum, Reason String, User Property, Wildcard Subscription Available, Subscriptions Identifiers Available, Shared Subscription Available, Server Keep Alive, Response Information, Server Reference, Authentication Method, Authentication Data |
| DISCONNECT | Session Expiry Interval, Reason String, User Property, Server Reference |

Table 1: MQTT v5.0 - Properties Grouped By Functionality - Connection Properties

Assigned Client Identifier \- MUST be set on CONNECT. If not set the broker MAY assign an identifier on CONNACK.

The Client Identifier is mandatory on CONNECT packets. However, the broker is allowed to accept an identifier with a length of zero bytes and assign an identifier to the client. In this case, the broker will return it on a CONNACK packet under this property.

Maximum Packet Size – MAY not be set, but cannot be set to zero.

The Maximum Packet Size “is the total number of bytes in an MQTT Control Packet”. It is used by the  Client and the Broker to define the maximum packet size they are willing to accept. If our client sets this property and the broker sends a packet greater than this limit, we must DISCONNECT with Reason Code 0x95 (Packet too large). The broker will not send Reason String(s) or some User Properties if its inclusion would make the packet greater than this property.

Maximum QoS – used by the broker on CONNACK’s and MAY not be set

The Maximum QoS informs the client about the broker's capability to handle QoS levels. If the broker accepts QoS Level 2, this property will not be set. Our client MUST abide by this server limitation by not sending PUBLISH packets with a higher QoS Level. We are allowed to support only QoS Level 0 in our client and still be conformant.

Non-normative comment

_"A Client does not need to support QoS 1 or QoS 2 PUBLISH packets. If this is the case, the Client simply restricts the maximum QoS field in any SUBSCRIBE commands it sends to a value it can support."_

Message Expiry Interval – MAY not be set.

The Message Expiry Interval is also part of the PUBLISH properties. It sets the lifetime of the Will Message. If not set, when the broker publishes the Will Message it will have no defined expiration time.

_"If the Message Expiry Interval has passed and the Server has not managed to start onward delivery to a matching subscriber, then it MUST delete the copy of the message for that subscriber."_

It can also be set on the Will Properties for the CONNECT Payload. If not set, the Message does not expire.

Payload Format Indicator – MUST be set if we are sending character data.

The Payload Format Indicator can be set on the Will Properties for the CONNECT Payload. Here it indicates if the Will Message is UTF-8 Encoded Character Data or if it is ‘unspecified bytes’.

“The Server MAY validate that the Will Message is of the format indicated, and if it is not send a CONNACK with the Reason Code of 0x99 (Payload format invalid) “

It is part of the PUBLISH properties and indicates the format of the Payload. The broker validation is optional too. But if it validates the payload format, we can expect a PUBACK, PUBREC, or DISCONNECT with the same Reason Code (0x99) if the format is different from the advertised.

If not set, the Payload format is assumed to be of ‘unspecified bytes’. What is to say that if we are sending character data, setting this property is mandatory.

Reason String – it MAY be used on all ACK’s, DISCONNECT, and AUTH by the client or the broker.

The Reason String is one of the new features in MQTT v5.0. It may be used to supplement the Reason Code with a human-readable diagnostic tool. If present, it can be used for logging, for example. We can request the broker to not send it by setting the ‘Request Problem Information’ to zero on CONNECT properties.

Receive Maximum – MAY not be set, but cannot be set to zero.

Our client can use this property on CONNECT to limit the number of QoS 1 and QoS 2 publications we are willing to handle concurrently on the current network connection. The broker may set it on CONNACK. If not set, it defaults to 65,535.

On QoS 0 there is no need to wait for PUBACK (QoS 1) or PUBCOMP (QoS 2) because, as we know, QoS 0 is ‘fire and forget’. This property sets how many messages our client or the broker is willing to send/receive before receiving the corresponding PUBACK or PUBCOMP. We can think of it as a means to say how many messages can be in a ‘pending confirmation’ state until new messages can be sent.

Request Problem Information – MAY be set on CONNECT.

The Request Problem Information is used to inform the server that we want to receive Reason String(s) and User Properties in case of failures. If we say nothing – by not setting it – then the broker can send them.

Request Response Information – MAY be set on CONNECT.

The Request Response Information is part of the Request/Response interaction via MQTT, instead of the regular pub/sub interaction. If unset we inform the broker that we do not want it to send response information. Otherwise, setting it we are allowing the broker to send us that response. Note that the broker is allowed to not send response information even if we request it. If this property is absent the value defaults to unset.

Response Information – MAY be set on CONNECT.

The Response Information is part of the Request/Response interaction via MQTT, instead of the regular pub/sub interaction.

Non-normative comment

_"A common use of this is to pass a globally unique portion of the topic tree which is reserved for this Client for at least the lifetime of its Session. This often cannot just be a random name as both the requesting Client and the responding Client need to be authorized to use it. It is normal to use this as the root of a topic tree for a particular Client. For the Server to return this information, it normally needs to be correctly configured. Using this mechanism allows this configuration to be done once in the Server rather than in each Client."_

Response Topic – MAY be set on CONNECT or PUBLISH.

The Response Topic is part of the Request/Response interaction via MQTT, instead of the regular pub/sub interaction. If we include it, the broker interprets the Will Message as a Request. Differently from the Topic Filter used on SUBSCRIBE packets, the Response Topic cannot have wildcard characters.

_“A Request Message is an Application Message with a Response Topic.”_

So, this is the property that characterizes an Application Message as being part of a Request/Response interaction.

Retain Available – MAY be present on CONNACK.

The Retain Available property informs our client if the broker supports retained messages. If absent retained messages are available.

Server Keep Alive – MAY be present on CONNACK.

The Server Keep Alive property has precedence over our client Keep Alive requested on CONNECT. If this property is not present on CONNACK, then we can use our Keep Alive. Otherwise, the Server Keep Alive rules.

Server Reference – MAY be present on CONNACK or DISCONNECT.

The Server Reference informs our client about server redirection. It can refer to a temporary or permanent redirection. In both cases, the other server may be already known to our client or it will be specified using this property.

Non-normative comment

_Examples of the Server Reference are:_

_myserver.xyz.org_

_myserver.xyz.org:8883_

_10.10.151.22:8883 \[fe80::9610:3eff:fe1c\]:1883_

The broker is allowed to not ever send this property and our client is allowed to ignore it.

Session Expiry Interval – MAY be set on CONNECT.

The Session Expiry Interval determines how long to retain the session after a disconnect. If unset or absent the Session ends when the connection is closed. It is possible to set the Session to not expire by setting this property to UINT\_MAX. We must store the Session State if this property is greater than zero. We can check it on the Session Present flag on CONNACK.

This property can be useful when the network connection is intermittent, allowing our client to resume the Session whenever the network connection resumes.

Shared Subscription Available – MAY be present on CONNACK.

The Shared Subscription Available informs our client if the broker supports Shared Subscriptions. If absent, then the broker supports it.

Subscription Identifiers Available – MAY be present on CONNACK.

The Subscription Identifiers Available informs our client if the broker supports Subscription Identifiers. If absent, then the broker supports it.

Topic Alias Maximum – MAY be set on CONNECT and MAY be present on CONNACK.

The Topic Alias Maximum informs the broker of the maximum number of Topic Alias our client is willing to accept on this specific connection. If we set it to zero or leave it blank, the broker will not send any Topic Alias in this connection. The reverse is also true: if this property is not present on CONNACK or is present but its value is zero, our client must not send any Topic Alias.

Wildcard Subscription Available – MAY be present on CONNACK.

If this property is unset (set to zero) the broker doesn’t support Wildcard Subscriptions. In this case the broker will DISCONNECT after receiving a SUBSCRIBE requesting Wildcard Subscription. But even if the broker supports the feature, it is allowed to reject a particular subscribe request containing wildcard subscriptions and return a SUBACK with the same Reason Code 0xA2 (Wildcard Subscriptions not supported). If the property is absent in CONNACK the broker supports the feature.

Will Delay Interval – MAY be set on Will Properties of the CONNECT payload.

This property sets the delay in seconds to be observed by the broker before sending the Will Message. This property is particularly useful to avoid the sending of the Will Message under unstable or intermittent network connections.

|  | Publishing Properties |
| --- | --- |
| PUBLISH | Payload Format Indicator, Message Expiry Interval, Topic Alias, Response Topic, Correlation Data, User Property, Subscription Identifier, Content Type |
| PUBACK | Reason String, User Property |
| PUBREC | Reason String, User Property |
| PUBREL | Reason String, User Property |
| PUBCOMP | Reason String, User Property |

Table 2: MQTT v5.0 - Properties Grouped By Functionality - Publishing Properties

Topic Alias – MAY be set on PUBLISH

Topic Alias is also a new feature of MQTT v5.0. It allows the broker or the client to reduce the size of the packets by replacing the Topic Name with a small integer, the alias. The overhead reduction can be expressive, since Topic names are strings that may extend to 65,535 bytes (UINT\_MAX).

Correlation Data – MAY be set on PUBLISH and Will Properties

The Correlation Data is part of the Request/Response interaction via MQTT, instead of the regular pub/sub interaction. Its value only has meaning to the application (broker and clients). It is binary data used in Request/Response “by the sender of the Request Message to identify which request the Response Message is for when it is received.”

Content Type – MAY be set on PUBLISH and Will Properties

Content Type may also be used in CONNECT for setting the Will Message Content Type.

The broker only validates the encoding of the property itself. It is up to the client the meaning of this property.

|  | Subscribing/Unsubscribing Properties |
| --- | --- |
| SUBSCRIBE | Subscription Identifier, User Property |
| SUBACK | Reason String, User Property |
| UNSUBSCRIBE | User Property |
| UNSUBACK | Reason String, User Property |

Table 3: MQTT v5.0 - Properties Grouped By Functionality - Subscribing/Unsubscribing Properties

Subscription Identifier – MAY be set on SUBSCRIBE

This is a numeric identifier that can be set on SUBSCRIBE. It will be returned on the message by the broker, allowing the client(s) to determine which subscription(s) caused the message to be delivered. It can have the value of 1 to 268,435,455. MUST not be set to zero and MUST not be used in PUBLISH from Client to Server.

|  | Authentication Properties |
| --- | --- |
| AUTH | Authentication Method, Authentication Data, Reason String, User Property |

Table 4: MQTT v5.0 - Properties Grouped By Functionality - Authentication Properties

Without suprise, these properties may also be used on connections.

Authentication Method

Besides basic network authentication with username and password, MQTT v5.0 allows ‘Extended Authentication’. This property informs the method of choice. The method of choice is defined by the application developers. The broker will inform if the method is supported.

The Authentication Method is commonly a SASL mechanism, and using such a registered name aids interchange. However, the Authentication Method is not constrained to using registered SASL mechanisms

Authentication Data

This property is used by the client and the broker to exchange authentication data, according to the authentication method of choice.

### How we are reading MQTT v5.0 Properties in MQL5

Until now, in the previous parts of this series, we have been dealing with ‘per Session’ settings configured via bit flags, namely, the Connect Flags on CONNECT, the CONNACK Reason Code, and the CONNACK Session Present flag. Those settings are read/write/persisted once per Session. But with properties things are different. They are part of the Application Message and may carry large amounts of critical data in some application profiles. Thus, our client must be prepared to read and write properties all the time.

To write a test for reading properties sent by the server we need a sample byte array. We will start with a sample byte array for a CONNACK packet because it is the first packet our Client will be dealing with. As with all MQTT Control Packets, it has a two-byte fixed header, and a two-byte variable header, being one byte for the Connect Acknowledge Flags, and one byte for the Connect Reason Code. The properties are the last field in the CONNACK packet, and it has no packet identifier and no payload.

![MQTT 5.0 - Structure of a CONNACK packet](https://c.mql5.com/2/59/MQTT-v5-connack-structure-properties.JPG)

Fig. 02: Structure of an MQTT 5.0 CONNACK packet

From the Standard, we know that:

_“The set of Properties is composed of a Property Length followed by the Properties.”_

We also know that:

_“The Property Length is encoded as a Variable Byte Integer. The Property Length does not include the bytes used to encode itself, but includes the length of the Properties. If there are no properties, this MUST be indicated by including a Property Length of zero.”_

Thus, the Fixed Header remaining length and the Property Length, both encoded as a Variable Byte Integer are the first pieces of information we need to read before accessing the properties. If the Property Length is zero, there is nothing to be read.

So, our sample byte array may look like this for a CONNACK without properties:

uchar connack\_response\[\] = {2, X, 0, 0, 0};

Where X is the Fixed Header remaining length. The algorithm for decoding a Variable Byte Integer is provided by the Standard. In MQL5 this algorithm may be write like this:

```
uint DecodeVariableByteInteger(uint &buf[], uint idx)
  {
   uint multiplier = 1;
   uint value = 0;
   uint encodedByte;
   do
     {
      encodedByte = buf[idx];
      value += (encodedByte & 127) * multiplier;
      if(multiplier > 128 * 128 * 128)
        {
         Print("Error(Malformed Variable Byte Integer)");
         return -1;
        }
      multiplier *= 128;
     }
   while((encodedByte & 128) != 0);
   return value;
  };
```

Where buf\[idx\] represents the ‘next byte from stream’.

Although the algorithm for decoding a Variable Byte Integer is provided by the Standard, we wrote a very simple test for it too, just to be sure the implementation is working as expected at this stage:

```
bool TEST_DecodeVariableByteInteger()
  {
   Print(__FUNCTION__);
   uint buf[] = {1, 127, 0, 0, 0};
   uint expected = 127;
   uint result = DecodeVariableByteInteger(buf, 1);
   ZeroMemory(buf);
   return AssertEqual(expected, result);
  }
```

Obviously, for testing purposes the remaining length value will be hard coded. For the above CONNACK without any properties, it would be:

uchar connack\_response\[\] = {2, 3, 0, 0, 0};

A sample byte array for a CONNACK with a one-byte Payload Format Indicator property set to UTF-8 Encoded String payload format could be something like:

uchar connack\_response\_one\_byte\_property = {2, 5, 0, 0, 2, 1, 1};

As you can see, checking for the presence of properties in a CONNACK is pretty straightforward. We just need to read the fifth byte containing the Property Length. If it is non-zero we have properties.Our first test looks like this:

```
bool TestProtectedMethods::TEST_HasProperties_CONNACK_No_Props()
  {
   Print(__FUNCTION__);
//--- Arrange
   bool expected = false;
   uchar connack_no_props[5] = {2, 3, 0, 0, 0};
//--- Act
   CSrvResponse *cut = new CSrvResponse();
   bool result =  this.HasProperties(connack_no_props);
//--- Assert
   bool isTrue = AssertEqual(expected, result);
//--- cleanup
   delete cut;
   ZeroMemory(result);
   return  isTrue ? true : false;
  }
```

Please, take a look at our previous article to see [how we are testing protected methods](https://www.mql5.com/en/articles/13388), and also take a look at the attached code to see the FAIL variant of the test.

The first implementation, just enough to pass the tests at the current stage looks like this:

```
bool CSrvResponse::HasProperties(uchar &resp_buf[])
  {
   return resp_buf[4] != 0 ? true : false;
  }
```

Here, we are using the [ternary operator](https://www.mql5.com/en/docs/basis/operators/ternary) to keep the code at its minimum.

Note that we must take into account that the location of the Property Length byte depends on the packet type. That is because, although the properties are always the last field in the Variable Header, there are packets that require a two-byte Packet Identifier before the Property Length. In a CONNACK this is a non-issue.

“Then this code will not work for other packet types!”, you may say. And yes, you are right. But, please, remember that we are using a TDD approach here. One of the main benefits of this practice is to keep us focused on the task at hand, and NOT trying to cope with all the possible future issues while in the first stages of development. We will deal with other packet types when the time comes in and our test fails. Then, we will rewrite our test(s), eventually refactoring the code.

Although it could seem a bit counter-intuitive, one cannot write code as before once used to it. If not for other reasons, because it makes our job easier and even a joy. By the way, in the next part of this series, we will start writing and reading properties for PUBLISH packets. So stay tuned!

If the Property Length is non-zero, we can look for a Property Identifier in the next byte. The Property Identifier gives us the Property datatype.

_“A Property consists of an Identifier which defines its usage and data type, followed by a value.”_

```
uchar CSrvResponse::GetPropertyIdentifier(uchar &resp_buf[])
  {
   return resp_buf[5];
  }
```

The data type gives us the number of bytes to be read. The data type can be one of:

One Byte Integer

```
uchar CSrvResponse::ReadOneByteProperty(uchar &resp_buf[])
  {
   return resp_buf[6];
  }
```

Two Byte Integer

```
void CSrvResponse::ReadTwoByteProperty(uchar &resp_buf[], uchar &dest_buf[])
  {
   ArrayCopy(dest_buf, resp_buf, 0, 6, 2);
  }
```

Four Byte Integer

```
void CSrvResponse::ReadFourByteProperty(uchar &resp_buf[], uchar &dest_buf[])
  {
   ArrayCopy(dest_buf, resp_buf, 0, 6, 4);
  }
```

Variable Byte Integer  (only for Subscription Identifier)

```
void CSrvResponse::ReadVariableByteProperty(uint &resp_buf[], uint &dest_buf[], uint start_idx)
  {
   uint value = DecodeVariableByteInteger(resp_buf, start_idx);
   ArrayResize(dest_buf,value,7);
   ArrayFill(dest_buf, 0, 1, value);
  }
```

The reading/decoding (and writing/encoding) of this property requires a lot more than what this test is checking right now.

_"The Variable Byte Integer is encoded using an encoding scheme which uses a single byte for values up to 127. Larger values are handled as follows. The least significant seven bits of each byte encode the data, and the most significant bit is used to indicate whether there are bytes following in the representation. Thus, each byte encodes 128 values and a ‘ **continuation bit**’ "_(emphasis is ours)

We will deal with Variable Byte Integer properties when implementing SUBSCRIBE packets since it is used only for the Subscription Identifier property. Also the following three datatypes: UTF-8 encoded strings, binary data, and UTF-8 string pairs. They will be detailed in the context of the use of Request/Response and the implementation of the special case of the User Property.

UTF-8 encoded strings are prefixed with their length.

_“Each of these strings is prefixed with a Two Byte Integer length field that gives the number of bytes in UTF-8 encoded string itself, as illustrated in Figure 1.1 Structure of UTF-8 Encoded Strings below. Consequently, the maximum size of a UTF-8 Encoded String is 65,535 bytes._ _Unless stated otherwise all UTF-8 encoded strings can have any length in the range 0 to 65,535 bytes.”_

![MQTT-v5-utf8-encoded-strings-structure-OASIS](https://c.mql5.com/2/59/MQTT-v5-utf8-encoded-strings-structure-oasis.JPG)

Fig. 03: MQTT 5.0 - Structure of UTF-8 Encoded Strings - screen capture from OASIS table

It is worth noting that UTF-8 encoded strings must be validated for the presence of disallowed Unicode code points. (more on this later)

_"Section 1.6.4 describes the Disallowed Unicode code points, which should not be included in a UTF-8 Encoded String. A Client or Server implementation can choose whether to validate that these code points are not used in UTF-8 Encoded Strings such as the Topic Name or Properties."_

Binary Data are also prefixed with their length.

_“Binary Data is represented by a Two Byte Integer length which indicates the number of data bytes, followed by that number of bytes. Thus, the length of Binary Data is limited to the range of 0 to 65,535 Bytes.”_

We keep counting the number of bytes read so we can know when we have read all the properties. We do not need to worry about the order of properties.

_“There is no significance in the order of Properties with different Identifiers.”_

### How Properties may be used to extend the protocol

As stated at the top of this article, properties are part of the ‘extensibility mechanism’ of MQTT 5.0 and the most prominent property for this mechanism is the User Property that can be used in any MQTT Control Packet. User Properties are key-value pairs whose meaning is opaque to the protocol. What is to say, its meaning is defined by the application.

Let’s imagine a use-case for our domain here: a receiver is copying trading signals from three different providers. Each provider is using different brokers. Each broker may assign different symbol names to the same asset, let’s say, gold.

- Broker A uses GOLD
- Broker B uses XAUUSD
- Broker C uses XAUUSD.s

Besides that, each signal provider may use more than one broker. So, the pair **signal\_provider : provider\_broker** may change at any time, even while a trading session is going on. (Yes, we have a quasi-combinatorial explosion here.) The receiver needs to know, ideally in milliseconds, the meaning of the symbol name it is receiving to be able to translate it to the symbol name that their broker is using to properly replicate the trade request.

Without User Properties, as was the case with previous versions of the protocol, that metadata (signal\_provider : provider\_broker) would have to be embedded in the payload, where one would expect to find (and parse) only the required trade signal data.

By contrast, if each signal provider has their own User Property with their broker name, the payload can have only the required signal data.

This is a simplistic example of this use case. But remember that this metadata can be extended to any critical information, including JSON/XML strings and even whole files. So, the possibilities are, in a sense, unlimited.

### Conclusion

In this fourth part of our series we presented a brief description of what properties are in MQTT v5.0, their semantics, and some use cases. We also reported how we are implementing them for the CONNACK, and provided a simple example of how they can be used to extend the protocol. In the next part, we will apply them in the context of the PUBLISH packets, always using a TDD approach to cope with the complexity of the specs.

If you think that you can contribute to the development of this native MQL5 client that will be part of our [Code Base](https://www.mql5.com/en/code), please, drop a note in the comments below or our Chat. Any help is welcome! :)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/13651.zip "Download all attachments in the single ZIP archive")

[MQTT-part4-headers-and-tests.zip](https://www.mql5.com/en/articles/download/13651/mqtt-part4-headers-and-tests.zip "Download MQTT-part4-headers-and-tests.zip")(8.73 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/456696)**
(1)


![Yousuf Mesalm](https://c.mql5.com/avatar/2022/5/6288CAC8-33C2.jpg)

**[Yousuf Mesalm](https://www.mql5.com/en/users/20163440)**
\|
19 Nov 2023 at 20:13

thanks for your great effort

waiting for part 5


![Neural networks made easy (Part 47): Continuous action space](https://c.mql5.com/2/55/Neural_Networks_Part_47_avatar.png)[Neural networks made easy (Part 47): Continuous action space](https://www.mql5.com/en/articles/12853)

In this article, we expand the range of tasks of our agent. The training process will include some aspects of money and risk management, which are an integral part of any trading strategy.

![MQL5 Wizard Techniques you should know (Part 07): Dendrograms](https://c.mql5.com/2/59/Dendrograms_Logo.png)[MQL5 Wizard Techniques you should know (Part 07): Dendrograms](https://www.mql5.com/en/articles/13630)

Data classification for purposes of analysis and forecasting is a very diverse arena within machine learning and it features a large number of approaches and methods. This piece looks at one such approach, namely Agglomerative Hierarchical Classification.

![Neural networks made easy (Part 48): Methods for reducing overestimation of Q-function values](https://c.mql5.com/2/56/NN_part_48_avatar.png)[Neural networks made easy (Part 48): Methods for reducing overestimation of Q-function values](https://www.mql5.com/en/articles/12892)

In the previous article, we introduced the DDPG method, which allows training models in a continuous action space. However, like other Q-learning methods, DDPG is prone to overestimating Q-function values. This problem often results in training an agent with a suboptimal strategy. In this article, we will look at some approaches to overcome the mentioned issue.

![Neural networks made easy (Part 46): Goal-conditioned reinforcement learning (GCRL)](https://c.mql5.com/2/55/Neural_Networks_Part_46_avatar.png)[Neural networks made easy (Part 46): Goal-conditioned reinforcement learning (GCRL)](https://www.mql5.com/en/articles/12816)

In this article, we will have a look at yet another reinforcement learning approach. It is called goal-conditioned reinforcement learning (GCRL). In this approach, an agent is trained to achieve different goals in specific scenarios.

[![](https://www.mql5.com/ff/si/dwquj7nmuxsb297n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F994%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dhow.use.vps%26utm_content%3Drent.vps%26utm_campaign%3D0622.MQL5.com.Internal&a=enhudadyvnrfwcvutcjazdvrxjyrzhyf&s=8f8a773cbff7e7ca26346dfb885f4f329a8b1f2c99472f858f32c0b06b662998&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=uqsfaugvukjhfbmhbwsauekizhckftta&ssn=1769178137583473628&ssn_dr=0&ssn_sr=0&fv_date=1769178137&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F13651&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Developing%20an%20MQTT%20client%20for%20Metatrader%205%3A%20a%20TDD%20approach%20%E2%80%94%20Part%204%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176917813716511740&fz_uniq=5068149091386455576&sv=2552)

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