---
title: Developing an MQTT client for MetaTrader 5: a TDD approach — Part 3
url: https://www.mql5.com/en/articles/13388
categories: Integration
relevance_score: 6
scraped_at: 2026-01-23T17:22:27.634496
---

[![](https://www.mql5.com/ff/sh/7h2yc16rtqsn2m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
MQL5 Channels - Market analysis\\
\\
Dozens of channels, thousands of subscribers and daily updates. Learn more about trading.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=glufvbpblsoxonicqfngsyuzwfebnilr&s=103cc3ab372a16872ca1698fc86368ffe3b3eaa21b59b4006d5c6c10f48ad545&uid=&ref=https://www.mql5.com/en/articles/13388&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5068151333359384098)

MetaTrader 5 / Integration


> > > > > > _“How can you consider yourself to be a professional if you do not know that all your code works? How can you know all your code works if you don’t test it every time you make a change? How can you test it every time you make a change if you don’t have automated unit tests with very high coverage? How can you get automated unit tests with very high coverage without practicing TDD?”_ (Robert 'Uncle Bob' Martin, The Clean Coder, 2011)

### Introduction

Until now, in the [Part 1](https://www.mql5.com/en/articles/12857) and in the Part 2 of this series, we have been dealing with a small fraction of the **non-operational** section of the MQTT protocol. We organized in two separate header files, all of the protocol definitions, enumerations, and some common functions that will be shared among our classes. Also, we wrote an interface to stand by as the root of the object hierarchy and implemented it in one class with the single purpose of building a conformant MQTT CONNECT packet. Meanwhile, we have been writing unit tests for each function involved in building the packets. Although we sent our generated packet to our local MQTT broker to check if it would be recognized as a well-formed MQTT packet, technically this step was not required. Since we were using fixture data to feed our function parameters, we knew that we were testing them in isolation, we knew that we were testing them in a state-independent way. That is good and we will strive to keep writing our tests – and consequently, our functions – that way. This will make our code more flexible, allowing us to change a function implementation without even changing our test code, as long as we have the same function signature.

From now on we will be dealing with the operational part of the MQTT protocol. Not surprisingly, it is named [**Operational Behavior** in the OASIS Standard](https://www.mql5.com/go?link=https://docs.oasis-open.org/mqtt/mqtt/v5.0/os/mqtt-v5.0-os.html%23_Toc3901229 "https://docs.oasis-open.org/mqtt/mqtt/v5.0/os/mqtt-v5.0-os.html#_Toc3901229"). What is to say, from now on we need to deal with the packets sent from the server. Our client must be able to identify the server packet type, and its semantics, and must be able to choose the appropriate behavior in a given context, the appropriate behavior in each possible client state.

To be able to cope with this task, we must identify the server packet type in the first byte of the response. If it is a CONNACK packet, we must read its Connect Reason Code, and react accordingly.

### (CONNECT) Setting the client Connect Flags

When our client requests a connection with the server, it must inform the server about

- some desired capabilities of the broker,
- if it will need authentication with a username and password,
- and if this connection is meant to be a new Session or it is resuming an earlier opened Session.

This is done by setting some bit flags at the start of the Variable Header, right after the Protocol Name and Protocol Version. These bit flags on the CONNECT packet are named Connect Flags.

Remember that bit flags are [boolean values](https://www.mql5.com/en/docs/basis/types/integer/boolconst). They may be given different names or representations, but boolean values have only two possible values, usually true or false.

![Fig 01 - Common terms used to represent boolean values](https://c.mql5.com/2/58/Fig_01_-_Common_terms_used_to_represent_boolean_values.PNG)

Fig 01 - Common terms used to represent boolean values

The OASIS Standard uses 1 (one) and 0 (zero) consistently. We will be using true and false here most of the time, and eventually, we will use set and unset. This should make the text more readable. Moreover, our public API is using true and false consistently for setting these values, thus using these terms should make this article more easy to understand for those readers who are following the development of the library.

![Fig 02 - OASIS Connect Flag bits](https://c.mql5.com/2/58/Fig_02_-_OASIS_Connect_Flag_bits.PNG)

Fig 02 - OASIS Connect Flag bits

As you can see in the OASIS table in the above image, the first bit (bit\_0) is reserved and we must leave it alone: zeroed, unchecked, boolean false, unset. If we set it, we will have a Malformed Packet.

Clean Start (bit\_1)

The first bit we can set is the second bit. It is used for setting the Clean Start flag – if true, the server will do a Clean Start and discard any existing Session associated with our Client Identifier. The server will start a new Session. If not set, the server will resume our previous conversation, if any, or start a new Session if there is no existing Session associated with our Client Identifier.

This is what our function for setting/unsetting this flag looks like now.

```
void CPktConnect::SetCleanStart(const bool cleanStart)
  {
   cleanStart ? m_connect_flags |= CLEAN_START : m_connect_flags &= ~CLEAN_START;
   ArrayFill(ByteArray, ArraySize(ByteArray) - 1, 1, m_connect_flags);
  }
```

We toggle the values by means of [bitwise operations](https://www.mql5.com/en/docs/basis/operations/bit). We are using a [ternary operator](https://www.mql5.com/en/docs/basis/operators/ternary) to toggle the boolean and [compound assignments](https://www.mql5.com/en/docs/basis/operations/assign) to make the code more compact. Then we store the result in the m\_connect\_flags private class member. Finally, we update the byte array that represents our CONNECT packet with the new values by calling the built-in function ArrayFill. (Note that this late step using – array filling – is an implementation detail that we will, probably, change later.)

This line from one of our tests shows how it is called.

```
   CPktConnect *cut = new CPktConnect(buf);
//--- Act
   cut.SetCleanStart(true);
```

| Bit | 7 | 6 | 5 | 4 | 3 | 2 | 1 | 0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | User Name Flag | Password Flag | Will Retain | Will  QoS 2 | Will QoS 1 | Will Flag | Clean Start | Reserved |
|  | X | X | X | X | X | X | 1 | 0 |

Table 01 - bit flag Clean Start (bit\_1) set to true – MQTT v5.0

We’ll be using this pattern extensively to toggle boolean flags: ternary operator and bitwise operations with compound assignments.

The following three flags named Will ‘Something’ are meant to express our desire that the server has some capabilities. They inform the server that we are ‘willing’ the server to be able to

1. store the Will Message(s) and associate them with our client Session (more on this later);
2. provide a specific QoS Level, usually above QoS 0, which is the default if nothing is set;
3. retain Will Message(s) and publish them as ‘retained’ (see below) if the Will Message is set to true

Will Flag (bit\_2)

The third bit is used for setting the Will Flag – if set to true, our client must provide a Will Message to be published “in cases where the Network Connection is not close normally”. One can think of Will Message as a kind of the client's ‘last words’ after dying in the face of the subscribers.

```
void CPktConnect::SetWillFlag(const bool willFlag)
  {
   willFlag ? m_connect_flags |= WILL_FLAG : m_connect_flags &= ~WILL_FLAG;
   ArrayFill(ByteArray, ArraySize(ByteArray) - 1, 1, m_connect_flags);
  }
```

It is called in the same way as the previous function.

```
//--- Act
   CPktConnect *cut = new CPktConnect(buf);
   cut.SetWillFlag(true);
```

| Bit | 7 | 6 | 5 | 4 | 3 | 2 | 1 | 0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | User Name Flag | Password Flag | Will Retain | Will  QoS 2 | Will QoS 1 | Will Flag | Clean Start | Reserved |
|  | X | X | X | X | X | 1 | X | 0 |

Table 02 - bit flag Will Flag (bit\_2) set to true – MQTT v5.0

Will QoS (bit\_3, bit\_4)

Differently from the two previous flags, this feature requires two bits to be set if the client is requesting the QoS Level 2, the fourth and fifth bits. QoS stands for Quality of Service and can be one of three.

![Fig 03 - OASIS QoS Definitions](https://c.mql5.com/2/58/Fig_03_-_OASIS_QoS_Definitions.PNG)

Fig 03 - OASIS QoS Definitions

From the least reliable to the most reliable delivery system, they are:

**QoS 0**

The QoS 0 sets at most once delivery. It is a kind of “fire and forget”. The sender will try once. The message may be lost. There is no acknowledgment from the server. It is the default, meaning, if nothing is set on bits 3 and 4, then the QoS Level being requested by the client is QoS 0.

**QoS 1**

The QoS 1 sets at least once delivery. It has a PUBACK confirming the delivery.

Same function definition pattern.

```
void CPktConnect::SetWillQoS_1(const bool willQoS_1)
  {
   willQoS_1 ? m_connect_flags |= WILL_QOS_1 : m_connect_flags &= ~WILL_QOS_1;
   ArrayFill(ByteArray, ArraySize(ByteArray) - 1, 1, m_connect_flags);
  }
```

Same function call pattern.

```
//--- Act
   CPktConnect *cut = new CPktConnect(buf);
   cut.SetWillQoS_1(true);
```

| Bit | 7 | 6 | 5 | 4 | 3 | 2 | 1 | 0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | User Name Flag | Password Flag | Will Retain | Will  QoS 2 | Will QoS 1 | Will Flag | Clean Start | Reserved |
|  | X | X | X | X | 1 | X | X | 0 |

Table 03 - bit flag Will QoS 1 (bit\_3) set to true – MQTT v5.0

**QoS 2**

The QoS 2 set exactly once delivery. This QoS requires that there is no loss or duplication. The sender will acknowledge the message with a PUBREC and the delivery with a PUBREL.

One can think of this level as sending a registered parcel. The postal system gives you a receipt when you transfer the packet to their hands acknowledging that, for now on, they are responsible for delivering it to the right address. And when it happens, when they deliver the parcel, they send you a signed receipt from the recipient, acknowledging the parcel delivery.

Idem.

```
void CPktConnect::SetWillQoS_2(const bool willQoS_2)
  {
   willQoS_2 ? m_connect_flags |= WILL_QOS_2 : m_connect_flags &= ~WILL_QOS_2;
   ArrayFill(ByteArray, ArraySize(ByteArray) - 1, 1, m_connect_flags);
  }
```

Ibidem.

```
//--- Act
   CPktConnect *cut = new CPktConnect(buf);
   cut.SetWillQoS_2(true);
```

| Bit | 7 | 6 | 5 | 4 | 3 | 2 | 1 | 0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | User Name Flag | Password Flag | Will Retain | Will  QoS 2 | Will QoS 1 | Will Flag | Clean Start | Reserved |
|  | X | X | X | 1 | X | X | X | 0 |

Table 04 - bit flag Will QoS 2 (bit\_4) set to true – MQTT v5.0

The server will tell us about its accepted Maximum QoS level in the CONNACK Reason Codes and in the CONNACK Properties. The client can request, but the server capabilities are mandatory. If we receive a CONNACK with a Maximum QoS, we must abide by this server limitation and not send a PUBLISH with a greater QoS. Otherwise, the server will DISCONNECT.

QoS 2 is the highest QoS level available on MQTT v5.0 and there is a significant overhead associated with it, since the delivery protocol is symmetric, meaning that any of the sides (the server and the client) can act both as sender or receiver for this matter.

_**NOTE**_: We can say that the QoS is the core of the protocol, from a user viewpoint. It defines the application profile and affects dozens of other aspects of the protocol. So, we will deep dive into the QoS Level and its settings in the context of the implementation of the PUBLISH packet.

It is worth noting that QoS 1 and QoS 2 are optional for client implementations. As OASIS says in a non-normative comment:

“A Client does not need to support QoS 1 or QoS 2 PUBLISH packets. If this is the case, the Client simply restricts the maximum QoS field in any SUBSCRIBE commands it sends to a value it can support.”

Will RETAIN (bit\_5)

In the sixth byte, we set the Will Retain flag. This flag is tied with the above Will Flag,

- If the Will Flag is unset, then Will Retain must be unset too.
- If the Will Flag is set and the Will Retain is unset, the server will publish the Will Message as a non-retained message.
- If both are set, the server will publish the Will Message as a retained message.

Code omitted for brevity in this and following two remaining flags since the function definition and the function call patterns are rigorously the same as the previous flags. Please, see the attached files for details and testing.

| Bit | 7 | 6 | 5 | 4 | 3 | 2 | 1 | 0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | User Name Flag | Password Flag | Will Retain | Will  QoS 2 | Will QoS 1 | Will Flag | Clean Start | Reserved |
|  | X | X | 1 | X | X | X | X | 0 |

Table 05 - bit flag Will Retain (bit\_5) set to true – MQTT v5.0

We must wait for the CONNACK packet to check this flag before starting to send PUBLISH packets. If the server receives a PUBLISH packet with Will Retain set to 1 and it doesn’t support retained messages, the server will DISCONNECT. You might be thinking: but wait? Is it possible to start publishing even before receiving a CONNACK packet? Yes, it is possible. The Standard allows this behavior. But they also make a remark about it:

“Clients that send MQTT Control Packets before they receive CONNACK will be unaware of the Server constraints”

Thus, we must check this flag on CONNACK packets before sending any PUBLISH packets with Will Retain set to 1 (one).

Password flag (bit\_6)

In the seventh bit, we inform the server if we will be sending a Password in the Payload or not. If this flag is set, a password field must be present in the Payload. If it is unset, a password field must not be present in the Payload.

“This version of the protocol allows the sending of a Password with no User Name, where MQTT v3.1.1 did not. This reflects the common use of Password for credentials other than a password.” (OASIS Standard, 3.1.2.9)

| Bit | 7 | 6 | 5 | 4 | 3 | 2 | 1 | 0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | User Name Flag | Password Flag | Will Retain | Will  QoS 2 | Will QoS 1 | Will Flag | Clean Start | Reserved |
|  | X | 1 | X | X | X | X | X | 0 |

Table 06 - bit flag Password Flag (bit\_6) set to true – MQTT v5.0

User Name flag (bit\_7)

And, finally, on the eight-bit, we inform the server if we will be sending a User Name in the Payload, or not. As is the case with the Password Flag above, if this flag is set a User Name field must be present in the Payload. Otherwise, a User Name field must not be present in the Payload.

| Bit | 7 | 6 | 5 | 4 | 3 | 2 | 1 | 0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | User Name Flag | Password Flag | Will Retain | Will  QoS 2 | Will QoS 1 | Will Flag | Clean Start | Reserved |
|  | 1 | X | X | X | X | X | X | 0 |

Table 07 - bit flag User Name (bit\_7) set to true – MQTT v5.0

So, a Connect Flags byte with the following bit sequence…

| Bit | 7 | 6 | 5 | 4 | 3 | 2 | 1 | 0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | User Name Flag | Password Flag | Will Retain | Will  QoS 2 | Will QoS 1 | Will Flag | Clean Start | Reserved |
|  | X | X | 1 | 1 | X | 1 | 1 | 0 |

Table 08 - bit flags set for Clean Start, Will Flag, Will QoS2 and Will Retain – MQTT v5.0

... could be translated as something like: Open a connection for a new Session, with QoS Level 2, and be prepared to store my Will Message and publish it as a retained message. And, by the way, Mr. Server, I’ll not need to authenticate with a username and password.

The server will kindly answer if it can fulfill our will. It may be able to fulfill them completely, partially or none at all. And the server will send its answer in the form of Connect Reason Codes in CONNACK packets.

### (CONNACK) Getting the Reason Codes related to Connect Flags

There are forty-four Reason Codes in MQTT v5.0. We have gathered them in our Defines.mqh header. The CONNACK (and other packet types) has a single Reason Code as part of the Variable Header. They are named Connect Reason Codes.

| Value | Hex | Reason Code name | Description |
| --- | --- | --- | --- |
| 0 | 0x00 | Success | The Connection is accepted. |
| 128 | 0x80 | Unspecified error | The Server does not wish to reveal the reason for the failure, or none of the other Reason Codes apply. |
| 129 | 0x81 | Malformed Packet | Data within the CONNECT packet could not be correctly parsed. |
| 130 | 0x82 | Protocol Error | Data in the CONNECT packet does not conform to this specification. |
| 131 | 0x83 | Implementation specific error | The CONNECT is valid but is not accepted by this Server. |
| 132 | 0x84 | Unsupported Protocol Version | The Server does not support the version of the MQTT protocol requested by the Client. |
| 133 | 0x85 | Client Identifier not valid | The Client Identifier is a valid string but is not allowed by the Server. |
| 134 | 0x86 | Bad User Name or Password | The Server does not accept the User Name or Password specified by the Client |
| 135 | 0x87 | Not authorized | The Client is not authorized to connect. |
| 136 | 0x88 | Server unavailable | The MQTT Server is not available. |
| 137 | 0x89 | Server busy | The Server is busy. Try again later. |
| 138 | 0x8A | Banned | This Client has been banned by administrative action. Contact the server administrator. |
| 140 | 0x8C | Bad authentication method | The authentication method is not supported or does not match the authentication method currently in use. |
| 144 | 0x90 | Topic Name invalid | The Will Topic Name is not malformed, but is not accepted by this Server. |
| 149 | 0x95 | Packet too large | The CONNECT packet exceeded the maximum permissible size. |
| 151 | 0x97 | Quota exceeded | An implementation or administrative imposed limit has been exceeded. |
| 153 | 0x99 | Payload format invalid | The Will Payload does not match the specified Payload Format Indicator. |
| 154 | 0x9A | Retain not supported | The Server does not support retained messages, and Will Retain was set to 1. |
| 155 | 0x9B | QoS not supported | The Server does not support the QoS set in Will QoS. |
| 156 | 0x9C | Use another server | The Client should temporarily use another server. |
| 157 | 0x9D | Server moved | The Client should permanently use another server. |
| 159 | 0x9F | Connection rate exceeded | The connection rate limit has been exceeded. |

Table 08 - Connect Reason Code values

The Standard is explicit about the server being required to send Connect Reason Codes on the CONNACK:

"The Server sending the CONNACK packet MUST use one of the Connect Reason Code values \[MQTT-3.2.2-8\]."

The Connect Reason Codes are of special interest to us at this point. Because we need to check them before going forward with the communication. They will inform us about some server capabilities and limitations, like the QoS Level available and the availability of retained messages. Also, as you can see in their names and descriptions in the table above, they will inform us if our CONNECT attempt was successful or not.

To get the Reason Codes, first, we need to identify the packet type, because we are interested only in CONNACK packets.

We will take advantage of the fact that we need a very simple function to get the packet type, to describe how we are using Test-Driven Development, do a bit of reasoning about this technique, and provide a couple of short examples. You can get all the details in the attached files.

### (CONNACK) Identifying the server packet type

We know for sure that the first byte of any MQTT Control Packet encodes the packet’s type. So we will just read this first byte as soon as possible and we have the server packet type.

```
uchar pkt_type = server_response_buffer[0];
```

Stop. Done. Next problem. Right?

Well, it is not wrong, for sure. The code is clear, the variables are well-named, and it should be performant and lightweight.

But wait! How the code that will use our library is expected to call this statement? The packet type will be returned by a public function call? Or this information can be hidden as an implementation detail behind a private member? If it is returned by a function call, where this function should be hosted? On the CPktConnect class? Or it should be hosted in any of our header files since it will used by many different classes? If it is stored in a private member, in which class it should live?

(TMTOWTDI\*) “There is more than one way to do it” is an acronym that became very popular. TDD is another acronym that became very popular, for different reasons. As it used to be, both acronyms were abused, and hyped, and some of them even became “fashion”, as is the case of TDD:

\_\_ “I’m tddying, mom! It’s cool.”

But the groups who coined them did it after years of hard work dealing with the same basic question: how to write more performant, idiomatic, and robust code while improving developer productivity? How to keep developers focused on what must be done, instead of wandering around what may be done? How to keep each of them focused on one task  - and only one task – at a time? How to be sure what they are doing will not introduce regression bugs and break the system?

In a word, these acronyms, the ideas that they carry, and the techniques they recommend, consolidate years of practice of hundreds of very different individuals with expertise in software development. TDD is not a theory, it is a practice. We can say that TDD is a technique of problem-solving by closing the scope by breaking down the problem into its constituents. We must define the single task that will move us one step ahead. Just one single step. Frequently a baby-step.

So, what is our problem now? We need to identify if a server response is a CONNACK packet. Simple as that. Because, according to the specs, we need to read the CONNACK Response Codes to decide what to do next. I mean, identifying the type of packet we received from the server as a response is required for us to advance from the connecting state to the publishing state.

How can we identify if a server response is a CONNACK packet? Well, it is easy. It has a specific type, coded in our MQTT.mqh header as an Enumeration, namely, ENUM\_PKT\_TYPE.

```
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
...
```

So, maybe we can start with a function that when passed a network byte array coming from an MQTT broker returns the type of the packet.

That sounds good. Let's write a test for this function.

```
bool TEST_GetPktType_FAIL()
  {
   Print(__FUNCTION__);
//--- Arrange
   uchar expected[] = {(uchar)CONNACK};
   uchar result[1] = {};
   uchar wrong_first_byte[] = {'X'};
//--- Act
   CSrvResponse *cut = new CSrvResponse();
   ENUM_PKT_TYPE pkt_type = cut.GetPktType(wrong_first_byte);
   ArrayFill(result,0,1,(uchar)pkt_type);
//--- Assert
   bool isTrue = AssertNotEqual(expected, result);
//--- cleanup
   delete cut;
   ZeroMemory(result);
   return  isTrue ? true : false;
  }
```

Let’s try to understand the test function since we will be using this pattern in all our tests and we will not be detailing all of them here for the sake of brevity.

In the comment lines we are describing each step of the pattern:

Arrange

First, we initialize an array with one single element that is the byte value we expect to be returned by our function.

Second, we initialize an empty char buffer with a size of one to receive the result of our function call.

We initialize our fixture to be passed to the function we are testing. It stands for the first single byte to be read from the server response fixed header in order to identify the packet type. In this case, it has a wrong\_first\_byte and we are making it explicit by naming the variable accordingly.

Act

We instantiate the Class Under Test (cut) and call our function.

Assert

We assert the inequality of the expected and the result arrays, both in content and size, using the MQL5 ArrayCompare function (see the attached test file).

Clean-Up

Finally, we clean up resources by deleting de ‘cut’ instance and by passing our ‘result’ buffer to ZeroMemory. This should avoid memory leaks and test contamination.

![Fig. 04 - TEST_CSrvResponse - FAIL - undeclared identifier](https://c.mql5.com/2/58/Fig._04_-_TEST_CSrvResponse_-_FAIL_-_undeclared_identifier.PNG)

Fig. 04 - TEST\_CSrvResponse - FAIL - undeclared identifier

It fails on compilation because the function does not exist yet. We need to write it. But where should it be hosted?

We already know that we will need to identify the response packet type all the time. Whenever we send a packet to our broker it will send us one of these “response thing”. And this “thing” is a kind of MQTT Control Packet. So, since it is a kind of some “thing”, it seems natural that it should have its own class under the group of that similar “things”. Let’s say a class to represent all server responses under the group of control packets.

Let’s say, a CSrvResponse class that implements the IControlPacket interface.

We could be tempted to make it another function in our already existent CPktConnect class. But we would be violating an important principle of Object-Oriented Programming: the Single Responsibility Principle (SRP).

“you should separate those things that change for different reasons, and group together those things that change for the same reasons.” (R. Martin, The Clean Coder, 2011).

For one side, our CPktConnect class will change whenever we change the way we build CONNECT packets, and for the other side, our (non-existent) CSrvResponse class will change whenever we change the way we read our CONNACK, PUBACK, SUBACK, and other server responses. So they have very clear different responsibilities and it is pretty easy to see that in this case. But sometimes it may be tricky to decide if an entity of the domain we are modeling should be declared in a proper class. By applying the SRP you have an objective guideline to decide about these “things”.

So let's write it, just enough to pass the test.

```
ENUM_PKT_TYPE CSrvResponse::GetPktType(uchar &resp_buf[])
  {
   return (ENUM_PKT_TYPE)resp_buf[0];
  }
```

Test compiles, but fail as expected, since we passed a “wrong” server response in order to have it fail.

![Fig. 05 - TEST_CSrvResponse - FAIL - wrong packet](https://c.mql5.com/2/58/Fig._05_-_TEST_CSrvResponse_-_FAIL_-_wrong_packet.PNG)

Fig. 05 - TEST\_CSrvResponse - FAIL - wrong packet

Let pass the “right” CONNACK packet type as a server response. Note that, again, we are explicitly naming the fixture: right\_first\_byte. The name per se is just a label. What matters is that its meaning should be clear for anyone reading our code. Including ourselves six months or six years later.

```
bool TEST_GetPktType()
  {
   Print(__FUNCTION__);
//--- Arrange
   uchar expected[] = {(uchar)CONNACK};
   uchar result[1] = {};
   uchar right_first_byte[] = {2};
//--- Act
   CSrvResponse *cut = new CSrvResponse();
   ENUM_PKT_TYPE pkt_type = cut.GetPktType(right_first_byte);
   ArrayFill(result,0,1,(uchar)pkt_type);
//--- Assert
   bool isTrue = AssertEqual(expected, result);
//--- cleanup
   delete cut;
   ZeroMemory(result);
   return  isTrue ? true : false;
  }
```

![Fig. 06 - TEST_CSrvResponse - PASS](https://c.mql5.com/2/58/Fig._06_-_TEST_CSrvResponse_-_PASS.PNG)

Fig. 06 - TEST\_CSrvResponse - PASS

Ok. Now the test is passing and we know that, at least for these two arguments, one wrong, and one right, it is failing or passing, respectively. Later we may test it more extensively, if needed.

In these simple steps are embedded the three basic “laws” of TDD as summarized by R. Martin.

1. You are not allowed to write any production code until you have first written a failing unit test.
2. You are not allowed to write more of a unit test than is sufficient to fail, and not compiling is failing.
3. You are not allowed to write more production code than is sufficient to pass the currently failing unit test.

Ok. Enough about TDD for now. Let’s go back to our task at hand and read those Connect Reason Codes on CONNACK packets arriving from the server.

### (Connect Reason Codes) What to do with unavailable capabilities on the server?

There are two of the Connect Reason Codes that deserve our attention at this point.

1. QoS not supported
2. Retain not supported

They are somewhat special because they do not indicate an error, instead they indicate a server limitation. If our CONNACK has any of these Connect Reason Codes, the server is saying that our network connection was successful, that our CONNECT was a well-formed packet, and that the server is online and working, but it is not able to fulfill our requirements. We need to take action. We need to choose what to do next.

What should we do if we send a CONNECT with Will QoS 2 and the server replies with QoS Maximum 1? Should we re-send the CONNECT with the downgraded QoS flag? Or should we DISCONNECT before downgrading? If that was the case with the RETAIN feature, can we just ignore it as irrelevant and start publishing anyway? Or should we re-send the CONNECT with downgraded flags before publishing?

What should we do after we receive a successful CONNACK, meaning the server accepted our connection and has all the capabilities we requested? Must we immediately start sending PUBLISH packets? Or we can leave the connection open by sending successive PINGREQ packets until we are ready to publish a message? By the way, must we SUBSCRIBE to a topic before publishing?

Most of these questions are answered by the Standard. They are required to implement AS-IS in order to have an MQTT v5.0 conformant client. Many choices are left to the application developers. For now, we will be dealing only with what is required, so we can have a conformant client as soon as possible.

According to the Standard, the client can only request a QoS Level > 0, if the Will Flag is also set to 1, meaning we are allowed to request a QoS Level > 0 only if we are also sending a Will Message in the CONNECT packet. But we do not want, or better, we do not need to deal with Will Message(s) right now. So, our decision is a compromise between understanding only what we need to know now and trying to understand all the intricacies of the Standard, eventually writing code that may be of no use later.

We only need to know what our client will do if the QoS Level requested or Retain is not available on the server. And we need to know this as soon as a new CONNACK arrives. So we are checking for it in the CSrvResponse’s constructor. If the response is a CONNACK, the constructor calls the protected method GetConnectReasonCode.

```
CSrvResponse::CSrvResponse(uchar &resp_buf[])
  {
   if(GetPktType(resp_buf) == CONNACK
      && GetConnectReasonCode(resp_buf)
      == (MQTT_REASON_CODE_QOS_NOT_SUPPORTED || MQTT_REASON_CODE_RETAIN_NOT_SUPPORTED))
     {
      CSrvProfile *serverProfile = new CSrvProfile();
      serverProfile.Update("000.000.00.00", resp_buf);
     }
  }
```

If the Connect Reason Code is one of MQTT\_REASON\_CODE\_QOS\_NOT\_SUPPORTED or MQTT\_REASON\_CODE\_RETAIN\_NOT\_SUPPORTED, it will store this information on the Server Profile. For now, we will only store this information about the server and wait for a DISCONNECT. Later we will use it when requesting a new connection on this same server. Note that this ‘later’ may be a few milliseconds after the first connection attempt. Or it may be weeks late. The point is that we will have this information stored on the Server Profile.

### How we are testing protected methods

To test protected methods we have created a class on our test script, a class derived from our Class Under Test, the CSrvResponse in this case. Then we call the protected methods on CSrvResponse through this ‘for testing purposes only’ derived class that we named TestProtectedMethods.

```
class TestProtectedMethods: public CSrvResponse
  {
public:
                     TestProtectedMethods() {};
                    ~TestProtectedMethods() {};
   bool              TEST_GetConnectReasonCode_FAIL();
   bool              TEST_GetConnectReasonCode();
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool TestProtectedMethods::TEST_GetConnectReasonCode_FAIL()
  {
   Print(__FUNCTION__);
//--- Arrange
   uchar expected = MQTT_REASON_CODE_SUCCESS;
   uchar reason_code_banned[4];
   reason_code_banned[0] = B'00100000'; // packet type
   reason_code_banned[1] = 2; // remaining length
   reason_code_banned[2] = 0; // connect acknowledge flags
   reason_code_banned[3] = MQTT_REASON_CODE_BANNED;
//--- Act
   CSrvResponse *cut = new CSrvResponse();
   uchar result = this.GetConnectReasonCode(reason_code_banned);
//--- Assert
   bool isTrue = AssertNotEqual(expected, result);
//--- cleanup
   delete cut;
   ZeroMemory(result);
   return  isTrue ? true : false;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool TestProtectedMethods::TEST_GetConnectReasonCode()
  {
   Print(__FUNCTION__);
//--- Arrange
   uchar expected = MQTT_REASON_CODE_SUCCESS;
   uchar reason_code_success[4];
   reason_code_success[0] = B'00100000'; // packet type
   reason_code_success[1] = 2; // remaining length
   reason_code_success[2] = 0; // connect acknowledge flags
   reason_code_success[3] = MQTT_REASON_CODE_SUCCESS;
//--- Act
   CSrvResponse *cut = new CSrvResponse();
   uchar result = this.GetConnectReasonCode(reason_code_success);
//--- Assert
   bool isTrue = AssertEqual(expected, result);
//--- cleanup
   delete cut;
   ZeroMemory(result);
   return  isTrue ? true : false;
  }
```

Note that we are not storing anything on the Server Profile. In fact, the Server Profile doesn’t even exist yet. We are just printing a message saying that the Server Profile is being updated. That is because the Server Profile is to be persisted between Sessions of our client, and we are not dealing with persistence yet. Later, when implementing persistence, we may change this stub function to persist the server profile in an SQLite database, for example, without even having to remove the printed (or logged) message. It is just the case that it is not being implemented right now. As said above, at this point we only have to know what to do if the server does not match our requested capabilities: we store the information to be reused later.

### Conclusion

In this article we described how we are starting to deal with the [Operational Behavior section of the MQTT v5.0 protocol](https://www.mql5.com/go?link=https://docs.oasis-open.org/mqtt/mqtt/v5.0/os/mqtt-v5.0-os.html%23_Toc3901229 "https://docs.oasis-open.org/mqtt/mqtt/v5.0/os/mqtt-v5.0-os.html#_Toc3901229"), as required by the OASIS Standard in order to have a conformant client working as soon as possible. We described how we are implementing the CSrvResponse class to identify the server response type and its associated Reason Codes. We also described how our client will react to unavailable server capabilities.

In the next step, we will implement PUBLISH, get a better understanding of the Operational Behavior for the QoS Level(s), and deal with Session(s), and its near-required persistence.

\\*\\* Other useful acronyms: DRY, KISS, YAGNI. Each of them embeds some practical wisdom but YMMV :)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/13388.zip "Download all attachments in the single ZIP archive")

[headers.zip](https://www.mql5.com/en/articles/download/13388/headers.zip "Download headers.zip")(7.16 KB)

[tests.zip](https://www.mql5.com/en/articles/download/13388/tests.zip "Download tests.zip")(3 KB)

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

**[Go to discussion](https://www.mql5.com/en/forum/454900)**

![Neural networks made easy (Part 38): Self-Supervised Exploration via Disagreement](https://c.mql5.com/2/54/self_supervised_exploration_via_disagreement_038_avatar.png)[Neural networks made easy (Part 38): Self-Supervised Exploration via Disagreement](https://www.mql5.com/en/articles/12508)

One of the key problems within reinforcement learning is environmental exploration. Previously, we have already seen the research method based on Intrinsic Curiosity. Today I propose to look at another algorithm: Exploration via Disagreement.

![Category Theory in MQL5 (Part 21): Natural Transformations with LDA](https://c.mql5.com/2/58/Category-Theory-p21-avatar.png)[Category Theory in MQL5 (Part 21): Natural Transformations with LDA](https://www.mql5.com/en/articles/13390)

This article, the 21st in our series, continues with a look at Natural Transformations and how they can be implemented using linear discriminant analysis. We present applications of this in a signal class format, like in the previous article.

![Developing a Replay System — Market simulation (Part 07): First improvements (II)](https://c.mql5.com/2/54/replay-p7-avatar.png)[Developing a Replay System — Market simulation (Part 07): First improvements (II)](https://www.mql5.com/en/articles/10784)

In the previous article, we made some fixes and added tests to our replication system to ensure the best possible stability. We also started creating and using a configuration file for this system.

![Estimate future performance with confidence intervals](https://c.mql5.com/2/58/estimate_future_performance_acavatar.png)[Estimate future performance with confidence intervals](https://www.mql5.com/en/articles/13426)

In this article we delve into the application of boostrapping techniques as a means to estimate the future performance of an automated strategy.

[![](https://www.mql5.com/ff/sh/592yc11u3j4rs5z9z2/01.png)How AI helps create robots for MetaTrader 5Learn from our book "Neural Networks in Algo Trading with MQL5"Read](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/neurobook%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.neurobook%26utm_content=visit.page%26utm_campaign=neurobook.promo.04.2024&a=ghrobswocqgvhztzjldphupateyllpro&s=9929cb0b8629585b5a42fabc06c525e41f6c0ebdf3045d044a5413b93ea88b47&uid=&ref=https://www.mql5.com/en/articles/13388&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5068151333359384098)

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