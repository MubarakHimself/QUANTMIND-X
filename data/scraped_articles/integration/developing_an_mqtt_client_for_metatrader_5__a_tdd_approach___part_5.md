---
title: Developing an MQTT client for Metatrader 5: a TDD approach — Part 5
url: https://www.mql5.com/en/articles/13998
categories: Integration
relevance_score: 6
scraped_at: 2026-01-23T17:21:58.252337
---

[![](https://www.mql5.com/ff/sh/20jc81m23z78s5z9z2/01.png)![](https://www.mql5.com/ff/sh/20jc81m23z78s5z9z2/02.png)Create your own AI for tradingRead our book "Neural Networks in Algo Trading with MQL5"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/neurobook%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.neurobook%26utm_content=visit.page%26utm_campaign=neurobook.promo.04.2024&a=elbyupbppbqpzzvzhxtydvlupfcbmnmb&s=0d2f8feb92df3772a11aca1f195d2996b59d6539e283cdf4a18ccff02e5ad43d&uid=&ref=https://www.mql5.com/en/articles/13998&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5068143795691779588)

MetaTrader 5 / Integration


“ _Premature optimization is the root of all evil_.” (Donald Knuth)

### Introduction

MQTT is a pub/sub message sharing protocol. So we can expect that its core is on PUBLISH and SUBSCRIBE packets. All other packet types exist to get on them.

Besides being able to write PUBLISH packets, we must also be able to read them, since the messages our Client will receive from other Clients are PUBLISH packets too. That is because the delivery protocol is symmetric.

"A PUBLISH packet is sent from a Client to a Server or from a Server to a Client to transport an Application Message."

PUBLISH packets have a different fixed header with Publish Flags and a variable header with a required Topic Name encoded as UFT-8 string, and a required Packet Identifier (if QoS > 0). Besides that, it can eventually use almost all the properties and user properties introduced in MQTT 5.0, including those properties related to the Request/Response interaction mode.

In this article, we will see the structure of its headers and how we are testing and implementing the Publish Flags, the Topic Name(s), and the Packet Identifier(s).

In the descriptions that follow, we are using the terms MUST and MAY as they are used by the OASIS Standard, which in turn uses them as described in [IETF RFC 2119](https://www.mql5.com/go?link=https://www.rfc-editor.org/info/rfc2119 "https://www.rfc-editor.org/info/rfc2119").

Also, unless otherwise stated, all quotes are from the [OASIS Standard](https://www.mql5.com/go?link=https://docs.oasis-open.org/mqtt/mqtt/v5.0/mqtt-v5.0.html "https://docs.oasis-open.org/mqtt/mqtt/v5.0/mqtt-v5.0.html").

### Structure of the Fixed Header of an MQTT 5.0 PUBLISH Packet

The PUBLISH packet fixed header follows the same two-byte basic structure of all other control packet types. The first byte is dedicated to carrying the packet type. The second byte is the host of the packet Remaining Length encoded as a Variable Byte Integer.

But while all other packet types have the first four bits of the first byte in RESERVED status, the PUBLISH packet uses these four bits to encode three features: RETAIN, QoS Level, and DUP.

| MQTT Control Packet | Fixed Header flags | Bit 3 | Bit 2 | Bit 1 | Bit 0 |
| --- | --- | --- | --- | --- | --- |
| CONNECT | Reserved | 0 | 0 | 0 | 0 |
| CONNACK | Reserved | 0 | 0 | 0 | 0 |
| PUBLISH | Used in MQTT v5.0 | DUP | QoS 2 | QoS 1 | RETAIN |
| PUBACK | Reserved | 0 | 0 | 0 | 0 |
| PUBREC | Reserved | 0 | 0 | 0 | 0 |
| PUBREL | Reserved | 0 | 0 | 1 | 0 |
| PUBCOMP | Reserved | 0 | 0 | 0 | 0 |
| SUBSCRIBE | Reserved | 0 | 0 | 1 | 0 |
| SUBACK | Reserved | 0 | 0 | 0 | 0 |
| UNSUBSCRIBE | Reserved | 0 | 0 | 1 | 0 |
| UNSUBACK | Reserved | 0 | 0 | 0 | 0 |
| PINGREQ | Reserved | 0 | 0 | 0 | 0 |
| PINGRESP | Reserved | 0 | 0 | 0 | 0 |
| DISCONNECT | Reserved | 0 | 0 | 0 | 0 |
| AUTH | Reserved | 0 | 0 | 0 | 0 |

Table 1 - Reproduction of Table 2-3 Flag Bits From MQTT 5.0 Oasis Standard

"Where a flag bit is marked as “Reserved”, it is reserved for future use and MUST be set to the value listed."

By consequence of this fixed header difference between PUBLISH packets and all other Control Packets, the function we have been using to generate fixed headers cannot be used here.

```
//+------------------------------------------------------------------+
//|                     SetFixedHeader                               |
//+------------------------------------------------------------------+
void SetFixedHeader(ENUM_PKT_TYPE pkt_type, uchar& buf[], uchar& dest_buf[])
  {
   dest_buf[0] = (uchar)pkt_type << 4;
   dest_buf[1] = EncodeVariableByteInteger(buf);
  }
```

As you can see, the function parameters have only the packet type and references to two arrays, one being the source and the other being the destination of the fixed header array. The first line then takes the integer value of the packet type from an Enum and left-shift the integer value by four bits, assigning the result of the bitwise operation to the first byte of the fixed header array (dest\_buf\[0\]). This bitwise operation ensures that the first four bits are left unassigned, or “Reserved”, as the Standard requires.

The second line calls the function that calculates the packet Remaining Length, assigning the value to the second byte of the fixed header array (dest\_buf\[1\]) encoded as a Variable Byte Integer.

But this function doesn’t provide any means for setting the Publish Flags.

![Fig. 1 - MQTT 5.0 PUBLISH Packet Fixed Header RETAIN, QoS Level, and DUP flags](https://c.mql5.com/2/63/MQTT_5.0_PUBLISH_packet_Fixed_Header_RETAINz_QoS_Levelb_and_DUP_flags.PNG)

Fig. 1 - MQTT 5.0 PUBLISH packet Fixed Header RETAIN, QoS Level, and DUP flags

Thus, we added a Switch to accommodate the PUBLISH packets and one last parameter to receive the Publish Flags. We could have overloaded the function to receive the Publish Flags, slightly modifying its body to implement the specificities of the PUBLISH packets. But this is a perfect use case for a Switch since we have only one exception (PUBLISH) and all other cases default to the previous implementation.

The last parameter defaults to zero, meaning it can be ignored when setting all the packet's fixed headers. It will change the dest\_buf only if any Publish Flags are set.

```
//+------------------------------------------------------------------+
//|                     SetFixedHeader                               |
//+------------------------------------------------------------------+
void SetFixedHeader(ENUM_PKT_TYPE pkt_type,
                    uchar& buf[], uchar& dest_buf[], uchar publish_flags = 0)
  {
   switch(pkt_type)
     {
      case PUBLISH:
         dest_buf[0] = (uchar)pkt_type << 4;
         dest_buf[0] |= publish_flags;
         dest_buf[1] = EncodeVariableByteInteger(buf);
         break;
      default:
         dest_buf[0] = (uchar)pkt_type << 4;
         dest_buf[1] = EncodeVariableByteInteger(buf);
         break;
     }
  }
```

As you can see, the destination buffer hosting the fixed header is modified through a OR bitwise operation combined with the assignment to the first byte of it. We have been using this pattern extensively to toggle the Connect Flags and now we are using the same pattern to toggle the Publish Flags.

For example, the RETAIN flag is being set/unset with the following code.

```
//+------------------------------------------------------------------+
//|               CPktPublish::SetRetain                             |
//+------------------------------------------------------------------+
void CPktPublish::SetRetain(const bool retain)
  {
   retain ? m_publish_flags |= RETAIN_FLAG : m_publish_flags &= ~RETAIN_FLAG;
   SetFixedHeader(PUBLISH, m_buf, ByteArray, m_publish_flags);
  }
```

The QoS\_1 level flag (stripped of similar function signature).

```
QoS_1 ? m_publish_flags |= QoS_1_FLAG : m_publish_flags &= ~QoS_1_FLAG;
SetFixedHeader(PUBLISH, m_buf, ByteArray, m_publish_flags);
```

The QoS\_2 level flag.

```
QoS_2 ? m_publish_flags |= QoS_2_FLAG : m_publish_flags &= ~QoS_2_FLAG;
SetFixedHeader(PUBLISH, m_buf, ByteArray, m_publish_flags);
```

The DUP flag.

```
dup ? m_publish_flags |= DUP_FLAG : m_publish_flags &= ~DUP_FLAG;
SetFixedHeader(PUBLISH, m_buf, ByteArray, m_publish_flags);
```

The value of the flags (flag masks) are constants defined in an Enum as power-of-two values according to the position of the respective bit on the byte being toggled.

```
//+------------------------------------------------------------------+
//|             PUBLISH - FIXED HEADER - PUBLISH FLAGS               |
//+------------------------------------------------------------------+
enum ENUM_PUBLISH_FLAGS
  {
   RETAIN_FLAG  	= 0x01,
   QoS_1_FLAG           = 0x02,
   QoS_2_FLAG           = 0x04,
   DUP_FLAG             = 0x08
  };
```

Thus, the flags have the following binary values and position on the byte.

RETAIN

| Decimal 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **1** |

QoS 1

| Decimal 2 | 0 | 0 | 0 | 0 | 0 | 0 | **1** | 0 |

QoS 2

| Decimal 4 | 0 | 0 | 0 | 0 | 0 | **1** | 0 | 0 |

DUP

| Decimal 8 | 0 | 0 | 0 | 0 | **1** | 0 | 0 | 0 |

The decimal value of the PUBLISH packet is 3.

| Decimal 3 | 0 | 0 | 0 | 0 | 0 | 0 | **1** | **1** |

We left-shifted the packet type value by four bits (dest\_buf\[0\] = (uchar)pkt\_type << 4).

|     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Decimal 48 | 0 | 0 | **1** | **1** | 0 | 0 | 0 | 0 |

When we apply the bitwise OR operation ( dest\_buf\[0\] \|= publish\_flags; ) to the binary representation of the packet type value and the flags, we are essentially merging the bits. So the binary representation of the left-shifted PUBLISH packet value with the DUP flag set becomes the following.

|     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Decimal 56 | 0 | 0 | **1** | **1** | **1** | 0 | 0 | 0 |

With RETAIN and QoS 2 flags set the bits of the first byte of the fixed header would look like this.

|     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Decimal 53 | 0 | 0 | **1** | **1** | 0 | **1** | 0 | **1** |

Conversely, the AND bitwise operation between the packet type value and the one’s complement (~) of the flags binary representation do the opposite, unsetting the flag ( m\_publish\_flags &= ~RETAIN\_FLAG ).

So, if the byte was set with QoS 1 without DUP or RETAIN, it would look like this.

|     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Decimal 50 | 0 | 0 | 1 | 1 | 0 | 0 | 1 | 0 |

The one’s complement of the QoS 1 flag above is the value of all its bits flipped.

| QoS\_1 Flag | 0 | 0 | 1 | 0 |
| ~QoS\_1 Flag | 1 | 1 | 0 | 1 |

Since any value AND zero is zero, we are effectively unsetting the flag.

Now, please, note that obviously the binary value of the byte changes as we set the flags. With all flags unset it has the decimal value of 48 after the left-shift of the decimal value of 3 by four bits. When we set the RETAIN flag it has the decimal value of 49. The value becomes 51 with RETAIN and QoS 1. And so on.

Those decimal values are the values we are looking for when exploring all possible combinations of setting/unsetting the flags in our tests.

```
//+------------------------------------------------------------------+
//|              TEST_SetFixedHeader_DUP_QoS2_RETAIN                 |
//+------------------------------------------------------------------+
bool TEST_SetFixedHeader_DUP_QoS2_RETAIN()
  {
   Print(__FUNCTION__);
//--- Arrange
   static uchar expected[] = {61, 0};
   uchar buf[] = {};
//--- Act
   CPktPublish *cut = new CPktPublish(buf);
   cut.SetDup(true);
   cut.SetQoS_2(true);
   cut.SetRetain(true);
   uchar result[];
   ArrayCopy(result, cut.ByteArray);
//--- Assert
   bool isTrue = AssertEqual(expected, result);
//--- cleanup
   delete cut;
   ZeroMemory(result);
   return isTrue;
  }
```

This somewhat naive tests (and others a bit more involved) written before the implementation are leading our development because, besides keeping us focused on the task at hand, they work pretty well as a ‘safety net’ when we need to change or refactor the code. You will find plenty of them in the attached files.

By running the tests you should see something like this.

![Fig 2 - MQTT 5.0 PUBLISH Test Output Fixed Header](https://c.mql5.com/2/63/MQTT_5.0_PUBLISH_Test_Output_Fixed_Header.PNG)

Fig 2 - MQTT 5.0 PUBLISH Test Output Fixed Header

If the publish/subscribe cycle is the core of the protocol, these three features (RETAIN, DUP, and QoS) are the core of the protocol's Operational Behavior. They will arguably have a major impact on the Session state management. So let’s go a bit beyond the strict protocol specification and try to have a sensible understanding of their semantics.

**RETAIN**

As we saw in the first part of this series, the publish/subscribe pattern is tied to a specific Topic Name: a client publishes a message with a Topic Name or subscribe to a Topic Name, and all clients receive messages published under the Topic Name they subscribed for.

When publishing, we can use the RETAIN flag set to 1 (one/true) to instruct the server to store the message and deliver it as a ‘retained message’ to the new subscribers. There is always only one retained message and we set RETAIN to 1 to store/replace existing retained messages. We send a zero-byte payload with this flag set to 1 to clean up retained messages. We unset it to 0 to instruct the server to do nothing with retained messages under this Topic Name, neither storing, replacing, or cleaning up.

When subscribing to a Topic Name, we receive the retained message. On Shared Subscriptions the retained message will be sent to only one of the clients of the shared Topic Filter. We will deep dive into Shared Subscription(s) when dealing with the SUBSCRIBE packets.

This feature works in tandem with the Retain Available and Retain Not Supported flags on CONNACK packets sent from the server.

Retained messages expire as any other message according to the Message Expiry Interval set on PUBLISH or on the Will Properties of the CONNECT payload.

We must take into account that RETAIN is a dynamic broker feature, meaning that it may change from ‘available’ to ‘not supported’ and vice-versa in the same Session.

**QoS Level**

We already talked about QoS Level in the introductory article of this series when enumerating some design choices made by the creators of the protocol.

_"Despite the fact that it was designed to be robust, fast and cheap due to tech stack limitations and expensive network costs, it was required to provide quality of service data delivery with continuous session awareness, which allows to cope with unreliable or even intermittent internet connections."_

And in the context of Connect Flags, we saw the table below with the definition of each QoS Level.

| QoS Value | Bit 2 | Bit 1 | Description |
| --- | --- | --- | --- |
| 0 | 0 | 0 | At most once delivery |
| 1 | 0 | 1 | At least once delivery |
| 2 | 1 | 0 | Exactly once delivery |
| - | 1 | 1 | Reserved - must not be used |

Table 2 - Reproduction of Table 3-9 QoS Definitions From MQTT 5.0 Oasis Standard

While describing the use of QoS Levels and other features we have been using the terms “server” and “broker” to designate the service that will distribute our messages. But according to the Standard,

_"The delivery protocol is symmetric, in the description below the Client and Server can each take the role of either sender or receiver. The delivery protocol is concerned solely with the delivery of an application message from a single sender to a single receiver. When the Server is delivering an Application Message to more than one Client, each Client is treated independently. The QoS level used to deliver an Application Message outbound to the Client could differ from that of the inbound Application Message."_ (emphasis is ours)

So, the use of the terms “server” and “ broker” in the sense we have been using it until now is justified because we are talking from the perspective of the Client in a broad sense, but keep in mind this symmetry in the delivery protocol.

The default QoS Level is 0, meaning, if we do not set this flag we will be informing the server that 0 (zero) is the maximum QoS Level we are willing to accept. Any compliant broker accepts this level. It is a “fire and forget” publishing with the sender accepting that both loss and duplication may occur on delivery.

![Fig.3 - MQTT 5.0 - QoS Level 0 Client-Server Flow Diagram](https://c.mql5.com/2/63/MQTT_5.0_-_QoS_Level_0_Client-Server_Flow_Diagram.PNG)

Fig. 3 - MQTT 5.0 - QoS Level 0 Client-Server Flow Diagram

QoS Level 1 accepts that duplication may occur on delivery, but doesn’t accept losses. The server will acknowledge the message with a PUBACK.

![Fig. 4 - MQTT 5.0 - QoS Level 1 Client-Server Flow Diagram](https://c.mql5.com/2/63/MQTT_5.0_-_QoS_Level_1_Client-Server_Flow_Diagram.PNG)

Fig. 4 - MQTT 5.0 - QoS Level 1 Client-Server Flow Diagram

QoS Level 2 requires no loss or duplication. There are four packets involved in this level. The server will recognize the delivery starts with a PUBREC. Then the client will ask for the release of that specific Packet Identifier with a PUBREL, and finally, the server will notify the delivery completion with a PUBCOMP.

![Fig. 5 - MQTT 5.0 - QoS Level 2 Client-Server Flow Diagram](https://c.mql5.com/2/63/MQTT_5.0_-_QoS_Level_2_Client-Server_Flow_Diagram.PNG)

Fig. 5 - MQTT 5.0 - QoS Level 2 Client-Server Flow Diagram

An analogy taken from [the previous article](https://www.mql5.com/en/articles/13651) when we were talking about Connect Flags:

" _One can think of this \[QoS 2\] level as sending a registered parcel. The postal system gives you a receipt when you transfer the packet to their hands acknowledging that, for now on, they are responsible for delivering it to the right address. And when it happens, when they deliver the parcel, they send you a signed receipt from the recipient, acknowledging the parcel delivery._"

Quality of Service can be required for the Will Message, for a Subscription (including Shared Subscriptions), or for a specific message.

| Will Message | Subscription | Message |
| --- | --- | --- |
| CONNECT Will QoS | SUBSCRIBE Subscription Options | PUBLISH QoS Level Flag |

Table 3 - MQTT 5.0 Packets and Flags Where QoS Level Can Be Set

The attentive reader may have noted that both QoS 1 and QoS 2 involves some kind of Session state. We will deal with the Session state and the corresponding persistence layer in an article dedicated solely to this extensive topic.

**DUP**

When set, the DUP flag is saying that we are retrying to send a previous failed PUBLISH packet. It MUST be unset to 0 (zero) for all QoS 0 messages. The duplication refers to the packet itself and not to the message.

### Variable Header of an MQTT 5.0 PUBLISH packet: Topic Name, Packet Identifier, and Properties

The Variable Header of an MQTT 5.0 PUBLISH packet MUST have a Topic Name and, if the QoS is greater than 0 (zero), it also MUST have a Packet Identifier. These two fields are usually followed by a set of properties and a payload, but a PUBLISH packet without properties and a zero-length payload is a valid packet. In other words, the most simple valid PUBLISH packet is the one with a Fixed Header with QoS 0, no DUP and no RETAIN flags, and a Variable Header with only a Topic Name.

**Topic Name**

Since all interactions between clients and server – and by extension, all interactions between the users/devices – in a pub/sub message sharing protocol revolves around publishing to a topic and subscribing to a topic, we can say that the Topic Name field deserves special attention here. In many real-time services, we will find the term “channel” instead of Topic Name. It makes sense because the Topic Name represents the information channel the clients are subscribed to.

A Topic Name is a UTF-8 encoded string organized in a hierarchical tree structure. The forward slash ( /  U+002F ) is used as a topic-level separator.

broker1/account12345/EURUSD

They are case-sensitive. So these are two different topics.

- broker1/account12345/EURUSD
- broker1/account12345/eurusd

These level separators have significance only when either of the Topic Filter wildcard characters (see below) is present on the client subscription. There are no limits for the number of levels, except the limit of the UTF-8 string itself. Eventually, the Topic Name may be replaced by a Topic Alias.

" _A Topic Alias is an integer value that is used to identify the Topic instead of using the Topic Name. This reduces the size of the PUBLISH packet, and is useful when the Topic Names are long and the same Topic Names are used repetitively within a Network Connection._"

**Packet Identifier**

Packet Identifier is a two-byte integer field required for PUBLISH packets with QoS > 0. It is used in all packets directly involved in the pub/sub cycle for Session state management. Packet Identifier MUST NOT be used in PUBLISH with QoS 0.

It is used to connect PUBLISH with their related ACK’s.

Please, remember that, since the delivery protocol is symetric, when using QoS 1 our Client may receive a PUBLISH from the server with the same packet ID before receiving the PUBACK related to a previous PUBLISH we sent.

"It is possible for a Client to send a PUBLISH packet with Packet Identifier 0x1234 and then receive a different PUBLISH packet with Packet Identifier 0x1234 from its Server before it receives a PUBACK for the PUBLISH packet that it sent."

It is worth noting that the Packet Identifier is also used to connect the related ACK’s in SUBSCRIBE and UNSUBSCRIBE packets.

### How we are writing Topic Name(s)

The Topic Name is the first field in the Variable Header. It is encoded as a UTF-8 string with some disallowed Unicode code points, and there is a catch here. Please, take a look at these three statements with some of the requirements to encode a UTF-8 string for MQTT 5.0.

“\[…\] the character data MUST NOT include encodings of code points between U+D800 and U+DFFF. If the Client or Server receives an MQTT Control Packet containing ill-formed UTF-8 it is a Malformed Packet.”

“A UTF-8 Encoded String MUST NOT include an encoding of the null character U+0000. If a receiver (Server or Client) receives an MQTT Control Packet containing U+0000 it is a Malformed Packet.”

“The data SHOULD NOT include encodings of the Unicode \[Unicode\] code points listed below. If a receiver (Server or Client) receives an MQTT Control Packet containing any of them it MAY treat it as a Malformed Packet. These are the Disallowed Unicode code points.

U+0001..U+001F control characters

U+007F..U+009F control characters

Code points defined in the Unicode specification \[Unicode\] to be non-characters”

As you can see, both the first and the second statement above are strict (MUST NOT), meaning that any compliant implementation will check for the presence of the disallowed code points, while the third statement is a recommendation (SHOULD NOT), meaning that an implementation may not check for the presence of the disallowed code points and still be considered compliant.

Since a Malformed Packet is a reason for a DISCONNECT, if we allow these code points in our Client and our broker chooses to not treat them as a Malformed Packet, we may cause the disconnection of other clients that enforce the recommendation. So, despite the exclusion of Unicode control characters and non-characters being only a recommendation, we are not allowing them in our implementation.

For now, our function to encode strings as UTF-8 looks like this:

```
//+------------------------------------------------------------------+
//|                    Encode UTF-8 String                           |
//+------------------------------------------------------------------+
void EncodeUTF8String(string str, ushort& dest_buf[])
  {
   uint str_len = StringLen(str);
// check for disallowed Unicode code points
   uint iter_pos = 0;
   while(iter_pos < str_len)
     {
      Print("Checking disallowed code points");
      ushort code_point = StringGetCharacter(str, iter_pos);
      if(IsDisallowedCodePoint(code_point))
        {
         printf("Found disallowed code point at position %d", iter_pos);
         ZeroMemory(dest_buf);
         return;
        }
      printf("Iter position %d", iter_pos);
      iter_pos++;
     }
   if(str_len == 0)
     {
      Print("Cleaning buffer: string empty");
      ZeroMemory(dest_buf);
      return;
     }
// we have no disallowed code points and the string is not empty: encode it.
   printf("Encoding %d bytes ", str_len);
   ArrayResize(dest_buf, str_len + 2);
   dest_buf[0] = (char)str_len >> 8; // MSB
   dest_buf[1] = (char)str_len % 256; // LSB
   ushort char_array[];
   StringToShortArray(str, char_array, 0, str_len);// to Unicode
   ArrayCopy(dest_buf, char_array, 2);
   ZeroMemory(char_array);
  }
```

If the string passed to this function has a disallowed code point, we log its position on the string, pass the destination buffer to ZeroMemory, and return immediately. As a Topic Name has a minimum required length of 1, if the string is empty we do the same: log, cleanup the buffer, and return.

By the way, note that we are using StringToShortArray to convert the string to a Unicode array. If we were converting it to an ASCII array, we would use StringToCharArray. You can find the detailed explanation and much more in [the book recently included in the documentation](https://www.mql5.com/en/book/common/strings/strings_codepages), or in this comprehensive [article about MQL5 strings](https://www.mql5.com/en/articles/585).

Also note that in this same call to StringToShortArray, we are using the length of the string as the last parameter, instead of the function default. That is because we do not want the null character (0x00) in our array, and according to the function documentation,

_“Default value is -1, which means copying up to the array end, or till terminal 0. Terminal 0 will also be copied to the recipient array”_

while StringLen return value is the

_“Number of symbols in a string without the ending zero.”_

The function to check for disallowed code points is trivial.

```
//+------------------------------------------------------------------+
//|              IsDisallowedCodePoint                               |
//|   https://unicode.org/faq/utf_bom.html#utf16-2                   |
//+------------------------------------------------------------------+
bool IsDisallowedCodePoint(ushort code_point)
  {
   if((code_point >= 0xD800 && code_point <= 0xDFFF) // Surrogates
      || (code_point > 0x00 && code_point <= 0x1F) // C0 - Control Characters
      || (code_point >= 0x7F && code_point <= 0x9F) // C0 - Control Characters
      || (code_point == 0xFFF0 || code_point == 0xFFFF)) // Specials - non-characters
     {
      return true;
     }
   return false;
  };
```

Besides disallowed code points, we also need to check for the two wildcard characters that are used in subscriptions Topic Filters, but forbidden in Topic Name: the plus sign (‘+’ U+002B) and the number sign (‘#’ U+0023).

The function to check for disallowed code points will be of general use to encode any string, so it is hosted on our MQTT.mqh header, while the function to check for wildcard characters is specific to Topic Name, so it is part of our _CPktPublish_ class.

```
//+------------------------------------------------------------------+
//|            CPktPublish::HasWildcardChar                          |
//+------------------------------------------------------------------+
bool CPktPublish::HasWildcardChar(const string str)
  {
   if(StringFind(str, "#") > -1 || StringFind(str, "+") > -1)
     {
      printf("Wildcard char not allowed in Topic Names");
      return true;
     }
   return false;
  }
```

The built-in function StringFind returns the start position of the matching substring and -1 if the matching substring is not found. So we just check for any value above -1. Then we call it from the main function.

```
//+------------------------------------------------------------------+
//|            CPktPublish::SetTopicName                             |
//+------------------------------------------------------------------+
void CPktPublish::SetTopicName(const string topic_name)
  {
   if(HasWildcardChar(topic_name) || StringLen(topic_name) == 0)
     {
      ArrayFree(ByteArray);
      return;
     }
   ushort encoded_string[];
   EncodeUTF8String(topic_name, encoded_string);
   ArrayCopy(ByteArray, encoded_string, 2);
   ByteArray[1] = EncodeVariableByteInteger(encoded_string);
  }
```

At this point, if a wildcard is found, we do the same “error handling” we have been doing: we log the info, clear the buffer, and return immediately. Later we can improve this, by raising alerts, for example.

The last line of the function assigns the packet's remaining length to the second byte of our fixed header using the algorithm suggested by the Standard. We commented about it in the first article of this series.

Our tests also follow the exact same structure.

```
//+------------------------------------------------------------------+
//|           TEST_SetTopicName_WildcardChar_NumberSign              |
//+------------------------------------------------------------------+
bool TEST_SetTopicName_WildcardChar_NumberSign()
  {
   Print(__FUNCTION__);
//--- Arrange
   static uchar expected[] = {};
   uchar payload[] = {};
//--- Act
   CPktPublish *cut = new CPktPublish(payload);
   cut.SetTopicName("a#");
   uchar result[];
   ArrayCopy(result, cut.ByteArray);
//--- Assert
   bool isTrue = AssertEqual(expected, result);
//--- cleanup
   delete cut;
   ZeroMemory(result);
   return isTrue;
  }
```

If you run the tests, you should see something like this:

![Fig. 6 - MQTT 5.0 - PUBLISH Test Output Topic Name](https://c.mql5.com/2/63/MQTT_5.0_-_PUBLISH_Test_Output_Topic_Name.PNG)

Fig. 6 - MQTT 5.0 - PUBLISH Test Output Topic Name

### How we are writing Packet Identifier(s)

The Packet Identifier is NOT meant to be assigned by the user. Instead, it MUST be assigned by the Client to any PUBLISH packet where QoS Level is > 0, and MUST NOT be assigned otherwise. In other words, every time we build a PUBLISH packet with QoS 1 or QoS 2 we must set its Packet Identifier.

We can start testing this now. All that we need is to instantiate a packet and set its required Topic Name and its QoS to 1 or 2. The resulting packet byte array should have a packet ID.

```
//+------------------------------------------------------------------+
//|            TEST_SetPacketID_QoS2_TopicName1Char                  |
//+------------------------------------------------------------------+
bool TEST_SetPacketID_QoS2_TopicName5Char()
  {
   Print(__FUNCTION__);
// Arrange
   uchar payload[] = {};
   uchar result[]; // expected {52, 9, 0, 1, 'a', 'b', 'c', 'd', 'e', pktID MSB, pktID LSB}
// Act
   CPktPublish *cut = new CPktPublish(payload);
// FIX: if we call SetQoS first this test breaks
   cut.SetTopicName("abcde");
   cut.SetQoS_2(true);
   ArrayCopy(result, cut.ByteArray);
// Assert
   ArrayPrint(result);
   bool is_true = result[9] > 0 || result[10] > 0;
// cleanup
   delete cut;
   ZeroMemory(result);
   return is_true;
  }
```

Note that we cannot test for the value of the generated packet ID because it is a (pseudo) random generated number, as you can see below in the stub implementation. We are testing for its presence instead. Also, note that we have a FIX to be done. The order of functions calling for SetTopicName and SetQoS\_X is affecting the resulting byte array in an unexpected way. It is not a good idea to have a calling-order dependency between functions. This would be a bug, but as the saying goes, a bug is a test not written. So, we will be writing a test for not having this calling-order dependency in the next iteration. For now, we are only concerned with making this test pass.

Of course, the test will not even compile until we have an implementation of the function to set packet ID’s. Since Packet Identifier(s) are required in several control packets, the function to write it should NOT be a member of the CPktPublish class. The MQTT.mqh header seems to be a more suitable file to host it.

```
//+------------------------------------------------------------------+
//|            SetPacketID                                           |
//+------------------------------------------------------------------+
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
   buf[start_idx + 1] = (uchar)packet_id % 256; //LSB
  }
```

We are using the built-in function MathRand to generate packet identifiers. It requires that we call MathSrand before. We must pass to this function the ‘seed’ for the random generator. We chose TimeLocal as the seed, following the recommendation we found in the book recently added to the documentation with a clear reference about [pseudo-random number generation in MQL5](https://www.mql5.com/en/book/common/maths/maths_rand).

To set the packet ID, we resize the original byte array to open room for the packet ID (two bytes integer) and set the values of the most significant byte and the least significant byte starting from the position passed as argument (start\_idx). The last step is to call the function from our CPktPublish class on the methods SetQoS\_1 and SetQoS\_2.

```
//+------------------------------------------------------------------+
//|            CPktPublish::SetQoS_2                                 |
//+------------------------------------------------------------------+
void CPktPublish::SetQoS_2(const bool QoS_2)
  {
   QoS_2 ? m_publish_flags |= QoS_2_FLAG : m_publish_flags &= ~QoS_2_FLAG;
   SetFixedHeader(PUBLISH, m_buf, ByteArray, m_publish_flags);
   SetPacketID(ByteArray, ByteArray.Size());
  }
```

By running the tests included on the attached files you should see something like this (stripped for brevity here):

![Fig. 7 - MQTT 5.0 - PUBLISH Test Output Packet Identifier](https://c.mql5.com/2/63/MQTT_5.0_-_PUBLISH_Test_Output_Packet_Identifier.PNG)

Fig. 7 - MQTT 5.0 - PUBLISH Test Output Packet Identifier

### Conclusion

By being at the core of protocol, PUBLISH packets are a bit more demanding to implement: they have different fixed headers, they require a variable header with the Topic Name encoded as UTF-8 and guarded against some disallowed code points, they require a packet identifier if QoS > 0, and they may use almost all the properties and user properties available in MQTT 5.0.

In this article, we reported how we are building valid PUBLISH headers with Publish Flags, Topic Name, and Packet Identifier. In the next article of this series, we will see how we are writing its Properties.

As a side note about the last changes: If you are following the development of this MQTT Client, you may have noted that we changed several function signatures, variable names, field access levels, test fixtures, etc. Some of these changes are the ones expected in any software development, but most of them are due to the fact that we are using a TDD approach and striving to keep as faithful as possible to this methodology so it can be reported here in these articles. We can expect a lot of change before we have a first deliverable.

As you know, no developer alone knows everything that is needed to develop a Client like this for our Code Base. TDD is helping a lot in our “huge specs, baby steps” journey, but if you can help, please drop a note in our Community Chat or in the comments below. Any help is more than welcome. Thank you.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/13998.zip "Download all attachments in the single ZIP archive")

[MQTT-part5-headers-and-tests.zip](https://www.mql5.com/en/articles/download/13998/mqtt-part5-headers-and-tests.zip "Download MQTT-part5-headers-and-tests.zip")(208.79 KB)

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

**[Go to discussion](https://www.mql5.com/en/forum/460359)**

![Introduction to MQL5 (Part 2): Navigating Predefined Variables, Common Functions, and  Control Flow Statements](https://c.mql5.com/2/64/Introduction_to_MQL5_4Part_2c__LOGO-transformed.png)[Introduction to MQL5 (Part 2): Navigating Predefined Variables, Common Functions, and Control Flow Statements](https://www.mql5.com/en/articles/13997)

Embark on an illuminating journey with Part Two of our MQL5 series. These articles are not just tutorials, they're doorways to an enchanted realm where programming novices and wizards alike unite. What makes this journey truly magical? Part Two of our MQL5 series stands out with its refreshing simplicity, making complex concepts accessible to all. Engage with us interactively as we answer your questions, ensuring an enriching and personalized learning experience. Let's build a community where understanding MQL5 is an adventure for everyone. Welcome to the enchantment!

![Data label for time series mining (Part 5)：Apply and Test in EA Using Socket](https://c.mql5.com/2/64/Data_label_for_time_series_miningbPart_50_Apply_and_Test_in_EA_Using_Socket_____LOGO.png)[Data label for time series mining (Part 5)：Apply and Test in EA Using Socket](https://www.mql5.com/en/articles/13254)

This series of articles introduces several time series labeling methods, which can create data that meets most artificial intelligence models, and targeted data labeling according to needs can make the trained artificial intelligence model more in line with the expected design, improve the accuracy of our model, and even help the model make a qualitative leap!

![Data Science and Machine Learning (Part 18): The battle of Mastering Market Complexity, Truncated SVD Versus NMF](https://c.mql5.com/2/64/Data_Science_and_Machine_Learning_pPart_183_Truncated_SVD_Versus_NMF__LOGO.png)[Data Science and Machine Learning (Part 18): The battle of Mastering Market Complexity, Truncated SVD Versus NMF](https://www.mql5.com/en/articles/13968)

Truncated Singular Value Decomposition (SVD) and Non-Negative Matrix Factorization (NMF) are dimensionality reduction techniques. They both play significant roles in shaping data-driven trading strategies. Discover the art of dimensionality reduction, unraveling insights, and optimizing quantitative analyses for an informed approach to navigating the intricacies of financial markets.

![Implementation of the Augmented Dickey Fuller test in MQL5](https://c.mql5.com/2/64/Implementation_of_the_Augmented_Dickey_Fuller_test_in_MQL5__LOGO.png)[Implementation of the Augmented Dickey Fuller test in MQL5](https://www.mql5.com/en/articles/13991)

In this article we demonstrate the implementation of the Augmented Dickey-Fuller test, and apply it to conduct cointegration tests using the Engle-Granger method.

[![](https://www.mql5.com/ff/si/3p2yc19r7qvs297n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F618%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dsignal.advantage%26utm_content%3Dsubscribe.signal%26utm_campaign%3D0622.MQL5.com.Internal&a=bewozmaxwejekdopjicjtsbzmjgfjyvt&s=e49ac7e84b713650e3af82ec3c6b4d02fdf06617c5821011b1e499af5edd01f4&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=erylyqdkdhwcllpchzgjorxpnmzdotvv&ssn=1769178116711581199&ssn_dr=1&ssn_sr=0&fv_date=1769178116&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F13998&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Developing%20an%20MQTT%20client%20for%20Metatrader%205%3A%20a%20TDD%20approach%20%E2%80%94%20Part%205%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176917811701255694&fz_uniq=5068143795691779588&sv=2552)

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