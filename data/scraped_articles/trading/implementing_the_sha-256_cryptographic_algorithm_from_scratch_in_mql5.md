---
title: Implementing the SHA-256 Cryptographic Algorithm from Scratch in MQL5
url: https://www.mql5.com/en/articles/16357
categories: Trading, Integration, Expert Advisors
relevance_score: 9
scraped_at: 2026-01-22T17:28:08.507259
---

[Need a reliable hosting solution for your robots?Contact your broker and find out about available Sponsored MetaTrader VPS offeringsLearn more![](https://www.mql5.com/ff/sh/0pw0dk81s56qy774z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=vljwvezfjkfbvviocwskggexlvgykvob&s=70cf8e354b9a125332533ffb65d7365abe8dde5b5c1ede9caac479a9e9df4f25&uid=&ref=https://www.mql5.com/en/articles/16357&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5049154267807327734)

MetaTrader 5 / Examples


Advances in MQL5 enable efficient implementation of cryptographic algorithms without third-party solutions, enhancing security, optimization, and stability. Benefits of custom implementation include task-specific adjustments, elimination of vulnerabilities, and consistent performance across environments. Cryptography is now integral to APIs, transaction verification, and data integrity, requiring developers to deeply understand these algorithms. The convergence of cryptography and trading systems has become a mandatory aspect of modern platform development.

### API Signature Compatibility Challenges

The primary driver for custom SHA-256 implementation lies in the fundamental incompatibility between MQL5's native hashing functions and cryptocurrency exchange requirements. This incompatibility manifests in several critical ways that directly impact trading operations.

When trading platforms make API calls to cryptocurrency exchanges like Binance or Bybit, they must generate cryptographic signatures that perfectly match the exchange's expectations. These signatures serve as proof of authenticity for each request, ensuring that orders and other sensitive operations come from authorized sources. However, MQL5's built-in cryptographic functions often generate signatures that differ from those produced by standard implementations in other programming languages.

This signature mismatch occurs because cryptocurrency exchanges typically expect signatures generated according to specific standards, often based on implementations in languages like Python or JavaScript. MQL5's native functions, while functional for general purposes, may handle certain aspects of the hashing process differently, such as:

1. Byte ordering in the input processing
2. Padding implementations
3. Character encoding handling
4. Memory representation of intermediate values
5. Handling of special characters in the input string

These differences lead to several serious operational issues in production environments:

First, API requests get rejected by exchanges because the signatures don't match their expected values. This rejection happens at the authentication layer, before the actual trading logic even begins to execute. Consider a scenario where a trading algorithm identifies a profitable opportunity and attempts to place an order. If the signature is invalid, the exchange immediately rejects the request, causing the system to miss the trading opportunity entirely.

Second, authentication failures occur consistently across different types of API calls. Even simple operations like retrieving account balances or checking order status become impossible because the exchange cannot verify the request's authenticity. This creates a systemic problem that affects every aspect of the trading system's operation.

Third, order placement becomes particularly problematic. When attempting to place orders, invalid signatures prevent the execution of trades, potentially leading to:

- Missed entry points for planned trades
- Inability to exit positions when needed
- Failed risk management operations
- Incomplete execution of trading strategies

Fourth, message verification becomes unreliable. Many exchanges use hash-based verification for various types of messages and notifications. Inconsistent hash outputs mean the trading system cannot reliably verify these messages, potentially missing important updates or acting on unverified information.

The impact of these compatibility issues extends beyond immediate technical problems. They affect the entire trading operation in subtle but significant ways:

- Trading strategies cannot be executed reliably
- Risk management systems may fail to operate as intended
- System monitoring and logging become less reliable
- Integration with exchange APIs requires constant workarounds

For example, consider a typical API call to a cryptocurrency exchange. The same input data might produce entirely different signatures:

Using MQL5's built-in function:

```
void OnStart()
{
   // The text to hash
   string text = "Hello";
   string key_text = "key";

   // Method 1: Using HashCalculate() - Returns uchar array
   uchar data[];
   uchar key[];
   uchar result[];
   StringToCharArray(text, data);
   StringToCharArray(key_text, key);

  int res =  CryptEncode(CRYPT_HASH_SHA256, data, key, result);

  Print(ArrayToHex(result));
}

/*Result
 D9D3734CD05564A131946ECF9E240E0319CA2F5BA321BD9F87D634A24A29EF4D
*/
```

Using standard implementation:

```

import hashlib
import hmac

text = "Hello" #message
key = "key"  #password

hash_object = hmac.new(bytes(key, 'utf-8'), text.encode('utf-8'), hashlib.sha256)
hex_dig = hash_object.hexdigest()

hex_dig

##output >>> "c70b9f4d665bd62974afc83582de810e72a41a58db82c538a9d734c9266d321e"
```

Discrepancies in signature generation can cause valid trading signals and strategies to fail. Different exchanges may have unique signature requirements, necessitating a flexible, customizable implementation.

Custom SHA-256 implementations provide developers with control to:

    Align with specific exchange requirements.

    Adapt to evolving standards.

    Debug signature-related issues.

    Ensure consistency across the trading system.

This control is crucial for high-frequency trading and complex strategies where reliability and speed are paramount. While challenging, custom implementations enhance reliability and integration in production trading systems.

**Difference in implementation of HMAC and SHA-256**

- Python uses the hmac library, which performs HMAC (Hash-based Message Authentication Code). This is not just hashing the text text with the key , but a combined process:
1. The key is supplemented (or cut) to the required size.
2. Two hashing steps are performed: first with the key and the message, then with the key and the intermediate result.
- MQL5 in this code only performs SHA-256 calculation via CryptEncode() without full HMAC implementation. This is regular text hashing with a key, not HMAC.

**Conclusion** : Python uses HMAC, while MQL5 uses only SHA-256, which already guarantees different results.

### Performance Optimization: A Deep Dive

When implementing SHA-256 in a trading environment, performance optimization becomes crucial because every millisecond can impact trading outcomes. A custom implementation offers several avenues for optimization that wouldn't be possible with built-in functions.

Trading systems often exhibit specific patterns in their cryptographic operations. For instance, order signatures typically contain similar components like timestamps, symbols, and quantities. By understanding these patterns, we can optimize our SHA-256 implementation specifically for trading-related data structures.

Consider how order placement works in a typical trading scenario. Each order requires multiple pieces of information: the trading pair, order type, quantity, price, and timestamp. In standard implementations, this data would be processed as a completely new input each time. However, we can optimize this process by recognizing that many components remain constant or follow predictable patterns.

Let's examine how this optimization works in practice. When generating signatures for orders, much of the data structure remains consistent:

```
baseEndpoint/symbol=BTCUSDT&side=BUY&type=LIMIT&quantity=0.1&price=50000&timestamp=1234567890
```

In this string, only a few elements typically change between orders: the quantity, price, and timestamp. By structuring our implementation to efficiently handle these partial changes, we can significantly improve performance. This might involve:

- Creating specialized preprocessing functions that efficiently handle common trading data structures,
- Implementing intelligent buffer management for frequently used components,
- Developing optimized parsing routines for numeric values that commonly appear in trading operations.

Memory management becomes particularly important in the MQL5 environment, which has its own specific constraints and characteristics. The MetaTrader platform operates with limited resources, and inefficient memory usage can impact the entire trading system's performance. A custom implementation allows us to fine-tune memory allocation and deallocation based on the exact needs of our trading operations.

We can implement sophisticated caching strategies that recognize the temporal locality of trading operations. For instance, during high-frequency trading sessions, certain trading pairs or order types might be used repeatedly. By caching the intermediate hash states for these common patterns, we can reduce the computational overhead for subsequent operations.

### Future-Proofing: Ensuring Long-Term Viability

The cryptocurrency landscape is notably dynamic, with exchanges frequently updating their requirements and security protocols. A custom SHA-256 implementation provides the flexibility needed to adapt to these changes while maintaining system reliability.

Consider how exchanges might modify their signature requirements over time. They might:

- Change the order of parameters in the signature string
- Add new required fields to the signature
- Modify how certain characters or special cases are handled
- Implement new security measures that affect how signatures are generated

With a custom implementation, adapting to these changes becomes straightforward. We maintain complete control over every aspect of the hashing process, from initial data preprocessing to final output formatting. This control allows us to quickly respond to new requirements without waiting for updates to MQL5's built-in functions.

For example, if an exchange decides to modify how it handles Unicode characters in signatures, we can immediately update our implementation to match the new requirements. This level of adaptability becomes crucial when dealing with multiple exchanges, each potentially having different requirements.

Independence from MQL5's built-in functions provides another significant advantage. As MetaTrader evolves and updates its platform, built-in functions might change in subtle ways that could affect signature generation. A custom implementation remains stable across different MetaTrader versions, ensuring consistent behavior regardless of platform updates.

The future-proofing aspect extends beyond just signature requirements. Cryptocurrency exchanges might introduce new security features or authentication methods that build upon SHA-256. Having a custom implementation allows us to:

- Extend the basic SHA-256 functionality to support new security features
- Modify the implementation to work with new authentication schemes
- Integrate additional cryptographic operations seamlessly
- Maintain backward compatibility while adding new capabilities

Furthermore, a custom implementation provides a foundation for implementing other cryptographic functions that might become necessary in the future. The code structure and optimization techniques developed for SHA-256 can serve as a template for implementing other hash functions or cryptographic operations.

Consider a scenario where an exchange introduces a new requirement for double hashing or combines SHA-256 with another algorithm. With a custom implementation, adding these features becomes a matter of extending the existing code rather than trying to work around the limitations of built-in functions.

This extensibility is particularly valuable in the rapidly evolving cryptocurrency trading landscape. New trading patterns, security requirements, and technological advances can emerge quickly, and having a flexible, customizable implementation allows trading systems to adapt and evolve alongside the market.

The combination of performance optimization and future-proofing capabilities makes a custom SHA-256 implementation invaluable for serious cryptocurrency trading operations. It provides the control, flexibility, and efficiency needed to maintain competitive advantage in a fast-moving market while ensuring long-term viability as requirements evolve.

### Understanding the SHA-256 Algorithm

SHA-256 stands for Secure Hash Algorithm 256-bit, which is a cryptographic hash function taking arbitrary input and mapping a fixed-size 256-bit (32-byte) output. It belongs to the SHA-2 family of algorithms and follows a very systematic and well-defined set of steps in its process—a feature that balances both security and determinism.

**MQL5 Implementation**

Let's examine each component in detail. The complete implementation is available in the final section for reference. In this implementation, we will have two classes, similar to the Python implementation where the HMAC and SHA256 classes are used:

Here is the structure of the SHA256 and HMAC class.

```
class CSha256Class
  {
private:

   uint              total_message[];
   uint              paddedMessage[64];
   void              Initialize_H();//This function initializes the values of h0-h7
   uint              RawMessage[];//Keeps track of the raw message sent in
public:
   //hash values from h0 - h7
   uint              h0;
   uint              h1;
   uint              h2;
   uint              h3;
   uint              h4;
   uint              h5;
   uint              h6;
   uint              h7;

   uint              K[64];
   uint              W[64];

                     CSha256Class(void);
                    ~CSha256Class() {};

   void              PreProcessMessage(uint &message[], int messageLength);
   void              CreateMessageSchedule();
   void              Compression();
   void              UpdateHash(uint &message[], int message_len);
   void              GetDigest(uint &digest[]);
   string            GetHex();
  };
```

```
class HMacSha256
  {
private:

public:
   uint              k_ipad[64];
   uint              k_opad[64];
   uint              K[];
   string            hexval;
                     HMacSha256(string key, string message);
                    ~HMacSha256() {};
   CSha256Class      myshaclass;
   void              ProcessKey(string key);
  };
```

### Step-by-Step Guide to Implementing These Classes

This module implements the HMAC algorithm as described by [RFC 2104](https://www.mql5.com/go?link=https://datatracker.ietf.org/doc/html/rfc2104.html "https://datatracker.ietf.org/doc/html/rfc2104.html").

Step 1:

Creating a Function to Process the Key.

```
void HMacSha256::ProcessKey(string key)
  {
   int keyLength = StringLen(key);//stores the length of the key

   if(keyLength>64)
     {
      uchar keyCharacters[];

      StringToCharArray(key, keyCharacters);
      uint KeyCharuint[];
      ArrayResize(KeyCharuint, keyLength);

      //Converts the keys to their characters
      for(int i=0;i<keyLength;i++)
         KeyCharuint[i] = (uint)keyCharacters[i];

      //Time to hash the
      CSha256Class keyhasher;
      keyhasher.UpdateHash(KeyCharuint, keyLength);

      uint digestValue[];
      keyhasher.GetDigest(digestValue);
      ArrayResize(K, 64);

      for(int i=0;i<ArraySize(digestValue);i++)
         K[i] = digestValue[i];

      for(int i=ArraySize(digestValue);i<64;i++)
         K[i] = 0x00;

     }
   else
     {
      uchar keyCharacters[];

      StringToCharArray(key, keyCharacters);
      ArrayResize(K, 64);

      for(int i=0;i<keyLength;i++)
         K[i] = (uint)keyCharacters[i];

      for(int i=keyLength;i<64;i++)
         K[i] = 0x00;
     }
  }
```

This implementation takes into accout the length of the key, if its greater than 64 then its first hashed using our CSha256Class.

Step 2:

Full Implementation of HMAC.

```
HMacSha256::HMacSha256(string key,string message)
  {
//process key and add zeros to complete n bytes of 64
   ProcessKey(key);

   for(int i=0;i<64;i++)
     {
      uint keyval = K[i];
      k_ipad[i] = 0x36 ^ keyval;
      k_opad[i] = 0x5c ^ keyval;
     }

//text chars
   uchar messageCharacters[];
   StringToCharArray(message, messageCharacters);
   int innerPaddingLength = 64+StringLen(message);

   uint innerPadding[];
   ArrayResize(innerPadding, innerPaddingLength);

   for(int i=0;i<64;i++)
      innerPadding[i] = k_ipad[i];

   int msg_counts = 0;
   for(int i=64;i<innerPaddingLength;i++)
     {
      innerPadding[i] = (uint)messageCharacters[msg_counts];
      msg_counts +=1;
     }

//send inner padding for hashing
   CSha256Class innerpaddHasher;
   innerpaddHasher.UpdateHash(innerPadding, ArraySize(innerPadding));

   uint ipad_digest_result[];
   innerpaddHasher.GetDigest(ipad_digest_result);

//   merge digest with outer padding
   uint outerpadding[];
   int outerpaddSize = 64 + ArraySize(ipad_digest_result);

   ArrayResize(outerpadding, outerpaddSize);
   for(int i=0;i<64;i++)
      outerpadding[i] = k_opad[i];

   int inner_counts = 0;
   for(int i=64;i<outerpaddSize;i++)
     {
      outerpadding[i] = ipad_digest_result[inner_counts];
      inner_counts+=1;
     }

   CSha256Class outerpaddHash;
   outerpaddHash.UpdateHash(outerpadding, ArraySize(outerpadding));
   hexval = outerpaddHash.GetHex();
  }
```

Other functions are available in the complete code attached.

The key component of this implementation is the Hash function, which is handled by the CSHa256Class.

The Hashing function involves several steps of operation on the text.

Step 1: Preprocessing of values.

We need to ensure the whole data is a multiple of 512 bits, So we apply some padding operations.

1. Convert the message to its binary form.
2. Append 1 to the end.
3. Add zeros as padding until the data length reaches 448 bits (which is 512 - 64 bits). This ensures we have exactly 64 bits remaining in the 512-bit block. In other words, we pad the message so it's congruent to 448 modulo 512.

4. Append 64 bits to the end, where the 64 bits are a big-endian integer representing the length of the original input in binary.

Step 2: Initialization of the Hash values.

Step 3: Initialization of round Constants.

Step 4: Chunk the total message bits into 512 bits per chunk and perform the following operation on each chunk iteratively.

```
   int chunks_count = (int)MathFloor(ArraySize(total_message)/64.0);
   int copied = 0;

   for(int i=0; i<chunks_count; i++)
     {
      uint newChunk[];
      ArrayResize(newChunk, 64);
      ArrayInitialize(newChunk, 0);  // Initialize chunk array

      for(int j=0; j<64; j++)
        {
         newChunk[j] = total_message[copied];
         copied += 1;
        }

      PreProcessMessage(newChunk, ArraySize(newChunk));
      CreateMessageSchedule();
      Compression();
     }
```

Step 5: Pre-process message by copying the chunk array into a new array where each entry is a 32-bit word.

```
void CSha256Class::PreProcessMessage(uint &message[],int messageLength)
  {
   ArrayInitialize(paddedMessage, 0);
   for(int i=0; i < messageLength; i++)
      paddedMessage[i] = message[i];
  }
```

Step 6: Create the message schedule for this chunk.

```
void CSha256Class::CreateMessageSchedule()
  {
   ArrayInitialize(W, 0);

   int counts = 0;
   for(int i=0; i<ArraySize(paddedMessage); i+=4)
     {
      //32 bits is equivalent to 4 bytes from message
      uint byte1 = paddedMessage[i];
      uint byte2 = paddedMessage[i+1];
      uint byte3 = paddedMessage[i+2];
      uint byte4 = paddedMessage[i+3];

      uint combined = ((byte1 << 24) | (byte2 << 16) | (byte3 << 8) | byte4);
      W[counts] = combined & 0xFFFFFFFF;

      counts += 1;
     }
   for(int i=counts; i<64; i++)
      W[i] = 0x00000000;
//preserve previous counts

   int prev_counts = counts;
   int left_count = 64-counts;
   for(int i=counts; i<64; i++)
     {
      uint s0 = (RightRotate(W[i-15], 7)) ^ (RightRotate(W[i-15],18)) ^ (W[i-15] >> 3);
      uint s1 = (RightRotate(W[i-2], 17)) ^ (RightRotate(W[i-2],19)) ^ (W[i-2] >> 10);

      W[i] = (W[i-16] + s0 + W[i-7] + s1) & 0xFFFFFFFF;
     }
  }
```

Step 7:  Apply the compression loop.

```
void CSha256Class::Compression(void)
  {
   uint a = h0;
   uint b = h1;
   uint c = h2;
   uint d = h3;
   uint e = h4;
   uint f = h5;
   uint g = h6;
   uint h = h7;

   for(int i=0; i<64; i++)
     {
      uint S1 = (RightRotate(e, 6) ^ RightRotate(e,11) ^ RightRotate(e,25)) & 0xFFFFFFFF;
      uint ch = ((e & f) ^ ((~e) & g))& 0xFFFFFFFF;
      uint temp1 = (h + S1 + ch + K[i] + W[i]) & 0xFFFFFFFF;
      uint S0 = (RightRotate(a, 2) ^ RightRotate(a, 13) ^ RightRotate(a, 22)) & 0xFFFFFFFF;
      uint maj = ((a & b) ^ (a & c) ^ (b & c)) & 0xFFFFFFFF;
      uint temp2 = (S0 + maj) & 0xFFFFFFFF;
      h = g & 0xFFFFFFFF;
      g = f & 0xFFFFFFFF;
      f = e & 0xFFFFFFFF;
      e = (d + temp1) & 0xFFFFFFFF;
      d = c & 0xFFFFFFFF;
      c = b & 0xFFFFFFFF;
      b = a & 0xFFFFFFFF;
      a = (temp1 + temp2)&0xFFFFFFFF;
     }

   h0 = h0 + a;
   h1 = h1 + b;
   h2 = h2 + c;
   h3 = h3 + d;
   h4 = h4 + e;
   h5 = h5 + f;
   h6 = h6 + g;
   h7 = h7 + h;

  }
```

After completing the compression loop, we update the final hash values by adding the modified working variables to the original hash values.

This addition operation is performed after processing each 512-bit chunk of the message, and the updated hash values become the starting point for processing the next chunk. This chaining mechanism ensures that each block's hash depends on all previous blocks, making the final hash value dependent on the entire message.

### How to use this class

To generate API signatures for either Binance or Bybit, create an HMAC (Hash-based Message Authentication Code) instance, similar to Python's implementation.

```
void OnStart()
{
   // The text to hash
   string text = "Hello";
   string key_text = "key";

  HMacSha256 sha256(key_text, text);
  Print(sha256.hexval);
}

>>> C70B9F4D665BD62974AFC83582DE810E72A41A58DB82C538A9D734C9266D321E
```

By comparing the signatures generated by both implementations, you'll see they produce identical results.

### Conclusion

Traders dealing with cryptocurrencies often face issues with API signature compatibility, performance optimization, and future-proofing when using built-in cryptographic functions in trading systems like MetaTrader 5. This article guides traders on implementing SHA-256 from scratch in MQL5 to overcome these problems.

In conclusion, creating a custom SHA-256 implementation offers traders compatibility with exchanges, enhanced performance, and flexibility to adapt to future changes, making it an essential strategy for secure and efficient cryptocurrency trading operations.

Remember to regularly test the implementation against standard test vectors and validate signatures with your target exchanges before deploying to production. As cryptocurrency exchanges evolve their security requirements, having this flexible, custom implementation will prove invaluable for quick adaptation and maintenance.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/16357.zip "Download all attachments in the single ZIP archive")

[Sha256Algorithm.mqh](https://www.mql5.com/en/articles/download/16357/sha256algorithm.mqh "Download Sha256Algorithm.mqh")(18.56 KB)

[Sha256TestFile.mq5](https://www.mql5.com/en/articles/download/16357/sha256testfile.mq5 "Download Sha256TestFile.mq5")(0.89 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/479956)**
(6)


![Abdulkudus Okatengwu Kadir](https://c.mql5.com/avatar/avatar_na2.png)

**[Abdulkudus Okatengwu Kadir](https://www.mql5.com/en/users/abdulkudus)**
\|
27 Feb 2025 at 14:08

**Sergey Zhilinskiy [#](https://www.mql5.com/en/forum/479956#comment_56022724):**

If key\_text is more than 64 characters, then HMacSha256 calculates incorrectly. What should be corrected for this?

My current implementation adapts to keys with more than 64 characters length. Do you have any specific key,message pair for which it did not work?

![Sergey Zhilinskiy](https://c.mql5.com/avatar/2019/12/5DEB1CBA-61DB.png)

**[Sergey Zhilinskiy](https://www.mql5.com/en/users/zhserg)**
\|
27 Feb 2025 at 15:32

**Abdulkudus Okatengwu Kadir [#](https://www.mql5.com/en/forum/479956#comment_56025183):**

My current implementation adapts to keys with more than 64 characters length. Do you have any specific key,message pair for which it did not work?

string text = "Hello";

string key\_text = "1234567890123456789012345678901234567890123456789012345678901234";

[https://www.devglan.com/online-tools/hmac-sha256-online](https://www.mql5.com/go?link=https://www.devglan.com/online-tools/hmac-sha256-online "https://www.devglan.com/online-tools/hmac-sha256-online") -\> 7558a77ff19ed6cb4777355e4bbc4772759a8130e1bb0913ba62b88411fdbaf8

Test script -> 2025.02.27 22:28:43.792Sha256TestFile (EURUSD,M5)6d8ee9dc1d16261fd986fafb97d919584aa206ca76706fb3deccc63ab2b7f6b

if  string key\_text = "123456789012345678901234567890123456789012345678901234567890123" - OK

![Abdulkudus Okatengwu Kadir](https://c.mql5.com/avatar/avatar_na2.png)

**[Abdulkudus Okatengwu Kadir](https://www.mql5.com/en/users/abdulkudus)**
\|
28 Feb 2025 at 11:39

**Sergey Zhilinskiy [#](https://www.mql5.com/en/forum/479956#comment_56026132):**

string text = "Hello";

string key\_text = "1234567890123456789012345678901234567890123456789012345678901234";

[https://www.devglan.com/online-tools/hmac-sha256-online](https://www.mql5.com/go?link=https://www.devglan.com/online-tools/hmac-sha256-online "https://www.devglan.com/online-tools/hmac-sha256-online") -\> 7558a77ff19ed6cb4777355e4bbc4772759a8130e1bb0913ba62b88411fdbaf8

Test script -> 2025.02.27 22:28:43.792Sha256TestFile (EURUSD,M5)6d8ee9dc1d16261fd986fafb97d919584aa206ca76706fb3deccc63ab2b7f6b

if  string key\_text = "123456789012345678901234567890123456789012345678901234567890123" - OK

I just tried it on my terminal and got same as the online hash tool:

2025.02.28 12:37:16.468hashin\_example\_code (EURUSD,M5)7558A77FF19ED6CB4777355E4BBC4772759A8130E1BB0913BA62B88411FDBAF8

using the code below:

```
void Hash()
{
   // The text to hash
   string text = "Hello";
   string key_text = "1234567890123456789012345678901234567890123456789012345678901234";
   HMacSha256 myhash(key_text, text);
   Print(myhash.hexval);
}
```

You might want to share your code.

![Sergey Zhilinskiy](https://c.mql5.com/avatar/2019/12/5DEB1CBA-61DB.png)

**[Sergey Zhilinskiy](https://www.mql5.com/en/users/zhserg)**
\|
1 Mar 2025 at 02:50

Yes, it works correctly with the original Sha256Algorithm.mqh. I made some changes, maybe that's why it didn't work?

```
 string CSha256Class::GetHex( void )
  {
   string result  = "" ;
 /*
   result += UintToHex(h0);
   result += UintToHex(h1);
   result += UintToHex(h2);
   result += UintToHex(h3);
   result += UintToHex(h4);
   result += UintToHex(h5);
   result += UintToHex(h6);
   result += UintToHex(h7);
*/
   result += StringFormat ( "%.2x" ,h0);
   result += StringFormat ( "%.2x" ,h1);
   result += StringFormat ( "%.2x" ,h2);
   result += StringFormat ( "%.2x" ,h3);
   result += StringFormat ( "%.2x" ,h4);
   result += StringFormat ( "%.2x" ,h5);
   result += StringFormat ( "%.2x" ,h6);
   result += StringFormat ( "%.2x" ,h7);

   return (result);
  }
```

Sorry to bother you!

![Stanislav Korotky](https://c.mql5.com/avatar/2010/10/4CA7CFA0-1F0C.jpg)

**[Stanislav Korotky](https://www.mql5.com/en/users/marketeer)**
\|
2 Jul 2025 at 11:29

There is an implementation of [SHA256, SHA384, SHA512 in the codebase](https://www.mql5.com/en/code/21065), it works in MQL5 as well.


![Introduction to MQL5 (Part 11): A Beginner's Guide to Working with Built-in Indicators in MQL5 (II)](https://c.mql5.com/2/112/Introduction_to_MQL5_Part_10___LOGO.png)[Introduction to MQL5 (Part 11): A Beginner's Guide to Working with Built-in Indicators in MQL5 (II)](https://www.mql5.com/en/articles/16956)

Discover how to develop an Expert Advisor (EA) in MQL5 using multiple indicators like RSI, MA, and Stochastic Oscillator to detect hidden bullish and bearish divergences. Learn to implement effective risk management and automate trades with detailed examples and fully commented source code for educational purposes!

![Adaptive Social Behavior Optimization (ASBO): Two-phase evolution](https://c.mql5.com/2/85/Adaptive_Social_Behavior_Optimization__Part_2__LOGO.png)[Adaptive Social Behavior Optimization (ASBO): Two-phase evolution](https://www.mql5.com/en/articles/15329)

We continue dwelling on the topic of social behavior of living organisms and its impact on the development of a new mathematical model - ASBO (Adaptive Social Behavior Optimization). We will dive into the two-phase evolution, test the algorithm and draw conclusions. Just as in nature a group of living organisms join their efforts to survive, ASBO uses principles of collective behavior to solve complex optimization problems.

![Developing a Calendar-Based News Event Breakout Expert Advisor in MQL5](https://c.mql5.com/2/107/News_logo.png)[Developing a Calendar-Based News Event Breakout Expert Advisor in MQL5](https://www.mql5.com/en/articles/16752)

Volatility tends to peak around high-impact news events, creating significant breakout opportunities. In this article, we will outline the implementation process of a calendar-based breakout strategy. We'll cover everything from creating a class to interpret and store calendar data, developing realistic backtests using this data, and finally, implementing execution code for live trading.

![The Liquidity Grab Trading Strategy](https://c.mql5.com/2/110/The_Liquidity_Grab_Trading_Strategy__2__LOGO.png)[The Liquidity Grab Trading Strategy](https://www.mql5.com/en/articles/16518)

The liquidity grab trading strategy is a key component of Smart Money Concepts (SMC), which seeks to identify and exploit the actions of institutional players in the market. It involves targeting areas of high liquidity, such as support or resistance zones, where large orders can trigger price movements before the market resumes its trend. This article explains the concept of liquidity grab in detail and outlines the development process of the liquidity grab trading strategy Expert Advisor in MQL5.

[![](https://www.mql5.com/ff/si/s2n3m9ymjh52n07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F523%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dchoose.signals%26utm_content%3Dsubscribe.signal%26utm_campaign%3D0622.MQL5.com.Internal&a=fyznzyduwsltgnhlftytumasbfgbwlqw&s=91bc0eca8f132d3df7d14cdb1baebac753aef179403d60dc83856af55a4d6769&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=daalbxeoifpltzsqqcaqgkbsvrzfbael&ssn=1769092087812254813&ssn_dr=0&ssn_sr=0&fv_date=1769092087&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F16357&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Implementing%20the%20SHA-256%20Cryptographic%20Algorithm%20from%20Scratch%20in%20MQL5%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176909208738966751&fz_uniq=5049154267807327734&sv=2552)

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