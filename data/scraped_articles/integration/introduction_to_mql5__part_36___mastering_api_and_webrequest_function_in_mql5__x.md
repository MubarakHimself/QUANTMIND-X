---
title: Introduction to MQL5 (Part 36): Mastering API and WebRequest Function in MQL5 (X)
url: https://www.mql5.com/en/articles/20938
categories: Integration, Expert Advisors
relevance_score: 6
scraped_at: 2026-01-23T11:53:52.502274
---

[![](https://www.mql5.com/ff/sh/rvgkjnsrvj1mzh89z2/01.png)Best VPS for tradersTwo-click launch from MetaTrader, minimum ping to broker, 15 USD/monthLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=wpjhvzsogglsviotmypjoyhhtuxlrzhi&s=aa6c5782a1658c2f617954d478dea9989a27ae26ecabc09d0ab1204277fdf8e3&uid=&ref=https://www.mql5.com/en/articles/20938&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5062797377452026143)

MetaTrader 5 / Integration


### Introduction

Welcome back to Part 36 of the Introduction to MQL5 series! In [Part 31](https://www.mql5.com/en/articles/20546) of this series, we discussed how to communicate with external platforms like Binance, but the communication was direct and did not involve exchanging any sensitive information. We’ll get into the security aspect of API connectivity in this article. We go over key subjects including signatures, HMAC (Hash-based Message Authentication Code), and SHA256 hashing, which are used to authenticate requests, confirm request integrity, and stop tampering. When making critical API calls, like placing orders or managing trades, these security measures are extremely crucial.

It's crucial to note that the theory and building blocks are the only topics that will be covered in this article. Sensitive information will not yet be sent in live queries. We will combine these concepts to securely connect with a server in the upcoming article. You will have a basic firm grasp of how HMAC, SHA256, and signatures combine to create the foundation for secure MQL5 API calls by the end of this section.

### **What is a signature?**

In APIs, a signature functions similarly to a special seal that is appended to each request. The server can verify two things with this seal: the request is truly from you, and nothing was altered before its arrival. Put differently, the signature informs the server that "this request is safe to trust." Without it, anyone may impersonate your requests, which is risky when managing delicate transactions like placing orders or making trades.

To verify requests, signatures are crucial. The authenticity of a request must be verified by servers. This is accomplished by employing cryptographic methods such as HMAC-SHA256 to create a signature using the request data and a secret key. This signature is sent with the request to the server, where the same computation is carried out. The request's legitimacy is confirmed by a matched result. Think of it as a signed letter to help you grasp it. Someone can be certain that a handwritten letter bearing your signature at the bottom was written by you. The receiver would be able to tell the letter wasn't genuine if someone attempted to fake it without your signature. Similar to this, in APIs, the signature guarantees that your request is identified as originating from you and hasn't been altered by anybody else while it was being transmitted.

### **Understanding HMAC-SHA256**

Before discussing HMAC, it's critical to grasp what MAC stands for. Message Authentication Code is referred to as MAC. A message and a secret key are used to create a MAC, which is a little piece of data. Its primary goal is to simultaneously demonstrate that the communication originated from the intended sender and that it hasn't been altered. The server uses the same secret key to recalculate the MAC after receiving a message. The communication is recognized as genuine if the results match. MACs come in various varieties. While some are created using hash functions, others are created using block ciphers. They all have the same purpose, message authentication and integrity checking, regardless of the type. HMAC is one of the most popular and secure methods among these.

It is used to provide a signature that verifies an API request is legitimate and hasn't been tampered with. When sending sensitive requests, like placing orders, platforms like Binance frequently employ it. We will not go into all of HMAC-SHA256's internal mathematical details in this article. Rather, we will concentrate solely on the components you must understand to properly create and sign MQL5 API queries. Here, practical understanding is more important than mastery of cryptography. This mechanism's fundamental components are a message authentication method and the SHA256 hashing algorithm. The end product is a signature, which is a fixed-length value. The WebRequest method is then used to send the request after this signature has been added to the request URL.

A fixed 256-bit result is produced by this cryptographic hash method, which takes inputs of different lengths. Whatever the input, the output size stays the same. The outcome is deterministic, guaranteeing repeatable and predictable outputs for the same input data even though it appears random. The avalanche effect is a significant feature of SHA256. A single character modification in the input will result in an entirely different hash. SHA256 is therefore particularly helpful for identifying data changes. SHA256 alone, however, cannot establish the sender of the data. A message can be hashed by anyone. Herein lies the role of HMAC.

HMAC incorporates a secret key into the hashing process, extending SHA256. This private key is only shared by you and the server, such as between Binance and your MT5 software. A signature that can only be produced by someone with the right secret key is produced when the message and secret key are combined and hashed using SHA256. This implies that an attacker cannot produce a legitimate signature without the secret key, even if they can view the message being delivered. There is no need to get into the specifics of how the internal hashing procedures operate for our use case. The most important lesson is that if you use the same message and secret key, you will always get the same signature; if you change either one, you will get a different signature.

In reality, the signed "message" is typically a query string composed of request elements like timestamp. Your API secret key and this string are sent to the HMAC-SHA256 process. The signature is what is produced. An additional parameter to the request URL is the produced signature. The server recalculates the signature using the same cryptographic technique and a copy of your secret key after receiving the request. The request is approved if the freshly generated signature matches the one you supplied; if not, it is denied. Although this procedure is founded on sophisticated cryptographic principles, regular MQL5 API usage does not require those specifics. This article does not go into core mechanics like padding, block sizes, or hash rounds; instead, it concentrates on what you need to create signatures effectively and understand their function.

This targeted method guarantees that you can confidently go to the next section of the series, where we will actually sign actual API requests using HMAC-SHA256 and submit them safely using MQL5's WebRequest function. Two essential components are needed to generate a signature for the Binance API. The first is a timestamp, which indicates the precise time the request was made and is always updating. The second is your secret key, which Binance gives you and which you must always keep confidential. Even if all other parameters stay the same, the timestamp guarantees that each request is distinct.

Binance does not immediately obtain your private key when you submit a request. Instead, your system generates a signature using HMAC-SHA256 utilizing that secret key and the request data, including the date. After that, the request is sent to Binance with this signature attached. Binance uses a copy of your secret key that they already possess to carry out the same computation on their end. The request is approved if the signature they produce matches the one you sent.

![Figure 1. HMAC](https://c.mql5.com/2/190/figure_1.png)

This procedure makes sure that the request hasn't been changed while it's on the network. The verification would fail if even a minor modification to the request parameters resulted in an entirely different signature. This is crucial for security, particularly when handling delicate tasks like placing orders.

![Figure 2. Minor Modification](https://c.mql5.com/2/190/figure_2.png)

Why the request never contains the secret key itself is another crucial element. It might be intercepted if it were sent across the network. Only a derived signature is sent when you use a cryptographic signing technique; the secret key stays securely on your computer. Therefore, intercepting the request does not provide an attacker with sufficient information to counterfeit legitimate requests or reproduce the secret key.

Analogy:

In a city full of mail carriers, picture yourself delivering a crucial letter to a reliable friend. Make sure your friend is aware that the letter truly originated from you and that no one altered its contents. You have a secret wax seal that only you and your pal can create to accomplish this. Similar to your private key, this seal is always in your possession and cannot be duplicated by anyone else. Imagine now that each letter you send has the precise time it was written imprinted on it. This timestamp is crucial since the letters will appear different even if you send the identical message twice due to different timings. Without the timestamp, someone might attempt to deceive your friend by sending an old letter again. Like a fingerprint for that particular instant, the timestamp guarantees that every letter is distinct.

You put the timestamp and your secret wax seal on the letter before sending it. This combination gives the letter a unique mark that verifies its authenticity. Similar to the HMAC-SHA256 signature, the mark is specific to that message at that precise moment because it is created using your secret wax seal and the time. Your acquaintance checks the mark on the letter by using a copy of your wax seal. To make sure the letter is up to date, they also look at the timestamp. Your acquaintance will know the letter is authentic and unaltered if the mark is correct and the time makes sense. Otherwise, they reject it right away.

The secret wax seal never leaves your desk thanks to this system's elegance. The letter is sent with only the mark or signature. Your seal cannot be replicated or a fresh letter forged, even if a cunning mail carrier spots the letter and the mark. Your secret seal and the timestamp work together to keep every letter safe, authentic, and distinct. The secret key and a timestamp are used to create a signature when submitting requests to the Binance API. The timestamp stops the same request from being used again, and the secret key verifies that it originates from you. The legitimacy and integrity of the request are then verified by Binance by validating the signature and timestamp.

### **Creating API Signatures in MQL5**

The process of creating API signatures using HMAC SHA256 in MQL5 is where all the previously studied theory begins to come together in a useful way. The objective is now straightforward: to create a signature that demonstrates to an external service, like an exchange API, that the request is actually coming from you and hasn't been changed while en route. This process takes place solely on your local computer in MQL5, before the request is delivered via the WebRequest function.

First, it's important to know exactly what we are signing. The majority of APIs, including Binance, do not generate the signature using random input. Rather, the request parameters are represented by a string that is used to generate it. Values like a timestamp, order information, quantities, or other endpoint-required parameters are frequently included in this string. Because it is constantly changing, the timestamp is very significant. This ensures that every request is unique even if every other parameter remains unchanged. There is a serious security risk if a valid request is logged without a timestamp and replayed later.

This parameter string gets merged with your secret key once it is ready. The API provider gives you the secret key, which should never be shared or communicated over the network. This secret key is only used in MQL5 as an input for the HMAC SHA256 procedure. The actual key never leaves your computer. Rather, it functions as an unseen component that affects the finished product. The identical signature for the same message can exclusively be produced by someone who is aware of this secret key.

A unique hashing procedure is applied to the message and secret key in MQL5. The main idea is the outcome, so you don't need to worry about the intricate computations. The result is a fixed-length string that is entirely consistent despite appearing random. Your signature is this string. You will receive the same signature if you sign the same mail again using the same secret key. The signature will be entirely different if you use a different key or change just one character.

The signature is created and then added to the request, typically as an extra URL parameter. The server uses its stored copy of your secret key and the supplied parameters to carry out the same signature calculation when the request reaches the server. A matching signature indicates to the server that the data has not changed during network transmission and that the request was made by the authorized key owner. The CryptEncode function in MQL5 is used to create signatures and hashes. It can generate hashes like SHA256 or HMAC-based signatures and is one of the platform's integrated cryptography tools. In essence, CryptEncode takes incoming data, transforms it using a cryptographic method, adds a key if desired, and outputs a binary value. We utilize that binary value as the signature or hash.

Converting the Message and Secret Key into Byte Data

Converting the message and the secret key into a character-based format that cryptographic operations can understand is the first stage in establishing a signature. Normal text strings are not directly compatible with hashing and HMAC operations in MQL5. Rather, they work with unprocessed byte data. This means that before any cryptographic processing can start, the human-readable message and the secret key must be converted into a low-level representation. The data you wish to protect is typically the message being transformed, such as a query string composed of request parameters. To ensure that each request is distinct, this frequently contains a timestamp. The private value that the API provider gives you is known as the secret key. To ensure that the same input consistently generates the same signature on both your system and the server side, both values must be treated carefully and translated consistently.

In this conversion procedure, a standard text encoding is used to convert each letter in the message and the secret key into its corresponding byte value. Because it is widely supported and guarantees that the data is interpreted consistently across many systems, UTF-8 is frequently utilized. Two byte sequences, one for the message and the other for the secret key, are the outcome of this stage. Because cryptographic techniques depend on precise byte sequences, this translation is essential. A signature would be entirely different if there was even a slight variation in encoding or representation. We guarantee the accuracy of the subsequent cryptographic process and the server's ability to replicate the identical signature for verification by transforming both the message and the secret key into byte form in a controlled and predictable manner.

The message and secret key are in the proper format to be utilized in the HMAC SHA256 procedure once this step is finished. All subsequent operations operate on these byte-level representations, enabling the creation of a secure signature without ever disclosing the secret key.

Example:

```
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
//---

   string message = "71772777";
   string key = "ABCDER2y8VAzqopxzLVEhVYABCDEV2AxIYLueComud3aSEez8Z8fvgHPZTXABCDE";

   uchar uMsg[], uKey[];

   StringToCharArray(message, uMsg, 0, StringLen(message), CP_UTF8);
   StringToCharArray(key, uKey, 0, StringLen(key), CP_UTF8);
  }
```

Explanation:

Defining the message and the secret key in a readable format is the first stage. The data that needs to be signed is represented by the message; in actual API usage, this is frequently a timestamp or a mix of request parameters. Only your system and the server should be aware of the secret key, which is the private value provided by the platform. Both of them are now only available as plain text, which is useful for people but unsuitable for cryptographic operations.

Setting up storage capable of storing raw byte data is the next stage. Since text is an abstract representation, cryptographic techniques do not directly deal with it. Rather, they need bytes. For this reason, specific byte containers are set up to store the secret key and the message's transformed versions. During the conversion process, these containers will be filled from their initial empty state. The message is then transformed from text to its byte representation. Consistency is ensured by using UTF-8 encoding for this conversion. The same characters will always convert into the same byte sequence thanks to UTF-8. This is crucial since a slight variation in bytes could result in an entirely different signature.

The secret key is then subjected to the same conversion procedure. The key's characters are converted into bytes and kept in separate containers. The secret key has not yet been shared and is still confidential. It has only been produced in a way that cryptographic functions may comprehend. The message and secret key byte arrays are now stored in the system when both conversions are finished. Signing and hashing have not yet taken place. Preparation is the only purpose of this step. In order for the HMAC SHA256 procedure to generate a legitimate and verifiable signature in the subsequent stage, it guarantees that the data is in the proper format.

Analogy:

Let's say the secret key and the message are two handwritten notes. The information you wish to communicate, such as a timestamp or request specifics, is contained in a single note. Only you and the recipient are aware of the unique code on the other note. Both notes are now written in plain handwriting, which is readable by humans but unreliable for security checks by machines. The next step is to prepare two empty envelopes. These envelopes are merely receptacles where the notes will be placed in a machine-readable format; they are not yet intended for mailing. The envelopes are now empty and awaiting filling. Imagine now that you take the message from the initial handwritten note and convert every word into a rigid, standard language that you and the recipient have decided upon.

This is similar to turning handwriting into typewritten text using a certain alphabet in which each letter has the same appearance. This ensures that the message will always be understood in the same manner and without ambiguity. You then follow the same procedure with the secret key. You copy it in the same machine-friendly, standardized language and put it in its own envelope, but it stays private and never leaves your desk. The trick is just being ready for the next step; it is not shared. You then have two envelopes: one with the translated secret key and the other containing the translated message. Both are just prepared to be utilized to establish a secure signature in the following step; nothing has been sealed or secured yet.

Preparing the Inner and Outer Pads for Signing

The objective at this point is to convert the secret key into two unique working forms that will be utilized to safely combine the key and the message. These two types are frequently referred to as the outer pad and the inner pad. They are not sent anyplace and are not arbitrary. They are only present throughout the signing process. The process can be compared to creating two distinct sealing stamps using the same master key. Although both stamps originate from the same secret, they are slightly modified to fulfill distinct functions during the verification process. The process starts with one stamp and ends with the other. An additional layer of security is added, and the signature is far more difficult to forge when two separate pads are used rather than just one.

Here, the fixed size is crucial. Blocks of a certain length are used by HMAC; with SHA256, this length is 64 bytes. The technique guarantees consistency and security by making the key fit into this precise size and then blending it in two separate ways. The procedure always follows a predictable structure, regardless of how long or short the initial secret key was.

Example:

```
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
//---

   string message = "71772777";
   string key = "ABCDER2y8VAzqopxzLVEhVYABCDEV2AxIYLueComud3aSEez8Z8fvgHPZTXABCDE";

   uchar uMsg[], uKey[];

   StringToCharArray(message, uMsg, 0, StringLen(message), CP_UTF8);
   StringToCharArray(key, uKey, 0, StringLen(key), CP_UTF8);

   int blockSize = 64;

   uchar ipad[], opad[];
   ArrayResize(ipad, blockSize);
   ArrayResize(opad, blockSize);

   for(int i = 0; i < blockSize; i++)
     {
      ipad[i] = uKey[i] ^ 0x36;
      opad[i] = uKey[i] ^ 0x5C;
     }

  }
```

Explanation:

Setting the block size for the SHA-256 hashing procedure is the first thing we do in this stage. Data is processed by SHA-256 in 64-byte fixed-size blocks. The length of the inner pad (ipad) and outer pad (opad) utilized in HMAC is determined by this block size. We guarantee that both pads are the proper length for the HMAC computation by specifying the block size. The inner and outer pads are then stored in two arrays, ipad and opad. These arrays are adjusted to correspond with a 64-byte block size. A single byte, obtained from the secret key and altered to get it ready for the hashing operation, will be stored in each array place.

In this stage, each byte of the secret key is processed one at a time using a loop. To prepare each byte for the inner and outer padding, it is coupled with a fixed constant. This guarantees that, without altering the original key, the key is appropriately altered for the hashing procedure. This phase essentially scrambles the key for the hashing process by altering each byte in a consistent and predictable manner. To distinguish the inner and outer pads, two unique constants are selected. Because the inner and outer pads are employed in different hashing phases that collectively result in a secure signature, this distinction is crucial.

By the time this stage is finished, we have two prepared pads. One for the inner section and one for the outside part. Both are based on the secret key but have been subtly altered to prepare them for hashing. The message and inner pad are first concatenated and hashed. The final signature is then created by combining that hash with the outer pad. This guarantees that the signature is exclusive to your key and cannot be altered.

Analogy:

Let's say you are using a secure mailing system to send a particularly important document. This system has a rigorous restriction that requires all documents to be sealed within identical-sized envelopes before they can be processed. The block size of 64 is that fixed envelope size. Before proceeding, everything must be ready to fit that precise size, regardless of how lengthy or short your message is. Now consider the secret key as a unique piece of evidence that the document truly originates from you. You must use that strip to create two unique protection layers before sealing the document. The inner pad and the outside pad are these two layers.

They can be compared to two specially designed security sleeves that would encircle the message at various verification phases. Each sleeve is meticulously made to match the needed size precisely because they must be the same size as the envelope. The customizing phase follows. A portion of the secret key is combined with a predetermined pattern for each spot in these sleeves. The outer sleeve has a different pattern than the inner sleeve. This mixing procedure is similar to employing two distinct stamping patterns to emboss your distinctive seal into both sleeves. The end product becomes something that only you and the recipient can duplicate since the patterns are set and the seal is confidential.

Generating the Signature

The actual signature is created in this stage using the inner and outer pads that we previously produced. The inner hash and the outer hash are the two primary hashing steps in the procedure. The message we wish to sign is first joined with the inner pad, and this combined data is then hashed. The inner hash is the outcome of this initial hash. The outside pad and the inner hash are then combined. The final hash is then generated by hashing this new combination once more. The original message and the secret key are uniquely represented by this final value, which functions as the secure signature.

Lastly, the hash we generated is converted from a string of bytes into a readable format, typically hexadecimal, just the way a secret code is converted into a written message. This message is used as the API request's signature. It is used by the server to verify that the message actually originated from the secret key holder and that it hasn't been altered during transmission. This stage creates a safe "digital seal" that is ready for use by locking the message and key together.

Example:

```
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
//---

   string message = "71772777";
   string key = "ABCDER2y8VAzqopxzLVEhVYABCDEV2AxIYLueComud3aSEez8Z8fvgHPZTXABCDE";

   uchar uMsg[], uKey[];

   StringToCharArray(message, uMsg, 0, StringLen(message), CP_UTF8);
   StringToCharArray(key, uKey, 0, StringLen(key), CP_UTF8);

   int blockSize = 64;

   uchar ipad[], opad[];
   ArrayResize(ipad, blockSize);
   ArrayResize(opad, blockSize);

   for(int i = 0; i < blockSize; i++)
     {
      ipad[i] = uKey[i] ^ 0x36;
      opad[i] = uKey[i] ^ 0x5C;
     }

   uchar innerData[], innerHash[];
   ArrayCopy(innerData, ipad);
   ArrayCopy(innerData, uMsg, blockSize);
   CryptEncode(CRYPT_HASH_SHA256, innerData, uKey, innerHash);

   uchar outerData[], finalHash[];
   ArrayCopy(outerData, opad);
   ArrayCopy(outerData, innerHash, blockSize);
   CryptEncode(CRYPT_HASH_SHA256, outerData, uKey, finalHash);

   string signature = "";
   for(int i = 0; i < ArraySize(finalHash); i++)
      signature += StringFormat("%02x", finalHash[i]);

   Print(signature);

  }
```

Explanation:

The inner data and the outcome of the inner hash are stored in two arrays that are created in the first section. The inner pad (iPad) and the message you wish to sign are combined to create the inner data. The message is appended right after the inner pad is copied into the new array using the ArrayCopy method. This guarantees that the message comes after the inner pad in the proper byte order for the inner hash input. Because even one incorrect byte will result in an entirely different hash, proper ordering is essential. The program uses the CryptEncode function, the SHA256 algorithm, and the secret key to calculate the inner hash once the inner data is ready. This inner hash functions as a secure fingerprint of the message and inner pad together and is connected to your secret key. It is the first hashing layer of HMAC-SHA256 and ensures that the final signature is dependent on both the message and your secret key.

Preparing the outside data is the next phase. This procedure is comparable to the inner data preparation: an array is formed, and ArrayCopy is used to copy the outside pad (opad) into it. The inner hash is then added to the same array following the outer pad. The input for the final hash computation is made up of the outer pad and inner hash. Once more, for HMAC to function properly, the precise sequence must be maintained. The final hash is created by applying the CryptEncode function once more using SHA256 and the secret key after the outer data has been prepared. The final hash is the signature of HMAC-SHA256. It is a special fixed-length byte array that safely stores the secret key and the message. The same process can then be used by the server to confirm that the message was created by a person who is aware of the secret key and has not been altered.

You can't send the final hash straight over the internet; it's like a stack of raw materials. The program begins with an empty container known as the signature to render them usable. Every byte from the hash is converted into a two-character hexadecimal fragment before being dropped into the container. The secure signature is represented as a single, tidy hexadecimal string that you may attach to your API request after repeating this process for each byte. Lastly, the hexadecimal signature is printed by the software. API requests can now include this signature, guaranteeing the request's legitimacy. Your secret key never needs to leave your computer since only someone with the right secret key can produce a matching signature. Because the signature cannot be reverse-engineered to reveal the secret key, this offers security even if the network is intercepted.

Analogy:

Imagine the procedure as if you were delivering a pal a locked treasure chest with a unique seal. Preparing the chest's interior is the initial step. After inserting your message, you lock it with a pad that can only be used with your secret key. The message and the inner pad are combined to tie the contents to your secret in a way that cannot be faked, much like when you create the inner hash. After that, you secure the chest by figuring out a special lock pattern using the message and the inside pad. The inner hash is represented by this lock pattern. Similar to a complicated lock, even a minor alteration to the message or the pad will produce a different pattern, making it impossible for anyone to secretly tamper with the chest.

The exterior portion of the chest is prepared once the inner lock is in place. You insert the inner lock after adding a cover that also comes from your secret key as an additional layer of security. Similar to creating the exterior data, this double-layered strategy makes sure that someone cannot duplicate the lock without the secret key, even if they view the inner half. Sealing the outer lock, which is the same as calculating the outer hash, is the last stage. This results in the signature, a distinctive mark that verifies the chest originated from you and that its contents have not been altered. You can safely share the signature with your friend. Your friend can examine the locks with their copy of your secret key after they receive the chest. They can be certain that the chest is genuine and unaltered if everything matches.

Lastly, you transform the lock pattern into a format that everyone can use, such as a unique code on a label, because it is a string of letters and numbers that is difficult to read directly. This is the signature's hexadecimal string. You include this code with your API request. Similar to your friend examining the chest's seal to ensure it truly originated from you, the server will verify authenticity by comparing this code to its own computation.

### **Conclusion**

This article taught you how to create an API signature in MQL5. You learned how to use inner and outer pads, transform messages and secret keys into hashed values, and produce a signature that can properly validate requests without disclosing sensitive information. We also witnessed the necessity of timestamps and secret keys to guarantee that every request is distinct and impervious to manipulation. We will put this into effect in the upcoming post by making a signature and utilizing it to ask Binance for sensitive data.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/20938.zip "Download all attachments in the single ZIP archive")

[Project\_26\_API\_SIGNATURE.mq5](https://www.mql5.com/en/articles/download/20938/Project_26_API_SIGNATURE.mq5 "Download Project_26_API_SIGNATURE.mq5")(1.79 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Introduction to MQL5 (Part 35): Mastering API and WebRequest Function in MQL5 (IX)](https://www.mql5.com/en/articles/20859)
- [Introduction to MQL5 (Part 34): Mastering API and WebRequest Function in MQL5 (VIII)](https://www.mql5.com/en/articles/20802)
- [Introduction to MQL5 (Part 33): Mastering API and WebRequest Function in MQL5 (VII)](https://www.mql5.com/en/articles/20700)
- [Introduction to MQL5 (Part 32): Mastering API and WebRequest Function in MQL5 (VI)](https://www.mql5.com/en/articles/20591)
- [Introduction to MQL5 (Part 31): Mastering API and WebRequest Function in MQL5 (V)](https://www.mql5.com/en/articles/20546)
- [Introduction to MQL5 (Part 30): Mastering API and WebRequest Function in MQL5 (IV)](https://www.mql5.com/en/articles/20425)

**[Go to discussion](https://www.mql5.com/en/forum/503955)**

![Central Force Optimization (CFO) algorithm](https://c.mql5.com/2/127/Central_Force_Optimization___LOGO.png)[Central Force Optimization (CFO) algorithm](https://www.mql5.com/en/articles/17167)

The article presents the Central Force Optimization (CFO) algorithm inspired by the laws of gravity. It explores how principles of physical attraction can solve optimization problems where "heavier" solutions attract less successful counterparts.

![MQL5 Trading Tools (Part 12): Enhancing the Correlation Matrix Dashboard with Interactivity](https://c.mql5.com/2/191/20962-mql5-trading-tools-part-12-logo.png)[MQL5 Trading Tools (Part 12): Enhancing the Correlation Matrix Dashboard with Interactivity](https://www.mql5.com/en/articles/20962)

In this article, we enhance the correlation matrix dashboard in MQL5 with interactive features like panel dragging, minimizing/maximizing, hover effects on buttons and timeframes, and mouse event handling for improved user experience. We add sorting of symbols by average correlation strength in ascending/descending modes, toggle between correlation and p-value views, and incorporate light/dark theme switching with dynamic color updates.

![Forex Arbitrage Trading: Relationship Assessment Panel](https://c.mql5.com/2/125/Forex_Arbitrage_Trading_Relationship_Assessment_Dashboard___LOGO.png)[Forex Arbitrage Trading: Relationship Assessment Panel](https://www.mql5.com/en/articles/17422)

Let's consider creating an arbitrage panel in MQl5. How to get fair exchange rates on Forex in different ways? Create an indicator to obtain deviations of market prices from fair exchange rates, as well as to assess the benefits of arbitrage ways of exchanging one currency for another (as in triangular arbitrage).

![Larry Williams Market Secrets (Part 6): Measuring Volatility Breakouts Using Market Swings](https://c.mql5.com/2/191/20862-larry-williams-market-secrets-logo.png)[Larry Williams Market Secrets (Part 6): Measuring Volatility Breakouts Using Market Swings](https://www.mql5.com/en/articles/20862)

This article demonstrates how to design and implement a Larry Williams volatility breakout Expert Advisor in MQL5, covering swing-range measurement, entry-level projection, risk-based position sizing, and backtesting on real market data.

[![](https://www.mql5.com/ff/sh/rvgkjnsrvj1mzh89z2/01.png)Best VPS for tradersTwo-click launch from MetaTrader, minimum ping to broker, 15 USD/monthLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=wpjhvzsogglsviotmypjoyhhtuxlrzhi&s=aa6c5782a1658c2f617954d478dea9989a27ae26ecabc09d0ab1204277fdf8e3&uid=&ref=https://www.mql5.com/en/articles/20938&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5062797377452026143)

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