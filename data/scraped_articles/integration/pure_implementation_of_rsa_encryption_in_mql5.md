---
title: Pure implementation of RSA encryption in MQL5
url: https://www.mql5.com/en/articles/20273
categories: Integration, Expert Advisors, Machine Learning
relevance_score: 6
scraped_at: 2026-01-23T11:54:41.857809
---

[![](https://www.mql5.com/ff/sh/zf7a2k61x98jzh89z2/01.png)Speed up your tradingUse our high-speed VPS for MetaTrader 4 and 5Learn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=qtrrsuiwuicrscmckjynyanztbditglq&s=c617dc80d90cfd3783ec1345eec2b419b281f10fec6eac77b3218984ac337259&uid=&ref=https://www.mql5.com/en/articles/20273&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5062808402633075024)

MetaTrader 5 / Integration


Imagine a common situation: your Expert Advisor needs to send signals, keys, login details, or other important data to a server. You might use HTTP or even HTTPS and consider the connection reasonably secure. But sooner or later you discover that the real vulnerability isn’t in the connection at all — it’s inside the code itself.

In MQL5, there are moments when you simply have no choice but to embed sensitive values directly into the EA. Algorithm parameters, private keys, passwords — all of it can end up compiled into the file. And even though the code is compiled and obfuscated, someone with enough skill can still extract those values. This creates a significant security issue: your transport may be encrypted, yet the Expert Advisor itself remains a potential leak because it has to _store_ secrets internally. At the same time, using DLLs for encryption is forbidden for Market products. MQL5 doesn’t provide RSA out of the box, and lightweight “encryption” tricks like XOR offer no real protection. Many developers get stuck at this point: they need a way to secure communication and protect their keys, but the available tools are either restricted or simply not strong enough.

This article aims to address these limitations. It explains how to ensure that your Expert Advisor (EA) no longer stores any critical data internally, but instead retrieves it securely from a trusted server in encrypted form. You will also learn how to implement proper hybrid encryption—using RSA for exchanging keys and AES for fast messaging—entirely within MQL5, without relying on DLLs or external libraries.

### Introduction To RSA

The name “RSA” comes from the initials of Ron Rivest, Adi Shamir, and Leonard Adleman, who introduced the algorithm in 1977 while working at MIT. Their work was the first practical implementation of a public-key cryptosystem based on one-way mathematical functions, turning what had been a theoretical idea into a usable technology. Although the concept of public-key cryptography had been proposed earlier by Diffie and Hellman in 1976, it was the RSA construction that provided a concrete mechanism for secure encryption and digital signatures.

At its core, RSA relies on the mathematical difficulty of factoring large composite numbers — a problem that remains computationally infeasible for sufficiently large key sizes. This property makes RSA a foundational technology in modern cryptography. For decades, it has been used to secure web traffic, authenticate digital documents, exchange keys, and protect sensitive data across countless platforms and protocols.

Today, RSA remains one of the most widely adopted asymmetric encryption algorithms, forming a critical part of standards such as SSL/TLS, PGP, SSH, and many secure communication systems. Despite the emergence of elliptic-curve cryptography (ECC) and other modern alternatives, RSA continues to be valued for its simplicity, robustness, and long-established security model.

![](https://c.mql5.com/2/185/RSA-Working-Diagram.png)

### The Math Behind It

RSA is a simple and elegant algorithm:

Here’s how it works: you encrypt your message with a public key, and then only the matching private key can decrypt it and recover the original message. The core of RSA is based on the formula shown below. If you’re wondering who discovered it, the credit goes to the brilliant mathematician Leonhard Euler. Euler's theorem (also known as the Fermat–Euler theorem or Euler's totient theorem) is the foundation of the RSA system. You can read more about it in [this article](https://www.mql5.com/go?link=https://www.geeksforgeeks.org/maths/eulers-theorem/ "https://www.geeksforgeeks.org/maths/eulers-theorem/").

![Main formula](https://c.mql5.com/2/185/main.png)

Here’s what those letters mean:

- **m**: is your message (just written as a number).
- **e**and **n:**together make up the public key.
- **d:** is the private key.
- **n:**is called the modulus.

Let us take a closer look at the main equation. Even with limited background in mathematics, it is possible to see that certain mathematical operations are applied to the message (m) and that the result of the operation recovers the original message. This provides the insight to split the equation into two asymmetric parts using the distributive property of modulo operations. The message _m_ can be encrypted by raising it to the power of _e_modulo _n_, which results in ciphertext,_Z_ (encrypted message).

![Encryption formula](https://c.mql5.com/2/185/encryption.png)

And encrypted message _Z_can be decrypted using the private key _d_:

![Decryption formula](https://c.mql5.com/2/185/decryption.png)

Let's demonstrate this concept with a simple numerical example using m = 2, e = 3, d = 3 and n = 15. Using the encryption formula we have:

Z = (m ^ e) mod n = 2^3 mod 15 = 8

The encrypted value can now be decrypted using the private key d = 3:

m = (Z^d) mod n = 8^3 mod 15 = 2

We can even make a small MQL5 program to implement the above numerical example:

```
//+------------------------------------------------------------------+
//| Simple RSA Numerical Example (for educational purposes only)     |
//+------------------------------------------------------------------+

// Fast modular exponentiation: computes (base^exp) % mod
int ModPow(int base, int exp, int mod)
{
   long result = 1;
   long b = base % mod;

   while(exp > 0)
   {
      if(exp & 1)
         result = (result * b) % mod;

      b = (b * b) % mod;
      exp >>= 1;
   }
   return (int)result;
}

// RSA "encryption": c = m^e % n
int EncryptRSA(int m, int e, int n)
{
   return ModPow(m, e, n);
}

// RSA "decryption": m = c^d % n
int DecryptRSA(int c, int d, int n)
{
   return ModPow(c, d, n);
}

//+------------------------------------------------------------------+
//| Example based on small numbers (not secure!)                     |
//+------------------------------------------------------------------+
void OnStart()
{
   int m = 2;     // message
   int e = 3;     // public exponent
   int d = 3;     // private exponent
   int n = 15;    // modulus

   Print("Original message: ", m);

   int encrypted = EncryptRSA(m, e, n);
   Print("Encrypted: ", encrypted, "   // 2^3 % 15 = 8");

   int decrypted = DecryptRSA(encrypted, d, n);
   Print("Decrypted: ", decrypted, "   // 8^3 % 15 = 2");

   // Verifying the main RSA concept:
   // (m^e)^d % n = m
   int check = ModPow(ModPow(m, e, n), d, n);
   Print("Check (m^e)^d % n = ", check);
}
```

Tips and Tricks:

First, you’ve got to turn text into numbers before you can encrypt it. That’s where ASCII or Unicode comes in—each letter, space, or symbol becomes a number. Take “Hello world!” for example. You can write it out as one huge number: decimal → 22405534230753963835153736737 or in hex →  0x48656c6c6f20776f726c6421. However, the modulus can’t be a small number such as 15; such a value is far too small for real messages. The RSA modulus must be larger than the numeric representation of the message; otherwise, the encryption mathematics will not function correctly. RSA encryption is computationally intensive. Encrypting large numbers with large exponents, such as 65537, requires many multiplications. Without optimization, the CPU may become heavily loaded.

Regarding security, basic RSA encryption alone is insufficient. It is vulnerable to replay attacks and other potential exploits. To address this, padding is required. For example, PKCS#1 padding adds random bytes before the actual message, enhancing security by preventing predictable ciphertext patterns.

```
[0x00][0x02][random bytes][0x00][message]
```

With padding, sending the same message multiple times produces different encrypted outputs. This makes it significantly more difficult for an attacker to compromise the encryption. Without proper padding and correct key handling, RSA encryption becomes vulnerable to several attack classes. In the following step, we demonstrate how to generate a proper key pair in practice.

Key generation using OpenSSL:

First, verify that OpenSSL is installed on your operating system. RSA keys can be generated using the following OpenSSL command (Windows environment):

```
//bash
openssl genpkey -algorithm RSA -out private_key.pem -pkeyopt rsa_keygen_bits:2048
```

Public certificates contain the public exponent and modulus, while private keys contain the private exponent. The following command displays both the public and private key components:

```
//bash
openssl rsa -in private_key.pem -text -noout
openssl rsa -in public_key.pem -pubin -text -noout
```

After running these commands, you will see output similar to the image shown below.

![generated private and public keys ](https://c.mql5.com/2/182/key.png)

This is how browsers handle secure connections with SSL or TLS. In summary, RSA enables secure communication even over insecure channels. For production use, it is important to apply proper padding and choose a sufficiently large key—2048 bits or more. Tools such as OpenSSL simplify key generation and management. You can also find other online tools for RSA encryption and decryption or even key generation and use them for development but, never trust them for production. Although Windows provides RSA functionality through CryptoAPI, calling external DLLs adds complexity and potential security risks, particularly for Expert Advisors (EAs). When clients use your EA and see that it requires DLL access, they may hesitate or even lose trust in your product.

### Algorithm Design (Step-by-Step)

The encryption process consists of several stages, and to make it easier to follow, the code explanation has been divided into three parts: Part 1 covers the core concepts and initialization of the class requirements. Part 2 introduces the necessary arithmetic functions for big integers. And Part 3 presents the final implementation of the RSA encryption process.

**Part 1**

Creating the RSA Class:

The main class _MQL5\_RSA_ encapsulates the complete encryption workflow and includes functionalities for data format conversions, such as Base64 and HexToByte. It implements big number arithmetic, PKCS#1 v1.5 padding to introduce randomness into the message, and an optional debug mode for tracing operations.

We start by defining an RSA class that stores the modulus, exponent, and helper methods. It also seeds the random number generator once, which is important because PKCS#1 v1.5 padding requires nonzero random bytes.

```
public class MQL5_RSA
{
private:
   uchar modulus[];  //will store the RSA modulus (n) as a big-endian byte array.
   int   exponent;   // holds the public exponent (usually 65537).
   bool  debugMode;  //toggles console logging.

public:
   MQL5_RSA(bool debug=false)
   {
      debugMode = debug;
      MathSrand((int)TimeLocal());   // seed RNG once
   }
};
```

Initialization:

Load the modulus (public key) as a hex string, convert to bytes, normalize, and store the public exponent (usually 65537).

```
void Init(string modulusHex, int e)
{
   ArrayResize(modulus, 0);  // clear previous values

   if(debugMode)
      PrintFormat("Init: modulus hex len=%d, e=%d", StringLen(modulusHex), e);

   HexToBytes(modulusHex, modulus);  // convert hex → bytes
   Normalize(modulus);               // remove leading zeros (important!)
   exponent = e;

   if(debugMode)
      PrintFormat("Init completed: modulus bytes=%d", ArraySize(modulus));
}
```

Fixing and Writing HexToBytes():

This function is crucial because the original MetaQuotes code had a bug; it appends character codes, not hex pairs. We write a correct and demanded version:

```
void HexToBytes(string hex, uchar &out[])
{
   string clean = "";
   int L = StringLen(hex);

   // keep only valid hex characters
   for(int i = 0; i < L; i++)
   {
      ushort c = StringGetCharacter(hex, i);
      if((c >= '0' && c <= '9') ||
         (c >= 'A' && c <= 'F') ||
         (c >= 'a' && c <= 'f'))
         clean += StringSubstr(hex, i, 1);
   }

   // ensure even number of hex chars (pad if necessary)
   if(StringLen(clean) % 2 == 1)
      clean = "0" + clean;

   int n = StringLen(clean) / 2;
   ArrayResize(out, n);

   for(int i = 0; i < n; i++)
   {
      ushort c1 = StringGetCharacter(clean, i*2);
      ushort c2 = StringGetCharacter(clean, i*2+1);

      int high = HexNibble(c1);   // convert 1 hex char -> 0..15
      int low  = HexNibble(c2);

      out[i] = (uchar)((high << 4) | low); // combine two nibbles → byte
   }
}
```

Normalizing Big Integers:

RSA numbers are big-endian arrays. This means that index 0 is the most significant byte. We never want leading zero bytes, so we trim them:

```
void Normalize(uchar &a[])
{
   int size = ArraySize(a);
   int leading = 0;

   // count leading zero bytes
   while(leading < size && a[leading] == 0)
      leading++;

   if(leading == 0)
      return;  // already normalized

   int newSize = size - leading;
   if(newSize <= 0)
   {
      ArrayResize(a, 0);
      return;
   }

   uchar temp[];
   ArrayResize(temp, newSize);

   // copy only the non-zero part
   ArrayCopy(temp, a, 0, leading, newSize);

   ArrayResize(a, newSize);
   ArrayCopy(a, temp);
}
```

At this point, the RSA class provides the essential core functionality. The class can store the debug state, accept and normalize modulus, correctly convert hexadecimal values to bytes, remove leading zeros, and store the exponent value. This foundation is essential before implementing big-integer arithmetic operations such as subtraction, comparison, and multiplication, followed by modular arithmetic and PKCS#1 padding. With these functionalities in place, the RSA encryption process can be executed.

**Part 2**

Building the Big-Integer Engine: _The Nuts and Bolts of RSA_

Next, we build the big-integer arithmetic system required for RSA. MQL5 does not provide a native big-integer type, so RSA must simulate large integers using arrays of bytes. This section develops the mathematical backbone that supports PKCS#1 padding and modular exponentiation. The following functions will be implemented:

1. Compare();
2. SubtractInPlace();
3. LeftShiftBytes();
4. Multiply();
5. Mod();
6. MulMod();

These functions collectively emulate real big-integer arithmetic operations.

1.Compare()—Determining Which Big Integer Is Larger:

Before performing subtraction, division, or modulo operations, it is necessary to be able to compare two numbers.

```
a > b
  or
a < b
  or
a == b
```

```
int Compare(const uchar &a_in[], const uchar &b_in[])
{
   // copy to avoid modifying original arrays
   uchar a[];
   ArrayCopy(a, a_in);
   uchar b[];
   ArrayCopy(b, b_in);

   Normalize(a);
   Normalize(b);

   int na = ArraySize(a), nb = ArraySize(b);

   // first compare lengths
   if(na > nb) return 1;
   if(na < nb) return -1;

   // lengths equal → compare byte by byte
   for(int i=0; i<na; i++)
   {
      if(a[i] > b[i]) return 1;
      if(a[i] < b[i]) return -1;
   }

   return 0; // equal
}
```

2\. SubtractInPlace()—Big-endian Subtraction

This function performs a = a − b. The operation is performed byte by byte, starting from the right, since the last byte represents the least significant portion of the number. Borrowing must be handled manually.This function is extensively used in Mod() during the long-division reduction phase, and it also appears in intermediate multiplication steps.

```
void SubtractInPlace(uchar &a[], const uchar &b[])
{
   // assume a >= b (Compare() must ensure this)
   int na = ArraySize(a);
   int nb = ArraySize(b);
   int borrow = 0;

   for(int i = 0; i < na; i++)
   {
      int ai = (int)a[na - 1 - i];               // rightmost byte of a
      int bi = (i < nb) ? (int)b[nb - 1 - i] : 0; // rightmost byte of b

      int diff = ai - bi - borrow;

      if(diff < 0)
      {
         diff += 256;  // wrap around as unsigned byte
         borrow = 1;
      }
      else
      {
         borrow = 0;
      }

      a[na - 1 - i] = (uchar)diff;
   }

   Normalize(a); // remove leading zeros
}
```

3\. LeftShiftBytes() — Multiply by 256ⁿ:

During long-division operations, it is often necessary to shift a number to the left, effectively multiplying it by powers of 256 (i.e., shifting by whole bytes).

```
void LeftShiftBytes(const uchar &in[], int shiftBytes, uchar &out[])
{
   int n = ArraySize(in);
   if(n == 0 || shiftBytes == 0)
   {
      ArrayCopy(out, in);
      return;
   }
   ArrayResize(out, n + shiftBytes);

   // copy original bytes
   for(int i=0; i<n; i++)
      out[i] = in[i];
   // append zeros on the right (least significant side)
   for(int i=n; i<n+shiftBytes; i++)
      out[i] = 0;
}
```

During modulus calculation, the divisor may be shifted until it aligns with the dividend.

This is the core of the long-division algorithm.

4\. Multiply() — Schoolbook Multiplication:

The standard multiplication function is now implemented using the classical “schoolbook” method in below:

```
void Multiply(const uchar &a[], const uchar &b[], uchar &result[])
{
   int na = ArraySize(a);
   int nb = ArraySize(b);

   if(na==0 || nb==0)
   {
      ArrayResize(result, 0);
      return;
   }

   int nRes = na + nb;
   int temp[];
   ArrayResize(temp, nRes);
   ArrayInitialize(temp, 0);  // accumulator array (int for safety)

   // classic schoolbook multiplication
   for(int i = na - 1; i >= 0; i--)
   {
      int carry = 0;

      for(int j = nb - 1; j >= 0; j--)
      {
         int prod = (int)a[i] * (int)b[j] + temp[i + j + 1] + carry;
         temp[i + j + 1] = prod % 256;
         carry = prod / 256;
      }

      temp[i] += carry;
   }

   ArrayResize(result, nRes);
   for(int i=0; i<nRes; i++)
      result[i] = (uchar)temp[i];

   Normalize(result);
}
```

This multiplication is the slowest part of pure-MQL5 RSA implementation, but it is necessary for MulMod() and ModExp() functions. These functions will be discussed in detail later.

5\. Mod(), Big-Integer Modulo Using Long Division:

This is the single most important big-integer routine.We will implement a simplified long-division. Let us start the remainder as an empty array then for each byte of the dividend, shift remainder left, append next byte and subtract the modulus while remainder ≥ modulus. The final remainder would be our desired result.

```
bool Mod(const uchar &a_in[], const uchar &m_in[], uchar &result[])
{
   if(ArraySize(m_in) == 0) return false;

   uchar dividend[]; ArrayCopy(dividend, a_in); Normalize(dividend);
   uchar modv[];     ArrayCopy(modv, m_in);   Normalize(modv);

   // if dividend < modulus → done
   if(Compare(dividend, modv) < 0)
   {
      ArrayCopy(result, dividend);
      return true;
   }

   int m = ArraySize(dividend);
   uchar rem[];
   ArrayResize(rem, 0);

   for(int i = 0; i < m; i++)
   {
      // shift remainder left by 1 byte
      int rlen = ArraySize(rem);
      ArrayResize(rem, rlen + 1);
      rem[rlen] = dividend[i];
      Normalize(rem);

      // subtract modulus while remainder >= modulus
      while(Compare(rem, modv) >= 0)
      {
         SubtractInPlace(rem, modv);
      }
   }

   Normalize(rem);
   ArrayCopy(result, rem);
   return true;
}
```

6\. MulMod() — Multiplication Followed by Mod:

This simply glues together:

Multiply(a, b) → temp

Mod(temp, m)   → out

```
bool MulMod(const uchar &a[], const uchar &b[], const uchar &m[], uchar &out[])
{
   uchar product[];
   Multiply(a, b, product);
   return Mod(product, m, out);
}
```

**Part 3**

We now have all the necessary tools for RSA arithmetic. First, the security of the message must be enhanced by applying PKCS#1 v1.5 padding before encryption. Next, the ModExp() function is implemented to compute: result = base^exp mod n. This is the core operation of RSA that produces the ciphertext. Finally, the output can be standardized using the Base64Encode() helper, allowing the ciphertext to be displayed conveniently.

Modular Exponentiation (ModExp) —The heart of RSA:

The ModExp() function efficiently computes result = base^exp mod n for large exponents using repeated squaring and modular multiplication. It relies on MulMod() (multiply then reduce) so all intermediate values stay small enough to handle.

```
bool ModExp(const uchar &base[], int exp, const uchar &modn[], uchar &result[])
{
   if(debugMode)
      PrintFormat("ModExp: exp=%d", exp);

   // Work on a copy to avoid mutating caller arrays
   uchar baseCopy[];
   ArrayCopy(baseCopy, base);
   Normalize(baseCopy);

   // Reduce base modulo modn first: base = base % modn
   if(Compare(baseCopy, modn) >= 0)
   {
      if(!Mod(baseCopy, modn, baseCopy))
         return false; // mod failed
   }

   // Initialize result = 1 (big-int representation)
   uchar res[];
   ArrayResize(res, 1);
   res[0] = 1;
   Normalize(res);

   // basePow holds current power of base (base^(2^i))
   uchar basePow[];
   ArrayCopy(basePow, baseCopy);

   int e = exp; // local copy so we can shift it

   // Square-and-multiply loop
   while(e > 0)
   {
      // If current LSB is 1 → multiply into result
      if((e & 1) == 1)
      {
         uchar tmp[];
         if(!MulMod(res, basePow, modn, tmp))
            return false; // MulMod failed
         ArrayCopy(res, tmp); // res = (res * basePow) % modn
      }

      // Move to next bit
      e >>= 1;

      // If still bits left, square basePow: basePow = (basePow * basePow) % modn
      if(e > 0)
      {
         uchar tmp2[];
         if(!MulMod(basePow, basePow, modn, tmp2))
            return false; // MulMod failed
         ArrayCopy(basePow, tmp2);
      }
   }

   Normalize(res);
   ArrayCopy(result, res); // return result via out param
   return true;
}
```

PKCS#1 v1.5 Padding and Encryption ( EncryptPKCS1v15 ):

First, we should make the padded message, EM = 0x00 \|\| 0x02 \|\| PS \|\| 0x00 \|\| M, where PS is non-zero random bytes, and make sure the message length is smaller than k - 11 (k = modulus length in bytes) because overhead bytes are counted for final message length.

```
bool EncryptPKCS1v15(uchar &plain[], uchar &cipher[])
{
   int k = ArraySize(modulus);      // modulus size (bytes)
   int mlen = ArraySize(plain);     // message length (bytes)

   if(debugMode)
      PrintFormat("Encrypt: key bytes=%d, plain=%d, max=%d", k, mlen, k-11);

   // PKCS#1 v1.5 requires at least 11 bytes of overhead
   if(mlen > k - 11)
   {
      if(debugMode) Print("Error: data too large for key");
      return false;
   }

   // Build the encoded message EM (k bytes)
   uchar em[];
   ArrayResize(em, k);
   ArrayInitialize(em, 0);    // default everything to 0

   em[0] = 0x00;              // leading zero by spec
   em[1] = 0x02;              // block type 2 (encryption)

   int psLen = k - mlen - 3;  // length of padding string PS

   // Fill PS with non-zero random bytes
   for(int i=0; i<psLen; i++)
   {
      uchar b = 0;
      // loop until rand byte is non-zero (spec requirement)
      do
      {
         b = (uchar)(MathRand() % 256);
      }
      while(b == 0);

      em[2 + i] = b;
   }

   // Separator before message
   em[2 + psLen] = 0x00;

   // Copy message into EM at the right position
   ArrayCopy(em, plain, 3 + psLen, 0, mlen);

   if(debugMode) Print("EM constructed");

   // Modular exponentiation: C = EM^e mod n
   uchar cBig[];
   if(!ModExp(em, exponent, modulus, cBig))
   {
      if(debugMode) Print("ModExp failed");
      return false;
   }

   // Ensure ciphertext byte array has exact length k
   int clen = ArraySize(cBig);
   ArrayResize(cipher, k);
   if(clen < k)
   {
      // left-pad with zeros
      ArrayInitialize(cipher, 0);
      ArrayCopy(cipher, cBig, k - clen, 0, clen);
   }
   else
   {
      ArrayCopy(cipher, cBig); // already k bytes or longer (shouldn't be longer normally)
   }

   if(debugMode) Print("Encryption finished");
   return true;
}
```

The padding string (PS) must consist of non-zero random bytes. In this implementation, MathRand() —seeded in the constructor—is used. While this is an improvement over an unseeded random number generator, it is not cryptographically secure. Production systems should use a cryptographically secure RNG. PKCS#1 v1.5 padding is widely supported, but it has known vulnerabilities, such as padding oracle attacks. For new systems, RSAES-OAEP is preferred. The padding function returns false if the message is too long for the chosen key.

Base64 Output Helper (Base64Encode):

Ciphertexts are binary data. For logging, HTTP transport, or text-based storage, Base64 provides a convenient and standardized representation.

```
string Base64Encode(uchar &data[])
{
   string chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
   int len = ArraySize(data);
   string out = "";

   // process 3 bytes => 4 base64 chars
   for(int i = 0; i < len; i += 3)
   {
      int b0 = data[i];
      int b1 = (i + 1 < len) ? data[i + 1] : 0;
      int b2 = (i + 2 < len) ? data[i + 2] : 0;
      int val = (b0 << 16) | (b1 << 8) | b2;

      // append 4 chars, using '=' padding for missing bytes
      out += StringSubstr(chars, (val >> 18) & 0x3F, 1);
      out += StringSubstr(chars, (val >> 12) & 0x3F, 1);
      out += (i + 1 < len) ? StringSubstr(chars, (val >> 6) & 0x3F, 1) : "=";
      out += (i + 2 < len) ? StringSubstr(chars, val & 0x3F, 1) : "=";
   }

   return out;
}
```

### Putting It All Together

To maintain readability, the full source code of the RSA class is not included in here. The complete implementation is provided as the **RSA.mqh** library, available for download at the bottom of the page. The library contains all components discussed in this article, as well as a commented example demonstrating how to use the class. It can be used for rapid integration or extended and modified once the internal workings are understood. If any errors occur or clarification is needed for any part of the implementation, readers are encouraged to leave a comment below.

**Example Usages**

Raw RSA Implementation:

This simple example demonstrates how the RSA encryption could be implemented. First, create an instance of the RSA class then prepare your message (the data that you want to encrypt) as uchar\[\], where each element is one byte. Finally, call EncryptPKCS1v15 (plain, cipher) to do the encryption. The result is as a char array and stored in cipherData\[\]:

```
void OnStart()
{
   MQL5_RSA rsa;
   string modulusHex = "A1B2C3D4E5F6..."; // Your modulus in hex
   int exponent = 65537; // exponent value

   rsa.Init(modulusHex, exponent);

   string plainText = "Hello RSA!"; //Data that you want to be encrypted.
   uchar plainData[];
   StringToCharArray(plainText, plainData, 0, StringLen(plainText));

   uchar cipherData[];
   if(rsa.EncryptPKCS1v15(plainData, cipherData))
   {
      string base64Cipher = rsa.Base64Encode(cipherData);
      Print("Encrypted: ", base64Cipher);
   }
   else
   {
      Print("Encryption failed!");
   }
}
```

Using RSA + AES in a real project:

![AES-RSA hybrid encryption](https://c.mql5.com/2/186/jklh.jpg)

In real-world cryptographic systems—including secure trading platforms, web servers, and distributed APIs—RSA is rarely used to encrypt large data directly. RSA encryption is computationally intensive and limited by key size, making it unsuitable for large payloads. To achieve both high-performance and strong security, RSA is typically combined with AES in a hybrid encryption model that leverages the strengths of both algorithms.

The hybrid encryption workflow generally follows these steps:

1. Generate a random AES session key (for example, 128 or 256 bits).
2. Encrypt this AES key using the RSA public key. Only the owner of the corresponding private key can later decrypt it.
3. Encrypt the actual data payload using AES — a fast, symmetric cipher ideal for large messages or files.
4. Transmit both components together:
   - The AES-encrypted data (fast and compact).
   - The RSA-encrypted AES key (small but secure).

This approach combines the advantages of both algorithms. RSA enables secure key exchange between two parties without a pre-shared secret, while AES provides high-speed encryption for the actual data. In this model, RSA functions as the key protector, and AES functions as the data protector. Together, they form the foundation of modern secure communication protocols, such as SSL/TLS, HTTPS, and VPN tunnels.

In the context of MQL5, this strategy allows developers to secure communications between Expert Advisors, indicators, and external servers, even when using standard HTTP or socket connections. The RSA class implemented in this article can be used to encrypt the AES key, while the built-in MQL5 functions CryptEncode() and CryptDecode() handle AES encryption and decryption of the actual message.

This enables a fully self-contained security layer inside MetaTrader 5, effectively creating a lightweight version of “HTTPS over HTTP.” It can be used to protect sensitive data, such as trading commands, authentication credentials, or configuration messages, without relying on external encryption libraries or DLLs.

Step 1—Server generates RSA public key:

Use Python, Java, OpenSSL, or any backend environment to generate modulus (hex string), exponent (normally 65537), and private key (kept on your server side).

The EA will only use the public values. Do not expose the private key to the client side.

```
modulusHex = "A1B2C3…";
exponent   = 65537;
```

Step 2— EA creates an RSA instance and loads the key:

```
#include <RSA.mqh>

MQL5_RSA rsa;
rsa.Init(modulusHex, exponent);
```

Step 3 — EA prepares a request message:

Example payload can be a login request containing:EA ID, account number, and timestamp. Make a proper JSON string for your request. You can use any third-party library to make the JSON string. The snippet below demonstrates what the expected JSON string should look like:

```
string json =
   "{\"cmd\":\"login\","
   "\"account\":" + IntegerToString(AccountInfoInteger(ACCOUNT_LOGIN)) + ","
   "\"ts\":" + IntegerToString((int)TimeCurrent()) + ","
   "}";
uchar plain[];
StringToCharArray(json, plain, 0, StringLen(json));
```

Step 4 — EA encrypts the AES session key with RSA:

The AES key used for the rest of the session:

```
uchar aesKey[];
GenerateAESKey(aesKey);   // 16 random bytes (AES-128)
uchar encryptedAes[];
rsa.EncryptPKCS1v15(aesKey, encryptedAes);
string encryptedAesBase64 = rsa.Base64Encode(encryptedAes);
```

Step 5 — EA encrypts the actual data with AES:

```
uchar encryptedPayload[];
CryptEncode(CRYPT_AES128, plain, aesKey, encryptedPayload);
string payloadBase64 = CryptBase64Encode(encryptedPayload);
```

Step 6 —EA sends everything to the server:

```
string body =
   "{\"key\":\"" + encryptedAesBase64 + "\","
   + "\"data\":\"" + payloadBase64 + "\"}";

string result;
char headers[];
int status = WebRequest("POST", url, headers, 5000, body, result);
```

Step 7 — Server-side decryption (conceptual):

On the server, first perform base64-decode for the RSA-encrypted AES key, then decrypt AES key using the private RSA key. Once the AES key is recovered, it can be used to decrypt the data payload. Since the AES key is randomly generated for each session, each message remains unique and secure, even if an attacker intercepts the traffic. This mechanism effectively establishes a secure session, analogous to a simplified HTTPS tunnel.

Step 8 — Client side decrypts server response:

```
uchar responseCipher[];
CryptBase64Decode(result, responseCipher);
uchar responsePlain[];
CryptDecode(CRYPT_AES128, responseCipher, aesKey, responsePlain);

string serverReply = CharArrayToString(responsePlain);
Print("Server replied: ", serverReply);
```

By following these steps, real-world cryptography can be implemented directly within the MetaTrader 5 environment, resulting in a reliable and fully portable solution for end users.

### Conclusion

This article presented a complete and functional RSA encryption library written entirely in MQL5, demonstrating that even complex mathematical algorithms can be implemented directly within the MetaTrader 5 environment without external dependencies. Through this work, we have shown that MQL5 is not limited to trading automation—it is also capable of performing advanced computational tasks such as modular arithmetic, and public-key cryptography.

With this foundation, developers now have a practical toolkit for building secure messaging systems, protecting configuration files, and supporting secure key exchange and encrypted communication directly within MetaTrader 5. This enables Expert Advisors and indicators to communicate securely with remote services, authenticate data sources, and exchange encrypted trade commands without exposing sensitive information to the network.

The implementation also underscores an important concept: cryptography does not need to rely on external DLLs or third-party libraries to be effective. When algorithms are implemented natively in MQL5, the code remains transparent, portable, and fully controllable by the developer. This ensures both auditability and compliance with MetaQuotes' security policies, which is particularly relevant when developing commercial trading tools distributed through the Market.

For developers interested in extending this work, several directions are possible. The RSA library can be combined with digital signatures to verify message authenticity and integrity, or expanded with modern padding schemes such as OAEP for improved resistance to cryptanalysis. Likewise, integrating AES in authenticated modes like GCM or CBC-HMAC can provide both encryption and tamper detection in a single step. Such additions would bring the implementation closer to modern standards like TLS and PGP, entirely inside the MQL5 environment.

From a practical perspective, this article encourages developers to approach MQL5 not merely as a trading scripting language, but as a full-featured programming environment where security, networking, and computation coexist. By combining mathematical precision with careful design, MQL5 developers can create secure, self-contained systems capable of communicating confidently with external APIs, brokers, or cloud services.

In summary, cryptography in MQL5 is both possible and powerful. Developers now have the tools to experiment with, extend, and integrate secure communication mechanisms directly into their projects. As the trading ecosystem evolves toward more connected and data-driven architectures, the ability to handle encryption and authentication within the terminal will become increasingly valuable. With this foundation, every developer can take meaningful steps toward building safer, smarter, and more resilient trading systems.

**Authors and Programmers**

- Vahid Irak
- [Siavash Nourmohammadi](https://www.mql5.com/en/users/siavashnourmohammadi)

**References and Further Reading**

For readers interested in the mathematical foundations and modern applications of RSA and AES encryption, the following references provide reliable, in-depth explanations:

- [Wikipedia – PKCS #1: RSA Cryptography Standard](https://en.wikipedia.org/wiki/PKCS_1 "https://en.wikipedia.org/wiki/PKCS_1")

Explains the official RSA standard, including padding schemes such as PKCS#1 v1.5 and OAEP, used for encryption and signatures.

The official MQL5 documentation for CryptEncode(), CryptDecode(), and other built-in functions for AES and hashing.
- Paar, C. & Pelzl, J. (2024). _Understanding Cryptography – A Textbook for Students and Practitioners_, Chapter 7: The RSA Cryptosystem. Springer. [link](https://www.mql5.com/go?link=https://link.springer.com/chapter/10.1007/978-3-662-69007-9_7 "https://link.springer.com/chapter/10.1007/978-3-662-69007-9_7")
- Mollin, R. A. (2023). _RSA and Public-Key Cryptography_. Routledge. [link](https://www.mql5.com/go?link=https://www.routledge.com/RSA-and-Public-Key-Cryptography/Mollin/p/book/9780367395650 "https://www.routledge.com/RSA-and-Public-Key-Cryptography/Mollin/p/book/9780367395650")
- National Institute of Standards and Technology. _FIPS 197: Advanced Encryption Standard (AES)_. 2001 (updated). [link](https://www.mql5.com/go?link=https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.197-upd1.pdf "https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.197-upd1.pdf")
- Tuo, Z. (2023). “A Comparative Analysis of AES and RSA Algorithms and Their Integrated Application.” _Theoretical and Natural Science_, Vol 25, pp. 28-35. [link](https://www.mql5.com/go?link=https://doi.org/10.54254/2753-8818/25/20240893 "https://doi.org/10.54254/2753-8818/25/20240893")
- “RSA-AES Hybrid Encryption: Combining the Strengths of Symmetric & Asymmetric Algorithms.” IJRAR, 2023. [link](https://www.mql5.com/go?link=https://www.ijrar.org/papers/IJRAR23B1852.pdf "https://www.ijrar.org/papers/IJRAR23B1852.pdf")
- Ganesh, R., Khan, A. R., et al. (2025). “A Panoramic Survey of the Advanced Encryption Standard (AES)”. _International Journal of Information Security_. [link](https://www.mql5.com/go?link=https://link.springer.com/article/10.1007/s10207-025-01116-x "https://link.springer.com/article/10.1007/s10207-025-01116-x")
- [Simplified explanation of how RSA message encryption/decryption works](https://www.mql5.com/go?link=https://hereket.com/posts/rsa-algorithm/ "https://hereket.com/posts/rsa-algorithm/")

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/20273.zip "Download all attachments in the single ZIP archive")

[RSA.mqh](https://www.mql5.com/en/articles/download/20273/RSA.mqh "Download RSA.mqh")(24.79 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

**[Go to discussion](https://www.mql5.com/en/forum/502115)**

![Statistical Arbitrage Through Cointegrated Stocks (Part 9): Backtesting Portfolio Weights Updates](https://c.mql5.com/2/186/20657-statistical-arbitrage-through-logo.png)[Statistical Arbitrage Through Cointegrated Stocks (Part 9): Backtesting Portfolio Weights Updates](https://www.mql5.com/en/articles/20657)

This article describes the use of CSV files for backtesting portfolio weights updates in a mean-reversion-based strategy that uses statistical arbitrage through cointegrated stocks. It goes from feeding the database with the results of a Rolling Windows Eigenvector Comparison (RWEC) to comparing the backtest reports. In the meantime, the article details the role of each RWEC parameter and its impact in the overall backtest result, showing how the comparison of the relative drawdown can help us to further improve those parameters.

![Larry Williams Market Secrets (Part 1): Building a Swing Structure Indicator in MQL5](https://c.mql5.com/2/186/20511-larry-williams-market-secrets-logo.png)[Larry Williams Market Secrets (Part 1): Building a Swing Structure Indicator in MQL5](https://www.mql5.com/en/articles/20511)

A practical guide to building a Larry Williams–style market structure indicator in MQL5, covering buffer setup, swing-point detection, plot configuration, and how traders can apply the indicator in technical market analysis.

![Creating Custom Indicators in MQL5 (Part 2): Building a Gauge-Style RSI Display with Canvas and Needle Mechanics](https://c.mql5.com/2/186/20632-creating-custom-indicators-logo__1.png)[Creating Custom Indicators in MQL5 (Part 2): Building a Gauge-Style RSI Display with Canvas and Needle Mechanics](https://www.mql5.com/en/articles/20632)

In this article, we develop a gauge-style RSI indicator in MQL5 that visualizes Relative Strength Index values on a circular scale with a dynamic needle, color-coded ranges for overbought and oversold levels, and customizable legends. We utilize the Canvas class to draw elements like arcs, ticks, and pies, ensuring smooth updates on new RSI data.

![Creating Custom Indicators in MQL5 (Part 1): Building a Pivot-Based Trend Indicator with Canvas Gradient](https://c.mql5.com/2/186/20610-creating-custom-indicators-logo__1.png)[Creating Custom Indicators in MQL5 (Part 1): Building a Pivot-Based Trend Indicator with Canvas Gradient](https://www.mql5.com/en/articles/20610)

In this article, we create a Pivot-Based Trend Indicator in MQL5 that calculates fast and slow pivot lines over user-defined periods, detects trend directions based on price relative to these lines, and signals trend starts with arrows while optionally extending lines beyond the current bar. The indicator supports dynamic visualization with separate up/down lines in customizable colors, dotted fast lines that change color on trend shifts, and optional gradient filling between lines, using a canvas object for enhanced trend-area highlighting.

[![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/01.png)![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/02.png)Powerful analytics for traders of any levelAll the necessary trading reports for beginners and professionals](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/454106&a=muccpajyfystoakuukdobwigjejzmpqn&s=52daad60fa795e635264e6f94898f05493bca3b5124d4cca8eb7e82333c2ef12&uid=&ref=https://www.mql5.com/en/articles/20273&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5062808402633075024)

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