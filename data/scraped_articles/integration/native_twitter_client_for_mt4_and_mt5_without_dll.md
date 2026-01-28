---
title: Native Twitter Client for MT4 and MT5 without DLL
url: https://www.mql5.com/en/articles/8270
categories: Integration
relevance_score: 3
scraped_at: 2026-01-23T21:13:59.453300
---

[![](https://www.mql5.com/ff/sh/9nb0c8df2rmwfn89z2/01.png) MetaTrader VPS vs regular cloud hosting services8 reasons why our solution is the best option for automated tradingRead](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/450486&a=dgmsfszgoedimaicrqqmagvqzpwuxkur&s=c59e3617ccf44fd54d4c50a03b44fd689ff7507b8fe4990c83772cc5419e627d&uid=&ref=https://www.mql5.com/en/articles/8270&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5071676178659486822)

MetaTrader 5 / Examples


### Introduction

Twitter provides free platform for anyone to post anything on their site. It can be as valuable as financial tips or as valueless as any prominent person can be in expressing her/his thoughts. Since this article primary focus on the media instead of its contents, let's get started.

Please sign-up on Twitter to indulge yourself with a bunch of tokens required to access Twitter API.

These tokens can be quite confusing at first, simply because there are a lot of them with similar names. Basically you need following tokens to be able to use Twitter API:

- customer\_token
- customer\_token\_secret
- access\_token
- access\_token\_secret

Note:

If you're familiar with public and private keys used for digital signing then you're on the right side.

The customer\_token and customer\_token\_secret are public and private keys to identify your "Twitter app". A Twitter app is, simply said, an identification of your service and/or access utilizing Twitter API.

The access\_token and access\_token\_secret are public and private keys to identify you as "Twitter user", in Twitter's term it is called as " **user auth"** or "user context". Based on this access\_token Twitter API can identify who is accessing it.

There exists another so called **bearer\_token** which allow "anonymous" access using Twitter API. This method of access is called as " **app auth**" or "app context". Without "user context" some Twitter APIs are not accessible which are well documented on [Twitter API reference](https://www.mql5.com/go?link=https://developer.twitter.com/en/docs/api-reference-index "https://developer.twitter.com/en/docs/api-reference-index")

For those who can code in other programming languages might find these [Twitter Libraries](https://www.mql5.com/go?link=https://developer.twitter.com/en/docs/tools-and-libraries "https://developer.twitter.com/en/docs/developer-utilities/twitter-libraries") are useful for reference. They are great resources that provide great insight into implementation details which are sometime not obvious from simply reading API documentation only.

### Twitter API and authorization

We focus on using the above mentioned tokens:

- customer\_token
- customer\_token\_secret
- access\_token
- access\_token\_secret

There are plenty of guides and/or tutorials on [YouTube](https://www.youtube.com/results?search_query=tweeter+tokens "https://www.youtube.com/results?search_query=tweeter+tokens") on how to get these tokens.

[OAuth](https://en.wikipedia.org/wiki/OAuth "https://en.wikipedia.org/wiki/OAuth") is well-accepted and widely-used standard for authentication and authorization of web based API, which is Twitter API is also using.

Simply said, the OAuth is digital signature, a method to sign digital content so that any attempt to manipulate the content shall invalidate it.

To verify the content and its signature correctly, specific methods and process of creating that content shall be followed precisely.

Again, Twitter API documentation has done a great job in documenting this whole process.

- [Creating a signature](https://www.mql5.com/go?link=https://developer.twitter.com/en/docs/authentication/oauth-1-0a/creating-a-signature "https://developer.twitter.com/en/docs/authentication/oauth-1-0a/creating-a-signature")
- [Authorizing a request](https://www.mql5.com/go?link=https://developer.twitter.com/en/docs/authentication/oauth-1-0a/authorizing-a-request "https://developer.twitter.com/en/docs/basics/authentication/oauth-1-0a/authorizing-a-request")

To keep this process simple, Twitter API requires a [specific method of encoding](https://www.mql5.com/go?link=https://developer.twitter.com/en/docs/authentication/oauth-1-0a/percent-encoding-parameters "https://developer.twitter.com/en/docs/basics/authentication/oauth-1-0a/percent-encoding-parameters") requests over HTTP. This method will be described in the next chapter.

### URL encoding and parameters sorting

To assure absolute correctness of digitally signed content, the content itself shall be well defined. For that purpose Twitter API (to be precise, the OAuth method) requires HTTP POST and/or   HTTP GET parameters to go through well defined steps before they are digitally signed.

- [URL encoding](https://www.mql5.com/go?link=https://developer.twitter.com/en/docs/authentication/oauth-1-0a/percent-encoding-parameters "https://developer.twitter.com/en/docs/basics/authentication/oauth-1-0a/percent-encoding-parameters")
- [Alphabetically sorted parameters](https://www.mql5.com/go?link=https://developer.twitter.com/en/docs/authentication/oauth-1-0a/creating-a-signature "https://developer.twitter.com/en/docs/authentication/oauth-1-0a/creating-a-signature")


It is imperative to encode HTTP POST / HTTP GET parameters as follows:

- All characters shall be "percent encoded (%XX)", except: alphanumeric (0-9, A-Z, a-z) and these special characters ( ‘-‘, ‘.’, ‘\_’, ‘~’ )
- The hex value of the "percent encoding" shall use capital letter (0-9 A B C D E F)

Please note in the [reference documentation](https://www.mql5.com/go?link=https://developer.twitter.com/en/docs/authentication/oauth-1-0a/percent-encoding-parameters "https://developer.twitter.com/en/docs/basics/authentication/oauth-1-0a/percent-encoding-parameters") and [this documentation too](https://www.mql5.com/go?link=https://developer.twitter.com/en/docs/authentication/oauth-1-0a/creating-a-signature "https://developer.twitter.com/en/docs/authentication/oauth-1-0a/creating-a-signature") regarding the "%" and "+" characters which shall be encoded correctly as well.

Please also pay attention regarding sorting of the parameters as quoted here from the [reference documentation](https://www.mql5.com/go?link=https://developer.twitter.com/en/docs/authentication/oauth-1-0a/creating-a-signature%23f1 "https://developer.twitter.com/en/docs/authentication/oauth-1-0a/creating-a-signature#f1"):

- The OAuth spec says to sort lexicographically, which is the default alphabetical sort for many libraries.
- In the case of two parameters with the same encoded key, the OAuth spec says to continue sorting based on value. However, Twitter does not accept duplicate keys in API requests


A simple implementation of this requirement is as follow:

```
string hex(int i)
  {
   static string h="0123456789ABCDEF";
   string ret="";
   int a = i % 16;
   int b = (i-a)/16;
   if(b>15)
      StringConcatenate(ret,ret,hex(b),StringSubstr(h,a,1));
   else
      StringConcatenate(ret,ret,StringSubstr(h,b,1),StringSubstr(h,a,1));
   return (ret);
  }

string URLEncode(string toCode)
  {
   int max=StringLen(toCode);

   string RetStr="";
   for(int i=0; i<max; i++)
     {
      string c = StringSubstr(toCode,i,1);
      ushort asc = StringGetCharacter(c, 0);

      if((asc >= '0' && asc <= '9')
         || (asc >= 'a' && asc <= 'z')
         || (asc >= 'A' && asc <= 'Z')
         || (asc == '-')
         || (asc == '.')
         || (asc == '_')
         || (asc == '~'))
         StringAdd(RetStr,c);
      else
        {
         StringConcatenate(RetStr,RetStr,"%",hex(asc));
        }
     }
   return (RetStr);
  }

string arrayEncode(string &array[][2])
  {
   string ret="";
   string key,val;
   int l=ArrayRange(array,0);
   for(int i=0; i<l; i++)
     {
      key = URLEncode(array[i,0]);
      val = URLEncode(array[i,1]);
      StringConcatenate(ret,ret,key,"=",val);
      if(i+1<l)
         StringConcatenate(ret,ret,"&");
     }
   return (ret);
  }

void sortParam(string&arr[][2])
  {
   string k1, k2;
   string v1, v2;
   int n = ArrayRange(arr,0);

// bubble sort
   int i, j;
   for(i = 0; i < n-1; i++)
     {
      // Last i elements are already in place
      for(j = 0; j < n-i-1; j++)
        {
         int x = j+1;
         k1 = arr[j][0];
         k2 = arr[x][0];
         if(k1 > k2)
           {
            // swap values
            v1 = arr[j][1];
            v2 = arr[x][1];
            arr[j][1] = v2;
            arr[x][1] = v1;
            // swap keys
            arr[j][0] = k2;
            arr[x][0] = k1;
           }
        }
     }
  }

void addParam(string key,string val,string&array[][2])
  {
   int x=ArrayRange(array,0);
   if(ArrayResize(array,x+1)>-1)
     {
      array[x][0]=key;
      array[x][1]=val;
     }
  }
```

An example on using the above functions is as follow:

```
   string params[][2];

   addParam("oauth_callback", "oob", params);
   addParam("oauth_consumer_key", consumer_key, params);

   sortParam(params);
```

Please note:

For simplification the parameters sorting is incomplete, it does not consider multiple same keys parameters. You might want to improve it in case you're using parameters with same keys, i.e radio buttons, check boxes on your html form.

### HMAC-SHA1 as easy as possible

Another hurdle before we can create the OAuth signature is the lack of HMAC-SHA1 native support in MQL. It turns out that MQL [CryptEncode()](https://www.mql5.com/en/docs/common/cryptencode) is of little use here, since it supports only building SHA1-HASH, hence the flag: CRYPT\_HASH\_SHA1.

So, let's code our own HMAC-SHA1 with help of [CryptEncode()](https://www.mql5.com/en/docs/common/cryptencode)

```
string hmac_sha1(string smsg, string skey, uchar &dstbuf[])
  {
// HMAC as described on:
// https://en.wikipedia.org/wiki/HMAC
//
   uint n;
   uint BLOCKSIZE=64;
   uchar key[];
   uchar msg[];
   uchar i_s[];
   uchar o_s[];
   uchar i_sha1[];
   uchar keybuf[];
   uchar i_key_pad[];
   uchar o_key_pad[];
   string s = "";

   if((uint)StringLen(skey)>BLOCKSIZE)
     {
      uchar tmpkey[];
      StringToCharArray(skey,tmpkey,0,StringLen(skey));
      CryptEncode(CRYPT_HASH_SHA1, tmpkey, keybuf, key);
      n=(uint)ArraySize(key);
     }
   else
      n=(uint)StringToCharArray(skey,key,0,StringLen(skey));

   if(n<BLOCKSIZE)
     {
      ArrayResize(key,BLOCKSIZE);
      ArrayFill(key,n,BLOCKSIZE-n,0);
     }

   ArrayCopy(i_key_pad,key);
   for(uint i=0; i<BLOCKSIZE; i++)
      i_key_pad[i]=key[i]^(uchar)0x36;

   ArrayCopy(o_key_pad,key);
   for(uint i=0; i<BLOCKSIZE; i++)
      o_key_pad[i]=key[i]^(uchar)0x5c;

   n=(uint)StringToCharArray(smsg,msg,0,StringLen(smsg));
   ArrayResize(i_s,BLOCKSIZE+n);
   ArrayCopy(i_s,i_key_pad);
   ArrayCopy(i_s,msg,BLOCKSIZE);

   CryptEncode(CRYPT_HASH_SHA1, i_s, keybuf, i_sha1);
   ArrayResize(o_s,BLOCKSIZE+ArraySize(i_sha1));
   ArrayCopy(o_s,o_key_pad);
   ArrayCopy(o_s,i_sha1,BLOCKSIZE);
   CryptEncode(CRYPT_HASH_SHA1, o_s, keybuf, dstbuf);

   for(int i=0; i < ArraySize(dstbuf); i++)
      StringConcatenate(s, s, StringFormat("%02x",(dstbuf[i])));

   return s;
  }
```

To verify its correctness we can compare with the hash created on [Twitter API documentation](https://developer.twitter.com/en/docs/authentication/oauth-1-0a/creating-a-signature#f1):

```
   uchar hashbuf[];
   string base_string = "POST&https%3A%2F%2Fapi.twitter.com%2F1.1%2Fstatuses%2Fupdate.json&include_entities%3Dtrue%26oauth_consumer_key%3Dxvz1evFS4wEEPTGEFPHBog%26oauth_nonce%3DkYjzVBB8Y0ZFabxSWbWovY3uYSQ2pTgmZeNu2VS4cg%26oauth_signature_method%3DHMAC-SHA1%26oauth_timestamp%3D1318622958%26oauth_token%3D370773112-GmHxMAgYyLbNEtIKZeRNFsMKPR9EyMZeS9weJAEb%26oauth_version%3D1.0%26status%3DHello%2520Ladies%2520%252B%2520Gentlemen%252C%2520a%2520signed%2520OAuth%2520request%2521";
   string signing_key = "kAcSOqF21Fu85e7zjz7ZN2U4ZRhfV3WpwPAoE3Z7kBw&LswwdoUaIvS8ltyTt5jkRh4J50vUPVVHtR2YPi5kE";
   string hash = hmac_sha1(base_string, signing_key, hashbuf);

   Print(hash); // 842b5299887e88760212a056ac4ec2ee1626b549

   uchar not_use[];
   uchar base64buf[];
   CryptEncode(CRYPT_BASE64, hashbuf, not_use, base64buf);
   string base64 = CharArrayToString(base64buf);

   Print(base64); // hCtSmYh+iHYCEqBWrE7C7hYmtUk=
```

### WebRequest to the rescue

Thanks to [WebRequest()](https://www.mql5.com/en/docs/network/webrequest) it is now easily to access any REST API over web without using any external DLL.

Following code will simplify accessing Twitter API using [WebRequest()](https://www.mql5.com/en/docs/network/webrequest)

```
#define WEBREQUEST_TIMEOUT 5000
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
string SendRequest(string method, string url, string headers="", string params="", int timeout=WEBREQUEST_TIMEOUT)
  {
   char data[];
   char result[];
   string resp_headers;
   ResetLastError();
   StringToCharArray(params, data);
   ArrayResize(data, ArraySize(data)-1);
   int res = WebRequest(method, url, headers, timeout, data, result, resp_headers);
   if(res != -1)
     {
      string resp = CharArrayToString(result);
      if(verbose)
        {
         Print("***");
         Print("Data:");
         Print(CharArrayToString(data));
         Print("Resp Headers:");
         Print(resp_headers);
         Print("Resp:");
         Print("***");
         Print(resp);
        }
      return resp;
     }
   else
     {
      int err = GetLastError();
      PrintFormat("* WebRequest error: %d (%d)", res, err);
      if(verbose)
        {
         Print("***");
         Print("Data:");
         Print(CharArrayToString(data));
        }
      if (err == 4014)
      {
         string msg = "* PLEASE allow https://api.twitter.com in WebRequest listed URL";
         Print(msg);
      }
     }
   return "";
  }
```

### PLEASE READ THE DOCUMENTATION OF WEBREQUEST().

QUOTE:

To use the WebRequest() function, add the addresses of the required servers in the list of allowed URLs in the "Expert Advisors" tab of the "Options" window. Server port is automatically selected on the basis of the specified protocol - 80 for "http://" and 443 for "https://".

### Helper functions to build Twitter REST API

Below some helper functions useful in building Twitter API signature.

```
string getNonce()
  {
   const string alnum = "abcdef0123456789";
   char base[];
   StringToCharArray(alnum, base);
   int x, len = StringLen(alnum);
   char res[32];
   for(int i=0; i<32; i++)
     {
      x = MathRand() % len;
      res[i] = base[x];
     }
   return CharArrayToString(res);
  }

string getBase(string&params[][2], string url, string method="POST")
  {
   string s = method;
   StringAdd(s, "&");
   StringAdd(s, URLEncode(url));
   StringAdd(s, "&");
   bool first = true;
   int x=ArrayRange(params,0);
   for(int i=0; i<x; i++)
     {
      if(first)
         first = false;
      else
         StringAdd(s, "%26");  // URLEncode("&")
      StringAdd(s, URLEncode(params[i][0]));
      StringAdd(s, "%3D"); // URLEncode("=")
      StringAdd(s, URLEncode(params[i][1]));
     }
   return s;
  }

string getQuery(string&params[][2], string url = "")
  {
   string key;
   string s = url;
   string sep = "";
   if(StringLen(s) > 0)
     {
      if(StringFind(s, "?") < 0)
        {
         sep = "?";
        }
     }
   bool first = true;
   int x=ArrayRange(params,0);
   for(int i=0; i<x; i++)
     {
      key = params[i][0];
      if(StringFind(key, "oauth_")==0)
         continue;
      if(first)
        {
         first = false;
         StringAdd(s, sep);
        }
      else
         StringAdd(s, "&");
      StringAdd(s, params[i][0]);
      StringAdd(s, "=");
      StringAdd(s, params[i][1]);
     }
   return s;
  }

string getOauth(string&params[][2])
  {
   string key;
   string s = "OAuth ";
   bool first = true;
   int x=ArrayRange(params,0);
   for(int i=0; i<x; i++)
     {
      key = params[i][0];
      if(StringFind(key, "oauth_")!=0)
         continue;
      if(first)
         first = false;
      else
         StringAdd(s, ", ");
      StringAdd(s, URLEncode(key));
      StringAdd(s, "=\"");
      StringAdd(s, URLEncode(params[i][1]));
      StringAdd(s, "\"");
     }
   return s;
  }
```

### Sample script

Now we are ready to send our first Twitter API request.

```
void verifyCredentials()
  {

   string _api_key = consumer_key;
   string _api_secret = consumer_secret;
   string _token = access_token;
   string _secret = access_secret;

   string url = "https://api.twitter.com/1.1/account/verify_credentials.json";

   string params[][2];
   addParam("oauth_consumer_key", _api_key, params);
   string oauth_nonce = getNonce();
   addParam("oauth_nonce", oauth_nonce, params);
   addParam("oauth_signature_method", "HMAC-SHA1", params);

   string oauth_timestamp = IntegerToString(TimeGMT());
   addParam("oauth_timestamp", oauth_timestamp, params);

   addParam("oauth_token", _token, params);
   addParam("oauth_version", "1.0", params);

   sortParam(params);

   string query = getQuery(params, url);
   string base = getBase(params, url, "GET");
   uchar buf[];
   string key = URLEncode(_api_secret);
   StringAdd(key, "&");
   StringAdd(key, URLEncode(_secret));

   uchar hashbuf[], base64buf[], nokey[];
   string hash = hmac_sha1(base, key, hashbuf);
   CryptEncode(CRYPT_BASE64, hashbuf, nokey, base64buf);
   string base64 = CharArrayToString(base64buf);

   addParam("oauth_signature", base64, params);
   sortParam(params);
   string o = getOauth(params);

   string headers = "Host:api.twitter.com\r\nContent-Encoding: identity\r\nConnection: close\r\n";
   StringAdd(headers, "Authorization: ");
   StringAdd(headers, o);
   StringAdd(headers, "\r\n\r\n");

   string resp = SendRequest("GET", query, headers);
   Print(resp);

   // if everything works well, we shall receive JSON-response similar to the following
   // NOTE: Identity has been altered to protect the innocent.
   // {"id":122,"id_str":"122","name":"xxx","screen_name":"xxx123","location":"","description":"", ...
  }
```

### A sample Twitter client on a MT5 chart

The following picture shows a Twitter client displaying Tweets of an Indonesian news channel. I'm preparing a followup article with more Twitter API implementation, which I'm hoping to publish it as soon as possible.

![Twitter client on MT5 chart](https://c.mql5.com/2/40/tweets.png)

**Figure 1. Tweets displayed on chart**

Another screenshot showing tweet posted from MT5 terminal.

[![Tweet from MT5 terminal](https://c.mql5.com/2/40/post__1.png)](https://c.mql5.com/2/40/post.png "https://c.mql5.com/2/40/post.png")

**Figure 2. A tweet posted from MT5 terminal**

### Future enhancement

The above described method works well as it is, albeit it is far from complete to cover all Twitter API. It is left as good exercise for those who want to dive deeper into Twitter API. Some more details on posting media, including chart screenshots will be described on next following article.

You might want to build a TwitterAPI class or even a general OAuth client class.

### A note to 3-legged authorization and PIN based authorization

You might want to dive deeper into the so called [3-legged authorization](https://www.mql5.com/go?link=https://developer.twitter.com/en/docs/basics/authentication/oauth-1-0a/obtaining-user-access-tokens "https://developer.twitter.com/en/docs/basics/authentication/oauth-1-0a/obtaining-user-access-tokens") and also the [PIN based authorization](https://www.mql5.com/go?link=https://developer.twitter.com/en/docs/basics/authentication/oauth-1-0a/pin-based-oauth "https://developer.twitter.com/en/docs/basics/authentication/oauth-1-0a/pin-based-oauth").

It is beyond of this article to describe them, please feel free to contact me in case you need further guidance.

### Conclusion

Twitter is a well accepted platform that provides anyone to publish almost anything.

With this article and its code described above, I'm hoping to contribute to MQL community with the result of my little adventure in understanding OAuth in accessing Twitter API .

Looking forward for encouraging and improving feedbacks. Feel free to use all code provide in any free and/or commercial projects of yours.

I want to thank[Grzegorz Korycki](https://www.mql5.com/en/users/angreeee "Grzegorz Korycki (angreeee)") for providing this library code ( [SHA256, SHA384 and SHA512 + HMAC - library for MetaTrader 4](https://www.mql5.com/en/code/21065)) which has inspired me in creating the HMAC-SHA1 function.

Let's tweet for fun and profit!

Enjoy.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/8270.zip "Download all attachments in the single ZIP archive")

[tweeter\_mql.mq5](https://www.mql5.com/en/articles/download/8270/tweeter_mql.mq5 "Download tweeter_mql.mq5")(9.67 KB)

[urlencode.mqh](https://www.mql5.com/en/articles/download/8270/urlencode.mqh "Download urlencode.mqh")(3.55 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Native Twitter Client: Part 2](https://www.mql5.com/en/articles/8318)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/348564)**
(8)


![IraSveta](https://c.mql5.com/avatar/2020/8/5F3A8BF0-F0A5.gif)

**[IraSveta](https://www.mql5.com/en/users/irasveta)**
\|
8 Sep 2020 at 16:20

[Twitter](https://www.mql5.com/en/articles/8270 "Article: Writing a Twitter Client for MetaTrader 4 and MetaTrader 5 without DLL ") is a useful thing. Thank you


![Joao Luiz Sa Marchioro](https://c.mql5.com/avatar/2017/11/5A1389EC-103A.JPG)

**[Joao Luiz Sa Marchioro](https://www.mql5.com/en/users/joaoluiz_sa)**
\|
29 Sep 2020 at 18:43

Excellent article! Very good! I'll be able to implement an idea based on it.

Simple code and very well written, easy to understand. Thank you very much!

![Ronei Toporcov](https://c.mql5.com/avatar/avatar_na2.png)

**[Ronei Toporcov](https://www.mql5.com/en/users/roneit)**
\|
4 Mar 2021 at 16:36

Please correct the message:

"Please Allow tweeter ..."

for :

[twitter](https://www.mql5.com/en/articles/8270 "Article: Writing a Twitter Client for MetaTrader 4 and MetaTrader 5 without DLL ")

Thanks for the help with your work.

![danielt132](https://c.mql5.com/avatar/avatar_na2.png)

**[danielt132](https://www.mql5.com/en/users/danielt132)**
\|
23 Jun 2021 at 18:08

Thank you alot


![](https://c.mql5.com/avatar/avatar_na2.png)

**\[Deleted\]**
\|
29 Nov 2021 at 03:10

Thank you for sharing this information, and code, greatly appreciated 👍😊


![Timeseries in DoEasy library (part 42): Abstract indicator buffer object class](https://c.mql5.com/2/38/MQL5-avatar-doeasy-library__7.png)[Timeseries in DoEasy library (part 42): Abstract indicator buffer object class](https://www.mql5.com/en/articles/7821)

In this article, we start the development of the indicator buffer classes for the DoEasy library. We will create the base class of the abstract buffer which is to be used as a foundation for the development of different class types of indicator buffers.

![Timeseries in DoEasy library (part 41): Sample multi-symbol multi-period indicator](https://c.mql5.com/2/38/MQL5-avatar-doeasy-library__6.png)[Timeseries in DoEasy library (part 41): Sample multi-symbol multi-period indicator](https://www.mql5.com/en/articles/7804)

In the article, we will consider a sample multi-symbol multi-period indicator using the timeseries classes of the DoEasy library displaying the chart of a selected currency pair on a selected timeframe as candles in a subwindow. I am going to modify the library classes a bit and create a separate file for storing enumerations for program inputs and selecting a compilation language.

![Native Twitter Client: Part 2](https://c.mql5.com/2/40/mql_twitter__1.png)[Native Twitter Client: Part 2](https://www.mql5.com/en/articles/8318)

A Twitter client implemented as MQL class to allow you to send tweets with photos. All you need is to include a single self contained include file and off you go to tweet all your wonderful charts and signals.

![Continuous Walk-Forward Optimization (Part 7): Binding Auto Optimizer's logical part with graphics and controlling graphics from the program](https://c.mql5.com/2/38/MQL5-avatar-continuous_optimization__4.png)[Continuous Walk-Forward Optimization (Part 7): Binding Auto Optimizer's logical part with graphics and controlling graphics from the program](https://www.mql5.com/en/articles/7747)

This article describes the connection of the graphical part of the auto optimizer program with its logical part. It considers the optimization launch process, from a button click to task redirection to the optimization manager.

[![](https://www.mql5.com/ff/sh/6xjc81sb5f2g45z9z2/01.png)Follow MQL5.community on social mediaWe publish the best technical materials from experts – free from advertising and irrelevant contentLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=yexgeaiatphxecqagtoxizolvboismyb&s=4e531fd1f983c26570e2dac7588b735354f2f9e0aea561427c030e4a1d2f060b&uid=&ref=https://www.mql5.com/en/articles/8270&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5071676178659486822)

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