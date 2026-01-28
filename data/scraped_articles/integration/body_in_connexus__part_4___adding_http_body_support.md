---
title: Body in Connexus (Part 4): Adding HTTP body support
url: https://www.mql5.com/en/articles/16098
categories: Integration, Expert Advisors
relevance_score: 1
scraped_at: 2026-01-23T21:42:06.820763
---

[![](https://www.mql5.com/ff/sh/592yc11u3j4rs5z9z2/01.png)How AI helps create robots for MetaTrader 5Learn from our book "Neural Networks in Algo Trading with MQL5"Read](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/neurobook%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.neurobook%26utm_content=visit.page%26utm_campaign=neurobook.promo.04.2024&a=ghrobswocqgvhztzjldphupateyllpro&s=9929cb0b8629585b5a42fabc06c525e41f6c0ebdf3045d044a5413b93ea88b47&uid=&ref=https://www.mql5.com/en/articles/16098&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5072029190611481341)

MetaTrader 5 / Examples


### Introduction

This article is a continuation of a series of articles where we will build a library called Connexus. In the [first article](https://www.mql5.com/en/articles/15795), we understood the basic operation of the [WebRequest](https://www.mql5.com/en/docs/network/webrequest) function, understanding each of its parameters, and also created an example code that demonstrates the use of this function and its difficulties. In the [last article](https://www.mql5.com/en/articles/16043) we understood how a request works, what headers are and how to use headers to make a request, and in the end we developed support for different headers in the library.

In API development, communication between client and server involves sending essential information through an HTTP request. And, if the _headers_ are like the envelope of this communication, the body is the letter itself — containing the actual data you want to transmit. In today’s article, we’ll explore the role of the body in an HTTP request, its importance, and how to configure it properly with Connexus. Let’s get started!

### What is a body in HTTP?

In the HTTP protocol, the body of a request or response refers to the actual content that is being sent or received. In simple terms, the body is where the data that interests us, that we want to send to the server, or receive from the server, is stored. It is the main component of requests of the POST , PUT and PATH types, in which it is necessary to send information such as forms, structured data in formats such as JSON or XML, and even a file. In this series of articles, the main focus will be on the use of the JSON format, which is the most used to consume APIs, but know that we can send in other formats.

In a GET type request there is usually no body, since this type of request is used to query data, that is, to receive information and not send it. Normally the server responds to this type of request with a body containing the results that were requested. However, in a POST request, the body is essential, because it is through it that the data is sent to the server to be processed. The server may or may not respond to this type of request with another body.

The HTTP body is used to transmit information from the client to the server, or vice versa, depending on the type of request. It is vital in operations that involve creating, updating or even removing data. The main function of the body is, therefore, to carry the "real content" that the server needs to process. Without it, HTTP com Done! Now just convert it to a char array: munication would be limited to mere requests for information, without the possibility of transmitting complex data or performing more sophisticated actions.

Now that we understand the role of the body in HTTP communication, it is important to know how to use it correctly in requests. Depending on the type of data you want to send, the _body_ can have different formats, such as JSON, XML, or binary data (in cases such as file uploads). Let's look at some practical examples.

1. **JSON Body**: The JSON (JavaScript Object Notation) format is the most common in modern applications, especially in REST APIs. It is lightweight, easy to read and ideal for transporting structured data, and will be the most widely used format when using HTTP in MQL5. Let's see how JSON would be used in a request body:




```
{
     "type": "BUY",
     "symbol": "EURUSD",
     "price": 1.09223
     "volume": 0.01
     "tp": 1.09233
     "sl": 1.09213
}
```




In this example, we are sending transaction data to the server, so that the server can receive and process the data, saving it in a database and generating a dashboard with performance metrics for the account transactions, or it can simply resend the transaction data to other accounts, creating a transaction copy system. To use this body, you must specify in the header that the content type is JSON:




```
Content-Type: application/json
```


2. **Body Text**: Another common way to send data in an HTTP request is in plain text format. This format is advantageous due to its simplicity. You just need to write what you want to send without following many rules or conventions, as long as the server supports what is being sent. It is not recommended for sending data, since it is difficult to organize a lot of data. For this scenario, the most recommended format is JSON. Let's look at an example of plain text:




```
This is a plain text example
```




Here, the form fields are concatenated and separated by & , and each value is assigned a specific key. To use this format, the header should be set to:




```
Content-Type: application/text-plain
```




This is one of the oldest formats, but it is still widely used in various applications. Our Connexus library will support this content type, allowing developers to choose the approach that best suits their use cases.


### Adding a body to an HTTP request

Let's take a practical example where we will send a JSON in the body of a request and check if the server received it correctly. To do this check, we will continue using [httpbin](https://www.mql5.com/go?link=https://httpbin.org/ "https://httpbin.org/"), which has already been mentioned in previous articles. Right from the start, I will create another file called TestBody.mq5 in the Experts/Connexus/Test/TestBody.mq5 folder, and I will add a simple post request that was used in the last article. All the files used in this article are attached at the end.

```
//+------------------------------------------------------------------+
//|                                                     TestBody.mq5 |
//|                                  Copyright 2024, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"

#include <Connexus/Data/Json.mqh>
#include <Connexus/URL/URL.mqh>
#include <Connexus/Header/HttpHeader.mqh>
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   //--- URL
   CURL url;
   url.Parse("https://httpbin.org");
   url.Path("post");

   //--- Data to be sent
   string method = "POST";
   char body_send[];

   //--- Headers that will be sent separated by "\n"
   CHttpHeader headers_send;
   headers_send.Add("User-Agent","Connexus/1.0 (MetaTrader 5 Terminal)");
   headers_send.Add("Content-Type","application/json");

   //--- Data that will be received
   char body_receive[];
   string headers_receive;

   //--- Send request
   int status_code = WebRequest(method,url.FullUrl(),headers_send.Serialize(),5000,body_send,body_receive,headers_receive);

   //--- Show response
   Print("Respose: ",CharArrayToString(body_receive));
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
```

The body of the request must be present in the body\_send variable, which is an array of type char. You might be wondering, why not be of type string instead of char\[\], string would be ideal for sending JSON or text? For several reasons related to flexibility, performance, and compatibility. Here are some detailed reasons:

- **Data Type Flexibility**: The body of an HTTP request can contain various types of data, such as text, JSON, binary (files, images), or other formats. Using a char array allows the function to accept any type of data, regardless of whether it is represented as text or binary. If the body contains a file (image, audio, etc.), it is necessary to manipulate it as an array of bytes, which can be handled directly in a char array. A string would only be useful for purely textual data, which limits the use of the function to scenarios where the content is binary.
- **Performance**: char arrays are more efficient in terms of low-level data manipulation. Because they do not involve the overhead of dynamic string management (such as memory allocations and reallocations), they allow networking code to work closer to the hardware, which is essential for applications that require high performance, such as large file transmissions or low-latency requests. When sending a request with a large image file, using a char array avoids the need to convert the binary content to a string, saving CPU cycles.
- **HTTP Protocol Compatibility**: The HTTP protocol works directly with bytes when transmitting data over networks. A char array (which is a sequence of bytes) better reflects this low-level behavior of the protocol. Therefore, by using char arrays, the HTTP request function will be aligned with the way data is handled and transmitted at the network level.

Using a char array for the body of an HTTP request offers superior flexibility, performance, and compatibility, especially in scenarios involving binary data or large volumes of data. It allows the function to directly deal with data in its most basic form (bytes), avoiding the overhead and limitations of working with strings , which are more suitable for textual data.

Now that we understand this, let's add a body to this request, for this we will use the [StringToCharArray()](https://www.mql5.com/en/docs/convert/stringtochararray) function, let's explore what the documentation says about this function:

| Parameter | Type | Description |
| --- | --- | --- |
| text\_string | string | String to copy. |
| array\[\] | char\[\] | Array of uchar type. |
| start | int | Position from which copying starts. Default - 0. |
| count | int | Number of array elements to copy. Defines length of a resulting string. Default value is -1, which means copying up to the array end, or till terminal 0. Terminal 0 will also be copied to the recipient array, in this case the size of a dynamic array can be increased if necessary to the size of the string. If the size of the dynamic array exceeds the length of the string, the size of the array will not be reduced. |
| codepage | uint | The value of the code page. When converting string variables to char type arrays and vice versa, the encoding that by default corresponds to the current ANSI of the Windows operating system (CP\_ACP) is used in MQL5. If you want to specify a different type of encoding, it can be set in this parameter. |

The table below lists the internal constants of some of the most popular [codepages](https://www.mql5.com/en/docs/constants/io_constants/codepageusage). Unspecified codepages can be specified using the corresponding code page.

| Constant | Value | Description |
| --- | --- | --- |
| CP\_ACP | 0 | The current Windows ANSI code page. |
| CP\_OEMCP | 1 | The current system OEM code page. |
| CP\_MACCP | 2 | The current system Macintosh code page. This value is mostly used in earlier created program codes and is of no use now, since modern Macintosh computers use Unicode for encoding. |
| CP\_THREAD\_ACP | 3 | The Windows ANSI code page for the current thread. |
| CP\_SYMBOL | 42 | Symbol code page |
| CP\_UTF7 | 65000 | UTF-7 code page. |
| CP\_UTF8 | 65001 | UTF-8 code page. |

For most APIs, we will use [UTF8](https://en.wikipedia.org/wiki/UTF-8 "https://en.wikipedia.org/wiki/UTF-8") which is a standard encoding type for email, web pages and others. So let's add a body in json format following the UTF8 encoding:

```
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   //--- Data to be sent
   string method = "POST";
   CJson body;
   body["type"] = "BUY";
   body["symbol"] = "EURUSD";
   body["price"] = 1.09223;
   body["volume"] = 0.01;
   body["tp"] = 1.09233;
   body["sl"] = 1.09213;
   //--- {"price": 1.09223,"sl": 1.09213,"symbol": "EURUSD","tp": 1.09233,"type": "BUY","volume": 0.01}

   //--- Char that will be sent
   char body_send[];

   //--- Convert string to char (UTF8)
   StringToCharArray(body.Serialize(),body_send,0,WHOLE_ARRAY,CP_UTF8);

   //--- Show char array
   ArrayPrint(body_send);

   return(INIT_SUCCEEDED);
  }
```

When you run this code, this will be displayed in the toolbox:

![toolbox | experts](https://c.mql5.com/2/97/toolbox1.png)

Note that in the last position of the array we have a value of “0”, which may be a problem for the server when reading the request body. To avoid this, we will remove the last position of the array using the [ArrayRemove()](https://www.mql5.com/en/docs/array/arrayremove) and [ArraySize()](https://www.mql5.com/en/docs/array/arraysize) functions.

```
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   //--- Data to be sent
   string method = "POST";
   CJson body;
   body["type"] = "BUY";
   body["symbol"] = "EURUSD";
   body["price"] = 1.09223;
   body["volume"] = 0.01;
   body["tp"] = 1.09233;
   body["sl"] = 1.09213;
   //--- {"price": 1.09223,"sl": 1.09213,"symbol": "EURUSD","tp": 1.09233,"type": "BUY","volume": 0.01}

   //--- Char that will be sent
   char body_send[];

   //--- Convert string to char (UTF8)
   StringToCharArray(body.Serialize(),body_send,0,WHOLE_ARRAY,CP_UTF8);
   ArrayRemove(body_send,ArraySize(body_send)-1);

   //--- Show char array
   ArrayPrint(body_send);

   return(INIT_SUCCEEDED);
  }
```

When you run it again, this will be displayed in the toolbox:

![toolbox | experts](https://c.mql5.com/2/97/toolbox2.png)

Now that we've got this little tweak sorted out, let's actually add this body to an http request:

```
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   //--- URL
   CURL url;
   url.Parse("https://httpbin.org");
   url.Path("post");

   //--- Data to be sent
   string method = "POST";
   CJson body;
   body["type"] = "BUY";
   body["symbol"] = "EURUSD";
   body["price"] = 1.09223;
   body["volume"] = 0.01;
   body["tp"] = 1.09233;
   body["sl"] = 1.09213;
   char body_send[];
   StringToCharArray(body.Serialize(),body_send,0,WHOLE_ARRAY,CP_UTF8);
   ArrayRemove(body_send,ArraySize(body_send)-1);

   //--- Headers that will be sent separated by "\n"
   CHttpHeader headers_send;
   headers_send.Add("User-Agent","Connexus/1.0 (MetaTrader 5 Terminal)");
   headers_send.Add("Content-Type","application/json");

   //--- Data that will be received
   char body_receive[];
   string headers_receive;

   //--- Send request
   int status_code = WebRequest(method,url.FullUrl(),headers_send.Serialize(),5000,body_send,body_receive,headers_receive);

   //--- Show response
   Print("Respose: ",CharArrayToString(body_receive));
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
```

When running this code, we will get this response from httpbin:

```
Respose: {
  "args": {},
  "data": "{\"type\":\"BUY\",\"symbol\":\"EURUSD\",\"price\":1.09223000,\"volume\":0.01000000,\"tp\":1.09233000,\"sl\":1.09213000}",
  "files": {},
  "form": {},
  "headers": {
    "Accept": "*/*",
    "Accept-Encoding": "gzip, deflate",
    "Accept-Language": "pt,en;q=0.5",
    "Content-Length": "103",
    "Content-Type": "text/plain",
    "Host": "httpbin.org",
    "User-Agent": "Connexus/1.0 (MetaTrader 5 Terminal)",
    "X-Amzn-Trace-Id": "Root=1-67081b9c-3def4b1527d04edc1511cc6b"
  },
  "json": {
    "price": 1.09223,
    "sl": 1.09213,
    "symbol": "EURUSD",
    "tp": 1.09233,
    "type": "BUY",
    "volume": 0.01
  },
  "origin": "189.74.63.39",
  "url": "https://httpbin.org/post"
}
```

Notice two interesting things, the first is that the “data” field contains the JSON that we sent in the body in string format, which means that the server was able to receive and correctly interpret the data sent. Another thing to note is that the “json” field contains the JSON that we sent, showing once again that the server correctly received the data. Working perfectly!

### Creating the CHttpBody class

Now that we understand how the body works, what it is for and how to use it, let's create a class in the Connexus library to work with the request body. The name of this class will be CHttpBody, and it will have methods to work with the body, being able to add, update or remove data. It will also be possible to define the encoding used (default will be UTF8).

Let's create a new file called HttpBody.mqh in the Include/Connexus/Header/HttpBody,mqh folder. When creating the file it will initially look similar to this:

```
//+------------------------------------------------------------------+
//|                                                     HttpBody.mqh |
//|                                  Copyright 2024, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
//+------------------------------------------------------------------+
//| class : CHttpBody                                                |
//|                                                                  |
//| [PROPERTY]                                                       |
//| Name        : CHttpBody                                          |
//| Heritage    : No heritage                                        |
//| Description : Responsible for organizing and storing the body of |
//|               a request.                                         |
//|                                                                  |
//+------------------------------------------------------------------+
class CHttpBody
  {
public:
                     CHttpBody(void);
                    ~CHttpBody(void);

  };
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CHttpBody::CHttpBody(void)
  {
  }
//+------------------------------------------------------------------+
//| Destructor                                                       |
//+------------------------------------------------------------------+
CHttpBody::~CHttpBody(void)
  {
  }
//+------------------------------------------------------------------+
```

Let's define some private properties of the class, they are:

- m\_body : Will store the request body as a char array.
- m\_codepage : Used to store the defined encoding

```
//+------------------------------------------------------------------+
//| Include the file CJson class                                     |
//+------------------------------------------------------------------+
#include "../Data/Json.mqh"
//+------------------------------------------------------------------+
//| class : CHttpBody                                                |
//|                                                                  |
//| [PROPERTY]                                                       |
//| Name        : CHttpBody                                          |
//| Heritage    : No heritage                                        |
//| Description : Responsible for organizing and storing the body of |
//|               a request.                                         |
//|                                                                  |
//+------------------------------------------------------------------+
class CHttpBody
  {
private:

   char              m_body[];                           // Will store the request body as a char array
   uint              m_codepage;                         // Used to store the defined encoding
  };;
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CHttpBody::CHttpBody(void)
  {
   m_codepage = CP_UTF8;
  }
//+------------------------------------------------------------------+
```

Now let's define some public methods to handle the request body:

- Add Data to Body
  - AddString(string data) : Adds a text string to the request body.
  - AddJson(CJson data) : Adds JSON-formatted data to the body. This may involve converting a JSON object to a formatted string.
  - AddBinary(char &data\[\]) : Allows you to add binary data (such as files) directly to the request body.
- Remove Data from Body
  - Clear(void) : Removes all the content from the request body, allowing you to start from scratch.
- Get the Contents of the Body
  - GetAsString(void) : Returns the request body as a string.
  - GetAsJson(void) : Converts the request body to a JSON object, useful when the body contains structured data. - GetAsChar(char &body\[\]) : Returns the body as an array of bytes, useful for working with binary data.
- Checking the Body Size
  - GetSize(void) : Returns the size of the request body, usually in bytes.
- Encoding
  - GetCodePage(void) : Returns the defined codepage
  - SetCodePage(uint codepage) : Sets the codepage to be used

Let's add these methods to the classes, in the end it will look like this:

```
//+------------------------------------------------------------------+
//| class : CHttpBody                                                |
//|                                                                  |
//| [PROPERTY]                                                       |
//| Name        : CHttpBody                                          |
//| Heritage    : No heritage                                        |
//| Description : Responsible for organizing and storing the body of |
//|               a request.                                         |
//|                                                                  |
//+------------------------------------------------------------------+
class CHttpBody
  {
private:

   char              m_body[];                           // Will store the request body as a char array
   uint              m_codepage;                         // Used to store the defined encoding

public:
                     CHttpBody(void);
                    ~CHttpBody(void);

   //--- Add data to the body
   void              AddString(string data);             // Adds a text string to the request body
   void              AddJson(CJson &data);               // Adds data in JSON format to the body
   void              AddBinary(char &data[]);            // Allows you to add binary data

   //--- Clear the body
   void              Clear(void);                        // Remove all body content

   //--- Gets the body content
   string            GetAsString(void);                  // Returns the request body as a string
   CJson             GetAsJson(void);                    // Converts the request body into a JSON object, useful when the body contains structured data
   void              GetAsBinary(char &body[]);          // Returns the body as an array of bytes, useful for working with binary data

   //--- Size in bytes
   int               GetSize(void);                      // Returns the size of the request body, usually in bytes

   //--- Codepage
   uint              GetCodePage(void);                  // Returns the defined codepage
   void              SetCodePage(uint codepage);         // Defines the codepage to be used
  };
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CHttpBody::CHttpBody(void)
  {
   m_codepage = CP_UTF8;
  }
//+------------------------------------------------------------------+
//| Destructor                                                       |
//+------------------------------------------------------------------+
CHttpBody::~CHttpBody(void)
  {
  }
//+------------------------------------------------------------------+
//| Adds a text string to the request body                           |
//+------------------------------------------------------------------+
void CHttpBody::AddString(string data)
  {
   StringToCharArray(data,m_body,this.GetSize()-1,WHOLE_ARRAY,m_codepage);
   ArrayRemove(m_body,this.GetSize()-1);
  }
//+------------------------------------------------------------------+
//| Adds data in JSON format to the body                             |
//+------------------------------------------------------------------+
void CHttpBody::AddJson(CJson &data)
  {
   this.AddString(data.Serialize());
  }
//+------------------------------------------------------------------+
//| Allows you to add binary data                                    |
//+------------------------------------------------------------------+
void CHttpBody::AddBinary(char &data[])
  {
   ArrayCopy(m_body,data);
  }
//+------------------------------------------------------------------+
//| Remove all body content                                          |
//+------------------------------------------------------------------+
void CHttpBody::Clear(void)
  {
   ArrayFree(m_body);
  }
//+------------------------------------------------------------------+
//| Returns the request body as a string                             |
//+------------------------------------------------------------------+
string CHttpBody::GetAsString(void)
  {
   return(CharArrayToString(m_body,0,WHOLE_ARRAY,m_codepage));
  }
//+------------------------------------------------------------------+
//| Converts the request body into a JSON object, useful when the    |
//| body contains structured data                                    |
//+------------------------------------------------------------------+
CJson CHttpBody::GetAsJson(void)
  {
   CJson json;
   json.Deserialize(this.GetAsString());
   return(json);
  }
//+------------------------------------------------------------------+
//| Returns the body as an array of bytes, useful for working with   |
//| binary data                                                      |
//+------------------------------------------------------------------+
void CHttpBody::GetAsBinary(char &body[])
  {
   ArrayCopy(body,m_body);
  }
//+------------------------------------------------------------------+
//| Returns the size of the request body, usually in bytes           |
//+------------------------------------------------------------------+
int CHttpBody::GetSize(void)
  {
   return(ArraySize(m_body));
  }
//+------------------------------------------------------------------+
//| Returns the defined codepage                                     |
//+------------------------------------------------------------------+
uint CHttpBody::GetCodePage(void)
  {
   return(m_codepage);
  }
//+------------------------------------------------------------------+
//| Defines the codepage to be used                                  |
//+------------------------------------------------------------------+
void CHttpBody::SetCodePage(uint codepage)
  {
   m_codepage = codepage;
  }
//+------------------------------------------------------------------+
```

These methods are simple and straightforward—the largest of them is only three lines long. However, don’t let their simplicity fool you. They are extremely useful and will have a significant impact on reducing the total number of lines of code in your library. Not only will they make your code leaner, but they will also make your library much easier to use and maintain.

### Tests

Let's move on to the tests and see how the class behaves. I will use the same file from the beginning of the article, TestBody.mq5 .

```
{
  "type": "BUY",
  "symbol": "EURUSD",
  "price": 1.09223
  "volume": 0.01
  "tp": 1.09233
  "sl": 1.09213
}
```

In this test, we will add this JSON to the body of a POST request. We will create the json object with the data:

```
CJson body_json;
body_json["type"] = "BUY";
body_json["symbol"] = "EURUSD";
body_json["price"] = 1.09223;
body_json["volume"] = 0.01;
body_json["tp"] = 1.09233;
body_json["sl"] = 1.09213;
```

Let's create an instance of the CHttpBody class and add this JSON inside it:

```
CHttpBody body;
body.AddJson(body_json);
```

Done! Now just convert it to a char array:

```
//--- Body in char array
char body_send[];
body.GetAsBinary(body_send);
```

It's that simple, we add the JSON to the request without any complications. This last step of converting it to a char array will not be necessary at the end of the library, as we are still in development we still do everything "by hand". In the end the code will look like this:

```
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   //--- URL
   CURL url;
   url.Parse("https://httpbin.org");
   url.Path("post");

   //--- Data to be sent
   string method = "POST";
   CJson body_json;
   body_json["type"] = "BUY";
   body_json["symbol"] = "EURUSD";
   body_json["price"] = 1.09223;
   body_json["volume"] = 0.01;
   body_json["tp"] = 1.09233;
   body_json["sl"] = 1.09213;
   CHttpBody body;
   body.AddJson(body_json);

   //--- Body in char array
   char body_send[];
   body.GetAsBinary(body_send);

   //--- Headers that will be sent separated by "\n"
   CHttpHeader headers_send;
   headers_send.Add("User-Agent","Connexus/1.0 (MetaTrader 5 Terminal)");
   headers_send.Add("Content-Type","application/json");

   //--- Data that will be received
   char body_receive[];
   string headers_receive;

   //--- Send request
   int status_code = WebRequest(method,url.FullUrl(),headers_send.Serialize(),5000,body_send,body_receive,headers_receive);

   //--- Show response
   Print("Respose: ",CharArrayToString(body_receive));
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
```

When running we will have this result:

```
Respose: {
  "args": {},
  "data": "{\"type\":\"BUY\",\"symbol\":\"EURUSD\",\"price\":1.09223000,\"volume\":0.01000000,\"tp\":1.09233000,\"sl\":1.09213000}",
  "files": {},
  "form": {},
  "headers": {
    "Accept": "*/*",
    "Accept-Encoding": "gzip, deflate",
    "Accept-Language": "pt,en;q=0.5",
    "Content-Length": "103",
    "Content-Type": "application/json",
    "Host": "httpbin.org",
    "User-Agent": "Connexus/1.0 (MetaTrader 5 Terminal)",
    "X-Amzn-Trace-Id": "Root=1-670902d1-5a796b1e1fe2053f18a07654"
  },
  "json": {
    "price": 1.09223,
    "sl": 1.09213,
    "symbol": "EURUSD",
    "tp": 1.09233,
    "type": "BUY",
    "volume": 0.01
  },
  "origin": "189.74.63.39",
  "url": "https://httpbin.org/post"
}
```

Notice once again that the “data” and “json” fields contain an object inside them, which means that the server correctly received the data we sent in the body. If you want to send a body with plain text, without formatting, just insert it in the CHttpBody class as a string, and change the header to text/plain as we saw in the [last article](https://www.mql5.com/en/articles/16043):

```
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   //--- URL
   CURL url;
   url.Parse("https://httpbin.org");
   url.Path("post");

   //--- Data to be sent
   string method = "POST";
   CHttpBody body;
   body.AddString("My simple text");

   //--- Body in char array
   char body_send[];
   body.GetAsBinary(body_send);

   //--- Headers that will be sent separated by "\n"
   CHttpHeader headers_send;
   headers_send.Add("User-Agent","Connexus/1.0 (MetaTrader 5 Terminal)");
   headers_send.Add("Content-Type","text/plain");

   //--- Data that will be received
   char body_receive[];
   string headers_receive;

   //--- Send request
   int status_code = WebRequest(method,url.FullUrl(),headers_send.Serialize(),5000,body_send,body_receive,headers_receive);

   //--- Show response
   Print("Respose: ",CharArrayToString(body_receive));
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
```

### Conclusion

In this article, we explored the concept of body in HTTP requests, showing its fundamental role in data transmission between client and server. We understand that the body is where we place the data we want to send, and it can be formatted in different ways such as JSON, XML and even file, but we explored more the use of JSON, which is the most used in the context of APIs, which is our focus for now. In addition, we discussed the headers required for each type of body, ensuring that the request is interpreted correctly by the server.

We also introduced the creation of the CHttpBody class in the Connexus library, which will be responsible for facilitating work with the body of the request. This class allows easy manipulation of the data that will be sent, without worrying about low-level formatting (bytes).

In the next article in the series, we will delve even deeper into the operations of the HTTP protocol, exploring some methods, such as GET, POST, PUT, DELETE and others. We will also discuss HTTP status codes, such as 200 (Ok), perhaps the most famous on the web, 404 (Not Found) and others like 500 (Internal Server Error), and what they mean for client-server communication. Stay tuned!

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/16098.zip "Download all attachments in the single ZIP archive")

[Body\_in\_Connexus\_mPart\_42\_Adding\_HTTP\_body\_support.zip](https://www.mql5.com/en/articles/download/16098/body_in_connexus_mpart_42_adding_http_body_support.zip "Download Body_in_Connexus_mPart_42_Adding_HTTP_body_support.zip")(19.59 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Simplifying Databases in MQL5 (Part 2): Using metaprogramming to create entities](https://www.mql5.com/en/articles/19594)
- [Simplifying Databases in MQL5 (Part 1): Introduction to Databases and SQL](https://www.mql5.com/en/articles/19285)
- [Mastering Log Records (Part 10): Avoiding Log Replay by Implementing a Suppression](https://www.mql5.com/en/articles/19014)
- [Mastering Log Records (Part 9): Implementing the builder pattern and adding default configurations](https://www.mql5.com/en/articles/18602)
- [Mastering Log Records (Part 8): Error Records That Translate Themselves](https://www.mql5.com/en/articles/18467)
- [Mastering Log Records (Part 7): How to Show Logs on Chart](https://www.mql5.com/en/articles/18291)
- [Mastering Log Records (Part 6): Saving logs to database](https://www.mql5.com/en/articles/17709)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/474787)**
(2)


![yucong tang](https://c.mql5.com/avatar/2025/5/6817664A-322F.png)

**[yucong tang](https://www.mql5.com/en/users/yucongtang)**
\|
4 Jun 2025 at 22:37

How to send Chinese content? Chinese content is messy.


![joaopedrodev](https://c.mql5.com/avatar/2024/9/66da07c2-0125.png)

**[joaopedrodev](https://www.mql5.com/en/users/joaopedrodev)**
\|
10 Sep 2025 at 23:38

The problem usually isn't the Chinese content itself, but rather the character encoding used to send and interpret the HTTP request body. Ideally, UTF-8 should be used, as it's compatible with ASCII and supports all Chinese characters. Additionally, it's important to ensure that the Content-Type header specifies charset=utf-8 and that the server is also configured to interpret content in that format.


![How to Create an Interactive MQL5 Dashboard/Panel Using the Controls Class (Part 1): Setting Up the Panel](https://c.mql5.com/2/97/How_to_Create_an_Interactive_MQL5_Dashboard___LOGO.png)[How to Create an Interactive MQL5 Dashboard/Panel Using the Controls Class (Part 1): Setting Up the Panel](https://www.mql5.com/en/articles/16084)

In this article, we create an interactive trading dashboard using the Controls class in MQL5, designed to streamline trading operations. The panel features a title, navigation buttons for Trade, Close, and Information, and specialized action buttons for executing trades and managing positions. By the end of the article, you will have a foundational panel ready for further enhancements in future installments.

![Developing a Replay System (Part 47): Chart Trade Project (VI)](https://c.mql5.com/2/75/Desenvolvendo_um_sistema_de_Replay_iParte_47i___LOGO.png)[Developing a Replay System (Part 47): Chart Trade Project (VI)](https://www.mql5.com/en/articles/11760)

Finally, our Chart Trade indicator starts interacting with the EA, allowing information to be transferred interactively. Therefore, in this article, we will improve the indicator, making it functional enough to be used together with any EA. This will allow us to access the Chart Trade indicator and work with it as if it were actually connected with an EA. But we will do it in a much more interesting way than before.

![Reimagining Classic Strategies (Part X): Can AI Power The MACD?](https://c.mql5.com/2/97/Reimagining_Classic_Strategies_Part_X___LOGO.png)[Reimagining Classic Strategies (Part X): Can AI Power The MACD?](https://www.mql5.com/en/articles/16066)

Join us as we empirically analyzed the MACD indicator, to test if applying AI to a strategy, including the indicator, would yield any improvements in our accuracy on forecasting the EURUSD. We simultaneously assessed if the indicator itself is easier to predict than price, as well as if the indicator's value is predictive of future price levels. We will furnish you with the information you need to decide whether you should consider investing your time into integrating the MACD in your AI trading strategies.

![Data Science and ML (Part 31): Using CatBoost AI Models for Trading](https://c.mql5.com/2/97/Data_Science_and_ML_Part_31___LOGO.png)[Data Science and ML (Part 31): Using CatBoost AI Models for Trading](https://www.mql5.com/en/articles/16017)

CatBoost AI models have gained massive popularity recently among machine learning communities due to their predictive accuracy, efficiency, and robustness to scattered and difficult datasets. In this article, we are going to discuss in detail how to implement these types of models in an attempt to beat the forex market.

[![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/01.png)![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/02.png)Powerful analytics for traders of any levelAll the necessary trading reports for beginners and professionals](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/454106&a=muccpajyfystoakuukdobwigjejzmpqn&s=52daad60fa795e635264e6f94898f05493bca3b5124d4cca8eb7e82333c2ef12&uid=&ref=https://www.mql5.com/en/articles/16098&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5072029190611481341)

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