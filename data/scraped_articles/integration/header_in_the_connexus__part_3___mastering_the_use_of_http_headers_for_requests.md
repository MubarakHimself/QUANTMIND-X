---
title: Header in the Connexus (Part 3): Mastering the Use of HTTP Headers for Requests
url: https://www.mql5.com/en/articles/16043
categories: Integration, Expert Advisors
relevance_score: 0
scraped_at: 2026-01-24T14:03:12.338446
---

[![](https://www.mql5.com/ff/sh/9nb0c8df2rmwfn89z2/01.png) MetaTrader VPS vs regular cloud hosting services8 reasons why our solution is the best option for automated tradingRead](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/450486&a=dgmsfszgoedimaicrqqmagvqzpwuxkur&s=c59e3617ccf44fd54d4c50a03b44fd689ff7507b8fe4990c83772cc5419e627d&uid=&ref=https://www.mql5.com/en/articles/16043&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5083304587865102573)

MetaTrader 5 / Examples


### Introduction

This article is the continuation of a series of articles where we will build a library called Connexus. In the [first article](https://www.mql5.com/en/articles/15795), we understood the basic functioning of the [WebRequest](https://www.mql5.com/en/docs/network/webrequest) function, understanding each of its parameters and also created an example code that demonstrates the use of this function and its difficulties. In this article, we will explore the importance and usefulness of headers in HTTP communication, and how these elements are used for different purposes in the modern web.

The structure of an HTTP message, whether a response or a request, is composed of two fundamental elements that we will delve into: headers and body. Each of them has its role in the communication process, ensuring that the data is transmitted in an organized, efficient and secure way.

To begin, let's briefly recap how an HTTP request and response structure works.

### Structure of an HTTP Request

An HTTP request typically follows this format:

```
HTTP Method | URL | HTTP Version
Headers
Body (Optional)
```

- **HTTP Method**: Defines the intent of the request (GET, POST, PUT, DELETE, etc.).
- **URL**: Identifies the resource being requested. (We discussed this in more detail in the last article)
- **HTTP Version**: Specifies the protocol version being used.
- **Headers**: Request metadata, such as content type, authentication, etc.
- **Body**: The content of the request, usually present in methods such as POST or PUT

### Structure of an HTTP Response

An HTTP response follows a similar structure:

```
HTTP Version | Status Code | Status Message
Headers
Body (Optional)
```

- **Status Code**: Indicates the result (200 OK, 404 Not Found, 500 Internal Server Error, etc.).
- **Headers**: Information about the content returned, such as size, data type, etc.
- **Body**: The actual content of the response, such as the HTML of a page or JSON data. In our case, it will be more common to receive JSON data, but be aware that some APIs may return HTML.

In the [last article](https://www.mql5.com/en/articles/15897) we delved deeper into the format of a URL, understanding each element in isolation and grouping them together to form a complete address. In this article we will delve deeper into the header of an HTTP request. What is it for? How to use it? What are the possible values, etc.

### Headers

First, let's understand what headers are. In the HTTP protocol, a header is a set of additional data that is sent with the request or response. HTTP headers are essential functions in client-server communications. Their main purpose is to provide details about the request that are not directly part of the URL or the message body. They help control the flow of communication and provide context to ensure that the server interprets and processes the data correctly. These headers serve to help both the server and the client better understand the context of the request or response, such as the content type, caching, authentication, among others. In other words, they act as metadata that informs the server about how the request can be processed and to the client how the response should be interpreted.

Let's mention some of the main functions of HTTP headers:

1. **Authentication**: One of the most common uses of headers is to authenticate the client, so that the server knows who is sending the request and whether they have access to the information. For example, the Authorization header sends a token or credentials that the server receives and can use to validate the client before processing the request.
2. **Cache Control**: Headers like Cache-Control allow the client and server to configure how data can be cached, which can be useful for avoiding another unnecessary request from the client to the server or from the server to some other service. Cached data can be stored on the client, proxies, or other intermediate points.
3. **Content-Type specification**: The Content-Type header allows the client to inform the server what type of data is being sent or received. Typically, formats such as JSON , XML , or HTML are used. This ensures that both sides of the communication know how to correctly interpret the data.
4. **Content Negotiation**: The client can use the Accept header to inform the server which response formats are acceptable, such as application/json (the server will send data in json format) or text/html (the server will send it in html format). This allows the server to send the response in a format that the client is prepared to receive. The most commonly used format is JSON and will be our focus here, but there are other formats supported.
5. **Security**: Headers such as Strict-Transport-Security help to reinforce the use of HTTPS (HTTPS is the same as HTTP, but it contains an additional layer of web security. The request, response, headers, body, URL and other formats are exactly the same, which is why it is recommended to always use HTTPS). Other headers such as CORS (Cross Origin Resource Sharing) define which domains can access an API's resources, increasing security. In this way, the server is filtering who it sends the information to, sending it only to a previously defined domain, so that no one other than that domain can access the data.
6. **Rate Limiting**: Some services return headers such as X-RateLimit-Limit and X-RateLimit-Reaming to inform the client how many requests are allowed in a given period of time. This prevents a large number of requests from overloading the server.

Thus, headers play a crucial role in HTTP communication, providing control, clarity, and security over how the request and response should be handled.

### Knowing Some Possible Values

Let's see what the possible values of the most common headers are. HTTP headers are highly versatile and can be customized as needed for communication between the client and the server. Understand that the values that a header can contain depend on the context of the communication. Below are some examples of the most common headers and their possible values:

1. **Authorization**: It is used to send credentials or authentication tokens from the client to the server, allowing access to protected resources. It can assume different values depending on the authentication method used. There are several authentication methods, let's look at the most commonly used ones:
   - **Bearer Token**: One of the most common formats, especially in modern APIs that use token-based authentication. This value is typically a JWT (JSON Web Token) token that the client receives after logging in or authenticating to an authentication server.




     ```
     Authorization: Bearer <my_token_jwt>
     ```




     The <my\_token\_jwt> value is the authentication token itself, typically a long string of Base64-encoded characters. This token contains user authorization information and has a limited validity.

   - **Basic**: This value uses HTTP basic authentication, where the username and password are Base64 encoded and sent directly in the header. This method is less secure, as the credentials can be easily decoded, and should not be used without HTTPS.




     ```
     Authorization: Basic base64(username:password)
     ```




     The value base64(username:password) is the Base64 encoding of the username:password pair. Although this method is simple, it is vulnerable to attack unless used over encrypted connections.

   - **Digest**: A more secure method than Basic , Digest uses a hash of the user's credentials instead of sending the credentials directly. While less common today with the rise of OAuth and JWT, it is still found in some APIs.




     ```
     Authorization: Digest username="admin", realm="example", nonce="dcd98b7102dd2f0e8b11d0f600bfb0c093", uri="/dir/index.html", response="6629fae49393a05397450978507c4ef1"
     ```




     Here, the value contains several fields, such as the username, the nonce (a number used once), the realm (scope of authentication), and the hashed encrypted response.

2. **Content-Type**: This header defines the type of content present in the body of the request or response. Let's look at some of the most commonly used possible values:
   - **application/json**: This is the most common value when dealing with APIs, as JSON is a lightweight and easy-to-read and write format for transferring data. Usage example:




     ```
     Content-Type: application/json
     ```

   - **application/xml**: XML (Extensible Markup Language) was widely used before JSON, and is still used in some legacy systems or old APIs. But most current APIs support JSON format, so don't worry about XML for now.




     ```
     Content-Type: application/xml
     ```

   - **multipart/form-data**: This value is used to send mixed data, especially when the client needs to upload files or form data. It's not our focus here for now, but it's good to know that this possibility exists.

   - **text/html**: Used when the content to be sent or received is HTML. This value is common in web page requests and responses.




     ```
     Content-Type: text/html
     ```

3. **User-Agent**: This header is used by the client to identify itself. It usually contains data about the type of device, browser, and operating system that is making the request. There is no specific rule for the values here, but there are some standards used:
   - **Web Browsers**: The value sent by browsers usually contains details about the browser name and version, as well as the operating system.




     ```
     User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36
     ```

   - **APIs or Application** s: Applications can use the custom value, such as:




     ```
     User-Agent: Connexus/1.0 (MetaTrader 5 Terminal)
     ```




     This lets the server know that the request is coming from a specific application, making it easier to debug or monitor if necessary.

4. **Accept**: This is used by the client to inform which content it accepts as a response. The possible values for this header are similar to Content-Type, let's look at some of them:
   - **application/json**, **application/xml**, **text/html**: You already know these, we saw them in the Content-Type section.

   - **image/png**, **image/jpeg**, **image/webp**: These values are used when the client expects to receive an image. For example, in APIs that provide graphical data, such as thumbnails or generated charts. Or simply images such as a user's avatar, or the website logo, etc.




     ```
     Accept: image/png, image/jpeg
     ```




     Here, the client is informing that they accept both PNG and JPEG images.

   - **\*/\*** _:_ This value indicates that the client accepts any type of content as a response. It is used when the client does not have a preference for a specific format or is prepared to deal with any type of response.

### How to Organize Header?

Now that we understand what headers are for and some possible values, let's understand how headers are organized. Headers are organized as a set of key and value pairs, and are inserted in requests. Although some headers are mandatory for some operations, most of them are optional and depend on the application's needs. Example of a basic header for an HTTP request:

```
GET /api/resource HTTP/1.1
Host: example.com
Authorization: Bearer token123
Content-Type: application/json
User-Agent: Connexus/1.0 (MetaTrader 5 Terminal)
```

As shown in this example, we have 3 headers:

- Authorization: Provides our access token “token123”
- Content-Type: Provides information on how the data is being sent, we use JSON
- User-Agent: Provides information to the server so that the server knows how to process the request. In this example, we use “Connexus/1.0 (MetaTrader 5 Terminal)”

These headers provide information so that the server knows how to process the request. This is just a simple example, and we will soon go into more detail about which sets are most commonly used, possible values, and what each one is for.

### Hand on the Code

We already understand how headers work, what they are for and how to use them, now let's get to the practical part. Remember httpbin? It's a free service that simply works as a mirror, everything we send it returns to us again, we'll use this to check which headers we're sending, and if there are any headers that the terminal itself adds automatically. To do this, I'll create a file called TestHeader.mq5 in the Experts/Connexus/Test/TestHeader.mq5 folder. Let's create a POST request without sending anything in the header and let's see what it responds to us:

```
#include <Connexus2/Data/Json.mqh>
#include <Connexus2/URL/URL.mqh>

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
   string headers_send;

   //--- Data that will be received
   char body_receive[];
   string headers_receive;

   //--- Send request
   int status_code = WebRequest(method,url.FullUrl(),headers_send,5000,body_send,body_receive,headers_receive);

   //--- Show response
   Print("Respose: ",CharArrayToString(body_receive));
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
```

When executing this code in the terminal, we will have this response:

```
Respose: {
  "args": {},
  "data": "",
  "files": {},
  "form": {},
  "headers": {
    "Accept": "*/*",
    "Accept-Encoding": "gzip, deflate",
    "Accept-Language": "pt,en;q=0.5",
    "Content-Length": "0",
    "Content-Type": "application/x-www-form-urlencoded",
    "Host": "httpbin.org",
    "User-Agent": "MetaTrader 5 Terminal/5.4518 (Windows NT 11.0.22631; x64)",
    "X-Amzn-Trace-Id": "Root=1-66feb3d9-50de44d019af8b0c1058436b"
  },
  "json": null,
  "origin": "189.74.63.39",
  "url": "https://httpbin.org/post"
}
```

Note the value of “headers”, it contains another json object with some header values that are automatically defined by the terminal. Remember that in the request we do not send any data, either in the header or in the body. Let's understand each of these headers that are sent automatically:

- **Accept**: As we saw previously, it tells the server what types of content the client is willing to accept as a response. In this case, the value \*/\* as seen previously, means that the client accepts any type of content as a response.

- **Accept-Encoding**: specifies the types of content encoding that the client can accept. Encodings are used to **compress data** to save network bandwidth.

  - **gzip**: This is a compression format used to reduce the size of the response sent.
  - **deflate**: It is another form of data compression that is similar to gzip, but with some technical differences in the algorithm.
- **Accept-Language**: This header tells the server which languages the client prefers, but the server must support multiple languages. It helps the server deliver content in the language most appropriate to the user.

  - **pt**: The client prefers to receive the response in Portuguese.
  - **en;q=0.5**: Here, en represents English, and q=0.5 would be a "quality factor" (from 0 to 1) that tells the relative priority of the language. A value of q=1.0 would be the maximum preference, while q=0.5 informs that the client accepts English, but prefers Portuguese.
- **Content-Length**: indicates the size of the request body in bytes. In this case the value is 0 , which means that there is no content, as said, we do not send data in the request body.

- **Content-Type**: informs the server of the type of data being sent to the server. In this case, the value application/x-www-form-urlencoded means that the data was sent in URL-encoded form format. I'm not sure why MetaTrader 5 sets this format by default.

- **Host**: specifies the name of the server to which the request is being sent. This is necessary so that the server knows which domain or IP is being requested, especially if it is serving multiple domains on a single IP address.

- **User-Agent**: is a string that identifies the **HTTP client** (in this case, MetaTrader 5) that is making the request. It contains details about the software and operating system being used.

- **X-Amzn-Trace-Id**: The X-Amzn-Trace-Id header is used by AWS (Amazon Web Services) services to trace requests across their distributed infrastructure. It helps identify and debug issues in applications running in the Amazon cloud.


  - **Root=1-66feb3d9-50de44d019af8b0c1058436b**: This value represents a unique tracking ID that can be used to identify the specific transaction. It is useful for diagnostics and performance monitoring.
  - **Meaning**: This identifier is automatically assigned by the AWS system and can be used to follow the path of a request as it passes through different services within the Amazon infrastructure.
  - **Common usage**: This header is used internally by AWS services to collect tracking and monitoring information. It is especially useful in microservices architectures to diagnose bottlenecks and latency issues.

I believe that MetaTrader 5 adds this automatically so that it can track requests made by the terminal for diagnostics such as the platform or to help fix a bug.

Now that we have seen which headers are added by default by MetaTrader 5, let's change some of them. Remember that headers cannot have two identical keys, that is, you cannot use Content-Type: application/json and Content-Type: application/x-www-form-urlencoded . Therefore, if we define the value of Content-Type it will overwrite the old value. Let's change some data, defining other values for the following headers:

- **Content-Type**: Content-Type: application/json
- **User-Agent**: Connexus/1.0 (MetaTrader 5 Terminal)

Let's add these values to the “headers\_send” variable, which is a string. Remember to add the “\\n” to separate the headers from each other. Here is the modified code:

```
#include <Connexus2/Data/Json.mqh>
#include <Connexus2/URL/URL.mqh>

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
   string headers_send = "User-Agent: Connexus/1.0 (MetaTrader 5 Terminal)\nContent-Type: application/json";

   //--- Data that will be received
   char body_receive[];
   string headers_receive;

   //--- Send request
   int status_code = WebRequest(method,url.FullUrl(),headers_send,5000,body_send,body_receive,headers_receive);

   //--- Show response
   Print("Respose: ",CharArrayToString(body_receive));
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
```

When executing the code, we will have this result:

```
Respose: {
  "args": {},
  "data": "",
  "files": {},
  "form": {},
  "headers": {
    "Accept": "*/*",
    "Accept-Encoding": "gzip, deflate",
    "Accept-Language": "pt,en;q=0.5",
    "Content-Length": "0",
    "Content-Type": "application/json",
    "Host": "httpbin.org",
    "User-Agent": "Connexus/1.0 (MetaTrader 5 Terminal)",
    "X-Amzn-Trace-Id": "Root=1-66feb90f-037374b15f220d3e28e1cb32"
  },
  "json": null,
  "origin": "189.74.63.39",
  "url": "https://httpbin.org/post"
}
```

Notice that we changed the User-Agent and Content-Type values to the ones we defined. Now that we have a simple example of a request sending some custom headers, let's add this headers feature to our library. Our goal is to create a class to work with headers. This class should be easy to use, have a simple and intuitive interface, and be able to add, remove or update headers easily.

### Creating the HttpHeaders Class

Let's create a new folder inside the Connexus folder, which is inside Includes. This new folder will be called Headers, and inside this new folder we create a new file called HttpHeaders.mqh. In the end it will look like this:

![path](https://c.mql5.com/2/96/path.png)

The file should have an empty class, similar to this. I added some comments:

```
//+------------------------------------------------------------------+
//|                                                       Header.mqh |
//|                                  Copyright 2024, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2023, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
//+------------------------------------------------------------------+
//| class : CHttpHeader                                              |
//|                                                                  |
//| [PROPERTY]                                                       |
//| Name        : CHttpHeader                                        |
//| Heritage    : No heritage                                        |
//| Description : Responsible for organizing and storing the headers |
//|               of a request.                                      |
//|                                                                  |
//+------------------------------------------------------------------+
class CHttpHeader
  {
public:
                     CHttpHeader(void);
                    ~CHttpHeader(void);
  };
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CHttpHeader::CHttpHeader(void)
  {
  }
//+------------------------------------------------------------------+
//| Destructor                                                       |
//+------------------------------------------------------------------+
CHttpHeader::~CHttpHeader(void)
  {
  }
//+------------------------------------------------------------------+
```

To store these headers we will use a json object, where the json key will be the header key, and in the same way the json value will be the header value. To do this we will import the json class and create a new instance called m\_headers inside the class.

```
//+------------------------------------------------------------------+
//| Include the file CJson class                                     |
//+------------------------------------------------------------------+
#include "../Data/Json.mqh"
//+------------------------------------------------------------------+
//| class : CHttpHeader                                              |
//|                                                                  |
//| [PROPERTY]                                                       |
//| Name        : CHttpHeader                                        |
//| Heritage    : No heritage                                        |
//| Description : Responsible for organizing and storing the headers |
//|               of a request.                                      |
//|                                                                  |
//+------------------------------------------------------------------+
class CHttpHeader
  {
private:

   CJson             m_headers;

public:
                     CHttpHeader(void);
                    ~CHttpHeader(void);
  };
//+------------------------------------------------------------------+
```

With the json object ready to store the data, the next step will be to define which methods this class should have. Initially, we will create the following methods:

- Add(string key, string value) : Adds a new header to the HTTP request or updates it if it already exists
- Get(string key) : Returns the value of a specific header, given its name.
- Remove(string key) : Removes a specific header.
- Has(string key) : Checks if a header with the specified key is present.
- Clear() : Removes all headers from the request.
- Count() : Returns the number of headers.

Let's add these methods that are simpler to the class

```
//+------------------------------------------------------------------+
//| class : CHttpHeader                                              |
//|                                                                  |
//| [PROPERTY]                                                       |
//| Name        : CHttpHeader                                        |
//| Heritage    : No heritage                                        |
//| Description : Responsible for organizing and storing the headers |
//|               of a request.                                      |
//|                                                                  |
//+------------------------------------------------------------------+
class CHttpHeader
  {
private:

   CJson             m_headers;

public:
                     CHttpHeader(void);
                    ~CHttpHeader(void);

   //--- Functions to manage headers
   void              Add(string key, string value);      // Adds a new header to the HTTP request or updates it if it already exists
   string            Get(string key);                    // Returns the value of a specific header, given its name.
   void              Remove(string key);                 // Removes a specific header.
   bool              Has(string key);                    // Checks whether a header with the specified key is present.
   void              Clear(void);                        // Removes all headers from the request.
   int               Count(void);                        // Returns the number of headers.
  };
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CHttpHeader::CHttpHeader(void)
  {
  }
//+------------------------------------------------------------------+
//| Destructor                                                       |
//+------------------------------------------------------------------+
CHttpHeader::~CHttpHeader(void)
  {
  }
//+------------------------------------------------------------------+
//| Adds a new header to the HTTP request or updates it if it already|
//| exists                                                           |
//+------------------------------------------------------------------+
void CHttpHeader::Add(string key,string value)
  {
   m_headers[key] = value;
  }
//+------------------------------------------------------------------+
//| Returns the value of a specific header, given its name.          |
//+------------------------------------------------------------------+
string CHttpHeader::Get(string key)
  {
   return(m_headers[key].ToString());
  }
//+------------------------------------------------------------------+
//| Removes a specific header.                                       |
//+------------------------------------------------------------------+
void CHttpHeader::Remove(string key)
  {
   m_headers.Remove(key);
  }
//+------------------------------------------------------------------+
//| Checks whether a header with the specified key is present.       |
//+------------------------------------------------------------------+
bool CHttpHeader::Has(string key)
  {
   return(m_headers.FindKey(key) != NULL);
  }
//+------------------------------------------------------------------+
//| Removes all headers from the request.                            |
//+------------------------------------------------------------------+
void CHttpHeader::Clear(void)
  {
   m_headers.Clear();
  }
//+------------------------------------------------------------------+
//| Returns the number of headers.                                   |
//+------------------------------------------------------------------+
int CHttpHeader::Count(void)
  {
   return(m_headers.Size());
  }
//+------------------------------------------------------------------+
```

Now that we have added the simplest methods, let's add the last two methods that are the heart of the class, they are:

- Serialize() : Returns all headers in string format, ready to be sent in an HTTP request.
- Parse(string headers) : Converts a string containing headers (usually received in an HTTP response) into a format usable in the class.

```
//+------------------------------------------------------------------+
//| class : CHttpHeader                                              |
//|                                                                  |
//| [PROPERTY]                                                       |
//| Name        : CHttpHeader                                        |
//| Heritage    : No heritage                                        |
//| Description : Responsible for organizing and storing the headers |
//|               of a request.                                      |
//|                                                                  |
//+------------------------------------------------------------------+
class CHttpHeader
  {
private:

   CJson             m_headers;

public:
                     CHttpHeader(void);
                    ~CHttpHeader(void);

   //--- Auxiliary methods
   string            Serialize(void);                    // Returns all headers in string format, ready to be sent in an HTTP request.
   bool              Parse(string headers);              // Converts a string containing headers (usually received in an HTTP response) into a format usable by the class.
  };
//+------------------------------------------------------------------+
//| Returns all headers in string format, ready to be sent in an HTTP|
//| request.                                                         |
//+------------------------------------------------------------------+
string CHttpHeader::Serialize(void)
  {
   //--- String with the result
   string headers;

   //--- Get size
   int size = this.Count();
   for(int i=0;i<size;i++)
     {
      //--- Adds the header to the string in the format: "key: value"
      headers += m_headers[i].m_key + ": " + m_headers[i].ToString();

      //--- If it's not the last time it adds "\n" at the end of the string
      if(i != size -1)
        {
         headers += "\n";
        }
     }

   //--- Return result
   return(headers);
  }
//+------------------------------------------------------------------+
//| Converts a string containing headers (usually received in an HTTP|
//| response) into a format usable by the class.                     |
//+------------------------------------------------------------------+
bool CHttpHeader::Parse(string headers)
  {
   //--- Array to store the key value sets
   string params[];

   //--- Separate the string, using the "\n" character as a separator
   int size = StringSplit(headers,StringGetCharacter("\n",0),params);
   for(int i=0;i<size;i++)
     {
      //--- With the header separated using ": "
      int pos = StringFind(params[i],": ");
      if(pos >= 0)
        {
         //--- Get key and value
         string key = StringSubstr(params[i],0,pos);
         string value = StringSubstr(params[i],pos+2);

         //--- Clear value
         StringTrimRight(value);

         //--- Add in json
         this.Add(key,value);
        }
     }
   return(true);
  }
//+------------------------------------------------------------------+
```

### Tests

To perform the class tests, I will use the same file that we created at the beginning of the article, TestHeader.mq5. Import the HttpHeader file, then create an instance of the CHttpHeader class, passing the data to the class. Then I use the Serialize() function to format it as a string.

```
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

   //--- Headers
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

When executing the code in the terminal, we get the same response:

```
Respose: {
  "args": {},
  "data": "",
  "files": {},
  "form": {},
  "headers": {
    "Accept": "*/*",
    "Accept-Encoding": "gzip, deflate",
    "Accept-Language": "pt,en;q=0.5",
    "Content-Length": "0",
    "Content-Type": "application/json",
    "Host": "httpbin.org",
    "User-Agent": "Connexus/1.0 (MetaTrader 5 Terminal)",
    "X-Amzn-Trace-Id": "Root=1-66fed891-0d6adb5334becd71123795c9"
  },
  "json": null,
  "origin": "189.74.63.39",
  "url": "https://httpbin.org/post"
}
```

### Conclusion

In short, HTTP headers are like the little notes you pass during class so the server knows what to do with your request. They can authenticate, set the content type, instruct about caching, and much more. Without them, HTTP communication would be as chaotic as trying to order a coffee without specifying the size, amount of sugar, or type of milk. Currently, the class diagram we have created so far looks like this:

![diagram connexus](https://c.mql5.com/2/96/diagram.png)

Now that you understand headers and how essential they are, it’s time to tackle something even more interesting: the request body. In the next article, we’ll dive into the heart of HTTP communication — the actual content you want to send to the server. After all, there’s no point in sending a note without content, right?

Get ready, the next chapter will explore how to elegantly and effectively package this information in the body. See you there!

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/16043.zip "Download all attachments in the single ZIP archive")

[Connexus\_aPart\_3b\_Adding\_Header\_Support.zip](https://www.mql5.com/en/articles/download/16043/connexus_apart_3b_adding_header_support.zip "Download Connexus_aPart_3b_Adding_Header_Support.zip")(16.53 KB)

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

**[Go to discussion](https://www.mql5.com/en/forum/474422)**

![Developing a multi-currency Expert Advisor (Part 12): Developing prop trading level risk manager](https://c.mql5.com/2/79/Developing_a_multi-currency_advisor_Part_12__LOGO__1.png)[Developing a multi-currency Expert Advisor (Part 12): Developing prop trading level risk manager](https://www.mql5.com/en/articles/14764)

In the EA being developed, we already have a certain mechanism for controlling drawdown. But it is probabilistic in nature, as it is based on the results of testing on historical price data. Therefore, the drawdown can sometimes exceed the maximum expected values (although with a small probability). Let's try to add a mechanism that ensures guaranteed compliance with the specified drawdown level.

![Reimagining Classic Strategies (Part IX): Multiple Time Frame Analysis (II)](https://c.mql5.com/2/96/Reimagining_Classic_Strategies_Part_IX___LOGO.png)[Reimagining Classic Strategies (Part IX): Multiple Time Frame Analysis (II)](https://www.mql5.com/en/articles/15972)

In today's discussion, we examine the strategy of multiple time-frame analysis to learn on which time frame our AI model performs best. Our analysis leads us to conclude that the Monthly and Hourly time-frames produce models with relatively low error rates on the EURUSD pair. We used this to our advantage and created a trading algorithm that makes AI predictions on the Monthly time frame, and executes its trades on the Hourly time frame.

![Creating an MQL5 Expert Advisor Based on the PIRANHA Strategy by Utilizing Bollinger Bands](https://c.mql5.com/2/97/PIRANHA_Strategy_by_Utilizing_Bollinger_Bands____LOGO.png)[Creating an MQL5 Expert Advisor Based on the PIRANHA Strategy by Utilizing Bollinger Bands](https://www.mql5.com/en/articles/16034)

In this article, we create an Expert Advisor (EA) in MQL5 based on the PIRANHA strategy, utilizing Bollinger Bands to enhance trading effectiveness. We discuss the key principles of the strategy, the coding implementation, and methods for testing and optimization. This knowledge will enable you to deploy the EA in your trading scenarios effectively

![Creating a Trading Administrator Panel in MQL5 (Part III): Extending Built-in Classes for Theme Management (II)](https://c.mql5.com/2/96/Creating_a_Trading_Administrator_Panel_in_MQL5_Part_III___V2____LOGO.png)[Creating a Trading Administrator Panel in MQL5 (Part III): Extending Built-in Classes for Theme Management (II)](https://www.mql5.com/en/articles/16045)

In this discussion, we will carefully extend the existing Dialog library to incorporate theme management logic. Furthermore, we will integrate methods for theme switching into the CDialog, CEdit, and CButton classes utilized in our Admin Panel project. Continue reading for more insightful perspectives.

[![](https://www.mql5.com/ff/sh/20jc81m23z78s5z9z2/01.png)![](https://www.mql5.com/ff/sh/20jc81m23z78s5z9z2/02.png)Create your own AI for tradingRead our book "Neural Networks in Algo Trading with MQL5"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/neurobook%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.neurobook%26utm_content=visit.page%26utm_campaign=neurobook.promo.04.2024&a=elbyupbppbqpzzvzhxtydvlupfcbmnmb&s=0d2f8feb92df3772a11aca1f195d2996b59d6539e283cdf4a18ccff02e5ad43d&uid=&ref=https://www.mql5.com/en/articles/16043&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5083304587865102573)

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