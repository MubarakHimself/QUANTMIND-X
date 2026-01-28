---
title: Connexus Helper (Part 5): HTTP Methods and Status Codes
url: https://www.mql5.com/en/articles/16136
categories: Integration, Expert Advisors
relevance_score: 0
scraped_at: 2026-01-24T14:03:01.876900
---

[![](https://www.mql5.com/ff/sh/0uquj7zv5pmx2m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
Market analytics in MQL5 Channels\\
\\
Tens of thousands of traders have chosen this messaging app to receive trading tips.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=epadtzgppsywkaeumqycnulasoijfbgz&s=9615c3e5c371aa0d7b34529539d05c10df73b35a1e2213e4ceee008933c7ede0&uid=&ref=https://www.mql5.com/en/articles/16136&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5083302891353020643)

MetaTrader 5 / Examples


### Introduction

This article is a continuation of a series of articles where we will build a library called Connexus. In the [first article](https://www.mql5.com/en/articles/15795), we understood the basic operation of the [WebRequest](https://www.mql5.com/en/docs/network/webrequest) function, understanding each of its parameters, and also created an example code that demonstrates the use of this function and its difficulties. In the last article, we understood how a request works, what the body of a request is, and how to send data to the server, and in the end, we developed support for requests with a body.

In this fifth article of the series on building the Connexus library, we will cover an essential aspect of the HTTP protocol: methods and status codes. Understanding how each HTTP verb works and how to handle status codes is crucial to creating reliable interactions between clients and servers. Let's go!

### HTTP Methods

HTTP methods are the actions we ask the server to perform. When you make an HTTP request, such as accessing a page or sending data, you are "talking" to the server using these verbs. Here are the main ones:

- GET: This is the good old “give me that”. The browser asks to see something on the server, be it a page, an image, or a file. It just gets the information, without changing anything. Like asking for the restaurant menu, just to see what's on offer.
- POST: POST is the guy who delivers a package. Here, you are sending data to the server. This is common in forms, like when you register on a website. Think of it as if you were sending a letter: you wait for it to arrive at its destination and do something there, like register you.
- PUT: When you use PUT, you are basically saying: “change this here for this new version”. It is used to update an existing resource. Like changing the oil in your car – it’s the same car, but now with something new.
- DELETE: Pretty straightforward, right? It’s “get that out of there.” You’re asking the server to delete something. Goodbye, see you never again.
- PATCH: PATCH is more delicate. It only changes part of the resource. It’s like fixing a broken part of a toy – you don’t have to change everything, just adjust what’s broken.
- HEAD: This is GET, but without the body. You only want the header information, not the content. It’s like reading the title of a book without opening the pages.

There are other methods such as CONNECT, OPTIONS, and TRACE, but they are rarely used in the day-to-day work of developers. I will not mention details of each one here, but my goal with the library is that it can support all HTTP methods. If you want to understand more about all HTTP methods, access the complete protocol documentation [here](https://www.mql5.com/go?link=https://datatracker.ietf.org/doc/html/rfc7231%23autoid-34 "https://datatracker.ietf.org/doc/html/rfc7231#autoid-34"). But trust me, the most common requests in the day-to-day work of developers, such as GET, POST, and DELETE, are enough for most problems.

I want to emphasize that we only use one method per request, that is, a request cannot be of the GET and POST types at the same time.

### Creating the CHttpMethod class

Now that we understand each HTTP method, what it is for and when to use each one, let's go straight to the code. We will implement in Connexus a class responsible for storing the HTTP method that will be used to make the requests. This will be a very simple class, regardless of your programming level, you will understand most of what is happening. The purpose of the class is only to store the method used, but since we are in a class, we will add more features to make it as easy as possible for the end user of the Connexus library.

Let's start by creating a new folder called Constants , inside it a new file called CHttpMethod , in the end the path will look like this: Include/Constants/HttpMethod.mqh . Inside this new file we will create the CHttpMethod class:

```
//+------------------------------------------------------------------+
//|                                                  HttpMethods.mqh |
//|                                  Copyright 2024, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
//+------------------------------------------------------------------+
//| class : CHttpMethods                                             |
//|                                                                  |
//| [PROPERTY]                                                       |
//| Name        : CHttpMethods                                       |
//| Heritage    : No heritage                                        |
//| Description : Saved http method.                                 |
//|                                                                  |
//+------------------------------------------------------------------+
class CHttpMethod
  {
public:
                     CHttpMethod(void);
                    ~CHttpMethod(void);
  };
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CHttpMethod::CHttpMethod(void)
  {
  }
//+------------------------------------------------------------------+
//| Destructor                                                       |
//+------------------------------------------------------------------+
CHttpMethod::~CHttpMethod(void)
  {
  }
//+------------------------------------------------------------------+
```

To contain all possible http methods, let's create an [enum](https://www.mql5.com/en/docs/basis/types/integer/enumeration) that will contain all possible methods of an http request. Let's also add to the class a new variable of the ENUM\_HTTP\_METHOD type, called m\_method:

```
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
enum ENUM_HTTP_METHOD
  {
   HTTP_METHOD_NULL = 0,   // [0] Null
   HTTP_METHOD_CONNECT,    // [1] Connect
   HTTP_METHOD_DELETE,     // [2] Delete
   HTTP_METHOD_GET,        // [3] Get
   HTTP_METHOD_HEAD,       // [4] Head
   HTTP_METHOD_OPTION,     // [5] Option
   HTTP_METHOD_PATCH,      // [6] Patch
   HTTP_METHOD_POST,       // [7] Post
   HTTP_METHOD_PUT,        // [8] Put
   HTTP_METHOD_TRACE,      // [9] Trace
  };
//+------------------------------------------------------------------+
//| class : CHttpMethods                                             |
//|                                                                  |
//| [PROPERTY]                                                       |
//| Name        : CHttpMethods                                       |
//| Heritage    : No heritage                                        |
//| Description : Saved http method.                                 |
//|                                                                  |
//+------------------------------------------------------------------+
class CHttpMethod
  {
private:
   ENUM_HTTP_METHOD  m_method;                           // Stores the method that will be used
  };
//+------------------------------------------------------------------+
```

Now that we have the variable that will be used for storage, let's create auxiliary methods to define and retrieve the HTTP method, in addition to overloading the = operator to make it easier to use, main functions:

- **Assignment operator (operator= )**: Allows you to define the HTTP method using the = operator directly.
- **SetMethod(ENUM\_HTTP\_METHOD) and SetMethod(string)**: Define the HTTP method by enum or string. When the parameter is of type string, it uses the [StringToUpper()](https://www.mql5.com/en/docs/strings/stringtoupper) function to form the string correctly.
- **GetMethod() and GetMethodDescription()**: Obtain the HTTP method and its textual description.
- **void operator=(ENUM\_HTTP\_METHOD method):** This is an overloaded operation, but in simple terms, it is used to define the HTTP method using the “=” operator. Here is an example of this operator in practice:



```
CHttpMethod method;
method = HTTP_METHOD_POST;
```

- **Verification functions (IsPost(), IsGet(), etc.)**: They facilitate the verification of the method, making the code more readable and simplified. An example of how these functions can help us



```
CHttpMethod method;
method.SetMethod("POST")

if(method.GetMethod() == HTTP_METHOD_POST)
    {
           //--- Action
    }
//--- Or
if(method.IsPost())
    {
           //--- Action
    }
```

This way we avoid explicit method comparisxfons.

The implementation of these methods is as follows:

```
//+------------------------------------------------------------------+
//| class : CHttpMethod                                              |
//|                                                                  |
//| [PROPERTY]                                                       |
//| Name        : CHttpMethod                                        |
//| Heritage    : No heritage                                        |
//| Description : Saved http method.                                 |
//|                                                                  |
//+------------------------------------------------------------------+
class CHttpMethod
  {
private:

   ENUM_HTTP_METHOD  m_method;                           // Stores the method that will be used

public:
                     CHttpMethod(void);
                    ~CHttpMethod(void);
   //--- Get and set
   void              operator=(ENUM_HTTP_METHOD method);
   void              Set(ENUM_HTTP_METHOD method);
   bool              Set(string method);
   ENUM_HTTP_METHOD  Get(void);
   string            GetAsString(void);

   //--- Check method
   bool              IsConnect(void);
   bool              IsGet(void);
   bool              IsPost(void);
   bool              IsPut(void);
   bool              IsDelete(void);
   bool              IsPatch(void);
   bool              IsHead(void);
   bool              IsOption(void);
   bool              IsTrace(void);
  };
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CHttpMethod::CHttpMethod(void)
  {
  }
//+------------------------------------------------------------------+
//| Destructor                                                       |
//+------------------------------------------------------------------+
CHttpMethod::~CHttpMethod(void)
  {
  }
//+------------------------------------------------------------------+
//|  Defines the http method                                         |
//+------------------------------------------------------------------+
void CHttpMethod::operator=(ENUM_HTTP_METHOD method)
  {
   m_method = method;
  }
//+------------------------------------------------------------------+
//|  Defines the http method                                         |
//+------------------------------------------------------------------+
void CHttpMethod::Set(ENUM_HTTP_METHOD method)
  {
   m_method = method;
  }
//+------------------------------------------------------------------+
//|  Defines the http method                                         |
//+------------------------------------------------------------------+
bool CHttpMethod::Set(string method)
  {
   string method_upper = method;
   StringToUpper(method_upper);
   if(method_upper == "CONNECT")
     {
      m_method = HTTP_METHOD_CONNECT;
      return(true);
     }
   else if(method_upper == "DELETE")
     {
      m_method = HTTP_METHOD_DELETE;
      return(true);
     }
   else if(method_upper == "GET")
     {
      m_method = HTTP_METHOD_GET;
      return(true);
     }
   else if(method_upper == "HEAD")
     {
      m_method = HTTP_METHOD_HEAD;
      return(true);
     }
   else if(method_upper == "OPTIONS")
     {
      m_method = HTTP_METHOD_OPTION;
      return(true);
     }
   else if(method_upper == "PATCH")
     {
      m_method = HTTP_METHOD_PATCH;
      return(true);
     }
   else if(method_upper == "POST")
     {
      m_method = HTTP_METHOD_POST;
      return(true);
     }
   else if(method_upper == "PUT")
     {
      m_method = HTTP_METHOD_PUT;
      return(true);
     }
   else if(method_upper == "TRACE")
     {
      m_method = HTTP_METHOD_TRACE;
      return(true);
     }
   return(false);
  }
//+------------------------------------------------------------------+
//| Get http method                                                  |
//+------------------------------------------------------------------+
ENUM_HTTP_METHOD CHttpMethod::Get(void)
  {
   return(m_method);
  }
//+------------------------------------------------------------------+
//| Get the description of the selected http method                  |
//+------------------------------------------------------------------+
string CHttpMethod::GetAsString(void)
  {
   switch(m_method)
     {
      case HTTP_METHOD_NULL:
         return "NULL";
      case HTTP_METHOD_CONNECT:
         return "CONNECT";
      case HTTP_METHOD_DELETE:
         return "DELETE";
      case HTTP_METHOD_GET:
         return "GET";
      case HTTP_METHOD_HEAD:
         return "HEAD";
      case HTTP_METHOD_OPTION:
         return "OPTIONS";
      case HTTP_METHOD_PATCH:
         return "PATCH";
      case HTTP_METHOD_POST:
         return "POST";
      case HTTP_METHOD_PUT:
         return "PUT";
      case HTTP_METHOD_TRACE:
         return "TRACE";

      default:
         return "Unknown HTTP Method";
     }
  }
//+------------------------------------------------------------------+
//| Check if method is connect                                       |
//+------------------------------------------------------------------+
bool CHttpMethod::IsConnect(void)
  {
   return(m_method == HTTP_METHOD_CONNECT);
  }
//+------------------------------------------------------------------+
//| Check if method is get                                           |
//+------------------------------------------------------------------+
bool CHttpMethod::IsGet(void)
  {
   return(m_method == HTTP_METHOD_GET);
  }
//+------------------------------------------------------------------+
//| Check if method is post                                          |
//+------------------------------------------------------------------+
bool CHttpMethod::IsPost(void)
  {
   return(m_method == HTTP_METHOD_POST);
  }
//+------------------------------------------------------------------+
//| Check if method is put                                           |
//+------------------------------------------------------------------+
bool CHttpMethod::IsPut(void)
  {
   return(m_method == HTTP_METHOD_PUT);
  }
//+------------------------------------------------------------------+
//| Check if method is delete                                        |
//+------------------------------------------------------------------+
bool CHttpMethod::IsDelete(void)
  {
   return(m_method == HTTP_METHOD_DELETE);
  }
//+------------------------------------------------------------------+
//| Check if method is patch                                         |
//+------------------------------------------------------------------+
bool CHttpMethod::IsPatch(void)
  {
   return(m_method == HTTP_METHOD_PATCH);
  }
//+------------------------------------------------------------------+
//| Check if method is head                                          |
//+------------------------------------------------------------------+
bool CHttpMethod::IsHead(void)
  {
   return(m_method == HTTP_METHOD_HEAD);
  }
//+------------------------------------------------------------------+
//| Check if method is option                                        |
//+------------------------------------------------------------------+
bool CHttpMethod::IsOption(void)
  {
   return(m_method == HTTP_METHOD_OPTION);
  }
//+------------------------------------------------------------------+
//| Check if method is trace                                         |
//+------------------------------------------------------------------+
bool CHttpMethod::IsTrace(void)
  {
   return(m_method == HTTP_METHOD_TRACE);
  }
//+------------------------------------------------------------------+
```

This concludes the class responsible for storing the HTTP method used. We have some auxiliary functions to save some code in the future. Let's move on to the next auxiliary class.

### Status Code

Status codes, quite simply, are numbers. This number is standardized and is sent by the server to the client after processing a request. Each code has three digits, the first indicates the category to which the status belongs (1xx, 2xx, 3xx, 4xx and 5xx). This code simply tells you what the result of the request was, whether it was completed successfully, whether an error was generated on the server side or whether the request was sent incorrectly. There are several status codes. This number can vary between 100 and 599, and they are separated into 5 categories. To find out which category the code is in, simply identify which range it is in, see the table with the values:

| Range status code | Description |
| --- | --- |
| 100-199 | It's not very common to see. They are responses like "I'm processing, please wait". It's the server saying that it's working on your request, but it's not finished yet. |
| 200-299 | Ah, this is the answer we like the most! It means that everything went well. The most famous is the 200 OK – a sign that “everything went well”. You asked for it, it delivered. Simple as that. |
| 300-399 | The server is basically telling you, “Oops, you’re in the wrong place, go there.” 301 Moved Permanently is the permanent redirect, while 302 Found is temporary – something like, “I’m under renovation, but you can find this here for now.” |
| 400-499 | These are the infamous errors that we, as users, often cause. The most well-known is 404 Not Found, when the page you are looking for simply does not exist. It is like arriving at an address and discovering that the building has been demolished. |
| 500-599 | Now, when the problem is on the other side, the blame falls on the server. The famous 500 Internal Server Error is basically the server throwing its hands up and saying “something went wrong here”. |

There are several possible codes, I won't cover them all so as not to make the article too long. If you want to learn more about all the possible values, read [here](https://www.mql5.com/go?link=https://datatracker.ietf.org/doc/html/rfc7231%23autoid-58 "https://datatracker.ietf.org/doc/html/rfc7231#autoid-58"), it explains each possible code in detail, it's worth reading if you want to learn more about the HTTP protocol. In the day-to-day life of a developer, only a few codes are used, and most of them are rarely found on the web. I'll list the most common ones:

1xx: Informational

- 100 Continue: The server received the headers and the client can continue sending the request body.
- 101 Switching Protocols: The client requested a protocol change, and the server accepted.

2xx: Success

- 200 OK: The request was successful.
- 201 Created: The request was successful and a new resource was created. - **204 No Content**: The request was successful, but there is no content in the response body.

3xx: Redirection

- 301 Moved Permanently: The resource has been permanently moved to a new URL.
- 302 Found: The resource has been temporarily moved to a new URL.
- 304 Not Modified: The resource has not been modified since the last request, allowing the client to use the cached version.

4xx: Client errors

- 400 Bad Request: The request is invalid or malformed.
- 401 Unauthorized: Access is not authorized; authentication is required.
- 403 Forbidden: The server understood the request, but denies access.
- 404 Not Found: The requested resource was not found.
- 405 Method Not Allowed: The HTTP method used is not allowed for the requested resource.

5xx: Server Errors

- 500 Internal Server Error: A generic error occurred on the server.
- 502 Bad Gateway: The server received an invalid response while trying to fulfill the request.
- 503 Service Unavailable: The server is temporarily unavailable, usually due to overload or maintenance.
- 504 Gateway Timeout: The server did not receive a response in time from another server it was trying to connect to.

Now that we understand all the status categories, and the most commonly used ones, I want the library to support all possible statuses. To do this, we will work on a class capable of processing any status code received by the server, as long as it is valid.

### Creating the CHttpStatusCode class

Now that we know all the possible status codes, let's add the appropriate implementation to the library. The goal here is simple: create a class that will be responsible for storing the received status code. The class should also contain support for describing each status and it should also be possible to quickly identify which category that status belongs to.

Let's go to the code. Within this same folder previously created for the CHttpMethod class, we will create a new file called HttpStatusCode.mqh . At the end, the full path will be Includes/Connexus/Constants/HttpStatusCode.mqh .

```
//+------------------------------------------------------------------+
//|                                               HttpStatusCode.mqh |
//|                                  Copyright 2024, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
//+------------------------------------------------------------------+
//| class : CHttpStatusCodes                                         |
//|                                                                  |
//| [PROPERTY]                                                       |
//| Name        : CHttpStatusCodes                                   |
//| Heritage    : No heritage                                        |
//| Description : Saved http status code.                            |
//|                                                                  |
//+------------------------------------------------------------------+
class CHttpStatusCodes
  {
public:
                     CHttpStatusCodes(void);
                    ~CHttpStatusCodes(void);
  };
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CHttpStatusCodes::CHttpStatusCodes(void)
  {
  }
//+------------------------------------------------------------------+
//| Destructor                                                       |
//+------------------------------------------------------------------+
CHttpStatusCodes::~CHttpStatusCodes(void)
  {
  }
//+------------------------------------------------------------------+
```

To contain all possible status codes, we will create an enum with all possible values. The name of the enum will be ENUM\_HTTP\_STATUS, the value of the enum will be the respective status code.

```
enum ENUM_HTTP_STATUS
  {
//--- Mql error
   HTTP_STATUS_URL_NOT_ALLOWED = -1,
   HTTP_STATUS_URL_NOT_DEFINED = 1,
   HTTP_STATUS_METHOD_NOT_DEFINED = 2,

//--- Informational
   HTTP_STATUS_CONTINUE = 100,
   HTTP_STATUS_SWITCHING_PROTOCOLS = 101,
   HTTP_STATUS_PROCESSING = 102,
   HTTP_STATUS_EARLY_HINTS = 103,

//--- Successul
   HTTP_STATUS_OK = 200,
   HTTP_STATUS_CREATED = 201,
   HTTP_STATUS_ACCEPTED = 202,
   HTTP_STATUS_NON_AUTHORITATIVE_INFORMATION = 203,
   HTTP_STATUS_NO_CONTENT = 204,
   HTTP_STATUS_RESET_CONTENT = 205,
   HTTP_STATUS_PARTIAL_CONTENT = 206,
   HTTP_STATUS_MULTI_STATUS = 207,
   HTTP_STATUS_ALREADY_REPORTED = 208,

//--- Redirection messages
   HTTP_STATUS_MULTIPLE_CHOICES = 300,
   HTTP_STATUS_MOVED_PERMANENTLY = 301,
   HTTP_STATUS_FOUND = 302,
   HTTP_STATUS_SEE_OTHER = 303,
   HTTP_STATUS_NOT_MODIFIED = 304,
   HTTP_STATUS_USE_PROXY = 305,
   HTTP_STATUS_SWITCH_PROXY = 306,
   HTTP_STATUS_TEMPORARY_REDIRECT = 307,
   HTTP_STATUS_PERMANENT_REDIRECT = 308,

//--- Client error
   HTTP_STATUS_BAD_REQUEST = 400,
   HTTP_STATUS_UNAUTHORIZED = 401,
   HTTP_STATUS_PAYMENT_REQUIRED = 402,
   HTTP_STATUS_FORBIDDEN = 403,
   HTTP_STATUS_NOT_FOUND = 404,
   HTTP_STATUS_METHOD_NOT_ALLOWED = 405,
   HTTP_STATUS_NOT_ACCEPTABLE = 406,
   HTTP_STATUS_PROXY_AUTHENTICATION_REQUIRED = 407,
   HTTP_STATUS_REQUEST_TIMEOUT = 408,
   HTTP_STATUS_CONFLICT = 409,
   HTTP_STATUS_GONE = 410,
   HTTP_STATUS_LENGTH_REQUIRED = 411,
   HTTP_STATUS_PRECONDITION_FAILED = 412,
   HTTP_STATUS_PAYLOAD_TOO_LARGE = 413,
   HTTP_STATUS_URI_TOO_LONG = 414,
   HTTP_STATUS_UNSUPPORTED_MEDIA_TYPE = 415,
   HTTP_STATUS_RANGE_NOT_SATISFIABLE = 416,
   HTTP_STATUS_EXPECTATION_FAILED = 417,
   HTTP_STATUS_MISDIRECTED_REQUEST = 421,
   HTTP_STATUS_UNPROCESSABLE_ENTITY = 422,
   HTTP_STATUS_LOCKED = 423,
   HTTP_STATUS_FAILED_DEPENDENCY = 424,
   HTTP_STATUS_TOO_EARLY = 425,
   HTTP_STATUS_UPGRADE_REQUIRED = 426,
   HTTP_STATUS_PRECONDITION_REQUIRED = 428,
   HTTP_STATUS_TOO_MANY_REQUESTS = 429,
   HTTP_STATUS_REQUEST_HEADER_FIELDS_TOO_LARGE = 431,
   HTTP_STATUS_UNAVAILABLE_FOR_LEGAL_REASONS = 451,

//--- Server error
   HTTP_STATUS_INTERNAL_SERVER_ERROR = 500,
   HTTP_STATUS_NOT_IMPLEMENTED = 501,
   HTTP_STATUS_BAD_GATEWAY = 502,
   HTTP_STATUS_SERVICE_UNAVAILABLE = 503,
   HTTP_STATUS_GATEWAY_TIMEOUT = 504,
   HTTP_STATUS_HTTP_VERSION_NOT_SUPPORTED = 505,
   HTTP_STATUS_VARIANT_ALSO_NEGOTIATES = 506,
   HTTP_STATUS_INSUFFICIENT_STORAGE = 507,
   HTTP_STATUS_LOOP_DETECTED = 508,
   HTTP_STATUS_NOT_EXTENDED = 510,
   HTTP_STATUS_NETWORK_AUTHENTICATION_REQUIRED = 511
  };
```

Please note that I added some comments separating the categories of each status. Note the first values of the enum, I added these values as a “custom status code”, which will be automatically generated by the library, they are:

- HTTP\_STATUS\_URL\_NOT\_ALLOWED: The URL was not added to the list of allowed URLs in the terminal
- HTTP\_STATUS\_URL\_NOT\_DEFINED: The URL was not defined in the request
- HTTP\_STATUS\_METHOD\_NOT\_DEFINED: A valid method was not defined for the request

When the WebRequest function is called and returns -1, it means that the URL is not added to the terminal, as stated in the MQL5 documentation, so the library should automatically return HTTP\_STATUS\_URL\_NOT\_ALLOWED . Similar logic will be applied to the other custom codes, but we will not focus on that at the moment.

Let's continue with the development of the class, this class will be similar to CHttpMethod , we will add a new private variable called m\_status of type ENUM\_HTTP\_STATUS, and we will also add some auxiliary methods to set and get the value of this variable:

- **operator=(int) and operator=(ENUM\_HTTP\_STATUS)**: Set the value of m\_status using the = operator, receiving an integer or an enum value.
- **Set(ENUM\_HTTP\_STATUS)**: Sets the status code using an enum value.
- **Get() and GetMessage():** Returns the current status code or only the message corresponding to the stored code.

Below is the code with the implementation of these methods:

```
//+------------------------------------------------------------------+
//| class : CHttpStatusCodes                                         |
//|                                                                  |
//| [PROPERTY]                                                       |
//| Name        : CHttpStatusCodes                                   |
//| Heritage    : No heritage                                        |
//| Description : Saved http status code.                            |
//|                                                                  |
//+------------------------------------------------------------------+
class CHttpStatusCodes
  {
private:

   ENUM_HTTP_STATUS  m_status;                           // Stores the status used

public:
                     CHttpStatusCodes(void);
                    ~CHttpStatusCodes(void);

   //--- Set
   void              operator=(int status);
   void              operator=(ENUM_HTTP_STATUS status);
   void              Set(ENUM_HTTP_STATUS status);

   //--- Get
   ENUM_HTTP_STATUS  Get(void);
   string            GetMessage(void);
  };
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CHttpStatusCodes::CHttpStatusCodes(void)
  {
  }
//+------------------------------------------------------------------+
//| Destructor                                                       |
//+------------------------------------------------------------------+
CHttpStatusCodes::~CHttpStatusCodes(void)
  {
  }

//+------------------------------------------------------------------+
//| Assignment operator to set status from integer value             |
//+------------------------------------------------------------------+
void CHttpStatusCodes::operator=(int status)
  {
   m_status = (ENUM_HTTP_STATUS)status;
  }

//+------------------------------------------------------------------+
//| Assignment operator to set status from ENUM_HTTP_STATUS          |
//+------------------------------------------------------------------+
void CHttpStatusCodes::operator=(ENUM_HTTP_STATUS status)
  {
   m_status = status;
  }

//+------------------------------------------------------------------+
//| Sets the HTTP status code                                        |
//+------------------------------------------------------------------+
void CHttpStatusCodes::Set(ENUM_HTTP_STATUS status)
  {
   m_status = status;
  }

//+------------------------------------------------------------------+
//| Returns the stored HTTP status code                              |
//+------------------------------------------------------------------+
ENUM_HTTP_STATUS CHttpStatusCodes::Get(void)
  {
   return(m_status);
  }

//+------------------------------------------------------------------+
//| Returns a message corresponding to the stored HTTP status code   |
//+------------------------------------------------------------------+
string CHttpStatusCodes::GetMessage(void)
  {
   switch(m_status)
     {
      case HTTP_STATUS_URL_NOT_ALLOWED:
         return "The URL was not added to the list of allowed URLs in the terminal";
      case HTTP_STATUS_URL_NOT_DEFINED:
         return "URL was not defined in the request";
      case HTTP_STATUS_METHOD_NOT_DEFINED:
         return "Method was not defined in the request";

      case HTTP_STATUS_CONTINUE:
         return "Continue";
      case HTTP_STATUS_SWITCHING_PROTOCOLS:
         return "Switching Protocols";
      case HTTP_STATUS_PROCESSING:
         return "Processing";
      case HTTP_STATUS_EARLY_HINTS:
         return "Early Hints";

      case HTTP_STATUS_OK:
         return "OK";
      case HTTP_STATUS_CREATED:
         return "Created";
      case HTTP_STATUS_ACCEPTED:
         return "Accepted";
      case HTTP_STATUS_NON_AUTHORITATIVE_INFORMATION:
         return "Non-Authoritative Information";
      case HTTP_STATUS_NO_CONTENT:
         return "No Content";
      case HTTP_STATUS_RESET_CONTENT:
         return "Reset Content";
      case HTTP_STATUS_PARTIAL_CONTENT:
         return "Partial Content";
      case HTTP_STATUS_MULTI_STATUS:
         return "Multi-Status";
      case HTTP_STATUS_ALREADY_REPORTED:
         return "Already Reported";

      case HTTP_STATUS_MULTIPLE_CHOICES:
         return "Multiple Choices";
      case HTTP_STATUS_MOVED_PERMANENTLY:
         return "Moved Permanently";
      case HTTP_STATUS_FOUND:
         return "Found";
      case HTTP_STATUS_SEE_OTHER:
         return "See Other";
      case HTTP_STATUS_NOT_MODIFIED:
         return "Not Modified";
      case HTTP_STATUS_USE_PROXY:
         return "Use Proxy";
      case HTTP_STATUS_SWITCH_PROXY:
         return "Switch Proxy";
      case HTTP_STATUS_TEMPORARY_REDIRECT:
         return "Temporary Redirect";
      case HTTP_STATUS_PERMANENT_REDIRECT:
         return "Permanent Redirect";

      case HTTP_STATUS_BAD_REQUEST:
         return "Bad Request";
      case HTTP_STATUS_UNAUTHORIZED:
         return "Unauthorized";
      case HTTP_STATUS_PAYMENT_REQUIRED:
         return "Payment Required";
      case HTTP_STATUS_FORBIDDEN:
         return "Forbidden";
      case HTTP_STATUS_NOT_FOUND:
         return "Not Found";
      case HTTP_STATUS_METHOD_NOT_ALLOWED:
         return "Method Not Allowed";
      case HTTP_STATUS_NOT_ACCEPTABLE:
         return "Not Acceptable";
      case HTTP_STATUS_PROXY_AUTHENTICATION_REQUIRED:
         return "Proxy Authentication Required";
      case HTTP_STATUS_REQUEST_TIMEOUT:
         return "Request Timeout";
      case HTTP_STATUS_CONFLICT:
         return "Conflict";
      case HTTP_STATUS_GONE:
         return "Gone";
      case HTTP_STATUS_LENGTH_REQUIRED:
         return "Length Required";
      case HTTP_STATUS_PRECONDITION_FAILED:
         return "Precondition Failed";
      case HTTP_STATUS_PAYLOAD_TOO_LARGE:
         return "Payload Too Large";
      case HTTP_STATUS_URI_TOO_LONG:
         return "URI Too Long";
      case HTTP_STATUS_UNSUPPORTED_MEDIA_TYPE:
         return "Unsupported Media Type";
      case HTTP_STATUS_RANGE_NOT_SATISFIABLE:
         return "Range Not Satisfiable";
      case HTTP_STATUS_EXPECTATION_FAILED:
         return "Expectation Failed";
      case HTTP_STATUS_MISDIRECTED_REQUEST:
         return "Misdirected Request";
      case HTTP_STATUS_UNPROCESSABLE_ENTITY:
         return "Unprocessable Entity";
      case HTTP_STATUS_LOCKED:
         return "Locked";
      case HTTP_STATUS_FAILED_DEPENDENCY:
         return "Failed Dependency";
      case HTTP_STATUS_TOO_EARLY:
         return "Too Early";
      case HTTP_STATUS_UPGRADE_REQUIRED:
         return "Upgrade Required";
      case HTTP_STATUS_PRECONDITION_REQUIRED:
         return "Precondition Required";
      case HTTP_STATUS_TOO_MANY_REQUESTS:
         return "Too Many Requests";
      case HTTP_STATUS_REQUEST_HEADER_FIELDS_TOO_LARGE:
         return "Request Header Fields Too Large";
      case HTTP_STATUS_UNAVAILABLE_FOR_LEGAL_REASONS:
         return "Unavailable For Legal Reasons";

      case HTTP_STATUS_INTERNAL_SERVER_ERROR:
         return "Internal Server Error";
      case HTTP_STATUS_NOT_IMPLEMENTED:
         return "Not Implemented";
      case HTTP_STATUS_BAD_GATEWAY:
         return "Bad Gateway";
      case HTTP_STATUS_SERVICE_UNAVAILABLE:
         return "Service Unavailable";
      case HTTP_STATUS_GATEWAY_TIMEOUT:
         return "Gateway Timeout";
      case HTTP_STATUS_HTTP_VERSION_NOT_SUPPORTED:
         return "HTTP Version Not Supported";
      case HTTP_STATUS_VARIANT_ALSO_NEGOTIATES:
         return "Variant Also Negotiates";
      case HTTP_STATUS_INSUFFICIENT_STORAGE:
         return "Insufficient Storage";
      case HTTP_STATUS_LOOP_DETECTED:
         return "Loop Detected";
      case HTTP_STATUS_NOT_EXTENDED:
         return "Not Extended";
      case HTTP_STATUS_NETWORK_AUTHENTICATION_REQUIRED:
         return "Network Authentication Required";
      default:
         return "Unknown HTTP Status";
     }
  }
//+------------------------------------------------------------------+
```

Let's add some more auxiliary methods, which will be responsible for checking if the status code stored in the class is in a specific category, they are:

- bool IsInformational(void): Informational range (100 - 199)
- bool IsSuccess(void): Success range (200 - 299)
- bool IsRedirection(void): Redirection range (300 - 399)
- bool IsClientError(void): Client error range (400 - 499)
- bool IsServerError(void): Server error range (500 - 599)

```
//+------------------------------------------------------------------+
//| class : CHttpStatusCodes                                         |
//|                                                                  |
//| [PROPERTY]                                                       |
//| Name        : CHttpStatusCodes                                   |
//| Heritage    : No heritage                                        |
//| Description : Saved http status code.                            |
//|                                                                  |
//+------------------------------------------------------------------+
class CHttpStatusCodes
  {
private:

   ENUM_HTTP_STATUS  m_status;                           // Stores the status used

public:
                     CHttpStatusCodes(void);
                    ~CHttpStatusCodes(void);

   //--- Check which group the code is in
   bool              IsInformational(void);              // Checks if the status code is in the informational response range (100 - 199)
   bool              IsSuccess(void);                    // Check if the status code is in the success range (200 - 299)
   bool              IsRedirection(void);                // Check if the status code is in the redirect range (300 - 399)
   bool              IsClientError(void);                // Checks if the status code is in the client error range (400 - 499)
   bool              IsServerError(void);                // Check if the status code is in the server error range (500 - 599)
  };
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CHttpStatusCodes::CHttpStatusCodes(void)
  {
  }
//+------------------------------------------------------------------+
//| Destructor                                                       |
//+------------------------------------------------------------------+
CHttpStatusCodes::~CHttpStatusCodes(void)
  {
  }
//+------------------------------------------------------------------+
//| Checks if the status code is in the informational response       |
//| range (100 - 199)                                                |
//+------------------------------------------------------------------+
bool CHttpStatusCodes::IsInformational(void)
  {
   return(m_status >= 100 && m_status <= 199);
  }
//+------------------------------------------------------------------+
//| Check if the status code is in the success range (200 - 299)     |
//+------------------------------------------------------------------+
bool CHttpStatusCodes::IsSuccess(void)
  {
   return(m_status >= 200 && m_status <= 299);
  }
//+------------------------------------------------------------------+
//| Check if the status code is in the redirect range (300 - 399)    |
//+------------------------------------------------------------------+
bool CHttpStatusCodes::IsRedirection(void)
  {
   return(m_status >= 300 && m_status <= 399);
  }
//+------------------------------------------------------------------+
//| Checks if the status code is in the client error range           |
//| (400 - 499)                                                      |
//+------------------------------------------------------------------+
bool CHttpStatusCodes::IsClientError(void)
  {
   return(m_status >= 400 && m_status <= 499);
  }
//+------------------------------------------------------------------+
//| Check if the status code is in the server error range (500 - 599)|
//+------------------------------------------------------------------+
bool CHttpStatusCodes::IsServerError(void)
  {
   return(m_status >= 500 && m_status <= 599);
  }
//+------------------------------------------------------------------+
```

### Conclusion

To make the library's progress more visual, follow the diagram below:

![Diagram](https://c.mql5.com/2/98/diagram1__2.png)

We have all the auxiliary classes ready, which are responsible for handling each HTTP element independently. We have the CQueryParam , CHttpHeader and CHttpBody classes using the CJson class, but there is no inheritance relationship between them. The classes created in this article have not yet been connected to the others, in the next article we will connect everything by creating an HTTP request and response.

In this article, we understand the HTTP methods and also the status codes, two very important pieces for web communication between client and server. Understanding what each method does gives you the control to make requests more precisely, informing the server what action you want to perform and making it more efficient. Each method has a role in communication and using it correctly makes communication with the API clearer, both for the client and the server, without surprises.

In addition, we talk about status codes, which are the server's direct response about what happened with the request. These range from a simple “all good” (200 OK) to client-side (4xx) or server-side (5xx) error messages. Knowing how to handle these codes is a valuable skill, because often an error does not mean the end of the line, but rather an opportunity to adjust or try again.

During this process of building the Connexus library, we learned how to handle each of these elements, which makes our library more robust and capable of handling the nuances of HTTP communication. From here, our class responsible for the methods and status codes will be ready to give the developer a greater level of control and security when interacting with APIs.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/16136.zip "Download all attachments in the single ZIP archive")

[Connexus\_Helper\_jPart\_5z\_HTTP\_Methods\_and\_Status\_Codes.zip](https://www.mql5.com/en/articles/download/16136/connexus_helper_jpart_5z_http_methods_and_status_codes.zip "Download Connexus_Helper_jPart_5z_HTTP_Methods_and_Status_Codes.zip")(24.99 KB)

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

**[Go to discussion](https://www.mql5.com/en/forum/475445)**

![Neural Networks Made Easy (Part 90): Frequency Interpolation of Time Series (FITS)](https://c.mql5.com/2/78/Neural_networks_are_easy_tPart_90x__LOGO.png)[Neural Networks Made Easy (Part 90): Frequency Interpolation of Time Series (FITS)](https://www.mql5.com/en/articles/14913)

By studying the FEDformer method, we opened the door to the frequency domain of time series representation. In this new article, we will continue the topic we started. We will consider a method with which we can not only conduct an analysis, but also predict subsequent states in a particular area.

![Neural Network in Practice: Straight Line Function](https://c.mql5.com/2/78/Rede_neural_na_prdtica_Fundso_de_reta____LOGO2.png)[Neural Network in Practice: Straight Line Function](https://www.mql5.com/en/articles/13696)

In this article, we will take a quick look at some methods to get a function that can represent our data in the database. I will not go into detail about how to use statistics and probability studies to interpret the results. Let's leave that for those who really want to delve into the mathematical side of the matter. Exploring these questions will be critical to understanding what is involved in studying neural networks. Here we will consider this issue quite calmly.

![Artificial Cooperative Search (ACS) algorithm](https://c.mql5.com/2/79/Artificial_Cooperative_Search____LOGO__1.png)[Artificial Cooperative Search (ACS) algorithm](https://www.mql5.com/en/articles/15004)

Artificial Cooperative Search (ACS) is an innovative method using a binary matrix and multiple dynamic populations based on mutualistic relationships and cooperation to find optimal solutions quickly and accurately. ACS unique approach to predators and prey enables it to achieve excellent results in numerical optimization problems.

![MQL5 Wizard Techniques you should know (Part 44): Average True Range (ATR) technical indicator](https://c.mql5.com/2/99/MQL5_Wizard_Techniques_you_should_know_Part_44___LOGO.png)[MQL5 Wizard Techniques you should know (Part 44): Average True Range (ATR) technical indicator](https://www.mql5.com/en/articles/16213)

The ATR oscillator is a very popular indicator for acting as a volatility proxy, especially in the forex markets where volume data is scarce. We examine this, on a pattern basis as we have with prior indicators, and share strategies & test reports thanks to the MQL5 wizard library classes and assembly.

[![](https://www.mql5.com/ff/sh/9nb0c8df2rmwfn89z2/01.png) MetaTrader VPS vs regular cloud hosting services8 reasons why our solution is the best option for automated tradingRead](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/450486&a=dgmsfszgoedimaicrqqmagvqzpwuxkur&s=c59e3617ccf44fd54d4c50a03b44fd689ff7507b8fe4990c83772cc5419e627d&uid=&ref=https://www.mql5.com/en/articles/16136&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5083302891353020643)

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