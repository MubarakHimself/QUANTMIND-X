---
title: Requesting in Connexus (Part 6): Creating an HTTP Request and Response
url: https://www.mql5.com/en/articles/16182
categories: Integration, Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T21:08:38.317176
---

[![](https://www.mql5.com/ff/sh/wm94j0jmkwd29943z2/ddfa713cb3cdd580c3e81e0e13b5b1b8.jpg)\\
Revised MetaTrader 5 Web Terminal\\
\\
Trade with no restrictions from any mobile device, OS and web browser\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=fkjlpstbxdmrrwpblfatcsdjyrxbizyj&s=f462f051eb7aaec36d6b31792d312d60d3f5a50c83b12d0d66e85d5d61bd941b&uid=&ref=https://www.mql5.com/en/articles/16182&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5071607493542488868)

MetaTrader 5 / Examples


### Introduction

This article is part of an ongoing series where we are building a library called Connexus. In the [first article](https://www.mql5.com/en/articles/15795), we covered the basic functionality of the [WebRequest](https://www.mql5.com/en/docs/network/webrequest) function, understanding each of its parameters and creating a sample code that demonstrates its use and the challenges associated with it. In the previous article, we explored what HTTP methods are and the status codes returned by the server, indicating whether the request was successfully processed or if there was an error on the client or server side.

In this sixth article of the Connexus library series, we will focus on a complete HTTP request, covering each component that makes up a request. We will create a class that represents the request as a whole, which will help us bring together the previously created classes. Additionally, we will develop a similar class to handle the server’s response, containing the response data, status code, and even the duration of the request.

In HTTP communication, a request is composed of several components that together form the complete request. We have already explored all of these components in previous articles and created individual classes for each element of the request. Let's recap the elements below:

- **URL**: Defines the address of the server on the web and is composed of smaller parts such as domain, port, path, etc. This was covered in more detail in [Part 2](https://www.mql5.com/en/articles/15897) of Connexus, where we created a class to properly format a URL.
- **Header**: These are additional data sent with the request, intended to provide details about the request that are not part of the body or the URL. This was covered in more detail in [Part 3](https://www.mql5.com/en/articles/16043) of Connexus, where we also created a class responsible for organizing the request header.
- **Body**: Refers to the actual content being sent or received. Simply put, the body is where the data that interests us is stored, which we want to send to the server. In [Part 4](https://www.mql5.com/en/articles/16098), we discussed this in more detail and also created a class responsible for storing the request body, which supports adding the body in different formats, such as plain text (string), JSON, or char\[\] (binary).
- **HTTP Method**: HTTP methods are used by the client to tell the server what action it wants to perform. This was discussed in more detail in [Part 5](https://www.mql5.com/en/articles/16136) of Connexus.
- **Timeout**: Timeout was not covered in previous articles, but to explain briefly, it is the time in milliseconds the server has to respond. If the request takes longer than the allotted time, it is terminated, resulting in a timeout error. This value can be useful to avoid scenarios where the server takes too long to respond, as the WebRequest function is synchronous, meaning the Expert Advisor remains "paused" while waiting for the server's response. This could be problematic if the request takes much longer than expected. To avoid this scenario, it's recommended to use a timeout. The timeout value can vary depending on your needs, but for most servers, 5000 milliseconds (5 seconds) is sufficient for the request to be processed correctly.

Each of these components plays a fundamental role in constructing a proper HTTP request, but it's important to note that not all of them are mandatory. The only required elements are the URL and the HTTP method (GET, POST, PUT, etc.).

To help guide us on the progress of the library and the classes we've already created, let's take a look at the current diagram of the library:

![current diagram](https://c.mql5.com/2/99/diagram1.png)

Note that we already have classes ready to handle HTTP Methods, Status Codes, URLs with query parameters, Headers, and the Body.

### Facade Design Pattern

To continue building the library, we will implement a design pattern. If you're not familiar with design patterns, I recommend reading the series of articles [Design Patterns in Software Development and MQL5](https://www.mql5.com/en/articles/13724), written by [Mohamed Abdelmaaboud](https://www.mql5.com/en/users/m.aboud). In that series, the author describes several design patterns with sample code. For the Connexus library, we will implement the "Facade" design pattern, which is a well-known pattern in programming. **This pattern provides a simplified interface to a more complex set of subsystems, hiding the internal complexity and allowing the client to interact with the system in a much simpler way.**

Let's imagine an example in the context of the library: you want to create a request. To do so, you would need to create an instance of each of the request elements and configure them, something like this:

```
CURL m_url;
m_url.Parse("http://example.com/api");

CHttpBody m_body;
m_body.AddString("my text");

CHttpHeader m_headers;
m_headers.Add("content-type","text/plain");

CHttpMethod m_method;
m_method = HTTP_METHOD_GET;

int m_timeout = 5000;
```

This approach makes the code cluttered, requiring the creation of multiple instances, consuming many lines, and making the code less readable. The situation worsens when working with numerous requests, as managing multiple objects like headers, body, and URL becomes complicated and difficult to maintain. This is where the **Facade** design pattern comes into play. Going back to the concept: **This pattern provides a simplified interface to a more complex set of subsystems, hiding internal complexity and allowing the client to interact with the system in a much simpler way**.

In this context, the subsystems are the classes for request elements such as CHttpBody , CHttpHeaders , etc., and the goal is to create a more intuitive interface for them. This pattern solves the problem by introducing a class or interface that acts as a "Facade" to access the subsystems. The final developer interacts only with this simplified interface.

In summary, the _Facade_ architecture offers the following benefits:

1. **Simplified Interaction**: Instead of dealing directly with a series of complex classes, the developer can use a simplified interface that hides the internal details.
2. **Reduced Coupling**: Since the client is not directly coupled to the internal subsystems, changes in these subsystems can be made without affecting the client.
3. **Improved Maintainability**: The clear separation between the _Facade_ interface and the internal subsystems makes the code easier to maintain and expand, as any internal changes can be abstracted by the facade.

How would this design pattern be implemented in the library? To achieve this, we will create a new class called CHttpRequest , which will contain the subsystems. For the library user, the use of this class should look something like this:

```
CHttpRequest request;

request.Method() = HTTP_METHOD_GET;
request.Url().Parse("http://example.com/api");
request.Body().AddString("my text");
request.Header().Add("content-type","text/plain");
```

Notice how the code has become much simpler and more readable, which is exactly the idea behind this design pattern. The Facade pattern provides a simplified interface (CHttpRequest) for a more complex set of subsystems (managing instances of CHttpBody, CHttpHeader, etc.), hiding the internal complexity and allowing the client to interact with the system in a much simpler way.

### Creating the CHttpRequest Class

Now that we understand the Facade concept, we can apply this architecture to the CHttpRequest class. Let's create this class within a new folder called core. The file will be named HttpRequest.mqh, and the final path will be Include/Connexus/Core/HttpRequest.mqh. Initially, the file will look like this:

```
//+------------------------------------------------------------------+
//|                                                  HttpRequest.mqh |
//|                                  Copyright 2024, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
//+------------------------------------------------------------------+
//| class : CHttpRequest                                             |
//|                                                                  |
//| [PROPERTY]                                                       |
//| Name        : CHttpRequest                                       |
//| Heritage    : No heritage                                        |
//| Description : Gathers elements of an http request such as url,   |
//|               body, header, method and timeout                   |
//|                                                                  |
//+------------------------------------------------------------------+
class CHttpRequest
  {
public:
                     CHttpRequest(void);
                    ~CHttpRequest(void);
  };
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CHttpRequest::CHttpRequest(void)
  {
  }
//+------------------------------------------------------------------+
//| Destructor                                                       |
//+------------------------------------------------------------------+
CHttpRequest::~CHttpRequest(void)
  {
  }
//+------------------------------------------------------------------+
```

Let's import the classes created in the last files: CURL, CHttpBody, CHttpHeader, and CHttpMethod. We'll create an instance of each, adding the "\*" to inform the compiler that we will be managing the pointer manually. Additionally, we'll add a variable named m\_timeout of type int, which will store the timeout value for the request.

```
//+------------------------------------------------------------------+
//|                                                  HttpRequest.mqh |
//|                                  Copyright 2024, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include "../URL/URL.mqh"
#include "../Header/HttpBody.mqh"
#include "../Header/HttpHeader.mqh"
#include "../Constants/HttpMethod.mqh"
//+------------------------------------------------------------------+
//| class : CHttpRequest                                             |
//|                                                                  |
//| [PROPERTY]                                                       |
//| Name        : CHttpRequest                                       |
//| Heritage    : No heritage                                        |
//| Description : Gathers elements of an http request such as url,   |
//|               body, header, method and timeout                   |
//|                                                                  |
//+------------------------------------------------------------------+
class CHttpRequest
  {
private:

   CURL              *m_url;
   CHttpBody         *m_body;
   CHttpHeader       *m_headers;
   CHttpMethod       *m_method;
   int               m_timeout;

public:
                     CHttpRequest(void);
                    ~CHttpRequest(void);
  };
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CHttpRequest::CHttpRequest(void)
  {
   m_url = new CURL();
   m_body = new CHttpBody();
   m_headers = new CHttpHeader();
   m_method = new CHttpMethod();
   m_timeout = 5000; // Default timeout (5 seconds)
  }
//+------------------------------------------------------------------+
//| Destructor                                                       |
//+------------------------------------------------------------------+
CHttpRequest::~CHttpRequest(void)
  {
   delete m_url;
   delete m_body;
   delete m_headers;
   delete m_method;
  }
//+------------------------------------------------------------------+
```

Now, let's add some methods to access the pointer for each instance, as well as methods to set and get the timeout value:

- CURL \*Url(void) : Returns the pointer to the URL
- CHttpBody \*Body(void) : Returns the pointer to the body
- CHttpHeader \*Header(void) : Returns the pointer to the header
- CHttpMethod \*Method(void) : Returns the pointer to the method
- CHttpRequest \*Timeout(int timeout) : Sets the timeout
- int Timeout(void) : Gets the timeout

In addition to these methods, we'll add some auxiliary methods:

- void Clear(void) : Removes all data from the instances

- string FormatString(void) : Generates a string containing all the request data

Example of a formatted request:


```
HTTP Request:
  ---------------
Method: GET
URL: https://api.example.com/resource?id=123&filter=active

Headers:
  ---------------
Authorization: Bearer token
Content-Type: application/json
User-Agent: MyHttpClient/1.0

Body:
  ---------------
{
      "key": "value",
      "data": [1, 2, 3]
}
  ---------------
```


Here is the implementation of the methods.

```
//+------------------------------------------------------------------+
//|                                                  HttpRequest.mqh |
//|                                  Copyright 2024, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
//+------------------------------------------------------------------+
//| class : CHttpRequest                                             |
//|                                                                  |
//| [PROPERTY]                                                       |
//| Name        : CHttpRequest                                       |
//| Heritage    : No heritage                                        |
//| Description : Gathers elements of an http request such as url,   |
//|               body, header, method and timeout                   |
//|                                                                  |
//+------------------------------------------------------------------+
class CHttpRequest
  {
public:
   //--- HTTP
   CURL              *Url(void);                         // Get url object
   CHttpBody         *Body(void);                        // Get body object
   CHttpHeader       *Header(void);                      // Get header object
   CHttpMethod       *Method(void);                      // Get method object

   //--- Timeout
   CHttpRequest      *Timeout(int timeout);              // Set timeout
   int               Timeout(void);                      // Get timeout

   //--- Auxiliary methods
   void              Clear(void);                        // Reset data
   string            FormatString(void);                 // Format data
  };
//+------------------------------------------------------------------+
//| Get url object                                                   |
//+------------------------------------------------------------------+
CURL *CHttpRequest::Url(void)
  {
   return(GetPointer(m_url));
  }
//+------------------------------------------------------------------+
//| Get body object                                                  |
//+------------------------------------------------------------------+
CHttpBody *CHttpRequest::Body(void)
  {
   return(GetPointer(m_body));
  }
//+------------------------------------------------------------------+
//| Get header object                                                |
//+------------------------------------------------------------------+
CHttpHeader *CHttpRequest::Header(void)
  {
   return(GetPointer(m_headers));
  }
//+------------------------------------------------------------------+
//| Get method object                                                |
//+------------------------------------------------------------------+
CHttpMethod *CHttpRequest::Method(void)
  {
   return(GetPointer(m_method));
  }
//+------------------------------------------------------------------+
//| Set timeout                                                      |
//+------------------------------------------------------------------+
CHttpRequest *CHttpRequest::Timeout(int timeout)
  {
   m_timeout = timeout;
   return(GetPointer(this));
  }
//+------------------------------------------------------------------+
//| Get timeout                                                      |
//+------------------------------------------------------------------+
int CHttpRequest::Timeout(void)
  {
   return(m_timeout);
  }
//+------------------------------------------------------------------+
//| Reset data                                                       |
//+------------------------------------------------------------------+
void CHttpRequest::Clear(void)
  {
   m_url.Clear();
   m_body.Clear();
   m_headers.Clear();
   m_timeout = 5000;
  }
//+------------------------------------------------------------------+
//| Format data                                                      |
//+------------------------------------------------------------------+
string CHttpRequest::FormatString(void)
  {
   return(
      "HTTP Request:"+
    "\n---------------"+
    "\nMethod: "+m_method.GetMethodDescription()+
    "\nURL: "+m_url.FullUrl()+
    "\n"+
    "\n---------------"+
    "\nHeaders:"+
    "\n"+m_headers.Serialize()+
    "\n"+
    "\n---------------"+
    "\nBody:"+
    "\n"+m_body.GetAsString()+
    "\n---------------"
   );
  }
//+------------------------------------------------------------------+
```

Thus, we have completed this class, which serves as a facade to access the objects that form an HTTP request. It is important to note that I am only adding the parts that have been modified in the code. The full file can be found at the end of the attached article.

With this new CHttpRequest class, the updated library diagram looks like this:

![diagram 2](https://c.mql5.com/2/99/diagram2.png)

In summary, CHttpRequest acts as a facade, simplifying the process of configuring and sending an HTTP request. Internally, the CHttpHeaders class handles the header logic, while CHttpBody takes care of constructing the request body. The developer using this library doesn't need to worry about the details of how the headers or body are handled – they simply set the values using the methods of CHttpRequest, and the facade class takes care of the rest.

### Creating the CHttpResponse Class

Following the same idea, let's create another class that will be used to represent the server's response data. It will follow a structure similar to CHttpRequest . These are the elements that form a response:

- **Header**: Just like the request, the response also includes a header, which informs the client with metadata about the request.
- **Body**: This will contain the body of the server's response. This is where the server sends the data that we want to obtain, and this is the core of the message.
- **Status Code**: Contains the status code, which we discussed in more detail in Part 5 of the series. This code is a 3-digit number that informs whether the request was successfully processed or if it encountered an error from the client or server.
- **Duration**: Stores the time the request took to complete, measured in milliseconds.

We will create a new file inside the _core_ folder called HttpResponse , and the final path will be Include/Connexus/Core/HttpResponse.mqh . The created class should look like this:

```
//+------------------------------------------------------------------+
//|                                                 HttpResponse.mqh |
//|                                  Copyright 2024, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
//+------------------------------------------------------------------+
//| class : CHttpResponse                                            |
//|                                                                  |
//| [PROPERTY]                                                       |
//| Name        : CHttpResponse                                      |
//| Heritage    : No heritage                                        |
//| Description : gathers elements of an http response such as body, |
//|               header, status code and duration                   |
//|                                                                  |
//+------------------------------------------------------------------+
class CHttpResponse
  {
public:
                     CHttpResponse(void);
                    ~CHttpResponse(void);
  };
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CHttpResponse::CHttpResponse(void)
  {
  }
//+------------------------------------------------------------------+
//| Destructor                                                       |
//+------------------------------------------------------------------+
CHttpResponse::~CHttpResponse(void)
  {
  }
//+------------------------------------------------------------------+
```

We will import the elements of a response, which are: CHttpHeader, CHttpBody, and CHttpStatusCode. After that, we will create an instance of each of these classes and also a private variable of type ulong, which will store the duration of the request. The final code will look like this:

```
//+------------------------------------------------------------------+
//|                                                 HttpResponse.mqh |
//|                                  Copyright 2024, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include "../Constants/HttpStatusCode.mqh"
#include "../Header/HttpBody.mqh"
#include "../Header/HttpHeader.mqh"
//+------------------------------------------------------------------+
//| class : CHttpResponse                                            |
//|                                                                  |
//| [PROPERTY]                                                       |
//| Name        : CHttpResponse                                      |
//| Heritage    : No heritage                                        |
//| Description : gathers elements of an http response such as body, |
//|               header, status code and duration                   |
//|                                                                  |
//+------------------------------------------------------------------+
class CHttpResponse
  {
private:

   CHttpHeader       *m_header;
   CHttpBody         *m_body;
   CHttpStatusCodes  *m_status_code;
   ulong              m_duration;

public:
                     CHttpResponse(void);
                    ~CHttpResponse(void);
  };
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CHttpResponse::CHttpResponse(void)
  {
   m_header = new CHttpHeader();
   m_body = new CHttpBody();
   m_status_code = new CHttpStatusCodes();
   m_duration = 0; // Reset duration
  }
//+------------------------------------------------------------------+
//| Destructor                                                       |
//+------------------------------------------------------------------+
CHttpResponse::~CHttpResponse(void)
  {
   delete m_header;
   delete m_body;
   delete m_status_code;
  }
//+------------------------------------------------------------------+
```

Now let's move on to the methods that will be added to this class. We'll start with the simplest ones, which simply return the pointers of each of the instances of the classes, and methods to set and get the duration:

- CHttpHeader \*Header(void) : Returns the pointer to the header.
- CHttpBody \*Body(void) : Returns the pointer to the body.
- CHttpStatusCodes \*StatusCode(void) : Returns the pointer to the status code.
- void Duration(ulong duration) : Sets the duration.
- ulong Duration(void) : Gets the duration.

We'll create the same helper methods for CHttpRequest for this class:

- void Clear(void) : Removes all the data from the instances.

- string FormatString(void) : Generates a string containing all the response data.

Example of a formatted response


```
HTTP Response:
  ---------------
Status Code: 200 OK
Duration: 120ms

Headers:
  ---------------
Content-Type: application/json
Content-Length: 256
Server: Apache/2.4.41 (Ubuntu)
Date: Wed, 02 Oct 2024 12:34:56 GMT

Body:
  ---------------
{
      "message": "Success",
      "data": {
          "id": 123,
          "name": "Sample Item",
          "status": "active"
      }
}
  ---------------
```


Here is the code with the implementation of these functions:

```
//+------------------------------------------------------------------+
//|                                                 HttpResponse.mqh |
//|                                  Copyright 2024, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
//+------------------------------------------------------------------+
//| class : CHttpResponse                                            |
//|                                                                  |
//| [PROPERTY]                                                       |
//| Name        : CHttpResponse                                      |
//| Heritage    : No heritage                                        |
//| Description : gathers elements of an http response such as body, |
//|               header, status code and duration                   |
//|                                                                  |
//+------------------------------------------------------------------+
class CHttpResponse
  {
public:
   //--- HTTP
   CHttpHeader       *Header(void);                      // Get header object
   CHttpBody         *Body(void);                        // Get body object
   CHttpStatusCodes  *StatusCode(void);                  // Get status code object

   //--- Duration
   void              Duration(ulong duration);           // Set duration
   ulong             Duration(void);                     // Get duration

   //--- Auxiliary methods
   void              Clear(void);                        // Reset data
   string            FormatString(void);                 // Format data
  };
//+------------------------------------------------------------------+
//| Get header object                                                |
//+------------------------------------------------------------------+
CHttpHeader *CHttpResponse::Header(void)
  {
   return(GetPointer(m_header));
  }
//+------------------------------------------------------------------+
//| Get body object                                                  |
//+------------------------------------------------------------------+
CHttpBody *CHttpResponse::Body(void)
  {
   return(GetPointer(m_body));
  };
//+------------------------------------------------------------------+
//| Get status code object                                           |
//+------------------------------------------------------------------+
CHttpStatusCodes *CHttpResponse::StatusCode(void)
  {
   return(GetPointer(m_status_code));
  };
//+------------------------------------------------------------------+
//| Set duration                                                     |
//+------------------------------------------------------------------+
void CHttpResponse::Duration(ulong duration)
  {
   m_duration = duration;
  }
//+------------------------------------------------------------------+
//| Get duration                                                     |
//+------------------------------------------------------------------+
ulong CHttpResponse::Duration(void)
  {
   return(m_duration);
  }
//+------------------------------------------------------------------+
//| Reset data                                                       |
//+------------------------------------------------------------------+
void CHttpResponse::Clear(void)
  {
   m_header.Clear();
   m_body.Clear();
   m_status_code.SetStatusCode(HTTP_STATUS_URL_NOT_ALLOWED);
  }
//+------------------------------------------------------------------+
//| Format data                                                      |
//+------------------------------------------------------------------+
string CHttpResponse::FormatString(void)
  {
   return(
      "HTTP Response:"+
    "\n---------------"+
    "\nStatus Code: "+m_status_code.GetStatusCodeFormat()+
    "\nDuration: "+IntegerToString(m_duration)+" ms"+
    "\n"+
    "\n---------------"+
    "\nHeaders:"+
    "\n"+m_header.Serialize()+
    "\n---------------"+
    "\nBody:"+
    "\n"+m_body.GetAsString()+
    "\n---------------"
   );
  }
//+------------------------------------------------------------------+
```

Just to remind you, here I'm only including the changes to the class so that the article doesn't become too lengthy. All the code used is attached at the end of the article.

This concludes the response class. These are relatively simple classes, and the goal is one thing: to group the elements of a request or response. This way, we can work with a request as a single object, significantly simplifying the use of HTTP requests and responses. With this new CHttpResponse class, the updated diagram looks like this:

![diagram 3](https://c.mql5.com/2/99/diagram3.png)

Usage examples

Briefly, I will give some usage examples of these classes. Starting with CHttpRequest, I will construct an HTTP request using this class.

```
//--- Example 1 - GET
CHttpRequest request;
request.Method() = HTTP_METHOD_GET;
request.Url().Parse("http://example.com/api/symbols");

//--- Example 2 - POST
CHttpRequest request;
request.Method() = HTTP_METHOD_POST;
request.Url().Parse("http://example.com/api/symbols");
request.Body().AddToString("{\"EURUSD\":1.08312}");
request.Header().Add("content-type","application/json");

//--- Example 3 - DELETE
CHttpRequest request;
request.Method() = HTTP_METHOD_DELETE;
request.Url().Parse("http://example.com/api/symbols?symbol=EURUSD");
```

The big advantage here is the simplicity of creating a request, adding a header and body, or changing the HTTP method.

For responses, what we need is the opposite of this, that is, instead of creating a request object easily, we want to read a response object easily, and both work in a similar way, which allows for simple creation and reading. I will include an image showing what the developer screen looks like using Connexus:

![example 1](https://c.mql5.com/2/99/example1.png)

This is what will appear to you, the developer. I marked each of the response data that can be accessed with different colors. Let's access the response body:

![example 2](https://c.mql5.com/2/99/example2.png)

Note that we can access the response body in different formats such as string, json or binary. Here we directly access the CHttpBody class pointer, then we access all the methods it has. When we created a class for each of these elements of a response, I was thinking of getting to this part of the library, where we use each of the classes.

### Conclusion

In this article, we explored the importance of having a CHttpRequest object that groups all the components of an HTTP request, ensuring clarity, reusability, and easy maintenance of the code. We also saw how the _Facade_ design pattern can be applied to simplify the interaction with complex components, hiding the internal complexity and offering a clean and efficient interface. In the context of Connexus, applying the _Facade_ pattern to the ChttpRequest class makes it easier to create complete HTTP requests, simplifying the process for the developer, and also creating the CHttpResponse class that follows the same format, but should facilitate access to the data of an HTTP response.

In the next article, we will delve deeper into the transport layer, which must receive a CHttpRequest object and convert the data to the WebRequest function, which is the final layer of the library, must be able to process the response data and return an object of type CHttpResponse for the developer to then use the response data. See you there!

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/16182.zip "Download all attachments in the single ZIP archive")

[Requesting\_in\_Connexus\_ePart\_6r\_Creating\_an\_HTTP\_Request\_and\_Response.zip](https://www.mql5.com/en/articles/download/16182/requesting_in_connexus_epart_6r_creating_an_http_request_and_response.zip "Download Requesting_in_Connexus_ePart_6r_Creating_an_HTTP_Request_and_Response.zip")(29.13 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/475956)**
(1)


![Jedidiah](https://c.mql5.com/avatar/avatar_na2.png)

**[Jedidiah](https://www.mql5.com/en/users/seleucus)**
\|
6 Nov 2024 at 10:39

You have made a tremendous contribution to mql5 and all of us. Making it easier than ever to work with [web requests](https://www.mql5.com/en/docs/network/webrequest "MQL5 documentation: WebRequest function") and http in general. This is excellent stuff. Thanks man. Can’t wait for the final article to seal the deal with the connexus library.


![How to view deals directly on the chart without weltering in trading history](https://c.mql5.com/2/80/How_to_avoid_drowning_in_trading_history_and_easily_glide_right_along_the_chart____LOGO.png)[How to view deals directly on the chart without weltering in trading history](https://www.mql5.com/en/articles/15026)

In this article, we will create a simple tool for convenient viewing of positions and deals directly on the chart with key navigation. This will allow traders to visually examine individual deals and receive all the information about trading results right on the spot.

![Developing a Replay System (Part 51): Things Get Complicated (III)](https://c.mql5.com/2/79/Desenvolvendo_um_sistema_de_Replay_Parte_51____LOGO.png)[Developing a Replay System (Part 51): Things Get Complicated (III)](https://www.mql5.com/en/articles/11877)

In this article, we will look into one of the most difficult issues in the field of MQL5 programming: how to correctly obtain a chart ID, and why objects are sometimes not plotted on the chart. The materials presented here are for didactic purposes only. Under no circumstances should the application be viewed for any purpose other than to learn and master the concepts presented.

![News Trading Made Easy (Part 5): Performing Trades (II)](https://c.mql5.com/2/99/news-trading-made-easy-4__LOGO.png)[News Trading Made Easy (Part 5): Performing Trades (II)](https://www.mql5.com/en/articles/16169)

This article will expand on the trade management class to include buy-stop and sell-stop orders to trade news events and implement an expiration constraint on these orders to prevent any overnight trading. A slippage function will be embedded into the expert to try and prevent or minimize possible slippage that may occur when using stop orders in trading, especially during news events.

![Self Optimizing Expert Advisor With MQL5 And Python (Part VI): Taking Advantage of Deep Double Descent](https://c.mql5.com/2/100/Self_Optimizing_Expert_Advisor_With_MQL5_And_Python_Part_VI__LOGO.png)[Self Optimizing Expert Advisor With MQL5 And Python (Part VI): Taking Advantage of Deep Double Descent](https://www.mql5.com/en/articles/15971)

Traditional machine learning teaches practitioners to be vigilant not to overfit their models. However, this ideology is being challenged by new insights published by diligent researches from Harvard, who have discovered that what appears to be overfitting may in some circumstances be the results of terminating your training procedures prematurely. We will demonstrate how we can use the ideas published in the research paper, to improve our use of AI in forecasting market returns.

[![](https://www.mql5.com/ff/sh/5z040u47jcv59943z2/6c76c03a8b37e08b8655a1a085770b7a.jpg)\\
MetaTrader 5 for iOS and Android\\
\\
Fully featured platform for any devices and web browsers\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=ddonqpipxfqlnsvzlwuowsuwlejpyjxk&s=9daba65b69f40afc3c35f95b1f84ef5824d68c47f29ce96a6dc5b164a2727baa&uid=&ref=https://www.mql5.com/en/articles/16182&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5071607493542488868)

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