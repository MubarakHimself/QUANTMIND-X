---
title: Introduction to Connexus (Part 1): How to Use the WebRequest Function?
url: https://www.mql5.com/en/articles/15795
categories: Integration, Expert Advisors
relevance_score: 0
scraped_at: 2026-01-24T14:03:31.405392
---

[![](https://www.mql5.com/ff/sh/0hvxp984jjj79943z2/6373d9e5710a718ffa6a7d50a5db9dd1.jpg)\\
Web terminal on your iPhone or Android\\
\\
Full-featured MetaTrader 5 platform for any devices and web browsers\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=uyigsjnbfcdvysiynusmriwvhincciwd&s=c95531ae2fd8a81b0fac3def2e4cf820a67584bbf4b02f76ec75f808942dbbd2&uid=&ref=https://www.mql5.com/en/articles/15795&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5083307826270443777)

MetaTrader 5 / Examples


### Introduction

In the world of financial programming, especially in the context of MetaTrader 5, the ability to interact with remote servers via HTTP is vital. Whether it’s to obtain real-time market data, send trading orders to an API, or even query third-party services, HTTP requests play a crucial role. In MQL5, the [WebRequest](https://www.mql5.com/en/docs/network/webrequest) function is the native tool provided for such communications, but its limitations make it impractical in many scenarios.

HTTP (Hypertext Transfer Protocol) is the basis of web communication, and mastering its use allows developers to create rich and dynamic integrations between MetaTrader 5 and other services. For example, an Expert Advisor (EA) may need to access a news API to adjust its trading strategies based on global events. Another common application is querying cryptocurrency prices on exchanges that offer HTTP APIs, allowing the EA to trade in sync with these markets.

Despite the importance of HTTP requests, the implementation of the WebRequest function in MQL5 is not exactly what one would expect from a modern and flexible tool. This puts developers in a challenging position: either adapt to the limitations or look for workarounds. The series of articles we are starting here aims to do just that—explore the weaknesses of the WebRequest function and build a library, Connexus, that overcomes these limitations and makes the work of MQL5 developers easier.

The WebRequest function opens up a wide range of possibilities for integration with external services. From collecting financial data, which can be essential for automated trading decisions, to fully automating processes, this function allows EAs to interact directly with the web. This allows, for example, your trading robot to obtain real-time information from external sources, such as economic news or market data from other platforms. This data can be processed and used to automatically adjust your trading strategies, increasing both the accuracy and efficiency of your operations.

However, as will be demonstrated in the examples presented, using the WebRequest function may not be trivial. Sending an HTTP request might seem simple, but you will quickly encounter technical challenges, such as sending the correct headers, formatting JSON data, properly handling server responses, and even dealing with errors and exceptions that may occur during communication. These challenges will illustrate that, although powerful, the function requires a solid understanding of protocols and communication between systems, which can present a significant barrier for developers who are just starting to explore this area.

This is exactly where the need for a more accessible and efficient tool will come in. The Connexus  library, which will be developed and improved in the upcoming articles in this series, aims to overcome these limitations and make the integration process via WebRequest more user-friendly and intuitive. With Connexus  , the idea will be that developers can focus on what truly matters: the logic of their applications and EAs, without having to deal directly with the technical details of the lower layers of network programming. Instead of wasting time debugging formatting or header errors, you will be able to focus on integrating your systems efficiently, with a clear and functional interface.

This series of articles will be dedicated to thoroughly exploring the weaknesses of the WebRequest function, its limitations, and how we will work around them when developing a robust solution. In addition to continuing the discussion of the HTTP protocol, we will cover aspects such as authentication in APIs, handling large volumes of data, and implementing advanced features like response time control and handling multiple simultaneous requests.

So, if you are interested in improving your MQL5 development skills, learning more about system integration, and optimizing HTTP communication processes, stay tuned for the upcoming publications. We will continue to expand the scope of this project, guiding you in developing the **Connexus** library so that it becomes an indispensable tool in your development arsenal. The knowledge gained here will be useful not only for those who work with MetaTrader, but also for any developer who needs to integrate APIs and web services into their applications.

### Getting to Know WebRequest

The WebRequest function is the main tool provided by MQL5 for making HTTP requests. In simple terms, it allows an MQL5 program to send a request to a server and receive a response. Although it may seem simple, practical use of WebRequest reveals a number of pitfalls and complexities that can complicate development.

The basic syntax of the function has two versions:

```
int  WebRequest(
   const string      method,           // HTTP method
   const string      url,              // URL
   const string      cookie,           // cookie
   const string      referer,          // referer
   int               timeout,          // timeout
   const char        &data[],          // the array of the HTTP message body
   int               data_size,        // data[] array size in bytes
   char              &result[],        // an array containing server response data
   string            &result_headers   // headers of server response
   );
```

```
int  WebRequest(
   const string      method,           // HTTP method
   const string      url,              // URL
   const string      headers,          // headers
   int               timeout,          // timeout
   const char        &data[],          // the array of the HTTP message body
   char              &result[],        // an array containing server response data
   string            &result_headers   // headers of server response
   );
```

The parameters of the WebRequest function are vital to its proper functioning, and a detailed understanding of them is essential for any developer who wants to use it effectively. Let's explore each of these parameters:

| Parameter Name | Type | Input/Output | Description |
| --- | --- | --- | --- |
| method | string | in | This parameter defines the type of HTTP request you want to make. The most common types are GET and POST. The GET type is used to request data from a server, while the POST type is used to send data to the server. |
| url | string | in | This is the address of the server to which the request will be sent. The URL must be complete, including the protocol (http or https). |
| headers | string | in | Additional HTTP headers can be passed as an array of strings. Each string must be in the format "Key: Value". These headers are used to pass additional information, such as authentication tokens or content type, separated by a newline "\\r\\n". |
| cookie | string | in | Cookie value. |
| referer | string | in | Value of the Referer header of the HTTP request. |
| timeout | int | in | Sets the maximum time to wait for a response from the server, in milliseconds. A proper timeout is crucial to ensure that the EA does not get stuck waiting for a response that may never arrive. |
| data | char\[\] | in | For POST requests, this parameter is used to send the data to the server. This data needs to be in the form of a byte array, which can be challenging for developers who are not familiar with handling binary data. |
| data\_size | int | in | This is the size of the data to be sent. It must match the size of the data array for the request to work correctly. |
| result | char\[\] | out | This parameter receives the response from the server, also as an array of bytes. After the function is called, the array needs to be decoded to extract the useful information. |
| result\_headers | string | out | This array of strings receives the response headers from the server, which may contain important information such as content type and authentication status. |

Each of these settings must be configured carefully to ensure that the request is made correctly. An error in any of these parameters can result in a malformed request or a complete failure to communicate with the server.

The function returns an HTTP status code that indicates the success or failure of the operation. While WebRequest covers the basic concepts, its implementation leaves much to be desired. It requires the developer to manually manage the creation of headers, the handling of different data types, and the error checking, making the process tedious and error-prone. One good thing about WebRequest is that it supports both GET and POST requests, which allows it to interact with a wide range of APIs.

### Practical Example of the WebRequest Function

To illustrate the use of the WebRequest function, let's build a simple example. For this we will use [httpbin.org](https://www.mql5.com/go?link=https://httpbin.org/ "https://httpbin.org/"), which is a free online service that allows you to make and test HTTP requests. It was created by [kennethreitz](https://www.mql5.com/go?link=https://kennethreitz.org/ "https://kennethreitz.org/"), it is an OpenSource project ( [link](https://www.mql5.com/go?link=https://github.com/postmanlabs/httpbin "https://github.com/postmanlabs/httpbin")). This service works in a very simple and uncomplicated way. It is basically a "mirror". You stand in front of the mirror and strike a pose or ask a question. This is like sending a request to HTTP Bin. The mirror reflects exactly what you are doing. This is like HTTP Bin receiving and reflecting what you sent. It is a useful tool for developers who want to check exactly what is being sent in their HTTP requests or who need to simulate different types of requests and responses. Some common features of httpbin include:

1. **Sending Requests:** You can send HTTP requests of different types, such as GET, POST, PUT, DELETE, etc., to see how the server responds.
2. **HTTP Header Testing:** You can send custom headers and view the server’s response, which is useful for debugging header-related issues.
3. **Sending Data in the Request Body:** It’s possible to test sending data in the request body and see how the server handles it.

4. **HTTP Status Simulation:** You can request the server to return specific status codes, such as 200, 404, 500, etc., to test how your application handles different status responses.

5. **Simulation of Delays and Redirects:** httpbin.org allows simulating response delays or redirects, helping to test system behavior in more complex scenarios.

6. **Cookie Testing:** You can manipulate cookies, seeing how they are set and returned by the server.


It is a practical tool for integrating systems that use the HTTP protocol. Let’s make the simplest possible GET request using WebRequest.

**Step 1: Add URL in the Terminal**

According to the documentation: To use the WebRequest() function, add the required server addresses to the list of allowed URLs in the "Tools" tab of the "Options" window. The server’s port is automatically selected based on the specified protocol - 80 for "http://" and 443 for "https://".

![](https://c.mql5.com/2/94/125611016856.png)

![](https://c.mql5.com/2/94/4158435562154.png)

![](https://c.mql5.com/2/94/5673630630579.png)

**Step 2: Hands on Code**

In the directory <data\_folder>/MQL5/Experts , create a folder called Connexus. We will place our test files in this folder. To find the data folder, in the MetaTrader or MetaEditor main menu, select File > Open Data Folder. Inside that folder, create a file named “WebRequest.mq5” and you’ll have something like this:

```
//+------------------------------------------------------------------+
//|                                                   WebRequest.mq5 |
//|                                  Copyright 2023, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2023, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---

//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---

  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//---

  }
//+------------------------------------------------------------------+
```

We will only use the OnInit() event for testing for now. Let's define some variables to perform the request and pass them to the WebRequest function.

```
int OnInit()
  {
//--- Defining variables
   string method = "GET";                    // HTTP verb in string (GET, POST, etc...)
   string url = "https://httpbin.org/get";   // Destination URL
   string headers = "";                      // Request header
   int timeout = 5000;                       // Maximum waiting time 5 seconds
   char data[];                              // Data we will send (body) array of type char
   char result[];                            // Data received as an array of type char
   string result_headers;                    // String with response headers

//--- Calling the function and getting the status_code
   int status_code = WebRequest(method,url,headers,timeout,data,result,result_headers);

//--- Print the data
   Print("Status code: ",status_code);
   Print("Response: ",CharArrayToString(result)); // We use CharArrayToString to display the response in string form.

//---
   return(INIT_SUCCEEDED);
  }
```

When we insert the robot into the chart, the terminal will print the following response:

```
WebRequest (WINV24,M1)  Status code: 200
WebRequest (WINV24,M1)  Resposta: {
WebRequest (WINV24,M1)    "args": {},
WebRequest (WINV24,M1)    "headers": {
WebRequest (WINV24,M1)      "Accept": "*/*",
WebRequest (WINV24,M1)      "Accept-Encoding": "gzip, deflate",
WebRequest (WINV24,M1)      "Accept-Language": "pt,en;q=0.5",
WebRequest (WINV24,M1)      "Host": "httpbin.org",
WebRequest (WINV24,M1)      "User-Agent": "MetaTrader 5 Terminal/5.4476 (Windows NT 11.0.22631; x64)",
WebRequest (WINV24,M1)      "X-Amzn-Trace-Id": "Root=1-66d8cd53-0d6cd16368aa22e455db461c"
WebRequest (WINV24,M1)    },
WebRequest (WINV24,M1)    "origin": "XXX.XXX.XX.XXX",
WebRequest (WINV24,M1)    "url": "https://httpbin.org/get"
WebRequest (WINV24,M1)  }
WebRequest (WINV24,M1)
```

Note that the status code received was 200, meaning the request was successful, and in the response we received a JSON with some data. In the next article we will explore in more detail how communication works via the HTTP protocol. Now we will change the HTTP verb to POST and send data in the body of the request.

```
int OnInit()
  {
//--- Defining variables
   string method = "POST";                    // HTTP verb in string (GET, POST, etc...)
   string url = "https://httpbin.org/post";   // Destination URL
   string headers = "";                       // Request header
   int timeout = 5000;                        // Maximum waiting time 5 seconds
   char data[];                               // Data we will send (body) array of type char
   char result[];                             // Data received as an array of type char
   string result_headers;                     // String with response headers

//--- Treating body
   string body = "{\"key1\":\"value1\",\"key2\":\"value2\"}";
   StringToCharArray(body,data,0,WHOLE_ARRAY,CP_UTF8); // Converts a string to a byte array

//--- Calling the function and getting the status_code
   int status_code = WebRequest(method,url,headers,timeout,data,result,result_headers);

//--- Print the data
   Print("Status code: ",status_code);
   Print("Response: ",CharArrayToString(result)); // We use CharArrayToString to display the response in string form.

//---
   return(INIT_SUCCEEDED);
  }
```

Okay, now we have everything working, considering that httpbin will return everything we send, it should return the body we sent, right? Executing the POST code will return:

```
WebRequest (WINV24,M1)  Status code: 200
WebRequest (WINV24,M1)  Resposta: {
WebRequest (WINV24,M1)    "args": {},
WebRequest (WINV24,M1)    "data": "",
WebRequest (WINV24,M1)    "files": {},
WebRequest (WINV24,M1)    "form": {
WebRequest (WINV24,M1)      "{\"key1\":\"value1\",\"key2\":\"value2\"}\u0000": ""
WebRequest (WINV24,M1)    },
WebRequest (WINV24,M1)    "headers": {
WebRequest (WINV24,M1)      "Accept": "*/*",
WebRequest (WINV24,M1)      "Accept-Encoding": "gzip, deflate",
WebRequest (WINV24,M1)      "Accept-Language": "pt,en;q=0.5",
WebRequest (WINV24,M1)      "Content-Length": "34",
WebRequest (WINV24,M1)      "Content-Type": "application/x-www-form-urlencoded",
WebRequest (WINV24,M1)      "Host": "httpbin.org",
WebRequest (WINV24,M1)      "User-Agent": "MetaTrader 5 Terminal/5.4476 (Windows NT 11.0.22631; x64)",
WebRequest (WINV24,M1)      "X-Amzn-Trace-Id": "Root=1-66d9bc77-314c004a607c383b3197c15a"
WebRequest (WINV24,M1)    },
WebRequest (WINV24,M1)    "json": null,
WebRequest (WINV24,M1)    "origin": "200.103.20.126",
WebRequest (WINV24,M1)    "url": "https://httpbin.org/post"
WebRequest (WINV24,M1)  }
WebRequest (WINV24,M1)
```

Note that we have some additional fields, such as “json” and “data”. Let’s understand each of them. The “data” field should show the body that we sent in string format, while the “json” field should show the body that we sent in json format. But why are both empty if we sent the body in the request? Because we have to inform the server that the content type will be json. To do this, we configure the request header, like this:

```
int OnInit()
  {
//--- Defining variables
   string method = "POST";                    // HTTP verb in string (GET, POST, etc...)
   string url = "https://httpbin.org/post";   // Destination URL
   string headers = "Content-Type: application/json";// Request header
   int timeout = 5000;                        // Maximum waiting time 5 seconds
   char data[];                               // Data we will send (body) array of type char
   char result[];                             // Data received as an array of type char
   string result_headers;                     // String with response headers

//--- Treating body
   string body = "{\"key1\":\"value1\",\"key2\":\"value2\"}";
   StringToCharArray(body,data,0,WHOLE_ARRAY,CP_UTF8);

//--- Calling the function and getting the status_code
   int status_code = WebRequest(method,url,headers,timeout,data,result,result_headers);

//--- Print the data
   Print("Status code: ",status_code);
   Print("Response: ",CharArrayToString(result)); // We use CharArrayToString to display the response in string form.

//---
   return(INIT_SUCCEEDED);
  }
```

```
WebRequest (WINV24,M1)  Status code: 200
WebRequest (WINV24,M1)  Resposta: {
WebRequest (WINV24,M1)    "args": {},
WebRequest (WINV24,M1)    "data": "{\"key1\":\"value1\",\"key2\":\"value2\"}\u0000",
WebRequest (WINV24,M1)    "files": {},
WebRequest (WINV24,M1)    "form": {},
WebRequest (WINV24,M1)    "headers": {
WebRequest (WINV24,M1)      "Accept": "*/*",
WebRequest (WINV24,M1)      "Accept-Encoding": "gzip, deflate",
WebRequest (WINV24,M1)      "Accept-Language": "pt,en;q=0.5",
WebRequest (WINV24,M1)      "Content-Length": "34",
WebRequest (WINV24,M1)      "Content-Type": "application/json",
WebRequest (WINV24,M1)      "Host": "httpbin.org",
WebRequest (WINV24,M1)      "User-Agent": "MetaTrader 5 Terminal/5.4476 (Windows NT 11.0.22631; x64)",
WebRequest (WINV24,M1)      "X-Amzn-Trace-Id": "Root=1-66d9be03-59060f042f7090092787855e"
WebRequest (WINV24,M1)    },
WebRequest (WINV24,M1)    "json": null,
WebRequest (WINV24,M1)    "origin": "200.103.20.126",
WebRequest (WINV24,M1)    "url": "https://httpbin.org/post"
WebRequest (WINV24,M1)  }
WebRequest (WINV24,M1)
```

Notice that now the data we sent is in the “data” field, which indicates that we are on the right track, but notice that the \\u0000 character appears because the StringToCharArray method includes the null terminator when converting the string to a byte array. To avoid this, we can adjust the size of the array. Let’s print the body that we are sending to see where this “\\u0000” character is coming from.

```
int OnInit()
  {
//--- Defining variables
   string method = "POST";                    // HTTP verb in string (GET, POST, etc...)
   string url = "https://httpbin.org/post";   // Destination URL
   string headers = "Content-Type: application/json";// Request header
   int timeout = 5000;                        // Maximum waiting time 5 seconds
   char data[];                               // Data we will send (body) array of type char
   char result[];                             // Data received as an array of type char
   string result_headers;                     // String with response headers

//--- Tratando body
   string body = "{\"key1\":\"value1\",\"key2\":\"value2\"}";
   StringToCharArray(body,data,0,WHOLE_ARRAY,CP_UTF8);
   Print("Body: ",body);
   Print("Body Size: ",StringLen(body));
   ArrayPrint(data);
   Print("Array Size: ",ArraySize(data));

//--- Calling the function and getting the status_code
   int status_code = WebRequest(method,url,headers,timeout,data,result,result_headers);

//--- Print the data
   Print("Status code: ",status_code);
   Print("Response: ",CharArrayToString(result)); // We use CharArrayToString to display the response in string form.

//---
   return(INIT_SUCCEEDED);
  }
```

```
WebRequest (WINV24,M1)  Body: {"key1":"value1","key2":"value2"}
WebRequest (WINV24,M1)  Body Size: 33
WebRequest (WINV24,M1)  123  34 107 101 121  49  34  58  34 118  97 108 117 101  49  34  44  34 107 101 121  50  34  58  34 118  97 108 117 101  50  34 125   0
WebRequest (WINV24,M1)  Array Size: 34
WebRequest (WINV24,M1)  Status code: 200
WebRequest (WINV24,M1)  Resposta: {
WebRequest (WINV24,M1)    "args": {},
WebRequest (WINV24,M1)    "data": "{\"key1\":\"value1\",\"key2\":\"value2\"}\u0000",
WebRequest (WINV24,M1)    "files": {},
WebRequest (WINV24,M1)    "form": {},
WebRequest (WINV24,M1)    "headers": {
WebRequest (WINV24,M1)      "Accept": "*/*",
WebRequest (WINV24,M1)      "Accept-Encoding": "gzip, deflate",
WebRequest (WINV24,M1)      "Accept-Language": "pt,en;q=0.5",
WebRequest (WINV24,M1)      "Content-Length": "34",
WebRequest (WINV24,M1)      "Content-Type": "application/json",
WebRequest (WINV24,M1)      "Host": "httpbin.org",
WebRequest (WINV24,M1)      "User-Agent": "MetaTrader 5 Terminal/5.4476 (Windows NT 11.0.22631; x64)",
WebRequest (WINV24,M1)      "X-Amzn-Trace-Id": "Root=1-66d9bed3-2ebfcda024f637f436fc1d82"
WebRequest (WINV24,M1)    },
WebRequest (WINV24,M1)    "json": null,
WebRequest (WINV24,M1)    "origin": "200.103.20.126",
WebRequest (WINV24,M1)    "url": "https://httpbin.org/post"
WebRequest (WINV24,M1)  }
WebRequest (WINV24,M1)
```

Note that the body string is valid JSON, that is, it opens and closes square brackets, values ​​are separated by commas, and respects the key:value rule. Let's see the byte array that is generated by the StringToCharArray function. Note that it prints the size of the string and the array, but they are different. The byte array is one position larger than the string. Also note that in the list of values, the last value is "0" where it should be 125, which is the character "}". So to solve this, we will remove the last position of the array using [ArrayRemove()](https://www.mql5.com/en/docs/array/arrayremove).

```
int OnInit()
  {
//--- Defining variables
   string method = "POST";                    // HTTP verb in string (GET, POST, etc...)
   string url = "https://httpbin.org/post";   // Destination URL
   string headers = "Content-Type: application/json";// Request header
   int timeout = 5000;                        // Maximum waiting time 5 seconds
   char data[];                               // Data we will send (body) array of type char
   char result[];                             // Data received as an array of type char
   string result_headers;                     // String with response headers

//--- Tratando body
   string body = "{\"key1\":\"value1\",\"key2\":\"value2\"}";
   StringToCharArray(body,data,0,WHOLE_ARRAY,CP_UTF8);
   ArrayRemove(data,ArraySize(data)-1);
   Print("Body: ",body);
   Print("Body Size: ",StringLen(body));
   ArrayPrint(data);
   Print("Array Size: ",ArraySize(data));

//--- Calling the function and getting the status_code
   int status_code = WebRequest(method,url,headers,timeout,data,result,result_headers);

//--- Print the data
   Print("Status code: ",status_code);
   Print("Response: ",CharArrayToString(result)); // We use CharArrayToString to display the response in string form.

//---
   return(INIT_SUCCEEDED);
  }
```

```
WebRequest (WINV24,M1)  Body: {"key1":"value1","key2":"value2"}
WebRequest (WINV24,M1)  Body Size: 33
WebRequest (WINV24,M1)  123  34 107 101 121  49  34  58  34 118  97 108 117 101  49  34  44  34 107 101 121  50  34  58  34 118  97 108 117 101  50  34 125
WebRequest (WINV24,M1)  Array Size: 33
WebRequest (WINV24,M1)  Status code: 200
WebRequest (WINV24,M1)  Resposta: {
WebRequest (WINV24,M1)    "args": {},
WebRequest (WINV24,M1)    "data": "{\"key1\":\"value1\",\"key2\":\"value2\"}",
WebRequest (WINV24,M1)    "files": {},
WebRequest (WINV24,M1)    "form": {},
WebRequest (WINV24,M1)    "headers": {
WebRequest (WINV24,M1)      "Accept": "*/*",
WebRequest (WINV24,M1)      "Accept-Encoding": "gzip, deflate",
WebRequest (WINV24,M1)      "Accept-Language": "pt,en;q=0.5",
WebRequest (WINV24,M1)      "Content-Length": "33",
WebRequest (WINV24,M1)      "Content-Type": "application/json",
WebRequest (WINV24,M1)      "Host": "httpbin.org",
WebRequest (WINV24,M1)      "User-Agent": "MetaTrader 5 Terminal/5.4476 (Windows NT 11.0.22631; x64)",
WebRequest (WINV24,M1)      "X-Amzn-Trace-Id": "Root=1-66d9c017-5985f48331dba63439d8192d"
WebRequest (WINV24,M1)    },
WebRequest (WINV24,M1)    "json": {
WebRequest (WINV24,M1)      "key1": "value1",
WebRequest (WINV24,M1)      "key2": "value2"
WebRequest (WINV24,M1)    },
WebRequest (WINV24,M1)    "origin": "200.103.20.126",
WebRequest (WINV24,M1)    "url": "https://httpbin.org/post"
WebRequest (WINV24,M1)  }
WebRequest (WINV24,M1)
```

Now, the size of the string and array are aligned, and the server correctly found the content as valid JSON. This can be provided in the response return, where the server returns the JSON sent in the "json" and "data" fields. After some configuration, we were able to perform a simple HTTP POST request, sending data correctly. However, using the WebRequest function is not trivial; it requires a good understanding of how the protocol works and the structures we are manipulating. Often, even small drawbacks can become complicated, as we saw when adjusting the code to obtain a valid response. The goal of the Connexus library is precisely to simplify this process, allowing the end user to not have to deal with these lower and more abstract layers of programming, avoiding complex problems like the one we are facing now, making the use of WebRequest more accessible and efficient.

### Conclusion

Throughout this article, we have explored in detail the use of the WebRequest function in MetaTrader 5, an essential tool for developers looking to expand the capabilities of their Expert Advisors (EAs) by communicating with external servers and APIs. We have used the httpbin.org service as a practical example to perform GET and POST requests, send and receive data in JSON format, and understand the server responses. This introduction is just the beginning of a journey of system integration via HTTP, providing the basis for much more complex projects in the future.

The journey is just beginning. Together, let's transform the WebRequest function into a simplified, powerful and accessible tool, simplifying the development of Expert Advisors and opening doors to new possibilities for automation and integration in MetaTrader 5.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/15795.zip "Download all attachments in the single ZIP archive")

[WebRequest.mq5](https://www.mql5.com/en/articles/download/15795/webrequest.mq5 "Download WebRequest.mq5")(5.14 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/473403)**
(5)


![J M](https://c.mql5.com/avatar/avatar_na2.png)

**[J M](https://www.mql5.com/en/users/joelmaloti)**
\|
4 Dec 2024 at 17:37

Hello engineer joaopedrodev!

In any case, many thanks for this very interesting article.

But just a small problem. It is that, when I try to test in real conditions the code that you provided, namely, the "WebRequest" function with the POST method, the request arrives without problem on my server. Only, the data passed in parameters (

```
"{\"key1\":\"value1\",\"key2\":\"value2\"}"
```

), they, they do not arrive on the server. Is it possible for you to have an idea to solve this? Thank you

Automatic translation applied by moderator. On the English forum, please write in English. Either use the automatic translation tool, or post in one of the other language forums.

![joaopedrodev](https://c.mql5.com/avatar/2024/9/66da07c2-0125.png)

**[joaopedrodev](https://www.mql5.com/en/users/joaopedrodev)**
\|
4 Jan 2025 at 18:15

**J M [#](https://www.mql5.com/en/forum/473403#comment_55300992):**

Hello engineer joaopedrodev!

In any case, many thanks for this very interesting article.

But just a small problem. It is that, when I try to test in real conditions the code that you provided, namely, the "WebRequest" function with the POST method, the request arrives without problem on my server. Only, the data passed in parameters (

), they, they do not arrive on the server. Is it possible for you to have an idea to solve this? Thank you

Automatic translation applied by moderator. On the English forum, please write in English. Either use the automatic translation tool, or post in one of the other language forums.

Hellow [@J M](https://www.mql5.com/en/users/joelmaloti)

Make sure you are sending this data in the body of the request, so the server will correctly receive the data you want to send. I also ask that you use the latest version of the library that is attached in the last article [Connexus Observer (Part 8): Adding a Request Observer](https://www.mql5.com/en/articles/16377)

![Good Beer](https://c.mql5.com/avatar/2021/12/61AB97ED-184E.jpg)

**[Good Beer](https://www.mql5.com/en/users/g_beer)**
\|
6 Apr 2025 at 11:37

I am also interested in the problem of using MT5 for DeFi. In general, I see that MT5, with all its attractiveness (namely, the ability to write tools with inbuilt tools) is not designed to work without brokers. it is brokers who pay for terminal support, while it is free for traders. Existing commercial [projects](https://www.mql5.com/en/articles/7863 "Article: Projects allow you to create profitable trading robots! But it's not accurate "), which allow to freely connect external resources, are quite expensive for their use. A trader's bread is usually irregular, so constant expenses are inappropriate for us. WebRequest() is not initially convenient for loading quotes via API, because APIs of different exchanges are different and are not optimised for WebRequest(). In fact, WebRequest() is a narrowly focused GET method, but in response comes a set of different types of data, which should be adapted (better) in .csv, structured for MT5. Python lists are much better suited for this than MQL arrays. But that's not the point, we need a symbol in the market overview that can load quotes from an external source (SQL for example). Through an EA it is cumbersome and requires connection to a left broker. So the free use of MT5 is still questionable, and WebRequest() is utopia.

![Janis Ozols](https://c.mql5.com/avatar/2016/8/57B428E0-C711.png)

**[Janis Ozols](https://www.mql5.com/en/users/glavforex)**
\|
7 Apr 2025 at 04:20

**Good Beer [#](https://www.mql5.com/ru/forum/484172#comment_56373493):**

But that's not the point, we need a symbol in the market overview that can load quotes from an external source (SQL, for example).

For this purpose, I created a custom symbol and a service that in the background updates its history via CustomRatesUpdate (via WebRequest) and CustomTicksAdd (via websocket) functions.

**Good Beer [#](https://www.mql5.com/ru/forum/484172#comment_56373493):**

Through an EA it is cumbersome and requires a connection to a left broker.

Through EA is really cumbersome. Use a service. You do not need to connect to a "left" broker, get quotes directly from the provider where you plan to trade.

**Good Beer [#](https://www.mql5.com/ru/forum/484172#comment_56373493):**

So free use of MT5 is still in question, and WebRequest() is utopia.

I use MT5 freely, no issues whatsoever. WebRequest is good for infrequent history updates and sending trade requests. For real time updates, use websockets.

![Good Beer](https://c.mql5.com/avatar/2021/12/61AB97ED-184E.jpg)

**[Good Beer](https://www.mql5.com/en/users/g_beer)**
\|
8 Apr 2025 at 19:33

That's right! Service! I didn't study them and forgot about them. Thank you!


![MQL5 Wizard Techniques you should know (Part 40): Parabolic SAR](https://c.mql5.com/2/94/MQL5_Wizard_Techniques_you_should_know_Part_40__LOGO.png)[MQL5 Wizard Techniques you should know (Part 40): Parabolic SAR](https://www.mql5.com/en/articles/15887)

The Parabolic Stop-and-Reversal (SAR) is an indicator for trend confirmation and trend termination points. Because it is a laggard in identifying trends its primary purpose has been in positioning trailing stop losses on open positions. We, however, explore if indeed it could be used as an Expert Advisor signal, thanks to custom signal classes of wizard assembled Expert Advisors.

![Scalping Orderflow for MQL5](https://c.mql5.com/2/94/Scalping_Orderflow_for_MQL5__LOGO2.png)[Scalping Orderflow for MQL5](https://www.mql5.com/en/articles/15895)

This MetaTrader 5 Expert Advisor implements a Scalping OrderFlow strategy with advanced risk management. It uses multiple technical indicators to identify trading opportunities based on order flow imbalances. Backtesting shows potential profitability but highlights the need for further optimization, especially in risk management and trade outcome ratios. Suitable for experienced traders, it requires thorough testing and understanding before live deployment.

![Neural Networks Made Easy (Part 88): Time-Series Dense Encoder (TiDE)](https://c.mql5.com/2/76/Neural_networks_are_easy_7Part_88j___LOGO.png)[Neural Networks Made Easy (Part 88): Time-Series Dense Encoder (TiDE)](https://www.mql5.com/en/articles/14812)

In an attempt to obtain the most accurate forecasts, researchers often complicate forecasting models. Which in turn leads to increased model training and maintenance costs. Is such an increase always justified? This article introduces an algorithm that uses the simplicity and speed of linear models and demonstrates results on par with the best models with a more complex architecture.

![Self Optimizing Expert Advisor With MQL5 And Python (Part IV): Stacking Models](https://c.mql5.com/2/94/Self_Optimizing_Expert_Advisor_With_MQL5_And_Python_Part_IV___LOGO__1.png)[Self Optimizing Expert Advisor With MQL5 And Python (Part IV): Stacking Models](https://www.mql5.com/en/articles/15886)

Today, we will demonstrate how you can build AI-powered trading applications capable of learning from their own mistakes. We will demonstrate a technique known as stacking, whereby we use 2 models to make 1 prediction. The first model is typically a weaker learner, and the second model is typically a more powerful model that learns the residuals of our weaker learner. Our goal is to create an ensemble of models, to hopefully attain higher accuracy.

[We've created a channel for MQL5 developersFollow MQL5.community on social media and be the first to receive important updatesLearn more![](https://www.mql5.com/ff/sh/a83xrgctr82w45z9z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=pwgdbvtemvkwsfltqonysypfvtrtufji&s=e99a66a1660cd810b1edbac65597df695e2c2220d1e937834f402f9aeabd4289&uid=&ref=https://www.mql5.com/en/articles/15795&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5083307826270443777)

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