---
title: HTTP and Connexus (Part 2): Understanding HTTP Architecture and Library Design
url: https://www.mql5.com/en/articles/15897
categories: Integration, Expert Advisors
relevance_score: 0
scraped_at: 2026-01-24T14:03:22.022510
---

[![](https://www.mql5.com/ff/si/3fgkjn78mkxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=ocndbzpeklfncxysjbwfhhbalbrsdbtv&s=a4309643278437a00bdd33c5809fc6b4b4032749c00fccd07b3b84e7b8b45126&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=hamxirpbinulhrdpnefpznieflpdoxrs&ssn=1769252600835189128&ssn_dr=0&ssn_sr=0&fv_date=1769252600&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F15897&back_ref=https%3A%2F%2Fwww.google.com%2F&title=HTTP%20and%20Connexus%20(Part%202)%3A%20Understanding%20HTTP%20Architecture%20and%20Library%20Design%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176925260052930332&fz_uniq=6370966655430826232&sv=2552)

MetaTrader 5 / Examples


### Introduction

This article is the continuation of a series of articles where we will build a library called Connexus. In the [first article](https://www.mql5.com/en/articles/15795), we understood the basic functioning of the WebRequest function, understanding each of its parameters and also created an example code that demonstrates the use of this function and its difficulties. In this article, we will continue to understand a little more about the HTTP protocol, how a URL works and what elements are used to build one, and create two initial classes, which are:

- **CQueryParam**: Class to manage query parameters in URL
- **CURL**: Class that contains all the elements of a URL, including an instance of CQueryParam

### What is HTTP?

HTTP is a communication protocol used to transfer data on the web. It works on top of the TCP/IP protocol, which establishes the connection between the client and the server. HTTP is a stateless protocol, which means that each request is independent, without knowledge of previous requests. An HTTP request and response is made up of three main parts:

**1\. Request Line**

The request line contains three elements:

- **HTTP Method**: Defines the action to be performed (GET, POST, PUT, DELETE, etc.).
- **URL**: Specifies the requested resource.
- **HTTP Version**: Indicates the protocol version used (HTTP/1.1, HTTP/2, etc.).

Request and response example:

```
REQUEST
GET /index.html HTTP/1.1

RESPONSE
HTTP/1.1 200 OK
```

**2. Request Headers**

Headers provide additional information about the request, such as content type, user agent (browser), and authentication. Example:

```
Host: www.exemplo.com
User-Agent: Mozilla/5.0
Accept-Language: en-US,en;q=0.5
```

**3\. Request Body**

Not all requests have a body, but it is common in methods like POST and PUT, where data (like forms or files) is sent to the server.

### Common HTTP Methods

HTTP methods are essential for determining the type of action that is being requested by the client from the server. Each method defines a specific purpose, such as fetching data, sending information, or modifying a resource. Let's take a deeper dive into the most common HTTP methods, their uses, and best practices.

- **GET**: The GET method is the most widely used in the HTTP protocol. It is used to retrieve or search for data from a server without changing its state. The main characteristic of GET is that it is idempotent, that is, making multiple GET requests for the same resource does not change the state of the application.

  - **Features**

    - **No side effects**: Only read data. Should not change anything on the server.
    - **Empty request body**: Generally, there is no body in a GET request.
    - **Cacheable**: GET responses can be cached by the browser to improve performance.

  - **When to use?**

    - Fetch static data such as HTML pages, images, files, or information in a format such as JSON.
    - Retrieve information from a database without altering data.

- **POST**: The POST method is used to send data to the server, usually to create new resources. Unlike GET, it changes the state of the server. POST is not idempotent, meaning that if you send the same request multiple times, you can create multiple resources.

  - **Features**

    - **Changes the server state:** Typically used to create new resources.
    - **Request body present:** Contains the data that will be sent to the server.
    - **Non-cacheable:** Generally, POST requests should not be cached.

  - **When to use?**

    - Submit forms with data.
    - Create new resources, such as adding a new item to a database.

- **PUT**: The PUT method is used to update a resource on the server. If the resource does not already exist, PUT can be used to create it. The main characteristic of PUT is that it is idempotent: making multiple PUT requests with the same request body will always produce the same result.

  - **Features**

    - **Idempotent:** Repeated requests with the same body will have the same effect.
    - **Full resource sent:** The request body typically contains a complete representation of the resource being updated.
    - **Can create or update:** If the resource does not exist, PUT can create it.

  - **When to use?**

    - Completely update or replace an existing resource.
    - Create a resource, if it does not exist, through a specific URL.

- **DELETE**: The DELETE method is used to remove a specific resource from the server. Like PUT, it is idempotent, which means that if you perform multiple DELETE requests for the same resource, the result will be the same: the resource will be removed (or will remain missing if it has already been deleted).

  - **Features**

    - **Idempotent:** If you delete a resource that has already been deleted, the server will return success.
    - **No body:** Normally, the DELETE request does not have a body.

  - **When to use?**

    - Remove specific server resources, such as deleting data from a database.

- **PATCH**: The PATCH method is used for partial updates of a resource. Unlike PUT, where you need to send the complete representation of the resource, PATCH allows you to modify only the fields that need updating.
- **HEAD**: Similar to GET, but without the response body. Used to check information about a resource.
- **OPTIONS**: Used to describe the communication options with the server, including the supported methods.

- **TRACE**: Used for diagnostics, returns what was sent by the client to the server.


### HTTP Status Codes

HTTP status codes are responses that the server sends to the client to inform them of the outcome of a request. These codes are numeric and indicate whether the request was successful or failed, as well as errors or redirects. They are divided into five main categories, each with a specific range of numbers, providing clear and detailed feedback about what happened during the processing of the request.

Here is a more in-depth look at each category and some of the most commonly used codes.

**1xx: Informative Responses**: Status codes in the 1xx series indicate that the server has received the request and that the client should wait for more information. These responses are rarely used in everyday practice, but can be important in certain scenarios.

**2xx: Success**: Status codes in the \*\*2xx\*\* series indicate that the request was \*\*successful\*\*. These are the codes we want to see most of the time, as they indicate that the interaction between the client and server went as expected.

**3xx: Redirects**: Codes in the 3xx series indicate that the client needs to take some additional action to complete the request, typically a redirect to another URL.

**4xx: Client Errors**: Codes in the 4xx series indicate that there was an error in the request made by the client. These errors can be caused by an incorrect data format, missing authentication, or attempts to access non-existent resources.

**5xx: Server Errors**: Codes in the 5xx series indicate that there was an internal error on the server while trying to process the request. These errors are usually backend issues, such as internal service failures, configuration errors, or system overload.

### Building a URL

A URL (Uniform Resource Locator) is the way we identify and access resources on the web. It is made up of several elements that provide essential information, such as the server location, the requested resource, and optionally additional parameters that can be used to filter or customize responses.

Below, we'll detail each component of a URL and how query parameters are used to pass additional information in an HTTP request.

**URL Structure**

A typical URL follows this format:

```
protocol://domain:port/path?query=params#fragment
```

Each part has a specific role:

1. **Protocol**: Indicates which protocol will be used for communication, such as http or https . Example: https:// .
2. **Domain**: Name of the server where the resource is hosted. It can be a domain name (e.g. example.com ) or an IP address (e.g. 192.168.1.1 ).
3. **Port**: Optional number that specifies the server port that should be used for communication. If omitted, the browser uses the default ports, such as 80 for http and 443 for https . Example: :8080 .
4. **Path**: Specifies the resource or route on the server. This can represent pages, API endpoints or files. Example: /api/v1/users .
5. **Query Params**: Used to pass additional information to the server. They follow the question mark ( ? ) and are formed by key-value pairs. Multiple parameters are separated by & . Example: ?name=John&age=30 .
6. **Fragment**: Indicates a specific part of the resource, such as an anchor point within an HTML page. Follows the hash character ( # ). Example: #section2 . Normally, **fragments are not useful** nor used to consume data from APIs. This is because fragments are processed exclusively on the client side, that is, by the browser or by the interface of the application that is consuming the web page. The server **does not receive the fragment** of the URL, and therefore it cannot be used in HTTP requests sent to the server, such as when consuming an API. For this reason, our library will not support fragments.

### Complete Example

Let's analyze the URL below:

```
https://www.exemplo.com:8080/market/symbols?active=EURUSD&timeframe=h1
```

Here we have:

- **Protocol**: https
- **Domain**: www.example.com
- **Port**: 8080
- **Path**: /market/symbols
- **Query Params**: active=EURUSD&timeframe=h1

### Hands On: Starting Construction of the Connexus Library

To start building the Connexus library, we will focus on the classes responsible for building and manipulating URLs and query parameters. We will build a module that helps in creating URLs and adding query parameters dynamically and programmatically.

### Class Structure

We will start by creating a CURL class, which will be responsible for building and manipulating URLs. It will allow the user to easily add query parameters, build the base URL, and handle different elements of the URL efficiently. To manage the **Query Parameters** of a URL in an organized and efficient way, we will use a class called CJson . The goal of this class is to convert the **query params** (which are normally passed as a string in the URL) into a structured and easy-to-manage format: **JSON**.

### What is JSON?

Before we dive into the functionality of the CJson class, it's important to understand the **JSON** (JavaScript Object Notation) format, if you're not already familiar with it. JSON is a very common data format used on the web to represent structured data. It consists of **key** pairs, where each key has an associated value. These pairs are separated by commas and grouped between curly braces “{}”

Example of a JSON object:

```
{
  "name": "John",
  "age": 30,
  "city": "New York"
}
```

Here, we have a JSON object that contains three key pairs “name”, “age” and “city”, with their respective values. In the case of **query params** of a URL, each parameter works in a similar way: there is a key (parameter name) and a value (the value associated with that key).

### Purpose of the CJson Class

The CJson class will be used to organize the URL query parameters in a **JSON** format. This makes it easier to manipulate, read, and even validate these parameters before including them in the final URL. Instead of dealing with a string of parameters like ?name=John&age=30 , you can deal with a structured object, making the code cleaner and more understandable. The CJson class will also be useful for sending and receiving data, which we will see in the next articles.

### Creating the first classes

We start by creating a folder called Connexus in includes in the Metaeditor. Inside the Connexus folder create another folder called URL and another called Data , Create a file called URL and QueryParam inside the URL folder and I will also leave attached to the article the CJson class, which should be added to the Data folder. I will not go into much detail about the implementation of this class, but it is easy to use, trust me. The structure should look something like this

```
MQL5
 |--- include
 |--- |--- Connexus
 |--- |--- |--- Data
 |--- |--- |--- |--- Json.mqh
 |--- |--- |--- URL
 |--- |--- |--- |--- QueryParam.mqh
 |--- |--- |--- |--- URL.mqh
```

### QueryParam

Let's start by working with the CQueryParam class. This class will be responsible for adding, removing, searching, and serializing query parameters, as well as offering auxiliary methods such as cleaning data and parsing query strings. We start by creating the class with a private object of type CJson, to store the query parameters as key-value pairs.

```
//+------------------------------------------------------------------+
//| class : CQueryParam                                              |
//|                                                                  |
//| [PROPERTY]                                                       |
//| Name        : CQueryParam                                        |
//| Heritage    : No heritage                                        |
//| Description : Manages query parameters for HTTP requests         |
//|                                                                  |
//+------------------------------------------------------------------+
class CQueryParam
  {
private:

   CJson             m_query_param;                      // Storage for query parameters

public:
                     CQueryParam(void);
                    ~CQueryParam(void);

   //--- Functions to manage query parameters
   void              AddParam(string key, string value); // Add a key-value pair
   void              AddParam(string param);             // Add a single key-value parameter
   void              AddParams(const string &params[]);  // Add multiple parameters
   void              RemoveParam(const string key);      // Remove a parameter by key
   string            GetParam(const string key) const;   // Retrieve a parameter value by key
   bool              HasParam(const string key);         // Check if parameter exists
   int               ParamSize(void);                    // Get the number of parameters

   //--- Auxiliary methods
   bool              Parse(const string query_param);    // Parse a query string
   string            Serialize(void);                    // Serialize parameters into a query string
   void              Clear(void);                        // Clear all parameters
  };
```

Now, let's explore the main methods and understand how each of them contributes to the functioning of the class.

- **AddParam(string key, string value)**: This method is responsible for adding a new parameter to the list of query params. It receives the key and the value as parameters and stores them in the m\_query\_param object.
- **AddParam(string param)**: This method adds a parameter already formatted as key=value . It checks if the string has the = character and, if so, splits the string into two values, one for the key and one for the value, and stores them.
- **AddParams(const string &params\[\])**: This method adds multiple parameters at once. It receives an array of strings in the key=value format and calls the AddParam method for each item in the array.
- **RemoveParam(const string key**): This method removes a parameter from the list of query params. It locates the key and removes it from the m\_query\_param object. - GetParam(const string key): This method returns the value of a specific parameter, using the key as input.
- **HasParam(const string key**): This method checks if a given parameter has already been added.
- **ParamSize(void)**: This method returns the number of stored query parameters.
- **Parse(const string query\_param**): The Parse() method receives a string of query params and converts them to key-value pairs, storing them in the m\_query\_param object. It splits the string by the characters & (which separate the parameters) and = (which separates key and value).
- **Serialize(void)**: The Serialize() method generates a formatted string containing all stored query params. It concatenates the parameters in the key=value format and separates each pair with & .
- **Clear(void)**: The Clear() method clears all stored parameters, resetting the object.

Below is the code with the implemented functions, remember to add the CJSON import:

```
//+------------------------------------------------------------------+
//| Include the file CJson class                                     |
//+------------------------------------------------------------------+
#include "../Data/Json.mqh"
//+------------------------------------------------------------------+
//| class : CQueryParam                                              |
//|                                                                  |
//| [PROPERTY]                                                       |
//| Name        : CQueryParam                                        |
//| Heritage    : No heritage                                        |
//| Description : Manages query parameters for HTTP requests         |
//|                                                                  |
//+------------------------------------------------------------------+
class CQueryParam
  {
private:

   CJson             m_query_param;                      // Storage for query parameters

public:
                     CQueryParam(void);
                    ~CQueryParam(void);

   //--- Functions to manage query parameters
   void              AddParam(string key, string value); // Add a key-value pair
   void              AddParam(string param);             // Add a single key-value parameter
   void              AddParams(const string &params[]);  // Add multiple parameters
   void              RemoveParam(const string key);      // Remove a parameter by key
   string            GetParam(const string key) const;   // Retrieve a parameter value by key
   bool              HasParam(const string key);         // Check if parameter exists
   int               ParamSize(void);                    // Get the number of parameters

   //--- Auxiliary methods
   bool              Parse(const string query_param);    // Parse a query string
   string            Serialize(void);                    // Serialize parameters into a query string
   void              Clear(void);                        // Clear all parameters
  };
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CQueryParam::CQueryParam(void)
  {
  }
//+------------------------------------------------------------------+
//| Destructor                                                       |
//+------------------------------------------------------------------+
CQueryParam::~CQueryParam(void)
  {
  }
//+------------------------------------------------------------------+
//| Adds a key-value pair to the query parameters                    |
//+------------------------------------------------------------------+
void CQueryParam::AddParam(string key, string value)
  {
   m_query_param[key] = value;
  }
//+------------------------------------------------------------------+
//| Adds a single parameter from a formatted string                  |
//+------------------------------------------------------------------+
void CQueryParam::AddParam(string param)
  {
   //--- Check if the input string contains an "=" symbol, which indicates a key-value pair
   if(StringFind(param,"=") >= 0)
     {
      //--- Declare an array to hold the key and value after splitting the string
      string key_value[];

      //--- Split the input string using "=" as the delimiter and store the result in the key_value array
      int size = StringSplit(param,StringGetCharacter("=",0),key_value);

      //--- If the size of the split result is exactly 2 (meaning a valid key-value pair was found)
      if(size == 2)
        {
         // Add the key-value pair to the m_query_param map
         // key_value[0] is the key, key_value[1] is the value
         m_query_param[key_value[0]] = key_value[1];
        }
     }
  }
//+------------------------------------------------------------------+
//| Adds multiple parameters from an array of formatted strings      |
//+------------------------------------------------------------------+
void CQueryParam::AddParams(const string &params[])
  {
   //--- Get the size of the input array 'params'
   int size = ArraySize(params);

   //--- Loop through each element in the 'params' array.
   for(int i=0;i<size;i++)
     {
      //--- Call the AddParam function to add each parameter to the m_query_param map.
      this.AddParam(params[i]);
     }
  }
//+------------------------------------------------------------------+
//| Removes a parameter by key                                       |
//+------------------------------------------------------------------+
void CQueryParam::RemoveParam(const string key)
  {
   m_query_param.Remove(key);
  }
//+------------------------------------------------------------------+
//| Retrieves a parameter value by key                               |
//+------------------------------------------------------------------+
string CQueryParam::GetParam(const string key) const
  {
   return(m_query_param[key].ToString());
  }
//+------------------------------------------------------------------+
//| Checks if a parameter exists by key                              |
//+------------------------------------------------------------------+
bool CQueryParam::HasParam(const string key)
  {
   return(m_query_param.FindKey(key) != NULL);
  }
//+------------------------------------------------------------------+
//| Returns the number of parameters stored                          |
//+------------------------------------------------------------------+
int CQueryParam::ParamSize(void)
  {
   return(m_query_param.Size());
  }
//+------------------------------------------------------------------+
//| Parses a query string into parameters                            |
//| Input: query_param - A string formatted as a query parameter     |
//| Output: bool - Always returns true, indicating successful parsing|
//+------------------------------------------------------------------+
bool CQueryParam::Parse(const string query_param)
  {
   //--- Split the input string by '&', separating the individual parameters
   string params[];
   int size = StringSplit(query_param, StringGetCharacter("&",0), params);

   //--- Iterate through each parameter string
   for(int i=0; i<size; i++)
     {
      //--- Split each parameter string by '=', separating the key and value
      string key_value[];
      StringSplit(params[i], StringGetCharacter("=",0), key_value);

      //--- Check if the split resulted in exactly two parts: key and value
      if (ArraySize(key_value) == 2)
        {
         //--- Assign the value to the corresponding key in the map
         m_query_param[key_value[0]] = key_value[1];
        }
     }
   //--- Return true indicating that parsing was successful
   return(true);
  }
//+------------------------------------------------------------------+
//| Serializes the stored parameters into a query string             |
//| Output: string - A string representing the serialized parameters |
//+------------------------------------------------------------------+
string CQueryParam::Serialize(void)
  {
   //--- Initialize an empty string to build the query parameter string
   string query_param = "";

   //--- Iterate over each key-value pair in the parameter map
   for(int i=0; i<m_query_param.Size(); i++)
     {
      //--- Append a '?' at the beginning to indicate the start of parameters
      if(i == 0)
        {
         query_param = "?";
        }

      //--- Construct each key-value pair as 'key=value'
      if(i == m_query_param.Size()-1)
        {
         //--- If it's the last pair, don't append '&'
         query_param += m_query_param[i].m_key + "=" + m_query_param[i].ToString();
        }
      else
        {
         //--- Otherwise, append '&' after each pair
         query_param += m_query_param[i].m_key + "=" + m_query_param[i].ToString() + "&";
        }
     }

   //--- Return the constructed query parameter string
   return(query_param);
  }
//+------------------------------------------------------------------+
//| Clears all stored parameters                                     |
//+------------------------------------------------------------------+
void CQueryParam::Clear(void)
  {
   m_query_param.Clear();
  }
//+------------------------------------------------------------------+
```

### URL

Now that we have a class responsible for working with query params, let's work on the CURL class that will do the rest, using protocol, host, port, etc. Here is an initial implementation of the CURL class in MQL5, remember to import the CQueryParam class:

```
//+------------------------------------------------------------------+
//|                                                          URL.mqh |
//|                                  Copyright 2023, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2023, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#include "QueryParam.mqh"
class CURL
  {
public:
                     CURL(void);
                    ~CURL(void);
  };
CURL::CURL(void)
  {
  }
CURL::~CURL(void)
  {
  }
//+------------------------------------------------------------------+
```

Let's create an ENUM that contains the most popular protocols

```
//+------------------------------------------------------------------+
//| Enum to represent different URL protocol                         |
//+------------------------------------------------------------------+
enum ENUM_URL_PROTOCOL
  {
   URL_PROTOCOL_NULL = 0,  // No protocol defined
   URL_PROTOCOL_HTTP,      // HTTP protocol
   URL_PROTOCOL_HTTPS,     // HTTPS protocol
   URL_PROTOCOL_WS,        // WebSocket (WS) protocol
   URL_PROTOCOL_WSS,       // Secure WebSocket (WSS) protocol
   URL_PROTOCOL_FTP        // FTP protocol
  };
```

In the private field of the class we will add a structure that forms the basic elements of a URL, and an instance of this structure called m\_url

```
private:

   //--- Structure to hold components of a URL
   struct MqlURL
     {
      ENUM_URL_PROTOCOL protocol;      // URL protocol
      string            host;          // Host name or IP
      uint              port;          // Port number
      string            path;          // Path after the host
      CQueryParam       query_param;   // Query parameters as key-value pairs
     };
   MqlURL            m_url;            // Instance of MqlURL to store the URL details
```

We create the setters and getters, and their implementations

```
//+------------------------------------------------------------------+
//| class : CURL                                                     |
//|                                                                  |
//| [PROPERTY]                                                       |
//| Name        : CURL                                               |
//| Heritage    : No heritage                                        |
//| Description : Define a class CURL to manage and manipulate URLs  |
//|                                                                  |
//+------------------------------------------------------------------+
class CURL
  {
public:
                     CURL(void);
                    ~CURL(void);


   //--- Methods to access and modify URL components
   ENUM_URL_PROTOCOL Protocol(void) const;                  // Get the protocol
   void              Protocol(ENUM_URL_PROTOCOL protocol);  // Set the protocol
   string            Host(void) const;                      // Get the host
   void              Host(const string host);               // Set the host
   uint              Port(void) const;                      // Get the port
   void              Port(const uint port);                 // Set the port
   string            Path(void) const;                      // Get the path
   void              Path(const string path);               // Set the path
   CQueryParam       *QueryParam(void);                     // Access query parameters
  };
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CURL::CURL(void)
  {
   this.Clear();
  }
//+------------------------------------------------------------------+
//| Destructor                                                       |
//+------------------------------------------------------------------+
CURL::~CURL(void)
  {
  }
//+------------------------------------------------------------------+
//| Getter for protocol                                              |
//+------------------------------------------------------------------+
ENUM_URL_PROTOCOL CURL::Protocol(void) const
  {
   return(m_url.protocol);
  }
//+------------------------------------------------------------------+
//| Setter for protocol                                              |
//+------------------------------------------------------------------+
void CURL::Protocol(ENUM_URL_PROTOCOL protocol)
  {
   m_url.protocol = protocol;
  }
//+------------------------------------------------------------------+
//| Getter for host                                                  |
//+------------------------------------------------------------------+
string CURL::Host(void) const
  {
   return(m_url.host);
  }
//+------------------------------------------------------------------+
//| Setter for host                                                  |
//+------------------------------------------------------------------+
void CURL::Host(const string host)
  {
   m_url.host = host;
  }
//+------------------------------------------------------------------+
//| Getter for port                                                  |
//+------------------------------------------------------------------+
uint CURL::Port(void) const
  {
   return(m_url.port);
  }
//+------------------------------------------------------------------+
//| Setter for port                                                  |
//+------------------------------------------------------------------+
void CURL::Port(const uint port)
  {
   m_url.port = port;
  }
//+------------------------------------------------------------------+
//| Getter for path                                                  |
//+------------------------------------------------------------------+
string CURL::Path(void) const
  {
   return(m_url.path);
  }
//+------------------------------------------------------------------+
//| Setter for path                                                  |
//+------------------------------------------------------------------+
void CURL::Path(const string path)
  {
   m_url.path = path;
  }
//+------------------------------------------------------------------+
//| Accessor for query parameters (returns a pointer)                |
//+------------------------------------------------------------------+
CQueryParam *CURL::QueryParam(void)
  {
   return(GetPointer(m_url.query_param));
  }
//+------------------------------------------------------------------+
```

Now we will work on the engine of our class, adding new functions to work with this data, they are:

- Clear(void) : The Clear() method is responsible for clearing all data stored in the class, resetting its attributes to empty or default values. This method is useful when you want to reuse the class instance to build a new URL or when you need to ensure that no old data is accidentally included in a new operation. In other words, it "resets" the class, removing all previously stored information.

**How it works:**


  - Sets the class attributes to empty or null, depending on the data type (empty string for protocol, domain, etc.).
  - Removes all query parameters and resets the path to the default value.
  - After calling Clear() , the class instance will be in an initial state, as if it had just been created.

**Example:**

If the class previously stored:

  - Protocol: https
  - Domain: api.example.com
  - Path: /v1/users
  - Query Params: id=123&active=true

After calling Clear() , all of these values will be reset to:

  - Protocol: ""
  - Domain: ""
  - Path: ""
  - Query Params: ""

This leaves the class ready to build a new URL from scratch.

- BaseUrl(void) : This method is responsible for generating and returning the **base part of the URL**, which is composed of the protocol (e.g., http , https ), the domain (such as www.example.com ) and, optionally, the port (e.g., :8080 ). The method ensures that the essential elements for communicating with the server are correct. This method allows you to have the flexibility to compose dynamic URLs, always starting from the base part. It can be useful when you want to reuse the base of the URL to access different resources on the same server.

- PathAndQuery(void) : This method is responsible for generating the path part of the resource and concatenating the query parameters you added earlier. The path usually specifies the resource you want to access on the server, while the query parameters allow you to provide additional details, such as filters or pagination. By separating the path and query parameters from the base URL, you can compose different parts of the URL in a more organized way. This method returns a string that can be used directly in an HTTP request or in other methods that need this structure.

- FullUrl(void) : This is the method that "compiles" all parts of the URL and returns the complete, ready-to-use URL. It combines **BaseURL()** and **PathAndQuery()** to form the final URL that you can use directly in an HTTP request. If you need the full URL to send an HTTP request, this method is the easiest way to ensure that the URL is properly formatted. It prevents errors such as forgetting to concatenate the base and query parameters.

**Example:** If the class has stored the following values:


  - Protocol: https
  - Domain: api.example.com
  - Path: /v1/users
  - Query Params: id=123&active=true

When calling Serialize() , the function will return:

```
https://api.exemplo.com/v1/users?id=123&active=true
```

- Parse(const string url) : Does the opposite of FullUrl(void) . It takes a complete URL as an argument and separates its components in an organized way. The goal is to decompose a URL into smaller parts (protocol, domain, port, path, query parameters, etc.), so that the programmer can work with these elements individually. This is especially useful if you are receiving a URL and need to understand its details or modify them programmatically.

**How it works:**


  - Receives a string containing a complete URL.
  - Analyzes (or "parses") the string, identifying each part of the URL: the protocol ( http , https ), the domain, the port (if any), the path, and any query parameters.
  - Assigns these values to the internal attributes of the class, such as protocol , host , path , queryParams . - Correctly handles separators such as :// , / , ? , and & to split the URL into its parts.

**Example:** Given the URL:

```
https://api.example.com:8080/v1/users?id=123&active=true
```

When calling Parse() , the function will assign the following values:

  - Protocol: https
  - Domain: api.example.com
  - Port: 8080
  - Path: /v1/users
  - Query Params: id=123 , active=true

This allows you to access each part of the URL programmatically, making it easier to manipulate or parse it.

- UrlProtocolToStr(ENUM\_URL\_PROTOCOL protocol) : Returns the protocol in a string, useful for converting ENUM\_URL\_PROTOCOL to a simple string, for example:

  - URL\_PROTOCOL\_HTTP → “http”
  - URL\_PROTOCOL\_HTTPS → “httpS”
  - URL\_PROTOCOL\_WSS → “wss”
  - etc…

Each of these methods plays an essential role in constructing and manipulating URLs. With these features, the Connexus library becomes highly flexible to meet the dynamic needs of APIs, whether creating URLs from scratch or parsing existing URLs. By implementing these methods, developers can compose URLs programmatically, avoiding errors and optimizing communication with servers. Below is the code with the implemented functions:

```
//+------------------------------------------------------------------+
//| Define constants for different URL protocols                     |
//+------------------------------------------------------------------+
#define HTTP "http"
#define HTTPS "https"
#define WS "ws"
#define WSS "wss"
#define FTP "ftp"
//+------------------------------------------------------------------+
//| class : CURL                                                     |
//|                                                                  |
//| [PROPERTY]                                                       |
//| Name        : CURL                                               |
//| Heritage    : No heritage                                        |
//| Description : Define a class CURL to manage and manipulate URLs  |
//|                                                                  |
//+------------------------------------------------------------------+
class CURL
  {
private:
   string            UrlProtocolToStr(ENUM_URL_PROTOCOL protocol); // Helper method to convert protocol enum to string

public:
   //--- Methods to parse and serialize the URL
   void              Clear(void);                           // Clear/reset the URL
   string            BaseUrl(void);                         // Return the base URL (protocol, host, port)
   string            PathAndQuery(void);                    // Return the path and query part of the URL
   string            FullUrl(void);                         // Return the complete URL
   bool              Parse(const string url);               // Parse a URL string into components
  };
//+------------------------------------------------------------------+
//| Convert URL protocol enum to string                              |
//+------------------------------------------------------------------+
string CURL::UrlProtocolToStr(ENUM_URL_PROTOCOL protocol)
  {
   if(protocol == URL_PROTOCOL_HTTP)   { return(HTTP);   }
   if(protocol == URL_PROTOCOL_HTTPS)  { return(HTTPS);  }
   if(protocol == URL_PROTOCOL_WS)     { return(WS);     }
   if(protocol == URL_PROTOCOL_WSS)    { return(WSS);    }
   if(protocol == URL_PROTOCOL_FTP)    { return(FTP);    }
   return(NULL);
  }
//+------------------------------------------------------------------+
//| Clear or reset the URL structure                                 |
//+------------------------------------------------------------------+
void CURL::Clear(void)
  {
   m_url.protocol = URL_PROTOCOL_NULL;
   m_url.host = "";
   m_url.port = 0;
   m_url.path = "";
   m_url.query_param.Clear();
  }
//+------------------------------------------------------------------+
//| Construct the base URL from protocol, host, and port             |
//+------------------------------------------------------------------+
string CURL::BaseUrl(void)
  {
   //--- Checks if host is not null or empty
   if(m_url.host != "" && m_url.host != NULL)
     {
      MqlURL url = m_url;

      //--- Set default protocol if not defined
      if(url.protocol == URL_PROTOCOL_NULL)
        {
         url.protocol = URL_PROTOCOL_HTTPS;
        }

      //--- Set default port based on the protocol
      if(url.port == 0)
        {
         url.port = (url.protocol == URL_PROTOCOL_HTTPS) ? 443 : 80;
        }

      //--- Construct base URL (protocol + host)
      string serialized_url = this.UrlProtocolToStr(url.protocol) + "://" + url.host;

      //--- Include port in URL only if it's not the default port for the protocol
      if(!(url.protocol == URL_PROTOCOL_HTTP && url.port == 80) &&
         !(url.protocol == URL_PROTOCOL_HTTPS && url.port == 443))
        {
         serialized_url += ":" + IntegerToString(m_url.port);
        }

      return(serialized_url);
     }
   else
     {
      return("Error: Invalid host");
     }
  }
//+------------------------------------------------------------------+
//| Construct path and query string from URL components              |
//+------------------------------------------------------------------+
string CURL::PathAndQuery(void)
  {
   MqlURL url = m_url;

   //--- Ensure path starts with a "/"
   if(url.path == "")
     {
      url.path = "/";
     }
   else if(StringGetCharacter(url.path,0) != '/')
     {
      url.path = "/" + url.path;
     }

   //--- Remove any double slashes from the path
   StringReplace(url.path,"//","/");

   //--- Check for invalid spaces in the path
   if(StringFind(url.path," ") >= 0)
     {
      return("Error: Invalid characters in path");
     }

   //--- Return the full path and query string
   return(url.path + url.query_param.Serialize());
  }
//+------------------------------------------------------------------+
//| Return the complete URL (base URL + path + query)                |
//+------------------------------------------------------------------+
string CURL::FullUrl(void)
  {
   return(this.BaseUrl() + this.PathAndQuery());
  }
//+------------------------------------------------------------------+
//| Parse a URL string and extract its components                    |
//+------------------------------------------------------------------+
bool CURL::Parse(const string url)
  {
   //--- Create an instance of MqlURL to hold the parsed data
   MqlURL urlObj;

   //--- Parse protocol from the URL
   int index_end_protocol = 0;

   //--- Check if the URL starts with "http://"
   if(StringFind(url,"http://") >= 0)
     {
      urlObj.protocol = URL_PROTOCOL_HTTP;
      index_end_protocol = 7;
     }
   else if(StringFind(url,"https://") >= 0)
     {
      urlObj.protocol = URL_PROTOCOL_HTTPS;
      index_end_protocol = 8;
     }
   else if(StringFind(url,"ws://") >= 0)
     {
      urlObj.protocol = URL_PROTOCOL_WS;
      index_end_protocol = 5;
     }
   else if(StringFind(url,"wss://") >= 0)
     {
      urlObj.protocol = URL_PROTOCOL_WSS;
      index_end_protocol = 6;
     }
   else if(StringFind(url,"ftp://") >= 0)
     {
      urlObj.protocol = URL_PROTOCOL_FTP;
      index_end_protocol = 6;
     }
   else
     {
      return(false); // Unsupported protocol
     }

   //--- Separate the endpoint part after the protocol
   string endpoint = StringSubstr(url,index_end_protocol);  // Get the URL part after the protocol
   string parts[];                                          // Array to hold the split components of the URL

   //--- Split the endpoint by the "/" character to separate path and query components
   int size = StringSplit(endpoint,StringGetCharacter("/",0),parts);

   //--- Handle the host and port part of the URL
   string host_port[];

   //--- If the first part (host) contains a colon (":"), split it into host and port
   if(StringSplit(parts[0],StringGetCharacter(":",0),host_port) > 1)
     {
      urlObj.host = host_port[0];                        // Set the host
      urlObj.port = (uint)StringToInteger(host_port[1]); // Convert and set the port
     }
   else
     {
      urlObj.host = parts[0];

      //--- Set default port based on the protocol
      if(urlObj.protocol == URL_PROTOCOL_HTTP)
        {
         urlObj.port = 80;
        }
      if(urlObj.protocol == URL_PROTOCOL_HTTPS)
        {
         urlObj.port = 443;
        }
     }

   //--- If there's no path, default to "/"
   if(size == 1)
     {
      urlObj.path += "/"; // Add a default root path "/"
     }

   //--- Loop through the remaining parts of the URL (after the host)
   for(int i=1;i<size;i++)
     {
      //--- If the path contains an empty part, return false (invalid URL)
      if(parts[i] == "")
        {
         return(false);
        }
      //--- If the part contains a "?" (indicating query parameters)
      else if(StringFind(parts[i],"?") >= 0)
        {
         string resource_query[];

         //--- Split the part by "?" to separate the resource and query
         if(StringSplit(parts[i],StringGetCharacter("?",0),resource_query) > 0)
           {
            urlObj.path += "/"+resource_query[0];
            urlObj.query_param.Parse(resource_query[1]);
           }
        }
      else
        {
         //--- Otherwise, add to the path as part of the URL
         urlObj.path += "/"+parts[i];
        }
     }

   //--- Assign the parsed URL object to the member variable
   m_url = urlObj;
   return(true);
  }
//+------------------------------------------------------------------+
```

Finally, we will add two more new functions to help developers, they are:

- **ShowData(void)**: Prints the URL elements separately, helping us debug and understand what data is stored in the class. For example:

> ```
> https://api.exemplo.com/v1/users?id=123&active=true
> ```

The function should return this:

> ```
> Protocol: https
> Host: api.exemplo.com
> Port: 443
> Path: /v1/users
> Query Param: {
>    "id":123,
>    "active":true
> }
> ```

- **Compare(CURL &url)**: This function receives another instance of the CURL class. It should return true if the URLs stored in both instances are the same, otherwise it should return false. It can be useful to avoid comparing serialized URLs, saving time. Example without the Compare() function

```
// Example without using the Compare() method
CURl url1;
CURl url2;

if(url1.FullUrl() == url2.FullUrl())
    {
     Print("Equals URL");
    }


// Example with method Compare()
CURl url1;
CURl url2;

if(url1.Compare(url2))
    {
     Print("Equals URL");
    }



```


Below is the code for implementing each of these functions:

```
//+------------------------------------------------------------------+
//| class : CURL                                                     |
//|                                                                  |
//| [PROPERTY]                                                       |
//| Name        : CURL                                               |
//| Heritage    : No heritage                                        |
//| Description : Define a class CURL to manage and manipulate URLs  |
//|                                                                  |
//+------------------------------------------------------------------+
class CURL
  {
public:
   //--- Auxiliary methods
   bool              Compare(CURL &url);                    // Compare two URLs
   string            ShowData();                            // Show URL details as a string
  };
//+------------------------------------------------------------------+
//| Compare the current URL with another URL                         |
//+------------------------------------------------------------------+
bool CURL::Compare(CURL &url)
  {
   return (m_url.protocol == url.Protocol() &&
           m_url.host == url.Host() &&
           m_url.port == url.Port() &&
           m_url.path == url.Path() &&
           m_url.query_param.Serialize() == url.QueryParam().Serialize());
  }
//+------------------------------------------------------------------+
//| Display the components of the URL as a formatted string          |
//+------------------------------------------------------------------+
string CURL::ShowData(void)
  {
   return(
      "Protocol: "+EnumToString(m_url.protocol)+"\n"+
      "Host: "+m_url.host+"\n"+
      "Port: "+IntegerToString(m_url.port)+"\n"+
      "Path: "+m_url.path+"\n"+
      "Query Param: "+m_url.query_param.Serialize()+"\n"
   );
  }
//+------------------------------------------------------------------+
```

We have finished both classes to work with URLs, let's move on to testing

### Tests

Now that we have our initial classes ready, let's create URLs through the classes and also do the reverse process, from a URL we will separate the element using the class. To perform the tests I will create a file called TestUrl.mq5 following this path Experts/Connexus/TestUrl.mq5.

```
int OnInit()
  {
   //--- Creating URL
   CURL url;
   url.Host("example.com");
   url.Path("/api/v1/data");
   Print("Test1 | # ",url.FullUrl() == "https://example.com/api/v1/data");

   //--- Changing parts of the URL
   url.Host("api.example.com");
   Print("Test2 | # ",url.FullUrl() == "https://api.example.com/api/v1/data");

   //--- Parse URL
   url.Clear();
   string url_str = "https://api.example.com/api/v1/data";
   Print("Test3 | # ",url.Parse(url_str));
   Print("Test3 | - Protocol # ",url.Protocol() == URL_PROTOCOL_HTTPS);
   Print("Test3 | - Host # ",url.Host() == "api.example.com");
   Print("Test3 | - Port # ",url.Port() == 443);
   Print("Test3 | - Path # ",url.Path() == "/api/v1/data");

//---
   return(INIT_SUCCEEDED);
  }
```

When running the EA, we have the following data in the terminal:

```
Test1 | # true
Test2 | # true
Test3 | # true
Test3 | - Protocol # true
Test3 | - Host # true
Test3 | - Port # true
Test3 | - Path # true
```

### Conclusion

In this article, we explored in depth how the HTTP protocol works, from basic concepts such as HTTP verbs (GET, POST, PUT, DELETE) to response status codes that help us interpret the return of requests. To facilitate the management of URLs in your MQL5 applications, we built the \`CQueryParam\` class, which offers a simple and efficient way to manipulate query params. In addition, we implemented the \`CURL\` class, which allows dynamic modification of parts of the URL, making the process of creating and handling HTTP requests more flexible and robust.

With these resources in hand, you already have a good foundation for integrating your applications with external APIs, facilitating communication between your code and web servers. However, we are just getting started. In the next article, we will continue our journey into the HTTP world, where we will build dedicated classes to work with the \*\*headers\*\* and \*\*body\*\* of requests, allowing even more control over HTTP interactions.

Stay tuned for upcoming posts as we build an essential library that will take your API integration skills to the next level!

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
**[Go to discussion](https://www.mql5.com/en/forum/473809)**
(1)


![Denis Kirichenko](https://c.mql5.com/avatar/2019/5/5CEDB8D2-7CB7.jpg)

**[Denis Kirichenko](https://www.mql5.com/en/users/denkir)**
\|
28 Apr 2025 at 15:28

Interesting stuff, thanks to the author.

A small spoonful of tartness. Imho, the name of CURL class is very unfortunate. It would be better to use something like CiURL. Because there may be confusion with [CURL](https://ru.wikipedia.org/wiki/CURL "https://ru.wikipedia.org/wiki/CURL").

![Risk manager for algorithmic trading](https://c.mql5.com/2/77/Risk_manager_for_algorithmic_trading___LOGO__2.png)[Risk manager for algorithmic trading](https://www.mql5.com/en/articles/14634)

The objectives of this article are to prove the necessity of using a risk manager and to implement the principles of controlled risk in algorithmic trading in a separate class, so that everyone can verify the effectiveness of the risk standardization approach in intraday trading and investing in financial markets. In this article, we will create a risk manager class for algorithmic trading. This is a logical continuation of the previous article in which we discussed the creation of a risk manager for manual trading.

![Creating an MQL5-Telegram Integrated Expert Advisor (Part 7): Command Analysis for Indicator Automation on Charts](https://c.mql5.com/2/95/Creating_an_MQL5-Telegram_Integrated_Expert_Advisor_Part_7__LOGO.png)[Creating an MQL5-Telegram Integrated Expert Advisor (Part 7): Command Analysis for Indicator Automation on Charts](https://www.mql5.com/en/articles/15962)

In this article, we explore how to integrate Telegram commands with MQL5 to automate the addition of indicators on trading charts. We cover the process of parsing user commands, executing them in MQL5, and testing the system to ensure smooth indicator-based trading

![Developing a multi-currency Expert Advisor (Part 11): Automating the optimization (first steps)](https://c.mql5.com/2/78/Developing_a_multi-currency_advisor_4Part_111___LOGO.png)[Developing a multi-currency Expert Advisor (Part 11): Automating the optimization (first steps)](https://www.mql5.com/en/articles/14741)

To get a good EA, we need to select multiple good sets of parameters of trading strategy instances for it. This can be done manually by running optimization on different symbols and then selecting the best results. But it is better to delegate this work to the program and engage in more productive activities.

![Building A Candlestick Trend Constraint Model (Part 9): Multiple Strategies Expert Advisor (I)](https://c.mql5.com/2/95/Building_A_Candlestick_Trend_Constraint_Model_Part_9____LOGO__1.png)[Building A Candlestick Trend Constraint Model (Part 9): Multiple Strategies Expert Advisor (I)](https://www.mql5.com/en/articles/15509)

Today, we will explore the possibilities of incorporating multiple strategies into an Expert Advisor (EA) using MQL5. Expert Advisors provide broader capabilities than just indicators and scripts, allowing for more sophisticated trading approaches that can adapt to changing market conditions. Find, more in this article discussion.

[![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/01.png)![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/02.png)Powerful analytics for traders of any levelAll the necessary trading reports for beginners and professionals](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/454106&a=muccpajyfystoakuukdobwigjejzmpqn&s=52daad60fa795e635264e6f94898f05493bca3b5124d4cca8eb7e82333c2ef12&uid=&ref=https://www.mql5.com/en/articles/15897&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=6370966655430826232)

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