---
title: Implementing Practical Modules from Other Languages in MQL5 (Part 02): Building the REQUESTS Library, Inspired by Python
url: https://www.mql5.com/en/articles/18728
categories: Integration, Expert Advisors
relevance_score: 9
scraped_at: 2026-01-22T17:40:42.592656
---

[![](https://www.mql5.com/ff/sh/0wxx5f0vuwq7xh89z2/01.png)VPS for 24/7 tradingContact your broker and find out how to get a free hosting subscriptionLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=nhetzvgituppcfrhndpblbihmzziogdh&s=d00c975c8bda3d8c1b29f042ad33ac81952ccea2f130a8f1ffa9015bab8ade87&uid=&ref=https://www.mql5.com/en/articles/18728&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5049297285923317992)

MetaTrader 5 / Integration


**Contents**

- [Introduction](https://www.mql5.com/en/articles/18728#para1)
- [Making a web request](https://www.mql5.com/en/articles/18728#making-web-request)
- [Different web requests in different functions](https://www.mql5.com/en/articles/18728#diff-web-requests-in-diff-functions)
- [Uploading files to the web](https://www.mql5.com/en/articles/18728#uploading-files-to-web)
- [Receiving and downloading files from the web](https://www.mql5.com/en/articles/18728#receiving-downloading-files)
- [Session and Cookies handling](https://www.mql5.com/en/articles/18728#session-cookies-handling)
- [Basic authentication](https://www.mql5.com/en/articles/18728#basic-auth)
- [Dealing with URL parameters](https://www.mql5.com/en/articles/18728#working-w-url-parameters)
- [Conclusion](https://www.mql5.com/en/articles/18728#para2)

### Introduction

An ability to send HTTP requests to the web directly from MetaTrader 5 is one of the best things that's ever happened to the MQL5 programming language. With this ability, traders can communicate with their external websites, servers, trading apps, etc.

This makes us capable of doing almost everything inside the trading platform, like getting data from external sources, sending trading notifications to our peers, and much more.

This ability has been made possible by the function [WebRequest](https://www.mql5.com/en/docs/network/webrequest) available in MQL5, which enables us to perform any HTTP action such as:

- Sending "POST" requests for sending information to external servers.
- Getting information from the web using the famous "GET" request.
- Sending PATCH requests to the web for modifying information from the server's database.
- Sending PUT requests to the web for updating values present in the server's database.


_To name a few HTTP actions._

However, this single function can be overwhelming sometimes, and it's not user-friendly.

It takes a lot of work to send a simple web request to perform any of the actions mentioned above, and more as you have to worry about HTTP methods, headers, and much more. Not to mention, you need to hardcode the process of handling the data you want to send or receive.

![](https://c.mql5.com/2/154/requests-sidebar.png)

If you've ever used a web framework or modules for similar tasks in your coding career, you might notice that most frameworks outside MQL5 are equipped to handle most of the basic tasks and operations that this WebRequest function in MQL5 doesn't.

One of the modules is [requests](https://www.mql5.com/go?link=https://pypi.org/project/requests/ "https://pypi.org/project/requests/") from Python programming language.

The Requests module — dubbed as _HTTP for Humans:_

Is a simple, yet elegant, HTTP library that allows programmers to send HTTP/1.1 requests extremely easily.

Using this library, Python developers don't need to manually add query strings in URLs or to form-encode PUT and POST data, and much more as everything is well crafted and simplified even for the "non-techy" developers.

In this article, we are going to implement a similar module with similar functions, syntax, and capabilities. Hoping to make it easy and convenient to perform webrequests in MQL5 as in Python.

### Making a Web Request

This is the main functionality in the requests module. The request function is the one that sends and receives stuff and information in different formats from the web.

Before mimicking a similar method offered in the requests module in Python, let's take a look at this function first.

_The request method in Python looks like the following._

```
def request(
    method: str | bytes,
    url: str | bytes,
    *,
    params: _Params | None = ...,
    data: _Data | None = ...,
    headers: _HeadersMapping | None = ...,
    cookies: CookieJar | _TextMapping | None = ...,
    files: _Files | None = ...,
    auth: _Auth | None = ...,
    timeout: _Timeout | None = ...,
    allow_redirects: bool = ...,
    proxies: _TextMapping | None = ...,
    hooks: _HooksInput | None = ...,
    stream: bool | None = ...,
    verify: _Verify | None = ...,
    cert: _Cert | None = ...,
    json: Any | None = None
) -> Response
```

This function takes a couple of arguments for the HTTP web request. Let's start by building it with a few parameters for now: The method, url, data, headers, timeout, and json parameters.

```
CResponse CSession::request(const string method, //HTTP method GET, POST, etc
                             const string url, //endpoint url
                             const string data = "", //The data you want to send if the method is POST or PUT
                             const string headers = "", //HTTP headers
                             const int timeout = 5000, //Request timeout (milliseconds)
                             const bool is_json=true) //Checks whether the given data is in JSON format
  {
   char data_char[];
   char result[];
   string result_headers;
   string temp_data = data;

   CResponse response; //a structure containing various response fields

//--- Managing the headers

   string temp_headers = m_headers;
   if (headers != "") // If the user provided additional headers, append them
      temp_headers += headers;

//--- Updating the headers with the information received

   if(is_json) //If the information parsed is
     {
         //--- Convert dictionary to JSON string

         CJAVal js(NULL, jtUNDEF);
         bool b = js.Deserialize(data, CP_UTF8);

         string json;
         js.Serialize(json); //Get the serialized Json outcome
         temp_data = json; //Assign the resulting serialized Json to the temporary data array

         //--- Set "Content-Type: application/json"

         temp_headers = UpdateHeader(temp_headers, "Content-Type", "application/json");
         if (MQLInfoInteger(MQL_DEBUG))
           printf("%s: %s",__FUNCTION__,temp_headers);
     }
    else
      {
         temp_headers = UpdateHeader(temp_headers, headers);
         if (MQLInfoInteger(MQL_DEBUG))
           printf("%s: %s",__FUNCTION__,temp_headers);
      }

//--- Convert data to byte array (for POST, PUT, etc.)

   if (StringToCharArray(temp_data, data_char, 0, StringLen(temp_data), CP_UTF8)<0) //Convert the data in a string format to a uchar
      {
         printf("%s, Failed to convert data to a Char array. Error = %s",__FUNCTION__,ErrorDescription(GetLastError()));
         return response;
      }

//--- Perform the WebRequest

   uint start = GetTickCount(); //starting time of the request

   int status = WebRequest(method, url, temp_headers, timeout, data_char, result, result_headers); //trigger a webrequest function
   if(status == -1)
     {
      PrintFormat("WebRequest failed with error %s", ErrorDescription(GetLastError()));
      response.status_code = 0;
      return response;
     }

//--- Fill the response struct

   response.elapsed = (GetTickCount() - start);

   string results_string = CharArrayToString(result);
   response.text = results_string;

   CJAVal js;

   if (!js.Deserialize(result))
      if (MQLInfoInteger(MQL_DEBUG))
         printf("Failed to serialize data received from WebRequest");

   response.json = js;

   response.cookies = js["cookies"].ToStr();

   response.status_code = status;
   ArrayCopy(response.content, result);
   response.headers = result_headers;
   response.url = url;
   response.ok = (status >= 200 && status < 400);
   response.reason = WebStatusText(status); // a custom helper for translating status codes

   return response;
  }
```

The request function in Python gives us two options for passing the data to a webrequest using two different arguments; the data argument for passing all the non-JSON data and the **json** argument for passing the JSON data.

Since both arguments provide the information to be fed to the request, and only one argument can be used for sending the data, i.e,. _You either send raw data (HTML, Plain text, etc) or the JSON formatted data_.

So, the function in Python detects the type of given data (JSON or otherwise) and adjusts its headers according to the headers given by the user. For example, it appends or modifies the header to Content-Type: application/json when the JSON data is given.

While we could also have both arguments in the MQL5 function as well, I find the idea so confusing and adds unnecessary complexity. So the function in our MQL5 class takes only one argument named _data_ for sending data to the web, the boolean argument named **is\_json** is the one responsible for distinguishing the information received in the variable named _data_ (between JSON and other types) inside the function.

Again, when the argument _is\_json_ is set to _true_ the function appends or modifies the received headers with the value ( _Content-Type: application/json_). But, before that, the received data from the argument _data_ is serialized to the right JSON format before sending it to the web.

```
//--- Updating the headers with the information received

   if(is_json) //If the information parsed is
     {
         //--- Convert dictionary to JSON string

         CJAVal js(NULL, jtUNDEF);
         bool b = js.Deserialize(data, CP_UTF8);

         string json;
         js.Serialize(json); //Get the serialized Json outcome
         temp_data = json; //Assign the resulting serialized Json to the temporary data array

         //--- Set "Content-Type: application/json"

         temp_headers = UpdateHeader(temp_headers, "Content-Type", "application/json");
         if (MQLInfoInteger(MQL_DEBUG))
           printf("%s: %s",__FUNCTION__,temp_headers);
     }
    else
      {
         temp_headers = UpdateHeader(temp_headers, headers);
         if (MQLInfoInteger(MQL_DEBUG))
           printf("%s: %s",__FUNCTION__,temp_headers);
      }
```

The request method offered in the requests module in Python, returns a plenty of variables according to the [HTTP Response](https://www.mql5.com/go?link=https://www.w3schools.com/python/ref_requests_response.asp "https://www.w3schools.com/python/ref_requests_response.asp"); containing information about the state of the request, data received, errors, and much more.

```
import requests

r = requests.get('https://api.github.com/events')

# Print the raw response
print("Raw response:", r)

# Status code
print("Status Code:", r.status_code)

# Reason phrase
print("Reason:", r.reason)

# URL (final URL after redirects)
print("URL:", r.url)

# Headers (dictionary)
print("Headers:")

#... other responses
```

To achieve this, we need a similar structure in our MQL5 class.

Inside the file _requests.mqh_, below is the _CResponse class_.

```
struct CResponse
  {
   int               status_code; // HTTP status code (e.g., 200, 404)
   string            text;        // Raw response body as string

   CJAVal            json;        // Parses response as JSON
   uchar             content[];   // Raw bytes of the response
   string            headers;     // Dictionary of response headers
   string            cookies;     // Cookies set by the server
   string            url;         // Final URL after redirects
   bool              ok;          // True if status_code < 400
   uint              elapsed;     // Time taken for the response in ms
   string            reason;      // Text reason (e.g., "OK", "Not Found")
  };
```

This is the structure that the function _request_ returns inside the _CSession_ class — _we'll discuss it in a second._

For the record, below is the tabulated list of all variables and information they hold inside the **_CResponse class._**

| Variable | Datatype | Description |
| --- | --- | --- |
| status\_code | int | HTTP status code returned by the server (e.g., 200=OK, 404 = Not Found, etc,.) |
| text | string | The full response body as a raw string (e.g., HTML, JSON, or text) |
| json | [CJAVal](https://www.mql5.com/en/code/13663) | A Parsed JSON object from the response body, _if serialization process was a success._ |
| content\[\] | uchar | Raw byte array of the response body (useful for binary responses). |
| headers | string | A dictionary containing HTTP response headers. _This can be converted to JSON if you want._ |
| cookies | string | Cookies set by the server (extracted from _Set-cookie_ headers, if any). |
| url | string | The final URL after any redirects. |
| ok | bool | Becomes _true_ if the status code is < 400, i.e., no client/server errors occurred in the process. |
| elapsed | uint | The time in milliseconds taken to complete the request |
| reason | string | HTTP status code in text format — human readable. |

Below is how to use the _request_ function.

_For this example to work on your machine make sure you add [https://httpbin.org](https://www.mql5.com/go?link=https://httpbin.org/ "https://httpbin.org/") (a testing URL) to the list of allowed URLs in MetaTrader 5._

![](https://c.mql5.com/2/154/setting_httpbin.org.gif)

**(a): Sending the JSON data**

```
#include <requests.mqh>
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
//---

     string json = "{\"username\": \"omega\", \"password\": \"secret123\"}";

     CResponse response = CSession::request("POST","https://httpbin.org/post", json, NULL, 5000, true);

      Print("Status Code: ", response.status_code);
      Print("Reason: ", response.reason);
      Print("URL: ", response.url);
      Print("OK: ", (string)response.ok);
      Print("Elapsed Time (ms): ", response.elapsed);

      Print("Headers:\n", response.headers);
      Print("Cookies: ", response.cookies);
      Print("Response: ",response.text);
      Print("JSON:\n", response.json["url"].ToStr());
 }
```

Outputs.

```
JF      0       08:10:33.226    Requests test (XAUUSD,D1)       CSession::request: Content-Type: application/json
GF      0       08:10:33.226    Requests test (XAUUSD,D1)
MI      0       08:10:34.578    Requests test (XAUUSD,D1)       Status Code: 200
PS      0       08:10:34.578    Requests test (XAUUSD,D1)       Reason: OK
LH      0       08:10:34.578    Requests test (XAUUSD,D1)       URL: https://httpbin.org/post
HP      0       08:10:34.578    Requests test (XAUUSD,D1)       OK: true
IF      0       08:10:34.578    Requests test (XAUUSD,D1)       Elapsed Time (ms): 1343
HO      0       08:10:34.578    Requests test (XAUUSD,D1)       Headers:
QE      0       08:10:34.578    Requests test (XAUUSD,D1)       Date: Fri, 04 Jul 2025 05:10:35 GMT
MG      0       08:10:34.578    Requests test (XAUUSD,D1)       Content-Type: application/json
HQ      0       08:10:34.578    Requests test (XAUUSD,D1)       Content-Length: 619
IH      0       08:10:34.578    Requests test (XAUUSD,D1)       Connection: keep-alive
JL      0       08:10:34.578    Requests test (XAUUSD,D1)       Server: gunicorn/19.9.0
QD      0       08:10:34.578    Requests test (XAUUSD,D1)       Access-Control-Allow-Origin: *
IO      0       08:10:34.578    Requests test (XAUUSD,D1)       Access-Control-Allow-Credentials: true
PI      0       08:10:34.578    Requests test (XAUUSD,D1)
GO      0       08:10:34.578    Requests test (XAUUSD,D1)       Cookies:
HH      0       08:10:34.578    Requests test (XAUUSD,D1)       Response: {
QL      0       08:10:34.578    Requests test (XAUUSD,D1)         "args": {},
FK      0       08:10:34.578    Requests test (XAUUSD,D1)         "data": "{\"username\":\"omega\",\"password\":\"secret123\"}",
MN      0       08:10:34.578    Requests test (XAUUSD,D1)         "files": {},
RH      0       08:10:34.578    Requests test (XAUUSD,D1)         "form": {},
OP      0       08:10:34.578    Requests test (XAUUSD,D1)         "headers": {
CH      0       08:10:34.578    Requests test (XAUUSD,D1)           "Accept": "*/*",
GM      0       08:10:34.578    Requests test (XAUUSD,D1)           "Accept-Encoding": "gzip, deflate",
HQ      0       08:10:34.578    Requests test (XAUUSD,D1)           "Accept-Language": "en;q=0.5",
GH      0       08:10:34.578    Requests test (XAUUSD,D1)           "Content-Length": "43",
IR      0       08:10:34.578    Requests test (XAUUSD,D1)           "Content-Type": "application/json",
EI      0       08:10:34.578    Requests test (XAUUSD,D1)           "Host": "httpbin.org",
NH      0       08:10:34.578    Requests test (XAUUSD,D1)           "User-Agent": "MetaTrader 5 Terminal/5.5120 (Windows NT 10.0.19045; x64)",
HK      0       08:10:34.578    Requests test (XAUUSD,D1)           "X-Amzn-Trace-Id": "Root=1-6867624b-422e528808290adf61a651a3"
IN      0       08:10:34.578    Requests test (XAUUSD,D1)         },
GD      0       08:10:34.578    Requests test (XAUUSD,D1)         "json": {
CN      0       08:10:34.578    Requests test (XAUUSD,D1)           "password": "secret123",
KD      0       08:10:34.578    Requests test (XAUUSD,D1)           "username": "omega"
QM      0       08:10:34.578    Requests test (XAUUSD,D1)         },
EJ      0       08:10:34.578    Requests test (XAUUSD,D1)         "origin": "197.250.227.26",
FQ      0       08:10:34.578    Requests test (XAUUSD,D1)         "url": "https://httpbin.org/post"
EG      0       08:10:34.578    Requests test (XAUUSD,D1)       }
PR      0       08:10:34.578    Requests test (XAUUSD,D1)
NF      0       08:10:34.578    Requests test (XAUUSD,D1)       JSON:
CR      0       08:10:34.578    Requests test (XAUUSD,D1)       https://httpbin.org/post
```

**(b): Sending the non-JSON data**

Using form data as an example.

```
#include <requests.mqh>
CSession requests;
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
//---

     string form_data = "username=omega&password=secret123"; //mimicking how the final form data is collected in the web

     CResponse response = CSession::post("https://httpbin.org/post", form_data, "Content-Type: application/x-www-form-urlencoded", 5000, false);

      Print("Status Code: ", response.status_code);
      Print("Reason: ", response.reason);
      Print("URL: ", response.url);
      Print("OK: ", (string)response.ok);
      Print("Elapsed Time (ms): ", response.elapsed);

      Print("Headers:\n", response.headers);
      Print("Cookies: ", response.cookies);
      Print("Response: ",response.text);
      Print("JSON:\n", response.json["url"].ToStr());
 }
```

Outputs.

```
DD      0       08:20:01.411    Requests test (XAUUSD,D1)       Status Code: 200
IN      0       08:20:01.411    Requests test (XAUUSD,D1)       Reason: OK
EG      0       08:20:01.411    Requests test (XAUUSD,D1)       URL: https://httpbin.org/post
QM      0       08:20:01.411    Requests test (XAUUSD,D1)       OK: true
RI      0       08:20:01.411    Requests test (XAUUSD,D1)       Elapsed Time (ms): 1547
QR      0       08:20:01.411    Requests test (XAUUSD,D1)       Headers:
EH      0       08:20:01.411    Requests test (XAUUSD,D1)       Date: Fri, 04 Jul 2025 05:20:02 GMT
DL      0       08:20:01.411    Requests test (XAUUSD,D1)       Content-Type: application/json
MD      0       08:20:01.411    Requests test (XAUUSD,D1)       Content-Length: 587
PM      0       08:20:01.411    Requests test (XAUUSD,D1)       Connection: keep-alive
OK      0       08:20:01.411    Requests test (XAUUSD,D1)       Server: gunicorn/19.9.0
HS      0       08:20:01.411    Requests test (XAUUSD,D1)       Access-Control-Allow-Origin: *
PD      0       08:20:01.411    Requests test (XAUUSD,D1)       Access-Control-Allow-Credentials: true
IF      0       08:20:01.411    Requests test (XAUUSD,D1)
RD      0       08:20:01.411    Requests test (XAUUSD,D1)       Cookies:
QM      0       08:20:01.411    Requests test (XAUUSD,D1)       Response: {
HK      0       08:20:01.411    Requests test (XAUUSD,D1)         "args": {},
OR      0       08:20:01.411    Requests test (XAUUSD,D1)         "data": "",
JE      0       08:20:01.411    Requests test (XAUUSD,D1)         "files": {},
RL      0       08:20:01.411    Requests test (XAUUSD,D1)         "form": {
DF      0       08:20:01.411    Requests test (XAUUSD,D1)           "password": "secret123",
PM      0       08:20:01.411    Requests test (XAUUSD,D1)           "username": "omega"
RE      0       08:20:01.411    Requests test (XAUUSD,D1)         },
PL      0       08:20:01.411    Requests test (XAUUSD,D1)         "headers": {
DE      0       08:20:01.411    Requests test (XAUUSD,D1)           "Accept": "*/*",
LP      0       08:20:01.411    Requests test (XAUUSD,D1)           "Accept-Encoding": "gzip, deflate",
CF      0       08:20:01.411    Requests test (XAUUSD,D1)           "Accept-Language": "en;q=0.5",
EK      0       08:20:01.411    Requests test (XAUUSD,D1)           "Content-Length": "33",
EL      0       08:20:01.411    Requests test (XAUUSD,D1)           "Content-Type": "application/x-www-form-urlencoded",
PH      0       08:20:01.411    Requests test (XAUUSD,D1)           "Host": "httpbin.org",
GO      0       08:20:01.411    Requests test (XAUUSD,D1)           "User-Agent": "MetaTrader 5 Terminal/5.5120 (Windows NT 10.0.19045; x64)",
FQ      0       08:20:01.411    Requests test (XAUUSD,D1)           "X-Amzn-Trace-Id": "Root=1-68676482-6439a0a071c275773681a193"
LM      0       08:20:01.411    Requests test (XAUUSD,D1)         },
JE      0       08:20:01.411    Requests test (XAUUSD,D1)         "json": null,
LM      0       08:20:01.411    Requests test (XAUUSD,D1)         "origin": "197.250.227.26",
KH      0       08:20:01.411    Requests test (XAUUSD,D1)         "url": "https://httpbin.org/post"
LN      0       08:20:01.411    Requests test (XAUUSD,D1)       }
IK      0       08:20:01.411    Requests test (XAUUSD,D1)
CO      0       08:20:01.411    Requests test (XAUUSD,D1)       JSON:
NK      0       08:20:01.411    Requests test (XAUUSD,D1)       https://httpbin.org/post
```

Awesome! We were able to send two distinct data types using the same Web request function.

Notice that, while sending the non-JSON data type (Form data in this case), I explicitly set  "Content-type" in the headers argument of the request function to accommodate the form data.

So, unless you are sending the JSON data which is automatically serialized and the right HTTP header is automatically updated to accommodate this data type, you have to explicitly set the Content-type to accommodate the kind of data you want to send.

_For more information read_ -> [https://beeceptor.com/docs/concepts/content-type/index.html](https://www.mql5.com/go?link=https://beeceptor.com/docs/concepts/content-type/index.html "https://beeceptor.com/docs/concepts/content-type/index.html")

Now, this _request_ function is capable of sending all kinds of HTTP requests similarly to the native MQL5 function, _This is  an extension of it roughly._

Since this function is so flexible, it becomes overwhelming and error-prone. Suppose you want to send a GET request to receive some information from the web.

We all know that it is inappropriate to send data with the GET request because it is not meant to do so in the first place. To reduce the room for errors, the requests module in Python has several high-level functions for sending various types of webrequests that takes into account the needs of a particular request.

### Different Web Requests in Different Functions

We can build several functions on top of the request function implemented in the previous section, for different HTTP actions as follows:

**(a): The get Function**

```
static CResponse         get(const string url, const string headers = "", const int timeout = 5000)
  {
    return request("GET", url, "", headers, timeout, false);
  }
```

This function sends a GET request to the specified URL. No data is sent to the URL when this function is called as it is meant for receiving the information only.

Usage.

```
#include <requests.mqh>
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
//---

      CResponse response = CSession::get("https://httpbin.org/get");

      Print("Status Code: ", response.status_code);
      Print("Reason: ", response.reason);
      Print("URL: ", response.url);
      Print("OK: ", (string)response.ok);
      Print("Elapsed Time (ms): ", response.elapsed);

      Print("Headers:\n", response.headers);
      Print("Cookies: ", response.cookies);
      Print("Response: ",response.text);
      Print("JSON:\n", response.json["url"].ToStr());
  }
```

Outputs.

```
LM      0       09:50:48.904    Requests test (XAUUSD,D1)       Status Code: 200
QF      0       09:50:48.904    Requests test (XAUUSD,D1)       Reason: OK
GQ      0       09:50:48.904    Requests test (XAUUSD,D1)       URL: https://httpbin.org/get
CD      0       09:50:48.904    Requests test (XAUUSD,D1)       OK: true
EQ      0       09:50:48.904    Requests test (XAUUSD,D1)       Elapsed Time (ms): 1782
KJ      0       09:50:48.904    Requests test (XAUUSD,D1)       Headers:
JQ      0       09:50:48.904    Requests test (XAUUSD,D1)       Date: Fri, 04 Jul 2025 06:50:49 GMT
JD      0       09:50:48.904    Requests test (XAUUSD,D1)       Content-Type: application/json
NL      0       09:50:48.904    Requests test (XAUUSD,D1)       Content-Length: 379
NE      0       09:50:48.904    Requests test (XAUUSD,D1)       Connection: keep-alive
MS      0       09:50:48.904    Requests test (XAUUSD,D1)       Server: gunicorn/19.9.0
NH      0       09:50:48.904    Requests test (XAUUSD,D1)       Access-Control-Allow-Origin: *
FL      0       09:50:48.904    Requests test (XAUUSD,D1)       Access-Control-Allow-Credentials: true
KM      0       09:50:48.904    Requests test (XAUUSD,D1)
LL      0       09:50:48.904    Requests test (XAUUSD,D1)       Cookies:
CE      0       09:50:48.904    Requests test (XAUUSD,D1)       Response: {
FS      0       09:50:48.904    Requests test (XAUUSD,D1)         "args": {},
DJ      0       09:50:48.904    Requests test (XAUUSD,D1)         "headers": {
PR      0       09:50:48.904    Requests test (XAUUSD,D1)           "Accept": "*/*",
DF      0       09:50:48.904    Requests test (XAUUSD,D1)           "Accept-Encoding": "gzip, deflate",
KO      0       09:50:48.904    Requests test (XAUUSD,D1)           "Accept-Language": "en;q=0.5",
JL      0       09:50:48.904    Requests test (XAUUSD,D1)           "Host": "httpbin.org",
QK      0       09:50:48.904    Requests test (XAUUSD,D1)           "User-Agent": "MetaTrader 5 Terminal/5.5120 (Windows NT 10.0.19045; x64)",
JI      0       09:50:48.904    Requests test (XAUUSD,D1)           "X-Amzn-Trace-Id": "Root=1-686779c9-147d77346060e72a2fa2282f"
FH      0       09:50:48.904    Requests test (XAUUSD,D1)         },
RI      0       09:50:48.904    Requests test (XAUUSD,D1)         "origin": "197.250.227.26",
CJ      0       09:50:48.904    Requests test (XAUUSD,D1)         "url": "https://httpbin.org/get"
PR      0       09:50:48.904    Requests test (XAUUSD,D1)       }
QG      0       09:50:48.904    Requests test (XAUUSD,D1)
KK      0       09:50:48.904    Requests test (XAUUSD,D1)       JSON:
DQ      0       09:50:48.904    Requests test (XAUUSD,D1)       https://httpbin.org/get
```

_While the get function enables you to pass the headers to the HTTP request, you can not control the type of content received from the server even if you set your headers to some Content-type **.**_

**(b): The post Function**

```
static CResponse         post(const string url, const string data = "", const string headers = "", const int timeout = 5000, const bool is_json=true)
  {
    return request("POST", url, data, headers, timeout, is_json);
  }
```

This function sends a POST request to the given URL with an optional data payload. Similary to the _request_ function, it automatically sets _Content-Type_ when the argument _is\_json_ is set to _true_.

Usage.

```
#include <requests.mqh>
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
//---

      string json = "{\"username\": \"omega\", \"password\": \"secret123\"}";
      CResponse response = CSession::post("https://httpbin.org/post",json);

      Print("Status Code: ", response.status_code);
      Print("Reason: ", response.reason);
      Print("URL: ", response.url);
      Print("OK: ", (string)response.ok);
      Print("Elapsed Time (ms): ", response.elapsed);

      Print("Headers:\n", response.headers);
      Print("Cookies: ", response.cookies);
      Print("Response: ",response.text);
      Print("JSON:\n", response.json["url"].ToStr());
 }
```

Outputs.

```
NK      0       15:32:13.093    Requests test (XAUUSD,D1)       Status Code: 200
OP      0       15:32:13.093    Requests test (XAUUSD,D1)       Reason: OK
KE      0       15:32:13.093    Requests test (XAUUSD,D1)       URL: https://httpbin.org/post
GR      0       15:32:13.093    Requests test (XAUUSD,D1)       OK: true
LD      0       15:32:13.093    Requests test (XAUUSD,D1)       Elapsed Time (ms): 1578
GL      0       15:32:13.093    Requests test (XAUUSD,D1)       Headers:
PK      0       15:32:13.093    Requests test (XAUUSD,D1)       Date: Fri, 04 Jul 2025 12:32:13 GMT
NR      0       15:32:13.093    Requests test (XAUUSD,D1)       Content-Type: application/json
GG      0       15:32:13.093    Requests test (XAUUSD,D1)       Content-Length: 619
JN      0       15:32:13.093    Requests test (XAUUSD,D1)       Connection: keep-alive
II      0       15:32:13.093    Requests test (XAUUSD,D1)       Server: gunicorn/19.9.0
RF      0       15:32:13.093    Requests test (XAUUSD,D1)       Access-Control-Allow-Origin: *
JR      0       15:32:13.093    Requests test (XAUUSD,D1)       Access-Control-Allow-Credentials: true
OK      0       15:32:13.093    Requests test (XAUUSD,D1)
HQ      0       15:32:13.093    Requests test (XAUUSD,D1)       Cookies:
GK      0       15:32:13.093    Requests test (XAUUSD,D1)       Response: {
RN      0       15:32:13.093    Requests test (XAUUSD,D1)         "args": {},
EN      0       15:32:13.093    Requests test (XAUUSD,D1)         "data": "{\"username\":\"omega\",\"password\":\"secret123\"}",
NL      0       15:32:13.093    Requests test (XAUUSD,D1)         "files": {},
QJ      0       15:32:13.093    Requests test (XAUUSD,D1)         "form": {},
PS      0       15:32:13.093    Requests test (XAUUSD,D1)         "headers": {
DJ      0       15:32:13.093    Requests test (XAUUSD,D1)           "Accept": "*/*",
HO      0       15:32:13.093    Requests test (XAUUSD,D1)           "Accept-Encoding": "gzip, deflate",
GG      0       15:32:13.093    Requests test (XAUUSD,D1)           "Accept-Language": "en;q=0.5",
HJ      0       15:32:13.093    Requests test (XAUUSD,D1)           "Content-Length": "43",
JM      0       15:32:13.093    Requests test (XAUUSD,D1)           "Content-Type": "application/json",
FK      0       15:32:13.093    Requests test (XAUUSD,D1)           "Host": "httpbin.org",
MN      0       15:32:13.093    Requests test (XAUUSD,D1)           "User-Agent": "MetaTrader 5 Terminal/5.5120 (Windows NT 10.0.19045; x64)",
IM      0       15:32:13.093    Requests test (XAUUSD,D1)           "X-Amzn-Trace-Id": "Root=1-6867c9cd-64a83cd77115a2575214fecd"
JS      0       15:32:13.093    Requests test (XAUUSD,D1)         },
HJ      0       15:32:13.093    Requests test (XAUUSD,D1)         "json": {
DL      0       15:32:13.093    Requests test (XAUUSD,D1)           "password": "secret123",
LK      0       15:32:13.093    Requests test (XAUUSD,D1)           "username": "omega"
RS      0       15:32:13.093    Requests test (XAUUSD,D1)         },
FG      0       15:32:13.093    Requests test (XAUUSD,D1)         "origin": "197.250.227.26",
EO      0       15:32:13.093    Requests test (XAUUSD,D1)         "url": "https://httpbin.org/post"
FE      0       15:32:13.093    Requests test (XAUUSD,D1)       }
OL      0       15:32:13.093    Requests test (XAUUSD,D1)
MD      0       15:32:13.093    Requests test (XAUUSD,D1)       JSON:
DD      0       15:32:13.093    Requests test (XAUUSD,D1)       https://httpbin.org/post
```

**(c): The put Function**

This function sends a PUT request to update a resource at the URL with the given data.

```
static CResponse         put(const string url, const string data = "", const string headers = "", const int timeout = 5000, const bool is_json=true)
  {
    return request("PUT", url, data, headers, timeout, is_json);
  }
```

Example usage.

```
#include <requests.mqh>
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
//---

      string json = "{\"username\": \"omega\", \"password\": \"secret123\"}";
      CResponse response = CSession::put("https://httpbin.org/put",json);

      Print("Status Code: ", response.status_code);
      Print("Reason: ", response.reason);
      Print("URL: ", response.url);
      Print("OK: ", (string)response.ok);
      Print("Elapsed Time (ms): ", response.elapsed);

      Print("Headers:\n", response.headers);
      Print("Cookies: ", response.cookies);
      Print("Response: ",response.text);
      Print("JSON:\n", response.json["url"].ToStr());
 }
```

Outputs.

```
#include <requests.mqh>
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
//---

      string json = "{\"update\": true}";
      CResponse response = CSession::put("https://httpbin.org/put", json, "", 5000, true);

      Print("Reason: ", response.reason);
      Print("Response: ",response.text);
  }
```

Outputs.

```
MG      0       15:51:02.874    Requests test (XAUUSD,D1)       Reason: OK
EQ      0       15:51:02.874    Requests test (XAUUSD,D1)       Response: {
HG      0       15:51:02.874    Requests test (XAUUSD,D1)         "args": {},
JM      0       15:51:02.874    Requests test (XAUUSD,D1)         "data": "{\"update\":true}",
LJ      0       15:51:02.874    Requests test (XAUUSD,D1)         "files": {},
CM      0       15:51:02.874    Requests test (XAUUSD,D1)         "form": {},
FE      0       15:51:02.874    Requests test (XAUUSD,D1)         "headers": {
RO      0       15:51:02.874    Requests test (XAUUSD,D1)           "Accept": "*/*",
JI      0       15:51:02.874    Requests test (XAUUSD,D1)           "Accept-Encoding": "gzip, deflate",
QL      0       15:51:02.874    Requests test (XAUUSD,D1)           "Accept-Language": "en;q=0.5",
KE      0       15:51:02.874    Requests test (XAUUSD,D1)           "Content-Length": "15",
LG      0       15:51:02.874    Requests test (XAUUSD,D1)           "Content-Type": "application/json",
PL      0       15:51:02.874    Requests test (XAUUSD,D1)           "Host": "httpbin.org",
KD      0       15:51:02.874    Requests test (XAUUSD,D1)           "User-Agent": "MetaTrader 5 Terminal/5.5120 (Windows NT 10.0.19045; x64)",
KH      0       15:51:02.874    Requests test (XAUUSD,D1)           "X-Amzn-Trace-Id": "Root=1-6867ce36-76b6e4053cd661ec5e4faa1b"
HI      0       15:51:02.874    Requests test (XAUUSD,D1)         },
NG      0       15:51:02.874    Requests test (XAUUSD,D1)         "json": {
DQ      0       15:51:02.874    Requests test (XAUUSD,D1)           "update": true
PG      0       15:51:02.874    Requests test (XAUUSD,D1)         },
PS      0       15:51:02.874    Requests test (XAUUSD,D1)         "origin": "197.250.227.26",
HD      0       15:51:02.874    Requests test (XAUUSD,D1)         "url": "https://httpbin.org/put"
RH      0       15:51:02.874    Requests test (XAUUSD,D1)       }
CI      0       15:51:02.874    Requests test (XAUUSD,D1)
```

**(d): The patch Function**

```
static CResponse         patch(const string url, const string data = "", const string headers = "", const int timeout = 5000, const bool is_json=true)
  {
    return request("PATCH", url, data, headers, timeout, is_json);
  }
```

This function sends a PATCH request to partially update a resource at the URL with the provided data.

Example usage.

```
#include <requests.mqh>
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
//---

      string json = "{\"patched\": 1}";
      CResponse response = CSession::patch("https://httpbin.org/patch", json, "", 5000, true);

      Print("Reason: ", response.reason);
      Print("Response: ",response.text);
  }
```

Outputs.

```
GR      0       16:33:45.258    Requests test (XAUUSD,D1)       Reason: OK
OF      0       16:33:45.258    Requests test (XAUUSD,D1)       Response: {
RR      0       16:33:45.258    Requests test (XAUUSD,D1)         "args": {},
GJ      0       16:33:45.258    Requests test (XAUUSD,D1)         "data": "{\"patched\":1}",
NL      0       16:33:45.258    Requests test (XAUUSD,D1)         "files": {},
IK      0       16:33:45.258    Requests test (XAUUSD,D1)         "form": {},
HS      0       16:33:45.258    Requests test (XAUUSD,D1)         "headers": {
LE      0       16:33:45.258    Requests test (XAUUSD,D1)           "Accept": "*/*",
HO      0       16:33:45.258    Requests test (XAUUSD,D1)           "Accept-Encoding": "gzip, deflate",
OF      0       16:33:45.258    Requests test (XAUUSD,D1)           "Accept-Language": "en;q=0.5",
CJ      0       16:33:45.258    Requests test (XAUUSD,D1)           "Content-Length": "13",
RM      0       16:33:45.258    Requests test (XAUUSD,D1)           "Content-Type": "application/json",
FK      0       16:33:45.258    Requests test (XAUUSD,D1)           "Host": "httpbin.org",
MN      0       16:33:45.258    Requests test (XAUUSD,D1)           "User-Agent": "MetaTrader 5 Terminal/5.5120 (Windows NT 10.0.19045; x64)",
GN      0       16:33:45.258    Requests test (XAUUSD,D1)           "X-Amzn-Trace-Id": "Root=1-6867d837-47b9eb772b7ec6300016aa79"
JS      0       16:33:45.258    Requests test (XAUUSD,D1)         },
HJ      0       16:33:45.258    Requests test (XAUUSD,D1)         "json": {
MR      0       16:33:45.258    Requests test (XAUUSD,D1)           "patched": 1
FH      0       16:33:45.258    Requests test (XAUUSD,D1)         },
RI      0       16:33:45.258    Requests test (XAUUSD,D1)         "origin": "197.250.227.26",
KK      0       16:33:45.258    Requests test (XAUUSD,D1)         "url": "https://httpbin.org/patch"
DS      0       16:33:45.258    Requests test (XAUUSD,D1)       }
```

**(e): The delete Function**

```
static CResponse         delete_(const string url, const string headers = "", const int timeout = 5000, const bool is_json=true)
  {
    return request("DELETE", url, "", headers, timeout, is_json);
  }
```

This function sends a DELETE request to remove a resource at the given URL. No data payload is used.

Example usage.

```
#include <requests.mqh>
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
//---

      CResponse response = CSession::delete_("https://httpbin.org/delete", "", 5000, true);

      Print("Reason: ", response.reason);
      Print("Response: ",response.text);
 }
```

Outputs.

```
ML      0       16:43:03.046    Requests test (XAUUSD,D1)       Reason: OK
EL      0       16:43:03.046    Requests test (XAUUSD,D1)       Response: {
HH      0       16:43:03.046    Requests test (XAUUSD,D1)         "args": {},
OR      0       16:43:03.046    Requests test (XAUUSD,D1)         "data": "",
JD      0       16:43:03.046    Requests test (XAUUSD,D1)         "files": {},
ER      0       16:43:03.046    Requests test (XAUUSD,D1)         "form": {},
DK      0       16:43:03.046    Requests test (XAUUSD,D1)         "headers": {
PR      0       16:43:03.046    Requests test (XAUUSD,D1)           "Accept": "*/*",
LG      0       16:43:03.046    Requests test (XAUUSD,D1)           "Accept-Encoding": "gzip, deflate",
KO      0       16:43:03.046    Requests test (XAUUSD,D1)           "Accept-Language": "en;q=0.5",
QQ      0       16:43:03.046    Requests test (XAUUSD,D1)           "Content-Length": "0",
HE      0       16:43:03.046    Requests test (XAUUSD,D1)           "Content-Type": "application/json",
LS      0       16:43:03.046    Requests test (XAUUSD,D1)           "Host": "httpbin.org",
GF      0       16:43:03.046    Requests test (XAUUSD,D1)           "User-Agent": "MetaTrader 5 Terminal/5.5120 (Windows NT 10.0.19045; x64)",
JK      0       16:43:03.046    Requests test (XAUUSD,D1)           "X-Amzn-Trace-Id": "Root=1-6867da67-1926ebf246353ab24461aed9"
LK      0       16:43:03.046    Requests test (XAUUSD,D1)         },
FL      0       16:43:03.046    Requests test (XAUUSD,D1)         "json": null,
PG      0       16:43:03.046    Requests test (XAUUSD,D1)         "origin": "197.250.227.26",
PO      0       16:43:03.046    Requests test (XAUUSD,D1)         "url": "https://httpbin.org/delete"
HE      0       16:43:03.046    Requests test (XAUUSD,D1)       }
QL      0       16:43:03.046    Requests test (XAUUSD,D1)
```

### Uploading Files to the Web

The _requests_ module in Python makes it effortless to share and receive files from the internet, unlike the built-in function for Web request in MQL5.

Being able to send files to the web is a necessity for effective communications in today's world. We often want to send chart screenshots to show our trading progress and sometimes demonstrate trading setups and some sort of visual signals directly from the MetaTrader 5 chart.

This is the trickiest part in our CSession-MQL5 class because we have to be mindful of the file types users can upload to the web and apply the right encoding to the binary information extracted directly from the files. Not to mention, we need the right HTTP header(s) for each type of file.

```
CResponse CSession::request(const string method,
                             const string url,
                             const string data,
                             const string &files[],
                             const string headers = "",
                             const int timeout = 5000,
                             const bool is_json=true)
 {
   char result[];
   string result_headers;
   string temp_headers = m_headers;
   string boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"; //for setting boundaries between data types and files in the form data
   CArrayChar final_body; //Final body uchar array

   CResponse response; //a structure containing various response fields

   // Append user headers
   if (headers != "")
      temp_headers += headers;

   bool use_multipart = ArraySize(files) > 0; //Check if files are attached

//--- Create a multi part request

   if (use_multipart) // If multipart, assemble full body (JSON + files)
    {
      temp_headers = UpdateHeader(temp_headers, "Content-Type", "multipart/form-data; boundary=" + boundary + "\r\n"); //Update the headers

      //--- JSON part (or form data)
      if (StringLen(data) > 0)
      {
         string json_data = "";
         if (is_json) //if Json data is given alongside the files
          {
            CJAVal js(NULL, jtUNDEF);
            if (js.Deserialize(data, CP_UTF8))
               js.Serialize(json_data); //Serialize the JSON data
          }

         string json_part = "--" + boundary + "\r\n";
         json_part += "Content-Disposition: form-data; name=\"metadata\"\r\n";
         json_part += "Content-Type: application/json\r\n\r\n";
         json_part += json_data + "\r\n";

         char json_bytes[];
         StringToCharArray(json_part, json_bytes, 0, StringLen(json_part), CP_UTF8);

         final_body.AddArray(json_bytes);
      }

      //--- File parts
      for (uint i = 0; i < files.Size(); i++)
       {
         string filename = GetFileName(files[i]);

         char file_data[]; //for storing the file data in binary format

         int file_handle = FileOpen(filename, FILE_BIN | FILE_SHARE_READ); // Read the file in binary format
         if (file_handle == INVALID_HANDLE)
          {
            printf("func=%s line=%d, Failed to read the file '%s'. Error = %s",__FUNCTION__,__LINE__,filename,ErrorDescription(GetLastError()));
            continue; //skip to the next file if the current file is invalid
          }

         int fsize = (int)FileSize(file_handle);
         ArrayResize(file_data, fsize);
         if (FileReadArray(file_handle, file_data, 0, fsize)==0)
           {
              printf("func=%s line=%d, No data found in the file '%s'. Error = %s",__FUNCTION__,__LINE__,filename,ErrorDescription(GetLastError()));
              FileClose(file_handle);
              continue; //skip to the next file if the current file is invalid
           }

         FileClose(file_handle); //close the current file

         //--- Append files header and content type as detected to the request

         string file_part = "--" + boundary + "\r\n";
         file_part += StringFormat("Content-Disposition: form-data; name=\"file\"; filename=\"%s\"\r\n", filename);
         file_part += StringFormat("Content-Type: %s\r\n\r\n", GuessContentType(filename));

         char file_header[];
         StringToCharArray(file_part, file_header, 0, StringLen(file_part), CP_UTF8); //UTF-8 Encoding is a must

         final_body.AddArray(file_header); //Add the file header
         final_body.AddArray(file_data); //Add the file in binary format, the actual file

         //--- append the new line — critical for HTTP form parsing.

         final_body.Add('\r');
         final_body.Add('\n');
      }

      //--- Final boundary
      string closing = "--" + boundary + "--\r\n";
      char closing_part[];
      StringToCharArray(closing, closing_part);

      final_body.AddArray(closing_part);
    }
   else // no files attached
    {
      //--- If it's just JSON or plain form data
      string body_data = data;
      if (is_json)
       {
         CJAVal js(NULL, jtUNDEF);
         if (js.Deserialize(data, CP_UTF8))
            js.Serialize(body_data);

         temp_headers = UpdateHeader(temp_headers, "Content-Type", "application/json");
       }
      else
         temp_headers = UpdateHeader(temp_headers, headers);

      //---

      char array[];
      StringToCharArray(body_data, array, 0, StringLen(body_data), CP_UTF8); //Use UTF-8 similar requests in Python, This is very crucial
      final_body.AddArray(array);
    }

   char final_body_char_arr[];
   CArray2Array(final_body, final_body_char_arr);

   if (MQLInfoInteger(MQL_DEBUG))
     Print("Final body:\n",CharArrayToString(final_body_char_arr, 0 , final_body.Total(), CP_UTF8));

//--- Send the request

   uint start = GetTickCount(); //starting time of the request
   int status = WebRequest(method, url, temp_headers, timeout, final_body_char_arr, result, result_headers); //trigger a webrequest function

   if(status == -1)
     {
      PrintFormat("WebRequest failed with error %s", ErrorDescription(GetLastError()));
      response.status_code = 0;
      return response;
     }

//--- Fill the response struct

   response.elapsed = GetTickCount() - start;
   response.text = CharArrayToString(result);
   response.status_code = status;
   response.headers = result_headers;
   response.url = url;
   response.ok = (status >= 200 && status < 400);
   response.reason = WebStatusText(status);
   ArrayCopy(response.content, result);

//---

   CJAVal js;
   if (js.Deserialize(response.text))
      response.json = js;

   return response;
 }
```

Similarly to the previous request function for performing any request to the web, this one does a similar task but, it is capable of detecting the files given by the user in the function argument and embed them in the HTTP request.

When the files array is empty, i.e,. The user gives no files. The above function performs a regular HTTP request as discussed prior, but when the files are received, it updates the HTTP header into a multipart/form-data content type that enables the HTTP request to distinguish between different information and data types given.

```
temp_headers = UpdateHeader(temp_headers, "Content-Type", "multipart/form-data; boundary=" + boundary + "\r\n"); //Update the headers
```

The _final\_body_  array is responsible for gluing all the data together (content and files) into a single character (char) array variable, similarly to what a form does on the web. This is done inside a loop that iterates throught the files array which carries all the files you'd like to send to the server at once.

```
      //--- File parts
      for (uint i = 0; i < files.Size(); i++)
       {
         string filename = GetFileName(files[i]);

         char file_data[]; //for storing the file data in binary format

         int file_handle = FileOpen(filename, FILE_BIN | FILE_SHARE_READ); // Read the file in binary format
         if (file_handle == INVALID_HANDLE)
          {
            printf("func=%s line=%d, Failed to read the file '%s'. Error = %s",__FUNCTION__,__LINE__,filename,ErrorDescription(GetLastError()));
            continue; //skip to the next file if the current file is invalid
          }

         int fsize = (int)FileSize(file_handle);
         ArrayResize(file_data, fsize);
         if (FileReadArray(file_handle, file_data, 0, fsize)==0)
           {
              printf("func=%s line=%d, No data found in the file '%s'. Error = %s",__FUNCTION__,__LINE__,filename,ErrorDescription(GetLastError()));
              FileClose(file_handle);
              continue; //skip to the next file if the current file is invalid
           }

         FileClose(file_handle); //close the current file

         //--- Append files header and content type as detected to the request

         string file_part = "--" + boundary + "\r\n";
         file_part += StringFormat("Content-Disposition: form-data; name=\"file\"; filename=\"%s\"\r\n", filename);
         file_part += StringFormat("Content-Type: %s\r\n\r\n", GuessContentType(filename));

         char file_header[];
         StringToCharArray(file_part, file_header, 0, StringLen(file_part), CP_UTF8); //UTF-8 Encoding is a must

         final_body.AddArray(file_header); //Add the file header
         final_body.AddArray(file_data); //Add the file in binary format, the actual file

         //--- append the new line — critical for HTTP form parsing.

         final_body.Add('\r');
         final_body.Add('\n');
      }
```

To allow different types of files (videos, images, micosoft documents, etc) to be sent to the server using this one function.

The function _GuessContentType_  detects the type of file given based on the file's extension and returns the right Content-type to be added to the HTTP multipart-form header.

```
string CSession::GuessContentType(string filename)
{
   StringToLower(filename); // Normalize for case-insensitivity

   if(StringFind(filename, ".txt")   >= 0) return "text/plain";
   if(StringFind(filename, ".json")  >= 0) return "application/json";
   if(StringFind(filename, ".xml")   >= 0) return "application/xml";
   //... other files

   //--- Images
   if(StringFind(filename, ".png")   >= 0) return "image/png";
   if(StringFind(filename, ".jpg")   >= 0 || StringFind(filename, ".jpeg") >= 0) return "image/jpeg";
   if(StringFind(filename, ".gif")   >= 0) return "image/gif";
   //...etc

   //--- Audio
   if(StringFind(filename, ".mp3")   >= 0) return "audio/mpeg";
   if(StringFind(filename, ".wav")   >= 0) return "audio/wav";
   if(StringFind(filename, ".ogg")   >= 0) return "audio/ogg";

   //--- Video
   if(StringFind(filename, ".mp4")   >= 0) return "video/mp4";
   if(StringFind(filename, ".avi")   >= 0) return "video/x-msvideo";
   if(StringFind(filename, ".mov")   >= 0) return "video/quicktime";
   if(StringFind(filename, ".webm")  >= 0) return "video/webm";
   if(StringFind(filename, ".mkv")   >= 0) return "video/x-matroska";

   //--- Applications
   if(StringFind(filename, ".pdf")   >= 0) return "application/pdf";
   if(StringFind(filename, ".zip")   >= 0) return "application/zip";
   //... etc

   //--- Microsoft Office
   if(StringFind(filename, ".doc")   >= 0) return "application/msword";
   if(StringFind(filename, ".docx")  >= 0) return "application/vnd.openxmlformats-officedocument.wordprocessingml.document";
   if(StringFind(filename, ".xls")   >= 0) return "application/vnd.ms-excel";
   //...etc

   return "application/octet-stream"; // Default fallback
}
```

**Example usage.**

Suppose we have an image — a screenshot taken from the MetaTrader 5 chart, that we want to send to a server.

_To work with these files easily, you have to ensure they are under the [MQL5 DataPath.](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/cterminalinfo/cterminalinfodatapath)_

![](https://c.mql5.com/2/156/940832026802.gif)

Using [tempfiles.org](https://www.mql5.com/go?link=https://tmpfiles.org/ "https://tmpfiles.org/") server as our API endpoint.

Again, for this to work, make sure you add the URL [tempfiles.org](https://www.mql5.com/go?link=https://tmpfiles.org/ "https://tmpfiles.org/") to the list of allowed URLs in MetaTrader 5;

![](https://c.mql5.com/2/156/5987986839830.png)

```
#include <requests.mqh>
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
//---

     string files[] = {"chart.jpg"};

     CResponse response = CSession::request("POST","https://tmpfiles.org/api/v1/upload","",files);

      Print("Status Code: ", response.status_code);
      Print("Reason: ", response.reason);
      Print("URL: ", response.url);
      Print("OK: ", (string)response.ok);
      Print("Elapsed Time (ms): ", response.elapsed);

      Print("Headers:\n", response.headers);
      Print("Cookies: ", response.cookies);
      Print("Response: ",response.text);
      Print("JSON:\n", response.json["url"].ToStr());
  }
```

Outputs.

```
CF      0       17:06:25.063    Requests test (XAUUSD,D1)       Status Code: 200
RL      0       17:06:25.063    Requests test (XAUUSD,D1)       Reason: OK
QM      0       17:06:25.063    Requests test (XAUUSD,D1)       URL: https://tmpfiles.org/api/v1/upload
FN      0       17:06:25.063    Requests test (XAUUSD,D1)       OK: true
OH      0       17:06:25.063    Requests test (XAUUSD,D1)       Elapsed Time (ms): 1594
FQ      0       17:06:25.063    Requests test (XAUUSD,D1)       Headers:
RN      0       17:06:25.063    Requests test (XAUUSD,D1)       Server: nginx/1.22.1
QO      0       17:06:25.063    Requests test (XAUUSD,D1)       Content-Type: application/json
KJ      0       17:06:25.063    Requests test (XAUUSD,D1)       Transfer-Encoding: chunked
CS      0       17:06:25.063    Requests test (XAUUSD,D1)       Connection: keep-alive
KE      0       17:06:25.063    Requests test (XAUUSD,D1)       Cache-Control: no-cache, private
RM      0       17:06:25.063    Requests test (XAUUSD,D1)       Date: Thu, 10 Jul 2025 14:06:25 GMT
DF      0       17:06:25.063    Requests test (XAUUSD,D1)       X-RateLimit-Limit: 60
GN      0       17:06:25.063    Requests test (XAUUSD,D1)       X-RateLimit-Remaining: 59
CN      0       17:06:25.063    Requests test (XAUUSD,D1)       Access-Control-Allow-Origin: *
NF      0       17:06:25.063    Requests test (XAUUSD,D1)
ED      0       17:06:25.063    Requests test (XAUUSD,D1)       Cookies:
RM      0       17:06:25.063    Requests test (XAUUSD,D1)       Response: {"status":"success","data":{"url":"http://tmpfiles.org/5459540/chart.png"}}
LS      0       17:06:25.063    Requests test (XAUUSD,D1)       JSON:
HJ      0       17:06:25.063    Requests test (XAUUSD,D1)
```

Upon a successful POST request, [tempfiles.org](https://www.mql5.com/go?link=https://tmpfiles.org/ "https://tmpfiles.org/") returns a JSON response containing a URL endpoint of where the file is hosted. We can go to this link and observe the image file in the web browser.

![](https://c.mql5.com/2/156/257685366700.gif)

### Receiving and Downloading Files from the Web

Again, the internet is meant for sharing various information and files, being able to receive different files in MQL5 is handy as it helps in receiving data in CSV, and Excel formats; receiving trained machine learning models and their parameters in different binary formats, and much more.

The implemented _get_ function is already capable of doing this when given the right API endpoint.

For example; Let's get an image from [httpbin.org](https://www.mql5.com/go?link=http://httpbin.org/ "http://httpbin.org/") and save it in the MQL5 datapath.

```
#include <requests.mqh>
CSession requests;
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
      //--- Get the image file from the web
      CResponse response = requests.get("https://httpbin.org/image/jpeg");
 }
```

When this function is executed successfully, it returns the image file/data encrypted (in binary format).

This file is located in the variable CResponse::content.

```
void OnStart()
  {
      //--- Get the image file from the web
      CResponse response = requests.get("https://httpbin.org/image/jpeg");

      //--- Saving an image received in binary format stored in response.content

      int handle = FileOpen("image.jpg", FILE_WRITE|FILE_BIN|FILE_SHARE_WRITE); //Open a .jpg file for writting an image to it
      if (handle == INVALID_HANDLE) //Check the handle
         {
            printf("Failed to open an Image. Error=%s",ErrorDescription(GetLastError()));
            return;
         }

      if (FileWriteArray(handle, response.content)==0) //write all binary data to a image.jpg file
         {
            printf("Failed to write an Image. Error=%s",ErrorDescription(GetLastError()));
            return;
         }

      FileClose(handle);
  }
```

Outputs.

An image containing a fox _(or whatever animal that is)_ is stored under the _MQL5/Files folder._

![](https://c.mql5.com/2/156/5902296982157.png)

Session and Cookies Handling

You may have noticed that all the functions in the _CSession_ class are static, making this class a "static one".

```
class CSession
  {
protected:
   //.... other lines of code

public:

                            CSession(const string headers, const string cookies=""); // Provides headers cookies persistance
                           ~CSession(void);

   static void SetCookie(const string cookie)
      {
         if (StringLen(m_cookies) > 0)
            m_cookies += "; ";
         m_cookies += cookie;
      }

   static void ClearCookies()
      {
         m_cookies = "";
      }

   static void              SetBasicAuth(const string username, const string password);

   //---

   static CResponse         request(const string method, const string url, const string data, const string &files[], const string headers = "", const int timeout = 5000, const bool is_json=true);

   // High-level request helpers
   static CResponse         get(const string url, const string headers = "", const int timeout = 5000)
     {
       string files[];
       return request("GET", url, "", files, headers, timeout, false);
     }

     //... other functions
 }
```

This is aimed to give an option for developers to either use the requests library partially or as a whole to mimick how the requests module in Python operates.

**(a): Using the entire class object**

```
#include <requests.mqh>
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
//---

     string headers = "Content-Type: application/json;";
     string cookies = "sessionid=abc123";

     CSession session(headers, cookies); //New session, class constructor calle with
     CResponse response = session.get("https://httpbin.org/cookies"); //receive cookies from the server

      Print("HTTP response");
      Print("--> Status Code: ", response.status_code);
      Print("--> Reason: ", response.reason);
      Print("--> URL: ", response.url);
      Print("--> OK: ", (string)response.ok);
      Print("--> Elapsed Time (ms): ", response.elapsed);

      Print("--> Headers:\n", response.headers);
      Print("--> Cookies: ", response.cookies);
      Print("--> Response text: ",response.text);
      Print("--> JSON:\n", response.json.ToStr());
}
```

Outputs.

```
DS      0       11:38:36.272    Requests test (XAUUSD,D1)       HTTP response
RI      0       11:38:36.272    Requests test (XAUUSD,D1)       --> Status Code: 200
KR      0       11:38:36.272    Requests test (XAUUSD,D1)       --> Reason: OK
NG      0       11:38:36.272    Requests test (XAUUSD,D1)       --> URL: https://httpbin.org/cookies
IF      0       11:38:36.272    Requests test (XAUUSD,D1)       --> OK: true
QQ      0       11:38:36.272    Requests test (XAUUSD,D1)       --> Elapsed Time (ms): 2141
IH      0       11:38:36.272    Requests test (XAUUSD,D1)       --> Headers:
FQ      0       11:38:36.272    Requests test (XAUUSD,D1)       Date: Sat, 12 Jul 2025 08:38:36 GMT
RE      0       11:38:36.272    Requests test (XAUUSD,D1)       Content-Type: application/json
FO      0       11:38:36.272    Requests test (XAUUSD,D1)       Content-Length: 49
DE      0       11:38:36.272    Requests test (XAUUSD,D1)       Connection: keep-alive
CR      0       11:38:36.272    Requests test (XAUUSD,D1)       Server: gunicorn/19.9.0
PK      0       11:38:36.272    Requests test (XAUUSD,D1)       Access-Control-Allow-Origin: *
HM      0       11:38:36.272    Requests test (XAUUSD,D1)       Access-Control-Allow-Credentials: true
QN      0       11:38:36.272    Requests test (XAUUSD,D1)
DL      0       11:38:36.272    Requests test (XAUUSD,D1)       --> Cookies:
LG      0       11:38:36.272    Requests test (XAUUSD,D1)       --> Response text: {
EP      0       11:38:36.272    Requests test (XAUUSD,D1)         "cookies": {
HJ      0       11:38:36.272    Requests test (XAUUSD,D1)           "sessionid": "abc123"
RN      0       11:38:36.272    Requests test (XAUUSD,D1)         }
DE      0       11:38:36.272    Requests test (XAUUSD,D1)       }
EM      0       11:38:36.272    Requests test (XAUUSD,D1)
MD      0       11:38:36.272    Requests test (XAUUSD,D1)       --> JSON:
GH      0       11:38:36.272    Requests test (XAUUSD,D1)
```

Using the entire class by calling the class constructor and passing the header and cookies (optional), enables you to work with global header values and use the same cookies across all the HTTP request that will be made using the same class instance, _this is what we call a HTTP session._

**(b): Using functions from the class separately**

For making simple HTTP requests without being in an HTTP session, i.e., for managing HTTP new headers and cookies every time.

Below is how you use the functions from the _CSession class_ directly without instantiating the class object.

```
#include <requests.mqh>
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
//---

     CResponse response = CSession::get("https://httpbin.org/get"); //The get request

      Print("HTTP response");
      Print("--> Status Code: ", response.status_code);
      Print("--> Reason: ", response.reason);
      Print("--> URL: ", response.url);
      Print("--> OK: ", (string)response.ok);
      Print("--> Elapsed Time (ms): ", response.elapsed);

      Print("--> Headers:\n", response.headers);
      Print("--> Cookies: ", response.cookies);
      Print("--> Response text: ",response.text);
      Print("--> JSON:\n", response.json.ToStr());
 }
```

### Basic Authentication

All the functions in the _requests_ module offered in Python have an option to send details for a basic (simple) authentication to the server.

```
import requests

response = requests.get("https://httpbin.org/headers", auth=("user", "pass"))
print(response.text)
```

Below is a similar functionality in our MQL5 class.

```
void CSession::SetBasicAuth(const string username, const string password)
  {
   string credentials = username + ":" + password;
   string encoded = Base64Encode(credentials); //Encode the credentials

   m_headers = UpdateHeader(m_headers, "Authorization", "Basic " + encoded); //Update HTTP headers with the authentication information
  }
```

Unlike in the Python module, where developers can send these authentication parameters directly in a function, our MQL5 class does it slightly differently by allowing users to send these basic authentication parameters in a separate function.

The SetBasicAuth function updates the headers in the class by adding the authorization credentials; these values will be available to all the functions called afterward using the same class instance.

```
#include <requests.mqh>
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
//---

     CSession::SetBasicAuth("user", "pass");   //Sets authentication parameters to the HTTP header
     CResponse response = CSession::get("https://httpbin.org/headers"); //Get the headers from the server

      Print("HTTP response");
      Print("--> Headers:\n", response.headers);
      Print("--> Response text: ",response.text);
 }
```

Outputs.

```
IE      0       14:23:23.885    Requests test (XAUUSD,D1)       HTTP response
FQ      0       14:23:23.885    Requests test (XAUUSD,D1)       --> Headers:
KI      0       14:23:23.885    Requests test (XAUUSD,D1)       Date: Sat, 12 Jul 2025 11:23:23 GMT
MM      0       14:23:23.885    Requests test (XAUUSD,D1)       Content-Type: application/json
HK      0       14:23:23.885    Requests test (XAUUSD,D1)       Content-Length: 385
IR      0       14:23:23.885    Requests test (XAUUSD,D1)       Connection: keep-alive
JJ      0       14:23:23.885    Requests test (XAUUSD,D1)       Server: gunicorn/19.9.0
IS      0       14:23:23.885    Requests test (XAUUSD,D1)       Access-Control-Allow-Origin: *
QE      0       14:23:23.885    Requests test (XAUUSD,D1)       Access-Control-Allow-Credentials: true
HF      0       14:23:23.885    Requests test (XAUUSD,D1)
KG      0       14:23:23.885    Requests test (XAUUSD,D1)       --> Response text: {
KP      0       14:23:23.885    Requests test (XAUUSD,D1)         "headers": {
GH      0       14:23:23.885    Requests test (XAUUSD,D1)           "Accept": "*/*",
CM      0       14:23:23.885    Requests test (XAUUSD,D1)           "Accept-Encoding": "gzip, deflate",
DQ      0       14:23:23.885    Requests test (XAUUSD,D1)           "Accept-Language": "en;q=0.5",
NE      0       14:23:23.885    Requests test (XAUUSD,D1)           "Authorization": "Basic dXNlcjpwYXNz",
NS      0       14:23:23.885    Requests test (XAUUSD,D1)           "Cookie": "session=abc123;max-age=60;",
KL      0       14:23:23.885    Requests test (XAUUSD,D1)           "Host": "httpbin.org",
HK      0       14:23:23.885    Requests test (XAUUSD,D1)           "User-Agent": "MetaTrader 5 Terminal/5.5120 (Windows NT 10.0.19045; x64)",
NI      0       14:23:23.885    Requests test (XAUUSD,D1)           "X-Amzn-Trace-Id": "Root=1-687245ab-2065056c28f0024b71a6446f"
GH      0       14:23:23.885    Requests test (XAUUSD,D1)         }
IF      0       14:23:23.885    Requests test (XAUUSD,D1)       }
```

### Dealing with URL Parameters

Another cool thing that the requests module in Python does is that it manages the URL received and its parameters before sending the final web request.

```
import requests

response = requests.get("https://httpbin.org/get", params={"param1": "value1"})
print(response.url)
```

Outputs.

```
https://httpbin.org/get?param1=value1
```

We take a slightly different approach in our MQL5 class. Instead of passing URL parameters to all the functions, which complicates them, we have a separate utility function that helps in creating the final URL with parameters given the original one and its associated parameters.

```
#include <requests.mqh>
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
//---
      string keys[]   = {"user", "id", "lang"};
      string values[] = {"omega", "123", "mql5 test"};

      string parent_url = "https://httpbin.org/get";
      string final_url =  CSession::BuildUrlWithParams(parent_url, keys, values); //builds a URL with the given parameters

      Print("final url: ",final_url);
 }
```

Outputs.

```
2025.07.12 15:41:57.924 Requests test (XAUUSD,D1)       final url: https://httpbin.org/get?user=omega&id=123&lang=mql5+test
```

After crafting a URL with its associated parameters, you can then use it in making HTTP web requests.

```
      CResponse response = CSession::get(final_url); //Get the headers from the server

      Print("HTTP response");
      Print("--> Headers:\n", response.headers);
      Print("--> Response text: ",response.text);
```

Outputs.

```
JK      0       15:42:00.398    Requests test (XAUUSD,D1)       --> Headers:
QP      0       15:42:00.398    Requests test (XAUUSD,D1)       Date: Sat, 12 Jul 2025 12:41:59 GMT
QK      0       15:42:00.398    Requests test (XAUUSD,D1)       Content-Type: application/json
PM      0       15:42:00.398    Requests test (XAUUSD,D1)       Content-Length: 525
ED      0       15:42:00.398    Requests test (XAUUSD,D1)       Connection: keep-alive
FP      0       15:42:00.398    Requests test (XAUUSD,D1)       Server: gunicorn/19.9.0
MH      0       15:42:00.398    Requests test (XAUUSD,D1)       Access-Control-Allow-Origin: *
EK      0       15:42:00.398    Requests test (XAUUSD,D1)       Access-Control-Allow-Credentials: true
LM      0       15:42:00.398    Requests test (XAUUSD,D1)
OI      0       15:42:00.398    Requests test (XAUUSD,D1)       --> Response text: {
RQ      0       15:42:00.398    Requests test (XAUUSD,D1)         "args": {
CR      0       15:42:00.398    Requests test (XAUUSD,D1)           "id": "123",
KF      0       15:42:00.398    Requests test (XAUUSD,D1)           "lang": "mql5 test",
HI      0       15:42:00.398    Requests test (XAUUSD,D1)           "user": "omega"
KP      0       15:42:00.398    Requests test (XAUUSD,D1)         },
MP      0       15:42:00.398    Requests test (XAUUSD,D1)         "headers": {
IH      0       15:42:00.398    Requests test (XAUUSD,D1)           "Accept": "*/*",
QL      0       15:42:00.398    Requests test (XAUUSD,D1)           "Accept-Encoding": "gzip, deflate",
JQ      0       15:42:00.398    Requests test (XAUUSD,D1)           "Accept-Language": "en;q=0.5",
NF      0       15:42:00.398    Requests test (XAUUSD,D1)           "Cookie": "session=abc123;max-age=60;",
KQ      0       15:42:00.398    Requests test (XAUUSD,D1)           "Host": "httpbin.org",
HP      0       15:42:00.398    Requests test (XAUUSD,D1)           "User-Agent": "MetaTrader 5 Terminal/5.5120 (Windows NT 10.0.19045; x64)",
ND      0       15:42:00.398    Requests test (XAUUSD,D1)           "X-Amzn-Trace-Id": "Root=1-68725817-67dc04cf43ac75b012094481"
KE      0       15:42:00.398    Requests test (XAUUSD,D1)         },
CQ      0       15:42:00.398    Requests test (XAUUSD,D1)         "origin": "197.250.227.235",
MD      0       15:42:00.398    Requests test (XAUUSD,D1)         "url": "https://httpbin.org/get?user=omega&id=123&lang=mql5+test"
IK      0       15:42:00.398    Requests test (XAUUSD,D1)       }
LN      0       15:42:00.398    Requests test (XAUUSD,D1)
```

_Awesome!_ The server has even responded with the args key in our response JSON text, which indicates the process of building a URL with its associated parameters was a success.

### The Bottom Line

With all the free, open-source knowledge and information available to the public. Coding should not be difficult in today's world.

After learning from how the _requests_ module operates in Python, I was able to implement a similar module in MQL5 to aid us in making HTTP requests to external servers from MetaTrader 5.

However, while the syntax and function calls might look similar to the ones offered in the requests module offered in Python, this MQL5 module is far from complete; we need to test it vigorously and keep improving it together that's why I created a Repository on Forge MQL5 linked -> [https://forge.mql5.io/omegajoctan/Requests](https://www.mql5.com/go?link=https://forge.mql5.io/omegajoctan/Requests "https://forge.mql5.io/omegajoctan/Requests") .

_So, don't hesitate to update the code from there and let us know your thoughts in the discussion section._

Peace out.

**Attachments Table**

| Filename | Descrioption & Usage |
| --- | --- |
| Include\\errordescription.mqh | Contains descriptions of all error codes produced by MQL5 and MetaTrader 5 |
| [Include\\Jason.mqh](https://www.mql5.com/en/code/13663) | The library for serializing and deserializing of strings in a JSON-like format. |
| Include\\requests.mqh | The main module resembling the requests module from Python. |
| Scripts\\Requests test.mq5 | The main script for testing all functions and methods described in this post. |

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/18728.zip "Download all attachments in the single ZIP archive")

[Attachments.zip](https://www.mql5.com/en/articles/download/18728/attachments.zip "Download Attachments.zip")(17.51 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Python-MetaTrader 5 Strategy Tester (Part 03): MT5-Like Trading Operations — Handling and Managing](https://www.mql5.com/en/articles/20782)
- [Python-MetaTrader 5 Strategy Tester (Part 02): Dealing with Bars, Ticks, and Overloading Built-in Functions in a Simulator](https://www.mql5.com/en/articles/20455)
- [Implementing Practical Modules from Other Languages in MQL5 (Part 06): Python-Like File IO operations in MQL5](https://www.mql5.com/en/articles/20695)
- [Data Science and ML (Part 47): Forecasting the Market Using the DeepAR model in Python](https://www.mql5.com/en/articles/20571)
- [Implementing Practical Modules from Other Languages in MQL5 (Part 05): The Logging module from Python, Log Like a Pro](https://www.mql5.com/en/articles/20458)
- [Implementing Practical Modules from Other Languages in MQL5 (Part 04): time, date, and datetime modules from Python](https://www.mql5.com/en/articles/19035)
- [Python-MetaTrader 5 Strategy Tester (Part 01): Trade Simulator](https://www.mql5.com/en/articles/18971)

**[Go to discussion](https://www.mql5.com/en/forum/491249)**

![MQL5 Wizard Techniques you should know (Part 75): Using Awesome Oscillator and the Envelopes](https://c.mql5.com/2/157/18842-mql5-wizard-techniques-you-logo__1.png)[MQL5 Wizard Techniques you should know (Part 75): Using Awesome Oscillator and the Envelopes](https://www.mql5.com/en/articles/18842)

The Awesome Oscillator by Bill Williams and the Envelopes Channel are a pairing that could be used complimentarily within an MQL5 Expert Advisor. We use the Awesome Oscillator for its ability to spot trends, while the envelopes channel is incorporated to define our support/resistance levels. In exploring this indicator pairing, we use the MQL5 wizard to build and test any potential these two may possess.

![MQL5 Trading Tools (Part 4): Improving the Multi-Timeframe Scanner Dashboard with Dynamic Positioning and Toggle Features](https://c.mql5.com/2/157/18786-mql5-trading-tools-part-4-improving-logo.png)[MQL5 Trading Tools (Part 4): Improving the Multi-Timeframe Scanner Dashboard with Dynamic Positioning and Toggle Features](https://www.mql5.com/en/articles/18786)

In this article, we upgrade the MQL5 Multi-Timeframe Scanner Dashboard with movable and toggle features. We enable dragging the dashboard and a minimize/maximize option for better screen use. We implement and test these enhancements for improved trading flexibility.

![Self Optimizing Expert Advisors in MQL5 (Part 9): Double Moving Average Crossover](https://c.mql5.com/2/157/18793-self-optimizing-expert-advisors-logo.png)[Self Optimizing Expert Advisors in MQL5 (Part 9): Double Moving Average Crossover](https://www.mql5.com/en/articles/18793)

This article outlines the design of a double moving average crossover strategy that uses signals from a higher timeframe (D1) to guide entries on a lower timeframe (M15), with stop-loss levels calculated from an intermediate risk timeframe (H4). It introduces system constants, custom enumerations, and logic for trend-following and mean-reverting modes, while emphasizing modularity and future optimization using a genetic algorithm. The approach allows for flexible entry and exit conditions, aiming to reduce signal lag and improve trade timing by aligning lower-timeframe entries with higher-timeframe trends.

![Cycles and trading](https://c.mql5.com/2/103/Cycles_and_Trading___LOGO.png)[Cycles and trading](https://www.mql5.com/en/articles/16494)

This article is about using cycles in trading. We will consider building a trading strategy based on cyclical models.

[![](https://www.mql5.com/ff/si/0nfwvn6yhmgzf97n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F117%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dorder.expert%26utm_content%3Dorder.freelance%26utm_campaign%3D0622.MQL5.com.Internal&a=tunpwtbhegzufrqocbwiszessdutnobs&s=d9e7484e15300021b4066b1df77a94a1352f9e7c326d5113006bb4f6476bafeb&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=lomwxpoeegrqwuprgaqvnjqczdfcqern&ssn=1769092840553898188&ssn_dr=0&ssn_sr=0&fv_date=1769092840&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F18728&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Implementing%20Practical%20Modules%20from%20Other%20Languages%20in%20MQL5%20(Part%2002)%3A%20Building%20the%20REQUESTS%20Library%2C%20Inspired%20by%20Python%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176909284067159450&fz_uniq=5049297285923317992&sv=2552)

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