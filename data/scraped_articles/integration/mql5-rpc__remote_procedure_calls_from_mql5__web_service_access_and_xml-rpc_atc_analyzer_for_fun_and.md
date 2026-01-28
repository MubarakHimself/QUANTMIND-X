---
title: MQL5-RPC. Remote Procedure Calls from MQL5: Web Service Access and XML-RPC ATC Analyzer for Fun and Profit
url: https://www.mql5.com/en/articles/342
categories: Integration
relevance_score: 3
scraped_at: 2026-01-23T21:20:59.807787
---

[![](https://www.mql5.com/ff/si/h2ryn394uwcpxwmxc2.jpg)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Feconomic-calendar%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dopen.calendar%26utm_content%3Deconomic.calendar%26utm_campaign%3Den.0009.desktop.default&a=qdeulxgvibgwytgewnvfatbocjnnninc&s=5c0c60f00ff5f5bedb0fdf65d9d79eb820442eb43ffac2b85aa003224f9dba14&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=xwdkbvyszxjoitojpwzberveigpwbusp&ssn=1769192457130275365&ssn_dr=0&ssn_sr=0&fv_date=1769192457&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F342&back_ref=https%3A%2F%2Fwww.google.com%2F&title=MQL5-RPC.%20Remote%20Procedure%20Calls%20from%20MQL5%3A%20Web%20Service%20Access%20and%20XML-RPC%20ATC%20Analyzer%20for%20Fun%20and%20Profit%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176919245796685715&fz_uniq=5071765080187547145&sv=2552)

MetaTrader 5 / Integration


### Introduction

This article will describe the MQL5-RPC framework I have built during the last weeks. It covers [XML-RPC access basics](https://en.wikipedia.org/wiki/Remote_procedure_call "https://en.wikipedia.org/wiki/Remote_procedure_call"), description of MQL5 implementation and two real world MQL5-RPC usage examples. First one will be a remote procedure call on an external forex website's webservice and the second one will be a client to our own XML-RPC server that is used to parse, analyze and provide results gathered from [Automated Trading Championship 2011](https://championship.mql5.com/2011/en "https://championship.mql5.com/2011/en"). If you are interested on how to implement and analyze different statistics from [ATC 2011](https://championship.mql5.com/2011/en "https://championship.mql5.com/2011/en") in real time, this article is just for you.

### XML-RPC basics

Let's start with XML-RPC basics. XML-RPC stands for [XML Remote Procedure Call](https://en.wikipedia.org/wiki/Remote_procedure_call "https://en.wikipedia.org/wiki/Remote_procedure_call"). This is a network protocol that uses XML to encode and decode parameters passed to call an external method. It uses HTTP protocol as the transport mechanism to exchange data. By external method I mean another computer program or a webservice that exposes remote procedures.

The exposed method can be called by any computer language from any machine connected to the network provided that it also uses XML-RPC protocol stack and has network access to the server. This also means that that XML-RPC can be used to call a method on the same machine written in another programming language. This will be shown in the second part of the article.

### XML-RPC data model

XML-RPC specification uses six basic data types: int, double, boolean, string, datetime, base64 and two compound data types: array and struct. Array can consist of any basic elements and struct provides name-value pairs such as associative arrays or object properties.

| Basic data types in XML-RPC |
| :-: |
| Type | Value | Examples |
| int or i4 | 32-bit integers between - 2,147,483,648 and 2,147,483,647. | <int>11<int><br><i4>12345<i4> |
| double | 64-bit floating-point numbers | <double>30.02354</double><br><double>-1.53525</double> |
| Boolean | true (1) or false (0) | <boolean>1</boolean><br><boolean>0</boolean> |
| string | ASCII text, many implementations support Unicode | <string>Hello</string><br><string>MQL5</string> |
| dateTime.iso8601 | Dates in ISO8601 format: CCYYMMDDTHH:MM:SS | <dateTime.iso8601><br>20111125T02:20:04<br></dateTime.iso8601><br><dateTime.iso8601><br>20101104T17:27:30<br></dateTime.iso8601> |
| base64 | Binary information encoded as defined in RFC 2045 | <base64><br>TDVsbG8sIFdvdwxkIE==<br></base64> |

Table 1. Basic data types in XML-RPC

Array can hold any of the basic types, not necessarily of the same type. Array element must be nested inside value element. It contains one data element and one or more value elements in data element. The example below shows an array of four integer values.

```
<value>
   <array>
      <data>
         <value><int>111</int></value>
         <value><int>222</int></value>
         <value><int>-3456</int></value>
         <value><int>666</int></value>
      </data>
   </array>
</value>
```

The second example shows an array of five string values.

```
<value>
   <array>
      <data>
         <value><string>MQL5</string></value>
         <value><string>is </string></value>
         <value><string>a</string></value>
         <value><string>great</string></value>
         <value><string>language.</string></value>
      </data>
   </array>
</value>
```

I am convinced you will be able to spot similarities in those two examples to build another XML-RPC array.

Structs have a struct element inside value element and member sections within the struct element. Each member consists of its name and the value it holds. It is therefore easy to pass values of an associative array or members of an object by using structs.

Please see the example below.

```
<value>
   <struct>
      <member>
         <name>AccountHolder</name>
         <value><string>John Doe</string></value>
      </member>
      <member>
         <name>Age</name>
         <value><int>77</int></value>
      </member>
      <member>
         <name>Equity</name>
         <value><double>1000000.0</double></value>
      </member>
   </struct>
</value>
```

Having acquainted with XML-RPC data model we go further to request and response structures.This will form a basis for implementing XML-RPC client in [MQL5](https://www.mql5.com/en/docs).

### XML-RPC request structures

XML-RPC request is composed of message header and message payload. Message header specifies [HTTP](https://en.wikipedia.org/wiki/HTTP "https://en.wikipedia.org/wiki/HTTP") sending method (POST), relative path to XML-RPC service, HTTP protocol version, user-agent name, host IP address, content type (text/xml) and content length in bytes.

```
POST /xmlrpc HTTP 1.1
User-Agent: mql5-rpc/1.0
Host: 10.12.10.10
Content-Type: text/xml
Content-Length: 188
```

Payload of the XML-RPC request is XML document. Root element of the XML tree must be called methodCall. A methodCall contains a single methodName element whose content is executed method name. MethodName element contains zero or one params element.

The params element contains one or more value, array or struct elements. All values are encoded accordingly with data type (see table above). Please see an example payload below showing 'multiply' method execution request with two double values to pass to the function.

```
<?xml version="1.0"?>
<methodCall>
   <methodName>multiply</methodName>
      <params>
         <param>
            <value><double>8654.41</double></value>
         </param>
         <param>
            <value><double>7234.00</double></value>
         </param>
      </params>
</methodCall>
```

The header and payload are sent through HTTP to the server that accepts the input. If server is available it checks the method name and parameter list and executes the desired method. After it finished processing it prepares XML-RPC response structure that can be read by the client.

### XML-RPC response structures

Similar to XML-RPC Request, XML-RPC Response consists of a header and a payload. The header is text and payload is XML document. If request was correct header's first line informs that the server was found (200 code) and specifies protocol version. The header must also contain Content-Type text/xml and Content-Length which is length of the payload in bytes.

```
HTTP/1.1 200 OK
Date: Tue, 08 Nov 2011 23:00:01 GMT
Server: Unix
Connection: close
Content-Type: text/xml
Content-Length: 124
```

Not surprisingly payload of the response is XML document, too. The root element of the XML tree must be called methodResponse. The methodResponse element contains one params on success element or one fault element on failure. The params element contains exactly one param element. The param element contains exactly one value element.

The example of a success response is presented below:

```
<?xml version="1.0"?>
<methodResponse>
   <params>
      <param>
         <value><double>62606001.94</double></value>
      </param>
   </params>
</methodResponse>
```

Failure response is prepared if there was a problem in processing the XML-RPC request.

The fault element, like the params element, has only a single output value.

```
<?xml version="1.0"?>
<methodResponse>
   <fault>
      <value><string>No such method!</string></value>
   </fault>
</methodResponse>
```

Since XML-RPC does not standarize error codes, therefore error messages are dependent on implementation.

### Introducing MQL5-RPC

I came across two articles by Alex Sergeev ["Using WinInet.dll for Data Exchange between Terminals via the Internet"](https://www.mql5.com/en/articles/73) and ["Using WinInet in MQL5. Part 2: POST Requests and Files"](https://www.mql5.com/en/articles/276) and realized I could implement an XML-RPC client for MetaTrader 5. After going through specifications I implemented my own from scratch. This is an ongoing project and does not cover the whole specifcation yet (base64 support will be added in the near future), but you can already use it to make a large subset of XML-RPC calls from MetaTrader 5.

### MQL5-RPC data model

The hardest part in implementation for me was to figure out a correct data model for MQL5. I decided that it must be the simpliest as possible to the framework user, therefore I built several classes that encapsulate its functionality. First decision was to make a request param data as a single CObject\* pointer. This pointer keeps an array of pointers to arrays derived from [CObject](https://www.mql5.com/en/docs/standardlibrary/cobject) class.

There are standard classes that hold CObject arrays; [CArrayInt](https://www.mql5.com/en/docs/standardlibrary/datastructures/carrayint), [CArrayDouble](https://www.mql5.com/en/docs/standardlibrary/datastructures/carraydouble), [CArrayString](https://www.mql5.com/en/docs/standardlibrary/datastructures/carraystring), therefore I based on that and implemented CArrayBool, CArrayDatetime to complete basic data types and CArrayMqlRates to add array of structs. Only base64 type is missing for now but it will be supported in the near future. If array contains just one element it is encapsulated in XML as a single value element.

I wrote an example on how to add different arrays to CObject\* array and display the whole array of arrays of different types. It is available below.

```
//+------------------------------------------------------------------+
//|                                                 ArrayObjTest.mq5 |
//|                                      Copyright 2011, Investeo.pl |
//|                                               http://Investeo.pl |
//+------------------------------------------------------------------+
#property copyright "Copyright 2011, Investeo.pl"
#property link      "http://Investeo.pl"
#property version   "1.00"
//---
#include <Arrays\ArrayObj.mqh>
#include <Arrays\ArrayInt.mqh>
#include <Arrays\ArrayDouble.mqh>
#include <Arrays\ArrayString.mqh>
#include <Arrays\ArrayBool.mqh>
#include <Arrays\ArrayMqlRates.mqh>
#include <Arrays\ArrayDatetime.mqh>
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
//---
   CArrayObj* params = new CArrayObj;

   CArrayInt* arrInt = new CArrayInt;

   arrInt.Add(1001);
   arrInt.Add(1002);
   arrInt.Add(1003);
   arrInt.Add(1004);

   CArrayDouble* arrDouble = new CArrayDouble;

   arrDouble.Add(1001.0);
   arrDouble.Add(1002.0);
   arrDouble.Add(1003.0);
   arrDouble.Add(1004.0);

   CArrayString* arrString = new CArrayString;

   arrString.Add("s1001.0");
   arrString.Add("s1002.0");
   arrString.Add("s1003.0");
   arrString.Add("s1004.0");

   CArrayDatetime* arrDatetime = new CArrayDatetime;

   arrDatetime.Add(TimeCurrent());
   arrDatetime.Add(TimeTradeServer()+3600);
   arrDatetime.Add(TimeCurrent()+3600*24);
   arrDatetime.Add(TimeTradeServer()+3600*24*7);

   CArrayBool* arrBool = new CArrayBool;

   arrBool.Add(false);
   arrBool.Add(true);
   arrBool.Add(true);
   arrBool.Add(false);

   CArrayMqlRates* arrRates = new CArrayMqlRates;

   MqlRates rates[];
   ArraySetAsSeries(rates,true);
   int copied=CopyRates(Symbol(),0,0,4,rates);

   arrRates.Add(rates[0]);
   arrRates.Add(rates[1]);
   arrRates.Add(rates[2]);
   arrRates.Add(rates[3]);

   params.Add(arrInt);
   params.Add(arrDouble);
   params.Add(arrString);
   params.Add(arrDatetime);
   params.Add(arrBool);
   params.Add(arrRates);

   Print("params has " + IntegerToString(params.Total()) + " arrays.");

   for (int p=0; p<params.Total(); p++)
   {
      int type = params.At(p).Type();

      switch (type) {
         case TYPE_INT: {
            CArrayInt *arr = params.At(p);
            for (int i=0; i<arr.Total(); i++)
               PrintFormat("%d %d %d", p, i, arr.At(i));
            break; }
         case TYPE_DOUBLE: {
            CArrayDouble *arr = params.At(p);
            for (int i=0; i<arr.Total(); i++)
               PrintFormat("%d %d %f", p, i, arr.At(i));
            break; }
         case TYPE_STRING: {
            CArrayString *arr = params.At(p);
            for (int i=0; i<arr.Total(); i++)
               PrintFormat("%d %d %s", p, i, arr.At(i));
            break; }
         case TYPE_BOOL: {
            CArrayBool *arr = params.At(p);
            for (int i=0; i<arr.Total(); i++)
               if (arr.At(i) == true)
                  PrintFormat("%d %d true", p, i);
               else
                  PrintFormat("%d %d false", p, i);
            break; }
         case TYPE_DATETIME: {
            CArrayDatetime *arr = params.At(p);
            for (int i=0; i<arr.Total(); i++)
               PrintFormat("%d %d %s", p, i, TimeToString(arr.At(i), TIME_DATE|TIME_MINUTES));
            break; }
         case TYPE_MQLRATES: {  //
            CArrayMqlRates *arr = params.At(p);
            for (int i=0; i<arr.Total(); i++)
               PrintFormat("%d %d %f %f %f %f", p, i, arr.At(i).open, arr.At(i).high, arr.At(i).low, arr.At(i).close);
            break; }
      };

   };
   delete params;

  }
//+------------------------------------------------------------------+
```

The result should be clear: there are 6 subarrays: array of integer values, array of double values, array of strings, array of datetimes, array of boolean values and array of MqlRates.

```
ArrayObjTest (EURUSD,H1)        23:01:54        params has 6 arrays.
ArrayObjTest (EURUSD,H1)        23:01:54        0 0 1001
ArrayObjTest (EURUSD,H1)        23:01:54        0 1 1002
ArrayObjTest (EURUSD,H1)        23:01:54        0 2 1003
ArrayObjTest (EURUSD,H1)        23:01:54        0 3 1004
ArrayObjTest (EURUSD,H1)        23:01:54        1 0 1001.000000
ArrayObjTest (EURUSD,H1)        23:01:54        1 1 1002.000000
ArrayObjTest (EURUSD,H1)        23:01:54        1 2 1003.000000
ArrayObjTest (EURUSD,H1)        23:01:54        1 3 1004.000000
ArrayObjTest (EURUSD,H1)        23:01:54        2 0 s1001.0
ArrayObjTest (EURUSD,H1)        23:01:54        2 1 s1002.0
ArrayObjTest (EURUSD,H1)        23:01:54        2 2 s1003.0
ArrayObjTest (EURUSD,H1)        23:01:54        2 3 s1004.0
ArrayObjTest (EURUSD,H1)        23:01:54        3 0 2011.11.11 23:00
ArrayObjTest (EURUSD,H1)        23:01:54        3 1 2011.11.12 00:01
ArrayObjTest (EURUSD,H1)        23:01:54        3 2 2011.11.12 23:00
ArrayObjTest (EURUSD,H1)        23:01:54        3 3 2011.11.18 23:01
ArrayObjTest (EURUSD,H1)        23:01:54        4 0 false
ArrayObjTest (EURUSD,H1)        23:01:54        4 1 true
ArrayObjTest (EURUSD,H1)        23:01:54        4 2 true
ArrayObjTest (EURUSD,H1)        23:01:54        4 3 false
ArrayObjTest (EURUSD,H1)        23:01:54        5 0 1.374980 1.374980 1.374730 1.374730
ArrayObjTest (EURUSD,H1)        23:01:54        5 1 1.375350 1.375580 1.373710 1.375030
ArrayObjTest (EURUSD,H1)        23:01:54        5 2 1.374680 1.375380 1.373660 1.375370
ArrayObjTest (EURUSD,H1)        23:01:54        5 3 1.375270 1.377530 1.374360 1.374690
```

You may be interested on how I implemented arrays of other datatypes. In CArrayBool and CArrayDatetime I simply based on CArrayInt but in CArrayMqlRates it was a little bit different, since structure must be passed as reference and there was no TYPE\_MQLRATES defined.

Below you can find a partial source code of CArrayMqlRates class. Other classes are available as the attachment to the article.

```
//+------------------------------------------------------------------+
//|                                                ArrayMqlRates.mqh |
//|                                      Copyright 2011, Investeo.pl |
//|                                               http://Investeo.pl |
//|                                              Revision 2011.03.03 |
//+------------------------------------------------------------------+
#include "Array.mqh"
//+------------------------------------------------------------------+
//| Class CArrayMqlRates.                                            |
//| Purpose: Class of dynamic array of structs                       |
//|          of MqlRates type.                                       |
//|          Derived from CArray class.                              |
//+------------------------------------------------------------------+
#define TYPE_MQLRATES 7654

class CArrayMqlRates : public CArray
  {
protected:
   MqlRates          m_data[];           // data array
public:
                     CArrayMqlRates();
                    ~CArrayMqlRates();
   //--- method of identifying the object
   virtual int       Type() const        { return(TYPE_MQLRATES); }
   //--- methods for working with files
   virtual bool      Save(int file_handle);
   virtual bool      Load(int file_handle);
   //--- methods of managing dynamic memory
   bool              Reserve(int size);
   bool              Resize(int size);
   bool              Shutdown();
   //--- methods of filling the array
   bool              Add(MqlRates& element);
   bool              AddArray(const MqlRates &src[]);
   bool              AddArray(const CArrayMqlRates *src);
   bool              Insert(MqlRates& element,int pos);
   bool              InsertArray(const MqlRates &src[],int pos);
   bool              InsertArray(const CArrayMqlRates *src,int pos);
   bool              AssignArray(const MqlRates &src[]);
   bool              AssignArray(const CArrayMqlRates *src);
   //--- method of access to the array
   MqlRates          At(int index) const;
   //--- methods of changing
   bool              Update(int index,MqlRates& element);
   bool              Shift(int index,int shift);
   //--- methods of deleting
   bool              Delete(int index);
   bool              DeleteRange(int from,int to);
protected:
   int               MemMove(int dest,int src,int count);
  };
//+------------------------------------------------------------------+
//| Constructor CArrayMqlRates.                                      |
//| INPUT:  no.                                                      |
//| OUTPUT: no.                                                      |
//| REMARK: no.                                                      |
//+------------------------------------------------------------------+
void CArrayMqlRates::CArrayMqlRates()
  {
//--- initialize protected data
   m_data_max=ArraySize(m_data);
  }
//+------------------------------------------------------------------+
//| Destructor CArrayMqlRates.                                       |
//| INPUT:  no.                                                      |
//| OUTPUT: no.                                                      |
//| REMARK: no.                                                      |
//+------------------------------------------------------------------+
void CArrayMqlRates::~CArrayMqlRates()
  {
   if(m_data_max!=0) Shutdown();
  }
...

//+------------------------------------------------------------------+
//| Adding an element to the end of the array.                       |
//| INPUT:  element - variable to be added.                          |
//| OUTPUT: true if successful, false if not.                        |
//| REMARK: no.                                                      |
//+------------------------------------------------------------------+
bool CArrayMqlRates::Add(MqlRates& element)
  {
//--- checking/reserve elements of array
   if(!Reserve(1)) return(false);
//--- adding
   m_data[m_data_total++]=element;
   m_sort_mode=-1;
//---
   return(true);
  }
//+------------------------------------------------------------------+
//| Adding an element to the end of the array from another array.    |
//| INPUT:  src - source array.                                      |
//| OUTPUT: true if successful, false if not.                        |
//| REMARK: no.                                                      |
//+------------------------------------------------------------------+
bool CArrayMqlRates::AddArray(const MqlRates &src[])
  {
   int num=ArraySize(src);
//--- checking/reserving elements of array
   if(!Reserve(num)) return(false);
//--- adding
   for(int i=0;i<num;i++) m_data[m_data_total++]=src[i];
   m_sort_mode=-1;
//---
   return(true);
  }

...
```

All MQL5 data must be converted to XML values before sending them as RPC request therefore I designed a CXMLRPCEncoder helper class that takes a value and encodes it as XML string.

```
class CXMLRPCEncoder
  {
    public:
                    CXMLRPCEncoder(){};
   string            header(string path,int contentLength);
   string            fromInt(int param);
   string            fromDouble(double param);
   string            fromBool(bool param);
   string            fromString(string param);
   string            fromDateTime(datetime param);
   string            fromMqlRates(MqlRates &param);
  };
```

I pasted three of the implemented methods below. All take one parameter (bool, string, datetime) and return string that is XML valid data type for XML-RPC protocol.

```
//+------------------------------------------------------------------+
//| fromBool                                                         |
//+------------------------------------------------------------------+
string CXMLRPCEncoder::fromBool(bool param)
  {
   CString s_bool;
   s_bool.Clear();
   s_bool.Append(VALUE_B);
   s_bool.Append(BOOL_B);
   if(param==true)
      s_bool.Append("1");
   else s_bool.Append("0");
   s_bool.Append(BOOL_E);
   s_bool.Append(VALUE_E);

   return s_bool.Str();
  }
//+------------------------------------------------------------------+
//| fromString                                                       |
//+------------------------------------------------------------------+
string CXMLRPCEncoder::fromString(string param)
  {
   CString s_string;
   s_string.Clear();
   s_string.Append(VALUE_B);
   s_string.Append(STRING_B);
   s_string.Append(param);
   s_string.Append(STRING_E);
   s_string.Append(VALUE_E);

   return s_string.Str();
  }
//+------------------------------------------------------------------+
//| fromDateTime                                                     |
//+------------------------------------------------------------------+
string CXMLRPCEncoder::fromDateTime(datetime param)
  {
   CString s_datetime;
   s_datetime.Clear();
   s_datetime.Append(VALUE_B);
   s_datetime.Append(DATETIME_B);
   CString s_iso8601;
   s_iso8601.Assign(TimeToString(param, TIME_DATE|TIME_MINUTES));
   s_iso8601.Replace(" ", "T");
   s_iso8601.Remove(":");
   s_iso8601.Remove(".");
   s_datetime.Append(s_iso8601.Str());
   s_datetime.Append(DATETIME_E);
   s_datetime.Append(VALUE_E);

   return s_datetime.Str();
  }
```

You may notice that there are certain tags with \_B suffix meaning 'tag begin' and \_E suffix meaning 'tag end'.

I decided to use a header file that keeps names of XML tags and headers since it made implementation much more transparent

```
//+------------------------------------------------------------------+
//|                                                   xmlrpctags.mqh |
//|                                      Copyright 2011, Investeo.pl |
//|                                           http://www.investeo.pl |
//+------------------------------------------------------------------+
#property copyright "Copyright 2011, Investeo.pl"
#property link      "http://www.investeo.pl"

#define HEADER_1a      "POST"
#define HEADER_1b      "HTTP/1.1"
#define HEADER_2       "User-Agent: MQL5RPC/1.1"
#define HEADER_3       "Host: host.com"
#define HEADER_4       "Content-Type: text/xml"
#define HEADER_5       "Content-Length: "
#define HEADER_6       "<?xml version='1.0'?>"
#define METHOD_B       "<methodCall>"
#define METHOD_E       "</methodCall>"
#define METHOD_NAME_B  "<methodName>"
#define METHOD_NAME_E  "</methodName>"
#define RESPONSE_B     "<methodResponse>"
#define RESPONSE_E     "</methodResponse>"
#define PARAMS_B       "<params>"
#define PARAMS_E       "</params>"
#define PARAM_B        "<param>"
#define PARAM_E        "</param>"
#define VALUE_B        "<value>"
#define VALUE_E        "</value>"
#define INT_B          "<int>"
#define INT_E          "</int>"
#define I4_B           "<i4>"
#define I4_E           "</i4>"
#define BOOL_B         "<boolean>"
#define BOOL_E         "</boolean>"
#define DOUBLE_B       "<double>"
#define DOUBLE_E       "</double>"
#define STRING_B       "<string>"
#define STRING_E       "</string>"
#define DATETIME_B     "<dateTime.iso8601>"
#define DATETIME_E     "</dateTime.iso8601>"
#define BASE64_B       "<base64>"
#define BASE64_E       "</base64>"
#define ARRAY_B        "<array>"
#define ARRAY_E        "</array>"
#define DATA_B         "<data>"
#define DATA_E         "</data>"
#define STRUCT_B       "<struct>"
#define STRUCT_E       "</struct>"
#define MEMBER_B       "<member>"
#define MEMBER_E       "</member>"
#define NAME_B         "<name>"
#define NAME_E         "</name>"
//+------------------------------------------------------------------+
```

Having defined the data model of MQL5-RPC we can proceed to building a full XML-RPC request.

### MQL5-RPC request

As previously mentioned, XML-RPC request consists of a request header and XML payload. I designed CXMLRPCQuery class that automatically constructs a query object from MQL5 data arrays. The class uses CXMLRPCEncoder to encapsulate data in XML and adds method name inside methodName tag.

```
class CXMLRPCQuery
  {
private:
   CString           s_query;
   void              addValueElement(bool start,bool array);
public:
                    CXMLRPCQuery() {};
                    CXMLRPCQuery(string method="",CArrayObj *param_array=NULL);
   string            toString();
  };
```

Constructor of the class has two parameters: method name and pointer to [CArrayObj](https://www.mql5.com/en/docs/standardlibrary/datastructures/carrayobj) holding parameters to call the method. All parameters are wrapped in XML as described in the previous section and a query header is added. Whole XML query can be displayed using toString() method.

```
CXMLRPCQuery::CXMLRPCQuery(string method="",CArrayObj *param_array=NULL)
  {
//--- constructs a single XMLRPC Query
   this.s_query.Clear();

   CXMLRPCEncoder encoder;
   this.s_query.Append(HEADER_6);
   this.s_query.Append(METHOD_B);
   this.s_query.Append(METHOD_NAME_B);
   this.s_query.Append(method);
   this.s_query.Append(METHOD_NAME_E);
   this.s_query.Append(PARAMS_B);

   for(int i=0; i<param_array.Total(); i++)
     {
      int j=0;
      this.s_query.Append(PARAM_B);

      int type=param_array.At(i).Type();
      int elements=0;

      switch(type)
        {
         case TYPE_INT:
           {
            CArrayInt *arr=param_array.At(i);
            elements=arr.Total();
            if(elements==1) addValueElement(true,false); else addValueElement(true,true);
            for(j=0; j<elements; j++) this.s_query.Append(encoder.fromInt(arr.At(j)));
            break;
           }
         case TYPE_DOUBLE:
           {
            CArrayDouble *arr=param_array.At(i);
            elements=arr.Total();
            if(elements==1) addValueElement(true,false); else addValueElement(true,true);
            for(j=0; j<elements; j++) this.s_query.Append(encoder.fromDouble(arr.At(j)));
            break;
           }
         case TYPE_STRING:
           {
            CArrayString *arr=param_array.At(i);
            elements=arr.Total();
            if(elements==1) addValueElement(true,false); else addValueElement(true,true);
            for(j=0; j<elements; j++) this.s_query.Append(encoder.fromString(arr.At(j)));
            break;
           }
         case TYPE_BOOL:
           {
            CArrayBool *arr=param_array.At(i);
            elements=arr.Total();
            if(elements==1) addValueElement(true,false); else addValueElement(true,true);
            for(j=0; j<elements; j++) this.s_query.Append(encoder.fromBool(arr.At(j)));
            break;
           }
         case TYPE_DATETIME:
           {
            CArrayDatetime *arr=param_array.At(i);
            elements=arr.Total();
            if(elements==1) addValueElement(true,false); else addValueElement(true,true);
            for(j=0; j<elements; j++) this.s_query.Append(encoder.fromDateTime(arr.At(j)));
            break;
           }
         case TYPE_MQLRATES:
           {
            CArrayMqlRates *arr=param_array.At(i);
            elements=arr.Total();
            if(elements==1) addValueElement(true,false); else addValueElement(true,true);
            for(j=0; j<elements; j++)
              {
               MqlRates tmp=arr.At(j);
               this.s_query.Append(encoder.fromMqlRates(tmp));
              }
            break;
           }
        };

      if(elements==1) addValueElement(false,false); else addValueElement(false,true);

      this.s_query.Append(PARAM_E);
     }

   this.s_query.Append(PARAMS_E);
   this.s_query.Append(METHOD_E);
  }
```

Please see a query test example below. This is not a simpliest one, since I want to show that it is possible to call quite complex methods.

Input parameters are: array of [double](https://www.mql5.com/en/docs/basis/types/double) values, array of [integer](https://www.mql5.com/en/docs/basis/types/integer) values, array of [string](https://www.mql5.com/en/docs/basis/types/stringconst) values, array of [bool](https://www.mql5.com/en/docs/basis/types/integer/boolconst) values, a single [datetime](https://www.mql5.com/en/docs/basis/types/integer/datetime) value and array of [MqlRates](https://www.mql5.com/en/docs/constants/structures/mqlrates) structs.

```
//+------------------------------------------------------------------+
//|                                          MQL5-RPC_query_test.mq5 |
//|                                      Copyright 2011, Investeo.pl |
//|                                           http://www.investeo.pl |
//+------------------------------------------------------------------+
#property copyright "Copyright 2011, Investeo.pl"
#property link      "http://www.investeo.pl"
#property version   "1.00"

#include <MQL5-RPC.mqh>
#include <Arrays\ArrayObj.mqh>
#include <Arrays\ArrayInt.mqh>
#include <Arrays\ArrayDouble.mqh>
#include <Arrays\ArrayString.mqh>
#include <Arrays\ArrayBool.mqh>
#include <Arrays\ArrayDatetime.mqh>
#include <Arrays\ArrayMqlRates.mqh>
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
//--- query test
   CArrayObj* params = new CArrayObj;

   CArrayDouble*   param1 = new CArrayDouble;
   CArrayInt*      param2 = new CArrayInt;
   CArrayString*   param3 = new CArrayString;
   CArrayBool*     param4 = new CArrayBool;
   CArrayDatetime* param5 = new CArrayDatetime;
   CArrayMqlRates* param6 = new CArrayMqlRates;

   for (int i=0; i<4; i++)
      param1.Add(20000.0 + i*100.0);

   params.Add(param1);

   for (int i=0; i<4; i++)
      param2.Add(i);

   params.Add(param2);

   param3.Add("first_string");
   param3.Add("second_string");
   param3.Add("third_string");

   params.Add(param3);

   param4.Add(false);
   param4.Add(true);
   param4.Add(false);

   params.Add(param4);

   param5.Add(TimeCurrent());
   params.Add(param5);

   const int nRates = 3;
   MqlRates rates[3];
   ArraySetAsSeries(rates,true);
   int copied=CopyRates(Symbol(),0,0,nRates,rates);
   if (copied==nRates) {
      param6.AddArray(rates);
      params.Add(param6);
   }

   CXMLRPCQuery query("sampleMethodname", params);

   Print(query.toString());

   delete params;
  }
//+------------------------------------------------------------------+
```

The resulting query is as follows. Normally this is a one-line string value but I present it as a tabbed text to make easy to distinguish XML-RPC sections and single values.

```
<?xml version="1.0" ?>
<methodCall>
      <methodName>sampleMethodname</methodName>
      <params>
            <param>
                  <value>
                        <array>
                              <data>
                                    <value>
                                          <double>20000.0000000000000000</double>
                                    </value>
                                    <value>
                                          <double>20100.0000000000000000</double>
                                    </value>
                                    <value>
                                          <double>20200.0000000000000000</double>
                                    </value>
                                    <value>
                                          <double>20300.0000000000000000</double>
                                    </value>
                              </data>
                        </array>
                  </value>
            </param>
            <param>
                  <value>
                        <array>
                              <data>
                                    <value>
                                          <int>0</int>
                                    </value>
                                    <value>
                                          <int>1</int>
                                    </value>
                                    <value>
                                          <int>2</int>
                                    </value>
                                    <value>
                                          <int>3</int>
                                    </value>
                              </data>
                        </array>
                  </value>
            </param>
            <param>
                  <value>
                        <array>
                              <data>
                                    <value>
                                          <string>first_string</string>
                                    </value>
                                    <value>
                                          <string>second_string</string>
                                    </value>
                                    <value>
                                          <string>third_string</string>
                                    </value>
                              </data>
                        </array>
                  </value>
            </param>
            <param>
                  <value>
                        <array>
                              <data>
                                    <value>
                                          <boolean>0</boolean>
                                    </value>
                                    <value>
                                          <boolean>1</boolean>
                                    </value>
                                    <value>
                                          <boolean>0</boolean>
                                    </value>
                              </data>
                        </array>
                  </value>
            </param>
            <param>
                  <value>
                        <dateTime.iso8601>20111111T2042</dateTime.iso8601>
                  </value>
            </param>
            <param>
                  <value>
                        <array>
                              <data>
                                    <value>
                                          <struct>
                                                <member>
                                                      <name>open</name>
                                                      <value>
                                                            <double>1.02902000</double>
                                                      </value>
                                                </member>
                                                <member>
                                                      <name>high</name>
                                                      <value>
                                                            <double>1.03032000</double>
                                                      </value>
                                                </member>
                                                <member>
                                                      <name>low</name>
                                                      <value>
                                                            <double>1.02842000</double>
                                                      </value>
                                                </member>
                                                <member>
                                                      <name>close</name>
                                                      <value>
                                                            <double>1.02867000</double>
                                                      </value>
                                                </member>
                                                <member>
                                                      <name>time</name>
                                                      <value>
                                                            <dateTime.iso8601>
                                                                  <value>
                                                                        <dateTime.iso8601>20111111T1800</dateTime.iso8601>
                                                                  </value>
                                                            </dateTime.iso8601>
                                                      </value>
                                                </member>
                                                <member>
                                                      <name>tick_volume</name>
                                                      <value>
                                                            <double>4154.00000000</double>
                                                      </value>
                                                </member>
                                                <member>
                                                      <name>real_volume</name>
                                                      <value>
                                                            <double>0.00000000</double>
                                                      </value>
                                                </member>
                                                <member>
                                                      <name>spread</name>
                                                      <value>
                                                            <double>30.00000000</double>
                                                      </value>
                                                </member>
                                          </struct>
                                    </value>
                                    <value>
                                          <struct>
                                                <member>
                                                      <name>open</name>
                                                      <value>
                                                            <double>1.02865000</double>
                                                      </value>
                                                </member>
                                                <member>
                                                      <name>high</name>
                                                      <value>
                                                            <double>1.02936000</double>
                                                      </value>
                                                </member>
                                                <member>
                                                      <name>low</name>
                                                      <value>
                                                            <double>1.02719000</double>
                                                      </value>
                                                </member>
                                                <member>
                                                      <name>close</name>
                                                      <value>
                                                            <double>1.02755000</double>
                                                      </value>
                                                </member>
                                                <member>
                                                      <name>time</name>
                                                      <value>
                                                            <dateTime.iso8601>
                                                                  <value>
                                                                        <dateTime.iso8601>20111111T1900</dateTime.iso8601>
                                                                  </value>
                                                            </dateTime.iso8601>
                                                      </value>
                                                </member>
                                                <member>
                                                      <name>tick_volume</name>
                                                      <value>
                                                            <double>3415.00000000</double>
                                                      </value>
                                                </member>
                                                <member>
                                                      <name>real_volume</name>
                                                      <value>
                                                            <double>0.00000000</double>
                                                      </value>
                                                </member>
                                                <member>
                                                      <name>spread</name>
                                                      <value>
                                                            <double>30.00000000</double>
                                                      </value>
                                                </member>
                                          </struct>
                                    </value>
                                    <value>
                                          <struct>
                                                <member>
                                                      <name>open</name>
                                                      <value>
                                                            <double>1.02760000</double>
                                                      </value>
                                                </member>
                                                <member>
                                                      <name>high</name>
                                                      <value>
                                                            <double>1.02901000</double>
                                                      </value>
                                                </member>
                                                <member>
                                                      <name>low</name>
                                                      <value>
                                                            <double>1.02756000</double>
                                                      </value>
                                                </member>
                                                <member>
                                                      <name>close</name>
                                                      <value>
                                                            <double>1.02861000</double>
                                                      </value>
                                                </member>
                                                <member>
                                                      <name>time</name>
                                                      <value>
                                                            <dateTime.iso8601>
                                                                  <value>
                                                                        <dateTime.iso8601>20111111T2000</dateTime.iso8601>
                                                                  </value>
                                                            </dateTime.iso8601>
                                                      </value>
                                                </member>
                                                <member>
                                                      <name>tick_volume</name>
                                                      <value>
                                                            <double>1845.00000000</double>
                                                      </value>
                                                </member>
                                                <member>
                                                      <name>real_volume</name>
                                                      <value>
                                                            <double>0.00000000</double>
                                                      </value>
                                                </member>
                                                <member>
                                                      <name>spread</name>
                                                      <value>
                                                            <double>30.00000000</double>
                                                      </value>
                                                </member>
                                          </struct>
                                    </value>
                              </data>
                        </array>
                  </value>
            </param>
      </params>
</methodCall>
```

This is quite a developed XML tree and the protocol is flexible to call even more complex methods.

### MQL5-RPC response

As with XML-RPC request, XML-RPC response is constructed from headers and XML payload, but this time the processing order must be reversed, that is XML response must be converted into MQL5 data. CXMLRPCResult class was designed to cope with this task.

Result can be received as a string that is passed to class constructor or can be automatically retrieved from CXMLServerProxy class after desired method is executed. In situations when the result contains structs that may not be known beforehand parseXMLResponseRAW() method can look for all <value> tags and returns [CArrayObj](https://www.mql5.com/en/docs/standardlibrary/datastructures/carrayobj) pointer that contains array of all found value elements.

```
class CXMLRPCResult
  {
private:
   CArrayObj        *m_resultsArr;

   CString           m_cstrResponse;
   CArrayString      m_params;

   bool              isValidXMLResponse();
   bool              parseXMLValuesToMQLArray(CArrayString *subArr,CString &val);
   bool              parseXMLValuesToMQLArray(CArrayDouble *subArr,CString &val);
   bool              parseXMLValuesToMQLArray(CArrayInt *subArr,CString &val);
   bool              parseXMLValuesToMQLArray(CArrayBool *subArr,CString &val);
   bool              parseXMLValuesToMQLArray(CArrayDatetime *subArr,CString &val);
   bool              parseXMLValuesToMQLArray(CArrayMqlRates *subArr,CString &val);
   bool              parseXMLResponse();

public:
                     CXMLRPCResult() {};
                    ~CXMLRPCResult();
                     CXMLRPCResult(string resultXml);

   CArrayObj        *getResults();
   bool              parseXMLResponseRAW();
   string            toString();
  };
```

The constructor of the class scans through every header in the response XML and calls private method parseXMLValuesToMQLArray() that does the hard job of converting XML to MQL5 data behind the scenes.

It recognizes if param is an array or single element and fills appropriate arrays added to [CArrayObj](https://www.mql5.com/en/docs/standardlibrary/datastructures/carrayobj) array result.

```
bool CXMLRPCResult::parseXMLResponse()
  {
   CArrayObj *results=new CArrayObj;

   m_params.Clear();

   //--- find params and put them in m_params array
   int tagStartIdx= 0;
   int tagStopIdx = 0;
   while((tagStartIdx!=-1) && (tagStopIdx!=-1))
     {

      tagStartIdx= m_cstrResponse.Find(tagStartIdx,PARAM_B);
      tagStopIdx = m_cstrResponse.Find(tagStopIdx,PARAM_E);

      if((tagStartIdx!=-1) && (tagStopIdx!=-1))
        {
         m_params.Add(m_cstrResponse.Mid(tagStartIdx+StringLen(PARAM_B),tagStopIdx-tagStartIdx-StringLen(PARAM_B)));
         tagStartIdx++; tagStopIdx++;
        };

     };

   for(int i=0; i<m_params.Total(); i++)
     {
      CString val;
      val.Assign(m_params.At(i));

      //--- parse value tag

      val.Assign(val.Mid(StringLen(VALUE_B),val.Len()-StringLen(VALUE_B)-StringLen(VALUE_E)));

      //--- now check first tag and handle it approprietaly

      string param_type=val.Mid(0,val.Find(0,">")+1);

      if(param_type==INT_B || param_type==I4_B)
        {
         val.Assign(m_params.At(i));
         CArrayInt *subArr=new CArrayInt;
         bool isValid=parseXMLValuesToMQLArray(subArr,val);
         if(isValid==true)
            results.Add(subArr);
        }
      else if(param_type==BOOL_B)
        {
         val.Assign(m_params.At(i));
         CArrayBool *subArr=new CArrayBool;
         bool isValid=parseXMLValuesToMQLArray(subArr,val);
         if(isValid==true)
            results.Add(subArr);
        }
      else if(param_type==DOUBLE_B)
        {
         val.Assign(m_params.At(i));
         CArrayDouble *subArr=new CArrayDouble;
         bool isValid=parseXMLValuesToMQLArray(subArr,val);
         if(isValid==true)
            results.Add(subArr);
        }
      else if(param_type==STRING_B)
        {
         val.Assign(m_params.At(i));
         CArrayString *subArr=new CArrayString;
         bool isValid=parseXMLValuesToMQLArray(subArr,val);
         if(isValid==true)
            results.Add(subArr);
        }
      else if(param_type==DATETIME_B)
        {
         val.Assign(m_params.At(i));
         CArrayDatetime *subArr=new CArrayDatetime;
         bool isValid=parseXMLValuesToMQLArray(subArr,val);
         if(isValid==true)
            results.Add(subArr);
        }
      else if(param_type==ARRAY_B)
        {
         val.Assign(val.Mid(StringLen(ARRAY_B)+StringLen(DATA_B),val.Len()-StringLen(ARRAY_B)-StringLen(DATA_E)));
         //--- find first type and define array
         string array_type=val.Mid(StringLen(VALUE_B),val.Find(StringLen(VALUE_B)+1,">")-StringLen(VALUE_B)+1);

         if(array_type==INT_B || array_type==I4_B)
           {
            CArrayInt *subArr=new CArrayInt;
            bool isValid=parseXMLValuesToMQLArray(subArr,val);
            if(isValid==true)
               results.Add(subArr);
           }
         else if(array_type==BOOL_B)
           {
            CArrayBool *subArr=new CArrayBool;
            bool isValid=parseXMLValuesToMQLArray(subArr,val);
            if(isValid==true)
               results.Add(subArr);
           }
         else if(array_type==DOUBLE_B)
           {
            CArrayDouble *subArr=new CArrayDouble;
            bool isValid=parseXMLValuesToMQLArray(subArr,val);
            if(isValid==true)
               results.Add(subArr);
           }
         else if(array_type==STRING_B)
           {
            CArrayString *subArr=new CArrayString;
            bool isValid=parseXMLValuesToMQLArray(subArr,val);
            if(isValid==true)
               results.Add(subArr);

           }
         else if(array_type==DATETIME_B)
           {
            CArrayDatetime *subArr=new CArrayDatetime;
            bool isValid=parseXMLValuesToMQLArray(subArr,val);
            if(isValid==true)
               results.Add(subArr);
           }
        }
     };

   m_resultsArr=results;

   return true;
  }
```

The conversion takes place inside overloaded parseXMLValuesToMQLArray methods. The string value is extracted from XML and converted to basic MQL5 variables.

Three of the methods used for conversion are pasted below for reference.

```
//+------------------------------------------------------------------+
//| parseXMLValuesToMQLArray                                         |
//+------------------------------------------------------------------+
bool CXMLRPCResult::parseXMLValuesToMQLArray(CArrayBool *subArr,CString &val)
  {
   //--- parse XML values and populate MQL array
   int tagStartIdx=0; int tagStopIdx=0;

   while((tagStartIdx!=-1) && (tagStopIdx!=-1))
     {
      tagStartIdx= val.Find(tagStartIdx,VALUE_B);
      tagStopIdx = val.Find(tagStopIdx,VALUE_E);
      if((tagStartIdx!=-1) && (tagStopIdx!=-1))
        {
         CString e;
         e.Assign(val.Mid(tagStartIdx+StringLen(VALUE_B)+StringLen(BOOL_B),
                  tagStopIdx-tagStartIdx-StringLen(VALUE_B)-StringLen(BOOL_B)-StringLen(BOOL_E)));
         if(e.Str()=="0")
            subArr.Add(false);
         if(e.Str()=="1")
            subArr.Add(true);

         tagStartIdx++; tagStopIdx++;
        };
     }
   if(subArr.Total()<1) return false;
   return true;
  }
//+------------------------------------------------------------------+
//| parseXMLValuesToMQLArray                                         |
//+------------------------------------------------------------------+
bool CXMLRPCResult::parseXMLValuesToMQLArray(CArrayInt *subArr,CString &val)
  {
    //--- parse XML values and populate MQL array
   int tagStartIdx=0; int tagStopIdx=0;

   while((tagStartIdx!=-1) && (tagStopIdx!=-1))
     {
      tagStartIdx= val.Find(tagStartIdx,VALUE_B);
      tagStopIdx = val.Find(tagStopIdx,VALUE_E);
      if((tagStartIdx!=-1) && (tagStopIdx!=-1))
        {
         CString e;
         e.Assign(val.Mid(tagStartIdx+StringLen(VALUE_B)+StringLen(INT_B),
                  tagStopIdx-tagStartIdx-StringLen(VALUE_B)-StringLen(INT_B)-StringLen(INT_E)));
         subArr.Add((int)StringToInteger(e.Str()));
         tagStartIdx++; tagStopIdx++;
        };
     }
   if(subArr.Total()<1) return false;
   return true;
  }
//+------------------------------------------------------------------+
//| parseXMLValuesToMQLArray                                         |
//+------------------------------------------------------------------+
bool CXMLRPCResult::parseXMLValuesToMQLArray(CArrayDatetime *subArr,CString &val)
  {
   // parse XML values and populate MQL array
   int tagStartIdx=0; int tagStopIdx=0;

   while((tagStartIdx!=-1) && (tagStopIdx!=-1))
     {
      tagStartIdx= val.Find(tagStartIdx,VALUE_B);
      tagStopIdx = val.Find(tagStopIdx,VALUE_E);
      if((tagStartIdx!=-1) && (tagStopIdx!=-1))
        {
         CString e;
         e.Assign(val.Mid(tagStartIdx+StringLen(VALUE_B)+StringLen(DATETIME_B),
                  tagStopIdx-tagStartIdx-StringLen(VALUE_B)-StringLen(DATETIME_B)-StringLen(DATETIME_E)));
         e.Replace("T"," "); e.Insert(4,"."); e.Insert(7,".");
         subArr.Add(StringToTime(e.Str()));
         tagStartIdx++; tagStopIdx++;
        };
     }
   if(subArr.Total()<1) return false;
   return true;
  }
```

As you can see string values are extracted from XML-RPC tags using methods available in [CString](https://www.mql5.com/en/docs/standardlibrary/stringoperations/cstring) class; [Assign()](https://www.mql5.com/en/docs/standardlibrary/stringoperations/cstring/cstringassign), [Mid()](https://www.mql5.com/en/docs/standardlibrary/stringoperations/cstring/cstringmid), [Find()](https://www.mql5.com/en/docs/standardlibrary/stringoperations/cstring/cstringfind) and are further converted into MQL5 using [StringToTime()](https://www.mql5.com/en/docs/convert/stringtotime), [StringToInteger()](https://www.mql5.com/en/docs/convert/stringtointeger), [StringToDouble()](https://www.mql5.com/en/docs/convert/stringtodouble) or a custom method as in case of bool variables.

After all values are parsed all data available can be displayed using toString() method. I simply use colon to separate values.

```
string CXMLRPCResult::toString(void)
  {
   // returns results array of arrays as a string
   CString r;

   for(int i=0; i<m_resultsArr.Total(); i++)
     {
      int rtype=m_resultsArr.At(i).Type();
      switch(rtype)
        {
         case(TYPE_STRING) :
           {
            CArrayString *subArr=m_resultsArr.At(i);
            for(int j=0; j<subArr.Total(); j++)
              {
               r.Append(subArr.At(j)+":");
              }
            break;
           };
         case(TYPE_DOUBLE) :
           {
            CArrayDouble *subArr=m_resultsArr.At(i);
            for(int j=0; j<subArr.Total(); j++)
              {
               r.Append(DoubleToString(NormalizeDouble(subArr.At(j),8))+":");
              }
            break;
           };
         case(TYPE_INT) :
           {
            CArrayInt *subArr=m_resultsArr.At(i);
            for(int j=0; j<subArr.Total(); j++)
              {
               r.Append(IntegerToString(subArr.At(j))+":");
              }
            break;
           };
         case(TYPE_BOOL) :
           {
            CArrayBool *subArr=m_resultsArr.At(i);
            for(int j=0; j<subArr.Total(); j++)
              {
               if(subArr.At(j)==false) r.Append("false:");
               else r.Append("true:");
              }
            break;
           };
         case(TYPE_DATETIME) :
           {
            CArrayDatetime *subArr=m_resultsArr.At(i);
            for(int j=0; j<subArr.Total(); j++)
              {
               r.Append(TimeToString(subArr.At(j),TIME_DATE|TIME_MINUTES|TIME_SECONDS)+" : ");
              }
            break;
           };
        };
     }

   return r.Str();
  }
```

I implemented a result test for you to see explicitly how XML-RPC result parameters are converted into MQL5.

```
//+------------------------------------------------------------------+
//|                                         MQL5-RPC_result_test.mq5 |
//|                                      Copyright 2011, Investeo.pl |
//|                                           http://www.investeo.pl |
//+------------------------------------------------------------------+
#property copyright "Copyright 2011, Investeo.pl"
#property link      "http://www.investeo.pl"
#property version   "1.00"
//---
#include <MQL5-RPC.mqh>
#include <Arrays\ArrayObj.mqh>
#include <Arrays\ArrayInt.mqh>
#include <Arrays\ArrayDouble.mqh>
#include <Arrays\ArrayString.mqh>
#include <Arrays\ArrayBool.mqh>
#include <Arrays\ArrayDatetime.mqh>
#include <Arrays\ArrayMqlRates.mqh>
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
//---
    string sampleXMLResult = "<?xml version='1.0'?>"
                            "<methodResponse>" +
                            "<params>" +
                            "<param>" +
                            "<value><array><data>" +
                            "<value><string>Xupypr</string></value>" +
                            "<value><string>anuta</string></value>" +
                            "<value><string>plencing</string></value>" +
                            "<value><string>aharata</string></value>" +
                            "<value><string>beast</string></value>" +
                            "<value><string>west100</string></value>" +
                            "<value><string>ias</string></value>" +
                            "<value><string>Tim</string></value>" +
                            "<value><string>gery18</string></value>" +
                            "<value><string>ronnielee</string></value>" +
                            "<value><string>investeo</string></value>" +
                            "<value><string>droslan</string></value>" +
                            "<value><string>Better</string></value>" +
                            "</data></array></value>" +
                            "</param>" +
                            "<param>" +
                            "<value><array><data>" +
                            "<value><double>1.222</double></value>" +
                            "<value><double>0.456</double></value>" +
                            "<value><double>1000000000.10101</double></value>" +
                            "</data></array></value>" +
                            "</param>" +
                            "<param>" +
                            "<value><array><data>" +
                            "<value><boolean>1</boolean></value>" +
                            "<value><boolean>0</boolean></value>" +
                            "<value><boolean>1</boolean></value>" +
                            "</data></array></value>" +
                            "</param>" +
                            "<param>" +
                            "<value><array><data>" +
                            "<value><int>-1</int></value>" +
                            "<value><int>0</int></value>" +
                            "<value><int>1</int></value>" +
                            "</data></array></value>" +
                            "</param>" +
                            "<param>" +
                            "<value><array><data>" +
                            "<value><dateTime.iso8601>20021125T02:20:04</dateTime.iso8601></value>" +
                            "<value><dateTime.iso8601>20111115T00:00:00</dateTime.iso8601></value>" +
                            "<value><dateTime.iso8601>20121221T00:00:00</dateTime.iso8601></value>" +
                            "</data></array></value>" +
                            "</param>" +
                            "<param><value><string>Single string value</string></value></param>" +
                            "<param><value><dateTime.iso8601>20111115T00:00:00</dateTime.iso8601></value></param>" +
                            "<param><value><int>-111</int></value></param>" +
                            "<param><value><boolean>1</boolean></value></param>" +
                            "</params>" +
                            "</methodResponse>";

   CXMLRPCResult* testResult = new CXMLRPCResult(sampleXMLResult);

   Print(testResult.toString());

   delete testResult;
  }
//+------------------------------------------------------------------+
```

You can find the result below:

```
MQL5-RPC__result_test (EURUSD,H1)       23:16:57
Xupypr:anuta:plencing:aharata:beast:west100:ias:Tim:gery18:ronnielee:investeo:
droslan:Better:1.22200000:0.45600000:1000000000.10099995:true:false:true:-1:0:
1:2002.11.25 02:20:04 : 2011.11.15 00:00:00 : 2012.12.21 00:00:00 :
Single string value:2011.11.15 00:00:00 :-111:true:
```

This is a displayed array of arrays of different MQL5 values converted from XML. As you can see there are [strings](https://www.mql5.com/en/docs/basis/types/stringconst), [double](https://www.mql5.com/en/docs/basis/types/double) values, [boolean](https://www.mql5.com/en/docs/basis/types/integer/boolconst) values, [integer](https://www.mql5.com/en/docs/basis/types/integer) values, [datetime](https://www.mql5.com/en/docs/basis/types/integer/datetime) values that can be accessed from single [CArrayObj](https://www.mql5.com/en/docs/standardlibrary/datastructures/carrayobj) pointer.

### MQL5-RPC proxy class

CXMLRPCServer proxy is a core class for HTTP communication. I used ["Using WinInet in MQL5. Part 2: POST Requests and Files"](https://www.mql5.com/en/articles/276) article to implement HTTP functionality and added custom header consistent with XML-RPC specification.

```
CXMLRPCServerProxy::CXMLRPCServerProxy(string s_proxy,int timeout=0)
  {
   CString proxy;
   proxy.Assign(s_proxy);
   //--- find query path
   int sIdx = proxy.Find(0,"/");
   if (sIdx == -1)
      m_query_path = "/";
   else  {
       m_query_path = proxy.Mid(sIdx, StringLen(s_proxy) - sIdx) + "/";
       s_proxy = proxy.Mid(0, sIdx);
    };
   //--- find query port. 80 is default
   int query_port = 80;
   int pIdx = proxy.Find(0,":");
   if (pIdx != -1) {
      query_port = (int)StringToInteger(proxy.Mid(pIdx+1, sIdx-pIdx));
      s_proxy = proxy.Mid(0, pIdx);
   };
   //Print(query_port);
   //Print(proxy.Mid(pIdx+1, sIdx-pIdx));
   if(InternetAttemptConnect(0)!=0)
     {
      this.m_connectionStatus="InternetAttemptConnect failed.";
      this.m_session=-1;
      this.m_isConnected=false;
      return;
     }
   string agent = "Mozilla";
   string empty = "";

   this.m_session=InternetOpenW(agent,OPEN_TYPE_PRECONFIG,empty,empty,0);

   if(this.m_session<=0)
     {
      this.m_connectionStatus="InternetOpenW failed.";
      this.m_session=-2;
      this.m_isConnected=true;
      return;
     }
   this.m_connection=InternetConnectW(this.m_session,s_proxy,query_port,empty,empty,SERVICE_HTTP,0,0);
   if(this.m_connection<=0)
     {
      this.m_connectionStatus="InternetConnectW failed.";
      return;
     }
   this.m_connectionStatus="Connected.";

  }
```

The CXMLRPCQuery object must be passed to CXMLRPCServerProxy execute() method to trigger XML-RPC call.

The methods returns pointer to CXMLRPCResult object that can be further used in the script that called XML-RPC call.

```
CXMLRPCResult *CXMLRPCServerProxy::execute(CXMLRPCQuery &query)
  {

   //--- creating descriptor of the request
   string empty_string = "";
   string query_string = query.toString();
   string query_method = HEADER_1a;
   string http_version = HEADER_1b;

   uchar data[];

   StringToCharArray(query.toString(),data);

   int ivar=0;
   int hRequest=HttpOpenRequestW(this.m_connection,query_method,m_query_path,http_version,
                                 empty_string,0,FLAG_KEEP_CONNECTION|FLAG_RELOAD|FLAG_PRAGMA_NOCACHE,0);
   if(hRequest<=0)
     {
      Print("-Err OpenRequest");
      InternetCloseHandle(this.m_connection);
      return(new CXMLRPCResult);
     }
   //-- sending the request
   CXMLRPCEncoder encoder;
   string header=encoder.header(m_query_path,ArraySize(data));

   int aH=HttpAddRequestHeadersW(hRequest,header,StringLen(header),HTTP_ADDREQ_FLAG_ADD|HTTP_ADDREQ_FLAG_REPLACE);
   bool hSend=HttpSendRequestW(hRequest,empty_string,0,data,ArraySize(data)-1);
   if(hSend!=true)
     {
      int err=0;
      err=GetLastError();
      Print("-Err SendRequest= ",err);
     }
   string res;

   ReadPage(hRequest,res,false);
   CString out;
   out.Assign(res);
   out.Remove("\n");

   //--- closing all handles
   InternetCloseHandle(hRequest); InternetCloseHandle(hSend);
   CXMLRPCResult* result = new CXMLRPCResult(out.Str());
   return result;
  }
```

### Example 1 - Web Service access

The first working example of MQL5-RPC is a call to external webservice. The example I found uses current currency exchange rate to convert a specific amount of one currency into antoher. The [exact specification](https://www.mql5.com/go?link=http://forums.asp.net/t/1629938.aspx/1 "http://forums.asp.net/t/1629938.aspx/1") of method parameters is available online.

The web service exposes foxrate.currencyConvert method that accepts three parameters, two strings and one float value:

- from currency (eg: USD) = string;
- to currency (eg: GBP) = string;
- amount to convert (eg:100.0) = float.

The implementation is quite short and takes only a couple of lines.

```
//+------------------------------------------------------------------+
//|                              MQL5-RPC_ex1_ExternalWebService.mq5 |
//|                                      Copyright 2011, Investeo.pl |
//|                                           http://www.investeo.pl |
//+------------------------------------------------------------------+
#property copyright "Copyright 2011, Investeo.pl"
#property link      "http://www.investeo.pl"
#property version   "1.00"
//---
#include <MQL5-RPC.mqh>
#include <Arrays\ArrayObj.mqh>
#include <Arrays\ArrayDouble.mqh>
#include <Arrays\ArrayString.mqh>
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
//---
   //--- web service test
   CArrayObj* params = new CArrayObj;

   CArrayString* from   = new CArrayString;
   CArrayString* to     = new CArrayString;
   CArrayDouble* amount = new CArrayDouble;

   from.Add("GBP"); to.Add("USD"); amount.Add(10000.0);
   params.Add(from); params.Add(to); params.Add(amount);

   CXMLRPCQuery query("foxrate.currencyConvert", params);

   Print(query.toString());

   CXMLRPCServerProxy s("foxrate.org/rpc");
   CXMLRPCResult* result;

   result = s.execute(query);

   result.parseXMLResponseRAW();
   Print(result.toString());

   delete params;
   delete result;
  }
//+------------------------------------------------------------------+
```

Since the method returns complex struct I parsed the result using parseXMLResponseRAW() method.

The XML-RPC query in this example looks as follows:

```
<?xml version="1.0" ?>
<methodCall>
      <methodName>foxrate.currencyConvert</methodName>
      <params>
            <param>
                  <value>
                        <string>GBP</string>
                  </value>
            </param>
            <param>
                  <value>
                        <string>USD</string>
                  </value>
           </param>
           <param>
                  <value>
                        <double>10000.00000000</double>
                  </value>
           </param>
      </params>
</methodCall>
```

And the response is returned as a struct wrapped in XML.

```
<?xml version="1.0" ?>
<methodResponse>
      <params>
            <param>
                  <value>
                        <struct>
                              <member>
                                    <name>flerror</name>
                                    <value>
                                          <int>0</int>
                                    </value>
                              </member>
                              <member>
                                    <name>amount</name>
                                    <value>
                                          <double>15773</double>
                                    </value>
                              </member>
                              <member>
                                    <name>message</name>
                                    <value>
                                          <string>cached</string>
                                    </value>
                              </member>
                        </struct>
                  </value>
            </param>
      </params>
</methodResponse>
```

The output from toString() reveals that three values are available in the result: 0 - no error, the second value is the amount of base currency needed for exchange and third parameter is the time of latest exchange rate (it was cached).

```
0:15773.00000000:cached:
```

Let's continue with more interesting use case.

### Example 2 - XML-RPC ATC 2011 Analyzer

Imagine you wanted to grab statistics from [Automated Trading Championship 2011](https://championship.mql5.com/2011/en "https://championship.mql5.com/2011/en") (I did) and use this information in the [MetaTrader 5](https://www.metatrader5.com/ "https://www.metatrader5.com/") terminal. If you would like to know how to achieve that, read on.

Since [MetaQuotes Software Corp.](https://www.metaquotes.net/en "https://www.metaquotes.net/en") is preparing a [Signal following service](https://www.mql5.com/en/signals) that will enable subscribing signals from particular Expert Advisors, I feel the combination of that service with the analyzer will be a very powerful way to perform a sophisticated analysis and provide new ways to benefit from [ATC](https://championship.mql5.com/ "https://championship.mql5.com/"). In fact you could grab data from a few different sources using this method and do a complex Expert Advisor based on it.

### XML-RPC ATC 2011 Analyzer server

First step in making an ATC Analyser server is to prepare output analysis. Let's say we want to grab all participants that accounts equities are above certain threshold and are interested in the positions they all currently hold and display the amount of buy and sell positions statistics of the currency pair we are interested in.

I used Python language and [BeautifulSoup](https://www.mql5.com/go?link=https://www.crummy.com/software/BeautifulSoup/ "http://www.crummy.com/software/BeautifulSoup/") library to retrieve and parse data. If you have never used Python before I strongly recommend to point your browser to [http:/Python.org](https://www.mql5.com/go?link=https://www.python.org/ "http://www.python.org/") and read what this language has to offer, you will not regret it.

If you would like to quickly put your hands on the analyzer instead, [download Python 2.7.1 installer](https://www.mql5.com/go?link=https://www.python.org/ftp/python/2.7.2/python-2.7.2.msi "http://python.org/ftp/python/2.7.2/python-2.7.2.msi") and [setup\_tools](https://www.mql5.com/go?link=http://pypi.python.org/packages/2.7/s/setuptools/setuptools-0.6c11.win32-py2.7.exe[hash]md5%3d57e1e64f6b7c7f1d2eddfc9746bbaf20 "http://pypi.python.org/packages/2.7/s/setuptools/setuptools-0.6c11.win32-py2.7.exe[hash]md5=57e1e64f6b7c7f1d2eddfc9746bbaf20") packages and install both of them. After that run Windows console and change directory to C:\\Python27\\Scripts or whatever the folder you installed Python to. Then issue 'easy\_install BeautifulSoup' command. This will install BeautifulSoup package:

```
C:\Python27\Scripts>easy_install BeautifulSoup
Searching for BeautifulSoup
Reading http://pypi.python.org/simple/BeautifulSoup/
Reading http://www.crummy.com/software/BeautifulSoup/
Reading http://www.crummy.com/software/BeautifulSoup/download/
Best match: BeautifulSoup 3.2.0
Downloading http://www.crummy.com/software/BeautifulSoup/download/3.x/BeautifulS
oup-3.2.0.tar.gz
Processing BeautifulSoup-3.2.0.tar.gz
Running BeautifulSoup-3.2.0\setup.py -q bdist_egg --dist-dir c:\users\przemek\ap
pdata\local\temp\easy_install-zc1s2v\BeautifulSoup-3.2.0\egg-dist-tmp-hnpwoo
zip_safe flag not set; analyzing archive contents...
C:\Python27\lib\site-packages\setuptools\command\bdist_egg.py:422: UnicodeWarnin
g: Unicode equal comparison failed to convert both arguments to Unicode - interp
reting them as being unequal
  symbols = dict.fromkeys(iter_symbols(code))
Adding beautifulsoup 3.2.0 to easy-install.pth file

Installed c:\python27\lib\site-packages\beautifulsoup-3.2.0-py2.7.egg
Processing dependencies for BeautifulSoup
Finished processing dependencies for BeautifulSoup

C:\Python27\Scripts>
```

After that you should be able to run Python console and issue 'import BeautifulSoup' command with no error.

```
Python 2.7.1 (r271:86832, Nov 27 2010, 18:30:46) [MSC v.1500 32 bit (Intel)] on win32
Type "copyright", "credits" or "license()" for more information.
>>> import BeautifulSoup
>>>
```

And run the attached analyzer with 'python ContestantParser.py' command. The analyzer fetches the website, does some keyword lookup and puts the output on the console.

```
import re
import urllib
from BeautifulSoup import BeautifulSoup

class ContestantParser(object):
    URL_PREFIX = 'https://championship.mql5.com/2011/en/users/'

    def __init__(self, name):
        print "User:", name
        url = ContestantParser.URL_PREFIX + name
        feed = urllib.urlopen(url)
        s = BeautifulSoup(feed.read())
        account = s.findAll('td', attrs={'class' : 'alignRight'})
        self.balance = float(account[0].contents[0].replace(' ', ''))
        self.equity = float(account[2].contents[0].replace(' ', ''))
        terminals = s.findAll('table', attrs={'class': 'terminal'})
        terminal = terminals[0]
        trs = terminal.findAll('tr')
        pairs = terminal.findAll('span', attrs={'class':'stateText'})
        self.stats = {}
        for i in range(len(pairs)-1):
            if len(pairs[i].string)> 6:
                break
            self.stats[str(pairs[i].string)] = str(trs[i+1].contents[7].string)

    def __str__(self):
        for k in self.stats.keys():
            print k, self.stats[k]

        return "Bal " + str(self.balance) + " Equ " +  str(self.equity)

    def position(self, pair):
        if pair in self.stats.keys():
            return self.stats[pair]
        else:
            return "none"

class ContestantsFinder(object):
    URL_PREFIX = "https://championship.mql5.com/2011/en/users/index/page"
    URL_SUFFIX = "?orderby=equity&dir=down"

    def __init__(self, min_equity):
        self.min_equity = min_equity
        self.user_list = []
        self.__find()

    def __find(self):
        isLastPage = False
        pageCnt = 1
        while isLastPage == False:
            url = ContestantsFinder.URL_PREFIX + str(pageCnt) + \
                  ContestantsFinder.URL_SUFFIX
            feed = urllib.urlopen(url)
            s = BeautifulSoup(feed.read())
            urows = s.findAll('tr', attrs={'class' : re.compile('row.*')})
            for row in urows:
                user_name = row.contents[5].a.string
                equity = float(row.contents[19].string.replace(' ',''))
                if equity <= self.min_equity:
                    isLastPage = True
                    break
                self.user_list.append(str(user_name))
                print user_name, equity
            pageCnt += 1


    def list(self):
        return self.user_list



if __name__ == "__main__":

    # find all contestants with equity larger than threshold
    contestants = ContestantsFinder(20000.0)
    print contestants.list()
    # display statistics
    print "* Statistics *"
    for contestant in contestants.list():
        u = ContestantParser(contestant)
        print u
        print u.position('eurusd')
        print '-' * 60

```

This was achieved by simply looking inside HTML source code and finding relations between tags. If this code is executed directly from the Python console it outputs contestants names, their balance and equity and all current open positions.

Please observe the output of a sample query:

```
* Statistics *
User: Tim
Bal 31459.2 Equ 31459.2
none
------------------------------------------------------------
User: enivid
eurusd sell
euraud sell
Bal 26179.98 Equ 29779.89
sell
------------------------------------------------------------
User: ias
eurjpy sell
usdchf sell
gbpusd buy
eurgbp buy
eurchf sell
audusd buy
gbpjpy sell
usdjpy buy
usdcad buy
euraud buy
Bal 15670.0 Equ 29345.66
none
------------------------------------------------------------
User: plencing
eurusd buy
Bal 30233.2 Equ 29273.2
buy
------------------------------------------------------------
User: anuta
audusd buy
usdcad sell
gbpusd buy
Bal 28329.85 Equ 28359.05
none
------------------------------------------------------------
User: gery18
Bal 27846.7 Equ 27846.7
none
------------------------------------------------------------
User: tornhill
Bal 27402.4 Equ 27402.4
none
------------------------------------------------------------
User: rikko
eurusd sell
Bal 25574.8 Equ 26852.8
sell
------------------------------------------------------------
User: west100
eurusd buy
Bal 27980.5 Equ 26255.5
buy
------------------------------------------------------------
...
```

Looks good, isn't it? With that data we can gather interesting statistics and implement a XML-RPC service that will provide them on demand. The XMLRPC client from [MetaTrader 5](https://www.metatrader5.com/ "https://www.metatrader5.com/") could fetch those statistics when needed.

In order to make a XMLRPC server I used python's SimpleXMLRPC server library. The server exposes two methods to the outer world: listContestants and getStats. While the former only displays contestant names that have equity above certain threshold, the latter displays how many of them have open position on a given currency pair, and what is the ratio of buy to sell on those positions.

Together with signals service and/or trade copier from article you might want to confirm that the setup you are actually trading has more chance to be a profitable one.

```
from SimpleXMLRPCServer import SimpleXMLRPCServer
from SimpleXMLRPCServer import SimpleXMLRPCRequestHandler

from PositionGrabberTest import ContestantParser, ContestantsFinder

class RequestHandler( SimpleXMLRPCRequestHandler ):
    rpc_path = ( '/RPC2', )

class ATC2011XMLRPCServer(SimpleXMLRPCServer):

    def __init__(self):
        SimpleXMLRPCServer.__init__( self, ("192.168.235.168", 6666), requestHandler=RequestHandler, logRequests = False)

        self.register_introspection_functions()
        self.register_function( self.xmlrpc_contestants, "listContestants" )
        self.register_function( self.xmlrpc_get_position_stats, "getStats" )


    def xmlrpc_contestants(self, min_equity):
        try:
            min_equity = float(min_equity)
        except ValueError as error:
            print error
            return []

        self.contestants = ContestantsFinder(min_equity)

        return self.contestants.list()

    def xmlrpc_get_position_stats(self, pair):
        total_users = len(self.contestants.list())
        holding_pair = 0
        buy_positions = 0

        for username in self.contestants.list():
            u = ContestantParser(username)
            position = u.position(pair)
            if position != "none":
                holding_pair += 1
                if position == "buy":
                    buy_positions += 1

        return [ total_users, holding_pair, buy_positions ]


if __name__ == '__main__':
    server = ATC2011XMLRPCServer()
    server.serve_forever()
```

This server can be access directly from Python console. When running this script remember to change IP address to the one your system is using.

```
C:\Program Files\MetaTrader 5\MQL5>python
Python 2.7.2 (default, Jun 12 2011, 15:08:59) [MSC v.1500 32 bit (Intel)] on win
32
Type "help", "copyright", "credits" or "license" for more information.
>>> from xmlrpclib import ServerProxy
>>> u = ServerProxy('http://192.168.235.168:6666')
>>> u.listContestants(20000.0)
['lf8749', 'ias', 'aharata', 'Xupypr', 'beast', 'Tim', 'tmt0086', 'yyy999', 'bob\
sley', 'Diubakin', 'Pirat', 'notused', 'AAA777', 'ronnielee', 'samclider-2010',\
'gery18', 'p96900', 'Integer', 'GradyLee', 'skarkalakos', 'zioliberato', 'kgo',\
'enivid', 'Loky', 'Gans-deGlucker', 'zephyrrr', 'InvestProm', 'QuarkSpark', 'ld7\
3', 'rho2011', 'tornhill', 'botaxuper']
>>>
```

Probably when you are reading this the contestants results have totally changed but you should get the idea on what I was trying to achieve.

Calling getStats() method with "eurusd" param returns three numbers:

```
In : u.getStats("eurusd")
Out: [23, 12, 9]
```

First being a number of contestants that have equity above the threshold, second being a number of contestants that hold open EURUSD position and the third number is a number of contestants that have long eurusd position. In this case two thirds of winning contestants have opened long EURUSD therefore this can serve as a confirmation signal when your robot or your indicators generated 'open eurusd long' signal.

### MQL5-RPC ATC 2011 Analyzer client

It is time to use MQL5-RPC framework to fetch this data on demand in MetaTrader 5. In fact its usage is self explanatory if you follow the source code.

```
//+------------------------------------------------------------------+
//|                           MQL5-RPC_ex2_ATC2011AnalyzerClient.mq5 |
//|                                      Copyright 2011, Investeo.pl |
//|                                           http://www.investeo.pl |
//+------------------------------------------------------------------+
#property copyright "Copyright 2011, Investeo.pl"
#property link      "http://www.investeo.pl"
#property version   "1.00"

#include <MQL5-RPC.mqh>
#include <Arrays\ArrayObj.mqh>
#include <Arrays\ArrayInt.mqh>
#include <Arrays\ArrayDouble.mqh>
#include <Arrays\ArrayString.mqh>
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
//---
   //--- ATC 2011 analyzer test
   CXMLRPCServerProxy s("192.168.235.168:6666");
   CXMLRPCResult* result;

   //--- Get list of contestants
   CArrayObj* params1 = new CArrayObj;
   CArrayDouble* amount = new CArrayDouble;

   amount.Add(20000.0); params1.Add(amount);
   CXMLRPCQuery query("listContestants", params1);

   Print(query.toString());

   result = s.execute(query);
   Print(result.toString());
   delete result;

   //--- Get position statistics
   CArrayObj* params2 = new CArrayObj;
   CArrayString* pair = new CArrayString;

   pair.Add("eurusd"); params2.Add(pair);
   CXMLRPCQuery query2("getStats", params2);

   Print(query2.toString());

   result = s.execute(query2);

   CArrayObj* resultArray = result.getResults();
   CArrayInt* stats = resultArray.At(0);

   Print("Contestants = " + IntegerToString(stats.At(0)) +
         ". EURUSD positions = " + IntegerToString(stats.At(1)) +
         ". BUY positions = " + IntegerToString(stats.At(2)));

   delete params1;
   delete params2;

   delete result;
  }
//+------------------------------------------------------------------+
```

This is what happens when the script is called. The query is wrapped into XML:

```
<?xml version="1.0" ?>
<methodCall>
      <methodName>listContestants</methodName>
      <params>
            <param>
                  <value>
                        <double>20000.00000000</double>
                  </value>
            </param>
      </params>
</methodCall>
```

and after a while response is received from XML-RPC server. In my case I used Linux host that run python XML-RPC service and called it from within Windows VirtualBox guest installation that run MetaTrader 5.

```
<?xml version="1.0" ?>
<methodResponse>
      <params>
            <param>
                  <value>
                        <array>
                              <data>
                                    <value>
                                          <string>lf8749</string>
                                    </value>
                                    <value>
                                          <string>ias</string>
                                    </value>
                                    <value>
                                          <string>Xupypr</string>
                                    </value>
                                    <value>
                                          <string>aharata</string>
                                    </value>
                                    <value>
                                          <string>beast</string>
                                    </value>
                                    <value>
                                          <string>Tim</string>
                                    </value>
                                    <value>
                                          <string>tmt0086</string>
                                    </value>
                                    <value>
                                          <string>yyy999</string>
                                    </value>
                                    <value>
                                          <string>bobsley</string>
                                    </value>
                                    <value>
                                          <string>Diubakin</string>
                                    </value>
                                    <value>
                                          <string>Pirat</string>
                                    </value>
                                    <value>
                                          <string>AAA777</string>
                                    </value>
                                    <value>
                                          <string>notused</string>
                                    </value>
                                    <value>
                                          <string>ronnielee</string>
                                    </value>
                                    <value>
                                          <string>samclider-2010</string>
                                    </value>
                                    <value>
                                          <string>gery18</string>
                                    </value>
                                    <value>
                                          <string>Integer</string>
                                    </value>
                                    <value>
                                          <string>GradyLee</string>
                                    </value>
                                    <value>
                                          <string>p96900</string>
                                    </value>
                                    <value>
                                          <string>skarkalakos</string>
                                    </value>
                                    <value>
                                          <string>Loky</string>
                                    </value>
                                    <value>
                                          <string>zephyrrr</string>
                                    </value>
                                    <value>
                                          <string>Medvedev</string>
                                    </value>
                                    <value>
                                          <string>Gans-deGlucker</string>
                                    </value>
                                    <value>
                                          <string>InvestProm</string>
                                    </value>
                                    <value>
                                          <string>zioliberato</string>
                                    </value>
                                    <value>
                                          <string>QuarkSpark</string>
                                    </value>
                                    <value>
                                          <string>rho2011</string>
                                    </value>
                                    <value>
                                          <string>ld73</string>
                                    </value>
                                    <value>
                                          <string>enivid</string>
                                    </value>
                                    <value>
                                          <string>botaxuper</string>
                                    </value>
                              </data>
                        </array>
                  </value>
            </param>
      </params>
</methodResponse>
```

The result of the first query displayed using toString() method is as follows:

```
lf8749:ias:Xupypr:aharata:beast:Tim:tmt0086:yyy999:bobsley:Diubakin:Pirat:
AAA777:notused:ronnielee:samclider-2010:gery18:Integer:GradyLee:p96900:
skarkalakos:Loky:zephyrrr:Medvedev:Gans-deGlucker:InvestProm:zioliberato:
QuarkSpark:rho2011:ld73:enivid:botaxuper:
```

The second query is used to call getStats() method.

```
<?xml version="1.0" ?>
<methodCall>
      <methodName>getStats</methodName>
      <params>
            <param>
                  <value>
                        <string>eurusd</string>
                  </value>
            </param>
      </params>
</methodCall>
```

The XML-RPC response is simple, it contains only three integer values.

```
<?xml version="1.0" ?>
<methodResponse>
      <params>
            <param>
                  <value>
                        <array>
                              <data>
                                    <value>
                                          <int>31</int>
                                    </value>
                                    <value>
                                          <int>10</int>
                                    </value>
                                    <value>
                                          <int>3</int>
                                    </value>
                              </data>
                        </array>
                  </value>
            </param>
      </params>
</methodResponse>
```

This time I took another approach and accessed returned values as MQL5 variables.

```
MQL5-RPC_ex2_ATC2011AnalyzerClient (AUDUSD,H1)  12:39:23

lf8749:ias:Xupypr:aharata:beast:Tim:tmt0086:yyy999:bobsley:Diubakin:Pirat:
AAA777:notused:ronnielee:samclider-2010:gery18:Integer:GradyLee:p96900:
skarkalakos:Loky:zephyrrr:Medvedev:Gans-deGlucker:InvestProm:zioliberato:
QuarkSpark:rho2011:ld73:enivid:botaxuper:

MQL5-RPC_ex2_ATC2011AnalyzerClient (AUDUSD,H1)  12:39:29

Contestants = 31. EURUSD positions = 10. BUY positions = 3
```

As you can see the ouput is in an easily understandable form.

### Conclusion

I presented a new MQL5-RPC framework that enables MetaTrader 5 to execute remote procecure calls using [XML-RPC protocol](https://en.wikipedia.org/wiki/XML-RPC "https://en.wikipedia.org/wiki/XML-RPC"). Two use cases were presented, the first one being access to a web service and the second one was a custom analyzer for [Automated Trading Championship 2011](https://championship.mql5.com/2011/en "https://championship.mql5.com/2011/en"). Those two examples should serve as a basis for further experiments.

I believe MQL5-RPC is a very powerful feature that can be used in many different ways. I have decided to make the framework Open Source and published it for under [GPL licence](https://en.wikipedia.org/wiki/GNU_General_Public_License "https://en.wikipedia.org/wiki/GNU_General_Public_License") at [http://code.google.com/p/mql5-rpc/](https://www.mql5.com/go?link=https://code.google.com/archive/p/mql5-rpc "http://code.google.com/p/mql5-rpc/"). If anyone would like to help to make the code bullet proof, refactor it or make bugfixes the doors are open to join the project. The source code with all examples is also available as the attachment to the article.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/342.zip "Download all attachments in the single ZIP archive")

[mql5-rpc.zip](https://www.mql5.com/en/articles/download/342/mql5-rpc.zip "Download mql5-rpc.zip")(22.18 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Securing MQL5 code: Password Protection, Key Generators, Time-limits, Remote Licenses and Advanced EA License Key Encryption Techniques](https://www.mql5.com/en/articles/359)
- [Applying The Fisher Transform and Inverse Fisher Transform to Markets Analysis in MetaTrader 5](https://www.mql5.com/en/articles/303)
- [Advanced Adaptive Indicators Theory and Implementation in MQL5](https://www.mql5.com/en/articles/288)
- [Using MetaTrader 5 Indicators with ENCOG Machine Learning Framework for Timeseries Prediction](https://www.mql5.com/en/articles/252)
- [Exposing C# code to MQL5 using unmanaged exports](https://www.mql5.com/en/articles/249)
- [Moving Mini-Max: a New Indicator for Technical Analysis and Its Implementation in MQL5](https://www.mql5.com/en/articles/238)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/5566)**
(17)


![nos40](https://c.mql5.com/avatar/avatar_na2.png)

**[nos40](https://www.mql5.com/en/users/nos40)**
\|
10 May 2014 at 12:08

Hi,

The link for this article seems to be broken : error 404

![Sergey Golubev](https://c.mql5.com/avatar/2012/12/50D09349-3D90.jpg)

**[Sergey Golubev](https://www.mql5.com/en/users/newdigital)**
\|
10 May 2014 at 14:06

**nos40:**

Hi,

The link for this article seems to be broken : error 404

Thanks for informing.

I reported to Servicedesk


![Alain Verleyen](https://c.mql5.com/avatar/2024/5/663a6cdf-e866.jpg)

**[Alain Verleyen](https://www.mql5.com/en/users/angevoyageur)**
\|
10 May 2014 at 14:11

**newdigital:**

Thanks for informing.

I reported to Servicedesk

Me too ![](https://c.mql5.com/3/33/bigsmile-fixed__3.png) (by the way I reported that there is a lot a wrong link in Articles section.)


![Nono Momo](https://c.mql5.com/avatar/avatar_na2.png)

**[Nono Momo](https://www.mql5.com/en/users/cubeer)**
\|
23 Feb 2015 at 06:00

**MetaQuotes:**

New Article [MQL5-RPC Remote Procedure Calls from MQL5: Web Service Access and XML-RPC Automated Trading Tournament Analytics for Fun and Profit](https://www.mql5.com/en/articles/342) has been released:

Author: [investeo](https://www.mql5.com/en/users/investeo "investeo")

mark

![Stefano Cerbioni](https://c.mql5.com/avatar/2016/6/5755B8A8-5894.png)

**[Stefano Cerbioni](https://www.mql5.com/en/users/faustf)**
\|
26 Jul 2023 at 12:48

exist a version for mt4 ? thanks


![Custom Graphical Controls. Part 2. Control Library](https://c.mql5.com/2/0/Graphic_Controls_Library_MQL5.png)[Custom Graphical Controls. Part 2. Control Library](https://www.mql5.com/en/articles/313)

The second article of the "Custom Graphical Controls" series introduces a control library for handling the main problems arising in interaction between a program (Expert Advisor, script, indicator) and a user. The library contains a great number of classes (CInputBox, CSpinInputBox, CCheckBox, CRadioGroup, CVSсrollBar, CHSсrollBar, CList, CListMS, CComBox, CHMenu, CVMenu, CHProgress, CDialer, CDialerInputBox, CTable) and examples of their use.

![Interview with Valery Mazurenko (ATC 2011)](https://c.mql5.com/2/0/ava_notused.png)[Interview with Valery Mazurenko (ATC 2011)](https://www.mql5.com/en/articles/553)

The task of writing an Expert Advisor trading on multiple currency pairs is complex both in terms of finding suitable strategies and from the technological side. But if the goal is set clear, nothing is impossible then. It was four times already that Vitaly Mazurenko (notused) submitted his multi-currency Expert Advisor. It seems, he has managed to find the right way this time.

![Interview with Li Fang (ATC 2011)](https://c.mql5.com/2/0/Li_Fang_interview.png)[Interview with Li Fang (ATC 2011)](https://www.mql5.com/en/articles/554)

On the seventh week of the Championship, Li Fang's Expert Advisor (lf8749) set a new record - it earned over $100,000 in 10 trades. This successful series helped the Expert Advisor to stay at the very top of the Automated Trading Championship 2011 rating for two weeks. In this interview we tried to find out the secret of Li Fang's success.

![The Basics of Object-Oriented Programming](https://c.mql5.com/2/0/OOP_avatar.png)[The Basics of Object-Oriented Programming](https://www.mql5.com/en/articles/351)

You don't need to know what are polymorphism, encapsulation, etc. all about in to use object-oriented programming (OOP)... you may simply use these features. This article covers the basics of OOP with hands-on examples.

[![](https://www.mql5.com/ff/sh/592yc11u3j4rs5z9z2/01.png)How AI helps create robots for MetaTrader 5Learn from our book "Neural Networks in Algo Trading with MQL5"Read](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/neurobook%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.neurobook%26utm_content=visit.page%26utm_campaign=neurobook.promo.04.2024&a=ghrobswocqgvhztzjldphupateyllpro&s=9929cb0b8629585b5a42fabc06c525e41f6c0ebdf3045d044a5413b93ea88b47&uid=&ref=https://www.mql5.com/en/articles/342&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5071765080187547145)

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