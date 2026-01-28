---
title: Building AI-Powered Trading Systems in MQL5 (Part 1): Implementing JSON Handling for AI APIs
url: https://www.mql5.com/en/articles/19562
categories: Trading Systems, Integration, Expert Advisors
relevance_score: 15
scraped_at: 2026-01-22T17:09:35.388044
---

[Need a reliable hosting solution for your robots?Contact your broker and find out about available Sponsored MetaTrader VPS offeringsLearn more![](https://www.mql5.com/ff/sh/0pw0dk81s56qy774z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=vljwvezfjkfbvviocwskggexlvgykvob&s=70cf8e354b9a125332533ffb65d7365abe8dde5b5c1ede9caac479a9e9df4f25&uid=&ref=https://www.mql5.com/en/articles/19562&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5048828039271391091)

MetaTrader 5 / Trading systems


### Introduction

In this article series, we introduce the integration of [Artificial Intelligence](https://en.wikipedia.org/wiki/Artificial_intelligence "https://en.wikipedia.org/wiki/Artificial_intelligence") (AI) into trading systems using [MetaQuotes Language 5](https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5 "https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5") (MQL5), starting with this part, where we develop a robust [JavaScript Object Notation](https://en.wikipedia.org/wiki/JSON "https://en.wikipedia.org/wiki/JSON") (JSON) parsing framework to handle data exchange for AI [Application Programming Interface](https://en.wikipedia.org/wiki/API "https://en.wikipedia.org/wiki/API") (API) interactions, such as with ChatGPT. We focus on creating a foundation for processing JSON structures to enable seamless communication with AI services for future trading applications. We will cover the following topics:

1. [Understanding JSON and Its Role in AI Integration](https://www.mql5.com/en/articles/19562#para1)
2. [Implementation in MQL5](https://www.mql5.com/en/articles/19562#para2)
3. [Testing the JSON Parser](https://www.mql5.com/en/articles/19562#para3)
4. [Conclusion](https://www.mql5.com/en/articles/19562#para4)

By the end, you’ll have a solid foundation for handling JSON data, setting the stage for AI-driven trading systems—let’s dive in!

### Understanding JSON and Its Role in AI Integration

[JSON](https://en.wikipedia.org/wiki/JSON "https://en.wikipedia.org/wiki/JSON") (JavaScript Object Notation) is a lightweight, text-based data interchange format widely used for structuring and transmitting data between systems, particularly in web-based APIs, due to its simplicity, readability, and compatibility across programming languages. In the context of AI-powered trading systems, like we want to build, serves as the standard format for exchanging data with AI APIs, such as OpenAI’s ChatGPT, enabling MQL5 applications to send trading-related prompts and receive structured responses for decision-making. Our approach in this article focuses on building a JSON parsing framework to handle these API interactions, laying the groundwork for integrating AI-driven insights into automated trading strategies.

What is JSON and Why It Matters

JSON represents data as key-value pairs, arrays, and nested objects, making it ideal for encoding complex information like market data, trading signals, or AI responses in a format that is both human-readable and machine-parsable. For example, a JSON object might look like this:

```
{
  "model": "gpt-3.5-turbo",
  "messages": [\
    {"role": "user", "content": "Analyze EURUSD trend"},\
    {"role": "assistant", "content": "EURUSD shows a bullish trend"}\
  ],
  "max_tokens": 500
}
```

This structure includes strings, numbers, arrays, and nested objects, which an MQL5 Expert Advisor (EA) must parse to extract relevant information, such as the AI’s response to a trading query. JSON’s role in AI integration is critical here because APIs return responses in JSON format, requiring the program to [serialize](https://en.wikipedia.org/wiki/Serialization "https://en.wikipedia.org/wiki/Serialization") inputs (convert data to JSON) and deserialize outputs (parse JSON into usable data) to enable dynamic trading decisions. If this sounds like a jargon to you, here is a quick visualization of what [data serialization and deserialization](https://www.mql5.com/go?link=https://medium.com/@khemanta/data-serialization-and-deserialization-what-is-it-29b5ca7a756f "https://medium.com/@khemanta/data-serialization-and-deserialization-what-is-it-29b5ca7a756f") means.

![SERIALIZATION & DESERIALIZATION](https://c.mql5.com/2/169/Screenshot_2025-09-16_142039.png)

Roadmap for Implementation

Our implementation plan involves creating a JSON handling [class](https://www.mql5.com/en/docs/basis/types/classes) that supports the following functionalities:

- Data Representation: A class to store JSON values with attributes for type, key, and value (e.g., [string](https://www.mql5.com/en/book/basis/builtin_types/strings), number, or boolean), and an array for child elements to handle nested structures.
- Parsing Logic: Methods to [deserialize](https://www.mql5.com/go?link=https://medium.com/@khemanta/data-serialization-and-deserialization-what-is-it-29b5ca7a756f "https://medium.com/@khemanta/data-serialization-and-deserialization-what-is-it-29b5ca7a756f") JSON strings into objects, processing characters to identify structures like objects, arrays, and primitive types, while handling whitespace and [escape](https://en.wikipedia.org/wiki/Escape_character "https://en.wikipedia.org/wiki/Escape_character") sequences.
- Serialization Logic: Methods to convert internal data back into JSON strings, ensuring proper formatting for API requests, including escaping special characters.
- Error Handling: Robust checks for invalid JSON, type mismatches, and out-of-bounds access to prevent crashes during API communication.
- User Interface Preparation: Laying the groundwork for future integration with a user interface to input prompts and display AI responses, which will rely on parsed JSON data.

We will test this framework to ensure it can parse typical AI API responses, such as those from OpenAI, and serialize trading-related prompts accurately as shown below.

![MQL5 AI RESPONSES](https://c.mql5.com/2/169/Screenshot_2025-09-14_132805.png)

By mastering JSON parsing in this article, we ensure that future AI-driven programs can process complex data structures, paving the way for sophisticated trading strategies that combine price action, chart patterns, and AI-generated insights. Let’s proceed to the implementation!

### Implementation in MQL5

To implement the integration, we will first create a comprehensive parsing [class](https://www.mql5.com/en/docs/basis/types/classes) that we will use for our first prompts and future applications. Here is how we achieve that.

```
//+------------------------------------------------------------------+
//|                                            a. JSON Code File.mq5 |
//|                           Copyright 2025, Allan Munene Mutiiria. |
//|                                   https://t.me/Forex_Algo_Trader |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, Allan Munene Mutiiria."
#property link "https://t.me/Forex_Algo_Trader"
#property version "1.00"
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit(){return(INIT_SUCCEEDED);}
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason){}
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick(){}
//+------------------------------------------------------------------+
//| JSON (JavaScript Object Notation)                                |
//+------------------------------------------------------------------+
#define DEBUG_PRINT false
//+------------------------------------------------------------------+
//| Enumeration of JSON value types                                  |
//+------------------------------------------------------------------+
enum JsonValueType {JsonUndefined,JsonNull,JsonBoolean,JsonInteger,JsonDouble,JsonString,JsonArray,JsonObject};
//+------------------------------------------------------------------+
//| Class representing a JSON value                                  |
//+------------------------------------------------------------------+
class JsonValue{
   public:
      // CONSTRUCTOR
      virtual void Reset(){
         m_parent=NULL;                                         //--- Set parent pointer to NULL
         m_key="";                                              //--- Clear the key string
         m_type=JsonUndefined;                                  //--- Set type to undefined
         m_booleanValue=false;                                  //--- Set boolean value to false
         m_integerValue=0;                                      //--- Set integer value to zero
         m_doubleValue=0;                                       //--- Set double value to zero
         m_stringValue="";                                      //--- Clear the string value
         ArrayResize(m_children,0);                             //--- Resize children array to zero
      }
      virtual bool CopyFrom(const JsonValue &source){
         m_key=source.m_key;                                    //--- Copy the key from source
         CopyDataFrom(source);                                  //--- Copy data from source
         return true;                                           //--- Return success
      }
      virtual void CopyDataFrom(const JsonValue &source){
         m_type=source.m_type;                                  //--- Copy the type from source
         m_booleanValue=source.m_booleanValue;                  //--- Copy the boolean value from source
         m_integerValue=source.m_integerValue;                  //--- Copy the integer value from source
         m_doubleValue=source.m_doubleValue;                    //--- Copy the double value from source
         m_stringValue=source.m_stringValue;                    //--- Copy the string value from source
         CopyChildrenFrom(source);                              //--- Copy children from source
      }
      virtual void CopyChildrenFrom(const JsonValue &source){
         int numChildren=ArrayResize(m_children,ArraySize(source.m_children)); //--- Resize children array to match source size
         for(int index=0; index<numChildren; index++){          //--- Loop through each child
            m_children[index]=source.m_children[index];         //--- Copy child from source
            m_children[index].m_parent=GetPointer(this);        //--- Set parent of child to current object
         }
      }
   public:
      JsonValue m_children[];                                   //--- Array to hold child JSON values
      string m_key;                                             //--- Key for this JSON value
      string m_temporaryKey;                                    //--- Temporary key used during parsing
      JsonValue *m_parent;                                      //--- Pointer to parent JSON value
      JsonValueType m_type;                                     //--- Type of this JSON value
      bool m_booleanValue;                                      //--- Boolean value storage
      long m_integerValue;                                      //--- Integer value storage
      double m_doubleValue;                                     //--- Double value storage
      string m_stringValue;                                     //--- String value storage
      static int encodingCodePage;                              //--- Static code page for encoding
   public:
      JsonValue(){
         Reset();                                               //--- Call reset to initialize
      }
      JsonValue(JsonValue *parent,JsonValueType type){
         Reset();                                               //--- Call reset to initialize
         m_type=type;                                           //--- Set the type
         m_parent=parent;                                       //--- Set the parent
      }
      JsonValue(JsonValueType type,string value){
         Reset();                                               //--- Call reset to initialize
         SetFromString(type,value);                             //--- Set value from string based on type
      }
      JsonValue(const int integerValue){
         Reset();                                               //--- Call reset to initialize
         m_type=JsonInteger;                                    //--- Set type to integer
         m_integerValue=integerValue;                           //--- Set integer value
         m_doubleValue=(double)m_integerValue;                  //--- Convert to double
         m_stringValue=IntegerToString(m_integerValue);         //--- Convert to string
         m_booleanValue=m_integerValue!=0;                      //--- Set boolean based on integer
      }
      JsonValue(const long longValue){
         Reset();                                               //--- Call reset to initialize
         m_type=JsonInteger;                                    //--- Set type to integer
         m_integerValue=longValue;                              //--- Set integer value
         m_doubleValue=(double)m_integerValue;                  //--- Convert to double
         m_stringValue=IntegerToString(m_integerValue);         //--- Convert to string
         m_booleanValue=m_integerValue!=0;                      //--- Set boolean based on integer
      }
      JsonValue(const double doubleValue){
         Reset();                                               //--- Call reset to initialize
         m_type=JsonDouble;                                     //--- Set type to double
         m_doubleValue=doubleValue;                             //--- Set double value
         m_integerValue=(long)m_doubleValue;                    //--- Convert to integer
         m_stringValue=DoubleToString(m_doubleValue);           //--- Convert to string
         m_booleanValue=m_integerValue!=0;                      //--- Set boolean based on integer
      }
      JsonValue(const bool booleanValue){
         Reset();                                               //--- Call reset to initialize
         m_type=JsonBoolean;                                    //--- Set type to boolean
         m_booleanValue=booleanValue;                           //--- Set boolean value
         m_integerValue=m_booleanValue;                         //--- Convert to integer
         m_doubleValue=m_booleanValue;                          //--- Convert to double
         m_stringValue=IntegerToString(m_integerValue);         //--- Convert to string
      }
      JsonValue(const JsonValue &other){
         Reset();                                               //--- Call reset to initialize
         CopyFrom(other);                                       //--- Copy from other object
      }
      // DECONSTRUCTOR
      ~JsonValue(){
         Reset();                                               //--- Call reset to clean up
      }
}
```

We begin the implementation of the [JSON parsing](https://en.wikipedia.org/wiki/JSON "https://en.wikipedia.org/wiki/JSON") framework, focusing on the foundational "JsonValue" [class](https://www.mql5.com/en/docs/basis/types/classes) to handle JSON data for AI API integration. First, we [define a macro](https://www.mql5.com/en/docs/basis/preprosessor/constant) "DEBUG\_PRINT" set to false to control debugging output, ensuring minimal logging during production. Then, we establish an [enumeration](https://www.mql5.com/en/docs/basis/types/integer/enumeration) "JsonValueType" with values (JsonUndefined, JsonNull, JsonBoolean, JsonInteger, JsonDouble, JsonString, JsonArray, JsonObject) to categorize JSON [data types](https://www.mql5.com/en/docs/basis/types), enabling the class to handle diverse structures like those returned by APIs.

Next, we implement the "JsonValue" class with [public members](https://www.mql5.com/en/book/oop/classes_and_interfaces/classes_access_rights): an array "m\_children" for nested JSON elements, strings "m\_key" and "m\_temporaryKey" for key storage during parsing, a pointer "m\_parent" for hierarchical relationships, a "JsonValueType" variable "m\_type" for the data type, and variables "m\_booleanValue", "m\_integerValue", "m\_doubleValue", and "m\_stringValue" for storing respective data, plus a static "encodingCodePage" for character encoding. We provide multiple [constructors](https://www.mql5.com/en/book/oop/classes_and_interfaces/classes_ctors) to initialize "JsonValue" objects: a default constructor calling "Reset", one with parent and type, one with type and string value, and others for integer, long, double, boolean, and copy construction, ensuring flexibility in creating JSON elements.

The "Reset" [method](https://www.mql5.com/en/docs/basis/function) clears all members to default values (e.g., null parent, empty strings, undefined type, zeroed values, and empty children array), while "CopyFrom", "CopyDataFrom", and "CopyChildrenFrom" methods facilitate deep copying of JSON structures, including children, with proper parent reassignment. This foundational implementation sets up the structure for parsing and manipulating JSON data, critical for future AI API interactions. We can then proceed to implement some pure [virtual function](https://www.mql5.com/en/docs/basis/oop/virtual) of the class, still under a [public access specifier](https://www.mql5.com/en/book/oop/classes_and_interfaces/classes_access_rights) for completeness. We use virtual to make the system flexible and polymorphic.

```
public:
   virtual bool IsNumericValue(){
      return (m_type==JsonDouble || m_type==JsonInteger);    //--- Check if type is double or integer
   }
   virtual JsonValue *FindChildByKey(string key){
      for(int index=ArraySize(m_children)-1; index>=0; --index){ //--- Loop backwards through children
         if(m_children[index].m_key==key){                   //--- Check if key matches
            return GetPointer(m_children[index]);            //--- Return pointer to matching child
         }
      }
      return NULL;                                           //--- Return NULL if no match
   }
   virtual JsonValue *HasChildWithKey(string key,JsonValueType type=JsonUndefined);
   virtual JsonValue *operator[](string key);
   virtual JsonValue *operator[](int index);
   void operator=(const JsonValue &value){
      CopyFrom(value);                                       //--- Copy from value
   }
   void operator=(const int integerValue){
      m_type=JsonInteger;                                    //--- Set type to integer
      m_integerValue=integerValue;                           //--- Set integer value
      m_doubleValue=(double)m_integerValue;                  //--- Convert to double
      m_booleanValue=m_integerValue!=0;                      //--- Set boolean based on integer
   }
   void operator=(const long longValue){
      m_type=JsonInteger;                                    //--- Set type to integer
      m_integerValue=longValue;                              //--- Set integer value
      m_doubleValue=(double)m_integerValue;                  //--- Convert to double
      m_booleanValue=m_integerValue!=0;                      //--- Set boolean based on integer
   }
   void operator=(const double doubleValue){
      m_type=JsonDouble;                                     //--- Set type to double
      m_doubleValue=doubleValue;                             //--- Set double value
      m_integerValue=(long)m_doubleValue;                    //--- Convert to integer
      m_booleanValue=m_integerValue!=0;                      //--- Set boolean based on integer
   }
   void operator=(const bool booleanValue){
      m_type=JsonBoolean;                                    //--- Set type to boolean
      m_booleanValue=booleanValue;                           //--- Set boolean value
      m_integerValue=(long)m_booleanValue;                   //--- Convert to integer
      m_doubleValue=(double)m_booleanValue;                  //--- Convert to double
   }
   void operator=(string stringValue){
      m_type=(stringValue!=NULL)?JsonString:JsonNull;        //--- Set type to string or null
      m_stringValue=stringValue;                             //--- Set string value
      m_integerValue=StringToInteger(m_stringValue);         //--- Convert to integer
      m_doubleValue=StringToDouble(m_stringValue);           //--- Convert to double
      m_booleanValue=stringValue!=NULL;                      //--- Set boolean based on string presence
   }

   bool operator==(const int integerValue){return m_integerValue==integerValue;}  //--- Compare integer value
   bool operator==(const long longValue){return m_integerValue==longValue;}       //--- Compare long value
   bool operator==(const double doubleValue){return m_doubleValue==doubleValue;}  //--- Compare double value
   bool operator==(const bool booleanValue){return m_booleanValue==booleanValue;} //--- Compare boolean value
   bool operator==(string stringValue){return m_stringValue==stringValue;}        //--- Compare string value

   bool operator!=(const int integerValue){return m_integerValue!=integerValue;}  //--- Check inequality for integer
   bool operator!=(const long longValue){return m_integerValue!=longValue;}       //--- Check inequality for long
   bool operator!=(const double doubleValue){return m_doubleValue!=doubleValue;}  //--- Check inequality for double
   bool operator!=(const bool booleanValue){return m_booleanValue!=booleanValue;} //--- Check inequality for boolean
   bool operator!=(string stringValue){return m_stringValue!=stringValue;}        //--- Check inequality for string

   long ToInteger() const{return m_integerValue;}            //--- Return integer value
   double ToDouble() const{return m_doubleValue;}            //--- Return double value
   bool ToBoolean() const{return m_booleanValue;}            //--- Return boolean value
   string ToString(){return m_stringValue;}                  //--- Return string value

   virtual void SetFromString(JsonValueType type,string stringValue){
      m_type=type;                                           //--- Set the type
      switch(m_type){                                        //--- Handle based on type
      case JsonBoolean:
         m_booleanValue=(StringToInteger(stringValue)!=0);   //--- Convert string to boolean
         m_integerValue=(long)m_booleanValue;                //--- Set integer from boolean
         m_doubleValue=(double)m_booleanValue;               //--- Set double from boolean
         m_stringValue=stringValue;                          //--- Set string value
         break;                                              //--- Exit case
      case JsonInteger:
         m_integerValue=StringToInteger(stringValue);        //--- Convert string to integer
         m_doubleValue=(double)m_integerValue;               //--- Set double from integer
         m_stringValue=stringValue;                          //--- Set string value
         m_booleanValue=m_integerValue!=0;                   //--- Set boolean from integer
         break;                                              //--- Exit case
      case JsonDouble:
         m_doubleValue=StringToDouble(stringValue);          //--- Convert string to double
         m_integerValue=(long)m_doubleValue;                 //--- Set integer from double
         m_stringValue=stringValue;                          //--- Set string value
         m_booleanValue=m_integerValue!=0;                   //--- Set boolean from integer
         break;                                              //--- Exit case
      case JsonString:
         m_stringValue=UnescapeString(stringValue);          //--- Unescape the string
         m_type=(m_stringValue!=NULL)?JsonString:JsonNull;   //--- Set type based on string presence
         m_integerValue=StringToInteger(m_stringValue);      //--- Convert to integer
         m_doubleValue=StringToDouble(m_stringValue);        //--- Convert to double
         m_booleanValue=m_stringValue!=NULL;                 //--- Set boolean based on string
         break;                                              //--- Exit case
      }
   }
   virtual string GetSubstringFromArray(char &jsonCharacterArray[],int startPosition,int substringLength){
      #ifdef __MQL4__
            if(substringLength<=0) return "";                //--- Return empty if length invalid in MQL4
      #endif
      char temporaryArray[];                                 //--- Declare temporary array
      ArrayCopy(temporaryArray,jsonCharacterArray,0,startPosition,substringLength); //--- Copy substring to temporary array
      return CharArrayToString(temporaryArray, 0, WHOLE_ARRAY, JsonValue::encodingCodePage); //--- Convert to string using code page
   }
   virtual void SetValue(const JsonValue &value){
      if(m_type==JsonUndefined) {m_type=JsonObject;}         //--- Set type to object if undefined
      CopyDataFrom(value);                                   //--- Copy data from value
   }
   virtual void SetArrayValues(const JsonValue &list[]);
   virtual JsonValue *AddChild(const JsonValue &item){
      if(m_type==JsonUndefined){m_type=JsonArray;}           //--- Set type to array if undefined
      return AddChildInternal(item);                         //--- Call internal add child
   }
   virtual JsonValue *AddChild(const int integerValue){
      JsonValue item(integerValue);                          //--- Create item from integer
      return AddChild(item);                                 //--- Add the item
   }
   virtual JsonValue *AddChild(const long longValue){
      JsonValue item(longValue);                             //--- Create item from long
      return AddChild(item);                                 //--- Add the item
   }
   virtual JsonValue *AddChild(const double doubleValue){
      JsonValue item(doubleValue);                           //--- Create item from double
      return AddChild(item);                                 //--- Add the item
   }
   virtual JsonValue *AddChild(const bool booleanValue){
      JsonValue item(booleanValue);                          //--- Create item from boolean
      return AddChild(item);                                 //--- Add the item
   }
   virtual JsonValue *AddChild(string stringValue){
      JsonValue item(JsonString,stringValue);                //--- Create item from string
      return AddChild(item);                                 //--- Add the item
   }
   virtual JsonValue *AddChildInternal(const JsonValue &item){
      int currentSize=ArraySize(m_children);                 //--- Get current children size
      ArrayResize(m_children,currentSize+1);                 //--- Resize array to add one more
      m_children[currentSize]=item;                          //--- Add the item
      m_children[currentSize].m_parent=GetPointer(this);     //--- Set parent to current object
      return GetPointer(m_children[currentSize]);            //--- Return pointer to added child
   }
   virtual JsonValue *CreateNewChild(){
      if(m_type==JsonUndefined) {m_type=JsonArray;}          //--- Set type to array if undefined
      return CreateNewChildInternal();                       //--- Call internal create new child
   }
   virtual JsonValue *CreateNewChildInternal(){
      int currentSize=ArraySize(m_children);                 //--- Get current children size
      ArrayResize(m_children,currentSize+1);                 //--- Resize array to add one more
      return GetPointer(m_children[currentSize]);            //--- Return pointer to new child
   }

   virtual string EscapeString(string value);
   virtual string UnescapeString(string value);
```

Still under the public access specifier, we implement the "IsNumericValue" method, which checks if the JSON value is a number by returning true if "m\_type" is "JsonDouble" or "JsonInteger", enabling type-specific handling for numeric data. Then, we develop the "FindChildByKey" [method](https://www.mql5.com/en/docs/basis/function), which iterates backward through the "m\_children" array to locate a child with a matching key, returning a [pointer](https://www.mql5.com/en/docs/basis/types/object_pointers) to it or [NULL](https://www.mql5.com/en/docs/basis/types/void) if not found, facilitating access to nested JSON objects.

Next, we define overloaded assignment operators ("operator=") for "JsonValue", integer, long, double, boolean, and string types, updating "m\_type" and corresponding value fields ("m\_integerValue", "m\_doubleValue", "m\_stringValue", "m\_booleanValue") while ensuring type consistency, such as converting integers to doubles and strings for unified storage. We also implement comparison operators ("operator==" and "operator!=") for integer, long, double, boolean, and string types, allowing direct value comparisons with the respective fields.

Additionally, we provide conversion methods "ToInteger", "ToDouble", "ToBoolean", and "ToString" to retrieve values in their respective formats. The "SetFromString" method sets the JSON value based on the specified "JsonValueType", handling boolean, integer, double, and [string types](https://www.mql5.com/en/book/basis/builtin_types/strings) by converting the input string and updating related fields, with string values unescaped via "UnescapeString". The "GetSubstringFromArray" method extracts a substring from a character array, copying a specified portion and converting it to a string using "CharArrayToString" with the defined "encodingCodePage".

Finally, we implement methods to manage child elements: "AddChild" (for "JsonValue", integer, long, double, boolean, string) adds a new child to "m\_children", setting the parent and type as needed; "CreateNewChild" and "CreateNewChildInternal" create an empty child; and "SetValue" copies data from another "JsonValue", ensuring the class can construct, manipulate, and access JSON structures. Finally, we can conclude the class by including methods to [serialize](https://www.mql5.com/go?link=https://medium.com/@khemanta/data-serialization-and-deserialization-what-is-it-29b5ca7a756f "https://medium.com/@khemanta/data-serialization-and-deserialization-what-is-it-29b5ca7a756f"), deserialize, and encode characters from responses and prompts.

```
public:
   virtual void SerializeToString(string &jsonString,bool includeKey=false,bool includeComma=false);
   virtual string SerializeToString(){
      string jsonString;                                     //--- Declare json string
      SerializeToString(jsonString);                         //--- Call serialize with default params
      return jsonString;                                     //--- Return the serialized string
   }
   virtual bool DeserializeFromArray(char &jsonCharacterArray[],int arrayLength,int &currentIndex);
   virtual bool ExtractStringFromArray(char &jsonCharacterArray[],int arrayLength,int &currentIndex);
   virtual bool DeserializeFromString(string jsonString,int encoding=CP_ACP){
      int currentIndex=0;                                    //--- Initialize current index
      Reset();                                               //--- Reset the object
      JsonValue::encodingCodePage=encoding;                  //--- Set encoding code page
      char characterArray[];                                 //--- Declare character array
      int arrayLength=StringToCharArray(jsonString,characterArray,0,WHOLE_ARRAY,JsonValue::encodingCodePage); //--- Convert string to char array
      return DeserializeFromArray(characterArray,arrayLength,currentIndex); //--- Call deserialize from array
   }
   virtual bool DeserializeFromArray(char &jsonCharacterArray[],int encoding=CP_ACP){
      int currentIndex=0;                                    //--- Initialize current index
      Reset();                                               //--- Reset the object
      JsonValue::encodingCodePage=encoding;                  //--- Set encoding code page
      return DeserializeFromArray(jsonCharacterArray,ArraySize(jsonCharacterArray),currentIndex); //--- Call deserialize with size
   }
```

Here, we implement the "SerializeToString" method with parameters for a string reference, "includeKey", and "includeComma", which converts the JSON structure into a string; it adds a comma if "includeComma" is true, includes the key if "includeKey" is true, and handles different types ("JsonNull" as "null", "JsonBoolean" as "true" or "false", "JsonInteger" and "JsonDouble" via string conversion, "JsonString" with escaped values, "JsonArray" with bracketed child elements, and "JsonObject" with keyed child elements), recursively serializing children for nested structures. A convenience overload of "SerializeToString" creates a string, calls the parameterized version, and returns the result.

Next, we implement "DeserializeFromString", which takes a JSON string and encoding (default [CP\_ACP](https://www.mql5.com/en/docs/constants/io_constants/codepageusage)), resets the object, sets "encodingCodePage", converts the string to a character array using [StringToCharArray](https://www.mql5.com/en/docs/convert/stringtochararray), and delegates to "DeserializeFromArray". The "DeserializeFromArray" overload with encoding initializes the index, resets the object, sets "encodingCodePage", and calls the main "DeserializeFromArray" method, which parses a character array by iterating through characters, handling whitespace, arrays ("\[" to "\]"), objects ("{" to "}"), booleans ("true" or "false"), null ("null"), numbers (detecting integers or doubles via valid characters), and strings (with escape handling), creating child elements as needed and managing hierarchy with "m\_parent" and "m\_temporaryKey".

Finally, "ExtractStringFromArray" parses strings within the array, handling escaped characters (e.g., "\\n", "\\t", Unicode "\\uXXXX") until a closing quote, just in case we decide to use [Unicode](https://en.wikipedia.org/wiki/Unicode "https://en.wikipedia.org/wiki/Unicode") characters like emojis, ensuring accurate string extraction. Here is how they look like.

![UNICODE SAMPLE](https://c.mql5.com/2/169/Screenshot_2025-09-14_140522.png)

We can now define this methods so they do their intended work accurately.

```
int JsonValue::encodingCodePage=CP_ACP;                      //--- Initialize static code page
//+------------------------------------------------------------------+
//| Checks if child with specific key and optional type exists       |
//+------------------------------------------------------------------+
JsonValue *JsonValue::HasChildWithKey(string key,JsonValueType type){
   for(int index=0; index<ArraySize(m_children); index++) if(m_children[index].m_key==key){ //--- Loop through children
      if(type==JsonUndefined || type==m_children[index].m_type){ //--- Check type condition
         return GetPointer(m_children[index]);               //--- Return matching child
      }
      break;                                                 //--- Exit loop
   }
   return NULL;                                              //--- Return NULL if no match
}
//+------------------------------------------------------------------+
//| Accessor for object key, creates if not exists                   |
//+------------------------------------------------------------------+
JsonValue *JsonValue::operator[](string key){
   if(m_type==JsonUndefined){m_type=JsonObject;}             //--- Set type to object if undefined
   JsonValue *value=FindChildByKey(key);                     //--- Find child by key
   if(value){return value;}                                  //--- Return if found
   JsonValue newValue(GetPointer(this),JsonUndefined);       //--- Create new undefined value
   newValue.m_key=key;                                       //--- Set key for new value
   value=AddChild(newValue);                                 //--- Add new value as child
   return value;                                             //--- Return the new value
}
//+------------------------------------------------------------------+
//| Accessor for array index, expands if necessary                   |
//+------------------------------------------------------------------+
JsonValue *JsonValue::operator[](int index){
   if(m_type==JsonUndefined) m_type=JsonArray;               //--- Set type to array if undefined
   while(index>=ArraySize(m_children)){                      //--- Loop to expand array if needed
      JsonValue newElement(GetPointer(this),JsonUndefined);  //--- Create new undefined element
      if(CheckPointer(AddChild(newElement))==POINTER_INVALID){return NULL;} //--- Add and check pointer
   }
   return GetPointer(m_children[index]);                     //--- Return pointer to element at index
}
//+------------------------------------------------------------------+
//| Sets array values from list                                      |
//+------------------------------------------------------------------+
void JsonValue::SetArrayValues(const JsonValue &list[]){
   if(m_type==JsonUndefined){m_type=JsonArray;}              //--- Set type to array if undefined
   int numChildren=ArrayResize(m_children,ArraySize(list));  //--- Resize children to list size
   for(int index=0; index<numChildren; ++index){             //--- Loop through list
      m_children[index]=list[index];                         //--- Copy from list
      m_children[index].m_parent=GetPointer(this);           //--- Set parent to current
   }
}
```

To define the class methods outside the body, we use the [scope resolution operator](https://www.mql5.com/en/docs/basis/operations/other). We could define them inside, but this way makes the code modular. We initialize the static member "encodingCodePage" to [CP\_ACP](https://www.mql5.com/en/docs/constants/io_constants/codepageusage) (Active Code Page) to define the default character encoding for string conversions, ensuring compatibility with JSON data from AI APIs. Then, we implement the "HasChildWithKey" method, which iterates forward through the "m\_children" array to find a child with a matching key and, optionally, a specific "JsonValueType" (defaulting to "JsonUndefined" to ignore type), returning a pointer to the matching child or [NULL](https://www.mql5.com/en/docs/basis/types/void) if not found, improving efficiency over "FindChildByKey" by breaking early and allowing type-specific checks.

Next, we develop the [overloaded](https://www.mql5.com/en/docs/basis/oop/overload) "operator" method, which sets "m\_type" to "JsonObject" if undefined, searches for a child with the given key using "FindChildByKey", and, if not found, creates a new "JsonValue" with the key and adds it to "m\_children" via "AddChild", returning a [pointer](https://www.mql5.com/en/docs/basis/types/object_pointers) to the existing or new child for seamless object access. Similarly, the "operator" method sets "m\_type" to "JsonArray" if undefined, expands the "m\_children" array with undefined elements if the index exceeds its size using "AddChild", and returns a pointer to the element at the specified index, ensuring safe array access.

Last, we implement "SetArrayValues", which sets "m\_type" to "JsonArray" if undefined, resizes "m\_children" to match the input list size, copies each "JsonValue" from the list to "m\_children", and sets the parent of each child to the current object, enabling bulk assignment of array values. For the other methods, we will use a similar format. Let us start with methods to serialize and deserialize strings.

```
//+------------------------------------------------------------------+
//| Serializes the JSON value to string                              |
//+------------------------------------------------------------------+
void JsonValue::SerializeToString(string &jsonString,bool includeKey,bool includeComma){
   if(m_type==JsonUndefined){return;}                        //--- Return if undefined
   if(includeComma){jsonString+=",";}                        //--- Add comma if needed
   if(includeKey){jsonString+=StringFormat("\"%s\":", m_key);} //--- Add key if needed
   int numChildren=ArraySize(m_children);                    //--- Get number of children
   switch(m_type){                                           //--- Handle based on type
   case JsonNull:
      jsonString+="null";                                    //--- Append null
      break;                                                 //--- Exit case
   case JsonBoolean:
      jsonString+=(m_booleanValue?"true":"false");           //--- Append true or false
      break;                                                 //--- Exit case
   case JsonInteger:
      jsonString+=IntegerToString(m_integerValue);           //--- Append integer as string
      break;                                                 //--- Exit case
   case JsonDouble:
      jsonString+=DoubleToString(m_doubleValue);             //--- Append double as string
      break;                                                 //--- Exit case
   case JsonString:
   {
      string value=EscapeString(m_stringValue);              //--- Escape the string
      if(StringLen(value)>0){jsonString+=StringFormat("\"%s\"",value);} //--- Append escaped string if not empty
      else{jsonString+="null";}                              //--- Append null if empty
   }
   break;                                                    //--- Exit case
   case JsonArray:
      jsonString+="[";                                       //--- Start array\
      for(int index=0; index<numChildren; index++){m_children[index].SerializeToString(jsonString,false,index>0);} //--- Serialize each child\
      jsonString+="]";                                       //--- End array
      break;                                                 //--- Exit case
   case JsonObject:
      jsonString+="{";                                       //--- Start object
      for(int index=0; index<numChildren; index++){m_children[index].SerializeToString(jsonString,true,index>0);} //--- Serialize each child with key
      jsonString+="}";                                       //--- End object
      break;                                                 //--- Exit case
   }
}
//+------------------------------------------------------------------+
//| Deserializes from character array                                |
//+------------------------------------------------------------------+
bool JsonValue::DeserializeFromArray(char &jsonCharacterArray[],int arrayLength,int &currentIndex){
   string validNumericCharacters="0123456789+-.eE";          //--- Define valid number characters
   int startPosition=currentIndex;                           //--- Set start position
   for(; currentIndex<arrayLength; currentIndex++){          //--- Loop through array
      char currentCharacter=jsonCharacterArray[currentIndex]; //--- Get current character
      if(currentCharacter==0){break;}                        //--- Break if null character
      switch(currentCharacter){                              //--- Handle based on character
      case '\t':
      case '\r':
      case '\n':
      case ' ':                                              //--- Skip whitespace
         startPosition=currentIndex+1;                       //--- Update start position
         break;                                              //--- Exit case
      case '[':                                              //--- Start of array\
      {\
         startPosition=currentIndex+1;                       //--- Update start position\
         if(m_type!=JsonUndefined){                          //--- Check if type is undefined\
            if(DEBUG_PRINT){Print(m_key+" "+string(__LINE__));} //--- Print debug if error\
            return false;                                    //--- Return false on error\
         }\
         m_type=JsonArray;                                   //--- Set type to array\
         currentIndex++;                                     //--- Increment index\
         JsonValue childValue(GetPointer(this),JsonUndefined); //--- Create child value\
         while(childValue.DeserializeFromArray(jsonCharacterArray,arrayLength,currentIndex)){ //--- Deserialize children\
            if(childValue.m_type!=JsonUndefined){AddChild(childValue);} //--- Add if not undefined\
            if(childValue.m_type==JsonInteger || childValue.m_type==JsonDouble || childValue.m_type==JsonArray){currentIndex++;} //--- Increment if certain types\
            childValue.Reset();                              //--- Reset child\
            childValue.m_parent=GetPointer(this);            //--- Set parent\
            if(jsonCharacterArray[currentIndex]==']'){break;} //--- Break if end of array
            currentIndex++;                                  //--- Increment index
            if(currentIndex>=arrayLength){                   //--- Check bounds
               if(DEBUG_PRINT){Print(m_key+" "+string(__LINE__));} //--- Print debug
               return false;                                 //--- Return false
            }
         }
         return (jsonCharacterArray[currentIndex]==']' || jsonCharacterArray[currentIndex]==0); //--- Return true if properly ended
      }
      break;                                                 //--- Exit case
      case ']':                                              //--- End of array
         if(!m_parent){return false;}                        //--- Return false if no parent
         return (m_parent.m_type==JsonArray);                //--- Check parent is array
      case ':':                                              //--- Key-value separator
      {
         if(m_temporaryKey==""){                             //--- Check temporary key
            if(DEBUG_PRINT){Print(m_key+" "+string(__LINE__));} //--- Print debug
            return false;                                    //--- Return false
         }
         JsonValue childValue(GetPointer(this),JsonUndefined); //--- Create child
         JsonValue *addedChild=AddChild(childValue);           //--- Add child
         addedChild.m_key=m_temporaryKey;                      //--- Set key
         m_temporaryKey="";                                    //--- Clear temporary key
         currentIndex++;                                       //--- Increment index
         if(!addedChild.DeserializeFromArray(jsonCharacterArray,arrayLength,currentIndex)){ //--- Deserialize child
            if(DEBUG_PRINT){Print(m_key+" "+string(__LINE__));} //--- Print debug
            return false;                                    //--- Return false
         }
         break;                                              //--- Exit case
      }
      case ',':                                              //--- Value separator
         startPosition=currentIndex+1;                       //--- Update start
         if(!m_parent && m_type!=JsonObject){                //--- Check conditions
            if(DEBUG_PRINT){Print(m_key+" "+string(__LINE__));} //--- Print debug
            return false;                                    //--- Return false
         }
         else if(m_parent){                                  //--- If has parent
            if(m_parent.m_type!=JsonArray && m_parent.m_type!=JsonObject){ //--- Check parent type
               if(DEBUG_PRINT){Print(m_key+" "+string(__LINE__));} //--- Print debug
               return false;                                //--- Return false
            }
            if(m_parent.m_type==JsonArray && m_type==JsonUndefined){return true;} //--- Return true for undefined in array
         }
         break;                                              //--- Exit case
      case '{':                                              //--- Start of object
         startPosition=currentIndex+1;                       //--- Update start
         if(m_type!=JsonUndefined){                          //--- Check type
            if(DEBUG_PRINT){Print(m_key+" "+string(__LINE__));} //--- Print debug
            return false;                                    //--- Return false
         }
         m_type=JsonObject;                                  //--- Set type to object
         currentIndex++;                                     //--- Increment index
         if(!DeserializeFromArray(jsonCharacterArray,arrayLength,currentIndex)){ //--- Recurse deserialize
            if(DEBUG_PRINT){Print(m_key+" "+string(__LINE__));} //--- Print debug
            return false;                                    //--- Return false
         }
         return (jsonCharacterArray[currentIndex]=='}' || jsonCharacterArray[currentIndex]==0); //--- Check end
         break;                                              //--- Exit case
      case '}':                                              //--- End of object
         return (m_type==JsonObject);                        //--- Check type is object
      case 't':
      case 'T':                                              //--- Start of true
      case 'f':
      case 'F':                                              //--- Start of false
         if(m_type!=JsonUndefined){                          //--- Check type
            if(DEBUG_PRINT){Print(m_key+" "+string(__LINE__));} //--- Print debug
            return false;                                    //--- Return false
         }
         m_type=JsonBoolean;                                 //--- Set type to boolean
         if(currentIndex+3<arrayLength){                     //--- Check for true
            if(StringCompare(GetSubstringFromArray(jsonCharacterArray, currentIndex, 4), "true", false)==0){ //--- Compare substring
               m_booleanValue=true;                          //--- Set to true
               currentIndex+=3;                              //--- Advance index
               return true;                                  //--- Return true
            }
         }
         if(currentIndex+4<arrayLength){                     //--- Check for false
            if(StringCompare(GetSubstringFromArray(jsonCharacterArray, currentIndex, 5), "false", false)==0){ //--- Compare substring
               m_booleanValue=false;                         //--- Set to false
               currentIndex+=4;                              //--- Advance index
               return true;                                  //--- Return true
            }
         }
         if(DEBUG_PRINT){Print(m_key+" "+string(__LINE__));} //--- Print debug
         return false;                                       //--- Return false
         break;                                              //--- Exit case
      case 'n':
      case 'N':                                              //--- Start of null
         if(m_type!=JsonUndefined){                          //--- Check type
            if(DEBUG_PRINT){Print(m_key+" "+string(__LINE__));} //--- Print debug
            return false;                                    //--- Return false
         }
         m_type=JsonNull;                                    //--- Set type to null
         if(currentIndex+3<arrayLength){                     //--- Check bounds
            if(StringCompare(GetSubstringFromArray(jsonCharacterArray,currentIndex,4),"null",false)==0){ //--- Compare substring
               currentIndex+=3;                              //--- Advance index
               return true;                                  //--- Return true
            }
         }
         if(DEBUG_PRINT){Print(m_key+" "+string(__LINE__));} //--- Print debug
         return false;                                       //--- Return false
         break;                                              //--- Exit case
      case '0':
      case '1':
      case '2':
      case '3':
      case '4':
      case '5':
      case '6':
      case '7':
      case '8':
      case '9':
      case '-':
      case '+':
      case '.':                                              //--- Start of number
      {
         if(m_type!=JsonUndefined){                          //--- Check type
            if(DEBUG_PRINT){Print(m_key+" "+string(__LINE__));} //--- Print debug
            return false;                                    //--- Return false
         }
         bool isDouble=false;                                //--- Initialize double flag
         int startOfNumber=currentIndex;                     //--- Set start of number
         while(jsonCharacterArray[currentIndex]!=0 && currentIndex<arrayLength){ //--- Loop to parse number
            currentIndex++;                                  //--- Increment index
            if(StringFind(validNumericCharacters,GetSubstringFromArray(jsonCharacterArray,currentIndex,1))<0){break;} //--- Break if invalid char
            if(!isDouble){isDouble=(jsonCharacterArray[currentIndex]=='.' || jsonCharacterArray[currentIndex]=='e' || jsonCharacterArray[currentIndex]=='E');} //--- Set double flag
         }
         m_stringValue=GetSubstringFromArray(jsonCharacterArray,startOfNumber,currentIndex-startOfNumber); //--- Get number string
         if(isDouble){                                       //--- If double
            m_type=JsonDouble;                               //--- Set type to double
            m_doubleValue=StringToDouble(m_stringValue);     //--- Convert to double
            m_integerValue=(long)m_doubleValue;              //--- Convert to integer
            m_booleanValue=m_integerValue!=0;                //--- Set boolean
         }
         else{                                               //--- Else integer
            m_type=JsonInteger;                              //--- Set type to integer
            m_integerValue=StringToInteger(m_stringValue);   //--- Convert to integer
            m_doubleValue=(double)m_integerValue;            //--- Convert to double
            m_booleanValue=m_integerValue!=0;                //--- Set boolean
         }
         currentIndex--;                                     //--- Decrement index
         return true;                                        //--- Return true
         break;                                              //--- Exit case
      }
      case '\"':                                             //--- Start of string or key
         if(m_type==JsonObject){                             //--- If object type
            currentIndex++;                                  //--- Increment index
            int startOfString=currentIndex;                  //--- Set start of string
            if(!ExtractStringFromArray(jsonCharacterArray,arrayLength,currentIndex)){ //--- Extract string
               if(DEBUG_PRINT){Print(m_key+" "+string(__LINE__));} //--- Print debug
               return false;                                 //--- Return false
            }
            m_temporaryKey=GetSubstringFromArray(jsonCharacterArray,startOfString,currentIndex-startOfString); //--- Set temporary key
         }
         else{                                               //--- Else value string
            if(m_type!=JsonUndefined){                       //--- Check type
               if(DEBUG_PRINT){Print(m_key+" "+string(__LINE__));} //--- Print debug
               return false;                                 //--- Return false
            }
            m_type=JsonString;                               //--- Set type to string
            currentIndex++;                                  //--- Increment index
            int startOfString=currentIndex;                  //--- Set start of string
            if(!ExtractStringFromArray(jsonCharacterArray,arrayLength,currentIndex)){ //--- Extract string
               if(DEBUG_PRINT){Print(m_key+" "+string(__LINE__));} //--- Print debug
               return false;                                 //--- Return false
            }
            SetFromString(JsonString,GetSubstringFromArray(jsonCharacterArray,startOfString,currentIndex-startOfString)); //--- Set from extracted string
            return true;                                     //--- Return true
         }
         break;                                              //--- Exit case
      }
   }
   return true;
      //--- Return true at end
}
```

We continue the implementation of the parsing framework, focusing on the critical [serialization and deserialization](https://www.mql5.com/go?link=https://medium.com/@khemanta/data-serialization-and-deserialization-what-is-it-29b5ca7a756f "https://medium.com/@khemanta/data-serialization-and-deserialization-what-is-it-29b5ca7a756f") methods.

First, we define the "SerializeToString" method, which converts a JSON structure into a string; it skips undefined types, adds a comma if "includeComma" is true, appends the key if "includeKey" is true using [StringFormat](https://www.mql5.com/en/docs/convert/stringformat), and handles each "m\_type" case: "JsonNull" appends "null", "JsonBoolean" appends "true" or "false" based on "m\_booleanValue", "JsonInteger" uses [IntegerToString](https://www.mql5.com/en/docs/convert/IntegerToString) for "m\_integerValue", "JsonDouble" uses [DoubleToString](https://www.mql5.com/en/docs/convert/doubletostring) for "m\_doubleValue", "JsonString" escapes "m\_stringValue" with "EscapeString" and wraps it in quotes (or "null" if empty), "JsonArray" wraps children in brackets with recursive calls to "SerializeToString" (without keys, adding commas for non-first elements), and "JsonObject" wraps children in braces with keys included. A convenience overload of "SerializeToString" creates a string, calls the parameterized version with default arguments, and returns the result.

Next, we implement "DeserializeFromArray", which parses a character array to construct a JSON structure, using a string of valid numeric characters ("0123456789+-.eE") and iterating through the array; it skips whitespace, handles arrays ("\[") by setting "m\_type" to "JsonArray" and recursively deserializing children until a closing bracket, validates array closure, processes key-value separators (":") by assigning "m\_temporaryKey" to a new child, handles value separators (","), processes objects ("{") by setting "m\_type" to "JsonObject" and deserializing children until a closing brace, validates object closure, parses booleans ("true" or "false") by setting "m\_booleanValue", parses null ("null") by setting "m\_type" to "JsonNull", parses numbers by detecting decimal or exponential notation to set "m\_type" to "JsonDouble" or "JsonInteger", and parses strings (quoted) by using "ExtractStringFromArray" to handle escaped characters.\
\
The "DeserializeFromArray" method ensures proper parent-child relationships and error handling via "DEBUG\_PRINT" logs for invalid structures, making it robust for parsing responses. Finally, we can define the escape methods.\
\
```\
//+------------------------------------------------------------------+\
//| Extracts string from array handling escapes                      |\
//+------------------------------------------------------------------+\
bool JsonValue::ExtractStringFromArray(char &jsonCharacterArray[],int arrayLength,int &currentIndex){\
   for(; jsonCharacterArray[currentIndex]!=0 && currentIndex<arrayLength; currentIndex++){ //--- Loop through string\
      char currentCharacter=jsonCharacterArray[currentIndex]; //--- Get current char\
      if(currentCharacter=='\"') break;                      //--- Break on closing quote\
      if(currentCharacter=='\\' && currentIndex+1<arrayLength){ //--- Handle escape\
         currentIndex++;                                     //--- Increment for escaped char\
         currentCharacter=jsonCharacterArray[currentIndex];  //--- Get escaped char\
         switch(currentCharacter){                           //--- Handle escaped type\
         case '/':\
         case '\\':\
         case '\"':\
         case 'b':\
         case 'f':\
         case 'r':\
         case 'n':\
         case 't':\
            break;                                           //--- Allowed escapes\
         case 'u':                                           //--- Unicode escape\
         {\
            currentIndex++;                                  //--- Increment\
            for(int hexDigitIndex=0; hexDigitIndex<4 && currentIndex<arrayLength && jsonCharacterArray[currentIndex]!=0; hexDigitIndex++,currentIndex++){ //--- Loop hex digits\
               if(!((jsonCharacterArray[currentIndex]>='0' && jsonCharacterArray[currentIndex]<='9') || (jsonCharacterArray[currentIndex]>='A' && jsonCharacterArray[currentIndex]<='F') || (jsonCharacterArray[currentIndex]>='a' && jsonCharacterArray[currentIndex]<='f'))){ //--- Check hex\
                  if(DEBUG_PRINT){Print(m_key+" "+CharToString(jsonCharacterArray[currentIndex])+" "+string(__LINE__));} //--- Print debug\
                  return false;                                 //--- Return false on invalid hex\
               }\
            }\
            currentIndex--;                                  //--- Decrement after loop\
            break;                                           //--- Exit case\
         }\
         default:\
            break;                                           //--- Handle other (commented return false)\
         }\
      }\
   }\
   return true;                                              //--- Return true\
}\
//+------------------------------------------------------------------+\
//| Escapes special characters in string                             |\
//+------------------------------------------------------------------+\
string JsonValue::EscapeString(string stringValue){\
   ushort inputCharacters[], escapedCharacters[];               //--- Declare arrays\
   int inputLength=StringToShortArray(stringValue, inputCharacters); //--- Convert string to short array\
   if(ArrayResize(escapedCharacters, 2*inputLength)!=2*inputLength){return NULL;} //--- Resize escaped array, return NULL on fail\
   int escapedIndex=0;                                          //--- Initialize escaped index\
   for(int inputIndex=0; inputIndex<inputLength; inputIndex++){ //--- Loop through input\
      switch(inputCharacters[inputIndex]){                      //--- Handle special chars\
      case '\\':\
         escapedCharacters[escapedIndex]='\\';                  //--- Add escape\
         escapedIndex++;                                        //--- Increment\
         escapedCharacters[escapedIndex]='\\';                  //--- Add backslash\
         escapedIndex++;                                        //--- Increment\
         break;                                                 //--- Exit case\
      case '"':\
         escapedCharacters[escapedIndex]='\\';                  //--- Add escape\
         escapedIndex++;                                        //--- Increment\
         escapedCharacters[escapedIndex]='"';                   //--- Add quote\
         escapedIndex++;                                        //--- Increment\
         break;                                                 //--- Exit case\
      case '/':\
         escapedCharacters[escapedIndex]='\\';                  //--- Add escape\
         escapedIndex++;                                        //--- Increment\
         escapedCharacters[escapedIndex]='/';                   //--- Add slash\
         escapedIndex++;                                        //--- Increment\
         break;                                                 //--- Exit case\
      case 8:\
         escapedCharacters[escapedIndex]='\\';                  //--- Add escape\
         escapedIndex++;                                        //--- Increment\
         escapedCharacters[escapedIndex]='b';                   //--- Add backspace\
         escapedIndex++;                                        //--- Increment\
         break;                                                 //--- Exit case\
      case 12:\
         escapedCharacters[escapedIndex]='\\';                  //--- Add escape\
         escapedIndex++;                                        //--- Increment\
         escapedCharacters[escapedIndex]='f';                   //--- Add form feed\
         escapedIndex++;                                        //--- Increment\
         break;                                                 //--- Exit case\
      case '\n':\
         escapedCharacters[escapedIndex]='\\';                  //--- Add escape\
         escapedIndex++;                                        //--- Increment\
         escapedCharacters[escapedIndex]='n';                   //--- Add newline\
         escapedIndex++;                                        //--- Increment\
         break;                                                 //--- Exit case\
      case '\r':\
         escapedCharacters[escapedIndex]='\\';                  //--- Add escape\
         escapedIndex++;                                        //--- Increment\
         escapedCharacters[escapedIndex]='r';                   //--- Add carriage return\
         escapedIndex++;                                        //--- Increment\
         break;                                                 //--- Exit case\
      case '\t':\
         escapedCharacters[escapedIndex]='\\';                  //--- Add escape\
         escapedIndex++;                                        //--- Increment\
         escapedCharacters[escapedIndex]='t';                   //--- Add tab\
         escapedIndex++;                                        //--- Increment\
         break;                                                 //--- Exit case\
      default:\
         escapedCharacters[escapedIndex]=inputCharacters[inputIndex]; //--- Copy normal char\
         escapedIndex++;                                        //--- Increment\
         break;                                                 //--- Exit case\
      }\
   }\
   stringValue=ShortArrayToString(escapedCharacters,0,escapedIndex); //--- Convert back to string\
   return stringValue;                                          //--- Return escaped string\
}\
//+------------------------------------------------------------------+\
//| Unescapes special characters in string                           |\
//+------------------------------------------------------------------+\
string JsonValue::UnescapeString(string stringValue){\
   ushort inputCharacters[], unescapedCharacters[];             //--- Declare arrays\
   int inputLength=StringToShortArray(stringValue, inputCharacters); //--- Convert to short array\
   if(ArrayResize(unescapedCharacters, inputLength)!=inputLength){return NULL;} //--- Resize, return NULL on fail\
   int outputIndex=0,inputIndex=0;                              //--- Initialize indices\
   while(inputIndex<inputLength){                               //--- Loop through input\
      ushort currentCharacter=inputCharacters[inputIndex];      //--- Get current char\
      if(currentCharacter=='\\' && inputIndex<inputLength-1){   //--- Handle escape\
         switch(inputCharacters[inputIndex+1]){                 //--- Handle escaped type\
         case '\\':\
            currentCharacter='\\';                              //--- Set to backslash\
            inputIndex++;                                       //--- Increment\
            break;                                              //--- Exit case\
         case '"':\
            currentCharacter='"';                               //--- Set to quote\
            inputIndex++;                                       //--- Increment\
            break;                                              //--- Exit case\
         case '/':\
            currentCharacter='/';                               //--- Set to slash\
            inputIndex++;                                       //--- Increment\
            break;                                              //--- Exit case\
         case 'b':\
            currentCharacter=8;                                 //--- Set to backspace\
            inputIndex++;                                       //--- Increment\
            break;                                              //--- Exit case\
         case 'f':\
            currentCharacter=12;                                //--- Set to form feed\
            inputIndex++;                                       //--- Increment\
            break;                                              //--- Exit case\
         case 'n':\
            currentCharacter='\n';                              //--- Set to newline\
            inputIndex++;                                       //--- Increment\
            break;                                              //--- Exit case\
         case 'r':\
            currentCharacter='\r';                              //--- Set to carriage return\
            inputIndex++;                                       //--- Increment\
            break;                                              //--- Exit case\
         case 't':\
            currentCharacter='\t';                              //--- Set to tab\
            inputIndex++;                                       //--- Increment\
            break;                                              //--- Exit case\
         }\
      }\
      unescapedCharacters[outputIndex]=currentCharacter;        //--- Copy to output\
      outputIndex++;                                            //--- Increment output\
      inputIndex++;                                             //--- Increment input\
   }\
   stringValue=ShortArrayToString(unescapedCharacters,0,outputIndex); //--- Convert back to string\
   return stringValue;                                          //--- Return unescaped string\
}\
```\
\
We finalize the implementation by focusing on the string handling methods. First, we implement the "ExtractStringFromArray" method, which parses a string from a character array by iterating until a closing quote or array end, handling escape sequences (e.g., "", """, "/", "\\b", "\\f", "\\n", "\\r", "\\t") and [Unicode](https://en.wikipedia.org/wiki/Unicode "https://en.wikipedia.org/wiki/Unicode") escapes ("\\uXXXX") by validating four [hexadecimal](https://en.wikipedia.org/wiki/Hexadecimal "https://en.wikipedia.org/wiki/Hexadecimal") digits, returning false on invalid hex or bounds errors, and using "DEBUG\_PRINT" for debugging invalid cases, ensuring accurate string extraction for JSON parsing.\
\
Then, we develop the "EscapeString" method, which converts a string to its JSON-compliant form by transforming it into a short array with [StringToShortArray](https://www.mql5.com/en/docs/convert/stringtoshortarray), resizing an output array to twice the input length to accommodate escapes, and iterating through each character to handle special cases: backslash, quote, slash, backspace (8), form feed (12), newline, carriage return, and tab are escaped with a preceding backslash (e.g., "\\n" for [newline](https://en.wikipedia.org/wiki/Newline "https://en.wikipedia.org/wiki/Newline")), while other characters are copied directly, returning the escaped string via the [ShortArrayToString](https://www.mql5.com/en/docs/convert/shortarraytostring) function.\
\
Last, we implement the "UnescapeString" method, which reverses the escaping process by converting the input string to a short array, resizing an output array to the input length, and iterating to process escape sequences: when a backslash is encountered, the next character is interpreted (e.g., "\\n" to newline, "\\t" to tab), copying the unescaped character to the output, while non-escaped characters are copied directly, returning the unescaped string via "ShortArrayToString" or [NULL](https://www.mql5.com/en/docs/basis/types/void) on resize failure. These methods ensure the "JsonValue" class can handle special characters in JSON strings, critical for correctly formatting API requests and parsing AI responses in a robust and reliable manner. Here is an example of the usage of special characters that will be interpreted correctly for easier communication and seamless understanding.\
\
![EXAMPLE USAGE OF SPECIAL CHARACTERS](https://c.mql5.com/2/169/Screenshot_2025-09-14_143752.png)\
\
With the class implementation done, we are now all okay to start using the parsing logic. But let us first test the parser and make sure everything is okay. We will do that in the next section below.\
\
### Testing the JSON Parser\
\
To ensure the reliability of our JSON parsing framework, we rigorously test the "JsonValue" [class](https://www.mql5.com/en/docs/basis/types/classes) in MQL5 to verify its ability to handle various JSON structures critical for AI API integration. Below, we outline the testing approach, including test cases, expected outcomes, and methods to validate the parser’s functionality in a trading environment. We will be defining a code function and calling it in the [OnInit](https://www.mql5.com/en/docs/event_handlers/oninit) event handler, printing the contents. Here is the first code. We will use the same file. Let's first print "Hello world", as in the case of a start-up.\
\
```\
//+------------------------------------------------------------------+\
//| Test Functions                                                   |\
//+------------------------------------------------------------------+\
void TestBasicSerialization(){\
   Print("\n--- Testing Basic Serialization ---");\
\
   JsonValue root;\
   root["string"] = "hello world";\
   root["number"] = 42;\
   root["double"] = 3.14159;\
   root["boolean"] = true;\
   root["empty"] = "";\
\
   string json = root.SerializeToString();\
   Print("Serialized JSON: ", json);\
\
   JsonValue parsed;\
   if(parsed.DeserializeFromString(json)){\
      Print("Deserialization successful");\
      Print("String: ", parsed["string"].ToString());\
      Print("Number: ", (int)parsed["number"].ToInteger());\
      Print("Double: ", parsed["double"].ToDouble());\
      Print("Boolean: ", parsed["boolean"].ToBoolean());\
      Print("Empty: ", parsed["empty"].ToString());\
   } else {\
      Print("Deserialization failed!");\
   }\
}\
\
//+------------------------------------------------------------------+\
//| Expert initialization function                                   |\
//+------------------------------------------------------------------+\
int OnInit(){\
   TestBasicSerialization();\
return(INIT_SUCCEEDED);\
}\
```\
\
We create a "TestBasicSerialization" function and create a "JsonValue" object named "root" and assign values using "operator\[\]": "string" to "hello world", "number" to 42, "double" to 3.14159, "boolean" to true, and "empty" to an empty string, covering key JSON data types (JsonString, JsonInteger, JsonDouble, JsonBoolean). Then, we call "SerializeToString" on "root" to generate a JSON string, store it in "json", and print it to the [MetaTrader 5](https://www.metatrader5.com/ "https://www.metatrader5.com/") terminal with [Print](https://www.mql5.com/en/docs/common/print) for verification.\
\
Next, we create a new "JsonValue" object named "parsed" and call "DeserializeFromString" with the "json" string to reconstruct the JSON structure, checking if it returns true to confirm success; if successful, we print "Deserialization successful" and use "ToString", "ToInteger" (cast to int), "ToDouble", and "ToBoolean" to retrieve and print the values of "string", "number", "double", "boolean", and "empty" via "operator\[\]"; if it fails, we print "Deserialization failed!". Last, in [OnInit](https://www.mql5.com/en/docs/event_handlers/oninit), we call "TestBasicSerialization" to execute the test upon program initialization and return "INIT\_SUCCEEDED" to indicate successful setup. Here is the output we get.\
\
![INITIAL BASIC SERIALIZATION](https://c.mql5.com/2/169/Screenshot_2025-09-14_150108.png)\
\
From the image, we can see basic serialization is successful. Let us now test basic deserialization and see.\
\
```\
void TestBasicDeserialization(){\
   Print("\n--- Testing Basic Deserialization ---");\
\
   string testJson = "{\"name\":\"John\",\"age\":30,\"isStudent\":false,\"salary\":1500.75}";\
\
   JsonValue parsed;\
   if(parsed.DeserializeFromString(testJson)){\
      Print("Parsed successfully:");\
      Print("Name: ", parsed["name"].ToString());\
      Print("Age: ", (int)parsed["age"].ToInteger());\
      Print("Is Student: ", parsed["isStudent"].ToBoolean());\
      Print("Salary: ", parsed["salary"].ToDouble());\
   } else {\
      Print("Failed to parse JSON");\
   }\
}\
```\
\
On testing, we get the following outcome.\
\
![INITIAL BASIC DESERIALIZATION](https://c.mql5.com/2/169/Screenshot_2025-09-14_151608.png)\
\
That was a success too. Let us now increase the complexity gauge and see.\
\
```\
void TestComplexObject(){\
   Print("\n--- Testing Complex Object ---");\
\
   JsonValue root;\
   root["person"]["name"] = "Alice";\
   root["person"]["age"] = 25;\
   root["person"]["isActive"] = true;\
   root["person"]["score"] = 95.5;\
   root["person"]["address"]["street"] = "123 Main St";\
   root["person"]["address"]["city"] = "New York";\
   root["person"]["address"]["zipcode"] = "10001";\
\
   root["person"]["hobbies"].AddChild("reading");\
   root["person"]["hobbies"].AddChild("gaming");\
   root["person"]["hobbies"].AddChild("coding");\
\
   root["person"]["preferences"]["theme"] = "dark";\
   root["person"]["preferences"]["notifications"] = true;\
\
   string json = root.SerializeToString();\
   Print("Complex JSON: ", json);\
\
   JsonValue parsed;\
   if(parsed.DeserializeFromString(json)){\
      Print("Round-trip successful");\
      Print("Name: ", parsed["person"]["name"].ToString());\
      Print("Age: ", (int)parsed["person"]["age"].ToInteger());\
      Print("City: ", parsed["person"]["address"]["city"].ToString());\
      Print("Zipcode: ", parsed["person"]["address"]["zipcode"].ToString());\
      Print("Theme: ", parsed["person"]["preferences"]["theme"].ToString());\
      Print("Hobby count: ", ArraySize(parsed["person"]["hobbies"].m_children));\
   }\
}\
\
void TestArrayHandling(){\
   Print("\n--- Testing Array Handling ---");\
\
   JsonValue root;\
\
   // Array of numbers\
   for(int i = 0; i < 5; i++){\
      root["numbers"].AddChild(i * 10);\
   }\
\
   // Array of mixed types\
   root["mixed"].AddChild("string");\
   root["mixed"].AddChild(123);\
   root["mixed"].AddChild(45.67);\
   root["mixed"].AddChild(true);\
\
   // Array of objects\
   JsonValue item;\
   item["id"] = 1;\
   item["name"] = "Item 1";\
   item["price"] = 19.99;\
   root["items"].AddChild(item);\
\
   item["id"] = 2;\
   item["name"] = "Item 2";\
   item["price"] = 29.99;\
   root["items"].AddChild(item);\
\
   // Nested arrays\
   JsonValue nestedArray;\
   nestedArray.AddChild("nested1");\
   nestedArray.AddChild("nested2");\
   root["nested"].AddChild(nestedArray);\
\
   string json = root.SerializeToString();\
   Print("Array JSON: ", json);\
\
   JsonValue parsed;\
   if(parsed.DeserializeFromString(json)){\
      Print("Numbers array length: ", ArraySize(parsed["numbers"].m_children));\
      Print("Mixed array length: ", ArraySize(parsed["mixed"].m_children));\
      Print("Items array length: ", ArraySize(parsed["items"].m_children));\
      Print("First item name: ", parsed["items"][0]["name"].ToString());\
      Print("Second item price: ", parsed["items"][1]["price"].ToDouble());\
   }\
}\
\
void TestErrorHandling(){\
   Print("\n--- Testing Error Handling ---");\
\
   JsonValue parsed;\
\
   // Test invalid JSON\
   string invalidJson = "{\"name\": \"test\", \"age\": }";\
   if(!parsed.DeserializeFromString(invalidJson)){\
      Print("✓ Correctly rejected invalid JSON");\
   }\
\
   // Test malformed JSON\
   string malformedJson = "{\"name\": \"test\" \"age\": 30}"; // Missing comma\
   if(!parsed.DeserializeFromString(malformedJson)){\
      Print("✓ Correctly rejected malformed JSON");\
   }\
\
   // Test empty string\
   if(!parsed.DeserializeFromString("")){\
      Print("✓ Correctly handled empty string");\
   }\
\
   // Test incomplete object\
   string incompleteJson = "{\"name\": \"test\"";\
   if(!parsed.DeserializeFromString(incompleteJson)){\
      Print("✓ Correctly rejected incomplete JSON");\
   }\
}\
```\
\
We test and we get the following outcome.\
\
![COMPLEX TESTS](https://c.mql5.com/2/169/Screenshot_2025-09-14_152728.png)\
\
Let us now test performance, nested structures, and data types.\
\
```\
void TestPerformance(){\
   Print("\n--- Testing Performance ---");\
\
   int startTime = GetTickCount();\
   int iterations = 100;\
   int successCount = 0;\
\
   for(int i = 0; i < iterations; i++){\
      JsonValue root;\
      root["test_id"] = i;\
      root["name"] = "Test Item " + IntegerToString(i);\
      root["value"] = i * 1.5;\
      root["active"] = (i % 2 == 0);\
\
      // Add array\
      for(int j = 0; j < 5; j++){\
         root["tags"].AddChild("tag" + IntegerToString(j));\
      }\
\
      string json = root.SerializeToString();\
\
      JsonValue parsed;\
      if(parsed.DeserializeFromString(json)){\
         successCount++;\
      }\
   }\
\
   int endTime = GetTickCount();\
   Print("Performance: ", iterations, " iterations in ", endTime - startTime, "ms");\
   Print("Success rate: ", successCount, "/", iterations, " (", DoubleToString(successCount*100.0/iterations, 1), "%)");\
}\
\
void TestNestedStructures(){\
   Print("\n--- Testing Nested Structures ---");\
\
   JsonValue root;\
\
   // Deep nesting\
   root["level1"]["level2"]["level3"]["level4"]["value"] = "deep_nested";\
   root["level1"]["level2"]["level3"]["level4"]["number"] = 999;\
\
   // Array of objects with nesting\
   JsonValue user;\
   user["name"] = "John";\
   user["profile"]["age"] = 30;\
   user["profile"]["settings"]["theme"] = "dark";\
   user["profile"]["settings"]["notifications"] = true;\
   root["users"].AddChild(user);\
\
   user["name"] = "Jane";\
   user["profile"]["age"] = 25;\
   user["profile"]["settings"]["theme"] = "light";\
   user["profile"]["settings"]["notifications"] = false;\
   root["users"].AddChild(user);\
\
   string json = root.SerializeToString();\
   Print("Nested JSON: ", json);\
\
   JsonValue parsed;\
   if(parsed.DeserializeFromString(json)){\
      Print("✓ Nested structures parsed successfully");\
      Print("Deep value: ", parsed["level1"]["level2"]["level3"]["level4"]["value"].ToString());\
      Print("User count: ", ArraySize(parsed["users"].m_children));\
      Print("First user theme: ", parsed["users"][0]["profile"]["settings"]["theme"].ToString());\
      Print("Second user age: ", (int)parsed["users"][1]["profile"]["age"].ToInteger());\
   }\
}\
\
void TestDataTypes(){\
   Print("\n--- Testing Data Types ---");\
\
   JsonValue root;\
\
   // All supported data types\
   root["string_type"] = "hello world";\
   root["int_type"] = 42;\
   root["long_type"] = 1234567890123;\
   root["double_type"] = 3.14159265358979;\
   root["bool_true"] = true;\
   root["bool_false"] = false;\
   root["empty_string"] = "";\
\
   // Scientific notation\
   root["scientific_positive"] = 1.23e+10;\
   root["scientific_negative"] = 1.23e-10;\
\
   string json = root.SerializeToString();\
   Print("Data Types JSON: ", json);\
\
   JsonValue parsed;\
   if(parsed.DeserializeFromString(json)){\
      Print("✓ All data types parsed successfully");\
      Print("String: ", parsed["string_type"].ToString());\
      Print("Integer: ", (int)parsed["int_type"].ToInteger());\
      Print("Long: ", parsed["long_type"].ToInteger());\
      Print("Double: ", parsed["double_type"].ToDouble());\
      Print("Bool True: ", parsed["bool_true"].ToBoolean());\
      Print("Bool False: ", parsed["bool_false"].ToBoolean());\
      Print("Scientific Positive: ", parsed["scientific_positive"].ToDouble());\
      Print("Scientific Negative: ", parsed["scientific_negative"].ToDouble());\
   }\
}\
```\
\
Here is the output.\
\
![PERFORMANCE, NESTED STRUCTURES & DATA TYPE TEST](https://c.mql5.com/2/169/Screenshot_2025-09-14_153802.png)\
\
That was a success. Let us test the escape characters now.\
\
```\
void TestEscapeCharacters(){\
   Print("\n--- Testing Escape Characters ---");\
\
   JsonValue root;\
\
   // Various escape sequences - using only MQL5 supported escapes\
   root["backslash"] = "\\\\"; // Double backslash for single backslash\
   root["quote"] = "\\\"";     // Escaped quote\
   root["newline"] = "\\n";    // Escaped newline\
   root["tab"] = "\\t";        // Escaped tab\
   root["carriage_return"] = "\\r"; // Escaped carriage return\
\
   // For form feed and backspace, we need to use their actual ASCII codes\
   // since MQL5 doesn't support \f and \b escape sequences\
   root["form_feed"] = "\\u000C"; // Unicode escape for form feed\
   root["backspace"] = "\\u0008"; // Unicode escape for backspace\
\
   root["mixed_escapes"] = "Line1\\nLine2\\tTabbed\\\"Quoted\\\"\\\\Backslash";\
\
   string json = root.SerializeToString();\
   Print("Escape Chars JSON: ", json);\
\
   JsonValue parsed;\
   if(parsed.DeserializeFromString(json)){\
      Print("✓ Escape characters parsed successfully");\
      Print("Backslash: '", parsed["backslash"].ToString(), "'");\
      Print("Quote: '", parsed["quote"].ToString(), "'");\
      Print("Newline: '", parsed["newline"].ToString(), "'");\
      Print("Tab: '", parsed["tab"].ToString(), "'");\
      Print("Carriage Return: '", parsed["carriage_return"].ToString(), "'");\
      Print("Form Feed: '", parsed["form_feed"].ToString(), "'");\
      Print("Backspace: '", parsed["backspace"].ToString(), "'");\
      Print("Mixed: '", parsed["mixed_escapes"].ToString(), "'");\
   } else {\
      Print("✗ Escape characters parsing failed");\
   }\
}\
```\
\
We get the following outcome.\
\
![ESCAPE CHARACTERS](https://c.mql5.com/2/169/Screenshot_2025-09-14_154757.png)\
\
That was a success. Since we have confirmed that our JSON implementation is usable, we can now use it to create AI integrations in future programs that we will be creating.\
\
### Conclusion\
\
In conclusion, we’ve developed a JSON (JavaScript Object Notation) parsing framework in MQL5, implementing a "JsonValue" class to handle serialization and deserialization of JSON data critical for API (Application Programming Interface) interactions. Through methods like "SerializeToString", "DeserializeFromArray", and "EscapeString", and testing via functions like "TestBasicSerialization", we ensure reliable processing of diverse JSON structures, laying a solid foundation for future AI-driven trading systems. In the subsequent parts, we will be integrating and interacting with the AIs in our trading applications. Keep tuned.\
\
**Attached files** \|\
\
\
[Download ZIP](https://www.mql5.com/en/articles/download/19562.zip "Download all attachments in the single ZIP archive")\
\
[a.\_JSON\_Code\_File.mq5](https://www.mql5.com/en/articles/download/19562/a._JSON_Code_File.mq5 "Download a._JSON_Code_File.mq5")(103.2 KB)\
\
**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.\
\
This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.\
\
#### Other articles by this author\
\
- [MQL5 Trading Tools (Part 12): Enhancing the Correlation Matrix Dashboard with Interactivity](https://www.mql5.com/en/articles/20962)\
- [Creating Custom Indicators in MQL5 (Part 5): WaveTrend Crossover Evolution Using Canvas for Fog Gradients, Signal Bubbles, and Risk Management](https://www.mql5.com/en/articles/20815)\
- [MQL5 Trading Tools (Part 11): Correlation Matrix Dashboard (Pearson, Spearman, Kendall) with Heatmap and Standard Modes](https://www.mql5.com/en/articles/20945)\
- [Creating Custom Indicators in MQL5 (Part 4): Smart WaveTrend Crossover with Dual Oscillators](https://www.mql5.com/en/articles/20811)\
- [Building AI-Powered Trading Systems in MQL5 (Part 8): UI Polish with Animations, Timing Metrics, and Response Management Tools](https://www.mql5.com/en/articles/20722)\
- [Creating Custom Indicators in MQL5 (Part 3): Multi-Gauge Enhancements with Sector and Round Styles](https://www.mql5.com/en/articles/20719)\
- [Creating Custom Indicators in MQL5 (Part 2): Building a Gauge-Style RSI Display with Canvas and Needle Mechanics](https://www.mql5.com/en/articles/20632)\
\
**Last comments \|**\
**[Go to discussion](https://www.mql5.com/en/forum/495719)**\
(2)\
\
\
![Peter Wlliams](https://c.mql5.com/avatar/2022/11/636D8A83-906E.png)\
\
**[Peter Wlliams](https://www.mql5.com/en/users/peterwlliams)**\
\|\
23 Sep 2025 at 23:44\
\
**MetaQuotes:**\
\
Check out the new article: [Building AI-Powered Trading Systems in MQL5 (Part 1): Implementing JSON Handling for AI APIs](https://www.mql5.com/en/articles/19562).\
\
Author: [Allan Munene Mutiiria](https://www.mql5.com/en/users/29210372 "29210372")\
\
Looks fantastic. Where to start?\
\
Is the download file a.\_JSON\_Code\_File.mq5 and EA?\
\
I renamed A\_JSON\_Code\_File.mq5 (looked more familiar), then compiled.  No errorss\
\
However I didn't see any reference as an EXPERT on my platform - i.e. I couldn't load & run.\
\
Just need to 'play' and try and understand what happens & why.\
\
Many thanks for sharing what appears to be a fantastic opportinity to utilise AI\
\
![Allan Munene Mutiiria](https://c.mql5.com/avatar/2022/11/637df59b-9551.jpg)\
\
**[Allan Munene Mutiiria](https://www.mql5.com/en/users/29210372)**\
\|\
24 Sep 2025 at 11:44\
\
**Peter Wlliams [#](https://www.mql5.com/en/forum/495719#comment_58104282):**\
\
Looks fantastic. Where to start?\
\
Is the download file a.\_JSON\_Code\_File.mq5 and EA?\
\
I renamed A\_JSON\_Code\_File.mq5 (looked more familiar), then compiled.  No errorss\
\
However I didn't see any reference as an EXPERT on my platform - i.e. I couldn't load & run.\
\
Just need to 'play' and try and understand what happens & why.\
\
Many thanks for sharing what appears to be a fantastic opportinity to utilise AI\
\
Thanks for the feedback. Seems like you really need to read the article. It's about JSON to enable AI integration in future versions.\
\
\
![From Novice to Expert: Animated News Headline Using MQL5 (XI)—Correlation in News Trading](https://c.mql5.com/2/170/19343-from-novice-to-expert-animated-logo.png)[From Novice to Expert: Animated News Headline Using MQL5 (XI)—Correlation in News Trading](https://www.mql5.com/en/articles/19343)\
\
In this discussion, we will explore how the concept of Financial Correlation can be applied to improve decision-making efficiency when trading multiple symbols during major economic events announcement. The focus is on addressing the challenge of heightened risk exposure caused by increased volatility during news releases.\
\
![Introduction to MQL5 (Part 21): Automating Harmonic Pattern Detection](https://c.mql5.com/2/170/19331-introduction-to-mql5-part-21-logo.png)[Introduction to MQL5 (Part 21): Automating Harmonic Pattern Detection](https://www.mql5.com/en/articles/19331)\
\
Learn how to detect and display the Gartley harmonic pattern in MetaTrader 5 using MQL5. This article explains each step of the process, from identifying swing points to applying Fibonacci ratios and plotting the full pattern on the chart for clear visual confirmation.\
\
![Functions for activating neurons during training: The key to fast convergence?](https://c.mql5.com/2/112/Functions_of_neuronal_activation_during_learning___LOGO.png)[Functions for activating neurons during training: The key to fast convergence?](https://www.mql5.com/en/articles/16845)\
\
This article presents a study of the interaction of different activation functions with optimization algorithms in the context of neural network training. Particular attention is paid to the comparison of the classical ADAM and its population version when working with a wide range of activation functions, including the oscillating ACON and Snake functions. Using a minimalistic MLP (1-1-1) architecture and a single training example, the influence of activation functions on the optimization is isolated from other factors. The article proposes an approach to manage network weights through the boundaries of activation functions and a weight reflection mechanism, which allows avoiding problems with saturation and stagnation in training.\
\
![Pipelines in MQL5](https://c.mql5.com/2/169/19544-pipelines-in-mql5-logo.png)[Pipelines in MQL5](https://www.mql5.com/en/articles/19544)\
\
In this piece, we look at a key data preparation step for machine learning that is gaining rapid significance. Data Preprocessing Pipelines. These in essence are a streamlined sequence of data transformation steps that prepare raw data before it is fed to a model. As uninteresting as this may initially seem to the uninducted, this ‘data standardization’ not only saves on training time and execution costs, but it goes a long way in ensuring better generalization. In this article we are focusing on some SCIKIT-LEARN preprocessing functions, and while we are not exploiting the MQL5 Wizard, we will return to it in coming articles.\
\
[Best articles and CodeBase updates in MQL5.community channelsFollow us to ensure you never miss out on important updates![](https://www.mql5.com/ff/sh/n9yf51p2srwzfqh5z2/02.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=dgazvhktsxqakdvarucjbvmvzenwlyje&s=98a038fe082e458df8c4a1d8e116e3a6646fd5517f06e48b2356b7ee005817d6&uid=&ref=https://www.mql5.com/en/articles/19562&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5048828039271391091)\
\
This website uses cookies. Learn more about our [Cookies Policy](https://www.mql5.com/en/about/cookies).\
\
![close](https://c.mql5.com/i/close.png)\
\
![MQL5 - Language of trade strategies built-in the MetaTrader 5 client terminal](https://c.mql5.com/i/registerlandings/logo-2.png)\
\
You are missing trading opportunities:\
\
- Free trading apps\
- Over 8,000 signals for copying\
- Economic news for exploring financial markets\
\
RegistrationLog in\
\
latin characters without spaces\
\
a password will be sent to this email\
\
An error occurred\
\
\
- [Log in With Google](https://www.mql5.com/en/auth_oauth2?provider=Google&amp;return=popup&amp;reg=1)\
\
You agree to [website policy](https://www.mql5.com/en/about/privacy) and [terms of use](https://www.mql5.com/en/about/terms)\
\
If you do not have an account, please [register](https://www.mql5.com/en/auth_register)\
\
Allow the use of cookies to log in to the MQL5.com website.\
\
Please enable the necessary setting in your browser, otherwise you will not be able to log in.\
\
[Forgot your login/password?](https://www.mql5.com/en/auth_forgotten?return=popup)\
\
- [Log in With Google](https://www.mql5.com/en/auth_oauth2?provider=Google&amp;return=popup)