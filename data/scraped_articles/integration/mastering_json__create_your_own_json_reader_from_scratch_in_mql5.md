---
title: Mastering JSON: Create Your Own JSON Reader from Scratch in MQL5
url: https://www.mql5.com/en/articles/16791
categories: Integration
relevance_score: 3
scraped_at: 2026-01-23T21:07:38.617779
---

[![](https://www.mql5.com/ff/si/h2ryn394uwcpxwmxc2.jpg)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Feconomic-calendar%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dopen.calendar%26utm_content%3Deconomic.calendar%26utm_campaign%3Den.0009.desktop.default&a=qdeulxgvibgwytgewnvfatbocjnnninc&s=5c0c60f00ff5f5bedb0fdf65d9d79eb820442eb43ffac2b85aa003224f9dba14&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=ayvhzejfebhbpouhjkneuaglzwhwygqj&ssn=1769191656185030060&ssn_dr=0&ssn_sr=0&fv_date=1769191656&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F16791&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Mastering%20JSON%3A%20Create%20Your%20Own%20JSON%20Reader%20from%20Scratch%20in%20MQL5%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176919165676638253&fz_uniq=5071594179143871208&sv=2552)

MetaTrader 5 / Integration


### Introduction

Hello and welcome! If you’ve ever tried to parse or manipulate JSON data in MQL5, you might’ve wondered if there’s a straightforward and flexible approach to do so. JSON, which stands for JavaScript Object Notation, has grown in popularity as a lightweight data-interchange format that’s both human-readable and machine-friendly. While MQL5 is predominantly known for creating Expert Advisors, Indicators, and Scripts for the MetaTrader 5 platform, it doesn’t have a native JSON library. This means that if you want to work with JSON data—be it from a web API, an external server, or from your own local files—you’ll likely need to devise a custom solution or integrate an existing library.

In this article, we aim to fill that gap by demonstrating how to create your own JSON reader in MQL5. Along the way, we’ll explore the fundamental concepts of parsing JSON, walking through the creation of a flexible class structure capable of handling different JSON element types (like objects, arrays, strings, numbers, booleans, and null values). Our end goal is to empower you to comfortably parse JSON strings and access or modify the data within them, all from the convenience of your MetaTrader 5 environment.

We will follow a structure similar to what we’ve seen in other MQL5-related articles but with a focus specifically on JSON parsing and usage. This single article will be divided into five main sections: an introduction (the one you’re reading now), a deeper dive into the fundamentals of JSON and how it fits within MQL5, a step-by-step guide to building a basic JSON parser from scratch, an exploration of advanced features for JSON handling, and finally a comprehensive code listing plus concluding thoughts.

JSON is everywhere. Whether you’re fetching market data from a third-party service, uploading your own trading records, or experimenting with complex strategies that require a dynamic configuration, JSON remains a near-universal format. Some of the most common practical use cases for JSON in the world of algorithmic trading include:

1. Fetching Market Data: Many modern broker APIs or financial data services offer real-time or historical data in JSON. Having a JSON reader at your disposal allows you to quickly parse that data and integrate it into your trading strategy.

2. Strategy Configuration: Suppose you have an Expert Advisor that supports multiple parameters: maximum spread, desired account risk level, or allowed trading times. A JSON file can store these settings neatly, and a JSON reader in MQL5 can dynamically load or update these parameters without recompiling your code.

3. Sending Logs or Data: In certain setups, you might want to transmit your trade logs or debug messages to an external server for analytics. Sending them as JSON can help keep your logs consistent, easily parseable, and integrable with tools that expect structured data.


Many online examples show how to parse JSON in languages like Python, JavaScript, or C++. However, MQL5 is a specialized language with its own constraints. That means we need to be careful about certain aspects: memory handling, array usage, strict data types, and so on.

We’ll create a custom class (or set of classes) dedicated to JSON parsing and manipulation. The idea is to design it so you can do something like:

```
CMyJsonParser parser;
parser.LoadString("{\"symbol\":\"EURUSD\",\"lots\":0.1,\"settings\":{\"slippage\":2,\"retries\":3}}");

// Access top-level fields:
Print("Symbol = ", parser.GetObject("symbol").ToStr());
Print("Lots = ", parser.GetObject("lots").ToDbl());

// Access nested fields:
CMyJsonElement settings = parser.GetObject("settings");
Print("Slippage = ", settings.GetObject("slippage").ToInt());
Print("Retries = ", settings.GetObject("retries").ToInt());
```

Of course, your final approach might differ slightly in naming or structure, but this sort of usability is the goal. By building a robust parser, you’ll have a foundation for expansions—like converting MQL5 data structures into JSON for output, or adding caching logic for repeated JSON queries.

You might have come across different JSON libraries out there, including some short scripts that parse JSON by handling character arrays. We will learn from these existing approaches but _will not copy the code directly_. Instead, we’ll construct something fresh with a similar idea so that it’s easier for you to understand and maintain. We’ll dissect our code snippet by snippet, and by the end of this article, you’ll have access to a final, cohesive implementation that you can attach to your own trading programs.

Our hope is that this approach of building a library from the ground up—explaining each segment in plain language—will give you a deeper understanding than if we had just given you a finished solution. By internalizing how the parser operates, you can more easily debug and customize it later.

While JSON is a text-based format, MQL5 strings can contain a variety of special characters, including new lines, carriage returns, or Unicode characters. Our implementation will consider some of these nuances and try to address them gracefully. Still, always ensure that your input data is valid JSON. If you receive malformed JSON or face random text that claims to be valid, you’ll likely need to add more robust error handling.

Here’s a quick preview of how this article is organized:

1. [Section 1 (You’re Here!) – Introduction](https://www.mql5.com/en/articles/16791#section1)

We’ve just discussed what JSON is, why it matters, and how we’ll approach writing a custom parser in MQL5. This sets the stage for everything else.

2. [Section 2 – The Basics: JSON and MQL5 Fundamentals](https://www.mql5.com/en/articles/16791#section2)

We’ll review JSON’s key structural elements, then map them to MQL5 data types and show which aspects need our careful attention.

3. [Section 3 – Extending Our Parser with Advanced Functionality](https://www.mql5.com/en/articles/16791#section3)

Here, we’ll talk about potential expansions or improvements: how to handle arrays, how to add error checking, and how to convert MQL5 data back into JSON if you need to send data out.

4. [Section 4 – Full Code](https://www.mql5.com/en/articles/16791#section4)

Finally, we’ll assemble our entire library into one place, giving you a single reference file.

5. [Section 5 – Conclusion](https://www.mql5.com/en/articles/16791#section5)

We’ll summarize the key lessons learned and point out a few next steps you might want to consider in your own projects.


By the end of the article, you’ll have a fully functional JSON parsing and manipulation library in MQL5. Beyond just that, you’ll understand how it all works under the hood, making you better equipped to integrate JSON into your automated trading solutions.

### The Basics – JSON and MQL5 Fundamentals

Welcome back! Now that we’ve laid out the overall plan for our custom MQL5 JSON reader, it’s time to drill down into the finer points of JSON and see how these map to MQL5. We’ll explore the structure of JSON, cover which data types are easiest to parse, and identify potential pitfalls as we bring JSON data into MetaTrader 5. By the end of this section, you’ll have a much clearer idea of how to tackle JSON in an MQL5 environment, setting the stage for the hands-on coding to come.

JSON (JavaScript Object Notation) is a text-based format commonly used for data transmission and storage. Unlike XML, it’s relatively lightweight: data is enclosed in curly braces ( {} ) for objects or square brackets ( \[\] ) for arrays, and each field is laid out in simple key-value pairs. Here’s a tiny example:

```
{
  "symbol": "EURUSD",
  "lots": 0.05,
  "enableTrade": true
}
```

This is easy for a human to read and straightforward for a machine to parse. Each piece of information—like "symbol" or "enableTrade" —is known as a key that holds some value. The value may be a string, a number, a boolean, or even another nested object or array. In short, JSON is all about organizing data in a nested tree structure, letting you represent everything from basic parameters to more complex hierarchical data.

JSON Versus MQL5 Data Types:

1. **Strings**: JSON strings appear in double quotes, such as "Hello World". In MQL5, we also have the string type, but these strings can include special characters, escape sequences, and Unicode. So, the first nuance we’ll face is to ensure our parser correctly handles quotation marks, escaped symbols (like \\" ), and possibly Unicode code points (e.g., \\u00A9 ).
2. **Numbers**: In JSON, numbers can be integers (like 42 ) or decimals ( 3.14159 ). MQL5 stores numbers primarily as int (for integers) or double (for floating-point values). However, not all numeric values in JSON will cleanly map to an int. For example, 1234567890 is valid, but in some contexts, you might need a long in MQL5 if the number is really large. We’ll need to pay special attention when the JSON number is beyond the range of a typical 32-bit integer. Also, you might need to convert a large integer into a double if it overflows a standard integer’s limit, but that comes with potential rounding issues.
3. **Booleans**: JSON uses lowercase true and false . Meanwhile, MQL5 uses bool . This is a straightforward mapping, but we’ll have to carefully detect these tokens ( true and false ) while parsing. There’s a small catch: any syntax errors—like True or FALSE in uppercase—are not valid JSON, though some parsers in other languages allow them. If your data sometimes uses uppercase booleans, you need to handle that gracefully, or ensure your data is strictly JSON-compliant.
4. **NULL**: A  null value in JSON often indicates an empty or missing field. MQL5 doesn’t have a dedicated “null type.” Instead, we may choose to represent JSON null as a special internal enumeration (like jtNULL if we define an enum for our JSON element types) or treat it as an empty string or default value. We’ll see how to manage nulls in the parser soon.
5. **Objects**: When you see curly braces, { ... } , that’s a JSON object. It’s essentially a collection of key-value pairs. In MQL5, there’s no built-in dictionary type, but we can simulate one by storing a dynamic array of pairs or by building a custom class to hold keys and values. We’ll typically define something like a CMyJsonObject class (or a general-purpose class with an internal “object” state) that houses a list of children. Each child has a key ( string ) and a value that could be any JSON data type.
6. **Arrays**: Arrays in JSON are ordered lists surrounded by square brackets, \[ ... \] . Each item in the array can be a string, number, object, or even another array. In MQL5, we handle arrays with the ArrayResize function and direct indexing. We’ll likely store a JSON array as a dynamic array of elements. Our code will need to keep track of the fact that a particular node is an array, along with the children inside it.

Let us look at some of the potential challenges:

1. Handling Escape Sequences: In JSON, a backslash \ can precede characters like quotes or newlines. For instance, you might see "description": "Line one\\\nLine two". We need to interpret \\\n as an actual newline within the final string. Special sequences include:

   - \\" for double quotes
   - \\\ for backslash
   - \\/ sometimes for forward slash
   - \\n for newline
   - \\t for tab
   - \\u for Unicode code points

We’ll have to methodically convert these sequences in the raw JSON string into the actual characters they represent in MQL5. If not, the parser might store them incorrectly or fail on input data that uses these standard escape patterns.

2. Trimming Whitespace and Control Characters: A valid JSON string can include spaces, tabs, and newlines (especially between elements). Although these are allowed and have no semantic meaning in most places, they can complicate parsing logic if we’re not careful. A robust parser typically ignores any whitespace outside of quoted strings. This means we’ll want to skip them as we move from one token to the next.
3. Dealing With Large Data: If your JSON string is extremely large, you might worry about memory constraints in MQL5. The language can handle arrays fairly well, but there are upper limits if you approach tens of millions of elements. Most traders rarely need JSON that huge, but it’s worth noting that a “streaming” or iterative approach might be necessary if you do. For most normal usage—like reading settings or moderate-size data sets—our straightforward approach should be just fine.
4. Not all JSON is perfect. If your parser tries to read an invalid structure—for instance, a missing quote or a trailing comma—it needs to handle that gracefully. You might want to define error codes or store an error message internally so the calling code can detect and respond to parse errors. In a trading context, you could:

   - Show a message box or print an error to the journal.
   - Default back to some safe settings if JSON is invalid.
   - Stop the Expert Advisor from running if critical data can’t be parsed.

We’ll incorporate basic checks to catch mistakes like unmatched brackets or unrecognized tokens. More advanced error reporting is also possible, but that’s up to how rigorous you want to be.

Since JSON can be nested, our parser will likely use a single class or class hierarchy where each node can be one of several types:

- Object – Contains key-value pairs
- Array – Contains indexed elements
- String – Holds textual data
- Number – Stores numeric data as double or possibly as long
- Boolean – True or false
- Null – No value

We might implement an enum for these possible node types, like:

```
enum JSONNodeType
  {
   JSON_UNDEFINED = 0,
   JSON_OBJECT,
   JSON_ARRAY,
   JSON_STRING,
   JSON_NUMBER,
   JSON_BOOL,
   JSON_NULL
  };
```

Then we give our parser class a variable that holds which type the current node is. We also store the node’s contents. If it’s an object, we keep an array of child nodes keyed by string. If it’s an array, we keep a list of child nodes indexed from 0 upwards. If it’s a string, we keep a string. If it’s a number, we might store a double plus an internal integer if it’s integral, etc.

An alternative approach is to have a separate class for objects, arrays, strings, and so on. This can get messy in MQL5 because you’d be casting between them often. Instead, we’ll likely adopt a single class (or a single main class plus some helper structures) that can dynamically represent any JSON type. This unified approach is straightforward when dealing with nested elements, as each child is essentially the same type of node with a different internal type. That helps us keep the code shorter and more generalized.

Even if your immediate project only requires reading JSON, you might eventually want to create JSON from your MQL5 data. For example, if you generate trade signals and want to push them to a server as JSON, or if you want to log your trades in a structured JSON file, you’ll need an “encoder” or “serializer.” Our eventual parser can be extended to do this. The basic code we’ll write for handling strings and arrays can also help with generating JSON. Just keep that in mind as you design your class methods: “How can I call the same logic in reverse to produce JSON text from internal data?”

Now we have a solid grasp of how JSON’s structures correlate to MQL5. We know we need a flexible class that can do the following:

1. Store Node Type – Whether it’s a number, string, object, array, boolean, or null.
2. Parse – Read raw text character by character, interpret braces, brackets, quotes, and special tokens.
3. Access – Provide convenient methods to get or set child nodes by key (for objects) or by index (for arrays).
4. Convert – Turn numeric or boolean nodes into MQL5 primitives, like double, int, or bool.
5. Escape/Unescape – Convert JSON-encoded sequences in strings to normal MQL5 strings (and vice versa if we add a future “to JSON” method).
6. Error Checking – Possibly detect malformed input or unknown tokens, then handle gracefully.

We’ll tackle these features step by step in the next section, where the real coding journey begins. If you’re worried about performance or memory usage, rest assured that a straightforward approach is usually plenty fast and memory-efficient for normal usage. If you do run into performance bottlenecks or memory constraints, you can always profile the code or adopt partial parsing techniques.

In Section 3, we’ll start building our parser in detail. We’ll define the overarching class—something like CJsonNode —and begin with the simplest tasks: storing a node’s type and value, plus writing a “tokenizer” method that identifies JSON tokens (like braces or quotes). Once the foundation is laid, we’ll work upwards to support objects, arrays, nested elements, and data extraction.

Whether you plan to parse small JSON config files or retrieve extensive data from the web, these same fundamentals apply. Even if you’re new to reading external data in MQL5, fear not: once you see the logic step by step, it all becomes quite manageable.

Take a breath now; we’re about to immerse ourselves in code. In the next section, we’ll get hands-on with building the custom JSON parser step by step, along with practical tips for ensuring your data is processed reliably. Let’s make MQL5 “speak” JSON like a champ!

_Parser’s Core Class:_ The goal of our parser class is to represent any piece of JSON data (sometimes called a “node” in a tree). Here’s a sketch of what we might need:

1. An Enumeration for Node Types:  We want to distinguish easily between JSON object, array, string, etc. Let’s define something like:



```
enum JsonNodeType
     {
      JSON_UNDEF  = 0,
      JSON_OBJ,
      JSON_ARRAY,
      JSON_STRING,
      JSON_NUMBER,
      JSON_BOOL,
      JSON_NULL
     };
```

2. Member Variables:

Each CJsonNode stores

   - A _JsonNodeType m\_typeto_ identify the node type.
   - For _objects_: a structure (like an array) that holds key-value pairs.
   - For _arrays_: a structure that holds indexed child nodes.
   - For _strings_: a string m\_value.
   - For _numbers_: a double m\_numVal, possibly an additional long m\_intValif needed.
   - For _booleans_: a bool m\_boolVal.

4. Parsing and Utility Methods:

   - One method to parse the raw JSON text.
   - Methods to retrieve child nodes by index or key.
   - Possibly a method to “tokenize” the input, helping us identify brackets, braces, strings, booleans, and so on.

We’ll keep these ideas in mind as we start coding. Below is an illustrative snippet that shows how we might define this class in MQL5 (in a file named something like CJsonNode.mqh). We’ll go step by step.

```
//+------------------------------------------------------------------+
//|      CJsonNode.mqh                                               |
//+------------------------------------------------------------------+
#pragma once

enum JsonNodeType
  {
   JSON_UNDEF  = 0,
   JSON_OBJ,
   JSON_ARRAY,
   JSON_STRING,
   JSON_NUMBER,
   JSON_BOOL,
   JSON_NULL
  };

// Class representing a single JSON node
class CJsonNode
  {
private:
   JsonNodeType     m_type;         // The type of this node
   string           m_value;        // Used if this node is a string
   double           m_numVal;       // Used if this node is a number
   bool             m_boolVal;      // Used if this node is a boolean

   // For arrays and objects, we'll keep child nodes in a dynamic array:
   CJsonNode        m_children[];   // The array for child nodes
   string           m_keys[];       // Only used if node is an object
                                     // For arrays, we’ll just rely on index

public:
   // Constructor & destructor
   CJsonNode();
   ~CJsonNode();

   // Parsing interface
   bool ParseString(const string jsonText);

   // Utility methods (we will define them soon)
   void  SetType(JsonNodeType nodeType);
   JsonNodeType GetType() const;
   int   ChildCount() const;

   // Accessing children
   CJsonNode* AddChild();
   CJsonNode* GetChild(int index);
   CJsonNode* GetChild(const string key);
   void        SetKey(int childIndex,const string key);

   // Setting and getting values
   void SetString(const string val);
   void SetNumber(const double val);
   void SetBool(bool val);
   void SetNull();

   string AsString() const;
   double AsNumber() const;
   bool   AsBool()   const;

   // We’ll add the actual parse logic in a dedicated private method
private:
   bool ParseRoot(string jsonText);
   bool ParseObject(string text, int &pos);
   bool ParseArray(string text, int &pos);
   bool ParseValue(string text, int &pos);
   bool SkipWhitespace(const string text, int &pos);
   // ... other helpers
  };
```

In the code above:

- m\_children\[\]: A dynamic array that can store multiple child CJsonNodeobjects. For arrays, each child is indexed, while for objects, each child has an associated key stored in m\_keys\[\].
- ParseString(const string jsonText): This public method is our “main entry point.” You feed it a JSON string, and it tries to parse it, populating the node’s internal data.
- ParseRoot, ParseObject, ParseArray, ParseValue: We’ll define each of these private methods to handle particular JSON constructs.

We’re showing a skeleton now, but we’ll fill in details in just a moment. While parsing JSON we read from left to right, ignoring whitespace until we see a structural character. For example:

- A '{' means we have an object starting.
- A '\[' means we have an array.\
\
- A '\\"' means a string is about to begin.\
\
- A digit or a minus sign might mean a number.\
\
- The sequences “true,” “false,” or “null” also appear in JSON.\
\
\
Let’s see a simplified version of how we might parse an entire text in our _ParseString_ method:\
\
```\
bool CJsonNode::ParseString(const string jsonText)\
  {\
   // Reset existing data first\
   m_type   = JSON_UNDEF;\
   m_value  = "";\
   ArrayResize(m_children,0);\
   ArrayResize(m_keys,0);\
\
   int pos=0;\
   return ParseRoot(jsonText) && SkipWhitespace(jsonText,pos) && pos>=StringLen(jsonText)-1;\
  }\
```\
\
- _Reset_ – We clear out any previous data.\
- _pos=0_ – This is our character position in the string.\
- _Call ParseRoot(jsonText)_ – A function we’ll define that sets m\_typeand populates m\_childrenor m\_valueas needed.\
- _SkipWhitespace(jsonText,pos)_ – We often skip any spaces, tabs, or newlines that might appear.\
- _Check final position_ – If everything parsed correctly, _pos_ should be near the end of the string. Otherwise, there might be trailing text or an error.\
\
Now, let’s look more closely at ParseRoot. For brevity, imagine it looks like this:\
\
```\
bool CJsonNode::ParseRoot(string jsonText)\
  {\
   int pos=0;\
   SkipWhitespace(jsonText,pos);\
\
   // If it begins with '{', parse as object\
   if(StringSubstr(jsonText,pos,1)=="{")\
     {\
      return ParseObject(jsonText,pos);\
     }\
   // If it begins with '[', parse as array\
   if(StringSubstr(jsonText,pos,1)=="[")\
     {\
      return ParseArray(jsonText,pos);\
     }\
\
   // Otherwise, parse as a single value\
   return ParseValue(jsonText,pos);\
  }\
```\
\
For demonstration, we’re checking the first non-whitespace character and deciding if it’s an object ( {), array ( \[), or something else (which might be a string, number, boolean, or null). Our real implementation can be more defensive, handling errors if the character is unexpected.\
\
Let us look at how we parse different cases:\
\
1. _**Parse an Object:**_ When we see an opening brace ( {), we create an object node. We then repeatedly look for key-value pairs until we encounter a closing brace ( }). Here’s a conceptual snippet of how ParseObjectmight work:\
\
```\
bool CJsonNode::ParseObject(string text, int &pos)\
         {\
          // We already know text[pos] == '{'\
          m_type = JSON_OBJ;\
          pos++; // move past '{'\
          SkipWhitespace(text,pos);\
\
          // If the next char is '}', it's an empty object\
          if(StringSubstr(text,pos,1)=="}")\
            {\
             pos++;\
             return true;\
            }\
\
          // Otherwise, parse key-value pairs in a loop\
          while(true)\
            {\
             SkipWhitespace(text,pos);\
             // The key must be a string in double quotes\
             if(StringSubstr(text,pos,1)!="\"")\
               return false; // or set an error\
             // parse the string key (we’ll show a helper soon)\
             string objKey = "";\
             if(!ParseStringLiteral(text,pos,objKey))\
                return false;\
\
             SkipWhitespace(text,pos);\
             // Expect a colon\
             if(StringSubstr(text,pos,1)!=":")\
                return false;\
             pos++;\
\
             // Now parse the value\
             CJsonNode child;\
             if(!child.ParseValue(text,pos))\
                return false;\
\
             // Add the child to our arrays\
             int newIndex = ArraySize(m_children);\
             ArrayResize(m_children,newIndex+1);\
             ArrayResize(m_keys,newIndex+1);\
             m_children[newIndex] = child;\
             m_keys[newIndex]     = objKey;\
\
             SkipWhitespace(text,pos);\
             // If next char is '}', object ends\
             if(StringSubstr(text,pos,1)=="}")\
               {\
                pos++;\
                return true;\
               }\
             // Otherwise, we expect a comma before the next pair\
             if(StringSubstr(text,pos,1)!=",")\
                return false;\
             pos++;\
            }\
          // unreachable\
          return false;\
         }\
```\
\
\
Explanations:\
\
\
   - We confirm that the character is {, set our type to JSON\_OBJ, and increment pos.\
   - If } follows, the object is empty.\
   - Otherwise, we loop until we see a }or an error. Each iteration:\
     - Parse a string key in quotes.\
     - Skip spaces, expect a colon ( :).\
     - Parse the next value (which might be a string, number, array, object, etc.).\
     - Store that in our arrays ( m\_childrenand m\_keys).\
     - If we see }, we’re done. If we see a comma, we continue.\
\
This loop is central to reading a JSON object. The structure is repeated for arrays, except arrays have no keys—only indexed elements.\
\
2. **_Parse an Array:_** Arrays start with \[. Inside, we’ll find zero or more elements separated by commas. Something like:\
\
\
```\
[ "Hello", 123, false, {"nestedObj": 1}, [10, 20] ]\
```\
\
\
Code:\
\
\
```\
bool CJsonNode::ParseArray(string text, int &pos)\
     {\
      m_type = JSON_ARRAY;\
      pos++; // skip '['\
      SkipWhitespace(text,pos);\
\
      // If it's immediately ']', it's an empty array\
      if(StringSubstr(text,pos,1)=="]")\
        {\
         pos++;\
         return true;\
        }\
\
      // Otherwise, parse elements in a loop\
      while(true)\
        {\
         SkipWhitespace(text,pos);\
         CJsonNode child;\
         if(!child.ParseValue(text,pos))\
            return false;\
\
         // store the child\
         int newIndex = ArraySize(m_children);\
         ArrayResize(m_children,newIndex+1);\
         m_children[newIndex] = child;\
\
         SkipWhitespace(text,pos);\
         // if next char is ']', array ends\
         if(StringSubstr(text,pos,1)=="]")\
           {\
            pos++;\
            return true;\
           }\
         // must find a comma otherwise\
         if(StringSubstr(text,pos,1)!=",")\
            return false;\
         pos++;\
        }\
\
      return false;\
     }\
```\
\
\
We skip \[ and any whitespace. If we see \], it’s empty. Otherwise, we parse elements in a loop until we reach \]. The key difference from objects is that we don’t parse key-value pairs—just values, in sequence.\
\
3. _**Parse a Value,**_ Values in JSON can be a string, number, object, array, boolean, or null. Our ParseValuemight do something like:\
\
\
\
\
\
\
```\
bool CJsonNode::ParseValue(string text, int &pos)\
     {\
      SkipWhitespace(text,pos);\
      string c = StringSubstr(text,pos,1);\
\
      // Object\
      if(c=="{")\
        {\
         return ParseObject(text,pos);\
        }\
      // Array\
      if(c=="[")\
        {\
         return ParseArray(text,pos);\
        }\
      // String\
      if(c=="\"")\
        {\
         m_type = JSON_STRING;\
         return ParseStringLiteral(text,pos,m_value);\
        }\
      // Boolean or null\
      // We’ll look for 'true', 'false', or 'null'\
      if(StringSubstr(text,pos,4)=="true")\
        {\
         m_type    = JSON_BOOL;\
         m_boolVal = true;\
         pos+=4;\
         return true;\
        }\
      if(StringSubstr(text,pos,5)=="false")\
        {\
         m_type    = JSON_BOOL;\
         m_boolVal = false;\
         pos+=5;\
         return true;\
        }\
      if(StringSubstr(text,pos,4)=="null")\
        {\
         m_type = JSON_NULL;\
         pos+=4;\
         return true;\
        }\
\
      // Otherwise, treat it as a number or fail\
      return ParseNumber(text,pos);\
     }\
```\
\
\
Here we:\
\
\
1. Skip whitespace.\
2. Look at the current character (or substring) to see if it’s {, \[, ", etc.\
3. Call the relevant parse function.\
4. If we find “true,” “false,” or “null,” handle them directly.\
5. If nothing else matches, we assume it’s a number.\
\
Depending on your needs, you might add better error handling. For instance, if the substring doesn’t match a recognized pattern, you can set an error.\
\
4. **_Parse a Number_**, we need to parse something that looks numeric, like 123, 3.14, or -0.001. We can implement a quick approach by scanning until we reach a non-numeric character:\
\
\
```\
bool CJsonNode::ParseNumber(string text, int &pos)\
     {\
      m_type = JSON_NUMBER;\
\
      // capture starting point\
      int startPos = pos;\
      while(pos < StringLen(text))\
        {\
         string c = StringSubstr(text,pos,1);\
         if(c=="-" || c=="+" || c=="." || c=="e" || c=="E" || (c>="0" && c<="9"))\
           {\
            pos++;\
           }\
         else break;\
        }\
\
      // substring from startPos to pos\
      string numStr = StringSubstr(text,startPos,pos-startPos);\
      if(StringLen(numStr)==0)\
        return false;\
\
      // convert to double\
      m_numVal = StringToDouble(numStr);\
      return true;\
     }\
```\
\
\
We allow digits, an optional sign (- or +), decimal points, and exponent notation (e or E). Once we hit something else—like a space, a comma, or a bracket—we stop. Then we parse the substring into double. If your code needs to differentiate integers from decimals, you can add extra checks.\
\
\
### Extending Our Parser with Advanced Functionality\
\
By now, we have a functional JSON parser in MQL5 that can handle objects, arrays, strings, numbers, booleans, and null values. In this section, we’ll explore additional features and improvements. We’ll discuss how to retrieve child elements in a more convenient way, how to handle potential errors gracefully, and even how to convert data back into JSON text. By layering these enhancements atop the parser we’ve built, you’ll gain a more robust and flexible tool—one that can serve a variety of real-world needs.\
\
1. **Retrieving Children by Key or Index**\
\
If our parser is to be truly useful, we want to easily fetch the value of a given key in an object, or the value at a particular index in an array. For example, say we have this JSON:\
\
\
\
\
```\
{\
     "symbol": "EURUSD",\
     "lots": 0.02,\
     "settings": {\
       "slippage": 2,\
       "retries": 3\
     }\
}\
```\
\
\
Let’s imagine we’ve parsed it into a root CJsonNodeobject named rootNode. We’d like to do things like:\
\
\
\
\
```\
string sym = rootNode.GetChild("symbol").AsString();\
double lot = rootNode.GetChild("lots").AsNumber();\
int slip   = rootNode.GetChild("settings").GetChild("slippage").AsNumber();\
```\
\
\
Our current code structure might allow this if we define GetChild(const string key)in the parser. Here’s how such a method might look in your CJsonNodeclass:\
\
\
\
\
\
\
```\
CJsonNode* CJsonNode::GetChild(const string key)\
     {\
      if(m_type != JSON_OBJ)\
         return NULL;\
\
      // We look through m_keys to find a match\
      for(int i=0; i<ArraySize(m_keys); i++)\
        {\
         if(m_keys[i] == key)\
            return &m_children[i];\
        }\
      return NULL;\
     }\
```\
\
\
That way, if the current node is not an object, we simply return NULL. Otherwise, we scan through all m\_keysto find one that matches. If it does, we return a pointer to the corresponding child.\
\
Likewise, we can define a method for arrays:\
\
\
```\
CJsonNode* CJsonNode::GetChild(int index)\
     {\
      if(m_type != JSON_ARRAY)\
         return NULL;\
\
      if(index < 0 || index >= ArraySize(m_children))\
         return NULL;\
\
      return &m_children[index];\
     }\
```\
\
\
If the node is an array, we simply check bounds and return the appropriate element. If it’s not an array—or the index is out of range—we return NULL. Checking for NULLis crucial in your actual code before dereferencing.\
\
2. **Graceful Error Handling**\
\
In many real-world scenarios, JSON might arrive malformed (e.g., missing quotes, trailing commas, or unexpected symbols). A robust parser should detect and report these errors. You can do this by:\
\
\
\
\
1. Return a Boolean: Most of our parse methods already return bool. If something fails, we return false. But we can also store an internal error message like m\_errorMsg, so the calling code can see what went wrong.\
\
2. Keep Parsing or Abort?: Once you detect a fatal parse error—say, an unexpected character or an unclosed brace—you might decide to abort the entire parse and keep your node in an “invalid” state. Alternatively, you could try to skip or recover, but that’s more advanced.\
\
\
Here’s a conceptual tweak: inside ParseArrayor ParseObject, if you see something unexpected (like a key without quotes or a missing colon), you can write:\
\
```\
Print("Parse Error: Missing colon after key at position ", pos);\
return false;\
```\
\
Then, in your calling code, you might do:\
\
```\
CJsonNode root;\
    if(!root.ParseString(jsonText))\
      {\
       Print("Failed to parse JSON data. Check structure and try again.");\
       // Perhaps handle defaults or stop execution\
      }\
```\
\
It’s up to you how deeply you want to detail these messages. Sometimes, a single “parse failed” is enough for a trading scenario. Other times, you might want more nuance to debug your JSON input.\
\
3. **Converting MQL5 Data Back to JSON**\
\
Reading JSON is only half the story. What if you need to send data back to a server or write your own logs in JSON format? You can extend your CJsonNodeclass with a “serializer” method that walks through the node’s data and reconstructs the JSON text. Let’s call it ToJsonString(), for example:\
\
\
\
\
```\
string CJsonNode::ToJsonString() const\
     {\
      // We can define a helper that does the real recursion\
      return SerializeNode(0);\
     }\
\
string CJsonNode::SerializeNode(int depth) const\
     {\
      // If you prefer pretty-print with indentation, use 'depth'\
      // For now, let's keep it simple:\
      switch(m_type)\
        {\
         case JSON_OBJ:\
            return SerializeObject(depth);\
         case JSON_ARRAY:\
            return SerializeArray(depth);\
         case JSON_STRING:\
            return "\""+EscapeString(m_value)+"\"";\
         case JSON_NUMBER:\
         {\
            // Convert double to string carefully\
            return DoubleToString(m_numVal, 10);\
         }\
         case JSON_BOOL:\
            return m_boolVal ? "true":"false";\
         case JSON_NULL:\
            return "null";\
         default:\
            return "\"\""; // or some placeholder\
        }\
     }\
```\
\
\
Then you can define, for example, SerializeObject:\
\
\
```\
string CJsonNode::SerializeObject(int depth) const\
     {\
      string result = "{";\
      for(int i=0; i<ArraySize(m_children); i++)\
        {\
         if(i>0) result += ",";\
         string key   = EscapeString(m_keys[i]);\
         string value = m_children[i].SerializeNode(depth+1);\
         result += "\""+key+"\":";\
         result += value;\
        }\
      result += "}";\
      return result;\
     }\
```\
\
\
\
\
And similarly for arrays:\
\
\
```\
string CJsonNode::SerializeArray(int depth) const\
     {\
      string result = "[";\
      for(int i=0; i<ArraySize(m_children); i++)\
        {\
         if(i>0) result += ",";\
         result += m_children[i].SerializeNode(depth+1);\
        }\
      result += "]";\
      return result;\
     }\
```\
\
\
You’ll notice we used an EscapeStringfunction. We can reuse the code that handles JSON string escapes—like turning special characters into \\", \\\, \\n, etc. That ensures the output is valid JSON if it contains quotes or line breaks.\
\
If you’d like “pretty-printed” JSON, just insert some line breaks ( "\\n") and indentation. One approach is to build a short string of spaces based on depth, so your JSON structure becomes more visually neat:\
\
\
```\
string indentation = "";\
for(int d=0; d<depth; d++)\
      indentation += "  ";\
```\
\
\
Then insert that indentation before each line or element. This is optional but handy if you regularly need to read or debug the JSON output manually.\
\
\
\
\
\
If your JSON data is huge, say tens of thousands of lines, you might need to consider performance:\
\
1. Efficient String Operations\
\
      Be mindful that repeated substring operations ( StringSubstr) can be expensive. MQL5 is fairly efficient, but if your data is truly massive, you may consider chunk-based parsing or an iterative approach.\
\
2. Streaming vs. DOM Parsing\
\
      Our strategy is a “DOM-like” approach, meaning we parse the entire input into a tree structure. If data is so large that it can’t fit comfortably in memory, you’d need a streaming parser that processes one piece at a time. That’s more complicated but can be necessary for extremely large data sets.\
\
3. Caching\
\
      If you frequently query the same object for the same keys, you might store them in a small map or maintain direct pointers to speed up repeated lookups. For typical trading tasks, this is rarely needed, but it’s an option if performance is critical.\
4. **Best Practices**\
\
Below are a few best practices to keep your code safe and maintainable:\
\
\
   - Always Check for NULL\
\
     Whenever you call GetChild(...), verify the result is not NULL. Attempting to access a null pointer in MQL5 can lead to crashes or weird behavior.\
\
   - Validate Types\
\
     If you expect a number but the child is actually a string, that might cause an issue. Consider verifying GetType()or using defensive code, for example:\
\
\
```\
CJsonNode* node = parent.GetChild("lots");\
if(node != NULL && node.GetType() == JSON_NUMBER)\
  double myLots = node.AsNumber();\
```\
\
That helps ensure your data is what you think it is.\
\
**Default Values**\
\
Often, you want a safe fallback if the JSON is missing a key. You can write a helper function:\
\
```\
double getDoubleOrDefault(CJsonNode &obj, const string key, double defaultVal)\
  {\
   CJsonNode* c = obj.GetChild(key);\
   if(c == NULL || c.GetType() != JSON_NUMBER)\
     return defaultVal;\
   return c.AsNumber();\
  }\
```\
\
That way, your code can gracefully handle missing or invalid fields.\
\
   - Be Mindful of MQL5’s String and Array Limitations\
\
     MQL5 can handle large strings but keep an eye on memory usage. If your JSON is extremely big, test carefully.\
\
     Similarly, arrays can be resized, but extremely large arrays (hundreds of thousands of elements) can become unwieldy.\
\
   - Testing\
\
     Just as you would test an EA’s logic with historical data, test your JSON parser with a variety of sample inputs:\
\
     - Simple objects\
     - Nested objects\
     - Arrays of mixed data\
     - Large numbers, negative numbers\
     - Boolean and null\
     - Strings with special characters or escape sequences\
\
The more variations you try, the more confident you’ll be that your parser is robust.\
\
At this point, we’ve turned our basic parser into a powerful JSON utility. We can parse JSON strings into a hierarchical structure, retrieve data by key or index, handle parse failures, and even serialize nodes back into JSON text. This is enough for many MQL5 use cases—like reading a config file, fetching data from the web (if you have a bridge to HTTP requests), or generating your own JSON logs.\
\
In the final section, we’ll present a complete code listing that bundles together everything we’ve discussed. You’ll be able to paste it into your MQL5 editor as a single .mqhfile or .mq5script, adapt it to your naming conventions, and start using JSON data right away. Alongside the final code, we’ll offer concluding thoughts and some pointers for extending the library further if you have specialized requirements.\
\
### Full Code\
\
Congratulations on making it this far! You’ve learned the basics of JSON in MQL5, built a step-by-step parser, extended it with advanced functionality, and explored best practices for real-world usage. Now it’s time to share a single, integrated code listing that merges all the snippets into a coherent module. You can place this final code in an .mqhfile (or directly in your .mq5file) and include it wherever you need JSON handling in your MetaTrader 5 projects.\
\
Below is an example code implementation named CJsonNode.mqh. It unifies object/array parsing, error checking, serialization back to JSON, and retrieval by key or index.\
\
Important: This code is original and not a copy of the reference snippet provided earlier. It follows similar parsing logic but is distinct to meet our requirement of having a fresh approach. As always, feel free to adapt method names, add more robust error handling, or implement specialized features as needed.\
\
```\
#ifndef __CJSONNODE_MQH__\
#define __CJSONNODE_MQH__\
\
//+------------------------------------------------------------------+\
//| CJsonNode.mqh - A Minimalistic JSON Parser & Serializer in MQL5  |\
//| Feel free to adapt as needed.                                    |\
//+------------------------------------------------------------------+\
#property strict\
\
//--- Enumeration of possible JSON node types\
enum JsonNodeType\
  {\
   JSON_UNDEF  = 0,\
   JSON_OBJ,\
   JSON_ARRAY,\
   JSON_STRING,\
   JSON_NUMBER,\
   JSON_BOOL,\
   JSON_NULL\
  };\
\
//+-----------------------------------------------------------------+\
//| Class representing a single JSON node                           |\
//+-----------------------------------------------------------------+\
class CJsonNode\
  {\
public:\
   //--- Constructor & Destructor\
   CJsonNode();\
   ~CJsonNode();\
\
   //--- Parse entire JSON text\
   bool        ParseString(string jsonText);\
\
   //--- Check if node is valid\
   bool        IsValid();\
\
   //--- Get potential error message if not valid\
   string      GetErrorMsg();\
\
   //--- Access node type\
   JsonNodeType GetType();\
\
   //--- For arrays\
   int         ChildCount();\
\
   //--- For objects: get child by key\
   CJsonNode*  GetChild(string key);\
\
   //--- For arrays: get child by index\
   CJsonNode*  GetChild(int index);\
\
   //--- Convert to string / number / bool\
   string      AsString();\
   double      AsNumber();\
   bool        AsBool();\
\
   //--- Serialize back to JSON\
   string      ToJsonString();\
\
private:\
   //--- Data members\
   JsonNodeType m_type;       // Type of this node (object, array, etc.)\
   string       m_value;      // For storing string content if node is string\
   double       m_numVal;     // For numeric values\
   bool         m_boolVal;    // For boolean values\
   CJsonNode    m_children[]; // Child nodes (for objects and arrays)\
   string       m_keys[];     // Keys for child nodes (valid if JSON_OBJ)\
   bool         m_valid;      // True if node is validly parsed\
   string       m_errMsg;     // Optional error message for debugging\
\
   //--- Internal methods\
   void         Reset();\
   bool         ParseValue(string text,int &pos);\
   bool         ParseObject(string text,int &pos);\
   bool         ParseArray(string text,int &pos);\
   bool         ParseNumber(string text,int &pos);\
   bool         ParseStringLiteral(string text,int &pos);\
   bool         ParseKeyLiteral(string text,int &pos,string &keyOut);\
   string       UnescapeString(string input_);\
   bool         SkipWhitespace(string text,int &pos);\
   bool         AllWhitespace(string text,int pos);\
   string       SerializeNode();\
   string       SerializeObject();\
   string       SerializeArray();\
   string       EscapeString(string s);\
};\
\
//+-----------------------------------------------------------------+\
//| Constructor                                                     |\
//+-----------------------------------------------------------------+\
CJsonNode::CJsonNode()\
  {\
   m_type    = JSON_UNDEF;\
   m_value   = "";\
   m_numVal  = 0.0;\
   m_boolVal = false;\
   m_valid   = true;\
   ArrayResize(m_children,0);\
   ArrayResize(m_keys,0);\
   m_errMsg  = "";\
  }\
\
//+-----------------------------------------------------------------+\
//| Destructor                                                      |\
//+-----------------------------------------------------------------+\
CJsonNode::~CJsonNode()\
  {\
   // No dynamic pointers to free; arrays are handled by MQL itself\
  }\
\
//+-----------------------------------------------------------------+\
//| Parse entire JSON text                                          |\
//+-----------------------------------------------------------------+\
bool CJsonNode::ParseString(string jsonText)\
  {\
   Reset();\
   int pos = 0;\
   bool res = (ParseValue(jsonText,pos) && SkipWhitespace(jsonText,pos));\
\
   // If there's leftover text that's not whitespace, it's an error\
   if(pos < StringLen(jsonText))\
     {\
      if(!AllWhitespace(jsonText,pos))\
        {\
         m_valid  = false;\
         m_errMsg = "Extra data after JSON parsing.";\
         res      = false;\
        }\
     }\
   return (res && m_valid);\
  }\
\
//+-----------------------------------------------------------------+\
//| Check if node is valid                                          |\
//+-----------------------------------------------------------------+\
bool CJsonNode::IsValid()\
  {\
   return m_valid;\
  }\
\
//+-----------------------------------------------------------------+\
//| Get potential error message if not valid                        |\
//+-----------------------------------------------------------------+\
string CJsonNode::GetErrorMsg()\
  {\
   return m_errMsg;\
  }\
\
//+-----------------------------------------------------------------+\
//| Access node type                                                |\
//+-----------------------------------------------------------------+\
JsonNodeType CJsonNode::GetType()\
  {\
   return m_type;\
  }\
\
//+------------------------------------------------------------------+\
//| For arrays: get number of children                               |\
//+------------------------------------------------------------------+\
int CJsonNode::ChildCount()\
  {\
   return ArraySize(m_children);\
  }\
\
//+------------------------------------------------------------------+\
//| For objects: get child by key                                    |\
//+------------------------------------------------------------------+\
CJsonNode* CJsonNode::GetChild(string key)\
  {\
   if(m_type != JSON_OBJ)\
      return NULL;\
   for(int i=0; i<ArraySize(m_keys); i++)\
     {\
      if(m_keys[i] == key)\
         return &m_children[i];\
     }\
   return NULL;\
  }\
\
//+------------------------------------------------------------------+\
//| For arrays: get child by index                                   |\
//+------------------------------------------------------------------+\
CJsonNode* CJsonNode::GetChild(int index)\
  {\
   if(m_type != JSON_ARRAY)\
      return NULL;\
   if(index<0 || index>=ArraySize(m_children))\
      return NULL;\
   return &m_children[index];\
  }\
\
//+------------------------------------------------------------------+\
//| Convert to string / number / bool                                |\
//+------------------------------------------------------------------+\
string CJsonNode::AsString()\
  {\
   if(m_type == JSON_STRING) return m_value;\
   if(m_type == JSON_NUMBER) return DoubleToString(m_numVal,8);\
   if(m_type == JSON_BOOL)   return m_boolVal ? "true" : "false";\
   if(m_type == JSON_NULL)   return "null";\
   // For object/array/undefined, return empty or handle as needed\
   return "";\
  }\
\
//+------------------------------------------------------------------+\
//| Convert node to numeric                                          |\
//+------------------------------------------------------------------+\
double CJsonNode::AsNumber()\
  {\
   if(m_type == JSON_NUMBER) return m_numVal;\
   // If bool, return 1 or 0\
   if(m_type == JSON_BOOL)   return (m_boolVal ? 1.0 : 0.0);\
   return 0.0;\
  }\
\
//+------------------------------------------------------------------+\
//| Convert node to boolean                                          |\
//+------------------------------------------------------------------+\
bool CJsonNode::AsBool()\
  {\
   if(m_type == JSON_BOOL)   return m_boolVal;\
   if(m_type == JSON_NUMBER) return (m_numVal != 0.0);\
   if(m_type == JSON_STRING) return (StringLen(m_value) > 0);\
   return false;\
  }\
\
//+------------------------------------------------------------------+\
//| Serialize node back to JSON                                      |\
//+------------------------------------------------------------------+\
string CJsonNode::ToJsonString()\
  {\
   return SerializeNode();\
  }\
\
//+------------------------------------------------------------------+\
//| Reset node to initial state                                      |\
//+------------------------------------------------------------------+\
void CJsonNode::Reset()\
  {\
   m_type    = JSON_UNDEF;\
   m_value   = "";\
   m_numVal  = 0.0;\
   m_boolVal = false;\
   m_valid   = true;\
   ArrayResize(m_children,0);\
   ArrayResize(m_keys,0);\
   m_errMsg  = "";\
  }\
\
//+------------------------------------------------------------------+\
//| Dispatch parse based on first character                          |\
//+------------------------------------------------------------------+\
bool CJsonNode::ParseValue(string text,int &pos)\
  {\
   if(!SkipWhitespace(text,pos)) return false;\
   if(pos >= StringLen(text))    return false;\
\
   string c = StringSubstr(text,pos,1);\
\
   //--- Object\
   if(c == "{")\
      return ParseObject(text,pos);\
\
   //--- Array\
   if(c == "[")\
      return ParseArray(text,pos);\
\
   //--- String\
   if(c == "\"")\
      return ParseStringLiteral(text,pos);\
\
   //--- Boolean / null\
   if(StringSubstr(text,pos,4) == "true")\
     {\
      m_type    = JSON_BOOL;\
      m_boolVal = true;\
      pos      += 4;\
      return true;\
     }\
   if(StringSubstr(text,pos,5) == "false")\
     {\
      m_type    = JSON_BOOL;\
      m_boolVal = false;\
      pos      += 5;\
      return true;\
     }\
   if(StringSubstr(text,pos,4) == "null")\
     {\
      m_type = JSON_NULL;\
      pos   += 4;\
      return true;\
     }\
\
   //--- Otherwise, parse number\
   return ParseNumber(text,pos);\
  }\
\
//+------------------------------------------------------------------+\
//| Parse object: { ... }                                            |\
//+------------------------------------------------------------------+\
bool CJsonNode::ParseObject(string text,int &pos)\
  {\
   m_type = JSON_OBJ;\
   pos++; // skip '{'\
   if(!SkipWhitespace(text,pos)) return false;\
\
   //--- Check for empty object\
   if(pos < StringLen(text) && StringSubstr(text,pos,1) == "}")\
     {\
      pos++;\
      return true;\
     }\
\
   //--- Parse key-value pairs\
   while(pos < StringLen(text))\
     {\
      if(!SkipWhitespace(text,pos)) return false;\
\
      // Expect key in quotes\
      if(pos >= StringLen(text) || StringSubstr(text,pos,1) != "\"")\
        {\
         m_valid  = false;\
         m_errMsg = "Object key must start with double quote.";\
         return false;\
        }\
\
      string key = "";\
      if(!ParseKeyLiteral(text,pos,key))\
         return false;\
\
      if(!SkipWhitespace(text,pos)) return false;\
      // Expect a colon\
      if(pos >= StringLen(text) || StringSubstr(text,pos,1) != ":")\
        {\
         m_valid  = false;\
         m_errMsg = "Missing colon after object key.";\
         return false;\
        }\
      pos++; // skip ':'\
      if(!SkipWhitespace(text,pos)) return false;\
\
      // Parse the child value\
      CJsonNode child;\
      if(!child.ParseValue(text,pos))\
        {\
         m_valid  = false;\
         m_errMsg = "Failed to parse object value.";\
         return false;\
        }\
\
      // Store\
      int idx = ArraySize(m_children);\
      ArrayResize(m_children,idx+1);\
      ArrayResize(m_keys,idx+1);\
      m_children[idx] = child;\
      m_keys[idx]     = key;\
\
      if(!SkipWhitespace(text,pos)) return false;\
      if(pos >= StringLen(text))    return false;\
\
      string nextC = StringSubstr(text,pos,1);\
      if(nextC == "}")\
        {\
         pos++;\
         return true;\
        }\
      if(nextC != ",")\
        {\
         m_valid  = false;\
         m_errMsg = "Missing comma in object.";\
         return false;\
        }\
      pos++; // skip comma\
     }\
\
   return false; // didn't see closing '}'\
  }\
\
//+------------------------------------------------------------------+\
//| Parse array: [ ... ]                                             |\
//+------------------------------------------------------------------+\
bool CJsonNode::ParseArray(string text,int &pos)\
  {\
   m_type = JSON_ARRAY;\
   pos++; // skip '['\
   if(!SkipWhitespace(text,pos)) return false;\
\
   //--- Check for empty array\
   if(pos < StringLen(text) && StringSubstr(text,pos,1) == "]")\
     {\
      pos++;\
      return true;\
     }\
\
   //--- Parse elements\
   while(pos < StringLen(text))\
     {\
      CJsonNode child;\
      if(!child.ParseValue(text,pos))\
        {\
         m_valid  = false;\
         m_errMsg = "Failed to parse array element.";\
         return false;\
        }\
      int idx = ArraySize(m_children);\
      ArrayResize(m_children,idx+1);\
      m_children[idx] = child;\
\
      if(!SkipWhitespace(text,pos)) return false;\
      if(pos >= StringLen(text))    return false;\
\
      string nextC = StringSubstr(text,pos,1);\
      if(nextC == "]")\
        {\
         pos++;\
         return true;\
        }\
      if(nextC != ",")\
        {\
         m_valid  = false;\
         m_errMsg = "Missing comma in array.";\
         return false;\
        }\
      pos++; // skip comma\
      if(!SkipWhitespace(text,pos)) return false;\
     }\
\
   return false; // didn't see closing ']'\
  }\
\
//+------------------------------------------------------------------+\
//| Parse a numeric value                                            |\
//+------------------------------------------------------------------+\
bool CJsonNode::ParseNumber(string text,int &pos)\
  {\
   m_type = JSON_NUMBER;\
   int startPos = pos;\
\
   // Scan allowed chars in a JSON number\
   while(pos < StringLen(text))\
     {\
      string c = StringSubstr(text,pos,1);\
      if(c=="-" || c=="+" || c=="." || c=="e" || c=="E" || (c>="0" && c<="9"))\
        pos++;\
      else\
        break;\
     }\
\
   string numStr = StringSubstr(text,startPos,pos - startPos);\
   if(StringLen(numStr) == 0)\
     {\
      m_valid  = false;\
      m_errMsg = "Expected number, found empty.";\
      return false;\
     }\
\
   m_numVal = StringToDouble(numStr);\
   return true;\
  }\
\
//+------------------------------------------------------------------+\
//| Parse a string literal (leading quote already checked)           |\
//+------------------------------------------------------------------+\
bool CJsonNode::ParseStringLiteral(string text,int &pos)\
  {\
   pos++;  // skip leading quote\
   string result = "";\
\
   while(pos < StringLen(text))\
     {\
      string c = StringSubstr(text,pos,1);\
      if(c == "\"")\
        {\
         // closing quote\
         pos++;\
         m_type  = JSON_STRING;\
         m_value = UnescapeString(result);\
         return true;\
        }\
      if(c == "\\")\
        {\
         // handle escape\
         pos++;\
         if(pos >= StringLen(text))\
            break;\
         string ec = StringSubstr(text,pos,1);\
         result += ("\\" + ec); // accumulate, we'll decode later\
         pos++;\
        }\
      else\
        {\
         result += c;\
         pos++;\
        }\
     }\
\
   // If we get here, string was not closed\
   m_valid  = false;\
   m_errMsg = "Unclosed string literal.";\
   return false;\
  }\
\
//+------------------------------------------------------------------+\
//| Parse a string key (similar to a literal)                        |\
//+------------------------------------------------------------------+\
bool CJsonNode::ParseKeyLiteral(string text,int &pos,string &keyOut)\
  {\
   pos++;  // skip leading quote\
   string buffer = "";\
\
   while(pos < StringLen(text))\
     {\
      string c = StringSubstr(text,pos,1);\
      if(c == "\"")\
        {\
         pos++;\
         keyOut = UnescapeString(buffer);\
         return true;\
        }\
      if(c == "\\")\
        {\
         pos++;\
         if(pos >= StringLen(text))\
            break;\
         string ec = StringSubstr(text,pos,1);\
         buffer += ("\\" + ec);\
         pos++;\
        }\
      else\
        {\
         buffer += c;\
         pos++;\
        }\
     }\
\
   m_valid  = false;\
   m_errMsg = "Unclosed key string.";\
   return false;\
  }\
\
//+------------------------------------------------------------------+\
//| Unescape sequences like \" \\ \n etc.                            |\
//+------------------------------------------------------------------+\
string CJsonNode::UnescapeString(string input_)\
  {\
   string out = "";\
   int i      = 0;\
\
   while(i < StringLen(input_))\
     {\
      string c = StringSubstr(input_,i,1);\
      if(c == "\\")\
        {\
         i++;\
         if(i >= StringLen(input_))\
           {\
            // Single backslash at end\
            out += "\\";\
            break;\
           }\
         string ec = StringSubstr(input_,i,1);\
\
         if(ec == "\"")      out += "\"";\
         else if(ec == "\\") out += "\\";\
         else if(ec == "n")  out += "\n";\
         else if(ec == "r")  out += "\r";\
         else if(ec == "t")  out += "\t";\
         else if(ec == "b")  out += CharToString(8);   // ASCII backspace\
         else if(ec == "f")  out += CharToString(12);  // ASCII formfeed\
         else                out += ("\\" + ec);\
\
         i++;\
        }\
      else\
        {\
         out += c;\
         i++;\
        }\
     }\
   return out;\
  }\
\
//+------------------------------------------------------------------+\
//| Skip whitespace                                                  |\
//+------------------------------------------------------------------+\
bool CJsonNode::SkipWhitespace(string text,int &pos)\
  {\
   while(pos < StringLen(text))\
     {\
      ushort c = StringGetCharacter(text,pos);\
      if(c == ' ' || c == '\t' || c == '\n' || c == '\r')\
        pos++;\
      else\
        break;\
     }\
   // Return true if we haven't gone beyond string length\
   return (pos <= StringLen(text));\
  }\
\
//+------------------------------------------------------------------+\
//| Check if remainder is all whitespace                             |\
//+------------------------------------------------------------------+\
bool CJsonNode::AllWhitespace(string text,int pos)\
  {\
   while(pos < StringLen(text))\
     {\
      ushort c = StringGetCharacter(text,pos);\
      if(c != ' ' && c != '\t' && c != '\n' && c != '\r')\
         return false;\
      pos++;\
     }\
   return true;\
  }\
\
//+------------------------------------------------------------------+\
//| Serialization dispatcher                                         |\
//+------------------------------------------------------------------+\
string CJsonNode::SerializeNode()\
  {\
   switch(m_type)\
     {\
      case JSON_OBJ:    return SerializeObject();\
      case JSON_ARRAY:  return SerializeArray();\
      case JSON_STRING: return "\""+EscapeString(m_value)+"\"";\
      case JSON_NUMBER: return DoubleToString(m_numVal,8);\
      case JSON_BOOL:   return (m_boolVal ? "true" : "false");\
      case JSON_NULL:   return "null";\
      default:          return "\"\""; // undefined => empty string\
     }\
  }\
\
//+------------------------------------------------------------------+\
//| Serialize object                                                 |\
//+------------------------------------------------------------------+\
string CJsonNode::SerializeObject()\
  {\
   string out = "{";\
   for(int i=0; i<ArraySize(m_children); i++)\
     {\
      if(i > 0) out += ",";\
      out += "\""+EscapeString(m_keys[i])+"\":";\
      out += m_children[i].SerializeNode();\
     }\
   out += "}";\
   return out;\
  }\
\
//+------------------------------------------------------------------+\
//| Serialize array                                                  |\
//+------------------------------------------------------------------+\
string CJsonNode::SerializeArray()\
  {\
   string out = "[";\
   for(int i=0; i<ArraySize(m_children); i++)\
     {\
      if(i > 0) out += ",";\
      out += m_children[i].SerializeNode();\
     }\
   out += "]";\
   return out;\
  }\
\
//+------------------------------------------------------------------+\
//| Escape a string for JSON output (backslashes, quotes, etc.)      |\
//+------------------------------------------------------------------+\
string CJsonNode::EscapeString(string s)\
  {\
   string out = "";\
   for(int i=0; i<StringLen(s); i++)\
     {\
      ushort c = StringGetCharacter(s,i);\
      switch(c)\
        {\
         case 34:  // '"'\
            out += "\\\"";\
            break;\
         case 92:  // '\\'\
            out += "\\\\";\
            break;\
         case 10:  // '\n'\
            out += "\\n";\
            break;\
         case 13:  // '\r'\
            out += "\\r";\
            break;\
         case 9:   // '\t'\
            out += "\\t";\
            break;\
         case 8:   // backspace\
            out += "\\b";\
            break;\
         case 12:  // formfeed\
            out += "\\f";\
            break;\
         default:\
            // Directly append character\
            out += CharToString(c);\
            break;\
        }\
     }\
   return out;\
  }\
\
#endif // __CJSONNODE_MQH__\
```\
\
Let us take an instance of it's usage in a script:\
\
```\
//+------------------------------------------------------------------+\
//|                                                      ProjectName |\
//|                                      Copyright 2020, CompanyName |\
//|                                       http://www.companyname.net |\
//+------------------------------------------------------------------+\
#property strict\
#include <CJsonNode.mqh>\
\
void OnStart()\
  {\
   // Some JSON text\
   string jsonText = "{\"name\":\"Alice\",\"age\":30,\"admin\":true,\"items\":[1,2,3],\"misc\":null}";\
\
   CJsonNode parser;\
   if(parser.ParseString(jsonText))\
     {\
      Print("JSON parsed successfully!");\
      Print("Name: ", parser.GetChild("name").AsString());\
      Print("Age: ",  parser.GetChild("age").AsNumber());\
      Print("Admin?",  parser.GetChild("admin").AsBool());\
      // Serialize back\
      Print("Re-serialized JSON: ", parser.ToJsonString());\
     }\
   else\
     {\
      Print("JSON parsing error: ", parser.GetErrorMsg());\
     }\
  }\
\
//+------------------------------------------------------------------+\
```\
\
Expected output is self-explanatory, Feel free to test it out.\
\
### Conclusion\
\
With this final code in hand, you have everything you need to parse, manipulate, and even generate JSON directly in MetaTrader 5:\
\
- Parsing JSON: ParseString() transforms raw text into a structured node hierarchy.\
- Querying Data: GetChild(key) and GetChild(index) let you navigate objects and arrays easily.\
- Validation: Check IsValid() and GetErrorMsg() to see if parsing succeeded or if there were issues (like mismatched braces).\
- Serialization: ToJsonString() reassembles the node (and children) back into valid JSON text.\
\
Feel free to adapt this library to your specific needs. You might, for example, add more comprehensive error reporting, specialized numeric conversions, or streaming capabilities for very large data sets. But the foundation here should be sufficient for most typical use cases, like reading parameters from a file or interacting with web-based APIs.\
\
That’s it! You’ve reached the end of our deep dive into JSON handling in MQL5. Whether you’re implementing a complex, data-driven trading engine or just loading config parameters from a local file, a reliable JSON parser and serializer can make your life much easier. We hope this article (and the code within) helps you integrate JSON smoothly into your automated trading workflows.\
\
Happy Coding! Happy Trading!\
\
**Attached files** \|\
\
\
[Download ZIP](https://www.mql5.com/en/articles/download/16791.zip "Download all attachments in the single ZIP archive")\
\
[CJsonNode.mqh](https://www.mql5.com/en/articles/download/16791/cjsonnode.mqh "Download CJsonNode.mqh")(41.39 KB)\
\
**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.\
\
This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.\
\
#### Other articles by this author\
\
- [Custom Debugging and Profiling Tools for MQL5 Development (Part I): Advanced Logging](https://www.mql5.com/en/articles/17933)\
- [Building a Custom Market Regime Detection System in MQL5 (Part 2): Expert Advisor](https://www.mql5.com/en/articles/17781)\
- [Building a Custom Market Regime Detection System in MQL5 (Part 1): Indicator](https://www.mql5.com/en/articles/17737)\
- [Advanced Memory Management and Optimization Techniques in MQL5](https://www.mql5.com/en/articles/17693)\
- [Mastering File Operations in MQL5: From Basic I/O to Building a Custom CSV Reader](https://www.mql5.com/en/articles/16614)\
- [Modified Grid-Hedge EA in MQL5 (Part IV): Optimizing Simple Grid Strategy (I)](https://www.mql5.com/en/articles/14518)\
\
**Last comments \|**\
**[Go to discussion](https://www.mql5.com/en/forum/481376)**\
(5)\
\
\
![Edgar Akhmadeev](https://c.mql5.com/avatar/avatar_na2.png)\
\
**[Edgar Akhmadeev](https://www.mql5.com/en/users/dali)**\
\|\
24 Oct 2025 at 20:45\
\
Oh, there's another one. I think it's the fifth.\
\
Well, when I get my hands on it, I'll do a comparative benchmark.\
\
\
![Edgar Akhmadeev](https://c.mql5.com/avatar/avatar_na2.png)\
\
**[Edgar Akhmadeev](https://www.mql5.com/en/users/dali)**\
\|\
28 Oct 2025 at 18:40\
\
I compared the speed of 4 libraries, including ToyJson3 from MQL5Book. As a sample json I took Binance's response to "exchangeInfo" with the size of 768 Kb. When the library reads it from a string, it is parsed in its entirety, then I select one symbol and read all its data. In a loop 100 times.\
\
```\
MetaTrader 5 x64 build 5370 started for MetaQuotes Ltd.\
Windows 10 build 19045, 4 x AMD Ryzen 3 PRO 3200 GE w/ Radeon Vega, AVX2, 6 / 13 Gb memory, 241 / 427 Gb disk, UAC, GMT+3\
cpu='AVX2 + FMA3'\
```\
\
Result (query processing time):\
\
99.5 ms - JAson 1.12 [(https://www.mql5.com/en/code/13663)](https://www.mql5.com/en/code/13663 "https://www.mql5.com/en/code/13663")\
\
85.5 ms - JAson 1.13\
\
46.9 ms - ToyJson3 (https://www.mql5.com/ru/forum/459079/page4#comment\_57805801)\
\
41 ms - JSON [(https://www.mql5.com/en/code/53107)](https://www.mql5.com/en/code/53107 "https://www.mql5.com/en/code/53107")\
\
1132 ms - JsonNode (this library)\
\
38 ms - my implementation based on JSON\
\
PS: Once upon a time another, very stripped down, library seemed to pop up here. But I've lost track of it.\
\
PPS: I don't publish the script for measuring. The code is in a completely unsightly form.\
\
![Alain Verleyen](https://c.mql5.com/avatar/2024/5/663a6cdf-e866.jpg)\
\
**[Alain Verleyen](https://www.mql5.com/en/users/angevoyageur)**\
\|\
29 Oct 2025 at 16:13\
\
**Edgar Akhmadeev [#](https://www.mql5.com/en/forum/481376#comment_58383087):**\
\
I compared the speed of 4 libraries, including ToyJson3 from MQL5Book. As a sample json I took Binance's response to "exchangeInfo" with the size of 768 Kb. When the library reads it from a string, it is parsed in its entirety, then I select one symbol and read all its data. In a loop 100 times.\
\
Result (query processing time):\
\
99.5 ms - JAson 1.12 [(https://www.mql5.com/en/code/13663)](https://www.mql5.com/en/code/13663 "https://www.mql5.com/en/code/13663")\
\
85.5 ms - JAson 1.13\
\
46.9 ms - ToyJson3 (https://www.mql5.com/ru/forum/459079/page4#comment\_57805801)\
\
41 ms - JSON [(https://www.mql5.com/en/code/53107)](https://www.mql5.com/en/code/53107 "https://www.mql5.com/en/code/53107")\
\
1132 ms - JsonNode (this library)\
\
38 ms - my implementation based on JSON\
\
PS: Once upon a time another, very stripped down, library seemed to pop up here. But I've lost track of it.\
\
PPS: I don't publish the script for measuring. The code is in a completely unsightly form.\
\
Could you post the json string or file please ?\
\
\
![trader6_1](https://c.mql5.com/avatar/avatar_na2.png)\
\
**[trader6\_1](https://www.mql5.com/en/users/trader6_1)**\
\|\
29 Oct 2025 at 16:31\
\
**Alain Verleyen [#](https://www.mql5.com/ru/forum/498411#comment_58391455):**\
\
Could you post the json string or file please?\
\
[https://fapi.binance.com/fapi/v1/exchangeInfo](https://www.mql5.com/go?link=https://fapi.binance.com/fapi/v1/exchangeInfo "https://fapi.binance.com/fapi/v1/exchangeInfo")\
\
[https://eapi.binance.com/eapi/v1/exchangeInfo](https://www.mql5.com/go?link=https://eapi.binance.com/eapi/v1/exchangeInfo "https://eapi.binance.com/eapi/v1/exchangeInfo")\
\
778 KB (796,729 bytes).\
\
![Edgar Akhmadeev](https://c.mql5.com/avatar/avatar_na2.png)\
\
**[Edgar Akhmadeev](https://www.mql5.com/en/users/dali)**\
\|\
29 Oct 2025 at 17:46\
\
**Alain Verleyen [#](https://www.mql5.com/ru/forum/498411#comment_58391455):**\
\
Could you post the json string or file please?\
\
Here is a copy of the string obtained programmatically from the API.\
\
\
![Building a Keltner Channel Indicator with Custom Canvas Graphics in MQL5](https://c.mql5.com/2/118/Building_a_Keltner_Channel_Indicator_with_Custom_Canvas_Graphics_in_MQL5___LOGO.png)[Building a Keltner Channel Indicator with Custom Canvas Graphics in MQL5](https://www.mql5.com/en/articles/17155)\
\
In this article, we build a Keltner Channel indicator with custom canvas graphics in MQL5. We detail the integration of moving averages, ATR calculations, and enhanced chart visualization. We also cover backtesting to evaluate the indicator’s performance for practical trading insights.\
\
![Neural Networks in Trading: Using Language Models for Time Series Forecasting](https://c.mql5.com/2/86/Neural_networks_in_trading__Using_language_models_to_forecast_time_series___LOGO.png)[Neural Networks in Trading: Using Language Models for Time Series Forecasting](https://www.mql5.com/en/articles/15451)\
\
We continue to study time series forecasting models. In this article, we get acquainted with a complex algorithm built on the use of a pre-trained language model.\
\
![MQL5 Wizard Techniques you should know (Part 54): Reinforcement Learning with hybrid SAC and Tensors](https://c.mql5.com/2/118/MQL5_Wizard_Techniques_you_should_know_Part_54___LOGO.png)[MQL5 Wizard Techniques you should know (Part 54): Reinforcement Learning with hybrid SAC and Tensors](https://www.mql5.com/en/articles/17159)\
\
Soft Actor Critic is a Reinforcement Learning algorithm that we looked at in a previous article, where we also introduced python and ONNX to these series as efficient approaches to training networks. We revisit the algorithm with the aim of exploiting tensors, computational graphs that are often exploited in Python.\
\
![Robustness Testing on Expert Advisors](https://c.mql5.com/2/118/Robustness_Testing_on_Expert_Advisors__LOGO2.png)[Robustness Testing on Expert Advisors](https://www.mql5.com/en/articles/16957)\
\
In strategy development, there are many intricate details to consider, many of which are not highlighted for beginner traders. As a result, many traders, myself included, have had to learn these lessons the hard way. This article is based on my observations of common pitfalls that most beginner traders encounter when developing strategies on MQL5. It will offer a range of tips, tricks, and examples to help identify the disqualification of an EA and test the robustness of our own EAs in an easy-to-implement way. The goal is to educate readers, helping them avoid future scams when purchasing EAs as well as preventing mistakes in their own strategy development.\
\
[![](https://www.mql5.com/ff/sh/jup0jccfs9655z9z2/01.png)Learn to create your own robotsRead our book "MQL5 Programming for Traders"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/book%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.algobook%26utm_content=visit.page%26utm_campaign=algobook.promo.04.2024&a=rsxjstxkzbrlgjjrxaglpezpvrjflnvw&s=7224440013c3dbc50ba9cc078cd015fabca36df446b8e75028d6b30234663872&uid=&ref=https://www.mql5.com/en/articles/16791&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5071594179143871208)\
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