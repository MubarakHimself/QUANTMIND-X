---
title: Regular expressions for traders
url: https://www.mql5.com/en/articles/2432
categories: Integration
relevance_score: 0
scraped_at: 2026-01-24T14:06:36.939298
---

[![](https://www.mql5.com/ff/si/3fgkjn78mkxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=ocndbzpeklfncxysjbwfhhbalbrsdbtv&s=a4309643278437a00bdd33c5809fc6b4b4032749c00fccd07b3b84e7b8b45126&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=pzyjlmoakyomlkkjklrhgdhjbgozyaws&ssn=1769252795713939402&ssn_dr=0&ssn_sr=0&fv_date=1769252795&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F2432&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Regular%20expressions%20for%20traders%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176925279577229493&fz_uniq=5083341601893259707&sv=2552)

MetaTrader 5 / Examples


### Introduction

_Regular expression_ is a special tool and language for handling texts based on a specified pattern. Multiple metacharacters and rules are defined by syntax of regular expressions.

They are able to carry out two main functions:

- search for a pattern in strings;
- replace a pattern found.

When creating patterns for regular expressions, as previously mentioned, special characters, metacharacters and classes (sets) of characters are used. It means a regular expression is a regular string, and all non-special (non-reserved) characters are considered regular.

The search of a specified pattern in a string is performed by a regular expression handler.  In .NET Framework, and, therefore, in the [RegularExpressions library for MQL5](https://www.mql5.com/en/code/15242), a regular expression handler backtracks for regular expressions. It is a variation of a traditional NFA (Nondeterministic Finite Automaton), the same as those applied in Perl, Python, Emacs and Tcl. It is used to replace pattern-matches found in a string.

### 1\. Basics for regular expressions

Metacharacters are special characters that specify commands and managing sequences operating like MQL5 and C# managing sequences. Such characters are preceded by a backslash(\\), and each of them has a special purpose.

MQL5 and C# metacharacters of regular expressions in the following tables are grouped according to meaning.

**1.1. Character classes:**

| Character | Description | Example | Matches |
| --- | --- | --- | --- |
| \[...\] | Any character indicated in brackets | \[a-z\] | A source string may have any character of English alphabet in lowercase |
| --- | --- | --- | --- |
| \[^...\] | Any character not indicated in brackets | \[^0-9\] | A source string may have any character, apart from numbers |
| --- | --- | --- | --- |
| . | Any character apart from line feed or other separator of Unicode string | ta.d | "trad" in "trade" string |
| --- | --- | --- | --- |
| \\w | Any word character that is not a space, tab character etc. | \\w | "M","Q","L","5" in "MQL 5" string |
| --- | --- | --- | --- |
| \\W | Any character that is not a word character | \\W | " ", "." in "MQL 5" string. |
| --- | --- | --- | --- |
| \\s | Any white-space character from the Unicode set | \\w\\s | "L " in "MQL 5" string |
| --- | --- | --- | --- |
| \\S | Any non-white-space character from the Unicode set. Please note <br>that \\w and \\S characters are not the same | \\S | "M", "Q", "L", "5", "." in <br>"MQL 5" string |
| --- | --- | --- | --- |
| \\d | Any ASCII digits. Equivalent to \[0-9\] | \\d | "5" in "MQL 5." |
| --- | --- | --- | --- |

**1.2. Characters of repetition:**

| Character | Description | Example | Matches |
| --- | --- | --- | --- |
| {n,m} | Corresponds to a previous pattern repeated no less than n or no more than m times | s{2,4} | "Press", "ssl", "progressss" |
| --- | --- | --- | --- |
| {n,} | Corresponds to a previous pattern repeated n or more times | s{1,} | "ssl" |
| --- | --- | --- | --- |
| {n} | Precisely matches n instances of a previous pattern | s{2} | "Press", "ssl", but not "progressss" |
| --- | --- | --- | --- |
| ? | Corresponds to zero or one instance of a previous pattern;<br>previous pattern is not obligatory | Equivalent to {0,1} |  |
| --- | --- | --- | --- |
| + | Corresponds to one or more instances of a previous pattern | Equivalent to {1,} |  |
| --- | --- | --- | --- |
| \* | Corresponds to zero or more instances of a previous pattern | Equivalent to {0,} |  |
| --- | --- | --- | --- |

**1.3. Characters of regular expressions of selection:**

| Character | Description | Example | Matches |
| --- | --- | --- | --- |
| \| | Corresponds either to subexpressions on the left, or subexpression on the right (analog of logical operation OR). | 1(1\|2)0 | "110", "120" in <br>"100, 110, 120, 130" string |
| --- | --- | --- | --- |
| (...) | Grouping. Groups elements in a single whole that can be used with characters \*, +, ?, \|, etc.<br>Also remembers characters that correspond to this group for using in subsequent links. |  |  |
| --- | --- | --- | --- |
| (?:...) | Only grouping. Groups elements in a single whole, but doesn't remember characters that correspond to this group. |  |  |
| --- | --- | --- | --- |

**1.4. Anchor characters of regular expressions:**

| Character | Description | Example | Matches |
| --- | --- | --- | --- |
| ^ | Corresponds to the start of a string expression or the start of a string at multi-string search. | ^Hello | "Hello, world", but not "Ok, Hello world" because the word <br>"Hello" in this string is placed not at the start |
| --- | --- | --- | --- |
| $ | Corresponds to the end of a string expression or the end of a string at multi-string search. | Hello$ | "World, Hello" |
| --- | --- | --- | --- |
| \\b | Corresponds to word boundary, i.e. corresponds to the position between \\w and \\W characters<br>or between \\w character and the start or the end of a string. | \\b(my)\\b | In the "Hello my world" string, the word "my" is selected |
| --- | --- | --- | --- |

For more information about elements of regular expressions, please read the article on the official [Microsoft](https://www.mql5.com/go?link=https://docs.microsoft.com/en-us/dotnet/standard/base-types/regular-expression-language-quick-reference "https://msdn.microsoft.com/en-us/library/az24scfc%28v=vs.110%29.aspx") website.

### 2\. Features of implementing regular expressions for MQL5

#### 2.1. Third-party files stored in the Internal folder

In order to achieve close implementation of [RegularExpressions for MQL5](https://www.mql5.com/en/code/15242) to the .Net source code, it was also required to transfer a fraction of third party files. All of them are stored in the **Internal** folder and potentially could be very interesting.

Let's have a closer look at the content of the **Internal** folder.

1. Generic — this folder contains files for implementing strictly typified collections, enumerations and their interfaces. A more detailed description is provided below.
2. TimeSpan — files for implementing the [TimeSpan](https://www.mql5.com/go?link=https://msdn.microsoft.com/en-us/library/system.timespan.aspx "/go?link=https://msdn.microsoft.com/en-us/library/system.timespan.aspx") structure that provides a time interval.
3. Array.mqh — the Array class with a number of static methods for operating with arrays is implemented in this file. For example: sorting, binary search, receiving enumeration, receiving element index, etc.
4. DynamicMatrix.mqh — this file has two main classes for implementing multidimensional dynamic arrays. These are pattern classes that, therefore, are appropriate for standard types and pointers to classes.
5. IComparable.mqh — file that implements the IComparable interface that is necessary to support a number of methods in typified collections.
6. Wrappers.mqh — wrappers for standard types and methods for finding hash codes on them.

**Generic** has three strictly typified collections implemented:

1. List<T> presents a strictly typified list of objects available by index. Supports methods for searching by list, sorting and other operations with lists.
2. Dictionary<TKey,TValue> presents a collection of keys and values.
3. LinkedList<T>  presents a doubly linked list.

Let's see the use of List<T> from the TradeHistoryParsing Expert Advisor. This EA reads all trade history from .html file and filters it by selected columns and records. The trade history consists of two tables: Deals and Orders. The **OrderRecord** and **DealRecord** classes interpret one record (tuple) from the tables with Orders and Trades, respectively. Therefore, every column can be presented as a list of its records:

```
List<OrderRecord*>*m_list1 = new List<OrderRecord*>();
List<DealRecord*>*m_list2 = new List<DealRecord*>();
```

Since the **List<T>** class supports sorting methods, it means that T type objects must be compared between each other. In other words, <,>,== operations are implemented for this type. There are no issues with standard elements, but if we need to create List<T>, where T indicates the custom class, then we receive error. There are two ways to handle this issue. First, we can explicitly reload the comparison operators in our class. Another solution is to make the class a descendant of the IComparable interface. The second option is considerably shorter in implementation, however, it disrupts correct sorting. In cases when it is necessary to sort out custom classes, we must reload all comparison operators. In addition to that it is advisable to implement inheritance.

This is just one of the features of the List<T> class. More information is provided below.

**Dictionary<TKey,TValue>** — a kind of a dictionary with sets of values and unique keys that correspond to them. At the same time, several values can be attached to one key. Types of keys and values are determined by users at the stage of creating an object. As seen from the description, the Dictionary<TKey,TValue> class is very suitable for the hash table's role. In order to speed up the operation with Dictionary<TKey,TValue>, you should create a new class, that is a descendant of the IEqualityComparer<T> class, and reload two functions:

- **bool Equals(T x,T y) —** the function returns **true**, if x equals y, and **false**— if otherwise.
- **int GetHashCode(T obj) —** the function returns a hash code from obj.

In the RegularExpressions library for MQL5, this feature is used for all dictionaries with strings used as keys.

Implementation of StringEqualityComparer:

```
class StringEqualityComparer : public IEqualityComparer<string>
  {
public:
   //--- Methods:
   //+------------------------------------------------------------------+
   //| Determines whether the specified objects are equal.              |
   //+------------------------------------------------------------------+
   virtual bool Equals(string x,string y)
     {
      if(StringLen(x)!=StringLen(y)){ return (false); }
      else
        {
         for(int i=0; i<StringLen(x); i++)
             if(StringGetCharacter(x,i)!=StringGetCharacter(y,i)){ return (false); }
        }
      return (true);
     }
   int GetHashCode(string obj)
     {
      return (::GetHashCode(obj));
     }
  };
```

Now, when creating a new object that belongs to the Dictionary<TKey,TValue> class with strings used as keys, we will send the pointer to the StringEqualityComparer object as a parameter in the constructor:

```
Dictionary<string,int> *dictionary= new Dictionary<string,int>(new StringEqualityComparer);
```

**LinkedList<T>** is a data structure that includes a number of elements. Every element contains an informative part and two pointers to previous and following elements. Therefore, two elements positioned next to each other mutually refer to one another. Nodes of this list are implemented by the LinkedListNode<T> objects. There is a standard set in every node that contains value, pointer to the list and pointers to adjacent nodes.

![](https://c.mql5.com/2/23/2_en__7.png)

Also, enumerators are implemented for all three above mentioned collections. **Enumerator** is a generalized IEnumerator<T> interface. IEnumerator<T> allows to implement a full bypass of collection, regardless of its structure.

In order to obtain the enumerator, we must call the GetEnumerator() method from the object, whose class implements the IEnumerable interface:

```
List<int>* list = new List<int>();
list.Add(0);
list.Add(1);
list.Add(2);
IEnumerator<int> *en = list.GetEnumerator();
while(en.MoveNext())
  {
   Print(en.Current());
  }
delete en;
delete list;
```

In this example, we iterate over the whole list and print every value. All this could be achieved by arranging a simple for loop, but, frequently, the approach with enumerators in more convenient. In fact, this solution is suitable when creating iteration over Dictionary<TKey,TValue>.

#### 2.2. Features of the RegularExpressions library for MQL5.

**1\.** In order to include all functionality of regular expressions in our project, the following section must be added:

```
#include <RegularExpressions\Regex.mqh>
```

**2.** Due to lack of namespaces in MQL5, and, therefore, the [internal](https://www.mql5.com/go?link=https://docs.microsoft.com/en-us/dotnet/csharp/language-reference/keywords/internal "/go?link=https://docs.microsoft.com/en-us/dotnet/csharp/language-reference/keywords/internal") access modifier, we have access to all internal classes and many methods of the library. In fact, this is considered unnecessary when working with regular expressions.

The following classes will be of interest to us for operation with regular expressions :

- Capture — provides results of one successful record of subexpression.
- CaptureCollection — provides a set of records made by one group.
- Group — provides results of a separate record group.
- GroupCollection — returns a set of recorded groups in a single search match.
- Match — provides results from a separate match of regular expression.
- MatchCollection — presents a set of successful matches found by iterative application of regular expression pattern to input string.
- Regex — represents immutable regular expression.

In addition to the above mentioned classes we will use:

- MatchEvaluator — pointer to the function that presents a method called each time when a match of regular expression is found.
- RegexOptions — enumeration that presents values to be used when specifying parameters of regular expressions.

RegexOptions — incomplete copy of the source enumeration from .Net, includes the following elements:

| Parameter | Description |
| **None** | Parameters are not specified. |
| **IgnoreCase** | Search of matches is not case-sensitive. |
| **Multiline** | Indicates multiline mode. |
| **ExplicitCapture** | Not to cover groups without name. Only valid selections — explicitly named or numbered groups in the format **(?<** _name_ **\>** _subexpression_ **)**. |
| **Singleline** | Indicates single line mode. |
| **IgnorePatternWhitespace** | Removes white spaces from a pattern without escape-sequence and enables comments marked with "#" character. |
| **RightToLeft** | Indicates that search will be performed from right to left, not from left to right. |
| **Debug** | Indicates that the program works using debugger. |
| **ECMAScript** | Enables ECMAScript compatible behavior for expression. This value can be used only with IgnoreCase and Multiline. |

These options are used to create a new object of the Regex class or when calling its static methods.

The examples of using all these classes, pointer and enumeration can be found in the source code of the Tests.mq5 Expert Advisor.

**3\.** As in .Net Framework version, a storage (static cache memory) of regular expressions is implemented. All regular expressions that are not explicitly created (examples of the Regex class) are placed into this storage. Such approach speeds up the operation of scripts, since it is no longer necessary to build regular expressions from scratch, if they match one of the existing patterns. The storage size by default equals 15. The **Regex::CacheSize()** method returns or specifies a maximum number of records in the current static cache storage of compiled regular expressions.

**4\.** The above mentioned storage needs to be cleared. The **Regex::ClearCache()** static function is called for this purpose. Clearing the storage is advisable after you finish working with regular expressions, otherwise there is a high risk to remove pointers and objects that may be required.

**5\.** C# syntax allows to place '@' character before strings in order to ignore all formatting signs. MQL5 doesn't provide this approach, therefore all control characters in the pattern of regular expressions should be explicitly specified.

### 3\. Example of analyzing trade history

The following operations are implied in this example.

1. Reading trade history from a sandbox in .html format.
2. Selecting a table from "Orders" or "Deals" for a subsequent work.
3. Choosing filters for a table.
4. Graphic presentation of a filtered table.
5. Brief mathematical statistics based on a filtered table.
6. Option to save filtered table.

All these 6 points are implemented in the TradeHistoryParsing.mq5 Expert Advisor.

First of all, when operating with an Expert Advisor, a trade history should be downloaded. Therefore, in the MetaTrader5 terminal we go to "Toolbox" panel, "History" tab, and right click to open a dialog window, select "Report" and then HTML (Internet Explorer).

![](https://c.mql5.com/2/23/HTML.png)

We save the file in the [sand box](https://www.mql5.com/en/docs/files) (\\MetaTrader 5\\MQL5\\Files).

Now, while running the Expert Advisor in the dialog window we go to "Inputs" tab and enter the name of our file in the file\_name field:

![](https://c.mql5.com/2/23/Image_1.png)

After pressing "OK", the interface of EA will appear:

![](https://c.mql5.com/2/23/HistoryParsing__1.png)

As previously mentioned, both tables are presented in the Expert Advisor in the form of two typified list: List<OrderRecord\*> and List<DealRecord\*>.

Constructors for OrderRecord and DealRecord classes use string array as a parameter that is a single record from the table.

In order to create these arrays we are going to need regular expressions. The entire analysis of history is performed in the constructor of the TradeHistory class, where presentations of both columns are stored, and methods are implemented by their filter. The constructor of this class takes one parameter — path, in our case it is a name of the .html history file:

```
                TradeHistory(const string path)
{
 m_file_name=path;
 m_handel= FileOpen(path,FILE_READ|FILE_TXT);
 m_list1 = new List<OrderRecord*>();
 m_list2 = new List<DealRecord*>();
 Regex *rgx=new Regex("(>)([^<>]*)(<)");
 while(!FileIsEnding(m_handel))
   {
    string str=FileReadString(m_handel);
    MatchCollection *matches=rgx.Matches(str);
    if(matches.Count()==23)
      {
       string in[11];
       for(int i=0,j=1; i<11; i++,j+=2)
         {
          in[i]=StringSubstr(matches[j].Value(),1,StringLen(matches[j].Value())-2);
         }
       m_list1.Add(new OrderRecord(in));
      }
    else if(matches.Count()==27)
      {
       string in[13];
       for(int i=0,j=1; i<13; i++,j+=2)
         {
          in[i]=StringSubstr(matches[j].Value(),1,StringLen(matches[j].Value())-2);
         }
       m_list2.Add(new DealRecord(in));
      }
    delete matches;
   }
 FileClose(m_handel);
 delete rgx;
 Regex::ClearCache();
}
```

The code of this constructor shows that we use only one regular expression with a pattern "(>)(\[^<>\]\*)(<)" for analyzing trade history. Let's consider this pattern carefully:

| (>) | Search of '>' character |
| (^\[<>\]\*) | Any character apart from '>' and '<', that is repeated zero or more times |
| (<) | Search of '<' character |

This regular expression searches through all substrings that begin with '>' and end with '<'. The text between them shouldn't begin with '<' or '>'. In other words, we obtain the text between tags in the .html file. There will be unnecessary brackets on the sides, but they will be shortly removed. All found substrings are stored in MatchCollection, which is a collection of all substrings that satisfy the pattern of regular expression and are found in the source string. Due to the .html file structure, we can accurately determine, whether our string is a record from the Orders table, the Deals table or other string, by simply calculating the total of matches. This way, the string is a record from the Orders table, if the number of matches equals 23, or from the Deals table upon the result of 27 matches. In any other cases, we are not interested in this string. Now, we will extract all even elements ("><" strings in odd elements) from our collection, trim the first and the last character and record the ready string into array:

```
in[i]=StringSubstr(matches[j].Value(),1,StringLen(matches[j].Value())-2);
```

Each time when reading a new string, a collection of matches should be deleted. After reading the entire file, we should close it, delete the regular expression and clear the buffer.

Now, we need to implement the table filter, specifically, by selecting a column and certain value from it to obtain a trimmed table. In our case, a list should generate a sublist. For this purpose we can create a new list, arrange full iteration of all elements of the old list, and if it satisfies the specified conditions, we will add it to a new list.

There is also another way based on the FindAll(Predicate match) method for List<T>. It extracts all elements that satisfy conditions of the specified predicate that is a pointer to the function:

```
typedef bool (*Predicate)(IComparable*);
```

We have mentioned the IComparable interface before.

It remains to implement the actual **match** function, where we already know the rule applied to accept or decline the list. In our case this is a column number and its value inside. In order to solve this issue in the Record class that is descendant of the OrderRecord and DealRecord classes, two static methods SetIndex(const int index) and SetValue(const string value) are applied. They accept and store the column number and value. This data will then be used to implement our method for searching:

```
static bool FindRecord(IComparable *value)
  {
   Record *record=dynamic_cast<Record*>(value);
   if(s_index>=ArraySize(record.m_data) || s_index<0)
     {
      Print("Iindex out of range.");
      return(false);
     }
   return (record.m_data[s_index] == s_value);
  }
```

Here, s\_index is a static variable with a value set by the SetIndex
method, and s\_value is a static variable with a value set by SetValue.

Now, by specifying necessary values of the column number and the value inside, we will easily obtain the reduced version of our list:

```
Record::SetValue(value);
Record::SetIndex(columnIndex);
List<Record*> *new_list = source_list.FindAll(Record::FindRecord);
```

These filtered lists will be displayed in the graphic interface of the Expert Advisor.

There is an option to save these filtered tables in .csv files, if necessary. The file will also be saved in the sand box called **Result.csv**.

IMPORTANT! The same name should be used when saving files. This way, if it is required to save two or more columns, we must save them one by one, and change their names accordingly. Otherwise, we will end up re-writing the same file.

### 4\. Example of analyzing results of EA optimization

This example handles the .xml file of the EA optimization result from the MetaTrader5 terminal. It has a graphic presentation for data obtained during optimization and an option to filter it. All data is divided into two tables:

- "Tester results table" — contains statistical data obtained during testing;
-  "Input parameters table" —  stores all values of input parameters. A limit of ten input parameters is applied to this table. Parameters exceeding the allowed number won't be displayed.

In order to set a filter to one of the tables, we should select the column name and set a range of values.

A graphic interface of the example is shown below:

![](https://c.mql5.com/2/23/R2.png)

This image shows the "Tester results table" with active columns "Pass", "Result", "Profit Factor", "Recovery Factor" and two filters:

1. Values in the "Pass" column should belong to \[0; 10\] range of values ;
2. Values in the "Profit Factor" column should belong to \[0.4; 0.5\] range of values.

### 5\. Brief description of samples from the RegularExpressions library for MQL5

Except for two described EAs, the RegularExpressions library for MQL5 offers 20 examples. They present implementation of various features of regular expressions and this library. They are all located in the Tests.mq5 Expert Advisor:

![](https://c.mql5.com/2/23/tests1_en.png)

We will consider which specific features and options of the library are applied in each example.

01. MatchExamples — shows two possible options of iterating over all matches by creating MatchCollection or using the Match.NextMatch() method.
02. MatchGroups — displays the way of obtaining results of a separate group of captures (Group) and a further operation with them.
03. MatchResult — demonstrates the use of Match.Result(string) method that returns the extension of the specified replacement pattern.
04. RegexConstructor — displays 3 various options of creating the Regex class: based on pattern, pattern with specified parameters, pattern with parameters and value that indicates for how long the method of comparison with pattern must attempt to find a match before timeout expires.
05. RegexEscape — demonstrates operation of the Regex::Escape(string) method.
06. RegexExample — indicates the process of creating regular expressions and their subsequent handling.
07. RegexGetGroupNames — provides the example of using the Regex.GetGroupNames(string) method;
08. RegexGetGroupNumbers — provides the example of using the Regex.GetGroupNumbers(int) method;
09. RegexGroupNameFromNumber — provides the example of using the Regex.GroupNameFromNumber(int) method;
10. RegexIsMatch — provides the example of using all options of the Regex::IsMatch() static method;
11. RegexReplace — provides the example of using main options of the Regex::Replace() static method;
12. RegexSplit — provides the example of using main options of the Regex::Split() static method;
13. Capture — example of operation with a result of a successful capture of expression (Capture).
14. CaptureCollection — example of operation with a set of captures made by a group of captures (CaptureCollection).
15. Group — example of operation with results of a separate Group.
16. GroupCollection — example of operation with a set of captured groups in a single search match (GroupCollection).
17. MatchCollectionItem — creating MatchCollection with the Regex::Matches(string,string) static method;
18. MatchEvaluator — example of creating and using a pointer to the function of the MatchEvaluator type.
19. RegexMatchCollectionCount — demonstration of the MatchCollection.Count() method;
20. RegexOptions — demonstration of the impact of the RegexOptions parameter to handle a regular expression.

The majority of examples have similar functionality and mainly serve for testing how the library works.

### Conclusion

This article briefly describes the features and application of regular expressions. For more detailed information we recommend reading articles available from the following links. Regular expression syntax on .Net has a lot in common with implementation on MQL5, therefore Help information from Microsoft will be relevant at least partially. The same applies to classes from the Internal folder.

### References

1. [http://professorweb.ru/my/csharp/charp\_theory/level4/4\_10.php](https://www.mql5.com/go?link=http://professorweb.ru/my/csharp/charp_theory/level4/4_10.php "http://professorweb.ru/my/csharp/charp_theory/level4/4_10.php")
2. [https://habrahabr.ru/post/115825/](https://www.mql5.com/go?link=https://habrahabr.ru/post/115825/ "https://habrahabr.ru/post/115825/")
3. [https://habrahabr.ru/post/115436/](https://www.mql5.com/go?link=https://habrahabr.ru/post/115436/ "https://habrahabr.ru/post/115436/")
4. [https://msdn.microsoft.com/en-us/library/hs600312.aspx](https://www.mql5.com/go?link=https://docs.microsoft.com/en-us/dotnet/standard/base-types/regular-expressions "https://msdn.microsoft.com/en-us/library/hs600312.aspx")
5. [https://msdn.microsoft.com/en-us/library/ae5bf541.aspx](https://www.mql5.com/go?link=https://msdn.microsoft.com/en-us/library/ae5bf541.aspx "https://msdn.microsoft.com/en-us/library/ae5bf541.aspx")

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/2432](https://www.mql5.com/ru/articles/2432)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/2432.zip "Download all attachments in the single ZIP archive")

[OptimizationResultParsing.mq5](https://www.mql5.com/en/articles/download/2432/optimizationresultparsing.mq5 "Download OptimizationResultParsing.mq5")(2.18 KB)

[OptimizatorResultView.mqh](https://www.mql5.com/en/articles/download/2432/optimizatorresultview.mqh "Download OptimizatorResultView.mqh")(24.45 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

#### Other articles by this author

- [Getting Started with MQL5 Algo Forge](https://www.mql5.com/en/articles/18518)
- [Installing MetaTrader 5 and Other MetaQuotes Apps on HarmonyOS NEXT](https://www.mql5.com/en/articles/18612)
- [MetaTrader 5 on macOS](https://www.mql5.com/en/articles/619)
- [How to earn money by fulfilling traders' orders in the Freelance service](https://www.mql5.com/en/articles/1019)
- [MetaTrader 4 on macOS](https://www.mql5.com/en/articles/1356)
- [Working with ONNX models in float16 and float8 formats](https://www.mql5.com/en/articles/14330)
- [Regression models of the Scikit-learn Library and their export to ONNX](https://www.mql5.com/en/articles/13538)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/90676)**
(3)


![Alexander](https://c.mql5.com/avatar/avatar_na2.png)

**[Alexander](https://www.mql5.com/en/users/kuva)**
\|
5 Jun 2016 at 13:27

I can't run your examples on MT5 build1340, because when compiling TableListView.mqh I get the error "can't open "C:\\Program Files\\MetaTrader 5\\MetaTrader 5\\mql5\\15242\\Include\\Controls\\WndClient.mqh" include file TableListView.mqh 7 11" and "can't open "C:\\Program Files\\MetaTrader 5\\mql5\\15242\\Include\\Controls\\Edit.mqh" include file TableListView.mqh 8 11". Maybe they should be unpacked into the standard MQL5 "Include" folder instead of "MQL5\\15242\\Include" ?

![Andrei Novichkov](https://c.mql5.com/avatar/2016/11/58342001-4AC3.png)

**[Andrei Novichkov](https://www.mql5.com/en/users/andreifx60)**
\|
9 Jun 2016 at 21:04

This is a very interesting topic. It never occurred to me to use regulars in MQL before.


![Rashid Umarov](https://c.mql5.com/avatar/2012/5/4FC60566-2EEC.jpg)

**[Rashid Umarov](https://www.mql5.com/en/users/rosh)**
\|
10 Jun 2016 at 09:48

**Alexander:**

I can't run your examples on my MT5 build1340, because during compilation.

Thanks for the message, fixed


![Using text files for storing input parameters of Expert Advisors, indicators and scripts](https://c.mql5.com/2/23/avatar__3.png)[Using text files for storing input parameters of Expert Advisors, indicators and scripts](https://www.mql5.com/en/articles/2564)

The article describes the application of text files for storing dynamic objects, arrays and other variables used as properties of Expert Advisors, indicators and scripts. The files serve as a convenient addition to the functionality of standard tools offered by MQL languages.

![Creating a trading robot for Moscow Exchange. Where to start?](https://c.mql5.com/2/23/expert-moex-avatar.png)[Creating a trading robot for Moscow Exchange. Where to start?](https://www.mql5.com/en/articles/2513)

Many traders on Moscow Exchange would like to automate their trading algorithms, but they do not know where to start. The MQL5 language offers a huge range of trading functions, and it additionally provides ready classes that help users to make their first steps in algo trading.

![Working with sockets in MQL, or How to become a signal provider](https://c.mql5.com/2/23/server_client_exchange.png)[Working with sockets in MQL, or How to become a signal provider](https://www.mql5.com/en/articles/2599)

Sockets… What in our IT world could possibly exist without them? Dating back to 1982, and hardly changed up to the present time, they smoothly work for us every second. This is the foundation of network, the nerve endings of the Matrix we all live in.

![How to create bots for Telegram in MQL5](https://c.mql5.com/2/22/telegram-avatar.png)[How to create bots for Telegram in MQL5](https://www.mql5.com/en/articles/2355)

This article contains step-by-step instructions for creating bots for Telegram in MQL5. This information may prove useful for users who wish to synchronize their trading robot with a mobile device. There are samples of bots in the article that provide trading signals, search for information on websites, send information about the account balance, quotes and screenshots of charts to you smart phone.

[![](https://www.mql5.com/ff/sh/bhdtjfb1zry09943z2/267b575d2182c180804d340af38ce02c.jpg)\\
Trade from your iPhone or Android device\\
\\
You only need an internet connection to use the new powerful MetaTrader 5 Web terminal\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=wtigumvtenarnsocpyfoqnanxrilnbxx&s=ec8c539e52b83881ff2d16eaff6913b25803952eb277cac55f670a102b2edc1f&uid=&ref=https://www.mql5.com/en/articles/2432&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5083341601893259707)

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