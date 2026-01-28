---
title: A DLL for MQL5 in 10 Minutes (Part II): Creating with Visual Studio 2017
url: https://www.mql5.com/en/articles/5798
categories: Trading Systems, Integration
relevance_score: 6
scraped_at: 2026-01-23T11:49:00.759467
---

[![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/01.png)![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/02.png)Powerful analytics for traders of any levelAll the necessary trading reports for beginners and professionals](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/454106&a=muccpajyfystoakuukdobwigjejzmpqn&s=52daad60fa795e635264e6f94898f05493bca3b5124d4cca8eb7e82333c2ef12&uid=&ref=https://www.mql5.com/en/articles/5798&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5062740451955484666)

MetaTrader 5 / Trading systems


### Introduction

This article was created as a development of ideas from the earlier published [article](https://www.mql5.com/en/articles/18) related to DLL creation with Visual Studio 2005/2008. The original basic article has not lost its relevance and thus if you are interested in this topic, be sure to read the first article. However much time has passed since then, so the current Visual Studio 2017 features an updated interface. The MetaTrader 5 platform has also acquired new features. Obviously, there is a need to update the information and consider some new features. In this article, we will pass all steps from the development of the DLL project in Visual Studio 2017 to connection of a ready DLL to the terminal and its use.

The article is intended for beginners who want to learn how to create and connect C++ libraries to the terminal.

### Why connect DLL to the terminal?

Some developers believe that no libraries should be connected to the terminal, because there are no tasks for which such connection is necessary, while the required functionality can be implemented using MQL means. This opinion is true to some extent. There are very few tasks which require libraries. Most of the required tasks can be solved using MQL tools. Moreover, when connecting a library, one should understand that the Expert Advisor or indicator, by which this library is used, will not be able to operate without that DLL. If you need to transfer such an application to a third party, you will have to transfer two files, i.e. the application itself and the library. Sometimes this can be very inconvenient or even impossible. Another weak point is that libraries can be unsafe and can contain harmful code.

However, libraries have their advantages, which definitely outweigh the disadvantages,. Examples:

- Libraries can help solve the problems which cannot be solved by MQL means. For example for mailing lists, when you need to send emails with attachments. DLLs can write to Skype. Etc.
- Some tasks which can be implemented in MQL, can be performed faster and more efficiently with the libraries. These include HTML page parsing and use of regular expressions.

If you want to solve such complex tasks, you should master your skills and properly learn how to create and connect libraries.

We have considered the "Pros" and "Cons" of DLL use in our projects. Now let us consider step by step the process of DLL creation with Visual Studio 2017.

### Creating a simple DLL

The whole process was already described in the [original article](https://www.mql5.com/en/articles/18). Now we will repeat it taking into account software updates and changes.

Open Visual Studio 2017 and navigate to File -> New -> Project. In the left part of the new project window, expand the Visual C++ list and select Windows Desktop from it. Select the Windows Desktop Wizard line in the middle part. Using entry fields at the bottom part, you can edit the project name (it is advisable to set your own meaningful name) and set project location (it is recommended to keep as suggested). Click OK and proceed to the next window:

![](https://c.mql5.com/2/35/cmpl1.png)

Select Dynamic Link Library (.dll) from the drop-down list and check "Export Symbols". Checking this item is optional, but beginners are recommended to do so. In this case, a demo code will be added to the project files. This code can be viewed and then deleted or commented. A click on "OK" creates project files, which can then be edited. However, first we need to consider the project settings. Firstly, remember that MetaTrader 5 only works with 64-bit libraries. If you try to connect a 32-bit DLL, you will receive the following messages:

' _E:\\...\\MQL5\\Libraries\\Project2.dll' is not 64-bit version_

_Cannot load 'E:\\MetaTrader 5\\MQL5\\Libraries\\Project2.dll' \[193\]_

Thus you will not be able to use this library.

An opposite limitation applies to MetaTrader 4 DLLs: only 32-bit libraries are allowed, while 64-bit DLLs cannot be connected. Keep this in mind and create a suitable version for your platform.

Now proceed to project settings. From the "Project" menu select "Name Properties...",  where "Name" is the project name specified by the developer at the creation stage. This opens a window with a variety of different settings. First of all you should enable Unicode. In the left part of the window select "General". In the right part select the header line in the first column: "Character Set". A drop-down list will become available in the second column. Select "Use Unicode Character Set" from that list. In some cases Unicode support is not required. We will discuss such cases later.

Another very useful (but not necessary) change in project properties: copy the finished library into the "Library" folder of the terminal. In the original article, this was done by changing the "Output Directory" parameter, which is in the same window of the project's "General" element. No need to do this in Visual Studio 2017. Do not change this parameter. However, pay attention to the "Build Events" item: you should select its "Post Build Events" sub-element. The "Command Line" parameter will appear in the first column of the right window. Select it to open an editable list in the second column. This should be a list of actions which Visual Studio 2017 will perform after building the library. Add the following line to this list:

_xcopy "$(TargetDir)$(TargetFileName)" "E:\\...\\MQL5\\Libraries\\" /s /i /y_

Here instead of ... you should specify the full path to the appropriate terminal folder. After successful library building, your DLL will be copied to the specified folder. All files in "Output Directory" will be preserved in this case, which can be important for further version-controlled development.

The last and very important project setup step is the following. Imagine that the library is already built and includes one function which can be used by the terminal. Suppose this function has the following simple prototype:

```
int fnExport(wchar_t* t);
```

This function can be called from the terminal script as follows:

```
#import "Project2.dll"
int fnExport(string str);
#import
```

However, the following error message will be returned in this case:

![](https://c.mql5.com/2/35/pic2.png)

How to solve this situation? During library code generation, Visual Studio 2017 formed the following macro:

```
#ifdef PROJECT2_EXPORTS
#define PROJECT2_API __declspec(dllexport)
#else
#define PROJECT2_API __declspec(dllimport)
#endif
```

The full prototype of the desired function looks as follows:

```
PROJECT2_API int fnExport(wchar_t* t);
```

View the export table after the library compilation:

![](https://c.mql5.com/2/35/pic3__1.png)

To view it, select the library file in the Total Commander window and press F3. Pay attention to the name of the exported function. Now let's edit the above macro (this is how it was done in the original article):

```
#ifdef PROJECT2_EXPORTS
#define PROJECT2_API extern "C" __declspec(dllexport)
#else
#define PROJECT2_API __declspec(dllimport)
#endif
```

Here

```
extern "C"
```

means the use of a simple function signature generation (in C language style) when receiving object files. In particular, this prohibits the C++ compiler from "decorating" of the function name with additional characters when exporting to a DLL. Re-compile and view the export table:

![](https://c.mql5.com/2/35/pic4.png)

Changes in the export table are obvious and no error occurs now when calling the function from the script. However, the method has a disadvantage: you have to edit the script which has been created by the compiler. There is a safer way to perform the same, which is however a bit longer:

### Definition file

This is a plain text file with the .def extension, usually with a name matching the project name. In our case this will be the file Project2.def. The file is created in a regular notepad. Never use Word or similar editors. The file contents will be as follows:

```
; PROJECT2.def : Declares the module parameters for the DLL.

LIBRARY      "PROJECT2"
DESCRIPTION  'PROJECT2 Windows Dynamic Link Library'

EXPORTS
    ; Explicit exports can go here
        fnExport @1
        fnExport2 @2
        fnExport3 @3
        ....
```

The header is followed by the list of exported functions. Characters @1, @2 etc. indicate the desired order of functions in the library. Save this file in the project folder.

Now let's create this file and connect to the project. In the left part of the project properties window, select the "Linker" element and its "Input" sub-element. Then select the "Module Definition File" parameter in the right part. As in previous cases, get access to the editable list and add the file name: "Project2.def". Click OK and repeat compilation. The result is the same as in the previous screenshot. The name is not decorated and no errors are encountered when the function is called by the script. We have analyzed the project settings. Now let's start writing the library code.

### Creating a library and DllMain

The original article provides a comprehensive description of issues related to data exchange and various functions calls from DLLs and thus we will not dwell on this. Let's create a simple code in the library to view some specific features:

1\. Add the following function to export (do not forget to edit the definition file):

```
PROJECT2_API int fnExport1(void) {
        return GetSomeParam();
}
```

2\. Create and add the Header1.h header file to the project and add another function to it:

```
const int GetSomeParam();
```

3\. Edit the dllmain.cpp file:

```
#include "stdafx.h"
#include "Header1.h"

int iParam;

BOOL APIENTRY DllMain( HMODULE hModule,
                       DWORD  ul_reason_for_call,
                       LPVOID lpReserved
                     )
{
    switch (ul_reason_for_call)
    {
    case DLL_PROCESS_ATTACH:
                iParam = 7;
                break;
    case DLL_THREAD_ATTACH:
                iParam += 1;
                break;
    case DLL_THREAD_DETACH:
    case DLL_PROCESS_DETACH:
        break;
    }
    return TRUE;
}

const int GetSomeParam() {
        return iParam;
}
```

The code purpose should be clear: a variable is added to the library. Its value is calculated in the DllMain function and is available using the fnExport1 function. Let's call the function in the script:

```
#import "Project2.dll"
int fnExport1(void);
#import
...
void OnStart() {
Print("fnExport1: ",fnExport1() );
```

The following entry is output:

_fnExport1: 7_

It means that this part of the DllMain code is not executed:

```
    case DLL_THREAD_ATTACH:
                iParam += 1;
                break;
```

Is this important? In my opinion, it is crucial, because if a developer adds here part of library initialization code, expecting it to be executed when connecting the library to the stream, the operation will fail. However no error will be returned and thus it will be difficult to spot issue.

### Strings

Operation with strings is described in the original article. This work is not difficult. However I would like to clarify the following specific point.

Let's create a simple function in the library (and edit the definition file):

```
PROJECT2_API void SamplesW(wchar_t* pChar) {
        size_t len = wcslen(pChar);
        wcscpy_s(pChar + len, 255, L" Hello from C++");
}
```

Call this function in a script:

```
#import "Project2.dll"
void SamplesW(string& pChar);
#import

void OnStart() {

string t = "Hello from MQL5";
SamplesW(t);
Print("SamplesW(): ", t);
```

The following expected message will be received:

_SamplesW(): Hello from MQL5 Hello from C++_

Now edit the function call:

```
#import "Project2.dll"
void SamplesW(string& pChar);
#import

void OnStart() {

string t;
SamplesW(t);
Print("SamplesW(): ", t);
```

This time we get an error message:

_Access violation at 0x00007FF96B322B1F read to 0x0000000000000008_

Initialize the string which is passed to the library function and repeat the script execution:

```
string t="";
```

No error message is received so we get the expected output:

_SamplesW():  Hello from C++_

The above code suggests the following: the strings which are passed to the functions exported by the library must be initialized!

It's time to get back to the use of Unicode. If you do not plan to pass strings to DLL (as is shown in the last example), then Unicode support is not required. However I recommend enabling Unicode support in any case, because exported function signatures can change, new functions can be added and the developer can forget about the absence of Unicode support.

Symbol arrays are passed and received in a common way, which was described in the original article. Therefore we do not need to discuss them again.

### Structures

Let's define the simplest structure in the library and in the script:

```
//In dll:
typedef struct E_STRUCT {
        int val1;
        int val2;
}ESTRUCT, *PESTRUCT;

//In MQL script:
struct ESTRUCT {
   int val1;
   int val2;
};
```

Add a function for working with the structure to the library:

```
PROJECT2_API void SamplesStruct(PESTRUCT s) {
        int t;
        t = s->val2;
        s->val2 = s->val1;
        s->val1 = t;
}
```

As can be seen from the code, the function simply swaps its own fields.

Call this function from the script:

```
#import "Project2.dll"
void SamplesStruct(ESTRUCT& s);
#import
....
ESTRUCT e;
e.val1 = 1;
e.val2 = 2;
SamplesStruct(e);
Print("SamplesStruct: val1: ",e.val1," val2: ",e.val2);
```

Run the script and get the expected result:

_SamplesStruct: val1: 2 val2: 1_

The object was passed to the called function by reference. The function processed the object and returned it to the calling code.

However, we often need more complex structures. Let's complicate the task: add one more field having a different type to the structure:

```
typedef struct E_STRUCT1 {
        int val1;
        char cval;
        int val2;
}ESTRUCT1, *PESTRUCT1;
```

Also add a function to work with it:

```
PROJECT2_API void SamplesStruct1(PESTRUCT1 s) {
        int t;
        t = s->val2;
        s->val2 = s->val1;
        s->val1 = t;
        s->cval = 'A';
}
```

Like in the previous case, the function swaps its fields of type int and assigns a value to a 'char' type field. Call this function in the script (in exactly the same way as the previous function). However, this time the result is as follows:

_SamplesStruct1: val1: -2144992512 cval: A val2: 33554435_

Structure fields of type int contain wrong data. It is not an exception, but is random incorrect data. What happened? The reason is in the [alignment](https://www.mql5.com/en/docs/basis/types/classes#pack)! Alignment is not a very complicated concept. The document section [pack](https://www.mql5.com/en/docs/basis/types/classes#pack) related to structures provides a detailed description of alignment. Visual Studio C++ also provides comprehensive materials related to alignment.

In our example, the error occurred because the library and the script have different alignments. There are two ways to solve the problem:

1. Specify a new alignment in the script. This can be done using the **pack(n)** attribute. Let's try to align the structure according to the largest field, i.e. **int**:


```
struct ESTRUCT1 pack(sizeof(int)){
           int val1;
           char cval;
           int val2;
};
```



    Let us repeat the script execution. The entry in the log has changed: _SamplesStruct1: val1: 3 cval: A val2: 2_ . Thus the error has been solved.

2. Specify new library alignment. The default alignment of MQL structures is pack(1). Apply the same to the library:


```
#pragma pack(1)
typedef struct E_STRUCT1 {
           int val1;
           char cval;
           int val2;
}ESTRUCT1, *PESTRUCT1;
#pragma pack()
```



    Build the library and run the script: the result is correct and is the same as it was with the first method.

Check one more thing. What happens if a structure contains methods in addition to the data fields? This is quite possible. Also programmers can add a constructor (which is not a method), a destructor or something else. Let us check these cases in the following library structure:

```
#pragma pack(1)
typedef struct E_STRUCT2 {
        E_STRUCT2() {
                val2 = 15;
        }
        int val1;
        char cval;
        int val2;
}ESTRUCT2, *PESTRUCT2;
#pragma pack()
```

The structure will be used by the following function:

```
PROJECT2_API void SamplesStruct2(PESTRUCT2 s) {
        int t;
        t = s->val2;
        s->val2 = s->val1;
        s->val1 = t;
        s->cval = 'B';
}
```

Make the appropriate changes in the script:

```
struct ESTRUCT2 pack(1){
        ESTRUCT2 () {
           val1 = -1;
           val2 = 10;
        }
        int val1;
        char cval;
        int f() { int val3 = val1 + val2; return (val3);}
        int val2;
};

#import "Project2.dll"
void SamplesStruct2(ESTRUCT2& s);
#import
...
ESTRUCT2 e2;
e2.val1 = 4;
e2.val2 = 5;
SamplesStruct2(e2);
t = CharToString(e2.cval);
Print("SamplesStruct2: val1: ",e2.val1," cval: ",t," val2: ",e2.val2);
```

Note that the f() method has been added to the structure thus providing more differences from the structure in the library. Run the script. The following entry is written to the journal: SamplesStruct2:  _val1: 5 cval: B val2: 4_ The execution is correct! The presence of a constructor and an additional method in our structure did not affect the result.

The last experiment. Remove the constructor and the method from the structure in the script, while leaving only data fields. The structure in the library remains unchanged. Again the script execution generates a correct result. This enables the drawing of a final conclusion: the presence of additional methods in structures does not affect the result.

This library project for Visual Studio 2017 and a MetaTrader 5 script are attached below.

### What you cannot do

There are certain limitations on operations with DLLs, which are described in the related documentation. We will not repeat them here. Here is an example:

```
struct BAD_STRUCT {
   string simple_str;
};
```

This structure cannot be passed to a DLL. This is a string wrapped in a structure. More complex objects cannot be passed to a DLL without getting an exception.

### What to do, if anything cannot be done

Often we need to pass to DLLs the objects, which are not allowed. These include structures with [dynamic objects](https://www.mql5.com/en/docs/basis/types/dynamic_array), gear array, etc. What can be done in this case? Without having access to the library code, this solution cannot be used. Access to the code can help solve the issue.

We will not consider changes in data design since we should try to solve it using available means and to avoid an exception. Some clarification is needed. The article is not intended for experienced users, therefore we will only outline possible solutions to the problem.

1. The use of the [StructToCharArray()](https://www.mql5.com/en/docs/convert/structtochararray) function. This seems to be a nice opportunity, which enables the use of the following code in the script:



```
struct Str
     {
        ...
     };

Str s;
uchar ch[];
StructToCharArray(s,ch);

SomeExportFunc(ch);
```



    Code in the cpp library file:


```
#pragma pack(1)
typedef struct D_a {
...
}Da, *PDa;
#pragma pack()

void SomeExportFunc(char* pA)
     {
           PDa = (PDa)pA;
           ......
     }
```



    In addition to security and quality issues, the very idea is useless: StructToCharArray() only works with [POD structures](https://www.mql5.com/en/docs/basis/types/classes#simple_structure), which can be passed to libraries without additional conversions. I haven't tested this function operation on actual code.

2. Create your own packer/unpacker of structures into an object that can be passed to the library. This method is possible but is very complicated and resource and time intensive. However, this method suggests a completely acceptable solution:

3. All objects which cannot be passed to the library directly should be packaged into a JSON string in the script and unpacked into structures in the library. And vice-versa. There are available tools for that: Parsers for JSON are available for C++, C# and [for MQL](https://www.mql5.com/en/code/13663). This method can be used if you are ready to spend some time for packing/unpacking the objects. However, apart from obvious time losses there are advantages. The method enables operation with very complex structures (as well as other objects). Moreover, you can refine existing packer/unpacker instead of writing one from scratch.

so keep in mind that there is a possibility to pass (and receive) a complex object to the library.

### Practical use

Now let us try to create a useful library. This library will send emails. Please note the following moments:

- The library cannot be used for spam emailing.
- The library can send emails from the address and server other than those specified in terminal settings. Moreover, the use of email can be disabled in terminal settings, but this will not affect the library operation.


And the last thing. Most of the C++ code is not mine, but was downloaded from Microsoft forums. This is and old and proven example, its variants are also available on VBS.

Let's start. Create a project in Visual Studio 2017 and change its settings as described at the article beginning. Create a definition file and connect it to the project. There is only one exported function:

```
SENDSOMEMAIL_API bool  SendSomeMail(LPCWSTR addr_from,
        LPCWSTR addr_to,
        LPCWSTR subject,
        LPCWSTR text_body,

        LPCWSTR smtp_server,
        LPCWSTR smtp_user,
        LPCWSTR smtp_password);
```

The meaning of its arguments is clear, so here is a brief explanation:

- addr\_from, addr\_to — sender and recipient email addresses.
- subject, text\_body — subject and email body.
- smtp\_server, smtp\_user, smtp\_password — the SMTP server address, user login and password for the server.

Pay attention to the following moments:

- As seen from the description of the arguments, to send mail you need to have an account on the mail server and to know its address. Thus the sender cannot be anonymous.

- The port number is hard coded in the library. This is the standard port number 25.

- The library receives the required data, connects to the server and sends email to it. In one call, an email can be sent only to one address. To send more, repeat the function call with the new address.

I will not provide the C++ code here. This code as well as the entire project are available below in the attached SendSomeMail.zip project. The used CDO object has many features and should be used for further library development and improvement.

In addition to this project, let us write a simple script to call the library function (it is located in the attached SendSomeMail.mq5 file):

```
#import "SendSomeMail.dll"
bool  SendSomeMail(string addr_from,string addr_to,string subject,string text_body,string smtp_server,string smtp_user,string smtp_password);
#import

//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
   bool b = SendSomeMail("XXX@XXX.XX", "XXXXXX@XXXXX.XX", "hello", "hello from me to you","smtp.XXX.XX", "XXXX@XXXX.XXX", "XXXXXXXXX");
   Print("Send mail: ", b);

  }
```

Add your own account details instead of X characters. Thus the development is complete. Add your own details, make any additions to the code which you may need and the library will be ready to use.

### Conclusion

Using the [original article](https://www.mql5.com/en/articles/18) and taking account updates contained in this article, anyone can quickly master the basics and move on to more complex and interesting projects.

I would like to dwell on one more interesting fact, which can be very important under specific situations. How to protect the dll code? The standard solution is to use a packer. There are a lot of different packers, many of which can provide a good protection level. I have two packers: Themida 2.4.6.0 and VMProtect Ultimate v. 3.0.9 . Let's use them to pack our first simple Project2.dll in two variants for each packer. After that, call the exported functions in the terminal using the existing script. Everything works fine! The terminal can work with such libraries. However normal operation of libraries protected by other packers is not guaranteed. Project2.dll packed in two methods is available in Project2\_Pack.zip

That's about it. Good luck in further developments.

### Programs used in the article

| # | Name | Type | Description |
| --- | --- | --- | --- |
| 1 | Project2.zip | Archive | Simple DLL project |
| 2 | Project2.mq5 | Script | Script for operations with DLL |
| 3 | SendSomeMail.zip | Archive | Emails ending DLL project |
| 4 | SendSomeMail.mq5 | Script | Script for operation with the SendSomeMail library dll |
| 5 | Project2\_Pack.zip | Archive | Project2.dll protected with Themida and VMProtect. |

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/5798](https://www.mql5.com/ru/articles/5798)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/5798.zip "Download all attachments in the single ZIP archive")

[Project2.mq5](https://www.mql5.com/en/articles/download/5798/project2.mq5 "Download Project2.mq5")(3.42 KB)

[SendSomeMail.mq5](https://www.mql5.com/en/articles/download/5798/sendsomemail.mq5 "Download SendSomeMail.mq5")(1.15 KB)

[SendSomeMail.zip](https://www.mql5.com/en/articles/download/5798/sendsomemail.zip "Download SendSomeMail.zip")(16.62 KB)

[Project2\_Pack.zip](https://www.mql5.com/en/articles/download/5798/project2_pack.zip "Download Project2_Pack.zip")(4645.25 KB)

[Project2.zip](https://www.mql5.com/en/articles/download/5798/project2.zip "Download Project2.zip")(18.65 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [MVC design pattern and its application (Part 2): Diagram of interaction between the three components](https://www.mql5.com/en/articles/10249)
- [MVC design pattern and its possible application](https://www.mql5.com/en/articles/9168)
- [Using cryptography with external applications](https://www.mql5.com/en/articles/8093)
- [Building an Expert Advisor using separate modules](https://www.mql5.com/en/articles/7318)
- [Parsing HTML with curl](https://www.mql5.com/en/articles/7144)
- [Arranging a mailing campaign by means of Google services](https://www.mql5.com/en/articles/6975)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/313358)**
(10)


![Andrei Novichkov](https://c.mql5.com/avatar/2016/11/58342001-4AC3.png)

**[Andrei Novichkov](https://www.mql5.com/en/users/andreifx60)**
\|
9 Apr 2019 at 10:58

**Maxim Kuznetsov:**

I'm not arguing with you ;-)

so - some notes that I remembered, maybe it will be useful to someone.

without arguing with the article

what kind of ["novice programmers](https://www.mql5.com/en/articles/447 "Article: Quick Dive into MQL5")" are there at the junction of two languages ?

PS/ by the way, you have a lot of memory there.

What's with the memory, did I mess up somewhere?

Beginner programmer is a stretching concept ) I am definitely a beginner in Python ) Or in java script. And many other things I can call myself a beginner in. Here too, what kind of situation is there, if a person hasn't made libraries before, but has been doing CAD for twenty years, or writing plugins for Adobe programs? Of course, he is a beginner in a new field, but experienced in his old one. Anyway, it's all right, this terminology is not so important here.

![Genaro Cancino](https://c.mql5.com/avatar/2020/2/5E3CF4CF-0561.jpg)

**[Genaro Cancino](https://www.mql5.com/en/users/gcancino)**
\|
26 Jul 2023 at 18:13

Can i create custom menus within Metatrader with a c++ dll?


![Andrei Novichkov](https://c.mql5.com/avatar/2016/11/58342001-4AC3.png)

**[Andrei Novichkov](https://www.mql5.com/en/users/andreifx60)**
\|
26 Jul 2023 at 19:11

No)


![Ivan Titov](https://c.mql5.com/avatar/2024/9/66d71f0c-3796.png)

**[Ivan Titov](https://www.mql5.com/en/users/goldrat)**
\|
20 Apr 2024 at 06:47

Thanks for the article. Is there a sequel planned for VS 2022 with the backlog of changes?


![Andrei Novichkov](https://c.mql5.com/avatar/2016/11/58342001-4AC3.png)

**[Andrei Novichkov](https://www.mql5.com/en/users/andreifx60)**
\|
20 Apr 2024 at 13:20

There are no plans for now


![Developing a cross-platform grider EA](https://c.mql5.com/2/35/mql5_ea_adviser_grid.png)[Developing a cross-platform grider EA](https://www.mql5.com/en/articles/5596)

In this article, we will learn how to create Expert Advisors (EAs) working both in MetaTrader 4 and MetaTrader 5. To do this, we are going to develop an EA constructing order grids. Griders are EAs that place several limit orders above the current price and the same number of limit orders below it simultaneously.

![Using MATLAB 2018 computational capabilities in MetaTrader 5](https://c.mql5.com/2/35/ext_infin2.png)[Using MATLAB 2018 computational capabilities in MetaTrader 5](https://www.mql5.com/en/articles/5572)

After the upgrade of the MATLAB package in 2015, it is necessary to consider a modern way of creating DLL libraries. The article uses a sample predictive indicator to illustrate the peculiarities of linking MetaTrader 5 and MATLAB using modern 64-bit versions of the platforms, which are utilized nowadays. With the entire sequence of connecting MATLAB considered, MQL5 developers will be able to create applications with advanced computational capabilities much faster, avoiding «pitfalls».

![Studying candlestick analysis techniques (part IV): Updates and additions to Pattern Analyzer](https://c.mql5.com/2/35/Logo__3.png)[Studying candlestick analysis techniques (part IV): Updates and additions to Pattern Analyzer](https://www.mql5.com/en/articles/6301)

The article presents a new version of the Pattern Analyzer application. This version provides bug fixes and new features, as well as the revised user interface. Comments and suggestions from previous article were taken into account when developing the new version. The resulting application is described in this article.

![MTF indicators as the technical analysis tool](https://c.mql5.com/2/35/mtf-avatar.png)[MTF indicators as the technical analysis tool](https://www.mql5.com/en/articles/2837)

Most of traders agree that the current market state analysis starts with the evaluation of higher chart timeframes. The analysis is performed downwards to lower timeframes until the one, at which deals are performed. This analysis method seems to be a mandatory part of professional approach for successful trading. In this article, we will discuss multi-timeframe indicators and their creation ways, as well as we will provide MQL5 code examples. In addition to the general evaluation of advantages and disadvantages, we will propose a new indicator approach using the MTF mode.

[![](https://www.mql5.com/ff/sh/6xjc81sb5f2g45z9z2/01.png)Follow MQL5.community on social mediaWe publish the best technical materials from experts – free from advertising and irrelevant contentLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=yexgeaiatphxecqagtoxizolvboismyb&s=4e531fd1f983c26570e2dac7588b735354f2f9e0aea561427c030e4a1d2f060b&uid=&ref=https://www.mql5.com/en/articles/5798&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5062740451955484666)

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