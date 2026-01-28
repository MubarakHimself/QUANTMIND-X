---
title: How to Exchange Data: A DLL for MQL5 in 10 Minutes
url: https://www.mql5.com/en/articles/18
categories: Integration
relevance_score: 3
scraped_at: 2026-01-23T21:22:35.254374
---

[![](https://www.mql5.com/ff/si/6pp0j40fqxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=luckhiizjxvmvgigcufevttapwwrwbld&s=08cd1d929f27358481aded3c1c5f4e75a9bd5f52c477127afef2a5c532aec5c5&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=kggcbnjgoaliadcmiqnryclsbzfvsmpk&ssn=1769192554507362296&ssn_dr=0&ssn_sr=0&fv_date=1769192554&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F18&back_ref=https%3A%2F%2Fwww.google.com%2F&title=How%20to%20Exchange%20Data%3A%20A%20DLL%20for%20MQL5%20in%2010%20Minutes%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176919255438956066&fz_uniq=5071786683873046122&sv=2552)

MetaTrader 5 / Examples


As a matter of fact, there are not many developers who remember exactly how to write a simple DLL library and what are features of binding different systems.

Using several examples, I will try to show the entire process of the simple DLL's creation in 10 minutes, as well as to discuss some technical details of our binding implementation. We will use Visual Studio 2005/2008; its Express versions are free and can be downloaded from the [Microsoft website.](https://www.mql5.com/go?link=https://www.visualstudio.com/ "http://www.microsoft.com/visualstudio/en-us/")

### 1\. Creating a DLL project in C++ in Visual Studio 2005/2008

Run the Win32 Application Wizard using the ' **File -> New**' menu, select the project type as ' **Visual C++**', choose ' **Win32 Console Application**' template and define the project name (for example, ' **MQL5DLLSamples**'). Select a root directory for storing project ' **Location**', instead of the default offered one, disable the checkbox of ' **Create directory for solution**' and click ' **OK**':

![](https://c.mql5.com/2/0/dll_vs2008_wizard_1.png)

Fig. 1. Win32 Application Wizard, DLL project creation

On the next step press ' **Next**' to go to the settings page:

![](https://c.mql5.com/2/0/dll_vs2008_wizard_2.png)

Fig. 2. Win32 Application Wizard, project settings

On the final page, select the ' **DLL**' application type, leaving other fields empty as they are, and click ' **Finish**'. Don't set the ' **Export symbols**' option, if you don't want to remove the demonstration code added automatically:

![](https://c.mql5.com/2/0/dll_vs2008_wizard_3.png)

Fig. 3. Win32 Application Wizard, Application settings

As a result you will have an empty project:

![](https://c.mql5.com/2/0/dll_vs2008_project_1.png)

Fig. 4. The empty DLL project prepared by Wizard

To simplify testing, it's better to specify in ' **Output Directory**' options the output of DLL files directly to ' **...\\MQL5\\Libraries**' of the client terminal - further, it will save you much time:

![](https://c.mql5.com/2/0/dll_vs2008_settings_2.png)

Fig. 5. DLL output directory

### 2\. Preparing to Add Functions

Add ' **\_DLLAPI**' macro at end of the **stdafx.h** file, so that you can conveniently and easily describe exported functions:

```
//+------------------------------------------------------------------+
//|                                                 MQL5 DLL Samples |
//|                   Copyright 2001-2010, MetaQuotes Software Corp. |
//|                                        https://www.metaquotes.net |
//+------------------------------------------------------------------+
#pragma once

#include "targetver.h"

#define WIN32_LEAN_AND_MEAN             // Exclude rarely-used stuff from Windows headers
#include <windows.h>

//---
#define _DLLAPI extern "C" __declspec(dllexport)
//+------------------------------------------------------------------+
```

The DLL imported functions calls in MQL5 should have the [stdcall and cdecl calling convention](https://www.mql5.com/en/docs/runtime/imports). Though stdcall and cdecl differ in the ways of parameter extracting from a stack, the MQL5 runtime environment can safely use both versions because of the special wrapper of DLL calls.

The C++ compiler uses
\_\_cdecl calling by default, but I recommend explicitly specifying the \_\_stdcall mode for exported functions.

A correctly written export function must have the following form:

```
_DLLAPI int __stdcall fnCalculateSpeed(int &res1,double &res2)
  {
   return(0);
  }
```

In a MQL5 program, the function should be defined and called as follows:

```
#import "MQL5DLLSamples.dll"
int  fnCalculateSpeed(int &res1,double &res2);
#import

//--- call
   speed=fnCalculateSpeed(res_int,res_double);
```

After the project compilation, this stdcall will be displayed in the export table as **\_fnCalculateSpeed@8**, where the compiler adds an underscore and number of
bytes, transmitted through the stack. Such a decoration allows to better control
the security of DLL functions calls due to the fact that the
caller knows exactly how many (but not the type of!) data that should be placed in
the stack.

If the final size of the parameter block has an error in the DLL
function import description, the function won't be called, and the new message
will appear in the journal: ' **Cannot find 'fnCrashTestParametersStdCall' in 'MQL5DLLSamples.dll**'. In such cases it is necessary to carefully check all the parameters both in the function prototype and in the DLL source.

The search for the simplified
description without decoration is used for compatibility in case the export table does not contain the full function name. Names like **fnCalculateSpeed** are created if functions are defined in the \_\_cdecl format.

```
_DLLAPI int fnCalculateSpeed(int &res1,double &res2)
  {
   return(0);
  }
```

### 3\. Methods to Pass Parameters and Exchange Data

Let's consider several variants of passed parameters:

1. **Receiving and passing of simple variables**

The case of simple variables is easy - they can be passed by value or by reference using &.



```
_DLLAPI int __stdcall fnCalculateSpeed(int &res1,double &res2)
     {
      int    res_int=0;
      double res_double=0.0;
      int    start=GetTickCount();
//--- simple math calculations
      for(int i=0;i<=10000000;i++)
        {
         res_int+=i*i;
         res_int++;
         res_double+=i*i;
         res_double++;
        }
//--- set calculation results
      res1=res_int;
      res2=res_double;
//--- return calculation time
      return(GetTickCount()-start);
     }

```

Call from MQL5:



```
#import "MQL5DLLSamples.dll"
int  fnCalculateSpeed(int &res1,double &res2);
#import

//--- calling the function for calculations
      int    speed=0;
      int    res_int=0;
      double res_double=0.0;

      speed=fnCalculateSpeed(res_int,res_double);
      Print("Time ",speed," msec, int: ",res_int," double: ",res_double);
```

The output is:



```
MQL5DLL Test (GBPUSD,M1) 19:56:42 Time  16  msec, int:  -752584127  double:  17247836076609
```

2. **Receiving and passing of an array with elements filling**


Unlike other MQL5 programs, array passing is performed through the direct reference to the data buffer without access to proprietary information about the dimensions and sizes. That's why the array dimension and size should be passed separately.




```
_DLLAPI void __stdcall fnFillArray(int *arr,const int arr_size)
     {
//--- check for the input parameters
      if(arr==NULL || arr_size<1) return;
//--- fill array with values
      for(int i=0;i<arr_size;i++) arr[i]=i;
     }

```

Call from MQL5:



```
#import "MQL5DLLSamples.dll"
void fnFillArray(int &arr[],int arr_size);
#import

//--- call for the array filling
      int    arr[];
      string result="Array: ";
      ArrayResize(arr,10);

      fnFillArray(arr,ArraySize(arr));
      for(int i=0;i<ArraySize(arr);i++) result=result+IntegerToString(arr[i])+" ";
      Print(result);
```

The output is:



```
MQL5DLL Test (GBPUSD,M1) 20:31:12 Array: 0 1 2 3 4 5 6 7 8 9
```

3. **Passing and modification of strings**

The unicode strings are passed using direct references to its buffer addresses without passing of any additional information.



```
_DLLAPI void fnReplaceString(wchar_t *text,wchar_t *from,wchar_t *to)
     {
      wchar_t *cp;
//--- parameters check
      if(text==NULL || from==NULL || to==NULL) return;
      if(wcslen(from)!=wcslen(to))             return;
//--- search for substring
      if((cp=wcsstr(text,from))==NULL)         return;
//--- replace it
      memcpy(cp,to,wcslen(to)*sizeof(wchar_t));
     }
```

Call from MQL5:



```
#import "MQL5DLLSamples.dll"
void fnReplaceString(string text,string from,string to);
#import

//--- modify the string
      string text="A quick brown fox jumps over the lazy dog";

      fnReplaceString(text,"fox","cat");
      Print("Replace: ",text);
```

The result is:



```
MQL5DLL Test (GBPUSD,M1) 19:56:42 Replace:  A quick brown fox jumps over the lazy dog
```

It's turned out that the line hadn't changed! This is a common mistake of newbies when they transmit copies of objects (a string is an object), instead of referring to them. The copy of the string 'text' has been automatically created that has been modified in the DLL, and then it has been removed automatically without affecting the original.



To remedy this situation, it's necessary to pass a string by reference. To do
it, simply modify the block of importing by adding & to the "text" parameter:



```
#import "MQL5DLLSamples.dll"
void fnReplaceString(string &text,string from,string to);
#import
```

After compilation and start we will get the right result:



```
MQL5DLL Test (GBPUSD,M1) 19:58:31 Replace:  A quick brown cat jumps over the lazy dog
```


### 4\. Catching of exceptions in DLL functions

To prevent the terminal crushes, each DLL call is protected automatically by Unhandled Exception Wrapping. This mechanism allows protecting from the most of standard errors (memory access errors, division by zero, etc.)

To see how the mechanism works, let's create the following code:

```
_DLLAPI void __stdcall fnCrashTest(int *arr)
  {
//--- wait for receipt of a zero reference to call the exception
   *arr=0;
  }
```

and call it from the client terminal:

```
#import "MQL5DLLSamples.dll"
void fnCrashTest(int arr);
#import

//--- call for the crash (the execution environment will catch the exception and prevent the client terminal crush)
   fnCrashTest(NULL);
   Print("You won't see this text!");
//---
```

As s result, it will try to write to the zero address and generate an exception. The client terminal will catch it, log it to the journal and continue its work:

```
MQL5DLL Test (GBPUSD,M1) 20:31:12 Access violation write to 0x00000000
```

### 5\. DLL calls wrapper and loss of speed on calls

As already described above, every call of DLL
functions is wrapped into a special wrapper in order to ensure safety. This binding masks the basic
code, replaces the stack, supports stdcall / cdecl agreements and
monitors exceptions within the functions called.

This volume of works doesn't lead to a significant delay of function calling.

### 6\. The final build

Let's collect all the above DLL functions examples in the **'MQL5DLLSamples.cpp'** file, and MQL5 examples in the script **'MQL5DLL Test.mq5'**. The final project for Visual Studio 2008 and the script in MQL5 are attached to the article.

```
//+------------------------------------------------------------------+
//|                                                 MQL5 DLL Samples |
//|                   Copyright 2001-2010, MetaQuotes Software Corp. |
//|                                        https://www.metaquotes.net |
//+------------------------------------------------------------------+
#include "stdafx.h"

//+------------------------------------------------------------------+
//| Passing and receving of simple variables                         |
//+------------------------------------------------------------------+
_DLLAPI int __stdcall fnCalculateSpeed(int &res1,double &res2)
  {
   int    res_int=0;
   double res_double=0.0;
   int    start=GetTickCount();
//--- simple math calculations
   for(int i=0;i<=10000000;i++)
     {
      res_int+=i*i;
      res_int++;
      res_double+=i*i;
      res_double++;
     }
//--- set calculation results
   res1=res_int;
   res2=res_double;
//--- return calculation time
   return(GetTickCount()-start);
  }
//+------------------------------------------------------------------+
//| Filling the array with values                                    |
//+------------------------------------------------------------------+
_DLLAPI void __stdcall fnFillArray(int *arr,const int arr_size)
  {
//--- check input variables
   if(arr==NULL || arr_size<1) return;
//--- fill array with values
   for(int i=0;i<arr_size;i++) arr[i]=i;
  }
//+------------------------------------------------------------------+
//| The substring replacement of the text string                     |
//| the string is passed as direct reference to the string content   |
//+------------------------------------------------------------------+
_DLLAPI void fnReplaceString(wchar_t *text,wchar_t *from,wchar_t *to)
  {
   wchar_t *cp;
//--- parameters checking
   if(text==NULL || from==NULL || to==NULL) return;
   if(wcslen(from)!=wcslen(to))             return;
//--- search for substring
   if((cp=wcsstr(text,from))==NULL)         return;
//--- replace it
   memcpy(cp,to,wcslen(to)*sizeof(wchar_t));
  }
//+------------------------------------------------------------------+
//| Call for the crush                                               |
//+------------------------------------------------------------------+
_DLLAPI void __stdcall fnCrashTest(int *arr)
  {
//--- wait for receipt of a zero reference to call the exception
   *arr=0;
  }
//+------------------------------------------------------------------+
```

```
//+------------------------------------------------------------------+
//|                                                 MQL5DLL Test.mq5 |
//|                        Copyright 2010, MetaQuotes Software Corp. |
//|                                              https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "2010, MetaQuotes Software Corp."
#property link      "https://www.mql5.com"
#property version   "1.00"
//---
#import "MQL5DLLSamples.dll"
int  fnCalculateSpeed(int &res1,double &res2);
void fnFillArray(int &arr[],int arr_size);
void fnReplaceString(string text,string from,string to);
void fnCrashTest(int arr);
#import

//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
//--- calling the function for calculations
   int    speed=0;
   int    res_int=0;
   double res_double=0.0;

   speed=fnCalculateSpeed(res_int,res_double);
   Print("Time ",speed," msec, int: ",res_int," double: ",res_double);
//--- call for the array filling
   int    arr[];
   string result="Array: ";
   ArrayResize(arr,10);

   fnFillArray(arr,ArraySize(arr));
   for(int i=0;i<ArraySize(arr);i++) result=result+IntegerToString(arr[i])+" ";
   Print(result);
//--- modifying the string
   string text="A quick brown fox jumps over the lazy dog";

   fnReplaceString(text,"fox","cat");
   Print("Replace: ",text);
//--- and finally call a crash
//--- (the execution environment will catch the exception and prevent the client terminal crush)
   fnCrashTest(NULL);
   Print("You won't see this text!");
//---
  }
//+------------------------------------------------------------------+
```

Thank you for your interest! I am ready to answer any questions.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/18](https://www.mql5.com/ru/articles/18)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/18.zip "Download all attachments in the single ZIP archive")

[mql5dll\_test.mq5](https://www.mql5.com/en/articles/download/18/mql5dll_test.mq5 "Download mql5dll_test.mq5")(1.83 KB)

[mql5dllsamples.zip](https://www.mql5.com/en/articles/download/18/mql5dllsamples.zip "Download mql5dllsamples.zip")(4.62 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/370)**
(31)


![Hu Zhao](https://c.mql5.com/avatar/2017/8/5983FF9A-5180.jpg)

**[Hu Zhao](https://www.mql5.com/en/users/reichtiger)**
\|
11 Dec 2017 at 04:24

Nice article.

But one question : How can I call a MQ5 function from DLL ?

How can I call a MQ5 function from DLL ? Besides, I created a thread via CreateThread and run a window in DLL , it runs ok but when unload MQ5, dll window can be closed but MT4 crush...

What should I do next ?

Thanks in advance.

![Jorge Fernando De Los Rios De Los Rios](https://c.mql5.com/avatar/2016/5/5730ECE0-95FF.jpg)

**[Jorge Fernando De Los Rios De Los Rios](https://www.mql5.com/en/users/jfdelosrios)**
\|
12 Jul 2018 at 08:05

Hi

I had this bug

[![](https://c.mql5.com/3/207/image__14.png)](https://c.mql5.com/3/207/image__13.png "https://c.mql5.com/3/207/image__13.png")

I solved it changing this to 64 bits

[![](https://c.mql5.com/3/207/image__16.png)](https://c.mql5.com/3/207/image__15.png "https://c.mql5.com/3/207/image__15.png")

![rezaeee](https://c.mql5.com/avatar/avatar_na2.png)

**[rezaeee](https://www.mql5.com/en/users/rezaeee)**
\|
10 Dec 2018 at 16:05

Hi,

Thanks for your great job!

As I am a beginner in this field, I will be very happy if you tell me does this DLL help me or not?

What I wanna do is, [exporting](https://www.mql5.com/en/economic-calendar/united-states/exports "US Economic Calendar: Exports") data (online) from MT5 to my C++ app, do some analysis on it, then send the result as buy/sell commands to MT5 from my app. May you guide me how can I reach to this goal?

![Gonzalo Rios](https://c.mql5.com/avatar/2020/1/5E0EB5CC-FB0B.jpg)

**[Gonzalo Rios](https://www.mql5.com/en/users/griosm)**
\|
26 Mar 2019 at 07:17

Excellent, thank you very much, I work perfect with [Visual Studio](https://www.mql5.com/en/articles/5798 "Article: How to write a DLL in MQL5 in 10 minutes (Part II): Writing in Visual Studio 2017 ") 2017 Community Edition


![jimmywen](https://c.mql5.com/avatar/avatar_na2.png)

**[jimmywen](https://www.mql5.com/en/users/jimmywen)**
\|
8 Apr 2020 at 11:38

Why in the DLL can call functions in MQL5: GetTickCount ()

![The Price Histogram (Market Profile) and its implementation in MQL5](https://c.mql5.com/2/0/price__1.png)[The Price Histogram (Market Profile) and its implementation in MQL5](https://www.mql5.com/en/articles/17)

The Market Profile was developed by trully brilliant thinker Peter Steidlmayer. He suggested to use the alternative representation of information about "horizontal" and "vertical" market movements that leads to completely different set of models. He assumed that there is an underlying pulse of the market or a fundamental pattern called the cycle of equilibrium and disequilibrium. In this article I will consider Price Histogram — a simplified model of Market Profile, and will describe its implementation in MQL5.

![Data Exchange between Indicators: It's Easy](https://c.mql5.com/2/0/v5__1.png)[Data Exchange between Indicators: It's Easy](https://www.mql5.com/en/articles/19)

We want to create such an environment, which would provide access to data of indicators attached to a chart, and would have the following properties: absence of data copying; minimal modification of the code of available methods, if we need to use them; MQL code is preferable (of course, we have to use DLL, but we will use just a dozen of strings of C++ code). The article describes an easy method to develop a program environment for the MetaTrader terminal, that would provide means for accessing indicator buffers from other MQL programs.

![Applying One Indicator to Another](https://c.mql5.com/2/0/indikators_001.png)[Applying One Indicator to Another](https://www.mql5.com/en/articles/15)

When writing an indicator that uses the short form of the OnCalculate() function call, you might miss the fact that an indicator can be calculated not only by price data, but also by data of some other indicator (no matter whether it is a built-in or custom one). Do you want to improve an indicator for its correct application to the other indicator's data? In this article we'll review all the steps required for such modification.

![MQL5: Create Your Own Indicator](https://c.mql5.com/2/0/indikator__1.png)[MQL5: Create Your Own Indicator](https://www.mql5.com/en/articles/10)

What is an indicator? It is a set of calculated values that we want to be displayed on the screen in a convenient way. Sets of values are represented in programs as arrays. Thus, creation of an indicator means writing an algorithm that handles some arrays (price arrays) and records results of handling to other arrays (indicator values). By describing creation of True Strength Index, the author shows how to write indicators in MQL5.

[![](https://www.mql5.com/ff/sh/bhdtjfb1zry09943z2/267b575d2182c180804d340af38ce02c.jpg)\\
Trade from your iPhone or Android device\\
\\
You only need an internet connection to use the new powerful MetaTrader 5 Web terminal\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=wtigumvtenarnsocpyfoqnanxrilnbxx&s=ec8c539e52b83881ff2d16eaff6913b25803952eb277cac55f670a102b2edc1f&uid=&ref=https://www.mql5.com/en/articles/18&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5071786683873046122)

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