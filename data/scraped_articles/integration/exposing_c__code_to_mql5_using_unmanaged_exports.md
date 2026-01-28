---
title: Exposing C# code to MQL5 using unmanaged exports
url: https://www.mql5.com/en/articles/249
categories: Integration
relevance_score: 3
scraped_at: 2026-01-23T21:21:28.376883
---

[![](https://www.mql5.com/ff/sh/6zw0dkux8bqt7m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
MQL5 Channels - Messenger for traders\\
\\
Install the app and receive market analytics and trading tips.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=iuciwacmrxvmiibwyujliagqikizpsoo&s=268cbb13914c54b6c5c875db99b154944f6e0122b3400b54c9ac0d4f69f0f0d6&uid=&ref=https://www.mql5.com/en/articles/249&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=6465837780010216998)

MetaTrader 5 / Integration


### Introduction

I was looking for a long time to find a simple solution that would enable me to use managed mode C# DLLs in MQL5. After reading many articles I was ready to implement C++ wrapper for managed DLL when I came across a brilliant solution that saved me many hours of work.

The solution provided a simple example of exporting managed C# code to be consumed by unmanaged application. In this article I will provide a background on managed mode DLLs, describe why they cannot be accessed directly from MetaTrader and introduce the solutions I found that enable to use managed code from MetaTrader.

I will provide an example of simple usage of unmanaged exports template and will continue with all I discovered. This should provide sound background for anyone trying to use C# DLL code in MetaTrader 5.

### 1\. Managed vs Unmanaged code

Since most of the readers may be not aware of the difference between managed and unmanaged code, I will describe it in a few sentences. Basically, MetaTrader uses MQL language to implement trading rules, indicators, expert advisors and scripts. It can however make use of already implemented libraries in other languages and link them dynamically during runtime. Those libraries are also called DLLs, or [Dynamic Link Libraries](https://en.wikipedia.org/wiki/Dynamic-link_library "http://en.wikipedia.org/wiki/Dynamic-link_library").

The libraries are in fact binary files that contain compiled source code that can be invoked by a number of external programs to perform specific operations. For example neural network library can export functions for neural network training and testing, derivative library can export calculations of different derivatives, matrix library can export operations on matrices. DLLs for MetaTrader became increasingly popular as they made possible to hide parts of implementation of indicators or expert advisors. A main reason though to use libraries is to reuse existing code without need to implement it over and over again.

Before .NET existed all DLLs that were compiled by Visual Basic, Delphi, VC++, be it [COM](https://en.wikipedia.org/wiki/Component_Object_Model "http://en.wikipedia.org/wiki/Component_Object_Model"), Win32, or plain C++, could be directly executed by the operating system. We refer to this code as unmanaged or native code. Then .NET came into existence and provided very different environment.

The code is controlled (or managed) by .NET Common Language Runtime - [CLR](https://en.wikipedia.org/wiki/Common_Language_Runtime "http://en.wikipedia.org/wiki/Common_Language_Runtime"). CLR compilers are required to produce from the source code, that may be written in several different languages, Metadata and Common Intermediate Language - [CIL](https://en.wikipedia.org/wiki/Common_Intermediate_Language "http://en.wikipedia.org/wiki/Common_Intermediate_Language").

CIL is machine - independent higher level language and Metadata fully describe types of the objects described by CIL according to Common Type Specification - [CTS](https://www.mql5.com/go?link=http://www.dotnetspider.com/resources/34535-CTS-Common-Type-Specification.aspx "http://www.dotnetspider.com/resources/34535-CTS-Common-Type-Specification.aspx"). Since CLR knows everything about the types it can provide us with managed execution environment. Managing can be thought of as garbage collection - automatic memory management and objects deletion and providing security - protection against common mistakes in native languages that could cause alien code execution with administrator privilges or simply memory overriding.

It has to be mentioned that CIL code is never directly executed - it is always translated into native machine code by [JIT](https://en.wikipedia.org/wiki/Just-in-time_compilation "http://en.wikipedia.org/wiki/Just-in-time_compilation") (Just-In-Time) compilation or by pre-compiling CIL into assembly. For a person that reads this for a first time the notion of managed mode code can be confusing, therefore I am pasting the general flow within CLR below:

![](https://c.mql5.com/2/2/400px-CLR_diag0svg.png)

Figure 1. Common Language Runtime

### 2\. Possible implementations of accessing managed code from MQL5

In the following paragraph I will describe methods that enable to access managed code from unmanaged code.

I think it is worth to mention them all as there may be someone who would prefer to use other method over the one that I am using. The methods that are in use are COM Interop, Reverse P/Invoke, C++ IJW, C++/Cli wrapper class and Unmanaged Exports.

**2.1. COM Interop**

Component Object Model ( [COM](https://en.wikipedia.org/wiki/Component_Object_Model "http://en.wikipedia.org/wiki/Component_Object_Model")) is a binary interface standard introduced by Microsoft in early nineties. The core idea of this technology is to enable object created in different programming languages to be used by any other COM object without knowing its internal implementation. Such requirement enforces implementing strict well-defined interface of the COM that is fully separate from the implementation.

In fact COM was superseded by .NET technology and Microsoft pushes to use .NET instead of COM. In order to provide backward compatibility with older code, .NET can cooperate with COM in both directions, that is .NET can call COM methods and COM object can make use of .NET managed code.

This functionality is called COM Interopability or COM Interop. COM interop API is in the managed System.Runtime.InteropServices namespace.

![Figure 2. COM Interoperability model ](https://c.mql5.com/2/2/CCWInteraction4.png)

Figure 2. COM Interoperability model

The following COM interop code calls a single function raw\_factorial.

Please notice CoInitialize() CoCreateInstance() and CoUninitialize() functions and interface calling function:

```
#include "windows.h"
#include <stdio.h>
#import "CSDll.tlb" named_guids

int main(int argc, char* argv[])
{
    HRESULT hRes = S_OK;
    CoInitialize(NULL);
    CSDll::IMyManagedInterface *pManagedInterface = NULL;

    hRes = CoCreateInstance(CSDll::CLSID_Class1, NULL, CLSCTX_INPROC_SERVER,
     CSDll::IID_IMyManagedInterface, reinterpret_cast<void**> (&pManagedInterface));

    if (S_OK == hRes)
    {
        long retVal =0;
        hRes = pManagedInterface->raw_factorial(4, &retVal);
        printf("The value returned by the dll is %ld\n",retVal);
        pManagedInterface->Release();
    }

    CoUninitialize();
    return 0;
}
```

For further reading on COM Interop please read detailed documentation at [Introduction to COM Interop](https://www.mql5.com/go?link=https://www.microsoft.com/en-us/download/details.aspx?id=55979 "http://msdn.microsoft.com/en-us/library/kew41ycz(v=vs.71).aspx") and the usage example I found on msdn blog: [How to call C++ code from Managed, and vice versa (Interop)](https://www.mql5.com/go?link=https://blogs.msdn.microsoft.com/deeptanshuv/2005/06/26/how-to-call-c-code-from-managed-and-vice-versa-interop/ "http://blogs.msdn.com/b/deeptanshuv/archive/2005/06/26/432870.aspx").

**2.2. Reverse P/Invoke**

Platform Invoke, referred to as P/Invoke enables .NET to call any function in any unmanaged language as long as its signature is redeclared. This is achieved by executing a native function pointer from .NET. The usage is well described in [Platform Invoke Tutorial](https://www.mql5.com/go?link=https://www.microsoft.com/en-us/download/details.aspx?id=55979 "http://msdn.microsoft.com/en-us/library/aa288468(v=vs.71).aspx").

The basic usage is to use DllImport attribute to mark the imported function:

```
// PInvokeTest.cs
using System;
using System.Runtime.InteropServices;

class PlatformInvokeTest
{
    [DllImport("msvcrt.dll")]
    public static extern int puts(string c);
    [DllImport("msvcrt.dll")]
    internal static extern int _flushall();

    public static void Main()
    {
        puts("Test");
        _flushall();
    }
}
```

The reverse operation can be described as providing a managed delegate callback to unmanaged code.

This is called Reverse P/Invoke and is achieved by implementing a public delegate function in managed environment and importing caller function implemented in native DLL:

```
#include <stdio.h>
#include <string.h>
typedef void (__stdcall *callback)(wchar_t * str);
extern "C" __declspec(dllexport) void __stdcall caller(wchar_t * input, int count, callback call)
{
      for(int i = 0; i < count; i++)
      {
            call(input);
      }
}
```

The managed code example is as follows:

```
using System.Runtime.InteropServices;
public class foo
{
    public delegate void callback(string str);
    public static void callee(string str)
    {
        System.Console.WriteLine("Managed: " +str);
    }
    public static int Main()
    {
        caller("Hello World!", 10, new callback(foo.callee));
        return 0;
    }
    [DllImport("nat.dll",CallingConvention=CallingConvention.StdCall)]
    public static extern void caller(string str, int count, callback call);
}
```

The main point of this solution is that this requires the managed side to begin the interaction.

For further reference please read [Gotchas with Reverse Pinvoke (unmanaged to managed code callbacks)](https://www.mql5.com/go?link=https://blogs.msdn.microsoft.com/davidnotario/2006/01/13/gotchas-with-reverse-pinvoke-unmanaged-to-managed-code-callbacks/ "http://blogs.msdn.com/b/davidnotario/archive/2006/01/13/512436.aspx") and [PInvoke-Reverse PInvoke and  stdcall - cdecl](https://www.mql5.com/go?link=https://blogs.msdn.microsoft.com/thottams/2007/06/02/pinvoke-reverse-pinvoke-and-__stdcall-__cdecl/ "http://blogs.msdn.com/b/thottams/archive/2007/06/02/pinvoke-reverse-pinvoke-and-stdcall-cdecl.aspx").

[http://blogs.msdn.com/b/thottams/archive/2007/06/02/pinvoke-reverse-pinvoke-and-stdcall-cdecl.aspx](https://www.mql5.com/go?link=https://blogs.msdn.microsoft.com/thottams/2007/06/02/pinvoke-reverse-pinvoke-and-__stdcall-__cdecl/ "http://blogs.msdn.com/b/thottams/archive/2007/06/02/pinvoke-reverse-pinvoke-and-stdcall-cdecl.aspx")

**2.3. C++ IJW**

C++ interop, referred to as It Just Works (IJW)) is a C++ specific feature, provided by [Managed Extensions for C++](https://www.mql5.com/go?link=https://www.microsoft.com/en-us/download/details.aspx?id=55979 "http://msdn.microsoft.com/en-us/library/aa712574(v=vs.71).aspx"):

```
#using <mscorlib.dll>
using namespace System;
using namespace System::Runtime::InteropServices;

#include <stdio.h>

int main()
{
   String * pStr = S"Hello World!";
   char* pChars = (char*)Marshal::StringToHGlobalAnsi(pStr).ToPointer();

   puts(pChars);

   Marshal::FreeHGlobal(pChars);
}
```

This solution might be useful for people wanting to use their managed C++ in unmanaged application. For full reference please read [Interoperability in Managed Extensions for C++](https://www.mql5.com/go?link=https://www.microsoft.com/en-us/download/details.aspx?id=55979 "http://msdn.microsoft.com/en-us/library/zbz07712(v=vs.71).aspx") and [Using IJW in Managed C++](https://www.mql5.com/go?link=https://www.codeproject.com/Articles/2234/Using-IJW-in-Managed-C "http://www.codeproject.com/KB/mcpp/nishijw01.aspx").

**2.4. C++/Cli wrapper class**

C++/Cli wrapper class implementation takes its name from embedding or wrapping managed class by another class written in C++/Cli mode. The first step to write the wrapper DLL is to write the C++ class that wraps the methods of original managed class.

The wrapper class must contain a handle to .NET object using gcroot<> template and must delegate all calls from original class. The wrapper class is compiled to IL (intermediate language) format, and therefore is a managed one.

The next step is to write native C++ class with #pragma unmanaged directive that wraps IL class and delegates all calls with [\_\_declspec(dllexport)](https://www.mql5.com/go?link=https://www.microsoft.com/en-us/download/details.aspx?id=55984 "http://msdn.microsoft.com/en-us/library/a90k134d(VS.80).aspx") directive. Those steps will make native C++ DLL that can be used by any unmanaged application.

Please see the example implementation. The first step is to implement C# code.

The example calculator class contains two public methods:

```
public class Calculator
{
    public int Add(int first, int second)
    {
        return first + second;
    }
    public string FormatAsString(float i)
    {
        return i.ToString();
    }
}
```

The next step is to write a managed wrapper that will delegate all methods from calculator class:

```
#pragma once
#pragma managed

#include <vcclr.h>

class ILBridge_CppCliWrapper_Calculator {
private:
    //Aggregating the managed class
    gcroot<CppCliWrapper::Calculator^> __Impl;
public:
    ILBridge_CppCliWrapper_Calculator() {
        __Impl = gcnew CppCliWrapper::Calculator;
    }
    int Add(int first, int second) {
        System::Int32 __Param_first = first;
        System::Int32 __Param_second = second;
        System::Int32 __ReturnVal = __Impl->Add(__Param_first, __Param_second);
        return __ReturnVal;
    }
    wchar_t* FormatAsString(float i) {
        System::Single __Param_i = i;
        System::String __ReturnVal = __Impl->FormatAsString(__Param_i);
        wchar_t* __MarshaledReturnVal = marshal_to<wchar_t*>(__ReturnVal);
        return __MarshaledReturnVal;
    }
};
```

Please note the reference to the original Calculator class is stored using gcnew instruction and stored as gcroot<> template. All wrapped methods can have the same name as original ones and the parameters and return values are preceded by \_\_Param and \_\_ReturnVal respectively.

Now the unmanaged C++ class that wraps the C++/Cli and exports native C++ DLL methods must be implemented.

The header file should contain class definition with \_\_declspec(dllexport) directive and store the pointer to the wrapper class.

```
#pragma once
#pragma unmanaged

#ifdef THISDLL_EXPORTS
#define THISDLL_API __declspec(dllexport)
#else
#define THISDLL_API __declspec(dllimport)
#endif

//Forward declaration for the bridge
class ILBridge_CppCliWrapper_Calculator;

class THISDLL_API NativeExport_CppCliWrapper_Calculator {
private:
    //Aggregating the bridge
    ILBridge_CppCliWrapper_Calculator* __Impl;
public:
    NativeExport_CppCliWrapper_Calculator();
    ~NativeExport_CppCliWrapper_Calculator();
    int Add(int first, int second);
    wchar_t* FormatAsString(float i);
};
```

And its implementation:

```
#pragma managed
#include "ILBridge_CppCliWrapper_Calculator.h"
#pragma unmanaged
#include "NativeExport_CppCliWrapper_Calculator.h"

NativeExport_CppCliWrapper_Calculator::NativeExport_CppCliWrapper_Calculator() {
    __Impl = new ILBridge_CppCliWrapper_Calculator;
}
NativeExport_CppCliWrapper_Calculator::~NativeExport_CppCliWrapper_Calculator()
{
    delete __Impl;
}
int NativeExport_CppCliWrapper_Calculator::Add(int first, int second) {
    int __ReturnVal = __Impl->Add(first, second);
    return __ReturnVal;
}
wchar_t* NativeExport_CppCliWrapper_Calculator::FormatAsString(float i) {
    wchar_t* __ReturnVal = __Impl->FormatAsString(i);
    return __ReturnVal;
}
```

A step by step guide to make this wrapper class is described at [.NET to C++ Bridge](https://www.mql5.com/go?link=http://blogs.microsoft.co.il/sasha/2008/02/16/net-to-c-bridge/ "http://blogs.microsoft.co.il/blogs/sasha/archive/2008/02/16/net-to-c-bridge.aspx").

A full reference for creating wrappers is available at [Mixing .NET and native code](https://www.mql5.com/go?link=https://www.codeproject.com/Articles/35041/Mixing-NET-and-native-code "http://www.codeproject.com/KB/mcpp/mixnetnative.aspx") and for general information about declaring handles in Native Types please read [How to: Declare Handles in Native Types](https://www.mql5.com/go?link=https://www.microsoft.com/en-us/download/details.aspx?id=55984 "http://msdn.microsoft.com/en-us/library/481fa11f(VS.80).aspx").

**2.5. Unmanaged exports**

This technique is fully described by [Expert .NET 2.0 IL Assembler](https://www.mql5.com/go?link=http://www.apress.com/book/view/9781590596463 "http://www.apress.com/book/view/9781590596463") book which I recommend for anyone that would like to read about details of the .NET compiler. The main idea is to expose managed methods as unmanaged exports of a managed DLL by decompiling already compiled module into IL code using ILDasm, changing module's VTable and VTableFixup tables and recompiling the DLL using ILAsm.

This task may look like a daunting one but the result from this operation will be to produce a DLL that can be used from within any unmanaged application. One has to remember that it is still a managed assembly, so .NET framework has to be installed.  A step-by-step tutorial to do this is available at [Export Managed Code as Unmanaged](https://www.mql5.com/go?link=http://www.c-sharpcorner.com/article/export-managed-code-as-unmanaged/ "http://www.c-sharpcorner.com/UploadFile/JTeeuwen/ExportManagedCodeasUnmanaged11262005051206AM/ExportManagedCodeasUnmanaged.aspx").

After decompiling DLL using ILDasm we get source code in IL language. Please observe a simple example of IL code with unmanaged export pasted below:

```
assembly extern mscorlib {}
..assembly UnmExports {}
..module UnmExports.dll
..corflags 0x00000002
..vtfixup [1] int32 fromunmanaged at VT_01
..data VT_01 = int32(0)
..method public static void foo()
{
..vtentry 1:1
..export [1] as foo
ldstr "Hello from managed world"
call void [mscorlib]System.Console::WriteLine(string)
ret
}
```

The IL source code lines responsible for implementing unmanaged exports are:

```
..vtfixup [1] int32 fromunmanaged at VT_01
..data VT_01 = int32(0)
```

and

```
..vtentry 1:1
..export [1] as foo
```

First part is responsible for adding function entry in VTableFixup table and setting VT\_01 virtual address to the function. Second part specifies which VTEntry is to be used for this function and export alias for the function to be exported.

Pros of this solution are that during DLL implementation phase we do not need to implement any additional code apart from the usual managed C# DLL and as stated by the book, that this method fully opens the managed world with all its security and class libraries to unmanaged clients.

The drawback is that getting into .NET assembly language is not suitable for all people. I was conviced that I would write c++ wrapper class instead until I found unmanaged exports template by Robert Giesecke: [http://sites.google.com/site/robertgiesecke/](https://www.mql5.com/go?link=https://sites.google.com/site/robertgiesecke/ "http://sites.google.com/site/robertgiesecke/") that enables to use unmanaged exports without any need to get inside IL code.

### 3\. Unmanaged exports C\# template

Template for unmanaged exports C# projects by R.Giesecke uses [MSBuild task](https://www.mql5.com/go?link=https://msdn.microsoft.com/en-us/library/ms171466(v=vs.90).aspx "http://msdn.microsoft.com/en-us/library/ms171466(v=vs.90).aspx") that automatically adds the appropriate VT-fixups after the build, therefore there is no need to change IL code at all. The template package only needs to be downloaded as a zip file and copied into ProjectTemplates folder of the Visual Studio.

After compiling the project the resulting DLL file can be flawlessly imported by MetaTrader, I will provide the examples in the next sections.

### 4\. Examples

It was quite a challenging task to figure out how to pass variables, arrays and structs between MetaTrader and C# using correct Marshalling method and I think that information provided here will save you a lot of time. All examples were compiled on Windows Vista with .NET 4.0 and Visual C# Express 2010. I am also attaching sample DLL with MQL5 code that invokes functions from C# DLL to the article.

**4.1. Example 1. Adding two integer, double or float variables in DLL function and returning the result to MetaTrader**

```
using System;
using System.Text;
using RGiesecke.DllExport;
using System.Runtime.InteropServices;

namespace Testme
{
    class Test
    {

        [DllExport("Add", CallingConvention = CallingConvention.StdCall)]
        public static int Add(int left, int right)
        {
            return left + right;
        }

        [DllExport("Sub", CallingConvention = CallingConvention.StdCall)]
        public static int Sub(int left, int right)
        {
            return left - right;
        }

        [DllExport("AddDouble", CallingConvention = CallingConvention.StdCall)]
        public static double AddDouble(double left, double right)
        {
            return left + right;
        }

        [DllExport("AddFloat", CallingConvention = CallingConvention.StdCall)]
        public static float AddFloat(float left, float right)
        {
            return left + right;
        }

    }
}
```

As you may have noticed, every exported function is preceded by DllExport directive. The first parameter describes alias of the exported function and the second parameter calling convention, for MetaTrader we must use CallingConvention.StdCall.

MQL5 code that imports and uses the functions exported from the DLL is straightforward and does not differ from any other DLL written in native C++. At first one must to declare imported functions inside #import block, and indicate which functions from the DLL can be later used from the MQL5 code:

```
//+------------------------------------------------------------------+
//|                                  UnmanagedExportsDLLExample1.mq5 |
//|                                      Copyright 2010, Investeo.pl |
//|                                                http:/Investeo.pl |
//+------------------------------------------------------------------+
#property copyright "Copyright 2010, Investeo.pl"
#property link      "http:/Investeo.pl"
#property version   "1.00"

#import "Testme.dll"
   int Add(int left,int right);
   int Sub(int left,int right);
   float AddFloat(float left,float right);
   double AddDouble(double left,double right);
#import

//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
//---
   for(int i=0; i<3; i++)
     {
      Print(Add(i,666));
      Print(Sub(666,i));
      Print(AddDouble(666.5,i));
      Print(AddFloat(666.5,-i));
     }
  }
//+------------------------------------------------------------------+
```

Result

```
2011.01.30 21:28:18     UnmanagedExportsDLLExample1 (EURUSD,M1) 664.50000
2011.01.30 21:28:18     UnmanagedExportsDLLExample1 (EURUSD,M1) 668.5
2011.01.30 21:28:18     UnmanagedExportsDLLExample1 (EURUSD,M1) 664
2011.01.30 21:28:18     UnmanagedExportsDLLExample1 (EURUSD,M1) 668
2011.01.30 21:28:18     UnmanagedExportsDLLExample1 (EURUSD,M1) 665.50000
2011.01.30 21:28:18     UnmanagedExportsDLLExample1 (EURUSD,M1) 667.5
2011.01.30 21:28:18     UnmanagedExportsDLLExample1 (EURUSD,M1) 665
2011.01.30 21:28:18     UnmanagedExportsDLLExample1 (EURUSD,M1) 667
2011.01.30 21:28:18     UnmanagedExportsDLLExample1 (EURUSD,M1) 666.50000
2011.01.30 21:28:18     UnmanagedExportsDLLExample1 (EURUSD,M1) 666.5
2011.01.30 21:28:18     UnmanagedExportsDLLExample1 (EURUSD,M1) 666
2011.01.30 21:28:18     UnmanagedExportsDLLExample1 (EURUSD,M1) 666
```

**4.2. Example 2. One dimensional array access**

```
        [DllExport("Get1DInt", CallingConvention = CallingConvention.StdCall)]
        public static int Get1DInt([MarshalAs(UnmanagedType.LPArray, SizeParamIndex = 1)]  int[] tab, int i, int idx)
        {
            return tab[idx];
        }

        [DllExport("Get1DFloat", CallingConvention = CallingConvention.StdCall)]
        public static float Get1DFloat([MarshalAs(UnmanagedType.LPArray, SizeParamIndex = 1)]  float[] tab, int i, int idx)
        {
            return tab[idx];
        }

        [DllExport("Get1DDouble", CallingConvention = CallingConvention.StdCall)]
        public static double Get1DDouble([MarshalAs(UnmanagedType.LPArray, SizeParamIndex = 1)]  double[] tab, int i, int idx)
        {
            return tab[idx];
        }
```

In order to marshal a one-dimensional array, MarshalAs directive must pass UnmanagedType.LPArray as the first parameter and SizeParamIndex as the second parameter. SizeParamIndex indicates which parameter (counting from 0) is the parameter containing array size.

In the examples above i is the array size and idx is the index of the element to return.

MQL5 example code using array access is below:

```
//+------------------------------------------------------------------+
//|                                  UnmanagedExportsDLLExample2.mq5 |
//|                                      Copyright 2010, Investeo.pl |
//|                                                http:/Investeo.pl |
//+------------------------------------------------------------------+
#property copyright "Copyright 2010, Investeo.pl"
#property link      "http:/Investeo.pl"
#property version   "1.00"

#import "Testme.dll"
   int Get1DInt(int &t[],int i,int idx);
   float Get1DFloat(float &t[],int i,int idx);
   double Get1DDouble(double &t[],int i,int idx);
#import
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
//---
   int tab[3];
   tab[0] = 11;
   tab[1] = 22;
   tab[2] = 33;

   float tfloat[3]={0.5,1.0,1.5};
   double tdouble[3]={0.5,1.0,1.5};

   for(int i=0; i<3; i++)
     {
      Print(tab[i]);
      Print(Get1DInt(tab,3,i));
      Print(Get1DFloat(tfloat,3,i));
      Print(Get1DDouble(tdouble,3,i));

     }
  }
//+------------------------------------------------------------------+
```

Result

```
2011.01.30 21:46:25     UnmanagedExportsDLLExample2 (EURUSD,M1) 1.5
2011.01.30 21:46:25     UnmanagedExportsDLLExample2 (EURUSD,M1) 1.50000
2011.01.30 21:46:25     UnmanagedExportsDLLExample2 (EURUSD,M1) 33
2011.01.30 21:46:25     UnmanagedExportsDLLExample2 (EURUSD,M1) 33
2011.01.30 21:46:25     UnmanagedExportsDLLExample2 (EURUSD,M1) 1
2011.01.30 21:46:25     UnmanagedExportsDLLExample2 (EURUSD,M1) 1.00000
2011.01.30 21:46:25     UnmanagedExportsDLLExample2 (EURUSD,M1) 22
2011.01.30 21:46:25     UnmanagedExportsDLLExample2 (EURUSD,M1) 22
2011.01.30 21:46:25     UnmanagedExportsDLLExample2 (EURUSD,M1) 0.5
2011.01.30 21:46:25     UnmanagedExportsDLLExample2 (EURUSD,M1) 0.50000
2011.01.30 21:46:25     UnmanagedExportsDLLExample2 (EURUSD,M1) 11
2011.01.30 21:46:25     UnmanagedExportsDLLExample2 (EURUSD,M1) 11
```

**4.3. Example 3. Populating one dimensional array and returning it to MetaTrader**

```
        [DllExport("SetFiboArray", CallingConvention = CallingConvention.StdCall)]
        public static int SetFiboArray([MarshalAs(UnmanagedType.LPArray, SizeParamIndex = 1)]
        int[] tab, int len, [In, Out, MarshalAs(UnmanagedType.LPArray, SizeParamIndex = 1)] int[] res)
        {
            res[0] = 0;
            res[1] = 1;

            if (len < 3) return -1;
            for (int i=2; i<len; i++)
                res[i] = res[i-1] + res[i-2];
            return 0;
        }
```

This example uses two input arrays to compare input parameter convention. If changed elements are to be returned back to Metatrader (passing by reference) it is enough to put \[In, Out,\] attributes before MarshalAs attribute.

```
//+------------------------------------------------------------------+
//|                                  UnmanagedExportsDLLExample3.mq5 |
//|                                      Copyright 2011, Investeo.pl |
//|                                                http:/Investeo.pl |
//+------------------------------------------------------------------+
#property copyright "Copyright 2011, Investeo.pl"
#property link      "http:/Investeo.pl"
#property version   "1.00"

#import "Testme.dll"
    int SetFiboArray(int& t[], int i, int& o[]);
#import

//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
//---
int fibo[10];
static int o[10];

   for (int i=0; i<4; i++)
   { fibo[i]=i; o[i] = i; }

   SetFiboArray(fibo, 6, o);

   for (int i=0; i<6; i++)
      Print(IntegerToString(fibo[i])+":"+IntegerToString(o[i]));

  }
//+------------------------------------------------------------------+
```

Result

```
2011.01.30 22:01:39     UnmanagedExportsDLLExample3 (EURUSD,M1) 0:5
2011.01.30 22:01:39     UnmanagedExportsDLLExample3 (EURUSD,M1) 0:3
2011.01.30 22:01:39     UnmanagedExportsDLLExample3 (EURUSD,M1) 3:2
2011.01.30 22:01:39     UnmanagedExportsDLLExample3 (EURUSD,M1) 2:1
2011.01.30 22:01:39     UnmanagedExportsDLLExample3 (EURUSD,M1) 1:1
2011.01.30 22:01:39     UnmanagedExportsDLLExample3 (EURUSD,M1) 0:0
```

**4.4. Example 4. Access to two dimensional array**

```
        public static int idx(int a, int b) {int cols = 2; return a * cols + b; }

        [DllExport("Set2DArray", CallingConvention = CallingConvention.StdCall)]
        public static int Set2DArray([In, Out, MarshalAs(UnmanagedType.LPArray, SizeParamIndex = 1)] int[] tab, int len)
        {
            tab[idx(0, 0)] = 0;
            tab[idx(0, 1)] = 1;
            tab[idx(1, 0)] = 2;
            tab[idx(1, 1)] = 3;
            tab[idx(2, 0)] = 4;
            tab[idx(2, 1)] = 5;

            return 0;
        }
```

Two dimensional array is not so simple to marshal, but I used a trick - namely passing 2D array as one dimensional and accessing array elements by auxiliary idx function.

```
//+------------------------------------------------------------------+
//|                                  UnmanagedExportsDLLExample4.mq5 |
//|                                      Copyright 2011, Investeo.pl |
//|                                                http:/Investeo.pl |
//+------------------------------------------------------------------+
#property copyright "Copyright 2011, Investeo.pl"
#property link      "http:/Investeo.pl"
#property version   "1.00"

#import "Testme.dll"
   int Set2DArray(int &t[][2],int i);
#import
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
//---
   int t2[3][2];

   Set2DArray(t2,6);

   for(int row=0; row<3; row++)
      for(int col=0; col<2; col++)
         Print("t2["+IntegerToString(row)+"]["+IntegerToString(col)+"]="+IntegerToString(t2[row][col]));

  }
//+------------------------------------------------------------------+
```

Result

```
2011.01.30 22:13:01     UnmanagedExportsDLLExample4 (EURUSD,M1) t2[2][1]=5
2011.01.30 22:13:01     UnmanagedExportsDLLExample4 (EURUSD,M1) t2[2][0]=4
2011.01.30 22:13:01     UnmanagedExportsDLLExample4 (EURUSD,M1) t2[1][1]=3
2011.01.30 22:13:01     UnmanagedExportsDLLExample4 (EURUSD,M1) t2[1][0]=2
2011.01.30 22:13:01     UnmanagedExportsDLLExample4 (EURUSD,M1) t2[0][1]=1
2011.01.30 22:13:01     UnmanagedExportsDLLExample4 (EURUSD,M1) t2[0][0]=0
```

**4.5. Example 5. Replacing string contents**

```
  	 [DllExport("ReplaceString", CallingConvention = CallingConvention.StdCall)]
        public static int ReplaceString([In, Out, MarshalAs(UnmanagedType.LPWStr)] StringBuilder str,
        [MarshalAs(UnmanagedType.LPWStr)]string a, [MarshalAs(UnmanagedType.LPWStr)]string b)
        {
            str.Replace(a, b);

            if (str.ToString().Contains(a)) return 1;
            else  return 0;
        }
```

This example is short but took me quite a long time to implement as I tried to use string parameter using \[In,Out\] attributes or with ref or out keywords with no success.

The solution is to use StringBuilder instead of string variable.

```
//+------------------------------------------------------------------+
//|                                  UnmanagedExportsDLLExample5.mq5 |
//|                                      Copyright 2011, Investeo.pl |
//|                                                http:/Investeo.pl |
//+------------------------------------------------------------------+
#property copyright "Copyright 2011, Investeo.pl"
#property link      "http:/Investeo.pl"
#property version   "1.00"

#import "Testme.dll"
   int ReplaceString(string &str,string a,string b);
#import
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
//---
   string str="A quick brown fox jumps over the lazy dog";
   string stra = "fox";
   string strb = "cat";

   Print(str);
   Print(ReplaceString(str,stra,strb));
   Print(str);

  }
//+------------------------------------------------------------------+
```

Result

```
2011.01.30 22:18:36     UnmanagedExportsDLLExample5 (EURUSD,M1) A quick brown cat jumps over the lazy dog
2011.01.30 22:18:36     UnmanagedExportsDLLExample5 (EURUSD,M1) 0
2011.01.30 22:18:36     UnmanagedExportsDLLExample5 (EURUSD,M1) A quick brown fox jumps over the lazy dog
```

**4.6. Example 6. Sending and changing MqlTick struct**

```
	 private static List<MqlTick> list;

	 [StructLayout(LayoutKind.Sequential, Pack = 1)]
        public struct MqlTick
        {
            public Int64 Time;
            public Double Bid;
            public Double Ask;
            public Double Last;
            public UInt64 Volume;
        }

        [DllExport("AddTick", CallingConvention = CallingConvention.StdCall)]
        public static int AddTick(ref MqlTick tick, ref double bidsum)
        {
            bidsum = 0.0;

            if (list == null) list = new List<MqlTick>();

            tick.Volume = 666;
            list.Add(tick);

            foreach (MqlTick t in list) bidsum += t.Ask;

            return list.Count;
        }
```

MqlTick struct is passed as reference, marked by ref keyword. The MqlTick struct itself has to be preceded by \[StructLayout (LayoutKind.Sequential, Pack =1)\] attribute.

Pack parameter describes data alignment in the struct, please read [StructLayoutAttribute.Pack Field](https://www.mql5.com/go?link=https://msdn.microsoft.com/en-us/library/system.runtime.interopservices.structlayoutattribute.pack.aspx "http://msdn.microsoft.com/en-us/library/system.runtime.interopservices.structlayoutattribute.pack.aspx") for details.

```
//+------------------------------------------------------------------+
//|                                  UnmanagedExportsDLLExample6.mq5 |
//|                                      Copyright 2011, Investeo.pl |
//|                                                http:/Investeo.pl |
//+------------------------------------------------------------------+
#property copyright "Copyright 2011, Investeo.pl"
#property link      "http:/Investeo.pl"
#property version   "1.00"

#import "Testme.dll"
   int AddTick(MqlTick &tick, double& bidsum);
#import
//+------------------------------------------------------------------+
//| Custom indicator initialization function                         |
//+------------------------------------------------------------------+
int OnInit()
  {
//--- indicator buffers mapping

//---
   return(0);
  }
//+------------------------------------------------------------------+
//| Custom indicator iteration function                              |
//+------------------------------------------------------------------+
int OnCalculate(const int rates_total,
                const int prev_calculated,
                const datetime& time[],
                const double& open[],
                const double& high[],
                const double& low[],
                const double& close[],
                const long& tick_volume[],
                const long& volume[],
                const int& spread[])
  {
//---
   MqlTick newTick;
   double bidsum;

   SymbolInfoTick(Symbol(), newTick);

   Print("before = " + IntegerToString(newTick.volume));

   Print(AddTick(newTick, bidsum));

   Print("after = " + IntegerToString(newTick.volume) + " : " + DoubleToString(bidsum));


//--- return value of prev_calculated for next call
   return(rates_total);
  }
//+------------------------------------------------------------------+
```

Result

```
2011.01.30 23:59:05     TickDLLSend (EURUSD,M1) after = 666 : 8.167199999999999
2011.01.30 23:59:05     TickDLLSend (EURUSD,M1) 6
2011.01.30 23:59:05     TickDLLSend (EURUSD,M1) before = 0
2011.01.30 23:59:05     TickDLLSend (EURUSD,M1) after = 666 : 6.806
2011.01.30 23:59:05     TickDLLSend (EURUSD,M1) 5
2011.01.30 23:59:05     TickDLLSend (EURUSD,M1) before = 0
2011.01.30 23:59:05     TickDLLSend (EURUSD,M1) after = 666 : 5.4448
2011.01.30 23:59:05     TickDLLSend (EURUSD,M1) 4
2011.01.30 23:59:05     TickDLLSend (EURUSD,M1) before = 0
2011.01.30 23:59:05     TickDLLSend (EURUSD,M1) after = 666 : 4.0836
2011.01.30 23:59:05     TickDLLSend (EURUSD,M1) 3
2011.01.30 23:59:05     TickDLLSend (EURUSD,M1) before = 0
2011.01.30 23:59:05     TickDLLSend (EURUSD,M1) after = 666 : 2.7224
2011.01.30 23:59:05     TickDLLSend (EURUSD,M1) 2
2011.01.30 23:59:05     TickDLLSend (EURUSD,M1) before = 0
2011.01.30 23:59:05     TickDLLSend (EURUSD,M1) after = 666 : 1.3612
2011.01.30 23:59:05     TickDLLSend (EURUSD,M1) 1
2011.01.30 23:59:04     TickDLLSend (EURUSD,M1) before = 0
```

### Conclusion

In this article I presented different methods of interaction between MQL5 code and managed C# code.

I also provided several examples on how to marshal MQL5 structures against C# and how to invoke exported DLL functions in MQL5 scripts. I believe that the provided examples may serve as a basis for future research in writing DLLs in managed code.

This article also open doors for MetaTrader to use many libraries that are already implemented in C#. For further reference please read the articles that are linked in References section.

To test it, please locate the files to the following folders:

MQL5\\Libraries\\testme.dll

MQL5\\Scripts\\unmanagedexportsdllexample1.mq5

MQL5\\Scripts\\unmanagedexportsdllexample2.mq5

MQL5\\Scripts\\unmanagedexportsdllexample3.mq5

MQL5\\Scripts\\unmanagedexportsdllexample4.mq5

MQL5\\Scripts\\unmanagedexportsdllexample5.mq5

MQL5\\Experts\\unmanagedexportsdllexample6.mq5

### References

01. [Exporting .NET DLLs with Visual Studio 2005 to be Consumed by Native Applications](https://www.mql5.com/go?link=http://www.codeguru.com/csharp/.net/cpp_managed/windowsservices/article.php/c14735/Exporting-NET-DLLs-with-Visual-Studio-2005-to-be-Consumed-by-Native-Applications.htm "http://www.codeguru.com/csharp/.net/cpp_managed/windowsservices/article.php/c14735/")

02. [Interoperating with Unmadged Coded](https://www.mql5.com/go?link=https://msdn.microsoft.com/en-us/library/sd10k43k(v=vs.71).aspx "http://msdn.microsoft.com/en-us/library/sd10k43k(v=vs.71).aspx")
03. [Introduction to COM Interop](https://www.mql5.com/go?link=https://www.microsoft.com/en-us/download/details.aspx?id=55979 "http://msdn.microsoft.com/en-us/library/kew41ycz(v=vs.71).aspx")

04. [Component Object Model (COM)](https://en.wikipedia.org/wiki/Component_Object_Model "https://en.wikipedia.org/wiki/Component_Object_Model")

05. [Exporting from a DLL Using \_\_declspec(dllexport)](https://www.mql5.com/go?link=https://www.microsoft.com/en-us/download/details.aspx?id=55984 "http://msdn.microsoft.com/en-us/library/a90k134d(VS.80).aspx")

06. [How to: Declare Handles in Native Types](https://www.mql5.com/go?link=https://www.microsoft.com/en-us/download/details.aspx?id=55984 "http://msdn.microsoft.com/en-us/library/481fa11f(VS.80).aspx")
07. [How to call C++ code from Managed, and vice versa (Interop)](https://www.mql5.com/go?link=https://blogs.msdn.microsoft.com/deeptanshuv/2005/06/26/how-to-call-c-code-from-managed-and-vice-versa-interop/ "http://blogs.msdn.com/b/deeptanshuv/archive/2005/06/26/432870.aspx")

08. [Reverse P/Invoke and exception](https://www.mql5.com/go?link=https://blogs.msdn.microsoft.com/junfeng/2008/01/28/reverse-pinvoke-and-exception/ "http://blogs.msdn.com/b/junfeng/archive/2008/01/28/reverse-p-invoke-and-exception.aspx")
09. [How to call a managed DLL from native Visual C++ code in Visual Studio.NET or in Visual Studio 2005](https://www.mql5.com/go?link=https://support.microsoft.com/en-us/help/828736/how-to-call-a-managed-dll-from-native-visual-c-code-in-visual-studio-n?fr=1 "http://support.microsoft.com/kb/828736/en-us?fr=1")

10. [Platform Invoke Tutorial](https://www.mql5.com/go?link=https://www.microsoft.com/en-us/download/details.aspx?id=55979 "http://msdn.microsoft.com/en-us/library/aa288468(v=vs.71).aspx")
11. [PInvoke-Reverse PInvoke and \_\_stdcall - \_\_cdecl](https://www.mql5.com/go?link=https://blogs.msdn.microsoft.com/thottams/2007/06/02/pinvoke-reverse-pinvoke-and-__stdcall-__cdecl/ "http://blogs.msdn.com/b/thottams/archive/2007/06/02/pinvoke-reverse-pinvoke-and-stdcall-cdecl.aspx")

12. [Gotchas with Reverse Pinvoke (unmanaged to managed code callbacks)](https://www.mql5.com/go?link=https://blogs.msdn.microsoft.com/davidnotario/2006/01/13/gotchas-with-reverse-pinvoke-unmanaged-to-managed-code-callbacks/ "http://blogs.msdn.com/b/davidnotario/archive/2006/01/13/512436.aspx")
13. [Mixing .NET and native code](https://www.mql5.com/go?link=https://www.codeproject.com/Articles/35041/Mixing-NET-and-native-code "http://www.codeproject.com/KB/mcpp/mixnetnative.aspx")

14. [Export Managed Code as Unmanaged](https://www.mql5.com/go?link=http://www.c-sharpcorner.com/article/export-managed-code-as-unmanaged/ "http://www.c-sharpcorner.com/UploadFile/JTeeuwen/ExportManagedCodeasUnmanaged11262005051206AM/ExportManagedCodeasUnmanaged.aspx")
15. [Understanding Classic COM Interoperability With .NET Applications](https://www.mql5.com/go?link=https://www.codeproject.com/Articles/990/Understanding-Classic-COM-Interoperability-With-NE "http://www.codeproject.com/KB/COM/cominterop.aspx")

16. [Managed Extensions for C++ Programming](https://www.mql5.com/go?link=https://www.microsoft.com/en-us/download/details.aspx?id=55979 "http://msdn.microsoft.com/en-us/library/aa712574(v=vs.71).aspx")

17. [Robert Giesecke's site](https://www.mql5.com/go?link=https://sites.google.com/site/robertgiesecke/ "http://sites.google.com/site/robertgiesecke/")
18. [MSBuild Tasks](https://www.mql5.com/go?link=https://msdn.microsoft.com/en-us/library/ms171466(v=vs.90).aspx "http://msdn.microsoft.com/en-us/library/ms171466(v=vs.90).aspx")
19. [Common Language Runtime](https://en.wikipedia.org/wiki/Common_Language_Runtime "https://en.wikipedia.org/wiki/Common_Language_Runtime")

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/249.zip "Download all attachments in the single ZIP archive")

[unmanagedexportsdllexample1.mq5](https://www.mql5.com/en/articles/download/249/unmanagedexportsdllexample1.mq5 "Download unmanagedexportsdllexample1.mq5")(1.13 KB)

[unmanagedexportsdllexample2.mq5](https://www.mql5.com/en/articles/download/249/unmanagedexportsdllexample2.mq5 "Download unmanagedexportsdllexample2.mq5")(1.27 KB)

[unmanagedexportsdllexample3.mq5](https://www.mql5.com/en/articles/download/249/unmanagedexportsdllexample3.mq5 "Download unmanagedexportsdllexample3.mq5")(1.1 KB)

[unmanagedexportsdllexample4.mq5](https://www.mql5.com/en/articles/download/249/unmanagedexportsdllexample4.mq5 "Download unmanagedexportsdllexample4.mq5")(1.07 KB)

[unmanagedexportsdllexample5.mq5](https://www.mql5.com/en/articles/download/249/unmanagedexportsdllexample5.mq5 "Download unmanagedexportsdllexample5.mq5")(1.05 KB)

[unmanagedexportsdllexample6.mq5](https://www.mql5.com/en/articles/download/249/unmanagedexportsdllexample6.mq5 "Download unmanagedexportsdllexample6.mq5")(1.89 KB)

[unmanagedexports.zip](https://www.mql5.com/en/articles/download/249/unmanagedexports.zip "Download unmanagedexports.zip")(1.02 KB)

[testme.zip](https://www.mql5.com/en/articles/download/249/testme.zip "Download testme.zip")(13.02 KB)

[testmedll.zip](https://www.mql5.com/en/articles/download/249/testmedll.zip "Download testmedll.zip")(3.22 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Securing MQL5 code: Password Protection, Key Generators, Time-limits, Remote Licenses and Advanced EA License Key Encryption Techniques](https://www.mql5.com/en/articles/359)
- [MQL5-RPC. Remote Procedure Calls from MQL5: Web Service Access and XML-RPC ATC Analyzer for Fun and Profit](https://www.mql5.com/en/articles/342)
- [Applying The Fisher Transform and Inverse Fisher Transform to Markets Analysis in MetaTrader 5](https://www.mql5.com/en/articles/303)
- [Advanced Adaptive Indicators Theory and Implementation in MQL5](https://www.mql5.com/en/articles/288)
- [Using MetaTrader 5 Indicators with ENCOG Machine Learning Framework for Timeseries Prediction](https://www.mql5.com/en/articles/252)
- [Moving Mini-Max: a New Indicator for Technical Analysis and Its Implementation in MQL5](https://www.mql5.com/en/articles/238)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/3154)**
(63)


![Trader Fortis](https://c.mql5.com/avatar/avatar_na2.png)

**[Trader Fortis](https://www.mql5.com/en/users/fortistrader)**
\|
28 Feb 2021 at 22:16

Does this approach work for .NET version 5?


![Igor Makanu](https://c.mql5.com/avatar/2018/10/5BB56740-A283.jpg)

**[Igor Makanu](https://www.mql5.com/en/users/igorm)**
\|
1 Mar 2021 at 06:15

**Trader Fortis:**

Does this approach work for .NET version 5?

I haven't tested it, but I doubt it will work

MT4 is very difficult to interact with C# - there are always some pitfalls.

It's easier to switch to MT5.

but if you want to use MT4 in principle, then as an option - launch .dll in C# according to the methodology from the article, and in it launch any C# code in a separate thread and organise the exchange, I have launched 64-bit C# libraries this way

![Trader Fortis](https://c.mql5.com/avatar/avatar_na2.png)

**[Trader Fortis](https://www.mql5.com/en/users/fortistrader)**
\|
1 Mar 2021 at 18:04

Thanks for the reply!

I have a question about MT5 - it claims native support for .NET libraries, but I can't run the library on .NET5, only on .NET4.

![Igor Makanu](https://c.mql5.com/avatar/2018/10/5BB56740-A283.jpg)

**[Igor Makanu](https://www.mql5.com/en/users/igorm)**
\|
1 Mar 2021 at 20:23

**Trader Fortis:**

Thanks for the reply!

I have a question about MT5 - it claims native support for .NET libraries, but I can't run the library on .NET5, only on .NET4.

I also had .NET5 not working on MT5 last year, I didn't figure it out.

ask the developers if and when support will be available in the topic of new MT5 releases, for example here [https://www.mql5.com/ru/forum/363680](https://www.mql5.com/ru/forum/363680).

![Moatle Thompson](https://c.mql5.com/avatar/2021/8/611EC400-9187.png)

**[Moatle Thompson](https://www.mql5.com/en/users/maot)**
\|
19 Apr 2023 at 16:30

The wrapper function , how is it done.


![MQL5 Wizard: How to Create a Risk and Money Management Module](https://c.mql5.com/2/0/CMoney_MQL5.png)[MQL5 Wizard: How to Create a Risk and Money Management Module](https://www.mql5.com/en/articles/230)

The generator of trading strategies of the MQL5 Wizard greatly simplifies testing of trading ideas. The article describes how to develop a custom risk and money management module and enable it in the MQL5 Wizard. As an example we've considered a money management algorithm, in which the size of the trade volume is determined by the results of the previous deal. The structure and format of description of the created class for the MQL5 Wizard are also discussed in the article.

![Orders, Positions and Deals in MetaTrader 5](https://c.mql5.com/2/0/TradeIndo_MQL5.png)[Orders, Positions and Deals in MetaTrader 5](https://www.mql5.com/en/articles/211)

Creating a robust trading robot cannot be done without an understanding of the mechanisms of the MetaTrader 5 trading system. The client terminal receives the information about the positions, orders, and deals from the trading server. To handle this data properly using the MQL5, it's necessary to have a good understanding of the interaction between the MQL5-program and the client terminal.

![MQL5 Wizard: How to Create a Module of Trailing of Open Positions](https://c.mql5.com/2/0/MQL5_Wizard_Trailing_Stop__1.png)[MQL5 Wizard: How to Create a Module of Trailing of Open Positions](https://www.mql5.com/en/articles/231)

The generator of trade strategies MQL5 Wizard greatly simplifies the testing of trading ideas. The article discusses how to write and connect to the generator of trade strategies MQL5 Wizard your own class of managing open positions by moving the Stop Loss level to a lossless zone when the price goes in the position direction, allowing to protect your profit decrease drawdowns when trading. It also tells about the structure and format of the description of the created class for the MQL5 Wizard.

![How to Copy Trading from MetaTrader 5 to MetaTrader 4](https://c.mql5.com/2/0/MetaTrader5_MetaTrader4_MQL5.png)[How to Copy Trading from MetaTrader 5 to MetaTrader 4](https://www.mql5.com/en/articles/189)

Is it possible to trade on a real MetaTrader 5 account today? How to organize such trading? The article contains the theory of these questions and the working codes used for copying trades from the MetaTrader 5 terminal to MetaTrader 4. The article will be useful both for the developers of Expert Advisors and for practicing traders.

[![](https://www.mql5.com/ff/sh/9nb0c8df2rmwfn89z2/01.png) MetaTrader VPS vs regular cloud hosting services8 reasons why our solution is the best option for automated tradingRead](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/450486&a=dgmsfszgoedimaicrqqmagvqzpwuxkur&s=c59e3617ccf44fd54d4c50a03b44fd689ff7507b8fe4990c83772cc5419e627d&uid=&ref=https://www.mql5.com/en/articles/249&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=6465837780010216998)

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