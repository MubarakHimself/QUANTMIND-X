---
title: Promote Your Development Projects Using EX5 Libraries
url: https://www.mql5.com/en/articles/362
categories: Integration
relevance_score: 0
scraped_at: 2026-01-24T14:07:57.554741
---

[![](https://www.mql5.com/ff/sh/20jc81m23z78s5z9z2/01.png)![](https://www.mql5.com/ff/sh/20jc81m23z78s5z9z2/02.png)Create your own AI for tradingRead our book "Neural Networks in Algo Trading with MQL5"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/neurobook%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.neurobook%26utm_content=visit.page%26utm_campaign=neurobook.promo.04.2024&a=elbyupbppbqpzzvzhxtydvlupfcbmnmb&s=0d2f8feb92df3772a11aca1f195d2996b59d6539e283cdf4a18ccff02e5ad43d&uid=&ref=https://www.mql5.com/en/articles/362&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5083355715155794444)

MetaTrader 5 / Examples


### Introduction

A sophisticated reader does not need an explanation of the purpose of hiding function and class implementations in libraries. Those of you who are actively searching for new ideas may want to know that the hiding of the implementation details of classes/functions in an .ex5 file will enable you to share your know-how algorithms with other developers, set up common projects and promote them in the Web.

And while the MetaQuotes team spares no effort to bring about the possibility of direct inheritance of ex5 library classes, we are going to implement it right now.

**Table of Contents**

[1\. Export and Import of Functions](https://www.mql5.com/en/articles/362#expfunc)

[2\. Export of the Hidden Implementation of a Class](https://www.mql5.com/en/articles/362#expclass)

[3\. Initialization of Variables in .ex5 File](https://www.mql5.com/en/articles/362#expini)

[4\. Inheritance of Export Classes](https://www.mql5.com/en/articles/362#expparent)

[5\. Publishing of ex5 Libraries](https://www.mql5.com/en/articles/362#exppubl)

### 1\. Export and Import of Functions

This is a basic method underlying the export of classes. There are three key things that shall be taken into account for your functions to be available to other programs:

1. The file to be created should have the extension .mq5 (not .mqh) in order to be compiled into .ex5 file;
2. The file shall contain the _[#property](https://www.mql5.com/en/docs/basis/preprosessor/compilation) library_ preprocessor directive;

3. The key word "export" shall be put after the headers of the required exported functions

```
Example 1. Let us create a function to be used in other programs

//--- library.mq5
#property library
int libfunc (int a, int b) export
{
  int c=a+b;
  Print("a+b="+string(с));
  return(с);
}
```

After compiling this file, you will get the _library.ex5_ file from where the _libfunc_ can then be used in another program.

The process of importing functions is also very simple. It is carried out using the _#import_ preprocessor directive _._

```
Example 2. We will use the export function libfunc() in our script

//--- uses.mq5
#import "library.ex5"
  int libfunc(int a, int b);
#import

void OnStart()
{
  libfunc(1, 2);
}
```

Bear in mind that the compiler will search for .ex5 files in the _MQL5\\Libraries_ folder. So if the _library.ex5_ is not located in that folder, you will have to specify the relative pathname.

E.g.:

```
#import "..\Include\MyLib\library.ex5" // the file is located in the MQL5\Include\MyLib folder
#import "..\Experts\library.ex5" // the file is located in the MQL5\Experts\ folder
```

For your future use, functions can be imported not only into the target .mq5 file but also into .mqh files.

In order to illustrate the practical application, let us use some graphics.

We are going to create a library of functions for export. These functions will display graphical objects such as Button, Edit, Label and Rectangle Label on a chart, delete the objects from the chart and reset the color parameters of the chart.

This can be schematically shown as follows:

![Class method export scheme](https://c.mql5.com/2/3/image001_En.gif)

The complete file _Graph.mq5_ can be found at the end of the article. Here we will only give one template example of the drawing function Edit.

```
//+------------------------------------------------------------------+
//| SetEdit                                                          |
//+------------------------------------------------------------------+
void SetEdit(long achart,string name,int wnd,string text,color txtclr,color bgclr,color brdclr,
             int x,int y,int dx,int dy,int corn=0,int fontsize=8,string font="Tahoma",bool ro=false) export
  {
   ObjectCreate(achart,name,OBJ_EDIT,wnd,0,0);
   ObjectSetInteger(achart,name,OBJPROP_CORNER,corn);
   ObjectSetString(achart,name,OBJPROP_TEXT,text);
   ObjectSetInteger(achart,name,OBJPROP_COLOR,txtclr);
   ObjectSetInteger(achart,name,OBJPROP_BGCOLOR,bgclr);
   ObjectSetInteger(achart,name,OBJPROP_BORDER_COLOR,brdclr);
   ObjectSetInteger(achart,name,OBJPROP_FONTSIZE,fontsize);
   ObjectSetString(achart,name,OBJPROP_FONT,font);
   ObjectSetInteger(achart,name,OBJPROP_XDISTANCE,x);
   ObjectSetInteger(achart,name,OBJPROP_YDISTANCE,y);
   ObjectSetInteger(achart,name,OBJPROP_XSIZE,dx);
   ObjectSetInteger(achart,name,OBJPROP_YSIZE,dy);
   ObjectSetInteger(achart,name,OBJPROP_SELECTABLE,false);
   ObjectSetInteger(achart,name,OBJPROP_READONLY,ro);
   ObjectSetInteger(achart,name,OBJPROP_BORDER_TYPE,0);
   ObjectSetString(achart,name,OBJPROP_TOOLTIP,"");
  }
```

The import of the required functions and their use will be implemented in the target file Spiro.mq5:

```
Example 3. Using imported functions

//--- Spiro.mq5 – the target file of the Expert Advisor

//--- importing some graphics functions
#import "Graph.ex5"
  void SetLabel(long achart, string name, int wnd, string text, color clr,
               int x, int y, int corn=0, int fontsize=8, string font="Tahoma");
  void SetEdit(long achart, string name, int wnd, string text, color txtclr, color bgclr, color brdclr,
                 int x, int y, int dx, int dy, int corn=0, int fontsize=8, string font="Tahoma", bool ro=false);
  void SetButton(long achart, string name, int wnd, string text, color txtclr, color bgclr,
                int x, int y, int dx, int dy, int corn=0, int fontsize=8, string font="Tahoma", bool state=false);
  void HideChart(long achart, color BackClr);
#import

//--- prefix for chart objects
string sID;
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+

void OnInit()
{
  HideChart(0, clrWhite);
  sID="spiro.";
  DrawParam();
}
//+------------------------------------------------------------------+
//| DrawParam                                                        |
//+------------------------------------------------------------------+
void DrawParam()
{
  color bgclr=clrWhite, clr=clrBlack;
//--- bigger radius
  SetLabel(0, sID+"stR.", 0, "R", clr, 10, 10+3);
  SetEdit(0, sID+"R.", 0, "100", clr, bgclr, clr, 40, 10, 50, 20);
//--- smaller radius
  SetLabel(0, sID+"str.", 0, "r", clr, 10, 35+3);
  SetEdit(0, sID+"r.", 0, "30", clr, bgclr, clr, 40, 35, 50, 20);
//--- distance to the center
  SetLabel(0, sID+"stD.", 0, "D", clr, 10, 60+3);
  SetEdit(0, sID+"D.", 0, "40", clr, bgclr, clr, 40, 60, 50, 20);
//--- drawing accuracy
  SetLabel(0, sID+"stA.", 0, "Alfa", clr, 10, 85+3);
  SetEdit(0, sID+"A.", 0, "0.04", clr, bgclr, clr, 40, 85, 50, 20);
//--- drawing accuracy
  SetLabel(0, sID+"stN.", 0, "Rotor", clr, 10, 110+3);
  SetEdit(0, sID+"N.", 0, "10", clr, bgclr, clr, 40, 110, 50, 20);
//--- draw button
  SetButton(0, sID+"draw.", 0, "DRAW", bgclr, clr, 39, 135, 51, 20);
}
```

Following the run of the Expert Advisor, the objects will appear on the chart:

![Example of using library objects](https://c.mql5.com/2/3/image003__2.png)

As can be seen, the process of exporting and importing functions is not at all difficult but be sure to read about certain limitations in the Help: [export](https://www.mql5.com/en/docs/basis/function/export), [import](https://www.mql5.com/en/docs/basis/preprosessor/import).

### 2\. Export of the Hidden Implementation of a Class

Since classes in MQL5 cannot be exported directly as yet, we will have to resort to a somewhat fancy method. It is based on [polymorphism](https://www.mql5.com/en/docs/basis/oop/polymorphism) and [virtual](https://www.mql5.com/en/docs/basis/oop/virtual) functions. As a matter of fact, it is not the class itself that is returned from the ex5 module but a created object thereof. Let us call it the **hidden implementation object**.

The essence of the method is to divide the required class into two so that the declaration of functions and variables is open for public access and their implementation details are hidden in a closed .ex5 file.

This can be simply exemplified as below. There is **CSpiro** class which we would like to share with other developers without disclosing the implementation details. Assume, it contains variables, constructor, destructor and working functions.

In order to export the class, we shall do as follows:

- Create a clone of the _CSpiro_ class descendant. Let us call it _ISpiro_ (the first letter C is replaced with I, as derived from the word "interface")
- Leave all variables and dummy functions in the initial _CSpiro_ class.
- The function implementation details shall form a new _ISpiro_ class.
- Add to it the export function that will create an instance of the closed _ISpiro_.
- Note! All required functions shall have the _**virtual**_ prefix


As a result, we have two files:

```
Example 4. Hiding of the class implementation in the ex5 module

//--- Spiro.mqh – public file, the so called header file

//+------------------------------------------------------------------+
//| Class CSpiro                                                     |
//| Spirograph draw class                                       |
//+------------------------------------------------------------------+
class CSpiro
  {
public:
   //--- prefix of the chart objects
   string            m_sID;
   //--- offset of the chart center
   int               m_x0,m_y0;
   //--- color of the line
   color             m_clr;
   //--- chart parameters
   double            m_R,m_r,m_D,m_dAlfa,m_nRotate;

public:
   //--- constructor
                     CSpiro() { };
   //--- destructor
                    ~CSpiro() { };
   virtual void Init(int ax0,int ay0,color aclr,string asID) { };
   virtual void SetData(double aR,double ar,double aD,double adAlpha,double anRotate) { };

public:
   virtual void DrawSpiro() { };
   virtual void SetPoint(int x,int y) { };
  };
```

Please note that all the function classes are declared with the key word **virtual**.

```
//--- ISpiro.mq5 – hidden implementation file

#include "Spiro.mqh"

//--- importing some functions
#import "..\Experts\Spiro\Graph.ex5"
void SetPoint(long achart,string name,int awnd,int ax,int ay,color aclr);
void ObjectsDeleteAll2(long achart=0,int wnd=-1,int type=-1,string pref="",string excl="");
#import

CSpiro *iSpiro() export { return(new ISpiro); }
//+------------------------------------------------------------------+
//| Сlass ISpiro                                                     |
//| Spirograph draw class                                       |
//+------------------------------------------------------------------+
class ISpiro : public CSpiro
  {
public:
                     ISpiro() { m_x0=0; m_y0=0; };
                    ~ISpiro() { ObjectsDeleteAll(0,0,-1); };
   virtual void      Init(int ax0,int ay0,color aclr,string asID);
   virtual void      SetData(double aR,double ar,double aD,double adAlpha,double anRotate);

public:
   virtual void      DrawSpiro();
   virtual void      SetPoint(int x,int y);
  };
//+------------------------------------------------------------------+
//| Init                                                             |
//+------------------------------------------------------------------+
void ISpiro::Init(int ax0,int ay0,color aclr,string asID)
  {
   m_x0=ax0;
   m_y0=ay0;
   m_clr=aclr;
   m_sID=asID;
   m_R=0;
   m_r=0;
   m_D=0;
  }
//+------------------------------------------------------------------+
//| SetData                                                          |
//+------------------------------------------------------------------+
void ISpiro::SetData(double aR,double ar,double aD,double adAlpha,double anRotate)
  {
   m_R=aR; m_r=ar; m_D=aD; m_dAlfa=adAlpha; m_nRotate=anRotate;
  }
//+------------------------------------------------------------------+
//| DrawSpiro                                                        |
//+------------------------------------------------------------------+
void ISpiro::DrawSpiro()
  {
   if(m_r<=0) { Print("Error! r==0"); return; }
   if(m_D<=0) { Print("Error! D==0"); return; }
   if(m_dAlfa==0) { Print("Error! Alpha==0"); return; }
   ObjectsDeleteAll2(0,0,-1,m_sID+"pnt.");
   int n=0; double a=0;
   while(a<m_nRotate*2*3.1415926)
     {
      double x=(m_R-m_r)*MathCos(a)+m_D*MathCos((m_R-m_r)/m_r*a);
      double y=(m_R-m_r)*MathSin(a)-m_D*MathSin((m_R-m_r)/m_r*a);
      SetPoint(int(m_x0+x),int(m_y0+y));
      a+=m_dAlfa;
     }
   ChartRedraw(0);
  }
//+------------------------------------------------------------------+
//| SetPoint                                                         |
//+------------------------------------------------------------------+
void ISpiro::SetPoint(int x,int y)
  {
   Graph::SetPoint(0,m_sID+"pnt."+string(x)+"."+string(y),0,x,y,m_clr);
  }
//+------------------------------------------------------------------+
```

As can be seen, the hidden class has been implemented in _.mq5_ file and contains the preprocessor command _#property library._ So all the rules set forth in the previous section have been observed.

Also note the [scope resolution](https://www.mql5.com/en/docs/basis/operations/other#context_allow) operator for the _SetPoint_ function. It is declared both in the _Graph_ library and _CSpiro_ class. In order for the compiler to call the required function, we explicitly specify it using the **::** action and give the file name.

```
  Graph::SetPoint(0, m_sID+"pnt."+string(x)+"."+string(y), 0, x, y, m_clr);
```

We can now include the header file and import its implementation into our resulting Expert Advisor.

This can be schematically shown as follows:

![Scheme for working with methods of the library classes](https://c.mql5.com/2/3/image002_En.gif)

```
Example 5. Using export objects

//--- Spiro.mq5 - the target file of the Expert Advisor

//--- importing some functions
#import "Graph.ex5"
  void SetLabel(long achart, string name, int wnd, string text, color clr,
               int x, int y, int corn=0, int fontsize=8, string font="Tahoma");
  void SetEdit(long achart, string name, int wnd, string text, color txtclr, color bgclr, color brdclr,
              int x, int y, int dx, int dy, int corn=0, int fontsize=8, string font="Tahoma", bool ro=false);
  void SetButton(long achart, string name, int wnd, string text, color txtclr, color bgclr,
                int x, int y, int dx, int dy, int corn=0, int fontsize=8, string font="Tahoma", bool state=false);
  void HideChart(long achart, color BackClr);
#import

//--- including the chart class
#include <Spiro.mqh>

//--- importing the object
#import "ISpiro.ex5"
  CSpiro *iSpiro();
#import

//--- object instance
CSpiro *spiro;
//--- prefix for chart objects
string sID;
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
void OnInit()
{
  HideChart(0, clrWhite);
  sID="spiro.";
  DrawParam();
//--- object instance created
  spiro=iSpiro();
//--- initializing the drawing
  spiro.Init(250, 200, clrBlack, sID);
//--- setting the calculation parameters
  spiro.SetData(100, 30, 40, 0.04, 10);
//--- drawing
  spiro.DrawSpiro();
}
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
  delete spiro; // deleting the object
}
```

As a result, you will be able to change the object parameters in the chart and draw the chart of the object

![Graphical object parameters](https://c.mql5.com/2/3/image007__2.png)

### 3\. Initialization of Variables in .ex5 File

It is often the case that your _ISuperClass_ uses variables from the include _globals.mqh_ file. These variables can be included in a similar manner to be used in your other files.

E.g.:

```
Example 6. Public include file

//--- globals.mqh

#include <Trade\Trade.mqh>
//--- instance of the trade function object
extern CTrade *_trade;
```

The only instance of the \_ _trade_ object is initialized in your program, yet it is used in the hidden _ISuperClass_ class.

For this purpose, a pointer to the object you created shall be passed from the _ISuperClass_ class to the .ex5 file.

It is easiest done when the object is received from the .ex5 file, as below:

```
Example 7. Initialization of variables upon creation of the object

//--- ISuperClass.mq5 –hidden implementation file

#property library
CSuperClass *iSuperClass(CTrade *atrade) export
{
//--- saving the pointer
   _trade=atrade;
//--- returning the object of the hidden implementation of ISuperClass of the open CSuperClass class
  return(new ISuperClass);
}
//... the remaining code
```

Thus, all the required variables are initialized upon receipt of the object in its module.

In fact, there may be a lot of public global variables which may be of different types. Those who are not eager to change the header of the _iSuperClass_ function all the time, should better create a special **class aggregating** all global variables and functions for working with it.

```
Example 8. Public include file

//--- globals.mqh
#include <Trade\Trade.mqh>

//--- trade "object"
extern CTrade *_trade;
//--- name of the Expert Advisor of the system
extern string _eaname;

//+------------------------------------------------------------------+
//| class __extern                                                   |
//+------------------------------------------------------------------+
class __extern // all extern parameters for passing between the ex5 modules are accumulated here
{
public:
//--- the list of all public global variables to be passed
//--- trade "object"
  CTrade *trade;
//--- name of the Expert Advisor of the system
  string eaname;

public:
  __extern() { };
  ~__extern() { };

//--- it is called when passing the parameters into the .ex5 file
  void Get() { trade=_trade; eaname=_eaname; };  // getting the variables

 //--- it is called in the .ex5 file
  void Set() { _trade=trade; _eaname=eaname; };  // setting the variables

};
//--- getting the variables and pointer for passing the object into the .ex5 file
__extern *_GetExt() { _ext.Get(); return(GetPointer(_ext)); }

//--- the only instance for operation
extern __extern _ext;
```

The _ISuperClass.mq5_ file will be implemented as follows:

```
Example 9.

//--- ISuperClass.mq5 –hidden implementation file

#property library
CSuperClass *iSuperClass(__extern *aext) export
{
//--- taking in all the parameters
  aext.Set();
//--- returning the object
  return(new ISuperClass);
}
//--- ... the remaining code
```

The function call will now be transformed into a simplified and, what is most important, extensible form.

```
Example 10. Using export objects in the presence of public global variables

//--- including global variables (usually located in SuperClass.mqh)
#include "globals.mqh"

//--- including the public header class
#include "SuperClass.mqh"
//--- getting the hidden implementation object
#import "ISuperClass.ex5"
  CSuperClass *iSuperClass();
#import

//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
{
//--- creating the hidden implementation object providing for the passing of all parameters
  CSuperClass *sc=iSuperClass(_GetExt());
  //--- ... the remaining code
}
```

### 4\. Inheritance of Export Classes

You must have already understood that this way of exporting objects implies that the direct and simple inheritance is out of the question. Export of the hidden implementation object suggests that the object itself is the last link of the inheritance chain and is the one that can ultimately be used.

In the general case you can create an "emulation" of inheritance by writing an additional intermediate class. And here we will of course need polymorphism and virtuality.

```
Example 11. Emulation of inheritance of hidden classes

//--- including the public header class
#include "SuperClass.mqh"

//--- getting the hidden implementation object
#import "ISuperClass.ex5"
  CSuperClass *iSuperClass();
#import

class _CSuperClass
{
public:
//--- instance of the hidden implementation object
  CSuperClass *_base;
public:
//--- constructor
  _CSuperClass() {  _base=iSuperClass(_GetExt()); };
//--- destructor
  ~_CSuperClass() { delete _base; };
//--- further followed by all functions of the base CSuperClass class
//--- working function called from the hidden implementation object
  virtual int func(int a, int b) { _base.func(a,b); };
};
```

The only issue here is access to variables of _CSuperClass_. As can be seen, they are not present in declaration of the descendant and are located in the variable \_ _base_. Usually it does not affect the usability provided that there is a header class _SuperClass.mqh_.

Naturally, if you are mainly focused on know-how functions, you do not have to create a wrapper of _ISuperClass_ with regard to them in advance. It will suffice to export those know-how functions and let the outside developers create their own wrapper classes which will then be easy to inherit.

Thus, when preparing your developments for other developers, you should care to create a whole set of necessary export functions, .mqh and .ex5 files and classes:

1. Export of class independent functions
2. Header .mqh files and their .ex5 implementations
3. Initialization of variables in the .ex5 files

### 5\. Publishing of ex5 Libraries

In November 2011, MetaQuotes started to provide access to a files repository. More on that can be found in the [announcement](https://www.mql5.com/en/forum/5155).

This repository allows you to store your developments and, what is more important, to provide access thereto for other developers. This tool you will enable you to easily publish new versions of your files to ensure fast access to them for the developers who may be using these files.

Moreover, the company website gives you an opportunity to offer your own function libraries in the [Market](https://www.mql5.com/en/market) on a commercial basis or free of charge.

### Conclusion

You now know how to create ex5 libraries with export of their functions or class objects and can apply your knowledge in practice. All these resources will allow you to establish a closer cooperation with other developers: to work on common projects, promote them in the Market or provide access to ex5 library functions.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/362](https://www.mql5.com/ru/articles/362)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/362.zip "Download all attachments in the single ZIP archive")

[spiro.zip](https://www.mql5.com/en/articles/download/362/spiro.zip "Download spiro.zip")(3.91 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Working with sockets in MQL, or How to become a signal provider](https://www.mql5.com/en/articles/2599)
- [SQL and MQL5: Working with SQLite Database](https://www.mql5.com/en/articles/862)
- [Getting Rid of Self-Made DLLs](https://www.mql5.com/en/articles/364)
- [Using WinInet in MQL5. Part 2: POST Requests and Files](https://www.mql5.com/en/articles/276)
- [Tracing, Debugging and Structural Analysis of Source Code](https://www.mql5.com/en/articles/272)
- [The Prototype of a Trading Robot](https://www.mql5.com/en/articles/132)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/6216)**
(21)


![Alain Verleyen](https://c.mql5.com/avatar/2024/5/663a6cdf-e866.jpg)

**[Alain Verleyen](https://www.mql5.com/en/users/angevoyageur)**
\|
26 Aug 2014 at 10:04

**Renat:**

...

Is it planned to implement [export](https://www.mql5.com/en/economic-calendar/japan/exports-yy "Volume of exports (Exports )") for Class or something similar ?


![MetaQuotes](https://c.mql5.com/avatar/2009/11/4AF883AB-83DE.jpg)

**[Renat Fatkhullin](https://www.mql5.com/en/users/renat)**
\|
26 Aug 2014 at 10:14

**angevoyageur:**

Is it planned to implement export for Class or something similar ?

Yes, but not now.


![Alain Verleyen](https://c.mql5.com/avatar/2024/5/663a6cdf-e866.jpg)

**[Alain Verleyen](https://www.mql5.com/en/users/angevoyageur)**
\|
26 Aug 2014 at 10:16

**Renat:**

Yes, but not now.

Thank you.


![Alexandr Gavrilin](https://c.mql5.com/avatar/2025/12/694aad80-f58e.png)

**[Alexandr Gavrilin](https://www.mql5.com/en/users/dken)**
\|
19 Nov 2021 at 07:04

At least some possibility to export classes.

Will it work in MT4?

![Alain Verleyen](https://c.mql5.com/avatar/2024/5/663a6cdf-e866.jpg)

**[Alain Verleyen](https://www.mql5.com/en/users/angevoyageur)**
\|
19 Nov 2021 at 13:56

**Alexandr Gavrilin [#](https://www.mql5.com/ru/forum/5815/page2#comment_25963897) :**

Well, at least there's some possibility of exporting classes.

Will it work in MT4?

After 7 years it's still "not now".

I forgot about MT4, it's in the past.

![Trademinator 3: Rise of the Trading Machines](https://c.mql5.com/2/0/Terminator_3_Rise_of_the_Machines.png)[Trademinator 3: Rise of the Trading Machines](https://www.mql5.com/en/articles/350)

In the article "Dr. Tradelove..." we created an Expert Advisor, which independently optimizes parameters of a pre-selected trading system. Moreover, we decided to create an Expert Advisor that can not only optimize parameters of one trading system underlying the EA, but also select the best one of several trading systems. Let's see what can come of it...

![Securing MQL5 code: Password Protection, Key Generators, Time-limits, Remote Licenses and Advanced EA License Key Encryption Techniques](https://c.mql5.com/2/0/MQL5_protection_methods.png)[Securing MQL5 code: Password Protection, Key Generators, Time-limits, Remote Licenses and Advanced EA License Key Encryption Techniques](https://www.mql5.com/en/articles/359)

Most developers need to have their code secured. This article will present a few different ways to protect MQL5 software - it presents methods to provide licensing capabilities to MQL5 Scripts, Expert Advisors and Indicators. It covers password protection, key generators, account license, time-limit evaluation and remote protection using MQL5-RPC calls.

![Time Series Forecasting Using Exponential Smoothing (continued)](https://c.mql5.com/2/0/Exponent_Smoothing2.png)[Time Series Forecasting Using Exponential Smoothing (continued)](https://www.mql5.com/en/articles/346)

This article seeks to upgrade the indicator created earlier on and briefly deals with a method for estimating forecast confidence intervals using bootstrapping and quantiles. As a result, we will get the forecast indicator and scripts to be used for estimation of the forecast accuracy.

![The All or Nothing Forex Strategy](https://c.mql5.com/2/0/allVSzero.png)[The All or Nothing Forex Strategy](https://www.mql5.com/en/articles/336)

The purpose of this article is to create the most simple trading strategy that implements the "All or Nothing" gaming principle. We don't want to create a profitable Expert Advisor - the goal is to increase the initial deposit several times with the highest possible probability. Is it possible to hit the jackpot on ForEx or lose everything without knowing anything about technical analysis and without using any indicators?

[![](https://www.mql5.com/ff/sh/x8fwvn495ta7y774z2/01.png)Does your broker offer sponsored hosting for trading?Now it's even easier to get MetaTrader VPS for free – contact your broker for details](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=xscnzeyhifcgygpwvysykhqydcmmbgpp&s=f87b748147e376d34c8f0fdb9737b1766f20cc2174769a0e6b9975b5c2e8ddae&uid=&ref=https://www.mql5.com/en/articles/362&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5083355715155794444)

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