---
title: Cross-Platform Expert Advisor: Introduction
url: https://www.mql5.com/en/articles/2569
categories: Integration, Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T21:18:29.974211
---

[What's wrong with regular VPS?Here are the 8 most common problems that algorithmic traders may encounterRead![](https://www.mql5.com/ff/sh/hzatb686qjqxwtr4z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/450486&a=drhremihlwuaqyvgpzfddbtmgciejpba&s=c37d25bcceb93ed153b814e6ba4d4839461a9b2d68dd82b95b142be06d310f3f&uid=&ref=https://www.mql5.com/en/articles/2569&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5071735535107517812)

MetaTrader 5 / Integration


### **Table** **of Contents**

- [Introduction](https://www.mql5.com/en/articles/2569#Introduction)
- [Hello\\
World EA Sample](https://www.mql5.com/en/articles/2569#HelloWorld)
- [Source and Header Files](https://www.mql5.com/en/articles/2569#SourceAndHeader)

- [Conditional\\
Compilation](https://www.mql5.com/en/articles/2569#ConditionalCompilation)
- [Split\\
Implementation](https://www.mql5.com/en/articles/2569#SplitImplementation)
- [Splitting\\
Directories and Files](https://www.mql5.com/en/articles/2569#SplittingDirectories)

  - [Including\\
     of Files](https://www.mql5.com/en/articles/2569#IncludingOfFiles)
  - [Inheritance](https://www.mql5.com/en/articles/2569#Inheritance)

- [Limitations](https://www.mql5.com/en/articles/2569#Limitations)

- [Conclusion](https://www.mql5.com/en/articles/2569#Conclusion)

### Introduction

Among the reasons to create cross-platform experts advisors in MetaTrader
are the following:

- You are
interested in sharing expert advisors with others, regardless of
what trading platform version they use.
- You
want to understand the differences between MQL4 and MQL5.
- You
want to save time coding.
- If
MetaTrader 4 suddenly becomes legacy software, you will have less
trouble migrating your trading robots to MetaTrader 5.
- You are
already a MetaTrader 5 user, but for some reason, you want to test
your expert advisor in MetaTrader 4.

- You are
still a MetaTrader 4 user, but you would like to use the MQL5 Cloud
Service to test and optimize your trading robots.


When
developing expert advisors, and even indicators and scripts, the
developer typically pursues the following courses of action:

1. Develop
    the software using one language (MQL4 or MQL5)
2. Thoroughly
    test the developed software
3. Re-implement
    the same software for the other language


It has several disadvantages:

1. All the
    aspects of the software would need to be re-implemented, including
    parts or features which both versions share
2. Debugging
    and maintenance can be difficult
3. Reduces
    productivity


Having a
separate, parallel implementation would nearly double the amount of
code needed: one for MQL4, and another one for MQL5. Debugging and
maintenance can even be more challenging. If one version needs to be
updated, the same update may need to be introduced as well to the
other version. And due to the differences between MQL4 and MQL5, the
two versions of the same software would have to diverge at some
point. This potentially brings in more problems, since the deviations
in code are often not clearly laid out using separate, parallel
implementations.

### Hello World EA Sample

Let us begin
with a simple expert advisor written in MQL5: a hello world expert
advisor. In the said MQL version, we typically write it as
is shown in the following source code:

(HelloWorld.mq5)

```
#include <Object.mqh>
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CHelloWorld : public CObject
  {
public:
                     CHelloWorld(void);
                    ~CHelloWorld(void);
   virtual void      Greeting(void);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CHelloWorld::CHelloWorld(void)
  {
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CHelloWorld::~CHelloWorld(void)
  {
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CHelloWorld::Greeting(void)
  {
   Print("Hello World!");
  }

CHelloWorld hello;
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---

//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---

  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//---
   hello.Greeting();
   ExpertRemove();
  }
```

In MQL4, we
also write the application in the same manner:

HelloWorld.mq4

```
#include <Object.mqh>
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CHelloWorld : public CObject
  {
public:
                     CHelloWorld(void);
                    ~CHelloWorld(void);
   virtual void      Greeting(void);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CHelloWorld::CHelloWorld(void)
  {
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CHelloWorld::~CHelloWorld(void)
  {
  }

CHelloWorld hello;
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---

//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---

  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//---
   hello.Greeting();
   ExpertRemove();
  }
```

### Source and Header Files

Note that
the two source files shown earlier are identical. It is not possible
to have a single source file that is cross-platform compatible. This
is due to how the source files are being compiled:

- Compiling
an MQ4 source file results to the generation of an EX4 file
- Compiling
an MQ5 source file results to the generation of an EX5 file.



It may not
be possible to have a single source file that works on both
platforms. However, it is possible to have both source files to
reference a single header file, as what is illustrated in the following
figure:

![Source and Header Files](https://c.mql5.com/2/23/1__1.png)

Ideally, we
would want to have everything on the header file, with the two source
files only having a single line of source code: a statement linking
the header file. We can then rewrite the Hello World expert advisor header file like
the following:

HelloWorld\_SingleHeader.mqh

```
#include <Object.mqh>
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CHelloWorld : public CObject
  {
public:
                     CHelloWorld(void);
                    ~CHelloWorld(void);
   virtual void      Greeting(void);
   virtual void      Greeting(const string str1,const string str2);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CHelloWorld::CHelloWorld(void)
  {
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CHelloWorld::~CHelloWorld(void)
  {
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CHelloWorld::Greeting(void)
  {
   Print("Hello World!");
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CHelloWorld::Greeting(const string str1,const string str2)
  {
   string str=NULL;
   Print(StringConcatenate(str,str1,str2));
  }
CHelloWorld hello;
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---

//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---

  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//---
   hello.Greeting("Hello ","World!");
   ExpertRemove();
  }
//+------------------------------------------------------------------+
```

The MQL4 and MQL5 source files will each contain a single line of code, which is an #include directive to reference the header file above:

HelloWorld\_SingleHeader.mq4 and HelloWorld\_SingleHeader.mq5

```
#include <HelloWorld_SingleHeader.mqh>
```

Using this
approach has a couple of advantages. First, we can potentially
decrease the amount of source code written for the two platforms by
up to 50% (at least for this example). The second advantage is that
this setup would allow us to work on just a single implementation,
rather than two, separate ones. Since there is only one source to work on, changes made on the MQL4 version will also apply to the MQL5 version, and vice versa.

Using the
normal approach, if one is to make changes to one source file, he
also has to apply the changes separately to the other source file for
the other platform. And expert advisors are rarely written like this
example code. They are much more complex. And as an expert advisor
becomes increasingly complex, it would also become increasingly hard
to maintain two separate versions.

### Conditional Compilation

MQL4 and
MQL5 share a lot of things in common, but they also differ from each
other in many ways. Among these differences is the implementation of
the StringConcatenate function. In MQL4, the function is defined as
the following:

```
string  StringConcatenate(
   void argument1,        // first parameter of any simple type
   void argument2,        // second parameter of any simple type
   ...                    // next parameter of any simple type
   );
```

In MQL5, the
function has a slightly different implementation:

```
int  StringConcatenate(
   string&  string_var,   // string to form
   void argument1         // first parameter of any simple type
   void argument2         // second parameter of any simple type
   ...                    // next parameter of any simple type
   );
```

We can use
this function on the Hello World application by overloading the
Greeting() method of our class. The new method will accept two string
arguments, and whose concatenated result would be printed on the
terminal. We update our header file like the following:

(HelloWorld\_SingleHeader.mqh)

```
#include <Object.mqh>
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CHelloWorld : public CObject
  {
public:
                     CHelloWorld(void);
                    ~CHelloWorld(void);
   virtual void      Greeting(void);
   virtual void      Greeting(const string str1,const string str2);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CHelloWorld::CHelloWorld(void)
  {
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CHelloWorld::~CHelloWorld(void)
  {
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CHelloWorld::Greeting(void)
  {
   Print("Hello World!");
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CHelloWorld::Greeting(const string str1,const string str2)
  {
   string str=NULL;
   Print(StringConcatenate(str,str1,str2));
  }
CHelloWorld hello;
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---

//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---

  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//---
   hello.Greeting("Hello ","World!");
   ExpertRemove();
  }
//+------------------------------------------------------------------+
```

Using this updated version, in MetaTrader 4, we
would see the following result printed on the terminal:

Hello World!

In MetaTrader 5, we
would see a result different from what was originally intended:

12

In MQL4, the
function returns a string representing the concatenated text. On the
other hand, in MQL5, an integer value representing the size of the
concatenated string is returned instead. In order to make the
application exhibit the same behavior on two platforms without having
to redo most of the code, we may simply use [conditional compilation](https://www.mql5.com/en/docs/basis/preprosessor/conditional_compilation "MQL5 Reference: Conditional Compilation"),
as shown in the following code:

```
CHelloWorld::Greeting(const string str1,const string str2)
  {
   #ifdef __MQL5__
      string str=NULL;
      StringConcatenate(str,str1,str2);
      Print(str);
   #else
      Print(StringConcatenate(str1,str2));
   #endif
  }
```

Note that
this is a pre-processor directive. There might be an additional
overhead on compile time, but not on execution time. In MQL4, the
compiler would interpret the code above as the following:

```
CHelloWorld::Greeting(const string str1,const string str2)
  {
      Print(StringConcatenate(str1,str2));
}
```

On the other
hand, the MQL5 compiler will see the code as follows:

```
CHelloWorld::Greeting(const string str1,const string str2)
  {
      string str=NULL;
      StringConcatenate(str,str1,str2);
      Print(str);
}
```

### Split Implementation

At this
point, we can already understand what types of code exist in order to
create a cross-platform compatible expert advisor:

1. Compatible

   - Shared functions
   - Calculation


3. Incompatible

   - Functions that behave differently
   - Functions that are available in one, but not on the other
   - Different mode of execution

Between MQL4
and MQL5, there exists a set of functions that behave identically.
The Print() function is one example. It behaves the same no matter
which platform version an expert advisor uses. Compatible source
codes can also be seen in the form of pure calculations. The result
of 1+1 will be the same for both MQL4 and MQL5, as well as any real-world programming language. In both cases, a
split implementation is rarely needed.

In cases
where a particular portion of source code will either not compile or
will execute differently in one platform, a split implementation will
be needed. The StringConcatenate function is an example of the first
case of incompatible code. Despite having the same name, they behave differently in MQL4 and MQL5. There are also some functions
that have no direct counterpart in the other language. An example is
the OrderCalcMargin function, which, at least up to the time of
this writing, has no equivalent in MQL4. The third case is probably
the most difficult to handle for cross-platform development, as it is
here where the implementation may vary from one developer to another.
In this case, finding the common denominator between the two
platforms may be needed in order to lessen code length, and then split
implementations as necessary.

Now, solely
relying on conditional compilation may be a bad idea. As the code
becomes long, having a lot of these statements can make debugging or code maintenance very difficult. In object-oriented programming, we may
need to split the implementation into three parts: (1) the base
implementation, (2) the MQL4-specific implementation, and (3) the
MQL5-specific implementation.

The base
class implementation will contain the code that is shared by both
versions. In cases where incompatibility arises, one may deviate from
the base implementation, or even leave the base implementation empty,
and apply separate implementations for both languages.

For the
Hello World expert advisor, we declare a base class, give it a name
such as CHelloWorldBase, and then use it to contain code that is
shared by both MQL4 and MQL5. This includes the initial
Greeting() method we defined at the start of this article:

HelloWorld\_SingleHeader.mqh

```
#include <Object.mqh>
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CHelloWorldBase : public CObject
  {
public:
                     CHelloWorldBase(void);
                    ~CHelloWorldBase(void);
   virtual void      Greeting(void);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CHelloWorldBase::CHelloWorldBase(void)
  {
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CHelloWorldBase::~CHelloWorldBase(void)
  {
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CHelloWorldBase::Greeting(void)
  {
   Print("Hello World!");
  }
//+------------------------------------------------------------------+
```

We then make
the platform- or language-specific class objects to inherit from the
base class, and introduce different implementations in order to
achieve the same desired result:

HelloWorld\_SingleHeader\_MQL4.mqh

```
#include "HelloWorld_SingleHeader.mqh"
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CHelloWorld : public CHelloWorldBase
  {
public:
                     CHelloWorld(void);
                    ~CHelloWorld(void);
   virtual void      Greeting(const string str1,const string str2);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CHelloWorld::CHelloWorld(void)
  {
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CHelloWorld::~CHelloWorld(void)
  {
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CHelloWorld::Greeting(const string str1,const string str2)
  {
   Print(StringConcatenate(str1,str2));
  }
//+------------------------------------------------------------------+
```

HelloWorld\_SingleHeader\_MQL5.mqh

```
#include "HelloWorld_SingleHeader.mqh"
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CHelloWorld : public CHelloWorldBase
  {
public:
                     CHelloWorld(void);
                    ~CHelloWorld(void);
   virtual void      Greeting(const string str1,const string str2);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CHelloWorld::CHelloWorld(void)
  {
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CHelloWorld::~CHelloWorld(void)
  {
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CHelloWorld::Greeting(const string str1,const string str2)
  {
   string str=NULL;
   StringConcatenate(str,str1,str2);
   Print(str);
  }
//+------------------------------------------------------------------+
```

We then move the event functions back to where they are normally found, which is within the main source file:

HelloWorld\_SingleHeader.mq5

```
#include <HelloWorld_SingleHeader_MQL5.mqh>
CHelloWorld hello;
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---

//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---

  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//---
   hello.Greeting("Hello ","World!");
   ExpertRemove();
  }
//+------------------------------------------------------------------+
```

HelloWorld\_SingleHeader.mq4

```
#include <HelloWorld_SingleHeader_MQL4.mqh>
CHelloWorld hello;
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---

//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---

  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//---
   hello.Greeting("Hello ","World!");
   ExpertRemove();
  }
//+------------------------------------------------------------------+
```

In this particular example, it is more practical to use a single header, containing the base class, and the two descendant classes within a conditional compilation directive. However, in most cases, moving the classes to separate files is necessary, especially if the source codes involved are long.

### Including of Files

It is
natural for a developer to simply reference the header file
containing the actual class definition to be used in the program. For
example, in the MQL5 implementation of the HelloWorld expert advisor,
we can see that the two versions (HelloWorld\_SingleHeader.mq4 and HelloWorld\_SingleHeader.mq5) are virtually the same, except for the specific header file they include.

```
#include <HelloWorld_SingleHeader_MQL4.mqh>
```

```
#include <HelloWorld_SingleHeader_MQL5.mqh>
```

Another
approach, is to reference the header file containing the base
implementation. Then, at the end of that header file, we can use a
conditional compilation directive to reference the header file containing the applicable descendant,
depending on the type of compiler being used:

```
#include <Object.mqh>
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CHelloWorldBase : public CObject
  {
public:
                     CHelloWorldBase(void);
                    ~CHelloWorldBase(void);
   virtual void      Greeting(void);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CHelloWorldBase::CHelloWorldBase(void)
  {
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CHelloWorldBase::~CHelloWorldBase(void)
  {
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CHelloWorldBase::Greeting(void)
  {
   Print("Hello World!");
  }
//+------------------------------------------------------------------+
#ifdef __MQL5__
   #include "HelloWorld_SingleHeader_MQL5.mqh"
#else
   #include "HelloWorld_SingleHeader_MQL4.mqh"
#endif
//+------------------------------------------------------------------+
```

We then
reference this header file in the main source file, rather than the
language-specific header file:

```
#include <HelloWorld_SingleHeader.mqh>
CHelloWorld hello;
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---

//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---

  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//---
   hello.Greeting("Hello ","World!");
   ExpertRemove();
  }
//+------------------------------------------------------------------+
```

After this, we then remove the #include directives on the header files containing the language-specific implementation (strike-through text shows code deleted):

```
#include "HelloWorld_SingleHeader.mqh"
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CHelloWorld : public CHelloWorldBase
  {
public:
                     CHelloWorld(void);
                    ~CHelloWorld(void);
   virtual void      Greeting(const string str1,const string str2);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CHelloWorld::CHelloWorld(void)
  {
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CHelloWorld::~CHelloWorld(void)
  {
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CHelloWorld::Greeting(const string str1,const string str2)
  {
   Print(StringConcatenate(str1,str2));
  }
//+------------------------------------------------------------------+
```

```
#include "HelloWorld_SingleHeader.mqh"
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CHelloWorld : public CHelloWorldBase
  {
public:
                     CHelloWorld(void);
                    ~CHelloWorld(void);
   virtual void      Greeting(const string str1,const string str2);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CHelloWorld::CHelloWorld(void)
  {
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CHelloWorld::~CHelloWorld(void)
  {
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CHelloWorld::Greeting(const string str1,const string str2)
  {
   string str=NULL;
   StringConcatenate(str,str1,str2);
   Print(str);
  }
//+------------------------------------------------------------------+
```

This
approach is recommended and has several advantages. First, it keeps the
include directives the same for both MQL4 and MQL5 main source files. It
also saves the mental overhead of thinking about which particular header file to include and which path (e.g. include MQL4/
or MQL5/), in a given include processor directive. The third advantage
is that it keeps the base inclusions on the base header files. If one
is to use include directives on the language-specific header files,
it would be used exclusively for that version only (MQL4 or MQL5).

### Splitting Directories and Files

When
developing expert advisors in OOP, it is indeed unlikely for one to
code everything under a single class definition. One proof of this
are the trading strategy classes of the MQL5 Standard Library. As the
lines of codes increase, it may be more practical to divide the
code among various header files. This article recommends the following
directory format:

**\|-Include**

> **\|-Base**

> **\|-MQL4**

> **\|-MQL5**

The
three directories can be placed right within the Include directory
within the data folder, or in a sub-directory within the said folder.

For our example code, we will adopt the following directory structure:

**\|-Include**

> **\|-MQLx-Intro**

> > **\|-Base**
> >
> > > **HelloWorldBase.mqh**
>
> > **\|-MQL4**
> >
> > > **HelloWorld.mqh**
>
> > **\|-MQL5**
> >
> > > **HelloWorld.mqh**

Using a directory structure such as this will give our code better organization. It would also eliminate the problem of file naming clashes that we were trying to avoid earlier.

Due to the change in directory locations for our header files, we need to update our main header file for the class with the new locations of its two descendants:

```
#include <Object.mqh>
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CHelloWorldBase : public CObject
  {
public:
                     CHelloWorldBase(void);
                    ~CHelloWorldBase(void);
   virtual void      Greeting(void);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CHelloWorldBase::CHelloWorldBase(void)
  {
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CHelloWorldBase::~CHelloWorldBase(void)
  {
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CHelloWorldBase::Greeting(void)
  {
   Print("Hello World!");
  }
//+------------------------------------------------------------------+
#ifdef __MQL5__
   #include "..\MQL5\HelloWorld.mqh"
#else
   #include "..\MQL4\HelloWorld.mqh"
#endif
//+------------------------------------------------------------------+
```

We also update our main source file with the updated location for our base class. For both versions, the source files will already be identical at this phase:

HelloWorld\_Sample.mq4 and HelloWorld\_Sample.mq5

```
#include <MQLx-Intro\Base\HelloWorldBase.mqh>
CHelloWorld hello;
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---

//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---

  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//---
   hello.Greeting("Hello ","World!");
   ExpertRemove();
  }
//+------------------------------------------------------------------+
```

### Inheritance

Let us
assume that we would like to extend the CHelloWorld class we defined
earlier, like a class named CGoodByeWorld. This class will use the Greeting() method of CHelloWorld in order to create a message that says "Goodbye World!". One way to do it
(recommended) is to reference the ancestor's base class, which is
CHelloWorldBase. Then, similar to CHelloWorldBase, include a
pre-processor conditional compilation directive at the end of this
file, referencing the correct descendant. The inheritance hierarchy would look like the following

![Inheritance Hierarchy](https://c.mql5.com/2/23/2.png)

However, the
way the header files are included will be a little different:

![Include Structure](https://c.mql5.com/2/23/3.png)

The class diagram is shown in the figure below. The initial Greeting function, is in the CHelloWorldBase class, and this method is used (inherited) across all the other descendant classes. The same is true for the CGoodByeWorld class, which also a new method named GoodBye. It is also possible for this class to even extend the method for CHelloWorldBase, so that the greeting will say "goodbye" rather than "hello".

![goodbye-world-uml](https://c.mql5.com/2/24/intro-split2.png)

We only include the header files for the base class. In this case where a single class hierarchy is involved, we only include the header file of the base class with the greatest abstraction (GoodByeWorldBase.mqh), since referencing this file automatically includes the other needed header files. Notice that we do not use #include to reference platform-specific header files, since it would be the responsibility of the base header files to include them.

Our directory structure would also be updated, which at completion, would already contain the new header files:

**\|-Include**

> **\|-MQLx-Intro**

> > **\|-Base**
> >
> > > **HelloWorldBase.mqh**
> > >
> > > **GoodByeWorldBase.mqh**
>
> > **\|-MQL4**
> >
> > > **HelloWorld.mqh**
> > >
> > > **GoodByeWorld.mqh**
>
> > **\|-MQL5**
>
> > > **HelloWorld.mqh**
> > >
> > >  **GoodByeWorld.mqh**

The following is the implementation of the CGoodByeWorldBase class:

```
#include "HelloWorldBase.mqh"
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CGoodByeWorldBase : public CHelloWorld
  {
public:
                     CGoodByeWorldBase(void);
                    ~CGoodByeWorldBase(void);
   virtual void      GoodBye(void);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CGoodByeWorldBase::CGoodByeWorldBase(void)
  {
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CGoodByeWorldBase::~CGoodByeWorldBase(void)
  {
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CGoodByeWorldBase::GoodBye(void)
  {
   Greeting("Goodbye ","World!");
  }
//+------------------------------------------------------------------+
#ifdef __MQL5__
   #include "..\MQL5\GoodByeWorld.mqh"
#else
   #include "..\MQL4\GoodByeWorld.mqh"
#endif
//+------------------------------------------------------------------+
```

Note that even though the file included "HelloWorldBase.mqh", the CGoodByeWorldBase class inherits from CHelloWorld, not CHelloWorldBase. The version of CHelloWorld used will ultimately depend on the version of MQL compiler is being used. Extending CHelloWorldBase will also work in another case. However, in this example, since the Goodbye() method uses the Greeting() method, CGoodByeWorldBase will need to inherit directly from a platform-specific implementation of CHelloWorld.

Since the GoodBye() method can be shared between the two versions, it would be ideal to keep this within the base implementation. And since there is no other additional method for this class object, the descendants will be lacking any new class methods. We can then
implement the descendant in the following manner:

```
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CGoodByeWorld : public CGoodByeWorldBase
  {
  };
//+------------------------------------------------------------------+
```

The main source file would need to be updated as well, this time, instantiating an object based on CGoodByeWorld, and calling the GoodBye() method within the OnTick handler.

HelloWorld\_Sample.mq4 and HelloWorld\_Sample.mq5

```
#include <MQLx-Intro\Base\GoodByeWorldBase.mqh>
CGoodByeWorld hello;
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---

//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---

  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//---
   hello.Greeting("Hello ","World!");
   hello.GoodBye();
   ExpertRemove();
  }
//+------------------------------------------------------------------+
```

Running the versions of the expert advisor, will print the following result on the terminal:

Hello World!

Goodbye World!

ExpertRemove() function called

### Limitations

In most cases, this approach would allow a programmer to develop cross-platform expert advisors faster and more efficiently. However, readers are cautioned to be aware of certain limitations which may make it difficult or even impossible for one to apply the method demonstrated in this article:

1\. Limitations in MetaTrader 4

2\. Largely different execution or conventions between the two platforms

MetaTrader 4, being the older trading platform, lacks certain features available in MetaTrader 5. In cases where an expert advisor requires a feature that is lacking in one platform, one has to develop a custom-made solution exclusively for the other version. This is mainly a problem with MetaTrader 5 native expert advisors that will need to have a counterpart in MetaTrader 4. MetaTrader 4 users have less to worry in this regard, as most features of MetaTrader 4, if not all, have a counterpart or at least an easy workaround in MetaTrader 5.

The two platforms largely differ in some operations. This is especially true for trade operations. In this case, the developer will have to choose which convention to adopt. He may for example, use MetaTrader 4 conventions and translate them to MetaTrader 5 conventions, in order to achieve the same end-behavior. Or the exact opposite, in which case he has to apply the customary approach to trading in MetaTrader 5 to MetaTrader 4 expert advisors.

### Conclusion

In this
article, we have demonstrated a method by which cross-platform expert
advisors can be possibly developed. The said method proposed the use
of a base class, which contains the implementations shared by both
trading platforms. In areas where the two languages deviate from each
other, split implementations can be introduced as descendant classes
which inherit from this base class. The same method is repeated for
classes that would need to be defined further into the class
hierarchy. This method may prove to be helpful in developing
cross-platform applications with less time, and make code maintenance
easier by avoiding the need for implementing separate, parallel
implementations.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/2569.zip "Download all attachments in the single ZIP archive")

[MQLx-Intro.zip](https://www.mql5.com/en/articles/download/2569/mqlx-intro.zip "Download MQLx-Intro.zip")(69.5 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Cross-Platform Expert Advisor: The CExpertAdvisor and CExpertAdvisors Classes](https://www.mql5.com/en/articles/3622)
- [Cross-Platform Expert Advisor: Custom Stops, Breakeven and Trailing](https://www.mql5.com/en/articles/3621)
- [Cross-Platform Expert Advisor: Stops](https://www.mql5.com/en/articles/3620)
- [Cross-Platform Expert Advisor: Time Filters](https://www.mql5.com/en/articles/3395)
- [Cross-Platform Expert Advisor: Money Management](https://www.mql5.com/en/articles/3280)
- [Cross-Platform Expert Advisor: Signals](https://www.mql5.com/en/articles/3261)
- [Cross-Platform Expert Advisor: Order Manager](https://www.mql5.com/en/articles/2961)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/93854)**
(7)


![fxsaber](https://c.mql5.com/avatar/2019/8/5D67260D-44C9.png)

**[fxsaber](https://www.mql5.com/en/users/fxsaber)**
\|
17 Aug 2016 at 16:53

https://www.mql5.com/ru/code/16006


![Rashid Umarov](https://c.mql5.com/avatar/2012/5/4FC60566-2EEC.jpg)

**[Rashid Umarov](https://www.mql5.com/en/users/rosh)**
\|
17 Aug 2016 at 18:17

**fxsaber:**

The idea (cross-platform) is correct. But here it is suggested to create a certain meta-language and use it to write cross-platform Expert Advisors. The meta-language seems superfluous in this solution, because you can write everything in MQL4. And to run EAs not only on MT4, but also on MT5.

If I understood correctly, this is a translation of the article. Therefore, to contact the author, apparently, you need to write to the original. English version?

Yes


![Yuriy Asaulenko](https://c.mql5.com/avatar/2018/3/5AB0EFAF-DA65.png)

**[Yuriy Asaulenko](https://www.mql5.com/en/users/yuba)**
\|
17 Aug 2016 at 18:59

**MetaQuotes Software Corp.:**

Published article [Cross-platform trading advisor: Introduction](https://www.mql5.com/en/articles/2569):

Author: [Enrico Lambino](https://www.mql5.com/en/users/Iceron "Iceron")

I'm sorry, but what the hell is the point of this?


![TheXpert](https://c.mql5.com/avatar/2016/7/5783C6E7-AEEE.png)

**[TheXpert](https://www.mql5.com/en/users/thexpert)**
\|
17 Aug 2016 at 22:40

**Yuriy Asaulenko:**

I'm sorry, but what's the point?

So you don't have to do any porting.


![Vasiliy Sokolov](https://c.mql5.com/avatar/2017/9/59C3C7E4-C9E1.png)

**[Vasiliy Sokolov](https://www.mql5.com/en/users/c-4)**
\|
18 Aug 2016 at 13:19

[A cross-platform Expert Advisor](https://www.mql5.com/en/articles/2574 "Article: Cross-platform Expert Advisor: Reusing Components from the MQL5 Standard Library ") can be created only if it is based on a cross-platform trading engine, where the trading API and data access will be replaced by OO-versions, the internal implementation of which will be determined by macros #ifdef \_\_MQL5\_\_. Considering the above, the author's article is at least naive. Of course, it is great that the author discovered the #ifdef \_\_MQL5\_\_ macro, but it is not enough by itself. You need to write an engine with #ifdef at every step, and this is much more complicated.


![Graphical Interfaces VIII: the File Navigator Control (Chapter 3)](https://c.mql5.com/2/23/av8__2.png)[Graphical Interfaces VIII: the File Navigator Control (Chapter 3)](https://www.mql5.com/en/articles/2541)

In the previous chapters of the eighth part of the series, our library has been reinforced by several classes for developing mouse pointers, calendars and tree views. The current article deals with the file navigator control that can also be used as part of an MQL application graphical interface.

![Graphical Interfaces VIII: The Tree View Control (Chapter 2)](https://c.mql5.com/2/23/av8__1.png)[Graphical Interfaces VIII: The Tree View Control (Chapter 2)](https://www.mql5.com/en/articles/2539)

The previous chapter of part VIII on graphical interfaces has focused on the elements of static and drop-down calendar. The second chapter will be dedicated to an equally complex element — a tree view, that is included in every complete library used for creating graphical interfaces. A tree view implemented in this article contains multiple flexible settings and modes, thus allowing to adjust this element of control to your needs.

![LifeHack for trader: four backtests are better than one](https://c.mql5.com/2/23/ava__3.png)[LifeHack for trader: four backtests are better than one](https://www.mql5.com/en/articles/2552)

Before the first single test, every trader faces the same question — "Which of the four modes to use?" Each of the provided modes has its advantages and features, so we will do it the easy way - run all four modes at once with a single button! The article shows how to use the Win API and a little magic to see all four testing chart at the same time.

![Graphical Interfaces VIII: The Calendar Control (Chapter 1)](https://c.mql5.com/2/23/av8.png)[Graphical Interfaces VIII: The Calendar Control (Chapter 1)](https://www.mql5.com/en/articles/2537)

In the part VIII of the series of articles dedicated to creating graphical interfaces in MetaTrader, we will consider complex composite controls like calendars, tree view, and file navigator. Due to the large amount of information, there are separate articles written for every subject. The first chapter of this part describes the calendar control and its expanded version — a drop down calendar.

[Best articles and CodeBase updates in MQL5.community channelsFollow us to ensure you never miss out on important updates![](https://www.mql5.com/ff/sh/n9yf51p2srwzfqh5z2/02.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=dgazvhktsxqakdvarucjbvmvzenwlyje&s=98a038fe082e458df8c4a1d8e116e3a6646fd5517f06e48b2356b7ee005817d6&uid=&ref=https://www.mql5.com/en/articles/2569&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5071735535107517812)

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