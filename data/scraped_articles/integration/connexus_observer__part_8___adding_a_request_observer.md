---
title: Connexus Observer (Part 8): Adding a Request Observer
url: https://www.mql5.com/en/articles/16377
categories: Integration, Expert Advisors
relevance_score: 0
scraped_at: 2026-01-24T14:02:40.289431
---

[![](https://www.mql5.com/ff/sh/qv94j0cd8n2n55z9z2/01.png)![](https://www.mql5.com/ff/sh/qv94j0cd8n2n55z9z2/02.png)Boost your trading experienceRead our book "MQL5 Programming for Traders"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/book%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.algobook%26utm_content=visit.page%26utm_campaign=algobook.promo.04.2024&a=heclgjpfbvfghpmyaciuaesdtswflupo&s=4255fbe1b8cbc4d1b40afbaebf4235e5ace8b5103cba60d996897a03d588556f&uid=&ref=https://www.mql5.com/en/articles/16377&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5083299670127548622)

MetaTrader 5 / Examples


### Introduction

This article is the continuation of a series of articles where we will build a library called Connexus. In the [first article](https://www.mql5.com/en/articles/15795), we understood the basic operation of the [WebRequest](https://www.mql5.com/en/docs/network/webrequest) function, understanding each of its parameters and also created an example code that demonstrates the use of this function and its difficulties. In the last article, we created the client layer, a simple and intuitive class responsible for sending requests, receiving a request object (CHttpRequest) and returning a response (CHttpResponse) that contains information about the request, such as status code, duration, body and response header. We also created a decoupling of the class with the WebRequest function, making the library more flexible, creating a new layer called CHttpTransport.

In this eighth article of the series, we will understand and implement an Observer in the library, to facilitate the management of multiple requests by the client. Let's go!

Just to remind you of the current state of the library, here is the current diagram:

![Diagram1](https://c.mql5.com/2/101/diagram1.png)

### What is an Observer?

Imagine the Observer as that friend who, half hidden, just watches from afar, paying attention to everything, but without interfering. In the programming world, the Observer pattern does something very similar: it allows certain objects to be "notified" when something changes, but without needing to know exactly who caused the change. It's almost like a magic touch: someone moves a part and whoever needs it, at the time, already knows. This pattern is one of the classics for keeping the gears of business logic and the interface working side by side. This way, the system gains that air of fluidity, with several parts automatically adjusting to events.

The idea was born to solve one of those annoying problems in very rigid systems, where one object depends on another and is "stuck" to it, without much room to breathe. The solution? Decouple, make it lighter. Before the Observer gained a name and surname, programmers were already looking to make systems lighter. It was then that, back in 1994, [\*\*Erich Gamma](https://en.wikipedia.org/wiki/Erich_Gamma "https://en.wikipedia.org/wiki/Erich_Gamma"), [Richard Helm](https://en.wikipedia.org/wiki/Richard_Helm "https://en.wikipedia.org/wiki/Richard_Helm"), [Ralph Johnson](https://en.wikipedia.org/wiki/Ralph_Johnson "https://en.wikipedia.org/wiki/Ralph_Johnson") and [John Vlissides](https://en.wikipedia.org/wiki/John_Vlissides "https://en.wikipedia.org/wiki/John_Vlissides")\\*\\* in their work [_Design Patterns: Elements of Reusable Object-Oriented Software_ (1994)](https://en.wikipedia.org/wiki/Design_Patterns "https://en.wikipedia.org/wiki/Design_Patterns") put forward the Observer as the ideal way to keep multiple objects up to date with changes to a single object, without the chains of a strong binding.

### Why Use the Observer Pattern?

The Observer is perfect for when we need decoupling, when we want to make things more independent from each other. The Subject doesn't need to know who is watching, it just needs to shout "change in sight!" and move on. It can also be useful for real-time updates\*\*,\*\* systems that need to update instantly, such as interactive interfaces or automatic notifications, are much more agile with the Observer.

### Components of the Observer Pattern

1. **Subject**: This is the "owner of the piece" whose state changes and that needs to notify the observers about these changes. It keeps a list of Observers and has methods to add or remove someone from the list.
2. **Observer**: Each Observer is like a "listener", always ready to react to the Subject's changes. It implements an update method that the Subject calls every time a change occurs.

I'll add a diagram below that shows how the observer pattern works:

![Diagram 2](https://c.mql5.com/2/101/diagram2.png)

1. Main Classes
   - **Subject**: This class maintains a collection of observers ( observerCollection ) and provides methods to manage these observers. Its function is to notify observers whenever a state change occurs.
     - **Methods:**
       - registerObserver(observer) : Adds an observer to the collection.
       - unregisterObserver(observer) : Removes an observer from the collection.
       - notifyObservers() : Notifies all observers by calling the update() method of each observer in the observerCollection .
   - **Observer**: This is an interface or abstract class that defines the update() method. All concrete observer classes (concretions of Observer ) must implement this method, which is called when the Subject notifies changes.
2. Concrete Classes
   - **ConcreteObserverA** and **ConcreteObserverB**: These are concrete implementations of the Observer interface. Each implements the update() method, which defines the specific response to a change in the Subject .
3. Relationship between Subject and Observer
   - The Subject maintains a list of Observers and notifies them by calling observer.update() for each observer in the collection.
   - The concrete Observers react to changes that occur in the Subject according to their specific implementation in the update() method.

How will this be useful in the Connexus library? We will use this pattern to inform the client code when a request was sent, when a response was received, or even when an unexpected error was generated. Using this pattern, the client will be informed that this happened. This makes it easier to use the library, because it avoids creating conditions in the code such as, for example, “if an error was generated, do this”, “if a request was made, then do this”, “if a response was received, then do this”.

### Hands on code

First I'll demonstrate a diagram that shows how we're going to add this pattern in the context of the library:

![Diagram 3](https://c.mql5.com/2/101/diagram3.png)

Let's understand better how the implementation will be

1. Note that the diagram has the same structure as the reference diagram
2. I added two methods that the observers will have access to:
3. OnSend() → When a request is sent.
4. OnRecv() → When a response is obtained.
5. IHttpObserver will not be an abstract class, but rather an interface.

### Creating the IHttpClient interface

First we create the IHttpClient interface at the path <Connexus/Interface/IHttpClient.mqh> . And we define the two notification functions

```
//+------------------------------------------------------------------+
//|                                                IHttpObserver.mqh |
//|                                  Copyright 2024, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
#include "../Core/HttpRequest.mqh"
#include "../Core/HttpResponse.mqh"
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
interface IHttpObserver
  {
   void OnSend(CHttpRequest *request);
   void OnRecv(CHttpResponse *response);
  };
//+------------------------------------------------------------------+
```

### Creating the list of observers in CHttpClient

Let's add the interface import.

```
#include "../Interface/IHttpObserver.mqh"
```

Now let's create an array of observers in the private field of the class. Remember that this array should store [pointers](https://www.mql5.com/en/docs/basis/types/object_pointers), so we need to add the “\*” before the variable name. We will also create public methods for adding, removing and notifying all observers that are stored in the array.

```
//+------------------------------------------------------------------+
//|                                                   HttpClient.mqh |
//|                                  Copyright 2024, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include "HttpRequest.mqh"
#include "HttpResponse.mqh"
#include "../Constants/HttpMethod.mqh"
#include "../Interface/IHttpTransport.mqh"
#include "../Interface/IHttpObserver.mqh"
#include "HttpTransport.mqh"
//+------------------------------------------------------------------+
//| class : CHttpClient                                              |
//|                                                                  |
//| [PROPERTY]                                                       |
//| Name        : CHttpClient                                        |
//| Heritage    : No heritage                                        |
//| Description : Class responsible for linking the request and      |
//|               response object with the transport layer.          |
//|                                                                  |
//+------------------------------------------------------------------+
class CHttpClient
  {
private:
   IHttpObserver     *m_observers[];                     // Array of observers

public:
   //--- Observers
   void              RegisterObserver(IHttpObserver *observer);
   void              UnregisterObserver(IHttpObserver *observer);
   void              OnSendNotifyObservers(CHttpRequest *request);
   void              OnRecvNotifyObservers(CHttpResponse *response);
  };
//+------------------------------------------------------------------+
//| Add observer pointer to observer list                            |
//+------------------------------------------------------------------+
void CHttpClient::RegisterObserver(IHttpObserver *observer)
  {
   int size = ArraySize(m_observers);
   ArrayResize(m_observers,size+1);
   m_observers[size] = observer;
  }
//+------------------------------------------------------------------+
//| Remove observer pointer to observer list                         |
//+------------------------------------------------------------------+
void CHttpClient::UnregisterObserver(IHttpObserver *observer)
  {
   int size = ArraySize(m_observers);
   for(int i=0;i<size;i++)
     {
      if(GetPointer(m_observers[i]) == GetPointer(observer))
        {
         ArrayRemove(m_observers,i,1);
         break;
        }
     }
  }
//+------------------------------------------------------------------+
//| Notifies observers that a request has been made                  |
//+------------------------------------------------------------------+
void CHttpClient::OnSendNotifyObservers(CHttpRequest *request)
  {
   int size = ArraySize(m_observers);
   for(int i=0;i<size;i++)
     {
      m_observers[i].OnSend(request);
     }
  }
//+------------------------------------------------------------------+
//| Notifies observers that a response has been received             |
//+------------------------------------------------------------------+
void CHttpClient::OnRecvNotifyObservers(CHttpResponse *response)
  {
   int size = ArraySize(m_observers);
   for(int i=0;i<size;i++)
     {
      m_observers[i].OnRecv(response);
     }
  }
//+------------------------------------------------------------------+
```

Finally, we call the notification function inside the function that sends the request so that the observers are actually informed:

```
//+------------------------------------------------------------------+
//| class : CHttpClient                                              |
//|                                                                  |
//| [PROPERTY]                                                       |
//| Name        : CHttpClient                                        |
//| Heritage    : No heritage                                        |
//| Description : Class responsible for linking the request and      |
//|               response object with the transport layer.          |
//|                                                                  |
//+------------------------------------------------------------------+
class CHttpClient
  {
private:

   IHttpObserver     *m_observers[];                     // Array of observers

public:
   //--- Basis function
   bool              Send(CHttpRequest &request, CHttpResponse &response);
  };
//+------------------------------------------------------------------+
//| Basis function                                                   |
//+------------------------------------------------------------------+
bool CHttpClient::Send(CHttpRequest &request, CHttpResponse &response)
  {
   //--- Request
   uchar body_request[];
   request.Body().GetAsBinary(body_request);

   //--- Response
   uchar body_response[];
   string headers_response;

   //--- Notify observer of request
   this.OnSendNotifyObservers(GetPointer(request));

   //--- Send
   ulong start = GetMicrosecondCount();
   int status_code = m_transport.Request(request.Method().GetMethodDescription(),request.Url().FullUrl(),request.Header().Serialize(),request.Timeout(),body_request,body_response,headers_response);
   ulong end = GetMicrosecondCount();

   //--- Notify observer of response
   this.OnRecvNotifyObservers(GetPointer(response));

   //--- Add data in Response
   response.Clear();
   response.Duration((end-start)/1000);
   response.StatusCode() = status_code;
   response.Body().AddBinary(body_response);
   response.Header().Parse(headers_response);

   //--- Return is success
   return(response.StatusCode().IsSuccess());
  }
//+------------------------------------------------------------------+
```

Work done, and simpler than it seems when we write the code, isn't it? With this, we have completed the entire implementation within the library. We need to create the observers, that is, the concrete classes that implement the IHttpObserver. We will do this in the next topic, the tests.

### Tests

Now all we need to do is use the library. To do this, I will create a new test file called TestObserver.mq5, in the pat <Experts/Connexus/Tests/TestObserver.mq5>. We will import the library and leave only the OnInit() event.

```
//+------------------------------------------------------------------+
//|                                                 TestObserver.mq5 |
//|                                  Copyright 2024, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
#include <Connexus/Core/HttpClient.mqh>
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
```

Just below the import I will create a concrete class that implements the IHttpClient interface, it will just print on the terminal console the data that was sent and received using the library:

```
//+------------------------------------------------------------------+
//|                                                 TestObserver.mq5 |
//|                                  Copyright 2024, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
#include <Connexus/Core/HttpClient.mqh>
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CMyObserver : public IHttpObserver
  {
public:
                     CMyObserver(void);
                    ~CMyObserver(void);

   void              OnSend(CHttpRequest *request);
   void              OnRecv(CHttpResponse *response);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CMyObserver::CMyObserver(void)
  {
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CMyObserver::~CMyObserver(void)
  {
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CMyObserver::OnSend(CHttpRequest *request)
  {
   Print("-----------------------------------------------");
   Print("Order sent notification received in CMyObserver");
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CMyObserver::OnRecv(CHttpResponse *response)
  {
   Print("-----------------------------------------------");
   Print("Response notification received in CMyObserver");
  }
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   //--- Create objects
   CHttpRequest request;
   CHttpResponse response;
   CHttpClient client;
   CMyObserver *my_observer = new CMyObserver();

   //--- Configure request
   request.Method() = HTTP_METHOD_GET;
   request.Url().Parse("https://httpbin.org/get");

   //--- Adding observer
   client.RegisterObserver(my_observer);

   //--- Send
   client.Send(request,response);

   //--- Delete pointer
   delete my_observer;
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
```

When running this on the graph we get this result:

![](https://c.mql5.com/2/101/1393212663119.png)

This shows that the functions of the CMyObserver class were called within the library, this changes everything, we complete the library with the main objective accomplished, that it be flexible.

The most interesting part is that we can have several observers in different parts of the code, if we have an EA that is divided into several classes, we can have each of these classes create an implementation of IHttpObserver and that's it! We will be notified as soon as the request is sent or a response is received.

Now with these observer inclusions this is the current diagram of the library:

![Diagram 4](https://c.mql5.com/2/101/diagram4.png)

### Refactoring the folders

Currently the directory of all the library files looks like this:

```
|--- Connexus
|--- |--- Constants
|--- |--- |--- HttpMethod.mqh
|--- |--- |--- HttpStatusCode.mqh
|--- |--- Core
|--- |--- |--- HttpClient.mqh
|--- |--- |--- HttpRequest.mqh
|--- |--- |--- HttpResponse.mqh
|--- |--- |--- HttpTransport.mqh
|--- |--- Data
|--- |--- |--- Json.mqh
|--- |--- Header
|--- |--- |--- HttpBody.mqh
|--- |--- |--- HttpHeader.mqh
|--- |--- Interface
|--- |--- |--- IHttpObserver.mqh
|--- |--- |--- IHttpTransport.mqh
|--- |--- URL
|--- |--- |--- QueryParam.mqh
|--- |--- |--- URL.mqh
```

We'll make two adjustments: include the files from the URL folder in the Data folder, and rename them to Utils, thus simplifying both folders that contain files with similar purposes. We'll also add the interfaces folder inside the Core folder, since the interfaces are part of the core of the library. In the end, the library's folder structure looks like this:

```
|--- Connexus
|--- |--- Constants
|--- |--- |--- HttpMethod.mqh
|--- |--- |--- HttpStatusCode.mqh
|--- |--- Core
|--- |--- |--- Interface
|--- |--- |--- |--- IHttpObserver.mqh
|--- |--- |--- |--- IHttpTransport.mqh
|--- |--- |--- HttpClient.mqh
|--- |--- |--- HttpRequest.mqh
|--- |--- |--- HttpResponse.mqh
|--- |--- |--- HttpTransport.mqh
|--- |--- Utils
|--- |--- |--- Json.mqh
|--- |--- |--- QueryParam.mqh
|--- |--- |--- URL.mqh
|--- |--- Header
|--- |--- |--- HttpBody.mqh
|--- |--- |--- HttpHeader.mqh
```

### Renaming some methods

When it comes to writing code that is easy to understand, maintain, and improve, adopting a **standard coding style** makes all the difference. Having a consistent standard when creating libraries goes far beyond simple aesthetics; it brings clarity, predictability, and a solid foundation for anyone who will use or collaborate with the code, today or in the future. This uniform style is not just a matter of organization; it is an investment in quality, robustness, and healthy growth of the library over time. And even though it may seem like just a detail at first, it ends up being the guiding thread that makes the code safe and more prepared to evolve.

### Why is having a Standard Style Essential?

- **Consistency and Readability**: Well-structured code with a uniform style makes reading more fluid and understandable for any developer. With a well-defined standard, people don't need to waste time deciphering variations or inconsistencies; instead, they can focus on what really matters: the logic of the code. Aspects such as spacing, indentation, and naming are details that, together, create a more intuitive and straightforward experience. Everything is aligned, facilitating navigation and reducing the stumbling blocks caused by varied and disconnected styles.
- **Ease of Maintenance and Expansion**: Libraries rarely stand still in time; new challenges arise and it is natural that they need adjustments. With a standardized coding style, maintenance becomes simpler and less prone to errors. This not only saves time when fixing problems, but also makes it easier for new developers to quickly understand the code and collaborate efficiently. And, of course, a library that is well-structured from the beginning is much easier to scale, since each new feature finds a predictable and organized environment in which to integrate.

That said, let's define some standards in the code, mainly in the naming of functions. Some other standards have already been applied, such as:

- All classes use the prefix “ C ” before the name
- All interfaces use the prefix “ I ” before the name
- Private variables use the prefix “ m\_ ”
- Methods must always start with capital letters
- ENUM values must be written in uppercase

All these standards have already been applied in the library during development, but we will add others, such as:

- Methods to set/get attributes of a class must use the prefix Get or Set
- Methods that return the size of the array must be named “ Size ”
- Methods that reset the class attributes must be named “ Clear ”
- Methods to convert to string must be named “ ToString ”
- Avoid name redundancy in context classes. For example, the CQueryParam class has the AddParam() method, which doesn't make sense. The ideal would be just Add() since we are already in the context of parameters.

That said, I will not list all the methods in the library that I will rename, nor will I provide the source code, since I am not changing the implementation of the method, only the name. But with the changes, I will leave a diagram below that shows all the classes in the library with the names of the updated methods and their relationships.

![Diagram 5](https://c.mql5.com/2/101/diagram5.png)

### Conclusion

With this last article, we conclude the series on the creation of the Connexus library, designed to simplify HTTP communication. It was quite a journey: we went through the basics, delved into more advanced design and code refinement techniques, and explored the Observer pattern to give Connexus the reactivity that is essential in dynamic and flexible systems. We implemented this pattern in practice, so that various parts of the application can automatically react to changes, creating a robust and adaptable structure.

In addition to the Observer, we organized the entire file and folder architecture, making the code modular and intuitive. We also renamed methods to increase clarity and make the use of the library more direct and consistent, details that make all the difference when it comes to clean code and long-term maintenance.

Connexus was designed to make HTTP integration as simple and intuitive as possible, and we hope that this series has shown each important point in the process, revealing the design choices that made this possible. With this final article, I hope that Connexus not only simplifies your HTTP integrations, but also inspires continuous improvements. Thank you for embarking on this journey with me, may Connexus be an ally in your projects!

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/16377.zip "Download all attachments in the single ZIP archive")

[Connexus\_Observer\_0Part\_8y\_Adding\_a\_Request\_Observer.zip](https://www.mql5.com/en/articles/download/16377/connexus_observer_0part_8y_adding_a_request_observer.zip "Download Connexus_Observer_0Part_8y_Adding_a_Request_Observer.zip")(31.69 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Simplifying Databases in MQL5 (Part 2): Using metaprogramming to create entities](https://www.mql5.com/en/articles/19594)
- [Simplifying Databases in MQL5 (Part 1): Introduction to Databases and SQL](https://www.mql5.com/en/articles/19285)
- [Mastering Log Records (Part 10): Avoiding Log Replay by Implementing a Suppression](https://www.mql5.com/en/articles/19014)
- [Mastering Log Records (Part 9): Implementing the builder pattern and adding default configurations](https://www.mql5.com/en/articles/18602)
- [Mastering Log Records (Part 8): Error Records That Translate Themselves](https://www.mql5.com/en/articles/18467)
- [Mastering Log Records (Part 7): How to Show Logs on Chart](https://www.mql5.com/en/articles/18291)
- [Mastering Log Records (Part 6): Saving logs to database](https://www.mql5.com/en/articles/17709)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/476889)**
(2)


![Rashid Umarov](https://c.mql5.com/avatar/2012/5/4FC60566-2EEC.jpg)

**[Rashid Umarov](https://www.mql5.com/en/users/rosh)**
\|
30 Sep 2025 at 12:39

**Kristian Kafarov [#](https://www.mql5.com/ru/forum/489355#comment_58155846):**

Hello! I copied all the files from this article + additional files from the previous one to the MQL5 folder. Here is what I got when trying to compile Connexus\\Test\\TestRequest.mq5:

Flash to the very first error, fix it and everything will work


![Rashid Umarov](https://c.mql5.com/avatar/2012/5/4FC60566-2EEC.jpg)

**[Rashid Umarov](https://www.mql5.com/en/users/rosh)**
\|
30 Sep 2025 at 13:41

In short, implicit signed/unsigned array type conversion is no longer allowed.

Some changes need to be made to the code.

![Automating Trading Strategies in MQL5 (Part 1): The Profitunity System (Trading Chaos by Bill Williams)](https://c.mql5.com/2/102/Automating_Trading_Strategies_in_MQL5_Part_1_LOGO.png)[Automating Trading Strategies in MQL5 (Part 1): The Profitunity System (Trading Chaos by Bill Williams)](https://www.mql5.com/en/articles/16365)

In this article, we examine the Profitunity System by Bill Williams, breaking down its core components and unique approach to trading within market chaos. We guide readers through implementing the system in MQL5, focusing on automating key indicators and entry/exit signals. Finally, we test and optimize the strategy, providing insights into its performance across various market scenarios.

![Chemical reaction optimization (CRO) algorithm (Part I): Process chemistry in optimization](https://c.mql5.com/2/81/Algorithm_for_optimization_by_chemical_reactions__LOGO___2.png)[Chemical reaction optimization (CRO) algorithm (Part I): Process chemistry in optimization](https://www.mql5.com/en/articles/15041)

In the first part of this article, we will dive into the world of chemical reactions and discover a new approach to optimization! Chemical reaction optimization (CRO) uses principles derived from the laws of thermodynamics to achieve efficient results. We will reveal the secrets of decomposition, synthesis and other chemical processes that became the basis of this innovative method.

![Neural Networks Made Easy (Part 93): Adaptive Forecasting in Frequency and Time Domains (Final Part)](https://c.mql5.com/2/80/Neural_networks_are_easy_Part_93____LOGO.png)[Neural Networks Made Easy (Part 93): Adaptive Forecasting in Frequency and Time Domains (Final Part)](https://www.mql5.com/en/articles/15024)

In this article, we continue the implementation of the approaches of the ATFNet model, which adaptively combines the results of 2 blocks (frequency and time) within time series forecasting.

![Visualizing deals on a chart (Part 2): Data graphical display](https://c.mql5.com/2/80/Visualization_of_trades_on_a_chart_Part_2_____LOGO.png)[Visualizing deals on a chart (Part 2): Data graphical display](https://www.mql5.com/en/articles/14961)

Here we are going to develop a script from scratch that simplifies unloading print screens of deals for analyzing trading entries. All the necessary information on a single deal is to be conveniently displayed on one chart with the ability to draw different timeframes.

[![](https://www.mql5.com/ff/si/mbxx5fzr169cx07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F498%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dhow.buy.expert%26utm_content%3Dbuy.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=yiuacrhbffqmmulobpsgnypolteeimpt&s=949562ee5e6aca93c0231542844344e241ce4a26ab488f494b70624c190b74d7&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=wxtifnjqllmigxwjrncgxenneavojidy&ssn=1769252559706057507&ssn_dr=0&ssn_sr=0&fv_date=1769252559&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F16377&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Connexus%20Observer%20(Part%208)%3A%20Adding%20a%20Request%20Observer%20-%20MQL5%20Articles&scr_res=1920x1080&ac=17692525591865409&fz_uniq=5083299670127548622&sv=2552)

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