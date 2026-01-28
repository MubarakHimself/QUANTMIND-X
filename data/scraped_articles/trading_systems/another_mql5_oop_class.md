---
title: Another MQL5 OOP Class
url: https://www.mql5.com/en/articles/703
categories: Trading Systems, Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T19:47:16.610988
---

[We've created a channel for MQL5 developersFollow MQL5.community on social media and be the first to receive important updatesLearn more![](https://www.mql5.com/ff/sh/a83xrgctr82w45z9z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=pwgdbvtemvkwsfltqonysypfvtrtufji&s=e99a66a1660cd810b1edbac65597df695e2c2220d1e937834f402f9aeabd4289&uid=&ref=https://www.mql5.com/en/articles/703&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5070560028328400913)

MetaTrader 5 / Examples


### Introduction

Building a complete object-oriented EA that actually works is in my humble opinion a challenging task which requires many skills all put together: logical reasoning, divergent thinking, capacity for analysis and synthesis, imagination, etc. Let's say that if the automated trading system that we have to solve was a game of chess, its trading idea would be the chess strategy. And the execution of the chess strategy through tactics would be the programming of the robot through the use of technical indicators, chart figures, fundamental economic ideas and conceptual axioms.

![Detail of The School of Athens by Raffaello Sanzio](https://c.mql5.com/2/5/plato-aristotle.jpg)

Figure 1. Detail of The School of Athens by Raffaello Sanzio. In this picture we see the philosophers Plato and Aristotle in deep discussion.

Plato represents here the conceptual world and Aristotle the empiricist world.

I am aware of the difficulty of this exercise. An OO EA is not extremely difficult to program, but it is true that presents a certain degree of difficulty to people with little experience in application development. As in any other discipline, this is, in turn, due to the lack of experience itself, that is why I try to teach you this topic through a specific example that I am sure you will understand. Do not get discouraged if you still feel insecure in handling OOP concepts, you will find things much easier once you have implemented your first five EAs, let's say. You don't have to build anything from scratch for now, just understand what I explain here!

The whole process of conceiving and implementing a trading system can be simplified in many senses when it is performed by several persons, though this fact poses the problem of communication. I mean that the person who conceives the investment strategy is not required to handle the programming concepts who does handle their interlocutor, the programmer. And the MQL5 developer may not understand at first some important aspects of their customer's trading strategy.

This is a classical problem in software engineering that has led to the creation of many software development methodologies such as [Scrum](https://en.wikipedia.org/wiki/Scrum_(software_development) "https://en.wikipedia.org/wiki/Scrum_(software_development)"), [Test Driven Development (TDD)](https://en.wikipedia.org/wiki/Test-driven_development "https://en.wikipedia.org/wiki/Test-driven_development"), [Extreme programming (XP)](https://en.wikipedia.org/wiki/Extreme_programming "https://en.wikipedia.org/wiki/Extreme_programming"), etc. It is crucial being aware of the pitfalls of language. By the way, to your knowledge, Wikipedia states: "software development methodology or system development methodology in software engineering is a framework that is used to structure, plan, and control the process of developing an information system".

We are going to assume that we are capable of both quickly conceiving and implementing successful trading ideas. We can also suppose that we are at the end of an iterative development process where the trading system thought by our customer has already been well defined and understood by everybody. As you prefer. By the way, from now on I will refer in this text to some educational articles available in [MQL5 Programming Articles](https://www.mql5.com/en/articles) so that you can quickly check and recall some ideas, when necessary, to successfully carry out this exercise. Are you ready?

### 1\. Your First Steps in Adopting the New Paradigm

**1.1. Why is OOP Good for Programming Your Forex EAs?**

Maybe you are wondering at this point why you have to do something like this. Let me tell you first of all that you are not forced to be OOP. Anyway it is highly recommended to be an OOP person in order to take a step further in your knowledge of programming automated trading systems.

The classical way to develop applications, the so-called procedural programming, has these drawbacks:

- It makes it difficult to model problems. Under this old paradigm the solution of the main problem is reduced to split it into simpler subproblems which are solved by functional modules, that is to say, functions and procedures.
- It makes it difficult to reuse code, which in turn hampers cost, reliability, flexibility and maintenance.

Reusing code is easier with the new object-oriented style. This is very important! Many experts believe that reusing code is the real solution to most problems of software development.

At this point we must mention [abstract data types (ADT)](https://en.wikipedia.org/wiki/Abstract_data_type "https://en.wikipedia.org/wiki/Abstract_data_type"). OOP enables the creation of ADTs. An ADT is an abstraction of the traditional concept of data type which is present in all programming languages. Its main use is to comfortably define the data domain of applications. **Include\\Arrays\\Array.mqh**, **Include\\Arrays\\List.mqh** and **Include\\Arrays\\Tree.mqh** are some examples of MQL5 abstract data types.

In short, the object-oriented programming paradigm wants you to design your applications in a conceptual level in order for you to benefit from code reuse, reliability, flexibility and ease of maintenance.

**1.2. Are You a Conceptual Reasoner? UML Comes to the Rescue**

Have you ever heard about [UML](https://en.wikipedia.org/wiki/Unified_Modeling_Language "https://en.wikipedia.org/wiki/Unified_Modeling_Language")? UML stands for Unified Modeling Language. It is a graphical language for designing object-oriented systems. We humans are supposed to first think our systems in the analysis phase and then code them with a programming language. Going from up to bottom is less insane for developers. Nevertheless, my experience as an analyst says that sometimes this is not possible because of several things: the app must be done in a very short time, there is no one in the team who can quickly apply their knowledge of UML, or perhaps some people on the team do not know some parts of UML.

UML is in my opinion a good analysis tool that you can use if you feel comfortable with it, and if circumstances surrounding the project are ok. Please, read [How to Develop an Expert Advisor using UML Tools](https://www.mql5.com/en/articles/304) if you are interested in exploring the topic of UML. That article is maybe a little overwhelming but it serves to get a big picture about how professional software engineers work their analysis. You would need to complete a course of several weeks to fully understand UML! For now it's okay to know what UML is. According to my experience this analysis tool is not always used in all software projects due to several circumstances present in the real world.

![UML logo](https://c.mql5.com/2/5/UML_logo.gif)

Figure 2. UML logo.

**1.3. Hello World! Your First OO Class**

If you are a complete newbie to object-oriented programming I recommend you first read the official documentation about OO available in [MQL5 Reference](https://www.mql5.com/en/docs) and then take a look at [Writing an Expert Advisor Using the MQL5 Object-Oriented Programming Approach](https://www.mql5.com/en/articles/116) to get the basics. Please be sure to complement these readings with some other materials. From now on I assume that you already know some OOP, so you will easily understand the classic example of the class Person which in MQL5 is as follows:

```
//+------------------------------------------------------------------+
//| CPerson Class                                                    |
//+------------------------------------------------------------------+
class CPerson
  {
protected:
   string            m_first_name;
   string            m_surname;
   datetime          m_birth;

public:
   //--- Constructor and destructor methods
                     CPerson(void);
                    ~CPerson(void);
   //--- Getter methods
   string            GetFirstName(void);
   string            GetSurname(void);
   datetime          GetBirth(void);
   //--- Setter methods
   void              SetFirstName(string first_name);
   void              SetSurname(string surname);
   void              SetBirth(datetime birth);
  };
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CPerson::CPerson(void)
  {
   Alert("Hello world! I am run when an object of type CPerson is created!");
  }
//+------------------------------------------------------------------+
//| Destructor                                                       |
//+------------------------------------------------------------------+
CPerson::~CPerson(void)
  {
   Alert("Goodbye world! I am run when the object is destroyed!");
  }
//+------------------------------------------------------------------+
//| GetFirstName                                                     |
//+------------------------------------------------------------------+
string CPerson::GetFirstName(void)
  {
   return m_first_name;
  }
//+------------------------------------------------------------------+
//| GetSurname                                                       |
//+------------------------------------------------------------------+
string CPerson::GetSurname(void)
  {
   return m_surname;
  }
//+------------------------------------------------------------------+
//| GetBirth                                                         |
//+------------------------------------------------------------------+
datetime CPerson::GetBirth(void)
  {
   return m_birth;
  }
//+------------------------------------------------------------------+
//| SetFirstName                                                     |
//+------------------------------------------------------------------+
void CPerson::SetFirstName(string first_name)
  {
   m_first_name=first_name;
  }
//+------------------------------------------------------------------+
//| SetSurname                                                       |
//+------------------------------------------------------------------+
void CPerson::SetSurname(string surname)
  {
   m_surname=surname;
  }
//+------------------------------------------------------------------+
//| SetBirth                                                         |
//+------------------------------------------------------------------+
void CPerson::SetBirth(datetime birth)
  {
   m_birth=birth;
  }
//+------------------------------------------------------------------+
```

The following is for purist developers obsessed in writing quality code. Unlike some examples available in [MQL5 Programming Articles](https://www.mql5.com/en/articles) and many other software available in [Code Base](https://www.mql5.com/en/code), the above class does use the same programming conventions applied by MetaQuotes Software Corp. for coding their MQL5 framework. I encourage you to write your code just as MetaQuotes does. By the way, the thread entitled [About conventions in OOP MQL5 programs](https://www.mql5.com/en/forum/12728) covers this topic.

In a nutshell, some important conventions applied in writing **Person.mqh** are:

- The class name **CPerson** starts with the letter C capital.
- Method names are camel cased and start with a capital letter, for instance, **GetFirstName**, **SetSurname**, etc.
- Protected properties names are preceded by the prefix **m\_**, for instance, **m\_first\_name**, **m\_surname** and **m\_birth**.
- The reserved word **this** is not used for referencing class members inside the class itself.

Please, have a look at some MQL5 framework files, for example, **Include\\Arrays\\Array.mqh**, **Include\\Arrays\\List.mqh**, **Include\\Trade\\Trade.mqh**, and see how the original MQL5 code is written.

### 2\. Let's Program Our First Object-Oriented EA

**2.1. The Trading System's Idea**

Our trading idea is simple: "Short trends of volatile markets are near random". That's it! This has been observed by several experts under some circumstances. If this hypothesis is true our Forex robot must work by necessity. Given a random point on a chart the next movement can obviously go both up and down, we don't know. The thing is that if the difference between the established SL and TP levels is sufficiently small then that difference is not significant in absolute terms, and we have therefore reached the mathematical expectation. This system is just going to let it work the mathematical expectation. Once you get the EA's code in this article and run the backtest you will see that it requires a very simple money management policy.

**2.2. The Robot's OOP Skeleton**

In this section we are developing the strategy above through the abstract reasoning required by object-oriented programming. So why don't we start thinking of our EA as if it was a living creature? Under this vision our Forex machine can be composed of three main parts: a brain, something that we will call evolution, and a chart.

The brain is the part of the robot that contains the data needed to operate, something like a read-only memory (ROM). The chart is the information piece emulating the graphic on which the robot operates. Finally, the so-called evolution is a piece of data containing temporal information such as the status of the robot at a given moment, the history of the operations performed, etc. It is as if we were designing a human being through their organs, something like a Frankenstein, because we have to develop an app for the health sector. Each organ is in this context a unique semantic concept associated with some other parts of the whole.

First of all let's create the folder **MQL5\\Include\\Mine** to store our custom stuff. This is just an idea to organize your code. It is good to know you can do this in your developments but of course you are not forced to. We will then create the file **MQL5\\Include\\Mine\\Enums.mqh** in order to store the enums created by us:

```
//+------------------------------------------------------------------+
//| Status enumeration                                               |
//+------------------------------------------------------------------+
enum ENUM_STATUS_EA
  {
   BUY,
   SELL,
   DO_NOTHING
  };
//+------------------------------------------------------------------+
//| Lifetime enumeration                                             |
//+------------------------------------------------------------------+
enum ENUM_LIFE_EA
  {
   HOUR,
   DAY,
   WEEK,
   MONTH,
   YEAR
  };
//+------------------------------------------------------------------+
```

Next, it is time to create our EA's embryo which will be named **ExpertSimpleRandom.mq5**! So, please, create the folder **MQL5\\Experts\\SimpleRandom** and then create inside the file **ExpertSimpleRandom.mq5** with this code:

```
//+------------------------------------------------------------------+
//|                                           ExpertSimpleRandom.mq5 |
//|                               Copyright © 2013, Jordi Bassagañas |
//+------------------------------------------------------------------+

#property copyright     "Copyright © 2013, laplacianlab"
#property link          "https://www.mql5.com/en/articles"
#property version       "1.00"

#include <Trade\Trade.mqh>
#include <Trade\SymbolInfo.mqh>
#include <Trade\PositionInfo.mqh>
#include <Indicators\Indicators.mqh>
#include <Mine\Enums.mqh>
#include <..\Experts\SimpleRandom\CSimpleRandom.mqh>

input int               StopLoss;
input int               TakeProfit;
input double            LotSize;
input ENUM_LIFE_EA      TimeLife;

MqlTick tick;
CSimpleRandom *SR=new CSimpleRandom(StopLoss,TakeProfit,LotSize,TimeLife);
//+------------------------------------------------------------------+
//| Initialization function                                          |
//+------------------------------------------------------------------+
int OnInit(void)
  {
   SR.Init();
   return(0);
  }
//+------------------------------------------------------------------+
//| Deinitialization function                                        |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   SR.Deinit();
   delete(SR);
  }
//+------------------------------------------------------------------+
//| OnTick event function                                            |
//+------------------------------------------------------------------+
void OnTick()
  {
   SymbolInfoTick(_Symbol,tick);
   SR.Go(tick.ask,tick.bid);
  }
//+------------------------------------------------------------------+
```

This is only one approach of the many possible out there. All of this is basically to illustrate how OOP works in MQL5. As you see, the Expert Advisor's main class is named **CSimpleRandom.mqh**, please, save it in **MQL5\\Experts\\SimpleRandom\\CSimpleRandom.mqh**:

```
//+------------------------------------------------------------------+
//|                                           ExpertSimpleRandom.mq5 |
//|                               Copyright © 2013, Jordi Bassagañas |
//+------------------------------------------------------------------+
#include <Trade\Trade.mqh>
#include <Mine\Enums.mqh>
#include <..\Experts\SimpleRandom\CBrain.mqh>
#include <..\Experts\SimpleRandom\CEvolution.mqh>
#include <..\Experts\SimpleRandom\CGraphic.mqh>
//+------------------------------------------------------------------+
//| CSimpleRandom Class                                              |
//+------------------------------------------------------------------+
class CSimpleRandom
  {
protected:
   CBrain           *m_brain;
   CEvolution       *m_evolution;
   CGraphic         *m_graphic;
   CTrade           *m_trade;
   CPositionInfo    *m_positionInfo;
public:
   //--- Constructor and destructor methods
                     CSimpleRandom(int stop_loss,int take_profit,double lot_size,ENUM_LIFE_EA time_life);
                    ~CSimpleRandom(void);
   //--- Getter methods
   CBrain           *GetBrain(void);
   CEvolution       *GetEvolution(void);
   CGraphic         *GetGraphic(void);
   CTrade           *GetTrade(void);
   CPositionInfo    *GetPositionInfo(void);
   //--- Specific methods of CSimpleRandom
   bool              Init();
   void              Deinit(void);
   bool              Go(double ask,double bid);
  };
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CSimpleRandom::CSimpleRandom(int stop_loss,int take_profit,double lot_size,ENUM_LIFE_EA time_life)
  {
   int lifeInSeconds;

   switch(time_life)
     {
      case HOUR:

         lifeInSeconds=3600;

         break;

      case DAY:

         lifeInSeconds=86400;

         break;

      case WEEK:

         lifeInSeconds=604800;

         break;

      case MONTH:

         lifeInSeconds=2592000;

         break;

         // One year

      default:

         lifeInSeconds=31536000;

         break;
     }

   m_brain=new CBrain(TimeLocal(),TimeLocal()+lifeInSeconds,lot_size,stop_loss,take_profit);
   m_evolution=new CEvolution(DO_NOTHING);
   m_graphic=new CGraphic(_Symbol);
   m_trade=new CTrade();
  }
//+------------------------------------------------------------------+
//| Destructor                                                       |
//+------------------------------------------------------------------+
CSimpleRandom::~CSimpleRandom(void)
  {
   delete(m_brain);
   delete(m_evolution);
   delete(m_graphic);
   delete(m_trade);
  }
//+------------------------------------------------------------------+
//| GetBrain                                                         |
//+------------------------------------------------------------------+
CBrain *CSimpleRandom::GetBrain(void)
  {
   return m_brain;
  }
//+------------------------------------------------------------------+
//| GetBrain                                                         |
//+------------------------------------------------------------------+
CEvolution *CSimpleRandom::GetEvolution(void)
  {
   return m_evolution;
  }
//+------------------------------------------------------------------+
//| GetGraphic                                                       |
//+------------------------------------------------------------------+
CGraphic *CSimpleRandom::GetGraphic(void)
  {
   return m_graphic;
  }
//+------------------------------------------------------------------+
//| GetTrade                                                         |
//+------------------------------------------------------------------+
CTrade *CSimpleRandom::GetTrade(void)
  {
   return m_trade;
  }
//+------------------------------------------------------------------+
//| GetPositionInfo                                                  |
//+------------------------------------------------------------------+
CPositionInfo *CSimpleRandom::GetPositionInfo(void)
  {
   return m_positionInfo;
  }
//+------------------------------------------------------------------+
//| CSimpleRandom initialization                                     |
//+------------------------------------------------------------------+
bool CSimpleRandom::Init(void)
  {
// Initialization logic here...
   return true;
  }
//+------------------------------------------------------------------+
//| CSimpleRandom deinitialization                                   |
//+------------------------------------------------------------------+
void CSimpleRandom::Deinit(void)
  {
// Deinitialization logic here...
   delete(m_brain);
   delete(m_evolution);
   delete(m_graphic);
   delete(m_trade);
  }
//+------------------------------------------------------------------+
//| CSimpleRandom Go                                                 |
//+------------------------------------------------------------------+
bool CSimpleRandom::Go(double ask,double bid)
  {
   double tp;
   double sl;

   int coin=m_brain.GetRandomNumber(0,1);

// Is there any open position?

   if(!m_positionInfo.Select(_Symbol))
     {
      // If not, we open one

      if(coin==0)
        {
         GetEvolution().SetStatus(BUY);
        }
      else
        {
         GetEvolution().SetStatus(SELL);
        }
     }

// If so, let it work the mathematical expectation.

   else GetEvolution().SetStatus(DO_NOTHING);

   switch(GetEvolution().GetStatus())
     {
      case BUY:

         tp = ask + m_brain.GetTakeProfit() * _Point;
         sl = bid - m_brain.GetStopLoss() * _Point;

         GetTrade().PositionOpen(_Symbol,ORDER_TYPE_BUY,m_brain.GetSize(),ask,sl,tp);

         break;

      case SELL:

         sl = ask + m_brain.GetStopLoss() * _Point;
         tp = bid - m_brain.GetTakeProfit() * _Point;

         GetTrade().PositionOpen(_Symbol,ORDER_TYPE_SELL,m_brain.GetSize(),bid,sl,tp);

         break;

      case DO_NOTHING:

         // Nothing...

         break;
     }

// If there is some error we return false, for now we always return true

   return(true);
  }
//+------------------------------------------------------------------+
```

**2.3. Binding CSimpleRandom to Objects of Complex Type**

Note how the custom objects of type **CBrain**, **CEvolution** and **CGraphic** are linked to **CSimpleRandom**.

First we define the corresponding protected properties:

```
protected:
   CBrain           *m_brain;
   CEvolution       *m_evolution;
   CGraphic         *m_graphic;
```

And right after we instantiate those objects inside the constructor:

```
m_brain=new CBrain(TimeLocal(), TimeLocal() + lifeInSeconds, lot_size, stop_loss, take_profit);
m_evolution=new CEvolution(DO_NOTHING);
m_graphic=new CGraphic(_Symbol);
```

What we do is dynamically create objects of complex type just as the official docs explain in [Object Pointers](https://www.mql5.com/en/docs/basis/types/object_pointers). With this scheme we can access **CBrain**'s, **CEvolution**'s and **CGraphic**'s functionality directly from **CSimpleRandom**. We could run for instance the following code in **ExpertSimpleRandom.mq5**:

```
//+------------------------------------------------------------------+
//| OnTick event function                                            |
//+------------------------------------------------------------------+
void OnTick()
   {
      // ...

      int randNumber=SR.GetBrain().GetRandomNumber(4, 8);

      // ...
  }
```

I write now the code of **CBrain**, **CEvolution** and **CGraphic** to conclude this section. Please, note that there are some parts which are not coded because they are not strictly needed to backtest **SimpleRandom**. It is left as an exercise for you to code the missing parts of these classes, feel free to develop them as you want.  For example, **m\_death** is not actually used, though the idea behind it is to know from the beginning the date in which the robot will finish its activity.

```
//+------------------------------------------------------------------+
//|                                               ExpertSimpleRandom |
//|                               Copyright © 2013, Jordi Bassagaсas |
//+------------------------------------------------------------------+

//+------------------------------------------------------------------+
//| CBrain Class                                                     |
//+------------------------------------------------------------------+
class CBrain
  {
protected:
   ENUM_TIMEFRAMES   m_period;               // period must always be initialized to PERIOD_M1 to fit the system's idea
   datetime          m_birth;               // The datetime in which the robot is initialized for the first time
   datetime          m_death;               // The datetime in which the robot will die
   double            m_size;                // The size of the positions
   int               m_stopLoss;            // Stop loss
   int               m_takeProfit;          // Take profit

public:
   //--- Constructor and destructor methods
                     CBrain(datetime birth,datetime death,double size,int stopLoss,int takeProfit);
                    ~CBrain(void);
   //--- Getter methods
   datetime          GetBirth(void);
   datetime          GetDeath(void);
   double            GetSize(void);
   int               GetStopLoss(void);
   int               GetTakeProfit(void);
   //--- Setter methods
   void              SetBirth(datetime birth);
   void              SetDeath(datetime death);
   void              SetSize(double size);
   void              SetStopLoss(int stopLoss);
   void              SetTakeProfit(int takeProfit);
   //--- Brain specific logic
   int               GetRandomNumber(int a,int b);
  };
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CBrain::CBrain(datetime birth,datetime death,double size,int stopLoss,int takeProfit)
  {
   MathSrand(GetTickCount());

   m_period=PERIOD_M1;
   m_birth=birth;
   m_death=death;
   m_size=size;
   m_stopLoss=stopLoss;
   m_takeProfit=takeProfit;
  }
//+------------------------------------------------------------------+
//| Destructor                                                       |
//+------------------------------------------------------------------+
CBrain::~CBrain(void)
  {
  }
//+------------------------------------------------------------------+
//| GetBirth                                                         |
//+------------------------------------------------------------------+
datetime CBrain::GetBirth(void)
  {
   return m_birth;
  }
//+------------------------------------------------------------------+
//| GetDeath                                                         |
//+------------------------------------------------------------------+
datetime CBrain::GetDeath(void)
  {
   return m_death;
  }
//+------------------------------------------------------------------+
//| GetSize                                                          |
//+------------------------------------------------------------------+
double CBrain::GetSize(void)
  {
   return m_size;
  }
//+------------------------------------------------------------------+
//| GetStopLoss                                                      |
//+------------------------------------------------------------------+
int CBrain::GetStopLoss(void)
  {
   return m_stopLoss;
  }
//+------------------------------------------------------------------+
//| GetTakeProfit                                                    |
//+------------------------------------------------------------------+
int CBrain::GetTakeProfit(void)
  {
   return m_takeProfit;
  }
//+------------------------------------------------------------------+
//| SetBirth                                                         |
//+------------------------------------------------------------------+
void CBrain::SetBirth(datetime birth)
  {
   m_birth=birth;
  }
//+------------------------------------------------------------------+
//| SetDeath                                                         |
//+------------------------------------------------------------------+
void CBrain::SetDeath(datetime death)
  {
   m_death=death;
  }
//+------------------------------------------------------------------+
//| SetSize                                                          |
//+------------------------------------------------------------------+
void CBrain::SetSize(double size)
  {
   m_size=size;
  }
//+------------------------------------------------------------------+
//| SetStopLoss                                                      |
//+------------------------------------------------------------------+
void CBrain::SetStopLoss(int stopLoss)
  {
   m_stopLoss=stopLoss;
  }
//+------------------------------------------------------------------+
//| SetTakeProfit                                                    |
//+------------------------------------------------------------------+
void CBrain::SetTakeProfit(int takeProfit)
  {
   m_takeProfit=takeProfit;
  }
//+------------------------------------------------------------------+
//| GetRandomNumber                                                  |
//+------------------------------------------------------------------+
int CBrain::GetRandomNumber(int a,int b)
  {
   return(a+(MathRand()%(b-a+1)));
  }
//+------------------------------------------------------------------+
```

```
//+------------------------------------------------------------------+
//|                                               ExpertSimpleRandom |
//|                               Copyright © 2013, Jordi Bassagaсas |
//+------------------------------------------------------------------+
#include <Indicators\Indicators.mqh>
#include <Mine\Enums.mqh>
//+------------------------------------------------------------------+
//| CEvolution Class                                                 |
//+------------------------------------------------------------------+
class CEvolution
  {
protected:
   ENUM_STATUS_EA    m_status;            // The current EA's status
   CArrayObj*        m_operations;        // History of the operations performed by the EA

public:
   //--- Constructor and destructor methods
                     CEvolution(ENUM_STATUS_EA status);
                    ~CEvolution(void);
   //--- Getter methods
   ENUM_STATUS_EA    GetStatus(void);
   CArrayObj        *GetOperations(void);
   //--- Setter methods
   void              SetStatus(ENUM_STATUS_EA status);
   void              SetOperation(CObject *operation);
  };
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CEvolution::CEvolution(ENUM_STATUS_EA status)
  {
   m_status=status;
   m_operations=new CArrayObj;
  }
//+------------------------------------------------------------------+
//| Destructor                                                       |
//+------------------------------------------------------------------+
CEvolution::~CEvolution(void)
  {
   delete(m_operations);
  }
//+------------------------------------------------------------------+
//| GetStatus                                                        |
//+------------------------------------------------------------------+
ENUM_STATUS_EA CEvolution::GetStatus(void)
  {
   return m_status;
  }
//+------------------------------------------------------------------+
//| GetOperations                                                    |
//+------------------------------------------------------------------+
CArrayObj *CEvolution::GetOperations(void)
  {
   return m_operations;
  }
//+------------------------------------------------------------------+
//| SetStatus                                                        |
//+------------------------------------------------------------------+
void CEvolution::SetStatus(ENUM_STATUS_EA status)
  {
   m_status=status;
  }
//+------------------------------------------------------------------+
//| SetOperation                                                     |
//+------------------------------------------------------------------+
void CEvolution::SetOperation(CObject *operation)
  {
   m_operations.Add(operation);
  }
//+------------------------------------------------------------------+
```

```
//+------------------------------------------------------------------+
//|                                           ExpertSimpleRandom.mq5 |
//|                               Copyright © 2013, Jordi Bassagaсas |
//+------------------------------------------------------------------+
#include <Trade\SymbolInfo.mqh>
#include <Arrays\ArrayObj.mqh>
//+------------------------------------------------------------------+
//| CGrapic Class                                                    |
//+------------------------------------------------------------------+
class CGraphic
  {
protected:
   ENUM_TIMEFRAMES   m_period;            // Graphic's timeframe
   string            m_pair;              // Graphic's pair
   CSymbolInfo*      m_symbol;            // CSymbolInfo object
   CArrayObj*        m_bars;              // Array of bars

public:
   //--- Constructor and destructor methods
                     CGraphic(string pair);
                    ~CGraphic(void);
   //--- Getter methods
   string            GetPair(void);
   CSymbolInfo      *GetSymbol(void);
   CArrayObj        *GetBars(void);
   //--- Setter methods
   void              SetPair(string pair);
   void              SetSymbol(CSymbolInfo *symbol);
   void              SetBar(CObject *bar);
  };
//+------------------------------------------------------------------+
//| Constuctor                                                       |
//+------------------------------------------------------------------+
CGraphic::CGraphic(string pair)
  {
   m_period=PERIOD_M1;
   m_pair=pair;
  }
//+------------------------------------------------------------------+
//| Destructor                                                       |
//+------------------------------------------------------------------+
CGraphic::~CGraphic(void)
  {
  }
//+------------------------------------------------------------------+
//| GetPair                                                          |
//+------------------------------------------------------------------+
string CGraphic::GetPair(void)
  {
   return m_pair;
  }
//+------------------------------------------------------------------+
//| GetSymbol                                                        |
//+------------------------------------------------------------------+
CSymbolInfo *CGraphic::GetSymbol(void)
  {
   return m_symbol;
  }
//+------------------------------------------------------------------+
//| GetBars                                                          |
//+------------------------------------------------------------------+
CArrayObj *CGraphic::GetBars(void)
  {
   return m_bars;
  }
//+------------------------------------------------------------------+
//| SetPair                                                          |
//+------------------------------------------------------------------+
void CGraphic::SetPair(string pair)
  {
   m_pair=pair;
  }
//+------------------------------------------------------------------+
//| SetSymbol                                                        |
//+------------------------------------------------------------------+
void CGraphic::SetSymbol(CSymbolInfo *symbol)
  {
   m_symbol=symbol;
  }
//+------------------------------------------------------------------+
//| SetBar                                                           |
//+------------------------------------------------------------------+
void CGraphic::SetBar(CObject *bar)
  {
   m_bars.Add(bar);
  }
//+------------------------------------------------------------------+
```

### 3\. Backtesting ExpertSimpleRandom.mq5

This random trading system has demonstrated to be valid only for certain stop loss and take profit levels, as expected. Of course, these winner SL/TP intervals are not the same for all symbols. This is because every symbol shows its own personality at a given time, or put another way, all currency pairs move differently with respect to the rest. So please identify first these levels in backtesting before running **ExpertSimpleRandom.mq5** in a real environment.

I share here some sample data for which the idea exposed in this article seems to be winner. This can be deduced after running many times **ExpertSimpleRandom.mq5** in MetaTrader 5 Strategy Tester.

Some winner inputs for EURUSD, January 2012, are:

- StopLoss: 400
- TakeProfit: 600
- LotSize: 0.01
- TimeLife: MONTH

Run number 1:

![EURUSD, January 2012](https://c.mql5.com/2/6/1.png)

Run number 2:

![EURUSD, January 2012](https://c.mql5.com/2/6/2.png)

Run number 3:

![EURUSD, January 2012](https://c.mql5.com/2/6/3.png)

### Conclusion

We have learned to apply object-oriented programming in our automated trading systems. In order to do this we have first had to define a mechanical trading strategy. Our trading idea has been very simple: "Short trends of volatile markets are near random". This has been observed by several experts under some circumstances.

Right after we have thought of our EA in real-world terms as if it was a living creature. Thanks to this vision we have seen that our Forex machine can be composed of three main parts: a brain, something that we have called evolution, and a chart.

And finally we have programmed the system's expert advisor which incorporates the logic needed to run the backtest, have run many times the robot in January 2012, and we have found that most of the times the system is winning. The idea behind this system has been proved true but its effectiveness is not very high because of its simplicity.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/703.zip "Download all attachments in the single ZIP archive")

[cbrain.mqh](https://www.mql5.com/en/articles/download/703/cbrain.mqh "Download cbrain.mqh")(5.64 KB)

[cevolution.mqh](https://www.mql5.com/en/articles/download/703/cevolution.mqh "Download cevolution.mqh")(2.97 KB)

[cgraphic.mqh](https://www.mql5.com/en/articles/download/703/cgraphic.mqh "Download cgraphic.mqh")(3.55 KB)

[enums.mqh](https://www.mql5.com/en/articles/download/703/enums.mqh "Download enums.mqh")(0.92 KB)

[expertsimplerandom.mq5](https://www.mql5.com/en/articles/download/703/expertsimplerandom.mq5 "Download expertsimplerandom.mq5")(1.79 KB)

[csimplerandom.mqh](https://www.mql5.com/en/articles/download/703/csimplerandom.mqh "Download csimplerandom.mqh")(12.79 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Building a Social Technology Startup, Part II: Programming an MQL5 REST Client](https://www.mql5.com/en/articles/1044)
- [Building a Social Technology Startup, Part I: Tweet Your MetaTrader 5 Signals](https://www.mql5.com/en/articles/925)
- [Raise Your Linear Trading Systems to the Power](https://www.mql5.com/en/articles/734)
- [Marvel Your MQL5 Customers with a Usable Cocktail of Technologies!](https://www.mql5.com/en/articles/728)
- [Extending MQL5 Standard Library and Reusing Code](https://www.mql5.com/en/articles/741)
- [Building an Automatic News Trader](https://www.mql5.com/en/articles/719)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/13081)**
(7)


![Jose Ignacio Martin Somokurzio](https://c.mql5.com/avatar/2016/2/56CDA1C8-9462.jpg)

**[Jose Ignacio Martin Somokurzio](https://www.mql5.com/en/users/jimsb)**
\|
24 Feb 2016 at 12:21

Agree with you....great job.Thnx.


![Evgeniy Chernish](https://c.mql5.com/avatar/2024/3/65eac9b5-9233.png)

**[Evgeniy Chernish](https://www.mql5.com/en/users/vp999369)**
\|
2 Oct 2023 at 16:07

Afternoon!

In the **CSimpleRandom.mqh** file, the author omitted the creation of an object of **[CPositionInfo](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/cpositioninfo "Standard library: Class CPositionInfo")** class

```
82 строчка  m_positionInfo=new CPositionInfo();
```

Without it, the Expert Advisor generated the error **invalid pointer access in 'CSimpleRandom.mqh'**.

Or did he omit it on purpose so that beginners could practice? If yes, he succeeded =)

![Artyom Trishkin](https://c.mql5.com/avatar/2022/7/62C4775C-ABD6.jpg)

**[Artyom Trishkin](https://www.mql5.com/en/users/artmedia70)**
\|
3 Oct 2023 at 14:15

**vp999369 CPositionInfo** class object

Without it, the Expert Advisor generated the error **invalid pointer access in 'CSimpleRandom.mqh'**

Or did he omit it on purpose so that beginners could practice? If yes, he succeeded =)

In the CSimplrRandom class file you suggest to create an object of the same class? That's not correct.

See the file \\MQL5\\Experts\\Expertsimplerandom.mq5. It contains the creation of an object of this class. Line 23:

```
//+------------------------------------------------------------------+
//|ExpertSimpleRandom.mq5 |
//|Copyright © 2013, Jordi Bassagaças |
//+------------------------------------------------------------------+

#property copyright     "Copyright © 2013, laplacianlab"
#property link          "http://www.mql5.com/en/articles"
#property version       "1.00"

#include <Trade\Trade.mqh>
#include <Trade\SymbolInfo.mqh>
#include <Trade\PositionInfo.mqh>
#include <Indicators\Indicators.mqh>
#include <Mine\Enums.mqh>
#include <..\Experts\SimpleRandom\CSimpleRandom.mqh>

input int               StopLoss;
input int               TakeProfit;
input double            LotSize;
input ENUM_LIFE_EA      TimeLife;

MqlTick tick;
CSimpleRandom *SR=new CSimpleRandom(StopLoss,TakeProfit,LotSize,TimeLife);
//+------------------------------------------------------------------+
//| Initialisation function|
//+------------------------------------------------------------------+
```

Compile and run the Expert Advisor \\MQL5\\Experts\\Expertsimplerandom.mq5.

![Artyom Trishkin](https://c.mql5.com/avatar/2022/7/62C4775C-ABD6.jpg)

**[Artyom Trishkin](https://www.mql5.com/en/users/artmedia70)**
\|
3 Oct 2023 at 14:43

Ah, no, I didn't understand you. Indeed, you need to add the [creation of](https://www.mql5.com/en/docs/basis/operators/newoperator "MQL5 Documentation: Object creation operator new") the class [object](https://www.mql5.com/en/docs/basis/operators/newoperator "MQL5 Documentation: Object creation operator new").


![Artyom Trishkin](https://c.mql5.com/avatar/2022/7/62C4775C-ABD6.jpg)

**[Artyom Trishkin](https://www.mql5.com/en/users/artmedia70)**
\|
3 Oct 2023 at 15:17

**vp999369 CPositionInfo** class object

Without it, the Expert Advisor generated the error **invalid pointer access in 'CSimpleRandom.mqh'**

Or did he omit it on purpose so that beginners could practice? If yes, he succeeded =)

The article files have been reuploaded. Thanks for the message.

![Money Management Revisited](https://c.mql5.com/2/17/801_12.gif)[Money Management Revisited](https://www.mql5.com/en/articles/1367)

The article deals with some issues arising when traders apply various money management systems to Forex trading. Experimental data obtained from performing trading deals using different money management (MM) methods is also described.

![Alert and Comment for External Indicators (Part Two)](https://c.mql5.com/2/17/825_12.gif)[Alert and Comment for External Indicators (Part Two)](https://www.mql5.com/en/articles/1372)

Since I published the article "Alert and Comment for External Indicators", I have been receiving requests and questions regarding the possibility of developing an external informer operating based on indicator lines. Having analyzed the questions, I have decided to continue with the subject. Getting data stored in indicator buffers turned out to be another area of interest to users.

![MQL5 Cookbook: Writing the History of Deals to a File and Creating Balance Charts for Each Symbol in Excel](https://c.mql5.com/2/0/avatar11.png)[MQL5 Cookbook: Writing the History of Deals to a File and Creating Balance Charts for Each Symbol in Excel](https://www.mql5.com/en/articles/651)

When communicating in various forums, I often used examples of my test results displayed as screenshots of Microsoft Excel charts. I have many times been asked to explain how such charts can be created. Finally, I now have some time to explain it all in this article.

![Alert and Comment for External Indicators. Multi-Currency Analysis Through External Scanning](https://c.mql5.com/2/17/832_8.png)[Alert and Comment for External Indicators. Multi-Currency Analysis Through External Scanning](https://www.mql5.com/en/articles/1371)

Alert for multi-currency and multiple time frame analysis of external indicators. The article deals with a method of getting event information in respect of events in external indicators, without having to attach indicators to a chart or open charts themselves. We will call it external scanning.

[Launching MetaTrader VPS for the first time?Read our comprehensive, step-by-step instructions![](https://www.mql5.com/ff/sh/0xb0c8bjq5sadh89z2/01.png)Read](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/articles/13586&a=gxygkojxdwrcfbbgfrchvjgelflsnelu&s=49eab2fb45d89f59a191e88145774dcd7f9533039acb10dd9c28061b04fa92fe&uid=&ref=https://www.mql5.com/en/articles/703&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5070560028328400913)

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