---
title: Design Patterns in software development and MQL5 (Part 4): Behavioral Patterns 2
url: https://www.mql5.com/en/articles/13876
categories: Trading, Trading Systems, Indicators, Expert Advisors
relevance_score: 4
scraped_at: 2026-01-23T17:37:40.135896
---

[![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/01.png)![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/02.png)Powerful analytics for traders of any levelAll the necessary trading reports for beginners and professionals](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/454106&a=muccpajyfystoakuukdobwigjejzmpqn&s=52daad60fa795e635264e6f94898f05493bca3b5124d4cca8eb7e82333c2ef12&uid=&ref=https://www.mql5.com/en/articles/13876&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5068460571004697010)

MetaTrader 5 / Trading


### Introduction

In this article, we will complete the behavioral design patterns to complete the topic of the design patterns in software and how we can use them in the MQL5. We have identified in the previous articles all creational patterns through the article of [Design Patterns in Software Development and MQL5 (Part I): Creational Patterns](https://www.mql5.com/en/articles/13622) and we have identified all structural patterns through the article of [Design Patterns in Software Development and MQL5 (Part 2): Structural Patterns](https://www.mql5.com/en/articles/13724). Then, we identified in the previous article of [Design Patterns in Software Development and MQL5 (Part 3): Behavioral Patterns](https://www.mql5.com/en/articles/13796) 1 some of the behavioral patterns and they were the same as the following: Chain of responsibility, Command, Interpreter, Iterator, and Mediator. We also identified what are design patterns, which are patterns that can be used to define and manage the method of communication between objects.

In this article, we will complete what remains in the behavioral patterns through the following topics:

- [Memento](https://www.mql5.com/en/articles/13876#memento): It can be used to restore an object to a stored state by capturing and externalizing the object's internal state without violating encapsulation.
- [Observer](https://www.mql5.com/en/articles/13876#observer): It can be used to define a one-to-many dependency between objects, so that when one object changes its state, all its dependencies receive a notification and are automatically updated.
- [State](https://www.mql5.com/en/articles/13876#state): It can be used to make an object change its behavior when its internal state changes. The object appears to change its class.
- [Strategy](https://www.mql5.com/en/articles/13876#strategy): it can be used to identify a family of algorithms, encapsulate them, and make them interchangeable. The strategy allows the algorithm to vary independently of the clients that use it.
- [Template Method](https://www.mql5.com/en/articles/13876#template): it can be used to identify the basics of an algorithm in an operation, leaving some steps to subclasses by allowing subclasses to re-identify them without changing the structure of the algorithms.
- [Visitor](https://www.mql5.com/en/articles/13876#visitor): It can be used to identify a new operation without having any effect on the classes of elements on which the operation is performed.
- [Conclusion](https://www.mql5.com/en/articles/13876#conclusion)

We identified before that behavioral patterns are the patterns of behavior that are concerned with the assignment and determination of responsibilities between objects. They also identify how objects can communicate or interact with each other. They characterize a complex flow of control that's hard to follow at run time. They allow you to focus only on the way objects are connected, shifting your focus away from the control flow.

If you have read my previous articles in this series, you will be familiar with our approach to presenting each pattern through the following points:

- What does the pattern do?
- What does the pattern solve?
- How can we use it in MQL5?

It is important to mention that if you understand the OOP (object Oriented Programing) topic this will help to well understand design patterns topic and if you want to read or learn some of this topic you can also read my previous [Understanding MQL5 Object-Oriented Programming (OOP)](https://www.mql5.com/en/articles/12813) article. I hope that you find my articles are useful in your journey to improve your programming skills by identifying one of the most important topics which is design patterns to write clean, extendable, reusable, and well-tested code.

Disclaimer: All information provided 'as is' only for educational purposes and is not prepared for trading purposes or advice. The information does not guarantee any kind of result. If you choose to use these materials on any of your trading accounts, you will do that at your own risk and you will be the only responsible.

### Memento

In this section, we are going to identify the Memento pattern as a behavioral design pattern. The Memento pattern can be used to externalize the state of an object to provide rollback functionality, and it is also known as a token.

**What does the pattern do?**

We can use the Memento pattern when we need to store a snapshot of the state of the object to be restored at a later time, and when a direct interface to get the state would expose the details of the execution and break the encapsulation of the object. So this pattern will capture and externalize the object state to be restored later, the following is a diagram of the structure of this pattern that shows how it can work:

![Memento](https://c.mql5.com/2/63/Memento.png)

As we can see in the previous graph we have the following as participants:

- **Memento:** it stores the state as necessary at the discretion of the Originator object which stores its state. It grants no access to objects except the originator. It may store as much or as little of the internal state of the originator as is necessary at the discretion of the originator. it has two interfaces narrow or wide based on the point of view of the caretaker or originator.
- **Originator:** it creates the memento that contains the current internal state snapshot and restores the internal state by using the memento.
- **Caretaker**: it is responsible for saving the memento without examining the memento's content.

There are some pitfalls in the use of this type of pattern and they are the same as those listed below:

- It can be expensive if there is a large copy.
- There is a loss of history as there are a limited number of slots for snapshots.
- It does not expose any information other than the memento.

**What does design pattern solve?**

By using this pattern, the following problems can be solved:

- It preserves the encapsulation boundaries.
- It identifies narrow and wide interfaces.
- It can be used to simplify the originator.

**How can we use it in MQL5?**

In this part, we are going to try to use the Memento pattern in MQL5 and it can be done by going through the following steps:

Declare the Memento class by using the class keyword

```
class Memento
  {
protected:
   string            m_state;
public:
   string            GetState(void);
   void              SetState(string);
                     Memento(string);
  };
Memento::Memento(string state):
   m_state(state)
  {
  }
string Memento::GetState(void)
  {
   return m_state;
  }
void Memento::SetState(string state)
  {
   m_state=state;
  }
```

Declare the Originator class

```
class Originator
  {
protected:
   string            m_state;
public:
   void              SetMemento(Memento& memento);
   Memento*          CreateMemento(void);
   string            State(void);
   void              State(string);
  };
void Originator::SetMemento(Memento& memento)
  {
   m_state=memento.GetState();
  }
Memento* Originator::CreateMemento(void)
  {
   return new Memento(m_state);
  }
void Originator::State(string state)
  {
   m_state=state;
  }
string Originator::State(void)
  {
   return m_state;
  }
```

Declare the Caretaker class

```
class Caretaker
  {
public:
   Memento*          memento;
                    ~Caretaker(void);
  };
Caretaker::~Caretaker(void)
  {
   if(CheckPointer(memento)==1)
     {
      delete memento;
     }
  }
```

So the following is the complete code in one block for the use of the Memento pattern in MQL5:

```
//+------------------------------------------------------------------+
//|                                                      Memento.mqh |
//+------------------------------------------------------------------+
class Memento
  {
protected:
   string            m_state;
public:
   string            GetState(void);
   void              SetState(string);
                     Memento(string);
  };
Memento::Memento(string state):
   m_state(state)
  {
  }
string Memento::GetState(void)
  {
   return m_state;
  }
void Memento::SetState(string state)
  {
   m_state=state;
  }
class Originator
  {
protected:
   string            m_state;
public:
   void              SetMemento(Memento& memento);
   Memento*          CreateMemento(void);
   string            State(void);
   void              State(string);
  };
void Originator::SetMemento(Memento& memento)
  {
   m_state=memento.GetState();
  }
Memento* Originator::CreateMemento(void)
  {
   return new Memento(m_state);
  }
void Originator::State(string state)
  {
   m_state=state;
  }
string Originator::State(void)
  {
   return m_state;
  }
class Caretaker
  {
public:
   Memento*          memento;
                    ~Caretaker(void);
  };
Caretaker::~Caretaker(void)
  {
   if(CheckPointer(memento)==1)
     {
      delete memento;
     }
  }
```

### Observer

In this part, we will identify another behavioral pattern which is the Observer pattern. It defines a one-to-many dependency between objects so that when one object changes its state, all its dependencies will be notified and will be automatically updated. It is also known as Dependents and Publish-Subscribe.

**What does the pattern do?**

We can use the Observer pattern when we need to encapsulate aspects of abstraction in separate objects to be able to vary and reuse them independently, when there is a need to change other objects when changing one and we do not know the number of objects needed to be changed, and when we need the object to send a notification to other objects without coupling between these objects.

Based on the foregoing, this pattern can be represented in the following diagram, to have a better understanding of how this pattern works:

![Observer](https://c.mql5.com/2/63/Observer.png)

As we can see in the previous graph we can say that we have participants in this pattern the same as the following:

- **Subject:** The subject identifies its observers and how many observers the subject may be observed by these observers. For attaching and detaching objects of the observer, the subject provides the interface for that.
- **Observer:** it identifies the updated interface for objects that need to be notified of changes in the subject.
- **ConcreteSubject:** it saves or stores the state to ConcreteObserver objects and when there is a change in the state, it sends a notification to the observers with that.
- **ConcreteObserver:** this will play the role of doing the following three objectives:

  - Maintaining the reference to the object of the ConcreteSubject.
  - Saving the state.
  - Executing the interface of the observer updating.

There are pitfalls when using the observer pattern and they are the same as the following:

- The observable does not identify which object updated its state.
- Large updates.
- It can be difficult to debug.

**What does design pattern solve?**

According to what we understand in the way that the Observer pattern works we can say that it will be able to solve or there are the following benefits when using it:

- It allows the abstract coupling between the subject and the observer.
- It provides support for the broadcast interaction or communication.

**How can we use it in MQL5?**

Now, it is time to see a method to use the Observer pattern in MQL5 and it will be the same as the following:

Using the interface keyword to declare the Observer area to determine the classes, and functions that we need

```
interface Observer
  {
   void Update(string state);
  };
```

Using the class keyword to declare the Subject

```
class Subject
  {
public:
                     Subject(void);
                    ~Subject(void);
   void              Attach(Observer* observer);
   void              Detach(Observer& observer);
   void              Notify(void);
   void              State(string state);
   string            State(void) {return m_state;}

protected:
   string            m_state;
   Observer*         m_observers[];

   int               Find(Observer& observer);
  };
Subject::Subject(void):
   m_state(NULL)
  {
  }
Subject::~Subject(void)
  {
   int itotal=ArraySize(m_observers);
   for(int i=0; i<itotal; i++)
     {
      Observer* item=m_observers[i];
      if(CheckPointer(item)==1)
        {
         delete item;
        }
     }
  }
void Subject::State(string state)
  {
   m_state=state;
  }
void Subject::Notify(void)
  {
   int itotal=ArraySize(m_observers);
   for(int i=0; i<itotal; i++)
     {
      m_observers[i].Update(m_state);
     }
  }
void Subject::Attach(Observer *observer)
  {
   int size=ArraySize(m_observers);
   ArrayResize(m_observers,size+1);
   m_observers[size]=observer;
  }
void Subject::Detach(Observer &observer)
  {
   int find=Find(observer);
   if(find==-1)
      return;
   Observer* item=m_observers[find];
   if(CheckPointer(item)==1)
      delete item;
   ArrayRemove(m_observers,find,1);
  }
int Subject::Find(Observer &observer)
  {
   int itotal=ArraySize(m_observers);
   for(int i=0; i<itotal; i++)
     {
      Observer* item=m_observers[i];
      if(item==&observer)
         return i;
     }
   return -1;
  }
```

Using the class keyword to declare the ConcreteSubject

```
class ConcreteSubject:public Subject
  {
  public:
   void              State(string state);
   string            State(void) {return m_state;}
  };
void ConcreteSubject::State(string state)
  {
   m_state=state;
  }
```

Using the class keyword to declare the ConcreteObserver

```
class ConcreteObserver:public Observer
  {
public:
   void              Update(string state);
                     ConcreteObserver(ConcreteSubject& subject);
protected:
   string            m_observer_state;
   ConcreteSubject*  m_subject;
  };
ConcreteObserver::ConcreteObserver(ConcreteSubject& subject):
   m_subject(&subject)
  {
  }
void ConcreteObserver::Update(string state)
  {
   m_observer_state=state;
  }
```

So, the following is the full code for using the Observer behavioral design pattern in MQL5:

```
//+------------------------------------------------------------------+
//|                                                     Observer.mqh |
//+------------------------------------------------------------------+
interface Observer
  {
   void Update(string state);
  };
class Subject
  {
public:
                     Subject(void);
                    ~Subject(void);
   void              Attach(Observer* observer);
   void              Detach(Observer& observer);
   void              Notify(void);
   void              State(string state);
   string            State(void) {return m_state;}

protected:
   string            m_state;
   Observer*         m_observers[];

   int               Find(Observer& observer);
  };
Subject::Subject(void):
   m_state(NULL)
  {
  }
Subject::~Subject(void)
  {
   int itotal=ArraySize(m_observers);
   for(int i=0; i<itotal; i++)
     {
      Observer* item=m_observers[i];
      if(CheckPointer(item)==1)
        {
         delete item;
        }
     }
  }
void Subject::State(string state)
  {
   m_state=state;
  }
void Subject::Notify(void)
  {
   int itotal=ArraySize(m_observers);
   for(int i=0; i<itotal; i++)
     {
      m_observers[i].Update(m_state);
     }
  }
void Subject::Attach(Observer *observer)
  {
   int size=ArraySize(m_observers);
   ArrayResize(m_observers,size+1);
   m_observers[size]=observer;
  }
void Subject::Detach(Observer &observer)
  {
   int find=Find(observer);
   if(find==-1)
      return;
   Observer* item=m_observers[find];
   if(CheckPointer(item)==1)
      delete item;
   ArrayRemove(m_observers,find,1);
  }
int Subject::Find(Observer &observer)
  {
   int itotal=ArraySize(m_observers);
   for(int i=0; i<itotal; i++)
     {
      Observer* item=m_observers[i];
      if(item==&observer)
         return i;
     }
   return -1;
  }
class ConcreteSubject:public Subject
  {
  public:
   void              State(string state);
   string            State(void) {return m_state;}
  };
void ConcreteSubject::State(string state)
  {
   m_state=state;
  }
class ConcreteObserver:public Observer
  {
public:
   void              Update(string state);
                     ConcreteObserver(ConcreteSubject& subject);
protected:
   string            m_observer_state;
   ConcreteSubject*  m_subject;
  };
ConcreteObserver::ConcreteObserver(ConcreteSubject& subject):
   m_subject(&subject)
  {
  }
void ConcreteObserver::Update(string state)
  {
   m_observer_state=state;
  }
```

### State

In this part, we will identify the State pattern which allows the object to change its behavior in case of changing in its internal state and the object will appear to change its class. It is also know as Objects for States. We can use it when the behavior of the object depends on its state by changing the behavior of the object based on the state at run-time. It can be used also when we have operations with large conditional statements and all of that depends on the state of the object.

**What does the pattern do?**

If we need to understand how the State behavioral pattern works, we can see through the following graph:

![State](https://c.mql5.com/2/63/State.png)

As we can see in the previous graph we have the following participants:

**Context:** it identifies the interface of the interest to the client. It also maintains the subclass of the instance of the ConcreteState that identifies the current state.

**State:** it identifies the interface to encapsulate the behavior of a specific state of the context.

**ConcreteState subclasses:** the subclass executes the behavior of the state of the context and this is for every subclass.

According to what we said about the State pattern, it can be used when we have an object that behaves differently depending on its current state. But there is a pitfalls when using this pattern that is we will have more classes which means more code.

**What does design pattern solve?**

Although there are the same pitfalls that we have presented when using the State pattern, there are advantages that can be considered as problems that can be solved when using it. These advantages are the same as the following:

- It helps to localize the behavior that is specific to each state and separates the behavior based on the characteristics of each state.
- It explicitly defines the transitions between the different states.
- It helps to share state objects.

**How can we use it in MQL5?**

The following steps are for a method about how we can use the State pattern in MQL5:

Declare the Context class

```
class Context;
```

Declare the State interface

```
interface State
  {
   void Handle(Context& context);
  };
```

Declare the m\_state object

```
State*            m_state;
```

Declare the Context class

```
class Context
  {
public:
                     Context(State& state);
                    ~Context(void);
   State*            State(void) {return m_state;}
   void              State(State& state);
   void              Request(void);
  };
Context::~Context(void)
  {
   if(CheckPointer(m_state)==1)
      delete m_state;
  }
void Context::State(State& state)
  {
   delete m_state;
   m_state=&state;
  }
void Context::Request(void)
  {
   m_state.Handle(this);
  }
```

Declare the ConcreteStateA class

```
class ConcreteStateA:public State
  {
public:
   void              Handle(Context& context);
  };
void ConcreteStateA::Handle(Context& context)
  {
   context.State(new ConcreteStateB);
  }
```

Declare the ConcreteStateB class

```
class ConcreteStateB:public State
  {
public:
   void              Handle(Context& context);
  };
void ConcreteStateB::Handle(Context& context)
  {
   context.State(new ConcreteStateA);
  }
```

So, the following is the one block of code to use the State design pattern in MQL5:

```
//+------------------------------------------------------------------+
//|                                                        State.mqh |
//+------------------------------------------------------------------+
class Context;
interface State
  {
   void Handle(Context& context);
  };
State*            m_state;
class Context
  {
public:
                     Context(State& state);
                    ~Context(void);
   State*            State(void) {return m_state;}
   void              State(State& state);
   void              Request(void);
  };
Context::~Context(void)
  {
   if(CheckPointer(m_state)==1)
      delete m_state;
  }
void Context::State(State& state)
  {
   delete m_state;
   m_state=&state;
  }
void Context::Request(void)
  {
   m_state.Handle(this);
  }

class ConcreteStateA:public State
  {
public:
   void              Handle(Context& context);
  };
void ConcreteStateA::Handle(Context& context)
  {
   context.State(new ConcreteStateB);
  }

class ConcreteStateB:public State
  {
public:
   void              Handle(Context& context);
  };
void ConcreteStateB::Handle(Context& context)
  {
   context.State(new ConcreteStateA);
  }
```

### Strategy

Here is another behavioral pattern which is the Strategy pattern. It identifies a group of algorithms, encapsulates them, and makes them interchangeable. This allows the algorithm to vary independently of the clients that use it. So, we can use it when we need to enable the algorithm to be selected at run-time, and when we need to eliminate conditional statements. It is also known as the Policy.

So, we can say that we can use the Strategy pattern when:

- Many related classes are only different in the way they behave. Strategies provide a method to configure to configure a class to behave in one of many ways.
- We need to use different variants of the same algorithm. and we need the ability to select the algorithm at run-time.
- An algorithm is a use of data that clients shouldn't have access to.
- A class defines many behaviors, which appear as multiple conditional statements in its operations. Rather than having many related conditional branches, it can be in a separate strategy class.
- Conditional statements elimination.

**What does the pattern do?**

If we want to understand how this Strategy pattern works, we can do that through the following graph:

![Strategy](https://c.mql5.com/2/63/Strategy.png)

As we can see in the previous graph the Strategy pattern has the following as participants:

- **Strategy:** it declares the interface common to all supported algorithms to be used by the context to be able to call the algorithm identified by the ConcreteStrategy.
- **ConcreteStrategy:** by using the interface of the Strategy, the context implements the algorithm.
- **Context:** it is constructed by the ConcreteStrategy object, it maintains the reference to the strategy object, and for the purpose of letting the Strategy access its data it can identify the interface for that.

Although benefits can be obtained when using this strategy pattern, there are pitfalls and they are the same as the following:

- The client has to know about strategies.
- There is an increased number of classes.

**What does design pattern solve?**

The following are the consequences or benefits of using the Strategy pattern:

- It helps to apply reusability because it helps to identify groups of related algorithms that can be reused by the context.
- It can be considered as another way to support a variety of behaviors instead of subclassing.
- It helps as a strategy to eliminate conditional statements.
- It allows one to choose between different implementations of the same behavior.

**How can we use it in MQL5?**

Now we will identify a way to use the Strategy pattern, which will be the same as the following steps:

Declare the Strategy interface to declare classes, and functions within:

```
interface Strategy
  {
   void AlgorithmInterface(void);
  };
```

Declare the Context class

```
class Context
  {
public:
                     Context(Strategy& strategy);
                    ~Context(void);

   void              ContextInterface(void);
protected:
   Strategy*         m_strategy;
  };
Context::Context(Strategy& strategy)
  {
   m_strategy=&strategy;
  }
Context::~Context(void)
  {
   if(CheckPointer(m_strategy)==1)
      delete m_strategy;
  }
void Context::ContextInterface(void)
  {
   m_strategy.AlgorithmInterface();
  }
```

Declare the ConcreteStrategyA class

```
class ConcreteStrategyA : public Strategy
  {
public:
   void              AlgorithmInterface(void);
  };
void ConcreteStrategyA::AlgorithmInterface(void)
  {
  }
```

Declare the ConcreteStrategyB class

```
class ConcreteStrategyB : public Strategy
  {
public:
   void              AlgorithmInterface(void);
  };
void ConcreteStrategyB::AlgorithmInterface(void)
  {
  }
```

Declare the ConcreteStrategyC class

```
class ConcreteStrategyC : public Strategy
  {
public:
   void              AlgorithmInterface(void);
  };
void ConcreteStrategyC::AlgorithmInterface(void)
  {
  }
```

So, the following code is the full code in one block to use the Strategy pattern in MQL5:

```
//+------------------------------------------------------------------+
//|                                                     Strategy.mqh |
//+------------------------------------------------------------------+
interface Strategy
  {
   void AlgorithmInterface(void);
  };
class Context
  {
public:
                     Context(Strategy& strategy);
                    ~Context(void);

   void              ContextInterface(void);
protected:
   Strategy*         m_strategy;
  };
Context::Context(Strategy& strategy)
  {
   m_strategy=&strategy;
  }
Context::~Context(void)
  {
   if(CheckPointer(m_strategy)==1)
      delete m_strategy;
  }
void Context::ContextInterface(void)
  {
   m_strategy.AlgorithmInterface();
  }
class ConcreteStrategyA : public Strategy
  {
public:
   void              AlgorithmInterface(void);
  };
void ConcreteStrategyA::AlgorithmInterface(void)
  {
  }
class ConcreteStrategyB : public Strategy
  {
public:
   void              AlgorithmInterface(void);
  };
void ConcreteStrategyB::AlgorithmInterface(void)
  {
  }
class ConcreteStrategyC : public Strategy
  {
public:
   void              AlgorithmInterface(void);
  };
void ConcreteStrategyC::AlgorithmInterface(void)
  {
  }
```

### Template Method

The Template Method is another behavioral design pattern that can be useful in software development. It can be used to recognize the basic components of an algorithm within an operation and to delegate certain steps to subclasses. This allows subclasses to redefine these steps without changing the overall structure of the algorithm.

So, we can say that we can use it when:

- We have an algorithm and we need to change some steps from it only without affecting the overall structure of this algorithm.
- We need to implement the invariant aspects of an algorithm only once and delegate responsibility for the implementation of variable behavior to subclasses.
- We need to void code duplication when the behavior that is common to the subclasses should be factored in and localized in a common class.
- We need to control subclass extensions.

**What does the pattern do?**

If we need to understand how this pattern works, we can do this with the help of the following chart, which is a representation of the pattern:

![ Template Method](https://c.mql5.com/2/63/Template_Method.png)

As we can see in the previous graph we have the following participants:

- **AbstractClass:** it can be used to specify abstract primitive operations that must be defined by concrete subclasses to implement the different steps of an algorithm. It can be used also to execute a template method that outlines the framework of an algorithm. This template method calls both primitive operations and operations specified in AbstractClass or belonging to other objects.
- **ConcreteClass:** It can be used to perform the primitive operations required to carry out the steps of the algorithm specific to the subclass.

**What does design pattern solve?**

There are benefits and issues that can be solved when using the template method behavioral pattern and these are the same as the following:

- Using the Template Method allows us to create reusable code because these methods are a basic technique for code reuse, especially in class libraries.
- It can let us change the steps of the algorithm without changing the structure of it as we mentioned.

Despite these benefits, there is a pitfall when using this pattern that is all classes must follow the algorithm without exceptions.

**How can we use it in MQL5?**

If we need to use the Template Method pattern in MQL5 we can do that through the following steps as a method to do that:

Declare the AbstractClass by using the class keyword

```
class AbstractClass
  {
public:
   virtual void      PrimitiveOperation1(void)=0;
   virtual void      PrimitiveOperation2(void)=0;
   virtual void      TemplateMethod(void);
  };
void AbstractClass::TemplateMethod(void)
  {
   PrimitiveOperation1();
   PrimitiveOperation2();
  }
```

Declare the ConcreteClass by using the class keyword

```
  class ConcreteClass : public AbstractClass
  {
public:
   void              PrimitiveOperation1(void);
   void              PrimitiveOperation2(void);
  };
void ConcreteClass::PrimitiveOperation1(void)
  {
  }
void ConcreteClass::PrimitiveOperation2(void)
  {
  }
```

So, the following block of code is the full code to use the Template Method pattern in MQL5:

```
//+------------------------------------------------------------------+
//|                                              Template_Method.mqh |
//+------------------------------------------------------------------+
class AbstractClass
  {
public:
   virtual void      PrimitiveOperation1(void)=0;
   virtual void      PrimitiveOperation2(void)=0;
   virtual void      TemplateMethod(void);
  };
void AbstractClass::TemplateMethod(void)
  {
   PrimitiveOperation1();
   PrimitiveOperation2();
  }
  class ConcreteClass : public AbstractClass
  {
public:
   void              PrimitiveOperation1(void);
   void              PrimitiveOperation2(void);
  };
void ConcreteClass::PrimitiveOperation1(void)
  {
  }
void ConcreteClass::PrimitiveOperation2(void)
  {
  }
```

### Visitor

Here is the last type of behavior pattern, the Visitor pattern. It makes it possible to identify a new operation without having an impact on the classes of elements on which the operation is performed. We can use this pattern when we have an object structure containing many classes of objects and interfaces that are different and we need to execute operations on these objects, we have many different and unrelated operations that need to be executed on objects within the object structure, and there is a desire to avoid the "pollution" of their classes with these operations, and we have the classes that define the structure of the object rarely change, but you do need to define new operations on the structure.

**What does the pattern do?**

In the following graph, we can see how the pattern of the visitors is working through its structure:

![Visitor](https://c.mql5.com/2/63/Visitor.png)

As we can see in the previous graph we have the following participants:

- **Visitor:** for each class of ConcreteElement in the object structure, a 'Visit' operation is declared. The name and the signature of the operation uniquely identify the class that sends the request to visit the visitor. This design allows direct access to the element through its specific interface so that the visitor can determine the concrete class of the element being visited.
- **ConcreteVisitor:** it performs the implementation of each operation declared by the visitor. Each operation implements a fragment of the algorithm that is defined for the corresponding class of the object in the structure. The concrete visitor provides context for the algorithm and stores its local state. This state is often an accumulation of results during the iteration through the structure.
- **Element:** specifies an accepting operation that takes a visitor as an argument.
- **ConcreteElement:** it executes an accepting operation that takes the visitor as an argument.
- **ObjectStructure:** it is able to list its elements, it may provide a high-level interface to enable the visitor to visit its elements, and such as a list or a set, it can be either a composite or a collection.

**What does design pattern solve?**

According to what we mentioned about the visitor pattern we can find that it solves or has the following benefits are some of them when applying or using it:

- It makes so easy to add new operations.
- It lets us differentiate between related and unrelated operations because it gathers related and separate unrelated ones.

**How can we use it in MQL5?**

If we need to use the Visitor pattern in MQL5 the following are steps of a method to do that:

Declare the Visitor interface

```
interface Visitor;
```

Declare the Element class

```
class Element
  {
protected:
   Visitor*          m_visitor;
public:
                    ~Element(void);
   virtual void      Accept(Visitor* visitor)=0;
protected:
   void              Switch(Visitor* visitor);
  };
Element::~Element(void)
  {
   if(CheckPointer(m_visitor)==1)
      delete m_visitor;
  }
void Element::Switch(Visitor *visitor)
  {
   if(CheckPointer(m_visitor)==1)
      delete m_visitor;
   m_visitor=visitor;
  }
```

Declare the ConcreteElementA class

```
class ConcreteElementA : public Element
  {
public:
   void              Accept(Visitor*);
   void              OperationA(void);
  };
void ConcreteElementA::OperationA(void)
  {
  }
void ConcreteElementA::Accept(Visitor *visitor)
  {
   Switch(visitor);
   visitor.VisitElementA(&this);
  }
```

Declare the ConcreteElementB class

```
class ConcreteElementB : public Element
  {
public:
   void              Accept(Visitor* visitor);
   void              OperationB(void);
  };
void ConcreteElementB::OperationB(void)
  {
  }
void ConcreteElementB::Accept(Visitor *visitor)
  {
   Switch(visitor);
   visitor.VisitElementB(&this);
  }
```

Using the interface keyword to declare the Visitor to define VisitElementA and  VisitElementB within

```
interface Visitor
{
   void VisitElementA(ConcreteElementA*);
   void VisitElementB(ConcreteElementB*);
};
```

Declare the ConcreteVisitor1 class

```
class ConcreteVisitor1 : public Visitor
  {
public:
   void              VisitElementA(ConcreteElementA* visitor);
   void              VisitElementB(ConcreteElementB* visitor);
  };
void ConcreteVisitor1::VisitElementA(ConcreteElementA* visitor)
  {
   visitor.OperationA();
  }
void ConcreteVisitor1::VisitElementB(ConcreteElementB* visitor)
  {
   visitor.OperationB();
  }
```

Declare the ConcreteVisitor2 class

```
class ConcreteVisitor2 : public Visitor
  {
public:
   void              VisitElementA(ConcreteElementA*);
   void              VisitElementB(ConcreteElementB*);
  };
void ConcreteVisitor2::VisitElementA(ConcreteElementA* visitor)
  {
   visitor.OperationA();
  }
void ConcreteVisitor2::VisitElementB(ConcreteElementB* visitor)
  {
   visitor.OperationB();
  }
```

So, the following block of code is the full code to see the Visitor pattern in MQL5:

```
//+------------------------------------------------------------------+
//|                                                      Visitor.mqh |
//+------------------------------------------------------------------+
interface Visitor;
class Element
  {
protected:
   Visitor*          m_visitor;
public:
                    ~Element(void);
   virtual void      Accept(Visitor* visitor)=0;
protected:
   void              Switch(Visitor* visitor);
  };
Element::~Element(void)
  {
   if(CheckPointer(m_visitor)==1)
      delete m_visitor;
  }
void Element::Switch(Visitor *visitor)
  {
   if(CheckPointer(m_visitor)==1)
      delete m_visitor;
   m_visitor=visitor;
  }
class ConcreteElementA : public Element
  {
public:
   void              Accept(Visitor*);
   void              OperationA(void);
  };
void ConcreteElementA::OperationA(void)
  {
  }
void ConcreteElementA::Accept(Visitor *visitor)
  {
   Switch(visitor);
   visitor.VisitElementA(&this);
  }
class ConcreteElementB : public Element
  {
public:
   void              Accept(Visitor* visitor);
   void              OperationB(void);
  };
void ConcreteElementB::OperationB(void)
  {
  }
void ConcreteElementB::Accept(Visitor *visitor)
  {
   Switch(visitor);
   visitor.VisitElementB(&this);
  }
interface Visitor
  {
   void VisitElementA(ConcreteElementA*);
   void VisitElementB(ConcreteElementB*);
  };
class ConcreteVisitor1 : public Visitor
  {
public:
   void              VisitElementA(ConcreteElementA* visitor);
   void              VisitElementB(ConcreteElementB* visitor);
  };
void ConcreteVisitor1::VisitElementA(ConcreteElementA* visitor)
  {
   visitor.OperationA();
  }
void ConcreteVisitor1::VisitElementB(ConcreteElementB* visitor)
  {
   visitor.OperationB();
  }
class ConcreteVisitor2 : public Visitor
  {
public:
   void              VisitElementA(ConcreteElementA*);
   void              VisitElementB(ConcreteElementB*);
  };
void ConcreteVisitor2::VisitElementA(ConcreteElementA* visitor)
  {
   visitor.OperationA();
  }
void ConcreteVisitor2::VisitElementB(ConcreteElementB* visitor)
  {
   visitor.OperationB();
  }
```

### Conclusion

After reading this article and others in this series, you will be able to identify all the types of design patterns that are patterns (creative, structural, and behavioral). We identified each one by understanding how it works when we can use it, what issues or problems can be solved after using each pattern, and how we can use these patterns in the MQL5 to write clean code, which means we get maintainable, reusable, and well-tested code.

We have identified patterns that are the same as the following ones:

- Creational patterns

  - Abstract Factory
  - Builder
  - Factory Method
  - Prototype
  - Singleton

- Structural patterns

  - Adapter
  - Bridge
  - Composite
  - Decorator
  - Facade
  - Flyweight
  - Proxy

- Behavioral patterns

  - Chain of Responsibility
  - Command
  - Interpreter
  - Iterator
  - Mediator
  - Memento
  - Observer
  - State
  - Template
  - Visitor

You can read the other articles in this series by clicking on the links below:

- [Design Patterns in software development and MQL5 (Part I): Creational Patterns](https://www.mql5.com/en/articles/13622)
- [Design Patterns in software development and MQL5 (Part 2): Structural Patterns](https://www.mql5.com/en/articles/13724)
- [Design Patterns in software development and MQL5 (Part 3): Behavioral Patterns 1](https://www.mql5.com/en/articles/13796)

I cannot stress enough the importance of understanding the topic of design patterns because it can be very useful in creating any software as we have mentioned, so I recommend reading more on this topic and the following some references on the same topic:

- Design Patterns - Elements of Reusable Object-Oriented Software by Eric Gamma, Richard Helm, Ralph Johnson, and John Vlissides
- Design Patterns for Dummies by Steve Holzner
- Head First Design Patterns by Eric Freeman, Elisabeth Robson, Bert Bates, and Kathy Sierra

I hope that you found this article and series useful for your journey to improve your programming skills in general and MQL5 in particular. If you need to read more articles about programming by the MQL5 and how you can create trading systems based on the most popular technical indicators like moving averages, RSI, Bollinger Bands, and MACD for examples you can check my [publications](https://www.mql5.com/en/users/m.aboud/publications) page and I hope that you find them useful also to improve your trading and programming background.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/13876.zip "Download all attachments in the single ZIP archive")

[Memento.mqh](https://www.mql5.com/en/articles/download/13876/memento.mqh "Download Memento.mqh")(1.34 KB)

[Observer.mqh](https://www.mql5.com/en/articles/download/13876/observer.mqh "Download Observer.mqh")(2.5 KB)

[State.mqh](https://www.mql5.com/en/articles/download/13876/state.mqh "Download State.mqh")(1.19 KB)

[Strategy.mqh](https://www.mql5.com/en/articles/download/13876/strategy.mqh "Download Strategy.mqh")(1.24 KB)

[Template\_Method.mqh](https://www.mql5.com/en/articles/download/13876/template_method.mqh "Download Template_Method.mqh")(0.77 KB)

[Visitor.mqh](https://www.mql5.com/en/articles/download/13876/visitor.mqh "Download Visitor.mqh")(2.14 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [How to build and optimize a cycle-based trading system (Detrended Price Oscillator - DPO)](https://www.mql5.com/en/articles/19547)
- [How to build and optimize a volume-based trading system (Chaikin Money Flow - CMF)](https://www.mql5.com/en/articles/16469)
- [MQL5 Integration: Python](https://www.mql5.com/en/articles/14135)
- [How to build and optimize a volatility-based trading system (Chaikin Volatility - CHV)](https://www.mql5.com/en/articles/14775)
- [Advanced Variables and Data Types in MQL5](https://www.mql5.com/en/articles/14186)
- [Building and testing Keltner Channel trading systems](https://www.mql5.com/en/articles/14169)
- [Building and testing Aroon Trading Systems](https://www.mql5.com/en/articles/14006)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/459454)**
(3)


![DidMa](https://c.mql5.com/avatar/2023/12/65866a53-9ed8.png)

**[DidMa](https://www.mql5.com/en/users/didma)**
\|
23 Dec 2023 at 05:03

Thank you for this great article.

I think I implemented the strategy pattern in an Expert Advisor, but maybe in a different way

This could give a more practical example for our beloved readers mind, and for me maybe I would like to get your feedback if I made conceptual mistakes:

I'll simplify the process here:

the Expert itself is an object and becomes the Context:

- I have an Enum to be able to choose the strategy (as an example STRATEGY\_RSI, STRATEGY\_MA,...)
- I have a switch statement in the OnInit(), on the strategy enum, and depending on the result, I

```
switch(strategyName)
        {
         case STRATEGY_RSI :
            m_strategy = new Strategy_RSI();
            break;
         case STRATEGY_MA :
            m_strategy = new Strategy_MA();
            break;
         default:
            break;
}
```

- I set the input parameters into the strategy through a parameter object (like mqlparams)

```
[...]
// add the signal parameters to the strategy through the expert Method
Expert.AddParameterObject(signal_RSI_Params); // signal_RSI_Params is the parameter object that has the input parameters inside

//detail of the AddParameterObject method (simplified):
bool CExpert::AddParameterObject(CParameter & parameter)
  {
         if(!m_strategy.AddParameterObject(parameter))
           {
            return(false);
           }
     }
   return true;
  }

// add the generic expert parameters.
Expert.InitStrategyParameters(expert_1_Params);

//detail of the method  (simplified) :
bool CExpert::InitStrategyParameters(CParameter & expertParameters)
  {
//
   if(!m_strategy.InitStrategyParameters(expertParameters))
     {
     [...]
      return false;
     }
   return true;
  }
```

- All the strategies implement the same methods (open a signal, close trade,...)
- All the strategies can load a Signal or its [custom indicators](https://www.mql5.com/en/articles/5 "Article: Step on New Rails: Custom Indicators in MQL5 ")


the in the OnTick function of the expert I call the generic Strategy methods

```
bool CExpert::CheckOpenLong(void)
  {
    [...]
    if(true == m_strategy.CheckOpenLong([...]))
     {
      [...]
     }
    [...]
}
```

It's working ver well as I can also test the several strategies / optimisation with the same expert.

The only thing I never found out, besides putting the input parameters in a separate file, is how to load dynamically input parameters according to what the user is choosing as this particularinput parameter (i.e. the strategy input).

I'd like to get your feedback on the way I implemented, is it a strategy pattern, or a hybrid, what could be improved?

And do you have a idea how we could have (I don't think it's possible with MT5) dynamic contextual input parameters ?

Thank you so much

Didma

![Altan Karakaya](https://c.mql5.com/avatar/2023/11/654e0408-75ed.png)

**[Altan Karakaya](https://www.mql5.com/en/users/tradewizards)**
\|
24 Dec 2023 at 08:46

very instructive

![Yuriy Yepifanov](https://c.mql5.com/avatar/2024/3/65f05b85-1fc9.jpg)

**[Yuriy Yepifanov](https://www.mql5.com/en/users/eurweb)**
\|
16 Apr 2024 at 12:06

Why do functions with the same functionality have different names

e.g. here

```
void Memento::SetState(string state)
  {
   m_state=state;
  }


void Originator::State(string state)
  {
   m_state=state;
  }
```

![Neural networks made easy (Part 56): Using nuclear norm to drive research](https://c.mql5.com/2/57/nuclear_norm_utilization_avatar.png)[Neural networks made easy (Part 56): Using nuclear norm to drive research](https://www.mql5.com/en/articles/13242)

The study of the environment in reinforcement learning is a pressing problem. We have already looked at some approaches previously. In this article, we will have a look at yet another method based on maximizing the nuclear norm. It allows agents to identify environmental states with a high degree of novelty and diversity.

![Data Science and Machine Learning (Part 17): Money in the Trees? The Art and Science of Random Forests in Forex Trading](https://c.mql5.com/2/63/midjourney_image_13765_54_491__3-logo.png)[Data Science and Machine Learning (Part 17): Money in the Trees? The Art and Science of Random Forests in Forex Trading](https://www.mql5.com/en/articles/13765)

Discover the secrets of algorithmic alchemy as we guide you through the blend of artistry and precision in decoding financial landscapes. Unearth how Random Forests transform data into predictive prowess, offering a unique perspective on navigating the complex terrain of stock markets. Join us on this journey into the heart of financial wizardry, where we demystify the role of Random Forests in shaping market destiny and unlocking the doors to lucrative opportunities

![Making a dashboard to display data in indicators and EAs](https://c.mql5.com/2/57/information_panel_for_displaying_data_avatar.png)[Making a dashboard to display data in indicators and EAs](https://www.mql5.com/en/articles/13179)

In this article, we will create a dashboard class to be used in indicators and EAs. This is an introductory article in a small series of articles with templates for including and using standard indicators in Expert Advisors. I will start by creating a panel similar to the MetaTrader 5 data window.

![Neural networks made easy (Part 55): Contrastive intrinsic control (CIC)](https://c.mql5.com/2/57/cic-055-avatar.png)[Neural networks made easy (Part 55): Contrastive intrinsic control (CIC)](https://www.mql5.com/en/articles/13212)

Contrastive training is an unsupervised method of training representation. Its goal is to train a model to highlight similarities and differences in data sets. In this article, we will talk about using contrastive training approaches to explore different Actor skills.

[![](https://www.mql5.com/ff/si/s2n3m9ymjh52n07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F523%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dchoose.signals%26utm_content%3Dsubscribe.signal%26utm_campaign%3D0622.MQL5.com.Internal&a=fyznzyduwsltgnhlftytumasbfgbwlqw&s=91bc0eca8f132d3df7d14cdb1baebac753aef179403d60dc83856af55a4d6769&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=cyokktkpdopgfybcdvbhsslvbtdlirwq&ssn=1769179058483502432&ssn_dr=0&ssn_sr=0&fv_date=1769179058&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F13876&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Design%20Patterns%20in%20software%20development%20and%20MQL5%20(Part%204)%3A%20Behavioral%20Patterns%202%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176917905886789461&fz_uniq=5068460571004697010&sv=2552)

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