---
title: Design Patterns in software development and MQL5 (Part 3): Behavioral Patterns 1
url: https://www.mql5.com/en/articles/13796
categories: Trading, Trading Systems, Indicators, Expert Advisors
relevance_score: 4
scraped_at: 2026-01-23T17:37:59.701172
---

[![](https://www.mql5.com/ff/si/h2ryn394uwcpxwmxc2.jpg)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Feconomic-calendar%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dopen.calendar%26utm_content%3Deconomic.calendar%26utm_campaign%3Den.0009.desktop.default&a=qdeulxgvibgwytgewnvfatbocjnnninc&s=5c0c60f00ff5f5bedb0fdf65d9d79eb820442eb43ffac2b85aa003224f9dba14&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=pcaqptsaawfayymzsgqnzwqvqzpqkfjw&ssn=1769179078586514697&ssn_dr=0&ssn_sr=0&fv_date=1769179078&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F13796&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Design%20Patterns%20in%20software%20development%20and%20MQL5%20(Part%203)%3A%20Behavioral%20Patterns%201%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176917907842047415&fz_uniq=5068467387117795782&sv=2552)

MetaTrader 5 / Trading


### Introduction

In this article, we will continue our series about the Design Patterns topic in the software domain. We identified two types of these patterns which are the Creational and Structural patterns in the previous two articles in this series and in this article, we will identify the Behavioral Design patterns which is the third type after identifying and understanding what behavioral patterns and how they can be useful when creating, building, or developing our software.  After that, we will learn how we can use them in the MQL5 to create our software for MetaTrader 5 to create a reliable, maintainable, reusable, well-tested, and extendable software.

The following topics are about what we will mention to cover this important type of pattern:

- [Behavioral patterns](https://www.mql5.com/en/articles/13796#behavioral)
- [Chain of responsibility](https://www.mql5.com/en/articles/13796#chain)
- [Command](https://www.mql5.com/en/articles/13796#command)
- [Interpreter](https://www.mql5.com/en/articles/13796#interpreter)
- [Iterator](https://www.mql5.com/en/articles/13796#iterator)
- [Mediator](https://www.mql5.com/en/articles/13796#mediator)
- [Conclusion](https://www.mql5.com/en/articles/13796#conclusion)

If this is the first reading article in this series, I hope that you read other articles about [Creational](https://www.mql5.com/en/articles/13622) and [Structural](https://www.mql5.com/en/articles/13724) patterns to take an overall view of one of the most important topics in software development which is Design Patterns because it can be very useful in your way to create your software.

Disclaimer: All information provided 'as is' only for educational purposes and is not prepared for trading purposes or advice. The information does not guarantee any kind of result. If you choose to use these materials on any of your trading accounts, you will do that at your own risk and you will be the only responsible.

### Behavioral patterns

In the context of talking about Design Patterns and after talking about Creational and Structural Patterns, we will talk about the last type of these design patterns which are behavioral Patterns. We learned that creational patterns are these patterns that help to create an independent software or system by creating, composing, and representing objects. In addition to that learning that structural patterns are these patterns that can be used to build larger structures by using created objects and classes.

In this article, we will provide the behavioral patterns that are concerned with assigning and setting how are responsibilities between objects. They also identify how objects can communicate or interact with each other and there are many patterns under this type, the same as the following:

- Chain of responsibility
- Command
- Interpreter
- Iterator
- Mediator
- Memento
- Observer
- State
- Strategy
- Template Method
- Visitor

Because there are many patterns that can not be covered in one article, our focus in this article will be on the first five patterns only the same as the following:

- Chain of responsibility: it helps to apply decoupling the sender and its receiver by giving the opportunity for more than one object to handle the request from the sender. It helps also to chain receiving objects and pass the request along this chain to complete handling from an object.
- Command: it gives the permission to set parameters to clients with different requests, queue or log requests, and support undoable operations after encapsulating the request as an object.
- Interpreter: it defines a representation of the grammar of a given language along with an interpreter that can use this defined representation to interpret what is needed as sentences in the language.
- Iterator: if we need a sequence access to elements of an aggregate object without exposing its underlying representation, this pattern helps to provide us with a method to do that.
- Mediator: it defines how a set of objects interact through an encapsulated object and promotes decoupling by letting us vary the interaction of objects independently.

If you read the previous two articles about Design Patterns, you will be familiar with the approach that we will use to cover every pattern and it will be the same as the following:

- What does the pattern do?
- What does design pattern solve?
- How can we use it in MQL5?

### Chain of responsibility

In this part, we will understand what is the Chain of responsibility by learning what it can do, and solve, and how we can use it in the MQL5. When we need to handle a request from the client, in case we have many objects that can handle the requests of client based on the responsibility of everyone, we can use this pattern to handle this case.

Despite the advantages of using this pattern, there are pitfalls that we can face such as the following:

- Efficiency issue in case of long chains.
- No guarantee for request handling because the request has no specified receiver, so this request can be passed to objects in the chain without handling. In addition to that we need to know that the request can not be handled if we have no properly configured chain.

**What does the pattern do?**

This pattern can be useful in decoupling the sender of any request and the receiver of that request by giving the opportunity of many objects to handle the request. This happens through chaining the receiving objects and passing the request to everyone to check which one can handle the request.

The following is the graph for the structure of the pattern:

![CHAIN OF RESPONSIBILITY](https://c.mql5.com/2/61/CHAIN_OF_RESPONSIBILITY.png)

As we can see in the previous graph, we have the following participants:

- **Client:** the client is initiating the request to be handled by the object in the chain.
- **Handler:** it defines the interface to handle the request. It can also implement the successor link.
- **ConcreteHandler:** it is the object that handles the request based on its responsibility, it has an access to a successor that can pass to it when it cannot handle the request.

**What does design pattern solve?**

This pattern can be used if the following is applicable:

- We have many objects that can handle requests.
- We need to decouple the sender and the receiver.
- We need to produce a request to one of many objects without mentioning the receiver.
- We have a dynamically specified set of objects that can handle the request.

So, this pattern can solve the following:

- Coupling reduction because it helps decoupling the sender and the receiver which means that it helps to apply independent changes.
- It grants a flexibility when assigning and distributing responsibilities to objects.

**How can we use it in MQL5?**

In this part, we will learn how to use this pattern in MQL5 to create effective MetaTrader5 software, so, the following are steps to code the Chain of Responsibility in the MQL5:

Declaring the Chain\_Of\_Responsibility area to include functions and variables of the pattern within by using the namespace keyword

```
namespace Chain_Of_Responsibility
```

Declare the participant of the Handler class that handles requests from the client and may implement the successor link

```
class Handler
  {
public:
   Handler*          successor;
   virtual void      HandleRequest(int)=0;
                    ~Handler(void);
  };
Handler::~Handler(void)
  {
   delete successor;
  }
```

Declare the participant of the ConcreteHandler1 class that handles requests that it is responsible for or pass the request to its successor if can handle it

```
class ConcreteHandler1:public Handler
  {
public:
   void              HandleRequest(int);
  };
void ConcreteHandler1::HandleRequest(int request)
  {
   if(request==1)
      Print("The request: ",request,". The request handled by: ",&this);
   else
      if(CheckPointer(successor))
        {
         Print("The request: ",request,". The request cannot be handled by ",&this,", but it is forwarding to the successor...");
         successor.HandleRequest(request);
        }
  }
```

Declare the ConcreteHandler2 class as a participant also

```
class ConcreteHandler2:public Handler
  {
public:
   void              HandleRequest(int);
  };
void ConcreteHandler2::HandleRequest(int request)
  {
   if(request==2)
      Print("The request: ",request,". The request handled by: ",&this);
   else
      if(CheckPointer(successor))
        {
         Print("The request: ",request,". The request cannot be handled by ",&this,", forwarding to successor...");
         successor.HandleRequest(request);
        }
  }
```

Declare the client class that initiates the request to the concrete handler in the chain

```
class Client
  {
public:
   string            Output();
   void              Run();
  };
string Client::Output()
  {
   return __FUNCTION__;
  }
```

The function of running the client is to send the request to the chain to be handled or passed to the successor

```
void   Client::Run()
  {
   Handler* h1=new ConcreteHandler1();
   Handler* h2=new ConcreteHandler2();
   h1.successor=h2;
   h1.HandleRequest(1);
   h1.HandleRequest(2);
   delete h1;
  }
```

So, the following is the full code to use the Chain of Responsibility pattern in the MQL5 in one block

```
//+------------------------------------------------------------------+
//|                                      Chain_Of_Responsibility.mqh |
//+------------------------------------------------------------------+
namespace Chain_Of_Responsibility
{
class Handler
  {
public:
   Handler*          successor;
   virtual void      HandleRequest(int)=0;
                    ~Handler(void);
  };
Handler::~Handler(void)
  {
   delete successor;
  }
class ConcreteHandler1:public Handler
  {
public:
   void              HandleRequest(int);
  };
void ConcreteHandler1::HandleRequest(int request)
  {
   if(request==1)
      Print("The request: ",request,". The request handled by: ",&this);
   else
      if(CheckPointer(successor))
        {
         Print("The request: ",request,". The request cannot be handled by ",&this,", but it is forwarding to the successor...");
         successor.HandleRequest(request);
        }
  }
class ConcreteHandler2:public Handler
  {
public:
   void              HandleRequest(int);
  };
void ConcreteHandler2::HandleRequest(int request)
  {
   if(request==2)
      Print("The request: ",request,". The request handled by: ",&this);
   else
      if(CheckPointer(successor))
        {
         Print("The request: ",request,". The request cannot be handled by ",&this,", but it is forwarding to successor...");
         successor.HandleRequest(request);
        }
  }
class Client
  {
public:
   string            Output();
   void              Run();
  };
string Client::Output()
  {
   return __FUNCTION__;
  }
void   Client::Run()
  {
   Handler* h1=new ConcreteHandler1();
   Handler* h2=new ConcreteHandler2();
   h1.successor=h2;
   h1.HandleRequest(1);
   h1.HandleRequest(2);
   delete h1;
  }
}
```

### Command

In this part, we will identify another Behavioral pattern which is the Command pattern also known as Action and Transaction. This Pattern helps to encapsulate the request in an object and this allows us to set our parameters for different requests without changing the sender or receiver which means that now there is a decoupling applied between the sender, the processor, and the receiver. This is very helpful when we have huge functionalities within classes. This pattern also supports undo operations.

The same as most things there are some pitfalls when using this pattern and they are the same as the following:

- It is usually used in combination with other patterns.
- It results high number of classes and objects to handle all different commands or requests.
- Creating an object for each command is against the Object Oriented Design.

**What does the pattern do?**

Simply, It creates an encapsulated invoker to receive the command and send it to the receiver.

The following is for the graph of the Command pattern:

![COMMAND](https://c.mql5.com/2/61/COMMAND.png)

As we can see in the previous graph of the structure of the Command pattern we have the following as participants:

- **Command:** This command can declare the interface to execute the operation.
- **ConcreteCommand:** This ConcreteCommand create a link between the receiver and the action or the command in addition to that through invoking the corresponding operation on the receiver, it implements the execution.
- **Client:** this can apply two things creating the ConcreteCommand and setting its receiver.
- **Invoker:** it receives the command to execute the request.
- **Receiver:** It identifies the method of the operations associated with the execution of the request and it can be any class.

**What does design pattern solve?**

- We can use this pattern when we find the following cases are applicable:
- We need to set parameters to objects by action to perform.
- We need to operate like specification, queuing, and executing at different times.
- We need something that can be used to support the undo operation.
- In case of crashing in a system, we need support of logging changes to be reapplied in that case.
- We need high-level operations built on primitive ones as a structure of the system.

**How can we use it in MQL5?**

In this part, we will take a look at a method that can be used to use this Command pattern in the MQL5 and the following steps are for doing that:

Declaring our Command space for specifying our functions, variables, classes ... etc by using the namespace keyword

```
namespace Command
```

Declaring the Receiver class as a participant which identifies the method to perform operations of the request

```
class Receiver
  {
public:
                     Receiver(void);
                     Receiver(Receiver&);
   void              Action(void);
  };
Receiver::Receiver(void)
  {
  }
Receiver::Receiver(Receiver &src)
  {
  }
void Receiver::Action(void)
  {
  }
```

Declaring the Command class as a participant to declare the operation interface

```
class Command
  {
protected:
   Receiver*         m_receiver;
public:
                     Command(Receiver*);
                    ~Command(void);
   virtual void      Execute(void)=0;
  };
Command::Command(Receiver* receiver)
  {
   m_receiver=new Receiver(receiver);
  }
Command::~Command(void)
  {
   if(CheckPointer(m_receiver)==1)
     {
      delete m_receiver;
     }
  }
```

Declaring the ConcreteCommand class as a participant to create the link between the receiver and the action or the command and implement execute() after calling the receiver operation

```
class ConcreteCommand:public Command
  {
protected:
   int               m_state;
public:
                     ConcreteCommand(Receiver*);
   void              Execute(void);
  };
ConcreteCommand::ConcreteCommand(Receiver* receiver):
   Command(receiver),
   m_state(0)
  {
  }
void ConcreteCommand::Execute(void)
  {
   m_receiver.Action();
   m_state=1;
  }
```

Declaring the Invoker class as a participant to receive the command to execute the request

```
class Invoker
  {
public:
                    ~Invoker(void);
   void              StoreCommand(Command*);
   void              Execute(void);
protected:
   Command*          m_command;
  };
Invoker::~Invoker(void)
  {
   if(CheckPointer(m_command)==1)
     {
      delete m_command;
     }
  }
void Invoker::StoreCommand(Command* command)
  {
   m_command=command;
  }
void Invoker::Execute(void)
  {
   m_command.Execute();
  }
```

Declaring the client class as a participant to create the concrete command set its receiver and run it

```
class Client
  {
public:
   string            Output(void);
   void              Run(void);
  };
string Client::Output(void)
  {
   return __FUNCTION__;
  }
void Client::Run(void)
  {
   Receiver receiver;
   Invoker invoker;
   invoker.StoreCommand(new ConcreteCommand(&receiver));
   invoker.Execute();
  }
```

So, we can find the following code is for the full code in one block of code

```
//+------------------------------------------------------------------+
//|                                                      Command.mqh |
//+------------------------------------------------------------------+
namespace Command
{
class Receiver
  {
public:
                     Receiver(void);
                     Receiver(Receiver&);
   void              Action(void);
  };
Receiver::Receiver(void)
  {
  }
Receiver::Receiver(Receiver &src)
  {
  }
void Receiver::Action(void)
  {
  }
class Command
  {
protected:
   Receiver*         m_receiver;
public:
                     Command(Receiver*);
                    ~Command(void);
   virtual void      Execute(void)=0;
  };
Command::Command(Receiver* receiver)
  {
   m_receiver=new Receiver(receiver);
  }
Command::~Command(void)
  {
   if(CheckPointer(m_receiver)==1)
     {
      delete m_receiver;
     }
  }
class ConcreteCommand:public Command
  {
protected:
   int               m_state;
public:
                     ConcreteCommand(Receiver*);
   void              Execute(void);
  };
ConcreteCommand::ConcreteCommand(Receiver* receiver):
   Command(receiver),
   m_state(0)
  {
  }
void ConcreteCommand::Execute(void)
  {
   m_receiver.Action();
   m_state=1;
  }
class Invoker
  {
public:
                    ~Invoker(void);
   void              StoreCommand(Command*);
   void              Execute(void);
protected:
   Command*          m_command;
  };
Invoker::~Invoker(void)
  {
   if(CheckPointer(m_command)==1)
     {
      delete m_command;
     }
  }
void Invoker::StoreCommand(Command* command)
  {
   m_command=command;
  }
void Invoker::Execute(void)
  {
   m_command.Execute();
  }
class Client
  {
public:
   string            Output(void);
   void              Run(void);
  };
string Client::Output(void)
  {
   return __FUNCTION__;
  }
void Client::Run(void)
  {
   Receiver receiver;
   Invoker invoker;
   invoker.StoreCommand(new ConcreteCommand(&receiver));
   invoker.Execute();
  }
}
```

### Interpreter

Another Behavioral pattern that can be used to help us to set interaction between objects through given a language and define the representation for rules or grammar with an interpreter that can use this representation later to explain and interpret the content of this language.

There are pitfalls when using this type of pattern the same as the following:

- The complexity of rules or grammar and when the degree of complexity is high, this means difficulty when aiming to maintain.
- It can be used in specific situations.

**What does the pattern do?**

This pattern helps us describe how to define the grammar of a language, represent the content of this language, and get an interpretation of this content.

If we need to see the graph of the interpreter pattern, it will be the same as the following:

![Interpreter](https://c.mql5.com/2/61/Interpreter.png)

As we can see in the previous graph we have the following participants in this pattern:

- **AbstractExpression:** It can declare the operation of abstract interpret (context).
- **TerminalExpression:** It can implement the operation of interpret that is associated with symbols of the terminal in the grammar, creating an instance as a requirement for each terminal symbol in the content.
- **NonterminalExression:** For every rule in the grammar, there is required one class, it maintains instances variables of the AbstractExpression for every rule. it implements the operation of the interpret for nonterminal symbols in the grammar.
- **Context:** It contains global information for the interpreter.
- **Client:** It builds the abstract syntax tree to represent the content in the language that we need it to be defined by the grammar, and it invokes the operation of the interpret.

**What does design pattern solve?**

As we identified, this pattern can be used when we have a language that we need to interpret and we can define or represent content in the language.

So, the following are the best cases that we can use the pattern for:

- We have a simple grammar of the language because if the grammar is complex the hierarchy of the class will become large and this can lead to an unmanaged state.
- If the efficiency factor in the interpreter is not so crucial.

So, by using this pattern we can get the following benefits:

- Using this pattern makes it easy and smooth to update or extend the grammar by using inheritance.
- It makes it easy also to implement the grammar.
- It makes it easier to add new ways to interpret expressions.

**How can we use it in MQL5?**

In this part of the article, we will present a simple method to code or use this type of pattern. The following are steps to use this Interpreter in the MQL5:

Declare the area of Interpreter that we will use to define and declare our functions, variables, and classes the same as we know by using the namespace keyword

```
namespace Interpreter
```

Declare the context class as a participant

```
class Context
  {
public:
   string            m_source;
   char              m_vocabulary;
   int               m_position;
   bool              m_result;
   //---
                     Context(char,string);
   void              Result(void);
  };
Context::Context(char vocabulary,string source):
   m_source(source),
   m_vocabulary(vocabulary),
   m_position(0),
   m_result(false)
  {
  }
void Context::Result(void)
  {
  }
```

Declare the Abstract class as a participant

```
class AbstractExpression
  {
public:
   virtual void      Interpret(Context&)=0;
  };
```

Declare the TerminalExpression class as a participant to implement the interpret method

```
class TerminalExpression:public AbstractExpression
  {
public:
   void              Interpret(Context&);
  };
void TerminalExpression::Interpret(Context& context)
  {
   context.m_result=
      StringSubstr(context.m_source,context.m_position,1)==
      CharToString(context.m_vocabulary);
  }
```

Declare the NonterminalExpression class as a participant

```
class NonterminalExpression:public AbstractExpression
  {
protected:
   AbstractExpression* m_nonterminal_expression;
   AbstractExpression* m_terminal_expression;
public:
   void              Interpret(Context&);
                    ~NonterminalExpression(void);
  };
NonterminalExpression::~NonterminalExpression(void)
  {
   delete m_nonterminal_expression;
   delete m_terminal_expression;
  }
void NonterminalExpression::Interpret(Context& context)
  {
   if(context.m_position<StringLen(context.m_source))
     {
      m_terminal_expression=new TerminalExpression;
      m_terminal_expression.Interpret(context);
      context.m_position++;
      if(context.m_result)
        {
         m_nonterminal_expression=new NonterminalExpression;
         m_nonterminal_expression.Interpret(context);
        }
     }
  }
```

Declare the client as a participant

```
class Client
  {
public:
   string            Output(void);
   void              Run(void);
  };
string Client::Output(void) {return __FUNCTION__;}
void Client::Run(void)
  {
   Context context_1('a',"aaa");
   Context context_2('a',"aba");
   AbstractExpression* expression;
   expression=new NonterminalExpression;
   expression.Interpret(context_1);
   context_1.Result();
   delete expression;
   expression=new NonterminalExpression;
   expression.Interpret(context_2);
   context_2.Result();
   delete expression;
  }
```

So, the following is the full code to use the interpreter pattern in one block of code

```
//+------------------------------------------------------------------+
//|                                                  Interpreter.mqh |
//+------------------------------------------------------------------+
namespace Interpreter
{
class Context
  {
public:
   string            m_source;
   char              m_vocabulary;
   int               m_position;
   bool              m_result;
                     Context(char,string);
   void              Result(void);
  };
Context::Context(char vocabulary,string source):
   m_source(source),
   m_vocabulary(vocabulary),
   m_position(0),
   m_result(false)
  {
  }
void Context::Result(void)
  {
  }
class AbstractExpression
  {
public:
   virtual void      Interpret(Context&)=0;
  };
class TerminalExpression:public AbstractExpression
  {
public:
   void              Interpret(Context&);
  };
void TerminalExpression::Interpret(Context& context)
  {
   context.m_result=
      StringSubstr(context.m_source,context.m_position,1)==
      CharToString(context.m_vocabulary);
  }
class NonterminalExpression:public AbstractExpression
  {
protected:
   AbstractExpression* m_nonterminal_expression;
   AbstractExpression* m_terminal_expression;
public:
   void              Interpret(Context&);
                    ~NonterminalExpression(void);
  };
NonterminalExpression::~NonterminalExpression(void)
  {
   delete m_nonterminal_expression;
   delete m_terminal_expression;
  }
void NonterminalExpression::Interpret(Context& context)
  {
   if(context.m_position<StringLen(context.m_source))
     {
      m_terminal_expression=new TerminalExpression;
      m_terminal_expression.Interpret(context);
      context.m_position++;
      if(context.m_result)
        {
         m_nonterminal_expression=new NonterminalExpression;
         m_nonterminal_expression.Interpret(context);
        }
     }
  }
class Client
  {
public:
   string            Output(void);
   void              Run(void);
  };
string Client::Output(void) {return __FUNCTION__;}
void Client::Run(void)
  {
   Context context_1('a',"aaa");
   Context context_2('a',"aba");
   AbstractExpression* expression;
   expression=new NonterminalExpression;
   expression.Interpret(context_1);
   context_1.Result();
   delete expression;
   expression=new NonterminalExpression;
   expression.Interpret(context_2);
   context_2.Result();
   delete expression;
  }
}
```

### Iterator

We will identify the Iterator design pattern which is one of the behavioral patterns to set the method of interaction or communication between objects. This pattern helps that an aggregate object such as a list is present or gives us a method to access elements in a sequenced way without exposing its underlying details of representation or internal structure. It is also known as Cursor.

Despite the benefits we can get when using this Iterator pattern there are pitfalls the same as below:

- If we have a collection, there is no access to the index of it.
- Unidirectional state in some case when we need to be directed to the previous for example but there is solutions for that in some languages.
- In some cases, if we create an index and loop through it, it will be faster than using the pattern.

**What does the pattern do?**

This pattern can support variations in case of complex aggregates that may be traversed in different ways because it is easy to update the traversal algorithm by replacing the instance of the iterator in addition to defining subclasses for the iterator to support the updated traversals. It makes the interface of the aggregate simple. Keeping track of the traversal state of the iterator allows that we can have many traversals processed at one time.

We can find the graph of this pattern the same as the following graph:

![ITERATOR](https://c.mql5.com/2/61/ITERATOR.png)

Based on the previous graph of the Iterator structure we can find the following participants:

- **Iterator:** it defines the interface that can be used to access and traverse elements.
- **ConcreteIterator:** it allows the interface of the iterator to be implemented, it keeps track of the aggregate traversal.
- **Aggregate:** it helps to identify the interface that can be used to create the object iterator.
- **ConcreteAggregate:** it can be helpful to implement the iterator interface to get the instance of the suitable ConcreteIterator as a return.

**What does design pattern solve?**

This iterator pattern can be used when we have the following applicable:

- If we need to access the content of the aggregate object but we do not need to expose the internal representation of it.
- If we have many traversals of aggregate objects we need to support that.
- If we have different aggregate structures we need to present a uniform interface for traversing them.

**How can we use it in MQL5?**

In this part, we will learn how we can use this pattern in the MQL5 through the following steps:

Defining ERRITERAOR-UT-OF-BOUNDS by using the preprocessor #define

```
#define ERR_ITERATOR_OUT_OF_BOUNDS 1
```

We will use the template keyword and declare T as a CurrentItem in the defined Iterator interface

```
template<typename T>
interface Iterator
  {
   void     First(void);
   void     Next(void);
   bool     IsDone(void);
   T        CurrentItem(void);
  };
```

Also, we will use the template keyword and declare T as an operator in the defined Aggregate interface

```
template<typename T>
interface Aggregate
  {
   Iterator<T>*   CreateIterator(void);
   int            Count(void);
   T              operator[](int at);
   void           operator+=(T item);
  };
```

Implementing the iterator interface and keeping track of the aggregate traversal as a current position after declaring the ConcreteIterator class

```
template<typename T>
class ConcreteIterator:public Iterator<T>
  {
public:
   void              First(void);
   void              Next(void);
   bool              IsDone(void);
   T                 CurrentItem(void);
                     ConcreteIterator(Aggregate<T>&);
protected:
   Aggregate<T>*     m_aggregate;
   int               m_current;
  };
template<typename T>
   ConcreteIterator::ConcreteIterator(Aggregate<T>& aggregate):
   m_aggregate(&aggregate),
   m_current(0)
  {
  }
template<typename T>
void ConcreteIterator::First(void)
  {
   m_current=0;
  }
template<typename T>
void ConcreteIterator::Next(void)
  {
   m_current++;
   if(!IsDone())
     {
     }
  }
template<typename T>
bool ConcreteIterator::IsDone(void)
  {
   return m_current>=m_aggregate.Count();
  }
template<typename T>
string ConcreteIterator::CurrentItem(void)
  {
   if(IsDone())
     {
      SetUserError(ERR_ITERATOR_OUT_OF_BOUNDS);
      return NULL;
     }
   return m_aggregate[m_current];
  }
```

Implementing the iterator creation interface to get the instance of the suitable concrete iterator as a return value

```
class ConcreteAggregate:public Aggregate<string>
  {
public:
   Iterator<string>* CreateIterator(void);
   int               Count(void);
   void              operator+=(string item);
   string            operator[](int at);
protected:
   string            m_items[];
  };
Iterator<string>* ConcreteAggregate::CreateIterator(void)
  {
   return new ConcreteIterator<string>(this);
  }
void ConcreteAggregate::operator+=(string item)
  {
   int size=ArraySize(m_items);
   ArrayResize(m_items,size+1);
   m_items[size]=item;
  }
string ConcreteAggregate::operator[](int at)
  {
   return m_items[at];
  }
int ConcreteAggregate::Count()
  {
   return ArraySize(m_items);
  }
```

So, the following is the full code in one block of code to use the iterator pattern in the MQL5

```
//+------------------------------------------------------------------+
//|                                                201021_104101.mqh |
//+------------------------------------------------------------------+
#define ERR_ITERATOR_OUT_OF_BOUNDS 1
template<typename T>
interface Iterator
  {
   void     First(void);
   void     Next(void);
   bool     IsDone(void);
   T        CurrentItem(void);
  };
template<typename T>
interface Aggregate
  {
   Iterator<T>*   CreateIterator(void);
   int            Count(void);
   T              operator[](int at);
   void           operator+=(T item);
  };

template<typename T>
class ConcreteIterator:public Iterator<T>
  {
public:
   void              First(void);
   void              Next(void);
   bool              IsDone(void);
   T                 CurrentItem(void);
                     ConcreteIterator(Aggregate<T>&);
protected:
   Aggregate<T>*     m_aggregate;
   int               m_current;
  };
template<typename T>
   ConcreteIterator::ConcreteIterator(Aggregate<T>& aggregate):
   m_aggregate(&aggregate),
   m_current(0)
  {
  }
template<typename T>
void ConcreteIterator::First(void)
  {
   m_current=0;
  }
template<typename T>
void ConcreteIterator::Next(void)
  {
   m_current++;
   if(!IsDone())
     {
     }
  }
template<typename T>
bool ConcreteIterator::IsDone(void)
  {
   return m_current>=m_aggregate.Count();
  }
template<typename T>
string ConcreteIterator::CurrentItem(void)
  {
   if(IsDone())
     {
      SetUserError(ERR_ITERATOR_OUT_OF_BOUNDS);
      return NULL;
     }
   return m_aggregate[m_current];
  }
class ConcreteAggregate:public Aggregate<string>
  {
public:
   Iterator<string>* CreateIterator(void);
   int               Count(void);
   void              operator+=(string item);
   string            operator[](int at);
protected:
   string            m_items[];
  };
Iterator<string>* ConcreteAggregate::CreateIterator(void)
  {
   return new ConcreteIterator<string>(this);
  }
void ConcreteAggregate::operator+=(string item)
  {
   int size=ArraySize(m_items);
   ArrayResize(m_items,size+1);
   m_items[size]=item;
  }
string ConcreteAggregate::operator[](int at)
  {
   return m_items[at];
  }
int ConcreteAggregate::Count()
  {
   return ArraySize(m_items);
  }
```

### Mediator

Another Behavioral design pattern that can be used in setting how objects can interact with each other. This pattern is the Mediator pattern which can be used when we have a set of objects and we need to define an encapsulated object to describe how this set of objects can interact. It allows or applies decoupling also which can be useful and let us vary the interaction of objects independently.

The same as anything that can have benefits and pitfalls, here are the following pitfalls for the Mediator design pattern:

- There is one created Mediator for everything.
- It is used with other patterns.

**What does the pattern do?**

As per what we mentioned as an identification for the Mediator pattern we can say that it helps to set the interaction method between objects without mentioning each object in an explicit way. So, it helps apply decoupling between objects. It also may be used as a router and it is used for communication management.

The following is a graph for the Mediator pattern to see its structure:

![Mediator](https://c.mql5.com/2/61/Mediator.png)

![Mediator(2)](https://c.mql5.com/2/61/Mediator_n2w.png)

As we can see in the previous graph of the structure of the pattern we have the following participants for the Mediator pattern:

- **Mediator:** it identifies the interface to communicate with objects of colleagues.
- **ConcreteMediator:** Through coordinating objects of colleagues, it implements the behavior of cooperation. its colleagues are known and can be maintainable by the ConcreteMediator.
- **Colleague classes:** the object's Mediator is known by each class of colleague. In addition to that, each colleague can communicate with its mediator at any time this communication is needed or will communicate with another colleague.

**What does design pattern solve?**

Throughout what we understood till now this pattern can be used to solve or when we have the following:

- We have a set of objects that can communicate with each other after defining the way of that well but the way of this communication is complex.
- We have difficulty reusing an object because it communicates with different other objects.
- We need to customize the behavior that is distributed between classes without creating many subclasses.

So, we can say that:

- It helps to limit subclassing.
- It helps decoupling colleagues.
- It makes object protocols simple.
- The cooperation between objects is abstracted.
- The control is centralized.

**How can we use it in MQL5?**

If we need to know how we can use this pattern in the MQL5 we can do that through the following steps:

Using the interface keyword to create the Colleague interface

```
interface Colleague
  {
   void Send(string message);
  };
```

Using the interface keyword to create the Mediator interface

```
interface Mediator
  {
   void Send(string message,Colleague& colleague);
  };
```

Declare the ConcreteColleague1 class

```
class ConcreteColleague1:public Colleague
  {
protected:
   Mediator*         m_mediator;
public:
                     ConcreteColleague1(Mediator& mediator);
   void              Notify(string message);
   void              Send(string message);
  };
ConcreteColleague1::ConcreteColleague1(Mediator& meditor):
   m_mediator(&meditor)
  {
  }
void ConcreteColleague1::Notify(string message)
  {
  }
void ConcreteColleague1::Send(string message)
  {
   m_mediator.Send(message,this);
  }
```

Declare the ConcreteColleague2 class

```
class ConcreteColleague2:public Colleague
  {
protected:
   Mediator*         m_mediator;
public:
                     ConcreteColleague2(Mediator& mediator);
   void              Notify(string message);
   void              Send(string message);
  };
ConcreteColleague2::ConcreteColleague2(Mediator& mediator):
   m_mediator(&mediator)
  {
  }
void ConcreteColleague2::Notify(string message)
  {
  }
void ConcreteColleague2::Send(string message)
  {
   m_mediator.Send(message,this);
  }
```

Declare the ConcreteMediator class

```
class ConcreteMediator:public Mediator
  {
public:
   ConcreteColleague1*  colleague_1;
   ConcreteColleague2*  colleague_2;
   void              Send(string message,Colleague& colleague);
  };
void ConcreteMediator::Send(string message,Colleague& colleague)
  {
   if(colleague_1==&colleague)
      colleague_2.Notify(message);
   else
      colleague_1.Notify(message);
  }
```

So, we can find the full code to use the Mediator design pattern in the MQL5 in one block of code the same as the following

```
//+------------------------------------------------------------------+
//|                                                     Mediator.mqh |
//+------------------------------------------------------------------+
interface Colleague
  {
   void Send(string message);
  };
interface Mediator
  {
   void Send(string message,Colleague& colleague);
  };
class ConcreteColleague1:public Colleague
  {
protected:
   Mediator*         m_mediator;
public:
                     ConcreteColleague1(Mediator& mediator);
   void              Notify(string message);
   void              Send(string message);
  };
ConcreteColleague1::ConcreteColleague1(Mediator& meditor):
   m_mediator(&meditor)
  {
  }
void ConcreteColleague1::Notify(string message)
  {
  }
void ConcreteColleague1::Send(string message)
  {
   m_mediator.Send(message,this);
  }
class ConcreteColleague2:public Colleague
  {
protected:
   Mediator*         m_mediator;
public:
                     ConcreteColleague2(Mediator& mediator);
   void              Notify(string message);
   void              Send(string message);
  };
ConcreteColleague2::ConcreteColleague2(Mediator& mediator):
   m_mediator(&mediator)
  {
  }
void ConcreteColleague2::Notify(string message)
  {
  }
void ConcreteColleague2::Send(string message)
  {
   m_mediator.Send(message,this);
  }
class ConcreteMediator:public Mediator
  {
public:
   ConcreteColleague1*  colleague_1;
   ConcreteColleague2*  colleague_2;
   void              Send(string message,Colleague& colleague);
  };
void ConcreteMediator::Send(string message,Colleague& colleague)
  {
   if(colleague_1==&colleague)
      colleague_2.Notify(message);
   else
      colleague_1.Notify(message);
  }
```

### Conclusion

Now, it is supposed that you got information about the third type of design patterns which is one of the most important topics in programming and software development. We mentioned in this article, some behavioral design patterns and identifying what they are and how they can be useful to create reusable, extended, maintainable, and tested software throughout learning what every pattern can do, problems or issues that can solved by using each pattern, benefits and pitfalls of each patterns, and how we can use each pattern in the MQL5 to create effective trading systems for the MetaTrader 5.

We mentioned the following patterns from the behavioral design patterns:

- Chain of responsibilities.
- Command
- Interpreter
- Iterator
- Mediator

If this is the first article that you read for me about design patterns, I recommend to read my other articles about [Design Patterns in software development and MQL5 (Part I): Creational Patterns](https://www.mql5.com/en/articles/13622) and [Design Patterns in software development and MQL5 (Part 2): Structural Patterns](https://www.mql5.com/en/articles/13724) if you need to learn more about other types of design patterns and I hope that you find them useful.

I recommend also reading more about the design patterns topic as it will help you to create effective software the following are some useful resources about that topic:

- Design Patterns - Elements of Reusable Object-Oriented Software by Eric Gamma, Richard Helm, Ralph Johnson, and John Vlissides
- Design Patterns for Dummies by Steve Holzner
- Head First Design Patterns by Eric Freeman, Elisabeth Robson, Bert Bates, and Kathy Sierra

If you want to read more articles about creating trading systems for the MetaTrader 5 using the most popular technical indicators you can check my other articles about that through my publication page and I hope that you find them useful for your trading to get useful insights and enhance your results or develop your background as a developer to improve projects that you work on.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/13796.zip "Download all attachments in the single ZIP archive")

[Chain\_Of\_Responsibility.mqh](https://www.mql5.com/en/articles/download/13796/chain_of_responsibility.mqh "Download Chain_Of_Responsibility.mqh")(3.52 KB)

[Command.mqh](https://www.mql5.com/en/articles/download/13796/command.mqh "Download Command.mqh")(4.04 KB)

[Interpreter.mqh](https://www.mql5.com/en/articles/download/13796/interpreter.mqh "Download Interpreter.mqh")(4.83 KB)

[Iterator.mqh](https://www.mql5.com/en/articles/download/13796/iterator.mqh "Download Iterator.mqh")(2.31 KB)

[Mediator.mqh](https://www.mql5.com/en/articles/download/13796/mediator.mqh "Download Mediator.mqh")(1.72 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/458517)**
(5)


![Maxim Kuznetsov](https://c.mql5.com/avatar/2016/1/56935A91-AF51.png)

**[Maxim Kuznetsov](https://www.mql5.com/en/users/nektomk)**
\|
4 Apr 2024 at 17:22

after

[![](https://c.mql5.com/3/432/2181988011009__1.png)](https://c.mql5.com/3/432/2181988011009.png "https://c.mql5.com/3/432/2181988011009.png")

you don't have to read any further

![Rashid Umarov](https://c.mql5.com/avatar/2012/5/4FC60566-2EEC.jpg)

**[Rashid Umarov](https://www.mql5.com/en/users/rosh)**
\|
5 Apr 2024 at 08:51

**Maxim Kuznetsov [#](https://www.mql5.com/ru/forum/465147#comment_52943418):**

after

you don't have to read any further

It's a translation from the original English.

![](https://c.mql5.com/3/432/2416257173374.png)

You can mentally replace it with "handler". Or just don't read it and write your own articles.

Article on the topic [https://habr.com/ru/articles/113995/](https://www.mql5.com/go?link=https://habr.com/ru/articles/113995/ "https://habr.com/ru/articles/113995/")

![Maxim Kuznetsov](https://c.mql5.com/avatar/2016/1/56935A91-AF51.png)

**[Maxim Kuznetsov](https://www.mql5.com/en/users/nektomk)**
\|
5 Apr 2024 at 17:45

**Rashid Umarov [#](https://www.mql5.com/ru/forum/465147#comment_52950734):**

This is a translation from the English original.

You can mentally replace it with "handler". Or just don't read it, but write your own articles.

Article on the topic [https://habr.com/ru/articles/113995/](https://www.mql5.com/go?link=https://habr.com/ru/articles/113995/ "https://habr.com/ru/articles/113995/")

it's not about translation... there is almost no text in the article, that's why there is a strict bias towards the code.

does the code from the screenshot have a chance to pass code-review ?

and about "write your own" - you are in the know, I suggested a series about using gcc and msys2 environment, but it turned out that you can't except MSVC

![trampampam](https://c.mql5.com/avatar/avatar_na2.png)

**[trampampam](https://www.mql5.com/en/users/trampampam)**
\|
5 Apr 2024 at 21:21

**Maxim Kuznetsov [#](https://www.mql5.com/ru/forum/465147#comment_52957283):**

What do you think is the "right" thing to do?

![Denis Kirichenko](https://c.mql5.com/avatar/2019/5/5CEDB8D2-7CB7.jpg)

**[Denis Kirichenko](https://www.mql5.com/en/users/denkir)**
\|
22 Apr 2024 at 12:52

```
template<typename T>
void ConcreteIterator::Next(void)
  {
   m_current++;
   if(!IsDone())
     {
     }
  }
```

What isthis even for? Looked at the material on iterators, there are these options:

1)

```
template<typename T>
void ConcreteIterator::Next(void)
  {
   m_current++;
  }
```

2)

```
template<typename T>
void ConcreteIterator::Next(void)
  {
   if(!IsDone())
     {
       m_current++;
     }
  }
```

![Developing a Replay System — Market simulation (Part 18): Ticks and more ticks (II)](https://c.mql5.com/2/56/replay-p18-avatar.png)[Developing a Replay System — Market simulation (Part 18): Ticks and more ticks (II)](https://www.mql5.com/en/articles/11113)

Obviously the current metrics are very far from the ideal time for creating a 1-minute bar. That's the first thing we are going to fix. Fixing the synchronization problem is not difficult. This may seem hard, but it's actually quite simple. We did not make the required correction in the previous article since its purpose was to explain how to transfer the tick data that was used to create the 1-minute bars on the chart into the Market Watch window.

![Data label for time series mining (Part 4)：Interpretability Decomposition Using Label Data](https://c.mql5.com/2/61/Data_label_for_time_series_mining_nPart_45Interpretability_Decomposition_Using_Label_Data_LOGO.png)[Data label for time series mining (Part 4)：Interpretability Decomposition Using Label Data](https://www.mql5.com/en/articles/13218)

This series of articles introduces several time series labeling methods, which can create data that meets most artificial intelligence models, and targeted data labeling according to needs can make the trained artificial intelligence model more in line with the expected design, improve the accuracy of our model, and even help the model make a qualitative leap!

![Modified Grid-Hedge EA in MQL5 (Part I): Making a Simple Hedge EA](https://c.mql5.com/2/62/Modified_Grid-Hedge_EA_in_MQL5_4Part_Ip_Making_a_Simple_Hedge_EA__LOGO.png)[Modified Grid-Hedge EA in MQL5 (Part I): Making a Simple Hedge EA](https://www.mql5.com/en/articles/13845)

We will be creating a simple hedge EA as a base for our more advanced Grid-Hedge EA, which will be a mixture of classic grid and classic hedge strategies. By the end of this article, you will know how to create a simple hedge strategy, and you will also get to know what people say about whether this strategy is truly 100% profitable.

![Market Reactions and Trading Strategies in Response to Dividend Announcements: Evaluating the Efficient Market Hypothesis in Stock Trading](https://c.mql5.com/2/61/Evaluating_the_Efficient_Market_Hypothesis_in_Stock_Trading_LOGO.png)[Market Reactions and Trading Strategies in Response to Dividend Announcements: Evaluating the Efficient Market Hypothesis in Stock Trading](https://www.mql5.com/en/articles/13850)

In this article, we will analyse the impact of dividend announcements on stock market returns and see how investors can earn more returns than those offered by the market when they expect a company to announce dividends. In doing so, we will also check the validity of the Efficient Market Hypothesis in the context of the Indian Stock Market.

[![](https://www.mql5.com/ff/sh/0hvxp984jjj79943z2/6373d9e5710a718ffa6a7d50a5db9dd1.jpg)\\
Web terminal on your iPhone or Android\\
\\
Full-featured MetaTrader 5 platform for any devices and web browsers\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=uyigsjnbfcdvysiynusmriwvhincciwd&s=c95531ae2fd8a81b0fac3def2e4cf820a67584bbf4b02f76ec75f808942dbbd2&uid=&ref=https://www.mql5.com/en/articles/13796&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5068467387117795782)

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