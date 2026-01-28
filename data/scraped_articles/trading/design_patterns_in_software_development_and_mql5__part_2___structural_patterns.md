---
title: Design Patterns in software development and MQL5 (Part 2): Structural Patterns
url: https://www.mql5.com/en/articles/13724
categories: Trading, Trading Systems, Indicators, Expert Advisors
relevance_score: 4
scraped_at: 2026-01-23T17:38:09.735255
---

[![](https://www.mql5.com/ff/sh/x8fwvn495ta7y774z2/01.png)Does your broker offer sponsored hosting for trading?Now it's even easier to get MetaTrader VPS for free – contact your broker for details](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=xscnzeyhifcgygpwvysykhqydcmmbgpp&s=f87b748147e376d34c8f0fdb9737b1766f20cc2174769a0e6b9975b5c2e8ddae&uid=&ref=https://www.mql5.com/en/articles/13724&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5068471385732348367)

MetaTrader 5 / Trading


### Introduction

Welcome to a new article talking about a very essential topic of software development which is Design Patterns by continuing other types of them. We talked about the Creational Design Patterns in a previous article, if you want to learn more about this type you can read [Design Patterns in software development and MQL5 (Part I): Creational Patterns](https://www.mql5.com/en/articles/13622). If you are new to the topic of Design Patterns I recommend to read this mentioned article to learn about the design pattern topic in general and learn how much they are useful in software development.

Design Patterns are essential to learn about them if you want to move your software development skills to the next level, these patterns give you the readymade blueprint to solve specific problems instead of reinventing the wheel but using them to get very practical and testing solutions.

In this article, we will continue by presenting the Structural Design Patterns and learn how they can be very useful in the world of software development to form larger structures by using what we have as classes. The most interesting part of the article is learning how we can use these patterns in the MQL5 programming language to benefit from them and design effective software to be used in the trading field by using the MetaTrader 5 trading terminal.

We will cover the design patterns of the Structural type through the following topics:

- [Structural Patterns](https://www.mql5.com/en/articles/13724#structural)
- [Adapter](https://www.mql5.com/en/articles/13724#adapter)
- [Bridge](https://www.mql5.com/en/articles/13724#bridge)
- [Composite](https://www.mql5.com/en/articles/13724#composite)
- [Decorator](https://www.mql5.com/en/articles/13724#decorator)
- [Facade](https://www.mql5.com/en/articles/13724#facade)
- [Flyweight](https://www.mql5.com/en/articles/13724#flyweight)
- [Proxy](https://www.mql5.com/en/articles/13724#proxy)
- [Conclusion](https://www.mql5.com/en/articles/13724#conclusion)

I hope that you find this article useful to develop your development and programming skills by learning a very interesting topic. It is good to mention also that your knowledge about the topic of Object Oriented Programming (OOP) will help you a lot to understand the Design Patterns topic. if you want to read about this topic you can do that by reading my previous article on this topic [Understanding MQL5 Object-Oriented Programming (OOP)](https://www.mql5.com/en/articles/12813) and I hope that it will be useful in this context.

Disclaimer: All information provided 'as is' only for educational purposes and is not prepared for trading purposes or advice. The information does not guarantee any kind of result. If you choose to use these materials on any of your trading accounts, you will do that at your own risk and you will be the only responsible.

### Structural Patterns

In this part, we will identify what are Structural Design Patterns and their types and structures. Structural patterns are concerned with the method of how classes and objects are structured to be components to build larger structures. These patterns compose interfaces and implementations by using the inheritance concept. This inheritance concept means that we will have a class that has or combines the properties of its parent classes. The need for this pattern will be more significant when we need to make developed classes work together independently.

There are many types of Structural patterns the same as the following:

- Adapter: it helps in getting another interface clients expect by converting the interface of a class.
- Bridge: the abstraction and its implementation can vary independently by decoupling them.
- Composite: for the sake of representation of part-whole hierarchies, it helps to compose objects into tree structures. In addition to that composite allows uniform treatment of individuals and compositions of objects by clients.
- Decorator: it can be used to attach more responsibilities to current objects in a dynamic way and it can be used as a flexible alternative to subclassing providers for the sake of extending functionality.
- Facade: it can be used when we need a unified interface to a set of interfaces in a subsystem and it helps to use the subsystem easily by defining a higher-level interface.
- Flyweight: it helps to effectively support large numbers of fine-grained objects by using sharing.
- Proxy: it can be used when we need to control access to an object by getting an alternative or placeholder for it.

We will cover these patterns through the following approach or by answering the following inquiries:

- What does the pattern do?
- What does design pattern solve?
- How can we use it in MQL5?

### Adapter

In this part, we will start identifying types of structural design patterns by identifying the first type which is the Adapter. The keyword to understand this pattern is adaptability. Simple, if we have an interface that can be used in specific circumstances and then some updates happen with these circumstances that make it is essential to make updates in the interface that let the code adapt and work effectively with these new circumstances. This is what this pattern can do because it converts the interface of the class that we have into another one that can be useable by the clients the same as they expect. So this Adapter pattern allows classes to work together in some cases of incompatible interfaces. This pattern is also known as Wrapper because it seems to give a Wrapper for the interface to adapt and work as another one.

**What does the pattern do?**

As we mentioned this pattern can be used when the designed interface doesn't match the application requirement of the domain-specific interface to convert the interface of this class into another one and allow classes to work together.

The following graphs represent the structure of the Adapter design pattern:

![Adapter1](https://c.mql5.com/2/60/Adapter_1.png)

![Adapter1](https://c.mql5.com/2/60/Adapter_2.png)

As we can see in the previous graphs, based on multiple inheritance support in the programming language, we have the class adapter and the object adapter. We have the target that identifies the domain-specific of the new interface that the client uses, the client that participates with objects adapting to the target interface, the adaptee identifies the existing interface (old one) that we need to make it adaptable, and the adapter that make the adaptee interface adaptable to the target interface.

**What does design pattern solve?**

- Using the existing class with an interface that does not match the interface that we need.
- Creating reusable classes that can work with unrelated classes whether these classes have compatible or incompatible interfaces.
- Adapting the interface of the parent's class when we need to use many existing subclasses.

**How can we use it in MQL5?**

In this part, we will learn how we can use this pattern (AdapterClass and ObjectClass) in the MQL5 programming language and it will be the same as the following:

Using the namespace function to declare the area (AdapterClass) we will define our functions, variables, and classes within

```
namespace AdapterClass
```

Using the interface function to declare (Target) allows determining specific functionality that can be implemented later by the class or to define the domain-specific that clients use

```
interface Target
  {
   void Request();
  };
```

Using the class function to define the Adaptee that defines the existing interface that we need it to be adaptable with one public member (SpecificRequest())

```
class Adaptee
  {
public:
   void              SpecificRequest();
  };
```

Printing a message when the request is executed by the Adaptee

```
void Adaptee::SpecificRequest(void)
  {
   Print("A specific request is executing by the Adaptee");
  }
```

Declaring the Adapter class that adapts the adaptee's interface to the interfaces of the target that inherit from the target and adaptee as multi-inheritance

```
class Adapter;
class AdapterAsTarget:public Target
  {
public:
   Adapter*          asAdaptee;
   void              Request();
  };
void AdapterAsTarget::Request()
  {
   printf("The Adapter requested Operation");
   asAdaptee.SpecificRequest();
  }
class Adapter:public Adaptee
  {
public:
   AdapterAsTarget*  asTarget;
                     Adapter();
                    ~Adapter();
  };
void Adapter::Adapter(void)
  {
   asTarget=new AdapterAsTarget;
   asTarget.asAdaptee=&this;
  }
void Adapter::~Adapter(void)
  {
   delete asTarget;
  }
```

Declaring the Client class

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

Running the Client

```
void Client::Run()
  {
   Adapter adapter;
   Target* target=adapter.asTarget;
   target.Request();
  }
```

So the following is the full code of the Adapter class in the MQL5 in one block of code

```
namespace AdapterClass
{
interface Target
  {
   void Request();
  };
class Adaptee
  {
public:
   void              SpecificRequest();
  };
void Adaptee::SpecificRequest(void)
  {
   Print("A specific request is executing by the Adaptee");
  }
class Adapter;
class AdapterAsTarget:public Target
  {
public:
   Adapter*          asAdaptee;
   void              Request();
  };
void AdapterAsTarget::Request()
  {
   printf("The Adapter requested Operation");
   asAdaptee.SpecificRequest();
  }
class Adapter:public Adaptee
  {
public:
   AdapterAsTarget*  asTarget;
                     Adapter();
                    ~Adapter();
  };
void Adapter::Adapter(void)
  {
   asTarget=new AdapterAsTarget;
   asTarget.asAdaptee=&this;
  }
void Adapter::~Adapter(void)
  {
   delete asTarget;
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
void Client::Run()
  {
   Adapter adapter;
   Target* target=adapter.asTarget;
   target.Request();
  }
}
```

The following is for how we can use the object adapter in MQL5:

Using the namespace to create a declaration area for functions, variables, and classes of the AdapterObject

```
namespace AdapterObject
```

Using the interface to define the target which is the domain-specific that clients use

```
interface Target
  {
   void Request();
  };
```

Creating the Adaptee class to define the existing interface that we need to be adaptable

```
class Adaptee
  {
public:
   void              SpecificRequest();
  };
void Adaptee::SpecificRequest(void)
  {
   Print("The specific Request");
  }
class Adapter:public Target
  {
public:
   void              Request();
protected:
   Adaptee           adaptee;
  };
void Adapter::Request(void)
  {
   Print("The request of Operation requested");
   adaptee.SpecificRequest();
  }
```

Declaring the Client

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

Running the Client when clients call operations on the instance of the Adapter

```
void Client::Run()
  {
   Target* target=new Adapter;
   target.Request();
   delete target;
  }
```

The following is the full code in one block:

```
namespace AdapterObject
{
interface Target
  {
   void Request();
  };
class Adaptee
  {
public:
   void              SpecificRequest();
  };
void Adaptee::SpecificRequest(void)
  {
   Print("The specific Request");
  }
class Adapter:public Target
  {
public:
   void              Request();
protected:
   Adaptee           adaptee;
  };
void Adapter::Request(void)
  {
   Print("The request of Operation requested");
   adaptee.SpecificRequest();
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
void Client::Run()
  {
   Target* target=new Adapter;
   target.Request();
   delete target;
  }
}
```

### Bridge

In this part, we will identify the Bridge design pattern as one of the structural patterns. The main idea of using this pattern is when we need to decouple the abstraction from its implementations to avoid any future conflicts that may happen if there are updates or changes in one of them. It is also known as Handle or Body.

**What does the pattern do?**

As we mentioned the Bridge pattern can be used when we have an abstraction that has many possible implementations and instead of using the usual method of inheritance which links the implementation to the abstraction always we can use this pattern to decouple the abstraction from its implementations to avoid any issues in case of changes or updates. This can be very useful to create a clean code that can be reusable, extendable, and easily tested.

The following is a graph for the Bridge design pattern:

![Bridge](https://c.mql5.com/2/60/Bridge.png)

As we can see through the previous graph for the Bridge pattern structure we have the following participants:

- Abstraction: that defines an interface of the abstraction and maintains a reference to implementor type object.
- RefinedAbstraction: that extends the abstraction interface.
- Implementor: that identifies implementation classes interface.
- ConcreteImplementor: that implements the interface of the implementor and identifies the concrete implementation of this interface.

**What does design pattern solve?**

This Bridge pattern can be used when we need the following as examples:

- Avoiding the continuous linking between the abstraction and its implementation because this pattern helps to decouple them.
- Combining different abstractions and implementations and extending each one independently without any conflict.
- Avoiding impact on clients when there are changes in the implementation of the abstraction.
- Hiding the implementation of the abstraction from clients completely in C++.

**How can we use it in MQL5?**

In this part, we will identify how can use this pattern in the MQL5 programming language to benefit from its useful benefits to create effective software. The following is for how we can code the structure of the Bridge pattern in the MQL5:

Creating the declaring area to define variables, functions, and classes of the pattern

```
namespace Bridge
```

Creating the Implementor interface by using the interface keyword to allow determining functionality to be implemented by the class

```
interface Implementor
  {
   void OperationImp();
  };
```

Creating the class of Abstraction with public and protected members as a participant and maintaining a reference to the object of the implementor

```
class Abstraction
  {
public:
   virtual void      Operation();
                     Abstraction(Implementor*);
                     Abstraction();
                    ~Abstraction();
protected:
   Implementor*      implementor;
  };
void Abstraction::Abstraction(void) {}
void Abstraction::Abstraction(Implementor*i):implementor(i) {}
void Abstraction::~Abstraction()
  {
   delete implementor;
  }
void Abstraction::Operation()
  {
   implementor.OperationImp();
  }
```

Creating the class of RefinedAbstraction as a participant

```
class RefinedAbstraction:public Abstraction
  {
public:
                     RefinedAbstraction(Implementor*);
   void              Operation();
  };
void RefinedAbstraction::RefinedAbstraction(Implementor*i):Abstraction(i) {}
void RefinedAbstraction::Operation()
  {
   Abstraction::Operation();
  }
```

Creating classes of ConcreteImplementorA and B

```
class ConcreteImplementorA:public Implementor
  {
public:
   void              OperationImp();
  };
void ConcreteImplementorA::OperationImp(void)
  {
   Print("The implementor A");
  }
class ConcreteImplementorB:public Implementor
  {
public:
   void              OperationImp();
  };
void ConcreteImplementorB::OperationImp(void)
  {
   Print("The implementor B");
  }
```

Creating the Client class

```
class Client
  {
public:
   string            Output();
   void              Run();
  };
string Client::Output(void)
  {
   return __FUNCTION__;
  }
```

Running the Client

```
void Client::Run(void)
  {
   Abstraction* abstraction;
   abstraction=new RefinedAbstraction(new ConcreteImplementorA);
   abstraction.Operation();
   delete abstraction;
   abstraction=new RefinedAbstraction(new ConcreteImplementorB);
   abstraction.Operation();
   delete abstraction;
  }
```

So, the following code is the full code of the Bridge pattern structure

```
namespace Bridge
{
interface Implementor
  {
   void OperationImp();
  };
class Abstraction
  {
public:
   virtual void      Operation();
                     Abstraction(Implementor*);
                     Abstraction();
                    ~Abstraction();
protected:
   Implementor*      implementor;
  };
void Abstraction::Abstraction(void) {}
void Abstraction::Abstraction(Implementor*i):implementor(i) {}
void Abstraction::~Abstraction()
  {
   delete implementor;
  }
void Abstraction::Operation()
  {
   implementor.OperationImp();
  }
class RefinedAbstraction:public Abstraction
  {
public:
                     RefinedAbstraction(Implementor*);
   void              Operation();
  };
void RefinedAbstraction::RefinedAbstraction(Implementor*i):Abstraction(i) {}
void RefinedAbstraction::Operation()
  {
   Abstraction::Operation();
  }
class ConcreteImplementorA:public Implementor
  {
public:
   void              OperationImp();
  };
void ConcreteImplementorA::OperationImp(void)
  {
   Print("The implementor A");
  }
class ConcreteImplementorB:public Implementor
  {
public:
   void              OperationImp();
  };
void ConcreteImplementorB::OperationImp(void)
  {
   Print("The implementor B");
  }
class Client
  {
public:
   string            Output();
   void              Run();
  };
string Client::Output(void)
  {
   return __FUNCTION__;
  }
void Client::Run(void)
  {
   Abstraction* abstraction;
   abstraction=new RefinedAbstraction(new ConcreteImplementorA);
   abstraction.Operation();
   delete abstraction;
   abstraction=new RefinedAbstraction(new ConcreteImplementorB);
   abstraction.Operation();
   delete abstraction;
  }
}
```

### Composite

In this part, we will identify another structural pattern which is the Composite pattern. this pattern helps to compose objects into a tree as a structure and it allows for uniform treatment from clients for individual objects and compositions.

**What does the pattern do?**

As we mentioned this Composite pattern depends on that we need to compose objects into tree structures and the tree is the main key in this pattern. So, if we have a component we can find as per the tree structure that there are two things under this component that we have Leaf which has operation only and the other thing is Composite which has more operations like adding, removing, and calling child.

The following is the graph to see what the Composite design pattern looks like:

![Composite](https://c.mql5.com/2/60/Composite.png)

As we can see in the previous graph we have the following participants:

- Component: it declares the objects' interface, and implements the interface's default behavior to classes, for accessing and managing the interface's components that are declared for that.
- Leaf: it represents objects of the leaf in a composition and this leaf has no Childs, identifies the behavior of objects that can be considered as primitive in the composition.
- Composite: it identifies the behavior of components with Childs, stores these Childs of components, and implements operations of Childs in the interface of components.
- Client: through the component interface the client manipulates objects.

**What does design pattern solve?**

This Composite pattern can be used when we need:

- Part-whole hierarchies of object representation.
- All objects in the composite will be treated uniformly by the client.

**How can we use it in MQL5?**

In this part we will present how we can code the Composite pattern in MQL5 it will be the same as the following:

Create the Composite space or area to declare all functions, variables, and classes by using the namespace keyword

```
namespace Composite
```

Creating the Component class with public and protected members and accessing the component's parent

```
class Component
  {
public:
   virtual void      Operation(void)=0;
   virtual void      Add(Component*)=0;
   virtual void      Remove(Component*)=0;
   virtual Component*   GetChild(int)=0;
                     Component(void);
                     Component(string);
protected:
   string            name;
  };
Component::Component(void) {}
Component::Component(string a_name):name(a_name) {}
```

Defining a user error of adding and removing to leaf and creating the Leaf class

```
#define ERR_INVALID_OPERATION_EXCEPTION   1
class Leaf:public Component
  {
public:
   void              Operation(void);
   void              Add(Component*);
   void              Remove(Component*);
   Component*        GetChild(int);
                     Leaf(string);
  };
void Leaf::Leaf(string a_name):Component(a_name) {}
void Leaf::Operation(void)
  {
   Print(name);
  }
void Leaf::Add(Component*)
  {
   SetUserError(ERR_INVALID_OPERATION_EXCEPTION);
  }
void Leaf::Remove(Component*)
  {
   SetUserError(ERR_INVALID_OPERATION_EXCEPTION);
  }
Component* Leaf::GetChild(int)
  {
   SetUserError(ERR_INVALID_OPERATION_EXCEPTION);
   return NULL;
  }
```

Creating the Composite class as a participant then operation, component adding, removing, and GetChild(int)

```
class Composite:public Component
  {
public:
   void              Operation(void);
   void              Add(Component*);
   void              Remove(Component*);
   Component*        GetChild(int);
                     Composite(string);
                    ~Composite(void);
protected:
   Component*        nodes[];
  };
Composite::Composite(string a_name):Component(a_name) {}
Composite::~Composite(void)
  {
   int total=ArraySize(nodes);
   for(int i=0; i<total; i++)
     {
      Component* i_node=nodes[i];
      if(CheckPointer(i_node)==1)
        {
         delete i_node;
        }
     }
  }
void Composite::Operation(void)
  {
   Print(name);
   int total=ArraySize(nodes);
   for(int i=0; i<total; i++)
     {
      nodes[i].Operation();
     }
  }
void Composite::Add(Component *src)
  {
   int size=ArraySize(nodes);
   ArrayResize(nodes,size+1);
   nodes[size]=src;
  }
void Composite::Remove(Component *src)
  {
   int find=-1;
   int total=ArraySize(nodes);
   for(int i=0; i<total; i++)
     {
      if(nodes[i]==src)
        {
         find=i;
         break;
        }
     }
   if(find>-1)
     {
      ArrayRemove(nodes,find,1);
     }
  }
Component* Composite::GetChild(int i)
  {
   return nodes[i];
  }
```

Creating the Client class as a participant

```
class Client
  {
public:
   string            Output(void);
   void              Run(void);
  };
string Client::Output(void) {return __FUNCTION__;}
```

Running the Client

```
void Client::Run(void)
  {
   Component* root=new Composite("root");
   Component* branch1=new Composite("The branch 1");
   Component* branch2=new Composite("The branch 2");
   Component* leaf1=new Leaf("The leaf 1");
   Component* leaf2=new Leaf("The leaf 2");
   root.Add(branch1);
   root.Add(branch2);
   branch1.Add(leaf1);
   branch1.Add(leaf2);
   branch2.Add(leaf2);
   branch2.Add(new Leaf("The leaf 3"));
   Print("The tree");
   root.Operation();
   root.Remove(branch1);
   Print("Removing one branch");
   root.Operation();
   delete root;
   delete branch1;
  }
```

### Decorator

The Decorator pattern is another structural design pattern that can be used to form larger structures for created or existing objects. This pattern can be used to add additional features, behaviors, or responsibilities to the object in a dynamic method at run-time because it can provide an alternative to subclassing flexibly. it is also known as Wrapper.

**What does the pattern do?**

As we said this pattern will help us to add responsibilities to any individual object without doing that in the entire class as a Wrapper instead of using the subclassing way.

The following graph is for the structure of the Decorator design pattern:

![Decorator](https://c.mql5.com/2/60/Decorator.png)

As we can see in the previous graph we have the following as participants:

- Component: it identifies the objects' interface and that they have additional roles to them in a dynamic way.
- ConcreteComponent: it identifies which is the object that can attach additional responsibilities to it.
- Decorator: it helps to maintain a reference to the object of the component and identify the interface that fits the component's interface
- ConcreteDecorator: it is responsible for adding responsibilities to the component.

**What does design pattern solve?**

This Decorator design pattern can be used when we need:

- Adding additional responsibilities to individual objects dynamically and transparently without an impact on other objects.
- Withdrawing responsibilities from objects.
- Finding the subclassing method is impractical in the case of extension.

**How can we use it in MQL5?**

If we need to code this Decorator pattern in the MQL5 to use it in the created software the following will be steps to do that:

Creating the area of declaring our Decorator to declare all that we need within

```
namespace Decorator
```

Creating the Component class with a public member to define the interface of objects

```
class Component
  {
public:
   virtual void      Operation(void)=0;
  };
```

Creating the Decorator class as a participant

```
class Decorator:public Component
  {
public:
   Component*        component;
   void              Operation(void);
  };
void Decorator::Operation(void)
  {
   if(CheckPointer(component)>0)
     {
      component.Operation();
     }
  }
```

Creating the ConcreteComponent class as a participant

```
class ConcreteComponent:public Component
  {
public:
   void              Operation(void);
  };
void ConcreteComponent::Operation(void)
  {
   Print("The concrete operation");
  }
```

Creating the ConcreteDecoratorA and B

```
class ConcreteDecoratorA:public Decorator
  {
protected:
   string            added_state;
public:
                     ConcreteDecoratorA(void);
   void              Operation(void);
  };
ConcreteDecoratorA::ConcreteDecoratorA(void):
   added_state("The added state()")
  {
  }
void ConcreteDecoratorA::Operation(void)
  {
   Decorator::Operation();
   Print(added_state);
  }
class ConcreteDecoratorB:public Decorator
  {
public:
   void              AddedBehavior(void);
   void              Operation(void);
  };
void ConcreteDecoratorB::AddedBehavior(void)
  {
   Print("The added behavior()");
  }
void ConcreteDecoratorB::Operation(void)
  {
   Decorator::Operation();
   AddedBehavior();
  }
```

Creating the Client class

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
```

Running the Client

```
void Client::Run(void)
  {
   Component* component=new ConcreteComponent();
   Decorator* decorator_a=new ConcreteDecoratorA();
   Decorator* decorator_b=new ConcreteDecoratorB();
   decorator_a.component=component;
   decorator_b.component=decorator_a;
   decorator_b.Operation();
   delete component;
   delete decorator_a;
   delete decorator_b;
  }
```

So, if we need to see the full code in one block of code we can see that the same as the following

```
namespace Decorator
{
class Component
  {
public:
   virtual void      Operation(void)=0;
  };
class Decorator:public Component
  {
public:
   Component*        component;
   void              Operation(void);
  };
void Decorator::Operation(void)
  {
   if(CheckPointer(component)>0)
     {
      component.Operation();
     }
  }
class ConcreteComponent:public Component
  {
public:
   void              Operation(void);
  };
void ConcreteComponent::Operation(void)
  {
   Print("The concrete operation");
  }
class ConcreteDecoratorA:public Decorator
  {
protected:
   string            added_state;
public:
                     ConcreteDecoratorA(void);
   void              Operation(void);
  };
ConcreteDecoratorA::ConcreteDecoratorA(void):
   added_state("The added state()")
  {
  }
void ConcreteDecoratorA::Operation(void)
  {
   Decorator::Operation();
   Print(added_state);
  }
class ConcreteDecoratorB:public Decorator
  {
public:
   void              AddedBehavior(void);
   void              Operation(void);
  };
void ConcreteDecoratorB::AddedBehavior(void)
  {
   Print("The added behavior()");
  }
void ConcreteDecoratorB::Operation(void)
  {
   Decorator::Operation();
   AddedBehavior();
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
   Component* component=new ConcreteComponent();
   Decorator* decorator_a=new ConcreteDecoratorA();
   Decorator* decorator_b=new ConcreteDecoratorB();
   decorator_a.component=component;
   decorator_b.component=decorator_a;
   decorator_b.Operation();
   delete component;
   delete decorator_a;
   delete decorator_b;
  }
}
```

### Facade

The Facade is another structural pattern that can be used in software development to create other larger structures. It identifies an interface with a higher level to make use of subsystems smoother and easier.

**What does the pattern do?**

As we mentioned the Facade is a way to decouple the Client from the complexity of the subsystem because it provides a unified interface for a set of subsystem interfaces. So, the client will interact with this unified interface to get what he requests but this interface will interact with the subsystem to return what the client requested.

If we need to see the structure of the Facade design pattern we can find it the same as the following graph:

![Facade](https://c.mql5.com/2/60/Facade.png)

As we can see in the previous graph there are the following participants in this pattern:

- Facade: it knows which subsystem can request, and delegates requests of the client to the suitable objects of the subsystem.
- Subsystem classes: they perform functions of the subsystem, when receiving the request from the Facade they handle it, they have no references to the Facade.

**What does design pattern solve?**

This Facade pattern can be used when we need:

- Simplifying the complexity of the subsystem by providing a simple interface.
- Decoupling the subsystem from clients and other subsystems to change the existing dependencies between clients and implementations of classes of the abstraction to the subsystem independence and portability.
- Defining entry points to every subsystem level by layering them.

**How can we use it in MQL5?**

In this part we will provide the code to use the Facade pattern in the MQL5 and the following are steps to do that:

Creating the space of Facade to declare what we need within by using the namespace

```
namespace Facade
```

Declaring the SubSystemA, SubSystemB, and SubSystemC classes

```
class SubSystemA
  {
public:
   void              Operation(void);
  };
void SubSystemA::Operation(void)
  {
   Print("The operation of the subsystem A");
  }
class SubSystemB
  {
public:
   void              Operation(void);
  };
void SubSystemB::Operation(void)
  {
   Print("The operation of the subsystem B");
  }
class SubSystemC
  {
public:
   void              Operation(void);
  };
void SubSystemC::Operation(void)
  {
   Print("The operation of the subsystem C");
  }
```

Declaring the Facade class

```
class Facade
  {
public:
   void              Operation_A_B(void);
   void              Operation_B_C(void);
protected:
   SubSystemA        subsystem_a;
   SubSystemB        subsystem_b;
   SubSystemC        subsystem_c;
  };
void Facade::Operation_A_B(void)
  {
   Print("The facade of the operation of A & B");
   Print("The request of the facade of the subsystem A operation");
   subsystem_a.Operation();
   Print("The request of the facade of the subsystem B operation");
   subsystem_b.Operation();
  }
void Facade::Operation_B_C(void)
  {
   Print("The facade of the operation of B & C");
   Print("The request of the facade of the subsystem B operation");
   subsystem_b.Operation();
   Print("The request of the facade of the subsystem C operation");
   subsystem_c.Operation();
  }
```

Declaring the Client

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
```

Running the Client

```
void Client::Run(void)
  {
   Facade facade;
   Print("The request of client of the facade operation A & B");
   facade.Operation_A_B();
   Print("The request of client of the facade operation B & C");
   facade.Operation_B_C();
  }
```

So, the following is the full code in one block of code

```
namespace Facade
{
class SubSystemA
  {
public:
   void              Operation(void);
  };
void SubSystemA::Operation(void)
  {
   Print("The operation of the subsystem A");
  }
class SubSystemB
  {
public:
   void              Operation(void);
  };
void SubSystemB::Operation(void)
  {
   Print("The operation of the subsystem B");
  }
class SubSystemC
  {
public:
   void              Operation(void);
  };
void SubSystemC::Operation(void)
  {
   Print("The operation of the subsystem C");
  }
class Facade
  {
public:
   void              Operation_A_B(void);
   void              Operation_B_C(void);
protected:
   SubSystemA        subsystem_a;
   SubSystemB        subsystem_b;
   SubSystemC        subsystem_c;
  };
void Facade::Operation_A_B(void)
  {
   Print("The facade of the operation of A & B");
   Print("The request of the facade of the subsystem A operation");
   subsystem_a.Operation();
   Print("The request of the facade of the subsystem B operation");
   subsystem_b.Operation();
  }
void Facade::Operation_B_C(void)
  {
   Print("The facade of the operation of B & C");
   Print("The request of the facade of the subsystem B operation");
   subsystem_b.Operation();
   Print("The request of the facade of the subsystem C operation");
   subsystem_c.Operation();
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
   Facade facade;
   Print("The request of client of the facade operation A & B");
   facade.Operation_A_B();
   Print("The request of client of the facade operation B & C");
   facade.Operation_B_C();
  }
}
```

### Flyweight

The Flyweight structural pattern is another pattern that can be useful when there are large numbers of fine-grained objects as it uses sharing in this case to support that.

**What does the pattern do?**

As we mentioned about this pattern by using sharing as a support this can be helpful also in terms of memory and this is the reason that it is named Flyweight.

The following are graphs of the Flyweights design pattern structure:

![Flyweight1](https://c.mql5.com/2/60/Flyweight_1.png)

![Flyweight2](https://c.mql5.com/2/60/Flyweight_2.png)

As we can see in the previous graph we have the following participants:

- Flyweight.
- ConcreteFlyweight.
- UnsharedConcreteFlyweight.
- FlyweightFactory.
- Client.

**What does design pattern solve?**

This pattern can be used when:

- A large number of objects are used in an application.
- We need to cut the expensive cost of storage.
- If most of the object state can be made extrinsic.
- If we remove the extrinsic state many groups of objects may be replaced by fewer shared objects relatively.
- The identity of the object is not so important for the application in terms of dependency.

**How can we use it in MQL5?**

In case of desire to code this pattern in the MQL5 we will create the code the same as the following:

Creating our Flyweight space by using the namespace keyword to declare all that we need with

```
namespace Flyweight
```

Using the interface keyword to declare the Flyweight

```
interface Flyweight;
```

Creating the Pair class with protected and public members as participant

```
class Pair
  {
protected:
   string            key;
   Flyweight*        value;
public:
                     Pair(void);
                     Pair(string,Flyweight*);
                    ~Pair(void);
   Flyweight*        Value(void);
   string            Key(void);
  };
Pair::Pair(void){}
Pair::Pair(string a_key,Flyweight *a_value):
   key(a_key),
   value(a_value){}
Pair::~Pair(void)
  {
   delete value;
  }
string Pair::Key(void)
  {
   return key;
  }
Flyweight* Pair::Value(void)
  {
   return value;
  }
```

Creating the Reference class and defining its constructor and deconstructor

```
class Reference
  {
protected:
   Pair*             pairs[];
public:
                     Reference(void);
                    ~Reference(void);
   void              Add(string,Flyweight*);
   bool              Has(string);
   Flyweight*        operator[](string);
protected:
   int               Find(string);
  };
Reference::Reference(void){}
Reference::~Reference(void)
  {
   int total=ArraySize(pairs);
   for(int i=0; i<total; i++)
     {
      Pair* ipair=pairs[i];
      if(CheckPointer(ipair))
        {
         delete ipair;
        }
     }
  }
int Reference::Find(string key)
  {
   int total=ArraySize(pairs);
   for(int i=0; i<total; i++)
     {
      Pair* ipair=pairs[i];
      if(ipair.Key()==key)
        {
         return i;
        }
     }
   return -1;
  }
bool Reference::Has(string key)
  {
   return (Find(key)>-1)?true:false;
  }
void Reference::Add(string key,Flyweight *value)
  {
   int size=ArraySize(pairs);
   ArrayResize(pairs,size+1);
   pairs[size]=new Pair(key,value);
  }
Flyweight* Reference::operator[](string key)
  {
   int find=Find(key);
   return (find>-1)?pairs[find].Value():NULL;
  }
```

Declaring the Flyweight interface to act on the extrinsic state

```
interface Flyweight
  {
   void Operation(int extrinsic_state);
  };
```

Declaring the ConcreteFlyweight class

```
class ConcreteFlyweight:public Flyweight
  {
public:
   void              Operation(int extrinsic_state);
protected:
   int               intrinsic_state;
  };
void ConcreteFlyweight::Operation(int extrinsic_state)
  {
   intrinsic_state=extrinsic_state;
   printf("The intrinsic state - %d",intrinsic_state);
  }
```

Declaring the UnsharedConcreteFlyweight class

```
class UnsharedConcreteFlyweight:public Flyweight
  {
protected:
   int               all_state;
public:
   void              Operation(int extrinsic_state);
  };
void UnsharedConcreteFlyweight::Operation(int extrinsic_state)
  {
   all_state=extrinsic_state;
   Print("all state - %d",all_state);
  }
```

Declaring the FlyweightFactory class

```
class FlyweightFactory
  {
protected:
   Reference        pool;
public:
                     FlyweightFactory(void);
   Flyweight*        Flyweight(string key);
  };
FlyweightFactory::FlyweightFactory(void)
  {
   pool.Add("1",new ConcreteFlyweight);
   pool.Add("2",new ConcreteFlyweight);
   pool.Add("3",new ConcreteFlyweight);
  }
Flyweight* FlyweightFactory::Flyweight(string key)
  {
   if(!pool.Has(key))
     {
      pool.Add(key,new ConcreteFlyweight());
     }
   return pool[key];
  }
```

Declaring the Client class

```
class Client
  {
public:
   string            Output();
   void              Run();
  };
string Client::Output(void)
  {
   return __FUNCTION__;
  }
```

Running the Client

```
void Client::Run(void)
  {
   int extrinsic_state=7;
   Flyweight* flyweight;
   FlyweightFactory factory;
   flyweight=factory.Flyweight("1");
   flyweight.Operation(extrinsic_state);
   flyweight=factory.Flyweight("10");
   flyweight.Operation(extrinsic_state);
   flyweight=new UnsharedConcreteFlyweight();
   flyweight.Operation(extrinsic_state);
   delete flyweight;
  }
```

The following is for the full code in one block to code the Flyweight pattern in the MQL5

```
namespace Flyweight
{
interface Flyweight;
class Pair
  {
protected:
   string            key;
   Flyweight*        value;
public:
                     Pair(void);
                     Pair(string,Flyweight*);
                    ~Pair(void);
   Flyweight*        Value(void);
   string            Key(void);
  };
Pair::Pair(void){}
Pair::Pair(string a_key,Flyweight *a_value):
   key(a_key),
   value(a_value){}
Pair::~Pair(void)
  {
   delete value;
  }
string Pair::Key(void)
  {
   return key;
  }
Flyweight* Pair::Value(void)
  {
   return value;
  }
class Reference
  {
protected:
   Pair*             pairs[];
public:
                     Reference(void);
                    ~Reference(void);
   void              Add(string,Flyweight*);
   bool              Has(string);
   Flyweight*        operator[](string);
protected:
   int               Find(string);
  };
Reference::Reference(void){}
Reference::~Reference(void)
  {
   int total=ArraySize(pairs);
   for(int i=0; i<total; i++)
     {
      Pair* ipair=pairs[i];
      if(CheckPointer(ipair))
        {
         delete ipair;
        }
     }
  }
int Reference::Find(string key)
  {
   int total=ArraySize(pairs);
   for(int i=0; i<total; i++)
     {
      Pair* ipair=pairs[i];
      if(ipair.Key()==key)
        {
         return i;
        }
     }
   return -1;
  }
bool Reference::Has(string key)
  {
   return (Find(key)>-1)?true:false;
  }
void Reference::Add(string key,Flyweight *value)
  {
   int size=ArraySize(pairs);
   ArrayResize(pairs,size+1);
   pairs[size]=new Pair(key,value);
  }
Flyweight* Reference::operator[](string key)
  {
   int find=Find(key);
   return (find>-1)?pairs[find].Value():NULL;
  }
interface Flyweight
  {
   void Operation(int extrinsic_state);
  };
class ConcreteFlyweight:public Flyweight
  {
public:
   void              Operation(int extrinsic_state);
protected:
   int               intrinsic_state;
  };
void ConcreteFlyweight::Operation(int extrinsic_state)
  {
   intrinsic_state=extrinsic_state;
   Print("The intrinsic state - %d",intrinsic_state);
  }
class UnsharedConcreteFlyweight:public Flyweight
  {
protected:
   int               all_state;
public:
   void              Operation(int extrinsic_state);
  };
void UnsharedConcreteFlyweight::Operation(int extrinsic_state)
  {
   all_state=extrinsic_state;
   Print("all state - %d",all_state);
  }
class FlyweightFactory
  {
protected:
   Reference        pool;
public:
                     FlyweightFactory(void);
   Flyweight*        Flyweight(string key);
  };
FlyweightFactory::FlyweightFactory(void)
  {
   pool.Add("1",new ConcreteFlyweight);
   pool.Add("2",new ConcreteFlyweight);
   pool.Add("3",new ConcreteFlyweight);
  }
Flyweight* FlyweightFactory::Flyweight(string key)
  {
   if(!pool.Has(key))
     {
      pool.Add(key,new ConcreteFlyweight());
     }
   return pool[key];
  }
class Client
  {
public:
   string            Output();
   void              Run();
  };
string Client::Output(void)
  {
   return __FUNCTION__;
  }
void Client::Run(void)
  {
   int extrinsic_state=7;
   Flyweight* flyweight;
   FlyweightFactory factory;
   flyweight=factory.Flyweight("1");
   flyweight.Operation(extrinsic_state);
   flyweight=factory.Flyweight("10");
   flyweight.Operation(extrinsic_state);
   flyweight=new UnsharedConcreteFlyweight();
   flyweight.Operation(extrinsic_state);
   delete flyweight;
  }
}
```

### Proxy

Now, we will identify the last structural design pattern which is the proxy. This pattern has many types in terms of representative, but in general we can say that The Proxy can be used to presenting an alternative or placeholder for another object to complete control in terms of access to this object. It is also known as a Surrogate.

**What does the pattern do?**

This pattern the same as we mentioned provides the surrogate to control the access to an object.

Now, the following graph is for the structure of the Proxy design pattern:

![Proxy](https://c.mql5.com/2/60/Proxy.png)

As we can see in the previous graph we have the following participants:

- Proxy.
- Subject.
- Real subject.

**What does design pattern solve?**

The following situations are common in which we can use the Proxy:

- If we need a local representative for an object in a different address space, we can use the remote proxy that provides that.
- If we need an expensive object on demand we can use the virtual proxy that creates these objects.
- If we need to control access to the primary or original object we can use the protection proxy that can do that.
- if we need a replacement for a bare pointer we can use the smart reference.

**How can we use it in MQL5?**

If we need to code the Proxy pattern in the MQL5 to create effective software we can find that through the following steps

Declaring the space of Proxy to declare all we need within it in terms of variables, functions, classes, ... etc.

```
namespace Proxy
```

Declaring the Subject class as a participant

```
class Subject
  {
public:
   virtual void      Request(void)=0;
  };
```

Creating the RealSubject class

```
class RealSubject:public Subject
  {
public:
   void              Request(void);
  };
void RealSubject::Request(void)
  {
   Print("The real subject");
  }
```

Creating the Proxy class as a participant

```
class Proxy:public Subject
  {
protected:
   RealSubject*      real_subject;
public:
                    ~Proxy(void);
   void              Request(void);
  };
Proxy::~Proxy(void)
  {
   delete real_subject;
  }
void Proxy::Request(void)
  {
   if(!CheckPointer(real_subject))
     {
      real_subject=new RealSubject;
     }
   real_subject.Request();
  }
```

Declaring the Client class

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
```

Running the Client

```
void Client::Run(void)
  {
   Subject* subject=new Proxy;
   subject.Request();
   delete subject;
  }
```

The following is the full code in one block of code

```
namespace Proxy
{
class Subject
  {
public:
   virtual void      Request(void)=0;
  };
class RealSubject:public Subject
  {
public:
   void              Request(void);
  };
void RealSubject::Request(void)
  {
   Print("The real subject");
  }
class Proxy:public Subject
  {
protected:
   RealSubject*      real_subject;
public:
                    ~Proxy(void);
   void              Request(void);
  };
Proxy::~Proxy(void)
  {
   delete real_subject;
  }
void Proxy::Request(void)
  {
   if(!CheckPointer(real_subject))
     {
      real_subject=new RealSubject;
     }
   real_subject.Request();
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
   Subject* subject=new Proxy;
   subject.Request();
   delete subject;
  }
}
```

### Conclusion

At the end of this article, we provided a simple introduction and information about the structural design patterns topic. Through this article, we identified each type of structural pattern to understand how we can write clean code which can be reusable, extendable, and easily tested by understanding each type of these patterns deeply by identifying what is the pattern, what does the pattern do, what is the structure of it, what the pattern solves as design issues.

So, we identified the following structural design patterns:

- Adapter
- Bridge
- Composite
- Decorator
- Facade
- Flyweight
- Proxy

As we said before in part one, the design pattern is very important to learn as a software developer because it will save a lot of time and let you avoid reinventing the wheel by using predetermined, tested, and practical solutions for specific issues and problems and your knowledge about the Object-Oriented-Programming is very helpful as a topic to understand Design Patterns topic.

I recommend reading more about this important topic and I recommend the following resources to learn more:

- Design Patterns - Elements of Reusable Object-Oriented Software by Eric Gamma, Richard Helm, Ralph Johnson, and John Vlissides
- Design Patterns for Dummies by Steve Holzner
- Head First Design Patterns by Eric Freeman, Elisabeth Robson, Bert Bates, and Kathy Sierra

I hope that you found this article useful that it added value to your knowledge and awareness in the field of software development and you benefit from it developing more effective software by the MQL5 programming language. If you found this article helpful and got value from it and you need to read more articles for me you can navigate my publication section and you will find many articles about MQL5 programming language and also you will find many articles about how to create trading systems based on the most popular technical indicators like RSI, MACD, Bollinger Bands, Moving averages, Stochastics, and others. I hope to get benefit from them and enhance your knowledge and results in software development and trading.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/13724.zip "Download all attachments in the single ZIP archive")

[Adapter\_Class.mqh](https://www.mql5.com/en/articles/download/13724/adapter_class.mqh "Download Adapter_Class.mqh")(1.07 KB)

[Adapter\_Object.mqh](https://www.mql5.com/en/articles/download/13724/adapter_object.mqh "Download Adapter_Object.mqh")(0.72 KB)

[Bridge.mqh](https://www.mql5.com/en/articles/download/13724/bridge.mqh "Download Bridge.mqh")(3.38 KB)

[Composite.mqh](https://www.mql5.com/en/articles/download/13724/composite.mqh "Download Composite.mqh")(3.09 KB)

[Decorator.mqh](https://www.mql5.com/en/articles/download/13724/decorator.mqh "Download Decorator.mqh")(3.53 KB)

[Facade.mqh](https://www.mql5.com/en/articles/download/13724/facade.mqh "Download Facade.mqh")(3.43 KB)

[Flyweight.mqh](https://www.mql5.com/en/articles/download/13724/flyweight.mqh "Download Flyweight.mqh")(3.47 KB)

[Proxy.mqh](https://www.mql5.com/en/articles/download/13724/proxy.mqh "Download Proxy.mqh")(1.79 KB)

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

**[Go to discussion](https://www.mql5.com/en/forum/457846)**

![How to create a simple Multi-Currency Expert Advisor using MQL5 (Part 4): Triangular moving average — Indicator Signals](https://c.mql5.com/2/60/rj-article-image-60x60.png)[How to create a simple Multi-Currency Expert Advisor using MQL5 (Part 4): Triangular moving average — Indicator Signals](https://www.mql5.com/en/articles/13770)

The Multi-Currency Expert Advisor in this article is Expert Advisor or trading robot that can trade (open orders, close orders and manage orders for example: Trailing Stop Loss and Trailing Profit) for more than one symbol pair only from one symbol chart. This time we will use only 1 indicator, namely Triangular moving average in multi-timeframes or single timeframe.

![Developing a quality factor for Expert Advisors](https://c.mql5.com/2/55/Desenvolvendo_um_fator_de_qualidade_para_os_EAs_Avatar.png)[Developing a quality factor for Expert Advisors](https://www.mql5.com/en/articles/11373)

In this article, we will see how to develop a quality score that your Expert Advisor can display in the strategy tester. We will look at two well-known calculation methods – Van Tharp and Sunny Harris.

![Developing a Replay System — Market simulation (Part 14): Birth of the SIMULATOR (IV)](https://c.mql5.com/2/55/Desenvolvendo_um_sistema_de_Replay_Parte_14_avatar.png)[Developing a Replay System — Market simulation (Part 14): Birth of the SIMULATOR (IV)](https://www.mql5.com/en/articles/11058)

In this article we will continue the simulator development stage. this time we will see how to effectively create a RANDOM WALK type movement. This type of movement is very intriguing because it forms the basis of everything that happens in the capital market. In addition, we will begin to understand some concepts that are fundamental to those conducting market analysis.

![Combinatorially Symmetric Cross Validation In MQL5](https://c.mql5.com/2/60/aticleicon.png)[Combinatorially Symmetric Cross Validation In MQL5](https://www.mql5.com/en/articles/13743)

In this article we present the implementation of Combinatorially Symmetric Cross Validation in pure MQL5, to measure the degree to which a overfitting may occure after optimizing a strategy using the slow complete algorithm of the Strategy Tester.

[What's wrong with regular VPS?Here are the 8 most common problems that algorithmic traders may encounterRead![](https://www.mql5.com/ff/sh/hzatb686qjqxwtr4z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/450486&a=drhremihlwuaqyvgpzfddbtmgciejpba&s=c37d25bcceb93ed153b814e6ba4d4839461a9b2d68dd82b95b142be06d310f3f&uid=&ref=https://www.mql5.com/en/articles/13724&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5068471385732348367)

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