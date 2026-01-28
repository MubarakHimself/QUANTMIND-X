---
title: Developing a multi-currency Expert Advisor (Part 10): Creating objects from a string
url: https://www.mql5.com/en/articles/14739
categories: Trading, Trading Systems, Integration, Expert Advisors
relevance_score: 12
scraped_at: 2026-01-22T17:12:14.637732
---

[![](https://www.mql5.com/ff/sh/jup0jccfs9655z9z2/01.png)Learn to create your own robotsRead our book "MQL5 Programming for Traders"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/book%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.algobook%26utm_content=visit.page%26utm_campaign=algobook.promo.04.2024&a=rsxjstxkzbrlgjjrxaglpezpvrjflnvw&s=7224440013c3dbc50ba9cc078cd015fabca36df446b8e75028d6b30234663872&uid=&ref=https://www.mql5.com/en/articles/14739&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5048964215504478780)

MetaTrader 5 / Trading


### Introduction

In the previous [article](https://www.mql5.com/en/articles/14680), I have outlined a general plan for developing the EA, which includes several stages. Each stage generates a certain amount of information to be used in the stages that follow. I decided to save this information in a database and created a table in it, in which we can place the results of single passes of the strategy tester for various EAs.

In order to be able to use this information in the next steps, we need to have some way of creating the necessary objects (trading strategies, their groups and EAs) from the information stored in the database. There is no option to save objects directly to the database. The best thing that can be suggested is to convert all the properties of objects into a string, save it in the database, then read this string from the database and create the required object from it.

Creating an object from a string can be implemented in different ways. For example, we can create an object of the desired class with default parameters, and then use a special method or function to parse the string read from the database and assign the corresponding values to the object properties. Alternatively, we can create an additional object constructor that will accept only one string as an input. This string will be parsed into parts inside the constructor and the corresponding values will be assigned to the object properties there. To understand which option is better, let's first look at how we store information about objects in the database.

### Storing information about objects

Let's open the table in the database that we filled in the previous article and look at the last columns. The _params_ and _inputs_ columns store the result of converting the _CSimpleVolumesStrategy_ class trading strategy object into a string and the inputs of a single optimization pass.

![](https://c.mql5.com/2/76/4312041235618.png)

Fig. 1. The fragment of the passes table with information about the applied strategy and test parameters

Although they are related, there are differences between them: the _inputs_ column features the names of inputs (although they do not exactly match the names of the strategy object properties), but some parameters, such as the symbol and period, are missing. Therefore, it will be more convenient for us to use the entry form from the _params_ column to recreate the object.

Let's recall where we got the implementation of converting a strategy object into a string. In the [fourth](https://www.mql5.com/en/articles/14246) part of the article series, I implemented saving the EA state to a file so that it can be restored after a restart. To prevent the EA from accidentally using a file featuring data from another similar EA, I implemented saving data about the parameters of all instances of strategies used in this EA to the file.

In other words, the original task was to ensure that instances of trading strategies with different parameters generated different strings. Therefore, I was not particularly concerned about the possibility to create a new trading strategy object based on such strings. In the [ninth](https://www.mql5.com/en/articles/14680) part, I took the existing string conversion mechanism without any additional modification, since my objective was to debug the process of adding such information to the database.

### Moving on to implementation

Now is the time to think about how we can recreate objects using such strings. So, we have a string that looks something like this:

class CSimpleVolumesStrategy(EURGBP,PERIOD\_H1,17,0.70,0.90,50,10000.00,750.00,10000,3)

If we pass it to the _CSimpleVolumesStrategy_ class object constructor, it should do the following:

- remove the part that comes before the first opening parenthesis;
- split the remaining part up to the closing parenthesis by comma symbols;
- assign each obtained part to the corresponding object properties converting them into numbers if necessary.

Looking at the list of these actions, we can see that the first action can be performed at a higher level. Indeed, if we first take the class name from this line, then we can define the class of the created object. Then it is more convenient for the constructor to pass only that part of the string that is inside the parentheses.

Moreover, the needs for creating objects from a string are not limited to this single class. First, we may not have just one trading strategy. Secondly, we will need to create _CVirtualStrategyGroup_ class objects, that is, groups of several instances of trading strategies with different parameters. This will be useful for the stage of combining several previously selected groups into one group. Third, what prevents us from providing the ability to create an EA object (the _CVirtualAdvisor_ class) from a string? This will allow us to write a universal EA that can load from a file a text description of all groups of strategies that should be used. By changing the description in the file, it will be possible to completely update the composition of the strategies included in it without recompiling the EA.

If we try to imagine what the initialization string of _CVirtualStrategyGroup_ class objects might look like, then we get something like this:

class CVirtualStrategyGroup(\[\
\
class CSimpleVolumesStrategy(EURGBP,PERIOD\_H1,17,0.70,0.90,50,10000.00,750.00,10000,3),\
\
class CSimpleVolumesStrategy(EURGBP,PERIOD\_H1,27,0.70,0.90,60,10000.00,550.00,10000,3),\
\
class CSimpleVolumesStrategy(EURGBP,PERIOD\_H1,37,0.70,0.90,80,10000.00,150.00,10000,3)\
\
\], 0.33)

The first parameter of the _CVirtualStrategyGroup_ class object constructor is an array of trading strategy objects or an array of trading strategy group objects. Therefore, it is necessary to learn how to parse a part of a string, which will be an array of similar object descriptions. As you can see, I have used the standard notation applied in JSON or Python to represent a list (array) of elements: the element entries are separated by commas and located inside a pair of square brackets.

We will also have to learn how to extract parts from a string not only between commas, but also those that represent a description of another nested class object. By chance, to convert the trading strategy object to a string, we used the _typename()_ function, which returns the class name of an object as a string preceded by the _class_ words. We can now use this word when parsing a string as a sign that what follows is a string describing an object of a certain class, and not a simple value such as a number or string.

Thus, we come to an understanding of the need to implement the Factory design pattern, when a special object will be engaged in the creation of objects of various classes upon request. The objects that a factory can produce should typically have a common ancestor in the class hierarchy. So let's start by creating a new common class all classes (whose objects can be created from the initialization string) will eventually be derived from.

### New base class

So far, our base classes participating in the inheritance hierarchy have been:

- **_СAdvisor_**. The class for creating EAs the _CVirtualAdvisor_ class is derived from.
- **_CStrategy_**. The class for creating trading strategies. _CSimpleVolumesStrategy_ is derived from it
- _**CVirtualStrategyGroup**_. The class for groups of trading strategies. It has no descendants and none are expected.

- So, is that all?

Yes, I don't see any more base classes having descendants that need the ability to be initialized with a string. This means that these three classes need to have a common ancestor, in which to collect all the necessary auxiliary methods to ensure initialization with a string.

The name I have chosen for the new ancestor is not very meaningful yet. I would like to somehow emphasize that the class descendants will be able to be produced in the Factory, so they will be "factoryable". Further on, while developing the code, the letter "y" disappeared somewhere, and only the name CFaсtorable remained.

Initially, the class looked something like this:

```
//+------------------------------------------------------------------+
//| Base class of objects created from a string                      |
//+------------------------------------------------------------------+
class CFactorable {
protected:
   virtual void      Init(string p_params) = 0;
public:
   virtual string    operator~() = 0;

   static string     Read(string &p_params);
};
```

So, the descendants of this class were required to have the _Init()_ method, which will do all the work necessary to convert the input initialization string into object property values, and the tilde operator, which deals with the reverse conversion of properties into an initialization string. The existence of the _Read()_ static method is also stated. It should be able to read some of the data from the initialization string. By a data part, we mean a substring that contains either a valid initialization string of another object, or an array of other data parts, or a number, or a string constant.

Although this implementation was brought to a working state, I decided to make significant changes to it.

First, the _Init()_ method appeared because I wanted to keep both the old object constructors and the new constructor (which accepts the initialization string). To avoid duplicating code, I implemented it once in the _Init()_ method and called it from several possible constructors. But in the end it turned out that there was no need for different constructors. We can get by with just one new constructor. Therefore the _Init()_ method code moved to the new constructor, while the method itself was removed.

Second, the initial implementation did not contain any means of checking the validity of initialization strings and error reports. We expect to generate initialization strings automatically, which almost completely eliminates the occurrence of such errors, but if suddenly we mess up something with the generated initialization strings, it would be nice to know about it in a timely manner and be able to find the error. For these purposes, I have added a new _m\_isValid_ logical property, which indicates whether all of the object constructor code executed successfully, or whether some parts of the initialization string contained errors. The property is made private, while the appropriate _IsValid()_ and _SetInvalid()_ methods are added to get and set its value. Moreover, the property is always _true_ initially, while the _SetInvalid()_ method can only set its value to _false_.

Third, the _Read()_ method became too cumbersome because of the implemented checks and error handling. So, it was split into several separate methods specializing in reading different types of data from the initialization string. Several auxiliary private methods have also been added for data reading methods. It is worth noting separately that the data reading methods modify the initialization string that is passed to them. When the next part of the data is successfully read, it is returned as the result of the method, and the passed initialization string loses the part it read.

Fourth, the method of converting an object back to an initialization string can be made almost identical for objects of different classes if the original initialization string is remembered with the parameters of the created object. Therefore, the _m\_params_ property was added to the base class to store the initialization string in the object constructor.

Considering the additions made, declaring the _CFactorable_ class looks like this:

```
//+------------------------------------------------------------------+
//| Base class of objects created from a string                      |
//+------------------------------------------------------------------+
class CFactorable {
private:
   bool              m_isValid;  // Is the object valid?

   // Clear empty characters from left and right in the initialization string
   static void       Trim(string &p_params);

   // Find a matching closing bracket in the initialization string
   static int        FindCloseBracket(string &p_params, char closeBraket = ')');

   // Clear the initialization string with a check for the current object validity
   bool              CheckTrimParams(string &p_params);

protected:
   string            m_params;   // Current object initialization string

   // Set the current object to the invalid state
   void              SetInvalid(string function = NULL, string message = NULL);

public:
                     CFactorable() : m_isValid(true) {}  // Constructor
   bool              IsValid();                          // Is the object valid?

   // Convert object to string
   virtual string    operator~() = 0;

   // Does the initialization string start with the object definition?
   static bool       IsObject(string &p_params, const string className = "");

   // Does the initialization string start with defining an object of the desired class?
   static bool       IsObjectOf(string &p_params, const string className);

   // Read the object class name from the initialization string
   static string     ReadClassName(string &p_params, bool p_removeClassName = true);

   // Read an object from the initialization string
   string            ReadObject(string &p_params);

   // Read an array from the initialization string as a string
   string            ReadArrayString(string &p_params);

   // Read a string from the initialization string
   string            ReadString(string &p_params);

   // Read a number from the initialization string as a string
   string            ReadNumber(string &p_params);

   // Read a real number from the initialization string
   double            ReadDouble(string &p_params);

   // Read an integer from the initialization string
   long              ReadLong(string &p_params);
};
```

I will not dwell on the implementation of the class methods here. However, I would like to note that the work of all reading methods involves performing a roughly similar set of actions. First, we check that the initialization string is not empty and the object is valid. The object could have entered an invalid state, for example, as a result of a previous unsuccessful operation to read part of the data from the implementation string. Therefore, such a check helps to avoid performing unnecessary actions on an obviously faulty object.

Then certain conditions are checked to ensure that the initialization string contains data of the correct type (object, array, string or number). If so, then we find the location where that piece of data ends in the initialization string. Everything located to the left of this place is used to get the return value, and everything to the right replaces the initialization string.

If at some stage of the checks we receive a negative result, then call the method of setting the current object to the invalid state, while passing information about the error location and nature to it.

Save the code of the class in the _Factorable.mqh_ file in the current folder.

### Object factory

Since the object initialization strings always include the class name, we can make a public function or static method that will act as an object "factory". We will pass an initialization string to it receiving a pointer to the created object of the given class.

Of course, for objects of the classes whose name in a given place in the program can take on a single value, the presence of such a factory is not necessary. We can create an object in the standard way using the _new_ operator by passing the initialization string with the parameters of the created object to the constructor. But if we have to create objects whose class name can be different (for example, different trading strategies), then the _new_ operator is unable to help us, since we first need to define the class of the object we are about to create. Let's entrust this work to the factory, or rather, to its only static method - _Create()._

```
//+------------------------------------------------------------------+
//| Object factory class                                             |
//+------------------------------------------------------------------+
class CVirtualFactory {
public:
   // Create an object from the initialization string
   static CFactorable* Create(string p_params) {
      // Read the object class name
      string className = CFactorable::ReadClassName(p_params);

      // Pointer to the object being created
      CFactorable* object = NULL;

      // Call the corresponding constructor  depending on the class name
      if(className == "CVirtualAdvisor") {
         object = new CVirtualAdvisor(p_params);
      } else if(className == "CVirtualStrategyGroup") {
         object = new CVirtualStrategyGroup(p_params);
      } else if(className == "CSimpleVolumesStrategy") {
         object = new CSimpleVolumesStrategy(p_params);
      }

      // If the object is not created or is created in the invalid state, report an error
      if(!object) {
         PrintFormat(__FUNCTION__" | ERROR: Constructor not found for:\nclass %s(%s)",
                     className, p_params);
      } else if(!object.IsValid()) {
         PrintFormat(__FUNCTION__
                     " | ERROR: Created object is invalid for:\nclass %s(%s)",
                     className, p_params);
         delete object; // Remove the invalid object
         object = NULL;
      }

      return object;
   }
};
```

Save this code in the _VirtualFactory.mqh_ file of the current folder.

Create two useful macros to make it easier for us to use the factory in the future. The first one will create an object from the initialization string replacing itself with calling the _CVirtualFactory::Create()_ method:

```
// Create an object in the factory from a string
#define NEW(Params) CVirtualFactory::Create(Params)
```

The second macro will only be run from the constructor of some other object, which should be the _CFactorable_ class descendant. In other words, this will happen only if we create the main object, while implementing other (nested) objects from the initialization string inside its constructor. The macro is to receive three parameters: created object class name ( _Class_), name of the variable receiving the pointer to the created object ( _Object_) and initialization string ( _Params_).

At the beginning, the macro will declare a pointer variable with the given name and class and initialize it with the NULL value. Then we check whether the main object is valid. If yes, then call the object creation method in the factory via the NEW() macro. Then try to cast the created pointer to the required class. Using the dynamic\_cast<>() operator for this purpose avoids a runtime error if the factory creates an object of a different Class than the one currently required. In this case, the Object pointer will simply remain equal to NULL, and the program will continue running. Then we check the pointer validity. If it is empty or invalid, set the main object to the invalid state, report an error and abort the main object constructor execution.

This is what the macro looks like:

```
// Creating a child object in the factory from a string with verification.
// Called only from the current object constructor.
// If the object is not created, the current object becomes invalid
// and exit from the constructor is performed
#define CREATE(Class, Object, Params)                                                                       \
    Class *Object = NULL;                                                                                   \
    if (IsValid()) {                                                                                        \
       Object = dynamic_cast<C*> (NEW(Params));                                                             \
       if(!Object) {                                                                                        \
          SetInvalid(__FUNCTION__, StringFormat("Expected Object of class %s() at line %d in Params:\n%s",  \
                                                #Class, __LINE__, Params));                                 \
          return;                                                                                           \
       }                                                                                                    \
    }                                                                                                       \
```

Add these macros to the beginning of the _Factorable.mqh_ file.

### Modification of the previous base classes

Add the _CFactorable_ class as a base one to all the previous base classes: _СAdvisor_, _СStrategy_ and _СVirtualStrategyGroup_. The first two will not require any further changes:

```
//+------------------------------------------------------------------+
//| EA base class                                                    |
//+------------------------------------------------------------------+
class CAdvisor : public CFactorable {
protected:
   CStrategy         *m_strategies[];  // Array of trading strategies
   virtual void      Add(CStrategy *strategy);  // Method for adding a strategy
public:
                    ~CAdvisor();                // Destructor
   virtual void      Tick();                    // OnTick event handler
   virtual double    Tester() {
      return 0;
   }
};
//+------------------------------------------------------------------+
//| Base class of the trading strategy                               |
//+------------------------------------------------------------------+
class CStrategy : public CFactorable {
public:
   virtual void      Tick() = 0; // Handle OnTick events
};
```

_СVirtualStrategyGroup_ has undergone more serious changes. Since this is no longer an abstract base class, we needed to write an implementation of the constructor in it that creates an object from the initialization string. In doing so, we got rid of two separate constructors that took either an array of strategies or an array of groups. Also, the method of converting to a string has now changed. In the method, we now simply add the class name to the saved initialization string with parameters. The _Scale()_ scaling method has remained unchanged.

```
//+------------------------------------------------------------------+
//| Class of trading strategies group(s)                             |
//+------------------------------------------------------------------+
class CVirtualStrategyGroup : public CFactorable {
protected:
   double            m_scale;                // Scaling factor
   void              Scale(double p_scale);  // Scaling the normalized balance
public:
                     CVirtualStrategyGroup(string p_params); // Constructor

   virtual string    operator~() override;      // Convert object to string

   CVirtualStrategy      *m_strategies[];       // Array of strategies
   CVirtualStrategyGroup *m_groups[];           // Array of strategy groups
};

//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CVirtualStrategyGroup::CVirtualStrategyGroup(string p_params) {
// Save the initialization string
   m_params = p_params;

// Read the initialization string of the array of strategies or groups
   string items = ReadArrayString(p_params);

// Until the string is empty
   while(items != NULL) {
      // Read the initialization string of one strategy or group object
      string itemParams = ReadObject(items);

      // If this is a group of strategies,
      if(IsObjectOf(itemParams, "CVirtualStrategyGroup")) {
         // Create a strategy group and add it to the groups array
         CREATE(CVirtualStrategyGroup, group, itemParams);
         APPEND(m_groups, group);
      } else {
         // Otherwise, create a strategy and add it to the array of strategies
         CREATE(CVirtualStrategy, strategy, itemParams);
         APPEND(m_strategies, strategy);
      }
   }

// Read the scaling factor
   m_scale = ReadDouble(p_params);

// Correct it if necessary
   if(m_scale <= 0.0) {
      m_scale = 1.0;
   }

   if(ArraySize(m_groups) > 0 && ArraySize(m_strategies) == 0) {
      // If we filled the array of groups, and the array of strategies is empty, then
      // Scale all groups
      Scale(m_scale / ArraySize(m_groups));
   } else if(ArraySize(m_strategies) > 0 && ArraySize(m_groups) == 0) {
      // If we filled the array of strategies, and the array of groups is empty, then
      // Scale all strategies
      Scale(m_scale / ArraySize(m_strategies));
   } else {
      // Otherwise, report an error in the initialization string
      SetInvalid(__FUNCTION__, StringFormat("Groups or strategies not found in Params:\n%s", p_params));
   }
}

//+------------------------------------------------------------------+
//| Convert an object to a string                                    |
//+------------------------------------------------------------------+
string CVirtualStrategyGroup::operator~() {
   return StringFormat("%s(%s)", typename(this), m_params);
}

    ...
```

Save the changes made to the _VirtualStrategyGroup.mqh_ file in the current folder.

### Modification of the EA class

In the previous article, the _CVirtualAdvisor_ EA class received the _Init()_ method, which was supposed to remove code duplication for different EA constructors. We had a constructor that took a single strategy as its first argument, and a constructor that took a strategy group object as its first argument. It probably will not be difficult for us to agree that there will be only one constructor - the one which accepts a group of strategies. If we need to use one instance of a trading strategy, we first simply create a group with this one strategy and pass the created group to the EA constructor. Then there is no need for the _Init()_ method and additional constructors. Therefore, I will leave one constructor that creates an EA object from the initialization string:

```
//+------------------------------------------------------------------+
//| Class of the EA handling virtual positions (orders)              |
//+------------------------------------------------------------------+
class CVirtualAdvisor : public CAdvisor {
   ...

public:
                     CVirtualAdvisor(string p_param);    // Constructor
                    ~CVirtualAdvisor();         // Destructor

   virtual string    operator~() override;      // Convert object to string

   ...
};

...

//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CVirtualAdvisor::CVirtualAdvisor(string p_params) {
// Save the initialization string
   m_params = p_params;

// Read the initialization string of the strategy group object
   string groupParams = ReadObject(p_params);

// Read the magic number
   ulong p_magic = ReadLong(p_params);

// Read the EA name
   string p_name = ReadString(p_params);

// Read the work flag only at the bar opening
   m_useOnlyNewBar = (bool) ReadLong(p_params);

// If there are no read errors,
   if(IsValid()) {
      // Create a strategy group
      CREATE(CVirtualStrategyGroup, p_group, groupParams);

      // Initialize the receiver with the static receiver
      m_receiver = CVirtualReceiver::Instance(p_magic);

      // Initialize the interface with the static interface
      m_interface = CVirtualInterface::Instance(p_magic);

      m_name = StringFormat("%s-%d%s.csv",
                            (p_name != "" ? p_name : "Expert"),
                            p_magic,
                            (MQLInfoInteger(MQL_TESTER) ? ".test" : "")
                           );

      // Save the work (test) start time
      m_fromDate = TimeCurrent();

      // Reset the last save time
      m_lastSaveTime = 0;

      // Add the contents of the group to the EA
      Add(p_group);

      // Remove the group object
      delete p_group;
   }
}
```

In the constructor, we first read all the data from the initialization string. If any discrepancy is detected at this stage, the currently created EA object will go into an invalid state. If all is well, the constructor will create a strategy group, add its strategies to its strategy array, and set the remaining properties based on the data read from the initialization string.

But now, due to the validity check before creating the receiver and interface objects in the constructor, these objects may not be created. Therefore, in the destructor, we need to add a check for the correctness of pointers to these objects before deleting them:

```
//+------------------------------------------------------------------+
//| Destructor                                                       |
//+------------------------------------------------------------------+
void CVirtualAdvisor::~CVirtualAdvisor() {
   if(!!m_receiver)  delete m_receiver;         // Remove the recipient
   if(!!m_interface) delete m_interface;        // Remove the interface
   DestroyNewBar();           // Remove the new bar tracking objects
}
```

Save the changes in the _VirtualAdvisor.mqh_ file of the current folder.

### Modification of the trading strategy class

In the _CSimpleVolumesStrategy_ strategy class, remove the constructor with separate parameters and rewrite the code of the constructor that accepts the initialization string using the _CFactorable_ class methods.

In the constructor, read the parameters from the initialization string having previously saved its initial state in the _m\_params_ property. If no errors occurred during reading that would cause the strategy object to become invalid, perform the basic actions to initialize the object: fill the array of virtual positions, initialize the indicator and register the event handler for a new bar on the minute timeframe.

The method of converting an object to a string has also changed. Instead of forming it from the parameters, we will simply concatenate the class name and the saved initialization string, as we did in the two previous considered classes.

```
//+------------------------------------------------------------------+
//| Trading strategy using tick volumes                              |
//+------------------------------------------------------------------+
class CSimpleVolumesStrategy : public CVirtualStrategy {
   ...

public:
   //--- Public methods
                     CSimpleVolumesStrategy(string p_params); // Constructor

   virtual string    operator~() override;         // Convert object to string

   virtual void      Tick() override;              // OnTick event handler
};

//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CSimpleVolumesStrategy::CSimpleVolumesStrategy(string p_params) {
// Save the initialization string
   m_params = p_params;

// Read the parameters from the initialization string
   m_symbol = ReadString(p_params);
   m_timeframe = (ENUM_TIMEFRAMES) ReadLong(p_params);
   m_signalPeriod = (int) ReadLong(p_params);
   m_signalDeviation = ReadDouble(p_params);
   m_signaAddlDeviation = ReadDouble(p_params);
   m_openDistance = (int) ReadLong(p_params);
   m_stopLevel = ReadDouble(p_params);
   m_takeLevel = ReadDouble(p_params);
   m_ordersExpiration = (int) ReadLong(p_params);
   m_maxCountOfOrders = (int) ReadLong(p_params);
   m_fittedBalance = ReadDouble(p_params);

// If there are no read errors,
   if(IsValid()) {
      // Request the required number of virtual positions
      CVirtualReceiver::Get(GetPointer(this), m_orders, m_maxCountOfOrders);

      // Load the indicator to get tick volumes
      m_iVolumesHandle = iVolumes(m_symbol, m_timeframe, VOLUME_TICK);

      // If the indicator is loaded successfully
      if(m_iVolumesHandle != INVALID_HANDLE) {

         // Set the size of the tick volume receiving array and the required addressing
         ArrayResize(m_volumes, m_signalPeriod);
         ArraySetAsSeries(m_volumes, true);

         // Register the event handler for a new bar on the minimum timeframe
         IsNewBar(m_symbol, PERIOD_M1);
      } else {
         // Otherwise, set the object state to invalid
         SetInvalid(__FUNCTION__, "Can't load iVolumes()");
      }
   }
}

//+------------------------------------------------------------------+
//| Convert an object to a string                                    |
//+------------------------------------------------------------------+
string CSimpleVolumesStrategy::operator~() {
   return StringFormat("%s(%s)", typename(this), m_params);
}
```

We also removed the _Save()_ and _Load()_ methods from the class since their implementation in the _CVirtualStrategy_ parent class proved to be quite sufficient to carry out the tasks assigned to it.

Save the changes in the _CSimpleVolumesStrategy.mqh_ file of the current folder.

### The EA for a single instance of the trading strategy

To optimize the parameters of a single trading strategy instance, we need to change only the _OnInit()_ initialization function. In this function, we should form a string for initializing the trading strategy object from the EA inputs and then use it to substitute the EA object into the initialization string.

Thanks to our implementation of methods for reading data from the initialization string, we are free to use additional spaces and line feeds inside it. Then when outputting to the log or making an entry in the database, we can see the initialization string formatted approximately like this:

```
Core 1  2023.01.01 00:00:00   OnInit | Expert Params:
Core 1  2023.01.01 00:00:00   class CVirtualAdvisor(
Core 1  2023.01.01 00:00:00       class CVirtualStrategyGroup(
Core 1  2023.01.01 00:00:00          [\
Core 1  2023.01.01 00:00:00           class CSimpleVolumesStrategy("EURGBP",16385,17,0.70,0.90,150,10000.00,85.00,10000,3,0.00)\
Core 1  2023.01.01 00:00:00          ],1
Core 1  2023.01.01 00:00:00       ),
Core 1  2023.01.01 00:00:00       ,27181,SimpleVolumesSingle,1
Core 1  2023.01.01 00:00:00   )
```

In the _OnDeinit()_ function, we need to make sure the pointer to the EA object is correct before removing it. Now we can no longer guarantee that the EA object will always be created, since theoretically we could have an incorrect initialization string, which would lead to the early deletion of the EA object by the factory.

```
...

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit() {
   CMoney::FixedBalance(fixedBalance_);

// Prepare the initialization string for a single strategy instance
   string strategyParams = StringFormat(
                              "class CSimpleVolumesStrategy(\"%s\",%d,%d,%.2f,%.2f,%d,%.2f,%.2f,%d,%d,%.2f)",
                              symbol_, timeframe_,
                              signalPeriod_, signalDeviation_, signaAddlDeviation_,
                              openDistance_, stopLevel_, takeLevel_, ordersExpiration_,
                              maxCountOfOrders_, 0
                           );

// Prepare the initialization string for an EA with a group of a single strategy
   string expertParams = StringFormat(
                            "class CVirtualAdvisor(\n"
                            "    class CVirtualStrategyGroup(\n"
                            "       [\n"\
                            "        %s\n"\
                            "       ],1\n"
                            "    ),\n"
                            "    ,%d,%s,%d\n"
                            ")",
                            strategyParams, magic_, "SimpleVolumesSingle", true
                         );

   PrintFormat(__FUNCTION__" | Expert Params:\n%s", expertParams);

// Create an EA handling virtual positions
   expert = NEW(expertParams);

   if(!expert) return INIT_FAILED;

   return(INIT_SUCCEEDED);
}

...

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason) {
   if(!!expert) delete expert;
}
```

Save the obtained code in the file _SimpleVolumesExpertSingle.mq5_ of the current folder.

### EA for multiple instances

To test the EA creation with multiple trading strategy instances, take the EA from the [eighth part](https://www.mql5.com/en/articles/14574), which we used to perform load testing. In the _OnInit()_ function, replace the EA creation mechanism with the one developed in this article. To do this, after loading the strategy parameters from the CSV file, we will supplement the initialization string of the strategy array based on them. Then we will use it to form the initialization string for the strategy group and the EA itself:

```
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit() {
// Load strategy parameter sets
   int totalParams = LoadParams(fileName_, strategyParams);

// If nothing is loaded, report an error
   if(totalParams == 0) {
      PrintFormat(__FUNCTION__" | ERROR: Can't load data from file %s.\n"
                  "Check that it exists in data folder or in common data folder.",
                  fileName_);
      return(INIT_PARAMETERS_INCORRECT);
   }

// Report an error if
   if(count_ < 1) { // number of instances is less than 1
      return INIT_PARAMETERS_INCORRECT;
   }

   ArrayResize(strategyParams, count_);

// Set parameters in the money management class
   CMoney::DepoPart(expectedDrawdown_ / 10.0);
   CMoney::FixedBalance(fixedBalance_);

// Prepare the initialization string for the array of strategy instances
   string strategiesParams;
   FOREACH(strategyParams, strategiesParams += StringFormat(" class CSimpleVolumesStrategy(%s),\n      ",
                                                            strategyParams[i % totalParams]));

// Prepare the initialization string for an EA with the strategy group
   string expertParams = StringFormat("class CVirtualAdvisor(\n"
                                      "   class CVirtualStrategyGroup(\n"
                                      "      [\n"\
                                      "      %s],\n"
                                      "      %.2f\n"
                                      "   ),\n"
                                      "   %d,%s,%d\n"
                                      ")",
                                      strategiesParams, scale_,
                                      magic_, "SimpleVolumes_BenchmarkInstances", useOnlyNewBars_);

// Create an EA handling virtual positions
   expert = NEW(expertParams);

   PrintFormat(__FUNCTION__" | Expert Params:\n%s", expertParams);

   if(!expert) return INIT_FAILED;

   return(INIT_SUCCEEDED);
}
```

Similar to the previous EA, the _OnDeinit()_ function receives the ability to check the validity of the pointer to the EA object before deleting it.

Save the obtained code in the _BenchmarkInstancesExpert.mq5_ file of the current folder.

### Checking functionality

Let's use the _BenchmarkInstancesExpert.mq5_ EA from the [eighth part](https://www.mql5.com/en/articles/14574) and the same EA from the current article. Launch them with the same parameters: 256 instances of trading strategies from the _Params\_SV\_EURGBP\_H1.csv_ file, the year of 2022 serves as a test period.

![](https://c.mql5.com/2/76/6068947895047.png)

![](https://c.mql5.com/2/76/3208549520357.png)

Fig. 2. The test results of the two EA versions are identical

The results have turned out to be identical. Therefore, they are displayed as one instance in the image. This is very good, as we can now use the latest version for its further development.

### Conclusion

So, we have managed to provide the ability to create all the necessary objects using initialization strings. So far we have been generating these lines almost manually, but in the future we will be able to read them from the database. This is, in general, why we started such a revision of the already working code.

Identical results of testing EAs that differ only in the method of creating objects, i.e. working with the same sets of trading strategy instances, justify the changes made.

Now we can move on and proceed to the automation of the first planned stage - the sequential launch of several processes of EA optimization to select the parameters of a single instance of the trading strategy. We will do this in the coming articles.

Thank you for your attention! See you soon!

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/14739](https://www.mql5.com/ru/articles/14739)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/14739.zip "Download all attachments in the single ZIP archive")

[Advisor.mqh](https://www.mql5.com/en/articles/download/14739/advisor.mqh "Download Advisor.mqh")(4.5 KB)

[BenchmarkInstancesExpert.mq5](https://www.mql5.com/en/articles/download/14739/benchmarkinstancesexpert.mq5 "Download BenchmarkInstancesExpert.mq5")(14.82 KB)

[Factorable.mqh](https://www.mql5.com/en/articles/download/14739/factorable.mqh "Download Factorable.mqh")(32.51 KB)

[SimpleVolumesExpertSingle.mq5](https://www.mql5.com/en/articles/download/14739/simplevolumesexpertsingle.mq5 "Download SimpleVolumesExpertSingle.mq5")(11.85 KB)

[SimpleVolumesStrategy.mqh](https://www.mql5.com/en/articles/download/14739/simplevolumesstrategy.mqh "Download SimpleVolumesStrategy.mqh")(27.07 KB)

[Strategy.mqh](https://www.mql5.com/en/articles/download/14739/strategy.mqh "Download Strategy.mqh")(1.87 KB)

[VirtualAdvisor.mqh](https://www.mql5.com/en/articles/download/14739/virtualadvisor.mqh "Download VirtualAdvisor.mqh")(21.18 KB)

[VirtualFactory.mqh](https://www.mql5.com/en/articles/download/14739/virtualfactory.mqh "Download VirtualFactory.mqh")(4.31 KB)

[VirtualStrategy.mqh](https://www.mql5.com/en/articles/download/14739/virtualstrategy.mqh "Download VirtualStrategy.mqh")(8.1 KB)

[VirtualStrategyGroup.mqh](https://www.mql5.com/en/articles/download/14739/virtualstrategygroup.mqh "Download VirtualStrategyGroup.mqh")(8.83 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Moving to MQL5 Algo Forge (Part 4): Working with Versions and Releases](https://www.mql5.com/en/articles/19623)
- [Moving to MQL5 Algo Forge (Part 3): Using External Repositories in Your Own Projects](https://www.mql5.com/en/articles/19436)
- [Moving to MQL5 Algo Forge (Part 2): Working with Multiple Repositories](https://www.mql5.com/en/articles/17698)
- [Moving to MQL5 Algo Forge (Part 1): Creating the Main Repository](https://www.mql5.com/en/articles/17646)
- [Developing a multi-currency Expert Advisor (Part 24): Adding a new strategy (II)](https://www.mql5.com/en/articles/17328)
- [Developing a multi-currency Expert Advisor (Part 24): Adding a new strategy (I)](https://www.mql5.com/en/articles/17277)
- [Developing a multi-currency Expert Advisor (Part 23): Putting in order the conveyor of automatic project optimization stages (II)](https://www.mql5.com/en/articles/16913)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/473051)**
(2)


![Viktor Kudriavtsev](https://c.mql5.com/avatar/2020/8/5F496EFA-E815.jpg)

**[Viktor Kudriavtsev](https://www.mql5.com/en/users/ale5312)**
\|
16 May 2024 at 14:05

Yuri hello. Thank you for the interesting series of articles.

Yuri, could you post the strategy file with which you tested the Expert Advisor from the current article? This is the one you got the screenshot at the bottom of the article. If it is posted somewhere, please tell me where, I have not found it under other articles. Should I put it in the folder C:\\Users\\Admin/AppData\\Roaming\\MetaQuotes\\Terminal\\Common\\Files or in the [terminal folder](https://www.mql5.com/en/docs/constants/environment_state/terminalstatus#enum_terminal_info_string "MQL5 Documentation: Client Terminal Status")? I want to see if I get the same results in the terminal as in your screenshot.

![Yuriy Bykov](https://c.mql5.com/avatar/avatar_na2.png)

**[Yuriy Bykov](https://www.mql5.com/en/users/antekov)**
\|
16 May 2024 at 18:35

Hello Victor.

This file can be obtained by running the EA optimisation with one strategy instance and after finishing it, saving its results first in XML and then saving it to CSV from Excel. This was explained in Part [6](https://www.mql5.com/ru/articles/14478).

![Creating a Trading Administrator Panel in MQL5 (Part III): Enhancing the GUI with Visual Styling (I)](https://c.mql5.com/2/93/Creating_a_Trading_Administrator_Panel_in_MQL5_Part_III___LOGO.png)[Creating a Trading Administrator Panel in MQL5 (Part III): Enhancing the GUI with Visual Styling (I)](https://www.mql5.com/en/articles/15419)

In this article, we will focus on visually styling the graphical user interface (GUI) of our Trading Administrator Panel using MQL5. We’ll explore various techniques and features available in MQL5 that allow for customization and optimization of the interface, ensuring it meets the needs of traders while maintaining an attractive aesthetic.

![How to Implement Auto Optimization in MQL5 Expert Advisors](https://c.mql5.com/2/93/Implementing_Auto_Optimization_in_MQL5_Expert_Advisors__LOGO.png)[How to Implement Auto Optimization in MQL5 Expert Advisors](https://www.mql5.com/en/articles/15837)

Step by step guide for auto optimization in MQL5 for Expert Advisors. We will cover robust optimization logic, best practices for parameter selection, and how to reconstruct strategies with back-testing. Additionally, higher-level methods like walk-forward optimization will be discussed to enhance your trading approach.

![Example of CNA (Causality Network Analysis), SMOC (Stochastic Model Optimal Control) and Nash Game Theory with Deep Learning](https://c.mql5.com/2/94/Example_of_CNA_b_SMOC_and_Nash_Game__LOGO2.png)[Example of CNA (Causality Network Analysis), SMOC (Stochastic Model Optimal Control) and Nash Game Theory with Deep Learning](https://www.mql5.com/en/articles/15819)

We will add Deep Learning to those three examples that were published in previous articles and compare results with previous. The aim is to learn how to add DL to other EA.

![Creating an MQL5-Telegram Integrated Expert Advisor (Part 6): Adding Responsive Inline Buttons](https://c.mql5.com/2/93/Creating_an_MQL5-Telegram_Integrated_Expert_Advisor_Part_6__LOGO.png)[Creating an MQL5-Telegram Integrated Expert Advisor (Part 6): Adding Responsive Inline Buttons](https://www.mql5.com/en/articles/15823)

In this article, we integrate interactive inline buttons into an MQL5 Expert Advisor, allowing real-time control via Telegram. Each button press triggers specific actions and sends responses back to the user. We also modularize functions for handling Telegram messages and callback queries efficiently.

[![](https://www.mql5.com/ff/sh/ub4fqgrk4rkv8gz9z2/01.png)![](https://www.mql5.com/ff/sh/ub4fqgrk4rkv8gz9z2/02.png)Explore your trading for freeUpdated statistics in MetaTrader 5 will help you to thoroughly evaluate results and reduce risksLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/454106&a=bkbqgaxtrafeuegfvjisjjwjohagrvnr&s=25c5856d7857fc6b6db7cffb15ae4ce40fd19d1ab594d8a900ad65673d9ffa0e&uid=&ref=https://www.mql5.com/en/articles/14739&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5048964215504478780)

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