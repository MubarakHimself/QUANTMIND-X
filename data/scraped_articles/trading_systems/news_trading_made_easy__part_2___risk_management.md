---
title: News Trading Made Easy (Part 2): Risk Management
url: https://www.mql5.com/en/articles/14912
categories: Trading Systems
relevance_score: -2
scraped_at: 2026-01-24T14:17:03.024910
---

[![](https://www.mql5.com/ff/si/fx5m8s6u6uxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=ysducdhemkrdsdtzzbfkclolrllnhezk&s=33f180a31db6c3b846d77732b0bc78169421a47b8cf9f076ca717f4e4846d1c7&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=wsfsjzkisnyzfaqogovhblkbqlxahubl&ssn=1769253419622610811&ssn_dr=0&ssn_sr=0&fv_date=1769253419&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F14912&back_ref=https%3A%2F%2Fwww.google.com%2F&title=News%20Trading%20Made%20Easy%20(Part%202)%3A%20Risk%20Management%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176925341927360648&fz_uniq=5083461534560033834&sv=2552)

MetaTrader 5 / Examples


## Introduction

A quick refresher for [the previous article](https://www.mql5.com/en/articles/14324 "News Trading Made Easy (Part 1): Creating a Database") in the News Trading Made Easy series. In part 1, we went through the concept of DST(Daylight Savings Time) and the various versions for different countries that essentially change their time zones by an hour ahead and behind during a financial year. This will change trading schedules for the related brokers using DST. The reasons for creating a database and the benefits were addressed. A database was created to store the news events from the [MQL5 Economic Calendar](https://www.mql5.com/en/economic-calendar "MQL5 Economic Calendar") with subsequent changes to the event time data to reflect the broker's DST schedule for accurate back-testing in the future. In the project files, an SQL script results in an Excel format was provided for all the unique events accessible through the MQL5 Calendar for all the different countries.

However, in this article, we will make a few changes to our previous code in part 1. Firstly by implementing inheritance to the existing code and upcoming new code, the previous news/calendar database will get a revamp into something more useable and practical. Additionally, we will tackle risk management and create different risk profiles to choose from for users with different risk appetites or preferences.

## Inheritance

![Types of Inheritance](https://c.mql5.com/2/78/TypesOfInheritance.png)

### What is Inheritance?

![Inheritance](https://c.mql5.com/2/78/inheritance2.png)

[Inheritance](https://www.mql5.com/go?link=https://www.programiz.com/cpp-programming/inheritance "https://www.programiz.com/cpp-programming/inheritance") is a fundamental concept in object-oriented programming (OOP) that allows a new class (called a subclass or derived class) to inherit properties and behaviors (fields and methods) from an existing class (called a superclass or base class). This mechanism provides a way to create a new class by extending or modifying the behavior of an existing class, promoting code reuse and the creation of a more logical and hierarchical class structure.

### What is the purpose of Inheritance?

Inheritance enables the reuse of existing code. By inheriting from a base class, a subclass can leverage existing methods and fields without having to rewrite them. This reduces redundancy and makes the code more maintainable.  Inheritance helps in organizing code into a hierarchical structure, which is easier to understand and manage. Classes can be grouped based on shared attributes and behaviors, leading to a clear and logical organization of the codebase. Inheritance is closely related to polymorphism, which allows objects of different classes to be treated as objects of a common superclass. This is particularly useful for implementing flexible and extensible code. For instance, a function can operate on objects of different classes as long as they inherit from the same base class, allowing for dynamic method binding and more generalized code.

Inheritance allows for extending the functionality of existing classes. Subclasses can add new methods and fields or override existing ones to introduce specific behaviors without modifying the original class. This promotes the open/closed principle, where classes are open for extension but closed for modification. Inheritance supports encapsulation by allowing a subclass to access the protected and public members of a superclass while keeping the implementation details private. This ensures a clear separation between the interface and implementation, enhancing modularity and reducing the risk of unintended interactions.

### What are Access Modifiers?

Access modifiers are keywords in object-oriented programming languages that set the accessibility of classes, methods/functions, and other members. They control the visibility and accessibility of these elements from different parts of the code, thus enforcing encapsulation and protecting the integrity of the data.

Types of Access Modifiers

- Public
- Private
- Protected

1\. **Public**

Purpose: To make a class, function, or variable available for use in other classes or programs.

2\. **Private**

Purpose: To restrict access to members of a class, thus protecting the integrity of the data.

3\. **Protected**

Purpose: To allow subclasses to inherit and access the members while still restricting access from class's objects.

### Inheritance Example related to MQL5

We will first create a [UML class diagram](https://www.mql5.com/go?link=https://www.visual-paradigm.com/guide/uml-unified-modeling-language/uml-class-diagram-tutorial/ "UML Class Diagram tutorial") for the example to visualize the classes and their relationships and attributes.

![UML Class Diagram](https://c.mql5.com/2/79/Example_uml_class_diagram_resize2.png)

Classes UnitedStates and Switzerland have singular Inheritance from class NewsData:

```
class NewsData
  {
private://Properties are only accessible from this class
   string            Country;//Private variable
   struct EventDetails//Private structure
     {
      int            EventID;
      string         EventName;
      datetime       EventDate;
     };
protected:
   //-- Protected Array Only accessible from this class and its children
   EventDetails      News[];
   //-- Proctected virtual void Function(to be expanded on via child classes)
   virtual void      SetNews();
   //-- Protected Function Only accessible from this class and its children
   void              SetCountry(string myCountry) {Country=myCountry;}
public:
   void              GetNews()//Public function to display 'News' array details
     {
      PrintFormat("+---------- %s ----------+",Country);
      for(uint i=0;i<News.Size();i++)
        {
         Print("ID: ",News[i].EventID," Name: ",News[i].EventName," Date: ",News[i].EventDate);
        }
     }
                     NewsData(void) {}//Class constructor
                    ~NewsData(void) {ArrayFree(News);}//Class destructor
  };

//+------------------------------------------------------------------+
//|(Subclass/Child) for 'NewsData'                                   |
//+------------------------------------------------------------------+
class UnitedStates:private NewsData
//private inheritance from NewsData,
//'UnitedStates' class's objects and children
//will not have access to 'NewsData' class's properties
  {
private:
   virtual void      SetNews()//private Function only Accessible in 'UnitedStates' class
     {
      ArrayResize(News,News.Size()+1,News.Size()+2);
      News[News.Size()-1].EventID = 1;
      News[News.Size()-1].EventName = "NFP(Non-Farm Payrolls)";
      News[News.Size()-1].EventDate = D'2024.01.03 14:00:00';
     }
public:
   void              myNews()//public Function accessible via class's object
     {
      SetCountry("United States");//Calling function from 'NewsData'
      GetNews();//Calling Function from private inherited class 'NewsData'
     }
                     UnitedStates(void) {SetNews();}//Class constructor
  };

//+------------------------------------------------------------------+
//|(Subclass/Child) for 'NewsData'                                   |
//+------------------------------------------------------------------+
class Switzerland: public NewsData
//public inheritance from NewsData
  {
public:
   virtual void      SetNews()//Public Function to set News data
     {
      ArrayResize(News,News.Size()+1,News.Size()+2);//Adjusting News structure array's size
      News[News.Size()-1].EventID = 0;//Setting event id to '0'
      News[News.Size()-1].EventName = "Interest Rate Decision";//Assigning event name
      News[News.Size()-1].EventDate = D'2024.01.06 10:00:00';//Assigning event date
     }
                     Switzerland(void) {SetCountry("Switerland"); SetNews();}//Class construct
  };
```

In this Example:

The (Parent/Base/Super) class would be NewsData and any private declarations will be only accessible by this class. The private declarations will be inaccessible to the class's objects and children. Whilst the protected declarations will be accessible to both the class and its children. Whereas all public declarations will be accessible to the class, its children and objects.

Accessibility Table For NewsData:

| Class's Properties | Class | Children | Objects |
| --- | --- | --- | --- |
| Variable: Country(Private) | ✔ | ✘ | ✘ |
| Structure: EventDetails(Private) | ✔ | ✘ | ✘ |
| Variable: News(Protected) | ✔ | ✔ | ✘ |
| Function: SetNews(Protected) | ✔ | ✔ | ✘ |
| Function: SetCountry(Protected) | ✔ | ✔ | ✘ |
| Function: GetNews(Public) | ✔ | ✔ | ✔ |
| Constructor: NewsData(Public) | ✔ | ✘ | ✘ |
| Destructor: ~NewsData(Public) | ✔ | ✘ | ✘ |

```
class NewsData
  {
private://Properties are only accessible from this class
   string            Country;//Private variable
   struct EventDetails//Private structure
     {
      int            EventID;
      string         EventName;
      datetime       EventDate;
     };
protected:
   //-- Protected Array Only accessible from this class and its children
   EventDetails      News[];
   //-- Proctected virtual void Function(to be expanded on via child classes)
   virtual void      SetNews();
   //-- Protected Function Only accessible from this class and its children
   void              SetCountry(string myCountry) {Country=myCountry;}
public:
   void              GetNews()//Public function to display 'News' array details
     {
      PrintFormat("+---------- %s ----------+",Country);
      for(uint i=0;i<News.Size();i++)
        {
         Print("ID: ",News[i].EventID," Name: ",News[i].EventName," Date: ",News[i].EventDate);
        }
     }
                     NewsData(void) {}//Class constructor
                    ~NewsData(void) {ArrayFree(News);}//Class destructor
  };
```

Visible Properties from the NewsData Object:

![Visible properties from object](https://c.mql5.com/2/79/NewsData_AccessableProperties.png)![Public function from object](https://c.mql5.com/2/79/NewsData_GetNews.png)

Result from GetNews function in NewsData:

![Function output](https://c.mql5.com/2/78/NewsData_GetNews_Result.png)

Inheritance is Implemented with the both the remaining classes:

In the (Child/Sub/Derived) class  UnitedStates, it inherits the parent(NewsData) class privately.

Meaning that the sub(UnitedStates) class can access the protected and public properties from the parent(NewsData) class, but the children for the UnitedStates class and its objects will not have access to any properties of the parent(NewsData) class. If the inheritance [Access Modifier](https://www.mql5.com/go?link=https://www.geeksforgeeks.org/access-modifiers-in-c/ "www.geeksforgeeks.org/access-modifiers-in-c/") was protective the UnitedStates class's children would have access to both the parent(NewsData) class's protective and public properties, but the sub(UnitedStates) class's objects would not have any access to the parent class's properties.

Accessibility Table For UnitedStates:

| Class's Properties | Class | Children | Objects |
| --- | --- | --- | --- |
| Inheritance(Private) Variable: Country(Private) | ✘ | ✘ | ✘ |
| Inheritance(Private) Structure: EventDetails(Private) | ✘ | ✘ | ✘ |
| Inheritance(Private) Variable: News(Protected) | ✔ | ✘ | ✘ |
| Inheritance(Private) Function: SetNews(Protected) | ✔ | ✘ | ✘ |
| Inheritance(Private) Function: SetCountry(Protected) | ✔ | ✘ | ✘ |
| Inheritance(Private) Function: GetNews(Public) | ✔ | ✘ | ✘ |
| Inheritance(Private) Constructor: NewsData(Public) | ✘ | ✘ | ✘ |
| Inheritance(Private) Destructor: ~NewsData(Public) | ✘ | ✘ | ✘ |
| Function: SetNews(Private) | ✔ | ✘ | ✘ |
| Function: myNews(Public) | ✔ | ✔ | ✔ |
| Constructor: UnitedStates(Public) | ✔ | ✘ | ✘ |

```
class UnitedStates:private NewsData
//private inheritance from NewsData,
//'UnitedStates' class's objects and children
//will not have access to 'NewsData' class's properties
  {
private:
   virtual void      SetNews()//private Function only Accessible in 'UnitedStates' class
     {
      ArrayResize(News,News.Size()+1,News.Size()+2);
      News[News.Size()-1].EventID = 1;
      News[News.Size()-1].EventName = "NFP(Non-Farm Payrolls)";
      News[News.Size()-1].EventDate = D'2024.01.03 14:00:00';
     }
public:
   void              myNews()//public Function accessible via class's object
     {
      SetCountry("United States");//Calling function from 'NewsData'
      GetNews();//Calling Function from private inherited class 'NewsData'
     }
                     UnitedStates(void) {SetNews();}//Class constructor
  };
```

Visible Properties from the UnitedStates Object:

![Visible properties from object](https://c.mql5.com/2/79/UnitedStates_AccessableProperties.png)![Private Inherited function from NewsData](https://c.mql5.com/2/79/UnitedStates_GetNews.png)

A compile error arises from attempting to access GetNews function which is privately inherited from NewsData, this prevents the UnitedStates object from access to the function.

![Compile error](https://c.mql5.com/2/78/UnitedStates_GetNews_Result.png)

In the (Child/Sub/Derived) class Switzerland.

The inheritance access modifier is public. This provides the sub(Switzerland) class's children access to the parent(NewsData) class's public and protective properties, whereas the Switzerland class's objects only have access to the public properties of all related classes.

Accessibility Table for Switzerland:

| Class's Properties | Class | Children | Objects |
| --- | --- | --- | --- |
| Inheritance(Public) Variable: Country(Private) | ✘ | ✘ | ✘ |
| Inheritance(Public) Structure: EventDetails(Private) | ✘ | ✘ | ✘ |
| Inheritance(Public) Variable: News(Protected) | ✔ | ✔ | ✘ |
| Inheritance(Public) Function: SetNews(Protected) | ✔ | ✔ | ✘ |
| Inheritance(Public) Function: SetCountry(Protected) | ✔ | ✔ | ✘ |
| Inheritance(Public) Function: GetNews(Public) | ✔ | ✔ | ✔ |
| Inheritance(Public) Constructor: NewsData(Public) | ✘ | ✘ | ✘ |
| Inheritance(Public) Destructor: ~NewsData(Public) | ✘ | ✘ | ✘ |
| Function: SetNews(Public) | ✔ | ✔ | ✔ |
| Constructor: Switzerland(Public) | ✔ | ✘ | ✘ |

```
class Switzerland: public NewsData
//public inheritance from NewsData
  {
public:
   virtual void      SetNews()//Public Function to set News data
     {
      ArrayResize(News,News.Size()+1,News.Size()+2);//Adjusting News structure array's size
      News[News.Size()-1].EventID = 0;//Setting event id to '0'
      News[News.Size()-1].EventName = "Interest Rate Decision";//Assigning event name
      News[News.Size()-1].EventDate = D'2024.01.06 10:00:00';//Assigning event date
     }
                     Switzerland(void) {SetCountry("Switerland"); SetNews();}//Class construct
  };
```

Visible Properties from the Switzerland Object:

![Visible properties from object](https://c.mql5.com/2/79/Switzerland_AccessableProperties.png)

Results:

![Objects results](https://c.mql5.com/2/78/overall_results.png)

## Daylight Savings Classes

### In News Trading Made Easy (Part 1):

UML Class Diagram

![UML Class Diagram(DaylightSavings Classes) Part 1](https://c.mql5.com/2/78/DaylightSavings_Part1_UML.png)

Project files:

![DaylightSavings Classes Part 1](https://c.mql5.com/2/78/DaylightSavingsClasses_Part1.png)

In the previous code we had three classes for Daylight Savings namely:

- CDaylightSavings\_AU
- CDaylightSavings\_UK
- CDaylightSavings\_US

### In Part 2:

UML Class Diagram

![UML Class Diagram(Daylight Savings Classes) Part 2](https://c.mql5.com/2/79/DaylightSavings_Part2_UML_resize.png)

Project files:

![Daylight Savings Classes Part 2](https://c.mql5.com/2/78/DaylightSavingsClasses_Part2.png)

We will have the following classes for Daylight Savings namely:

- CDaylightSavings
- CDaylightSavings\_AU
- CDaylightSavings\_UK
- CDaylightSavings\_US

### Why create another Daylight savings class?

In the previous classes' code there was a lot of repetition among the classes, which essentially was the same code written over again for different values in a list. Instead of this repetition occurring amongst similar classes, we will put all the commonalities in one separate class and inherit the common features into the different daylight savings classes.

### What are virtual Functions?

In object-oriented programming (OOP), a [virtual function](https://www.mql5.com/go?link=https://www.geeksforgeeks.org/virtual-function-cpp/ "https://www.geeksforgeeks.org/virtual-function-cpp/") is a member function in a base class that you can override in a derived class. When a function is declared as virtual, it enables polymorphism, allowing the derived class to provide a specific implementation of the function that can be called through a base class pointer or reference.

Purpose?

- Polymorphism: Virtual functions allow for dynamic (runtime) polymorphism. This means that the method that gets executed is determined at runtime based on the actual type of the object being referenced, rather than the type of the reference or pointer.
- Flexibility: They enable more flexible and reusable code by allowing derived classes to modify or extend the base class behavior.
- Decoupling: Virtual functions help in decoupling the code by separating the interface from the implementation, making it easier to change the implementation without affecting the code that uses the base class interface.

### CDaylightSavings Class

In this class all commonalities of the previous codes are put into one class and we will declare a few virtualfunctions to initialize the different lists for the respective Daylight Savings schedule.

CDaylightSavings class has a singular Inheritance from CObject class.

CDaylightSavings class has Inclusions from classes:

- CArrayObj
- CTimeManagement

```
//+------------------------------------------------------------------+
//|                                                      NewsTrading |
//|                                  Copyright 2024, MetaQuotes Ltd. |
//|                            https://www.mql5.com/en/users/kaaiblo |
//+------------------------------------------------------------------+

#include <Object.mqh>

#include <Arrays\ArrayObj.mqh>
#include "../TimeManagement.mqh"

//+------------------------------------------------------------------+
//|DaylightSavings class                                             |
//+------------------------------------------------------------------+
class CDaylightSavings: public CObject
  {

protected:
   CTimeManagement   Time;
                     CDaylightSavings(datetime startdate,datetime enddate);
   CObject           *List() { return savings;}//Gets the list of Daylightsavings time
   datetime          StartDate;
   datetime          EndDate;
   CArrayObj         *savings;
   CArrayObj         *getSavings;
   CDaylightSavings      *dayLight;
   virtual void      SetDaylightSavings_UK();//Initialize UK Daylight Savings Dates into List
   virtual void      SetDaylightSavings_US();//Initialize US Daylight Savings Dates into List
   virtual void      SetDaylightSavings_AU();//Initialize AU Daylight Savings Dates into List

public:
                     CDaylightSavings(void);
                    ~CDaylightSavings(void);
   bool              isDaylightSavings(datetime Date);//This function checks if a given date falls within Daylight Savings Time.
   bool              DaylightSavings(int Year,datetime &startDate,datetime &endDate);//Check if DaylightSavings Dates are available for a certain Year
   string            adjustDaylightSavings(datetime EventDate);//Will adjust the date's timezone depending on DaylightSavings
  };

//+------------------------------------------------------------------+
//|Constructor                                                       |
//+------------------------------------------------------------------+
CDaylightSavings::CDaylightSavings(void)
  {
  }

//+------------------------------------------------------------------+
//|Initialize variables                                              |
//+------------------------------------------------------------------+
CDaylightSavings::CDaylightSavings(datetime startdate,datetime enddate)
  {
   StartDate = startdate;//Assign class's global variable StartDate value from parameter variable startdate
   EndDate = enddate;//Assign class's global variable EndDate value from parameter variable enddate
  }

//+------------------------------------------------------------------+
//|checks if a given date falls within Daylight Savings Time         |
//+------------------------------------------------------------------+
bool CDaylightSavings::isDaylightSavings(datetime Date)
  {
// Initialize a list to store daylight savings periods.
   getSavings = List();
// Iterate through all the periods in the list.
   for(int i=0; i<getSavings.Total(); i++)
     {
      // Access the current daylight savings period.
      dayLight = getSavings.At(i);
      // Check if the given date is within the current daylight savings period.
      if(Time.DateIsInRange(dayLight.StartDate,dayLight.EndDate,Date))
        {
         // If yes, return true indicating it is daylight savings time.
         return true;
        }
     }
// If no period matches, return false indicating it is not daylight savings time.
   return false;
  }

//+------------------------------------------------------------------+
//|Check if DaylightSavings Dates are available for a certain Year   |
//+------------------------------------------------------------------+
bool CDaylightSavings::DaylightSavings(int Year,datetime &startDate,datetime &endDate)
  {
// Initialize a list to store daylight savings periods.
   getSavings = List();
   bool startDateDetected=false,endDateDetected=false;
// Iterate through all the periods in the list.
   for(int i=0; i<getSavings.Total(); i++)
     {
      dayLight = getSavings.At(i);
      if(Year==Time.ReturnYear(dayLight.StartDate))//Check if a certain year's date is available within the DaylightSavings start dates in the List
        {
         startDate = dayLight.StartDate;
         startDateDetected = true;
        }
      if(Year==Time.ReturnYear(dayLight.EndDate))//Check if a certain year's date is available within the DaylightSavings end dates in the List
        {
         endDate = dayLight.EndDate;
         endDateDetected = true;
        }
      if(startDateDetected&&endDateDetected)//Check if both DaylightSavings start and end dates are found for a certain Year
        {
         return true;
        }
     }

   startDate = D'1970.01.01 00:00:00';//Set a default start date if no DaylightSaving date is found
   endDate = D'1970.01.01 00:00:00';//Set a default end date if no DaylightSaving date is found
   return false;
  }

//+------------------------------------------------------------------+
//|Will adjust the date's timezone depending on DaylightSavings      |
//+------------------------------------------------------------------+
string CDaylightSavings::adjustDaylightSavings(datetime EventDate)
  {
   if(isDaylightSavings(TimeTradeServer()))//Check if the current tradeserver time is already within the DaylightSavings Period
     {
      if(isDaylightSavings(EventDate))//Checks if the event time is during daylight savings
        {
         return TimeToString(EventDate);//normal event time
        }
      else
        {
         return TimeToString((datetime)(EventDate-Time.HoursS()));//event time minus an hour for DST
        }
     }
   else
     {
      if(isDaylightSavings(EventDate))//Checks if the event time is during daylight savings
        {
         return TimeToString((datetime)(Time.HoursS()+EventDate));//event time plus an hour for DST
        }
      else
        {
         return TimeToString(EventDate);//normal event time
        }
     }
  }

//+------------------------------------------------------------------+
//|Destructor                                                        |
//+------------------------------------------------------------------+
CDaylightSavings::~CDaylightSavings(void)
  {
   delete savings;//Delete CArrayObj Pointer
   delete dayLight;//Delete CDaylightSavings Pointer
   delete getSavings;//Delete CArrayObj Pointer
  }
//+------------------------------------------------------------------+
```

### CDaylightSavings\_AU Class

In this class we expand upon the virtual void SetDaylightSavings\_AU and proceed to add the Daylight Savings Schedule for Australia.

Australian Daylight Savings Time dates were [found here](https://www.mql5.com/go?link=https://www.timeanddate.com/time/change/australia "https://www.timeanddate.com/time/change/australia").

CDaylightSavings\_AU class has multilevel Inheritance from classes:

- CDaylightSavings
- CObject

CDaylightSavings\_AU class has hierarchical Inheritance from classes:

- CDaylightSavings
- CArrayObj
- CTimeManagement

```
//+------------------------------------------------------------------+
//|                                                      NewsTrading |
//|                                  Copyright 2024, MetaQuotes Ltd. |
//|                            https://www.mql5.com/en/users/kaaiblo |
//+------------------------------------------------------------------+

#include "DaylightSavings.mqh"

//+------------------------------------------------------------------+
//|DaylightSavings_AU class                                          |
//+------------------------------------------------------------------+
class CDaylightSavings_AU: public CDaylightSavings
  {

public:

                     CDaylightSavings_AU(void);
  };

//+------------------------------------------------------------------+
//|Set Daylight Savings Schedule for Australia                       |
//+------------------------------------------------------------------+
void CDaylightSavings::SetDaylightSavings_AU()
  {
   savings = new CArrayObj();
//Daylight savings dates to readjust dates in the database for accurate testing in the strategy tester
   savings.Add(new CDaylightSavings(D'2006.10.29 03:00:00',D'2007.03.25 02:00:00'));
   savings.Add(new CDaylightSavings(D'2007.10.28 03:00:00',D'2008.04.06 02:00:00'));
   savings.Add(new CDaylightSavings(D'2008.10.05 03:00:00',D'2009.04.05 02:00:00'));
   savings.Add(new CDaylightSavings(D'2009.10.04 03:00:00',D'2010.04.04 02:00:00'));
   savings.Add(new CDaylightSavings(D'2010.10.03 03:00:00',D'2011.04.03 02:00:00'));
   savings.Add(new CDaylightSavings(D'2011.10.02 03:00:00',D'2012.04.01 02:00:00'));
   savings.Add(new CDaylightSavings(D'2012.10.07 03:00:00',D'2013.04.07 02:00:00'));
   savings.Add(new CDaylightSavings(D'2013.10.06 03:00:00',D'2014.04.06 02:00:00'));
   savings.Add(new CDaylightSavings(D'2014.10.05 03:00:00',D'2015.04.05 02:00:00'));
   savings.Add(new CDaylightSavings(D'2015.10.04 03:00:00',D'2016.04.03 02:00:00'));
   savings.Add(new CDaylightSavings(D'2016.10.02 03:00:00',D'2017.04.02 02:00:00'));
   savings.Add(new CDaylightSavings(D'2017.10.01 03:00:00',D'2018.04.01 02:00:00'));
   savings.Add(new CDaylightSavings(D'2018.10.07 03:00:00',D'2019.04.07 02:00:00'));
   savings.Add(new CDaylightSavings(D'2019.10.06 03:00:00',D'2020.04.05 02:00:00'));
   savings.Add(new CDaylightSavings(D'2020.10.04 03:00:00',D'2021.04.04 02:00:00'));
   savings.Add(new CDaylightSavings(D'2021.10.03 03:00:00',D'2022.04.03 02:00:00'));
   savings.Add(new CDaylightSavings(D'2022.10.02 03:00:00',D'2023.04.02 02:00:00'));
   savings.Add(new CDaylightSavings(D'2023.10.01 03:00:00',D'2024.04.07 02:00:00'));
   savings.Add(new CDaylightSavings(D'2024.10.06 03:00:00',D'2025.04.06 02:00:00'));
   savings.Add(new CDaylightSavings(D'2025.10.05 03:00:00',D'2026.04.05 02:00:00'));
   savings.Add(new CDaylightSavings(D'2026.10.04 03:00:00',D'2027.04.04 02:00:00'));
   savings.Add(new CDaylightSavings(D'2027.10.03 03:00:00',D'2028.04.02 02:00:00'));
   savings.Add(new CDaylightSavings(D'2028.10.01 03:00:00',D'2029.04.01 02:00:00'));
  }

//+------------------------------------------------------------------+
//|Constructor                                                       |
//+------------------------------------------------------------------+
CDaylightSavings_AU::CDaylightSavings_AU(void)
  {
   SetDaylightSavings_AU();
  }
//+------------------------------------------------------------------+
```

### CDaylightSavings\_UK Class

In this class we expand upon the virtual void SetDaylightSavings\_UK and proceed to add the Daylight Savings Schedule for Europe.

United Kingdom Daylight Savings Time dates were [found here](https://www.mql5.com/go?link=https://www.timeanddate.com/time/change/uk "https://www.timeanddate.com/time/change/uk").

CDaylightSavings\_UK class has multilevel Inheritance from classes:

- CDaylightSavings
- CObject

CDaylightSavings\_UK class has hierarchical Inheritance from classes:

- CDaylightSavings
- CArrayObj
- CTimeManagement

```
//+------------------------------------------------------------------+
//|                                                      NewsTrading |
//|                                  Copyright 2024, MetaQuotes Ltd. |
//|                            https://www.mql5.com/en/users/kaaiblo |
//+------------------------------------------------------------------+

#include "DaylightSavings.mqh"

//+------------------------------------------------------------------+
//|DaylightSavings_UK class                                          |
//+------------------------------------------------------------------+
class CDaylightSavings_UK: public CDaylightSavings
  {

public:

                     CDaylightSavings_UK(void);
  };

//+------------------------------------------------------------------+
//|Set Daylight Savings Schedule for Europe                          |
//+------------------------------------------------------------------+
void CDaylightSavings::SetDaylightSavings_UK()
  {
   savings = new CArrayObj();
//Daylight savings dates to readjust dates in the database for accurate testing in the strategy tester
   savings.Add(new CDaylightSavings(D'2007.03.25 02:00:00',D'2007.10.28 01:00:00'));
   savings.Add(new CDaylightSavings(D'2008.03.30 02:00:00',D'2008.10.26 01:00:00'));
   savings.Add(new CDaylightSavings(D'2009.03.29 02:00:00',D'2009.10.25 01:00:00'));
   savings.Add(new CDaylightSavings(D'2010.03.28 02:00:00',D'2010.10.31 01:00:00'));
   savings.Add(new CDaylightSavings(D'2011.03.27 02:00:00',D'2011.10.30 01:00:00'));
   savings.Add(new CDaylightSavings(D'2012.03.25 02:00:00',D'2012.10.28 01:00:00'));
   savings.Add(new CDaylightSavings(D'2013.03.31 02:00:00',D'2013.10.27 01:00:00'));
   savings.Add(new CDaylightSavings(D'2014.03.30 02:00:00',D'2014.10.26 01:00:00'));
   savings.Add(new CDaylightSavings(D'2015.03.29 02:00:00',D'2015.10.25 01:00:00'));
   savings.Add(new CDaylightSavings(D'2016.03.27 02:00:00',D'2016.10.30 01:00:00'));
   savings.Add(new CDaylightSavings(D'2017.03.26 02:00:00',D'2017.10.29 01:00:00'));
   savings.Add(new CDaylightSavings(D'2018.03.25 02:00:00',D'2018.10.28 01:00:00'));
   savings.Add(new CDaylightSavings(D'2019.03.31 02:00:00',D'2019.10.27 01:00:00'));
   savings.Add(new CDaylightSavings(D'2020.03.29 02:00:00',D'2020.10.25 01:00:00'));
   savings.Add(new CDaylightSavings(D'2021.03.28 02:00:00',D'2021.10.31 01:00:00'));
   savings.Add(new CDaylightSavings(D'2022.03.27 02:00:00',D'2022.10.30 01:00:00'));
   savings.Add(new CDaylightSavings(D'2023.03.26 02:00:00',D'2023.10.29 01:00:00'));
   savings.Add(new CDaylightSavings(D'2024.03.31 02:00:00',D'2024.10.27 01:00:00'));
   savings.Add(new CDaylightSavings(D'2025.03.30 02:00:00',D'2025.10.26 01:00:00'));
   savings.Add(new CDaylightSavings(D'2026.03.29 02:00:00',D'2026.10.25 01:00:00'));
   savings.Add(new CDaylightSavings(D'2027.03.28 02:00:00',D'2027.10.31 01:00:00'));
   savings.Add(new CDaylightSavings(D'2028.03.26 02:00:00',D'2028.10.29 01:00:00'));
   savings.Add(new CDaylightSavings(D'2029.03.25 02:00:00',D'2029.10.28 01:00:00'));
  }

//+------------------------------------------------------------------+
//|Constructor                                                       |
//+------------------------------------------------------------------+
CDaylightSavings_UK::CDaylightSavings_UK(void)
  {
   SetDaylightSavings_UK();
  }
//+------------------------------------------------------------------+
```

### CDaylightSavings\_US Class

In this class we expand upon the virtual void SetDaylightSavings\_US and proceed to add the Daylight Savings Schedule for United States.

United States Daylight Savings Time dates were [found here](https://www.mql5.com/go?link=https://www.timeanddate.com/time/change/usa "https://www.timeanddate.com/time/change/usa").

CDaylightSavings\_US class has multilevel Inheritance from classes:

- CDaylightSavings
- CObject

CDaylightSavings\_US class has hierarchical Inheritance from classes:

- CDaylightSavings
- CArrayObj
- CTimeManagement

```
//+------------------------------------------------------------------+
//|                                                      NewsTrading |
//|                                  Copyright 2024, MetaQuotes Ltd. |
//|                            https://www.mql5.com/en/users/kaaiblo |
//+------------------------------------------------------------------+

#include "DaylightSavings.mqh"

//+------------------------------------------------------------------+
//|DaylightSavings_US Class                                          |
//+------------------------------------------------------------------+
class CDaylightSavings_US: public CDaylightSavings
  {

public:

                     CDaylightSavings_US(void);
  };

//+------------------------------------------------------------------+
//|Set Daylight Savings Schedule for the United States               |
//+------------------------------------------------------------------+
void CDaylightSavings::SetDaylightSavings_US()
  {
   savings = new CArrayObj();
//Daylight savings dates to readjust dates in the database for accurate testing in the strategy tester
   savings.Add(new CDaylightSavings(D'2007.03.11 03:00:00',D'2007.11.04 01:00:00'));
   savings.Add(new CDaylightSavings(D'2008.03.09 03:00:00',D'2008.11.02 01:00:00'));
   savings.Add(new CDaylightSavings(D'2009.03.08 03:00:00',D'2009.11.01 01:00:00'));
   savings.Add(new CDaylightSavings(D'2010.03.14 03:00:00',D'2010.11.07 01:00:00'));
   savings.Add(new CDaylightSavings(D'2011.03.13 03:00:00',D'2011.11.06 01:00:00'));
   savings.Add(new CDaylightSavings(D'2012.03.11 03:00:00',D'2012.11.04 01:00:00'));
   savings.Add(new CDaylightSavings(D'2013.03.10 03:00:00',D'2013.11.03 01:00:00'));
   savings.Add(new CDaylightSavings(D'2014.03.09 03:00:00',D'2014.11.02 01:00:00'));
   savings.Add(new CDaylightSavings(D'2015.03.08 03:00:00',D'2015.11.01 01:00:00'));
   savings.Add(new CDaylightSavings(D'2016.03.13 03:00:00',D'2016.11.06 01:00:00'));
   savings.Add(new CDaylightSavings(D'2017.03.12 03:00:00',D'2017.11.05 01:00:00'));
   savings.Add(new CDaylightSavings(D'2018.03.11 03:00:00',D'2018.11.04 01:00:00'));
   savings.Add(new CDaylightSavings(D'2019.03.10 03:00:00',D'2019.11.03 01:00:00'));
   savings.Add(new CDaylightSavings(D'2020.03.08 03:00:00',D'2020.11.01 01:00:00'));
   savings.Add(new CDaylightSavings(D'2021.03.14 03:00:00',D'2021.11.07 01:00:00'));
   savings.Add(new CDaylightSavings(D'2022.03.13 03:00:00',D'2022.11.06 01:00:00'));
   savings.Add(new CDaylightSavings(D'2023.03.12 03:00:00',D'2023.11.05 01:00:00'));
   savings.Add(new CDaylightSavings(D'2024.03.10 03:00:00',D'2024.11.03 01:00:00'));
   savings.Add(new CDaylightSavings(D'2025.03.09 03:00:00',D'2025.11.02 01:00:00'));
   savings.Add(new CDaylightSavings(D'2026.03.08 03:00:00',D'2026.11.01 01:00:00'));
   savings.Add(new CDaylightSavings(D'2027.03.14 03:00:00',D'2027.11.07 01:00:00'));
   savings.Add(new CDaylightSavings(D'2028.03.12 03:00:00',D'2028.11.05 01:00:00'));
   savings.Add(new CDaylightSavings(D'2029.03.11 03:00:00',D'2029.11.04 01:00:00'));
  }

//+------------------------------------------------------------------+
//|Constructor                                                       |
//+------------------------------------------------------------------+
CDaylightSavings_US::CDaylightSavings_US(void)
  {
   SetDaylightSavings_US();
  }
//+------------------------------------------------------------------+
```

## Symbol Properties Class

In this class we will set a symbol in which we would like to retrieve data from. This class will provide an easy and quick method for us to get symbol properties within other classes and therefore removing redundance in our code.

We will select a combination of properties to retrieve the list could be expanded upon but for not the list goes as follows:

- Ask Price
- Bid Price
- Contract Size
- Minimum Volume
- Maximum Volume
- Volume Step
- Volume Limit
- Spread
- Stops Level
- Freeze Level
- Symbol's Time
- Symbol's Normalized Price
- Symbol's Digits
- Symbol's Point
- Symbol's Trade Mode
- Sum of Symbol's Orders' Volume
- Sum of Symbol's Positions' Volume
- Symbol's Currency Base
- Symbol's Currency Profit
- Symbol's Currency Margin
- Symbol's Custom status
- Symbol's Background color

CSymbolProperties class has inclusion from class CSymbolInfo.

```
//+------------------------------------------------------------------+
//|                                                      NewsTrading |
//|                                  Copyright 2024, MetaQuotes Ltd. |
//|                            https://www.mql5.com/en/users/kaaiblo |
//+------------------------------------------------------------------+
#include <Trade/SymbolInfo.mqh>
//+------------------------------------------------------------------+
//|SymbolProperties class                                            |
//+------------------------------------------------------------------+
class CSymbolProperties
  {
private:
   double            ASK;//Store Ask Price
   double            BID;//Store Bid Price
   double            LOTSMIN;//Store Minimum Lotsize
   double            LOTSMAX;//Store Maximum Lotsize
   double            LOTSSTEP;//Store Lotsize Step
   double            LOTSLIMIT;//Store Lotsize Limit(Maximum sum of Volume)
   long              SPREAD;//Store Spread value
   long              STOPLEVEL;//Store Stop level
   long              FREEZELEVEL;//Store Freeze level
   long              TIME;//Store time
   long              DIGITS;//Store Digits
   double            POINT;//Store Point
   double            ORDERSVOLUME;//Store Orders volume
   double            POSITIONSVOLUME;//Store Positions volume
   long              CUSTOM;//Store if Symbol is Custom
   long              BACKGROUND_CLR;//Store Symbol's background color

protected:
   CSymbolInfo       CSymbol;//Creating class CSymbolInfo's Object
   bool              SetSymbolName(string SYMBOL)
     {
      //-- If Symbol's name was successfully set.
      if(!CSymbol.Name((SYMBOL==NULL)?Symbol():SYMBOL))
        {
         Print("Invalid Symbol: ",SYMBOL);
         return false;
        }
      return true;
     }

   //-- Retrieve Symbol's name
   string            GetSymbolName()
     {
      return CSymbol.Name();
     }

public:
                     CSymbolProperties(void);//Constructor
   double            Ask(string SYMBOL=NULL);//Retrieve Ask Price
   double            Bid(string SYMBOL=NULL);//Retrieve Bid Price
   double            ContractSize(string SYMBOL=NULL);//Retrieve Contract Size
   double            LotsMin(string SYMBOL=NULL);//Retrieve Min Volume
   double            LotsMax(string SYMBOL=NULL);//Retrieve Max Volume
   double            LotsStep(string SYMBOL=NULL);//Retrieve Volume Step
   double            LotsLimit(string SYMBOL=NULL);//Retrieve Volume Limit
   int               Spread(string SYMBOL=NULL);//Retrieve Spread
   int               StopLevel(string SYMBOL=NULL);//Retrieve Stop Level
   int               FreezeLevel(string SYMBOL=NULL);//Retrieve Freeze Level
   datetime          Time(string SYMBOL=NULL);//Retrieve Symbol's Time
   //-- Normalize Price
   double            NormalizePrice(const double price,string SYMBOL=NULL);
   int               Digits(string SYMBOL=NULL);//Retrieve Symbol's Digits
   double            Point(string SYMBOL=NULL);//Retrieve Symbol's Point
   ENUM_SYMBOL_TRADE_MODE TradeMode(string SYMBOL=NULL);//Retrieve Symbol's Trade Mode
   double            OrdersVolume(string SYMBOL=NULL);//Retrieve Symbol's Orders Volume
   double            PositionsVolume(string SYMBOL=NULL);//Retrieve Symbol's Positions Volume
   string            CurrencyBase(string SYMBOL=NULL);//Retrieve Symbol's Currency Base
   string            CurrencyProfit(string SYMBOL=NULL);//Retrieve Symbol's Currency Profit
   string            CurrencyMargin(string SYMBOL=NULL);//Retrieve Symbol's Currency Margin
   bool              Custom(string SYMBOL=NULL);//Retrieve Symbol's Custom status
   color             SymbolBackground(string SYMBOL=NULL,bool allow_black=false);//Retrieve Symbol's Background color
  };

//+------------------------------------------------------------------+
//|Constructor                                                       |
//+------------------------------------------------------------------+
//Initializing Variables
CSymbolProperties::CSymbolProperties(void):ASK(0.0),BID(0.0),
   LOTSMIN(0.0),LOTSMAX(0.0),
   LOTSSTEP(0.0),LOTSLIMIT(0.0),DIGITS(0),
   SPREAD(0),STOPLEVEL(0),ORDERSVOLUME(0.0),
   FREEZELEVEL(0),TIME(0),POINT(0.0),POSITIONSVOLUME(0.0),
   CUSTOM(0),BACKGROUND_CLR(0)
  {
  }

//+------------------------------------------------------------------+
//|Retrieve Ask Price                                                |
//+------------------------------------------------------------------+
double CSymbolProperties::Ask(string SYMBOL=NULL)
  {
   if(SetSymbolName(SYMBOL))//Set Symbol
     {
      if(CSymbol.InfoDouble(SYMBOL_ASK,ASK))
        {
         return ASK;
        }
     }
   Print("Unable to retrieve Symbol's Ask Price");
   return 0.0;
  }

//+------------------------------------------------------------------+
//|Retrieve Bid Price                                                |
//+------------------------------------------------------------------+
double CSymbolProperties::Bid(string SYMBOL=NULL)
  {
   if(SetSymbolName(SYMBOL))//Set Symbol
     {
      if(CSymbol.InfoDouble(SYMBOL_BID,BID))
        {
         return BID;
        }
     }
   Print("Unable to retrieve Symbol's Bid Price");
   return 0.0;
  }

//+------------------------------------------------------------------+
//|Retrieve Contract Size                                            |
//+------------------------------------------------------------------+
double CSymbolProperties::ContractSize(string SYMBOL=NULL)
  {
   if(SetSymbolName(SYMBOL))//Set Symbol
     {
      if(CSymbol.Refresh())
        {
         return CSymbol.ContractSize();
        }
     }
   Print("Unable to retrieve Symbol's Contract size");
   return 0.0;
  }

//+------------------------------------------------------------------+
//|Retrieve Min Volume                                               |
//+------------------------------------------------------------------+
double CSymbolProperties::LotsMin(string SYMBOL=NULL)
  {
   if(SetSymbolName(SYMBOL))//Set Symbol
     {
      if(CSymbol.InfoDouble(SYMBOL_VOLUME_MIN,LOTSMIN))
        {
         return LOTSMIN;
        }
     }
   Print("Unable to retrieve Symbol's LotsMin");
   return 0.0;
  }

//+------------------------------------------------------------------+
//|Retrieve Max Volume                                               |
//+------------------------------------------------------------------+
double CSymbolProperties::LotsMax(string SYMBOL=NULL)
  {
   if(SetSymbolName(SYMBOL))//Set Symbol
     {
      if(CSymbol.InfoDouble(SYMBOL_VOLUME_MAX,LOTSMAX))
        {
         return LOTSMAX;
        }
     }
   Print("Unable to retrieve Symbol's LotsMax");
   return 0.0;
  }

//+------------------------------------------------------------------+
//|Retrieve Volume Step                                              |
//+------------------------------------------------------------------+
double CSymbolProperties::LotsStep(string SYMBOL=NULL)
  {
   if(SetSymbolName(SYMBOL))//Set Symbol
     {
      if(CSymbol.InfoDouble(SYMBOL_VOLUME_STEP,LOTSSTEP))
        {
         return LOTSSTEP;
        }
     }
   Print("Unable to retrieve Symbol's LotsStep");
   return 0.0;
  }

//+------------------------------------------------------------------+
//|Retrieve Volume Limit                                             |
//+------------------------------------------------------------------+
double CSymbolProperties::LotsLimit(string SYMBOL=NULL)
  {
   if(SetSymbolName(SYMBOL))//Set Symbol
     {
      if(CSymbol.InfoDouble(SYMBOL_VOLUME_LIMIT,LOTSLIMIT))
        {
         return LOTSLIMIT;
        }
     }
   Print("Unable to retrieve Symbol's LotsLimit");
   return 0.0;
  }

//+------------------------------------------------------------------+
//|Retrieve Spread                                                   |
//+------------------------------------------------------------------+
int CSymbolProperties::Spread(string SYMBOL=NULL)
  {
   if(SetSymbolName(SYMBOL))//Set Symbol
     {
      if(CSymbol.InfoInteger(SYMBOL_SPREAD,SPREAD))
        {
         return int(SPREAD);
        }
     }
   Print("Unable to retrieve Symbol's Spread");
   return 0;
  }

//+------------------------------------------------------------------+
//|Retrieve Stop Level                                               |
//+------------------------------------------------------------------+
int CSymbolProperties::StopLevel(string SYMBOL=NULL)
  {
   if(SetSymbolName(SYMBOL))//Set Symbol
     {
      if(CSymbol.InfoInteger(SYMBOL_TRADE_STOPS_LEVEL,STOPLEVEL))
        {
         return int(STOPLEVEL);
        }
     }
   Print("Unable to retrieve Symbol's StopLevel");
   return 0;
  }

//+------------------------------------------------------------------+
//|Retrieve Freeze Level                                             |
//+------------------------------------------------------------------+
int CSymbolProperties::FreezeLevel(string SYMBOL=NULL)
  {
   if(SetSymbolName(SYMBOL))//Set Symbol
     {
      if(CSymbol.InfoInteger(SYMBOL_TRADE_FREEZE_LEVEL,FREEZELEVEL))
        {
         return int(FREEZELEVEL);
        }
     }
   Print("Unable to retrieve Symbol's FreezeLevel");
   return 0;
  }

//+------------------------------------------------------------------+
//|Retrieve Symbol's Time                                            |
//+------------------------------------------------------------------+
datetime CSymbolProperties::Time(string SYMBOL=NULL)
  {
   if(SetSymbolName(SYMBOL))//Set Symbol
     {
      if(CSymbol.InfoInteger(SYMBOL_TIME,TIME))
        {
         return datetime(TIME);
        }
     }
   Print("Unable to retrieve Symbol's Time");
   TIME=0;
   return datetime(TIME);
  }

//+------------------------------------------------------------------+
//|Normalize Price                                                   |
//+------------------------------------------------------------------+
double CSymbolProperties::NormalizePrice(const double price,string SYMBOL=NULL)
  {
   if(SetSymbolName(SYMBOL))//Set Symbol
     {
      if(CSymbol.Refresh()&&CSymbol.RefreshRates())
        {
         return CSymbol.NormalizePrice(price);
        }
     }
   Print("Unable to Normalize Symbol's Price");
   return price;
  }

//+------------------------------------------------------------------+
//|Retrieve Symbol's Digits                                          |
//+------------------------------------------------------------------+
int CSymbolProperties::Digits(string SYMBOL=NULL)
  {
   if(SetSymbolName(SYMBOL))//Set Symbol
     {
      if(CSymbol.InfoInteger(SYMBOL_DIGITS,DIGITS))
        {
         return int(DIGITS);
        }
     }
   Print("Unable to retrieve Symbol's Digits");
   return 0;
  }

//+------------------------------------------------------------------+
//|Retrieve Symbol's Point                                           |
//+------------------------------------------------------------------+
double CSymbolProperties::Point(string SYMBOL=NULL)
  {
   if(SetSymbolName(SYMBOL))//Set Symbol
     {
      if(CSymbol.InfoDouble(SYMBOL_POINT,POINT))
        {
         return POINT;
        }
     }
   Print("Unable to retrieve Symbol's Point");
   return 0.0;
  }

//+------------------------------------------------------------------+
//|Retrieve Symbol's Trade Mode                                      |
//+------------------------------------------------------------------+
ENUM_SYMBOL_TRADE_MODE CSymbolProperties::TradeMode(string SYMBOL=NULL)
  {
   if(SetSymbolName(SYMBOL))//Set Symbol
     {
      if(CSymbol.Refresh())
        {
         return CSymbol.TradeMode();
        }
     }
   Print("Unable to retrieve Symbol's TradeMode");
   return SYMBOL_TRADE_MODE_DISABLED;
  }

//+------------------------------------------------------------------+
//|Retrieve Symbol's Orders Volume                                   |
//+------------------------------------------------------------------+
double CSymbolProperties::OrdersVolume(string SYMBOL=NULL)
  {
   if(SetSymbolName(SYMBOL))//Set Symbol
     {
      for(int i=0; i<OrdersTotal(); i++)
        {
         if(OrderSelect(OrderGetTicket(i)))
           {
            if(OrderGetString(ORDER_SYMBOL)==GetSymbolName())
              {
               ORDERSVOLUME+=OrderGetDouble(ORDER_VOLUME_CURRENT);
              }
           }
        }
     }
   else
     {
      Print("Unable to retrieve Symbol's OrdersVolume");
      return 0.0;
     }
   return ORDERSVOLUME;
  }

//+------------------------------------------------------------------+
//|Retrieve Symbol's Positions Volume                                |
//+------------------------------------------------------------------+
double CSymbolProperties::PositionsVolume(string SYMBOL=NULL)
  {
   if(SetSymbolName(SYMBOL))//Set Symbol
     {
      for(int i=0; i<PositionsTotal(); i++)
        {
         if(PositionGetTicket(i)>0)
           {
            if(PositionGetString(POSITION_SYMBOL)==GetSymbolName())
              {
               POSITIONSVOLUME+=PositionGetDouble(POSITION_VOLUME);
              }
           }
        }
     }
   else
     {
      Print("Unable to retrieve Symbol's PositionsVolume");
      return 0.0;
     }
   return POSITIONSVOLUME;
  }

//+------------------------------------------------------------------+
//|Retrieve Symbol's Currency Base                                   |
//+------------------------------------------------------------------+
string CSymbolProperties::CurrencyBase(string SYMBOL=NULL)
  {
   if(SetSymbolName(SYMBOL))//Set Symbol
     {
      if(CSymbol.Refresh())
        {
         return CSymbol.CurrencyBase();
        }
     }
   Print("Unable to retrieve Symbol's CurrencyBase");
   return "";
  }
//+------------------------------------------------------------------+
//|Retrieve Symbol's Currency Profit                                 |
//+------------------------------------------------------------------+
string CSymbolProperties::CurrencyProfit(string SYMBOL=NULL)
  {
   if(SetSymbolName(SYMBOL))//Set Symbol
     {
      if(CSymbol.Refresh())
        {
         return CSymbol.CurrencyProfit();
        }
     }
   Print("Unable to retrieve Symbol's CurrencyProfit");
   return "";
  }
//+------------------------------------------------------------------+
//|Retrieve Symbol's Currency Margin                                 |
//+------------------------------------------------------------------+
string CSymbolProperties::CurrencyMargin(string SYMBOL=NULL)
  {
   if(SetSymbolName(SYMBOL))//Set Symbol
     {
      if(CSymbol.Refresh())
        {
         return CSymbol.CurrencyMargin();
        }
     }
   Print("Unable to retrieve Symbol's CurrencyMargin");
   return "";
  }

//+------------------------------------------------------------------+
//|Retrieve Symbol's Custom status                                   |
//+------------------------------------------------------------------+
bool CSymbolProperties::Custom(string SYMBOL=NULL)
  {
   if(SetSymbolName(SYMBOL))//Set Symbol
     {
      if(CSymbol.InfoInteger(SYMBOL_CUSTOM,CUSTOM))
        {
         return bool(CUSTOM);
        }
     }
   Print("Unable to retrieve if Symbol is Custom");
   return false;
  }

//+------------------------------------------------------------------+
//|Retrieve Symbol's Background color                                |
//+------------------------------------------------------------------+
color CSymbolProperties::SymbolBackground(string SYMBOL=NULL,bool allow_black=false)
  {
   if(SetSymbolName(SYMBOL))//Set Symbol
     {
      if(CSymbol.InfoInteger(SYMBOL_BACKGROUND_COLOR,BACKGROUND_CLR))
        {
         /*Avoid any Symbol black background color */
         BACKGROUND_CLR = ((ColorToString(color(BACKGROUND_CLR))=="0,0,0"||
                            color(BACKGROUND_CLR)==clrBlack)&&!allow_black)?
                          long(StringToColor("236,236,236")):BACKGROUND_CLR;
         return color(BACKGROUND_CLR);
        }
     }
   Print("Unable to retrieve Symbol's Background color");
   return color(StringToColor("236,236,236"));//Retrieve a lightish gray color
  }
//+------------------------------------------------------------------+
```

In the class's constructor below, we initialize the variables we declared earlier such as the double variable ASK, and we assign ASK the value 0.0 as we don't yet have the Ask price for the Symbol.

```
//Initializing Variables
CSymbolProperties::CSymbolProperties(void):ASK(0.0),BID(0.0),
   LOTSMIN(0.0),LOTSMAX(0.0),
   LOTSSTEP(0.0),LOTSLIMIT(0.0),DIGITS(0),
   SPREAD(0),STOPLEVEL(0),ORDERSVOLUME(0.0),
   FREEZELEVEL(0),TIME(0),POINT(0.0),POSITIONSVOLUME(0.0),
   CUSTOM(0),BACKGROUND_CLR(0)
  {
  }
```

In the code below we go through and order of steps to finally retrieve the Symbol's ask price.

![Ask Price](https://c.mql5.com/2/78/Ask_Price.png)

1. Firstly we have an optional parameter where we can decide to input or edit the Symbol's name in the variable SYMBOL.

2. We then set the Symbol's name to the parameter value. If the parameter value is still the default value of NULL we will assume that we want the Symbol's properties for the current chart's Symbol - Symbol().

3. If we cannot find the Symbol's name we will Print an error message to notify the user that the Symbol's Ask price cannot be retrieved and return 0.0

4. Once we were able to set the Symbol's name we will retrieve the ask price for that specific Symbol and return the value.

```
double CSymbolProperties::Ask(string SYMBOL=NULL)
  {
   if(SetSymbolName(SYMBOL))
     {
      if(CSymbol.InfoDouble(SYMBOL_ASK,ASK))
        {
         return ASK;
        }
     }
   Print("Unable to retrieve Symbol's Ask Price");
   return 0.0;
  }
```

```
   bool              SetSymbolName(string SYMBOL)
     {
      //-- If Symbol's name was successfully set.
      if(!CSymbol.Name((SYMBOL==NULL)?Symbol():SYMBOL))
        {
         Print("Invalid Symbol: ",SYMBOL);
         return false;
        }
      return true;
     }
```

The function below will retrieve the Symbol's Bid Price based of the SYMBOL variable.

![Bid Price](https://c.mql5.com/2/78/Bid_Price.png)

```
double CSymbolProperties::Bid(string SYMBOL=NULL)
  {
   if(SetSymbolName(SYMBOL))//Set Symbol
     {
      if(CSymbol.InfoDouble(SYMBOL_BID,BID))
        {
         return BID;
        }
     }
   Print("Unable to retrieve Symbol's Bid Price");
   return 0.0;
  }
```

The function below will retrieve the Symbol's Contract size. A Symbol's contract size affects the trader as a higher contract size increases risk on the individual's trades. Whereas a lower contract size will decrease risk on an individual's trades.

![Contract size](https://c.mql5.com/2/79/Contract_Size.png)

```
double CSymbolProperties::ContractSize(string SYMBOL=NULL)
  {
   if(SetSymbolName(SYMBOL))//Set Symbol
     {
      if(CSymbol.Refresh())
        {
         return CSymbol.ContractSize();
        }
     }
   Print("Unable to retrieve Symbol's Contract size");
   return 0.0;
  }
```

The function below will retrieve the Symbol's Minimum allowed lot-size/volume. Meaning the trader cannot open a position with a lot-size less than the minimum.

![Minimum volume](https://c.mql5.com/2/78/Minimal_Volume.png)

```
double CSymbolProperties::LotsMin(string SYMBOL=NULL)
  {
   if(SetSymbolName(SYMBOL))//Set Symbol
     {
      if(CSymbol.InfoDouble(SYMBOL_VOLUME_MIN,LOTSMIN))
        {
         return LOTSMIN;
        }
     }
   Print("Unable to retrieve Symbol's LotsMin");
   return 0.0;
  }
```

The function below will retrieve the Symbol's Maximum allowed lot-size/volume. This implies that the trader will not be allowed to open a position with a higher lot-size/volume than the maximum, but they may open multiple positions that could sum up to a higher lot-size/volume than the maximum depending on the broker's volume limit and account's orders limit.

![Maximum Volume](https://c.mql5.com/2/78/Maximum_Volume.png)

```
double CSymbolProperties::LotsMax(string SYMBOL=NULL)
  {
   if(SetSymbolName(SYMBOL))//Set Symbol
     {
      if(CSymbol.InfoDouble(SYMBOL_VOLUME_MAX,LOTSMAX))
        {
         return LOTSMAX;
        }
     }
   Print("Unable to retrieve Symbol's LotsMax");
   return 0.0;
  }
```

The function below will retrieve the Symbol's Volume/lot-size Step. Meaning that the lot-size should have an interval of this value. Example if the volume step is 1 the trader cannot select a lot-size/volume of 1.5.

![Volume Step](https://c.mql5.com/2/78/Volume_Step.png)

```
double CSymbolProperties::LotsStep(string SYMBOL=NULL)
  {
   if(SetSymbolName(SYMBOL))//Set Symbol
     {
      if(CSymbol.InfoDouble(SYMBOL_VOLUME_STEP,LOTSSTEP))
        {
         return LOTSSTEP;
        }
     }
   Print("Unable to retrieve Symbol's LotsStep");
   return 0.0;
  }
```

The function below will retrieve the Symbol's Volume/lot-size Limit. This is the sum of volume/lot-sizes that are allowed to be placed before restrictions are implemented on the trader's account for the specific Symbol.

![Volume Limit](https://c.mql5.com/2/78/Volume_Limit.png)

```
double CSymbolProperties::LotsLimit(string SYMBOL=NULL)
  {
   if(SetSymbolName(SYMBOL))//Set Symbol
     {
      if(CSymbol.InfoDouble(SYMBOL_VOLUME_LIMIT,LOTSLIMIT))
        {
         return LOTSLIMIT;
        }
     }
   Print("Unable to retrieve Symbol's LotsLimit");
   return 0.0;
  }
```

The function below will retrieve the Symbol's Spread. The spread affects traders as the higher the Symbol's spread is the less profitable the trader will be, depending on the spread, a strategy could be profitable or not, obviously there are many different circumstances that could make a strategy unprofitable but the Symbol's spread could be a significant circumstance. Spreads are essentially a form of income for your broker, you could think of it as the broker's tax for that Symbol.

![Spread](https://c.mql5.com/2/78/Spread.png)

```
int CSymbolProperties::Spread(string SYMBOL=NULL)
  {
   if(SetSymbolName(SYMBOL))//Set Symbol
     {
      if(CSymbol.InfoInteger(SYMBOL_SPREAD,SPREAD))
        {
         return int(SPREAD);
        }
     }
   Print("Unable to retrieve Symbol's Spread");
   return 0;
  }
```

The function below will retrieve the Symbol's Stops level. This is a restriction on the minimum distance between an open-price and stoploss or take-profit as well as the minimum distance between the current price Ask or Bid and the price of opening an order.

![Stops Level](https://c.mql5.com/2/78/Stops_level.png)

```
int CSymbolProperties::StopLevel(string SYMBOL=NULL)
  {
   if(SetSymbolName(SYMBOL))//Set Symbol
     {
      if(CSymbol.InfoInteger(SYMBOL_TRADE_STOPS_LEVEL,STOPLEVEL))
        {
         return int(STOPLEVEL);
        }
     }
   Print("Unable to retrieve Symbol's StopLevel");
   return 0;
  }
```

The function below will retrieve the Symbol's Freeze level. This is the minimum distance a price has to move from it's open-price for the trade to be allowed closure(when you are allowed to close the specific trade in a profit or loss).

```
int CSymbolProperties::FreezeLevel(string SYMBOL=NULL)
  {
   if(SetSymbolName(SYMBOL))//Set Symbol
     {
      if(CSymbol.InfoInteger(SYMBOL_TRADE_FREEZE_LEVEL,FREEZELEVEL))
        {
         return int(FREEZELEVEL);
        }
     }
   Print("Unable to retrieve Symbol's FreezeLevel");
   return 0;
  }
```

The function below will retrieve the Symbol's Time.

![Time](https://c.mql5.com/2/78/Time.png)

```
datetime CSymbolProperties::Time(string SYMBOL=NULL)
  {
   if(SetSymbolName(SYMBOL))//Set Symbol
     {
      if(CSymbol.InfoInteger(SYMBOL_TIME,TIME))
        {
         return datetime(TIME);
        }
     }
   Print("Unable to retrieve Symbol's Time");
   TIME=0;
   return datetime(TIME);
  }
```

The function below will attempt to normalize a price for a specific symbol.

For example if the Ask price for EURUSD is 1.07735 and you try to open a buy trade at a price level 1.077351. You may get an error like invalid price, as the number of decimal digits are more than the allowed number for example 5 digits. This function will take the price with 6 digits and convert it into 5 digits, therefore normalizing the price.

```
double CSymbolProperties::NormalizePrice(const double price,string SYMBOL=NULL)
  {
   if(SetSymbolName(SYMBOL))//Set Symbol
     {
      if(CSymbol.Refresh()&&CSymbol.RefreshRates())
        {
         return CSymbol.NormalizePrice(price);
        }
     }
   Print("Unable to Normalize Symbol's Price");
   return price;
  }
```

The function below will retrieve the Symbol's Digits. The Digits are represented as the Symbol's price's decimal places.

![Digits](https://c.mql5.com/2/78/Digits.png)

![Symbol's Decimal places are 3](https://c.mql5.com/2/79/DecimalPlaces.png)

```
int CSymbolProperties::Digits(string SYMBOL=NULL)
  {
   if(SetSymbolName(SYMBOL))//Set Symbol
     {
      if(CSymbol.InfoInteger(SYMBOL_DIGITS,DIGITS))
        {
         return int(DIGITS);
        }
     }
   Print("Unable to retrieve Symbol's Digits");
   return 0;
  }
```

The function below will retrieve the Symbol's Point.

```
double CSymbolProperties::Point(string SYMBOL=NULL)
  {
   if(SetSymbolName(SYMBOL))//Set Symbol
     {
      if(CSymbol.InfoDouble(SYMBOL_POINT,POINT))
        {
         return POINT;
        }
     }
   Print("Unable to retrieve Symbol's Point");
   return 0.0;
  }
```

The function below will retrieve the Symbol's Trade Mode.

![Trade Mode](https://c.mql5.com/2/78/TradeMode.png)

```
ENUM_SYMBOL_TRADE_MODE CSymbolProperties::TradeMode(string SYMBOL=NULL)
  {
   if(SetSymbolName(SYMBOL))//Set Symbol
     {
      if(CSymbol.Refresh())
        {
         return CSymbol.TradeMode();
        }
     }
   Print("Unable to retrieve Symbol's TradeMode");
   return SYMBOL_TRADE_MODE_DISABLED;
  }
```

The function below will retrieve the sum of the Symbol's Orders' volume/lots.

```
double CSymbolProperties::OrdersVolume(string SYMBOL=NULL)
  {
   if(SetSymbolName(SYMBOL))//Set Symbol
     {
      for(int i=0; i<OrdersTotal(); i++)
        {
         if(OrderSelect(OrderGetTicket(i)))
           {
            if(OrderGetString(ORDER_SYMBOL)==GetSymbolName())
              {
               ORDERSVOLUME+=OrderGetDouble(ORDER_VOLUME_CURRENT);
              }
           }
        }
     }
   else
     {
      Print("Unable to retrieve Symbol's OrdersVolume");
      return 0.0;
     }
   return ORDERSVOLUME;
  }
```

The function below will retrieve the sum of the Symbol's Positions' volume/lots.

```
double CSymbolProperties::PositionsVolume(string SYMBOL=NULL)
  {
   if(SetSymbolName(SYMBOL))//Set Symbol
     {
      for(int i=0; i<PositionsTotal(); i++)
        {
         if(PositionGetTicket(i)>0)
           {
            if(PositionGetString(POSITION_SYMBOL)==GetSymbolName())
              {
               POSITIONSVOLUME+=PositionGetDouble(POSITION_VOLUME);
              }
           }
        }
     }
   else
     {
      Print("Unable to retrieve Symbol's PositionsVolume");
      return 0.0;
     }
   return POSITIONSVOLUME;
  }
```

The function below will retrieve the Symbol's Base Currency.

```
string CSymbolProperties::CurrencyBase(string SYMBOL=NULL)
  {
   if(SetSymbolName(SYMBOL))//Set Symbol
     {
      if(CSymbol.Refresh())
        {
         return CSymbol.CurrencyBase();
        }
     }
   Print("Unable to retrieve Symbol's CurrencyBase");
   return "";
  }
```

The function below will retrieve the Symbol's Profit Currency.

![Profit Currency](https://c.mql5.com/2/78/Profit_Currency.png)

```
string CSymbolProperties::CurrencyProfit(string SYMBOL=NULL)
  {
   if(SetSymbolName(SYMBOL))//Set Symbol
     {
      if(CSymbol.Refresh())
        {
         return CSymbol.CurrencyProfit();
        }
     }
   Print("Unable to retrieve Symbol's CurrencyProfit");
   return "";
  }
```

The function below will retrieve the Symbol's Margin Currency.

![Margin Currency](https://c.mql5.com/2/78/Margin_Currency.png)

```
string CSymbolProperties::CurrencyMargin(string SYMBOL=NULL)
  {
   if(SetSymbolName(SYMBOL))//Set Symbol
     {
      if(CSymbol.Refresh())
        {
         return CSymbol.CurrencyMargin();
        }
     }
   Print("Unable to retrieve Symbol's CurrencyMargin");
   return "";
  }
```

The function below will retrieve a Boolean value to identify if a Symbol is custom or not.

```
bool CSymbolProperties::Custom(string SYMBOL=NULL)
  {
   if(SetSymbolName(SYMBOL))//Set Symbol
     {
      if(CSymbol.InfoInteger(SYMBOL_CUSTOM,CUSTOM))
        {
         return bool(CUSTOM);
        }
     }
   Print("Unable to retrieve if Symbol is Custom");
   return false;
  }
```

The function below will retrieve the Symbol's background color. And has an optional parameter allow\_black which is false by default, this is because we will use the Symbol's background color to set the chart's background color later on and we do want to retrieve a black color as other elements of our chart will be in black. If we allowed the black color, based on our anticipated chart format it would cause the chart to be distracting and unreadable.

Example of black background chart with our new chart format which will be established later.

![Black background](https://c.mql5.com/2/78/Incorrect_chart_format.png)

```
color CSymbolProperties::SymbolBackground(string SYMBOL=NULL,bool allow_black=false)
  {
   if(SetSymbolName(SYMBOL))//Set Symbol
     {
      if(CSymbol.InfoInteger(SYMBOL_BACKGROUND_COLOR,BACKGROUND_CLR))
        {
        /*Avoid any Symbol black background color */
         BACKGROUND_CLR = ((ColorToString(color(BACKGROUND_CLR))=="0,0,0"||
                            color(BACKGROUND_CLR)==clrBlack)&&!allow_black)?
                          long(StringToColor("236,236,236")):BACKGROUND_CLR;
         return color(BACKGROUND_CLR);
        }
     }
   Print("Unable to retrieve Symbol's Background color");
   return color(StringToColor("236,236,236"));//Retrieve a lightish gray color
  }
```

## Time Management Class

In this class, I will highlight the new functions added to the class's functionality. The purpose of this class is to manipulate and or interact with time data.

```
//+------------------------------------------------------------------+
//|                                                      NewsTrading |
//|                                  Copyright 2024, MetaQuotes Ltd. |
//|                            https://www.mql5.com/en/users/kaaiblo |
//+------------------------------------------------------------------+

//+------------------------------------------------------------------+
//|TimeManagement class                                              |
//+------------------------------------------------------------------+
class CTimeManagement
  {

private:

   MqlDateTime       today;//private variable
   MqlDateTime       timeFormat;//private variable

public:

   //-- Checks if a date is within two other dates
   bool              DateIsInRange(datetime FirstTime,datetime SecondTime,datetime compareTime);
   //-- Check if two dates(Start&End) are within CompareStart & CompareEnd
   bool              DateIsInRange(datetime Start,datetime End,datetime CompareStart,datetime CompareEnd);
   bool              DateisToday(datetime TimeRepresented);//Checks if a date is within the current day
   int               SecondsS(int multiple=1);//Returns seconds
   int               MinutesS(int multiple=1);//Returns Minutes in seconds
   int               HoursS(int multiple=1);//Returns Hours in seconds
   int               DaysS(int multiple=1);//Returns Days in seconds
   int               WeeksS(int multiple=1);//Returns Weeks in seconds
   int               MonthsS(int multiple=1);//Returns Months in seconds
   int               YearsS(int multiple=1);//Returns Years in seconds
   int               ReturnYear(datetime time);//Returns the Year for a specific date
   int               ReturnMonth(datetime time);//Returns the Month for a specific date
   int               ReturnDay(datetime time);//Returns the Day for a specific date
   //-- Will return a datetime type of a date with an subtraction offset in seconds
   datetime          TimeMinusOffset(datetime standardtime,int timeoffset);
   //-- Will return a datetime type of a date with an addition offset in seconds
   datetime          TimePlusOffset(datetime standardtime,int timeoffset);
  };

//+------------------------------------------------------------------+
//|Checks if a date is within two other dates                        |
//+------------------------------------------------------------------+
bool CTimeManagement::DateIsInRange(datetime FirstTime,datetime SecondTime,datetime compareTime)
  {
   return(FirstTime<=compareTime&&SecondTime>compareTime);
  }

//+------------------------------------------------------------------+
//|Check if two dates(Start&End) are within CompareStart & CompareEnd|
//+------------------------------------------------------------------+
bool CTimeManagement::DateIsInRange(datetime Start,datetime End,datetime CompareStart,datetime CompareEnd)
  {
   return(Start<=CompareStart&&CompareEnd<End);
  }

//+------------------------------------------------------------------+
//|Checks if a date is within the current day                        |
//+------------------------------------------------------------------+
bool CTimeManagement::DateisToday(datetime TimeRepresented)
  {
   MqlDateTime TiM;
   TimeToStruct(TimeRepresented,TiM);
   TimeCurrent(today);
   return(TiM.year==today.year&&TiM.mon==today.mon&&TiM.day==today.day);
  }

//+------------------------------------------------------------------+
//|Returns seconds                                                   |
//+------------------------------------------------------------------+
int CTimeManagement::SecondsS(int multiple=1)
  {
   return (1*multiple);
  }

//+------------------------------------------------------------------+
//|Returns Minutes in seconds                                        |
//+------------------------------------------------------------------+
int CTimeManagement::MinutesS(int multiple=1)
  {
   return (SecondsS(60)*multiple);
  }

//+------------------------------------------------------------------+
//|Returns Hours in seconds                                          |
//+------------------------------------------------------------------+
int CTimeManagement::HoursS(int multiple=1)
  {
   return (MinutesS(60)*multiple);
  }

//+------------------------------------------------------------------+
//|Returns Days in seconds                                           |
//+------------------------------------------------------------------+
int CTimeManagement::DaysS(int multiple=1)
  {
   return (HoursS(24)*multiple);
  }

//+------------------------------------------------------------------+
//|Returns Weeks in seconds                                          |
//+------------------------------------------------------------------+
int CTimeManagement::WeeksS(int multiple=1)
  {
   return (DaysS(7)*multiple);
  }

//+------------------------------------------------------------------+
//|Returns Months in seconds                                         |
//+------------------------------------------------------------------+
int CTimeManagement::MonthsS(int multiple=1)
  {
   return (WeeksS(4)*multiple);
  }

//+------------------------------------------------------------------+
//|Returns Years in seconds                                          |
//+------------------------------------------------------------------+
int CTimeManagement::YearsS(int multiple=1)
  {
   return (MonthsS(12)*multiple);
  }

//+------------------------------------------------------------------+
//|Returns the Year for a specific date                              |
//+------------------------------------------------------------------+
int CTimeManagement::ReturnYear(datetime time)
  {
   TimeToStruct(time,timeFormat);
   return timeFormat.year;
  }

//+------------------------------------------------------------------+
//|Returns the Month for a specific date                             |
//+------------------------------------------------------------------+
int CTimeManagement::ReturnMonth(datetime time)
  {
   TimeToStruct(time,timeFormat);
   return timeFormat.mon;
  }

//+------------------------------------------------------------------+
//|Returns the Day for a specific date                               |
//+------------------------------------------------------------------+
int CTimeManagement::ReturnDay(datetime time)
  {
   TimeToStruct(time,timeFormat);
   return timeFormat.day;
  }

//+------------------------------------------------------------------+
//|Will return a datetime type of a date with an subtraction offset  |
//|in seconds                                                        |
//+------------------------------------------------------------------+
datetime CTimeManagement::TimeMinusOffset(datetime standardtime,int timeoffset)
  {
   standardtime-=timeoffset;
   return standardtime;
  }

//+------------------------------------------------------------------+
//|Will return a datetime type of a date with an addition offset     |
//|in seconds                                                        |
//+------------------------------------------------------------------+
datetime CTimeManagement::TimePlusOffset(datetime standardtime,int timeoffset)
  {
   standardtime+=timeoffset;
   return standardtime;
  }
//+------------------------------------------------------------------+
```

## Chart Properties Class

The purpose of chart properties is to store the chart configuration before we change the layout of the chart. Once the expert is removed from the chart, the class's destructor will restore the charts configuration's state before the changes we would have made to the chart.

Making changes to the chart is not necessary for the experts functionality, but it is visually pleasing to have a chart the isn't just green and black(default chart layout) and may be hard to make out chart prices and or trade levels once trades are placed.

CChartProperties has a singular Inheritance from class CSymbolProperties.

CChartProperties has hierarchical Inheritance from classes:

- CSymbolProperties

- CSymbolInfo

```
//+------------------------------------------------------------------+
//|                                                      NewsTrading |
//|                                  Copyright 2024, MetaQuotes Ltd. |
//|                            https://www.mql5.com/en/users/kaaiblo |
//+------------------------------------------------------------------+
#include "SymbolProperties.mqh"
//+------------------------------------------------------------------+
//|ChartProperties class                                             |
//+------------------------------------------------------------------+
class CChartProperties : public CSymbolProperties
  {
private:
   struct ChartFormat
     {
      ulong             CHART_MODE;//Chart Candle Mode
      ulong             CHART_COLOR_BACKGROUND;//Chart Background Color
      ulong             CHART_COLOR_FOREGROUND;//Chart Foreground Color
      ulong             CHART_COLOR_CHART_LINE;//Chart Line Color
      ulong             CHART_COLOR_CANDLE_BEAR;//Chart Bear Candle Color
      ulong             CHART_COLOR_CHART_DOWN;//Chart Down Candle Color
      ulong             CHART_COLOR_CANDLE_BULL;//Chart Bull Candle Color
      ulong             CHART_COLOR_CHART_UP;//Chart Up Candle Color
      ulong             CHART_COLOR_ASK;//Chart Ask Color
      ulong             CHART_COLOR_BID;//Chart Bid Color
      ulong             CHART_COLOR_STOP_LEVEL;//Chart Stoplevel Color
      ulong             CHART_SHOW_PERIOD_SEP;//Chart Show Period Separator
      ulong             CHART_SCALE;//Chart Scale
      ulong             CHART_FOREGROUND;//Chart Show Foreground
      ulong             CHART_SHOW_ASK_LINE;//Chart Show Ask Line
      ulong             CHART_SHOW_BID_LINE;//Chart Show Bid Line
      ulong             CHART_SHOW_TRADE_LEVELS;//Chart Show Trade Levels
      ulong             CHART_SHOW_OHLC;//Chart Show Open-High-Low-Close
      ulong             CHART_SHOW_GRID;//Chart Show Grid
      ulong             CHART_SHOW_VOLUMES;//Chart Show Volumes
      ulong             CHART_AUTOSCROLL;//Chart Auto Scroll
      double            CHART_SHIFT_SIZE;//Chart Shift Size
      ulong             CHART_SHIFT;//Chart Shift
      ulong             CHART_SHOW_ONE_CLICK;//Chart One Click Trading
     };
   ulong             ChartConfig[65];//Array To Store Chart Properties
   void              ChartSet();//Apply Chart format
   void              ChartConfigure();//Set Chart Values
   ChartFormat       Chart;//Variable of type ChartFormat

public:
                     CChartProperties(void);//Constructor
                    ~CChartProperties(void);//Destructor
   void              ChartRefresh() {ChartConfigure();}
  };

//+------------------------------------------------------------------+
//|Constructor                                                       |
//+------------------------------------------------------------------+
CChartProperties::CChartProperties(void)//Class Constructor
  {
   for(int i=0;i<65;i++)//Iterating through ENUM_CHART_PROPERTY_INTEGER Elements
     {
      ChartGetInteger(0,(ENUM_CHART_PROPERTY_INTEGER)i,0,ChartConfig[i]);//Storing Chart values into ChartConfig array
     }
   ChartConfigure();
  }

//+------------------------------------------------------------------+
//|Destructor                                                        |
//+------------------------------------------------------------------+
CChartProperties::~CChartProperties(void)
  {
   for(int i=0;i<65;i++)//Iterating through ENUM_CHART_PROPERTY_INTEGER Elements
     {
      ChartSetInteger(0,(ENUM_CHART_PROPERTY_INTEGER)i,0,ChartConfig[i]);//Restoring Chart values from ChartConfig array
     }
  }

//+------------------------------------------------------------------+
//|Set Chart Properties                                              |
//+------------------------------------------------------------------+
void CChartProperties::ChartSet()
  {
   ChartSetInteger(0,CHART_MODE,Chart.CHART_MODE);//Set Chart Candle Mode
   ChartSetInteger(0,CHART_COLOR_BACKGROUND,Chart.CHART_COLOR_BACKGROUND);//Set Chart Background Color
   ChartSetInteger(0,CHART_COLOR_FOREGROUND,Chart.CHART_COLOR_FOREGROUND);//Set Chart Foreground Color
   ChartSetInteger(0,CHART_COLOR_CHART_LINE,Chart.CHART_COLOR_CHART_LINE);//Set Chart Line Color
   ChartSetInteger(0,CHART_COLOR_CANDLE_BEAR,Chart.CHART_COLOR_CANDLE_BEAR);//Set Chart Bear Candle Color
   ChartSetInteger(0,CHART_COLOR_CHART_DOWN,Chart.CHART_COLOR_CHART_DOWN);//Set Chart Down Candle Color
   ChartSetInteger(0,CHART_COLOR_CANDLE_BULL,Chart.CHART_COLOR_CANDLE_BULL);//Set Chart Bull Candle Color
   ChartSetInteger(0,CHART_COLOR_CHART_UP,Chart.CHART_COLOR_CHART_UP);//Set Chart Up Candle Color
   ChartSetInteger(0,CHART_COLOR_ASK,Chart.CHART_COLOR_ASK);//Set Chart Ask Color
   ChartSetInteger(0,CHART_COLOR_BID,Chart.CHART_COLOR_BID);//Set Chart Bid Color
   ChartSetInteger(0,CHART_COLOR_STOP_LEVEL,Chart.CHART_COLOR_STOP_LEVEL);//Set Chart Stop Level Color
   ChartSetInteger(0,CHART_FOREGROUND,Chart.CHART_FOREGROUND);//Set if Chart is in Foreground Visibility
   ChartSetInteger(0,CHART_SHOW_ASK_LINE,Chart.CHART_SHOW_ASK_LINE);//Set Chart Ask Line Visibility
   ChartSetInteger(0,CHART_SHOW_BID_LINE,Chart.CHART_SHOW_BID_LINE);//Set Chart Bid Line Visibility
   ChartSetInteger(0,CHART_SHOW_PERIOD_SEP,Chart.CHART_SHOW_PERIOD_SEP);//Set Chart Period Separator Visibility
   ChartSetInteger(0,CHART_SHOW_TRADE_LEVELS,Chart.CHART_SHOW_TRADE_LEVELS);//Set Chart Trade Levels Visibility
   ChartSetInteger(0,CHART_SHOW_OHLC,Chart.CHART_SHOW_OHLC);//Set Chart Open-High-Low-Close Visibility
   ChartSetInteger(0,CHART_SHOW_GRID,Chart.CHART_SHOW_GRID);//Set Chart Grid Visibility
   ChartSetInteger(0,CHART_SHOW_VOLUMES,Chart.CHART_SHOW_VOLUMES);//Set Chart Volumes Visibility
   ChartSetInteger(0,CHART_SCALE,Chart.CHART_SCALE);//Set Chart Scale Value
   ChartSetInteger(0,CHART_AUTOSCROLL,Chart.CHART_AUTOSCROLL);//Set Chart Auto Scroll Option
   ChartSetDouble(0,CHART_SHIFT_SIZE,Chart.CHART_SHIFT_SIZE);//Set Chart Shift Size Value
   ChartSetInteger(0,CHART_SHIFT,Chart.CHART_SHIFT);//Set Chart Shift Option
   ChartSetInteger(0,CHART_SHOW_ONE_CLICK,Chart.CHART_SHOW_ONE_CLICK);//Set Chart One Click Trading
  }

//+------------------------------------------------------------------+
//|Initialize Chart Properties                                       |
//+------------------------------------------------------------------+
void CChartProperties::ChartConfigure(void)
  {
   Chart.CHART_MODE=(ulong)CHART_CANDLES;//Assigning Chart Mode of CHART_CANDLES
   Chart.CHART_COLOR_BACKGROUND=ulong(SymbolBackground());//Assigning Chart Background Color of Symbol's Background color
   Chart.CHART_COLOR_FOREGROUND=(ulong)clrBlack;//Assigning Chart Foreground Color of clrBalck(Black color)
   Chart.CHART_COLOR_CHART_LINE=(ulong)clrBlack;//Assigning Chart Line Color of clrBlack(Black color)
   Chart.CHART_COLOR_CANDLE_BEAR=(ulong)clrBlack;//Assigning Chart Bear Candle Color of clrBlack(Black color)
   Chart.CHART_COLOR_CHART_DOWN=(ulong)clrBlack;//Assigning Chart Down Candle Color of clrBlack(Black color)
   Chart.CHART_COLOR_CANDLE_BULL=(ulong)clrWhite;//Assigning Chart Bull Candle Color of clrWhite(White color)
   Chart.CHART_COLOR_CHART_UP=(ulong)clrBlack;//Assigning Chart Up Candle Color of clrBlack(Black color)
   Chart.CHART_COLOR_ASK=(ulong)clrBlack;//Assigning Chart Ask Color of clrBlack(Black color)
   Chart.CHART_COLOR_BID=(ulong)clrBlack;//Assigning Chart Bid Color of clrBlack(Black color)
   Chart.CHART_COLOR_STOP_LEVEL=(ulong)clrBlack;//Assigning Chart Stop Level Color of clrBlack(Black color)
   Chart.CHART_FOREGROUND=(ulong)false;//Assigning Chart Foreground Boolean Value of 'false'
   Chart.CHART_SHOW_ASK_LINE=(ulong)true;//Assigning Chart Ask Line Boolean Value of 'true'
   Chart.CHART_SHOW_BID_LINE=(ulong)true;//Assigning Chart Bid Line Boolean Value of 'true'
   Chart.CHART_SHOW_PERIOD_SEP=(ulong)true;//Assigning Chart Period Separator Boolean Value of 'true'
   Chart.CHART_SHOW_TRADE_LEVELS=(ulong)true;//Assigning Chart Trade Levels Boolean Value of 'true'
   Chart.CHART_SHOW_OHLC=(ulong)false;//Assigning Chart Open-High-Low-Close Boolean Value of 'false'
   Chart.CHART_SHOW_GRID=(ulong)false;//Assigning Chart Grid Boolean Value of 'false'
   Chart.CHART_SHOW_VOLUMES=(ulong)false;//Assigning Chart Volumes Boolean Value of 'false'
   Chart.CHART_SCALE=(ulong)3;//Assigning Chart Scale Boolean Value of '3'
   Chart.CHART_AUTOSCROLL=(ulong)true;//Assigning Chart Auto Scroll Boolean Value of 'true'
   Chart.CHART_SHIFT_SIZE=30;//Assigning Chart Shift Size Value of '30'
   Chart.CHART_SHIFT=(ulong)true;//Assigning Chart Shift Boolean Value of 'true'
   Chart.CHART_SHOW_ONE_CLICK=ulong(false);//Assigning Chart One Click Trading a value of 'false'
   ChartSet();//Calling Function to set chart format
  }
//+------------------------------------------------------------------+
```

In the structure ChartFormat we declared, we store different chart variables that we will change on the current chart the expert is on.

```
   struct ChartFormat
     {
      ulong             CHART_MODE;//Chart Candle Mode
      ulong             CHART_COLOR_BACKGROUND;//Chart Background Color
      ulong             CHART_COLOR_FOREGROUND;//Chart Foreground Color
      ulong             CHART_COLOR_CHART_LINE;//Chart Line Color
      ulong             CHART_COLOR_CANDLE_BEAR;//Chart Bear Candle Color
      ulong             CHART_COLOR_CHART_DOWN;//Chart Down Candle Color
      ulong             CHART_COLOR_CANDLE_BULL;//Chart Bull Candle Color
      ulong             CHART_COLOR_CHART_UP;//Chart Up Candle Color
      ulong             CHART_COLOR_ASK;//Chart Ask Color
      ulong             CHART_COLOR_BID;//Chart Bid Color
      ulong             CHART_COLOR_STOP_LEVEL;//Chart Stoplevel Color
      ulong             CHART_SHOW_PERIOD_SEP;//Chart Show Period Separator
      ulong             CHART_SCALE;//Chart Scale
      ulong             CHART_FOREGROUND;//Chart Show Foreground
      ulong             CHART_SHOW_ASK_LINE;//Chart Show Ask Line
      ulong             CHART_SHOW_BID_LINE;//Chart Show Bid Line
      ulong             CHART_SHOW_TRADE_LEVELS;//Chart Show Trade Levels
      ulong             CHART_SHOW_OHLC;//Chart Show Open-High-Low-Close
      ulong             CHART_SHOW_GRID;//Chart Show Grid
      ulong             CHART_SHOW_VOLUMES;//Chart Show Volumes
      ulong             CHART_AUTOSCROLL;//Chart Auto Scroll
      double            CHART_SHIFT_SIZE;//Chart Shift Size
      ulong             CHART_SHIFT;//Chart Shift
      ulong             CHART_SHOW_ONE_CLICK;//Chart One Click Trading
     };
```

The ChartConfig array will store all the chart properties before we make changes to the chart.

```
ulong             ChartConfig[65];//Array To Store Chart Properties
```

In the Function SetBackground we get the current Symbol's MarketWatch background color:

![MarketWatch](https://c.mql5.com/2/78/MarketWatch_BackgroundColor.png)

and set the current chart's background color to this color:

![Chart Background Color](https://c.mql5.com/2/78/Chart_BackgroundColor.png)

In the class's constructor we get all the chart properties of type Integer and store them into ChartConfig array.

```
CChartProperties::CChartProperties(void)//Class Constructor
  {
   for(int i=0;i<65;i++)//Iterating through ENUM_CHART_PROPERTY_INTEGER Elements
     {
      ChartGetInteger(0,(ENUM_CHART_PROPERTY_INTEGER)i,0,ChartConfig[i]);//Storing Chart values into ChartConfig array
     }
   ChartConfigure();
  }
```

We also initialize the variable Chart which is of type structure ChartFormat previously mentioned and give it the appropriate values to customize the chart to our liking in the function ChartConfigure.

```
void CChartProperties::ChartConfigure(void)
  {
   Chart.CHART_MODE=(ulong)CHART_CANDLES;//Assigning Chart Mode of CHART_CANDLES
   Chart.CHART_COLOR_BACKGROUND=ulong(SymbolBackground());//Assigning Chart Background Color of Symbol's Background color
   Chart.CHART_COLOR_FOREGROUND=(ulong)clrBlack;//Assigning Chart Foreground Color of clrBalck(Black color)
   Chart.CHART_COLOR_CHART_LINE=(ulong)clrBlack;//Assigning Chart Line Color of clrBlack(Black color)
   Chart.CHART_COLOR_CANDLE_BEAR=(ulong)clrBlack;//Assigning Chart Bear Candle Color of clrBlack(Black color)
   Chart.CHART_COLOR_CHART_DOWN=(ulong)clrBlack;//Assigning Chart Down Candle Color of clrBlack(Black color)
   Chart.CHART_COLOR_CANDLE_BULL=(ulong)clrWhite;//Assigning Chart Bull Candle Color of clrWhite(White color)
   Chart.CHART_COLOR_CHART_UP=(ulong)clrBlack;//Assigning Chart Up Candle Color of clrBlack(Black color)
   Chart.CHART_COLOR_ASK=(ulong)clrBlack;//Assigning Chart Ask Color of clrBlack(Black color)
   Chart.CHART_COLOR_BID=(ulong)clrBlack;//Assigning Chart Bid Color of clrBlack(Black color)
   Chart.CHART_COLOR_STOP_LEVEL=(ulong)clrBlack;//Assigning Chart Stop Level Color of clrBlack(Black color)
   Chart.CHART_FOREGROUND=(ulong)false;//Assigning Chart Foreground Boolean Value of 'false'
   Chart.CHART_SHOW_ASK_LINE=(ulong)true;//Assigning Chart Ask Line Boolean Value of 'true'
   Chart.CHART_SHOW_BID_LINE=(ulong)true;//Assigning Chart Bid Line Boolean Value of 'true'
   Chart.CHART_SHOW_PERIOD_SEP=(ulong)true;//Assigning Chart Period Separator Boolean Value of 'true'
   Chart.CHART_SHOW_TRADE_LEVELS=(ulong)true;//Assigning Chart Trade Levels Boolean Value of 'true'
   Chart.CHART_SHOW_OHLC=(ulong)false;//Assigning Chart Open-High-Low-Close Boolean Value of 'false'
   Chart.CHART_SHOW_GRID=(ulong)false;//Assigning Chart Grid Boolean Value of 'false'
   Chart.CHART_SHOW_VOLUMES=(ulong)false;//Assigning Chart Volumes Boolean Value of 'false'
   Chart.CHART_SCALE=(ulong)3;//Assigning Chart Scale Boolean Value of '3'
   Chart.CHART_AUTOSCROLL=(ulong)true;//Assigning Chart Auto Scroll Boolean Value of 'true'
   Chart.CHART_SHIFT_SIZE=30;//Assigning Chart Shift Size Value of '30'
   Chart.CHART_SHIFT=(ulong)true;//Assigning Chart Shift Boolean Value of 'true'
   Chart.CHART_SHOW_ONE_CLICK=ulong(false);//Assigning Chart One Click Trading a value of 'false'
   ChartSet();//Calling Function to set chart format
  }
```

In the ChartSet function will set the values of the selected chart properties from the variable Chart of structure type ChartFormat.

```
void CChartProperties::ChartSet()
  {
   ChartSetInteger(0,CHART_MODE,Chart.CHART_MODE);//Set Chart Candle Mode
   ChartSetInteger(0,CHART_COLOR_BACKGROUND,Chart.CHART_COLOR_BACKGROUND);//Set Chart Background Color
   ChartSetInteger(0,CHART_COLOR_FOREGROUND,Chart.CHART_COLOR_FOREGROUND);//Set Chart Foreground Color
   ChartSetInteger(0,CHART_COLOR_CHART_LINE,Chart.CHART_COLOR_CHART_LINE);//Set Chart Line Color
   ChartSetInteger(0,CHART_COLOR_CANDLE_BEAR,Chart.CHART_COLOR_CANDLE_BEAR);//Set Chart Bear Candle Color
   ChartSetInteger(0,CHART_COLOR_CHART_DOWN,Chart.CHART_COLOR_CHART_DOWN);//Set Chart Down Candle Color
   ChartSetInteger(0,CHART_COLOR_CANDLE_BULL,Chart.CHART_COLOR_CANDLE_BULL);//Set Chart Bull Candle Color
   ChartSetInteger(0,CHART_COLOR_CHART_UP,Chart.CHART_COLOR_CHART_UP);//Set Chart Up Candle Color
   ChartSetInteger(0,CHART_COLOR_ASK,Chart.CHART_COLOR_ASK);//Set Chart Ask Color
   ChartSetInteger(0,CHART_COLOR_BID,Chart.CHART_COLOR_BID);//Set Chart Bid Color
   ChartSetInteger(0,CHART_COLOR_STOP_LEVEL,Chart.CHART_COLOR_STOP_LEVEL);//Set Chart Stop Level Color
   ChartSetInteger(0,CHART_FOREGROUND,Chart.CHART_FOREGROUND);//Set if Chart is in Foreground Visibility
   ChartSetInteger(0,CHART_SHOW_ASK_LINE,Chart.CHART_SHOW_ASK_LINE);//Set Chart Ask Line Visibility
   ChartSetInteger(0,CHART_SHOW_BID_LINE,Chart.CHART_SHOW_BID_LINE);//Set Chart Bid Line Visibility
   ChartSetInteger(0,CHART_SHOW_PERIOD_SEP,Chart.CHART_SHOW_PERIOD_SEP);//Set Chart Period Separator Visibility
   ChartSetInteger(0,CHART_SHOW_TRADE_LEVELS,Chart.CHART_SHOW_TRADE_LEVELS);//Set Chart Trade Levels Visibility
   ChartSetInteger(0,CHART_SHOW_OHLC,Chart.CHART_SHOW_OHLC);//Set Chart Open-High-Low-Close Visibility
   ChartSetInteger(0,CHART_SHOW_GRID,Chart.CHART_SHOW_GRID);//Set Chart Grid Visibility
   ChartSetInteger(0,CHART_SHOW_VOLUMES,Chart.CHART_SHOW_VOLUMES);//Set Chart Volumes Visibility
   ChartSetInteger(0,CHART_SCALE,Chart.CHART_SCALE);//Set Chart Scale Value
   ChartSetInteger(0,CHART_AUTOSCROLL,Chart.CHART_AUTOSCROLL);//Set Chart Auto Scroll Option
   ChartSetDouble(0,CHART_SHIFT_SIZE,Chart.CHART_SHIFT_SIZE);//Set Chart Shift Size Value
   ChartSetInteger(0,CHART_SHIFT,Chart.CHART_SHIFT);//Set Chart Shift Option
   ChartSetInteger(0,CHART_SHOW_ONE_CLICK,Chart.CHART_SHOW_ONE_CLICK);//Set Chart One Click Trading
  }
```

In the destructor we will restore the previous chart's integer values.

```
CChartProperties::~CChartProperties(void)
  {
   for(int i=0;i<65;i++)//Iterating through ENUM_CHART_PROPERTY_INTEGER Elements
     {
      ChartSetInteger(0,(ENUM_CHART_PROPERTY_INTEGER)i,0,ChartConfig[i]);//Restoring Chart values from ChartConfig array
     }
  }
```

## Candle Properties Class

CCandleProperties has multilevel Inheritance from classes:

- CChartProperties
- CSymbolProperties

CCandleProperties has Inclusion from class CTimeManagement.

CCandleProperties has hierarchical Inheritance from classes:

- CSymbolProperties
- CSymbolInfo

```
//+------------------------------------------------------------------+
//|                                                      NewsTrading |
//|                                  Copyright 2024, MetaQuotes Ltd. |
//|                            https://www.mql5.com/en/users/kaaiblo |
//+------------------------------------------------------------------+
#include "TimeManagement.mqh"
#include "ChartProperties.mqh"
//+------------------------------------------------------------------+
//|CandleProperties class                                            |
//+------------------------------------------------------------------+
class CCandleProperties : public CChartProperties
  {
private:
   CTimeManagement   Time;

public:
   double            Open(int CandleIndex,ENUM_TIMEFRAMES Period=PERIOD_CURRENT,string SYMBOL=NULL);//Retrieve Candle Open-Price
   double            Close(int CandleIndex,ENUM_TIMEFRAMES Period=PERIOD_CURRENT,string SYMBOL=NULL);//Retrieve Candle Close-Price
   double            High(int CandleIndex,ENUM_TIMEFRAMES Period=PERIOD_CURRENT,string SYMBOL=NULL);//Retrieve Candle High-Price
   double            Low(int CandleIndex,ENUM_TIMEFRAMES Period=PERIOD_CURRENT,string SYMBOL=NULL);//Retrieve Candle Low-Price
   bool              IsLargerThanPreviousAndNext(datetime CandleTime,int Offset,string SYMBOL);//Determine if one candle is larger than two others
  };

//+------------------------------------------------------------------+
//|Retrieve Candle Open-Price                                        |
//+------------------------------------------------------------------+
double CCandleProperties::Open(int CandleIndex,ENUM_TIMEFRAMES Period=PERIOD_CURRENT,string SYMBOL=NULL)
  {
   return (SetSymbolName(SYMBOL))?iOpen(GetSymbolName(),Period,CandleIndex):0;//return candle open price
  }

//+------------------------------------------------------------------+
//|Retrieve Candle Close-Price                                       |
//+------------------------------------------------------------------+
double CCandleProperties::Close(int CandleIndex,ENUM_TIMEFRAMES Period=PERIOD_CURRENT,string SYMBOL=NULL)
  {
   return (SetSymbolName(SYMBOL))?iClose(GetSymbolName(),Period,CandleIndex):0;//return candle close price
  }

//+------------------------------------------------------------------+
//|Retrieve Candle High-Price                                        |
//+------------------------------------------------------------------+
double CCandleProperties::High(int CandleIndex,ENUM_TIMEFRAMES Period=PERIOD_CURRENT,string SYMBOL=NULL)
  {
   return (SetSymbolName(SYMBOL))?iHigh(GetSymbolName(),Period,CandleIndex):0;//return candle high price
  }

//+------------------------------------------------------------------+
//|Retrieve Candle Low-Price                                         |
//+------------------------------------------------------------------+
double CCandleProperties::Low(int CandleIndex,ENUM_TIMEFRAMES Period=PERIOD_CURRENT,string SYMBOL=NULL)
  {
   return (SetSymbolName(SYMBOL))?iLow(GetSymbolName(),Period,CandleIndex):0;//return candle low price
  }

//+------------------------------------------------------------------+
//|Determine if one candle is larger than two others                |
//+------------------------------------------------------------------+
bool CCandleProperties::IsLargerThanPreviousAndNext(datetime CandleTime,int Offset,string SYMBOL)
  {
   int CandleIndex = iBarShift(SYMBOL,PERIOD_M15,CandleTime);//Assign candle index of candletime
//--Assign candle index of candletime minus time offset
   int CandleIndexMinusOffset = iBarShift(SYMBOL,PERIOD_M15,Time.TimeMinusOffset(CandleTime,Offset));
//--Assign candle index of candletime plus time offset
   int CandleIndexPlusOffset = iBarShift(SYMBOL,PERIOD_M15,Time.TimePlusOffset(CandleTime,Offset));
//--Assign height of M15 candletime in pips
   double CandleHeight = High(CandleIndex,PERIOD_M15,SYMBOL)-Low(CandleIndex,PERIOD_M15,SYMBOL);
//--Assign height of M15 candletime  minus offset in Pips
   double CandleHeightMinusOffset = High(CandleIndexMinusOffset,PERIOD_M15,SYMBOL)-Low(CandleIndexMinusOffset,PERIOD_M15,SYMBOL);
//--Assign height of M15 candletime plus offset in Pips
   double CandleHeightPlusOffset = High(CandleIndexPlusOffset,PERIOD_M15,SYMBOL)-Low(CandleIndexPlusOffset,PERIOD_M15,SYMBOL);
//--Determine if candletime height is greater than candletime height minus offset and candletime height plus offset
   if(CandleHeight>CandleHeightMinusOffset&&CandleHeight>CandleHeightPlusOffset)
     {
      return true;//Candletime is likely when the news event occured
     }
   return false;//Candletime is unlikely when the real news data was released
  }
//+------------------------------------------------------------------+
```

## Object Properties Class

This class will be responsible for creating and deleting chart objects.

CObjectProperties has multilevel Inheritance from classes:

- CChartProperties
- CSymbolProperties

CObjectProperties has hierarchical Inheritance from classes:

- CSymbolProperties
- CSymbolInfo

```
//+------------------------------------------------------------------+
//|                                                      NewsTrading |
//|                                  Copyright 2024, MetaQuotes Ltd. |
//|                            https://www.mql5.com/en/users/kaaiblo |
//+------------------------------------------------------------------+
#include "ChartProperties.mqh"
//+------------------------------------------------------------------+
//|ObjectProperties class                                            |
//+------------------------------------------------------------------+
class CObjectProperties:public CChartProperties
  {
private:
   //Simple  chart objects structure
   struct ObjStruct
     {
      long           ChartId;
      string         Name;
     } Objects[];//ObjStruct variable array

   //-- Add chart object to Objects array
   void              AddObj(long chart_id,string name)
     {
      ArrayResize(Objects,Objects.Size()+1,Objects.Size()+2);
      Objects[Objects.Size()-1].ChartId=chart_id;
      Objects[Objects.Size()-1].Name=name;
     }

public:
                     CObjectProperties(void) {}//Class constructor

   //-- Create Rectangle chart object
   void              Square(long chart_ID,string name,int x_coord,int y_coord,int width,int height,ENUM_ANCHOR_POINT Anchor);

   //-- Create text chart object
   void              TextObj(long chartID,string name,string text,int x_coord,int y_coord,
                             ENUM_BASE_CORNER Corner=CORNER_LEFT_UPPER,int fontsize=10);

   //-- Create Event object
   void               EventObj(long chartID,string name,string description,datetime eventdate);

   //-- Class destructor removes all chart objects created previously
                    ~CObjectProperties(void)
     {
      for(uint i=0;i<Objects.Size();i++)
        {
         ObjectDelete(Objects[i].ChartId,Objects[i].Name);
        }
     }
  };

//+------------------------------------------------------------------+
//|Create Rectangle chart object                                     |
//+------------------------------------------------------------------+
void CObjectProperties::Square(long chart_ID,string name,int x_coord,int y_coord,int width,int height,ENUM_ANCHOR_POINT Anchor)
  {
   const int              sub_window=0;             // subwindow index
   const int              x=x_coord;                // X coordinate
   const int              y=y_coord;                // Y coordinate
   const color            back_clr=clrBlack;        // background color
   const ENUM_BORDER_TYPE border=BORDER_SUNKEN;     // border type
   const color            clr=clrRed;               // flat border color (Flat)
   const ENUM_LINE_STYLE  style=STYLE_SOLID;        // flat border style
   const int              line_width=0;             // flat border width
   const bool             back=false;               // in the background
   const bool             selection=false;          // highlight to move
   const bool             hidden=true;              // hidden in the object list

   ObjectDelete(chart_ID,name);//Delete previous object with the same name and chart id
   if(ObjectCreate(chart_ID,name,OBJ_RECTANGLE_LABEL,sub_window,0,0))//create rectangle object label
     {
      AddObj(chart_ID,name);//Add object to array
      ObjectSetInteger(chart_ID,name,OBJPROP_XDISTANCE,x);//Set x Distance/coordinate
      ObjectSetInteger(chart_ID,name,OBJPROP_YDISTANCE,y);//Set y Distance/coordinate
      ObjectSetInteger(chart_ID,name,OBJPROP_XSIZE,width);//Set object's width/x-size
      ObjectSetInteger(chart_ID,name,OBJPROP_YSIZE,height);//Set object's height/y-size
      ObjectSetInteger(chart_ID,name,OBJPROP_BGCOLOR,back_clr);//Set object's background color
      ObjectSetInteger(chart_ID,name,OBJPROP_BORDER_TYPE,border);//Set object's border type
      ObjectSetInteger(chart_ID,name,OBJPROP_ANCHOR,Anchor);//Set objects anchor point
      ObjectSetInteger(chart_ID,name,OBJPROP_COLOR,clr);//Set object's color
      ObjectSetInteger(chart_ID,name,OBJPROP_STYLE,style);//Set object's style
      ObjectSetInteger(chart_ID,name,OBJPROP_WIDTH,line_width);//Set object's flat border width
      ObjectSetInteger(chart_ID,name,OBJPROP_BACK,back);//Set if object is in foreground or not
      ObjectSetInteger(chart_ID,name,OBJPROP_SELECTABLE,selection);//Set if object is selectable/dragable
      ObjectSetInteger(chart_ID,name,OBJPROP_SELECTED,selection);//Set if object is Selected
      ObjectSetInteger(chart_ID,name,OBJPROP_HIDDEN,hidden);//Set if object is hidden in object list
      ChartRedraw(chart_ID);
     }
   else
     {
      Print("Failed to create object: ",name);
     }
  }

//+------------------------------------------------------------------+
//|Create text chart object                                          |
//+------------------------------------------------------------------+
void CObjectProperties::TextObj(long chartID,string name,string text,int x_coord,int y_coord,
                                ENUM_BASE_CORNER Corner=CORNER_LEFT_UPPER,int fontsize=10)
  {
   ObjectDelete(chartID,name);//Delete previous object with the same name and chart id
   if(ObjectCreate(chartID,name,OBJ_LABEL,0,0,0))//Create object label
     {
      AddObj(chartID,name);//Add object to array
      ObjectSetInteger(chartID,name,OBJPROP_XDISTANCE,x_coord);//Set x Distance/coordinate
      ObjectSetInteger(chartID,name,OBJPROP_YDISTANCE,y_coord);//Set y Distance/coordinate
      ObjectSetInteger(chartID,name,OBJPROP_CORNER,Corner);//Set object's corner anchor
      ObjectSetString(chartID,name,OBJPROP_TEXT,text);//Set object's text
      ObjectSetInteger(chartID,name,OBJPROP_COLOR,SymbolBackground());//Set object's color
      ObjectSetInteger(chartID,name,OBJPROP_FONTSIZE,fontsize);//Set object's font-size
     }
   else
     {
      Print("Failed to create object: ",name);
     }
  }

//+------------------------------------------------------------------+
//|Create Event object                                               |
//+------------------------------------------------------------------+
void CObjectProperties::EventObj(long chartID,string name,string description,datetime eventdate)
  {
   ObjectDelete(chartID,name);//Delete previous object with the same name and chart id
   if(ObjectCreate(chartID,name,OBJ_EVENT,0,eventdate,0))//Create object event
     {
      AddObj(chartID,name);//Add object to array
      ObjectSetString(chartID,name,OBJPROP_TEXT,description);//Set object's text
      ObjectSetInteger(chartID,name,OBJPROP_COLOR,clrBlack);//Set object's color
     }
   else
     {
      Print("Failed to create object: ",name);
     }
  }
//+------------------------------------------------------------------
```

The array variable Objects will store all chart objects created in this class CObjectProperties.

```
struct ObjStruct
     {
      long           ChartId;
      string         Name;
     } Objects[];//ObjStruct variable array
```

The function AddObj will add the chart object's chart id and object name into the Objects array.

```
   //-- Add chart object to Objects array
   void              AddObj(long chart_id,string name)
     {
      ArrayResize(Objects,Objects.Size()+1,Objects.Size()+2);
      Objects[Objects.Size()-1].ChartId=chart_id;
      Objects[Objects.Size()-1].Name=name;
     }
```

The function Square's purpose is to create a Rectangle object with specific properties to allow for customization.

```
void CObjectProperties::Square(long chart_ID,string name,int x_coord,int y_coord,int width,int height,ENUM_ANCHOR_POINT Anchor)
  {
   const int              sub_window=0;             // subwindow index
   const int              x=x_coord;                // X coordinate
   const int              y=y_coord;                // Y coordinate
   const color            back_clr=clrBlack;        // background color
   const ENUM_BORDER_TYPE border=BORDER_SUNKEN;     // border type
   const color            clr=clrRed;               // flat border color (Flat)
   const ENUM_LINE_STYLE  style=STYLE_SOLID;        // flat border style
   const int              line_width=0;             // flat border width
   const bool             back=false;               // in the background
   const bool             selection=false;          // highlight to move
   const bool             hidden=true;              // hidden in the object list

   ObjectDelete(chart_ID,name);//Delete previous object with the same name and chart id
   if(ObjectCreate(chart_ID,name,OBJ_RECTANGLE_LABEL,sub_window,0,0))//create rectangle object label
     {
      AddObj(chart_ID,name);//Add object to array
      ObjectSetInteger(chart_ID,name,OBJPROP_XDISTANCE,x);//Set x Distance/coordinate
      ObjectSetInteger(chart_ID,name,OBJPROP_YDISTANCE,y);//Set y Distance/coordinate
      ObjectSetInteger(chart_ID,name,OBJPROP_XSIZE,width);//Set object's width/x-size
      ObjectSetInteger(chart_ID,name,OBJPROP_YSIZE,height);//Set object's height/y-size
      ObjectSetInteger(chart_ID,name,OBJPROP_BGCOLOR,back_clr);//Set object's background color
      ObjectSetInteger(chart_ID,name,OBJPROP_BORDER_TYPE,border);//Set object's border type
      ObjectSetInteger(chart_ID,name,OBJPROP_ANCHOR,Anchor);//Set objects anchor point
      ObjectSetInteger(chart_ID,name,OBJPROP_COLOR,clr);//Set object's color
      ObjectSetInteger(chart_ID,name,OBJPROP_STYLE,style);//Set object's style
      ObjectSetInteger(chart_ID,name,OBJPROP_WIDTH,line_width);//Set object's flat border width
      ObjectSetInteger(chart_ID,name,OBJPROP_BACK,back);//Set if object is in foreground or not
      ObjectSetInteger(chart_ID,name,OBJPROP_SELECTABLE,selection);//Set if object is selectable/dragable
      ObjectSetInteger(chart_ID,name,OBJPROP_SELECTED,selection);//Set if object is Selected
      ObjectSetInteger(chart_ID,name,OBJPROP_HIDDEN,hidden);//Set if object is hidden in object list
      ChartRedraw(chart_ID);
     }
   else
     {
      Print("Failed to create object: ",name);
     }
  }
```

The function TextObj will create text objects on the chart.

```
void CObjectProperties::TextObj(long chartID,string name,string text,int x_coord,int y_coord,
                                ENUM_BASE_CORNER Corner=CORNER_LEFT_UPPER,int fontsize=10)
  {
   ObjectDelete(chartID,name);//Delete previous object with the same name and chart id
   if(ObjectCreate(chartID,name,OBJ_LABEL,0,0,0))//Create object label
     {
      AddObj(chartID,name);//Add object to array
      ObjectSetInteger(chartID,name,OBJPROP_XDISTANCE,x_coord);//Set x Distance/coordinate
      ObjectSetInteger(chartID,name,OBJPROP_YDISTANCE,y_coord);//Set y Distance/coordinate
      ObjectSetInteger(chartID,name,OBJPROP_CORNER,Corner);//Set object's corner anchor
      ObjectSetString(chartID,name,OBJPROP_TEXT,text);//Set object's text
      ObjectSetInteger(chartID,name,OBJPROP_COLOR,SymbolBackground());//Set object's color
      ObjectSetInteger(chartID,name,OBJPROP_FONTSIZE,fontsize);//Set object's font-size
     }
   else
     {
      Print("Failed to create object: ",name);
     }
  }
```

The function EventObj will create event objects on the chart to show the economic events the have or will occur.

```
void CObjectProperties::EventObj(long chartID,string name,string description,datetime eventdate)
  {
   ObjectDelete(chartID,name);//Delete previous object with the same name and chart id
   if(ObjectCreate(chartID,name,OBJ_EVENT,0,eventdate,0))//Create object event
     {
      AddObj(chartID,name);//Add object to array
      ObjectSetString(chartID,name,OBJPROP_TEXT,description);//Set object's text
      ObjectSetInteger(chartID,name,OBJPROP_COLOR,clrBlack);//Set object's color
     }
   else
     {
      Print("Failed to create object: ",name);
     }
  }
```

## News Class

Calendar Tables in Part 1:

![Database Tables in Part 1](https://c.mql5.com/2/78/Database_Tables_Part1.png)

![Calendar Database Part 1](https://c.mql5.com/2/78/Calendar_Database_Part1.png)

In our previous database in part 1, the file size was massive and pretty slow to store all the news data into the database. This is due to the news data being stored inefficiently. The biggest contributor to the file size and sluggish performance, is storing similar data repeatedly.

The tables:

- Data\_AU
- Data\_None
- Data\_UK
- Data\_US

Store the same news data with a variation to the time data.

New Design:

Calendar Contents in Part 2:

![Database Contents](https://c.mql5.com/2/78/Database_Tables_Part2.png)

![Calendar Database Part 2](https://c.mql5.com/2/78/Calendar_Database_Part2.png)

Instead of repeatedly storing the same news event data with different times. We will create a singular table to store all the news event data called MQL5Calendar. And to store the differing time data between each DST schedule, we will have another table called TimeSchedule. This layout will therefore reduce the file size by more than half of the previous calendar database and increase performance.

In the new database design we will have these contents:

- AutoDST Table
- Calendar\_AU View
- Calendar\_NONE View
- Calendar\_UK View
- Calendar\_US View
- MQL5Calendar Table
- Record Table
- TimeSchedule Table
- OnlyOne\_AutoDST Trigger
- OnlyOne\_Record Trigger

We are going to Normalize the Tables from the previous database, these tables are Data\_AU, Data\_None, Data\_UK and Data\_US.

[What is Database Normalization?](https://www.mql5.com/go?link=https://learn.microsoft.com/en-us/office/troubleshoot/access/database-normalization-description "https://learn.microsoft.com/en-us/office/troubleshoot/access/database-normalization-description")

Database normalization is a process in database design used to organize a database into tables and columns to minimize redundancy and dependency. The main goals are to eliminate redundant data (for example, storing the same data in more than one table) and ensure data dependencies make sense (only storing related data in a table). This process results in a set of tables that can be more easily maintained and reduces the chances of data anomalies.

![UML](https://c.mql5.com/2/78/UML_Tables.png)

We will also create triggers to assure that only one record is stored in tables AutoDST and Record. Additionally we will create views for each DST schedule to show the News events for the last updated day, ideally to easily navigate which news events were up to date without having to repeatedly run queries on the tables with thousands of entries.

### Okay so what is a Trigger?

A [trigger in SQLite](https://www.mql5.com/go?link=https://www.sqlitetutorial.net/sqlite-trigger/ "https://www.sqlitetutorial.net/sqlite-trigger/") is a special kind of stored procedure that automatically executes a specified set of actions in response to certain events on a particular table. These events can be insertions, updates, or deletions of rows in the table.

### What is a View?

A [view in SQLite](https://www.mql5.com/go?link=https://www.sqlitetutorial.net/sqlite-create-view/ "https://www.sqlitetutorial.net/sqlite-create-view/") is a virtual table that is based on the result set of a SELECT query. Unlike a table, a view does not store data physically. Instead, it provides a way to present data from one or more tables in a specific structure or format, often simplifying complex queries and enhancing data security.

Before we start creating new tables and adding to the already large size of the database. We need a way to know which tables to delete and which to keep. A simple solution would be to check for each table we know exists in our previous database, which would include Data\_AU and others. But we can't just hard code which tables to delete from our memory, our program needs to find the tables we do not need anymore by itself. In order to do this we need to check which tables exist in our database, and iterate through the tables we want to delete and skip those we want to keep.

In SQLite there is a table called SQLITE\_MASTER/SQLITE\_SCHEMA that stores database's metadata, including information about all objects in the database and the SQL used to define them. It's the most important system catalog in SQLite. The query below is used to get all the information from the database.

```
SELECT * FROM SQLITE_MASTER;
```

Database output:

```
type    name		    		tbl_name        rootpage        sql
table   Data_None       		Data_None       2       	CREATE TABLE Data_None(ID INT NOT NULL,EVENTID  INT   NOT NULL,COUNTRY  STRING   NOT NULL,EVENTNAME   STRING   NOT NULL,EVENTTYPE   STRING   NOT NULL,EVENTIMPORTANCE   STRING   NOT NULL,EVENTDATE   STRING   NOT NULL,EVENTCURRENCY  STRING   NOT NULL,EVENTCODE   STRING   NOT NULL,EVENTSECTOR STRING   NOT NULL,EVENTFORECAST  STRING   NOT NULL,EVENTPREVALUE  STRING   NOT NULL,EVENTIMPACT STRING   NOT NULL,EVENTFREQUENCY STRING   NOT NULL,PRIMARY KEY(ID))
index   sqlite_autoindex_Data_None_1    Data_None       3
table   Data_US 			Data_US 	4       	CREATE TABLE Data_US(ID INT NOT NULL,EVENTID  INT   NOT NULL,COUNTRY  STRING   NOT NULL,EVENTNAME   STRING   NOT NULL,EVENTTYPE   STRING   NOT NULL,EVENTIMPORTANCE   STRING   NOT NULL,EVENTDATE   STRING   NOT NULL,EVENTCURRENCY  STRING   NOT NULL,EVENTCODE   STRING   NOT NULL,EVENTSECTOR STRING   NOT NULL,EVENTFORECAST  STRING   NOT NULL,EVENTPREVALUE  STRING   NOT NULL,EVENTIMPACT STRING   NOT NULL,EVENTFREQUENCY STRING   NOT NULL,PRIMARY KEY(ID))
index   sqlite_autoindex_Data_US_1      Data_US 	5
table   Data_UK 			Data_UK 	6       	CREATE TABLE Data_UK(ID INT NOT NULL,EVENTID  INT   NOT NULL,COUNTRY  STRING   NOT NULL,EVENTNAME   STRING   NOT NULL,EVENTTYPE   STRING   NOT NULL,EVENTIMPORTANCE   STRING   NOT NULL,EVENTDATE   STRING   NOT NULL,EVENTCURRENCY  STRING   NOT NULL,EVENTCODE   STRING   NOT NULL,EVENTSECTOR STRING   NOT NULL,EVENTFORECAST  STRING   NOT NULL,EVENTPREVALUE  STRING   NOT NULL,EVENTIMPACT STRING   NOT NULL,EVENTFREQUENCY STRING   NOT NULL,PRIMARY KEY(ID))
index   sqlite_autoindex_Data_UK_1      Data_UK 	7
table   Data_AU 			Data_AU 	8       	CREATE TABLE Data_AU(ID INT NOT NULL,EVENTID  INT   NOT NULL,COUNTRY  STRING   NOT NULL,EVENTNAME   STRING   NOT NULL,EVENTTYPE   STRING   NOT NULL,EVENTIMPORTANCE   STRING   NOT NULL,EVENTDATE   STRING   NOT NULL,EVENTCURRENCY  STRING   NOT NULL,EVENTCODE   STRING   NOT NULL,EVENTSECTOR STRING   NOT NULL,EVENTFORECAST  STRING   NOT NULL,EVENTPREVALUE  STRING   NOT NULL,EVENTIMPACT STRING   NOT NULL,EVENTFREQUENCY STRING   NOT NULL,PRIMARY KEY(ID))
index   sqlite_autoindex_Data_AU_1      Data_AU 	9
table   Records 			Records 	38774   	CREATE TABLE Records(RECORDEDTIME INT NOT NULL)
table   AutoDST 			AutoDST 	38775   	CREATE TABLE AutoDST(DST STRING NOT NULL)
```

As seen from the database output, there is something called an index, which we did not create before.

### What is an Index?

An [index in SQLite](https://www.mql5.com/go?link=https://www.sqlitetutorial.net/sqlite-index/ "https://www.sqlitetutorial.net/sqlite-index/") is a database object that provides a way to improve the performance of query operations by facilitating faster retrieval of records from a table. Indexes are particularly useful for speeding up searches, sorting, and join operations by creating a sorted data structure (typically a B-tree) that allows the database engine to locate rows more quickly.

Why was the index created if we didn't create it previously?

When a table has a primary key in SQLite an index is automatically created for that particular table.

In our database output previously, we finally got all the objects and metadata in the database, we can now find the tables which are no longer necessary and delete them. In order to do this we will create an array of the tables we need and the SQL statements that create these tables, so we can compare them to the one in the database output and therefore remove the ones that do not match.

CNews has multilevel Inheritance from classes:

- CCandleProperties
- CChartProperties
- CSymbolProperties

CNews has Inclusions from classes:

- CDaylightSavings\_UK
- CDaylightSavings\_US
- CDaylightSavings\_AU

CNews has an inclusion from header file CommonVariables.mqh

CNews has hierarchical Inheritance from classes:

- CSymbolProperties
- CSymbolInfo
- CCandleProperties
- CTimeManagement

```
//+------------------------------------------------------------------+
//|                                                      NewsTrading |
//|                                  Copyright 2024, MetaQuotes Ltd. |
//|                            https://www.mql5.com/en/users/kaaiblo |
//+------------------------------------------------------------------+
#include "CommonVariables.mqh"
#include "DayLightSavings/DaylightSavings_UK.mqh"
#include "DayLightSavings/DaylightSavings_US.mqh"
#include "DayLightSavings/DaylightSavings_AU.mqh"
#include "CandleProperties.mqh"

//+------------------------------------------------------------------+
//|News class                                                        |
//+------------------------------------------------------------------+
class CNews : private CCandleProperties
  {
   //Private Declarations Only accessable by this class/header file
private:

   //-- To keep track of what is in our database
   enum CalendarComponents
     {
      AutoDST_Table,//AutoDST Table
      CalendarAU_View,//View for DST_AU
      CalendarNONE_View,//View for DST_NONE
      CalendarUK_View,//View for DST_UK
      CalendarUS_View,//View for DST_US
      Record_Table,// Record Table
      TimeSchedule_Table,//TimeSchedule Table
      MQL5Calendar_Table,//MQL5Calendar Table
      AutoDST_Trigger,//Table Trigger for AutoDST
      Record_Trigger//Table Trigger for Record
     };

   //-- structure to retrieve all the objects in the database
   struct SQLiteMaster
     {
      string         type;//will store object's type
      string         name;//will store object's name
      string         tbl_name;//will store table name
      int            rootpage;//will store rootpage
      string         sql;//Will store the sql create statement
     } DBContents[];//Array of type SQLiteMaster

   //--  MQL5CalendarContents inherits from SQLiteMaster structure
   struct MQL5CalendarContents:SQLiteMaster
     {
      CalendarComponents  Content;
      string         insert;//Will store the sql insert statement
     } CalendarContents[10];//Array to Store objects in our database

   CTimeManagement   Time;//TimeManagement Object declaration
   CDaylightSavings_UK  Savings_UK;//DaylightSavings Object for the UK and EU
   CDaylightSavings_US  Savings_US;//DaylightSavings Object for the US
   CDaylightSavings_AU  Savings_AU;//DaylightSavings Object for the AU

   bool              AutoDetectDST(DST_type &dstType);//Function will determine Broker DST
   DST_type          DSTType;//variable of DST_type enumeration declared in the CommonVariables class/header file
   bool              InsertIntoTables(int db,Calendar &Evalues[]);//Function for inserting Economic Data in to a database's table
   void              CreateAutoDST(int db);//Function for creating and inserting Recommend DST for the Broker into a table
   bool              CreateCalendarTable(int db,bool &tableExists);//Function for creating a table in a database
   bool              CreateTimeTable(int db,bool &tableExists);//Function for creating a table in a database
   void              CreateCalendarViews(int db);//Function for creating a view in a database
   void              CreateRecordTable(int db);//Creates a table to store the record of when last the Calendar database was updated/created
   bool              UpdateRecords();//Checks if the main Calendar database needs an update or not
   void              EconomicDetails(Calendar &NewsTime[]);//Gets values from the MQL5 economic Calendar
   string            DropRequest;//Variable for dropping tables in the database

   //-- Function for retrieving the MQL5CalendarContents structure for the enumartion type CalendarComponents
   MQL5CalendarContents CalendarStruct(CalendarComponents Content)
     {
      MQL5CalendarContents Calendar;
      for(uint i=0;i<CalendarContents.Size();i++)
        {
         if(CalendarContents[i].Content==Content)
           {
            return CalendarContents[i];
           }
        }
      return Calendar;
     }

   //Public declarations accessable via a class's Object
public:
                     CNews(void);
                    ~CNews(void);//Deletes a text file created when the Calendar database is being worked on
   void              CreateEconomicDatabase();//Creates the Calendar database for a specific Broker
   datetime          GetLatestNewsDate();//Gets the lastest/newest date in the Calendar database
  };

//+------------------------------------------------------------------+
//|Constructor                                                       |
//+------------------------------------------------------------------+
CNews::CNews(void):DropRequest("PRAGMA foreign_keys = OFF; "
                                  "PRAGMA secure_delete = ON; "
                                  "Drop %s IF EXISTS %s; "
                                  "Vacuum; "
                                  "PRAGMA foreign_keys = ON;")//Sql drop statement
  {
//-- initializing properties for the AutoDST table
   CalendarContents[0].Content = AutoDST_Table;
   CalendarContents[0].name = "AutoDST";
   CalendarContents[0].sql = "CREATE TABLE AutoDST(DST TEXT NOT NULL DEFAULT 'DST_NONE')STRICT;";
   CalendarContents[0].tbl_name = "AutoDST";
   CalendarContents[0].type = "table";
   CalendarContents[0].insert = "INSERT INTO 'AutoDST'(DST) VALUES ('%s');";

   string views[] = {"UK","US","AU","NONE"};
   string view_sql = "CREATE VIEW IF NOT EXISTS Calendar_%s "
                     "AS "
                     "SELECT C.Eventid,C.Eventname,C.Country,T.DST_%s as Time,C.EventCurrency,C.Eventcode from MQL5Calendar C,Record R "
                     "Inner join TimeSchedule T on C.ID=T.ID "
                     "Where DATE(REPLACE(T.DST_%s,'.','-'))=R.Date "
                     "Order by T.DST_%s Asc;";

//-- Sql statements for creating the table views
   for(uint i=1;i<=views.Size();i++)
     {
      CalendarContents[i].Content = (CalendarComponents)i;
      CalendarContents[i].name = StringFormat("Calendar_%s",views[i-1]);
      CalendarContents[i].sql = StringFormat(view_sql,views[i-1],views[i-1],views[i-1],views[i-1]);
      CalendarContents[i].tbl_name = StringFormat("Calendar_%s",views[i-1]);
      CalendarContents[i].type = "view";
     }

//-- initializing properties for the Record table
   CalendarContents[5].Content = Record_Table;
   CalendarContents[5].name = "Record";
   CalendarContents[5].sql = "CREATE TABLE Record(Date TEXT NOT NULL)STRICT;";
   CalendarContents[5].tbl_name="Record";
   CalendarContents[5].type = "table";
   CalendarContents[5].insert = "INSERT INTO 'Record'(Date) VALUES (Date(REPLACE('%s','.','-')));";

//-- initializing properties for the TimeSchedule table
   CalendarContents[6].Content = TimeSchedule_Table;
   CalendarContents[6].name = "TimeSchedule";
   CalendarContents[6].sql = "CREATE TABLE TimeSchedule(ID INT NOT NULL,DST_UK   TEXT   NOT NULL,DST_US   TEXT   NOT NULL,"
                             "DST_AU   TEXT   NOT NULL,DST_NONE   TEXT   NOT NULL,FOREIGN KEY (ID) REFERENCES MQL5Calendar (ID))STRICT;";
   CalendarContents[6].tbl_name="TimeSchedule";
   CalendarContents[6].type = "table";
   CalendarContents[6].insert = "INSERT INTO 'TimeSchedule'(ID,DST_UK,DST_US,DST_AU,DST_NONE) "
                                "VALUES (%d,'%s','%s', '%s', '%s');";

//-- initializing properties for the MQL5Calendar table
   CalendarContents[7].Content = MQL5Calendar_Table;
   CalendarContents[7].name = "MQL5Calendar";
   CalendarContents[7].sql = "CREATE TABLE MQL5Calendar(ID INT NOT NULL,EVENTID  INT   NOT NULL,COUNTRY  TEXT   NOT NULL,"
                             "EVENTNAME   TEXT   NOT NULL,EVENTTYPE   TEXT   NOT NULL,EVENTIMPORTANCE   TEXT   NOT NULL,"
                             "EVENTCURRENCY  TEXT   NOT NULL,EVENTCODE   TEXT   NOT NULL,EVENTSECTOR TEXT   NOT NULL,"
                             "EVENTFORECAST  TEXT   NOT NULL,EVENTPREVALUE  TEXT   NOT NULL,EVENTIMPACT TEXT   NOT NULL,"
                             "EVENTFREQUENCY TEXT   NOT NULL,PRIMARY KEY(ID))STRICT;";
   CalendarContents[7].tbl_name="MQL5Calendar";
   CalendarContents[7].type = "table";
   CalendarContents[7].insert = "INSERT INTO 'MQL5Calendar'(ID,EVENTID,COUNTRY,EVENTNAME,EVENTTYPE,EVENTIMPORTANCE,EVENTCURRENCY,EVENTCODE,"
                                "EVENTSECTOR,EVENTFORECAST,EVENTPREVALUE,EVENTIMPACT,EVENTFREQUENCY) "
                                "VALUES (%d,%d,'%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s');";

//-- Sql statement for creating the AutoDST table's trigger
   CalendarContents[8].Content = AutoDST_Trigger;
   CalendarContents[8].name = "OnlyOne_AutoDST";
   CalendarContents[8].sql = "CREATE TRIGGER IF NOT EXISTS OnlyOne_AutoDST "
                             "BEFORE INSERT ON AutoDST "
                             "BEGIN "
                             "Delete from AutoDST; "
                             "END;";
   CalendarContents[8].tbl_name="AutoDST";
   CalendarContents[8].type = "trigger";

//-- Sql statement for creating the Record table's trigger
   CalendarContents[9].Content = Record_Trigger;
   CalendarContents[9].name = "OnlyOne_Record";
   CalendarContents[9].sql = "CREATE TRIGGER IF NOT EXISTS OnlyOne_Record "
                             "BEFORE INSERT ON Record "
                             "BEGIN "
                             "Delete from Record; "
                             "END;";
   CalendarContents[9].tbl_name="Record";
   CalendarContents[9].type = "trigger";
  }

//+------------------------------------------------------------------+
//|Destructor                                                        |
//+------------------------------------------------------------------+
CNews::~CNews(void)
  {
   if(FileIsExist(NEWS_TEXT_FILE,FILE_COMMON))//Check if the news database open text file exists
     {
      FileDelete(NEWS_TEXT_FILE,FILE_COMMON);
     }
  }

//+------------------------------------------------------------------+
//|Gets values from the MQL5 economic Calendar                       |
//+------------------------------------------------------------------+
void CNews::EconomicDetails(Calendar &NewsTime[])
  {
   int Size=0;//to keep track of the size of the events in the NewsTime array
   MqlCalendarCountry countries[];
   string Country_code="";

   for(int i=0,count=CalendarCountries(countries); i<count; i++)
     {
      MqlCalendarValue values[];
      datetime date_from=0;//Get date from the beginning
      datetime date_to=(datetime)(Time.MonthsS()+iTime(Symbol(),PERIOD_D1,0));//Date of the next month from the current day
      if(CalendarValueHistory(values,date_from,date_to,countries[i].code))
        {
         for(int x=0; x<(int)ArraySize(values); x++)
           {
            MqlCalendarEvent event;
            ulong event_id=values[x].event_id;//Get the event id
            if(CalendarEventById(event_id,event))
              {
               ArrayResize(NewsTime,Size+1,Size+2);//Readjust the size of the array to +1 of the array size
               StringReplace(event.name,"'","");//Removing or replacing single quotes(') from event name with an empty string
               NewsTime[Size].CountryName = countries[i].name;//storing the country's name from the specific event
               NewsTime[Size].EventName = event.name;//storing the event's name
               NewsTime[Size].EventType = EnumToString(event.type);//storing the event type from (ENUM_CALENDAR_EVENT_TYPE) to a string
               //-- storing the event importance from (ENUM_CALENDAR_EVENT_IMPORTANCE) to a string
               NewsTime[Size].EventImportance = EnumToString(event.importance);
               NewsTime[Size].EventId = event.id;//storing the event id
               NewsTime[Size].EventDate = TimeToString(values[x].time);//storing normal event time
               NewsTime[Size].EventCurrency = countries[i].currency;//storing event currency
               NewsTime[Size].EventCode = countries[i].code;//storing event code
               NewsTime[Size].EventSector = EnumToString(event.sector);//storing event sector from (ENUM_CALENDAR_EVENT_SECTOR) to a string
               if(values[x].HasForecastValue())//Checks if the event has a forecast value
                 {
                  NewsTime[Size].EventForecast = (string)values[x].forecast_value;//storing the forecast value into a string
                 }
               else
                 {
                  NewsTime[Size].EventForecast = "None";//storing 'None' as the forecast value
                 }

               if(values[x].HasPreviousValue())//Checks if the event has a previous value
                 {
                  NewsTime[Size].EventPreval = (string)values[x].prev_value;//storing the previous value into a string
                 }
               else
                 {
                  NewsTime[Size].EventPreval = "None";//storing 'None' as the previous value
                 }
               //-- storing the event impact from (ENUM_CALENDAR_EVENT_IMPACT) to a string
               NewsTime[Size].EventImpact =  EnumToString(values[x].impact_type);
               //-- storing the event frequency from (ENUM_CALENDAR_EVENT_FREQUENCY) to a string
               NewsTime[Size].EventFrequency =  EnumToString(event.frequency);
               Size++;//incrementing the Calendar array NewsTime
              }
           }
        }
     }
  }

//+------------------------------------------------------------------+
//|Checks if the main Calendar database needs an update or not       |
//+------------------------------------------------------------------+
bool CNews::UpdateRecords()
  {
//initialize variable to true
   bool perform_update=true;
//--- open/create
//-- try to open database Calendar
   int db=DatabaseOpen(NEWS_DATABASE_FILE, DATABASE_OPEN_READWRITE | DATABASE_OPEN_CREATE| DATABASE_OPEN_COMMON);
   if(db==INVALID_HANDLE)//Checks if the database was able to be opened
     {
      //if opening the database failed
      if(!FileIsExist(NEWS_DATABASE_FILE,FILE_COMMON))//Checks if the database Calendar exists in the common folder
        {
         return perform_update;//Returns true when the database was failed to be opened and the file doesn't exist in the common folder
        }
     }

   int MasterRequest = DatabasePrepare(db,"select * from sqlite_master where type<>'index';");
   if(MasterRequest==INVALID_HANDLE)
     {
      Print("DB: ",NEWS_DATABASE_FILE, " request failed with code ", GetLastError());
     }
   else
     {
      SQLiteMaster ReadContents;
      //Assigning values from the sql query into DBContents array
      for(int i=0; DatabaseReadBind(MasterRequest,ReadContents); i++)
        {
         ArrayResize(DBContents,i+1,i+2);
         DBContents[i].type = ReadContents.type;
         DBContents[i].name = ReadContents.name;
         DBContents[i].tbl_name = ReadContents.tbl_name;
         DBContents[i].rootpage = ReadContents.rootpage;
         /*Check if the end of the sql string has a character ';' if not add this character to the string*/
         DBContents[i].sql = (StringFind(ReadContents.sql,";",StringLen(ReadContents.sql)-1)==
                              (StringLen(ReadContents.sql)-1))?ReadContents.sql:ReadContents.sql+";";;
        }

      uint contents_exists = 0;
      for(uint i=0;i<DBContents.Size();i++)
        {
         bool isCalendarContents = false;
         for(uint x=0;x<CalendarContents.Size();x++)
           {
            /*Store Sql query from CalendarContents without string ' IF NOT EXISTS'*/
            string CalendarSql=CalendarContents[x].sql;
            StringReplace(CalendarSql," IF NOT EXISTS","");
            //-- Check if the Db object is in our list
            if(DBContents[i].name==CalendarContents[x].name&&
               (DBContents[i].sql==CalendarSql||
                DBContents[i].sql==CalendarContents[x].sql)&&
               CalendarContents[x].type==DBContents[i].type&&
               CalendarContents[x].tbl_name==DBContents[i].tbl_name)
              {
               contents_exists++;
               isCalendarContents = true;
              }
           }
         if(!isCalendarContents)
           {
            //-- Print DBcontent's name if it does not match with CalendarContents
            PrintFormat("DBContent: %s is not needed!",DBContents[i].name);
            //-- We will drop the table if it is not neccessary
            DatabaseExecute(db,StringFormat(DropRequest,DBContents[i].type,DBContents[i].name));
            Print("Attempting To Clean Database...");
           }
        }
      /*If not all the CalendarContents exist in the Calendar Database before an update */
      if(contents_exists!=CalendarContents.Size())
        {
         return perform_update;
        }
     }
   if(!DatabaseTableExists(db,CalendarStruct(Record_Table).name))//If the database table 'Record' doesn't exist
     {
      DatabaseClose(db);
      return perform_update;
     }

//-- Sql query to determine the lastest or maximum date recorded
   /* If the last recorded date data in the 'Record' table is not equal to the current day, perform an update! */
   string request_text=StringFormat("SELECT Date FROM %s where Date=Date(REPLACE('%s','.','-'))",
                                    CalendarStruct(Record_Table).name,TimeToString(TimeTradeServer()));
   int request=DatabasePrepare(db,request_text);//Creates a handle of a request, which can then be executed using DatabaseRead()
   if(request==INVALID_HANDLE)//Checks if the request failed to be completed
     {
      Print("DB: ",NEWS_DATABASE_FILE, " request failed with code ", GetLastError());
      DatabaseClose(db);
      return perform_update;
     }

   if(DatabaseRead(request))//Will be true if there are results from the sql query/request
     {
      DatabaseFinalize(request);//Removes a request created in DatabasePrepare()
      DatabaseClose(db);//Closes the database
      perform_update=false;
      return perform_update;
     }
   else
     {
      DatabaseFinalize(request);//Removes a request created in DatabasePrepare()
      DatabaseClose(db);//Closes the database
      return perform_update;
     }
  }

//+------------------------------------------------------------------+
//|Creates the Calendar database for a specific Broker               |
//+------------------------------------------------------------------+
void CNews::CreateEconomicDatabase()
  {
   if(FileIsExist(NEWS_DATABASE_FILE,FILE_COMMON))//Check if the database exists
     {
      if(!UpdateRecords())//Check if the database is up to date
        {
         return;//will terminate execution of the rest of the code below
        }
     }
   if(FileIsExist(NEWS_TEXT_FILE,FILE_COMMON))//Check if the database is open
     {
      return;//will terminate execution of the rest of the code below
     }

   Calendar Evalues[];//Creating a Calendar array variable
   bool failed=false,tableExists=false;
   int file=INVALID_HANDLE;
//--- open/create the database 'Calendar'
//-- will try to open/create in the common folder
   int db=DatabaseOpen(NEWS_DATABASE_FILE, DATABASE_OPEN_READWRITE | DATABASE_OPEN_CREATE| DATABASE_OPEN_COMMON);
   if(db==INVALID_HANDLE)//Checks if the database 'Calendar' failed to open/create
     {
      Print("DB: ",NEWS_DATABASE_FILE, " open failed with code ", GetLastError());
      return;//will terminate execution of the rest of the code below
     }
   else
     {
      //-- try to create a text file 'NewsDatabaseOpen' in common folder
      file=FileOpen(NEWS_TEXT_FILE,FILE_WRITE|FILE_ANSI|FILE_TXT|FILE_COMMON);
      if(file==INVALID_HANDLE)
        {
         DatabaseClose(db);//Closes the database 'Calendar' if the News text file failed to be created
         return;//will terminate execution of the rest of the code below
        }
     }

   DatabaseTransactionBegin(db);//Starts transaction execution
   Print("Please wait...");

//-- attempt to create the MQL5Calendar and TimeSchedule tables
   if(!CreateCalendarTable(db,tableExists)||!CreateTimeTable(db,tableExists))
     {
      FileClose(file);//Closing the file 'NewsDatabaseOpen.txt'
      FileDelete(NEWS_TEXT_FILE,FILE_COMMON);//Deleting the file 'NewsDatabaseOpen.txt'
      return;//will terminate execution of the rest of the code below
     }

   EconomicDetails(Evalues);//Retrieving the data from the Economic Calendar
   if(tableExists)//Checks if there is an existing table within the Calendar Database
     {
      //if there is an existing table we will notify the user that we are updating the table.
      PrintFormat("Updating %s",NEWS_DATABASE_FILE);
     }
   else
     {
      //if there isn't an existing table we will notify the user that we about to create one
      PrintFormat("Creating %s",NEWS_DATABASE_FILE);
     }

//-- attempt to insert economic event data into the calendar tables
   if(!InsertIntoTables(db,Evalues))
     {
      //-- Will assign true if inserting economic vaules failed in the MQL5Calendar and TimeSchedule tables
      failed=true;
     }

   if(failed)
     {
      //--- roll back all transactions and unlock the database
      DatabaseTransactionRollback(db);
      PrintFormat("%s: DatabaseExecute() failed with code %d", __FUNCTION__, GetLastError());
      FileClose(file);//Close the text file 'NEWS_TEXT_FILE'
      FileDelete(NEWS_TEXT_FILE,FILE_COMMON);//Delete the text file, as we are reverted/rolled-back the database
      ArrayRemove(Evalues,0,WHOLE_ARRAY);//Removes the values in the array
     }
   else
     {
      CreateCalendarViews(db);
      CreateRecordTable(db);//Will create the 'Record' table and insert the  current time
      CreateAutoDST(db);//Will create the 'AutoDST' table and insert the broker's DST schedule
      FileClose(file);//Close the text file 'NEWS_TEXT_FILE'
      FileDelete(NEWS_TEXT_FILE,FILE_COMMON);//Delete the text file, as we are about to close the database
      ArrayRemove(Evalues,0,WHOLE_ARRAY);//Removes the values in the array
      if(tableExists)
        {
         //Let the user/trader know that the database was updated
         PrintFormat("%s Updated",NEWS_DATABASE_FILE);
        }
      else
        {
         //Let the user/trader know that the database was created
         PrintFormat("%s Created",NEWS_DATABASE_FILE);
        }
     }
//--- all transactions have been performed successfully - record changes and unlock the database
   DatabaseTransactionCommit(db);
   DatabaseClose(db);//Close the database
  }

//+------------------------------------------------------------------+
//|Function for creating a table in a database                       |
//+------------------------------------------------------------------+
bool CNews::CreateCalendarTable(int db,bool &tableExists)
  {
//-- Checks if a table 'MQL5Calendar' exists
   if(DatabaseTableExists(db,CalendarStruct(MQL5Calendar_Table).name))
     {
      tableExists=true;//Assigns true to tableExists variable
      //-- Checks if a table 'TimeSchedule' exists in the database 'Calendar'
      if(DatabaseTableExists(db,CalendarStruct(TimeSchedule_Table).name))
        {
         //-- We will drop the table if the table already exists
         if(!DatabaseExecute(db,StringFormat("Drop Table %s",CalendarStruct(TimeSchedule_Table).name)))
           {
            //If the table failed to be dropped/deleted
            PrintFormat("Failed to drop table %s with code %d",CalendarStruct(TimeSchedule_Table).name,GetLastError());
            DatabaseClose(db);//Close the database
            return false;//will terminate execution of the rest of the code below and return false, when the table cannot be dropped
           }
        }
      //--We will drop the table if the table already exists
      if(!DatabaseExecute(db,StringFormat("Drop Table %s",CalendarStruct(MQL5Calendar_Table).name)))
        {
         //If the table failed to be dropped/deleted
         PrintFormat("Failed to drop table %s with code %d",CalendarStruct(MQL5Calendar_Table).name,GetLastError());
         DatabaseClose(db);//Close the database
         return false;//will terminate execution of the rest of the code below and return false, when the table cannot be dropped
        }
     }
//-- If the database table 'MQL5Calendar' doesn't exist
   if(!DatabaseTableExists(db,CalendarStruct(MQL5Calendar_Table).name))
     {
      //--- create the table 'MQL5Calendar'
      if(!DatabaseExecute(db,CalendarStruct(MQL5Calendar_Table).sql))//Checks if the table was successfully created
        {
         Print("DB: create the Calendar table failed with code ", GetLastError());
         DatabaseClose(db);//Close the database
         return false;//Function returns false if creating the table failed
        }
     }
   return true;//Function returns true if creating the table was successful
  }

//+------------------------------------------------------------------+
//|Function for creating a table in a database                       |
//+------------------------------------------------------------------+
bool CNews::CreateTimeTable(int db,bool &tableExists)
  {
//-- If the database table 'TimeSchedule' doesn't exist
   if(!DatabaseTableExists(db,CalendarStruct(TimeSchedule_Table).name))
     {
      //--- create the table 'TimeSchedule'
      if(!DatabaseExecute(db,CalendarStruct(TimeSchedule_Table).sql))//Checks if the table was successfully created
        {
         Print("DB: create the Calendar table failed with code ", GetLastError());
         DatabaseClose(db);//Close the database
         return false;//Function returns false if creating the table failed
        }
     }
   return true;//Function returns true if creating the table was successful
  }

//+------------------------------------------------------------------+
//|Function for creating views in a database                         |
//+------------------------------------------------------------------+
void CNews::CreateCalendarViews(int db)
  {
   for(uint i=1;i<=4;i++)
     {
      if(!DatabaseExecute(db,CalendarStruct((CalendarComponents)i).sql))//Checks if the view was successfully created
        {
         Print("DB: create the Calendar view failed with code ", GetLastError());
        }
     }
  }

//+------------------------------------------------------------------+
//|Function for inserting Economic Data in to a database's table     |
//+------------------------------------------------------------------+
bool CNews::InsertIntoTables(int db,Calendar &Evalues[])
  {
   for(uint i=0; i<Evalues.Size(); i++)//Looping through all the Economic Events
     {
      string request_insert_into_calendar =
         StringFormat(CalendarStruct(MQL5Calendar_Table).insert,
                      i,
                      Evalues[i].EventId,
                      Evalues[i].CountryName,
                      Evalues[i].EventName,
                      Evalues[i].EventType,
                      Evalues[i].EventImportance,
                      Evalues[i].EventCurrency,
                      Evalues[i].EventCode,
                      Evalues[i].EventSector,
                      Evalues[i].EventForecast,
                      Evalues[i].EventPreval,
                      Evalues[i].EventImpact,
                      Evalues[i].EventFrequency);//Inserting all the columns for each event record
      if(DatabaseExecute(db,request_insert_into_calendar))//Check if insert query into calendar was successful
        {
         string request_insert_into_time =
            StringFormat(CalendarStruct(TimeSchedule_Table).insert,
                         i,
                         //-- Economic EventDate adjusted for UK DST(Daylight Savings Time)
                         Savings_UK.adjustDaylightSavings(StringToTime(Evalues[i].EventDate)),
                         //-- Economic EventDate adjusted for US DST(Daylight Savings Time)
                         Savings_US.adjustDaylightSavings(StringToTime(Evalues[i].EventDate)),
                         //-- Economic EventDate adjusted for AU DST(Daylight Savings Time)
                         Savings_AU.adjustDaylightSavings(StringToTime(Evalues[i].EventDate)),
                         Evalues[i].EventDate//normal Economic EventDate
                        );//Inserting all the columns for each event record
         if(!DatabaseExecute(db,request_insert_into_time))
           {
            Print(GetLastError());
            //-- Will print the sql query to check for any errors or possible defaults in the query/request
            Print(request_insert_into_time);
            return false;//Will end the loop and return false, as values failed to be inserted into the table
           }
        }
      else
        {
         Print(GetLastError());
         //-- Will print the sql query to check for any errors or possible defaults in the query/request
         Print(request_insert_into_calendar);
         return false;//Will end the loop and return false, as values failed to be inserted into the table
        }
     }
   return true;//Will return true, all values were inserted into the table successfully
  }

//+------------------------------------------------------------------+
//|Creates a table to store the record of when last the Calendar     |
//|database was updated/created                                      |
//+------------------------------------------------------------------+
void CNews::CreateRecordTable(int db)
  {
   bool failed=false;
   if(!DatabaseTableExists(db,CalendarStruct(Record_Table).name))//Checks if the table 'Record' exists in the databse 'Calendar'
     {
      //--- create the table
      if(!DatabaseExecute(db,CalendarStruct(Record_Table).sql))//Will attempt to create the table 'Record'
        {
         Print("DB: create the Records table failed with code ", GetLastError());
         DatabaseClose(db);//Close the database
         return;//Exits the function if creating the table failed
        }
      else//If Table was created Successfully then Create Trigger
        {
         DatabaseExecute(db,CalendarStruct(Record_Trigger).sql);
        }
     }
   else
     {
      DatabaseExecute(db,CalendarStruct(Record_Trigger).sql);
     }
//Sql query/request to insert the current time into the 'Date' column in the table 'Record'
   string request_text=StringFormat(CalendarStruct(Record_Table).insert,TimeToString(TimeTradeServer()));
   if(!DatabaseExecute(db, request_text))//Will attempt to run this sql request/query
     {
      Print(GetLastError());
      PrintFormat(CalendarStruct(Record_Table).insert,TimeToString(TimeTradeServer()));
      failed=true;//assign true if the request failed
     }
   if(failed)
     {
      //--- roll back all transactions and unlock the database
      DatabaseTransactionRollback(db);
      PrintFormat("%s: DatabaseExecute() failed with code %d", __FUNCTION__, GetLastError());
     }
  }

//+------------------------------------------------------------------+
//|Function for creating and inserting Recommend DST for the Broker  |
//|into a table                                                      |
//+------------------------------------------------------------------+
void CNews::CreateAutoDST(int db)
  {
   bool failed=false;//boolean variable
   if(!AutoDetectDST(DSTType))//Check if AutoDetectDST went through all the right procedures
     {
      return;//will terminate execution of the rest of the code below
     }

   if(!DatabaseTableExists(db,CalendarStruct(AutoDST_Table).name))//Checks if the table 'AutoDST' exists in the databse 'Calendar'
     {
      //--- create the table AutoDST
      if(!DatabaseExecute(db,CalendarStruct(AutoDST_Table).sql))//Will attempt to create the table 'AutoDST'
        {
         Print("DB: create the AutoDST table failed with code ", GetLastError());
         DatabaseClose(db);//Close the database
         return;//Exits the function if creating the table failed
        }
      else//If Table was created Successfully then Create Trigger
        {
         DatabaseExecute(db,CalendarStruct(AutoDST_Trigger).sql);
        }
     }
   else
     {
      //Create trigger if AutoDST table exists
      DatabaseExecute(db,CalendarStruct(AutoDST_Trigger).sql);
     }
//Sql query/request to insert the recommend DST for the Broker using the DSTType variable to determine which string data to insert
   string request_text=StringFormat(CalendarStruct(AutoDST_Table).insert,EnumToString(DSTType));
   if(!DatabaseExecute(db, request_text))//Will attempt to run this sql request/query
     {
      Print(GetLastError());
      PrintFormat(CalendarStruct(AutoDST_Table).insert,EnumToString(DSTType));//Will print the sql query if failed
      failed=true;//assign true if the request failed
     }
   if(failed)
     {
      //--- roll back all transactions and unlock the database
      DatabaseTransactionRollback(db);
      PrintFormat("%s: DatabaseExecute() failed with code %d", __FUNCTION__, GetLastError());
     }
  }

//+------------------------------------------------------------------+
//|Gets the latest/newest date in the Calendar database              |
//+------------------------------------------------------------------+
datetime CNews::GetLatestNewsDate()
  {
//--- open the database 'Calendar' in the common folder
   int db=DatabaseOpen(NEWS_DATABASE_FILE, DATABASE_OPEN_READONLY|DATABASE_OPEN_COMMON);

   if(db==INVALID_HANDLE)//Checks if 'Calendar' failed to be opened
     {
      if(!FileIsExist(NEWS_DATABASE_FILE,FILE_COMMON))//Checks if 'Calendar' database exists
        {
         Print("Could not find Database!");
         return 0;//Will return the earliest date which is 1970.01.01 00:00:00
        }
     }
   string latest_record="1970.01.01";//string variable with the first/earliest possible date in MQL5
//Sql query to determine the lastest or maximum recorded time from which the database was updated.
   string request_text="SELECT REPLACE(Date,'-','.') FROM 'Record'";
   int request=DatabasePrepare(db,request_text);
   if(request==INVALID_HANDLE)
     {
      Print("DB: ",NEWS_DATABASE_FILE, " request failed with code ", GetLastError());
      DatabaseClose(db);//Close Database
      return 0;
     }
   if(DatabaseRead(request))//Will read the one record in the 'Record' table
     {
      //-- Will assign the first column(column 0) value to the variable 'latest_record'
      if(!DatabaseColumnText(request,0,latest_record))
        {
         Print("DatabaseRead() failed with code ", GetLastError());
         DatabaseFinalize(request);//Finalize request
         DatabaseClose(db);//Closes the database 'Calendar'
         return D'1970.01.01';//Will end the for loop and will return the earliest date which is 1970.01.01 00:00:00
        }
     }
   DatabaseFinalize(request);
   DatabaseClose(db);//Closes the database 'Calendar'
   return (datetime)latest_record;//Returns the string latest_record converted to datetime
  }

//+------------------------------------------------------------------+
//|Function will determine Broker DST                                |
//+------------------------------------------------------------------+
bool CNews::AutoDetectDST(DST_type &dstType)
  {
   MqlCalendarValue values[];//Single array of MqlCalendarValue type
   string eventtime[];//Single string array variable to store NFP(Nonfarm Payrolls) dates for the 'United States' from the previous year
//-- Will store the previous year into an integer
   int lastyear = Time.ReturnYear(Time.TimeMinusOffset(iTime(Symbol(),PERIOD_CURRENT,0),Time.YearsS()));
//-- Will store the start date for the previous year
   datetime lastyearstart = StringToTime(StringFormat("%s.01.01 00:00:00",(string)lastyear));
//-- Will store the end date for the previous year
   datetime lastyearend = StringToTime(StringFormat("%s.12.31 23:59:59",(string)lastyear));
//-- Getting last year's calendar values for CountryCode = 'US'
   if(CalendarValueHistory(values,lastyearstart,lastyearend,"US"))
     {
      for(int x=0; x<(int)ArraySize(values); x++)
        {
         if(values[x].event_id==840030016)//Get only NFP Event Dates
           {
            ArrayResize(eventtime,eventtime.Size()+1,eventtime.Size()+2);//Increasing the size of eventtime array by 1
            eventtime[eventtime.Size()-1] = TimeToString(values[x].time);//Storing the dates in an array of type string
           }
        }
     }
//-- datetime variables to store the broker's timezone shift(change)
   datetime ShiftStart=D'1970.01.01 00:00:00',ShiftEnd=D'1970.01.01 00:00:00';
   string   EURUSD="";//String variables declarations for working with EURUSD
   bool     EurusdIsFound=false;//Boolean variables declarations for working with EURUSD
   for(int i=0;i<SymbolsTotal(true);i++)//Will loop through all the Symbols inside the Market Watch
     {
      string SymName = SymbolName(i,true);//Assign the Symbol Name of index 'i' from the list of Symbols inside the Market Watch
      //-- Check if the Symbol outside the Market Watch has a SYMBOL_CURRENCY_BASE of EUR
      //-- and a SYMBOL_CURRENCY_PROFIT of USD, and this Symbol is not a Custom Symbol(Is not from the broker)
      if(((CurrencyBase(SymName)=="EUR"&&CurrencyProfit(SymName)=="USD")||
          (StringFind(SymName,"EUR")>-1&&CurrencyProfit(SymName)=="USD"))&&!Custom(SymName))
        {
         EURUSD = SymName;//Assigning the name of the EURUSD Symbol found inside the Market Watch
         EurusdIsFound = true;//EURUSD Symbol was found in the Trading Terminal for your Broker
         break;//Will end the for loop
        }
     }
   if(!EurusdIsFound)//Check if EURUSD Symbol was already Found in the Market Watch
     {
      for(int i=0; i<SymbolsTotal(false); i++)//Will loop through all the available Symbols outside the Market Watch
        {
         string SymName = SymbolName(i,false);//Assign the Symbol Name of index 'i' from the list of Symbols outside the Market Watch
         //-- Check if the Symbol outside the Market Watch has a SYMBOL_CURRENCY_BASE of EUR
         //-- and a SYMBOL_CURRENCY_PROFIT of USD, and this Symbol is not a Custom Symbol(Is not from the broker)
         if(((CurrencyBase(SymName)=="EUR"&&CurrencyProfit(SymName)=="USD")||
             (StringFind(SymName,"EUR")>-1&&CurrencyProfit(SymName)=="USD"))&&!Custom(SymName))
           {
            EURUSD = SymName;//Assigning the name of the EURUSD Symbol found outside the Market Watch
            EurusdIsFound = true;//EURUSD Symbol was found in the Trading Terminal for your Broker
            break;//Will end the for loop
           }
        }
     }
   if(!EurusdIsFound)//Check if EURUSD Symbol was Found in the Trading Terminal for your Broker
     {
      Print("Cannot Find EURUSD!");
      Print("Cannot Create Database!");
      Print("Server DST Cannot be Detected!");
      dstType = DST_NONE;//Assigning enumeration value DST_NONE, Broker has no DST(Daylight Savings Time)
      return false;//Returning False, Broker's DST schedule was not found
     }

   struct DST
     {
      bool           result;
      datetime       date;
     } previousresult,currentresult;

   bool timeIsShifted;//Boolean variable declaration will be used to determine if the broker changes it's timezone
   for(uint i=0;i<eventtime.Size();i++)
     {
      //-- Store the result of if the eventdate is the larger candlestick
      currentresult.result = IsLargerThanPreviousAndNext((datetime)eventtime[i],Time.HoursS(),EURUSD);
      currentresult.date = (datetime)eventtime[i];//Store the eventdate from eventtime[i]
      //-- Check if there is a difference between the previous result and the current result
      timeIsShifted = ((currentresult.result!=previousresult.result&&i>0)?true:false);

      //-- Check if the Larger candle has shifted from the previous event date to the current event date in eventtime[i] array
      if(timeIsShifted)
        {
         if(ShiftStart==D'1970.01.01 00:00:00')//Check if the ShiftStart variable has not been assigned a relevant value yet
           {
            ShiftStart=currentresult.date;//Store the eventdate for when the timeshift began
           }
         ShiftEnd=previousresult.date;//Store the eventdate timeshift
        }
      previousresult.result = currentresult.result;//Store the previous result of if the eventdate is the larger candlestick
      previousresult.date = currentresult.date;//Store the eventdate from eventtime[i]
     }
//-- Check if the ShiftStart variable has not been assigned a relevant value and the eventdates are more than zero
   if(ShiftStart==D'1970.01.01 00:00:00'&&eventtime.Size()>0)
     {
      Print("Broker ServerTime unchanged!");
      dstType = DST_NONE;//Assigning enumeration value DST_NONE, Broker has no DST(Daylight Savings Time)
      return true;//Returning True, Broker's DST schedule was found successfully
     }

   datetime DaylightStart,DaylightEnd;//Datetime variables declarations for start and end dates for DaylightSavings
   if(Savings_AU.DaylightSavings(lastyear,DaylightStart,DaylightEnd))
     {
      if(Time.DateIsInRange(DaylightStart,DaylightEnd,ShiftStart,ShiftEnd))
        {
         Print("Broker ServerTime Adjusted For AU DST");
         dstType = DST_AU;//Assigning enumeration value AU_DST, Broker has AU DST(Daylight Savings Time)
         return true;//Returning True, Broker's DST schedule was found successfully
        }
     }
   else
     {
      Print("Something went wrong!");
      Print("Cannot Find Daylight-Savings Date For AU");
      Print("Year: %d Cannot Be Found!",lastyear);
      dstType = DST_NONE;//Assigning enumeration value DST_NONE, Broker has no DST(Daylight Savings Time)
      return false;//Returning False, Broker's DST schedule was not found
     }

   if(Savings_UK.DaylightSavings(lastyear,DaylightStart,DaylightEnd))
     {
      if(Time.DateIsInRange(DaylightStart,DaylightEnd,ShiftStart,ShiftEnd))
        {
         Print("Broker ServerTime Adjusted For UK DST");
         dstType = DST_UK;//Assigning enumeration value UK_DST, Broker has UK/EU DST(Daylight Savings Time)
         return true;//Returning True, Broker's DST schedule was found successfully
        }
     }
   else
     {
      Print("Something went wrong!");
      Print("Cannot Find Daylight-Savings Date For UK");
      Print("Year: %d Cannot Be Found!",lastyear);
      dstType = DST_NONE;//Assigning enumeration value DST_NONE, Broker has no DST(Daylight Savings Time)
      return false;//Returning False, Broker's DST schedule was not found
     }

   if(Savings_US.DaylightSavings(lastyear,DaylightStart,DaylightEnd))
     {
      if(Time.DateIsInRange(DaylightStart,DaylightEnd,ShiftStart,ShiftEnd))
        {
         Print("Broker ServerTime Adjusted For US DST");
         dstType = DST_US;//Assigning enumeration value US_DST, Broker has US DST(Daylight Savings Time)
         return true;//Returning True, Broker's DST schedule was found successfully
        }
     }
   else
     {
      Print("Something went wrong!");
      Print("Cannot Find Daylight-Savings Date For US");
      Print("Year: %d Cannot Be Found!",lastyear);
      dstType = DST_NONE;//Assigning enumeration value DST_NONE, Broker has no DST(Daylight Savings Time)
      return false;//Returning False, Broker's DST schedule was not found
     }
   Print("Cannot Detect Broker ServerTime Configuration!");
   dstType = DST_NONE;//Assigning enumeration value DST_NONE, Broker has no DST(Daylight Savings Time)
   return false;//Returning False, Broker's DST schedule was not found
  }
//+------------------------------------------------------------------+
```

We will give every component in our desired database design an enumeration value, as a form of identity.

```
 enum CalendarComponents
     {
      AutoDST_Table,//AutoDST Table
      CalendarAU_View,//View for DST_AU
      CalendarNONE_View,//View for DST_NONE
      CalendarUK_View,//View for DST_UK
      CalendarUS_View,//View for DST_US
      Record_Table,// Record Table
      TimeSchedule_Table,//TimeSchedule Table
      MQL5Calendar_Table,//MQL5Calendar Table
      AutoDST_Trigger,//Table Trigger for AutoDST
      Record_Trigger//Table Trigger for Record
     };
```

The structure SQLiteMaster purpose is to store the current database object's properties such as type, name and etc. So we can keep track of all the objects in the DBContentsarray.

```
//-- structure to retrieve all the objects in the database
   struct SQLiteMaster
     {
      string         type;//will store object type
      string         name;//will store object's name
      string         tbl_name;//will store table name
      int            rootpage;//will store rootpage
      string         sql;//Will store the sql create statement
     } DBContents[];//Array of type SQLiteMaster
```

In the structure MQL5CalendarContents, we will store an additional properties which are Content and insert variables.

Our string variable insert, will store the SQL insertion statements for our SQL objects. Whereas the CalendarComponents variable Content will store the enumeration value for our SQL object as a from of identity, so we can know which SQL object is which, once all object properties are stored in the CalendarContents structure array.

```
//--  MQL5CalendarContents inherits from SQLiteMaster structure
   struct MQL5CalendarContents:SQLiteMaster
     {
      CalendarComponents  Content;
      string         insert;//Will store the sql insert statement
     } CalendarContents[10];//Array to Store objects in our database
```

CalendarStruct function will return the structure MQL5CalendarContents value, when the Content parameter is equal to the enumeration value in the variable Content in the CalendarContents array structure.

```
//-- Function for retrieving the MQL5CalendarContents structure for the enumartion type CalendarComponents
   MQL5CalendarContents CalendarStruct(CalendarComponents Content)
     {
      MQL5CalendarContents Calendar;
      for(uint i=0;i<CalendarContents.Size();i++)
        {
         if(CalendarContents[i].Content==Content)
           {
            return CalendarContents[i];
           }
        }
      return Calendar;
     }
```

The string variable DropRequest will be responsible for dropping database objects we no longer need or want. In the SQL query we make use of PRAGMA statements.

### What is a PRAGMA statement?

A [PRAGMA statement](https://www.mql5.com/go?link=https://www.sqlite.org/pragma.html "https://www.sqlite.org/pragma.html") in SQLite is a special command used to modify the operation of the SQLite library or to query the internal state of the database engine. PRAGMAs are not part of the standard SQL but are specific to SQLite. They provide a way to control various environmental settings and database behaviors.

### What are the purposes of PRAGMA statements?

- Configuration: PRAGMA statements allow you to configure the database environment, such as enabling or disabling foreign key constraints, setting the journal mode, or adjusting memory usage parameters.
- Diagnostics: They can be used to retrieve information about the database, such as checking the integrity of the database, obtaining the current settings, or viewing the status of the SQLite engine.
- Optimization: PRAGMAs help in optimizing database performance by tuning parameters like cache size, locking mode, and synchronous settings.
- Maintenance: They are useful for maintenance tasks such as rebuilding indexes, analyzing tables, and managing the auto-vacuum setting.

In our first PRAGMA statement we disable any foreign key constraints that would prevent us from dropping any table with foreign key constraints.

In our second PRAGMA statement we enable secure\_delete, which controls whether or not deleted content is zeroed out before being removed from the database file. In this case the database will overwrite deleted content with zeros only if doing so does not increase the amount of I/O.

In our third statement we will drop the SQL object if it exists. We will then use the Vacuum command, which will rebuild the database file, repacking it into a minimal amount of disk space. This process can help to optimize the database's performance and reclaim unused space.

Finally we will re-enable foreign key constraints.

```
CNews::CNews(void):DropRequest("PRAGMA foreign_keys = OFF; "
                                  "PRAGMA secure_delete = ON; "
                                  "Drop %s IF EXISTS %s; "
                                  "Vacuum; "
                                  "PRAGMA foreign_keys = ON;")//Sql drop statement
```

We will store the properties for our AutoDST table into the first index of CalendarContents array. Here in the Content variable we assign an enumeration value of AutoDST\_Table.

We then assign the name, table name, type and insert statement. In the SQL statement to create the table, we give the column 'DST' a default value of 'DST\_NONE' and end the statement will a ' [STRICT](https://www.mql5.com/go?link=https://www.sqlite.org/stricttables.html%23%3a%7e%3atext%3dThe%2520STRICT%2520keyword%2520at%2520the%2c(except%2520as%2520noted%2520below). "SQLite documentation")' keyword.

The keyword STRICT, enforces that the data inserted into a column must match the declared type of that column. This provides more predictability and consistency in the type of data stored.

Whereas without the keyword STRICT, essentially any datatype can be inserted into the table and the declared column's datatype is treated as a recommendation rather than a requirement.

```
//-- initializing properties for the AutoDST table
   CalendarContents[0].Content = AutoDST_Table;
   CalendarContents[0].name = "AutoDST";
   CalendarContents[0].sql = "CREATE TABLE AutoDST(DST TEXT NOT NULL DEFAULT 'DST_NONE')STRICT;";
   CalendarContents[0].tbl_name = "AutoDST";
   CalendarContents[0].type = "table";
   CalendarContents[0].insert = "INSERT INTO 'AutoDST'(DST) VALUES ('%s');";
```

We will now initialize the Calendar views' properties for UK,US,AU and NONE.

In the view\_sql variable we store our SQL statement to create the individual views. In our SQL statement we select the Eventid, Eventname, Country, EventCurrency and Eventcode from the MQL5Calendar table.

```
ID      EVENTID 	COUNTRY 	EVENTNAME       				EVENTTYPE       	EVENTIMPORTANCE 		EVENTCURRENCY   EVENTCODE       EVENTSECTOR     		EVENTFORECAST   EVENTPREVALUE   EVENTIMPACT     		EVENTFREQUENCY
18742   999020002       European Union  Eurogroup Meeting       			CALENDAR_TYPE_EVENT     CALENDAR_IMPORTANCE_MODERATE    EUR     	EU      	CALENDAR_SECTOR_GOVERNMENT      None    	None    	CALENDAR_IMPACT_NA      	CALENDAR_FREQUENCY_NONE
18746   999010020       European Union  ECB Executive Board Member Lane Speech  	CALENDAR_TYPE_EVENT     CALENDAR_IMPORTANCE_MODERATE    EUR     	EU      	CALENDAR_SECTOR_MONEY   	None    	None    	CALENDAR_IMPACT_NA      	CALENDAR_FREQUENCY_NONE
34896   392010004       Japan   	Coincident Index        			CALENDAR_TYPE_INDICATOR CALENDAR_IMPORTANCE_LOW 	JPY     	JP      	CALENDAR_SECTOR_BUSINESS        113900000       113900000       CALENDAR_IMPACT_NEGATIVE        CALENDAR_FREQUENCY_MONTH
34897   392010005       Japan   	Leading Index   				CALENDAR_TYPE_INDICATOR CALENDAR_IMPORTANCE_LOW 	JPY     	JP      	CALENDAR_SECTOR_BUSINESS        111400000       111400000       CALENDAR_IMPACT_POSITIVE        CALENDAR_FREQUENCY_MONTH
34898   392010011       Japan   	Coincident Index m/m    			CALENDAR_TYPE_INDICATOR CALENDAR_IMPORTANCE_LOW 	JPY     	JP      	CALENDAR_SECTOR_BUSINESS        None    	2400000 	CALENDAR_IMPACT_NA      	CALENDAR_FREQUENCY_MONTH
34899   392010012       Japan   	Leading Index m/m       			CALENDAR_TYPE_INDICATOR CALENDAR_IMPORTANCE_LOW 	JPY     	JP      	CALENDAR_SECTOR_BUSINESS        None    	-700000 	CALENDAR_IMPACT_NA      	CALENDAR_FREQUENCY_MONTH
55462   156010014       China   	Industrial Profit YTD y/y       		CALENDAR_TYPE_INDICATOR CALENDAR_IMPORTANCE_LOW 	CNY     	CN      	CALENDAR_SECTOR_BUSINESS        None    	4300000 	CALENDAR_IMPACT_NA      	CALENDAR_FREQUENCY_MONTH
72568   276030001       Germany 	Ifo Business Expectations       		CALENDAR_TYPE_INDICATOR CALENDAR_IMPORTANCE_MODERATE    EUR     	DE      	CALENDAR_SECTOR_BUSINESS        92000000        89900000        CALENDAR_IMPACT_NA      	CALENDAR_FREQUENCY_MONTH
72569   276030002       Germany 	Ifo Current Business Situation  		CALENDAR_TYPE_INDICATOR CALENDAR_IMPORTANCE_MODERATE    EUR     	DE      	CALENDAR_SECTOR_BUSINESS        88800000        88900000        CALENDAR_IMPACT_NA      	CALENDAR_FREQUENCY_MONTH
72570   276030003       Germany 	Ifo Business Climate    			CALENDAR_TYPE_INDICATOR CALENDAR_IMPORTANCE_HIGH        EUR     	DE      	CALENDAR_SECTOR_BUSINESS        89900000        89400000        CALENDAR_IMPACT_NA      	CALENDAR_FREQUENCY_MONTH
72571   276050007       Germany 	Bbk Executive Board Member Mauderer Speech      CALENDAR_TYPE_EVENT     CALENDAR_IMPORTANCE_MODERATE    EUR     	DE      	CALENDAR_SECTOR_MONEY   	None    	None    	CALENDAR_IMPACT_NA      	CALENDAR_FREQUENCY_NONE
78850   250020001       France  	3-Month BTF Auction     			CALENDAR_TYPE_INDICATOR CALENDAR_IMPORTANCE_LOW 	EUR     	FR      	CALENDAR_SECTOR_MARKET  	None    	3746000 	CALENDAR_IMPACT_NA      	CALENDAR_FREQUENCY_NONE
78851   250020002       France  	6-Month BTF Auction     			CALENDAR_TYPE_INDICATOR CALENDAR_IMPORTANCE_LOW 	EUR     	FR      	CALENDAR_SECTOR_MARKET  	None    	3657000 	CALENDAR_IMPACT_NA      	CALENDAR_FREQUENCY_NONE
78852   250020003       France  	12-Month BTF Auction    			CALENDAR_TYPE_INDICATOR CALENDAR_IMPORTANCE_LOW 	EUR     	FR      	CALENDAR_SECTOR_MARKET  	None    	3467000 	CALENDAR_IMPACT_NA      	CALENDAR_FREQUENCY_NONE
84771   76020007        Brazil  	BCB Bank Lending m/m    			CALENDAR_TYPE_INDICATOR CALENDAR_IMPORTANCE_LOW 	BRL     	BR      	CALENDAR_SECTOR_MONEY   	400000  	1200000 	CALENDAR_IMPACT_NA      	CALENDAR_FREQUENCY_MONTH
84772   76020001        Brazil  	BCB Focus Market Report 			CALENDAR_TYPE_EVENT     CALENDAR_IMPORTANCE_MODERATE    BRL     	BR      	CALENDAR_SECTOR_MONEY   	None    	None    	CALENDAR_IMPACT_NA      	CALENDAR_FREQUENCY_NONE
94938   344020004       Hong Kong       Exports y/y     				CALENDAR_TYPE_INDICATOR CALENDAR_IMPORTANCE_LOW 	HKD     	HK      	CALENDAR_SECTOR_TRADE   	18100000        4700000 	CALENDAR_IMPACT_NA      	CALENDAR_FREQUENCY_MONTH
94939   344020005       Hong Kong       Imports y/y     				CALENDAR_TYPE_INDICATOR CALENDAR_IMPORTANCE_LOW 	HKD     	HK      	CALENDAR_SECTOR_TRADE   	15000000        5300000 	CALENDAR_IMPACT_NA      	CALENDAR_FREQUENCY_MONTH
94940   344020006       Hong Kong       Trade Balance   				CALENDAR_TYPE_INDICATOR CALENDAR_IMPORTANCE_MODERATE    HKD     	HK      	CALENDAR_SECTOR_TRADE   	-29054000       -45000000       CALENDAR_IMPACT_NA      	CALENDAR_FREQUENCY_MONTH
102731  578020001       Norway  	Unemployment Rate       			CALENDAR_TYPE_INDICATOR CALENDAR_IMPORTANCE_MODERATE    NOK     	NO      	CALENDAR_SECTOR_JOBS    	3700000 	4000000 	CALENDAR_IMPACT_NA      	CALENDAR_FREQUENCY_MONTH
102732  578020020       Norway  	General Public Domestic Loan Debt y/y   	CALENDAR_TYPE_INDICATOR CALENDAR_IMPORTANCE_LOW 	NOK     	NO      	CALENDAR_SECTOR_MONEY   	3300000 	3500000 	CALENDAR_IMPACT_NA      	CALENDAR_FREQUENCY_MONTH
147163  840031004       United States   Memorial Day    				CALENDAR_TYPE_HOLIDAY   CALENDAR_IMPORTANCE_NONE        USD     	US      	CALENDAR_SECTOR_HOLIDAYS        None    	None    	CALENDAR_IMPACT_NA      	CALENDAR_FREQUENCY_NONE
162245  826090005       United Kingdom  Spring Bank Holiday     			CALENDAR_TYPE_HOLIDAY   CALENDAR_IMPORTANCE_NONE        GBP     	GB      	CALENDAR_SECTOR_HOLIDAYS        None    	None    	CALENDAR_IMPACT_NA      	CALENDAR_FREQUENCY_NONE
```

We then select the DST column for the respective schedule from TimeSchedule.

```
ID      DST_UK  		DST_US  		DST_AU  		DST_NONE
18742   2024.05.27 02:00        2024.05.27 02:00        2024.05.27 02:00        2024.05.27 02:00
18746   2024.05.27 14:00        2024.05.27 14:00        2024.05.27 14:00        2024.05.27 14:00
34896   2024.05.27 07:00        2024.05.27 07:00        2024.05.27 07:00        2024.05.27 07:00
34897   2024.05.27 07:00        2024.05.27 07:00        2024.05.27 07:00        2024.05.27 07:00
34898   2024.05.27 07:00        2024.05.27 07:00        2024.05.27 07:00        2024.05.27 07:00
34899   2024.05.27 07:00        2024.05.27 07:00        2024.05.27 07:00        2024.05.27 07:00
55462   2024.05.27 03:30        2024.05.27 03:30        2024.05.27 03:30        2024.05.27 03:30
72568   2024.05.27 10:30        2024.05.27 10:30        2024.05.27 10:30        2024.05.27 10:30
72569   2024.05.27 10:30        2024.05.27 10:30        2024.05.27 10:30        2024.05.27 10:30
72570   2024.05.27 10:30        2024.05.27 10:30        2024.05.27 10:30        2024.05.27 10:30
72571   2024.05.27 15:30        2024.05.27 15:30        2024.05.27 15:30        2024.05.27 15:30
78850   2024.05.27 14:50        2024.05.27 14:50        2024.05.27 14:50        2024.05.27 14:50
78851   2024.05.27 14:50        2024.05.27 14:50        2024.05.27 14:50        2024.05.27 14:50
78852   2024.05.27 14:50        2024.05.27 14:50        2024.05.27 14:50        2024.05.27 14:50
84771   2024.05.27 13:30        2024.05.27 13:30        2024.05.27 13:30        2024.05.27 13:30
84772   2024.05.27 13:30        2024.05.27 13:30        2024.05.27 13:30        2024.05.27 13:30
94938   2024.05.27 10:30        2024.05.27 10:30        2024.05.27 10:30        2024.05.27 10:30
94939   2024.05.27 10:30        2024.05.27 10:30        2024.05.27 10:30        2024.05.27 10:30
94940   2024.05.27 10:30        2024.05.27 10:30        2024.05.27 10:30        2024.05.27 10:30
102731  2024.05.27 08:00        2024.05.27 08:00        2024.05.27 08:00        2024.05.27 08:00
102732  2024.05.27 08:00        2024.05.27 08:00        2024.05.27 08:00        2024.05.27 08:00
147163  2024.05.27 02:00        2024.05.27 02:00        2024.05.27 02:00        2024.05.27 02:00
162245  2024.05.27 02:00        2024.05.27 02:00        2024.05.27 02:00        2024.05.27 02:00
```

And join the two tables MQL5Calendar and TimeSchedule on the same ID.

We filter this list with the Date from the Record table.

```
Date
27-05-2024
```

Once we get the results from the query, the results are sorted in ascending order based of the respective DST time from TimeSchedule.

```
   string views[] = {"UK","US","AU","NONE"};
   string view_sql = "CREATE VIEW IF NOT EXISTS Calendar_%s "
                     "AS "
                     "SELECT C.Eventid,C.Eventname,C.Country,T.DST_%s as Time,C.EventCurrency,C.Eventcode from MQL5Calendar C,Record R "
                     "Inner join TimeSchedule T on C.ID=T.ID "
                     "Where DATE(REPLACE(T.DST_%s,'.','-'))=R.Date "
                     "Order by T.DST_%s Asc;";

//-- Sql statements for creating the table views
   for(uint i=1;i<=views.Size();i++)
     {
      CalendarContents[i].Content = (CalendarComponents)i;
      CalendarContents[i].name = StringFormat("Calendar_%s",views[i-1]);
      CalendarContents[i].sql = StringFormat(view_sql,views[i-1],views[i-1],views[i-1],views[i-1]);
      CalendarContents[i].tbl_name = StringFormat("Calendar_%s",views[i-1]);
      CalendarContents[i].type = "view";
     }
```

We will have a look at one of the views and see what the query result will produce:

```
SELECT * FROM 'Calendar_UK';
```

Output:

```
EVENTID 	EVENTNAME       				COUNTRY 	Time    		EVENTCURRENCY   EVENTCODE
999020002       Eurogroup Meeting       			European Union  2024.05.27 02:00        EUR     	EU
840031004       Memorial Day    				United States   2024.05.27 02:00        USD     	US
826090005       Spring Bank Holiday     			United Kingdom  2024.05.27 02:00        GBP     	GB
156010014       Industrial Profit YTD y/y       		China   	2024.05.27 03:30        CNY     	CN
392010004       Coincident Index        			Japan   	2024.05.27 07:00        JPY     	JP
392010005       Leading Index   				Japan   	2024.05.27 07:00        JPY     	JP
392010011       Coincident Index m/m    			Japan   	2024.05.27 07:00        JPY     	JP
392010012       Leading Index m/m       			Japan   	2024.05.27 07:00        JPY     	JP
578020001       Unemployment Rate       			Norway  	2024.05.27 08:00        NOK     	NO
578020020       General Public Domestic Loan Debt y/y   	Norway  	2024.05.27 08:00        NOK     	NO
276030001       Ifo Business Expectations       		Germany 	2024.05.27 10:30        EUR     	DE
276030002       Ifo Current Business Situation  		Germany 	2024.05.27 10:30        EUR     	DE
276030003       Ifo Business Climate    			Germany 	2024.05.27 10:30        EUR     	DE
344020004       Exports y/y     				Hong Kong       2024.05.27 10:30        HKD     	HK
344020005       Imports y/y     				Hong Kong       2024.05.27 10:30        HKD     	HK
344020006       Trade Balance   				Hong Kong       2024.05.27 10:30        HKD     	HK
76020007        BCB Bank Lending m/m    			Brazil  	2024.05.27 13:30        BRL     	BR
76020001        BCB Focus Market Report 			Brazil  	2024.05.27 13:30        BRL     	BR
999010020       ECB Executive Board Member Lane Speech  	European Union  2024.05.27 14:00        EUR     	EU
250020001       3-Month BTF Auction     			France  	2024.05.27 14:50        EUR     	FR
250020002       6-Month BTF Auction     			France  	2024.05.27 14:50        EUR     	FR
250020003       12-Month BTF Auction    			France  	2024.05.27 14:50        EUR     	FR
276050007       Bbk Executive Board Member Mauderer Speech      Germany 	2024.05.27 15:30        EUR     	DE
```

We technically have a new table called Record, this table will replace our previous Records table, as we will now only be storing one a record. Our table will have the datatype of TEXT and column name 'Date' which shouldn't be confused with the Date function in SQLite.

```
//-- initializing properties for the Record table
   CalendarContents[5].Content = Record_Table;
   CalendarContents[5].name = "Record";
   CalendarContents[5].sql = "CREATE TABLE Record(Date TEXT NOT NULL)STRICT;";
   CalendarContents[5].tbl_name="Record";
   CalendarContents[5].type = "table";
   CalendarContents[5].insert = "INSERT INTO 'Record'(Date) VALUES (Date(REPLACE('%s','.','-')));";
```

Our TimeSchedule table will store all the individual events time data and will use the foreign key reference 'ID' to link the table(create a relationship) to the MQL5Calendar table.

```
//-- initializing properties for the TimeSchedule table
   CalendarContents[6].Content = TimeSchedule_Table;
   CalendarContents[6].name = "TimeSchedule";
   CalendarContents[6].sql = "CREATE TABLE TimeSchedule(ID INT NOT NULL,DST_UK   TEXT   NOT NULL,DST_US   TEXT   NOT NULL,"
                             "DST_AU   TEXT   NOT NULL,DST_NONE   TEXT   NOT NULL,FOREIGN KEY (ID) REFERENCES MQL5Calendar (ID))STRICT;";
   CalendarContents[6].tbl_name="TimeSchedule";
   CalendarContents[6].type = "table";
   CalendarContents[6].insert = "INSERT INTO 'TimeSchedule'(ID,DST_UK,DST_US,DST_AU,DST_NONE) "
                                "VALUES (%d,'%s','%s', '%s', '%s');";
```

MQL5Calendar table will have a Primary key called 'ID' which will be unique for each news event record in the table.

```
//-- initializing properties for the MQL5Calendar table
   CalendarContents[7].Content = MQL5Calendar_Table;
   CalendarContents[7].name = "MQL5Calendar";
   CalendarContents[7].sql = "CREATE TABLE MQL5Calendar(ID INT NOT NULL,EVENTID  INT   NOT NULL,COUNTRY  TEXT   NOT NULL,"
                             "EVENTNAME   TEXT   NOT NULL,EVENTTYPE   TEXT   NOT NULL,EVENTIMPORTANCE   TEXT   NOT NULL,"
                             "EVENTCURRENCY  TEXT   NOT NULL,EVENTCODE   TEXT   NOT NULL,EVENTSECTOR TEXT   NOT NULL,"
                             "EVENTFORECAST  TEXT   NOT NULL,EVENTPREVALUE  TEXT   NOT NULL,EVENTIMPACT TEXT   NOT NULL,"
                             "EVENTFREQUENCY TEXT   NOT NULL,PRIMARY KEY(ID))STRICT;";
   CalendarContents[7].tbl_name="MQL5Calendar";
   CalendarContents[7].type = "table";
   CalendarContents[7].insert = "INSERT INTO 'MQL5Calendar'(ID,EVENTID,COUNTRY,EVENTNAME,EVENTTYPE,EVENTIMPORTANCE,EVENTCURRENCY,EVENTCODE,"
                                "EVENTSECTOR,EVENTFORECAST,EVENTPREVALUE,EVENTIMPACT,EVENTFREQUENCY) "
                                "VALUES (%d,%d,'%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s');";
```

We will create a trigger called OnlyOne\_AutoDST. The trigger will begin when we attempt to insert a value into the AutoDST and delete all records from AutoDSTbefore inserting a new record.

```
//-- Sql statement for creating the AutoDST table's trigger
   CalendarContents[8].Content = AutoDST_Trigger;
   CalendarContents[8].name = "OnlyOne_AutoDST";
   CalendarContents[8].sql = "CREATE TRIGGER IF NOT EXISTS OnlyOne_AutoDST "
                             "BEFORE INSERT ON AutoDST "
                             "BEGIN "
                             "Delete from AutoDST; "
                             "END;";
   CalendarContents[8].tbl_name="AutoDST";
   CalendarContents[8].type = "trigger";
```

The same can be said for OnlyOne\_Record, but this trigger is in relation to the Record table.

```
//-- Sql statement for creating the Record table's trigger
   CalendarContents[9].Content = Record_Trigger;
   CalendarContents[9].name = "OnlyOne_Record";
   CalendarContents[9].sql = "CREATE TRIGGER IF NOT EXISTS OnlyOne_Record "
                             "BEFORE INSERT ON Record "
                             "BEGIN "
                             "Delete from Record; "
                             "END;";
   CalendarContents[9].tbl_name="Record";
   CalendarContents[9].type = "trigger";
```

Now in our function UpdateRecords we will determine if our Calendar database requires an update.

The changes to this function from the previous in part 1 are namely:

1. We will read all the objects that are not indexes present in the database with the SQL query "select \* from sqlite\_master where type<>'index' ; ".

2. We will store all the object's attributes into the array DBContents and if there is no semi-colon present at the end of the sql statement we will add one.

3. We will compare the object's found in our database and the object's we initialize in our array CalendarContents. We will remove ' IF NOT EXISTS' from the CalendarContents sql.

4. We when do not find a match between DBContents and CalendarContents we will proceed to drop the object in the DBcontents index.

5. If the SQL object matches are not equal to the size of CalendarContents we will perform an update.

```
bool CNews::UpdateRecords()
  {
//initialize variable to true
   bool perform_update=true;
//--- open/create
//-- try to open database Calendar
   int db=DatabaseOpen(NEWS_DATABASE_FILE, DATABASE_OPEN_READWRITE | DATABASE_OPEN_CREATE| DATABASE_OPEN_COMMON);
   if(db==INVALID_HANDLE)//Checks if the database was able to be opened
     {
      //if opening the database failed
      if(!FileIsExist(NEWS_DATABASE_FILE,FILE_COMMON))//Checks if the database Calendar exists in the common folder
        {
         return perform_update;//Returns true when the database was failed to be opened and the file doesn't exist in the common folder
        }
     }

   int MasterRequest = DatabasePrepare(db,"select * from sqlite_master where type<>'index';");
   if(MasterRequest==INVALID_HANDLE)
     {
      Print("DB: ",NEWS_DATABASE_FILE, " request failed with code ", GetLastError());
     }
   else
     {
      SQLiteMaster ReadContents;
      //Assigning values from the sql query into DBContents array
      for(int i=0; DatabaseReadBind(MasterRequest,ReadContents); i++)
        {
         ArrayResize(DBContents,i+1,i+2);
         DBContents[i].type = ReadContents.type;
         DBContents[i].name = ReadContents.name;
         DBContents[i].tbl_name = ReadContents.tbl_name;
         DBContents[i].rootpage = ReadContents.rootpage;
         /*Check if the end of the sql string has a character ';' if not add this character to the string*/
         DBContents[i].sql = (StringFind(ReadContents.sql,";",StringLen(ReadContents.sql)-1)==
                              (StringLen(ReadContents.sql)-1))?ReadContents.sql:ReadContents.sql+";";;
        }

      uint contents_exists = 0;
      for(uint i=0;i<DBContents.Size();i++)
        {
         bool isCalendarContents = false;
         for(uint x=0;x<CalendarContents.Size();x++)
           {
            /*Store Sql query from CalendarContents without string ' IF NOT EXISTS'*/
            string CalendarSql=CalendarContents[x].sql;
            StringReplace(CalendarSql," IF NOT EXISTS","");
            //-- Check if the Db object is in our list
            if(DBContents[i].name==CalendarContents[x].name&&
               (DBContents[i].sql==CalendarSql||
                DBContents[i].sql==CalendarContents[x].sql)&&
               CalendarContents[x].type==DBContents[i].type&&
               CalendarContents[x].tbl_name==DBContents[i].tbl_name)
              {
               contents_exists++;
               isCalendarContents = true;
              }
           }
         if(!isCalendarContents)
           {
            //-- Print DBcontent's name if it does not match with CalendarContents
            PrintFormat("DBContent: %s is not needed!",DBContents[i].name);
            //-- We will drop the table if it is not neccessary
            DatabaseExecute(db,StringFormat(DropRequest,DBContents[i].type,DBContents[i].name));
            Print("Attempting To Clean Database...");
           }
        }
      /*If not all the CalendarContents exist in the Calendar Database before an update */
      if(contents_exists!=CalendarContents.Size())
        {
         return perform_update;
        }
     }
   if(!DatabaseTableExists(db,CalendarStruct(Record_Table).name))//If the database table 'Record' doesn't exist
     {
      DatabaseClose(db);
      return perform_update;
     }

//-- Sql query to determine the lastest or maximum date recorded
   /* If the last recorded date data in the 'Record' table is not equal to the current day, perform an update! */
   string request_text=StringFormat("SELECT Date FROM %s where Date=Date(REPLACE('%s','.','-'))",
                                    CalendarStruct(Record_Table).name,TimeToString(TimeTradeServer()));
   int request=DatabasePrepare(db,request_text);//Creates a handle of a request, which can then be executed using DatabaseRead()
   if(request==INVALID_HANDLE)//Checks if the request failed to be completed
     {
      Print("DB: ",NEWS_DATABASE_FILE, " request failed with code ", GetLastError());
      DatabaseClose(db);
      return perform_update;
     }

   if(DatabaseRead(request))//Will be true if there are results from the sql query/request
     {
      DatabaseFinalize(request);//Removes a request created in DatabasePrepare()
      DatabaseClose(db);//Closes the database
      perform_update=false;
      return perform_update;
     }
   else
     {
      DatabaseFinalize(request);//Removes a request created in DatabasePrepare()
      DatabaseClose(db);//Closes the database
      return perform_update;
     }
  }
```

In the function CreateCalendarTable, we will check if the MQL5Calendar table already exists in the calendar database, we will also check if the TimeSchedule table already exists and attempt to drop each table if they exist. Since TimeSchedule requires MQL5Calendar we cannot drop MQL5Calendar without dropping TimeSchedule first.

Once MQL5Calendar does not exist we will create its table.

```
bool CNews::CreateCalendarTable(int db,bool &tableExists)
  {
//-- Checks if a table 'MQL5Calendar' exists
   if(DatabaseTableExists(db,CalendarStruct(MQL5Calendar_Table).name))
     {
      tableExists=true;//Assigns true to tableExists variable
      //-- Checks if a table 'TimeSchedule' exists in the database 'Calendar'
      if(DatabaseTableExists(db,CalendarStruct(TimeSchedule_Table).name))
        {
         //-- We will drop the table if the table already exists
         if(!DatabaseExecute(db,StringFormat("Drop Table %s",CalendarStruct(TimeSchedule_Table).name)))
           {
            //If the table failed to be dropped/deleted
            PrintFormat("Failed to drop table %s with code %d",CalendarStruct(TimeSchedule_Table).name,GetLastError());
            DatabaseClose(db);//Close the database
            return false;//will terminate execution of the rest of the code below and return false, when the table cannot be dropped
           }
        }
      //--We will drop the table if the table already exists
      if(!DatabaseExecute(db,StringFormat("Drop Table %s",CalendarStruct(MQL5Calendar_Table).name)))
        {
         //If the table failed to be dropped/deleted
         PrintFormat("Failed to drop table %s with code %d",CalendarStruct(MQL5Calendar_Table).name,GetLastError());
         DatabaseClose(db);//Close the database
         return false;//will terminate execution of the rest of the code below and return false, when the table cannot be dropped
        }
     }
//-- If the database table 'MQL5Calendar' doesn't exist
   if(!DatabaseTableExists(db,CalendarStruct(MQL5Calendar_Table).name))
     {
      //--- create the table 'MQL5Calendar'
      if(!DatabaseExecute(db,CalendarStruct(MQL5Calendar_Table).sql))//Checks if the table was successfully created
        {
         Print("DB: create the Calendar table failed with code ", GetLastError());
         DatabaseClose(db);//Close the database
         return false;//Function returns false if creating the table failed
        }
     }
   return true;//Function returns true if creating the table was successful
  }
```

In the function CreateTimeTable, we verify if the table exists in the Calendar database if not we create it.

```
bool CNews::CreateTimeTable(int db,bool &tableExists)
  {
//-- If the database table 'TimeSchedule' doesn't exist
   if(!DatabaseTableExists(db,CalendarStruct(TimeSchedule_Table).name))
     {
      //--- create the table 'TimeSchedule'
      if(!DatabaseExecute(db,CalendarStruct(TimeSchedule_Table).sql))//Checks if the table was successfully created
        {
         Print("DB: create the Calendar table failed with code ", GetLastError());
         DatabaseClose(db);//Close the database
         return false;//Function returns false if creating the table failed
        }
     }
   return true;//Function returns true if creating the table was successful
  }
```

In the function CreateCalendarViews, we create all the views using CalendarComponents to find the identity(Enumeration value) for each view and create it.

```
void CNews::CreateCalendarViews(int db)
  {
   for(uint i=1;i<=4;i++)
     {
      if(!DatabaseExecute(db,CalendarStruct((CalendarComponents)i).sql))//Checks if the view was successfully created
        {
         Print("DB: create the Calendar view failed with code ", GetLastError());
        }
     }
  }
```

In the function InsertIntoTables, we will insert every record from Evalues array into both MQL5Calendar table and TimeSchedule table respectively. The event dates will be adjusted for the various DST schedules in TimeSchedule.

```
bool CNews::InsertIntoTables(int db,Calendar &Evalues[])
  {
   for(uint i=0; i<Evalues.Size(); i++)//Looping through all the Economic Events
     {
      string request_insert_into_calendar =
         StringFormat(CalendarStruct(MQL5Calendar_Table).insert,
                      i,
                      Evalues[i].EventId,
                      Evalues[i].CountryName,
                      Evalues[i].EventName,
                      Evalues[i].EventType,
                      Evalues[i].EventImportance,
                      Evalues[i].EventCurrency,
                      Evalues[i].EventCode,
                      Evalues[i].EventSector,
                      Evalues[i].EventForecast,
                      Evalues[i].EventPreval,
                      Evalues[i].EventImpact,
                      Evalues[i].EventFrequency);//Inserting all the columns for each event record
      if(DatabaseExecute(db,request_insert_into_calendar))//Check if insert query into calendar was successful
        {
         string request_insert_into_time =
            StringFormat(CalendarStruct(TimeSchedule_Table).insert,
                         i,
                         //-- Economic EventDate adjusted for UK DST(Daylight Savings Time)
                         Savings_UK.adjustDaylightSavings(StringToTime(Evalues[i].EventDate)),
                         //-- Economic EventDate adjusted for US DST(Daylight Savings Time)
                         Savings_US.adjustDaylightSavings(StringToTime(Evalues[i].EventDate)),
                         //-- Economic EventDate adjusted for AU DST(Daylight Savings Time)
                         Savings_AU.adjustDaylightSavings(StringToTime(Evalues[i].EventDate)),
                         Evalues[i].EventDate//normal Economic EventDate
                        );//Inserting all the columns for each event record
         if(!DatabaseExecute(db,request_insert_into_time))
           {
            Print(GetLastError());
            //-- Will print the sql query to check for any errors or possible defaults in the query/request
            Print(request_insert_into_time);
            return false;//Will end the loop and return false, as values failed to be inserted into the table
           }
        }
      else
        {
         Print(GetLastError());
         //-- Will print the sql query to check for any errors or possible defaults in the query/request
         Print(request_insert_into_calendar);
         return false;//Will end the loop and return false, as values failed to be inserted into the table
        }
     }
   return true;//Will return true, all values were inserted into the table successfully
  }
```

In the function CreateRecordTable, we check if the Record table already exists, if it doesn't exist we will create the table. Once the Record table exists we will create the Trigger of it. And proceed to insert the Current server date.

### Why use TimeTradeServer instead of TimeCurrent?

If we use TimeCurrent we will get the time data for the current chart symbol and symbol times may vary between each other as the time data is updated every new tick. This is potentially problematic when the symbols don't have the same trading hours and when the current chart symbol could be closed, meaning no new ticks are being received therefore the TimeCurrent could return a date which is a day or more behind the actual date. Whereas TimeTradeServer is consistently updated regardless of the type of Symbol.

```
void CNews::CreateRecordTable(int db)
  {
   bool failed=false;
   if(!DatabaseTableExists(db,CalendarStruct(Record_Table).name))//Checks if the table 'Record' exists in the databse 'Calendar'
     {
      //--- create the table
      if(!DatabaseExecute(db,CalendarStruct(Record_Table).sql))//Will attempt to create the table 'Record'
        {
         Print("DB: create the Records table failed with code ", GetLastError());
         DatabaseClose(db);//Close the database
         return;//Exits the function if creating the table failed
        }
      else//If Table was created Successfully then Create Trigger
        {
         DatabaseExecute(db,CalendarStruct(Record_Trigger).sql);
        }
     }
   else
     {
      DatabaseExecute(db,CalendarStruct(Record_Trigger).sql);
     }
//Sql query/request to insert the current time into the 'Date' column in the table 'Record'
   string request_text=StringFormat(CalendarStruct(Record_Table).insert,TimeToString(TimeTradeServer()));
   if(!DatabaseExecute(db, request_text))//Will attempt to run this sql request/query
     {
      Print(GetLastError());
      PrintFormat(CalendarStruct(Record_Table).insert,TimeToString(TimeTradeServer()));
      failed=true;//assign true if the request failed
     }
   if(failed)
     {
      //--- roll back all transactions and unlock the database
      DatabaseTransactionRollback(db);
      PrintFormat("%s: DatabaseExecute() failed with code %d", __FUNCTION__, GetLastError());
     }
  }
```

In the function CreateAutoDST, we check the broker's DST schedule, if we successfully get the DST schedule we then check if the AutoDST table exists in the Calendar database. If the AutoDST table doesn't exist it will be created. Once the AutoDST table exists we then create its Trigger and attempt to insert the DST schedule converted from an enumeration into a string.

```
void CNews::CreateAutoDST(int db)
  {
   bool failed=false;//boolean variable
   if(!AutoDetectDST(DSTType))//Check if AutoDetectDST went through all the right procedures
     {
      return;//will terminate execution of the rest of the code below
     }

   if(!DatabaseTableExists(db,CalendarStruct(AutoDST_Table).name))//Checks if the table 'AutoDST' exists in the databse 'Calendar'
     {
      //--- create the table AutoDST
      if(!DatabaseExecute(db,CalendarStruct(AutoDST_Table).sql))//Will attempt to create the table 'AutoDST'
        {
         Print("DB: create the AutoDST table failed with code ", GetLastError());
         DatabaseClose(db);//Close the database
         return;//Exits the function if creating the table failed
        }
      else//If Table was created Successfully then Create Trigger
        {
         DatabaseExecute(db,CalendarStruct(AutoDST_Trigger).sql);
        }
     }
   else
     {
      //Create trigger if AutoDST table exists
      DatabaseExecute(db,CalendarStruct(AutoDST_Trigger).sql);
     }
//Sql query/request to insert the recommend DST for the Broker using the DSTType variable to determine which string data to insert
   string request_text=StringFormat(CalendarStruct(AutoDST_Table).insert,EnumToString(DSTType));
   if(!DatabaseExecute(db, request_text))//Will attempt to run this sql request/query
     {
      Print(GetLastError());
      PrintFormat(CalendarStruct(AutoDST_Table).insert,EnumToString(DSTType));//Will print the sql query if failed
      failed=true;//assign true if the request failed
     }
   if(failed)
     {
      //--- roll back all transactions and unlock the database
      DatabaseTransactionRollback(db);
      PrintFormat("%s: DatabaseExecute() failed with code %d", __FUNCTION__, GetLastError());
     }
  }
```

## Risk Management Class

Risk management is a critical component of successful trading. The primary goal of risk management is to protect trading capital. Without capital, a trader cannot continue to trade. Implementing strategies to limit losses ensures that traders can stay in the market longer, providing more opportunities to recover from setbacks and achieve overall profitability. In this case we will provide the user with different risk profiles to choose from and find the most suitable option.

Disclaimer: I will reference Lot, Lot-size and volume interchangeably. Consider them the same in context to risk management.

List of risk profiles

- Minimum Lot-size
- Maximum Lot-size
- Percentage of Balance
- Percentage of Free-Margin
- Risk in Amount per Balance
- Risk in Amount per Free-Margin
- Lot-size per Balance
- Lots-size per Free-Margin
- Custom Lot-size
- Percentage of Risk

### Minimum Lot-size:

In this Risk option, we will use the minimum allowed lot-size for the current symbol.

### Maximum Lot-size:

In this Risk option, we will use the maximum allowed lot-size for the current symbol.

### Percentage of Balance:

In this Risk option, we will first get the amount of risk.

```
amount_of_risk = Balance*Percent;
```

Let Percent = 5% and the account balance is 10,000.

```
amount_of_risk = 10000*(5/100);
amount_of_risk = 500;
```

We will then need an open-price and close-price to calculate the Minimum risk for the specific trade when using the Minimum lot-size.

```
OrderCalcProfit(ORDER_TYPE,Symbol(),Minimum_lotsize,OpenPrice,ClosePrice,Minimum_risk);
```

Once the Minimum\_risk is returned, we will use the following equation to get the required Lot-size for the amount\_of\_risk

```
required_lotsize = (amount_of_risk/Minimum_risk)*Minimum_lotsize;
```

Let  Minimum\_risk = 100  and  Minimum\_lotsize = 0.1;

```
required_lotsize = (500/100)*0.1;
required_lotsize = 5*0.1;
required_lotsize = 0.5;
```

### Percentage of Free-Margin:

This risk option is similar to Percentage of Balance. But the advantages of this risk option comes into play when there are open trades in the traders account.

Whereas the risk remains the same in Percentage of Balance regardless of open trades as long as the balance is the same. With Percentage of Free-Margin the risk changes as the open trades' profit fluctuates. This gives a more accurate risk calculation for the current account's condition.

### Risk in Amount per Balance:

In this Risk option, we first need to obtain the [quotient](https://www.mql5.com/go?link=https://www.cuemath.com/numbers/quotient/ "what is a quotient?") between Risk in Amount(the divisor) and Balance(Dividend).

```
risk = Balance/Risk_in_Amount;
```

We will then let Balance = 10,000 and Risk\_in\_Amount = 800;

In this case we basically want to risk $800 per trade for every $10,000 in the traders account balance.

```
risk = 10000/800;
risk = 12.5;
```

We will then divide risk by the actual account balance to get our risk amount.

```
amount_of_risk = AccountBalance/risk;
```

Let AccountBalance = 5000;

```
amount_of_risk = 5000/12.5;
amount_of_risk = 400;
```

Now we know that the trader wants to risk 400 dollars for this particular trade.

### Risk in Amount per Free-Margin:

This risk option is similar to Risk in Amount per Balance, we will just go through another example.

```
risk = FreeMargin/Risk_in_Amount;
```

We will let FreeMargin = 150 and Risk\_in\_Amount = 1;

In this case we will risk $1 for every $150 in FreeMargin.

```
risk = 150/1;
risk = 150;

amount_of_risk = AccountFreeMargin/risk;

//-- Let AccountFreeMargin = 750

amount_of_risk = 750/150;
amount_of_risk = 5;
```

After getting the risk amount we will then calculate the required Lot-size for that specific trade to meet 5 dollars in risk.

### Lot-size per Balance:

In this Risk option the trader will provide the lot-size the would like to risk for a certain account balance.

```
required_lotsize = (AccountBalance/Balance)*lotsize;
```

Where AccountBalance is the traders actual account balance, Balance and lotsize are input values provided by the trader.

Let AccountBalance = 10,000 and Balance = 350 and lotsize = 0.01

In this case the trader wants to risk 0.01 lots for every 350 dollars in their account balance.

```
required_lotsize = (10000/350)*0.01;
required_lotsize = 0.285;
```

The required\_lotsize is 0.285, the actual value is a lot longer. Lets assume that the Volume step for the specific Symbol we want to open a trade in is 0.01. Trying to open a trade with a 0.285 lot-size when the Volume step is 0.01 will cause an error.

To prevent this we will normalize the lot-size, basically we will format the lot-size in accordance with the Volume step.

```
required_lotsize = Volume_Step*MathFloor(0.285/Volume_Step);
requred_lotsize = 0.01*MathFloor(0.285/0.01);
required_lotsize = 0.01*MathFloor(28.5);
required_lotsize = 0.01*28;
required_lotsize = 0.28;
```

### Lot-size per Free-Margin:

This Risk option is similar to Lot-size per Balance, we will provide another example.

```
required_lotsize = (AccountFreeMargin/FreeMargin)*lotsize;
```

Let AccountFreeMargin = 134,560 and FreeMargin = 1622 and lot-size = 0.0056

In this case:

```
required_lotsize = (134560/1622)*0.0056;
required_lotsize = 0.464;
```

Lets assume that the Volume step is 0.02.

We will normalize required\_lotsize in accordance with Volume step.

```
//-- normalize for Volume Step
required_lotsize = Volume_Step*MathFloor(0.464/Volume_Step);
requred_lotsize = 0.02*MathFloor(0.464/0.02);
required_lotsize = 0.02*MathFloor(23.2);
required_lotsize = 0.02*23;
required_lotsize = 0.46;
```

### Custom Lot-size:

In this Risk option we will utilize the input lot-size provided by the trader.

### Percentage of Risk:

In this Risk option, we will calculate the percentage of maximum allowable risk for the current symbol using Free-Margin and Margin requirements for the specific symbol.

CRiskManagement class has multilevel Inheritance from classes:

- CSymbolProperties
- CChartProperties

CRiskManagement class has Inclusion from CAccountInfo class.

CRiskManagement class has hierarchical Inheritance from classes:

- CSymbolProperties
- CSymbolInfo

```
//+------------------------------------------------------------------+
//|                                                      NewsTrading |
//|                                  Copyright 2024, MetaQuotes Ltd. |
//|                            https://www.mql5.com/en/users/kaaiblo |
//+------------------------------------------------------------------+
#include "ChartProperties.mqh"
#include <Trade/AccountInfo.mqh>
CAccountInfo      Account;

//-- Enumeration declaration for Risk options
enum RiskOptions
  {
   MINIMUM_LOT,//MINIMUM LOTSIZE
   MAXIMUM_LOT,//MAXIMUM LOTSIZE
   PERCENTAGE_OF_BALANCE,//PERCENTAGE OF BALANCE
   PERCENTAGE_OF_FREEMARGIN,//PERCENTAGE OF FREE-MARGIN
   AMOUNT_PER_BALANCE,//AMOUNT PER BALANCE
   AMOUNT_PER_FREEMARGIN,//AMOUNT PER FREE-MARGIN
   LOTSIZE_PER_BALANCE,//LOTSIZE PER BALANCE
   LOTSIZE_PER_FREEMARGIN,//LOTSIZE PER FREE-MARGIN
   CUSTOM_LOT,//CUSTOM LOTSIZE
   PERCENTAGE_OF_MAXRISK//PERCENTAGE OF MAX-RISK
  } RiskProfileOption;//variable for Risk options

//-- Enumeration declaration for Risk floor
enum RiskFloor
  {
   RiskFloorMin,//MINIMUM LOTSIZE
   RiskFloorMax,//MAX-RISK
   RiskFloorNone//NONE
  } RiskFloorOption;//variable for Risk floor

//-- Enumeration declaration for Risk ceiling(Maximum allowable risk in terms of lot-size)
enum RiskCeil
  {
   RiskCeilMax,//MAX LOTSIZE
   RiskCeilMax2,//MAX LOTSIZE(x2)
   RiskCeilMax3,//MAX LOTSIZE(x3)
   RiskCeilMax4,//MAX LOTSIZE(x4)
   RiskCeilMax5,//MAX LOTSIZE(x5)
  } RiskCeilOption;//variable for Risk ceiling

//-- Structure declaration for Risk options (AMOUNT PER BALANCE and AMOUNT PER FREE-MARGIN)
struct RISK_AMOUNT
  {
   double            RiskAmountBoF;//store Balance or Free-Margin
   double            RiskAmount;//store risk amount
  } Risk_Profile_2;//variable for Risk options (AMOUNT PER BALANCE and AMOUNT PER FREE-MARGIN)

//-- Structure declaration for Risk options (LOTSIZE PER BALANCE and LOTSIZE PER FREE-MARGIN)
struct RISK_LOT
  {
   double            RiskLotBoF;//store Balance or Free-Margin
   double            RiskLot;//store lot-size
  } Risk_Profile_3;//variable for Risk options (LOTSIZE PER BALANCE and LOTSIZE PER FREE-MARGIN)

double            RiskFloorPercentage;//variable for RiskFloorMax
double            Risk_Profile_1;//variable for Risk options (PERCENTAGE OF BALANCE and PERCENTAGE OF FREE-MARGIN)
double            Risk_Profile_4;//variable for Risk option (CUSTOM LOTSIZE)
double            Risk_Profile_5;//variable for Risk option (PERCENTAGE OF MAX-RISK)

//+------------------------------------------------------------------+
//|RiskManagement class                                              |
//+------------------------------------------------------------------+
class CRiskManagement : public CChartProperties
  {

private:
   double            Medium;//variable to store actual Account (Balance or Free-Margin)
   double            RiskAmount,MinimumAmount;
   double            Lots;//variable to store Lot-size to open trade
   const double      max_percent;//variable to store percentage for Maximum risk

   //-- enumeration for dealing with account balance/free-margin
   enum RiskMedium
     {
      BALANCE,
      MARGIN
     };

   //-- calculations for Risk options (PERCENTAGE OF BALANCE and PERCENTAGE OF FREE-MARGIN)
   double              RiskProfile1(const RiskMedium R_Medium);
   //-- calculations for Risk options (AMOUNT PER BALANCE and AMOUNT PER FREE-MARGIN)
   double              RiskProfile2(const RiskMedium R_Medium);
   //-- calculations for Risk options (LOTSIZE PER BALANCE and LOTSIZE PER FREE-MARGIN)
   double              RiskProfile3(const RiskMedium R_Medium);
   //-- calculations for Maximum allowable Risk
   double              MaxRisk(const double percent);
   //-- Store Trade's Open-price
   double              OpenPrice;
   //-- Store Trade's Close-price
   double              ClosePrice;
   //-- Store Ordertype between (ORDER_TYPE_BUY or ORDER_TYPE_SELL) for risk calaculations
   ENUM_ORDER_TYPE     ORDERTYPE;
   //-- Set Medium variable value
   void                SetMedium(const RiskMedium R_Medium) {Medium = (R_Medium==BALANCE)?Account.Balance():Account.FreeMargin();}
   //-- Get Minimum Risk for a Trade using Minimum Lot-size
   bool                GetMinimumRisk()
     {
      return OrderCalcProfit(ORDERTYPE,Symbol(),LotsMin(),OpenPrice,ClosePrice,MinimumAmount);
     }
   //-- Retrieve Risk amount based on Risk inputs
   double            GetRisk(double Amount)
     {
      if(!GetMinimumRisk()||Amount==0)
         return 0.0;
      return ((Amount/MinimumAmount)*LotsMin());
     }

protected:
   //-- Application of Lot-size limits
   void              ValidateLotsize(double &Lotsize);
   //-- Set ORDERTYPE variable to (ORDER_TYPE_BUY or ORDER_TYPE_SELL) respectively
   void              SetOrderType(ENUM_ORDER_TYPE Type)
     {
      if(Type==ORDER_TYPE_BUY||Type==ORDER_TYPE_BUY_LIMIT||Type==ORDER_TYPE_BUY_STOP)
        {
         ORDERTYPE = ORDER_TYPE_BUY;
        }
      else
         if(Type==ORDER_TYPE_SELL||Type==ORDER_TYPE_SELL_LIMIT||Type==ORDER_TYPE_SELL_STOP)
           {
            ORDERTYPE = ORDER_TYPE_SELL;
           }
     }

public:

                     CRiskManagement();//Class's constructor
   //-- Retrieve user's Risk option
   string            GetRiskOption()
     {
      switch(RiskProfileOption)
        {
         case  MINIMUM_LOT://MINIMUM LOTSIZE - Risk Option
            return "MINIMUM LOTSIZE";
            break;
         case MAXIMUM_LOT://MAXIMUM LOTSIZE - Risk Option
            return "MAXIMUM LOTSIZE";
            break;
         case PERCENTAGE_OF_BALANCE://PERCENTAGE OF BALANCE - Risk Option
            return "PERCENTAGE OF BALANCE";
            break;
         case PERCENTAGE_OF_FREEMARGIN://PERCENTAGE OF FREE-MARGIN - Risk Option
            return "PERCENTAGE OF FREE-MARGIN";
            break;
         case AMOUNT_PER_BALANCE://AMOUNT PER BALANCE - Risk Option
            return "AMOUNT PER BALANCE";
            break;
         case AMOUNT_PER_FREEMARGIN://AMOUNT PER FREE-MARGIN - Risk Option
            return "AMOUNT PER FREE-MARGIN";
            break;
         case LOTSIZE_PER_BALANCE://LOTSIZE PER BALANCE - Risk Option
            return "LOTSIZE PER BALANCE";
            break;
         case LOTSIZE_PER_FREEMARGIN://LOTSIZE PER FREE-MARGIN - Risk Option
            return "LOTSIZE PER FREE-MARGIN";
            break;
         case CUSTOM_LOT://CUSTOM LOTSIZE - Risk Option
            return "CUSTOM LOTSIZE";
            break;
         case PERCENTAGE_OF_MAXRISK://PERCENTAGE OF MAX-RISK - Risk Option
            return "PERCENTAGE OF MAX-RISK";
            break;
         default:
            return "";
            break;
        }
     }
   //-- Retrieve user's Risk Floor Option
   string            GetRiskFloor()
     {
      switch(RiskFloorOption)
        {
         case RiskFloorMin://MINIMUM LOTSIZE for Risk floor options
            return "MINIMUM LOTSIZE";
            break;
         case RiskFloorMax://MAX-RISK for Risk floor options
            return "MAX-RISK";
            break;
         case RiskFloorNone://NONE for Risk floor options
            return "NONE";
            break;
         default:
            return "";
            break;
        }
     }
   //-- Retrieve user's Risk Ceiling option
   string            GetRiskCeil()
     {
      switch(RiskCeilOption)
        {
         case  RiskCeilMax://MAX LOTSIZE for Risk ceiling options
            return "MAX LOTSIZE";
            break;
         case RiskCeilMax2://MAX LOTSIZE(x2) for Risk ceiling options
            return "MAX LOTSIZE(x2)";
            break;
         case RiskCeilMax3://MAX LOTSIZE(x3) for Risk ceiling options
            return "MAX LOTSIZE(x3)";
            break;
         case RiskCeilMax4://MAX LOTSIZE(x4) for Risk ceiling options
            return "MAX LOTSIZE(x4)";
            break;
         case RiskCeilMax5://MAX LOTSIZE(x5) for Risk ceiling options
            return "MAX LOTSIZE(x5)";
            break;
         default:
            return "";
            break;
        }
     }

   double            Volume();//Get risk in Volume
   //Apply fixes to lot-size where applicable
   void              NormalizeLotsize(double &Lotsize);
  };

//+------------------------------------------------------------------+
//|Constructor                                                       |
//+------------------------------------------------------------------+
//Initialize values
CRiskManagement::CRiskManagement(void):Lots(0.0),max_percent(100),
   ORDERTYPE(ORDER_TYPE_BUY),OpenPrice(Ask()),
   ClosePrice(NormalizePrice(Ask()+Ask()*0.01))

  {
  }

//+------------------------------------------------------------------+
//|Get risk in Volume                                                |
//+------------------------------------------------------------------+
double CRiskManagement::Volume()
  {
   switch(RiskProfileOption)
     {
      case  MINIMUM_LOT://MINIMUM LOTSIZE - Risk Option
         return LotsMin();
         break;
      case MAXIMUM_LOT://MAXIMUM LOTSIZE - Risk Option
         Lots = LotsMax();
         break;
      case PERCENTAGE_OF_BALANCE://PERCENTAGE OF BALANCE - Risk Option
         Lots = RiskProfile1(BALANCE);
         break;
      case PERCENTAGE_OF_FREEMARGIN://PERCENTAGE OF FREE-MARGIN - Risk Option
         Lots = RiskProfile1(MARGIN);
         break;
      case AMOUNT_PER_BALANCE://AMOUNT PER BALANCE - Risk Option
         Lots = RiskProfile2(BALANCE);
         break;
      case AMOUNT_PER_FREEMARGIN://AMOUNT PER FREE-MARGIN - Risk Option
         Lots = RiskProfile2(MARGIN);
         break;
      case LOTSIZE_PER_BALANCE://LOTSIZE PER BALANCE - Risk Option
         Lots =  RiskProfile3(BALANCE);
         break;
      case LOTSIZE_PER_FREEMARGIN://LOTSIZE PER FREE-MARGIN - Risk Option
         Lots = RiskProfile3(MARGIN);
         break;
      case CUSTOM_LOT://CUSTOM LOTSIZE - Risk Option
         Lots = Risk_Profile_4;
         break;
      case PERCENTAGE_OF_MAXRISK://PERCENTAGE OF MAX-RISK - Risk Option
         Lots = MaxRisk(Risk_Profile_5);
         break;
      default:
         Lots = 0.0;
         break;
     }
   ValidateLotsize(Lots);//Check/Adjust Lotsize Limits
   NormalizeLotsize(Lots);//Normalize Lotsize
   return Lots;
  }

//+------------------------------------------------------------------+
//|calculations for Risk options                                     |
//|(PERCENTAGE OF BALANCE and PERCENTAGE OF FREE-MARGIN)             |
//+------------------------------------------------------------------+
//-- calculations for Risk options (PERCENTAGE OF BALANCE and PERCENTAGE OF FREE-MARGIN)
double CRiskManagement::RiskProfile1(const RiskMedium R_Medium)
  {
   SetMedium(R_Medium);
   RiskAmount = Medium*(Risk_Profile_1/100);
   return GetRisk(RiskAmount);
  }

//+------------------------------------------------------------------+
//|calculations for Risk options                                     |
//|(AMOUNT PER BALANCE and AMOUNT PER FREE-MARGIN)                   |
//+------------------------------------------------------------------+
//-- calculations for Risk options (AMOUNT PER BALANCE and AMOUNT PER FREE-MARGIN)
double CRiskManagement::RiskProfile2(const RiskMedium R_Medium)
  {
   SetMedium(R_Medium);
   double risk = (Risk_Profile_2.RiskAmountBoF/Risk_Profile_2.RiskAmount);
   risk = (risk<1)?1:risk;

   if(Medium<=0)
      return 0.0;

   RiskAmount = Medium/risk;
   return GetRisk(RiskAmount);
  }

//+------------------------------------------------------------------+
//|calculations for Risk options                                     |
//|(LOTSIZE PER BALANCE and LOTSIZE PER FREE-MARGIN)                 |
//+------------------------------------------------------------------+
//-- calculations for Risk options (LOTSIZE PER BALANCE and LOTSIZE PER FREE-MARGIN)
double CRiskManagement::RiskProfile3(const RiskMedium R_Medium)
  {
   SetMedium(R_Medium);
   return (Medium>0)?((Medium/Risk_Profile_3.RiskLotBoF)*Risk_Profile_3.RiskLot):0.0;
  }

//+------------------------------------------------------------------+
//|calculations for Maximum allowable Risk                           |
//+------------------------------------------------------------------+
//-- calculations for Maximum allowable Risk
double CRiskManagement::MaxRisk(const double percent)
  {
   double margin=0.0,max_risk=0.0;
//--- checks
   if(percent<0.01 || percent>100)
     {
      Print(__FUNCTION__," invalid parameters");
      return(0.0);
     }
//--- calculate margin requirements for 1 lot
   if(!OrderCalcMargin(ORDERTYPE,Symbol(),1.0,OpenPrice,margin) || margin<0.0)
     {
      Print(__FUNCTION__," margin calculation failed");
      return(0.0);
     }
//--- calculate maximum volume
   max_risk=Account.FreeMargin()*(percent/100.0)/margin;
//--- return volume
   return(max_risk);
  }

//+------------------------------------------------------------------+
//|Apply fixes to lot-size where applicable                          |
//+------------------------------------------------------------------+
void CRiskManagement::NormalizeLotsize(double &Lotsize)
  {
   if(Lotsize<=0.0)
      return;

//-- Check if the is a Volume limit for the current Symbol
   if(LotsLimit()>0.0)
     {
      if((Lots+PositionsVolume()+OrdersVolume())>LotsLimit())
        {
         //-- calculation of available lotsize remaining
         double remaining_avail_lots = (LotsLimit()-(PositionsVolume()+OrdersVolume()));
         if(remaining_avail_lots>=LotsMin())
           {
            if(RiskFloorOption==RiskFloorMin)//Check if Risk floor option is MINIMUM LOTSIZE
              {
               Print("Warning: Volume Limit Reached, minimum Lotsize selected.");
               Lotsize = LotsMin();
              }
            else
               if(RiskFloorOption==RiskFloorMax)//Check if Risk floor option is MAX-RISK
                 {
                  Print("Warning: Volume Limit Reached, Lotsize Reduced.");
                  Lotsize = ((remaining_avail_lots*(RiskFloorPercentage/100))>LotsMin())?
                            (remaining_avail_lots*(RiskFloorPercentage/100)):LotsMin();
                 }
           }
         else
           {
            Print("Volume Limit Reached!");
            Lotsize=0.0;
            return;
           }
        }
     }

//Check if there is a valid Volume Step for the current Symbol
   if(LotsStep()>0.0)
      Lotsize=LotsStep()*MathFloor(Lotsize/LotsStep());
  }

//+------------------------------------------------------------------+
//|Application of Lot-size limits                                    |
//+------------------------------------------------------------------+
void CRiskManagement::ValidateLotsize(double &Lotsize)
  {
   switch(RiskFloorOption)
     {
      case RiskFloorMin://MINIMUM LOTSIZE for Risk floor options
         //-- Check if lot-size is not less than Minimum lot or more than maximum allowable risk
         if(Lotsize<LotsMin()||Lotsize>MaxRisk(max_percent))
           {
            Lotsize=LotsMin();
           }
         break;
      case RiskFloorMax://MAX-RISK for Risk floor options
         //-- Check if lot-size is more the maximum allowable risk
         if(Lotsize>MaxRisk(max_percent))
           {
            Lotsize=(MaxRisk(RiskFloorPercentage)>LotsMin())?MaxRisk(RiskFloorPercentage):LotsMin();
           }
         else
            if(Lotsize<LotsMin())//Check if lot-size is less than Minimum lot
              {
               Lotsize=LotsMin();
              }
         break;
      case RiskFloorNone://NONE for Risk floor options
         //Check if lot-size is less than Minimum lot
         if(Lotsize<LotsMin())
           {
            Lotsize=0.0;
           }
         break;
      default:
         Lotsize=0.0;
         break;
     }

   switch(RiskCeilOption)
     {
      case  RiskCeilMax://MAX LOTSIZE for Risk ceiling options
         //Check if lot-size is more than Maximum lot
         if(Lotsize>LotsMax())
            Lotsize=LotsMax();
         break;
      case RiskCeilMax2://MAX LOTSIZE(x2) for Risk ceiling options
         //Check if lot-size is more than Maximum lot times two
         if(Lotsize>(LotsMax()*2))
            Lotsize=(LotsMax()*2);
         break;
      case RiskCeilMax3://MAX LOTSIZE(x3) for Risk ceiling options
         //Check if lot-size is more than Maximum lot times three
         if(Lotsize>(LotsMax()*3))
            Lotsize=(LotsMax()*3);
         break;
      case RiskCeilMax4://MAX LOTSIZE(x4) for Risk ceiling options
         //Check if lot-size is more than Maximum lot times four
         if(Lotsize>(LotsMax()*4))
            Lotsize=(LotsMax()*4);
         break;
      case RiskCeilMax5://MAX LOTSIZE(x5) for Risk ceiling options
         //Check if lot-size is more than Maximum lot times five
         if(Lotsize>(LotsMax()*5))
            Lotsize=(LotsMax()*5);
         break;
      default:
         break;
     }
  }
//+------------------------------------------------------------------+
```

The variable RiskProfileOption of type enumeration RiskOptions, will store the user/trader's Risk profile option which will be an input for the expert.

```
//-- Enumeration declaration for Risk options
enum RiskOptions
  {
   MINIMUM_LOT,//MINIMUM LOTSIZE
   MAXIMUM_LOT,//MAXIMUM LOTSIZE
   PERCENTAGE_OF_BALANCE,//PERCENTAGE OF BALANCE
   PERCENTAGE_OF_FREEMARGIN,//PERCENTAGE OF FREE-MARGIN
   AMOUNT_PER_BALANCE,//AMOUNT PER BALANCE
   AMOUNT_PER_FREEMARGIN,//AMOUNT PER FREE-MARGIN
   LOTSIZE_PER_BALANCE,//LOTSIZE PER BALANCE
   LOTSIZE_PER_FREEMARGIN,//LOTSIZE PER FREE-MARGIN
   CUSTOM_LOT,//CUSTOM LOTSIZE
   PERCENTAGE_OF_MAXRISK//PERCENTAGE OF MAX-RISK
  } RiskProfileOption;//variable for Risk options
```

The variable RiskFloorOption of type enumeration RiskFloor, will store the user/trader's Minimum Risk Preference which will be an input for the expert.

```
//-- Enumeration declaration for Risk floor
enum RiskFloor
  {
   RiskFloorMin,//MINIMUM LOTSIZE
   RiskFloorMax,//MAX-RISK
   RiskFloorNone//NONE
  } RiskFloorOption;//variable for Risk floor
```

The variable RiskCeilOption of type enumeration RiskCeil, will store the user/trader's Maximum Risk Preference which will be an input for the expert.

```
//-- Enumeration declaration for Risk ceiling(Maximum allowable risk in terms of lot-size)
enum RiskCeil
  {
   RiskCeilMax,//MAX LOTSIZE
   RiskCeilMax2,//MAX LOTSIZE(x2)
   RiskCeilMax3,//MAX LOTSIZE(x3)
   RiskCeilMax4,//MAX LOTSIZE(x4)
   RiskCeilMax5,//MAX LOTSIZE(x5)
  } RiskCeilOption;//variable for Risk ceiling
```

The user/trader's (Account Balance or Free-Margin) will be stored in the double variable RiskAmountBoF and the double variable RiskAmount will store the Risk amount value. Risk\_Profile\_2 will be used to store Risk profiles' AMOUNT PER BALANCE and AMOUNT PER FREE-MARGIN properties.

```
//-- Structure declaration for Risk options (AMOUNT PER BALANCE and AMOUNT PER FREE-MARGIN)
struct RISK_AMOUNT
  {
   double            RiskAmountBoF;//store Balance or Free-Margin
   double            RiskAmount;//store risk amount
  } Risk_Profile_2;//variable for Risk options (AMOUNT PER BALANCE and AMOUNT PER FREE-MARGIN)
```

The variable Risk\_Profile\_3 of type structure RISK\_LOT, will store Risk profiles' LOTSIZE PER BALANCE and LOTSIZE PER FREE-MARGIN properties.

```
//-- Structure declaration for Risk options (LOTSIZE PER BALANCE and LOTSIZE PER FREE-MARGIN)
struct RISK_LOT
  {
   double            RiskLotBoF;//store Balance or Free-Margin
   double            RiskLot;//store lot-size
  } Risk_Profile_3;//variable for Risk options (LOTSIZE PER BALANCE and LOTSIZE PER FREE-MARGIN)
```

The variable RiskFloorPercentage will store the Percentage of Max-Risk for Riskfloor option RiskFloorMax.

```
double            RiskFloorPercentage;//variable for RiskFloorMax
```

The variable Risk\_Profile\_1 will store the Risk Percentage for Risk options  PERCENTAGE OF BALANCE or PERCENTAGE OF FREE-MARGIN.

```
double            Risk_Profile_1;//variable for Risk options (PERCENTAGE OF BALANCE and PERCENTAGE OF FREE-MARGIN)
```

The variable Risk\_Profile\_4 will store the Custom lot-size for Risk option CUSTOM LOTSIZE.

```
double            Risk_Profile_4;//variable for Risk option (CUSTOM LOTSIZE)
```

The variable Risk\_Profile\_5 will store the Percentage of max-risk for Risk option PERCENTAGE OF MAX-RISK.

```
double            Risk_Profile_5;//variable for Risk option (PERCENTAGE OF MAX-RISK)
```

In the function GetRiskOption, we will retrieve the user/trader's Risk option in a string datatype.

```
   //-- Retrieve user's Risk option
   string            GetRiskOption()
     {
      switch(RiskProfileOption)
        {
         case  MINIMUM_LOT://MINIMUM LOTSIZE - Risk Option
            return "MINIMUM LOTSIZE";
            break;
         case MAXIMUM_LOT://MAXIMUM LOTSIZE - Risk Option
            return "MAXIMUM LOTSIZE";
            break;
         case PERCENTAGE_OF_BALANCE://PERCENTAGE OF BALANCE - Risk Option
            return "PERCENTAGE OF BALANCE";
            break;
         case PERCENTAGE_OF_FREEMARGIN://PERCENTAGE OF FREE-MARGIN - Risk Option
            return "PERCENTAGE OF FREE-MARGIN";
            break;
         case AMOUNT_PER_BALANCE://AMOUNT PER BALANCE - Risk Option
            return "AMOUNT PER BALANCE";
            break;
         case AMOUNT_PER_FREEMARGIN://AMOUNT PER FREE-MARGIN - Risk Option
            return "AMOUNT PER FREE-MARGIN";
            break;
         case LOTSIZE_PER_BALANCE://LOTSIZE PER BALANCE - Risk Option
            return "LOTSIZE PER BALANCE";
            break;
         case LOTSIZE_PER_FREEMARGIN://LOTSIZE PER FREE-MARGIN - Risk Option
            return "LOTSIZE PER FREE-MARGIN";
            break;
         case CUSTOM_LOT://CUSTOM LOTSIZE - Risk Option
            return "CUSTOM LOTSIZE";
            break;
         case PERCENTAGE_OF_MAXRISK://PERCENTAGE OF MAX-RISK - Risk Option
            return "PERCENTAGE OF MAX-RISK";
            break;
         default:
            return "";
            break;
        }
     }
```

In the function GetRiskFloor, we will retrieve the user/trader's Risk floor option in a string datatype.

```
   //-- Retrieve user's Risk Floor Option
   string            GetRiskFloor()
     {
      switch(RiskFloorOption)
        {
         case RiskFloorMin://MINIMUM LOTSIZE for Risk floor options
            return "MINIMUM LOTSIZE";
            break;
         case RiskFloorMax://MAX-RISK for Risk floor options
            return "MAX-RISK";
            break;
         case RiskFloorNone://NONE for Risk floor options
            return "NONE";
            break;
         default:
            return "";
            break;
        }
     }
```

In the function GetRiskCeil, we will retrieve the user/trader's Risk ceiling option in a string datatype.

```
   //-- Retrieve user's Risk Ceiling option
   string            GetRiskCeil()
     {
      switch(RiskCeilOption)
        {
         case  RiskCeilMax://MAX LOTSIZE for Risk ceiling options
            return "MAX LOTSIZE";
            break;
         case RiskCeilMax2://MAX LOTSIZE(x2) for Risk ceiling options
            return "MAX LOTSIZE(x2)";
            break;
         case RiskCeilMax3://MAX LOTSIZE(x3) for Risk ceiling options
            return "MAX LOTSIZE(x3)";
            break;
         case RiskCeilMax4://MAX LOTSIZE(x4) for Risk ceiling options
            return "MAX LOTSIZE(x4)";
            break;
         case RiskCeilMax5://MAX LOTSIZE(x5) for Risk ceiling options
            return "MAX LOTSIZE(x5)";
            break;
         default:
            return "";
            break;
        }
     }
```

In Risk Management class's constructor we will initialize the variables previously declared with a default value. The default value for variable ORDERTYPE is ORDER\_TYPE\_BUY, so for the risk options that require an order type to calculate risk the order type will be set by this variable and will be used to simulate risk calculations for opening trades. The default open-price will be stored in the variable OpenPrice and will be the Symbol's Ask price for our ORDERTYPE. The default close-price will be a 1% price deviation from the Ask price stored into the variable ClosePrice.

```
//Initialize values
CRiskManagement::CRiskManagement(void):Lots(0.0),max_percent(100),
   ORDERTYPE(ORDER_TYPE_BUY),OpenPrice(Ask()),
   ClosePrice(NormalizePrice(Ask()+Ask()*0.01))

  {
  }
```

The function Volume will retrieve the lot-size for the user/trader's Risk profile option and adjust the lot-size from the selected risk option in accordance to the Risk Ceiling option and Risk Floor option selected by the user/trader.

Afterwards the lot-size will be normalize so an actual trade can be opened with the specific lot-size.

```
double CRiskManagement::Volume()
  {
   switch(RiskProfileOption)
     {
      case  MINIMUM_LOT://MINIMUM LOTSIZE - Risk Option
         return LotsMin();
         break;
      case MAXIMUM_LOT://MAXIMUM LOTSIZE - Risk Option
         Lots = LotsMax();
         break;
      case PERCENTAGE_OF_BALANCE://PERCENTAGE OF BALANCE - Risk Option
         Lots = RiskProfile1(BALANCE);
         break;
      case PERCENTAGE_OF_FREEMARGIN://PERCENTAGE OF FREE-MARGIN - Risk Option
         Lots = RiskProfile1(MARGIN);
         break;
      case AMOUNT_PER_BALANCE://AMOUNT PER BALANCE - Risk Option
         Lots = RiskProfile2(BALANCE);
         break;
      case AMOUNT_PER_FREEMARGIN://AMOUNT PER FREE-MARGIN - Risk Option
         Lots = RiskProfile2(MARGIN);
         break;
      case LOTSIZE_PER_BALANCE://LOTSIZE PER BALANCE - Risk Option
         Lots =  RiskProfile3(BALANCE);
         break;
      case LOTSIZE_PER_FREEMARGIN://LOTSIZE PER FREE-MARGIN - Risk Option
         Lots = RiskProfile3(MARGIN);
         break;
      case CUSTOM_LOT://CUSTOM LOTSIZE - Risk Option
         Lots = Risk_Profile_4;
         break;
      case PERCENTAGE_OF_MAXRISK://PERCENTAGE OF MAX-RISK - Risk Option
         Lots = MaxRisk(Risk_Profile_5);
         break;
      default:
         Lots = 0.0;
         break;
     }
   ValidateLotsize(Lots);//Check/Adjust Lotsize Limits
   NormalizeLotsize(Lots);//Normalize Lotsize
   return Lots;
  }
```

The function SetMedium will assign the double variable Medium the value of the user/trader's Account Balance or Account Free-Margin based of the enumeration variable R\_Medium.

```
//-- Set Medium variable value
   void                SetMedium(const RiskMedium R_Medium) {Medium = (R_Medium==BALANCE)?Account.Balance():Account.FreeMargin();}
```

The function GetMinimumRisk will assign the variable MinimumAmount with the minimum required risk for a specific trade with the Minimum lot-size.

```
//-- Get Minimum Risk for a Trade using Minimum Lot-size
   bool                GetMinimumRisk()
     {
      return OrderCalcProfit(ORDERTYPE,Symbol(),LotsMin(),OpenPrice,ClosePrice,MinimumAmount);
     }
```

The function GetRisk will get the required lot-size for a specified risk amount in the argument variable Amount. When the MinimumAmount(minimum risk amount for a specific trade) is established. Amount is divided by MinimumAmount to get a quotient to multiple with the minimum lot-size in order to get the required lot-size for Amount.

```
//-- Retrieve Risk amount based on Risk inputs
   double            GetRisk(double Amount)
     {
      if(!GetMinimumRisk()||Amount==0)
         return 0.0;
      return ((Amount/MinimumAmount)*LotsMin());
     }
```

In the function RiskProfile1 we calculate and return the lot-size for Risk options PERCENTAGE OF BALANCE or PERCENTAGE OF FREE-MARGIN.

```
//-- calculations for Risk options (PERCENTAGE OF BALANCE and PERCENTAGE OF FREE-MARGIN)
double CRiskManagement::RiskProfile1(const RiskMedium R_Medium)
  {
   SetMedium(R_Medium);
   RiskAmount = Medium*(Risk_Profile_1/100);
   return GetRisk(RiskAmount);
  }
```

In the function RiskProfile2 we calculate and return the lot-size for Risk options AMOUNT PER BALANCE or AMOUNT PER FREE-MARGIN.

```
//-- calculations for Risk options (AMOUNT PER BALANCE and AMOUNT PER FREE-MARGIN)
double CRiskManagement::RiskProfile2(const RiskMedium R_Medium)
  {
   SetMedium(R_Medium);
   double risk = (Risk_Profile_2.RiskAmountBoF/Risk_Profile_2.RiskAmount);
   risk = (risk<1)?1:risk;

   if(Medium<=0)
      return 0.0;

   RiskAmount = Medium/risk;
   return GetRisk(RiskAmount);
  }
```

In the function RiskProfile3 we calculate and return the lot-size for Risk options LOTSIZE PER BALANCE or LOTSIZE PER FREE-MARGIN.

```
//-- calculations for Risk options (LOTSIZE PER BALANCE and LOTSIZE PER FREE-MARGIN)
double CRiskManagement::RiskProfile3(const RiskMedium R_Medium)
  {
   SetMedium(R_Medium);
   return (Medium>0)?((Medium/Risk_Profile_3.RiskLotBoF)*Risk_Profile_3.RiskLot):0.0;
  }
```

In the function ValidateLotsize, adjustments are made to the variable Lotsize passed by reference.

In the first Switch Expression RiskFloorOption:

- In case RiskFloorMin: We check if the variable Lotsize is outside of it's limits then assign it the current Symbol's minimum lot-size. We check for the lower limit which is whether the variable value is less than the minimum lot-size. The upper limit is when the Lotsize variable is above the maximum risk possible.
- In case RiskFloorMax: We first check if the Lotsize variable is more than the maximum risk possible, if it is we then check if the maximum desired minimum risk is more than the minimum lot-size, if it is we assign Lotsize with the maximum desired minimum risk for that trade, else we assign the minimum lot-size . If the Lotsize was initially less than the maximum risk possible and less than the minimum lot-size, we then assign the minimum lot-size.

In the second Switch Expression RiskCeilOption, for each case we check if Lotsize is above any of the maximum risk based of the lot-size value and set the Lotsize value to the lot-size limit if reached.

```
void CRiskManagement::ValidateLotsize(double &Lotsize)
  {
   switch(RiskFloorOption)
     {
      case RiskFloorMin://MINIMUM LOTSIZE for Risk floor options
         //-- Check if lot-size is not less than Minimum lot or more than maximum allowable risk
         if(Lotsize<LotsMin()||Lotsize>MaxRisk(max_percent))
           {
            Lotsize=LotsMin();
           }
         break;
      case RiskFloorMax://MAX-RISK for Risk floor options
         //-- Check if lot-size is more the maximum allowable risk
         if(Lotsize>MaxRisk(max_percent))
           {
            Lotsize=(MaxRisk(RiskFloorPercentage)>LotsMin())?MaxRisk(RiskFloorPercentage):LotsMin();
           }
         else
            if(Lotsize<LotsMin())//Check if lot-size is less than Minimum lot
              {
               Lotsize=LotsMin();
              }
         break;
      case RiskFloorNone://NONE for Risk floor options
         //Check if lot-size is less than Minimum lot
         if(Lotsize<LotsMin())
           {
            Lotsize=0.0;
           }
         break;
      default:
         Lotsize=0.0;
         break;
     }

   switch(RiskCeilOption)
     {
      case  RiskCeilMax://MAX LOTSIZE for Risk ceiling options
         //Check if lot-size is more than Maximum lot
         if(Lotsize>LotsMax())
            Lotsize=LotsMax();
         break;
      case RiskCeilMax2://MAX LOTSIZE(x2) for Risk ceiling options
         //Check if lot-size is more than Maximum lot times two
         if(Lotsize>(LotsMax()*2))
            Lotsize=(LotsMax()*2);
         break;
      case RiskCeilMax3://MAX LOTSIZE(x3) for Risk ceiling options
         //Check if lot-size is more than Maximum lot times three
         if(Lotsize>(LotsMax()*3))
            Lotsize=(LotsMax()*3);
         break;
      case RiskCeilMax4://MAX LOTSIZE(x4) for Risk ceiling options
         //Check if lot-size is more than Maximum lot times four
         if(Lotsize>(LotsMax()*4))
            Lotsize=(LotsMax()*4);
         break;
      case RiskCeilMax5://MAX LOTSIZE(x5) for Risk ceiling options
         //Check if lot-size is more than Maximum lot times five
         if(Lotsize>(LotsMax()*5))
            Lotsize=(LotsMax()*5);
         break;
      default:
         break;
     }
  }
```

In the function NormalizeLotsize it's purpose is to check if the lot-size is within the Symbol's volume limit and that the lot-size is in accordance with the volume step.

If the lot-size violates the Symbol's volume limit, we then calculate the remaining available lot-sizes before the volume limit. Thereafter we check if the remaining lot-sizes is more than or equal to the minimum lot-size for the current Symbol.

- RiskFloorMin: We set the Lotsize variable to the Minimum lot-size.
- RiskFloorMax: If the RiskFloorPercentage of the remaining lot-size is more than the minimum lot-size, we set the Lotsize variable to this value. If the RiskFloorPercentage of the remaining lot-size is less than or equal to the minimum lot-size, we set the Lotsize variable to the minimum lot-size.

```
void CRiskManagement::NormalizeLotsize(double &Lotsize)
  {
   if(Lotsize<=0.0)
      return;

//-- Check if the is a Volume limit for the current Symbol
   if(LotsLimit()>0.0)
     {
      if((Lots+PositionsVolume()+OrdersVolume())>LotsLimit())
        {
         //-- calculation of available lotsize remaining
         double remaining_avail_lots = (LotsLimit()-(PositionsVolume()+OrdersVolume()));
         if(remaining_avail_lots>=LotsMin())
           {
            if(RiskFloorOption==RiskFloorMin)//Check if Risk floor option is MINIMUM LOTSIZE
              {
               Print("Warning: Volume Limit Reached, minimum Lotsize selected.");
               Lotsize = LotsMin();
              }
            else
               if(RiskFloorOption==RiskFloorMax)//Check if Risk floor option is MAX-RISK
                 {
                  Print("Warning: Volume Limit Reached, Lotsize Reduced.");
                  Lotsize = ((remaining_avail_lots*(RiskFloorPercentage/100))>LotsMin())?
                            (remaining_avail_lots*(RiskFloorPercentage/100)):LotsMin();
                 }
           }
         else
           {
            Print("Volume Limit Reached!");
            Lotsize=0.0;
            return;
           }
        }
     }

//Check if there is a valid Volume Step for the current Symbol
   if(LotsStep()>0.0)
      Lotsize=LotsStep()*MathFloor(Lotsize/LotsStep());
  }
```

## Common Graphics Class

Common graphics class will display general properties of the current Symbol and some of the risk options set by the trader.

CommonGraphics has multilevel Inheritance from classes:

- CObjectProperties
- CChartProperties
- CSymbolProperties

CommonGraphics has inclusion from CRiskManagement class.

CommonGraphics has hierarchical Inheritance from classes:

- CSymbolProperties
- CSymbolInfo

```
//+------------------------------------------------------------------+
//|                                                      NewsTrading |
//|                                  Copyright 2024, MetaQuotes Ltd. |
//|                            https://www.mql5.com/en/users/kaaiblo |
//+------------------------------------------------------------------+
#include "ObjectProperties.mqh"
#include "RiskManagement.mqh"
//+------------------------------------------------------------------+
//|CommonGraphics class                                              |
//+------------------------------------------------------------------+
class CCommonGraphics:CObjectProperties
  {
private:
   CRiskManagement   CRisk;//Risk management class object

public:
                     CCommonGraphics(void);//class constructor
                    ~CCommonGraphics(void) {}//class destructor
   void              GraphicsRefresh();//will create the chart objects
  };

//+------------------------------------------------------------------+
//|Constructor                                                       |
//+------------------------------------------------------------------+
CCommonGraphics::CCommonGraphics(void)
  {
   GraphicsRefresh();//calling GraphicsRefresh function
  }

//+------------------------------------------------------------------+
//|Specify Chart Objects                                             |
//+------------------------------------------------------------------+
void CCommonGraphics::GraphicsRefresh()
  {
//-- Will create the rectangle object
   Square(0,"Symbol Properties",2,20,330,183,ANCHOR_LEFT_UPPER);
//-- Will create the text object for the Symbol's name
   TextObj(0,"Symbol Name",Symbol(),5,23);
//-- Will create the text object for the contract size
   TextObj(0,"Symbol Contract Size","Contract Size: "+string(ContractSize()),5,40,CORNER_LEFT_UPPER,9);
//-- Will create the text object for the Symbol's Minimum lotsize
   TextObj(0,"Symbol MinLot","Minimum Lot: "+string(LotsMin()),5,60,CORNER_LEFT_UPPER,9);
//-- Will create the text object for the Symbol's Maximum lotsize
   TextObj(0,"Symbol MaxLot","Max Lot: "+string(LotsMax()),5,80,CORNER_LEFT_UPPER,9);
//-- Will create the text object for the Symbol's Volume Step
   TextObj(0,"Symbol Volume Step","Volume Step: "+string(LotsStep()),5,100,CORNER_LEFT_UPPER,9);
//-- Will create the text object for the Symbol's Volume Limit
   TextObj(0,"Symbol Volume Limit","Volume Limit: "+string(LotsLimit()),5,120,CORNER_LEFT_UPPER,9);
//-- Will create the text object for the trader's Risk Option
   TextObj(0,"Risk Option","Risk Option: "+CRisk.GetRiskOption(),5,140,CORNER_LEFT_UPPER,9);
//-- Will create the text object for the trader's Risk Floor
   TextObj(0,"Risk Floor","Risk Floor: "+CRisk.GetRiskFloor(),5,160,CORNER_LEFT_UPPER,9);
//-- Will create the text object for the trader's Risk Ceiling
   TextObj(0,"Risk Ceil","Risk Ceiling: "+CRisk.GetRiskCeil(),5,180,CORNER_LEFT_UPPER,9);
  }
//+------------------------------------------------------------------+
```

In the function GraphicsRefresh we set the properties for our graphical chart objects.

![Visible Chart Objects ](https://c.mql5.com/2/79/ChartObjects2.png)

![Chart Objects](https://c.mql5.com/2/79/ChartObjects.png)

```
void CCommonGraphics::GraphicsRefresh()
  {
//-- Will create the rectangle object
   Square(0,"Symbol Properties",2,20,330,183,ANCHOR_LEFT_UPPER);
//-- Will create the text object for the Symbol's name
   TextObj(0,"Symbol Name",Symbol(),5,23);
//-- Will create the text object for the contract size
   TextObj(0,"Symbol Contract Size","Contract Size: "+string(ContractSize()),5,40,CORNER_LEFT_UPPER,9);
//-- Will create the text object for the Symbol's Minimum lotsize
   TextObj(0,"Symbol MinLot","Minimum Lot: "+string(LotsMin()),5,60,CORNER_LEFT_UPPER,9);
//-- Will create the text object for the Symbol's Maximum lotsize
   TextObj(0,"Symbol MaxLot","Max Lot: "+string(LotsMax()),5,80,CORNER_LEFT_UPPER,9);
//-- Will create the text object for the Symbol's Volume Step
   TextObj(0,"Symbol Volume Step","Volume Step: "+string(LotsStep()),5,100,CORNER_LEFT_UPPER,9);
//-- Will create the text object for the Symbol's Volume Limit
   TextObj(0,"Symbol Volume Limit","Volume Limit: "+string(LotsLimit()),5,120,CORNER_LEFT_UPPER,9);
//-- Will create the text object for the trader's Risk Option
   TextObj(0,"Risk Option","Risk Option: "+CRisk.GetRiskOption(),5,140,CORNER_LEFT_UPPER,9);
//-- Will create the text object for the trader's Risk Floor
   TextObj(0,"Risk Floor","Risk Floor: "+CRisk.GetRiskFloor(),5,160,CORNER_LEFT_UPPER,9);
//-- Will create the text object for the trader's Risk Ceiling
   TextObj(0,"Risk Ceil","Risk Ceiling: "+CRisk.GetRiskCeil(),5,180,CORNER_LEFT_UPPER,9);
  }
```

## Expert

Once again we won't be opening any trades in this article.

```
//+------------------------------------------------------------------+
//|                                                      NewsTrading |
//|                                  Copyright 2024, MetaQuotes Ltd. |
//|                            https://www.mql5.com/en/users/kaaiblo |
//+------------------------------------------------------------------+
//--- width and height of the canvas (used for drawing)
#define IMG_WIDTH  200
#define IMG_HEIGHT 100
//--- enable to set color format
ENUM_COLOR_FORMAT clr_format=COLOR_FORMAT_XRGB_NOALPHA;
//--- drawing array (buffer)
uint ExtImg[IMG_WIDTH*IMG_HEIGHT];

#include "News.mqh"
CNews NewsObject;//Class CNews Object 'NewsObject'
#include "TimeManagement.mqh"
CTimeManagement CTM;//Class CTimeManagement Object 'CTM'
#include "WorkingWithFolders.mqh"
CFolders Folder();//Calling Class's Constructor
#include "ChartProperties.mqh"
CChartProperties CChart;//Class CChartProperties Object 'CChart'
#include "RiskManagement.mqh"
CRiskManagement CRisk;//Class CRiskManagement Object 'CRisk'
#include "CommonGraphics.mqh"
CCommonGraphics CGraphics();//Calling Class's Constructor

enum iSeparator
  {
   Delimiter//__________________________
  };

sinput group "+--------| RISK MANAGEMENT |--------+";
input RiskOptions RISK_Type=MINIMUM_LOT;//SELECT RISK OPTION
input RiskFloor RISK_Mini=RiskFloorMin;//RISK FLOOR
input double RISK_Mini_Percent=75;//MAX-RISK [100<-->0.01]%
input RiskCeil  RISK_Maxi=RiskCeilMax;//RISK CEILING
sinput iSeparator iRisk_1=Delimiter;//__________________________
sinput iSeparator iRisk_1L=Delimiter;//PERCENTAGE OF [BALANCE | FREE-MARGIN]
input double Risk_1_PERCENTAGE=3;//[100<-->0.01]%
sinput iSeparator iRisk_2=Delimiter;//__________________________
sinput iSeparator iRisk_2L=Delimiter;//AMOUNT PER [BALANCE | FREE-MARGIN]
input double Risk_2_VALUE=1000;//[BALANCE | FREE-MARGIN]
input double Risk_2_AMOUNT=10;//EACH AMOUNT
sinput iSeparator iRisk_3=Delimiter;//__________________________
sinput iSeparator iRisk_3L=Delimiter;//LOTSIZE PER [BALANCE | FREE-MARGIN]
input double Risk_3_VALUE=1000;//[BALANCE | FREE-MARGIN]
input double Risk_3_LOTSIZE=0.1;//EACH LOTS(VOLUME)
sinput iSeparator iRisk_4=Delimiter;//__________________________
sinput iSeparator iRisk_4L=Delimiter;//CUSTOM LOTSIZE
input double Risk_4_LOTSIZE=0.01;//LOTS(VOLUME)
sinput iSeparator iRisk_5=Delimiter;//__________________________
sinput iSeparator iRisk_5L=Delimiter;//PERCENTAGE OF MAX-RISK
input double Risk_5_PERCENTAGE=1;//[100<-->0.01]%

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---

//Initializing CRiskManagement variable for Risk options
   RiskProfileOption = RISK_Type;
//Initializing CRiskManagement variable for Risk floor
   RiskFloorOption = RISK_Mini;
//Initializing CRiskManagement variable for RiskFloorMax
   RiskFloorPercentage = (RISK_Mini_Percent>100)?100:
                         (RISK_Mini_Percent<0.01)?0.01:RISK_Mini_Percent;//Percentage cannot be more than 100% or less than 0.01%
//Initializing CRiskManagement variable for Risk ceiling
   RiskCeilOption = RISK_Maxi;
//Initializing CRiskManagement variable for Risk options (PERCENTAGE OF BALANCE and PERCENTAGE OF FREE-MARGIN)
   Risk_Profile_1 = (Risk_1_PERCENTAGE>100)?100:
                    (Risk_1_PERCENTAGE<0.01)?0.01:Risk_1_PERCENTAGE;//Percentage cannot be more than 100% or less than 0.01%
//Initializing CRiskManagement variables for Risk options (AMOUNT PER BALANCE and AMOUNT PER FREE-MARGIN)
   Risk_Profile_2.RiskAmountBoF = Risk_2_VALUE;
   Risk_Profile_2.RiskAmount = Risk_2_AMOUNT;
//Initializing CRiskManagement variables for Risk options (LOTSIZE PER BALANCE and LOTSIZE PER FREE-MARGIN)
   Risk_Profile_3.RiskLotBoF = Risk_3_VALUE;
   Risk_Profile_3.RiskLot = Risk_3_LOTSIZE;
//Initializing CRiskManagement variable for Risk option (CUSTOM LOTSIZE)
   Risk_Profile_4 = Risk_4_LOTSIZE;
//Initializing CRiskManagement variable for Risk option (PERCENTAGE OF MAX-RISK)
   Risk_Profile_5 = (Risk_5_PERCENTAGE>100)?100:
                    (Risk_5_PERCENTAGE<0.01)?0.01:Risk_5_PERCENTAGE;//Percentage cannot be more than 100% or less than 0.01%

   CChart.ChartRefresh();//Load chart configurations
   CGraphics.GraphicsRefresh();//-- Create/Re-create chart objects

   if(!MQLInfoInteger(MQL_TESTER))//Checks whether the program is in the strategy tester
     {
      //--- create OBJ_BITMAP_LABEL object for drawing
      ObjectCreate(0,"STATUS",OBJ_BITMAP_LABEL,0,0,0);
      ObjectSetInteger(0,"STATUS",OBJPROP_XDISTANCE,5);
      ObjectSetInteger(0,"STATUS",OBJPROP_YDISTANCE,22);
      //--- specify the name of the graphical resource
      ObjectSetString(0,"STATUS",OBJPROP_BMPFILE,"::PROGRESS");
      uint   w,h;          // variables for receiving text string sizes
      uint    x,y;          // variables for calculation of the current coordinates of text string anchor points

      /*
      In the Do while loop below, the code will check if the terminal is connected to the internet.
      If the the program is stopped the loop will break, if the program is not stopped and the terminal
      is connected to the internet the function CreateEconomicDatabase will be called from the News.mqh header file's
      object called NewsObject and the loop will break once called.
      */
      bool done=false;
      do
        {
         //--- clear the drawing buffer array
         ArrayFill(ExtImg,0,IMG_WIDTH*IMG_HEIGHT,0);

         if(!TerminalInfoInteger(TERMINAL_CONNECTED))
           {
            //-- integer dots used as a loading animation
            static int dots=0;
            //--- set the font
            TextSetFont("Arial",-150,FW_EXTRABOLD,0);
            TextGetSize("Waiting",w,h);//get text width and height values
            //--- calculate the coordinates of the 'Waiting' text
            x=10;//horizontal alignment
            y=IMG_HEIGHT/2-(h/2);//alignment for the text to be centered vertically
            //--- output the 'Waiting' text to ExtImg[] buffer
            TextOut("Waiting",x,y,TA_LEFT|TA_TOP,ExtImg,IMG_WIDTH,IMG_HEIGHT,ColorToARGB(CChart.SymbolBackground()),clr_format);
            //--- calculate the coordinates for the dots after the 'Waiting' text
            x=w+13;//horizontal alignment
            y=IMG_HEIGHT/2-(h/2);//alignment for the text to be centered vertically
            TextSetFont("Arial",-160,FW_EXTRABOLD,0);
            //--- output of dots to ExtImg[] buffer
            TextOut(StringSubstr("...",0,dots),x,y,TA_LEFT|TA_TOP,ExtImg,IMG_WIDTH,IMG_HEIGHT,ColorToARGB(CChart.SymbolBackground()),clr_format);
            //--- update the graphical resource
            ResourceCreate("::PROGRESS",ExtImg,IMG_WIDTH,IMG_HEIGHT,0,0,IMG_WIDTH,clr_format);
            //--- force chart update
            ChartRedraw();
            dots=(dots==3)?0:dots+1;
            //-- Notify user that program is waiting for connection
            Print("Waiting for connection...");
            Sleep(500);
            continue;
           }
         else
           {
            //--- set the font
            TextSetFont("Arial",-120,FW_EXTRABOLD,0);
            TextGetSize("Getting Ready",w,h);//get text width and height values
            x=20;//horizontal alignment
            y=IMG_HEIGHT/2-(h/2);//alignment for the text to be centered vertically
            //--- output the text 'Getting Ready...' to ExtImg[] buffer
            TextOut("Getting Ready...",x,y,TA_LEFT|TA_TOP,ExtImg,IMG_WIDTH,IMG_HEIGHT,ColorToARGB(CChart.SymbolBackground()),clr_format);
            //--- update the graphical resource
            ResourceCreate("::PROGRESS",ExtImg,IMG_WIDTH,IMG_HEIGHT,0,0,IMG_WIDTH,clr_format);
            //--- force chart update
            ChartRedraw();
            //-- Notify user that connection is successful
            Print("Connection Successful!");
            NewsObject.CreateEconomicDatabase();//calling the database create function
            done=true;
           }
        }
      while(!done&&!IsStopped());
      //-- Delete chart object
      ObjectDelete(0,"STATUS");
      //-- force chart to update
      ChartRedraw();
     }
   else
     {
      //Checks whether the database file exists
      if(!FileIsExist(NEWS_DATABASE_FILE,FILE_COMMON))
        {
         Print("Necessary Files Do not Exist!");
         Print("Run Program outside of the Strategy Tester");
         Print("Necessary Files Should be Created First");
         return(INIT_FAILED);
        }
      else
        {
         //Checks whether the lastest database date includes the time and date being tested
         datetime latestdate = CTM.TimePlusOffset(NewsObject.GetLatestNewsDate(),CTM.DaysS());//Day after the lastest recorded time in the database
         if(latestdate<TimeCurrent())
           {
            Print("Necessary Files outdated!");
            Print("To Update Files: Run Program outside of the Strategy Tester");
           }
         Print("Database Dates End at: ",latestdate);
         PrintFormat("Dates after %s will not be available for backtest",TimeToString(latestdate));
        }
     }
//-- the volume calculations and the risk type set by the trader
   Print("Lots: ",CRisk.Volume()," || Risk type: ",CRisk.GetRiskOption());
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

  }
//+------------------------------------------------------------------+
```

![Project Files](https://c.mql5.com/2/79/Project_files__1.png)

![Expert](https://c.mql5.com/2/79/Add_Expert_to_chart.png)

Once everything is compiled we will go through some steps that will occur once the Expert is placed on the chart.

![US30 chart](https://c.mql5.com/2/79/wUS30H1_BeforeExpert.png)

Once you decide which Symbol chart window to open, your chart could look similar to the chart above before attaching the Expert.

Now we will attach the Expert.

![Attach the Expert](https://c.mql5.com/2/79/Attach_Expert.png)

We can first configure the Risk management input variables, I'll leave the default values to begin with.

![Expert Inputs](https://c.mql5.com/2/79/Expert_inputs.png)

If it is the first time running NewsTrading 2.00 and you didn't previously run NewsTrading 1.00 on your broker and the Calendar database does not exist in the common folder.

Your chart will appear in this manner, chart colors may differ from broker to broker.

![Expert chart #1](https://c.mql5.com/2/79/Expert_Firsttime_NoCalendarDB.png)

As seen in the message below the Lots will be displayed according to the Risk options selected in the settings.

![Expert Chart #1 Message](https://c.mql5.com/2/79/Expert_Firsttime_NoCalendarDBbMessage.png)

If the terminal cannot find a connection.

2024 05 28 17 03 16 - YouTube

[Photo image of MQL5.community](https://www.youtube.com/channel/UC8bwZMk2yh5WDToFos49OjQ?embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F14912)

MQL5.community

1.91K subscribers

[2024 05 28 17 03 16](https://www.youtube.com/watch?v=WzfJSfAISxA)

MQL5.community

Search

Watch later

Share

Copy link

Info

Shopping

Tap to unmute

If playback doesn't begin shortly, try restarting your device.

More videos

## More videos

You're signed out

Videos you watch may be added to the TV's watch history and influence TV recommendations. To avoid this, cancel and sign in to YouTube on your computer.

CancelConfirm

Share

Include playlist

An error occurred while retrieving sharing information. Please try again later.

[Watch on](https://www.youtube.com/watch?v=WzfJSfAISxA&embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F14912)

0:00

0:00 / 0:06

•Live

•

When running the NewsTrading 2.00 Expert for the first time and the calendar database was created from NewsTrading 1.00 Expert.

NewsTrading 2.00 will delete all tables from NewsTrading 1.00

![Expert Attached and Calendar DD Exists from Previous version](https://c.mql5.com/2/79/Expert_Firsttime_CalendarExists.png)

## Conclusion

In this article we explored how Inheritance works, with an example for illustrative purposes. We created created a new Daylight Savings class to act as a parent for the different DST schedule classes. We created a Symbol properties class to retrieve Symbol properties from all classes that Inherit from it. We also created a Chart properties class to customize the chart. We went through different SQLite methodologies and a simple way to improve the overall efficiency of the database. We created an Object properties class to create chart objects and delete them, we then created a Risk management class to cater for different risk profiles and traders. Finally we created the last class called Common Graphics to display the Symbol properties onto the chart along with the trader's risk options. In the next article, 'News Trading Made Easy (Part 3): Performing Trades' we will finally begin opening trades and I cannot wait to finish it in a reasonable timeline!

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/14912.zip "Download all attachments in the single ZIP archive")

[NewsTrading\_Part2\_ProjectFiles.zip](https://www.mql5.com/en/articles/download/14912/newstrading_part2_projectfiles.zip "Download NewsTrading_Part2_ProjectFiles.zip")(440.93 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [News Trading Made Easy (Part 6): Performing Trades (III)](https://www.mql5.com/en/articles/16170)
- [News Trading Made Easy (Part 5): Performing Trades (II)](https://www.mql5.com/en/articles/16169)
- [News Trading Made Easy (Part 4): Performance Enhancement](https://www.mql5.com/en/articles/15878)
- [News Trading Made Easy (Part 3): Performing Trades](https://www.mql5.com/en/articles/15359)
- [News Trading Made Easy (Part 1): Creating a Database](https://www.mql5.com/en/articles/14324)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/468090)**
(6)


![Kabelo Frans Mampa](https://c.mql5.com/avatar/2023/1/63bd510f-63d8.png)

**[Kabelo Frans Mampa](https://www.mql5.com/en/users/kaaiblo)**
\|
11 Jul 2024 at 19:13

**Jermaine Wedderburn [#](https://www.mql5.com/en/forum/468090#comment_53947373):**

Was trying to test it but i don't know where to put al the files, so the newstrading is not compiling

Step 1: Move the NewsTrading folder into the Experts Folder

![Step 1](https://c.mql5.com/3/439/step1.png)

Step 2:Open The NewsTrading Project File.

![Step 2](https://c.mql5.com/3/439/step2.png)

Step 3: Click on the NewsTrading mq5 file and open it.

![Step 3](https://c.mql5.com/3/439/step3.png)

Step 4: Compile the Application.

![Step 4](https://c.mql5.com/3/439/final.png)

![Christian Edward Bannard](https://c.mql5.com/avatar/2021/4/6078E99A-355C.jpg)

**[Christian Edward Bannard](https://www.mql5.com/en/users/traderd)**
\|
25 Oct 2024 at 00:50

When switching Stop Loss to 0, this is what happens:

[![](https://c.mql5.com/3/447/1756679750171__1.png)](https://c.mql5.com/3/447/1756679750171.png "https://c.mql5.com/3/447/1756679750171.png")

[![](https://c.mql5.com/3/447/811562456350__1.png)](https://c.mql5.com/3/447/811562456350.png "https://c.mql5.com/3/447/811562456350.png")

Great programming, though needs a bit of a tidy up. Really interesting article and I can see how much effort was put into it.

![Kabelo Frans Mampa](https://c.mql5.com/avatar/2023/1/63bd510f-63d8.png)

**[Kabelo Frans Mampa](https://www.mql5.com/en/users/kaaiblo)**
\|
25 Oct 2024 at 06:15

**Christian Edward Bannard [#](https://www.mql5.com/en/forum/468090#comment_54929865):**

When switching Stop Loss to 0, this is what happens:

Great programming, though needs a bit of a tidy up. Really interesting article and I can see how much effort was put into it.

Hi Christian, thanks for the kind words. This issue has been noticed and resolved in my later articles which are still in the process of being published. In terms of the commission, are you suggesting that the expert adjust for commissions when calculating risk?


![Christian Edward Bannard](https://c.mql5.com/avatar/2021/4/6078E99A-355C.jpg)

**[Christian Edward Bannard](https://www.mql5.com/en/users/traderd)**
\|
28 Oct 2024 at 06:01

**Kabelo Frans Mampa [#](https://www.mql5.com/en/forum/468090#comment_54930850):**

Hi Christian, thanks for the kind words. This issue has been noticed and resolved in my later articles which are still in the process of being published. In terms of the commission, are you suggesting that the expert adjust for commissions when calculating risk?

I believe factoring in commissions is an important aspect to profitability and when calculating risk, otherwise traders using automated systems may be closing out trades thinking they're in profit and realistically profit minus expenses = net profit, which reflects the real world.

If a trade is in loss or open for an extended time, the price simply returning back to your entry point is likely not going to cover costs.

Sure, in a demo account, all good for testing purposes, though most brokers charge some kind of commission, so breakeven is never going to be returning to the entry point, it will always be a few points above/below your original entry point depending on whether the broker charges both for entry and exit or whether they charge a percentage. Swaps should also be factored into any breakeven point also.

![Kabelo Frans Mampa](https://c.mql5.com/avatar/2023/1/63bd510f-63d8.png)

**[Kabelo Frans Mampa](https://www.mql5.com/en/users/kaaiblo)**
\|
30 Oct 2024 at 04:54

**Christian Edward Bannard [#](https://www.mql5.com/ru/forum/475326#comment_54953089) :**

I believe that accounting for fees is an important aspect of profitability and risk calculation, otherwise traders using automated systems may close trades thinking they are in profit, when in fact profit minus expenses = net profit, which reflects the real world.

If a trade is in the red or has been open for a long time, the price simply returning to the entry point will most likely not cover the costs.

Of course, a demo account is fine for testing purposes, but most brokers charge some commission, so breakeven will never be back to your entry point, it will always be a few pips above/below your original entry point depending on whether the broker charges entry and exit fees or charges a percentage. Swaps must also be factored into any breakeven point.

Thanks for the feedback, it is appreciated!

![Neural networks made easy (Part 72): Trajectory prediction in noisy environments](https://c.mql5.com/2/64/Neural_networks_made_easy_6Part_72m__Predicting_trajectories_in_the_presence_of_noise___LOGO-FNYbN4B.png)[Neural networks made easy (Part 72): Trajectory prediction in noisy environments](https://www.mql5.com/en/articles/14044)

The quality of future state predictions plays an important role in the Goal-Conditioned Predictive Coding method, which we discussed in the previous article. In this article I want to introduce you to an algorithm that can significantly improve the prediction quality in stochastic environments, such as financial markets.

![Reimagining Classic Strategies: Crude Oil](https://c.mql5.com/2/79/Reimagining_Classic_Strategies____Crude_Oil____LOGO___5.png)[Reimagining Classic Strategies: Crude Oil](https://www.mql5.com/en/articles/14855)

In this article, we revisit a classic crude oil trading strategy with the aim of enhancing it by leveraging supervised machine learning algorithms. We will construct a least-squares model to predict future Brent crude oil prices based on the spread between Brent and WTI crude oil prices. Our goal is to identify a leading indicator of future changes in Brent prices.

![Neural networks made easy (Part 73): AutoBots for predicting price movements](https://c.mql5.com/2/64/Neural_networks_are_easy_jPart_73u__AutoBots_for_predicting_price_movement_LOGO.png)[Neural networks made easy (Part 73): AutoBots for predicting price movements](https://www.mql5.com/en/articles/14095)

We continue to discuss algorithms for training trajectory prediction models. In this article, we will get acquainted with a method called "AutoBots".

![Developing a multi-currency Expert Advisor (Part 2): Transition to virtual positions of trading strategies](https://c.mql5.com/2/69/Developing_a_multi-currency_advisor_0Part_1g___LOGO__3.png)[Developing a multi-currency Expert Advisor (Part 2): Transition to virtual positions of trading strategies](https://www.mql5.com/en/articles/14107)

Let's continue developing a multi-currency EA with several strategies working in parallel. Let's try to move all the work associated with opening market positions from the strategy level to the level of the EA managing the strategies. The strategies themselves will trade only virtually, without opening market positions.

[We've created a channel for MQL5 developersFollow MQL5.community on social media and be the first to receive important updatesLearn more![](https://www.mql5.com/ff/sh/a83xrgctr82w45z9z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=pwgdbvtemvkwsfltqonysypfvtrtufji&s=e99a66a1660cd810b1edbac65597df695e2c2220d1e937834f402f9aeabd4289&uid=&ref=https://www.mql5.com/en/articles/14912&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5083461534560033834)

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