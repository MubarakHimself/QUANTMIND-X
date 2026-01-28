---
title: MQL for "Dummies": How to Design and Construct Object Classes
url: https://www.mql5.com/en/articles/53
categories: Expert Advisors
relevance_score: 0
scraped_at: 2026-01-24T14:13:34.726259
---

[![](https://www.mql5.com/ff/si/fx5m8s6u6uxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=ysducdhemkrdsdtzzbfkclolrllnhezk&s=33f180a31db6c3b846d77732b0bc78169421a47b8cf9f076ca717f4e4846d1c7&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=ttscsqcxqnjppvnxackyqrltgwezaevm&ssn=1769253213245158269&ssn_dr=0&ssn_sr=0&fv_date=1769253213&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F53&back_ref=https%3A%2F%2Fwww.google.com%2F&title=MQL%20for%20%22Dummies%22%3A%20How%20to%20Design%20and%20Construct%20Object%20Classes%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176925321317848112&fz_uniq=5083419512600009564&sv=2552)

MetaTrader 5 / Expert Advisors


### Introduction to Object-Oriented Programming (OOP)

Question of "dummies": Having only the vaguest understanding of procedural programming, is it possible to master OOP and use it in writing automated trading strategies? Or is this task beyond a common user?

By and large, it is possible to use the object-oriented programming language to write a MQL5 Expert Advisor or indicator, without the use of the [object-oritented programming](https://www.mql5.com/en/docs/basis/oop) principles. The use of new technologies in your developments is not mandatory. Choose the way that you believe to be the simplest. In addition, the application of the OOP more so can not guarantee the profitability of trading robots, which you create.

However, the transition to a new (object oriented) approach, opens the grounds for applying more complex adaptive mathematical models of trading strategies to their Expert Advisors, which will react to external changes and synchronize with the market.

So let's take a look at the technologies that OOP is based on:

1. Events

2. Object classes


Events are the main base of the OOP. The entire logic of the program is built on processing the constantly incoming events. The appropriate reactions to them are defined and described in the object classes. In other words, a class object works by intercepting and processing the flow of events.

The second basis is the class of objects, which in its turn rests on the "three pillars":

1. [Encapsulation](https://www.mql5.com/en/docs/basis/oop/incapsulation) \- Protection of class based on a "black box" principle : the object reacts to events, but its factual implementation remains unknown.

2. [Inheritance](https://www.mql5.com/en/docs/basis/oop/inheritance) \- the possibility to create a new class from an existing one, while preserving all the properties and methods of the "ancestor" class.

3. [Polymorphism](https://www.mql5.com/en/docs/basis/oop/polymorphism) \- the ability to change the implementation of an inherited method in a "descendant" class.

Basic concepts are best demonstrated in the Expert Advisor code.

**[Events:](https://www.mql5.com/en/docs/basis/function/events)**

```
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()                        // OnInit event processing
  {
   return(0);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)     // OnDeInit event processing
  {
  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()                       // OnTick event processing
  {
  }
//+------------------------------------------------------------------+
//| Expert Timer function                                            |
//+------------------------------------------------------------------+
void OnTimer()                      // OnTimer event processing
  {
  }
//+------------------------------------------------------------------+
//| Expert Chart event function                                      |
//+------------------------------------------------------------------+
void OnChartEvent(const int id,     // OnChartEvent event processing
                  const long &lparam,
                  const double &dparam,
                  const string &sparam)
  {
  }
```

**[Object Class:](https://www.mql5.com/en/docs/basis/types/classes#class)**

```
class CNew:public CObject
  {
private:
   int               X,Y;
   void              EditXY();
protected:
   bool              on_event;      //events processing flag
public:
   // Class constructor
   void              CNew();
   // OnChart event processing method
   virtual void      OnEvent(const int id,
                             const long &lparam,
                             const double &dparam,
                             const string &sparam);
  };
```

**[Encapsulation:](https://www.mql5.com/en/docs/basis/oop/incapsulation)**

```
private:
   int               X,Y;
   void              EditXY();
```

**[Inheritance:](https://www.mql5.com/en/docs/basis/oop/inheritance)**

```
class CNew: public CObject
```

**[Polymorphism:](https://www.mql5.com/en/docs/basis/oop/polymorphism)**

```
   // OnChart event processing method
   virtual void      OnEvent(const int id,
                             const long &lparam,
                             const double &dparam,
                             const string &sparam);
```

The [virtual](https://www.mql5.com/en/docs/basis/oop/virtual) modifier of this method means that OnEvent handler can be overridden, but the name of the method in this case remains the same as that of the ancestor class.

### 2\. Designing Classes

One of the most significant advantages of the OOP is its extensible - which means that the existing system is able to work with new components, without making any changes to it. New components can be added at this stage.

Consider the design process by creating a program of visual design of MasterWindows classes for MQL5.

**2.1.** **I Stage: Project Draft**

The design process begins with a sketch, drawn in pencil on a sheet of paper. This is one of the most challenging and exciting moments in programming. We must consider not only the dialogue between the program and the user (the interface), but also the organization of data processing. This process may take more than one day. It is best to begin with the interface, because it can become (in some cases, as in our example) defining when structuring an algorithm.

For the organization of the dialogue of the created program, we will use the form, similar to the Windows application window (see sketch in Figure 1). It contains lines, and these in turn consist of cells, and cells of the graphical objects. And so, as early as on the stage of conceptual design, we begin to see the structure of the program and the classification of objects.

![Figure 1. Form of the classes constructor (sketch)](https://c.mql5.com/2/1/figure1.png)

Figure 1\. Form of the
classes constructor (sketch)

With a sufficiently large number of rows and cells (fields) in the form, they are constructed out of only two types of graphic objects: [OBJ\_EDIT](https://www.mql5.com/en/docs/constants/objectconstants/enum_object) and [OBJ\_BUTTON](https://www.mql5.com/en/docs/constants/objectconstants/enum_object) . Thus, once we determine the visual appearance, the structure, and the basic objects created by the program, we can assume that the draft of the design is ready and it's time to move on to the next stage.

**2.2** **Stage II: Designing the Base Class**

There are three such classes so far, and more can be added later (if necessary):

- class cell CCell;

- class row CRow, consists of cells of class CCell;
- class window CWin, consists of lines of class CRow.

We can now proceed directly to programming classes, but ... we have yet to solve a very important task - the exchange of data between objects of classes. For such purposes, the language of MQL5 contains, aside from the usual variables, a new type - [structure](https://www.mql5.com/en/docs/basis/types/classes) . Of course, at this stage of design, we can not see all of the connections and it is difficult to calculate them. Therefore, we will gradually fill the description of classes and structures as the project progresses. Moreover, the principles of the OOP not only do not hinder this, but in fact the opposite - encourage the technology or programming.

**WinCell Structure:**

```
struct WinCell
  {
   color             TextColor;     // text color
   color             BGColor;       // background color
   color             BGEditColor;   // background color while editing
   ENUM_BASE_CORNER  Corner;         // anchor corner
   int               H;            // cell height
   int               Corn;         // displacement direction (1;-1)
  };
```

Structures that do not contain strings and objects of dynamic arrays are called simple structure. The variables of such structures can be freely copied into each other, even if they are different structures. The established structure is exactly of this type. We will evaluate its effectiveness later.

**Base class CCell:**

```
//+------------------------------------------------------------------+
//| CCell base class                                                 |
//+------------------------------------------------------------------+
class CCell
  {
private:
protected:
   bool              on_event;      // event processing flag
   ENUM_OBJECT       type;           // cell type
public:
   WinCell           Property;     // cell property
   string            name;          // cell name
   //+---------------------------------------------------------------+
   // Class constructor
   void              CCell();
   virtual     // Draw method
   void              Draw(string m_name,
                          int m_xdelta,
                          int m_ydelta,
                          int m_bsize);
   virtual     // Event processing method
   void              OnEvent(const int id,
                             const long &lparam,
                             const double &dparam,
                             const string &sparam);
  };
```

**Base class CRow:**

```
//+------------------------------------------------------------------+
//| CRow base class                                                  |
//+------------------------------------------------------------------+
class CRow
  {
protected:
   bool              on_event;      // event processing flag
public:
   string            name;          // row name
   WinCell           Property;     // row property
   //+---------------------------------------------------------------+
   // Class constructor
   void              CRow();
   virtual     // Draw method
   void              Draw(string m_name,
                          int m_xdelta,
                          int m_ydelta,
                          int m_bsize);
   virtual     // Event processing method
   void              OnEvent(const int id,
                             const long &lparam,
                             const double &dparam,
                             const string &sparam);
  };
```

**Base class CWin:**

```
//+------------------------------------------------------------------+
//| Base CWin class (WINDOW)                                         |
//+------------------------------------------------------------------+
class CWin
  {
private:
   void              SetXY(int m_corner); //Coordinates
protected:
   bool              on_event;   // event processing flag
public:
   string            name;       // window name
   int               w_corner;   // window corner
   int               w_xdelta;   // vertical delta
   int               w_ydelta;   // horizontal detla
   int               w_xpos;     // X coordinate
   int               w_ypos;     // Y coordinate
   int               w_bsize;    // Window width
   int               w_hsize;    // Window height
   int               w_h_corner; // hide mode corner
   WinCell           Property;   // Property
   //---
   CRowType1         STR1;       // CRowType1
   CRowType2         STR2;       // CRowType2
   CRowType3         STR3;       // CRowType3
   CRowType4         STR4;       // CRowType4
   CRowType5         STR5;       // CRowType5
   CRowType6         STR6;       // CRowType6
   //+---------------------------------------------------------------+
   // Class constructor
   void              CWin();
   // Set window properties
   void              SetWin(string m_name,
                            int m_xdelta,
                            int m_ydelta,
                            int m_bsize,
                            int m_corner);
   virtual     // Draw window method
   void              Draw(int &MMint[][3],
                          string &MMstr[][3],
                          int count);
   virtual     // OnEventTick handler
   void              OnEventTick();
   virtual     // OnChart event handler method
   void              OnEvent(const int id,
                             const long &lparam,
                             const double &dparam,
                             const string &sparam);
  };
```

Explanations and recommendations:

- All of the base classes (in this project) contain methods of processing events. They are required for intercepting and transmitting events further along the chain. Without a mechanism for receiving and sending events, the program (or module) loses its interactivity.
- When developing a base class, try to build it with a minimal number of methods. Then, implement various extensions of this class in the "descending" classes, which will boost the functionality of created objects.
- Do not use a direct appeal to the internal data of another class!


**2.3.** **Stage III: Working Project**

At this point we begin a step by step creation of the program. Begining with the supporting framework, we will increase its functional components and fill it with contents. During this, we will monitor the correctness of the work, apply debugging with an optimized code and track the appearing errors.

Let's stop here and consider the technology of the framework creation, which will work for almost any program. The main requirement for it - it should be immediately operational (compile without errors and run on execution). The language designers have taken care of this and advise to use the Expert Advisor template, which is generated by the MQL5 Wizard, as a framework.

As an example, let's consider our own version of this template:

**1) Program = Expert Advisor**

```
//+------------------------------------------------------------------+
//|                                                MasterWindows.mq5 |
//|                                                 Copyright DC2008 |
//|                                              http://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "DC2008"
#property link      "http://www.mql5.com"
#property version   "1.00"
//--- include files with classes
#include <ClassMasterWindows.mqh>
//--- Main module declaration
CMasterWindows    MasterWin;
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//--- Launch of the main module
   MasterWin.Run();
   return(0);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//--- Deinitialization of the main module
   MasterWin.Deinit();
  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//--- call OnTick event handler of main module
   MasterWin.OnEventTick();
  }
//+------------------------------------------------------------------+
//| Expert Event function                                            |
//+------------------------------------------------------------------+
void OnChartEvent(const int id,
                  const long &lparam,
                  const double &dparam,
                  const string &sparam)
  {
//--- call OnChartEvent handler of main module
   MasterWin.OnEvent(id,lparam,dparam,sparam);
  }
```

This is the completed code of the Expert Advisor. No additional changes need to be added throughout the project!

**2) The main module = class**

All of the main and auxiliary modules of the project will begin their development from here. This approach eases the programming of complex multi-modular projects and facilitates the search for possible errors. But finding them is very difficult. Sometimes it is easier and faster to write a new project rather than seek out the elusive "bugs".

```
//+------------------------------------------------------------------+
//|                                           ClassMasterWindows.mqh |
//|                                                 Copyright DC2008 |
//|                                              http://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "DC2008"
#property link      "http://www.mql5.com"
//+------------------------------------------------------------------+
//| Main module: CMasterWindows class                                |
//+------------------------------------------------------------------+
class CMasterWindows
  {
protected:
   bool              on_event;   // event processing flag
public:
   // Class constructor
   void              CMasterWindows();
   // Method of launching the main module (core algorithm)
   void              Run();
   // Deinitialization method
   void              Deinit();
   // OnTick event processing method
   void              OnEventTick();
   // OnChartEvent event processing method
   void              OnEvent(const int id,
                             const long &lparam,
                             const double &dparam,
                             const string &sparam);
  };
```

Below is a rough initial description of the main methods of the class.

```
//+------------------------------------------------------------------+
//| CMasterWindows class constructor                                 |
//+------------------------------------------------------------------+
void CMasterWindows::CMasterWindows()
   {
//--- class members initialization
   on_event=false;   // disable events processing
   }
//+------------------------------------------------------------------+
//| Метод запуска главного модуля (основной алгоритм)                |
//+------------------------------------------------------------------+
void CMasterWindows::Run()
   {
//--- Main functional of the class: runs additional modules
   ObjectsDeleteAll(0,0,-1);
   Comment("MasterWindows for MQL5     © DC2008");
//---
   on_event=true;   // enable events processing
   }
//+------------------------------------------------------------------+
//| Deinitialization method                                          |
//+------------------------------------------------------------------+
void CMasterWindows::Deinit()
   {
//---
   ObjectsDeleteAll(0,0,-1);
   Comment("");
   }
//+------------------------------------------------------------------+
//| OnTick() event processing method                                 |
//+------------------------------------------------------------------+
void CMasterWindows::OnEventTick()
   {
   if(on_event) // event processing is enabled
     {
     //---
     }
   }
//+------------------------------------------------------------------+
//| OnChartEvent() event processing method                           |
//+------------------------------------------------------------------+
void CMasterWindows::OnEvent(const int id,
                             const long &lparam,
                             const double &dparam,
                             const string &sparam)
  {
   if(on_event) // event processing is enabled
     {
     //---
     }
  }
```

**3) The library of basic and derived classes**

The library can contain any number of derived classes and it is best to group them in separate files, which are included along with the base class (if any). This way, it will be easier to make the necessary changes and additions, as well as - search for errors.

And so, we now have the framework of the program. Let's test it and see if it works correctly: compile and execute. If the test is successful, then we can begin filling the project with additional modules.

Let's start with the connection of derived classes, and begin with the cells:

| Name class | Image |
| --- | --- |
| Class CCellText | ![](https://c.mql5.com/2/1/Textfield.png) |
| Class CCellEdit | ![](https://c.mql5.com/2/1/defaultvalue.png) |
| Class CCellButton | ![](https://c.mql5.com/2/1/largebutton.png) |
| Class CCellButtonType | ![](https://c.mql5.com/2/1/knopki_ver1.gif) |

Table 1\. Library of cell classes

Let's take a detailed look at the creation of a single derived class of CCellButtonType. This class creates buttons of various types.

```
//+------------------------------------------------------------------+
//| CCellButtonType class                                            |
//+------------------------------------------------------------------+
class CCellButtonType:public CCell
  {
public:
   ///Class constructor
   void              CCellButtonType();
   virtual     ///Draw method
   void              Draw(string m_name,
                          int m_xdelta,
                          int m_ydelta,
                          int m_type);
  };
//+------------------------------------------------------------------+
//| CCellButtonType class constructor                                |
//+------------------------------------------------------------------+
void CCellButtonType::CCellButtonType()
  {
   type=OBJ_BUTTON;
   on_event=false;   //disable events processing
  }
//+------------------------------------------------------------------+
//| CCellButtonType class Draw method                                |
//+------------------------------------------------------------------+
void CCellButtonType::Draw(string m_name,
                           int m_xdelta,
                           int m_ydelta,
                           int m_type)
  {
//--- creating an object with specified name
   if(m_type<=0) m_type=0;
   name=m_name+".Button"+(string)m_type;
   if(ObjectCreate(0,name,type,0,0,0,0,0)==false)
      Print("Function ",__FUNCTION__," error ",GetLastError());
//--- object properties initializartion
   ObjectSetInteger(0,name,OBJPROP_COLOR,Property.TextColor);
   ObjectSetInteger(0,name,OBJPROP_BGCOLOR,Property.BGColor);
   ObjectSetInteger(0,name,OBJPROP_CORNER,Property.Corner);
   ObjectSetInteger(0,name,OBJPROP_XDISTANCE,m_xdelta);
   ObjectSetInteger(0,name,OBJPROP_YDISTANCE,m_ydelta);
   ObjectSetInteger(0,name,OBJPROP_XSIZE,Property.H);
   ObjectSetInteger(0,name,OBJPROP_YSIZE,Property.H);
   ObjectSetInteger(0,name,OBJPROP_SELECTABLE,0);
   if(m_type==0) // Hide button
     {
      ObjectSetString(0,name,OBJPROP_TEXT,CharToString(MIN_WIN));
      ObjectSetString(0,name,OBJPROP_FONT,"Webdings");
      ObjectSetInteger(0,name,OBJPROP_FONTSIZE,12);
     }
   if(m_type==1) // Close button
     {
      ObjectSetString(0,name,OBJPROP_TEXT,CharToString(CLOSE_WIN));
      ObjectSetString(0,name,OBJPROP_FONT,"Wingdings 2");
      ObjectSetInteger(0,name,OBJPROP_FONTSIZE,8);
     }
   if(m_type==2) // Return button
     {
      ObjectSetString(0,name,OBJPROP_TEXT,CharToString(MAX_WIN));
      ObjectSetString(0,name,OBJPROP_FONT,"Webdings");
      ObjectSetInteger(0,name,OBJPROP_FONTSIZE,12);
     }
   if(m_type==3) // Plus button
     {
      ObjectSetString(0,name,OBJPROP_TEXT,"+");
      ObjectSetString(0,name,OBJPROP_FONT,"Arial");
      ObjectSetInteger(0,name,OBJPROP_FONTSIZE,10);
     }
   if(m_type==4) // Minus button
     {
      ObjectSetString(0,name,OBJPROP_TEXT,"-");
      ObjectSetString(0,name,OBJPROP_FONT,"Arial");
      ObjectSetInteger(0,name,OBJPROP_FONTSIZE,13);
     }
   if(m_type==5) // PageUp button
     {
      ObjectSetString(0,name,OBJPROP_TEXT,CharToString(PAGE_UP));
      ObjectSetString(0,name,OBJPROP_FONT,"Wingdings 3");
      ObjectSetInteger(0,name,OBJPROP_FONTSIZE,8);
     }
   if(m_type==6) // PageDown button
     {
      ObjectSetString(0,name,OBJPROP_TEXT,CharToString(PAGE_DOWN));
      ObjectSetString(0,name,OBJPROP_FONT,"Wingdings 3");
      ObjectSetInteger(0,name,OBJPROP_FONTSIZE,8);
     }
   if(m_type>6) // empty button
     {
      ObjectSetString(0,name,OBJPROP_TEXT,"");
      ObjectSetString(0,name,OBJPROP_FONT,"Arial");
      ObjectSetInteger(0,name,OBJPROP_FONTSIZE,13);
     }
   on_event=true;   //enable events processing
  }
//+------------------------------------------------------------------+
```

Necessary explanations:

- We introduce a ban on the processing of events into the class constructor. This is necessary to prepare the object for work and eliminate the distractions of incoming events. Upon the completion of all necessary operations, we will allow such processing, and object will begin to fully function.
- The draw method uses internal data and receives external data. Therefore, the data should be first tested for compliance, and only then be processed, in order to avoid exceptional situations. But we will not perform this test in this particular case. Why? Imagine that the class object is a soldier, and soldiers do not necessarily need to know the plans of the generals. Their job is to clearly, quickly and rigorously follow the orders of their commanders, instead of analyzing the received commands and making independent decisions. Therefore, all of the external data must be complied before we begin working with his class.


Now we must test the entire library of cells. To do this, we will insert the following code into the main module (temporarily for testing purposes) and run the Expert Advisor.

```
//--- include file with classes
#include <ClassUnit.mqh>
//+------------------------------------------------------------------+
//| Main module: CMasterWindows class                                |
//+------------------------------------------------------------------+
class CMasterWindows
  {
protected:
   bool              on_event;   // events processing flag
   WinCell           Property;   // cell property
   CCellText         Text;
   CCellEdit         Edit;
   CCellButton       Button;
   CCellButtonType   ButtonType;
public:
   // Class constructor
   void              CMasterWindows();
   // Main module run method (core algorithm)
   void              Run();
   // Deinitialization method
   void              Deinit();
   // OnTick event processing method
   void              OnEventTick();
   // OnChart event processing method
   void              OnEvent(const int id,
                             const long &lparam,
                             const double &dparam,
                             const string &sparam);
  };
//+------------------------------------------------------------------+
//| Main module run method (core algorithm)                          |
//+------------------------------------------------------------------+
void CMasterWindows::Run()
  {
//--- core algorithm - it launches additional modules
   ObjectsDeleteAll(0,0,-1);
   Comment("MasterWindows for MQL5     © DC2008");
//--- Text field
   Text.Draw("Text",50,50,150,"Text field");
//--- Edit field
   Edit.Draw("Edit",205,50,150,"default value",true);
//--- LARGE BUTTON
   Button.Draw("Button",50,80,200,"LARGE BUTTON");
//--- Hide button
   ButtonType.Draw("type0",50,100,0);
//--- Close button
   ButtonType.Draw("type1",70,100,1);
//--- Return  button
   ButtonType.Draw("type2",90,100,2);
//--- Plus button
   ButtonType.Draw("type3",110,100,3);
//--- Minus button
   ButtonType.Draw("type4",130,100,4);
//--- None button
   ButtonType.Draw("type5",150,100,5);
//--- None button
   ButtonType.Draw("type6",170,100,6);
//--- None button
   ButtonType.Draw("type7",190,100,7);
//---
   on_event=true;   // enable events processing
  }
```

And we must not forget to transfer events for the resulting classes! If this is not done, handling projects can become very difficult or even impossible.

```
//+------------------------------------------------------------------+
//| CMasterWindows class OnChart event processing method             |
//+------------------------------------------------------------------+
void CMasterWindows::OnEvent(const int id,
                             const long &lparam,
                             const double &dparam,
                             const string &sparam)
  {
   if(on_event) // event processing is enabled
     {
      //--- process events for the cell class objects
      Text.OnEvent(id,lparam,dparam,sparam);
      Edit.OnEvent(id,lparam,dparam,sparam);
      Button.OnEvent(id,lparam,dparam,sparam);
      ButtonType.OnEvent(id,lparam,dparam,sparam);
     }
  }
```

As a result we see all of the available options for objects of the library of cell classes.

![Figure 2. Library of cell classes ](https://c.mql5.com/2/1/figure2.png)

Figure 2\. Library of cell
classes

Let's test the working efficiency and the responses of objects to events:

- We enter into the editing field different variables, instead of "default". If the values vary, then testing was successful.

- We press the buttons, they remain in the pressed state until they are pressed again. This is not, however, a satisfying reaction. We need the button to return to its original state automatically, after we press it once. And this is where we can demonstrate the power of the OOP - the possibility of inheritance. Our program can be using more than a dozen buttons and it isn't necessary to add the desired functionality for each one of them separately. It is suffice enough to change the CCell base class, and all of the objects of derived classes will miraculously start working properly!


```
//+------------------------------------------------------------------+
//| CCell class OnChart event processing method                      |
//+------------------------------------------------------------------+
void CCell::OnEvent(const int id,
                    const long &lparam,
                    const double &dparam,
                    const string &sparam)
  {
   if(on_event) // event processing is enabled
     {
      //--- button click event
      if(id==CHARTEVENT_OBJECT_CLICK && StringFind(sparam,".Button",0)>0)
        {
         if(ObjectGetInteger(0,sparam,OBJPROP_STATE)==1)
           {
            //--- if button stays pressed
            Sleep(TIME_SLEEP);
            ObjectSetInteger(0,sparam,OBJPROP_STATE,0);
            ChartRedraw();
           }
        }
     }
  }
```

Thus, the library of class cells is tested and linked to the project.

The next step is adding a library of lines:

| Name class | Image |
| --- | --- |
| Class CRowType1 (0) | ![](https://c.mql5.com/2/1/vid1.gif) |
| Class CRowType1 (1) | ![](https://c.mql5.com/2/1/vid2__1.gif) |
| Class CRowType1 (2) | ![](https://c.mql5.com/2/1/vid3__1.gif) |
| Class CRowType1 (3) | ![](https://c.mql5.com/2/1/vid4__1.gif) |
| Class CRowType2 | ![](https://c.mql5.com/2/1/vid2__2.gif) |
| Class CRowType3 | ![](https://c.mql5.com/2/1/vid3__2.gif) |
| Class CRowType4 | ![](https://c.mql5.com/2/1/vid4__2.gif) |
| Class CRowType5 | ![](https://c.mql5.com/2/1/vid5__1.gif) |
| Class CRowType6 | ![](https://c.mql5.com/2/1/vid6__1.gif) |

Table 2\. Library of line classes

and we test it in the same way. After all of the testing we proceed to the next stage.

**2.4 Stage IV: Constructing the Project**

At this point, all of the necessary modules have been created and tested. Now we proceed to constructing the project. First we create a cascade: the shape of the window as in Figure 1 and fill it with functionality, i.e. programmed reactions of all of the elements and modules to incoming events.

To do this, we have a ready frame of the program and preparation of the main module. Let's begin with this. It is one of the "descendant" classes, of the CWin base class, therefore, all of the public methods and fields of the "ancestor" class were passed on to it by inheritance. Therefore we simply need to override a few methods and a new CMasterWindows class is ready:

```
//--- include files with classes
#include <ClassWin.mqh>
#include <InitMasterWindows.mqh>
#include <ClassMasterWindowsEXE.mqh>
//+------------------------------------------------------------------+
//| CMasterWindows class                                             |
//+------------------------------------------------------------------+
class CMasterWindows:public CWin
  {
protected:
   CMasterWindowsEXE WinEXE;     // executable module
public:
   void              Run();      // Run method
   void              Deinit();   // Deinitialization method
   virtual                       // OnChart event processing method
   void              OnEvent(const int id,
                             const long &lparam,
                             const double &dparam,
                             const string &sparam);
  };
//+------------------------------------------------------------------+
//| CMasterWindows class deinitialization method                     |
//+------------------------------------------------------------------+
void CMasterWindows::Deinit()
  {
//---(delete all objects)
   ObjectsDeleteAll(0,0,-1);
   Comment("");
  }
//+------------------------------------------------------------------+
//| CMasterWindows class Run method                                  |
//+------------------------------------------------------------------+
void CMasterWindows::Run()
  {
   ObjectsDeleteAll(0,0,-1);
   Comment("MasterWindows for MQL5     © DC2008");
//--- creating designer window and launch executable object
   SetWin("CWin1",1,30,250,CORNER_RIGHT_UPPER);
   Draw(Mint,Mstr,21);
   WinEXE.Init("CWinNew",30,18);
   WinEXE.Run();
  }
//+------------------------------------------------------------------+
//| CMasterWindows class event processing method                     |
//+------------------------------------------------------------------+
void CMasterWindows::OnEvent(const int id,
                             const long &lparam,
                             const double &dparam,
                             const string &sparam)
  {
   if(on_event) // event processing is enabled
     {
      //--- Close button click in the main window
      if(id==CHARTEVENT_OBJECT_CLICK
         && StringFind(sparam,"CWin1",0)>=0
         && StringFind(sparam,".Button1",0)>0)
        {
         ExpertRemove();
        }
      //--- OnChart event processing for all objects
      STR1.OnEvent(id,lparam,dparam,sparam);
      STR2.OnEvent(id,lparam,dparam,sparam);
      STR3.OnEvent(id,lparam,dparam,sparam);
      STR4.OnEvent(id,lparam,dparam,sparam);
      STR5.OnEvent(id,lparam,dparam,sparam);
      STR6.OnEvent(id,lparam,dparam,sparam);
      WinEXE.OnEvent(id,lparam,dparam,sparam);
     }
  }
```

By itself, the main module is pretty small, since it is responsible for nothing else but the creation of the application window. Next it passes the control to the executable WinEXE module, where the most interesting thing takes place - the reaction to incoming events.

Previously, we created a simple WinCell structure for the exchange of data between objects, and now, all of the advantages of this approach become clear. The process of copying all of the members of the structure is very rational and compact:

```
   STR1.Property = Property;
   STR2.Property = Property;
   STR3.Property = Property;
   STR4.Property = Property;
   STR5.Property = Property;
   STR6.Property = Property;
```

At this stage we can end the detailed consideration of the class design and move on to the visual technology of their construction, which considerably speeds up the process of creating new classes.

### 3\. Visual design of classes

A class can be constructed much faster, and can be visualized easier, in the mode of visual MasterWindows design for MQL5:

![Figure 3. The process of visual design](https://c.mql5.com/2/1/figure3.png)

Figure 3. The process of visual design

All that is required from the developer - is to draw the window form, using the means of MasterWindows form, and then, simply to determine the reaction to the planned event. The code itself is created automatically. And that's it! The project is completed.

An example of a generated code of the CMasterWindows class, as well as of the Expert Advisor, is shown in Figure 4 (a file is created in the folder ...\\MQL5\\Files):

```
//****** Project (Expert Advisor): project1.mq5
//+------------------------------------------------------------------+
//|        Code has been generated by MasterWindows Copyright DC2008 |
//|                                              http://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "DC2008"
//--- include files with classes
#include <ClassWin.mqh>
int Mint[][3]=
  {
     {1,0,0},
     {2,100,0},
     {1,100,0},
     {3,100,0},
     {4,100,0},
     {5,100,0},
     {6,100,50},
     {}
  };
string Mstr[][3]=
  {
     {"New window","",""},
     {"NEW1","new1",""},
     {"NEW2","new2",""},
     {"NEW3","new3",""},
     {"NEW4","new4",""},
     {"NEW5","new5",""},
     {"NEW6","new6",""},
     {}
  };
//+------------------------------------------------------------------+
//| CMasterWindows class (main unit)                                 |
//+------------------------------------------------------------------+
class CMasterWindows:public CWin
  {
private:
   long              Y_hide;          // Window shift vertical in hide mode
   long              Y_obj;           // Window shift vertical
   long              H_obj;           // Window shift horizontal
public:
   bool              on_hide;         // HIDE mode flag
   CArrayString      units;           // Main window lines
   void              CMasterWindows() {on_event=false; on_hide=false;}
   void              Run();           // Run method
   void              Hide();          // Hide method
   void              Deinit()         {ObjectsDeleteAll(0,0,-1); Comment("");}
   virtual void      OnEvent(const int id,
                             const long &lparam,
                             const double &dparam,
                             const string &sparam);
  };
//+------------------------------------------------------------------+
//| CMasterWindows class Run method                                  |
//+------------------------------------------------------------------+
void CMasterWindows::Run()
  {
   ObjectsDeleteAll(0,0,-1);
   Comment("Code has been generated by MasterWindows for MQL5 © DC2008");
//--- creating main window and launch executable module
   SetWin("project1.Exp",50,100,250,CORNER_LEFT_UPPER);
   Draw(Mint,Mstr,7);
  }
//+------------------------------------------------------------------+
//| CMasterWindows class Hide method                                 |
//+------------------------------------------------------------------+
void CMasterWindows::Hide()
  {
   Y_obj=w_ydelta;
   H_obj=Property.H;
   Y_hide=ChartGetInteger(0,CHART_HEIGHT_IN_PIXELS,0)-Y_obj-H_obj;;
//---
   if(on_hide==false)
     {
      int n_str=units.Total();
      for(int i=0; i<n_str; i++)
        {
         long y_obj=ObjectGetInteger(0,units.At(i),OBJPROP_YDISTANCE);
         ObjectSetInteger(0,units.At(i),OBJPROP_YDISTANCE,(int)y_obj+(int)Y_hide);
         if(StringFind(units.At(i),".Button0",0)>0)
            ObjectSetString(0,units.At(i),OBJPROP_TEXT,CharToString(MAX_WIN));
        }
     }
   else
     {
      int n_str=units.Total();
      for(int i=0; i<n_str; i++)
        {
         long y_obj=ObjectGetInteger(0,units.At(i),OBJPROP_YDISTANCE);
         ObjectSetInteger(0,units.At(i),OBJPROP_YDISTANCE,(int)y_obj-(int)Y_hide);
         if(StringFind(units.At(i),".Button0",0)>0)
            ObjectSetString(0,units.At(i),OBJPROP_TEXT,CharToString(MIN_WIN));
        }
     }
//---
   ChartRedraw();
   on_hide=!on_hide;
  }
//+------------------------------------------------------------------+
//| CMasterWindows class OnChartEvent event processing method        |
//+------------------------------------------------------------------+
void CMasterWindows::OnEvent(const int id,
                             const long &lparam,
                             const double &dparam,
                             const string &sparam)
  {
   if(on_event // event handling is enabled
      && StringFind(sparam,"project1.Exp",0)>=0)
     {
      //--- call of OnChartEvent handlers
      STR1.OnEvent(id,lparam,dparam,sparam);
      STR2.OnEvent(id,lparam,dparam,sparam);
      STR3.OnEvent(id,lparam,dparam,sparam);
      STR4.OnEvent(id,lparam,dparam,sparam);
      STR5.OnEvent(id,lparam,dparam,sparam);
      STR6.OnEvent(id,lparam,dparam,sparam);
      //--- creating graphic object
      if(id==CHARTEVENT_OBJECT_CREATE)
        {
         if(StringFind(sparam,"project1.Exp",0)>=0) units.Add(sparam);
        }
      //--- edit [NEW1] in Edit STR1
      if(id==CHARTEVENT_OBJECT_ENDEDIT
         && StringFind(sparam,".STR1",0)>0)
        {
        //--- event processing code
        }
      //--- edit [NEW3] : Plus button STR3
      if(id==CHARTEVENT_OBJECT_CLICK
         && StringFind(sparam,".STR3",0)>0
         && StringFind(sparam,".Button3",0)>0)
        {
        //--- event processing code
        }
      //--- edit [NEW3] : Minus button STR3
      if(id==CHARTEVENT_OBJECT_CLICK
         && StringFind(sparam,".STR3",0)>0
         && StringFind(sparam,".Button4",0)>0)
        {
        //--- event processing code
        }
      //--- edit [NEW4] : Plus button STR4
      if(id==CHARTEVENT_OBJECT_CLICK
         && StringFind(sparam,".STR4",0)>0
         && StringFind(sparam,".Button3",0)>0)
        {
        //--- event processing code
        }
      //--- edit [NEW4] : Minus button STR4
      if(id==CHARTEVENT_OBJECT_CLICK
         && StringFind(sparam,".STR4",0)>0
         && StringFind(sparam,".Button4",0)>0)
        {
        //--- event processing code
        }
      //--- edit [NEW4] : Up button STR4
      if(id==CHARTEVENT_OBJECT_CLICK
         && StringFind(sparam,".STR4",0)>0
         && StringFind(sparam,".Button5",0)>0)
        {
        //--- event processing code
        }
      //--- edit [NEW4] : Down button STR4
      if(id==CHARTEVENT_OBJECT_CLICK
         && StringFind(sparam,".STR4",0)>0
         && StringFind(sparam,".Button6",0)>0)
        {
        //--- event processing code
        }
      //--- [new5] button click STR5
      if(id==CHARTEVENT_OBJECT_CLICK
         && StringFind(sparam,".STR5",0)>0
         && StringFind(sparam,".Button",0)>0)
        {
        //--- event processing code
        }
      //--- [NEW6] button click STR6
      if(id==CHARTEVENT_OBJECT_CLICK
         && StringFind(sparam,".STR6",0)>0
         && StringFind(sparam,"(1)",0)>0)
        {
        //--- event processing code
        }
      //--- [new6] button click STR6
      if(id==CHARTEVENT_OBJECT_CLICK
         && StringFind(sparam,".STR6",0)>0
         && StringFind(sparam,"(2)",0)>0)
        {
        //--- event processing code
        }
      //--- button click [] STR6
      if(id==CHARTEVENT_OBJECT_CLICK
         && StringFind(sparam,".STR6",0)>0
         && StringFind(sparam,"(3)",0)>0)
        {
        //--- event processing code
        }
      //--- Close button click in the main window
      if(id==CHARTEVENT_OBJECT_CLICK
         && StringFind(sparam,".Button1",0)>0)
        {
         ExpertRemove();
        }
      //--- Hide button click in the main window
      if(id==CHARTEVENT_OBJECT_CLICK
         && StringFind(sparam,".Button0",0)>0)
        {
         Hide();
        }
     }
  }
//--- Main module declaration
CMasterWindows MasterWin;
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//--- launch main module
   MasterWin.Run();
   return(0);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//--- main module deinitialization
   MasterWin.Deinit();
  }
//+------------------------------------------------------------------+
//| Expert Event function                                            |
//+------------------------------------------------------------------+
void OnChartEvent(const int id,
                  const long &lparam,
                  const double &dparam,
                  const string &sparam)
  {
//--- call OnChartEvent event handler
   MasterWin.OnEvent(id,lparam,dparam,sparam);
  }
```

With the launch of this, we see the following designed window:

![Figure 4. Advisor project1 - the result of visual design of classes](https://c.mql5.com/2/1/figure4.png)

Figure 4\. Expert Advisor project1 - the result of visual design of classes

### Conclusion

1. Classes need to be designed stage by stage. By breaking down the task into modules, a separate class is created for each one of them. The modules, in turn, are broken down into micromodules of derived or base classes.

2. Try not to overload the base classes with built-in methods - the number of these should be kept to a minimum.

3. The design of classes with the use of visual design environment is very simple, even for a "dummie", because the code is generated automatically.

Location of attachments:

- masterwindows.mq5 - ...\\MQL5\\Experts\
- remaining in the folder - ...\\MQL5\\Include\


Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/53](https://www.mql5.com/ru/articles/53)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/53.zip "Download all attachments in the single ZIP archive")

[masterwindows-doc-en.zip](https://www.mql5.com/en/articles/download/53/masterwindows-doc-en.zip "Download masterwindows-doc-en.zip")(274.93 KB)

[masterwindows-en.zip](https://www.mql5.com/en/articles/download/53/masterwindows-en.zip "Download masterwindows-en.zip")(13.39 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Developing multi-module Expert Advisors](https://www.mql5.com/en/articles/3133)
- [3D Modeling in MQL5](https://www.mql5.com/en/articles/2828)
- [Statistical distributions in the form of histograms without indicator buffers and arrays](https://www.mql5.com/en/articles/2714)
- [The ZigZag Indicator: Fresh Approach and New Solutions](https://www.mql5.com/en/articles/646)
- [Calculation of Integral Characteristics of Indicator Emissions](https://www.mql5.com/en/articles/610)
- [Testing Performance of Moving Averages Calculation in MQL5](https://www.mql5.com/en/articles/106)
- [Migrating from MQL4 to MQL5](https://www.mql5.com/en/articles/81)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/1063)**
(54)


![alventa](https://c.mql5.com/avatar/avatar_na2.png)

**[alventa](https://www.mql5.com/en/users/alventa)**
\|
3 May 2016 at 11:03

When trying to compile MasterWindows it complains "'CMasterWindowsEXE::Run' - cannot call protected member [function](https://www.mql5.com/en/docs/constants/namedconstants/compilemacros "MQL5 Documentation: Predefined Macro Substitutions") ClassMasterWindows.mqh" on the line "WinEXE.Run()". Can you tell me how to fight it?


![Sergey Pavlov](https://c.mql5.com/avatar/2010/2/4B7AECD8-6F67.jpg)

**[Sergey Pavlov](https://www.mql5.com/en/users/dc2008)**
\|
3 May 2016 at 16:32

**alventa:**

When trying to compile MasterWindows it complains "'CMasterWindowsEXE::Run' - cannot call protected member function ClassMasterWindows.mqh" on the line "WinEXE.Run()". Can you tell me how to fight it?

[MasterWindows](https://www.mql5.com/ru/code/15059) library is available in **CodeBase**. There are also examples of using this library in panels [\[1](https://www.mql5.com/ru/code/15224), [2](https://www.mql5.com/ru/code/15241), [3](https://www.mql5.com/ru/code/15412)\].

![Vitaliy Kostrubko](https://c.mql5.com/avatar/2016/8/579E94F7-83FB.png)

**[Vitaliy Kostrubko](https://www.mql5.com/en/users/bbk30)**
\|
16 Nov 2024 at 08:59

**Sergey Pavlov [#](https://www.mql5.com/ru/forum/782#comment_4405) 4:**

Try the compiled file.

Sergey hello !

... I downloaded your compiled file (from post #4) to familiarise myself with the possibilities ... threw it into the folder \\MQL5\\Experts, but it DOES NOT APPEAR on the chart (!) :(

![Vitaliy Kostrubko](https://c.mql5.com/avatar/2016/8/579E94F7-83FB.png)

**[Vitaliy Kostrubko](https://www.mql5.com/en/users/bbk30)**
\|
20 Nov 2024 at 17:34

**Sergey Pavlov [#](https://www.mql5.com/ru/forum/782/page3#comment_4478) 29:**

I recommend to update MasterWindows. Now two files are generated simultaneously: Expert Advisor and indicator.

... downloaded from this post ... when compiling it gives errors :

[![](https://c.mql5.com/3/448/1035816389366__1.png)](https://c.mql5.com/3/448/1035816389366.png "https://c.mql5.com/3/448/1035816389366.png")

![lynxntech](https://c.mql5.com/avatar/2022/7/62CF9DBF-A3CD.png)

**[lynxntech](https://www.mql5.com/en/users/lynxntech)**
\|
21 Nov 2024 at 05:29

**Vitaliy Kostrubko [#](https://www.mql5.com/ru/forum/782/page5#comment_55176833):**

... downloaded from this post ... when compiling it gives errors :

it looks like you are starting from the wrong place,

You should start with the basics of MQL5 programming, and then classes and try to build a panel and embed trading functions.

If you need to trade with this panel, it is better to buy a ready-made one, there is a choice, and you should not learn programming from this article at all.

If you build a panel here, what will you do with it? The one who can do all the things you have described will say that you don't need your developments..... and use his own from scratch.

And the one who just knows programming, not panels, will spend half a year building these trade desires for the panel under your sketch from the article.

add

this article may be useful for those who want to improve their procedural skills before Classes.

![OOP in MQL5 by Example: Processing Warning and Error Codes](https://c.mql5.com/2/0/mistake.png)[OOP in MQL5 by Example: Processing Warning and Error Codes](https://www.mql5.com/en/articles/70)

The article describes an example of creating a class for working with the trade server return codes and all the errors that occur during the MQL-program run. Read the article, and you will learn how to work with classes and objects in MQL5. At the same time, this is a convenient tool for handling errors; and you can further change this tool according to your specific needs.

![Creating Active Control Panels in MQL5 for Trading](https://c.mql5.com/2/0/panel__2.png)[Creating Active Control Panels in MQL5 for Trading](https://www.mql5.com/en/articles/62)

The article covers the problem of development of active control panels in MQL5. Interface elements are managed by the event handling mechanism. Besides, the option of a flexible setup of control elements properties is available. The active control panel allows working with positions, as well setting, modifying and deleting market and pending orders.

![The Algorithm of Ticks' Generation within the Strategy Tester of the MetaTrader 5 Terminal](https://c.mql5.com/2/0/Ticks_Modelling_Algorithm_Metatrader5.png)[The Algorithm of Ticks' Generation within the Strategy Tester of the MetaTrader 5 Terminal](https://www.mql5.com/en/articles/75)

MetaTrader 5 allows us to simulate automatic trading, within an embedded strategy tester, by using Expert Advisors and the MQL5 language. This type of simulation is called testing of Expert Advisors, and can be implemented using multithreaded optimization, as well as simultaneously on a number of instruments. In order to provide a thorough testing, a generation of ticks based on the available minute history, needs to be performed. This article provides a detailed description of the algorithm, by which the ticks are generated for the historical testing in the MetaTrader 5 client terminal.

![Practical Application Of Databases For Markets Analysis](https://c.mql5.com/2/0/dar.png)[Practical Application Of Databases For Markets Analysis](https://www.mql5.com/en/articles/69)

Working with data has become the main task for modern software - both for standalone and network applications. To solve this problem a specialized software were created. These are Database Management Systems (DBMS), that can structure, systematize and organize data for their computer storage and processing. As for trading, the most of analysts don't use databases in their work. But there are tasks, where such a solution would have to be handy. This article provides an example of indicators, that can save and load data from databases both with client-server and file-server architectures.

[![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/01.png)![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/02.png)Powerful analytics for traders of any levelAll the necessary trading reports for beginners and professionals](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/454106&a=muccpajyfystoakuukdobwigjejzmpqn&s=52daad60fa795e635264e6f94898f05493bca3b5124d4cca8eb7e82333c2ef12&uid=&ref=https://www.mql5.com/en/articles/53&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5083419512600009564)

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