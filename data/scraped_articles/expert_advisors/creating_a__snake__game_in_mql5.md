---
title: Creating a "Snake" Game in MQL5
url: https://www.mql5.com/en/articles/65
categories: Expert Advisors
relevance_score: 6
scraped_at: 2026-01-23T17:31:25.247031
---

[![](https://www.mql5.com/ff/sh/ub4fqgrk4rkv8gz9z2/01.png)![](https://www.mql5.com/ff/sh/ub4fqgrk4rkv8gz9z2/02.png)Explore your trading for freeUpdated statistics in MetaTrader 5 will help you to thoroughly evaluate results and reduce risksLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/454106&a=bkbqgaxtrafeuegfvjisjjwjohagrvnr&s=25c5856d7857fc6b6db7cffb15ae4ce40fd19d1ab594d8a900ad65673d9ffa0e&uid=&ref=https://www.mql5.com/en/articles/65&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5068334466469918778)

MetaTrader 5 / Expert Advisors


### Introduction

In this article, we will consider an example of writing a "Snake" game in MQL5.

Since the 5th version of MQL, the game programming became possible, primarily due to [events processing](https://www.mql5.com/en/docs/basis/function/events) features, including the [custom events](https://www.mql5.com/en/docs/eventfunctions/eventchartcustom). The object-oriented programming simplifies the design of such programs, makes the code clearer and reduces the number of errors.

After the reading of this article, you will learn about the [OnChart events processing](https://www.mql5.com/en/docs/basis/function/events), examples of use of the [MQL5 standard library](https://www.mql5.com/en/docs/standardlibrary) classes and recipes for cyclic calls of the function after a certain time to perform any calculations.

### **Game Description**

The "Snake" game has been chosen as an example, primarily, because of the simplicity of its implementation. Everyone, who had a great interest in programming is able to write this game.

According to Wikipedia:

Snake is a video game first released during the mid 1970s in arcades and has maintained popularity since then, becoming somewhat of a classic.

The player controls a long, thin creature, resembling a snake, which roams around on a bordered plane, picking up food (or some other item), trying to avoid hitting its own tail or the "walls" that surround the playing area. In some variations on the field there are additional obstacles. Each time the snake eats a piece of food, its tail grows longer, making the game increasingly difficult. The user controls the direction of the snake's head (up, down, left, or right), and the snake's body follows. The player cannot stop the snake from moving while the game is in progress, and cannot make the snake go in reverse.

This implementation of "Snake" in MQL5 will have some limitations and features.

The number of levels is equal to 6 (from 0 to 5). There are 5 lives available at each level. After the use of all lives or after  he successful passing of all levels, the game will return to the initial level. You can create your own levels. The snake speed and its maximal length is the same for each level.

The game field consists of 4 elements:

1. Game title. It's used for the game positioning on the chart. By moving the title, all the game elements are moved.

2. Playing field. It's an array (table) of cells with dimensions 20x20. Each cell has a size of 20x20 pixels. The elements on the playing field are:

   - The Snake. It consists of at least three consecutive elements - head, body and tail. The head can be moved left, right, up and down. All other snake elements are moved after the head.
   - The Obstacle. It's represented as grey rectangle, for the case of snake's head collision with the obstacle, the current level is restarted and the number of lives is decreased by 1.
   - The Food. The food is presented by berry, for the case of snake's head collision with the food, the snake's size (length of its body) is increased.
      After the eating of 12 pieces, the snake goes to the next level.

4. The Information Panel (status bar of the game). It consists of three elements:

   - Level. Shows the current level.
   - Food left over. Shows how much of berries is left to eat.
   - Lives. Shows the number of lives available.

6. Panel. It consists of three buttons:
   - Button "Start". Starts the current level.
   - The button "Pause". Pauses the game.
   - The button "Stop". Stops the game, while the transition occurs at the initial level.

All these elements can be seen at Figure 1:

![](https://c.mql5.com/2/1/Image1_en.png)

Fig. 1. Elements of the "Snake"game

The game title is an object of "Button" type. All playing field elements are the objects of "BmpLabel" type. The Information panel consists of three objects of "Edit" type, the Control Panel consists of three objects "Button" type. All objects are positioned by definition of distances along X and Y in pixels relative to the upper left corner of the chart.

It should be noted that playing field edges isn't an obstacle for the snake's movement. For example, then the snake comes through the left edge, it appears on the right. It can be seen at Figure 2:

![](https://c.mql5.com/2/1/Image2___1.png)

Figure 2. Passing of the snake through the playing field edge

The snake's head and its tail, unlike the body can be rotated. The direction of head is determined by snake movement direction or by position of its neighboring elements. The tail's direction is determined only by position of the neighboring element.

For example, if the neighbour tail element is on the left side, the tail is turned to the left. A little differently is with the head. The head is turned left, if its neighboring element is on the right. side The examples of head and tail directions are presented at the figures below. Pay attention to the turn of the head and tail relative to their neighboring elements.

| ![](https://c.mql5.com/2/1/pic1__1.png) | ![](https://c.mql5.com/2/1/pic2__1.png) | ![](https://c.mql5.com/2/1/pic3__1.png) | ![](https://c.mql5.com/2/1/pic4__1.png) |
| --- | --- | --- | --- |
| The head and tail are directed to the left | The head and tail are directed to the right | The head and tail are directed to the down | The head and tail are directed to the up |

The snake movement is carried out in three stages:

1. The one cell head movement to the right, left, up or down depending on direction.
2. The movement of the last element of a snake body on the previous head place.
3. Moving the snake's tail on the previous place of the last body element.
    The movement of snake's tail on the previous place of the last snake's body element.

If the snake eats the food, the tail does not moved. Instead, a new element of the body is created, that moved on the past place of the last snake's body element.

An example of snake movement to the left is presented at figures below:

| ![](https://c.mql5.com/2/1/pic5__1.png) | ![](https://c.mql5.com/2/1/pic6__1.png) | ![](https://c.mql5.com/2/1/pic7__1.png) | ![](https://c.mql5.com/2/1/pic8__1.png) |
| --- | --- | --- | --- |
| Initial position | One cell to the left head movement | Movement of the last body element<br> to the previous place of the head | Tail movement on the past place<br> of the last body element |

### Theory

Next we will discuss the tools and techniques which are used when writing games.

### The MQL5 Standard Library

It's convenient to use the arrays of objects of the same type (for example, playing field cells, snake elements) to manipulate them (create, move, delete). These arrays and objects can be implemented using the  MQL5 Standard Library classes.

The use of the [MQL5 Standard Library](https://www.mql5.com/en/docs/standardlibrary) classes allows to simplify the process of programs writing. For the game, we will use the following library classes:

1. The [CArrayObj](https://www.mql5.com/en/docs/standardlibrary/datastructures/carrayobj) is class for the [data organization](https://www.mql5.com/en/docs/standardlibrary/datastructures) (dynamic array of pointers).
2. The [CChartObjectEdit](https://www.mql5.com/en/docs/standardlibrary/chart_object_classes/obj_controls/cchartobjectedit) , [CChartObjectButton](https://www.mql5.com/en/docs/standardlibrary/chart_object_classes/obj_controls/cchartobjectbutton) , [CChartObjectBmpLabel](https://www.mql5.com/en/docs/standardlibrary/chart_object_classes/obj_controls/cchartobjectbmplabel) \- are [control classes](https://www.mql5.com/en/docs/standardlibrary/chart_object_classes/obj_controls) , representing, respectively, "Edit", "Button" and "BmpLabel" respectively.

To use the [MQL5 Standard Library](https://www.mql5.com/en/docs/standardlibrary) classes, it's necessary to include them using the following compiler directive:

```
#include <path_to_the_file_with_classes_description>
```

For example, for using of objects of [CChartObjectButton](https://www.mql5.com/en/docs/standardlibrary/chart_object_classes/obj_controls/cchartobjectbutton) type we need to write:

```
#include <ChartObjects\ChartObjectsTxtControls.mqh>
```

File paths can be found in the [MQL5 Reference](https://www.mql5.com/en/docs/standardlibrary).

When working with the [MQL5 Standard Library](https://www.mql5.com/en/docs/standardlibrary) classes it's important to understand that some of them inherit each other. For example, the [CChartObjectButton](https://www.mql5.com/en/docs/standardlibrary/chart_object_classes/obj_controls/cchartobjectbutton) class inherits the [CChartObjectEdit](https://www.mql5.com/en/docs/standardlibrary/chart_object_classes/obj_controls/cchartobjectedit) class, in turn the [CChartObjectEdit](https://www.mql5.com/en/docs/standardlibrary/chart_object_classes/obj_controls/cchartobjectedit) class inherits the [CChatObjectLabel](https://www.mql5.com/en/docs/standardlibrary/chart_object_classes/obj_controls/cchartobjectlabel) class, etc. This means that for the derived classes, the parent class properties and methods are available.

To understand the advantages of use of the [MQL5 Standard Library](https://www.mql5.com/en/docs/standardlibrary) classes, let's consider an example of button creation and implement it in two ways (without and with use of the classes).

Here is an example without the use of the classes:

```
//Creating a button with name "button"
ObjectCreate(0,"button",OBJ_BUTTON,0,0,0);
//Specifying the text on the button
ObjectSetString(0,"button",OBJPROP_TEXT,"Button text");
//Specifying the button size
ObjectSetInteger(0,"button",OBJPROP_XSIZE,100);
ObjectSetInteger(0,"button",OBJPROP_YSIZE,20);
//Specifying the button position
ObjectSetInteger(0,"button",OBJPROP_XDISTANCE,10);
ObjectSetInteger(0,"button",OBJPROP_YDISTANCE,10);
```

An example with use of the classes:

```
CChartObjectButton *button;
//Creating an object of CChartObjectButton class and assign a pointer to the button variable
button=new CChartObjectButton;
//Creating a button with properties (in pixels): (width=100, height=20, positions: X=10,Y=10)
button.Create(0,"button",0,10,10,100,20);
//Specifying the text on the button
button.Description("Button text");
```

One can see, it simplier to work with the classes. In addition, the class objects can be stored in arrays and easily handled.

The methods and properties of the [Object controls](https://www.mql5.com/en/docs/standardlibrary/chart_object_classes/obj_controls) classes are well and
clearly described in the [MQL5 Reference](https://www.mql5.com/en/docs/standardlibrary) to the Standard Library classes.

We will use the [CArrayObj](https://www.mql5.com/en/docs/standardlibrary/datastructures/carrayobj) class of the [Standard Library](https://www.mql5.com/en/docs/standardlibrary) to arrange the objects array, it allows to relieve the user from many routine operations (such as an array resizing when adding a new element, the deletion of objects in the array, etc.)

### CArrayObj Class Features

The [CArrayObj](https://www.mql5.com/en/docs/standardlibrary/datastructures/carrayobj) class allows to organize a dynamic array of pointers to the objects of [CObject](https://www.mql5.com/en/docs/standardlibrary/cobject) class type. The CObject is a parent class for all classes of the [Standard Library](https://www.mql5.com/en/docs/standardlibrary). It means that we can create a dynamic array of pointers to the objects of any class of tge [Standard Library](https://www.mql5.com/en/docs/standardlibrary) classes. If you need to create a dynamic array of objects of your own class, it should be inherited from the CObject class.

In the following example, the compiler will not print errors, because the custom class is the successor of CObject class:

```
#include <Arrays\ArrayObj.mqh>
class CMyClass:public CObject
  {
   //fields and methods
  };
//creating an object of CMyClass type and assign it to the value of the my_obj variable
CMyClass *my_obj=new CMyClass;
//declaring a dynamic array of object pointers
CArrayObj array_obj;
//adding the my_obj object pointer at the end of the array_obj array
array_obj.Add(my_obj);
```

For the next case, the compiler will generate an error, because the my\_obj is not a pointer to the CObject class, or a class that inherits CObject class:

```
#include <Arrays\ArrayObj.mqh>
class CMyClass
  {
   //fields and methods
  };
//creating an object of CMyClass type and assing it to the value of the my_obj variable
CMyClass *my_obj=new CMyClass;
//declaring a dynamic array of object pointers
CArrayObj array_obj;
//adding the my_obj object pointer at the end of the array_obj array
array_obj.Add(my_obj);
```

When writing the game will use the following methods of class [CArrayObj](https://www.mql5.com/en/docs/standardlibrary/datastructures/carrayobj) :

- [Add](https://www.mql5.com/en/docs/standardlibrary/datastructures/carrayobj/carrayobjadd) \- Adds an element at the end of the array.
- [Insert](https://www.mql5.com/en/docs/standardlibrary/datastructures/carrayobj/carrayobjinsert) \- Inserts an element at the specified position of the array.
- [Detach](https://www.mql5.com/en/docs/standardlibrary/datastructures/carrayobj/carrayobjdetach) \- Deletes the element at the specified position (the element is removed from the array).
- [Total](https://www.mql5.com/en/docs/standardlibrary/datastructures/carray/carraytotal) \- Gets the number of elements in the array.
- [At](https://www.mql5.com/en/docs/standardlibrary/datastructures/carrayobj/carrayobjat) \- Gets the element at the specified position (the element isn't removed from the array).

Here is an example of work with the [CArrayObj](https://www.mql5.com/en/docs/standardlibrary/datastructures/carrayobj) class:

```
#include <Arrays\ArrayObj.mqh>
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CMyClass:public CObject
  {
   public:
   char   s;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void MyPrint(CArrayObj *array_obj)
  {
   CMyClass *my_obj;
   for(int i=0;i<array_obj.Total();i++)
     {
      my_obj=array_obj.At(i);
      printf("%C",my_obj.s);
     }
  }
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   //creating the pointer to the object of CArrayObj class
   CArrayObj *array_obj=new CArrayObj();
   //declaring the CMyClass object pointer
   CMyClass *my_obj;
   //filling the array_obj dynamic array
   for(int i='a';i<='c';i++)
     {
      //creating the CMyClass class object
      my_obj=new CMyClass();
      my_obj.s=char(i);
      //adding an object of CMyClass class at the end of the array_obj dynamic array
      array_obj.Add(my_obj);
     }
   //printing result
   MyPrint(array_obj);
   //creating new object of CMyClass class
   my_obj=new CMyClass();
   my_obj.s='d';
   //inserting new element at the first position of the array
   array_obj.Insert(my_obj,1);
   //printing result
   MyPrint(array_obj);
   //detaching the element from the third position of the array
   my_obj=array_obj.Detach(2);
   //printing result
   MyPrint(array_obj);
   //deleting the dynamic array and all objects with pointers of the array
   delete array_obj;
   return(0);
  }
```

In this example, the [OnInit](https://www.mql5.com/en/docs/basis/function/events) function creates a dynamic array with three elements. The output of the array contents is performed by the call of the MyPrint function.

After filling the array using the [Add](https://www.mql5.com/en/docs/standardlibrary/datastructures/carrayobj/carrayobjadd) method, its contents can be represented as (a, b, c).

After applying the [Insert](https://www.mql5.com/en/docs/standardlibrary/datastructures/carrayobj/carrayobjinsert) method, the contents of the array can be represented as (a, d, b, c).

Finally, after applying the [Detach](https://www.mql5.com/en/docs/standardlibrary/datastructures/carrayobj/carrayobjdetach) method, the array will look like (a, d, c).

When the [delete](https://www.mql5.com/en/docs/basis/operators/deleteoperator) operator is appled to the array\_obj variable, the [CArrayObj](https://www.mql5.com/en/docs/standardlibrary/datastructures/carrayobj) class desctructor is called, which removes not only the array\_obj array, but also the objects, whose pointers are stored in it.
To prevent it, before applying the [delete](https://www.mql5.com/en/docs/basis/operators/deleteoperator) commandm, the memory management flag of the [CArrayObj](https://www.mql5.com/en/docs/standardlibrary/datastructures/carrayobj) class should be set to false. This flag is set by [FreeMode](https://www.mql5.com/en/docs/standardlibrary/datastructures/carrayobj/carrayobjfreemodeconst) method .

If it isn't necessary to delete the objects, whose pointers are stored in the dynamic array when removing a dynamic array of object pointers, you should write the following code:

```
array_obj.FreeMode(false);
delete array_obj;
```

### Event Handling

If there is set of events generated, they accumulate in the queue, then they arrive consistently to the event processing function.

For the [events handling](https://www.mql5.com/en/docs/eventfunctions) generated when working with chart, as well as [custom events](https://www.mql5.com/en/docs/eventfunctions/eventchartcustom), MQL5 has the [OnChartEvent](https://www.mql5.com/en/docs/basis/function/events) function. An each event has an identifier and parameters, passed to the [OnChartEvent](https://www.mql5.com/en/docs/basis/function/events) function.

The [OnChartEvent](https://www.mql5.com/en/docs/basis/function/events) function is called only when the thread is out of all other program functions. Thus, in the following example, the [OnChartEvent](https://www.mql5.com/en/docs/basis/function/events) will never get control.

```
#include <ChartObjects\ChartObjectsTxtControls.mqh>
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void MyFunction()
  {
   CChartObjectButton *button;
   button=new CChartObjectButton;
   button.Create(0,"button",0,10,10,100,20);
   button.Description("Button text");
   while(true)
     {
      //The code, that should be called periodically
     }
  }
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   MyFunction();
   return(0);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void OnChartEvent(const int id,
                const long &lparam,
                const double &dparam,
                const string &sparam)
  {
   if(id==CHARTEVENT_OBJECT_CLICK && sparam=="button") Alert("Button click");
  }
```

An infinite [while](https://www.mql5.com/en/docs/basis/operators/while) loop doesn't allow to return from the MyFunction function. The [OnChartEvent](https://www.mql5.com/en/docs/basis/function/events) function cannot get control. Thus, press of the button doesn't calls the [Alert](https://www.mql5.com/en/docs/common/alert) function.

### Periodic Code Execution with the Events Handling

In the game, the periodic call of the snake movement function with the ability of events handling after a certain time interval is needed. But as was shown above, an endless loop leads to the fact, that the [OnChartEvent](https://www.mql5.com/en/docs/basis/function/events) function is not called and event handling becomes impossible.

So it's necessary to invent another way of periodic code execution.

### Using OnTimer

The MQL5 language has a special [OnTimer](https://www.mql5.com/en/docs/basis/function/events) function, that called periodically according to the predefined number of seconds. To do it, we will use the [EventSetTimer](https://www.mql5.com/en/docs/eventfunctions/eventsettimer) function.

The previous example can be rewritten as:

```
#include <ChartObjects\ChartObjectsTxtControls.mqh>
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void MyFunction()
  {
   //The code, that should be executed periodically
  }
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   CChartObjectButton *button;
   button=new CChartObjectButton;
   button.Create(0,"button",0,10,10,100,20);
   button.Description("Button text");
   EventSetTimer(1);
   return(0);
  }
void OnTimer()
{
   MyFunction();
}
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void OnChartEvent(const int id,
                  const long &lparam,
                  const double &dparam,
                  const string &sparam)
  {
   if(id==CHARTEVENT_OBJECT_CLICK && sparam=="button") Alert("Button click");
  }
```

In the [OnInit](https://www.mql5.com/en/docs/basis/function/events) function, the button created and defined a period
equal to one second, for call of the [OnTimer](https://www.mql5.com/en/docs/basis/function/events) function, The call of the function [OnTimer](https://www.mql5.com/en/docs/basis/function/events) is performed every second, the [OnTimer](https://www.mql5.com/en/docs/basis/function/events) function calls the code (MyFunction), that should be executed periodically.

Pay attention that call of the [OnTimer](https://www.mql5.com/en/docs/basis/function/events) function is multiple of seconds. To call the function after a specified number of milliseconds, the other method is needed. This method is the use of custom events.

### Using the Custom Events

The custom event is generated by [EventChartCustom](https://www.mql5.com/en/docs/eventfunctions/eventchartcustom) function,
the event ID and its parameters is defined in the input parameters of the [EventChartCustom](https://www.mql5.com/en/docs/eventfunctions/eventchartcustom) function. The number of custom defined IDs can be up to 65536 - from 0 to 65535. The MQL5 compiler adds automatically the [CHARTEVENT\_CUSTOM](https://www.mql5.com/en/docs/constants/chartconstants/enum_chartevents) constant identifier to the ID to distinguish the custom events from the other types of events. Thus, the actual range of the custom IDs is from [CHARTEVENT\_CUSTOM](https://www.mql5.com/en/docs/constants/chartconstants/enum_chartevents) to [CHARTEVENT\_CUSTOM](https://www.mql5.com/en/docs/constants/chartconstants/enum_chartevents) +65535 ( [CHARTEVENT\_CUSTOM\_LAST](https://www.mql5.com/en/docs/constants/chartconstants/enum_chartevents) ).

An example of a periodic call of MyFunction using the custom events is presented below:

```
#include <ChartObjects\ChartObjectsTxtControls.mqh>
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void MyFunction()
  {
   //The code, that should be executed periodically
   Sleep(200);
   EventChartCustom(0,0,0,0,"");
  }
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   CChartObjectButton *button;
   button=new CChartObjectButton;
   button.Create(0,"button",0,10,10,100,20);
   button.Description("Button text");
   MyFunction();
   return(0);
  }
//+------------------------------------------------------------------+
//| OnChartEvent processing function                                 |
//+------------------------------------------------------------------+
void OnChartEvent(const int id,
                  const long &lparam,
                  const double &dparam,
                  const string &sparam)
  {
   if(id==CHARTEVENT_OBJECT_CLICK && sparam=="button") Alert("Button click");
   if(id==CHARTEVENT_CUSTOM) MyFunction();
  }
```

In this example, before the MyFunction function there is a delay of 200 ms (the time of the periodic call this function) and custom event is generated. The [OnChartEvent](https://www.mql5.com/en/docs/basis/function/events) function handles all events, for the case of the custom event, it calls again the MyFunction function. Thus, the periodic call of the MyFunction function is implemented this way, and it's possible to set the call period equal to milliseconds.

### Practical Part

Let's consider example of writing a "Snake" game.

### Defining the Constants and Levels Map

Map levels in a separate includes (header) file "Snake.mqh" and is a three-dimensional array level \[6\] \[20\] \[20\].
The levels map is located in a separate header file "Snake.mqh" and represented as a game\_level\[6\]\[20\]\[20\] three-dimensional array. An each element of this array is a two-dimensional array, that contains the description of the individual level. If the value of an element is equal to 9, it's an obstacle. If the value of an array element is equal to 1,2 or 3, it's the head, body or tail of the snake respectively, that defines its initial position on the playing field. You can add new levels or modify the existing ones to the level array.

In addition, the file "Snake.mqh" contains the constants, that are used in the game. For example, by changing the SPEED\_SNAKE and MAX\_LENGTH\_SNAKE contants, you can increase/decrease the snake speed and its maximal length at each level. All constants are commented.

```
//+------------------------------------------------------------------+
//|                                                        Snake.mqh |
//|                                        Copyright Roman Martynyuk |
//|                                              http://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Roman Martynyuk"
#property link      "http://www.mql5.com"
#include <VirtualKeys.mqh>                                                    //File with keycodes
#include <Arrays\ArrayObj.mqh>                                                //File with CArrayObj class
#include <ChartObjects\ChartObjectsBmpControls.mqh>                           //File with CChartObjectBmpLabel class
#include <ChartObjects\ChartObjectsTxtControls.mqh>                           //File with CChartObjectButton and CChartObjectEdit classes
#define CRASH_NO                          0                                   //No crash
#define CRASH_OBSTACLE_OR_SNAKE           1                                   //Crash with an "Obstacle" or snake body
#define CRASH_FOOD                        2                                   //Crash with a "Food""
#define DIRECTION_LEFT                    0                                   //Left
#define DIRECTION_UP                      1                                   //Up
#define DIRECTION_RIGHT                   2                                   //Right
#define DIRECTION_DOWN                    3                                   //Down
#define COUNT_COLUMNS                     ArrayRange(game_level,2)            //Number of columns of playing field
#define COUNT_ROWS                        ArrayRange(game_level,1)            //Number of rows of playing field
#define COUNT_LEVELS                      ArrayRange(game_level,0)             //Number of levels
#define START_POS_X                       0                                   //Starting X position of the game
#define START_POS_Y                       0                                   //Starting Y position of the game
#define SQUARE_WIDTH                      20                                  //Square (cell) width (in pixels)
#define SQUARE_HEIGHT                     20                                  //Square (cell) height (in pixels)
#define IMG_FILE_NAME_SQUARE              "\\Images\\Games\\Snake\\square.bmp"          //Path to the "Square" image
#define IMG_FILE_NAME_OBSTACLE            "\\Images\\Games\\Snake\\obstacle.bmp"        //Path to the "Obstacle" image
#define IMG_FILE_NAME_SNAKE_HEAD_LEFT     "\\Images\\Games\\Snake\\head_left.bmp"       //Path to the snake's head (left) image
#define IMG_FILE_NAME_SNAKE_HEAD_UP       "\\Images\\Games\\Snake\\head_up.bmp"         //Path to the snake's head (up) image
#define IMG_FILE_NAME_SNAKE_HEAD_RIGHT    "\\Images\\Games\\Snake\\head_right.bmp"      //Path to the snake's head (right) image
#define IMG_FILE_NAME_SNAKE_HEAD_DOWN     "\\Images\\Games\\Snake\\head_down.bmp"       //Path to the snake's head (down) image
#define IMG_FILE_NAME_SNAKE_BODY          "\\Images\\Games\\Snake\\body.bmp"            //Path to the snake's body image
#define IMG_FILE_NAME_SNAKE_TAIL_LEFT     "\\Images\\Games\\Snake\\tail_left.bmp"       //Path to the snake's tail (left) image
#define IMG_FILE_NAME_SNAKE_TAIL_UP       "\\Images\\Games\\Snake\\tail_up.bmp"         //Path to the snake's tail (up) image
#define IMG_FILE_NAME_SNAKE_TAIL_RIGHT    "\\Images\\Games\\Snake\\tail_right.bmp"      //Path to the snake's tail (right) image
#define IMG_FILE_NAME_SNAKE_TAIL_DOWN     "Games\\Snake\\tail_down.bmp"       //Path to the snake's tail (down) image
#define IMG_FILE_NAME_FOOD                "Games\\Snake\food.bmp"             //Path to the "Food" image
#define SQUARE_BMP_LABEL_NAME             "snake_square_%u_%u"                //Name of the "Square" graphic label
#define OBSTACLE_BMP_LABEL_NAME           "snake_obstacle_%u_%u"              //Name of the "Obstacle" graphic label
#define SNAKE_ELEMENT_BMP_LABEL_NAME      "snake_element_%u"                  //Name of the "Snake" graphic label
#define FOOD_BMP_LABEL_NAME               "snake_food_%u"                     //Name of the "Food" graphic label
#define LEVEL_EDIT_NAME                   "snake_level_edit"                  //Name of the "Level" edit
#define LEVEL_EDIT_TEXT                   "Level: %u of %u"                   //Text of the "Level" edit
#define FOOD_LEFT_OVER_EDIT_NAME          "snake_food_available_edit"         //Name of the "Food left" edit
#define FOOD_LEFT_OVER_EDIT_TEXT          "Food left over: %u"                //Text of the "Food left" edit
#define LIVES_EDIT_NAME                   "snake_lives_edit"                  //Name of the "Lives" edit
#define LIVES_EDIT_TEXT                   "Lives: %u"                         //Text of the "Lives" edit
#define START_GAME_BUTTON_NAME            "snake_start_game_button"           //Name of the "Start" button
#define START_GAME_BUTTON_TEXT            "Start"                             //Text of the "Start" button
#define PAUSE_GAME_BUTTON_NAME            "snake_pause_game_button"           //Name of the "Pause" button
#define PAUSE_GAME_BUTTON_TEXT            "Pause"                             //Text of the "Pause" button
#define STOP_GAME_BUTTON_NAME             "snake_stop_game_button"            //Name of the "Stop" button
#define STOP_GAME_BUTTON_TEXT             "Stop"                              //Text of the "Stop" button
#define CONTROL_WIDTH                     (COUNT_COLUMNS*(SQUARE_WIDTH-1)+1)/3//Control Panel Width (1/3 of playing field width)
#define CONTROL_HEIGHT                    40                                  //Control Panel Height
#define CONTROL_BACKGROUND                C'240,240,240'                      //Color of Control Panel buttons
#define CONTROL_COLOR                     Black                               //Text Color of Control Panel Buttons
#define STATUS_WIDTH                      (COUNT_COLUMNS*(SQUARE_WIDTH-1)+1)/3//Status Panel Width (1/3 of playing field width)
#define STATUS_HEIGHT                     40                                  //Status Panel Height
#define STATUS_BACKGROUND                 LemonChiffon                        //Status Panel Background Color
#define STATUS_COLOR                      Black                               //Status Panel Text Color
#define HEADER_BUTTON_NAME                "snake_header_button"               //Name of the "Header" button
#define HEADER_BUTTON_TEXT                "Snake"                             //Text of the "Header" button
#define HEADER_WIDTH                      COUNT_COLUMNS*(SQUARE_WIDTH-1)+1    //Width of the "Header" button (playing field width)
#define HEADER_HEIGHT                     40                                  //Height of the "Header" button
#define HEADER_BACKGROUND                 BurlyWood                           //Header Background Color
#define HEADER_COLOR                      Black                               //Headet Text Color
#define COUNT_FOOD                        3                                   //Number of "Foods" at playing field
#define LIVES_SNAKE                       5                                   //Number of snake lives at each level
#define SPEED_SNAKE                       100                                 //Snake Speed (in milliseconds)
#define MAX_LENGTH_SNAKE                  15                                  //Maximal Snake Length
#define MAX_LEVEL                         COUNT_LEVELS-1                      //Maximal Level
int game_level[][20][20]=
  {
     {
        {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,3,2,1,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}
     }
      ,
     {
        {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,3,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,1,2,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0,0,0,0,9,9,9,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0,0,0,0,9,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0,0,0,0,9,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0,0,0,0,9,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0,0,0,0,9,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,9,9,9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}
     }
      ,
     {
        {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,9,9,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,9,9,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,9,9,1,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,3,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0,0,0,9,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0,0,0,9,9,9,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0,0,0,9,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0,0,0,9,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,9,9,9,9,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,9,9,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}
     }
      ,
     {
        {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,9,9,9,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0},
        {0,0,0,0,0,9,9,0,0,0,0,0,3,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0,0,9,9,9,9,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,9,9,9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,9,9,9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}
     }
      ,
     {
        {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,1,2,3,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0,9,9,9,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0,0,0,9,9,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0,0,0,9,9,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,9,9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,9,9,9,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,9,0,0,0,0,0,0,0,9,9,9,9,0},
        {0,0,0,0,0,0,0,9,0,0,0,0,0,0,0,0,9,0,0,0},
        {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}
     }
      ,
     {
        {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,9,9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,9,9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0,0,0,9,0,0,0,0,0,0,0,0},
        {0,1,2,3,0,0,0,0,0,0,0,9,9,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0,0,0,9,9,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,9,0,9,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,9,9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,9,9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,9,9,9,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,9,9,0,0,0,0,0,0,9,9,9,9,0},
        {0,0,0,0,0,0,0,9,0,0,0,0,0,0,0,0,9,0,0,0},
        {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0,0,0,0,0,9,9,0,0,0,0,0}
     }
  };
//+------------------------------------------------------------------+
```

Note, the definition of constant [#define](https://www.mql5.com/en/docs/basis/preprosessor/constant) SQUARE\_BMP\_LABEL\_NAME "snake\_square\_% u\_% U". We will create the playing field. An each cell of the playing field is a bitmap label, it should have a unique name. The name of a cell is defined by this constant, the [format specification](https://www.mql5.com/en/docs/common/printformat) of a cell name is %u, it means the insigned integer.

If you will specify the name when creating the BmpLabel like: [StringFormat](https://www.mql5.com/en/docs/convert/stringformat) (SQUARE\_BMP\_LABEL\_NAME, 1,0), the name will be be equal to "snake\_square\_1\_0".

### The Classes

There are two custom classes have been developed for the game, they are
located in the "Snake.mq5" file.

The ChartFieldElement class:

```
//+------------------------------------------------------------------+
//| CChartFieldElement class                                         |
//+------------------------------------------------------------------+
class CChartFieldElement:public CChartObjectBmpLabel
  {
private:
   int               pos_x,pos_y;
public:
   int               GetPosX(){return pos_x;}
   int               GetPosY(){return pos_y;}
   //setting position (pos_x,pos_y) in internal coordinates
   void              SetPos(int val_pos_x,int val_pos_y)
     {
      pos_x=(val_pos_x==-1)?COUNT_COLUMNS-1:((val_pos_x==COUNT_COLUMNS)?0:val_pos_x);
      pos_y=(val_pos_y==-1)?COUNT_ROWS-1:((val_pos_y==COUNT_ROWS)?0:val_pos_y);
     }
   //conversion of internal coordinates to absolute and object movement on the chart
   void              Move(int start_pos_x,int start_pos_y)
     {
      X_Distance(start_pos_x+pos_x*SQUARE_WIDTH-pos_x+(SQUARE_WIDTH-X_Size())/2);
      Y_Distance(start_pos_y+pos_y*SQUARE_HEIGHT-pos_y+(SQUARE_HEIGHT-Y_Size())/2);
     }
  };
```

The CChartFiledElement class inherits the [CChartObjectBmpLabel](https://www.mql5.com/en/docs/standardlibrary/chart_object_classes/obj_controls/cchartobjectbmplabel) class, thus extends it. All the playing field, such as cell barrier, head, body and tail of the snake, and the "food" are the objects of this class. The _pos\_x_ and _pos\_y_ properties are relative coordinates of elements on the playing field, that is the row and column indexes of the element. The SetPos method sets these coordinates. The Move method converts the relative coordinates to the distances along the X and Y axes in pixels and moves the element. To do it, it calls the [X\_Distance](https://www.mql5.com/en/docs/standardlibrary/chart_object_classes/obj_controls/cchartobjectbmplabel/cchartobjectbmplabelx_distance) and [YDistance](https://www.mql5.com/en/docs/standardlibrary/chart_object_classes/obj_controls/cchartobjectbmplabel/cchartobjectbmplabely_distance) methods of the [CChartObjectBmpLabel](https://www.mql5.com/en/docs/standardlibrary/chart_object_classes/obj_controls/cchartobjectbmplabel) class.

The CSnakeGame class:

```
//+------------------------------------------------------------------+
//| CSnakeGame class                                                 |
//+------------------------------------------------------------------+
class CSnakeGame
  {
private:
   CArrayObj        *square_obj_arr;                     //Array of playing field cells
   CArrayObj        *control_panel_obj_arr;              //Array of control panel buttons
   CArrayObj        *status_panel_obj_arr;               //Array of control panel edits
   CArrayObj        *obstacle_obj_arr;                   //Array of an obstacles
   CArrayObj        *food_obj_arr;                       //Array of "Food"
   CArrayObj        *snake_element_obj_arr;              //Array of snake elements
   CChartObjectButton *header;                           //Header
   int               direction;                          //Snake movement direction
   int               current_lives;                      //Number of snake Lives
   int               current_level;                      //Level
   int               header_left;                        //Left position of a header (X)
   int               header_top;                         //Top position of a header (Y)
public:
   //class constructor
   void              CSnakeGame()
     {
      current_lives=LIVES_SNAKE;
      current_level=0;
      header_left=START_POS_X;
      header_top=START_POS_Y;
     }
   //method for definition of header_left and header_top fields
   void              SetHeaderPos(int val_header_left,int val_header_top)
     {
      header_left=val_header_left;
      header_top=val_header_top;
     };
   //Get/Set direction methods
   void              SetDirection(int d){direction=d;}
   int               GetDirection(){return direction;}
   //Header creation and deletion methods
   void              CreateHeader();
   void              DeleteHeader();
   //Playing field creation, movement and deletion methods
   void              CreateField();
   void              FieldMoveOnChart();
   void              DeleteField();
   //Obstacle creation, movement and deletion methods
   void              CreateObstacle();
   void              ObstacleMoveOnChart();
   void              DeleteObstacle();
   //Snake creation, movement and deletion methods
   void              CreateSnake();
   void              SnakeMoveOnChart();
   void              SnakeMoveOnField();                 //snake movement on the playing field
   void              SetTrueSnake();                     //setting the images of the current snake's head and tail
   int               Check();                            //check for the collision with the playing field elements
   void              DeleteSnake();
   //Food creation, movement and deletion methods
   void              CreateFood();
   void              FoodMoveOnChart();
   void              FoodMoveOnField(int food_num);
   void              DeleteFood();
   //Status panel creation, movement and deletion methods
   void              CreateControlPanel();
   void              ControlPanelMoveOnChart();
   void              DeleteControlPanel();
   //Control panel creation, movement and deletion methods
   void              CreateStatusPanel();
   void              StatusPanelMoveOnChart();
   void              DeleteStatusPanel();
   //Move all elements on the chart
   void              AllMoveOnChart();
   //Game initialization
   void              Init();
   //Game deinitialization
   void              Deinit();
   //Game control methods
   void              StartGame();
   void              PauseGame();
   void              StopGame();
   void              ResetGame();
   void              NextLevel();
  };
```

The CSnakeGame is the main class of the game, it contains the fields and methods of creation, movement and removal of the game elements. One can see, at the beginning of the class description, the fields for the
organization of dynamic arrays of pointers of game elements are declared. For example, the pointers of the snake elements are stored into the snake\_element\_obj\_arr field. The zeroth array index of the snake\_element\_obj\_arr array will be a head of snake, and the last - its tail. Knowing it, you can easily manipulate the snake on the playing field.

Let's consider all methods of the CSnakeGame class. The methods are implemented on the basis of the theory, presented in the "Theory" chapter of this article.

### The game header

```
//+------------------------------------------------------------------+
//| Header creation method                                           |
//+------------------------------------------------------------------+
void CSnakeGame::CreateHeader(void)
  {
   //creating a new object of CChartObjectButton class and specifying the properties of header of CSnakeGame class
   header=new CChartObjectButton;
   header.Create(0,HEADER_BUTTON_NAME,0,header_left,header_top,HEADER_WIDTH,HEADER_HEIGHT);
   header.BackColor(HEADER_BACKGROUND);
   header.Color(HEADER_COLOR);
   header.Description(HEADER_BUTTON_TEXT);
   //the header is selectable
   header.Selectable(true);
  }
//+------------------------------------------------------------------+
//| Header deletion method                                           |
//+------------------------------------------------------------------+
void CSnakeGame::DeleteHeader(void)
  {
   delete header;
  }
```

### The playing field

```
//+------------------------------------------------------------------+
//| Playing Field creation method                                    |
//+------------------------------------------------------------------+
void CSnakeGame::CreateField()
  {
   int i,j;
   CChartFieldElement *square_obj;
   //creating an object of CArrayObj class and assign the square_obj_arr properties of CSnakeGame class
   square_obj_arr=new CArrayObj();
   for(i=0;i<COUNT_ROWS;i++)
      for(j=0;j<COUNT_COLUMNS;j++)
        {
         square_obj=new CChartFieldElement();
         square_obj.Create(0,StringFormat(SQUARE_BMP_LABEL_NAME,i,j),0,0,0);
         square_obj.BmpFileOn(IMG_FILE_NAME_SQUARE);
         //specifying the internal coordinates of the cell
         square_obj.SetPos(j,i);
         square_obj_arr.Add(square_obj);
        }
   //moving the playing field cells
   FieldMoveOnChart();
   ChartRedraw();
  }
//+------------------------------------------------------------------+
//| The movement of playing field cells on the chart                 |
//+------------------------------------------------------------------+
void CSnakeGame::FieldMoveOnChart()
  {
   CChartFieldElement *square_obj;
   int i;
   i=0;
   while((square_obj=square_obj_arr.At(i))!=NULL)
     {
      square_obj.Move(header_left,header_top+HEADER_HEIGHT);
      i++;
     }
   ChartRedraw();
  }
//+------------------------------------------------------------------+
//| Deletion of a playing field                                      |
//+------------------------------------------------------------------+
void CSnakeGame::DeleteField()
  {
   delete square_obj_arr;
   ChartRedraw();
  }
```

### The Obstacles

```
//+------------------------------------------------------------------+
//| Creation of the obstacles                                        |
//+------------------------------------------------------------------+
void CSnakeGame::CreateObstacle()
  {
   int i,j;
   CChartFieldElement *obstacle_obj;
   //creating an object of CArrayObj class and assign the obstacle_obj_arr properties of CSnakeGame class
   obstacle_obj_arr=new CArrayObj();
   for(i=0;i<COUNT_ROWS;i++)
      for(j=0;j<COUNT_COLUMNS;j++)
         if(game_level[current_level][i][j]==9)
           {
            obstacle_obj=new CChartFieldElement();
            obstacle_obj.Create(0,StringFormat(OBSTACLE_BMP_LABEL_NAME,i,j),0,0,0);
            obstacle_obj.BmpFileOn(IMG_FILE_NAME_OBSTACLE);
            //specifying the internal coordinates of the obstacle
            obstacle_obj.SetPos(j,i);
            obstacle_obj_arr.Add(obstacle_obj);
           }
   //moving the obstacle on the chart
   ObstacleMoveOnChart();
   ChartRedraw();
  }
//+------------------------------------------------------------------+
//| Obstacle movement method                                         |
//+------------------------------------------------------------------+
void CSnakeGame::ObstacleMoveOnChart()
  {
   CChartFieldElement *obstacle_obj;
   int i;
   i=0;
   while((obstacle_obj=obstacle_obj_arr.At(i))!=NULL)
     {
      obstacle_obj.Move(header_left,header_top+HEADER_HEIGHT);
      i++;
     }
   ChartRedraw();
  }
//+------------------------------------------------------------------+
//| Obstacle deletion method                                         |
//+------------------------------------------------------------------+
void CSnakeGame::DeleteObstacle()
  {
   delete obstacle_obj_arr;
   ChartRedraw();
  }
```

### The Snake

```
//+------------------------------------------------------------------+
//| Snake creation method                                            |
//+------------------------------------------------------------------+
void CSnakeGame::CreateSnake()
  {
   int i,j;
   CChartFieldElement *snake_element_obj,*snake_arr[];
   ArrayResize(snake_arr,3);
   //creating an object of CArrayObj class and assign it to the snake_element_obj_arr properties of CSnakeGame class
   snake_element_obj_arr=new CArrayObj();
   for(i=0;i<COUNT_COLUMNS;i++)
      for(j=0;j<COUNT_ROWS;j++)
         if(game_level[current_level][i][j]==1 || game_level[current_level][i][j]==2 || game_level[current_level][i][j]==3)
           {
            snake_element_obj=new CChartFieldElement();
            snake_element_obj.Create(0,StringFormat(SNAKE_ELEMENT_BMP_LABEL_NAME,game_level[current_level][i][j]),0,0,0);
            snake_element_obj.BmpFileOn(IMG_FILE_NAME_SNAKE_BODY);
            //specifying the internal coordinates of the snake element
            snake_element_obj.SetPos(j,i);
            snake_arr[game_level[current_level][i][j]-1]=snake_element_obj;
           }
   snake_element_obj_arr.Add(snake_arr[0]);
   snake_element_obj_arr.Add(snake_arr[1]);
   snake_element_obj_arr.Add(snake_arr[2]);
   //moving the snake on the chart
   SnakeMoveOnChart();
   //setting the correct images of the snake's head and tail
   SetTrueSnake();
   ChartRedraw();
  }
//+------------------------------------------------------------------+
//| Snake movement on the chart                                      |
//+------------------------------------------------------------------+
void CSnakeGame::SnakeMoveOnChart()
  {
   CChartFieldElement *snake_element_obj;
   int i;
   i=0;
   while((snake_element_obj=snake_element_obj_arr.At(i))!=NULL)
     {
      snake_element_obj.Move(header_left,header_top+HEADER_HEIGHT);
      i++;
     }
   ChartRedraw();
  }
//+------------------------------------------------------------------+
//| Snake movement on the playing field                              |
//+------------------------------------------------------------------+
void CSnakeGame::SnakeMoveOnField()
  {
   int prev_x,prev_y,next_x,next_y,check;
   CChartFieldElement *snake_head_obj,*snake_body_obj,*snake_tail_obj;
   //getting the snake's head from the array
   snake_head_obj=snake_element_obj_arr.At(0);
   //saving the coordinates of a head
   prev_x=snake_head_obj.GetPosX();
   prev_y=snake_head_obj.GetPosY();
   //setting the new internal coordinates for the head depending on the movement direction
   switch(direction)
     {
      case DIRECTION_LEFT:snake_head_obj.SetPos(prev_x-1,prev_y);break;
      case DIRECTION_UP:snake_head_obj.SetPos(prev_x,prev_y-1);break;
      case DIRECTION_RIGHT:snake_head_obj.SetPos(prev_x+1,prev_y);break;
      case DIRECTION_DOWN:snake_head_obj.SetPos(prev_x,prev_y+1);break;
     }
   //moving the snake's head
   snake_head_obj.Move(header_left,header_top+HEADER_HEIGHT);
   //check for the snake's head collision with the other playing field elements (obstacle, snake body, food)
   check=Check();
   //getting the last element of the snake's body
   snake_body_obj=snake_element_obj_arr.Detach(snake_element_obj_arr.Total()-2);
   //saving coordinates of the snake's body
   next_x=snake_body_obj.GetPosX();
   next_y=snake_body_obj.GetPosY();
   //moving the snake's body to the previous head's position
   snake_body_obj.SetPos(prev_x,prev_y);
   snake_body_obj.Move(header_left,header_top+HEADER_HEIGHT);
   //saving the previous position of the snake's body
   prev_x=next_x;
   prev_y=next_y;
   //inserting the snake's body to the first position of the snake_element_obj_arr array
   snake_element_obj_arr.Insert(snake_body_obj,1);
   //if the snake's head has collided with the "Food"
   if(check>=CRASH_FOOD)
     {
      //creating new element of the snake's body
      snake_body_obj=new CChartFieldElement();
      snake_body_obj.Create(0,StringFormat(SNAKE_ELEMENT_BMP_LABEL_NAME,snake_element_obj_arr.Total()+1),0,0,0);
      snake_body_obj.BmpFileOn(IMG_FILE_NAME_SNAKE_BODY);
      //moving the body element to the end of the snake before the tail
      snake_body_obj.SetPos(prev_x,prev_y);
      snake_body_obj.Move(header_left,header_top+HEADER_HEIGHT);
      //adding the body to the penultimate position of the snake_element_obj_arr array
      snake_element_obj_arr.Insert(snake_body_obj,snake_element_obj_arr.Total()-1);
      //if snake's body isn't equal to the maximal snake length
      if(snake_element_obj_arr.Total()!=MAX_LENGTH_SNAKE)
        {
         //moving the eaten element on the new place on the playing field
         FoodMoveOnField(check-CRASH_FOOD);
        }
      //else we generate the custom event, that indicates that current snake length is the maximal possible
      else EventChartCustom(0,2,0,0,"");
     }
   //else if there isn't collision with the food, moving the tail to the position of the snake's body
   else
     {
      snake_tail_obj=snake_element_obj_arr.At(snake_element_obj_arr.Total()-1);
      snake_tail_obj.SetPos(prev_x,prev_y);
      snake_tail_obj.Move(header_left,header_top+HEADER_HEIGHT);
     }
   //setting the correct images for the head and tail
   SetTrueSnake();
   ChartRedraw();
   //generating the custom event for periodic call of this snake movement function
   EventChartCustom(0,0,0,0,"");
   Sleep(SPEED_SNAKE);
  }
//+------------------------------------------------------------------+
//| Setting the correct images for the snake's head and tail         |
//+------------------------------------------------------------------+
void CSnakeGame::SetTrueSnake()
  {
   CChartFieldElement *snake_head,*snake_body,*snake_tail;
   int total,x1,x2,y1,y2;
   total=snake_element_obj_arr.Total();
   //getting the snake's head
   snake_head=snake_element_obj_arr.At(0);
   //saving position of a head
   x1=snake_head.GetPosX();
   y1=snake_head.GetPosY();
   //getting the first element of the snake's body
   snake_body=snake_element_obj_arr.At(1);
   //saving coordinates of the body
   x2=snake_body.GetPosX();
   y2=snake_body.GetPosY();
   //choosing the file with an image depening on the position of the head and the first body element relative to each other
   //setting the snake's movement direction depending on the snake's head direction
   if(x1-x2==1 || x1-x2==-(COUNT_COLUMNS-1))
     {
      snake_head.BmpFileOn(IMG_FILE_NAME_SNAKE_HEAD_RIGHT);
      direction=DIRECTION_RIGHT;
     }
   else if(y1-y2==1 || y1-y2==-(COUNT_ROWS-1))
     {
      snake_head.BmpFileOn(IMG_FILE_NAME_SNAKE_HEAD_DOWN);
      direction=DIRECTION_DOWN;
     }
   else if(x1-x2==-1 || x1-x2==COUNT_COLUMNS-1)
     {
      snake_head.BmpFileOn(IMG_FILE_NAME_SNAKE_HEAD_LEFT);
      direction=DIRECTION_LEFT;
     }
   else
     {
      snake_head.BmpFileOn(IMG_FILE_NAME_SNAKE_HEAD_UP);
      direction=DIRECTION_UP;
     }
   //getting the last element of the snake's body
   snake_body=snake_element_obj_arr.At(total-2);
   //saving coordinates of the body
   x1=snake_body.GetPosX();
   y1=snake_body.GetPosY();
   //getting the tail of the snake
   snake_tail=snake_element_obj_arr.At(total-1);
   //saving coordinates of the tail
   x2=snake_tail.GetPosX();
   y2=snake_tail.GetPosY();

   //choosing the file with an image depening on the position of the tail and the last body element relative to each other
   if(x1-x2==1 || x1-x2==-(COUNT_COLUMNS-1))    snake_tail.BmpFileOn(IMG_FILE_NAME_SNAKE_TAIL_RIGHT);
   else if(y1-y2==1 || y1-y2==-(COUNT_ROWS-1))  snake_tail.BmpFileOn(IMG_FILE_NAME_SNAKE_TAIL_DOWN);
   else if(x1-x2==-1 || x1-x2==COUNT_COLUMNS-1) snake_tail.BmpFileOn(IMG_FILE_NAME_SNAKE_TAIL_LEFT);
   else                                         snake_tail.BmpFileOn(IMG_FILE_NAME_SNAKE_TAIL_UP);
  }
//+------------------------------------------------------------------+
//| Check for snake's head collision with the playing field elements |
//+------------------------------------------------------------------+
int CSnakeGame::Check()
  {
   int i;
   CChartFieldElement *snake_head_obj,*snake_element_obj,*obstacle_obj,*food_obj;
   //getting the snake's head
   snake_head_obj=snake_element_obj_arr.At(0);
   i=0;
   //check for the head's collision with the obstacle
   while((obstacle_obj=obstacle_obj_arr.At(i))!=NULL)
     {
      if(snake_head_obj.GetPosX()==obstacle_obj.GetPosX() && snake_head_obj.GetPosY()==obstacle_obj.GetPosY())
        {
         EventChartCustom(0,1,0,0,"");
         return CRASH_OBSTACLE_OR_SNAKE;
        }
      i++;
     }
   i=0;
   //check for the collision of head with the food
   while((food_obj=food_obj_arr.At(i))!=NULL)
     {
      if(snake_head_obj.GetPosX()==food_obj.GetPosX() && snake_head_obj.GetPosY()==food_obj.GetPosY())
        {
         //hiding the food
         food_obj.Background(true);
         return(CRASH_FOOD+i);
        }
      i++;
     }
   i=3;
   //check for the collision of a head with the body and tail
   while((snake_element_obj=snake_element_obj_arr.At(i))!=NULL)
     {
      //we don't check for the collision with the last snake's element, because it hasn't been moved yet
      if(snake_element_obj_arr.At(i+1)==NULL)
         break;
      if(snake_head_obj.GetPosX()==snake_element_obj.GetPosX() && snake_head_obj.GetPosY()==snake_element_obj.GetPosY())
        {
         EventChartCustom(0,1,0,0,"");
         //hiding the snake's element we have collided
         snake_element_obj.Background(true);
         return CRASH_OBSTACLE_OR_SNAKE;
        }
      i++;
     }
   return CRASH_NO;
  }
//+------------------------------------------------------------------+
//| Snake deletion                                                   |
//+------------------------------------------------------------------+
void CSnakeGame::DeleteSnake()
  {
   delete snake_element_obj_arr;
   ChartRedraw();
  }
```

After the snake's head is moved, it checked for the collision by Check() function, which returns the collision identifier.

The SetTrueSnake() function is used to specify the correct drawing of snake's head and tail , depending on the position of their neighboring elements.

### The Food for the Snake

```
//+------------------------------------------------------------------+
//| Food creation                                                    |
//+------------------------------------------------------------------+
void CSnakeGame::CreateFood()
  {
   int i;
   CChartFieldElement *food_obj;
   MathSrand(uint(TimeLocal()));
   //creating an object of CArrayObj class and assign it to the food_obj_arr properties of CSnakeGame class
   food_obj_arr=new CArrayObj();
   i=0;
   while(i<COUNT_FOOD)
     {
      //creating the food
      food_obj=new CChartFieldElement;
      food_obj.Create(0,StringFormat(FOOD_BMP_LABEL_NAME,i),0,0,0);
      food_obj.BmpFileOn(IMG_FILE_NAME_FOOD);
      food_obj_arr.Add(food_obj);
      //setting the field coordinates on the field and moving it on the playing field
      FoodMoveOnField(i);
      i++;
     }
  }
//+------------------------------------------------------------------+
//| Food movement method                                             |
//+------------------------------------------------------------------+
void CSnakeGame::FoodMoveOnChart()
  {
   CChartFieldElement *food_obj;
   int i;
   i=0;
   while((food_obj=food_obj_arr.At(i))!=NULL)
     {
      food_obj.Move(header_left,header_top+HEADER_HEIGHT);
      i++;
     }
   ChartRedraw();
  }
//+---------------------------------------------------------------------------+
//| A method to set coordinates of a food and to move it on the playing field |
//+---------------------------------------------------------------------------+
void CSnakeGame::FoodMoveOnField(int food_num)
  {
   int i,j,k,n,m;
   CChartFieldElement *snake_element_obj,*food_obj;
   CChartObjectEdit *edit_obj;
   //setting a new value for "Foods left" on the status panel
   edit_obj=status_panel_obj_arr.At(1);
   edit_obj.Description(StringFormat(spaces2+FOOD_LEFT_OVER_EDIT_TEXT,MAX_LENGTH_SNAKE-snake_element_obj_arr.Total()));
   bool b;
   b=false;
   k=0;
   //generating randomly the food coordinates until the we get the free cells
   while(true)
     {
      //generating a row number
      i=(int)(MathRand()/32767.0*(COUNT_ROWS-1));
      //generating a column number
      j=(int)(MathRand()/32767.0*(COUNT_COLUMNS-1));
      n=0;
      //check, if there are any elements of the snake
      while((snake_element_obj=snake_element_obj_arr.At(n))!=NULL)
        {
         if(j!=snake_element_obj.GetPosX() && i!=snake_element_obj.GetPosY())
            b=true;
         else
           {
            b=false;
            break;
           }
         n++;
        }
      //checking for the other food presence
      if(b==true)
        {
         n=0;
         while((food_obj=food_obj_arr.At(n))!=NULL)
           {
            if(j!=food_obj.GetPosX() && i!=food_obj.GetPosY())
               b=true;
            else
              {
               b=false;
               break;
              }
            n++;
           }
        }
      //checking for the presence of the obstacle
      if(b==true && game_level[current_level][i][j]!=9) break;
      k++;
     }
   food_obj=food_obj_arr.At(food_num);
   //show food
   food_obj.Background(false);
   //setting new coordinates
   food_obj.SetPos(j,i);
   //moving the food
   food_obj.Move(header_left,header_top+HEADER_HEIGHT);
   ChartRedraw();
  }
//+------------------------------------------------------------------+
//| Food deletion                                                    |
//+------------------------------------------------------------------+
void CSnakeGame::DeleteFood()
  {
   delete food_obj_arr;
   ChartRedraw();
  }
```

Location of food on the playing field is set at random, provided that the cell field, which will be located the food does not contain other elements.

### The Status Panel

```
//+------------------------------------------------------------------+
//| Status Panel Creation                                            |
//+------------------------------------------------------------------+
void CSnakeGame::CreateStatusPanel()
  {
   CChartObjectEdit *edit_obj;
   //creating an object of CArrayObj class and assign it to the status_panel_obj_arr properties of CSnakeGame class
   status_panel_obj_arr=new CArrayObj();
   //creating the "Level" edit
   edit_obj=new CChartObjectEdit;
   edit_obj.Create(0,LEVEL_EDIT_NAME,0,0,0,CONTROL_WIDTH,CONTROL_HEIGHT);
   edit_obj.BackColor(STATUS_BACKGROUND);
   edit_obj.Color(STATUS_COLOR);
   edit_obj.Description(StringFormat(spaces6+LEVEL_EDIT_TEXT,current_level,MAX_LEVEL));
   edit_obj.Selectable(false);
   edit_obj.ReadOnly(true);
   status_panel_obj_arr.Add(edit_obj);
   //creating the "Food left over" edit
   edit_obj=new CChartObjectEdit;
   edit_obj.Create(0,FOOD_LEFT_OVER_EDIT_NAME,0,0,0,CONTROL_WIDTH,CONTROL_HEIGHT);
   edit_obj.BackColor(STATUS_BACKGROUND);
   edit_obj.Color(STATUS_COLOR);
   edit_obj.Description(StringFormat(spaces2+FOOD_LEFT_OVER_EDIT_TEXT,MAX_LENGTH_SNAKE-3));
   edit_obj.Selectable(false);
   edit_obj.ReadOnly(true);
   status_panel_obj_arr.Add(edit_obj);
   //creating the "Lives" edit
   edit_obj=new CChartObjectEdit;
   edit_obj.Create(0,LIVES_EDIT_NAME,0,0,0,CONTROL_WIDTH,CONTROL_HEIGHT);
   edit_obj.BackColor(STATUS_BACKGROUND);
   edit_obj.Color(STATUS_COLOR);
   edit_obj.Description(StringFormat(spaces8+LIVES_EDIT_TEXT,current_lives));
   edit_obj.Selectable(false);
   edit_obj.ReadOnly(true);
   status_panel_obj_arr.Add(edit_obj);
   //moving the status panel
   StatusPanelMoveOnChart();
   ChartRedraw();
  }
//+------------------------------------------------------------------+
//| Status Panel movement method                                     |
//+------------------------------------------------------------------+
void CSnakeGame::StatusPanelMoveOnChart()
  {
   CChartObjectEdit *edit_obj;
   int x,y,i;
   x=header_left;
   y=header_top+HEADER_HEIGHT+COUNT_ROWS*(SQUARE_HEIGHT-1)+1;
   i=0;
   while((edit_obj=status_panel_obj_arr.At(i))!=NULL)
     {
      edit_obj.X_Distance(x+i*CONTROL_WIDTH);
      edit_obj.Y_Distance(y);
      i++;
     }
   ChartRedraw();
  }
//+------------------------------------------------------------------+
//| Status Panel deletion method                                     |
//+------------------------------------------------------------------+
void CSnakeGame::DeleteStatusPanel()
  {
   delete status_panel_obj_arr;
   ChartRedraw();
  }
```

### The Control Panel

```
//+------------------------------------------------------------------+
//| Control Panel creation method                                    |
//+------------------------------------------------------------------+
void CSnakeGame::CreateControlPanel()
  {
   CChartObjectButton *button_obj;
   //creating an object of CArrayObj class and assign it to the control_panel_obj_arr properties of CSnakeGame class
   control_panel_obj_arr=new CArrayObj();
   //creating the "Start" button
   button_obj=new CChartObjectButton;
   button_obj.Create(0,START_GAME_BUTTON_NAME,0,0,0,CONTROL_WIDTH,CONTROL_HEIGHT);
   button_obj.BackColor(CONTROL_BACKGROUND);
   button_obj.Color(CONTROL_COLOR);
   button_obj.Description(START_GAME_BUTTON_TEXT);
   button_obj.Selectable(false);
   control_panel_obj_arr.Add(button_obj);
   //creating the "Pause" button
   button_obj=new CChartObjectButton;
   button_obj.Create(0,PAUSE_GAME_BUTTON_NAME,0,0,0,CONTROL_WIDTH,CONTROL_HEIGHT);
   button_obj.BackColor(CONTROL_BACKGROUND);
   button_obj.Color(CONTROL_COLOR);
   button_obj.Description(PAUSE_GAME_BUTTON_TEXT);
   button_obj.Selectable(false);
   control_panel_obj_arr.Add(button_obj);
   //creating the "Stop" button
   button_obj=new CChartObjectButton;
   button_obj.Create(0,STOP_GAME_BUTTON_NAME,0,0,0,CONTROL_WIDTH,CONTROL_HEIGHT);
   button_obj.BackColor(CONTROL_BACKGROUND);
   button_obj.Color(CONTROL_COLOR);
   button_obj.Description(STOP_GAME_BUTTON_TEXT);
   button_obj.Selectable(false);
   control_panel_obj_arr.Add(button_obj);
   //moving the control panel
   ControlPanelMoveOnChart();
   ChartRedraw();
  }
//+------------------------------------------------------------------+
//| Control Panel movement method                                    |
//+------------------------------------------------------------------+
void CSnakeGame::ControlPanelMoveOnChart()
  {
   CChartObjectButton *button_obj;
   int x,y,i;
   x=header_left;
   y=header_top+HEADER_HEIGHT+COUNT_ROWS*(SQUARE_HEIGHT-1)+1;
   i=0;
   while((button_obj=control_panel_obj_arr.At(i))!=NULL)
     {
      button_obj.X_Distance(x+i*CONTROL_WIDTH);
      button_obj.Y_Distance(y+CONTROL_HEIGHT);
      i++;
     }
   ChartRedraw();
  }
//+------------------------------------------------------------------+
//| Control Panel deletion method                                    |
//+------------------------------------------------------------------+
void CSnakeGame::DeleteControlPanel()
  {
   delete control_panel_obj_arr;
   ChartRedraw();
  }
```

### The Game Initialization, Deinitialization and Game Elements Movement

```
//+------------------------------------------------------------------+
//| Game elements movement method                                    |
//+------------------------------------------------------------------+
void  CSnakeGame::AllMoveOnChart()
  {
   FieldMoveOnChart();
   StatusPanelMoveOnChart();
   ControlPanelMoveOnChart();
   ObstacleMoveOnChart();
   SnakeMoveOnChart();
   FoodMoveOnChart();
  }
//+------------------------------------------------------------------+
//| Game initialization                                              |
//+------------------------------------------------------------------+
void CSnakeGame::Init()
  {
   CreateHeader();
   CreateField();
   CreateStatusPanel();
   CreateControlPanel();
   CreateObstacle();
   CreateSnake();
   CreateFood();
   ChartRedraw();
  }
//+------------------------------------------------------------------+
//| Game deinitialization                                            |
//+------------------------------------------------------------------+
void  CSnakeGame::Deinit()
  {
   DeleteFood();
   DeleteSnake();
   DeleteObstacle();
   DeleteControlPanel();
   DeleteStatusPanel();
   DeleteField();
   DeleteHeader();
  }
```

### The Game Control

```
//+------------------------------------------------------------------+
//| Dummy Start game method                                          |
//+------------------------------------------------------------------+
void CSnakeGame::StartGame()
  {
   return;
  }
//+------------------------------------------------------------------+
//| Dummy Pause game method                                          |
//+------------------------------------------------------------------+
void CSnakeGame::PauseGame()
  {
   return;
  }
//+------------------------------------------------------------------+
//| Stop game method                                                 |
//+------------------------------------------------------------------+
void CSnakeGame::StopGame()
  {
   CChartObjectEdit *edit_obj;
   current_level=0;
   current_lives=LIVES_SNAKE;
   //setting new value for the "Level" field of the status panel
   edit_obj=status_panel_obj_arr.At(0);
   edit_obj.Description(StringFormat(spaces6+LEVEL_EDIT_TEXT,current_level,MAX_LEVEL));
   //setting new value for the "Lives" field of the status panel
   edit_obj=status_panel_obj_arr.At(2);
   edit_obj.Description(StringFormat(spaces8+LIVES_EDIT_TEXT,current_lives));
   DeleteFood();
   DeleteSnake();
   DeleteObstacle();
   CreateObstacle();
   CreateSnake();
   CreateFood();
  }
//+------------------------------------------------------------------+
//| Level restart method                                             |
//+------------------------------------------------------------------+
void CSnakeGame::ResetGame()
  {
   CChartObjectEdit *edit_obj;
   if(current_lives-1==-1)StopGame();
   else
     {
      //decreasing the number of lives
      current_lives--;
      //updating it at the status panel
      edit_obj=status_panel_obj_arr.At(2);
      edit_obj.Description(StringFormat(spaces8+LIVES_EDIT_TEXT,current_lives));
      DeleteFood();
      DeleteSnake();
      CreateSnake();
      CreateFood();
     }
  }
//+------------------------------------------------------------------+
//| Next level method                                                |
//+------------------------------------------------------------------+
void CSnakeGame::NextLevel()
  {
   CChartObjectEdit *edit_obj;
   current_lives=LIVES_SNAKE;
   //to the initial level if there isn't next level
   if(current_level+1>MAX_LEVEL)StopGame();
   else
     {
      //else increasing the level and updating the startus panel contents
      current_level++;
      edit_obj=status_panel_obj_arr.At(0);
      edit_obj.Description(StringFormat(spaces6+LEVEL_EDIT_TEXT,current_level,MAX_LEVEL));
      edit_obj=status_panel_obj_arr.At(2);
      edit_obj.Description(StringFormat(spaces8+LIVES_EDIT_TEXT,current_lives));
      DeleteFood();
      DeleteSnake();
      DeleteObstacle();
      CreateObstacle();
      CreateSnake();
      CreateFood();
     }
  }
```

### The Event Handling (final code)

```
// Declaring and creating an object of CSnakeGame type at global level
CSnakeGame snake_field;
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   snake_field.Init();
   EventSetTimer(1);
   return(0);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   snake_field.Deinit();
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void OnTimer()
  {
   //setting the buttons unpressed
   if(ObjectFind(0,START_GAME_BUTTON_NAME)>=0 && ObjectGetInteger(0,START_GAME_BUTTON_NAME,OBJPROP_STATE)==true)
      ObjectSetInteger(0,START_GAME_BUTTON_NAME,OBJPROP_STATE,false);
   if(ObjectFind(0,PAUSE_GAME_BUTTON_NAME)>=0 && ObjectGetInteger(0,PAUSE_GAME_BUTTON_NAME,OBJPROP_STATE)==true)
      ObjectSetInteger(0,PAUSE_GAME_BUTTON_NAME,OBJPROP_STATE,false);
   if(ObjectFind(0,STOP_GAME_BUTTON_NAME)>=0 && ObjectGetInteger(0,STOP_GAME_BUTTON_NAME,OBJPROP_STATE)==true)
      ObjectSetInteger(0,STOP_GAME_BUTTON_NAME,OBJPROP_STATE,false);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void OnChartEvent(const int id,
                  const long &lparam,
                  const double &dparam,
                  const string &sparam)
  {
   long x,y;
   static bool press_key=true;
   static bool press_button=false;
   static bool move=false;
   //if key has been pressed and the snake has moved, let's specify the new movement direction
   if(id==CHARTEVENT_KEYDOWN && press_key==false)
     {
      if((lparam==VK_LEFT) && (snake_field.GetDirection()!=DIRECTION_LEFT && snake_field.GetDirection()!=DIRECTION_RIGHT))
         snake_field.SetDirection(DIRECTION_LEFT);
      else if((lparam==VK_RIGHT) && (snake_field.GetDirection()!=DIRECTION_LEFT && snake_field.GetDirection()!=DIRECTION_RIGHT))
         snake_field.SetDirection(DIRECTION_RIGHT);
      else if((lparam==VK_DOWN) && (snake_field.GetDirection()!=DIRECTION_UP && snake_field.GetDirection()!=DIRECTION_DOWN))
         snake_field.SetDirection(DIRECTION_DOWN);
      else if((lparam==VK_UP) && (snake_field.GetDirection()!=DIRECTION_UP && snake_field.GetDirection()!=DIRECTION_DOWN))
         snake_field.SetDirection(DIRECTION_UP);
      press_key=true;
     }
   //if "Start" button has been pressed and press_button=false
   if(id==CHARTEVENT_OBJECT_CLICK && sparam==START_GAME_BUTTON_NAME && press_button==false)
     {
         //waiting 1 second
         Sleep(1000);
         //generating new event for snake movement
         EventChartCustom(0,0,0,0,"");
         press_button=true;
     }
   //if "Pause" button has been pressed
   else if(id==CHARTEVENT_OBJECT_CLICK && sparam==PAUSE_GAME_BUTTON_NAME)
     {
      press_button=false;
     }
   //if "Stop" button has been pressed
   else if(id==CHARTEVENT_OBJECT_CLICK && sparam==STOP_GAME_BUTTON_NAME)
     {
      snake_field.StopGame();
      press_key=true;
      press_button=false;
     }
   //processing of the snake movement event, if press_button=true
   else if(id==CHARTEVENT_CUSTOM && press_button==true)
     {
      snake_field.SnakeMoveOnField();
      press_key=false;
     }
   //processing of the game restart event
   else if(id==CHARTEVENT_CUSTOM+1)
     {
      snake_field.ResetGame();
      Sleep(1000);
      press_key=true;
      press_button=false;
     }
   //processing of the next level event
   else if(id==CHARTEVENT_CUSTOM+2)
     {
      snake_field.NextLevel();
      Sleep(1000);
      press_key=true;
      press_button=false;
     }
   //processing of the header movement event
   else if(id==CHARTEVENT_OBJECT_DRAG && sparam==HEADER_BUTTON_NAME)
     {
      x=ObjectGetInteger(0,sparam,OBJPROP_XDISTANCE);
      y=ObjectGetInteger(0,sparam,OBJPROP_YDISTANCE);
      snake_field.SetHeaderPos(x,y);
      snake_field.AllMoveOnChart();
     }
  }
//+------------------------------------------------------------------+
```

The _press\_key_ and _press\_button_ are two [static variables](https://www.mql5.com/en/docs/basis/variables/static), defined in the [OnChartEvent](https://www.mql5.com/en/docs/basis/function/events) event handler function.

The game start is allowed if _press\_button_ variable is false. After the click on the "Start" button, the _press\_button_ variable is set to true (it prohibits the re-execution of code that starts the game), this state remains  the same until one of the following events:

- Restart of the current level;
- Transition to the next level;
- The game pause (the "Pause" button has pressed);
- The game stop (the "Stop" button has pressed).


The change of the snake movement direction is possible if it's perpendicular to the current direction, as well as after the snake has moved on the playing field (the value of _press\_key_ variable indicates about it)


These conditions are taken into account in the [CHARTEVENT\_KEYDOWN](https://www.mql5.com/en/docs/constants/chartconstants/enum_chartevents) event processing function (keypress event).

Then you move the header, the [CHARTEVENT\_OBJECT\_DRAG](https://www.mql5.com/en/docs/constants/chartconstants/enum_chartevents) event is generated, the _header\_left_ and _header\_top_  fields of the CSnakeGame class are redefined. The movement of the other game elements is determined by the values of these fields.

The movement of playing field is implemented the way, presented in the [TradePad\_Sample](https://www.mql5.com/en/code/68).

### Conclusion

In this article we have considered an example of writing the games in MQL5.

We have introduced the [Standard Library](https://www.mql5.com/en/docs/standardlibrary) classes (the [control classes](https://www.mql5.com/en/docs/standardlibrary/chart_object_classes/obj_controls)), the [CArrayObj](https://www.mql5.com/en/docs/standardlibrary/datastructures/carrayobj) class, and also have learned how to perform the periodical function call with events handling.

An archive with source codes of the "Snake" game can be downloaded at the reference below. The archive should be unpacked into the folder **terminal\_data\_folder.**

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/65](https://www.mql5.com/ru/articles/65)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/65.zip "Download all attachments in the single ZIP archive")

[snake-mql5-doc.zip](https://www.mql5.com/en/articles/download/65/snake-mql5-doc.zip "Download snake-mql5-doc.zip")(405.1 KB)

[snake\_game\_mql5\_en.zip](https://www.mql5.com/en/articles/download/65/snake_game_mql5_en.zip "Download snake_game_mql5_en.zip")(16.1 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [AutoElliottWaveMaker - MetaTrader 5 Tool for Semi-Automatic Analysis of Elliott Waves](https://www.mql5.com/en/articles/378)
- [The Implementation of Automatic Analysis of the Elliott Waves in MQL5](https://www.mql5.com/en/articles/260)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/913)**
(15)


![Евгений](https://c.mql5.com/avatar/avatar_na2.png)

**[Евгений](https://www.mql5.com/en/users/space_cowboy)**
\|
12 Apr 2010 at 13:59

**MoneyJinnписал(а) [#](https://www.mql5.com/ru/forum/757#comment_4289) :**

The brighter and prettier, the closer you get to the casino.

I agree, all this beauty only increases the excitement, which leads to a drain.


![Vasily](https://c.mql5.com/avatar/2010/2/4B89624F-52CB.jpg)

**[Vasily](https://www.mql5.com/en/users/corewintt)**
\|
12 Apr 2010 at 15:28

I wonder! will happy farmer be implemented on mcle?


![Morndyk](https://c.mql5.com/avatar/2020/7/5F1F0994-563A.png)

**[Morndyk](https://www.mql5.com/en/users/morndyk)**
\|
27 Jul 2020 at 17:09

This is an extremely interesting [project](https://www.mql5.com/en/articles/7863 "Article: Projects allow you to create profitable trading robots! But it is not exactly"), especially considering that the game is currently growing. It is worth paying attention to the web applications of the gaming industry, so I recommend moving in this direction and further.


![SBS NCPL](https://c.mql5.com/avatar/2024/9/66D61E69-E111.png)

**[SBS NCPL](https://www.mql5.com/en/users/sbsncpl)**
\|
2 Sep 2024 at 20:26

Wow, creating a [Snake game](https://www.mql5.com/en/articles/65 "Article: Creating a \"Snake\" Game in MQL5") in MQL5 is quite an ambitious project! I'm impressed by your coding skills. I've dabbled in game development myself, mostly with modding existing games like Hollow Knight APK mod menu, but creating a game from scratch is a whole different level. I'm curious, what inspired you to choose MQL5 as the platform for your game? And how did you handle the movement and collision detection logic? I've found that to be one of the trickiest parts of game development. Looking forward to seeing the finished product.


![Clemence Benjamin](https://c.mql5.com/avatar/2025/3/67df27c6-2936.png)

**[Clemence Benjamin](https://www.mql5.com/en/users/billionaire2024)**
\|
21 Mar 2025 at 00:11

I've been searching for something like this for a long time!

Finally, found it here. Thank you so much!

![MetaTrader 5: Publishing trading forecasts and live trading statements via e-mail on blogs, social networks and dedicated websites](https://c.mql5.com/2/0/social-network.png)[MetaTrader 5: Publishing trading forecasts and live trading statements via e-mail on blogs, social networks and dedicated websites](https://www.mql5.com/en/articles/80)

This article aims to present ready-made solutions for publishing forecasts using MetaTrader 5. It covers a range of ideas: from using dedicated websites for publishing MetaTrader statements, through setting up one's own website with virtually no web programming experience needed and finally integration with a social network microblogging service that allows many readers to join and follow the forecasts. All solutions presented here are 100% free and possible to setup by anyone with a basic knowledge of e-mail and ftp services. There are no obstacles to use the same techniques for professional hosting and commercial trading forecast services.

![Creating Tick Indicators in MQL5](https://c.mql5.com/2/0/Untitled7.png)[Creating Tick Indicators in MQL5](https://www.mql5.com/en/articles/60)

In this article, we will consider the creation of two indicators: the tick indicator, which plots the tick chart of the price and tick candle indicator, which plot candles with the specified number of ticks. Each of the indicators writes the incoming prices into a file, and uses the saved data after the restart of the indicator (these data also can be used by the other programs)

![Creating an Indicator with Graphical Control Options](https://c.mql5.com/2/0/macd__1.png)[Creating an Indicator with Graphical Control Options](https://www.mql5.com/en/articles/42)

Those who are familiar with market sentiments, know the MACD indicator (its full name is Moving Average Convergence/Divergence) - the powerful tool for analyzing the price movement, used by traders from the very first moments of appearance of the computer analysis methods. In this article we'll consider possible modifications of MACD and implement them in one indicator with the possibility to graphically switch between the modifications.

![Using the Object Pointers in MQL5](https://c.mql5.com/2/0/pointer_small.png)[Using the Object Pointers in MQL5](https://www.mql5.com/en/articles/36)

By default, all objects in MQL5 are passed by reference, but there is a possibility to use the object pointers. However it's necessary to perform the pointer checking, because the object may be not initialized. In this case, the MQL5 program will be terminated with critical error and unloaded. The objects, created automatically, doesn't cause such an error, so in this sence, they are quite safe. In this article, we will try to understand the difference between the object reference and object pointer, and consider how to write secure code, that uses the pointers.

[![](https://www.mql5.com/ff/sh/qv94j0cd8n2n55z9z2/01.png)![](https://www.mql5.com/ff/sh/qv94j0cd8n2n55z9z2/02.png)Boost your trading experienceRead our book "MQL5 Programming for Traders"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/book%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.algobook%26utm_content=visit.page%26utm_campaign=algobook.promo.04.2024&a=heclgjpfbvfghpmyaciuaesdtswflupo&s=4255fbe1b8cbc4d1b40afbaebf4235e5ace8b5103cba60d996897a03d588556f&uid=&ref=https://www.mql5.com/en/articles/65&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5068334466469918778)

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