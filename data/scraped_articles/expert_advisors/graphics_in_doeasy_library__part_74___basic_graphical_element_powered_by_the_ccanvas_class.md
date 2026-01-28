---
title: Graphics in DoEasy library (Part 74): Basic graphical element powered by the CCanvas class
url: https://www.mql5.com/en/articles/9493
categories: Expert Advisors
relevance_score: 0
scraped_at: 2026-01-24T14:11:51.118237
---

[![](https://www.mql5.com/ff/sh/0uquj7zv5pmx2m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
Market analytics in MQL5 Channels\\
\\
Tens of thousands of traders have chosen this messaging app to receive trading tips.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=epadtzgppsywkaeumqycnulasoijfbgz&s=9615c3e5c371aa0d7b34529539d05c10df73b35a1e2213e4ceee008933c7ede0&uid=&ref=https://www.mql5.com/en/articles/9493&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5083397110050593525)

MetaTrader 5 / Examples


### Contents

- [Concept](https://www.mql5.com/en/articles/9493#node01)
- [Improving library classes](https://www.mql5.com/en/articles/9493#node02)
- [Base object of all library graphical objects based on canvas](https://www.mql5.com/en/articles/9493#node03)
- [Test](https://www.mql5.com/en/articles/9493#node04)
- [What's next?](https://www.mql5.com/en/articles/9493#node05)


### Concept

[In the previous article](https://www.mql5.com/en/articles/9442), I started working on the large library section for handling graphics. Namely, I started the development of the form object to be the main object of all library graphical objects [based on the CCanvas standard library class](https://www.mql5.com/en/docs/standardlibrary/canvasgraphics/ccanvas). I also tested some mechanics and made preparations for further development. However, a careful analysis showed that the selected concept differs from the concept of constructing library objects, and the form object is much more complex than the base object.

I will introduce the concept of the "element" for the base graphical object on the canvas. This concept is to be used to build the rest of the graphical objects. For example, the form object is also a minimally sufficient object for drawing graphical constructions in a program but it can already be an independent object for design. It already has the ability to draw the object frame, various shapes and a text. In contrast, the element object serves as the basis for creation of all subsequent objects in the library "graphical" hierarchy, for example:

- **The base graphical object** is a descendant of [CObject](https://www.mql5.com/en/docs/standardlibrary/cobject). It contains the properties inherent in graphical objects that can be built in the terminal;

- **Element object on the canvas** has the properties of the object based on the canvas object;

- **Form object** features additional properties and functionality for designing the appearance of the element object;

- **Window object** is a composite object based on element and form objects;

- etc.

Based on the new concept, I am going to rework the basic class of the CGBaseObj library graphical objects and create a new "graphical element" object that fully repeats the entire concept of building basic library objects. Later, such an approach will allow us to quickly search for the necessary graphical objects, as well as sort them and manage their behavior and rendering.

### Improving library classes

In MQL5\\Include\\DoEasy\ **Data.mqh**, add the new message index:

```
   MSG_LIB_SYS_FAILED_CREATE_STORAGE_FOLDER,          // Failed to create folder for storing files. Error:
   MSG_LIB_SYS_FAILED_ADD_ACC_OBJ_TO_LIST,            // Error. Failed to add current account object to collection list
   MSG_LIB_SYS_FAILED_CREATE_CURR_ACC_OBJ,            // Error. Failed to create account object with current account data
   MSG_LIB_SYS_FAILED_OPEN_FILE_FOR_WRITE,            // Could not open file for writing
   MSG_LIB_SYS_INPUT_ERROR_NO_SYMBOL,                 // Input error: no symbol
   MSG_LIB_SYS_FAILED_CREATE_SYM_OBJ,                 // Failed to create symbol object
   MSG_LIB_SYS_FAILED_ADD_SYM_OBJ,                    // Failed to add symbol
   MSG_LIB_SYS_FAILED_CREATE_ELM_OBJ,                 // Failed to create the graphical element object
```

and the text corresponding to a newly added index:

```
   {"Не удалось создать папку хранения файлов. Ошибка: ","Could not create file storage folder. Error: "},
   {"Ошибка. Не удалось добавить текущий объект-аккаунт в список-коллекцию","Error. Failed to add current account object to collection list"},
   {"Ошибка. Не удалось создать объект-аккаунт с данными текущего счёта","Error. Failed to create account object with current account data"},
   {"Не удалось открыть для записи файл ","Could not open file for writing: "},
   {"Ошибка входных данных: нет символа ","Input error: no "},
   {"Не удалось создать объект-символ ","Failed to create symbol object "},
   {"Не удалось добавить символ ","Failed to add "},
   {"Не удалось создать объект-графический элемент ","Failed to create graphic element object "},
```

For the new "graphical element" object in \\MQL5\\Include\\DoEasy\ **Defines.mqh**, add its type to the enumeration list of graphical object types, as well as its integer and string properties:

```
//+------------------------------------------------------------------+
//| The list of graphical element types                              |
//+------------------------------------------------------------------+
enum ENUM_GRAPH_ELEMENT_TYPE
  {
   GRAPH_ELEMENT_TYPE_ELEMENT,                        // Element
   GRAPH_ELEMENT_TYPE_FORM,                           // Form
   GRAPH_ELEMENT_TYPE_WINDOW,                         // Window
  };
//+------------------------------------------------------------------+
//| Integer properties of the graphical element on the canvas        |
//+------------------------------------------------------------------+
enum ENUM_CANV_ELEMENT_PROP_INTEGER
  {
   CANV_ELEMENT_PROP_ID = 0,                          // Form ID
   CANV_ELEMENT_PROP_TYPE,                            // Graphical element type
   CANV_ELEMENT_PROP_NUM,                             // Element index in the list
   CANV_ELEMENT_PROP_CHART_ID,                        // Chart ID
   CANV_ELEMENT_PROP_WND_NUM,                         // Chart subwindow index
   CANV_ELEMENT_PROP_COORD_X,                         // Form's X coordinate on the chart
   CANV_ELEMENT_PROP_COORD_Y,                         // Form's Y coordinate on the chart
   CANV_ELEMENT_PROP_WIDTH,                           // Form width
   CANV_ELEMENT_PROP_HEIGHT,                          // Form height
   CANV_ELEMENT_PROP_RIGHT,                           // Form right border
   CANV_ELEMENT_PROP_BOTTOM,                          // Form bottom border
   CANV_ELEMENT_PROP_ACT_SHIFT_LEFT,                  // Active area offset from the left edge of the form
   CANV_ELEMENT_PROP_ACT_SHIFT_TOP,                   // Active area offset from the top edge of the form
   CANV_ELEMENT_PROP_ACT_SHIFT_RIGHT,                 // Active area offset from the right edge of the form
   CANV_ELEMENT_PROP_ACT_SHIFT_BOTTOM,                // Active area offset from the bottom edge of the form
   CANV_ELEMENT_PROP_OPACITY,                         // Form opacity
   CANV_ELEMENT_PROP_COLOR_BG,                        // Form background color
   CANV_ELEMENT_PROP_MOVABLE,                         // Form moveability flag
   CANV_ELEMENT_PROP_ACTIVE,                          // Form activity flag
   CANV_ELEMENT_PROP_COORD_ACT_X,                     // X coordinate of the form's active area
   CANV_ELEMENT_PROP_COORD_ACT_Y,                     // Y coordinate of the form's active area
   CANV_ELEMENT_PROP_ACT_RIGHT,                       // Right border of the form's active area
   CANV_ELEMENT_PROP_ACT_BOTTOM,                      // Bottom border of the form's active area
  };
#define CANV_ELEMENT_PROP_INTEGER_TOTAL (23)          // Total number of integer properties
#define CANV_ELEMENT_PROP_INTEGER_SKIP  (0)           // Number of integer properties not used in sorting
//+------------------------------------------------------------------+
//| Real properties of the graphical element on the canvas           |
//+------------------------------------------------------------------+
enum ENUM_CANV_ELEMENT_PROP_DOUBLE
  {
   CANV_ELEMENT_PROP_DUMMY = CANV_ELEMENT_PROP_INTEGER_TOTAL, // DBL stub
  };
#define CANV_ELEMENT_PROP_DOUBLE_TOTAL  (1)           // Total number of real properties
#define CANV_ELEMENT_PROP_DOUBLE_SKIP   (1)           // Number of real properties not used in sorting
//+------------------------------------------------------------------+
//| String properties of the graphical element on the canvas         |
//+------------------------------------------------------------------+
enum ENUM_CANV_ELEMENT_PROP_STRING
  {
   CANV_ELEMENT_PROP_NAME_OBJ = (CANV_ELEMENT_PROP_INTEGER_TOTAL+CANV_ELEMENT_PROP_DOUBLE_TOTAL), // Form object name
   CANV_ELEMENT_PROP_NAME_RES,                        // Graphical resource name
  };
#define CANV_ELEMENT_PROP_STRING_TOTAL  (2)           // Total number of string properties
//+------------------------------------------------------------------+
```

Since the canvas-based objects have no real properties yet, while the concept of constructing library objects requires their presence, I have added the real property stub as the only real property.

In order to sort the graphical element objects by properties, add the enumeration with the possible sorting criteria:

```
//+------------------------------------------------------------------+
//| Possible sorting criteria of graphical elements on the canvas    |
//+------------------------------------------------------------------+
#define FIRST_CANV_ELEMENT_DBL_PROP  (CANV_ELEMENT_PROP_INTEGER_TOTAL-CANV_ELEMENT_PROP_INTEGER_SKIP)
#define FIRST_CANV_ELEMENT_STR_PROP  (CANV_ELEMENT_PROP_INTEGER_TOTAL-CANV_ELEMENT_PROP_INTEGER_SKIP+CANV_ELEMENT_PROP_DOUBLE_TOTAL-CANV_ELEMENT_PROP_DOUBLE_SKIP)
enum ENUM_SORT_CANV_ELEMENT_MODE
  {
//--- Sort by integer properties
   SORT_BY_CANV_ELEMENT_ID = 0,                       // Sort by form ID
   SORT_BY_CANV_ELEMENT_TYPE,                         // Sort by graphical element type
   SORT_BY_CANV_ELEMENT_NUM,                          // Sort by form index in the list
   SORT_BY_CANV_ELEMENT_CHART_ID,                     // Sort by chart ID
   SORT_BY_CANV_ELEMENT_WND_NUM,                      // Sort by chart window index
   SORT_BY_CANV_ELEMENT_COORD_X,                      // Sort by the form X coordinate on the chart
   SORT_BY_CANV_ELEMENT_COORD_Y,                      // Sort by the form Y coordinate on the chart
   SORT_BY_CANV_ELEMENT_WIDTH,                        // Sort by the form width
   SORT_BY_CANV_ELEMENT_HEIGHT,                       // Sort by the form height
   SORT_BY_CANV_ELEMENT_RIGHT,                        // Sort by the form right border
   SORT_BY_CANV_ELEMENT_BOTTOM,                       // Sort by the form bottom border
   SORT_BY_CANV_ELEMENT_ACT_SHIFT_LEFT,               // Sort by the active area offset from the left edge of the form
   SORT_BY_CANV_ELEMENT_ACT_SHIFT_TOP,                // Sort by the active area offset from the top edge of the form
   SORT_BY_CANV_ELEMENT_ACT_SHIFT_RIGHT,              // Sort by the active area offset from the right edge of the form
   SORT_BY_CANV_ELEMENT_ACT_SHIFT_BOTTOM,             // Sort by the active area offset from the bottom edge of the form
   SORT_BY_CANV_ELEMENT_OPACITY,                      // Sort by the form opacity
   SORT_BY_CANV_ELEMENT_COLOR_BG,                     // Sort by the form background color
   SORT_BY_CANV_ELEMENT_MOVABLE,                      // Sort by the form moveability flag
   SORT_BY_CANV_ELEMENT_ACTIVE,                       // Sort by the form activity flag
   SORT_BY_CANV_ELEMENT_COORD_ACT_X,                  // Sort by X coordinate of the form active area
   SORT_BY_CANV_ELEMENT_COORD_ACT_Y,                  // Sort by Y coordinate of the form active area
   SORT_BY_CANV_ELEMENT_ACT_RIGHT,                    // Sort by the right border of the form active area
   SORT_BY_CANV_ELEMENT_ACT_BOTTOM,                   // Sort by the bottom border of the form active area
//--- Sort by real properties

//--- Sort by string properties
   SORT_BY_CANV_ELEMENT_NAME_OBJ = FIRST_CANV_ELEMENT_STR_PROP,// Sort by the form object name
   SORT_BY_CANV_ELEMENT_NAME_RES,                     // Sort by the graphical resource name
  };
//+------------------------------------------------------------------+
```

All these enumerations were described [in the first article](https://www.mql5.com/en/articles/5654) and considered many times, so I will not dwell on them here.

Before creating the "graphical element" object, revise the base object class of all library graphical objects in MQL5\\Include\\DoEasy\\Objects\\Graph\ **GBaseObj.mqh**.

The object is to store all common properties of any graphical objects, such as created object type, chart ID and index of the subwindow, in which the graphical object, its name and name prefix are set. Any graphical objects of the library are to be inherited from the class.

It is more convenient to completely re-create this class rather than fix the existing one. Therefore, simply remove everything from the file and add the necessary things:

```
//+------------------------------------------------------------------+
//|                                                     GBaseObj.mqh |
//|                                  Copyright 2021, MetaQuotes Ltd. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2021, MetaQuotes Ltd."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
#property strict    // Necessary for mql4
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include "..\..\Services\DELib.mqh"
#include <Graphics\Graphic.mqh>
//+------------------------------------------------------------------+
//| Class of the base object of the library graphical objects        |
//+------------------------------------------------------------------+
class CGBaseObj : public CObject
  {
private:
   int               m_type;                             // Object type
protected:
   string            m_name_prefix;                      // Object name prefix
   string            m_name;                             // Object name
   long              m_chart_id;                         // Chart ID
   int               m_subwindow;                        // Subwindow index
   int               m_shift_y;                          // Subwindow Y coordinate shift

public:
//--- Return the values of class variables
   string            Name(void)                          const { return this.m_name;      }
   long              ChartID(void)                       const { return this.m_chart_id;  }
   int               SubWindow(void)                     const { return this.m_subwindow; }
//--- The virtual method returning the object type
   virtual int       Type(void)                          const { return this.m_type;      }

//--- Constructor/destructor
                     CGBaseObj();
                    ~CGBaseObj();
  };
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CGBaseObj::CGBaseObj() : m_shift_y(0), m_type(0), m_name_prefix(::MQLInfoString(MQL_PROGRAM_NAME)+"_")
  {
  }
//+------------------------------------------------------------------+
//| Destructor                                                       |
//+------------------------------------------------------------------+
CGBaseObj::~CGBaseObj()
  {
  }
//+------------------------------------------------------------------+
```

The file of the library service functions and [the Standard Library CGraphic class file](https://www.mql5.com/en/docs/standardlibrary/graphics/cgraphic) are immediately included into the file. The CCanvas class file is already included into CGraphic. At the same time, the CGraphic class has a wide range of methods for drawing various graphs. We will also need this in the future.

The class is inherited from [the base class](https://www.mql5.com/en/docs/standardlibrary/cobject) of the Standard Library allowing us to create graphical elements as CObject class objects and store the lists of the library graphical objects the same way as I already store all our objects in the appropriate collections.

The **m\_type** private variable is to store the object type from the ENUM\_GRAPH\_ELEMENT\_TYPE enumeration I discussed above.

By default, the object type is equal to zero and is returned by the Type() virtual method of the Standard Library base class:

```
   //--- method of identifying the object
   virtual int       Type(void)                                    const { return(0);      }
```

Here I have also redefined that method, so that it returns the **m\_type** variable depending on the time of the created graphical object.

**Protected class variables:**

- **m\_name\_prefix** — here I will store a name prefix of objects for identifying graphical objects by their affiliation with the program. Accordingly, here I will store the name of the program based on the library.

- **m\_name** stores a graphical object name. The full object name is created by summing the prefix and the name. Thus, when creating objects, we will only need to specify a unique name for a newly created object, while the "graphical element" object class adds a prefix to the name on its own. The prefix allows identifying the object with the program that created it.

- **m\_chart\_id** — here I will set an ID of the chart the graphical object is to be created on.
- **m\_subwindow** — subwindow of the chart the graphical object is built on.

- **m\_shift\_y** — offset of the Y coordinate of an object created in a chart subwindow.

**Public methods** simply return the values of the appropriate class variables:

```
public:
//--- Return the values of class variables
   string            Name(void)                          const { return this.m_name;      }
   long              ChartID(void)                       const { return this.m_chart_id;  }
   int               SubWindow(void)                     const { return this.m_subwindow; }
//--- The virtual method returning the object type
   virtual int       Type(void)                          const { return this.m_type;      }
```

In the initialization list of the **class constructor**, set the Y coordinate offset, object type (the default is 0) and the name prefix consisting of the program name and an underscore:

```
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CGBaseObj::CGBaseObj() : m_shift_y(0), m_type(0), m_name_prefix(::MQLInfoString(MQL_PROGRAM_NAME)+"_")
  {
  }
//+------------------------------------------------------------------+
```

### Base object of all library graphical objects based on canvas

Let's start the development of the "graphical element" object class [based on the CCanvas class](https://www.mql5.com/en/docs/standardlibrary/canvasgraphics/ccanvas).

In \\MQL5\\Include\\DoEasy\\Objects\ **Graph\**, create the new file **GCnvElement.mqh** of the **CGCnvElement** class.

Include the file of the library base graphical objectthe class should be inherited from into the class file:

```
//+------------------------------------------------------------------+
//|                                                  GCnvElement.mqh |
//|                                  Copyright 2021, MetaQuotes Ltd. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2021, MetaQuotes Ltd."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
#property strict    // Necessary for mql4
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include "GBaseObj.mqh"
//+------------------------------------------------------------------+
//| Class of the base object of the library graphical objects        |
//+------------------------------------------------------------------+
class CGCnvElement : public CGBaseObj
  {
  }
```

In the protected section of the class, declare the objects of the CCanvas and CPause classes and two methods returning the position of the specified coordinates relative to the element and its active area:

```
protected:
   CCanvas           m_canvas;                                 // CCanvas class object
   CPause            m_pause;                                  // Pause class object
//--- Return the cursor position relative to the (1) entire element and (2) the element's active area
   bool              CursorInsideElement(const int x,const int y);
   bool              CursorInsideActiveArea(const int x,const int y);

private:
```

In the private section of the class, declare the arrays for storing object properties and write two methods returning real indices of the specified properties in the appropriate arrays:

```
private:
   long              m_long_prop[ORDER_PROP_INTEGER_TOTAL];    // Integer properties
   double            m_double_prop[ORDER_PROP_DOUBLE_TOTAL];   // Real properties
   string            m_string_prop[ORDER_PROP_STRING_TOTAL];   // String properties

//--- Return the index of the array the order's (1) double and (2) string properties are located at
   int               IndexProp(ENUM_CANV_ELEMENT_PROP_DOUBLE property)  const { return(int)property-CANV_ELEMENT_PROP_INTEGER_TOTAL;                                 }
   int               IndexProp(ENUM_CANV_ELEMENT_PROP_STRING property)  const { return(int)property-CANV_ELEMENT_PROP_INTEGER_TOTAL-CANV_ELEMENT_PROP_DOUBLE_TOTAL;  }

public:
```

The public section of the class features standard methods of library class objects for setting the properties to the arrays and returning the properties from the arrays, the methods returning the flags of the object supporting the specified property and the methods comparing two objects:

```
public:
//--- Set object's (1) integer, (2) real and (3) string properties
   void              SetProperty(ENUM_CANV_ELEMENT_PROP_INTEGER property,long value)   { this.m_long_prop[property]=value;                      }
   void              SetProperty(ENUM_CANV_ELEMENT_PROP_DOUBLE property,double value)  { this.m_double_prop[this.IndexProp(property)]=value;    }
   void              SetProperty(ENUM_CANV_ELEMENT_PROP_STRING property,string value)  { this.m_string_prop[this.IndexProp(property)]=value;    }
//--- Return object’s (1) integer, (2) real and (3) string property from the properties array
   long              GetProperty(ENUM_CANV_ELEMENT_PROP_INTEGER property)        const { return this.m_long_prop[property];                     }
   double            GetProperty(ENUM_CANV_ELEMENT_PROP_DOUBLE property)         const { return this.m_double_prop[this.IndexProp(property)];   }
   string            GetProperty(ENUM_CANV_ELEMENT_PROP_STRING property)         const { return this.m_string_prop[this.IndexProp(property)];   }

//--- Return the flag of the object supporting this property
   virtual bool      SupportProperty(ENUM_CANV_ELEMENT_PROP_INTEGER property)          { return true; }
   virtual bool      SupportProperty(ENUM_CANV_ELEMENT_PROP_DOUBLE property)           { return true; }
   virtual bool      SupportProperty(ENUM_CANV_ELEMENT_PROP_STRING property)           { return true; }

//--- Compare CGCnvElement objects with each other by all possible properties (for sorting the lists by a specified object property)
   virtual int       Compare(const CObject *node,const int mode=0) const;
//--- Compare CGCnvElement objects with each other by all properties (to search equal objects)
   bool              IsEqual(CGCnvElement* compared_obj) const;
//--- Creates the control
```

All these methods are standard for library objects. I considered them [in the first article](https://www.mql5.com/en/articles/5654).

The public section of the class features the method for creating the "graphical element" object on the canvas, the method returning the pointer to the created canvas object, the method setting the canvas update frequency, the method for shifting the canvas on the chart and the methods for a simplified access to the object properties:

```
//--- Creates the control
   bool              Create(const long chart_id,
                            const int wnd_num,
                            const string name,
                            const int x,
                            const int y,
                            const int w,
                            const int h,
                            const color colour,
                            const uchar opacity,
                            const bool redraw=false);

//--- Return the pointer to a canvas object
   CCanvas          *CanvasObj(void)                                                   { return &this.m_canvas;               }
//--- Set the canvas update frequency
   void              SetFrequency(const ulong value)                                   { this.m_pause.SetWaitingMSC(value);   }
//--- Update the coordinates (shift the canvas)
   bool              Move(const int x,const int y,const bool redraw=false);

//--- Constructors/Destructor
                     CGCnvElement(const ENUM_GRAPH_ELEMENT_TYPE element_type,
                                  const int     element_id,
                                  const int     element_num,
                                  const long    chart_id,
                                  const int     wnd_num,
                                  const string  name,
                                  const int     x,
                                  const int     y,
                                  const int     w,
                                  const int     h,
                                  const color   colour,
                                  const uchar   opacity,
                                  const bool    movable=true,
                                  const bool    activity=true,
                                  const bool    redraw=false);
                     CGCnvElement(){;}
                    ~CGCnvElement();

//+------------------------------------------------------------------+
//| Methods of simplified access to object properties                |
//+------------------------------------------------------------------+
//--- Set the (1) X, (2) Y coordinates, (3) element width and (4) height,
   bool              SetCoordX(const int coord_x);
   bool              SetCoordY(const int coord_y);
   bool              SetWidth(const int width);
   bool              SetHeight(const int height);
//--- Set the shift of the (1) left, (2) top, (3) right, (4) bottom edge of the active area relative to the element,
//--- (5) all shifts of the active area edges relative to the element and (6) the element opacity
   void              SetActiveAreaLeftShift(const int value)   { this.SetProperty(CANV_ELEMENT_PROP_ACT_SHIFT_LEFT,fabs(value));    }
   void              SetActiveAreaRightShift(const int value)  { this.SetProperty(CANV_ELEMENT_PROP_ACT_SHIFT_RIGHT,fabs(value));   }
   void              SetActiveAreaTopShift(const int value)    { this.SetProperty(CANV_ELEMENT_PROP_ACT_SHIFT_TOP,fabs(value));     }
   void              SetActiveAreaBottomShift(const int value) { this.SetProperty(CANV_ELEMENT_PROP_ACT_SHIFT_BOTTOM,fabs(value));  }
   void              SetActiveAreaShift(const int left_shift,const int bottom_shift,const int right_shift,const int top_shift);
   void              SetOpacity(const uchar value,const bool redraw=false);

//--- Return the shift (1) of the left, (2) right, (3) top and (4) bottom edge of the element active area
   int               ActiveAreaLeftShift(void)           const { return (int)this.GetProperty(CANV_ELEMENT_PROP_ACT_SHIFT_LEFT);    }
   int               ActiveAreaRightShift(void)          const { return (int)this.GetProperty(CANV_ELEMENT_PROP_ACT_SHIFT_RIGHT);   }
   int               ActiveAreaTopShift(void)            const { return (int)this.GetProperty(CANV_ELEMENT_PROP_ACT_SHIFT_TOP);     }
   int               ActiveAreaBottomShift(void)         const { return (int)this.GetProperty(CANV_ELEMENT_PROP_ACT_SHIFT_BOTTOM);  }
//--- Return the coordinate (1) of the left, (2) right, (3) top and (4) bottom edge of the element active area
   int               ActiveAreaLeft(void)                const { return int(this.CoordX()+this.ActiveAreaLeftShift());              }
   int               ActiveAreaRight(void)               const { return int(this.RightEdge()-this.ActiveAreaRightShift());          }
   int               ActiveAreaTop(void)                 const { return int(this.CoordY()+this.ActiveAreaTopShift());               }
   int               ActiveAreaBottom(void)              const { return int(this.BottomEdge()-this.ActiveAreaBottomShift());        }
//--- Return (1) the opacity, coordinate (2) of the right and (3) bottom element edge
   uchar             Opacity(void)                       const { return (uchar)this.GetProperty(CANV_ELEMENT_PROP_OPACITY);         }
   int               RightEdge(void)                     const { return this.CoordX()+this.m_canvas.Width();                        }
   int               BottomEdge(void)                    const { return this.CoordY()+this.m_canvas.Height();                       }
//--- Return the (1) X, (2) Y coordinates, (3) element width and (4) height,
   int               CoordX(void)                        const { return (int)this.GetProperty(CANV_ELEMENT_PROP_COORD_X);           }
   int               CoordY(void)                        const { return (int)this.GetProperty(CANV_ELEMENT_PROP_COORD_Y);           }
   int               Width(void)                         const { return (int)this.GetProperty(CANV_ELEMENT_PROP_WIDTH);             }
   int               Height(void)                        const { return (int)this.GetProperty(CANV_ELEMENT_PROP_HEIGHT);            }
//--- Return the element (1) moveability and (2) activity flag
   bool              Movable(void)                       const { return (bool)this.GetProperty(CANV_ELEMENT_PROP_MOVABLE);          }
   bool              Active(void)                        const { return (bool)this.GetProperty(CANV_ELEMENT_PROP_ACTIVE);           }
//--- Return (1) the object name, (2) the graphical resource name, (3) the chart ID and (4) the chart subwindow index
   string            NameObj(void)                       const { return this.GetProperty(CANV_ELEMENT_PROP_NAME_OBJ);               }
   string            NameRes(void)                       const { return this.GetProperty(CANV_ELEMENT_PROP_NAME_RES);               }
   long              ChartID(void)                       const { return this.GetProperty(CANV_ELEMENT_PROP_CHART_ID);               }
   int               WindowNum(void)                     const { return (int)this.GetProperty(CANV_ELEMENT_PROP_WND_NUM);           }

  };
//+------------------------------------------------------------------+
```

Let's consider the implementation of the declared methods in details.

**The parametric class constructor:**

```
//+------------------------------------------------------------------+
//| Parametric constructor                                           |
//+------------------------------------------------------------------+
CGCnvElement::CGCnvElement(const ENUM_GRAPH_ELEMENT_TYPE element_type,
                           const int      element_id,
                           const int      element_num,
                           const long     chart_id,
                           const int      wnd_num,
                           const string   name,
                           const int      x,
                           const int      y,
                           const int      w,
                           const int      h,
                           const color    colour,
                           const uchar    opacity,
                           const bool     movable=true,
                           const bool     activity=true,
                           const bool     redraw=false)

  {
   this.m_name=this.m_name_prefix+name;
   this.m_chart_id=chart_id;
   this.m_subwindow=wnd_num;
   if(this.Create(chart_id,wnd_num,this.m_name,x,y,w,h,colour,opacity,redraw))
     {
      this.SetProperty(CANV_ELEMENT_PROP_NAME_RES,this.m_canvas.ResourceName()); // Graphical resource name
      this.SetProperty(CANV_ELEMENT_PROP_CHART_ID,CGBaseObj::ChartID());         // Chart ID
      this.SetProperty(CANV_ELEMENT_PROP_WND_NUM,CGBaseObj::SubWindow());        // Chart subwindow index
      this.SetProperty(CANV_ELEMENT_PROP_NAME_OBJ,CGBaseObj::Name());            // Element object name
      this.SetProperty(CANV_ELEMENT_PROP_TYPE,element_type);                     // Graphical element type
      this.SetProperty(CANV_ELEMENT_PROP_ID,element_id);                         // Element ID
      this.SetProperty(CANV_ELEMENT_PROP_NUM,element_num);                       // Element index in the list
      this.SetProperty(CANV_ELEMENT_PROP_COORD_X,x);                             // Element's X coordinate on the chart
      this.SetProperty(CANV_ELEMENT_PROP_COORD_Y,y);                             // Element's Y coordinate on the chart
      this.SetProperty(CANV_ELEMENT_PROP_WIDTH,w);                               // Element width
      this.SetProperty(CANV_ELEMENT_PROP_HEIGHT,h);                              // Element height
      this.SetProperty(CANV_ELEMENT_PROP_ACT_SHIFT_LEFT,0);                      // Active area offset from the left edge of the element
      this.SetProperty(CANV_ELEMENT_PROP_ACT_SHIFT_TOP,0);                       // Active area offset from the upper edge of the element
      this.SetProperty(CANV_ELEMENT_PROP_ACT_SHIFT_RIGHT,0);                     // Active area offset from the right edge of the element
      this.SetProperty(CANV_ELEMENT_PROP_ACT_SHIFT_BOTTOM,0);                    // Active area offset from the bottom edge of the element
      this.SetProperty(CANV_ELEMENT_PROP_OPACITY,opacity);                       // Element opacity
      this.SetProperty(CANV_ELEMENT_PROP_COLOR_BG,colour);                       // Element color
      this.SetProperty(CANV_ELEMENT_PROP_MOVABLE,movable);                       // Element moveability flag
      this.SetProperty(CANV_ELEMENT_PROP_ACTIVE,activity);                       // Element activity flag
      this.SetProperty(CANV_ELEMENT_PROP_RIGHT,this.RightEdge());                // Element right border
      this.SetProperty(CANV_ELEMENT_PROP_BOTTOM,this.BottomEdge());              // Element bottom border
      this.SetProperty(CANV_ELEMENT_PROP_COORD_ACT_X,this.ActiveAreaLeft());     // X coordinate of the element active area
      this.SetProperty(CANV_ELEMENT_PROP_COORD_ACT_Y,this.ActiveAreaTop());      // Y coordinate of the element active area
      this.SetProperty(CANV_ELEMENT_PROP_ACT_RIGHT,this.ActiveAreaRight());      // Right border of the element active area
      this.SetProperty(CANV_ELEMENT_PROP_ACT_BOTTOM,this.ActiveAreaBottom());    // Bottom border of the element active area
     }
   else
     {
      ::Print(CMessage::Text(MSG_LIB_SYS_FAILED_CREATE_ELM_OBJ),this.m_name);
     }
  }
//+------------------------------------------------------------------+
```

Here we first create an object name consisting of the object name prefix created in the parent class and the name passed in the constructor parameters. Thus, the unique object name looks as "Prefix\_Object\_Name".

Next, set the chart ID and subwindow index passed in the parameters to the parent class variables.

The method of creating the graphical object on the canvas is called afterwards. If the object is created successfully, write all data to the element object properties. If failed to create the graphical object of the CCanvas class, inform of that in the journal. The name with the prefix will already have been created and the chart ID will have been set together with its subwindow. Thus, we can try to create the CCanvas class object again by a new call of the Create() method. By default, the active area offset is set to zero from each side when creating an object, i.e. the object active area matches the size of the created graphical element. After its creation, the size and position of the active area can always be changed using the appropriate methods considered below.

**In the class destructor,** destroy the created object of the CCanvas class:

```
//+------------------------------------------------------------------+
//| Destructor                                                       |
//+------------------------------------------------------------------+
CGCnvElement::~CGCnvElement()
  {
   this.m_canvas.Destroy();
  }
//+------------------------------------------------------------------+
```

**The method comparing the graphical element objects by a specified property:**

```
//+----------------------------------------------------------------------+
//|Compare CGCnvElement objects with each other by the specified property|
//+----------------------------------------------------------------------+
int CGCnvElement::Compare(const CObject *node,const int mode=0) const
  {
   const CGCnvElement *obj_compared=node;
//--- compare integer properties of two objects
   if(mode<CANV_ELEMENT_PROP_INTEGER_TOTAL)
     {
      long value_compared=obj_compared.GetProperty((ENUM_CANV_ELEMENT_PROP_INTEGER)mode);
      long value_current=this.GetProperty((ENUM_CANV_ELEMENT_PROP_INTEGER)mode);
      return(value_current>value_compared ? 1 : value_current<value_compared ? -1 : 0);
     }
//--- compare real properties of two objects
   else if(mode<CANV_ELEMENT_PROP_DOUBLE_TOTAL+CANV_ELEMENT_PROP_INTEGER_TOTAL)
     {
      double value_compared=obj_compared.GetProperty((ENUM_CANV_ELEMENT_PROP_DOUBLE)mode);
      double value_current=this.GetProperty((ENUM_CANV_ELEMENT_PROP_DOUBLE)mode);
      return(value_current>value_compared ? 1 : value_current<value_compared ? -1 : 0);
     }
//--- compare string properties of two objects
   else if(mode<ORDER_PROP_DOUBLE_TOTAL+ORDER_PROP_INTEGER_TOTAL+ORDER_PROP_STRING_TOTAL)
     {
      string value_compared=obj_compared.GetProperty((ENUM_CANV_ELEMENT_PROP_STRING)mode);
      string value_current=this.GetProperty((ENUM_CANV_ELEMENT_PROP_STRING)mode);
      return(value_current>value_compared ? 1 : value_current<value_compared ? -1 : 0);
     }
   return 0;
  }
//+------------------------------------------------------------------+
```

The method is standard for all library objects. It was considered earlier. In short, the method receives the object whose specified parameter should be compared with the appropriate parameter of the current object. Depending on the passed parameter, get a similar one and return the result of comparing the parameters of two objects (1, -1 and 0 for 'more', 'less' and 'equal', accordingly).

**The method comparing the graphical element objects by all properties:**

```
//+------------------------------------------------------------------+
//| Compare CGCnvElement objects with each other by all properties   |
//+------------------------------------------------------------------+
bool CGCnvElement::IsEqual(CGCnvElement *compared_obj) const
  {
   int beg=0, end=CANV_ELEMENT_PROP_INTEGER_TOTAL;
   for(int i=beg; i<end; i++)
     {
      ENUM_CANV_ELEMENT_PROP_INTEGER prop=(ENUM_CANV_ELEMENT_PROP_INTEGER)i;
      if(this.GetProperty(prop)!=compared_obj.GetProperty(prop)) return false;
     }
   beg=end; end+=CANV_ELEMENT_PROP_DOUBLE_TOTAL;
   for(int i=beg; i<end; i++)
     {
      ENUM_CANV_ELEMENT_PROP_DOUBLE prop=(ENUM_CANV_ELEMENT_PROP_DOUBLE)i;
      if(this.GetProperty(prop)!=compared_obj.GetProperty(prop)) return false;
     }
   beg=end; end+=CANV_ELEMENT_PROP_STRING_TOTAL;
   for(int i=beg; i<end; i++)
     {
      ENUM_CANV_ELEMENT_PROP_STRING prop=(ENUM_CANV_ELEMENT_PROP_STRING)i;
      if(this.GetProperty(prop)!=compared_obj.GetProperty(prop)) return false;
     }
   return true;
  }
//+------------------------------------------------------------------+
```

The method is also standard for all library objects. In short, the method receives the object whose parameters should be compared with the current object ones. In three loops by all object properties, compare each new property of two objects. If there are unequal properties, the method returns false  — compared objects are not equal. Upon completing three loops, true is returned — all properties of the two compared objects are equal.

**The method creating the graphical element object:**

```
//+------------------------------------------------------------------+
//| Create the graphical element object                              |
//+------------------------------------------------------------------+
bool CGCnvElement::Create(const long chart_id,     // Chart ID
                          const int wnd_num,       // Chart subwindow
                          const string name,       // Element name
                          const int x,             // X coordinate
                          const int y,             // Y coordinate
                          const int w,             // Width
                          const int h,             // Height
                          const color colour,      // Background color
                          const uchar opacity,     // Opacity
                          const bool redraw=false) // Flag indicating the need to redraw

  {
   if(this.m_canvas.CreateBitmapLabel(chart_id,wnd_num,name,x,y,w,h,COLOR_FORMAT_ARGB_NORMALIZE))
     {
      this.m_canvas.Erase(::ColorToARGB(colour,opacity));
      this.m_canvas.Update(redraw);
      this.m_shift_y=(int)::ChartGetInteger(chart_id,CHART_WINDOW_YDISTANCE,wnd_num);
      return true;
     }
   return false;
  }
//+------------------------------------------------------------------+
```

The method receives all parameters necessary for the construction, and the second form of the [CreateBitmapLabel()](https://www.mql5.com/en/docs/standardlibrary/canvasgraphics/ccanvas/ccanvascreatebitmaplabel) method of the CCanvas class is called. If the graphical resource bound to the chart object is created successfully, [the graphical element is filled in with color](https://www.mql5.com/en/docs/standardlibrary/canvasgraphics/ccanvas/ccanvaserase) and the [Update() method](https://www.mql5.com/en/docs/standardlibrary/canvasgraphics/ccanvas/ccanvasupdate) is called for displaying the implemented changes on the screen. The method receives the screen redraw flag. If we update a composite object consisting of several graphical elements, the chart should be redrawn after making changes in all composite object elements to avoid multiple chart updates after each element is changed. Next, the **m\_shift** parent class variable receives the offset of the Y coordinate for the subwindow and true is returned. If no CCanvas class object is created, return false.

**The method returning the cursor position relative to the element:**

```
//+------------------------------------------------------------------+
//| Return the cursor position relative to the element               |
//+------------------------------------------------------------------+
bool CGCnvElement::CursorInsideElement(const int x,const int y)
  {
   return(x>=this.CoordX() && x<=this.RightEdge() && y>=this.CoordY() && y<=this.BottomEdge());
  }
//+------------------------------------------------------------------+
```

The method receives the integer coordinates of the X and Y cursor coordinates and the position of passed coordinates relative to the element dimensions — true is returned only if the cursor is located inside the element.

**The method returning the cursor position relative to the element active area:**

```
//+------------------------------------------------------------------+
//| Return the cursor position relative to the element active area   |
//+------------------------------------------------------------------+
bool CGCnvElement::CursorInsideActiveArea(const int x,const int y)
  {
   return(x>=this.ActiveAreaLeft() && x<=this.ActiveAreaRight() && y>=this.ActiveAreaTop() && y<=this.ActiveAreaBottom());
  }
//+------------------------------------------------------------------+
```

The method logic is similar to the previous method one. But the cursor coordinates position is returned relative to the borders of the element active area — true is returned only if the cursor is inside the active area.

**The method updating the element coordinates:**

```
//+------------------------------------------------------------------+
//| Update the coordinate elements                                   |
//+------------------------------------------------------------------+
bool CGCnvElement::Move(const int x,const int y,const bool redraw=false)
  {
//--- Leave if the element is not movable or inactive
   if(!this.Movable())
      return false;
//--- If failed to set new values into graphical object properties, return 'false'
   if(!this.SetCoordX(x) || !this.SetCoordY(y))
      return false;
   //--- If the update flag is activated, redraw the chart.
   if(redraw)
      ::ChartRedraw(this.ChartID());
   //--- Return 'true'
   return true;
  }
//+------------------------------------------------------------------+
```

The method receives the new coordinates of the upper left corner of the graphical element it should be placed on, as well as the chart redrawing flag. Next, check the object moveability flag and leave if the object is not movable. If failed to set the new coordinates to the object using the methods considered below, return false. Next, update the chart if the chart redrawing flag is set. As a result, return true.

**The method setting the new X coordinate:**

```
//+------------------------------------------------------------------+
//| Set the new  X coordinate                                        |
//+------------------------------------------------------------------+
bool CGCnvElement::SetCoordX(const int coord_x)
  {
   int x=(int)::ObjectGetInteger(this.ChartID(),this.NameObj(),OBJPROP_XDISTANCE);
   if(coord_x==x)
     {
      if(coord_x==GetProperty(CANV_ELEMENT_PROP_COORD_X))
         return true;
      this.SetProperty(CANV_ELEMENT_PROP_COORD_X,coord_x);
      return true;
     }
   if(::ObjectSetInteger(this.ChartID(),this.NameObj(),OBJPROP_XDISTANCE,coord_x))
     {
      this.SetProperty(CANV_ELEMENT_PROP_COORD_X,coord_x);
      return true;
     }
   return false;
  }
//+------------------------------------------------------------------+
```

The method receives the required X coordinate value. Next, get this coordinate from the object. If the passed coordinate is equal to the object one, the object should not be moved. But we need to check whether the same value is set in the object properties. If the values match, return true, otherwise set the passed new coordinate value to the object property and return true.

If the passed and the object coordinates do not match, set the new coordinate to the object. If setting is successful, write the value to the object property and return true. In all other cases, return false.

**The method setting the new Y coordinate:**

```
//+------------------------------------------------------------------+
//| Set the new Y coordinate                                         |
//+------------------------------------------------------------------+
bool CGCnvElement::SetCoordY(const int coord_y)
  {
   int y=(int)::ObjectGetInteger(this.ChartID(),this.NameObj(),OBJPROP_YDISTANCE);
   if(coord_y==y)
     {
      if(coord_y==GetProperty(CANV_ELEMENT_PROP_COORD_Y))
         return true;
      this.SetProperty(CANV_ELEMENT_PROP_COORD_Y,coord_y);
      return true;
     }
   if(::ObjectSetInteger(this.ChartID(),this.NameObj(),OBJPROP_YDISTANCE,coord_y))
     {
      this.SetProperty(CANV_ELEMENT_PROP_COORD_Y,coord_y);
      return true;
     }
   return false;
  }
//+------------------------------------------------------------------+
```

The method logic is similar to setting the X coordinate considered above.

**The method setting the new object width:**

```
//+------------------------------------------------------------------+
//| Set the new width                                                |
//+------------------------------------------------------------------+
bool CGCnvElement::SetWidth(const int width)
  {
   return this.m_canvas.Resize(width,this.m_canvas.Height());
  }
//+------------------------------------------------------------------+
```

The method receives the new width of the object and the result of calling the [Resize()](https://www.mql5.com/en/docs/standardlibrary/canvasgraphics/ccanvas/ccanvasresize) method resizing the graphical resource.

The Resize() method passes the new width and the current height of the object.

**The method setting the new height of the object:**

```
//+------------------------------------------------------------------+
//| Set the new height                                               |
//+------------------------------------------------------------------+
bool CGCnvElement::SetHeight(const int height)
  {
   return this.m_canvas.Resize(this.m_canvas.Width(),height);
  }
//+------------------------------------------------------------------+
```

The method receives the new height of the object and the result of calling the [Resize()](https://www.mql5.com/en/docs/standardlibrary/canvasgraphics/ccanvas/ccanvasresize) method resizing the graphical resource.

The Resize() method passes the current width and the new height of the object.

Note that when resizing the resource, the previous image drawn on the canvas is overwritten.

Therefore, these methods will be refined later.

**The method setting all shifts of the active area relative to the element:**

```
//+------------------------------------------------------------------+
//| Set all shifts of the active area relative to the element        |
//+------------------------------------------------------------------+
void CGCnvElement::SetActiveAreaShift(const int left_shift,const int bottom_shift,const int right_shift,const int top_shift)
  {
   this.SetActiveAreaLeftShift(left_shift);
   this.SetActiveAreaBottomShift(bottom_shift);
   this.SetActiveAreaRightShift(right_shift);
   this.SetActiveAreaTopShift(top_shift);
  }
//+------------------------------------------------------------------+
```

The method receives all values of the necessary inward shifts from the edges of the "graphical element" object. All four shifts are set one by one by calling the appropriate methods.

**The method setting the element opacity:**

```
//+------------------------------------------------------------------+
//| Set the element opacity                                          |
//+------------------------------------------------------------------+
void CGCnvElement::SetOpacity(const uchar value,const bool redraw=false)
  {
   this.m_canvas.TransparentLevelSet(value);
   this.SetProperty(CANV_ELEMENT_PROP_OPACITY,value);
   this.m_canvas.Update(redraw);
  }
//+------------------------------------------------------------------+
```

The method receives the required object opacity value (0 — completely transparent, 255 — completely opaque) and the chart redrawing flag.

Next, call the [TransparentLevelSet()](https://www.mql5.com/en/docs/standardlibrary/canvasgraphics/ccanvas/ccanvastransparentlevelset) method of the CCanvas class, write the new property value to the object properties and update the object with the passed redrawing flag.

The "graphic element" object is ready. Now we need the ability to sort these objects in the lists where they are to be stored. To achieve this, we need the CSelect class, in which we set the methods of sorting and searching all library objects.

Open \\MQL5\\Include\\DoEasy\\Services\ **Select.mqh** and add the inclusion of the "graphical element" object class file, as well as declaring the methods of sorting and searching for "graphical element" objects by their properties to the end of the class body:

```
//+------------------------------------------------------------------+
//|                                                       Select.mqh |
//|                        Copyright 2020, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2020, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include <Arrays\ArrayObj.mqh>
#include "..\Objects\Orders\Order.mqh"
#include "..\Objects\Events\Event.mqh"
#include "..\Objects\Accounts\Account.mqh"
#include "..\Objects\Symbols\Symbol.mqh"
#include "..\Objects\PendRequest\PendRequest.mqh"
#include "..\Objects\Series\SeriesDE.mqh"
#include "..\Objects\Indicators\Buffer.mqh"
#include "..\Objects\Indicators\IndicatorDE.mqh"
#include "..\Objects\Indicators\DataInd.mqh"
#include "..\Objects\Ticks\DataTick.mqh"
#include "..\Objects\Book\MarketBookOrd.mqh"
#include "..\Objects\MQLSignalBase\MQLSignal.mqh"
#include "..\Objects\Chart\ChartObj.mqh"
#include "..\Objects\Graph\GCnvElement.mqh"
//+------------------------------------------------------------------+
```

...

```
//+--------------------------------------------------------------------------+
//| The methods of working with data of the graphical elements on the canvas |
//+--------------------------------------------------------------------------+
   //--- Return the list of objects with one of (1) integer, (2) real and (3) string properties meeting a specified criterion
   static CArrayObj *ByGraphCanvElementProperty(CArrayObj *list_source,ENUM_CANV_ELEMENT_PROP_INTEGER property,long value,ENUM_COMPARER_TYPE mode);
   static CArrayObj *ByGraphCanvElementProperty(CArrayObj *list_source,ENUM_CANV_ELEMENT_PROP_DOUBLE property,double value,ENUM_COMPARER_TYPE mode);
   static CArrayObj *ByGraphCanvElementProperty(CArrayObj *list_source,ENUM_CANV_ELEMENT_PROP_STRING property,string value,ENUM_COMPARER_TYPE mode);
   //--- Return the chart index with the maximum value of the (1) integer, (2) real and (3) string properties
   static int        FindGraphCanvElementMax(CArrayObj *list_source,ENUM_CANV_ELEMENT_PROP_INTEGER property);
   static int        FindGraphCanvElementMax(CArrayObj *list_source,ENUM_CANV_ELEMENT_PROP_DOUBLE property);
   static int        FindGraphCanvElementMax(CArrayObj *list_source,ENUM_CANV_ELEMENT_PROP_STRING property);
   //--- Return the chart index with the minimum value of the (1) integer, (2) real and (3) string properties
   static int        FindGraphCanvElementMin(CArrayObj *list_source,ENUM_CANV_ELEMENT_PROP_INTEGER property);
   static int        FindGraphCanvElementMin(CArrayObj *list_source,ENUM_CANV_ELEMENT_PROP_DOUBLE property);
   static int        FindGraphCanvElementMin(CArrayObj *list_source,ENUM_CANV_ELEMENT_PROP_STRING property);
//---
  };
//+------------------------------------------------------------------+
```

At the end of the file listing, add the implementation of new declared methods:

```
//+------------------------------------------------------------------+
//+---------------------------------------------------------------------------+
//| The methods of working with data of the graphical elements on the canvas  |
//+---------------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| Return the list of objects with one integer                      |
//| property meeting the specified criterion                         |
//+------------------------------------------------------------------+
CArrayObj *CSelect::ByGraphCanvElementProperty(CArrayObj *list_source,ENUM_CANV_ELEMENT_PROP_INTEGER property,long value,ENUM_COMPARER_TYPE mode)
  {
   if(list_source==NULL) return NULL;
   CArrayObj *list=new CArrayObj();
   if(list==NULL) return NULL;
   list.FreeMode(false);
   ListStorage.Add(list);
   int total=list_source.Total();
   for(int i=0; i<total; i++)
     {
      CGCnvElement *obj=list_source.At(i);
      if(!obj.SupportProperty(property)) continue;
      long obj_prop=obj.GetProperty(property);
      if(CompareValues(obj_prop,value,mode)) list.Add(obj);
     }
   return list;
  }
//+------------------------------------------------------------------+
//| Return the list of objects with one real                         |
//| property meeting the specified criterion                         |
//+------------------------------------------------------------------+
CArrayObj *CSelect::ByGraphCanvElementProperty(CArrayObj *list_source,ENUM_CANV_ELEMENT_PROP_DOUBLE property,double value,ENUM_COMPARER_TYPE mode)
  {
   if(list_source==NULL) return NULL;
   CArrayObj *list=new CArrayObj();
   if(list==NULL) return NULL;
   list.FreeMode(false);
   ListStorage.Add(list);
   for(int i=0; i<list_source.Total(); i++)
     {
      CGCnvElement *obj=list_source.At(i);
      if(!obj.SupportProperty(property)) continue;
      double obj_prop=obj.GetProperty(property);
      if(CompareValues(obj_prop,value,mode)) list.Add(obj);
     }
   return list;
  }
//+------------------------------------------------------------------+
//| Return the list of objects with one string                       |
//| property meeting the specified criterion                         |
//+------------------------------------------------------------------+
CArrayObj *CSelect::ByGraphCanvElementProperty(CArrayObj *list_source,ENUM_CANV_ELEMENT_PROP_STRING property,string value,ENUM_COMPARER_TYPE mode)
  {
   if(list_source==NULL) return NULL;
   CArrayObj *list=new CArrayObj();
   if(list==NULL) return NULL;
   list.FreeMode(false);
   ListStorage.Add(list);
   for(int i=0; i<list_source.Total(); i++)
     {
      CGCnvElement *obj=list_source.At(i);
      if(!obj.SupportProperty(property)) continue;
      string obj_prop=obj.GetProperty(property);
      if(CompareValues(obj_prop,value,mode)) list.Add(obj);
     }
   return list;
  }
//+------------------------------------------------------------------+
//| Return the object index in the list                              |
//| with the maximum integer property value                          |
//+------------------------------------------------------------------+
int CSelect::FindGraphCanvElementMax(CArrayObj *list_source,ENUM_CANV_ELEMENT_PROP_INTEGER property)
  {
   if(list_source==NULL) return WRONG_VALUE;
   int index=0;
   CGCnvElement *max_obj=NULL;
   int total=list_source.Total();
   if(total==0) return WRONG_VALUE;
   for(int i=1; i<total; i++)
     {
      CGCnvElement *obj=list_source.At(i);
      long obj1_prop=obj.GetProperty(property);
      max_obj=list_source.At(index);
      long obj2_prop=max_obj.GetProperty(property);
      if(CompareValues(obj1_prop,obj2_prop,MORE)) index=i;
     }
   return index;
  }
//+------------------------------------------------------------------+
//| Return the object index in the list                              |
//| with the maximum real property value                             |
//+------------------------------------------------------------------+
int CSelect::FindGraphCanvElementMax(CArrayObj *list_source,ENUM_CANV_ELEMENT_PROP_DOUBLE property)
  {
   if(list_source==NULL) return WRONG_VALUE;
   int index=0;
   CGCnvElement *max_obj=NULL;
   int total=list_source.Total();
   if(total==0) return WRONG_VALUE;
   for(int i=1; i<total; i++)
     {
      CGCnvElement *obj=list_source.At(i);
      double obj1_prop=obj.GetProperty(property);
      max_obj=list_source.At(index);
      double obj2_prop=max_obj.GetProperty(property);
      if(CompareValues(obj1_prop,obj2_prop,MORE)) index=i;
     }
   return index;
  }
//+------------------------------------------------------------------+
//| Return the object index in the list                              |
//| with the maximum string property value                           |
//+------------------------------------------------------------------+
int CSelect::FindGraphCanvElementMax(CArrayObj *list_source,ENUM_CANV_ELEMENT_PROP_STRING property)
  {
   if(list_source==NULL) return WRONG_VALUE;
   int index=0;
   CGCnvElement *max_obj=NULL;
   int total=list_source.Total();
   if(total==0) return WRONG_VALUE;
   for(int i=1; i<total; i++)
     {
      CGCnvElement *obj=list_source.At(i);
      string obj1_prop=obj.GetProperty(property);
      max_obj=list_source.At(index);
      string obj2_prop=max_obj.GetProperty(property);
      if(CompareValues(obj1_prop,obj2_prop,MORE)) index=i;
     }
   return index;
  }
//+------------------------------------------------------------------+
//| Return the object index in the list                              |
//| with the minimum integer property value                          |
//+------------------------------------------------------------------+
int CSelect::FindGraphCanvElementMin(CArrayObj* list_source,ENUM_CANV_ELEMENT_PROP_INTEGER property)
  {
   int index=0;
   CGCnvElement *min_obj=NULL;
   int total=list_source.Total();
   if(total==0) return WRONG_VALUE;
   for(int i=1; i<total; i++)
     {
      CGCnvElement *obj=list_source.At(i);
      long obj1_prop=obj.GetProperty(property);
      min_obj=list_source.At(index);
      long obj2_prop=min_obj.GetProperty(property);
      if(CompareValues(obj1_prop,obj2_prop,LESS)) index=i;
     }
   return index;
  }
//+------------------------------------------------------------------+
//| Return the object index in the list                              |
//| with the minimum real property value                             |
//+------------------------------------------------------------------+
int CSelect::FindGraphCanvElementMin(CArrayObj* list_source,ENUM_CANV_ELEMENT_PROP_DOUBLE property)
  {
   int index=0;
   CGCnvElement *min_obj=NULL;
   int total=list_source.Total();
   if(total== 0) return WRONG_VALUE;
   for(int i=1; i<total; i++)
     {
      CGCnvElement *obj=list_source.At(i);
      double obj1_prop=obj.GetProperty(property);
      min_obj=list_source.At(index);
      double obj2_prop=min_obj.GetProperty(property);
      if(CompareValues(obj1_prop,obj2_prop,LESS)) index=i;
     }
   return index;
  }
//+------------------------------------------------------------------+
//| Return the object index in the list                              |
//| with the minimum string property value                           |
//+------------------------------------------------------------------+
int CSelect::FindGraphCanvElementMin(CArrayObj* list_source,ENUM_CANV_ELEMENT_PROP_STRING property)
  {
   int index=0;
   CGCnvElement *min_obj=NULL;
   int total=list_source.Total();
   if(total==0) return WRONG_VALUE;
   for(int i=1; i<total; i++)
     {
      CGCnvElement *obj=list_source.At(i);
      string obj1_prop=obj.GetProperty(property);
      min_obj=list_source.At(index);
      string obj2_prop=min_obj.GetProperty(property);
      if(CompareValues(obj1_prop,obj2_prop,LESS)) index=i;
     }
   return index;
  }
//+------------------------------------------------------------------+
```

The methods are described [in the third article](https://www.mql5.com/en/articles/5687#node01) where we discussed creating the CSelect class.

**Let's test the results.**

### Test

To perform the test, let's use the EA from the previous article and save it in \\MQL5\\Experts\\TestDoEasy\ **Part74\** as **TestDoEasyPart74.mq5**.

Include the file [with the class of the dynamic array of pointers to the instances of the CObject class and its descendants](https://www.mql5.com/en/docs/standardlibrary/datastructures/carrayobj), the standard library, CSelect and CGCnvElement library class files, specify the number of created "graphical element" objects and declare the list to store created graphical elements:

```
//+------------------------------------------------------------------+
//|                                             TestDoEasyPart74.mq5 |
//|                                  Copyright 2021, MetaQuotes Ltd. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2021, MetaQuotes Ltd."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//--- includes
#include <Arrays\ArrayObj.mqh>
#include <DoEasy\Services\Select.mqh>
#include <DoEasy\Objects\Graph\GCnvElement.mqh>
//--- defines
#define        FORMS_TOTAL (2)
//--- input parameters
sinput   bool  InpMovable  = true;  // Movable flag
//--- global variables
CArrayObj      list_elements;
//+------------------------------------------------------------------+
```

In the EA's **OnInit()** handler, create new graphical element objects by passing all the necessary parameters to the class constructor:

```
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//--- Set the permissions to send cursor movement and mouse scroll events
   ChartSetInteger(ChartID(),CHART_EVENT_MOUSE_MOVE,true);
   ChartSetInteger(ChartID(),CHART_EVENT_MOUSE_WHEEL,true);
//--- Set EA global variables

//--- Create the specified number of graphical elements on the canvas
   int total=FORMS_TOTAL;
   for(int i=0;i<total;i++)
     {
      //--- When creating an object, pass all the required parameters to it
      CGCnvElement *element=new CGCnvElement(GRAPH_ELEMENT_TYPE_ELEMENT,i,0,ChartID(),0,"Element_0"+(string)(i+1),300,40+(i*80),100,70,clrSilver,200,InpMovable,true,true);
      if(element==NULL)
         continue;
      //--- Add objects to the list
      if(!list_elements.Add(element))
        {
         delete element;
         continue;
        }
     }
//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
```

In the **OnDeinit()** handler, remove all comments from the chart:

```
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//--- destroy timer
   EventKillTimer();
   Comment("");
  }
//+------------------------------------------------------------------+
```

In the **OnChartEvent()** handler, capture the click on the object, get the element object with the name corresponding to the name of the clicked object set in the sparam handler parameter and increase its opacity level by 5. Display the message with the handled object name and opacity level in the chart comment:

```
//+------------------------------------------------------------------+
//| ChartEvent function                                              |
//+------------------------------------------------------------------+
void OnChartEvent(const int id,
                  const long &lparam,
                  const double &dparam,
                  const string &sparam)
  {
//--- If clicking on an object
   if(id==CHARTEVENT_OBJECT_CLICK)
     {
      //--- In the new list, get the element object with the name corresponding to the sparam string parameter value of the OnChartEvent() handler
      CArrayObj *obj_list=CSelect::ByGraphCanvElementProperty(GetPointer(list_elements),CANV_ELEMENT_PROP_NAME_OBJ,sparam,EQUAL);
      if(obj_list!=NULL && obj_list.Total()>0)
        {
         //--- Get the pointer to the object in the list
         CGCnvElement *obj=obj_list.At(0);
         //--- and set the new opacity level for it
         uchar opasity=obj.Opacity();
         if((opasity+5)>255)
            opasity=0;
         else
            opasity+=5;
         //--- Set the new opacity to the object and display the object name and opacity level in the journal
         obj.SetOpacity(opasity);
         Comment(DFUN,"Object name: ",obj.NameObj(),", opasity=",opasity);
        }
     }
  }
//+------------------------------------------------------------------+
```

Compile the EA and launch it on a symbol chart. When clicking on any "graphical element" object, its opacity is increased up to 255, then, upon reaching the maximum value (255), it is increased from 0 to 255, while the name of the clicked object and its opacity level are displayed in the chart comment:

![](https://c.mql5.com/2/42/kFt0y515KB.gif)

### What's next?

In the next article, I will continue the development of the "graphical element" object and start adding methods for displaying graphical primitives and text on it.

All files of the current version of the library are attached below together with the test EA file for MQL5 for you to test and download.

Leave your questions and suggestions in the comments.

[Back to contents](https://www.mql5.com/en/articles/9493#node00)

**\*Previous articles within the series:**

[Graphics in DoEasy library (Part 73): Form object of a graphical element](https://www.mql5.com/en/articles/9442)

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/9493](https://www.mql5.com/ru/articles/9493)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/9493.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/9493/mql5.zip "Download MQL5.zip")(3957.08 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Tables in the MVC Paradigm in MQL5: Customizable and sortable table columns](https://www.mql5.com/en/articles/19979)
- [How to publish code to CodeBase: A practical guide](https://www.mql5.com/en/articles/19441)
- [Tables in the MVC Paradigm in MQL5: Integrating the Model Component into the View Component](https://www.mql5.com/en/articles/19288)
- [The View and Controller components for tables in the MQL5 MVC paradigm: Resizable elements](https://www.mql5.com/en/articles/18941)
- [The View and Controller components for tables in the MQL5 MVC paradigm: Containers](https://www.mql5.com/en/articles/18658)
- [The View and Controller components for tables in the MQL5 MVC paradigm: Simple controls](https://www.mql5.com/en/articles/18221)
- [The View component for tables in the MQL5 MVC paradigm: Base graphical element](https://www.mql5.com/en/articles/17960)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/372173)**
(3)


![Jagg](https://c.mql5.com/avatar/avatar_na2.png)

**[Jagg](https://www.mql5.com/en/users/jagg)**
\|
28 Jun 2021 at 13:49

I'm looking forward to the next CCanvas article(s) - great series - thanks for that...


![Artyom Trishkin](https://c.mql5.com/avatar/2022/7/62C4775C-ABD6.jpg)

**[Artyom Trishkin](https://www.mql5.com/en/users/artmedia70)**
\|
28 Jun 2021 at 14:32

**Jagg :**

I'm looking forward to the next CCanvas article(s) - great series - thanks for that...

There are already 77 articles in the RU segment of MQL5.com. Articles are first published in the Russian segment of the resource, and then translated into other languages.

[75](https://www.mql5.com/ru/articles/9515) , [76](https://www.mql5.com/ru/articles/9553) , [77](https://www.mql5.com/ru/articles/9575) .

They have not yet been published in the EN-segment, you can familiarize yourself using the translator.

![Jagg](https://c.mql5.com/avatar/avatar_na2.png)

**[Jagg](https://www.mql5.com/en/users/jagg)**
\|
19 Jul 2021 at 08:37

**Artyom Trishkin:**

There are already 77 articles in the RU segment of MQL5.com. Articles are first published in the Russian segment of the resource, and then translated into other languages.

[75](https://www.mql5.com/ru/articles/9515) , [76](https://www.mql5.com/ru/articles/9553) , [77](https://www.mql5.com/ru/articles/9575) .

They have not yet been published in the EN-segment, you can familiarize yourself using the translator.

Thanks for the info!

![Graphics in DoEasy library (Part 75): Methods of handling primitives and text in the basic graphical element](https://c.mql5.com/2/42/MQL5-avatar-doeasy-library3-2__2.png)[Graphics in DoEasy library (Part 75): Methods of handling primitives and text in the basic graphical element](https://www.mql5.com/en/articles/9515)

In this article, I will continue the development of the basic graphical element class of all library graphical objects powered by the CCanvas Standard Library class. I will create the methods for drawing graphical primitives and for displaying a text on a graphical element object.

![Graphics in DoEasy library (Part 73): Form object of a graphical element](https://c.mql5.com/2/42/MQL5-avatar-doeasy-library3-2.png)[Graphics in DoEasy library (Part 73): Form object of a graphical element](https://www.mql5.com/en/articles/9442)

The article opens up a new large section of the library for working with graphics. In the current article, I will create the mouse status object, the base object of all graphical elements and the class of the form object of the library graphical elements.

![Graphics in DoEasy library (Part 76): Form object and predefined color themes](https://c.mql5.com/2/42/MQL5-avatar-doeasy-library3-2__3.png)[Graphics in DoEasy library (Part 76): Form object and predefined color themes](https://www.mql5.com/en/articles/9553)

In this article, I will describe the concept of building various library GUI design themes, create the Form object, which is a descendant of the graphical element class object, and prepare data for creating shadows of the library graphical objects, as well as for further development of the functionality.

![Cluster analysis (Part I): Mastering the slope of indicator lines](https://c.mql5.com/2/42/mql5-avatar-cluster_analysis.png)[Cluster analysis (Part I): Mastering the slope of indicator lines](https://www.mql5.com/en/articles/9527)

Cluster analysis is one of the most important elements of artificial intelligence. In this article, I attempt applying the cluster analysis of the indicator slope to get threshold values for determining whether a market is flat or following a trend.

[![](https://www.mql5.com/ff/sh/9nb0c8df2rmwfn89z2/01.png) MetaTrader VPS vs regular cloud hosting services8 reasons why our solution is the best option for automated tradingRead](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/450486&a=dgmsfszgoedimaicrqqmagvqzpwuxkur&s=c59e3617ccf44fd54d4c50a03b44fd689ff7507b8fe4990c83772cc5419e627d&uid=&ref=https://www.mql5.com/en/articles/9493&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5083397110050593525)

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