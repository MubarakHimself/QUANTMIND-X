---
title: Graphics in DoEasy library (Part 76): Form object and predefined color themes
url: https://www.mql5.com/en/articles/9553
categories: Expert Advisors
relevance_score: 0
scraped_at: 2026-01-24T14:11:30.512315
---

[![](https://www.mql5.com/ff/sh/0wxx5f0vuwq7xh89z2/01.png)VPS for 24/7 tradingContact your broker and find out how to get a free hosting subscriptionLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=nhetzvgituppcfrhndpblbihmzziogdh&s=d00c975c8bda3d8c1b29f042ad33ac81952ccea2f130a8f1ffa9015bab8ade87&uid=&ref=https://www.mql5.com/en/articles/9553&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5083393004061858528)

MetaTrader 5 / Examples


### Contents

- [Concept](https://www.mql5.com/en/articles/9553#node01)
- [Improving library classes](https://www.mql5.com/en/articles/9553#node02)
- [Color themes and form types](https://www.mql5.com/en/articles/9553#node03)
- [Form object class](https://www.mql5.com/en/articles/9553#node04)
- [Test](https://www.mql5.com/en/articles/9553#node05)
- [What's next?](https://www.mql5.com/en/articles/9553#node06)


### Concept

In the previous article, I have developed the basic graphical element object class used as a basis for creating more complex library graphical objects, as well as created the methods for drawing graphical primitives and texts. Today, I will use this graphical element object as a basis for its descendant object class — form object. The form object can already be an absolutely independent unit for designing and presenting some controls and performing visualization in programs based on the library.

But before making the form object, let's talk about GUI and its design methods, as well as create a starting set of color themes and types of graphical object presentation.

Many programs using graphical data representation and providing interaction with the outside world via their graphics engine allow users to quickly change the appearance and design of their graphical objects. I will use a set of themes to quickly change the appearance and color scheme. The parameters of created themes will be contained in a separate library file, in which a program user or a programmer can quickly change various settings for the graphical object appearance and color.

In the current article, I will start the development of two themes, which will gradually feature more and more different parameters and their values as I develop new library objects and functionality.

There is no need to use the themes developed in the library for creating custom graphical objects, but they can serve as a good example of how exactly to make certain objects for their subsequent usage.

### Improving library classes

As usual, add the indices of new messages to \\MQL5\\Include\\DoEasy\ **Data.mqh**:

```
   MSG_LIB_SYS_FAILED_ADD_SYM_OBJ,                    // Failed to add symbol
   MSG_LIB_SYS_FAILED_CREATE_ELM_OBJ,                 // Failed to create the graphical element object
   MSG_LIB_SYS_OBJ_ALREADY_IN_LIST,                   // Such an object is already present in the list
   MSG_LIB_SYS_FAILED_GET_DATA_GRAPH_RES,             // Failed to receive graphical resource data
```

...

```
   MSG_LIB_SYS_FAILED_ADD_BUFFER,                     // Failed to add buffer object to the list
   MSG_LIB_SYS_FAILED_CREATE_BUFFER_OBJ,              // Failed to create \"Indicator buffer\" object
   MSG_LIB_SYS_FAILED_OBJ_ADD_TO_LIST,                // Could not add object to the list
```

and message texts corresponding to newly added indices:

```
   {"Не удалось добавить символ ","Failed to add "},
   {"Не удалось создать объект-графический элемент ","Failed to create graphic element object "},
   {"Такой объект уже есть в списке","Such an object is already in the list"},
   {"Не удалось получить данные графического ресурса","Failed to get graphic resource data"},
```

...

```
   {"Не удалось добавить объект-буфер в список","Failed to add buffer object to list"},
   {"Не удалось создать объект \"Индикаторный буфер\"","Failed to create object \"Indicator buffer\""},
   {"Не удалось добавить объект в список","Failed to add object to the list"},
```

While developing the form object, I will make a blank for the subsequent creation of shadows cast by the form on the objects, above which the form is located. The form needs a small space around it. This space is to be used to draw the shadow. To set the size of the space, we need a macro substitution indicating the size of one side of this space in pixels. If we set five pixels, the form will have free space at the top, bottom, left and right — five pixels on each side.

Handling the graphical element object shows that some of its properties are not needed in the property list. They will not be used for searching and sorting objects. Therefore, they should be removed from the enumeration of the element object integer properties. They will be contained in ordinary protected class member variables.

Let's open \\MQL5\\Include\\DoEasy\ **Defines.mqh** and implement the above mentioned changes:

In the list of canvas parameters, add the offset of one side to insert shadows:

```
//--- Parameters of the DOM snapshot series
#define MBOOKSERIES_DEFAULT_DAYS_COUNT (1)                        // The default required number of days for DOM snapshots in the series
#define MBOOKSERIES_MAX_DATA_TOTAL     (200000)                   // Maximum number of stored DOM snapshots of a single symbol
//--- Canvas parameters
#define PAUSE_FOR_CANV_UPDATE          (16)                       // Canvas update frequency
#define NULL_COLOR                     (0x00FFFFFF)               // Zero for the canvas with the alpha channel
#define OUTER_AREA_SIZE                (5)                        // Size of one side of the outer area around the workspace
//+------------------------------------------------------------------+
//| Enumerations                                                     |
//+------------------------------------------------------------------+
```

Remove two unnecessary properties from the list of integer properties of the graphical element:

```
   CANV_ELEMENT_PROP_ACT_SHIFT_BOTTOM,                // Active area offset from the bottom edge of the element
   CANV_ELEMENT_PROP_OPACITY,                         // Element opacity
   CANV_ELEMENT_PROP_COLOR_BG,                        // Element background color
   CANV_ELEMENT_PROP_MOVABLE,                         // Element moveability flag
```

Now the list of integer properties will look like this:

```
//+------------------------------------------------------------------+
//| Integer properties of the graphical element on the canvas        |
//+------------------------------------------------------------------+
enum ENUM_CANV_ELEMENT_PROP_INTEGER
  {
   CANV_ELEMENT_PROP_ID = 0,                          // Element ID
   CANV_ELEMENT_PROP_TYPE,                            // Graphical element type
   CANV_ELEMENT_PROP_NUM,                             // Element index in the list
   CANV_ELEMENT_PROP_CHART_ID,                        // Chart ID
   CANV_ELEMENT_PROP_WND_NUM,                         // Chart subwindow index
   CANV_ELEMENT_PROP_COORD_X,                         // Form's X coordinate on the chart
   CANV_ELEMENT_PROP_COORD_Y,                         // Form's Y coordinate on the chart
   CANV_ELEMENT_PROP_WIDTH,                           // Element width
   CANV_ELEMENT_PROP_HEIGHT,                          // Element height
   CANV_ELEMENT_PROP_RIGHT,                           // Element right border
   CANV_ELEMENT_PROP_BOTTOM,                          // Element bottom border
   CANV_ELEMENT_PROP_ACT_SHIFT_LEFT,                  // Active area offset from the left edge of the element
   CANV_ELEMENT_PROP_ACT_SHIFT_TOP,                   // Active area offset from the upper edge of the element
   CANV_ELEMENT_PROP_ACT_SHIFT_RIGHT,                 // Active area offset from the right edge of the element
   CANV_ELEMENT_PROP_ACT_SHIFT_BOTTOM,                // Active area offset from the bottom edge of the element
   CANV_ELEMENT_PROP_MOVABLE,                         // Element moveability flag
   CANV_ELEMENT_PROP_ACTIVE,                          // Element activity flag
   CANV_ELEMENT_PROP_COORD_ACT_X,                     // X coordinate of the element active area
   CANV_ELEMENT_PROP_COORD_ACT_Y,                     // Y coordinate of the element active area
   CANV_ELEMENT_PROP_ACT_RIGHT,                       // Right border of the element active area
   CANV_ELEMENT_PROP_ACT_BOTTOM,                      // Bottom border of the element active area
  };
#define CANV_ELEMENT_PROP_INTEGER_TOTAL (21)          // Total number of integer properties
#define CANV_ELEMENT_PROP_INTEGER_SKIP  (0)           // Number of integer properties not used in sorting
//+------------------------------------------------------------------+
```

Reduce the total number of integer properties by 2 — set **21** instead of 23.

Also, remove two unnecessary constants from the enumeration of possible canvas-based graphical element sorting criteria:

```
   SORT_BY_CANV_ELEMENT_ACT_SHIFT_BOTTOM,             // Sort by the active area offset from the bottom edge of the element
   SORT_BY_CANV_ELEMENT_OPACITY,                      // Sort by the element opacity
   SORT_BY_CANV_ELEMENT_COLOR_BG,                     // Sort by the element background color
   SORT_BY_CANV_ELEMENT_MOVABLE,                      // Sort by the element moveability flag
```

The complete list is to be as follows:

```
//+------------------------------------------------------------------+
//| Possible sorting criteria of graphical elements on the canvas    |
//+------------------------------------------------------------------+
#define FIRST_CANV_ELEMENT_DBL_PROP  (CANV_ELEMENT_PROP_INTEGER_TOTAL-CANV_ELEMENT_PROP_INTEGER_SKIP)
#define FIRST_CANV_ELEMENT_STR_PROP  (CANV_ELEMENT_PROP_INTEGER_TOTAL-CANV_ELEMENT_PROP_INTEGER_SKIP+CANV_ELEMENT_PROP_DOUBLE_TOTAL-CANV_ELEMENT_PROP_DOUBLE_SKIP)
enum ENUM_SORT_CANV_ELEMENT_MODE
  {
//--- Sort by integer properties
   SORT_BY_CANV_ELEMENT_ID = 0,                       // Sort by element ID
   SORT_BY_CANV_ELEMENT_TYPE,                         // Sort by graphical element type
   SORT_BY_CANV_ELEMENT_NUM,                          // Sort by form index in the list
   SORT_BY_CANV_ELEMENT_CHART_ID,                     // Sort by chart ID
   SORT_BY_CANV_ELEMENT_WND_NUM,                      // Sort by chart window index
   SORT_BY_CANV_ELEMENT_COORD_X,                      // Sort by the element X coordinate on the chart
   SORT_BY_CANV_ELEMENT_COORD_Y,                      // Sort by the element Y coordinate on the chart
   SORT_BY_CANV_ELEMENT_WIDTH,                        // Sort by the element width
   SORT_BY_CANV_ELEMENT_HEIGHT,                       // Sort by the element height
   SORT_BY_CANV_ELEMENT_RIGHT,                        // Sort by the element right border
   SORT_BY_CANV_ELEMENT_BOTTOM,                       // Sort by the element bottom border
   SORT_BY_CANV_ELEMENT_ACT_SHIFT_LEFT,               // Sort by the active area offset from the left edge of the element
   SORT_BY_CANV_ELEMENT_ACT_SHIFT_TOP,                // Sort by the active area offset from the top edge of the element
   SORT_BY_CANV_ELEMENT_ACT_SHIFT_RIGHT,              // Sort by the active area offset from the right edge of the element
   SORT_BY_CANV_ELEMENT_ACT_SHIFT_BOTTOM,             // Sort by the active area offset from the bottom edge of the element
   SORT_BY_CANV_ELEMENT_MOVABLE,                      // Sort by the element moveability flag
   SORT_BY_CANV_ELEMENT_ACTIVE,                       // Sort by the element activity flag
   SORT_BY_CANV_ELEMENT_COORD_ACT_X,                  // Sort by X coordinate of the element active area
   SORT_BY_CANV_ELEMENT_COORD_ACT_Y,                  // Sort by Y coordinate of the element active area
   SORT_BY_CANV_ELEMENT_ACT_RIGHT,                    // Sort by the right border of the element active area
   SORT_BY_CANV_ELEMENT_ACT_BOTTOM,                   // Sort by the bottom border of the element active area
//--- Sort by real properties

//--- Sort by string properties
   SORT_BY_CANV_ELEMENT_NAME_OBJ = FIRST_CANV_ELEMENT_STR_PROP,// Sort by an element object name
   SORT_BY_CANV_ELEMENT_NAME_RES,                     // Sort by the graphical resource name
  };
//+------------------------------------------------------------------+
```

All created graphical objects are based on the graphical element object. The object itself is a descendant of the basic object of all library graphical objects (which, in turn, is a [descendant of the CObject](https://www.mql5.com/en/docs/standardlibrary/cobject) Standard Library basic class). All properties of each parent class are handed down to its descendants. Therefore, if we need properties that are common for all graphical objects, they should be located in the basic objects of the entire derivation tree. The [CGBaseObj class object](https://www.mql5.com/en/articles/9442#node04) serves as an object for graphical library objects.

We will have to manage the visibility of graphical objects on the chart. We do not need to remove or hide the graphical object for that. All we have to do is specify the necessary flags in the [OBJPROP\_TIMEFRAMES](https://www.mql5.com/en/docs/constants/objectconstants/visible) property of the graphical object so that it is removed or shown on the chart on top of all others. Thus, we will be able to manage the object visibility on the chart, as well as place the required object above all others. It will serve as the current object the program user is to work with.

Out of the entire [set of object flags](https://www.mql5.com/en/docs/constants/objectconstants/visible), we will need the following ones: OBJ\_NO\_PERIODS — for hiding the object and OBJ\_ALL\_PERIODS — for displaying the object on a chart. To move the object to the foreground, simply hide and then show it again.

Let's add new properties and methods to the \\MQL5\\Include\\DoEasy\\Objects\\Graph\ **GBaseObj.mqh** base object file.

In the protected section of the class, declare the variable for storing the object visibility property:

```
//+------------------------------------------------------------------+
//| Class of the base object of the library graphical objects        |
//+------------------------------------------------------------------+
class CGBaseObj : public CObject
  {
private:

protected:
   string            m_name_prefix;                      // Object name prefix
   string            m_name;                             // Object name
   long              m_chart_id;                         // Chart ID
   int               m_subwindow;                        // Subwindow index
   int               m_shift_y;                          // Subwindow Y coordinate shift
   int               m_type;                             // Object type
   bool              m_visible;                          // Object visibility

//--- Create (1) the object structure and (2) the object from the structure
   virtual bool      ObjectToStruct(void)                      { return true; }
   virtual void      StructToObject(void){;}

public:
```

In the public section of the class, write the method for setting the object visibility flag, as well as simultaneous setting of the property itself to the object and the method for returning the object visibility on the chart:

```
public:
//--- Return the values of class variables
   string            Name(void)                          const { return this.m_name;      }
   long              ChartID(void)                       const { return this.m_chart_id;  }
   int               SubWindow(void)                     const { return this.m_subwindow; }
//--- (1) Set and (2) return the object visibility
   void              SetVisible(const bool flag)
                       {
                        long value=(flag ? OBJ_ALL_PERIODS : 0);
                        if(::ObjectSetInteger(this.m_chart_id,this.m_name,OBJPROP_TIMEFRAMES,value))
                           this.m_visible=flag;
                       }
   bool              IsVisible(void)                     const { return this.m_visible;   }

//--- The virtual method returning the object type
   virtual int       Type(void)                          const { return this.m_type;      }
```

The method setting the object visibility first checks the flag value. Depending on the passed value (true or false), it sends a request for setting a value to the object, or OBJ\_ALL\_PERIODS for displaying an object on a chart, or 0 for hiding it. If a request is successfully placed in the chart events queue, the **m\_visible** variable receives the flag value passed to the method. The value can be revealed using the IsVisible() method returning the variable value.

In the initialization list of the class constructor, initialize a new variable using false:

```
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CGBaseObj::CGBaseObj() : m_shift_y(0), m_type(0),m_visible(false), m_name_prefix(::MQLInfoString(MQL_PROGRAM_NAME)+"_")
  {
  }
//+------------------------------------------------------------------+
```

Let's improve the graphical element object class in \\MQL5\\Include\\DoEasy\\Objects\\Graph\ **GCnvElement.mqh**.

In the protected section of the class, declare the flag variable, in which the object shadow presence/absence is set, and the variable for storing the chart background color — we will need it in the future when drawing shadows:

```
//+------------------------------------------------------------------+
//| Class of the base object of the library graphical objects        |
//+------------------------------------------------------------------+
class CGCnvElement : public CGBaseObj
  {
protected:
   CCanvas           m_canvas;                                 // CCanvas class object
   CPause            m_pause;                                  // Pause class object
   bool              m_shadow;                                 // Shadow presence
   color             m_chart_color_bg;                         // Chart background color
//--- Return the cursor position relative to the (1) entire element and (2) the element's active area
   bool              CursorInsideElement(const int x,const int y);
   bool              CursorInsideActiveArea(const int x,const int y);
//--- Create (1) the object structure and (2) the object from the structure
   virtual bool      ObjectToStruct(void);
   virtual void      StructToObject(void);

private:
```

Since we have removed two constants from the enumeration of the object integer properties, now we need to store them in the class variables.

Declare them in the private section:

```
   long              m_long_prop[ORDER_PROP_INTEGER_TOTAL];    // Integer properties
   double            m_double_prop[ORDER_PROP_DOUBLE_TOTAL];   // Real properties
   string            m_string_prop[ORDER_PROP_STRING_TOTAL];   // String properties

   ENUM_TEXT_ANCHOR  m_text_anchor;                            // Current text alignment
   color             m_color_bg;                               // Element background color
   uchar             m_opacity;                                // Element opacity

//--- Return the index of the array the order's (1) double and (2) string properties are located at
```

The "element background color" and "element opacity" are now set and read in these variables.

To design the appearance of graphical objects, we need a method that allows changing the color lightness.

This is one of the components of the HSL color model:

**HSL**, **HLS** or **HSI** (hue, saturation, lightness (intensity)) is a color model, in which hue, saturation and lightness are used as color coordinates. HSV and HSL are two different color models ( _lightness_ should not be confused with brightness).

When drawing graphical primitives, we need to lighten the conventionally lit parts of the drawing and darken the conventionally shaded parts. The color of the image itself should not be affected. To ensure this, I will apply the method converting the ARGB color model into HSL and change the brightness of pixels of the desired part of the image.

Declare this method in the private section of the class:

```
//--- Update the coordinates (shift the canvas)
   bool              Move(const int x,const int y,const bool redraw=false);

//--- Change the color lightness by the specified amount
   uint              ChangeColorLightness(const uint clr,const double change_value);

protected:
```

The graphical element object is a main object for creating more complex graphical objects that will be its descendants. Keeping the concept of constructing library objects (where the parent class features the protected constructor specifying the parameters of the created descendant object) in mind, we need to create a protected parametric constructor for the element object. It is to receive clarifying parameters about the descendant object type to be created on the basis of the graphical element (here it is form object).

In the protected section of the class, declare a new protected parametric constructor:

```
protected:
//--- Protected constructor
                     CGCnvElement(const ENUM_GRAPH_ELEMENT_TYPE element_type,
                                  const long    chart_id,
                                  const int     wnd_num,
                                  const string  name,
                                  const int     x,
                                  const int     y,
                                  const int     w,
                                  const int     h);
public:
```

It is to receive only basic parameters for creating the object. All other parameters are set only after its successful creation. This will be done in the library graphical object collection class, which I have not yet started to do.

Add the shadow presence flag initialization and chart background color to the default (non-parametric) constructor, namely to its initialization list:

```
public:
//--- Parametric constructor
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
//--- Default constructor/Destructor
                     CGCnvElement() : m_shadow(false),m_chart_color_bg((color)::ChartGetInteger(::ChartID(),CHART_COLOR_BACKGROUND)) {;}
```

Add the new methods for setting object properties in the blocks of methods for a simplified access to object parameters:

```
//+------------------------------------------------------------------+
//| Methods of simplified access to object properties                |
//+------------------------------------------------------------------+
//--- Set the (1) X, (2) Y coordinates, (3) element width, (4) height, (5) right (6) and bottom edge,
   bool              SetCoordX(const int coord_x);
   bool              SetCoordY(const int coord_y);
   bool              SetWidth(const int width);
   bool              SetHeight(const int height);
   void              SetRightEdge(void)                        { this.SetProperty(CANV_ELEMENT_PROP_RIGHT,this.RightEdge());           }
   void              SetBottomEdge(void)                       { this.SetProperty(CANV_ELEMENT_PROP_BOTTOM,this.BottomEdge());         }
//--- Set the shift of the (1) left, (2) top, (3) right, (4) bottom edge of the active area relative to the element,
//--- (5) all shifts of the active area edges relative to the element, (6) the element background color and (7) the element opacity
   void              SetActiveAreaLeftShift(const int value)   { this.SetProperty(CANV_ELEMENT_PROP_ACT_SHIFT_LEFT,fabs(value));       }
   void              SetActiveAreaRightShift(const int value)  { this.SetProperty(CANV_ELEMENT_PROP_ACT_SHIFT_RIGHT,fabs(value));      }
   void              SetActiveAreaTopShift(const int value)    { this.SetProperty(CANV_ELEMENT_PROP_ACT_SHIFT_TOP,fabs(value));        }
   void              SetActiveAreaBottomShift(const int value) { this.SetProperty(CANV_ELEMENT_PROP_ACT_SHIFT_BOTTOM,fabs(value));     }
   void              SetActiveAreaShift(const int left_shift,const int bottom_shift,const int right_shift,const int top_shift);
   void              SetColorBackground(const color colour)    { this.m_color_bg=colour;                                               }
   void              SetOpacity(const uchar value,const bool redraw=false);

//--- Set the flag of (1) object moveability, (2) activity, (3) element ID, (4) element index in the list and (5) shadow presence
   void              SetMovable(const bool flag)               { this.SetProperty(CANV_ELEMENT_PROP_MOVABLE,flag);                     }
   void              SetActive(const bool flag)                { this.SetProperty(CANV_ELEMENT_PROP_ACTIVE,flag);                      }
   void              SetID(const int id)                       { this.SetProperty(CANV_ELEMENT_PROP_ID,id);                            }
   void              SetNumber(const int number)               { this.SetProperty(CANV_ELEMENT_PROP_NUM,number);                       }
   void              SetShadow(const bool flag);

//--- Return the shift (1) of the left, (2) right, (3) top and (4) bottom edge of the element active area
```

The methods for returning the background color and opacity now return the values set in the new declared variables:

```
//--- Return (1) the background color, (2) the opacity, coordinate (3) of the right and (4) bottom element edge
   color             ColorBackground(void)               const { return this.m_color_bg;                                               }
   uchar             Opacity(void)                       const { return this.m_opacity;                                                }
   int               RightEdge(void)                     const { return this.CoordX()+this.m_canvas.Width();                           }
   int               BottomEdge(void)                    const { return this.CoordY()+this.m_canvas.Height();                          }
//--- Return the (1) X, (2) Y coordinates, (3) element width and (4) height,
```

At the end of the list, add the method for returning the flag of drawing the shadow cast by the object, the method returning the chart background color

and the method moving the object to the foreground (above all other graphical objects on the charts):

```
//--- Return (1) the element ID, (2) element index in the list, (3) flag of the form shadow presence and (4) the chart background color
   int               ID(void)                            const { return (int)this.GetProperty(CANV_ELEMENT_PROP_ID);                   }
   int               Number(void)                        const { return (int)this.GetProperty(CANV_ELEMENT_PROP_NUM);                  }
   bool              IsShadow(void)                      const { return this.m_shadow;                                                 }
   color             ChartColorBackground(void)          const { return this.m_chart_color_bg;                                         }
//--- Set the object above all
   void              BringToTop(void)                          { CGBaseObj::SetVisible(false); CGBaseObj::SetVisible(true);            }
//+------------------------------------------------------------------+
```

As we can see, in order to set the object above all others, we should hideand immediately show it again using the parent class methods considered above.

Delete the strings that set the now removed properties from the parametric constructor:

```
      this.SetProperty(CANV_ELEMENT_PROP_ACT_SHIFT_BOTTOM,0);                    // Active area offset from the bottom edge of the element
      this.SetProperty(CANV_ELEMENT_PROP_OPACITY,opacity);                       // Element opacity
      this.SetProperty(CANV_ELEMENT_PROP_COLOR_BG,colour);                       // Element color
      this.SetProperty(CANV_ELEMENT_PROP_MOVABLE,movable);                       // Element moveability flag
```

Now these, as well as new properties, are set in the new variables:

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
                           const bool     redraw=false) : m_shadow(false)

  {
   this.m_chart_color_bg=(color)::ChartGetInteger(chart_id,CHART_COLOR_BACKGROUND);
   this.m_name=this.m_name_prefix+name;
   this.m_chart_id=chart_id;
   this.m_subwindow=wnd_num;
   this.m_type=element_type;
   this.SetFont("Calibri",8);
   this.m_text_anchor=0;
   this.m_color_bg=colour;
   this.m_opacity=opacity;
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

The new protected paramteric constructor does not differ from the one considered above:

```
//+------------------------------------------------------------------+
//| Protected constructor                                            |
//+------------------------------------------------------------------+
CGCnvElement::CGCnvElement(const ENUM_GRAPH_ELEMENT_TYPE element_type,
                           const long    chart_id,
                           const int     wnd_num,
                           const string  name,
                           const int     x,
                           const int     y,
                           const int     w,
                           const int     h) : m_shadow(false)
  {
   this.m_chart_color_bg=(color)::ChartGetInteger(chart_id,CHART_COLOR_BACKGROUND);
   this.m_name=this.m_name_prefix+name;
   this.m_chart_id=chart_id;
   this.m_subwindow=wnd_num;
   this.m_type=element_type;
   this.SetFont("Calibri",8);
   this.m_text_anchor=0;
   this.m_color_bg=NULL_COLOR;
   this.m_opacity=0;
   if(this.Create(chart_id,wnd_num,this.m_name,x,y,w,h,this.m_color_bg,this.m_opacity,false))
     {
      this.SetProperty(CANV_ELEMENT_PROP_NAME_RES,this.m_canvas.ResourceName()); // Graphical resource name
      this.SetProperty(CANV_ELEMENT_PROP_CHART_ID,CGBaseObj::ChartID());         // Chart ID
      this.SetProperty(CANV_ELEMENT_PROP_WND_NUM,CGBaseObj::SubWindow());        // Chart subwindow index
      this.SetProperty(CANV_ELEMENT_PROP_NAME_OBJ,CGBaseObj::Name());            // Element object name
      this.SetProperty(CANV_ELEMENT_PROP_TYPE,element_type);                     // Graphical element type
      this.SetProperty(CANV_ELEMENT_PROP_ID,0);                                  // Element ID
      this.SetProperty(CANV_ELEMENT_PROP_NUM,0);                                 // Element index in the list
      this.SetProperty(CANV_ELEMENT_PROP_COORD_X,x);                             // Element's X coordinate on the chart
      this.SetProperty(CANV_ELEMENT_PROP_COORD_Y,y);                             // Element's Y coordinate on the chart
      this.SetProperty(CANV_ELEMENT_PROP_WIDTH,w);                               // Element width
      this.SetProperty(CANV_ELEMENT_PROP_HEIGHT,h);                              // Element height
      this.SetProperty(CANV_ELEMENT_PROP_ACT_SHIFT_LEFT,0);                      // Active area offset from the left edge of the element
      this.SetProperty(CANV_ELEMENT_PROP_ACT_SHIFT_TOP,0);                       // Active area offset from the upper edge of the element
      this.SetProperty(CANV_ELEMENT_PROP_ACT_SHIFT_RIGHT,0);                     // Active area offset from the right edge of the element
      this.SetProperty(CANV_ELEMENT_PROP_ACT_SHIFT_BOTTOM,0);                    // Active area offset from the bottom edge of the element
      this.SetProperty(CANV_ELEMENT_PROP_MOVABLE,false);                         // Element moveability flag
      this.SetProperty(CANV_ELEMENT_PROP_ACTIVE,false);                          // Element activity flag
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

It receives less values, the element background color is set to transparent white and the full element transparency is set.

Remove the now unnecessary strings from the object structure creation method:

```
   this.m_struct_obj.act_shift_bottom=(int)this.GetProperty(CANV_ELEMENT_PROP_ACT_SHIFT_BOTTOM);// Active area offset from the bottom edge of the element
   this.m_struct_obj.opacity=(uchar)this.GetProperty(CANV_ELEMENT_PROP_OPACITY);                // Element opacity
   this.m_struct_obj.color_bg=(color)this.GetProperty(CANV_ELEMENT_PROP_COLOR_BG);              // Element background color
   this.m_struct_obj.movable=(bool)this.GetProperty(CANV_ELEMENT_PROP_MOVABLE);                 // Element moveability flag
```

and add saving the parameters from the new variables below:

```
//+------------------------------------------------------------------+
//| Create the object structure                                      |
//+------------------------------------------------------------------+
bool CGCnvElement::ObjectToStruct(void)
  {
//--- Save integer properties
   this.m_struct_obj.id=(int)this.GetProperty(CANV_ELEMENT_PROP_ID);                            // Element ID
   this.m_struct_obj.type=(int)this.GetProperty(CANV_ELEMENT_PROP_TYPE);                        // Graphical element type
   this.m_struct_obj.number=(int)this.GetProperty(CANV_ELEMENT_PROP_NUM);                       // Eleemnt ID in the list
   this.m_struct_obj.chart_id=this.GetProperty(CANV_ELEMENT_PROP_CHART_ID);                     // Chart ID
   this.m_struct_obj.subwindow=(int)this.GetProperty(CANV_ELEMENT_PROP_WND_NUM);                // Chart subwindow index
   this.m_struct_obj.coord_x=(int)this.GetProperty(CANV_ELEMENT_PROP_COORD_X);                  // Form's X coordinate on the chart
   this.m_struct_obj.coord_y=(int)this.GetProperty(CANV_ELEMENT_PROP_COORD_Y);                  // Form's Y coordinate on the chart
   this.m_struct_obj.width=(int)this.GetProperty(CANV_ELEMENT_PROP_WIDTH);                      // Element width
   this.m_struct_obj.height=(int)this.GetProperty(CANV_ELEMENT_PROP_HEIGHT);                    // Element height
   this.m_struct_obj.edge_right=(int)this.GetProperty(CANV_ELEMENT_PROP_RIGHT);                 // Element right edge
   this.m_struct_obj.edge_bottom=(int)this.GetProperty(CANV_ELEMENT_PROP_BOTTOM);               // Element bottom edge
   this.m_struct_obj.act_shift_left=(int)this.GetProperty(CANV_ELEMENT_PROP_ACT_SHIFT_LEFT);    // Active area offset from the left edge of the element
   this.m_struct_obj.act_shift_top=(int)this.GetProperty(CANV_ELEMENT_PROP_ACT_SHIFT_TOP);      // Active area offset from the top edge of the element
   this.m_struct_obj.act_shift_right=(int)this.GetProperty(CANV_ELEMENT_PROP_ACT_SHIFT_RIGHT);  // Active area offset from the right edge of the element
   this.m_struct_obj.act_shift_bottom=(int)this.GetProperty(CANV_ELEMENT_PROP_ACT_SHIFT_BOTTOM);// Active area offset from the bottom edge of the element

   this.m_struct_obj.movable=(bool)this.GetProperty(CANV_ELEMENT_PROP_MOVABLE);                 // Element moveability flag
   this.m_struct_obj.active=(bool)this.GetProperty(CANV_ELEMENT_PROP_ACTIVE);                   // Element activity flag
   this.m_struct_obj.coord_act_x=(int)this.GetProperty(CANV_ELEMENT_PROP_COORD_ACT_X);          // X coordinate of the element active area
   this.m_struct_obj.coord_act_y=(int)this.GetProperty(CANV_ELEMENT_PROP_COORD_ACT_Y);          // Y coordinate of the element active area
   this.m_struct_obj.coord_act_right=(int)this.GetProperty(CANV_ELEMENT_PROP_ACT_RIGHT);        // Right border of the element active area
   this.m_struct_obj.coord_act_bottom=(int)this.GetProperty(CANV_ELEMENT_PROP_ACT_BOTTOM);      // Bottom border of the element active area
   this.m_struct_obj.color_bg=this.m_color_bg;                                                  // Element background color
   this.m_struct_obj.opacity=this.m_opacity;                                                    // Element opacity
//--- Save real properties

//--- Save string properties
   ::StringToCharArray(this.GetProperty(CANV_ELEMENT_PROP_NAME_OBJ),this.m_struct_obj.name_obj);// Graphical element object name
   ::StringToCharArray(this.GetProperty(CANV_ELEMENT_PROP_NAME_RES),this.m_struct_obj.name_res);// Graphical resource name
   //--- Save the structure to the uchar array
   ::ResetLastError();
   if(!::StructToCharArray(this.m_struct_obj,this.m_uchar_array))
     {
      ::Print(DFUN,CMessage::Text(MSG_LIB_SYS_FAILED_SAVE_OBJ_STRUCT_TO_UARRAY),(string)::GetLastError());
      return false;
     }
   return true;
  }
//+------------------------------------------------------------------+
```

The same isdone in the method of creating the object from the structure:

```
//+------------------------------------------------------------------+
//| Create the object from the structure                             |
//+------------------------------------------------------------------+
void CGCnvElement::StructToObject(void)
  {
//--- Save integer properties
   this.SetProperty(CANV_ELEMENT_PROP_ID,this.m_struct_obj.id);                                 // Element ID
   this.SetProperty(CANV_ELEMENT_PROP_TYPE,this.m_struct_obj.type);                             // Graphical element type
   this.SetProperty(CANV_ELEMENT_PROP_NUM,this.m_struct_obj.number);                            // Element index in the list
   this.SetProperty(CANV_ELEMENT_PROP_CHART_ID,this.m_struct_obj.chart_id);                     // Chart ID
   this.SetProperty(CANV_ELEMENT_PROP_WND_NUM,this.m_struct_obj.subwindow);                     // Chart subwindow index
   this.SetProperty(CANV_ELEMENT_PROP_COORD_X,this.m_struct_obj.coord_x);                       // Form's X coordinate on the chart
   this.SetProperty(CANV_ELEMENT_PROP_COORD_Y,this.m_struct_obj.coord_y);                       // Form's Y coordinate on the chart
   this.SetProperty(CANV_ELEMENT_PROP_WIDTH,this.m_struct_obj.width);                           // Element width
   this.SetProperty(CANV_ELEMENT_PROP_HEIGHT,this.m_struct_obj.height);                         // Element height
   this.SetProperty(CANV_ELEMENT_PROP_RIGHT,this.m_struct_obj.edge_right);                      // Element right edge
   this.SetProperty(CANV_ELEMENT_PROP_BOTTOM,this.m_struct_obj.edge_bottom);                    // Element bottom edge
   this.SetProperty(CANV_ELEMENT_PROP_ACT_SHIFT_LEFT,this.m_struct_obj.act_shift_left);         // Active area offset from the left edge of the element
   this.SetProperty(CANV_ELEMENT_PROP_ACT_SHIFT_TOP,this.m_struct_obj.act_shift_top);           // Active area offset from the upper edge of the element
   this.SetProperty(CANV_ELEMENT_PROP_ACT_SHIFT_RIGHT,this.m_struct_obj.act_shift_right);       // Active area offset from the right edge of the element
   this.SetProperty(CANV_ELEMENT_PROP_ACT_SHIFT_BOTTOM,this.m_struct_obj.act_shift_bottom);     // Active area offset from the bottom edge of the element


   this.SetProperty(CANV_ELEMENT_PROP_MOVABLE,this.m_struct_obj.movable);                       // Element moveability flag
   this.SetProperty(CANV_ELEMENT_PROP_ACTIVE,this.m_struct_obj.active);                         // Element activity flag
   this.SetProperty(CANV_ELEMENT_PROP_COORD_ACT_X,this.m_struct_obj.coord_act_x);               // X coordinate of the element active area
   this.SetProperty(CANV_ELEMENT_PROP_COORD_ACT_Y,this.m_struct_obj.coord_act_y);               // Y coordinate of the element active area
   this.SetProperty(CANV_ELEMENT_PROP_ACT_RIGHT,this.m_struct_obj.coord_act_right);             // Right border of the element active area
   this.SetProperty(CANV_ELEMENT_PROP_ACT_BOTTOM,this.m_struct_obj.coord_act_bottom);           // Bottom border of the element active area
   this.m_color_bg=this.m_struct_obj.color_bg;                                                  // Element background color
   this.m_opacity=this.m_struct_obj.opacity;                                                    // Element opacity
//--- Save real properties

//--- Save string properties
   this.SetProperty(CANV_ELEMENT_PROP_NAME_OBJ,::CharArrayToString(this.m_struct_obj.name_obj));// Graphical element object name
   this.SetProperty(CANV_ELEMENT_PROP_NAME_RES,::CharArrayToString(this.m_struct_obj.name_res));// Graphical resource name
  }
//+------------------------------------------------------------------+
```

In the method creating the graphical object element, the object background is now completely erased and filled with white transparent color:

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
      this.Erase(NULL_COLOR);
      this.m_canvas.Update(redraw);
      this.m_shift_y=(int)::ChartGetInteger(chart_id,CHART_WINDOW_YDISTANCE,wnd_num);
      return true;
     }
   return false;
  }
//+------------------------------------------------------------------+
```

In the method setting the element opacity, the opacity will be now set in the variable instead of an object removed property:

```
//+------------------------------------------------------------------+
//| Set the element opacity                                          |
//+------------------------------------------------------------------+
void CGCnvElement::SetOpacity(const uchar value,const bool redraw=false)
  {
   this.m_canvas.TransparentLevelSet(value);
   this.m_opacity=value;
   this.m_canvas.Update(redraw);
  }
//+------------------------------------------------------------------+
```

**The new method changing the color lightness by the specified amount:**

```
//+------------------------------------------------------------------+
//| Change the color lightness by the specified amount               |
//+------------------------------------------------------------------+
uint CGCnvElement::ChangeColorLightness(const uint clr,const double change_value)
  {
   if(change_value==0.0)
      return clr;
   double a=GETRGBA(clr);
   double r=GETRGBR(clr);
   double g=GETRGBG(clr);
   double b=GETRGBB(clr);
   double h=0,s=0,l=0;
   CColors::RGBtoHSL(r,g,b,h,s,l);
   double nl=l+change_value;
   if(nl>1.0) nl=1.0;
   if(nl<0.0) nl=0.0;
   CColors::HSLtoRGB(h,s,nl,r,g,b);
   return ARGB(a,r,g,b);
  }
//+------------------------------------------------------------------+
```

Here:

Check the passed value, by which the lightness should be changed. If zero is passed, nothing should be changed — return the color unchanged.

Next, receive each of the ARGB color components of the color passed to the method and convert RGB components into the HSL color model.

After converting the value of each component, HSL models are set into the corresponding variables (we need the l component).

Add the value passed to the method ( **change\_value** values may vary from -1.0 to 1.0) and adjust it when going beyond acceptable value ranges.

Next, convert the HSL model back into RGB and return the ARGB model obtained from the new color components generated by converting from the HSL model to RGB.

### Color themes and form types

The library is to support creation of various objects — graphical elements, relevant forms, windows, etc. Each form, window or image (frames, separators, drop-down lists, etc.) may have different display styles. However, it would be strange to have various objects with different drawing styles, colors and decorations within a single program.

To simplify the development of identically looking objects belonging to a single program, I will introduce drawing styles, object types and color schemes. This will allow end users to select the desired style and color theme in the program settings, while the programmer will not have to care about all that too much — selected themes and styles will immediately rebuild all objects according to a single criterion. I only need to make the necessary changes and additions to the file of graphical settings featuring all the necessary colors, as well as object and primitive parameters.

I have already used such a tactic when developing the [library message class](https://www.mql5.com/en/articles/7176/). It features the list of message indices and the array of texts corresponding to message indices. The first thing I usually do at the beginning of each article is setting new data.

The file of graphical settings will be arranged in the same way: it will feature the enumeration of color themes and object styles, as well as the appropriate arrays, which will gradually receive new parameters and their values for each newly added property, theme or color.

In \\MQL5\\Include\ **DoEasy\** root folder, create the new **GraphINI.mqh** include file and add the number of color themes to it:

```
//+------------------------------------------------------------------+
//|                                                     GraphINI.mqh |
//|                                  Copyright 2021, MetaQuotes Ltd. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2021, MetaQuotes Ltd."
#property link      "https://mql5.com/en/users/artmedia70"
//+------------------------------------------------------------------+
//| Macro substitutions                                              |
//+------------------------------------------------------------------+
#define TOTAL_COLOR_THEMES             (2)      // Number of color schemes
//+------------------------------------------------------------------+
```

Two color themes will be sufficient to demonstrate the usage of the settings file. I will increase their number subsequently.

Below I set the indices of color schemes and the indices of one theme parameters. Each theme will have the same number of parameters:

```
//+------------------------------------------------------------------+
//|                                                     GraphINI.mqh |
//|                                  Copyright 2021, MetaQuotes Ltd. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2021, MetaQuotes Ltd."
#property link      "https://mql5.com/en/users/artmedia70"
//+------------------------------------------------------------------+
//| Macro substitutions                                              |
//+------------------------------------------------------------------+
#define TOTAL_COLOR_THEMES             (2)      // Number of color schemes
//+------------------------------------------------------------------+
//| Enumerations                                                     |
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| List of color scheme indices                                     |
//+------------------------------------------------------------------+
enum ENUM_COLOR_THEMES
  {
   COLOR_THEME_BLUE_STEEL,                      // Blue steel
   COLOR_THEME_LIGHT_CYAN_GRAY,                 // Light cyan gray
  };
//+------------------------------------------------------------------+
//| List of indices of color scheme parameters                       |
//+------------------------------------------------------------------+
enum ENUM_COLOR_THEME_COLORS
  {
   COLOR_THEME_COLOR_FORM_BG,                   // Form background color
   COLOR_THEME_COLOR_FORM_FRAME,                // Form frame color
   COLOR_THEME_COLOR_FORM_FRAME_OUTER,          // Form outer frame color
   COLOR_THEME_COLOR_FORM_SHADOW,               // Form shadow color
  };
#define TOTAL_COLOR_THEME_COLORS       (4)      // Number of parameters in the color theme
//+------------------------------------------------------------------+
```

It will be convenient to access each color theme and the required parameter by the enumeration constant names.

Below, write a two-dimensional array containing color themes in the first dimension and color parameter indices for drawing various object properties in the second one:

```
//+------------------------------------------------------------------+
//|                                                     GraphINI.mqh |
//|                                  Copyright 2021, MetaQuotes Ltd. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2021, MetaQuotes Ltd."
#property link      "https://mql5.com/en/users/artmedia70"
//+------------------------------------------------------------------+
//| Macro substitutions                                              |
//+------------------------------------------------------------------+
#define TOTAL_COLOR_THEMES             (2)      // Number of color schemes
//+------------------------------------------------------------------+
//| Enumerations                                                     |
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| List of color scheme indices                                     |
//+------------------------------------------------------------------+
enum ENUM_COLOR_THEMES
  {
   COLOR_THEME_BLUE_STEEL,                      // Blue steel
   COLOR_THEME_LIGHT_CYAN_GRAY,                 // Light cyan gray
  };
//+------------------------------------------------------------------+
//| List of indices of color scheme parameters                       |
//+------------------------------------------------------------------+
enum ENUM_COLOR_THEME_COLORS
  {
   COLOR_THEME_COLOR_FORM_BG,                   // Form background color
   COLOR_THEME_COLOR_FORM_FRAME,                // Form frame color
   COLOR_THEME_COLOR_FORM_FRAME_OUTER,          // Form outer frame color
   COLOR_THEME_COLOR_FORM_SHADOW,               // Form shadow color
  };
#define TOTAL_COLOR_THEME_COLORS       (4)      // Number of parameters in the color theme
//+------------------------------------------------------------------+
//| The array containing color schemes                               |
//+------------------------------------------------------------------+
color array_color_themes[TOTAL_COLOR_THEMES][TOTAL_COLOR_THEME_COLORS]=
  {
//--- Parameters of the "Blue steel" color scheme
   {
      C'134,160,181',                           // Form background color
      C'134,160,181',                           // Form frame color
      clrDimGray,                               // Form outer frame color
      C'46,85,117',                             // Form shadow color
   },
//--- Parameters of the "Light cyan gray" color scheme
   {
      C'181,196,196',                           // Form background color
      C'181,196,196',                           // Form frame color
      clrGray,                                  // Form outer frame color
      C'130,147,153',                           // Form shadow color
   },
  };
//+------------------------------------------------------------------+
```

This is the array which is to gradually receive new colors for each newly added graphical object parameter whose color should depend on the selected color scheme.

Next, add enumerations of smoothing types when drawing primitives, frame styles, types and form styles followed by the enumerations of form style property indices and their parameters similar to the ones designed for color themes:

```
//+------------------------------------------------------------------+
//| Smoothing types                                                  |
//+------------------------------------------------------------------+
enum ENUM_SMOOTHING_TYPE
  {
   SMOOTHING_TYPE_NONE,                         // No smoothing
   SMOOTHING_TYPE_AA,                           // Anti-aliasing
   SMOOTHING_TYPE_WU,                           // Wu
   SMOOTHING_TYPE_THICK,                        // Thick
   SMOOTHING_TYPE_DUAL,                         // Dual
  };
//+------------------------------------------------------------------+
//| Frame styles                                                     |
//+------------------------------------------------------------------+
enum ENUM_FRAME_STYLE
  {
   FRAME_STYLE_SIMPLE,                          // Simple frame
   FRAME_STYLE_FLAT,                            // Flat frame
   FRAME_STYLE_BEVEL,                           // Embossed (convex)
   FRAME_STYLE_STAMP,                           // Embossed (concave)
  };
//+------------------------------------------------------------------+
//| Form types                                                       |
//+------------------------------------------------------------------+
enum ENUM_FORM_TYPE
  {
   FORM_TYPE_SQUARE,                            // Rectangular
  };
//+------------------------------------------------------------------+
//| Form styles                                                      |
//+------------------------------------------------------------------+
enum ENUM_FORM_STYLE
  {
   FORM_STYLE_FLAT,                             // Flat form
   FORM_STYLE_BEVEL,                            // Embossed form
  };
#define TOTAL_FORM_STYLES
//+------------------------------------------------------------------+
//| List of form style parameter indices                             |
//+------------------------------------------------------------------+
enum ENUM_FORM_STYLE_PARAMS
  {
   FORM_STYLE_FRAME_WIDTH_LEFT,                 // Form frame width to the left
   FORM_STYLE_FRAME_WIDTH_RIGHT,                // Form frame width to the right
   FORM_STYLE_FRAME_WIDTH_TOP,                  // Form frame width on top
   FORM_STYLE_FRAME_WIDTH_BOTTOM,               // Form frame width below
   FORM_STYLE_FRAME_SHADOW_OPACITY,             // Shadow opacity
  };
#define TOTAL_FORM_STYLE_PARAMS        (5)      // Number of form style parameters
//+------------------------------------------------------------------+
//| Array containing form style parameters                           |
//+------------------------------------------------------------------+
int array_form_style[TOTAL_FORM_STYLES][TOTAL_FORM_STYLE_PARAMS]=
  {
//--- "Flat form" style parameters
   {
      3,                                        // Form frame width to the left
      3,                                        // Form frame width to the right
      3,                                        // Form frame width on top
      3,                                        // Form frame width below
      80,                                       // Shadow opacity
   },
//--- "Embossed form" style parameters
   {
      4,                                        // Form frame width to the left
      4,                                        // Form frame width to the right
      4,                                        // Form frame width on top
      4,                                        // Form frame width below
      100,                                      // Shadow opacity
   },
  };
//+------------------------------------------------------------------+
```

The second array, whose construction logic is identical to the color theme array, will gradually receive new parameters of constructing elements, forms, windows and other objects whose parameters should depend on a selected style of designing object appearances.

Add new enumerations for program inputs in \\MQL5\\Include\\DoEasy\ **InpData.mqh** to be able to select the necessary style of constructing objects and a color theme. At its very beginning, include the newly created GraphINI.mqh file:

```
//+------------------------------------------------------------------+
//|                                                      InpData.mqh |
//|                        Copyright 2020, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2020, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include "GraphINI.mqh"
//+------------------------------------------------------------------+
```

In the code block for compiling in Englishand Russian, add new input enumerations for selecting a color theme:

```
//+------------------------------------------------------------------+
//|                                                      InpData.mqh |
//|                        Copyright 2020, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2020, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include "GraphINI.mqh"
//+------------------------------------------------------------------+
//| Macro substitutions                                              |
//+------------------------------------------------------------------+
#define COMPILE_EN // Comment out the string for compilation in Russian
//+------------------------------------------------------------------+
//| Input enumerations                                               |
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| English language inputs                                          |
//+------------------------------------------------------------------+
#ifdef COMPILE_EN
//+------------------------------------------------------------------+
//| Modes of working with symbols                                    |
//+------------------------------------------------------------------+
enum ENUM_SYMBOLS_MODE
  {
   SYMBOLS_MODE_CURRENT,                              // Work only with the current Symbol
   SYMBOLS_MODE_DEFINES,                              // Work with a given list of Symbols
   SYMBOLS_MODE_MARKET_WATCH,                         // Working with Symbols from the "Market Watch" window
   SYMBOLS_MODE_ALL                                   // Work with a complete list of Symbols
  };
//+------------------------------------------------------------------+
//| Mode of working with timeframes                                  |
//+------------------------------------------------------------------+
enum ENUM_TIMEFRAMES_MODE
  {
   TIMEFRAMES_MODE_CURRENT,                           // Work only with the current timeframe
   TIMEFRAMES_MODE_LIST,                              // Work with a given list of timeframes
   TIMEFRAMES_MODE_ALL                                // Work with a complete list of timeframes
  };
//+------------------------------------------------------------------+
//| "Yes"/"No"                                                       |
//+------------------------------------------------------------------+
enum ENUM_INPUT_YES_NO
  {
   INPUT_NO  = 0,                                     // No
   INPUT_YES = 1                                      // Yes
  };
//+------------------------------------------------------------------+
//| Select color themes                                              |
//+------------------------------------------------------------------+
enum ENUM_INPUT_COLOR_THEME
  {
   INPUT_COLOR_THEME_BLUE_STEEL,                      // Blue steel
   INPUT_COLOR_THEME_LIGHT_CYAN_GRAY,                 // Light cyan gray
  };
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| Russian language inputs                                          |
//+------------------------------------------------------------------+
#else
//+------------------------------------------------------------------+
//| Modes of working with symbols                                    |
//+------------------------------------------------------------------+
enum ENUM_SYMBOLS_MODE
  {
   SYMBOLS_MODE_CURRENT,                              // Работа только с текущим символом
   SYMBOLS_MODE_DEFINES,                              // Работа с заданным списком символов
   SYMBOLS_MODE_MARKET_WATCH,                         // Работа с символами из окна "Обзор рынка"
   SYMBOLS_MODE_ALL                                   // Работа с полным списком символов
  };
//+------------------------------------------------------------------+
//| Mode of working with timeframes                                  |
//+------------------------------------------------------------------+
enum ENUM_TIMEFRAMES_MODE
  {
   TIMEFRAMES_MODE_CURRENT,                           // Работа только с текущим таймфреймом
   TIMEFRAMES_MODE_LIST,                              // Работа с заданным списком таймфреймов
   TIMEFRAMES_MODE_ALL                                // Работа с полным списком таймфреймов
  };
//+------------------------------------------------------------------+
//| "Да"/"Нет"                                                       |
//+------------------------------------------------------------------+
enum ENUM_INPUT_YES_NO
  {
   INPUT_NO  = 0,                                     // Нет
   INPUT_YES = 1                                      // Да
  };
//+------------------------------------------------------------------+
//| Select color themes                                              |
//+------------------------------------------------------------------+
enum ENUM_COLOR_THEME
  {
   COLOR_THEME_BLUE_STEEL,                            // Голубая сталь
   COLOR_THEME_LIGHT_CYAN_GRAY,                       // Светлый серо-циановый
  };
//+------------------------------------------------------------------+
#endif
//+------------------------------------------------------------------+
```

This will allow us to select the desired color scheme when starting the program. Later, I will add the selection of object drawing styles and construction types.

### Form object class

The form object is a more advanced version of a graphical object. It allows drawing "dimensional" frames and other primitives, as well as attach other elements to it. Of course, you can draw everything you need "manually" but the form allows you to automate your work.

In E:\\MetaQuotes\\MetaTrader 5\\MQL5\\Include\\DoEasy\\Objects\ **Graph\**, create the new **Form.mqh** file of the **CForm** class. The class should be inherited from the graphical element object, which means the element object file should be included as well:

```
//+------------------------------------------------------------------+
//|                                                         Form.mqh |
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
#include "GCnvElement.mqh"
//+------------------------------------------------------------------+
//| Form object class                                                |
//+------------------------------------------------------------------+
class CForm : public CGCnvElement
  {
  }
```

In the private section of the class, declare the necessary objects, variables and class auxiliary methods:

```
//+------------------------------------------------------------------+
//| Form object class                                                |
//+------------------------------------------------------------------+
class CForm : public CGCnvElement
  {
private:
   CArrayObj         m_list_elements;                          // List of attached elements
   CGCnvElement     *m_shadow_obj;                             // Pointer to the shadow object
   color             m_color_frame;                            // Form frame color
   color             m_color_shadow;                           // Form shadow color
   int               m_frame_width_left;                       // Form frame width to the left
   int               m_frame_width_right;                      // Form frame width to the right
   int               m_frame_width_top;                        // Form frame width at the top
   int               m_frame_width_bottom;                     // Form frame width at the bottom

//--- Initialize the variables
   void              Initialize(void);

//--- Create a new graphical object
   CGCnvElement     *CreateNewGObject(const ENUM_GRAPH_ELEMENT_TYPE type,
                                      const int element_num,
                                      const string name,
                                      const int x,
                                      const int y,
                                      const int w,
                                      const int h,
                                      const color colour,
                                      const uchar opacity,
                                      const bool movable,
                                      const bool activity);

public:
```

The public section of the class, features the methods that are standard for the library objects and several constructors — default ones as well as the ones allowing users to create an object form on a specified chart and subwindow, on a specified subwindow of the current chart and on the current chart in the main window:

```
public:
   //--- Constructors
                     CForm(const long chart_id,
                           const int subwindow,
                           const string name,
                           const int x,
                           const int y,
                           const int w,
                           const int h);
                     CForm(const int subwindow,
                           const string name,
                           const int x,
                           const int y,
                           const int w,
                           const int h);
                     CForm(const string name,
                           const int x,
                           const int y,
                           const int w,
                           const int h);
                     CForm() { this.Initialize(); }
//--- Destructor
                    ~CForm();

//--- Supported form properties (1) integer and (2) string ones
   virtual bool      SupportProperty(ENUM_CANV_ELEMENT_PROP_INTEGER property) { return true; }
   virtual bool      SupportProperty(ENUM_CANV_ELEMENT_PROP_STRING property)  { return true; }

//--- Return (1) the list of attached objects and (2) the shadow object
   CArrayObj        *GetList(void)                                            { return &this.m_list_elements;  }
   CGCnvElement     *GetShadowObj(void)                                       { return this.m_shadow_obj;      }
```

Below are the methods of working with the form object:

```
//--- Set the form (1) color scheme and (2) style
   virtual void      SetColorTheme(const ENUM_COLOR_THEMES theme,const uchar opacity);
   virtual void      SetFormStyle(const ENUM_FORM_STYLE style,
                                  const ENUM_COLOR_THEMES theme,
                                  const uchar opacity,
                                  const bool shadow=false,
                                  const bool redraw=false);

//--- Create a new attached element
   bool              CreateNewElement(const int element_num,
                                      const string name,
                                      const int x,
                                      const int y,
                                      const int w,
                                      const int h,
                                      const color colour,
                                      const uchar opacity,
                                      const bool movable,
                                      const bool activity);

//--- Create a shadow object
   void              CreateShadow(const uchar opacity);
//--- Draw an object shadow
   void              DrawShadow(const uchar opacity);

//--- Draw the form frame
   void              DrawFormFrame(const int wd_top,                          // Frame upper segment width
                                   const int wd_bottom,                       // Frame lower segment width
                                   const int wd_left,                         // Frame left segment width
                                   const int wd_right,                        // Frame right segment width
                                   const color colour,                        // Frame color
                                   const uchar opacity,                       // Frame opacity
                                   const ENUM_FRAME_STYLE style);             // Frame style
//--- Draw a simple frame
   void              DrawFrameSimple(const int x,                             // X coordinate relative to the form
                                     const int y,                             // Y coordinate relative to the form
                                     const int width,                         // Frame width
                                     const int height,                        // Frame height
                                     const int wd_top,                        // Frame upper segment width
                                     const int wd_bottom,                     // Frame lower segment width
                                     const int wd_left,                       // Frame left segment width
                                     const int wd_right,                      // Frame right segment width
                                     const color colour,                      // Frame color
                                     const uchar opacity);                    // Frame opacity
//--- Draw a flat frame
   void              DrawFrameFlat(const int x,                               // X coordinate relative to the form
                                   const int y,                               // Y coordinate relative to the form
                                   const int width,                           // Frame width
                                   const int height,                          // Frame height
                                   const int wd_top,                          // Frame upper segment width
                                   const int wd_bottom,                       // Frame lower segment width
                                   const int wd_left,                         // Frame left segment width
                                   const int wd_right,                        // Frame right segment width
                                   const color colour,                        // Frame color
                                   const uchar opacity);                      // Frame opacity

//--- Draw an embossed (convex) frame
   void              DrawFrameBevel(const int x,                              // X coordinate relative to the form
                                    const int y,                              // Y coordinate relative to the form
                                    const int width,                          // Frame width
                                    const int height,                         // Frame height
                                    const int wd_top,                         // Frame upper segment width
                                    const int wd_bottom,                      // Frame lower segment width
                                    const int wd_left,                        // Frame left segment width
                                    const int wd_right,                       // Frame right segment width
                                    const color colour,                       // Frame color
                                    const uchar opacity);                     // Frame opacity

//--- Draw an embossed (concave) frame
   void              DrawFrameStamp(const int x,                              // X coordinate relative to the form
                                    const int y,                              // Y coordinate relative to the form
                                    const int width,                          // Frame width
                                    const int height,                         // Frame height
                                    const int wd_top,                         // Frame upper segment width
                                    const int wd_bottom,                      // Frame lower segment width
                                    const int wd_left,                        // Frame left segment width
                                    const int wd_right,                       // Frame right segment width
                                    const color colour,                       // Frame color
                                    const uchar opacity);                     // Frame opacity

//--- Draw a simple field
   void              DrawFieldFlat(const int x,                               // X coordinate relative to the form
                                   const int y,                               // Y coordinate relative to the form
                                   const int width,                           // Field width
                                   const int height,                          // Field height
                                   const color colour,                        // Field color
                                   const uchar opacity);                      // Field opacity

//--- Draw an embossed (convex) field
   void              DrawFieldBevel(const int x,                              // X coordinate relative to the form
                                    const int y,                              // Y coordinate relative to the form
                                    const int width,                          // Field width
                                    const int height,                         // Field height
                                    const color colour,                       // Field color
                                    const uchar opacity);                     // Field opacity

//--- Draw an embossed (concave) field
   void              DrawFieldStamp(const int x,                              // X coordinate relative to the form
                                    const int y,                              // Y coordinate relative to the form
                                    const int width,                          // Field width
                                    const int height,                         // Field height
                                    const color colour,                       // Field color
                                    const uchar opacity);                     // Field opacity

//+------------------------------------------------------------------+
//| Methods of simplified access to object properties                |
//+------------------------------------------------------------------+
//--- (1) Set and (2) get the form frame color
   void              SetColorFrame(const color colour)                        { this.m_color_frame=colour;  }
   color             ColorFrame(void)                                   const { return this.m_color_frame;  }
//--- (1) Set and (2) return the form shadow color
   void              SetColorShadow(const color colour)                       { this.m_color_shadow=colour; }
   color             ColorShadow(void)                                  const { return this.m_color_shadow; }

  };
//+------------------------------------------------------------------+
```

Let's consider the declared methods in details.

**Constructor indicating the chart and subwindow ID:**

```
//+------------------------------------------------------------------+
//| Constructor indicating the chart and subwindow ID                |
//+------------------------------------------------------------------+
CForm::CForm(const long chart_id,
             const int subwindow,
             const string name,
             const int x,
             const int y,
             const int w,
             const int h) : CGCnvElement(GRAPH_ELEMENT_TYPE_FORM,chart_id,subwindow,name,x,y,w,h)
  {
   this.Initialize();
  }
//+------------------------------------------------------------------+
```

The constructor receives the ID of the chart and the index of the subwindow used to create a form object, its name, coordinates of the upper left form angle and its size. In the initialization list, call the element object class constructor specifying the Form object type. In the class body, call the initialization method.

**Current chart constructor specifying the subwindow:**

```
//+------------------------------------------------------------------+
//| Current chart constructor specifying the subwindow               |
//+------------------------------------------------------------------+
CForm::CForm(const int subwindow,
             const string name,
             const int x,
             const int y,
             const int w,
             const int h) : CGCnvElement(GRAPH_ELEMENT_TYPE_FORM,::ChartID(),subwindow,name,x,y,w,h)
  {
   this.Initialize();
  }
//+------------------------------------------------------------------+
```

The constructor receives the number of the subwindow the form object should be created on (the current chart), form object name, coordinates of the upper left form angle and its size. In the initialization list, call the element object class constructor specifying the Form object type and the current chart ID. In the class body, call the initialization method.

**Constructor on the current chart in the main chart window:**

```
//+------------------------------------------------------------------+
//| Constructor on the current chart in the main chart window        |
//+------------------------------------------------------------------+
CForm::CForm(const string name,
             const int x,
             const int y,
             const int w,
             const int h) : CGCnvElement(GRAPH_ELEMENT_TYPE_FORM,::ChartID(),0,name,x,y,w,h)
  {
   this.Initialize();
  }
//+------------------------------------------------------------------+
```

The constructor receives the name of the form object, coordinates of the upper left form angle and its size. In the initialization list, call the element object class constructor, specifying the Form object type and the current chart ID, and the main window index (0). In the class body, call the initialization method.

**In the class destructor**, check the validity of the pointer to the shadow object and remove the object if it exists:

```
//+------------------------------------------------------------------+
//| Destructor                                                       |
//+------------------------------------------------------------------+
CForm::~CForm()
  {
   if(m_shadow_obj!=NULL)
      delete m_shadow_obj;
  }
//+------------------------------------------------------------------+
```

**The variable initialization method:**

```
//+------------------------------------------------------------------+
//| Initialize the variables                                         |
//+------------------------------------------------------------------+
void CForm::Initialize(void)
  {
   this.m_list_elements.Clear();
   this.m_list_elements.Sort();
   this.m_shadow_obj=NULL;
   this.m_shadow=false;
   this.m_frame_width_right=2;
   this.m_frame_width_left=2;
   this.m_frame_width_top=2;
   this.m_frame_width_bottom=2;
  }
//+------------------------------------------------------------------+
```

Here I clear the list of elements attached to the form, set the sorted list flag to it and specify the default values for the shadow object pointer (NULL), the shadow drawing flag (false) and the form frame size (two pixels on each side).

**The private method creating a new graphical object:**

```
//+------------------------------------------------------------------+
//| Create a new graphical object                                    |
//+------------------------------------------------------------------+
CGCnvElement *CForm::CreateNewGObject(const ENUM_GRAPH_ELEMENT_TYPE type,
                                      const int obj_num,
                                      const string obj_name,
                                      const int x,
                                      const int y,
                                      const int w,
                                      const int h,
                                      const color colour,
                                      const uchar opacity,
                                      const bool movable,
                                      const bool activity)
  {
   int pos=::StringLen(::MQLInfoString(MQL_PROGRAM_NAME));
   string pref=::StringSubstr(NameObj(),pos+1);
   string name=pref+"_"+obj_name;
   CGCnvElement *element=new CGCnvElement(type,this.ID(),obj_num,this.ChartID(),this.SubWindow(),name,x,y,w,h,colour,opacity,movable,activity);
   if(element==NULL)
      ::Print(DFUN,CMessage::Text(MSG_LIB_SYS_FAILED_CREATE_ELM_OBJ),": ",name);
   return element;
  }
//+------------------------------------------------------------------+
```

The method receives all the parameters that are necessary for creating a new object — its type, its index in the list of attached objects, name, coordinates, size, color, opacity and the flags of the object moveability and activity.

In the class body, retrieve the ending from the name object (the name consists of the program name and object name assigned during its creation). We need to retrieve the object name during its creation and add the name passed to the method.

For example, in case of the name _"Program\_name\_Form01"_, we retrieve the _"Form01"_ substring and add the name passed to the method. If we create a shadow object and pass the name _"Shadow"_, the object name is _"Form01\_ Shadow "_, while the final name of the created object is as follows: _"Program\_name\_Form01\_Shadow"_.

Next, we create a new object specifying its type, parameters of the chart (the current form object has been created on), specific name and other parameters passed to the method. Return the method pointer to the created object or NULL in case of a failure.

**The method that creates a new attached element:**

```
//+------------------------------------------------------------------+
//| Create a new attached element                                    |
//+------------------------------------------------------------------+
bool CForm::CreateNewElement(const int element_num,
                             const string element_name,
                             const int x,
                             const int y,
                             const int w,
                             const int h,
                             const color colour,
                             const uchar opacity,
                             const bool movable,
                             const bool activity)
  {
   CGCnvElement *obj=this.CreateNewGObject(GRAPH_ELEMENT_TYPE_ELEMENT,element_num,element_name,x,y,w,h,colour,opacity,movable,activity);
   if(obj==NULL)
      return false;
   this.m_list_elements.Sort(SORT_BY_CANV_ELEMENT_NAME_OBJ);
   int index=this.m_list_elements.Search(obj);
   if(index>WRONG_VALUE)
     {
      ::Print(DFUN,CMessage::Text(MSG_LIB_SYS_OBJ_ALREADY_IN_LIST),": ",obj.NameObj());
      delete obj;
      return false;
     }
   if(!this.m_list_elements.Add(obj))
     {
      ::Print(DFUN,CMessage::Text(MSG_LIB_SYS_FAILED_OBJ_ADD_TO_LIST),": ",obj.NameObj());
      delete obj;
      return false;
     }
   return true;
  }
//+------------------------------------------------------------------+
```

The method creates a new graphical element object using the above method and adds it to the list of attached objects in the form object. If failed to create a new object or to add it to the list of attached objects, display the error message and return false. When a new element is successfully created and added to the list, return true.

**The method creating the shadow object:**

```
//+------------------------------------------------------------------+
//| Create the shadow object                                         |
//+------------------------------------------------------------------+
void CForm::CreateShadow(const uchar opacity)
  {
//--- If the shadow flag is disabled, exit
   if(!this.m_shadow)
      return;
//--- Calculate the shadow object coordinates according to the offset from the top and left
   int x=this.CoordX()-OUTER_AREA_SIZE;
   int y=this.CoordY()-OUTER_AREA_SIZE;
//--- Calculate the width and height in accordance with the top, bottom, left and right offsets
   int w=this.Width()+OUTER_AREA_SIZE*2;
   int h=this.Height()+OUTER_AREA_SIZE*2;
//--- Create a new element object and set the pointer to it in the variable
   this.m_shadow_obj=this.CreateNewGObject(GRAPH_ELEMENT_TYPE_ELEMENT,-1,"Shadow",x,y,w,h,this.m_chart_color_bg,opacity,Movable(),false);
   if(this.m_shadow_obj==NULL)
      return;
//--- Move the form object to the foreground
   this.BringToTop();
  }
//+------------------------------------------------------------------+
```

The method logic features comments in the listing. In short, since the element object the shadow is to be drawn on should be bigger than the form object it is created for (we need free space at the top, bottom, left and right to draw the shadow), the size of the new object is calculated depending on the OUTER\_AREA\_SIZE macro substitution value.

After the object has been successfully created, it is automatically set above the form object it has been created on. Therefore, we need to forcefully move the form object to the foreground. This is done at the very end of the method.

**The shadow drawing method:**

```
//+------------------------------------------------------------------+
//| Draw the shadow                                                  |
//+------------------------------------------------------------------+
void CForm::DrawShadow(const uchar opacity)
  {
//--- If the shadow flag is disabled, exit
   if(!this.m_shadow)
      return;
//--- Calculate rectangle coordinates relative to the shadow object borders
   int x=OUTER_AREA_SIZE+1;
   int y=OUTER_AREA_SIZE+1;
//--- Draw a filled rectangle starting from the calculated coordinates and having the size of the current form object
   m_shadow_obj.DrawRectangleFill(x,y,x+Width(),y+Height(),this.ColorShadow(),opacity);
//--- Update the shadow object for displaying changes
   m_shadow_obj.Update();
   return;
  }
//+------------------------------------------------------------------+
```

The method logic features comments in its code. Currently, this method is just a workpiece for creating a full-fledged method of drawing object shadows. Currently, the method simply draws a simple rectangle shifted to the bottom right "under" the current object on the element object created for drawing shadows.

**The method setting a color scheme:**

```
//+------------------------------------------------------------------+
//| Set a color scheme                                               |
//+------------------------------------------------------------------+
void CForm::SetColorTheme(const ENUM_COLOR_THEMES theme,const uchar opacity)
  {
   this.SetOpacity(opacity);
   this.SetColorBackground(array_color_themes[theme][COLOR_THEME_COLOR_FORM_BG]);
   this.SetColorFrame(array_color_themes[theme][COLOR_THEME_COLOR_FORM_FRAME]);
   this.SetColorShadow(array_color_themes[theme][COLOR_THEME_COLOR_FORM_SHADOW]);
  }
//+------------------------------------------------------------------+
```

The method is used for setting a specified color theme for the object. The method receives the required theme and the form object opacity value. The form opacity, form background color, form frame color and form shadow color are then set from the values written in the array of color themes I have created above.

**The method setting the form style:**

```
//+------------------------------------------------------------------+
//| Set the form style                                               |
//+------------------------------------------------------------------+
void CForm::SetFormStyle(const ENUM_FORM_STYLE style,
                         const ENUM_COLOR_THEMES theme,
                         const uchar opacity,
                         const bool shadow=false,
                         const bool redraw=false)
  {
//--- Set opacity parameters and the size of the form frame side
   this.m_shadow=shadow;
   this.m_frame_width_top=array_form_style[style][FORM_STYLE_FRAME_WIDTH_TOP];
   this.m_frame_width_bottom=array_form_style[style][FORM_STYLE_FRAME_WIDTH_BOTTOM];
   this.m_frame_width_left=array_form_style[style][FORM_STYLE_FRAME_WIDTH_LEFT];
   this.m_frame_width_right=array_form_style[style][FORM_STYLE_FRAME_WIDTH_RIGHT];
//--- Set the color scheme
   this.SetColorTheme(theme,opacity);
//--- Create the shadow object and draw a simple distinct shadow
   this.CreateShadow((uchar)array_form_style[style][FORM_STYLE_FRAME_SHADOW_OPACITY]);
   this.DrawShadow((uchar)array_form_style[style][FORM_STYLE_FRAME_SHADOW_OPACITY]);
//--- Fill in the form background with color and opacity
   this.Erase(this.ColorBackground(),this.Opacity());
//--- Depending on the selected form style, draw the corresponding form frame and the outer bounding frame
   switch(style)
     {
      case FORM_STYLE_BEVEL   :
        this.DrawFormFrame(this.m_frame_width_top,this.m_frame_width_bottom,this.m_frame_width_left,this.m_frame_width_right,this.ColorFrame(),this.Opacity(),FRAME_STYLE_BEVEL);
        this.DrawRectangle(0,0,Width()-1,Height()-1,array_color_themes[theme][COLOR_THEME_COLOR_FORM_FRAME_OUTER],this.Opacity());
        break;
      //---FORM_STYLE_FLAT
      default:
        this.DrawFormFrame(this.m_frame_width_top,this.m_frame_width_bottom,this.m_frame_width_left,this.m_frame_width_right,this.ColorFrame(),this.Opacity(),FRAME_STYLE_FLAT);
        this.DrawRectangle(0,0,Width()-1,Height()-1,array_color_themes[theme][COLOR_THEME_COLOR_FORM_FRAME_OUTER],this.Opacity());
        break;
     }
  }
//+------------------------------------------------------------------+
```

The method logic features comments in the listing.

In fact, this method is an example of creating a form object with the required parameters.

**The method drawing a form frame:**

```
//+------------------------------------------------------------------+
//| Draw the form frame                                              |
//+------------------------------------------------------------------+
void CForm::DrawFormFrame(const int wd_top,              // Frame upper segment width
                          const int wd_bottom,           // Frame lower segment width
                          const int wd_left,             // Frame left segment width
                          const int wd_right,            // Frame right segment width
                          const color colour,            // Frame color
                          const uchar opacity,           // Frame opacity
                          const ENUM_FRAME_STYLE style)  // Frame style
  {
//--- Depending on the passed frame style
   switch(style)
     {
      //--- draw a dimensional (convex) frame
      case FRAME_STYLE_BEVEL :
         DrawFrameBevel(0,0,Width(),Height(),wd_top,wd_bottom,wd_left,wd_right,colour,opacity);
        break;
      //--- draw a dimensional (concave) frame
      case FRAME_STYLE_STAMP :
         DrawFrameStamp(0,0,Width(),Height(),wd_top,wd_bottom,wd_left,wd_right,colour,opacity);
        break;
      //--- draw a flat frame
      case FRAME_STYLE_FLAT :
         DrawFrameFlat(0,0,Width(),Height(),wd_top,wd_bottom,wd_left,wd_right,colour,opacity);
        break;
      //--- draw a simple frame
      default:
        //---FRAME_STYLE_SIMPLE
         DrawFrameSimple(0,0,Width(),Height(),wd_top,wd_bottom,wd_left,wd_right,colour,opacity);
        break;
     }
  }
//+------------------------------------------------------------------+
```

Depending on the frame style, draw the appropriate form frame.

**The method drawing a simple frame:**

```
//+------------------------------------------------------------------+
//| Draw a simple frame                                              |
//+------------------------------------------------------------------+
void CForm::DrawFrameSimple(const int x,           // X coordinate relative to the form
                            const int y,           // Y coordinate relative to the form
                            const int width,       // Frame width
                            const int height,      // Frame height
                            const int wd_top,      // Frame upper segment width
                            const int wd_bottom,   // Frame lower segment width
                            const int wd_left,     // Frame left segment width
                            const int wd_right,    // Frame right segment width
                            const color colour,    // Frame color
                            const uchar opacity)   // Frame opacity
  {
//--- Set rectangle coordinates
   int x1=x, y1=y;
   int x2=x1+width-1;
   int y2=y1+height-1;
//--- Draw the first rectangle
   CGCnvElement::DrawRectangle(x1,y1,x2,y2,colour,opacity);
//--- If the frame width exceeds 1 on all sides, draw the second rectangle
   if(wd_left>1 || wd_right>1 || wd_top>1 || wd_bottom>1)
      CGCnvElement::DrawRectangle(x1+wd_left-1,y1+wd_top-1,x2-wd_right+1,y2-wd_bottom+1,colour,opacity);
//--- Search for "voids" between the lines of two rectangles and fill them with color
   if(wd_left>2 && wd_right>2 && wd_top>2 && wd_bottom>2)
      this.Fill(x1+1,y1+1,colour,opacity);
   else if(wd_left>2 && wd_top>2)
      this.Fill(x1+1,y1+1,colour,opacity);
   else if(wd_right>2 && wd_bottom>2)
      this.Fill(x2-1,y2-1,colour,opacity);
   else if(wd_left<3 && wd_right<3)
     {
      if(wd_top>2)
         this.Fill(x1+1,y1+1,colour,opacity);
      if(wd_bottom>2)
         this.Fill(x1+1,y2-1,colour,opacity);
     }
   else if(wd_top<3 && wd_bottom<3)
     {
      if(wd_left>2)
         this.Fill(x1+1,y1+1,colour,opacity);
      if(wd_right>2)
         this.Fill(x2-1,y1+1,colour,opacity);
     }
  }
//+------------------------------------------------------------------+
```

The method logic features detailed comments in the code. In short, draw two rectangles — one inside the other. If there is a void between the rectangles at the spots forming the sides of the future frame (the sides of the rectangles do not overlap), fill them with color matching the one used to draw the rectangles.

**The method drawing a flat frame:**

```
//+------------------------------------------------------------------+
//| Draw the flat frame                                              |
//+------------------------------------------------------------------+
void CForm::DrawFrameFlat(const int x,
                          const int y,
                          const int width,
                          const int height,
                          const int wd_top,
                          const int wd_bottom,
                          const int wd_left,
                          const int wd_right,
                          const color colour,
                          const uchar opacity)
  {
//--- Draw a simple frame
   this.DrawFrameSimple(x,y,width,height,wd_top,wd_bottom,wd_left,wd_right,colour,opacity);
//--- If the width of the frame top and bottom exceeds one pixel
   if(wd_top>1 && wd_bottom>1)
     {
      //--- Darken the horizontal sides of the frame
      for(int i=0;i<width;i++)
        {
         this.m_canvas.PixelSet(x+i,y,CGCnvElement::ChangeColorLightness(this.GetPixel(x+i,y),-0.05));
         this.m_canvas.PixelSet(x+i,y+height-1,CGCnvElement::ChangeColorLightness(this.GetPixel(x+i,y+height-1),-0.07));
        }
     }
//--- If the width of the frame left and right sides exceeds one pixel
   if(wd_left>1 && wd_right>1)
     {
      //--- Darken the vertical sides of the frame
      for(int i=1;i<height-1;i++)
        {
         this.m_canvas.PixelSet(x,y+i,CGCnvElement::ChangeColorLightness(this.GetPixel(x,y+1),-0.01));
         this.m_canvas.PixelSet(x+width-1,y+i,CGCnvElement::ChangeColorLightness(this.GetPixel(x+width-1,y+1),-0.02));
        }
     }
  }
//+------------------------------------------------------------------+
```

**The method drawing an embossed (convex) frame:**

```
//+------------------------------------------------------------------+
//| Draw an embossed (convex) frame                                  |
//+------------------------------------------------------------------+
void CForm::DrawFrameBevel(const int x,
                           const int y,
                           const int width,
                           const int height,
                           const int wd_top,
                           const int wd_bottom,
                           const int wd_left,
                           const int wd_right,
                           const color colour,
                           const uchar opacity)
  {
//--- Draw a simple frame
   this.DrawFrameSimple(x,y,width,height,wd_top,wd_bottom,wd_left,wd_right,colour,opacity);
//--- If the width of the frame top and bottom exceeds one pixel
   if(wd_top>1 && wd_bottom>1)
     {
      //--- Lighten and darken the required sides of the frame edges
      for(int i=0;i<width;i++)
        {
         this.m_canvas.PixelSet(x+i,y,CGCnvElement::ChangeColorLightness(this.GetPixel(x+i,y),0.25));
         this.m_canvas.PixelSet(x+i,y+height-1,CGCnvElement::ChangeColorLightness(this.GetPixel(x+i,y+height-1),-0.2));
        }
      for(int i=wd_left;i<width-wd_right;i++)
        {
         this.m_canvas.PixelSet(x+i,y+wd_top-1,CGCnvElement::ChangeColorLightness(this.GetPixel(x+i,y+wd_top-1),-0.2));
         this.m_canvas.PixelSet(x+i,y+height-wd_bottom,CGCnvElement::ChangeColorLightness(this.GetPixel(x+i,y+height-wd_bottom),0.1));
        }
     }
//--- If the width of the frame left and right sides exceeds one pixel
   if(wd_left>1 && wd_right>1)
     {
      //--- Lighten and darken the required sides of the frame edges
      for(int i=1;i<height-1;i++)
        {
         this.m_canvas.PixelSet(x,y+i,CGCnvElement::ChangeColorLightness(this.GetPixel(x,y+i),0.1));
         this.m_canvas.PixelSet(x+width-1,y+i,CGCnvElement::ChangeColorLightness(this.GetPixel(x+width-1,y+i),-0.1));
        }
      for(int i=wd_top;i<height-wd_bottom;i++)
        {
         this.m_canvas.PixelSet(x+wd_left-1,y+i,CGCnvElement::ChangeColorLightness(this.GetPixel(x+wd_left-1,y+i),-0.1));
         this.m_canvas.PixelSet(x+width-wd_right,y+i,CGCnvElement::ChangeColorLightness(this.GetPixel(x+width-wd_right,y+i),0.1));
        }
     }
  }
//+------------------------------------------------------------------+
```

**The method drawing an embossed (concave) frame:**

```
//+------------------------------------------------------------------+
//| Draw an embossed (concave) frame                                 |
//+------------------------------------------------------------------+
void CForm::DrawFrameStamp(const int x,
                           const int y,
                           const int width,
                           const int height,
                           const int wd_top,
                           const int wd_bottom,
                           const int wd_left,
                           const int wd_right,
                           const color colour,
                           const uchar opacity)
  {
//--- Draw a simple frame
   this.DrawFrameSimple(x,y,width,height,wd_top,wd_bottom,wd_left,wd_right,colour,opacity);
//--- If the width of the frame top and bottom exceeds one pixel
   if(wd_top>1 && wd_bottom>1)
     {
      //--- Lighten and darken the required sides of the frame edges
      for(int i=0;i<width;i++)
        {
         this.m_canvas.PixelSet(x+i,y,CGCnvElement::ChangeColorLightness(this.GetPixel(x+i,y),-0.25));
         this.m_canvas.PixelSet(x+i,y+height-1,CGCnvElement::ChangeColorLightness(this.GetPixel(x+i,y+height-1),0.2));
        }
      for(int i=wd_left;i<width-wd_right;i++)
        {
         this.m_canvas.PixelSet(x+i,y+wd_top-1,CGCnvElement::ChangeColorLightness(this.GetPixel(x+i,y+wd_top-1),0.2));
         this.m_canvas.PixelSet(x+i,y+height-wd_bottom,CGCnvElement::ChangeColorLightness(this.GetPixel(x+i,y+height-wd_bottom),-0.25));
        }
     }
//--- If the width of the frame left and right sides exceeds one pixel
   if(wd_left>1 && wd_right>1)
     {
      //--- Lighten and darken the required sides of the frame edges
      for(int i=1;i<height-1;i++)
        {
         this.m_canvas.PixelSet(x,y+i,CGCnvElement::ChangeColorLightness(this.GetPixel(x,y+i),-0.1));
         this.m_canvas.PixelSet(x+width-1,y+i,CGCnvElement::ChangeColorLightness(this.GetPixel(x+width-1,y+i),0.2));
        }
      for(int i=wd_top;i<height-wd_bottom;i++)
        {
         this.m_canvas.PixelSet(x+wd_left-1,y+i,CGCnvElement::ChangeColorLightness(this.GetPixel(x+wd_left-1,y+i),0.2));
         this.m_canvas.PixelSet(x+width-wd_right,y+i,CGCnvElement::ChangeColorLightness(this.GetPixel(x+width-wd_right,y+i),-0.2));
        }
     }
  }
//+------------------------------------------------------------------+
```

**The methods drawing the fields (simple and embossed ones):**

```
//+------------------------------------------------------------------+
//| Draw a simple field                                              |
//+------------------------------------------------------------------+
void CForm::DrawFieldFlat(const int x,const int y,const int width,const int height,const color colour,const uchar opacity)
  {
//--- Draw a filled rectangle
   CGCnvElement::DrawRectangleFill(x,y,x+width-1,y+height-1,colour,opacity);
//--- Darken all its edges
   for(int i=0;i<width;i++)
     {
      this.m_canvas.PixelSet(x+i,y,CGCnvElement::ChangeColorLightness(this.GetPixel(x+i,y),-0.05));
      this.m_canvas.PixelSet(x+i,y+height-1,CGCnvElement::ChangeColorLightness(this.GetPixel(x+i,y+height-1),-0.05));
     }
   for(int i=1;i<height-1;i++)
     {
      this.m_canvas.PixelSet(x,y+i,CGCnvElement::ChangeColorLightness(this.GetPixel(x,y+1),-0.05));
      this.m_canvas.PixelSet(x+width-1,y+i,CGCnvElement::ChangeColorLightness(this.GetPixel(x+width-1,y+1),-0.05));
     }
  }
//+------------------------------------------------------------------+
//| Draw an embossed (convex) field                                  |
//+------------------------------------------------------------------+
void CForm::DrawFieldBevel(const int x,const int y,const int width,const int height,const color colour,const uchar opacity)
  {
//--- Draw a filled rectangle
   CGCnvElement::DrawRectangleFill(x,y,x+width-1,y+height-1,colour,opacity);
//--- Lighten its top and left and darken its bottom and right
   for(int i=0;i<width;i++)
     {
      this.m_canvas.PixelSet(x+i,y,CGCnvElement::ChangeColorLightness(this.GetPixel(x+i,y),0.1));
      this.m_canvas.PixelSet(x+i,y+height-1,CGCnvElement::ChangeColorLightness(this.GetPixel(x+i,y+height-1),-0.1));
     }
   for(int i=1;i<height-1;i++)
     {
      this.m_canvas.PixelSet(x,y+i,CGCnvElement::ChangeColorLightness(this.GetPixel(x,y+1),0.05));
      this.m_canvas.PixelSet(x+width-1,y+i,CGCnvElement::ChangeColorLightness(this.GetPixel(x+width-1,y+1),-0.05));
     }
  }
//+------------------------------------------------------------------+
//| Draw an embossed (concave) field                                 |
//+------------------------------------------------------------------+
void CForm::DrawFieldStamp(const int x,const int y,const int width,const int height,const color colour,const uchar opacity)
  {
//--- Draw a filled rectangle
   CGCnvElement::DrawRectangleFill(x,y,x+width-1,y+height-1,colour,opacity);
//--- Darken its top and left and lighten its bottom and right
   for(int i=0;i<width;i++)
     {
      this.m_canvas.PixelSet(x+i,y,CGCnvElement::ChangeColorLightness(this.GetPixel(x+i,y),-0.1));
      this.m_canvas.PixelSet(x+i,y+height-1,CGCnvElement::ChangeColorLightness(this.GetPixel(x+i,y+height-1),0.1));
     }
   for(int i=1;i<height-1;i++)
     {
      this.m_canvas.PixelSet(x,y+i,CGCnvElement::ChangeColorLightness(this.GetPixel(x,y+1),-0.05));
      this.m_canvas.PixelSet(x+width-1,y+i,CGCnvElement::ChangeColorLightness(this.GetPixel(x+width-1,y+1),0.05));
     }
  }
//+------------------------------------------------------------------+
```

The logic of all the above methods is almost identical and features comments in the method code. I believe, the methods are quite comprehensible and easy to grasp. In any case, you are welcome to use the comments section.

**This completes the creation of the form object for now.**

### Test

Today's test will be pretty simple. I will create two different forms with different construction styles and color themes. After the forms are created, I will add the fields to them. The upper form will get a dimensional concave field, while the second form will get a semi-transparent dimensional concave field.

To perform the test, let's use [the EA from the previous article](https://www.mql5.com/en/articles/9515#node05) and save it to \\MQL5\\Experts\\TestDoEasy\ **Part76\** as **TestDoEasyPart76.mq5**.

Include the library form object file into the EA and rename the list of element objects into the list of form objects:

```
//+------------------------------------------------------------------+
//|                                             TestDoEasyPart76.mq5 |
//|                                  Copyright 2021, MetaQuotes Ltd. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2021, MetaQuotes Ltd."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//--- includes
#include <Arrays\ArrayObj.mqh>
#include <DoEasy\Services\Select.mqh>
#include <DoEasy\Objects\Graph\Form.mqh>
//--- defines
#define        FORMS_TOTAL (2)   // Number of created forms
//--- input parameters
sinput   bool  InpMovable  = true;  // Movable flag
//--- global variables
CArrayObj      list_forms;
//+------------------------------------------------------------------+
```

In the **OnInit()** handler, create two forms and draw concave fields on them:

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

//--- Create the specified number of form objects
   list_forms.Clear();
   int total=FORMS_TOTAL;
   for(int i=0;i<total;i++)
     {
      //--- When creating an object, pass all the required parameters to it
      CForm *form=new CForm("Form_0"+(string)(i+1),300,40+(i*80),100,70);
      if(form==NULL)
         continue;
      //--- Set activity and moveability flags for the form
      form.SetActive(true);
      form.SetMovable(false);
      //--- Set the form ID equal to the loop index and the index in the list of objects
      form.SetID(i);
      form.SetNumber(0);   // (0 - main form object) Auxiliary objects may be attached to the main one. The main object is able to manage them
      //--- Set the full opacity for the top form and the partial opacity for the bottom one
      uchar opacity=(i==0 ? 255 : 250);
      //--- Set the form style and its color theme depending on the loop index
      ENUM_FORM_STYLE style=(ENUM_FORM_STYLE)i;
      ENUM_COLOR_THEMES theme=(ENUM_COLOR_THEMES)i;
      //--- Set the form style and theme
      form.SetFormStyle(style,theme,opacity,true);
      //--- If this is the first (top) form
      if(i==0)
        {
         //--- Draw a concave field slightly shifted from the center of the form downwards
         form.DrawFieldStamp(3,10,form.Width()-6,form.Height()-13,form.ColorBackground(),form.Opacity());
         form.Update(true);
        }
      //--- If this is the second (bottom) form
      if(i==1)
        {
         //--- Draw a concave semi-transparent "tainted glass" field in the center
         form.DrawFieldStamp(10,10,form.Width()-20,form.Height()-20,clrWheat,200);
         form.Update(true);
        }
      //--- Add objects to the list
      if(!list_forms.Add(form))
        {
         delete form;
         continue;
        }
     }
//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
```

Remove handling mouse clicks on objects from the **OnChartEvent()** handler:

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

     }
  }
//+------------------------------------------------------------------+
```

Compile the EA and launch it on the chart:

![](https://c.mql5.com/2/42/jbZuzu7s4h.gif)

As you can see, we have managed to create two different forms with different colors of the components and drawing style by simply specifying the desired style and color theme.

### What's next?

In the next article, I will continue the development of the form object and supplement its functionality.

All files of the current version of the library are attached below together with the test EA file for MQL5 for you to test and download.

Leave your questions and suggestions in the comments.

[Back to contents](https://www.mql5.com/en/articles/9553#node00)

**\*Previous articles within the series:**

[Graphics in DoEasy library (Part 73): Form object of a graphical element](https://www.mql5.com/en/articles/9442)

[Graphics in DoEasy library (Part 74): Basic graphical element powered by the CCanvas class](https://www.mql5.com/en/articles/9493)

[Graphics in DoEasy library (Part 75): Methods of handling primitives and text in the basic graphical element](https://www.mql5.com/en/articles/9515)

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/9553](https://www.mql5.com/ru/articles/9553)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/9553.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/9553/mql5.zip "Download MQL5.zip")(3980.07 KB)

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

**[Go to discussion](https://www.mql5.com/en/forum/373786)**

![Graphics in DoEasy library (Part 77): Shadow object class](https://c.mql5.com/2/42/MQL5-avatar-doeasy-library3-2__4.png)[Graphics in DoEasy library (Part 77): Shadow object class](https://www.mql5.com/en/articles/9575)

In this article, I will create a separate class for the shadow object, which is a descendant of the graphical element object, as well as add the ability to fill the object background with a gradient fill.

![Graphics in DoEasy library (Part 75): Methods of handling primitives and text in the basic graphical element](https://c.mql5.com/2/42/MQL5-avatar-doeasy-library3-2__2.png)[Graphics in DoEasy library (Part 75): Methods of handling primitives and text in the basic graphical element](https://www.mql5.com/en/articles/9515)

In this article, I will continue the development of the basic graphical element class of all library graphical objects powered by the CCanvas Standard Library class. I will create the methods for drawing graphical primitives and for displaying a text on a graphical element object.

![Better Programmer (Part 01): You must stop doing these 5 things to become a successful MQL5 programmer](https://c.mql5.com/2/42/Article_image.png)[Better Programmer (Part 01): You must stop doing these 5 things to become a successful MQL5 programmer](https://www.mql5.com/en/articles/9643)

There are a lot of bad habits that newbies and even advanced programmers are doing that are keeping them from becoming the best they can be to their coding career. We are going to discuss and address them in this article. This article is a must read for everyone who wants to become successful developer in MQL5.

![Graphics in DoEasy library (Part 74): Basic graphical element powered by the CCanvas class](https://c.mql5.com/2/42/MQL5-avatar-doeasy-library3-2__1.png)[Graphics in DoEasy library (Part 74): Basic graphical element powered by the CCanvas class](https://www.mql5.com/en/articles/9493)

In this article, I will rework the concept of building graphical objects from the previous article and prepare the base class of all graphical objects of the library powered by the Standard Library CCanvas class.

[![](https://www.mql5.com/ff/si/w766tj9vyj3g607n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fmarket%2Fmt5%2Fexpert%3FHasRent%3Don%26utm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Drent.expert%26utm_content%3Drent.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=sorsafcerhkgwrjzwwrpvelbicxjwzon&s=ae91b1eae8acb61167455495742e6cc8eb55ccedb33fd953f8256b68cbe9c3b4&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=oqyjqtwywenyqdcgkosuzkfscxlehtbz&ssn=1769253088374238980&ssn_dr=0&ssn_sr=0&fv_date=1769253088&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F9553&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Graphics%20in%20DoEasy%20library%20(Part%2076)%3A%20Form%20object%20and%20predefined%20color%20themes%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176925308848555214&fz_uniq=5083393004061858528&sv=2552)

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