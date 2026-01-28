---
title: Graphics in DoEasy library (Part 73): Form object of a graphical element
url: https://www.mql5.com/en/articles/9442
categories: Expert Advisors
relevance_score: 0
scraped_at: 2026-01-24T14:12:01.860291
---

[Running robots on virtual hosting is easyFollow our step-by-step MetaTrader VPS guide for beginnersRead![](https://www.mql5.com/ff/sh/au4fqg4kms7s9mq1z2/01.png)![](https://www.mql5.com/ff/sh/au4fqg4kms7s9mq1z2/02.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/articles/13586&a=uzpprdshbcrtxvjxpmescehprypbymxc&s=516438f25b531570d9b7d49dcfb29c82fa1021f5ede6571df8026dbfbafcd13f&uid=&ref=https://www.mql5.com/en/articles/9442&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5083399132980189951)

MetaTrader 5 / Examples


### Contents

- [Concept](https://www.mql5.com/en/articles/9442#node01)
- [Improving library classes](https://www.mql5.com/en/articles/9442#node02)
- [Mouse status class](https://www.mql5.com/en/articles/9442#node03)
- [Class of the base object of all library graphical elements](https://www.mql5.com/en/articles/9442#node04)
- [Class of the form object of graphical elements](https://www.mql5.com/en/articles/9442#node05)
- [Test](https://www.mql5.com/en/articles/9442#node06)
- [What's next?](https://www.mql5.com/en/articles/9442#node07)


### Concept

Modern programs, especially analytical ones, are able to use huge amounts of data in their calculations. However, it would be difficult to understand something without visualization. Besides, it would be quite challenging to use the program to its full extent without a clear and convenient interface. Naturally, the ability to work with graphics is a must for our library as well. Therefore, the article starts a large section about working with graphical elements.

My objective is to create a convenient functionality for creating a wide range of different graphical objects, allow all main library classes to interactively work with graphics using custom graphical objects, as well as create graphical objects featuring component hierarchies of any complexity.

Let's get started with graphical objects based on [the CCanvas standard library class](https://www.mql5.com/en/docs/standardlibrary/canvasgraphics/ccanvas). The class allows for easy creation of any custom images and using them as "bricks" to build more complex objects. It is possible to either use ready-made images, or draw custom images on a created canvas. Personally, I find the latter option more exciting. So I am going to widely use it to design my graphical objects.

The hierarchy of a single object always looks as follows:

1. The base object of all library graphical elements based on the [CObject class](https://www.mql5.com/en/docs/standardlibrary/cobject). The CCanvas class object is declared in it. Besides, it contains all parameters common to graphical elements, including width, height, chart coordinates, right and bottom object borders, etc.,
2. The object-form of a graphical element — it represents a basis (canvas) of any graphical object. It is to contain all other elements of the composite object. Its parameters enable you to set parameters for the entire graphical object. The object of the class providing the methods of working with mouse status (cursor coordinates and pressed buttons) is declared here as well.

The hierarchy forms a hardcore of the base element of all library graphical objects based on the CCanvas class. All other created objects are based on this object and inherit its basic properties.

But first, let's slightly improve the ready-made library classes and add new data for the objects I am going to create here.

### Improving library classes

Add the new subsection of canvas parameters to \\MQL5\\Include\\DoEasy\ **Defines.mqh** and add a macro substitution with its update frequency:

```
//--- Parameters of the DOM snapshot series
#define MBOOKSERIES_DEFAULT_DAYS_COUNT (1)                        // The default required number of days for DOM snapshots in the series
#define MBOOKSERIES_MAX_DATA_TOTAL     (200000)                   // Maximum number of stored DOM snapshots of a single symbol
//--- Canvas parameters
#define PAUSE_FOR_CANV_UPDATE          (16)                       // Canvas update frequency
//+------------------------------------------------------------------+
//| Enumerations                                                     |
//+------------------------------------------------------------------+
```

Canvas-based objects should be updated (re-drawn) not more often than 16 milliseconds. This eliminates unnecessary redrawing of the screen, which is invisible to the human eye but still puts an extra load on the system. So, before updating a canvas-based object, we need to check how many milliseconds have passed since its previous update. Setting an optimal latency enables us to achieve an acceptable display of the screen featuring graphical objects.

I will create the class of the mouse status object to define the status of mouse buttons, as well as of Shift and Ctrl keys. To achieve this, I will need two enumerations: the list of possible mouse buttons, as well Shift and Ctrl keys states and the list of possible mouse states relative to the form. Add them to the end of the listing file:

```
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| Data for working with mouse                                      |
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| The list of possible mouse buttons, Shift and Ctrl keys states   |
//+------------------------------------------------------------------+
enum ENUM_MOUSE_BUTT_KEY_STATE
  {
   MOUSE_BUTT_KEY_STATE_NONE              = 0,        // Nothing is clicked
//--- Mouse buttons
   MOUSE_BUTT_KEY_STATE_LEFT              = 1,        // The left mouse button is clicked
   MOUSE_BUTT_KEY_STATE_RIGHT             = 2,        // The right mouse button is clicked
   MOUSE_BUTT_KEY_STATE_MIDDLE            = 16,       // The middle mouse button is clicked
   MOUSE_BUTT_KEY_STATE_WHELL             = 128,      // Scrolling the mouse wheel
   MOUSE_BUTT_KEY_STATE_X1                = 32,       // The first additional mouse button is clicked
   MOUSE_BUTT_KEY_STATE_X2                = 64,       // The second additional mouse button is clicked
   MOUSE_BUTT_KEY_STATE_LEFT_RIGHT        = 3,        // The left and right mouse buttons clicked
//--- Keyboard keys
   MOUSE_BUTT_KEY_STATE_SHIFT             = 4,        // Shift is being held
   MOUSE_BUTT_KEY_STATE_CTRL              = 8,        // Ctrl is being held
   MOUSE_BUTT_KEY_STATE_CTRL_CHIFT        = 12,       // Ctrl and Shift are being held
//--- Left mouse button combinations
   MOUSE_BUTT_KEY_STATE_LEFT_WHELL        = 129,      // The left mouse button is clicked and the wheel is being scrolled
   MOUSE_BUTT_KEY_STATE_LEFT_SHIFT        = 5,        // The left mouse button is clicked and Shift is being held
   MOUSE_BUTT_KEY_STATE_LEFT_CTRL         = 9,        // The left mouse button is clicked and Ctrl is being held
   MOUSE_BUTT_KEY_STATE_LEFT_CTRL_CHIFT   = 13,       // The left mouse button is clicked, Ctrl and Shift are being held
//--- Right mouse button combinations
   MOUSE_BUTT_KEY_STATE_RIGHT_WHELL       = 130,      // The right mouse button is clicked and the wheel is being scrolled
   MOUSE_BUTT_KEY_STATE_RIGHT_SHIFT       = 6,        // The right mouse button is clicked and Shift is being held
   MOUSE_BUTT_KEY_STATE_RIGHT_CTRL        = 10,       // The right mouse button is clicked and Ctrl is being held
   MOUSE_BUTT_KEY_STATE_RIGHT_CTRL_CHIFT  = 14,       // The right mouse button is clicked, Ctrl and Shift are being held
//--- Middle mouse button combinations
   MOUSE_BUTT_KEY_STATE_MIDDLE_WHEEL      = 144,      // The middle mouse button is clicked and the wheel is being scrolled
   MOUSE_BUTT_KEY_STATE_MIDDLE_SHIFT      = 20,       // The middle mouse button is clicked and Shift is being held
   MOUSE_BUTT_KEY_STATE_MIDDLE_CTRL       = 24,       // The middle mouse button is clicked and Ctrl is being held
   MOUSE_BUTT_KEY_STATE_MIDDLE_CTRL_CHIFT = 28,       // The middle mouse button is clicked, Ctrl and Shift are being held
  };
//+------------------------------------------------------------------+
//| The list of possible mouse states relative to the form           |
//+------------------------------------------------------------------+
enum ENUM_MOUSE_FORM_STATE
  {
   MOUSE_FORM_STATE_NONE = 0,                         // Undefined state
//--- Outside the form
   MOUSE_FORM_STATE_OUTSIDE_NOT_PRESSED,              // The cursor is outside the form, the mouse buttons are not clicked
   MOUSE_FORM_STATE_OUTSIDE_PRESSED,                  // The cursor is outside the form, any mouse button is clicked
   MOUSE_FORM_STATE_OUTSIDE_WHEEL,                    // The cursor is outside the form, the mouse wheel is being scrolled
//--- Within the form
   MOUSE_FORM_STATE_INSIDE_NOT_PRESSED,               // The cursor is inside the form, the mouse buttons are not clicked
   MOUSE_FORM_STATE_INSIDE_PRESSED,                   // The cursor is inside the form, any mouse button is clicked
   MOUSE_FORM_STATE_INSIDE_WHEEL,                     // The cursor is inside the form, the mouse wheel is being scrolled
//--- Within the window header area
   MOUSE_FORM_STATE_INSIDE_ACTIVE_AREA_NOT_PRESSED,   // The cursor is inside the active area, the mouse buttons are not clicked
   MOUSE_FORM_STATE_INSIDE_ACTIVE_AREA_PRESSED,       // The cursor is inside the active area,  any mouse button is clicked
   MOUSE_FORM_STATE_INSIDE_ACTIVE_AREA_WHEEL,         // The cursor is inside the active area, the mouse wheel is being scrolled
//--- Within the window scrolling area
   MOUSE_FORM_STATE_INSIDE_SCROLL_NOT_PRESSED,        // The cursor is within the window scrolling area, the mouse buttons are not clicked
   MOUSE_FORM_STATE_INSIDE_SCROLL_PRESSED,            // The cursor is within the window scrolling area, any mouse button is clicked
   MOUSE_FORM_STATE_INSIDE_SCROLL_WHEEL,              // The cursor is within the window scrolling area, the mouse wheel is being scrolled
  };
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| Data for handling graphical elements                             |
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| The list of graphical element types                              |
//+------------------------------------------------------------------+
enum ENUM_GRAPH_ELEMENT_TYPE
  {
   GRAPH_ELEMENT_TYPE_FORM,                           // Simple form
   GRAPH_ELEMENT_TYPE_WINDOW,                         // Window
  };
//+------------------------------------------------------------------+
```

The list of graphical element types has been added to "secure seats" for subsequent classes that are based on the ones created here — the lists will be filled in and used in future articles.

The list of possible mouse buttons, Shift and Ctrl states features basic mouse and keys events, as well as some of their combinations that will probably be needed most often.

The mouse states are in fact a simple set of bit flags described in the help for the[CHARTEVENT\_MOUSE\_MOVE](https://www.mql5.com/en/docs/constants/chartconstants/enum_chartevents) event.

The table specifies bits and corresponding mouse buttons, Shift and Ctrl states:

| Bit | Description | Value |
| --- | --- | --- |
| 0 | Left mouse button status | 1 |
| 1 | Right mouse button status | 2 |
| 2 | SHIFT status | 4 |
| 3 | CTRL status | 8 |
| 4 | Middle mouse button status | 16 |
| 5 | The first additional mouse button status | 32 |
| 6 | The second additional mouse button status | 64 |

The table allows us to define mouse events by the number set in the variable storing mouse status bits:

- If only the left button is clicked, the variable is equal to 1
- If only the left button is clicked, then 2
- If both buttons are clicked, 1 + 2 = 3
- If only the left button is clicked, while Shift is being held, 1 + 4 = 5

It is for this reason that the values in the ENUM\_MOUSE\_BUTT\_KEY\_STATE enumeration are set exactly in accordance with the displayed calculation of the variable, while the flags described by the enumeration constants are activated.

The ENUM\_MOUSE\_FORM\_STATE enumeration serves for specifying the mouse cursor position relative to the form with clicked/released mouse buttons. We will need the values of the enumeration constants to define the relative position of the mouse cursor, its buttons and the object we are to interact with.

We will be able to store these two enumerations in two bytes of the ushort variable to immediately grasp the whole picture of what is happening with the mouse and its interaction object. The table contains the entire bitmap of the variable:

| Bit | Byte | State | Value |
| --- | --- | --- | --- |
| 0 | 0 | left mouse button | 1 |
| 1 | 0 | right mouse button | 2 |
| 2 | 0 | SHIFT | 4 |
| 3 | 0 | CTRL | 8 |
| 4 | 0 | middle mouse button | 16 |
| 5 | 0 | first additional mouse button | 32 |
| 6 | 0 | second additional mouse button | 64 |
| 7 | 0 | scrolling the wheel | 128 |
| 8 (0) | 1 | cursor inside the form | 256 |
| 9 (1) | 1 | cursor inside the form active area | 512 |
| 10 (2) | 1 | cursor inside the window control area (minimize/maximize/close, etc.) | 1024 |
| 11 (3) | 1 | cursor within the window scrolling area | 2048 |
| 12 (4) | 1 | cursor at the left edge of the form | 4096 |
| 13 (5) | 1 | cursor at the bottom edge of the form | 8192 |
| 14 (6) | 1 | cursor at the right edge of the form | 16384 |
| 15 (7) | 1 | cursor at the top edge of the form | 32768 |

The flags indicating mouse states and cursor positions relative to the form object and window object based on the form are sufficient for now.

Let's slightly improve the pause class object in \\MQL5\\Include\\DoEasy\\Services\ **Pause.mqh**.

Its method SetTimeBegin(), apart from setting a new pause countdown time, also sets the time passed to the method, namely to the **m\_time\_begin** variable.

This is only required to send data to the journal and it is not needed if we just want to count a pause somewhere inside the method. We can easily pass any time (including zero) to the method, but I have decided to implement the method overload without specifying the time:

```
//--- Set the new (1) countdown start time and (2) pause in milliseconds
   void              SetTimeBegin(const ulong time)         { this.m_time_begin=time; this.SetTimeBegin();              }
   void              SetTimeBegin(void)                     { this.m_start=this.TickCount();                            }
   void              SetWaitingMSC(const ulong pause)       { this.m_wait_msc=pause;                                    }
```

Now we are able to create the mouse status object class.

### Mouse status class

In the \\MQL5\\Include\\DoEasy\ **Services\** folder of service functions and classes, create the CMouseState class in **MouseState.mqh**.

In the private section of the class, declare the variables for storing object parameters and two methods for setting the flags of mouse buttons and keys states. Leave the instructions concerning the location of the bit flags in the ushort variable for storing the bit flags of mouse states:

```
//+------------------------------------------------------------------+
//|                                                   MouseState.mqh |
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
#include "DELib.mqh"
//+------------------------------------------------------------------+
//| Mouse status class                                               |
//+------------------------------------------------------------------+
class CMouseState
  {
private:
   int               m_coord_x;                             // X coordinate
   int               m_coord_y;                             // Y coordinate
   int               m_delta_wheel;                         // Mouse wheel scroll value
   int               m_window_num;                          // Subwindow index
   long              m_chart_id;                            // Chart ID
   ushort            m_state_flags;                         // Status flags

//--- Set the status of mouse buttons, as well as of Shift and Ctrl keys
   void              SetButtonKeyState(const int id,const long lparam,const double dparam,const ushort flags);
//--- Set the mouse buttons and keys status flags
   void              SetButtKeyFlags(const short flags);

//--- Data location in the ushort value of the button status
   //-----------------------------------------------------------------
   //   bit    |    byte   |            state            |    dec    |
   //-----------------------------------------------------------------
   //    0     |     0     | left mouse button           |     1     |
   //-----------------------------------------------------------------
   //    1     |     0     | right mouse button          |     2     |
   //-----------------------------------------------------------------
   //    2     |     0     | SHIFT key                   |     4     |
   //-----------------------------------------------------------------
   //    3     |     0     | CTRL key                    |     8     |
   //-----------------------------------------------------------------
   //    4     |     0     | middle mouse button         |    16     |
   //-----------------------------------------------------------------
   //    5     |     0     | 1 add. mouse button         |    32     |
   //-----------------------------------------------------------------
   //    6     |     0     | 2 add. mouse button         |    64     |
   //-----------------------------------------------------------------
   //    7     |     0     | scrolling the wheel         |    128    |
   //-----------------------------------------------------------------
   //-----------------------------------------------------------------
   //    0     |     1     | cursor inside the form      |    256    |
   //-----------------------------------------------------------------
   //    1     |     1     | cursor inside active area   |    512    |
   //-----------------------------------------------------------------
   //    2     |     1     | cursor in the control area  |   1024    |
   //-----------------------------------------------------------------
   //    3     |     1     | cursor in the scrolling area|   2048    |
   //-----------------------------------------------------------------
   //    4     |     1     | cursor at the left edge     |   4096    |
   //-----------------------------------------------------------------
   //    5     |     1     | cursor at the bottom edge   |   8192    |
   //-----------------------------------------------------------------
   //    6     |     1     | cursor at the right edge    |   16384   |
   //-----------------------------------------------------------------
   //    7     |     1     | cursor at the top edge      |   32768   |
   //-----------------------------------------------------------------

public:
```

In the public section of the class, set the methods returning the object property values:

```
public:
//--- Reset the states of all buttons and keys
   void              ResetAll(void);
//--- Set (1) the subwindow index and (2) the chart ID
   void              SetWindowNum(const int wnd_num)           { this.m_window_num=wnd_num;        }
   void              SetChartID(const long id)                 { this.m_chart_id=id;               }
//--- Return the variable with the mouse status flags
   ushort            GetMouseFlags(void)                       { return this.m_state_flags;        }
//--- Return (1-2) the cursor coordinates, (3) scroll wheel value, (4) status of the mouse buttons and Shift/Ctrl keys
   int               CoordX(void)                        const { return this.m_coord_x;            }
   int               CoordY(void)                        const { return this.m_coord_y;            }
   int               DeltaWheel(void)                    const { return this.m_delta_wheel;        }
   ENUM_MOUSE_BUTT_KEY_STATE ButtKeyState(const int id,const long lparam,const double dparam,const string flags);

//--- Return the flag of the clicked (1) left, (2) right, (3) middle, (4) first and (5) second additional mouse buttons
   bool              IsPressedButtonLeft(void)           const { return this.m_state_flags==1;     }
   bool              IsPressedButtonRight(void)          const { return this.m_state_flags==2;     }
   bool              IsPressedButtonMiddle(void)         const { return this.m_state_flags==16;    }
   bool              IsPressedButtonX1(void)             const { return this.m_state_flags==32;    }
   bool              IsPressedButtonX2(void)             const { return this.m_state_flags==64;    }
//--- Return the flag of the pressed (1) Shift, (2) Ctrl, (3) Shift+Ctrl key and the flag of scrolling the mouse wheel
   bool              IsPressedKeyShift(void)             const { return this.m_state_flags==4;     }
   bool              IsPressedKeyCtrl(void)              const { return this.m_state_flags==8;     }
   bool              IsPressedKeyCtrlShift(void)         const { return this.m_state_flags==12;    }
   bool              IsWheel(void)                       const { return this.m_state_flags==128;   }

//--- Return the flag indicating the status of the left mouse button and (1) the mouse wheel, (2) Shift, (3) Ctrl, (4) Ctrl+Shift
   bool              IsPressedButtonLeftWheel(void)      const { return this.m_state_flags==129;   }
   bool              IsPressedButtonLeftShift(void)      const { return this.m_state_flags==5;     }
   bool              IsPressedButtonLeftCtrl(void)       const { return this.m_state_flags==9;     }
   bool              IsPressedButtonLeftCtrlShift(void)  const { return this.m_state_flags==13;    }
//--- Return the flag indicating the status of the right mouse button and (1) the mouse wheel, (2) Shift, (3) Ctrl, (4) Ctrl+Shift
   bool              IsPressedButtonRightWheel(void)     const { return this.m_state_flags==130;   }
   bool              IsPressedButtonRightShift(void)     const { return this.m_state_flags==6;     }
   bool              IsPressedButtonRightCtrl(void)      const { return this.m_state_flags==10;    }
   bool              IsPressedButtonRightCtrlShift(void) const { return this.m_state_flags==14;    }
//--- Return the flag indicating the status of the middle mouse button and (1) the mouse wheel, (2) Shift, (3) Ctrl, (4) Ctrl+Shift
   bool              IsPressedButtonMiddleWheel(void)    const { return this.m_state_flags==144;   }
   bool              IsPressedButtonMiddleShift(void)    const { return this.m_state_flags==20;    }
   bool              IsPressedButtonMiddleCtrl(void)     const { return this.m_state_flags==24;    }
   bool              IsPressedButtonMiddleCtrlShift(void)const { return this.m_state_flags==28;    }

//--- Constructor/destructor
                     CMouseState();
                    ~CMouseState();
  };
//+------------------------------------------------------------------+
```

Here we have the methods returning the class variables and some methods returning predefined mouse buttons and Ctrl/Shift keys states.

**In the class constructor**, call the methodresetting the states of button and key flags, as well as resetting the mouse wheel scrolling value:

```
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CMouseState::CMouseState() : m_delta_wheel(0),m_coord_x(0),m_coord_y(0),m_window_num(0)
  {
   this.ResetAll();
  }
//+------------------------------------------------------------------+
//| Destructor                                                       |
//+------------------------------------------------------------------+
CMouseState::~CMouseState()
  {
  }
//+------------------------------------------------------------------+
//| Reset the states of all buttons and keys                         |
//+------------------------------------------------------------------+
void CMouseState::ResetAll(void)
  {
   this.m_delta_wheel = 0;
   this.m_state_flags = 0;
  }
//+------------------------------------------------------------------+
```

**The method setting the status of mouse buttons, as well as of Shift/Ctrl keys:**

```
//+------------------------------------------------------------------+
//| Set the status of mouse buttons, as well as of Shift/Ctrl keys   |
//+------------------------------------------------------------------+
void CMouseState::SetButtonKeyState(const int id,const long lparam,const double dparam,const ushort flags)
  {
   //--- Reset the values of all mouse status bits
   this.ResetAll();
   //--- If a chart or an object is left-clicked
   if(id==CHARTEVENT_CLICK || id==CHARTEVENT_OBJECT_CLICK)
     {
      //--- Write the appropriate chart coordinates and set the bit of 0
      this.m_coord_x=(int)lparam;
      this.m_coord_y=(int)dparam;
      this.m_state_flags |=(0x0001);
     }
   //--- otherwise
   else
     {
      //--- in case of a mouse wheel scrolling
      if(id==CHARTEVENT_MOUSE_WHEEL)
        {
         //--- get the cursor coordinates and the total scroll value (the minimum of +120 or -120)
         this.m_coord_x=(int)(short)lparam;
         this.m_coord_y=(int)(short)(lparam>>16);
         this.m_delta_wheel=(int)dparam;
         //--- Call the method of setting flags indicating the states of the mouse buttons and Shift/Ctrl keys
         this.SetButtKeyFlags((short)(lparam>>32));
         //--- and set the bit of 8
         this.m_state_flags &=0xFF7F;
         this.m_state_flags |=(0x0001<<7);
        }
      //--- If this is a cursor movement, write its coordinates and
      //--- call the method of setting flags indicating the states of the mouse buttons and Shift/Ctrl keys
      if(id==CHARTEVENT_MOUSE_MOVE)
        {
         this.m_coord_x=(int)lparam;
         this.m_coord_y=(int)dparam;
         this.SetButtKeyFlags(flags);
        }
     }
  }
//+------------------------------------------------------------------+
```

Here we check, which chart event is being handled.

First, reset all bits in the variable storing mouse status bit flags.

Next, at the event of a mouse click on a chart or an object, set the bit of 0 to the variable storing the bit flags.

In case of a mouse wheel scrolling event, the lparam integer parameter contains data on the cursor coordinates, scrolling size and bit flags of button and Ctrl/Shift states. Extract all the data from the lparam variable and write them to the variables storing the cursor coordinates and to the custom variable with bit flags so that the bit order described in the private section of the class is observed. Next, set the bit of 8 indicating the mouse wheel scrolling event.

When moving the cursor over the chart, write the cursor coordinates to the variables and call the method of setting the bit flags indicating the mouse buttons and Ctrl/Shift status.

**The method setting the flags indicating mouse buttons and keys states:**

```
//+------------------------------------------------------------------+
//| Set the mouse buttons and keys status flags                      |
//+------------------------------------------------------------------+
void CMouseState::SetButtKeyFlags(const short flags)
  {
//--- Left mouse button status
   if((flags & 0x0001)!=0) this.m_state_flags |=(0x0001<<0);
//--- Right mouse button status
   if((flags & 0x0002)!=0) this.m_state_flags |=(0x0001<<1);
//--- SHIFT status
   if((flags & 0x0004)!=0) this.m_state_flags |=(0x0001<<2);
//--- CTRL status
   if((flags & 0x0008)!=0) this.m_state_flags |=(0x0001<<3);
//--- Middle mouse button status
   if((flags & 0x0010)!=0) this.m_state_flags |=(0x0001<<4);
//--- The first additional mouse button status
   if((flags & 0x0020)!=0) this.m_state_flags |=(0x0001<<5);
//--- The second additional mouse button status
   if((flags & 0x0040)!=0) this.m_state_flags |=(0x0001<<6);
  }
//+------------------------------------------------------------------+
```

Here all is simple: the method receives the variable featuring the mouse status flags. Apply the bit mask to it with the verified bit installed one at a time. The value obtained after applying the bit mask will be true due to the bitwise "AND" only if both verified bits are installed (1). If the variable with the applied mask is not equal to zero (the verified bit is installed), write the appropriate bit to the variable for storing bit flags.

**The method returning the mouse buttons and Shift/Ctrl keys states:**

```
//+------------------------------------------------------------------+
//| Return the mouse buttons and Shift/Ctrl keys states              |
//+------------------------------------------------------------------+
ENUM_MOUSE_BUTT_KEY_STATE CMouseState::ButtKeyState(const int id,const long lparam,const double dparam,const string flags)
  {
   this.SetButtonKeyState(id,lparam,dparam,(ushort)flags);
   return (ENUM_MOUSE_BUTT_KEY_STATE)this.m_state_flags;
  }
//+------------------------------------------------------------------+
```

Here we first call the method checking and setting all mouse buttons and Ctrl/Shift keys status flags, and return the **m\_state\_flags** variable value as the ENUM\_MOUSE\_BUTT\_KEY\_STATE enumeration. In the enumeration, the values of all constants correspond to the value obtained by the set of installed variable bits. So, we immediately return one of the enumeration values, which is then to be handled in the classes requiring the states of the mouse, its buttons and Ctrl/Shift keys. The method is called from the [OnChartEvent()](https://www.mql5.com/en/docs/event_handlers/onchartevent) handler.

### Class of the base object of all library graphical elements

Just like the main library classes are descendants of the [standard library base class](https://www.mql5.com/en/docs/standardlibrary/cobject), all classes of graphical element objects should be inherited from it. Such an inheritance allows working with each graphical object as with a standard MQL5 object. Namely, it is important for us to be able to handle different types of graphical objects the way we handle the CObject class object. To achieve this, we need to create a new basic object which is a descendant of the CObject object and contains the common variables and methods for each (and any) library graphical object.

Below are the common properties inherent in each graphical object and present in the base graphical object:

- object coordinates on a chart;
- width and height of an element (canvas), which is to feature other elements of composite objects (containing the same properties that are common to all objects);
- coordinates of the right and bottom canvas edges (the left and top edges correspond to the coordinates);
- various object IDs (object type, name, as well as chart and subwindow IDs);
- and some additional flags that specify the behavior of the object when interacting with it.


The class will be very simple: private variables, protected methods for setting and public methods for returning their values.

The class is to be a descendant of the base class of the CObject standard library.

In \\MQL5\\Include\\DoEasy\ **Objects\**, create the **Graph\** folder containing the **GBaseObj.mqh** file of the CGBaseObj class:

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
//+------------------------------------------------------------------+
//| Class of the base object of the library graphical objects        |
//+------------------------------------------------------------------+
class CGBaseObj : public CObject
  {
private:
   int               m_type;                                // Object type
   string            m_name_obj;                            // Object name
   long              m_chart_id;                            // Chart ID
   int               m_wnd_num;                             // Chart subwindow index
   int               m_coord_x;                             // Canvas X coordinate
   int               m_coord_y;                             // Canvas Y coordinate
   int               m_width;                               // Width
   int               m_height;                              // Height
   bool              m_movable;                             // Object movability flag
   bool              m_selectable;                          // Object selectability flag

protected:
//--- Set the values to class variables
   void              SetNameObj(const string name)             { this.m_name_obj=name;                            }
   void              SetChartID(const long chart_id)           { this.m_chart_id=chart_id;                        }
   void              SetWindowNum(const int wnd_num)           { this.m_wnd_num=wnd_num;                          }
   void              SetCoordX(const int coord_x)              { this.m_coord_x=coord_x;                          }
   void              SetCoordY(const int coord_y)              { this.m_coord_y=coord_y;                          }
   void              SetWidth(const int width)                 { this.m_width=width;                              }
   void              SetHeight(const int height)               { this.m_height=height;                            }
   void              SetMovable(const bool flag)               { this.m_movable=flag;                             }
   void              SetSelectable(const bool flag)            { this.m_selectable=flag;                          }

public:
//--- Return the values of class variables
   string            NameObj(void)                       const { return this.m_name_obj;                          }
   long              ChartID(void)                       const { return this.m_chart_id;                          }
   int               WindowNum(void)                     const { return this.m_wnd_num;                           }
   int               CoordX(void)                        const { return this.m_coord_x;                           }
   int               CoordY(void)                        const { return this.m_coord_y;                           }
   int               Width(void)                         const { return this.m_width;                             }
   int               Height(void)                        const { return this.m_height;                            }
   int               RightEdge(void)                     const { return this.m_coord_x+this.m_width;              }
   int               BottomEdge(void)                    const { return this.m_coord_y+this.m_height;             }
   bool              Movable(void)                       const { return this.m_movable;                           }
   bool              Selectable(void)                    const { return this.m_selectable;                        }

//--- The virtual method returning the object type
   virtual int       Type(void)                          const { return this.m_type;                              }

//--- Constructor/destructor
                     CGBaseObj();
                    ~CGBaseObj();
  };
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CGBaseObj::CGBaseObj() : m_chart_id(::ChartID()),
                         m_type(WRONG_VALUE),
                         m_wnd_num(0),
                         m_coord_x(0),
                         m_coord_y(0),
                         m_width(0),
                         m_height(0),
                         m_movable(false),
                         m_selectable(false)
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

The CObject base object class features the virtual Type() method returning the object type (for identifying objects by their types). The original method always returns zero:

```
   //--- method of identifying the object
   virtual int       Type(void)                                    const { return(0);      }
```

By re-defining the method in the descendants, we return the object type set in the **m\_type** variable.

Graphical object types are set in subsequent articles when creating object classes. In the meantime, the method will return -1 (this is the value that we set in the initialization list of the class constructor).

### Class of the form object of graphical elements

The form object is a basis for creating the remaining classes of the library graphical elements based on the CCanvas class. It is to be used as a "canvas" for drawing data necessary for various objects and arranging other elements, which will ultimately display the ready-made object.

For now, this will be a simple form featuring basic parameters and functionality (the ability to set the active area used for interacting with the cursor), as well as the ability to move it along the chart.

In \\MQL5\\Include\\DoEasy\\Objects\ **Graph\**, create the **Form.mqh** file of the CForm class.

The class should be a descendant of the base object of all library graphical objects. Therefore, the files of the base graphical objectand the mouse property object class should be included into it:

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
#include <Canvas\Canvas.mqh>
#include "GBaseObj.mqh"
#include "..\..\Services\MouseState.mqh"
//+------------------------------------------------------------------+
//| Class of the base object of the library graphical objects        |
//+------------------------------------------------------------------+
class CForm : public CGBaseObj
  {
  }
```

In the protected class section, declare the CCanvas standard library class, CPause and CMouseState library class objects, the variable for storing the mouse status values, the variable for storing mouse status bit flags and the variables for storing object properties:

```
//+------------------------------------------------------------------+
//| Class of the base object of the library graphical objects        |
//+------------------------------------------------------------------+
class CForm : public CGBaseObj
  {
protected:
   CCanvas              m_canvas;                              // CCanvas class object
   CPause               m_pause;                               // Pause class object
   CMouseState          m_mouse;                               // "Mouse status" class object
   ENUM_MOUSE_FORM_STATE m_mouse_state;                        // Mouse status relative to the form
   ushort               m_mouse_state_flags;                   // Mouse status flags

   int                  m_act_area_left;                       // Left border of the active area (offset from the left border inward)
   int                  m_act_area_right;                      // Right border of the active area (offset from the right border inward)
   int                  m_act_area_top;                        // Upper border of the active area (offset from the upper border inward)
   int                  m_act_area_bottom;                     // Lower border of the active area (offset from the lower border inward)
   uchar                m_opacity;                             // Opacity
   int                  m_shift_y;                             // Y coordinate shift for the subwindow

private:
```

In the private section of the class, declare auxiliary methods for the class operation:

```
private:
//--- Set and return the flags indicating the states of mouse buttons and Shift/Ctrl keys
   ENUM_MOUSE_BUTT_KEY_STATE MouseButtonKeyState(const int id,const long lparam,const double dparam,const string sparam)
                       {
                        return this.m_mouse.ButtKeyState(id,lparam,dparam,sparam);
                       }
//--- Return the cursor position relative to the (1) form and (2) active area
   bool              CursorInsideForm(const int x,const int y);
   bool              CursorInsideActiveArea(const int x,const int y);

public:
```

The MouseButtonKeyState() method returns the value returned by the same-name method from the mouse status class object. Two other methods are necessary for defining the mouse cursor position relative to the form and form active area. I will consider them a bit later.

The public section of the class features the methods for creating a form, installing it and returning its parameters:

```
public:
//--- Create a form
   bool              CreateForm(const long chart_id,
                                const int wnd_num,
                                const string name,
                                const int x,
                                const int y,
                                const int w,
                                const int h,
                                const color colour,
                                const uchar opacity,
                                const bool movable=true,
                                const bool selectable=true);

//--- Return the pointer to a canvas object
   CCanvas          *CanvasObj(void)                           { return &this.m_canvas;                           }
//--- Set (1) the form update frequency, (2) the movability flag and (3) selectability flag for interaction
   void              SetFrequency(const ulong value)           { this.m_pause.SetWaitingMSC(value);               }
   void              SetMovable(const bool flag)               { CGBaseObj::SetMovable(flag);                     }
   void              SetSelectable(const bool flag)            { CGBaseObj::SetSelectable(flag);                  }
//--- Update the form coordinates (shift the form)
   bool              Move(const int x,const int y,const bool redraw=false);

//--- Return the mouse status relative to the form
   ENUM_MOUSE_FORM_STATE MouseFormState(const int id,const long lparam,const double dparam,const string sparam);
//--- Return the flag of the clicked (1) left, (2) right, (3) middle, (4) first and (5) second additional mouse buttons
   bool              IsPressedButtonLeftOnly(void)             { return this.m_mouse.IsPressedButtonLeft();       }
   bool              IsPressedButtonRightOnly(void)            { return this.m_mouse.IsPressedButtonRight();      }
   bool              IsPressedButtonMiddleOnly(void)           { return this.m_mouse.IsPressedButtonMiddle();     }
   bool              IsPressedButtonX1Only(void)               { return this.m_mouse.IsPressedButtonX1();         }
   bool              IsPressedButtonX2Only(void)               { return this.m_mouse.IsPressedButtonX2();         }
//--- Return the flag of the pressed (1) Shift and (2) Ctrl key
   bool              IsPressedKeyShiftOnly(void)               { return this.m_mouse.IsPressedKeyShift();         }
   bool              IsPressedKeyCtrlOnly(void)                { return this.m_mouse.IsPressedKeyCtrl();          }

//--- Set the shift of the (1) left, (2) top, (3) right, (4) bottom edge of the active area relative to the form,
//--- (5) all shifts of the active area edges relative to the form and (6) the form opacity
   void              SetActiveAreaLeftShift(const int value)   { this.m_act_area_left=fabs(value);                }
   void              SetActiveAreaRightShift(const int value)  { this.m_act_area_right=fabs(value);               }
   void              SetActiveAreaTopShift(const int value)    { this.m_act_area_top=fabs(value);                 }
   void              SetActiveAreaBottomShift(const int value) { this.m_act_area_bottom=fabs(value);              }
   void              SetActiveAreaShift(const int left_shift,const int bottom_shift,const int right_shift,const int top_shift);
   void              SetOpacity(const uchar value)             { this.m_opacity=value;                            }
//--- Return the coordinate (1) of the left, (2) right, (3) top and (4) bottom edge of the form active area
   int               ActiveAreaLeft(void)                const { return this.CoordX()+this.m_act_area_left;       }
   int               ActiveAreaRight(void)               const { return this.RightEdge()-this.m_act_area_right;   }
   int               ActiveAreaTop(void)                 const { return this.CoordY()+this.m_act_area_top;        }
   int               ActiveAreaBottom(void)              const { return this.BottomEdge()-this.m_act_area_bottom; }
//--- Return (1) the form opacity, coordinate (2) of the right and (3) bottom form edge
   uchar             Opacity(void)                       const { return this.m_opacity;                           }
   int               RightEdge(void)                     const { return CGBaseObj::RightEdge();                   }
   int               BottomEdge(void)                    const { return CGBaseObj::BottomEdge();                  }

//--- Event handler
   void              OnChartEvent(const int id,const long& lparam,const double& dparam,const string& sparam);

//--- Constructors/Destructor
                     CForm(const long chart_id,
                           const int wnd_num,
                           const string name,
                           const int x,
                           const int y,
                           const int w,
                           const int h,
                           const color colour,
                           const uchar opacity,
                           const bool movable=true,
                           const bool selectable=true);
                     CForm(){;}
                    ~CForm();
  };
//+------------------------------------------------------------------+
```

Let's consider the class methods in detail.

**In the parametric constructor,** create a form object with the parameters passed to the constructor:

```
//+------------------------------------------------------------------+
//| Parametric constructor                                           |
//+------------------------------------------------------------------+
CForm::CForm(const long chart_id,
             const int wnd_num,
             const string name,
             const int x,
             const int y,
             const int w,
             const int h,
             const color colour,
             const uchar opacity,
             const bool movable=true,
             const bool selectable=true) : m_act_area_bottom(0),
                                           m_act_area_left(0),
                                           m_act_area_right(0),
                                           m_act_area_top(0),
                                           m_mouse_state(0),
                                           m_mouse_state_flags(0)

  {
   if(this.CreateForm(chart_id,wnd_num,name,x,y,w,h,colour,opacity,movable,selectable))
     {
      this.m_shift_y=(int)::ChartGetInteger(chart_id,CHART_WINDOW_YDISTANCE,wnd_num);
      this.SetWindowNum(wnd_num);
      this.m_pause.SetWaitingMSC(PAUSE_FOR_CANV_UPDATE);
      this.m_pause.SetTimeBegin();
      this.m_mouse.SetChartID(chart_id);
      this.m_mouse.SetWindowNum(wnd_num);
      this.m_mouse.ResetAll();
      this.m_mouse_state_flags=0;
      CGBaseObj::SetMovable(movable);
      CGBaseObj::SetSelectable(selectable);
      this.SetOpacity(opacity);
     }
  }
//+------------------------------------------------------------------+
```

Here we first initialize all variables in the constructor initialization list. Then call the form creation method. If the form is created successfully, set the parameters passed to the constructor.

**In the class destructor**, remove the created graphical object:

```
//+------------------------------------------------------------------+
//| Destructor                                                       |
//+------------------------------------------------------------------+
CForm::~CForm()
  {
   ::ObjectsDeleteAll(this.ChartID(),this.NameObj());
  }
//+------------------------------------------------------------------+
```

**The method creating the graphical form object:**

```
//+------------------------------------------------------------------+
//| Create the graphical form object                                 |
//+------------------------------------------------------------------+
bool CForm::CreateForm(const long chart_id,
                       const int wnd_num,
                       const string name,
                       const int x,
                       const int y,
                       const int w,
                       const int h,
                       const color colour,
                       const uchar opacity,
                       const bool movable=true,
                       const bool selectable=true)
  {
   if(this.m_canvas.CreateBitmapLabel(chart_id,wnd_num,name,x,y,w,h,COLOR_FORMAT_ARGB_NORMALIZE))
     {
      this.SetChartID(chart_id);
      this.SetWindowNum(wnd_num);
      this.SetNameObj(name);
      this.SetCoordX(x);
      this.SetCoordY(y);
      this.SetWidth(w);
      this.SetHeight(h);
      this.SetActiveAreaLeftShift(1);
      this.SetActiveAreaRightShift(1);
      this.SetActiveAreaTopShift(1);
      this.SetActiveAreaBottomShift(1);
      this.SetOpacity(opacity);
      this.SetMovable(movable);
      this.SetSelectable(selectable);
      this.m_canvas.Erase(::ColorToARGB(colour,this.Opacity()));
      this.m_canvas.Update();
      return true;
     }
   return false;
  }
//+------------------------------------------------------------------+
```

Use the [CreateBitmapLabel() method of the CCanvas class](https://www.mql5.com/en/docs/standardlibrary/canvasgraphics/ccanvas/ccanvascreatebitmaplabel) to create a graphical resource using the chart ID and subwindow index (the second method calling form). If the graphical resource has been created successfully, set all parameters passed to the method, fill the form with color, set the opacity using the [Erase()](https://www.mql5.com/en/docs/standardlibrary/canvasgraphics/ccanvas/ccanvaserase) method and display changes on the screen using the [Update()](https://www.mql5.com/en/docs/standardlibrary/canvasgraphics/ccanvas/ccanvasupdate) method.

I would like to clarify the term "opacity" or color density. The CCanvas class allows setting transparency for objects. 0 means a completely transparent color, while 255 stands for a completely opaque color. So, everything seems to be inverted here. Therefore, I decided to use the term "opacity" since the values of 0 — 255 correspond exactly to an increase in color density from zero (completely transparent) to 255 (completely opaque).

**CForm class event handler:**

```
//+------------------------------------------------------------------+
//| Event handler                                                    |
//+------------------------------------------------------------------+
void CForm::OnChartEvent(const int id,const long &lparam,const double &dparam,const string &sparam)
  {
//--- Get the status of mouse buttons, Shift/Ctrl keys and the state of a mouse relative to the form
   ENUM_MOUSE_BUTT_KEY_STATE mouse_state=this.m_mouse.ButtKeyState(id,lparam,dparam,sparam);
   this.m_mouse_state=this.MouseFormState(id,lparam,dparam-this.m_shift_y,sparam);
//--- Initialize the difference between X and Y coordinates of the form and cursor
   static int diff_x=0;
   static int diff_y=0;
//--- In case of a chart change event, recalculate the shift by Y for the subwindow
   if(id==CHARTEVENT_CHART_CHANGE)
     {
      this.m_shift_y=(int)::ChartGetInteger(this.ChartID(),CHART_WINDOW_YDISTANCE,this.WindowNum());
     }
//--- If the cursor is inside the form, disable chart scrolling, context menu and Crosshair tool
   if((this.m_mouse_state_flags & 0x0100)!=0)
     {
      ::ChartSetInteger(this.ChartID(),CHART_MOUSE_SCROLL,false);
      ::ChartSetInteger(this.ChartID(),CHART_CONTEXT_MENU,false);
      ::ChartSetInteger(this.ChartID(),CHART_CROSSHAIR_TOOL,false);
     }
//--- Otherwise, if the cursor is outside the form, allow chart scrolling, context menu and Crosshair tool
   else
     {
      ::ChartSetInteger(this.ChartID(),CHART_MOUSE_SCROLL,true);
      ::ChartSetInteger(this.ChartID(),CHART_CONTEXT_MENU,true);
      ::ChartSetInteger(this.ChartID(),CHART_CROSSHAIR_TOOL,true);
     }
//--- If the mouse movement event and the cursor are located in the form active area
   if(id==CHARTEVENT_MOUSE_MOVE && m_mouse_state==MOUSE_FORM_STATE_INSIDE_ACTIVE_AREA_PRESSED)
     {
      //--- If only the left mouse button is being held and the form is moved,
      //--- set the new parameters of moving the form relative to the cursor
      if(IsPressedButtonLeftOnly() && this.Move(this.m_mouse.CoordX()-diff_x,this.m_mouse.CoordY()-diff_y))
        {
         diff_x=this.m_mouse.CoordX()-this.CoordX();
         diff_y=this.m_mouse.CoordY()-this.CoordY();
        }
     }
//--- In any other cases, set the parameters of shifting the form relative to the cursor
   else
     {
      diff_x=this.m_mouse.CoordX()-this.CoordX();
      diff_y=this.m_mouse.CoordY()-this.CoordY();
     }
//--- Test display of mouse states on the chart
   Comment(EnumToString(mouse_state),"\n",EnumToString(this.m_mouse_state));
  }
//+------------------------------------------------------------------+
```

The entire logic is clarified in the comments. The method should be called [from the OnChartEvent() standard handler](https://www.mql5.com/en/docs/event_handlers/onchartevent) of the program and it has exactly the same parameters.

Let me explain the dedicated calculation passed to the MouseFormState() method. If the form is located in the main chart window, the **m\_shift\_y** variable is equal to zero and the expression dparam-this.m\_shift\_y returns the accurate Y cursor coordinate. But if the form is located in the chart subwindow, the shift in the **m\_shift\_y** variable exceeds zero to adjust the Y cursor coordinate to the subwindow coordinates. Accordingly, we also need to pass the Y coordinate with the shift set in **m\_shift\_y to the methods of calculating the cursor coordinates.** Otherwise, the object coordinates will point higher than it actually is by the number of pixels of the shift specified in the variable.

**The method returning the cursor position relative to the form:**

```
//+------------------------------------------------------------------+
//| Return the cursor position relative to the form                  |
//+------------------------------------------------------------------+
bool CForm::CursorInsideForm(const int x,const int y)
  {
   return(x>=this.CoordX() && x<this.RightEdge() && y>=this.CoordY() && y<=this.BottomEdge());
  }
//+------------------------------------------------------------------+
```

The method receives the cursor X and Y coordinates.

If

- (the cursor X coordinate exceeds or is equal to the X coordinate of the form, and the cursor X coordinate is less or equal to the coordinate of the form right edge) and
- (the cursor Y coordinate exceeds or is equal to the Y coordinate of the form, and the cursor Y coordinate is less or equal to the coordinate of the form bottom edge),

true is returned — the cursor is located inside the form object.

**The method returning the cursor position relative to the form active area:**

```
//+------------------------------------------------------------------+
//| Return the cursor position relative to the form active area      |
//+------------------------------------------------------------------+
bool CForm::CursorInsideActiveArea(const int x,const int y)
  {
   return(x>=this.ActiveAreaLeft() && x<this.ActiveAreaRight() && y>=this.ActiveAreaTop() && y<=this.ActiveAreaBottom());
  }
//+------------------------------------------------------------------+
```

The method receives the cursor X and Y coordinates.

If

- (the cursor X coordinate exceeds or is equal to the X coordinate of the form active area, and the cursor X coordinate is less or equal to the coordinate of the form active area right edge) and
- (the cursor Y coordinate exceeds or is equal to the Y coordinate of the form active area, and the cursor Y coordinate is less or equal to the coordinate of the form active area bottom edge),

true is returned — the cursor is located inside the form object active area.

**The method returning the mouse status relative to the form:**

```
//+------------------------------------------------------------------+
//| Return the mouse status relative to the form                     |
//+------------------------------------------------------------------+
ENUM_MOUSE_FORM_STATE CForm::MouseFormState(const int id,const long lparam,const double dparam,const string sparam)
  {
//--- Get the mouse status relative to the form, as well as the states of mouse buttons and Shift/Ctrl keys
   ENUM_MOUSE_FORM_STATE form_state=MOUSE_FORM_STATE_NONE;
   ENUM_MOUSE_BUTT_KEY_STATE state=this.MouseButtonKeyState(id,lparam,dparam,sparam);
//--- Get the mouse status flags from the CMouseState class object and save them in the variable
   this.m_mouse_state_flags=this.m_mouse.GetMouseFlags();
//--- If the cursor is inside the form
   if(this.CursorInsideForm(m_mouse.CoordX(),m_mouse.CoordY()))
     {
      //--- Set bit 8 responsible for the "cursor inside the form" flag
      this.m_mouse_state_flags |= (0x0001<<8);
      //--- If the cursor is inside the active area, set bit 9 "cursor inside the active area"
      if(CursorInsideActiveArea(m_mouse.CoordX(),m_mouse.CoordY()))
         this.m_mouse_state_flags |= (0x0001<<9);
      //--- otherwise, release the bit "cursor inside the active area"
      else this.m_mouse_state_flags &=0xFDFF;
      //--- If one of the mouse buttons is clicked, check the cursor location in the active area and
      //--- return the appropriate value of the pressed key (in the active area or the form area)
      if((this.m_mouse_state_flags & 0x0001)!=0 || (this.m_mouse_state_flags & 0x0002)!=0 || (this.m_mouse_state_flags & 0x0010)!=0)
         form_state=((m_mouse_state_flags & 0x0200)!=0 ? MOUSE_FORM_STATE_INSIDE_ACTIVE_AREA_PRESSED : MOUSE_FORM_STATE_INSIDE_PRESSED);
      //--- otherwise, check the cursor location in the active area and
      //--- return the appropriate value of the unpressed key (in the active area or the form area)
      else
         form_state=((m_mouse_state_flags & 0x0200)!=0 ? MOUSE_FORM_STATE_INSIDE_ACTIVE_AREA_NOT_PRESSED : MOUSE_FORM_STATE_INSIDE_NOT_PRESSED);
     }
   return form_state;
  }
//+------------------------------------------------------------------+
```

Each code string is clarified in the code comments. In brief, we get a ready-made mouse status from the mouse status class object and write it to the **m\_mouse\_state\_flags** variable. Next, depending on the cursor location relative to the form, supplement the mouse status bit flags with new data and return the mouse status in the ENUM\_MOUSE\_FORM\_STATE enumeration format we considered above at the beginning of the article.

**The method updating the form coordinates (shifting the form on the chart):**

```
//+------------------------------------------------------------------+
//| Update the form coordinates                                      |
//+------------------------------------------------------------------+
bool CForm::Move(const int x,const int y,const bool redraw=false)
  {
//--- If the form is not movable, leave
   if(!this.Movable())
      return false;
//--- If new values are successfully set into graphical object properties
   if(::ObjectSetInteger(this.ChartID(),this.NameObj(),OBJPROP_XDISTANCE,x) &&
      ::ObjectSetInteger(this.ChartID(),this.NameObj(),OBJPROP_YDISTANCE,y))
     {
      //--- set the new values of X and Y coordinate properties
      this.SetCoordX(x);
      this.SetCoordY(y);
      //--- If the update flag is activated, redraw the chart.
      if(redraw)
         ::ChartRedraw(this.ChartID());
      //--- Return 'true'
      return true;
     }
//--- Something is wrong...
   return false;
  }
//+------------------------------------------------------------------+
```

The method receives the coordinates you want to shift the form object to. If the new coordinate parameters are successfully set to the graphical form object, write these coordinates to the object properties and redraw the chart only if the redraw flag (which is passed to the method as well) is activated. Redrawing by the flag value is necessary to avoid multiple chart redrawing in case the graphical object consists of many forms. In this case, we need to first move all the forms of one object. After each form receives new coordinates, update the chart once.

**The method setting all shifts of the active area relative to the form:**

```
//+------------------------------------------------------------------+
//| Set all shifts of the active area relative to the form           |
//+------------------------------------------------------------------+
void CForm::SetActiveAreaShift(const int left_shift,const int bottom_shift,const int right_shift,const int top_shift)
  {
   this.SetActiveAreaLeftShift(left_shift);
   this.SetActiveAreaBottomShift(bottom_shift);
   this.SetActiveAreaRightShift(right_shift);
   this.SetActiveAreaTopShift(top_shift);
  }
//+------------------------------------------------------------------+
```

We have the methods for setting active area borders separately. But sometimes it is required to set all borders within one call of a single method. This is exactly what the metod does — it sets new values of the active area border offset from the form edge using the calls of the appropriate methods.

This completes the creation of the first version of the form object. Let's test the results.

### Test

To perform the test, let's create a single form object on the chart and try moving it using the cursor. Besides, I am going to display the states of mouse buttons and Ctrl/Shift keys, as well as the status of the cursor relative to the active area form and borders.

In \\MQL5\\Experts\\TestDoEasy\ **Part73\**, create the new EA file **TestDoEasyPart73.mq5**.

When creating the EA file, specify that we need the InpMovable input of the bool type and the initial true value:

![](https://c.mql5.com/2/42/MetaEditor64_A9qDnmazM8__1.png)

Next, specify that we need the additional OnChartEvent() handler:

![](https://c.mql5.com/2/42/WKjsveaw9o__1.png)

As a result, we obtain the following EA work piece:

```
//+------------------------------------------------------------------+
//|                                             TestDoEasyPart73.mq5 |
//|                                  Copyright 2021, MetaQuotes Ltd. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2021, MetaQuotes Ltd."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//--- input parameters
input bool     InpMovable=true;
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---

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
//| ChartEvent function                                              |
//+------------------------------------------------------------------+
void OnChartEvent(const int id,
                  const long &lparam,
                  const double &dparam,
                  const string &sparam)
  {
//---

  }
//+------------------------------------------------------------------+
```

Include the newly created form object class to the EA file and declare the two global variables — object name prefix and CForm class object:

```
//+------------------------------------------------------------------+
//|                                             TestDoEasyPart73.mq5 |
//|                                  Copyright 2021, MetaQuotes Ltd. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2021, MetaQuotes Ltd."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//--- includes
#include <DoEasy\Objects\Graph\Form.mqh>
//--- input parameters
sinput   bool  InpMovable  = true;  // Movable flag
//--- global variables
string         prefix;
CForm          form;
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
```

In the OnInit() handler, enable the permission to send mouse cursor movement and mouse scroll events, set the value for object name prefixes as (file name)+"\_" and create the form object on the chart. After creating it, set the offset of 10 pixels for the boundaries of the active zone:

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
   prefix=MQLInfoString(MQL_PROGRAM_NAME)+"_";
//--- If the form is created, set an active area for it with the offset of 10 pixels from the edges
   if(form.CreateForm(ChartID(),0,prefix+"Form_01",300,20,100,70,clrSilver,200,InpMovable))
     {
      form.SetActiveAreaShift(10,10,10,10);
     }
//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
```

Now it remains to call the OnChartEvent() handler of the form object from the OnChartEvent() handler of the EA:

```
//+------------------------------------------------------------------+
//| ChartEvent function                                              |
//+------------------------------------------------------------------+
void OnChartEvent(const int id,
                  const long &lparam,
                  const double &dparam,
                  const string &sparam)
  {
//---
   form.OnChartEvent(id,lparam,dparam,sparam);
  }
//+------------------------------------------------------------------+
```

Compile the EA and launch it on a symbol chart:

![](https://c.mql5.com/2/42/GJ8LiuCp71.gif)

As we can see, the status of the buttons and the cursor is displayed correctly. The form object moves only when grabbed by the mouse within its active area.

When clicking the right and middle mouse buttons within the form, the context menu and the Crosshair tool are not activated. Here we face a funny glitch: if we enable the Crosshair tool outside the window and then hover with it (with the left mouse button pressed) over the active area of the form, it starts to shift. This is an incorrect behavior. But this is only the beginning. I will make improvements and add the new functionality to the form object in subsequent articles.

### What's next?

In the next article, I will continue the development of the form object class.

All files of the current version of the library are attached below together with the test EA file for MQL5 for you to test and download.

Leave your questions and suggestions in the comments.

[Back to contents](https://www.mql5.com/en/articles/9442#node00)

**\*The final article of the last series:**

[Other classes in DoEasy library (Part 72): Tracking and recording chart object parameters in the collection](https://www.mql5.com/en/articles/9385/)

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/9442](https://www.mql5.com/ru/articles/9442)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/9442.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/9442/mql5.zip "Download MQL5.zip")(3954.84 KB)

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

**[Go to discussion](https://www.mql5.com/en/forum/371873)**

![Graphics in DoEasy library (Part 74): Basic graphical element powered by the CCanvas class](https://c.mql5.com/2/42/MQL5-avatar-doeasy-library3-2__1.png)[Graphics in DoEasy library (Part 74): Basic graphical element powered by the CCanvas class](https://www.mql5.com/en/articles/9493)

In this article, I will rework the concept of building graphical objects from the previous article and prepare the base class of all graphical objects of the library powered by the Standard Library CCanvas class.

![Cluster analysis (Part I): Mastering the slope of indicator lines](https://c.mql5.com/2/42/mql5-avatar-cluster_analysis.png)[Cluster analysis (Part I): Mastering the slope of indicator lines](https://www.mql5.com/en/articles/9527)

Cluster analysis is one of the most important elements of artificial intelligence. In this article, I attempt applying the cluster analysis of the indicator slope to get threshold values for determining whether a market is flat or following a trend.

![Graphics in DoEasy library (Part 75): Methods of handling primitives and text in the basic graphical element](https://c.mql5.com/2/42/MQL5-avatar-doeasy-library3-2__2.png)[Graphics in DoEasy library (Part 75): Methods of handling primitives and text in the basic graphical element](https://www.mql5.com/en/articles/9515)

In this article, I will continue the development of the basic graphical element class of all library graphical objects powered by the CCanvas Standard Library class. I will create the methods for drawing graphical primitives and for displaying a text on a graphical element object.

![Other classes in DoEasy library (Part 72): Tracking and recording chart object parameters in the collection](https://c.mql5.com/2/42/MQL5-avatar-doeasy-library__10.png)[Other classes in DoEasy library (Part 72): Tracking and recording chart object parameters in the collection](https://www.mql5.com/en/articles/9385)

In this article, I will complete working with chart object classes and their collection. I will also implement auto tracking of changes in chart properties and their windows, as well as saving new parameters to the object properties. Such a revision allows the future implementation of an event functionality for the entire chart collection.

[![](https://www.mql5.com/ff/sh/zf7a2k61x98jzh89z2/01.png)Speed up your tradingUse our high-speed VPS for MetaTrader 4 and 5Learn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=qtrrsuiwuicrscmckjynyanztbditglq&s=c617dc80d90cfd3783ec1345eec2b419b281f10fec6eac77b3218984ac337259&uid=&ref=https://www.mql5.com/en/articles/9442&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5083399132980189951)

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