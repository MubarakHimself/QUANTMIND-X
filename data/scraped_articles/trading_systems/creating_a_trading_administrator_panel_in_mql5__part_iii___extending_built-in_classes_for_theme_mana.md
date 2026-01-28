---
title: Creating a Trading Administrator Panel in MQL5 (Part III): Extending Built-in Classes for Theme Management (II)
url: https://www.mql5.com/en/articles/16045
categories: Trading Systems, Integration
relevance_score: 6
scraped_at: 2026-01-23T11:39:11.745404
---

[![](https://www.mql5.com/ff/sh/6zw0dkux8bqt7m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
MQL5 Channels - Messenger for traders\\
\\
Install the app and receive market analytics and trading tips.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=iuciwacmrxvmiibwyujliagqikizpsoo&s=268cbb13914c54b6c5c875db99b154944f6e0122b3400b54c9ac0d4f69f0f0d6&uid=&ref=https://www.mql5.com/en/articles/16045&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5062620059727209901)

MetaTrader 5 / Examples


### Contents:

- [Introduction](https://www.mql5.com/en/articles/16045#para1)
- [Understanding Classes in MQL5](https://www.mql5.com/en/articles/16045#para2)
- [Adding Theme Management Methods to (CDialog, CEdit, and CButton)](https://www.mql5.com/en/articles/16045#para3)
- [CButton Class Theme Management](https://www.mql5.com/en/articles/16045#para4)
- [CEdit Class Theme Management](https://www.mql5.com/en/articles/16045#para5)
- [Adjusting the Admin Panel for Theme Switching](https://www.mql5.com/en/articles/16045#para6)
- [Final Code and Results](https://www.mql5.com/en/articles/16045#para7)
- [Conclusion](https://www.mql5.com/en/articles/16045#para8)

### Introduction

It is possible to modify and create new library classes for MQL5. However, since the built-in libraries are shared by the platform, any changes we make to these files may lead to either positive enhancements or negative impacts on the current platform features. In our [recent article](https://www.mql5.com/en/articles/15419#para7), we briefly discussed how we edited the root [Dialog class color](https://www.mql5.com/en/articles/15419#para7) to affect the appearance of our panel. While our theme-switching button successfully changed the text color, it did not alter the panel skin or the button background color.

Through research, we have finally identified methods to safely integrate theme-changing functionalities into the available classes. After successfully implementing these changes, we adjusted the Admin Panel algorithm to align with the newly integrated features.

![New Panel Theme](https://c.mql5.com/2/96/Panel_Switching__1.png)

Theme switching successful

Today's discussion focuses on the process we undertook to achieve the visually appealing panel displayed on the right. The theme colors shown are based on my opinion on color selection during the development; they can be optimized in the code to suit other user preferences, allowing you to experiment with different colors to find what resonates with you. It's important to highlight the key components of our program that contribute to the overall functionality of the panel.

I will list them down:

- Text color
- Button skin color
- Borders
- Background color

Essentially, those are the most visible features of our program. When we trigger a theme change, each component must respond by altering its display properties to showcase the desired colors as defined in the code. By the end of this discussion, we aim to empower you with the skills needed to modify and extend the available classes when working with interfaces, as demonstrated in this project.

### Understanding Classes in MQL5.

To ensure that both experts and novices can follow along, I would like to start by familiarizing everyone with the concept of classes as employed in MQL5. Below are the definitions and key concepts that will help us understand how classes function within this programming environment.

**Classes:**

Classes are the foundation of Object-Oriented Programming (OOP) in MQL5, allowing developers to group related variables (attributes) and functions (methods) into a single unit to represent complex concepts and behaviors in a program.

Breaking down a class into two:

1. Attributes: Variables that store the state or data for objects of the class.
2. Methods: Functions that define the behavior or actions for objects of the class

**Outline of the main features of a Class:**

- Encapsulation in a class involves bundling data (variables) and methods (functions) that operate on that data, ensuring it is protected from external access and misuse.
- Inheritance allows a class to inherit properties and methods from another class, promoting code reuse and creating a hierarchical structure.
- Polymorphism enables method overriding, allowing subclasses to provide specific implementations for methods already defined in their parent classes.
- Abstraction simplifies the modeling of complex systems by focusing only on the relevant data and methods, hiding unnecessary details from the user.

To access the MetaQuotes header files that contain the GUI classes useful for our project, refer to the following image, which illustrates how we can locate these files.

![Locating MQ Header Files](https://c.mql5.com/2/96/ShareX_E5yHSvpAQ2.gif)

Locating the MQL5 header files

I have used a typical MQL5 class source extract to help us clearly understand the classes and their structures at a practical angle. See the code snippet below, and I have explained its construction just below it in tabular form.

```
//Basic parts of a class.

class CDialog : public CWndContainer
{
public:
   // Constructor and Destructor (Methods)
   CDialog(void);   // Constructor
   ~CDialog(void);  // Destructor

   // Public Methods (Functions)
   virtual bool Create(const long chart, const string name, const int subwin, const int x1, const int y1, const int x2, const int y2);
   virtual bool OnEvent(const int id, const long &lparam, const double &dparam, const string &sparam);
   string Caption(void) const;
   bool Caption(const string text);
   bool Add(CWnd *control);   // Add control by pointer
   bool Add(CWnd &control);   // Add control by reference
   virtual bool Save(const int file_handle);
   virtual bool Load(const int file_handle);
   void UpdateThemeColors(bool darkTheme);

protected:
   // Attributes (Variables)
   bool m_panel_flag;        // Panel visibility flag
   bool m_minimized;         // Minimized state flag
   CWnd m_caption;           // Caption control
   CWnd m_client_area;       // Client area control
   CRect m_norm_rect;        // Normal (non-minimized) rectangle
   CRect m_min_rect;         // Minimized rectangle
   CWnd m_white_border;      // White border control

   // Protected Methods (Internal functions)
   virtual bool CreateWhiteBorder(void);
   virtual bool CreateBackground(void);
   virtual bool CreateCaption(void);
   virtual bool CreateButtonClose(void);
   virtual bool CreateClientArea(void);
   virtual void OnClickCaption(void);
   virtual void OnClickButtonClose(void);
   virtual bool OnDialogDragStart(void);
   virtual bool OnDialogDragProcess(void);
   virtual bool OnDialogDragEnd(void);
};
```

This table is a summary of the attributes available in the above code snippets and their description.

| Attributes (Properties) | Description |
| --- | --- |
| ```<br>bool m_panel_flag;<br>``` | Flag to indicate if the panel is visible. |
| ```<br>bool m_minimized;<br>``` | Flag to indicate if the dialog is minimized |
| ```<br>CWnd m_caption;<br>``` | Control for the caption text. |
| ```<br>CWnd m_client_area;<br>``` | Control for the client area where other elements reside. |
| ```<br>CRect m_norm_rect;<br>``` | Coordinates for the normal (non-minimized) state. |
| ```<br>CRect m_min_rect;<br>``` | Coordinates for the minimized state. |
| ```<br>CWnd m_white_border;<br>``` | Control for the white border around the dialog. |

This table summarizes the methods used in the example class code.

| Methods | Description |
| --- | --- |
| ```<br>CDialog(void)<br>``` | Constructor that initializes the dialog. |
| ```<br>~CDialog(void)<br>``` | Destructor to clean up resources. |
| ```<br>Create(...)<br>``` | Creates the dialog window and its controls. |
| ```<br>OnEvent(...)<br>``` | Handles chart events for the dialog. |
| ```<br>Caption(void)<br>``` | Returns the current caption text. |
| ```<br>Caption(const string text)<br>``` | Sets the caption text. |
| ```<br>Add(CWnd *control)<br>``` | Adds a control to the client area by pointer. |
| ```<br>Add(CWnd &control)<br>``` | Adds a control to the client area by reference. |
| ```<br>Save(const int file_handle)<br>``` | Saves the dialog state to a file. |
| ```<br>Load(const int file_handle)<br>``` | Loads the dialog state from a file. |
| ```<br>UpdateThemeColors(bool darkTheme)<br>``` | Updates the theme colors (dark or light). |
| ```<br>CreateWhiteBorder(void)<br>``` | Creates the white border for the dialog. |
| ```<br>CreateBackground(void)<br>``` | Creates the background of the dialog. |
| ```<br>CreateCaption(void)<br>``` | Creates the caption area. |
| ```<br>CreateButtonClose(void)<br>``` | Creates the close button. |
| ```<br>CreateClientArea(void)<br>``` | Creates the client area. |
| ```<br>OnClickCaption(void)<br>``` | Handles the caption click event. |
| ```<br>OnClickButtonClose(void)<br>``` | Handles the close button click event. |
| ```<br>OnDialogDragStart(void)<br>``` | Handles the start of a dialog drag event. |
| ```<br>OnDialogDragProcess(void)<br>``` | Handles the drag process of the dialog. |
| ```<br>OnDialogDragEnd(void)<br>``` | Handles the end of a dialog drag event. |

Let's briefly look at one of the prominent classes that we are using in our program below.

### Adding Theme Management Methods to (CDialog, CEdit, and CButton)

Now, I believe we have a clearer understanding of the methods we need to implement to achieve our goal of theme switching. The Dialog library already contains the essential features required, and our next step will be to incorporate the necessary methods.

**[CDialog](https://www.mql5.com/en/docs/standardlibrary/controls/cdialog?utm_campaign=search&utm_medium=special&utm_source=mt5editor) Theme Management Methods:**

**CDialog:**

The CDialog class in MQL5 is responsible for creating and managing custom graphical dialog windows or panels within the MetaTrader 5 platform. It allows developers to construct dialog boxes containing UI components such as captions, client areas, borders, and close buttons. The class handles user interactions like clicking and dragging the dialog, as well as dynamically updating its theme (e.g., switching between dark and light modes). Additionally, it provides methods to save and load the dialog’s state, ensuring its size, position, and minimization status are preserved. Controls such as buttons and text fields can be added to the dialog, making it a versatile tool for building interactive and visually appealing interfaces in trading applications.

In the CDialog class, we introduced a method for handling dynamic theme updates. This method is responsible for updating the dialog's visual appearance based on whether the darkTheme is active or not. Here’s how the method is incorporated and how it ties in with the other components of the CDialog class. I will explain in two steps. However, you may consider skipping the first step if you do not intend to define colors.

S **tep 1: Define Theme Colors**

It is necessary to define the colors so that the program knows the alternatives when a theme change is called. In this implementation, our method uses specific color definitions for both dark and light themes. These could be predefined constants or passed through parameters.

```
// Theme colors that can be defined elsewhere in our program
const color DARK_THEME_BG = clrBlack;
const color DARK_THEME_BORDER = clrGray;
const color LIGHT_THEME_BG = clrWhite;
const color LIGHT_THEME_BORDER = clrSilver;
```

**Step 2: The Update Theme Colors method**

This function checks whether the darkTheme is active (true or false) and applies the respective colors to the key components: the white border (m\_white\_border) is updated with both background and border colors; the background (m\_background) adjusts its background and border colors; the caption (m\_caption) changes the text and background colors of the title bar; and the client area (m\_client\_area) applies color changes to the client area. Finally, the function calls Redraw() to ensure that the new theme is visually applied without recreating objects. If you had skipped to step two, then the highlighted color definitions will not work and color need to be put as e.g. ClrBlack or ClrBlue etc

```
//+------------------------------------------------------------------+
//| Method for dynamic theme updates                                 |
//+------------------------------------------------------------------+

void CDialog::UpdateThemeColors(bool darkTheme)
{
   color backgroundColor = darkTheme ? DARK_THEME_BG : LIGHT_THEME_BG;
   color borderColor = darkTheme ? DARK_THEME_BORDER : LIGHT_THEME_BORDER;

   // Update White Border colors
   m_white_border.ColorBackground(backgroundColor);
   m_white_border.ColorBorder(borderColor);

   // Update Background colors
   m_background.ColorBackground(backgroundColor);
   m_background.ColorBorder(borderColor);

   // Update Caption colors (optional for text-based themes)
   m_caption.Color(darkTheme ? clrWhite : clrBlack);
   m_caption.ColorBackground(backgroundColor);

   // Update Client Area colors
   m_client_area.ColorBackground(backgroundColor);
   m_client_area.ColorBorder(borderColor);

   // Redraw the controls to reflect the theme changes
   Redraw();
}
```

### [CButton](https://www.mql5.com/en/docs/standardlibrary/controls/cbutton?utm_campaign=search&utm_medium=special&utm_source=mt5editor) Class Theme Management

Using the same terms as above, we added methods SetTextColor, SetBackgroundColor, and SetBorderColor to the CButton class. These methods allow us to set the button's text, background, and border colors, respectively. Here's the code snippet showing the implementation of methods.

```
 //--- theme methods
   void              SetTextColor(color clr)       { m_button.Color(clr);                           }
   void              SetBackgroundColor(color clr) { m_button.BackColor(clr);                       }
   void              SetBorderColor(color clr)     { m_button.BorderColor(clr);                     }
```

**CButton Default Program from MQL5**

```
//+------------------------------------------------------------------+
//|                                                       Button.mqh |
//|                             Copyright 2000-2024, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#include "WndObj.mqh"
#include <ChartObjects\ChartObjectsTxtControls.mqh>
//+------------------------------------------------------------------+
//| Class CButton                                                    |
//| Usage: control that is displayed by                              |
//|             the CChartObjectButton object                        |
//+------------------------------------------------------------------+
class CButton : public CWndObj
  {
private:
   CChartObjectButton m_button;             // chart object

public:
                     CButton(void);
                    ~CButton(void);
   //--- create
   virtual bool      Create(const long chart,const string name,const int subwin,const int x1,const int y1,const int x2,const int y2);
   //--- state
   bool              Pressed(void)          const { return(m_button.State());                       }
   bool              Pressed(const bool pressed)  { return(m_button.State(pressed));                }
   //--- properties
   bool              Locking(void)          const { return(IS_CAN_LOCK);                            }
   void              Locking(const bool flag);

protected:
   //--- handlers of object settings
   virtual bool      OnSetText(void)              { return(m_button.Description(m_text));           }
   virtual bool      OnSetColor(void)             { return(m_button.Color(m_color));                }
   virtual bool      OnSetColorBackground(void)   { return(m_button.BackColor(m_color_background)); }
   virtual bool      OnSetColorBorder(void)       { return(m_button.BorderColor(m_color_border));   }
   virtual bool      OnSetFont(void)              { return(m_button.Font(m_font));                  }
   virtual bool      OnSetFontSize(void)          { return(m_button.FontSize(m_font_size));         }
   //--- internal event handlers
   virtual bool      OnCreate(void);
   virtual bool      OnShow(void);
   virtual bool      OnHide(void);
   virtual bool      OnMove(void);
   virtual bool      OnResize(void);
   //--- íîâûå îáðàáîò÷èêè
   virtual bool      OnMouseDown(void);
   virtual bool      OnMouseUp(void);
  };
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CButton::CButton(void)
  {
   m_color           =CONTROLS_BUTTON_COLOR;
   m_color_background=CONTROLS_BUTTON_COLOR_BG;
   m_color_border    =CONTROLS_BUTTON_COLOR_BORDER;
  }
//+------------------------------------------------------------------+
//| Destructor                                                       |
//+------------------------------------------------------------------+
CButton::~CButton(void)
  {
  }
//+------------------------------------------------------------------+
//| Create a control                                                 |
//+------------------------------------------------------------------+
bool CButton::Create(const long chart,const string name,const int subwin,const int x1,const int y1,const int x2,const int y2)
  {
//--- call method of the parent class
   if(!CWndObj::Create(chart,name,subwin,x1,y1,x2,y2))
      return(false);
//--- create the chart object
   if(!m_button.Create(chart,name,subwin,x1,y1,Width(),Height()))
      return(false);
//--- call the settings handler
   return(OnChange());
  }
//+------------------------------------------------------------------+
//| Locking flag                                                     |
//+------------------------------------------------------------------+
void CButton::Locking(const bool flag)
  {
   if(flag)
      PropFlagsSet(WND_PROP_FLAG_CAN_LOCK);
   else
      PropFlagsReset(WND_PROP_FLAG_CAN_LOCK);
  }
//+------------------------------------------------------------------+
//| Create object on chart                                           |
//+------------------------------------------------------------------+
bool CButton::OnCreate(void)
  {
//--- create the chart object by previously set parameters
   return(m_button.Create(m_chart_id,m_name,m_subwin,m_rect.left,m_rect.top,m_rect.Width(),m_rect.Height()));
  }
//+------------------------------------------------------------------+
//| Display object on chart                                          |
//+------------------------------------------------------------------+
bool CButton::OnShow(void)
  {
   return(m_button.Timeframes(OBJ_ALL_PERIODS));
  }
//+------------------------------------------------------------------+
//| Hide object from chart                                           |
//+------------------------------------------------------------------+
bool CButton::OnHide(void)
  {
   return(m_button.Timeframes(OBJ_NO_PERIODS));
  }
//+------------------------------------------------------------------+
//| Absolute movement of the chart object                            |
//+------------------------------------------------------------------+
bool CButton::OnMove(void)
  {
//--- position the chart object
   return(m_button.X_Distance(m_rect.left) && m_button.Y_Distance(m_rect.top));
  }
//+------------------------------------------------------------------+
//| Resize the chart object                                          |
//+------------------------------------------------------------------+
bool CButton::OnResize(void)
  {
//--- resize the chart object
   return(m_button.X_Size(m_rect.Width()) && m_button.Y_Size(m_rect.Height()));
  }
//+------------------------------------------------------------------+
//| Handler of click on the left mouse button                        |
//+------------------------------------------------------------------+
bool CButton::OnMouseDown(void)
  {
   if(!IS_CAN_LOCK)
      Pressed(!Pressed());
//--- call of the method of the parent class
   return(CWnd::OnMouseDown());
  }
//+------------------------------------------------------------------+
//| Handler of click on the left mouse button                        |
//+------------------------------------------------------------------+
bool CButton::OnMouseUp(void)
  {
//--- depress the button if it is not fixed
   if(m_button.State() && !IS_CAN_LOCK)
      m_button.State(false);
//--- call of the method of the parent class
   return(CWnd::OnMouseUp());
  }
//+------------------------------------------------------------------+
```

**CButton With Theme Management Method Incorporated:**

See the highlighted section.

```
//+------------------------------------------------------------------+
//|                                                       Button.mqh |
//|                             Copyright 2000-2024, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#include "WndObj.mqh"
#include <ChartObjects\ChartObjectsTxtControls.mqh>
//+------------------------------------------------------------------+
//| Class CButton                                                    |
//| Usage: control that is displayed by                              |
//|             the CChartObjectButton object                        |
//+------------------------------------------------------------------+
class CButton : public CWndObj
  {
private:
   CChartObjectButton m_button;             // chart object

public:
                     CButton(void);
                    ~CButton(void);
   //--- create
   virtual bool      Create(const long chart,const string name,const int subwin,const int x1,const int y1,const int x2,const int y2);
   //--- state
   bool              Pressed(void)          const { return(m_button.State());                       }
   bool              Pressed(const bool pressed)  { return(m_button.State(pressed));                }
   //--- properties
   bool              Locking(void)          const { return(IS_CAN_LOCK);                            }
   void              Locking(const bool flag);

   //--- theme methods
   void              SetTextColor(color clr)       { m_button.Color(clr);                           }
   void              SetBackgroundColor(color clr) { m_button.BackColor(clr);                       }
   void              SetBorderColor(color clr)     { m_button.BorderColor(clr);                     }

protected:
   //--- handlers of object settings
   virtual bool      OnSetText(void)              { return(m_button.Description(m_text));           }
   virtual bool      OnSetColor(void)             { return(m_button.Color(m_color));                }
   virtual bool      OnSetColorBackground(void)   { return(m_button.BackColor(m_color_background)); }
   virtual bool      OnSetColorBorder(void)       { return(m_button.BorderColor(m_color_border));   }
   virtual bool      OnSetFont(void)              { return(m_button.Font(m_font));                  }
   virtual bool      OnSetFontSize(void)          { return(m_button.FontSize(m_font_size));         }
   //--- internal event handlers
   virtual bool      OnCreate(void);
   virtual bool      OnShow(void);
   virtual bool      OnHide(void);
   virtual bool      OnMove(void);
   virtual bool      OnResize(void);
   virtual bool      OnMouseDown(void);
   virtual bool      OnMouseUp(void);
  };
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CButton::CButton(void)
  {
   m_color           =CONTROLS_BUTTON_COLOR;
   m_color_background=CONTROLS_BUTTON_COLOR_BG;
   m_color_border    =CONTROLS_BUTTON_COLOR_BORDER;
  }
//+------------------------------------------------------------------+
//| Destructor                                                       |
//+------------------------------------------------------------------+
CButton::~CButton(void)
  {
  }
//+------------------------------------------------------------------+
//| Create a control                                                 |
//+------------------------------------------------------------------+
bool CButton::Create(const long chart,const string name,const int subwin,const int x1,const int y1,const int x2,const int y2)
  {
//--- call method of the parent class
   if(!CWndObj::Create(chart,name,subwin,x1,y1,x2,y2))
      return(false);
//--- create the chart object
   if(!m_button.Create(chart,name,subwin,x1,y1,Width(),Height()))
      return(false);
//--- call the settings handler
   return(OnChange());
  }
//+------------------------------------------------------------------+
//| Locking flag                                                     |
//+------------------------------------------------------------------+
void CButton::Locking(const bool flag)
  {
   if(flag)
      PropFlagsSet(WND_PROP_FLAG_CAN_LOCK);
   else
      PropFlagsReset(WND_PROP_FLAG_CAN_LOCK);
  }
//+------------------------------------------------------------------+
//| Create object on chart                                           |
//+------------------------------------------------------------------+
bool CButton::OnCreate(void)
  {
//--- create the chart object by previously set parameters
   return(m_button.Create(m_chart_id,m_name,m_subwin,m_rect.left,m_rect.top,m_rect.Width(),m_rect.Height()));
  }
//+------------------------------------------------------------------+
//| Display object on chart                                          |
//+------------------------------------------------------------------+
bool CButton::OnShow(void)
  {
   return(m_button.Timeframes(OBJ_ALL_PERIODS));
  }
//+------------------------------------------------------------------+
//| Hide object from chart                                           |
//+------------------------------------------------------------------+
bool CButton::OnHide(void)
  {
   return(m_button.Timeframes(OBJ_NO_PERIODS));
  }
//+------------------------------------------------------------------+
//| Absolute movement of the chart object                            |
//+------------------------------------------------------------------+
bool CButton::OnMove(void)
  {
//--- position the chart object
   return(m_button.X_Distance(m_rect.left) && m_button.Y_Distance(m_rect.top));
  }
//+------------------------------------------------------------------+
//| Resize the chart object                                          |
//+------------------------------------------------------------------+
bool CButton::OnResize(void)
  {
//--- resize the chart object
   return(m_button.X_Size(m_rect.Width()) && m_button.Y_Size(m_rect.Height()));
  }
//+------------------------------------------------------------------+
//| Handler of click on the left mouse button                        |
//+------------------------------------------------------------------+
bool CButton::OnMouseDown(void)
  {
   if(!IS_CAN_LOCK)
      Pressed(!Pressed());
//--- call of the method of the parent class
   return(CWnd::OnMouseDown());
  }
//+------------------------------------------------------------------+
//| Handler of click on the left mouse button                        |
//+------------------------------------------------------------------+
bool CButton::OnMouseUp(void)
  {
//--- depress the button if it is not fixed
   if(m_button.State() && !IS_CAN_LOCK)
      m_button.State(false);
//--- call of the method of the parent class
   return(CWnd::OnMouseUp());
  }
//+------------------------------------------------------------------+
```

### [CEdit](https://www.mql5.com/en/docs/standardlibrary/controls/cedit?utm_campaign=search&utm_medium=special&utm_source=mt5editor) Class Theme Management

This is one of the key classes in our project that controls the input box where we will enter our message. By default, our panel and its components are set to a white background with black foreground text. When we click the theme switch button, the foreground color changes to white. However, during development, I noticed that the input box color remained unchanged, causing it to occasionally blend with the text during theme switching. Therefore, we need to add a method to the CEdit class to handle theme switching and ensure that the text input box aligns with our theme objectives.

The default CEdit class already has methods for setting colors (OnSetColor, OnSetColorBackground, and OnSetColorBorder). We can use these methods to update the appearance of the CEdit object when the theme changes. We employ new methods for Theme Switching say by adding these terms: SetTextColor, SetBackgroundColor, and SetBorderColor methods to the CEdit class. These methods update the respective colors and call the existing methods (OnSetColor, OnSetColorBackground, OnSetColorBorder) to apply the changes to the chart object.

```
//+------------------------------------------------------------------+
//| Set text color                                                   |
//+------------------------------------------------------------------+
bool CEdit::SetTextColor(const color clr)
  {
   m_color = clr;
   return(OnSetColor());
  }

//+------------------------------------------------------------------+
//| Set background color                                             |
//+------------------------------------------------------------------+
bool CEdit::SetBackgroundColor(const color clr)
  {
   m_color_background = clr;
   return(OnSetColorBackground());
  }

//+------------------------------------------------------------------+
//| Set border color                                                 |
//+------------------------------------------------------------------+
bool CEdit::SetBorderColor(const color clr)
  {
   m_color_border = clr;
   return(OnSetColorBorder());
  }
```

We will look at the unedited CEdit class source code below and will move on to share the incorporated program just below it.

**CEdit Default From MQL5:**

```
//+------------------------------------------------------------------+
//|                                                         Edit.mqh |
//|                             Copyright 2000-2024, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#include "WndObj.mqh"
#include <ChartObjects\ChartObjectsTxtControls.mqh>
//+------------------------------------------------------------------+
//| Class CEdit                                                      |
//| Usage: control that is displayed by                              |
//|             the CChartObjectEdit object                          |
//+------------------------------------------------------------------+
class CEdit : public CWndObj
  {
private:
   CChartObjectEdit  m_edit;                // chart object
   //--- parameters of the chart object
   bool              m_read_only;           // "read-only" mode flag
   ENUM_ALIGN_MODE   m_align_mode;          // align mode

public:
                     CEdit(void);
                    ~CEdit(void);
   //--- create
   virtual bool      Create(const long chart,const string name,const int subwin,const int x1,const int y1,const int x2,const int y2);
   //--- chart event handler
   virtual bool      OnEvent(const int id,const long &lparam,const double &dparam,const string &sparam);
   //--- parameters of the chart object
   bool              ReadOnly(void)         const { return(m_read_only);                          }
   bool              ReadOnly(const bool flag);
   ENUM_ALIGN_MODE   TextAlign(void)        const { return(m_align_mode);                         }
   bool              TextAlign(const ENUM_ALIGN_MODE align);
   //--- data access
   string            Text(void)             const { return(m_edit.Description());                 }
   bool              Text(const string value)     { return(CWndObj::Text(value));                 }

protected:
   //--- handlers of object events
   virtual bool      OnObjectEndEdit(void);
   //--- handlers of object settings
   virtual bool      OnSetText(void)              { return(m_edit.Description(m_text));           }
   virtual bool      OnSetColor(void)             { return(m_edit.Color(m_color));                }
   virtual bool      OnSetColorBackground(void)   { return(m_edit.BackColor(m_color_background)); }
   virtual bool      OnSetColorBorder(void)       { return(m_edit.BorderColor(m_color_border));   }
   virtual bool      OnSetFont(void)              { return(m_edit.Font(m_font));                  }
   virtual bool      OnSetFontSize(void)          { return(m_edit.FontSize(m_font_size));         }
   virtual bool      OnSetZOrder(void)            { return(m_edit.Z_Order(m_zorder));             }
   //--- internal event handlers
   virtual bool      OnCreate(void);
   virtual bool      OnShow(void);
   virtual bool      OnHide(void);
   virtual bool      OnMove(void);
   virtual bool      OnResize(void);
   virtual bool      OnChange(void);
   virtual bool      OnClick(void);
  };
//+------------------------------------------------------------------+
//| Common handler of chart events                                   |
//+------------------------------------------------------------------+
bool CEdit::OnEvent(const int id,const long &lparam,const double &dparam,const string &sparam)
  {
   if(m_name==sparam && id==CHARTEVENT_OBJECT_ENDEDIT)
      return(OnObjectEndEdit());
//--- event was not handled
   return(CWndObj::OnEvent(id,lparam,dparam,sparam));
  }
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CEdit::CEdit(void) : m_read_only(false),
                     m_align_mode(ALIGN_LEFT)
  {
   m_color           =CONTROLS_EDIT_COLOR;
   m_color_background=CONTROLS_EDIT_COLOR_BG;
   m_color_border    =CONTROLS_EDIT_COLOR_BORDER;
  }
//+------------------------------------------------------------------+
//| Destructor                                                       |
//+------------------------------------------------------------------+
CEdit::~CEdit(void)
  {
  }
//+------------------------------------------------------------------+
//| Create a control                                                 |
//+------------------------------------------------------------------+
bool CEdit::Create(const long chart,const string name,const int subwin,const int x1,const int y1,const int x2,const int y2)
  {
//--- call method of the parent class
   if(!CWndObj::Create(chart,name,subwin,x1,y1,x2,y2))
      return(false);
//--- create the chart object
   if(!m_edit.Create(chart,name,subwin,x1,y1,Width(),Height()))
      return(false);
//--- call the settings handler
   return(OnChange());
  }
//+------------------------------------------------------------------+
//| Set parameter                                                    |
//+------------------------------------------------------------------+
bool CEdit::ReadOnly(const bool flag)
  {
//--- save new value of parameter
   m_read_only=flag;
//--- set up the chart object
   return(m_edit.ReadOnly(flag));
  }
//+------------------------------------------------------------------+
//| Set parameter                                                    |
//+------------------------------------------------------------------+
bool CEdit::TextAlign(const ENUM_ALIGN_MODE align)
  {
//--- save new value of parameter
   m_align_mode=align;
//--- set up the chart object
   return(m_edit.TextAlign(align));
  }
//+------------------------------------------------------------------+
//| Create object on chart                                           |
//+------------------------------------------------------------------+
bool CEdit::OnCreate(void)
  {
//--- create the chart object by previously set parameters
   return(m_edit.Create(m_chart_id,m_name,m_subwin,m_rect.left,m_rect.top,m_rect.Width(),m_rect.Height()));
  }
//+------------------------------------------------------------------+
//| Display object on chart                                          |
//+------------------------------------------------------------------+
bool CEdit::OnShow(void)
  {
   return(m_edit.Timeframes(OBJ_ALL_PERIODS));
  }
//+------------------------------------------------------------------+
//| Hide object from chart                                           |
//+------------------------------------------------------------------+
bool CEdit::OnHide(void)
  {
   return(m_edit.Timeframes(OBJ_NO_PERIODS));
  }
//+------------------------------------------------------------------+
//| Absolute movement of the chart object                            |
//+------------------------------------------------------------------+
bool CEdit::OnMove(void)
  {
//--- position the chart object
   return(m_edit.X_Distance(m_rect.left) && m_edit.Y_Distance(m_rect.top));
  }
//+------------------------------------------------------------------+
//| Resize the chart object                                          |
//+------------------------------------------------------------------+
bool CEdit::OnResize(void)
  {
//--- resize the chart object
   return(m_edit.X_Size(m_rect.Width()) && m_edit.Y_Size(m_rect.Height()));
  }
//+------------------------------------------------------------------+
//| Set up the chart object                                          |
//+------------------------------------------------------------------+
bool CEdit::OnChange(void)
  {
//--- set up the chart object
   return(CWndObj::OnChange() && ReadOnly(m_read_only) && TextAlign(m_align_mode));
  }
//+------------------------------------------------------------------+
//| Handler of the "End of editing" event                            |
//+------------------------------------------------------------------+
bool CEdit::OnObjectEndEdit(void)
  {
//--- send the ON_END_EDIT notification
   EventChartCustom(CONTROLS_SELF_MESSAGE,ON_END_EDIT,m_id,0.0,m_name);
//--- handled
   return(true);
  }
//+------------------------------------------------------------------+
//| Handler of the "click" event                                     |
//+------------------------------------------------------------------+
bool CEdit::OnClick(void)
  {
//--- if editing is enabled, send the ON_START_EDIT notification
   if(!m_read_only)
     {
      EventChartCustom(CONTROLS_SELF_MESSAGE,ON_START_EDIT,m_id,0.0,m_name);
      //--- handled
      return(true);
     }
//--- else send the ON_CLICK notification
   return(CWnd::OnClick());
  }
//+------------------------------------------------------------------+
```

**CEdit With Theme Management Method Incorporated:**

See the highlighted sections.

```
//+------------------------------------------------------------------+
//|                                                         Edit.mqh |
//|                             Copyright 2000-2024, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#include "WndObj.mqh"
#include <ChartObjects\ChartObjectsTxtControls.mqh>

//+------------------------------------------------------------------+
//| Class CEdit                                                      |
//| Usage: control that is displayed by                              |
//|             the CChartObjectEdit object                          |
//+------------------------------------------------------------------+
class CEdit : public CWndObj
  {
private:
   CChartObjectEdit  m_edit;                // chart object
   //--- parameters of the chart object
   bool              m_read_only;           // "read-only" mode flag
   ENUM_ALIGN_MODE   m_align_mode;          // align mode

public:
                     CEdit(void);
                    ~CEdit(void);
   //--- create
   virtual bool      Create(const long chart,const string name,const int subwin,const int x1,const int y1,const int x2,const int y2);
   //--- chart event handler
   virtual bool      OnEvent(const int id,const long &lparam,const double &dparam,const string &sparam);
   //--- parameters of the chart object
   bool              ReadOnly(void)         const { return(m_read_only);                          }
   bool              ReadOnly(const bool flag);
   ENUM_ALIGN_MODE   TextAlign(void)        const { return(m_align_mode);                         }
   bool              TextAlign(const ENUM_ALIGN_MODE align);
   //--- data access
   string            Text(void)             const { return(m_edit.Description());                 }
   bool              Text(const string value)     { return(CWndObj::Text(value));                 }
   //--- theme handling
   bool              SetTextColor(const color clr);
   bool              SetBackgroundColor(const color clr);
   bool              SetBorderColor(const color clr);

protected:
   //--- handlers of object events
   virtual bool      OnObjectEndEdit(void);
   //--- handlers of object settings
   virtual bool      OnSetText(void)              { return(m_edit.Description(m_text));           }
   virtual bool      OnSetColor(void)             { return(m_edit.Color(m_color));                }
   virtual bool      OnSetColorBackground(void)   { return(m_edit.BackColor(m_color_background)); }
   virtual bool      OnSetColorBorder(void)       { return(m_edit.BorderColor(m_color_border));   }
   virtual bool      OnSetFont(void)              { return(m_edit.Font(m_font));                  }
   virtual bool      OnSetFontSize(void)          { return(m_edit.FontSize(m_font_size));         }
   virtual bool      OnSetZOrder(void)            { return(m_edit.Z_Order(m_zorder));             }
   //--- internal event handlers
   virtual bool      OnCreate(void);
   virtual bool      OnShow(void);
   virtual bool      OnHide(void);
   virtual bool      OnMove(void);
   virtual bool      OnResize(void);
   virtual bool      OnChange(void);
   virtual bool      OnClick(void);
  };

//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CEdit::CEdit(void) : m_read_only(false),
                     m_align_mode(ALIGN_LEFT)
  {
   m_color           =CONTROLS_EDIT_COLOR;
   m_color_background=CONTROLS_EDIT_COLOR_BG;
   m_color_border    =CONTROLS_EDIT_COLOR_BORDER;
  }

//+------------------------------------------------------------------+
//| Destructor                                                       |
//+------------------------------------------------------------------+
CEdit::~CEdit(void)
  {
  }

//+------------------------------------------------------------------+
//| Create a control                                                 |
//+------------------------------------------------------------------+
bool CEdit::Create(const long chart,const string name,const int subwin,const int x1,const int y1,const int x2,const int y2)
  {
   //--- call method of the parent class
   if(!CWndObj::Create(chart,name,subwin,x1,y1,x2,y2))
      return(false);
   //--- create the chart object
   if(!m_edit.Create(chart,name,subwin,x1,y1,Width(),Height()))
      return(false);
   //--- call the settings handler
   return(OnChange());
  }

//+------------------------------------------------------------------+
//| Set parameter                                                    |
//+------------------------------------------------------------------+
bool CEdit::ReadOnly(const bool flag)
  {
   //--- save new value of parameter
   m_read_only=flag;
   //--- set up the chart object
   return(m_edit.ReadOnly(flag));
  }

//+------------------------------------------------------------------+
//| Set parameter                                                    |
//+------------------------------------------------------------------+
bool CEdit::TextAlign(const ENUM_ALIGN_MODE align)
  {
   //--- save new value of parameter
   m_align_mode=align;
   //--- set up the chart object
   return(m_edit.TextAlign(align));
  }

//+------------------------------------------------------------------+
//| Set text color                                                   |
//+------------------------------------------------------------------+
bool CEdit::SetTextColor(const color clr)
  {
   m_color = clr;
   return(OnSetColor());
  }

//+------------------------------------------------------------------+
//| Set background color                                             |
//+------------------------------------------------------------------+
bool CEdit::SetBackgroundColor(const color clr)
  {
   m_color_background = clr;
   return(OnSetColorBackground());
  }

//+------------------------------------------------------------------+
//| Set border color                                                 |
//+------------------------------------------------------------------+
bool CEdit::SetBorderColor(const color clr)
  {
   m_color_border = clr;
   return(OnSetColorBorder());
  }

//+------------------------------------------------------------------+
//| Create object on chart                                           |
//+------------------------------------------------------------------+
bool CEdit::OnCreate(void)
  {
   //--- create the chart object by previously set parameters
   return(m_edit.Create(m_chart_id,m_name,m_subwin,m_rect.left,m_rect.top,m_rect.Width(),m_rect.Height()));
  }

//+------------------------------------------------------------------+
//| Display object on chart                                          |
//+------------------------------------------------------------------+
bool CEdit::OnShow(void)
  {
   return(m_edit.Timeframes(OBJ_ALL_PERIODS));
  }

//+------------------------------------------------------------------+
//| Hide object from chart                                           |
//+------------------------------------------------------------------+
bool CEdit::OnHide(void)
  {
   return(m_edit.Timeframes(OBJ_NO_PERIODS));
  }

//+------------------------------------------------------------------+
//| Absolute movement of the chart object                            |
//+------------------------------------------------------------------+
bool CEdit::OnMove(void)
  {
   //--- position the chart object
   return(m_edit.X_Distance(m_rect.left) && m_edit.Y_Distance(m_rect.top));
  }

//+------------------------------------------------------------------+
//| Resize the chart object                                          |
//+------------------------------------------------------------------+
bool CEdit::OnResize(void)
  {
   //--- resize the chart object
   return(m_edit.X_Size(m_rect.Width()) && m_edit.Y_Size(m_rect.Height()));
  }

//+------------------------------------------------------------------+
//| Set up the chart object                                          |
//+------------------------------------------------------------------+
bool CEdit::OnChange(void)
  {
   //--- set up the chart object
   return(CWndObj::OnChange() && ReadOnly(m_read_only) && TextAlign(m_align_mode));
  }

//+------------------------------------------------------------------+
//| Handler of the "End of editing" event                            |
//+------------------------------------------------------------------+
bool CEdit::OnObjectEndEdit(void)
  {
   //--- send the ON_END_EDIT notification
   EventChartCustom(CONTROLS_SELF_MESSAGE,ON_END_EDIT,m_id,0.0,m_name);
   //--- handled
   return(true);
  }

//+------------------------------------------------------------------+
//| Handler of the "click" event                                     |
//+------------------------------------------------------------------+
bool CEdit::OnClick(void)
  {
   //--- if editing is enabled, send the ON_START_EDIT notification
   if(!m_read_only)
     {
      EventChartCustom(CONTROLS_SELF_MESSAGE,ON_START_EDIT,m_id,0.0,m_name);
      //--- handled
      return(true);
     }
   //--- else send the ON_CLICK notification
   return(CWnd::OnClick());
  }

//+------------------------------------------------------------------+
```

We have successfully prepared our control include files for the Admin Panel, and we are closer to completing our project than ever before. In the next segment, we will finalize our efforts by adjusting the Admin Panel Expert Advisor code to support theme switching to align with the recent development.

### Adjusting the Admin Panel for Theme Switching.

There are logically four key areas in our theme management.

- The theme switching functionality in our Admin panel must be centered around the **darkTheme** boolean variable and the **UpdateThemeColors()** function. Here's how it works:

```
bool darkTheme = false;
```

- The above flag determines whether the current theme is dark or light. It is toggled upon pressing the **toggleThemeButton**, as seen below.

```
void OnToggleThemeButtonClick()
{
    darkTheme = !darkTheme;
    UpdateThemeColors();
    Print("Theme toggled: ", darkTheme ? "Dark" : "Light");
}
```

- Clicking the toggle theme button invokes this function, which flips the **darkTheme** flag and subsequently updates the UI's theme via **UpdateThemeColors()**.

```
void UpdateThemeColors()
{
    // Determine colors based on the current theme
    color textColor = darkTheme ? clrWhite : clrBlack;
    color buttonBgColor = darkTheme ? clrDarkSlateGray : clrGainsboro;
    color borderColor = darkTheme ? clrSlateGray : clrGray;
    color bgColor     = darkTheme ? clrDarkBlue : clrWhite;

    // Set text box colors
    inputBox.SetTextColor(textColor);
    inputBox.SetBackgroundColor(bgColor);
    inputBox.SetBorderColor(borderColor);

    // Update button colors
    UpdateButtonTheme(clearButton, textColor, buttonBgColor, borderColor);
    UpdateButtonTheme(sendButton, textColor, buttonBgColor, borderColor);
    UpdateButtonTheme(toggleThemeButton, textColor, buttonBgColor, borderColor);
    UpdateButtonTheme(changeFontButton, textColor, buttonBgColor, borderColor);
    UpdateButtonTheme(minimizeButton, textColor, buttonBgColor, borderColor);
    UpdateButtonTheme(maximizeButton, textColor, buttonBgColor, borderColor);
    UpdateButtonTheme(closeButton, textColor, buttonBgColor, borderColor);

    // Update quick message buttons
    for (int i = 0; i < ArraySize(quickMessageButtons); i++)
    {
        UpdateButtonTheme(quickMessageButtons[i], textColor, buttonBgColor, borderColor);
    }

    // Update character counter color
    charCounter.Color(textColor);

    // Redraw chart to apply changes
    ChartRedraw();
}
```

Based on the **darkTheme** flag, we chose different colors for text, button backgrounds, borders, and backgrounds. Colors are applied to various UI components as follows:

- Text Box (inputBox): Functions **SetTextColor**, **SetBackgroundColor**, and **SetBorderColor** are used to apply the theme.
- Buttons: The **UpdateButtonTheme()** function is called for each button, setting their text color, background color, and border color as determined.
- Character Counter: Directly sets its color when we click the theme button.

```
//Theme button application
void UpdateButtonTheme(CButton &button, color textColor, color bgColor, color borderColor)
{
    button.SetTextColor(textColor);
    button.SetBackgroundColor(bgColor);
    button.SetBorderColor(borderColor);
}
```

We employed a helper function above to apply all relevant theme-related color settings to any button. This cleans up the repeated code and ensures consistency across buttons. Summing up all the code snippets and integrating them into the main Admin Panel program, we have all the features performing as per goal.

### Final Code and Results

Here's the final draft of our program with the new features.

```
//+------------------------------------------------------------------+
//|                                             Admin Panel.mq5      |
//|                     Copyright 2024, Clemence Benjamin            |
//|     https://www.mql5.com/en/users/billionaire2024/seller         |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, Clemence Benjamin"
#property link      "https://www.mql5.com/en/users/billionaire2024/seller"
#property version   "1.12"

#include <Trade\Trade.mqh>
#include <Controls\Dialog.mqh>
#include <Controls\Button.mqh>
#include <Controls\Edit.mqh>
#include <Controls\Label.mqh>

// Input parameters
input string QuickMessage1 = "Updates";
input string QuickMessage2 = "Close all";
input string QuickMessage3 = "In deep profits";
input string QuickMessage4 = "Hold position";
input string QuickMessage5 = "Swing Entry";
input string QuickMessage6 = "Scalp Entry";
input string QuickMessage7 = "Book profit";
input string QuickMessage8 = "Invalid Signal";
input string InputChatId = "Enter Chat ID from Telegram bot API";
input string InputBotToken = "Enter BOT TOKEN from your Telegram bot";

// Global variables
CDialog adminPanel;
CButton sendButton, clearButton, changeFontButton, toggleThemeButton;
CButton quickMessageButtons[8], minimizeButton, maximizeButton, closeButton;
CEdit inputBox;
CLabel charCounter;
bool minimized = false;
bool darkTheme = false;
int MAX_MESSAGE_LENGTH = 4096;
string availableFonts[] = { "Arial", "Courier New", "Verdana", "Times New Roman" };
int currentFontIndex = 0;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
    // Initialize the Dialog
    if (!adminPanel.Create(ChartID(), "Admin Panel", 0, 30, 30, 500, 500))
    {
        Print("Failed to create dialog");
        return INIT_FAILED;
    }

    // Create controls
    if (!CreateControls())
    {
        Print("Control creation failed");
        return INIT_FAILED;
    }

    adminPanel.Show();
    UpdateThemeColors();

    Print("Initialization complete");
    return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Create necessary UI controls                                     |
//+------------------------------------------------------------------+
bool CreateControls()
{
    long chart_id = ChartID();

    // Create the input box
    if (!inputBox.Create(chart_id, "InputBox", 0, 5, 25, 460, 95))
    {
        Print("Failed to create input box");
        return false;
    }
    adminPanel.Add(inputBox);

    // Character counter
    if (!charCounter.Create(chart_id, "CharCounter", 0, 380, 5, 460, 25))
    {
        Print("Failed to create character counter");
        return false;
    }
    charCounter.Text("0/" + IntegerToString(MAX_MESSAGE_LENGTH));
    adminPanel.Add(charCounter);

    // Clear button
    if (!clearButton.Create(chart_id, "ClearButton", 0, 235, 95, 345, 125))
    {
        Print("Failed to create clear button");
        return false;
    }
    clearButton.Text("Clear");
    adminPanel.Add(clearButton);

    // Send button
    if (!sendButton.Create(chart_id, "SendButton", 0, 350, 95, 460, 125))
    {
        Print("Failed to create send button");
        return false;
    }
    sendButton.Text("Send");
    adminPanel.Add(sendButton);

    // Change font button
    if (!changeFontButton.Create(chart_id, "ChangeFontButton", 0, 95, 95, 230, 115))
    {
        Print("Failed to create change font button");
        return false;
    }
    changeFontButton.Text("Font<>");
    adminPanel.Add(changeFontButton);

    // Toggle theme button
    if (!toggleThemeButton.Create(chart_id, "ToggleThemeButton", 0, 5, 95, 90, 115))
    {
        Print("Failed to create toggle theme button");
        return false;
    }
    toggleThemeButton.Text("Theme<>");
    adminPanel.Add(toggleThemeButton);

    // Minimize button
    if (!minimizeButton.Create(chart_id, "MinimizeButton", 0, 375, -22, 405, 0))
    {
        Print("Failed to create minimize button");
        return false;
    }
    minimizeButton.Text("_");
    adminPanel.Add(minimizeButton);

    // Maximize button
    if (!maximizeButton.Create(chart_id, "MaximizeButton", 0, 405, -22, 435, 0))
    {
        Print("Failed to create maximize button");
        return false;
    }
    maximizeButton.Text("[ ]");
    adminPanel.Add(maximizeButton);

    // Close button
    if (!closeButton.Create(chart_id, "CloseButton", 0, 435, -22, 465, 0))
    {
        Print("Failed to create close button");
        return false;
    }
    closeButton.Text("X");
    adminPanel.Add(closeButton);

    // Quick messages
    return CreateQuickMessageButtons();
}

//+------------------------------------------------------------------+
//| Create quick message buttons                                     |
//+------------------------------------------------------------------+
bool CreateQuickMessageButtons()
{
    string quickMessages[8] = { QuickMessage1, QuickMessage2, QuickMessage3, QuickMessage4, QuickMessage5, QuickMessage6, QuickMessage7, QuickMessage8 };
    int startX = 5, startY = 160, width = 222, height = 65, spacing = 5;

    for (int i = 0; i < 8; i++)
    {
        if (!quickMessageButtons[i].Create(ChartID(), "QuickMessageButton" + IntegerToString(i + 1), 0, startX + (i % 2) * (width + spacing), startY + (i / 2) * (height + spacing), startX + (i % 2) * (width + spacing) + width, startY + (i / 2) * (height + spacing) + height))
        {
            Print("Failed to create quick message button ", i + 1);
            return false;
        }
        quickMessageButtons[i].Text(quickMessages[i]);
        adminPanel.Add(quickMessageButtons[i]);
    }
    return true;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    adminPanel.Destroy();
    Print("Deinitialization complete");
}

//+------------------------------------------------------------------+
//| Handle chart events                                              |
//+------------------------------------------------------------------+
void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
{
    switch (id)
    {
        case CHARTEVENT_OBJECT_CLICK:
            if (sparam == "SendButton") OnSendButtonClick();
            else if (sparam == "ClearButton") OnClearButtonClick();
            else if (sparam == "ChangeFontButton") OnChangeFontButtonClick();
            else if (sparam == "ToggleThemeButton") OnToggleThemeButtonClick();
            else if (sparam == "MinimizeButton") OnMinimizeButtonClick();
            else if (sparam == "MaximizeButton") OnMaximizeButtonClick();
            else if (sparam == "CloseButton") OnCloseButtonClick();
            else if (StringFind(sparam, "QuickMessageButton") != -1)
            {
                long index = StringToInteger(StringSubstr(sparam, 18));
                OnQuickMessageButtonClick(index - 1);
            }
            break;

        case CHARTEVENT_OBJECT_ENDEDIT:
            if (sparam == "InputBox") OnInputChange();
            break;
    }
}

//+------------------------------------------------------------------+
//| Handle custom message send button click                          |
//+------------------------------------------------------------------+
void OnSendButtonClick()
{
    string message = inputBox.Text();
    if (message != "")
    {
        if (SendMessageToTelegram(message))
            Print("Custom message sent: ", message);
        else
            Print("Failed to send custom message.");
    }
    else
    {
        Print("No message entered.");
    }
}

//+------------------------------------------------------------------+
//| Handle clear button click                                        |
//+------------------------------------------------------------------+
void OnClearButtonClick()
{
    inputBox.Text("");
    OnInputChange();
    Print("Input box cleared.");
}

//+------------------------------------------------------------------+
//| Handle quick message button click                                |
//+------------------------------------------------------------------+
void OnQuickMessageButtonClick(int index)
{
    string quickMessages[8] = { QuickMessage1, QuickMessage2, QuickMessage3, QuickMessage4, QuickMessage5, QuickMessage6, QuickMessage7, QuickMessage8 };
    string message = quickMessages[index];

    if (SendMessageToTelegram(message))
        Print("Quick message sent: ", message);
    else
        Print("Failed to send quick message.");
}

//+------------------------------------------------------------------+
//| Update character counter                                         |
//+------------------------------------------------------------------+
void OnInputChange()
{
    int currentLength = StringLen(inputBox.Text());
    charCounter.Text(IntegerToString(currentLength) + "/" + IntegerToString(MAX_MESSAGE_LENGTH));
    ChartRedraw();
}

//+------------------------------------------------------------------+
//| Handle toggle theme button click                                 |
//+------------------------------------------------------------------+
void OnToggleThemeButtonClick()
{
    darkTheme = !darkTheme;
    UpdateThemeColors();
    Print("Theme toggled: ", darkTheme ? "Dark" : "Light");
}

//+------------------------------------------------------------------+
//| Update theme colors for the panel                                |
//+------------------------------------------------------------------+
void UpdateThemeColors()
{
    // Use the dialog's theme update method as a placeholder.
    adminPanel.UpdateThemeColors(darkTheme);

    color textColor = darkTheme ? clrWhite : clrBlack;
    color buttonBgColor = darkTheme ? clrDarkSlateGray : clrGainsboro;
    color borderColor = darkTheme ? clrSlateGray : clrGray;
    color bgColor     = darkTheme?  clrDarkBlue : clrWhite;

          inputBox.SetTextColor(textColor);
          inputBox.SetBackgroundColor(bgColor);
          inputBox.SetBorderColor(borderColor);

    UpdateButtonTheme(clearButton, textColor, buttonBgColor, borderColor);
    UpdateButtonTheme(sendButton, textColor, buttonBgColor, borderColor);
    UpdateButtonTheme(toggleThemeButton, textColor, buttonBgColor, borderColor);
    UpdateButtonTheme(changeFontButton, textColor, buttonBgColor, borderColor);
    UpdateButtonTheme(minimizeButton, textColor, buttonBgColor, borderColor);
    UpdateButtonTheme(maximizeButton, textColor, buttonBgColor,borderColor);
    UpdateButtonTheme(closeButton, textColor, buttonBgColor, borderColor);


    for (int i = 0; i < ArraySize(quickMessageButtons); i++)
    {
        UpdateButtonTheme(quickMessageButtons[i], textColor, buttonBgColor, borderColor);
    }

    charCounter.Color(textColor);

    ChartRedraw();
}

//+------------------------------------------------------------------+
//| Apply theme settings to a button                                 |
//+------------------------------------------------------------------+
void UpdateButtonTheme(CButton &button, color textColor, color bgColor, color borderColor)
{
    button.SetTextColor(textColor);
    button.SetBackgroundColor(bgColor);
    button.SetBorderColor(borderColor);
}

//+------------------------------------------------------------------+
//| Handle change font button click                                  |
//+------------------------------------------------------------------+
void OnChangeFontButtonClick()
{
    currentFontIndex = (currentFontIndex + 1) % ArraySize(availableFonts);

    inputBox.Font(availableFonts[currentFontIndex]);
    clearButton.Font(availableFonts[currentFontIndex]);
    sendButton.Font(availableFonts[currentFontIndex]);
    toggleThemeButton.Font(availableFonts[currentFontIndex]);
    changeFontButton.Font(availableFonts[currentFontIndex]);

    for (int i = 0; i < ArraySize(quickMessageButtons); i++)
    {
        quickMessageButtons[i].Font(availableFonts[currentFontIndex]);
    }

    Print("Font changed to: ", availableFonts[currentFontIndex]);
    ChartRedraw();
}

//+------------------------------------------------------------------+
//| Handle minimize button click                                     |
//+------------------------------------------------------------------+
void OnMinimizeButtonClick()
{
    minimized = true;
    adminPanel.Hide();
    minimizeButton.Hide();
    maximizeButton.Show();
    closeButton.Show();
    Print("Panel minimized.");
}

//+------------------------------------------------------------------+
//| Handle maximize button click                                     |
//+------------------------------------------------------------------+
void OnMaximizeButtonClick()
{
    if (minimized)
    {
        adminPanel.Show();
        minimizeButton.Show();
        maximizeButton.Hide();
        closeButton.Hide();
        Print("Panel maximized.");
    }
}

//+------------------------------------------------------------------+
//| Handle close button click                                        |
//+------------------------------------------------------------------+
void OnCloseButtonClick()
{
    ExpertRemove();
    Print("Admin Panel closed.");
}

//+------------------------------------------------------------------+
//| Send the message to Telegram                                     |
//+------------------------------------------------------------------+
bool SendMessageToTelegram(string message)
{
    string url = "https://api.telegram.org/bot" + InputBotToken + "/sendMessage";
    string jsonMessage = "{\"chat_id\":\"" + InputChatId + "\", \"text\":\"" + message + "\"}";
    char post_data[];
    ArrayResize(post_data, StringToCharArray(jsonMessage, post_data, 0, WHOLE_ARRAY) - 1);

    int timeout = 5000;
    char result[];
    string responseHeaders;

    int res = WebRequest("POST", url, "Content-Type: application/json\r\n", timeout, post_data, result, responseHeaders);

    if (res == 200)
    {
        Print("Message sent successfully: ", message);
        return true;
    }
    else
    {
        Print("Failed to send message. HTTP code: ", res, " Error code: ", GetLastError());
        Print("Response: ", CharArrayToString(result));
        return false;
    }
}
```

After successfully compiling, we launched our program as demonstrated below, showcasing some impressive effects from the theme buttons. The theme switching functionality is working effectively, significantly enhancing the visual presentation. One amazing aspect is that these colors can be optimized within the code to suit your preferences. Furthermore, we could incorporate color input options to enable color customization outside the code.

![Advanced  Admin Panel](https://c.mql5.com/2/96/ShareX_xUjUZblfNl.gif)

New themed Admin Panel

The image below illustrates all the operations performed on the panel, including error handling. The failure to send a custom message can be attributed to missing or incorrect [Telegram bot token and chat ID](https://www.mql5.com/en/articles/14968#para3) entries. As seen in the image, these fields were left blank. It’s important to ensure these credentials are input correctly, as they are crucial for the operation. Please remember to keep these credentials secure to prevent unauthorized access.

![Experts Log](https://c.mql5.com/2/96/Experts_Log.PNG)

Experts Log

### Conclusion

This marks another milestone in the development of our Trading Systems Admin Panel. We have successfully incorporated theme management algorithms into existing classes without any observed performance issues affecting other platform features that rely on the same libraries. These advancements are primarily for learning and research purposes. However, modifying classes and integrating new methods can have a positive impact, but they also carry the risk of undesirable outcomes if not implemented correctly. Our project has now become more complex, integrating both [Telegram](https://www.mql5.com/en/articles/14968#para3) functionality and advanced visualization features.

I am pleased with our progress and hope that you have gained valuable insights from working with the library files in MQL5. There is still much more that can be achieved using some of the approaches employed in this project. I have attached the modified source files below. Please note that the theme switching feature relies on the presence of these library classes. If you encounter any issues, consider reinstalling [MetaTrader 5](https://download.mql5.com/cdn/web/metaquotes.ltd/mt5/mt5setup.exe?utm_source=mt5editor&utm_campaign=search "https://download.mql5.com/cdn/web/metaquotes.ltd/mt5/mt5setup.exe?utm_source=mt5editor&utm_campaign=search") to restore the system files and reset the classes to their default state.

The modifications and discussions regarding MQL5 library files, particularly those related to GUI components, are provided for educational purposes only. While editing these files can yield visually appealing results, please proceed with caution. Modifying header files may introduce unexpected behavior, instability, or compatibility issues within your trading applications. It is essential to thoroughly understand the changes being made and to keep backups of the original files. We recommend testing any modifications in a safe environment before deploying them to live trading systems. The author of this material bear no responsibility for any issues that may arise from the editing of MQL5 header files.

[Back to Contents page](https://www.mql5.com/en/articles/16045#para0)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/16045.zip "Download all attachments in the single ZIP archive")

[Admin\_Panel.mq5](https://www.mql5.com/en/articles/download/16045/admin_panel.mq5 "Download Admin_Panel.mq5")(14.69 KB)

[Extended\_MQL5\_Header\_Files.zip](https://www.mql5.com/en/articles/download/16045/extended_mql5_header_files.zip "Download Extended_MQL5_Header_Files.zip")(10.06 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [From Novice to Expert: Higher Probability Signals](https://www.mql5.com/en/articles/20658)
- [From Novice to Expert: Navigating Market Irregularities](https://www.mql5.com/en/articles/20645)
- [From Novice to Expert: Automating Trade Discipline with an MQL5 Risk Enforcement EA](https://www.mql5.com/en/articles/20587)
- [From Novice to Expert: Trading the RSI with Market Structure Awareness](https://www.mql5.com/en/articles/20554)
- [From Novice to Expert: Developing a Geographic Market Awareness with MQL5 Visualization](https://www.mql5.com/en/articles/20417)
- [The MQL5 Standard Library Explorer (Part 5): Multiple Signal Expert](https://www.mql5.com/en/articles/20289)
- [The MQL5 Standard Library Explorer (Part 4): Custom Signal Library](https://www.mql5.com/en/articles/20266)

**[Go to discussion](https://www.mql5.com/en/forum/474355)**

![Reimagining Classic Strategies (Part IX): Multiple Time Frame Analysis (II)](https://c.mql5.com/2/96/Reimagining_Classic_Strategies_Part_IX___LOGO.png)[Reimagining Classic Strategies (Part IX): Multiple Time Frame Analysis (II)](https://www.mql5.com/en/articles/15972)

In today's discussion, we examine the strategy of multiple time-frame analysis to learn on which time frame our AI model performs best. Our analysis leads us to conclude that the Monthly and Hourly time-frames produce models with relatively low error rates on the EURUSD pair. We used this to our advantage and created a trading algorithm that makes AI predictions on the Monthly time frame, and executes its trades on the Hourly time frame.

![Self Optimizing Expert Advisor With MQL5 And Python (Part V): Deep Markov Models](https://c.mql5.com/2/96/Self_Optimizing_Expert_Advisor_With_MQL5_And_Python_Part_V___LOGO.png)[Self Optimizing Expert Advisor With MQL5 And Python (Part V): Deep Markov Models](https://www.mql5.com/en/articles/16030)

In this discussion, we will apply a simple Markov Chain on an RSI Indicator, to observe how price behaves after the indicator passes through key levels. We concluded that the strongest buy and sell signals on the NZDJPY pair are generated when the RSI is in the 11-20 range and 71-80 range, respectively. We will demonstrate how you can manipulate your data, to create optimal trading strategies that are learned directly from the data you have. Furthermore, we will demonstrate how to train a deep neural network to learn to use the transition matrix optimally.

![Header in the Connexus (Part 3): Mastering the Use of HTTP Headers for Requests](https://c.mql5.com/2/99/http60x60__3.png)[Header in the Connexus (Part 3): Mastering the Use of HTTP Headers for Requests](https://www.mql5.com/en/articles/16043)

We continue developing the Connexus library. In this chapter, we explore the concept of headers in the HTTP protocol, explaining what they are, what they are for, and how to use them in requests. We cover the main headers used in communications with APIs, and show practical examples of how to configure them in the library.

![Developing a robot in Python and MQL5 (Part 2): Model selection, creation and training, Python custom tester](https://c.mql5.com/2/79/Robot_development_in_Python_and_MQL5____Part_2____LOGO__2.png)[Developing a robot in Python and MQL5 (Part 2): Model selection, creation and training, Python custom tester](https://www.mql5.com/en/articles/14910)

We continue the series of articles on developing a trading robot in Python and MQL5. Today we will solve the problem of selecting and training a model, testing it, implementing cross-validation, grid search, as well as the problem of model ensemble.

[Running robots on virtual hosting is easyFollow our step-by-step MetaTrader VPS guide for beginnersRead![](https://www.mql5.com/ff/sh/au4fqg4kms7s9mq1z2/01.png)![](https://www.mql5.com/ff/sh/au4fqg4kms7s9mq1z2/02.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/articles/13586&a=uzpprdshbcrtxvjxpmescehprypbymxc&s=516438f25b531570d9b7d49dcfb29c82fa1021f5ede6571df8026dbfbafcd13f&uid=&ref=https://www.mql5.com/en/articles/16045&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5062620059727209901)

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