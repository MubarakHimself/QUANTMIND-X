---
title: Using Layouts and Containers for GUI Controls: The CBox Class
url: https://www.mql5.com/en/articles/1867
categories: Integration
relevance_score: 0
scraped_at: 2026-01-24T14:07:07.250859
---

[![](https://www.mql5.com/ff/sh/7h2yc16rtqsn2m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
MQL5 Channels - Market analysis\\
\\
Dozens of channels, thousands of subscribers and daily updates. Learn more about trading.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=glufvbpblsoxonicqfngsyuzwfebnilr&s=103cc3ab372a16872ca1698fc86368ffe3b3eaa21b59b4006d5c6c10f48ad545&uid=&ref=https://www.mql5.com/en/articles/1867&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5083348443776162264)

MetaTrader 5 / Examples


### Table of Contents

- [1\. Introduction](https://www.mql5.com/en/articles/1867#para2)
- [2\. Objectives](https://www.mql5.com/en/articles/1867#para3)
- [3\. The CBox Class](https://www.mql5.com/en/articles/1867#para4)
  - [3.1. Layout Styles](https://www.mql5.com/en/articles/1867#para5)
  - [3.2. Calculating Space Between Controls](https://www.mql5.com/en/articles/1867#para6)
  - [3.3. Alignment](https://www.mql5.com/en/articles/1867#para7)
  - [3.4. Rendering Components](https://www.mql5.com/en/articles/1867#para8)
  - [3.5. Component Resizing](https://www.mql5.com/en/articles/1867#para9)
  - [3.6. Recursive Rendering](https://www.mql5.com/en/articles/1867#para10)
- [4\. Implementation in a Dialog Window](https://www.mql5.com/en/articles/1867#para11)
- [5\. Examples](https://www.mql5.com/en/articles/1867#para12)
  - [5.1. Example #1: A Simple Pip Value Calculator](https://www.mql5.com/en/articles/1867#para13)
  - [5.2. Example #2: Reconstructing the Controls Example](https://www.mql5.com/en/articles/1867#para14)
- [6\. Advantages and Disadvantages](https://www.mql5.com/en/articles/1867#para15)
- [7\. Conclusion](https://www.mql5.com/en/articles/1867#para16)

### 1\. Introduction

Absolute positioning of controls within an application dialog window is the most direct way of creating a graphical user interface for an application. However, in some cases, this approach to [graphical user interface](https://www.mql5.com/en/docs/standardlibrary/controls "MQL5 Standard Library Controls") (GUI) design can be inconvenient, or even impractical. This article presents an alternative method of GUI creation based on layouts and containers, using one layout manager — the CBox class.

The layout manager class implemented and used in this article is roughly equivalent to those found in some mainstream programming languages such as [BoxLayout](https://www.mql5.com/go?link=https://docs.oracle.com/javase/tutorial/uiswing/layout/box.html "Java BoxLayout") (Java) and [Pack geometry manager](https://www.mql5.com/go?link=http://effbot.org/tkinterbook/pack.htm "Python Tkinter Pack geometry manager") (Python/Tkinter).

### 2\. Objectives

Looking at the SimplePanel and Controls examples available in MetaTrader 5, we can see that the controls within these panels are positioned pixel-by-pixel (absolute positioning). Each control is created and assigned a definite position in the client area, and the position of each control will depend on the control created before it, with some additional offsets. Although this is the natural approach, such a level of precision is not needed in most cases, and using this method can be disadvantageous on many levels.

Any programmer with sufficient skill would be able to design graphical user interfaces using precise pixel positions for graphical controls. However, it has the following disadvantages:

- It is usually impossible to prevent other components from being affected when one component's size or position is modified.
- Most of the code is not reusable — minor changes in the interface may sometimes require major changes in the code.
- It can be time-consuming, especially when designing more complex interfaces.

This prompts us to create a layout system with the following objectives:

- The code should be reusable.
- Changing one part of the interface should have minimal impact on other components.
- The positioning of components within the interface should be automatically calculated.

One implementation of such a system is introduced in this article using a container — the CBox Class.

### 3\. The CBox Class

An instance of the CBox class acts as a container or box — controls are added to that box, and CBox would automatically calculate the positioning of the controls within its allocated space. A typical instance of the CBox class would have the following layout:

![CBox Layout](https://c.mql5.com/2/19/box__1.gif)

Figure 1. CBox Layout

The outer box represents the size of the entire container, while the dotted box inside it represents the boundaries of the padding. The blue area represents the padding space. The remaining white space would be the entire space available for positioning the controls inside the container.

Depending on the complexity of the panel, the CBox class can be used in various ways. For example, it is possible for a container (CBox) to hold another container holding a set of controls. Or a container containing a control and another container. However, using containers only as siblings in a given parent container is highly recommended.

We construct CBox by extending [CWndClient](https://www.mql5.com/en/docs/standardlibrary/controls/cwndclient "CWndClient Control") (without the scrollbars), as shown in the following snippet:

```
#include <Controls\WndClient.mqh>
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CBox : public CWndClient
  {
public:
                     CBox();
                    ~CBox();
   virtual bool      Create(const long chart,const string name,const int subwin,
                           const int x1,const int y1,const int x2,const int y2);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CBox::CBox()
  {
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CBox::~CBox()
  {
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CBox::Create(const long chart,const string name,const int subwin,
                  const int x1,const int y1,const int x2,const int y2)
  {
   if(!CWndContainer::Create(chart,name,subwin,x1,y1,x2,y2))
      return(false);
   if(!CreateBack())
      return(false);
   if(!ColorBackground(CONTROLS_DIALOG_COLOR_CLIENT_BG))
      return(false);
   if(!ColorBorder(clrNONE))
      return(false);
   return(true);
  }
//+------------------------------------------------------------------+
```

It is also possible for the CBox class to directly inherit from [CWndContainer](https://www.mql5.com/en/docs/standardlibrary/controls/cwndcontainer "CWndContainer control"). However, doing this would deprive the class of some useful features, such as the background and border. Alternatively, a much simpler version may be achieved by directly extending from [CWndObj](https://www.mql5.com/en/docs/standardlibrary/controls/cwndobj "CWndObj control"), but you would need to add an instance of [CArrayObj](https://www.mql5.com/en/docs/standardlibrary/datastructures/carrayobj "CArrayObj") as one of its [private or protected members](https://www.mql5.com/en/docs/basis/oop/inheritance) and recreate the class methods involving the objects to be stored on that instance.

**3.1. Layout Styles**

CBox has two layout styles: vertical style and horizontal style.

A horizontal style would have the following basic layout:

![Horizontal Style for CBox](https://c.mql5.com/2/19/horizontalbox.gif)

Figure 2. Horizontal Style (Centered)

A vertical style would have the following basic layout:

![Vertical Style for CBox](https://c.mql5.com/2/19/verticalbox.gif)

Figure 3. Vertical Style (Centered)

CBox would use a horizontal style by default.

Using a combination of these two layouts (possibly using multiple containers), it is possible to recreate virtually any type of GUI panel design. Furthermore, placing controls within containers would allow for a segmented design. That is, it allows one to customize the sizes and positioning of the controls in a given container, without affecting those held by other containers.

In order to implement horizontal and vertical styles in CBox, we would need to declare an enumeration, which we will then store as one of the members of the said class:

```
enum LAYOUT_STYLE
  {
   LAYOUT_STYLE_VERTICAL,
   LAYOUT_STYLE_HORIZONTAL
  };
```

**3.2. Calculating Space Between Controls**

CBox would maximize the available space allocated within, and will use it to position the controls it contains evenly, as shown in earlier figures.

Looking at the figures above, we can then derive the formula for calculating the space between controls in a given CBox container, using the pseudocode below:

```
for horizontal layout:
x space = ((available space x)-(total x size of all controls))/(total number of controls + 1)
y space = ((available space y)-(y size of control))/2

for vertical layout:
x space = ((available space x)-(x size of control))/2
y space = ((available space y)-(total y size of all controls))/(total number of controls + 1)
```

**3.3. Alignment**

The calculation of the space between controls, as mentioned in the previous section, only applies to a centered alignment. We would want the CBox class to accommodate more alignments as well, and so we will need some minor changes in the calculation.

For the horizontal alignment, the available options, aside from a centered container, are left, right, and center (no sides), as shown in the following figures:

![Horizontal box - align left](https://c.mql5.com/2/19/horizontalbox_alignleft.gif)

Figure 4. Horizontal Style (Align Left)

![Horizontal box - align right](https://c.mql5.com/2/19/horizontalbox_alignright.gif)

Figure 5. Horizontal Style (Align Right)

![Horizontal box - align center (no sides)](https://c.mql5.com/2/19/horizontalbox_aligncenter_nosides.gif)

Figure 6. Horizontal Style (Centered, No Sides)

For the vertical alignment, the available options, aside from the centered alignment, are top, bottom, center, and center (no sides), as shown below:

![Vertical box - align top](https://c.mql5.com/2/19/verticalbox_aligntop.gif)![Vertical box - align center (no sides)](https://c.mql5.com/2/19/verticalbox_aligncenter_nosides.gif)![Vertical box - align bottom](https://c.mql5.com/2/19/verticalbox_alignbottom.gif)

Figure 7. Vertical Style Alignments: (Left) Align Top, (Center) Align Center - No Sides, (Right) Align Bottom

Note that the CBox class should automatically calculate the x- and y-spacing between controls based on these alignments. Thus, rather than using a divisor of

```
(total number of controls + 1)
```

to get the space between controls, we use the total number of controls for skewed alignments (right, left, top, and bottom alignments) as divisor, and (total number of controls - 1) for a centered container with no margin on sides.

Similar to the layout styles, implementing alignment features to the CBox class would require enumerations. We will declare one enumeration for each alignment style, as follows:

```
enum VERTICAL_ALIGN
  {
   VERTICAL_ALIGN_CENTER,
   VERTICAL_ALIGN_CENTER_NOSIDES,
   VERTICAL_ALIGN_TOP,
   VERTICAL_ALIGN_BOTTOM
  };
enum HORIZONTAL_ALIGN
  {
   HORIZONTAL_ALIGN_CENTER,
   HORIZONTAL_ALIGN_CENTER_NOSIDES,
   HORIZONTAL_ALIGN_LEFT,
   HORIZONTAL_ALIGN_RIGHT
  };
```

**3.4. Rendering Components**

Normally, we create controls by specifying the x1, y1, x2, and y2 parameters, such as the following snippet when creating a [button](https://www.mql5.com/en/docs/standardlibrary/controls/cbutton "Button control"):

```
CButton m_button;
int x1 = currentX;
int y1 = currentY;
int x2 = currentX+BUTTON_WIDTH;
int y2 = currentY+BUTTON_HEIGHT
if(!m_button.Create(m_chart_id,m_name+"Button",m_subwin,x1,y1,x2,y2))
      return(false);
```

where x2 minus x1 and y2 minus y1 are equivalent to the width and height of the control, respectively. Rather than using this method, we can create the same button with CBox using a much simpler method, as shown in the following snippet:

```
if(!m_button.Create(m_chart_id,m_name+"Button",m_subwin,0,0,BUTTON_WIDTH,BUTTON_HEIGHT))
      return(false);
```

The CBox class would automatically reposition the component later in the creation of the panel window. The Pack() method, which calls the Render() method, should be invoked for repositioning of controls and containers:

```
bool CBox::Pack(void)
  {
   GetTotalControlsSize();
   return(Render());
  }
```

The Pack() method simply gets the combined size of the containers, and then calls the Render() method, where most of the action takes place. The snippet below shows the actual rendering of the controls within the container, through the Render() method:

```
bool CBox::Render(void)
  {
   int x_space=0,y_space=0;
   if(!GetSpace(x_space,y_space))
      return(false);
   int x=Left()+m_padding_left+
      ((m_horizontal_align==HORIZONTAL_ALIGN_LEFT||m_horizontal_align==HORIZONTAL_ALIGN_CENTER_NOSIDES)?0:x_space);
   int y=Top()+m_padding_top+
      ((m_vertical_align==VERTICAL_ALIGN_TOP||m_vertical_align==VERTICAL_ALIGN_CENTER_NOSIDES)?0:y_space);
   for(int j=0;j<ControlsTotal();j++)
     {
      CWnd *control=Control(j);
      if(control==NULL)
         continue;
      if(control==GetPointer(m_background))
         continue;
      control.Move(x,y);
      if (j<ControlsTotal()-1)
         Shift(GetPointer(control),x,y,x_space,y_space);
     }
   return(true);
  }
```

**3.5. Component Resizing**

When the size of a control is larger than the available space within its container, the control should be resized in order to fit the available space. Otherwise, the control will spill over the container, causing issues in the appearance of the entire panel. This approach is also convenient when you wanted a certain control to maximize its space and take the entire width or height of the client area, or its container. If the width or height of a given control exceeds the width or height of the container minus the padding (both sides), the control will be resized to the maximum width or maximum height available.

Note that CBox would not resize the container when the total size of all the controls it holds exceeds the available space. In this case, either the size of the main dialog window ( [CDialog](https://www.mql5.com/en/docs/standardlibrary/controls/cdialog "CDialog control") or [CAppDialog](https://www.mql5.com/en/docs/standardlibrary/controls/cappdialog "CAppDialog control")), or that of the individual controls would need to be manually adjusted.

**3.6. Recursive Rendering**

For a simple usage of CBox, a single call to the Pack() method would be sufficient. However, for nested containers, the same method will need to be called so that all containers would be able to position their individual controls or containers. We can prevent this by adding a method to the function to implement the same method to its own controls if and only if the graphical control in question is an instance of the CBox class or any layout class. To do this, first, we define a [macro](https://www.mql5.com/en/docs/basis/preprosessor/constant "Preprocessor constants") and assign it a unique value:

```
#define CLASS_LAYOUT 999
```

Then, we override the Type() method of the [CObject](https://www.mql5.com/en/docs/standardlibrary/cobject "CObject") class so that it returns the value of the macro we just prepared:

```
virtual int       Type() const {return CLASS_LAYOUT;}
```

Finally, within the Pack() method of the CBox class, we will perform the rendering method to its child containers that are instances of a layout class:

```
for(int j=0;j<ControlsTotal();j++)
     {
      CWnd *control=Control(j);
      if(control==NULL)
         continue;
      if(control==GetPointer(m_background))
         continue;
      control.Move(x,y);

      //call control Pack() method if it is a layout class
      if(control.Type()==CLASS_LAYOUT)
        {
         CBox *container=control;
         container.Pack();
        }

      if (j<ControlsTotal()-1)
         Shift(GetPointer(control),x,y,x_space,y_space);
     }
```

The rendering method begins by calculating the available space for the controls within the container. These values are stored in m\_total\_x and m\_total\_y, respectively. The next task is to calculate the space between the controls, based on the layout style and alignment. The last step is to implement the actual repositioning of the controls within the container.

CBox keeps a tally of controls to be repositioned, as there are objects within the container that do not require repositioning, such as the [CWndClient](https://www.mql5.com/en/docs/standardlibrary/controls/cwndclient "CWndClient control") native background object, or possibly some other controls should CBox be extended.

CBox also keeps the minimum control size within the container (except the background), defined by m\_min\_size (struct CSize). Its purpose is to keep the controls uniformly stacked within the container, whether horizontally or vertically. Its definition is rather counter-intuitive, since this is actually the size of the largest control. However, here we define it as a minimum, since CBox would assume that this size is the minimum size, and would calculate the available space based on this size.

Note that the Shift() method follows a routine similar to the usual implementation of control positioning (absolute positioning). The rendering methods keep references to both of the x- and y- coordinates which are remembered and updated as CBox repositions each control. However, with CBox, this is done automatically, leaving the developer of the panel to simply set the actual size for each control to be used.

### 4\. Implementation in a Dialog Window

When using CBox, we are practically replacing the functionality of the native client area of [CDialog](https://www.mql5.com/en/docs/standardlibrary/controls/cdialog "CDialog control") or [CAppDialog](https://www.mql5.com/en/docs/standardlibrary/controls/cappdialog "CAppDialog control"), m\_client\_area, which is an instance of [CWndClient](https://www.mql5.com/en/docs/standardlibrary/controls/cwndclient "CWndClient control"). Thus, we have at least three options in this case:

1. Extend/Rewrite [CAppDialog](https://www.mql5.com/en/docs/standardlibrary/controls/cappdialog) or [CDialog](https://www.mql5.com/en/docs/standardlibrary/controls/cdialog "CDialog control") to have CBox replace the client area.
2. Use containers and add them on the client area.
3. Use a main CBox container to host other small containers.

Using the first option may take a lot of effort, as we will need to rewrite the dialog objects to make it use the new client area object. Alternatively, the dialog objects can be extended to use the custom container class, but will leave us with an instance of [CWndClient](https://www.mql5.com/en/docs/standardlibrary/controls/cwndclient "CWndClient control") (m\_client\_area) unused, taking memory space unnecessarily.

The second option is also feasible. We simply place controls within CBox containers, and then use pixel-positioning in order to add them to the client area. But this option does not fully utilize the potential of the CBox class, which is to design panels without having to be bothered about the positioning of individual controls and containers.

The third option is recommended. That is, we create a main CBox container to hold all other smaller containers and controls. This main container would occupy the width of the entire native client area, and will be added to it as its only child. This would make the native client area a bit redundant, but at least, it is still being used. Furthermore, we can avoid a great deal of coding/recoding using this option.

### 5\. Examples

**5.1. Example #1: A Simple Pip Value Calculator**

Now, we use the CBox class to implement a simple panel: a pip value calculator. The pip value calculator dialog would contain three fields of type CEdit, namely:

- [name of symbol](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/csymbolinfo/csymbolinfoname "CSymbolInfo object") or instrument;
- size of 1 pip for the input symbol or instrument;
- value of 1 pip for the symbol or instrument.

This gives us a total of 7 different controls, including the labels ( [CLabel](https://www.mql5.com/en/docs/standardlibrary/controls/clabel "CLabel control")) for each field, and a button ( [CButton](https://www.mql5.com/en/docs/standardlibrary/controls/cbutton "Button control")) for executing a calculation. A screen shot of the calculator is shown below:

![Pip value calculator - screen shot](https://c.mql5.com/2/19/pipvaluecalc.png)

Figure 8. Pip Value Calculator

By looking at the calculator panel, we can deduct that it will use 5 different CBox containers. There should be 3 horizontal containers for each of the fields, and another horizontal container, aligned right, for the button. All these containers will be packaged inside the main container with a vertical style. And finally, this main container should be attached to the client area of the [CAppDialog](https://www.mql5.com/en/docs/standardlibrary/controls/cappdialog "CAppDialog control") instance. The following figure shows the layout of the containers. The violet boxes represent the horizontal rows. The white boxes represent the essential controls, while the large gray box containing all the smaller boxes is the main box window.

![Pip Value Calculator - dialog layout](https://c.mql5.com/2/19/pipvaluecalc.gif)

Figure 9. Pip Value Calculator layout

Notice that using CBox containers, we do not declare any macros for gaps and indentations. Rather we simply declare macros for control sizes, configure each CBox instance, and let them arrange the controls accordingly.

To construct this panel, first we begin by creating a header file, 'PipValueCalculator.mqh', which should be on the same folder as the main source file that we will prepare later (PipValueCalculator.mq5). We include in this file the CBox class header file, as well as other includes that we need for this panel. We would also require the [CSymbolInfo](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/csymbolinfo) class, which we will use for the actual calculation of pip value for any given symbol:

```
#include <Trade\SymbolInfo.mqh>
#include <Layouts\Box.mqh>
#include <Controls\Dialog.mqh>
#include <Controls\Label.mqh>
#include <Controls\Button.mqh>
```

The next step is to specify the width and height for the controls that we will use. It is possible to specify a certain size for each control, but for this panel, we will use a generic control size. That is, all essential controls will have the same width and height:

```
#define CONTROL_WIDTH   (100)
#define CONTROL_HEIGHT  (20)
```

Now, we move on to creating the actual panel class object. This is done in the usual way, by having a new class inherit from [CAppDialog](https://www.mql5.com/en/docs/standardlibrary/controls/cappdialog):

```
class CPipValueCalculatorDialog : public CAppDialog
```

The initial structure of the class will look similar to the following:

```
class CPipValueCalculatorDialog : public CAppDialog
  {
protected:
//protected class members here
public:
                     CPipValueCalculatorDialog();
                    ~CPipValueCalculatorDialog();

protected:
//protected class methods here
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CPipValueCalculatorDialog::CPipValueCalculatorDialog(void)
  {
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CPipValueCalculatorDialog::~CPipValueCalculatorDialog(void)
  {
  }
```

From the code snippet above, we now have a starting template for the pip value calculator panel class (actually, this can be reused to create similar panels). Now, we proceed with the creation of the main container class member, which will serve as the parent container of all other CBox containers found on the panel:

```
class CPipValueCalculatorDialog : public CAppDialog
  {
protected:
   CBox              m_main;
//more code here...
```

We have defined the main CBox container for the panel, but not the actual function for creating it. To do this, we add another class method to the panel class, as shown in the following:

```
// start of class definition
// ...
public:
                     CPipValueCalculatorDialog();
                    ~CPipValueCalculatorDialog();
protected:
   virtual bool      CreateMain(const long chart,const string name,const int subwin);
// the rest of the definition
// ...
```

Then, outside the class, we define the actual body of the class method (similar to [how the the body of the class constructor and destructor are defined](https://www.mql5.com/en/docs/basis/types/classes)):

```
bool CPipValueCalculatorDialog::CreateMain(const long chart,const string name,const int subwin)
  {
   //create main CBox container
   if(!m_main.Create(chart,name+"main",subwin,0,0,CDialog::m_client_area.Width(),CDialog::m_client_area.Height()))
      return(false);

   //apply vertical layout
   m_main.LayoutStyle(LAYOUT_STYLE_VERTICAL);

   //set padding to 10 px on all sides
   m_main.Padding(10);
   return(true);
  }
```

We use CDialog::m\_client\_area.Width() and CDialog::m\_client\_area.Height() to specify the width and height of the container. That is, it takes the entire space of the panel client area. We also apply some modifications to the container: the application of a vertical style, and setting the padding to 10 pixels on all sides. These functions are provided by the CBox Class.

Now that we have defined the main container class member and how it should be created, we then create the members for the rows as shown in Fig. 9. For the topmost row, which is the row for the symbol, we declare them by first creating the container, and then the essential controls contained within it, just below the declaration for the main container class member shown in the previous snippet:

```
CBox              m_main;
CBox              m_symbol_row;   //row container
CLabel            m_symbol_label; //label control
CEdit             m_symbol_edit;  //edit control
```

Similar to the main container, we also define a function for the creation of the actual row container:

```
bool CPipValueCalculatorDialog::CreateSymbolRow(const long chart,const string name,const int subwin)
  {
   //create CBox container for this row (symbol row)
   if(!m_symbol_row.Create(chart,name+"symbol_row",subwin,0,0,CDialog::m_client_area.Width(),CONTROL_HEIGHT*1.5))
      return(false);

   //create label control
   if(!m_symbol_label.Create(chart,name+"symbol_label",subwin,0,0,CONTROL_WIDTH,CONTROL_HEIGHT))
      return(false);
   m_symbol_label.Text("Symbol");

   //create edit control
   if(!m_symbol_edit.Create(chart,name+"symbol_edit",subwin,0,0,CONTROL_WIDTH,CONTROL_HEIGHT))
      return(false);
   m_symbol_edit.Text(m_symbol.Name());

   //add the essential controls to their parent container (row)
   if(!m_symbol_row.Add(m_symbol_label))
      return(false);
   if(!m_symbol_row.Add(m_symbol_edit))
      return(false);
   return(true);
  }
```

In this function, we first create the symbol row container. Notice that we use the entire width of the client area as its width, while making its height 50% larger than the control height we defined earlier.

After the creation of the row, we then create the individual controls. This time, they use the control width and height macros we defined earlier. Also notice how we create these controls:

```
Create(chart,name+"symbol_edit",subwin,0,0,CONTROL_WIDTH,CONTROL_HEIGHT))
```

The values in red are the x1 and y1 coordinates. This means that at creation, all controls are placed at the upper left hand side of the chart. These are then rearranged as soon as we call the Pack() method of CBox.

We have created the row container. We have also created the essential controls within the container. The next step was to add the controls we just created to the row container, represented in the last several lines of the function:

```
if(!m_symbol_row.Add(m_symbol_label))
   return(false);
if(!m_symbol_row.Add(m_symbol_edit))
   return(false);
```

For the other rows (pip size, pip value, and button rows), we implement roughly the same method as we have done for the symbol row.

The creation of the main container and other child rows is needed when using the CBox class. Now, we move on to more familiar ground, which is the creation of the panel object itself. This is done (whether or not using CBox) by overriding the virtual method Create() of the [CAppDialog](https://www.mql5.com/en/docs/standardlibrary/controls/cappdialog) class. It is under this method where the two methods we defined earlier will finally make sense, since we will invoke those methods here:

```
bool CPipValueCalculatorDialog::Create(const long chart,const string name,const int subwin,const int x1,const int y1,const int x2,const int y2)
  {
   //create CAppDialog panel
   if(!CAppDialog::Create(chart,name,subwin,x1,y1,x2,y2))
      return(false);

   //create main CBox container using the function we defined earlier
   if(!CreateMain(chart,name,subwin))
      return(false);

   //create symbol row CBox container using the function we defined earlier
   if(!CreateSymbolRow(chart,name,subwin))
      return(false);

   //add the symbol row CBox container as a child of the main CBox container
   if(!m_main.Add(m_symbol_row))
      return(false);

   //render the main CBox container and all its child containers (recursively)
   if (!m_main.Pack())
      return(false);

   //add the main CBox container as the only child of the panel client area
   if (!Add(m_main))
      return(false);
   return(true);
  }
```

Don't forget to declare the overridden Create() method in the CPipValueCalculatorDialog class, as follows:

```
public:
                     CPipValueCalculatorDialog();
                    ~CPipValueCalculatorDialog();
   virtual bool      Create(const long chart,const string name,const int subwin,const int x1,const int y1,const int x2,const int y2);
```

As shown in the code above, it should be a public class method, since we will call it outside the class. To be more specific, this will be needed in the main source file: PipValueCalculator.mq5:

```
#include "PipValueCalculator.mqh"
CPipValueCalculatorDialog ExtDialog;
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---
//--- create application dialog
   if(!ExtDialog.Create(0,"Pip Value Calculator",0,50,50,279,250))
      return(INIT_FAILED);
//--- run application
   if(!ExtDialog.Run())
      return(INIT_FAILED);
//--- ok
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---
   ExtDialog.Destroy(reason);
  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//---
  }
//+------------------------------------------------------------------+
void OnChartEvent(const int id,
                  const long &lparam,
                  const double &dparam,
                  const string &sparam)
  {
    //commented for now, to be discussed later in the section
    //ExtDialog.ChartEvent(id,lparam,dparam,sparam);
  }
//+------------------------------------------------------------------+
```

This code is very similar to what we usually see on panel main source files, except for three things:

1. We add 'PipValueCalculator.mqh' as a header file, rather than the include file for CAppDalog. 'PipValueCalculator.mqh' already includes the header file for it, so there is no longer any need to include it in the main source file. 'PipValueCalculator.mqh' is also responsible for including the CBox class header file.

2. We declare ExtDialog as an instance of the class we defined in 'PipValueCalculator.mqh' (PipValueCalculator class).
3. We specify a custom size for the panel that is more suited to it, as defined in ExtDialog.Create().

When compiled with only the symbol row, the panel would look similar to the following screenshot:

![Pip value calculator panel with one row](https://c.mql5.com/2/19/symbolrow_only.png)

Figure 10. Pip Value Calculator panel with one row

The main container has a vertical layout and is aligned center, while the symbol row shown as a horizontal layout (also centered horizontally and vertically). To make this panel resemble the one shown in Fig. 8, we need to add the other three rows, using basically the same method we implemented for the creation of the symbol row. One exception is the button row, which only contains a single essential control (button), and should be aligned right:

```
m_button_row.HorizontalAlign(HORIZONTAL_ALIGN_RIGHT);
```

The handling of events is beyond the scope of this article, but for the sake of completeness for this example, we will discuss it briefly here. We begin by declaring a new class member for the PipValueCalculator class, m\_symbol. We also include two additional members, m\_digits\_adjust and m\_points\_adjust, which will be used later to convert size in points to pips.

```
CSymbolInfo      *m_symbol;
int               m_digits_adjust;
double            m_points_adjust;
```

We initialize m\_symbol either in the class constructor or in Create() method, using the following code:

```
if (m_symbol==NULL)
      m_symbol=new CSymbolInfo();
if(m_symbol!=NULL)
{
   if (!m_symbol.Name(_Symbol))
      return(false);
}
```

If the symbol pointer is null, we create a new instance of CSymbolInfo. If it is not null, we assign the symbol name equal to the chart symbol name.

The next step would be to define a click event handler for the button. This is implemented by the OnClickButton() class method. We define its body as follows:

```
void CPipValueCalculatorDialog::OnClickButton()
  {
   string symbol=m_symbol_edit.Text();
   StringToUpper(symbol);
   if(m_symbol.Name(symbol))
     {
      m_symbol.RefreshRates();
      m_digits_adjust=(m_symbol.Digits()==3 || m_symbol.Digits()==5)?10:1;
      m_points_adjust=m_symbol.Point()*m_digits_adjust;
      m_pip_size_edit.Text((string)m_points_adjust);
      m_pip_value_edit.Text(DoubleToString(m_symbol.TickValue()*(StringToDouble(m_pip_size_edit.Text()))/m_symbol.TickSize(),2));
     }
   else Print("invalid input");
  }
```

The class method calculates the pip value by first getting the value of the m\_symbol\_edit control. It then passes the name of the symbol to an instance of the [CSymbolInfo](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/csymbolinfo) class. The said class gets the tick value of the chosen symbol, which is then adjusted by a certain multiplier in order to compute for the value of 1 pip.

The final step to enable event handling for the class is the definition of the event handler (also within the PipValueCalculator class). Under the public methods of the class, insert this line of code:

```
virtual bool      OnEvent(const int id,const long &lparam,const double &dparam,const string &sparam);
```

Then, we define the body of class method outside the class, using the following snippet:

```
EVENT_MAP_BEGIN(CPipValueCalculatorDialog)
   ON_EVENT(ON_CLICK,m_button,OnClickButton)
EVENT_MAP_END(CAppDialog)
```

**5.2. Example #2: Reconstructing the Controls Example**

The Controls panel example is automatically installed after a fresh installation of MetaTrader. On the navigator window, it can be found under Expert Advisors\\Examples\\Controls. A screenshot of this panel is shown below:

![Controls - dialog](https://c.mql5.com/2/19/controls_orig.png)

Figure 11. Controls Dialog (Original)

The layout of the dialog window shown above is detailed in the following figure. To reconstruct the panel using CBox instances, we are seeing 4 main horizontal rows (in violet) for the following set of controls:

1. the [Edit](https://www.mql5.com/en/docs/standardlibrary/controls/cedit) control;
2. the three [Button](https://www.mql5.com/en/docs/standardlibrary/controls/cbutton "Button control") controls;
3. the [SpinEdit](https://www.mql5.com/en/docs/standardlibrary/controls/cspinedit) and the DatePicker;
4. the [ComboBox](https://www.mql5.com/en/docs/standardlibrary/controls/ccombobox "ComboBox control"), [RadioGroup](https://www.mql5.com/en/docs/standardlibrary/controls/cradiogroup "RadioGroup control") and [CheckGroup](https://www.mql5.com/en/docs/standardlibrary/controls/ccheckgroup) (Column 1) and the [ListView](https://www.mql5.com/en/docs/standardlibrary/controls/clistview "ListView control") (Column 2).

The last horizontal container is a special case, since it is a nested horizontal container holding two other containers (columns 1 and 2, in green). These containers should have vertical layouts.

![Figure 12. Controls Dialog layout](https://c.mql5.com/2/19/controls__2.gif)

Figure 12. Controls Dialog layout

When reconstructing the Controls dialog, any lines of code invoking the [Add()](https://www.mql5.com/en/docs/standardlibrary/controls/cwndcontainer/cwndcontaineradd "CWndContainer Add method") method, should be removed, except for the main container, which would act as the sole child of the dialog client area. Meanwhile, the other controls and containers should be added to their designated parent containers from the deepest levels up to the main container and then finally to the native client area.

Once installed, compile and execute. Everything should work fine, except for the date picker, whose increment, decrement, and list buttons would refuse to work. This is due to the fact that the drop list of the CDatePicker class is set on the background of the other containers. To resolve this issue, look for the CDatePicker class file located at %Data Folder%\\MQL5\\Include\\Controls\\DatePicker.mqh. Find the ListShow() method and at the beginning of that function, insert this line of code:

```
BringToTop();
```

Recompile and test. This would make the drop list of the Date Picker to be placed on the foreground, and given priority for click events whenever it is shown. Here is a snippet of the entire function:

```
bool CDatePicker::ListShow(void)
  {
   BringToTop();
//--- set value
   m_list.Value(m_value);
//--- show the list
   return(m_list.Show());
  }
```

A screenshot of the reconstructed Controls dialog is shown below:

![Controls - dialog reconstructed](https://c.mql5.com/2/19/controls_cbox.png)

Figure 13. Controls Dialog (Using CBox)

Focusing on the big picture, it looks nearly identical to the original. But there is a striking difference, which was made incidentally — column 1 is perfectly aligned with column 2. In the original, we can see that the [CheckGroup](https://www.mql5.com/en/docs/standardlibrary/controls/ccheckgroup) is stacked uniformly with the [ListView](https://www.mql5.com/en/docs/standardlibrary/controls/clistview "ListView control") on the bottom. However, on the top, the [ComboBox](https://www.mql5.com/en/docs/standardlibrary/controls/ccombobox "ComboBox control") is not aligned with the top of the [ListView](https://www.mql5.com/en/docs/standardlibrary/controls/clistview "ListView control"). Of course, the coordinates can be repositioned on the original panel, but this would require the adjustment not just of the pixel coordinates for the [ComboBox](https://www.mql5.com/en/docs/standardlibrary/controls/ccombobox "ComboBoxcontrol"), but the coordinates of the [RadioGroup](https://www.mql5.com/en/docs/standardlibrary/controls/cradiogroup "RadioGroup control") and the gaps between the three controls as well. Using a CBox container, on the other hand, only required setting the top and bottom padding to zero, and using the correct alignment.

This does not mean, however, that using CBox or layouts is superior in terms of precision. Although admittedly less precise than encoding exact coordinates for controls, using containers and layouts would still be able to provide a decent level of precision, while at the same time making GUI design a bit easier.

### 6\. Advantages and Disadvantages

Advantages:

- Its code is reusable — you can reuse CBox or any layout class across different applications and dialogs.
- It is scalable — although using it may make the source code longer in small applications, its benefits can be more appreciated in more complex panels and dialogs.
- Segmentation of control sets — it allows you to modify a set of controls without affecting much the positioning of other controls.
- Automated positioning — indents, gaps, and spacings are not needed to be manually encoded, these are automatically calculated by the layout class.

Disadvantages:

- Additional controls will need to be created for the containers, as well as the creation of additional functions that utilize them.
- Less precise — the positioning is limited to the layouts and alignment options available.
- Can become problematic or too complex when used to contain controls with varying sizes — in this case, either the differences in the sizes can be kept at a minimum, or make use of nested containers.

### 7\. Conclusion

In this article, we have considered the possibility of using layouts and containers in the design of graphical panels. This approach has allowed us to automate the process of positioning various controls using layout and alignment styles. It can make designing graphical panels easier, and in some cases, reduce coding time.

The CBox class is an auxiliary control that acts as a container for essential controls in a GUI panel. In this article, we have demonstrated its operation and how it can be used in real applications. Although less precise than absolute positioning, it can still provide a level of precision that would prove to be convenient in many applications.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/1867.zip "Download all attachments in the single ZIP archive")

[Box.mqh](https://www.mql5.com/en/articles/download/1867/box.mqh "Download Box.mqh")(11.88 KB)

[Controls2.mq5](https://www.mql5.com/en/articles/download/1867/controls2.mq5 "Download Controls2.mq5")(2.13 KB)

[ControlsDialog2.mqh](https://www.mql5.com/en/articles/download/1867/controlsdialog2.mqh "Download ControlsDialog2.mqh")(20.18 KB)

[PipValueCalculator.mq5](https://www.mql5.com/en/articles/download/1867/pipvaluecalculator.mq5 "Download PipValueCalculator.mq5")(1.89 KB)

[PipValueCalculator.mqh](https://www.mql5.com/en/articles/download/1867/pipvaluecalculator.mqh "Download PipValueCalculator.mqh")(9.47 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Cross-Platform Expert Advisor: The CExpertAdvisor and CExpertAdvisors Classes](https://www.mql5.com/en/articles/3622)
- [Cross-Platform Expert Advisor: Custom Stops, Breakeven and Trailing](https://www.mql5.com/en/articles/3621)
- [Cross-Platform Expert Advisor: Stops](https://www.mql5.com/en/articles/3620)
- [Cross-Platform Expert Advisor: Time Filters](https://www.mql5.com/en/articles/3395)
- [Cross-Platform Expert Advisor: Money Management](https://www.mql5.com/en/articles/3280)
- [Cross-Platform Expert Advisor: Signals](https://www.mql5.com/en/articles/3261)
- [Cross-Platform Expert Advisor: Order Manager](https://www.mql5.com/en/articles/2961)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/61323)**
(17)


![Azman Waras](https://c.mql5.com/avatar/2018/3/5AA03CFF-2695.png)

**[Azman Waras](https://www.mql5.com/en/users/azmanwaras)**
\|
8 Mar 2018 at 01:44

**Alain Verleyen :**

Kode ini tidak dikompilasi Bangun 1702

'pipvaluecalculator.mq5' pipvaluecalculator.mq5 1 1

...

'm\_client\_area' - kesalahan akses anggota pribadi PipValueCalculator.mqh 118 60

'm\_client\_area' - kesalahan akses anggota pribadi PipValueCalculator.mqh 118 91

'm\_client\_area' - kesalahan akses anggota pribadi PipValueCalculator.mqh 129 72

'm\_client\_area' - kesalahan akses anggota pribadi PipValueCalculator.mqh 148 75

'm\_client\_area' - kesalahan akses anggota pribadi PipValueCalculator.mqh 168 77

'm\_client\_area' - kesalahan akses anggota pribadi PipValueCalculator.mqh 187 72

6 error (s), 0 warning (s) 7 1

Jadi catatan, kecuali jika saya melewatkannya, anda harus menentukan di mana tempat file-file tersebut. Sebagai catatan, kecuali jika saya melewatkannya, Anda harus menentukan di mana menempatkan file-file tersebut. Kita perlu mencoba dan melihat kode untuk mengetahui bahwa kita perlu membuat folder "Layouts" di Include dan menempatkan file Box.mqh di dalamnya.Kita perlu mencoba dan melihat kode untuk mengetahui bahwa kita perlu membuat folder "Layouts" di Include dan menempatkan file Box.mqh di dalamnya.

![Stanislav Korotky](https://c.mql5.com/avatar/2010/10/4CA7CFA0-1F0C.jpg)

**[Stanislav Korotky](https://www.mql5.com/en/users/marketeer)**
\|
29 Feb 2020 at 19:49

I have encountered a problem with this useful add-on.

I ask for advice from those who know about standard controls.

Since this add-in with panels uses [standard classes](https://www.mql5.com/en/articles/179 "Article: Create your view of the market through ready-made classes") CWnd, CWndClient, I assume that knowledge of their internal structure may be enough to understand where the "dog is in the fight".

The essence of the problem. We take the Controls2.mq5 programme from the article (for compilation we also need ControlsDialog2.mqh and Box.mqh), compile it, run it.

Almost everything works fine except for the "datapicker". It only opens and closes, but in the open state it does not react to clicks, skipping events to the underlying controls.

The original similar demo from MQ (Experts/Examples/Controls.mq5) works with "datapicker" normally.

![Stanislav Korotky](https://c.mql5.com/avatar/2010/10/4CA7CFA0-1F0C.jpg)

**[Stanislav Korotky](https://www.mql5.com/en/users/marketeer)**
\|
1 Mar 2020 at 19:58

**Stanislav Korotky:**

Almost everything works fine except for the "datapicker". It only opens and closes, but in the open state it doesn't react to presses, skipping events to underlying controls.

The original similar demo from MQ (Experts/Examples/Controls.mq5) works with "datapicker" normally.

The question is removed. In CDatePicker there is no BringToTop call when the dropdown is expanded, as is done in CComboBox, for example. In the standard example, the CDatePicker works due to the fact that its initialisation was moved (intentionally or accidentally) after the creation of the "leafbox", which is topologically below it. And the controls in CWndContainer::OnMouseEvent are bypassed from the last added to the first.

To fix it normally, it would be necessary to override CDatePicker::ListShow, but it is not virtual. We have to redefine CDatePicker::OnClickButton and add BringToTop there. However, we cannot do it correctly as well as with any virtual method in the [standard library](https://www.mql5.com/en/docs/standardlibrary "MQL5 Documentation: Standard Library"), because all member variables are declared private. In particular, it is impossible to write:

```
bool MyDatePicker::OnClickButton(void) // override
{
    return ((m_drop.Pressed()) ? BringToTop() && ListShow() : ListHide());
}
```

because m\_drop is not available. We have to call BringToTop both when opening and collapsing.

```
#include <Controls/DatePicker.mqh>

class CDatePickerFixed: public CDatePicker
{
  protected:
    virtual bool OnClickButton() override
    {
      BringToTop();
      return CDatePicker::OnClickButton();
    }
};
```

![Vitaliy Kostrubko](https://c.mql5.com/avatar/2016/8/579E94F7-83FB.png)

**[Vitaliy Kostrubko](https://www.mql5.com/en/users/bbk30)**
\|
3 Nov 2024 at 16:28

and there's a few mistakes:

[![](https://c.mql5.com/3/447/4376968880682__1.png)](https://c.mql5.com/3/447/4376968880682.png "https://c.mql5.com/3/447/4376968880682.png")

![CFA LAU](https://c.mql5.com/avatar/avatar_na2.png)

**[CFA LAU](https://www.mql5.com/en/users/cfaliu)**
\|
17 Jan 2025 at 07:45

Thank you for sharing


![Statistical Verification of the Labouchere Money Management System](https://c.mql5.com/2/18/labouchere.png)[Statistical Verification of the Labouchere Money Management System](https://www.mql5.com/en/articles/1800)

In this article, we test the statistical properties of the Labouchere money management system. It is considered to be a less aggressive kind of Martingale, since bets are not doubled, but are raised by a certain amount instead.

![Identifying Trade Setups by Support, Resistance and Price Action](https://c.mql5.com/2/19/avatar.png)[Identifying Trade Setups by Support, Resistance and Price Action](https://www.mql5.com/en/articles/1734)

This article shows how price action and the monitoring of support and resistance levels can be used for well-timed market entry. It discusses a trading system that effectively combines the two for the determination of trade setups. Corresponding MQL4 code is explained that can be utilized in the EAs based on these trading concepts.

![Drawing Dial Gauges Using the CCanvas Class](https://c.mql5.com/2/19/gg_cases.png)[Drawing Dial Gauges Using the CCanvas Class](https://www.mql5.com/en/articles/1699)

We can find dial gauges in cars and airplanes, in industrial production and everyday life. They are used in all spheres which require quick response to behavior of a controlled value. This article describes the library of dial gauges for MetaTrader 5.

![Tips for Selecting a Trading Signal to Subscribe. Step-By-Step Guide](https://c.mql5.com/2/18/signals__1.png)[Tips for Selecting a Trading Signal to Subscribe. Step-By-Step Guide](https://www.mql5.com/en/articles/1838)

This step-by-step guide is dedicated to the Signals service, examination of trading signals, a system approach to the search of a required signal which would satisfy criteria of profitability, risk, trading ambitions, working on various types of accounts and financial instruments.

[We've created a channel for MQL5 developersFollow MQL5.community on social media and be the first to receive important updatesLearn more![](https://www.mql5.com/ff/sh/a83xrgctr82w45z9z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=pwgdbvtemvkwsfltqonysypfvtrtufji&s=e99a66a1660cd810b1edbac65597df695e2c2220d1e937834f402f9aeabd4289&uid=&ref=https://www.mql5.com/en/articles/1867&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5083348443776162264)

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