---
title: Using Layouts and Containers for GUI Controls: The CGrid Class
url: https://www.mql5.com/en/articles/1998
categories: Integration, Indicators, Expert Advisors
relevance_score: 0
scraped_at: 2026-01-24T14:06:47.076847
---

[![](https://www.mql5.com/ff/si/3fgkjn78mkxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=ocndbzpeklfncxysjbwfhhbalbrsdbtv&s=a4309643278437a00bdd33c5809fc6b4b4032749c00fccd07b3b84e7b8b45126&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=mbnekgrkaflzbhwccrtllwuyrrbcvcpl&ssn=1769252805404639884&ssn_dr=0&ssn_sr=0&fv_date=1769252805&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F1998&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Using%20Layouts%20and%20Containers%20for%20GUI%20Controls%3A%20The%20CGrid%20Class%20-%20MQL5%20Articles&scr_res=1920x1080&ac=17692528053401685&fz_uniq=5083343203916061124&sv=2552)

MetaTrader 5 / Examples


### Table of Contents

- [1\. Introduction](https://www.mql5.com/en/articles/1998#1)
- [2\. Objectives](https://www.mql5.com/en/articles/1998#2)
- [3\. The CGrid Class](https://www.mql5.com/en/articles/1998#3)
  - [3.1. Initialization](https://www.mql5.com/en/articles/1998#3.1)
  - [3.2. Space Between Controls](https://www.mql5.com/en/articles/1998#3.2)
  - [3.3. Control Resizing](https://www.mql5.com/en/articles/1998#3.3)
- [4\. Example #1: A Simple Grid of Buttons](https://www.mql5.com/en/articles/1998#4)
- [5\. Example #2: Sliding Puzzle](https://www.mql5.com/en/articles/1998#5)
  - [5.1. Creation of the Dialog Window](https://www.mql5.com/en/articles/1998#5.1)
  - [5.2. Buttons](https://www.mql5.com/en/articles/1998#5.2)
  - [5.3. Checking for Adjacent Tiles](https://www.mql5.com/en/articles/1998#5.3)
  - [5.4. Shuffling the Tiles](https://www.mql5.com/en/articles/1998#5.4)
  - [5.5. Button Click Event](https://www.mql5.com/en/articles/1998#5.5)
  - [5.6. Checking](https://www.mql5.com/en/articles/1998#5.6)
- [6\. The CGridTk Class](https://www.mql5.com/en/articles/1998#6)
  - [6.1. Problems with the CGrid Class](https://www.mql5.com/en/articles/1998#6.1)
  - [6.2. CGridTk: An Improved CGrid](https://www.mql5.com/en/articles/1998#6.2)
    - [6.2.1. Row Span and Column Span](https://www.mql5.com/en/articles/1998#6.2.1)
    - [6.2.2. The CConstraints class](https://www.mql5.com/en/articles/1998#6.2.2)
    - [6.2.3. Default Positioning](https://www.mql5.com/en/articles/1998#6.2.3)
- [7\. Example #3: Sliding Puzzle (Improved)](https://www.mql5.com/en/articles/1998#7)
- [8\. Container Nesting](https://www.mql5.com/en/articles/1998#8)
- [9\. Advantages and Disadvantages](https://www.mql5.com/en/articles/1998#9)
- [10\. Conclusion](https://www.mql5.com/en/articles/1998#10)

### 1\. Introduction

The CGrid class is a layout manager used in the design of GUI controls for dialog windows in MetaTrader. It is one of the custom container classes that can be used in GUI design without relying on absolute positioning.

It is highly recommended to read the [article about the CBox class](https://www.mql5.com/en/articles/1867) before proceeding to the concepts discussed in this article.

### 2\. Objectives

Using the CBox class is sufficient for most simple dialog windows. However, as the number of controls increase in the dialog window, using multiple CBox containers may have the following disadvantages:

- Deeper nesting of controls.
- More container controls needed in the layout.
- More lines of code in order to get some simple things done.

Most, if not all, of these issues with the CBox class can be prevented if its controls can be placed in a grid rather than individual box containers. The objectives of this article are the following:

- Implement a class to arrange controls within a predefined grid.
- Implement an easier alternative to nested CBox containers.

And similar to the CBox class, the following objectives would also need to be satisfied:

- The code should be reusable.
- Changing one part of the interface should have minimal impact on other components.
- The positioning of components within the interface should be automatically calculated.

In this article, we aim to define a layout manager that achieves the objectives mentioned above using the CGrid class.

### 3\. The CGrid Class

The CGrid class creates a container for one or more GUI controls and presents them in a grid arrangement. An example layout of an instance of the CGrid class is shown in the following illustration:

![CGrid Layout](https://c.mql5.com/2/19/grid.png)

Figure 1. Grid Layout

Using this class can be convenient, especially if the controls to be added to the grid have identical dimensions, such as a set of buttons or edit boxes within the client area.

The example above is a grid of 4x4 cells (4 columns and 4 rows). However, we aim to develop a class that would be able to accommodate any number of rows and columns in a grid.

We will declare the CGrid class as a child class of the CBox class. With this, we would be able to easily override the virtual functions of the parent class. Furthermore, this will give us the capability to manipulate the instances of this class like the instances of CBox:

```
#include "Box.mqh"
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CGrid : public CBox
  {
protected:
   int               m_cols;
   int               m_rows;
   int               m_hgap;
   int               m_vgap;
   CSize             m_cell_size;
public:
                     CGrid();
                     CGrid(int rows,int cols,int hgap=0,int vgap=0);
                    ~CGrid();
   virtual int       Type() const {return CLASS_LAYOUT;}
   virtual bool      Init(int rows,int cols,int hgap=0,int vgap=0);
   virtual bool      Create(const long chart,const string name,const int subwin,
                            const int x1,const int y1,const int x2,const int y2);
   virtual int       Columns(){return(m_cols);}
   virtual void      Columns(int cols){m_cols=cols;}
   virtual int       Rows(){return(m_rows);}
   virtual void      Rows(int rows){m_rows=rows;}
   virtual int       HGap(){return(m_hgap);}
   virtual void      HGap(int gap){m_hgap=gap;}
   virtual int       VGap(){return(m_vgap);}
   virtual void      VGap(int gap){m_vgap=gap;}
   virtual bool      Pack();
protected:
   virtual void      CheckControlSize(CWnd *control);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CGrid::CGrid()
  {
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CGrid::CGrid(int rows,int cols,int hgap=0,int vgap=0)
  {
   Init(rows,cols,hgap,vgap);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CGrid::~CGrid()
  {
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CGrid::Init(int rows,int cols,int hgap=0,int vgap=0)
  {
   Columns(cols);
   Rows(rows);
   HGap(hgap);
   VGap(vgap);
   return(true);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CGrid::Create(const long chart,const string name,const int subwin,
                   const int x1,const int y1,const int x2,const int y2)
  {
   return(CBox::Create(chart,name,subwin,x1,y1,x2,y2));
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CGrid::Pack()
  {
   CSize size=Size();
   m_cell_size.cx = (size.cx-((m_cols+1)*m_hgap))/m_cols;
   m_cell_size.cy = (size.cy-((m_rows+1)*m_vgap))/m_rows;
   int x=Left(),y=Top();
   int cnt=0;
   for(int i=0;i<ControlsTotal();i++)
     {
      CWnd *control=Control(i);
      if(control==NULL)
         continue;
      if(control==GetPointer(m_background))
         continue;
      if(cnt==0 || Right()-(x+m_cell_size.cx)<m_cell_size.cx+m_hgap)
        {
         if(cnt==0)
            y+=m_vgap;
         else y+=m_vgap+m_cell_size.cy;
         x=Left()+m_hgap;
        }
      else x+=m_cell_size.cx+m_hgap;
      CheckControlSize(control);
      control.Move(x,y);
      if(control.Type()==CLASS_LAYOUT)
        {
         CBox *container=control;
         container.Pack();
        }
      cnt++;
     }
   return(true);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CGrid::CheckControlSize(CWnd *control)
  {
   control.Size(m_cell_size.cx,m_cell_size.cy);
  }
//+------------------------------------------------------------------+
```

### 3.1. Initialization

Similar to other containers and controls, we create the actual grid by calling the Create() method of the class. However, similar to any instance of CBox, specifying the position of the control is optional at this point. We can simply declare the width and height of the control using the _x2_ and _y2_ property. If the grid is the sole container (main container) to attach to the client area, the following code can be used (with _m\_main_ as an instance of CGrid):

```
if(!m_main.Create(chart,name+"main",subwin,0,0,CDialog::m_client_area.Width(),CDialog::m_client_area.Height()))
      return(false);
```

Right after the creation, we will need to initialize the grid by calling its Init() method. To initialize an instance of CGrid, we will need to specify the number of columns and rows by which the main client area (or a subsection of it) will be divided into, as well as the space (horizontal and vertical) between each cell on the grid. To actually initialize the grid, we need to call the Init() method in the source code. The following code will create a 4x4 grid with horizontal and vertical gaps between each cell of 2 pixels each:

```
m_main.Init(4,4,2,2);
```

The Init() method has 4 parameters:

1. number of rows;
2. number of columns;
3. horizontal gap (in pixels);
4. vertical gap (in pixels).

The horizontal and vertical gap between cells are optional parameters. By default, these values would be zero unless initialized with custom values.

### 3.2. Space Between Controls

The _hgap_ (horizontal gap) and _vgap_ (vertical gap) parameters determine the spacing between each cell on the grid. Since the grid maximizes the use of the entire client area or container, the total remaining space for controls in any given horizontal or vertical orientation is shown in the following formula:

```
total size left for controls = total area space - (gap * (number of cells+1))
```

The formula above is used within the Pack() function of the class.

### 3.3. Control Resizing

In the CGrid class, the size of each control in the grid will be resized to occupy the full size of the cell. Thus, using this layout, it is acceptable to create or initialize control elements with zero size. The control will be resized later on during the creation of the main dialog window ( [CDialog](https://www.mql5.com/en/docs/standardlibrary/controls/cdialog) or [CAppDialog](https://www.mql5.com/en/docs/standardlibrary/controls/cappdialog)) as the Pack() method for the class instance is called.

The total size left (horizontal or vertical) calculated in the formula given in the previous section will determine the x- or y-size of any particular cell within the grid. For the size of each cell, the grid will use the following formulas:

```
xsize = total size left for controls / total number of columns

ysize = total size left for controls / total number of rows
```

The actual resizing is done within the CheckControlSize() method of the class.

### 4\. Example \#1: A Simple Grid of Buttons

To illustrate a basic example of using the CGrid class, we present a simple grid of buttons. A screenshot of the GUI is shown in the following:

![A Simple Grid of Buttons](https://c.mql5.com/2/19/gridsample.png)

Figure 2. A Simple Grid of Buttons

As we can see, the dialog shown above contains a grid of 3x3 cells, with each cell containing a button. Each button is placed uniformly across the entire grid, which occupies the entire client area of the dialog window.

In order to create this grid, we need to construct an EA or indicator following the format described in the article about CBox, which is also essentially similar to the example controls in MetaTrader. That is, we declare a main source file, which contains the declaration of an instance of a custom CAppDialog window (along with other event handlers), and link it with a header file containing the actual declaration of the class being used.

For the 3x3 grid, we need to have an instance of the CGrid class as a member of the class, along with a set of 9 buttons (1 for each grid cell):

```
class CGridSampleDialog : public CAppDialog
  {
protected:
   CGrid             m_main;
   CButton           m_button1;
   CButton           m_button2;
   CButton           m_button3;
   CButton           m_button4;
   CButton           m_button5;
   CButton           m_button6;
   CButton           m_button7;
   CButton           m_button8;
   CButton           m_button9;
public:
                     CGridSampleDialog();
                    ~CGridSampleDialog();
  };
```

The next step would be to override the public virtual functions of the CAppDialog class.

```
public:
                     CGridSampleDialog();
                    ~CGridSampleDialog();
   virtual bool      Create(const long chart,const string name,const int subwin,const int x1,const int y1,const int x2,const int y2);
   virtual bool      OnEvent(const int id,const long &lparam,const double &dparam,const string &sparam);
```

```
bool CGridSampleDialog::Create(const long chart,const string name,const int subwin,const int x1,const int y1,const int x2,const int y2)
  {
   if(!CAppDialog::Create(chart,name,subwin,x1,y1,x2,y2))
      return(false);
   if(!CreateMain(chart,name,subwin))
      return(false);
   for(int i=1;i<=9;i++)
     {
      if(!CreateButton(i,chart,"button",subwin))
         return(false);
     }
   if(!m_main.Pack())
      return(false);
   if(!Add(m_main))
      return(false);
   return(true);
  }
```

```
EVENT_MAP_BEGIN(CGridSampleDialog)
EVENT_MAP_END(CAppDialog)
```

The event map is empty for this example, since we will not assign any event handling to the buttons.

The final step would be to declare the protected functions of the class, which will be the ones actually used for constructing the grid with its controls:

```
protected:
   virtual bool      CreateMain(const long chart,const string name,const int subwin);
   virtual bool      CreateButton(const int button_id,const long chart,const string name,const int subwin);
```

Using this example, we can see some of the advantages of CGrid over CBox. In order to construct a similar layout, using CBox alone would require 4 different containers. This is because CBox can only handle a single column or a single row. With CGrid, on the other hand, we have reduced the number of containers from 4 to 1, which took fewer declarations and lines of code.

```
bool CGridSampleDialog::CreateMain(const long chart,const string name,const int subwin)
  {
   if(!m_main.Create(chart,name+"main",subwin,0,0,CDialog::m_client_area.Width(),CDialog::m_client_area.Height()))
      return(false);
   m_main.Init(3,3,5,5);
   return(true);
  }
```

The CreateMain() class method is responsible for constructing the grid control itself. It works similarly when creating the control for CBox. The only difference is that CGrid requires an additional method, which is Init(). On the other hand, CBox does not need this.

The implementation for the CreateButton() class member is shown in the code snippet below:

```
bool CGridSampleDialog::CreateButton(const int button_id,const long chart,const string name,const int subwin)
  {
   CButton *button;
   switch(button_id)
     {
      case 1: button = GetPointer(m_button1); break;
      case 2: button = GetPointer(m_button2); break;
      case 3: button = GetPointer(m_button3); break;
      case 4: button = GetPointer(m_button4); break;
      case 5: button = GetPointer(m_button5); break;
      case 6: button = GetPointer(m_button6); break;
      case 7: button = GetPointer(m_button7); break;
      case 8: button = GetPointer(m_button8); break;
      case 9: button = GetPointer(m_button9); break;
      default: return(false);
     }
   if (!button.Create(chart,name+IntegerToString(button_id),subwin,0,0,100,100))
      return(false);
   if (!button.Text(name+IntegerToString(button_id)))
      return(false);
   if (!m_main.Add(button))
      return(false);
   return(true);
  }
```

Since the processes of creating the buttons are rather similar, instead of using a method for creating each button, we will use a generic function to create all the buttons. This is done by the CreateButton() class method implemented above. We will call this method within the Create() virtual class method right after we have created the dialog window and the grid. As shown in the code snippet for the Create() virtual member method, we implemented a [for](https://www.mql5.com/en/docs/basis/operators/for) loop in order to accomplish this. Since the buttons are statically declared within the class, the buttons are already created upon declaration, so there is no need to use the [new](https://www.mql5.com/en/docs/basis/operators/newoperator) operator. We simply get the [pointer](https://www.mql5.com/en/docs/constants/namedconstants/enum_pointer_type) (automatic) of each button and then call each of their Create() methods.

### 5\. Example \#2: Sliding Puzzle

Our second example involves a game called sliding puzzle. In this game, the user is given a set of numbers ranging from 1 to 15 in a 4 x 4 grid. The goal for the user is to rearrange the tiles so that the numbers are arranged in order, from left to right and top to bottom. The game is considered complete as soon as the user has sorted the number tiles in the correct order, as shown in the following screenshot:

![Sliding Puzzle](https://c.mql5.com/2/19/slidingpuzzle1__1.png)

Figure 3. Sliding Puzzle

Apart from the class methods involved in constructing a dialog window, creating this application would require the following additional features:

- method for creating the buttons;
- method for random shuffling of tiles;
- method for checking if a certain cell is beside an empty tile;
- method for checking whether or not the puzzle is already solved;
- click event method for each button on the grid.

### 5.1. Creation of the Dialog Window

We declare the class as an extension of the CAppDialog class, with its protected (or private) members, constructor and destructor:

```
class CSlidingPuzzleDialog : public CAppDialog
  {
protected:
   CGrid             m_main;
   CButton           m_button1;
   CButton           m_button2;
   CButton           m_button3;
   CButton           m_button4;
   CButton           m_button5;
   CButton           m_button6;
   CButton           m_button7;
   CButton           m_button8;
   CButton           m_button9;
   CButton           m_button10;
   CButton           m_button11;
   CButton           m_button12;
   CButton           m_button13;
   CButton           m_button14;
   CButton           m_button15;
   CButton           m_button16;
   CButton          *m_empty_cell;
public:
                     CSlidingPuzzleDialog();
                    ~CSlidingPuzzleDialog();
  };
```

The following code shows the Create() method for the class.

Declaration (under the class definition, public member functions):

```
virtual bool      Create(const long chart,const string name,const int subwin,const int x1,const int y1,const int x2,const int y2);
```

Implementation:

```
bool CSlidingPuzzleDialog::Create(const long chart,const string name,const int subwin,const int x1,const int y1,const int x2,const int y2)
  {
   if(!CAppDialog::Create(chart,name,subwin,x1,y1,x2,y2))
      return(false);
   if(!CreateMain(chart,name,subwin))
      return(false);
   for(int i=1;i<=16;i++)
     {
      if(!CreateButton(i,chart,"button",subwin))
         return(false);
     }
   m_empty_cell=GetPointer(m_button16);
   if(!m_main.Pack())
      return(false);
   if(!Add(m_main))
      return(false);
   Shuffle();
   return(true);
  }
```

From here, we can see that the dialog has the functions CreateMain(), which will be used for constructing the grid, and CreateButton() in a for loop, which is used for creating the buttons for the grid. We can also see here the Pack() method of the CGrid instance being called (for re-positioning of controls), and the attachment of the grid to the main client area, using the Add() class method. The initialization of the game is also present through the Shuffle() method.

### 5.2. Buttons

The following shows the code snippet for the CreateButton() class method:

```
bool CSlidingPuzzleDialog::CreateButton(const int button_id,const long chart,const string name,const int subwin)
  {
   CButton *button;
   switch(button_id)
     {
      case 1: button = GetPointer(m_button1); break;
      case 2: button = GetPointer(m_button2); break;
      case 3: button = GetPointer(m_button3); break;
      case 4: button = GetPointer(m_button4); break;
      case 5: button = GetPointer(m_button5); break;
      case 6: button = GetPointer(m_button6); break;
      case 7: button = GetPointer(m_button7); break;
      case 8: button = GetPointer(m_button8); break;
      case 9: button = GetPointer(m_button9); break;
      case 10: button = GetPointer(m_button10); break;
      case 11: button = GetPointer(m_button11); break;
      case 12: button = GetPointer(m_button12); break;
      case 13: button = GetPointer(m_button13); break;
      case 14: button = GetPointer(m_button14); break;
      case 15: button = GetPointer(m_button15); break;
      case 16: button = GetPointer(m_button16); break;
      default: return(false);
     }
   if(!button.Create(chart,name+IntegerToString(button_id),subwin,0,0,100,100))
      return(false);
   if(button_id<16)
     {
      if(!button.Text(IntegerToString(button_id)))
         return(false);
     }
   else if(button_id==16)
     {
      button.Hide();
     }
   if(!m_main.Add(button))
      return(false);
   return(true);
  }
```

Here we can see that the class method is similar to the CreateButton() method for the previous example. Under this method, we assign each cell an initial value (from 1-16). We also hide the 16th cell, since it would serve as an empty cell.

### 5.3. Checking for Adjacent Tiles

It is necessary to check if an adjacent tile in a given direction exists. Otherwise, the empty cell will be swapping values with a button that does not exist. The actual checking for adjacent tiles is done using the functions HasNorth(), HasSouth(), HasEast(), and HasSouth(). The following code snippet shows the HasNorth() method:

```
bool CSlidingPuzzleDialog::HasNorth(CButton *button,int id,bool shuffle=false)
  {
   if(id==1 || id==2 || id==3 || id==4)
      return(false);
   CButton *button_adj=m_main.Control(id-4);
   if(!CheckPointer(button_adj))
      return(false);
   if(!shuffle)
     {
      if(button_adj.IsVisible())
         return(false);
     }
   return(true);
  }
```

These functions check whether or not a button (or an empty cell) is allowed to move in the cardinal directions, which are also the directions where the empty cell can freely go. If a certain button is found around the center of the grid, it would be free to move in all four directions. However, if it is found on one of the the sides, then there would be some tiles that do not exist. For example, not considering empty cells, the first cell on the grid can move right or down, but it cannot move to its left or top, whereas the sixth cell can move freely in all four directions.

### 5.4. Shuffling the Tiles

The following code snippet shows the Shuffle() method of the class:

```
void CSlidingPuzzleDialog::Shuffle(void)
  {
   m_empty_cell=m_main.Control(16);
   for(int i=1;i<m_main.ControlsTotal()-1;i++)
     {
      CButton *button=m_main.Control(i);
      button.Text((string)i);
     }
   MathSrand((int)TimeLocal());
   CButton *target=NULL;
   for(int i=0;i<30;i++)
     {
      int empty_cell_id=(int)StringToInteger(StringSubstr(m_empty_cell.Name(),6));
      int random=MathRand()%4+1;
      if(random==1 && HasNorth(m_empty_cell,empty_cell_id,true))
         target= m_main.Control(empty_cell_id-4);
      else if(random==2 && HasEast(m_empty_cell,empty_cell_id,true))
         target=m_main.Control(empty_cell_id+1);
      else if(random==3 && HasSouth(m_empty_cell,empty_cell_id,true))
         target=m_main.Control(empty_cell_id+4);
      else if(random==4 && HasWest(m_empty_cell,empty_cell_id,true))
         target=m_main.Control(empty_cell_id-1);
      if(CheckPointer(target))
         Swap(target);
     }
  }
```

When shuffling the tiles, the process should involve some form of randomness. Otherwise, the tiles would always shuffle in the same order. We will use the functions [MathSrand](https://www.mql5.com/en/docs/math/mathsrand) and [MathRand](https://www.mql5.com/en/docs/math/mathrand) in order to accomplish this, and use the local time as the initial seed.

Before any shuffling should occur, we need to initialize the values of the buttons to their default values first. This prevents any event where the puzzle would become unsolvable, or perhaps too difficult to solve. We do this by reassigning the empty cell to the 16th tile, and assigning the values accordingly. We also assign the 16th cell to the empty cell pointer (class member) we have declared earlier.

At the end of the class method, the sorting of the tiles is performed. The buttons are not actually switched. Rather, their values are simply swapped, giving the illusion of movement. And as we can see, this is the easier approach. Each loop would check if there is an adjacent tile, and if the tile is an empty cell, then the values of the empty button and the button selected randomly, will be exchanged.

We also indicate a default value on how many times the swapping of tiles occurs. The default value is 30, but this value can also be changed in order to increase or decrease the difficulty. The shuffling may be more or less difficult than the difficulty setting, depending on whether or not the target button acquired a valid pointer for each iteration.

### 5.5. Button Click Event

In order to process the click events for each button, we would need to declare a click event handler. However, in order to lessen the code duplication, we will declare a class method that processes all the button click events:

```
CSlidingPuzzleDialog::OnClickButton(CButton *button)
  {
   if(IsMovable(button))
     {
      Swap(button);
      Check();
     }
  }
```

The IsMovable() function simply checks whether a certain number tile has any empty tile adjacent to it, using the functions involving cardinal directions (e.g. HasNorth(), HasSouth()). If the button has an empty tile adjacent to it, it is movable, and therefore the Swap() function is called, exchanging the value of the button with that of the empty cell. It also calls the Check() function right after each successful swap.

Then, we will create separate event handlers for each button. Here is an example of the event handler for the first button:

```
CSlidingPuzzleDialog::OnClickButton1(void)
  {
   OnClickButton(GetPointer(m_button1));
  }
```

Each of these event handlers would eventually call OnClickButton() at some point. We would also need to declare these class methods on the event map:

```
EVENT_MAP_BEGIN(CSlidingPuzzleDialog)
   ON_EVENT(ON_CLICK,m_button1,OnClickButton1)
   ON_EVENT(ON_CLICK,m_button2,OnClickButton2)
   ON_EVENT(ON_CLICK,m_button3,OnClickButton3)
   ON_EVENT(ON_CLICK,m_button4,OnClickButton4)
   ON_EVENT(ON_CLICK,m_button5,OnClickButton5)
   ON_EVENT(ON_CLICK,m_button6,OnClickButton6)
   ON_EVENT(ON_CLICK,m_button7,OnClickButton7)
   ON_EVENT(ON_CLICK,m_button8,OnClickButton8)
   ON_EVENT(ON_CLICK,m_button9,OnClickButton9)
   ON_EVENT(ON_CLICK,m_button10,OnClickButton10)
   ON_EVENT(ON_CLICK,m_button11,OnClickButton11)
   ON_EVENT(ON_CLICK,m_button12,OnClickButton12)
   ON_EVENT(ON_CLICK,m_button13,OnClickButton13)
   ON_EVENT(ON_CLICK,m_button14,OnClickButton14)
   ON_EVENT(ON_CLICK,m_button15,OnClickButton15)
   ON_EVENT(ON_CLICK,m_button16,OnClickButton16)
EVENT_MAP_END(CAppDialog)
```

Alternatively, it is possible to invoke the click event handler for each of the buttons on the event map itself, in order to prevent having to declare separate event handler class members for each button.

Finally, add the OnEvent() public member function to the class declaration:

```
virtual bool      OnEvent(const int id,const long &lparam,const double &dparam,const string &sparam);
```

### 5.6. Checking

We will need to check the order of the cells upon each button click, to see if the puzzle is already solved. This is performed by the Check() member function:

```
bool CSlidingPuzzleDialog::Check(void)
  {
   for(int i=1;i<m_main.ControlsTotal()-1;i++)
     {
      CButton *button=m_main.Control(i);
      if(CheckPointer(button))
        {
         if(button.Text()!=IntegerToString(i))
           {
            Print("status: not solved: "+button.Text()+" "+IntegerToString(i));
            return(false);
           }
        }
     }
   Print("status: solved");
   return(true);
  }
```

The checking is performed from the 2nd control up to the second-to-last control only. The first control would always be the background, which is not a button, while the final control would be the empty cell, which no longer needs to be checked.

### 6\. The CGridTk Class

### 6.1. Problems with the CGrid Class

We encounter several problems when using the CGrid class:

- Empty space — CGrid would simply place the next control on the next column in the current row, and will move on to the next row only when the current row is full.
- Custom positioning and size of controls — the layout can be useful in a number of cases, but can be rigid in some. This is because every control within the grid will need to occupy exactly one cell.

In the case of placing empty cells, we may sometimes want a particular control to be positioned far away from any of its siblings (either horizontally, vertically, or both), probably farther away than the horizontal and vertical gaps available on the grid. An example would be separating a set of buttons from another set of buttons, or positioning a button on the left side of the client area (pull-left) with another one on the other side (pull-right). We often see these types of GUI designs on the various forms we encounter on web pages.

For the first problem mentioned above, we can resolve it by creating empty controls. It can be some control with not many cosmetic components such as a button or a label. Furthermore, we can render such invisible controls by calling their Hide() method similar to what we have done for the 16th cell in the first example. And finally, we place such controls in a cell within the grid where we would like to create some breathing ground or space. This would give the illusion of space for such a cell. But in reality, the cell is occupied by an invisible control.

This solution can be handy for simple dialog windows, but in more complex dialogs, it can be inefficient and impractical. The code will tend to be longer due to the number of controls to be declared, especially if more than one empty cell is involved. Furthermore, maintaining the code could be difficult as the number of empty cells increase (e.g. an entire row or column of empty cells).

The second problem has something to do with the position and size of the controls. With respect to the positioning of the individual controls in the cell, we do not have the problem if all the controls follow the same size and distance from each other. But if they don't, then we have to implement a different approach. Most likely, the solution would be to put the asymmetrical controls out of the grid and place them somewhere else through the absolute positioning. Another alternative would be to place them on another container such as CBox or another instance of CGrid.

### 6.2. CGridTk: An Improved CGrid

The standard CGrid class can have a wide range of applications. However, its capabilities as a grid container are very limited. Based on two problems involved with the use of the standard CGrid class discussed in the previous section, we can derive a much improved class from it with the following features (on top of CGrid):

- Allows the creation of empty cells without any GUI control.
- Allows the placement of controls with a custom width and height, expressed as a multiple of the pixel size of one grid cell.

With these features, we would be able to resolve the problems presented in the previous section. Furthermore, this would give us more freedom in the actual placement and positioning of cells, akin to absolute positioning. However, unlike absolute positioning, we are using cell size as the basic unit of positioning, rather than 1 pixel. Again, we sacrifice precision for the sake of convenience when designing, it is easier to visualize the size of 1 cell in a grid than, let's say, 100 pixels on the screen.

We will rename the class to GridTk. Its code is shown below:

```
#include "Grid.mqh"
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CGridTk : public CGrid
  {
protected:
   CArrayObj         m_constraints;
public:
                     CGridTk();
                    ~CGridTk();
   bool              Grid(CWnd *control,int row,int column,int rowspan,int colspan);
   bool              Pack();
   CGridConstraints     *GetGridConstraints(CWnd *control);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CGridTk::CGridTk(void)
  {
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CGridTk::~CGridTk(void)
  {
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CGridTk::Grid(CWnd *control,int row,int column,int rowspan=1,int colspan=1)
  {
   CGridConstraints *constraints=new CGridConstraints(control,row,column,rowspan,colspan);
   if(!CheckPointer(constraints))
      return(false);
   if(!m_constraints.Add(constraints))
      return(false);
   return(Add(control));
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CGridTk::Pack()
  {
   CGrid::Pack();
   CSize size=Size();
   m_cell_size.cx = (size.cx-(m_cols+1)*m_hgap)/m_cols;
   m_cell_size.cy = (size.cy-(m_rows+1)*m_vgap)/m_rows;
   for(int i=0;i<ControlsTotal();i++)
     {
      int x=0,y=0,sizex=0,sizey=0;
      CWnd *control=Control(i);
      if(control==NULL)
         continue;
      if(control==GetPointer(m_background))
         continue;
      CGridConstraints *constraints = GetGridConstraints(control);
      if (constraints==NULL)
         continue;
      int column = constraints.Column();
      int row = constraints.Row();
      x = (column*m_cell_size.cx)+((column+1)*m_hgap);
      y = (row*m_cell_size.cy)+((row+1)*m_vgap);
      int colspan = constraints.ColSpan();
      int rowspan = constraints.RowSpan();
      control.Size(colspan*m_cell_size.cx+((colspan-1)*m_hgap),rowspan*m_cell_size.cy+((rowspan-1)*m_vgap));
      control.Move(x,y);
      if(control.Type()==CLASS_LAYOUT)
        {
         CBox *container=control;
         container.Pack();
        }
     }
   return(true);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CGridConstraints *CGridTk::GetGridConstraints(CWnd *control)
  {
   for(int i=0;i<m_constraints.Total();i++)
     {
      CGridConstraints *constraints=m_constraints.At(i);
      CWnd *ctrl=constraints.Control();
      if(ctrl==NULL)
         continue;
      if(ctrl==control)
         return(constraints);
     }
   return (NULL);
  }
```

In addition to the Add() method, we introduce a new method for adding controls to the grid, the Grid() method. When this class method is used, the control can be assigned a custom position and size, based on a multiple of the size of 1 cell.

We can see that the class has a member of the CConstraints class, which will be discussed later in this section.

### 6.2.1. Row Span and Column Span

With the row and column spans, we can now define how long or wide the control should be. This is an improvement over having a default size of one grid cell, but still less precise than absolute positioning. However, it is worth noting that the CGridTk class no longer uses the CheckControlSize() method of CBox and CGrid. Rather, it already performs the actual resizing of controls within the Pack() method itself.

### 6.2.2. The CConstraints Class

For each control, we will need to define a set of constraints that will define how each control will be positioned in the grid, what cells it should occupy, as well as how they should be resized. We can directly reposition and resize the controls themselves as soon as they are added through the use of the Grid() method of CGridTk. However, for the sake of consistency we will delay the resizing and repositioning until the Pack() method is called (similar to what is done within the CBox class). In order to do this, we need to store the constraints in the memory, which is the very purpose of the CConstraints class:

```
class CGridConstraints : public CObject
  {
protected:
   CWnd             *m_control;
   int               m_row;
   int               m_col;
   int               m_rowspan;
   int               m_colspan;
public:
                     CGridConstraints(CWnd *control,int row,int column,int rowspan=1,int colspan=1);
                    ~CGridConstraints();
   CWnd             *Control(){return(m_control);}
   int               Row(){return(m_row);}
   int               Column(){return(m_col);}
   int               RowSpan(){return(m_rowspan);}
   int               ColSpan(){return(m_colspan);}
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CGridConstraints::CGridConstraints(CWnd *control,int row,int column,int rowspan=1,int colspan=1)
  {
   m_control = control;
   m_row = MathMax(0,row);
   m_col = MathMax(0,column);
   m_rowspan = MathMax(1,rowspan);
   m_colspan = MathMax(1,colspan);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CGridConstraints::~CGridConstraints()
  {
  }
```

Through the class object's constructor alone, we can conclude that the CConstraints class stores the rows, columns, rowspan, and colspan for each control. But this is only possible whenever the Grid() method is called, as can be seen on the implementation for the CGridTk class. Furthermore, the said class only stores information. How the information is used is implemented under CGridTk.

### 6.3.3. Default Positioning

If a certain control is not added to the grid using the Grid() method, the default positioning would be used. Such a control was added to the grid using the Add() method, and this means that the grid has no constraints (no CGridConstraints object stored in the grid class instance). Thus, updated methods in CGridTk would not be able to do anything on those controls as far as positioning or resizing are concerned. The placement method would be similar to the CGrid class method, as a fallback or default method of positioning. That is, such controls would be stacked like bricks of a wall, but starting from the top left portion of the client area, as shown in the first example.

### 7\. Example \#3: Sliding Puzzle (Improved)

In order to further improve the sliding puzzle, we need to make some changes on the second example:

1. Create a "New Game" button, so the expert advisor no longer needs to be restarted to begin a new game.
2. Create a control on the dialog window showing the status of the game to eliminate the need to open the journal tab of the terminal window.
3. Implement a different size for the new controls.
4. Make some cosmetic changes, such as coloring of the tiles and showing all tiles on the grid (optional).


The improved sliding puzzle is shown in the following screenshot:

![Sliding Puzzle (Improved)](https://c.mql5.com/2/19/slidingpuzzle2__1.png)

Figure 4. Sliding Puzzle (improved)

As seen on the screenshot, we have added new components to the dialog window. There is a button that allows us to create a new game (reshuffle) as well as a text box showing the current status of the game. Now, we would not want these buttons to be resized to a size of 1 grid cell, just like the other 16 buttons. That might cause some confusion for the users, as they may find it hard to see the descriptions or text for these controls.

In order to construct this dialog, we will need to extend the class in the second example, or copy the said class and then modify it. Here, we will choose to simply copy rather than extend the class in the second example.

With this new dialog, two additional controls were added. We will need to declare member functions that would perform the creation of these controls, namely, CreateButtonNew() and CreateLabel(). First, we will need to declare them as members of the class:

```
protected:
   //protected member methods start

   virtual bool      CreateButtonNew(const long chart,const string name,const int subwin);
   virtual bool      CreateLabel(const long chart,const string name,const int subwin);

   //more protected member methods below..
```

The actual implementation of the member functions is shown below:

```
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CSlidingPuzzleDialog::CreateButtonNew(const long chart,const string name,const int subwin)
  {
   if(!m_button_new.Create(chart,name+"buttonnew",m_subwin,0,0,101,101))
      return(false);
   m_button_new.Text("New");
   m_button_new.ColorBackground(clrYellow);
   if(!m_main.Grid(GetPointer(m_button_new),4,0,1,2))
      return(false);
   return(true);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CSlidingPuzzleDialog::CreateLabel(const long chart,const string name,const int subwin)
  {
   if(!m_label.Create(chart,name+"labelnew",m_subwin,0,0,102,102))
      return(false);
   m_label.Text("click new");
   m_label.ReadOnly(true);
   m_label.TextAlign(ALIGN_CENTER);
   if(!m_main.Grid(GetPointer(m_label),4,2,1,2))
      return(false);
   return(true);
  }
```

We would also need to slightly modify some functions. Since new controls are added to the grid, member functions such as Check(), HasNorth(), HasSouth(), HasWest(), and HasEast() will need to be modified. This is to make sure that the actual tiles do not switch values with the wrong control. First, we will give the number tiles with the prefix 'block' (as an argument on CreateButton()), and then use this prefix in order to identify whether or not the selected control is actually a number tile. The following code shows the updated member function, Check():

```
bool CSlidingPuzzleDialog::Check(void)
  {
   for(int i=0;i<m_main.ControlsTotal();i++)
     {
      CWnd *control=m_main.Control(i);
      if(StringFind(control.Name(),"block")>=0)
        {
         CButton *button=control;
         if(CheckPointer(button))
           {
            if(button.Text()!=IntegerToString(i))
              {
               m_label.Text("not solved");
               return(false);
              }
           }
        }
     }
   m_label.Text("solved");
   m_solved=true;
   return(true);
  }
```

Here, we use the [StringFind](https://www.mql5.com/en/docs/strings/stringfind) function to make sure that the selected control is indeed a button, and that it is a number tile. This is necessary, otherwise, we will receive errors such as "incorrect casting of pointers" when we assign the control to an instance of CButton, which is done in one of the lines of code that follow. In this code, we also see that rather than using the Print function to display the status in the terminal window, we simply edit the text on the CEdit control.

### 8\. Container Nesting

It is possible to place a grid within another container such as a box container or a larger grid. When placed inside a CBox container, the entire grid would follow the layout and alignment of its parent container. However, just like any controls or containers placed inside an instance of CBox, the grid should be designated a preferred height and width. On the other hand, when placed inside another grid, the size of the grid will be automatically calculated.

### 9\. Advantages and Disadvantages

Advantages:

- Can potentially reduce the number of containers needed for the dialog window, especially if identical controls are present.
- More manageable and maintainable than CBox.
- More convenient to use than absolute positioning.

Disadvantages:

- Less precise than absolute positioning.
- The alignment may be a little off on the right and bottom side if the size of the client area is not in proportion to the client area. This can occur when the size of the client area minus the space for each cell yields a whole number (not integer) when divided by the number of cells or columns. As a pixel cannot be further divided, any excess pixels or remainders would accumulate at these sides, resulting in a slightly uneven look. However, this can be easily resolved by resizing the main dialog window.

### 10\. Conclusion

In this article, we have considered the possibility of using a grid layout in the construction and design of graphical panels. This additional layout class provides an additional tool for the easier construction of GUI controls in MetaTrader. In some cases, we have seen the advantages of using this layout class to the standard box layout.

We have presented two classes for creating a grid: the CGrid and the the CGridTk class. The CGrid class is an auxilliary control that acts as a container for essential controls in a GUI panel. It adds essential controls as its child components and rearranges them into an organized grid. The CGridTk class is an extension of the CGrid class, and provides more features for custom control positioning and resizing. These classes can serve as fundamental building blocks for the easier creation of graphical controls in MetaTrader.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/1998.zip "Download all attachments in the single ZIP archive")

[Grid.zip](https://www.mql5.com/en/articles/download/1998/grid.zip "Download Grid.zip")(12.35 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/64024)**
(4)


![Amir Yacoby](https://c.mql5.com/avatar/avatar_na2.png)

**[Amir Yacoby](https://www.mql5.com/en/users/amir_avatar)**
\|
6 Oct 2015 at 02:39

Great work as usual, Enrico.

I tried but couldn't make it, to place two [CEdit](https://www.mql5.com/en/docs/standardlibrary/controls/cedit "Standard library: Class CEdit") fields one just by anoter without any spacing (fixed positioning) - but it seems the class only auto positions by the layout style.

Can it be done? For instance, I want a field name and the value like this:

**Total Orders:** 3

using two CEdit's but placing them one by the other?

Thanks

![Enrico Lambino](https://c.mql5.com/avatar/2014/10/54465D5F-0757.jpg)

**[Enrico Lambino](https://www.mql5.com/en/users/iceron)**
\|
10 Oct 2015 at 07:51

**Amir Yacoby:**

Great work as usual, Enrico.

I tried but couldn't make it, to place two CEdit fields one just by anoter without any spacing (fixed positioning) - but it seems the class only auto positions by the layout style.

Can it be done? For instance, I want a field name and the value like this:

**Total Orders:** 3

using two CEdit's but placing them one by the other?

Thanks

You're welcome.

The grid only accepts a single component per cell. If you are to place more than one [control](https://www.mql5.com/en/articles/310 "Article: Custom Graphic Controls Part 1. Creating a simple control") on any given cell, you should nest them inside CBox or CGrid.

Another option would be to extend CGrid(tk) or CBox so that you can directly specify which controls should use fixed positioning, and which ones should follow layout styles.

![Vladimir Karputov](https://c.mql5.com/avatar/2024/2/65d8b5a2-f9d9.jpg)

**[Vladimir Karputov](https://www.mql5.com/en/users/barabashkakvn)**
\|
30 Mar 2016 at 10:59

When compiling the "Grid Sample.mq5" I get the error:

```
'm_client_area' - private member access error   GridSample.mqh  78      60
'm_client_area' - private member access error   GridSample.mqh  78      91
```

[![Errors](https://c.mql5.com/3/92/1toin0upzx.png)](https://c.mql5.com/3/92/0l9g6sx32k.png "https://c.mql5.com/3/92/0l9g6sx32k.png")

![Enrico Lambino](https://c.mql5.com/avatar/2014/10/54465D5F-0757.jpg)

**[Enrico Lambino](https://www.mql5.com/en/users/iceron)**
\|
30 Mar 2016 at 17:42

**Karputov Vladimir:**

When compiling the "Grid Sample.mq5" I get the error:

I guess the language was updated. Before it was possible to call the superclass that way. But now, I see, it is now possible to call the functions [ClientAreaWidth](https://www.mql5.com/en/docs/standardlibrary/controls/cdialog/cdialogclientareawidth) and [ClientAreaHeight](https://www.mql5.com/en/docs/standardlibrary/controls/cdialog/cdialogclientareaheight) for instances of CDialog and its heirs. The code:

```
m_main.Create(chart,name+"main",subwin,0,0,ClientAreaWidth(),ClientAreaHeight())
```

would be the equivalent statement.

![An Introduction to Fuzzy Logic](https://c.mql5.com/2/19/avatar__4.png)[An Introduction to Fuzzy Logic](https://www.mql5.com/en/articles/1991)

Fuzzy logic expands our boundaries of mathematical logic and set theory. This article reveals the basic principles of fuzzy logic as well as describes two fuzzy inference systems using Mamdani-type and Sugeno-type models. The examples provided will describe implementation of fuzzy models based on these two systems using the FuzzyNet library for MQL5.

![Price Action. Automating the Inside Bar Trading Strategy](https://c.mql5.com/2/19/PA.png)[Price Action. Automating the Inside Bar Trading Strategy](https://www.mql5.com/en/articles/1771)

The article describes the development of a MetaTrader 4 Expert Advisor based on the Inside Bar strategy, including Inside Bar detection principles, as well as pending and stop order setting rules. Test and optimization results are provided as well.

![How to Develop a Profitable Trading Strategy](https://c.mql5.com/2/14/289_2.png)[How to Develop a Profitable Trading Strategy](https://www.mql5.com/en/articles/1447)

This article provides an answer to the question: "Is it possible to formulate an automated trading strategy based on history data with neural networks?".

![Managing the MetaTrader Terminal via DLL](https://c.mql5.com/2/19/MetaTrader-dll.png)[Managing the MetaTrader Terminal via DLL](https://www.mql5.com/en/articles/1903)

The article deals with managing MetaTrader user interface elements via an auxiliary DLL library using the example of changing push notification delivery settings. The library source code and the sample script are attached to the article.

[![](https://www.mql5.com/ff/si/d9hnbkyp2d47h07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fsignals%2Fmt5%2Fpage1%3Fpreset%3D2%26utm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dmax.profit.signals%26utm_content%3Dsubscribe.signal%26utm_campaign%3D0622.MQL5.com.Internal&a=hgyovyikvykcdukcncnktswvlctghemf&s=545653d14172edfb3c9c02ca8e948778c29f9c1b70be9a587e8d4b040fb23539&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=gxhvoyvtbdiloklndfgindkulcxvgczn&ssn=1769252805404639884&ssn_dr=0&ssn_sr=0&fv_date=1769252805&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F1998&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Using%20Layouts%20and%20Containers%20for%20GUI%20Controls%3A%20The%20CGrid%20Class%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176925280534096536&fz_uniq=5083343203916061124&sv=2552)

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