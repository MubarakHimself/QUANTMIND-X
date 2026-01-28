---
title: Expert Advisor featuring GUI: Creating the panel (part I)
url: https://www.mql5.com/en/articles/4715
categories: Expert Advisors
relevance_score: 6
scraped_at: 2026-01-23T17:30:24.641639
---

[![](https://www.mql5.com/ff/sh/9nb0c8df2rmwfn89z2/01.png) MetaTrader VPS vs regular cloud hosting services8 reasons why our solution is the best option for automated tradingRead](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/450486&a=dgmsfszgoedimaicrqqmagvqzpwuxkur&s=c59e3617ccf44fd54d4c50a03b44fd689ff7507b8fe4990c83772cc5419e627d&uid=&ref=https://www.mql5.com/en/articles/4715&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=6412191147402131455)

MetaTrader 5 / Examples


### Contents

- [Introduction](https://www.mql5.com/en/articles/4715#para1)
- [GUI elements](https://www.mql5.com/en/articles/4715#para3)
- [Assembling the GUI](https://www.mql5.com/en/articles/4715#para4)

  - [Form for controls](https://www.mql5.com/en/articles/4715#para5)
  - [Status bar](https://www.mql5.com/en/articles/4715#para6)
  - [Group of tabs](https://www.mql5.com/en/articles/4715#para7)
  - [Input field](https://www.mql5.com/en/articles/4715#para8)
  - [Button](https://www.mql5.com/en/articles/4715#para9)
  - [Combo box with a drop-down list](https://www.mql5.com/en/articles/4715#para10)
  - [Checkbox](https://www.mql5.com/en/articles/4715#para11)
  - [Table](https://www.mql5.com/en/articles/4715#para12)
  - [Standard chart](https://www.mql5.com/en/articles/4715#para13)
  - [Progress bar](https://www.mql5.com/en/articles/4715#para14)

- [EasyAndFast library updates](https://www.mql5.com/en/articles/4715#para2)
- [Conclusion](https://www.mql5.com/en/articles/4715#para21)

### Introduction

Despite the active development of algorithmic trading, many traders still prefer manual trading. However, it is hardly possible to completely avoid the automation of routine operations.

The article shows the development of a multi-symbol signal Expert Advisor for manual trading. As an example, let's examine signals of **Stochastic** from the terminal standard delivery. You can use this code to develop your own EAs with graphical interface (GUI): any other indicator can be included into it, as well as results of certain calculations can be used to make decisions.

For those who execute other traders' orders, the article may be useful as an example of a technical task to demonstrate to customers. This example may save your time when preparing a comprehensible requirements specification for developing a program that features GUI.

Here are the issues to be discussed in detail in this article.

- Creating a GUI.
- Getting a list of symbols with specified properties.
- Trading operation controls.
- Fast switching of symbols and timeframes on charts without the EA re-initialization.
- Managing chart properties via the user interface.
- Receiving indicator signals from multiple symbols with color indication.
- Working with open positions.
- [EasyAndFast](https://www.mql5.com/en/code/19703) library updates.

The article is to be published in two parts. In this article, we consider developing the panel, while the next one describes filling it with functions.

### GUI elements

We will start developing the EA with the construction of a GUI enabling users to interact with the program and allowing data visualization. It is possible to create a GUI using the capabilities of the standard library, but in my example, it is implemented based on the [EasyAndFast](https://www.mql5.com/en/code/19703) library. Its rich features make it possible to focus on the functionality of the program itself without being distracted by refining its graphical part.

First, let's outline the general GUI structure. The diagram below shows the GUI window consisting of two tabs. The lists display the functions to be placed on them. This is a simplified example. A customer and a developer may elaborate on it in more detail during the discussion.

![Fig. 1. General GUI view with notes](https://c.mql5.com/2/33/image__2.png)

Fig. 1. General GUI view with notes

There may be a lot of GUI controls. Therefore, let's list them in a hierarchical form at first:

- Form for controls
  - Status bar for displaying additional summary information
  - Group of tabs:

    - **Trade**:

      - Input field with a checkbox for sorting out the symbol list
      - Request button to start forming the list of symbols
      - Button for SELL
      - Button for BUY
      - Trade volume input field
      - Button for closing all open positions
      - Input field for setting a sell signal level
      - Input field for setting a buy signal level
      - Table of symbols for trading
      - Chart for visualizing symbol data. For more convenience, let managing some chart properties using the following group of elements:
        - Combo box with a drop-down list for selecting timeframe switching
        - Checkbox for enabling/disabling the time scale
        - Checkbox for enabling/disabling the price scale
        - Scale management input field
        - Button for enabling an indent
        - Checkbox for displaying the indicator

    - **Positions**:
      - Table of positions

  - Indicator for replaying frames

In the main program class ( **CProgram**), declare methods and class instances of the elements listed above. The code of methods for creating elements is provided in a separate file and included into the file with an MQL program class:

```
//+------------------------------------------------------------------+
//| Application development class                                    |
//+------------------------------------------------------------------+
class CProgram : public CWndEvents
  {
private:
   //--- Window
   CWindow           m_window1;
   //--- Status bar
   CStatusBar        m_status_bar;
   //--- Tabs
   CTabs             m_tabs1;
   //--- Input fields
   CTextEdit         m_symb_filter;
   CTextEdit         m_lot;
   CTextEdit         m_up_level;
   CTextEdit         m_down_level;
   CTextEdit         m_chart_scale;
   //--- Buttons
   CButton           m_request;
   CButton           m_chart_shift;
   CButton           m_buy;
   CButton           m_sell;
   CButton           m_close_all;
   //--- Combo boxes
   CComboBox         m_timeframes;
   //--- Checkboxes
   CCheckBox         m_date_scale;
   CCheckBox         m_price_scale;
   CCheckBox         m_show_indicator;
   //--- Tables
   CTable            m_table_positions;
   CTable            m_table_symb;
   //--- Standard chart
   CStandardChart    m_sub_chart1;
   //--- Progress bar
   CProgressBar      m_progress_bar;
   //---
public:
   //--- Create the GUI
   bool              CreateGUI(void);
   //---
private:
   //--- Form
   bool              CreateWindow(const string text);
   //--- Status bar
   bool              CreateStatusBar(const int x_gap,const int y_gap);
   //--- Tabs
   bool              CreateTabs1(const int x_gap,const int y_gap);
   //--- Input fields
   bool              CreateSymbFilter(const int x_gap,const int y_gap,const string text);
   bool              CreateLot(const int x_gap,const int y_gap,const string text);
   bool              CreateUpLevel(const int x_gap,const int y_gap,const string text);
   bool              CreateDownLevel(const int x_gap,const int y_gap,const string text);
   bool              CreateChartScale(const int x_gap,const int y_gap,const string text);
   //--- Buttons
   bool              CreateRequest(const int x_gap,const int y_gap,const string text);
   bool              CreateChartShift(const int x_gap,const int y_gap,const string text);
   bool              CreateBuy(const int x_gap,const int y_gap,const string text);
   bool              CreateSell(const int x_gap,const int y_gap,const string text);
   bool              CreateCloseAll(const int x_gap,const int y_gap,const string text);
   //--- Combo box
   bool              CreateComboBoxTF(const int x_gap,const int y_gap,const string text);
   //--- Checkboxes
   bool              CreateDateScale(const int x_gap,const int y_gap,const string text);
   bool              CreatePriceScale(const int x_gap,const int y_gap,const string text);
   bool              CreateShowIndicator(const int x_gap,const int y_gap,const string text);
   //--- Tables
   bool              CreatePositionsTable(const int x_gap,const int y_gap);
   bool              CreateSymbolsTable(const int x_gap,const int y_gap);
   //--- Standard chart
   bool              CreateSubChart1(const int x_gap,const int y_gap);
   //--- Progress bar
   bool              CreateProgressBar(const int x_gap,const int y_gap,const string text);
  };
//+------------------------------------------------------------------+
//| Methods for creating controls                                    |
//+------------------------------------------------------------------+
#include "CreateGUI.mqh"
//+------------------------------------------------------------------+
```

Next, let's focus on assembling the GUI, methods for creating its elements and their properties.

### Assembling the GUI

We will use the elements of ten types in the GUI of the developed application.

- Form for controls ( **CWindow**)
- Status bar ( **CStatusBar**)
- Group of tabs ( **CTabs**)
- Input field ( **CTextEdit**)
- Button ( **CButton**)
- Combo box with a drop-down list ( **CComboBox**)
- Checkbox ( **CCheckBox**)
- Table ( **CTable**)
- Standard chart ( **CStandardChart**)
- Progress bar ( **CProgressBar**)

We will need more than one element in some categories from this list, so we will consider only one element per each group. Let's consider the methods of their creation in the same sequence.

### Form for controls

Below is the code of the method for creating a form all other elements are to be located on. First, we need to add the form into the list of the program GUI elements. To do this, call the **CWndContainer::AddWindow**() method by passing the **CWindow** type element object to it. Then, set its properties before creating the form. We set the following properties (in the same order as in the listing below):

- Form size (width and height)
- Header font size
- Form moving mode (within the chart)
- Mode of manually re-sizing the form (by dragging the borders)
- Form buttons. Visibility of each button is customizable. In this case, the following ones are used:

  - Closing the form. When clicking the button, the program closure confirmation window appears in the program main form.
  - Maximizing/minimizing the form.
  - Element tooltips. This button also has two states. If enabled, specified tooltips are displayed in the controls.
  - Expanding the form for the entire chart area. This can be undone by clicking the button again.

- A tooltip can be set for each button.

After setting the properties, call the form creation method — **CWindow::CreateWindow**() and pass to it:

- chart ID,

- chart subwindow index,

- header text,
- form initial coordinates.

```
//+------------------------------------------------------------------+
//| Create the controls form                                         |
//+------------------------------------------------------------------+
bool CProgram::CreateWindow(const string caption_text)
  {
//--- Add the window pointer to the windows array
   CWndContainer::AddWindow(m_window1);
//--- Properties
   m_window1.XSize(750);
   m_window1.YSize(450);
   m_window1.FontSize(9);
   m_window1.IsMovable(true);
   m_window1.ResizeMode(true);
   m_window1.CloseButtonIsUsed(true);
   m_window1.CollapseButtonIsUsed(true);
   m_window1.TooltipsButtonIsUsed(true);
   m_window1.FullscreenButtonIsUsed(true);
//--- Set tooltips
   m_window1.GetCloseButtonPointer().Tooltip("Close");
   m_window1.GetTooltipButtonPointer().Tooltip("Tooltips");
   m_window1.GetFullscreenButtonPointer().Tooltip("Fullscreen");
   m_window1.GetCollapseButtonPointer().Tooltip("Collapse/Expand");
//--- Create the form
   if(!m_window1.CreateWindow(m_chart_id,m_subwin,caption_text,1,1))
      return(false);
//---
   return(true);
  }
```

It is recommended to compile the program and check the result each time you add a new GUI element:

![Fig. 2. Form for controls](https://c.mql5.com/2/32/013.png)

Fig. 2. Form for controls

Screenshots of all the intermediate results are shown below.

### Status bar

The code of the method for creating a status bar starts with specifying the main element, which is used to calculate the elements bound to it and align their size. This saves time when developing an application: a whole group of related elements can be moved by changing the coordinates only of the main one. To bind the element, its pointer is passed to the **CElement::MainPointer**() method. In this example, we bind the status bar to the form, therefore, the form object is passed to the method.

Then, set the status bar properties. It is to contain three sections displaying information for users.

- In order not to specify dimensions relative to the form, make sure the width changes automatically when the form width is changed.
- The indent of the element's right edge is 1 pixel.
- Bind the status bar to the bottom of the form, so that it is automatically adjusted by the bottom edge when the form height changes.
- Then, when adding points, set the width for each of them.

After the properties are set, create the element. Now, it is ready for work, and we can change the text in its sections during the runtime. In our example, set the text "For Help, press F1" in the first section.

At the end of the method, make sure to save the pointer of the created element in the general list of the GUI elements. To do this, call the **CWndContainer::AddToElementsArray**() method and pass the form index and element object to it. Since we have only a single form, its index will be 0.

```
//+------------------------------------------------------------------+
//| Create the status bar                                            |
//+------------------------------------------------------------------+
bool CProgram::CreateStatusBar(const int x_gap,const int y_gap)
  {
#define STATUS_LABELS_TOTAL 3
//--- Save the pointer to the window
   m_status_bar.MainPointer(m_window1);
//--- Properties
   m_status_bar.AutoXResizeMode(true);
   m_status_bar.AutoXResizeRightOffset(1);
   m_status_bar.AnchorBottomWindowSide(true);
//--- Set the number of parts and their properties
   int width[STATUS_LABELS_TOTAL]={0,200,110};
   for(int i=0; i<STATUS_LABELS_TOTAL; i++)
      m_status_bar.AddItem(width[i]);
//--- Create a control
   if(!m_status_bar.CreateStatusBar(x_gap,y_gap))
      return(false);
//--- Set a text in the status bar sections
   m_status_bar.SetValue(0,"For Help, press F1");
//--- Add the object to the general array of object groups
   CWndContainer::AddToElementsArray(0,m_status_bar);
   return(true);
  }
```

The remaining elements of the [EasyAndFast](https://www.mql5.com/en/code/19703) library are created based on the same principle. Therefore, we will only consider the customizable properties to be used in our EA.

![Fig. 3. Adding the status bar](https://c.mql5.com/2/32/014.png)

Fig. 3. Adding the status bar

### Group of tabs

Let's set the following element properties in the method for creating tab groups:

- Center the texts in the tab.
- Place the tabs in the upper part of the working area.
- The size is automatically adjusted to the main element (form) area. In this case, it is not necessary to specify the size of the tab group area.
- Set the indents of the right and bottom edges of the main element. If the form changes its size, these indents are left intact.
- When adding the following tabs, the tab name and width are also transferred to the method.

Below is the method code:

```
//+------------------------------------------------------------------+
//| Create tab group 1                                               |
//+------------------------------------------------------------------+
bool CProgram::CreateTabs1(const int x_gap,const int y_gap)
  {
#define TABS1_TOTAL 2
//--- Save the pointer to the main element
   m_tabs1.MainPointer(m_window1);
//--- Properties
   m_tabs1.IsCenterText(true);
   m_tabs1.PositionMode(TABS_TOP);
   m_tabs1.AutoXResizeMode(true);
   m_tabs1.AutoYResizeMode(true);
   m_tabs1.AutoXResizeRightOffset(3);
   m_tabs1.AutoYResizeBottomOffset(25);
//--- Add tabs with specified properties
   string tabs_names[TABS1_TOTAL]={"Trade","Positions"};
   for(int i=0; i<TABS1_TOTAL; i++)
      m_tabs1.AddTab(tabs_names[i],100);
//--- Create the control
   if(!m_tabs1.CreateTabs(x_gap,y_gap))
      return(false);
//--- Add the object to the general array of object groups
   CWndContainer::AddToElementsArray(0,m_tabs1);
   return(true);
  }
```

![Fig. 4. Adding the tab group](https://c.mql5.com/2/32/015.png)

Fig. 4. Adding the tab group

### Input field

For example, let's consider an input field, in which a user can specify currencies and/or currency pairs to form a list of symbols in a table. Its main element is a group of tabs. Here we need to specify the tab the input field should be displayed on. To do this, call the **CTabs::AddToElementsArray**() method and pass the tab index and the attached element object to it.

Now, let's consider the properties for this input field.

- By default, "USD" is entered in the input field: the program will collect symbols having USD in the table. Currencies and/or symbols in this input field are comma-separated. Below, I will show the method of forming a list of symbols by comma-separated lines in this input field.
- The input field will feature a checkbox. After disabling the checkbox, a text in the input field is ignored. All detected currency pairs are included into the symbol list.
- The input field width is equal to the entire width of the main element and is adjusted when the tabs area width changes.
- The **Request** button is placed to the right of the input field. While the program is running, you can specify other currencies and/or symbols in the input field. Click this button to generate the list. Since the input field width is to be changed automatically, while the **Request** button should always be located to the right of it, make sure that the right side of the input field has an indent from the right edge of the main element.

The element of **CTextEdit** type consists of several other elements. Therefore, if you need to change their properties, you can get pointers to them. We had to change some properties of the text input field ( **CTextBox**). Let's consider them in the same order as implemented in the code listing below.

- Input field indent from the left edge of the main element ( **CTextEdit**).
- Automatic change of the width relative to the main element.
- When activating the input field (by left-clicking the input field), the text is highlighted fully automatically for a quick replacement.
- If there is no text in the input field at all, the following prompt is displayed: "Example: EURUSD, GBP, NOK".

The checkbox of the input field is enabled by default. To do this, activate it immediately after creating the element.

```
//+------------------------------------------------------------------+
//| Create a checkbox with the "Symbols filter" input field          |
//+------------------------------------------------------------------+
bool CProgram::CreateSymbolsFilter(const int x_gap,const int y_gap,const string text)
  {
//--- Save the pointer to the main element
   m_symb_filter.MainPointer(m_tabs1);
//--- Reserve for the tab
   m_tabs1.AddToElementsArray(0,m_symb_filter);
//--- Properties
   m_symb_filter.SetValue("USD"); // "EUR,USD" "EURUSD,GBPUSD" "EURUSD,GBPUSD,AUDUSD,NZDUSD,USDCHF"
   m_symb_filter.CheckBoxMode(true);
   m_symb_filter.AutoXResizeMode(true);
   m_symb_filter.AutoXResizeRightOffset(90);
   m_symb_filter.GetTextBoxPointer().XGap(100);
   m_symb_filter.GetTextBoxPointer().AutoXResizeMode(true);
   m_symb_filter.GetTextBoxPointer().AutoSelectionMode(true);
   m_symb_filter.GetTextBoxPointer().DefaultText("Example: EURUSD,GBP,NOK");
//--- Create a control
   if(!m_symb_filter.CreateTextEdit(text,x_gap,y_gap))
      return(false);
//--- Enable the check box
   m_symb_filter.IsPressed(true);
//--- Add the object to the common array of object groups
   CWndContainer::AddToElementsArray(0,m_symb_filter);
   return(true);
  }
```

In addition to the text input field, there will be numeric ones in the GUI. For example, **Lot** input field (volume for opening positions). Other properties of this type should be specified for input fields.

- Total width of the element.
- Maximum value to enter.
- Minimum value to enter.
- Step used when switching using the increment and decrement buttons.
- Number of digits after the decimal point.
- In order for the input field to become numeric, we should specify this using the **CTextEdit::SpinEditMode**() method.
- Default value after downloading the program to the terminal chart.
- Input field width.
- Automatic text highlighting in the input field when clicking on it.
- Anchoring the input field to the right edge of the element.

Here is the code for the method:

```
//+------------------------------------------------------------------+
//| Create the "Lot" input field                                     |
//+------------------------------------------------------------------+
bool CProgram::CreateLot(const int x_gap,const int y_gap,const string text)
  {
//--- Save the pointer to the main element
   m_lot.MainPointer(m_tabs1);
//--- Reserve for the tab
   m_tabs1.AddToElementsArray(0,m_lot);
//--- Properties
   m_lot.XSize(80);
   m_lot.MaxValue(1000);
   m_lot.MinValue(0.01);
   m_lot.StepValue(0.01);
   m_lot.SetDigits(2);
   m_lot.SpinEditMode(true);
   m_lot.SetValue((string)0.1);
   m_lot.GetTextBoxPointer().XSize(50);
   m_lot.GetTextBoxPointer().AutoSelectionMode(true);
   m_lot.GetTextBoxPointer().AnchorRightWindowSide(true);
//--- Create a control
   if(!m_lot.CreateTextEdit(text,x_gap,y_gap))
      return(false);
//--- Add the object to the common array of object groups
   CWndContainer::AddToElementsArray(0,m_lot);
   return(true);
  }
```

![Fig. 5. Adding the input fields](https://c.mql5.com/2/32/016.png)

Fig. 5. Adding the input fields

The image does not look very logical, but when you add other elements, everything turns out to be in place.

### Button

Let's add a few buttons to the EA's GUI. We will consider the one with the most properties: the button for opening SELL positions.

- Button width.
- The button text is accurately centered both vertically and horizontally.
- Button background color.
- Background color when hovering the cursor.
- Background color when left-clicking.
- Button text color.
- Text color when hovering the cursor.
- Text color when left-clicking.
- Button frame color.
- Frame color when hovering the cursor.
- Frame color when left-clicking.

The same properties are changed for the BUY button except for specified background colors.

```
//+------------------------------------------------------------------+
//| Create the 'Sell' button                                         |
//+------------------------------------------------------------------+
bool CProgram::CreateSell(const int x_gap,const int y_gap,const string text)
  {
//--- Save the pointer to the main element
   m_sell.MainPointer(m_tabs1);
//--- Reserve for the tab
   m_tabs1.AddToElementsArray(0,m_sell);
//--- Properties
   m_sell.XSize(80);
   m_sell.IsCenterText(true);
   m_sell.BackColor(C'255,51,51');
   m_sell.BackColorHover(C'255,100,100');
   m_sell.BackColorPressed(C'195,0,0');
   m_sell.LabelColor(clrWhite);
   m_sell.LabelColorHover(clrWhite);
   m_sell.LabelColorPressed(clrWhite);
   m_sell.BorderColor(clrBlack);
   m_sell.BorderColorHover(clrBlack);
   m_sell.BorderColorPressed(clrBlack);
//--- Create a control
   if(!m_sell.CreateButton(text,x_gap,y_gap))
      return(false);
//--- Add the element pointer to the database
   CWndContainer::AddToElementsArray(0,m_sell);
   return(true);
  }
```

![Fig. 6. Adding the buttons](https://c.mql5.com/2/32/017.png)

Fig. 6. Adding the buttons

### Combo box with a drop-down list

To change a timeframe, let's make a combo box with a drop-down list. Define the properties for its configuration.

- Total width of the element.
- Number of list items (in our case, there are 21 of them corresponding to the number of timeframes in the terminal).
- The element is to be bound to the right part of the tabs area.
- Combo box button width.
- The button is bound to the right part of the element.

Values are assigned to each list item, and some properties are set for the list via the pointer after that.

- Highlighting items when hovering the mouse cursor.
- Highlighted item. In our case, this is the point by index 18 (D1 timeframe).

Below is the code of the method for creating the combo box:

```
//+------------------------------------------------------------------+
//| Create a combo box for creating timeframes                       |
//+------------------------------------------------------------------+
bool CProgram::CreateComboBoxTF(const int x_gap,const int y_gap,const string text)
  {
//--- Total amount of list items
#define ITEMS_TOTAL2 21
//--- Pass the panel object
   m_timeframes.MainPointer(m_tabs1);
//--- Anchor to the tab
   m_tabs1.AddToElementsArray(0,m_timeframes);
//--- Properties
   m_timeframes.XSize(115);
   m_timeframes.ItemsTotal(ITEMS_TOTAL2);
   m_timeframes.AnchorRightWindowSide(true);
   m_timeframes.GetButtonPointer().XSize(50);
   m_timeframes.GetButtonPointer().AnchorRightWindowSide(true);
//--- Save item values to the combo box list
   string items_text[ITEMS_TOTAL2]={"M1","M2","M3","M4","M5","M6","M10","M12","M15","M20","M30","H1","H2","H3","H4","H6","H8","H12","D1","W1","MN"};
   for(int i=0; i<ITEMS_TOTAL2; i++)
      m_timeframes.SetValue(i,items_text[i]);
//--- Get the list pointer
   CListView *lv=m_timeframes.GetListViewPointer();
//--- Set the list properties
   lv.LightsHover(true);
   lv.SelectItem(18);
//--- Create the control
   if(!m_timeframes.CreateComboBox(text,x_gap,y_gap))
      return(false);
//--- Add the element pointer to the database
   CWndContainer::AddToElementsArray(0,m_timeframes);
   return(true);
  }
```

![Fig. 7. Adding the combo box](https://c.mql5.com/2/32/018.png)

Fig. 7. Adding the combo box

### Checkbox

A checkbox is the simplest element. Only two properties should be specified for it.

- Width.
- Location of the main element's right part.

After creating the element, we can enable the checkbox programmatically.

```
//+------------------------------------------------------------------+
//| Create the "Date scale" checkbox                                 |
//+------------------------------------------------------------------+
bool CProgram::CreateDateScale(const int x_gap,const int y_gap,const string text)
  {
//--- Save the window pointer
   m_date_scale.MainPointer(m_tabs1);
//--- Reserve for the tab
   m_tabs1.AddToElementsArray(0,m_date_scale);
//--- Properties
   m_date_scale.XSize(70);
   m_date_scale.AnchorRightWindowSide(true);
//--- Create a control
   if(!m_date_scale.CreateCheckBox(text,x_gap,y_gap))
      return(false);
//--- Enable the checkbox
   m_date_scale.IsPressed(true);
//--- Add the object to the common array of object groups
   CWndContainer::AddToElementsArray(0,m_date_scale);
   return(true);
  }
```

![Fig. 8. Adding the checkboxes](https://c.mql5.com/2/32/019.png)

Fig. 8. Adding the checkboxes

### Table

The GUI will feature two tables. Let's consider the one that visualizes the formed list of symbols and signals for opening positions. It is located on the first tab. First, declare and initialize the arrays for placing the table properties. Set the following properties.

- Element width.
- Table dimensions (number of columns and rows).
- Column width (values are passed in the array).
- Text alignment (values are passed in the array).
- Indent of the text from the cell edges.
- Displaying the headers.
- Ability to select rows.
- Ability to manually re-size columns by dragging a header border.
- Displaying formatting in Zebra style.
- Automatic change of the vertical size relative to the main element.
- Indent from the bottom edge of the main element.

Texts for headers are to be specified after creating the table:

```
//+------------------------------------------------------------------+
//| Create a symbol table                                            |
//+------------------------------------------------------------------+
bool CProgram::CreateSymbolsTable(const int x_gap,const int y_gap)
  {
#define COLUMNS1_TOTAL 2
#define ROWS1_TOTAL    1
//--- Save the pointer to the main element
   m_table_symb.MainPointer(m_tabs1);
//--- Reserve for the tab
   m_tabs1.AddToElementsArray(0,m_table_symb);
//--- Column width array
   int width[COLUMNS1_TOTAL]={95,58};
//--- Text alignment array in columns
   ENUM_ALIGN_MODE align[COLUMNS1_TOTAL]={ALIGN_LEFT,ALIGN_RIGHT};
//--- Text indent array in columns by X axis
   int text_x_offset[COLUMNS1_TOTAL]={5,5};
//--- Properties
   m_table_symb.XSize(168);
   m_table_symb.TableSize(COLUMNS1_TOTAL,ROWS1_TOTAL);
   m_table_symb.ColumnsWidth(width);
   m_table_symb.TextAlign(align);
   m_table_symb.TextXOffset(text_x_offset);
   m_table_symb.ShowHeaders(true);
   m_table_symb.SelectableRow(true);
   m_table_symb.ColumnResizeMode(true);
   m_table_symb.IsZebraFormatRows(clrWhiteSmoke);
   m_table_symb.AutoYResizeMode(true);
   m_table_symb.AutoYResizeBottomOffset(2);
//--- Create a control
   if(!m_table_symb.CreateTable(x_gap,y_gap))
      return(false);
//--- Set the header names
   m_table_symb.SetHeaderText(0,"Symbol");
   m_table_symb.SetHeaderText(1,"Values");
//--- Add the object to the common array of object groups
   CWndContainer::AddToElementsArray(0,m_table_symb);
   return(true);
  }
```

The second table displays some properties of open positions. Ten columns display the following data.

- Position symbol.
- Number of positions.
- Total volume of all open positions.
- Volume of BUY positions.
- Volume of SELL positions.
- The current total result of all open positions.
- The current result of all open BUY positions.
- The current result of all open SELL positions.
- Deposit load per each individual symbol.
- Median price

The following properties should be additionally configured in the second table.

- Image indents from the right and upper cell edges.
- Ability to sort values.
- Automatic change of the horizontal size relative to the main element.
- Indent from the right edge of the main element.

Images in the cells of the first column will symbolize the buttons, by clicking which you can close a position or all of them if this is a hedge account on a specified symbol.

```
//+------------------------------------------------------------------+
//| Create a position table                                          |
//+------------------------------------------------------------------+
bool CProgram::CreatePositionsTable(const int x_gap,const int y_gap)
  {
...
//--- Properties
   m_table_positions.TableSize(COLUMNS2_TOTAL,ROWS2_TOTAL);
   m_table_positions.ColumnsWidth(width);
   m_table_positions.TextAlign(align);
   m_table_positions.TextXOffset(text_x_offset);
   m_table_positions.ImageXOffset(image_x_offset);
   m_table_positions.ImageYOffset(image_y_offset);
   m_table_positions.ShowHeaders(true);
   m_table_positions.IsSortMode(true);
   m_table_positions.SelectableRow(true);
   m_table_positions.ColumnResizeMode(true);
   m_table_positions.IsZebraFormatRows(clrWhiteSmoke);
   m_table_positions.AutoXResizeMode(true);
   m_table_positions.AutoYResizeMode(true);
   m_table_positions.AutoXResizeRightOffset(2);
   m_table_positions.AutoYResizeBottomOffset(2);
...
   return(true);
  }
```

![Fig. 9. Adding the table on the first tab](https://c.mql5.com/2/32/020.png)

Fig. 9. Adding the table on the first tab

![Fig. 10. Adding the table on the second tab](https://c.mql5.com/2/32/001_02__1.png)

Fig. 10. Adding the table on the second tab

Details on working with the tables in the main program file ( **CProgram**) are to be revealed in one of the following article sections.

### Standard chart

The element of **CStandardChart** type is meant to visualize data by symbols. EURUSD D1 is displayed by default. It features the following properties.

- Horizontal scrolling.
- Width auto adjustment.
- Height auto adjustment.
- Indent from the right edge of the main element.
- Indent from the bottom edge of the main element.

If necessary, it is possible to create an array of charts aligned in a horizontal row. To do this, use the **CStandardChart::AddSubChart**() method by passing a symbol and chart timeframe as arguments. However, in this case, we need a single chart, while symbols and timeframes are switched using other controls.

```
//+------------------------------------------------------------------+
//| Create the standard chart 1                                      |
//+------------------------------------------------------------------+
bool CProgram::CreateSubChart1(const int x_gap,const int y_gap)
  {
//--- Save the pointer to the window
   m_sub_chart1.MainPointer(m_tabs1);
//--- Reserve for the 1st tab
   m_tabs1.AddToElementsArray(0,m_sub_chart1);
//--- Properties
   m_sub_chart1.XScrollMode(true);
   m_sub_chart1.AutoXResizeMode(true);
   m_sub_chart1.AutoYResizeMode(true);
   m_sub_chart1.AutoXResizeRightOffset(125);
   m_sub_chart1.AutoYResizeBottomOffset(2);
//--- Add charts
   m_sub_chart1.AddSubChart("EURUSD",PERIOD_D1);
//--- Create a control
   if(!m_sub_chart1.CreateStandardChart(x_gap,y_gap))
      return(false);
//--- Add object to the common array of object groups
   CWndContainer::AddToElementsArray(0,m_sub_chart1);
   return(true);
  }
```

![Fig. 11. Adding the chart](https://c.mql5.com/2/32/001__2.png)

Fig. 11. Adding the chart

### Progress bar

The progress bar allows users to understand what the program is doing now. So, let's add it to the GUI. Below are the properties for our example (in the same order as in the code).

- Total height of the element.
- The indicator height (progress bar line).
- Bar indent by X.
- Bar indent by Y.
- Indent of the main text label by X.
- Indent of the main text label by Y.
- Indent of the percentage text label by X.
- Indent of the percentage text label by Y.
- Indication of a drop-down control (for auto hiding).
- Font.
- Indicator frame color.
- Indicator background color.
- Indicator progress bar line.
- Width auto adjustment.
- Indent from the right edge of the main element.

Examples of applying the progress bar are to be displayed below.

```
//+------------------------------------------------------------------+
//| Create the progress bar                                          |
//+------------------------------------------------------------------+
bool CProgram::CreateProgressBar(const int x_gap,const int y_gap,const string text)
  {
//--- Save the pointer to the main element
   m_progress_bar.MainPointer(m_status_bar);
//--- Properties
   m_progress_bar.YSize(17);
   m_progress_bar.BarYSize(14);
   m_progress_bar.BarXGap(0);
   m_progress_bar.BarYGap(1);
   m_progress_bar.LabelXGap(5);
   m_progress_bar.LabelYGap(2);
   m_progress_bar.PercentXGap(5);
   m_progress_bar.PercentYGap(2);
   m_progress_bar.IsDropdown(true);
   m_progress_bar.Font("Consolas");
   m_progress_bar.BorderColor(clrSilver);
   m_progress_bar.IndicatorBackColor(clrWhiteSmoke);
   m_progress_bar.IndicatorColor(clrLightGreen);
   m_progress_bar.AutoXResizeMode(true);
   m_progress_bar.AutoXResizeRightOffset(2);
//--- Create the element
   if(!m_progress_bar.CreateProgressBar(text,x_gap,y_gap))
      return(false);
//--- Add the element pointer to the database
   CWndContainer::AddToElementsArray(0,m_progress_bar);
   return(true);
  }
```

We have described all the controls to be used in our EA's GUI. At the moment, this is just a graphical shell. Further on, we will develop all the necessary methods to make all this work in accordance with the original idea.

### EasyAndFast library updates

In the [EasyAndFast](https://www.mql5.com/en/code/19703) library, the **CTable::SortData**() public method has been revised in the **CTable** class. Now, as the second argument, you can specify the table sorting direction (optional parameter). Previously, the **CTable::SortData**() method call started sorting in the opposite direction from the current one. Also, the methods for receivingthe current sorting direction and sorted column index have been added. If the table has been sorted by a user manually (by clicking on a header), and then the data in the table has not been updated in the same sequence, it is possible to restore it after finding the current sorting direction.

```
//+------------------------------------------------------------------+
//| Class for creating a drawn table                                 |
//+------------------------------------------------------------------+
class CTable : public CElement
  {
public:
...
   //--- Sort data by a specified column
   void              SortData(const uint column_index=0,const int direction=WRONG_VALUE);
   //--- (1) Current sorting direction, (2) sorted array index
   int               IsSortDirection(void)             const { return(m_last_sort_direction);    }
   int               IsSortedColumnIndex(void)         const { return(m_is_sorted_column_index); }
...
  };
//+------------------------------------------------------------------+
//| Sort data by a specified column                                  |
//+------------------------------------------------------------------+
void CTable::SortData(const uint column_index=0,const int direction=WRONG_VALUE)
  {
//--- Exit if exceeding the table borders
   if(column_index>=m_columns_total)
      return;
//--- Index the sorting is to start from
   uint first_index=0;
//--- Last index
   uint last_index=m_rows_total-1;
//--- Direction is not managed by a user
   if(direction==WRONG_VALUE)
     {
      //--- First time, it is sorted in ascending order. Every other time it is sorted in the opposite direction
      if(m_is_sorted_column_index==WRONG_VALUE || column_index!=m_is_sorted_column_index || m_last_sort_direction==SORT_DESCEND)
         m_last_sort_direction=SORT_ASCEND;
      else
         m_last_sort_direction=SORT_DESCEND;
     }
   else
     {
      m_last_sort_direction=(ENUM_SORT_MODE)direction;
     }
//--- Remember the index of the last sorted data column
   m_is_sorted_column_index=(int)column_index;
//--- Sorting
   QuickSort(first_index,last_index,column_index,m_last_sort_direction);
  }
```

Another small addition has been madeto the **CKeys** class of the **CKeys::KeySymbol**() method. The numeric keypad (a separate block of keys on the right side of the keyboard) has not been processed previously. Now you can enter numbers and special characters from this section of the keyboard as well.

```
//+------------------------------------------------------------------+
//| Return pressed button symbol                                     |
//+------------------------------------------------------------------+
string CKeys::KeySymbol(const long key_code)
  {
   string key_symbol="";
//--- If a space is needed (Space key)
   if(key_code==KEY_SPACE)
     {
      key_symbol=" ";
     }
//--- If (1) an alphabetic character or (2) numeric symbol or (3) a special symbol is required
   else if((key_code>=KEY_A && key_code<=KEY_Z) ||
           (key_code>=KEY_0 && key_code<=KEY_9) ||
           (key_code>=KEY_NUMLOCK_0 && key_code<=KEY_NUMLOCK_SLASH) ||
           (key_code>=KEY_SEMICOLON && key_code<=KEY_SINGLE_QUOTE))
     {
      key_symbol=::ShortToString(::TranslateKey((int)key_code));
     }
//--- Return a symbol
   return(key_symbol);
  }
```

New versions of the **CTable** and **CKeys** classes can be downloaded at the end of the article.

### Conclusion

This was the first part of the article. We have discussed how to develop GUIs for programs of any complexity without excessive effort. You can continue to develop this program and use it for your own purposes. In the second part of the article I will show how to work with GUI, and most importantly — how to fill it with functionality.

Below, you can download the files for testing and detailed study of the code provided in the article.

| File name | Comment |
| --- | --- |
| MQL5\\Experts\\TradePanel\\TradePanel.mq5 | The EA for manual trading with GUI |
| MQL5\\Experts\\TradePanel\\Program.mqh | File with the program class |
| MQL5\\Experts\\TradePanel\\CreateGUI.mqh | File implementing methods for developing GUI from the program class in Program.mqh |
| MQL5\\Include\\EasyAndFastGUI\\Controls\\Table.mqh | Updated CTable class |
| MQL5\\Include\\EasyAndFastGUI\\Keys.mqh | Updated CKeys class |

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/4715](https://www.mql5.com/ru/articles/4715)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/4715.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/4715/mql5.zip "Download MQL5.zip")(48 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Magic of time trading intervals with Frames Analyzer tool](https://www.mql5.com/en/articles/11667)
- [The power of ZigZag (part II). Examples of receiving, processing and displaying data](https://www.mql5.com/en/articles/5544)
- [The power of ZigZag (part I). Developing the base class of the indicator](https://www.mql5.com/en/articles/5543)
- [Universal RSI indicator for working in two directions simultaneously](https://www.mql5.com/en/articles/4828)
- [Expert Advisor featuring GUI: Adding functionality (part II)](https://www.mql5.com/en/articles/4727)
- [Visualizing optimization results using a selected criterion](https://www.mql5.com/en/articles/4636)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/269651)**
(15)


![fobos3_1](https://c.mql5.com/avatar/2021/1/6013393A-14DD.jpg)

**[fobos3\_1](https://www.mql5.com/en/users/fobos3_1)**
\|
4 Feb 2021 at 03:15

Excellent, very detailed, thank you so much for sharing.


![Guilherme Mendonca](https://c.mql5.com/avatar/2018/9/5B98163A-29AC.jpg)

**[Guilherme Mendonca](https://www.mql5.com/en/users/billy-gui)**
\|
8 Mar 2021 at 14:48

Is there any way to fix this message bellow?

```
deprecated behavior, hidden method calling will be disabled in a future MQL compiler version
```

![Ali Reza Emami Bistgani](https://c.mql5.com/avatar/avatar_na2.png)

**[Ali Reza Emami Bistgani](https://www.mql5.com/en/users/alien4ufo)**
\|
16 Sep 2022 at 12:56

**Mario Trinchero [#](https://www.mql5.com/en/forum/269651#comment_8238291):**

Why this errors?

How I can correct it?

Thanks

Could you find any solutions?

![Ali Reza Emami Bistgani](https://c.mql5.com/avatar/avatar_na2.png)

**[Ali Reza Emami Bistgani](https://www.mql5.com/en/users/alien4ufo)**
\|
16 Sep 2022 at 12:57

**MetaQuotes:**

New article [Expert Advisor featuring GUI: Creating the panel (part I)](https://www.mql5.com/en/articles/4715) has been published:

Author: [Anatoli Kazharski](https://www.mql5.com/en/users/tol64 "tol64")

It has so many errors .... what i gotta do? any one?

![Marco](https://c.mql5.com/avatar/avatar_na2.png)

**[Marco](https://www.mql5.com/en/users/gigino5)**
\|
10 Dec 2023 at 10:51

I have an error and I don't know how to fix it

'AddItem' - wrong parameters countGraphicalPanel\_v2.00.mqh16122

I copied and pasted the function and only changed the class name. Thanks

![Deep Neural Networks (Part VI). Ensemble of neural network classifiers: bagging](https://c.mql5.com/2/48/Deep_Neural_Networks_06.png)[Deep Neural Networks (Part VI). Ensemble of neural network classifiers: bagging](https://www.mql5.com/en/articles/4227)

The article discusses the methods for building and training ensembles of neural networks with bagging structure. It also determines the peculiarities of hyperparameter optimization for individual neural network classifiers that make up the ensemble. The quality of the optimized neural network obtained in the previous article of the series is compared with the quality of the created ensemble of neural networks. Possibilities of further improving the quality of the ensemble's classification are considered.

![How to analyze the trades of the Signal selected in the chart](https://c.mql5.com/2/32/bv8-az2zxypg7t6xs-7r9h1l-nlm87q3q35-vvyg13n-gv-dcau99f.png)[How to analyze the trades of the Signal selected in the chart](https://www.mql5.com/en/articles/4751)

The trade Signals service develops in leaps and bounds. Trusting our funds to a signal provider, we would like to minimize the risk of losing our deposit. So how to puzzle out in this forest of trade signals? How to find the one that would produce profits? This paper proposes to create a tool for visually analyzing the history of trades on trade signals in a symbol chart.

![Applying the Monte Carlo method for optimizing trading strategies](https://c.mql5.com/2/32/Monte_Carlo.png)[Applying the Monte Carlo method for optimizing trading strategies](https://www.mql5.com/en/articles/4347)

Before launching a robot on a trading account, we usually test and optimize it on quotes history. However, a reasonable question arises: how can past results help us in the future? The article describes applying the Monte Carlo method to construct custom criteria for trading strategy optimization. In addition, the EA stability criteria are considered.

![Improving Panels: Adding transparency, changing background color and inheriting from CAppDialog/CWndClient](https://c.mql5.com/2/32/Advanced_Pane.png)[Improving Panels: Adding transparency, changing background color and inheriting from CAppDialog/CWndClient](https://www.mql5.com/en/articles/4575)

In this article, we continue studying the use of CAppDialog. Now we will learn how to set color for the background, borders and header of the dialog box. Also, this article provides a step-by-step description of how to add transparency for an application window when dragging it within the chart. We will consider how to create child classes of CAppDialog or CWndClient and analyze new specifics of working with controls. Finally, we will review new Projects from a new perspective.

[![](https://www.mql5.com/ff/sh/0wxx5f0vuwq7xh89z2/01.png)VPS for 24/7 tradingContact your broker and find out how to get a free hosting subscriptionLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=nhetzvgituppcfrhndpblbihmzziogdh&s=d00c975c8bda3d8c1b29f042ad33ac81952ccea2f130a8f1ffa9015bab8ade87&uid=&ref=https://www.mql5.com/en/articles/4715&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=6412191147402131455)

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