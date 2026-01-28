---
title: Improve Your Trading Charts With Interactive GUI's in MQL5 (Part III): Simple Movable Trading GUI
url: https://www.mql5.com/en/articles/12923
categories: Trading, Trading Systems, Indicators, Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T18:06:24.670372
---

[![](https://www.mql5.com/ff/si/fx5m8s6u6uxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=ysducdhemkrdsdtzzbfkclolrllnhezk&s=33f180a31db6c3b846d77732b0bc78169421a47b8cf9f076ca717f4e4846d1c7&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=rpxyebizshtrtdrrrialenhwqyrafqiz&ssn=1769180782630611458&ssn_dr=1&ssn_sr=0&fv_date=1769180782&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F12923&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Improve%20Your%20Trading%20Charts%20With%20Interactive%20GUI%27s%20in%20MQL5%20(Part%20III)%3A%20Simple%20Movable%20Trading%20GUI%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918078303223371&fz_uniq=5069097252661690478&sv=2552)

MetaTrader 5 / Trading


### Introduction

Hello and welcome back to part 3 of our series "Improve Your Trading Charts With Interactive GUI's in MQL5".

Before we venture into new territory, let's quickly recap what we've covered in Parts I and II:

> 1\. In Part I, we started by understanding the concept of chart events, and from there, we created two simple movable dashboards on the same chart.
>
> 2\. For Part II, we took a step further. We utilized classes within a .mqh file to make our code more efficient and versatile, ready for integration with full-scale EAs/Indicators.

And now, we are ready for Part III! In this part, we're going to focus on enhancing our dashboards by integrating GUIs into them. Because without GUIs, dashboards won't serve their intended purpose.

Here's a quick overview of what we'll tackle in this article:

1. [What are we creating?](https://www.mql5.com/en/articles/12923#what_are_we_creating?)
2. [Creating a simple trading static dashboard](https://www.mql5.com/en/articles/12923#creating_a_simple_trading_static_dashboard)
3. [Discussing the Approach to make our static dashboard move with all elements inside it](https://www.mql5.com/en/articles/12923#discussing_the_Approach_to_make_our_static_dashboard_move_with_all_elements_inside_it)
4. [Using discussed approach to make our static dashboard movable](https://www.mql5.com/en/articles/12923/133041/editusing_discussed_approach_to_make_our_static_dashboard_movable)
5. [Conclusion](https://www.mql5.com/en/articles/12923/133041/editconclusion)

### What are we creating?

We aim to create a Movable Dashboard with a GUI, and for that, we need to decide what we will be creating. I've chosen a simple EA, specifically, the Simple Trading EA, as our basis.

First, we need to construct this static dashboard i.e., the Simple Trading EA. It's crucial to do this efficiently since we're creating a full-fledged EA. By efficiency, I mean that we cannot merely open a file and write all the code there. Instead, we need a well-considered plan that allows us to write the bare minimum code across several well-organized .mqh files. Most importantly, we must avoid duplicating the same code repeatedly to create the required static GUIs for our movable dashboard.

Here is the basic static dashboard that we will be creating for our purpose:

![Fig 1. Simple Static Dashboard](https://c.mql5.com/2/57/Fig_1._Simple_Static_Dashboard.png)

**Fig 1. Simple Static Dashboard**

It comprises:

| Element | Description |
| --- | --- |
| Label 1 | Title Text (Simple Trading EA V1.0) |
| Label 2 | Lot Size |
| Edit 1 | The white-colored edit box you see in the image above, with 0.01 written inside it. |
| Button 1 | The green-colored Buy button. |
| Button 2 | The red-colored Sell button. |
| Rectangle Label 1 | Title bar, the dark blue-colored bar on which "Simple Trading EA V1.0" is written. |
| Rectangle Label 2 | Main dashboard area, the light blue-colored dashboard. |

So, our dashboard consists of these seven components combined. If you ask me, I'd say that's a pretty good-looking dashboard we've created just by combining these seven elements.

Now, let's start coding this dashboard.

### Creating a simple trading static dashboard

What classes are we going to write? Let's think...

We will need 2 Labels, 2 Buttons, 1 Edit, and 2 Rectangle Labels. So, let's create 4 .mqh files, one for each of these. Here's our project's folder structure:

- Simple Trading EA/

  - SimpleTradingEA.mq5
  - Button.mqh
  - Label.mqh
  - Edit.mqh
  - RectangleLabel.mqh

These are the files in which we will be writing our code. Now, let's create our first file, "SimpleTradingEA.mq5", which is our main EA file.

I have removed the OnTick() function as we won't be needing it for this project. Here's what the file looks like at the moment:

```
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
   {
    return(INIT_SUCCEEDED);
   }

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
   {
   }
//+------------------------------------------------------------------+
```

Let's create a plan now. We will construct our static dashboard in the following order:

1. Title bar
2. Main Dashboard Body
3. Title Text
4. "Lot Size:" Text

5. Edit Box

6. Buy and Sell Button
7. Add any necessary finishing touches

This seems like a reasonable sequence to follow. Let's Start,

01. **Title bar**

     To create the Title bar, we need to use the Rectangle Label Object. So, let's create a class that will handle everything related to the Rectangle Label Object. We will be creating an .mqh file; let's name it "RectangleLabel.mqh" to keep things simple, and let's name the class "RectangleLabel," again to keep things simple.

    Here's the empty class we created:





    ```
    //+------------------------------------------------------------------+
    //| Class Definition: RectangleLabel                                 |
    //+------------------------------------------------------------------+
    class RectangleLabel
       {
    public:
                         RectangleLabel(void);
                        ~RectangleLabel(void);
       };

    //+------------------------------------------------------------------+
    //| Constructor: RectangleLabel                                      |
    //+------------------------------------------------------------------+
    RectangleLabel::RectangleLabel(void)
       {
       }

    //+------------------------------------------------------------------+
    //| Destructor: RectangleLabel                                       |
    //+------------------------------------------------------------------+
    RectangleLabel::~RectangleLabel(void)
       {
       }
    //+------------------------------------------------------------------+
    ```


    We will be needing some functions, let's see


    1. Create             -> To create the rectangle label
    2. Destroy            -> To destroy the dashboard
    3. SetBorderType  -> To set border type

    4. SetBGColor       -> To set background color

Let's declare the above functions in member function list. Now our class looks like this:

```
//+------------------------------------------------------------------+
//| Class Definition: RectangleLabel                                 |
//+------------------------------------------------------------------+
class RectangleLabel
   {
public:
                     RectangleLabel(void); // Constructor
                    ~RectangleLabel(void); // Destructor
    void             Create(string name, int xDis, int yDis, int xSize, int ySize); //Creates a Rectangle Label with the given parameters
    void             Destroy(); // Destroys the Rectangle Label
    void             SetBorderType(ENUM_BORDER_TYPE borderType); // Sets the border type of the Rectangle Label
    void             SetBGColor(color col); // Sets the background color of the Rectangle Label
   };
//+------------------------------------------------------------------+
```

Let's write down a basic create function:

```
//+------------------------------------------------------------------+
//| RectangleLabel Class - Create Method                             |
//+------------------------------------------------------------------+
void RectangleLabel::Create(string name, int xDis, int yDis, int xSize, int ySize)
   {
    ObjectCreate(0, name, OBJ_RECTANGLE_LABEL, 0, 0, 0); // Create the Rectangle Label object
    ObjectSetInteger(0, name, OBJPROP_XDISTANCE, xDis); // Set the X-axis distance
    ObjectSetInteger(0, name, OBJPROP_YDISTANCE, yDis); // Set the Y-axis distance
    ObjectSetInteger(0, name, OBJPROP_XSIZE, xSize); // Set the X size
    ObjectSetInteger(0, name, OBJPROP_YSIZE, ySize); // Set the Y size
   }
//+------------------------------------------------------------------+
```

Let's create Destroy, SetBorderType and SetBGColor in the same line as they only require one line. Here's our updated class:

```
//+------------------------------------------------------------------+
//| Class Definition for the Rectangle Label                         |
//+------------------------------------------------------------------+
class RectangleLabel
   {
private:
    string           _name; // Name of the rectangle label
public:
                     RectangleLabel(void); // Constructor
                    ~RectangleLabel(void); // Destructor

    void             Create(string name, int xDis, int yDis, int xSize, int ySize); // Method to create a rectangle label with given dimensions

    void             Destroy() {ObjectDelete(0, _name);} // Method to delete the object using the object's name

    void             SetBorderType(ENUM_BORDER_TYPE borderType) {ObjectSetInteger(0, _name, OBJPROP_BORDER_TYPE, borderType);} // Method to set the border type for the rectangle label

    void             SetBGColor(color col) {ObjectSetInteger(0, _name, OBJPROP_BGCOLOR, col);} // Method to set the background color for the rectangle label
   };
//+------------------------------------------------------------------+
```

Also we added a private variable named "\_name" as ObjectDelete requires a name and we set "\_name" in Create function, It now looks like:

```
//+------------------------------------------------------------------+
//| Rectangle Label Creation Method                                  |
//+------------------------------------------------------------------+
void RectangleLabel::Create(string name, int xDis, int yDis, int xSize, int ySize)
   {
    ObjectCreate(0, name, OBJ_RECTANGLE_LABEL, 0, 0, 0); // Create rectangle label object
    ObjectSetInteger(0, name, OBJPROP_XDISTANCE, xDis); // Set X distance
    ObjectSetInteger(0, name, OBJPROP_YDISTANCE, yDis); // Set Y distance
    ObjectSetInteger(0, name, OBJPROP_XSIZE, xSize); // Set X size
    ObjectSetInteger(0, name, OBJPROP_YSIZE, ySize); // Set Y size
    _name = name; // Assign the name to the member variable
   }
//+------------------------------------------------------------------+
```

we simply added "\_name = name;" in the last line to set the \_name variable to the name of the rectangle label when it was created.

If you are wandering where is the code that will make it movable, we are ignoring that aspect at the moment to keep things simple until we create a simple static dashboard.

Now Let's use this class in main file i.e. SimpleTradingEA.mq5 to see the result:

We first included the RectangleLabel.mqh file using "#include" and created a instance of the class named TitleBar as we are creating Title bar of the dashboard with this instance of the RectangleLabel class, We will be using it again for the Main Dashboard Body.

Then we used this instance to create a Rectangle Label on the chart at (100,100) coordinate with dimensions of 200x20. Then we set its border to Flat (BORDER\_FLAT) as that looks better according to me; you may change it according to your preference. Then we use the ChartRedraw(0) function to redraw the chart; that way, the dashboard will be created on the chart immediately. Otherwise, it may wait for the next price update, i.e., tick.

That was all in OnInit(), i.e., execute only once to create and show the dashboard on the chart.

Finally, we destroy the dashboard using our created Destroy() function in OnDeinit(), i.e., when the EA is removed from the chart.

Result:

![Fig 2. Title bar](https://c.mql5.com/2/57/Fig_2._Title_bar.png)

**Fig 2. Title bar**

04. **Main Dashboard Body**




    Let's again use the RectangleLabel class to create the main body. It's simple; we just need to create another instance; let's name it "MainDashboardBody" and add the below simple code in OnInit() after we create the title bar and then finally add MainDashboardBody.Destroy() in OnDeinit():


    ```
    // Creating a rectangle label called "MainDashboardBody" with specific dimensions
    MainDashboardBody.Create("MainDashboardBody", 100, 119, 200, 100);
    // Setting the border type of the "MainDashboardBody" rectangle label to be flat
    MainDashboardBody.SetBorderType(BORDER_FLAT);
    ```

    After the our code looks like this:



    ```
    #include "RectangleLabel.mqh" // Including the RectangleLabel class definition
    RectangleLabel TitleBar; // Declaration of a TitleBar object
    RectangleLabel MainDashboardBody; // Declaration of a MainDashboardBody object

    //+------------------------------------------------------------------+
    //| Expert initialization function                                   |
    //+------------------------------------------------------------------+
    int OnInit()
       {
        TitleBar.Create("TitleBar", 100, 100, 200, 20); // Creating the TitleBar with specified dimensions
        TitleBar.SetBorderType(BORDER_FLAT); // Setting the border type of TitleBar to be flat

        MainDashboardBody.Create("MainDashboardBody", 100, 119, 200, 100); // Creating the MainDashboardBody with specified dimensions
        MainDashboardBody.SetBorderType(BORDER_FLAT); // Setting the border type of MainDashboardBody to be flat

        ChartRedraw(0); // Redrawing the chart to reflect changes
        return(INIT_SUCCEEDED); // Indicating successful initialization
       }

    //+------------------------------------------------------------------+
    //| Expert deinitialization function                                 |
    //+------------------------------------------------------------------+
    void OnDeinit(const int reason)
       {
        MainDashboardBody.Destroy(); // Destroying the MainDashboardBody object
        TitleBar.Destroy(); // Destroying the TitleBar object
       }
    //+------------------------------------------------------------------+
    ```


    With that our result looks quite good:



    ![Fig 3. Added Main Dashboard Body](https://c.mql5.com/2/57/Fig_3._Added_Main_Dashboard_Body.png)



    **Fig 3. Added Main Dashboard Body**


06. **Title Text**




    To add title text, we need to create a class similar to RectangleLabel but specifically for labels, allowing us to add text. Here's the code for a new class named Label :


    ```
    //+------------------------------------------------------------------+
    //| Label class definition                                           |
    //+------------------------------------------------------------------+
    class Label
       {
    private:
        string           _name; // Name of the label
    public:
                         Label(void); // Constructor
                        ~Label(void); // Destructor

        void             Create(string name, int xDis, int yDis); // Method to create a label
        void             Destroy() {ObjectDelete(0, _name);} // Method to destroy a label
        void             SetTextColor(color col) {ObjectSetInteger(0, _name, OBJPROP_COLOR, col);} // Method to set the text color
        void             SetText(string text) {ObjectSetString(0, _name, OBJPROP_TEXT, text);} // Method to set the text content
        string           GetText() {return ObjectGetString(0, _name, OBJPROP_TEXT);} // Method to retrieve the text content
        void             SetFontSize(int fontSize) {ObjectSetInteger(0, _name, OBJPROP_FONTSIZE, fontSize);} // Method to set the font size
        void             SetFont(string fontName) {ObjectSetString(0, _name, OBJPROP_FONT, fontName);} // Method to set the font name
       };

    //+------------------------------------------------------------------+
    //| Constructor                                                      |
    //+------------------------------------------------------------------+
    Label::Label(void)
       {

       }

    //+------------------------------------------------------------------+
    //| Destructor                                                       |
    //+------------------------------------------------------------------+
    Label::~Label(void)
       {

       }

    //+------------------------------------------------------------------+
    //| Method to create a label object                                  |
    //+------------------------------------------------------------------+
    void Label::Create(string name, int xDis, int yDis)
       {
        // Code to create label object, set its position, and assign its name
        ObjectCreate(0, name, OBJ_LABEL, 0, 0, 0);
        ObjectSetInteger(0, name, OBJPROP_XDISTANCE, xDis);
        ObjectSetInteger(0, name, OBJPROP_YDISTANCE, yDis);
        _name = name;
       }
    //+------------------------------------------------------------------+
    ```



    - Created class named Label in a new .mqh file named Label.mqh
    - Decalred a private variable named \_name to store name privately
    - Created a function named Create with 3 requried parameters: name, xDis, yDis. Size is irrelevant for a Label object, To change Text Size we change Font Size
    - Craete a function named Destroy to destroy the Label
    - Created a function SetTextColor to set Text Color
    - Created a function to Set Text of the Label Object
    - Create a function GetText to get the text of the Label object which of course returns string
    - Created a function to SetFontSize to of course set Font Size
    - Created a function to set Font to set Font, requires name of the Font in string and of course font should be available/installed in the Operating System

That is it for the Label. Now Let's use it to create a label object on chart, No actually 2 label objects on the chart.

Now Our SimpleTradingEA.mq5 looks like:

```
#include "RectangleLabel.mqh" // Including the RectangleLabel class definition
RectangleLabel TitleBar; // Declaration of a TitleBar object
RectangleLabel MainDashboardBody; // Declaration of a MainDashboardBody object

#include "Label.mqh" // Including the Label class definition
Label TitleText; // Declaration of a Label object

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
   {
    TitleBar.Create("TitleBar", 100, 100, 200, 20); // Creating the TitleBar with specified dimensions
    TitleBar.SetBorderType(BORDER_FLAT); // Setting the border type of TitleBar to be flat

    MainDashboardBody.Create("MainDashboardBody", 100, 119, 200, 100); // Creating the MainDashboardBody with specified dimensions
    MainDashboardBody.SetBorderType(BORDER_FLAT); // Setting the border type of MainDashboardBody to be flat

    TitleText.Create("TitleText", 110, 101); // Creating the TitleText at (110,101)
    TitleText.SetText("Simple Trading EA V1.0"); // Setting its text to "Simple Trading EA V1.0"
    TitleText.SetFontSize(10); // Setting its font size to 10
    TitleText.SetTextColor(clrBlack); // Setting its text color to clrBlack

    ChartRedraw(0); // Redrawing the chart to reflect changes
    return(INIT_SUCCEEDED); // Indicating successful initialization
   }

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
   {
    MainDashboardBody.Destroy(); // Destroying the MainDashboardBody object
    TitleBar.Destroy(); // Destroying the TitleBar object
    TitleText.Destroy(); // Destroying the TitleText object
   }
//+------------------------------------------------------------------+
```

    - Created Label instance named TitleText
    - Used TitleText.Create function to create the TitleText
    - Used TitleText.SetText to set TitleText to "Simple Trading EA V1.0"
    - Used TitleText.SetFontSize to Set FontSize to 10
    - Used TitleText.SetTextColor to set color to Black
    - Used TitleText.Destroy to destroy the the TitleText object in OnDeinit

Result:

![Fig 4. Added Title Text](https://c.mql5.com/2/57/Fig_4._Added_Title_Text.png)

**Fig 4. Added Title Text**

08. **"Lot Size:" Text**




    For the "Lot Size:" text, you'll follow a process similar to the title text. The final code is as follows:




    ```
    #include "RectangleLabel.mqh" // Including the RectangleLabel class definition
    RectangleLabel TitleBar; // Declaration of a TitleBar object
    RectangleLabel MainDashboardBody; // Declaration of a MainDashboardBody object

    #include "Label.mqh" // Including the Label class definition
    Label TitleText; // Declaration of a Label object
    Label LotSizeText; // Declaration of a LotSizeText object

    //+------------------------------------------------------------------+
    //| Expert initialization function                                   |
    //+------------------------------------------------------------------+
    int OnInit()
       {
        TitleBar.Create("TitleBar", 100, 100, 200, 20); // Creating the TitleBar with specified dimensions
        TitleBar.SetBorderType(BORDER_FLAT); // Setting the border type of TitleBar to be flat

        MainDashboardBody.Create("MainDashboardBody", 100, 119, 200, 100); // Creating the MainDashboardBody with specified dimensions
        MainDashboardBody.SetBorderType(BORDER_FLAT); // Setting the border type of MainDashboardBody to be flat

        TitleText.Create("TitleText", 110, 101); // Creating the TitleText at (110,101)
        TitleText.SetText("Simple Trading EA V1.0"); // Setting its text to "Simple Trading EA V1.0"
        TitleText.SetFontSize(10); // Setting its font size to 10
        TitleText.SetTextColor(clrBlack); // Setting its text color to clrBlack

        LotSizeText.Create("LotSizeText", 110, 140); // Creating the LotSizeText at (110,140)
        LotSizeText.SetText("Lot Size:"); // Setting its text to "Lot Size:"
        LotSizeText.SetFontSize(12); // Setting its font size to 12
        LotSizeText.SetTextColor(clrBlack); // Setting its text color to clrBlack

        ChartRedraw(0); // Redrawing the chart to reflect changes
        return(INIT_SUCCEEDED); // Indicating successful initialization
       }

    //+------------------------------------------------------------------+
    //| Expert deinitialization function                                 |
    //+------------------------------------------------------------------+
    void OnDeinit(const int reason)
       {
        MainDashboardBody.Destroy(); // Destroying the MainDashboardBody object
        TitleBar.Destroy(); // Destroying the TitleBar object
        TitleText.Destroy(); // Destroying the TitleText object
        LotSizeText.Destroy(); // Destroying the LotSizeText object
       }
    //+------------------------------------------------------------------+
    ```


    - Created Label instance named LotSizeText
    - Used LotSizeText.Create function to create the Lot Size Text
    - Used LotSizeText.SetText to set text to "Lot Size:"
    - Used LotSizeText.SetFontSize to Set FontSize to 12
    - Used LotSizeText.SetTextColor to set color to Black
    - Used LotSizeText.Destroy to destroy the the Label object in OnDeinit

That is all for it. Result:

![Fig 5. Added Lot Size Text](https://c.mql5.com/2/57/Fig_5._Added_Lot_Size_Text.png)

**Fig 5. Added "Lot Size:" Text**

14. **Edit Box**




    For the Edit Box, you'll create a class quite similar to the Label class. Here's the code for a new class named Edit :




    ```
    //+------------------------------------------------------------------+
    //| Edit class definition                                            |
    //+------------------------------------------------------------------+
    class Edit
       {
    private:
        string           _name; // Name of the edit control
    public:
                         Edit(void); // Constructor
                        ~Edit(void); // Destructor

        void             Create(string name, int xDis, int yDis, int xSize, int ySize); // Method to create an edit control
        void             Destroy() {ObjectDelete(0, _name);} // Method to destroy an edit control
        void             SetBorderColor(color col) {ObjectSetInteger(0, _name, OBJPROP_BORDER_COLOR, col);} // Method to set the border color
        void             SetBGColor(color col) {ObjectSetInteger(0, _name, OBJPROP_BGCOLOR, col);} // Method to set the background color
        void             SetTextColor(color col) {ObjectSetInteger(0, _name, OBJPROP_COLOR, col);} // Method to set the text color
        void             SetText(string text) {ObjectSetString(0, _name, OBJPROP_TEXT, text);} // Method to set the text content
        string           GetText() {return ObjectGetString(0, _name, OBJPROP_TEXT);} // Method to retrieve the text content
       };

    //+------------------------------------------------------------------+
    //| Constructor                                                      |
    //+------------------------------------------------------------------+
    Edit::Edit(void)
       {

       }

    //+------------------------------------------------------------------+
    //| Destructor                                                       |
    //+------------------------------------------------------------------+
    Edit::~Edit(void)
       {

       }

    //+------------------------------------------------------------------+
    //| Method to create an edit control object                          |
    //+------------------------------------------------------------------+
    void Edit::Create(string name, int xDis, int yDis, int xSize, int ySize)
       {
        // Code to create edit control object, set its position, size, and assign its name
        ObjectCreate(0, name, OBJ_EDIT, 0, 0, 0);
        ObjectSetInteger(0, name, OBJPROP_XDISTANCE, xDis);
        ObjectSetInteger(0, name, OBJPROP_YDISTANCE, yDis);
        ObjectSetInteger(0, name, OBJPROP_XSIZE, xSize);
        ObjectSetInteger(0, name, OBJPROP_YSIZE, ySize);
        _name = name;
       }
    //+------------------------------------------------------------------+
    ```



    - Created class named Edit in a new .mqh file named Edit.mqh
    - Decalred a private variable named \_name to store name privately
    - Created a function named Create with 5 requried parameters: name, xDis, yDis, xSize, ySize
    - Created a function named Destroy to destroy the Edit Object
    - Created a function SetBorderColor to set Border Color
    - Created a function SetBGColor to se Background color to WhiteSmoke
    - Created a function SetTextColor to set the text color of the text inside the edit box
    - Created a function SetText to set text
    - Created a function GetText to get text

You can now use the Edit class in SimpleTradingEA, as shown below:

```
#include "RectangleLabel.mqh" // Including the RectangleLabel class definition
RectangleLabel TitleBar; // Declaration of a TitleBar object
RectangleLabel MainDashboardBody; // Declaration of a MainDashboardBody object

#include "Label.mqh" // Including the Label class definition
Label TitleText; // Declaration of a Label object
Label LotSizeText; // Declaration of a LotSizeText object

#include "Edit.mqh" // Including the Edit class definition
Edit LotSize; // Declaration of a LotSize object

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
   {
    TitleBar.Create("TitleBar", 100, 100, 200, 20); // Creating the TitleBar with specified dimensions
    TitleBar.SetBorderType(BORDER_FLAT); // Setting the border type of TitleBar to be flat

    MainDashboardBody.Create("MainDashboardBody", 100, 119, 200, 100); // Creating the MainDashboardBody with specified dimensions
    MainDashboardBody.SetBorderType(BORDER_FLAT); // Setting the border type of MainDashboardBody to be flat

    TitleText.Create("TitleText", 110, 101); // Creating the TitleText at (110,101)
    TitleText.SetText("Simple Trading EA V1.0"); // Setting its text to "Simple Trading EA V1.0"
    TitleText.SetFontSize(10); // Setting its font size to 10
    TitleText.SetTextColor(clrBlack); // Setting its text color to clrBlack

    LotSizeText.Create("LotSizeText", 110, 140); // Creating the LotSizeText at (110,140)
    LotSizeText.SetText("Lot Size:"); // Setting its text to "Lot Size:"
    LotSizeText.SetFontSize(12); // Setting its font size to 12
    LotSizeText.SetTextColor(clrBlack); // Setting its text color to clrBlack

    LotSize.Create("LotSize", 220, 140, 50, 20); // Creating the LotSize with specified dimensions
    LotSize.SetBorderColor(clrBlack); // Setting its Border Color to clrBlack
    LotSize.SetBGColor(clrWhiteSmoke); // Setting its BG Color to clrWhiteSmoke
    LotSize.SetText("0.01"); // Setting its text to 0.01
    LotSize.SetTextColor(clrBlack); // Setting its text color to clrBlack

    ChartRedraw(0); // Redrawing the chart to reflect changes
    return(INIT_SUCCEEDED); // Indicating successful initialization
   }

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
   {
    MainDashboardBody.Destroy(); // Destroying the MainDashboardBody object
    TitleBar.Destroy(); // Destroying the TitleBar object
    TitleText.Destroy(); // Destroying the TitleText object
    LotSizeText.Destroy(); // Destroying the LotSizeText object
    LotSize.Destroy(); // Destroying the LotSize object
   }
//+------------------------------------------------------------------+
```

    - Created Edit instance named LotSize
    - Used LotSize.Create function to create the Edit Object
    - Used LotSize.SetBorderColor to set border color to Black
    - Used LotSize.SetBGColor to Set background color to WhiteSmoke
    - Used LotSize.SetText to set text to 0.01 representing lot size
    - Used LotSize.SetTextColor to set the text color inside the edit box to Black
    - Used LotSize.Destroy to destroy the the Edit object in OnDeinit

15. **Buy and Sell Buttons**




    Finally, we come to the buttons. Let's create a class for buttons in a similar way as we did for others:




    ```
    //+------------------------------------------------------------------+
    //| Button class definition                                          |
    //+------------------------------------------------------------------+
    class Button
       {
    private:
        string           _name; // Name of the button control

    public:
                         Button(void); // Constructor
                        ~Button(void); // Destructor

        void             Create(string name, int xDis, int yDis, int xSize, int ySize); // Method to create a button control
        void             SetBorderColor(color col) {ObjectSetInteger(0, _name, OBJPROP_BORDER_COLOR, col);} // Method to set the border color
        void             SetBGColor(color col) {ObjectSetInteger(0, _name, OBJPROP_BGCOLOR, col);} // Method to set the background color
        void             SetText(string text) {ObjectSetString(0, _name, OBJPROP_TEXT, text);} // Method to set the text content
        void             Destroy() {ObjectDelete(0, _name);} // Method to destroy a button control
       };

    //+------------------------------------------------------------------+
    //| Constructor                                                      |
    //+------------------------------------------------------------------+
    Button::Button(void)
       {

       }

    //+------------------------------------------------------------------+
    //| Destructor                                                       |
    //+------------------------------------------------------------------+
    Button::~Button(void)
       {

       }

    //+------------------------------------------------------------------+
    //| Method to create a button control object                         |
    //+------------------------------------------------------------------+
    void Button::Create(string name, int xDis = 0, int yDis = 0, int xSize = 0, int ySize = 0)
       {
        // Code to create button control object, set its position, size, and assign its name
        ObjectCreate(0, name, OBJ_BUTTON, 0, 0, 0);
        ObjectSetInteger(0, name, OBJPROP_XDISTANCE, xDis);
        ObjectSetInteger(0, name, OBJPROP_YDISTANCE, yDis);
        ObjectSetInteger(0, name, OBJPROP_XSIZE, xSize);
        ObjectSetInteger(0, name, OBJPROP_YSIZE, ySize);
        _name = name;
       }
    //+------------------------------------------------------------------+
    ```


    In a new .mqh file named Button.mqh , we have created a class named Button . We've declared a private variable named \_name to store the name privately. We have also created the following functions:






    - A function named Create with 5 required parameters: name, xDis, yDis, xSize, ySize.
    - A function named Destroy to destroy the Button Object.
    - A function named SetBorderColor to set the Border Color.
    - A function named SetBGColor to set the Background color to WhiteSmoke.
    - A function named SetText to set text.

Now let's look at the main SimpleTradingEA.mq5 file after adding the buttons. You'll notice that it now includes instances for RectangleLabel , Label , Edit , Button for BuyButton , and SellButton.

```
#include "RectangleLabel.mqh" // Including the RectangleLabel class definition
RectangleLabel TitleBar; // Declaration of a TitleBar object
RectangleLabel MainDashboardBody; // Declaration of a MainDashboardBody object

#include "Label.mqh" // Including the Label class definition
Label TitleText; // Declaration of a Label object
Label LotSizeText; // Declaration of a LotSizeText object

#include "Edit.mqh" // Including the Edit class definition
Edit LotSize; // Declaration of a LotSize object

#include "Button.mqh" // Including the Button class definition
Button BuyButton; // Declaration of a BuyButton object
Button SellButton; // Declaration of a SellButton object

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
   {
    TitleBar.Create("TitleBar", 100, 100, 200, 20); // Creating the TitleBar with specified dimensions
    TitleBar.SetBorderType(BORDER_FLAT); // Setting the border type of TitleBar to be flat

    MainDashboardBody.Create("MainDashboardBody", 100, 119, 200, 100); // Creating the MainDashboardBody with specified dimensions
    MainDashboardBody.SetBorderType(BORDER_FLAT); // Setting the border type of MainDashboardBody to be flat

    TitleText.Create("TitleText", 110, 101); // Creating the TitleText at (110,101)
    TitleText.SetText("Simple Trading EA V1.0"); // Setting its text to "Simple Trading EA V1.0"
    TitleText.SetFontSize(10); // Setting its font size to 10
    TitleText.SetTextColor(clrBlack); // Setting its text color to clrBlack

    LotSizeText.Create("LotSizeText", 110, 140); // Creating the LotSizeText at (110,140)
    LotSizeText.SetText("Lot Size:"); // Setting its text to "Lot Size:"
    LotSizeText.SetFontSize(12); // Setting its font size to 12
    LotSizeText.SetTextColor(clrBlack); // Setting its text color to clrBlack

    LotSize.Create("LotSize", 220, 140, 50, 20); // Creating the LotSize with specified dimensions
    LotSize.SetBorderColor(clrBlack); // Setting its Border Color to clrBlack
    LotSize.SetBGColor(clrWhiteSmoke); // Setting its BG Color to clrWhiteSmoke
    LotSize.SetText("0.01"); // Setting its text to 0.01
    LotSize.SetTextColor(clrBlack); // Setting its text color to clrBlack

    BuyButton.Create("BuyButton", 110, 180, 80, 25); // Creating the BuyButton with specified dimensions
    BuyButton.SetBorderColor(clrBlack); // Setting its Border Color to clrBlack
    BuyButton.SetText("Buy"); // Setting its text to "Buy"
    BuyButton.SetBGColor(clrLime); // Setting its BG Color to clrLime

    SellButton.Create("SellButton", 210, 180, 80, 25); // Creating the SellButton with specified dimensions
    SellButton.SetBorderColor(clrBlack); // Setting its Border Color to clrBlack
    SellButton.SetText("Sell"); // Setting its text to "Sell"
    SellButton.SetBGColor(clrRed); // Setting its BG Color to clrRed

    ChartRedraw(0); // Redrawing the chart to reflect changes
    return(INIT_SUCCEEDED); // Indicating successful initialization
   }

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
   {
    MainDashboardBody.Destroy(); // Destroying the MainDashboardBody object
    TitleBar.Destroy(); // Destroying the TitleBar object
    TitleText.Destroy(); // Destroying the TitleText object
    LotSizeText.Destroy(); // Destroying the LotSizeText object
    LotSize.Destroy(); // Destroying the LotSize object
    BuyButton.Destroy(); // Destroying the BuyButton object
    SellButton.Destroy(); // Destroying the SellButton object
   }
//+------------------------------------------------------------------+
```

    - Created Button instance named BuyButton
    - Used BuyButton.Create function to create the Edit Object
    - Used BuyButton.SetBorderColor to set border color to Black
    - Used BuyButton.SetBGColor to Set background color to Lime
    - Used BuyButton.SetText to set text Buy
    - Used BuyButton.Destroy to destroy the the Button object in OnDeinit

Now for Sell button:

    - Created Button instance named SellButton
    - Used SellButton.Create function to create the Button Object
    - Used SellButton.SetBorderColor to set border color to Black
    - Used SellButton.SetBGColor to Set background color to Red
    - Used SellButton.SetText to set text Sell
    - Used SellButton.Destroy to destroy the the Button object in OnDeinit

Result:

![Fig 6. Added Buy and Sell Buttons](https://c.mql5.com/2/57/Fig_6._Added_Buy_and_Sell_Buttons.png)

**Fig 6. Added Buy and Sell Buttons**

16. **Finishing touches**

Now for the finishing touches, let's make it colorful. We'll make the following changes:

Let's do the folowing:

    - Change Title Bar color to Dark Blue
    - Change Main Dashboard Body color to Light Blue
    - Change Title Text color to White from Black
    - Change Lot Size Text color to White from Black
    - Add Buy/Sell functionality

The final SimpleTradingEA.mq5 code includes color changes and includes the trading library. It also creates an OnChartEvent function so that when the Buy or Sell button is clicked, the corresponding order is placed immediately.

```
#include "RectangleLabel.mqh" // Including the RectangleLabel class definition
RectangleLabel TitleBar; // Declaration of a TitleBar object
RectangleLabel MainDashboardBody; // Declaration of a MainDashboardBody object

#include "Label.mqh" // Including the Label class definition
Label TitleText; // Declaration of a Label object
Label LotSizeText; // Declaration of a LotSizeText object

#include "Edit.mqh" // Including the Edit class definition
Edit LotSize; // Declaration of a LotSize object

#include "Button.mqh" // Including the Button class definition
Button BuyButton; // Declaration of a BuyButton object
Button SellButton; // Declaration of a SellButton object

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
   {
    TitleBar.Create("TitleBar", 100, 100, 200, 20); // Creating the TitleBar with specified dimensions
    TitleBar.SetBorderType(BORDER_FLAT); // Setting the border type of TitleBar to be flat
    TitleBar.SetBGColor(C'27, 59, 146'); // Setting the color to RGB code: C'27, 59, 146'

    MainDashboardBody.Create("MainDashboardBody", 100, 119, 200, 100); // Creating the MainDashboardBody with specified dimensions
    MainDashboardBody.SetBorderType(BORDER_FLAT); // Setting the border type of MainDashboardBody to be flat
    MainDashboardBody.SetBGColor(C'102, 152, 250'); // Setting the color to RGB code: C'102, 152, 250'

    TitleText.Create("TitleText", 110, 101); // Creating the TitleBar at (110,101)
    TitleText.SetText("Simple Trading EA V1.0"); // Setting its text to "Simple Trading EA V1.0"
    TitleText.SetFontSize(10); // Setting its font size to 10
    TitleText.SetTextColor(clrWhite); // Setting its text color to clrWhite

    LotSizeText.Create("LotSizeText", 110, 140); // Creating the LotSizeText at (110,140)
    LotSizeText.SetText("Lot Size:"); // Setting its text to "Lot Size:"
    LotSizeText.SetFontSize(12); // Setting its font size to 12
    LotSizeText.SetTextColor(clrWhite); // Setting its text color to clrWhite

    LotSize.Create("LotSize", 220, 140, 50, 20); // Creating the LotSize with specified dimensions
    LotSize.SetBorderColor(clrBlack); // Setting its Border Color to clrBlack
    LotSize.SetBGColor(clrWhiteSmoke); // Setting its BG Color to clrWhiteSmoke
    LotSize.SetText("0.01"); // Setting its text to 0.01
    LotSize.SetTextColor(clrBlack); // Setting its text color to clrBlack

    BuyButton.Create("BuyButton", 110, 180, 80, 25); // Creating the BuyButton with specified dimensions
    BuyButton.SetBorderColor(clrBlack); // Setting its Border Color to clrBlack
    BuyButton.SetText("Buy"); // Setting its text to "Buy"
    BuyButton.SetBGColor(clrLime); // Setting its BG Color to clrLime

    SellButton.Create("SellButton", 210, 180, 80, 25); // Creating the SellButton with specified dimensions
    SellButton.SetBorderColor(clrBlack); // Setting its Border Color to clrBlack
    SellButton.SetText("Sell"); // Setting its text to "Sell"
    SellButton.SetBGColor(clrRed); // Setting its BG Color to clrRed

    ChartRedraw(0); // Redrawing the chart to reflect changes
    return(INIT_SUCCEEDED); // Indicating successful initialization
   }

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
   {
    MainDashboardBody.Destroy(); // Destroying the MainDashboardBody object
    TitleBar.Destroy(); // Destroying the TitleBar object
    TitleText.Destroy(); // Destroying the TitleText object
    LotSizeText.Destroy(); // Destroying the LotSizeText object
    LotSize.Destroy(); // Destroying the LotSize object
    BuyButton.Destroy(); // Destroying the BuyButton object
    SellButton.Destroy(); // Destroying the SellButton object
   }

//+------------------------------------------------------------------+
//| Chart event handling function                                    |
//+------------------------------------------------------------------+
void OnChartEvent(const int id, const long& lparam, const double& dparam, const string& sparam)
   {
    // Handles click events for Buy and Sell buttons and opens corresponding positions
    if(id == CHARTEVENT_OBJECT_CLICK) {
        double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
        double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
        if(sparam == "BuyButton") {
            trade.PositionOpen(_Symbol, ORDER_TYPE_BUY, (double)LotSize.GetText(), ask, 0, 0);
        }
        if(sparam == "SellButton") {
            trade.PositionOpen(_Symbol, ORDER_TYPE_SELL, (double)LotSize.GetText(), bid, 0, 0);
        }
    }
   }
//+------------------------------------------------------------------+
```

Changes:

    1. Color Modifications:

       - Changed the Title Bar's background color to dark blue using TitleBar.SetBGColor(C'27, 59, 146') .
       - Updated the Main Dashboard Body's color to light blue with MainDashboardBody.SetBGColor(C'102, 152, 250') .
       - Altered the Title Text's color to white via TitleText.SetTextColor(clrWhite) .
       - Adjusted the Lot Size Text's color to white using LotSizeText.SetTextColor(clrWhite) .
    2. Inclusion of the Trading Library:

       - Integrated the Trading Library and created an instance named trade with the following code:



         ```
         #include <Trade/Trade.mqh>
         CTrade trade;
         ```
    3. Creation of the OnChartEvent Function:



       Implemented an OnChartEvent function that executes a corresponding order immediately when either the Buy or Sell button is clicked. The code is as follows:




       ```
       //+------------------------------------------------------------------+
       //| Chart event handling function                                    |
       //+------------------------------------------------------------------+
       void OnChartEvent(const int id, const long& lparam, const double& dparam, const string& sparam)
          {
           if(id == CHARTEVENT_OBJECT_CLICK) {
               double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
               double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
               if(sparam == "BuyButton") {
                   trade.PositionOpen(_Symbol, ORDER_TYPE_BUY, (double)LotSize.GetText(), ask, 0, 0);
               }
               if(sparam == "SellButton") {
                   trade.PositionOpen(_Symbol, ORDER_TYPE_SELL, (double)LotSize.GetText(), bid, 0, 0);
               }
           }
          }
       //+------------------------------------------------------------------+
       ```

       If the event ID equals CHARTEVENT\_OBJECT\_CLICK, the function detects an Object Click, retrieves the clicked object's name through sparam, checks whether the object name is "BuyButton" or "SellButton", and then places the respective trade using the Trade library.

Final Result:

![Fig 7. Completed Simple Trading EA (Static)](https://c.mql5.com/2/57/Fig_7._Completed_Simple_Trading_EA_rStatic0.png)

**Fig 7. Completed Simple Trading EA (Static)**

This section concludes with that.

### Discussing the Approach to make our static dashboard move with all elements inside it

Now the real work begins. How do we make everything movable? Let's ponder this.

At the moment, we can make any single element movable. But what we need is for all the elements to move. Then, let's make one element move and have all others follow it. We can make other elements literally follow the main element using CustomChartEvent , but unfortunately, that method is slow and thus inefficient. So, what I found to be the most efficient approach is to move our main element (around which all other elements will move) and move other elements simultaneously. That's the theory, but how do we apply it practically?

Let's call our main element the Central Element, and let's make our title bar the Central Element. Now we will move all other elements around it.

Previously, we were moving a single element using a function defined in its class named OnEvent . Now we will modify this function so that it moves a single element and then moves all other elements by exactly the same amount.

Here's our current OnEvent function:

```
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void RectangleLabel::OnEvent(int id, long lparam, double dparam, string sparam)
  {
   //Verify the event that triggered the OnChartEvent was CHARTEVENT_MOUSE_MOVE because we only want to execute out code when that is the case
   if(id == CHARTEVENT_MOUSE_MOVE)
     {
      //define X, Y, XDistance, YDistance, XSize, YSize
      int X = (int)lparam;
      int Y = (int)dparam;
      int MouseState = (int)sparam;

      string name = Name;
      int XDistance = (int)ObjectGetInteger(0, name, OBJPROP_XDISTANCE); //Should be 100 initially as we set it in OnInit()
      int YDistance = (int)ObjectGetInteger(0, name, OBJPROP_YDISTANCE); //Should be 100 initially as we set it in OnInit()
      int XSize = (int)ObjectGetInteger(0, name, OBJPROP_XSIZE); //Should be 200 initially as we set it in OnInit()
      int YSize = (int)ObjectGetInteger(0, name, OBJPROP_YSIZE); //Should be 200 initially as we set it in OnInit()

      if(previousMouseState == 0 && MouseState == 1) //Check if this was the MLB first click
        {
         mlbDownX = X; //Set mlbDownX (Variable that stores the initial MLB X location) equal to the current X
         mlbDownY = Y; //Set mlbDownY (Variable that stores the initial MLB Y location) equal to the current Y
         mlbDownXDistance = XDistance; //Set mlbDownXDistance (Variable that stores the initial XDistance i.e. Width of the dashboard) equal to the current XDistance
         mlbDownYDistance = YDistance; //Set mlbDownYDistance (Variable that stores the initial YDistance i.e. Height of the dashboard) equal to the current YDistance

         if(X >= XDistance && X <= XDistance + XSize && Y >= YDistance && Y <= YDistance + YSize) //Check if the click was on the dashboard
           {
            movingState = true; //If yes the set movingState to True
           }

        }

      if(movingState)//if movingState is true, Update the Dashboard position
        {
         ChartSetInteger(0, CHART_MOUSE_SCROLL, false);//Restrict Chart to be moved by Mouse
         ObjectSetInteger(0, name, OBJPROP_XDISTANCE, mlbDownXDistance + X - mlbDownX);//Update XDistance to: mlbDownXDistance + (X - mlbDownX)
         ObjectSetInteger(0, name, OBJPROP_YDISTANCE, mlbDownYDistance + Y - mlbDownY);//Update YDistance to: mlbDownYDistance + (Y - mlbDownY)
         ChartRedraw(0); //Redraw Chart
        }

      if(MouseState == 0)//Check if MLB is not pressed
        {
         movingState = false;//set movingState again to false
         ChartSetInteger(0, CHART_MOUSE_SCROLL, true);//allow the cahrt to be moved again
        }

      previousMouseState = MouseState;//update the previousMouseState at the end so that we can use it next time and copare it with new value
     }
  }
//+------------------------------------------------------------------+
```

I know we still haven't added this function to the RectangleLabel class; we'll do that after discussing the approach.

Now, what do we need to move any object? Its name, right?

What we are going to do is quite simple: we'll loop through those names and move the objects by the same amount as we moved the central element. But there's a major flaw here that's harder to see.

Whenever the mouse moves, we set the XDis and YDis of the central element like this:

```
ObjectSetInteger(0, name, OBJPROP_XDISTANCE, mlbDownXDistance + X - mlbDownX);//Update XDistance to: mlbDownXDistance + (X - mlbDownX)
ObjectSetInteger(0, name, OBJPROP_YDISTANCE, mlbDownYDistance + Y - mlbDownY);//Update YDistance to: mlbDownYDistance + (Y - mlbDownY)
```

Here, we know the XDis and YDis of the central element when the mouse MLB was pressed. So, we need to know that information for other elements as well. However, this would make the function very complicated or inefficient, so we need a better approach.

Upon closer inspection, a better approach is right in front of us. We simply need to maintain the "X Distance and Y Distance between Central Elements and Other Elements." Yes, it's that simple.

So, we'll note the "X Distance and Y Distance between Central Elements and Other Elements," and maintain that distance. How do we record those distances? Well, at some point, we'll be adding our other elements to the central element, and at that point, we'll note the "X Distance and Y Distance between Central Elements and Other Elements."

To reiterate, at some point, we'll be using the names of the other elements to add them to the central element, and at that point, we'll save the "X Distance and Y Distance between Central Elements and Other Elements." Then we'll maintain this distance between the other elements and the central element. We'll update this distance after updating the central element's position.

That's our approach for the task. Now, let's put it into action.

### Using discussed approach to make our static dashboard movable

So let's discuss where we will store the Name, X Distance, and Y Distance between Central Elements and Other Elements. These are the only two categories of information we need to store.

We will create a function named Add in the RectangleLabel class. Using that function, we will store the following two things:

1. **Name in the addedNames array**
2. **X Distance and Y Distance between Central Elements and Other Elements in addedXDisDifference and addedYDisDifference , respectively**

Regarding the naming conventions, "added" implies the variable is related to another element added to the central element, while "XDis" and "YDis" are fairly straightforward. "Difference" suggests that the variable has something to do with a difference, so it is a reasonable name. The reason for discussing the name is to alleviate any confusion, as the correct variable name can minimize misunderstanding.

Let's declare these variables:

```
string           addedNamed[];
int              addedXDisDiffrence[], addedYDisDiffrence[];
```

Please note that we declare them as Private, as we won't need them to be Public. Also, they are all arrays.

Now let's create the Add function:

```
//+------------------------------------------------------------------+
//| Method to add an object by name to the rectangle label           |
//+------------------------------------------------------------------+
void RectangleLabel::Add(string name)
   {
    ArrayResize(addedNames, ArraySize(addedNames) + 1);
    ArrayResize(addedXDisDiffrence, ArraySize(addedXDisDiffrence) + 1);
    ArrayResize(addedYDisDiffrence, ArraySize(addedYDisDiffrence) + 1);

    addedNames[ArraySize(addedNames) - 1] = name;
    addedXDisDiffrence[ArraySize(addedXDisDiffrence) - 1] = ObjectGetInteger(0, _name, OBJPROP_XDISTANCE) - ObjectGetInteger(0, name, OBJPROP_XDISTANCE);
    addedYDisDiffrence[ArraySize(addedYDisDiffrence) - 1] = ObjectGetInteger(0, _name, OBJPROP_YDISTANCE) - ObjectGetInteger(0, name, OBJPROP_YDISTANCE);
   }
//+------------------------------------------------------------------+
```

This function is declared in the RectangleLabel class, as the TitleBar is our central element, and it is essentially a RECTANGLE\_LABEL object. We declare the variables in the same class, obviously, as we are using them in this function.

What this function does is accept the name as a parameter, then increase the size of those three arrays by one. At the last index, we store the corresponding data. For the Name, we simply store the name. For the Distance Differences (X and Y), we store the difference between the Central element (TitleBar in this case) and the element whose name is provided as a parameter. This constitutes our Add function.

Next, we need to modify the OnEvent function. We create a loop to iterate through the addedNames array and maintain the distance between the TitleBar and the named element, setting it equal to the new TitleBar X/Y Distance minus the difference value given in the respective arrays.

```
for(int i = 0; i < ArraySize(addedNames); i++)
   {
    ObjectSetInteger(0, addedNames[i], OBJPROP_XDISTANCE, mlbDownXDistance + X - mlbDownX - addedXDisDiffrence[i]);
    ObjectSetInteger(0, addedNames[i], OBJPROP_YDISTANCE, mlbDownYDistance + Y - mlbDownY - addedYDisDiffrence[i]);
   }
```

Please note that the underlined part is the new X/Y Distance of the TitleBar (Central Element), and we subtract the difference value given in the respective arrays (referring to the difference between X Distance and Y Distance between Central Elements and Other Elements).

Where do we place this loop? We put it just after the Central Element is updated.

Here's our new OnEvent function:

```
//+------------------------------------------------------------------+
//| Event handling for mouse movements                               |
//+------------------------------------------------------------------+
void RectangleLabel::OnEvent(int id, long lparam, double dparam, string sparam)
   {
    // Handle mouse movement events for dragging the rectangle label
    if(id == CHARTEVENT_MOUSE_MOVE)
       {
        int X = (int)lparam;
        int Y = (int)dparam;
        int MouseState = (int)sparam;

        string name = _name;
        int XDistance = (int)ObjectGetInteger(0, name, OBJPROP_XDISTANCE);
        int YDistance = (int)ObjectGetInteger(0, name, OBJPROP_YDISTANCE);
        int XSize = (int)ObjectGetInteger(0, name, OBJPROP_XSIZE);
        int YSize = (int)ObjectGetInteger(0, name, OBJPROP_YSIZE);

        if(previousMouseState == 0 && MouseState == 1)
           {
            mlbDownX = X;
            mlbDownY = Y;
            mlbDownXDistance = XDistance;
            mlbDownYDistance = YDistance;

            if(X >= XDistance && X <= XDistance + XSize && Y >= YDistance && Y <= YDistance + YSize)
               {
                movingState = true;
               }

           }

        if(movingState)
           {
            ChartSetInteger(0, CHART_MOUSE_SCROLL, false);
            ObjectSetInteger(0, name, OBJPROP_XDISTANCE, mlbDownXDistance + X - mlbDownX);
            ObjectSetInteger(0, name, OBJPROP_YDISTANCE, mlbDownYDistance + Y - mlbDownY);
            for(int i = 0; i < ArraySize(addedNames); i++)
               {
                ObjectSetInteger(0, addedNames[i], OBJPROP_XDISTANCE, mlbDownXDistance + X - mlbDownX - addedXDisDiffrence[i]);
                ObjectSetInteger(0, addedNames[i], OBJPROP_YDISTANCE, mlbDownYDistance + Y - mlbDownY - addedYDisDiffrence[i]);
               }
            ChartRedraw(0);
           }

        if(MouseState == 0)
           {
            movingState = false;
            ChartSetInteger(0, CHART_MOUSE_SCROLL, true);
           }

        previousMouseState = MouseState;
       }
   }
```

The highlighted part is our new loop.

Now we simply need to use the Add function to attach elements to the Central Element, as we have chosen the TitleBar. We use the Add function from the TitleBar instance, which we named "TitleBar."

Let's use the Add function in the TitleBar instance to add all other elements to the TitleBar:

```
// Add the other elements to the Central Element i.e. TitleBar object in this case
TitleBar.Add("MainDashboardBody");
TitleBar.Add("TitleText");
TitleBar.Add("LotSizeText");
TitleBar.Add("LotSize");
TitleBar.Add("BuyButton");
TitleBar.Add("SellButton");
```

With this, the names of all these elements are added to the addedNames array, allowing them to move. Also, their distances from the TitleBar are noted, so that distance will be maintained.

Now, let's make use of the OnEvent function. Without it, all of this would be for naught.

```
// Passes events to the TitleBar object
TitleBar.OnEvent(id, lparam, dparam, sparam);
```

We add this to the OnChartEvent() , and we are done at last. I know this was lengthy, but the final result should be well worth the effort.

![Fig 8. Final Result](https://c.mql5.com/2/57/Fig_8._Final_Result.gif)

**Fig 8. Final Result**

### Conclusion

With this, we come to the end of this article. Throughout our journey in this piece, we've accomplished a great deal, culminating in the completion of our part 3"Improve Your Trading Charts With Interactive GUIs in MQL5."

We have successfully achieved the objectives we set for ourselves in the "Movable GUI" series i.e. Part 1 and Part 2, bringing life to a dynamic and user-friendly interface for trading charts. Thank you for taking the time to read my articles. I hope you find them to be both informative and helpful in your endeavors.

If you have any ideas or suggestions for what you'd like to see in my next piece, please don't hesitate to share.

Happy Coding! Happy Trading!

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/12923.zip "Download all attachments in the single ZIP archive")

[SimpleTradingEA.mq5](https://www.mql5.com/en/articles/download/12923/simpletradingea.mq5 "Download SimpleTradingEA.mq5")(4.3 KB)

[RectangleLabel.mqh](https://www.mql5.com/en/articles/download/12923/rectanglelabel.mqh "Download RectangleLabel.mqh")(5.75 KB)

[Label.mqh](https://www.mql5.com/en/articles/download/12923/label.mqh "Download Label.mqh")(2.35 KB)

[Edit.mqh](https://www.mql5.com/en/articles/download/12923/edit.mqh "Download Edit.mqh")(2.53 KB)

[Button.mqh](https://www.mql5.com/en/articles/download/12923/button.mqh "Download Button.mqh")(2.31 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Custom Debugging and Profiling Tools for MQL5 Development (Part I): Advanced Logging](https://www.mql5.com/en/articles/17933)
- [Building a Custom Market Regime Detection System in MQL5 (Part 2): Expert Advisor](https://www.mql5.com/en/articles/17781)
- [Building a Custom Market Regime Detection System in MQL5 (Part 1): Indicator](https://www.mql5.com/en/articles/17737)
- [Advanced Memory Management and Optimization Techniques in MQL5](https://www.mql5.com/en/articles/17693)
- [Mastering JSON: Create Your Own JSON Reader from Scratch in MQL5](https://www.mql5.com/en/articles/16791)
- [Mastering File Operations in MQL5: From Basic I/O to Building a Custom CSV Reader](https://www.mql5.com/en/articles/16614)
- [Modified Grid-Hedge EA in MQL5 (Part IV): Optimizing Simple Grid Strategy (I)](https://www.mql5.com/en/articles/14518)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/452127)**
(21)


![Aleksandr Slavskii](https://c.mql5.com/avatar/2017/4/58E88E5E-2732.jpg)

**[Aleksandr Slavskii](https://www.mql5.com/en/users/s22aa)**
\|
10 Mar 2024 at 17:13

**Stanislav Korotky [#](https://www.mql5.com/ru/forum/458272#comment_52675372):**

The most practical and universal option is to design forms visually, without worrying about the layout code at all - that's exactly what classes should do. One of the possible solutions was in the [article](https://www.mql5.com/en/articles/7795).

Stanislav, well, under the article for beginners you provide a link to the article for advanced users :)

I read it, not read it, but just read it, because my level in programming is much lower than I need to understand what is written there.

I try to use what I don't understand as little as possible. A recent situation has rooted me in this even more strongly.

You are probably reading the topic "Errors, bugs, questions", so I had a task to display arrows from closed positions on a renko-graph, entry-exit between them - a line.

I decided not to write it myself, but to take a ready-made code from Saiber, as a result - half a day of wasted time. Saiber corrected its code in the end, but I lost time.

And if I wanted to take the code from your article and needed to tweak something, well, you understand, nothing good would come out of it.

I have a simpler task than you set for yourself when writing your article. I just need to make it so that I can build the next panel from pieces of ready-made code like children build houses from lego.

I don't need OOP for this purpose at all, in my opinion. I don't know it, so I don't like it.

The MVC principle is quite suitable for my purposes, if I understood it correctly)))))

In general, I already have a picture of how it should be.

By the way, can you tell me how to make it so that when I access the functions of the inheritor class from the programme, I don't see the functions of the base class? Well, if it is possible.

![Aleksandr Slavskii](https://c.mql5.com/avatar/2017/4/58E88E5E-2732.jpg)

**[Aleksandr Slavskii](https://www.mql5.com/en/users/s22aa)**
\|
10 Mar 2024 at 17:20

**Maxim Kuznetsov [#](https://www.mql5.com/ru/forum/458272/page2#comment_52675516):**

it came to mind :-)

for those who are not for the market, want more or less complex but beautiful, and it is too lazy to write a DLL, there is GtkServer [https://sourceforge.net/projects/gtk-server/](https://www.mql5.com/go?link=https://sourceforge.net/projects/gtk-server/ "https://sourceforge.net/projects/gtk-server/") and Glade form designer to it.

Methodology: GtkServer is launched as a tcp listener, advisor using SocketOpen/SocketSend sends text "load form" (or itself by steps forms gtk widgets) and also reads the result....

You too :)

I generallyperceive words like tcp listener, SocketOpen/SocketSend as profanity,  Idon't even know their meaning without google, and you suggest to use it.

Gentlemen, have a conscience, stop scaring people with your terminology)))))

![Stanislav Korotky](https://c.mql5.com/avatar/2010/10/4CA7CFA0-1F0C.jpg)

**[Stanislav Korotky](https://www.mql5.com/en/users/marketeer)**
\|
11 Mar 2024 at 12:10

**Aleksandr Slavskii [#](https://www.mql5.com/ru/forum/458272/page2#comment_52676114):**

Stanislav, well yo-ma-yo, under the article for beginners you throw a link to the article for advanced :)

I have read it, not read it, but just read it, because my level in programming is much lower than I need to understand what is written there.

I try to use what I don't understand as little as possible. A recent situation has rooted me in this even more strongly.

Unfortunately, the situation with software is such that it is impossible to understand everything yourself. According to this logic, the operating system should be customised (which is what some Linux apologists do).

That's why the modern approach is to take ready-made bricks (which do more or less what you need) and dock them (without going deep into implementation).

I was merely answering the question of how to do more universally. You can't argue with the fact that my article is complicated, but the point is that you can describe the necessary GUI as a template without programming, just by plugging in enclosures to get controllers, window dragging, resizing and so on.

![Frist001](https://c.mql5.com/avatar/avatar_na2.png)

**[Frist001](https://www.mql5.com/en/users/frist001)**
\|
3 May 2024 at 04:30

Excellent, very good, I can tinker with what I can't code, but there are some features I can't handle, is it easy to help me modify? Add an input box with a +- sign. Thank you!


![Retail Trading Realities LTD](https://c.mql5.com/avatar/2025/4/68116106-adc9.png)

**[Philip Kym Sang Nelson](https://www.mql5.com/en/users/rtr_ltd)**
\|
27 Oct 2024 at 00:01

Fantastic !

I couldn't figure it out on my own,

yes read it twice .

Thanks !

Philip

![Category Theory in MQL5 (Part 16): Functors with Multi-Layer Perceptrons](https://c.mql5.com/2/57/category-theory-p16-avatar.png)[Category Theory in MQL5 (Part 16): Functors with Multi-Layer Perceptrons](https://www.mql5.com/en/articles/13116)

This article, the 16th in our series, continues with a look at Functors and how they can be implemented using artificial neural networks. We depart from our approach so far in the series, that has involved forecasting volatility and try to implement a custom signal class for setting position entry and exit signals.

![The RSI Deep Three Move Trading Technique](https://c.mql5.com/2/57/The_RSI_Deep_Three_Move_avatar.png)[The RSI Deep Three Move Trading Technique](https://www.mql5.com/en/articles/12846)

Presenting the RSI Deep Three Move Trading Technique in MetaTrader 5. This article is based on a new series of studies that showcase a few trading techniques based on the RSI, a technical analysis indicator used to measure the strength and momentum of a security, such as a stock, currency, or commodity.

![Category Theory in MQL5 (Part 17): Functors and Monoids](https://c.mql5.com/2/57/Category-Theory-p17-avatar.png)[Category Theory in MQL5 (Part 17): Functors and Monoids](https://www.mql5.com/en/articles/13156)

This article, the final in our series to tackle functors as a subject, revisits monoids as a category. Monoids which we have already introduced in these series are used here to aid in position sizing, together with multi-layer perceptrons.

![Developing a Replay System — Market simulation (Part 04): adjusting the settings (II)](https://c.mql5.com/2/52/replay-p4-avatar.png)[Developing a Replay System — Market simulation (Part 04): adjusting the settings (II)](https://www.mql5.com/en/articles/10714)

Let's continue creating the system and controls. Without the ability to control the service, it is difficult to move forward and improve the system.

[Launching MetaTrader VPS for the first time?Read our comprehensive, step-by-step instructions![](https://www.mql5.com/ff/sh/0xb0c8bjq5sadh89z2/01.png)Read](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/articles/13586&a=gxygkojxdwrcfbbgfrchvjgelflsnelu&s=49eab2fb45d89f59a191e88145774dcd7f9533039acb10dd9c28061b04fa92fe&uid=&ref=https://www.mql5.com/en/articles/12923&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5069097252661690478)

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