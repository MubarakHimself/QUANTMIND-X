---
title: 100 best optimization passes (part 1). Developing optimization analyzer
url: https://www.mql5.com/en/articles/5214
categories: Trading Systems, Expert Advisors
relevance_score: 6
scraped_at: 2026-01-23T11:49:12.637179
---

[![](https://www.mql5.com/ff/sh/0uquj7zv5pmx2m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
Market analytics in MQL5 Channels\\
\\
Tens of thousands of traders have chosen this messaging app to receive trading tips.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=epadtzgppsywkaeumqycnulasoijfbgz&s=9615c3e5c371aa0d7b34529539d05c10df73b35a1e2213e4ceee008933c7ede0&uid=&ref=https://www.mql5.com/en/articles/5214&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5062742449115277317)

MetaTrader 5 / Tester


- [Introduction](https://www.mql5.com/en/articles/5214#para1)
- [Optimization analyzer structure](https://www.mql5.com/en/articles/5214#para2)
- [Graphics](https://www.mql5.com/en/articles/5214#para3)
- [Working with the database](https://www.mql5.com/en/articles/5214#para4)
- [Calculations](https://www.mql5.com/en/articles/5214#para5)
- [Presenter](https://www.mql5.com/en/articles/5214#para6)
- [Conclusion](https://www.mql5.com/en/articles/5214#para7)

### Introduction

Modern technology has now become so deeply ingrained into the field of financial trading that it is now almost impossible to imagine how we could do without it. Nevertheless, just a very short while ago, trading was conducted manually and there was a complex system of hand language (quickly heading into oblivion nowadays) ​​describing how much asset to buy or sell.

Personal computers rapidly superseded traditional trading methods by bringing online trading literally into our homes. Now we can look at asset quotes in real time and make appropriate decisions. Moreover, the advent of online technologies in the market industry causes the ranks of manual traders to dwindle at an increasing speed. Now, more than half of the deals are made by trading algorithms, and it is worth to say that MetaTrader 5 is number one among the most convenient terminals for this.

But despite all the advantages of this platform, it has a number of drawbacks I tried to mitigate with the application described here. The article describes developing the program written entirely in MQL5 using the [EasyAndFastGUI](https://www.mql5.com/en/code/19703) library designed to improve selection of trading algorithm optimization parameters. It also adds new features to the analysis of retrospective trading and general EA assessment.

Article 5214 - YouTube

[Photo image of MQL5.community](https://www.youtube.com/channel/UC8bwZMk2yh5WDToFos49OjQ?embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F5214)

MQL5.community

1.91K subscribers

[Article 5214](https://www.youtube.com/watch?v=Nk2a0-6JspY)

MQL5.community

Search

Watch later

Share

Copy link

Info

Shopping

Tap to unmute

If playback doesn't begin shortly, try restarting your device.

More videos

## More videos

You're signed out

Videos you watch may be added to the TV's watch history and influence TV recommendations. To avoid this, cancel and sign in to YouTube on your computer.

CancelConfirm

Share

Include playlist

An error occurred while retrieving sharing information. Please try again later.

[Watch on](https://www.youtube.com/watch?v=Nk2a0-6JspY&embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F5214)

0:00

0:00 / 11:04

•Live

•

First, optimization of EAs takes quite a long time. Of course, this is due to the fact that the tester generates ticks in a more high-quality manner (even when OHLC is selected, four ticks are generated for each candle), as well as other additions that allow for better EA evaluation. However, on home PCs that are not so powerful, optimization can take several days or weeks. It often happens that after choosing the EA parameters, we soon realize that they are incorrect, and there is nothing at hand besides optimization passes statistics and a few evaluation ratios.

It would be nice to have a full-fledged statistics per each optimization pass and filtration ability (including conditional filters) for each of them by multiple parameters. It would also be good to compare trading statistics with the Buy And Hold strategy and impose all statistics on each other. In addition, it is sometimes necessary to upload all the trading history data to a file for the subsequent processing of results of each deal.

Sometimes, we may also want to see what kind of slippage the algorithm is able to withstand and how the algorithm behaves on a certain time interval, since some strategies depend on the market type. A flat-based strategy can serve as an example. It loses during trend periods and makes profit during flat ones. It would also be good to view certain intervals (by dates) as a complete set of ratios and other additions (rather than simply on a price chart) separately from the general PL graph.

We should also pay attention to forward tests. They are very informative, but their graphs are displayed as a continuation of the previous one in the strategy tester's standard report. Novice traders may easily conclude that their robot sharply lost all profits and then started to recover (or worse — went negative). In the program described here, all data are reviewed in terms of optimization type (either forward, or history one).

It is also important to mention Grails many EA developers are so fond of searching for. Some robots make 1000% or more per month. It may seem that they outrun the market (Buy and Hold strategy), but in real practice, everything looks quite different. As the described program shows, these robots can really make 1000%, but they do not outrun the market.

The program features separation of an analysis between trading using a robot with a full lot (increasing/reducing it, etc…), as well as imitation of a trading by the robot using a single lot (minimum lot available for trading). When building the Buy and Hold trading graph, the described program considers lot management performed by the robot (i.e. it purchases some more asset when a lot is increased and reduces the amount of a purchased asset when a lot is decreased). If we compare these two graphs, it turns out that my test robot, which showed unrealistic results in one of its best optimization passes, could not outrun the market. Therefore, for more objective assessment of trading strategies, we should have a look at the one-lot trading graph, in which both the robot's and the Buy and Hold strategy's PL are displayed as if trading with the least allowable trading volume (PL= Profit/Loss — graph of obtained profit by time).

Now, let's have a more detailed look at how the program was developed.

### Optimization analyzer structure

The program structure can be graphically expressed as follows:

![](https://c.mql5.com/2/34/fig1__2.png)

The resulting optimization analyzer is not tied to any particular robot and is not part of it. However, due to the specifics of building graphical interfaces in MQL5, the MQL5 EA development template was used as the program's basis. Since the program turned out to be quite large (several thousands lines of code), for more specificity and consistency, it was divided into a number of blocks (displayed on the diagram above) that were in turn divided into classes. The robot template is only the starting point for launching the application. Each of the blocks will be considered in more details below. Here we will describe the relationships between them. To work with the application, we will need:

- The trading algorithm
- Dll Sqlite3
- The [graphical interface library](https://www.mql5.com/en/code/19703) mentioned above with necessary edits (described in the graphics block below)

The robot itself can be developed in any way you like (using OOP, a function inside the robot template, imported from Dll…). Most importantly, it should apply the robot development template provided by MQL5 Wizard. It connects one file from the database block where the class uploading required data to the database after each optimization pass is located. This part is independent and does not depend on the application itself, since the database is formed when launching the robot in the strategy tester.

**The calculation block** is an improved continuation of my previous article ["Custom presentation of trading history and creation of report diagrams"](https://www.mql5.com/en/articles/4803).

Database and calculation block are used both in the analyzed robot and in the described application. Therefore, they are placed into the Include directory. These blocks perform the bulk of the work and are connected to the graphical interface via the presenter class.

**Presenter class** connects separate program blocks. Each of the blocks has its own function in the graphical interface. It handles button pressing and other events, as well as redirects to other logical blocks. The data obtained from them are returned to the presenter, where they are processed and the appropriate graphs are plotted, tables are filled, and other interaction with the graphic part takes place.

The **graphical part** of the program does not perform any conceptual logic. Instead, it only builds a window with the required interface and calls the appropriate presenter functions during the button pressing events.

**The program itself** is written as the [MQL5 project](https://www.metatrader5.com/en/metaeditor/help/mql5storage/projects "https://www.metatrader5.com/en/metaeditor/help/projects") allowing you to develop it in a more structured way and put all the required files with code in one place. The project features yet another class that will be described in the calculation block. This class was written specifically for this program. It sorts optimization passes using the method I developed. In fact, it serves the entire "Optimisation selection" tab reducing data sampling by certain criteria.

**Universal sorting class** is an independent addition to the program. It dies not fit into any of the blocks, but it still remains an important part of the program. Therefore, we briefly consider it in this part of the article.

As the name implies, the class deals with data sorting. Its algorithm was taken from a third-party website — [Selection sort](https://www.mql5.com/go?link=https://function-x.ru/cpp_algoritmy_sortirovki.html%23paragraph1 "https://function-x.ru/cpp_algoritmy_sortirovki.html#paragraph1") (in Russian).

```
//+------------------------------------------------------------------+
//| E-num with a sorting style                                       |
//+------------------------------------------------------------------+
enum SortMethod
  {
   Sort_Ascending,// Ascending
   Sort_Descendingly// Descending
  };
//+------------------------------------------------------------------+
//| Class sorting the passed data type                               |
//+------------------------------------------------------------------+
class CGenericSorter
  {
public:
   // Default constructor
                     CGenericSorter(){method=Sort_Descendingly;}
   // Sorting method
   template<typename T>
   void              Sort(T &out[],ICustomComparer<T>*comparer);
   // Select sorting type
   void Method(SortMethod _method){method=_method;}
   // Get sorting method
   SortMethod Method(){return method;}
private:
   // Sorting method
   SortMethod        method;
  };
```

The class contains the template Sort method, which sorts the data. The template method allows sorting any passed data, including classes and structures. The data comparison method should be described in a separate class that implements the IСustomComparer<T> interface. I had to develop my own interface of IСomparer type only because in the conventional IСomparer interface of the Compare method, comprised data are not passed by reference, while passing by reference is one of the conditions of passing structures to a method in MQL5 language.

CGenericSorter::Method class method overloads return and accept data sorting type (in ascending or descending order). This class is used in all blocks of the program where the data are sorted.

### Graphics

| **_Warning!_** <br>_When developing the graphical interface, a bug was detected in the applied library (EasyAndFastGUI) — the ComboBox graphical element cleared some variables incompletely during its refilling. According to the [recommendations](https://www.mql5.com/ru/forum/225047/page10) (in Russian) of the library developer, the following changes should be made to fix this:_<br>_m\_item\_index\_focus =WRONG\_VALUE;_ _m\_prev\_selected\_item =WRONG\_VALUE;_ _m\_prev\_item\_index\_focus =WRONG\_VALUE;_<br>_to the method CListView::Clear(const bool redraw=false)._<br>_The method is located on the 600 th string of the ListView.mqh file. The file's path:_<br>_Include\\EasyAndFastGUI\\Controls._<br>_If you do not add these edits, the "Array out of range" error will sometimes pop up while opening ComboBox and the application will close abnormally._ |
| --- |

To create a window in MQL5 based on the EasyAndFastGUI library, a class is required that will serve as a container for all subsequent window filling. The class should be derived from the CwindEvents class. The methods should be redefined inside the class:

```
 //--- Initialization/deinitialization
   void              OnDeinitEvent(const int reason){CWndEvents::Destroy();};
   //--- Chart event handler
   virtual void      OnEvent(const int id,const long &lparam,const double &dparam,const string &sparam);//
```

The blank for creating the window should be as follows:

```
class CWindowManager : public CWndEvents
  {
public:
                     CWindowManager(void){presenter = NULL;};
                    ~CWindowManager(void){};
   //===============================================================================
   // Calling methods and events :
   //===============================================================================
   //--- Initialization/deinitialization
   void              OnDeinitEvent(const int reason){CWndEvents::Destroy();};
   //--- Chart event handler
   virtual void      OnEvent(const int id,const long &lparam,const double &dparam,const string &sparam);

//--- Create the program's graphical interface
   bool              CreateGUI(void);

private:
 //--- Main window
   CWindow           m_window;
  }
```

The window itself is created with the Cwindow type inside the class. However, a number of window properties should be defined before displaying the window. In this particular case, the window creation method looks as follows:

```
bool CWindowManager::CreateWindow(const string text)
  {
//--- Add the window pointer to the window array
   CWndContainer::AddWindow(m_window);
//--- Coordinates
   int x=(m_window.X()>0) ? m_window.X() : 1;
   int y=(m_window.Y()>0) ? m_window.Y() : 1;
//--- Properties
   m_window.XSize(WINDOW_X_SIZE+25);
   m_window.YSize(WINDOW_Y_SIZE);
   m_window.Alpha(200);
   m_window.IconXGap(3);
   m_window.IconYGap(2);
   m_window.IsMovable(true);
   m_window.ResizeMode(false);
   m_window.CloseButtonIsUsed(true);
   m_window.FullscreenButtonIsUsed(false);
   m_window.CollapseButtonIsUsed(true);
   m_window.TooltipsButtonIsUsed(false);
   m_window.RollUpSubwindowMode(true,true);
   m_window.TransparentOnlyCaption(true);

//--- Set tooltips
   m_window.GetCloseButtonPointer().Tooltip("Close");
   m_window.GetFullscreenButtonPointer().Tooltip("Fullscreen/Minimize");
   m_window.GetCollapseButtonPointer().Tooltip("Collapse/Expand");
   m_window.GetTooltipButtonPointer().Tooltip("Tooltips");
//--- Create the form
   if(!m_window.CreateWindow(m_chart_id,m_subwin,text,x,y))
      return(false);
//---
   return(true);
  }
```

The prerequisites for this method are the string that adds the window to the array of application windows and creating the form. Later, when the application is running and the OnEvent event is triggered, one of the library methods runs in a loop over all the windows listed in the array of windows. Then it goes over all the elements inside the window and looks for an event related to clicking on any management interface or highlighting a table row, etc. Therefore, when creating each new application window, a reference should be added to that window in the reference array.

The developed application featues the interface divided by tabs. There are 4 tab containers:

```
//--- Tabs
   CTabs             main_tab; // Main tabs
   CTabs             tab_up_1; // Tabs with settings and results table
   CTabs             tab_up_2; // Tabs with statistics and selection of parameters, as well as common graphs
   CTabs             tab_down; // Tabs with statistics and uploading to a file
```

They look as follows on the form (signed in red on the screenshot):

![](https://c.mql5.com/2/34/t7hu_4.PNG)

- main\_tab divides the table with all selected optimization passes ("Optimisation Data") from the rest of the program interface. This table contains all the results that satisfy the filter conditions on the settings tab. The results are then sorted by the ratio selected in ComboBox — Sort by. The obtained data are transferred to the described table in the sorted form. The tab with the rest of the program interface contains another 3 Tab containers.
- tab\_up\_1 contains a division into the initial settings of the program and the table with the sorted results. In addition to the mentioned conditional filters, the Settings tab serves for selecting the database and entering additional data. For example, you can select whether to enter all data already added to the Optimisation Data tab of the table to the data selection results table, or only a certain number of the best parameters (filtering in descending order by selected ratio) will be sufficient.
- tab\_up\_2 contains 3 tabs. Each of them contains the interface performing three different types of tasks. The first tab contains the full report on a selected optimization pass and allows simulating slippage, as well as considering trading history for a certain time period. The second one serves as the filter for optimization passes and helps define the strategy sensitivity to different parameters and narrow down the number of optimization results by selecting the most adequate intervals of the parameters of interest. The last tab serves as a graphical representation of the optimization results table and shows the total number of selected optimization parameters.
- tab\_down features five tabs, four of which is presentation of an EA's trading report during optimization with selected parameters, while the last tab is uploading data to a file. The first tab presents a table with estimated ratios. The second tab provides profit/loss distribution by trading days. The third tab represents profit and loss graph imposed on the Buy and Hold strategy (black graph), while the fourth tab represents changes in some selected ratios over time, as well as some additional interesting and informative types of graphs that can be obtained by analyzing the EA trading results.

The process of creating tabs is similar — the only difference is the content. As an example, I will provide the method of creating the main tab:

```
//+------------------------------------------------------------------+
//| Main Tab                                                         |
//+------------------------------------------------------------------+
bool CWindowManager::CreateTab_main(const int x_gap,const int y_gap)
  {
//--- Save the pointer to the main element
   main_tab.MainPointer(m_window);

//--- Array of tab width
   int tabs_width[TAB_MAIN_TOTAL];
   ::ArrayInitialize(tabs_width,45);
   tabs_width[0]=120;
   tabs_width[1]=120;
//---
   string tabs_names[TAB_UP_1_TOTAL]={"Analysis","Optimisation Data"};
//--- Properties
   main_tab.XSize(WINDOW_X_SIZE-23);
   main_tab.YSize(WINDOW_Y_SIZE);
   main_tab.TabsYSize(TABS_Y_SIZE);
   main_tab.IsCenterText(true);
   main_tab.PositionMode(TABS_LEFT);
   main_tab.AutoXResizeMode(true);
   main_tab.AutoYResizeMode(true);
   main_tab.AutoXResizeRightOffset(3);
   main_tab.AutoYResizeBottomOffset(3);
//---
   main_tab.SelectedTab((main_tab.SelectedTab()==WRONG_VALUE)? 0 : main_tab.SelectedTab());
//--- Add tabs with specified properties
   for(int i=0; i<TAB_MAIN_TOTAL; i++)
      main_tab.AddTab((tabs_names[i]!="")? tabs_names[i]: "Tab "+string(i+1),tabs_width[i]);
//--- Create a control element
   if(!main_tab.CreateTabs(x_gap,y_gap))
      return(false);
//--- Add an object to the common array of object groups
   CWndContainer::AddToElementsArray(0,main_tab);
   return(true);
  }
```

In addition to content that may vary, the main code strings are as follows:

1. Adding a pointer to the main element — the tab container should know the element it is assigned to
2. Control element creation string
3. Adding an element to the general list of controls.

The control elements are next according to the hierarchy. 11 control element types were used in the application. They are all created in a similar manner, therefore the methods adding the control elements have been written to create each of them. Let's consider the implementation of only one of them:

```
bool CWindowManager::CreateLable(const string text,
                                 const int x_gap,
                                 const int y_gap,
                                 CTabs &tab_link,
                                 CTextLabel &lable_link,
                                 int tabIndex,
                                 int lable_x_size)
  {
//--- Save the pointer to the main element
   lable_link.MainPointer(tab_link);
//--- Assign to the tab
   tab_link.AddToElementsArray(tabIndex,lable_link);

//--- Settings
   lable_link.XSize(lable_x_size);

//--- Creating
   if(!lable_link.CreateTextLabel(text,x_gap,y_gap))
      return false;

//--- Add an object to the general object groups array
   CWndContainer::AddToElementsArray(0,lable_link);
   return true;
  }
```

The passed control element (CTextLabel), together with the tabs, should remember the element it is assigned to as a container. In turn, the tab container remembers the tab the element is located at. After that, the element is filled with required settings and initial data. Eventually, the object is added to the general array of objects.

Similar to labels, other elements defined inside the class container as fields are added. I separated certain elements and placed some of them to the 'protected' class area. These are the elements that do not require access via the presenter. Some other elements were placed to 'public'. These are the elements defining some conditions or radio buttons, the state of which should be checked from the presenter. In other words, all elements and methods, access to which is not desirable, have their headers in the 'protected' or 'private' parts of the class together with the reference to the presenter. Adding the presenter reference is made in the form of a public method where the presence of an already added presenter is checked first, and if the reference to it has not been added yet, the presenter is saved. This is done to avoid dynamic presenter substitution during the program execution.

The window itself is created in the CreateGUI method:

```
bool CWindowManager::CreateGUI(void)
  {
//--- Create window
   if(!CreateWindow("Optimisation Selection"))
      return(false);

//--- Create tabs
   if(!CreateTab_main(120,20))
      return false;
   if(!CreateTab_up_1(3,44))
      return(false);
   int indent=WINDOW_Y_SIZE-(TAB_UP_1_BOTTOM_OFFSET+TABS_Y_SIZE-TABS_Y_SIZE);
   if(!CreateTab_up_2(3,indent))
      return(false);
   if(!CreateTab_down(3,33))
      return false;

//--- Create controls
   if(!Create_all_lables())
      return false;
   if(!Create_all_buttons())
      return false;
   if(!Create_all_comboBoxies())
      return false;
   if(!Create_all_dropCalendars())
      return false;
   if(!Create_all_textEdits())
      return false;
   if(!Create_all_textBoxies())
      return false;
   if(!Create_all_tables())
      return false;
   if(!Create_all_radioButtons())
      return false;
   if(!Create_all_SepLines())
      return false;
   if(!Create_all_Charts())
      return false;
   if(!Create_all_CheckBoxies())
      return false;

// Show window
   CWndEvents::CompletedGUI();

   return(true);
  }
```

As can be seen from its implementation, it does not directly create any control element itself, but only calls other methods for creating these elements. The main code string that should be included as a final one in this method is CWndEvents::CompletedGUI();

This string completes graphics creation and plots it on a user's screen. Creation of each control element (be it separation lines, labels or buttons) is implemented into methods having a similar content and applying the above mentioned approaches to creating graphical control elements. The method headers can be found in the 'private' part of the class:

```
//===============================================================================
// Controls creation:
//===============================================================================
//--- All Labels
   bool              Create_all_lables();
   bool              Create_all_buttons();
   bool              Create_all_comboBoxies();
   bool              Create_all_dropCalendars();
   bool              Create_all_textEdits();
   bool              Create_all_textBoxies();
   bool              Create_all_tables();
   bool              Create_all_radioButtons();
   bool              Create_all_SepLines();
   bool              Create_all_Charts();
   bool              Create_all_CheckBoxies();
```

Talking about graphics, it is impossible to skip the event model part. For correct processing in graphic applications developed using EasyAndFastGUI, you will need to perform the following steps:

Create the event handler method (for example, button pressing). This method should accept 'id' and 'lparam' as parameters. The first parameter indicates the type of a graphical event, while the second indicates ID of an object the interaction took place with. Implementation of the methods is similar in all cases:

```
//+------------------------------------------------------------------+
//| Btn_Update_Click                                                 |
//+------------------------------------------------------------------+
void CWindowManager::Btn_Update_Click(const int id,const long &lparam)
  {
   if(id==CHARTEVENT_CUSTOM+ON_CLICK_BUTTON && lparam==Btn_update.Id())
     {
      presenter.Btn_Update_Click();
     }
  }
```

First, check the condition (whether the button was pressed or the list element was selected…). Next, check lparam where ID passed to the method is compared to ID of the required list element.

All button pressing event declarations are located in the 'private' part of the class. The event should be called to obtain a respond to it. Declared events are called in the overloaded OnEvent method:

```
//+------------------------------------------------------------------+
//| OnEvent                                                          |
//+------------------------------------------------------------------+
void CWindowManager::OnEvent(const int id,const long &lparam,const double &dparam,const string &sparam)
  {
   Btn_Update_Click(id,lparam);
   Btn_Load_Click(id,lparam);
   OptimisationData_inMainTable_selected(id,lparam);
   OptimisationData_inResults_selected(id,lparam);
   Update_PLByDays(id,lparam);
   RealPL_pressed(id,lparam);
   OneLotPL_pressed(id,lparam);
   CoverPL_pressed(id,lparam);
   RealPL_pressed_2(id,lparam);
   OneLotPL_pressed_2(id,lparam);
   RealPL_pressed_4(id,lparam);
   OneLotPL_pressed_4(id,lparam);
   SelectHistogrameType(id,lparam);
   SaveToFile_Click(id,lparam);
   Deals_passed(id,lparam);
   BuyAndHold_passed(id,lparam);
   Optimisation_passed(id,lparam);
   OptimisationParam_selected(id,lparam);
   isCover_clicked(id,lparam);
   ChartFlag(id,lparam);
   show_FriquencyChart(id,lparam);
   FriquencyChart_click(id,lparam);
   Filtre_click(id,lparam);
   Reset_click(id,lparam);
   RealPL_pressed_3(id,lparam);
   OneLotPL_pressed_3(id,lparam);
   ShowAll_Click(id,lparam);
   DaySelect(id,lparam);
  }
```

The method, in turn, is called from the robot template. Thus, the event model stretches from the robot template (provided below) to the graphical interface. GUI performs all processing, sorting out and redirection for subsequent handling in the presenter. The robot template itself is a starting point of the program. It looks as follows:

```
#include "Presenter.mqh"

CWindowManager _window;
CPresenter Presenter(&_window);
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---
   if(!_window.CreateGUI())
     {
      Print(__FUNCTION__," > Failed to create the graphical interface!");
      return(INIT_FAILED);
     }
//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---
   _window.OnDeinitEvent(reason);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void OnChartEvent(const int    id,
                  const long   &lparam,
                  const double &dparam,
                  const string &sparam)
  {
   _window.ChartEvent(id,lparam,dparam,sparam);
  }
//+------------------------------------------------------------------+
```

### Working with the database

Before considering this rather extensive part of the project, it is worth saying a few words regarding the choice made. One of the initial project objectives was to provide ability to work with optimization results after completing the optimization itself, as well as availability of these results at any time. Saving data to a file was immediately discarded as unsuitable. It would require creation of multiple tables (forming, in fact, a single large table, but with a different number of rows) or files.

Neither is very convenient. Besides, the method is more difficult to implement. The second method is creating [optimization frames](https://www.mql5.com/en/docs/optimization_frames). The toolkit itself is good but we are not going to work with optimizations during the optimization process. Besides, the frames functionality is not as good as the database one. In addition, frames are designed for MetaTrader, while the database can be used in any third-party analytical program if required.

Selecting the right database was easy enough. We needed a fast and popular database that would be convenient to link up with and not requiring any additional software. [Sqlite](https://www.mql5.com/go?link=https://sqlite.org/index.html "https://sqlite.org/index.html") database meets all that criteria. The mentioned characteristics make it so popular. To use it, connect databases supplied by the provider to the Dll project. Dll data are written in C and are easily linked up with MQL5 applications, which is a nice addition since you do not have to write a single code line in a third-party language complicating the project. Among the disadvantages of this approach is that Dll Sqlite does not provide a convenient API for working with the database, and therefore it is required to describe at least the minimum wrapper for working with the database. An example of writing this functionality was efficiently presented in the article ["SQL and MQL5: Working with SQLite database"](https://www.mql5.com/en/articles/862). For this project, a part of the code related to interaction with WinApi and importing some functions from dll to MQL5 from the mentioned article was used. As for the wrapper, I decided to write it myself.

As a result, the database handling block consists of the Sqlite3 folder, where a convenient wrapper for working with the database is described, and the OptimisationSelector folder created specifically for the developed program. Both folders are located in the MQL5/Include directory. As mentioned earlier, a number of functions of the Windows standard library are used for working with the database. All functions of this part of the application are located in the WinApi directory. In addition to the mentioned borrowings, I also used the code for creating a shared resource ( [Mutex](https://www.mql5.com/en/code/1835)) from CodeBase. When working with the database from two sources (namely, if the optimization analyzer opens the database used during the optimization), data obtained by the program should always be complete. This is why a shared resource is required. It turns out that if one of the sides (optimization process or analyzer) activates the database, the second one waits till its counterpart completes its work. Sqlite database allows reading it from several threads. Due to the subject matter of the article, we will not consider in detail the resulting wrapper for working with the sqlite3 database from MQL5. Instead, we only describe some points of its implementation and application methods. As already mentioned, the wrapper of working with the database is located in the Sqlite3 folder. There are three files in it. Let's go over them in writing order.

- The first thing we need is to import the necessary functions for working with the database from the Dll. Since the goal was to create a wrapper containing the minimum required functionality, I did not import even 1% of the total number of functions provided by the database developers. All required functions are imported in the sqlite\_amalgmation.mqh file. These functions are well commented on the developer’s website, and are also labeled in the above file. If desired, you can import the entire header file the same way. The result will be a complete list of all functions and, accordingly, the ability to access them. The list of the imported functions is as follows:

```
#import "Sqlite3_32.dll"
int sqlite3_open(const uchar &filename[],sqlite3_p32 &paDb);// Open the database
int sqlite3_close(sqlite3_p32 aDb); // Close the database
int sqlite3_finalize(sqlite3_stmt_p32 pStmt);// Complete the statement
int sqlite3_reset(sqlite3_stmt_p32 pStmt); // Reset the statement
int sqlite3_step(sqlite3_stmt_p32 pStmt); // Move to the next line when reading the statement
int sqlite3_column_count(sqlite3_stmt_p32 pStmt); // Calculate the number of columns
int sqlite3_column_type(sqlite3_stmt_p32 pStmt,int iCol); // Get the type of the selected column
int sqlite3_column_int(sqlite3_stmt_p32 pStmt,int iCol);// Convert the value into int
long sqlite3_column_int64(sqlite3_stmt_p32 pStmt,int iCol); // Convert the value into int64
double sqlite3_column_double(sqlite3_stmt_p32 pStmt,int iCol); // Convert the value into double
const PTR32 sqlite3_column_text(sqlite3_stmt_p32 pStmt,int iCol);// Get the text value
int sqlite3_column_bytes(sqlite3_stmt_p32 apstmt,int iCol); // Get the number of bytes occupied by the line from the passed cell
int sqlite3_bind_int64(sqlite3_stmt_p32 apstmt,int icol,long a);// Combine the request having a value (of int64 type)
int sqlite3_bind_double(sqlite3_stmt_p32 apstmt,int icol,double a);// Combine the request having a value (of double type)
int sqlite3_bind_text(sqlite3_stmt_p32 apstmt,int icol,char &a[],int len,PTRPTR32 destr);// Combine the request having a value (string type (char* - in C++))
int sqlite3_prepare_v2(sqlite3_p32 db,const uchar &zSql[],int nByte,PTRPTR32 &ppStmt,PTRPTR32 &pzTail);// Prepare a request
int sqlite3_exec(sqlite3_p32 aDb,const char &sql[],PTR32 acallback,PTR32 avoid,PTRPTR32 &errmsg);// Sql execution
int sqlite3_open_v2(const uchar &filename[],sqlite3_p32 &ppDb,int flags,const char &zVfs[]); // Open the database with parameters
#import
```

Databases provided by the developers should be placed to the Libraries folder and named Sqlite3\_32.dll and Sqlite3\_64.dll according to their bit count for the dll database wrapper to work. You can take Dll data from the files attached to the article, compile them on your own from Sqlite Amalgmation or take them from the Sqlite developers' [website](https://www.mql5.com/go?link=https://www.sqlite.org/download.html "https://www.sqlite.org/download.html"). Their presence is a prerequisite for the program. You also need to allow the EA to import Dll.

- The second thing is to write a functional wrapper for connecting to the database. This should be a class that creates a connection to the database and releases it (disconnects from the database) in the destructor. Also, it should be able to execute simple string Sql commands, manage transactions and create queries (statements). All the described functionality was implemented in the CsqliteManager class — it is from its creation that the process of interacting with the database starts.

```
//+------------------------------------------------------------------+
//| Database connection and management class                         |
//+------------------------------------------------------------------+
class CSqliteManager
  {
public:
                     CSqliteManager(){db=NULL;} // Empty constructor
                     CSqliteManager(string dbName); // Pass the name
                     CSqliteManager(string dbName,int flags,string zVfs); // Pass the name and connection flags
                     CSqliteManager(CSqliteManager  &other) { db=other.db; } // Copying constructor
                    ~CSqliteManager(){Disconnect();};// Destructor

   void              Disconnect(); // Disconnect from the database
   bool              Connect(string dbName,int flags,string zVfs); // Parametric connection to the database
   bool              Connect(string dbName); // Connect to the database by name

   void operator=(CSqliteManager  &other){db=other.db;}// Assignment operator

   sqlite3_p64 DB() { return db; }; // Get the pointer to the database

   sqlite3_stmt_p64  Create_statement(const string sql); // Create the statement
   bool              Execute(string sql); // Execute the command
   void              Execute(string  sql,int &result_code,string &errMsg); // Execute the command, and provide the error code and message

   void              BeginTransaction(); // Transaction start
   void              RollbackTransaction(); // Transaction roll-back
   void              CommitTransaction(); // Confirm a transaction

private:
   sqlite3_p64       db; // Database

   void stringToUtf8(const string strToConvert,// String to be converted into an array in utf-8 encoding
                     uchar &utf8[],// Array in utf-8 encoding the converted strToConvert string is to be placed to
                     const bool untilTerminator=true)
     {    // Number of symbols converted to utf-8 encoding and to be copied to the utf-8 array
      //---
      int count=untilTerminator ? -1 : StringLen(strToConvert);
      StringToCharArray(strToConvert,utf8,0,count,CP_UTF8);
     }
  };
```

As can be seen from the code, the resulting class has the ability to create two types of connections in the database (textual and specifying parameters). The Create\_sttement method forms a request to the database and returns a pointer to it. Exequte method overloads perform simple string queries, while transaction methods create and accept/cancel transactions. Connection to the database itself is stored in the db variable. If we applied Disconnect method or just created the class using the default constructor (did not have time to connect to the database yet), the variable is NULL. When repeatedly calling the Connect method, we disconnect from the previously connected database and connect to the new one. Since connecting to the database requires passing of a string in the UTF-8 format, the class has a special 'private' method that converts the string to the required data format.

- The next task is creating a wrapper for convenient work with queries (statement). A request to the database should be created and destroyed. A request is created by CsqliteManager, while the memory is not managed by anything. In other words, after creating a request, it needs to be destroyed when it is no longer required, otherwise it will not allow to disconnect from the database, and when trying to complete the work with the database, we will get an exception indicating that the database is busy. Also, a statement wrapper class should be able to fill the request with passed parameters (when it is formed as "INSERT INTO table\_1 VALUES(@ID,@Param\_1,@Param\_2);"). In addition, a given class should be able to execute the query placed in it (Exequte method).

```
typedef bool(*statement_callback)(sqlite3_stmt_p64); // call-back performed when executing a query. If successful, a 'true' is performed
//+------------------------------------------------------------------+
//| Class of a query to the database                                 |
//+------------------------------------------------------------------+
class CStatement
  {
public:
                     CStatement(){stmt=NULL;} // empty constructor
                     CStatement(sqlite3_stmt_p64 _stmt){this.stmt=_stmt;} // Constructor with the parameter - pointer to the statement
                    ~CStatement(void){if(stmt!=NULL)Sqlite3_finalize(stmt);} // Destructor
   sqlite3_stmt_p64 get(){return stmt;} // Get pointer to statement
   void              set(sqlite3_stmt_p64 _stmt); // Set pointer to statement

   bool              Execute(statement_callback callback=NULL); // Execute statement
   bool              Parameter(int index,const long value); // Add the parameter
   bool              Parameter(int index,const double value); // Add the parameter
   bool              Parameter(int index,const string value); // Add the parameter

private:
   sqlite3_stmt_p64  stmt;
  };
```

Parameter method overloads fill the request parameters. The 'set' method saves the passed statement to the 'stmt' variable: if it is found out that an old request has alrady been saved in the class before saving the new one, the Sqlite3\_finalize method is called for the previously saved request.

- The concluding class in the database handling wrapper is CSqliteReader able to read a response from the database. Similar to previous classes, the class calls the sqlite3\_reset method in its destructor — it drops the request and allows you to work with it again. In the new versions of the database, calling this function is not necessary, but it has been left by the developers. I have used it in the wrapper just in case. Also this class should fulfill its main duties, namely, reading a response from the database string by string with the possibility of converting the read data into the appropriate format.

```
//+------------------------------------------------------------------+
//| Class reading responses from databases                           |
//+------------------------------------------------------------------+
class CSqliteReader
  {
public:
                     CSqliteReader(){statement=NULL;} // empty constructor
                     CSqliteReader(sqlite3_stmt_p64 _statement) { this.statement=_statement; }; // Constructor accepting the pointer to the statement
                     CSqliteReader(CSqliteReader  &other) : statement(other.statement) {} // Copying constructor
                    ~CSqliteReader() { Sqlite3_reset(statement); } // Destructor

   void              set(sqlite3_stmt_p64 _statement); // Add a reference to the statement
   void operator=(CSqliteReader  &other){statement=other.statement;}// Reader assignment operator
   void operator=(sqlite3_stmt_p64 _statement) {set(_statement);}// Statement assignment operator

   bool              Read(); // Read the string
   int               FieldsCount(); // Count the number of columns
   int               ColumnType(int col); // Get the column type

   bool              IsNull(int col); // Check if the value == SQLITE_NULL
   long              GetInt64(int col); // Convert into 'int'
   double            GetDouble(int col);// Convert into 'double'
   string            GetText(int col);// Convert into 'string'

private:
   sqlite3_stmt_p64  statement; // pointer to the statement
  };
```

Now that we have implemented the described classes using the functions for working with the database uploaded from Sqlite3.dll, it is time to describe the classes working with the database from the described program.

The structure of the created database is as follows:

_Buy And Hold table:_

1. Time — X axis (time interval label)
2. PL\_total — profit/loss if we increase a lot in proportion to the robot
3. PL\_oneLot — profit/loss if trading a single lot constantly
4. DD\_total — drawdown if trading a lot the same way the EA traded
5. DD\_oneLot — drawdown if trading a single lot
6. isForvard — forward graph property

_OptimisationParams table:_

1. ID — unique auto-filling entry index in the database
2. HistoryBorder — history optimization completion date
3. TF — timeframe
4. Param\_1...Param\_n — parameter
5. InitalBalance — initial balance size

_ParamsCoefitients table:_

01. ID — external key, reference to OptimisationParams(ID)
02. isForvard — forward optimization property
03. isOneLot — property of the chart the ratio was based on
04. DD — drawdown
05. averagePL — average profit/loss by PL graph
06. averageDD — average drawdown
07. averageProfit — average profit
08. profitFactor — profit factor
09. recoveryFactor — recovery factor
10. sharpRatio — Sharpe ratio
11. altman\_Z\_Score — Altman Z score
12. VaR\_absolute\_90 — VaR 90
13. VaR\_absolute\_95 — VaR 95
14. VaR\_absolute\_99 — VaR 99
15. VaR\_growth\_90 — VaR 90
16. VaR\_growth\_95 — VaR 95
17. VaR\_growth\_99 — VaR 99
18. winCoef — win ratio
19. customCoef — custom ratio

_ParamType table:_

1. ParamName — robot parameter name
2. ParamType — robot parameter type (int/double/string)

_TradingHistory table_

01. ID — external key reference to OptimisationParams(ID)
02. isForvard — forward test flag
03. Symbol — symbol
04. DT\_open — open date
05. Day\_open — open day
06. DT\_close — close date
07. Day\_close — close day
08. Volume — number of lots
09. isLong — long/short property
10. Price\_in — entry price
11. Price\_out — exit price
12. PL\_oneLot — profit when trading a single lot
13. PL\_forDeal — profit when trading as we did previously
14. OpenComment — entry comment
15. CloseComment — exit comment

Based on the provided database structure, we can see that some tables use the external key to refer to the OptimisationParams table where we store the EA parameters. Each column of an input parameter bears its name (for example, Fast/Slow — fast/slow moving average). Also, each column should have a specific data format. Many Sqlite databases are created without defining the table column data format. In this case, all data are stored as lines. However, we need to know the exact data format, since we should sort out ratios by a certain property, which means conversion of the data uploaded from the database to its original format.

To do this, we should know their format before entering the data into the database. Several options are possible: creating a template method and transfer converter into it or creating a class, which, in fact, is a universal storage of several data types (any data type can be converted to) combined with the name of the EA variable. I seleced the second option and created the CDataKeeper class. The described class can store 3 data types \[int, double, string\], while all other data types that can be used as the EA input formats can be converted to them one way or another.

```
//+------------------------------------------------------------------+
//| Types of input data of the EA parameter                          |
//+------------------------------------------------------------------+
enum DataTypes
  {
   Type_INTEGER,// int
   Type_REAL,// double, float
   Type_Text // string
  };
//+------------------------------------------------------------------+
//| Result of comparing two CDataKeeper                              |
//+------------------------------------------------------------------+
enum CoefCompareResult
  {
   Coef_Different,// different data types or variable names
   Coef_Equal,// variables are equal
   Coef_Less, // current variable is less than passed one
   Coef_More // current variable exceeds the passed one
  };
//+---------------------------------------------------------------------+
//| Class storing one specific robot input.                             |
//| It can store the data of the following types: [int, double, string] |
//+---------------------------------------------------------------------+
class CDataKeeper
  {
public:
                     CDataKeeper(); // Constructor
                     CDataKeeper(const CDataKeeper&other); // Copying constructor
                     CDataKeeper(string _variable_name,int _value); // Parametric constructor
                     CDataKeeper(string _variable_name,double _value); // Parametric constructor
                     CDataKeeper(string _variable_name,string _value); // Parametric constructor

   CoefCompareResult Compare(CDataKeeper &data); // Comparison method

   DataTypes         getType(){return variable_type;}; // Get the data type
   string            getName(){return variable_name;}; // Get the parameter name
   string            valueString(){return value_string;}; // Get the parameter
   int               valueInteger(){return value_int;}; // Get the parameter
   double            valueDouble(){return value_double;}; // Get the parameter
   string            ToString(); // Convert any parameter into string. If this is a string parameter, single quotes are added to the string from both sides <<'>>

private:
   string            variable_name,value_string; // variable name and string variable
   int               value_int; // Int variable
   double            value_double; // Double variable
   DataTypes         variable_type; // Variable type

   int compareDouble(double x,double y) // Comparing Double types accurate up to 10 decimal places
     {
      double diff=NormalizeDouble(x-y,10);
      if(diff>0) return 1;
      else if(diff<0) return -1;
      else return 0;
     }
  };
```

Three constructor overloads accept the variable name as the first parameter, while the value converted to one of the mentioned types is accepted as the second one. These values are saved in class global variables starting with 'value\_' followed by a type indication. The getType() method returns the type as an enumeration provided above, while the getName() method returns the variable name. Methods starting with 'value' return the variable of the required type, but if the valueDouble() method is called, while the variable stored in the class is of 'int' type, NULL is returned. The ToString() method converts the value of any of the variables to the string format. However, if the variable was a string initially, the single quotes are added to it (to form SQL requests more conveniently). The Compare(CDataKeeper &ther) method allows comparing two objects of the CDataKeeper type, while comparing:

1. EA variable name
2. Variable type
3. Variable value

If the first two comparisons do not pass, then we are trying to compare two different parameters (for example, the period of the fast moving average with the period of the slow one), and accordingly, we cannot do this because we only need to compare data of the same type. Therefore, we return the Coef\_Different value of the CoefCompareResult type. In other cases, a comparison is made and a required result is returned. The comparison method itself is implemented as follows:

```
//+------------------------------------------------------------------+
//| Compare the current parameter with the passed one                |
//+------------------------------------------------------------------+
CoefCompareResult CDataKeeper::Compare(CDataKeeper &data)
  {
   CoefCompareResult ans=Coef_Different;

   if(StringCompare(this. variable_name,data.getName())==0 &&
      this.variable_type==data.getType()) // Compare names and types
     {
      switch(this.variable_type) // Compare values
        {
         case Type_INTEGER :
            ans=(this.value_int==data.valueInteger() ? Coef_Equal :(this.value_int>data.valueInteger() ? Coef_More : Coef_Less));
            break;
         case Type_REAL :
            ans=(compareDouble(this.value_double,data.valueDouble())==0 ? Coef_Equal :(compareDouble(this.value_double,data.valueDouble())>0 ? Coef_More : Coef_Less));
            break;
         case Type_Text :
            ans=(StringCompare(this.value_string,data.valueString())==0 ? Coef_Equal :(StringCompare(this.value_string,data.valueString())>0 ? Coef_More : Coef_Less));
            break;
        }
     }
   return ans;
  }
```

The type-independent representation of variables allows using them in a more convenient form, taking into account both the name, the data type of the variable and its value.

The next task is creating the database described above. The CDatabaseWriter class is used for that.

```
//+---------------------------------------------------------------------------------+
//| Call-back calculating a user ratio                                              |
//| History data and flag of the history type the ratio calculation is required for |
//| are passed to the input                                                         |
//+---------------------------------------------------------------------------------+
typedef double(*customScoring_1)(const DealDetales &history[],bool isOneLot);
//+---------------------------------------------------------------------------------+
//| Call-back calculating a user ratio                                              |
//| Connection to the database (read only), history and requested ratio type flag   |
//| are passed to the input                                                         |
//+---------------------------------------------------------------------------------+
typedef double(*customScoring_2)(CSqliteManager *dbManager,const DealDetales &history[],bool isOneLot);
//+---------------------------------------------------------------------------------+
//| Class saving the data in the database and creating the database before that     |
//+---------------------------------------------------------------------------------+
class CDBWriter
  {
public:
   // Call one of resets to OnInit
   void              OnInitEvent(const string DBPath,const CDataKeeper &inputData_array[],customScoring_1 scoringFunction,double r,ENUM_TIMEFRAMES TF=PERIOD_CURRENT); // call-back 1
   void              OnInitEvent(const string DBPath,const CDataKeeper &inputData_array[],customScoring_2 scoringFunction,double r,ENUM_TIMEFRAMES TF=PERIOD_CURRENT); // call-back 2
   void              OnInitEvent(const string DBPath,const CDataKeeper &inputData_array[],double r,ENUM_TIMEFRAMES TF=PERIOD_CURRENT);// No call-back and no user ratio (equal to zero)
   double            OnTesterEvent();// Call in OnTester
   void              OnTickEvent();// Call in OnTick

private:
   CSqliteManager    dbManager; // Connector to the database
   CDataKeeper       coef_array[]; // Input parameters
   datetime          DT_Border; // The last candle date (calculated in OnTickEvent)
   double            r; // Risk free rate

   customScoring_1   scoring_1; // Call-back
   customScoring_2   scoring_2; // Call-back
   int               scoring_type; // Call-back type [1,2]
   string            DBPath; // Path to the database
   double            balance; // Balance
   ENUM_TIMEFRAMES   TF; // Timeframe

   void              CreateDB(const string DBPath,const CDataKeeper &inputData_array[],double r,ENUM_TIMEFRAMES TF);// Create the database and everything that goes with it
   bool              isForvard();// Define the current optimization type (history/forward)
   void              WriteLog(string s,string where);// File log entry

   int               setParams(bool IsForvard,CReportCreator *reportCreator,DealDetales &history[],double &customCoef);// Fill the inputs table
   void              setBuyAndHold(bool IsForvard,CReportCreator *reportCreator);// Fill in the Buy And Hold history
   bool              setTraidingHistory(bool IsForvard,DealDetales &history[],int ID);// Fill in the trading history
   bool              setTotalResult(TotalResult &coefData,bool isOneLot,long ID,bool IsForvard,double customCoef);// Fill in the tables with ratios
   bool              isHistoryItem(bool IsForvard,DealDetales &item,int ID); // Check if these parameters already exist in the trading history table
  };
```

The class is used only in the custom robot itself. Its objective is to create an input parameter for a described program, namely the database with a required structure and content. As we can see, it has 3 public methods (the overloaded method is considered as one):

- OnInitEvent
- OnTesterEvent
- OnTickEvent

Each of them is called in the corresponding call-backs of the robot template, where the required parameters are passed to them. The OnInitEvent method is designed to prepare the class for working with the database. Its overloads are implemented as follows:

```
//+------------------------------------------------------------------+
//| Create the database and connection                               |
//+------------------------------------------------------------------+
void CDBWriter::OnInitEvent(const string _DBPath,const CDataKeeper &inputData_array[],customScoring_2 scoringFunction,double _r,ENUM_TIMEFRAMES _TF)
  {
   CreateDB(_DBPath,inputData_array,_r,_TF);
   scoring_2=scoringFunction;
   scoring_type=2;
  }
//+------------------------------------------------------------------+
//| Create the database and connection                               |
//+------------------------------------------------------------------+
void CDBWriter::OnInitEvent(const string _DBPath,const CDataKeeper &inputData_array[],customScoring_1 scoringFunction,double _r,ENUM_TIMEFRAMES _TF)
  {
   CreateDB(_DBPath,inputData_array,_r,_TF);
   scoring_1=scoringFunction;
   scoring_type=1;
  }
//+------------------------------------------------------------------+
//| Create the database and connection                               |
//+------------------------------------------------------------------+
void CDBWriter::OnInitEvent(const string _DBPath,const CDataKeeper &inputData_array[],double _r,ENUM_TIMEFRAMES _TF)
  {
   CreateDB(_DBPath,inputData_array,_r,_TF);
   scoring_type=0;
  }
```

As we can see in the method implementation, it assigns required values to the class fields and creates the database. Call-back methods should be implemented by a user personally (if a custom ratio should be calculated) or an overload without a call-back is used — in this case, a custom ratio is equal to zero. A user's ratio is a custom method of assessing the EA optimization pass. In order to implement it, the pointers to two functions with two types of possible required data were created.

- The first one (customScoring\_1) receives the trading history and the flag defining the optimization pass the calculation is required for (actually traded lot or trading a single lot — all data for calculations are present in the passed array).
- The second call-back type (customScoring\_2) gets access to the database the work is performed from, but only with read-only rights to avoid unexpected edits by the user.

The CreateDB method is one of the main class methods. It performs full preparation for work:


- Assigning balance, timeframe and risk free rate values.
- Establishing connection to the database and occupying a shared resource (Mutex)
- Creating the table database if not created yet.

The OnTickEvent public tick saves the minute candle date at each tick. When testing a strategy, it is impossible to define if the current pass is forward or not, while the database has a similar parameter. But we know that the tester runs forward passes after historical ones. Thus, while overwriting the variable with a date at each tick, we find out the last date at the end of the optimization process. The OptimisationParams table features the HistoryBorder parameter. It is equal to the saved date. The lines are added to this table only during historical optimization. During the first pass with this parameters (same as historical optimization pass), the date is added to the required field in the database. If during one of the next passes, we see that the entry with these parameters is already present in the database, there are two options:

1. either a user for some reasons stopped historical optimization and then re-launched it,
2. or this is a forward optimization.

To filter one from another, we compare the last date stored in the current pass with the date from the database. If the current date is greater than the one in the database, then this is a forward pass, if it is less or equal, you deal with a historical one. Considering that optimization should be launched twice with the same ratios, we enter only the new data into the database or cancel all changes made during the current pass. The OnTesterEvent() method saves data in the database. It is implemented the following way:

```
//+------------------------------------------------------------------+
//| Save all the data in the database and return                     |
//| a custom ratio                                                   |
//+------------------------------------------------------------------+
double CDBWriter::OnTesterEvent()
  {

   DealDetales history[];

   CDealHistoryGetter historyGetter;
   historyGetter.getDealsDetales(history,0,TimeCurrent()); // Get trading history

   CMutexSync sync; // synchronization object
   if(!sync.Create(getMutexName(DBPath))) { Print(Symbol()+" MutexSync create ERROR!"); return 0; }
   CMutexLock lock(sync,(DWORD)INFINITE); // lock the segment within the brackets

   bool IsForvard=isForvard(); // Find out if the current tester iteration is a forward one
   CReportCreator rc;
   string Symb[];
   rc.Get_Symb(history,Symb); // Get the list of symbols
   rc.Create(history,Symb,balance,r); // Create a report (Buy And Hold report is created automatically)

   double ans=0;
   dbManager.BeginTransaction(); // Transaction start

   CStatement stmt(dbManager.Create_statement("INSERT OR IGNORE INTO ParamsType VALUES(@ParamName,@ParamType);")); // Request for saving the list of types of EA parameters
   if(stmt.get()!=NULL)
     {
      for(int i=0;i<ArraySize(coef_array);i++)
        {
         stmt.Parameter(1,coef_array[i].getName());
         stmt.Parameter(2,(int)coef_array[i].getType());
         stmt.Execute(); // save parameter types and their names
        }
     }

   int ID=setParams(IsForvard,&rc,history,ans); // Save EA parameters as well as valuation ratios and get ID
   if(ID>0)// If ID > 0, the parameters are saved successfully
     {
      if(setTraidingHistory(IsForvard,history,ID)) // Save the trading history and check if it is saved
        {
         setBuyAndHold(IsForvard,&rc); // Save the Buy And Hold history (saved only once - during the first saving)
         dbManager.CommitTransaction(); // Confirm the end of a transaction
        }
      else dbManager.RollbackTransaction(); // Otherwise, cancel the transaction
     }
   else dbManager.RollbackTransaction(); // Otherwise, cancel the transaction

   return ans;
  }
```

The first thing the method does is forming the trading history using the class described in my previous article. Then it takes the shared resource (Mutex) and saves the data. To achieve this, first define if the current optimization pass is forward (according to the method described above), then get the list of symbols (all symbols that were traded).

Accordingly, if a spread-trading EA was tested for instance, the trading history is uploaded on both symbols the trading was conducted on. After that, a report is generated (using the class reviewed below) and written to the database. A transaction is created for the correct record. The transaction is canceled if an error occurred when filling any of the tables or incorrect data were obtained. First, the ratios are saved, and then, if everything went smoothly, we save the trading history followed by the Buy and Hold history. The latter is saved only once during the first data entry. In case of a data saving error, Log File in the Common/Files folder is generated.

After creating the database, it should be read. The database reading class is already used in the described program. It is simpler and looks as follows:

```
//+------------------------------------------------------------------+
//| Class reading data from the database                             |
//+------------------------------------------------------------------+
class CDBReader
  {
public:
   void              Connect(string DBPath);// Method connecting to the database

   bool              getBuyAndHold(BuyAndHoldChart_item &data[],bool isForvard);// Method reading Buy And Hold history
   bool              getTraidingHistory(DealDetales &data[],long ID,bool isForvard);// Method reading history traded by the EA
   bool              getRobotParams(CoefData_item &data[],bool isForvard);// Method reading the EA parameters and ratios

private:
   CSqliteManager    dbManager; // Database manager
   string            DBPath; // Path to the database

   bool              getParamTypes(ParamType_item &data[]);// Read input types and their names.
  };
```

It implements 3 public methods reading 4 tables we are interested in and creating structure arrays with data from these tables.

- The first method (getBuyAndHold) returns BuyAndHold history by reference for forward and historical periods depending on the passed flag. If the upload is successful, the method returns 'true', otherwise 'false'. The upload is performed from the Buy And Hold table.
- The getTradingHistory method also returns trading history for passed ID and the isForvard flag accordingly. The upload is performed from the TradingHistory table.
- The getRobotParams method combines uploads from the two tables: ParamsCoefitients — from where the robot parameters are taken and OptimisationParams where calculated valuation ratios are located.

Thus, the written classes allow you to no longer work directly with the database, but with the classes that provide the required data hiding the entire algorithm for working with the database. These classes, in turn, work with the written wrapper for the database, which also simplifies the work. The mentioned wrapper works with the database via Dll provided by the database developers. The database itself meets all required conditions and in fact is a file making it convenient for transportation and processing both in this program and in other analytical applications. Another advantage of this approach is the fact that the long-term operation of a single algorithm enables you to collect databases from each optimization, thereby accumulating history and tracking parameter change patterns.

### Calculations

The block consists of two classes. The first one is meant for generating a trading report and is an improved version of the class generating a trading report described in the previous article.

The second one is a filter class. It sorts optimization samples in a passed range and is able to create a graph displaying a frequency of profitable and loss-making trades for each individual optimization ratio value. Another objective of this class is to create a normal distribution graph for the actually traded PL as of the end of optimization (i.e. PL for the entire optimization period). In other words, if there are 1000 optimization passes, we have 1000 optimization results (PL as of the optimization end). The distribution we are interested in is based on them.

This distribution shows, in which direction the asymmetry of the obtained values ​​is shifted. If the larger tail and the distribution center are in the profit zone, the robot generates mostly profitable optimization passes and, accordingly, is good, otherwise it generates mostly unprofitable passes. If the definition asymmetry is shifted to the loss-making zone, this also means that the selected parameters mostly cause losses rather than profits.

Let's have a look at this block starting with the class generating a trading report. The described class is located in the Include directory of the "History manager" folder and has the following header:

```
//+------------------------------------------------------------------+
//| Class for generating the trading history statistics              |
//+------------------------------------------------------------------+
class CReportCreator
  {
public:

   //=============================================================================================================================================
   // Calculation/ Recalculation:
   //=============================================================================================================================================

   void              Create(DealDetales &history[],DealDetales &BH_history[],const double balance,const string &Symb[],double r);
   void              Create(DealDetales &history[],DealDetales &BH_history[],const string &Symb[],double r);
   void              Create(DealDetales &history[],const string &Symb[],const double balance,double r);
   void              Create(DealDetales &history[],double r);
   void              Create(const string &Symb[],double r);
   void              Create(double r=0);

   //=============================================================================================================================================
   // Getters:
   //=============================================================================================================================================

   bool              GetChart(ChartType chart_type,CalcType calc_type,PLChart_item &out[]); // Get PL graphs
   bool              GetDistributionChart(bool isOneLot,DistributionChart &out); // Get distribution graphs
   bool              GetCoefChart(bool isOneLot,CoefChartType type,CoefChart_item &out[]); // Get ratio graphs
   bool              GetDailyPL(DailyPL_calcBy calcBy,DailyPL_calcType calcType,DailyPL &out); // Get PL graph by days
   bool              GetRatioTable(bool isOneLot,ProfitDrawdownType type,ProfitDrawdown &out); // Get the table of extreme points
   bool              GetTotalResult(TotalResult &out); // Get the TotalResult table
   bool              GetPL_detales(PL_detales &out); // Get the PL_detales table
   void              Get_Symb(const DealDetales &history[],string &Symb[]); // Get the array of symbols that were traded
   void              Clear(); // Clear statistics

private:
   //=============================================================================================================================================
   // Private data types:
   //=============================================================================================================================================
   // Structure of PL graph types
   struct PL_keeper
     {
      PLChart_item      PL_total[];
      PLChart_item      PL_oneLot[];
      PLChart_item      PL_Indicative[];
     };
   // Types structure of daily Profit/Loss graph
   struct DailyPL_keeper
     {
      DailyPL           avarage_open,avarage_close,absolute_open,absolute_close;
     };
   // Structure of the extreme points table
   struct RatioTable_keeper
     {
      ProfitDrawdown    Total_max,Total_absolute,Total_percent;
      ProfitDrawdown    OneLot_max,OneLot_absolute,OneLot_percent;
     };
   // Structures for calculating the amount of profits and losses in a row
   struct S_dealsCounter
     {
      int               Profit,DD;
     };
   struct S_dealsInARow : public S_dealsCounter
     {
      S_dealsCounter    Counter;
     };
   // Structures for calculating auxiliary data
   struct CalculationData_item
     {
      S_dealsInARow     dealsCounter;
      int               R_arr[];
      double            DD_percent;
      double            Accomulated_DD,Accomulated_Profit;
      double            PL;
      double            Max_DD_forDeal,Max_Profit_forDeal;
      double            Max_DD_byPL,Max_Profit_byPL;
      datetime          DT_Max_DD_byPL,DT_Max_Profit_byPL;
      datetime          DT_Max_DD_forDeal,DT_Max_Profit_forDeal;
      int               Total_DD_numDeals,Total_Profit_numDeals;
     };
   struct CalculationData
     {
      CalculationData_item total,oneLot;
      int               num_deals;
      bool              isNot_firstDeal;
     };
   // Structure for creating ratio graphs
   struct CoefChart_keeper
     {
      CoefChart_item    OneLot_ShartRatio_chart[],Total_ShartRatio_chart[];
      CoefChart_item    OneLot_WinCoef_chart[],Total_WinCoef_chart[];
      CoefChart_item    OneLot_RecoveryFactor_chart[],Total_RecoveryFactor_chart[];
      CoefChart_item    OneLot_ProfitFactor_chart[],Total_ProfitFactor_chart[];
      CoefChart_item    OneLot_AltmanZScore_chart[],Total_AltmanZScore_chart[];
     };
   // Class participating in sorting trading history by close date.
   class CHistoryComparer : public ICustomComparer<DealDetales>
     {
   public:
      int               Compare(DealDetales &x,DealDetales &y);
     };
   //=============================================================================================================================================
   // Keepers:
   //=============================================================================================================================================
   CHistoryComparer  historyComparer; // Comparing class
   CChartComparer    chartComparer; // Comparing class

                                    // Auxiliary structures
   PL_keeper         PL,PL_hist,BH,BH_hist;
   DailyPL_keeper    DailyPL_data;
   RatioTable_keeper RatioTable_data;
   TotalResult       TotalResult_data;
   PL_detales        PL_detales_data;
   DistributionChart OneLot_PDF_chart,Total_PDF_chart;
   CoefChart_keeper  CoefChart_data;

   double            balance,r; // Initial deposit and no-risk rate
                                // Sorting class
   CGenericSorter    sorter;

   //=============================================================================================================================================
   // Calculations:
   //=============================================================================================================================================
   // Calculate PL
   void              CalcPL(const DealDetales &deal,CalculationData &data,PLChart_item &pl_out[],CalcType type);
   // Calculate PL histograms
   void              CalcPLHist(const DealDetales &deal,CalculationData &data,PLChart_item &pl_out[],CalcType type);
   // Calculate auxiliary structures used for plotting
   void              CalcData(const DealDetales &deal,CalculationData &out,bool isBH);
   void              CalcData_item(const DealDetales &deal,CalculationData_item &out,bool isOneLot);
   // Calculate daily profit/loss
   void              CalcDailyPL(DailyPL &out,DailyPL_calcBy calcBy,const DealDetales &deal);
   void              cmpDay(const DealDetales &deal,ENUM_DAY_OF_WEEK etalone,PLDrawdown &ans,DailyPL_calcBy calcBy);
   void              avarageDay(PLDrawdown &day);
   // Compare symbols
   bool              isSymb(const string &Symb[],string symbol);
   // Calculate profit factor
   void              ProfitFactor_chart_calc(CoefChart_item &out[],CalculationData &data,const DealDetales &deal,bool isOneLot);
   // Calculate recovery factor
   void              RecoveryFactor_chart_calc(CoefChart_item &out[],CalculationData &data,const DealDetales &deal,bool isOneLot);
   // Calculate win ratio
   void              WinCoef_chart_calc(CoefChart_item &out[],CalculationData &data,const DealDetales &deal,bool isOneLot);
   // Calculate Sharpe ratio
   double            ShartRatio_calc(PLChart_item &data[]);
   void              ShartRatio_chart_calc(CoefChart_item &out[],PLChart_item &data[],const DealDetales &deal);
   // Calculate distribution
   void              NormalPDF_chart_calc(DistributionChart &out,PLChart_item &data[]);
   double            PDF_calc(double Mx,double Std,double x);
   // Calculate VaR
   double            VaR(double quantile,double Mx,double Std);
   // Calculate Z score
   void              AltmanZScore_chart_calc(CoefChart_item &out[],double N,double R,double W,double L,const DealDetales &deal);
   // Calculate the TotalResult_item structure
   void              CalcTotalResult(CalculationData &data,bool isOneLot,TotalResult_item &out);
   // Calculate the PL_detales_item structure
   void              CalcPL_detales(CalculationData_item &data,int deals_num,PL_detales_item &out);
   // Get day from the date
   ENUM_DAY_OF_WEEK  getDay(datetime DT);
   // Clear data
   void              Clear_PL_keeper(PL_keeper &data);
   void              Clear_DailyPL(DailyPL &data);
   void              Clear_RatioTable(RatioTable_keeper &data);
   void              Clear_TotalResult_item(TotalResult_item &data);
   void              Clear_PL_detales(PL_detales &data);
   void              Clear_DistributionChart(DistributionChart &data);
   void              Clear_CoefChart_keeper(CoefChart_keeper &data);

   //=============================================================================================================================================
   // Copy:
   //=============================================================================================================================================
   void              CopyPL(const PLChart_item &src[],PLChart_item &out[]); // Copy PL graphs
   void              CopyCoefChart(const CoefChart_item &src[],CoefChart_item &out[]); // Copy ratio graphs

  };
```

This class, unlike its previous version, calculates two times more data and builds more types of graphs. 'Create' method overloads also calculate the report.

In fact, the report is generated only once — at the time of the Create method call. Later on, only the previously calculated data are obtained in the methods starting with the Get word. The main loop, iterating over the input parameters once, is located in the Create method with the most arguments. This method iterates over arguments and immediately calculates a series of data, based on which all required data are built in the same iteration.

This allows building everything we are interested in within a single pass, while the previous version of this class for getting the graph iterated over initial data again. As a result, the calculation of all ratios lasts milliseconds, while obtaining the required data takes even less time. In the 'private' area of the class, there is a series of structures used only inside that class as more convenient data containers. Sorting trading history is performed using the Generic sorting method described above.

Let's describe data obtained when calling each of the getters:

| Method | Parameters | chart type |
| --- | --- | --- |
| GetChart | chart\_type = \_PL, calc\_type = \_Total | PL graph — according to actual trading history |
| GetChart | chart\_type = \_PL, calc\_type = \_OneLot | PL graph — when trading a single lot |
| GetChart | chart\_type = \_PL, calc\_type = \_Indicative | PL graph — indicative |
| GetChart | chart\_type = \_BH, calc\_type = \_Total | BH graph — if managing a lot as a robot |
| GetChart | chart\_type = \_BH, calc\_type = \_OneLot | BH graph — if trading a single lot |
| GetChart | chart\_type = \_BH, calc\_type = \_Indicative | BH graph — indicative |
| GetChart | chart\_type = \_Hist\_PL, calc\_type = \_Total | PL histogram — according to actual traded history |
| GetChart | chart\_type = \_Hist\_PL, calc\_type = \_OneLot | PL histogram — if trading a single lot |
| GetChart | chart\_type = \_Hist\_PL, calc\_type = \_Indicative | PL histogram — indicative |
| GetChart | chart\_type = \_Hist\_BH, calc\_type = \_Total | BH histogram — if managing a lot as a robot |
| GetChart | chart\_type = \_Hist\_BH, calc\_type = \_OneLot | BH histogram — if trading a single lot |
| GetChart | chart\_type = \_Hist\_BH, calc\_type = \_Indicative | BH histogram — indicative |
| GetDistributionChart | isOneLot = true | Distributions and VaR when trading a single lot |
| GetDistributionChart | isOneLot = false | Distributions and VaR when trading like we did previously |
| GetCoefChart | isOneLot = true, type=\_ShartRatio\_chart | Sharpe ratio by time when trading a single lot |
| GetCoefChart | isOneLot = true, type=\_WinCoef\_chart | Win ratio by time when trading a single lot |
| GetCoefChart | isOneLot = true, type=\_RecoveryFactor\_chart | Recovery factor by time when trading a single lot |
| GetCoefChart | isOneLot = true, type=\_ProfitFactor\_chart | Profit factor by time when trading a single lot |
| GetCoefChart | isOneLot = true, type=\_AltmanZScore\_chart | Z — Altman score by time when trading a single lot |
| GetCoefChart | isOneLot = false, type=\_ShartRatio\_chart | Sharpe ratio by time when trading like we did previously |
| GetCoefChart | isOneLot = false, type=\_WinCoef\_chart | Win ratio by time when trading like we did previously |
| GetCoefChart | isOneLot = false, type=\_RecoveryFactor\_chart | Recovery factor by time when trading like we did previously |
| GetCoefChart | isOneLot = false, type=\_ProfitFactor\_chart | Profit factor by time when trading like we did previously |
| GetCoefChart | isOneLot = false, type=\_AltmanZScore\_chart | Z — Altman score by time when trading like we did previously |
| GetDailyPL | calcBy=CALC\_FOR\_CLOSE, calcType=AVERAGE\_DATA | Average PL by days as of closing time |
| GetDailyPL | calcBy=CALC\_FOR\_CLOSE, calcType=ABSOLUTE\_DATA | Total PL by days as of closing time |
| GetDailyPL | calcBy=CALC\_FOR\_OPEN, calcType=AVERAGE\_DATA | Average PL by days as of opening time |
| GetDailyPL | calcBy=CALC\_FOR\_OPEN, calcType=ABSOLUTE\_DATA | Total PL by days as of opening time |
| GetRatioTable | isOneLot = true, type = \_Max | If trading one lot — maximum obtained profit/loss per trade |
| GetRatioTable | isOneLot = true, type = \_Absolute | If trading one lot — total profit/loss |
| GetRatioTable | isOneLot = true, type = \_Percent | If trading one lot — amount of profits/losses in % |
| GetRatioTable | isOneLot = false, type = \_Max | If trading like we did previously — maximum obtained profit/loss per trade |
| GetRatioTable | isOneLot = false, type = \_Absolute | If trading like we did previously — total profit/loss |
| GetRatioTable | isOneLot = false, type = \_Percent | If trading like we did previously — amount of profits/losses in % |
| GetTotalResult |  | Table with ratios |
| GetPL\_detales |  | PL curve brief summary |
| Get\_Symb |  | Array of symbols present in the trading history |

_PL graph — according to actual trading history:_

The graph is equal to a usual PL graph. We can see this in the terminal after all the tester passes.

_PL graph — when trading a single lot:_

This graph is similar to the previously described one differing in a traded volume. It is calculated as if we were trading a single lot all the time. Entry and exit prices are calculated as averaged prices by the total number of EA market entries and exits. A trading profit is also calculated based on the profit traded by the EA, but it is converted into the profit obtained as if trading a single lot via the proportion.

_PL graph — indicative:_

Normalized PL graph. If PL > 0, PL is divided by the maximum loss-making deal reached by this moment, otherwise PL is divided by the maximum profitable deal reached so far.

Histogram graphs are constructed in a similar way.

_Distributions and VaR_

Parametric VaR is built using both absolute data and growth.

The same is true for the distribution graph.

_Ratio graphs:_

Built at each loop iteration according to the appropriate equations throughout the entire history available for this particular iteration.

_Daily profit graphs:_

Built by 4 possible profit combinations mentioned in the table. Looks like a histogram.

The method creating all mentioned data looks as follows:

```
//+------------------------------------------------------------------+
//| Ratio calculation/re-calculation                                 |
//+------------------------------------------------------------------+
void CReportCreator::Create(DealDetales &history[],DealDetales &BH_history[],const double _balance,const string &Symb[],double _r)
  {
   Clear(); // Clear data
            // Save the balance
   this.balance=_balance;
   if(this.balance<=0)
     {
      CDealHistoryGetter dealGetter;
      this.balance=dealGetter.getBalance(history[ArraySize(history)-1].DT_open);
     }
   if(this.balance<0)
      this.balance=0;
// Save the rate without a risk
   if(_r<0) _r=0;
   this.r=r;

// Auxiliary structures
   CalculationData data_H,data_BH;
   ZeroMemory(data_H);
   ZeroMemory(data_BH);
// Sorting trading history
   sorter.Method(Sort_Ascending);
   sorter.Sort<DealDetales>(history,&historyComparer);
// loop by trading history
   for(int i=0;i<ArraySize(history);i++)
     {
      if(isSymb(Symb,history[i].symbol))
         CalcData(history[i],data_H,false);
     }
// Sorting Buy And Hold history and the appropriate loop
   sorter.Sort<DealDetales>(BH_history,&historyComparer);
   for(int i=0;i<ArraySize(BH_history);i++)
     {
      if(isSymb(Symb,BH_history[i].symbol))
         CalcData(BH_history[i],data_BH,true);
     }

// average daily PL (averaged type)
   avarageDay(DailyPL_data.avarage_close.Mn);
   avarageDay(DailyPL_data.avarage_close.Tu);
   avarageDay(DailyPL_data.avarage_close.We);
   avarageDay(DailyPL_data.avarage_close.Th);
   avarageDay(DailyPL_data.avarage_close.Fr);

   avarageDay(DailyPL_data.avarage_open.Mn);
   avarageDay(DailyPL_data.avarage_open.Tu);
   avarageDay(DailyPL_data.avarage_open.We);
   avarageDay(DailyPL_data.avarage_open.Th);
   avarageDay(DailyPL_data.avarage_open.Fr);

// Fill profit/loss ratio tables
   RatioTable_data.data_H.oneLot.Accomulated_Profit;
   RatioTable_data.data_H.oneLot.Accomulated_DD;
   RatioTable_data.data_H.oneLot.Max_Profit_forDeal;
   RatioTable_data.data_H.oneLot.Max_DD_forDeal;
   RatioTable_data.data_H.oneLot.Total_Profit_numDeals/data_H.num_deals;
   RatioTable_data.data_H.oneLot.Total_DD_numDeals/data_H.num_deals;

   RatioTable_data.Total_absolute.Profit=data_H.total.Accomulated_Profit;
   RatioTable_data.Total_absolute.Drawdown=data_H.total.Accomulated_DD;
   RatioTable_data.Total_max.Profit=data_H.total.Max_Profit_forDeal;
   RatioTable_data.Total_max.Drawdown=data_H.total.Max_DD_forDeal;
   RatioTable_data.Total_percent.Profit=data_H.total.Total_Profit_numDeals/data_H.num_deals;
   RatioTable_data.Total_percent.Drawdown=data_H.total.Total_DD_numDeals/data_H.num_deals;

// Calculate normal distribution
   NormalPDF_chart_calc(OneLot_PDF_chart,PL.PL_oneLot);
   NormalPDF_chart_calc(Total_PDF_chart,PL.PL_total);

// TotalResult
   CalcTotalResult(data_H,true,TotalResult_data.oneLot);
   CalcTotalResult(data_H,false,TotalResult_data.total);

// PL_detales
   CalcPL_detales(data_H.oneLot,data_H.num_deals,PL_detales_data.oneLot);
   CalcPL_detales(data_H.total,data_H.num_deals,PL_detales_data.total);
  }
```

As can be seen from its implementation, part of the data is calculated as the loop goes through the history, while some data is calculated after passing all the loops based on data from the structures: CalculationData data\_H, data\_BH.

The CalcData method is implemented in a way similar to the Create method. This is the only method that calls the methods supposed to perform calculations at each iteration. All methods calculating the final data are calculated based on the information contained in the above-mentioned structures. The filling/refilling of the described structures is carried out by the following method:

```
//+------------------------------------------------------------------+
//| Calculate auxiliary data                                         |
//+------------------------------------------------------------------+
void CReportCreator::CalcData_item(const DealDetales &deal,CalculationData_item &out,
                                   bool isOneLot)
  {
   double pl=(isOneLot ? deal.pl_oneLot : deal.pl_forDeal); // PL
   int n=0;
// Amount of profits and losses
   if(pl>=0)
     {
      out.Total_Profit_numDeals++;
      n=1;
      out.dealsCounter.Counter.DD=0;
      out.dealsCounter.Counter.Profit++;
     }
   else
     {
      out.Total_DD_numDeals++;
      out.dealsCounter.Counter.DD++;
      out.dealsCounter.Counter.Profit=0;
     }
   out.dealsCounter.DD=MathMax(out.dealsCounter.DD,out.dealsCounter.Counter.DD);
   out.dealsCounter.Profit=MathMax(out.dealsCounter.Profit,out.dealsCounter.Counter.Profit);

// Profit and loss series
   int s=ArraySize(out.R_arr);
   if(!(s>0 && out.R_arr[s-1]==n))
     {
      ArrayResize(out.R_arr,s+1,s+1);
      out.R_arr[s]=n;
     }

   out.PL+=pl; // Total PL
               // Max Profit / DD
   if(out.Max_DD_forDeal>pl)
     {
      out.Max_DD_forDeal=pl;
      out.DT_Max_DD_forDeal=deal.DT_close;
     }
   if(out.Max_Profit_forDeal<pl)
     {
      out.Max_Profit_forDeal=pl;
      out.DT_Max_Profit_forDeal=deal.DT_close;
     }
// Accumulated Profit / DD
   out.Accomulated_DD+=(pl>0 ? 0 : pl);
   out.Accomulated_Profit+=(pl>0 ? pl : 0);
// Extreme points by profit
   double maxPL=MathMax(out.Max_Profit_byPL,out.PL);
   if(compareDouble(maxPL,out.Max_Profit_byPL)==1/* || !isNot_firstDeal*/)// yet another check is required for saving the date
     {
      out.DT_Max_Profit_byPL=deal.DT_close;
      out.Max_Profit_byPL=maxPL;
     }
   double maxDD=out.Max_DD_byPL;
   double DD=0;
   if(out.PL>0)DD=out.PL-maxPL;
   else DD=-(MathAbs(out.PL)+maxPL);
   maxDD=MathMin(maxDD,DD);
   if(compareDouble(maxDD,out.Max_DD_byPL)==-1/* || !isNot_firstDeal*/)// yet another check is required for saving the date
     {
      out.Max_DD_byPL=maxDD;
      out.DT_Max_DD_byPL=deal.DT_close;
     }
   out.DD_percent=(balance>0 ?(MathAbs(DD)/(maxPL>0 ? maxPL : balance)) :(maxPL>0 ?(MathAbs(DD)/maxPL) : 0));
  }
```

This is the basic method that calculates all the input data for each of the calculation methods. This approach (moving calculation of input data to this method) allows avoiding excessive passes in the history loops that happened in the previous version of the class creating a trading report. This method is called inside the CalcData method.

Class of the optimization pass results filter has the following header:

```
//+--------------------------------------------------------------------------+
//| Class sorting optimization passes after unloading them from the database |
//+--------------------------------------------------------------------------+
class CParamsFiltre
  {
public:
                     CParamsFiltre(){sorter.Method(Sort_Ascending);} // Default constructor
   int               Total(){return ArraySize(arr_main);}; // Total number of unloaded parameters (according to the Optimisation Data table)
   void              Clear(){ArrayFree(arr_main);ArrayFree(arr_result);}; // Clear all arrays
   void              Add(LotDependency_item &customCoef,CDataKeeper &params[],long ID,double total_PL,bool addToResult); // Add new value to the array
   double            GetCustomCoef(long ID,bool isOneLot);// Get custom ratio by ID
   void              GetParamNames(CArrayString &out);// Get EA parameters name
   void              Get_UniqueCoef(UniqCoefData_item &data[],string paramName,CArrayString &coefValue); // Get unique ratios
   void              Filtre(string Name,string from,string till,long &ID_Arr[]);// Sort the arr_result array
   void              ResetFiltre(long &ID_arr[]);// Reset the filter

   bool              Get_Distribution(Chart_item &out[],bool isMainTable);// Create distribution by both arrays
   bool              Get_Distribution(Chart_item &out[],string Name,string value);// Create distribution by selected data

private:
   CGenericSorter    sorter; // Sorter
   CCoefComparer     cmp_coef;// Compare the ratios
   CChartComparer    cmp_chart;// Compare the graphs

   bool              selectCoefByName(CDataKeeper &_input[],CDataKeeper &out,string Name);// Select ratios by name
   double            Mx(CoefStruct &_arr[]);// Arithmetic mean
   double            Std(CoefStruct &_arr[],double _Mx);// Standard deviation

   CoefStruct        arr_main[]; // Optimisation data table equivalent
   CoefStruct        arr_result[];// Result table equivalent
  };
```

Analyze the structure of the class and tell about some of the methods in more detail. As we can see, the class has two global arrays: arr\_main and arr\_result. The arrays are optimization data storage. After unloading the table with optimization passes from the database, it is divided into two tables:

- main — all unloaded data is obtained except for the data discarded during a conditional sorting
- result — n initially selected best data is obtained. After that, the described class sorts this particular table and, accordingly, reduces or resets the number of its entries.

The described arrays store the EA's ID and parameters, as well as some other data from the above tables according to the array names. In essence, this class performs two functions — a convenient data storage for operations with tables and sorting the table of results of the selected optimization passes. The sorting class and two comparator classes are involved in the sorting process of the mentioned arrays, as well as in the sorting of distributions built according to the described tables.

Since this class operates with the EA ratios, namely, their representation in the form of the CdataKeeper class, a private method selectCoefByName is created. It selects one necessary ratio and returns the result by reference from the array of passed EA ratios of one specific optimization pass.

The Add method adds the line uploaded to the database (both arrays), considering that addToResult ==true, or only to the arr\_main array if addToResult ==false. ID is a unique parameter of each optimization pass, therefore, all the work on the definition of a particular selected pass is based on it. We get the user-calculated ratio for this parameter out of the provided arrays. The program itself does not know the equation for calculating a custom valuation, since the valuation is calculated during the EA optimization without the program participation. This is why we need to save a custom valuation to these arrays. When it is requested, we get it using the GetCustomCoef method by the passed ID.

The most important class methods are as follows:

- Filtre — sort the results table, so that it contains the values of a selected ratio in a passed range (from/till).
- ResetFiltre — reset the entire sorted info.
- Get\_Distribution(Chart\_item &out\[\],bool isMainTable) — build distribution by an actually traded PL according to the selected table specified using the isMainTable parameter.
- Get\_Distribution(Chart\_item &out\[\],string Name,string value) — create a new array where a selected parameter (Name) is equal to the passed value (value). In other words, the pass along the arr\_result array is performed in a loop. During each iteration of the loop, the parameter we are interested in is selected by its name (using the selectCoefByName function) out of all EA parameters. Also, it is checked whether its value is equal to the required one (value). If yes, the arr\_result array value is added to the temporary array. Then, a distribution by the temporary array is created and returned. In other words, this is how we select all optimization passes, where the value of the parameter selected by the name was detected that is equal to the passed value. This is necessary in order to estimate how much this particular parameter affects the EA as a whole. The implementation of the described class is adequately commented in the code, and therefore I will not provide the implementation of these methods here.

### Presenter

Presenter serves as a connector. This is a kind of a linkage between the graphic layer of the application and its logic described above. In this application, the presenter is implemented using abstractions — the IPresenter interface. This interface contains the name of the required call-back methods; they, in turn, are implemented in the presenter class, which should inherit the required interface. This division was created to finalize the application. If you need to rewrite the presenter block, this can be done easily without affecting the block of graphics or the application logic. The described interface is presented as follows:

```
//+------------------------------------------------------------------+
//| Presenter interface                                              |
//+------------------------------------------------------------------+
interface IPresenter
  {
   void Btn_Update_Click(); // Download data and build the entire form
   void Btn_Load_Click(); // Create a report
   void OptimisationData(bool isMainTable);// Select optimization line in the tables
   void Update_PLByDays(); // Upload profit and loss by days
   void DaySelect();// Select a day from the PL table by week days
   void PL_pressed(PLSelected_type type);// Build PL graph by selected history
   void PL_pressed_2(bool isRealPL);// Build "Other charts" graphs
   void SaveToFile_Click();// Save the data file (to the sandboxes)
   void SaveParam_passed(SaveParam_type type);// Select data to write to the file
   void OptimisationParam_selected(); // Select optimization parameter and fill the "Optimisation selection" tab
   void CompareTables(bool isChecked);// Build distribution by the result table (for correlation with the common (main) table)
   void show_FriquencyChart(bool isChecked);// Display profit/loss frequency graph
   void FriquencyChart_click();// Select a line in the ratio table and build a distribution
   void Filtre_click();// Sort by selected conditions
   void Reset_click();// Reset the filters
   void PL_pressed_3(bool isRealPL);// Build the profit/loss graphs by all data in the Result table
   void PL_pressed_4(bool isRealPL);// Build statistics tables
   void setChartFlag(bool isPlot);// Condition to build (or not to build) graphs from the PL_pressed_3(bool isRealPL) method;
  };
```

The presenter class implements the required interface and looks like this:

```
class CPresenter : public IPresenter
  {
public:
                     CPresenter(CWindowManager *_windowManager); // Constructor

   void              Btn_Update_Click();// Download data and build the entire form
   void              Btn_Load_Click();// Create a report
   void              OptimisationData(bool isMainTable);// Select optimization line in the tables
   void              Update_PLByDays();// Upload profit and loss by days
   void              PL_pressed(PLSelected_type type);// Build PL graph by selected history
   void              PL_pressed_2(bool isRealPL);// Build "Other charts" graphs
   void              SaveToFile_Click();// Save the data file (to the sandboxes)
   void              SaveParam_passed(SaveParam_type type);// Select data to write to the file
   void              OptimisationParam_selected();// Select optimization parameter and fill the "Optimisation selection" tab
   void              CompareTables(bool isChecked);// Build distribution by the result table (for correlation with the common (main) table)
   void              show_FriquencyChart(bool isChecked);// Display profit/loss frequency graph
   void              FriquencyChart_click();// Select a line in the ratio table and build a distribution
   void              Filtre_click();// Sort by selected conditions
   void              PL_pressed_3(bool isRealPL);// Build the profit/loss graphs by all data in the Result table
   void              PL_pressed_4(bool isRealPL);// Build statistics tables
   void              DaySelect();// Select a day from the PL table by week days
   void              Reset_click();// Reset the filters
   void              setChartFlag(bool isPlot);// Condition to build (or not to build) graphs from the PL_pressed_3(bool isRealPL) method;

private:
   CWindowManager   *windowManager;// Reference to the window class
   CDBReader         dbReader;// Class working with the database
   CReportCreator    reportCreator; // Class processing data

   CGenericSorter    sorter; // Sorting class
   CoefData_comparer coefComparer; // Class comparing data

   void              loadData();// Upload data from the database and fill in the tables

   void              insertDataTo_main_Table(bool isResult,const CoefData_item &data[]); // Insert the data to the results table and to the "Main" table (tables with optimization pass ratios)
   void              insertRowTo_main_Table(CTable *tb,int n,const CoefData_item &data); // Direct data insertion to the optimization pass tables
   void              selectChartByID(long ID,bool recalc=true);// Select graphs by ID
   void              createReport();// Create a report
   string            getCorrectPath(string path,string name);// Get correct path to the file
   bool              getPLChart(PLChart_item &data[],bool isOneLot,long ID);

   bool              curveAdd(CGraphic *chart_ptr,const PLChart_item &data[],bool isHist);// Add the graph to Other Charts
   bool              curveAdd(CGraphic *chart_ptr,const CoefChart_item &data[],double borderPoint);// Add the graph to Other Charts
   bool              curveAdd(CGraphic *chart_ptr,const Distribution_item &data);// Add the graph to Other Charts
   void              setCombobox(CComboBox *cb_ptr,CArrayString &arr,bool isFirstIndex=true);// Set the combo box parameters
   void              addPDF_line(CGraphic *chart_ptr,double &x[],color clr,int width,string _name=NULL);// Add the distribution graph's smooth line
   void              plotMainPDF();// Build distribution by the "Main" table (Optimisation Data)
   void              updateDT(CDropCalendar *dt_ptr,datetime DT);// Update drop-down calendars

   CParamsFiltre     coefKeeper;// Sort optimization passes (by distributions)
   CArrayString      headder; // Ratio tables header

   bool              _isUpbateClick; // Flag of the Update button pressing and data upload from the database
   long              _selectedID; // ID of a selected series of all PL graph (red if loss-making and green if profitable)
   long              _ID,_ID_Arr[];// Array of IDs selected for the Result table after uploading the data
   bool              _IsForvard_inTables,_IsForvard_inReport; // Flag of the optimization data type in the optimization pass tables
   datetime          _DT_from,_DT_till;
   double            _Gap; // Saved type of the added gap (spread extension / or slippage simulation...) of the previous selected optimization graph
  };
```

Each of the callbacks is quite well commented, so there is no need to dwell on them here. It is only necessary to say that this is exactly the part of the application where all the form behavior is implemented. It contains building graphs, filling in combo boxes, calling methods for uploading and handling data from the database, as well as other operations connecting various classes.

### Conclusion

We have developed the application handling the table with all possible optimization parameters passed through the tester, as well as the addition to the EA for saving all optimization passes to the database. In addition to the detailed trading report we obtain when selecting a parameter we are interested in, the program also allows us to thoroughly view an interval from the entire optimization history selected by time, as well as all ratios for a given time interval. It is also possible to simulate slippage by increasing the Gap parameter and see how this affects the behavior of graphs and ratios. Another addition is the ability to sort optimization results in a certain interval of ratio values.

The easiest way to get 100 best optimization passes is to connect the CDBWriter class to your robot, just like with the sample EA (in the attached files), set the conditional filter (for example, Profit Factor >= 1 immediately excludes all loss-making combinations) and click Update leaving the "Show n params" parameter equal to 100. In this case, 100 best optimization passes (according to your filter) are displayed in the result table. Each of the options of the resulting application, as well as more refined methods of selecting ratios will be discussed in more detail in the next article.

_The following files are attached to the article:_

**Experts/2MA\_Martin — test EA project**

- 2MA\_Martin.mq5 — EA template code. The DBWriter.mqh file that saves optimization data to the database is included into it.
- Robot.mq5 — EA logic
- Robot.mqh — header file implemented in the Robot.mq5 file
- Trade.mq5 — EA trading logic
- Trade.mqh — header file implemented in the Trade.mq5 file

**Experts/OptimisationSelector — described application project**

- OptimisationSelector.mq5 — template of an EA calling the entire project code
- ParamsFiltre.mq5 — filter and distributions by result tables
- ParamsFiltre.mqh — header file implemented in the ParamsFiltre.mq5 file
- Presenter.mq5 — presenter
- Presenter.mqh — header file implemented in the Presenter.mq5 file
- Presenter\_interface.mqh — presenter interface
- Window\_1.mq5 — graphics
- Window\_1.mqh — header file implemented in the Window\_1.mq5 file

**Include/CustomGeneric**

- GenericSorter.mqh — data sorting
- ICustomComparer.mqh — ICustomSorter interface

**Include/History manager**

- DealHistoryGetter.mqh — unloading trading history from the terminal and converting it into a required view
- ReportCreator.mqh — class creating the trading history

**Include/OptimisationSelector**

- DataKeeper.mqh — class for storing the ratios of the EA associated with the ratio name
- DBReader.mqh — class reading required tables from the database
- DBWriter.mqh — class writing to the database

**Include/Sqlite3**

- sqlite\_amalgmation.mqh — importing functions for working with the database
- SqliteManager.mqh — connector to the database and statement class
- SqliteReader.mqh — class reading responds from the database

**Include/WinApi**

- memcpy.mqh — import the memcpy function
- Mutex.mqh — import the Mutex creation functions
- strcpy.mqh — import the strcpy function
- strlen.mqh — import the strlen function

**Libraries**

- Sqlite3\_32.dll — Dll Sqlite for 32-bit terminals
- Sqlite3\_64.dll — Dll Sqlite for 64-bit terminals

**Test database**

- 2MA\_Martin optimisation data - Sqlite database

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/5214](https://www.mql5.com/ru/articles/5214)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/5214.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/5214/mql5.zip "Download MQL5.zip")(6783.27 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Continuous walk-forward optimization (Part 8): Program improvements and fixes](https://www.mql5.com/en/articles/7891)
- [Continuous Walk-Forward Optimization (Part 7): Binding Auto Optimizer's logical part with graphics and controlling graphics from the program](https://www.mql5.com/en/articles/7747)
- [Continuous Walk-Forward Optimization (Part 6): Auto optimizer's logical part and structure](https://www.mql5.com/en/articles/7718)
- [Continuous Walk-Forward Optimization (Part 5): Auto Optimizer project overview and creation of a GUI](https://www.mql5.com/en/articles/7583)
- [Continuous Walk-Forward Optimization (Part 4): Optimization Manager (Auto Optimizer)](https://www.mql5.com/en/articles/7538)
- [Continuous Walk-Forward Optimization (Part 3): Adapting a Robot to Auto Optimizer](https://www.mql5.com/en/articles/7490)
- [Continuous Walk-Forward Optimization (Part 2): Mechanism for creating an optimization report for any robot](https://www.mql5.com/en/articles/7452)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/288610)**
(16)


![Victor Volovin](https://c.mql5.com/avatar/avatar_na2.png)

**[Victor Volovin](https://www.mql5.com/en/users/victor2v)**
\|
18 Nov 2018 at 12:37

Thanks for your help. The project appeared but gave an error.

[![](https://c.mql5.com/3/251/image__21.png)](https://c.mql5.com/3/251/image__20.png "https://c.mql5.com/3/251/image__20.png")

![Andrey Azatskiy](https://c.mql5.com/avatar/2018/6/5B127D58-708F.jpg)

**[Andrey Azatskiy](https://www.mql5.com/en/users/andreykrivcov)**
\|
18 Nov 2018 at 19:12

**Victor Volovin:**

Thanks for your help. The project appeared but gave an error.

Thank you for your feedback. You need to download the EasyAndFast [graphics library](https://www.mql5.com/en/articles/2634 "Article: Graphical Interfaces X: Updates for Easy And Fast Library (build 2) ") and make the appropriate edits to it (there is a link in the article and it describes what edits to make where).

![Vasily Yuan](https://c.mql5.com/avatar/2018/7/5B60C3EA-F46D.jpeg)

**[Vasily Yuan](https://www.mql5.com/en/users/vasilyalexanov)**
\|
8 Jul 2019 at 18:35

The whole [project](https://www.mql5.com/en/articles/7863 "Article: Projects allow you to create profitable trading robots! But it is not exactly") is very interesting, but it would be a pleasure to know how can we those on mt4/mql4 version to try this out if possible?

![Andrey Azatskiy](https://c.mql5.com/avatar/2018/6/5B127D58-708F.jpg)

**[Andrey Azatskiy](https://www.mql5.com/en/users/andreykrivcov)**
\|
19 Aug 2019 at 12:42

**Vasily Yuan:**

The whole project is very interesting, but it would be a pleasure to know how can we those on mt4/mql4 version to try this out if possible?

Mt4 is not suported, sorry.

![Oluwatosin David Oyinleye](https://c.mql5.com/avatar/2022/2/621D1974-4D2E.png)

**[Oluwatosin David Oyinleye](https://www.mql5.com/en/users/tosyne07)**
\|
4 Mar 2023 at 04:28

Thank you so much for this article. Please how can I resolve this error when I compiled the GenericSorter [![](https://c.mql5.com/3/402/1819891489066__1.png)](https://c.mql5.com/3/402/1819891489066.png "https://c.mql5.com/3/402/1819891489066.png")

![EA remote control methods](https://c.mql5.com/2/34/RemoteControl_EA.png)[EA remote control methods](https://www.mql5.com/en/articles/5166)

The main advantage of trading robots lies in the ability to work 24 hours a day on a remote VPS server. But sometimes it is necessary to intervene in their work, while there may be no direct access to the server. Is it possible to manage EAs remotely? The article proposes one of the options for controlling EAs via external commands.

![Modeling time series using custom symbols according to specified distribution laws](https://c.mql5.com/2/33/Custom_series_modelling.png)[Modeling time series using custom symbols according to specified distribution laws](https://www.mql5.com/en/articles/4566)

The article provides an overview of the terminal's capabilities for creating and working with custom symbols, offers options for simulating a trading history using custom symbols, trend and various chart patterns.

![Movement continuation model - searching on the chart and execution statistics](https://c.mql5.com/2/34/wave_movie.png)[Movement continuation model - searching on the chart and execution statistics](https://www.mql5.com/en/articles/4222)

This article provides programmatic definition of one of the movement continuation models. The main idea is defining two waves — the main and the correction one. For extreme points, I apply fractals as well as "potential" fractals - extreme points that have not yet formed as fractals.

![Reversing: The holy grail or a dangerous delusion?](https://c.mql5.com/2/33/avatar5008.png)[Reversing: The holy grail or a dangerous delusion?](https://www.mql5.com/en/articles/5008)

In this article, we will study the reverse martingale technique and will try to understand whether it is worth using, as well as whether it can help improve your trading strategy. We will create an Expert Advisor to operate on historic data and to check what indicators are best suitable for the reversing technique. We will also check whether it can be used without any indicator as an independent trading system. In addition, we will check if reversing can turn a loss-making trading system into a profitable one.

[![](https://www.mql5.com/ff/si/0nfwvn6yhmgzf97n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F117%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dorder.expert%26utm_content%3Dorder.freelance%26utm_campaign%3D0622.MQL5.com.Internal&a=tunpwtbhegzufrqocbwiszessdutnobs&s=d9e7484e15300021b4066b1df77a94a1352f9e7c326d5113006bb4f6476bafeb&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=ofhzahvcywyfkkjuwzuslazoabkyovdf&ssn=1769158149299081788&ssn_dr=0&ssn_sr=0&fv_date=1769158149&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F5214&back_ref=https%3A%2F%2Fwww.google.com%2F&title=100%20best%20optimization%20passes%20(part%201).%20Developing%20optimization%20analyzer%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176915814987454121&fz_uniq=5062742449115277317&sv=2552)

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