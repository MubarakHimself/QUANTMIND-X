---
title: Tracing, Debugging and Structural Analysis of Source Code
url: https://www.mql5.com/en/articles/272
categories: Trading Systems, Integration
relevance_score: 0
scraped_at: 2026-01-24T13:55:54.039959
---

[![](https://www.mql5.com/ff/sh/9nb0c8df2rmwfn89z2/01.png) MetaTrader VPS vs regular cloud hosting services8 reasons why our solution is the best option for automated tradingRead](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/450486&a=dgmsfszgoedimaicrqqmagvqzpwuxkur&s=c59e3617ccf44fd54d4c50a03b44fd689ff7507b8fe4990c83772cc5419e627d&uid=&ref=https://www.mql5.com/en/articles/272&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5083220436570871608)

MetaTrader 5 / Examples


### Introduction

This article tells about one of the methods of creating a call stack during execution. The following features are described in the article:

- Making the structure of used classes, functions and files.
- Making the call stack keeping all previous stacks. Sequence of calling them.
- Viewing the state of Watch parameters during execution.
- Stepwise execution of code.
- Grouping and sorting obtained stacks, getting "extreme" information.


### Main Principles of Development

A common approach is chosen as the method of representation of the structure – displaying in the form of a tree. For this purpose, we need two informational classes. **CNode** \- a "node" used for writing all information about a stack. **CTreeCtrl** \- a "tree" that processes all nodes. And the tracer itself - **CTraceCtrl**, used for processing trees.

The classes are implemented according to the following hierarchy:

![](https://c.mql5.com/2/3/kgscnq2c__1.jpg)

The **CNodeBase** and **CTreeBase** classes describe basic properties and methods of working with nodes and trees.

The inherited class **CNode** extends the basic functionality of **CNodeBase**, and the **CTreeBase** class works with the derived class **CNode**. It is done due to the class **CNodeBase** is the parent of the other standard nodes, and it is isolated as an independent class for the convenience of hierarchy and inheritance.

Unlike [CTreeNode](https://www.mql5.com/en/docs/standardlibrary/datastructures/ctreenode) from the standard library, the **CNodeBase** class contains an array of pointers to nodes, thus the number of "branches" coming out of this node is unlimited.

The CNodeBase and CNode Classes

```
class CNode; // forward declaration
//------------------------------------------------------------------    class CNodeBase
class CNodeBase
  {
public:
   CNode   *m_next[]; // list of nodes it points to
   CNode   *m_prev; // parent node
   int     m_id; // unique number
   string  m_text; // text

public:
          CNodeBase() { m_id=0; m_text=""; } // constructor
          ~CNodeBase(); // destructor
  };

//------------------------------------------------------------------    class CNode
class CNode : public CNodeBase
  {
public:
   bool    m_expand; // expanded
   bool    m_check; // marked with a dot
   bool    m_select; // highlighted

   //--- run-time information
   int     m_uses; // number of calls of the node
   long    m_tick; // time spent in the node
   long    m_tick0; // time of entering the node
   datetime    m_last; // time of entering the node
   tagWatch   m_watch[]; // list of name/value parameters
   bool    m_break; // debug-pause

   //--- parameters of the call
   string   m_file; // file name
   int      m_line; // number of row in the file
   string   m_class; // class name
   string   m_func; // function name
   string   m_prop; // add. information

public:
           CNode(); // constructor
           ~CNode(); // destructor
   void    AddWatch(string watch,string val);
  };
```

You can find the implementation of all classes in the attached files. In the article, we're going to show only their headers and important functions.

According to the accepted classification, **CTreeBase** represents and oriented acyclic graph. The derived class **CTreeCtrl** uses **CNode** and serves all its functionality: adding, changing and deleting the **CNode** nodes.

**CTreeCtrl** and **CNode** can successfully substitute the corresponding classes of the standard library, since they have a slightly wider functionality.

The CTreeBase and CTreeCtrl Classes

```
//------------------------------------------------------------------    class CTreeBase
class CTreeBase
  {
public:
   CNode   *m_root; // first node of the tree
   int     m_maxid;    // counter of ID

   //--- base functions
public:
           CTreeBase(); // constructor
           ~CTreeBase(); // destructor
   void    Clear(CNode *root=NULL); // deletion of all nodes after a specified one
   CNode   *FindNode(int id,CNode *root=NULL); // search of a node by its ID starting from a specified node
   CNode   *FindNode(string txt,CNode *root=NULL); // search of a node by txt starting from a specified node
   int     GetID(string txt,CNode *root=NULL); // getting ID for a specified Text, the search starts from a specified node
   int     GetMaxID(CNode *root=NULL); // getting maximal ID in the tree
   int     AddNode(int id,string text,CNode *root=NULL); // adding a node to the list, search is performed by ID starting from a specified node
   int     AddNode(string txt,string text,CNode *root=NULL); // adding a node to the list, search is performed by text starting from a specified node
   int     AddNode(CNode *root,string text); // adding a node under root
  };

//------------------------------------------------------------------    class CTreeCtrl
class CTreeCtrl : public CTreeBase
  {
   //--- base functions
public:
          CTreeCtrl() { m_root.m_file="__base__"; m_root.m_line=0;
                        m_root.m_func="__base__"; m_root.m_class="__base__"; } // constructor
          ~CTreeCtrl() { delete m_root; m_maxid=0; } // destructor
   void    Reset(CNode *root=NULL); // reset the state of all nodes
   void    SetDataBy(int mode,int id,string text,CNode *root=NULL); // changing text for a specified ID, search is started from a specified node
   string  GetDataBy(int mode,int id,CNode *root=NULL); // getting text for a specified ID, search is started from a specified node

   //--- processing state
public:
   bool    IsExpand(int id,CNode *root=NULL); // getting the m_expand property for a specified ID, search is started from a specified node
   bool    ExpandIt(int id,bool state,CNode *root=NULL); // change the m_expand state, search is started from a specified node
   void    ExpandBy(int mode,CNode *node,bool state,CNode *root=NULL); // expand node of a specified node

   bool    IsCheck(int id,CNode *root=NULL); // getting the m_check property for a specified ID, search is started from a specified node
   bool    CheckIt(int id,bool state,CNode *root=NULL); // change the m_check state to a required one starting from a specified node
   void    CheckBy(int mode,CNode *node,bool state,CNode *root=NULL); // mark the whole tree

   bool    IsSelect(int id,CNode *root=NULL); // getting the m_select property for a specified ID, search is started from a specified node
   bool    SelectIt(int id,bool state,CNode *root=NULL); // change the m_select state to a required one starting from a specified node
   void    SelectBy(int mode,CNode *node,bool state,CNode *root=NULL); // highlight the whole tree

   bool    IsBreak(int id,CNode *root=NULL); // getting the m_break property for a specified ID, search is started from a specified node
   bool    BreakIt(int id,bool state,CNode *root=NULL); // change the m_break state, search is started from a specified node
   void    BreakBy(int mode,CNode *node,bool state,CNode *root=NULL); // set only for a selected one

   //--- operations with nodes
public:
   void    SortBy(int mode,bool ascend,CNode *root=NULL); // sorting by a property
   void    GroupBy(int mode,CTreeCtrl *atree,CNode *node=NULL); // grouping by a property
  };
```

The architecture ends with two classes: **CTraceCtrl** \- its only instance is used directly for tracing, it contains three instances of the **CTreeCtrl** class for creation of required structure of functions; and a temporary container - the **CIn** class. This is just an auxiliary class that is used for adding new nodes to **CTraceCtrl**.

The CTraceCtrl and CIn Classes

```
class CTraceView; // provisional declaration
//------------------------------------------------------------------    class CTraceCtrl
class CTraceCtrl
  {
public:
   CTreeCtrl   *m_stack; // object of graph
   CTreeCtrl   *m_info; // object of graph
   CTreeCtrl   *m_file; // grouping by files
   CTreeCtrl   *m_class; // grouping by classes
   CTraceView  *m_traceview; // pointer to displaying of class

   CNode   *m_cur; // pointer to the current node
           CTraceCtrl() { Create(); Reset(); } // tracer created
           ~CTraceCtrl() { delete m_stack; delete m_info; delete m_file; delete m_class; } // tracer deleted
   void    Create(); // tracer created
   void    In(string afile,int aline,string aname,int aid); // entering a specified node
   void    Out(int aid); // exit from a specified node
   bool    StepBack(); // exit from a node one step higher (going to the parent)
   void    Reset() { m_cur=m_stack.m_root; m_stack.Reset(); m_file.Reset(); m_class.Reset(); } // resetting all nodes
   void    Clear() { m_cur=m_stack.m_root; m_stack.Clear(); m_file.Clear(); m_class.Clear(); } // resetting all nodes

public:
   void    AddWatch(string name,string val); // checking the debug mode for a node
   void    Break(); // pause for a node
  };

//------------------------------------------------------------------    CIn
class CIn
  {
public:
   void In(string afile,int aline,string afunc)
     {
      if(NIL(m_trace)) return; // exit if there is no graph
      if(NIL(m_trace.m_tree)) return;
      if(NIL(m_trace.m_tree.m_root)) return;
      if(NIL(m_trace.m_cur)) m_trace.m_cur=m_trace.m_tree.m_root;
      m_trace.In(afile,aline,afunc,-1); // entering the next one
     }
   void ~CIn() { if(!NIL(m_trace)) m_trace.Out(-1); } // exiting higher
  };
```

### Model of Operation of the CIn Class

This class is in charge of creation of the stack tree.

Forming of the graph is performed stepwise in two stages using two **CTraceCtrl** functions:

```
void In(string afile, int aline, string aname, int aid); // entering a specified node
void Out(int aid);  // exit before a specified node
```

In other words, to form a tree, continuous calls of **In-Out-In-Out-In-In-Out-Out**, etc. are preformed.

The **In-Out** pair works in the following way:

_1\. Entering a block (function, cycle, condition, etc.), i.e. right after the bracket " **{**"._

When entering the block, a new instance of **CIn** is created, it gets the current **CTraceCtrl** that is already started with some previous nodes. The **CTraceCtrl** **::In** function is called in **CIn**, it creates a new node in the stack. The node is created under the current node **CTraceCtrl** **::m\_cur**. All the actual information about entering is written in it: file name, row number, class name, functions, current time, etc.

_2\. Exiting from the block when meeting a " **}**" bracket._

When exiting from the block, MQL automatically calls the destructor **CIn::~CIn**. **CTraceCtrl** **::Out** is called in the destructor. The pointer of the current node **CTraceCtrl** **::m\_cur** is raised one level higher in the tree. At that the destructor is not called for the new node, the node stays in the tree.

Scheme of Forming a Stack

![](https://c.mql5.com/2/3/h4ty__2.jpg)

Forming of the call stack in the form of a tree with filling all the information about a call is performed using the **CIn** container.

### Macros to Make Calls Easier

To avoid rewriting the long lines of code of creating the **CIn** object and entering a node in your code, it is convenient to replace it with the call of the macro:

```
#define _IN    CIn _in; _in.In(__FILE__, __LINE__, __FUNCTION__)
```

As you see, the **CIn** object is created and then we enter the node.

Since MQL gives a warning in case the names of local variables are the same as the global variables, it's better (more accurate and clear) to create 3-4 analogous definitions with the other names of variables in the following form:

```
#define _IN1    CIn _in1; _in1.In(__FILE__, __LINE__, __FUNCTION__)
#define _IN2    CIn _in2; _in2.In(__FILE__, __LINE__, __FUNCTION__)
#define _IN3    CIn _in3; _in3.In(__FILE__, __LINE__, __FUNCTION__)
```

As you go deeper to sub-blocks, use next macros \_ **INx**

```
bool CSampleExpert::InitCheckParameters(int digits_adjust)
  { _IN;
//--- initial data checks
   if(InpTakeProfit*digits_adjust<m_symbol.StopsLevel())
     { _IN1;
      printf("Take Profit must be greater than %d",m_symbol.StopsLevel());
```

With appearing of macros in the build 411, you can fully use passing of parameters using **#define**.

That is why in the **CTraceCtrl** class you'll find the following macro definition:

```
#define NIL(p)    (CheckPointer(p)==POINTER_INVALID)
```

It allows shortening the check of validity of the pointer.

For example, the line:

```
if (CheckPointer(m_tree))==POINTER_INVALID || CheckPointer(m_cur))==POINTER_INVALID) return;
```

is replaced with the shorter variant:

```
if (NIL(m_tree) || NIL(m_cur)) return;
```

### Preparing Your Files for Tracing

To control and get the stack, you need to take three steps.

_1\. Add the required files_

```
#include <Trace.mqh>
```

The entire standard library is based on the **CObject** class at the moment. Thus if it is also used as a base class in your files, it's sufficient to add **Trace.mqh** only to **Object.mqh**.

_2\. Place the **\_IN** macros to the required blocks (you can use search/replace)_

The example of using the \_IN macro:

```
bool CSampleExpert::InitCheckParameters(int digits_adjust)
  { _IN;
//--- initial data checks
   if(InpTakeProfit*digits_adjust<m_symbol.StopsLevel())
     { _IN1;
      printf("Take Profit must be greater than %d",m_symbol.StopsLevel());
```

3\. In the **OnInit**, **OnTime**, and **OnDeinit** functions consisting the main module of the program, add the creation, modification and deletion of the global object **CTraceCtrl** respectively. Below you can find the ready-made code for inserting:

Embedding the tracer in main code

```
//------------------------------------------------------------------    OnInit
int OnInit()
  {
   //****************
   m_traceview= new CTraceView; // created displaying of the graph
   m_trace= new CTraceCtrl; // created the graph
   m_traceview.m_trace=m_trace; // attached the graph
   m_trace.m_traceview=m_traceview; // attached displaying of the graph
   m_traceview.Create(ChartID()); // created chart
   //****************
   // remaining part of your code…
   return(0);
  }
//------------------------------------------------------------------    OnDeinit
void OnDeinit(const int reason)
  {
   //****************
   delete m_traceview;
   delete m_trace;
   //****************
   // remaining part of your code…
  }
//------------------------------------------------------------------    OnTimer
void OnTimer()
  {
   //****************
   if (m_traceview.IsOpenView(m_traceview.m_chart)) m_traceview.OnTimer();
   else { m_traceview.Deinit(); m_traceview.Create(ChartID()); } // if the window is accidentally closed
   //****************
   // remaining part of your code…
  }
//------------------------------------------------------------------    OnChartEvent
void OnChartEvent(const int id, const long& lparam, const double& dparam, const string& sparam)
  {
   //****************
   m_traceview.OnChartEvent(id, lparam, dparam, sparam);
   //****************
   // remaining part of your code…
  }
```

### Classes of Trace Displaying

So, the stack has been organized. Now let's consider displaying of obtained information.

For this purpose, we should create two classes. **CTreeView** – for displaying of the tree, and **CTraceView** – for controlling of displaying of trees and additional information about stack. Both classes are derived from the base class **CView**.

![](https://c.mql5.com/2/3/i904rhvl1__1.jpg)

The CTreeView and CTraceView Classes

```
//------------------------------------------------------------------    class CTreeView
class CTreeView: public CView
  {
   //--- basic functions
public:
           CTreeView(); // constructor
           ~CTreeView(); // destructor
   void    Attach(CTreeCtrl *atree); // attached the tree object for displaying it
   void    Create(long chart,string name,int wnd,color clr,color bgclr,color selclr,
                    int x,int y,int dx,int dy,int corn=0,int fontsize=8,string font="Arial");

   //--- functions of processing of state
public:
   CTreeCtrl        *m_tree; // pointer to the tree object to be displayed
   int     m_sid; // last selected object (for highlighting)
   int     OnClick(string name); // processing the event of clicking on an object

   //--- functions of displaying
public:
   int     m_ndx, m_ndy; // size of margins from button for drawing
   int     m_bdx, m_bdy; // size of button of nodes
   CScrollView       m_scroll;
   bool    m_bProperty; // show properties near the node

   void    Draw(); // refresh the view
   void    DrawTree(CNode *first,int xpos,int &ypos,int &up,int &dn); // redraw
   void    DeleteView(CNode *root=NULL,bool delparent=true); // delete all displayed elements starting from a specified node
  };

//------------------------------------------------------------------    class CTreeView
class CTraceView: public CView
  {
   //--- base functions
public:
           CTraceView() { }; // constructor
           ~CTraceView() { Deinit(); } // destructor
   void    Deinit(); // full deinitialization of representation
   void    Create(long chart); // create and activate the representation

   //--- function of processing of state
public:
   int     m_hagent; // handler of the indicator-agent for sending messages
   CTraceCtrl   *m_trace; // pointer to created tracer
   CTreeView    *m_viewstack; // tree for displaying the stack
   CTreeView    *m_viewinfo; // tree for displaying of node properties
   CTreeView    *m_viewfile; // tree for displaying of the stack with grouping by files
   CTreeView    *m_viewclass; // tree for displaying of stack with grouping by classes
   void    OnTimer(); // handler of timer
   void    OnChartEvent(const int,const long&,const double&,const string&); // handler of event

   //--- functions of displaying
public:
   void    Draw(); // refresh objects
   void    DeleteView(); // delete the view

   void    UpdateInfoTree(CNode *node,bool bclear); // displaying the window of detailed information about a node
   string  TimeSeparate(long time); // special function for transformation of time into string
  };
```

We have chosen to display the stack in a separate subwindow as an optimal variant.

In other words, when the **CTraceView** class is created in the function **CTraceView::Create**, the chart window is created and all the objects are drawn in it, despite the fact that **CTraceView** is created and works in Expert Advisor in another window. It is done to prevent impeding of operation of the source code of the traced program and displaying of its own information on the chart by the huge amount of information.

But to make the interaction between two windows possible, we need to add an indicator to window, which will send all events of the user to the base window with the traced program.

The indicator is created in the same function **CTraceView::Create**. It has only one external parameter - the ID of chart to which it should send all the events.

The TraceAgent Indicator

```
#property indicator_chart_window
input long cid=0; // чарт получателя
//------------------------------------------------------------------    OnCalculate
int OnCalculate(const int rates_total, const int prev_calculated, const int begin, const double& price[])
{ return(rates_total); }
//------------------------------------------------------------------    OnChartEvent
void OnChartEvent(const int id,const long& lparam,const double& dparam,const string& sparam)
  {
    EventChartCustom(cid, (ushort)id, lparam, dparam, sparam);
  }
```

In the result, we have a pretty structured representation of the stack.

![](https://c.mql5.com/2/3/EURUSDH3__1.png)

In the tree TRACE displayed to the left, the initial stack is displayed.

Below it there is the INFO window containing detailed information about the selected node ( **CTraceView::OnChartEvent** in this example). Two adjacent window containing trees display the same stack, but it is grouped **by classes** (the CLASS tree in the middle) and **by files** (the FILE tree to the right).

The trees of classes and files have the embedded mechanism of synchronization with the main tree of the stack as well as convenient means of controlling. For example, when you click on a class name in the tree of classes all the functions of this class are selected in the tree of stack and in the tree of files. And in the same way, when you click on a file name all the functions and classes in the that file are selected.

![](https://c.mql5.com/2/3/111__1.png)

This mechanism allows quick selecting and viewing the required groups of functions.

### Features of Working with the Stack

- Adding Watch Parameters

As you have already noticed, the parameters of the **CNode** node include the array of structures **tagWatch**. It is created only for the convenience of representation of information. It contains a named value of a variable or expression.

Structure of a Watch Value

```
//------------------------------------------------------------------    struct tagWatch
struct tagWatch
{
    string m_name;     // name
    string m_val;    // value
};
```

To add a new Watch value to the current node, you need to call the **CTrace::AddWatch** function and use the **\_WATCH** macro.

```
#define _WATCH(w, v)         if (!NIL(m_trace) && !NIL(m_trace.m_cur)) m_trace.m_cur.AddWatch(w, string(v));
```

The special limitation on added values (the same as with nodes) is controlling of uniqueness of names. It means that the name of a Watch value is checked for uniqueness before it is added to the **CNode::m\_watch\[\]** array. If the array contains a value with the same name, the new one won't be added, but the value of the existing one will be updated.

All the tracked Watch values are displayed in the information window.

![](https://c.mql5.com/2/3/1.png)

- Stepwise execution of code.

Another convenient feature given by MQL5 is organization of a forced break in the code during its execution.

The pause is implemented using a simple infinite loop **while (true)**. The convenience of MQL5 here is handling of the event of exit from this loop - clicking the controlling red button. To create a break point during execution, use the **CTrace::Break** function.

The Function for Implementation of Break Points

```
//------------------------------------------------------------------    Break
void CTraceCtrl::Break() // checking the debug mode of a node
  {
   if(NIL(m_traceview)) return; // check of validity
   m_stack.BreakBy(TG_ALL,NULL,false); // removed the m_break flags from all nodes
   m_cur.m_break=true; // activated only at the current one
   m_traceview.m_viewstack.m_sid=m_cur.m_id; // moved selection to it
   m_stack.ExpandBy(TG_UP,m_cur,true,m_cur); // expand parent node if they are closed
   m_traceview.Draw(); // drew everything
   string name=m_traceview.m_viewstack.m_name+string(m_cur.m_id)+".dbg"; // got name of the BREAK button
   bool state=ObjectGetInteger(m_traceview.m_chart,name,OBJPROP_STATE);
   while(!state) // button is not pressed, execute the loop
     {
      Sleep(1000); // made a pause
      state=ObjectGetInteger(m_traceview.m_chart,name,OBJPROP_STATE);  // check its state
      if(!m_traceview.IsOpenView()) break; // if the window is closed, exit
      m_traceview.Draw(); // drew possible changes
     }
   m_cur.m_break=false; // removed the flag
   m_traceview.Draw(); // drew the update
  }
```

When meeting such break point, the stack trees are synchronized to display the function that called this macro. If a node is closed, the parent node will be expanded to display it. And if necessary, the tree is scrolled up or down to bring the node to the visible area.

![](https://c.mql5.com/2/3/2.png)

To exit from **CTraceCtrl::Break**, click the red button located near the node name.

### Conclusion

Well, we have an interesting "toy" now. While writing the article, I have tried many variants of working with **CTraceCtrl** and made sure that MQL5 has unique perspectives of controlling Expert Advisors and organizing their operation. All the features used for developing the tracer are unavailable in MQL4, what proves the advantages of MQL5 and its wide possibilities once again.

In the attached code, you can find all the classes described in the article together with service libraries (the minimum required set of them, since they're not the aim). In addition, I've attached the ready-made example - updated files of the standard library where the **\_IN** macros are placed. All the experiments were conducted with the Expert Advisor included in the standard delivery of MetaTrader 5 - **MACD Sample.mq5.**

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/272](https://www.mql5.com/ru/articles/272)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/272.zip "Download all attachments in the single ZIP archive")

[mql5.zip](https://www.mql5.com/en/articles/download/272/mql5.zip "Download mql5.zip")(24.67 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Working with sockets in MQL, or How to become a signal provider](https://www.mql5.com/en/articles/2599)
- [SQL and MQL5: Working with SQLite Database](https://www.mql5.com/en/articles/862)
- [Getting Rid of Self-Made DLLs](https://www.mql5.com/en/articles/364)
- [Promote Your Development Projects Using EX5 Libraries](https://www.mql5.com/en/articles/362)
- [Using WinInet in MQL5. Part 2: POST Requests and Files](https://www.mql5.com/en/articles/276)
- [The Prototype of a Trading Robot](https://www.mql5.com/en/articles/132)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/3647)**
(13)


![Denis Kirichenko](https://c.mql5.com/avatar/2019/5/5CEDB8D2-7CB7.jpg)

**[Denis Kirichenko](https://www.mql5.com/en/users/denkir)**
\|
27 Mar 2011 at 18:33

**sergeev:**

I think so. But usually in scripts the code is not branched much (unless of course the script is in a loop).

Besides, there is an inconvenience - scripts don't handle OnChartEvent event.

And if my script uses many different classes, class hierarchies?

I think it is necessary to sharpen the tool for scripts as well...

![---](https://c.mql5.com/avatar/avatar_na2.png)

**[\-\-\-](https://www.mql5.com/en/users/sergeev)**
\|
27 Mar 2011 at 19:05

CTraceView class doesn't care who calls it. It will make a tree and display it.

But scripts have an unsolvable feedback problem. You will not be able to actively work with the tree.

![Artapov Alexandr](https://c.mql5.com/avatar/avatar_na2.png)

**[Artapov Alexandr](https://www.mql5.com/en/users/artall)**
\|
6 Apr 2011 at 22:02

Dear sergeev, help me to understand!

I can't even reproduce the EA tree from the example, what am I doing wrong? It should show it:(

I took mql5-3.zip (the last one), unpacked the MQH folder into include\\expert\ - indicator into Indicators, EXPERT (example) into Expert folder.

Yes, and in the Object I put <Trace>.

I fixed all the paths, compiled - everything worked.

But further - I throw the [Indicator on the chart - the window](https://www.mql5.com/en/docs/chart_operations/chartindicatoradd "MQL5 Documentation: ChartIndicatorAdd Function") does NOT open; I throw the Expert - and then in its properties there is NO "YES, OK" button, only "Cancel and Reset".

Thank you.

![---](https://c.mql5.com/avatar/avatar_na2.png)

**[\-\-\-](https://www.mql5.com/en/users/sergeev)**
\|
6 Apr 2011 at 22:09

**artall:**

And further - I throw the [Indicator on the chart - the window is](https://www.mql5.com/en/docs/chart_operations/chartindicatoradd "MQL5 Documentation: ChartIndicatorAdd Function") NOT opened; I throw the Expert Advisor - and then in its properties there is NO "YES, OK" button, only "Cancel and Reset".

1\. you don't need to throw the indicator anywhere. the Expert Advisor will throw it by itself.

2\. read the [manual](https://www.mql5.com/ru/forum/3429/page1#comment_53455). last line of the post.

![Sumet Saengkeaw](https://c.mql5.com/avatar/2017/12/5A2413B3-8D91.jpeg)

**[Sumet Saengkeaw](https://www.mql5.com/en/users/7486585)**
\|
3 Dec 2017 at 18:10

**MetaQuotes Software Corp.:**

New article [Tracing, Debugging and Structural Analysis of Source Code](https://www.mql5.com/en/articles/272) has been published:

Author: [o\_Omp](https://www.mql5.com/en/users/sergeev "sergeev").5

![How to Order an Expert Advisor and Obtain the Desired Result](https://c.mql5.com/2/0/Order_EA_MQL5_Job.png)[How to Order an Expert Advisor and Obtain the Desired Result](https://www.mql5.com/en/articles/235)

How to write correctly the Requirement Specifications? What should and should not be expected from a programmer when ordering an Expert Advisor or an indicator? How to keep a dialog, what moments to pay special attention to? This article gives the answers to these, as well as to many other questions, which often don't seem obvious to many people.

![The Indicators of the Micro, Middle and Main Trends](https://c.mql5.com/2/0/three_degrees_of_trend.png)[The Indicators of the Micro, Middle and Main Trends](https://www.mql5.com/en/articles/219)

The aim of this article is to investigate the possibilities of trade automation and the analysis, on the basis of some ideas from a book by James Hyerczyk "Pattern, Price & Time: Using Gann Theory in Trading Systems" in the form of indicators and Expert Advisor. Without claiming to be exhaustive, here we investigate only the Model - the first part of the Gann theory.

![Using MetaTrader 5 Indicators with ENCOG Machine Learning Framework for Timeseries Prediction](https://c.mql5.com/2/0/brain.png)[Using MetaTrader 5 Indicators with ENCOG Machine Learning Framework for Timeseries Prediction](https://www.mql5.com/en/articles/252)

This article presents connecting MetaTrader 5 to ENCOG - Advanced Neural Network and Machine Learning Framework. It contains description and implementation of a simple neural network indicator based on a standard technical indicators and an Expert Advisor based on a neural indicator. All source code, compiled binaries, DLLs and an exemplary trained network are attached to the article.

![The Implementation of Automatic Analysis of the Elliott Waves in MQL5](https://c.mql5.com/2/0/MQL5_Elliott_Waves_Automated.png)[The Implementation of Automatic Analysis of the Elliott Waves in MQL5](https://www.mql5.com/en/articles/260)

One of the most popular methods of market analysis is the Elliott Wave Principle. However, this process is quite complicated, which leads us to the use of additional tools. One of such instruments is the automatic marker. This article describes the creation of an automatic analyzer of Elliott Waves in MQL5 language.

[Best articles and CodeBase updates in MQL5.community channelsFollow us to ensure you never miss out on important updates![](https://www.mql5.com/ff/sh/n9yf51p2srwzfqh5z2/02.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=dgazvhktsxqakdvarucjbvmvzenwlyje&s=98a038fe082e458df8c4a1d8e116e3a6646fd5517f06e48b2356b7ee005817d6&uid=&ref=https://www.mql5.com/en/articles/272&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5083220436570871608)

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