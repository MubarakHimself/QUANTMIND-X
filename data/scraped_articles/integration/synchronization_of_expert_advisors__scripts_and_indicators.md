---
title: Synchronization of Expert Advisors, Scripts and Indicators
url: https://www.mql5.com/en/articles/1393
categories: Integration, Indicators, Expert Advisors
relevance_score: 6
scraped_at: 2026-01-23T17:27:38.806597
---

[Need a reliable hosting solution for your robots?Contact your broker and find out about available Sponsored MetaTrader VPS offeringsLearn more![](https://www.mql5.com/ff/sh/0pw0dk81s56qy774z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=vljwvezfjkfbvviocwskggexlvgykvob&s=70cf8e354b9a125332533ffb65d7365abe8dde5b5c1ede9caac479a9e9df4f25&uid=&ref=https://www.mql5.com/en/articles/1393&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5068260820665694041)

MetaTrader 4 / Examples


### Introduction

There are three kinds of programs written in MQL4 and executed in the MetaTrader
4 Client Terminal:

\- Expert Advisors;

\- scripts;

\- indicators.

Each of them is intended for solving a certain range of problems. Let us give a
brief description of the programs.

### 1\. Brief Description of MQL4 User Programs

**1.1. Expert Advisors**

Expert Advisors (EAs) are the main kind of programs used to realize profitable trading
strategies. The distinctive features of an EA are listed below:

1\. Ability to use embedded functions that support trades.

2\. Ability to modify external settings manually.

3\. Availability of rules that regulate launching of special function **start().** It is launched tickwise. At the moment when a new tick incomes, parameters of the
entire environment available for this function are updated. For example, such variables
as **bid** and **ask** take new values. Having completed the code execution, namely - having reached operator
**return,** function **start()** finishes its operation and sleeps until a new tick incomes.

**1.2. Scripts**

Scripts are very similar to Expert Advisors, but their features are a bit different.
Main feautres of scripts are listed below:

1\. Scripts can use the functions of trades, too.

2\. Parameters of external settings cannot be changed in scripts.

3\. The main feature of scripts is the rule, according to which special function
**start()** of a script will be launched only once, immediately after it has been attached
to the chart and initialized.

Expert Advisors and scripts are attached to the main window of a symbol, they cannot
have special subwindows.

**1.3. Indicators**

Unlike Expert Advisors and scripts, indicators have another intent:

1\. The main feature of indicators is the possibility to draw continuous curves that
display one or another principle according to the idea implied in them.

2\. Trade functions cannot be used in indicators.

3\. Indicators are launched tickwise.

4\. Regarding the parameters implied, the indicator can serve its purpose in the
main symbol window or in its own subwindow.

We listed above only main characteristics of custom programs, namely - those we
will need in our further speculations.

As we can see from the description above, no custom program has properties of all
programs: Expert Advisors and scripts cannot draw, indicators may not trade, etc.

If our trading system needs to use all properties of custom programs during trading,
the only solution will be simultaneous use of an Expert Advisor, a script and an
indicator.

### 2\. Problem Statement

Let us consider criteria that provide necessity of simultaneous use of all custom
programs.

**2.1. Timeliness**

Any control by the user must be executed immediately. A program based on an Expert
Advisor is not always suitable for this purpose. The main disadvantage of Expert
Advisors is its inonsusceptibility to external actions. The reason for this limitation
is very simple: Expert Advisor's basic code is launched tickwise. What will happen
if the user commands the EA to close an order and the EA is waiting for the next
tick? The answer to this question depends on how the Expert Advisor has been written.
In some cases, the command will be executed, but with some delay.

The program can be organized in such a way that the main code of the Expert Advisor
is executed continuously, without breaks between ticks. For this purpose, it is
necessary to organize in special function

**start()**

an infinite loop, in which the entire main code of the program will be placed. If
the environmental information is forcedly updated at the start of every loop, the
whole complex can work successfully. Disadvantage of a looped Expert Advisor is
the impossibility to open the setup panel. Try to loop an EA - and you won't be
able to set it up.

The same idea can be successfully realized using a script. This means an infinite
loop can be organized in a script. But there are no parameters to be set up in
scripts.

The customizability of the trading system and timeliness of executing all user's
commands in the continuous operation mode can be provided only using simultaneously
an Expert Advisor for setting up and a script for instant execution.

**2.2. Awareness**

In some cases, there is a necessity to get some information about trading. For example,
every trader would like to know that, at a certain moment (say two minutes prior
to important news being published), the dealing center has changed its normal 10
points of the minimal tolerable distance to place pending orders for 20 points.
Besides, as a rule, the trader wants to know the reason why the trade server refuses
to execute orders. This and other useful information can be shown as a text in
the indicator window. As well, on an on-going basis, the lines containing older
messages can be moved up to empty space for new messages from the trading system.
In this case, it is necessary to combine the trading Expert Advisor or a script
with a displaying indicator.

**2.3. Controls**

If you use a trading system that involves an enhanced interface, the controls (graphical
objects) are better to be placed in the indicator window. So we can be sure that
candle trend will not overlap our control items and, thus, will not disturb controlling.

**2.4. System Requirements**

The main requirement to the end product, in this case, is **synchronous operation**, so, developing a system based on all three types of programs, it is necessary
to dissociate tasks to be solved by all its components. Regarding special features
of each type of programs in our system, we can define the following properties
for them:

**script**\- gives basic code containing analytical and trading functions;

**Expert Advisor** \- provides setup panel;

**indicator** \- provides subwindow field to display controls and information.

### 3\. Software Solutions

Let us indicate at once that we consider here the structure of an application based
on three components within the minimum required range. If you decide to use the
application in practice, you will have to elaborate it by yourself as it concerns
analytical and trading operations. But the material given below is quite sufficient
for developing a structure.

**3.1. Expert Advisor**

Let us consider in details what an Expert Advisor consists of and how it works.

```
// Expert.mq4
//====================================================== include =======
#include <stdlib.mqh>
#include <stderror.mqh>
#include <WinUser32.mqh>
//======================================================================
#include <Peremen_exp.mq4>  // Description of the EA variables.
#include <Metsenat_exp.mq4> // Predefinition of the EA variables.
#include <Del_GV_exp.mq4>
// Deletion of all GlobalVariables created by the Expert Advisor.
#include <Component_exp.mq4> // Checking for availability of components.
#include <Component_uni.mq4>
// Message in the indicator that components are not available.
//======================================================================
//
//
//======================================================================
int init()
  {
    Fishka=1;          // We are in init()
    Metsenat_exp();   // Predefinition of the EA variables.
    Component_exp();  // Check for availability of components
    return;
 }
//=====================================================================
int start()
  {
    Fishka=2;         // We are in start()
    Component_exp();  // Check for availability of components
    return;
 }
//=====================================================================
int deinit()
  {
    Fishka=3;         // We are in deinit()
    Component_exp();  // check for availability of components
    Del_GV_exp();     // Deletion of the Expert Advisor's GlobalVariable.
    return;
 }
//======================================================================
```

In special function init(), two functions are working - Metsenat\_exp() and Component\_exp()

**Metsenat\_exp()**

\- a function of predefinition of some variables.

```
// Metsenat_exp.mq4
//=================================================================
int Metsenat_exp()
 {
//============================================ Predefinitions =====
   Symb     = "_"+Symbol();
   GV       = "MyGrafic_GV_";
//============================================= GlobalVariable ====
   GV_Ind_Yes = GV+"Ind_Yes"   +Symb;
// 0/1 confirms that the indicator is loaded
   GV_Scr_Yes = GV+"Scr_Yes"   +Symb;
// 0/1 confirms that the script is loaded
//-------------------------------------------- Public Exposure ----
   GV_Exp_Yes = GV+"Exp_Yes"   +Symb;
   GlobalVariableSet(GV_Exp_Yes, 1 );
   GV_Extern  = GV+"Extern"    +Symb;
   GlobalVariableSet(GV_Extern,  1 );
//  AAA is used as an example:
   GV_AAA     = GV+"AAA"       +Symb;
GlobalVariableSet(GV_AAA,   AAA );
//==================================================================
   return;
 }
//====================== End of Module =============================
```

One of the tasks of the entire application maintenance is the task of tracking the
availability of all components. This is why all components (the script, the Expert
Advisor and the indicator) must trace each other and, if a component is not available,
stop working and inform the user about this. For this purpose, each program informs
about its availability at startup through publishing a global variable. In the
given case, in function Metsenat\_exp() of the Expert Advisor, this will be done
in the line below:

```
   GV_Exp_Yes = GV+"Exp_Yes"   +Symb;
   GlobalVariableSet(GV_Exp_Yes, 1 );
```

Function Metsenat\_exp() is controlled by function init() of the EA, i.e., it is
used only once during loading or changing values of extern variables. The script
must 'know' about changed settings, this is why the Expert will inform the script
about it through changing the value of global variable GV\_Extern:

```
   GV_Extern  = GV+"Extern"    +Symb;
   GlobalVariableSet(GV_Extern,  1 );
```

**Component\_exp()**

\- a function intended for completeness controlling. The further scenario depends
on the Expert Advisor's special function, in which the Component\_exp() is used.

```
// Component_exp.mq4
//===============================================================================================
int Component_exp()
 {
//===============================================================================================
   while( Fishka < 3 &&     // We are in init() or in start() and..
      (GlobalVariableGet(GV_Ind_Yes)!=1 ||
       GlobalVariableGet(GV_Scr_Yes)!=1))
    {                            // ..while a program is not available.
      Complect=0;                // Since one is inavailable, it is a deficiency
      GlobalVariableSet(GV_Exp_Yes, 1);
// Inform that the EA is available
//-----------------------------------------------------------------------------------------------
      if(GlobalVariableGet(GV_Ind_Yes)==1 &&
         GlobalVariableGet(GV_Scr_Yes)!=1)
        {//If there is an indicator but there is no scrip available, then..
         Graf_Text = "Script is not installed.";
// Message text
         Component_uni();
// Write the message text in the indicator window.
        }
//-----------------------------------------------------------------------------------------------
      Sleep(300);
    }
//===============================================================================================
     if(Complect==0)
    {
      ObjectDelete("Necomplect_1");
// Deletion of unnecessary messages informing about inavailability of components
      ObjectDelete("Necomplect_2");
      ObjectsRedraw();              // For quick deletion
      Complect=1;        // If we have left the loop, it means that all components are available
    }
//===============================================================================================
   if(Fishka == 3 && GlobalVariableGet(GV_Ind_Yes)==1)
// We are in deinit(), and there is space to write the indicator into
    {
//-----------------------------------------------------------------------------------------------
      if(GlobalVariableGet(GV_Scr_Yes)!=1)  // If there is no script available,
       {
         Graf_Text = "Components Expert and Script are not installed";
// Message (since we are unloading)
         Component_uni();     // Write the message text in the indicator window
       }
//-----------------------------------------------------------------------------------------------
    }
//===============================================================================================
   return;
 }
//===================== End of the module =======================================================
```

Availability of script and indicator is traced on the basis of reading values of
the corresponding global variables - GV\_Scr\_Yes and GV\_Ind\_Yes. If neither of the
components is available, the control will be given to the infinite loop until the
completeness is achieved, i.e., until both indicator and script are installed.
The application will inform user about the current state through function

**Component\_uni()**. It is a universal function included in all components.

```
// Component_uni.mq4
//================================================================
int Component_uni()
 {
//================================================================
//----------------------------------------------------------------
   Win_ind = WindowFind("Indicator");
// What is our indicator's window number?
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   ObjectCreate ( "Necomplect_1", OBJ_LABEL, Win_ind, 0, 0  );
// Create an object in the indicator window
   ObjectSet    ( "Necomplect_1", OBJPROP_CORNER,        3  );
// coordinates related to the bottom-right corner
   ObjectSet    ( "Necomplect_1", OBJPROP_XDISTANCE,   450  );
// coordinates on X..
   ObjectSet    ( "Necomplect_1", OBJPROP_YDISTANCE,    16  );
// coordinates on Y..
   ObjectSetText("Necomplect_1", Graf_Text,10,"Courier New",Tomato);
// text, font, and color
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   Graf_Text = "Application does not work.";
 // Message text
   ObjectCreate ( "Necomplect_2", OBJ_LABEL, Win_ind, 0, 0);
// Create an object in the indicator window
   ObjectSet    ( "Necomplect_2", OBJPROP_CORNER,        3);
// coordinates related to the bottom-right corner
   ObjectSet    ( "Necomplect_2", OBJPROP_XDISTANCE,   450);
// coordinates on Х..
   ObjectSet    ( "Necomplect_2", OBJPROP_YDISTANCE,     2);
// coordinates on Y..
   ObjectSetText("Necomplect_2", Graf_Text,10,"Courier New",Tomato);
// text, font, color
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   ObjectsRedraw();                                // Redrawing.
   return;
//================================================================
 }
//===================== End of module ============================
```

As soon as the application is complete, the control in the EA will be given from
the loop to the sequent code where the unnecessary messagy about incompleteness
will be deleted.

When the EA is unloaded, special function deinit() will also call Component\_exp(),
but for another purpose - to inform about the unloading at the current moment.

In the Expert Advisor's deinit(), function **Del\_GV\_exp()** will be called, as well.

It is used to delete all GlobalVariables opened by the EA. According to the unwritten
rule, each program being unloaded must "clean the room", i.e., delete
global variables and graphical objects created before.

```
// Del_GV_exp.mq4
//=================================================================
int Del_GV_exp()
 {
//=================================================================
   GlobalVariableDel(GV_Exp_Yes      );
   GlobalVariableDel(GV_Extern       );
   GlobalVariableDel(GV_AAA          );
//=================================================================
   return;
 }
//====================== End of module ============================
```

Thus, the Expert Advisor starts working and tracks the availability of the two other
components at all stages: once in init(), once in deinit() and at every tick -
in start(). This EA construction makes it possible to use the program for solving
our task - **making the setup panel available.** The file describing variables contains, as an example, variable ААА and its corresponding global variable GV\_AAA,
the value of which is read from the script.

To get into details of how all this works, let us consider the structure of a script.

**3.2. Script**

Script code:

```
// Script.mq4
//==================================================== include ====
#include <stdlib.mqh>
#include <stderror.mqh>
#include <WinUser32.mqh>
//=================================================================
#include <Peremen_scr.mq4>
// File describing variables of the script.
#include <Metsenat_scr.mq4>
// Predefining of variables of the script.
#include <Mess_graf_scr.mq4>
// List of graphical messages.
#include <Novator_scr.mq4>
// Environment scanning, obtaining new values for some variables
#include <Del_GV_scr.mq4>
// Deletion of all GlobalVariables created by the script.
#include <Component_scr.mq4>
// Checking for components availability.
#include <Component_uni.mq4>
// Message in the indicator about inavailability of components.
#include <Del_Obj_scr.mq4>
// Deletion of all objects created by the program complex.
#include <Work_scr.mq4>
// The main working function of the script.
//=================================================================
//
//
//=================================================================
int init()
 {
   Fishka = 1;                                // We are in init()
   Metsenat_scr();       // Predefining of variables of the script.
   return;
 }
//================================================================
int start()
 {
   Fishka = 2;                               // We are in start()
   while(true)
    {
      Component_scr();  // Checking for availability of components
      Work_scr();       // The main working function of the script.
    }
   return;
 }
//=================================================================
int deinit()
 {
   Fishka = 3;                                // We are in deinit()
   Component_scr();      // Checking for availability of components
   Del_Obj_scr();          // Deletion of graphical objects created
   Del_GV_scr();        // Deletion of GlobalVariable of the script.
   return;
 }
//==================================================================
```

The basis of the code is the availability of an infinite loop in special function
start(). In the script code, functions with similar names and contents are applied.
We turn our attention to their special features. At the beginning of every loop,
function

**Component\_scr()**

is called.

```
// Component_scr.mq4
//====================================================================
int Component_scr()
 {
//====================================================================
   Iter=0;                               // Zeroize iteration counter
   while (Fishka <3 &&              // We are in init() or in start()
      (GlobalVariableGet(GV_Ind_Yes)!=1 ||
       GlobalVariableGet(GV_Exp_Yes)!=1))
    {                                 // Until a program is available
      GlobalVariableSet(GV_Scr_Yes, 1);
// Declare about the script availability
//--------------------------------------------------------------------
      Iter++;                                    // Iteration counter
      if(Iter==1)                         // Skip the first iteration
       {
         Sleep(500);
         continue;
       }
//--------------------------------------------------------------------
      if(Iter==2)             // Take measures on the second iteration
       {
         Complect=0; // Program is not available, it is incompleteness
         for (i=0;i<=31;i++)ObjectDelete(Name_Graf_Text[i]);
// Deletion of all strings
// Here, a function can be inserted that will zeroize trade queue.
       }
//--------------------------------------------------------------------
      if(GlobalVariableGet(GV_Ind_Yes)==1 &&
          GlobalVariableGet(GV_Exp_Yes)!=1)
       {                       // If there is an indicator, but no EA
         Graf_Text = "Expert has not been installed.";
// Message text
         Component_uni();
// Write the text message in the indicator window.
       }
//-----------------------------------------------------------------
      Sleep(300);
    }
//-----------------------------------------------------------------
   if(Complect==0)                // Process it once at completing.
    {
      ObjectDelete("Necomplect_1");
// Deletion of unnecessary messages..
      ObjectDelete("Necomplect_2");
// ..about incompleteness of components
      Mess_graf_scr(1);
// Inform the user about completeness
      if( IsExpertEnabled())
// The button is enabled
       {
         Mess_graf_scr(3000);
         Knopka_Old = 1;
       }
      if(!IsExpertEnabled())
// The button is disabled
       {
         Mess_graf_scr(4000);
         Knopka_Old = 0;
       }
      Complect=1;
// The minimal installation completeness reached
      Redraw = 1;
// For quick deletion
    }
//====================================================================
   if(Fishka == 3 && GlobalVariableGet(GV_Ind_Yes)==1)
// We are in deinit()
    {
      for(i=0;i<=31;i++)ObjectDelete(Name_Graf_Text[i]);
// Deletion of all strings
//--------------------------------------------------------------------
      if(GlobalVariableGet(GV_Exp_Yes)!=1)
// There is indicator, but no Expert Advisor
         Graf_Text="Components Expert and Script are not installed.";
// Message (as we're unloading)
      if(GlobalVariableGet(GV_Exp_Yes)==1)
// If there are both indicator and EA, then..
         Graf_Text="The Script has not been installed.";
// Message (as we're unloading)
      Component_uni();   // Write the message in the indicator window.
//--------------------------------------------------------------------
      ObjectsRedraw();                    // For quick deletion
    }
//====================================================================
   return;
 }
//====================== End of module ===============================
```

The first demand on a script is

**continuity** of its operation. During updating of extern variables, the Expert Advisor goes through the complete installation.
When pressing OK on the EA setup panel, it will be unloaded and give control to deinit(), then it will immediately
load again going through init() and start() sequentially. As a result, the Expert Advisorб though for a short
time, deletes from deinit() the global variable that confirms its availability.

In order the script does not suppose that the EA has not been loaded at all, function
**Component\_scr()** contains a small block that disables making a decision at the first iteration:

```
      Iter++;                         // Iteration counter
      if(Iter==1)              // Skip the first iteration
        {
          Sleep(500);
          continue;
        }
```

Five hundred milliseconds will be, in most cases, enough for complete loading of
the Expert Advisor. If you use a more consuming code in the Expert Advisor's init(),
the time must be prolonged. If the EA has not been detected at the second iteration,
the decision will be made that the EA is not available at all, and the script stops
operating.

```
      Complect = 0;    // A program is not available, incompleteness
```

The expression "the script stops operating" is used in the preceding paragraph.
In our example, there is no code responsible for this phenomenon, just because
it is beyond the scope of our topic and this article. in the place where you can
put calling for the corresponding functions, there is a comment in the code.

```
// Here, a function zeroizing trade counter can be inserted.
```

In function

**Work\_scr()** of a really working program, except for functions used in our example, other functions
are taken to be that are responsible for a certain order of events. For example,
if your program is adjusted to modify several orders, it will surely contain an
array, in which the queue for execution of trades will be stored if there is a
number of such trades occurring in the current tick.

If incompleteness occurs (for example, the Expert Advisor or the script has been
unloaded inadvertently) at the moment when such a queue takes place, it is necessary
to disable trading which can be achieved by zeroizing the above-mentioned array
of the trade queue and, perhaps, some other variables according to the situation.

The infinite loop of the script also contains function **Work\_scr().** This is the script main function where its entire main code must be placed.

```
// Work_scr.mq4
//=================================================================
int Work_scr()
 {
   Novator_scr();
//-----------------------------------------------------------------
   // Basic code of the entire application.
//------------------------------------------------ For example ----
   if(New_Tick==1)                             // At every new tick
    {
      Alert ("The current value of ААА = ", AAA);
    }
//-----------------------------------------------------------------
   if(Redraw==1)
    {
      ObjectsRedraw();                    // To display immediately
      Redraw=0;                         // Unflag objects redrawing
    }
//-----------------------------------------------------------------
   Mess_graf_scr(0);
   Sleep(1);                           // Just in case, from reload
   return;
 }
//====================== End of module ============================
```

The **Work\_scr()** contains function **Novator\_scr()** intended for updating environmental variables used in the basic code.

```
// Novator_scr.mq4
//===================================================================
int Novator_scr()
 {
//===================================================================
//---------------------------------------- Updating of settings -----
   if(GlobalVariableGet(GV_Extern)==1)
// There is an update in the EA
    {
      Metsenat_scr();         // Updating of the script variables.
      Mess_graf_scr(5000);    // Message about a new setting.
      Redraw=1;               // Redrawing at the end of the loop.
    }
//--------------------------------- EA button state -----------------
   Knopka = 0;                                         // Preset
   if( IsExpertEnabled()) Knopka = 1;
// Check for the real state of the button

   if(Knopka==1 && Knopka_Old==0)
// If the state has changed for ON
    {
      Knopka_Old = 1;                // This will be the old one
      Mess_graf_scr(3);              // Inform the user about changes
    }
   if(Knopka==0 && Knopka_Old==1)
// If the state has changed for OFF
    {
      Knopka_Old = 0;                 // This will be the old one
      Mess_graf_scr(4);              // Inform the user about changes
    }
//-------------------------------------------------- New tick --------
   New_Tick=0;                              // First of all, zeroize
   if (RefreshRates()==true) New_Tick=1;
// It is easy to catch a new tick if you know how to do it
//--------------------------------------------------------------------
//====================================================================
   return;
 }
//=====================; End of module ===============================
```

Let us consider the necessity of this function in more details. We mentioned at
the beginning of the article that every time when the EA is loaded, as well as
when its variables are updated, its subordinated function **Metsenat\_exp()** sets the value of variable GV\_Extern as **1**. For the script, it means that the settings must be updated. For this purpose,
function Novator\_scr() contains the following block:

```
//---------------------------------------- Updating settings ----
   if (GlobalVariableGet(GV_Extern)==1)
// An update has taken place in the EA
    {
      Metsenat_scr();              // Updating script settings.
      Mess_graf_scr(5000);         // New setting message.
      Redraw=1;                    // Redrawing at the end of the loop.
    }
```

The value of the above variable is analyzed here and, in case of necessity to update,
function

**Metsenat\_scr()** is called, which makes the updating (reading of new values of global variables).

```
// Metsenat_scr.mq4
//===================================================================
int Metsenat_scr()
  {
//========================================================== int ====
//======================================================= double ====
//======================================================= string ====
      MyGrafic    = "MyGrafic_";
    Mess_Graf   = "Mess_Graf_";
    Symb        = "_"+Symbol();
    GV          = "MyGrafic_GV_";
//=============================================== GlobalVariable ====
     GV_Ind_Yes  = GV+"Ind_Yes" +Symb;
// 0/1 confirms that the indicator has been loaded
     GV_Exp_Yes  = GV+"Exp_Yes" +Symb;
// 0/1 confirms that the EA has been loaded
//-------------------------------------------------- Publishing -----
     GV_Scr_Yes  = GV+"Scr_Yes" +Symb;
    GlobalVariableSet(GV_Scr_Yes,          1 );
    GV_Extern   = GV+"Extern"  +Symb;
    GlobalVariableSet(GV_Extern,           0 );
//--------------------------------------------------- Reading -------
                                             //  AAA is used as an example:
     GV_AAA      = GV+"AAA"     +Symb;
   AAA  = GlobalVariableGet(GV_AAA);
//===================================================================
     return;
 }
//======================== End of module ============================
```

Function Metsenat\_scr(), in its turn, sets value of global variable GV\_Extern as
**0**. In the subsequent history, this variable remains equal to 0 until the user opens
the EA's setup window.

Note that, in spite of the fact that due to its changed settings the EA goes through
all stages of unloading and loading, the script does not stop working during the
user is changing the settings or after that. Thus, the combined usage of the Expert
Advisor and the script helps to meet the requirement of operation **continuity** of the application and allows the user to change settings, i.e., to **control** the process.

In the subsequent blocks of function **Novator\_scr()**, the EA's button is controlled that enables it to trade. Then a new tick is detected.
If your trading system assumes using of those and similar parameters, it is function **Novator\_scr()** that is intended for such calculations.

For example, you can дcomplete this function with blocks that detect whether a

**new bar** has appeared, check whether critical event

**time** has come, detect whether the

**trade terms** have been changed (for example, spread size, minimal distance at which the stop
orders may be placed, etc.), as well as with other calculations necessary before
the basic analytical functions start operating.

Functions that make the basic content of the program are not shown in function **Work\_scr()**. In the article named [Considering Orders in a Large Program](https://www.mql5.com/en/articles/1390), we dealt with function **Terminal()** that considered orders. If you use the same considering principle in your trading system, function **Terminal()** should be included into function **Work\_scr()** immediately after function **Novator\_scr().**

The script has one more auxilliary function at its disposal - **Mess\_graf\_scr()** intended for displaying messages in the indicator window.

```
// Mess_graf_scr.mq4
//====================================================================
int Mess_graf_scr(int Mess_Number)
 {
//====================================================================
   if(Mess_Number== 0)        // This happens in every loop Work
    {
      if(Time_Mess>0 && GetTickCount()-Time_Mess>15000)
// Color print has outdated within the last
       {                       // ..15 sec, let's color lines gray
         ObjectSet(Name_Graf_Text[1],OBJPROP_COLOR,Gray);
// Last 2 lines
         ObjectSet(Name_Graf_Text[2],OBJPROP_COLOR,Gray);
// The last 2 lines
         Time_Mess=0;         // Additional flag not to color in vain
         Redraw=1;            // Then redraw
       }
      return;                 // It was a little step into
    }
//--------------------------------------------------------------------
   Time_Mess=GetTickCount(); // Remember the message publishing time
   Por_Nom_Mess_Graf++;      // Count lines. This is just a name part.
   Stroka_2=0;            // Presume that message in one line
   if(Mess_Number>1000)
// If a huge number occurs, the number will be brought to life,
// understand that the previous line is from the same message, i.e.,it
// should not be colored gray
    {
      Mess_Number=Mess_Number/1000;
      Stroka_2=1;
    }
//====================================================================
   switch(Mess_Number)
    {
//--------------------------------------------------------------------
      case 1:
         Graf_Text = "All necessary components installed.";
         Color_GT = LawnGreen;
         break;
//--------------------------------------------------------------------
      case 2:
         Graf_Text = " ";
         break;
//--------------------------------------------------------------------
      case 3:
         Graf_Text = "Expert Advisors enabled.";
         Color_GT = LawnGreen;
         break;
//--------------------------------------------------------------------
      case 4:
         Graf_Text = "Expert Advisors disabled.";
         Color_GT = Tomato;
         break;
//--------------------------------------------------------------------
      case 5:
         Graf_Text = "Expert Advisor settings have been updated.";
         Color_GT = White;
         break;
//---------------------------------------------------- default -------
      default:
         Graf_Text = "Line default "+ DoubleToStr( Mess_Number, 0);
         Color_GT = Tomato;
         break;
    }
//====================================================================
   ObjectDelete(Name_Graf_Text[30]);
// the 30th object is preempted, delete it
   int Kol_strok=Por_Nom_Mess_Graf;
   if(Kol_strok>30) Kol_strok=30;
//-----------------------------------------------------------------
   for(int lok=Kol_strok;lok>=2;lok--)
// Go through graphical text names
    {
      Name_Graf_Text[lok]=Name_Graf_Text[lok-1];
// Reassign them (normalize)
      ObjectSet(Name_Graf_Text[lok],OBJPROP_YDISTANCE,2+14*(lok-1));
//Change Y value (normalize)
      if(lok==3 || lok==4 || (lok==2 && Stroka_2==0))
         ObjectSet(Name_Graf_Text[lok],OBJPROP_COLOR,Gray);
//Color old lines gray..
    }
//-------------------------------------------------------------------
   Graf_Text_Number=DoubleToStr( Por_Nom_Mess_Graf, 0);
//The unique part of the name unite with the message number
   Name_Graf_Text[1] = MyGrafic + Mess_Graf + Graf_Text_Number;
// Form the message name.
   Win_ind= WindowFind("Indicator");
//What is the window number of our indicator?

   ObjectCreate ( Name_Graf_Text[1],OBJ_LABEL, Win_ind,0,0);
// Create an object in the indicator window
   ObjectSet    ( Name_Graf_Text[1],OBJPROP_CORNER, 3   );
// ..with coord, from the bottom-right corner..
   ObjectSet    ( Name_Graf_Text[1],OBJPROP_XDISTANCE,450);
// ..with coordinates on X..
   ObjectSet    ( Name_Graf_Text[1],OBJPROP_YDISTANCE, 2);
// ..with coordinates on Y..
   ObjectSetText(Name_Graf_Text[1],Graf_Text,10,"Courier New",
                 Color_GT);
//text font color
   Redraw=1;                                  // Then redraw
//=================================================================
   return;
 }
//====================== End of module ============================
```

There is no need to consider this function in all details. We can just mention some
of its special features.

1\. All messages are displayed by graphical means.

2\. The formal parameter to be passed to the function corresponds to the message
number.

3\. If the value of the passed parameter lies between 1 and 999, the preceding text
line in the indicator window will lose the color. If this value exceeds 1000, the
message will be displayed, the number of which equals to the passed value divided
by 1000. In this latter case, the preceding line will not lose its color.

4\. At the end of fifteen seconds after the last message, all lines will lose their
color.

5\. Maintaining the possibility to discolor lines, the function should be activated
from time to time. So, there is a call at the end of function **Work\_scr()**:

```
   Mess_graf_scr(0);
```

In the article named [Graphic Expert Advisor: AutoGraf](https://www.mql5.com/en/articles/1378), a really working program complex is represented where we use a similar function
that contains over 250 various messages. You can refer to this example to use all
or some of the messages in your trading.

**3.3. Indicator**

For our presentment to be complete, let us consider the indicator, as well, though
its code is rather simple.

```
// Indicator.mq4
//===================================================; include ==========
#include <stdlib.mqh>
#include <stderror.mqh>
#include <WinUser32.mqh>
//=======================================================================
#include <Peremen_ind.mq4>
// Description of the indicator variables.
#include <Metsenat_ind.mq4>
// Predefining the indicator variables.
#include <Del_GV_ind.mq4>
// Deletion of all GlobalVariables created by the indicator.
#include <Component_ind.mq4>
// Check components for availability.
#include <Component_uni.mq4>
// Message in the indicator about inavailability of components.
//=======================================================================
//
//
//=======================================================================
#property indicator_separate_window
//=======================================================================
//
//
//=======================================================================
int init()
 {
   Metsenat_ind();
   return;
 }
//=======================================================================
int start()
 {
   if(Component_ind()==0) return; // Check for availability of components
   //...
   return;
 }
//=======================================================================
int deinit()
 {
   Del_GV_ind();             // Deletion of the indicator GlobalVariable.
   return;
 }
//=======================================================================
```

Only one critical feature of an indicator should be emphasized here: Indicator is
shown in a separate window:

```
#property indicator_separate_window
```

Contents of functions Metsenat\_ind() and Del\_GV\_ind() are similar to those of previously
considered functions used in the Expert Advisor and in the script.

Content of function **Component\_ind()** is unsophisticated, too:

```
// Component_ind.mq4
//===================================================================
int Component_ind()
 {
//===================================================================
   if(GlobalVariableGet(GV_Exp_Yes)==1 &&
      GlobalVariableGet(GV_Scr_Yes)==1)
//If all components are available
    {                        // State about the indicator available
      if(GlobalVariableGet(GV_Ind_Yes)!=1)
          GlobalVariableSet(GV_Ind_Yes,1);
      return(1);
    }
//--------------------------------------------------------------------
   if(GlobalVariableGet(GV_Scr_Yes)!=1 &&
      GlobalVariableGet(GV_Exp_Yes)!=1)
    {                           // If there is neither script nor EA
      Graf_Text = "Components Expert and Script are not installed.";
// Message text
      Component_uni();        // Write the info message in ind. window
    }
//=====================================================================
   return(0);
 }
//=====================; End of module ================================
```

As we can see from the code, function Component\_ind() gives a message only if two other components have not been loaded
\- both the script and the Expert Advisor. If only one of programs is inavailable,
no actions will be made. This oresumes that, if they are available in the symbol
window, these programs will track the composition of the program complex and inform
the user about the results.

If it is necessary, the main property of the indicator - drawing - can be used,
too. In terms of the program complex, this property is not necessary, but it can
be used in the real trading, for example, to divide the subwindow into zones.

### 4\. Practical Use

The order of attaching the application components to the symbol window does not
signify. However, it would be recommended to attach indicator as the first since
it allows us to read comments as soon as it is attached.

Thus, the following must be done to demonstrate how the application works.

1\. Attach the **indicator** in the symbol window. This will be shown in the indicator window immediately:

|     |
| --- |
| **Components Expert and Script are not installed**<br>**Application does not work.** |

2\. Attach the **Expert Advisor** in the symbol window. Function Component\_exp() will trigger and the following message
will appear in the indicator window:

|     |
| --- |
| **The Script is not installed**<br>**Application does not work.** |

3\. Attach the **script** in the symbol window. This event will be processed in function Component\_scr()
of the script and displayed in the indicator window:

|     |
| --- |
| **All necessary components are installed.**<br>**Expert Advisors enabled.** |

If Expert Advisors were disabled, the message will look like this:

|     |
| --- |
| **All necessary components are installed.**<br>**Expert Advisors disabled.** |

4\. You can press the EA button several times and be sure that this event sequence
is processed by the application immediately and displayed in message lines:

|     |
| --- |
| **Expert Advisors disabled.**<br>**Expert Advisors enabled.**<br>**Expert Advisors disabled.**<br>**Expert Advisors enabled.**<br>**Expert Advisors disabled.** |

Please note that, due to the script with the looped basic code used in the program
complex, the program response to the user's controls is not made in multiples of
ticks, but immediately.

As an example, we placed in function Work\_scr() the tickwise dysplaying of a variable
from the EA settings using function Alert().

![](https://c.mql5.com/2/21/2016-01-19_11h31_21.png)

Let us consider this feature. Function Work\_scr() is a part of the script. The basic
loop of the script has time to turn hundreds of times between ticks while the message
is given by function Alert() in multiples of ticks.

5\. Open the EA setup toolbar and replace value AAA with 3. The script will track
this event and give a message in the indicator window:

|     |
| --- |
| **Expert Advisors enabled.**<br>**Expert Advisors disabled.**<br>**Expert Advisors enabled.**<br>**Expert Advisors disabled.**<br>**EA settings have been updated.** |

The window of function Alert() will display the new value of variable AAA tickwise:

![](https://c.mql5.com/2/21/2016-01-19_11h32_47.png)

6\. Now, you can load or unload any components in any sequence, play with EA button,
change value of ajustable variable, and make your own opinion about the quality
of the program complex.

### Conclusion

The main thing we have reached using the described technology is that **the script does not stop working regardless of whether events take place or not**
**in its environment**. The script will stop working if it finds that one or both other components (indicator,
Expert Advisor) are not available.

The described principle of creating a program complex is, slightly modified, usedin
a really working application **AutoGraf** that was described in the article named [Graphic Expert Advisor: AutoGraf](https://www.mql5.com/en/articles/1378) .

SK. Dnepropetrovsk. 2006

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/1393](https://www.mql5.com/ru/articles/1393)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/1393.zip "Download all attachments in the single ZIP archive")

[Exp\_Scr\_Ind.zip](https://www.mql5.com/en/articles/download/1393/exp_scr_ind.zip "Download Exp_Scr_Ind.zip")(33.24 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [My First "Grail"](https://www.mql5.com/en/articles/1413)
- [Considering Orders in a Large Program](https://www.mql5.com/en/articles/1390)
- [Graphic Expert Advisor: AutoGraf](https://www.mql5.com/en/articles/1378)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/39290)**
(7)


![](https://c.mql5.com/avatar/avatar_na2.png)

**\[Deleted\]**
\|
12 Jul 2007 at 11:24

Thank you Sergey for interesting article!

Much to study - think - learn about within...;)

As an English speaker \[only :(\], I am continually amazed at how much .ru MT4'ers
put into their articles - am thankful of such persons... as soooo much \[interesting\
things\] to learn and keeping the little grey cells firing!!! - especially at my
grey haired time of life, :)

![Maria Sountsova](https://c.mql5.com/avatar/avatar_na2.png)

**[Maria Sountsova](https://www.mql5.com/en/users/maria)**
\|
12 Jul 2007 at 13:22

**ukt:**

Thank you Sergey for interesting article!

Much to study - think - learn about within...;)

As an English speaker \[only :(\], I am continually amazed at how much .ru MT4'ers
put into their articles - am thankful of such persons... as soooo much \[interesting\
things\] to learn and keeping the little grey cells firing!!! - especially at my
grey haired time of life, :)

Hello, Tim! I think Sergey will reply to your comment, too.

As a translator of this and some others of his great articles into English, I would
like to ask you for informing me about possible \[raw\] errors made in translations.
If you once have time and mood to correct something, please write me at: maria
AT metaquotes DOT ru

We try to do our best to translate articles in the best practically way. But being
non-native speakers shows, doesn't it? ((


![Сергей Ковалев](https://c.mql5.com/avatar/2009/11/4B0DCA5C-32FF.jpg)

**[Сергей Ковалев](https://www.mql5.com/en/users/sk)**
\|
12 Jul 2007 at 21:09

Привет, Tim.

Спасибо за Ваш интерес к статье.

С тех пор, как вышла статья, в MQL4 появились новые функции. Это
позволило существенно усовершенствовать идею. 2я версия AutoGraf
будет значительно удобнее. А пока бесплатно: [http://autograf.dp.ua/](https://www.mql5.com/go?link=https://autograf4.com/ "http://autograf.dp.ua/")

Извините за формальный перевод.

Greetings, Tim.

Thanks for your interest to article.

Since there was article, in MQL4 there were new functions. It has allowed to improve idea essentially. 2я version
AutoGraf will be much more convenient. For now it is free-of-charge: [http://autograf.dp.ua/](https://www.mql5.com/go?link=https://autograf4.com/ "http://autograf.dp.ua/")

Excuse for formal translation.

![](https://c.mql5.com/avatar/avatar_na2.png)

**\[Deleted\]**
\|
10 Sep 2007 at 22:10

Where is the english version? I couldn't find it.

thanks,

Ed


![Wiley](https://c.mql5.com/avatar/avatar_na2.png)

**[Wiley](https://www.mql5.com/en/users/wiley)**
\|
11 Dec 2009 at 20:03

I know this article has been up for awhile, and I did rework all the code to my own purposes some time ago, and saw how it all worked.  The only real benefit is buttons that work in real time... but is all the code overhead worth that?

It causes me to wonder if matters couldn't be further simplified.  Why not just run the script and indicator, and dispense with the EA?  The script has the infinite loop working, can process orders, the indicator can provide the buy and [sell signals](https://www.mql5.com/en/docs/constants/objectconstants/enum_object "MQL5 documentation: Object Types").  Seems like this could be simpler.  I know that external variables might not be able to be set in the script in this situation but is that really necessary?  Just interested in any thoughts you might have along these lines... a simpler system?

![Events in МetaТrader 4](https://c.mql5.com/2/13/119_4.gif)[Events in МetaТrader 4](https://www.mql5.com/en/articles/1399)

The article deals with programmed tracking of events in the МetaТrader 4 Client Terminal, such as opening/closing/modifying orders, and is targeted at a user who has basic skills in working with the terminal and in programming in MQL 4.

![Poll: Traders’ Estimate of the Mobile Terminal](https://c.mql5.com/2/14/345_1.png)[Poll: Traders’ Estimate of the Mobile Terminal](https://www.mql5.com/en/articles/1471)

Unfortunately, there are no clear projections available at this moment about the future of the mobile trading. However, there are a lot of speculations surrounding this matter. In our attempt to resolve this ambiguity we decided to conduct a survey among traders to find out their opinion about our mobile terminals. Through the efforts of this survey, we have managed to established a clear picture of what our clients currently think about the product as well as their requests and wishes in future developments of our mobile terminals.

![Expert System 'Commentator'. Practical Use of Embedded Indicators in an MQL4 Program](https://c.mql5.com/2/13/129_2.gif)[Expert System 'Commentator'. Practical Use of Embedded Indicators in an MQL4 Program](https://www.mql5.com/en/articles/1406)

The article describes the use of technical indicators in programming on MQL4.

![MQL4  as a Trader's Tool, or The Advanced Technical Analysis](https://c.mql5.com/2/13/137_1.png)[MQL4 as a Trader's Tool, or The Advanced Technical Analysis](https://www.mql5.com/en/articles/1410)

Trading is, first of all, a calculus of probabilities. The proverb about idleness being an engine for progress reveals us the reason why all those indicators and trading systems have been developed. It comes that the major of newcomers in trading study "ready-made" trading theories. But, as luck would have it, there are some more undiscovered market secrets, and tools used in analyzing of price movements exist, basically, as those unrealized technical indicators or math and stat packages. Thanks awfully to Bill Williams for his contribution to the market movements theory. Though, perhaps, it's too early to rest on oars.

[What's wrong with regular VPS?Here are the 8 most common problems that algorithmic traders may encounterRead![](https://www.mql5.com/ff/sh/hzatb686qjqxwtr4z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/450486&a=drhremihlwuaqyvgpzfddbtmgciejpba&s=c37d25bcceb93ed153b814e6ba4d4839461a9b2d68dd82b95b142be06d310f3f&uid=&ref=https://www.mql5.com/en/articles/1393&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5068260820665694041)

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