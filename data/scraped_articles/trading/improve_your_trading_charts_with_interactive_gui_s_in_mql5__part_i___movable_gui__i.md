---
title: Improve Your Trading Charts With Interactive GUI's in MQL5 (Part I): Movable GUI (I)
url: https://www.mql5.com/en/articles/12751
categories: Trading, Integration, Indicators, Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T18:07:13.764462
---

[We've created a channel for MQL5 developersFollow MQL5.community on social media and be the first to receive important updatesLearn more![](https://www.mql5.com/ff/sh/a83xrgctr82w45z9z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=pwgdbvtemvkwsfltqonysypfvtrtufji&s=e99a66a1660cd810b1edbac65597df695e2c2220d1e937834f402f9aeabd4289&uid=&ref=https://www.mql5.com/en/articles/12751&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5069125225783689375)

MetaTrader 5 / Trading


### Introduction

Welcome to the exciting world of movable GUI in [MQL5](https://www.mql5.com/ "https://www.mql5.com/")! This guide is here to empower you with the knowledge to create dynamic, interactive GUI that will elevate your trading strategies. We'll start by decoding the essential concept of chart events, the engine driving our GUI interactivity. With this foundation, we'll then guide you through crafting your first movable GUI.

Then in the next part, we will advance to creating multiple GUI on a single chart efficiently (not just by copy pasting stuff), we'll delve into enhancing our GUI by adding and customizing various elements, tailoring it to your unique needs. We've also catered to those eager to dive right in with a streamlined guide offering quick steps to make your GUI  movable.

By the end of this journey, you'll have gained a valuable skill in creating and manipulating movable GUI in MQL5, a powerful tool for any trader. Even if you're in a hurry, we've got you covered with a quick-guide section for those who want to jump right into making their GUI movable. So, let's get started on this exciting journey!

We will be proceeding further in the following manner:

- [Decoding Chart Events: The Building Blocks of Movable GUI](https://www.mql5.com/en/articles/12751#decoding_chart_events)
- [Crafting Your First Movable GUI: A Step-by-Step Guide](https://www.mql5.com/en/articles/12751#crafting_first_gui)
- [Conclusion](https://www.mql5.com/en/articles/12751#conclusion)

### Decoding Chart Events: The Building Blocks of Movable GUI

As of now, the EA Code looks like this i.e. absolute basic EA:

You might be wondering why EA, why not Indicator? Well, The reason for that is EA is easier to understand for most peoples while Indicator may confuse some but rest assured, Same steps can be followed in Indicator too.

```
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

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void OnChartEvent(const int id, const long& lparam, const double& dparam, const string& sparam)
   {
//---
   }
//+------------------------------------------------------------------+
```

Now, To understand OnChartEvent better, Let's compare it with other predefined function that you already know about from the above absolute basic EA with OnChartEvent().

| Predefined functions | Their Roles |
| --- | --- |
| OnInit() | Run On Initialization i.e. The EA is Initialized (Attached) on the chart, This runs only once |
| OnTick() | Run On Tick i.e. When Symbol on the chart receives a tick from the broker, tick means price update |
| OnDeinit() | Run On Deinitialization i.e.  The EA is Deinitialized (Removed) from the chart, This runs only once |

In the same manner OnChartEvent() is a function that is executed when certain events happen. Well, What events are we talking about here?

There are 9 predefined events (excluding 2 custom onces):

1. CHARTEVENT\_KEYDOWN

2. CHARTEVENT\_MOUSE\_MOVE

3. CHARTEVENT\_OBJECT\_CREATE

4. CHARTEVENT\_OBJECT\_CHANGE

5. CHARTEVENT\_OBJECT\_DELETE

6. CHARTEVENT\_CLICK

7. CHARTEVENT\_OBJECT\_CLICK

8. CHARTEVENT\_OBJECT\_DRAG

9. CHARTEVENT\_OBJECT\_ENDEDIT

Brief Overview for the once that we will be using later on in the article:

1. CHARTEVENT\_KEYDOWN




When chart window is in focus (just click anywhere on the chart window to get it in focus) OnChartEvent() function is executed each time any key on the keyboard is clicked, to be specific key is pressed down

Holding it will mean clicking the key again and again at a rate of 30 Clicks per second.

On that key is pressed what can we do about it, NOT MUCH. It is useless untill we know what key is pressed, Then How do we know what key was pressed? Well that's where Parameters of  OnChartEvent() comes into the play.

What parameters are we talking about? There are 4 parameter we get when OnChartEvent() is executed






1. id -> integer
2. lparam -> long
3. dparam -> double
4. sparam -> string

These are simply put, Some data about the events for which OnChartEvent() is called for and We can use this data inside OnChartEvent() function.

For example In case on event CHARTEVENT\_KEYDOWN,

   - id contains  CHARTEVENT\_KEYDOWN itself so that we can identify for what event the  OnChartEvent() is called for and treat the othere params accordingly.
   - lparam contains keycode of the key pressed.
   - dparam contains number of keypresses generated while the key was held in the pressed state but when we hold the key it does not set the key in pressed state and instead does 30 clicks per second. So, This value is always 1.
   - sparam contains bit mask, simply put It desctibes the state of the key, pressed or clicked by showing 2 different values for a specific key (see the example below to understand it better).

Say we clicked/hold key "A" on the keyboard then OnChartEvent() will be executed with

   - id =  CHARTEVENT\_KEYDOWN
   - lparam = 65
   - dparam = 1
   - sparam = 30 for first click and 16414 for the continuous subsequent clicks while holding A at the rate of 30 Clicks per second.

Now that we have the information, we can use some if statements and do something when user press or holds A key.

2. CHARTEVENT\_MOUSE\_MOVE



    First of all it requires a booleon chart property names  CHART\_EVENT\_MOUSE\_MOVE to be set to True, It can simply be done by using a simple line of code:




```
//Set Chart property CHART_EVENT_MOUSE_DOWN to true
ChartSetInteger(0, CHART_EVENT_MOUSE_MOVE, true);
```




It is usually recommended to do this on OnInit(), Once this is done we can use CHARTEVENT\_MOUSE\_MOVE

Whenever Mouse is moved above the chart, OnChartEvent() will be executed with below information in the params:




   - id =  CHARTEVENT\_MOUSE\_MOVE
   - lparam = x-axis co-ordinate
   - dparam = y-axis co-ordinate
   - sparam =  bit mask, which describes the state of the mouse keys

bitmask holds below values ->

   - Mouse Left               --> 1
   - Mouse Right             --> 2
   - Mouse Middle           --> 16
   - Mouse First X Key     --> 32
   - Mouse Second X Key --> 64
   - Shift Key                  --> 4
   - Control Key              --> 8

Chart window is on 4th quadrant i.e. x-axis co-ordinate (lparam) is from Left of the chart window and y-axis co-ordinate (dparam) from Top the chart window. Now With all that information, we are ready to use CHARTEVENT\_MOUSE\_MOVE, We will be using it below to make GUI movable.

3. CHARTEVENT\_CLICK



It does not require any special property, We can use it directly i.e. Whenever mouse is clicked on the chart, OnChartEvent() is executed with below params


   - id =  CHARTEVENT\_CLICK
   - lparam = x-axis co-ordinate
   - dparam = y-axis co-ordinate
   - sparam =  "" i.e. empty string that means it doesn't provide any useful information

Now that we have the above information, we can use some if statements and do something when user clicks anywhere on the chart.

Above We have discussed 3 events that executes OnChartEvent() fundtion, CHARTEVENT\_KEYDOWN, CHART\_EVENT\_MOUSE\_MOVE, CHARTEVENT\_CLICK

I know if you have not used OnChartEvent() function before, All of this may seem bit confusing and overwhelming, Well have been there but We all have gotten out of that stage too by learning more and practicing more. We Will be Practicing the above knowledge below to make GUI movvable, Stick with me and you will feel confident and better very soon.

Read Documentation or Comment down below if you need details for the other events that execute OnChartEvent().

### Crafting Your First Movable GUI: A Step-by-Step Guide

Now that all the boring stuff is done, We can focus on Real Learning i.e. Applying the theory that we learned to do real stuff not just talking

So In this section Our goal is to create a very very simple looking GUI or we can say a Blank White Rectangle Label. Now now don't get demotivated just because we are not creating complex good looking GUI, things always start from the basics but they increase their level exponentially.

Do you think you can keep up with it? Try it to find out, Stick with me till the end of this article and see if you can say that all of this was easy. Let's call this GUI a dashboard as soon we will make it a dashboard. So, Let's get started with it without any further ado

First, Let's create a Basic Rectangle Label of 200x200 (XSize x YSize) at 100 pixels from Left (XDistance) and 100 pixels from Top (YDistance)

```
int OnInit()
   {
    //---
    //Set the name of the rectangle as "TestRectangle"
    string name = "TestRectangle";
    //Create a Rectangle Label Object at (time1, price1)=(0,0)
    ObjectCreate(0, name, OBJ_RECTANGLE_LABEL, 0, 0, 0);
    //Set XDistance to 100px i.e. Distance of Rectangle Label 100px from Left of the Chart Window
    ObjectSetInteger(0, name,OBJPROP_XDISTANCE, 100);
    //Set YDistance to 100px i.e. Distance of Rectangle Label 100px from Top of the Chart Window
    ObjectSetInteger(0, name,OBJPROP_YDISTANCE, 100);
    //Set XSize to 200px i.e. Width of Rectangle Label
    ObjectSetInteger(0, name,OBJPROP_XSIZE, 200);
    //Set YSize to 200px i.e. Height of Rectangle Label
    ObjectSetInteger(0, name,OBJPROP_YSIZE, 200);
    //---
    return(INIT_SUCCEEDED);
   }
```

Now when we attach the EA to chart we should see our created Rectangle Label like:

![Figure 1.Simple Rectangle Label Image](https://c.mql5.com/2/55/Fig_1.png)

**Figure 1.Simple Rectangle Label**

Now If we can make this movable with mouse drag then we can do a lot of things like move a very complex multiple dashboard freely on the chart window which can make very good interactive EA's/Indicator's, One of the best application for this can be seen [Trade Assistant EA](https://www.mql5.com/en/market/product/23415).

So How do we proceed for making it movable? Let's create a plan first:

- Conditions before moving it:


  - Mouse must be on the dashboard

  - Mouse Left button must be down
- Then If we move our mouse while holding Mouse Left button down, the dashboard should be moving

- But How much should it move?, It should move exactly as much as mouse has moved from the point mouse has satisfied both it's conditon.

So, That is a basic overview of our plan to make it movable, Now Let's code it step by step:

For the first condition, Mouse must be on the dashboard, We first need to find x and y co-ordinates of mouse position

Time to apply the theory, For x and y axes co-ordinate of mouse, We need to use OnChartEvent(), what do we do next? Try to remember.

1. We set Chart Property, CHART\_EVENT\_MOUSE\_MOVE to True




We put the below code in OnInit() to make it true when EA is initialized:






```
//Set Chart property CHART_EVENT_MOUSE_DOWN to true
ChartSetInteger(0, CHART_EVENT_MOUSE_MOVE, true);
```

2. Now we can get mouse co-ordinate in OnChartEvent()



```
void OnChartEvent(const int id, const long& lparam, const double& dparam, const string& sparam)
      {
       //Verify the event that triggered the OnChartEvent was CHARTEVENT_MOUSE_MOVE because we only want to execute out code when that is the case
       if(id == CHARTEVENT_MOUSE_MOVE)
          {
           //Comment the X and Y Axes Coordinates
           Comment("X: ", lparam, "\nY: ", dparam);
          }
      }
```




In OnChartEvent(), First we verify that the event that triggered the OnChartEvent was CHARTEVENT\_MOUSE\_MOVE using a simple if statement that check if the id is equal to CHARTEVENT\_MOUSE\_MOVE because we only want to execute our comment code when that is the case

Then we comment the X and Y Axes Coordinate (It will show on left top corner of the chart window in white color with a bit small font) like:

![Figure 2. X,Y Coordinate Comment Image](https://c.mql5.com/2/55/Fig_2_b30.gif)



**Figure 2. X,Y Coordinate Comment**


Now For our logic to find out if the mouse is on dashboard, see the below image:

![Figure 3. Formula Visualization Image](https://c.mql5.com/2/55/Fig_3.png)

**Figure 3. Formula Visualization**

To find out if the mouse is on dashboard,

- X >= XDistance             --> X>=100
- X <= XDIstance + XSize --> X<=300
- Y >= YDistance             --> Y>=100
- Y <= YDistance + YSize --> Y>=300

Converting into code:

```
void OnChartEvent(const int id, const long& lparam, const double& dparam, const string& sparam)
  {
   //Verify the event that triggered the OnChartEvent was CHARTEVENT_MOUSE_MOVE because we only want to execute out code when that is the case
   if(id == CHARTEVENT_MOUSE_MOVE)
     {
      //define X, Y, XDistance, YDistance, XSize, YSize
      int X = (int)lparam;
      int Y = (int)dparam;

      string name = "TestRectangle";
      int XDistance = ObjectGetInteger(0, name, OBJPROP_XDISTANCE); //Should be 100 initially as we set it in OnInit()
      int YDistance = ObjectGetInteger(0, name, OBJPROP_YDISTANCE); //Should be 100 initially as we set it in OnInit()
      int XSize = ObjectGetInteger(0, name, OBJPROP_XSIZE); //Should be 200 initially as we set it in OnInit()
      int YSize = ObjectGetInteger(0, name, OBJPROP_YSIZE); //Should be 200 initially as we set it in OnInit()

      //Check Mouse on Dashboard condition
      if(X >= XDistance && X <= XDistance + XSize && Y >= YDistance && Y <= YDistance + YSize)
        {
         //Comment the X and Y Axes Coordinates and Mouse is on the dashboard
         Comment("X: ", lparam, "\nY: ", dparam, "\nMouse is on the Dashboard");
        }
      else
        {
         //Comment the X and Y Axes Coordinates and Mouse is not on the dashboard
         Comment("X: ", lparam, "\nY: ", dparam, "\nMouse is NOT on the Dashboard");
        }

     }
  }
```

Defined the variables X, Y, name, XDistance, YDistance, XSize, YSize. Got the X from lparam, Y from dparam, name is just the string name we set above, XDistance, YDistance, XSize, YSize using ObjectGetInteger() function.

Remember our goal here is to just make the dashboard move smoothly then we worry about other things.

Result:

![Figure 4. Mouse is on the Dashboard GIF](https://c.mql5.com/2/55/Fig_4.gif)

**Figure 4. Mouse is on the Dashboard**

As you can see whenever mouse is on the dashboard, comment changes. So our logic is working and now we know if the mouse is on the dashboard or not.

Now we will be needing the mouse buttons state, Remember the theory if Mouse Left was clicked then sparam was 1, Let's use that

```
void OnChartEvent(const int id, const long& lparam, const double& dparam, const string& sparam)
  {
   //Verify the event that triggered the OnChartEvent was CHARTEVENT_MOUSE_MOVE because we only want to execute out code when that is the case
   if(id == CHARTEVENT_MOUSE_MOVE)
     {
      //define X, Y, XDistance, YDistance, XSize, YSize
      int X = (int)lparam;
      int Y = (int)dparam;
      int MouseState = (int)sparam;

      string name = "TestRectangle";
      int XDistance = ObjectGetInteger(0, name, OBJPROP_XDISTANCE); //Should be 100 initially as we set it in OnInit()
      int YDistance = ObjectGetInteger(0, name, OBJPROP_YDISTANCE); //Should be 100 initially as we set it in OnInit()
      int XSize = ObjectGetInteger(0, name, OBJPROP_XSIZE); //Should be 200 initially as we set it in OnInit()
      int YSize = ObjectGetInteger(0, name, OBJPROP_YSIZE); //Should be 200 initially as we set it in OnInit()

      //Check Dashboard move conditions
      if(X >= XDistance && X <= XDistance + XSize && Y >= YDistance && Y <= YDistance + YSize && MouseState == 1)
        {
         //Comment that the dashboard is ready to move
         Comment("Dashboard is ready to move.");
        }
      else
        {
         //Comment that the dashboard is not ready to move
         Comment("Dashboard is NOT ready to move.");
        }

     }
  }
```

I have added

```
int MouseState = (int)sparam; //To get the mouse state: 1 -> Mouse Left Button Down (You can check the other above)
```

in the variables and added the condition that

```
if(MouseState == 1) // This insures that Mouse Left button in pressed
```

in the if statement and changed the comments a little bit

Now Whenever mouse Left button is in pressed state on the dashboard we get a comment "Dashboard is ready to move." otherwise "Dashboard is NOT ready to move."

Let's see it in action:

![Figure 5. Dashboard is ready to move GIF](https://c.mql5.com/2/55/Fig_5.gif)

**Figure 5. Dashboard is ready to move**

See the comment change when Mouse Left button is held down

Now with that being done, We are ready to move our dashboard, So now How do we go about doing it? Let's try to find out.

We know that dashboard will be moving with our mouse and How do we change dashboard position? Ofcourse wIth XDistance and YDistance property. So We change XDistance and YDistance but by how much? because dashboard is moving with mouse, the dashboard should move same as mouse right?

Well Then by how much did our mouse move by? That we will have to figure out but again How? That was a lot of questions just now, let's make a plan out of those questions

Plan:

- Find out by how much our mouse moved after we pressed MLB (Mouse Left button)
- Move the dashboard by exactly the same amount as mouse is moved by
- Do this untill MLB is pressed down, Stop moving the dashboard after that

Let's go one step at a time,

We can always get the current mouse position right so what if we save the mouse position when MLB was clicked the first time?

and we can know if the MLB is pressed from MouseState variable we created that stores sparam

Below is the code that dectects the MLB click first time:

```
int previousMouseState = 0;
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void OnChartEvent(const int id, const long& lparam, const double& dparam, const string& sparam)
  {
   //Verify the event that triggered the OnChartEvent was CHARTEVENT_MOUSE_MOVE because we only want to execute out code when that is the case
   if(id == CHARTEVENT_MOUSE_MOVE)
     {
      int MouseState = (int)sparam;

      bool mblDownFirstTime = false;
      if(previousMouseState == 0 && MouseState == 1) {
         mblDownFirstTime = true;
      }

      previousMouseState = MouseState;
     }
  }
//+------------------------------------------------------------------+
```

I have removed other things to make it easier to see and understand. This method is quite commonly used in create Trailing functions fro EA or detecting new bar etc.

Let's break it down:

1. ```
int previousMouseState = 0;
```


We declare a int variable named previousMouseState on global space and set it to 0, this variable will store the MouseState value from the last time CHARTEVENT\_MOUSE\_MOVE event happened, Well How? Have some patience you will know soon enough.

2. ```
int previousMouseState = 0;
```


We declare a int variable named previousMouseState on global space and set it to 0, this variable will store the MouseState value from the last time CHARTEVENT\_MOUSE\_MOVE event happened, Well How? Have some patience you will know soon enough.

3. ```
int MouseState = (int)sparam;
bool mblDownFirstTime = false;
if(previousMouseState == 0 && MouseState == 1) {
      mblDownFirstTime = true;
}
```




First We declare one MouseState and set it equal to sparam which contains the Mouse State, two we declare a bool variable named mblDownFirstTime and set it's default value to false

Then we check for 2 conditions, one previousMouseState should be 0 (MLB Up, No mouse button pressed) and (&&) MouseState should be 1 (MLB Down)

This condition basically confirms or denies that MLB is down the first time or not, once we know it's the first time, We set mblDownFirstTime equal to true so that we can use this variable later.


Now With that, Our first step is completed, let move to Step 2 and 3.

Now, There are many ways we can proceed further but to get a very smooth and subtle movement, Below are the step we will be taking:

1. Create a bool global variable named movingState that we will set true once user clicks MLB on the the dashboard, also declare MLB Down X,  MLB Down Y,  MLB Down XDistance,  MLB Down YDistance (here, MLB Down  means the first the MLB was down) Those will be needed to modify the dashboards position

2. While movingState is true, We will be updating dashboard position according to change in mouse position from the mouse initial Position (when MLB was first clicked)

3. While movingState is true, We will be updating dashboard position according to change in mouse position from the mouse initial Position (when MLB was first clicked)


New Code:

```
int previousMouseState = 0;
int mlbDownX = 0;
int mlbDownY = 0;
int mlbDownXDistance = 0;
int mlbDownYDistance = 0;
bool movingState = false;
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void OnChartEvent(const int id, const long& lparam, const double& dparam, const string& sparam)
{
//Verify the event that triggered the OnChartEvent was CHARTEVENT_MOUSE_MOVE because we only want to execute out code when that is the case
if(id == CHARTEVENT_MOUSE_MOVE)
     {
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
//+------------------------------------------------------------------+
```

Let's break it down in simpler terms:

1. ```
int mlbDownX = 0;
int mlbDownY = 0;
int mlbDownXDistance = 0;
int mlbDownYDistance = 0;
bool movingState = false;
```


We created a few variables in Global Space, we have already talked about the previousMouseState other than that:




   - mlbDownX                -> To hold X Coordinate when MLB was down the first time
   - mlbDownY                -> To hold Y Coordinate when MLB was down the first time
   - mlbDownXDistance   -> To hold XDistance property of the dashboard when the MLB was pressed first time
   - mlbDownYDistance   -> To hold YDistance property of the dashboard when the MLB was pressed first time
   - movingState             -> To hold true if we are moving the dashboard else false

```
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
```

First we check if it is the first MLB click, if it is then we update mlbDownX, mlbDownY, mlbDownXDistance, mlbDownYDistance to the current X, Y, XDistance, YDistance respectively, we will be using them later.

Then we check if MLB was down on the dashboard and if it was down on the dashboard then we set movingState to true.

2. ```
if(movingState)
     {
      ChartSetInteger(0, CHART_MOUSE_SCROLL, false);
      ObjectSetInteger(0, name, OBJPROP_XDISTANCE, mlbDownXDistance + X - mlbDownX);
      ObjectSetInteger(0, name, OBJPROP_YDISTANCE, mlbDownYDistance + Y - mlbDownY);
      ChartRedraw(0);
     }
```


And if movingState is true then we modify the XDistance and YDistance,




```
X - mlbDownX // Change in Mouse X Position form the initial click
and
Y - mlbDownY // Change in Mouse X Position form the initial click
```


These above are the change in Mouse Position from the initial click and we add them to mlbDownXDistance and mlbDownYDistance to get the new dashboard position, make sense if you think about it.

And We also set CHART\_MOUSE\_SCROLL chart property to false so that chart doesn't move with our dashboard then we ofcourse redraw the chart to get a very smooth subtle movement.

3. ```
if(MouseState == 0)
     {
      movingState = false;
      ChartSetInteger(0, CHART_MOUSE_SCROLL, true);
     }
```


Now once we leave the MLB i.e. MouseState becomes 0

Then we set movingState to false and again allow the chart to be moved by setting CHART\_MOUSE\_SCROLL to true again.


Now, With that being done, Our Code is completed, Give your a pat on the back if you the followed till now because we are almost done for the Part I.

Our Complete Code:

```
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---
   //Set the name of the rectangle as "TestRectangle"
   string name = "TestRectangle";
   //Create a Rectangle Label Object at (time1, price1)=(0,0)
   ObjectCreate(0, name, OBJ_RECTANGLE_LABEL, 0, 0, 0);
   //Set XDistance to 100px i.e. Distance of Rectangle Label 100px from Left of the Chart Window
   ObjectSetInteger(0, name, OBJPROP_XDISTANCE, 100);
   //Set YDistance to 100px i.e. Distance of Rectangle Label 100px from Top of the Chart Window
   ObjectSetInteger(0, name, OBJPROP_YDISTANCE, 100);
   //Set XSize to 200px i.e. Width of Rectangle Label
   ObjectSetInteger(0, name, OBJPROP_XSIZE, 200);
   //Set YSize to 200px i.e. Height of Rectangle Label
   ObjectSetInteger(0, name, OBJPROP_YSIZE, 200);

//Set Chart property CHART_EVENT_MOUSE_DOWN to true
   ChartSetInteger(0, CHART_EVENT_MOUSE_MOVE, true);
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

//Declare some global variable that will be used in the OnChartEvent() function
int previousMouseState = 0;
int mlbDownX = 0;
int mlbDownY = 0;
int mlbDownXDistance = 0;
int mlbDownYDistance = 0;
bool movingState = false;
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void OnChartEvent(const int id, const long& lparam, const double& dparam, const string& sparam)
  {
//Verify the event that triggered the OnChartEvent was CHARTEVENT_MOUSE_MOVE because we only want to execute out code when that is the case
   if(id == CHARTEVENT_MOUSE_MOVE)
     {
      //define X, Y, XDistance, YDistance, XSize, YSize
      int X = (int)lparam;
      int Y = (int)dparam;
      int MouseState = (int)sparam;

      string name = "TestRectangle";
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

Now That simple code does the job

Result:

![Figure 6. Final Result](https://c.mql5.com/2/55/Fig_6.gif)

**Figure 6. Final Result**

Now with that we will wrap this section now

### **Conclusion**

I am really sorry to leave to on a kind of a cliffhanger because we have not created something very unique yet. But be assured another part will be coming in a few days in which:

- We  will advance to creating multiple GUI on a single chart efficiently (not just by copy pasting stuff)
- we'll delve into enhancing our GUI by adding and customizing various elements, tailoring it to your unique needs. We've also catered to those eager to dive right in with a streamlined guide offering quick steps to make your GUI  movable.

By the end of this journey, you'll have gained a valuable skill in creating and manipulating movable GUI in MQL5, a powerful tool for any trader. So, let's get started on this exciting journey!

Hope you liked it and it helped you in any slightest way. Hope to see you again in the next part.

Happy Coding

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/12751.zip "Download all attachments in the single ZIP archive")

[Movable\_Dashboard\_MQL5.mq5](https://www.mql5.com/en/articles/download/12751/movable_dashboard_mql5.mq5 "Download Movable_Dashboard_MQL5.mq5")(5.27 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/449259)**
(15)


![](https://c.mql5.com/avatar/avatar_na2.png)

**\[Deleted\]**
\|
26 Aug 2023 at 04:14

Millions thanks, this helps me a lot!


![Sahil Bagdi](https://c.mql5.com/avatar/2025/6/68402632-0431.jpg)

**[Sahil Bagdi](https://www.mql5.com/en/users/sahilbagdi)**
\|
31 Aug 2023 at 08:48

**Anna Gan [#](https://www.mql5.com/en/forum/449259#comment_48968039):**

Millions thanks, this helps me a lot!

You’re very welcome

I am glad that I was able to help you with my article.

![RadoWay](https://c.mql5.com/avatar/2024/4/66103fb1-60b2.jpg)

**[RadoWay](https://www.mql5.com/en/users/radoway)**
\|
12 May 2024 at 16:10

Thanks for the great article.

May i ask, what if the rectangle coordinate is set to the [upper right corner](https://www.mql5.com/en/docs/constants/objectconstants/enum_basecorner "MQL5 documentation: Chart Corner")? How can the mouse move find it if the mouse coordinate is set to the left upper corner?

Note: i ask this question because in my code the rectangle has to bet set to the right upper corner and it cant be changed to the left.

Thanks a lot in advance…

![Sahil Bagdi](https://c.mql5.com/avatar/2025/6/68402632-0431.jpg)

**[Sahil Bagdi](https://www.mql5.com/en/users/sahilbagdi)**
\|
13 May 2024 at 00:54

**Radwan Aly Abouelseoud Ara [#](https://www.mql5.com/en/forum/449259/page2#comment_53344064):**

Thanks for the great article.

May i ask, what if the rectangle coordinate is set to the [upper right corner](https://www.mql5.com/en/docs/constants/objectconstants/enum_basecorner "MQL5 documentation: Chart Corner")? How can the mouse move find it if the mouse coordinate is set to the left upper corner?

Note: i ask this question because in my code the rectangle has to bet set to the right upper corner and it cant be changed to the left.

Thanks a lot in advance…

For that you will have to understand and edit the code yourself.

I am glad you liked my article so much, it feels really good that someone is being helped from code written by me.

Thanks & Regards.

![Arch](https://c.mql5.com/avatar/avatar_na2.png)

**[Arch](https://www.mql5.com/en/users/dms-66-99-69-gmail)**
\|
18 Jul 2024 at 14:28

The old way.Read the coordinates of the object should be done after [clicking the mouse](https://www.mql5.com/en/docs/constants/chartconstants/enum_chartevents "MQL5 Documentation: Chart Event Types") and only after checking to disable the scroll chart .But as I did not try to optimise it will still lag at the minimum chart size with enabled volumes.(Perhaps it is a matter of PC power).


![Multilayer perceptron and backpropagation algorithm (Part 3): Integration with the Strategy Tester - Overview (I).](https://c.mql5.com/2/51/Perceptron_Multicamadas_60x60.png)[Multilayer perceptron and backpropagation algorithm (Part 3): Integration with the Strategy Tester - Overview (I).](https://www.mql5.com/en/articles/9875)

The multilayer perceptron is an evolution of the simple perceptron which can solve non-linear separable problems. Together with the backpropagation algorithm, this neural network can be effectively trained. In Part 3 of the Multilayer Perceptron and Backpropagation series, we'll see how to integrate this technique into the Strategy Tester. This integration will allow the use of complex data analysis aimed at making better decisions to optimize your trading strategies. In this article, we will discuss the advantages and problems of this technique.

![Automated exchange grid trading using stop pending orders on Moscow Exchange (MOEX)](https://c.mql5.com/2/54/moex-mesh-trading-avatar.png)[Automated exchange grid trading using stop pending orders on Moscow Exchange (MOEX)](https://www.mql5.com/en/articles/10671)

The article considers the grid trading approach based on stop pending orders and implemented in an MQL5 Expert Advisor on the Moscow Exchange (MOEX). When trading in the market, one of the simplest strategies is a grid of orders designed to "catch" the market price.

![How to Become a Successful Signal Provider on MQL5.com](https://c.mql5.com/2/55/How_to_Become_a_Successful_Signal_Provider_Avatar.png)[How to Become a Successful Signal Provider on MQL5.com](https://www.mql5.com/en/articles/12814)

My main goal in this article is to provide you with a simple and accurate account of the steps that will help you become a top signal provider on MQL5.com. Drawing upon my knowledge and experience, I will explain what it takes to become a successful signal provider, including how to find, test, and optimize a good strategy. Additionally, I will provide tips on publishing your signal, writing a compelling description and effectively promoting and managing it.

![Creating an EA that works automatically (Part 13): Automation (V)](https://c.mql5.com/2/51/aprendendo_construindo_013_avatar.png)[Creating an EA that works automatically (Part 13): Automation (V)](https://www.mql5.com/en/articles/11310)

Do you know what a flowchart is? Can you use it? Do you think flowcharts are for beginners? I suggest that we proceed to this new article and learn how to work with flowcharts.

[Need a reliable hosting solution for your robots?Contact your broker and find out about available Sponsored MetaTrader VPS offeringsLearn more![](https://www.mql5.com/ff/sh/0pw0dk81s56qy774z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=vljwvezfjkfbvviocwskggexlvgykvob&s=70cf8e354b9a125332533ffb65d7365abe8dde5b5c1ede9caac479a9e9df4f25&uid=&ref=https://www.mql5.com/en/articles/12751&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5069125225783689375)

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