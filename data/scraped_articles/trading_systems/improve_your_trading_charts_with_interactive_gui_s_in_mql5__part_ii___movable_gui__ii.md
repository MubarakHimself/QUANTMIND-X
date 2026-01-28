---
title: Improve Your Trading Charts With Interactive GUI's in MQL5 (Part II): Movable GUI (II)
url: https://www.mql5.com/en/articles/12880
categories: Trading Systems, Integration, Indicators, Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T19:25:29.662201
---

[![](https://www.mql5.com/ff/sh/a27a2kwmtszm2m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
MQL5 Channels - Messenger for traders\\
\\
Subscribe to traders' channels or create your own.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=vpcudokyepxfrcxrpjcktglhsjlemtza&s=f08ad2c1289e29bd5630f1ef977aef297d5cdbfcb686faed4a4b0f1e276d3c4a&uid=&ref=https://www.mql5.com/en/articles/12880&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5070271891857412853)

MetaTrader 5 / Trading systems


### Introduction

Welcome to the second part of this series. In the first part, we discussed how to create a simple, movable dashboard. This second part aims to achieve the same objective, but in a more efficient way that's suitable for full-fledged EA/Indicator applications.

For example, if we want to have two movable dashboards on the screen, we would need to duplicate the existing code and create six additional global variables, each with a different name. Now, if we decide we need three movable dashboards, the complexity of the code would increase significantly and become much harder to manage. Clearly, we need a more streamlined approach.

Fortunately, we can turn to .mqh files to simplify this process.

Here's what we'll cover in this article:

1. [Understanding the concept of classes](https://www.mql5.com/en/articles/12880#section2)
2. [Creating the same dashboard using .mqh file](https://www.mql5.com/en/articles/12880#section3)
3. [Setting up two dashboards on the same chart using .mqh file](https://www.mql5.com/en/articles/12880#section4)

### Understanding the concept of classes

Before we delve deeper, it's essential to first understand the concept of classes. Classes can become incredibly advanced and complex as we delve further into them, but for the time being, we'll only be covering the basics. Understanding and effectively utilizing these basic concepts are important steps before advancing to more intricate details.

So, what exactly are classes?

Simply put, a class is a complex data type, akin to int, string, and others, but with a touch more complexity.

There are numerous ways to define classes, but fundamentally, they can be viewed as clusters of code. What kind of code, you may ask? They are typically a collection of functions, often referred to as methods, and variables. Some might argue this definition is somewhat vague or slightly inaccurate. However, it's important to remember that we're not cramming for an exam here like we might have done in school. Our primary objective is to harness the power of classes to make coding more manageable and efficient, and for that, a rigid definition isn't crucial.

In essence, classes are a collection of functions and variables that we can utilize to our advantage.

Now, this understanding naturally leads to four fundamental questions:

1. Where do we create them?
2. How do we declare them?
3. How to write them?
4. How to use them?

By the way, if you're scratching your head wondering why we need classes in the first place, the answer is quite straightforward. Classes simplify the coding process and make code management a breeze.

1. **Where do we create them?**




The choice of file type for creating classes—be it .mq5 or .mqh—is flexible. However, typically, we opt for separate .mqh files.

The difference between creating classes in .mq5 and .mqh is noteworthy. If you develop your classes within an .mqh file, you'll need to import it into .mq5. This is because the creation of an EA/Indicator is exclusive to .mq5 files. However, if you establish the class directly within an .mq5 file, there's no need for any import process.

We generally favor separate .mqh files because they enhance code manageability. The importation process is straightforward—it merely requires a single line of code. For the sake of this discussion, we'll be using a separate .mqh file.

2. **How do we declare them?**




The declaration of a class is straightforward. Below is an example of how to declare a simple, empty class:


```
class YourClassName
     {
     };
```

In the above code snippet, 'YourClassName' is a placeholder. Replace 'YourClassName' with the actual name you want to assign to your class.

3. **What to write them?**




To understand this, let's start by discussing variables before we move on to functions.

Suppose you wish to declare two variables: one of 'int' type and the other of 'bool' type. You can do this in the following manner:


```
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class YourClassName
     {
      int var1;
      bool var2;
     };
//+------------------------------------------------------------------+
```


Importantly, note that you can't assign values to these variables directly within the class declaration. For example, the following code will lead to an error:






```
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class YourClassName
     {
      int var1 = "int var";
      bool var2 = true;
     };
//+------------------------------------------------------------------+
```

The error you'll encounter will say:


'=' \- illegal assignment use

'=' \- illegal assignment use





In scenarios where our logic depends on initial variable values, we can utilize something called a constructor. Additionally, we also have something known as a destructor, which essentially serves as the counterpart to a constructor.

Constructors and destructors are special functions always associated with a class, whether you declare them explicitly or not. If you don't declare them, they are implicitly considered as empty functions. The constructor executes when an instance of the class is declared, and the destructor executes when an instance of the class goes out of scope. It's important to note that in MQL5, there's no way to explicitly delete an instance of the class.

For instance, the OnInit() function in our code behaves as a constructor, and the OnDeinit() function acts as a destructor. The class here is hidden in the background to maintain simplicity. This pattern is common in many languages, including Java, which always includes a default class.



**We'll discuss what an 'instance' means shortly in next step (How to use them?).**


For now, understand that we can use constructors to assign initial values to our variables. The destructor won't be relevant for this discussion, but we'll surely cover it in the later parts of this series.






**Please note that it is highly recommended to use a constructor.**







While some programmers might ignore compiler warnings and assume that variables will be implicitly set to default if they don't explicitly define them, this approach isn't entirely correct nor completely wrong.



The assumption stems from the fact that in many programming languages (even in MQL4), variables without explicit definitions default to some value without causing inconsistencies in the code. However, in MQL5, you might encounter inconsistencies in your code if you don't explicitly define your variables.









Here are the presumed default values for the most commonly used data types:



| Type | Code to Declare | **Presumed** Default Value |
| --- | --- | --- |
| int | int test; | 0 |
| double | double test; | 0.0 |
| bool | bool test; | false |
| string | string test; | NULL |
| datetime | datetime test; | 1970.01.01 00:00:00 |




**Note:** The Presumed Default Value is the value you'll see when you Print the uninitialized variable. If you check the same with an if-statement, some will behave as expected, but others won't, potentially leading to issues.





Our testing script looks like this:


```
void OnStart()
     {
      type test;
      if(test == presumedDefaultValue)
        {
         Alert("Yes, This is the default value of the test variable.");
        }
      else
        {
         Alert("No, This is NOT the default value of the test variable.");
        }
     }
```


**_Replace 'type' with the variable type and 'presumedDefaultValue' with the value you anticipate to be the default._**

Here, you'll see that for bool and string, everything works perfectly and the alert "Yes, This is the default value of the test variable." will be triggered. However, for int, double, and datetime, it's not as straightforward. You'll get an alert that says, "No, This is NOT the default value of the test variable." This unexpected result could cause logical issues.





Now that we understand the importance of the constructor, let's see how to create one:


```
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class name
     {
private:

public:
      int               var1;
      bool              var2;
                        name();
                       ~name();
     };
//+------------------------------------------------------------------+

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
name::name()
     {
      var1 = 0;
      var2 = true;
     }
//+------------------------------------------------------------------+
```




Here, we've included public and private fields in the class, although private is currently empty. **_It is very important to declare constructor with public access modifier to use it in other files._** To explain this, let's discuss Access Modifiers (also known as Access Specifiers).



Access Modifiers define how the compiler can access variables, members of structures, or classes.



| Access Modifiers | Description |
| --- | --- |
| public | Allows unrestricted access to the variable or class method |
| private | Allows access from methods of this class, as well as from methods of [publicly inherited](https://docs.mql4.com/basis/oop/inheritance#public_inheritance "https://docs.mql4.com/basis/oop/inheritance#public_inheritance") classes. Other access is impossible; |
| protected | Allows access to variables and class methods only from methods of the same class. |
| virtual | Applies only to class methods (but not to methods of structures) and tells the compiler that this method should be placed in the table of virtual functions of the class. |


We will only understand public and private in this article and others are not used, we will surely cover them in later parts of the series.



We'll focus on public and private modifiers in this article, leaving out others for later parts of the series. Public essentially means that the variables and functions defined as public can be used/modified (variables) anywhere, including different .mq5 or .mqh files. On the other hand, private will only allow access within the function defined in the current class.

If you are wandering why need them in the first place?, There are many reason like Data Hiding, Abstraction, Maintainability, Reusability.



We've defined our constructor code like this:


```
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
name::name()
     {
      var1 = 0;
      var2 = true;
     }
//+------------------------------------------------------------------+
```


Constructor/Destructor does not have a return type.



Note that Constructors/Destructors do not have a return type. We're simply assigning value to uninitialized variables here because you may sometimes need the bool variable (i.e., var2 in this case) to be set to true as the initial value.



There's an alternative way of doing this:


```
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
name::name() : var1(0), var2(true)
     {
     }
//+------------------------------------------------------------------+
```


This constructor has no body and simply initializes the values of uninitialized variables to what we want. While you can use either method, it's generally better to use the second one. For instance, if you use const when declaring a variable to make it unchangeable, the first method of assigning value to an uninitialized variable won't work, but the second will. This is because in the first method we're assigning value, while in the second we're initializing value.



or you can alternatively write this in the member list too:




```
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class name
     {
private:

public:
      int               var1;
      bool              var2;
                        name() : var1(0), var2(true) {}
                       ~name();
     };
//+------------------------------------------------------------------+
```






Now that we've covered how to declare variables, constructors, and destructors, understanding how to create a function will be straightforward.

If you want to create a function named functionName() that takes one input as a string variable and simply prints the variable, it would look something like this:




```
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class className
     {
public:
      void              functionName(string printThis)
     };
//+------------------------------------------------------------------+

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void className::functionName(string printThis)
     {
      Print(printThis);
     }
//+------------------------------------------------------------------+
```




We declare the function in the member list in the class named "className" with public access modifier so that we can use this function anywhere. Then, we write the body of the function.



Note that the declaration of the function in the class i.e.





```
functionName(string printThis)
```



must match exactly when writing the function body.



This concludes our basic introduction to writing classes in MQL5.

4. **How to use them?**




To understand this better let's see our folder structure:






   - Test Project
     - mainFile.mq5
     - includeFile.mqh

Firstly, Let's see our complete class code that we written in includeFile.mqh:

In this example, we've declared a class named "className" that includes a constructor, a destructor, three variables (with one private and two public), and a public function.

   - Constructor: We initialize variables var0, var1, and var2 to 10, 0, and true, respectively.
   - Destructor: Currently, it's empty and therefore does nothing.
   - var0: This is a private integer variable, initialized to 10, and used in the function (functionName).
   - var1: This is a public integer variable, initialized to 0, and also used in the function (functionName).
   - functionName: This void function named "functionName" accepts an integer "printThisNumber" and prints the sum of printThisNumber, var0, and var1.

Next, let's examine our mainFile.mq5:

```
#include "includeFile.mqh"
className classInstance;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   classInstance.functionName(5)//This line will print (5+10+0) = 15
   classInstance.var1 = 50;//This will change the value of var1 to 50
   classInstance.functionName(5)//Now, this line will print (5+10+50) = 65
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
```

We first include the includeFile.mqh file in our mainFile.mq5. Then, we create an instance of the class using

```
className classInstance;
```

This instance enables us to modify and use variables or functions from the class. This process is called instantiation, but we won't delve into the deep definitions here.

You can create as many instances as you like and they all will be independent of each other.

We used "" instead of <> to search for the .mqh file because <> looks for the .mqh file in the 'include' folder, while "" searches for the .mqh file in the current directory.
This walkthrough provides a solid understanding of how to use the class.

### Creating the same dashboard using .mqh file

Let's embark on a journey to create a similar dashboard from scratch, but this time using a .mqh file. We'll borrow pieces of our previous code where necessary. To organize our code files effectively, we'll create a new folder, aptly named "Movable Dashboard MQL5".

Next, we'll generate two new files: the "Movable\_Dashboard\_MQL5.mq5" file, which will serve as our primary .mq5 file, and the "GUI\_Movable.mqh" file to hold the code for making the dashboard movable. Properly naming these files is crucial for managing multiple files with ease.

To begin, let's craft a 200x200 white dashboard using the Object Create method in our main .mq5 file (Movable\_Dashboard\_MQL5.mq5) within the OnInit():

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
   ChartRedraw();
//---
   return(INIT_SUCCEEDED);
  }
```

Result:

![Figure 1: Basic Dashboard Image](https://c.mql5.com/2/56/Screenshot_2023-07-01_071105.png)

**Figure 1. Basic Dashboard Image**

You might question why we're creating the dashboard in our main .mq5 file (Movable\_Dashboard\_MQL5.mq5) instead of the .mqh file (GUI\_Movable.mqh). This decision is mainly for simplicity's sake and can depend on your specific goals, we will be taking that approach in the next section.

Let's turn our attention to the .mqh file (GUI\_Movable.mqh), which currently looks like this:

```
//+------------------------------------------------------------------+
//| Class GUI_Movable                                                |
//+------------------------------------------------------------------+
class GUI_Movable
  {

  };
//+------------------------------------------------------------------+
```

Here, we've merely declared a class without explicitly defining a constructor and a destructor.

So, what's our objective? We aim to tailor this code so it can be implemented in our main file, thereby making our dashboard movable.

Alright, how do we achieve that? The only code that needs to make the dashboard movable in our previous .mq5 file (Movable\_Dashboard\_MQL5.mq5) is:

```
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

Now we will proceed in the following manner:

1. Write code for GUI\_Movable class
2. Create a instance of the class in main .mq5 file
3. Give name to that instance
4. Use GUI\_Movable class methods to make dashboard movable

These steps might appear daunting at first, but the process will become intuitive with practice.

1. **Write code for GUI\_Movable class:**
We need to plan out the components of our class. Here's a breakdown:


1. We need to declare six variables with private modifiers (previousMouseState, mlbDownX, mlbDownY, mlbDownXDistance, mlbDownYDistance, movingState). These variables include five integers and one boolean.
2. We must declare a seventh public variable that will store the name of the dashboard. We need to make this variable public because we'll need to modify it from our main .mq5 file.
3. We need to find a way to utilize the OnChartEvent function in the .mqh file, as all of our declared variables are located there, and we need these variables within the OnChartEvent function.


1. Let's start by declaring our six private variables in the class, five of which are integers and one boolean. We'll use a constructor to initialize these values:








      ```
      //+------------------------------------------------------------------+
      //| Class GUI_Movable                                                |
      //+------------------------------------------------------------------+
      class GUI_Movable
        {
      private:
         int               previousMouseState, mlbDownX, mlbDownY, mlbDownXDistance, mlbDownYDistance;
         bool              movingState;
      public:
                           GUI_Movable() : previousMouseState(0), mlbDownX(0), mlbDownY(0), mlbDownXDistance(0), mlbDownYDistance(0), movingState(false) {};
        };
      //+------------------------------------------------------------------+
      ```


      Now we need 0 for all int and false for bool as our intial values, so we have used constructor to intialize them.

2. Next, we'll declare a public variable to store the name of the dashboard. This variable needs to be accessible from our main .mq5 file.


      ```
      public:
         string Name;
      ```


      Intial value for it will ofcourse be NULL, but for formality we will initialize it to NULL and change our constructor to (formality becausestring doesn't cause inconsistencies)


      ```
      GUI_Movable() : previousMouseState(0), mlbDownX(0), mlbDownY(0), mlbDownXDistance(0), mlbDownYDistance(0), movingState(false), Name(NULL) {};
      ```

3. This step might seem a bit complicated, but it's straightforward once you understand it.



      We'll create a public function named OnEvent which will accept the following inputs: id, lparam, dparam, and sparam. Since OnChartEvent() doesn't return anything (void), we'll also make OnEvent() void.



      The OnEvent function will do everything that OnChartEvent() is designed to do, but it will do so in the .mqh file. We'll use OnEvent() in the actual OnChartEvent() function in the main file.



      To avoid errors caused by declaring OnChartEvent() in both the .mqh and main files, we created this separate function named OnEvent(). Here's how we declare it:


      ```
      public:
         string            Name;
         void              OnEvent(int id, long lparam, double dparam, string sparam);
      ```




      Now, let's write the function code. It'll do everything that the original OnChartEvent() was designed to do:


      ```
      //+------------------------------------------------------------------+
      //|                                                                  |
      //+------------------------------------------------------------------+
      void GUI_Movable::OnEvent(int id, long lparam, double dparam, string sparam)
        {

        }
      //+------------------------------------------------------------------+
      ```




      We placed this function in the global scope. Now, we can just put the same code in here and it will have access to the variables declared in the class.



      The complete function will look like this:


      ```
      //+------------------------------------------------------------------+
      //|                                                                  |
      //+------------------------------------------------------------------+
      void GUI_Movable::OnEvent(int id, long lparam, double dparam, string sparam)
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


      The only thing we change is


      ```
      string name = "TestRectangle";
      ```


      To


      ```
      string name = Name;
      ```


      Because ofcourse we need to use that Name variable that we set in out main .mq5 file.
2. **Create a instance of the class in main .mq5 file:**

That can very simply done in the following way:


```
#include "GUI_Movable.mqh"
GUI_Movable Dashboard;
```


Here, we have included the .mqh file, opting for "" instead of <> to specify the location of the file. <> searches for the .mqh file in the include folder, while "" looks for the .mqh file in the current directory, which in this case is a folder named "Movable Dashboard MQL5". We then declare an instance of the GUI\_Movable class and assign it a convenient name, "Dashboard". This name allows us to utilize the code we have written in the .mqh file.

3. **Give name to that instance:**



This can be executed effortlessly in the OnInit() function. This is how our OnInit() function should appear:




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
      //Set CHART_EVENT_MOUSE_MOVE to true to detect mouse move event
      ChartSetInteger(0, CHART_EVENT_MOUSE_MOVE, true);
      //Give dashboard's name to the class instance
      Dashboard.Name = name;

//---
      return(INIT_SUCCEEDED);
     }
```



In the end, we use

```
//Give dashboard's name to the class instance
Dashboard.Name = name;
```


to assign the "Name" variable in the Dashboard instance of the GUI\_Movable class. This will be utilized later in the OnEvent() function within the instance. It's essential to remember to set the CHART\_EVENT\_MOUSE\_MOVE property to true. This enables the detection of mouse events. We will repeat this step in the constructor later on, but for now, we're keeping things uncomplicated

4. **Use GUI\_Movable class methods to make dashboard movable:**



Despite its somewhat complex name, this step is straightforward.


```
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void OnChartEvent(const int id,const long& lparam,const double& dparam,const string& sparam)
     {
      Dashboard.OnEvent(id, lparam, dparam, sparam);
     }
```




In this phase, we place the OnEvent() function within OnChartEvent() to utilize OnChartEvent()'s functionality in the .mqh file.


Finally, Here is our complete code:

Folder Structure:

- Movable Dashboard MQL5
  - Movable\_Dashboard\_MQL5.mq5
  - GUI\_Movable.mqh

1. **Movable\_Dashboard\_MQL5.mq5**



```
#include "GUI_Movable.mqh"
GUI_Movable Dashboard;

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
      //Set CHART_EVENT_MOUSE_MOVE to true to detect mouse move event
      ChartSetInteger(0, CHART_EVENT_MOUSE_MOVE, true);
      //Give dashboard's name to the class instance
      Dashboard.Name = name;

//---
      return(INIT_SUCCEEDED);
     }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void OnChartEvent(const int id, const long& lparam, const double& dparam, const string& sparam)
     {
      Dashboard.OnEvent(id, lparam, dparam, sparam);
     }
//+------------------------------------------------------------------+
```

2. **GUI\_Movable.mqh**



```
//+------------------------------------------------------------------+
//| Class GUI_Movable                                                |
//+------------------------------------------------------------------+
class GUI_Movable
     {
private:
      int               previousMouseState, mlbDownX, mlbDownY, mlbDownXDistance, mlbDownYDistance;
      bool              movingState;
public:
   		     GUI_Movable() : previousMouseState(0), mlbDownX(0), mlbDownY(0), mlbDownXDistance(0), mlbDownYDistance(0), movingState(false), Name(NULL) {};
      string            Name;
      void              OnEvent(int id, long lparam, double dparam, string sparam);
     };
//+------------------------------------------------------------------+

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void GUI_Movable::OnEvent(int id, long lparam, double dparam, string sparam)
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


**First, compile the .mqh file, then the .mq5 file. This will create a .ex5 file that can be attached to the chart.**

With these steps, we've replicated what we accomplished in Part 1 with more efficient code. Notice the significant difference in the amount of code used in the main .mq5 file between Part 1 and Part 2. And the best part is, it's only going to get better from here.

Result:

![Figure 2. Simple Movable Dashboard](https://c.mql5.com/2/56/Fig_6.gif)

**Figure 2. Simple Movable Dashboard**

### **Setting up two dashboards on the same chart using .mqh file**

Now, instead of creating the dashboard using ObjectCreate in our main .mq5 file, we'll do this in our .mqh file. You'll see how simple things become afterward.

Let's delve into the changes we'll make to our .mqh file:

1. We need to change the modifier of the string variable "Name" from public to private. The "Name" is not required in our main file - we only need it in the .mqh file. So, we make it private. This can be done as follows:
From:



```
//+------------------------------------------------------------------+
//| Class GUI_Movable                                                |
//+------------------------------------------------------------------+
class GUI_Movable
     {
private:
      int               previousMouseState, mlbDownX, mlbDownY, mlbDownXDistance, mlbDownYDistance;
      bool              movingState;
public:
   	             GUI_Movable() : previousMouseState(0), mlbDownX(0), mlbDownY(0), mlbDownXDistance(0), mlbDownYDistance(0), movingState(false), Name(NULL) {};
      string            Name;
      void              OnEvent(int id, long lparam, double dparam, string sparam);
     };
```






To:


```
//+------------------------------------------------------------------+
//| Class GUI_Movable                                                |
//+------------------------------------------------------------------+
class GUI_Movable
     {
private:
      int               previousMouseState, mlbDownX, mlbDownY, mlbDownXDistance, mlbDownYDistance;
      bool              movingState;
      string            Name;
public:
   		     GUI_Movable() : previousMouseState(0), mlbDownX(0), mlbDownY(0), mlbDownXDistance(0), mlbDownYDistance(0), movingState(false), Name(NULL) {};
      void              OnEvent(int id, long lparam, double dparam, string sparam);
     };
```




We simply changed the location of




```
string            Name;
```

that changed the variables modifier from public to private

2. Next, we add a public method named CreateDashboard(). This method will take the following inputs: name (string), xDis (int), yDis (int), xSize (int), ySize (int).

We first add this to the class member list:


```
public:
      void              OnEvent(int id, long lparam, double dparam, string sparam);
      void              CreateDashboard(string name, int xDis, int yDis, int xSize, int ySize);
```


Now, let's define this function in the global space, copying the code from our main file as follows:


```
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void GUI_Movable::CreateDashboard(string name, int xDis, int yDis, int xSize, int ySize) {
      //Create a Rectangle Label Object at (time1, price1)=(0,0)
      ObjectCreate(0, name, OBJ_RECTANGLE_LABEL, 0, 0, 0);
      //Set XDistance to 100px i.e. Distance of Rectangle Label 100px from Left of the Chart Window
      ObjectSetInteger(0, name, OBJPROP_XDISTANCE, xDis);
      //Set YDistance to 100px i.e. Distance of Rectangle Label 100px from Top of the Chart Window
      ObjectSetInteger(0, name, OBJPROP_YDISTANCE, yDis);
      //Set XSize to 200px i.e. Width of Rectangle Label
      ObjectSetInteger(0, name, OBJPROP_XSIZE, xSize);
      //Set YSize to 200px i.e. Height of Rectangle Label
      ObjectSetInteger(0, name, OBJPROP_YSIZE, ySize);
      //Set CHART_EVENT_MOUSE_MOVE to true to detect mouse move event
      ChartSetInteger(0, CHART_EVENT_MOUSE_MOVE, true);
      //Give dashboard's name to the class instance
      Name = name;
      //Redraw Chart
      ChartRedraw();
}
//+------------------------------------------------------------------+
```


Following this, we need to modify our .mq5 file:

```
#include "GUI_Movable.mqh"
GUI_Movable Dashboard1;
GUI_Movable Dashboard2;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   Dashboard1.CreateDashboard("Dashboard1", 100, 100, 200, 200);
   Dashboard2.CreateDashboard("Dashboard2", 100, 350, 200, 200);
//---
   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void OnChartEvent(const int id, const long& lparam, const double& dparam, const string& sparam)
  {
   Dashboard1.OnEvent(id, lparam, dparam, sparam);
   Dashboard2.OnEvent(id, lparam, dparam, sparam);
  }
//+------------------------------------------------------------------+
```

To break it down to you:

We begin by including the "GUI\_Movable.mqh" file, which contains the GUI\_Movable class definition. This class has methods for creating and handling events related to a movable dashboard.

Next, we declare two instances of the GUI\_Movable class, Dashboard1 and Dashboard2. These instances represent the two dashboards we will be creating and controlling in our program.

In the OnInit() function, which is automatically called when the Expert Advisor starts, we create the two dashboards by calling the CreateDashboard() method on our two instances. We pass the name of the dashboard and its position and size (in pixels) as parameters to this method. The function then returns INIT\_SUCCEEDED to indicate that the initialization was successful.

Finally, we have the OnChartEvent() function, which is triggered whenever an event occurs on the chart (like a mouse click or move). In this function, we call the OnEvent() method on our two dashboard instances, passing all the received parameters. This allows each dashboard to handle the event independently according to the logic defined in the OnEvent() method of the GUI\_Movable class.

As you can see, this approach is simple and clean while maintaining the same functionality. This makes the code highly usable in fully-fledged EAs/Indicators.

Complete code of .mqh file:

```
//+------------------------------------------------------------------+
//| Class GUI_Movable                                                |
//+------------------------------------------------------------------+
class GUI_Movable
  {
private:
   int               previousMouseState, mlbDownX, mlbDownY, mlbDownXDistance, mlbDownYDistance;
   bool              movingState;
   string            Name;
public:
		     GUI_Movable() : previousMouseState(0), mlbDownX(0), mlbDownY(0), mlbDownXDistance(0), mlbDownYDistance(0), movingState(false), Name(NULL) {};
   void              OnEvent(int id, long lparam, double dparam, string sparam);
   void              CreateDashboard(string name, int xDis, int yDis, int xSize, int ySize);
  };
//+------------------------------------------------------------------+

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void GUI_Movable::OnEvent(int id, long lparam, double dparam, string sparam)
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

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void GUI_Movable::CreateDashboard(string name, int xDis, int yDis, int xSize, int ySize) {
   //Create a Rectangle Label Object at (time1, price1)=(0,0)
   ObjectCreate(0, name, OBJ_RECTANGLE_LABEL, 0, 0, 0);
   //Set XDistance to 100px i.e. Distance of Rectangle Label 100px from Left of the Chart Window
   ObjectSetInteger(0, name, OBJPROP_XDISTANCE, xDis);
   //Set YDistance to 100px i.e. Distance of Rectangle Label 100px from Top of the Chart Window
   ObjectSetInteger(0, name, OBJPROP_YDISTANCE, yDis);
   //Set XSize to 200px i.e. Width of Rectangle Label
   ObjectSetInteger(0, name, OBJPROP_XSIZE, xSize);
   //Set YSize to 200px i.e. Height of Rectangle Label
   ObjectSetInteger(0, name, OBJPROP_YSIZE, ySize);
   //Set CHART_EVENT_MOUSE_MOVE to true to detect mouse move event
   ChartSetInteger(0, CHART_EVENT_MOUSE_MOVE, true);
   //Give dashboard's name to the class instance
   Name = name;
   //Redraw Chart
   ChartRedraw();
}
//+------------------------------------------------------------------+
```

Result:

![Figure 3. Two Movable Dashboard on the same chart](https://c.mql5.com/2/56/Fid.gif)

**Figure 3. Two Movable Dashboard on the same chart**

If you're wondering about the hierarchy of these dashboards, the one created earlier in time will be at a lower position compared to the dashboard created later.

### Conclusion

**For those who are looking to make their pre-existing dashboard movable,** the process is quite straightforward. Referring to the section titled " _[Creating the same dashboard using .mqh file](https://www.mql5.com/en/articles/12880#section3)_", you'll find that you can transform any dashboard into a movable one with just a few lines of code in your existing EA/Indicator. All it requires is the inclusion of the GUI\_Movable.mqh file and the creation of an instance of the class, with the dashboard's name assigned to this instance. With these simple steps, your dashboard becomes interactive and can be easily moved around with your mouse.

With the completion of this second part, we've successfully learned how to enhance a dashboard's interactivity by making it movable. This can be applied to any pre-existing EA/Indicator, or when building a new one from scratch.

Though this was a lengthy article, and the concept of classes can be challenging to both explain and understand, I believe that this knowledge will significantly benefit your coding journey moving forward.

I genuinely hope you found this article helpful, even if in the smallest way. Looking forward to seeing you again in the next part of the series.

_Happy coding, Happy trading!_

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/12880.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/12880/mql5.zip "Download MQL5.zip")(5.01 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/450261)**
(12)


![komlos](https://c.mql5.com/avatar/avatar_na2.png)

**[komlos](https://www.mql5.com/en/users/komlos)**
\|
29 Jul 2023 at 10:38

**Kailash Bai Mina [#](https://www.mql5.com/en/forum/450261#comment_48428494):**

Hi,

I feel great that you liked my article! Thank you for your feedback.

I don't think you need to modify my code to achieve what you're aiming for, as your goal seems distinct from what my code was initially designed for. However, you can certainly utilize my code to make your dynamically created dashboards movable.

To detect mouse clicks on the chart, you could use the ChartEvent, specifically CHARTEVENT\_CLICK. This event will provide the coordinates of the mouse click, allowing you to create a new dashboard at that location. You could then apply my code to make this newly created dashboard movable.

If you want to learn more about ChartEvent, I would recommend referring to the first part of this series, specifically the section titled ' [Decoding Chart Events: The Building Blocks of Movable GUI](https://www.mql5.com/en/articles/12751#decoding_chart_events)'.

Additionally, I strongly suggest using classes, as they will simplify your task if used correctly. If you're unfamiliar with the concept of classes, you might want to refer to my article ' [Understanding the Concept of Classes](https://www.mql5.com/en/articles/12880#section3)' for the basics.

Hope this helps!

My problem is that I don't know how I can call the OnEvent function in OnChartEvent on every dynamically created dashboard instance (since every dashboard instance has to handle the event independently as you mention it in the article). The way your code currently works, OnEvent is called on every previously created instance by defining them separately in OnChartEvent in advance. But when creating dashboard instances dynamically you can't just define them in OnChartEvent in advance because they hasn't been created yet...

Thanks

![Sahil Bagdi](https://c.mql5.com/avatar/2025/6/68402632-0431.jpg)

**[Sahil Bagdi](https://www.mql5.com/en/users/sahilbagdi)**
\|
29 Jul 2023 at 17:33

**komlos [#](https://www.mql5.com/en/forum/450261#comment_48434741):**

My problem is that I don't know how I can call the OnEvent function in OnChartEvent on every dynamically created dashboard instance (since every dashboard instance has to handle the event independently as you mention it in the article). The way your code currently works, OnEvent is called on every previously created instance by defining them separately in OnChartEvent in advance. But when creating dashboard instances dynamically you can't just define them in OnChartEvent in advance because they hasn't been created yet...

Thanks

Oh, I see where you're having difficulty. Let me help you.

I've written an Expert Advisor (EA) named MultiDash exactly like you wanted by slightly modifying my code.

I've attached it below for your reference. Please check it out and don't hesitate to ask if there's anything in my code that you don't understand. I would be happy to help.

![komlos](https://c.mql5.com/avatar/avatar_na2.png)

**[komlos](https://www.mql5.com/en/users/komlos)**
\|
29 Jul 2023 at 19:59

**Kailash Bai Mina [#](https://www.mql5.com/en/forum/450261#comment_48437832):**

Oh, I see where you're having difficulty. Let me help you.

I've written an Expert Advisor (EA) named MultiDash exactly like you wanted by slightly modifying my code.

I've attached it below for your reference. Please check it out and don't hesitate to ask if there's anything in my code that you don't understand. I would be happy to help.

Wow, thank you, it is very kind of you! I will definitely check it out!


![CapeCoddah](https://c.mql5.com/avatar/avatar_na2.png)

**[CapeCoddah](https://www.mql5.com/en/users/capecoddah)**
\|
30 Jul 2023 at 21:17

HELP!

Your preview intrigued me and made me think that I don't need to make a class Text.  Instead, I plan on using your GUI as a base class that will be inherited by the child class of each of my unique panels.  The GUI class should contain the definition for a function Move(....) but does not contain any working code.  Each of the child classes are essentially a shell inheriting from the base class.  In addition the child class will contain a Move function which will take the x&y coordinates from the GUI onEvent function and contain code to assign these coordinates to the x y ordinates of each of the specific objects on the panel.

While I'm a good programmer, I am not so good object programmer, in fact I'm a newbie.  I am getting "clsGUI::CreatePanel - cannot access private member function"  I assume this means I need some other qualifiers to allow their use directly in the child class to resolve the error.  So far my references have not identified the solution.

The include file and program are attached and originated as your code but contain many changes i made in trying to resolve the problem.

WARNING TO ANYONE ELSE THAT USES THIS CODE, IT CONTAINS MANY ERRORS & i BEAR NO RESPONSIBILITY

Thanks so much for your assistance

CapeCoddah

![Sahil Bagdi](https://c.mql5.com/avatar/2025/6/68402632-0431.jpg)

**[Sahil Bagdi](https://www.mql5.com/en/users/sahilbagdi)**
\|
31 Jul 2023 at 18:59

**CapeCoddah [#](https://www.mql5.com/en/forum/450261/page2#comment_48449905):**

HELP!

Your preview intrigued me and made me think that I don't need to make a class Text.  Instead, I plan on using your GUI as a base class that will be inherited by the child class of each of my unique panels.  The GUI class should contain the definition for a function Move(....) but does not contain any working code.  Each of the child classes are essentially a shell inheriting from the base class.  In addition the child class will contain a Move function which will take the x&y coordinates from the GUI onEvent function and contain code to assign these coordinates to the x y ordinates of each of the specific objects on the panel.

While I'm a good programmer, I am not so good object programmer, in fact I'm a newbie.  I am getting "clsGUI::CreatePanel - cannot access private member function"  I assume this means I need some other qualifiers to allow their use directly in the child class to resolve the error.  So far my references have not identified the solution.

The include file and program are attached and originated as your code but contain many changes i made in trying to resolve the problem.

WARNING TO ANYONE ELSE THAT USES THIS CODE, IT CONTAINS MANY ERRORS & i BEAR NO RESPONSIBILITY

Thanks so much for your assistance

CapeCoddah

On Line number 103 in .mqh file:

```
class clsSample : clsGUI
```

to

```
class clsSample : public clsGUI
```

Problem solved.

Concept: Inheritance type ->

Here's what each type of inheritance means:

- **Public inheritance** ( class Child : public Parent ): Public and protected members of the Parent class become public and protected members of the Child class, respectively. In essence, public inheritance means "is-a". For instance, a "Child" is a type of "Parent".

- **Protected inheritance** ( class Child : protected Parent ): Public and protected members of the Parent class both become protected members of the Child class. This means they can be accessed from the Child class and its subclasses, but not from outside these classes.

- **Private inheritance** ( class Child : private Parent ): Both public and protected members of the Parent class become private members of the Child class. This means they can only be accessed from within the Child class itself, not from its subclasses or from outside the class.


Hope it helps!

PS: use Chart Redraw otherwise it waits for a price tick.


![Can Heiken-Ashi Combined With Moving Averages Provide Good Signals Together?](https://c.mql5.com/2/56/heiken_ashi_combined_moving_averages_avatar.png)[Can Heiken-Ashi Combined With Moving Averages Provide Good Signals Together?](https://www.mql5.com/en/articles/12845)

Combinations of strategies may offer better opportunities. We can combine indicators or patterns together, or even better, indicators with patterns, so that we get an extra confirmation factor. Moving averages help us confirm and ride the trend. They are the most known technical indicators and this is because of their simplicity and their proven track record of adding value to analyses.

![Developing an MQTT client for MetaTrader 5: a TDD approach](https://c.mql5.com/2/56/mqtt-avatar.png)[Developing an MQTT client for MetaTrader 5: a TDD approach](https://www.mql5.com/en/articles/12857)

This article reports the first attempts in the development of a native MQTT client for MQL5. MQTT is a Client Server publish/subscribe messaging transport protocol. It is lightweight, open, simple, and designed to be easy to implement. These characteristics make it ideal for use in many situations.

![MQL5 — You too can become a master of this language](https://c.mql5.com/2/51/Avatar_MQL5_Voch_tamb8m-pode-se-tornar-um-mestre-nesta-linguagem.png)[MQL5 — You too can become a master of this language](https://www.mql5.com/en/articles/12071)

This article will be a kind of interview with myself, in which I will tell you how I took my first steps in the MQL5 language. I will show you how you can become a great MQL5 programmer. I will explain the necessary bases for you to achieve this feat. The only prerequisite is a willingness to learn.

![Integrating ML models with the Strategy Tester (Part 3): Managing CSV files (II)](https://c.mql5.com/2/51/Avatar_Perceptron_Multicamadas_e_o-Algoritmo_Backpropagation_Parte_3_02.png)[Integrating ML models with the Strategy Tester (Part 3): Managing CSV files (II)](https://www.mql5.com/en/articles/12069)

This material provides a complete guide to creating a class in MQL5 for efficient management of CSV files. We will see the implementation of methods for opening, writing, reading, and transforming data. We will also consider how to use them to store and access information. In addition, we will discuss the limitations and the most important aspects of using such a class. This article ca be a valuable resource for those who want to learn how to process CSV files in MQL5.

[Best articles and CodeBase updates in MQL5.community channelsFollow us to ensure you never miss out on important updates![](https://www.mql5.com/ff/sh/n9yf51p2srwzfqh5z2/02.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=dgazvhktsxqakdvarucjbvmvzenwlyje&s=98a038fe082e458df8c4a1d8e116e3a6646fd5517f06e48b2356b7ee005817d6&uid=&ref=https://www.mql5.com/en/articles/12880&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5070271891857412853)

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