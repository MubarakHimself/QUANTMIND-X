---
title: Introduction to MQL5 (Part 4): Mastering Structures, Classes, and Time Functions
url: https://www.mql5.com/en/articles/14232
categories: Expert Advisors, Machine Learning
relevance_score: 3
scraped_at: 2026-01-23T21:24:54.613708
---

[We've created a channel for MQL5 developersFollow MQL5.community on social media and be the first to receive important updatesLearn more![](https://www.mql5.com/ff/sh/a83xrgctr82w45z9z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=pwgdbvtemvkwsfltqonysypfvtrtufji&s=e99a66a1660cd810b1edbac65597df695e2c2220d1e937834f402f9aeabd4289&uid=&ref=https://www.mql5.com/en/articles/14232&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5071815013477330676)

MetaTrader 5 / Expert Advisors


### Introduction

Welcome to Part Four of our MQL5 journey! As we delve into the simplicity of structures, classes, and time functions, our focus is on making complex concepts more approachable. The goal remains the same: to create an inclusive space where the language of MQL5 becomes accessible to all. Remember, your questions are not just welcomed but encouraged; they pave the way for clarity and understanding. Let's continue our exploration together, ensuring that the path to mastering MQL5 is not only educational but also enjoyable.

I understand that programming can be overwhelming, especially for beginners. But fear not, as this article is designed to unravel complexities and provide clear insights into the world of MQL5. Please don't hesitate to ask questions, for inquiring minds pave the way for better understanding. This chapter is a stepping stone toward mastering the art of algorithmic trading, with each line of code explained to foster comprehension. Let's dive into the realm of simplicity together, because learning should be an enjoyable journey. Happy coding, and remember, curiosity is your best companion!

In this article, we will cover the following topics:

- Structures

- Classes

- Time Functions in MQL5

Dear readers, as we navigate through Part Four of our MQL5 series, I'm excited to announce that, as usual, a summary video will be provided to refresh your memories of the concepts covered in [Part Three](https://www.mql5.com/en/articles/14099). This video aims to reinforce your understanding and serve as a quick recap, ensuring that you're well-prepared for the new material ahead. So, if you ever feel the need to revisit the basics or catch up on any missed details, the summary video is here to assist you. Keep those questions coming, and let's continue this learning journey together!

YouTube

### 1. Structures

**What are Structures?**

In programming, a structure is a composite data type that allows you to group different types of data under a single name. This enables you to organize and manage logically related pieces of information in a cohesive manner. A structure can include various data types, such as integers, floating-point numbers, and characters. The main purpose of a structure is to enhance code clarity and reusability and to represent entities with multiple attributes.

In MQL5, a structure serves the same purpose. It's a user-defined data type that allows you to create a container for storing related data elements. MQL5 structures are often used to represent real-world entities in trading algorithms, financial instruments, or any scenario where different attributes need to be associated with a single entity. The syntax for declaring structures in MQL5 is similar to that of other programming languages, providing a versatile way to organize and access data efficiently within trading scripts and expert advisors.

**Analogy**

Let's imagine a structure as a magical backpack. In this backpack, you can put different things like toys, snacks, and even a little map. Each thing in the backpack is like a different piece of information. For example, the toy is the type of toy, the snack is its flavor, and the map shows you where you want to go.

Now, think of the backpack as a structure in programming. The structure holds different types of information (toys, snacks, and maps), just like how a structure holds different pieces of data in a program. So, when you want to know about the toy, you look inside the backpack and find the toy. Similarly, in a program, you use the structure to find specific pieces of information. In short, a structure is like a magical backpack that helps you organize and carry different types of things, making it easy to find what you need.

**1.1. How to Declare a Structure**

Declaring a structure in MQL5 is similar to defining the structure of your recipe card before you start filling it out. You specify the types and names of the different pieces of information you want to include.

```
struct Person
 {
    string name;
    int age;
    string address;
};
```

**Explanation:**

**“struct Person”:**

- This line initiates the declaration of a structure named Person. Think of a structure as a user-defined data type that allows you to group different types of variables under one name.

**“string name;”:**

- Inside the structure, there are three members (fields or variables) declared. The first one is “name”, and it is of type “string”. This field is intended to store the name of a “person”.


**“int age;”:**

- The second member is “age”, and it is of type “int”. This field is meant to store the age of a person.

**“string address;”:**

- The third member is “address”, and it is also of type “string”. This field is designed to store the address of a person.

Think of a structure as a custom container or a form where you can neatly organize information about a person. This “Person” structure acts like a form with specific fields (name, age, and address) where you can fill in details for each individual. Once you have this structure, you can create instances of it for different people, and each instance will hold information about a specific person.

**1.2. Utilization**

Once you've declared your structure, it's time to put it into action. Think of the structure declaration as a blueprint for creating personalized instances or objects. These instances will hold specific data for each person, just like filling out a unique recipe card for each dish you want to cook.

```
void OnStart()
  {

   struct Person
     {
      string         name;
      int            age;
      string         address;
     };

// Creating variables of type Person
   Person person1, person2;

// Filling in details for person1
   person1.name = "Israel";
   person1.age = 25;
   person1.address = "123 Main St";

// Filling in details for person2
   person2.name = "Bob";
   person2.age = 30;
   person2.address = "456 Oak St";

// Now you can use person1 and person2 with their specific details
   Print("Details of Person 1: ", person1.name, ", ", person1.age, ", ", person1.address);
   Print("Details of Person 2: ", person2.name, ", ", person2.age, ", ", person2.address);

  }
```

**Explanation:**

**“void OnStart()”:**

- This is the starting point of the script execution. Any code within this function will run when the script is started or attached to a chart.


**“Person person1, person2;”:**

- Declares two variables, “person1” and “person2”, of type “Person”. These variables act as containers to store information about individuals.


**“person1.name = "Israel";”:**

- Assigns the value "Israel" to the “name” member of “person1”.


**“person1.age = 25;”:**

- Assigns the value “25” to the “age” member of “person1”.


**“person1.address = "123 Main St";”:**

- Assigns the value "123 Main St" to the “address” member of “person1”.


**“person2.name = "Bob";” to “person2.address = "456 Oak St";”:**

- Similar assignments for the person2 variable.


**“Print("Details of Person 1: ", person1.name, ", ", person1.age, ", ", person1.address);”:**

- Prints the details of “person1” using the “Print” function.


**“Print("Details of Person 2: ", person2.name, ", ", person2.age, ", ", person2.address);”:**

- Prints the details of “person2” using the “Print” function.


**Analogy:**

Imagine you're a master creator in a magical workshop, crafting personalized information cards for special individuals. These cards, let's call them "Person Cards," have distinct sections for the name, age, and address of each person. The creation process is meticulous, ensuring that each card holds the unique essence of the individual it represents.

**Crafting the Person Cards:**

- In your workshop, you lay out two blank cards on your worktable, naming them person1 and person2. Each card is carefully designed to encapsulate the distinct details of different individuals.


**Infusing Details into person1:**

- As you embark on creating the first card, person1, you inscribe the name "Israel" into the designated space under the "Name" section. With a stroke of your magical pen, you denote the age "25" in the section dedicated to age. The address "123 Main St" finds its place under the "Address" section.


**Weaving Details into person2:**

- The process repeats as you conjure the second card, person2. This time, the name "Bob," age "30," and address "456 Oak St" are etched onto the card.

**Revealing the Person Cards**:

- With a flourish, you present the first card, person1, and narrate its contents, "Here are the details of Person 1: Israel, 25 years old, living at 123 Main St." The second card, person2, follows suit, showcasing the unique details of its bearer.


In this magical workshop, the “Person” structure serves as your enchanted blueprint, dictating the sections on each card. The variables person1 and person2 come to life as the tangible representations of these magical cards, each encapsulating the essence of a different individual. As you unveil these cards, you offer a glimpse into the enchanting details they hold.

**Note:** In the journey of programming, understanding concepts like arrays and structures might seem a bit confusing initially. One common misconception for beginners is mixing up arrays and structures due to their apparent similarities. Both involve organizing data, but are different. Arrays are like neatly arranged lists where similar pieces of information (similar data types) sit in a sequential order, accessible through indexing. On the other hand, structures are more like containers where different types of information (different data types) can be bundled together for a specific entity.

So, while arrays deal with a collection of similar elements, structures allow you to create a complex data type that holds various pieces of information. Don't worry if it feels a bit tricky at first; understanding these subtleties is part of the exciting journey into the world of programming.

### 2\. Classes

**What are Classes?**

In MQL5, classes are like blueprints or templates that guide the creation of objects. Objects, in this context, are specific instances or realizations of the class. Think of a class as a set of instructions for building various objects, and each object is like a unique item created from those instructions.

Imagine you have a class called "Fruit" with instructions for creating different fruits like apples, bananas, and oranges. The objects would be the actual apples, bananas, and oranges you create based on those instructions. Each fruit object has its own specific attributes (like color and taste) and can perform actions (like being eaten). So, in MQL5, classes define the rules, and objects are the tangible things you create following those rules.

**Analogy**

Imagine you're playing with building blocks, and you have a special set of instructions on how to build a cool spaceship. This set of instructions is like a "class." It tells you what colors to use, how to stack the blocks, and even how to make the spaceship fly (just like the blueprint for a class in programming). Now, every time you follow these instructions and build a spaceship, that spaceship is like an "object" of the class. Each spaceship you build can have its own colors and features, but they all follow the same set of instructions.

So, in computer programming, a class is like those special instructions for building something fun and exciting. It helps programmers create things (we call them objects) with specific features and actions. It's a way to organize and plan how things should be made, making it easier to create lots of similar things without starting from scratch every time. Just like having a set of instructions for building awesome spaceships, cars, or even magical creatures!

Imagine you're running a magical cookie factory where you can create different types of cookies. Now, the blueprint or recipe for baking cookies is like a class. It defines what ingredients each cookie should have and how they should be baked.

**2.1. How to Declare a Class**

In MQL5, the syntax for declaring classes involves using the "class" keyword, followed by the class name and a code block containing class members.

**Example:**

```
void OnStart()
  {

   class Person
     {
   public:
      // Attributes (data members)
      string         name;
      int            age;

      // Methods (member functions)
      void           displayInfo()
        {
         Print("Name: ", name);
         Print("Age: ", age);
        }
     };

// Creating an object of the Person class
   Person person1;

// Initializing attributes of the object
   person1.name = "John";
   person1.age = 25;

// Calling the displayInfo method to print information
   person1.displayInfo();

  }
```

**Explanation:**

**Class Declaration:**

```
class Person {
```

- This line declares a class named "Person." The class keyword is followed by the class name, and the opening curly brace " {" indicates the start of the class definition.


**Public Section:**

```
public:
```

- This line declares the access specifier "public," indicating that the following class members (attributes and methods) are accessible from outside the class.


**Attributes (Data Members):**

```
string name;
int age;
```

- These lines declare two attributes (data members) within the class: a string attribute "name" and an integer attribute "age."


**Method (Member Function):**

**```**
**void displayInfo()**
**{**
**Print("Name: ", name);**
**Print("Age: ", age);**
**}**
**```**

- This defines a method (member function) named "displayInfo" within the class. It "prints" the values of the "name" and "age" attributes using the Print function.


**Closing Brace:**

```
};
```

- The closing curly brace " }" marks the end of the class definition.


**Usage in OnStart Function:**

```
void OnStart() {
```

- The " OnStart" function is a special function in MQL5 that is executed when the script or Expert Advisor starts running.


**Creating an Object:**

```
Person person1;
```

- This line creates an object named "person1" of type "Person." It instantiates a specific instance of the class.


**Setting Attributes:**

```
person1.name = "John";
person1.age = 25;
```

- These lines set the values of the "name" and "age" attributes for the "person1" object.


**Calling Method:**

```
person1.displayInfo();
```

- This line calls the "displayInfo" method for the "person1" object, which prints the information stored in the attributes.


While classes and structures in programming share similarities, they serve distinct purposes. Both are used to define custom data types by grouping related data together. However, the key difference lies in their functionality.

Structures are more straightforward, primarily serving as containers for data without incorporating methods or functions. They're like organized storage units, grouping various data elements for convenient access. On the other hand, classes in object-oriented programming encompass not only data but also behaviors through methods. Think of classes as versatile toolkits, not only holding information, but also providing the means to perform actions or functions. So, while both classes and structures organize data, classes offer an additional layer of functionality with their methods, making them powerful tools for modeling and solving complex problems in programming.

It's completely normal if the distinction between classes and structures feels a bit overwhelming at first. In the upcoming article, I'll delve into a summary video where I'll break down these concepts visually, making them easier to grasp. So, if you're finding it a bit confusing now, don't worry because clarity is just around the corner. Keep the questions coming, and we'll explore these programming fundamentals together!

### 3. Time Functions in MQL5

**Time**

In the context of programming and trading, "time" refers to the ongoing progression of hours, minutes, and seconds. It's a fundamental aspect of algorithmic trading, as financial markets operate within specific timeframes, and traders often need to synchronize their strategies with these temporal elements.

**Functions**

In programming, a "function" is a self-contained block of code designed to perform a specific task or operation. Functions help in organizing code, making it more modular and easier to understand. They take input, process it, and provide output.

**What are Time Functions?**

Time functions in programming, specifically in MQL5, are tools that allow developers and traders to work with and manipulate time-related data within their algorithms. These functions help in retrieving current server time, converting time values, and performing various operations that involve timestamps. In the context of algorithmic trading, time functions are crucial for synchronizing trading strategies with specific timeframes, setting precise execution times, and creating time-dependent conditions for trading signals. They provide the necessary tools to navigate the temporal aspects of financial markets, enhancing the precision and effectiveness of algorithmic trading strategies.

**Analogy**

Imagine you have a magical clock that tells you when to do certain things, like when to play, when to eat, or when to sleep. In the world of computer programs, time functions are like that magical clock. They help the computer keep track of time and decide when to do different tasks.

So, if you're writing a computer program for trading, you might use time functions to say, "Hey, computer, if it's morning, do this trading strategy, but if it's nighttime, do something else." Time functions are like the instructions you give to the computer about what to do at different times, just like you follow your magical clock's instructions throughout the day.

Before we delve deeper into the fascinating realm of MQL5 time functions, let's take a moment to unravel the mysteries of “datetime”

**3.1. datetime**

In MQL5, datetime is a special data type used to represent date and time. It's like having a digital clock that can keep track of the current date and time in the world of trading. This allows traders and algorithms to precisely timestamp events, helping in analyzing and executing strategies based on specific time conditions.

**3.1.1. How to Declare a datetime Variable**

Think of a “datetime” variable as a magical clock that can store the date and time. When you declare a “datetime” variable, you can set it to a particular moment.

**Example:**

```
void OnStart()
  {

// Declaration of datetime variable
   datetime magicalClock;

// Assigning a special date and time to our magical clock
   magicalClock = D'2024.01.15 12:30:00';

// Let's tell the computer to show us the date and time inside our
   Comment(magicalClock);

  }
```

**Explanation:**

**“datetime magicalClock;”:**

- Here, we declare a variable named “magicalClock” with the “datetime” data type. This variable will hold our magical date and time.


**“magicalClock = D'2024.01.15 12:30:00';”:**

- In this line, we assign a specific date and time (January 15, 2024, at 12:30:00) to our “magicalClock” variable using the “D'YYYY.MM.DD HH:MI:SS'” format.

- “D'” indicates that we're assigning a datetime constant.

- “2024” is the year.

- “01” is the month.

- “15” is the day.

- “12” is the hour.

- “30” is the minute.

- “00” is the second.


**“Comment(magicalClock);”:**

- Finally, we use the “Comment” function to display the content of our magical clock (“magicalClock”). The computer will output the stored date and time, allowing us to witness the enchanting moment captured within the “datetime” variable.

This code snippet showcases the declaration, assignment, and printing of a “datetime” variable, providing a glimpse into the world of managing dates and times in MQL5.

![Figure 1. Result after running code in MT5](https://c.mql5.com/2/69/tt.png)

**3.2. Core Time Functions**

**3.2.1. TimeCurrent**

In the context of algorithmic trading, the “TimeCurrent()” function in MQL5 plays a crucial role in providing the current server time. This function returns a “datetime” value, which essentially represents the current timestamp on the server where your trading activities are executed.

Why is this important? Imagine you're executing a trading strategy that involves specific actions based on the time of day. For instance, you might have different approaches for the market open, close, or during volatile hours. By using “TimeCurrent()”, you can programmatically access the current time and tailor your trading decisions accordingly. It's like having a clock synchronized with the server's time, allowing your algorithm to adapt dynamically to different market conditions throughout the day.

In practical terms, you can use “TimeCurrent()” to create time-sensitive conditions in your trading algorithms. For example, you might decide to initiate trades only during certain hours or adjust your risk management strategies based on the time of day. It's a valuable tool for ensuring your algorithms are in sync with the ever-changing landscape of financial markets.

**Analogy**

Let's imagine that time in programming is like a clock that tells us when things happen. So, “TimeCurrent()” is like asking the computer, "Hey, what's the current time on your clock right now?"

When we use “TimeCurrent()”, the computer happily tells us the current time in a special format, like "2024.01.15 12:30:00". We can use this information to know when certain events occurred or plan things in our program. It's like having a magical clock that always shows the current time in the world of programming!

**Example:**

**```**
**void OnStart()**
**{**

**// Ask the magical clock for the current time**
**datetime currentTime = TimeCurrent();**

**// Display the current time on the console**
**Print("The magical clock says it's now: ", currentTime);**

**}**
**```**

**Explanation:**

**“datetime currentTime = TimeCurrent();”:**

- Here, we declare a variable named “currentTime” of type “datetime” (a data type that holds date and time information). We use the “TimeCurrent()” function to get the current server time, and the result is assigned to the “currentTime” variable.


**“Print("The magical clock says it's now: ", currentTime);”:**

- The “Print()” function is used to display messages on the console. In this line, we print a message along with the value of “currentTime”, so it shows the current time as reported by our "magical clock."


So, in simple terms, the program checks the current time using the “TimeCurrent()” function and then tells us what it found by printing a message on the console.

![Figure 2. Result after running the code in MT5](https://c.mql5.com/2/69/time_print.png)

Now, let's explore another example to deepen our understanding. We'll delve deeper into using the “TimeCurrent()” function. I'll illustrate a scenario where we check if the current server time matches a predefined magical moment. This will help reinforce the understanding of time functions in MQL5. Stay tuned for another coding adventure!

**Example:**

```
void OnStart()
  {

// Declaration of datetime variable
   datetime magicalClock;

// Assigning a special date and time to our magical clock
   magicalClock = D'2024.01.15 12:30:00';

// Check if TimeCurrent is equal to our magical clock
   if(TimeCurrent() == magicalClock)
     {
      Print("The magic moment has arrived!");
     }
   else
     {
      Print("Not yet the magic time...");
     }

  }
```

**Explanation:**

**“datetime magicalClock;”:**

- Declares a variable named “magicalClock” of the “datetime” data type.

**“magicalClock = D'2024.01.15 12:30:00';”:**

- Assigns a specific date and time (January 15, 2024, at 12:30 PM) to “magicalClock” using the “D” prefix for “datetime” literals.

**“if (TimeCurrent() == magicalClock) { ... }”:**

- Compares the current server time (“TimeCurrent()”) with the predefined “magicalClock”. If they are equal, it prints "The magic moment has arrived!" to the console; otherwise, it prints "Not yet the magic time...".


In the journey of exploring MQL5 and algorithmic trading, it's normal to find certain concepts confusing, especially for beginners. Remember that the learning process involves asking questions and seeking clarification. Don't hesitate to reach out and ask whether it's about arrays, custom functions, preprocessors, event handling, or any other programming topic we’ve covered.

Programming, like any skill, can be a bit puzzling in the beginning, but through interaction and collaboration, we can make the journey more enjoyable and understandable. The uniqueness of our learning experience lies in the questions we ask and the discussions we have. So, embrace the learning process, ask away, and let's unravel the world of algorithmic trading together!

**3.2.2. TimeGMT**

Just as your magical clock operates on a standard time, TimeGMT() in MQL5 allows us to work with Greenwich Mean Time (GMT). Think of GMT as a universal clock that people all over the world use as a reference, ensuring a standardized measure of time globally. This function facilitates the coordination and synchronization of actions on a global scale, providing a common ground for time-related operations in the world of algorithmic trading.

The “TimeGMT” function in MQL5 returns the current Greenwich Mean Time (GMT) adjusted for Daylight Saving Time (DST) based on the local time of the computer where the client terminal is running. This adjustment ensures that the returned GMT considers whether Daylight Saving Time is currently in effect.

**Analogy**

Imagine there's a magical clock in a special town called Greenwich. This town is like the superhero headquarters for timekeeping. The time on the magical clock in Greenwich is considered the superhero time that everyone around the world follows. Now, suppose you have friends in different places, each with their own local time. When you want to plan a virtual play date or a game with them, it can get a bit tricky because everyone has a different time on their clocks.

GMT comes to the rescue! It's like a superhero clock that helps everyone synchronize. GMT is the time on the magical clock in Greenwich, and when you know what time it is there, you can coordinate activities with your friends more easily, no matter where they are in the world. So, GMT is like superhero time that helps people from different places agree on when to do things together.

Imagine that in our superhero town of Greenwich, sometimes they decide to make their magical clock go faster or slower during certain parts of the year. This is like a special power-up for their clock. When they make it go faster, it's called Daylight Saving Time (DST), and when they make it go slower, it's like regular time. Now, when our friends around the world want to know the superhero time from Greenwich, they need to be aware if the magical clock is in DST mode or not. MQL5's "TimeGMT()" function is like a messenger that tells them the superhero time, considering whether the magical clock is currently in DST mode or not.

In programming, this information is crucial because it helps traders and algorithms keep track of time accurately, considering any adjustments made to the superhero clock in Greenwich. So, "TimeGMT()" is our trustworthy messenger that delivers the correct superhero time, taking into account whether DST is active or not in our magical town.

**Example:**

```
void OnStart()
  {

// Declaration of a variable to store GMT time
   datetime gmtTime;

// Assigning the current GMT time to the variable
   gmtTime = TimeGMT();

// Printing the GMT time
   Print("Current GMT Time: ", TimeToString(gmtTime));

  }
```

**Explanation:**

**“datetime gmtTime;”:**

- This line declares a variable named “gmtTime” of the “datetime” type. “datetime” is a data type in MQL5 used to represent date and time values.


**“gmtTime = TimeGMT();”:**

- This line assigns the current GMT time to the “gmtTime” variable. The “TimeGMT()” function is called to retrieve the current GMT time.


**“Print("Current GMT Time: ", TimeToString(gmtTime));”:**

- This line uses the Print function to display a message in the console. It prints the text "Current GMT Time: " followed by the GMT time converted to a string using TimeToString(gmtTime).


In summary, this code snippet declares a variable to store GMT time, assigns the current GMT time to that variable, and then prints a message with the current GMT time to the console.

**3.2.3. TimeLocal**

The “TimeLocal()” function in MQL5 is like having a clock set to your computer's local time. It helps you know what time it is in your specific geographical location. When you use “TimeLocal()”, you're getting the time as per your computer's system clock, without adjusting for any global time zones.

Let's say you're in New York, and your computer's clock is set to New York time. If you use “TimeLocal()”, it will give you the current time based on your computer's clock, making it easy to relate to your daily routine and local time settings. This function is handy when you want to work with time in the context of your time zone without worrying about global variations.

**Analogy**

Imagine you have a magical clock that tells you the time. The TimeLocal function in MQL5 is like asking your magical clock, "Hey, what's the time right here where I am?" It gives you the time according to your local surroundings, just like how your magical clock understands the time in your room.

**Example:**

```
void OnStart()
  {

// Declaration of a variable to store local time
   datetime localTime;

// Assigning the current local time to the variable
   localTime = TimeLocal();

// Printing the local time
   Print("Current Local Time: ", TimeToString(localTime));

  }
```

**Explanation:**

This code asks the computer to use the “TimeLocal()” function to find out the time in your specific location. It then prints that time on your screen. So, if your magical clock were a computer program, this is how it would tell you the time in your room!

![Figure 3. Result after running the code in MT5](https://c.mql5.com/2/69/ttime.png)

**3.2.4. TimeGMTOffset**

The “TimeGMTOffset()” function in MQL5 helps you find the time difference between your local time and GMT (Greenwich Mean Time). It's like asking, "How many hours ahead or behind is my time compared to the standard global time?"

**Formula:**

TimeGMTOffset = TimeGMT() − TimeLocal()

TimeGMTOffset is a convenient function that directly provides the time difference (offset) between GMT (Greenwich Mean Time) and the local time on the computer where the trading terminal is running, in seconds.

**Analogy**

Alright, let's imagine you have a magic clock that tells you the time, but this clock works a bit differently. It not only tells you the time, but also the difference between your time and a special time everyone around the world follows. This special time is the time everyone agrees upon, called Greenwich Mean Time (GMT). Now, instead of doing some tricky math to find out how much your time differs from this special time, you have a magic button called TimeGMTOffset. When you press this button, it gives you the answer directly: the number of seconds your time is ahead or behind this universal time.

So, if you ever want to plan something at the same time with friends from different places or even with magical creatures in different time zones, this magic button helps you figure out when to meet without needing to calculate everything yourself. It's like having a helper who makes sure everyone is on the same page, no matter where they are!

**Example:**

```
void OnStart()
  {

// Declaration of variables
   int gmtOffsetSeconds, gmtOffsetMinutes;

// Assigning the current GMT offset to the variable in seconds
   gmtOffsetSeconds = TimeGMTOffset();

// Converting seconds to minutes
   gmtOffsetMinutes = gmtOffsetSeconds / 60;

// Printing the GMT offset in minutes
   Print("Current GMT Offset (in minutes): ", gmtOffsetMinutes);

  }
```

Explanation:

**“int gmtOffsetSeconds, gmtOffsetMinutes;”:**

- Here, we're declaring two variables (“gmtOffsetSeconds” and “gmtOffsetMinutes”) to store the GMT offset in seconds and minutes, respectively.


**“gmtOffsetSeconds = TimeGMTOffset();”:**

- We're using the “TimeGMTOffset()” function to get the current GMT offset in seconds and assign it to the variable “gmtOffsetSeconds”.


**“gmtOffsetMinutes = gmtOffsetSeconds / 60;”:**

- To convert the GMT offset from seconds to minutes, we're dividing “gmtOffsetSeconds” by "60" and storing the result in “gmtOffsetMinutes.”


**“Print("Current GMT Offset (in minutes): ", gmtOffsetMinutes);”:**

- Finally, we're printing the GMT offset in minutes to the console. The “Print” function displays the text inside the quotation marks along with the calculated GMT offset in minutes.


In this code, we first retrieve the GMT offset in seconds using “TimeGMTOffset()”, then we divide that value by 60 to convert it to minutes. Finally, we print the GMT offset in minutes. If you run this code in your MQL5 script, it will display the GMT offset in minutes.

![Figure 4. Result after running code in MT5](https://c.mql5.com/2/69/ggmt.png)

A GMT offset of -60 indicates that the local time is ahead of GMT by 60 minutes.

**3.2.5. TimeToStruct**

In MQL5, the “TimeToStruct()” function is used to convert a timestamp (represented as the number of seconds since January 1, 1970) into a structured format. This structured format is represented by the “MqlDateTime” predefined structure, which includes separate members for the year, month, day, hour, minute, and second.

**Example:**

**```**
**void OnStart()**
**{**

**// Declare an MqlDateTime variable**
**MqlDateTime myTime;**

**// Convert the current timestamp to a structured format**
**TimeToStruct(TimeCurrent(), myTime);**

**// Access individual components of the structured time**
**Print("Current Year: ", myTime.year);**
**Print("Current Month: ", myTime.mon);**
**Print("Current Day: ", myTime.day);**
**Print("Current Hour: ", myTime.hour);**
**Print("Current Minute: ", myTime.min);**
**Print("Current Second: ", myTime.sec);**

**}**
**```**

In the example above, “TimeCurrent()” returns the current timestamp, and “TimeToStruct” converts that timestamp into a structured format stored in the “MqlDateTime” variable “myTime”. After the conversion, you can access specific components of the time (year, month, day, etc.) using the members of the “myTime” structure.

Note: _More explanations to come on MqlDateTime as this article progresses._

**Analogy**

Let's dive a bit deeper into the magic of “TimeToStruct()”. Imagine we have a huge stash of seconds that started ticking away in the year 1970 (which is like the magic starting point in computer timekeeping). Now, these seconds have been piling up, and we want to know how many hours, minutes, days, and even years are in this big number.

Here's where “TimeToStruct()” steps in. It takes that massive pile of seconds and breaks them down into a more human-friendly format. So, if we have a whopping number like 100,000 seconds, “TimeToStruct()” will tell us, "Hey, that's about 27 hours, 46 minutes, and 40 seconds!" It's like having a magical calculator that transforms a raw number of seconds into a detailed breakdown of time, considering everything since the magical starting point in 1970.

Imagine you have a magical time capsule called “MqlDateTime”, and when you feed it a specific moment in time (like the current time), it opens up to reveal the details neatly organized. The “TimeToStruct()” function is like casting a spell on this time capsule, turning the raw seconds since 1970 into a beautifully organized set of information.

So, if you have a second count, “TimeToStruct()” waves its magic wand, and suddenly you have a clear understanding of the year, month, day, hour, minute, second, day of the week, and day of the year—all neatly packed into the “MqlDateTime” structure. It's like turning a mysterious time code into a readable date and time book, making your programs more versatile and time-savvy!

**Example:**

```
void OnStart()
  {

// Declare an MqlDateTime variable
   MqlDateTime myTime;

// Convert the number of seconds into a structured format
   TimeToStruct(100000, myTime);

// Now, myTime will tell us the breakdown since 1970
   Print("Years: ", myTime.year);
   Print("Months: ", myTime.mon);
   Print("Days: ", myTime.day);
   Print("Hours: ", myTime.hour);
   Print("Minutes: ", myTime.min);
   Print("Seconds: ", myTime.sec);

  }
```

**Explanation:**

**“MqlDateTime myTime;”:**

- We're declaring a magical clock named “myTime” using the “MqlDateTime” structure.


**“TimeToStruct(100000, myTime);”:**

- Here, we're casting a spell called “TimeToStruct” to convert the number of seconds (100,000 in this case) into the structured format stored in “myTime”. It's like telling the magical clock to decode a specific moment in time.


**“Print("Years: ", myTime.year);”:**

- Now, we command the magical clock to reveal the years since 1970 using its year attribute. The same goes for months, days, hours, minutes, and seconds.


In simpler terms, it's as if we took a moment in time (100,000 seconds since 1970) and asked our magical clock to break it down into years, months, days, and so on. It's a way to understand time in a more detailed and structured manner. How enchanting!

Dear coding enthusiasts, as you navigate the intricacies of time functions in MQL5, remember that every question is a key to unlocking new realms of understanding. This journey is a step-by-step exploration, much like wandering through a magical forest of code. If you find yourself pondering, don't hesitate to ask questions, as they are the lanterns guiding us through the realms of knowledge.

Note: _Today's insights into “MqlDateTime” are just the beginning. The adventure of predefined structures unfolds further in the next article._

### Conclusion

As we wrap up this chapter, we've delved into the intriguing realms of structures, classes, and the concept of time in MQL5. Remember, learning is a dynamic journey, and questions are the compass that steers you through uncharted territories. If you find yourself at a crossroads, don't hesitate to ask; your curiosity is the engine that drives progress. Stay tuned for the upcoming articles, where we'll unravel more layers of MQL5's programming magic. Your questions are not just welcome; they're the catalysts for deeper understanding. Happy coding, and let the quest for knowledge continue!

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Introduction to MQL5 (Part 36): Mastering API and WebRequest Function in MQL5 (X)](https://www.mql5.com/en/articles/20938)
- [Introduction to MQL5 (Part 35): Mastering API and WebRequest Function in MQL5 (IX)](https://www.mql5.com/en/articles/20859)
- [Introduction to MQL5 (Part 34): Mastering API and WebRequest Function in MQL5 (VIII)](https://www.mql5.com/en/articles/20802)
- [Introduction to MQL5 (Part 33): Mastering API and WebRequest Function in MQL5 (VII)](https://www.mql5.com/en/articles/20700)
- [Introduction to MQL5 (Part 32): Mastering API and WebRequest Function in MQL5 (VI)](https://www.mql5.com/en/articles/20591)
- [Introduction to MQL5 (Part 31): Mastering API and WebRequest Function in MQL5 (V)](https://www.mql5.com/en/articles/20546)
- [Introduction to MQL5 (Part 30): Mastering API and WebRequest Function in MQL5 (IV)](https://www.mql5.com/en/articles/20425)

**[Go to discussion](https://www.mql5.com/en/forum/462538)**

![Developing a Replay System — Market simulation (Part 22): FOREX (III)](https://c.mql5.com/2/57/replay_p22_avatar.png)[Developing a Replay System — Market simulation (Part 22): FOREX (III)](https://www.mql5.com/en/articles/11174)

Although this is the third article on this topic, I must explain for those who have not yet understood the difference between the stock market and the foreign exchange market: the big difference is that in the Forex there is no, or rather, we are not given information about some points that actually occurred during the course of trading.

![Population optimization algorithms: Stochastic Diffusion Search (SDS)](https://c.mql5.com/2/59/SDS_avatar.png)[Population optimization algorithms: Stochastic Diffusion Search (SDS)](https://www.mql5.com/en/articles/13540)

The article discusses Stochastic Diffusion Search (SDS), which is a very powerful and efficient optimization algorithm based on the principles of random walk. The algorithm allows finding optimal solutions in complex multidimensional spaces, while featuring a high speed of convergence and the ability to avoid local extrema.

![Neural networks are easy (Part 59): Dichotomy of Control (DoC)](https://c.mql5.com/2/58/logo__1.png)[Neural networks are easy (Part 59): Dichotomy of Control (DoC)](https://www.mql5.com/en/articles/13551)

In the previous article, we got acquainted with the Decision Transformer. But the complex stochastic environment of the foreign exchange market did not allow us to fully implement the potential of the presented method. In this article, I will introduce an algorithm that is aimed at improving the performance of algorithms in stochastic environments.

![Developing a Replay System — Market simulation (Part 21): FOREX (II)](https://c.mql5.com/2/57/replay_p21-avatar.png)[Developing a Replay System — Market simulation (Part 21): FOREX (II)](https://www.mql5.com/en/articles/11153)

We will continue to build a system for working in the FOREX market. In order to solve this problem, we must first declare the loading of ticks before loading the previous bars. This solves the problem, but at the same time forces the user to follow some structure in the configuration file, which, personally, does not make much sense to me. The reason is that by designing a program that is responsible for analyzing and executing what is in the configuration file, we can allow the user to declare the elements he needs in any order.

[![](https://www.mql5.com/ff/si/5k7a2kbftss6k97n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F1171%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dbest.vps%26utm_content%3Drent.vps%26utm_campaign%3D0622.MQL5.com.Internal&a=nwegcasiojnqcoyrdlgofmjtfardztwf&s=d64d6f3c87f2458cba81f6d7b6694dd9e89dd354d4abc1d0584e405285806c9f&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=iujhdbokasujnefppftqmmcrvyfihimx&ssn=1769192693190336466&ssn_dr=0&ssn_sr=0&fv_date=1769192693&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F14232&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Introduction%20to%20MQL5%20(Part%204)%3A%20Mastering%20Structures%2C%20Classes%2C%20and%20Time%20Functions%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176919269323342006&fz_uniq=5071815013477330676&sv=2552)

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