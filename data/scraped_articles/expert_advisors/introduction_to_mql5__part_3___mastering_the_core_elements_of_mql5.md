---
title: Introduction to MQL5 (Part 3): Mastering the Core Elements of MQL5
url: https://www.mql5.com/en/articles/14099
categories: Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T21:25:13.660952
---

[![](https://www.mql5.com/ff/sh/6zw0dkux8bqt7m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
MQL5 Channels - Messenger for traders\\
\\
Install the app and receive market analytics and trading tips.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=iuciwacmrxvmiibwyujliagqikizpsoo&s=268cbb13914c54b6c5c875db99b154944f6e0122b3400b54c9ac0d4f69f0f0d6&uid=&ref=https://www.mql5.com/en/articles/14099&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5071818878947897096)

MetaTrader 5 / Expert Advisors


### Introduction

Welcome back, fellow traders and aspiring algorithmic enthusiasts! As we step into the third chapter of our MQL5 journey, we stand at the crossroads of theory and practice, poised to unravel the secrets behind arrays, custom functions, preprocessors, and event handling. Our mission is to empower every reader, regardless of their programming background, with a profound understanding of these fundamental MQL5 elements.

What makes this journey unique is our commitment to clarity. Every line of code will be explained, ensuring that you not only grasp the concepts but also witness them come to life in practical applications. The goal is to create an inclusive space where the language of MQL5 becomes accessible to all. Your questions are not only welcome but encouraged as we navigate through the basics. This is just the beginning, laying the groundwork for the engaging coding adventures that lie ahead. Let's learn, build, and explore the world of MQL5 programming together.

So, tighten your focus and join us on this immersive exploration. Together, we'll transform theoretical knowledge into practical expertise, ensuring that the embrace of understanding extends to everyone. The stage is set, and the code is ready, so let's dive into the world of MQL5 mastery!

In this article, we will cover the following topics:

- Array

- Custom Functions

- Preprocessor

- Event Handling

Before we embark on the hands-on journey into the depths of MQL5 coding in Part 3, I have an exciting announcement. To enhance your learning experience, I'll be providing a video summary that summarizes the key concepts and highlights from [Part 2](https://www.mql5.com/en/articles/13997) of our exploration. This visual recap aims to reinforce your understanding, making complex topics more digestible.

YouTube

**1.  Array**

**What are Arrays?**

Arrays are foundational concepts in programming that provide efficient ways to organize and store data. It allows you to store and organize multiple values under a single variable name. Instead of creating separate variables for each piece of data, you can use an array to hold a collection of elements of the same type. These elements can be accessed and manipulated using indices.

**Analogy**

An array is like a magic box where you can keep a bunch of similar things together. Imagine you have a row of boxes, and each box has its number. These numbers help you find and organize your things. Let's say you have a row of boxes numbered 0, 1, 2, 3, and so on (counting always starts from 0). In each box, you can keep something special, like your favorite toys or candies. When you want to play with a specific toy, say the box number, and voila! You find exactly what you're looking for. So, an array in programming is like this row of numbered boxes. It lets you keep many similar items in one place and find them easily using their special numbers. It's a cool way to organize stuff!

**1.1. How to Declare an Array**

In MQL5, you declare an array by specifying its data type and name, followed by the array dimensions (if any).

**Here's the basic syntax:**

data\_type array\_name\[array\_size\];

- **data\_type:** The type of data the array will hold (e.g., int, double, string, etc.)
- **array\_name:** The name you give to the array.

- **array\_size:** The size or length of the array, specifying how many elements it can hold.


**Example:**

```
void OnStart()
  {

   int closePrices[5]; // An integer array with a size of 5, means the array has 5 integer elements
   double prices[10];  // A double-precision floating-point array with a size of 10

  }
```

**1.2. Assigning Values to Array**

In MQL5, you can assign values to an array during its declaration or separately after declaring it. Let's break down both methods:

**1.2.1. Assigning Values After Declaration**

In this method, you declare the array first and then assign values to each element separately. It allows you to assign values dynamically during the program's execution or based on certain conditions.

**Example:**

```
void OnStart()
  {

   int closePrices[5];
   closePrices[0] = 10;
   closePrices[1] = 20;
   closePrices[2] = 30;
   closePrices[3] = 40;
   closePrices[4] = 50;

  }
```

**1.2.2. Assigning Values during Declaration**

In this method, you declare the array and provide the values inside curly braces {}. The size of the array is determined by the number of values you provide. In this example, the array “closePrices” is declared with a size of 5 and is initialized with the values 10, 20, 30, 40, and 50.  Both methods are valid, and you can choose the one that fits your coding style or the requirements of your specific program.

**Example:**

```
void OnStart()
  {

   int closePrices[5] = {10, 20, 30, 40, 50};

  }
```

**1.3. How to Access Elements of an Array**

In most programming languages, including MQL5, array indices start from 0. So, by default, when you declare an array and access its elements, the counting begins with 0.

![Figure 1. Accessing Elements of an Array](https://c.mql5.com/2/64/array.png)

**Example:**

```
void OnStart()
  {

// Declare an integer array with a size of 5
   int closePrices[5];

// Assign values to array elements
   closePrices[0] = 10;
   closePrices[1] = 20;
   closePrices[2] = 30;
   closePrices[3] = 40;
   closePrices[4] = 50;

   // Access and print array elements
   Print("Element at index 0: ", closePrices[0]); // Output: 10
   Print("Element at index 1: ", closePrices[1]); // Output: 20
   Print("Element at index 2: ", closePrices[2]); // Output: 30
   Print("Element at index 3: ", closePrices[3]); // Output: 40
   Print("Element at index 4: ", closePrices[4]); // Output: 50

  }
```

**Explanation:**

**Array Declaration:**

```
int closePrices[5];
```

This line declares an integer array named “closePrices”  with a fixed size of 5 elements. In MQL5, arrays can store elements of the same data type, and indexing starts from 0, so this array has indices 0 to 4.

**Assigning Values to Array Elements:**

```
   closePrices[0] = 10;
   closePrices[1] = 20;
   closePrices[2] = 30;
   closePrices[3] = 40;
   closePrices[4] = 50;
```

Here, values are assigned to each element of the array using indices. The array “closePrices”  now holds the values 10, 20, 30, 40, and 50 in their respective positions.

**Accessing and Printing Array Elements:**

```
   Print("Element at index 0: ", closePrices[0]); // Output: 10
   Print("Element at index 1: ", closePrices[1]); // Output: 20
   Print("Element at index 2: ", closePrices[2]); // Output: 30
   Print("Element at index 3: ", closePrices[3]); // Output: 40
   Print("Element at index 4: ", closePrices[4]); // Output: 50
```

The “Print” statements demonstrate how to access and display the values stored at specific indices in the array. Indices 0 through 4 are accessed, and the corresponding values are printed to the console.

In this introductory exploration of MQL5 arrays, we've uncovered the basics, using the metaphor of organized containers for data. Through code examples, we've demonstrated the declaration, initialization, and access of array elements.  It's worth noting that MQL5 comes equipped with a range of powerful array functions. While these will be explored in detail later, for now, it's essential to familiarize yourself with the fundamental array concept.

**2\. Custom Functions**

**What are Common Functions?**

Custom functions, also known as user-defined functions, are specific segments of code that you create to perform a particular task or set of tasks in a programming language. In the context of MQL5, which is used for algorithmic trading, custom functions are written by the programmer to encapsulate a series of actions, calculations, or operations. These functions allow you to modularize your code, making it more readable, reusable, and easier to maintain. Instead of duplicating the same code in multiple places within your program, you can create a custom function that encapsulates that logic, and then call the function whenever that specific task needs to be executed.

Building upon the foundation laid in our previous article where we explored common functions like Alert, Comment, and Print, we now delve into the empowering realm of crafting your functions in MQL5. Just as a skilled artisan refines their tools, creating custom functions allows you to tailor your code precisely to the demands of your trading strategies.

**Analogy**

Imagine you're in a kitchen, preparing a delightful meal (your code). In the culinary world of programming, functions are like recipes, providing step-by-step instructions to create specific dishes. Now, let's explore the analogy of custom functions versus common functions. Think of common functions as widely recognized recipes available in cookbooks. These recipes are standard and commonly used, like making a cup of coffee or boiling an egg. In programming terms, functions like “Alert”, “Comment”, and “Print” are akin to these well-known recipes. They serve specific purposes and are readily available for immediate use.

On the other hand, custom functions are like your unique, secret family recipes. These are dishes you've created, tailored to your taste and preferences. Similarly, custom functions in coding are personalized sets of instructions crafted by the programmer to perform specific tasks. They encapsulate a series of steps to achieve a particular outcome, much like your secret recipe for the perfect chocolate cake.The main distinction lies in the level of customization. Common functions are like instant meals convenient, quick, and widely applicable. They're efficient for common tasks, just like a pre-made dinner you can heat up. On the flip side, custom functions offer precision and personalization. They're like cooking from scratch, allowing you to adjust ingredients, flavors, and techniques to meet the unique requirements of your coding cuisine.

**2.1. How to Create a Function**

Creating a function in MQL5 involves specifying the function's name, defining its parameters (if any), specifying its return type (if it returns a value), and providing the code block that constitutes the function's body.

**Step 1: Determine the Purpose of the Function**

Decide what task or calculation the function will perform. This will help you determine the function's name, input parameters, and return type (if any). In this case, lets create a function that multiplies 2 numbers.

**Step 2: Declare the Function**

Declare the function with the chosen name, input parameters, and return type (if applicable). The declaration should be done at the global scope, outside any specific function or event handler.

_Note: Event handlers in MQL5 are functions that automatically respond to specific events, such as market price changes (OnTick), chart interactions (OnChartEvent), and timed intervals (OnTimer). They streamline the execution of code based on real-time or scheduled occurrences in the MetaTrader platform.  More explanation to come._

**Syntax:**

```
return_type FunctionName(parameter1_type parameter1, parameter2_type parameter2, ...)
   {
    // Function body (code block)
    // Perform actions or calculations
    // Optionally, return a value using the 'return' keyword if the function has a return type
   }
```

**Step 3: Define the Function Body**

Inside the function, add the code that performs the desired task or calculation. Include any necessary variables and logic.

**Example:**

```
double MyFunction(int param1, double param2)
     {
      // Function body: Calculate the product of the two parameters
      double result = param1 * param2;

      // Return the result
      return result;
     }
```

**Explanation:**

**“double MyFunction(int param1, double param2)”:**

- This line defines a function named “MyFunction”.  It takes two parameters: “param1” of type “int” (integer) and “param2” of type “double” (floating-point number). The function is expected to return a value of type “double”.

**{:**

- The opening curly brace “{“ marks the beginning of the function's body, where the actual code of the function resides.


**“double result = param1 \* param2;”:**

- This line calculates the product of “param1” and “param2” and stores the result in a variable named result. Since both “param1” and “param2” are of numeric types, the multiplication operation (\*) results in a double.


**“return result;”:**

- This line exits the function (“MyFunction”) and returns the value stored in the result variable back to the code that called the function. Since the return type of the function is “double”, it means the caller expects to receive a double value.


**“}”:**

- The closing curly brace “}” marks the end of the function's body.


**Step 4: Use the Function**

Call the function from other parts of your code. You can use the function in event handlers, other functions, or any suitable context.

**Example:** **```**
**void OnStart()**
**{**

**// Calling the custom function**
**double result = MyFunction(5, 3.5);**

**// Printing the result**
**Print("The result is: ", result); // The output will be 17.5**

**}**
**```**

The function provided, “MyFunction”, has been designed to automatically multiply two numbers. By calling this function and providing the required numeric values as arguments, the calculation is performed seamlessly, and the result is returned. This abstraction simplifies the process of multiplying numbers, enhancing code readability and reusability.

I encourage you to engage actively with the content and feel free to ask any questions that may arise. Your understanding is my priority, and I am here to provide clear explanations or offer further clarification on any related topic. Your questions contribute to a richer learning experience, so don't hesitate to reach out. I'll do my best to respond promptly and assist you in comprehending the concepts discussed in the article.

**3\. Preprocessors**

**What are Preprocessors?**

In the dynamic world of MQL5 programming, preprocessors stand as instrumental tools shaping the way code is interpreted and executed. A preprocessor is a specialized software component that operates on the source code before the actual compilation process begins. Its primary role is to handle instructions and directives that influence the behavior, structure, and characteristics of the resulting program.

Preprocessors work by transforming or preprocessing the original source code. They interpret specific directives marked by a preceding hash symbol (“#”) and act accordingly, altering the code's content before it undergoes compilation. These directives, also known as preprocessor directives, encompass a range of functionalities, including defining constants, incorporating external files, setting program properties, and enabling conditional compilation. Each directive serves a distinct purpose in molding the code according to the developer's requirements.

The unique feature of preprocessors lies in their ability to execute before the actual compilation takes place. This early phase of code manipulation allows developers to establish certain conditions, configurations, or inclusions that will affect the final compiled program.

![Figure 2. Preprocessors in MetaEditor](https://c.mql5.com/2/65/Preprocessors.png)

**Note:** In MQL5 programming, it's essential to remember that preprocessors don't end with a semicolon (;) like regular statements, and they are declared in the global space of your code. These unique instructions, such as “#define” for macros or “#include” for file inclusion, play a vital role in shaping the behavior and structure of your program. So, when working with preprocessors, skip the semicolon and ensure they have their own space in the global spotlight!

**Analogy**

Let's imagine that writing code is like telling a robot what to do. Now, before we tell the robot, there's a special friend we have called the "Preprocessor." This friend helps us get everything ready for the robot so that it understands our instructions perfectly. Think of the Preprocessor as a magical helper that looks at our instructions before the robot does. Its job is to make things simpler and more organized. Sometimes, we want to use special words that mean something important, like saying "MAGIC\_NUMBER" instead of the number 10. The Preprocessor helps us by understanding these special words and replacing them with real numbers.

**Example:**

```
#define MAGIC_NUMBER 10

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void OnStart()
  {

   int result = MAGIC_NUMBER + 5;
   Comment(result); // Output will be 15

  }
```

**Explanation:**

**“#define MAGIC\_NUMBER 10”:**

- This line is like saying, "Hey Preprocessor, whenever you see the word 'MAGIC\_NUMBER' in my code, replace it with the number 10." It's like creating a special word that stands for a particular value.

**“void OnStart()”:**

- This part is where the main action begins. “OnStart” is a special function in MQL5 that gets executed when your script starts running.

**“int result = MAGIC\_NUMBER + 5;”:**

- Here, we're using our special word 'MAGIC\_NUMBER.' The Preprocessor sees this and replaces 'MAGIC\_NUMBER' with 10. So, this line is the same as saying “int result = 10 + 5;”

**“Comment(result); // Output will be 15”:**

- The “Comment” function is like telling the robot to say something. In this case, we're asking the robot to say the value of 'result,' which is 15. So, when you run this script, it will display "15" in the comments on your chart.

The cool thing is that the Preprocessor works even before we tell the robot what to do! It looks at our special words and prepares everything, so when the robot starts working, it already knows what to do. So, the Preprocessor is like our secret friend who makes sure our instructions are clear, our code is neat, and everything is set up perfectly for the robot to follow. It's like having a superhero friend in the coding world!

**3.1. Categories of Preprocessors**

**3.1.1. Macro Substitution (#define)**

In the coding world, Macros are used to create constants. Constants are like unchanging values. With macros, we can create special words that stand for these values.

**Example:**

```
#define MAGIC_NUMBER 10

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void OnStart()
  {

   int result = MAGIC_NUMBER + 5;

  }
```

Here, instead of using the number “10” every time, we simply use the word “MAGIC\_NUMBER” making our code cleaner and more readable.

**3.1.1.1.  Parameterized Macros (Function-Like Macros)**

Imagine you have a magical recipe, and sometimes you want to change a key ingredient. Parameterized macros let you create flexible instructions.

**Example:**

```
#define MULTIPLY(x, y) (x * y)

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void OnStart()
  {

   int result = MULTIPLY(3, 5); // Result is 15

  }
```

Here, the macro “MULTIPLY” takes two ingredients (numbers) and multiplies them. It's like having a magical function in your code!

**Explanation:**

**“#define MULTIPLY(x, y) (x \* y)”:**

- This line is like telling the computer, "Hey, whenever I say 'MULTIPLY,' I mean take two numbers (let's call them 'x' and 'y') and multiply them together." It's like creating a multiplication machine using our magic word.


**“int result = MULTIPLY(3, 5);”:**

- Here, we're using our magic word “MULTIPLY.” The computer sees this and knows it means "take the numbers 3 and 5 and multiply them." So, it replaces MULTIPLY(3, 5) with (3 \* 5).


**“// Result is 15”:**

- The result of multiplying 3 and 5 is 15. So, after the computer does the calculation, the value of 'result' becomes 15.

In essence, this code simplifies the multiplication process. Instead of writing “3 \* 5” directly, we use the magic word 'MULTIPLY' to make our code more readable and easy to understand. It's like having a mini math assistant in our code!

**3.1.1.2. The #undef directive**

The “#undef” directive is like saying, "Forget what I told you earlier; let's start fresh." In the world of MQL5 programming, it allows us to undefine or "uncreate" a previously defined macro. It's like erasing a note on a magic whiteboard.

**Example:**

```
#define MAGIC_NUMBER 10

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void OnStart()
  {

   int result = MAGIC_NUMBER + 5;
#undef MAGIC_NUMBER // Undefining MAGIC_NUMBER

  }
```

****Explanation:****

****“#define MAGIC\_NUMBER 10”:****

- Initially, we define a macro called “MAGIC\_NUMBER” and set its value to 10. It's like saying, "Whenever I use 'MAGIC\_NUMBER,' I mean the number 10."

**“int result = MAGIC\_NUMBER + 5;”:**

- We use the “MAGIC\_NUMBER” in a calculation, where the result becomes 15.

**“#undef  MAGIC\_NUMBER”:**

- With “#undef MAGIC\_NUMBER” we're saying, "Okay, forget what “MAGIC\_NUMBER” meant before." It's like erasing the definition, making “MAGIC\_NUMBER” undefined.

Now, if you try to use “MAGIC\_NUMBER” after the “#undef” line, it will result in an error because the computer no longer knows what 'MAGIC\_NUMBER' means. This can be useful if you want to redefine or stop using a macro in a specific part of your code. It's like telling the computer, "I used this magic word for a while, but now let's move on to something else."

**3.1.2. Program Properties (#property)**

In the coding world, every program has its unique traits, like a character in a story. With “#property” we can give our program special features and qualities. It's like saying, "Hey computer, here are some things that make my program special". Imagine you're writing a book or a song, you might want to tell people who created it and when it was made. “#property” helps us do the same for our programs. It's like adding a small note at the beginning of our code, saying, "This program is version 1.0, and I made it in 2022.

**Example:**

```
#property copyright "crownsoyin"
#property version   "1.00"
```

**Result After running code**

![Figure 3. Result after running the code in MT5](https://c.mql5.com/2/65/PROP.png)

“#property” is like the cover page of a book or the opening credits of a movie. It sets the tone, provides information about the program, and helps everyone, including the coder and others who might read the code later, understand what's going on.

**3.1.3 Include Directive (#include)**

**What is an Include file?**

Imagine you have a magical recipe book, but some recipes are stored in a different book. “#include” is like saying, "Hey, let's bring in those extra recipes from that other book and combine them into one big recipe collection." In coding, it helps us merge external files with our main code, making everything accessible in one place. Think of coding as constructing a house with different rooms. Each room has specific functions, like a kitchen or a bedroom. “#include” lets us reuse these rooms (functions and structures) in other houses (programs). It's like saying, "I built this fantastic kitchen; now, I can use it in all my houses without building it from scratch."

**Example:**

```
#include "extraRecipes.mqh" // Include external file with extra recipes
```

**Explanation:**

**“#include”:**

- Think of #include” as a magical command that says, "Bring in something special from another place and add it here." It's like having a secret portal to fetch additional ingredients for our recipe.

**‘"extraRecipes.mqh"’:**

- The text inside the double quotes is the name of the external file we want to include. In this case, it's "extraRecipes.mqh." This file contains additional recipes or code that we want to use in our main program.

So, when we write ‘#include "extraRecipes.mqh"’, it's akin to opening a secret cookbook and saying, "Let's add these extra recipes to our main cooking instructions." This helps keep our main code clean and organized, while still having access to the extra magic from "extraRecipes.mqh." It's like expanding our magical cookbook with more spells and enchantments! When you use “#include” to include an external file in your code, any functions or structures defined in that external file become accessible and can be used in your main code.

**Examples:**

External File: “extraRecipes.mqh”

```
// extraRecipes.mqh
int MultiplyByTwo(int number) // function from an external file
{
    return number * 2;
}
```

**Main Code:**

```
// Your main code
#include "extraRecipes.mqh" // Include external file with extra recipes
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void OnStart()
  {

   int result = MultiplyByTwo(5);  // Using the function from the included file
   Print("Result: ", result);      // Output: Result: 10

  }
```

In this example, the “MultiplyByTwo” function is defined in the external file “extraRecipes.mqh.” By using #include ‘"extraRecipes.mqh"’ in your main code, you can now call and use the “MultiplyByTwo” function as if it was defined directly in your main code.

Understanding the intricacies of code inclusion and function usage might seem a bit complex at first, and that's okay! Programming is like learning a new language, and every journey starts with the first step. If you find yourself scratching your head or feeling a bit puzzled, don't hesitate to ask questions. Your curiosity fuels the learning process, and I'm here to help unravel any mysteries. So, embrace the learning adventure, and remember that questions are the keys to unlocking a deeper understanding. Feel free to drop any queries you have, and let's embark on this coding journey together!

**3.1.4. Compilation Conditions  ( #ifdef, #ifndef, #else, #endif)**

Compilation conditions allow us to include or exclude portions of code during the compilation process. It's like having special instructions that guide the compiler on what to include based on certain conditions.

**3.1.4.1. “#ifdef” Directive**

The #ifdef directive in MQL5 is a preprocessor directive that checks whether a specific symbol is defined. If the symbol is defined, the code block following #ifdef is included during compilation; otherwise, the code block following #else or #endif is included.

**Example:**

```
#define MAGIC_NUMBER 10
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void OnStart()
  {

#ifdef MAGIC_NUMBER
   Print(MAGIC_NUMBER);
#else
   Print(MAGIC_NUMBER);
#endif

  }
```

**Explanation:**

- In this example, “MAGIC\_NUMBER” is defined using “#define”.
- The “#ifdef MAGIC\_NUMBER” checks if “MAGIC\_NUMBER” is defined.

- Since it is defined, the code block following “#ifdef” is included, resulting in the compilation of the first “Print” statement.

- If “MAGIC\_NUMBER” were not defined, the code block following “#else” would be included, and the second “Print” statement would be compiled.


“#ifdef” is often used for conditional compilation, allowing developers to include or exclude specific code sections based on predefined symbols. It’s a valuable tool for creating adaptable and configurable code, allowing developers to tailor their applications based on defined symbols or conditions during the compilation process.

**3.1.4.2. “#ifndef” Directive**

The “#ifndef” directive in MQL5 is a preprocessor directive that checks whether a specific symbol is not defined. If the macro is not defined, the code block following “#ifndef” is included during compilation; otherwise, the code block following “#else” or “#endif” is included.

**Example:**

```
void OnStart()
  {

#ifndef MAGIC_NUMBER
   Print(MAGIC_NUMBER);
#else
   Print(MAGIC_NUMBER);
#endif

  }
```

**Explanation:**

- The “#ifndef MAGIC\_NUMBER” checks whether “MAGIC\_NUMBER” is not defined.

- If “MAGIC\_NUMBER” is not defined, the code block following “#ifndef” is included, and it prints a message indicating that “MAGIC\_NUMBER” is not defined.

- If “MAGIC\_NUMBER” is defined, the code block following “#else” is included, and it prints the value of “MAGIC\_NUMBER”.


This code demonstrates the use of conditional compilation based on whether a specific macro (“MAGIC\_NUMBER” in this case) is defined or not. Depending on the presence or absence of the macro, different code blocks are included during compilation.

_Note: “MAGIC\_NUMBER” was not defined in this example_

**3.1.4.3. “#endif” Directive**

The “#endif” directive in MQL5 marks the end of a conditional block of code initiated by directives such as “#ifdef” or “#ifndef”. It serves as a signal to the preprocessor that the conditional compilation section is concluded, and the subsequent code should be processed for compilation. It doesn't have any conditions or parameters, its purpose is to denote the end of the conditional compilation block.

**Example:**

```
#define MAGIC_NUMBER 10
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void OnStart()
  {

#ifndef MAGIC_NUMBER
   Print(MAGIC_NUMBER);
#else
   Print(MAGIC_NUMBER);
#endif

  }
```

**Explanation:**

- “#endif” marks the end of the conditional block, and the subsequent code is processed normally.


_Note: Always pair “#endif” with an opening conditional directive (#ifdef or #ifndef) to maintain proper syntax and avoid compilation errors._

In this brief exploration of MQL5 preprocessors, we covered macros for creating constants, program properties (#properties) to define characteristics, include files for modularity, and conditional compilation for flexible code. While this serves as an introduction, there's much more to discover. These tools form the building blocks of efficient MQL5 programming, providing versatility and adaptability. Don't hesitate to ask questions and dive deeper into these concepts as you continue your journey in crafting powerful trading algorithms. Embrace curiosity, and let your understanding of preprocessors grow organically.

**4\. Event Handling**

**What are Events?**

An event in programming refers to a specific occurrence or happening that takes place during the execution of a program. Events can be triggered by various actions, such as user interactions, and changes in system states. In the context of MQL5, events are crucial for developing algorithmic trading scripts that respond dynamically to market conditions.

**What are Event Handlers?**

An event handler is a designated section of code that is responsible for responding to a specific type of event. In MQL5, event handlers are functions designed to execute when a particular event occurs. These functions are predefined and serve as the designated response mechanisms for various events. Each type of event has its corresponding event handler function.

**Analogy**

Imagine you're at a magical puppet show, and the puppets move and talk whenever certain things happen, like when the audience claps or when a special button is pressed. Now, think of the puppets as different parts of a computer program, and the magical button as an event. An event handler is like the puppet master behind the scenes, waiting for those specific moments (events) to occur. When an event happens, the event handler springs into action, making the program do something special, just like the puppet master making the puppets dance or talk when the audience cheers or a button is pressed. So, in the world of programming, an event handler is like the magical puppet master that brings things to life when certain events take place!

**4.1. Types of Event Handlers**

![Figure 4. Some Types of Event Handlers](https://c.mql5.com/2/65/event.png)

**4.1.1. OnInit**

In MQL5, “OnInit” is a special function used in Expert Advisors (EAs) to initialize the EA when it is first loaded onto a chart.

**Analogy:**

Alright, let's imagine that you have a magical robot friend. Before the robot starts doing anything exciting, like moving around or making funny sounds, it needs to get ready. The "getting ready" part is like the robot's “OnInit” moment. So, when you first switch on the robot, it goes into a special room (the OnInit function) where it prepares itself. This is where it sets up its favorite colors, decides how fast it should move, and makes sure everything is just right. Once it's all setup, the robot is ready to come out and start doing its cool tricks, like dancing or telling jokes.

In computer programs, the “OnInit” function works in a similar way. It's a special room where the program gets ready before it starts doing its tasks. It's like the opening ceremony for the program, ensuring everything is in place and ready to roll!

**Examples:**

```
// Declare global variables
double LotSize;
int TakeProfit;
int StopLoss;
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
// OnInit function
int OnInit()
  {
// Set EA parameters
   LotSize = 0.1;
   TakeProfit = 50;
   StopLoss = 30;

// Display initialization message
   Print("EA is getting ready. Lot Size: ", LotSize, ", Take Profit: ", TakeProfit, ", Stop Loss: ", StopLoss);

   return(INIT_SUCCEEDED);
  }
```

**Explanation:**

- We declare some global variables that the EA might use.

- The OnInit function is where we initialize these variables and perform any necessary setup. In this example, we set values for “LotSize”, “TakeProfit”, and “StopLoss”.

- The “Print” statement is like a message that the EA can send to the console to inform us about its initialization. It's like the robot saying, "I'm getting ready, and here are my settings."


When you attach this EA to a chart, the “OnInit” function runs once, and the console displays the initialization message. This ensures that the EA is ready with its settings before it starts trading or performing other actions.

**4.1.2. OnStart**

In MQL5, the “OnStart” function is a critical part of scripts and Expert Advisors (EAs). Its primary role is to execute commands only once when the script is activated or started. It's the initial entry point for the script's execution. In the case of a script, the “OnStart” function runs its defined logic, which may include placing trades, performing calculations, or other actions. However, unlike an Expert Advisor (EA) that continuously runs and re-evaluates conditions in the “OnStart” function, a script typically executes its logic just once and then stops.

So, if you have trade-related actions in the “OnStart” function of an MQL5 script, those actions will be carried out when the script is activated, but the script won't continually monitor the market or execute additional trades.

**Example:**

```
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
// This script prints a message to the console when activated

// Entry point when the script is started
void OnStart()
  {
// Print a message to the console
   Print("Script Activated! Hello, Traders!");
  }
```

**Explanation:**

- The “OnStart” function is the entry point of the script **.**

- The “Print” function is used to display a message in the "Experts" tab of the MetaTrader terminal's "Terminal" window.

- When you attach this script to a chart or run it in the MetaEditor, you'll see the specified message in the console. Feel free to modify the Print statement and add more complex logic based on your script's requirements.


When this script is attached to a chart, the “OnStart” function will run once. The purpose of the “OnStart” function in a script is typically to perform initialization tasks or execute certain actions at the moment the script is attached to the chart. After the script runs its logic, it completes its operation.

**4.1.3. OnTick**

In MQL5, the “OnTick” function is a crucial part of Expert Advisors (EAs). Its primary role is to contain the core trading logic and actions that should be executed on each tick of the market. EAs use “OnTick” to monitor price changes, analyze market conditions, and make decisions based on predefined trading strategies. Order-related actions, such as opening, modifying, or closing positions, are commonly placed within the “OnTick” function.

**Analogy**

Think of the “OnTick” function as your trading friend, TickTrader, exploring a busy market. Every tick is like a new chance or change. TickTrader is always looking for good trades, just like when you search for the best deals in a lively market.

When the market is calm, TickTrader may take it slow, just looking around without quickly buying or selling. Similarly, if the “OnTick” function sees things staying calm with every tick, it might suggest being careful, like casually looking around in the market. If prices suddenly go up, TickTrader might see a great deal and decide to make a purchase. Likewise, when the “OnTick” function notices a big price change with every tick, it may suggest taking advantage of good opportunities, just like getting great deals in the busy market.

In this lively market scene, TickTrader is always ready for the next opportunity, making decisions as the ticks keep coming. Similarly, the “OnTick” function works in real-time, adjusting to each tick and guiding trading actions in the dynamic market.

**Example:**

```
// Declare a variable to store the last tick's close price
double lastClose;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
// OnInit function
int OnInit()
  {

// Initialize the variable with the current close price
   lastClose = iClose(_Symbol, PERIOD_CURRENT, 0);

   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {

// Get the current close price
   double currentClose = iClose(_Symbol, PERIOD_CURRENT, 0);

// Check if the close price has changed
   if(currentClose != lastClose)
     {
      // Print a message when the close price changes
      Print("Close price changed! New close price: ", currentClose);

      // Update the last close price
      lastClose = currentClose;
     }

  }
```

**Explanation:**

**Variable Declaration:**

```
// Declare a variable to store the last tick's close price
double lastClose;
```

We declared a variable “lastClose” of type “double” to store the close price of the last tick.

**Initialization (OnInit function):**

```
nt OnInit()
  {

// Initialize the variable with the current close price
   lastClose = iClose(_Symbol, PERIOD_CURRENT, 0);

   return(INIT_SUCCEEDED);
  }
```

In the “OnInit” function, we initialize “lastClose” with the close price of the most recent candle using the “iClose” function. The parameters “\_Symbol”, “PERIOD\_CURRENT”, and “0” indicate the current symbol, the current timeframe, and the most recent candle, respectively **.**

**OnTick Function:**

```
void OnTick()
  {

// Get the current close price
   double currentClose = iClose(_Symbol, PERIOD_CURRENT, 0);

// Check if the close price has changed
   if(currentClose != lastClose)
     {
      // Print a message when the close price changes
      Print("Close price changed! New close price: ", currentClose);

      // Update the last close price
      lastClose = currentClose;
     }

  }
```

- In the “OnTick” function, we get the current close price using “iClose” and store it in the “currentClose” variable.

- We then check if the current close price is different from the last recorded close price (“currentClose != lastClose”).

- If there is a change, we print a message indicating the change and update the “lastClose” variable.


This code essentially monitors and prints a message whenever the close price changes on each tick. It showcases how the “OnTick” function can be used to respond to market dynamics in real-time.

**Conclusion**

In this article, we delved into the fundamental aspects of MQL5, exploring the concepts of arrays, custom functions, preprocessors, and event handling. I encourage you, dear readers, to embrace the learning journey and feel free to ask any questions that may arise. Your understanding of these foundational elements will pave the way for our upcoming discussions on creating powerful trading bots. Remember, knowledge grows with curiosity, and your questions are the seeds of a deeper understanding. Stay tuned for the next installments, where we'll embark on the exciting journey of building trading bots together. For now, take the time to familiarize yourself with these core concepts. Happy coding, and may your trading endeavors be as prosperous as your quest for knowledge!

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

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/461316)**
(6)


![Israel Pelumi Abioye](https://c.mql5.com/avatar/2023/11/6554a830-8858.png)

**[Israel Pelumi Abioye](https://www.mql5.com/en/users/13467913)**
\|
29 Jan 2024 at 14:28

**Oluwatosin Babalola [#](https://www.mql5.com/en/forum/461316#comment_51948946):**

Thank you for this wonderful beginner friendly article on Arrays. Kindly include beginner friendly explanation on structures in mql5 in your upcoming articles. Looking forward to that! Thank you!

Hello Oluwatosin,

Thank you so much for your kind words! I'm thrilled to hear that you enjoyed the article. Your request has been noted.

![leonroug2705](https://c.mql5.com/avatar/avatar_na2.png)

**[leonroug2705](https://www.mql5.com/en/users/leonroug2705)**
\|
22 Sep 2024 at 18:55

Thank you for being so kind about explaining MQL.  Many authors purposely write to confuse as to get you into a subscription plan.  Ever since QUE publishing was brought I have not been able to find a company that publish non bias books where anyone with a high school diploma can understand. Someone brought QUE and shut it down.  I was planning on leaning a programming language and publishing a book that's well explained and very understandable.  Thank You again, your time and effort are appreciated.


![Israel Pelumi Abioye](https://c.mql5.com/avatar/2023/11/6554a830-8858.png)

**[Israel Pelumi Abioye](https://www.mql5.com/en/users/13467913)**
\|
22 Sep 2024 at 20:31

**leonroug2705 [#](https://www.mql5.com/en/forum/461316#comment_54640707):**

Thank you for being so kind about explaining MQL.  Many authors purposely write to confuse as to get you into a subscription plan.  Ever since QUE publishing was brought I have not been able to find a company that publish non bias books where anyone with a high school diploma can understand. Someone brought QUE and shut it down.  I was planning on leaning a programming language and publishing a book that's well explained and very understandable.  Thank You again, your time and effort are appreciated.

You’re welcome. Thank you for your kind words.


![Ben Ho](https://c.mql5.com/avatar/2018/7/5B3F3C85-C0CA.png)

**[Ben Ho](https://www.mql5.com/en/users/bentlho)**
\|
5 Nov 2024 at 11:43

is it possible to code a price action strategy? i come from a [manual trading](https://www.mql5.com/en/articles/5268 "Article: Reversal: formalizing the entry point and writing a manual trading algorithm ") background so i focused on candle sticks reading.


![Ben Ho](https://c.mql5.com/avatar/2018/7/5B3F3C85-C0CA.png)

**[Ben Ho](https://www.mql5.com/en/users/bentlho)**
\|
7 Nov 2024 at 15:13

in the last bit " This code essentially monitors and prints a message whenever the close [price changes](https://www.mql5.com/en/docs/constants/errorswarnings/enum_trade_return_codes "MQL5 Documentation: Prices have changed") on each tick. It showcases how the “OnTick” function can be used to respond to market dynamics in real-time."

why do you put a if statement to compare the currentclose and last close?

why don't just Print currentclose price ? Wouldn't that be straight forward to get the job done?

![Data Science and Machine Learning (Part 19): Supercharge Your AI models with AdaBoost](https://c.mql5.com/2/65/Data_Science_and_Machine_Learning_4Part_19y_Supercharge_Your_AI_models_with_AdaBoost___LOGO.png)[Data Science and Machine Learning (Part 19): Supercharge Your AI models with AdaBoost](https://www.mql5.com/en/articles/14034)

AdaBoost, a powerful boosting algorithm designed to elevate the performance of your AI models. AdaBoost, short for Adaptive Boosting, is a sophisticated ensemble learning technique that seamlessly integrates weak learners, enhancing their collective predictive strength.

![ALGLIB numerical analysis library in MQL5](https://c.mql5.com/2/58/ALGLIB_in_MQL5_avatar.png)[ALGLIB numerical analysis library in MQL5](https://www.mql5.com/en/articles/13289)

The article takes a quick look at the ALGLIB 3.19 numerical analysis library, its applications and new algorithms that can improve the efficiency of financial data analysis.

![Ready-made templates for including indicators to Expert Advisors (Part 2): Volume and Bill Williams indicators](https://c.mql5.com/2/58/Volume_Bill_Williams_indicators_avatar.png)[Ready-made templates for including indicators to Expert Advisors (Part 2): Volume and Bill Williams indicators](https://www.mql5.com/en/articles/13277)

In this article, we will look at standard indicators of the Volume and Bill Williams' indicators category. We will create ready-to-use templates for indicator use in EAs - declaring and setting parameters, indicator initialization and deinitialization, as well as receiving data and signals from indicator buffers in EAs.

![How to create a simple Multi-Currency Expert Advisor using MQL5 (Part 6): Two RSI indicators cross each other's lines](https://c.mql5.com/2/64/rj-article-image-60x60.png)[How to create a simple Multi-Currency Expert Advisor using MQL5 (Part 6): Two RSI indicators cross each other's lines](https://www.mql5.com/en/articles/14051)

The multi-currency expert advisor in this article is an expert advisor or trading robot that uses two RSI indicators with crossing lines, the Fast RSI which crosses with the Slow RSI.

[![](https://www.mql5.com/ff/si/dwquj7nmuxsb297n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F994%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dhow.use.vps%26utm_content%3Drent.vps%26utm_campaign%3D0622.MQL5.com.Internal&a=enhudadyvnrfwcvutcjazdvrxjyrzhyf&s=8f8a773cbff7e7ca26346dfb885f4f329a8b1f2c99472f858f32c0b06b662998&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=fxjyufaodxypyopfwinjofhuxdguuhmh&ssn=1769192712866635713&ssn_dr=0&ssn_sr=0&fv_date=1769192712&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F14099&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Introduction%20to%20MQL5%20(Part%203)%3A%20Mastering%20the%20Core%20Elements%20of%20MQL5%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176919271235285503&fz_uniq=5071818878947897096&sv=2552)

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