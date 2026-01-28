---
title: Introduction to MQL5 (Part 1): A Beginner's Guide into Algorithmic Trading
url: https://www.mql5.com/en/articles/13738
categories: Expert Advisors
relevance_score: 9
scraped_at: 2026-01-22T17:42:37.062828
---

[![](https://www.mql5.com/ff/sh/zf7a2k61x98jzh89z2/01.png)Speed up your tradingUse our high-speed VPS for MetaTrader 4 and 5Learn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=qtrrsuiwuicrscmckjynyanztbditglq&s=c617dc80d90cfd3783ec1345eec2b419b281f10fec6eac77b3218984ac337259&uid=&ref=https://www.mql5.com/en/articles/13738&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5049320525991356763)

MetaTrader 5 / Tester


### Introduction

Without any prior programming experience, learning MQL5 can be difficult but not impossible. Understanding MQL5, a specialized language created for algorithmic trading, necessitates having both programming and financial market expertise. In my upcoming articles, I aim to bridge the gap for individuals without a programming background who are interested in learning MQL5, the content will be tailored to breaking down programming concepts and make MQL5 accessible to beginners.

I'll provide a solid foundation in programming fundamentals, breaking down concepts like variables, data types, loops, and conditional statements in a beginner-friendly manner. The articles will take a step-by-step approach, guiding readers through the basics of MQL5. I'll start with simple scripts and gradually introduce more complex topics, ensuring a smooth learning curve.

Learning is often more effective when accompanied by practical examples. I'll include real-world examples and scenarios to illustrate how MQL5 is applied in algorithmic trading. To enhance understanding, the articles will encourage interactive learning. Readers will be encouraged to practice writing code, experiment with examples, and apply their knowledge in a hands-on manner. In essence, these articles are designed to empower individuals without a programming background to confidently navigate the world of MQL5, fostering a gradual and enjoyable learning experience in the realm of algorithmic trading.

In this article, we will cover the following topics:

- Introduction to Programming

- Types of MQL5 Programs

- MetaEditor IDE

- MQL5 Language Basics

In the next articles, we will cover the following topics

- Predefined variable

- Common Functions

- Control Flow Statements

- Arrays and Loops

- Creating Expert Advisors

- Strategy Tester

**1. Introduction to Programming**

Have you ever wondered how your favorite games and software were built? How do your favorite game characters follow your commands? All these are possible because of programming, think of programming as a set of code written to make human and computer communication possible.

What is a programming language? The word “programming” means a process of setting rules or instructions to be followed to achieve a specific goal. On the other hand, the word “language” means a system of communication that enables individuals to express ideas, emotions, and thoughts to a computer. Computers do not understand human languages, how do we communicate our set of instructions to achieve our goals? Programing language serves as an intermediary between humans and computers by helping humans to communicate with the computer.

**1.1. Types of Programming Language**

Just as there are different human languages, there are also different programming languages, these languages are classified into two:

- **High-level programming language:** These are closer to human language and are easily relatable. Examples include MQL4, MQL5, Pythons, and JavaScript
- **Low-level programming language:** Closer to machine code and more challenging for humans to write and understand. These include Assembly language and are more closely related to the computer's hardware.

**Classification Based on Purpose**

- **General-Purpose programming languages:** These are designed to perform a wide range of functions. Examples include Pythons, Java, and C++

- **Domain-Specific Languages:** These are programming languages that perform a specific task. For instance, MQL5 is used for developing trading bots.

**1.3. Important Terminologies in Programming**

- **IDE (Integrated Development Environment):** A software application that provides comprehensive facilities to programmers for software development. It's a central platform where developers can write, test, debug, and deploy their software. Examples include Visual Studio Code, Code Blocks, and Meta Editor 5. As this article progresses, we’ll focus more on Meta Editor 5 which is the IDE used in MQL5.

- **Debugger:** This tool helps find and fix bugs in the code by allowing developers to step through the code, set breakpoints, and inspect variables during runtime.

- **Compiler/Interpreter:** Many IDEs come with built-in tools to compile and run code directly within the environment. They can execute the code, check for errors, and display the output

- **Data Types:** Different categories of data that a programming language can manipulate. Examples include integers, floats, strings, and booleans. As this article progresses, a section will be dedicated to explaining this.

- **Syntax:** The set of rules that define the combinations of symbols and words in a programming language. Every programming language has its own unique syntax that helps in the execution of codes.

- **Algorithm:** A set of well-defined instructions designed to solve a specific problem or perform a task.

- **Function:** A block of code that performs a specific task. It's reusable and can be called multiple times within a program.

- **Library:** A collection of pre-written code, functions, and routines that can be used to perform specific tasks, helping programmers avoid reinventing the wheel.

- **Conditional Statements:** Constructs like if-else or switch-case that allow the program to make decisions based on certain conditions. This means that if a condition is true the computer should execute a code.

- **Loop:** A control structure that repeats a block of code until a specific condition is met. Common types include for-loops, while-loops, and do-while loops. For example, the while loop will keep executing a block of code while a specific condition is met.

- **Variable:** A container used to store data. It has a name and can hold different values that can change during the program's execution.

_Note: As we progress in this article I will discuss more on some of these important terminologies._

**2\. MQL5 Programs**

MQL5 (MetaQuotes Language 5) is a high-level and domain-specific programming language. MQL5 is a domain-specific programming language because it only performs tasks related to trading. Examples include automating trading strategies, developing trading robots, and technical indicators.

In MQL5 programming language, various types of programs can be developed to automate trading, analyze markets, and enhance the overall trading experience. Here are some common types of MQL5 programs:

- **Expert Advisors (EAs):** EAs are the types of MQL5 programs that are used to automate trading strategies. It helps in the execution of trades based on predefined rules. EAs are bound to the chart and wait for a predefined event to occur no matter how long it takes before executing a command.
- **Scripts:** Scripts share similarities with EAs but they are different. Scripts are like one-time OTPs that execute a program immediately after they are activated, if the predefined conditions are met it will execute the program and if it doesn’t, it will be otherwise.

- **Custom Indicators:** Unlike EAs, custom indicators cannot execute trades, they are programs that indicate a particular event has occurred in the market based on your predefined rules.

- **Library:** A collection of pre-written code, functions, and routines that can be used to perform specific tasks, helping programmers avoid reinventing the wheel.

- **Include File:** In programming, an include file is a separate file that contains code or declarations that you want to reuse in multiple parts of your program. The ‘#include’ directive is used to include the content of another file in your main program file. Include files are commonly used to organize and modularize code, promoting code reuse and maintainability.


**3\. MetaEditor IDE**

MetaEditor 5 is the IDE used in the MQL5 programming language. An IDE is like an environment or page where you can write, debug, compile, and test your codes. MetaEditor 5 always comes with MetaTrader 5 when downloaded.

- To access Meta Editor, click on the IDE button in your MetaTrader 5.
![Figure 1. Opening MetaEditor from MetaTrade5](https://c.mql5.com/2/60/mte.png)

- The Navigator section (Ctrl+D) displays all the types of MQL5 programs. When you click on any of the programs, it shows all the codes you’ve written.
![Figure 2. The navigator section](https://c.mql5.com/2/60/navigator.png)

- To open the page for writing your codes, click the New (Ctrl+N) button, you need to choose the type of MQL5 program you want to develop, click the Next > button in the MQL Wizard section after choosing the type of program you want to develop.
![Figure 3. Choosing the type of MQL5 program to develop](https://c.mql5.com/2/60/type.png)

- The page to input your details will be displayed, click the Finish button in the MQL wizard section when you are done.
![Figure 4. Enter necessary information](https://c.mql5.com/2/60/details.png)

- Congratulations! You just have a fully ready Meta Editor to start writing your codes.
![Figure 5. MetaEditor 5](https://c.mql5.com/2/60/ready.png)


- This section displays the details you input earlier.
![Figure 6. Details.](https://c.mql5.com/2/60/properties.png)

- OnStart() is a function that is called in script programs, it ensures the codes in the curly bracket {} are executed once and immediately after the script program is activated.

![Figure 7. OnStart()](https://c.mql5.com/2/60/onstart.png)

- Let's write our first code, we are creating a script that alerts “Hello MQL5” when activated.


```
void OnStart()
    {

     Alert("Hello MQL5");

    }
```


_Note: Further explanation will be added as this article progresses_

- The compiler checks if the code is correct or not, it also makes suggestions on what to do to correct the code. When you are done writing your codes, click the Compile button and check the description section for errors and warnings.
![Figure 8. Cpmpiler.](https://c.mql5.com/2/60/compiler.png)


_Note: If the code has errors, the code will not run until you make corrections. The code can run without correcting the warnings but it’s better to address that to avoid issues with your program._

- To test the program, click the play button.
![Figure 9. Testing code.](https://c.mql5.com/2/60/play.png)

- The program then runs in your MetaTrader 5.

![Figure 10. Running code on MT5.](https://c.mql5.com/2/60/run_code.png)


So far, we've embarked on a brief yet crucial journey into the foundations of programming and the MetaEditor environment. We've laid the groundwork for understanding the language of algorithms, MQL5, and explored the creative space provided by MetaEditor. The MetaEditor is not just a tool; it's a gateway to transforming ideas into executable strategies. Stay tuned for the ahead, where we'll unravel the richness of MQL5, guiding you toward the mastery of programming for financial markets. Your adventure into algorithmic trading has just begun.

**4\. MQL5 Language Basics**

Before we proceed to develop MQL5 programs, it’s important to understand some basic things that make up a program. Examples include Data type, Functions, Variables, Comments, and MQL5 Syntax.

**4.1. Syntax**

Syntax refers to the set of rules that dictate how programs written in a particular language should be structured. It's like the grammar of a programming language. Just as human languages have rules for forming sentences, programming languages have rules for creating valid and understandable code.

Here are some key aspects of syntax:

**4.1.1. Statement Termination**

Each statement in MQL5 is typically terminated with a semicolon (‘;’). This informs the compiler that one statement has ended, and the next one is beginning.

**Example:**

```
void OnStart()
  {

   Alert("Hello MQL5");  // the semicolon means the end of the code line

  }
```

**4.1.2. Comments**

Comments in MQL5 are essential for providing explanations and documentation within your code. Think of comments like a tag you add to a line of code just to refer to it later or explain the purpose of a code. Here are detailed examples of how to use comments in various scenarios:

Single-Line Comments: Single-line comments are preceded by ‘//’. They are used for brief explanations.

Examples:

```
void OnStart()
  {

   Print("Hello MQL5");   // This line of code will print “Hello MQL5” when run

  }
```

_Note: The compiler completely ignores anything written after //_

**Multi-Line Comments**

Multi-line comments are enclosed within “/\* \*/”. They are suitable for more extensive explanations. This can be useful for writing out your plans or step-by-step guard to follow to develop a program.

Example:

```
void OnStart()
  {

   /* In many programming languages, including MQL5
    a semicolon “;” is used to indicate the end of a statement.
   It is a crucial element for the compiler to understand the structure of the program.
   */

  }
```

_The compiler completely ignores everything between the /\* and \*/_

Comments play a crucial role in making your code readable and understandable. They aid not only in your own understanding but also in anyone else who may read or collaborate on your code. Utilize comments generously to enhance the clarity of your MQL5 programs.

**4.1.3. Identifiers**

In programming, an identifier is a name given to a variable, function, or any other user-defined item in the code. Identifiers play a crucial role in making code readable and understandable. Here are some key points about identifiers in MQL5:

**Naming Rules**

Identifiers must follow specific naming rules. They should start with a letter (A-Z or a-z) and can be followed by letters, digits (0-9), or underscores (\_).

**Case Sensitivity**

MQL5 is case-sensitive, meaning uppercase and lowercase letters are distinct. Therefore, ‘myVariable’ and ‘MyVariable’ are considered different identifiers.

Example:

```
int myage = 25;
int MyAge = 25;  // myage is a different identifier from MyAge.
```

**Meaningful Names**

Choose meaningful and descriptive names for identifiers. This enhances code readability and makes it easier for others (or your future self) to understand the purpose of variables or functions.

**Reserved Words**

Avoid using reserved words as identifiers. Reserved words are words that have special meanings in the programming language and cannot be used for other purposes. Examples include Alert, Print, and Comment. Learn more about [reserved words.](https://www.mql5.com/en/docs/basis/syntax/reserved)

Example:

```
    int Age = 25;
    Print(Age);
    /*
    Int is the data type is an integer because 25 is a whole number.
    Age is the  identifier
    Print is a reserved word in MQL5 and it serves a special function. This means it can't be used as an identifier
    */
```

**4.2. Basic Data Types**

In programming, data types are like containers that hold different kinds of information. Imagine you have boxes to store things: some boxes can hold numbers, some can hold words, and some can hold more complex stuff. Examples include:

**Integer (int)**

Think of this as a box that only holds whole numbers. It can store numbers like 1, 5, or 10, but it can't store numbers with decimal points.

Example:

```
int myInteger = 10;
```

**Double**

This type represents numbers with decimal points.

Example:

```
double myDouble = 3.14;
```

**Character**

In MQL5, a character is a data type used to represent a single character, such as a letter, digit, or symbol. It is denoted by the 'char' keyword.

Example:

```
 // Declaring a character variable
      char myChar = 'A';

 // Printing the character to the console
      Print(myChar);
```

**String Type**

Strings are sequences of characters, typically used for text.

Example:

```
string myString = "Hello, MQL5!";

Alert(myString);
```

**Boolean Type**

Booleans represent true or false values.

Example:

```
bool iam25 = true;
```

**Arrays**

Arrays in MQL5 allow you to store multiple values of the same data type under a single variable name. They provide a convenient way to work with collections of data. Here are the key aspects of arrays in MQL5:

1\. **Array Declaration:** You declare an array by specifying its data type and name, followed by square brackets ‘\[\]’ indicating the array.

Example:

```
Integer array declaration

int numbers[5];
```

**2\. Initialization:** You can initialize an array at the time of declaration by providing a list of values enclosed in curly braces ‘{}’.

Example:

```
// Initializing an integer array

   int numbers[] = {1, 2, 3, 4, 5};
```

3\. Accessing Elements: Array elements are accessed using their index, starting from 0. For example, in the array ‘numbers’, ‘numbers\[0\]’ is the first element. Array in MQL5 can be used to get close prices or open prices of candle sticks.

Example:

```
// Accessing elements of an array

   int firstNumber = numbers[0];  // Accesses the first element
```

_Note: these are just a few data types and we only covered some basics. This is to guide you through each concept with clear examples, avoiding overwhelming details to make your learning journey enjoyable and easy._

### Conclusion

In conclusion, we've embarked on a journey exploring the foundations of programming and the specific world of MQL5. Starting with an introduction to programming, we delved into the types of MQL5 programs, understanding the significance of MetaEditor IDE in crafting our trading algorithms. Our exploration of MQL5 language basics has laid the groundwork for building more sophisticated programs. Remember, this is just the beginning, and as we progress, the power of coding with MQL5 will unfold, empowering you to create robust and efficient trading strategies. Stay tuned for more insights and articles into the exciting realm of algorithmic trading and MQL5 programming!

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
**[Go to discussion](https://www.mql5.com/en/forum/458325)**
(8)


![Israel Pelumi Abioye](https://c.mql5.com/avatar/2023/11/6554a830-8858.png)

**[Israel Pelumi Abioye](https://www.mql5.com/en/users/13467913)**
\|
6 Dec 2023 at 01:49

**JMBerlu [#](https://www.mql5.com/en/forum/458325#comment_50942027):**

Bravo! it is a very useful article for beginners. I am dying to read the next parts and I am sure I am not the only ones!

Thanks in advance for sharing your knowledges and the energy you will spend to do that!

Best regards,

JMB

Thank you so much, JMB! I'm thrilled to hear you found the article useful. Your encouragement means a lot, and I'm excited to share more insights in the upcoming parts. Stay tuned for more, and if you have any specific topics you'd like to see covered, feel free to let me know!

Best regards,

Abioye Israel Pelumi

![Tbor Yorgonson](https://c.mql5.com/avatar/2024/5/6640AF65-1A7E.png)

**[Tbor Yorgonson](https://www.mql5.com/en/users/tboryorgonson)**
\|
22 Aug 2024 at 02:41

Love your work mate thanks very much for the effort in these articles, really clears up the basics if you can't find what you are looking for or need context!


![Israel Pelumi Abioye](https://c.mql5.com/avatar/2023/11/6554a830-8858.png)

**[Israel Pelumi Abioye](https://www.mql5.com/en/users/13467913)**
\|
22 Aug 2024 at 03:13

**Tbor Yorgonson [#](https://www.mql5.com/en/forum/458325#comment_54365129):**

Love your work mate thanks very much for the effort in these articles, really clears up the basics if you can't find what you are looking for or need context!

Thank you for your kind words


![olenarvsv](https://c.mql5.com/avatar/2025/8/68a33100-af0f.jpg)

**[olenarvsv](https://www.mql5.com/en/users/olenarvsv)**
\|
20 Aug 2025 at 00:03

Thanks a lot for this article. On my week one studying MQL5, and this was really helpful! Looking forward to reading the series.


![Israel Pelumi Abioye](https://c.mql5.com/avatar/2023/11/6554a830-8858.png)

**[Israel Pelumi Abioye](https://www.mql5.com/en/users/13467913)**
\|
20 Aug 2025 at 08:44

**olenarvsv [#](https://www.mql5.com/en/forum/458325#comment_57845312):**

Thanks a lot for this article. On my week one studying MQL5, and this was really helpful! Looking forward to reading the series.

You're welcome

![Neural networks made easy (Part 53): Reward decomposition](https://c.mql5.com/2/57/decomposition_of_remuneration_053_avatar.png)[Neural networks made easy (Part 53): Reward decomposition](https://www.mql5.com/en/articles/13098)

We have already talked more than once about the importance of correctly selecting the reward function, which we use to stimulate the desired behavior of the Agent by adding rewards or penalties for individual actions. But the question remains open about the decryption of our signals by the Agent. In this article, we will talk about reward decomposition in terms of transmitting individual signals to the trained Agent.

![Developing a Replay System — Market simulation (Part 17): Ticks and more ticks (I)](https://c.mql5.com/2/55/replay-p17-avatar.png)[Developing a Replay System — Market simulation (Part 17): Ticks and more ticks (I)](https://www.mql5.com/en/articles/11106)

Here we will see how to implement something really interesting, but at the same time very difficult due to certain points that can be very confusing. The worst thing that can happen is that some traders who consider themselves professionals do not know anything about the importance of these concepts in the capital market. Well, although we focus here on programming, understanding some of the issues involved in market trading is paramount to what we are going to implement.

![MQL5 Wizard Techniques you should know (Part 08): Perceptrons](https://c.mql5.com/2/61/MQL5_Wizard_Techniques_you_should_know_xPart_08c_Perceptrons_LOGO.png)[MQL5 Wizard Techniques you should know (Part 08): Perceptrons](https://www.mql5.com/en/articles/13832)

Perceptrons, single hidden layer networks, can be a good segue for anyone familiar with basic automated trading and is looking to dip into neural networks. We take a step by step look at how this could be realized in a signal class assembly that is part of the MQL5 Wizard classes for expert advisors.

![Developing a Replay System — Market simulation (Part 16): New class system](https://c.mql5.com/2/55/replay-p16-avatar.png)[Developing a Replay System — Market simulation (Part 16): New class system](https://www.mql5.com/en/articles/11095)

We need to organize our work better. The code is growing, and if this is not done now, then it will become impossible. Let's divide and conquer. MQL5 allows the use of classes which will assist in implementing this task, but for this we need to have some knowledge about classes. Probably the thing that confuses beginners the most is inheritance. In this article, we will look at how to use these mechanisms in a practical and simple way.

[![](https://www.mql5.com/ff/si/q0vxp9pq0887p07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fvps%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Duse.vps%26utm_content%3Drent.vps%26utm_campaign%3D0622.MQL5.com.Internal&a=rktadgjlwhobyedohbrepzshvpcqrlpo&s=a93cef75a53eb5da24c98e0068b3c2b96015191a0af0d1857f5b4dd22e55e7bf&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=kvtoaxbvbaaefympbezhdiactixrxzad&ssn=1769092955771369281&ssn_dr=0&ssn_sr=0&fv_date=1769092955&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F13738&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Introduction%20to%20MQL5%20(Part%201)%3A%20A%20Beginner%27s%20Guide%20into%20Algorithmic%20Trading%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176909295589376912&fz_uniq=5049320525991356763&sv=2552)

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