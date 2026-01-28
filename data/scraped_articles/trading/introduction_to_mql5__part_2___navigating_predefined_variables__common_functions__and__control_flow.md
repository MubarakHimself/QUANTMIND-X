---
title: Introduction to MQL5 (Part 2): Navigating Predefined Variables, Common Functions, and  Control Flow Statements
url: https://www.mql5.com/en/articles/13997
categories: Trading, Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T18:03:50.878200
---

[![](https://www.mql5.com/ff/sh/rvgkjnsrvj1mzh89z2/01.png)Best VPS for tradersTwo-click launch from MetaTrader, minimum ping to broker, 15 USD/monthLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=wpjhvzsogglsviotmypjoyhhtuxlrzhi&s=aa6c5782a1658c2f617954d478dea9989a27ae26ecabc09d0ab1204277fdf8e3&uid=&ref=https://www.mql5.com/en/articles/13997&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5069003678209212372)

MetaTrader 5 / Examples


**Introduction**

Welcome back to our MQL5 journey! In [Part One](https://www.mql5.com/en/articles/13738), we embarked on the adventure of algorithmic trading, breaking down the complexities of MQL5  for beginners without prior programming experience. As we step into Part Two, the excitement continues as we delve even deeper into the essential building blocks of MQL5. Our goal is simple yet profound: to ensure everyone, regardless of their programming background, feels the embrace of understanding. Feel free to ask any questions, and let's unravel the intricacies of MQL5 together. Let's forge a community where every voice is heard and every journey in algorithmic trading is shared.

These articles stand out as beacons of simplicity. Embracing a unique approach, they unravel the intricacies of MQL5 basics with a focus on clarity, ensuring that anyone, regardless of their background, can embark on the enchanting journey of algorithmic trading.  This isn't just an article, it's a voyage into the extraordinary. Explore the world of MQL5 in a way that's uniquely immersive and captivating. Welcome to a new era of understanding where each article is not just read but experienced.

We'll unravel the mysteries of predefined variables, common functions, control flow statements, arithmetic operations, logical operations, and relational operations. As we progress through these Articles, the spotlight turns to creating Expert Advisors, the powerhouse of automated trading. You'll learn how to breathe life into your trading strategies, empowering your scripts to make decisions on your behalf.

And, of course, what good is a strategy without testing its mettle? The Strategy Tester takes center stage, allowing you to assess the robustness of your algorithms against historical data. It's where theory meets reality, and I'll guide you through the intricacies of this vital aspect of algorithmic trading. So buckle up, fellow enthusiasts! Part Two is a voyage into the heart of MQL5. Get ready for an immersive learning experience designed to equip you with the skills to navigate the fascinating world of algorithmic trading with MQL5 confidently.

In this article, we will cover the following topics:

- Predefined variables

- Common Functions

- Arithmetic, Relational, and Logical Operations

- Control Flow Statements


Before we venture into the depths of Part Two, stay tuned for an upcoming video where I'll provide a concise summary of the key concepts discussed in the first article. This video will serve as a quick refresher, ensuring we're all on the same page as we delve into more advanced MQL5 topics. Get ready for an insightful recap!

Intoduction to MQL5 Video 1 - YouTube

[Photo image of MQL5.community](https://www.youtube.com/channel/UC8bwZMk2yh5WDToFos49OjQ?embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F13997)

MQL5.community

1.91K subscribers

[Intoduction to MQL5 Video 1](https://www.youtube.com/watch?v=7MOMcTn7YII)

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

[Watch on](https://www.youtube.com/watch?v=7MOMcTn7YII&embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F13997)

0:00

0:00 / 5:43

•Live

•

**1\. Predefined Variables**

**What are Predefined Variables?**

In programming, variables are like containers that hold information. Predefined variables, as the name suggests, are special containers that are already defined by the programming language itself. These containers are filled with specific information about the current state of the program, trading environment, or market conditions.

**Why are they "Predefined"?**

They are called "predefined" because the programming language (in this case, MQL5) has already set them up and given them specific meanings. You don't need to create or declare these variables, they're ready to use whenever you need them. When you start an MQL5 program (like an Expert Advisor, script, or custom indicator), a special set of predefined variables comes into play. These variables already contain information about what the market looks like at that very moment. Details like prices, symbols, and time. They serve as a snapshot of the market conditions at the beginning of the program's execution, providing essential data for making decisions within the program.

**Analogy**

let's imagine predefined variables as magic boxes in a wizard's workshop. These magical boxes already have special powers, and the wizard gives them names to make it easy to understand what each box does. Now, imagine there's a special box called "\_Symbol." This magical box knows the name of the financial symbols (EURUSD, GBPUSD), like the name of a magical creature.

Another box, called "\_Period," knows the time period (Timeframe), almost like a special clock. These boxes are predefined because the wizard has already set them up with these powers before you even start your magical journey. So, when you, the young wizard, want to create a spell (or a program in our magical world), you can use these predefined boxes to quickly get important information like the name of a creature or the time of day.

You don't have to spend time creating these boxes yourself. They're already there, ready to help you perform your magical tasks! In the magical land of coding, predefined variables are like ready to use magic boxes that hold important information, making it easier for wizards (programmers) to create their spells (programs) without starting from scratch.

**1.1. Examples of Predefined Variables in MQL5**

**1.1.1. \_Symbol**

In the world of coding for finance, "\_Symbol" is like a magic wand that helps the program know exactly which financial instrument it's dealing with. It's like having a built-in GPS for trading symbols. Imagine a big store with lots of different items for sale, each item has its name, like "EURUSD" or "GBPJPY." "\_Symbol" is like a friendly assistant that quietly tells the program which item (symbol) it's looking at, without needing constant reminders.

For folks writing code to make trading decisions, knowing the specific item being traded is crucial. "\_Symbol" makes this easy, it's like having a helper who always knows which item is on the trading table. This helper ensures that the code can smoothly adapt to different items without needing constant manual adjustments.So, in simple terms, "\_Symbol" is the code's helpful companion, making sure it always knows which financial item it's working with. It's a handy tool for creating smart and flexible code that can handle the ever-changing world of trading symbols effortlessly.

**Example:**

```
void OnStart()
  {

   Alert(_Symbol);

  }
```

In this example, the ‘Alert’ function uses the "\_Symbol" predefined variable to fetch the current trading symbol and then send an alert to the chart. When you run this code on a MetaTrader platform, it will display the current trading symbol.

**1.1.2. \_Period**

The ‘\_Period’ predefined variable in MQL5 is another powerful tool that provides information about the timeframe or period of the chart currently under consideration. It serves as a built-in reference to the timeframe on which the trading algorithm is operating.

**Example:**

```
void OnStart()
  {

// a sample Script that prints the current timeframe to the user
   Print(_Period);

  }
```

Similar to "\_Symbol," "\_Period" contributes to the adaptability of trading algorithms by allowing them to be aware of and respond to the timeframe on which they are deployed. It's like having a clock that always tells the program the timeframe it's working with.

_Note: Think of '\_Symbol' as your guide to the current trading item, and '\_Period' as your clock showing the timeframe. These two are just the beginning of the magic. There's more to discover, but for now, enjoy the simplicity they bring to our coding adventure._

**1.1.3. \_Digits**

In the context of MQL5, the "\_Digits" predefined variable provides practical information about the number of decimal places associated with the current trading instrument. This detail is particularly crucial when dealing with the concept of a pip in the world of currency trading. A pip, or "percentage in point," represents the smallest price move in the exchange rate of a currency pair. The number of decimal places influences the precision of price movements, and "\_Digits" offers a quick reference for traders to adapt their calculations accordingly.

Consider this analogous to having a ruler that specifies the precision of each measurement, where each digit in the "\_Digits" variable corresponds to a decimal place. For instance, if "\_Digits" is set to 4, it indicates a precision of four decimal places, common in many currency pairs.

**Example:**

```
void OnStart()
  {

// Use the "_Digits" predefined variable to get the number of decimal places
   int decimalPlaces = _Digits;
// Print the result to the console
   Print("Number of Decimal Places: ", decimalPlaces, " (Each movement is a pip!)")

  }
```

**1.1.4. \_Point**

In the fascinating domain of MQL5, "\_Point" stands as a crucial predefined variable, offering insight into the minimum price movement of the current trading instrument. It's essentially the smallest measurable increment in the price, often referred to as a pip in the world of currency trading. To grasp this concept, envision "\_Point" as your financial ruler, indicating the tiniest unit by which prices can fluctuate. The "\_Point" value varies across different financial instruments, providing traders with a precise understanding of how much the price can change with each movement. This is important for setting of SL and TP in programs. This is just to familiarize you with the concept, as time goes on more explanation will be provided.

**Example:**

```
void OnStart()
  {

// setting sl by subtracting 15 pips from a ask price of 1.2222
   double Ask = 1.2222; // current market price
   double sl = Ask - (_Point * 15);
   Alert(sl); // Result will be 1.2207

  }
```

In this article, we've delved into a subset of fundamental predefined variables in MQL5, namely "\_Symbol," "\_Digits," "\_Point," and "\_Period." These variables serve as foundational pillars, offering insights into the current trading item, its precision, the minimum price movement, and the timeframe under consideration.  It's crucial to emphasize the deliberate simplicity of our approach. The complexity of MQL5 encompasses a plethora of predefined variables, yet our focus on a select few aims to provide a clear and easily digestible introduction. In keeping things simple, we pave the way for a more gradual and nuanced exploration of the intricate landscape of algorithmic trading in future articles. Stay tuned for further insights as our journey unfolds.

**2\. Common Functions**

**What are Common Functions?**

In MQL5, common functions are essential tools that traders and developers use to automate various aspects of algorithmic trading. These functions serve specific purposes, from displaying alerts and printing messages for debugging to managing orders, closing positions, and retrieving account information. Think of these functions as the building blocks that empower traders to implement their strategies efficiently and make informed decisions. They streamline the coding process, enhance strategy execution, and contribute to a more dynamic and responsive trading experience.

**Analogy**

Let's imagine you're the captain of a magical ship, sailing through the seas of trading adventures. In this journey, you have special tools called "common functions." These tools help you with different tasks, just like how a map guides you on your sea voyage. They are like magical words you can use in MQL5 to tell your program what to do.

**2.1. Examples of Common Functions in MQL5**

**2.1.1. Alert**

It's like a magical alarm that tells you when something exciting or important is happening during your journey. Imagine it as a friendly parrot that squawks to get your attention. When you want to be notified of something important, you use the ‘Alert’ function to send an alert to the chart when your program is activated or a certain condition is met.

**Examples:**

```
void OnStart()
  {

   Alert("Discoveries Await!");

  }
```

![Figure 1. Result after running  code in MetaTrader5](https://c.mql5.com/2/63/Alert.png)

In the enchanting world of MQL5, coding feels like speaking a language that's remarkably close to human expression. Take, for instance, the "Alert" function. In the magical script of MQL5, using the "Alert" function is akin to telling the program, "Hey, send an alarm when something interesting happens!" It's like having a conversation with your trading assistant in a language that feels intuitive and familiar, making the coding journey both accessible and enchanting.

**2.1.2. Print**

the "Print" function serves as our wizard's quill, effortlessly translating our thoughts into the script. When we invoke the "Print" incantation, it's as if we're jotting down notes in a magical journal, capturing the essence of our trading adventures. It's a whimsical act of storytelling, where each line of code becomes a chapter in our wizard's diary, chronicling the enchanting discoveries and wonders of the trading world. With "Print," our magical words come to life, creating a narrative that unfolds with every mystical trade.

**Example:**

```
void OnStart()
  {

   Print("Discoveries Await!");

  }
```

![Figure 2. Result after running code in MetaTrader5](https://c.mql5.com/2/63/COCO.png)

**2.1.3. Comment**

The "Comment" function is our magical brushstroke, adding vivid annotations to our trading canvas. When we cast the "Comment" spell, it's as if we're leaving secret messages for ourselves and fellow wizards to unravel. This incantation allows us to inscribe notes and symbols directly on our charts, turning them into mystical scrolls that tell the story of our trading strategies. With "Comment," our charts become living manuscripts, adorned with the wisdom and insights of our magical journey through the financial realms. It is like a magic word you use to display a message on your chart.

**Example:**

```
 {

  Comment("I love MQL5!");

 }
```

![Figure 3. Result after running code in MetaTrader5](https://c.mql5.com/2/63/Comments.png)

**2.1.4. PlaySound**

The “PlaySound” function in MQL5 is a powerful tool that allows traders to incorporate auditory signals into their scripts and expert advisors. This function can be used to play sound files in various situations, providing traders with additional cues or alerts beyond visual elements. Let's say you have a sound file named "alert.wav" in the "Sounds" directory, and you want to play this sound when a specific condition is met in your trading script.

Example:

```
void OnStart()
  {

// Check a condition
   if(5>4)
     {
      // Play the sound
      PlaySound("alert.wav");
     }
  }
```

alert.wav: The name of the sound file to be played. The file should be located in the "Sounds" directory of the terminal.

As we wrap up our exploration of common functions in MQL5, it's crucial to recognize that what we've encountered here is merely a glimpse into the expansive repertoire of tools available in the realm of algorithmic trading. The functions discussed Alert, Print, Comment, and PlaySound are but a selection from a diverse array of capabilities.

**3\. Arithmetic, Logical, and Relational Operation**

**3.1. Arithmetic Operations**

Arithmetic operations are fundamental mathematical operations that play a crucial role in programming, including MQL5 for algorithmic trading. These operations include addition (+), subtraction (-), multiplication (\*), division (/), modulus (%), increment operator (++), and decrement operator(--). They enable developers to perform mathematical calculations within their code.

**Addition (+) and Subtraction (-)**

Addition and subtraction are fundamental arithmetic operations that involve combining or separating quantities.

**Example:**

```
void OnStart()
  {

   int num1 = 8;
   int num2 = 5;

//addition of values
   int sum = num1 + num2;
//subtraction of values
   int minus = num1 - num2;

   Alert("Sum = ",sum,", minus = ",minus);

  }
```

**Multiplication (\*) and Division (/)**

**Example:**

```
void OnStart()
  {
   double price = 50.25;
   int quantity = 10;
   double totalCost = price * quantity; // Multiplication
   double averagePrice = totalCost / quantity; // Division

   Alert("totalCost = ",totalCost,", averagePrice = ",averagePrice);
  }
```

**Modulus**

The modulus operator, represented by the “%” symbol, is a mathematical operation that returns the remainder of the division of one number by another. The modulus operator is particularly useful in programming for tasks such as iterating through a specific number of elements, checking for even or odd numbers, or ensuring that values stay within a defined range.

**Example:**

```
void OnStart()
  {

   int num1 = 10;
   int num2 = 3;
   int remainder = num1 % num2; // The variable 'remainder' now holds the value 1

  }
```

**Increment and Decrement Operator**

The “++” and “--“ operators are known as increment and decrement operators, respectively. They are unary operators, meaning they operate on a single operand.

**Example:**

```
void OnStart()
  {

   int x = 5;
   int y = 8;
   x++; // After this line, the value of 'x' is 6
   y--; // After this line, the value of 'y' is 7

  }
```

_Note: These operators are often used to conveniently update the value of variables in loops, counters, or other scenarios where you want to increase or decrease a value by a fixed amount._

**3.2. Logical Operations**

Logical operations in programming involve evaluating conditions or expressions that result in a Boolean value (either true or false ). These operations are fundamental for making decisions and controlling the flow of a program. Logical operators are frequently used in conditional statements ( if , else , switch ), loops, and other situations where you need to make decisions based on certain conditions. They are integral to creating expressive and flexible logic within programming.

In many programming languages, including MQL5, the convention is that ' 1' typically represents true , and ' 0  ' represents false in the context of Boolean logic.

The common logical operators include:

| Symbol | Operation | Example | Explanation |
| --- | --- | --- | --- |
| \|\| | logical OR | y > 2 \|\| y < 7 | The value of y is greater than 2 or less than 7 |
| ! | logical NOT | !y | If the operand is true , the NOT operator makes it false, |
| && | logical AND | y > 2 && y < 7 | The value of y is greater than two and less than 7 |

**3.3. Relational Operation**

Relational operations in programming involve comparing values to determine the relationship between them. These operations typically result in a Boolean value ( true or false ).

Common relational operators include:

| Symbol | Operation | Example |
| --- | --- | --- |
| > | Checks if the value on the left is greater than the value on the right | x > y |
| < | Checks if the value on the left is less than the value on the right | x < y |
| != | Checks if two values are not equal | x != y |
| == | Checks if two values are equal | x == y |
| >= | Checks if the value on the left is greater than or equal to the value on the right | x >= y |
| <= | Checks if the value on the left is less than or equal to the value on the | x <= y |

_Note:  The equal sign ( = ) and double equal sign ( ==_ _) serve different purposes in programming and are used in distinct contexts._

The single equal '=' sign is an assignment operator used to assign a value to a variable.

**Example**:

```
int x = 10; // Assigns the value 10 to the variable x
```

The double equal '==' sign is a relational operator used for comparison to check for equality between two values.

**Example:**

```
int a = 5;
int b = 7;
bool isEqual = (a == b); // Checks if 'a' is equal to 'b'; 'isEqual' is false
```

**4\. Control Flow Statements**

In the vast landscape of programming, control flow statements emerge as the navigational compass, charting the course for a program's execution. Imagine your code as a ship sailing through the intricate waters of commands and logic. In this dynamic journey, control flow statements act as the captain's compass, steering the vessel through decision points, loops, and varied conditions.

Much like a ship's captain navigates turbulent seas, a programmer employs control flow statements to dictate the direction the code should take. Whether it's making decisions based on specific conditions, iterating through data, or executing certain actions repeatedly, these statements are the strategist's tools for orchestrating a purposeful voyage through the complexities of algorithmic logic

**How does it work?**

Imagine you have a magical pet, let's call it "Codey the Dragon." Codey loves following your instructions, but you need a way to tell Codey what to do in different situations. Here's where your magical wand (code) and special words (control flow statements) come into play.

**4.1. If/Else Statement**

You say to Codey, "If the sun is shining, fly high in the sky; else, stay on the ground and take a nap." It's like saying, "If something happens, do this; otherwise, do that."

**Example:**

```
 if(sun_is_shining)
     { Codey_fly_high_in_the_sky() }
 else
     { Codey_take_a_nap_on_the_ground() }
```

**4.2. While Loop**

The "while" loop is a powerful tool for performing repetitive tasks, iterating through sequences, and enchanting your code with dynamic, iterative magic. It allows your MQL5 programs to adapt and respond dynamically to changing conditions, making your trading strategies flexible and resilient in the ever shifting market landscape. It's more like saying to the computer to keep executing a command while a condition is true.

**Syntax:**

while (condition) { // Code to be executed while the condition is true }

**Example:**

```
void OnTick()
  {
// Initialization
   int numberOfTrades = 0;
// While loop condition
   while(numberOfTrades < 5)
     {
      // Magic within the loop
      Print("You can take a trade ", numberOfTrades + 1);
      // Counter increment
      numberOfTrades ++;
     }
// Magical finale
   Print("Trade complete!");
  }
```

_Note: More explanation to come on OnTick function later._

The "while" loop is like a magical construct that repeats a block of code as long as a specified condition remains true. It provides a way to perform iterative tasks until a certain condition is no longer met.

Let's break down the elements of the example:

**Initialization:**  int numberOfTrades = 0;

- We initialize a counter variable to keep track of our magical iterations

**While Loop Condition:**  while (numberOfTrades < 5)

- The loop checks whether the numberOfTrades is less than 5. If true, the code inside the loop executes. This condition acts as a magical   gatekeeper, allowing the loop to continue as long as the numberOfTrades is below 5.

**Magic Within the Loop:** Print("You can take a trade ", numberOfTrades + 1);

- Inside the loop, we perform some magical action. In this case, we're printing a message while numberOfTrades < 5.

**Counter Increment:** numberOfTrades++

- The numberOfTrades is incremented with each iteration, moving us closer to our magical goal which is 5.


**Loop Termination:** Print("Trade complete!");

- Once the loop has finished its iterations, The message is printed.


**4.3. For loop**

The "for" loop is a powerful enchantment that provides a concise way to perform iterative tasks. It's especially handy when you know the number of iterations you want to perform in advance. you can use for loop for almost everything you can use while loop for, but for loop has some advantages.

**Advantages:**

- **Conciseness:** The "for" loop condenses the initialization, condition check, and iteration statement into a single line, making your code more elegant.

- **Readability:** It enhances the clarity of your code, making it easier to understand at a glance.

- **Controlled Iteration:** With explicit control over the loop variable, you can precisely define the number of iterations.


**Syntax:**

for (initialization; condition; iteration) { // Code to be executed in each iteration }

**Example:**

```
void OnTick()
  {
// For loop to iterate from 1 to 5

   for(int i = 1; i <= 5; i++)
     {
      // Magic within the loop
      //i represents number of trades
      Print("You can take a trade ", i);
     }
// Magical finale
   Print("Trade complete!");

  }
```

**Explanation:**

**Initialization (int i = 1;):**

- We initialize a loop variable i to start our magical iterations from 1.


**Condition (i <= 5;):**

- The loop continues as long as the condition (i less than or equal to 5) holds true.


**Iteration (i++):**

- After each iteration, the loop variable i is incremented by 1.


**Magic Within the Loop:** Print("You can take a trade ", i);

- Inside the loop, we perform some magical action. In this case, we're printing a message as long as numberOfTrades < 5.

**Loop Finale:** Print("Trade complete!");

- Once the loop has finished its iterations, The message is printed.

**Conclusion**

As we conclude Part Two of our MQL5 series, we've navigated through the essentials. We covered predefined variables, common functions, and the intricate landscape of arithmetic, relational, and logical operations, along with control flow statements. What makes this journey exceptional is its simplicity and the invitation for interaction.  You are free to ask questions, turning learning into a dynamic exchange. As we close this chapter, anticipate the exciting next steps. Soon, we embark on the path of developing your own trading bot. Stay engaged, and let's continue this interactive journey towards mastering MQL5 and shaping your success in algorithmic trading.

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
**[Go to discussion](https://www.mql5.com/en/forum/460360)**
(4)


![Daniel Jose](https://c.mql5.com/avatar/2021/1/5FF740FF-26B8.jpg)

**[Daniel Jose](https://www.mql5.com/en/users/dj_tlog_831)**
\|
3 Apr 2024 at 10:27

Very good. But I missed you commenting on the ternary operator. ( ?: ) since it is quite practical at different times. And as in my codes I use it a lot. It would be nice if he was included. This way, people would be able to better understand the codes in my articles 🙂. Anyway, I liked your initiative to create an MQL5 mini course, in order to help with documentation. 😁👍


![Israel Pelumi Abioye](https://c.mql5.com/avatar/2023/11/6554a830-8858.png)

**[Israel Pelumi Abioye](https://www.mql5.com/en/users/13467913)**
\|
3 Apr 2024 at 15:30

**Daniel Jose [#](https://www.mql5.com/en/forum/460360#comment_52924154):**

Very good. But I missed you commenting on the ternary operator. ( ?: ) since it is quite practical at different times. And as in my codes I use it a lot. It would be nice if he was included. This way, people would be able to better understand the codes in my articles 🙂. Anyway, I liked your initiative to create an MQL5 mini course, in order to help with documentation. 😁👍

Hello Daniel,

Thank you for your input! I appreciate your suggestion regarding the ternary operator (?:). I'll include it in future articles to enhance code comprehension. Your feedback is valuable and contributes to improving our resources for the community.

![DHess10000](https://c.mql5.com/avatar/2024/5/6640e8eb-44c5.jpg)

**[DHess10000](https://www.mql5.com/en/users/dhess10000)**
\|
12 May 2024 at 15:56

Israel, I like your analogies and how you describe "making the magic" happen.  Thank you.


![Israel Pelumi Abioye](https://c.mql5.com/avatar/2023/11/6554a830-8858.png)

**[Israel Pelumi Abioye](https://www.mql5.com/en/users/13467913)**
\|
12 May 2024 at 17:37

**DHess10000 [#](https://www.mql5.com/en/forum/460360#comment_53344011):**

Israel, I like your analogies and how you describe "making the magic" happen.  Thank you.

Hello DHess,

You’re welcome.

![Data Science and Machine Learning (Part 18): The battle of Mastering Market Complexity, Truncated SVD Versus NMF](https://c.mql5.com/2/64/Data_Science_and_Machine_Learning_pPart_183_Truncated_SVD_Versus_NMF__LOGO.png)[Data Science and Machine Learning (Part 18): The battle of Mastering Market Complexity, Truncated SVD Versus NMF](https://www.mql5.com/en/articles/13968)

Truncated Singular Value Decomposition (SVD) and Non-Negative Matrix Factorization (NMF) are dimensionality reduction techniques. They both play significant roles in shaping data-driven trading strategies. Discover the art of dimensionality reduction, unraveling insights, and optimizing quantitative analyses for an informed approach to navigating the intricacies of financial markets.

![Developing an MQTT client for Metatrader 5: a TDD approach — Part 5](https://c.mql5.com/2/64/Developing_an_MQTT_client_for_Metatrader_5___Part_5___LOGO__1.png)[Developing an MQTT client for Metatrader 5: a TDD approach — Part 5](https://www.mql5.com/en/articles/13998)

This article is the fifth part of a series describing our development steps of a native MQL5 client for the MQTT 5.0 protocol. In this part we describe the structure of PUBLISH packets, how we are setting their Publish Flags, encoding Topic Name(s) strings, and setting Packet Identifier(s) when required.

![Ready-made templates for including indicators to Expert Advisors (Part 1): Oscillators](https://c.mql5.com/2/57/ready_made_templates_for_connecting_indicators_001_avatar.png)[Ready-made templates for including indicators to Expert Advisors (Part 1): Oscillators](https://www.mql5.com/en/articles/13244)

The article considers standard indicators from the oscillator category. We will create ready-to-use templates for their use in EAs - declaring and setting parameters, indicator initialization and deinitialization, as well as receiving data and signals from indicator buffers in EAs.

![Data label for time series mining (Part 5)：Apply and Test in EA Using Socket](https://c.mql5.com/2/64/Data_label_for_time_series_miningbPart_50_Apply_and_Test_in_EA_Using_Socket_____LOGO.png)[Data label for time series mining (Part 5)：Apply and Test in EA Using Socket](https://www.mql5.com/en/articles/13254)

This series of articles introduces several time series labeling methods, which can create data that meets most artificial intelligence models, and targeted data labeling according to needs can make the trained artificial intelligence model more in line with the expected design, improve the accuracy of our model, and even help the model make a qualitative leap!

[![](https://www.mql5.com/ff/sh/bhdtjfb1zry09943z2/267b575d2182c180804d340af38ce02c.jpg)\\
Trade from your iPhone or Android device\\
\\
You only need an internet connection to use the new powerful MetaTrader 5 Web terminal\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=wtigumvtenarnsocpyfoqnanxrilnbxx&s=ec8c539e52b83881ff2d16eaff6913b25803952eb277cac55f670a102b2edc1f&uid=&ref=https://www.mql5.com/en/articles/13997&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5069003678209212372)

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