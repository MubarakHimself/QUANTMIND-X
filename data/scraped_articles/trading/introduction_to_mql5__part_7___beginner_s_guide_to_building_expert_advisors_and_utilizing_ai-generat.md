---
title: Introduction to MQL5 (Part 7): Beginner's Guide to Building Expert Advisors and Utilizing AI-Generated Code in MQL5
url: https://www.mql5.com/en/articles/14651
categories: Trading, Indicators, Expert Advisors, Machine Learning
relevance_score: 9
scraped_at: 2026-01-22T17:29:46.900218
---

[![](https://www.mql5.com/ff/si/h2ryn394uwcpxwmxc2.jpg)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Feconomic-calendar%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dopen.calendar%26utm_content%3Deconomic.calendar%26utm_campaign%3Den.0009.desktop.default&a=qdeulxgvibgwytgewnvfatbocjnnninc&s=5c0c60f00ff5f5bedb0fdf65d9d79eb820442eb43ffac2b85aa003224f9dba14&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=hickysvfpiotmarezjywxcuxadqnymlk&ssn=1769092184282044433&ssn_dr=0&ssn_sr=0&fv_date=1769092184&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F14651&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Introduction%20to%20MQL5%20(Part%207)%3A%20Beginner%27s%20Guide%20to%20Building%20Expert%20Advisors%20and%20Utilizing%20AI-Generated%20Code%20in%20MQL5%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176909218446466686&fz_uniq=5049172250835396183&sv=2552)

MetaTrader 5 / Trading


### **Introduction**

Welcome back to our MQL5 journey! Prepare to dig in and explore the fascinating realm of real-world examples as we begin Part 7 of our MQL5 programming series. In this chapter, we're going to move from theory to practice, with practical exercises that help us put the ideas we've studied in earlier articles into practice.

Understanding the significance of pseudocode will be an essential component of our investigation. Pseudocode acts as a link between concrete code implementations and abstract algorithmic concepts. Before starting to code, it enables us to quickly and easily sketch out our ideas in a format that humans can read. In the upcoming sections, we'll look at the power of pseudocode and how it helps turn algorithmic strategies into workable code.

The rapid advancement of artificial intelligence in today's technological landscape has sparked discussions regarding the future of coding. Some argue that code produced by AI could replace the need for conventional coding knowledge. But nothing could be further from reality than this. This article seeks to dispel this myth and highlight the importance of knowing how to code. We're going to reveal ways you can incorporate AI-generated code into our programming projects in a seamless way so that it complements rather than replaces our skills. So, let's get started and utilize MQL5 programming fully without any delay. Prepare to apply theory to real-world situations, and let's work some magic!

In this article, we will cover the following topics:

- Mastering MQL5
- Pseudocode

- Generating MQL5 Codes with AI


In the previous article, I mentioned that I would include a video on array functions. This video will also serve as a reminder of where we left off.

Intoduction to MQL5 Video 5 Array Fuctions in MQL5 - YouTube

[Photo image of MQL5.community](https://www.youtube.com/channel/UC8bwZMk2yh5WDToFos49OjQ?embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F14651)

MQL5.community

1.91K subscribers

[Intoduction to MQL5 Video 5 Array Fuctions in MQL5](https://www.youtube.com/watch?v=-Y71d78249I)

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

[Watch on](https://www.youtube.com/watch?v=-Y71d78249I&embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F14651)

0:00

0:00 / 31:37

•Live

•

### **1\. Mastering MQL5**

Learning all the ins and outs of MQL5 at once might be overwhelming. Instead, focusing on projects can make the learning process more manageable and enjoyable. Begin by mastering fundamental concepts like variables, loops, conditions, functions, and all the other basic concepts discussed in previous articles. Once you have a solid grasp of these basics, start exploring projects that interest you, such as building **E** xpert **A** dvisors or custom indicators. Every project you work on will present fresh difficulties and teach you priceless lessons. Through project-based learning, you can put your theoretical knowledge to use in real-world scenarios, which helps you better grasp the complexities of MQL5. You will progressively acquire the knowledge and abilities required to handle the complexity of MQL5 programming as you work on a variety of projects.

Accepting this method of learning gives you the ability to advance as a programmer in a steady and self-assured manner. Instead of feeling intimidated by MQL5's vastness, you'll tackle it project by project, gradually increasing your proficiency. Therefore, accept the project-based learning journey; it is the most reliable way to master MQL5. I decided to use project-based learning throughout this series because of its effectiveness. Instead of throwing you headfirst into MQL5 complexities, we'll focus on practical projects that allow you to learn through experience. Working on real-world projects will help you better understand the topic and develop useful skills that will improve your trading strategies.

We will begin with basic, beginner-friendly projects and work our way up to more complex ones. By following this methodical approach, you can ensure that you have a solid understanding of MQL5 programming before advancing to more complex topics. You'll pick up new skills and ideas with each project, which will help you gradually improve. The project we will be working on for this article will act as a hands-on example of the sequential procedure for generating Expert Advisors in MQL5. We'll also look at optimizing the benefits of AI-generated code in the context of MQL5.

The beginner-friendly project we'll be working on is an alarm trading clock in MQL5. What sets this project apart is its use of a pre-recorded voice message instead of the standard sound provided by MQL5. You'll learn how to create an alarm system that allows users to set alerts using voice messages, adding a personalized touch to your trading experience.

The first step in creating an Expert Advisor (EA) in MQL5 is to list all the necessary components the EA will entail. To guarantee that the EA is created with a precise goal and plan in mind, this planning stage is essential. For the EA to analyze market data and make trading decisions, you should first decide on the trading strategy it will use. This includes specifying the entry and exit rules, risk management procedures, and any other special requirements.

Knowing the type of Expert Advisor (EA) you are working on is important when writing one. While some EAs are primarily concerned with carrying out trades using predetermined strategies, others have diverse functions like risk management, data monitoring, or even sending trade alerts without actually placing trades. Understanding the kind of EA you are building will direct your efforts and assist you in designing an EA that effectively and efficiently carries out its intended purpose. By the end of this article, readers will not only have a functional EA capable of sending voice alerts but also gain valuable insights into the process of EA development and the potential of AI-assisted trading strategies.

### **2\. Pseudocode**

**2.1. What is Pseudocode?**

Pseudocode is a way to define what you want to do before you start coding. You can specify your goals before you begin coding by using pseudocode. It's a means of outlining your goals for your program in plain, understandable language without worrying about programming language syntax. It makes it easier to write the actual code later on by assisting you in organizing the steps and logic your program will follow.

**Analogy**

Let's say you want to use a computer program to add two numbers to each other. You sit down and lay out all the steps you need to take before you write the actual code. The way we plan is similar to pseudocode! It assists you in outlining the functionality of your program before you begin writing any code.

**What your pseudocode will look like:**

First step: Input Numbers

- Ask the user to input the first and second numbers.

Second step: Adding Numbers and Printing the Result

- Add the first number to the second number and print out the result.


Each step in this pseudocode indicates a particular action you must perform to add the two numbers together. It assists in decomposing the process into smaller, more doable tasks, much like when you cook by following a recipe. Writing your program step-by-step becomes easier when it is planned out in this manner, much like when you follow directions to solve a math problem.

**Why is pseudocode important?**

Even experienced programmers frequently overlook pseudocode, despite it being a crucial component of programming. Though it's easy to become absorbed in the thrill of writing code and watching your program come to life, planning and outlining your ideas in pseudocode can speed up development significantly. Like many developers, I have also been guilty of jumping into coding without much of a plan. Nevertheless, this method may result in misunderstandings, ineffectiveness, and time loss.

Writing pseudocode first before going into coding has, in my experience, saved me many hours of frustration and debugging. When you're anxious to see results, it's easy to undervalue the significance of organization and planning. That being said, there is no denying that using pseudocode can help you with programming because it provides you with clarity, direction, and confidence.

Moreover, pseudocode plays a significant role in creating Expert Advisors (EAs) and makes using AI-generated code easier to understand in MQL5. When working with complex algorithms and AI-powered tools, pseudocode acts as a bridge between abstract concepts and practical implementation. It helps developers translate high-level ideas into actionable steps, making the process of integrating AI-generated code into MQL5 programs more straightforward and accessible. By using pseudocode to plan and outline the logic of your EAs, you can ensure that your code is well-structured, efficient, and easy to maintain.

Another advantage of understanding pseudocode is its usefulness for [freelance](https://www.mql5.com/en/job) opportunities within the MQL5 community. By effectively communicating your programming plans and strategies to clients through pseudocode, you can demonstrate your expertise and increase your chances of securing projects to build Expert Advisors tailored to their specific needs.

**2.2. How to Write Pseudocode**

The first step in writing pseudocode is to divide the goals of your program into smaller, more doable tasks. Next, explain each process step in simple, straightforward terms. Avoid becoming mired in the syntax of a particular programming language, and instead concentrate on the general logic and flow of your program.

Let's take a look at the EA we'll be working on. For instance, we'll be working on a trading alarm that enables you to set reminders at specific times. What's special about this project is that you'll have the option to choose the sound you want when setting up the reminder. We could write pseudocode for this in the following way:

The first step is to initialize the EA:

- Define and configure the required variables and settings, including.

The second step is to set up reminders:

- Allow users to input the desired alarm times and choose the sound they prefer for each reminder.
- Store the alarm times and selected sounds in the respective variables.

The third step is to monitor the trading time zone, which involves:

- Continuously comparing the current time with the designated alarm times.
- Trigger the audio alert when an alarm time is reached.

The fourth step is to sound the audio alert:

- Notify traders of the set reminder time.
- Play the selected sound file to alert traders.

Every step in this pseudocode represents a particular task or action the EA will carry out. Before beginning the actual coding process, we can efficiently plan out the logic and structure of our EA by breaking down the program's objectives into smaller steps and explaining them in simple terms.

### **3\. Generating MQL5 Codes with AI**

Coding processes have advanced significantly as a result of artificial intelligence (AI). The new approach completely transforms the way developers create Expert Advisors (EAs). Automation of code generation has become a major benefit of MQL5 programming with AI-powered tools. These tools use advanced algorithms to translate high-level instructions and pseudocode into executable MQL5 code by analyzing the input data. Artificial Intelligence shortens development timelines and increases code generation productivity by automating repetitive tasks and decreasing manual intervention.

But even with these developments, AI technology is still in its early stages and prone to coding errors. It's critical to recognize that the code produced by AI might not be flawless and might need to be fixed. Therefore, to effectively use AI-generated codes, developers still need to have a basic or intermediate understanding of MQL5. This guarantees the correctness and dependability of the code generated by the AI by allowing it to recognize and fix mistakes made by the system. Even though AI speeds up the process of creating code, developers still need to be informed and watchful to guarantee the caliber of the finished product.

**3.1. How to Generate Accurate AI Codes**

Here's the step-by-step process:

**3.1.1 EA Pseudocode**

Before using AI to generate MQL5 code, produce a thorough pseudocode. Divide the goals of your EA into smaller, more doable steps, and outline the conditions and actions for each step.

After selecting an AI-powered tool for generating MQL5 code, the subsequent step is to provide a thorough pseudocode outlining your Expert Advisor (EA) logic and functionality. During this stage, each step of the pseudocode will prompt you to ask questions and go deeper into the specifics of your EA's operation. By breaking down the pseudocode into actionable steps and addressing the questions that arise under each step, you can ensure that your EA is comprehensively designed and effectively implements your trading strategy.

The first step is to initialize the EA:

- Define and configure the required variables and settings, including:

This step prompts questions such as:

- Which variables are necessary for the EA to function?
- What data types should be used for the variables?
- How should default values for variables be determined? Are there any external dependencies or resources that need to be loaded?

With these questions, we can ensure that the first step of initializing the EA is executed effectively, with the necessary variables and settings configured, and the recorded voice message is properly loaded and prepared for playback during the designated trading sessions.

The second step is to set up reminders:

- Allow users to input the desired alarm times and choose the sound they prefer for each reminder.
- Store the alarm times and selected sounds in the respective variables.

This step prompts questions such as:

- What format should the alarm times be in?
- How will users select the sound they prefer for the reminder?
- Where will the selected alarm times and sounds be stored?
- Will users have the option to modify or delete existing reminders?

The third step is to monitor the trading time zone, which involves:

- Continuously comparing the current time with the designated alarm times.
- Trigger the audio alert when an alarm time is reached.

This step prompts questions such as:

- How will the EA continuously compare the current time with the designated alarm times?
- What action will be taken when an alarm time is reached?
- How frequently will the EA check for alarm times?

The fourth step is to set up reminders:

- The fourth step is to sound the audio alert:
- Notify traders of the set reminder time.
- Play the selected sound file to alert traders.

This step prompts questions such as:

- How do we add the recorded voice message to our MQL5 project?
- How will the selected sound file be played to alert traders?

These kinds of questions are essential to the process of creating AI code. By carefully responding to them, we create a precise roadmap for code generation powered by AI. This procedure makes sure that the final code accurately captures the complexities of our project, opening the door for the smooth incorporation of AI technology into the creation of our Expert Advisors (EAs).

**3.1.2. Code generation**

Based on the given actions and conditions, generate MQL5 code step-by-step using some of the questions derived from the pseudocode as a guide. Use the chosen AI-powered tool to decipher each pseudocode segment you don’t have answers to and convert it to MQL5 executable code. Make sure that the generated code appropriately reflects the intended logic and functionality by concentrating on addressing each step in turn.

It's crucial to understand that the code that the AI generates will not be the final version that we employ. Rather, they act as instructive models, demonstrating to you as a beginner how to accomplish particular objectives in your Expert Advisor (EA) creation process.

As a beginner, you can start to identify some of the answers to these questions, and by doing so, you'll better understand the next steps you should take with your project. It's critical to understand that this is also more about introducing the AI with your project's specifics, than merely asking it questions and expecting an answer. You can actively improve the AI's comprehension of your project by interacting with it and asking relevant inquiries. This exchange helps to improve the accuracy of the code that it eventually produces for you to use. The more you communicate with the AI and give it background information about your project, the more capable it is of helping you successfully with your coding tasks.

Note: Before posing questions to an AI regarding your project, it's essential to provide a detailed explanation of the project first. This introduction serves as a foundation for the AI to understand the context and objectives of the project.

To demonstrate how to generate code from the AI, let's begin with the first step using one of the questions:

**Which variables are necessary for the EA to function?**

**AI’s response:**

For the trading alarm project, several variables are necessary for the EA to function effectively. These may include:

- **Alarm Times Variables:** Variables to store the user-defined alarm times. These could be datetime variables representing specific times during the trading day when the reminders should be triggered.
- **Sound Selection Variables:** Variables to store the user's choice of sound for each alarm. This could be a string variable containing the file path or name of the selected sound file.
- **Current Time Variable:** A variable to hold the current time, which will be continuously compared with the alarm times to trigger the reminders.
- **Sound File Path Variables:** Variables to store the file paths of the available sound files that the user can choose from.
- **Notification Message Variable:** A variable to store any notification messages that may accompany the alarms.
- **Other Configuration Variables:** Additional variables may be necessary depending on the specific requirements of the project, such as variables to control the behavior of the EA or to store user preferences

These variables are essential for the EA to track the alarm times, manage user preferences, and trigger the audio alerts at the specified times.

Now, with these theoretical answers in hand, it's apparent that the AI's responses might not always be perfect. However, by providing a detailed explanation of the project beforehand, we've enabled the AI to provide more comprehensive answers. This detailed overview helps the AI understand the context of the questions and provides more insightful responses. This approach enhances the quality of the AI's responses and fosters a more fruitful interaction between the project requirements and the AI's capabilities.

For an even clearer explanation, you can now ask the AI to include a code example. This addition will further illustrate how the theoretical concepts discussed translate into actual code implementation, providing a practical demonstration of the project's functionality.

**AI’s response:**

```
// Define variables for alarm times
   datetime Alarm1;
   datetime Alarm2;
   datetime Alarm3;

// Define variables for sound selection
   string Sound1 = "alert1.wav";
   string Sound2 = "alert2.wav";
   string Sound3 = "alert3.wav";
// Define variables for current time and notification message
   datetime CurrentTime;
   string NotificationMessage;

// Define other configuration variables
   int NumberOfAlarms = 3;
```

In this example:

- Alarm1, Alarm2, and Alarm3 are datetime variables to store the user-defined alarm times.
- Sound1, Sound2, and Sound3 are string variables containing the file paths or names of the selected sound files for each alarm.
-  NotificationMessage is a string variable to store any notification messages.
- NumberOfAlarms is an integer variable indicating the total number of alarms the user can set.

These variables provide the necessary foundation for the EA to manage alarm times, sound selections, and other configurations required for the project

With the explanation provided for the AI, it offers some answers, but they're not perfect. However, from this response, you can extract some questions and ask the AI to refine them to your specifications.

The code provided by the AI helps identify the necessary variables for the project. However, for our specific project, we want the user to input the parameters for the sound and the alarm. This introduces a couple of questions related to Step Two of the pseudocode:

- How will users select the sound and time they prefer for the reminder?
- Where will the selected alarm times and sounds be stored?

These questions highlight the need to implement user input functionality within the EA. Users should be able to choose the sound they prefer for each reminder and specify the corresponding alarm times. The selected alarm times and sounds need to be stored in variables within the EA so that they can be accessed and utilized when needed. This ensures that the EA operates according to the user's preferences and effectively triggers alarms with the specified sounds at the designated times.

**AI’s response:**

To allow users to select the sound they prefer for each reminder, you can define a variable using the **input** keyword, specifying the type as **string**. This variable will store the filename of the sound file selected by the user. Here's how you can modify the code:

```
input string Alarm1 = "00:00:00";
input string Alarm2 = "00:00:00";
input string Alarm3 = "00:00:00";

input string Sound = ""; // User-selected sound filename
```

With this setup, users can input the filename of the sound they want to use for the reminders. You can then use this Sound variable to play the selected sound when the reminder is triggered.

This demonstrates how important it is to follow the pseudocode step by step. First, we determine which variables are essential for the EA to work as intended. The project's foundation is formed by these variables. The user interaction component is then addressed in the second step, where users are given the option to input their preferred sound files and desired alarm times. Following this methodical process guarantees that every aspect of the project is methodically handled, resulting in a more thorough comprehension. The interaction with the AI helps refine and clarify the project requirements, leading to a more robust implementation, even though the code provided at this stage might not be the final version.

We now proceed to the third step, which is keeping an eye on the trading time zone, by following the pseudocode's progression. This makes us think about the following queries:

- How will the EA keep track of the time difference between the set alarm times and the current time?
- When an alarm time is reached, what will happen?

By answering these questions, the development process will be guided, and the EA will be able to monitor the designated alarm times and take the appropriate action as needed.

**AI’s response:**

To continuously compare the current time with the designated alarm times, we can use a loop that runs on each tick of the trading platform. Within this loop, we'll check if the current time matches any of the alarm times set by the user.

```
// Inside the OnTick() function
// Get the current time
   datetime currentTime = TimeLocal();

// Check if the current time matches any of the alarm times
   if(currentTime == StringToTime(Alarm1) ||
      currentTime == StringToTime(Alarm2) ||
      currentTime == StringToTime(Alarm3))
     {
      // Trigger the audio alert
      PlaySound(Sound);
     }
```

In this code snippet:

- We obtain the current time using TimeLocal().
- We compare the current time with each of the alarm times specified by the user using StringToTime().
- If the current time matches any of the alarm times, we trigger the audio alert by playing the selected sound using PlaySound(Sound).

This code ensures that the EA continuously monitors the specified alarm times and takes action when an alarm time is reached

The responses from the AI are getting more precise as we move through the stages described in the pseudocode. This highlights the significance of methodically dividing the project into doable phases. The AI's explanations match our project objectives more closely with every step. It assists us in better understanding how to approach the project's more difficult aspects, as well as how to approach the development process itself.

Engaging with the AI iteratively is proving to be a very useful process. In addition to assisting us with the project, it is expanding our comprehension of how to handle its intricacies. We're getting more insight into how to effectively handle the project's multiple aspects as we interact with the AI.

As we transition to the fourth step of the pseudocode, new questions emerge that require consideration:

- How do we add the recorded voice message to our MQL5 project?
- How will the selected sound file be played to alert traders?

Addressing these questions will involve integrating audio files into our MQL5 project and implementing the logic to trigger the playback of the selected sound file at the designated alarm times. Let's explore these aspects further to ensure our alarm trading clock functions as intended.

**AI’s response:**

To add a voice message to your MQL5 project, follow these steps:

1. Prepare a recorded voice message in a compatible format (e.g., WAV).
2. Within the MetaTrader 5 installation directory, you'll find a folder named "Sounds." Open this folder.
3. Copy your sound file (in WAV format) into the "Sounds" folder. If your sound file is in a different format (e.g., MP3), you'll need to convert it to WAV format before copying it into the "Sounds" folder.
4. Once you've copied the sound file into the "Sounds" folder, it should be available for use within MetaTrader 5. You can reference the sound file by its filename when using functions like PlaySound() in your MQL5 code.
5. Trigger playback of the voice message using the PlaySound() function.
6. For example, if you've copied a sound file named "alert.wav" into the "Sounds" folder, you can play this sound file in your MQL5 code like this:

```
PlaySound("alert.wav");
```

Replace "alert.wav" with the filename of your sound file. Ensure that the filename matches exactly (including capitalization) to avoid errors when referencing the sound file in your MQL5 code.

That's it! Your sound file should now be successfully added to MetaTrader 5 and ready for use in your Expert Advisors or scripts.

All the responses produced by the AI provide ways to address the issues raised by the project. Rather than expecting AI to do all the work, which is usually not accurate, this methodical approach is the best way to work with AI in our projects. It's critical to realize that artificial intelligence (AI) is not here to replace humans; rather, it can help and provide insights that can enhance human expertise and creativity, making work easier. You can use AI in this way to work with them to improve their productivity and problem-solving abilities, as well as to gain a better understanding of their projects.

**3.1.3. Iterative Testing**

We can now better understand how to move forward with our project thanks to the AI, so the next step is to conduct step-by-step testing and debugging. Using scripts to analyze your code's performance and find errors or problems is the most effective way to debug it. You can more successfully isolate issues and make sure the entire program runs as intended by segmenting the code into smaller parts and testing each one separately. Testing and debugging the code iteratively is crucial to improving its quality and guaranteeing its dependability and functionality in practical situations.

Now, for example, let's test how all the input required from the user will work:

```
// Define input variables
input string Alarm1 = "00:00:00"; // Default value for first alarm
input string Alarm2 = "00:00:00"; // Default value for second alarm
input string Alarm3 = "00:00:00"; // Default value for third alarm
input string Sound = "alert.wav"; // Default sound file

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
// Initialization function, executed once when the EA is launched
   return 0; // Return 0 to indicate successful initialization
  }

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
// Deinitialization function, executed before the EA is removed
// Perform any cleanup or necessary actions here

  }

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
// Convert alarm strings to datetime objects
   datetime first_Alarm = StringToTime(Alarm1);
   datetime second_Alarm = StringToTime(Alarm2);
   datetime third_Alarm = StringToTime(Alarm3);

// Get current local time
   datetime Time = TimeLocal();

// Check if current time matches any of the alarm times
   if(first_Alarm == Time || second_Alarm == Time || third_Alarm == Time)
     {
      // Play sound if any alarm time is reached
      PlaySound(Sound);
     }

  }
```

**Exlanation:**

This code snippet serves as a testing ground to understand how user inputs will function within our program. It defines input variables Alarm1, Alarm2, Alarm3, and Sound, which allow users to specify alarm times and choose a sound. During initialization, the values of these variables are printed to the console, providing a visual confirmation of the user inputs. The placeholder functions [OnInit()](https://www.mql5.com/en/docs/event_handlers/oninit) and [OnDeinit()](https://www.mql5.com/en/docs/event_handlers/ondeinit) handle EA initialization and deinitialization respectively, while [OnTick()](https://www.mql5.com/en/docs/event_handlers/ontick) serves as a placeholder for processing ticks.

**Output:**

![Figure 1. Code Output in MetaTrader5](https://c.mql5.com/2/77/outtt1.png)

After confirming that our code functions correctly, we can proceed to test other parts of the program. For instance, let's say we want to test using a script. One aspect we might want to test is the printing of the local time. This test will help ensure that our program accurately retrieves and displays the current time, which is crucial for triggering alarms at the specified times.

Example:

```
void OnStart()
  {
// Get the current local time
   datetime localTime = TimeLocal();

// Print the local time to the console
   Print("Current Local Time: ", localTime);
  }
```

**Explanation:**

This code snippet defines an OnStart() function, which is triggered when a script starts running in MQL5. It fetches the current local time using the [TimeLocal()](https://www.mql5.com/en/docs/dateandtime/timelocal) function and then prints this time along with a descriptive message to the console using the Print() function. This allows the developer to quickly verify the accuracy of the time retrieval and output functionality during testing and debugging.

With this code, we've successfully tested our ability to retrieve the local time from the computer accurately. This ensures that our program can access and utilize local time data effectively, which is essential for various time-based functionalities in our project. Similarly, we should apply this method of testing and debugging to the rest of our code, ensuring that each component functions as intended before integrating them into the larger project.

Our ability to precisely retrieve the local time from the computer has been successfully tested with this code. Because many of the time-based features in our project depend on our program's ability to access and use local time data, this guarantees that it can do so. By making sure that every part works as intended before integrating it into the overall project, we should also use this testing and debugging approach for the remaining portions of our code.

You can ask the AI to integrate your tested code segments into a meaningful program now that every step of the code has been tested. You can, however, make the required adjustments on your own in certain situations if it falls short of your expectations. With its constant guidance and support, the AI has simplified your work and saved you a great deal of time.

**AI’s response:**

```
// Define input variables
input string Alarm1 = "00:00:00"; // Default value for first alarm
input string Alarm2 = "00:00:00"; // Default value for second alarm
input string Alarm3 = "00:00:00"; // Default value for third alarm
input string Sound = "alert.wav"; // Default sound file

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
// Initialization function, executed once when the EA is launched
   return 0; // Return 0 to indicate successful initialization
  }

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
// Deinitialization function, executed before the EA is removed
// Perform any cleanup or necessary actions here

  }

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
// Convert alarm strings to datetime objects
   datetime first_Alarm = StringToTime(Alarm1);
   datetime second_Alarm = StringToTime(Alarm2);
   datetime third_Alarm = StringToTime(Alarm3);

// Get current local time
   datetime Time = TimeLocal();

// Check if current time matches any of the alarm times
   if(first_Alarm == Time || second_Alarm == Time || third_Alarm == Time)
     {
      // Play sound if any alarm time is reached
      PlaySound(Sound);
     }

  }
```

This block defines the OnTick() function, executed on every tick of the chart. It's the heart of the EA's functionality. Here's what it does:

- It converts the alarm time strings (Alarm1, Alarm2, Alarm3) into datetime objects using the StringToTime() function.
- It fetches the current local time using the TimeLocal() function and stores it in the Time variable.
- It checks if the current time matches any of the alarm times. If so, it plays the specified sound using the PlaySound() function.

This code essentially monitors the current time and triggers an alarm sound when any of the specified alarm times are reached.

_Note: To edit your inputs press the F7 button._

### **Conclusion**

To sum up, this article has given a thorough overview of project-based learning for MQL5 mastery. By breaking down complex concepts into manageable projects, learners can deepen their understanding and proficiency in MQL5 programming. By using AI tools, iterative testing, and step-by-step pseudocode, developers can efficiently create Expert Advisors and improve their trading strategies. Recall that learning MQL5 is a marathon rather than a sprint and that embracing project-based learning is an essential first step on the path to becoming proficient. So go ahead and take on projects, try out some code, and keep learning and developing as a developer.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/14651.zip "Download all attachments in the single ZIP archive")

[Alarms.mq5](https://www.mql5.com/en/articles/download/14651/alarms.mq5 "Download Alarms.mq5")(2.19 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/466825)**
(12)


![Oluwatosin Mary Babalola](https://c.mql5.com/avatar/2024/2/65cfcb6a-f1c4.jpg)

**[Oluwatosin Mary Babalola](https://www.mql5.com/en/users/excel_om)**
\|
13 Jul 2024 at 21:11

I’ve been following your series and I must say the explanations are great and I’m looking forward to learning more from [project](https://www.mql5.com/en/articles/7863 "Article: Projects allow you to create profitable trading robots! But it is not exactly") based articles like this


![Israel Pelumi Abioye](https://c.mql5.com/avatar/2023/11/6554a830-8858.png)

**[Israel Pelumi Abioye](https://www.mql5.com/en/users/13467913)**
\|
13 Jul 2024 at 22:07

**Oluwatosin Mary Babalola [#](https://www.mql5.com/en/forum/466825#comment_53975622):**

I’ve been following your series and I must say the explanations are great and I’m looking forward to learning more from [project](https://www.mql5.com/en/articles/7863 "Article: Projects allow you to create profitable trading robots! But it is not exactly") based articles like this

Hello Oluwatosin, I look forward to sharing the next part with you. If you have any questions about the article, please feel free to reach out to me.

![Stanislav Korotky](https://c.mql5.com/avatar/2010/10/4CA7CFA0-1F0C.jpg)

**[Stanislav Korotky](https://www.mql5.com/en/users/marketeer)**
\|
3 Oct 2024 at 16:00

And why would a beginner's guide immediately teach them to code incorrectly?

Firstly, there are efficiency problems that will become noticeable as soon as a person starts writing something a bit more complex and realistic (but continues to use the same technique). On every tick, a [StringToTime](https://www.mql5.com/en/docs/convert/stringtotime "MQL5 documentation: StringToTime function") conversion of the same strings is called three times. It would be possible to clean up for the AI - make the conversion once in global variables:

```
input string Alarm1 = "00:00:00"; // Default value for first alarm
input string Alarm2 = "00:00:00"; // Default value for second alarm
input string Alarm3 = "00:00:00"; // Default value for third alarm
input string Sound = "alert.wav"; // Default sound file

// Define globals for persistent values
datetime first_Alarm, second_Alarm, third_Alarm;

int OnInit()
{
   // Initialization function, executed once when the EA is launched
   // Make conversions only once per inputs
   first_Alarm = StringToTime(Alarm1);
   second_Alarm = StringToTime(Alarm2);
   third_Alarm = StringToTime(Alarm3);
   return 0; // Return 0 to indicate successful initialization
}

void OnTick()
{
   // Get current local time
   datetime Time = TimeLocal();

   // Check if current time matches any of the alarm times
   if(first_Alarm == Time || second_Alarm == Time || third_Alarm == Time) // another logical error here (see below)
   {
      // Play sound if any alarm time is reached
      PlaySound(Sound);
   }
}
```

Secondly, there is no guarantee that the tick times will match the given alert times - most likely this code will miss them in the _if-times_. And even if we had made checks in the OnTimer handler, it would not guarantee that we would "catch" the necessary moments, because all events (OnTick, OnTimer, etc.) come from the queue, which is emptied as fast as possible, but may slow down if some other programme is doing heavy calculations. Therefore, the condition of checking the occurrence of alerts must be rewritten thoroughly, and at least without additional variables is not possible (in the future, it is suggested to remake alerts under classes, but of course not in this article).

![Stanislav Korotky](https://c.mql5.com/avatar/2010/10/4CA7CFA0-1F0C.jpg)

**[Stanislav Korotky](https://www.mql5.com/en/users/marketeer)**
\|
3 Oct 2024 at 18:11

**Stanislav Korotky [#](https://www.mql5.com/en/forum/466825#comment_54741302):**

And why would a beginner's guide immediately teach them to code incorrectly?

Firstly, there are efficiency problems that will become noticeable as soon as a person starts writing something a bit more complex and realistic (but continues to use the same technique). On every tick, a [StringToTime](https://www.mql5.com/en/docs/convert/stringtotime "MQL5 documentation: StringToTime function") conversion of the same strings is called three times. It would be possible to clean up for the AI - make the conversion once in global variables:

Secondly, there is no guarantee that the tick times will match the given alert times - most likely this code will miss them in the _if-times_. And even if we had made checks in the OnTimer handler, it would not guarantee that we would "catch" the necessary moments, because all events (OnTick, OnTimer, etc.) come from the queue, which is emptied as fast as possible, but may slow down if some other programme is doing heavy calculations. Therefore, the condition of checking the occurrence of alerts must be rewritten thoroughly, and at least without additional variables is not possible (in the future, it is suggested to remake alerts under classes, but of course not in this article).

I don't understand how this automatic translation feature works here. During initial writing the forum showed me a different, more accurate translation to English. But here it's translated very bad. Normally I reply in English, but this time hoped in vain that autotranslation will suffice, but it disoriented me with different wording here and there.

![Israel Pelumi Abioye](https://c.mql5.com/avatar/2023/11/6554a830-8858.png)

**[Israel Pelumi Abioye](https://www.mql5.com/en/users/13467913)**
\|
3 Oct 2024 at 21:41

**Stanislav Korotky [#](https://www.mql5.com/en/forum/466825/page2#comment_54742826):**

I don't understand how this automatic translation feature works here. During initial writing the forum showed me a different, more accurate translation to English. But here it's translated very bad. Normally I reply in English, but this time hoped in vain that autotranslation will suffice, but it disoriented me with different wording here and there.

Hello, Mr. Korotky,

Thank you very much for your comment. You raised some excellent points. At this series stage, I try to keep the material as simple as possible, as it's intended for beginners. I intentionally use this method to explain step-by-step, as the program is simple. When we start working on more complex projects, those issues will be addressed.

I appreciate your remarks and your insight on code efficiency. Thank you for your understanding and for sharing your experience.

![Building A Candlestick Trend Constraint Model(Part 2): Merging Native Indicators](https://c.mql5.com/2/77/Building_A_Candlestick_Trend_Constraint_ModelsPart_2t____LOGO.png)[Building A Candlestick Trend Constraint Model(Part 2): Merging Native Indicators](https://www.mql5.com/en/articles/14803)

This article focuses on taking advantage of in-built meta trader 5 indicators to screen out off-trend signals. Advancing from the previous article we will explore how to do it using MQL5 code to communicate our idea to the final program.

![MQL5 Wizard Techniques you should know (Part 18): Neural Architecture Search with Eigen Vectors](https://c.mql5.com/2/77/MQL5_Wizard_Techniques_you_should_know_fPart_18j___LOGO.png)[MQL5 Wizard Techniques you should know (Part 18): Neural Architecture Search with Eigen Vectors](https://www.mql5.com/en/articles/14845)

Neural Architecture Search, an automated approach at determining the ideal neural network settings can be a plus when facing many options and large test data sets. We examine how when paired Eigen Vectors this process can be made even more efficient.

![A feature selection algorithm using energy based learning in pure MQL5](https://c.mql5.com/2/78/A_feature_selection_algorithm_using_energy_based_learning_in_pure_MQL5____LOGO.png)[A feature selection algorithm using energy based learning in pure MQL5](https://www.mql5.com/en/articles/14865)

In this article we present the implementation of a feature selection algorithm described in an academic paper titled,"FREL: A stable feature selection algorithm", called Feature weighting as regularized energy based learning.

![Data Science and Machine Learning (Part 22): Leveraging Autoencoders Neural Networks for Smarter Trades by Moving from Noise to Signal](https://c.mql5.com/2/77/Data_Science_and_ML_gPart_22k_____LOGO.png)[Data Science and Machine Learning (Part 22): Leveraging Autoencoders Neural Networks for Smarter Trades by Moving from Noise to Signal](https://www.mql5.com/en/articles/14760)

In the fast-paced world of financial markets, separating meaningful signals from the noise is crucial for successful trading. By employing sophisticated neural network architectures, autoencoders excel at uncovering hidden patterns within market data, transforming noisy input into actionable insights. In this article, we explore how autoencoders are revolutionizing trading practices, offering traders a powerful tool to enhance decision-making and gain a competitive edge in today's dynamic markets.

[![](https://www.mql5.com/ff/si/w766tj9vyj3g607n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fmarket%2Fmt5%2Fexpert%3FHasRent%3Don%26utm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Drent.expert%26utm_content%3Drent.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=sorsafcerhkgwrjzwwrpvelbicxjwzon&s=ae91b1eae8acb61167455495742e6cc8eb55ccedb33fd953f8256b68cbe9c3b4&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=kumuxhohoerleolggkdbzzaeyhxzqyel&ssn=1769092184282044433&ssn_dr=0&ssn_sr=0&fv_date=1769092184&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F14651&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Introduction%20to%20MQL5%20(Part%207)%3A%20Beginner%27s%20Guide%20to%20Building%20Expert%20Advisors%20and%20Utilizing%20AI-Generated%20Code%20in%20MQL5%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176909218446414684&fz_uniq=5049172250835396183&sv=2552)

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