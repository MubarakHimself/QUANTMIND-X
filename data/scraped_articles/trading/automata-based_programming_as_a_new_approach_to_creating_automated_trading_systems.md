---
title: Automata-Based Programming as a New Approach to Creating Automated Trading Systems
url: https://www.mql5.com/en/articles/446
categories: Trading, Trading Systems, Integration, Expert Advisors
relevance_score: 12
scraped_at: 2026-01-22T17:12:57.119777
---

[Running robots on virtual hosting is easyFollow our step-by-step MetaTrader VPS guide for beginnersRead![](https://www.mql5.com/ff/sh/au4fqg4kms7s9mq1z2/01.png)![](https://www.mql5.com/ff/sh/au4fqg4kms7s9mq1z2/02.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/articles/13586&a=uzpprdshbcrtxvjxpmescehprypbymxc&s=516438f25b531570d9b7d49dcfb29c82fa1021f5ede6571df8026dbfbafcd13f&uid=&ref=https://www.mql5.com/en/articles/446&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5048972066704695910)

MetaTrader 5 / Trading


_**ἓν οἶδα ὅτι οὐδὲν οἶδα ( ο φιλόσοφος Σωκράτης )**_

_**I know that I know nothing (the philosopher Socrates)**_

### Introduction

To start off, this subject is completely new to traders developing EAs using the MetaQuotes Language 4/5 (MQL4/5). I could see it for myself when I tried to do a relevant search on the MetaQuotes website. There is nothing on this subject.

Every trader creates its own Expert Advisor which requires a serious approach to tackling all kinds of problems associated with programming and very complicated program logic. At the end of the day, the program is supposed to run ON ITS OWN like clockwork in any standard and force majeure situation.

But how can one embrace everything? This is extremely difficult which is why automatic control systems require proper programming of all control systems that can be best achieved using the only appropriate programming technology of Automata-Based Programming. In recent years, a great attention has been revolving around the development of programming technologies for embedded and real-time systems that set high requirements to the quality of software.

In 1991, the Russian author [A.A. Shalyto](https://www.mql5.com/go?link=http://www.ifmo.ru/person/94/person_94.htm "http://www.ifmo.ru/person/94/person_94.htm") (Lecturer, Professor, DSc in Engineering, Head of the "Programming Technologies" Department at [SPbSU ITMO](https://en.wikipedia.org/wiki/Saint_Petersburg_State_University_of_Information_Technologies,_Mechanics_and_Optics "https://en.wikipedia.org/wiki/Saint_Petersburg_State_University_of_Information_Technologies,_Mechanics_and_Optics")) developed a programming technology which he called "automata-based programming". I believe the readers may find it interesting to see how simple automata-based programming or SWITCH-technology can be. It allows to make the development of MTS using the MetaQuotes Language so convenient that it simply could not be better. And it blends great into the system of complex decision making.

### 1\. Working Up to the Problem

A long cherished dream of all problem originators and software developers must be to have a planned solution to the problem (algorithm) and an implementation of that algorithm fully consistent with it. But it does not seem to be working out that way for originators and developers. Algorithms tend to leave out what is important to developers for implementation, while the program text per se bears little resemblance to the algorithm.

Thus, there are two algorithms - one is on paper (for recording and documenting design solutions) which usually represents a certain design outcome instead of methods employed in obtaining a given result, while the second algorithm is in the developer's mind (which is, however, also saved textually).

The final version of the program text is often followed by attempts to modify the documentation whereby many things are again not taken into account. In this case, the program logic may likely be different from the logic of the algorithm, thus demonstrating a lack of correspondence. I intentionally say 'likely' as nobody is ever going to check somebody's program text.

If the program is large, it is impossible to check whether it corresponds with the algorithm by the text alone. The accuracy of implementation can be checked using a procedure called 'testing'. It basically checks how the developer grasped the algorithm (laid out on paper), transformed it into another algorithm in his mind and output it as a program. Eventually, the developer is the only holder of valuable information on the logic and all that was coined before the implementation becomes absolutely irrelevant.

It is not even that the developer can fall ill (or... resign). The point is that the underlying program logic would be different with every developer, depending on their intelligence and knowledge of a programming language. In any case, the developer introduces and uses a lot of intermediate variables as he deems fit. And if the program is large and logically complex, a more qualified specialist will be required to find the glitches (and I do not mean OS glitches or incorrect use of language functions here but rather an improper implementation in terms of logic) and work it out through the program text itself.

The majority of developers are, to say the least, not keen on writing down algorithms before programming (or even sketching them on paper), which is likely due to the fact that they will still need to think up something of their own along the way. Indeed, why waste time on drawing some rectangles, diamonds and arrows when it is better to immediately proceed to programming and then lay out a somewhat similar or a very general algorithm in documentation.

Everybody has got used to it - developers do it because it is easier that way, while problem originators do not always have programming skills to the extent required and even in the case they do, they are simply unable to make a timely change to what developers will come up with. Convenient programming environments also contribute to the validity of the specified order of development. Advanced tools for debugging and monitoring values of variables give us hope for detection of any error in the logic.

As time is slipping away and the project deadline is approaching, the developer is sitting and sketching on a 'napkin' solutions to a given logical problem which, by the way, still need to be implemented, let alone errors overlooked during testing for testing follows pretty much the same (chaotic) scenario... This is the current situation. Is there a solution or can it at least be improved? It feels that something important is lost in the transition from an algorithm laid out in a standard way to the program code.

### 2\. Logical Part of the Program

The author of "automata-based programming" proposed the following concept of the ideal logical part of a program. The entire logic of the program is based on the switch. Simply put, any control algorithm (automaton) can be implemented as shown below (do not think much about the meaning of the comments at this point, just take a look at the structure).

```
switch(int STATUS ) // Мulti-valued global state variable of the automaton.
{
  case 0:  // start

  // Checking arc and loop conditions (in order of priority),
  // transition (change of the value of the variable STATUS)
  // and execution of arc and loop actions (output function execution);
  // logging transitions and actions if the condition is met. 0

  // Calling nested automata.
  // Execution of output functions in the state.
  break ;

  case 1:
  // Checking arc and loop conditions (in order of priority),
  // transition (change of the value of the variable STATUS)
  // and execution of arc and loop actions (output function execution);
  // logging transitions and actions if the condition is met.

  // Calling nested automata.
  // Execution of output functions in the state.
  break ;

*********
*********
*********

 case N-1:
  // Checking arc and loop conditions (in order of priority),
  // transition (change of the value of the variable STATUS)
  // and execution of arc and loop actions (output function execution);
  // logging transitions and actions if the condition is met.

  // Calling nested automata.
  // Execution of output functions in the state.
  break ;

 case N:
  // Checking arc and loop conditions (in order of priority),
  // transition (change of the value of the variable STATUS)
  // and execution of arc and loop actions (output function execution);
  // logging transitions and actions if the condition is met.

  // Calling nested automata.
  // Execution of output functions in the state.
  break ;
}
```

### 3\. Automata-Based Programming As Explained by the Author A.A. Shalyto

Regardless of the development techniques, any program has states determined by all its data values at any specific time. There may be hundreds and even thousands of variables and several control flows in a large application program. A complete set of these variables describes the state of the application program at any specific time.

The program state can be treated in a simpler way as a collection of values of all control variables, the ones that take part in all transition conditions. A change in the value of one of control variables will then mean a change in the program state, and the number of program states will be determined by the maximum possible number of combinations of control variable values arising during the program operation. Suppose that only binary control variables (flags) are used in a program. The number of states of the program containing n binary control variables will in this case lie within the range from n to 2n.

It may be that the developer has provided for reactions to all combinations of control variable values (2n combinations in our case). However, it is more likely that some combinations of control variable values (up to 2n-n) turned out to be unspecified. Then if the unexpected combination of input actions arises, the program can transition into an unspecified state.

**It has the same effect as inaction of an EA for a trader in the following events:**

- **gap,**
- **loss of the deposit,**
- **falling into a negative balance situation with subsequent Margin Call,**
- **not taking good profit going to zero and further in the red,**
- **incorrect opening and closing of longs and shorts,**
- **other obviously adverse situations.**

Such states are called "unvisualized". Complexity causes the difficulty of enumerating, much less understanding, all the possible states of the program leading to its unreliability... Complexity of the structure is the source of unvisualized states that constitute security trapdoors. The program behavior in an unspecified state may range from memory protection faults to extending program to new functions and creating side effects of various nature.

A lot of PC users and probably all software developers have many a time come across situations where a program in use or under development gets into an unspecified state.

To eliminate the very possibility of unspecified states in the program, all required states should be explicitly specified as early as at the design stage and only one multi-valued control variable should be used to differentiate between them. It is then necessary to identify all possible transitions between the states and develop a program so that it cannot 'go astray'.

Three components are required to achieve rigor in the development of the program behavior:

- Mathematical model allowing to unambiguously identify program states and possible transitions between them;
- Graphical notation for that model;
- Universal method for implementation of algorithms expressed in this notation.

A finite automaton based on the notion of "state" is proposed to be used as a mathematical model. Automata-based programming supports such software development stages as design, implementation, debugging and documentation.

Whereas the term **'event'** is becoming more commonly used in programming in the recent years, the proposed approach is based on the notion of **'state'**. After coupling it with the term **'input action'** that can either be an input variable or an event, the term **'automaton without output'** can be introduced. The latter is followed by the term **'output action'** and the notion of (deterministic finite) **automaton** is further introduced. The area of programming based on this concept is therefore called automata-based programming and the respective development process is referred to as automata-based program design.

The specified approach is peculiar in that when it is applied, automata are represented by transition graphs. To differentiate between their nodes, the term **'state assignment'** is introduced. When choosing a 'multi-valued state assignment', the states whose number coincides with the number of values the selected variable can take on, can be differentiated using only one variable. This fact allowed to introduce into programming the term **'program observability'**.

Programming under the proposed approach is carried out through 'states' rather than 'variables' (flags) which helps to better understand and specify the problem and its components. Debugging in this case is done by logging in terms of automata.

Since the above approach proposes to move from the transition graph to the program code using a formal and isomorphic method, it appears to be more reasonable to do this applying switch structures when high-level programming languages are used. That is why it was decided to use the term 'SWITCH-technology' when referring to automata-based programming paradigm.

### 4\. Explicit State-Based Programming

The application of automata-based approach was further extended to event-driven systems that are also called 'reactive'. Reactive systems interact with the environment using messages at the rate set by the environment (an EA can be included in the same class).

The development of event-driven systems using automata was made possible by employing the procedural approach, from where the name **explicit state-based programming** was derived. Output actions are in this method assigned to arcs, loops or nodes of transition graphs (mixed automata - Moore and Mealy automata - are used). This allows to get a compact representation of the sequence of actions being reactions to relevant input actions.

The proposed approach to programming the given class of systems features an increased centralization of logic due to it being eliminated from event handlers and generation of a system of interconnected automata called from the handlers. The interaction between automata in such a system can be achieved by nesting, calling and exchanging numbers of states.

The system of interconnected automata forms a system-independent program part while a system-dependent part is formed by input and output action functions, handlers, etc.

Another key feature of the given approach is that when it is applied, automata are used in the triune way:

- for specification;
- for implementation (they remain in the program code);
- for logging in terms of automata (as specified above).

The latter allows to control the accuracy of automata system operation. Logging is performed automatically on the basis of the program developed and can be used for large-scale problems with complex program logic. Every log can in this case be considered as a relevant script.

Logs allow to monitor the program in operation and illustrate the fact that automata are not 'pictures' but real active entities. Automata-based approach is proposed to be used not only when creating a control system but also when modeling control objects.

### 5\. Basic Concepts of Automata-Based Programming

The basic concept of automata-based programming is **STATE**. The main property of the system state at any specific time t0 is to 'separate' the future (t > t0) from the past (t < t0) in the sense that the current state contains all information on the system's past that is necessary in determining its reactions to any input action generated at any given time t0.

When using the term **STATE** the knowledge of historical data is not required. State can be regarded as a special characteristic which implicitly combines all input actions of the past affecting the reaction of the entity at the present moment. The reaction now only depends on the input action and the current state.

The notion of **'input action'** is also one of the key notions in automata-based programming. An input action is most commonly a vector. Its components are divided into events and input variables, depending on the meaning and generation mechanism.

The combination of the finite set of states and finite set of input actions forms a (finite) automaton without output. Such automaton reacts to input actions by changing its current state in a certain way. The rules according to which the states can be changed are called the automaton transition function.

What is referred to as (finite) automaton in automata-based programming is basically the combination of **'automaton without output'** and **'input action'**. Such automaton reacts to the input action by not only changing its state but also by generating certain values at outputs. The rules of generating output actions are called the **automaton output function**.

When designing a system with complex behavior, it is necessary to take as a point of departure the existing control objects with a certain set of operations and a given set of events that may arise in the external ( **market**) environment.

In practice, the design is more commonly premised on control objects and events:

1. The initial data of the problem is not merely a verbal description of the target behavior of the system but also a (more or less) accurate specification of the set of events incoming to the system from the external environment and a great number of requests and commands of all control objects.

2. A set of control states is built.

3. Every request of control objects is assigned a corresponding input variable of the automaton, while every command is assigned a corresponding output variable. The automaton that is going to ensure a required system behavior is built based on control states, events, input and output variables.


### 6\. Program Features and Advantages

The first feature of an automata-based program is that the presence of an outer loop is essential. There basically seems to be nothing new; the main thing here is that this loop will be the only one in the logical part of the entire program! ( **i.e. new incoming tick**.)

The second feature follows from the first one. Any automaton contains a switch structure (in fact, it is virtually made of it) that comprises all logical operations. When an automaton is called, the control is transferred to one of the 'case' labels and following the relevant actions, the automaton (subprogram) operation is completed until the next start. These actions consist in checking transition conditions and should a certain condition be met, the relevant output functions are called and the automaton state is changed.

The main consequence of all stated above is that the implementation of an automaton is not only simple but most importantly that the program can do without many intermediate logical variables (flags) whose functionality in every automaton is provided by a multi-valued state variable.

The last statement is hard to believe as we got used to using a lot of global and local variables (flags) without thinking too much. How can we do without them?! These are very often flags signaling to the program that a condition is met. The flag is set (to TRUE) when the developer deems necessary but is then (usually only after the flag begins to give rise to wanted effects by being always TRUE) painfully sought to be reset to FALSE elsewhere in the program.

Sounds familiar, doesn't it? Now take a look at the example and see: no additional variables are used here; the change concerns only the value of the state number and only when a logical condition is met. Isn't it a worthy replacement for flags?!

Algorithm plays a major role in creating a logical part of a program. The key phrase to be remembered here is _'logical part'_. State underlies everything in this case. Another word that should be added is 'waiting'. And, in my opinion, we get a quite adequate definition of **'waiting state'**. When in the state, we wait for the appearance of input actions (attributes, values or events). Waiting can be either short or long. Or, in other words, there are states that can be unstable and stable.

**The first property of the state** is the fact that a limited set of input actions is waited for in the state. Any algorithm (and obviously any program) has input and output information. Output actions can be divided into two types: variables (e.g. object property operations) and functions (e.g. call of application start function, report function, etc.).

**The second property of the state** is provision of a set of accurate values of output variables. This reveals a very simple yet extremely important circumstance - all output variable values can be determined at any given time as the algorithm (program) is in a certain state at every point in time.

The number of states is limited, as is the number of output variable values. The function for logging transitions is smoothly integrated into the automaton function and the sequence of transitions between states, as well as the delivery of output actions can consequently always be determined.

The complete list of features is provided in section [2\. Features of the Proposed Technology](https://www.mql5.com/go?link=http://www.softcraft.ru/design/switch/sw02.shtml "http://www.softcraft.ru/design/switch/sw02.shtml"), and the complete list of advantages can be found in section [3\. Advantages of the Proposed Technology](https://www.mql5.com/go?link=http://www.softcraft.ru/design/switch/sw03.shtml "http://www.softcraft.ru/design/switch/sw03.shtml"). This article simply cannot cover the vast amount of information on the subject! After a thorough study of all the research literature written by Anatoly Shalyto, all theoretical questions should be directed toward him personally to shalyto@mail.ifmo.ru.

And being the user of his scientific ideas while having in mind our goals and problems, I will further below give three examples of my implementation of automata-based programming technology.

### 7\. Examples of Automata-Based Programming

**7.1. Example for Comprehension**

State is just a mode in which the system exists. For example, water exists in 3 states: solid, liquid or gas. It transitions from one state to another when influenced by one variable - the temperature (at constant pressure).

Assume, we have a time-based chart of Temperature (t) (in our case - the price value):

![](https://c.mql5.com/2/4/xhy_d0g_0q7k.jpg)

```
int STATUS=0; // a global integer is by all means always a variable !!! STATUS is a multi-valued flag
//----------------------------------------------------------------------------------------------//
int start() // outer loop is a must
  {
   switch(STATUS)
     {
      case 0:  //--- start state of the program
         if(T>0 && T<100) STATUS=1;
         if(T>=100)       STATUS=2;
         if(T<=0)         STATUS=3;
         break;

      case 1:  //---  liquid
         // set of calculations or actions in this situation (repeating the 1st status -- a loop in automata-based programming) //
         // and calls of other nested automata A4, A5;
         if(T>=100 )      { STATUS=2; /* set of actions when transitioning, calls of other nested automata A2, A3;*/}
         if(T<0)          { STATUS=3; /* set of actions when transitioning, calls of other nested automata A2, A3;*/}
         // logging transitions and actions when the condition is met.
         break;

      case 2:  //--- gas
         // set of calculations or actions in this situation (repeating the 2nd status -- a loop in automata-based programming) //
         // and calls of other nested automata A4, A5;
         if(T>0 && T<100) { STATUS=1; /* set of actions when transitioning, calls of other nested automata A2, A3;*/}
         if(T<=0)         { STATUS=3; /* set of actions when transitioning, calls of other nested automata A2, A3;*/}
         // logging transitions and actions when the condition is met.
         break;

      case 3:  //--- solid
         // set of calculations or actions in this situation (repeating the 3rd status -- a loop in automata-based programming) //
         // and calls of other nested automata A4, A5;
         if(T>0 && T<100) {STATUS=1; /* set of actions when transitioning, calls of other nested automata A2, A3;*/}
         if(T>=100)       {STATUS=2; /* set of actions when transitioning, calls of other nested automata A2, A3;*/}
         // logging transitions and actions when the condition is met.
         break;
     }
   return(0);
  }
```

The program can be made more sophisticated by adding the pressure parameter P and new states and introducing a complex dependency demonstrated in the chart:

![](https://c.mql5.com/2/4/b8crij64__1.png)

This automaton has 32 = 9 transition conditions so nothing can be left out or overlooked. This style can also be very convenient when writing instructions and laws! No loopholes and bypassing of laws are allowed here - all combinations of variants of succession of events must be covered and all cases described.

Automata-based programming requires us to take everything into account, even if some variants of succession of events would otherwise not be thought of which is why it is the main tool when checking laws, instructions and control systems for consistency and integrity. There is also a mathematical law:

**If there are N states (except for 0 start) in the system, the total of transition conditions is N2.**

Transition diagram: **N** = 3 states, the number of transitions and loops is **N2** = 9 (equal to the number of arrows).

![](https://c.mql5.com/2/4/cc1buxgu__1.PNG)

If the number of variables in the example was different, then:

![](https://c.mql5.com/2/4/oi4ap3n__3.PNG)

It shows that all the calculated values in the table rise **exponentially**, i.e. design is a complicated process that requires thoroughness when selecting the main systemic variables.

Even if there are only two parameters, it is very difficult to describe everything! However, in practice, it is all much easier! Depending on the logic and meaning, 50-95% transitions cannot exist physically and the number of states is also 60-95% less. This analysis of logic and meaning greatly decreases the difficulty of describing all transitions and states.

In more complicated cases, it is required to calculate the maximum number of states for all known input and output data in an EA. The solution to this problem can be found by applying combinatorics and [combination](https://en.wikipedia.org/wiki/Combination "https://en.wikipedia.org/wiki/Combination"), [permutation](https://en.wikipedia.org/wiki/Permutation "https://en.wikipedia.org/wiki/Permutation"), [arrangement](https://ru.wikipedia.org/wiki/%D0%A0%D0%B0%D0%B7%D0%BC%D0%B5%D1%89%D0%B5%D0%BD%D0%B8%D1%8F "http://ru.wikipedia.org/wiki/Размещения") and [enumerative combinatorics](https://en.wikipedia.org/wiki/Enumerative_combinatorics "https://en.wikipedia.org/wiki/Enumerative_combinatorics") formulas.

**7.2. Relay with Hysteresis**

Programming of relays, triggers, registers, counters, decoders, comparators and other non-linear digital and analog control system elements can be very convenient in an EA.

![](https://c.mql5.com/2/4/w95h.jpg)

- xmax = 100 - the maximum pick-up value;
- xmin = -100 - the minimum pick-up value;
- x = x(t) - signal at the input;
- Y = Y(t) - signal at the output.

```
int   status=0;  // at the beginning of the program we globally assign
//------------------------------------------------------------------//
switch(status)
  {
   case 0: //  start
      Y=x;
      if(x>xmax)  {status=1;}
      if(x<xmin)  {status=2;}
      break;

   case 1: //++++++++++++++++++++
      if(x>xmax)  Y=x;
      if(x<xmax)  Y=xmin;
      if(x<=xmin) {status=2; Y=xmin;}
      break;

   case 2: //--------------------
      if(x<xmin)  Y=x;
      if(x>xmin)  Y=xmax;
      if(x>=xmax) {status=1; Y=xmax;}
      break;
  }
```

The relay characteristic:

![](https://c.mql5.com/2/4/3p40.PNG)

**7.3. Template for 9 States and 81 Variants of Succession of Events**

Y is the current input state of the automaton from 1 to 9. The value of Y is generated in the EA outside the given subprogram. MEGASTATUS is the past state of Y.

```
int MEGASTATUS=0; // at the beginning of the program we globally assign
//---------------------------------------------------------------------//
void A0(int Y) // automaton template
  {
   switch(MEGASTATUS)
     {
      case 0:  // start
          MEGASTATUS=Y;
          break;

      case 1: // it was the past
          // it became current, repeating
          if(Y=1) { /*set of actions in this situation, calls of other nested automata A2, A3, ... */ } // Loop//
          // new current
          if(Y=2) { /* set of actions in this situation */ }
          if(Y=3) { /* set of actions in this situation */ }
          if(Y=4) { /* set of actions in this situation */ }
          if(Y=5) { /* set of actions in this situation */ }
          if(Y=6) { /* set of actions in this situation */ }
          if(Y=7) { /* set of actions in this situation */ }
          if(Y=8) { /* set of actions in this situation */ }
          if(Y=9) { /* set of actions in this situation */ }
          // logging transitions and actions when the condition is met.
          break;

      case 2: // it was the past
          // it has become current
          if(Y=1) { /* set of actions in this situation */ }
          if(Y=2) { /* set of actions in this situation */ } //Loop//
          if(Y=3) { /* set of actions in this situation */ }
          if(Y=4) { /* set of actions in this situation */ }
          if(Y=5) { /* set of actions in this situation */ }
          // e.g. if the transition from 2 to 6 is in essence impossible or does not exist, do not write anything
          if(Y=6) { /* set of actions in this situation */ }
          // the automaton will then be reduced but the automaton template shall be complete to count in everything
          if(Y=7) { /* set of actions in this situation */ }
          if(Y=8) { /* set of actions in this situation */ }
          if(Y=9) { /* set of actions in this situation */ }
          // logging transitions and actions when the condition is met.
          break;

      case 3: // it was the past
          // it has become current
          if(Y=1) { /* set of actions in this situation */ }
          if(Y=2) { /* set of actions in this situation */ }
          if(Y=3) { /* set of actions in this situation */ } //Loop//
          if(Y=4) { /* set of actions in this situation */ }
          if(Y=5) { /* set of actions in this situation */ }
          if(Y=6) { /* set of actions in this situation */ }
          if(Y=7) { /* set of actions in this situation */ }
          if(Y=8) { /* set of actions in this situation */ }
          if(Y=9) { /* set of actions in this situation */ }
          // logging transitions and actions when the condition is met.
          break;

      case 4: // it was the past
          // it has become current
          if(Y=1) { /* set of actions in this situation */ }
          if(Y=2) { /* set of actions in this situation */ }
          if(Y=3) { /* set of actions in this situation */ }
          if(Y=4) { /* set of actions in this situation */ } //Loop//
          if(Y=5) { /* set of actions in this situation */ }
          if(Y=6) { /* set of actions in this situation */ }
          if(Y=7) { /* set of actions in this situation */ }
          if(Y=8) { /* set of actions in this situation */ }
          if(Y=9) { /* set of actions in this situation */ }
          // logging transitions and actions when the condition is met.
          break;

      case 5: // it was the past
          // it has become current
          if(Y=1) { /* set of actions in this situation */ }
          if(Y=2) { /* set of actions in this situation */ }
          if(Y=3) { /* set of actions in this situation */ }
          if(Y=4) { /* set of actions in this situation */ }
          if(Y=5) { /* set of actions in this situation */ } //Loop//
          if(Y=6) { /* set of actions in this situation */ }
          if(Y=7) { /* set of actions in this situation */ }
          if(Y=8) { /* set of actions in this situation */ }
          if(Y=9) { /* set of actions in this situation */ }
          // logging transitions and actions when the condition is met.
          break;

      case 6: // it was the past
          // it has become current
          if(Y=1) { /* set of actions in this situation */ }
          if(Y=2) { /* set of actions in this situation */ }
          if(Y=3) { /* set of actions in this situation */ }
          if(Y=4) { /* set of actions in this situation */ }
          if(Y=5) { /* set of actions in this situation */ }
          if(Y=6) { /* set of actions in this situation */ } //Loop//
          if(Y=7) { /* set of actions in this situation */ }
          if(Y=8) { /* set of actions in this situation */ }
          if(Y=9) { /* set of actions in this situation */ }
          // logging transitions and actions when the condition is met.
          break;

      case 7: // it was the past
          //it has become current
          if(Y=1) { /* set of actions in this situation */ }
          if(Y=2) { /* set of actions in this situation */ }
          if(Y=3) { /* set of actions in this situation */ }
          if(Y=4) { /* set of actions in this situation */ }
          if(Y=5) { /* set of actions in this situation */ }
          if(Y=6) { /* set of actions in this situation */ }
          if(Y=7) { /* set of actions in this situation */ } //Loop//
          if(Y=8) { /* set of actions in this situation */ }
          if(Y=9) { /* set of actions in this situation */ }
          // logging transitions and actions when the condition is met.
          break;

      case 8: // it was the past
          // it has become current
          if(Y=1) { /* set of actions in this situation */ }
          if(Y=2) { /* set of actions in this situation */ }
          if(Y=3) { /* set of actions in this situation */ }
          if(Y=4) { /* set of actions in this situation */ }
          if(Y=5) { /* set of actions in this situation */ }
          if(Y=6) { /* set of actions in this situation */ }
          if(Y=7) { /* set of actions in this situation */ }
          if(Y=8) { /* set of actions in this situation */ } //Loop//
          if(Y=9) { /* set of actions in this situation */ }
          // logging transitions and actions when the condition is met.
          break;

      case 9: // it was the past
         // it has become current
          if(Y=1) { /* set of actions in this situation */ }
          if(Y=2) { /* set of actions in this situation */ }
          if(Y=3) { /* set of actions in this situation */ }
          if(Y=4) { /* set of actions in this situation */ }
          if(Y=5) { /* set of actions in this situation */ }
          if(Y=6) { /* set of actions in this situation */ }
          if(Y=7) { /* set of actions in this situation */ }
          if(Y=8) { /* set of actions in this situation */ }
          if(Y=9) { /* set of actions in this situation */ } //Loop//
          // logging transitions and actions when the condition is met.
          break;
     }
   MEGASTATUS=Y;
  }
```

**7.4. Audio Player Automaton**

Let us review a [Simple Audio Player](https://www.mql5.com/go?link=http://is.ifmo.ru/projects_en/player/ "http://is.ifmo.ru/projects_en/player/").

![](https://c.mql5.com/2/4/3tjqq.png)

This device can be in 6 states:

1. Ready;
2. No Track;
3. Playing;
4. Fast-Forward;
5. Rewind;
6. Pause.

The audio player control system is represented by an automaton. Buttons pressed are regarded as events having effect on the automaton. Transitions between tracks, playing, display control, etc. are output actions.

```
switch(STATUS)
  {
   case 0: //--- "Ready"
      if(Event == 3) { STATUS = 3; } //«>>» button pressed
      if(Event == 6) { STATUS = 1; } //Audio file not found
      if(Event == 1) { STATUS = 2; } //«PLAY» button pressed

      z1();  // Set the indicator to the initial state
      break;

   case 1: //--- "No Track"
      z6();  // Give the «No Track» message
      break;

   case 2: //--- "Playing"
      if(Event == 4) { STATUS = 4; } //«<<» button pressed
      if(Event == 5) { STATUS = 5; } //«PAUSE»( | | ) button pressed
      if(Event == 3) { STATUS = 3; } //«>>» button pressed
      if(Event == 2) { STATUS = 0; } //«STOP» button pressed
      z2(); // Playing
      break;

   case 3: //--- "Fast-Forward"
      z3();  // Next track
      { STATUS=2; }
      break;

   case 4: //--- "Rewind"
      z4(); // Previous track
      { STATUS=2; }
      break;

   case 5: //--- "Pause"
      if(Event == 5) { STATUS = 2; } //«PAUSE» button pressed
      if(Event == 1) { STATUS = 2; } //«PLAY» button pressed
      if(Event == 2) { STATUS = 0; } //«STOP» button pressed
      if(Event == 3) { STATUS = 3; } //«>>» button pressed
      if(Event == 4) { STATUS = 4; } //«<<» button pressed
      z5(); //Pause
      break;
  }
```

In theory, this automaton could contain 36 transition variants, but only 15 are really existent and all the details can be found in the description provided by the author.

### 8\. А.А. Shalyto's Recommendations Regarding Project Execution

Complete information on how to prepare and write project documentation can be found here [http://project.ifmo.ru/books/3](https://www.mql5.com/go?link=http://project.ifmo.ru/books/3 "http://project.ifmo.ru/books/3"), whereas in this article I am only going to give you a short extract:

01. The book by А.А. Shalyto "Logic Control. Methods of Hardware and Software Implementation of Algorithms. SPb.: Nauka, 2000", available at the specified website in the "Books" section can be taken as a prototype". It embodies a proper presentation of information as it was published by the oldest and the most reputable publishing company in Russia.
02. Introduction should give grounds for the relevance of the subject chosen, briefly state the problem under consideration and specify the programming language and operating system used in the project.
03. A detailed verbal description of the problem at hand should be provided in the "Problem Description" section along with figures, diagrams and screenshots clarifying the problem described.
04. When using the object-oriented programming, the "Design" section should include a diagram of classes. The main classes should be carefully described. It is advisable that a "Block diagram of the class" is prepared for every one of them aiming at presenting its interface and methods used along with indication of automata-based methods.
05. Three documents should be provided for every automaton in the "Automata" section: a verbal description, automaton link diagram and transition graph.
06. A verbal description should be quite detailed, however given the fact that the behavior of a complex automaton is difficult to be described in a clear way, it usually represents a "declaration of intent".
07. The automaton link diagram provides a detailed description of its interface. The left part of the diagram should feature:


    - data sources;
    - full name of every input variable;
    - full name of every event;
    - predicates with state numbers of other automata that are used in the given automaton as input actions. E.g. the Y8 == 6 predicate can be used which takes the value equal to one once the eighth automaton transitions to the sixth state);
    - input variables denoted as x with relevant indices;
    - events denoted as e with relevant indices;
    - variables for storing states of the automaton with the number N, denoted as YN.

The right part of the diagram should feature:
     - output variables denoted as z with relevant indices;
    - full name of every output variables;
    - events generated by the given automaton (if any);
    - full name of every generated event;
    - data receivers.
08. If complex computational algorithms are used in nodes or transitions, the "Computational Algorithms" section explains the choice of algorithms and provides their description (including mathematical description). These algorithms are designated by variables x and z, depending on whether the calculations are made at the input or output.
09. Peculiarities of the program implementation should be set forth in the "Implementation" section. It should particularly present a template for a formal and isomorphic implementation of automata. Implementations of automata should also be provided here.
10. "Conclusion" covers benefits and downsides of the completed project. It can also offer ways of improving the project.

### 9\. Conclusion

I encourage all of you to:

- explore this new approach to programming.
- implement this utterly new and extremely interesting approach to programming your ideas and trading strategies.

I hope that automata-based programming will:

- over time become the standard in programming and design for all traders and even MetaQuotes Language developers.
- be the basis in complex decision making when designing an EA.
- in the future develop into a new language - MetaQuotes Language 6 - supporting the automata-based programming approach and a new platform - MetaTrader 6.

If all trading developers follow this programming approach, the goal of creating a no-loss EA can be achieved. This first article is my attempt to show you a whole new outlet for creativity and research in the field of automata-based design and programming as an impulse to new inventions and discoveries.

And one more thing - I fully agree with the author's article and feel that it is important to provide it to you in a concise form (full text here [http://is.ifmo.ru/works/open\_doc/](https://www.mql5.com/go?link=https://www.codeproject.com/Articles/8043/New-Initiative-in-Programming-Foundation-for-Open "http://www.codeproject.com/Articles/8043/New-Initiative-in-Programming-Foundation-for-Open")):

### Why Source Codes are Not a Solution to Understanding Programs

The central issue in practical programming is the issue of understanding program codes. It is always beneficial to have source codes at hand but the problem is that this is often not sufficient. And additional documentation is usually needed in order to gain an understanding of a nontrivial program. This need grows exponentially as the amount of code increases.

Program code analysis aimed at restoring the original design decisions made by developers and understanding programs are two important branches of programming technology whose existence goes hand in hand with insufficiency of source codes for understanding programs.

Everyone who has ever been involved in a major software reconstruction project will always remember the feeling of helplessness and perplexity that comes when you first see a bunch of ill-documented (though not always badly written) source codes. The availability of source codes is not really of much help when there is no access to key developers. If the program is written in a relatively low-level language and is in addition ill-documented, all the main design decisions usually disperse in programming details and require reconstruction. In cases like that, the value of higher-level documentation, such as interface specification and architecture description, may outweigh the value of the source code itself.

The realization of the fact that source codes are inadequate for understanding programs gave rise to attempts to combine the code and a higher-level documentation.

If you miss the early stages of the project, the complexity and the amount of work will virtually 'lock away' the source codes from you, provided that there is no high-level documentation in place. Understanding the "prehistoric" code in the absence of the developers originally working on the project or adequate documentation allowing to sort out the relevant architectural decisions is probably one of the most difficult challenges programmers encounter».

### Why Programs Lack Design

So, while the absence of source codes can be bad, their availability can equally be poorly beneficial. What is still missing for a life "happy ever after"? The answer is simple - a detailed and accurate design documentation that includes program documentation as one of its components.

Bridges, roads and skyscrapers cannot normally be built without documentation at hand, which is not true about programs.

The situation that has come about in programming can be defined as follows: "If builders built buildings the way programmers wrote programs, then the first woodpecker that came along would destroy civilization."

Why is it that a good deal of detailed and clear design documentation is issued for hardware and can be relatively easily understood and modified by an average specialist even years after it was issued, but such documentation is either nonexistent for software or it is written in a purely formal way and a highly skilled specialist is required to modify it (if the developer is missing)?

Apparently, this situation can be explained as follows. First, development and manufacture of hardware are two different processes carried out by different organizations. Therefore, if the quality of documentation is poor, the development engineer will spend the rest of his life working in the 'plant' which is obviously not what he would wish for. When it comes to software development, the situation changes as in this case both the software developer and manufacturer are usually one and the same company and therefore, regardless of the list of documents, their contents will, as a rule, be quite superficial.

Second, hardware is 'hard', while software is 'soft'. It makes it easier to modify programs but does not give grounds for not issuing design documentation altogether. It is known that most programmers are pathologically reluctant to read, and all the more so to write, documentation.

The experience suggests that virtually none of newly qualified programmers, even the smartest ones, can prepare design documentation. And despite the fact that many of them took and passed long and complex courses in mathematics, it has almost no effect on their logic and rigor of writing documentation. They might use different notations for one and the same thing throughout the entire documentation (regardless of its size), thus calling it e.g. the bulb, the light bulb, the lamp or Lamp, writing it with a small or capital letter whenever they like. Imagine what happens when they give their fancy full scope!

Apparently, it happens due to the fact that when programming the compiler flags inconsistencies while design documentation is written without any prompts whatsoever.

The issue of the quality of software documentation is becoming an issue of increasing social significance. Software development is progressively becoming similar to show business with its strong profit motive. Everything is done in a mad rush, without thinking of what will become of the product in the future. Like show business, programming measures everything in terms of "profit and loss", rather than "good and bad". In the majority of cases, a good technology is not the one that is actually good but the one that pays.

The unwillingness to write design documentation is probably also associated with the fact that the more restricted (undocumented) the project is, the more indispensable the author.

Such work behavior unfortunately spreads out to the development of software for highly critical systems. It is largely due to the fact that programs are in most cases written, and not designed. "When designing, any technique more complicated than [CRC cards](https://en.wikipedia.org/wiki/Class-responsibility-collaboration_card "https://en.wikipedia.org/wiki/Class-responsibility-collaboration_card") or use case diagrams is considered to be too complex and is therefore not used. A programmer can always refuse to apply any given technology by reporting to the boss that he may not be able to meet the deadline".

This leads to situations where even the "users do not consider errors in the software something out of the ordinary".

It is presently widely thought that design and proper documentation shall be in place when it comes to large buildings, and not software.

In conclusion, it should be noted that such situation did not use to exist in programming in the past - when early large-scale computers were in use, programs were either designed or developed very carefully as in case of an error, the next attempt would normally take place as early as in a day. Thus, technical progress has led us to a less careful programming.

### References

Unfortunately, our problems and concerns cannot be traced on the website of the institute department where A.A. Shalyto works. They have their own problems and goals and are totally unfamiliar with, and unaware of, our concepts and definitions, hence no examples relevant to our subject.

The main books/textbooks by A.A. Shalyto:

01. Automata-Based Programming. [http://is.ifmo.ru/books/\_book.pdf](https://www.mql5.com/go?link=http://is.ifmo.ru/books/_book.pdf "http://is.ifmo.ru/books/_book.pdf")
02. Using Flow Graphs and Transition Graphs in Implementation of Logic Control Algorithms. [http://is.ifmo.ru/download/gsgp.pdf](https://www.mql5.com/go?link=http://is.ifmo.ru/download/gsgp.pdf "http://is.ifmo.ru/download/gsgp.pdf")
03. Automata-Based Programming. [http://is.ifmo.ru/works/\_2010\_09\_08\_automata\_progr.pdf](https://www.mql5.com/go?link=http://is.ifmo.ru/works/_2010_09_08_automata_progr.pdf "http://is.ifmo.ru/works/_2010_09_08_automata_progr.pdf")
04. Transformation of Iterative Algorithms into Automata-Based Algorithms. [http://is.ifmo.ru/download/iter.pdf](https://www.mql5.com/go?link=http://is.ifmo.ru/download/iter.pdf "http://is.ifmo.ru/download/iter.pdf")
05. Switch-Technology: Automata-Based Approach to Developing Software for Reactive Systems. [http://is.ifmo.ru/download/switch.pdf](https://www.mql5.com/go?link=http://is.ifmo.ru/download/switch.pdf "http://is.ifmo.ru/download/switch.pdf")
06. Automata-Based Program Design. Algorithmization and Programming of Logic Control Problems. [http://is.ifmo.ru/download/app-aplu.pdf](https://www.mql5.com/go?link=http://is.ifmo.ru/download/app-aplu.pdf "http://is.ifmo.ru/download/app-aplu.pdf")
07. Using Genetic Algorithm to Design Autopilot for a Simplified Helicopter Model. [http://is.ifmo.ru/works/2008/Vestnik/53/05-genetic-helicopter.pdf](https://www.mql5.com/go?link=http://is.ifmo.ru/works/2008/Vestnik/53/05-genetic-helicopter.pdf "http://is.ifmo.ru/works/2008/Vestnik/53/05-genetic-helicopter.pdf")
08. Explicit State-Based Programming. [http://is.ifmo.ru/download/mirpk1.pdf](https://www.mql5.com/go?link=http://is.ifmo.ru/download/mirpk1.pdf "http://is.ifmo.ru/download/mirpk1.pdf")
09. Algorithmization and Programming for Logic Control and Reactive Systems. [http://is.ifmo.ru/download/arew.pdf](https://www.mql5.com/go?link=http://is.ifmo.ru/download/arew.pdf "http://is.ifmo.ru/download/arew.pdf")
10. Object-Oriented Approach to Automata-Based Programming. [http://is.ifmo.ru/works/ooaut.pdf](https://www.mql5.com/go?link=http://is.ifmo.ru/works/ooaut.pdf "http://is.ifmo.ru/works/ooaut.pdf")
11. Graphical Notation for Inheritance of Automata-Based Classes. [http://is.ifmo.ru/works/\_12\_12\_2007\_shopyrin.pdf](https://www.mql5.com/go?link=http://is.ifmo.ru/works/_12_12_2007_shopyrin.pdf "http://is.ifmo.ru/works/_12_12_2007_shopyrin.pdf")
12. Programming in... 1 (One) Minute. [http://is.ifmo.ru/progeny/1minute/?i0=progeny&i1=1minute](https://www.mql5.com/go?link=http://is.ifmo.ru/progeny/1minute/?i0=progeny&i1=1minute "http://is.ifmo.ru/progeny/1minute/?i0=progeny&i1=1minute")

Projects:

1. Modeling Operation of ATMs. [http://is.ifmo.ru/unimod-projects/bankomat/](https://www.mql5.com/go?link=http://is.ifmo.ru/unimod-projects/bankomat/ "http://is.ifmo.ru/unimod-projects/bankomat/") [https://www.mql5.com/go?link=http://is.ifmo.ru/unimod-projects/bankomat/](https://www.mql5.com/go?link=http://is.ifmo.ru/unimod-projects/bankomat/ "https://www.mql5.com/go?link=http://is.ifmo.ru/unimod-projects/bankomat/")
2. Modeling Nuclear Reactor Control Process. [http://is.ifmo.ru/projects/reactor/](https://www.mql5.com/go?link=http://is.ifmo.ru/projects/reactor/)
3. Elevator Control System. [http://is.ifmo.ru/projects/elevator/](https://www.mql5.com/go?link=http://is.ifmo.ru/projects/elevator/ "https://www.mql5.com/go?link=http://is.ifmo.ru/projects/elevator/")
4. Automata-Based Development of Coffee Maker Control System. [http://is.ifmo.ru/projects/coffee2/](https://www.mql5.com/go?link=http://is.ifmo.ru/projects/coffee2/ "https://www.mql5.com/go?link=http://is.ifmo.ru/projects/coffee2/")
5. Design and Research of Automata for Driving. [http://is.ifmo.ru/projects/novohatko/](https://www.mql5.com/go?link=http://is.ifmo.ru/projects/novohatko/ "http://is.ifmo.ru/projects/novohatko/")
6. Modeling a Digital Camera Using Automata-Based Programming. [http://project.ifmo.ru/shared/files/200906/5\_80.pdf](https://www.mql5.com/go?link=http://project.ifmo.ru/shared/files/200906/5_80.pdf "http://project.ifmo.ru/shared/files/200906/5_80.pdf")
7. Using Automata-Based Programming to Model a Multi-Agent System for Unmanned Vehicles. [http://project.ifmo.ru/shared/files/200906/5\_41.pdf](https://www.mql5.com/go?link=http://project.ifmo.ru/shared/files/200906/5_41.pdf "http://project.ifmo.ru/shared/files/200906/5_41.pdf")
8. Visual Rubik's Cube Solution System. [http://is.ifmo.ru/projects/rubik/](https://www.mql5.com/go?link=http://is.ifmo.ru/projects/rubik/ "https://www.mql5.com/go?link=http://is.ifmo.ru/projects/rubik/")

and other interesting articles and projects: [http://project.ifmo.ru/projects/](https://www.mql5.com/go?link=http://project.ifmo.ru/projects/ "http://project.ifmo.ru/projects/"), [http://is.ifmo.ru/projects\_en/](https://www.mql5.com/go?link=http://is.ifmo.ru/projects_en/ "http://is.ifmo.ru/projects_en/") and [http://is.ifmo.ru/articles\_en/](https://www.mql5.com/go?link=http://is.ifmo.ru/articles_en/ "http://is.ifmo.ru/articles_en/").

### P.S.

The number of possible different events of a [Rubik's Cube](https://en.wikipedia.org/wiki/Rubik "https://en.wikipedia.org/wiki/Rubik") is (8! × 38−1) × (12! × 212−1)/2 = 43 252 003 274 489 856 000. But this number does not take into account that central squares can have different orientations.

Thus, considering the orientations of center faces, the number of events becomes 2048 times as large, i.e. 88 580 102 706 155 225 088 000.

The Forex market and exchange do not have so many variants of succession of events but problems associated with them can easily be solved in 100-200 steps using this programming paradigm. It is true! The market and EAs are in constant competition. It is like playing [chess](https://en.wikipedia.org/wiki/Computer_chess "https://en.wikipedia.org/wiki/Computer_chess") where nobody knows the upcoming moves of the opponent (just like us). However there are impressive computer programs, such as [Rybka](https://en.wikipedia.org/wiki/Rybka "https://en.wikipedia.org/wiki/Rybka") (very powerful chess engine) designed based on alpha-beta pruning algorithms.

May these successes of others in other areas of programming give you energy and commitment to our work! Although, we certainly all know that we know nothing.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/446](https://www.mql5.com/ru/articles/446)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/7913)**
(61)


![Denis Lazarev](https://c.mql5.com/avatar/avatar_na2.png)

**[Denis Lazarev](https://www.mql5.com/en/users/lazarev-d-m)**
\|
3 Dec 2012 at 23:56

**Heroix:**

No wonder they don't want to publish.

Not only are there no formulas, it doesn't make much sense.


![Dale Forsyth](https://c.mql5.com/avatar/2017/12/5A341D3C-9040.JPG)

**[Dale Forsyth](https://www.mql5.com/en/users/5734354d)**
\|
5 Feb 2018 at 13:06

Это сообщение всем, кто сыграл определенную роль в эволюции этой статьи, особенно автора.

Я не могу полагаться на 100% на текст или лингвистику, чтобы я мог передать свой ответ и реакцию на статью, которая была опубликована здесь. У меня так много, что я хотел бы выразить. Если я начну пытаться объяснить мои чувства с помощью текста, тогда я буду ей на половину жизни и все равно не смогу точно передать свое сообщение. С учетом сказанного мой выбор заключается не в том, чтобы сильно полагаться на текст и слова. Моя отзывчивая энергия найдет свой путь ко всем тем, кто сыграл определенную роль в событиях, ведущих к моему открытию.

Богатство мудрости, ожидающее здесь, нашло меня, и я понял с удивительной ясностью. Это в надежных руках.

Спасибо.

Большое спасибо.

С уважением,

Дол.  :-)

This is a message to all who have played a part in the evolution of this article, particularly the author.

I am not able to rely 100% on text or linguistics in order for me to convey my response and reaction to the article that has been published here. I have so much that i would like to express. If I begin to attempt to further explain my feelings through text, then i will be her for half a life time and still not be able to accurately convey my message. With that said my choice is not to rely heavily on text and words. My responsive energy will find its way to all of those that have played a part in events leading up to my discovery.

The wealth of wisdom waiting here has found me and I have understood with amazing clarity. It is in good hands.

Thank you.

Thank you very much.

Kind regards,

Dale.:-)

![Roann Manicad](https://c.mql5.com/avatar/2018/9/5BA85F6D-1091.png)

**[Roann Manicad](https://www.mql5.com/en/users/roannmanicad)**
\|
28 Sep 2018 at 04:27

Nice


![Mong Kol](https://c.mql5.com/avatar/2020/8/5F34653C-9848.png)

**[Mong Kol](https://www.mql5.com/en/users/33505028)**
\|
12 Aug 2020 at 21:57

<Deleted>

![Carlos Camargo](https://c.mql5.com/avatar/avatar_na2.png)

**[Carlos Camargo](https://www.mql5.com/en/users/camargo.cr)**
\|
2 Aug 2021 at 21:40

Hi,

Could, please, someone (specially professionals from Saint Petersburg University!) point out good resources (articles or books in English) on Switch-Technology (or Automata-based Programming) as originally teached by Prof. Dr. Shalyto on Saint Petersburg University? (I found some divulgation articles, but lack of in depth on methodology, techniques and methods, as well as tools, to adopt in real life programming.)

Thanks,

![Quick Start: Short Guide for Beginners](https://c.mql5.com/2/0/start_ava.png)[Quick Start: Short Guide for Beginners](https://www.mql5.com/en/articles/496)

Hello dear reader! In this article, I will try to explain and show you how you can easily and quickly get the hang of the principles of creating Expert Advisors, working with indicators, etc. It is beginner-oriented and will not feature any difficult or abstruse examples.

![Exploring Trading Strategy Classes of the Standard Library - Customizing Strategies](https://c.mql5.com/2/0/sl_article.png)[Exploring Trading Strategy Classes of the Standard Library - Customizing Strategies](https://www.mql5.com/en/articles/488)

In this article we are going to show how to explore the Standard Library of Trading Strategy Classes and how to add Custom Strategies and Filters/Signals using the Patterns-and-Models logic of the MQL5 Wizard. In the end you will be able easily add your own strategies using MetaTrader 5 standard indicators, and MQL5 Wizard will create a clean and powerful code and fully functional Expert Advisor.

![Interview with Achmad Hidayat (ATC 2012)](https://c.mql5.com/2/0/Achmad-ava.png)[Interview with Achmad Hidayat (ATC 2012)](https://www.mql5.com/en/articles/560)

Throughout the entire duration of the Automated Trading Championship 2012, we will be providing live coverage of the events - hot reports and interviews every week. This report spotlights Achmad Hidayat (achidayat), the participant from Indonesia. On the first day of the Championship his Expert Advisor secured its position in the third ten, which is a good start. Achmad has sparked our interest with his active participation in the MQL5 Market. He has already published over 20 products thus far.

![Fundamentals of Statistics](https://c.mql5.com/2/0/statistic.png)[Fundamentals of Statistics](https://www.mql5.com/en/articles/387)

Every trader works using certain statistical calculations, even if being a supporter of fundamental analysis. This article walks you through the fundamentals of statistics, its basic elements and shows the importance of statistics in decision making.

[We've created a channel for MQL5 developersFollow MQL5.community on social media and be the first to receive important updatesLearn more![](https://www.mql5.com/ff/sh/a83xrgctr82w45z9z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=pwgdbvtemvkwsfltqonysypfvtrtufji&s=e99a66a1660cd810b1edbac65597df695e2c2220d1e937834f402f9aeabd4289&uid=&ref=https://www.mql5.com/en/articles/446&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5048972066704695910)

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