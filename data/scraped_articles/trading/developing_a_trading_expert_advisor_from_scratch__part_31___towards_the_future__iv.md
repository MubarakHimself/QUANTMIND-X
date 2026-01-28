---
title: Developing a trading Expert Advisor from scratch (Part 31): Towards the future (IV)
url: https://www.mql5.com/en/articles/10678
categories: Trading, Trading Systems, Indicators
relevance_score: 6
scraped_at: 2026-01-22T20:45:14.057115
---

[![](https://www.mql5.com/ff/si/6pp0j40fqxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=luckhiizjxvmvgigcufevttapwwrwbld&s=08cd1d929f27358481aded3c1c5f4e75a9bd5f52c477127afef2a5c532aec5c5&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=oquctnxshknhwrwghuvbzcvihdwmlcmt&ssn=1769103912254182774&ssn_dr=0&ssn_sr=0&fv_date=1769103912&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F10678&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Developing%20a%20trading%20Expert%20Advisor%20from%20scratch%20(Part%2031)%3A%20Towards%20the%20future%20(IV)%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176910391218399354&fz_uniq=5051658117776790566&sv=2552)

MetaTrader 5 / Trading


### Introduction

After removing Chart Trade from the EA in the article " [Development of a trading EA from scratch (Part 29)](https://www.mql5.com/en/articles/10664)", we turned the Chart Trade panel into an indicator. How to do this, as well as how to adjust and maintain functions required to keep the indicator working, is described in [Part 30](https://www.mql5.com/en/articles/10653). This is one of the possible approaches, although there are actually other ways, with their advantages and disadvantages, but we will consider them another time.

So, we still have something left to remove from the EA. We will remove it now, and the article will be the last one in this series. The thing to remove is the sound system. This can be confusing if you haven't followed previous articles.

In order to understand what the whole process will be like (since it includes many things that need an explanation), we will use almost the same model based on the previous article. This will make the explanation simple and understandable, even for those who are not professional programmers. We will also complicate the system a bit, just to spice things up a bit.

This time the new part is the sound system. It will cease to be part of the EA. But this will provide many benefits in the future. However, let's not rush, because what's important is to understand what is going to happen here. The article will be relatively short, but interesting.

### Introduction of a sound service

All these changes can drive you crazy. But believe me, the idea is not to drive you crazy, well maybe bewilder a little, but to show how sometimes small changes can make a big difference and make using the MetaTrader 5 platform much more enjoyable. At the same time, you will see how all these actions enable the modulation of things.

This way you can choose what you need and what not. If something is really used, you will be able to add improvements to it later to make the feature even more useful and enjoyable. This will not require big changes or reprogramming of what was created a while ago. The idea is to **ALWAYS REUSE**.

One of such features is the sound system. It may seem that leaving this system inside the EA is a good idea. In a sense, this system does not interfere with the functioning of the EA as a whole. But if we remove it from the EA and implement some communication between them, you will see that it is possible to use sound alerts in a very simple way, as if it's a sound library. This solution will be very useful.

There is no point in placing an alert system only within an EA. It can be useful to have a sound system in indicators or even in scripts that run at specific times. This will be very helpful in analysis. Thus, the MetaTrader 5 platform can become a true monster in terms of analysis, where you can do huge calculations to better analyze the market at very specific moments, be it position entering or closing. All this can be done with a minimum of effort.

One may say: "But I can add all the sounds to an MQH file (Header File), embed them in executables and get the required behavior." Yes, you can. But think about the following scenario: Over time, this MQH file will grow, and as it does, some older programs may become incompatible with this header file (MQH). If you have to recompile such old files, you will run into problems. And if you create a modular system in which there is a communication protocol between processes, you can expand the functionality of the platform while maintaining compatibility with older programs.

This is the reason for these changes: to show how you can create and use any of the possible paths. And I'm showing this by taking things out of the EA, while keeping things as close to the original behavior as possible.

In the previous article, I showed how to recreate a Chart Trade so that it behaves the same as when it was integrated into the EA. However, having removed it from the EA, it was necessary to create some way for it to continue working in the same mode. The way I showed you is one of many possible, and although it is not the best, it works. Every solution requires a proper understanding of how things work in general. Sometimes, being limited to just one idea model does not help in solving specific situations, quite the contrary. Precisely due to lack of knowledge, many think that it is not possible to do something or they say that the system is limited, when in fact, the limitation lies in the lack of knowledge of the person responsible for planning and implementing the solution.

We saw this when implementing the order system without using any structure to store the data. Many thought that it was something impossible to do, that there is no way to do such things. But I showed that it was possible. The important thing is to know and understand what you are doing. The first step is to know the limitations of each type of solution.

So, let's learn how to make the sound system as modular as possible, bearing in mind that we will be expanding its functionality as the system grows further.

First of all, we will not touch the C\_Sound class except for the cases when we need to expand the functionality. So, there will be no big changes in this class. In fact, at this stage this class will remains unchanged, however, we need to make small additions to the system. The first of them is the header file shown below:

```
#property copyright "Daniel Jose"
//+------------------------------------------------------------------+
#define def_GlobalVariableAlert "Sound Alert"
//+------------------------------------------------------------------+
```

You might think that we are going to use this file in the EA, but no... the EA will not use this file, at least not yet. It will use another file which we will see later.

After that, we can create a file that will be the sound service. It is shown in the code below:

```
#property service
#property copyright "Daniel Jose"
//+------------------------------------------------------------------+
#include <NanoEA-SIMD\Auxiliar\C_Sounds.mqh>
#include <NanoEA-SIMD\Interprocess\Sound.mqh>
//+------------------------------------------------------------------+
void OnStart()
{
        union u00
        {
                double value;
                struct s00
                {
                        uint    i0,
                                i1;
                }info;
        }u_local;
        C_Sounds Sound;

        while (!IsStopped())
        {
                Sleep(500);
                if (GlobalVariableGet(def_GlobalVariableAlert, u_local.value))
                {
                        GlobalVariableDel(def_GlobalVariableAlert);
                        if (u_local.info.i1 == 0) Sound.PlayAlert((C_Sounds::eTypeSound)u_local.info.i0);
                        else Sound.PlayAlert(u_local.info.i1);
                }
        }
}
//+------------------------------------------------------------------+
```

This service watches MetaTrader 5 global variables. It will play the specified sound as soon as the variable whose name is declared in the header file is launched by any script, EA or indicator, no matter what it is and when it happens.

All you need to do is specify the index of the file to be played. Based on the above structure, you will be able to play a total of 4,294,967,295 different sound, which is the number for only for external files. You can have the same number of internal sounds, so you can do a lot of things.

In order for the system to know which sound type to play, it checks the value of the **u\_local.info.i1** variable: if the value is 0, then the sound to be reproduced will be embedded in the service file, and the index of the sound will be indicated by the **u\_local.info.i0** variable, but this value represents the enumerator inside the C\_Sound class.

Now we can compile the service and run it. As soon as the above conditions are satisfied, the service will perform its work, remembering that when the global variable is captured by the service, it will be removed so that it can be used at another time.

Before we go any further, let's think a little. Unlike the Chart Trade indicator, which will only communicate with the EA, the sound system can communicate with any type of program in the MetaTrader 5 platform. To play the desired sound, you need to set the value of the variable, which will always be double.

You may think it's easy but try it and you'll see it's not. Furthermore, you will have to keep creating the global variable with the correct name every time. So, you will have to do a lot of work every time you want to play a previously saved sound.

But there is a practical solution that avoids all this hassle. Because it's quite nice, we'll use this solution at its most basic level at this early stage. To see how it is done, let's move on to the next topic.

### Creating a library to access the sound service

The reason for creating a library is that it will make our lives easier in some way. No matter how, but it will make our lives easier. In the previous topic, I mentioned that when a program accesses the sound service, we don't need to know the name of the global variable, which gives access to the service. As strange as it may sound, the best way to pass information between processes is by adding a layer to the system. This layer is the library.

These libraries will "hide" the complexity of data modeling between processes, so that you no longer worry about which form the modeling should take. You only take care about the calls themselves and the expected results.

There are only 2 concerns when creating a library:

1. Clearly declare the functions that will be exported.
2. Hide the complexity of the internal modeling as much as possible, so that the library user does not need to know what is happening. The user should only see the data coming in and the result coming out.

So, any procedure or function within a library is designed to have a very simple behavior from the point of view of the user. But internally, there can be an extremely complex level of operations leading to the final results. But the programmer who will be using the library does not need to know what is happening inside it. It is important to know that the results are provided correctly.

So, let's take a look at our library that will hide the data modeling used in the sound service. Every program should report two things: the first is whether the sound is internal or external; the second is the index of the sound. Sounds complicated? Let's see the code of these calls inside the library:

```
void Sound_WAV(uint index) export { Sound(0, index); }
void Sound_Alert(uint index) export { Sound(index, 0); }
```

These two functions hide any complexity in data modeling. Note that we are using the keyword [export](https://www.mql5.com/en/docs/basis/function/export) which instructs the compiler to create a symbolic link to these functions. They are actually procedures because they don't return any value. This way they will be visible outside the file, as if this file were a DLL.

But if you look over the code, you won't find any function called **Sound**. Where is it? It is in the library itself, but it won't be visible outside of it. See the full library code below:

```
//+------------------------------------------------------------------+
#property library
#property copyright "Daniel Jose"
//+------------------------------------------------------------------+
#include <NanoEA-SIMD\Interprocess\Sound.mqh>
//+------------------------------------------------------------------+
void Sound_WAV(uint index) export { Sound(0, index); }
//+------------------------------------------------------------------+
void Sound_Alert(uint index) export { Sound(index, 0); }
//+------------------------------------------------------------------+
void Sound(uint value00, uint value01)
{
        union u00
        {
                double value;
                struct s00
                {
                        uint    i0,
                                i1;
                }info;
        }u_local;

        u_local.info.i0 = value00;
        u_local.info.i1 = value01;
        GlobalVariableTemp(def_GlobalVariableAlert);
        GlobalVariableSet(def_GlobalVariableAlert, u_local.value);
}
//+------------------------------------------------------------------+
```

Note that the Sound procedure will contain all the necessary complexity required to assemble the adequate value, so that the service can execute the task, which a script, indicator or EA requests. But instead of placing this code inside the program that will access the service, we will only use simplified calls, which makes program debugging more comfortable and less tiring.

To understand how this works, let's look at an example script:

```
#property copyright "Daniel Jose"
#property script_show_inputs
#import "Service_Sound.ex5"
        void Sound_WAV(uint);
        void Sound_Alert(uint);
#import
//+------------------------------------------------------------------+
input uint value00 = 1;         //Internal sound service...
input uint value01 = 10016;     //Sound in WAV file...
//+------------------------------------------------------------------+
void OnStart()
{
        Sound_WAV(value01);
        Sound_Alert(value00);
}
//+------------------------------------------------------------------+
```

Look at the code above. It is not necessary to know what type of communication is implemented, where and when the sound event will happen — it can happen anywhere, within the platform, within the operating system , or even remotely, it does not matter. The only things we need to inform is if the sound is internal or external to the system and its index.

Now, before continuing, I want you to do one experiment. Swap functions. In this case we run Sound\_WAV and then Sound\_Alert. Run it and see the result. Next, change the order: run Sound\_Alert, then Sound\_WAV and see the result. For those who don't understand, the code inside the OnStart event would look like this in the first situation:

```
void OnStart()
{
        Sound_WAV(value01);
        Sound_Alert(value00);
}
```

And like this in the second case:

```
void OnStart()
{
        Sound_Alert(value00);
        Sound_WAV(value01);
}
```

While it may seem silly, this experiment is necessary in order to understand a few things. Don't ignore it, it will be interesting to see the results.

Now that we have seen what we should add to our programs in order to be able to play sounds, we simply need to add the following code:

```
#import "Service_Sound.ex5"
        void Sound_WAV(uint);
        void Sound_Alert(uint);
#import
```

Whenever you need to play a sound, just use the right function with the right value, without worrying about how it will be done. The system itself will make sure that everything works perfectly. In our EA, the code will look like this:

```
// ...

#import "Service_Sound.ex5"
        void Sound_WAV(uint);
        void Sound_Alert(uint);
#import
//+------------------------------------------------------------------+
#include <NanoEA-SIMD\Trade\Control\C_IndicatorTradeView.mqh>
#include <NanoEA-SIMD\Interprocess\Sound.mqh>

// ...
```

The question arises: What does the highlighted code do there? Can't we simply use the library? Yes, but we can use an enumeration to identify the numeric codes if the sounds, as it was done before, and unless you are using a very low number of sounds or alerts, it can be very difficult to understand what each one represents just by looking at the code. For this reason, the header file Sound.mqh received an addition which is highlighted in the code below:

```
#property copyright "Daniel Jose"
//+------------------------------------------------------------------+
#define def_GlobalVariableAlert "Sound Alert"
//+------------------------------------------------------------------+
enum eTypeSound {TRADE_ALLOWED, OPERATION_BEGIN, OPERATION_END};
//+------------------------------------------------------------------+
```

So, we can end up with code like this:

```
int OnInit()
{
        if (!TerminalInfoInteger(TERMINAL_TRADE_ALLOWED))
        {
                Sound_Alert(TRADE_ALLOWED);
                return INIT_FAILED;
        }

// ... The rest of the function
```

It is much more representative than the same code that uses indexes instead of enumerations:

```
int OnInit()
{
        if (!TerminalInfoInteger(TERMINAL_TRADE_ALLOWED))
        {
                Sound_Alert(0);
                return INIT_FAILED;
        }

// ... Rest of the code
```

Which one is easier to understand?

After all this work, you will have the flow of information in the platform as is shown in the figure below:

![](https://c.mql5.com/2/46/001__2.png)

As you can see, no matter who provides the signal, we will always have the same destination.

### Conclusion

Although it may not seem like a big deal, what has been shown in this article goes a long way towards increasing the usability of your code, programs and information. As you start to program less and less, and become more and more productive, at the same time your code becomes more secure and stable, as reuse and testing are repeated in many different scenarios.

Here we have seen another path, which is different from the one seen in the previous article, but even so, this one can be improved a lot, giving us a plethora of new possibilities. But we will see this in another series, where you will learn how to make your programs and projects in MetaTrader 5 much more modular, with a much higher level of security, usability and stability than any of the methods shown here.

But the main and most important thing is to know how to design and use some different solutions, as there are cases in which one solution will be better than another, for one reason or another.

All codes are available in the attached file. For those who are not very used to this way of programming, using libraries, I advise to study this phase of EA development well. Do not put off till tomorrow what you can do today, because tomorrow may not come for you the way you expect.

This article completes this EA development stage. Soon I will present another type of material, focused on another type of situation, where the level of complexity involved is much higher, but nevertheless considerably more interesting. Big hug to all and see you later.

Translated from Portuguese by MetaQuotes Ltd.

Original article: [https://www.mql5.com/pt/articles/10678](https://www.mql5.com/pt/articles/10678)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/10678.zip "Download all attachments in the single ZIP archive")

[EA\_-\_j\_Parte\_31\_f.zip](https://www.mql5.com/en/articles/download/10678/ea_-_j_parte_31_f.zip "Download EA_-_j_Parte_31_f.zip")(14533.52 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Market Simulation (Part 09): Sockets (III)](https://www.mql5.com/en/articles/12673)
- [Market Simulation (Part 08): Sockets (II)](https://www.mql5.com/en/articles/12672)
- [Market Simulation (Part 07): Sockets (I)](https://www.mql5.com/en/articles/12621)
- [Market Simulation (Part 06): Transferring Information from MetaTrader 5 to Excel](https://www.mql5.com/en/articles/11794)
- [Market Simulation (Part 05): Creating the C\_Orders Class (II)](https://www.mql5.com/en/articles/12598)
- [Market Simulation (Part 04): Creating the C\_Orders Class (I)](https://www.mql5.com/en/articles/12589)
- [Market Simulation (Part 03): A Matter of Performance](https://www.mql5.com/en/articles/12580)

**[Go to discussion](https://www.mql5.com/en/forum/438300)**

![Category Theory in MQL5 (Part 1)](https://c.mql5.com/2/50/Category-Theory-avatar-001.png)[Category Theory in MQL5 (Part 1)](https://www.mql5.com/en/articles/11849)

Category Theory is a diverse and expanding branch of Mathematics which as of yet is relatively uncovered in the MQL community. These series of articles look to introduce and examine some of its concepts with the overall goal of establishing an open library that attracts comments and discussion while hopefully furthering the use of this remarkable field in Traders' strategy development.

![Magic of time trading intervals with Frames Analyzer tool](https://c.mql5.com/2/50/Frames_Analyzer_avatar.png)[Magic of time trading intervals with Frames Analyzer tool](https://www.mql5.com/en/articles/11667)

What is Frames Analyzer? This is a plug-in module for any Expert Advisor for analyzing optimization frames during parameter optimization in the strategy tester, as well as outside the tester, by reading an MQD file or a database that is created immediately after parameter optimization. You will be able to share these optimization results with other users who have the Frames Analyzer tool to discuss the results together.

![Population optimization algorithms: Ant Colony Optimization (ACO)](https://c.mql5.com/2/50/popular_algorithm_ant_colony_optimization_avatar.png)[Population optimization algorithms: Ant Colony Optimization (ACO)](https://www.mql5.com/en/articles/11602)

This time I will analyze the Ant Colony optimization algorithm. The algorithm is very interesting and complex. In the article, I make an attempt to create a new type of ACO.

![Neural networks made easy (Part 31): Evolutionary algorithms](https://c.mql5.com/2/50/Neural_networks_made_easy_021__1.png)[Neural networks made easy (Part 31): Evolutionary algorithms](https://www.mql5.com/en/articles/11619)

In the previous article, we started exploring non-gradient optimization methods. We got acquainted with the genetic algorithm. Today, we will continue this topic and will consider another class of evolutionary algorithms.

[Need a reliable hosting solution for your robots?Contact your broker and find out about available Sponsored MetaTrader VPS offeringsLearn more![](https://www.mql5.com/ff/sh/0pw0dk81s56qy774z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=vljwvezfjkfbvviocwskggexlvgykvob&s=70cf8e354b9a125332533ffb65d7365abe8dde5b5c1ede9caac479a9e9df4f25&uid=&ref=https://www.mql5.com/en/articles/10678&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5051658117776790566)

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