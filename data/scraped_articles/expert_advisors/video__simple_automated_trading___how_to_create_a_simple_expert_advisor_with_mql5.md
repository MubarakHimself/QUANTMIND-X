---
title: Video: Simple automated trading – How to create a simple Expert Advisor with MQL5
url: https://www.mql5.com/en/articles/10954
categories: Expert Advisors
relevance_score: 9
scraped_at: 2026-01-22T17:43:02.655052
---

[![](https://www.mql5.com/ff/sh/a27a2kwmtszm2m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
MQL5 Channels - Messenger for traders\\
\\
Subscribe to traders' channels or create your own.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=vpcudokyepxfrcxrpjcktglhsjlemtza&s=f08ad2c1289e29bd5630f1ef977aef297d5cdbfcb686faed4a4b0f1e276d3c4a&uid=&ref=https://www.mql5.com/en/articles/10954&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5049325314879891826)

MetaTrader 5 / Expert Advisors


### Part 1 – How to create a simple Expert Advisor

LEARN MQL5 TUTORIAL BASICS - HOW TO CREATE A SIMPLE EXPERT ADVISOR - YouTube

[Photo image of MQL5 Tutorial](https://www.youtube.com/channel/UCokIBdJXNOSOeYkKDvENWYA?embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F10954)

MQL5 Tutorial

27.5K subscribers

[LEARN MQL5 TUTORIAL BASICS - HOW TO CREATE A SIMPLE EXPERT ADVISOR](https://www.youtube.com/watch?v=9GVduCy1CgQ)

MQL5 Tutorial

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

[Watch on](https://www.youtube.com/watch?v=9GVduCy1CgQ&embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F10954)

0:00

0:00 / 1:41

•Live

•

A so-called Expert Advisor is what we are looking at right now. An Expert Advisor is an automated application that can operate within MetaTrader and can open and close positions on its own.

In this video, we will learn how to create an Expert Advisor in its most basic form. Please click on the small button here or press F4 in MetaTrader to open the MetaEditor window. After that, click on "File/ New/ Expert Advisor (template)" from the template, "Continue," and I'll call this version "SimpleExpertAdvisor." After that, click on "Continue," "Continue," and "Finish," and we're finished!

This version is readable for humans, and when we compile it, we create a readable version for the MetaTrader.  It's already available here, so let's open a new chart window, and we can drag our new file onto the chart. But it currently lacks logic, which we will add in the videos that will follow this one.

### Part 2 – What are functions

LEARN MQL5 TUTORIAL BASICS - 2 WHAT ARE FUNCTIONS - YouTube

[Photo image of MQL5 Tutorial](https://www.youtube.com/channel/UCokIBdJXNOSOeYkKDvENWYA?embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F10954)

MQL5 Tutorial

27.5K subscribers

[LEARN MQL5 TUTORIAL BASICS - 2 WHAT ARE FUNCTIONS](https://www.youtube.com/watch?v=zYdt_HnGaFY)

MQL5 Tutorial

Search

Watch later

Share

Copy link

Info

Shopping

Tap to unmute

If playback doesn't begin shortly, try restarting your device.

You're signed out

Videos you watch may be added to the TV's watch history and influence TV recommendations. To avoid this, cancel and sign in to YouTube on your computer.

CancelConfirm

Share

Include playlist

An error occurred while retrieving sharing information. Please try again later.

0:00

0:00 / 4:11

•Live

•

MQL5 uses functions to automate things, and we already have a few functions in our Expert Advisor template. The "OnInit" function is the expert initialization function, and it will run only once when you drop your Expert Advisor on a chart. It does have a return value, so you could check if the "OnInit" function was successful.

It will return one of these values here, which will tell us whether or not the "init" process worked. We also have a function called "OnDeinit" that cleans up before the Expert Advisor is closed. It doesn't return anything, which is what "void" means, "void" is used when something doesn't return a value.

In this case, the reason is very simple: "OnDeinit" is the last thing that will run in our program, and there's nothing we can return to because the program is closed after that. There is the "OnTick" function. This function runs whenever the price changes on the chart. Most of the coding logic would be triggered by the "OnTick" function, and whenever the price changes on one of your charts, everything between these two brackets will be run. We can now get rid of the rest of the template.

For our simple example, we only need the "OnTick" function, which will be used to call another function and show the local time. When we want to see something on our chart, we use the "Comment" function. This one will show a user value in the top left corner of the chart. It takes so-called parameters in round brackets, which could be a text like "Hello," and this is already a complete Expert Advisor.

We could now compile it, click the button here to go back to MetaTrader, and drag the "SimpleExpertAdvisor" onto the chart. When I click the "OK" button to confirm, it will show the text "Hello" that we set up here. But the "Comment" function can do more.

We wanted to show the local time, so let's say "The local time is." With a comma, we can pass another parameter, which in our case is "TimeLocal." Let's recompile the code, and now the Expert Advisor says "The local time is," and the local time shows up right away on our chart.

In this video, you learned about built-in functions like the "OnTick" function, which is called whenever the price changes. You also learned how to use the "Comment" function to output a text followed by a calculated value. For example, you used a few lines of MQL5 code to make MetaTrader 5 print out "The local time is" followed by the calculated time right on your chart.

### Part 3 – How to use the Strategy Tester

YouTube

We want to learn how to use the Strategy Tester in this video. This is a strategy test, which is also called a "back test." We use it to trade an Expert Advisor based on historical data to see if it makes money or not. Let's find out how to do that. Last time, we made a simple template that printed out the text "Hello MQL5". Now, we want to do a strategy test or back test with this simple example.

To do this, we click on "View/Strategy Tester" or press CTRL and R. Now, you should see this "Strategy Tester" panel here. We can choose a file, and we'll start with the simple Expert Advisor (SimpleExpertAdvisor) that we made last time.

We're going to use the currency pair Australian Dollar vs. Canadian Dollar (AUDCAD) for 1 minute and we want to pick a custom period, in our case the year 2017. One of the benefits of MetaTrader 5 is that you no longer have to download historical data. If you choose a time period for which you don't have historical data, this will happen automatically.

All of this happens in the background. This setting is for the quality. It's called OHLC, which stands for "open, high, low, and close." If you move your mouse over a candle, you'll see its open, high, low, and close prices.

This is what that looks like. A lot of people don't know that every tick for a candle that has more than one price change is simulated. Even if you choose "Every tick" here, if this big candle here was a 24 hour/day candle, every price change in these 24 hours except for the open, high, low, and close prices would be calculated randomly, even if this big candle here was a 24 hour/day candle.

If you want to know more about how it works, you can just open the "Help" file and read the instructions. I never use forwarding because I don't think anyone can predict the future. There is an execution setting which could simulate a slow network, so I don't use it either.

There is the amount of money in your testing account. You can set it to any amount in US dollars, Euros, or any other currency you like. There is the leverage. If the leverage is high, you can trade more money with a small account.

Optimization is something we'll talk about in a later video. For now, just click "Start," and you should see the text "Hello MQL5" on your chart. This isn't very exciting, so let's change it and use the function "TimeLocal."

Recompile the code, and when we restart the test, we'll see an output for the local time that is calculated every time the price changes. In this video, you learned how to use the Strategy Tester and how to output the local time. And you did it with just a few lines of MQL5 code.

### Part 4 - What are data types?

LEARN MQL5 TUTORIAL BASICS - 4 WHAT ARE DATA TYPES - YouTube

[Photo image of MQL5 Tutorial](https://www.youtube.com/channel/UCokIBdJXNOSOeYkKDvENWYA?embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F10954)

MQL5 Tutorial

27.5K subscribers

[LEARN MQL5 TUTORIAL BASICS - 4 WHAT ARE DATA TYPES](https://www.youtube.com/watch?v=_bNsvoLjz60)

MQL5 Tutorial

Search

Watch later

Share

Copy link

Info

Shopping

Tap to unmute

If playback doesn't begin shortly, try restarting your device.

Share

Include playlist

An error occurred while retrieving sharing information. Please try again later.

0:00

0:00 / 4:37

•Live

•

We're going to talk about data types and what they are in this video. In this strategy test, we see that the local time has a specific format. When you calculate something, it's important to use the right type of data, so let's find out how to do that.

Start by clicking on the little icon or pressing F4. You should now see the MetaEditor, where you want to click on "File/New/Expert Advisor (template)" from template, "Continue," and I'll call this file: "SimpleDataTypes," click "Continue," "Continue," and "Finish". You can now delete everything above the "OnTick" function and the two comment lines here.

Let's start with the most obvious type of data, which is text. We use the data type "string" to assign this text to a string variable called "Text," but you can't use string variables to do any calculations. To figure out anything, you have to use the right kind of value.

For example, to get the balance of our account, we use a "double" type, which is a floating-point type, along with the "AccountInfoDouble" function and this expression, which is all in uppercase. This should give us the right value, so let's use the "Comment" function to show the "Text" followed by the calculated value. Let's click on the "Compile" button here or press F7.

That should work without any problems, and if it does, you can click on the little icon here or press F4 to go back to MetaTrader. In the last video, we learned how to use this Strategy Tester, so let's click "File/Strategy Tester," pick the new file "SimpleDataTypes," and start a new test. You should now see that the value is 100,000.0.

Let's change that here, start a new test, and this time you'll see the digits behind the dot, which is why we use "double" when working with floating type values. You already know about the "TimeLocal" function from the last video. It returns a variable of type "datetime."

Let's make an output for this one, recompile the code, stop the previous test, and start a new one. Now you can see that the output is in a special format, so "datetime" is what we want to use whenever we need something with time and date. You can use the "integer" type for whole numbers.

In this case, we get the account number by using "AccountInfoInteger." We use this constant for account log in (ACCOUNT LOGIN). Let's see what this looks like.

When you only want a statement to be true or false, you use the "bool" type. Let's see what the output looks like for this one: "The value is: true." These are some common data types. If you click "Help" or press F1, you should find the MQL5 Reference article about data types. There are a few more types you can use, and it is also possible to use complex data types.

I would suggest writing little test programs like this one. In this short video, you saw how to put different types of data directly on your chart using a few lines of MQL5 code that you wrote yourself.

### Part 5 - How to do calculations

LEARN MQL5 TUTORIAL BASICS - 5 HOW TO DO CALCULATIONS - YouTube

[Photo image of MQL5 Tutorial](https://www.youtube.com/channel/UCokIBdJXNOSOeYkKDvENWYA?embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F10954)

MQL5 Tutorial

27.5K subscribers

[LEARN MQL5 TUTORIAL BASICS - 5 HOW TO DO CALCULATIONS](https://www.youtube.com/watch?v=-ELtMn-MJu4)

MQL5 Tutorial

Search

Watch later

Share

Copy link

Info

Shopping

Tap to unmute

If playback doesn't begin shortly, try restarting your device.

You're signed out

Videos you watch may be added to the TV's watch history and influence TV recommendations. To avoid this, cancel and sign in to YouTube on your computer.

CancelConfirm

More videos

## More videos

Share

Include playlist

An error occurred while retrieving sharing information. Please try again later.

[Why am I seeing this?](https://support.google.com/youtube/answer/9004474?hl=en)

[Watch on](https://www.youtube.com/watch?v=-ELtMn-MJu4&embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F10954)

0:00

0:00 / 4:33

•Live

•

In this video, we'll talk about simple math. When you use an Expert Advisor like this one to trade automatically, you'll need to do some math.

For example, in my case, I have calculated the maximum number of positions that are allowed, and you can see that the currency pair profit is also calculated. Let's learn how to do some basic calculations! To do this, click on the little icon here or press F4 to open your MetaEditor window.

Then, click "File/New/Expert Advisor (template)" from template, "Continue," name the file "SimpleCalculations," click "Continue," "Continue," and "Finish". Now, you can delete everything above the "OnTick" function here, and let's also get rid of the two comment lines. First, we need two "int" variables. "a" has a value of 5 and "b" has a value of 3.

We use "Comment" to show what "a" plus "b" gives us. Let's click the "Compile" button here. We don't have any mistakes, so we can now click here or press F4 to go back to MetaTrader.

When we go to "View/Strategy Tester" in MetaTrader, we choose the new "SimpleCalculations.ex5" file, turn on the visualization mode, and start the test. Here is the output: the result is 8, because 5+3=8, so let's try the next one: "a" - "b," recompile, and start a new test. This time, the result is 2, because 5-3=2.

Let's multiply the two numbers together: 5 x 3 = 15, so let's recompile. Try "a" divided by "b," what do you think? Oh, the answer is 1! We used integer variables, so 5/3 = 1, which is why the answer is 1.

So we don't have any floating type values here, which is where most people go crazy because the results don't match what they expect. Let's use "double" as a data type, recompile, and now the result is 1.6666666667, which is better.

What do you think will happen if you add "a" to "a" and multiply it by "b"? "a" equals 5, so "a" plus "a" equals 10. "b" equals 3, so 3 times 10 equals 30.

Let's add it all up, and the answer is 20! This is because multiplication or division is always done first. If you want to change that, you need to use brackets, because what's inside the brackets is calculated before multiplication or division, and our result this time is 30!

If you press F1 and open the MQL5 Reference, you will find lots of other math functions. I would suggest that you write little programs like this one to figure out how they work, because you will be doing more complicated calculations in the future, which makes it even harder to figure out what is working and why.

In this short video, you have learned how to do very basic math calculations, and you have done it yourself with a few lines of MQL5 code.

### Part 6 - How to check conditions with if

LEARN MQL5 TUTORIAL BASICS - 6 SIMPLE IF CONDITION - YouTube

[Photo image of MQL5 Tutorial](https://www.youtube.com/channel/UCokIBdJXNOSOeYkKDvENWYA?embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F10954)

MQL5 Tutorial

27.5K subscribers

[LEARN MQL5 TUTORIAL BASICS - 6 SIMPLE IF CONDITION](https://www.youtube.com/watch?v=Utr4IBg7FPQ)

MQL5 Tutorial

Search

Watch later

Share

Copy link

Info

Shopping

Tap to unmute

If playback doesn't begin shortly, try restarting your device.

You're signed out

Videos you watch may be added to the TV's watch history and influence TV recommendations. To avoid this, cancel and sign in to YouTube on your computer.

CancelConfirm

More videos

## More videos

Share

Include playlist

An error occurred while retrieving sharing information. Please try again later.

[Why am I seeing this?](https://support.google.com/youtube/answer/9004474?hl=en)

[Watch on](https://www.youtube.com/watch?v=Utr4IBg7FPQ&embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F10954)

0:00

0:00 / 4:58

•Live

•

In this video, we'll find out if a certain condition is true. When you use an automated program like this Expert Advisor, you need to check if something is true or false, like if the Stochastic is above or below the dotted lines or if the price is above or below the Moving Average.

Now we'll learn how to do that. Please click on this small icon or press F4 to open the MetaEditor. Here, you want to click on "File/New/Expert Advisor (template)" from template, "Continue," then "Continue," "Continue," and "Finish".

Now, you can delete everything above the "OnTick" function, and let's also get rid of the two comment lines. We'll start by using two integer variables: "a" is 5, and "b" should be 3. If "a" is greater than 5, I want to see the text "a is greater than 5"; if that's not the case, I can use the "else" statement, so whenever none of the conditions are true, I want to see the text "Conditions not true." Please click "Compile" or press F7, there are no mistakes here...

Oh, I got a warning because I forgot the "Comment" statement, but now everything is fine, so let's click this button or press F4 to go back to MetaTrader. And in MetaTrader, we click "View" and then "Strategy Tester." You could also press CTRL and R. Here, we want to choose the file "SimpleIfCondition.ex5".

Please turn on the visualization mode and start a test. Now we get the message "Conditions not true" because "a" is not greater than 5, so let's add another "if" statement to check if "a" is equal to 5. Now, let's recompile the code, stop the test, and start a new one.

This time, "a equals 5" will be shown as the result. What will happen if we add another "if" statement here to check if "b" equals 3? Let's recompile the code and run another test.

Now, we only see "b equals 3" on the screen. The statement in this case is ignored, so we could use two "if" statements instead. In the first one, we'll check to see if "a is 5", and in the second, we'll see if "b is 3."

The result will be "a is 5 and b is 3." Let's recompile the code, and the next time we test, we'll see "a = 5 and b = 3" on the screen.

So far, everything is fine, but if you add something here, like "c = a + b", and you click "Compile," you'll get two errors. This is because whenever you have more than one line behind a "if" statement, you need to use two curly braces. Let's recompile the code, and now it works, and in our last test, you'll see the message "c = 8."

This was a very simple example. There are many other ways to check if a condition is true, but in this short video, you learned how to use the "if" statement and a few lines of MQL5 code to check if a condition is true.

### Part 7 - How to use switch and case

LEARN MQL5 TUTORIAL BASICS - 7 HOW TO USE SWITCH AND CASE - YouTube

[Photo image of MQL5 Tutorial](https://www.youtube.com/channel/UCokIBdJXNOSOeYkKDvENWYA?embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F10954)

MQL5 Tutorial

27.5K subscribers

[LEARN MQL5 TUTORIAL BASICS - 7 HOW TO USE SWITCH AND CASE](https://www.youtube.com/watch?v=o2mN-ZwaUsQ)

MQL5 Tutorial

Search

Watch later

Share

Copy link

Info

Shopping

Tap to unmute

If playback doesn't begin shortly, try restarting your device.

You're signed out

Videos you watch may be added to the TV's watch history and influence TV recommendations. To avoid this, cancel and sign in to YouTube on your computer.

CancelConfirm

More videos

## More videos

Share

Include playlist

An error occurred while retrieving sharing information. Please try again later.

[Watch on](https://www.youtube.com/watch?v=o2mN-ZwaUsQ&embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F10954)

0:00

0:00 / 5:29

•Live

•

In this video, we'll learn how to change the way an Expert Advisor works by using the "switch" and "case" statements. This is an automated Expert Advisor. Right now, it doesn't do much; all it says is "customer wants RSI."

We need to figure out how to change this value using the "switch" and "case" commands. Click on the little icon here or press F4 to do that. Now you should see the MetaEditor window.

In this window, click "File/New/Expert Advisor (template)" from template, "Continue," and I'll call this file: "SimpleSwitchCase," click "Continue," "Continue," and "Finish". We can now get rid of everything above the "OnTick" function and the two comments here. We start by using an integer variable called "choice," which should have the value 5.

We also use a string variable called "entry" inside the "OnTick" function. We don't give it a value because we want to define the entry based on what we choose here. This is done with the "switch" statement.

We want to go through different options for our choice, so if the customer picks option 5, we say "the customer wants RSI." We are using a plus sign and an equal sign here, and I'll explain why later. Right now, we just want to break.

As soon as this code has been run, the "switch-case loop" will end because of this "break" statement. Let's add one more. If our choice is 4, the entry should say "the customer wants Bollinger Bands."

Again, we use the "break" statement to leave our "switch and case construct" here. Let's add another one. If our "choice" variable has the value 3, we want our "entry" statement to say "customer wants MACD".

You could also use an expression like 1+1 here instead of a number, but the case statement doesn't work with variables, so you'd get an error if you did. Let's also add a "default" option. "Default" will run if none of the other options are true, and in that case, we want our "entry" statement to say "the customer doesn't know."

Let's add two curly brackets here. Then, we want to use the "Comment" function to make an output for our "entry." When you're done, click the "Compile" button here.

This should work without any errors or warnings, and if it does, we can click a small button here or press F4 to go back to MetaTrader. Click "View/Strategy Tester" in MetaTrader or press CTRL and R. Here, we want to choose the new file "SimpleSwitchCase.ex5".

Please turn on the option for visualization and start a test. Here is what we made: "The customer wants RSI," so let's change "choice" to "3," recompile the code, and run another test. This time, we get "The customer wants MACD," since "choice" 3 is the same as "MACD."Let's change that to 11, recompile the code, and see what happens.

This time, it says "the customer doesn't know" because 11 isn't on our list of options, so that's the default. Okay, these "switch" and "case" statements are unique and it is possible that you can remove the "break" statement. Let's do that for the first two "case" statements, set the "choice" back to 5, recompile the code, and run another test.

This time, we get the messages "the customer wants RSI," "the customer wants Bollinger Bands," and "the customer wants MACD." This is because the first three parts of our "switch-case-construct" have been processed.  So, please remember that if you leave out the "break" operator here, our little Expert Advisor program will not leave the loop but will continue to check the next condition.

Now you learned how to use the "switch" and "case" statements in this short video. You did this with just a few lines of MQL5 code.

### Part 8 - How to use the while loop

LEARN MQL5 TUTORIAL BASICS - 8 HOW TO USE THE WHILE LOOP - YouTube

[Photo image of MQL5 Tutorial](https://www.youtube.com/channel/UCokIBdJXNOSOeYkKDvENWYA?embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F10954)

MQL5 Tutorial

27.5K subscribers

[LEARN MQL5 TUTORIAL BASICS - 8 HOW TO USE THE WHILE LOOP](https://www.youtube.com/watch?v=-zxk8ptjjbE)

MQL5 Tutorial

Search

Watch later

Share

Copy link

Info

Shopping

Tap to unmute

If playback doesn't begin shortly, try restarting your device.

You're signed out

Videos you watch may be added to the TV's watch history and influence TV recommendations. To avoid this, cancel and sign in to YouTube on your computer.

CancelConfirm

More videos

## More videos

Share

Include playlist

An error occurred while retrieving sharing information. Please try again later.

[Watch on](https://www.youtube.com/watch?v=-zxk8ptjjbE&embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F10954)

0:00

0:00 / 5:05

•Live

•

In this video, we'll learn how to use the "while" statement to wait until something happens, like right now. We waited until the delay counter reached 500,000, so let's learn how to use the "while" loop in MQL5. To do this, click the small button here or press F4 in your MetaTrader.

You should now see the MetaEditor window. In this window, click "File/New/Expert Advisor (template)" from template, "Continue," and then name the file "SimpleWhileLoop". Finally, click "Continue," "Continue," and "Finish". Everything above the "OnTick" function and the two comment lines can be deleted. So let's start by making a delay counter (DelayCounter).

We will use an integer variable because we only need whole numbers. We will give it a value of 1 and then increase it in the "OnTick" function. Let's say our minimum value is 500,000, and we only want to do something if our delay counter is higher than that.

Now, we can use the "while" statement inside the "OnTick" function to check if our delay counter is still below our minimum value. If it is, we want to do two things. First, we want to create an output that will show us the text "DelayCounter:" followed by the calculated value of the current delay counter.

Then, we want to increase the delay counter by 1 by writing "DelayCounter" equals "DelayCounter" plus 1. Let's open the Strategy Tester by clicking "View/Strategy Tester" or pressing CTRL and R. Here, we choose the new file "SimpleWhileLoop.ex5".

Please turn on the visualization mode and start your test. Now you can see that our counter is working, and when it reaches 500,000, this will move, so everything is fine. However, there is a catch: if you want to increase the delay counter outside of the while loop, we can still compile the code without any errors, but when we start the next test, nothing happens.

Why has the whole Strategy Tester stopped working? If you highlight the "while" statement and press F1, you should see the "Help" file, but nothing happens because everything is still frozen. This is because we've made an endless loop here.

When the first tick comes in and this expression is true, the "while" statement will start an endless loop. We wanted to increase the delay counter here, but since we did it outside of the "while" loop, this will never happen, so this expression will always be true. This is one of the reasons why I don't like to use "while" very often; you can check most of the conditions with "if."

Let's recompile the code, restart the test, and now we can see that the delay counter is working because the "if" statement will only be run once every time a tick occurs. Okay, so if you want to use the "while" statement, you need to make sure that something will stop the execution of your "while" loop, or your "while" loop will run forever. So let's try the test again, and this time it counts like it should.

In this short video, you learned how to use the "while" statement and how to avoid loops that never end. You also did it yourself with a few lines of MQL5 code.

### Part 9 - How to use the for loop

LEARN MQL5 TUTORIAL BASICS - 9 HOW TO USE THE FOR LOOP - YouTube

Tap to unmute

[LEARN MQL5 TUTORIAL BASICS - 9 HOW TO USE THE FOR LOOP](https://www.youtube.com/watch?v=pS0WHwhAWnw) [MQL5 Tutorial](https://www.youtube.com/channel/UCokIBdJXNOSOeYkKDvENWYA)

MQL5 Tutorial27.5K subscribers

[Watch on](https://www.youtube.com/watch?v=pS0WHwhAWnw)

In this video, we will learn how to use the "for" loop to change the value of a "counter." This one is counting until it reaches 10,000, so let's find out how to do that. To get started, click on the small icon here or press F4 in MetaTrader.

You should now see the MetaEditor window. Here, you want to click on: "File/New/Expert Advisor (template)" from template, "Continue," I'll call this file "SimpleForLoop," "Continue," "Continue," and "Finish".

Now you can delete everything above the "OnTick" function. Let's also get rid of the two comment lines. We start by making a "counter" variable.

This one is an integer (int) variable called "counter", and its initial value is 1. We also want to set an end value (endvalue), which is 10,000 in our case. Inside the "OnTick" function, we want to use the "for" loop.

If you have never seen a "for" loop before, this might look strange, but it is not. Here, the first expression is the starting value, which is the "counter" value 1 in our case. The second expression checks a condition.

In our case, we want to see if the value of the "counter" is less than the "endvalue." In the third expression, we do something to the "counter." In our case, we add 1 to it, which is what "counter++" means.

We could also say that "counter" equals "counter" plus 1. Whatever is inside these curly braces will be run as long as this condition is true. In our case, we just print out the text: "the counter So far, so good".

Please click on the "Compile" button here, and if you do not see any errors down here, you can click on the little symbol here or press F4 to go back to MetaTrader. In MetaTrader, we click on "View/Strategy Tester" or press CTRL and R, choose the new file "SimpleForLoop.ex5", turn on the visualization here, and start a test. And in the "Journal" tab, you should be able to see that the "counter" is working.

The number goes up until it reaches 9,999, which is good so far. You can also make a countdown. So let's switch the values around. Now the "counter" should start at 10,000 and the "endvalue" should be 1.

We will start with the "counter" value and count down by subtracting 1 from it as long as the "counter" is bigger than the "endvalue." Let's put that one together, start the test again, and this time add a countdown. Well, this one ends with "Counter = 2," which is because in our second expression we checked to see if "counter" was bigger than "endvalue".

We could also say "bigger or equal," so let's recompile. The last value should be 1 this time, and it is. You could also use the "for" loop to make the "counter" go up or down by other numbers.

Let's choose 10. One last compilation, and this time you can see that our countdown is working, and we have a step size of 10.

In this short video, you have learned how to use the "for" loop to count and increase or decrease "counter" values, and you have coded it yourself with a few lines of MQL5 code.

### Part 10 - How to code a simple function

LEARN MQL5 TUTORIAL BASICS - 10 HOW TO CODE A SIMPLE CUSTOM FUNCTION - YouTube

[Photo image of MQL5 Tutorial](https://www.youtube.com/channel/UCokIBdJXNOSOeYkKDvENWYA?embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F10954)

MQL5 Tutorial

27.5K subscribers

[LEARN MQL5 TUTORIAL BASICS - 10 HOW TO CODE A SIMPLE CUSTOM FUNCTION](https://www.youtube.com/watch?v=A_jqKTnoEXU)

MQL5 Tutorial

Search

Watch later

Share

Copy link

Info

Shopping

Tap to unmute

If playback doesn't begin shortly, try restarting your device.

You're signed out

Videos you watch may be added to the TV's watch history and influence TV recommendations. To avoid this, cancel and sign in to YouTube on your computer.

CancelConfirm

More videos

## More videos

Share

Include playlist

An error occurred while retrieving sharing information. Please try again later.

[Watch on](https://www.youtube.com/watch?v=A_jqKTnoEXU&embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F10954)

0:00

0:00 / 4:39

•Live

•

In this video, we will use MQL5 to make a custom function that will double a counter value every time the price changes. To do this, click on the small button here in MetaTrader or press F4 on your keyboard. You should now see the MetaEditor window. Click on "File/New/Expert Advisor (template)" from template, "Continue," and I'll call this file: "SimpleFunction," click "Continue," "Continue," and "Finish".

You can now delete everything above the "OnTick" function and the two comment lines. We start by making a static variable for our "counter." "Static" is a word you might not have seen before. In this case, we use "static" to make a local variable that will only exist in our function.

We want to pass the variable to another function that will be called "DoubleTheValue" (Double The Value), but it does not exist yet, so we need to make it. Inside the round braces, we will pass the value of our current "counter". And we want to use the Comment function to output the counter value.

The DoubleTheValue function still needs to be created. To make a custom function, you start by defining what the function will return. In our case, the function will return an integer (int) value.

This is the name of the custom function, which I call "DoubleTheValue," and this is what the function will take as a parameter. We pass the "counter," which is an integer (int). I call it "CounterValue" in the custom function, but you do not have to use the same name.

Our function will need two curly braces, and when I try to compile it this time, we get another error because our function isn't going to return anything. So let's first double the passed value, which we have called "CounterValue". To do this, we take "CounterValue" and multiply it by 2.

The result is stored in a new variable called "DoubleValue," and we can use the "return" operator to send the value back to the calling program. This is done by typing "return DoubleValue;". So let's compile the code again to make sure there are no errors.

And in MetaTrader, click "View/Strategy Tester" or press CTRL+R, choose the new file "SimpleFunction.ex5", check the "Visualization" box, and run a test. You should now see that the counter value doubles every time a new tick comes in, so our custom function is called whenever the price changes. This was a very simple example because we only passed one value and did one operation.

However, you could use the same kind of logic to do more complicated things. For example, you could go through all the open positions, calculate the profit for a currency pair, and return the value to the main function by using the "return" operator.

In this short video, you learned how to create a custom function in MQL5 and coded it yourself with a few lines of code.

### Conclusion

MQL5 does not need to be complicated. I hope, this first part will help you get started and give you a good understanding about how simple automated trading can be. Of course, there is much more content available and you will need to add more advanced parts to your own system, but by now you should know if MQL5 programming is for you...

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Video: How to setup MetaTrader 5 and MQL5 for simple automated trading](https://www.mql5.com/en/articles/10962)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/425709)**
(2)


![Dragan Drenjanin](https://c.mql5.com/avatar/2022/6/629F67A4-4A75.png)

**[Dragan Drenjanin](https://www.mql5.com/en/users/drgandra)**
\|
26 May 2022 at 12:57

Hi Raimund,

You have excellent video lessons for beginners.

![ForexmailkontoFrLeLa](https://c.mql5.com/avatar/avatar_na2.png)

**[ForexmailkontoFrLeLa](https://www.mql5.com/en/users/forexmailkontofrlela)**
\|
20 Jun 2022 at 16:02

Raimund,

you did a really good job with these videos.

Your calm manner and precise way of expression contributes a lot to understanding.

![Learn how to design a trading system by OBV](https://c.mql5.com/2/46/why-and-how__6.png)[Learn how to design a trading system by OBV](https://www.mql5.com/en/articles/10961)

This is a new article to continue our series for beginners about how to design a trading system based on some of the popular indicators. We will learn a new indicator that is On Balance Volume (OBV), and we will learn how we can use it and design a trading system based on it.

![Graphics in DoEasy library (Part 100): Making improvements in handling extended standard graphical objects](https://c.mql5.com/2/45/MQL5-avatar-doeasy-library3-2__4.png)[Graphics in DoEasy library (Part 100): Making improvements in handling extended standard graphical objects](https://www.mql5.com/en/articles/10634)

In the current article, I will eliminate obvious flaws in simultaneous handling of extended (and standard) graphical objects and form objects on canvas, as well as fix errors detected during the test performed in the previous article. The article concludes this section of the library description.

![Developing a trading Expert Advisor from scratch (Part 7): Adding Volume at Price (I)](https://c.mql5.com/2/45/variety_of_indicators__5.png)[Developing a trading Expert Advisor from scratch (Part 7): Adding Volume at Price (I)](https://www.mql5.com/en/articles/10302)

This is one of the most powerful indicators currently existing. Anyone who trades trying to have a certain degree of confidence must have this indicator on their chart. Most often the indicator is used by those who prefer “tape reading” while trading. Also, this indicator can be utilized by those who use only Price Action while trading.

![Multiple indicators on one chart (Part 06): Turning MetaTrader 5 into a RAD system (II)](https://c.mql5.com/2/45/variety_of_indicators__4.png)[Multiple indicators on one chart (Part 06): Turning MetaTrader 5 into a RAD system (II)](https://www.mql5.com/en/articles/10301)

In my previous article, I showed you how to create a Chart Trade using MetaTrader 5 objects and thus to turn the platform into a RAD system. The system works very well, and for sure many of the readers might have thought about creating a library, which would allow having extended functionality in the proposed system. Based on this, it would be possible to develop a more intuitive Expert Advisor with a nicer and easier to use interface.

[![](https://www.mql5.com/ff/si/5k7a2kbftss6k97n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F1171%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dbest.vps%26utm_content%3Drent.vps%26utm_campaign%3D0622.MQL5.com.Internal&a=nwegcasiojnqcoyrdlgofmjtfardztwf&s=d64d6f3c87f2458cba81f6d7b6694dd9e89dd354d4abc1d0584e405285806c9f&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=ilubpwgtobsdcbqyfbeoxhqxkgxgwtwh&ssn=1769092979636406305&ssn_dr=0&ssn_sr=0&fv_date=1769092979&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F10954&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Video%3A%20Simple%20automated%20trading%20%E2%80%93%20How%20to%20create%20a%20simple%20Expert%20Advisor%20with%20MQL5%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176909297921478586&fz_uniq=5049325314879891826&sv=2552)

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