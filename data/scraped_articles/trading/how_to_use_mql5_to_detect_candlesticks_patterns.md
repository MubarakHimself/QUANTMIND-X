---
title: How to use MQL5 to detect candlesticks patterns
url: https://www.mql5.com/en/articles/12385
categories: Trading, Trading Systems, Expert Advisors
relevance_score: 5
scraped_at: 2026-01-23T17:32:33.370154
---

[![](https://www.mql5.com/ff/si/fx5m8s6u6uxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=ysducdhemkrdsdtzzbfkclolrllnhezk&s=33f180a31db6c3b846d77732b0bc78169421a47b8cf9f076ca717f4e4846d1c7&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=nbmgpwdwrmwfatsgqtmamqcpsovbhxrj&ssn=1769178752476583473&ssn_dr=0&ssn_sr=0&fv_date=1769178752&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F12385&back_ref=https%3A%2F%2Fwww.google.com%2F&title=How%20to%20use%20MQL5%20to%20detect%20candlesticks%20patterns%20-%20MQL5%20Articles&scr_res=1920x1080&ac=17691787521776529&fz_uniq=5068357852566845567&sv=2552)

MetaTrader 5 / Trading


### Introduction

Candlesticks are a very helpful technical tool if we use them correctly, as we can find a potential movement based on their patterns. Candlesticks can form specific patterns on the chart and these patterns can be divided into two types single-candle patterns and blended candle patterns (more than one candle). In this article, we will learn how we can use MQL5 to detect some of these patterns automatically in the MetaTrader 5 trading terminal, we will cover that through the following topics:

- [Single candle patterns](https://www.mql5.com/en/articles/12385#single)
- [Dual candles patterns](https://www.mql5.com/en/articles/12385#dual)
- [Three candles patterns](https://www.mql5.com/en/articles/12385#three)
- [Conclusion](https://www.mql5.com/en/articles/12385#conclusion)

I need to mention that it is very important to use these patterns accompanied by other technical tools to get significant signals. So, you need to understand the main idea of detecting mentioned patterns by MQL5 to be a part of your trading system to ease your trading and get good results.

Disclaimer: All information provided 'as is' only for educational purposes and is not prepared for trading purposes or advice. The information does not guarantee any kind of result. If you choose to use these materials on any of your trading accounts, you will do that at your own risk and you will be the only responsible.

### Single candle pattern

In this part, we will see two examples of popular single-candle pattern which appears on the chart. You can see them in any time frame, it will be more significant when they come in its place relative to the price action. we will see the Doji and the Hammer patterns.

**Doji pattern:**

It is very popular among candlesticks patterns and it is the candle that has almost the same opening and closing price, we can see a very small body of the candle or a line on the chart for the same price with upper and lower body or even without these shadows. The following figure is for this candle:

![Doji](https://c.mql5.com/2/52/Doji__1.png)

This Doji candle indicates that there is a balance between buyers and sellers and no one controls the market to move the price higher or lower during the period when it appears at. It may signal a reversal or correction in the market if it appears at its suitable place on the chart before a correction or at the end of the trend and it will be more significant if it appears in the bigger time frame. There are many types and formations for this candle and everyone has a lot of information that can be used in our favor in trading like Dragonfly and Long-legged.

What we need to do is to inform the computer to detect the Doji pattern by defining the prices and time of the last candle and every tick we need the program to check and compare these values at this defined time and determine the positions of everyone. If the open is equal to the close we need the program to return a signal that this is a Doji candle pattern.

Now, we need to create a program that can detect this pattern and the following are steps of a method that can do that:

We will create a function for this Doji (getDoji) and we will call it in the OnTick() to check every tick searching for this pattern

```
void OnTick()
  {
   getDoji();
  }
```

Create the (getDoji) function by creating it as an integer variable

```
int getDoji()
```

Defining this function by defining time, open, high, low, and close prices of the last candle

Using the iTime function which returns the opening time of the candle, the iOpen which returns the open price of the candle, the iHigh which returns the high price, the iLow which returns the low price, and the iClose which returns the close price of the candle. parameters are the same for all of them:

- symbol: to define the symbol name, we will use (\_Symbol) for the current symbol.
- timeframe: to define the period or timeframe of the chart, we will use (PERIOD\_CURRENT) for the current timeframe.
- shift: to define the index of the returned value, we will use (1) to define the last candle.

```
   datetime time=iTime(_Symbol,PERIOD_CURRENT,1);
   double open=iOpen(_Symbol,PERIOD_CURRENT,1);
   double high=iHigh(_Symbol,PERIOD_CURRENT,1);
   double low=iLow(_Symbol,PERIOD_CURRENT,1);
   double close=iClose(_Symbol,PERIOD_CURRENT,1);
```

Setting the condition of the Doji that we need to detect by using the if statement

```
if(open==close)
```

If this condition is true, we need the program to create an object based on the createObj function that we will create with the parameters of time, price, code of arrow, color, and the text that we need. Then terminate the function by returning 1.

```
   if(open==close)
     {
      createObj(time,low,217, clrBlack,"Doji");
        {
         return 1;
        }
     }
```

We will return 0 to terminate the function of getDoji

```
   return 0;
```

Creating (createObj) function with the parameters of time, price, arrowcode, clr, and txt by using the void function

```
void createObj(datetime time, double price, int arrawCode, color clr, string txt)
```

Creating a string variable of (objName) assigned to (" ") value

```
string objName=" ";
```

Combining strings to assign them to the (objName) variable by using the (StringConcatenate) function which forms a string of passed parameters and returns the size of the formed string. Its parameters are:

- string\_var: to define the string that will be formed after concatenation, we will use the (objName).
- argument1: to define the parameter of any simple type, we will use the text of "Signal at  ".
- argument2: to define the time of the detected candle, we will use the time of the predetermined variable.
- argument3: we will set the text of " at ".
- argument4: we will set the text of the rounded price by using DoubleToString to convert the double type to string type.
- argument5: we will set the text of " (".
- argument6: we will assign a value for the predetermined integer variable (arrowcode) that we need. This code can be found by searching for the Wingdings in the mql5 reference.
- argument7: we will set the text of ")".

```
StringConcatenate(objName, "Signal at ",time, " at ",DoubleToString(price,_Digits)," (",arrawCode,")");
```

We will set a condition to be evaluated by using the if statement and the (ObjectCreate) function as an expression, the (ObjectCreate) function creates an object for us by using the predefined name (objName) and its parameters are:

- chart\_id: to identify the chart, we will use 0 for the current chart.
- name: to define the object name, we will use the predefined name (objName).
- type: to define the object type, we will use (OBJ\_ARROW).
- nwin: to define the number of the chart sub-window, we will use (0) for the main chart window.
- time1: to define the time of the anchor, we will use the predefined (time) variable.
- price1: to define the price of the anchor, we will use the predefined (price) variable.

```
if(ObjectCreate(0,objName,OBJ_ARROW,0,time,price))
```

Once this condition is true by creating the object we need to set its properties its shape by determining the arrow code and its color by using the (ObjectSetInteger) function which sets the value of the object property. Its parameters are:

- chart\_id: to identify the chart, we will use 0 for the current chart.
- name: to define the object name, we will use (objName).
- prop\_id: to define the property of the object, we will use one of the ENUM\_OBJECT\_PROPERTY\_INTEGER which will be the (OBJPROP\_ARROWCODE) for the arrow code and (OBJPROP\_COLOR) for the color.
- prop\_value: to define the property value, we will use (arrawCode) for the arrow code and the predefined variable (clr) for the color.

```
ObjectSetInteger(0,objName,OBJPROP_ARROWCODE,arrawCode);
ObjectSetInteger(0,objName,OBJPROP_COLOR,clr);
```

After that, we need to define the text that we need as a definition of the candle by creating a string variable for the (candleName) with an assignment of the predefined (objName) and (txt) variables

```
string candleName=objName+txt;
```

Creating and editing the text object by using the if statement and using (ObjectCreate) function as an expression and the operator will be (ObjectSetString) to set the string value of the object property and the (ObjectSetInteger) to set the color of the text object.

```
      ObjectSetString(0,candleName,OBJPROP_TEXT," "+txt);
      ObjectSetInteger(0,candleName,OBJPROP_COLOR,clr);
```

Now, we can see the full code of this Expert Advisor the same as the following:

```
//+------------------------------------------------------------------+
//|                                        Doji pattern detector.mq5 |
//|                                  Copyright 2023, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
void OnTick()
  {
   getDoji();
  }
int getDoji()
  {
   datetime time=iTime(_Symbol,PERIOD_CURRENT,1);
   double open=iOpen(_Symbol,PERIOD_CURRENT,1);
   double high=iHigh(_Symbol,PERIOD_CURRENT,1);
   double low=iLow(_Symbol,PERIOD_CURRENT,1);
   double close=iClose(_Symbol,PERIOD_CURRENT,1);
//Doji
   if(open==close)
     {
      createObj(time,low,217, clrBlack,"Doji");
        {
         return 1;
        }
     }
   return 0;
  }
void createObj(datetime time, double price, int arrawCode, color clr, string txt)
  {
   string objName=" ";
   StringConcatenate(objName, "Signal at ",time, " at ",DoubleToString(price,_Digits)," (",arrawCode,")");
   if(ObjectCreate(0,objName,OBJ_ARROW,0,time,price))
     {
      ObjectSetInteger(0,objName,OBJPROP_ARROWCODE,arrawCode);
      ObjectSetInteger(0,objName,OBJPROP_COLOR,clr);
     }
   string candleName=objName+txt;
   if(ObjectCreate(0,candleName,OBJ_TEXT,0,time,price))
     {
      ObjectSetString(0,candleName,OBJPROP_TEXT," "+txt);
      ObjectSetInteger(0,candleName,OBJPROP_COLOR,clr);
     }
  }
```

After compiling this code without errors, we can find it in the navigator window. By dragging it to be executed we can then get its signals detecting the Doji pattern and the following is an example from testing:

![Doji example](https://c.mql5.com/2/52/Doji_example.png)

As we can see in the previous chart we have a black arrow object below the candle and the Doji text to define the candle pattern.

**Hammer Pattern:**

The hammer pattern is a very popular candlesticks pattern that we can see on the chart in many timeframes. Its name refers to its shape as it has a long shadow and a small body there are two types of Hammer patterns, Hammer and Inverted Hammer as per the position of the small body. If it has a long lower shadow and the body of the candle is above, it is a Hammer and it can be a bullish or bearish candle based on the opening and closing price, the following figures are examples of this Hammer patterns:

- Bullish Hammer

![Bullish Hammer](https://c.mql5.com/2/52/Bullish_Hammer.png)

It indicates that the seller tried to push the price lower but the buyer controls the market and close higher than its opening which means strength for the buyer.

- Bearish Hammer

![Bearish Hammer](https://c.mql5.com/2/52/Bearish_Hammer.png)

It indicates that the seller tried to push the price lower but the buyer appears to close at around the opening which means that the buyer is still in the game.

If the candle has a long upper shadow and its body is below, it is an Inverted Hammer pattern and it can be also bullish or bearish based on the position of open and close prices. The following figures are examples of this Inverted Hammer.

- Bullish Inverted Hammer

![Bullish Inverted Hammer](https://c.mql5.com/2/52/Bullish_Inverted_Hammer.png)

It indicates that the buyer tried to push prices higher but the seller appears to close the candle around the open and the low which means that the seller is still in the game although the strength of the buyer.

- Bearish Inverted Hammer

![Bearish Inverted Hammer](https://c.mql5.com/2/52/Bearish_Inverted_Hammer.png)

It indicates that the buyer tried to push the price lower but the seller controls the market and close lower than its opening which means strength for the seller.

This pattern also the same as all candlesticks patterns will be more meaningful and significant when it combined with other technical tools

Now, we need to create a program that can be used to detect this kind of pattern, so we will let this program find the candle prices, time, and candle size to be compared to the body and shadows of the candle and we need the program to continuously check and comparing them every tick to determine their positions. When the program detects one of the Hammer or Inverted Hammer (Bullish or Bearish), we need the program to return an object on the chart with its type name and arrows color green or red based and below or above the candle based on the color of the candle (bullish or bearish).

The following is the full code to create this type of program:

```
//+------------------------------------------------------------------+
//|                                      Hammer pattern detector.mq5 |
//|                                  Copyright 2023, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
void OnTick()
  {
   getHammer(0.07,0.7);
  }
int getHammer(double smallShadowRatio, double longShadowRatio)
  {
   datetime time=iTime(_Symbol,PERIOD_CURRENT,1);
   double open=iOpen(_Symbol,PERIOD_CURRENT,1);
   double high=iHigh(_Symbol,PERIOD_CURRENT,1);
   double low=iLow(_Symbol,PERIOD_CURRENT,1);
   double close=iClose(_Symbol,PERIOD_CURRENT,1);
   double candleSize=high-low;
   if(open<close)
     {
      if(high-close < candleSize*smallShadowRatio)
        {
         if(open-low>candleSize*longShadowRatio)
            createObj(time,low,217, clrGreen,"Hammer");
           {
            return 1;
           }
        }
     }
   if(open>close)
     {
      if(high-open<candleSize*smallShadowRatio)
        {
         if(close-low>candleSize*longShadowRatio)
            createObj(time,high,218,clrRed,"Hammer");
           {
            return 1;
           }
        }
     }
   if(open<close)
     {
      if(open-low < candleSize*smallShadowRatio)
        {
         if(high-close>candleSize*longShadowRatio)
            createObj(time,low,217, clrGreen,"Inverted Hammer");
           {
            return -1;
           }
        }
     }
   if(open>close)
     {
      if(close-low < candleSize*smallShadowRatio)
        {
         if(high-open>candleSize*longShadowRatio)
            createObj(time,high,218, clrRed,"Inverted Hammer");
           {
            return -1;
           }
        }
     }
   return 0;
  }
void createObj(datetime time, double price, int arrawCode, color clr, string txt)
  {
   string objName=" ";
   StringConcatenate(objName, "Signal@",time, "at",DoubleToString(price,_Digits),"(",arrawCode,")");
   if(ObjectCreate(0,objName,OBJ_ARROW,0,time,price))
     {
      ObjectSetInteger(0,objName,OBJPROP_ARROWCODE,arrawCode);
      ObjectSetInteger(0,objName,OBJPROP_COLOR,clr);
      if(clr==clrGreen)
         ObjectSetInteger(0,objName,OBJPROP_ANCHOR,ANCHOR_TOP);
      if(clr==clrRed)
         ObjectSetInteger(0,objName,OBJPROP_ANCHOR,ANCHOR_BOTTOM);
     }
   string candleName=objName+txt;
   if(ObjectCreate(0,candleName,OBJ_TEXT,0,time,price))
     {
      ObjectSetString(0,candleName,OBJPROP_TEXT," "+txt);
      ObjectSetInteger(0,candleName,OBJPROP_COLOR,clr);
     }
  }
```

Differences in this code are the same as the following:

Calling the getHammer function in the OnTick() with determining desired parameters for the smallShadowRatio and longShadowRatio

```
void OnTick()
  {
   getHammer(0.07,0.7);
  }
```

Creating the (getHammer) function with parameters of two double variables (smallShadowRatio) and (longShadowRatio)

```
int getHammer(double smallShadowRatio, double longShadowRatio)
```

Creating a double variable for the (candleSize) to be compared to the ratio

```
double candleSize=high-low;
```

Conditions of the Hammer candle,

In the case of a Bullish Hammer (Open<Close), we need a last bullish candle, the upper shadow of the candle (high-close) is less than the ratio of small shadow which is 0.07, and the lower shadow (open-low) is greater than the long shadow ratio which is 0.7. Create a green arrow with the code (217) from Wingdings and the "Hammer" text object on the chart below the low of this Hammer candle, then terminate the function.

```
   if(open<close)
     {
      if(high-close < candleSize*smallShadowRatio)
        {
         if(open-low>candleSize*longShadowRatio)
            createObj(time,low,217, clrGreen,"Hammer");
           {
            return 1;
           }
        }
     }
```

In the case of a Bearish Hammer (open>close), we need a last bearish candle, the upper shadow of the candle (high-open) is less than the ratio of small shadow which is 0.07, and the lower shadow (close-low) is greater than the long shadow ratio which is 0.7. Create a red arrow with the code (218) from Wingdings and the "Hammer" text object on the chart above the high of this Hammer candle, then terminate the function.

```
   if(open>close)
     {
      if(high-open<candleSize*smallShadowRatio)
        {
         if(close-low>candleSize*longShadowRatio)
            createObj(time,high,218,clrRed,"Hammer");
           {
            return 1;
           }
        }
     }
```

In the case of a Bullish Inverted Hammer (open<close), we need a last bullish candle, the lower shadow of the candle (open-low) is less than the ratio of small shadow which is 0.07, and the upper shadow (high-close) is greater than the long shadow ratio which is 0.7. Create a green arrow with the code (217) from Wingdings and the "Inverted Hammer" text object on the chart below the low of this Hammer candle, then terminate the function.

```
   if(open<close)
     {
      if(open-low < candleSize*smallShadowRatio)
        {
         if(high-close>candleSize*longShadowRatio)
            createObj(time,low,217, clrGreen,"Inverted Hammer");
           {
            return -1;
           }
        }
     }
```

In the case of a Bearish Inverted Hammer (open>close), we need a last bearish candle, the lower shadow of the candle (close-low) is less than the ratio of small shadow which is 0.07, and the upper shadow (high-open) is greater than the long shadow ratio which is 0.7. Create a red arrow with the code (218) from Wingdings and the "Inverted Hammer" text object on the chart above the high of this candle, then terminate the function.

```
   if(open>close)
     {
      if(close-low < candleSize*smallShadowRatio)
        {
         if(high-open>candleSize*longShadowRatio)
            createObj(time,high,218, clrRed,"Inverted Hammer");
           {
            return -1;
           }
        }
     }
```

Let's edit the arrow position and color depending on the candle type using the 'if' operator. The expression will be the color, while the 'if' operator (if it is true) will be used as the arrow position with the help of the ObjectSetInteger function.

If green will be below the candle

```
      if(clr==clrGreen)
         ObjectSetInteger(0,objName,OBJPROP_ANCHOR,ANCHOR_TOP);
```

If red will be above the candle

```
      if(clr==clrRed)
         ObjectSetInteger(0,objName,OBJPROP_ANCHOR,ANCHOR_BOTTOM);
```

After compiling this code without errors and executing it we can get our signal and the following are examples from testing:

- Bullish Hammer:

![Bullish Hammer example](https://c.mql5.com/2/52/Bullish_Hammer_example.png)

As we can see that we have a green arrow and the green "Hammer" text object on the chart below the low of the bullish Hammer candle.

- Bearish Hammer:

![Bearish Hammer example](https://c.mql5.com/2/52/Bearish_Hammer_example.png)

As we can see that we have a red arrow and the red "Hammer" text object on the chart above the high of the bearish Hammer candle.

- Bullish Inverted Hammer:

![Bullish Inverted Hammer example](https://c.mql5.com/2/52/Bullish_Inverted_Hammer_example.png)

As we can see that we have a green arrow and the green "Inverted Hammer" text object on the chart below the low of the bullish Inverted Hammer candle.

- Bearish Inverted Hammer:

![Bearish Inverted Hammer example](https://c.mql5.com/2/52/Bearish_Inverted_Hammer_example.png)

As we can see that we a red arrow and the red "Inverted Hammer" text object on the chart above the high of the bearish Inverted Hammer candle.

### Dual candles pattern

In this part, we will see another type of candlesticks pattern that consists of two candles, we will see two popular patterns which are Engulfing (Bullish and Bearish) and the bullish Piercing line and its opposite bearish Dark Cloud patterns.

**Engulfing pattern:**

This candlestick pattern is also a very popular one on the chart and technical analysis that consists of two candles one of them engulfs the other one which means that it has a small candle followed by a larger one and this larger one covers the smaller one completely.

There are types of this Engulfing pattern based on the color or type of candles:

- The Bullish Engulfing:

It has a small bearish candle followed by a large bullish candle and this bullish one engulfs the smaller one and the following figure is for it:

![Bullish engulfing](https://c.mql5.com/2/52/Bullish_engulfing.png)

Depending on its significance, it indicates that the buyer controls the market and the price may continue to rise after it.

- The Bearish Engulfing:

It has a small bullish candle followed by a large bearish candle and this bearish one engulfs the smaller bullish one and the following figure is for it:

![Bearish engulfing](https://c.mql5.com/2/52/Bearish_engulfing.png)

Depending on its significance, it indicates that the seller controls the market and the price may continue to decline after it.

Now, if we want to create a program that can be used to detect this pattern automatically, we need to define the time of the last candle and prices of the last two candles, we need the program to continuously check these values every tick and determine its positions related to each other to check if we have this type of Engulfing pattern or not. Once we have this Engulfing pattern, we need the program to return a specific signal which is a colored arrow and text object based on its type (Bullish or Bearish).

The following is the full code to create this program:

```
//+------------------------------------------------------------------+
//|                                   Engulfing pattern detector.mq5 |
//|                                  Copyright 2023, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
void OnTick()
  {
   getEngulfing();
  }
int getEngulfing()
  {
   datetime time=iTime(_Symbol,PERIOD_CURRENT,1);
   double open=iOpen(_Symbol,PERIOD_CURRENT,1);
   double high=iHigh(_Symbol,PERIOD_CURRENT,1);
   double low=iLow(_Symbol,PERIOD_CURRENT,1);
   double close=iClose(_Symbol,PERIOD_CURRENT,1);
   double open2=iOpen(_Symbol,PERIOD_CURRENT,2);
   double high2=iHigh(_Symbol,PERIOD_CURRENT,2);
   double low2=iLow(_Symbol,PERIOD_CURRENT,2);
   double close2=iClose(_Symbol,PERIOD_CURRENT,2);
   if(open<close)
     {
      if(open2>close2)
        {
         if(high>high2&&low<low2)
           {
            if(close>open2&&open<close2)
              {
               createObj(time,low,217, clrGreen,"Bullish Engulfing");
                 {
                  return 1;
                 }
              }
           }
        }
     }
   if(open>close)
     {
      if(open2<close2)
        {
         if(high>high2&&low<low2)
           {
            if(close<open2&&open>close2)
              {
               createObj(time,high,218, clrRed,"Bearish Engulfing");
                 {
                  return -1;
                 }
              }
           }
        }
     }
   return 0;
  }
void createObj(datetime time, double price, int arrawCode, color clr, string txt)
  {
   string objName=" ";
   StringConcatenate(objName, "Signal@",time, "at",DoubleToString(price,_Digits),"(",arrawCode,")");
   if(ObjectCreate(0,objName,OBJ_ARROW,0,time,price))
     {
      ObjectSetInteger(0,objName,OBJPROP_ARROWCODE,arrawCode);
      ObjectSetInteger(0,objName,OBJPROP_COLOR,clr);
      if(clr==clrGreen)
         ObjectSetInteger(0,objName,OBJPROP_ANCHOR,ANCHOR_TOP);
      if(clr==clrRed)
         ObjectSetInteger(0,objName,OBJPROP_ANCHOR,ANCHOR_BOTTOM);
     }
   string candleName=objName+txt;
   if(ObjectCreate(0,candleName,OBJ_TEXT,0,time,price))
     {
      ObjectSetString(0,candleName,OBJPROP_TEXT," "+txt);
      ObjectSetInteger(0,candleName,OBJPROP_COLOR,clr);
     }
  }
```

Differences in this code:

Creating double variables for time for the last candle and prices the last two candles in creating the function of (getEngulfing), time, open, high, low, and close are for the last candle and  open2, high2, low2, and close2 are for the previous candle of the last one

```
   datetime time=iTime(_Symbol,PERIOD_CURRENT,1);
   double open=iOpen(_Symbol,PERIOD_CURRENT,1);
   double high=iHigh(_Symbol,PERIOD_CURRENT,1);
   double low=iLow(_Symbol,PERIOD_CURRENT,1);
   double close=iClose(_Symbol,PERIOD_CURRENT,1);
   double open2=iOpen(_Symbol,PERIOD_CURRENT,2);
   double high2=iHigh(_Symbol,PERIOD_CURRENT,2);
   double low2=iLow(_Symbol,PERIOD_CURRENT,2);
   double close2=iClose(_Symbol,PERIOD_CURRENT,2);
```

Conditions that define this type of candle pattern

In the case of Bullish Engulfing, The last candle is bullish (open<close), the previous of the last is bearish one (open2>close2), the high is greater than high2 and at the same low is lower than low2, the close greater than open2 and at the same time open is lower than close2. Once identifying it, create an object based on the created function (createObj) with the following parameters:

- time: it will be the time of the last candle which is the predefined variable.
- price: it will be the low of the last candle that we need the object below it.
- arrowCode: it will be 217 from the Wingdings.
- clr: it will be clrGreen.
- txt: it will be the "Bullish Engulfing".

Then terminate the function.

```
   if(open<close)
     {
      if(open2>close2)
        {
         if(high>high2&&low<low2)
           {
            if(close>open2&&open<close2)
              {
               createObj(time,low,217, clrGreen,"Bullish Engulfing");
                 {
                  return 1;
                 }
              }
           }
        }
     }
```

In the case of Bearish Engulfing, The last candle is bearish (open>close), the previous of the last is bullish one (open2<close2), the high is greater than high2, and at the same low is lower than low2, the close lower than open2 and at the same time open is greater than close2. Once identifying it, create an object based on the created function (createObj) with the following parameters:

- time: it will be the time of the last candle which is the predefined variable.
- price: it will be the high of the last candle that we need the object above it.
- arrowCode: it will be 218 from the Wingdings.
- clr: it will be clrRed.
- txt: it will be the "Bearish Engulfing".

Then terminate the function.

```
   if(open>close)
     {
      if(open2<close2)
        {
         if(high>high2&&low<low2)
           {
            if(close<open2&&open>close2)
              {
               createObj(time,high,218, clrRed,"Bearish Engulfing");
                 {
                  return -1;
                 }
              }
           }
        }
     }
```

After compiling this code without errors and executing its EA, we can get its signals the as the following examples from testing:

- The Bullish Engulfing:

![Bullish engulfing example](https://c.mql5.com/2/52/Bullish_engulfing_example.png)

The same as we can see on the previous chart we have a green arrow and Bullish Engulfing text below the low of the last candle in the pattern.

- The Bearish Engulfing:

![Bearish engulfing example](https://c.mql5.com/2/52/Bearish_engulfing_example.png)

The same as we can see on the previous chart we have a red arrow and Bearish Engulfing text above the high of the last candle in the pattern.

**Piercing Line and Dark Cloud Cover pattern:**

- Piercing Line Pattern:

It is a bullish candlestick and consists of two candles also as the first candle is bearish and followed by a bullish one with lower open than the bearish one then moves up and closes above the midpoint of the first bearish candle. The following figure is for a graph that describes it:

![Piercing Line](https://c.mql5.com/2/52/Piercing_Line.png)

It indicates that the buyer becomes stronger and controls the market after the control from the seller. So, it refers to a shift from selling to buying power as the buyer was able to push the price above the midpoint of the previous bearish candle although there was a gap when opening.

- Dark Cloud Cover pattern:

It is the opposite form from the piercing patterns, as it is a bearish pattern that has two candles structure the first is bullish and is followed by a bearish candle with an opening gap and closes below the midpoint of the first bullish one. The following is a graph for it:

![Dark Cloud Cover](https://c.mql5.com/2/52/Dark_Cloud_Cover.png)

It indicates that the seller becomes stronger and controls the market after the control from the buyer. So, it refers to a shift from buying to selling power as the seller was able to push the price below the midpoint of the previous bullish candle although there was a gap when opening.

When we want to create a program that can be used to detect this type of pattern we will need to define the time and prices of the first candle (time, open, high, low, and close) and prices for the second candle (open2, high2, low2, and close2), candle size of the first candle (candleSize2), and the midpoint of the first candle (candleMidPoint2). we need the program to continuously check these values and determine their positions related to each other and return a specific signal based on specific conditions based on the bullishness or bearishness.

The following is the full code to create this program:

```
//+------------------------------------------------------------------+
//|                      Piercing && Dark Cloud pattern detector.mq5 |
//|                                  Copyright 2023, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
void OnTick()
  {
   getPiercing();
  }
int getPiercing()
  {
   datetime time=iTime(_Symbol,PERIOD_CURRENT,1);
   double open=iOpen(_Symbol,PERIOD_CURRENT,1);
   double high=iHigh(_Symbol,PERIOD_CURRENT,1);
   double low=iLow(_Symbol,PERIOD_CURRENT,1);
   double close=iClose(_Symbol,PERIOD_CURRENT,1);
   double open2=iOpen(_Symbol,PERIOD_CURRENT,2);
   double high2=iHigh(_Symbol,PERIOD_CURRENT,2);
   double low2=iLow(_Symbol,PERIOD_CURRENT,2);
   double close2=iClose(_Symbol,PERIOD_CURRENT,2);
   double candleSize2=high2-low2;
   double candleMidPoint2=high2-(candleSize2/2);
   if(open<close)
     {
      if(open2>close2)
        {
         if(open<low2)
           {
            if(close>candleMidPoint2&&close<high2)
              {
               createObj(time,low,217, clrGreen,"Piercing");
                 {
                  return 1;
                 }
              }
           }
        }
     }
   if(open>close)
     {
      if(open2<close2)
        {
         if(open>high2)
           {
            if(close<candleMidPoint2&&close>low2)
              {
               createObj(time,high,218, clrRed,"Dark Cloud");
                 {
                  return -1;
                 }
              }
           }
        }
     }
   return 0;
  }
void createObj(datetime time, double price, int arrawCode, color clr, string txt)
  {
   string objName=" ";
   StringConcatenate(objName, "Signal@",time, "at",DoubleToString(price,_Digits),"(",arrawCode,")");
   if(ObjectCreate(0,objName,OBJ_ARROW,0,time,price))
     {
      ObjectSetInteger(0,objName,OBJPROP_ARROWCODE,arrawCode);
      ObjectSetInteger(0,objName,OBJPROP_COLOR,clr);
      if(clr==clrGreen)
         ObjectSetInteger(0,objName,OBJPROP_ANCHOR,ANCHOR_TOP);
      if(clr==clrRed)
         ObjectSetInteger(0,objName,OBJPROP_ANCHOR,ANCHOR_BOTTOM);
     }
   string candleName=objName+txt;
   if(ObjectCreate(0,candleName,OBJ_TEXT,0,time,price))
     {
      ObjectSetString(0,candleName,OBJPROP_TEXT," "+txt);
      ObjectSetInteger(0,candleName,OBJPROP_COLOR,clr);
     }
  }
```

Differences in this code

Defining candleSize2 and candleMidPoint2

```
   double candleSize2=high2-low2;
   double candleMidPoint2=high2-(candleSize2/2);
```

Conditions of the pattern

In the case of the Piercing Line pattern:

If the last candle is bullish (open<close), and the open2 is greater than close2, open is lower than low2, and close is greater than candleMidPoint2 and at the same time close is lower than high2, we need the program to return an object on the chart with green arrow and the text of "Piercing" below the low of the pattern, then terminate the function.

```
   if(open<close)
     {
      if(open2>close2)
        {
         if(open<low2)
           {
            if(close>candleMidPoint2&&close<high2)
              {
               createObj(time,low,217, clrGreen,"Piercing");
                 {
                  return 1;
                 }
              }
           }
        }
     }
```

In the case of the Dark Cloud Cover pattern:

If the last candle is bearish (open>close), and the open2 is lower than close2, open is greater than high2, and close is lower than candleMidPoint2 and at the same time close is greater than low2, we need the program to return an object on the chart with a red arrow and the text of "Dark Cloud" above the high of the pattern, then terminate the function.

```
   if(open>close)
     {
      if(open2<close2)
        {
         if(open>high2)
           {
            if(close<candleMidPoint2&&close>low2)
              {
               createObj(time,high,218, clrRed,"Dark Cloud");
                 {
                  return -1;
                 }
              }
           }
        }
     }
```

After compiling this code and executing its EA, we can get the desired signals the same as the following examples from testing:

- Piercing Line pattern:

![Piercing example](https://c.mql5.com/2/52/Piercing_example.png)

As we can see in the previous chart, we have our green arrow and piercing text below the low of the pattern the same as we need.

- Dark Cloud Cover pattern:

![Dark Cloud Cover example](https://c.mql5.com/2/52/Dark_Cloud_Cover_example.png)

As we can see in the previous chart, we have a red arrow and Dark Cloud text above the high of the pattern the same as we need.

### Three candles pattern

In this part, we will see two patterns from the blended patterns and they are the Star pattern (Morning, Evening) and the Three Inside Pattern (Up, Down).

**Star Pattern:**

- The Morning Star:

It is a three candles structure the same as we mentioned. It is formed by a small candle between two candles the first one is a long bearish and the second one is a long bullish one. The following is a graph for it:

![Morning star](https://c.mql5.com/2/52/Morning_star.png)

Based on its significance, it indicates a shift in power from selling to buying as the buyer controls the market and pushes the price higher after a decline by seller control.

- Evening Star pattern:

It is a three candles structure the same as we mentioned. It is formed by a small candle between two candles the first one is a long bullish and the second one is a long bearish one. The following is a graph for it:

![Evening star](https://c.mql5.com/2/52/Evening_star.png)

Based on its significance, it indicates a shift in power from buying to selling as the seller controls the market and pushes the price lower after a rally by buyer control.

When we want to create a program that can be used to detect this kind of pattern, we need to define the time and price of the last candle and the price data of the previous two candles of the last one, candleSize of the last three candles and compare them to each other to determine their positions related to each other to get specific signals based on specific conditions.

The following is the full code to create this program:

```
//+------------------------------------------------------------------+
//|                                        Star pattern detector.mq5 |
//|                                  Copyright 2023, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2023, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"
void OnTick()
  {
   getStar(0.5);
  }
int getStar(double middleCandleRatio)
  {
   datetime time=iTime(_Symbol,PERIOD_CURRENT,1);
   double open=iOpen(_Symbol,PERIOD_CURRENT,1);
   double high=iHigh(_Symbol,PERIOD_CURRENT,1);
   double low=iLow(_Symbol,PERIOD_CURRENT,1);
   double close=iClose(_Symbol,PERIOD_CURRENT,1);
   double open2=iOpen(_Symbol,PERIOD_CURRENT,2);
   double high2=iHigh(_Symbol,PERIOD_CURRENT,2);
   double low2=iLow(_Symbol,PERIOD_CURRENT,2);
   double close2=iClose(_Symbol,PERIOD_CURRENT,2);
   double open3=iOpen(_Symbol,PERIOD_CURRENT,3);
   double high3=iHigh(_Symbol,PERIOD_CURRENT,3);
   double low3=iLow(_Symbol,PERIOD_CURRENT,3);
   double close3=iClose(_Symbol,PERIOD_CURRENT,3);
   double candleSize=high-low;
   double candleSize2=high2-low2;
   double candleSize3=high3-low3;
   if(open<close)
     {
      if(open3>close3)
        {
         if(candleSize2<candleSize*middleCandleRatio && candleSize2<candleSize3*middleCandleRatio)
           {
            createObj(time,low,217, clrGreen,"Morning Star");
              {
               return 1;
              }
           }
        }

     }
   if(open>close)
     {
      if(open3<close3)
        {
         if(candleSize2<candleSize*middleCandleRatio && candleSize2<candleSize3*middleCandleRatio)
           {
            createObj(time,high,218, clrRed,"Evening Star");
              {
               return -1;
              }
           }
        }

     }
   return 0;
  }
void createObj(datetime time, double price, int arrawCode, color clr, string txt)
  {
   string objName=" ";
   StringConcatenate(objName, "Signal@",time, "at",DoubleToString(price,_Digits),"(",arrawCode,")");
   if(ObjectCreate(0,objName,OBJ_ARROW,0,time,price))
     {
      ObjectSetInteger(0,objName,OBJPROP_ARROWCODE,arrawCode);
      ObjectSetInteger(0,objName,OBJPROP_COLOR,clr);
      if(clr==clrGreen)
         ObjectSetInteger(0,objName,OBJPROP_ANCHOR,ANCHOR_TOP);
      if(clr==clrRed)
         ObjectSetInteger(0,objName,OBJPROP_ANCHOR,ANCHOR_BOTTOM);
     }
   string candleName=objName+txt;
   if(ObjectCreate(0,candleName,OBJ_TEXT,0,time,price))
     {
      ObjectSetString(0,candleName,OBJPROP_TEXT," "+txt);
      ObjectSetInteger(0,candleName,OBJPROP_COLOR,clr);
     }
  }
```

Differences in this code

Creating the (getStar) with the parameter on middleCandleRatio

```
int getStar(double middleCandleRatio)
```

Creating variables of time for the last candle and the price data (open, high, low, close) and the candle size for the last three candles (candlesize, candleSize2, and  candleSize3)

```
   datetime time=iTime(_Symbol,PERIOD_CURRENT,1);
   double open=iOpen(_Symbol,PERIOD_CURRENT,1);
   double high=iHigh(_Symbol,PERIOD_CURRENT,1);
   double low=iLow(_Symbol,PERIOD_CURRENT,1);
   double close=iClose(_Symbol,PERIOD_CURRENT,1);
   double open2=iOpen(_Symbol,PERIOD_CURRENT,2);
   double high2=iHigh(_Symbol,PERIOD_CURRENT,2);
   double low2=iLow(_Symbol,PERIOD_CURRENT,2);
   double close2=iClose(_Symbol,PERIOD_CURRENT,2);
   double open3=iOpen(_Symbol,PERIOD_CURRENT,3);
   double high3=iHigh(_Symbol,PERIOD_CURRENT,3);
   double low3=iLow(_Symbol,PERIOD_CURRENT,3);
   double close3=iClose(_Symbol,PERIOD_CURRENT,3);
   double candleSize=high-low;
   double candleSize2=high2-low2;
   double candleSize3=high3-low3;
```

Conditions of the pattern

In the case of the Morning Star pattern:

If the last candle is bullish (open<close), the third one is bearish (open3>close3), the candleSize2 is lower than the middleCandleRatio of the candleSize which is 0.5 and at the same time the candleSize2 is lower than the middleCandleRatio of the candleSize3, we need the program to return an object of green arrow and text of "Morning Star" below the low of the pattern, then terminate the function.

```
   if(open<close)
     {
      if(open3>close3)
        {
         if(candleSize2<candleSize*middleCandleRatio && candleSize2<candleSize3*middleCandleRatio)
           {
            createObj(time,low,217, clrGreen,"Morning Star");
              {
               return 1;
              }
           }
        }
     }
```

In the case of the Evening Star:

If the last candle is bearish (open>close), the third one is bullish (open3<close3), the candleSize2 is lower than the middleCandleRatio of the candleSize which is 0.5 and at the same time the candleSize2 is lower than the middleCandleRatio of the candleSize3, we need the program to return an object of red arrow and text of "Evening Star" above the high of the pattern, then terminate the function.

```
   if(open>close)
     {
      if(open3<close3)
        {
         if(candleSize2<candleSize*middleCandleRatio && candleSize2<candleSize3*middleCandleRatio)
           {
            createObj(time,high,218, clrRed,"Evening Star");
              {
               return -1;
              }
           }
        }
     }
```

After compiling this code without errors and executing its EA, we can get signals the same as the following examples from testing:

- The Morning Star:

![Morning star example](https://c.mql5.com/2/52/Morning_star_example.png)

As we can see that we have the desired signal of the required object on the chart below the detected pattern.

- The Evening Star:

![Evening star example](https://c.mql5.com/2/52/Evening_star_example.png)

As we can see that we have the desired signal of the required object on the chart below the detected pattern.

As a note for the Star pattern, the identical pattern formation is having a gap with the middle small candle, you can add it as an additional condition in the code, if you want to get the identical pattern.

**Three Inside pattern:**

- Three Inside Up:

It is a three-candle pattern also, the first candle is a long bearish, the second one is a small bullish candle that is trading inside the first one, and the third one is a long bullish candle that closes above the high of the first one. The following is a graph of this pattern.

![Three inside up](https://c.mql5.com/2/52/Three_inside_up.png)

Based on its significance, it indicates a potential bullishness by the control of the buyer.

- Three Inside Down:

It is a three-candle pattern also, the first candle is a long bullish, the second one is a small bearish candle that is trading inside the first one, and the third one is a long bearish candle that closes below the low of the first one. The following is a graph of this pattern.

![Three inside down](https://c.mql5.com/2/52/Three_inside_down.png)

Based on its significance, it indicates a potential bearishness by the control of the seller. If we want to create a program that can be used to detect this type of pattern, we will define also the time of the last candle and the price data of the last three candles, let the program check these values every tick and determine its positions related to each other to return a suitable signal as an object on the chart depending on the pattern. The following is the full code of this program:

```
//+------------------------------------------------------------------+
//|                                Three inside pattern detector.mq5 |
//|                                  Copyright 2023, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2023, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"
void OnTick()
  {
   getthreeInside();
  }
int getthreeInside()
  {
   datetime time=iTime(_Symbol,PERIOD_CURRENT,1);
   double open=iOpen(_Symbol,PERIOD_CURRENT,1);
   double high=iHigh(_Symbol,PERIOD_CURRENT,1);
   double low=iLow(_Symbol,PERIOD_CURRENT,1);
   double close=iClose(_Symbol,PERIOD_CURRENT,1);
   double open2=iOpen(_Symbol,PERIOD_CURRENT,2);
   double high2=iHigh(_Symbol,PERIOD_CURRENT,2);
   double low2=iLow(_Symbol,PERIOD_CURRENT,2);
   double close2=iClose(_Symbol,PERIOD_CURRENT,2);
   double open3=iOpen(_Symbol,PERIOD_CURRENT,3);
   double high3=iHigh(_Symbol,PERIOD_CURRENT,3);
   double low3=iLow(_Symbol,PERIOD_CURRENT,3);
   double close3=iClose(_Symbol,PERIOD_CURRENT,3);
   if(open3>close3)
     {
      if(open2<close2)
        {
         if(open2>low3&&close2<high3)
           {
            if(open<close&&open>open2&&open<close2)
              {
               if(close>high3)
                 {
                  createObj(time,low,217, clrGreen,"3 Inside Up");
                    {
                     return 1;
                    }
                 }
              }
           }
        }

     }
   if(open3<close3)
     {
      if(open2>close2)
        {
         if(open2<high3&&close2>low3)
           {
            if(open>close&&open<open2&&open>close2)
              {
               if(close<low3)
                 {
                  createObj(time,high,218, clrRed,"3 Inside Down");
                    {
                     return -1;
                    }
                 }
              }
           }
        }
     }
   return 0;
  }
void createObj(datetime time, double price, int arrawCode, color clr, string txt)
  {
   string objName=" ";
   StringConcatenate(objName, "Signal@",time, "at",DoubleToString(price,_Digits),"(",arrawCode,")");
   if(ObjectCreate(0,objName,OBJ_ARROW,0,time,price))
     {
      ObjectSetInteger(0,objName,OBJPROP_ARROWCODE,arrawCode);
      ObjectSetInteger(0,objName,OBJPROP_COLOR,clr);
      if(clr==clrGreen)
         ObjectSetInteger(0,objName,OBJPROP_ANCHOR,ANCHOR_TOP);
      if(clr==clrRed)
         ObjectSetInteger(0,objName,OBJPROP_ANCHOR,ANCHOR_BOTTOM);
     }
   string candleName=objName+txt;
   if(ObjectCreate(0,candleName,OBJ_TEXT,0,time,price))
     {
      ObjectSetString(0,candleName,OBJPROP_TEXT," "+txt);
      ObjectSetInteger(0,candleName,OBJPROP_COLOR,clr);
     }
  }
```

Differences in this code are conditions of the pattern

In the case of the Three Inside Up

```
   if(open3>close3)
     {
      if(open2<close2)
        {
         if(open2>low3&&close2<high3)
           {
            if(open<close&&open>open2&&open<close2)
              {
               if(close>high3)
                 {
                  createObj(time,low,217, clrGreen,"3 Inside Up");
                    {
                     return 1;
                    }
                 }
              }
           }
        }

     }
```

In the case of the Three Inside Down

```
   if(open3<close3)
     {
      if(open2>close2)
        {
         if(open2<high3&&close2>low3)
           {
            if(open>close&&open<open2&&open>close2)
              {
               if(close<low3)
                 {
                  createObj(time,high,218, clrRed,"3 Inside Down");
                    {
                     return -1;
                    }
                 }
              }
           }
        }
     }
```

After compiling this code without error and executing its EA, we can get signals the same as the following examples:

- Three Inside Up:

![Three inside up example](https://c.mql5.com/2/52/Three_inside_up_example.png)

As we can see on the chart we have the desired signal of the Three Inside Up.

- Three Inside Down:

![Three inside down example](https://c.mql5.com/2/52/Three_inside_down_example.png)

As we can see on the chart we have the desired signal of the Three Inside Down.

### Conclusion

After the previous topics in this article, it is supposed that you got the idea of how to write the code to detect the candlesticks patterns in their different formations, single candle, dual candles, and three candles patterns:

- Signal candle patterns: we learned how to detect Doji and Hammer patterns.
- Dual candle patterns: we learn how to detect Engulfing, Piecing Line, and Dark Cloud Cover patterns.
- Three candles patterns: we learned how to create a program that can detect Star patterns and Three Inside patterns.

I hope that you find this article useful to help you get better insights.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/12385.zip "Download all attachments in the single ZIP archive")

[Doji\_pattern\_detector.mq5](https://www.mql5.com/en/articles/download/12385/doji_pattern_detector.mq5 "Download Doji_pattern_detector.mq5")(1.42 KB)

[Hammer\_pattern\_detector.mq5](https://www.mql5.com/en/articles/download/12385/hammer_pattern_detector.mq5 "Download Hammer_pattern_detector.mq5")(2.75 KB)

[Engulfing\_pattern\_detector.mq5](https://www.mql5.com/en/articles/download/12385/engulfing_pattern_detector.mq5 "Download Engulfing_pattern_detector.mq5")(2.37 KB)

[Piercing\_ss\_Dark\_Cloud\_pattern\_detector.mq5](https://www.mql5.com/en/articles/download/12385/piercing_ss_dark_cloud_pattern_detector.mq5 "Download Piercing_ss_Dark_Cloud_pattern_detector.mq5")(2.44 KB)

[Star\_pattern\_detector.mq5](https://www.mql5.com/en/articles/download/12385/star_pattern_detector.mq5 "Download Star_pattern_detector.mq5")(3.32 KB)

[Three\_inside\_pattern\_detector.mq5](https://www.mql5.com/en/articles/download/12385/three_inside_pattern_detector.mq5 "Download Three_inside_pattern_detector.mq5")(2.88 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [How to build and optimize a cycle-based trading system (Detrended Price Oscillator - DPO)](https://www.mql5.com/en/articles/19547)
- [How to build and optimize a volume-based trading system (Chaikin Money Flow - CMF)](https://www.mql5.com/en/articles/16469)
- [MQL5 Integration: Python](https://www.mql5.com/en/articles/14135)
- [How to build and optimize a volatility-based trading system (Chaikin Volatility - CHV)](https://www.mql5.com/en/articles/14775)
- [Advanced Variables and Data Types in MQL5](https://www.mql5.com/en/articles/14186)
- [Building and testing Keltner Channel trading systems](https://www.mql5.com/en/articles/14169)
- [Building and testing Aroon Trading Systems](https://www.mql5.com/en/articles/14006)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/444822)**
(22)


![arispras](https://c.mql5.com/avatar/avatar_na2.png)

**[arispras](https://www.mql5.com/en/users/arispras)**
\|
19 Sep 2023 at 01:13

Good Article. Its very helpful for understand how to [create object](https://www.mql5.com/en/docs/constants/objectconstants/enum_object "MQL5 documentation: Object Types") pattern in mql5.

![Sylvanus Phillips](https://c.mql5.com/avatar/2023/10/65269448-C178.png)

**[Sylvanus Phillips](https://www.mql5.com/en/users/slyphill)**
\|
21 Mar 2024 at 17:46

Good stuff. thanks a lot .

What is the condition for both bullish and bearish Marubozu

![Ivan Titov](https://c.mql5.com/avatar/2024/9/66d71f0c-3796.png)

**[Ivan Titov](https://www.mql5.com/en/users/goldrat)**
\|
21 Aug 2024 at 15:45

**Sylvanus Phillips [#](https://www.mql5.com/en/forum/444822#comment_52799550):**

What is the condition for both bullish and bearish Marubozu

Something like this:

![](https://c.mql5.com/3/442/4893633010247.png)

![Ivan Titov](https://c.mql5.com/avatar/2024/9/66d71f0c-3796.png)

**[Ivan Titov](https://www.mql5.com/en/users/goldrat)**
\|
21 Aug 2024 at 15:53

**FINANSE-BOND [#](https://www.mql5.com/ru/forum/447191#comment_46976809):**

If someone is not too lazy, please add Buy and Sell orders to the code, since it is an Expert Advisor.

It seems easier to find a working alternative on the [Market](https://www.mql5.com/en/market/mt5?filter=%D0%BA%D0%BE%D0%BD%D1%81%D1%82%D1%80%D1%83%D0%BA%D1%82%D0%BE%D1%80).

![Adams.16](https://c.mql5.com/avatar/avatar_na2.png)

**[Adams.16](https://www.mql5.com/en/users/adams.16)**
\|
8 Dec 2024 at 10:55

Thank you very much for your work. Your work makes it possible to follow the trend and analyse with the greatest possible confidence... I have been working for a while on a pattern of two or more candles, up to 5, that can follow each other under the conditions. I would define the pattern by drawing a rectangle on it each time it appeared. After looking at your util subject, I replaced the rectangle with the signal, and the indicator was faster.But just as I failed with the rectangle, I also failed with the signal: I couldn't determine the time of the current signal as the end of a Fibo level that starts from the time of the previous signal. The level will be radius if we are at the last signal.Please help me to end my suffering related to this problem, God bless you.If you could help me please give me an example on engolfing model with high and Low as 0 and 100.God reward you.


![Category Theory in MQL5 (Part 5): Equalizers](https://c.mql5.com/2/53/Category-Theory-p5-avatar.png)[Category Theory in MQL5 (Part 5): Equalizers](https://www.mql5.com/en/articles/12417)

Category Theory is a diverse and expanding branch of Mathematics which is only recently getting some coverage in the MQL5 community. These series of articles look to explore and examine some of its concepts & axioms with the overall goal of establishing an open library that provides insight while also hopefully furthering the use of this remarkable field in Traders' strategy development.

![Moral expectation in trading](https://c.mql5.com/2/0/Moral_expectation_avatar.png)[Moral expectation in trading](https://www.mql5.com/en/articles/12134)

This article is about moral expectation. We will look at several examples of its use in trading, as well as the results that can be achieved with its help.

![Implementing the Janus factor in MQL5](https://c.mql5.com/2/53/Avatar_Implementing_the_Janus_factor_in_MQL5__1.png)[Implementing the Janus factor in MQL5](https://www.mql5.com/en/articles/12328)

Gary Anderson developed a method of market analysis based on a theory he dubbed the Janus Factor. The theory describes a set of indicators that can be used to reveal trends and assess market risk. In this article we will implement these tools in mql5.

![Canvas based indicators: Filling channels with transparency](https://c.mql5.com/2/52/filling-channels-avatar.png)[Canvas based indicators: Filling channels with transparency](https://www.mql5.com/en/articles/12357)

In this article I'll introduce a method for creating custom indicators whose drawings are made using the class CCanvas from standard library and see charts properties for coordinates conversion. I'll approach specially indicators which need to fill the area between two lines using transparency.

[![](https://www.mql5.com/ff/sh/ub4fqgrk4rkv8gz9z2/01.png)![](https://www.mql5.com/ff/sh/ub4fqgrk4rkv8gz9z2/02.png)Explore your trading for freeUpdated statistics in MetaTrader 5 will help you to thoroughly evaluate results and reduce risksLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/454106&a=bkbqgaxtrafeuegfvjisjjwjohagrvnr&s=25c5856d7857fc6b6db7cffb15ae4ce40fd19d1ab594d8a900ad65673d9ffa0e&uid=&ref=https://www.mql5.com/en/articles/12385&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5068357852566845567)

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