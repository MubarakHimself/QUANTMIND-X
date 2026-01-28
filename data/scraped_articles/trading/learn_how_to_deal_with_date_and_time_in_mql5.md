---
title: Learn how to deal with date and time in MQL5
url: https://www.mql5.com/en/articles/13466
categories: Trading, Trading Systems, Indicators, Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T18:05:44.641942
---

[Best articles and CodeBase updates in MQL5.community channelsFollow us to ensure you never miss out on important updates![](https://www.mql5.com/ff/sh/n9yf51p2srwzfqh5z2/02.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=dgazvhktsxqakdvarucjbvmvzenwlyje&s=98a038fe082e458df8c4a1d8e116e3a6646fd5517f06e48b2356b7ee005817d6&uid=&ref=https://www.mql5.com/en/articles/13466&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5069073939579207750)

MetaTrader 5 / Trading


### Introduction

It is not hidden by anyone in the financial market field, the importance of time, and how it can affect trading decisions and results. MQL5 (MetaQuotes Language 5) offers an amazing solution to deal with date and time effectively and this is what we will learn in this article because we will see how we can deal with this important topic through many applications that can be coded as a part of our trading system after understanding the most important aspects of this topic in the MQL5 programming language.

The following topics are what we will cover in this article:

- [datetime type](https://www.mql5.com/en/articles/13466#datetime)
- [MqlDateTime structure](https://www.mql5.com/en/articles/13466#structure)
- [OnTimer event](https://www.mql5.com/en/articles/13466#event)
- [NewBar application](https://www.mql5.com/en/articles/13466#NewBar)
- [Time filter application](https://www.mql5.com/en/articles/13466#filter)
- [tradeAtTime application](https://www.mql5.com/en/articles/13466#tradeAtTime)
- [Conclusion](https://www.mql5.com/en/articles/13466#conclusion)

By reading these topics, it is supposed that you will be able to use the datetime type of data in the mql5 to be used in your software as part of any one that you create because as we said the datetime topic is a very crucial topic in the trading and you must understand how to deal with as a trader or a trading system developer. So, I hope that you find this article useful and get insights about the topic of it or at least about any related topic.

Disclaimer: All information provided 'as is' only for educational purposes and is not prepared for trading purposes or advice. The information does not guarantee any kind of result. If you choose to use these materials on any of your trading accounts, you will do that at your own risk and you will be the only responsible.

### datetime type

The datetime variable type in mql5 is used to sort data of date and time in Unix time. The Unix time is the number of seconds elapsed since January 1, 1970. For more understanding, if we want to convert the date of Jan 01, 2023 at 00:00:00 GMT to Unix time, we can find that it is 1672531200 seconds. Accordingly, the time stamp is the number of seconds between a particular date and the Unix Epoch which is January 1st, 1970. We can also see the time in seconds compared to the known readable time the same the following:

| Known readable time | Seconds |
| --- | --- |
| 1 Year | 31556926 Seconds |
| 1 Month | 2629743 Seconds |
| 1 Week | 604800 Seconds |
| 1 Day | 86400 Seconds |
| 1 Hour | 3600 Seconds |

There is interesting information about the Unix time topic, if you want to read more you do that through the following [Unix Time](https://en.wikipedia.org/wiki/Unix_time "https://en.wikipedia.org/wiki/Unix_time") article in Wikipedia.

We may find ourselves needing to determine a specific date and time, this is very easy to do by using the datetime constant which begins with the D character with the date and time between single quotes the same as the following format:

D'yyyy.mm.dd hh:mm:ss'

Or:

D'dd.mm.yyyy hh:mm:ss'

The following examples from the mql5 [reference](https://www.mql5.com/en/docs/basis/types/integer/datetime) can be useful guide in this context:

```
datetime NY=D'2015.01.01 00:00';     // Time of beginning of year 2015
datetime d1=D'1980.07.19 12:30:27';  // Year Month Day Hours Minutes Seconds
datetime d2=D'19.07.1980 12:30:27';  // Equal to D'1980.07.19 12:30:27';
datetime d3=D'19.07.1980 12';        // Equal to D'1980.07.19 12:00:00'
datetime d4=D'01.01.2004';           // Equal to D'01.01.2004 00:00:00'
datetime compilation_date=__DATE__;             // Compilation date
datetime compilation_date_time=__DATETIME__;    // Compilation date and time
datetime compilation_time=__DATETIME__-__DATE__;// Compilation time
//--- Examples of declarations after which compiler warnings will be returned
datetime warning1=D'12:30:27';       // Equal to D'[date of compilation] 12:30:27'
datetime warning2=D'';               // Equal to __DATETIME__
```

The following is for printed results after compilation:

![datetime](https://c.mql5.com/2/58/datetime.png)

The same as we understood until now that datetime type holds date and time values in the number of seconds. This makes comparing or manipulating values of date and time easy. The following are examples of what we can do when dealing with date and time values:

- Comparing values of date and time.
- Adding or subtracting to date and time values.
- Converting time to string or string to time for other purposes.

**Comparing values of date and time:**

If we have two time and date values and we need to compare them for other purposes, we can do that in MQL5 by creating a variable for every value and then comparing them to each other to take an action based on the result. The following is an example of this case:

If the user enters two values of date and time, one is older than the other and we want to get a specific result for example a print statement in case the input one value is greater than the input two value then we need to receive a message of "Input one is a recent time" but if the input one value is less than the input two value then we need to get a message of "Input two is a recent time".

The following is for the code to do that:

```
//+------------------------------------------------------------------+
//|                                              dateTimeCompare.mq5 |
//+------------------------------------------------------------------+
input datetime inpDate1 = D'2023.09.01 00:00';
input datetime inpDate2 = D'2023.09.30 00:00';
//+------------------------------------------------------------------+
void OnTick()
  {
   if(inpDate1>inpDate2)
     {
      Print("Input one is a recent time");
     }
   else
      if(inpDate1<inpDate2)
        {
         Print("Input two is a recent time");
        }
  }
//+------------------------------------------------------------------+
```

After compiling this code, if we set date and time input values the same as the following:

![datetimeCompareInp](https://c.mql5.com/2/58/datetimeCompareInp.png)

As we can see, we have the input1 is older than the second one. So, we can find the result the same as the following:

![datetimeCompareresult1](https://c.mql5.com/2/58/datetimeCompareresult1__1.png)

If we changed the first input to be more recent the second one, the same as the following:

![ datetimeCompareInp2](https://c.mql5.com/2/58/datetimeCompareInp2.png)

According to the previous settings of inputs, we must find the result the same as the following:

![datetimeCompareresult2](https://c.mql5.com/2/58/datetimeCompareresult2__1.png)

The previous example of code is a sample of how we can deal with date and time as a comparison.

**Adding or subtracting to date and time values:**

We can also add to subtract to date and time values, let's say that we have two values of date and time the same as the following:

D'2023.09.01 00:00'

D'2023.09.30 00:00'

Now, we need to add one day to the first first value and subtract one day from the second value, one day in seconds is equal to 86400. so, after these two transactions, we can find the new values the same as the following:

D'2023.09.02 00:00'

D'2023.09.29 00:00'

The following code is about how to code these transactions:

```
//+------------------------------------------------------------------+
//|                                           dateTimeManipulate.mq5 |
//+------------------------------------------------------------------+
input datetime oldDate = D'2023.09.01 00:00';
input datetime newDate = D'2023.09.30 00:00';
//+------------------------------------------------------------------+
void OnTick()
  {
   datetime addToDate =   oldDate+86400;
   Print("Result of date addition - ",addToDate);
   datetime subtractFromDate =   newDate-86400;
   Print("Result of date subtraction - ",subtractFromDate);
  }
//+------------------------------------------------------------------+
```

The result of executing this software will be the same as we mentioned as the following:

![ dateTimeMani](https://c.mql5.com/2/58/dateTimeMani__1.png)

**Converting time to string or string to time for other purposes:**

We can find ourselves needing to convert the type of date and time values, at this time, we have two functions that can deal with converting between string and datetime. Either TimeToString or StringToTime, we will see now in this part an example of how we can deal with a case like that the same as the following code:

```
//+------------------------------------------------------------------+
//|                                              dateTimeConvert.mq5 |
//+------------------------------------------------------------------+
input datetime oldDate = D'2023.09.01 00:00';
input string newDate = "2023.09.30 00:00";
//+------------------------------------------------------------------+
void OnTick()
  {
   string newOldDate = TimeToString(oldDate);
   datetime newNewDate = StringToTime(newDate);
   Print("Time To String - ",newOldDate);
   Print("String To Time - ",newNewDate);
  }
//+------------------------------------------------------------------+
```

By executing this code in the trading terminal, we can find the returned values for each function first one for converting from TimeToString and the second one for converting from StringToTime the same as the following:

![ dateTimeConvert](https://c.mql5.com/2/58/dateTimeConvert.png)

Generally speaking, if we want to know and understand all operations about datetime type in the MQL5,

- currentTime: to return the current time.
- Comparing between values: logically, we use this operation to check about the time comparison.
- Manipulation with date and time data: like addition and distracting.
- format date and time.

### MqlDateTime structure

In this part, we will learn the MqlDateTime structure. It contains variables that can hold data of date and time or datetime value. The following is the structure the same as we can find the MQL5 [reference](https://www.mql5.com/en/docs/constants/structures/mqldatetime):

```
struct MqlDateTime
  {
   int year;           // Year
   int mon;            // Month
   int day;            // Day
   int hour;           // Hour
   int min;            // Minutes
   int sec;            // Seconds
   int day_of_week;    // Day of week (0-Sunday, 1-Monday, ... ,6-Saturday)
   int day_of_year;    // Day number of the year (January 1st is assigned the number value of zero)
  };
```

As we can see it contains eight fields of int type variables.

We use (TimeToStruct) function with this structure to convert the datetime value into a variable of MqlDateTime and parameters of this function are:

- dt: date and time
- dt\_struct: structure for the adoption of values

This function return a Boolean value.

We may use also the (StructToTime) function to convert the variable of the MqlDateTime structure into a datetime value. Its parameter is only dt\_struct. It returns datetime type containing the number of seconds.

We can see an example for this MqlDateTime structure the same as the following:

If we have the datetime of D'2023.10.10 07:07:07' or any other input value we need to use MqlDateTime with it. We can do that through the following code:

```
//+------------------------------------------------------------------+
//|                                                  MqlDateTime.mq5 |
//+------------------------------------------------------------------+
input datetime dtData = D'2023.10.10 07:07:07';
void OnTick()
  {
   MqlDateTime timeStruct;
   TimeToStruct(dtData,timeStruct);
   int year = timeStruct.year;
   Print("year: ",year);
   Print("=====================");
   int month = timeStruct.mon;
   Print("month: ",month);
   Print("=====================");
   int day = timeStruct.day;
   Print("day: ",day);
   Print("=====================");
   int hour = timeStruct.hour;
   Print("hour: ",hour);
   Print("=====================");
   int minute = timeStruct.min;
   Print("minute: ",minute);
   Print("=====================");
   int second = timeStruct.sec;
   Print("second: ",second);
   Print("=====================");
   int dayofWeek = timeStruct.day_of_week;
   Print("dayofWeek: ",dayofWeek);
   Print("=====================");
   int dayofYear = timeStruct.day_of_year;
   Print("dayofYear: ",dayofYear);
  }
//+------------------------------------------------------------------+
```

By executing this EA, we can find the printed messages the same as the following:

![ mqlDateTime](https://c.mql5.com/2/58/mqlDateTime.png)

It is good to mention here that the default value of the month or day is 01 if you use less than 1 as an input the maximum default value of the month is 12 and the day is 31 if the usable input larger than this maximum value. Default values of hour as a minimum 0 or maximum 23 and minutes as a minimum 0 or maximum 59.

### OnTimer() event

In this part, we will learn what is the OnTimer() event handler and how we can use it. It can be used when you need to execute your code or trading strategy every specified number of seconds. So, if you want your code to perform a specific action at a specific time interval you can place your code in the OnTimer() event handler.

The OnTimer() event handler is a void type without parameters and it can be added to the code if you need that. To use it you have to set the timer interval in the OnInit() part by using the EventSetTimer() function with the seconds parameter. the same as the following example:

```
int OnInit()
{
EventSetTimer(60);
}
```

The previous example means that the code within the OnTimer() event handler will executed every 60 seconds.

Now, if we need to stop running the code within the OnTimer() event we can do that by using the void EventKillTimer() type function in the onDeinit() event handler or the class destructor.

### NewBar application

In this application, we will create an include file that holds classes and functions that can be used in other files like EAs to perform our objective which is reporting if we have a new bar through printing a message of "A new bar painted".

First, the following steps are for creating the include file. The main idea behind creating the include file here is to learn and adapt to building files that can be used in many projects and work professionally as a developer if you are a beginner.

We'll declare the CNewBar class that has two private variables (time\[\] to store the time of the current bar and the other variable lastTime to store the time of the most recent bar. We will include a contractor of the Boolean newBarCheck function with two parameters:

- symbol string type
- timeFrame ENUM\_TIMEFRAMES type

```
class CNewBar
  {
private:
   datetime          time[], lastTime;
public:
   void              CNewBar();
   bool              newBarCheck(string symbol, ENUM_TIMEFRAMES timeFrame);
  };
```

Setting the time array as series

```
void CNewBar::CNewBar(void)
  {
   ArraySetAsSeries(time,true);
  }
```

Creating the newBarCheck function with two parameters (symbol and timeFrame) to check if there is a new bar or not, the function will return a bool type value

```
bool CNewBar::newBarCheck(string symbol, ENUM_TIMEFRAMES timeFrame)
```

The body of the function checks if there is a new bar, Creating two bool variables for first and newBar with a default false value.

```
   bool firstCheck = false;
   bool newBar = false;
```

Use the CopyTime function with the variants of call by the first position to get the time\_array history data of the bar opening time of the specified symbol-period pair in the specified quantity. Its parameters are:

- symbol\_name: to specify the symbol name that will be symbol as a defined string one in the function.
- timeframe: to specify the time frame that will be the timeFrame as a defined ENIM\_TIMEFRAMES in the function.
- start\_pos: to specify the starting position that will be 0.
- count: to specify data to copy that will be 2.
- time\_array\[\]: the array to copy the open times from that will be the time array.

```
CopyTime(symbol,timeFrame,0,2,time);
```

We will check the lastTime class variable, if it is the first time to check then the value of the lastTime variable will be equal to 0 then the firstCheck variable will be true

```
   if(lastTime == 0)
     {
      firstCheck = true;
     }
```

Checking if the time\[0\] is greater than the lastTime value then checking if the firstCheck value is equal to false, so the newBar will be true then lastTime will be the same value as time\[0\]

```
   if(time[0] > lastTime)
     {
      if(firstCheck == false)
      {
         newBar = true;
      }
      lastTime = time[0];
     }
```

Return the newBar value that will be a Boolean value.

```
return(newBar);
```

The following is for the full code of this include file in one block of code:

```
//+------------------------------------------------------------------+
//|                                                       newBar.mqh |
//+------------------------------------------------------------------+
class CNewBar
  {
private:
   datetime          time[], lastTime;
public:
   void              CNewBar();
   bool              newBarCheck(string symbol, ENUM_TIMEFRAMES timeFrame);
  };
//+------------------------------------------------------------------+
void CNewBar::CNewBar(void)
  {
   ArraySetAsSeries(time,true);
  }
//+------------------------------------------------------------------+
bool CNewBar::newBarCheck(string symbol, ENUM_TIMEFRAMES timeFrame)
  {
   bool firstCheck = false;
   bool newBar = false;
   CopyTime(symbol,timeFrame,0,2,time);
   if(lastTime == 0)
     {
      firstCheck = true;
     }
   if(time[0] > lastTime)
     {
      if(firstCheck == false)
      {
         newBar = true;
      }
      lastTime = time[0];
     }
   return(newBar);
  }
//+------------------------------------------------------------------+
```

If you want to learn more about the object oriented programming (OOP) and classes in MQL5, you can read my previous [Understanding MQL5 Object-Oriented Programming (OOP)](https://www.mql5.com/en/articles/12813) article you may find it useful.

Now, after creating the included file, we will create our EA that will detect the new bar and print a message of "A new bar painted, you can trade". First, we will include the created include file in the EA code by using the preprocessor #include

```
#include <dateTime.mqh>
```

Declare an object from the CNewBar class

```
CNewBar NewBar;
```

Create a bool variable as an input to choose if I will trade when a new bar is painted (true) or not (false)

```
bool newBarTrade=true;
```

After that, in the OnTick() part, we will create a bool variable of newBar with a default true value and another integer barShift variable with a default 0 value

```
   bool newBar=true;
   int barShift=0;
```

Checking if the newBarTrade is true, then the newBar will be equal to Newbarobject.newBarCheck with parameters the current symbol and the current time frame, and the barShift will be equal to 1.

```
   if(newBarTrade==true)
     {
      newBar = NewBar.newBarCheck(Symbol(),Period());
      barShift=1;
     }
```

Checking if the newBar is equal to true, this means that a new bar generated so we need to get the message of "A new bar painted, you can trade"

```
   if(newBar==true)
     {
      Print("A new bar painted, you can trade");
     }
```

The following is the full code for this application in one block of code

```
//+------------------------------------------------------------------+
//|                                                       newBar.mq5 |
//+------------------------------------------------------------------+
#include <dateTime.mqh>
CNewBar NewBar;
bool newBarTrade=true;
//+------------------------------------------------------------------+
void OnTick()
  {
   bool newBar=true;
   int barShift=0;
   if(newBarTrade==true)
     {
      newBar = NewBar.newBarCheck(Symbol(),Period());
      barShift=1;
     }
   if(newBar==true)
     {
      Print("A new bar painted, you can trade");
     }
  }
//+------------------------------------------------------------------+
```

After executing this EA, we can find the following result once a new bar generated

![newBar alert](https://c.mql5.com/2/58/newBar_alert.png)

This code can be developed to change the action when a new bar is painted as per your strategy like for example open a trade or something else but we are sharing in this article simple code to understand the main idea of the code and you can develop it as needed.

### Time filter application

In this application we need to create a filter application that can be used to allow or prevent trading based on a time filter or a specified time period. We'll add to the created include file a new class named CTimeFilter and create a public bool timeCheck function with three parameters (startTime, endTime, localTime with a default false value.

```
class CTimeFilter
  {
public:
   bool timeCheck(datetime startTime, datetime endTime, bool localTime = false);
  };
```

Create the body of the timeCheck function by checking if the data input was wrong by getting a startTime is greater than or equal to endTime then return an error alert "Error: Invalid Time input" and return false

```
   if(startTime >= endTime)
     {
      Alert("Error: Invalid Time input");
      return(false);
     }
```

Creating a datetime variable for the currentTime then checking if the localTime is true the TimeLocal() will be assigned to currentTime or TimeCurrent() will be assigned to currentTime

```
   datetime currentTime;
   if(localTime == true)
     {
      currentTime = TimeLocal();
     }
   else
      currentTime = TimeCurrent();
```

Creating a boolean variable named timeFilterActive with a default false value then checking if the currentTime is greater than or equal to the startTime and at the same time it is less than endTime then return true for the timeFilterActive. return the timeFilterActive value

```
   bool timeFilterActive = false;
   if(currentTime >= startTime && currentTime < endTime)
     {
      timeFilterActive = true;
     }
   return(timeFilterActive);
```

So, the following is the full code in the include file in one block

```
class CTimeFilter
  {
public:
   bool timeCheck(datetime startTime, datetime endTime, bool localTime = false);
  };
bool CTimeFilter::timeCheck(datetime startTime, datetime endTime, bool localTime = false)
  {
   if(startTime >= endTime)
     {
      Alert("Error: Invalid Time input");
      return(false);
     }
   datetime currentTime;
   if(localTime == true)
     {
      currentTime = TimeLocal();
     }
   else
      currentTime = TimeCurrent();
   bool timeFilterActive = false;
   if(currentTime >= startTime && currentTime < endTime)
     {
      timeFilterActive = true;
     }
   return(timeFilterActive);
  }
```

After creating the include file, we will create our timeFilterApp EA through the following step:

Include the dateTime file to use its content

```
#include <dateTime.mqh>
```

Creating two datetime variable as input values named StartTime and EndTime

```
input datetime StartTime = D'2023.10.10 10:00';
input datetime EndTime = D'2023.10.10 17:00';
```

Create an object named filter in the global scope

```
CTimeFilter filter;
```

In the OnTick() part, create timeFilterActive and assign to its filter.timeCheck with its parameters

```
bool timeFilterActive = filter.timeCheck(StartTime,EndTime,false);
```

To check if the timeFilterActive is true we need to get the message "Trading is active based on time filter" or the message "Trading is inactive based on time filter"

```
   if(timeFilterActive == true)
     {
      Print("Trading is active based on time filter");
     }
     else Print("Trading is inactive based on time filter");
```

The following is the full code in one block

```
//+------------------------------------------------------------------+
//|                                                timeFilterApp.mq5 |
//+------------------------------------------------------------------+
#include <dateTime.mqh>
input datetime StartTime = D'2023.10.10 10:00';
input datetime EndTime = D'2023.10.10 17:00';
CTimeFilter filter;
void OnTick()
  {
   bool timeFilterActive = filter.timeCheck(StartTime,EndTime,false);
   if(timeFilterActive == true)
     {
      Print("Trading is active based on time filter");
     }
     else Print("Trading is inactive based on time filter");
  }
```

After executing this application we can find one of three results as per time input.

In case of active time:

![ timeFilterAppinpactive](https://c.mql5.com/2/58/timeFilterAppinpactive.png)

![timeFilterApp](https://c.mql5.com/2/58/timeFilterApp.png)

In case of inactive time:

![timeFilterAppinpinactive](https://c.mql5.com/2/58/timeFilterAppinpinactive.png)

![ timeFilterAppinactive](https://c.mql5.com/2/58/timeFilterAppinactive.png)

In case of false inputs:

![ timeFilterAppinperror.](https://c.mql5.com/2/58/timeFilterAppinperror.png)

![timeFilterApperror](https://c.mql5.com/2/58/timeFilterApperror__1.png)

There is another method that we can use without include file and the following are steps to do that.

Create four input int variables for starting hour and minute and ending hour and minute globally

```
input int TimeStartHour =
10 ;
input int TimeStartMin = 0;
input int TimeEndHour = 17;
input int TimeEndMin =
0 ;
```

In the OnTick() part, we will create structTime variable using the MqlDateTime function that contains date and time data

```
MqlDateTime structTime;
```

Passing the created structTime variable to TimeCurrent function

```
TimeCurrent(structTime);
```

Setting second to 0, hour, and minutes for the starting time

```
   structTime.sec = 0;
   structTime.hour = TimeStartHour;
   structTime.min = TimeStartMin;
```

Creating a datetime variable for timeStart and passing structTime to the StructToTime function which converts a structure variable MqlDateTime into a value of datetime type then assign it to the timeStart variable

```
datetime timeStart = StructToTime(structTime);
```

Setting hours and minutes for the ending time

```
   structTime.hour = TimeEndHour;
   structTime.min = TimeEndMin;
```

Creating timeEnd variable the same as we did with timeStart

```
datetime timeEnd = StructToTime(structTime);
```

Creating a boolean isTime variable to determine the allowed time to trade

```
bool isTime = TimeCurrent() >= timeStart && TimeCurrent() < timeEnd;
```

Creating Print error message in case of false inputs

```
   if(TimeStartHour >= TimeEndHour)
     {
      Print("Error: Invalid Time input");
     }
```

Setting conditions of trading time

```
   if(isTime==true)
     {
      Print("Trading is active based on time filter");
     }
   else
      Print("Trading is inactive based on time filter");
```

The following is for the full code in one block:

```
//+------------------------------------------------------------------+
//|                                               timeFilterApp2.mq5 |
//+------------------------------------------------------------------+
input int TimeStartHour = 10;
input int TimeStartMin = 0;
input int TimeEndHour = 17;
input int TimeEndMin = 0;
//+------------------------------------------------------------------+
void OnTick()
  {
   MqlDateTime structTime;
   TimeCurrent(structTime);
   structTime.sec = 0;
   structTime.hour = TimeStartHour;
   structTime.min = TimeStartMin;
   datetime timeStart = StructToTime(structTime);
   structTime.hour = TimeEndHour;
   structTime.min = TimeEndMin;
   datetime timeEnd = StructToTime(structTime);
   bool isTime = TimeCurrent() >= timeStart && TimeCurrent() < timeEnd;
   if(TimeStartHour >= TimeEndHour)
     {
      Print("Error: Invalid Time input");
     }
   if(isTime==true)
     {
      Print("Trading is active based on time filter");
     }
   else
      Print("Trading is inactive based on time filter");
  }
//+------------------------------------------------------------------+
```

By executing this code we can find results the same as the following:

In case of active:

![ timeFilterApp2inpactive](https://c.mql5.com/2/58/timeFilterApp2inpactive.png)

![timeFilterApp2active](https://c.mql5.com/2/58/timeFilterApp2active.png)

In case of inactive time to trade

![timeFilterApp2inpinactive](https://c.mql5.com/2/58/timeFilterApp2inpinactive.png)

![ timeFilterApp2inactiv](https://c.mql5.com/2/58/timeFilterApp2inactive.png)

In case of false inputs

![ timeFilterApp2inperror](https://c.mql5.com/2/58/timeFilterApp2inperror.png)

![ timeFilterApp2error](https://c.mql5.com/2/58/timeFilterApp2error.png)

### tradeAtTime application

Let's say that we need to set a specific time that we need to only trade at, we will need to set this time as an allowed time to trade at and the following are steps to do that.

Creating input string type of openTime variable with a default value "10:00" globally

```
input string openTime="10:00";
```

Creating a datetime variable for lastTime

```
datetime lastTime;
```

In the OnTick() part, we will create a datatime dtOpenTime variable to assign to it the StringToTime values of the openTime

```
datetime dtOpenTime=StringToTime(openTime);
```

Checking the condition of allowed time, if the last allowed time is not equal to dtOpenTime and at the same time timeCurrent() is greater than the dtOpenTime, we need the update the value of lastTime with the dtOpenTime and print the message "Now is the allowed time to trade". This printed message will be printed one time only at the specified time

```
   if(lastTime !=dtOpenTime && TimeCurrent()>dtOpenTime)
     {
      lastTime=dtOpenTime;
      Print("Now is the allowed time to trade");
     }
```

the following is the full code of this software:

```
//+------------------------------------------------------------------+
//|                                                  tradeAtTime.mq5 |
//+------------------------------------------------------------------+
input string openTime="10:00";
datetime lastTime;
//+------------------------------------------------------------------+
void OnTick()
  {
   datetime dtOpenTime=StringToTime(openTime);
   if(lastTime !=dtOpenTime && TimeCurrent()>dtOpenTime)
     {
      lastTime=dtOpenTime;
      Print("Now is the allowed time to trade");
     }
  }
//+------------------------------------------------------------------+
```

After executing the code we can find its signal the same as the following:

![ tradeAtTime](https://c.mql5.com/2/58/tradeAtTime.png)

### Conclusion

At the end of this article, I hope that you understood what is the datetime and how we can deal with date and time in the MQL5 for customization based on time consideration. We identified the datetime in the MQL5, and how we can use it in addition to that we applied that to simple applications that had to enhance our understanding of this important topic. These applications can be part of any software to handle date and time effectively. So, it is supposed that we understood the following topic

- datetime type
- MqlDateTime structure
- OnTimer event
- NewBar application
- Time filter application
- tradeAtTime application

There are many applications that can be used in terms of date and time in our trading and we are not able to mention them all but by understanding the basics of the topic we can apply them to any application we need.

I hope that you found this article useful and helped you in the journey of learning the MQL5 programming language, if you found this article useful and you need to read more articles you can check my publication, you can find articles about how to create trading systems based on the most popular technical indicators and other articles in the context of learning the MQL5 programming language and I hope that you find them useful as well.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/13466.zip "Download all attachments in the single ZIP archive")

[dateTime.mqh](https://www.mql5.com/en/articles/download/13466/datetime.mqh "Download dateTime.mqh")(2.1 KB)

[newBar.mq5](https://www.mql5.com/en/articles/download/13466/newbar.mq5 "Download newBar.mq5")(0.68 KB)

[timeFilterApp.ex5](https://www.mql5.com/en/articles/download/13466/timefilterapp.ex5 "Download timeFilterApp.ex5")(6.52 KB)

[timeFilterApp2.mq5](https://www.mql5.com/en/articles/download/13466/timefilterapp2.mq5 "Download timeFilterApp2.mq5")(1.12 KB)

[tradeAtTime.mq5](https://www.mql5.com/en/articles/download/13466/tradeattime.mq5 "Download tradeAtTime.mq5")(0.62 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/455892)**
(11)


![Maxim Kuznetsov](https://c.mql5.com/avatar/2016/1/56935A91-AF51.png)

**[Maxim Kuznetsov](https://www.mql5.com/en/users/nektomk)**
\|
25 Feb 2024 at 16:32

**JRandomTrader [#](https://www.mql5.com/ru/forum/462997#comment_52516640):**

He's taking seconds from the beginning of the month to make the number shorter.

Although with almost (if you don't need to recalculate the object name to the day of the month) the same result you can take the remainder of dividing TimeCurrent() by a number close to the average number of seconds in the month.

and subtract from one date another date (the beginning of the month) does not allow some cunning religion ? :-)

In practice - if somewhere inside you need to know the exact beginning of the month, day, week, they are counted and memorised once. So that you don't have to recalculate them for every sneeze.

datetime thisDayTime, nextDayTime;

datetime thisYearTime,thisMonTime;

int thisYear, thisMon;

// вызывать внутри всех обработчиков OnXXX терминала- это прототип службы времени

void TimeService() {

    datetime now=TimeCurrent();

    if (now>=nextDayTime) {

        MqlDateTime dt; TimeToStruct(dt,now);

        dt.hour=dt.min=dt.sec=0;

        thisDayTime=StructToTime(dt);

        nextDayTime=thisDayTime+24\*60\*60;

        OnDay();  /// обработчик "начало дня"

        if (dt.month!=thisMon\|\|dt.year!=thisYear) {

               thisMon=dt.month;

               dt.day=0; thisMon=StructToTime(dt);

               OnMon();   /// обработчик "начало месяца"

        }

         if ( dt.year!=thisYear ) {

thisYear=dt.year;

               dt.month=0; thisYearTime=StructToTime(dt);

               OnYear();  /// обработчик "начало года"

        }

    }

}

![JRandomTrader](https://c.mql5.com/avatar/avatar_na2.png)

**[JRandomTrader](https://www.mql5.com/en/users/jrandomtrader)**
\|
26 Feb 2024 at 09:39

Sorry, I just thought of something on the subject:

```
Фредерик Браун
Конец начал

Профессор Джонс долгое время работал над теорией времени.
- И сегодня я нашел ключевое уравнение, - сказал он своей дочери как-то утром.
- Время это поле. Я создал  машину, которая способна управлять этим полем.
Он протянул руку и, нажимая кнопку, сказал:
- Это заставит время идти назад идти время заставит это
- :сказал, кнопку нажимая, и руку протянул он.
- Полем этим управлять способна которая, машину создал я. Поле это время,
- утром как-то дочери своей он сказал. - Уравнение ключевое нашел я сегодня и.
Времени теорией над работал время долгое Джонс профессор.

Начал конец
Браун Фредерик
```

![Sergey Gridnev](https://c.mql5.com/avatar/2014/5/53726F63-E57D.jpg)

**[Sergey Gridnev](https://www.mql5.com/en/users/contender)**
\|
26 Feb 2024 at 09:47

**JRandomTrader [#](https://www.mql5.com/ru/forum/462997#comment_52522562):**

Sorry, I just thought of something on the subject:

Probably should have reversed not only the word order, but also the letter order ;)


![Edgar Akhmadeev](https://c.mql5.com/avatar/avatar_na2.png)

**[Edgar Akhmadeev](https://www.mql5.com/en/users/dali)**
\|
26 Feb 2024 at 13:08

**JRandomTrader [#](https://www.mql5.com/ru/forum/462997#comment_52522562):**

Sorry, I just thought of something on the subject:

I remembered a Soviet sci-fi story where the daughter kept telling him "ukponk imzhan" before her father's experiment at the time institute, and he didn't understand her childish pampering.

And she saved him with it.

![Hilario Miguel Ofarril Gonzalez](https://c.mql5.com/avatar/avatar_na2.png)

**[Hilario Miguel Ofarril Gonzalez](https://www.mql5.com/en/users/hilariomiguelofarrilgonzalez)**
\|
22 Mar 2024 at 04:58

**MetaQuotes:**

Published article [We work with dates and times in MQL5](https://www.mql5.com/en/articles/13466):

Author: [Mohamed Abdelmaaboud](https://www.mql5.com/en/users/M.Aboud "M.Aboud")

Very valuable to be able to understand and then see clearly a very important point we know that a second is different in an hour marked with the distance and if we travel depending on where we are if we are not synchronized at the time of working our watches can not give the time with ezaptitud depending on how we move forward .


![Integrate Your Own LLM into EA (Part 1): Hardware and Environment Deployment](https://c.mql5.com/2/59/Hardware_icon_up__1.png)[Integrate Your Own LLM into EA (Part 1): Hardware and Environment Deployment](https://www.mql5.com/en/articles/13495)

With the rapid development of artificial intelligence today, language models (LLMs) are an important part of artificial intelligence, so we should think about how to integrate powerful LLMs into our algorithmic trading. For most people, it is difficult to fine-tune these powerful models according to their needs, deploy them locally, and then apply them to algorithmic trading. This series of articles will take a step-by-step approach to achieve this goal.

![Launching MetaTrader VPS: A step-by-step guide for first-time users](https://c.mql5.com/2/59/2023-10-17_15-20-00.png)[Launching MetaTrader VPS: A step-by-step guide for first-time users](https://www.mql5.com/en/articles/13586)

Everyone who uses trading robots or signal subscriptions sooner or later recognizes the need to rent a reliable 24/7 hosting server for their trading platform. We recommend using MetaTrader VPS for several reasons. You can conveniently pay and manage the subscription through your MQL5.community account.

![Discrete Hartley transform](https://c.mql5.com/2/57/discrete_hartley_transform_avatar.png)[Discrete Hartley transform](https://www.mql5.com/en/articles/12984)

In this article, we will consider one of the methods of spectral analysis and signal processing - the discrete Hartley transform. It allows filtering signals, analyzing their spectrum and much more. The capabilities of DHT are no less than those of the discrete Fourier transform. However, unlike DFT, DHT uses only real numbers, which makes it more convenient for implementation in practice, and the results of its application are more visual.

![Mastering ONNX: The Game-Changer for MQL5 Traders](https://c.mql5.com/2/59/Mastering_ONNX_logo_up.png)[Mastering ONNX: The Game-Changer for MQL5 Traders](https://www.mql5.com/en/articles/13394)

Dive into the world of ONNX, the powerful open-standard format for exchanging machine learning models. Discover how leveraging ONNX can revolutionize algorithmic trading in MQL5, allowing traders to seamlessly integrate cutting-edge AI models and elevate their strategies to new heights. Uncover the secrets to cross-platform compatibility and learn how to unlock the full potential of ONNX in your MQL5 trading endeavors. Elevate your trading game with this comprehensive guide to Mastering ONNX

[Best articles and CodeBase updates in MQL5.community channelsFollow us to ensure you never miss out on important updates![](https://www.mql5.com/ff/sh/n9yf51p2srwzfqh5z2/02.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=dgazvhktsxqakdvarucjbvmvzenwlyje&s=98a038fe082e458df8c4a1d8e116e3a6646fd5517f06e48b2356b7ee005817d6&uid=&ref=https://www.mql5.com/en/articles/13466&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5069073939579207750)

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