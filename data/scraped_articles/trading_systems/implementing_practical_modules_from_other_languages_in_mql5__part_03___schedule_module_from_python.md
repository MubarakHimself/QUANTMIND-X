---
title: Implementing Practical Modules from Other Languages in MQL5 (Part 03): Schedule Module from Python, the OnTimer Event on Steroids
url: https://www.mql5.com/en/articles/18913
categories: Trading Systems, Machine Learning, Strategy Tester
relevance_score: 6
scraped_at: 2026-01-23T11:34:57.894289
---

[![](https://www.mql5.com/ff/sh/0wxx5f0vuwq7xh89z2/01.png)VPS for 24/7 tradingContact your broker and find out how to get a free hosting subscriptionLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=nhetzvgituppcfrhndpblbihmzziogdh&s=d00c975c8bda3d8c1b29f042ad33ac81952ccea2f130a8f1ffa9015bab8ade87&uid=&ref=https://www.mql5.com/en/articles/18913&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5062568305371292846)

MetaTrader 5 / Tester


**Contents**

- [Introduction](https://www.mql5.com/en/articles/18913#para1)
- [What is the Schedule module offered in Python](https://www.mql5.com/en/articles/18913#what-is-schedule-module-Python)?
- [The Schedule class in MQL5](https://www.mql5.com/en/articles/18913#schedule-class-MQL5)
- [Running functions at a specific time](https://www.mql5.com/en/articles/18913#running-fxs-at-specific-time)
- [Passing arguments to a task](https://www.mql5.com/en/articles/18913#passing-arguments-to-task)
- [Running a task until a certain time](https://www.mql5.com/en/articles/18913#running-task-until)
- [Running all jobs regardless of their schedules](https://www.mql5.com/en/articles/18913#runing-all-jobs-at-once)
- [Managing schedules](https://www.mql5.com/en/articles/18913#managing-schedules)
- [Dealing with Timezones](https://www.mql5.com/en/articles/18913#dealing-w-timezones)
- [Applications of the schedule module in your trading apps](https://www.mql5.com/en/articles/18913#applications-of-schedule)
- [Schedule vs OnTimer](https://www.mql5.com/en/articles/18913#schedule-vs-ontimer)
- [Conclusion](https://www.mql5.com/en/articles/18913#para2)

### Introduction

Programming is intended to make our life easier by letting us automate many of the crucial and sometimes boring/repetitive tasks that we often want computers to carry out without any human interaction(s). A good example is the autosave function that we see in many text editors, instead of having to worry about saving documents every time you write a new word, text editors handle the saving process automatically so that you can focus on writing and not worry about losing your work once the unexpected happens.

This is not different in the trading space, where numerous repetitive activities and tasks aid in trading that we want to automate using some lines of code.

![source: unsplash.com](https://c.mql5.com/2/158/karolina-santos-PKzz5eF_2-c-unsplash.png)

In MQL5 programming language, we have the notorious [OnTimer function](https://www.mql5.com/en/docs/event_handlers/ontimer)  which helps in running certain functions and lines of code after a specific time interval set by the programmer.

Below is a simple example — _running the OnTimer function after every 10 seconds._

```
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {

   EventSetTimer(10); //Creates a timer with 10 seconds period

//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
    EventKillTimer(); //Destroy the timer after completing the work
  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//---

  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void OnTimer(void)
  {
     Print("Ontimer called: ",TimeLocal());   //This line of code will be run after every 10 seconds
  }
```

Outputs.

```
MN      0       10:16:39.455    Schedule test (XAUUSD,D1)       Ontimer called: 2025.07.21 10:16:39
CD      0       10:16:49.459    Schedule test (XAUUSD,D1)       Ontimer called: 2025.07.21 10:16:49
```

This function is decent, but it is crude and not flexible enough to allow multiple/different schedules to operate simultaneously in the same program.

Once you set the OnTimer event handler in your Expert Advisor or Indicator, you can only rely on that single "timer schedule". This is very limiting because we often have different tasks that we want to be executed at different times (intervals) in our programs.

For example, Printing or sending daily, weekly, or monthly trading reports to the users.

In the Python programming language, there is a module similar to the OnTimer function, but it is way better at running functions timely. In this article, we will discuss it and implement a similar module in the MQL5 programming language.

### What is the Schedule Module Offered in Python?

_[Dubbed as — Python job scheduling for humans](https://www.mql5.com/go?link=https://schedule.readthedocs.io/en/stable/examples.html%23examples "https://schedule.readthedocs.io/en/stable/examples.html#examples")_

This is a human-friendly Python module that helps in scheduling particular tasks to be run after a specific time of the day, week, etc. This module is simple to use and lightweight, making it a necessary module that every Python developer needs to be aware of.

Unlike the OnTimer function we have in MQL5, the _schedule_  module does not only allow us to schedule tasks to run after a specific time interval, it also gives us the flexibility to be more specific of when and how a particular task (function) is supposed to run.

Below are some of the functions offered by this module.

[Imports](https://www.mql5.com/go?link=https://schedule.readthedocs.io/en/stable/installation.html "https://schedule.readthedocs.io/en/stable/installation.html")

```
import schedule
```

| Function | Description |
| --- | --- |
| ```<br>schedule.every(10).minutes.do(job)<br>``` | Similarly to the OnTimer event, after every 10 minutes, the function named _job_ will be run. |
| ```<br>schedule.every().hour.do(job)<br>``` | The function named _job_ will be run every hour from the start of the script. |
| ```<br>schedule.every().day.at("10:30").do(job)<br>``` | The function named _job_ will run every day at a specific local time 10:30 in 24-Hour time. |
| ```<br>schedule.every().monday.do(job)<br>``` | The function named _job_ will be called into action every Monday at the exact time the script was initially run. |
| ```<br>schedule.every().wednesday.at("13:15").do(job)<br>``` | The function named _job_ will be called into action on every wednesday at 13:15. |
| ```<br>schedule.every().day.at("12:42", "Europe/Amsterdam").do(job)<br>``` | The function named _job_ will be called every day at 12:42 according to Europe/Amsterdam time. |
| ```<br>schedule.every().minute.at(":17").do(job) <br>``` | The function named _job_ will be called every minute at the 17th second. |

These are just a few, yet crucial functions offered in this module. Let's implement a familiar-looking class in MQL5.

### The Schedule Class in MQL5

The schedule module in Python is built to occupy separate functions for every task you want to schedule in a specific interval. The function named **do**  is the endpoint of all the function chaining from the schedule class.

```
schedule.every(10).minutes.do
```

To accomplish a similar syntax in MQL5, we have to make some functions in the class _CSchedule_ return an instance of the whole class, except the function named **dO**, which is the endpoint.

```
class CSchedule
  {
private:

   int               m_period; //the number of seconds, minutes, etc to use
   time_intervals_enum m_unit; //time interval: minutes, hours, etc
   int               m_time_seconds; //datetime in seconds
   JobFunction       m_func; //The function to run for the current schedule

public:

                     int  m_fixed_time;  // time from midnight in seconds

                     CSchedule(void);
                    ~CSchedule(void);

                     CSchedule*  every(int period = 1);
                     CSchedule*  seconds();
                     CSchedule*  minutes();
                     CSchedule*  hours();
                     CSchedule*  days();
                     CSchedule*  weeks();
                     CSchedule*  months();
                     CSchedule*  years();

                     void  dO(JobFunction func);
 }
```

This syntax allows us to have a similar fluent interface like the one offered by the _schedule_ module from Python.

```
#include <schedule.mqh>
CSchedule schedule;
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   schedule.every().minutes().dO(runthis);
  }
```

The function named _every_ is crucial for setting an interval of a specific timeframe. For example:

```
   schedule.every(10).minutes()
```

Means after every 10 minutes, a specific function received by the function named _dO_ should get triggered.

At its core, this function takes a given _period_ and assigns that variable to a variable named _m\_period_— _stored inside the class._

```
CSchedule* CSchedule::every(int period = 1)
  {
   m_period = period;
   return GetPointer(this);
  }
```

The functions: _seconds, minutes, hours,_ etc, assigns the timeframe variable according to all available time interval options given by the [enumerator](https://www.mql5.com/en/docs/basis/types/integer/enumeration) named _time\_intervals\_enum._

```
enum time_intervals_enum
 {
   SECONDS,
   MINUTES,
   HOURS,
   DAYS,
   WEEKS,
   MONTHS,
   YEARS
};
```

```
CSchedule* CSchedule::seconds()
 {
   m_unit = SECONDS;
   return GetPointer(this);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CSchedule* CSchedule::minutes()
 {
   m_unit = MINUTES;
   return GetPointer(this);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CSchedule* CSchedule::hours()
 {
   m_unit = HOURS;
   return GetPointer(this);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CSchedule* CSchedule::days()
  {
   m_unit = DAYS;
   return GetPointer(this);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CSchedule* CSchedule::weeks(void)
  {
   m_unit = WEEKS;
   return GetPointer(this);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CSchedule* CSchedule::months(void)
  {
   m_unit = MONTHS;
   return GetPointer(this);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CSchedule* CSchedule::years(void)
  {
   m_unit = YEARS;
   return GetPointer(this);
  }
```

Before we understand the function named _dO_, which is the endpoint all the scheduling functions. Let's understand how each job (function) is handled and stored inside the file named _schedule.mqh._

```
//+------------------------------------------------------------------+
//| Handling and storing every job (task) used in the class CSchedule|
//+------------------------------------------------------------------+
typedef void (*JobFunction)();  // For global functions

struct jobs_struct
{
    int prev_run;
    int next_run;
    int interval;

    JobFunction func;  // Store the function pointer
};

jobs_struct m_jobs[];  // Global job list

void jobs_add(jobs_struct &jobs_array[], const jobs_struct &job)
  {
   uint size = ArraySize(jobs_array);
   ArrayResize(jobs_array, size + 1);
   jobs_array[size] = job;
  }
```

Since the class _CSchedule_ has functions that refer to itself, handling each job object inside this class becomes complicated/confusing, and error-prone.

Having a global array named _m\_jobs_ provides a universal way of storing and handling all jobs used inside the class. _We'll discuss in a moment._

The function named _dO_ receives a function to run repeatedly according to a received "schedule". It calculates the last time a function ran and the next time it is expected to run.

A function received is stored in the structure named _jobs\_struct_ alongside other jobs' properties such as the last time, and next time the job will run.

All these values are then stored inside an array named _m\_jobs_ which is an array of _jobs\_struct_ type.

```
jobs_struct m_jobs[];  // Global job list
```

```
void CSchedule::dO(JobFunction func)
 {
      m_func = func;

      jobs_struct job;
      job.func = m_func;

      datetime now = TimeLocal();
      job.prev_run = (int)now;
      job.interval = timedelta(m_period, m_unit); //Get configs from the every() method and above
      job.next_run = job.prev_run + timedelta(m_period, m_unit);

    if (MQLInfoInteger(MQL_DEBUG))
      Print("The first function run is schedule at: ", TimeToString((datetime)job.next_run, TIME_DATE | TIME_SECONDS));

//---
      jobs_add(m_jobs, job); //store the job object to the list of jobs
 }
```

After a job is stored in its appropriate array, we need a universal function for constantly monitoring it and running its function when its scheduled time arrives.

```
void CSchedule::run_pending()
  {
   int now = (int)TimeLocal();
   for(int i = 0; i < ArraySize(m_jobs); i++)
     {
      if(now >= m_jobs[i].next_run)
        {
         if(m_jobs[i].func != NULL)
            m_jobs[i].func();

         m_jobs[i].prev_run = m_jobs[i].next_run;
         // Recalculate next_run
         m_jobs[i].next_run += m_jobs[i].interval;

         if (MQLInfoInteger(MQL_DEBUG))
            printf("Prev run: %s Next run: %s", TimeToString((datetime)m_jobs[i].prev_run, TIME_DATE|TIME_SECONDS), TimeToString((datetime)m_jobs[i].next_run, TIME_DATE|TIME_SECONDS));
        }
     }
  }
```

Let's schedule our very first job using this class.

```
#include <schedule.mqh>
CSchedule schedule;
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
//---

   schedule.every(10).seconds().dO(runthis);

   while (true)
    {
      schedule.run_pending();
    }
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void runthis()
 {
   Print(__FUNCTION__," called at: ",TimeLocal());
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
```

We want to call the function _runthis_ after every 10 seconds. _An infinite while loop is used just to keep the script running until stopped._

Below is the logged output when the script was run in [debug mode](https://www.metatrader5.com/en/metaeditor/help/development/debug "https://www.metatrader5.com/en/metaeditor/help/development/debug").

```
NO      0       14:51:56.301    schedule test (XAUUSD,D1)       The first function run is schedule at: 2025.07.21 14:52:06
GS      0       14:52:06.000    schedule test (XAUUSD,D1)       runthis called at: 2025.07.21 14:52:06
PJ      0       14:52:06.000    schedule test (XAUUSD,D1)       Prev run: 2025.07.21 14:52:06 Next run: 2025.07.21 14:52:16
QH      0       14:52:16.000    schedule test (XAUUSD,D1)       runthis called at: 2025.07.21 14:52:16
GR      0       14:52:16.000    schedule test (XAUUSD,D1)       Prev run: 2025.07.21 14:52:16 Next run: 2025.07.21 14:52:26
KP      0       14:52:26.000    schedule test (XAUUSD,D1)       runthis called at: 2025.07.21 14:52:26
FJ      0       14:52:26.000    schedule test (XAUUSD,D1)       Prev run: 2025.07.21 14:52:26 Next run: 2025.07.21 14:52:36
```

We can have multiple schedules using the same class.

```
void OnStart()
  {
//---

   schedule.every(10).seconds().dO(runthis); //run after every 10 seconds
   schedule.every().minute().dO(runthis2); //run on every minute

   while (true)
    {
      schedule.run_pending();
    }
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void runthis()
 {
   Print(__FUNCTION__," called at: ",TimeLocal());
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void runthis2()
 {
   Print("Hello world!, This function is called after a minute has passed");
 }
```

Outputs.

```
FK      0       15:00:55.079    schedule test (XAUUSD,D1)       The first function run is schedule at: 2025.07.21 15:01:05
IL      0       15:00:55.079    schedule test (XAUUSD,D1)       The first function run is schedule at: 2025.07.21 15:01:55
ER      0       15:01:05.000    schedule test (XAUUSD,D1)       runthis called at: 2025.07.21 15:01:05
RK      0       15:01:05.000    schedule test (XAUUSD,D1)       Prev run: 2025.07.21 15:01:05 Next run: 2025.07.21 15:01:15
OK      0       15:01:15.000    schedule test (XAUUSD,D1)       runthis called at: 2025.07.21 15:01:15
ES      0       15:01:15.000    schedule test (XAUUSD,D1)       Prev run: 2025.07.21 15:01:15 Next run: 2025.07.21 15:01:25
IS      0       15:01:25.000    schedule test (XAUUSD,D1)       runthis called at: 2025.07.21 15:01:25
LJ      0       15:01:25.000    schedule test (XAUUSD,D1)       Prev run: 2025.07.21 15:01:25 Next run: 2025.07.21 15:01:35
CK      0       15:01:35.000    schedule test (XAUUSD,D1)       runthis called at: 2025.07.21 15:01:35
KR      0       15:01:35.000    schedule test (XAUUSD,D1)       Prev run: 2025.07.21 15:01:35 Next run: 2025.07.21 15:01:45
MP      0       15:01:45.000    schedule test (XAUUSD,D1)       runthis called at: 2025.07.21 15:01:45
FJ      0       15:01:45.000    schedule test (XAUUSD,D1)       Prev run: 2025.07.21 15:01:45 Next run: 2025.07.21 15:01:55
GH      0       15:01:55.000    schedule test (XAUUSD,D1)       runthis called at: 2025.07.21 15:01:55
NM      0       15:01:55.000    schedule test (XAUUSD,D1)       Prev run: 2025.07.21 15:01:55 Next run: 2025.07.21 15:02:05
KR      0       15:01:55.000    schedule test (XAUUSD,D1)       Hello world!, This function is called after a minute has passed
MH      0       15:01:55.000    schedule test (XAUUSD,D1)       Prev run: 2025.07.21 15:01:55 Next run: 2025.07.21 15:02:55
NJ      0       15:02:05.001    schedule test (XAUUSD,D1)       runthis called at: 2025.07.21 15:02:05
HP      0       15:02:05.001    schedule test (XAUUSD,D1)       Prev run: 2025.07.21 15:02:05 Next run: 2025.07.21 15:02:15
GR      0       15:02:15.000    schedule test (XAUUSD,D1)       runthis called at: 2025.07.21 15:02:15
LK      0       15:02:15.000    schedule test (XAUUSD,D1)       Prev run: 2025.07.21 15:02:15 Next run: 2025.07.21 15:02:25
RK      0       15:02:25.001    schedule test (XAUUSD,D1)       runthis called at: 2025.07.21 15:02:25
FS      0       15:02:25.001    schedule test (XAUUSD,D1)       Prev run: 2025.07.21 15:02:25 Next run: 2025.07.21 15:02:35
OS      0       15:02:35.004    schedule test (XAUUSD,D1)       runthis called at: 2025.07.21 15:02:35
JK      0       15:02:35.004    schedule test (XAUUSD,D1)       Prev run: 2025.07.21 15:02:35 Next run: 2025.07.21 15:02:45
EK      0       15:02:45.000    schedule test (XAUUSD,D1)       runthis called at: 2025.07.21 15:02:45
KR      0       15:02:45.000    schedule test (XAUUSD,D1)       Prev run: 2025.07.21 15:02:45 Next run: 2025.07.21 15:02:55
OP      0       15:02:55.000    schedule test (XAUUSD,D1)       runthis called at: 2025.07.21 15:02:55
EJ      0       15:02:55.000    schedule test (XAUUSD,D1)       Prev run: 2025.07.21 15:02:55 Next run: 2025.07.21 15:03:05
FK      0       15:02:55.000    schedule test (XAUUSD,D1)       Hello world!, This function is called after a minute has passed
```

### Running Jobs (Functions) at a Specific Time

We often want to run our functions at a very specific time. For instance, running a function responsible for opening a trade at a certain time according to a trading session e.g. opening a trade at 19:00 in local time.

In the CSchedule class, all functions named _at_ are responsible for doing this when given a specific _"legitimate time"_.

For example.

```
   schedule.every().day().at(19, 10).dO(job);
```

This schedules for the function named _job_ to run every day at the time 19:10 hours.

Implementing this function in MQL5 is tricky as we need separate classes to be returned for every "timely" function, i.e., second, minute, hour, day, week.

```
class CSchedule
  {
private:

   int               m_period; //the number of seconds, minutes, etc to use
   time_intervals_enum m_unit; //time interval: minutes, hours, etc
   int               m_time_seconds; //datetime in seconds
   JobFunction       m_func;   //The function to run for the current schedule

   bool has_fixed_time() const { return m_fixed_time > 0; }
   datetime TodaysDate(datetime dt)
   {
      // Extract year, month, day — and build a new datetime at 00:00:00
      MqlDateTime tm;
      TimeToStruct(dt, tm);
      tm.hour = 0;
      tm.min = 0;
      tm.sec = 0;
      return StructToTime(tm);
   }

public:

                     int  m_fixed_time;  // time from midnight in seconds

                     CSchedule(void);
                    ~CSchedule(void);

                     CSchedule*  every(int period = 1);
                     CSchedule*  seconds();
                     CSchedule*  minutes();
                     CSchedule*  hours();
                     CSchedule*  days();
                     CSchedule*  weeks();

                     MinuteScheduleBuilder* minute();
                     HourScheduleBuilder* hour();
                     DayScheduleBuilder* day();
                     WeekScheduleBuilder* week();
```

Inside every "Builder" class (all classes ending with the word _Builder_), we have a function named _at_ **,** it is responsible for setting a specific time interval.

We also have the function named _dO_ **,** which inherits a namesake function from the _CSchedule_ class.

For example, _the WeekScheduleBuilder class._

```
class CSchedule; //forward declaration | VERY IMPORTANT

class WeekScheduleBuilder
  {
protected:

   CSchedule *m_schedule;

public:

                     WeekScheduleBuilder(CSchedule *schedule_) { m_schedule = schedule_; }
                    ~WeekScheduleBuilder(void) {};

                     CSchedule* at(ENUM_DAY_OF_WEEK dayofweek, uint hour=0, uint minutes = 0, uint seconds = 0)
                        {
                           if (CheckPointer(m_schedule) == POINTER_INVALID || m_schedule == NULL)
                              return NULL;

                           datetime now = TimeLocal();
                           MqlDateTime tm;
                           TimeToStruct(now, tm);

                           int today_dow = tm.day_of_week;

                           //--- Compute days until target day (next week if it's the same day or already passed)

                           int days_ahead = (int)dayofweek - today_dow;
                           if (days_ahead < 0)
                              days_ahead += 7;  // ensure it's next week

                           datetime next_target_date = now + timedelta(days_ahead, DAYS);
                           MqlDateTime target_tm;
                           TimeToStruct(next_target_date, target_tm);

                           //--- setting the correct time

                           target_tm.hour = (int)hour;
                           target_tm.min = (int)minutes;
                           target_tm.sec = (int)seconds;

                           m_schedule.m_fixed_time = (int)StructToTime(target_tm);
                           return m_schedule;
                        }

                     void dO(JobFunction func)
                       {
                          m_schedule.dO(func);
                       }
  };
```

All the functions named _at_  take a specific time passed to their arguments, the time ahead for the first function run, and assign the time value in seconds to a variable named _m\_fixed\_time_.

Inside the function named _dO_, we introduce a condition for checking whether the received time value is a fixed time value (eg,. 19:00) or a scheduled number of seconds, minutes, etc, for the next function run since, _both conditions requires a slightly different handling approach._

```
void CSchedule::dO(JobFunction func)
 {
      m_func = func;

      jobs_struct job;
      job.func = m_func;

      datetime now = TimeLocal();
      job.prev_run = (int)now;
      job.interval = timedelta(m_period, m_unit); //Get configs from the every() method and above

      if (has_fixed_time())
         {
            datetime scheduled_time = (datetime)m_fixed_time; //we add today's date to the fixed_time calculated

            //Add interval repeatedly until scheduled_time >= now
            while (scheduled_time <= now)
               scheduled_time += job.interval; //Schedule for the next time if the current time has passed

            job.next_run = (int)scheduled_time;
         }
      else
        {
          job.next_run = job.prev_run + job.interval;
        }

    if (MQLInfoInteger(MQL_DEBUG))
      Print("The first function run is schedule at: ", TimeToString((datetime)job.next_run, TIME_DATE | TIME_SECONDS));

//---
      jobs_add(m_jobs, job); //store the job object to the list of jobs
 }
```

Below is how you set multiple schedules to run repeatedly at a specified time.

```
void OnStart()
  {
//---

   schedule.every().minute().at(10).dO(job); //Runs on every minute at the 10th second
   schedule.every().hour().at(10).dO(job); //runs on every hour at the 10th minute
   schedule.every().day().at(19, 10).dO(job); //runs every day at 19:10 hours
   schedule.every().week().at(MONDAY).dO(job); //Runs every week on Monday at 00:00 (by default)

   while (true)
    {
      schedule.run_pending();
    }
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void job()
 {
   Print(__FUNCTION__," run at: ",TimeLocal());
 }
```

### Passing Arguments to a Task (Job)

As seen in the previous example output logs, it is hard to identify and track a job's progress, especially when we have multiple scheduled functions running simultaneously. To fix this, we need an optional variable named _jobs\_name_ in the function named _dO_ **,** this variable helps in labelling all scheduled tasks.

```
void CSchedule::dO(JobFunction func, const string jobs_name="")
 {
      jobs_struct job;

      job.func = func; //Assigns the function to job's sturucture
      job.name = jobs_name; //Assigns the name to job's structure

      datetime now = TimeLocal();
      job.prev_run = (int)now;
      job.interval = timedelta(m_period, m_unit); //Get configs from the every() method and above

      if (has_fixed_time())
         {
            datetime today_midnight = TodaysDate(now); //Get todays date at 00:00
            datetime scheduled_time = today_midnight + m_fixed_time; //we add today's date to the fixed_time calculated

            //Add interval repeatedly until scheduled_time >= now
            while (scheduled_time <= now)
               scheduled_time += job.interval; //Schedule for the next time

            job.next_run = (int)scheduled_time;
         }
      else
        {
          job.next_run = job.prev_run + job.interval;
        }

    if (MQLInfoInteger(MQL_DEBUG))
      printf("Job: %s -> first run schedule at: [%s]",job.name, TimeToString((datetime)job.next_run, TIME_DATE | TIME_SECONDS));

//---
      jobs_add(m_jobs, job); //store the job object to the list of jobs
 }
```

Now, we can track each Job's progress more effectively.

```
void OnStart()
  {
//---

   schedule.every().minute().at(10).dO(Greet, "Jacob");
   schedule.every().hour().at(10).dO(Greet, "Anne");
   schedule.every().day().at(08, 10).dO(Greet, "Chriss");
   schedule.every().week().at(MONDAY).dO(Greet, "Nobody");

   while (true)
    {
      schedule.run_pending();
    }
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void Greet()
 {
   Print("Hello there!");
 }
```

Outputs.

```
JI      0       06:57:54.817    schedule test (XAUUSD,D1)       Job: Jacob -> first run schedule at: [2025.07.22 06:58:10]
MH      0       06:57:54.817    schedule test (XAUUSD,D1)       Job: Anne -> first run schedule at: [2025.07.22 07:10:00]
FL      0       06:57:54.817    schedule test (XAUUSD,D1)       Job: Chriss -> first run schedule at: [2025.07.22 08:10:00]
JO      0       06:57:54.817    schedule test (XAUUSD,D1)       Job: Nobody -> first run schedule at: [2025.07.28 00:00:00]
GN      0       06:58:10.014    schedule test (XAUUSD,D1)       Hello there!
LF      0       06:58:10.014    schedule test (XAUUSD,D1)       Job: Jacob -> Prev run: [2025.07.22 06:58:10] Next run: [2025.07.22 06:59:10]
```

### Running a Task Until a Certain Time

Sometimes we have scheduled tasks that we don't to keep running forever. Putting a deadline to these jobs would help in this case.

Let's introduce the function named _until._ _A function similar to the one available in the schedule module offered in Python._

```
CSchedule* CSchedule::until(datetime expiry_date)
 {
   m_expiry_date = expiry_date;

   return GetPointer(this);
 }
```

Inside the function named _dO_, we take the expiry date from our class (received from the function named until) and assign it to the job's structure.

```
void CSchedule::dO(JobFunction func, const string jobs_name="")
 {
      jobs_struct job;

      job.func = func; //Assigns the function to job's sturucture
      job.name = jobs_name; //Assigns the name to job's structure

      datetime now = TimeLocal();
      job.prev_run = (int)now;
      job.interval = timedelta(m_period, m_unit); //Get configs from the every() method and above
      job.expiry_date = m_expiry_date;

      //Other lines of code
 }
```

Before running a job inside the function named _run\_pending_, we have to check if it hasn't expired.

```
void CSchedule::run_pending()
  {
   int now = (int)TimeLocal();
   for(int i = 0; i < ArraySize(m_jobs); i++)
     {
      if (now >= (int)m_jobs[i].expiry_date && expiry_date != 0) //Check if the job hasn't expired
        {
          if (MQLInfoInteger(MQL_DEBUG))
            printf("Job: %s -> Expired",m_jobs[i].name);

          continue; //skip all expired jobs
        }

    //... other checks
  }
```

Finally, we run a task and set the expiry date 5 minutes from the current time.

```
#include <schedule.mqh>
CSchedule schedule;
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
//---

   schedule.every(1).minutes().until(D'22.7.2025 10:15').dO(Greet, "Greet"); //The current time was 10:10, in the same date

   while (true)
    {
      schedule.run_pending();
      Sleep(1000);
    }
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void Greet()
 {
   Print("Hello there!");
 }
```

Outputs.

```
EM      0       10:10:02.849    schedule test (XAUUSD,D1)       Job: Greet -> first run schedule at: [2025.07.22 10:11:02]
CI      0       10:11:02.864    schedule test (XAUUSD,D1)       Hello there!
NS      0       10:11:02.864    schedule test (XAUUSD,D1)       Job: Greet -> Prev run: [2025.07.22 10:11:02] Next run: [2025.07.22 10:12:02]
FR      0       10:12:02.873    schedule test (XAUUSD,D1)       Hello there!
EJ      0       10:12:02.873    schedule test (XAUUSD,D1)       Job: Greet -> Prev run: [2025.07.22 10:12:02] Next run: [2025.07.22 10:13:02]
ND      0       10:13:02.861    schedule test (XAUUSD,D1)       Hello there!
OL      0       10:13:02.861    schedule test (XAUUSD,D1)       Job: Greet -> Prev run: [2025.07.22 10:13:02] Next run: [2025.07.22 10:14:02]
GM      0       10:14:02.922    schedule test (XAUUSD,D1)       Hello there!
PG      0       10:14:02.922    schedule test (XAUUSD,D1)       Job: Greet -> Prev run: [2025.07.22 10:14:02] Next run: [2025.07.22 10:15:02]
LH      0       10:15:00.945    schedule test (XAUUSD,D1)       Job: Greet -> Expired
```

### Running all Jobs Regardless of their Schedules

Sometimes you might need to run all the functions instantly, despite their schedules. Usually, for testing, and sometimes we just want to force all scheduled operations to run at once, for example, during the program startup.

In such situations, the function named _run\_all_ becomes handy.

```
void CSchedule::run_all(void)
  {
   datetime now = TimeLocal();
   for(int i = 0; i < ArraySize(m_jobs); i++)
     {
         if(m_jobs[i].func != NULL)
            m_jobs[i].func();

         m_jobs[i].prev_run = m_jobs[i].next_run;
         // Recalculate next_run
         m_jobs[i].next_run += m_jobs[i].interval;

         if (MQLInfoInteger(MQL_DEBUG))
            printf("%s run at: %s",m_jobs[i].name, TimeToString(now, TIME_DATE|TIME_SECONDS));
     }

    if (MQLInfoInteger(MQL_DEBUG))
      printf("%s -> All %I64u Jobs have been executed!",__FUNCTION__, m_jobs.Size());
  }
```

Example usage.

```
#include <schedule.mqh>
CSchedule schedule;
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
//---

   schedule.every(4).minutes().dO(Greet, "Greet every minute");
   schedule.every(4).hours().dO(Greet, "Greet hourly");
   schedule.every(4).days().dO(Greet, "Greet daily");
   schedule.every(4).weeks().dO(Greet, "Greet weekly");

   schedule.run_all();
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void Greet()
 {
   Print("Hello there!");
 }
```

Outputs.

```
CM      0       11:04:06.695    schedule test (XAUUSD,D1)       Job: Greet every minute -> first run schedule at: [2025.07.22 11:08:06]
RR      0       11:04:06.695    schedule test (XAUUSD,D1)       Job: Greet hourly -> first run schedule at: [2025.07.22 15:04:06]
HR      0       11:04:06.695    schedule test (XAUUSD,D1)       Job: Greet daily -> first run schedule at: [2025.07.26 11:04:06]
MN      0       11:04:06.695    schedule test (XAUUSD,D1)       Job: Greet weekly -> first run schedule at: [2025.08.19 11:04:06]
NI      0       11:04:06.695    schedule test (XAUUSD,D1)       Hello there!
QL      0       11:04:06.695    schedule test (XAUUSD,D1)       Greet every minute run at: 2025.07.22 11:04:06
RN      0       11:04:06.695    schedule test (XAUUSD,D1)       Hello there!
GF      0       11:04:06.695    schedule test (XAUUSD,D1)       Greet hourly run at: 2025.07.22 11:04:06
RL      0       11:04:06.695    schedule test (XAUUSD,D1)       Hello there!
KJ      0       11:04:06.695    schedule test (XAUUSD,D1)       Greet daily run at: 2025.07.22 11:04:06
DR      0       11:04:06.695    schedule test (XAUUSD,D1)       Hello there!
QK      0       11:04:06.695    schedule test (XAUUSD,D1)       Greet weekly run at: 2025.07.22 11:04:06
FL      0       11:04:06.695    schedule test (XAUUSD,D1)       CSchedule::run_all -> All 4 Jobs have been executed!
```

Despite setting these four functions to run after their fourth time frame interval, they were all executed at the same current time.

### Managing Schedules

We need different ways to access and cancel different schedules programmatically, as some schedules might become obsolete as time passes.

**Getting all Jobs**

| Function | Description |
| --- | --- |
| ```<br>void get_jobs(jobs_struct &jobs_struct_array[]) <br>  { <br>    ArrayResize(jobs_struct_array, m_jobs.Size());<br>    for (uint i=0; i<m_jobs.Size(); i++)<br>      jobs_struct_array[i] = m_jobs[i];<br>  }<br>``` | This function provides an argument that returns by reference, an array containing a structure with all the properties for all jobs/tasks. |
| ```<br>uint get_jobs() { return m_jobs.Size(); }<br>``` | It returns the number of jobs scheduled. |

**Cancelling scheduled Jobs**

| Function | Description |
| --- | --- |
| ```<br>bool Cancel(const string jobs_name);<br>``` | This cancels a scheduled job using it's name. |
| ```<br>bool Cancel(const uint jobs_index);<br>``` | It cancels a scheduled job using its index number (from 0 to +infinity), i.e,. If the job was the first to be scheduled, its index number is 0. |
| ```<br>bool Clear() { return ArrayResize(m_jobs, 0)==-1?false:true; } <br>``` | This clears (removes) all scheduled jobs from memory. _No job/task will run after this function is called._ |

Example usage.

```
#include <schedule.mqh>
CSchedule schedule;
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
//---

   schedule.every().minute().at(0).dO(Greet, "EveryMin Greetings"); //Job is set at index 0
   schedule.every().hour().at(20,10).dO(Greet, "Hourly Greetings"); //Job is set at index 1
   schedule.every().day().at(13,20,10).dO(Greet, "Daily Greetings"); //JOb is set at index 2
   schedule.every().week().at(MONDAY, 13, 56).dO(Greet, "Weekly Greetings"); //Job is set at index 3

   schedule.Cancel(0); //Cancel the job at index 0, the first one
   Print("Jobs remaining: ",schedule.get_jobs());

   schedule.Cancel("Hourly Greetings"); //Cancel the job with this name
   Print("Jobs remaining: ",schedule.get_jobs());

   schedule.Clear(); //Clear all schedules
   Print("Jobs remaining: ",schedule.get_jobs());

   while (true)
    {
      schedule.run_pending();
      Sleep(1000);
    }
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void Greet()
 {
   Print("Hello there!");
 }
```

Outputs.

```
LK      0       16:01:02.017    schedule test (XAUUSD,D1)       Job: EveryMin Greetings -> first run schedule at: [2025.07.22 16:02:00]
HH      0       16:01:02.017    schedule test (XAUUSD,D1)       Job: Hourly Greetings -> first run schedule at: [2025.07.22 17:20:10]
EH      0       16:01:02.017    schedule test (XAUUSD,D1)       Job: Daily Greetings -> first run schedule at: [2025.07.23 13:20:10]
PE      0       16:01:02.017    schedule test (XAUUSD,D1)       Job: Weekly Greetings -> first run schedule at: [2025.07.28 13:56:00]
OS      0       16:01:02.017    schedule test (XAUUSD,D1)       CSchedule::Cancel Job at index [0] removed
HS      0       16:01:02.017    schedule test (XAUUSD,D1)       Jobs remaining: 3
RG      0       16:01:02.018    schedule test (XAUUSD,D1)       CSchedule::Cancel Job 'Hourly Greetings' removed
DL      0       16:01:02.018    schedule test (XAUUSD,D1)       Jobs remaining: 2
DD      0       16:01:02.018    schedule test (XAUUSD,D1)       Jobs remaining: 0
```

### Dealing with Timezones

In all previously discussed examples and code implementations in our library, we used the local time. But, this is very limiting considering that we have multiple [time options](https://www.mql5.com/en/docs/dateandtime) that developers can use in MQL5 programming language. For example, you might want to schedule for a trading operation at a certain time according to broker's servers time or UTC time.

Inside the class, _CSchedule_  constructor, we add an optional variable that allows developers to select the kind of time to use for all the scheduling operations.

```
CSchedule::CSchedule(TIME_SOURCE_ENUM time_source=TIME_SOURCE_LOCAL):
 m_time_source(time_source)
 {
   if (MQLInfoInteger(MQL_DEBUG))
     printf("Schedule class initialized using %s, current time -> %s",EnumToString(time_source), (string)GetTime(m_time_source));
 }
```

_Below is "Time source enumeration" and its corresponding function._

```
enum TIME_SOURCE_ENUM
  {
   TIME_SOURCE_LOCAL,        // TimeLocal()
   TIME_SOURCE_CURRENT,      // TimeCurrent()
   TIME_SOURCE_TRADE_SERVER, // TimeTradeServer()
   TIME_SOURCE_GMT           // TimeGMT()
  };

datetime GetTime(TIME_SOURCE_ENUM source)
  {
   switch(source)
     {
      case TIME_SOURCE_LOCAL:
         return TimeLocal();

      case TIME_SOURCE_CURRENT:
         return TimeCurrent();

      case TIME_SOURCE_TRADE_SERVER:
         return TimeTradeServer();

      case TIME_SOURCE_GMT:
         return TimeGMT();

      default:
         return TimeLocal(); // Fallback
     }
  }
```

Example usage.

```
#include <schedule.mqh>
CSchedule schedule(TIME_SOURCE_GMT); //Using GMT
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
//---

   schedule.every().minute().at(0).dO(Greet, "EveryMin Greetings"); //Job is set at index 0
   schedule.every().hour().at(20,10).dO(Greet, "Hourly Greetings"); //Job is set at index 1
   schedule.every().day().at(13,20,10).dO(Greet, "Daily Greetings"); //JOb is set at index 2
   schedule.every().week().at(MONDAY, 13, 56).dO(Greet, "Weekly Greetings"); //Job is set at index 3

   while (true)
    {
      schedule.run_pending();
      Sleep(1000);
    }
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void Greet()
 {
   Print("Hello there!");
 }
```

Outputs.

```
LP      0       16:57:34.961    schedule test (XAUUSD,D1)       Schedule class initialized using TIME_SOURCE_GMT, current time -> 2025.07.22 13:57:34
QL      0       16:57:34.964    schedule test (XAUUSD,D1)       Job: EveryMin Greetings -> first run schedule at: [2025.07.22 13:58:00]
RM      0       16:57:34.964    schedule test (XAUUSD,D1)       Job: Hourly Greetings -> first run schedule at: [2025.07.22 14:20:10]
RE      0       16:57:34.964    schedule test (XAUUSD,D1)       Job: Daily Greetings -> first run schedule at: [2025.07.23 13:20:10]
KK      0       16:57:34.964    schedule test (XAUUSD,D1)       Job: Weekly Greetings -> first run schedule at: [2025.07.28 13:56:00]
HK      0       16:58:00.161    schedule test (XAUUSD,D1)       Hello there!
KO      0       16:58:00.161    schedule test (XAUUSD,D1)       Job: EveryMin Greetings -> Prev run: [2025.07.22 13:58:00] Next run: [2025.07.22 13:59:00]
```

### Applications of the Schedule Module in your Trading Applications

We've seen how you can use this module in simple functions and examples showcasing timely function runs. Below are a few real-world examples on how you can use this library in your trading applications.

**A More Effective NewBar Event Handling**

It is not always easy to write a function that effectively detects the opening of a new bar, since the CSchedule class has different ways of scheduling a task at a very specific time, we can use it in performing certain actions at the opening of a second, minute, hour, day, etc.

Inside the file **Schedule Testing EA.mq5**.

```
#include <Trade\Trade.mqh>
#include <Trade\SymbolInfo.mqh>
#include <Trade\PositionInfo.mqh>

CTrade m_trade;
CSymbolInfo m_symbol;
CPositionInfo m_position;

//---

#include <schedule.mqh>
CSchedule schedule(TIME_SOURCE_CURRENT); //Use the current broker's time
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

input int magic_number = 22072025;
input uint slippage = 100;
input uint stoploss = 500;
input uint takeprofit = 700;
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---

   m_trade.SetExpertMagicNumber(magic_number);
   m_trade.SetTypeFillingBySymbol(Symbol());
   m_trade.SetDeviationInPoints(slippage);

   if (!m_symbol.Name(Symbol()))
      {
         printf("%s -> Failed to select a symbol '%s'. Error = %d", __FUNCTION__,Symbol(),GetLastError());
         return INIT_FAILED;
      }

//--- Schedule

   schedule.every().hour().at(0,0).dO(MainTradingFunction); //every hour when the minute == 0 and second == 0

//---

   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {

  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//---

    schedule.run_pending(); //Constanly monitor all the scheduled tasks

  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool PosExists(ENUM_POSITION_TYPE type)
 {
    for (int i=PositionsTotal()-1; i>=0; i--)
      if (m_position.SelectByIndex(i))
         if (m_position.Symbol()==Symbol() && m_position.Magic() == magic_number && m_position.PositionType()==type)
            return (true);

    return (false);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CloseAllTrades()
 {
   for (int i = PositionsTotal() - 1; i >= 0; i--)
      if (m_position.SelectByIndex(i))
         if (m_position.Magic() == magic_number && m_position.Symbol() == Symbol())
             m_trade.PositionClose(m_position.Ticket(), slippage);
 }
//+------------------------------------------------------------------+
//|      The main function for opening trades and performing other   |
//|      trading related tasks                                       |
//+------------------------------------------------------------------+
void MainTradingFunction()
 {
   printf("New bar detected!");
//---

   if (!m_symbol.RefreshRates())
      return;

    if (!PosExists(POSITION_TYPE_BUY))
      m_trade.Buy(m_symbol.LotsMin(),
                  Symbol(),
                  m_symbol.Ask(),
                  m_symbol.Ask()-stoploss*m_symbol.Point(),
                  m_symbol.Ask()+takeprofit*m_symbol.Point()
                 );

    if (!PosExists(POSITION_TYPE_SELL))
      m_trade.Sell(m_symbol.LotsMin(),
                   Symbol(),
                   m_symbol.Bid(),
                   m_symbol.Bid()+stoploss*m_symbol.Point(),
                   m_symbol.Bid()-takeprofit*m_symbol.Point()
                   );
//---
 }
```

Outputs in the strategy tester.

![](https://c.mql5.com/2/158/433562118190.gif)

We have to explicitly set all inside the [OnInit](https://www.mql5.com/en/docs/event_handlers/oninit) function similarly to [EventSetTimer](https://www.mql5.com/en/docs/eventfunctions/eventsettimer) — using the _dO_ function.

Since the function _schedule\_pending_ is responsible for constantly monitoring the schedules, it should be run inside the OnTick function in Expert Advisors, inside the OnCalculate function in indicators, and under an infinite loop inside an MQL5 script.

**Sending Daily Trading Reports**

By tracking a few minutes or seconds before the market closing (e.g, 5 minutes before 00:00 hours), we can print or send daily trading reports to the users.

```
int OnInit()
  {
//... other lines of code

//--- Schedule

   schedule.every().hour().at(0,0).dO(MainTradingFunction);
   schedule.every().day().at(23, 55).dO(SendDailyTradingReport); //every day 5 minutes before market closing

//---

   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {

  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//---

    schedule.run_pending();

  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void SendDailyTradingReport()
 {
   string sdate = TimeToString (TimeCurrent(), TIME_DATE);
   datetime start = StringToTime(sdate);

   if (!HistorySelect(start, TimeCurrent()))
     {
       printf("%s, line %d failed to obtain closed deals from history error =%d",__FUNCTION__,__LINE__,GetLastError());
       return;
     }

   Comment("");

//---

   double pl = 0.0;

   int trades_count=0;
   string report_body = "";
   for(int i = 0; i < HistoryDealsTotal(); i++)
     {
      if (m_deal.SelectByIndex(i))
        if (m_deal.Entry() == DEAL_ENTRY_OUT && m_deal.Magic() == magic_number)
          {
            pl += m_deal.Profit();
            trades_count++;

            report_body += StringFormat("Trade[%d] -> | ticket: %I64u | type: %s | entry: %.5f | volume: %.3f | commision: %.3f\n",
                                          trades_count,
                                          m_deal.Ticket(),
                                          EnumToString(m_deal.DealType()),
                                          m_deal.Entry(),
                                          m_deal.Volume(),
                                          m_deal.Commission()
                                        );
          }
     }
    string report_header = StringFormat("<<< Daily Trading Report >>> \r\n\r\nAC Balance: %.3f\r\nAC Equity: %.3f\r\nPL: %.3f\r\nTotal Trades: %d \r\n\r\n",
                                          m_account.Balance(),
                                          m_account.Equity(),
                                          pl,
                                          trades_count
                                        );

//--- You might choose to send the reports instead of printing

   Comment(report_header+report_body);
   Print(report_header+report_body);
 }
```

Outputs on the strategy tester.

```
CS      0       11:33:47.902    Schedule testing EA (EURUSD,H1) 2025.03.13 23:55:00   <<< Daily Trading Report >>>
CS      0       11:33:47.902    Schedule testing EA (EURUSD,H1) 2025.03.13 23:55:00
CS      0       11:33:47.902    Schedule testing EA (EURUSD,H1) 2025.03.13 23:55:00   AC Balance: 2983.830
CS      0       11:33:47.902    Schedule testing EA (EURUSD,H1) 2025.03.13 23:55:00   AC Equity: 2983.200
CS      0       11:33:47.902    Schedule testing EA (EURUSD,H1) 2025.03.13 23:55:00   PL: -2.960
CS      0       11:33:47.902    Schedule testing EA (EURUSD,H1) 2025.03.13 23:55:00   Total Trades: 3
CS      0       11:33:47.902    Schedule testing EA (EURUSD,H1) 2025.03.13 23:55:00
CS      0       11:33:47.902    Schedule testing EA (EURUSD,H1) 2025.03.13 23:55:00   Trade[1] -> | ticket: 166 | type: DEAL_TYPE_SELL | entry: 1.00000 | volume: 0.010 | commision: 0.000
CS      0       11:33:47.902    Schedule testing EA (EURUSD,H1) 2025.03.13 23:55:00   Trade[2] -> | ticket: 168 | type: DEAL_TYPE_BUY | entry: 1.00000 | volume: 0.010 | commision: 0.000
CS      0       11:33:47.902    Schedule testing EA (EURUSD,H1) 2025.03.13 23:55:00   Trade[3] -> | ticket: 169 | type: DEAL_TYPE_SELL | entry: 1.00000 | volume: 0.010 | commision: 0.000
CS      0       11:33:47.902    Schedule testing EA (EURUSD,H1) 2025.03.13 23:55:00
```

_The same idea can be extended to generating weekly reports on every Friday, just before the market closing, and in generating monthly reports._

### Schedule Vs OnTimer

While a direct comparison between these two methods would be unfair, as they both work fine in their own ways, it is good to know their difference and when to and when not to use one or the other.

| OnTimer | Schedule |
| --- | --- |
| Since it allows one timer event (one schedule), this built-in functionality is useful when you want to schedule a single task. | It allows multiple tasks and different intervals for each task. This custom library is useful when you have multiple schedules that you want to carry out simultaneously. |
| It is fast and effective | Not fast as the built-in OnTimer; its effectiveness is yet to be explored. |
| It is limited to Expert Advisors and Indicators only. | It works in all MQL5 programs; EAs, Indicators, and Scripts. |
| It works 24/7 (reliable all the time) | It relies on the trading function (OnTick and OnCalculate) functions, which are triggered only when the market is opened.<br>Unless used inside an infinite loop in a script, the monitoring function named _run\_pending_ operates in less than 24 (hours)/5 (days of the week) — depending on the market. |

To fix the reliability issue that arises using this library as described in the last row of the comparison table, you have to run the class CSchedule inside the _OnTimer_ function.

```
#include <schedule.mqh>
CSchedule schedule(TIME_SOURCE_CURRENT); //Use the current broker's time
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//... other functions & lines of code

//--- Schedule

   schedule.every().hour().at(0,0).dO(MainTradingFunction);
   schedule.every().day().at(23, 55).dO(SendDailyTradingReport); //every day 5 minutes before market closing

//--- Ontimer

   EventSetTimer(1); //Run the Ontimer function after every 1 second (pretty much always)

   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
    EventKillTimer(); //Delete the timer
  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//---

    //schedule.run_pending(); //❎

  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void OnTimer(void)
  {
    schedule.run_pending(); //✅
  }
```

### Conclusion

Being able to set specific schedules and run tasks at precise times is crucial for algorithmic traders. Not only do many trading strategies depend on exact times of the day, but there are also numerous activities we often want to automate for repeated execution such as sending daily trading reports, monthly updates, and more.

The implemented class provides an easy way of setting repeated events in your MQL5 programs, similarly to the one offered in the schedule module offered in Python.

_While the OnTimer function is decent, it lacks some of the crucial ways of setting human-friendly schedules. So, feel free to use this library in areas where OnTimer is insufficient._

Peace out.

**Attachments Table**

| Filename | Description & Usage |
| --- | --- |
| Experts\\Schedule testing EA.mq5 | An Expert Advisor (EA) for scheduling trading operations. |
| Include\\schedule.mqh | Contains the CSchedule class useful for scheduling functions to run at specific times and intervals. |
| Scripts\\schedule test.mq5 | A script for testing the CSchedule class. |

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/18913.zip "Download all attachments in the single ZIP archive")

[Attachments.zip](https://www.mql5.com/en/articles/download/18913/attachments.zip "Download Attachments.zip")(7.22 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Python-MetaTrader 5 Strategy Tester (Part 03): MT5-Like Trading Operations — Handling and Managing](https://www.mql5.com/en/articles/20782)
- [Python-MetaTrader 5 Strategy Tester (Part 02): Dealing with Bars, Ticks, and Overloading Built-in Functions in a Simulator](https://www.mql5.com/en/articles/20455)
- [Implementing Practical Modules from Other Languages in MQL5 (Part 06): Python-Like File IO operations in MQL5](https://www.mql5.com/en/articles/20695)
- [Data Science and ML (Part 47): Forecasting the Market Using the DeepAR model in Python](https://www.mql5.com/en/articles/20571)
- [Implementing Practical Modules from Other Languages in MQL5 (Part 05): The Logging module from Python, Log Like a Pro](https://www.mql5.com/en/articles/20458)
- [Implementing Practical Modules from Other Languages in MQL5 (Part 04): time, date, and datetime modules from Python](https://www.mql5.com/en/articles/19035)
- [Python-MetaTrader 5 Strategy Tester (Part 01): Trade Simulator](https://www.mql5.com/en/articles/18971)

**[Go to discussion](https://www.mql5.com/en/forum/491807)**

![Building a Trading System (Part 1): A Quantitative Approach](https://c.mql5.com/2/159/18587-building-a-profitable-trading-logo__1.png)[Building a Trading System (Part 1): A Quantitative Approach](https://www.mql5.com/en/articles/18587)

Many traders evaluate strategies based on short-term performance, often abandoning profitable systems too early. Long-term profitability, however, depends on positive expectancy through optimized win rate and risk-reward ratio, along with disciplined position sizing. These principles can be validated using Monte Carlo simulation in Python with back-tested metrics to assess whether a strategy is robust or likely to fail over time.

![From Novice to Expert: Animated News Headline Using MQL5 (VII) — Post Impact Strategy for News Trading](https://c.mql5.com/2/159/18817-from-novice-to-expert-animated-logo.png)[From Novice to Expert: Animated News Headline Using MQL5 (VII) — Post Impact Strategy for News Trading](https://www.mql5.com/en/articles/18817)

The risk of whipsaw is extremely high during the first minute following a high-impact economic news release. In that brief window, price movements can be erratic and volatile, often triggering both sides of pending orders. Shortly after the release—typically within a minute—the market tends to stabilize, resuming or correcting the prevailing trend with more typical volatility. In this section, we’ll explore an alternative approach to news trading, aiming to assess its effectiveness as a valuable addition to a trader’s toolkit. Continue reading for more insights and details in this discussion.

![MetaTrader tick info access from MQL5 services to Python application using sockets](https://c.mql5.com/2/159/18680-metatrader-tick-info-access-logo.png)[MetaTrader tick info access from MQL5 services to Python application using sockets](https://www.mql5.com/en/articles/18680)

Sometimes everything is not programmable in the MQL5 language. And even if it is possible to convert existing advanced libraries in MQL5, it would be time-consuming. This article tries to show that we can bypass Windows OS dependency by transporting tick information such as bid, ask and time with MetaTrader services to a Python application using sockets.

![Price Action Analysis Toolkit Development (Part 33): Candle Range Theory Tool](https://c.mql5.com/2/159/18911-price-action-analysis-toolkit-logo.png)[Price Action Analysis Toolkit Development (Part 33): Candle Range Theory Tool](https://www.mql5.com/en/articles/18911)

Upgrade your market reading with the Candle-Range Theory suite for MetaTrader 5, a fully MQL5-native solution that converts raw price bars into real-time volatility intelligence. The lightweight CRangePattern library benchmarks each candle’s true range against an adaptive ATR and classifies it the instant it closes; the CRT Indicator then projects those classifications on your chart as crisp, color-coded rectangles and arrows that reveal tightening consolidations, explosive breakouts, and full-range engulfment the moment they occur.

[![](https://www.mql5.com/ff/si/m0dtjf9x3brdz07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fmarket%2Fmt5%2Fexpert%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dtop.experts%26utm_content%3Dbuy.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=widauvjabtsckwovwaperzkotrcrttvb&s=25ef75d39331f608a319410bf27ff02c1bd7986622ecc1eec8968a650f044731&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=pyvzjgnaftniqkfajapinphnovvvejvo&ssn=1769157295180642261&ssn_dr=0&ssn_sr=0&fv_date=1769157295&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F18913&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Implementing%20Practical%20Modules%20from%20Other%20Languages%20in%20MQL5%20(Part%2003)%3A%20Schedule%20Module%20from%20Python%2C%20the%20OnTimer%20Event%20on%20Steroids%20-%20MQL5%20Articles&scr_res=1920x1080&ac=1769157295730170&fz_uniq=5062568305371292846&sv=2552)

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