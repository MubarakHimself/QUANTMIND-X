---
title: Finding Errors and Logging
url: https://www.mql5.com/en/articles/150
categories: Integration
relevance_score: 0
scraped_at: 2026-01-24T14:08:37.157760
---

[![](https://www.mql5.com/ff/sh/5z040u47jcv59943z2/6c76c03a8b37e08b8655a1a085770b7a.jpg)\\
MetaTrader 5 for iOS and Android\\
\\
Fully featured platform for any devices and web browsers\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=ddonqpipxfqlnsvzlwuowsuwlejpyjxk&s=9daba65b69f40afc3c35f95b1f84ef5824d68c47f29ce96a6dc5b164a2727baa&uid=&ref=https://www.mql5.com/en/articles/150&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5083362337995364915)

MetaTrader 5 / Examples


### Introduction

Hello, dear readers!

In this article we will consider several ways of finding errors in Expert Advisors/Scripts/Indicators and methods of logging. Also I will offer you a small program to view logs - LogMon.

Finding errors is part and parcel of programming. As you write a new block of code, it is necessary to check if it works correctly and has no logic errors. You can find an error in your program in three different ways:


1. Evaluating the final result

2. Step-by-step debugging

3. Writing logical steps into log


Consider each way.


**1\. Evaluating the Final Result**

With this method, we analyze the result of program's work or part of its code. For example, take a simple code, that contains an obvious mistake made only for clarity:

```
void OnStart()
  {
//---
   int intArray[10];
   for(int i=0;i<9;i++)
     {
      intArray[i]=i;
     }
   Alert(intArray[9]);

  }
```

Compile and run, and the screen will display "0". By analyzing the results we expect number "9", hence we conclude that our program is not working as it should. This method of finding errors is common and can not find the error location. Considering the second way of finding errors, we will use the debugging.


**2\. Step-by-Step Debugging**

This method allows you to exactly find the place, where the program logic has been violated. In MetaEditor put a breakpoint inside the 'for' loop, begin debugging and add watch for the i variable:

![Debugging](https://c.mql5.com/2/2/001_Debug.png)

Next click "Resume debugging" for as long as we consider the whole process of program work. We see that as the "i" variable has the value of "8", we exit the loop, hence we conclude that the error is in the line:

```
for(int i=0;i<9;i++)
```

Namely, when the value of i and number 9 are compared. Fix the line "i<9" " to "i<10" or to "i<=9", check the results. We get the number 9 - exactly what we've expected. Using debugging, we have learned how the program acts at its runtime and were able to fix the problem. The cons of this method:

1. It is not clear where the error occurred, by intuition.

2. You need to add variables to the Watch list and to view them after each step.

3. This method can not detect errors during the execution of finished program, such as EA trading on real or demo account.


Finally, consider the third way of finding errors.


**3\. Writing Logical Steps into Log**

Using this method, we record the significant steps of our program. For example: initialization, making a deal, indicator calculation, etc. Upgrade our script with one line of code. Namely, we will print the _i_ variable value on each iteration:


```
void OnStart()
  {
//---
   int intArray[10];
   for(int i=0;i<9;i++)
     {
      intArray[i]=i;
      Alert(i);
     }
   Alert(intArray[9]);

  }
```

Run and see the log output - numbers "0 1 2 3 4 5 6 7 8 0". Conclude why it can be so and fix the script, just like the last time.

Pros and cons of this method of finding errors:


1. + No need to run program step-by-step, so that saves a lot of time.

2. + Often it is obvious, where the error lies.

3. + You can keep logging while the program runs.

4. + You can save log for later analysis and comparison (For example, when writing to a file. See below.).

5. - Source code grows in size due to added operators, that write data into log.

6. - Increased program runtime (mainly important for optimization).



**Summary:**

The first way of finding errors can't track where the error is actually located. We use it primarily for its speed. The second way - step-by-step debugging allows you to find the exact location of error, but it takes a lot of time. And if you'll slip past the desired block of code, you have to start all over again.

Finally, the third way - recording logical steps into log allows you to quickly analyze the program's work and save the result. While writing events of your Expert Advisors/Indicators/Scripts into the log , you can easily find an error and you **don't have to** look for the right conditions under which an error occurs, and don't have to debug your program for hours and hours. Next, we will consider the ways logging in details and compare them. Also I will offer you the way that is most convenient and fast.

### **When you need to log?**

Here are some reasons for logging:


1. Erroneous behavior of the program.

2. Too long program runtime (optimization).

3. Runtime monitoring (print notifications of opening/closing positions, actions executed, etc.).

4. Learning MQL5, for example - printing arrays.

5. Checking Expert Advisors before the championship, etc.


### Methods of Logging

There are a lot of ways to write messages into log, but some are used throughout, while others are needed in special cases. For example, sending log by email or via ICQ is not always necessary.


Here is the list of the most common methods used in MQL5 programming:

1. Using the [Comment()](https://www.mql5.com/en/docs/common/comment) function

2. Using the [Alert()](https://www.mql5.com/en/docs/common/alert) function

3. Using the [Print()](https://www.mql5.com/en/docs/common/print) function

4. Write log into file using the [FileWrite()](https://www.mql5.com/en/docs/files/filewrite) function


Next I'll give examples of each method with source codes and describe the features of each method. These source codes are rather abstract, so we won't go far away from the essence.

**Using the Comment() Function**

```
void OnStart()
  {
//---
   int intArray[10];
   for(int i=0;i<10;i++)
     {
      intArray[i]=i;
      Comment("Variable i: ",i);
      Sleep(5000);
     }
   Alert(intArray[9]);
  }
```

So, in the upper left corner we see the current value of the "i" variable:

![Comment()](https://c.mql5.com/2/2/002_Comment.png)

Thus we can monitor the current state of running program. Now the pros and cons:


1. + You can see the value immediately.

2. - Restriction of output.

3. - You can't select any particular message.

4. - You don't see its work all over the runtime, only the current state.

5. - Relatively slow method.

6. - Ill-suited for continuous monitoring of work, as you always have to watch the readout.


The Comment() function is useful to display the current state of Expert Advisor. For example, "Open 2 deal" or "buy GBRUSD lot: 0.7".


**Using the Alert() Function**

This function displays messages in a separate window with sound notification. The example of code:


```
void OnStart()
  {
//---
   Alert("Start script");
   int intArray[10];
   for(int i=0;i<10;i++)
     {
      intArray[i]=i;
      Alert("Variable i:", I);
      Sleep(1000);
     }
   Alert(intArray[9]);
   Alert("Stop script");
  }
```


The result of code execution:



![Alert()](https://c.mql5.com/2/2/003_Alert.png)

And now we are in seventh heaven, everything is immediately obvious even with sound. But now the pros and cons:


1. + All messages are recorded consistently.

2. + Sound notification.

3. + Everything is written into the "Terminal\_dir\\MQL5\\Logs\\data.txt" file.

4. - All messages from all Scripts/Expert Advisors/Indicators are written into one log.

5. - Does not work in Strategy Tester.

6. **- When called frequently it may freeze terminal for a long time (for example, when calling on every tick or when printing array in loop).**
7. - Impossible to group messages.

8. - Inconvenient viewing of log file.

9. - Can't save messages into folder different from standard Data Folder.


The sixth point is very critical in real trading, especially when scalping or modifying Stop Loss. There are quite a lot of cons and you can find others, but I think that's enough.


**Using the Print() Function**

This function writes log messages into special window called "Experts". Here is the code:


```
void OnStart()
  {
//---
   Print("Старт скрипта");
   int intArray[10];
   for(int i=0;i<10;i++)
     {
      intArray[i]=i;
      Print("Variable i: ",i);
     }
   Print(intArray[9]);
   Print("Stop script");
  }
```

![Print()](https://c.mql5.com/2/2/004_Print.png)

As you can see, this function is called just like the Alert() function, but now all messages are written without notifications into the "Experts" tab and into the "Terminal\_dir\\MQL5\\Logs\\data.txt" file. Consider the pros and cons of this method:


1. + All messages are recorded consistently.

2. + Everything is written into the "Terminal\_dir\\MQL5\\Logs\\data.txt" file.

3. + Suitable for continuous logging of program's work.

4. - All messages from all Scripts/Expert Advisors/Indicators are written into one log.

5. - Impossible to group messages.

6. - Inconvenient viewing of log file.

7. - Can't save messages into folder different from standard Data Folder.


This method is likely used by most of MQL5 programmers, it is pretty fast and well suited for large number of log records.


**Writing Log into File**

Consider the last way of logging - writing messages to files. This method is much more complicated than all previous ones, but with proper preparation ensures a good speed of writing and a comfortable viewing of log, as well as notifications. Here is the simplest code of writing log into a file:


```
void OnStart()
  {
//--- Open log file
   int fileHandle=FileOpen("log.txt",FILE_WRITE|FILE_TXT|FILE_SHARE_READ|FILE_UNICODE);
   FileWrite(fileHandle,"Start script");
   int intArray[10];
   for(int i=0;i<10;i++)
     {
      intArray[i]=i;
      FileWrite(fileHandle,"Variable i: ",i);
      // Sleep(1000);
     }
   FileWrite(fileHandle,intArray[9]);
   FileWrite(FileHandle,"Stop script");
   FileClose(fileHandle); // close log file
  }
```

Run and browse to the "Terminal\_dir\\MQL5\\Files" folder and open the "log.txt" file in text editor. Here are the contents:


![Log to File](https://c.mql5.com/2/2/005_Log_to_file.png)

As you can see, the output is consequent, no extra messages, just what we have written to the file. Consider the pros and cons:


1. + Fast.

2. + Writes only what we want.

3. + You can write messages from different programs into different files, so that that excludes the intersection of logs.

4. - No notifications of new messages in the log.

5. - Impossible to distinguish a particular message or category of message.

6. - It takes a long time to open the log, you must browse to the folder and open the file.


**Summary:**

All of the methods mentioned above have their drawbacks, but you can amend some of them. The first three methods of logging are not flexible, we almost can not influence their behavior. But the latter method, **Writing Log into File**, is the most flexible, we can decide how and when messages are logged. If you want to display a single number, of course it's easier to use the first three methods. But if you have a complicated program with lots of code, it will be difficult to use it without logging.


### New Approach to Logging

Now I will tell and show you how you can improve logging into a file and give you the handy tool to view logs. This is the application for Windows, that I have written in C++ and called it LogMon.

Let's begin with writing the class, that will do all the logging, namely:

1. Keep the location of file, to which the log and other log settings will be written.

2. Create log files depending on given name and date/time.

3. Convert the passed parameters into line of log.

4. Add time to the log message.

5. Add message color.

6. Add message category.

7. Cache messages and write them once per n-seconds or every n-messages.


Since MQL5 is object-oriented language and doesn't significantly differ from C++ in its speed, we will write a class specifically for MQL5. Let's begin.


### Implementation of Class of Writing Log into File

We will put our class in a separate include file with the mqh extension. Here is the general structure of class.

![CLogger](https://c.mql5.com/2/2/006_CLogger.png)

Now the source code of class with detailed comments:


```
//+------------------------------------------------------------------+
//|                                                      Clogger.mqh |
//|                                                             ProF |
//|                                                          http:// |
//+------------------------------------------------------------------+
#property copyright "ProF"
#property link      "http://"

// Max size of cache (quantity)
#define MAX_CACHE_SIZE   10000
// Max file size in megabytes
#define MAX_FILE_SIZEMB 10
//+------------------------------------------------------------------+
//|   Logger                                                         |
//+------------------------------------------------------------------+
class CLogger
  {
private:
   string            project,file;             // Name of project and log file
   string            logCache[MAX_CACHE_SIZE]; // Cache max size
   int               sizeCache;                // Cache counter
   int               cacheTimeLimit;           // Caching time
   datetime          cacheTime;                // Time of cache last flush into file
   int               handleFile;               // Handle of log file
   string            defCategory;              // Default category
   void              writeLog(string log_msg); // Writing message into log or file, and flushing cache
public:
   void              CLogger(void){cacheTimeLimit=0; cacheTime=0; sizeCache=0;};    // Constructor
   void             ~CLogger(void){};                                               // Destructor
   void              SetSetting(string project,string file_name,
                                string default_category="",int cache_time_limit=0); // Settings
   void              init();                   // Initialization, open file for writing
   void              deinit();                 // Deinitialization, closing file
   void              write(string msg,string category="");                                         // Generating message
   void              write(string msg,string category,color colorOfMsg,string file="",int line=0); // Generating message
   void              write(string msg,string category,uchar red,uchar green,uchar blue,
                           string file="",int line=0);                                             // Generating message
   void              flush(void);              // Flushing cache into file

  };
//+------------------------------------------------------------------+
//|  Settings                                                        |
//+------------------------------------------------------------------+
void CLogger::SetSetting(string project_name,string file_name,
                        string default_category="",int cache_time_limit=0)
  {
   project=project_name;             // Project name
   file=file_name;                   // File name
   cacheTimeLimit=cache_time_limit;  // Caching time
   if(default_category=="")          // Setting default category
     {  defCategory="Comment";   }
     else
     {defCategory = default_category;}
  }
//+------------------------------------------------------------------+
//|  Initialization                                                  |
//+------------------------------------------------------------------+
void CLogger::init(void)
  {
   string path;
   MqlDateTime date;
   int i=0;
   TimeToStruct(TimeCurrent(),date);                            // Get current time
   StringConcatenate(path,"log\\log_",project,"\\log_",file,"_",
                     date.year,date.mon,date.day);              // Generate path and file name
   handleFile=FileOpen(path+".txt",FILE_WRITE|FILE_READ|
                       FILE_UNICODE|FILE_TXT|FILE_SHARE_READ);  // Open or create new file
   while(FileSize(handleFile)>(MAX_FILE_SIZEMB*1000000))        // Check file size
     {
      // Open or create new log file
      i++;
      FileClose(handleFile);
      handleFile=FileOpen(path+"_"+(string)i+".txt",
                          FILE_WRITE|FILE_READ|FILE_UNICODE|FILE_TXT|FILE_SHARE_READ);
     }
   FileSeek(handleFile,0,SEEK_END);                             // Set pointer to the end of file
  }
//+------------------------------------------------------------------+
//|   Deinitialization                                               |
//+------------------------------------------------------------------+
void CLogger::deinit(void)
  {
   FileClose(handleFile); // Close file
  }
//+------------------------------------------------------------------+
//|   Write message into file or cache                               |
//+------------------------------------------------------------------+
void CLogger::writeLog(string log_msg)
  {
   if(cacheTimeLimit!=0)  // Check if cache is enabled
     {
      if((sizeCache<MAX_CACHE_SIZE-1 && TimeCurrent()-cacheTime<cacheTimeLimit)
         || sizeCache==0) // Check if cache time is out or if cache limit is reached
        {
         // Write message into cache
         logCache[sizeCache++]=log_msg;
        }
      else
        {
         // Write message into cache and flush cache into file
         logCache[sizeCache++]=log_msg;
         flush();
        }

     }
   else
     {
      // Cache is disabled, immediately write into file
      FileWrite(handleFile,log_msg);
     }
   if(FileTell(handleFile)>(MAX_FILE_SIZEMB*1000000)) // Check current file size
     {
      // File size exceeds allowed limit, close current file and open new
      deinit();
      init();
     }
  }
//+------------------------------------------------------------------+
//|   Generate message and write into log                            |
//+------------------------------------------------------------------+
void CLogger::write(string msg,string category="")
  {
   string msg_log;
   if(category=="")                // Check if passed category exists
     {   category=defCategory;   } // Set default category

// Generate line and call method of writing message
   StringConcatenate(msg_log,category,":|:",TimeToString(TimeCurrent(),TIME_SECONDS),"    ",msg);
   writeLog(msg_log);
  }
//+------------------------------------------------------------------+
//|    Generate message and write into log                           |
//+------------------------------------------------------------------+
void CLogger::write(string msg,string category,color colorOfMsg,string file="",int line=0)
  {
   string msg_log;
   int red,green,blue;
   red=(colorOfMsg  &Red);           // Select red color from constant
   green=(colorOfMsg  &0x00FF00)>>8; // Select green color from constant
   blue=(colorOfMsg  &Blue)>>16;     // Select blue color from constant
                                     // Check if file or line are passed, generate line and call method of writing message
   if(file!="" && line!=0)
     {
      StringConcatenate(msg_log,category,":|:",red,",",green,",",blue,
                        ":|:",TimeToString(TimeCurrent(),TIME_SECONDS),"    ",
                        "file: ",file,"   line: ",line,"   ",msg);
     }
   else
     {
      StringConcatenate(msg_log,category,":|:",red,",",green,",",blue,
                        ":|:",TimeToString(TimeCurrent(),TIME_SECONDS),"    ",msg);
     }
   writeLog(msg_log);
  }
//+------------------------------------------------------------------+
//|    Generate message and write into log                           |
//+------------------------------------------------------------------+
void CLogger::write(string msg,string category,uchar red,uchar green,uchar blue,string file="",int line=0)
  {
   string msg_log;

// Check if file or line are passed, generate line and call method of writing message
   if(file!="" && line!=0)
     {
      StringConcatenate(msg_log,category,":|:",red,",",green,",",blue,
                        ":|:",TimeToString(TimeCurrent(),TIME_SECONDS),"    ",
                        "file: ",file,"   line: ",line,"   ",msg);
     }
   else
     {
      StringConcatenate(msg_log,category,":|:",red,",",green,",",blue,
                        ":|:",TimeToString(TimeCurrent(),TIME_SECONDS),"    ",msg);
     }
   writeLog(msg_log);
  }
//+------------------------------------------------------------------+
//|    Flush cache into file                                         |
//+------------------------------------------------------------------+
void CLogger::flush(void)
  {
   for(int i=0;i<sizeCache;i++) // In loop write all messages into file
     {
      FileWrite(handleFile,logCache[i]);
     }
   sizeCache=0; // Reset cache counter
   cacheTime=TimeCurrent(); // Set time of reseting cache
  }
//+------------------------------------------------------------------+
```

In MetaEditor create the include file (.mqh), and copy the source code of class and save under the "CLogger.mqh" name. Now let's talk more about each method and how to apply the class.


**Using the CLogger class**

To start recording messages into log using this class, we need to include the class file into Expert Advisor/Indicator/Script:


```
#include <CLogger.mqh>
```

Next, you have to create an object of this class:


```
CLogger logger;
```

We will perform all the actions with the "logger" object. Now we need to adjust settings by calling the "SetSetting()" method. Into this method we need to pass the project name and the file name. There are also two optional parameters - the name of default category and cache lifetime (in seconds) during which cache is stored before it is written into file. If you specify zero, all messages will be written once.


```
SetSetting(string project,             // Project name
           string file_name,           // Log file name
           string default_category="", // Default category
           int cache_time_limit=0      // Cache lifetime in seconds
           );
```

Example of call:


```
logger.SetSetting("MyProject","myLog","Comment",60);
```

As a result, messages will be written into the "Client\_Terminal\_dir\\MQL5\\Files\\log\\log\_MyProject\\log\_myLog\_date.txt" file, default category is "Comment" and cache lifetime is 60 seconds. Then you need to call the init() method to open/create log file. Example of call is simple, as you don't need to pass parameters:


```
logger.init();
```

This method generates the path and the name of log file, opens it and checks if it doesn't exceed maximum size. If the size exceeds previously set constant value, then another file is opened, and 1 is concatenated to its name. Then again, the size is checked until the file with correct size is opened.

Then the pointer is moved to position at the end of file. Now the object is ready to write the log. We have overridden the write method. Thanks to this we can set different structures of messages, example of calling the write method and the result in the file:


```
// Write message with default caegory
logger.write("Test message");
// Write message with "Errors" category
logger.write("Test message", "Errors");
// Write message with "Errors" category, that will be highlighted with red color in LogMon
logger.write("Test message", "Errors",Red);
// Write message with "Errors" category, that will be highlighted with red color in LogMon
// Also message will contain current file name and current line
logger.write("Test message", "Errors",Red,__FILE__,__LINE__);
// Write message with "Errors" category, that will be highlighted with GreenYellow color in LogMon
// But now we specify each color independently as: red, green, blue. 0-black, 255 - white
logger.write("Test message", "Errors",173,255,47);
// Write message with "Errors" category, that will be highlighted with GreenYellow color in LogMon
// But now we specify each color independently as: red, green, blue. 0-black, 255 - white
// Also message will contain current file name and current line
logger.write("Test message", "Errors",173,255,47,__FILE__,__LINE__);
```

The log file will contain the following lines:


```
Comment:|:23:13:12    Test message
Errors:|:23:13:12    Test message
Errors:|:255,0,0:|:23:13:12    Test message
Errors:|:255,0,0:|:23:13:12    file: testLogger.mq5   line: 27   Test message
Errors:|:173,255,47:|:23:13:12    Test message
Errors:|:173,255,47:|:23:13:12    file: testLogger.mq5   line: 29   Test message
```

As you can see, everything is very simple. Call the write() method with required parameters anywhere and the message will be written to the file. At the end of your program you need to insert the call of two methods - flush() and deinit().


```
logger.flush();  // Forcibly flush cache to hard disk
logger.deinit(); // Close the log file
```

Here's a simple example of script that writes numbers in a loop to the log:


```
//+------------------------------------------------------------------+
//|                                                   testLogger.mq5 |
//|                                                             ProF |
//|                                                          http:// |
//+------------------------------------------------------------------+
#property copyright "ProF"
#property link      "http://"
#property version   "1.00"
#include <Сlogger.mqh>
CLogger logger;
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+

void OnStart()
  {
//---
   logger.SetSetting("proj","lfile");      // Settings
   logger.init();                          // Initialization
   logger.write("Start script","system");
   for(int i=0;i<100000;i++)               // Write 100000 messages to the log
     {
      logger.write("log: "+(string)i,"Comment",100,222,100,__FILE__,__LINE__);
     }
   logger.write("Stop script","system");
   logger.flush();                         // Flush buffer
   logger.deinit();                        // Deinitialization
  }
//+------------------------------------------------------------------+
```

Script executed in three seconds and created 2 files:


![Log files](https://c.mql5.com/2/2/007_Log_files.png)

File contents:


![Log file contents](https://c.mql5.com/2/2/008_Log_file_contents.png)

And so all the 100000 messages. As you can see, everything works pretty quickly. You can modify this class, add new features or optimize it.


### Level of Messages Output

As you write a program, you have to display several types of messages:


1. Critical errors (program does not behave properly)


3. Notifications of non-critical errors, trading operations, etc. (the program is experiencing temporary errors, or the program has made important action, that user must be notified of)


5. Debugging information (contents of arrays and variables, and other information that is not needed in real work).


Also it is advisable to be able to adjust what messages you want to print without changing the source code. We will implement this goal as a simple function and won't use classes and methods.

Declare the variable parameter that will store the level of messages output. The greater the number in variable, the more categories of messages will be displayed. If you want to completely disable messages output, assign it with value "-1".

```
input int dLvl=2;
```

Below is the source code of function, that must be declared after creating object of the CLogger class.


```
void debug(string debugMsg,             // Message text
          int lvl        )              // Message level
{
   if (lvl<=dLvl)                       // Compare message level with level of messages output
   {
       if (lvl==0)                      // If message is critical (level = 0)
       {logger.write(debugMsg,"",Red);} // mark it with red color
       else
       {logger.write(debugMsg);}        // Else print it with default color
   }
}
```

Now an example: specify level "0" to the most important messages, and any number (in ascending order from zero) to the most useless ones:


```
debug("Error in Expert Advisor!",0);      // Critical error
debug("Stop-Loss execution",1);      // Notification
int i = 99;
debug("Variable i:"+(string)i,2); // Debugging information, variable contents
```

### Easy Log Viewing Using LogMon

OK, now we have log files containing thousands of lines. But it is rather difficult to find information in them. They are not divided into categories and don't differ from each other. I have tried to solve this problem, namely, wrote a program to view logs generated by the CLogger class. Now I will briefly introduce you to the LogMon program, that is written in C++ using WinAPI. Due to this it is quick and small in size. The program is absolutely free.


To work with program you need to:


1. Copy it to the "Client\_Terminal\_dir\\MQL5\\Files\\" folder and run it - in case of normal mode.

2. Copy it to the "Agents\_dir\\Agent\\MQL5\\Files\\" folder and run it - in case of testing or optimization.


Program main window looks like this:

![LogMon main window](https://c.mql5.com/2/2/009_LogMon_Main_Window.png)

The main window contains the toolbar and the window with the tree view. To expand an item, double click it with left mouse button. Folder in the list - are the projects, located in the "Client\_Terminal\_dir\\MQL\\Files\\log\\" folder. You set the name of project in the CLogger class using the SetSetting() method. The files in folders list - are actually the log files. Messages in log files are divided into categories that you've specified in the write() method. Numbers in parentheses - are the numbers of messages in that category.

Now let's consider the buttons on the toolbar from left to right.

**Button to delete project or log file, as well as to reset the tree view**

When you press this button, the following window appears:


![Delete, Flush](https://c.mql5.com/2/2/010_LogMon_Delete_and_Flush.png)

If you press the "Delete and Flush" button, all threads of scanning files/folders will be stopped, tree view will be reset, and you'll be prompted to delete selected file or project (simply click an element to select it - you don't need to select the check box!). The "Reset" button will stop all threads of scanning files/folders and clear the tree view.

**Button to view the "About" dialog box**

Shows brief information about program and its author.


**Button to show program window always on top**

Places the program window above all other windows.

**Button to activate monitoring of new messages in log files**

This button hides the program window to tray ![System Tray](https://c.mql5.com/2/2/011_System_Tray.png)  and activates monitoring of new messages in log files. To select project/file/category that will be scanned, select check boxes next to the necessary elements.

If you select check box next to the category of messages, the notification will trigger on a new message in this project/file/category. If you select check box next to the file, the notification will trigger on a new message in this file for any category. Finally, if you select check box next to the project, the notification will trigger on a new log files and messages in them.


**Monitoring**

If you have activated monitoring and program window is minimized to tray, then when new message appears in selected elements, the main application window will be maximized with sound notification. To disable notifications, click anywhere in the list with left mouse button. To stop monitoring click the program icon in the tray ![LogMon icon](https://c.mql5.com/2/2/012_LogMon_icon.png). To change the sound of notification to your own, place .wav file with name "alert.wav" in the same folder with program executable file.


**View Log Category**

To view specific category simply double click it. Then you will see the message box:


![LogMon search](https://c.mql5.com/2/2/013_LogMon_Search.png)

In this window you can search messages, pin the window alway on top and toggle auto scrolling. The color of each message is set individually using the write() method of the CLogger class. The background of message will be highlighted with selected color.

When you double-click a message, it will open in a separate window. It will be handy if the message is too long and doesn't fit into the dialog box:


![LogMon message](https://c.mql5.com/2/2/014_LogMon_Message.png)

Now you have a handy tool to view and monitor log files. Hopefully, this program will aid you, as you develop and use MQL5 programs.

### Conclusion

Logging events in your program is very useful, it helps you to identify hidden errors and reveal opportunities to improve your program. In this article I've described methods and programs for the most simple logging into file, logs monitoring and viewing.


Your comments and suggestions will be appreciated!

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/150](https://www.mql5.com/ru/articles/150)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/150.zip "Download all attachments in the single ZIP archive")

[clogger.mqh](https://www.mql5.com/en/articles/download/150/clogger.mqh "Download clogger.mqh")(8.72 KB)

[testlogger.mq5](https://www.mql5.com/en/articles/download/150/testlogger.mq5 "Download testlogger.mq5")(1.29 KB)

[logmon\_source\_en.zip](https://www.mql5.com/en/articles/download/150/logmon_source_en.zip "Download logmon_source_en.zip")(119.02 KB)

[logmonen.zip](https://www.mql5.com/en/articles/download/150/logmonen.zip "Download logmonen.zip")(88.78 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Creating Multi-Colored Indicators in MQL5](https://www.mql5.com/en/articles/135)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/3255)**
(10)


![Brett Luedtke](https://c.mql5.com/avatar/avatar_na2.png)

**[Brett Luedtke](https://www.mql5.com/en/users/lugner)**
\|
27 Feb 2011 at 08:53

The attached .exe is still in Russian language. Can you please re-up it?

P.S. This is really useful! One of the first classes that should be declared in any OOP [project](https://www.mql5.com/en/articles/7863 "Article: Projects allow you to create profitable trading robots! But it is not exactly") is one to handle errors!

![MetaQuotes](https://c.mql5.com/avatar/2010/1/4B5DE8B4-9045.jpg)

**[MetaQuotes](https://www.mql5.com/en/users/metaquotes)**
\|
17 Mar 2011 at 15:48

Thank you for your comment. Now you can find correct English version of this utility in attached files.


![nelson75](https://c.mql5.com/avatar/avatar_na2.png)

**[nelson75](https://www.mql5.com/en/users/nelson75)**
\|
16 Aug 2013 at 23:21

If you forget to use flush, data loss may occur.

A better solution would be the following:

```
private:
   void              flush(void);              // Flushing cache into file
```

```
//+------------------------------------------------------------------+
//|   Deinitialization                                               |
//+------------------------------------------------------------------+
void CLogger::deinit(void)
  {
   flush();  // Flush data
   FileClose(handleFile); // Close file
  }
```

So that the data save is automatically.

ps.: I'm sorry if I offended spelling, the google translator helped me...

![Serhiy Dotsenko](https://c.mql5.com/avatar/avatar_na2.png)

**[Serhiy Dotsenko](https://www.mql5.com/en/users/thejobber)**
\|
4 Jan 2015 at 15:14

Dmitry Alexandrovich, thank you for your work, I have been looking for something like this for a long time and finally found it ))

suggestion to improve logmon.exe, to make a setting that would be able to specify the paths where to look for files with logs, because to have two copies in different folders (for the tester and standard work) somehow amateurish ))

although maybe I'll finish it myself when I get my hands on it ))

if you have anything else useful - post it, your style of programming and presentation of material is very braindead )).

![Sayberix](https://c.mql5.com/avatar/avatar_na2.png)

**[Sayberix](https://www.mql5.com/en/users/sayberix)**
\|
2 Mar 2021 at 12:36

I get an error when debugging on historical data: "MQL5 debugger failed to start debugging 'testlogger.ex5' on history". On real data it works fine.

Can you please tell me what I need to tweak to make it work on history?

![Building a Spectrum Analyzer](https://c.mql5.com/2/0/spectrum_MQL5__1.png)[Building a Spectrum Analyzer](https://www.mql5.com/en/articles/185)

This article is intended to get its readers acquainted with a possible variant of using graphical objects of the MQL5 language. It analyses an indicator, which implements a panel of managing a simple spectrum analyzer using the graphical objects. The article is meant for readers acquianted with basics of MQL5.

![Averaging Price Series for Intermediate Calculations Without Using Additional Buffers](https://c.mql5.com/2/0/averages_MQL5__1.png)[Averaging Price Series for Intermediate Calculations Without Using Additional Buffers](https://www.mql5.com/en/articles/180)

This article is about traditional and unusual algorithms of averaging packed in simplest and single-type classes. They are intended for universal usage in almost all developments of indicators. I hope that the suggested classes will be a good alternative to 'bulky' calls of custom and technical indicators.

![Parallel Calculations in MetaTrader 5](https://c.mql5.com/2/0/parallel.png)[Parallel Calculations in MetaTrader 5](https://www.mql5.com/en/articles/197)

Time has been a great value throughout the history of mankind, and we strive not to waste it unnecessarily. This article will tell you how to accelerate the work of your Expert Advisor if your computer has a multi-core processor. Moreover, the implementation of the proposed method does not require the knowledge of any other languages besides MQL5.

![The Implementation of a Multi-currency Mode in MetaTrader 5](https://c.mql5.com/2/0/Multicurrency_Expert_Advisor.png)[The Implementation of a Multi-currency Mode in MetaTrader 5](https://www.mql5.com/en/articles/234)

For a long time multi-currency analysis and multi-currency trading has been of interest to people. The opportunity to implement a full fledged multi-currency regime became possible only with the public release of MetaTrader 5 and the MQL5 programming language. In this article we propose a way to analyze and process all incoming ticks for several symbols. As an illustration, let's consider a multi-currency RSI indicator of the USDx dollar index.

[![](https://www.mql5.com/ff/sh/zf7a2k61x98jzh89z2/01.png)Speed up your tradingUse our high-speed VPS for MetaTrader 4 and 5Learn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=qtrrsuiwuicrscmckjynyanztbditglq&s=c617dc80d90cfd3783ec1345eec2b419b281f10fec6eac77b3218984ac337259&uid=&ref=https://www.mql5.com/en/articles/150&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5083362337995364915)

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