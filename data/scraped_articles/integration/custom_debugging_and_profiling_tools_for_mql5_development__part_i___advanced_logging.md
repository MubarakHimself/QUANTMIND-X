---
title: Custom Debugging and Profiling Tools for MQL5 Development (Part I): Advanced Logging
url: https://www.mql5.com/en/articles/17933
categories: Integration
relevance_score: 3
scraped_at: 2026-01-23T21:07:18.155101
---

[![](https://www.mql5.com/ff/sh/jup0jccfs9655z9z2/01.png)Learn to create your own robotsRead our book "MQL5 Programming for Traders"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/book%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.algobook%26utm_content=visit.page%26utm_campaign=algobook.promo.04.2024&a=rsxjstxkzbrlgjjrxaglpezpvrjflnvw&s=7224440013c3dbc50ba9cc078cd015fabca36df446b8e75028d6b30234663872&uid=&ref=https://www.mql5.com/en/articles/17933&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5071589501924485843)

MetaTrader 5 / Integration


Here's the plan:

1. [Introduction](https://www.mql5.com/en/articles/17933#para1)
2. [Building a Custom Logging Framework](https://www.mql5.com/en/articles/17933#para2)
3. [Using the Logging Framework](https://www.mql5.com/en/articles/17933#para3)
4. [Benefits of the Custom Logging Framework](https://www.mql5.com/en/articles/17933#para4)
5. [Conclusion](https://www.mql5.com/en/articles/17933#para5)

### Introduction

Anyone who has spent time writing Expert Advisors, indicators, or scripts in MQL5 knows the frustration: a live trade behaves strangely, a complex formula spits out the wrong number, or your EA grinds to a halt just when the market heats up. The usual quick fix—scattering Print() statements, firing up the Strategy Tester, and praying the problem shows itself—breaks down once your codebase grows large.

MQL5 poses debugging hurdles that ordinary programming languages don’t. Trading programs run in real time (so timing matters), handle real money (so mistakes are costly), and must stay lightning-fast even in volatile markets. MetaEditor’s built-ins—a step-through debugger, Print() and Comment() for basic output, and a high-level profiler—are helpful but generic. They simply weren’t crafted for the pinpoint diagnostics your trading algorithms need.

That’s why building your own debugging and profiling toolkit is a game-changer. Tailor-made utilities can deliver the fine-grained insight and custom workflows missing from the standard set, letting you catch bugs sooner, tune performance, and safeguard code quality.

This series will guide you through constructing just such a toolkit. We’ll start with the cornerstone—a versatile logging framework far more powerful than scattershot Print() calls—then layer on advanced debuggers, custom profilers, a unit-testing harness, and static code checkers. By the end, you’ll have a full suite that turns “fire-fighting” into proactive quality control.

Each installment is hands-on: complete, drop-in MQL5 examples, detailed explanations of how they work, and rationale for each design choice. You’ll walk away with tools you can use immediately and the know-how to adapt them to your own projects.

First up: the most basic diagnostic need of all—seeing exactly what your program does, moment by moment. Let’s build that custom logging framework.

### Building a Custom Logging Framework

In this section, we'll develop a flexible, powerful logging framework that goes far beyond the basic Print() function provided by MQL5. Our custom logger will support multiple output formats, severity levels, and contextual information that will make debugging complex trading systems much more effective.

**Why the Usual Print() Falls Short**

Before we roll up our sleeves and build the new system, it helps to see why relying on Print() alone just doesn’t cut it for professional projects:

1. No Severity Hierarchy – every message lands in the same bucket, so critical alerts get buried beneath routine chatter.
2. Sparse Context – Print can’t tell you which function triggered the message or what the application’s state was at the time.
3. One-Track Output – everything funnels into the Experts tab; there’s no built-in path to files or alternate targets.
4. Zero Filtering – you can’t silence verbose debug logs in production without also muting the errors you care about.
5. Unstructured Text – the free-form output is hard for tools to parse automatically.

Our bespoke logging framework tackles each of these pain points and lays a solid groundwork for troubleshooting sophisticated trading code.

**Architecting the Logger**

We’ll build a clean, modular, object-oriented system around three core pieces:

1. LogLevels: an enum that names the severity tiers (DEBUG, INFO, WARN, ERROR, FATAL).
2. ILogHandler: an interface that lets us plug in different sinks, such as FileLogHandler or ConsoleLogHandler.
3. CLogger: a singleton orchestrator that holds the handlers and exposes the logging API.

We’ll unpack each part next.

**Log Levels**

First, we define the severity levels in LogLevels.mqh:

```
enum LogLevel
{
   LOG_LEVEL_DEBUG = 0, // Detailed information for debugging purposes.
   LOG_LEVEL_INFO  = 1, // General information about the system's operation.
   LOG_LEVEL_WARN  = 2, // Warnings about potential issues that are not critical.
   LOG_LEVEL_ERROR = 3, // Errors that affect parts of the system but allow continuity.
   LOG_LEVEL_FATAL = 4, // Serious problems that interrupt the system's execution.
   LOG_LEVEL_OFF   = 5  // Turn off logging.
};
```

These levels allow us to categorize messages by importance and filter them accordingly. For example, during development, you might want to see all messages (including DEBUG), but in production, you might only want to see WARN and above.

**The Handler Interface**

Next, we define an interface for log handlers in ILogHandler.mqh :

```
#property strict

#include "LogLevels.mqh"
#include <Arrays/ArrayObj.mqh> // For managing handlers

//+------------------------------------------------------------------+
//| Interface: ILogHandler                                           |
//| Description: Defines the contract for log handling mechanisms.   |
//|              Each handler is responsible for processing and      |
//|              outputting log messages in a specific way (e.g., to |
//|              console, file, database).                           |
//+------------------------------------------------------------------+
interface ILogHandler
  {
//--- Method to configure the handler with specific settings
   virtual bool      Setup(const string settings="");

//--- Method to process and output a log message
   virtual void      Log(const datetime time, const LogLevel level, const string origin, const string message, const long expert_id=0);

//--- Method to perform any necessary cleanup
   virtual void      Shutdown();
  };
//+------------------------------------------------------------------+
```

This header file, ILogHandler.mqh, defines a crucial component of the logging framework: the ILogHandler interface. An interface in MQL5 acts as a blueprint or contract, specifying a set of methods that any class implementing it must provide. The purpose of ILogHandler is to standardize how different log output mechanisms (like writing to the console or a file) interact with the main logger class.

The ILogHandler interface itself declares three virtual methods that concrete handler classes must implement:

- _virtual bool Setup(const string settings="")_: This method is designed for initializing and configuring the specific log handler. It accepts an optional string argument (settings) which can be used to pass configuration parameters (like file paths, formatting strings, or minimum log levels) to the handler during its setup phase. The method returns true if the setup was successful and false otherwise, allowing the main logger to know if the handler is ready to use.
- _virtual void Log(const datetime time, const LogLevel level, const string origin, const string message, const long expert\_id=0)_: This is the core method responsible for processing and outputting a single log message. It receives all the necessary details about the log event: the timestamp (time), the severity level (level from LogLevels.mqh), the source or origin of the message (origin), the actual log message text (message), and an optional expert advisor ID (expert\_id). Each implementing class will define how to format and where to send this information based on its specific purpose (e.g., Print to console, write to a file).
- _virtual void Shutdown()_: This method is intended for performing cleanup operations when the log handler is no longer needed, typically during the shutdown sequence of the main logger or the application. Implementations might use this method to close open file handles, release allocated resources, or flush any buffered output to ensure all logs are saved before termination.

By defining this standard interface, the logging framework achieves flexibility and extensibility. The main CLogger class can manage a collection of different ILogHandler objects, sending log messages to each one without needing to know the specific details of how each handler works. New output destinations can be added simply by creating new classes that implement the ILogHandler interface.

**Console Log Handler**

This header file provides the  ConsoleLogHandler  class, a concrete implementation of the  ILogHandler  interface. Its specific purpose is to direct formatted log messages to the MetaTrader 5 platform\\'s "Experts" tab, which acts as the console output area during Expert Advisor (EA) or script execution.

```
#property strict

#include "ILogHandler.mqh"
#include "LogLevels.mqh"

//+------------------------------------------------------------------+
//| Class: ConsoleLogHandler                                         |
//| Description: Implements ILogHandler to output log messages to    |
//|              the MetaTrader 5 Experts tab (console).             |
//+------------------------------------------------------------------+
class ConsoleLogHandler : public ILogHandler
  {
private:
   LogLevel          m_min_level;       // Minimum level to log
   string            m_format;          // Log message format string

   //--- Helper to format the log message
   string            FormatMessage(const datetime time, const LogLevel level, const string origin, const string message);
   //--- Helper to get string representation of LogLevel
   string            LogLevelToString(const LogLevel level);

public:
                     ConsoleLogHandler(const LogLevel min_level = LOG_LEVEL_INFO, const string format = "[{time}] {level}: {origin} - {message}");
                    ~ConsoleLogHandler();

   //--- ILogHandler implementation
   virtual bool      Setup(const string settings="") override;
   virtual void      Log(const datetime time, const LogLevel level, const string origin, const string message, const long expert_id=0) override;
   virtual void      Shutdown() override;

   //--- Setters
   void              SetMinLevel(const LogLevel level) { m_min_level = level; }
   void              SetFormat(const string format)    { m_format = format; }
  };
```

The ConsoleLogHandler class inherits publicly from ILogHandler, meaning it promises to provide implementations for the Setup, Log, and Shutdown methods defined in the interface. It contains two private member variables: m\_min\_level of type LogLevel stores the minimum severity level a message must have to be logged by this handler, and m\_format of type string holds the template used to format the output message. It also declares private helper methods FormatMessage and LogLevelToString, and public methods for the interface implementation and setters for its private members.

```
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
ConsoleLogHandler::ConsoleLogHandler(const LogLevel min_level = LOG_LEVEL_INFO, const string format = "[{time}] {level}: {origin} - {message}")
  {
   m_min_level = min_level;
   m_format = format;
   // No specific setup needed for console logging initially
  }
//+------------------------------------------------------------------+
//| Destructor                                                       |
//+------------------------------------------------------------------+
ConsoleLogHandler::~ConsoleLogHandler()
  {
   // No specific cleanup needed
  }
```

The constructor initializes a new ConsoleLogHandler object. It accepts two optional arguments: min\_level (defaulting to LOG\_LEVEL\_INFO) and format (defaulting to a standard template "\[{time}\] {level}: {origin} - {message}"). These arguments are used to set the initial values of the m\_min\_level and m\_format member variables, respectively. This allows users to configure the handler\\'s filtering level and output appearance upon creation.

The destructor is responsible for cleaning up resources when a ConsoleLogHandler object is destroyed. In this specific implementation, there are no dynamically allocated resources or open handles managed directly by this class, so the destructor body is empty, indicating no special cleanup actions are required for this handler.

```
//+------------------------------------------------------------------+
//| Setup                                                            |
//+------------------------------------------------------------------+
bool ConsoleLogHandler::Setup(const string settings="")
  {
   // Settings could be used to parse format or min_level, but we use constructor args for now
   // Example: Parse settings string if needed
   return true;
  }
//+------------------------------------------------------------------+
//| Log                                                              |
//+------------------------------------------------------------------+
void ConsoleLogHandler::Log(const datetime time, const LogLevel level, const string origin, const string message, const long expert_id=0)
  {
   // Check if the message level meets the minimum requirement
   if(level >= m_min_level && level < LOG_LEVEL_OFF)
     {
      // Format and print the message to the Experts tab
      Print(FormatMessage(time, level, origin, message));
     }
  }
//+------------------------------------------------------------------+
//| Shutdown                                                         |
//+------------------------------------------------------------------+
void ConsoleLogHandler::Shutdown()
  {
   // No specific shutdown actions needed for console logging
   PrintFormat("%s: ConsoleLogHandler shutdown.", __FUNCTION__);
  }
```

1. Setup Method( _ConsoleLogHandler::Setup_):This method implements the Setup function required by the ILogHandler interface. While designed for configuration, the current implementation doesn\\'t utilize the settings string parameter, as the primary configuration (minimum level and format) is handled via the constructor. It simply returns true, indicating that the handler is always considered ready for use after construction.

2. Log Method ( _ConsoleLogHandler::Log_):

This is the core implementation of the logging action for the console. When called by the main CLogger, it first checks if the provided level of the incoming message is greater than or equal to the handler\\'s configured m\_min\_level and also less than LOG\_LEVEL\_OFF. If the message passes this filter, the method calls the private FormatMessage helper function to create the final output string based on the m\_format template and the provided log details (time, level, origin, message). Finally, it uses the built-in MQL5 Print function to display the formatted string in the Experts tab.

3. Shutdown Method ( _ConsoleLogHandler::Shutdown_):

This method implements the Shutdown function from the interface. Similar to the destructor, console logging doesn\\'t typically require specific shutdown actions like closing files. This implementation simply prints a message indicating that the console handler is shutting down, providing confirmation during the application\\'s termination sequence.

```
//+------------------------------------------------------------------+
//| FormatMessage                                                    |
//+------------------------------------------------------------------+
string ConsoleLogHandler::FormatMessage(const datetime time, const LogLevel level, const string origin, const string message)
  {
   string formatted_message = m_format;

   // Replace placeholders
   StringReplace(formatted_message, "{time}", TimeToString(time, TIME_DATE | TIME_SECONDS));
   StringReplace(formatted_message, "{level}", LogLevelToString(level));
   StringReplace(formatted_message, "{origin}", origin);
   StringReplace(formatted_message, "{message}", message);

   return formatted_message;
  }
//+------------------------------------------------------------------+
//| LogLevelToString                                                 |
//+------------------------------------------------------------------+
string ConsoleLogHandler::LogLevelToString(const LogLevel level)
  {
   switch(level)
     {
      case LOG_LEVEL_DEBUG: return "DEBUG";
      case LOG_LEVEL_INFO:  return "INFO";
      case LOG_LEVEL_WARN:  return "WARN";
      case LOG_LEVEL_ERROR: return "ERROR";
      case LOG_LEVEL_FATAL: return "FATAL";
      default:              return "UNKNOWN";
     }
  }
//+------------------------------------------------------------------+
```

1. Helper Method ( _FormatMessage_):

This private helper function takes the raw log details (time, level, origin, message) and the handler\\'s format string (m\_format) as input. It replaces placeholders like {time}, {level}, {origin}, and {message} within the format string with the actual corresponding values. It uses TimeToString for formatting the timestamp and calls LogLevelToString to get the string representation of the severity level. The resulting fully formatted string is then returned to the Log method for printing.

2. Helper Method ( _LogLevelToString_):

This private utility function converts a LogLevel enum value into its corresponding string representation (e.g., LOG\_LEVEL\_INFO becomes "INFO"). It uses a switch statement to handle the defined log levels and returns "UNKNOWN" for any unexpected values. This provides human-readable level indicators in the formatted log output.

3. Setter Methods ( _SetMinLevel, SetFormat_): These public methods allow the user to change the handler\\'s configuration after it has been created. SetMinLevel updates the m\_min\_level member variable, changing the filtering threshold for subsequent log messages. SetFormat updates the m\_format member variable, altering the template used for formatting future log messages.

#### File Log Handler

This header file contains the FileLogHandler class, another concrete implementation of the ILogHandler interface. This handler is designed for persistent logging, writing formatted log messages to files. It includes more advanced features compared to the console handler, such as automatic log file rotation based on date and size, and management of the number of log files retained.

```
#property strict

#include "ILogHandler.mqh"
#include "LogLevels.mqh"

//+------------------------------------------------------------------+
//| Class: FileLogHandler                                            |
//| Description: Implements ILogHandler to output log messages to    |
//|              files with rotation capabilities.                   |
//+------------------------------------------------------------------+
class FileLogHandler : public ILogHandler
  {
private:
   LogLevel          m_min_level;       // Minimum level to log
   string            m_format;          // Log message format string
   string            m_file_path;       // Base path for log files
   string            m_file_prefix;     // Prefix for log file names
   int               m_file_handle;     // Current file handle
   datetime          m_current_day;     // Current day for rotation
   int               m_max_size_kb;     // Maximum file size in KB before rotation
   int               m_max_files;       // Maximum number of log files to keep

   //--- Helper to format the log message
   string            FormatMessage(const datetime time, const LogLevel level, const string origin, const string message);
   //--- Helper to get string representation of LogLevel
   string            LogLevelToString(const LogLevel level);
   //--- Helper to create or rotate log file
   bool              EnsureFileOpen();
   //--- Helper to generate file name based on date
   string            GenerateFileName(const datetime time);
   //--- Helper to perform log rotation
   void              RotateLogFiles();
   //--- Helper to check if file size exceeds limit
   bool              IsFileSizeExceeded();
   // Add custom helper function to sort string arrays
   void              SortStringArray(string &arr[]);
   //--- New helper to clean file paths
   string CleanPath(const string path);

public:
   FileLogHandler(const string file_path="MQL5\\Logs",
                  const string file_prefix="EA_Log",
                  const LogLevel min_level=LOG_LEVEL_INFO,
                  const string format="[{time}] {level}: {origin} - {message}",
                  const int max_size_kb=1024,
                  const int max_files=5);
   virtual ~FileLogHandler();
   //--- ILogHandler implementation
   virtual bool      Setup(const string settings="") override;
   virtual void      Log(const datetime time, const LogLevel level, const string origin, const string message, const long expert_id=0) override;
   virtual void      Shutdown() override;

   //--- Setters
   void SetFilePath(const string path)    { m_file_path = CleanPath(path); }
   void              SetMinLevel(const LogLevel level) { m_min_level = level; }
   void              SetFormat(const string format)    { m_format = format; }
   void              SetFilePrefix(const string prefix){ m_file_prefix = prefix; }
   void              SetMaxSizeKB(const int size)      { m_max_size_kb = size; }
   void              SetMaxFiles(const int count)      { m_max_files = count; }
  };
```

The FileLogHandler class inherits from ILogHandler. It maintains several private member variables to manage its state and configuration: m\_min\_level and m\_format (similar to the console handler), m\_file\_path (the directory where logs are stored), m\_file\_prefix (a base name for log files), m\_file\_handle (the handle for the currently open log file), m\_current\_day (used for daily rotation logic), m\_max\_size\_kb (the size limit in kilobytes for a single log file), and m\_max\_files (the maximum number of log files to keep).

It also declares several private helper methods for formatting, file management, and rotation (FormatMessage, LogLevelToString, EnsureFileOpen, GenerateFileName, RotateLogFiles, IsFileSizeExceeded, SortStringArray, CleanPath). Public methods include the constructor, destructor, interface implementations, and setters for configuration.

```
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
FileLogHandler::FileLogHandler(const string file_path,
                               const string file_prefix,
                               const LogLevel min_level,
                               const string format,
                               const int max_size_kb,
                               const int max_files)
  {
   m_min_level = min_level;
   m_format = format;
   m_file_path = CleanPath(file_path);
   m_file_prefix = file_prefix;
   m_file_handle = INVALID_HANDLE;
   m_current_day = 0;
   m_max_size_kb = max_size_kb;
   m_max_files = max_files;

   // Create directory if it doesn't exist
   if(!FolderCreate(m_file_path))
     {
      if(GetLastError() != 0)
         Print("FileLogHandler: Failed to create directory: ", m_file_path, ", error: ", GetLastError());
     }
  }
//+------------------------------------------------------------------+
//| Destructor                                                       |
//+------------------------------------------------------------------+
FileLogHandler::~FileLogHandler()
  {
   Shutdown();
  }
```

1. Constructor ( _FileLogHandler::FileLogHandler_):

The constructor initializes the FileLogHandler. It takes arguments for the file path, prefix, minimum log level, format string, maximum file size, and maximum number of files, setting the corresponding member variables. It uses the CleanPath helper to ensure the file path uses correct directory separators. Crucially, it also attempts to create the specified log directory (m\_file\_path relative to the terminal\\'s data path) using FolderCreate if it doesn\\'t already exist, ensuring the handler has a place to write the files.

2. Destructor ( _FileLogHandler::~FileLogHandler_):

The destructor ensures proper cleanup by calling the Shutdown method. This guarantees that the currently open log file handle is closed when the FileLogHandler object is destroyed, preventing resource leaks.

```
//+------------------------------------------------------------------+
//| Setup                                                            |
//+------------------------------------------------------------------+
bool FileLogHandler::Setup(const string settings)
  {
   // Parse settings if provided
   // Format could be: "path=MQL5/Logs;prefix=MyEA;min_level=INFO;max_size=2048;max_files=10"
   if(settings != "")
     {
      string parts[];
      int count = StringSplit(settings, ';', parts);

      for(int i = 0; i < count; i++)
        {
         string key_value[];
         if(StringSplit(parts[i], '=', key_value) == 2)
           {
            string key = key_value[0];
            StringTrimLeft(key);
            StringTrimRight(key);
            string value = key_value[1];
            StringTrimLeft(value);
            StringTrimRight(value);

            if(key == "path")
               m_file_path = CleanPath(value);
            else if(key == "prefix")
               m_file_prefix = value;
            else if(key == "min_level")
              {
               if(value == "DEBUG")
                  m_min_level = LOG_LEVEL_DEBUG;
               else if(value == "INFO")
                  m_min_level = LOG_LEVEL_INFO;
               else if(value == "WARN")
                  m_min_level = LOG_LEVEL_WARN;
               else if(value == "ERROR")
                  m_min_level = LOG_LEVEL_ERROR;
               else if(value == "FATAL")
                  m_min_level = LOG_LEVEL_FATAL;
              }
            else if(key == "max_size")
               m_max_size_kb = (int)StringToInteger(value);
            else if(key == "max_files")
               m_max_files = (int)StringToInteger(value);
           }
        }
     }

   return true;
  }
//+------------------------------------------------------------------+
//| Log                                                              |
//+------------------------------------------------------------------+
void FileLogHandler::Log(const datetime time, const LogLevel level, const string origin, const string message, const long expert_id=0)
  {
   // Check if the message level meets the minimum requirement
   if(level >= m_min_level && level < LOG_LEVEL_OFF)
     {
      // Ensure file is open and ready for writing
      if(EnsureFileOpen())
        {
         // Format the message
         string formatted_message = FormatMessage(time, level, origin, message);

         // Write to file
         FileWriteString(m_file_handle, formatted_message + "\r\n");

         // Flush to ensure data is written immediately
         FileFlush(m_file_handle);

         // Check if rotation is needed
         if(IsFileSizeExceeded())
           {
            FileClose(m_file_handle);
            m_file_handle = INVALID_HANDLE;
            RotateLogFiles();
            EnsureFileOpen();
           }
        }
     }
  }
//+------------------------------------------------------------------+
//| Shutdown                                                         |
//+------------------------------------------------------------------+
void FileLogHandler::Shutdown()
  {
   if(m_file_handle != INVALID_HANDLE)
     {
      FileClose(m_file_handle);
      m_file_handle = INVALID_HANDLE;
     }
  }
```

1. Setup Method ( _FileLogHandler::Setup_):

This method implements the Setup function from the interface. It provides an alternative way to configure the handler after creation using a single settings string (e.g., "path=MQL5/Logs;prefix=MyEA;max\_size=2048"). It parses this string, splitting it into key-value pairs, and updates the corresponding member variables like m\_file\_path, m\_file\_prefix, m\_min\_level, m\_max\_size\_kb, and m\_max\_files. This allows for configuration loading from external sources if needed. It returns true after parsing.

2. Log Method ( _FileLogHandler::Log_):

This method handles the core file logging logic. It first checks if the message\\'s level meets the m\_min\_level requirement. If it does, it calls EnsureFileOpen to make sure a valid log file is open (handling daily rotation if necessary). If the file is successfully opened, it formats the message using FormatMessage and writes the formatted string followed by a newline (\\r\\n) to the file using FileWriteString. It then calls FileFlush to ensure the data is immediately written to the disk, which is important for capturing logs even if the application crashes. Finally, it checks if the current file size exceeds the m\_max\_size\_kb limit using IsFileSizeExceeded. If the limit is exceeded, it closes the current file, triggers RotateLogFiles to manage old files, and re-opens a new file using EnsureFileOpen.

3. Shutdown Method ( _FileLogHandler::Shutdown_):

This method implements the Shutdown requirement from the interface. Its primary responsibility is to close the currently open log file handle (m\_file\_handle) using FileClose if it\\'s valid (!= INVALID\_HANDLE). This ensures that the file is properly closed and all buffered data is written when the logger is shut down.

```
//+------------------------------------------------------------------+
//| FormatMessage                                                    |
//+------------------------------------------------------------------+
string FileLogHandler::FormatMessage(const datetime time, const LogLevel level, const string origin, const string message)
  {
   string formatted_message = m_format;

   // Replace placeholders
   StringReplace(formatted_message, "{time}", TimeToString(time, TIME_DATE | TIME_SECONDS));
   StringReplace(formatted_message, "{level}", LogLevelToString(level));
   StringReplace(formatted_message, "{origin}", origin);
   StringReplace(formatted_message, "{message}", message);

   return formatted_message;
  }
//+------------------------------------------------------------------+
//| LogLevelToString                                                 |
//+------------------------------------------------------------------+
string FileLogHandler::LogLevelToString(const LogLevel level)
  {
   switch(level)
     {
      case LOG_LEVEL_DEBUG: return "DEBUG";
      case LOG_LEVEL_INFO:  return "INFO";
      case LOG_LEVEL_WARN:  return "WARN";
      case LOG_LEVEL_ERROR: return "ERROR";
      case LOG_LEVEL_FATAL: return "FATAL";
      default:              return "UNKNOWN";
     }
  }
```

Helper Methods ( _FormatMessage, LogLevelToString_):These private helpers function identically to their counterparts in the ConsoleLogHandler, providing message formatting based on the m\_format string and converting LogLevel enums to readable strings.

```
//+------------------------------------------------------------------+
//| EnsureFileOpen                                                   |
//+------------------------------------------------------------------+
bool FileLogHandler::EnsureFileOpen()
  {
   datetime current_time = TimeCurrent();
   MqlDateTime time_struct;
   TimeToStruct(current_time, time_struct);

   // Create a datetime that represents just the current day (time set to 00:00:00)
   MqlDateTime day_struct;
   day_struct.year = time_struct.year;
   day_struct.mon = time_struct.mon;
   day_struct.day = time_struct.day;
   day_struct.hour = 0;
   day_struct.min = 0;
   day_struct.sec = 0;
   datetime current_day = StructToTime(day_struct);

   // Check if we need to open a new file (either first time or new day)
   if(m_file_handle == INVALID_HANDLE || m_current_day != current_day)
     {
      // Close existing file if open
      if(m_file_handle != INVALID_HANDLE)
        {
         FileClose(m_file_handle);
         m_file_handle = INVALID_HANDLE;
        }

      // Update current day
      m_current_day = current_day;

      // Generate new file name
      string file_name = GenerateFileName(current_time);

      // Open file for writing (append if exists)
      m_file_handle = FileOpen(file_name, FILE_WRITE | FILE_READ | FILE_TXT);

      if(m_file_handle == INVALID_HANDLE)
        {
         Print("FileLogHandler: Failed to open log file: ", file_name, ", error: ", GetLastError());
         return false;
        }

      // Move to end of file for appending
      FileSeek(m_file_handle, 0, SEEK_END);
     }

   return true;
  }
//+------------------------------------------------------------------+
//| GenerateFileName                                                 |
//+------------------------------------------------------------------+
string FileLogHandler::GenerateFileName(const datetime time)
  {
   MqlDateTime time_struct;
   TimeToStruct(time, time_struct);

   string date_str = StringFormat("%04d%02d%02d",
                                 time_struct.year,
                                 time_struct.mon,
                                 time_struct.day);

   return m_file_path + "\\" + m_file_prefix + "_" + date_str + ".log";
  }
//+------------------------------------------------------------------+
//| IsFileSizeExceeded                                               |
//+------------------------------------------------------------------+
bool FileLogHandler::IsFileSizeExceeded()
  {
   if(m_file_handle != INVALID_HANDLE)
     {
      // Get current position (file size)
      ulong size = FileSize(m_file_handle);

      // Check if size exceeds limit (convert KB to bytes)
      return (size > (ulong)m_max_size_kb * 1024);
     }

   return false;
  }
```

1. Helper Method ( _EnsureFileOpen_):

This crucial helper method manages the opening and daily rotation of log files. It compares the current date (derived from TimeCurrent()) with the stored m\_current\_day. If the file handle is invalid or the day has changed, it closes any existing handle, updates m\_current\_day, generates a new filename using GenerateFileName (which includes the date), and opens this new file in write/read mode (FILE\_WRITE \| FILE\_READ \| FILE\_TXT). It uses FileSeek to move to the end of the file, ensuring new logs are appended. It returns true if a file is successfully opened or already open, and false on failure.

2. Helper Method ( _GenerateFileName_):

This utility generates the full path for a log file based on the current time. It formats the date part of the time into a YYYYMMDD string and combines it with the configured m\_file\_path, m\_file\_prefix, and the .log extension.

3. Helper Method ( _IsFileSizeExceeded_):

This function checks if the size of the currently open log file (m\_file\_handle) has surpassed the configured m\_max\_size\_kb limit. It retrieves the file size using FileSize and compares it against the limit (converted to bytes). It returns true if the size is exceeded, false otherwise.

```
//+------------------------------------------------------------------+
//| RotateLogFiles                                                   |
//+------------------------------------------------------------------+
void FileLogHandler::RotateLogFiles()
  {
   // Get list of log files
   string terminal_path = TerminalInfoString(TERMINAL_DATA_PATH);
   string full_path = terminal_path + "\\" + m_file_path;
   string file_pattern = m_file_prefix + "_*.log";

   string files[];
   int file_count = 0;

   long search_handle = FileFindFirst(full_path + "\\" + file_pattern, files[file_count]);
   if(search_handle != INVALID_HANDLE)
     {
      file_count++;

      // Find all matching files
      while(FileFindNext(search_handle, files[file_count]))
        {
         file_count++;
         ArrayResize(files, file_count + 1);
        }

      // Close search handle
      FileFindClose(search_handle);
     }

   // Resize array to actual number of found files before sorting
   ArrayResize(files, file_count);
   // Sort the string array using the custom sorter
   SortStringArray(files);

   // Delete oldest files if we have too many
   int files_to_delete = file_count - m_max_files + 1; // +1 for the new file we'll create

   if(files_to_delete > 0)
     {
      for(int i = 0; i < files_to_delete; i++)
        {
         if(!FileDelete(m_file_path + "\\" + files[i]))
            Print("FileLogHandler: Failed to delete old log file: ", files[i], ", error: ", GetLastError());
        }
     }
  }
//+------------------------------------------------------------------+
//| SortStringArray                                                  |
//+------------------------------------------------------------------+
void FileLogHandler::SortStringArray(string &arr[])
  {
   int n = ArraySize(arr);
   for(int i = 0; i < n - 1; i++)
     {
      for(int j = i + 1; j < n; j++)
        {
         if(arr[i] > arr[j])
           {
            string temp = arr[i];
            arr[i] = arr[j];
            arr[j] = temp;
           }
        }
     }
  }
//+------------------------------------------------------------------+
//| New implementation: CleanPath                                    |
//+------------------------------------------------------------------+
string FileLogHandler::CleanPath(const string path)
  {
   string result = path;
   // Replace all "/" with "\\"
   StringReplace(result, "/", "\\");
   return result;
  }
//+------------------------------------------------------------------+
```

1. Helper Method ( _RotateLogFiles_):

This method implements the log file retention policy. It finds all files in the log directory matching the pattern (m\_file\_prefix\_\*.log) using FileFindFirst and FileFindNext. It stores the filenames in a string array, sorts them alphabetically (which usually corresponds to chronological order due to the date format in the filename) using the SortStringArray helper. It then calculates how many files exceed the m\_max\_files limit and deletes the oldest ones (those appearing earliest in the sorted list) using FileDelete.

2. Helper Method ( _SortStringArray_):

This is a simple bubble sort implementation specifically for sorting the array of log filenames obtained in RotateLogFiles. It\\'s used because MQL5\\'s standard library lacks a built-in sort function for string arrays.

3. Helper Method ( _CleanPath_):

This utility ensures that directory paths use the backslash (\\) separator expected by MQL5 file functions, replacing any forward slashes (/) found in the input path string.

4. Setter Methods ( _SetFilePath, SetMinLevel, etc._):

These public methods allow modification of the handler\\'s configuration parameters (path, prefix, level, format, size limits) after its initial creation, providing flexibility.

**CLogger**

This header file defines the CLogger class, which acts as the central orchestrator for the entire logging framework. It is implemented using the Singleton design pattern, ensuring that only one instance of the logger exists throughout the application. This single instance manages all registered log handlers and provides the primary interface for the user\\'s code to submit log messages.

```
#property strict

#include "LogLevels.mqh"
#include "ILogHandler.mqh"

//+------------------------------------------------------------------+
//| Class: CLogger                                                   |
//| Description: Singleton class for managing and dispatching log    |
//|              messages to registered handlers.                    |
//+------------------------------------------------------------------+
class CLogger
  {
private:
   static CLogger   *s_instance;
   ILogHandler*     m_handlers[];
   LogLevel          m_global_min_level;
   long              m_expert_magic;
   string            m_expert_name;

   //--- Private constructor for Singleton
                     CLogger();
                    ~CLogger();

public:
   //--- Get the singleton instance
   static CLogger*   Instance();
   //--- Cleanup the singleton instance
   static void       Release();

   //--- Handler management
   bool              AddHandler(ILogHandler *handler);
   void              ClearHandlers();

   //--- Configuration
   void              SetGlobalMinLevel(const LogLevel level);
   void              SetExpertInfo(const long magic, const string name);

   //--- Logging methods
   void              Log(const LogLevel level, const string origin, const string message);
   void              Debug(const string origin, const string message);
   void              Info(const string origin, const string message);
   void              Warn(const string origin, const string message);
   void              Error(const string origin, const string message);
   void              Fatal(const string origin, const string message);

   //--- Formatted logging methods
   void              LogFormat(const LogLevel level, const string origin, const string formatted_message);
   void              DebugFormat(const string origin, const string formatted_message);
   void              InfoFormat(const string origin, const string formatted_message);
   void              WarnFormat(const string origin, const string formatted_message);
   void              ErrorFormat(const string origin, const string formatted_message);
   void              FatalFormat(const string origin, const string formatted_message);
  };
```

The CLogger class contains several private members. s\_instance is a static pointer to hold the single instance of the class itself. m\_handlers is a dynamic array of ILogHandler pointers, storing references to all the active log handlers (like console or file handlers). m\_global\_min\_level sets a global filtering threshold; messages below this level are ignored even before being sent to individual handlers. m\_expert\_magic and m\_expert\_name store optional information about the Expert Advisor using the logger, which can be included in log messages for better context.

The constructor and destructor are private to enforce the Singleton pattern. Public methods provide access to the instance, handler management, configuration, and various logging functions.

```
//+------------------------------------------------------------------+
//| Static instance initialization                                   |
//+------------------------------------------------------------------+
CLogger *CLogger::s_instance = NULL;
//+------------------------------------------------------------------+
//| Get Singleton Instance                                           |
//+------------------------------------------------------------------+
CLogger* CLogger::Instance()
  {
   if(s_instance == NULL)
     {
      s_instance = new CLogger();
     }
   return s_instance;
  }
//+------------------------------------------------------------------+
//| Release Singleton Instance                                       |
//+------------------------------------------------------------------+
void CLogger::Release()
  {
   if(s_instance != NULL)
     {
      delete s_instance;
      s_instance = NULL;
     }
  }
//+------------------------------------------------------------------+
//| Constructor (Private)                                            |
//+------------------------------------------------------------------+
CLogger::CLogger()
  {
   m_global_min_level = LOG_LEVEL_DEBUG;
   m_expert_magic = 0;
   m_expert_name = "";
  }
//+------------------------------------------------------------------+
//| Destructor                                                       |
//+------------------------------------------------------------------+
CLogger::~CLogger()
  {
   ClearHandlers();
  }
```

1. Singleton Implementation ( _Instance, Release, private Constructor_):

The Singleton pattern is implemented through the static Instance() method, which creates the CLogger object on its first call and returns the same instance on subsequent calls. The constructor (CLogger::CLogger) is private, preventing direct instantiation from outside the class; it initializes default values for the global minimum level and expert info. The static Release() method is provided to explicitly delete the singleton instance and clean up resources, typically called during application shutdown.

2. Destructor ( _CLogger::~CLogger_):

The destructor is called when the singleton instance is deleted via the Release() method. Its primary responsibility is to clean up the managed handlers by calling the ClearHandlers method, ensuring that each handler\\'s Shutdown method is called and the handler objects themselves are deleted.

```
//+------------------------------------------------------------------+
//| AddHandler                                                       |
//+------------------------------------------------------------------+
bool CLogger::AddHandler(ILogHandler *handler)
  {
   if(CheckPointer(handler) == POINTER_INVALID)
     {
      Print("CLogger::AddHandler - Error: Invalid handler pointer.");
      return false;
     }
   int size = ArraySize(m_handlers);
   ArrayResize(m_handlers, size + 1);
   m_handlers[size] = handler;
   return true;
  }
//+------------------------------------------------------------------+
//| ClearHandlers                                                    |
//+------------------------------------------------------------------+
void CLogger::ClearHandlers()
  {
   for(int i = 0; i < ArraySize(m_handlers); i++)
     {
      ILogHandler *handler = m_handlers[i];
      if(CheckPointer(handler) != POINTER_INVALID)
        {
         handler.Shutdown();
         delete handler;
        }
     }
   ArrayResize(m_handlers, 0);
  }
//+------------------------------------------------------------------+
//| SetGlobalMinLevel                                                |
//+------------------------------------------------------------------+
void CLogger::SetGlobalMinLevel(const LogLevel level)
  {
   m_global_min_level = level;
  }
//+------------------------------------------------------------------+
//| SetExpertInfo                                                    |
//+------------------------------------------------------------------+
void CLogger::SetExpertInfo(const long magic, const string name)
  {
   m_expert_magic = magic;
   m_expert_name = name;
  }
```

1. Handler Management ( _AddHandler, ClearHandlers_):

The AddHandler method allows adding a new log handler (any object implementing ILogHandler) to the logger\\'s internal list (m\_handlers). It checks for a valid pointer, resizes the dynamic array, and adds the handler. The ClearHandlers method iterates through the m\_handlers array, calls the Shutdown method on each valid handler, deletes the handler object itself (assuming the logger takes ownership), and finally clears the array. This is crucial for proper resource cleanup.

2. Configuration ( _SetGlobalMinLevel, SetExpertInfo_):

These methods allow customization of the logger\\'s behavior. SetGlobalMinLevel adjusts the global filtering threshold (m\_global\_min\_level), affecting all messages before they reach the handlers. SetExpertInfo allows setting the magic number and name of the EA, which can then be automatically included in log messages by the handlers for better identification, especially when multiple EAs might be logging concurrently.

```
//+------------------------------------------------------------------+
//| Log                                                              |
//+------------------------------------------------------------------+
void CLogger::Log(const LogLevel level, const string origin, const string message)
  {
   // Check global level first
   if(level < m_global_min_level || level >= LOG_LEVEL_OFF)
      return;

   datetime current_time = TimeCurrent();
   string effective_origin = origin;
   if(m_expert_name != "")
      effective_origin = m_expert_name + "::" + origin;

   // Dispatch to all registered handlers
   for(int i = 0; i < ArraySize(m_handlers); i++)
     {
      ILogHandler *handler = m_handlers[i];
      if(CheckPointer(handler) != POINTER_INVALID)
        {
         handler.Log(current_time, level, effective_origin, message, m_expert_magic);
        }
     }
  }
//+------------------------------------------------------------------+
//| Convenience Logging Methods                                      |
//+------------------------------------------------------------------+
void CLogger::Debug(const string origin, const string message) { Log(LOG_LEVEL_DEBUG, origin, message); }
void CLogger::Info(const string origin, const string message)  { Log(LOG_LEVEL_INFO, origin, message); }
void CLogger::Warn(const string origin, const string message)  { Log(LOG_LEVEL_WARN, origin, message); }
void CLogger::Error(const string origin, const string message) { Log(LOG_LEVEL_ERROR, origin, message); }
void CLogger::Fatal(const string origin, const string message) { Log(LOG_LEVEL_FATAL, origin, message); }

//+------------------------------------------------------------------+
//| LogFormat                                                        |
//+------------------------------------------------------------------+
void CLogger::LogFormat(const LogLevel level, const string origin, const string formatted_message)
  {
   // Check global level first
   if(level < m_global_min_level || level >= LOG_LEVEL_OFF)
      return;
   Log(level, origin, formatted_message);
  }
//+------------------------------------------------------------------+
//| Convenience Formatted Logging Methods                            |
//+------------------------------------------------------------------+
void CLogger::DebugFormat(const string origin, const string formatted_message) { LogFormat(LOG_LEVEL_DEBUG, origin, formatted_message); }
void CLogger::InfoFormat(const string origin, const string formatted_message)  { LogFormat(LOG_LEVEL_INFO, origin, formatted_message); }
void CLogger::WarnFormat(const string origin, const string formatted_message)  { LogFormat(LOG_LEVEL_WARN, origin, formatted_message); }
void CLogger::ErrorFormat(const string origin, const string formatted_message) { LogFormat(LOG_LEVEL_ERROR, origin, formatted_message); }
void CLogger::FatalFormat(const string origin, const string formatted_message) { LogFormat(LOG_LEVEL_FATAL, origin, formatted_message); }
//+------------------------------------------------------------------+
```

1. Core Logging Method ( _Log_):

This is the central method that receives log requests. It first checks if the message\\'s level meets the m\_global\_min\_level. If it passes, it retrieves the current time and constructs an effective\_origin string, potentially prepending the configured m\_expert\_name. It then iterates through the m\_handlers array and calls the Log method of each valid handler, passing along the timestamp, level, origin, message, and expert magic number. This effectively dispatches the log message to all active output destinations.

2. Convenience Logging Methods ( _Debug, Info, Warn, Error, Fatal_):

These public methods provide a simpler interface for logging messages at specific severity levels. Each method (e.g., Debug, Info) simply calls the main Log method with the corresponding LogLevel enum value (LOG\_LEVEL\_DEBUG, LOG\_LEVEL\_INFO, etc.), reducing the amount of code needed in the user\\'s application to log a message.

3. Formatted Logging Methods ( _LogFormat, DebugFormat, etc._):

These methods offer an alternative way to log messages that are already formatted. LogFormat takes a pre-formatted message string and calls the main Log method. The convenience methods like DebugFormat, InfoFormat, etc., simply call LogFormat with the appropriate severity level. These are useful if the message formatting logic is complex and handled elsewhere before calling the logger.

With the CLogger implementation complete, it’s time to see it in action.

### Using the Logging Framework

This EA serves as a practical demonstration of how to integrate and utilize the custom MQL5 logging framework (comprising CLogger, ILogHandler, ConsoleLogHandler, and FileLogHandler). It showcases the setup, configuration, usage during operation, and cleanup of the logging components within a standard EA structure.

The initial section of LoggingExampleEA.mq5 sets up standard Expert Advisor properties and includes the necessary components from the custom logging framework.

Following the properties, the #include statements are crucial for integrating the logging functionality. CLogger.mqh  brings in the main logger class definition. ConsoleLogHandler.mqh  includes the class for logging to the MetaTrader console (Experts tab). FileLogHandler.mqh  includes the class responsible for logging to files. These includes make the classes and functions defined within those header files available for use within this EA.

**Input Parameters (input):**

```
// Input parameters
input int      MagicNumber = 654321;         // EA Magic Number
input double   LotSize     = 0.01;           // Fixed lot size
input int      StopLossPips = 50;            // Stop Loss in pips
input int      TakeProfitPips = 100;         // Take Profit in pips
input LogLevel ConsoleLogLevel = LOG_LEVEL_INFO; // Minimum level for console output
input LogLevel FileLogLevel = LOG_LEVEL_DEBUG;   // Minimum level for file output
```

This section defines the external parameters that users can configure when attaching the Expert Advisor to a chart. These inputs allow customization of the EA's trading behavior and, importantly, its logging settings.

- input int MagicNumber = 654321; : This is a standard EA parameter used to identify orders placed by this specific instance of the EA. It helps distinguish its trades from those of other EAs or manual trades.
- input double LotSize = 0.01; : Defines the fixed trading volume (lot size) to be used for orders placed by the EA.
- input int StopLossPips = 50; : Sets the stop loss distance in pips for orders.
- input int TakeProfitPips = 100; : Sets the take profit distance in pips for orders.

Crucially, the following inputs directly control the behavior of the custom logging framework:

- input LogLevel ConsoleLogLevel = LOG\_LEVEL\_INFO; : This parameter allows the user to select the minimum severity level for messages that should be displayed in the MetaTrader Experts tab (console). It uses the  LogLevel  enumeration type defined in  LogLevels.mqh. By default, it's set to  LOG\_LEVEL\_INFO, meaning INFO, WARN, ERROR, and FATAL messages will be shown on the console, while DEBUG messages will be suppressed.
- input LogLevel FileLogLevel = LOG\_LEVEL\_DEBUG; : Similarly, this input sets the minimum severity level for messages written to the log file. It also uses the  LogLevel  enumeration. The default is  LOG\_LEVEL\_DEBUG, indicating that all messages, including detailed debug information, will be saved to the log file. This allows for less verbose console output during normal operation while retaining detailed logs for later analysis or troubleshooting.

These logging-specific inputs demonstrate how the framework can be easily configured externally, allowing users to adjust logging verbosity without modifying the EA's code.

```
// Global logger pointer (optional, can use CLogger::Instance() directly)
CLogger *g_logger = NULL;
```

The EA declares a single global variable:  CLogger \*g\_logger = NULL; : This line declares a pointer named  g\_logger  that can point to an object of the  CLogger class. It is initialized to NULL, meaning it doesn't point to any valid object initially. This global pointer is intended to hold the single instance of the CLogger obtained via the singleton pattern (CLogger::Instance()).

While using the static CLogger::Instance() method directly wherever logging is needed is possible, storing the instance in this global variable after retrieving it in OnInit() provides a convenient way to access the logger object from different functions (OnTick, OnDeinit, OnChartEvent) without repeatedly calling CLogger::Instance(). It acts as a cached pointer to the singleton logger.

**OnInit():**

```
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//--- Get the logger instance
   g_logger = CLogger::Instance();
   if(CheckPointer(g_logger) == POINTER_INVALID)
     {
      Print("Critical Error: Failed to get Logger instance!");
      return(INIT_FAILED);
     }

//--- Set EA information for context in logs
   g_logger.SetExpertInfo(MagicNumber, MQL5InfoString(MQL5_PROGRAM_NAME));

//--- Configure Handlers ---
   // 1. Console Handler
   ConsoleLogHandler *console_handler = new ConsoleLogHandler(ConsoleLogLevel);
   if(CheckPointer(console_handler) != POINTER_INVALID)
     {
      // Optionally customize format
      // console_handler.SetFormat("[{level}] {message}");
      if(!g_logger.AddHandler(console_handler))
        {
         Print("Warning: Failed to add ConsoleLogHandler.");
         delete console_handler; // Clean up if not added
        }
     }
   else
     {
      Print("Warning: Failed to create ConsoleLogHandler.");
     }

   // 2. File Handler
   string log_prefix = MQL5InfoString(MQL5_PROGRAM_NAME) + "_" + IntegerToString(MagicNumber);
   FileLogHandler *file_handler = new FileLogHandler("MQL5/Logs/EA_Logs", // Directory relative to MQL5/Files
                                                   log_prefix,          // File name prefix
                                                   FileLogLevel,        // Minimum level to log to file
                                                   "[{time}] {level} ({origin}): {message}", // Format
                                                   2048, // Max file size in KB (e.g., 2MB)
                                                   10);  // Max number of log files to keep
   if(CheckPointer(file_handler) != POINTER_INVALID)
     {
      if(!g_logger.AddHandler(file_handler))
        {
         Print("Warning: Failed to add FileLogHandler.");
         delete file_handler; // Clean up if not added
        }
     }
   else
     {
      Print("Warning: Failed to create FileLogHandler.");
     }

//--- Log initialization message
   g_logger.Info(__FUNCTION__, "Expert Advisor initialized successfully.");
   g_logger.Debug(__FUNCTION__, StringFormat("Settings: Lots=%.2f, SL=%d, TP=%d, ConsoleLevel=%s, FileLevel=%s",
                                           LotSize, StopLossPips, TakeProfitPips,
                                           EnumToString(ConsoleLogLevel),
                                           EnumToString(FileLogLevel)));

//--- succeed
   return(INIT_SUCCEEDED);
  }
```

In this example, OnInit() is crucial for setting up and configuring the custom logging framework. The first step within OnInit is retrieving the singleton instance of the logger:

```
g_logger = CLogger::Instance();
```

This static method ensures that only one CLogger object exists. The returned pointer is stored in the global variable g\_logger for easier access later. Basic error checking follows using CheckPointer to ensure the instance was successfully obtained; if not, a critical error is printed to the standard log, and initialization fails (INIT\_FAILED).

```
g_logger.SetExpertInfo(MagicNumber, MQL5InfoString(MQL5_PROGRAM_NAME));
```

This line configures the logger with context about the EA using it. It passes the MagicNumber (from input parameters) and the EA's name (retrieved using MQL5InfoString(MQL5\_PROGRAM\_NAME)). This information can be automatically included in log messages by the handlers (depending on their format string), making it easier to identify logs from specific EAs, especially if multiple EAs are running.

A ConsoleLogHandler is created dynamically using \` new\`:

```
ConsoleLogHandler *console_handler = new ConsoleLogHandler(ConsoleLogLevel);
```

It's configured directly in the constructor with the minimum log level specified by the ConsoleLogLevel input parameter. The code includes a commented-out example (console\_handler.SetFormat("\[{level}\] {message}");) showing how the output format could be customized after creation if needed. The handler is then added to the main logger:

```
if(!g_logger.AddHandler(console_handler))
```

If adding the handler fails (returns false), a warning is printed, and the created handler object is deleted using delete to prevent memory leaks. Error checking is also included for the initial creation (new) of the handler.

Similarly, a FileLogHandler is created:

```
   // 2. File Handler
   string log_prefix = MQL5InfoString(MQL5_PROGRAM_NAME) + "_" + IntegerToString(MagicNumber);
   FileLogHandler *file_handler = new FileLogHandler("MQL5/Logs/EA_Logs", // Directory relative to MQL5/Files
                                                   log_prefix,          // File name prefix
                                                   FileLogLevel,        // Minimum level to log to file
                                                   "[{time}] {level} ({origin}): {message}", // Format
                                                   2048, // Max file size in KB (e.g., 2MB)
                                                   10);  // Max number of log files to keep
```

A log file prefix is constructed using the EA name and magic number for unique identification. The FileLogHandler constructor is called with several arguments: the directory path (\\"MQL5/Logs/EA\_Logs\\", relative to the terminal\\'s MQL5/Files directory), the generated prefix, the minimum level from the FileLogLevel input, a custom format string, the maximum file size in KB (2048 KB = 2MB), and the maximum number of log files to retain (10). Like the console handler, it's added to the logger using g\_logger.AddHandler(), with similar error handling and cleanup (delete) if creation or addition fails.

After setting up the handlers, the EA logs messages to confirm initialization:

```
g_logger.Info(__FUNCTION__, \"Expert Advisor initialized successfully.\");
g_logger.Debug(__FUNCTION__, StringFormat(\"Settings: ...\"));
```

An Info level message confirms success. A Debug level message logs the key input parameters using StringFormat. \_\_FUNCTION\_\_ is used as the origin string, automatically providing the name of the current function (OnInit). These messages will be processed by the added handlers based on their configured minimum levels.

Finally, if all initializations are successful, the function returns INIT\_SUCCEEDED, signaling to the terminal that the EA is ready to start processing ticks. If any critical error occurred (like failing to get the logger instance), it returns INIT\_FAILED.

**OnDeinit():**

```
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//--- Log deinitialization
   if(CheckPointer(g_logger) != POINTER_INVALID)
     {
      string reason_str = "Unknown reason";
      switch(reason)
        {
         case REASON_ACCOUNT: reason_str = "Account change"; break;
         case REASON_CHARTCHANGE: reason_str = "Chart symbol or period change"; break;
         case REASON_CHARTCLOSE: reason_str = "Chart closed"; break;
         case REASON_PARAMETERS: reason_str = "Input parameters changed"; break;
         case REASON_RECOMPILE: reason_str = "Recompiled"; break;
         case REASON_REMOVE: reason_str = "EA removed from chart"; break;
         case REASON_TEMPLATE: reason_str = "Template applied"; break;
         case REASON_CLOSE: reason_str = "Terminal closed"; break;
        }
      g_logger.Info(__FUNCTION__, "Expert Advisor shutting down. Reason: " + reason_str + " (" + IntegerToString(reason) + ")");

      // Release the logger instance (this calls Shutdown() on all handlers)
      CLogger::Release();
      g_logger = NULL; // Set pointer to NULL after release
     }
   else
     {
      Print("Logger instance was already invalid during Deinit.");
     }
//--- Print to standard log just in case logger failed
   Print(MQL5InfoString(MQL5_PROGRAM_NAME) + ": Deinitialized. Reason code: " + IntegerToString(reason));
  }
```

In LoggingExampleEA.mq5, OnDeinit focuses on gracefully shutting down the logging framework:

```
if(CheckPointer(g_logger) != POINTER_INVALID)
```

The function first checks if the global logger pointer g\_logger is still valid. This prevents errors if OnDeinit is called after the logger has already been released or if initialization failed.

Inside the if block, the code determines a human-readable string corresponding to the reason code passed to OnDeinit using a switch statement. This provides context about why the EA is stopping. An informational message is then logged using g\_logger.Info(), including the determined reason string and the original reason code.

```
string reason_str = "Unknown reason";
      switch(reason)
        {
         case REASON_ACCOUNT: reason_str = "Account change"; break;
         case REASON_CHARTCHANGE: reason_str = "Chart symbol or period change"; break;
...
...
         case REASON_CLOSE: reason_str = "Terminal closed"; break;
        }
      g_logger.Info(__FUNCTION__, "Expert Advisor shutting down. Reason: " + reason_str + " (" + IntegerToString(reason) + ")");
```

This ensures that the final moments of the EA\\'s operation, including the reason for stopping, are recorded in the logs (both console and file, depending on their configured levels).

This is the most critical step for logger cleanup:

```
CLogger::Release();
```

Calling the static Release() method of the CLogger class triggers the deletion of the singleton logger instance. As part of its destruction process, the CLogger destructor iterates through all added handlers (the console and file handlers in this case), calls their respective Shutdown() methods (which, for the FileLogHandler, involves closing the open log file), and then deletes the handler objects themselves. This ensures all resources are properly released and files are closed correctly.

Nullify Global Pointer:

```
g_logger = NULL;
```

After releasing the instance, the global pointer g\_logger is explicitly set back to NULL. This is good practice to indicate that the pointer no longer points to a valid object.

An else block handles the case where g\_logger was already invalid when OnDeinit was called, printing a message to the standard Experts log. Additionally, a final Print statement outside the logger logic ensures that a deinitialization message is always recorded in the standard log, even if the custom logger failed entirely.

This implementation demonstrates the correct procedure for shutting down the custom logging framework, ensuring that log files are closed properly and resources are released when the EA terminates.

**OnTick():**

```
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//--- Ensure logger is valid
   if(CheckPointer(g_logger) == POINTER_INVALID)
     {
      // Attempt to re-initialize logger if it became invalid unexpectedly
      // This is defensive coding, ideally it shouldn't happen if OnInit succeeded.
      Print("Error: Logger instance invalid in OnTick! Attempting re-init...");
      if(OnInit() != INIT_SUCCEEDED)
        {
         Print("Critical Error: Failed to re-initialize logger in OnTick. Stopping EA.");
         ExpertRemove(); // Stop the EA
         return;
        }
     }

//--- Log tick arrival
   MqlTick latest_tick;
   if(SymbolInfoTick(_Symbol, latest_tick))
     {
      g_logger.Debug(__FUNCTION__, StringFormat("New Tick: Time=%s, Bid=%.5f, Ask=%.5f, Volume=%d",
                                             TimeToString(latest_tick.time, TIME_DATE|TIME_SECONDS),
                                             latest_tick.bid, latest_tick.ask, (int)latest_tick.volume_real));
     }
   else
     {
      g_logger.Warn(__FUNCTION__, "Failed to get latest tick info. Error: " + IntegerToString(GetLastError()));
     }

//--- Example Logic: Check for a simple crossover
   // Note: Use more robust indicator handling in a real EA
   double ma_fast[], ma_slow[];
   int copied_fast = CopyBuffer(iMA(_Symbol, _Period, 10, 0, MODE_SMA, PRICE_CLOSE), 0, 0, 3, ma_fast);
   int copied_slow = CopyBuffer(iMA(_Symbol, _Period, 50, 0, MODE_SMA, PRICE_CLOSE), 0, 0, 3, ma_slow);

   if(copied_fast < 3 || copied_slow < 3)
     {
      g_logger.Warn(__FUNCTION__, "Failed to copy enough indicator data.");
      return; // Not enough data yet
     }

   // ArraySetAsSeries might be needed depending on how you access indices
   // ArraySetAsSeries(ma_fast, true);
   // ArraySetAsSeries(ma_slow, true);

   bool cross_up = ma_fast[1] > ma_slow[1] && ma_fast[2] <= ma_slow[2];
   bool cross_down = ma_fast[1] < ma_slow[1] && ma_fast[2] >= ma_slow[2];

   if(cross_up)
     {
      g_logger.Info(__FUNCTION__, "MA Cross Up detected. Potential Buy Signal.");
      // --- Add trading logic here ---
      // Example: SendBuyOrder();
     }
   else if(cross_down)
     {
      g_logger.Info(__FUNCTION__, "MA Cross Down detected. Potential Sell Signal.");
      // --- Add trading logic here ---
      // Example: SendSellOrder();
     }

   // Log account info periodically
   static datetime last_account_log = 0;
   if(TimeCurrent() - last_account_log >= 3600) // Log every hour
     {
      g_logger.Info(__FUNCTION__, StringFormat("Account Update: Balance=%.2f, Equity=%.2f, Margin=%.2f, FreeMargin=%.2f",
                                            AccountInfoDouble(ACCOUNT_BALANCE),
                                            AccountInfoDouble(ACCOUNT_EQUITY),
                                            AccountInfoDouble(ACCOUNT_MARGIN),
                                            AccountInfoDouble(ACCOUNT_MARGIN_FREE)));
      last_account_log = TimeCurrent();
     }
  }
```

Zooming In...

```
//--- Ensure logger is valid
   if(CheckPointer(g_logger) == POINTER_INVALID)
     {
      // Attempt to re-initialize logger if it became invalid unexpectedly
      // This is defensive coding, ideally it shouldn't happen if OnInit succeeded.
      Print("Error: Logger instance invalid in OnTick! Attempting re-init...");
      if(OnInit() != INIT_SUCCEEDED)
        {
         Print("Critical Error: Failed to re-initialize logger in OnTick. Stopping EA.");
         ExpertRemove(); // Stop the EA
         return;
        }
     }
```

Similar to OnDeinit, the function begins by checking if the g\_logger pointer is valid using CheckPointer. As a defensive measure, if the logger is found to be invalid (which ideally shouldn\\'t happen after a successful OnInit), it attempts to re-initialize the logger by calling OnInit() again. If re-initialization fails, it logs a critical error using the standard Print function and stops the EA using ExpertRemove().

Further, The EA attempts to retrieve the latest tick information using SymbolInfoTick().

```
//--- Log tick arrival
   MqlTick latest_tick;
   if(SymbolInfoTick(_Symbol, latest_tick))
     {
      g_logger.Debug(__FUNCTION__, StringFormat("New Tick: Time=%s, Bid=%.5f, Ask=%.5f, Volume=%d",
                                             TimeToString(latest_tick.time, TIME_DATE|TIME_SECONDS),
                                             latest_tick.bid, latest_tick.ask, (int)latest_tick.volume_real));
     }
   else
     {
      g_logger.Warn(__FUNCTION__, "Failed to get latest tick info. Error: " + IntegerToString(GetLastError()));
     }
```

If successful, it logs a Debug message containing the tick\\'s timestamp, bid price, ask price, and volume, formatted using StringFormat. This provides a detailed trace of incoming price data, useful for debugging. If SymbolInfoTick fails, a Warn message is logged, including the error code obtained via GetLastError().

The code further includes a simple example of checking for a moving average (MA) crossover:

```
//--- Example Logic: Check for a simple crossover
   // Note: Use more robust indicator handling in a real EA
   double ma_fast[], ma_slow[];
   int copied_fast = CopyBuffer(iMA(_Symbol, _Period, 10, 0, MODE_SMA, PRICE_CLOSE), 0, 0, 3, ma_fast);
   int copied_slow = CopyBuffer(iMA(_Symbol, _Period, 50, 0, MODE_SMA, PRICE_CLOSE), 0, 0, 3, ma_slow);

   if(copied_fast < 3 || copied_slow < 3)
     {
      g_logger.Warn(__FUNCTION__, "Failed to copy enough indicator data.");
      return; // Not enough data yet
     }

   // ArraySetAsSeries might be needed depending on how you access indices
   // ArraySetAsSeries(ma_fast, true);
   // ArraySetAsSeries(ma_slow, true);

   bool cross_up = ma_fast[1] > ma_slow[1] && ma_fast[2] <= ma_slow[2];
   bool cross_down = ma_fast[1] < ma_slow[1] && ma_fast[2] >= ma_slow[2];

   if(cross_up)
     {
      g_logger.Info(__FUNCTION__, "MA Cross Up detected. Potential Buy Signal.");
      // --- Add trading logic here ---
      // Example: SendBuyOrder();
     }
   else if(cross_down)
     {
      g_logger.Info(__FUNCTION__, "MA Cross Down detected. Potential Sell Signal.");
      // --- Add trading logic here ---
      // Example: SendSellOrder();
     }
```

It first attempts to copy data from two iMA indicators. If insufficient data is copied, a Warn message is logged, and the function returns. If data is available, it checks for a crossover condition between the fast and slow MAs on the previous two bars. When a crossover (cross\_up or cross\_down) is detected, an Info level message is logged, indicating the potential trading signal. This demonstrates logging significant events within the trading strategy.

Finally we log information periodically, rather than on every tick:

```
   // Log account info periodically
   static datetime last_account_log = 0;
   if(TimeCurrent() - last_account_log >= 3600) // Log every hour
     {
      g_logger.Info(__FUNCTION__, StringFormat("Account Update: Balance=%.2f, Equity=%.2f, Margin=%.2f, FreeMargin=%.2f",
                                            AccountInfoDouble(ACCOUNT_BALANCE),
                                            AccountInfoDouble(ACCOUNT_EQUITY),
                                            AccountInfoDouble(ACCOUNT_MARGIN),
                                            AccountInfoDouble(ACCOUNT_MARGIN_FREE)));
      last_account_log = TimeCurrent();
     }
```

A static variable last\_account\_log keeps track of the last time account information was logged. The code checks if the current time (TimeCurrent()) is at least 3600 seconds (1 hour) greater than the last log time. If it is, an Info message containing current account balance, equity, margin, and free margin is logged, and last\_account\_log is updated. This prevents flooding the logs with repetitive information while still providing regular status updates.

Overall, the OnTick function showcases how to use the logger for different purposes during EA execution: detailed debugging (Debug ticks), warnings for potential issues (Warn on data copy failure), informational messages for significant events (Info on signals), and periodic status updates (Info on account status).

**OnChartEvent():**

The OnChartEvent() function is an MQL5 event handler designed to process various events that occur directly on the chart where the EA is running. These events can include user interactions like keyboard presses or mouse movements, clicks on graphical objects, or custom events generated by the EA or other MQL5 programs.

```
//+------------------------------------------------------------------+
//| ChartEvent function                                              |
//+------------------------------------------------------------------+
void OnChartEvent(const int id,
                  const long &lparam,
                  const double &dparam,
                  const string &sparam)
  {
//--- Ensure logger is valid
   if(CheckPointer(g_logger) == POINTER_INVALID) return;

//--- Log chart events
   string event_name = "Unknown Chart Event";
   switch(id)
     {
      case CHARTEVENT_KEYDOWN: event_name = "KeyDown"; break;
      case CHARTEVENT_MOUSE_MOVE: event_name = "MouseMove"; break;
      // Add other CHARTEVENT cases as needed
      case CHARTEVENT_OBJECT_CLICK: event_name = "ObjectClick"; break;
      case CHARTEVENT_CUSTOM+1: event_name = "CustomEvent_1"; break; // Example custom event
     }

   g_logger.Debug(__FUNCTION__, StringFormat("Chart Event: ID=%d (%s), lparam=%d, dparam=%.5f, sparam='%s'",
                                           id, event_name, lparam, dparam, sparam));
  }
//+------------------------------------------------------------------+
```

As in OnTick and OnDeinit, the function starts by ensuring the global logger pointer g\_logger is valid:

```
if(CheckPointer(g_logger) == POINTER_INVALID) return;
```

If the logger is not valid, the function simply returns, preventing any further processing or potential errors.

The core of the function identifies the type of event and logs its details:

```
//--- Log chart events
   string event_name = "Unknown Chart Event";
   switch(id)
     {
      case CHARTEVENT_KEYDOWN: event_name = "KeyDown"; break;
      case CHARTEVENT_MOUSE_MOVE: event_name = "MouseMove"; break;
      // Add other CHARTEVENT cases as needed
      case CHARTEVENT_OBJECT_CLICK: event_name = "ObjectClick"; break;
      case CHARTEVENT_CUSTOM+1: event_name = "CustomEvent_1"; break; // Example custom event
     }

   g_logger.Debug(__FUNCTION__, StringFormat("Chart Event: ID=%d (%s), lparam=%d, dparam=%.5f, sparam='%s'",
                                           id, event_name, lparam, dparam, sparam));
```

A switch statement takes each incoming event ID and converts it into a human-friendly event\_name, such as CHARTEVENT\_KEYDOWN, CHARTEVENT\_MOUSE\_MOVE, or CHARTEVENT\_OBJECT\_CLICK. It even illustrates how to react to a user-defined signal (CHARTEVENT\_CUSTOM + 1).

Next, we issue a Debug-level message with g\_logger.Debug(). This entry records the event ID, the resolved event name, and the parameter values (lparam, dparam, sparam) formatted through StringFormat. Keeping this information a t De bug level is invaluable during development and testing, letting you trace chart interactions and follow custom event flows throughout your application.

### Benefits of the Custom Logging Framework

Our tailor-made logging system delivers several improvements over the basic Print() function:

- _Severity filtering:_ View only the messages that matter, ranked by priority.
- _Multiple outputs:_ Send logs to the console, files, or other destinations simultaneously.
- _Rich context:_ Timestamps, source, and EA details are added automatically.
- _Flexible formatting:_ Adjust message layouts to suit your reading preferences.
- _File rotation:_ Prevent log files from growing without limit.
- _Centralised control:_ Turn logging on or off globally or for individual handlers.

These capabilities make debugging complex trading systems far more efficient. You can pinpoint issues quickly, observe behaviour over time, and stay focused on the data that truly matters.

### Conclusion

Once this custom logging framework is in place, you can ditch the random Print() statements and step into a world where your code speaks in clear, context-rich, and fully adjustable messages. Critical faults jump out, exhaustive traces sit ready for later review, and log files stay tidy. Even better, the system bends to your habits: swap handlers, reshape formats, or dial verbosity up or down whenever you like. The next article will layer on profiling and unit-testing tools so you can spot performance hiccups and logic slips long before they show up on a live chart. That's what real MQL5 craftsmanship looks like.

And keep in mind, this is only the first leg of the journey. We still have advanced debugging tricks, custom profilers, a rock-solid unit-test harness, and automated code-quality scans on the agenda. By the end of the series, you'll trade reactive bug hunts for a disciplined, proactive quality routine.

Until then, happy trading and happy coding!

**File Overview:**

| File Name | File Description |
| --- | --- |
| LogLevels.mqh | Defines the LogLevel enumeration with DEBUG→OFF severity values used throughout the framework. |
| ILogHandler.mqh | Declares the ILogHandler interface (Setup/Log/Shutdown) that all concrete log-output classes implement. |
| ConsoleLogHandler.mqh | Implements ILogHandler to print formatted log messages to the MetaTrader “Experts” tab with level-based filtering. |
| FileLogHandler.mqh | Implements ILogHandler to write logs to rotating daily/size-limited files, keeping a configurable file history. |
| CLogger.mqh | Singleton logger that stores handlers, applies global severity filtering, and offers convenience log methods. |
| LoggingExampleEA.mq5 | Example Expert Advisor showing how to set up, use, and shut down the custom logging framework in practice. |

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/17933.zip "Download all attachments in the single ZIP archive")

[LogLevels.mqh](https://www.mql5.com/en/articles/download/17933/loglevels.mqh "Download LogLevels.mqh")(0.84 KB)

[ILogHandler.mqh](https://www.mql5.com/en/articles/download/17933/iloghandler.mqh "Download ILogHandler.mqh")(1.08 KB)

[ConsoleLogHandler.mqh](https://www.mql5.com/en/articles/download/17933/consoleloghandler.mqh "Download ConsoleLogHandler.mqh")(5.12 KB)

[FileLogHandler.mqh](https://www.mql5.com/en/articles/download/17933/fileloghandler.mqh "Download FileLogHandler.mqh")(14.38 KB)

[CLogger.mqh](https://www.mql5.com/en/articles/download/17933/clogger.mqh "Download CLogger.mqh")(8.51 KB)

[LoggingExampleEA.mq5](https://www.mql5.com/en/articles/download/17933/loggingexampleea.mq5 "Download LoggingExampleEA.mq5")(9.22 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Building a Custom Market Regime Detection System in MQL5 (Part 2): Expert Advisor](https://www.mql5.com/en/articles/17781)
- [Building a Custom Market Regime Detection System in MQL5 (Part 1): Indicator](https://www.mql5.com/en/articles/17737)
- [Advanced Memory Management and Optimization Techniques in MQL5](https://www.mql5.com/en/articles/17693)
- [Mastering JSON: Create Your Own JSON Reader from Scratch in MQL5](https://www.mql5.com/en/articles/16791)
- [Mastering File Operations in MQL5: From Basic I/O to Building a Custom CSV Reader](https://www.mql5.com/en/articles/16614)
- [Modified Grid-Hedge EA in MQL5 (Part IV): Optimizing Simple Grid Strategy (I)](https://www.mql5.com/en/articles/14518)

**[Go to discussion](https://www.mql5.com/en/forum/486401)**

![Automating Trading Strategies in MQL5 (Part 17): Mastering the Grid-Mart Scalping Strategy with a Dynamic Dashboard](https://c.mql5.com/2/141/18038-automating-trading-strategies-logo.png)[Automating Trading Strategies in MQL5 (Part 17): Mastering the Grid-Mart Scalping Strategy with a Dynamic Dashboard](https://www.mql5.com/en/articles/18038)

In this article, we explore the Grid-Mart Scalping Strategy, automating it in MQL5 with a dynamic dashboard for real-time trading insights. We detail its grid-based Martingale logic and risk management features. We also guide backtesting and deployment for robust performance.

![Neural Networks in Trading: Mask-Attention-Free Approach to Price Movement Forecasting](https://c.mql5.com/2/95/Neural_Networks_in_Trading_A_Maskless_Approach_to_Price_Movement_Forecasting__LOGO_2.png)[Neural Networks in Trading: Mask-Attention-Free Approach to Price Movement Forecasting](https://www.mql5.com/en/articles/15973)

In this article, we will discuss the Mask-Attention-Free Transformer (MAFT) method and its application in the field of trading. Unlike traditional Transformers that require data masking when processing sequences, MAFT optimizes the attention process by eliminating the need for masking, significantly improving computational efficiency.

![Data Science and ML (Part 39): News + Artificial Intelligence, Would You Bet on it?](https://c.mql5.com/2/141/17986-data-science-and-ml-part-39-logo.png)[Data Science and ML (Part 39): News + Artificial Intelligence, Would You Bet on it?](https://www.mql5.com/en/articles/17986)

News drives the financial markets, especially major releases like Non-Farm Payrolls (NFPs). We've all witnessed how a single headline can trigger sharp price movements. In this article, we dive into the powerful intersection of news data and Artificial Intelligence.

![Price Action Analysis Toolkit Development (Part 22): Correlation Dashboard](https://c.mql5.com/2/141/18052-price-action-analysis-toolkit-logo.png)[Price Action Analysis Toolkit Development (Part 22): Correlation Dashboard](https://www.mql5.com/en/articles/18052)

This tool is a Correlation Dashboard that calculates and displays real-time correlation coefficients across multiple currency pairs. By visualizing how pairs move in relation to one another, it adds valuable context to your price-action analysis and helps you anticipate inter-market dynamics. Read on to explore its features and applications.

[![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/01.png)![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/02.png)Powerful analytics for traders of any levelAll the necessary trading reports for beginners and professionals](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/454106&a=muccpajyfystoakuukdobwigjejzmpqn&s=52daad60fa795e635264e6f94898f05493bca3b5124d4cca8eb7e82333c2ef12&uid=&ref=https://www.mql5.com/en/articles/17933&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5071589501924485843)

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