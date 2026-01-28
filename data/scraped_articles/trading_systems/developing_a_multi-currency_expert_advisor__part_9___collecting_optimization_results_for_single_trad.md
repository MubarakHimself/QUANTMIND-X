---
title: Developing a multi-currency Expert Advisor (Part 9): Collecting optimization results for single trading strategy instances
url: https://www.mql5.com/en/articles/14680
categories: Trading Systems, Integration, Expert Advisors
relevance_score: 15
scraped_at: 2026-01-22T17:09:53.575822
---

[![](https://www.mql5.com/ff/si/6pp0j40fqxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=luckhiizjxvmvgigcufevttapwwrwbld&s=08cd1d929f27358481aded3c1c5f4e75a9bd5f52c477127afef2a5c532aec5c5&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=agrycbnsnnmiiqooscgwoxkmhoblpwtu&ssn=1769090428809824593&ssn_dr=0&ssn_sr=0&fv_date=1769090428&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F14680&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Developing%20a%20multi-currency%20Expert%20Advisor%20(Part%209)%3A%20Collecting%20optimization%20results%20for%20single%20trading%20strategy%20instances%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176909042887733993&fz_uniq=5048829533920010108&sv=2552)

MetaTrader 5 / Tester


### Introduction

We have already implemented a lot of interesting things in the previous [articles](https://www.mql5.com/ru/blogs/post/756958). We have a trading strategy or several trading strategies that we can implement in the EA. Besides, we have developed a structure for connecting many instances of trading strategies in a single EA, added tools for managing the maximum allowable drawdown, looked at possible ways of automated selection of sets of strategy parameters for their best work in a group, learned how to assemble an EA from groups of strategy instances and even from groups of different groups of strategy instances. But the value of the results already obtained will greatly increase if we manage to combine them together.

Let's try to outline a general structure within the article framework: single trading strategies are fed into the input, while the output is a ready-made EA, which uses selected and grouped copies of the original trading strategies that provide the best trading results.

After drawing up a rough road map, let's take a closer look at some section of it, analyze what we need to implement the selected stage, and get down to the actual implementation.

### Main stages

Let's list the main stages that we will have to go through while developing the EA:

1. **Implementing a trading strategy**. We develop the class derived from _CVirtualStrategy_, which implements the trading logic of opening, maintaining and closing virtual positions and orders. We did this in the first four parts of the series.

2. **Trading strategy optimization**. We select good sets of inputs for a trading strategy that show noteworthy results. If none are found, then we return to point 1.

As a rule, it is more convenient for us to perform optimization on one symbol and timeframe. For genetic optimization, we will most likely need to run it several times with different optimization criteria, including some of our own. It will only be possible to use brute force optimization in strategies with a very small number of parameters. Even in our model strategy, exhaustive search is too expensive. Therefore, further on, while speaking about optimization, I will imply genetic optimization in the MetaTrader 5 strategy tester. The optimization process was not described in detail in the articles, since it is pretty standard.

3. **Clustering of sets**. This step is not mandatory, but will save some time in the next step. Here we significantly reduce the number of sets of parameters of trading strategy instances, among which we will select suitable groups. This is described in the sixth part.

4. **Selecting groups of parameter sets**. Based on the results of the previous stage, perform optimization selecting

the most compatible sets of parameters of trading strategy instances that produce the best results. This is also mainly described in the sixth and seventh parts.

5. **Selecting groups from groups of parameter sets**. Now combine the results of the previous stage into groups using the same principle as when combining the sets of single instance parameter sets.

6. **Iterating through symbols and timeframes**. Repeat steps 2 - 5 for all desired symbols and timeframes. Perhaps, in addition to a symbol and timeframe, it is possible to conduct separate optimization on certain classes of other inputs for some trading strategies.

7. **Other strategies**. If you have other trading strategies in mind, then repeat steps 1 - 6 for each of them.

8. **Assembling the EA**. We collect all the best groups of groups found for different trading strategies, symbols, timeframes and other parameters into one final EA.


Each stage, upon completion, generates some data that needs to be saved and used in the next stages. So far we have been using temporary improvised means, convenient enough to use once or twice, but not particularly convenient for repeated use.

For example, we saved the optimization results after the second stage in an Excel file, then manually added the missing columns, and then, having saved it as a CSV file, used it in the third stage.

We either used the results of the third stage directly from the strategy tester interface, or saved them again in Excel files, carried out some processing there, and again used the results obtained from the tester interface.

We did not actually carry out the fifth stage, only noting the possibility of carrying it out. Therefore, it never came to fruition.

For all these received data, we would like to implement a single storage and usage structure.

### Implementation options

Essentially, the main type of data we need to store and use is the optimization results of multiple EAs. As you know, the strategy tester records all optimization results in a separate cache file with the \*.opt extension, which can then be reopened in the tester or even opened in the tester of another MetaTrader 5 terminal. The file name is determined from the hash calculated based on the name of the optimized EA and the optimization parameters. This allows us not to lose information about the passes already made when continuing optimization after its early interruption or after changing the optimization criterion.

Therefore, one of the options under consideration is the use of optimization cache files to store intermediate results. There is a good [library](https://www.mql5.com/en/code/26223) from [fxsaber](https://www.mql5.com/en/users/fxsaber) allowing us to access all saved information from MQL5 programs.

But as the number of optimizations performed increases, the number of files with their results will also increase. In order not to get confused, we will need to come up with some additional structure for arranging the storage and working with these cache files. If optimization is not carried out on one server, then it will be necessary to implement synchronization or storing all cache files in one place. In addition, for the next stage we will still need some processing to export the obtained optimization results to the EA at the next stage.

Then let's look at arranging the storage of all results in the database. At first glance, this would require quite a lot of time to implement. But this work can be broken down into smaller stages, and we will be able to use its results immediately, without waiting for full implementation. This approach also allows for greater freedom in choosing the most convenient means of intermediate processing of stored results. For example, we can assign some processing to simple SQL queries, something will be calculated in MQL5, and something in Python or R programs. We will be able to try different processing options and choose the most suitable one.

MQL5 offers built-in functions for working with the SQLite database. There were also implementations of third-party libraries that allow working, say, with MySQL. It is not yet clear whether SQLite capabilities will be enough for us, but most likely this database will be sufficient for our needs. If it is not sufficient, then we will think about migrating to another DBMS.

### Let's start designing the database

First, we need to identify the entities whose information we want to store. Of course, one test run is one of them. The fields of this entity will include test input data fields and test result fields. Generally, they can be distinguished as separate entities. The essence of the input data can be broken down into even smaller entities: the EA, optimization settings and EA single-pass parameters. But let's continue to be guided by the principle of least action. To begin with, one table with fields for the pass results that we used in previous articles and one or two text fields for placing the necessary information about the pass inputs will be sufficient for us.

Such a table can be created with the following SQL query:

```
CREATE TABLE passes (
    id                    INTEGER  PRIMARY KEY AUTOINCREMENT,
    pass                  INT,	-- pass index

    inputs                TEXT, -- pass input values
    params                TEXT, -- additional pass data

    initial_deposit       REAL, -- pass results...
    withdrawal            REAL,
    profit                REAL,
    gross_profit          REAL,
    gross_loss            REAL,
    max_profittrade       REAL,
    max_losstrade         REAL,
    conprofitmax          REAL,
    conprofitmax_trades   REAL,
    max_conwins           REAL,
    max_conprofit_trades  REAL,
    conlossmax            REAL,
    conlossmax_trades     REAL,
    max_conlosses         REAL,
    max_conloss_trades    REAL,
    balancemin            REAL,
    balance_dd            REAL,
    balancedd_percent     REAL,
    balance_ddrel_percent REAL,
    balance_dd_relative   REAL,
    equitymin             REAL,
    equity_dd             REAL,
    equitydd_percent      REAL,
    equity_ddrel_percent  REAL,
    equity_dd_relative    REAL,
    expected_payoff       REAL,
    profit_factor         REAL,
    recovery_factor       REAL,
    sharpe_ratio          REAL,
    min_marginlevel       REAL,
    deals                 REAL,
    trades                REAL,
    profit_trades         REAL,
    loss_trades           REAL,
    short_trades          REAL,
    long_trades           REAL,
    profit_shorttrades    REAL,
    profit_longtrades     REAL,
    profittrades_avgcon   REAL,
    losstrades_avgcon     REAL,
    complex_criterion     REAL,
    custom_ontester       REAL,
    pass_date             DATETIME DEFAULT (datetime('now') )
                                   NOT NULL
);
```

Let's create the auxiliary _CDatabase_ class, which will contain methods for working with the database. We can make it static, since we do not need many instances in one program, just one is sufficient. Since we are currently planning to accumulate all the information in one database, we can rigidly specify the database file name in the source code.

This class will contain the _s\_db_ field for storing the open database handle. The _Open()_ database opening method will set its value. If the database has not yet been created at the time of opening, it will be created by calling the _Create()_ method. Once opened, we can execute single SQL queries to the database using the Execute() method or bulk SQL queries in a single transaction using the _ExecuteTransaction()_ method. At the end, we will close the database using the _Close()_ method.

We can also declare a short macro that allows us to replace the long _CDatabase_ class name with the shorter _DB_.

```
#define DB CDatabase

//+------------------------------------------------------------------+
//| Class for handling the database                                  |
//+------------------------------------------------------------------+
class CDatabase {
   static int        s_db;          // DB connection handle
   static string     s_fileName;    // DB file name
public:
   static bool       IsOpen();      // Is the DB open?

   static void       Create();      // Create an empty DB
   static void       Open();        // Opening DB
   static void       Close();       // Closing DB

   // Execute one query to the DB
   static bool       Execute(string &query);

   // Execute multiple DB queries in one transaction
   static bool       ExecuteTransaction(string &queries[]);
};

int    CDatabase::s_db       =  INVALID_HANDLE;
string CDatabase::s_fileName = "database.sqlite";
```

In the database creation method, we will simply create an array with SQL queries for creating tables and execute them in one transaction:

```
//+------------------------------------------------------------------+
//| Create an empty DB                                               |
//+------------------------------------------------------------------+
void CDatabase::Create() {
   // Array of DB creation requests
   string queries[] = {
      "DROP TABLE IF EXISTS passes;",

      "CREATE TABLE passes ("
      "id                    INTEGER  PRIMARY KEY AUTOINCREMENT,"
      "pass                  INT,"
      "inputs                TEXT,"
      "params                TEXT,"
      "initial_deposit       REAL,"
      "withdrawal            REAL,"
      "profit                REAL,"
      "gross_profit          REAL,"
      "gross_loss            REAL,"
      ...
      "pass_date             DATETIME DEFAULT (datetime('now') ) NOT NULL"
      ");"
      ,
   };

   // Execute all requests
   ExecuteTransaction(queries);
}
```

In the open database method, we will first try to open an existing database file. If it does not exist, then we create and open it, after which we create the database structure by calling the _Create()_ method:

```
//+------------------------------------------------------------------+
//| Is the DB open?                                                  |
//+------------------------------------------------------------------+
bool CDatabase::IsOpen() {
   return (s_db != INVALID_HANDLE);
}
...

//+------------------------------------------------------------------+
//| Open DB                                                          |
//+------------------------------------------------------------------+
void CDatabase::Open() {
// Try to open an existing DB file
   s_db = DatabaseOpen(s_fileName, DATABASE_OPEN_READWRITE | DATABASE_OPEN_COMMON);

// If the DB file is not found, try to create it when opening
   if(!IsOpen()) {
      s_db = DatabaseOpen(s_fileName,
                          DATABASE_OPEN_READWRITE | DATABASE_OPEN_CREATE |
                          DATABASE_OPEN_COMMON);

      // Report an error in case of failure
      if(!IsOpen()) {
         PrintFormat(__FUNCTION__" | ERROR: %s open failed with code %d",
                     s_fileName, GetLastError());
         return;
      }

      // Create the database structure
      Create();
   }
   PrintFormat(__FUNCTION__" | Database %s opened successfully", s_fileName);
}
```

In the method of executing multiple _ExecuteTransaction()_ queries, we create a transaction and start executing all SQL queries in a loop one by one. If an error occurs while executing the next request, we interrupt the loop, report the error, and cancel all previous requests within this transaction. If no errors occur, confirm the transaction:

```
//+------------------------------------------------------------------+
//| Execute multiple DB queries in one transaction                   |
//+------------------------------------------------------------------+
bool CDatabase::ExecuteTransaction(string &queries[]) {
// Open a transaction
   DatabaseTransactionBegin(s_db);

   bool res = true;
// Send all execution requests
   FOREACH(queries, {
      res &= Execute(queries[i]);
      if(!res) break;
   });

// If an error occurred in any request, then
   if(!res) {
      // Report it
      PrintFormat(__FUNCTION__" | ERROR: Transaction failed, error code=%d", GetLastError());
      // Cancel transaction
      DatabaseTransactionRollback(s_db);
   } else {
      // Otherwise, confirm transaction
      DatabaseTransactionCommit(s_db);
      PrintFormat(__FUNCTION__" | Transaction done successfully");
   }
   return res;
}
```

Save the changes in the _Database.mqh_ file of the current folder.

### Modifying the EA to collect optimization data

When using only agents on the local computer in the optimization process, we can arrange saving the pass results to the database either in _OnTester()_, or _OnDeinit()_ handler. When using agents in a local network or in the MQL5 Cloud Network, it will be very difficult, if possible, to achieve saving the results. Fortunately, MQL5 offers a great standard way to get any information from test agents, wherever they are, by creating, sending and receiving data frames.

This mechanism is described in sufficient detail in the [reference](https://www.mql5.com/en/docs/optimization_frames) and in the [AlgoBook](https://www.mql5.com/en/book/automation/tester/tester_frameadd). In order to use it, we need to add three additional event handlers to the optimized: _OnTesterInit()_, _OnTesterPass()_ and _OnTesterDeinit()_.

Optimization is always launched from some MetaTrader 5 terminal, which we will henceforth conditionally call the main one. When an EA with such handlers is launched from the main terminal for optimization, a new chart is opened in the main terminal, and another instance of the EA is launched on this chart before distributing the EA instances to testing agents to perform normal optimization passes with different sets of parameters.

This instance is launched in a special mode: the standard _OnInit()_, _OnTick()_ and _OnDeinit()_ handlers are not executed in it. Only these three new handlers are executed instead. This mode even has its own name - the mode of collecting frames of optimization results. If necessary, we can check that the EA is running in this mode in the EA functions by calling the _MQLInfoInteger()_ function the following way:

```
// Check if the EA is running in data frame collection mode
bool isFrameMode = MQLInfoInteger(MQL_FRAME_MODE);
```

As the names suggest, in frame collection mode, the _OnTesterInit()_ handler runs once before optimization, _OnTesterPass()_ runs every time any of the test agents completes its pass, while _OnTesterDeinit()_ runs once after all scheduled optimization passes are completed or when optimization is interrupted.

The EA instance launched on the main terminal chart in the frame collection mode will be responsible for collecting data frames from all test agents. "Data frame" is just a convenient name to use when describing the data exchange between test agents and the EA in the main terminal. It denotes a data set with a name and a numeric ID that the test agent created and sent to the main terminal after completing a single optimization pass.

It should be noted that it makes sense to create data frames only in the EA instances operating in normal mode on the test agents, and to collect and handle data frames only in the EA instance in the main terminal operating in frame collection mode. So let's start with creating frames.

We can place the creation of frames in the EA in the OnTester() handler or in any function or method called from OnTester(). The handler is launched after the completion of the pass, and we can get in it the values of all statistical characteristics of the completed pass and, if necessary, calculate the value of the user criterion for evaluating the pass results.

We currently have the code in it that calculates a custom criterion showing the predicted profit that could be obtained given the maximum achievable drawdown of 10%:

```
//+------------------------------------------------------------------+
//| Test results                                                     |
//+------------------------------------------------------------------+
double OnTester(void) {
// Maximum absolute drawdown
   double balanceDrawdown = TesterStatistics(STAT_EQUITY_DD);

// Profit
   double profit = TesterStatistics(STAT_PROFIT);

// The ratio of possible increase in position sizes for the drawdown of 10% of fixedBalance_
   double coeff = fixedBalance_ * 0.1 / balanceDrawdown;

// Recalculate the profit
   double fittedProfit = profit * coeff;

   return fittedProfit;
}
```

Let's move this code from the _SimpleVolumesExpertSingle.mq5_ EA file to the new _CVirtualAdvisor_ method class, while the EA is left with returning the method call result:

```
//+------------------------------------------------------------------+
//| Test results                                                     |
//+------------------------------------------------------------------+
double OnTester(void) {
   return expert.Tester();
}
```

When moving, we should consider that we can no longer use the _fixedBalance\__ variable inside the method, since it may not be present in another EA as well. But its value can be obtained from the _CMoney_ static class by calling the _CMoney::FixedBalance()_ method. Along the way, we will make one more change to the calculation of our user criterion. After determining the projected profit, we will recalculate it per unit of time, for example, profit per year. This will allow us to roughly compare the results of passes over periods of different lengths.

To do this, we need to remember the test start date in the EA. Let's add the new property _m\_fromDate_, which is to store the current time in the EA object constructor.

```
//+------------------------------------------------------------------+
//| Class of the EA handling virtual positions (orders)              |
//+------------------------------------------------------------------+
class CVirtualAdvisor : public CAdvisor {
protected:
   ...

   datetime          m_fromDate;

public:
   ...
   virtual double    Tester() override;         // OnTester event handler
   ...
};

//+------------------------------------------------------------------+
//| OnTester event handler                                           |
//+------------------------------------------------------------------+
double CVirtualAdvisor::Tester() {
// Maximum absolute drawdown
   double balanceDrawdown = TesterStatistics(STAT_EQUITY_DD);

// Profit
   double profit = TesterStatistics(STAT_PROFIT);

// The ratio of possible increase in position sizes for the drawdown of 10% of fixedBalance_
   double coeff = CMoney::FixedBalance() * 0.1 / balanceDrawdown;

// Calculate the profit in annual terms
   long totalSeconds = TimeCurrent() - m_fromDate;
   double fittedProfit = profit * coeff * 365 * 24 * 3600 / totalSeconds ;

// Perform data frame generation on the test agent
   CTesterHandler::Tester(fittedProfit,
                          ~((CVirtualStrategy *) m_strategies[0]));

   return fittedProfit;
}
```

Later, we might make several custom optimization criteria, and then this code will be moved again to a new location. But for now, let's not get distracted by the extensive topic of studying various fitness functions for optimizing EAs and leave the code as is.

The _SimpleVolumesExpertSingle.mq5_ EA file now gets new handlers _OnTesterInit()_, _OnTesterPass()_ and _OnTesterDeinit()_. Since, according to our plan, the logic of these functions should be the same for all EAs, we will first lower their implementation to the EA level ( _CVirtualAdvisor_ class object).

It should be taken into account that when the EA is launched in the main terminal in the frame collection mode, the _OnInit()_ function, in which the EA instance is created, will not be executed. Therefore, in order not to add creation/deletion of an EA instance to new handlers, make the methods for handling these events static in the _CVirtualAdvisor_ class. Then we need to add the following code to the EA:

```
//+------------------------------------------------------------------+
//| Initialization before starting optimization                      |
//+------------------------------------------------------------------+
int OnTesterInit(void) {
   return CVirtualAdvisor::TesterInit();
}

//+------------------------------------------------------------------+
//| Actions after completing the next optimization pass              |
//+------------------------------------------------------------------+
void OnTesterPass() {
   CVirtualAdvisor::TesterPass();
}

//+------------------------------------------------------------------+
//| Actions after optimization is complete                           |
//+------------------------------------------------------------------+
void OnTesterDeinit(void) {
   CVirtualAdvisor::TesterDeinit();
}
```

Another change we can make for the future is to get rid of the separate call to the _CVirtualAdvisor::Add()_ method for adding trading strategies to the EA after it is created. Instead, we will immediately transfer information about strategies to the EA's constructor, while it will call the _Add()_ method on its own. Then this method can be removed from the public part.

With this approach, the _OnInit()_ EA initialization function will look as follows:

```
int OnInit() {
   CMoney::FixedBalance(fixedBalance_);

// Create an EA handling virtual positions
   expert = new CVirtualAdvisor(
      new CSimpleVolumesStrategy(
         symbol_, timeframe_,
         signalPeriod_, signalDeviation_, signaAddlDeviation_,
         openDistance_, stopLevel_, takeLevel_, ordersExpiration_,
         maxCountOfOrders_, 0), // One strategy instance
      magic_, "SimpleVolumesSingle", true);

   return(INIT_SUCCEEDED);
}
```

Save the changes in the _SimpleVolumesExpertSingle.mq5_ file of the current folder.

### Modifying the EA class

To avoid overloading the _CVirtualAdvisor_ EA class, let's move the code of the _TesterInit_, _TesterPass_ and _OnTesterDeinit_ event handlers to the separate _CTesterHandler_ class, in which we will create static methods to handle each of these events. In this case, we need to add to the _CVirtualAdvisor_ class approximately the same code as in the main EA file:

```
//+------------------------------------------------------------------+
//| Class of the EA handling virtual positions (orders)              |
//+------------------------------------------------------------------+
class CVirtualAdvisor : public CAdvisor {
   ...

public:
   ...
   static int        TesterInit();     // OnTesterInit event handler
   static void       TesterPass();     // OnTesterDeinit event handler
   static void       TesterDeinit();   // OnTesterDeinit event handler
};

//+------------------------------------------------------------------+
//| Initialization before starting optimization                      |
//+------------------------------------------------------------------+
int CVirtualAdvisor::TesterInit() {
   return CTesterHandler::TesterInit();
}

//+------------------------------------------------------------------+
//| Actions after completing the next optimization pass              |
//+------------------------------------------------------------------+
void CVirtualAdvisor::TesterPass() {
   CTesterHandler::TesterPass();
}

//+------------------------------------------------------------------+
//| Actions after optimization is complete                           |
//+------------------------------------------------------------------+
void CVirtualAdvisor::TesterDeinit() {
   CTesterHandler::TesterDeinit();
}
```

Let's also make some additions to the EA object constructor code. Move all actions from the constructor to the new _Init()_ initialization method with future improvements in mind. This will allow us to add multiple constructors with different sets of parameters that will all use the same initialization method after a little preprocessing of the parameters.

Let's add constructors whose first argument will be either a strategy object or a strategy group object. Then we can add strategies to the EA directly in the constructor. In this case, we no longer need to call the _Add()_ method in the _OnInit()_ EA function.

```
//+------------------------------------------------------------------+
//| Class of the EA handling virtual positions (orders)              |
//+------------------------------------------------------------------+
class CVirtualAdvisor : public CAdvisor {
protected:
   ...

   datetime          m_fromDate;

public:
                     CVirtualAdvisor(CVirtualStrategy *p_strategy, ulong p_magic = 1, string p_name = "", bool p_useOnlyNewBar = false); // Constructor
                     CVirtualAdvisor(CVirtualStrategyGroup *p_group, ulong p_magic = 1, string p_name = "", bool p_useOnlyNewBar = false); // Constructor
   void              CVirtualAdvisor::Init(CVirtualStrategyGroup *p_group,
                                           ulong p_magic = 1,
                                           string p_name = "",
                                           bool p_useOnlyNewBar = false
                                          );
   ...
};

...

//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CVirtualAdvisor::CVirtualAdvisor(CVirtualStrategy *p_strategy,
                                 ulong p_magic = 1,
                                 string p_name = "",
                                 bool p_useOnlyNewBar = false
                                ) {
   CVirtualStrategy *strategies[] = {p_strategy};
   Init(new CVirtualStrategyGroup(strategies), p_magic, p_name, p_useOnlyNewBar);
};

//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CVirtualAdvisor::CVirtualAdvisor(CVirtualStrategyGroup *p_group,
                                 ulong p_magic = 1,
                                 string p_name = "",
                                 bool p_useOnlyNewBar = false
                                ) {
   Init(p_group, p_magic, p_name, p_useOnlyNewBar);
};

//+------------------------------------------------------------------+
//| EA initialization method                                         |
//+------------------------------------------------------------------+
void CVirtualAdvisor::Init(CVirtualStrategyGroup *p_group,
                           ulong p_magic = 1,
                           string p_name = "",
                           bool p_useOnlyNewBar = false
                          ) {
// Initialize the receiver with a static receiver
   m_receiver = CVirtualReceiver::Instance(p_magic);
// Initialize the interface with the static interface
   m_interface = CVirtualInterface::Instance(p_magic);
   m_lastSaveTime = 0;
   m_useOnlyNewBar = p_useOnlyNewBar;
   m_name = StringFormat("%s-%d%s.csv",
                         (p_name != "" ? p_name : "Expert"),
                         p_magic,
                         (MQLInfoInteger(MQL_TESTER) ? ".test" : "")
                        );

   m_fromDate = TimeCurrent();

   Add(p_group);
   delete p_group;
};
```

Save the changes in the _VirtualExpert.mqh_ of the current folder.

### Optimization event handling class

Let's now focus directly on the implementation of actions performed before the start, after the completion of the pass, and after the completion of the optimization. We will create the _CTesterHandler_ class and add to it methods for handling the necessary events, as well as a couple of auxiliary methods placed in the closed part of the class:

```
//+------------------------------------------------------------------+
//| Optimization event handling class                                |
//+------------------------------------------------------------------+
class CTesterHandler {
   static string     s_fileName;                   // File name for writing frame data
   static void       ProcessFrames();              // Handle incoming frames
   static string     GetFrameInputs(ulong pass);   // Get pass inputs
public:
   static int        TesterInit();     // Handle the optimization start in the main terminal
   static void       TesterDeinit();   // Handle the optimization completion in the main terminal
   static void       TesterPass();     // Handle the completion of a pass on an agent in the main terminal

   static void       Tester(const double OnTesterValue,
                            const string params);  // Handle completion of tester pass for agent
};

string CTesterHandler::s_fileName = "data.bin";    // File name for writing frame data
```

The event handlers for the main terminal look very simple, since we will move the main code into auxiliary functions:

```
//+------------------------------------------------------------------+
//| Handling the optimization start in the main terminal             |
//+------------------------------------------------------------------+
int CTesterHandler::TesterInit(void) {
// Open / create a database
   DB::Open();

// If failed to open it, we do not start optimization
   if(!DB::IsOpen()) {
      return INIT_FAILED;
   }

// Close a successfully opened database
   DB::Close();

   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Handling the optimization completion in the main terminal        |
//+------------------------------------------------------------------+
void CTesterHandler::TesterDeinit(void) {
// Handle the latest data frames received from agents
   ProcessFrames();

// Close the chart with the EA running in frame collection mode
   ChartClose();
}

//+--------------------------------------------------------------------+
//| Handling the completion of a pass on an agent in the main terminal |
//+--------------------------------------------------------------------+
void CTesterHandler::TesterPass(void) {
// Handle data frames received from the agent
   ProcessFrames();
}
```

The actions performed after the completion of one pass will exist in two versions:

- **For the test agent**. It is there that, after the passage, the necessary information will be collected and a data frame will be created for sending to the main terminal. These actions will be collected in the _Tester()_ event handler.

- **For the main terminal**. Here we can receive data frames from test agents, parse the information received in the frame and enter it into the database. These actions will be collected in the _TesterPass()_ handler.

Generating a data frame for the test agent should be performed in the EA, namely inside the _OnTester_ handler. Since we moved its code to the EA object level (to the _CVirtualAdvisor_ class), then this is where we need to add the _CTesterHandler::Tester()_ method. We will pass the newly calculated value of the custom optimization criterion and a string describing the parameters of the strategy, that was used in the optimized EA, as the method parameters. To form such a string, we will use the already created ~ (tilde) for the _CVirtualStrategy_ class objects.

```
//+------------------------------------------------------------------+
//| OnTester event handler                                           |
//+------------------------------------------------------------------+
double CVirtualAdvisor::Tester() {
// Maximum absolute drawdown
   double balanceDrawdown = TesterStatistics(STAT_EQUITY_DD);

// Profit
   double profit = TesterStatistics(STAT_PROFIT);

// The ratio of possible increase in position sizes for the drawdown of 10% of fixedBalance_
   double coeff = CMoney::FixedBalance() * 0.1 / balanceDrawdown;

// Calculate the profit in annual terms
   long totalSeconds = TimeCurrent() - m_fromDate;
   double fittedProfit = profit * coeff * 365 * 24 * 3600 / totalSeconds ;

// Perform data frame generation on the test agent
   CTesterHandler::Tester(fittedProfit,
                          ~((CVirtualStrategy *) m_strategies[0]));

   return fittedProfit;
}
```

In the _CTesterHandler::Tester()_ method itself, go through all possible names of available statistical characteristics, get their values, convert them to strings and add these strings to the _stats_ array. Why did we need to convert real numeric characteristics to strings? Only so that they could be passed in one frame with a string description of the strategy parameters. In one frame, we can pass either an array of values of one of the simple types (strings do not apply to) or a pre-created file with any data. Therefore, in order to avoid the hassle of sending two different frames (one containing numbers and the other containing strings from a file), we will convert all the data into strings, write them to a file, and send its contents in one frame:

```
//+------------------------------------------------------------------+
//| Handling completion of tester pass for agent                     |
//+------------------------------------------------------------------+
void CTesterHandler::Tester(double custom,   // Custom criteria
                            string params    // Description of EA parameters in the current pass
                           ) {
// Array of names of saved statistical characteristics of the pass
   ENUM_STATISTICS statNames[] = {
      STAT_INITIAL_DEPOSIT,
      STAT_WITHDRAWAL,
      STAT_PROFIT,
      ...
   };

// Array for values of statistical characteristics of the pass as strings
   string stats[];
   ArrayResize(stats, ArraySize(statNames));

// Fill the array of values of statistical characteristics of the pass
   FOREACH(statNames, stats[i] = DoubleToString(TesterStatistics(statNames[i]), 2));

// Add the custom criterion value to it
   APPEND(stats, DoubleToString(custom, 2));

// Screen the quotes in the description of parameters just in case
   StringReplace(params, "'", "\\'");

// Open the file to write data for the frame
   int f = FileOpen(s_fileName, FILE_WRITE | FILE_TXT | FILE_ANSI);

// Write statistical characteristics
   FOREACH(stats, FileWriteString(f, stats[i] + ","));

// Write a description of the EA parameters
   FileWriteString(f, StringFormat("'%s'", params));

// Close the file
   FileClose(f);

// Create a frame with data from the recorded file and send it to the main terminal
   if(!FrameAdd("", 0, 0, s_fileName)) {
      PrintFormat(__FUNCTION__" | ERROR: Frame add error: %d", GetLastError());
   }
}
```

Finally, let's consider an auxiliary method that will accept data frames and save the information from them to the database. In this method, we receive in a loop all incoming frames that have not yet been handled at the current moment. From each frame, we obtain data in the form of a character array and convert them into a string. Next, we form a string with the names and values of the parameters of the pass with the given index. We use the obtained values to form an SQL query to insert a new row into the _passes_ table in our database. Add the created SQL query to the SQL query array.

After handling all currently received data frames in this way, we execute all SQL queries from the array within a single transaction.

```
//+------------------------------------------------------------------+
//| Handling incoming frames                                         |
//+------------------------------------------------------------------+
void CTesterHandler::ProcessFrames(void) {
// Open the database
   DB::Open();

// Variables for reading data from frames
   string   name;      // Frame name (not used)
   ulong    pass;      // Frame pass index
   long     id;        // Frame type ID (not used)
   double   value;     // Single frame value (not used)
   uchar    data[];    //  Frame data array as a character array

   string   values;    // Frame data as a string
   string   inputs;    // String with names and values of pass parameters
   string   query;     // A single SQL query string
   string   queries[]; // SQL queries for adding records to the database

// Go through frames and read data from them
   while(FrameNext(pass, name, id, value, data)) {
      // Convert the array of characters read from the frame into a string
      values = CharArrayToString(data);

      // Form a string with names and values of the pass parameters
      inputs = GetFrameInputs(pass);

      // Form an SQL query from the received data
      query = StringFormat("INSERT INTO passes "
                           "VALUES (NULL, %d, %s,\n'%s',\n'%s');",
                           pass, values, inputs,
                           TimeToString(TimeLocal(), TIME_DATE | TIME_SECONDS));

      // Add it to the SQL query array
      APPEND(queries, query);
   }

// Execute all requests
   DB::ExecuteTransaction(queries);

// Close the database
   DB::Close();
}
```

The _GetFrameInputs()_ auxiliary method for forming a string with names and values of input variables of the pass has been taken from the [AlgoBook](https://www.mql5.com/en/book/automation/tester/tester_framenext) and slightly supplemented to suit our needs.

Save the obtained code in the _TesterHandler.mqh_ file of the current folder.

### Checking operation

To test the functionality, let's run optimization with a small number of parameters to be iterated over a relatively short time period. After the optimization process is completed, we can look at the results in the strategy tester and in the created database.

![](https://c.mql5.com/2/76/6226191855886.png)

Fig. 1. Optimization results in the strategy tester

![](https://c.mql5.com/2/76/3562929987488.png)

Fig. 2. Optimization results in the database

As we can see, the database results match the results in the tester: with the same sorting by user criteria, we observe the same sequence of profit values in both cases. The best pass reports that the expected profit may exceed USD 5000 within a year with the initial deposit of USD 10,000 and a maximum achievable drawdown of 10% of the initial deposit (USD 1000). Currently, however, we are not so interested in the quantitative characteristics of the optimization results as in the fact that they can now be stored in a database.

### Conclusion

So, we are one step closer to our goal. We managed to save the results of the conducted optimizations of the EA parameters to our database. In this way, we have provided the foundation for further automated implementation of the second stage of the EA development.

There are still quite a few questions left behind the scenes. Many things had to be postponed for the future, since their implementation would require significant costs. But having received the current results, we can more clearly formulate the direction of further project development.

The implemented saving currently works only for one optimization process in the sense that we save information about the passes, but it is still difficult to extract groups of strings related to one optimization process from them. To do this, we will need to make changes to the database structure, which is now made extremely simple. In the future, we will try to automate the launch of several sequential optimization processes with preliminary assignment of different options for the parameters to be optimized.

Thank you for your attention! See you soon!

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/14680](https://www.mql5.com/ru/articles/14680)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/14680.zip "Download all attachments in the single ZIP archive")

[Advisor.mqh](https://www.mql5.com/en/articles/download/14680/advisor.mqh "Download Advisor.mqh")(4.4 KB)

[Database.mqh](https://www.mql5.com/en/articles/download/14680/database.mqh "Download Database.mqh")(13.37 KB)

[Interface.mqh](https://www.mql5.com/en/articles/download/14680/interface.mqh "Download Interface.mqh")(3.21 KB)

[Macros.mqh](https://www.mql5.com/en/articles/download/14680/macros.mqh "Download Macros.mqh")(2.28 KB)

[Money.mqh](https://www.mql5.com/en/articles/download/14680/money.mqh "Download Money.mqh")(4.61 KB)

[NewBarEvent.mqh](https://www.mql5.com/en/articles/download/14680/newbarevent.mqh "Download NewBarEvent.mqh")(11.52 KB)

[Receiver.mqh](https://www.mql5.com/en/articles/download/14680/receiver.mqh "Download Receiver.mqh")(1.79 KB)

[SimpleVolumesExpertSingle.mq5](https://www.mql5.com/en/articles/download/14680/simplevolumesexpertsingle.mq5 "Download SimpleVolumesExpertSingle.mq5")(9.92 KB)

[SimpleVolumesStrategy.mqh](https://www.mql5.com/en/articles/download/14680/simplevolumesstrategy.mqh "Download SimpleVolumesStrategy.mqh")(33.63 KB)

[Strategy.mqh](https://www.mql5.com/en/articles/download/14680/strategy.mqh "Download Strategy.mqh")(1.73 KB)

[TesterHandler.mqh](https://www.mql5.com/en/articles/download/14680/testerhandler.mqh "Download TesterHandler.mqh")(17.5 KB)

[VirtualAdvisor.mqh](https://www.mql5.com/en/articles/download/14680/virtualadvisor.mqh "Download VirtualAdvisor.mqh")(22.65 KB)

[VirtualChartOrder.mqh](https://www.mql5.com/en/articles/download/14680/virtualchartorder.mqh "Download VirtualChartOrder.mqh")(10.84 KB)

[VirtualInterface.mqh](https://www.mql5.com/en/articles/download/14680/virtualinterface.mqh "Download VirtualInterface.mqh")(8.41 KB)

[VirtualOrder.mqh](https://www.mql5.com/en/articles/download/14680/virtualorder.mqh "Download VirtualOrder.mqh")(39.52 KB)

[VirtualReceiver.mqh](https://www.mql5.com/en/articles/download/14680/virtualreceiver.mqh "Download VirtualReceiver.mqh")(17.43 KB)

[VirtualStrategy.mqh](https://www.mql5.com/en/articles/download/14680/virtualstrategy.mqh "Download VirtualStrategy.mqh")(9.22 KB)

[VirtualStrategyGroup.mqh](https://www.mql5.com/en/articles/download/14680/virtualstrategygroup.mqh "Download VirtualStrategyGroup.mqh")(6.1 KB)

[VirtualSymbolReceiver.mqh](https://www.mql5.com/en/articles/download/14680/virtualsymbolreceiver.mqh "Download VirtualSymbolReceiver.mqh")(33.82 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Moving to MQL5 Algo Forge (Part 4): Working with Versions and Releases](https://www.mql5.com/en/articles/19623)
- [Moving to MQL5 Algo Forge (Part 3): Using External Repositories in Your Own Projects](https://www.mql5.com/en/articles/19436)
- [Moving to MQL5 Algo Forge (Part 2): Working with Multiple Repositories](https://www.mql5.com/en/articles/17698)
- [Moving to MQL5 Algo Forge (Part 1): Creating the Main Repository](https://www.mql5.com/en/articles/17646)
- [Developing a multi-currency Expert Advisor (Part 24): Adding a new strategy (II)](https://www.mql5.com/en/articles/17328)
- [Developing a multi-currency Expert Advisor (Part 24): Adding a new strategy (I)](https://www.mql5.com/en/articles/17277)
- [Developing a multi-currency Expert Advisor (Part 23): Putting in order the conveyor of automatic project optimization stages (II)](https://www.mql5.com/en/articles/16913)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/472665)**
(1)


![Yuriy Bykov](https://c.mql5.com/avatar/avatar_na2.png)

**[Yuriy Bykov](https://www.mql5.com/en/users/antekov)**
\|
10 Nov 2024 at 12:49

Hi.

The database is created and resides on your computer in the terminal's data folder.

Look at the CDatabase::Create() method

![Creating an MQL5-Telegram Integrated Expert Advisor (Part 5): Sending Commands from Telegram to MQL5 and Receiving Real-Time Responses](https://c.mql5.com/2/92/MQL5-Telegram_Integrated_Expert_Advisor_lPart_5.png)[Creating an MQL5-Telegram Integrated Expert Advisor (Part 5): Sending Commands from Telegram to MQL5 and Receiving Real-Time Responses](https://www.mql5.com/en/articles/15750)

In this article, we create several classes to facilitate real-time communication between MQL5 and Telegram. We focus on retrieving commands from Telegram, decoding and interpreting them, and sending appropriate responses back. By the end, we ensure that these interactions are effectively tested and operational within the trading environment

![Introduction to MQL5 (Part 9): Understanding and Using Objects in MQL5](https://c.mql5.com/2/92/Introduction_to_MQL5_Part_9___LOGO____2.png)[Introduction to MQL5 (Part 9): Understanding and Using Objects in MQL5](https://www.mql5.com/en/articles/15764)

Learn to create and customize chart objects in MQL5 using current and historical data. This project-based guide helps you visualize trades and apply MQL5 concepts practically, making it easier to build tools tailored to your trading needs.

![Formulating Dynamic Multi-Pair EA (Part 1): Currency Correlation and Inverse Correlation](https://c.mql5.com/2/92/xurrency_Correlation_and_Inverse_Correlation___LOGO.png)[Formulating Dynamic Multi-Pair EA (Part 1): Currency Correlation and Inverse Correlation](https://www.mql5.com/en/articles/15378)

Dynamic multi pair Expert Advisor leverages both on correlation and inverse correlation strategies to optimize trading performance. By analyzing real-time market data, it identifies and exploits the relationship between currency pairs.

![Neural Networks Made Easy (Part 86): U-Shaped Transformer](https://c.mql5.com/2/75/Neural_networks_are_easy_vPart_86m____LOGO.png)[Neural Networks Made Easy (Part 86): U-Shaped Transformer](https://www.mql5.com/en/articles/14766)

We continue to study timeseries forecasting algorithms. In this article, we will discuss another method: the U-shaped Transformer.

[Best articles and CodeBase updates in MQL5.community channelsFollow us to ensure you never miss out on important updates![](https://www.mql5.com/ff/sh/n9yf51p2srwzfqh5z2/02.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=dgazvhktsxqakdvarucjbvmvzenwlyje&s=98a038fe082e458df8c4a1d8e116e3a6646fd5517f06e48b2356b7ee005817d6&uid=&ref=https://www.mql5.com/en/articles/14680&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5048829533920010108)

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