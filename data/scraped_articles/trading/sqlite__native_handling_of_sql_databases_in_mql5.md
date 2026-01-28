---
title: SQLite: Native handling of SQL databases in MQL5
url: https://www.mql5.com/en/articles/7463
categories: Trading, Integration
relevance_score: 3
scraped_at: 2026-01-23T18:16:40.446623
---

[![](https://www.mql5.com/ff/sh/zf7a2k61x98jzh89z2/01.png)Speed up your tradingUse our high-speed VPS for MetaTrader 4 and 5Learn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=qtrrsuiwuicrscmckjynyanztbditglq&s=c617dc80d90cfd3783ec1345eec2b419b281f10fec6eac77b3218984ac337259&uid=&ref=https://www.mql5.com/en/articles/7463&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5069302826976346837)

MetaTrader 5 / Tester


### Contents

- [Modern algorithmic trading in MetaTrader 5](https://www.mql5.com/en/articles/7463#algorithmic_trading)

- [Functions for working with databases](https://www.mql5.com/en/articles/7463#database_functions)
- [Simple query](https://www.mql5.com/en/articles/7463#simple_query)

- [Debugging SQL queries in MetaEditor](https://www.mql5.com/en/articles/7463#query_debug)
- [Auto reading of query results into the structure using \\
DatabaseReadBind()](https://www.mql5.com/en/articles/7463#databasereadbind_function)
- [Accelerating transactions by wrapping them into \\
DatabaseTransactionBegin()/DatabaseTransactionCommit()](https://www.mql5.com/en/articles/7463#transactions_speedup)
- [Handling trading history deals](https://www.mql5.com/en/articles/7463#deals_convert)
- [Portfolio analysis by strategies](https://www.mql5.com/en/articles/7463#portfolio_analysis)
- [Analyzing deals by symbols](https://www.mql5.com/en/articles/7463#analysis_by_symbols)
- [Analyzing deals by entry hours](https://www.mql5.com/en/articles/7463#analysis_by_entries)
- [Convenient data output to the EA log in DatabasePrint()](https://www.mql5.com/en/articles/7463#databaseprint_function)
- [Data import/export](https://www.mql5.com/en/articles/7463#import_export)
- [Saving optimization results to the database](https://www.mql5.com/en/articles/7463#optimization_store)
- [Optimizing query execution using indices](https://www.mql5.com/en/articles/7463#query_planning)
- [Integrating database handling into MetaEditor](https://www.mql5.com/en/articles/7463#metaeditor_database)

### Modern algorithmic trading in MetaTrader 5

MQL5 is a perfect solution for algorithmic trading since it is as close to C++ as possible in terms of both syntax and computation speed. The
MetaTrader 5 platform offers its users the modern specialized language for developing trading robots and custom indicators allowing them
to go beyond simple trading tasks and create analytical systems of any complexity.

In addition to asynchronous trading functions and [math \\
libraries](https://www.mql5.com/en/docs/standardlibrary/mathematics), traders also have access to the [network functions](https://www.mql5.com/en/docs/network),
importing data to [Python](https://www.mql5.com/en/docs/integration/python_metatrader5), parallel computing in [OpenCL](https://www.metatrader5.com/en/metaeditor/help/development/opencl "https://www.metatrader5.com/en/metaeditor/help/development/opencl"),
native [support \\
for .NET libraries](https://www.metatrader5.com/en/releasenotes/terminal/1898 "https://www.metatrader5.com/en/releasenotes/terminal/1898") with "smart" function import, [integration \\
with MS Visual Studio](https://www.metatrader5.com/en/metaeditor/help/development/c_dll "https://www.metatrader5.com/en/metaeditor/help/development/c_dll") and data visualization using [DirectX](https://www.mql5.com/en/docs/directx). These
indispensable tools in the arsenal of modern algorithmic trading currently allow users to solve a variety of tasks without leaving the
MetaTrader 5 trading platform.

### Functions for working with databases

The development of trading strategies is associated with handling large amounts of data. A trading algorithm in the form of a reliable and
fast MQL5 program is no longer sufficient. To obtain reliable results, traders also need to carry out a huge number of tests and
optimizations on a variety of trading instruments, save and handle the results, conduct an analysis and decide on where to go next.

Now, you are able to [work with databases](https://www.mql5.com/en/docs/database) using a simple and popular [SQLite](https://www.mql5.com/go?link=https://www.sqlite.org/index.html "https://www.sqlite.org/index.html")
engine directly in MQL5. [The \\
test results on the developers' website](https://www.mql5.com/go?link=https://sqlite.org/speed.html "https://sqlite.org/speed.html") show high speed of executing SQL queries. In most tasks, it outperformed PostgreSQL and MySQL.
In turn, we compared the speeds of these test executions on MQL5 and [LLVM \\
9.0.0](https://www.mql5.com/go?link=https://releases.llvm.org/9.0.0/tools/clang/docs/ReleaseNotes.html "https://releases.llvm.org/9.0.0/tools/clang/docs/ReleaseNotes.html") and showed them in the table. The execution results are given in milliseconds — the less the better.

| Name | Description | LLVM | MQL5 |
| --- | --- | --- | --- |
| Test 1 | 1000 INSERTs | 11572 | 8488 |
| Test 2 | 25000 INSERTs in a transaction | 59 | 60 |
| Test 3 | 25000 INSERTs into an indexed table | 102 | 105 |
| Test 4 | 100 SELECTs without an index | 142 | 150 |
| Test 5 | 100 SELECTs on a string comparison | 391 | 390 |
| Test 6 | Creating an index | 43 | 33 |
| Test 7 | 5000 SELECTs with an index | 385 | 307 |
| Test 8 | 1000 UPDATEs without an index | 58 | 54 |
| Test 9 | 25000 UPDATEs with an index | 161 | 165 |
| Test 10 | 25000 text UPDATEs with an index | 124 | 120 |
| Test 11 | INSERTs from a SELECT | 84 | 84 |
| Test 12 | DELETE without an index | 25 | 74 |
| Test 13 | DELETE with an index | 70 | 72 |
| Test 14 | A big INSERT after a big DELETE | 62 | 66 |
| Test 15 | A big DELETE followed by many small INSERTs | 33 | 33 |
| Test 16 | DROP TABLE: finished | 42 | 40 |

You can find the test details in the attached SqLiteTest.zip file. Specifications of the computer the measurements were taken on — Windows 10
x64, Intel Xeon  E5-2690 v3 @ 2.60GHz.

The results show that you can be sure of maximum performance when working with databases in MQL5. Those who have never encountered SQL before
will see that the language of structured queries allows them to solve many tasks quickly and elegantly without the need for complex loops and
samplings.

### Simple query

Databases store information in the form of tables, while receiving/modifying and adding new data is done using queries in [SQL](https://en.wikipedia.org/wiki/SQL "https://en.wikipedia.org/wiki/SQL")
language. Let's have a look at the creation of a simple database and obtaining data from it.

```
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
   string filename="company.sqlite";
//--- create or open the database in the common terminal folder
   int db=DatabaseOpen(filename, DATABASE_OPEN_READWRITE | DATABASE_OPEN_CREATE |DATABASE_OPEN_COMMON);
   if(db==INVALID_HANDLE)
     {
      Print("DB: ", filename, " open failed with code ", GetLastError());
      return;
     }
... working with the database

//--- close the database
   DatabaseClose(db);
  }
```

Creating and closing a database is similar to working with files. First, we create a handle for a database, then we check it and, finally, we close it.

Next, we check the presence of a table in the database. If the table already exists, the attempt to insert the data from the above example ends in an
error.

```
//--- if the COMPANY table exists, delete it
   if(DatabaseTableExists(db, "COMPANY"))
     {
      //--- delete the table
      if(!DatabaseExecute(db, "DROP TABLE COMPANY"))
        {
         Print("Failed to drop table COMPANY with code ", GetLastError());
         DatabaseClose(db);
         return;
        }
     }
//--- create the COMPANY table
   if(!DatabaseExecute(db, "CREATE TABLE COMPANY("
                       "ID INT PRIMARY KEY     NOT NULL,"
                       "NAME           TEXT    NOT NULL,"
                       "AGE            INT     NOT NULL,"
                       "ADDRESS        CHAR(50),"
                       "SALARY         REAL );"))
     {
      Print("DB: ", filename, " create table failed with code ", GetLastError());
      DatabaseClose(db);
      return;
     }
```

The table is created and deleted using queries, and the execution result should be checked at all times. The COMPANY table features only five
fields: entry ID, name, age, address and salary. The ID field is a key, i.e. a unique index. Indices allow for reliable definition of each
entry and can be used in different tables to link them together. This is similar to how a position ID links all deals and orders related to a
particular position.

Now the table should be filled with data. This is done using the INSERT query:

```
//--- enter data to the table
   if(!DatabaseExecute(db, "INSERT INTO COMPANY (ID,NAME,AGE,ADDRESS,SALARY) VALUES (1,'Paul',32,'California',25000.00); "
                       "INSERT INTO COMPANY (ID,NAME,AGE,ADDRESS,SALARY) VALUES (2,'Allen',25,'Texas',15000.00); "
                       "INSERT INTO COMPANY (ID,NAME,AGE,ADDRESS,SALARY) VALUES (3,'Teddy',23,'Norway',20000.00);"
                       "INSERT INTO COMPANY (ID,NAME,AGE,ADDRESS,SALARY) VALUES (4,'Mark',25,'Rich-Mond',65000.00);"))
     {
      Print("DB: ", filename, " insert failed with code ", GetLastError());
      DatabaseClose(db);
      return;
     }
```

As we can see, four entries are added to the COMPANY table. The sequence of fields and the values to be inserted to these fields are
specified for each entry. Each entry is inserted by a separate "INSERT...." query combined into a single query. In other words, we could
insert each entry into the table by a separate [DatabaseExecute()](https://www.mql5.com/en/docs/database/databaseexecute)
call.

Since, upon completion of the script operation, the database is saved to the _company.sqlite_ file, we would try to
write the same data to the COMPANY table having the same ID during the next launch of the script. This would result in an error. This is why we
deleted the table first so that to start the work from scratch every time the script is launched.

Now let's get all the entries from the COMPANY table where SALARY field > 15000. This is done using the [DatabasePrepare()](https://www.mql5.com/en/docs/database/databaseprepare)
function, which compiles the query text and returns the handle for it for subsequent use in [DatabaseRead()](https://www.mql5.com/en/docs/database/databaseread)
or DatabaseReadBind().

```
//--- create a query and get a handle for it
   int request=DatabasePrepare(db, "SELECT * FROM COMPANY WHERE SALARY>15000");
   if(request==INVALID_HANDLE)
     {
      Print("DB: ", filename, " request failed with code ", GetLastError());
      DatabaseClose(db);
      return;
     }
```

After the query has been successfully created, we need to obtain its execution results. We will do this using DatabaseRead(), which executes
the query during the first call and moves to the first entry in the results. With each subsequent call, it simply reads the next entry until it
reaches the end. In this case, it returns 'false', which means "no more entries".

```
//--- print all entries with the salary greater than 15000
   int    id, age;
   string name, address;
   double salary;
   Print("Persons with salary > 15000:");
   for(int i=0; DatabaseRead(request); i++)
     {
      //--- read the values of each field from the obtained entry
      if(DatabaseColumnInteger(request, 0, id) && DatabaseColumnText(request, 1, name) &&
         DatabaseColumnInteger(request, 2, age) && DatabaseColumnText(request, 3, address) && DatabaseColumnDouble(request, 4, salary))
         Print(i, ":  ", id, " ", name, " ", age, " ", address, " ", salary);
      else
        {
         Print(i, ": DatabaseRead() failed with code ", GetLastError());
         DatabaseFinalize(request);
         DatabaseClose(db);
         return;
        }
     }
//--- remove the query after use
   DatabaseFinalize(request);
```

The execution result is as follows:

```
Persons with salary > 15000:
0:  1 Paul 32 California 25000.0
1:  3 Teddy 23 Norway 20000.0
2:  4 Mark 25 Rich-Mond  65000.0
```

Find the complete sample code in the DatabaseRead.mq5 file.

### Debugging SQL queries in MetaEditor

All functions for working with the database return the error code in case of an unsuccessful code. Working with them should not cause any
issues if you follow four simple rules:

1. all query handles should be destroyed after use by DatabaseFinalize();
2. the database should be closed with DatabaseClose() before completion;
3. query execution results should be checked;
4. in case of an error, a query is destroyed first, while the database is closed afterwards.


The most difficult thing is to understand what the error is if the query has not been created. MetaEditor allows opening \*.sqlite files and
work with them using SQL queries. Let's see how this is done using the company.sqlite file as an example:

> 1\. Open the company.sqlite file in the common terminal folder.
>
> 2. After opening the database, we can see the COMPANY table in the Navigator. Double-click on it.
>
> 3\. The "SELECT \* FROM COMPANY" query is automatically created in the status bar.
>
> 4\. The query is executed automatically. It can also be executed by pressing F9 or clicking Execute.
>
> 5\. See the query execution result.
>
> 6\. If something is wrong, the errors are displayed in the editor’s Journal.

![](https://c.mql5.com/2/38/2020-02-14_16h51_51.gif)

SQL queries allow obtaining statistics on table fields, for example, the sum and the average. Let's make the queries and check if they work.

![Working with queries in MetaEditor](https://c.mql5.com/2/38/Check_SQL_request__1.png)

Now we can implement these queries in the MQL5 code:

```
   Print("Some statistics:");
//--- prepare a new query about the sum of salaries
   request=DatabasePrepare(db, "SELECT SUM(SALARY) FROM COMPANY");
   if(request==INVALID_HANDLE)
     {
      Print("DB: ", filename, " request failed with code ", GetLastError());
      DatabaseClose(db);
      return;
     }
   while(DatabaseRead(request))
     {
      double total_salary;
      DatabaseColumnDouble(request, 0, total_salary);
      Print("Total salary=", total_salary);
     }
//--- remove the query after use
   DatabaseFinalize(request);

//--- prepare a new query about the average salary
   request=DatabasePrepare(db, "SELECT AVG(SALARY) FROM COMPANY");
   if(request==INVALID_HANDLE)
     {
      Print("DB: ", filename, " request failed with code ", GetLastError());
      ResetLastError();
      DatabaseClose(db);
      return;
     }
   while(DatabaseRead(request))
     {
      double aver_salary;
      DatabaseColumnDouble(request, 0, aver_salary);
      Print("Average salary=", aver_salary);
     }
//--- remove the query after use
   DatabaseFinalize(request);
```

Compare the execution results:

```
Some statistics:
Total salary=125000.0
Average salary=31250.0
```

### Auto reading of query results into the structure using DatabaseReadBind()

The DatabaseRead() function allows going through all the query result entries and obtain full data on each column in the resulting table:

- [DatabaseColumnName](https://www.mql5.com/en/docs/database/databasecolumnname) — name,
- [DatabaseColumnType](https://www.mql5.com/en/docs/database/databasecolumntype)
— data type,
- [DatabaseColumnSize](https://www.mql5.com/en/docs/database/databasecolumnsize) — data size in bytes,
- [DatabaseColumnText](https://www.mql5.com/en/docs/database/databasecolumntext) — read as a text,
- [DatabaseColumnInteger](https://www.mql5.com/en/docs/database/databasecolumninteger) — get int type value,
- [DatabaseColumnLong](https://www.mql5.com/en/docs/database/databasecolumnlong) — get long type value,
- [DatabaseColumnDouble](https://www.mql5.com/en/docs/database/databasecolumndouble) — get double
type value,

- [DatabaseColumnBlob](https://www.mql5.com/en/docs/database/databasecolumnblob) — get data array.

These functions allow working with any query results in a unified manner. However, this benefit is counterbalanced by an excessive code. If the
structure of query results is known in advance, it is better to use the [DatabaseReadBind()](https://www.mql5.com/en/docs/database/databasereadbind "DatabaseReadBind")
function allowing you to immediately read the entire entry into the structure. We can redo the previous example the following way — first, declare the
Person structure:

```
struct Person
  {
   int               id;
   string            name;
   int               age;
   string            address;
   double            salary;
  };
```

Next, each entry is read from the query results using DatabaseReadBind(request, person):

```
//--- display obtained query results
   Person person;
   Print("Persons with salary > 15000:");
   for(int i=0; DatabaseReadBind(request, person); i++)
      Print(i, ":  ", person.id, " ", person.name, " ", person.age, " ", person.address, " ", person.salary);
//--- remove the query after use
   DatabaseFinalize(request);
```

This allows us to obtain the values of all fields from the current entry right away with no need to read them separately.

### Accelerating transactions by wrapping them into  DatabaseTransactionBegin()/DatabaseTransactionCommit()

When working with a table, it may be necessary to use the INSERT, UPDATE or DELETE commands en masse. The best way to do this is using
transactions. When conducting transactions, the database is first blocked ( [DatabaseTransactionBegin](https://www.mql5.com/en/docs/database/databasetransactionbegin)).
The bulk change commands are then performed and saved ( [DatabaseTransactionCommit](https://www.mql5.com/en/docs/database/databasetransactioncommit "DatabaseTransactionCommit"))
or canceled in case of an error ( [DatabaseTransactionRollback](https://www.mql5.com/en/docs/database/databasetransactionrollback "DatabaseTransactionRollback")).

The [DatabasePrepare](https://www.mql5.com/en/docs/database/databaseprepare "DatabasePrepare") function description
features an example of using transactions:

```
//--- auxiliary variables
   ulong    deal_ticket;         // deal ticket
   long     order_ticket;        // a ticket of an order a deal was executed by
   long     position_ticket;     // ID of a position a deal belongs to
   datetime time;                // deal execution time
   long     type ;               // deal type
   long     entry ;              // deal direction
   string   symbol;              // a symbol a deal was executed for
   double   volume;              // operation volume
   double   price;               // price
   double   profit;              // financial result
   double   swap;                // swap
   double   commission;          // commission
   long     magic;               // Magic number (Expert Advisor ID)
   long     reason;              // deal execution reason or source
//--- go through all deals and add them to the database
   bool failed=false;
   int deals=HistoryDealsTotal();
// --- lock the database before executing transactions
   DatabaseTransactionBegin(database);
   for(int i=0; i<deals; i++)
     {
      deal_ticket=    HistoryDealGetTicket(i);
      order_ticket=   HistoryDealGetInteger(deal_ticket, DEAL_ORDER);
      position_ticket=HistoryDealGetInteger(deal_ticket, DEAL_POSITION_ID);
      time= (datetime)HistoryDealGetInteger(deal_ticket, DEAL_TIME);
      type=           HistoryDealGetInteger(deal_ticket, DEAL_TYPE);
      entry=          HistoryDealGetInteger(deal_ticket, DEAL_ENTRY);
      symbol=         HistoryDealGetString(deal_ticket, DEAL_SYMBOL);
      volume=         HistoryDealGetDouble(deal_ticket, DEAL_VOLUME);
      price=          HistoryDealGetDouble(deal_ticket, DEAL_PRICE);
      profit=         HistoryDealGetDouble(deal_ticket, DEAL_PROFIT);
      swap=           HistoryDealGetDouble(deal_ticket, DEAL_SWAP);
      commission=     HistoryDealGetDouble(deal_ticket, DEAL_COMMISSION);
      magic=          HistoryDealGetInteger(deal_ticket, DEAL_MAGIC);
      reason=         HistoryDealGetInteger(deal_ticket, DEAL_REASON);
      //--- add each deal to the table using the following query
      string request_text=StringFormat("INSERT INTO DEALS (ID,ORDER_ID,POSITION_ID,TIME,TYPE,ENTRY,SYMBOL,VOLUME,PRICE,PROFIT,SWAP,COMMISSION,MAGIC,REASON)"
                                       "VALUES (%d, %d, %d, %d, %d, %d, '%s', %G, %G, %G, %G, %G, %d, %d)",
                                       deal_ticket, order_ticket, position_ticket, time, type, entry, symbol, volume, price, profit, swap, commission, magic, reason);
      if(!DatabaseExecute(database, request_text))
        {
         PrintFormat("%s: failed to insert deal #%d with code %d", __FUNCTION__, deal_ticket, GetLastError());
         PrintFormat("i=%d: deal #%d  %s", i, deal_ticket, symbol);
         failed=true;
         break;
        }
     }
//--- check for transaction execution errors
   if(failed)
     {
      //--- roll back all transactions and unlock the database
      DatabaseTransactionRollback(database);
      PrintFormat("%s: DatabaseExecute() failed with code %d", __FUNCTION__, GetLastError());
      return(false);
     }
//--- all transactions have been performed successfully - record changes and unlock the database
   DatabaseTransactionCommit(database);
```

Transactions allow accelerating bulk table operations hundreds of times as shown in the [DatabaseTransactionBegin](https://www.mql5.com/en/docs/database/databasetransactionbegin "DatabaseTransactionBegin")
example:

```
Result:
   Deals in the trading history: 2737
   Transations WITH    DatabaseTransactionBegin/DatabaseTransactionCommit: time=48.5 milliseconds
   Transations WITHOUT DatabaseTransactionBegin/DatabaseTransactionCommit: time=25818.9 milliseconds
   Use of DatabaseTransactionBegin/DatabaseTransactionCommit provided acceleration by 532.8 times
```

### Handling trading history deals

The power of SQL queries lies in the fact that you can easily sort, select and modify source data without writing code. Let's continue
analyzing the example from the [DatabasePrepare](https://www.mql5.com/en/docs/database/databaseprepare "DatabasePrepare")
function description showing how to obtain trades from deals via a singe query. A trade features data on position entry/exit dates and
prices, as well as symbol, direction and volume info. If we have a look at the [deal \\
structure](https://www.mql5.com/en/docs/constants/tradingconstants/dealproperties), we can see that entry/exit deals are linked by the common position ID. Thus, if we have a simple trading system on a hedging
account, we can easily combine two deals into a single trade. This is done using the following query:

```
//--- fill in the TRADES table using an SQL query based on DEALS table data
   ulong start=GetMicrosecondCount();
   if(DatabaseTableExists(db, "DEALS"))
     {
      //--- fill in the TRADES table
      if(!DatabaseExecute(db, "INSERT INTO TRADES(TIME_IN,TICKET,TYPE,VOLUME,SYMBOL,PRICE_IN,TIME_OUT,PRICE_OUT,COMMISSION,SWAP,PROFIT) "
                          "SELECT "
                          "   d1.time as time_in,"
                          "   d1.position_id as ticket,"
                          "   d1.type as type,"
                          "   d1.volume as volume,"
                          "   d1.symbol as symbol,"
                          "   d1.price as price_in,"
                          "   d2.time as time_out,"
                          "   d2.price as price_out,"
                          "   d1.commission+d2.commission as commission,"
                          "   d2.swap as swap,"
                          "   d2.profit as profit "
                          "FROM DEALS d1 "
                          "INNER JOIN DEALS d2 ON d1.position_id=d2.position_id "
                          "WHERE d1.entry=0 AND d2.entry=1"))
        {
         Print("DB: fillng the TRADES table failed with code ", GetLastError());
         return;
        }
     }
   ulong transaction_time=GetMicrosecondCount()-start;
```

The existing DEALS table is used here. The entries are created out of the deals with the identical DEAL\_POSITION\_ID using internal
combination via INNER JOIN. The result of the example operation from [DatabasePrepare](https://www.mql5.com/en/docs/database/databaseprepare "DatabasePrepare")
on a trading account:

```
Result:
   Deals in the trading history: 2741
   The first 10 deals:
       [ticket] [order_ticket] [position_ticket]              [time] [type] [entry] [symbol] [volume]   [price]   [profit] [swap] [commission] [magic] [reason]
   [0] 34429573              0                 0 2019.09.05 22:39:59      2       0 ""        0.00000   0.00000 2000.00000 0.0000      0.00000       0        0
   [1] 34432127       51447238          51447238 2019.09.06 06:00:03      0       0 "USDCAD"  0.10000   1.32320    0.00000 0.0000     -0.16000     500        3
   [2] 34432128       51447239          51447239 2019.09.06 06:00:03      1       0 "USDCHF"  0.10000   0.98697    0.00000 0.0000     -0.16000     500        3
   [3] 34432450       51447565          51447565 2019.09.06 07:00:00      0       0 "EURUSD"  0.10000   1.10348    0.00000 0.0000     -0.18000     400        3
   [4] 34432456       51447571          51447571 2019.09.06 07:00:00      1       0 "AUDUSD"  0.10000   0.68203    0.00000 0.0000     -0.11000     400        3
   [5] 34432879       51448053          51448053 2019.09.06 08:00:00      1       0 "USDCHF"  0.10000   0.98701    0.00000 0.0000     -0.16000     600        3
   [6] 34432888       51448064          51448064 2019.09.06 08:00:00      0       0 "USDJPY"  0.10000 106.96200    0.00000 0.0000     -0.16000     600        3
   [7] 34435147       51450470          51450470 2019.09.06 10:30:00      1       0 "EURUSD"  0.10000   1.10399    0.00000 0.0000     -0.18000     100        3
   [8] 34435152       51450476          51450476 2019.09.06 10:30:00      0       0 "GBPUSD"  0.10000   1.23038    0.00000 0.0000     -0.20000     100        3
   [9] 34435154       51450479          51450479 2019.09.06 10:30:00      1       0 "EURJPY"  0.10000 118.12000    0.00000 0.0000     -0.18000     200        3

   The first 10 trades:
                 [time_in] [ticket] [type] [volume] [symbol] [price_in]          [time_out] [price_out] [commission]   [swap]  [profit]
   [0] 2019.09.06 06:00:03 51447238      0  0.10000 "USDCAD"    1.32320 2019.09.06 18:00:00     1.31761     -0.32000  0.00000 -42.43000
   [1] 2019.09.06 06:00:03 51447239      1  0.10000 "USDCHF"    0.98697 2019.09.06 18:00:00     0.98641     -0.32000  0.00000   5.68000
   [2] 2019.09.06 07:00:00 51447565      0  0.10000 "EURUSD"    1.10348 2019.09.09 03:30:00     1.10217     -0.36000 -1.31000 -13.10000
   [3] 2019.09.06 07:00:00 51447571      1  0.10000 "AUDUSD"    0.68203 2019.09.09 03:30:00     0.68419     -0.22000  0.03000 -21.60000
   [4] 2019.09.06 08:00:00 51448053      1  0.10000 "USDCHF"    0.98701 2019.09.06 18:00:01     0.98640     -0.32000  0.00000   6.18000
   [5] 2019.09.06 08:00:00 51448064      0  0.10000 "USDJPY"  106.96200 2019.09.06 18:00:01   106.77000     -0.32000  0.00000 -17.98000
   [6] 2019.09.06 10:30:00 51450470      1  0.10000 "EURUSD"    1.10399 2019.09.06 14:30:00     1.10242     -0.36000  0.00000  15.70000
   [7] 2019.09.06 10:30:00 51450476      0  0.10000 "GBPUSD"    1.23038 2019.09.06 14:30:00     1.23040     -0.40000  0.00000   0.20000
   [8] 2019.09.06 10:30:00 51450479      1  0.10000 "EURJPY"  118.12000 2019.09.06 14:30:00   117.94100     -0.36000  0.00000  16.73000
   [9] 2019.09.06 10:30:00 51450480      0  0.10000 "GBPJPY"  131.65300 2019.09.06 14:30:01   131.62500     -0.40000  0.00000  -2.62000
   Filling the TRADES table took 12.51 milliseconds
```

Launch this script on your hedging account and compare the results with the positions in history. Previously, you may have had not enough
knowledge or time to code the loops to obtain such a result. Now you can do that by a single SQL query. You are able to view the script operation
result in MetaEditor. To do that, open the attached trades.sqlite file.

### Portfolio analysis by strategies

The results of the [DatabasePrepare](https://www.mql5.com/en/docs/database/databaseprepare "DatabasePrepare") script
operation shown above make it clear that trading is conducted on multiple currency pairs. Besides, the \[magic\] column shows the values from
100 to 600. This means that the trading account is managed by several strategies each of them having its own Magic Number to identify its
deals.

An SQL query allows us to analyze trading in context of _magic_ values:

```
//--- get trading statistics for Expert Advisors by Magic Number
   request=DatabasePrepare(db, "SELECT r.*,"
                           "   (case when r.trades != 0 then (r.gross_profit+r.gross_loss)/r.trades else 0 end) as expected_payoff,"
                           "   (case when r.trades != 0 then r.win_trades*100.0/r.trades else 0 end) as win_percent,"
                           "   (case when r.trades != 0 then r.loss_trades*100.0/r.trades else 0 end) as loss_percent,"
                           "   r.gross_profit/r.win_trades as average_profit,"
                           "   r.gross_loss/r.loss_trades as average_loss,"
                           "   (case when r.gross_loss!=0.0 then r.gross_profit/(-r.gross_loss) else 0 end) as profit_factor "
                           "FROM "
                           "   ("
                           "   SELECT MAGIC,"
                           "   sum(case when entry =1 then 1 else 0 end) as trades,"
                           "   sum(case when profit > 0 then profit else 0 end) as gross_profit,"
                           "   sum(case when profit < 0 then profit else 0 end) as gross_loss,"
                           "   sum(swap) as total_swap,"
                           "   sum(commission) as total_commission,"
                           "   sum(profit) as total_profit,"
                           "   sum(profit+swap+commission) as net_profit,"
                           "   sum(case when profit > 0 then 1 else 0 end) as win_trades,"
                           "   sum(case when profit < 0 then 1 else 0 end) as loss_trades "
                           "   FROM DEALS "
                           "   WHERE SYMBOL <> '' and SYMBOL is not NULL "
                           "   GROUP BY MAGIC"
                           "   ) as r");
```

Result:

```
Trade statistics by Magic Number
    [magic] [trades] [gross_profit] [gross_loss] [total_commission] [total_swap] [total_profit] [net_profit] [win_trades] [loss_trades] [expected_payoff] [win_percent] [loss_percent] [average_profit] [average_loss] [profit_factor]
[0]     100      242     2584.80000  -2110.00000          -33.36000    -93.53000      474.80000    347.91000          143            99           1.96198      59.09091       40.90909         18.07552      -21.31313         1.22502
[1]     200      254     3021.92000  -2834.50000          -29.45000    -98.22000      187.42000     59.75000          140           114           0.73787      55.11811       44.88189         21.58514      -24.86404         1.06612
[2]     300      250     2489.08000  -2381.57000          -34.37000    -96.58000      107.51000    -23.44000          134           116           0.43004      53.60000       46.40000         18.57522      -20.53078         1.04514
[3]     400      224     1272.50000  -1283.00000          -24.43000    -64.80000      -10.50000    -99.73000          131            93          -0.04687      58.48214       41.51786          9.71374      -13.79570         0.99182
[4]     500      198     1141.23000  -1051.91000          -27.66000    -63.36000       89.32000     -1.70000          116            82           0.45111      58.58586       41.41414          9.83819      -12.82817         1.08491
[5]     600      214     1317.10000  -1396.03000          -34.12000    -68.48000      -78.93000   -181.53000          116            98          -0.36883      54.20561       45.79439         11.35431      -14.24520         0.94346
```

4 out of 6 strategies have turned out to be profitable. We have received statistical values for each strategy:

- trades — number of trades by strategy,
- gross\_profit — total profit by strategy (the sum of all positive _profit_ values),
- gross\_loss — total loss by strategy (the sum of all negative _profit_ values),

- total\_commission — sum of all commissions by strategy trades,
- total\_swap — sum of all swaps by strategy trades,
- total\_profit — _gross\_profit_ and _gross\_loss_ sum,
- net\_profit — sum ( _gross\_profit_ \+ _gross\_loss_ \+ _total\_commission_ \+ _total\_swap_),
- win\_trades — number of trades where _profit_ >0,
- loss\_trades — number of trades where _profit_ <0,
- expected\_payoff — expected payoff for the trade excluding swaps and commissions = _net\_profit_/ _trades_,


- win\_percent — percentage of winning trades,
- loss\_percent — percentage of losing trades,
- average\_profit — average win = _gross\_profit_/ _win\_trades_,
- average\_loss — average loss = gross\_loss /loss\_trades,
- profit\_factor — Profit factor = _gross\_profit_/ _gross\_loss_.


Statistics for calculating profit and loss does not consider swaps and commissions accrued on the position. This allows you to see the net costs. It
may turn out that a strategy yields a small profit but is generally unprofitable due to swaps and commissions.

### Analyzing deals by symbols

We are able to analyze trading by symbols. To do this, make the following query:

```
//--- get trading statistics per symbols
   int request=DatabasePrepare(db, "SELECT r.*,"
                               "   (case when r.trades != 0 then (r.gross_profit+r.gross_loss)/r.trades else 0 end) as expected_payoff,"
                               "   (case when r.trades != 0 then r.win_trades*100.0/r.trades else 0 end) as win_percent,"
                               "   (case when r.trades != 0 then r.loss_trades*100.0/r.trades else 0 end) as loss_percent,"
                               "   r.gross_profit/r.win_trades as average_profit,"
                               "   r.gross_loss/r.loss_trades as average_loss,"
                               "   (case when r.gross_loss!=0.0 then r.gross_profit/(-r.gross_loss) else 0 end) as profit_factor "
                               "FROM "
                               "   ("
                               "   SELECT SYMBOL,"
                               "   sum(case when entry =1 then 1 else 0 end) as trades,"
                               "   sum(case when profit > 0 then profit else 0 end) as gross_profit,"
                               "   sum(case when profit < 0 then profit else 0 end) as gross_loss,"
                               "   sum(swap) as total_swap,"
                               "   sum(commission) as total_commission,"
                               "   sum(profit) as total_profit,"
                               "   sum(profit+swap+commission) as net_profit,"
                               "   sum(case when profit > 0 then 1 else 0 end) as win_trades,"
                               "   sum(case when profit < 0 then 1 else 0 end) as loss_trades "
                               "   FROM DEALS "
                               "   WHERE SYMBOL <> '' and SYMBOL is not NULL "
                               "   GROUP BY SYMBOL"
                               "   ) as r");
```

Result:

```
Trade statistics by Symbol
      [name] [trades] [gross_profit] [gross_loss] [total_commission] [total_swap] [total_profit] [net_profit] [win_trades] [loss_trades] [expected_payoff] [win_percent] [loss_percent] [average_profit] [average_loss] [profit_factor]
[0] "AUDUSD"      112      503.20000   -568.00000           -8.83000    -24.64000      -64.80000    -98.27000           70            42          -0.57857      62.50000       37.50000          7.18857      -13.52381         0.88592
[1] "EURCHF"      125      607.71000   -956.85000          -11.77000    -45.02000     -349.14000   -405.93000           54            71          -2.79312      43.20000       56.80000         11.25389      -13.47676         0.63512
[2] "EURJPY"      127     1078.49000  -1057.83000          -10.61000    -45.76000       20.66000    -35.71000           64            63           0.16268      50.39370       49.60630         16.85141      -16.79095         1.01953
[3] "EURUSD"      233     1685.60000  -1386.80000          -41.00000    -83.76000      298.80000    174.04000          127           106           1.28240      54.50644       45.49356         13.27244      -13.08302         1.21546
[4] "GBPCHF"      125     1881.37000  -1424.72000          -22.60000    -51.56000      456.65000    382.49000           80            45           3.65320      64.00000       36.00000         23.51712      -31.66044         1.32052
[5] "GBPJPY"      127     1943.43000  -1776.67000          -18.84000    -52.46000      166.76000     95.46000           76            51           1.31307      59.84252       40.15748         25.57145      -34.83667         1.09386
[6] "GBPUSD"      121     1668.50000  -1438.20000           -7.96000    -49.93000      230.30000    172.41000           77            44           1.90331      63.63636       36.36364         21.66883      -32.68636         1.16013
[7] "USDCAD"       99      405.28000   -475.47000           -8.68000    -31.68000      -70.19000   -110.55000           51            48          -0.70899      51.51515       48.48485          7.94667       -9.90563         0.85238
[8] "USDCHF"      206     1588.32000  -1241.83000          -17.98000    -65.92000      346.49000    262.59000          131            75           1.68199      63.59223       36.40777         12.12458      -16.55773         1.27902
[9] "USDJPY"      107      464.73000   -730.64000          -35.12000    -34.24000     -265.91000   -335.27000           50            57          -2.48514      46.72897       53.27103          9.29460      -12.81825         0.63606
```

Statistics shows that the net profit was received on 5 out of 10 symbols (net\_profit>0), while the profit factor was positive on 6 out of 10 symbols
(profit\_factor>1). This is exactly the case when swaps and commissions make the strategy unprofitable on EURJPY.

### Analyzing deals by entry hours

Even if trading is performed on a single symbol and a single strategy is applied, analyzing deals by market entry hours may still be useful. This
is done by the following SQL query:

```
//--- get trading statistics by market entry hours
   request=DatabasePrepare(db, "SELECT r.*,"
                           "   (case when r.trades != 0 then (r.gross_profit+r.gross_loss)/r.trades else 0 end) as expected_payoff,"
                           "   (case when r.trades != 0 then r.win_trades*100.0/r.trades else 0 end) as win_percent,"
                           "   (case when r.trades != 0 then r.loss_trades*100.0/r.trades else 0 end) as loss_percent,"
                           "   r.gross_profit/r.win_trades as average_profit,"
                           "   r.gross_loss/r.loss_trades as average_loss,"
                           "   (case when r.gross_loss!=0.0 then r.gross_profit/(-r.gross_loss) else 0 end) as profit_factor "
                           "FROM "
                           "   ("
                           "   SELECT HOUR_IN,"
                           "   count() as trades,"
                           "   sum(volume) as volume,"
                           "   sum(case when profit > 0 then profit else 0 end) as gross_profit,"
                           "   sum(case when profit < 0 then profit else 0 end) as gross_loss,"
                           "   sum(profit) as net_profit,"
                           "   sum(case when profit > 0 then 1 else 0 end) as win_trades,"
                           "   sum(case when profit < 0 then 1 else 0 end) as loss_trades "
                           "   FROM TRADES "
                           "   WHERE SYMBOL <> '' and SYMBOL is not NULL "
                           "   GROUP BY HOUR_IN"
                           "   ) as r");
```

Result:

```
Trade statistics by entry hour
     [hour_in] [trades] [volume] [gross_profit] [gross_loss] [net_profit] [win_trades] [loss_trades] [expected_payoff] [win_percent] [loss_percent] [average_profit] [average_loss] [profit_factor]
[ 0]         0       50  5.00000      336.51000   -747.47000   -410.96000           21            29          -8.21920      42.00000       58.00000         16.02429      -25.77483         0.45020
[ 1]         1       20  2.00000      102.56000    -57.20000     45.36000           12             8           2.26800      60.00000       40.00000          8.54667       -7.15000         1.79301
[ 2]         2        6  0.60000       38.55000    -14.60000     23.95000            5             1           3.99167      83.33333       16.66667          7.71000      -14.60000         2.64041
[ 3]         3       38  3.80000      173.84000   -200.15000    -26.31000           22            16          -0.69237      57.89474       42.10526          7.90182      -12.50938         0.86855
[ 4]         4       60  6.00000      361.44000   -389.40000    -27.96000           27            33          -0.46600      45.00000       55.00000         13.38667      -11.80000         0.92820
[ 5]         5       32  3.20000      157.43000   -179.89000    -22.46000           20            12          -0.70187      62.50000       37.50000          7.87150      -14.99083         0.87515
[ 6]         6       18  1.80000       95.59000   -162.33000    -66.74000           11             7          -3.70778      61.11111       38.88889          8.69000      -23.19000         0.58886
[ 7]         7       14  1.40000       38.48000   -134.30000    -95.82000            9             5          -6.84429      64.28571       35.71429          4.27556      -26.86000         0.28652
[ 8]         8       42  4.20000      368.48000   -322.30000     46.18000           24            18           1.09952      57.14286       42.85714         15.35333      -17.90556         1.14328
[ 9]         9      118 11.80000     1121.62000   -875.21000    246.41000           72            46           2.08822      61.01695       38.98305         15.57806      -19.02630         1.28154
[10]        10      206 20.60000     2280.59000  -2021.80000    258.79000          115            91           1.25626      55.82524       44.17476         19.83122      -22.21758         1.12800
[11]        11      138 13.80000     1377.02000   -994.18000    382.84000           84            54           2.77420      60.86957       39.13043         16.39310      -18.41074         1.38508
[12]        12      152 15.20000     1247.56000  -1463.80000   -216.24000           84            68          -1.42263      55.26316       44.73684         14.85190      -21.52647         0.85227
[13]        13       64  6.40000      778.27000   -516.22000    262.05000           36            28           4.09453      56.25000       43.75000         21.61861      -18.43643         1.50763
[14]        14       62  6.20000      536.93000   -427.47000    109.46000           38            24           1.76548      61.29032       38.70968         14.12974      -17.81125         1.25606
[15]        15       50  5.00000      699.92000   -413.00000    286.92000           28            22           5.73840      56.00000       44.00000         24.99714      -18.77273         1.69472
[16]        16       88  8.80000      778.55000   -514.00000    264.55000           51            37           3.00625      57.95455       42.04545         15.26569      -13.89189         1.51469
[17]        17       76  7.60000      533.92000  -1019.46000   -485.54000           44            32          -6.38868      57.89474       42.10526         12.13455      -31.85813         0.52373
[18]        18       52  5.20000      237.17000   -246.78000     -9.61000           24            28          -0.18481      46.15385       53.84615          9.88208       -8.81357         0.96106
[19]        19       52  5.20000      407.67000   -150.36000    257.31000           30            22           4.94827      57.69231       42.30769         13.58900       -6.83455         2.71129
[20]        20       18  1.80000       65.92000    -89.09000    -23.17000            9             9          -1.28722      50.00000       50.00000          7.32444       -9.89889         0.73993
[21]        21       10  1.00000       41.86000    -32.38000      9.48000            7             3           0.94800      70.00000       30.00000          5.98000      -10.79333         1.29277
[22]        22       14  1.40000       45.55000    -83.72000    -38.17000            6             8          -2.72643      42.85714       57.14286          7.59167      -10.46500         0.54408
[23]        23        2  0.20000        1.20000     -1.90000     -0.70000            1             1          -0.35000      50.00000       50.00000          1.20000       -1.90000         0.63158
```

It is clear that the largest number of trades is performed in the interval from 9 to 16 hours inclusive. Trading during other hours gives fewer
trades and is mostly unprofitable. Find the full source code with these three query types in the example for the [DatabaseExecute()](https://www.mql5.com/en/docs/database/databaseexecute "DatabaseExecute")
function.

### Convenient data output to the EA log in DatabasePrint()

In the previous examples, we had to read every entry into the structure and display entries one by one to display query results. It may often be
inconvenient to create a structure only to see the table or query result values. The [DatabasePrint()](https://www.mql5.com/en/docs/database/databaseprint)
function has been added for such cases:

```
long  DatabasePrint(
   int     database,          // database handle received in DatabaseOpen
   string  table_or_sql,      // a table or an SQL query
   uint    flags              // combination of flags
   );
```

It allows to print out not only an existing table but also query execution results that can be represented as a table. For example, display the
DEALS table values using the following query:

```
   DatabasePrint(db,"SELECT * from DEALS",0);
```

Result (the first 10 table rows are displayed):

```
  #|       ID ORDER_ID POSITION_ID       TIME TYPE ENTRY SYMBOL VOLUME   PRICE  PROFIT  SWAP COMMISSION MAGIC REASON
---+----------------------------------------------------------------------------------------------------------------
  1| 34429573        0           0 1567723199    2     0           0.0     0.0  2000.0   0.0        0.0     0      0
  2| 34432127 51447238    51447238 1567749603    0     0 USDCAD    0.1  1.3232     0.0   0.0      -0.16   500      3
  3| 34432128 51447239    51447239 1567749603    1     0 USDCHF    0.1 0.98697     0.0   0.0      -0.16   500      3
  4| 34432450 51447565    51447565 1567753200    0     0 EURUSD    0.1 1.10348     0.0   0.0      -0.18   400      3
  5| 34432456 51447571    51447571 1567753200    1     0 AUDUSD    0.1 0.68203     0.0   0.0      -0.11   400      3
  6| 34432879 51448053    51448053 1567756800    1     0 USDCHF    0.1 0.98701     0.0   0.0      -0.16   600      3
  7| 34432888 51448064    51448064 1567756800    0     0 USDJPY    0.1 106.962     0.0   0.0      -0.16   600      3
  8| 34435147 51450470    51450470 1567765800    1     0 EURUSD    0.1 1.10399     0.0   0.0      -0.18   100      3
  9| 34435152 51450476    51450476 1567765800    0     0 GBPUSD    0.1 1.23038     0.0   0.0       -0.2   100      3
 10| 34435154 51450479    51450479 1567765800    1     0 EURJPY    0.1  118.12     0.0   0.0      -0.18   200      3
```

### Data import/export

To simplify data import/export, the [DatabaseImport()](https://www.mql5.com/en/docs/database/databaseimport)
and [DatabaseExport()](https://www.mql5.com/en/docs/database/databaseexport) functions have been added. These
functions allow working with CSV files and data within ZIP archives.

DatabaseImport() imports data to a specified table. If no table with the specified name exists, it is created automatically. Names and field types in
the created table are also defined automatically based on the file data.

DatabaseExport() allows saving the table or query results to the file. If the query results are exported, the SQL query should begin with "SELECT" or
"select". In other words, the SQL query cannot alter the database status, otherwise DatabaseExport() fails with an error.

See the full description of the functions in the MQL5 Documentation.

### Saving optimization results to the database

The functions for working with databases can also be used for handling optimization results. Let's use the MACD Sample EA from the standard
delivery to illustrate the obtaining of test results using frames and saving the values of all optimization criteria into a single file
afterwards. To do this, create the CDatabaseFrames class, in which we define the OnTester() method for sending trading statistics:

```
//+------------------------------------------------------------------+
//| Tester function - sends trading statistics in a frame            |
//+------------------------------------------------------------------+
void               CDatabaseFrames::OnTester(const double OnTesterValue)
  {
//--- stats[] array to send data to a frame
   double stats[16];
//--- allocate separate variables for trade statistics to achieve more clarity
   int    trades=(int)TesterStatistics(STAT_TRADES);
   double win_trades_percent=0;
   if(trades>0)
      win_trades_percent=TesterStatistics(STAT_PROFIT_TRADES)*100./trades;
//--- fill in the array with test results
   stats[0]=trades;                                       // number of trades
   stats[1]=win_trades_percent;                           // percentage of profitable trades
   stats[2]=TesterStatistics(STAT_PROFIT);                // net profit
   stats[3]=TesterStatistics(STAT_GROSS_PROFIT);          // gross profit
   stats[4]=TesterStatistics(STAT_GROSS_LOSS);            // gross loss
   stats[5]=TesterStatistics(STAT_SHARPE_RATIO);          // Sharpe Ratio
   stats[6]=TesterStatistics(STAT_PROFIT_FACTOR);         // profit factor
   stats[7]=TesterStatistics(STAT_RECOVERY_FACTOR);       // recovery factor
   stats[8]=TesterStatistics(STAT_EXPECTED_PAYOFF);       // trade mathematical expectation
   stats[9]=OnTesterValue;                                // custom optimization criterion
//--- calculate built-in standard optimization criteria
   double balance=AccountInfoDouble(ACCOUNT_BALANCE);
   double balance_plus_profitfactor=0;
   if(TesterStatistics(STAT_GROSS_LOSS)!=0)
      balance_plus_profitfactor=balance*TesterStatistics(STAT_PROFIT_FACTOR);
   double balance_plus_expectedpayoff=balance*TesterStatistics(STAT_EXPECTED_PAYOFF);
   double balance_plus_dd=balance/TesterStatistics(STAT_EQUITYDD_PERCENT);
   double balance_plus_recoveryfactor=balance*TesterStatistics(STAT_RECOVERY_FACTOR);
   double balance_plus_sharpe=balance*TesterStatistics(STAT_SHARPE_RATIO);
//--- add the values of built-in optimization criteria
   stats[10]=balance;                                     // Balance
   stats[11]=balance_plus_profitfactor;                   // Balance+ProfitFactor
   stats[12]=balance_plus_expectedpayoff;                 // Balance+ExpectedPayoff
   stats[13]=balance_plus_dd;                             // Balance+EquityDrawdown
   stats[14]=balance_plus_recoveryfactor;                 // Balance+RecoveryFactor
   stats[15]=balance_plus_sharpe;                         // Balance+Sharpe
//--- create a data frame and send it to the terminal
   if(!FrameAdd(MQLInfoString(MQL_PROGRAM_NAME)+"_stats", STATS_FRAME, trades, stats))
      Print("Frame add error: ", GetLastError());
   else
      Print("Frame added, Ok");
  }
```

The second important method of the class is OnTesterDeinit(). After the optimization, it reads all obtained frames and saves statistics to
the database:

```
//+------------------------------------------------------------------+
//| TesterDeinit function - read data from frames                    |
//+------------------------------------------------------------------+
void               CDatabaseFrames::OnTesterDeinit(void)
  {
//--- take the EA name and optimization end time
   string filename=MQLInfoString(MQL_PROGRAM_NAME)+" "+TimeToString(TimeCurrent())+".sqlite";
   StringReplace(filename, ":", "."); // ":" character is not allowed in file names
//--- open/create the database in the common terminal folder
   int db=DatabaseOpen(filename, DATABASE_OPEN_READWRITE | DATABASE_OPEN_CREATE | DATABASE_OPEN_COMMON);
   if(db==INVALID_HANDLE)
     {
      Print("DB: ", filename, " open failed with code ", GetLastError());
      return;
     }
   else
      Print("DB: ", filename, " opened successful");
//--- create the PASSES table
   if(!DatabaseExecute(db, "CREATE TABLE PASSES("
                       "PASS               INT PRIMARY KEY NOT NULL,"
                       "TRADES             INT,"
                       "WIN_TRADES         INT,"
                       "PROFIT             REAL,"
                       "GROSS_PROFIT       REAL,"
                       "GROSS_LOSS         REAL,"
                       "SHARPE_RATIO       REAL,"
                       "PROFIT_FACTOR      REAL,"
                       "RECOVERY_FACTOR    REAL,"
                       "EXPECTED_PAYOFF    REAL,"
                       "ON_TESTER          REAL,"
                       "BL_BALANCE         REAL,"
                       "BL_PROFITFACTOR    REAL,"
                       "BL_EXPECTEDPAYOFF  REAL,"
                       "BL_DD              REAL,"
                       "BL_RECOVERYFACTOR  REAL,"
                       "BL_SHARPE          REAL );"))
     {
      Print("DB: ", filename, " create table failed with code ", GetLastError());
      DatabaseClose(db);
      return;
     }
//--- variables for reading frames
   string        name;
   ulong         pass;
   long          id;
   double        value;
   double        stats[];
//--- move the frame pointer to the beginning
   FrameFirst();
   FrameFilter("", STATS_FRAME); // select frames with trading statistics for further work
//--- variables to get statistics from the frame
   int trades;
   double win_trades_percent;
   double profit, gross_profit, gross_loss;
   double sharpe_ratio, profit_factor, recovery_factor, expected_payoff;
   double ontester_value;                              // custom optimization criterion
   double balance;                                     // Balance
   double balance_plus_profitfactor;                   // Balance+ProfitFactor
   double balance_plus_expectedpayoff;                 // Balance+ExpectedPayoff
   double balance_plus_dd;                             // Balance+EquityDrawdown
   double balance_plus_recoveryfactor;                 // Balance+RecoveryFactor
   double balance_plus_sharpe;                         // Balance+Sharpe
//--- block the database for the period of bulk transactions
   DatabaseTransactionBegin(db);
//--- go through frames and read data from them
   bool failed=false;
   while(FrameNext(pass, name, id, value, stats))
     {
      Print("Got pass #", pass);
      trades=(int)stats[0];
      win_trades_percent=stats[1];
      profit=stats[2];
      gross_profit=stats[3];
      gross_loss=stats[4];
      sharpe_ratio=stats[5];
      profit_factor=stats[6];
      recovery_factor=stats[7];
      expected_payoff=stats[8];
      stats[9];
      balance=stats[10];
      balance_plus_profitfactor=stats[11];
      balance_plus_expectedpayoff=stats[12];
      balance_plus_dd=stats[13];
      balance_plus_recoveryfactor=stats[14];
      balance_plus_sharpe=stats[15];
      PrintFormat("VALUES (%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%G,%.2f,%.2f,%2.f,%.2f,%.2f,%.2f,%.2f)",
                  pass, trades, win_trades_percent, profit, gross_profit, gross_loss, sharpe_ratio,
                  profit_factor, recovery_factor, expected_payoff, ontester_value, balance,
                  balance_plus_profitfactor, balance_plus_expectedpayoff, balance_plus_dd, balance_plus_recoveryfactor,
                  balance_plus_sharpe);
      //--- write data to the table
      string request=StringFormat("INSERT INTO PASSES (PASS,TRADES,WIN_TRADES, PROFIT,GROSS_PROFIT,GROSS_LOSS,"
                                  "SHARPE_RATIO,PROFIT_FACTOR,RECOVERY_FACTOR,EXPECTED_PAYOFF,ON_TESTER,"
                                  "BL_BALANCE,BL_PROFITFACTOR,BL_EXPECTEDPAYOFF,BL_DD,BL_RECOVERYFACTOR,BL_SHARPE) "
                                  "VALUES (%d, %d, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %G, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f)",
                                  pass, trades, win_trades_percent, profit, gross_profit, gross_loss, sharpe_ratio,
                                  profit_factor, recovery_factor, expected_payoff, ontester_value, balance,
                                  balance_plus_profitfactor, balance_plus_expectedpayoff, balance_plus_dd, balance_plus_recoveryfactor,
                                  balance_plus_sharpe);

      //--- execute a query to add a pass to the PASSES table
      if(!DatabaseExecute(db, request))
        {
         PrintFormat("Failed to insert pass %d with code %d", pass, GetLastError());
         failed=true;
         break;
        }
     }
//--- if an error occurred during a transaction, inform of that and complete the work
   if(failed)
     {
      Print("Transaction failed, error code=", GetLastError());
      DatabaseTransactionRollback(db);
      DatabaseClose(db);
      return;
     }
   else
     {
      DatabaseTransactionCommit(db);
      Print("Transaction done successful");
     }
//--- close the database
   if(db!=INVALID_HANDLE)
     {
      Print("Close database with handle=", db);
      DatabaseClose(db);
     }
```

In the MACD Sample EA, include the DatabaseFrames.mqh file and declare the CDatabaseFrames class variable:

```
#define MACD_MAGIC 1234502
//---
#include <Trade\Trade.mqh>
#include <Trade\SymbolInfo.mqh>
#include <Trade\PositionInfo.mqh>
#include <Trade\AccountInfo.mqh>
#include "DatabaseFrames.mqh"
...
CDatabaseFrames DB_Frames;
```

Next, add three functions at the end of the EA to be called only during optimization:

```
//+------------------------------------------------------------------+
//| TesterInit function                                              |
//+------------------------------------------------------------------+
int OnTesterInit()
  {
   return(DB_Frames.OnTesterInit());
  }
//+------------------------------------------------------------------+
//| TesterDeinit function                                            |
//+------------------------------------------------------------------+
void OnTesterDeinit()
  {
   DB_Frames.OnTesterDeinit();
  }
//+------------------------------------------------------------------+
//| Tester function                                                  |
//+------------------------------------------------------------------+
double OnTester()
  {
   double ret=0;
   //--- create a custom optimization criterion as the ratio of a net profit to a relative balance drawdown
   if(TesterStatistics(STAT_BALANCE_DDREL_PERCENT)!=0)
      ret=TesterStatistics(STAT_PROFIT)/TesterStatistics(STAT_BALANCE_DDREL_PERCENT);
   DB_Frames.OnTester(ret);
   return(ret);
  }
//+------------------------------------------------------------------+
```

Launch optimization and get the database file with trading statistics in the common terminal folder:

```
CDatabaseFrames::OnTesterInit: optimization launched at 15:53:27
DB: MACD Sample Database 2020.01.20 15.53.sqlite opened successful
Transaction done successful
Close database with handle=65537
Database stored in file 'MACD Sample Database 2020.01.20 15.53.sqlite'
```

The newly created database file can be opened in MetaEditor or used in another MQL5 application for further work.

![Working with the database in MetaEditor](https://c.mql5.com/2/38/Optimization_Results_Database.png)

Thus, you can prepare any data in the necessary form for further analysis or exchange with other traders. Find the source code, the ini file with
the optimization parameters and the execution result in the MACD.zip archive attached below.

### Optimizing query execution using indices

The best feature of SQL (in all its implementations, not just SQLite) is that it is a declarative language, not a procedural language. When
programming in SQL, you tell the system WHAT you want to compute, not HOW to compute it. The task of figuring out the 'how' is delegated to the
query planner subsystem within the SQL database engine.

For any given SQL statement, there might be hundreds or thousands of
different algorithms of performing the operation. All of these algorithms will get the correct answer, though some will run faster than
others. The query planner tries to pick the fastest and most efficient algorithm for each SQL statement.

Most of the time, the query planner in SQLite does a good job. However, the query planner needs indices to do its best. These indices should
normally be added by programmers. Sometimes, the query planner will make a suboptimal algorithm choice. In those cases, programmers may
want to provide additional hints to help the query planner do a better job.

**Lookup without indices**

Suppose that we have the DEALS table containing the specified 14 fields. Below are the first 10 entries of this table.

|     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rowid | ID | ORDER\_ID | POSITION\_ID | TIME | TYPE | ENTRY | SYMBOL | VOLUME | PRICE | PROFIT | SWAP | COMMISSION | MAGIC | REASON |
| 1 | 34429573 | 0 | 0 | 1567723199 | 2 | 0 |  | 0 | 0 | 2000 | 0 | 0 | 0 | 0 |
| 2 | 34432127 | 51447238 | 51447238 | 1567749603 | 0 | 0 | USDCAD | 0.1 | 1.3232 | 0 | 0 | -0.16 | 500 | 3 |
| 3 | 34432128 | 51447239 | 51447239 | 1567749603 | 1 | 0 | USDCHF | 0.1 | 0.98697 | 0 | 0 | -0.16 | 500 | 3 |
| 4 | 34432450 | 51447565 | 51447565 | 1567753200 | 0 | 0 | EURUSD | 0.1 | 1.10348 | 0 | 0 | -0.18 | 400 | 3 |
| 5 | 34432456 | 51447571 | 51447571 | 1567753200 | 1 | 0 | AUDUSD | 0.1 | 0.68203 | 0 | 0 | -0.11 | 400 | 3 |
| 6 | 34432879 | 51448053 | 51448053 | 1567756800 | 1 | 0 | USDCHF | 0.1 | 0.98701 | 0 | 0 | -0.16 | 600 | 3 |
| 7 | 34432888 | 51448064 | 51448064 | 1567756800 | 0 | 0 | USDJPY | 0.1 | 106.962 | 0 | 0 | -0.16 | 600 | 3 |
| 8 | 34435147 | 51450470 | 51450470 | 1567765800 | 1 | 0 | EURUSD | 0.1 | 1.10399 | 0 | 0 | -0.18 | 100 | 3 |
| 9 | 34435152 | 51450476 | 51450476 | 1567765800 | 0 | 0 | GBPUSD | 0.1 | 1.23038 | 0 | 0 | -0.2 | 100 | 3 |
| 10 | 34435154 | 51450479 | 51450479 | 1567765800 | 1 | 0 | EURJPY | 0.1 | 118.12 | 0 | 0 | -0.18 | 200 | 3 |

It features data from the [Deal \\
Properties](https://www.mql5.com/en/docs/constants/tradingconstants/dealproperties "Deal properties") section (except for DEAL\_TIME\_MSC, DEAL\_COMMENT and DEAL\_EXTERNAL\_ID) necessary for analyzing trading history. Apart
from the stored data, each table always features the **rowid** integer key followed by entry fields. _rowid_ key values are
created automatically and are unique within the table. They are increased when adding new entries. Deleting entries may cause numbering
gaps but table rows are always stored in _rowid_ ascending order.

If we need to find deals related to a certain position, for example, ID=51447571, we should write the following query:

```
SELECT * FROM deals WHERE position_id=51447571
```

In this case, a full table scan is performed — all rows are viewed and the POSITION\_ID is checked for equality to the value of 51447571 at each
row. Rows that satisfy this condition are displayed in the query execution results. If the table contains millions or tens of millions of
records, the search may take a long time. If we did the search by the rowid=5 condition rather than position\_id=51447571, the search time
would be reduced by thousands or even millions of times (depending on the table size).

```
SELECT * FROM deals WHERE rowid=5
```

The query execution result would be the same since the row with rowid=5 stores position\_id=51447571. Acceleration is achieved due to the
fact that the _rowid_ values are sorted in ascending order, and the binary search is used to get the result. Unfortunately, the search by _rowid_
is not suitable for us since we are interested in entries having the necessary position\_id value.

**Lookup by index**

To make a query execution more time efficient, we need to add the POSITION\_ID field index using the following query:

```
 CREATE INDEX Idx1 ON deals(position_id)
```

In this case, a separate table with two columns is generated. The first column consists of POSITION\_ID values sorted in ascending order,
while the second column consists of _rowid_.

| POSITION\_ID | rowid |
| 0 | 1 |
| 51447238 | 2 |
| 51447239 | 3 |
| 51447565 | 4 |
| 51447571 | 5 |
| 51448053 | 6 |
| 51448064 | 7 |
| 51450470 | 8 |
| 51450476 | 9 |
| 51450479 | 10 |

The _rowid_ sequence may already be violated, although it is preserved in our example, since POSITION\_ID is increased as well when opening
a position by time.

Now that we have the POSITION\_ID field index, our query

```
SELECT * FROM deals WHERE position_id=51447571
```

is performed differently. First, a binary search in the Idx1 index is performed by the POSITION\_ID column and all _rowids_
matching the condition are found. The second binary search in the original DEALS table looks for all entries by the known _rowid_ values. Thus, a
single full scan of the large table is now replaced with two consecutive lookups — first, by index and then by table row numbers. This allows
reducing the execution time of such queries by thousands or more times in case of a large number of rows in the table.

**General rule**: If some of the table fields are often used for searching/comparing/sorting, it is recommended to create
indices by these fields.

The DEALS table also features SYMBOL, MAGIC (EA ID) and ENTRY (entry direction) fields. If you need to take samples in these fields, then it is
reasonable to create the appropriate indices. For example:

```
CREATE INDEX Idx2 ON deals(symbol)
CREATE INDEX Idx3 ON deals(magic)
CREATE INDEX Idx4 ON deals(entry)
```

Keep in mind that creating indices requires additional memory, and each entry addition/deletion entails re-indexing. You can also create
multi-indices based on multiple fields. For example, if we want to select all deals performed by the EA having MAGIC= 500 on USDCAD, we can
create the following query:

```
SELECT * FROM deals WHERE magic=500 AND symbol='USDCAD'
```

In this case, you can create a multi-index by MAGIC and SYMBOL fields

```
CREATE INDEX Idx5 ON deals(magic, symbol)
```

and the following index table is created (the first 10 rows are shown schematically)

| MAGIC | SYMBOL | rowid |
| 100 | EURUSD | 4 |
| 100 | EURUSD | 10 |
| 100 | EURUSD | 20 |
| 100 | GBPUSD | 5 |
| 100 | GBPUSD | 11 |
| 200 | EURJPY | 6 |
| 200 | EURJPY | 12 |
| 200 | EURJPY | 22 |
| 200 | GBPJPY | 7 |
| 200 | GBPJPY | 13 |

In the newly created multi-index, the entries are first sorted in blocks by MAGIC and then – by SYMBOL field. Therefore, in case of AND
queries, the search in the index is first performed by the MAGIC column. The value of the SYMBOL column is checked afterwards. If both
conditions are met, _rowid_ is added to the result set to be used in the original table search. Generally speaking, such a multi-index
is no longer suitable for queries where SYMBOL is checked first

```
SELECT * FROM deals WHERE  symbol='USDCAD' AND magic=500
```

Although the query planner understands how to act correctly and performs the search in the right order in such cases, it would still be unwise to hope
that it will always automatically fix your errors in table and query design.

**OR queries**

Multi-indices are only suitable for AND queries. For example, suppose that we want to find all deals performed by the EA having MAGIC=100 or on EURUSD:

```
SELECT * FROM deals WHERE magic=100 OR symbol='EURUSD'
```

In this case, two separate lookups are implemented. All found rowids are then combined into a common selection for the final search by row
numbers in the source table.

```
SELECT * FROM deals WHERE magic=100
SELECT * FROM deals WHERE symbol='EURUSD'
```

But even in this case, it is necessary that both fields of the OR query have indices, otherwise the search will cause the full table scan.

**Sorting**

To speed up sorting, it is also recommended to have an index by the fields used to arrange query results. For example, suppose that we need to
select all deals on EURUSD sorted by deal time:

```
SELECT * FROM deals symbol='EURUSD' ORDER BY time
```

In this case, you should consider creating an index by TIME field. The need for indices depends on the table size. If the table has few entries,
then indexing can hardly save any time.

Here we examined only the very basics of query optimization. For better understanding, we recommend that you study the subject starting from
the [Query \\
Planning](https://www.mql5.com/go?link=https://www.sqlite.org/queryplanner.html "https://www.sqlite.org/queryplanner.html") section on the SQLite developers' website.

### Integrating database handling into MetaEditor

The MetaTrader 5 platform is in constant development. We have added the native support for SQL queries to the MQL5 language and integrated the
new functionality for handling databases into MetaEditor, including creating a database, inserting and deleting data and performing
bulk transactions. Creating a database is standard and involves MQL5 Wizard. Simply specify file and table names, and add all the necessary
fields indicating the type.

![Creating a database in MQL Wizard](https://c.mql5.com/2/38/MQL5_Wizard_Database__1.png)

Next, you can fill the table with data, perform a search and selection, introduce SQL queries, etc. Thus, you can work with databases not only
from MQL5 programs, but also manually. No third-party browsers are needed for that.

The introduction of SQLite in MetaTrader opens up new opportunities for traders in terms of handling large data amounts both
programmatically and manually. We have done our best to make sure that these functions are most convenient to use and are on equal footing
with other solutions in terms of speed. Study and apply the language of SQL queries in your work.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/7463](https://www.mql5.com/ru/articles/7463)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/7463.zip "Download all attachments in the single ZIP archive")

[SqLiteTest.zip](https://www.mql5.com/en/articles/download/7463/sqlitetest.zip "Download SqLiteTest.zip")(2708.45 KB)

[trades.sqlite](https://www.mql5.com/en/articles/download/7463/trades.sqlite "Download trades.sqlite")(340 KB)

[MACD.zip](https://www.mql5.com/en/articles/download/7463/macd.zip "Download MACD.zip")(8.27 KB)

[DatabaseRead.mq5](https://www.mql5.com/en/articles/download/7463/databaseread.mq5 "Download DatabaseRead.mq5")(10.11 KB)

[DatabaseReadBind.mq5](https://www.mql5.com/en/articles/download/7463/databasereadbind.mq5 "Download DatabaseReadBind.mq5")(9.62 KB)

[DatabaseTransactionBegin.mq5](https://www.mql5.com/en/articles/download/7463/databasetransactionbegin.mq5 "Download DatabaseTransactionBegin.mq5")(18.3 KB)

[DatabasePrepare.mq5](https://www.mql5.com/en/articles/download/7463/databaseprepare.mq5 "Download DatabasePrepare.mq5")(35.02 KB)

[DatabaseExecute.mq5](https://www.mql5.com/en/articles/download/7463/databaseexecute.mq5 "Download DatabaseExecute.mq5")(64.83 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

#### Other articles by this author

- [Getting Started with MQL5 Algo Forge](https://www.mql5.com/en/articles/18518)
- [Installing MetaTrader 5 and Other MetaQuotes Apps on HarmonyOS NEXT](https://www.mql5.com/en/articles/18612)
- [MetaTrader 5 on macOS](https://www.mql5.com/en/articles/619)
- [How to earn money by fulfilling traders' orders in the Freelance service](https://www.mql5.com/en/articles/1019)
- [MetaTrader 4 on macOS](https://www.mql5.com/en/articles/1356)
- [Working with ONNX models in float16 and float8 formats](https://www.mql5.com/en/articles/14330)
- [Regression models of the Scikit-learn Library and their export to ONNX](https://www.mql5.com/en/articles/13538)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/333176)**
(112)


![MetaQuotes](https://c.mql5.com/avatar/2010/1/4B5DE8B4-9045.jpg)

**[MetaQuotes](https://www.mql5.com/en/users/metaquotes)**
\|
7 Dec 2022 at 05:44

**Anatoli Kazharski [#](https://www.mql5.com/ru/forum/332994/page10#comment_43578188):**

In **MetaEditor**, the maximum number of database table columns to show is only **23.**

Is it possible to remove the limitation?

Fixed in beta 3531 with increasing columns to 64.


![DDFedor](https://c.mql5.com/avatar/2010/11/4CE0E2D8-74F3.jpg)

**[DDFedor](https://www.mql5.com/en/users/ddfedor)**
\|
17 Dec 2022 at 22:35

At some point somewhere there was an answer about single and [double quotes](https://www.mql5.com/en/docs/basis/types/integer/symbolconstants "MQL5 Documentation: Symbolic constants"). Not verbatim, but close to the text - "has been working for a long time with double quotes". At the moment, coming back to working with tables, an attempt to write text in double quotes into a table fails. However, by enclosing the text in single and then double quotes, the write completes successfully. What is the sound rule of single and double quotes for writing text to a table?

Options and outcomes :

successful -

```
         AddTable_TstDate(i,
                          iTime(Symbol(),PERIOD_CURRENT,i),
                          iHigh(Symbol(),PERIOD_CURRENT,i),
                          iTime(Symbol(),PERIOD_CURRENT,i),
                          iLow(Symbol(),PERIOD_CURRENT,i),
                          IntegerToString(iTime(Symbol(),PERIOD_CURRENT,i)),
                          1121,
                          "'string_no_error'");
```

not successful -

```
         AddTable_TstDate(i,
                          iTime(Symbol(),PERIOD_CURRENT,i),
                          iHigh(Symbol(),PERIOD_CURRENT,i),
                          iTime(Symbol(),PERIOD_CURRENT,i),
                          iLow(Symbol(),PERIOD_CURRENT,i),
                          IntegerToString(iTime(Symbol(),PERIOD_CURRENT,i)),
                          1121,
                          "string_error");
```

I would like to point out, translating an integer to a string does not produce an error when writing to a table.

![Denis Kirichenko](https://c.mql5.com/avatar/2019/5/5CEDB8D2-7CB7.jpg)

**[Denis Kirichenko](https://www.mql5.com/en/users/denkir)**
\|
29 Dec 2022 at 21:59

Dear developers, please tell me why it is not possible to attach an existing database from a file (ATTACH DATABASE)?

Then how is it possible to attach a base from the RAM...?

I have attached the code.

The **create\_databases.mq5** script creates a database. The **attach\_mem\_db.mq5** script attaches the database from the RAM. But the **attach\_other\_db.mq5** script  fails to attach an existing database.

![Faisal Mahmood](https://c.mql5.com/avatar/2022/2/621BC4C1-E102.jpeg)

**[Faisal Mahmood](https://www.mql5.com/en/users/xbotuk)**
\|
24 Jan 2023 at 19:46

At the moment the MT5's [SQLite](https://www.mql5.com/en/articles/7463 "Article: SQLite: Native handling of SQL databases in MQL5 ") has a max worker threads set to 8 in the binaries and cannot be increased. How to get the MT5 developers to have this increased in the next update? Ideally the max limit should be a high number like 50, and we can set the threads in the code.

![](https://c.mql5.com/avatar/avatar_na2.png)

**\[Deleted\]**
\|
12 Jan 2024 at 23:00

[![](https://c.mql5.com/3/426/2506030584234__1.png)](https://c.mql5.com/3/426/2506030584234.png "https://c.mql5.com/3/426/2506030584234.png")

Insert Delete Commit Rollback all are greyed out. May i know how to enable these when editing db

![Library for easy and quick development of MetaTrader programs (part XXVIII): Closure, removal and modification of pending trading requests](https://c.mql5.com/2/37/MQL5-avatar-doeasy__16.png)[Library for easy and quick development of MetaTrader programs (part XXVIII): Closure, removal and modification of pending trading requests](https://www.mql5.com/en/articles/7438)

This is the third article about the concept of pending requests. We are going to complete the tests of pending trading requests by creating the methods for closing positions, removing pending orders and modifying position and pending order parameters.

![Continuous Walk-Forward Optimization (Part 2): Mechanism for creating an optimization report for any robot](https://c.mql5.com/2/37/MQL5-avatar-continuous_optimization__1.png)[Continuous Walk-Forward Optimization (Part 2): Mechanism for creating an optimization report for any robot](https://www.mql5.com/en/articles/7452)

The first article within the Walk-Through Optimization series described the creation of a DLL to be used in our auto optimizer. This continuation is entirely devoted to the MQL5 language.

![Neural Networks Made Easy](https://c.mql5.com/2/48/Neural_networks_made_easy_001.png)[Neural Networks Made Easy](https://www.mql5.com/en/articles/7447)

Artificial intelligence is often associated with something fantastically complex and incomprehensible. At the same time, artificial intelligence is increasingly mentioned in everyday life. News about achievements related to the use of neural networks often appear in different media. The purpose of this article is to show that anyone can easily create a neural network and use the AI achievements in trading.

![Library for easy and quick development of MetaTrader programs (part XXVII): Working with trading requests - placing pending orders](https://c.mql5.com/2/37/MQL5-avatar-doeasy__15.png)[Library for easy and quick development of MetaTrader programs (part XXVII): Working with trading requests - placing pending orders](https://www.mql5.com/en/articles/7418)

In this article, we will continue the development of trading requests, implement placing pending orders and eliminate detected shortcomings of the trading class operation.

[![](https://www.mql5.com/ff/si/m0dtjf9x3brdz07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fmarket%2Fmt5%2Fexpert%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dtop.experts%26utm_content%3Dbuy.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=widauvjabtsckwovwaperzkotrcrttvb&s=25ef75d39331f608a319410bf27ff02c1bd7986622ecc1eec8968a650f044731&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=qejaenyrlkfsqpjwfkxtjkiddgrcdhid&ssn=1769181398804644216&ssn_dr=0&ssn_sr=0&fv_date=1769181398&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F7463&back_ref=https%3A%2F%2Fwww.google.com%2F&title=SQLite%3A%20Native%20handling%20of%20SQL%20databases%20in%20MQL5%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918139831934905&fz_uniq=5069302826976346837&sv=2552)

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