---
title: How to Access the MySQL Database from MQL5 (MQL4)
url: https://www.mql5.com/en/articles/932
categories: Integration
relevance_score: 3
scraped_at: 2026-01-23T21:19:30.972376
---

[![](https://www.mql5.com/ff/sh/5z040u47jcv59943z2/6c76c03a8b37e08b8655a1a085770b7a.jpg)\\
MetaTrader 5 for iOS and Android\\
\\
Fully featured platform for any devices and web browsers\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=ddonqpipxfqlnsvzlwuowsuwlejpyjxk&s=9daba65b69f40afc3c35f95b1f84ef5824d68c47f29ce96a6dc5b164a2727baa&uid=&ref=https://www.mql5.com/en/articles/932&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5071747402102156721)

MetaTrader 5 / Integration


### Introduction

The problem of interaction of MQL with databases is not new, however it's still relevant. Use of databases can greatly enhance the possibilities of MetaTrader: storage and analysis of the price history, copying trades from one trading platform to another, providing quotes/trades in real time, heavy analytical computations on the server side and/or using a schedule, monitoring and remote control of accounts using web technologies.

Anyway, there were many attempts to benefit from the combination of MQL and MySQL, some solutions are available in the CodeBase.

For example ["MySQL wrapper - library for MetaTrader 4"](https://www.mql5.com/en/code/8623) is the project, from which many programmers start their own developments with further additions. In my opinion, one of the disadvantages of this solution is allocation of special arrays for reading data from the database.

Another project ["MySQL logger 1 - EA for MetaTrader 4"](https://www.mql5.com/ru/code/9815) is highly specialized, it uses no wrapper to access the standard library libmysql.dll. Therefore it doesn't work in MetaTrader4 Build 600+, since the **char** character types have been replaced by **wchar\_t**, and the use of the **int** type instead of the **TMYSQL** structure pointer causes memory leaks in the project (the allocated memory cannot be controlled/freed).

Another interesting project is ["EAX\_Mysql - MySQL library - library for MetaTrader 5"](https://www.mql5.com/en/code/855). It's quite a good implementation. The list of disadvantages stated by the author imposes some restrictions on its use.

Anyone who ever needs to uses databases in their MQL projects has two options: either to develop their own solution and know every single part of it, or use/adapt any third-party solution, learn how to use it and detect all its defects that may hinder their project.

I faced such a necessity and the two options while developing a rather complex trading robot. Having searched through existing projects and studied a very large number of solutions, I realized that non of the found implementations could help bring my trading robot to the "professional level".

Moreover, there were also absurd solutions, for example: DML/DDL operations (insert/update/delete data, create/drop objects in database) were performed using the standard libmysql.dll, and data selection (SELECT) was actually implemented as a HTTP request (using inet.dll) to a PHP script located on the web server on the MySQL server side. The SQL queries were written in the PHP script.

In other words, to run the project, one needed to keep the following components available, configured and running: MySQL server, Apache/IIS web server, PHP/ASP scripts on the server side... A combination of quite a large number of technologies. Of course, in some circumstances this may be acceptable, but when the only task is to select data from the database - this is nonsense. In addition, supporting such a cumbersome solution is time-consuming.

Most of the solutions had no problems inserting data, creating objects and the like. The problem was data selection, as the data should be returned to the calling environment.

I thought using arrays for this purpose was impractical and inconvenient, simply because in the course of development/debugging/support of the main project, select queries to the database can be changed, while you should also control correct memory allocation for the arrays... Well, this can and must be avoided.

The hereinafter discussed MQL <-> MySql interfaced is based on a typical approach used in Oracle PL/SQL, MS SQL T-SQL, AdoDB - use of cursors. This interface was developed targeting the ease of programming and maintenance, plus a minimum of components. It is implemented as a DLL wrapper to the standard library libmysql.dll and a set of interface functions as an .mqh file.

### 1\. MQL <-> MySQL Interface

The interaction between the MetaTrader terminal (through MQL programs) can be implemented with the help of the below components:

![The scheme of MQL and MySQL interaction](https://c.mql5.com/2/11/n6dps.png)

1\. The interface library MQLMySQL.mqh. It is added to the project using the **#include** directory and can be modified to your taste.

It contains the directives for importing functions of the MQLMySQL.dll dynamic library, as well as functions for calling them and handling errors.

2\. The MQLMySQL.dll dynamic library. It is a wrapper to access the functionality of the standard library libmysql.dll.

Also, the MQLMySQL.dll library processes the results of operations and shared access to the database connections and cursors. It means that you can create and use multiple connections at a time (from one or more MQL programs), keep a few cursors open, with queries to one or more databases. Mutexes are used for separating access to shared resources.

3\. The standard dynamic library libmysql.dll is a native access driver. You can copy it from any MySql database distribution in C:\\Windows\\Sytem32 or <Terminal>\\MQL5\\Libraries (for MetaTrader 4 in <Terminal>\\MQL4\\Libraries).

In fact, it is responsible for sending queries to the database and retrieving the results.

Let's dwell on the main points, such as: opening/closing the connection, performing DML/DDL queries and data selection.

**1.1. Opening and Closing the Connection**

The MySqlConnect function has been implemented for opening connection with the MySQL database:

| **Type** | **Name** | **Parameters** | **Description** |
| --- | --- | --- | --- |
| **int** | MySqlConnect | This function implements connection with the database and returns a connection identifier. This ID will be required to query the database.<br>In case of a connection failure, the return value is "-1". For the error details, check the variables _**MySQLErrorNumber**_ and **_MySqlErrorDescription_**.<br>Typically, this function is called when handling the **_OnInit()_** event in the MQL program. |
| **string** pHost | The DNS name or IP address of the MySQL server |
| **string** pUser | Database user (for example, root) |
| **string** pPassword | The password of the database user |
| **string** pDatabase | The name of the database |
| **int** pPort | The TCP/IP port of the database (usually 3306) |
| **string** pSocket | The Unix socket (for the Unix based systems) |
| **int** pClientFlag | The combination of special flags (usually 0) |

The MySqlDisconnect interface function has been implemented for closing the connection:

| **Type** | **Name** | **Parameters** | **Description** |
| --- | --- | --- | --- |
| **void** | MySqlDisconnect | This function closes connection with the MySQL database. <br>Typically, this function is called when handling the **_OnDeinit()_** event in the MQL program. |
| **int** pConnection | Connection identifier |

It should be noted that the MySQL database can close the connection on its own in case of a hardware failure, network congestion or timeout (when no queries are sent to the database for a long time).

Often developers use the [OnTick()](https://www.mql5.com/en/docs/basis/function/events#ontick) event for writing data to the database. However, when weekend comes and the market is closed, the connection is still "hanging". In this case, MySQL will close its by timeout (the default is 8 hours).

And on Monday, when the market is open, errors are found in the project. Therefore it is strongly recommended to check the connection and/or reconnect to the database after a time interval smaller than the timeout specified in the settings of the MySQL server.

**1.2. Execution of DML/DDL Queries**

DML operations are used for data manipulations ( **D** ata **M** anipulation **L** anguage). Data manipulations include the following set of statements: INSERT, UPDATE and DELETE.

DDL operations are used for data definition ( **D** ata **D** efinition **L** anguage). This includes the creation (CREATE) of database objects (tables, views, stored procedures, triggers, etc.) and their modification (ALTER) and deletion (DROP).

It's not all DML/DDL statements, moreover, DCL ( **D** ata **C** ontrol **L** anguage) is used to separate data access, but we will not delve into the features of SQL. Any of these commands can be executed using the MySqlExecute interface function:

| **Type** | **Name** | **Parameters** | **Description** |
| --- | --- | --- | --- |
| **bool** | MySqlExecute | This function can be used for executing non-SELECT statements of SQL, after connection to the database has been successfully established (using the **MySqlConnect** function).<br>In case of successful command execution the function returns true, otherwise - false. For the error details, use the **_MySQLErrorNumber_** and **_MySqlErrorDescription_**. |
| **int** pConnection | Connection identifier |
| **string** pQuery | SQL Query |

As a SQL query, you can also use the USE command to select the database. I would like mention the use of multi-statement queries. It is a set of SQL commands separated by the character ";".

To enable the multi-statements mode, connection with the database should be opened with the CLIENT\_MULTI\_STATEMENTS flag:

```
...
int ClientFlag = CLIENT_MULTI_STATEMENTS; // Setting the multi-statements flag
int DB;

DB = MySqlConnect(Host, User, Password, Database, Port, Socket, ClientFlag); // Connection to the database

if (DB == -1)
   {
    // Handling the connection error
   }
...

// Preparing a SQL query to insert data (3 rows in one query)
string SQL;
SQL = "INSERT INTO EURUSD(Ask,Bid) VALUES (1.3601,1.3632);";
SQL = SQL + "INSERT INTO EURUSD(Ask,Bid) VALUES (1.3621,1.3643);";
SQL = SQL + "INSERT INTO EURUSD(Ask,Bid) VALUES (1.3605,1.3629);";
...

if (!MySqlExecute(DB,SQL))
   {
    // Showing an error message
   }
...
```

In this fragment, 3 entries will be inserted in the EURUSD table with a single call to the database. Each of the queries stored in the SQL variable is separated by ";".

This approach can be used for frequent insert/update/delete; a set of necessary commands is combined into one "package", thus relieving the network traffic and improving the database performance.

The INSERT syntax in MySQL is quite well developed in terms of exception handling.

For example, if the task is to move the price history, a table should be created for the currency pairs with the primary key of the datetime type, since the date and the time of a bar are unique. Moreover, it should be checked if the data on any particular bar exist in the database (to improve the stability of the data migration). With MySQL this check is not required, since the INSERT statement supports ON DUPLICATE KEY.

In more simple words, if an attempt is made to insert data, and the table already has an entry with the same date and time, the INSERT statement can be ignored or replaced by UPDATE for this row (see. [http://dev.mysql.com/doc/refman/5.0/en/insert-on-duplicate.html](https://www.mql5.com/go?link=https://dev.mysql.com/doc/refman/5.7/en/insert-on-duplicate.html "http://dev.mysql.com/doc/refman/5.0/en/insert-on-duplicate.html")).

**1.3. Data Selection**

The SQL SELECT statement is used for retrieving data from the database. The below sequence of actions is used for selecting data and retrieving the selection result:

1. Preparing the SELECT statement.
2. Opening the cursor.
3. Getting the number of rows returned by the query.
4. Looping and retrieving each row of the query.
5. Fetching data to the MQL variables inside the loop.
6. Closing the cursor.

Of course, this is a general scheme, so not all the operations are required for every case. For example, if you want to make sure that a row exists in the table (by any criteria), it will be enough to prepare a query, open a cursor, get the number of rows and close the cursor. In fact, the mandatory parts are - preparing the SELECT statement, opening and closing the cursor.

What is a cursor? This is a reference to the context memory area, in fact - the resulting set of values. When you send the SELECT query, the database allocates memory for the result and creates a pointer to a row that you can move from one row to another. Thus it is possible to access all the rows in the order of a queue defined by the query (ORDER BY clause of the SELECT statement).

The following interface functions are used for data selection:

Opening the cursor:

| **Type** | **Name** | **Parameters** | **Description** |
| --- | --- | --- | --- |
| **int** | MySqlCursorOpen | This function opens a cursor for the SELECT query and returns a cursor identifier in case of success. Otherwise, the function returns "-1". To find out the cause of the error, use the variables **_MySQLErrorNumber_** and **_MySqlErrorDescription_**. |
| **int** pConnection | Identifier of connection with the database |
| **string** pQuery | SQL query (the SELECT statement) |

Getting the number of rows returned by the query:

| **Type** | **Name** | **Parameters** | **Description** |
| --- | --- | --- | --- |
| **int** | MySqlCursorRows | This function returns the number of rows selected by the query. |
| **int** pCursorID | The cursor identifier returned by **MySqlCursorOpen** |

Fetching the query row:

| **Type** | **Name** | **Parameters** | **Description** |
| --- | --- | --- | --- |
| **bool** | MySqlCursorFetchRow | Fetches one row from the data set returned by the query. After successful execution, you can retrieve the data to MQL variables. The function returns true if successful, otherwise it returns false. |
| **int** pCursorID | The cursor identifier returned by **MySqlCursorOpen** |

Fetching data into MQL variables after fetching the query row:

| **Type** | **Name** | **Parameters** | **Description** |
| --- | --- | --- | --- |
| **int** | MySqlGetFieldAsInt | This function returns the representation of the table field value using the [**int**](https://www.mql5.com/en/docs/basis/types/integer/integertypes#int) data type. |
| **int** pCursorID | The cursor identifier returned by **MySqlCursorOpen** |
| **int** pField | The field number in the SELECT list (numbering starts with 0) |
| **double** | MySqlGetFieldAsDouble | This function returns the representation of the table field value using the [**double**](https://www.mql5.com/en/docs/basis/types/double) data type. |
| **int** pCursorID | The cursor identifier returned by **MySqlCursorOpen** |
| **int** pField | The field number in the SELECT list (numbering starts with 0) |
| **datetime** | MySqlGetFieldAsDatetime | This function returns the representation of the table field value using the [**datetime**](https://www.mql5.com/en/docs/basis/types/integer/datetime) data type. |
| **int** pCursorID | The cursor identifier returned by **MySqlCursorOpen** |
| **int** pField | The field number in the SELECT list (numbering starts with 0) |
| **string** | MySqlGetFieldAsString | This function returns the representation of the table field value using the [**string**](https://www.mql5.com/en/docs/basis/types/stringconst) data type. |
| **int** pCursorID | The cursor identifier returned by **MySqlCursorOpen** |
| **int** pField | The field number in the SELECT list (numbering starts with 0) |

All data returned by MySQL have native representation (presented without types as strings).

Therefore, using these functions, you can cast the selected data to the desired type. The only downside is specification of the column number (numbering starts from 0) in the SELECT list instead of its name. However, when developing an application, preparation of the SELECT statement and getting the results are almost always on one page, so you can see the SELECT query, when you prescribe the data fetching logic.

Thus, you always know the numbers of the fields in the SELECT list (this approach is also used when accessing data using AdoDB). Well, this part can further be revised in the future. But this will have little impact on the functionality of the developed solution.

Closing the cursor:

| **Type** | **Name** | **Parameters** | **Description** |
| --- | --- | --- | --- |
| **void** | MySqlCursorClose | This function closes the specified cursor and releases the memory. |
| **int** pCursorID | The cursor identifier returned by **MySqlCursorOpen** |

Closing a cursor is a critical operation. **Do not forget to close cursors.**

Imagine you open the cursor and forget to close it. Suppose, data are retrieved to the cursor with every tick while handling the [OnTick()](https://www.mql5.com/en/docs/basis/function/events#ontick) event, and every time a new cursor is opened, memory is allocated for it (both on the client side and the server side). At some point, the server will refuse the service because the limit of open cursors is reached, and this could cause buffer overflow.

Of course, it's exaggerated, such a result is possible when working with libmysql.dll directly. However, the MQLMySQL.DLL dynamic library distributes memory for cursors and will refuse to open a cursor that goes beyond the permissible limit.

When implementing real tasks, it is enough to keep 2-3 cursors open. Each cursor can handle one Cartesian measurement of data; using two-three cursors simultaneously (nested, for example, when one parametrically depends on another cursor) covers two or three dimensions. This is perfectly fine for the most tasks. In addition, for the implementation of complex data selection, you can always use these objects to represent the database (VIEW), create them on the server side and send queries to them from the MQL code as to tables.

**1.4. Additional Information**

The following can be mentioned as additional features:

_1.4.1. Reading data from an .INI file_

| **Type** | **Name** | **Parameters** | **Description** |
| --- | --- | --- | --- |
| **String** | ReadIni | Returns the value of a key of the given section of the INI-file. |
| string pFileName | The name of the INI file |
| string<br>pSection | The section name |
| string pKey | The key name |

Often storing information about connections to the database (IP address of the server, port, username, password, etc.) directly in the code MQL (or parameters of the Expert Advisor, indicator of script) is not rational, because the server can be moved, its address can change dynamically, etc. You will need to modify the MQL code in this case. Thus, all these data should better be stored in the standard .INI file, while only its name should be written in the MQL program. Then, use the ReadINI function to read connection parameters and use them.

For example, the INI file contains the following information:

```
[MYSQL]
Server = 127.0.0.1
User = root
Password = Adm1n1str@t0r
Database = mysql
Port = 3306
```

To get the IP address of the server, execute the following:

```
string vServer = ReadIni("C:\\MetaTrader5\\MQL5\\Experts\\MyConnection.ini", "MYSQL", "Server");
```

The INI file is located at C:\\MetaTrader5\\MQL5\\Experts and is called "MyConnection.ini", you access the **Server** key of the **MYSQL** section. In one INI file you can store settings to various servers used in your project.

_1.4.2. Tracing the Problem Areas_

In the interface library provides the trace mode, which can be enabled for debugging SQL queries anywhere in an MQL program.

Specify the following in the problem area:

```
SQLTrace = true;
```

and then

```
SQLTrace = false;
```

If you enable tracing at the beginning of the MQL program and do not disable it, all calls to the database will be logged. The Log is kept in the terminal console (using the Print command).

### 2\. Examples

This section provides a few examples of connection and use of the developed libraries. See them and estimate the usability of the software solution.

The **_MySQL-003.mq5_** example shows the following: connecting to a database (connection parameters are stored in the .ini file), creating a table, inserting data (also using multi-statements) and disconnecting from the database.

```
//+------------------------------------------------------------------+
//|                                                    MySQL-003.mq5 |
//|                                   Copyright 2014, Eugene Lugovoy |
//|                                              https://www.mql5.com |
//| Inserting data with multi-statement (DEMO)                       |
//+------------------------------------------------------------------+
#property copyright "Copyright 2014, Eugene Lugovoy."
#property link      "https://www.mql5.com"
#property version   "1.00"
#property strict

#include <MQLMySQL.mqh>

string INI;
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
{
 string Host, User, Password, Database, Socket; // database credentials
 int Port,ClientFlag;
 int DB; // database identifier

 Print (MySqlVersion());

 INI = TerminalInfoString(TERMINAL_PATH)+"\\MQL5\\Scripts\\MyConnection.ini";

 // reading database credentials from INI file
 Host = ReadIni(INI, "MYSQL", "Host");
 User = ReadIni(INI, "MYSQL", "User");
 Password = ReadIni(INI, "MYSQL", "Password");
 Database = ReadIni(INI, "MYSQL", "Database");
 Port     = (int)StringToInteger(ReadIni(INI, "MYSQL", "Port"));
 Socket   = ReadIni(INI, "MYSQL", "Socket");
 ClientFlag = CLIENT_MULTI_STATEMENTS; //(int)StringToInteger(ReadIni(INI, "MYSQL", "ClientFlag"));

 Print ("Host: ",Host, ", User: ", User, ", Database: ",Database);

 // open database connection
 Print ("Connecting...");

 DB = MySqlConnect(Host, User, Password, Database, Port, Socket, ClientFlag);

 if (DB == -1) { Print ("Connection failed! Error: "+MySqlErrorDescription); } else { Print ("Connected! DBID#",DB);}

 string Query;
 Query = "DROP TABLE IF EXISTS `test_table`";
 MySqlExecute(DB, Query);

 Query = "CREATE TABLE `test_table` (id int, code varchar(50), start_date datetime)";
 if (MySqlExecute(DB, Query))
    {
     Print ("Table `test_table` created.");

     // Inserting data 1 row
     Query = "INSERT INTO `test_table` (id, code, start_date) VALUES ("+(string)AccountInfoInteger(ACCOUNT_LOGIN)+",\'ACCOUNT\',\'"+TimeToString(TimeLocal(), TIME_DATE|TIME_SECONDS)+"\')";
     if (MySqlExecute(DB, Query))
        {
         Print ("Succeeded: ", Query);
        }
     else
        {
         Print ("Error: ", MySqlErrorDescription);
         Print ("Query: ", Query);
        }

     // multi-insert
     Query =         "INSERT INTO `test_table` (id, code, start_date) VALUES (1,\'EURUSD\',\'2014.01.01 00:00:01\');";
     Query = Query + "INSERT INTO `test_table` (id, code, start_date) VALUES (2,\'EURJPY\',\'2014.01.02 00:02:00\');";
     Query = Query + "INSERT INTO `test_table` (id, code, start_date) VALUES (3,\'USDJPY\',\'2014.01.03 03:00:00\');";
     if (MySqlExecute(DB, Query))
        {
         Print ("Succeeded! 3 rows has been inserted by one query.");
        }
     else
        {
         Print ("Error of multiple statements: ", MySqlErrorDescription);
        }
    }
 else
    {
     Print ("Table `test_table` cannot be created. Error: ", MySqlErrorDescription);
    }

 MySqlDisconnect(DB);
 Print ("Disconnected. Script done!");
}
```

Example _**MySQL-004.mq5**_ shows selection of data from a table created by the "MySQL-003.mq5" script.

```
//+------------------------------------------------------------------+
//|                                                    MySQL-004.mq5 |
//|                                   Copyright 2014, Eugene Lugovoy |
//|                                              https://www.mql5.com |
//| Select data from table (DEMO)                                    |
//+------------------------------------------------------------------+
#property copyright "Copyright 2014, Eugene Lugovoy."
#property link      "https://www.mql5.com"
#property version   "1.00"
#property strict

#include <MQLMySQL.mqh>

string INI;
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
{
 string Host, User, Password, Database, Socket; // database credentials
 int Port,ClientFlag;
 int DB; // database identifier

 Print (MySqlVersion());

 INI = TerminalInfoString(TERMINAL_PATH)+"\\MQL5\\Scripts\\MyConnection.ini";

 // reading database credentials from INI file
 Host = ReadIni(INI, "MYSQL", "Host");
 User = ReadIni(INI, "MYSQL", "User");
 Password = ReadIni(INI, "MYSQL", "Password");
 Database = ReadIni(INI, "MYSQL", "Database");
 Port     = (int)StringToInteger(ReadIni(INI, "MYSQL", "Port"));
 Socket   = ReadIni(INI, "MYSQL", "Socket");
 ClientFlag = (int)StringToInteger(ReadIni(INI, "MYSQL", "ClientFlag"));

 Print ("Host: ",Host, ", User: ", User, ", Database: ",Database);

 // open database connection
 Print ("Connecting...");

 DB = MySqlConnect(Host, User, Password, Database, Port, Socket, ClientFlag);

 if (DB == -1) { Print ("Connection failed! Error: "+MySqlErrorDescription); return; } else { Print ("Connected! DBID#",DB);}

 // executing SELECT statement
 string Query;
 int    i,Cursor,Rows;

 int      vId;
 string   vCode;
 datetime vStartTime;

 Query = "SELECT id, code, start_date FROM `test_table`";
 Print ("SQL> ", Query);
 Cursor = MySqlCursorOpen(DB, Query);

 if (Cursor >= 0)
    {
     Rows = MySqlCursorRows(Cursor);
     Print (Rows, " row(s) selected.");
     for (i=0; i<Rows; i++)
         if (MySqlCursorFetchRow(Cursor))
            {
             vId = MySqlGetFieldAsInt(Cursor, 0); // id
             vCode = MySqlGetFieldAsString(Cursor, 1); // code
             vStartTime = MySqlGetFieldAsDatetime(Cursor, 2); // start_time
             Print ("ROW[",i,"]: id = ", vId, ", code = ", vCode, ", start_time = ", TimeToString(vStartTime, TIME_DATE|TIME_SECONDS));
            }
     MySqlCursorClose(Cursor); // NEVER FORGET TO CLOSE CURSOR !!!
    }
 else
    {
     Print ("Cursor opening failed. Error: ", MySqlErrorDescription);
    }

 MySqlDisconnect(DB);
 Print ("Disconnected. Script done!");
}
```

The above examples contain the typical error handling used in real projects.

In fact, each query used in an MQL program should be debugged in any MySQL client (PHPMyAdmin, DB Ninja, MySQL console). I personally use and recommend professional software for database development Quest TOAD for MySQL.

### Conclusion

This article does not describe the details of implementation of MQLMySQL.DLL developed in the Microsoft Visual Studio 2010 (C/C++) environment. This software solution is designed for practical use and has more than 100 successful implementations in various areas of MQL software development (from the creation of complex trading systems to web publishing).

- The versions of the libraries for MQL4 and MQL5 are attached below. The attachments also include a zip file with the source code of MQLMySQL.DLL;
- Documentation is included in the archives;
- To use the examples, do not forget to specify the parameters of connection to your database in the file \\Scripts\\MyConnection.ini.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/932](https://www.mql5.com/ru/articles/932)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/932.zip "Download all attachments in the single ZIP archive")

[MQLMySQL\_for\_MQL4.zip](https://www.mql5.com/en/articles/download/932/mqlmysql_for_mql4.zip "Download MQLMySQL_for_MQL4.zip")(1124.85 KB)

[MQLMySQL\_for\_MQL5.zip](https://www.mql5.com/en/articles/download/932/mqlmysql_for_mql5.zip "Download MQLMySQL_for_MQL5.zip")(1125.19 KB)

[MQLMySQL\_DLL\_Project\_MSVS-2010.zip](https://www.mql5.com/en/articles/download/932/mqlmysql_dll_project_msvs-2010.zip "Download MQLMySQL_DLL_Project_MSVS-2010.zip")(1375.73 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/37085)**
(292)


![Alex FXPIP](https://c.mql5.com/avatar/2020/10/5F9DC28E-F0D0.PNG)

**[Alex FXPIP](https://www.mql5.com/en/users/rufxstrategy)**
\|
1 Nov 2024 at 18:28

Hi! Question for experts - how much data and how often I can read from MySQL to MT5 ?

For example, I have data 50 000 elements and I update them in the table every 0,1 sec (numbers). Will MT5 be able to pick them up from MySQL and update them every 0.1 sec? Is there any limitation of the functionality given in this article on KB per 1 query?


![Eugeniy Lugovoy](https://c.mql5.com/avatar/2014/4/5346C9F5-2917.jpg)

**[Eugeniy Lugovoy](https://www.mql5.com/en/users/elugovoy)**
\|
13 Dec 2024 at 15:47

**Alex Renko [#](https://www.mql5.com/ru/forum/36928/page11#comment_55005366):**

Hi! Question for experts - how much data and how often I can read from MySQL to MT5 ?

For example, I have data 50 000 elements and I update them in the table every 0,1 sec (numbers). Will MT5 be able to pick them up from MySQL and update them every 0.1 sec? Is there any limitation of the functionality given in this article on KB for 1 query?

Well, the question is certainly interesting ...

I must say that there are no limits on the number of SELECT query rows returned.

The size limit for the query itself is 64 Kb. So if you are trying to update 50k rows of data, it is better to split them into batches, say 1000 rows each, and thus send 50 queries.

As for the speed of 100 ms, if you have the database on one server, and your terminal in which you execute MQL with a connection to the database is somewhat remote, then most likely you will run into network latency, the size of the ping....

For example, if you have a ping of 60 ms between the database server and the terminal, the actual response from the server will be delayed = 60ms (query) + query processing time on the database side + 60ms (response).

This project is just a simple wrapper to access the functionality of dynamic mysql libraries.

The set of functionality is limited to the main practically useful functions, you can expand, add what you need, say add support for asynchronous queries, and then you can send all 50 queries on 1000 lines without waiting for the execution of each in turn.

P.S.: on Github you can see the library sources and preset limits [(https://github.com/elugovoy/MQLMySQL-Project/tree/master/MQLMySQL)](https://www.mql5.com/go?link=https://github.com/elugovoy/MQLMySQL-Project/tree/master/MQLMySQL "https://github.com/elugovoy/MQLMySQL-Project/tree/master/MQLMySQL").

P.P.S.: you can also download, modify at your discretion, compile and test.

![Eugeniy Lugovoy](https://c.mql5.com/avatar/2014/4/5346C9F5-2917.jpg)

**[Eugeniy Lugovoy](https://www.mql5.com/en/users/elugovoy)**
\|
13 Dec 2024 at 15:58

**andreysneg [#](https://www.mql5.com/ru/forum/36928/page10#comment_54804781):**

Is there any way to know the number of Fields that can be retrieved after MySqlCursorFetchRow ?

Maybe there is some hidden function like RowFieldsSize ...

As I understand, if there is no Field, MySqlGetFieldAsString returns empty. But also if String Field specifically contains an empty field, it also returns empty. I.e. it is not always possible to track the number of Fields by brute force.

As a crutch you can find out through sql command first, but then select again, but this is already an unnecessary Select.

Please develop the library, very useful thing. Of course, a couple of mysql should be built into mt a long time ago

Hm... and what kind of queries are so tricky at you that it is necessary to determine the number of fields returned by it?

Usually in the SELECT comand only list what is needed in a particular situation. Don't use SELECT **\***, select only what you need :) this is normal practice.

You should not make crutches, you can take the source code from Github and add a wrapper for [mysql\_fetch\_fields()](https://www.mql5.com/go?link=https://dev.mysql.com/doc/c-api/8.0/en/mysql-fetch-fields.html "5.4.20 mysql_fetch_fields()") MySQL API function

![Eugeniy Lugovoy](https://c.mql5.com/avatar/2014/4/5346C9F5-2917.jpg)

**[Eugeniy Lugovoy](https://www.mql5.com/en/users/elugovoy)**
\|
13 Dec 2024 at 16:03

**andreysneg [#](https://www.mql5.com/ru/forum/36928/page10#comment_54906137):**

Insert and Update query - only 16kb query limit ?

If the query is more than 16.000 characters, metatrader crashes (closes). if less, it is fine.

I attach an example of UPDATE for 32.000 characters.

Field for updating in the database - LONGTEXT

The library defines the size for queries in 64kb:

#define MAX\_QUERY\_SIZE 65535 // Max size of SQL query

I suppose in your case (and probably not only in your case, but in MQL string) there is 4-byte utf encoding, that is 16\*4 = 64 and the limit is reached....

Here either split queries or increase the buffer for the query and recompile.

![Gabms](https://c.mql5.com/avatar/avatar_na2.png)

**[Gabms](https://www.mql5.com/en/users/gabms)**
\|
15 Jan 2025 at 21:01

Awesome!

Its "too loud" use SELECTs with this wrapper in OnTick() function?

Thanks.

![MQL5 Cookbook: Handling BookEvent](https://c.mql5.com/2/11/OnBookEvent_MetaTrader5.png)[MQL5 Cookbook: Handling BookEvent](https://www.mql5.com/en/articles/1179)

This article considers BookEvent - a Depth of Market event, and the principle of its processing. An MQL program, handling states of Depth of Market, serves as an example. It is written using the object-oriented approach. Results of handling are displayed on the screen as a panel and Depth of Market levels.

![MQL5 Cookbook: Handling Custom Chart Events](https://c.mql5.com/2/11/avatar.png)[MQL5 Cookbook: Handling Custom Chart Events](https://www.mql5.com/en/articles/1163)

This article considers aspects of design and development of custom chart events system in the MQL5 environment. An example of an approach to the events classification can also be found here, as well as a program code for a class of events and a class of custom events handler.

![Why Virtual Hosting On The MetaTrader 4 And MetaTrader 5 Is Better Than Usual VPS](https://c.mql5.com/2/11/Virtual_hosting.png)[Why Virtual Hosting On The MetaTrader 4 And MetaTrader 5 Is Better Than Usual VPS](https://www.mql5.com/en/articles/1171)

The Virtual Hosting Cloud network was developed specially for MetaTrader 4 and MetaTrader 5 and has all the advantages of a native solution. Get the benefit of our free 24 hours offer - test out a virtual server right now.

![MQL5 Cookbook: Handling Typical Chart Events](https://c.mql5.com/2/11/OnChartEvent_MetaTrader5.png)[MQL5 Cookbook: Handling Typical Chart Events](https://www.mql5.com/en/articles/689)

This article considers typical chart events and includes examples of their processing. We will focus on mouse events, keystrokes, creation/modification/removal of a graphical object, mouse click on a chart and on a graphical object, moving a graphical object with a mouse, finish editing of text in a text field, as well as on chart modification events. A sample of an MQL5 program is provided for each type of event considered.

[Need a reliable hosting solution for your robots?Contact your broker and find out about available Sponsored MetaTrader VPS offeringsLearn more![](https://www.mql5.com/ff/sh/0pw0dk81s56qy774z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=vljwvezfjkfbvviocwskggexlvgykvob&s=70cf8e354b9a125332533ffb65d7365abe8dde5b5c1ede9caac479a9e9df4f25&uid=&ref=https://www.mql5.com/en/articles/932&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5071747402102156721)

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