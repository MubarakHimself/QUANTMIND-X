---
title: SQL and MQL5: Working with SQLite Database
url: https://www.mql5.com/en/articles/862
categories: Integration
relevance_score: 3
scraped_at: 2026-01-23T21:19:40.684661
---

[![](https://www.mql5.com/ff/si/3fgkjn78mkxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=ocndbzpeklfncxysjbwfhhbalbrsdbtv&s=a4309643278437a00bdd33c5809fc6b4b4032749c00fccd07b3b84e7b8b45126&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=clxiurxtthwlzubolfpcvgilpnlpzikd&ssn=1769192379055113523&ssn_dr=0&ssn_sr=0&fv_date=1769192379&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F862&back_ref=https%3A%2F%2Fwww.google.com%2F&title=SQL%20and%20MQL5%3A%20Working%20with%20SQLite%20Database%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176919237922181008&fz_uniq=5071749102909205946&sv=2552)

MetaTrader 5 / Examples


_Small. Fast. Reliable._

_Choose any of three._

### Introduction

Many developers consider using databases in their projects for data storage purposes and yet they remain hesitant about this, knowing how much extra time the SQL server installation may require. And whereas it may not be so difficult for programmers (if a database management system (DBMS) has already been installed for other purposes), it will certainly be an issue for a common user who might eventually be discouraged to install the software altogether.

So many developers choose not to deal with DBMS realizing that solutions they are currently working on will be used by very few people. As a result, they turn to working with files (often having to deal with more than one file, given the variety of data used): CSV, less often XML or JSON, or binary data files with strict structure size, etc.

However, it turns out there is a great alternative to SQL server! And you do not even need to install additional software as everything is done locally in your project, while still allowing you to use the full power of SQL. We are talking about SQLite.

The purpose of this article is to quickly get you started with SQLite. I will therefore not go into subtleties and all imaginable parameter sets and function flags but instead will create a light connection wrapper to execute SQL commands and will demonstrate its use.

To proceed with the article, you need to:

- Be in a good mood ;)
- Extract the archive files attached to the article to the MetaTrader 5 client terminal folder
- Install any convenient SQLite Viewer (e.g. [SQLiteStudio](https://www.mql5.com/go?link=http://sqlitestudio.pl/ "http://sqlitestudio.pl/"))
- Add the official documentation on SQLite [http://www.sqlite.org](https://www.mql5.com/go?link=http://www.sqlite.org/ "http://www.sqlite.org/") to Favorites

**Contents**

[1\. SQLite Principles](https://www.mql5.com/en/articles/862#sqlite_principles)

[2\. SQLite3 API](https://www.mql5.com/en/articles/862#sqlite_api)

[2.1. Opening and Closing a Database](https://www.mql5.com/en/articles/862#open_close_db)

[2.2. Execution of SQL Queries](https://www.mql5.com/en/articles/862#sql_queries)

[2.3. Getting Data from Tables](https://www.mql5.com/en/articles/862#get_data)

[2.4. Writing Parameter Data with Binding](https://www.mql5.com/en/articles/862#write_data)

[2.5. Transactions / Multirow Inserts (Example of Creating the Deal Table of a Trading Account)](https://www.mql5.com/en/articles/862#transactions)

[3\. Compiling 64-Bit Version (sqlite3\_64.dll)](https://www.mql5.com/en/articles/862#sqlite3_64)

### 1\. SQLite Principles

SQLite is an RDBMS whose key feature is the absence of a locally installed SQL server. Your application is seen here as a server. Working with SQLite database is basically working with a file (on a disk drive or **in the memory**). All data can be archived or moved to another computer without any need to install them in any specific way.

With SQLite, developers and users can benefit from several undeniable advantages:

- no need to install additional software;
- data are stored in a local file, thus offering transparency of management, i.e. you can view and edit them, independent of your application;
- ability to import and export tables to other DBMS;
- the code uses familiar SQL queries, which allows you to force the application to work with other DBMS at any time.

There are three ways of working with SQLite:

1. you can use the DLL file with a complete set of API functions;
2. you can use shell commands to an EXE file;
3. you can compile your project including source codes of SQLite API.

In this article, I will describe the first option, being the most customary in MQL5.

### **2\. SQLite3 API**

The connector operation will require the use of the following SQLite [functions](https://www.mql5.com/go?link=http://www.sqlite.org/c3ref/funclist.html "http://www.sqlite.org/c3ref/funclist.html"):

```
//--- general functions
sqlite3_open
sqlite3_prepare
sqlite3_step
sqlite3_finalize
sqlite3_close
sqlite3_exec

//--- functions for getting error descriptions
sqlite3_errcode
sqlite3_extended_errcode
sqlite3_errmsg

//--- functions for saving data
sqlite3_bind_null
sqlite3_bind_int
sqlite3_bind_int64
sqlite3_bind_double
sqlite3_bind_text
sqlite3_bind_blob

//--- functions for getting data
sqlite3_column_count
sqlite3_column_name
sqlite3_column_type
sqlite3_column_bytes
sqlite3_column_int
sqlite3_column_int64
sqlite3_column_double
sqlite3_column_text
sqlite3_column_blob
```

You will also need low-level msvcrt.dll functions for working with pointers:

```
strlen
strcpy
memcpy
```

Since I'm creating a connector that is supposed to work in 32 and 64 bit terminals, it is important to consider the size of the pointer sent to API functions. Let's separate their names:

```
// for a 32 bit terminal
#define PTR32                int
#define sqlite3_stmt_p32     PTR32
#define sqlite3_p32          PTR32
#define PTRPTR32             PTR32

// for a 64 bit terminal
#define PTR64                long
#define sqlite3_stmt_p64     PTR64
#define sqlite3_p64          PTR64
#define PTRPTR64             PTR64
```

If necessary, all API functions will be overloaded for 32 and 64 bit pointers. Please note that all pointers of the connector will be 64 bit. They will be converted to 32 bit directly in the overloaded API functions. The API function import source code is provided in **SQLite3Import.mqh**

**SQLite Data Types**

There are five data types in SQLite Version 3

| Type | Description |
| --- | --- |
| NULL | NULL value. |
| INTEGER | Integer stored in 1, 2, 3, 4, 6, or 8 bytes depending on the magnitude of the value stored. |
| REAL | 8-byte real number. |
| TEXT | Text string with the end character \\0 stored using UTF-8 or UTF-16 encoding. |
| BLOB | Arbitrary binary data |

You can also use other type names, e.g. BIGINT or INT accepted in various DBMS to specify the data type of a field when creating a table from a SQL query. In this case, SQLite will convert them in one of its intrinsic types, in this case to INTEGER. For further information about data types and their relations, please read the documentation [http://www.sqlite.org/datatype3.html](https://www.mql5.com/go?link=http://www.sqlite.org/datatype3.html "http://www.sqlite.org/datatype3.html")

**2.1. Opening and Closing a Database**

As you already know, a database in SQLite3 is a regular file. So opening a database is in fact equal to opening a file and getting its handle.

It is done using the **sqlite3\_open** function:

```
int sqlite3_open(const uchar &filename[], sqlite3_p64 &ppDb);

filename [in]  - a pathname or file name if the file is being opened at the current location.
ppDb     [out] - variable that will store the file handle address.

The function returns SQLITE_OK in case of success or else an error code.
```

A database file is closed using the **sqlite3\_close** function:

```
int sqlite3_close(sqlite3_p64 ppDb);

ppDb [in] - file handle

The function returns SQLITE_OK in case of success or else an error code.
```

Let us create database opening and closing functions in the connector.

```
//+------------------------------------------------------------------+
//| CSQLite3Base class                                               |
//+------------------------------------------------------------------+
class CSQLite3Base
  {
   sqlite3_p64       m_db;             // pointer to database file
   bool              m_bopened;        // flag "Is m_db handle valid"
   string            m_dbfile;         // path to database file

public:
                     CSQLite3Base();   // constructor
   virtual          ~CSQLite3Base();   // destructor

public:
   //--- connection to database
   bool              IsConnected();
   int               Connect(string dbfile);
   void              Disconnect();
   int               Reconnect();
  };
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CSQLite3Base::CSQLite3Base()
  {
   m_db=NULL;
   m_bopened=false;
  }
//+------------------------------------------------------------------+
//| Destructor                                                       |
//+------------------------------------------------------------------+
CSQLite3Base::~CSQLite3Base()
  {
   Disconnect();
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CSQLite3Base::IsConnected()
  {
   return(m_bopened && m_db);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int CSQLite3Base::Connect(string dbfile)
  {
   if(IsConnected())
      return(SQLITE_OK);
   m_dbfile=dbfile;
   return(Reconnect());
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CSQLite3Base::Disconnect()
  {
   if(IsConnected())
      ::sqlite3_close(m_db);
   m_db=NULL;
   m_bopened=false;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int CSQLite3Base::Reconnect()
  {
   Disconnect();
   uchar file[];
   StringToCharArray(m_dbfile,file);
   int res=::sqlite3_open(file,m_db);
   m_bopened=(res==SQLITE_OK && m_db);
   return(res);
  }
```

The connector can now open and close a database. Now check its performance with a simple script:

```
#include <MQH\Lib\SQLite3\SQLite3Base.mqh>

CSQLite3Base sql3; // database connector
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void OnStart()
{
//--- open database connection
   if(sql3.Connect("SQLite3Test.db3")!=SQLITE_OK)
      return;
//--- close connection
    sql3.Disconnect();
}
```

Run the script in debug mode, take a deep breath and check the operation of each string. As a result, a database file will appear in the MetaTrader 5 terminal installation folder. Congratulate yourself on this success and proceed to the next section.

**2.2. Execution of SQL Queries**

Any SQL query in SQLite3 has to go through at least three stages:

1. [sqlite3\_prepare](https://www.mql5.com/go?link=http://www.sqlite.org/c3ref/prepare.html "http://www.sqlite.org/c3ref/prepare.html") \- verification and receiving the list of statements;
2. [sqlite3\_step](https://www.mql5.com/go?link=http://www.sqlite.org/c3ref/step.html "http://www.sqlite.org/c3ref/step.html") \- executing these statements;
3. [sqlite3\_finalize](https://www.mql5.com/go?link=http://www.sqlite.org/c3ref/finalize.html "http://www.sqlite.org/c3ref/finalize.html") \- finalizing and memory clearing.

This structure is mainly suitable for creating or deleting tables, as well as for writing non-binary data, i.e. for the cases when an SQL query does not imply returning any data except for its execution success status.

If the query involves receiving data or writing binary data, _sqlite3\_column\_хх_ or _sqlite3\_bind\_хх_ function is used at the second stage, respectively. These functions are described in details in the next section.

Let's write _CSQLite3Base::Query_ method for executing a simple SQL query:

```
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int CSQLite3Base::Query(string query)
  {
//--- check connection
   if(!IsConnected())
      if(!Reconnect())
         return(SQLITE_ERROR);
//--- check query string
   if(StringLen(query)<=0)
      return(SQLITE_DONE);
   sqlite3_stmt_p64 stmt=0; // variable for pointer
//--- get pointer
   PTR64 pstmt=::memcpy(stmt,stmt,0);
   uchar str[];
   StringToCharArray(query,str);
//--- prepare statement and check result
   int res=::sqlite3_prepare(m_db,str,-1,pstmt,NULL);
   if(res!=SQLITE_OK)
      return(res);
//--- execute
   res=::sqlite3_step(pstmt);
//--- clean
   ::sqlite3_finalize(pstmt);
//--- return result
   return(res);
  }
```

As you can see, [sqlite3\_prepare](https://www.mql5.com/go?link=http://www.sqlite.org/c3ref/prepare.html "http://www.sqlite.org/c3ref/prepare.html"), [sqlite3\_step](https://www.mql5.com/go?link=http://www.sqlite.org/c3ref/step.html "http://www.sqlite.org/c3ref/step.html") and [sqlite3\_finalize](https://www.mql5.com/go?link=http://www.sqlite.org/c3ref/finalize.html "http://www.sqlite.org/c3ref/finalize.html") functions are following one after another.

Consider execution of _CSQLite3Base::Query_ when working with tables in SQLite:

```
// Create the table (CREATE TABLE)
sql3.Query("CREATE TABLE IF NOT EXISTS `TestQuery` (`ticket` INTEGER, `open_price` DOUBLE, `comment` TEXT)");
```

After executing this command, the table appears in the database:

![](https://c.mql5.com/2/6/01.png)

```
// Rename the table  (ALTER TABLE  RENAME)
sql3.Query("ALTER TABLE `TestQuery` RENAME TO `Trades`");

// Add the column (ALTER TABLE  ADD COLUMN)
sql3.Query("ALTER TABLE `Trades` ADD COLUMN `profit`");
```

After executing these commands, we receive the table with a new name and an additional field:

![](https://c.mql5.com/2/6/01__1.png)

```
// Add the row (INSERT INTO)
sql3.Query("INSERT INTO `Trades` VALUES(3, 1.212, 'info', 1)");

// Update the row (UPDATE)
sql3.Query("UPDATE `Trades` SET `open_price`=5.555, `comment`='New price'  WHERE(`ticket`=3)")
```

The following entry appears in the table after the new row is added and changed:

![](https://c.mql5.com/2/6/01__2.png)

Finally, the following commands should be executed one after another to clean up the database.

```
// Delete all rows from the table (DELETE FROM)
sql3.Query("DELETE FROM `Trades`")

// Delete the table (DROP TABLE)
sql3.Query("DROP TABLE IF EXISTS `Trades`");

// Compact database (VACUUM)
sql3.Query("VACUUM");
```

Before moving to the next section, we need the method that receives an error description. From my own experience I can say that the error code can provide plenty of information but the error's description shows the place in SQL query text where an error has appeared simplifying its detection and fixing.

```
const PTR64 sqlite3_errmsg(sqlite3_p64 db);

db [in] - handle received by function sqlite3_open

The pointer is returned to the string containing the error description.
```

In the connector, we should add the method for receiving this string from the pointer using _strcpy_ and _strlen_.

```
//+------------------------------------------------------------------+
//| Error message                                                    |
//+------------------------------------------------------------------+
string CSQLite3Base::ErrorMsg()
  {
   PTR64 pstr=::sqlite3_errmsg(m_db);  // get message string
   int len=::strlen(pstr);             // length of string
   uchar str[];
   ArrayResize(str,len+1);             // prepare buffer
   ::strcpy(str,pstr);                 // read string to buffer
   return(CharArrayToString(str));     // return string
  }
```

**2.3. Getting Data from Tables**

As I have already mentioned at the beginning of section 2.2, data reading is performed using _sqlite3\_column\_хх_ functions. This can be schematically shown as follows:

1. [sqlite3\_prepare](https://www.mql5.com/go?link=http://www.sqlite.org/c3ref/prepare.html "http://www.sqlite.org/c3ref/prepare.html")
2. [sqlite3\_column\_count](https://www.mql5.com/go?link=http://www.sqlite.org/c3ref/column_count.html "http://www.sqlite.org/c3ref/column_count.html") \- find out the number of columns of the obtained table
3. While current step result [sqlite3\_step](https://www.mql5.com/go?link=http://www.sqlite.org/c3ref/step.html "http://www.sqlite.org/c3ref/step.html") == _SQLITE\_ROW_

1. [sqlite3\_column\_хх](https://www.mql5.com/go?link=http://www.sqlite.org/c3ref/column_blob.html "http://www.sqlite.org/c3ref/column_blob.html") \- read the string cells

5. [sqlite3\_finalize](https://www.mql5.com/go?link=http://www.sqlite.org/c3ref/finalize.html "http://www.sqlite.org/c3ref/finalize.html")

Since we are approaching an extensive section concerning data reading and writing, it is a good time to describe three container classes used in the entire data exchange. The necessary data model depends on how the data is stored in the database:

Database

\|

**Table** is an array of rows.

\|

**Row** is an array of cells.

\|

**Cell** is a byte buffer of an arbitrary length.

```
//+------------------------------------------------------------------+
//| CSQLite3Table class                                              |
//+------------------------------------------------------------------+
class CSQLite3Table
  {

public:
   string            m_colname[]; // column name
   CSQLite3Row       m_data[];    // database rows
//...
  };
```

```
//+------------------------------------------------------------------+
//| CSQLite3Row class                                                |
//+------------------------------------------------------------------+
class CSQLite3Row
  {

public:
   CSQLite3Cell      m_data[];
//...
  };
```

```
//+------------------------------------------------------------------+
//| CSQLite3Cell class                                               |
//+------------------------------------------------------------------+
class CSQLite3Cell
  {

public:
   enCellType        type;
   CByteImg          buf;
//...
  };
```

As you can see, _CSQLite3Row_ and _CSQLite3Table_ connections are primitive - these are conventional data arrays. _CSQLite3Cell_ cell class also has _uchar_ data array + Data type field. Byte array is implemented in _CByteImage_ class (similar to well-known _[CFastFile](https://www.mql5.com/en/code/845)_).

I have created the following enumeration to facilitate connector's operation and manage cell data types:

```
enum enCellType
  {
   CT_UNDEF,
   CT_NULL,
   CT_INT,
   CT_INT64,
   CT_DBL,
   CT_TEXT,
   CT_BLOB,
   CT_LAST
  };
```

Note that _CT\_UNDEF_ type has been added to five basic SQLite3 types to identify the initial cell status. The entire INTEGER type divided into _CT\_INT_ and _CT\_INT64_ ones according to similarly divided _sqlite3\_bind\_intXX_ and _sqlite3\_column\_intXX_ functions.

**Getting Data**

In order to get data from the cell, we should create the method generalizing _sqlite3\_column\_хх_ type functions. It will check data type and size and write it to _CSQLite3Cell._

```
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CSQLite3Base::ReadStatement(sqlite3_stmt_p64 stmt,int column,CSQLite3Cell &cell)
  {
   cell.Clear();
   if(!stmt || column<0)
      return(false);
   int bytes=::sqlite3_column_bytes(stmt,column);
   int type=::sqlite3_column_type(stmt,column);
//---
   if(type==SQLITE_NULL)
      cell.type=CT_NULL;
   else if(type==SQLITE_INTEGER)
     {
      if(bytes<5)
         cell.Set(::sqlite3_column_int(stmt,column));
      else
         cell.Set(::sqlite3_column_int64(stmt,column));
     }
   else if(type==SQLITE_FLOAT)
      cell.Set(::sqlite3_column_double(stmt,column));
   else if(type==SQLITE_TEXT || type==SQLITE_BLOB)
     {
      uchar dst[];
      ArrayResize(dst,bytes);
      PTR64 ptr=0;
      if(type==SQLITE_TEXT)
         ptr=::sqlite3_column_text(stmt,column);
      else
         ptr=::sqlite3_column_blob(stmt,column);
      ::memcpy(dst,ptr,bytes);
      if(type==SQLITE_TEXT)
         cell.Set(CharArrayToString(dst));
      else
         cell.Set(dst);
     }
   return(true);
  }
```

The function is quite large, but it only reads data from the current statement and stores them in a cell.

We should also overload _CSQLite3Base::Query_ function by adding _CSQLite3Table_ container table for received data as the first parameter.

```
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int CSQLite3Base::Query(CSQLite3Table &tbl,string query)
  {
   tbl.Clear();
//--- check connection
   if(!IsConnected())
      if(!Reconnect())
         return(SQLITE_ERROR);
//--- check query string
   if(StringLen(query)<=0)
      return(SQLITE_DONE);
//---
   sqlite3_stmt_p64 stmt=NULL;
   PTR64 pstmt=::memcpy(stmt,stmt,0);
   uchar str[]; StringToCharArray(query,str);
   int res=::sqlite3_prepare(m_db, str, -1, pstmt, NULL); if(res!=SQLITE_OK) return(res);
   int cols=::sqlite3_column_count(pstmt); // get column count
   bool b=true;
   while(::sqlite3_step(pstmt)==SQLITE_ROW) // in loop get row data
     {
      CSQLite3Row row; // row for table
      for(int i=0; i<cols; i++) // add cells to row
        {
         CSQLite3Cell cell;
         if(ReadStatement(pstmt,i,cell)) row.Add(cell); else { b=false; break; }
        }
      tbl.Add(row); // add row to table
      if(!b) break; // if error enabled
     }
// get column name
   for(int i=0; i<cols; i++)
     {
      PTR64 pstr=::sqlite3_column_name(pstmt,i); if(!pstr) { tbl.ColumnName(i,""); continue; }
      int len=::strlen(pstr);
      ArrayResize(str,len+1);
      ::strcpy(str,pstr);
      tbl.ColumnName(i,CharArrayToString(str));
     }
   ::sqlite3_finalize(stmt);  // clean
   return(b?SQLITE_DONE:res); // return result code
  }
```

We have all functions necessary for receiving data. Let's pass to their examples:

```
// Read data (SELECT)
CSQLite3Table tbl;
sql3.Query(tbl, "SELECT * FROM `Trades`")
```

Print out the result of the query in the terminal using the following command **Print**( _TablePrint_( _tbl_)). We will see the following entries in the journal (the order is from bottom to top):

![](https://c.mql5.com/2/6/01__3.png)

```
// Sample calculation of stat. data from the tables (COUNT, MAX, AVG ...)
sql3.Query(tbl, "SELECT COUNT(*) FROM `Trades` WHERE(`profit`>0)")
sql3.Query(tbl, "SELECT MAX(`ticket`) FROM `Trades`")
sql3.Query(tbl, "SELECT SUM(`profit`) AS `sumprof`, AVG(`profit`) AS `avgprof` FROM `Trades`")
```

```
// Get the names of all tables in the base
sql3.Query(tbl, "SELECT `name` FROM `sqlite_master` WHERE `type`='table' ORDER BY `name`;");
```

The query result is printed out the same way using **Print**( _TablePrint_( _tbl_)). We can see the existing table:

![](https://c.mql5.com/2/6/01__4.png)

As can be seen from the examples, query execution results are placed into _tbl_ variable. After that, you can easily obtain and process them at your discretion.

**2.4. Writing Parameter Data with Binding**

Another topic that can be of importance for newcomers is writing data to the database having "incovenient" format. Of course, we mean binary data here. It cannot be passed directly in a common INSERT or UPDATE text statement, as a string is considered complete when the first zero is encountered. The same issue occurs when the string itself contains single quotes **'**.

Late binding may be useful in some cases, especially when the table is wide. It would be difficult and unreliable to write all fields to a single line, as you can easily miss something. The functions of [sqlite3\_bind\_хх](https://www.mql5.com/go?link=http://www.sqlite.org/c3ref/bind_blob.html "http://www.sqlite.org/c3ref/bind_blob.html") series are necessary for the binding operation.

In order to apply binding, a template should be inserted instead of passed data. I will consider one of the cases - **"?"** sign. In other words, UPDATE query will look as follows:

UPDATE \`Trades\` SET \`open\_price\`=?, \`comment\`=? WHERE(\`ticket\`=3)

Then, _sqlite3\_bind\_ **double**_ and _sqlite3\_bind\_ **text**_ functions should be executed one after another to place data to open\_price and comment. Generally, working with _bind_ functions can be represented the following way:

1. [sqlite3\_prepare](https://www.mql5.com/go?link=http://www.sqlite.org/c3ref/prepare.html "http://www.sqlite.org/c3ref/prepare.html")
2. Call [sqlite3\_bind\_хх](https://www.mql5.com/go?link=http://www.sqlite.org/c3ref/bind_blob.html "http://www.sqlite.org/c3ref/bind_blob.html") one after another and write required data to the statement
3. [sqlite3\_step](https://www.mql5.com/go?link=http://www.sqlite.org/c3ref/step.html "http://www.sqlite.org/c3ref/step.html")
4. [sqlite3\_finalize](https://www.mql5.com/go?link=http://www.sqlite.org/c3ref/finalize.html "http://www.sqlite.org/c3ref/finalize.html")

By the number of types _sqlite3\_bind\_xx_ completely repeats the reading functions described above. Thus, you can easily combine them in the connector at _CSQLite3Base::BindStatement_:

```
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CSQLite3Base::BindStatement(sqlite3_stmt_p64 stmt,int column,CSQLite3Cell &cell)
  {
   if(!stmt || column<0)
      return(false);
   int bytes=cell.buf.Len();
   enCellType type=cell.type;
//---
   if(type==CT_INT)        return(::sqlite3_bind_int(stmt, column+1, cell.buf.ViewInt())==SQLITE_OK);
   else if(type==CT_INT64) return(::sqlite3_bind_int64(stmt, column+1, cell.buf.ViewInt64())==SQLITE_OK);
   else if(type==CT_DBL)   return(::sqlite3_bind_double(stmt, column+1, cell.buf.ViewDouble())==SQLITE_OK);
   else if(type==CT_TEXT)  return(::sqlite3_bind_text(stmt, column+1, cell.buf.m_data, cell.buf.Len(), SQLITE_STATIC)==SQLITE_OK);
   else if(type==CT_BLOB)  return(::sqlite3_bind_blob(stmt, column+1, cell.buf.m_data, cell.buf.Len(), SQLITE_STATIC)==SQLITE_OK);
   else if(type==CT_NULL)  return(::sqlite3_bind_null(stmt, column+1)==SQLITE_OK);
   else                    return(::sqlite3_bind_null(stmt, column+1)==SQLITE_OK);
  }
```

The only objective of this method is to write the buffer of the passed cell to the statement.

Let's add _CQLite3Table::QueryBind_ method in a similar way. Its first argument is data string for writing:

```
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int CSQLite3Base::QueryBind(CSQLite3Row &row,string query) // UPDATE <table> SET <row>=?, <row2>=?  WHERE (cond)
  {
   if(!IsConnected())
      if(!Reconnect())
         return(SQLITE_ERROR);
//---
   if(StringLen(query)<=0 || ArraySize(row.m_data)<=0)
      return(SQLITE_DONE);
//---
   sqlite3_stmt_p64 stmt=NULL;
   PTR64 pstmt=::memcpy(stmt,stmt,0);
   uchar str[];
   StringToCharArray(query,str);
   int res=::sqlite3_prepare(m_db, str, -1, pstmt, NULL);
   if(res!=SQLITE_OK)
      return(res);
//---
   bool b=true;
   for(int i=0; i<ArraySize(row.m_data); i++)
     {
      if(!BindStatement(pstmt,i,row.m_data[i]))
        {
         b=false;
         break;
        }
     }
   if(b)
      res=::sqlite3_step(pstmt); // executed
   ::sqlite3_finalize(pstmt);    // clean
   return(b?res:SQLITE_ERROR);   // result
  }
```

Its objective is to write the string to the appropriate parameters.

**2.5. Transactions / Multirow Inserts**

Before proceeding with this topic, you need to know one more SQLite API function. In the previous section, I have described the three-stage handling of requests: _prepare_ + _step_ + _finalize_. However, there is an alternative (in some cases, simple or even critical) solution – [sqlite3\_exec](https://www.mql5.com/go?link=http://www.sqlite.org/c3ref/exec.html "http://www.sqlite.org/c3ref/exec.html") function:

```
int sqlite3_exec(sqlite3_p64 ppDb, const char &sql[], PTR64 callback, PTR64 pvoid, PTRPTR64 errmsg);

ppDb [in] - database handle
sql  [in] - SQL query
The remaining three parameters are not considered yet in relation to MQL5.

It returns SQLITE_OK in case of success or else an error code.
```

Its main objective is to execute the query in a single call without creating three-stage constructions.

Let's add its call to the connector:

```
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int CSQLite3Base::Exec(string query)
  {
   if(!IsConnected())
      if(!Reconnect())
         return(SQLITE_ERROR);
   if(StringLen(query)<=0)
      return(SQLITE_DONE);
   uchar str[];
   StringToCharArray(query,str);
   int res=::sqlite3_exec(m_db,str,NULL,NULL,NULL);
   return(res);
  }
```

The resulting method is easy to use. For example, you can execute table deletion (DROP TABLE) or database compact (VACUUM) command the following way:

```
sql3.Exec("DROP TABLE `Trades`");

sql3.Exec("VACUUM");
```

**Transactions**

Now, suppose that we have to add several thousands of rows to the table. If we insert all this into the loop:

```
for (int i=0; i<N; i++)
   sql3.Query("INSERT INTO `Table` VALUES(1, 2, 'text')");
```

the execution will be very slow (more than 10(!) seconds). Thus, such implementation is **not recommended** in SQLite. The most appropriate solution here is to use [transactions](https://www.mql5.com/go?link=http://www.sqlite.org/lang_transaction.html "http://www.sqlite.org/lang_transaction.html"): all SQL statements are entered into a common list and then passed as a single query.

The following SQL statements are used to write transaction start and end:

```
BEGIN
...
COMMIT
```

All the contents is executed at the last _COMMIT_ statement. ROLLBACK statement is used in case the loop should be interrupted or already added statements should not be executed.

As an example, all account deals are added to the table.

```
#include <MQH\Lib\SQLite3\SQLite3Base.mqh>
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void OnStart()
  {
   CSQLite3Base sql3;

//--- open database connection
   if(sql3.Connect("Deals.db3")!=SQLITE_OK) return;
//---
   if(sql3.Query("CREATE TABLE IF NOT EXISTS `Deals` (`ticket` INTEGER PRIMARY KEY, `open_price` DOUBLE, `profit` DOUBLE, `comment` TEXT)")!=SQLITE_DONE)
     {
      Print(sql3.ErrorMsg());
      return;
     }

//--- create transaction
   if(sql3.Exec("BEGIN")!=SQLITE_OK)
     {
      Print(sql3.ErrorMsg());
      return;
     }
   HistorySelect(0,TimeCurrent());
//--- dump all deals from terminal to table
   for(int i=0; i<HistoryDealsTotal(); i++)
     {
      CSQLite3Row row;
      long ticket=(long)HistoryDealGetTicket(i);
      row.Add(ticket);
      row.Add(HistoryDealGetDouble(ticket, DEAL_PRICE));
      row.Add(HistoryDealGetDouble(ticket, DEAL_PROFIT));
      row.Add(HistoryDealGetString(ticket, DEAL_COMMENT));
      if(sql3.QueryBind(row,"REPLACE INTO `Deals` VALUES("+row.BindStr()+")")!=SQLITE_DONE)
        {
         sql3.Exec("ROLLBACK");
         Print(sql3.ErrorMsg());
         return;
        }
     }
//--- end transaction
   if(sql3.Exec("COMMIT")!=SQLITE_OK)
      return;

//--- get statistical information from table
   CSQLite3Table tbl;
   CSQLite3Cell cell;

   if(sql3.Query(tbl,"SELECT COUNT(*) FROM `Deals` WHERE(`profit`>0)")!=SQLITE_DONE)
     {
      Print(sql3.ErrorMsg());
      return;
     }
   tbl.Cell(0,0,cell);
   Print("Count(*)=",cell.GetInt64());
//---
   if(sql3.Query(tbl,"SELECT SUM(`profit`) AS `sumprof`, AVG(`profit`) AS `avgprof` FROM `Deals`")!=SQLITE_DONE)
     {
      Print(sql3.ErrorMsg());
      return;
     }
   tbl.Cell(0,0,cell);
   Print("SUM(`profit`)=",cell.GetDouble());
   tbl.Cell(0,1,cell);
   Print("AVG(`profit`)=",cell.GetDouble());
  }
```

After the script is applied to the account, it inserts account deals to the table instantly.

![](https://c.mql5.com/2/6/02.png)

Statistics is displayed in the terminal journal

![](https://c.mql5.com/2/6/02__1.png)

You can toy around with the script: comment out the lines containing _BEGIN_, _ROLLBACK_ and _COMMIT_. If there are over hundreds of deals on your account, you will see the difference immediately. By the way, according to [some](https://www.mql5.com/go?link=https://habrahabr.ru/post/42121/ "http://habrahabr.ru/post/42121/") [tests](https://www.mql5.com/go?link=http://www.sqlite.org/speed.html "http://www.sqlite.org/speed.html"), SQLite transactions work faster than in MySQL or PostgreSQL.

### 3\. Compiling 64-Bit Version (sqlite3\_64.dll)

1. Download [SQLite source code](https://www.mql5.com/go?link=http://sqlite.org/download.html "/go?http://sqlite.org/download.html") (amalgamation) and find **sqlite3.c** file.
2. Download [sqlite-dll-win32](https://www.mql5.com/go?link=http://sqlite.org/download.html "http://sqlite.org/download.html") and extract **sqlite3.dll** file from it.
3. Execute **LIB.EXE /DEF:sqlite3.def** console command in the folder where dll file has been extracted. Make sure that the paths to lib.exe file are set in PATH system variable or find it in your Visual Studio.
4. Create DLL project selecting Release configuration for 64 bit platforms.
5. Add downloaded **sqlite3.c** and obtained **sqlite3.def** files to the project. If the compiler does not accept some functions from **def** file, just comment them out.
6. The following parameters should be set in the project settings:

**C/C++** --\> **General** --\> **Debug Information Format** = **Program Database (/Zi**)

**C/C++** --\> **Precompiled Headers** --\> **Create/Use Precompiled Header** = **Not Using Precompiled Headers (/Yu)**

7. Compile and get 64 bit dll.

### Conclusion

I hope that the article will become your indispensable guide in mastering SQLite. Perhaps, you will use it in your future projects. This brief overview has provided some insight into the functionality of SQLite as a perfect and reliable solution for applications.

In this article, I have described all cases you may face when handling trading data. As a homework, I recommend that you develop a simple tick collector inserting ticks into the table for each symbol. You can find class library source code and test scripts in the attachment below.

**I wish you good luck and big profits!**

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/862](https://www.mql5.com/ru/articles/862)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/862.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/862/mql5.zip "Download MQL5.zip")(790.12 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Working with sockets in MQL, or How to become a signal provider](https://www.mql5.com/en/articles/2599)
- [Getting Rid of Self-Made DLLs](https://www.mql5.com/en/articles/364)
- [Promote Your Development Projects Using EX5 Libraries](https://www.mql5.com/en/articles/362)
- [Using WinInet in MQL5. Part 2: POST Requests and Files](https://www.mql5.com/en/articles/276)
- [Tracing, Debugging and Structural Analysis of Source Code](https://www.mql5.com/en/articles/272)
- [The Prototype of a Trading Robot](https://www.mql5.com/en/articles/132)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/23404)**
(46)


![Sergey Likho](https://c.mql5.com/avatar/2022/6/62A1BDA2-1D02.jpg)

**[Sergey Likho](https://www.mql5.com/en/users/serler2)**
\|
13 May 2019 at 18:06

Question

Apparently, this is a peculiarity of encoding.

If you write Russian text in the database, it looks crooked in SQLite studio (rhombuses with a question).

And if you enter Russian text manually, then it looks crooked in MT4.

Question: how to make the text display in the correct encoding so that the Russian text is visible?

[![](https://c.mql5.com/3/279/prt5ql_kl7aau_2019-05-13_9_19.03.33.png)](https://c.mql5.com/3/279/fhbd7g_iuff86_2019-05-13_0_19.03.33.png "https://c.mql5.com/3/279/fhbd7g_iuff86_2019-05-13_0_19.03.33.png")

![Martin Bittencourt](https://c.mql5.com/avatar/2020/5/5EC064DD-BF5C.jpeg)

**[Martin Bittencourt](https://www.mql5.com/en/users/momergil)**
\|
9 Oct 2019 at 06:19

**MetaQuotes Software Corp.:**

New article [SQL and MQL5: Working with SQLite Database](https://www.mql5.com/en/articles/862) has been published:

Author: [o\_O](https://www.mql5.com/en/users/sergeev "sergeev")

Hey!

Since build 2170, this library is giving scope errors. Could you please update it as to follow the new MQL5 scope 'guidelines'? Thanks!

Btw great work!

![Marcos Silva](https://c.mql5.com/avatar/avatar_na2.png)

**[Marcos Silva](https://www.mql5.com/en/users/investguy)**
\|
17 Mar 2020 at 03:07

A new article implementing a native SQL solution can be found here: [https://www.mql5.com/en/articles/7463](https://www.mql5.com/en/articles/7463)

![EABASE](https://c.mql5.com/avatar/avatar_na2.png)

**[EABASE](https://www.mql5.com/en/users/eabase)**
\|
27 Sep 2020 at 04:10

**Quintos:**

I think I found a memory leak:

In SQLite3Base.mqh line 250

Should be:

Good catch!

(Any tools avilable to [check](https://www.mql5.com/en/docs/common/GetTickCount64 "MQL5 Documentation: GetTickCount64 function") these things?)


![benqzyp](https://c.mql5.com/avatar/avatar_na2.png)

**[benqzyp](https://www.mql5.com/en/users/benqzyp)**
\|
26 Mar 2022 at 11:46

ByteImg.mqh is running wrong, please fix it!


![Tales of Trading Robots: Is Less More?](https://c.mql5.com/2/0/tales.png)[Tales of Trading Robots: Is Less More?](https://www.mql5.com/en/articles/910)

Two years ago in "The Last Crusade" we reviewed quite an interesting yet currently not widely used method for displaying market information - point and figure charts. Now I suggest you try to write a trading robot based on the patterns detected on the point and figure chart.

![Common Errors in MQL4 Programs and How to Avoid Them](https://c.mql5.com/2/13/1152_84.png)[Common Errors in MQL4 Programs and How to Avoid Them](https://www.mql5.com/en/articles/1391)

To avoid critical completion of programs, the previous version compiler handled many errors in the runtime environment. For example, division by zero or array out of range are critical errors and usually lead to program crash. The new compiler can detect actual or potential sources of errors and improve code quality. In this article, we discuss possible errors that can be detected during compilation of old programs and see how to fix them.

![MQL5 Cookbook: Development of a Multi-Symbol Indicator to Analyze Price Divergence](https://c.mql5.com/2/0/avatar__11.png)[MQL5 Cookbook: Development of a Multi-Symbol Indicator to Analyze Price Divergence](https://www.mql5.com/en/articles/754)

In this article, we will consider the development of a multi-symbol indicator to analyze price divergence in a specified period of time. The core topics have been already discussed in the previous article on the programming of multi-currency indicators "MQL5 Cookbook: Developing a Multi-Symbol Volatility Indicator in MQL5". So this time we will dwell only on those new features and functions that have been changed dramatically. If you are new to the programming of multi-currency indicators, I recommend you to first read the previous article.

![Upgrade to MetaTrader 4 Build 600 and Higher](https://c.mql5.com/2/13/1145_130.png)[Upgrade to MetaTrader 4 Build 600 and Higher](https://www.mql5.com/en/articles/1389)

The new version of the MetaTrader 4 terminal features the updated structure of user data storage. In earlier versions all programs, templates, profiles etc. were stored directly in terminal installation folder. Now all necessary data required for a particular user are stored in a separate directory called data folder. Read the article to find answers to frequently asked questions.

[![](https://www.mql5.com/ff/sh/rvgkjnsrvj1mzh89z2/01.png)Best VPS for tradersTwo-click launch from MetaTrader, minimum ping to broker, 15 USD/monthLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=wpjhvzsogglsviotmypjoyhhtuxlrzhi&s=aa6c5782a1658c2f617954d478dea9989a27ae26ecabc09d0ab1204277fdf8e3&uid=&ref=https://www.mql5.com/en/articles/862&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5071749102909205946)

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