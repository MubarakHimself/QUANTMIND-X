---
title: Implementing Practical Modules from Other Languages in MQL5 (Part 01): Building the SQLite3 Library, Inspired by Python
url: https://www.mql5.com/en/articles/18640
categories: Trading Systems, Integration
relevance_score: 9
scraped_at: 2026-01-22T17:34:03.883977
---

[Best articles and CodeBase updates in MQL5.community channelsFollow us to ensure you never miss out on important updates![](https://www.mql5.com/ff/sh/n9yf51p2srwzfqh5z2/02.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=dgazvhktsxqakdvarucjbvmvzenwlyje&s=98a038fe082e458df8c4a1d8e116e3a6646fd5517f06e48b2356b7ee005817d6&uid=&ref=https://www.mql5.com/en/articles/18640&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5049220552037607257)

MetaTrader 5 / Trading systems


**Contents**

- [Introduction](https://www.mql5.com/en/articles/18640#para1)
- [What is a sqlite3 module offered in Python](https://www.mql5.com/en/articles/18640#what-is-sqlite3-python)
- [Connecting to a SQLite database](https://www.mql5.com/en/articles/18640#connecting-to-sqlite-db)
- [Executing SQL statements](https://www.mql5.com/en/articles/18640#executing-sql-statements)
- [Working with Text (string) values from the database](https://www.mql5.com/en/articles/18640#working-with-strings-from-db)
- [Database transactions control](https://www.mql5.com/en/articles/18640#db-transactions-control)
- [Other methods in the sqlite3 module](https://www.mql5.com/en/articles/18640#other-methods-in-sqlite)
- [Logging trades to the database](https://www.mql5.com/en/articles/18640#logging-trades-to-db)
- [Conclusion](https://www.mql5.com/en/articles/18640#para2)

### Introduction

Has it ever happened to you that you want one or two of your favorite modules, libraries, frameworks, etc., present in another programming language apart from MQL5 inside MQL5? _It happens a lot to me._

There is a huge number of developers in the MQL5 community that come from different coding backgrounds; some come from web development like myself, some come from Android development, and many more coding backgrounds present today. This means most programmers are familiar with different languages such as JavaScript, Java, Python, C++, C#, Just to name a few.

In those different languages, coders come across different coding tools (modules), useful modules that we just want to use anywhere possible. For example, I love using the NumPy module offered in Python for mathematical calculations, to the extent that I had to implement a similar library once in MQL5, [in this article](https://www.mql5.com/en/articles/17469).

While an attempt to implement a module, tool, framework, etc., from one language into another— in this case into MQL5, could produce a slightly different functionality and outcome(s) due to the distinct nature of programming languages, but having a similar syntax or experience might be sufficient to make product development in MQL5 easy and a fun experience for developers familiar with different languages. Not to mention, we might learn some valuable information in the process that could solidify our programming skills in general.

> ![](https://c.mql5.com/2/152/article_series_image.png)

In this new article series, we will be implementing not every module from other languages, but **every module practical in MQL5 from another language**. For example, modules for mathematical calculations, data storage, data analysis, etc.

Starting with the [sqlite3 module](https://www.mql5.com/go?link=https://docs.python.org/3/library/sqlite3.html "https://docs.python.org/3/library/sqlite3.html") that comes built-in with [Python programming language](https://www.mql5.com/go?link=https://www.python.org/ "https://www.python.org/").

### What is a Sqlite3 Module Offered in Python

Let us first understand a [SQLite database](https://www.mql5.com/go?link=https://sqlite.org/ "https://sqlite.org/"):

A SQLite database is a lightweight, self-contained, serverless SQL database engine. It is widely used in applications that need a simple, embedded, local data storage solution. It is a file-based database which stores (schema, tables, indexes, and data) all in a single **.sqlite** or **.db** file.

Unlike [MySQL](https://www.mql5.com/go?link=https://www.mysql.com/ "https://www.mysql.com/") or [PostgreSQL](https://www.mql5.com/go?link=https://www.postgresql.org/ "https://www.postgresql.org/") databases, which require some setup, a server, and administration configurations, SQLite runs right off the bat without any of that as it reads and writes directly to the disk.

The MQL5 programming language comes with built-in functions for working with a SQLite database; these built-in functions are sufficient. However, they are not that easy to use compared to if you were to use the sqlite3 module in Python.

For example, trying to create a simple "example" database and insert some information into a table named **users,** _the table is automatically created if it doesn't exist._

**Using Python sqlite module**

```
import sqlite3

conn = sqlite3.connect("example.db")

cursor = conn.cursor()

cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        age INTEGER,
        email TEXT UNIQUE
    )
""") # Execute a query

try:
    cursor.execute("INSERT INTO users (name, age, email) VALUES (?, ?, ?)",
                   ("Bruh", 30, "bruh@example.com"))

    conn.commit() # commit the transaction | save the new information to a database

except sqlite3.DatabaseError as e:
    print("Insert failed:", e)

conn.close() # closing the database
```

**Using native MQL5 functions**

```
void OnStart()
  {
//---

   int db_handle = DatabaseOpen("example.db", DATABASE_OPEN_READWRITE | DATABASE_OPEN_CREATE);
   if (db_handle == INVALID_HANDLE)
     {
       printf("Failed to open a database. Error = %s",ErrorDescription(GetLastError()));
       return;
     }

    string sql =
       "   CREATE TABLE IF NOT EXISTS users ("
       "   id INTEGER PRIMARY KEY AUTOINCREMENT,"
       "   name TEXT NOT NULL,"
       "   age INTEGER,"
       "   email TEXT UNIQUE"
       ")";

    if (!DatabaseExecute(db_handle, sql)) //Execute a sql query
      {
         printf("Failed to execute a query to the database. Error = %s",ErrorDescription(GetLastError()));
         return;
      }

    if (!DatabaseTransactionBegin(db_handle)) //Begin the transaction
      {
         printf("Failed to begin a transaction. Error = %s",ErrorDescription(GetLastError()));
         return;
      }

    sql = "INSERT INTO users (name, age, email) VALUES ('Bruh', 30, 'bruh@example.com')";

    if (!DatabaseExecute(db_handle, sql)) //Execute a query
      {
         printf("Failed to execute a query to the database. Error = %s",ErrorDescription(GetLastError()));
         return;
      }

   if (!DatabaseTransactionCommit(db_handle)) //Commit the transaction | push the changes to a database
      {
         printf("Failed to commit a transaction. Error = %s",ErrorDescription(GetLastError()));
         return;
      }

    DatabaseClose(db_handle); //Close the database
 }
```

We can all agree that using sqlite3 in Python made our code cleaner than using native MQL5 functions, which require us to handle errors, the information returned, or added to the database.

The sqlite3 module handles a plenty of unnecessary steps to executing a command and managing transactions, making it much easier for users to get and insert some data into and from the SQLite database without much hassle.

So, let's attempt to implement a similar module in MQL5.

### Connecting to a SQLite Database

The **connect**  method in sqlite3 Python creates a new database if the given name for the database doesn't exist and returns a connection object which represents the connection to the on-disk database.

```
import sqlite3
con = sqlite3.connect("example.db")
```

In MQL5, this connection is similar to the database **handle**  so technically, we don't need to return a handle in our MQL5 library as we will use it inside all the functions in our class.

```
class CSqlite3
  {
protected:

   int m_request;

public:
                     int m_db_handle;
 //... Other functions
 }
```

```
bool CSqlite3::connect(const string db_name, const bool common=false, const bool database_in_memory=false)
 {
   int flags = DATABASE_OPEN_READWRITE | DATABASE_OPEN_CREATE;

   if (common) //Open the database from the common folder
      flags |= DATABASE_OPEN_COMMON;

   if (database_in_memory) //Open the database in memory
      flags |= DATABASE_OPEN_MEMORY;

   m_db_handle = DatabaseOpen(db_name, flags);

//---

   if (m_db_handle == INVALID_HANDLE)
     {
       printf("func=%s line=%d, Failed to open a database. Error = %s",__FUNCTION__, __LINE__, ErrorDescription(GetLastError()));
       return false;
     }

   return true;
 }
```

MQL5 gives us an option to save the data from either the [datapath](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/cterminalinfo/cterminalinfodatapath) (folder) or the [common datapath (folder)](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/cterminalinfo/cterminalinfocommondatapath). When the **common** argument is set to **true,** the database will be read from the common datapath instead of a regular datapath.

We can also give the user an option to choose whether to open a database in memory (RAM) or on disk, so when the variable **database\_in\_memory** is set to **true**. The database will be [created in RAM](https://www.mql5.com/en/docs/database/databaseopen#:~:text=DATABASE_OPEN_MEMORY) instead of the disk and vice versa.

We are ignoring one flag i.e., DATABASE\_OPEN\_READONLY for opening the database in [ReadOnly](https://en.wikipedia.org/wiki/Read-only "https://en.wikipedia.org/wiki/Read-only") mode. The reason is simple; if you don't want to write to a database you won't execute the "INSERT" queries to it at all. _Makes sense?_

### Executing SQL Statements

This is one of the most crucial functions we often use when working with SQLite databases. This function enables us to get the information, insert, update, delete, and modify the values in the database, etc.

This function executes SQL statements and commands on our database directly.

In sqlite3 Python, the function **execute()** works seamlessly and effortless. It can take any command for either inserting or getting the information, and it automatically knows when and what to return and what not to.

In MQL5, we have a built-in function named [DatabaseExecute](https://www.mql5.com/en/docs/database/databaseexecute)  which is similar to **execute()** in sqlite3 Python. They both execute a request to a specified database or table. However, this MQL5 method is suitable for executing all requests but, all requests with the keyword "SELECT" for reading the information from the database.

To effectively read information from the database we use the function [DatabasePrepare](https://www.mql5.com/en/docs/database/databaseprepare) as it creates a handle of a request that can then be executed using [DatabaseRead](https://www.mql5.com/en/docs/database/databaseread).

To create a similar flexible function in MQL5 for executing functions regardless of the query type, we need to distinguish the types of SQL queries and return the right information with the right query.

```
execute_res_structure CSqlite3::execute(const string sql)
 {
   execute_res_structure res;

   string trimmed_sql = sql;
   StringTrimLeft(trimmed_sql); //Trim the leading white space

   string first_word = trimmed_sql;

   // Find the index of the first space (to isolate the first SQL keyword)
   int space_index = StringFind(trimmed_sql, " ");
   if (space_index != -1)
      first_word = StringSubstr(trimmed_sql, 0, space_index); // Extract the first word from the query

   StringToUpper(first_word); //Convert the first word in the query to uppercase for comparison

   if (first_word == "SELECT")
    {
      // SELECT query – prepare and expect data
      m_request = DatabasePrepare(m_db_handle, sql);
      res.request = m_request;

      return res;
    }
   else
    {
      // INSERT/UPDATE/DELETE – execute directly
       if (!m_transaction_active)
         {
            if (!this.begin())
               return res;
            else
                m_transaction_active_auto = true; //The transaction was started automatically by the execute function
         }

       ResetLastError();
       if (!DatabaseExecute(m_db_handle, sql))
         {
            printf("func=%s line=%d, Failed to execute a query to the database. Error = %s",__FUNCTION__,__LINE__,ErrorDescription(GetLastError()));
         }
    }

   return res;
 }
```

Similarly to the execute function in Python, all the non-SELECT statements implicitly open a transaction using the function **begin**, this transaction needs to be committed before changes are saved in the database.

After calling the function **execute**  and inserting some information into the database, you have to call the **commit** method to save the changes to the database. _We'll discuss this function later on in this post_.

When the function is called with a **non-SELECT** statement or query, it doesn't return any value as it is probably updating, modifying, and inserting the information into the database. However, when the function is called with SELECT type of query, it returns a data structure. Let's discuss this structure in detail.

**1\. The fetchone Method**

The sqlite3 module offered in Python is capable of dynamically returning a whole or part of the information received from the SQL statement.

Starting with the ability to return one row of information from the database.

```
import sqlite3
conn = sqlite3.connect("example.db")

cursor = conn.cursor()
print(conn.execute("SELECT * FROM users").fetchone())
```

Despite requesting all the information available in the SQLite database from the SQL statement, the fetchone function restricts the execute function from returning more than one row of data from the database.

Database.

![](https://c.mql5.com/2/153/2446137604720.png)

Outputs.

```
(mcm-env) C:\Users\Omega Joctan\OneDrive\Desktop\MCM>C:/Anaconda/envs/mcm-env/python.exe "c:/Users/Omega Joctan/OneDrive/Desktop/MCM/sqlite_test.py"
(1, 'Alice', 30, 'alice@example.com')
```

We need a similar function in our MQL5 class, inside the structure named **execute\_res\_structure** which is returned by the function **execute**.

```
struct execute_res_structure
  {
   int request;
   CLabelEncoder le;

   vector fetchone()
      {
         int cols = DatabaseColumnsCount(request);
         vector row = vector::Zeros(cols);

         while (DatabaseRead(request))  // Essential, read the entire database
          {
            string row_string = "(";

            for (int i = 0; i < cols; i++)
             {
               int int_val; //Integer variable
               double dbl_val; //double
               string str_val; //string variable

               ENUM_DATABASE_FIELD_TYPE col_type = DatabaseColumnType(request, i);
               string col_name;

               if (!DatabaseColumnName(request, i, col_name))
                {
                  printf("func=%s line=%d, Failed to read database column name. Error = %s", __FUNCTION__, __LINE__, ErrorDescription(GetLastError()));
                  continue;
                }

                 switch (col_type) //Detect a column datatype and assign the value read from every row of that column to the suitable variable
                  {
                     case DATABASE_FIELD_TYPE_INTEGER:
                        if (!DatabaseColumnInteger(request, i, int_val))
                           printf("func=%s line=%d, Failed to read Integer. Error = %s", __FUNCTION__, __LINE__, ErrorDescription(GetLastError()));
                        else
                          {
                             row_string += StringFormat("%d", int_val);
                             row[i] = int_val;
                           }
                        break;

                     case DATABASE_FIELD_TYPE_FLOAT:
                        if (!DatabaseColumnDouble(request, i, dbl_val))
                           printf("func=%s line=%d, Failed to read Double. Error = %s", __FUNCTION__, __LINE__, ErrorDescription(GetLastError()));
                        else
                          {
                            row_string += StringFormat("%.5f", dbl_val);
                            row[i] = dbl_val;
                          }
                        break;

                     case DATABASE_FIELD_TYPE_TEXT:
                        if (!DatabaseColumnText(request, i, str_val))
                           printf("func=%s line=%d, Failed to read Text. Error = %s", __FUNCTION__, __LINE__, ErrorDescription(GetLastError()));
                        else
                          {
                             row_string += "'" + str_val + "'";
                             row[i] = (double)str_val;
                          }
                        break;

                     default:
                             if (MQLInfoInteger(MQL_DEBUG))
                                PrintFormat("%s = <Unknown or Unsupported column Type by this Class>", col_name);
                        break;
                  }

               // Add comma if not last element
               if (i < cols - 1)
                  row_string += ", ";
             }

            row_string += ")";
            if (MQLInfoInteger(MQL_DEBUG))
               Print(row_string);  // Print the full row once

            break;
          }

         DatabaseFinalize(request);
         return row;  // Replace with actual parsed return if needed
      }

 //... Other functions
}
```

From a similar database in MetaEditor.

> ![](https://c.mql5.com/2/153/5261327482042.png)

Below is how we send a query that requests all the information available in the database table, then restricts the amount of information returned to a single row of data.

```
#include <sqlite3.mqh>

CSqlite3 sqlite3;
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
//---

    sqlite3.connect("example.db");
    Print(sqlite3.execute("SELECT * FROM users").fetchone());

    sqlite3.close();
  }
```

Outputs.

```
RJ      0       13:25:04.402    sqlite3 test (XAUUSD,H1)        (1, 'Alice', 30, 'zero@example.com')
DQ      0       13:25:04.402    sqlite3 test (XAUUSD,H1)        [1,0,30,0]
```

Looks great! You can now retrieve a single row of values from the database with minimal hassle. However, the current function ignores all binary values and doesn't handle or encode the strings, thats why, all the "TEXT" or string data types were assigned to zero in a returned vector.

Also, unlike Python arrays, which can hold values of different data types, MQL5 vectors and arrays can't; we'll see how we can deal with this later on.

**2\. The fetchall Method**

Unlike the **fetchone** method, this one receives all information requested in the SQL statement of the execute method.

```
struct execute_res_structure
  {
   int request;
  //... Other functions

   matrix fetchall()
      {
         int cols = DatabaseColumnsCount(request);
         vector row = vector::Zeros(cols);

         int CHUNK_SIZE = 1000; //For optimized matrix handling
         matrix results_matrix = matrix::Zeros(CHUNK_SIZE, cols);
         int rows_found = 0; //for counting the number of rows seen in the database

         while (DatabaseRead(request))  // Essential, read the entire database
          {
            string row_string = "("; //for printing purposes only. Similar to how Python prints

            for (int i = 0; i < cols; i++)
             {
               int int_val; //Integer variable
               double dbl_val; //double variable
               string str_val; //string variable

               ENUM_DATABASE_FIELD_TYPE col_type = DatabaseColumnType(request, i);
               string col_name;

               if (!DatabaseColumnName(request, i, col_name))
                {
                  printf("func=%s line=%d, Failed to read database column name. Error = %s", __FUNCTION__, __LINE__, ErrorDescription(GetLastError()));
                  continue;
                }

                 switch (col_type) //Detect a column datatype and assign the value read from every row of that column to the suitable variable
                  {
                     case DATABASE_FIELD_TYPE_INTEGER:
                        if (!DatabaseColumnInteger(request, i, int_val))
                           printf("func=%s line=%d, Failed to read Integer. Error = %s", __FUNCTION__, __LINE__, ErrorDescription(GetLastError()));
                        else
                          {
                             row_string += StringFormat("%d", int_val);  //For printing purposes only
                             row[i] = int_val;
                           }
                        break;

                     case DATABASE_FIELD_TYPE_FLOAT:
                        if (!DatabaseColumnDouble(request, i, dbl_val))
                           printf("func=%s line=%d, Failed to read Double. Error = %s", __FUNCTION__, __LINE__, ErrorDescription(GetLastError()));
                        else
                          {
                            row_string += StringFormat("%.5f",dbl_val);  //For printing purposes only
                            row[i] = dbl_val;
                          }
                        break;

                     case DATABASE_FIELD_TYPE_TEXT:
                        if (!DatabaseColumnText(request, i, str_val))
                           printf("func=%s line=%d, Failed to read Text. Error = %s", __FUNCTION__, __LINE__, ErrorDescription(GetLastError()));
                        else
                          {
                             row_string += "'" + str_val + "'";
                             row[i] = (double)str_val;
                          }
                        break;

                     default:
                             if (MQLInfoInteger(MQL_DEBUG))
                                PrintFormat("%s = <Unknown or Unsupported column Type by this Class>", col_name);
                        break;
                  }

               // Add comma if not last element
               if (i < cols - 1)
                  row_string += ", ";
             }

          //---

            row_string += ")";
            if (MQLInfoInteger(MQL_DEBUG))
               Print(row_string);  // Print the full row once

          //---

            rows_found++; //Increment the rows counter
            if (rows_found > (int)results_matrix.Rows())
               results_matrix.Resize(results_matrix.Rows()+CHUNK_SIZE, cols); //Resizing the array after 1000 rows

            results_matrix.Row(row, rows_found-1); //Insert a row into the matrix
          }

         results_matrix.Resize(rows_found, cols); //Resize the matrix according to the number of unknown rows found in the database | Final trim

         DatabaseFinalize(request); //Removes a request created in DatabasePrepare().
         return results_matrix;  // return the final matrix
      }

  //... Other lines of code
}
```

This time, we return a [matrix](https://www.mql5.com/en/docs/matrix) instead of a vector to make it possible to accomodate the **entire table** which **is two-dimensional**.

The tricky part in this function is handling the process of dynamically resizing the resulting matrix. Database tables can be huge sometimes (containing 100,000+ rows), and knowing the size of the database or tables in terms of rows is challenging. So, resizing this resulting matrix on every iteration can make this function extremely slow, the more we keep looping through the rows because **the [Resize](https://www.mql5.com/en/docs/matrix/matrix_manipulations/matrix_resize) function** is one of the most computationally expensive functions in MQL5.

This is the reason, I chose to resize the matrix after every 1000 iterations in the above function to reduce the number of times we call the Resize method.

Function usage.

```
#include <sqlite3.mqh>
CSqlite3 sqlite3;
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
//---

    sqlite3.connect("example.db");
    Print(sqlite3.execute("SELECT * FROM users").fetchall());

    sqlite3.close();
  }
```

Outputs in the Experts tab of MetaTrader 5.

```
IF      0       13:30:33.649    sqlite3 test (XAUUSD,H1)        (1, 'Alice', 30, 'zero@example.com')
FR      0       13:30:33.649    sqlite3 test (XAUUSD,H1)        (2, 'Alice', 30, 'alice@example.com')
FD      0       13:30:33.649    sqlite3 test (XAUUSD,H1)        (3, 'Alice', 30, 'bro@example.com')
QQ      0       13:30:33.650    sqlite3 test (XAUUSD,H1)        (4, 'Alice', 30, 'ishowspeed@example.com')
MO      0       13:30:33.650    sqlite3 test (XAUUSD,H1)        (5, 'Alice', 30, 'damn@example.com')
MD      0       13:30:33.650    sqlite3 test (XAUUSD,H1)        (6, 'Alice', 30, 'wth@example.com')
QN      0       13:30:33.650    sqlite3 test (XAUUSD,H1)        (7, 'Bruh', 30, 'stubborn@example.com')
NO      0       13:30:33.650    sqlite3 test (XAUUSD,H1)        (8, 'Bruh', 30, 'whathehelly@example.com')
ED      0       13:30:33.650    sqlite3 test (XAUUSD,H1)        (9, 'Bruh', 30, 'huh@example.com')
PO      0       13:30:33.650    sqlite3 test (XAUUSD,H1)        (10, 'Bruh', 30, 'whatsgoingon@example.com')
FS      0       13:30:33.650    sqlite3 test (XAUUSD,H1)        (11, 'Bruh', 30, 'bruh@example.com')
FF      0       13:30:33.650    sqlite3 test (XAUUSD,H1)        (12, 'Bruh', 30, 'how@example.com')
JO      0       13:30:33.650    sqlite3 test (XAUUSD,H1)        [[1,0,30,0]\
RG      0       13:30:33.650    sqlite3 test (XAUUSD,H1)         [2,0,30,0]\
QN      0       13:30:33.650    sqlite3 test (XAUUSD,H1)         [3,0,30,0]\
LE      0       13:30:33.650    sqlite3 test (XAUUSD,H1)         [4,0,30,0]\
KL      0       13:30:33.650    sqlite3 test (XAUUSD,H1)         [5,0,30,0]\
NS      0       13:30:33.650    sqlite3 test (XAUUSD,H1)         [6,0,30,0]\
MJ      0       13:30:33.650    sqlite3 test (XAUUSD,H1)         [7,0,30,0]\
HQ      0       13:30:33.650    sqlite3 test (XAUUSD,H1)         [8,0,30,0]\
GH      0       13:30:33.650    sqlite3 test (XAUUSD,H1)         [9,0,30,0]\
OL      0       13:30:33.650    sqlite3 test (XAUUSD,H1)         [10,0,30,0]\
RE      0       13:30:33.650    sqlite3 test (XAUUSD,H1)         [11,0,30,0]\
JS      0       13:30:33.650    sqlite3 test (XAUUSD,H1)         [12,0,30,0]]
```

All twelve rows available in the database were returned by the execute function.

Again, MQL5 arrays can't hold non-homogeneous variables in a single array, so we end up with zeros for every row containing strings from the database table. _At least for now._

**3\. The fetchmany method**

Similarly to the fetchall method, this function returns a matrix containing values from the table, but this method gives you control over the number of rows you want to return.

```
struct execute_res_structure
  {
   int request;

//... other lines of code

   matrix fetchmany(uint size)
      {
         int cols = DatabaseColumnsCount(request);
         vector row = vector::Zeros(cols);

         matrix results_matrix = matrix::Zeros(size, cols);
         int rows_found = 0;

         while (DatabaseRead(request))  // Essential, read the entire database
          {
            string row_string = "(";

            for (int i = 0; i < cols; i++)
             {
               int int_val; //Integer variable
               double dbl_val; //double variable
               string str_val; //string variable

               ENUM_DATABASE_FIELD_TYPE col_type = DatabaseColumnType(request, i);
               string col_name;

               if (!DatabaseColumnName(request, i, col_name))
                {
                  printf("func=%s line=%d, Failed to read database column name. Error = %s", __FUNCTION__, __LINE__, ErrorDescription(GetLastError()));
                  continue;
                }

                 switch (col_type) //Detect a column datatype and assign the value read from every row of that column to the suitable variable
                  {
                     case DATABASE_FIELD_TYPE_INTEGER:
                        if (!DatabaseColumnInteger(request, i, int_val))
                           printf("func=%s line=%d, Failed to read Integer. Error = %s", __FUNCTION__, __LINE__, ErrorDescription(GetLastError()));
                        else
                          {
                             row_string += StringFormat("%d", int_val);
                             row[i] = int_val;
                           }
                        break;

                     case DATABASE_FIELD_TYPE_FLOAT:
                        if (!DatabaseColumnDouble(request, i, dbl_val))
                           printf("func=%s line=%d, Failed to read Double. Error = %s", __FUNCTION__, __LINE__, ErrorDescription(GetLastError()));
                        else
                          {
                            row_string += StringFormat("%.5f", dbl_val);
                            row[i] = dbl_val;
                          }
                        break;

                     case DATABASE_FIELD_TYPE_TEXT:
                        if (!DatabaseColumnText(request, i, str_val))
                           printf("func=%s line=%d, Failed to read Text. Error = %s", __FUNCTION__, __LINE__, ErrorDescription(GetLastError()));
                        else
                          {
                             row_string += "'" + str_val + "'";
                             row[i] = (double)str_val;
                          }
                        break;

                     default:
                             if (MQLInfoInteger(MQL_DEBUG))
                                PrintFormat("%s = <Unknown or Unsupported column Type by this Class>", col_name);
                        break;
                  }
             }

            results_matrix.Row(row, rows_found);
            rows_found++;

            if (rows_found >= (int)size)
               break;

            row_string += ")";
            if (MQLInfoInteger(MQL_DEBUG))
               Print(row_string);  // Print the full row once
          }

         results_matrix.Resize(rows_found, cols); //Resize the matrix according to the number of unknown rows found in the database | Final trim

         DatabaseFinalize(request); //Removes a request created in DatabasePrepare().
         return results_matrix;  // return the final matrix
      }

  //... other functions
}
```

Function usage.

```
#include <sqlite3.mqh>
CSqlite3 sqlite3;
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
//---

    sqlite3.connect("example.db");
    Print(sqlite3.execute("SELECT * FROM users").fetchmany(5));

    sqlite3.close();
  }
```

Outputs.

```
EG      0       13:45:25.480    sqlite3 test (XAUUSD,H1)        (1'Alice'30'zero@example.com')
IR      0       13:45:25.481    sqlite3 test (XAUUSD,H1)        (2'Alice'30'alice@example.com')
EJ      0       13:45:25.481    sqlite3 test (XAUUSD,H1)        (3'Alice'30'bro@example.com')
NS      0       13:45:25.481    sqlite3 test (XAUUSD,H1)        (4'Alice'30'ishowspeed@example.com')
CD      0       13:45:25.481    sqlite3 test (XAUUSD,H1)        [[1,0,30,0]\
KR      0       13:45:25.481    sqlite3 test (XAUUSD,H1)         [2,0,30,0]\
LI      0       13:45:25.481    sqlite3 test (XAUUSD,H1)         [3,0,30,0]\
QP      0       13:45:25.481    sqlite3 test (XAUUSD,H1)         [4,0,30,0]\
EJ      0       13:45:25.481    sqlite3 test (XAUUSD,H1)         [5,0,30,0]]
```

Despite these three functions being capable of controlling the amount of data to store in a matrix, they still rely on your SQL statement; it is the one that controls what information to return, in the first place. The above 3 functions discussed are just gateways for receiving the data requested by the SQL statement.

**4\. Additional Method, Checking the execution Function Status**

By default, this function returns a structure full of functions useful for returning the data obtained after a "SELECT" type of SQL statement. However, we might often call the execute method for a non-select type of SQL statement, which makes this function to restrain from returning any data that could be used for checking if it succeeded or not.

To achieve this, we have to use the variable named boolean in the structure returned by the execute function.

```
{
    sqlite3.connect("indicators.db");

    sqlite3.execute(
       "   CREATE TABLE IF NOT EXISTS EURUSD ("
       "   id INTEGER PRIMARY KEY AUTOINCREMENT,"
       "   example_indicator FLOAT,"
       ")"
    );

    if (!sqlite3.execute(StringFormat("INSERT INTO USDJPY (example_indicator) VALUES (%.5f)",(double)rand())).boolean) //A SQL query with a purposefully placed error
      printf("The execute function failed!");
 }
```

_The boolean variable is a bool type of variable, it becomes true if the function succeeded and false if it failed._

Outputs.

```
QK      2       18:27:00.575    sqlite3 test (XAUUSD,H1)        database error, near ")": syntax error
DN      0       18:27:00.576    sqlite3 test (XAUUSD,H1)        func=CSqlite3::execute line=402, Failed to execute a query to the database. Error = Generic database error
GE      2       18:27:00.578    sqlite3 test (XAUUSD,H1)        database error, no such table: USDJPY
PS      0       18:27:00.578    sqlite3 test (XAUUSD,H1)        func=CSqlite3::execute line=402, Failed to execute a query to the database. Error = Generic database error
IS      0       18:27:00.578    sqlite3 test (XAUUSD,H1)        The execute function failed!
```

To check if the function succeeded when it is given with a "SELECT" type of query _that returns some data_, you have to assess the size of the returned matrix or vector.

If they are empty (rows==0 for a matrix and size==0 for a vector); it means the execute function failed.

### Working with Text (string) Values from the Database

As discussed in the previous section, the simplified **execute** function isn't capable of returning string values in a matrix or vector, and we know that; strings or string values can be as useful as other variables so, after getting all values from the database stored in a matrix, you have to extract all columns containing string values separately.

```
#include <sqlite3.mqh>
CSqlite3 sqlite3;
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
//---

    sqlite3.connect("example.db");

    Print("database matrix:\n",sqlite3.execute("SELECT * FROM users").fetchall());

    string name_col[];
    sqlite3.execute("SELECT name FROM users").fetch_column("name", name_col);

    ArrayPrint(name_col);
}
```

Given this array of string values, you can find ways to encode them into variables accepted by the matrix (double, float, etc) and plug them back into it.

Outputs.

```
LS      0       12:48:12.456    sqlite3 test (XAUUSD,H1)        (1, 'Alice', 30, 'zero@example.com')
KG      0       12:48:12.456    sqlite3 test (XAUUSD,H1)        (2, 'Alice', 30, 'alice@example.com')
KH      0       12:48:12.456    sqlite3 test (XAUUSD,H1)        (3, 'Alice', 30, 'bro@example.com')
OM      0       12:48:12.457    sqlite3 test (XAUUSD,H1)        (4, 'Alice', 30, 'ishowspeed@example.com')
GH      0       12:48:12.457    sqlite3 test (XAUUSD,H1)        (5, 'Alice', 30, 'damn@example.com')
KH      0       12:48:12.457    sqlite3 test (XAUUSD,H1)        (6, 'Alice', 30, 'wth@example.com')
OR      0       12:48:12.457    sqlite3 test (XAUUSD,H1)        (7, 'Bruh', 30, 'stubborn@example.com')
LJ      0       12:48:12.457    sqlite3 test (XAUUSD,H1)        (8, 'Bruh', 30, 'whathehelly@example.com')
OG      0       12:48:12.457    sqlite3 test (XAUUSD,H1)        (9, 'Bruh', 30, 'huh@example.com')
RS      0       12:48:12.457    sqlite3 test (XAUUSD,H1)        (10, 'Bruh', 30, 'whatsgoingon@example.com')
PF      0       12:48:12.457    sqlite3 test (XAUUSD,H1)        (11, 'Bruh', 30, 'bruh@example.com')
DS      0       12:48:12.457    sqlite3 test (XAUUSD,H1)        (12, 'Bruh', 30, 'how@example.com')
NL      0       12:48:12.457    sqlite3 test (XAUUSD,H1)        (13, 'John', 83, 'johndoe@example.com')
IE      0       12:48:12.457    sqlite3 test (XAUUSD,H1)        (14, 'John', 83, 'johndoe2@example.com')
KP      0       12:48:12.457    sqlite3 test (XAUUSD,H1)        (15, 'John', 83, 'johndoe3@example.com')
IN      0       12:48:12.457    sqlite3 test (XAUUSD,H1)        (16, 'John', 83, 'johndoe4@example.com')
KD      0       12:48:12.457    sqlite3 test (XAUUSD,H1)        database matrix:
HP      0       12:48:12.457    sqlite3 test (XAUUSD,H1)        [[1,0,30,0]\
PF      0       12:48:12.457    sqlite3 test (XAUUSD,H1)         [2,0,30,0]\
OM      0       12:48:12.457    sqlite3 test (XAUUSD,H1)         [3,0,30,0]\
ND      0       12:48:12.457    sqlite3 test (XAUUSD,H1)         [4,0,30,0]\
MK      0       12:48:12.457    sqlite3 test (XAUUSD,H1)         [5,0,30,0]\
LR      0       12:48:12.457    sqlite3 test (XAUUSD,H1)         [6,0,30,0]\
KI      0       12:48:12.457    sqlite3 test (XAUUSD,H1)         [7,0,30,0]\
JP      0       12:48:12.457    sqlite3 test (XAUUSD,H1)         [8,0,30,0]\
IG      0       12:48:12.457    sqlite3 test (XAUUSD,H1)         [9,0,30,0]\
QM      0       12:48:12.457    sqlite3 test (XAUUSD,H1)         [10,0,30,0]\
LF      0       12:48:12.457    sqlite3 test (XAUUSD,H1)         [11,0,30,0]\
KO      0       12:48:12.457    sqlite3 test (XAUUSD,H1)         [12,0,30,0]\
RP      0       12:48:12.457    sqlite3 test (XAUUSD,H1)         [13,0,83,0]\
MI      0       12:48:12.457    sqlite3 test (XAUUSD,H1)         [14,0,83,0]\
PR      0       12:48:12.457    sqlite3 test (XAUUSD,H1)         [15,0,83,0]\
DF      0       12:48:12.457    sqlite3 test (XAUUSD,H1)         [16,0,83,0]]
LQ      0       12:48:12.458    sqlite3 test (XAUUSD,H1)        "Alice" "Alice" "Alice" "Alice" "Alice" "Alice" "Bruh"  "Bruh"  "Bruh"  "Bruh"  "Bruh"  "Bruh"  "John"  "John"  "John"  "John"
```

### Database Transactions Control

The sqlite3 module in Python offers multiple methods of controlling whether, when, and how database transactions are opened and closed. We can adapt to the same format in our class, making it easier to handle transactions when working with SQLite databases.

| Function | Description | Notes |
| --- | --- | --- |
| ```<br>bool CSqlite3::commit(void)<br>``` | Commits the current transaction, making all changes to the database permanent. | It is required after **INSERT**, **UPDATE**, or **DELETE** types of SQL queries if **autocommit** is set to **false**. |
| ```<br>bool CSqlite3::rollback(void) <br>``` | Rolls back the current transaction, undoing all uncommitted changes. | Useful for error handling. |
| ```<br>bool CSqlite3::begin(void)<br>``` | Begins the transaction that will later be committed to the database. | More explicit syntax is used before making any changes to the database. The execute() function calls it automatically when autocommit is set to true. |
| ```<br>bool CSqlite3::in_transaction()<br>``` | A boolean function for checking if the transaction is active or not. | It returns true if the transaction is currently active. |
| ```<br>CSqlite3(bool autocommit=false)<br>``` |  | Optionally — not recommended, you can modify the **autocommit** value inside a class constructor to enable or disable automatically committing of the transaction inside the **execute** function. |

The three functions (begin, rollback, and commit) are built on top of MQL5's built-in functions for handling database transactions.

```
bool CSqlite3::commit(void)
 {
   if (!DatabaseTransactionCommit(m_db_handle))
      {
         printf("func=%s line=%d, Failed to commit a transaction. Error = %s",__FUNCTION__,__LINE__,ErrorDescription(GetLastError()));
         return false;
      }

   m_transaction_active = false; //Reset the transaction after commit
   m_transaction_active_auto = false;

   return true;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CSqlite3::begin(void)
  {
   if (m_transaction_active)
     {
       if (!m_transaction_active_auto) //print only if the user started the transaction not when it was started automatically by the execute() function
         printf("Can not begin, already in a transaction. Call the function rollback() to disregard it, or commit() to save the changes");
       return false;
     }

//---

   if (!DatabaseTransactionBegin(m_db_handle))
      {
         printf("func=%s line=%d, Failed to begin a transaction. Error = %s",__FUNCTION__,__LINE__,ErrorDescription(GetLastError()));
         return false;
      }

   m_transaction_active = true;
   m_transaction_active_auto = false;

   return m_transaction_active;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CSqlite3::rollback(void)
 {
   if (!DatabaseTransactionRollback(m_db_handle))
      {
         printf("func=%s line=%d, Failed to rollback a transaction. Error = %s",__FUNCTION__,__LINE__,ErrorDescription(GetLastError()));
         return false;
      }

   m_transaction_active = false; //Reset the transaction after rollback
   m_transaction_active_auto = false;

   return true;
 }
```

### Other Methods in the sqlite3 Module

These are less-used functions from this module, but they are still handy for various tasks.

**1\. The executemany Method**

This function grants you the ability to insert many rows into a database table in a single function call.

Suppose you have a matrix containing multiple rows, each column containing a specific types of indicator values that you want to insert into the database at once — this is a "goto" function.

```
#include <sqlite3.mqh>
CSqlite3 sqlite3;
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
//---

    sqlite3.connect("indicators.db");

    sqlite3.execute(
       "   CREATE TABLE IF NOT EXISTS EURUSD ("
       "   id INTEGER PRIMARY KEY AUTOINCREMENT,"
       "   INDICATOR01 FLOAT,"
       "   INDICATOR02 FLOAT,"
       "   INDICATOR03 FLOAT"
       ")"
    );

    matrix data = {{101, 25, 001},
                   {102, 32, 002},
                   {103, 29, 003}};

    sqlite3.executemany("INSERT INTO EURUSD (INDICATOR01, INDICATOR02, INDICATOR03) VALUES (?,?,?)", data);
    sqlite3.commit();
    sqlite3.close();
  }
```

Outputs.

![](https://c.mql5.com/2/153/3039934650933.png)

The number of question marks after the keyword **VALUES**  must be equal to the number of columns in the params matrix for this function to work without throwing errors.

Keep in mind that due to matrix limitations to homogeneous variables within a single matrix, you can not add different data types at once to your database using this function, e.g,. You can't insert a string and a double column into the database at the same time.

**2\. The executescript Method**

This function is handy when you want to execute multiple SQL statements in one go. For example, create tables, insert rows, etc., in one statement. The following are key features of this function.

```
void OnStart()
  {
//---
     sqlite3.connect("indicators.db");

     // Use executescript to log actions
     sqlite3.executescript(
              "CREATE TABLE IF NOT EXISTS logs ("
              "id INTEGER PRIMARY KEY AUTOINCREMENT,"
              "event TEXT NOT NULL"
          ");"

          "INSERT INTO logs (event) VALUES ('Users batch inserted');"
      );

     sqlite3.close();
  }
```

- It takes a string of multiple SQL statements separated by semicolons ";"
- It executes them all at once, not parameterized, and not parsed like execute() or executemany().
- It doesn’t return results, just executes the batch of commands.

Outputs.

![](https://c.mql5.com/2/153/460668556250.png)

The **executescript** function automatically commits any open transactions, so you usually don’t need to explicitly call the **commit**  function unless needed for control.

**3\. Additional Method, the print\_table Method**

To get the functionality similar to **cursor.description** offered by the sqlite3 module in Python, which returns the structure of a SQL query, database, or a table.

We can use MQL5 built-in function named [DatabasePrint](https://www.mql5.com/en/docs/database/databaseprint) to get a similar outcome.

```
void CSqlite3::print_table(const string table_name_or_sql, const int flags=0) // Prints a table or an SQL request execution result in the Experts journal.
  {
    if (DatabasePrint(m_db_handle, table_name_or_sql, flags)<0)
       printf("func=%s line=%d, Failed to print the table or query result. Error = %s",__FUNCTION__,__LINE__, ErrorDescription(GetLastError()));
  }
```

Usage.

```
sqlite3.print_table("SELECT * FROM users");
```

Outputs.

```
CM      0       13:17:19.028    sqlite3 test (XAUUSD,H1)         #| id name  age email
PJ      0       13:17:19.028    sqlite3 test (XAUUSD,H1)        --+--------------------------------------
MH      0       13:17:19.028    sqlite3 test (XAUUSD,H1)         1|  1 Alice  30 zero@example.com
KS      0       13:17:19.028    sqlite3 test (XAUUSD,H1)         2|  2 Alice  30 alice@example.com
NO      0       13:17:19.028    sqlite3 test (XAUUSD,H1)         3|  3 Alice  30 bro@example.com
NI      0       13:17:19.028    sqlite3 test (XAUUSD,H1)         4|  4 Alice  30 ishowspeed@example.com
IL      0       13:17:19.028    sqlite3 test (XAUUSD,H1)         5|  5 Alice  30 damn@example.com
LO      0       13:17:19.028    sqlite3 test (XAUUSD,H1)         6|  6 Alice  30 wth@example.com
EE      0       13:17:19.028    sqlite3 test (XAUUSD,H1)         7|  7 Bruh   30 stubborn@example.com
EM      0       13:17:19.028    sqlite3 test (XAUUSD,H1)         8|  8 Bruh   30 whathehelly@example.com
ER      0       13:17:19.028    sqlite3 test (XAUUSD,H1)         9|  9 Bruh   30 huh@example.com
HK      0       13:17:19.028    sqlite3 test (XAUUSD,H1)        10| 10 Bruh   30 whatsgoingon@example.com
IQ      0       13:17:19.028    sqlite3 test (XAUUSD,H1)        11| 11 Bruh   30 bruh@example.com
LQ      0       13:17:19.028    sqlite3 test (XAUUSD,H1)        12| 12 Bruh   30 how@example.com
GG      0       13:17:19.028    sqlite3 test (XAUUSD,H1)        13| 13 John   83 johndoe@example.com
GK      0       13:17:19.028    sqlite3 test (XAUUSD,H1)        14| 14 John   83 johndoe2@example.com
NQ      0       13:17:19.028    sqlite3 test (XAUUSD,H1)        15| 15 John   83 johndoe3@example.com
QF      0       13:17:19.028    sqlite3 test (XAUUSD,H1)        16| 16 John   83 johndoe4@example.com
```

### Logging Trades to the Database

We've seen some simple examples on how you can execute SQL statements, add some values to the database, and much more. Let us do something meaningful at last; insert all trades in MetaTrader 5 history into a SQLite database.

This example taken from [this article](https://www.mql5.com/en/articles/7463#transactions_speedup).

**Without sqlite3.**

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

**With sqlite3.**

```
#include <sqlite3.mqh>
CSqlite3 sqlite3;
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
//---
    sqlite3.connect("Trades_database.db");

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

   HistorySelect(0, TimeCurrent());
   int deals=HistoryDealsTotal();

   sqlite3.execute("CREATE TABLE IF NOT EXISTS DEALS ("
                     "ID          INT KEY NOT NULL,"
                     "ORDER_ID    INT     NOT NULL,"
                     "POSITION_ID INT     NOT NULL,"
                     "TIME        INT     NOT NULL,"
                     "TYPE        INT     NOT NULL,"
                     "ENTRY       INT     NOT NULL,"
                     "SYMBOL      CHAR(10),"
                     "VOLUME      REAL,"
                     "PRICE       REAL,"
                     "PROFIT      REAL,"
                     "SWAP        REAL,"
                     "COMMISSION  REAL,"
                     "MAGIC       INT,"
                     "REASON      INT );"
   ); //Creates a table if it doesn't exist

   sqlite3.begin(); //Start the transaction

// --- lock the database before executing transactions

   for(int i=0; i<deals; i++) //loop through all deals
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

      sqlite3.execute(request_text);
     }

    sqlite3.commit(); //Commit all deals to the database at once
    sqlite3.close(); //close the database
  }
```

Outcome.

![](https://c.mql5.com/2/153/3269356229342.gif)

### Final Thoughts

Recreating Python’s sqlite3 module in MQL5 was a rewarding challenge that highlights both the flexibility and the limitations of MQL5 when compared to Python. While MQL5 lacks the built-in support for advanced features like authorizers or context managers for a SQLite database, with careful abstraction and method design, one can achieve a very similar developer experience.

This custom CSqlite3 class now allows MQL5 developers to interact with SQLite databases in a structured, Pythonic way — complete with support for queries, transactions, commit/rollback control, and fetch operations like fetchone(), fetchmany(), fetchall(), etc,.

_If you're coming from Python, this module should feel familiar and hopefully, enjoyable to use._

Peace out.

**Attachments Table**

| Filename | Description & Usage |
| --- | --- |
| Include\\errordescription.mqh | Contains descriptions of all error codes produced by MQL5 and MetaTrader 5 |
| Include\\sqlite3.mqh | Contains the CSqlite3 class for working with SQLite databases in a Python-like way. |
| Scripts\\sqlite3 test.mq5 | A script for testing the CSqlite3 class. |

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/18640.zip "Download all attachments in the single ZIP archive")

[Attachments.zip](https://www.mql5.com/en/articles/download/18640/attachments.zip "Download Attachments.zip")(12.89 KB)

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

**[Go to discussion](https://www.mql5.com/en/forum/490294)**

![Developing a multi-currency Expert Advisor (Part 20): Putting in order the conveyor of automatic project optimization stages (I)](https://c.mql5.com/2/102/Developing_a_multi-currency_advisor_Part_20___LOGO.png)[Developing a multi-currency Expert Advisor (Part 20): Putting in order the conveyor of automatic project optimization stages (I)](https://www.mql5.com/en/articles/16134)

We have already created quite a few components that help arrange auto optimization. During the creation, we followed the traditional cyclical structure: from creating minimal working code to refactoring and obtaining improved code. It is time to start clearing up our database, which is also a key component in the system we are creating.

![Price Action Analysis Toolkit Development (Part 30): Commodity Channel Index (CCI), Zero Line EA](https://c.mql5.com/2/153/18551-price-action-analysis-toolkit-logo.png)[Price Action Analysis Toolkit Development (Part 30): Commodity Channel Index (CCI), Zero Line EA](https://www.mql5.com/en/articles/18551)

Automating price action analysis is the way forward. In this article, we utilize the Dual CCI indicator, the Zero Line Crossover strategy, EMA, and price action to develop a tool that generates trade signals and sets stop-loss (SL) and take-profit (TP) levels using ATR. Please read this article to learn how we approach the development of the CCI Zero Line EA.

![Using association rules in Forex data analysis](https://c.mql5.com/2/102/Using_Association_Rules_to_Analyze_Forex_Data___LOGO.png)[Using association rules in Forex data analysis](https://www.mql5.com/en/articles/16061)

How to apply predictive rules of supermarket retail analytics to the real Forex market? How are purchases of cookies, milk and bread related to stock exchange transactions? The article discusses an innovative approach to algorithmic trading based on the use of association rules.

![MQL5 Wizard Techniques you should know (Part 72): Using Patterns of MACD and the OBV with Supervised Learning](https://c.mql5.com/2/153/18697-mql5-wizard-techniques-you-logo.png)[MQL5 Wizard Techniques you should know (Part 72): Using Patterns of MACD and the OBV with Supervised Learning](https://www.mql5.com/en/articles/18697)

We follow up on our last article, where we introduced the indicator pair of the MACD and the OBV, by looking at how this pairing could be enhanced with Machine Learning. MACD and OBV are a trend and volume complimentary pairing. Our machine learning approach uses a convolution neural network that engages the Exponential kernel in sizing its kernels and channels, when fine-tuning the forecasts of this indicator pairing. As always, this is done in a custom signal class file that works with the MQL5 wizard to assemble an Expert Advisor.

[![](https://www.mql5.com/ff/si/m0dtjf9x3brdz07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fmarket%2Fmt5%2Fexpert%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dtop.experts%26utm_content%3Dbuy.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=widauvjabtsckwovwaperzkotrcrttvb&s=25ef75d39331f608a319410bf27ff02c1bd7986622ecc1eec8968a650f044731&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=lfqedelripabuwovornxgmxgtplmkxax&ssn=1769092441050837909&ssn_dr=0&ssn_sr=0&fv_date=1769092441&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F18640&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Implementing%20Practical%20Modules%20from%20Other%20Languages%20in%20MQL5%20(Part%2001)%3A%20Building%20the%20SQLite3%20Library%2C%20Inspired%20by%20Python%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176909244178566703&fz_uniq=5049220552037607257&sv=2552)

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