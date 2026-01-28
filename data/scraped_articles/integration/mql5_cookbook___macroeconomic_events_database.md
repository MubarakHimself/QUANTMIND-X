---
title: MQL5 Cookbook — Macroeconomic events database
url: https://www.mql5.com/en/articles/11977
categories: Integration
relevance_score: 3
scraped_at: 2026-01-23T21:11:45.619687
---

[![](https://www.mql5.com/ff/si/fx5m8s6u6uxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=ysducdhemkrdsdtzzbfkclolrllnhezk&s=33f180a31db6c3b846d77732b0bc78169421a47b8cf9f076ca717f4e4846d1c7&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=lzxipeatbjogkmmydvndlzbzvdtfxzbs&ssn=1769191903075289342&ssn_dr=0&ssn_sr=0&fv_date=1769191903&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F11977&back_ref=https%3A%2F%2Fwww.google.com%2F&title=MQL5%20Cookbook%20%E2%80%94%20Macroeconomic%20events%20database%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176919190386454397&fz_uniq=5071646272802204639&sv=2552)

MetaTrader 5 / Examples


### Introduction

This article will focus on how to group and manage data that describes macroeconomic calendar events.

In the modern world, information flows are all-pervasive. So, one has to deal with big data when analyzing events. Although, to a greater extent, the article covers issues related not to content, but to form, nevertheless, it seems that the correct organization and structuring of data contributes a lot to the fact that these data will turn into information.

We will solve these tasks by means of SQLite. The developer has added support for handling SQLite directly from MQL5 [in build 2265](https://www.metatrader5.com/en/releasenotes/terminal/2117 "https://www.metatrader5.com/en/releasenotes/terminal/2117") (December 6, 2019). Before that, I had to use various connectors, for example, as described in the article [SQL and MQL5: Working with SQLite database](https://www.mql5.com/en/articles/862).

### 1. Documentation and additional materials

Let's skim through [Documentation](https://www.mql5.com/en/docs/database), namely the sections about database handling. The developer provides 26 native features:

01. DatabaseOpen();
02. DatabaseClose();
03. DatabaseImport();
04. DatabaseExport();
05. DatabasePrint();
06. DatabaseTableExists();
07. DatabaseExecute();
08. DatabasePrepare();
09. DatabaseReset();
10. DatabaseBind();
11. DatabaseBindArray();
12. DatabaseRead();
13. DatabaseReadBind();
14. DatabaseFinalize();
15. DatabaseTransactionBegin();
16. DatabaseTransactionCommit();
17. DatabaseTransactionRollback();
18. DatabaseColumnsCount();
19. DatabaseColumnName();
20. DatabaseColumnType();
21. DatabaseColumnSize();
22. DatabaseColumnText();
23. DatabaseColumnInteger();
24. DatabaseColumnLong();
25. DatabaseColumnDouble();
26. DatabaseColumnBlob().

There are also blocks of statistical and mathematical functions that were added recently. The article [SQLite: Native handling of SQL databases in MQL5](https://www.mql5.com/en/articles/7463) can serve as a starting point for studying this functionality.

### 2\. CDatabase class

Let's create the CDatabase class for the convenience of handling databases. First, describe the class composition. Then check its operation using examples.

The data members of the CDatabase class include the following:

- m\_name  - database file name (with the extension);
- m\_handle - database handle;
- m\_flags - combination of flags;
- m\_table\_names – table names;
- m\_curr\_table\_name – current table name;
- m\_sql\_request\_ha – last SQL query handle;
- m\_sql\_request – last SQL query.

As for the methods, I would divide them into several groups:

1. Methods that include native functions for handling databases (API MQL5 functions);
2. Methods for handling tables;
3. Methods for handling requests;
4. Methods for working with views;
5. Methods for obtaining data member values (get methods).

[SQLite](https://www.mql5.com/go?link=https://www.sqlite.org/index.html "https://www.sqlite.org/index.html") features multiple request forms that can be both simple and complex. My objective is not to create a custom method in the CDatabase class for each such form.  If the class does not have a method for a particular request, then the request can be formed using the universal CDatabase::Select() method.

Now let's look at the examples of using CDatabase class features.

_2.1 Creating a database_

Let's create our first calendar database using the **_1\_create\_calendar\_db._ _mq5_** script.  The script is to have only a few lines of code.

```
//--- include
#include "..\CDatabase.mqh"
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
   CDatabase db_obj;
   string file_name="Databases\\test_calendar_db.sqlite";
   uint flags=DATABASE_OPEN_READWRITE | DATABASE_OPEN_CREATE;
   if(!db_obj.Open(file_name, flags))
      ::PrintFormat("Failed to create a calendar database \"%s\"!", file_name);
   db_obj.Close();
  }
//+------------------------------------------------------------------+
```

After running the script, we will see that the database file test\_calendar\_db.sqlite has appeared in %MQL5\\Files\\Databases (Fig. 1).

![test_calendar_db.sqlite database file](https://c.mql5.com/2/51/1__20.png)

_Fig. 1. test\_calendar\_db.sqlite database file_

If we open this file in the code editor, we will see that the database is empty (Fig. 2).

![test_calendar_db database](https://c.mql5.com/2/51/2__19.png)

_Fig. 2. _test\_calendar\_db_ database_

_2.2 Creating a table_

Let's try to fill in the database. To do this, create the COUNTRIES table, in which we will enter a list of countries whose calendar events will subsequently be processed by our queries. The _**2\_**_ **_create\__ _countries\__ _table._ _mq5_** script will do the job.

```
//--- include
#include "..\CDatabase.mqh"
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
   {
//--- open a database
   CDatabase db_obj;
   string file_name="Databases\\test_calendar_db.sqlite";
   uint flags=DATABASE_OPEN_READWRITE;
   if(!db_obj.Open(file_name, flags))
      {
      ::PrintFormat("Failed to open a calendar database \"%s\"!", file_name);
      db_obj.Close();
      return;
      }
//--- create a table
   string table_name="COUNTRIES";
   string params[]=
      {
      "COUNTRY_ID UNSIGNED BIG INT PRIMARY KEY NOT NULL,", // 1) country ID
      "NAME TEXT,"                                         // 2) country name
      "CODE TEXT,"                                         // 3) country code
      "CONTINENT TEXT,"                                    // 4) country continent
      "CURRENCY TEXT,"                                     // 5) currency code
      "CURRENCY_SYMBOL TEXT,"                              // 6) currency symbol
      "URL_NAME TEXT"                                      // 7) country URL
      };
   if(!db_obj.CreateTable(table_name, params))
      {
      ::PrintFormat("Failed to create a table \"%s\"!", table_name);
      db_obj.Close();
      return;
      }
   db_obj.Close();
   }
//+------------------------------------------------------------------+
```

After running the script, we can find that the COUNTRIES table has appeared in the database (Fig. 3).

![Empty COUNTRIES table](https://c.mql5.com/2/51/3__20.png)

_Fig. 3. Empty COUNTRIES table_

_2.3 Filling in the table_

Let's populate a new table with data. To do this, use the CiCalendarInfo class features. Find more details about the class in the article [MQL5 Cookbook – Economic Calendar](https://www.mql5.com/en/articles/9874). The task is performed by the **3\_ _fill\__ _in\__ _countries\__ _table._ _mq5_** script.

```
//--- include
#include "..\CalendarInfo.mqh"
#include "..\CDatabase.mqh"
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
   {
//--- open a database
   CDatabase db_obj;
   string file_name="Databases\\test_calendar_db.sqlite";
   uint flags=DATABASE_OPEN_READWRITE;
   if(!db_obj.Open(file_name, flags))
      {
      db_obj.Close();
      return;
      }
//--- open a table
   string table_name="COUNTRIES";
   if(db_obj.SelectTable(table_name))
      if(db_obj.EmptyTable())
         {
         db_obj.FinalizeSqlRequest();
         string col_names[]=
            {
            "COUNTRY_ID", "NAME", "CODE", "CONTINENT",
            "CURRENCY", "CURRENCY_SYMBOL", "URL_NAME"
            };
//--- fill in the table
         CiCalendarInfo calendar_info;
         if(calendar_info.Init())
            {
            MqlCalendarCountry calendar_countries[];
            if(calendar_info.GetCountries(calendar_countries))
               {
               if(db_obj.TransactionBegin())
                  for(int c_idx=0; c_idx<::ArraySize(calendar_countries); c_idx++)
                     {
                     MqlCalendarCountry curr_country=calendar_countries[c_idx];
                     string col_vals[];
                     ::ArrayResize(col_vals, 7);
                     col_vals[0]=::StringFormat("%I64u", curr_country.id);
                     col_vals[1]=::StringFormat("'%s'", curr_country.name);
                     col_vals[2]=::StringFormat("'%s'", curr_country.code);
                     col_vals[3]="NULL";
                     SCountryByContinent curr_country_continent_data;
                     if(curr_country_continent_data.Init(curr_country.code))
                        col_vals[3]=::StringFormat("'%s'",
                                                   curr_country_continent_data.ContinentDescription());
                     col_vals[4]=::StringFormat("'%s'", curr_country.currency);
                     col_vals[5]=::StringFormat("'%s'", curr_country.currency_symbol);
                     col_vals[6]=::StringFormat("'%s'", curr_country.url_name);
                     if(!db_obj.InsertSingleRow(col_names, col_vals))
                        {
                        db_obj.TransactionRollback();
                        db_obj.Close();
                        return;
                        }
                     db_obj.FinalizeSqlRequest();
                     }
               if(!db_obj.TransactionCommit())
                  ::PrintFormat("Failed to complete transaction execution, error %d", ::GetLastError());
               }
            //--- print
            if(db_obj.PrintTable()<0)
               ::PrintFormat("Failed to print the table \"%s\", error %d", table_name, ::GetLastError());
            }
         }
   db_obj.Close();
   }
//+------------------------------------------------------------------+
```

Print out the COUNTRIES table data in the log.

```
3_fill_in_countries_table (EURUSD,H1)    #| COUNTRY_ID NAME           CODE CONTINENT         CURRENCY CURRENCY_SYMBOL URL_NAME
3_fill_in_countries_table (EURUSD,H1)   --+-----------------------------------------------------------------------------------------
3_fill_in_countries_table (EURUSD,H1)    1|        554 New Zealand    NZ   Australia/Oceania NZD      $               new-zealand
3_fill_in_countries_table (EURUSD,H1)    2|        999 European Union EU   Europe            EUR      €               european-union
3_fill_in_countries_table (EURUSD,H1)    3|        392 Japan          JP   Asia              JPY      ¥               japan
3_fill_in_countries_table (EURUSD,H1)    4|        124 Canada         CA   North America     CAD      $               canada
3_fill_in_countries_table (EURUSD,H1)    5|         36 Australia      AU   Australia/Oceania AUD      $               australia
3_fill_in_countries_table (EURUSD,H1)    6|        156 China          CN   Asia              CNY      ¥               china
3_fill_in_countries_table (EURUSD,H1)    7|        380 Italy          IT   Europe            EUR      €               italy
3_fill_in_countries_table (EURUSD,H1)    8|        702 Singapore      SG   Asia              SGD      R$              singapore
3_fill_in_countries_table (EURUSD,H1)    9|        276 Germany        DE   Europe            EUR      €               germany
3_fill_in_countries_table (EURUSD,H1)   10|        250 France         FR   Europe            EUR      €               france
3_fill_in_countries_table (EURUSD,H1)   11|         76 Brazil         BR   South America     BRL      R$              brazil
3_fill_in_countries_table (EURUSD,H1)   12|        484 Mexico         MX   North America     MXN      Mex$            mexico
3_fill_in_countries_table (EURUSD,H1)   13|        710 South Africa   ZA   Africa            ZAR      R               south-africa
3_fill_in_countries_table (EURUSD,H1)   14|        344 Hong Kong      HK   Asia              HKD      HK$             hong-kong
3_fill_in_countries_table (EURUSD,H1)   15|        356 India          IN   Asia              INR      ₹               india
3_fill_in_countries_table (EURUSD,H1)   16|        578 Norway         NO   Europe            NOK      Kr              norway
3_fill_in_countries_table (EURUSD,H1)   17|        840 United States  US   North America     USD      $               united-states
3_fill_in_countries_table (EURUSD,H1)   18|        826 United Kingdom GB   Europe            GBP      £               united-kingdom
3_fill_in_countries_table (EURUSD,H1)   19|        756 Switzerland    CH   Europe            CHF      ₣               switzerland
3_fill_in_countries_table (EURUSD,H1)   20|        410 South Korea    KR   Asia              KRW      ₩               south-korea
3_fill_in_countries_table (EURUSD,H1)   21|        724 Spain          ES   Europe            EUR      €               spain
3_fill_in_countries_table (EURUSD,H1)   22|        752 Sweden         SE   Europe            SEK      Kr              sweden
3_fill_in_countries_table (EURUSD,H1)   23|          0 Worldwide      WW   World             ALL                      worldwide
```

In MetaEditor, the table looks like this (Fig. 4).

![Filled COUNTRIES table](https://c.mql5.com/2/51/4__7.png)

_Fig. 4. Filled COUNTRIES table_

_2.4 Selecting some of table columns_

Let's handle the COUNTRIES table data. Suppose that we want to select the following columns:

- "COUNTRY\_ID";
- "COUNTRY\_NAME";
- "COUNTRY\_CODE";
- "COUNTRY\_CONTINENT";
- "CURRENCY".

Create an SQL query using the _**4\_select\_some\_columns.mq5**_ script the following way:

```
//--- include
#include "..\CDatabase.mqh"
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
   {
//--- open a database
   CDatabase db_obj;
   string file_name="Databases\\test_calendar_db.sqlite";
   uint flags=DATABASE_OPEN_READONLY;
   if(!db_obj.Open(file_name, flags))
      {
      db_obj.Close();
      return;
      }
//--- check a table
   string table_name="COUNTRIES";
   if(db_obj.SelectTable(table_name))
      {
      string col_names_to_select[]=
         {
         "COUNTRY_ID", "NAME", "CODE",
         "CONTINENT", "CURRENCY"
         };
      if(!db_obj.SelectFrom(col_names_to_select))
         {
         db_obj.Close();
         return;
         }
      //--- print the SQL request
      if(db_obj.PrintSqlRequest()<0)
         ::PrintFormat("Failed to print the SQL request, error %d", ::GetLastError());
      db_obj.FinalizeSqlRequest();
      }
   db_obj.Close();
   }
//+------------------------------------------------------------------+
```

When printing out the query, we get:

```
4_select_some_columns (EURUSD,H1)        #| COUNTRY_ID NAME           CODE CONTINENT         CURRENCY
4_select_some_columns (EURUSD,H1)       --+----------------------------------------------------------
4_select_some_columns (EURUSD,H1)        1|        554 New Zealand    NZ   Australia/Oceania NZD
4_select_some_columns (EURUSD,H1)        2|        999 European Union EU   Europe            EUR
4_select_some_columns (EURUSD,H1)        3|        392 Japan          JP   Asia              JPY
4_select_some_columns (EURUSD,H1)        4|        124 Canada         CA   North America     CAD
4_select_some_columns (EURUSD,H1)        5|         36 Australia      AU   Australia/Oceania AUD
4_select_some_columns (EURUSD,H1)        6|        156 China          CN   Asia              CNY
4_select_some_columns (EURUSD,H1)        7|        380 Italy          IT   Europe            EUR
4_select_some_columns (EURUSD,H1)        8|        702 Singapore      SG   Asia              SGD
4_select_some_columns (EURUSD,H1)        9|        276 Germany        DE   Europe            EUR
4_select_some_columns (EURUSD,H1)       10|        250 France         FR   Europe            EUR
4_select_some_columns (EURUSD,H1)       11|         76 Brazil         BR   South America     BRL
4_select_some_columns (EURUSD,H1)       12|        484 Mexico         MX   North America     MXN
4_select_some_columns (EURUSD,H1)       13|        710 South Africa   ZA   Africa            ZAR
4_select_some_columns (EURUSD,H1)       14|        344 Hong Kong      HK   Asia              HKD
4_select_some_columns (EURUSD,H1)       15|        356 India          IN   Asia              INR
4_select_some_columns (EURUSD,H1)       16|        578 Norway         NO   Europe            NOK
4_select_some_columns (EURUSD,H1)       17|        840 United States  US   North America     USD
4_select_some_columns (EURUSD,H1)       18|        826 United Kingdom GB   Europe            GBP
4_select_some_columns (EURUSD,H1)       19|        756 Switzerland    CH   Europe            CHF
4_select_some_columns (EURUSD,H1)       20|        410 South Korea    KR   Asia              KRW
4_select_some_columns (EURUSD,H1)       21|        724 Spain          ES   Europe            EUR
4_select_some_columns (EURUSD,H1)       22|        752 Sweden         SE   Europe            SEK
4_select_some_columns (EURUSD,H1)       23|          0 Worldwide      WW   World             ALL
```

Obviously, the selection was made without any sorting.

_2.5 Selecting some of the sorted table columns_

Let's sort the data in the table by the "COUNTRY\_ID" column. This request has the following implementation in the **_5\_select\_some\_sorted\_columns._ _mq5_** script:

```
//--- include
#include "..\CDatabase.mqh"
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
   {
//--- open a database
   CDatabase db_obj;
   string file_name="Databases\\test_calendar_db.sqlite";
   uint flags=DATABASE_OPEN_READONLY;
   if(!db_obj.Open(file_name, flags))
      {
      db_obj.Close();
      return;
      }
//--- check a table
   string table_name="COUNTRIES";
   if(db_obj.SelectTable(table_name))
      {
      string col_names_to_select[]=
         {
         "COUNTRY_ID", "NAME", "CODE",
         "CONTINENT", "CURRENCY"
         };
      string ord_names[1];
      ord_names[0]=col_names_to_select[0];
      if(!db_obj.SelectFromOrderedBy(col_names_to_select, ord_names))
         {
         db_obj.Close();
         return;
         }
      //--- print the SQL request
      if(db_obj.PrintSqlRequest()<0)
         ::PrintFormat("Failed to print the SQL request, error %d", ::GetLastError());
      db_obj.FinalizeSqlRequest();
      }
   db_obj.Close();
   }
//+------------------------------------------------------------------+
```

The result of the query execution appears in the log:

```
5_select_some_sorted_columns (EURUSD,H1)         #| COUNTRY_ID NAME           CODE CONTINENT         CURRENCY
5_select_some_sorted_columns (EURUSD,H1)        --+----------------------------------------------------------
5_select_some_sorted_columns (EURUSD,H1)         1|          0 Worldwide      WW   World             ALL
5_select_some_sorted_columns (EURUSD,H1)         2|         36 Australia      AU   Australia/Oceania AUD
5_select_some_sorted_columns (EURUSD,H1)         3|         76 Brazil         BR   South America     BRL
5_select_some_sorted_columns (EURUSD,H1)         4|        124 Canada         CA   North America     CAD
5_select_some_sorted_columns (EURUSD,H1)         5|        156 China          CN   Asia              CNY
5_select_some_sorted_columns (EURUSD,H1)         6|        250 France         FR   Europe            EUR
5_select_some_sorted_columns (EURUSD,H1)         7|        276 Germany        DE   Europe            EUR
5_select_some_sorted_columns (EURUSD,H1)         8|        344 Hong Kong      HK   Asia              HKD
5_select_some_sorted_columns (EURUSD,H1)         9|        356 India          IN   Asia              INR
5_select_some_sorted_columns (EURUSD,H1)        10|        380 Italy          IT   Europe            EUR
5_select_some_sorted_columns (EURUSD,H1)        11|        392 Japan          JP   Asia              JPY
5_select_some_sorted_columns (EURUSD,H1)        12|        410 South Korea    KR   Asia              KRW
5_select_some_sorted_columns (EURUSD,H1)        13|        484 Mexico         MX   North America     MXN
5_select_some_sorted_columns (EURUSD,H1)        14|        554 New Zealand    NZ   Australia/Oceania NZD
5_select_some_sorted_columns (EURUSD,H1)        15|        578 Norway         NO   Europe            NOK
5_select_some_sorted_columns (EURUSD,H1)        16|        702 Singapore      SG   Asia              SGD
5_select_some_sorted_columns (EURUSD,H1)        17|        710 South Africa   ZA   Africa            ZAR
5_select_some_sorted_columns (EURUSD,H1)        18|        724 Spain          ES   Europe            EUR
5_select_some_sorted_columns (EURUSD,H1)        19|        752 Sweden         SE   Europe            SEK
5_select_some_sorted_columns (EURUSD,H1)        20|        756 Switzerland    CH   Europe            CHF
5_select_some_sorted_columns (EURUSD,H1)        21|        826 United Kingdom GB   Europe            GBP
5_select_some_sorted_columns (EURUSD,H1)        22|        840 United States  US   North America     USD
5_select_some_sorted_columns (EURUSD,H1)        23|        999 European Union EU   Europe            EUR
```

The script works correctly - the "COUNTRY\_ID" column starts at 0 and ends at 999.

_2.6 Selecting grouped results of a specified table column_

Now let's use the _**6\_select\_some\_grouped\_columns.mq5**_ script to get grouped country names by continent. The task is to get the number of countries included in each row of the continent. Countries are selected from the “NAME” column. After running the script, the following lines appear in the log:

```
6_select_some_grouped_columns (EURUSD,H1)       #| CONTINENT         COUNT(NAME)
6_select_some_grouped_columns (EURUSD,H1)       -+------------------------------
6_select_some_grouped_columns (EURUSD,H1)       1| Africa                      1
6_select_some_grouped_columns (EURUSD,H1)       2| Asia                        6
6_select_some_grouped_columns (EURUSD,H1)       3| Australia/Oceania           2
6_select_some_grouped_columns (EURUSD,H1)       4| Europe                      9
6_select_some_grouped_columns (EURUSD,H1)       5| North America               3
6_select_some_grouped_columns (EURUSD,H1)       6| South America               1
6_select_some_grouped_columns (EURUSD,H1)       7| World                       1
```

“Europe” includes the most countries – 9, while “Africa” and “South America”  have only 1 each. Besides, there is also “World”.

_2.7 Selecting unique values of a specified table column_

Now use the _**7\_select\_distinct\_columns.mq5**_ script to collect unique values in the CURRENCY column. There are countries using the same currency. To weed out repetitions, run the script. We can see the following result:

```
7_select_distinct_columns (EURUSD,H1)    1| NZD
7_select_distinct_columns (EURUSD,H1)    2| EUR
7_select_distinct_columns (EURUSD,H1)    3| JPY
7_select_distinct_columns (EURUSD,H1)    4| CAD
7_select_distinct_columns (EURUSD,H1)    5| AUD
7_select_distinct_columns (EURUSD,H1)    6| CNY
7_select_distinct_columns (EURUSD,H1)    7| SGD
7_select_distinct_columns (EURUSD,H1)    8| BRL
7_select_distinct_columns (EURUSD,H1)    9| MXN
7_select_distinct_columns (EURUSD,H1)   10| ZAR
7_select_distinct_columns (EURUSD,H1)   11| HKD
7_select_distinct_columns (EURUSD,H1)   12| INR
7_select_distinct_columns (EURUSD,H1)   13| NOK
7_select_distinct_columns (EURUSD,H1)   14| USD
7_select_distinct_columns (EURUSD,H1)   15| GBP
7_select_distinct_columns (EURUSD,H1)   16| CHF
7_select_distinct_columns (EURUSD,H1)   17| KRW
7_select_distinct_columns (EURUSD,H1)   18| SEK
7_select_distinct_columns (EURUSD,H1)   19| ALL
```

Thus, the calendar has events for a total of 18 currencies and one group of events that applies to all currencies.

It is easy to see that the methods for selecting grouped results and selecting unique values have similarities. Let's demonstrate this with an example.

The **_8\__ _compare\__ _grouped\__ _and\__ _distinct\__ _columns._** _**mq5** script_ displays the following results in the log:

```
8_compare_ grouped_and_distinct_columns (EURUSD,H1)
8_compare_ grouped_and_distinct_columns (EURUSD,H1)     Method CDatabase::SelectFromGroupBy()
8_compare_ grouped_and_distinct_columns (EURUSD,H1)     #| CONTINENT
8_compare_ grouped_and_distinct_columns (EURUSD,H1)     -+------------------
8_compare_ grouped_and_distinct_columns (EURUSD,H1)     1| Africa
8_compare_ grouped_and_distinct_columns (EURUSD,H1)     2| Asia
8_compare_ grouped_and_distinct_columns (EURUSD,H1)     3| Australia/Oceania
8_compare_ grouped_and_distinct_columns (EURUSD,H1)     4| Europe
8_compare_ grouped_and_distinct_columns (EURUSD,H1)     5| North America
8_compare_ grouped_and_distinct_columns (EURUSD,H1)     6| South America
8_compare_ grouped_and_distinct_columns (EURUSD,H1)     7| World
8_compare_ grouped_and_distinct_columns (EURUSD,H1)
8_compare_ grouped_and_distinct_columns (EURUSD,H1)     Method CDatabase::SelectDistinctFrom()
8_compare_ grouped_and_distinct_columns (EURUSD,H1)     #| CONTINENT
8_compare_ grouped_and_distinct_columns (EURUSD,H1)     -+------------------
8_compare_ grouped_and_distinct_columns (EURUSD,H1)     1| Australia/Oceania
8_compare_ grouped_and_distinct_columns (EURUSD,H1)     2| Europe
8_compare_ grouped_and_distinct_columns (EURUSD,H1)     3| Asia
8_compare_ grouped_and_distinct_columns (EURUSD,H1)     4| North America
8_compare_ grouped_and_distinct_columns (EURUSD,H1)     5| South America
8_compare_ grouped_and_distinct_columns (EURUSD,H1)     6| Africa
8_compare_ grouped_and_distinct_columns (EURUSD,H1)     7| World
```

The methods have returned the same content results because we defined the "CONTINENT" column as a grouping column (field) for the first method. Interestingly, the first method has also sorted the selection for us.

_2.8 Selecting ordered unique values of a specified table column_

CURRENCY column values were displayed by the _**7\_select\_distinct\_columns.mq5**_ script in an unsorted way. Let's make a selection with sorting ( **_9\_select\__ _sorted\_distinct\_columns.mq5_** script). Let the column "COUNTRY\_ID" become the sorting criterion. As a result of log manipulations, we get:

```
9_select_sorted_distinct_columns (EURUSD,H1)     #| CURRENCY
9_select_sorted_distinct_columns (EURUSD,H1)    --+---------
9_select_sorted_distinct_columns (EURUSD,H1)     1| ALL
9_select_sorted_distinct_columns (EURUSD,H1)     2| AUD
9_select_sorted_distinct_columns (EURUSD,H1)     3| BRL
9_select_sorted_distinct_columns (EURUSD,H1)     4| CAD
9_select_sorted_distinct_columns (EURUSD,H1)     5| CNY
9_select_sorted_distinct_columns (EURUSD,H1)     6| EUR
9_select_sorted_distinct_columns (EURUSD,H1)     7| HKD
9_select_sorted_distinct_columns (EURUSD,H1)     8| INR
9_select_sorted_distinct_columns (EURUSD,H1)     9| JPY
9_select_sorted_distinct_columns (EURUSD,H1)    10| KRW
9_select_sorted_distinct_columns (EURUSD,H1)    11| MXN
9_select_sorted_distinct_columns (EURUSD,H1)    12| NZD
9_select_sorted_distinct_columns (EURUSD,H1)    13| NOK
9_select_sorted_distinct_columns (EURUSD,H1)    14| SGD
9_select_sorted_distinct_columns (EURUSD,H1)    15| ZAR
9_select_sorted_distinct_columns (EURUSD,H1)    16| SEK
9_select_sorted_distinct_columns (EURUSD,H1)    17| CHF
9_select_sorted_distinct_columns (EURUSD,H1)    18| GBP
9_select_sorted_distinct_columns (EURUSD,H1)    19| USD
```

Now all currencies are sorted. By default, sorting is performed in ascending order.

2.9 _Selecting some of the table columns by condition_

Previously, we have already created an SQL query to select table columns. Now let's make it so that we can get the columns when some condition is met. Suppose that we want to select countries whose ID is equal to or greater than 392 and equal to or less than 840. This task is solved by the _**10\_select\_some\_columns\_where.mq5**_ script.

After running the script, we will see the following in the log:

```
10_select_some_columns_where (EURUSD,H1)         #| COUNTRY_ID NAME           CODE CONTINENT         CURRENCY
10_select_some_columns_where (EURUSD,H1)        --+----------------------------------------------------------
10_select_some_columns_where (EURUSD,H1)         1|        392 Japan          JP   Asia              JPY
10_select_some_columns_where (EURUSD,H1)         2|        410 South Korea    KR   Asia              KRW
10_select_some_columns_where (EURUSD,H1)         3|        484 Mexico         MX   North America     MXN
10_select_some_columns_where (EURUSD,H1)         4|        554 New Zealand    NZ   Australia/Oceania NZD
10_select_some_columns_where (EURUSD,H1)         5|        578 Norway         NO   Europe            NOK
10_select_some_columns_where (EURUSD,H1)         6|        702 Singapore      SG   Asia              SGD
10_select_some_columns_where (EURUSD,H1)         7|        710 South Africa   ZA   Africa            ZAR
10_select_some_columns_where (EURUSD,H1)         8|        724 Spain          ES   Europe            EUR
10_select_some_columns_where (EURUSD,H1)         9|        752 Sweden         SE   Europe            SEK
10_select_some_columns_where (EURUSD,H1)        10|        756 Switzerland    CH   Europe            CHF
10_select_some_columns_where (EURUSD,H1)        11|        826 United Kingdom GB   Europe            GBP
10_select_some_columns_where (EURUSD,H1)        12|        840 United States  US   North America     USD
```

In other words, the sample starts with the country code of 392 and ends with 840.

2.10 _Selecting some of the sorted table columns by condition_

Let's make the previous problem more complicated. Let's add a sorting criterion to the sample - this is the country's belonging to the continent. The current task is solved in the _**11\_select\_some\_sorted\_columns\_where.mq5**_ script. After running it, we will see the following rows in the log:

```
11_select_some_sorted_columns_where (EURUSD,H1)  #| COUNTRY_ID NAME           CODE CONTINENT         CURRENCY
11_select_some_sorted_columns_where (EURUSD,H1) --+----------------------------------------------------------
11_select_some_sorted_columns_where (EURUSD,H1)  1|        710 South Africa   ZA   Africa            ZAR
11_select_some_sorted_columns_where (EURUSD,H1)  2|        392 Japan          JP   Asia              JPY
11_select_some_sorted_columns_where (EURUSD,H1)  3|        410 South Korea    KR   Asia              KRW
11_select_some_sorted_columns_where (EURUSD,H1)  4|        702 Singapore      SG   Asia              SGD
11_select_some_sorted_columns_where (EURUSD,H1)  5|        554 New Zealand    NZ   Australia/Oceania NZD
11_select_some_sorted_columns_where (EURUSD,H1)  6|        578 Norway         NO   Europe            NOK
11_select_some_sorted_columns_where (EURUSD,H1)  7|        724 Spain          ES   Europe            EUR
11_select_some_sorted_columns_where (EURUSD,H1)  8|        752 Sweden         SE   Europe            SEK
11_select_some_sorted_columns_where (EURUSD,H1)  9|        756 Switzerland    CH   Europe            CHF
11_select_some_sorted_columns_where (EURUSD,H1) 10|        826 United Kingdom GB   Europe            GBP
11_select_some_sorted_columns_where (EURUSD,H1) 11|        484 Mexico         MX   North America     MXN
11_select_some_sorted_columns_where (EURUSD,H1) 12|        840 United States  US   North America     USD
```

As a result, "South Africa" comes first in the sample since the continent "Africa" comes first in the list of continents.

_2.11_ Updating _some of the table columns by condition_

Suppose that we are faced with the task of updating the rows in the selected columns. Moreover, we need to do this after fulfilling a preliminary condition.

Let's take the Asian countries and reset the values for them in the CURRENCY and CURRENCY\_SYMBOL columns. The task is performed by the _**12\_update\_some\_columns.mq5**_ script.

As a result of its execution, we get the following table:

```
12_update_some_columns (EURUSD,H1)       #| COUNTRY_ID NAME           CODE CONTINENT         CURRENCY CURRENCY_SYMBOL URL_NAME
12_update_some_columns (EURUSD,H1)      --+-----------------------------------------------------------------------------------------
12_update_some_columns (EURUSD,H1)       1|        554 New Zealand    NZ   Australia/Oceania NZD      $               new-zealand
12_update_some_columns (EURUSD,H1)       2|        999 European Union EU   Europe            EUR      €               european-union
12_update_some_columns (EURUSD,H1)       3|        392 Japan          JP   Asia              None     None            japan
12_update_some_columns (EURUSD,H1)       4|        124 Canada         CA   North America     CAD      $               canada
12_update_some_columns (EURUSD,H1)       5|         36 Australia      AU   Australia/Oceania AUD      $               australia
12_update_some_columns (EURUSD,H1)       6|        156 China          CN   Asia              None     None            china
12_update_some_columns (EURUSD,H1)       7|        380 Italy          IT   Europe            EUR      €               italy
12_update_some_columns (EURUSD,H1)       8|        702 Singapore      SG   Asia              None     None            singapore
12_update_some_columns (EURUSD,H1)       9|        276 Germany        DE   Europe            EUR      €               germany
12_update_some_columns (EURUSD,H1)      10|        250 France         FR   Europe            EUR      €               france
12_update_some_columns (EURUSD,H1)      11|         76 Brazil         BR   South America     BRL      R$              brazil
12_update_some_columns (EURUSD,H1)      12|        484 Mexico         MX   North America     MXN      Mex$            mexico
12_update_some_columns (EURUSD,H1)      13|        710 South Africa   ZA   Africa            ZAR      R               south-africa
12_update_some_columns (EURUSD,H1)      14|        344 Hong Kong      HK   Asia              None     None            hong-kong
12_update_some_columns (EURUSD,H1)      15|        356 India          IN   Asia              None     None            india
12_update_some_columns (EURUSD,H1)      16|        578 Norway         NO   Europe            NOK      Kr              norway
12_update_some_columns (EURUSD,H1)      17|        840 United States  US   North America     USD      $               united-states
12_update_some_columns (EURUSD,H1)      18|        826 United Kingdom GB   Europe            GBP      £               united-kingdom
12_update_some_columns (EURUSD,H1)      19|        756 Switzerland    CH   Europe            CHF      ₣               switzerland
12_update_some_columns (EURUSD,H1)      20|        410 South Korea    KR   Asia              None     None            south-korea
12_update_some_columns (EURUSD,H1)      21|        724 Spain          ES   Europe            EUR      €               spain
12_update_some_columns (EURUSD,H1)      22|        752 Sweden         SE   Europe            SEK      Kr              sweden
12_update_some_columns (EURUSD,H1)      23|          0 Worldwide      WW   World             ALL                      worldwide
```

_2.12_ _Replacing and adding some table rows_

Let's continue our work with the tables. Now let's try to replace some rows in the selected table.

Suppose that we need to replace the current symbol "Mex$" with "Peso mexicano" for “Mexico” in the CURRENCY\_SYMBOL column. We will entrust this task to the _**13\_replace\_some\_rows.mq5**_ script.

In the current version of the COUNTRIES table, Mexico has the following entry:

| COUNTRY\_ID | NAME | CODE | CONTINENT | CURRENCY | CURRENCY\_SYMBOL | URL\_NAME |
| --- | --- | --- | --- | --- | --- | --- |
| 484 | Mexico | MX | North America | MXN | Mex$ | mexico |

In order to replace this row in the table, we need to set a unique value for the selected row. Otherwise, SQLite will not understand what we want to replace.

Let's assume that this value will be the name of the country (the NAME column). Then the replacement function will be represented as follows in the code:

```
//--- the replaced row for "COUNTRY_NAME=Mexico"
string col_names[]=
  {
   "NAME", "CURRENCY_SYMBOL"
  };
string col_vals[2];
col_vals[0]=::StringFormat("'%s'", "Mexico");
col_vals[1]=::StringFormat("'%s'", "Peso mexicano");
if(!db_obj.Replace(col_names, col_vals))
  {
   db_obj.Close();
   return;
  }
```

When executing the script, we get the following error:

```
11_replace_some_rows (EURUSD,H1)        database error, NOT NULL constraint failed: COUNTRIES.COUNTRY_ID
11_replace_some_rows (EURUSD,H1)        CDatabase::Replace: failed with code 5619
```

Obviously, the NOT NULL constraint has been violated. The thing is that initially, when creating the table, it was specified that the COUNTRY\_ID column cannot contain null. So, it is necessary to add a value for this column. In order not to get a half-empty line, let's add values for all columns.

```
//--- the replaced row for "COUNTRY_NAME=Mexico"
string col_names[]=
  {
   "COUNTRY_ID", "NAME", "CODE",
   "CONTINENT", "CURRENCY", "CURRENCY_SYMBOL",
   "URL_NAME"
  };
string col_vals[7];
col_vals[0]=::StringFormat("%I64u", 484);
col_vals[1]=::StringFormat("'%s'", "Mexico");
col_vals[2]=::StringFormat("'%s'", "MX");
col_vals[3]=::StringFormat("'%s'", "North America");
col_vals[4]=::StringFormat("'%s'", "MXN");
col_vals[5]=::StringFormat("'%s'", "Peso mexicano");
col_vals[6]=::StringFormat("'%s'", "mexico");
if(!db_obj.Replace(col_names, col_vals))
  {
   db_obj.Close();
   return;
  }
```

Now the script will work just fine. The following entries appear in the log:

```
13_replace_some_rows (EURUSD,H1)         'Mexico' row before replacement
13_replace_some_rows (EURUSD,H1)        #| COUNTRY_ID NAME   CODE CONTINENT     CURRENCY CURRENCY_SYMBOL URL_NAME
13_replace_some_rows (EURUSD,H1)        -+-----------------------------------------------------------------------
13_replace_some_rows (EURUSD,H1)        1|        484 Mexico MX   North America MXN      Mex$            mexico
13_replace_some_rows (EURUSD,H1)
13_replace_some_rows (EURUSD,H1)         'Mexico' row after replacement
13_replace_some_rows (EURUSD,H1)        #| COUNTRY_ID NAME   CODE CONTINENT     CURRENCY CURRENCY_SYMBOL URL_NAME
13_replace_some_rows (EURUSD,H1)        -+-----------------------------------------------------------------------
13_replace_some_rows (EURUSD,H1)        1|        484 Mexico MX   North America MXN      Peso mexicano   mexico
```

It is worth noting that if there were no line containing data about Mexico, then it would simply be added. In other words, the replacement operation is also the operation of adding table rows.

_2.13_ _Deleting some of the table rows_

Now let's see how we can reduce the number of table rows rather than increasing it. To do this, create the _**14\_delete\_some\_rows.mq5**_ script, which will delete Asia-related rows from the selected table on request.

After running the script, display the final table:

```
14_delete_some_rows (EURUSD,H1)  #| COUNTRY_ID NAME           CODE CONTINENT         CURRENCY CURRENCY_SYMBOL URL_NAME
14_delete_some_rows (EURUSD,H1) --+-----------------------------------------------------------------------------------------
14_delete_some_rows (EURUSD,H1)  1|        554 New Zealand    NZ   Australia/Oceania NZD      $               new-zealand
14_delete_some_rows (EURUSD,H1)  2|        999 European Union EU   Europe            EUR      €               european-union
14_delete_some_rows (EURUSD,H1)  3|        124 Canada         CA   North America     CAD      $               canada
14_delete_some_rows (EURUSD,H1)  4|         36 Australia      AU   Australia/Oceania AUD      $               australia
14_delete_some_rows (EURUSD,H1)  5|        380 Italy          IT   Europe            EUR      €               italy
14_delete_some_rows (EURUSD,H1)  6|        276 Germany        DE   Europe            EUR      €               germany
14_delete_some_rows (EURUSD,H1)  7|        250 France         FR   Europe            EUR      €               france
14_delete_some_rows (EURUSD,H1)  8|         76 Brazil         BR   South America     BRL      R$              brazil
14_delete_some_rows (EURUSD,H1)  9|        710 South Africa   ZA   Africa            ZAR      R               south-africa
14_delete_some_rows (EURUSD,H1) 10|        578 Norway         NO   Europe            NOK      Kr              norway
14_delete_some_rows (EURUSD,H1) 11|        840 United States  US   North America     USD      $               united-states
14_delete_some_rows (EURUSD,H1) 12|        826 United Kingdom GB   Europe            GBP      £               united-kingdom
14_delete_some_rows (EURUSD,H1) 13|        756 Switzerland    CH   Europe            CHF      ₣               switzerland
14_delete_some_rows (EURUSD,H1) 14|        724 Spain          ES   Europe            EUR      €               spain
14_delete_some_rows (EURUSD,H1) 15|        752 Sweden         SE   Europe            SEK      Kr              sweden
14_delete_some_rows (EURUSD,H1) 16|          0 Worldwide      WW   World             ALL                      worldwide
14_delete_some_rows (EURUSD,H1) 17|        484 Mexico         MX   North America     MXN      Peso mexicano   mexico
```

No Asia-related rows were found.

_2.14 Adding columns to the table_

Adding new columns to a table is also a pretty common task.

Suppose that we need to expand our COUNTRIES table and add a column containing the number of macroeconomic events that fall into the calendar.

The task will be performed by the _**15\_add\_new\_column.mq5**_ script.

After executing the script, check the table (Fig. 5). Now it features the new column EVENTS\_NUM.

![New EVENTS_NUM column in the COUNTRIES table](https://c.mql5.com/2/51/5__7.png)

_Fig. 5. New EVENTS\_NUM column in the COUNTRIES table_

_2.15 Renaming columns in the table_

Renaming columns is done by CDatabase::RenameColumn(const string \_curr\_name, const string \_new\_name). Set the current and new column names as parameters. The **_16\_rename\_column.mq5_** script replaces the EVENTS\_NUM column name with EVENTS\_NUMBER.

![The renamed EVENTS_NUMBER column in the COUNTRIES country](https://c.mql5.com/2/51/6__2.png)

_Fig. 6. Renamed EVENTS\_NUMBER column in the COUNTRIES table_

Now the table looks as follows (Fig. 6).

_2.16 Concatenating rows_

Suppose that we need to concatenate the sampling results within a single table. This can be achieved by the CDatabase::Union() method. The task is performed by the **_17\_union\_some\_columns._ _mq5_** script.

Suppose that we have two tables - EUROPEAN\_COUNTRIES and NORTH\_AMERICAN\_COUNTRIES. The first table features European countries, while the second one contains North American countries. Let's first create tables to concatenate their rows. Each of the tables will be the resulting selection from the COUNTRIES table. This looks as follows in the code:

```
//--- create 2 tables
   string table1_name, table2_name, sql_request;
   table1_name="EUROPEAN_COUNTRIES";
   table2_name="NORTH_AMERICAN_COUNTRIES";
   sql_request="SELECT COUNTRY_ID AS id, NAME AS name, CURRENCY "
               "as currency FROM COUNTRIES "
               "WHERE CONTINENT='North America'";
   if(!db_obj.CreateTableAs(table2_name, sql_request, true))
      {
      db_obj.Close();
      return;
      }
   db_obj.FinalizeSqlRequest();
   sql_request="SELECT COUNTRY_ID AS id, NAME AS name, CURRENCY "
               "as currency FROM COUNTRIES "
               "WHERE CONTINENT='Europe'";
   if(!db_obj.CreateTableAs(table1_name, sql_request, true))
      {
      db_obj.Close();
      return;
      }
   db_obj.FinalizeSqlRequest();
```

While running the script, we get the following entries in the log:

```
16_union_some_columns (EURUSD,H1)        #|  id name           currency
16_union_some_columns (EURUSD,H1)       --+----------------------------
16_union_some_columns (EURUSD,H1)        1| 124 Canada         CAD
16_union_some_columns (EURUSD,H1)        2| 250 France         EUR
16_union_some_columns (EURUSD,H1)        3| 276 Germany        EUR
16_union_some_columns (EURUSD,H1)        4| 380 Italy          EUR
16_union_some_columns (EURUSD,H1)        5| 484 Mexico         MXN
16_union_some_columns (EURUSD,H1)        6| 578 Norway         NOK
16_union_some_columns (EURUSD,H1)        7| 724 Spain          EUR
16_union_some_columns (EURUSD,H1)        8| 752 Sweden         SEK
16_union_some_columns (EURUSD,H1)        9| 756 Switzerland    CHF
16_union_some_columns (EURUSD,H1)       10| 826 United Kingdom GBP
16_union_some_columns (EURUSD,H1)       11| 840 United States  USD
16_union_some_columns (EURUSD,H1)       12| 999 European Union EUR
```

The resulting sample includes European and North American countries.

_2.17 Sample difference_

Suppose that we have two samples. We need to find entries in the first sample that are not present in the second one. This can be achieved by the CDatabase::Except() method.

Let's take the COUNTRIES and EUROPEAN\_COUNTRIES tables as an example. Let's see which countries remain if the EXCEPT operator is applied to the first table.

The solution is provided by the _**18\_except\_some\_columns.mq5**_ script.

The following lines will be displayed in the log as a result of the script execution:

```
18_except_some_columns (EURUSD,H1)      #| COUNTRY_ID NAME          CURRENCY
18_except_some_columns (EURUSD,H1)      -+----------------------------------
18_except_some_columns (EURUSD,H1)      1|          0 Worldwide     ALL
18_except_some_columns (EURUSD,H1)      2|         36 Australia     AUD
18_except_some_columns (EURUSD,H1)      3|         76 Brazil        BRL
18_except_some_columns (EURUSD,H1)      4|        124 Canada        CAD
18_except_some_columns (EURUSD,H1)      5|        484 Mexico        MXN
18_except_some_columns (EURUSD,H1)      6|        554 New Zealand   NZD
18_except_some_columns (EURUSD,H1)      7|        710 South Africa  ZAR
18_except_some_columns (EURUSD,H1)      8|        840 United States USD
```

As we can see, the sample does not contain European countries. The Asian ones are absent as well since they were removed earlier.

_2.18  Sample intersection_

Now let's find out what common features the samples have. In other words, the task is to find common rows of samples.

First, let's update the COUNTRIES table and return it to its original form, which included Asian countries.

Create two temporary tables with "id", "name" and "currency" columns. The first will include countries whose value in the COUNTRY\_ID column does not exceed 578, and the second will include countries whose value in the same column is at least 392.

```
//--- create temporary tables
string table1_name, table2_name, sql_request;
table1_name="Table1";
table2_name="Table2";
sql_request="SELECT COUNTRY_ID AS id, NAME AS name, CURRENCY "
            "as currency FROM COUNTRIES "
            "WHERE COUNTRY_ID<=578";
if(!db_obj.CreateTableAs(table1_name, sql_request, true, true))
  {
   db_obj.Close();
   return;
  }
db_obj.FinalizeSqlRequest();
//--- print the temporary table
string temp_col_names[]= {"*"};
if(db_obj.SelectTable(table1_name, true))
   if(db_obj.SelectFrom(temp_col_names))
     {
      ::Print("   \nTable #1: ");
      db_obj.PrintSqlRequest();
      db_obj.FinalizeSqlRequest();
     }
sql_request="SELECT COUNTRY_ID AS id, NAME AS name, CURRENCY "
            "as currency FROM COUNTRIES "
            "WHERE COUNTRY_ID>=392";
if(!db_obj.CreateTableAs(table2_name, sql_request, true, true))
  {
   db_obj.Close();
   return;
  }
db_obj.FinalizeSqlRequest();
//--- print the temporary table
if(db_obj.SelectTable(table2_name, true))
   if(db_obj.SelectFrom(temp_col_names))
     {
      ::Print("   \nTable #2: ");
      db_obj.PrintSqlRequest();
      db_obj.FinalizeSqlRequest();
     }
```

Let's use the features of the CDatabase::Intersect() method in the _**19\_intersect\_some\_columns.mq5**_ script. As a result, we get the following lines in the log:

```
19_intersect_some_columns (EURUSD,H1)   #|  id name        currency
19_intersect_some_columns (EURUSD,H1)   -+-------------------------
19_intersect_some_columns (EURUSD,H1)   1| 392 Japan       JPY
19_intersect_some_columns (EURUSD,H1)   2| 410 South Korea KRW
19_intersect_some_columns (EURUSD,H1)   3| 484 Mexico      MXN
19_intersect_some_columns (EURUSD,H1)   4| 554 New Zealand NZD
19_intersect_some_columns (EURUSD,H1)   5| 578 Norway      NOK
```

The script has worked correctly. We got a list of countries where the country's minimum id value is 392, while the maximum one is 578.

_2.19 Creating views_

A [view](https://www.mql5.com/go?link=https://www.sqlitetutorial.net/sqlite-create-view/ "https://www.sqlitetutorial.net/sqlite-create-view/") is a sort of a virtual table. The convenience is that you can display data selected from any other table.

We will create views using the bool CDatabase::CreateView() and bool CDatabase::CreateViewWhere() methods.The first one creates some kind of unconditional view, while the second one creates a view according to a specified condition.

Let's consider the following example. We have the COUNTRIES table. Suppose that we need to select all countries by the NAME, CONTINENT, and CURRENCY columns for the new virtual table.

Let's solve this problem with the _**20\_create\_view.mq5**_ script. The result is the “All\_countries” view (Fig. 7).

![“All_countries” view ](https://c.mql5.com/2/51/7.png)

_Fig. 7. “All\_countries” view_

Let's complicate the example and now select European countries only. The **_21\_create\_view\_where.mq5_** script will do this. As a result, we have a virtual table containing only European countries (Fig. 8).

![“European” view ](https://c.mql5.com/2/51/8.png)

_Fig. 8. “European” view_

Views are not full-fledged tables since we cannot add, delete or update rows in them. But they can be used to conveniently aggregate the results of complex queries and select individual columns, while changing names without affecting the relationships between the tables themselves.

_2.20 Removing views_

We can remove a previously created view using the CDatabase::DropView() method.

The method is similar to its counterpart, which removes DropTable() tables. In the previous examples, it was the view deletion method that was called before the view was created.

Now it is time to say a few words about the IF EXISTS construction. If we try to delete a non-existent view with this construction, the method returns 'true', otherwise it returns 'false'.

Let's see how the script works **_22\_drop\_view.mq5_**.

```
//--- drop a view
string table_name="COUNTRIES";
if(db_obj.SelectTable(table_name))
   for(int idx=0; idx<2; idx++)
     {
      string view_name=::StringFormat("European%d", idx+1);
      bool if_exists=idx;
      if(db_obj.DropView(view_name, if_exists))
         ::PrintFormat("A view \"%s\" has been successfully dropped!", view_name);
      db_obj.FinalizeSqlRequest();
     }
```

It first tries to remove the non-existent "European\_countries1" view without calling IF EXISTS. As a result, we get error 5601:

```
22_drop_view (EURUSD,H1)        database error, no such view: European1
22_drop_view (EURUSD,H1)        CDatabase::Select: failed with code 5601
22_drop_view (EURUSD,H1)        A view "European2" has been successfully dropped!
```

After that, the script tries to delete the also non-existent "European\_countries2" view using IF EXISTS. Deleting the second view will be successful, even though no deletion actually takes place.

_2.21 Renaming the table_

Let's say that we are faced with the task of renaming the table itself. To do this, let's turn to the CDatabase::RenameTable() method. The renaming is done by the _**23\_rename\_table.mq5**_ script.

![Renamed COUNTRIES1 table](https://c.mql5.com/2/51/9__1.png)

_Fig. 9. Renamed COUNTRIES1 table_

As a result, the current table will be named COUNTRIES1 (Fig. 9).

### 3\. Database of macroeconomic events

In this section, I propose to start creating a relational database of macroeconomic events covered in the Calendar.

So, let's first create a table structure that will make up the future database. The article ["MQL5 Cookbook – Economic Calendar"](https://www.mql5.com/en/articles/9874) already considered of calendar structures. Therefore, in our case, it is quite easy to create database tables and set relationships for them.

_**3.1 Tables and relational connections**_

The database is to contain three source tables:

1. COUNTRIES;
2. EVENTS;
3. EVENT\_VALUES.

The relational connections between tables are shown in Fig. 10.

### ![Relationship structure between tables in the Calendar_DB database](https://c.mql5.com/2/51/10__1.png)      _Fig. 10. The structure of connections between tables in the Calendar\_DB database_

The COUNTRIES table becomes the parent one for the EVENTS table. The latter, in turn, becomes a child one for the former.

The primary key for the COUNTRIES table is the COUNTRY\_ID column (field). In the image, it is preceded by the "+" sign. For the EVENTS table, the key is the EVENT\_ID column, while the COUNTRY\_ID column is an external foreign key. In the structure, it is preceded by the "#" symbol.

The EVENTS table becomes the parent one for the EVENT\_VALUES table, while the second one becomes the child one for the first.

In the EVENT\_VALUES table, the primary key is the VALUE\_ID column (field), while the external key is EVENT\_ID.

The keys are needed precisely in order to implement the relationships between the tables indicated above. The relationships, in turn, contribute to the integrity of the data in the database.

The relationships between the three tables are in the one-to-many form (1..\*). I believe, it will not be difficult to decipher them. The first connection between countries and events can be represented as follows: one country has many macroeconomic events, while one event has only one country. The second connection between events and event values can be illustrated as follows: one event has many values, while any value has only one event.

Let's move on to the code. The _**sCreateAndFillCalendarDB.mq5**_ script features the following stages:

1. creating a calendar database;
2. creating database tables;
3. filling tables.

Let's see, for example, how the EVENTS table is created. The final query for creating this table looks like this:

```
CREATE TABLE IF NOT EXISTS EVENTS (
    EVENT_ID   [UNSIGNED BIG INT] PRIMARY KEY
                                  NOT NULL,
    TYPE       TEXT,
    SECTOR     TEXT,
    FREQUENCY  TEXT,
    TIME_MODE  TEXT,
    COUNTRY_ID [UNSIGNED BIG INT] NOT NULL,
    UNIT       TEXT,
    IMPORTANCE TEXT,
    MULTIPLIER TEXT,
    DIGITS     [UNSIGNED INT],
    SOURCE     TEXT,
    CODE       TEXT,
    NAME       TEXT,
    FOREIGN KEY (
        COUNTRY_ID
    )
    REFERENCES COUNTRIES (COUNTRY_ID) ON UPDATE CASCADE
                                      ON DELETE CASCADE
)
```

The strings where the external key is created are of particular interest. The FOREIGN KEY (COUNTRY\_ID) line means that the table has an external key by the COUNTRY\_ID field. The REFERENCES COUNTRIES(COUNTRY\_ID) construction is used to refer to the COUNTRIES parent table.

The ON UPDATE CASCADE and ON DELETE CASCADE expressions mean that if a related row is deleted or modified from the parent table, the rows in the child table will also be deleted or modified.

As for filling tables, below is a block of code where the COUNTRIES table is being filled.

```
//--- Table 1
MqlCalendarCountry calendar_countries[];
table_name="COUNTRIES";
if(db_obj.SelectTable(table_name))
   if(db_obj.EmptyTable())
     {
      db_obj.FinalizeSqlRequest();
      string col_names[]=
        {
         "COUNTRY_ID",     // 1
         "NAME",           // 2
         "CODE",           // 3
         "CONTINENT",      // 4
         "CURRENCY",       // 5
         "CURRENCY_SYMBOL",// 6
         "URL_NAME"        // 7
        };
      CiCalendarInfo calendar_info;
      if(calendar_info.Init())
        {
         if(calendar_info.GetCountries(calendar_countries))
           {
            if(db_obj.TransactionBegin())
               for(int c_idx=0; c_idx<::ArraySize(calendar_countries); c_idx++)
                 {
                  MqlCalendarCountry curr_country=calendar_countries[c_idx];
                  string col_vals[];
                  ::ArrayResize(col_vals, 7);
                  col_vals[0]=::StringFormat("%I64u", curr_country.id);
                  col_vals[1]=::StringFormat("'%s'", curr_country.name);
                  col_vals[2]=::StringFormat("'%s'", curr_country.code);
                  col_vals[3]="NULL";
                  SCountryByContinent curr_country_continent_data;
                  if(curr_country_continent_data.Init(curr_country.code))
                     col_vals[3]=::StringFormat("'%s'",
                                                curr_country_continent_data.ContinentDescription());
                  col_vals[4]=::StringFormat("'%s'", curr_country.currency);
                  col_vals[5]=::StringFormat("'%s'", curr_country.currency_symbol);
                  col_vals[6]=::StringFormat("'%s'", curr_country.url_name);
                  if(!db_obj.InsertSingleRow(col_names, col_vals))
                    {
                     db_obj.TransactionRollback();
                     db_obj.Close();
                     return;
                    }
                  db_obj.FinalizeSqlRequest();
                 }
            if(!db_obj.TransactionCommit())
               ::PrintFormat("Failed to complete transaction execution, error %d", ::GetLastError());
           }
         //--- print
         if(db_obj.PrintTable()<0)
            ::PrintFormat("Failed to print the table \"%s\", error %d", table_name, ::GetLastError());
        }
     }
```

First, for further work with the table, we need to select it using the CDatabase::SelectTable() method. Here we can draw an analogy with how a trading position is selected using the [::PositionSelect()](https://www.mql5.com/en/docs/trading/positionselect) native function for its further processing.

Then the CDatabase::EmptyTable() method preliminarily clears the table.

Next, go through the countries in a loop and fill the table by columns

- "COUNTRY\_ID",
- "COUNTRY\_NAME",
- "COUNTRY\_CODE",
- "CONTINENT",
- "CURRENCY",
- "CURRENCY\_SYMBOL",
- "URL\_NAME".

The final row is inserted into the table by the CDatabase::InsertSingleRow() method. Filling the table involves a transactional mechanism. Find out more in the section ["Accelerating transactions by wrapping them into DatabaseTransactionBegin()/DatabaseTransactionCommit()"](https://www.mql5.com/en/articles/7463#transactions_speedup).

As a result of filling three tables, the following results were obtained: the COUNTRIES table contains 23 entries, the  EVENTS table contains 1500 entries, while the  EVENT\_VALUES table has 158 696 entries (Fig. 11).

![Filled EVENT_VALUES table](https://c.mql5.com/2/51/11.png)

_Fig. 11. Filled_ _EVENT\_VALUES_ table

Now that we have the data, we can begin to form the queries.

**_3.2 Database queries_**

By and large, all database queries can be divided into two groups:

1) requests that receive data from the database;

2) queries that change data in the database.

First, let's handle the examples of getting information from the calendar database.

_3.2.1 Sample number of events by country_

Let's start by asking the database how many macroeconomic events there are in each country. Create the following query by referring to the "EVENTS" table:

```
SELECT COUNTRY_ID AS id,
       COUNT(EVENT_ID) AS events_num
  FROM EVENTS
 GROUP BY COUNTRY_ID
```

In the MQL5 code, such a request is implemented as follows:

```
//--- 1) group events number by country id
string table_name="EVENTS";
if(db_obj.SelectTable(table_name))
  {
   string col_names_to_select[]=
     {
      "COUNTRY_ID AS id", "COUNT(EVENT_ID) AS events_num"
     };
   string gr_names[]=
     {
      "COUNTRY_ID"
     };
   if(!db_obj.SelectFromGroupBy(col_names_to_select, gr_names))
     {
      db_obj.Close();
      return;
     }
//--- print the SQL request
   if(db_obj.PrintSqlRequest()<0)
      ::PrintFormat("Failed to print the SQL request, error %d", ::GetLastError());
   db_obj.FinalizeSqlRequest();
```

The result is the following selection by the COUNTRY\_ID and COUNT(EVENT\_ID) source columns displayed in the terminal log:

```
sRequest1 (EURUSD,H1)    #|  id events_num
sRequest1 (EURUSD,H1)   --+---------------
sRequest1 (EURUSD,H1)    1|   0          7
sRequest1 (EURUSD,H1)    2|  36         85
sRequest1 (EURUSD,H1)    3|  76         55
sRequest1 (EURUSD,H1)    4| 124         74
sRequest1 (EURUSD,H1)    5| 156         40
sRequest1 (EURUSD,H1)    6| 250         43
sRequest1 (EURUSD,H1)    7| 276         62
sRequest1 (EURUSD,H1)    8| 344         26
sRequest1 (EURUSD,H1)    9| 356         57
sRequest1 (EURUSD,H1)   10| 380         52
sRequest1 (EURUSD,H1)   11| 392        124
sRequest1 (EURUSD,H1)   12| 410         36
sRequest1 (EURUSD,H1)   13| 484         47
sRequest1 (EURUSD,H1)   14| 554         82
sRequest1 (EURUSD,H1)   15| 578         47
sRequest1 (EURUSD,H1)   16| 702         27
sRequest1 (EURUSD,H1)   17| 710         54
sRequest1 (EURUSD,H1)   18| 724         39
sRequest1 (EURUSD,H1)   19| 752         59
sRequest1 (EURUSD,H1)   20| 756         40
sRequest1 (EURUSD,H1)   21| 826        115
sRequest1 (EURUSD,H1)   22| 840        247
sRequest1 (EURUSD,H1)   23| 999         82
```

The selection does not look very readable, because the “id” column is a country ID, not a country name. The country names can be found in the COUNTRIES table.

To get the name of the country and the number of country events, we need to create a compound query (a query within a query).

The first version of such a compound query looks like this:

```
SELECT c.NAME AS country,
       ev.events_num AS events_number
  FROM COUNTRIES c
       JOIN
       (
           SELECT COUNTRY_ID AS id,
                  COUNT(EVENT_ID) AS events_num
             FROM EVENTS
            GROUP BY COUNTRY_ID
       )
       AS ev ON c.COUNTRY_ID = ev.id
```

This option uses the query we created at the beginning. But now it has become part of another query, thereby changing its form to the form of a subquery.

The second version of the request can be implemented in the form of [CTE](https://www.mql5.com/go?link=https://www.sqlite.org/lang_with.html "https://www.sqlite.org/lang_with.html"):

```
WITH ev_cnt AS (
    SELECT COUNTRY_ID AS id,
           COUNT(EVENT_ID) AS events_num
      FROM EVENTS
     GROUP BY COUNTRY_ID
)
SELECT c.NAME AS country,
       ev.events_num AS events_number
  FROM COUNTRIES c
       INNER JOIN
       ev_cnt AS ev ON c.COUNTRY_ID = ev.id
```

In the MQL5 code, a compound query is implemented as follows:

```
//--- 2) group events number by country name using a subquery
::Print("\nGroup events number by country name using a subquery:\n");
string subquery=db_obj.SqlRequest();
string new_sql_request=::StringFormat("SELECT c.NAME AS country,"
                                      "ev.events_num AS events_number FROM COUNTRIES c "
                                      "JOIN(%s) AS ev "
                                      "ON c.COUNTRY_ID=ev.id", subquery);
if(!db_obj.Select(new_sql_request))
  {
   db_obj.Close();
   return;
  }
//--- print the SQL request
if(db_obj.PrintSqlRequest()<0)
   ::PrintFormat("Failed to print the SQL request, error %d", ::GetLastError());
db_obj.FinalizeSqlRequest();
```

The table expression (CTE) is implemented as follows:

```
//--- 3) group events number by country name using CTE
::Print("\nGroup events number by country name using CTE:\n");
new_sql_request=::StringFormat("WITH ev_cnt AS (%s)"
                               "SELECT c.NAME AS country,"
                               "ev.events_num AS events_number FROM COUNTRIES c "
                               "INNER JOIN ev_cnt AS ev ON c.COUNTRY_ID=ev.id", subquery);
if(!db_obj.Select(new_sql_request))
  {
   db_obj.Close();
   return;
  }
//--- print the SQL request
if(db_obj.PrintSqlRequest()<0)
   ::PrintFormat("Failed to print the SQL request, error %d", ::GetLastError());
db_obj.FinalizeSqlRequest();
```

Both options print the following query results in the log:

```
sRequest1 (EURUSD,H1)    #| country        events_number
sRequest1 (EURUSD,H1)   --+-----------------------------
sRequest1 (EURUSD,H1)    1| Worldwide                  7
sRequest1 (EURUSD,H1)    2| Australia                 85
sRequest1 (EURUSD,H1)    3| Brazil                    55
sRequest1 (EURUSD,H1)    4| Canada                    74
sRequest1 (EURUSD,H1)    5| China                     40
sRequest1 (EURUSD,H1)    6| France                    43
sRequest1 (EURUSD,H1)    7| Germany                   62
sRequest1 (EURUSD,H1)    8| Hong Kong                 26
sRequest1 (EURUSD,H1)    9| India                     57
sRequest1 (EURUSD,H1)   10| Italy                     52
sRequest1 (EURUSD,H1)   11| Japan                    124
sRequest1 (EURUSD,H1)   12| South Korea               36
sRequest1 (EURUSD,H1)   13| Mexico                    47
sRequest1 (EURUSD,H1)   14| New Zealand               82
sRequest1 (EURUSD,H1)   15| Norway                    47
sRequest1 (EURUSD,H1)   16| Singapore                 27
sRequest1 (EURUSD,H1)   17| South Africa              54
sRequest1 (EURUSD,H1)   18| Spain                     39
sRequest1 (EURUSD,H1)   19| Sweden                    59
sRequest1 (EURUSD,H1)   20| Switzerland               40
sRequest1 (EURUSD,H1)   21| United Kingdom           115
sRequest1 (EURUSD,H1)   22| United States            247
sRequest1 (EURUSD,H1)   23| European Union            82
```

It is easy to see that the [canlendar](https://www.mql5.com/en/economic-calendar) pays the utmost attention to U.S. events - there are 247 of them in it.

Let's complicate the task a little and add a column to the sample, in which we calculate how many important events occur in a particular country. The importance is defined in the IMPORTANCE column. We will select only those events that have the High value.

First, let's work with the EVENTS table. Here we will need to create two samples. The first sample is a count of the number of events by country. This task has already been completed above. The second sample is a count of the number of important events by country. Finally, we need to combine two samples.

The SQL code of the query looks as follows:

```
SELECT evn.COUNTRY_ID AS id,
       COUNT(EVENT_ID) AS events_num,
       imp.high AS imp_events_num
  FROM EVENTS evn
       JOIN
       (
           SELECT COUNTRY_ID AS id,
                  COUNT(IMPORTANCE) AS high
             FROM EVENTS
            WHERE IMPORTANCE = 'High'
            GROUP BY COUNTRY_ID
       )
       AS imp ON evn.COUNTRY_ID = imp.id
 GROUP BY COUNTRY_ID
```

As for the MQL5 implementation, the code looks like this:

```
//--- 5) important events - ids, events number and important events number
::Print("\nGroup events number and important events number by country id");
subquery=db_obj.SqlRequest();
string new_sql_request4=::StringFormat("SELECT ev.COUNTRY_ID AS id, COUNT(EVENT_ID) AS events_num,"
                                       "imp.high AS imp_events_num "
                                       "FROM EVENTS ev JOIN (%s) AS imp "
                                       "ON ev.COUNTRY_ID=imp.id GROUP BY COUNTRY_ID", subquery);
if(!db_obj.Select(new_sql_request4))
  {
   db_obj.Close();
   return;
  }
//--- print the SQL request
if(db_obj.PrintSqlRequest()<0)
   ::PrintFormat("Failed to print the SQL request, error %d", ::GetLastError());
db_obj.FinalizeSqlRequest();
```

As a result, we get the following log entries:

```
sRequest1 (EURUSD,H1)   Group events number and important events number by country id:
sRequest1 (EURUSD,H1)
sRequest1 (EURUSD,H1)    #|  id events_num imp_events_num
sRequest1 (EURUSD,H1)   --+------------------------------
sRequest1 (EURUSD,H1)    1|   0          7              2
sRequest1 (EURUSD,H1)    2|  36         85              5
sRequest1 (EURUSD,H1)    3|  76         55              2
sRequest1 (EURUSD,H1)    4| 124         74             10
sRequest1 (EURUSD,H1)    5| 156         40              5
sRequest1 (EURUSD,H1)    6| 250         43              1
sRequest1 (EURUSD,H1)    7| 276         62              3
sRequest1 (EURUSD,H1)    8| 344         26              1
sRequest1 (EURUSD,H1)    9| 356         57              2
sRequest1 (EURUSD,H1)   10| 392        124              7
sRequest1 (EURUSD,H1)   11| 410         36              2
sRequest1 (EURUSD,H1)   12| 484         47              2
sRequest1 (EURUSD,H1)   13| 554         82              8
sRequest1 (EURUSD,H1)   14| 578         47              2
sRequest1 (EURUSD,H1)   15| 702         27              1
sRequest1 (EURUSD,H1)   16| 710         54              2
sRequest1 (EURUSD,H1)   17| 752         59              1
sRequest1 (EURUSD,H1)   18| 756         40              4
sRequest1 (EURUSD,H1)   19| 826        115             13
sRequest1 (EURUSD,H1)   20| 840        247             20
sRequest1 (EURUSD,H1)   21| 999         82             11
```

In the final selection, it remains only to replace the “id” column with “country”.

Let's create a compound query again. We will take advantage of the fact that its parts were written earlier. At the end, sort the sample in descending order of the values in the 'imp\_events\_number' column. The compound query looks like this:

```
SELECT c.NAME AS country,
       ev.events_num AS events_number,
       ev.imp_events_num AS imp_events_number
  FROM COUNTRIES c
       JOIN
       (
           SELECT ev.COUNTRY_ID AS id,
                  COUNT(EVENT_ID) AS events_num,
                  imp.high AS imp_events_num
             FROM EVENTS ev
                  JOIN
                  (
                      SELECT COUNTRY_ID AS id,
                             COUNT(IMPORTANCE) AS high
                        FROM EVENTS
                       WHERE IMPORTANCE = 'High'
                       GROUP BY COUNTRY_ID
                  )
                  AS imp ON ev.COUNTRY_ID = imp.id
            GROUP BY COUNTRY_ID
       )
       AS ev ON c.COUNTRY_ID = ev.id
 ORDER BY imp_events_number DESC
```

In the MQL5 code, the request is implemented as follows:

```
//--- 6) important events - countries, events number and important events number
::Print("\nGroup events number and important events number by country:\n");
subquery=db_obj.SqlRequest();
string new_sql_request5=::StringFormat("SELECT c.NAME AS country,"
                                       "ev.events_num AS events_number,"
                                       "ev.imp_events_num AS imp_events_number "
                                       "FROM COUNTRIES c "
                                       "JOIN(%s) AS ev "
                                       "ON c.COUNTRY_ID=ev.id "
                                       "ORDER BY imp_events_number DESC", subquery);
if(!db_obj.Select(new_sql_request5))
  {
   db_obj.Close();
   return;
  }
//--- print the SQL request
if(db_obj.PrintSqlRequest()<0)
   ::PrintFormat("Failed to print the SQL request, error %d", ::GetLastError());
db_obj.FinalizeSqlRequest();
```

Now we get the desired sample in the log:

```
sRequest1 (EURUSD,H1)   Group events number and important events number by country:
sRequest1 (EURUSD,H1)
sRequest1 (EURUSD,H1)    #| country        events_number imp_events_number
sRequest1 (EURUSD,H1)   --+-----------------------------------------------
sRequest1 (EURUSD,H1)    1| United States            247                20
sRequest1 (EURUSD,H1)    2| United Kingdom           115                13
sRequest1 (EURUSD,H1)    3| European Union            82                11
sRequest1 (EURUSD,H1)    4| Canada                    74                10
sRequest1 (EURUSD,H1)    5| New Zealand               82                 8
sRequest1 (EURUSD,H1)    6| Japan                    124                 7
sRequest1 (EURUSD,H1)    7| Australia                 85                 5
sRequest1 (EURUSD,H1)    8| China                     40                 5
sRequest1 (EURUSD,H1)    9| Switzerland               40                 4
sRequest1 (EURUSD,H1)   10| Germany                   62                 3
sRequest1 (EURUSD,H1)   11| Worldwide                  7                 2
sRequest1 (EURUSD,H1)   12| Brazil                    55                 2
sRequest1 (EURUSD,H1)   13| India                     57                 2
sRequest1 (EURUSD,H1)   14| South Korea               36                 2
sRequest1 (EURUSD,H1)   15| Mexico                    47                 2
sRequest1 (EURUSD,H1)   16| Norway                    47                 2
sRequest1 (EURUSD,H1)   17| South Africa              54                 2
sRequest1 (EURUSD,H1)   18| France                    43                 1
sRequest1 (EURUSD,H1)   19| Hong Kong                 26                 1
sRequest1 (EURUSD,H1)   20| Singapore                 27                 1
sRequest1 (EURUSD,H1)   21| Sweden                    59                 1
```

As can be seen from the sample, the United States has the most important events - 20. UK comes second - 13. The third place is occupied by EU - 11. Japan comes 6th - 7.

Let's use a query to find those countries that have no important events at all. To do this, we need to find the difference between the two samples. The first sample will include all countries taken from the COUNTRIES table, while the second one - the column with countries from the previous composite query.

In this case, the SQL code will be as follows:

```
SELECT NAME
  FROM COUNTRIES
EXCEPT
SELECT country
  FROM (
           SELECT c.NAME AS country,
                  ev.events_num AS events_number,
                  ev.imp_events_num AS imp_events_number
             FROM COUNTRIES c
                  JOIN
                  (
                      SELECT ev.COUNTRY_ID AS id,
                             COUNT(EVENT_ID) AS events_num,
                             imp.high AS imp_events_num
                        FROM EVENTS ev
                             JOIN
                             (
                                 SELECT COUNTRY_ID AS id,
                                        COUNT(IMPORTANCE) AS high
                                   FROM EVENTS
                                  WHERE IMPORTANCE = 'High'
                                  GROUP BY COUNTRY_ID
                             )
                             AS imp ON ev.COUNTRY_ID = imp.id
                       GROUP BY COUNTRY_ID
                  )
                  AS ev ON c.COUNTRY_ID = ev.id
       )
```

MQL5 code looks simpler as we take advantage of the fact that the previous query becomes our new subquery.

```
//--- 7) countries having no important events
::Print("\nCountries having no important events:\n");
string last_request=db_obj.SqlRequest();
string new_sql_request6=::StringFormat("SELECT NAME FROM COUNTRIES "
                                       "EXCEPT SELECT country FROM (%s)", last_request);
if(!db_obj.Select(new_sql_request6))
  {
   db_obj.Close();
   return;
  }
//--- print the SQL request
if(db_obj.PrintSqlRequest()<0)
   ::PrintFormat("Failed to print the SQL request, error %d", ::GetLastError());
db_obj.FinalizeSqlRequest();
```

Upon completion of the code execution, we will receive the following entries in the log:

```
sRequest1 (EURUSD,H1)   Countries having no important events:
sRequest1 (EURUSD,H1)
sRequest1 (EURUSD,H1)   #| NAME
sRequest1 (EURUSD,H1)   -+------
sRequest1 (EURUSD,H1)   1| Italy
sRequest1 (EURUSD,H1)   2| Spain
```

So, of all countries, only Italy and Spain do not have important events. Requests about country events in MQL5 were executed in the _**sRequest1.mq5**_ script.

_3.2.2 Sample of GDP values by country_

In this example, we will make a database query to get a selection of GDP values for various countries. As the value of GDP, we will take the parameter “Gross domestic product (GDP) q/q” (for the 3rd quarter).

There will be several samples, so the query will be compound.

First, let's find out the economies of which countries have a quarterly GDP indicator.

The SQL code looks as follows:

```
SELECT COUNTRY_ID,
       EVENT_ID
  FROM EVENTS
 WHERE (NAME LIKE 'GDP q/q' AND
        SECTOR = 'Gross Domestic Product')
```

MQL5 implementation looks as follows ( _**sRequest2.mq5**_ script):

```
//--- 1) countries by id where the indicator '%GDP q/q%' exists
string col_names[]= {"COUNTRY_ID", "EVENT_ID"};
string where_condition="(NAME LIKE 'GDP q/q' AND SECTOR='Gross Domestic Product')";
if(!db_obj.SelectFromWhere(col_names, where_condition))
  {
   db_obj.Close();
   return;
  }
::Print("\nCountries by id where the indicator 'GDP q/q' exists:\n");
//--- print the SQL request
if(db_obj.PrintSqlRequest()<0)
   ::PrintFormat("Failed to print the SQL request, error %d", ::GetLastError());
db_obj.FinalizeSqlRequest();
```

Below is a printout from the log after the query is executed:

```
sRequest2 (EURUSD,H1)   Countries by id where the indicator 'GDP q/q' exists:
sRequest2 (EURUSD,H1)
sRequest2 (EURUSD,H1)    #| COUNTRY_ID  EVENT_ID
sRequest2 (EURUSD,H1)   --+---------------------
sRequest2 (EURUSD,H1)    1|        554 554010024
sRequest2 (EURUSD,H1)    2|        999 999030016
sRequest2 (EURUSD,H1)    3|        392 392010001
sRequest2 (EURUSD,H1)    4|        124 124010022
sRequest2 (EURUSD,H1)    5|         36  36010019
sRequest2 (EURUSD,H1)    6|        156 156010004
sRequest2 (EURUSD,H1)    7|        380 380010020
sRequest2 (EURUSD,H1)    8|        702 702010004
sRequest2 (EURUSD,H1)    9|        276 276010008
sRequest2 (EURUSD,H1)   10|        250 250010005
sRequest2 (EURUSD,H1)   11|         76  76010010
sRequest2 (EURUSD,H1)   12|        484 484020016
sRequest2 (EURUSD,H1)   13|        710 710060009
sRequest2 (EURUSD,H1)   14|        344 344020002
sRequest2 (EURUSD,H1)   15|        578 578020012
sRequest2 (EURUSD,H1)   16|        840 840010007
sRequest2 (EURUSD,H1)   17|        826 826010037
sRequest2 (EURUSD,H1)   18|        756 756040001
sRequest2 (EURUSD,H1)   19|        410 410010011
sRequest2 (EURUSD,H1)   20|        724 724010005
sRequest2 (EURUSD,H1)   21|        752 752010019
```

As we can see, the required indicator exists in 21 countries. The indicator is not used in India as a global one ("Worldwide").

Now we need to get a sample of indicator values for the 3rd quarter and associate it with the first selection by event id.

The SQL query looks as follows:

```
SELECT evs.COUNTRY_ID AS country_id,
       evals.EVENT_ID AS event_id,
       evals.VALUE_ID AS value_id,
       evals.PERIOD AS period,
       evals.TIME AS time,
       evals.ACTUAL AS actual
  FROM EVENT_VALUES evals
       JOIN
       (
           SELECT COUNTRY_ID,
                  EVENT_ID
             FROM EVENTS
            WHERE (NAME LIKE 'GDP q/q' AND
                   SECTOR = 'Gross Domestic Product')
       )
       AS evs ON evals.event_id = evs.EVENT_ID
WHERE (period = '2022.07.01 00:00' )
```

As for the MQL5 code, the compound query is implemented as follows:

```
//--- 2)  'GDP y/y' event and last values
string subquery=db_obj.SqlRequest();
string new_sql_request1=::StringFormat("SELECT evs.COUNTRY_ID AS country_id,"
                                       "evals.EVENT_ID AS event_id,"
                                       "evals.VALUE_ID AS value_id,"
                                       "evals.PERIOD AS period,"
                                       "evals.TIME AS time,"
                                       "evals.ACTUAL AS actual "
                                       "FROM EVENT_VALUES evals "
                                       "JOIN(%s) AS evs ON evals.event_id = evs.event_id "
                                       " WHERE (period = \'2022.07.01 00:00\')", subquery);
if(!db_obj.Select(new_sql_request1))
  {
   db_obj.Close();
   return;
  }
::Print("\n'GDP y/y' event and last values:\n");
//--- print the SQL request
if(db_obj.PrintSqlRequest()<0)
   ::PrintFormat("Failed to print the SQL request, error %d", ::GetLastError());
db_obj.FinalizeSqlRequest();
```

After the execution, the following lines appear in the log:

```
sRequest2 (EURUSD,H1)   'GDP q/q' event and last values:
sRequest2 (EURUSD,H1)
sRequest2 (EURUSD,H1)    #| country_id  event_id value_id period           time             actual
sRequest2 (EURUSD,H1)   --+-----------------------------------------------------------------------
sRequest2 (EURUSD,H1)    1|        554 554010024   168293 2022.07.01 00:00 2022.12.14 23:45    2.0
sRequest2 (EURUSD,H1)    2|        999 999030016   158836 2022.07.01 00:00 2022.10.31 12:00    0.2
sRequest2 (EURUSD,H1)    3|        999 999030016   158837 2022.07.01 00:00 2022.11.15 12:00    0.2
sRequest2 (EURUSD,H1)    4|        999 999030016   158838 2022.07.01 00:00 2022.12.07 12:00    0.3
sRequest2 (EURUSD,H1)    5|        392 392010001   165181 2022.07.01 00:00 2022.11.15 01:50   -0.3
sRequest2 (EURUSD,H1)    6|        392 392010001   165182 2022.07.01 00:00 2022.12.08 01:50   -0.2
sRequest2 (EURUSD,H1)    7|        124 124010022   161963 2022.07.01 00:00 2022.11.29 15:30    0.7
sRequest2 (EURUSD,H1)    8|         36  36010019   173679 2022.07.01 00:00 2022.12.07 02:30    0.6
sRequest2 (EURUSD,H1)    9|        156 156010004   172459 2022.07.01 00:00 2022.10.24 04:00    3.9
sRequest2 (EURUSD,H1)   10|        380 380010020   162296 2022.07.01 00:00 2022.10.31 11:00    0.5
sRequest2 (EURUSD,H1)   11|        380 380010020   162297 2022.07.01 00:00 2022.11.30 11:00    0.5
sRequest2 (EURUSD,H1)   12|        702 702010004   167581 2022.07.01 00:00 2022.10.14 02:00    1.5
sRequest2 (EURUSD,H1)   13|        702 702010004   174527 2022.07.01 00:00 2022.11.23 02:00    1.1
sRequest2 (EURUSD,H1)   14|        276 276010008   172410 2022.07.01 00:00 2022.10.28 10:00    0.3
sRequest2 (EURUSD,H1)   15|        276 276010008   157759 2022.07.01 00:00 2022.11.25 09:00    0.4
sRequest2 (EURUSD,H1)   16|        250 250010005   169062 2022.07.01 00:00 2022.10.28 07:30    0.2
sRequest2 (EURUSD,H1)   17|        250 250010005   169389 2022.07.01 00:00 2022.11.30 09:45    0.2
sRequest2 (EURUSD,H1)   18|         76  76010010   173825 2022.07.01 00:00 2022.12.01 14:00    0.4
sRequest2 (EURUSD,H1)   19|        484 484020016   166108 2022.07.01 00:00 2022.10.31 14:00    1.0
sRequest2 (EURUSD,H1)   20|        484 484020016   166109 2022.07.01 00:00 2022.11.25 14:00    0.9
sRequest2 (EURUSD,H1)   21|        710 710060009   175234 2022.07.01 00:00 2022.12.06 11:30    1.6
sRequest2 (EURUSD,H1)   22|        344 344020002   155337 2022.07.01 00:00 2022.10.31 10:30   -2.6
sRequest2 (EURUSD,H1)   23|        344 344020002   155338 2022.07.01 00:00 2022.11.11 10:30   -2.6
sRequest2 (EURUSD,H1)   24|        578 578020012   172320 2022.07.01 00:00 2022.11.18 09:00    1.5
sRequest2 (EURUSD,H1)   25|        840 840010007   163417 2022.07.01 00:00 2022.10.27 14:30    2.6
sRequest2 (EURUSD,H1)   26|        840 840010007   163418 2022.07.01 00:00 2022.11.30 15:30    2.9
sRequest2 (EURUSD,H1)   27|        840 840010007   163419 2022.07.01 00:00 2022.12.22 15:30    3.2
sRequest2 (EURUSD,H1)   28|        826 826010037   157174 2022.07.01 00:00 2022.11.11 09:00   -0.2
sRequest2 (EURUSD,H1)   29|        826 826010037   157175 2022.07.01 00:00 2022.12.22 09:00   -0.3
sRequest2 (EURUSD,H1)   30|        756 756040001   159276 2022.07.01 00:00 2022.11.29 10:00    0.2
sRequest2 (EURUSD,H1)   31|        410 410010011   161626 2022.07.01 00:00 2022.10.27 01:00    0.3
sRequest2 (EURUSD,H1)   32|        410 410010011   161627 2022.07.01 00:00 2022.12.01 01:00    0.3
sRequest2 (EURUSD,H1)   33|        724 724010005   159814 2022.07.01 00:00 2022.10.28 09:00    0.2
sRequest2 (EURUSD,H1)   34|        724 724010005   159815 2022.07.01 00:00 2022.12.23 10:00    0.1
sRequest2 (EURUSD,H1)   35|        752 752010019   170359 2022.07.01 00:00 2022.10.28 08:00    0.7
sRequest2 (EURUSD,H1)   36|        752 752010019   171381 2022.07.01 00:00 2022.11.29 09:00    0.6
```

It is easy to see that there are several values in the sample for some events with the same event\_id. For example, entries 2-4 refer to the EU parameter. The GDP has been estimated in several readings, so there are several parameter values. As a result, the final sample contains 36 entries, which is clearly more than the number of countries the parameter has been calculated for.

If we need to make a sample by getting only the latest values for a given event, then we need to add the abilities to group and sort group results in the query. Then we get the following compound SQL query:

```
SELECT evs.COUNTRY_ID AS country_id,
       evals.EVENT_ID AS event_id,
       evals.VALUE_ID AS value_id,
       evals.PERIOD AS period,
       evals.TIME AS time,
       evals.ACTUAL AS actual
  FROM EVENT_VALUES evals
       JOIN
       (
           SELECT COUNTRY_ID,
                  EVENT_ID
             FROM EVENTS
            WHERE (NAME LIKE 'GDP q/q' AND
                   SECTOR = 'Gross Domestic Product')
       )
       AS evs ON evals.event_id = evs.EVENT_ID
WHERE (period = '2022.07.01 00:00' )
GROUP BY evals.event_id
HAVING MAX(value_id)
```

The entries will be grouped by the "event\_id" column (field). If there are multiple entries, the one with the maximum value by the "value\_id" column (field) is used. Thus, only one of the three entries for EU will be selected in this case:

| country\_id | event\_id | value\_id | period | time | actual |
| --- | --- | --- | --- | --- | --- |
| 999 | 999030016 | 158838 | 2022.07.01 00:00 | 2022.12.07 12:00 | 0.3 |

As a result, the following entries appear in the log:

```
sRequest2 (EURUSD,H1)   'GDP q/q' event and grouped last values:
sRequest2 (EURUSD,H1)
sRequest2 (EURUSD,H1)    #| country_id  event_id value_id period           time             actual
sRequest2 (EURUSD,H1)   --+-----------------------------------------------------------------------
sRequest2 (EURUSD,H1)    1|         36  36010019   173679 2022.07.01 00:00 2022.12.07 02:30    0.6
sRequest2 (EURUSD,H1)    2|         76  76010010   173825 2022.07.01 00:00 2022.12.01 14:00    0.4
sRequest2 (EURUSD,H1)    3|        124 124010022   161963 2022.07.01 00:00 2022.11.29 15:30    0.7
sRequest2 (EURUSD,H1)    4|        156 156010004   172459 2022.07.01 00:00 2022.10.24 04:00    3.9
sRequest2 (EURUSD,H1)    5|        250 250010005   169389 2022.07.01 00:00 2022.11.30 09:45    0.2
sRequest2 (EURUSD,H1)    6|        276 276010008   172410 2022.07.01 00:00 2022.10.28 10:00    0.3
sRequest2 (EURUSD,H1)    7|        344 344020002   155338 2022.07.01 00:00 2022.11.11 10:30   -2.6
sRequest2 (EURUSD,H1)    8|        380 380010020   162297 2022.07.01 00:00 2022.11.30 11:00    0.5
sRequest2 (EURUSD,H1)    9|        392 392010001   165182 2022.07.01 00:00 2022.12.08 01:50   -0.2
sRequest2 (EURUSD,H1)   10|        410 410010011   161627 2022.07.01 00:00 2022.12.01 01:00    0.3
sRequest2 (EURUSD,H1)   11|        484 484020016   166109 2022.07.01 00:00 2022.11.25 14:00    0.9
sRequest2 (EURUSD,H1)   12|        554 554010024   168293 2022.07.01 00:00 2022.12.14 23:45    2.0
sRequest2 (EURUSD,H1)   13|        578 578020012   172320 2022.07.01 00:00 2022.11.18 09:00    1.5
sRequest2 (EURUSD,H1)   14|        702 702010004   174527 2022.07.01 00:00 2022.11.23 02:00    1.1
sRequest2 (EURUSD,H1)   15|        710 710060009   175234 2022.07.01 00:00 2022.12.06 11:30    1.6
sRequest2 (EURUSD,H1)   16|        724 724010005   159815 2022.07.01 00:00 2022.12.23 10:00    0.1
sRequest2 (EURUSD,H1)   17|        752 752010019   171381 2022.07.01 00:00 2022.11.29 09:00    0.6
sRequest2 (EURUSD,H1)   18|        756 756040001   159276 2022.07.01 00:00 2022.11.29 10:00    0.2
sRequest2 (EURUSD,H1)   19|        826 826010037   157175 2022.07.01 00:00 2022.12.22 09:00   -0.3
sRequest2 (EURUSD,H1)   20|        840 840010007   163419 2022.07.01 00:00 2022.12.22 15:30    3.2
sRequest2 (EURUSD,H1)   21|        999 999030016   158838 2022.07.01 00:00 2022.12.07 12:00    0.3
```

Now there are 21 entries in the sample. Finally, we need to replace the country code with its name. Let's change the previous SLQ query to the following one:

```
SELECT c.NAME AS country,
       ev_evals.event_id AS event_id,
       ev_evals.value_id AS value_id,
       ev_evals.period AS period,
       ev_evals.TIME AS time,
       ev_evals.ACTUAL AS actual
  FROM COUNTRIES c
       JOIN
       (
           SELECT evs.COUNTRY_ID AS country_id,
                  evals.EVENT_ID AS event_id,
                  evals.VALUE_ID AS value_id,
                  evals.PERIOD AS period,
                  evals.TIME AS time,
                  evals.ACTUAL AS actual
             FROM EVENT_VALUES evals
                  JOIN
                  (
                      SELECT COUNTRY_ID,
                             EVENT_ID
                        FROM EVENTS
                       WHERE (NAME LIKE 'GDP q/q' AND
                              SECTOR = 'Gross Domestic Product')
                  )
                  AS evs ON evals.event_id = evs.EVENT_ID
            WHERE (period = '2022.07.01 00:00')
            GROUP BY evals.event_id
           HAVING MAX(value_id)
       )
       AS ev_evals ON c.COUNTRY_ID = ev_evals.country_id
```

Implement the following compound query in MQL5 along the way:

```
//--- 4)  'GDP q/q' event and grouped last values with country names
subquery=db_obj.SqlRequest();
string new_sql_request3=::StringFormat("SELECT c.NAME AS country,"
                                       "ev_evals.event_id AS event_id,"
                                       "ev_evals.value_id AS value_id,"
                                       "ev_evals.period AS period,"
                                       "ev_evals.TIME AS time,"
                                       "ev_evals.ACTUAL AS actual "
                                       "FROM COUNTRIES c JOIN (%s) "
                                       "AS ev_evals ON c.COUNTRY_ID = ev_evals.country_id",
                                       subquery);
if(!db_obj.Select(new_sql_request3))
  {
   db_obj.Close();
   return;
  }
::Print("\n'GDP q/q' event and grouped last values with country names:\n");
//--- print the SQL request
if(db_obj.PrintSqlRequest()<0)
   ::PrintFormat("Failed to print the SQL request, error %d", ::GetLastError());
db_obj.FinalizeSqlRequest();
```

The desired sample will be printed in the journal:

```
sRequest2 (EURUSD,H1)   'GDP q/q' event and grouped last values with country names:
sRequest2 (EURUSD,H1)
sRequest2 (EURUSD,H1)    #| country         event_id value_id period           time             actual
sRequest2 (EURUSD,H1)   --+---------------------------------------------------------------------------
sRequest2 (EURUSD,H1)    1| Australia       36010019   173679 2022.07.01 00:00 2022.12.07 02:30    0.6
sRequest2 (EURUSD,H1)    2| Brazil          76010010   173825 2022.07.01 00:00 2022.12.01 14:00    0.4
sRequest2 (EURUSD,H1)    3| Canada         124010022   161963 2022.07.01 00:00 2022.11.29 15:30    0.7
sRequest2 (EURUSD,H1)    4| China          156010004   172459 2022.07.01 00:00 2022.10.24 04:00    3.9
sRequest2 (EURUSD,H1)    5| France         250010005   169389 2022.07.01 00:00 2022.11.30 09:45    0.2
sRequest2 (EURUSD,H1)    6| Germany        276010008   172410 2022.07.01 00:00 2022.10.28 10:00    0.3
sRequest2 (EURUSD,H1)    7| Hong Kong      344020002   155338 2022.07.01 00:00 2022.11.11 10:30   -2.6
sRequest2 (EURUSD,H1)    8| Italy          380010020   162297 2022.07.01 00:00 2022.11.30 11:00    0.5
sRequest2 (EURUSD,H1)    9| Japan          392010001   165182 2022.07.01 00:00 2022.12.08 01:50   -0.2
sRequest2 (EURUSD,H1)   10| South Korea    410010011   161627 2022.07.01 00:00 2022.12.01 01:00    0.3
sRequest2 (EURUSD,H1)   11| Mexico         484020016   166109 2022.07.01 00:00 2022.11.25 14:00    0.9
sRequest2 (EURUSD,H1)   12| New Zealand    554010024   168293 2022.07.01 00:00 2022.12.14 23:45    2.0
sRequest2 (EURUSD,H1)   13| Norway         578020012   172320 2022.07.01 00:00 2022.11.18 09:00    1.5
sRequest2 (EURUSD,H1)   14| Singapore      702010004   174527 2022.07.01 00:00 2022.11.23 02:00    1.1
sRequest2 (EURUSD,H1)   15| South Africa   710060009   175234 2022.07.01 00:00 2022.12.06 11:30    1.6
sRequest2 (EURUSD,H1)   16| Spain          724010005   159815 2022.07.01 00:00 2022.12.23 10:00    0.1
sRequest2 (EURUSD,H1)   17| Sweden         752010019   171381 2022.07.01 00:00 2022.11.29 09:00    0.6
sRequest2 (EURUSD,H1)   18| Switzerland    756040001   159276 2022.07.01 00:00 2022.11.29 10:00    0.2
sRequest2 (EURUSD,H1)   19| United Kingdom 826010037   157175 2022.07.01 00:00 2022.12.22 09:00   -0.3
sRequest2 (EURUSD,H1)   20| United States  840010007   163419 2022.07.01 00:00 2022.12.22 15:30    3.2
sRequest2 (EURUSD,H1)   21| European Union 999030016   158838 2022.07.01 00:00 2022.12.07 12:00    0.3
```

Even though the problem was solved in several approaches, the possibility of including one query in another made it much easier.

### Conclusion

I hope that the article will arouse interest among those traders and developers who use macroeconomic data to create their strategies. I would also dare to suggest that no macro parameters allow building a good strategy. However, they may be useful as an addition to the original neural network data.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/11977](https://www.mql5.com/ru/articles/11977)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/11977.zip "Download all attachments in the single ZIP archive")

[CalendarDB.zip](https://www.mql5.com/en/articles/download/11977/calendardb.zip "Download CalendarDB.zip")(52.66 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [MQL5 Cookbook — Services](https://www.mql5.com/en/articles/11826)
- [MQL5 Cookbook – Economic Calendar](https://www.mql5.com/en/articles/9874)
- [MQL5 Cookbook: Trading strategy stress testing using custom symbols](https://www.mql5.com/en/articles/7166)
- [MQL5 Cookbook: Getting properties of an open hedge position](https://www.mql5.com/en/articles/4830)
- [MQL5 Cookbook - Pivot trading signals](https://www.mql5.com/en/articles/2853)
- [MQL5 Cookbook - Trading signals of moving channels](https://www.mql5.com/en/articles/1863)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/443614)**
(2)


![Mustafa Nail Sertoglu](https://c.mql5.com/avatar/2021/11/618E1649-9997.PNG)

**[Mustafa Nail Sertoglu](https://www.mql5.com/en/users/nail_mql5)**
\|
16 Mar 2023 at 11:51

Nice "Information System Design" ; thanks for all efforts, i will study on article..


![John May](https://c.mql5.com/avatar/2012/11/509707BE-5C43.jpg)

**[John May](https://www.mql5.com/en/users/cappinjack)**
\|
10 Mar 2024 at 21:15

Denis,

What a great article. I am still stepping through it. Every script has worked for me. and I have been getting the same output as you with one exception;

Section 2.18, you display the output. That had confused me for a moment because in the prior Section 2.13, you had deleted Asia counties.

I appreciate the time you put into it.

![Neural networks made easy (Part 33): Quantile regression in distributed Q-learning](https://c.mql5.com/2/50/Neural_Networks_Made_Easy_q-learning_avatar.png)[Neural networks made easy (Part 33): Quantile regression in distributed Q-learning](https://www.mql5.com/en/articles/11752)

We continue studying distributed Q-learning. Today we will look at this approach from the other side. We will consider the possibility of using quantile regression to solve price prediction tasks.

![Creating an EA that works automatically (Part 06): Account types (I)](https://c.mql5.com/2/50/aprendendo_construindo_006_avatar.png)[Creating an EA that works automatically (Part 06): Account types (I)](https://www.mql5.com/en/articles/11241)

Today we'll see how to create an Expert Advisor that simply and safely works in automatic mode. Our EA in its current state can work in any situation but it is not yet ready for automation. We still have to work on a few points.

![Creating an EA that works automatically (Part 07): Account types (II)](https://c.mql5.com/2/50/aprendendo_construindo_007_avatar.png)[Creating an EA that works automatically (Part 07): Account types (II)](https://www.mql5.com/en/articles/11256)

Today we'll see how to create an Expert Advisor that simply and safely works in automatic mode. The trader should always be aware of what the automatic EA is doing, so that if it "goes off the rails", the trader could remove it from the chart as soon as possible and take control of the situation.

![Data Science and Machine Learning (Part 12): Can Self-Training Neural Networks Help You Outsmart the Stock Market?](https://c.mql5.com/2/52/Self-Training-Neural-Networks-avatar.png)[Data Science and Machine Learning (Part 12): Can Self-Training Neural Networks Help You Outsmart the Stock Market?](https://www.mql5.com/en/articles/12209)

Are you tired of constantly trying to predict the stock market? Do you wish you had a crystal ball to help you make more informed investment decisions? Self-trained neural networks might be the solution you've been looking for. In this article, we explore whether these powerful algorithms can help you "ride the wave" and outsmart the stock market. By analyzing vast amounts of data and identifying patterns, self-trained neural networks can make predictions that are often more accurate than human traders. Discover how you can use this cutting-edge technology to maximize your profits and make smarter investment decisions.

[![](https://www.mql5.com/ff/si/mbxx5fzr169cx07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F498%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dhow.buy.expert%26utm_content%3Dbuy.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=yiuacrhbffqmmulobpsgnypolteeimpt&s=949562ee5e6aca93c0231542844344e241ce4a26ab488f494b70624c190b74d7&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=bggurtxrdvyjhfanyfdfpvgflpyeimpr&ssn=1769191903075289342&ssn_dr=0&ssn_sr=0&fv_date=1769191903&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F11977&back_ref=https%3A%2F%2Fwww.google.com%2F&title=MQL5%20Cookbook%20%E2%80%94%20Macroeconomic%20events%20database%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176919190386423188&fz_uniq=5071646272802204639&sv=2552)

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