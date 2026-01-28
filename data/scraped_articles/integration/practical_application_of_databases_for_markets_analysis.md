---
title: Practical Application Of Databases For Markets Analysis
url: https://www.mql5.com/en/articles/69
categories: Integration
relevance_score: 0
scraped_at: 2026-01-24T14:09:06.851187
---

[![](https://www.mql5.com/ff/sh/x8fwvn495ta7y774z2/01.png)Does your broker offer sponsored hosting for trading?Now it's even easier to get MetaTrader VPS for free – contact your broker for details](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=xscnzeyhifcgygpwvysykhqydcmmbgpp&s=f87b748147e376d34c8f0fdb9737b1766f20cc2174769a0e6b9975b5c2e8ddae&uid=&ref=https://www.mql5.com/en/articles/69&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5083368007352195665)

MetaTrader 5 / Integration


### Introduction

Working with data has become the main task for modern software - both for standalone and network applications. To solve this problem a specialized software were created. These are Database Management Systems (DBMS), that can structure, systematize and organize data for their computer storage and processing. These software are the foundation of information activities in all sectors - from manufacturing to finance and telecommunications.

As for trading, the most of analysts don't use databases in their work. But there are tasks, where such a solution would have to be handy.

This article covers one such task: tick indicator, that saves and loads data from database.

### BuySellVolume Algorithm

BuySellVolume - this simple name I gave to the indicator with even more simple algorithm: take the time (t) and price (p) of two sequential ticks (tick1 and tick2). Let's calculate the difference between them:

> **Δ** **t = t2 - t1**     (seconds)

> **Δ** **p = p2 - p1**    (points)

The volume value is calculated using this formula:

> **v2 = Δp / Δt**

So, our volume is directly proportional to the number of points, by which the price has moved, and is inversely proportional to the time, spent for it. If **Δt** = 0, then instead of it the 0.5 value is taken. Thus, we get a kind of activity value of buyers and sellers in the market.

### 1\. Indicator implementation of without using database

I think it would be logical first to consider an indicator with specified functionality, but without interaction with database. In my opinion, the best solution is to create a base class, that will do the appropriate calculations, and it's derivatives to realize the interaction with the database. To implement this we'll need [AdoSuite](https://www.mql5.com/en/code/93) library. So, click the link and download it.

First, create the BsvEngine.mqh file and connect AdoSuite data classes:

```
#include <Ado\Data.mqh>
```

Then create a base indicator class, which will implement all the necessary functions, except the work with database. It looks as following:

**Listing 1.1**

```
//+------------------------------------------------------------------+
// BuySellVolume indicator class (without storing to database)       |
//+------------------------------------------------------------------+
class CBsvEngine
  {
private:
   MqlTick           TickBuffer[];     // ticks buffer
   double            VolumeBuffer[];   // volume buffer
   int               TicksInBuffer;    // number of ticks in the buffer

   bool              DbAvailable;      // indicates, whether it's possible to work with the database

   long              FindIndexByTime(const datetime &time[],datetime barTime,long left,long right);

protected:

   virtual string DbConnectionString() { return NULL; }
   virtual bool DbCheckAvailable() { return false; }
   virtual CAdoTable *DbLoadData(const datetime startTime,const datetime endTime) { return NULL; }
   virtual void DbSaveData(CAdoTable *table) { return; }

public:
                     CBsvEngine();

   void              Init();
   void              ProcessTick(double &buyBuffer[],double &sellBuffer[]);
   void              LoadData(const datetime startTime,const datetime &time[],double &buyBuffer[],double &sellBuffer[]);
   void              SaveData();
  };
```

I want to note, that in order to increase solution productivity, data are put to the special buffers (TickBuffer and VolumeBuffer), and then after a certain period of time are uploaded into database.

Let's consider the order of class implementation. Let's start with constructor:

**Listing 1.2**

```
//+------------------------------------------------------------------+
// Constructor                                                       |
//+------------------------------------------------------------------+
CBsvEngine::CBsvEngine(void)
  {
// Initially, can be placed up to 500 ticks in a buffer
   ArrayResize(TickBuffer,500);
   ArrayResize(VolumeBuffer,500);
   TicksInBuffer=0;
   DbAvailable=false;
  }
```

Here, I think everything should be clear: variables are initialized and initial sizes of buffers are set.

Next come implementation of the Init() method:

**Listing 1.3**

```
//+-------------------------------------------------------------------+
// Function, called in the OnInit event                               |
//+-------------------------------------------------------------------+
CBsvEngine::Init(void)
  {
   DbAvailable=DbCheckAvailable();
   if(!DbAvailable)
      Alert("Unable to work with database. Working offline.");
  }
```

Here we check, whether it's possible to work with the database.In the base classDbCheckAvailable() always returns false, because working with database will be done only from derived classes.I think, you may have noticed that theDbConnectionString(),DbCheckAvailable(), DbLoadData(), DbSaveData()functions do not have any special meaning yet.These are the functions that we override in descendants to bind to a specific database.

Listing 1.4 shows the implementation of the ProcessTick() function, that is called on the new teak arrival, inserts teak in the buffer and calculates the values for our indicator. To do this 2 indicator buffers are passed to the function: one is used to store buyers activity, the other - to store sellers activity.

**Listing 1.4**

```
//+------------------------------------------------------------------+
// Processing incoming tick and updating indicator data              |
//+------------------------------------------------------------------+
CBsvEngine::ProcessTick(double &buyBuffer[],double &sellBuffer[])
  {
// if it's not enough of allocated buffer for ticks, let's increase it
   int bufSize=ArraySize(TickBuffer);
   if(TicksInBuffer>=bufSize)
     {
      ArrayResize(TickBuffer,bufSize+500);
      ArrayResize(VolumeBuffer,bufSize+500);
     }

// getting the last tick and writing it to the buffer
   SymbolInfoTick(Symbol(),TickBuffer[TicksInBuffer]);

   if(TicksInBuffer>0)
     {
      // calculating the time difference
      int span=(int)(TickBuffer[TicksInBuffer].time-TickBuffer[TicksInBuffer-1].time);
      // calculating the price difference
      int diff=(int)MathRound((TickBuffer[TicksInBuffer].bid-TickBuffer[TicksInBuffer-1].bid)*MathPow(10,_Digits));

      // calculating the volume. If the tick came in the same second as the previous one, we consider the time equal to 0.5 seconds
      VolumeBuffer[TicksInBuffer]=span>0 ?(double)diff/(double)span :(double)diff/0.5;

      // filling the indicator buffers with data
      int index=ArraySize(buyBuffer)-1;
      if(diff>0) buyBuffer[index]+=VolumeBuffer[TicksInBuffer];
      else sellBuffer[index]+=VolumeBuffer[TicksInBuffer];
     }

   TicksInBuffer++;
  }
```

The LoadData() function loads data from the database for the current timeframe for a specified period of time.

**Listing 1.5**

```
//+------------------------------------------------------------------+
// Loading historical data from the database                         |
//+------------------------------------------------------------------+
CBsvEngine::LoadData(const datetime startTime,const datetime &time[],double &buyBuffer[],double &sellBuffer[])
  {
// if the database is inaccessible, then does not load the data
   if(!DbAvailable) return;

// getting data from the database
   CAdoTable *table=DbLoadData(startTime,TimeCurrent());

   if(CheckPointer(table)==POINTER_INVALID) return;

// filling buffers with received data
   for(int i=0; i<table.Records().Total(); i++)
     {
      // get the record with data
      CAdoRecord *row=table.Records().GetRecord(i);

      // getting the index of corresponding bar
      MqlDateTime mdt;
      mdt=row.GetValue(0).ToDatetime();
      long index=FindIndexByTime(time,StructToTime(mdt));

      // filling buffers with data
      if(index!=-1)
        {
         buyBuffer[index]+=row.GetValue(1).ToDouble();
         sellBuffer[index]+=row.GetValue(2).ToDouble();
        }
     }
   delete table;
  }
```

LoadData() calls the DbLoadData() function, which must be overridden in successors and return a table with three columns - the bar time, the buyers buffer value and the sellers buffer value.

Another function is used here - FindIndexByTime().At the time of writing this article I have not found a binary search function for timeseries in the standard library, so I wrote it by myself.

And, finally, the SaveData() function for storing data:

**Listing 1.6**

```
//+---------------------------------------------------------------------+
// Saving data from the TickBuffer and VolumeBuffer buffers to database |
//+---------------------------------------------------------------------+
CBsvEngine::SaveData(void)
  {
   if(DbAvailable)
     {
      // creating a table for passing data to SaveDataToDb
      CAdoTable *table=new CAdoTable();
      table.Columns().AddColumn("Time", ADOTYPE_DATETIME);
      table.Columns().AddColumn("Price", ADOTYPE_DOUBLE);
      table.Columns().AddColumn("Volume", ADOTYPE_DOUBLE);

      // filling table with data from buffers
      for(int i=1; i<TicksInBuffer; i++)
        {
         CAdoRecord *row=table.CreateRecord();
         row.Values().GetValue(0).SetValue(TickBuffer[i].time);
         row.Values().GetValue(1).SetValue(TickBuffer[i].bid);
         row.Values().GetValue(2).SetValue(VolumeBuffer[i]);

         table.Records().Add(row);
        }

      // saving data to database
      DbSaveData(table);

      if(CheckPointer(table)!=POINTER_INVALID)
         delete table;
     }

// writing last tick to the beginning, to have something to compare
   TickBuffer[0] = TickBuffer[TicksInBuffer - 1];
   TicksInBuffer = 1;
  }
```

As we see, in the method a table is formed with necessary information for the indicator and it is passed to the DbSaveData() function, that saves data to the database.After recording, we just clear the buffer.

So, our framework is ready - now let's look at Listing 1.7 how do the BuySellVolume.mq5 indicator look like:

**Listing 1.7**

```
// including file with the indicator class
#include "BsvEngine.mqh"

//+------------------------------------------------------------------+
//| Indicator Properties                                             |
//+------------------------------------------------------------------+
#property indicator_separate_window

#property indicator_buffers 2
#property indicator_plots   2

#property indicator_type1   DRAW_HISTOGRAM
#property indicator_color1  Red
#property indicator_width1  2

#property indicator_type2   DRAW_HISTOGRAM
#property indicator_color2  SteelBlue
#property indicator_width2  2

//+------------------------------------------------------------------+
//| Data Buffers                                                     |
//+------------------------------------------------------------------+
double ExtBuyBuffer[];
double ExtSellBuffer[];

//+------------------------------------------------------------------+
//| Variables                                                        |
//+------------------------------------------------------------------+
// declaring indicator class
CBsvEngine bsv;
//+------------------------------------------------------------------+
//| OnInit
//+------------------------------------------------------------------+
int OnInit()
  {
// setting indicator properties
   IndicatorSetString(INDICATOR_SHORTNAME,"BuySellVolume");
   IndicatorSetInteger(INDICATOR_DIGITS,2);
// buffer for 'buy'
   SetIndexBuffer(0,ExtBuyBuffer,INDICATOR_DATA);
   PlotIndexSetString(0,PLOT_LABEL,"Buy");
// buffer for 'sell'
   SetIndexBuffer(1,ExtSellBuffer,INDICATOR_DATA);
   PlotIndexSetString(1,PLOT_LABEL,"Sell");

// setting the timer to clear buffers with ticks
   EventSetTimer(60);

   return(0);
  }
//+------------------------------------------------------------------+
//| OnDeinit
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   EventKillTimer();
  }
//+------------------------------------------------------------------+
//| OnCalculate
//+------------------------------------------------------------------+
int OnCalculate(const int rates_total,
                const int prev_calculated,
                const datetime &time[],
                const double &open[],
                const double &high[],
                const double &low[],
                const double &close[],
                const long &tick_volume[],
                const long &volume[],
                const int &spread[])
  {
// processing incoming tick
   bsv.ProcessTick(ExtBuyBuffer,ExtSellBuffer);

   return(rates_total);
  }
//+------------------------------------------------------------------+
//| OnTimer
//+------------------------------------------------------------------+
void OnTimer()
  {
// saving data
   bsv.SaveData();
  }
```

Very simple, in my opinion. In the indicator only two functions of the class are called: ProcessTick() and SaveData(). The ProcessTick() function is used for calculations and the SaveData() function is necessary to reset the buffer with tics, although it doesn't save data.

Let's try to compile and "voila" - the indicator begins to show values:

![](https://c.mql5.com/2/1/GBPUSDM1.png)

Figure 1.BuySellVolume indicator without link to the database on GBPUSD M1

Excellent! Ticks are ticking, indicator is calculating. The advantage of such a solution - we need need only indicator itself (ex5) for its work and nothing more. However, when changing the timeframe, or instrument, or when you close the terminal, data are irretrievably lost. To avoid this let's see, how we can add saving and loading data in our indicator.

### 2\. Linking to SQL Server 2008

At the moment I have two DBMSd installed on my computer - SQL Server 2008 and Db2 9.7.I've chosen SQL Server, since I assume that most readers are more familiar with SQL Server, than with Db2.

To begin, let's create a new database BuySellVolume for SQL Server 2008 (via SQL Server Management Studio or any other means) and a new file BsvMsSql.mqh, to which we will include the file with basic CBsvEngine class:

```
#include "BsvEngine.mqh"
```

SQL Server is equipped with OLE DB driver, so we can work with it through the OleDb provider, included in the AdoSuite library. To do this, include the necessary classes:

```
#include <Ado\Providers\OleDb.mqh>
```

And actually create a derived class:

**Listing 2.1**

```
//+------------------------------------------------------------------+
// Class of the BuySellVolume indicator, linked with MsSql database  |
//+------------------------------------------------------------------+
class CBsvSqlServer : public CBsvEngine
  {
protected:

   virtual string    DbConnectionString();
   virtual bool      DbCheckAvailable();
   virtual CAdoTable *DbLoadData(const datetime startTime,const datetime endTime);
   virtual void      DbSaveData(CAdoTable *table);

  };
```

All that we need - is to override four functions, that are responsible for working directly with the database. Let's start from the beginning. The DbConnectionString() method returns a string to connect to the database.

In my case it looks as follows:

**Listing 2.2**

```
//+------------------------------------------------------------------+
// Returns the string for connection to database                     |
//+------------------------------------------------------------------+
string CBsvSqlServer::DbConnectionString(void)
  {
   return "Provider=SQLOLEDB;Server=.\SQLEXPRESS;Database=BuySellVolume;Trusted_Connection=yes;";
  }
```

From the connection string we see that we work through the MS SQL OLE-DB driver with the SQLEXPRESS server, located on the local machine.We're connecting to the BuySellVolume database using Windows authentication (other option - explicitly enter login and password).

The next step is to implement the DbCheckAvailable() function.But first, let's see what really should do this function.

It was said that it checks the possibility to work with the database.To some extent this is true.In fact, it's main purpose - is to check, if there is a table to store data for the current instrument, and if it's not - to create it.If these actions will end with error, it will return false, that would mean that reading and writing indicator data from the table will be ignored, and indicator will work similar to that, which we have already implemented (see Listing 1.7).

I'm suggest to work with data via stored procedures (SP) of SQL Server.Why using them?I just wanted to.This is a matter of taste of course, but I think that using SPs is more elegant solution than to write queries in the code (which also require more time to compile, although it's not applicable to this case, since dynamic queries will be used :)

For DbCheckAvailable() stored procedure looks as follows:

**Listing 2.3**

```
CREATE PROCEDURE [dbo].[CheckAvailable]
        @symbol NVARCHAR(30)
AS
BEGIN
        SET NOCOUNT ON;

        -- If there is no table for instrument, we create it
        IF OBJECT_ID(@symbol, N'U') IS NULL
        EXEC ('
                -- Creating table for the instrument
                CREATE TABLE ' + @symbol + ' (Time DATETIME NOT NULL,
                        Price REAL NOT NULL,
                        Volume REAL NOT NULL);

                -- Creating index for the tick time
                CREATE INDEX Ind' + @symbol + '
                ON  ' + @symbol + '(Time);
        ');
END
```

We see that if the desired table is not in the database, dynamic query (as a string), which creates a table, is formed and executed. When the stored procedure is created - it's time to handle with the DbCheckAvailable() function:

**Listing 2.4**

```
//+------------------------------------------------------------------+
// Checks whether it's possible to connect to database               |
//+------------------------------------------------------------------+
bool CBsvSqlServer::DbCheckAvailable(void)
  {
// working with ms sql via Oledb provider
   COleDbConnection *conn=new COleDbConnection();
   conn.ConnectionString(DbConnectionString());

// using stored procedure to create a table
   COleDbCommand *cmd=new COleDbCommand();
   cmd.CommandText("CheckAvailable");
   cmd.CommandType(CMDTYPE_STOREDPROCEDURE);
   cmd.Connection(conn);

// passing parameters to stored procedure
   CAdoValue *vSymbol=new CAdoValue();
   vSymbol.SetValue(Symbol());
   cmd.Parameters().Add("@symbol",vSymbol);

   conn.Open();

// executing stored procedure
   cmd.ExecuteNonQuery();

   conn.Close();

   delete cmd;
   delete conn;

   if(CheckAdoError())
     {
      ResetAdoError();
      return false;
     }

   return true;
  }
```

As we see, we are able to work with stored procedures of server - we just need to set the CommandType property to CMDTYPE\_STOREDPROCEDURE, then to pass necessary parameters and to execute. As it was conceived, in case of an error the DbCheckAvailable function will return false.

Next, let's write a stored procedure for the DbLoadData function.Since the database stores data for each tick, we need to create data out of them for each bar of required period.I've made the following procedure:

**Listing 2.5**

```
CREATE PROCEDURE [dbo].[LoadData]
        @symbol NVARCHAR(30),   -- instrument
        @startTime DATETIME,    -- beginning of calculation
        @endTime DATETIME,      -- end of calculation
        @period INT             -- chart period (in minutes)
AS
BEGIN
        SET NOCOUNT ON;

        -- converting inputs to strings for passing to a dynamic query
        DECLARE @sTime NVARCHAR(20) = CONVERT(NVARCHAR, @startTime, 112) + ' ' + CONVERT(NVARCHAR, @startTime, 114),
                @eTime NVARCHAR(20) = CONVERT(NVARCHAR, @endTime, 112) + ' ' + CONVERT(NVARCHAR, @endTime, 114),
                @p NVARCHAR(10) = CONVERT(NVARCHAR, @period);

        EXEC('
                SELECT DATEADD(minute, Bar * ' + @p + ', ''' + @sTime + ''') AS BarTime,
                        SUM(CASE WHEN Volume > 0 THEN Volume ELSE 0 END) as Buy,
                        SUM(CASE WHEN Volume < 0 THEN Volume ELSE 0 END) as Sell
                FROM
                (
                        SELECT DATEDIFF(minute, ''' + @sTime + ''', TIME) / ' + @p + ' AS Bar,
                                Volume
                        FROM ' + @symbol + '
                        WHERE Time >= ''' + @sTime + ''' AND Time <= ''' + @eTime + '''
                ) x
                GROUP BY Bar
                ORDER BY 1;
        ');
END
```

The only thing to note - opening time of the first filled bar should be passed as @startTime, otherwise we'll get the offset.

Let's consider the DbLoadData() implementation from the following listing:

**Listing 2.6**

```
//+------------------------------------------------------------------+
// Loading data from the database                                    |
//+------------------------------------------------------------------+
CAdoTable *CBsvSqlServer::DbLoadData(const datetime startTime,const datetime endTime)
  {
// working with ms sql via Oledb provider
   COleDbConnection *conn=new COleDbConnection();
   conn.ConnectionString(DbConnectionString());

// using stored procedure to calculate data
   COleDbCommand *cmd=new COleDbCommand();
   cmd.CommandText("LoadData");
   cmd.CommandType(CMDTYPE_STOREDPROCEDURE);
   cmd.Connection(conn);

// passing parameters to stored procedure
   CAdoValue *vSymbol=new CAdoValue();
   vSymbol.SetValue(Symbol());
   cmd.Parameters().Add("@symbol",vSymbol);

   CAdoValue *vStartTime=new CAdoValue();
   vStartTime.SetValue(startTime);
   cmd.Parameters().Add("@startTime",vStartTime);

   CAdoValue *vEndTime=new CAdoValue();
   vEndTime.SetValue(endTime);
   cmd.Parameters().Add("@endTime",vEndTime);

   CAdoValue *vPeriod=new CAdoValue();
   vPeriod.SetValue(PeriodSeconds()/60);
   cmd.Parameters().Add("@period",vPeriod);

   COleDbDataAdapter *adapter=new COleDbDataAdapter();
   adapter.SelectCommand(cmd);

// creating table and filling it with data, that were returned by stored procedure
   CAdoTable *table=new CAdoTable();
   adapter.Fill(table);

   delete adapter;
   delete conn;

   if(CheckAdoError())
     {
      delete table;
      ResetAdoError();
      return NULL;
     }

   return table;
  }
```

Here we're calling stored procedure, passing tools, calculation start date, calculation end date and current chart period in minutes. Then using the COleDbDataAdapter class we're reading the result into the table, from which the buffers of our indicator will be filled.

And the final step will be to implement the DbSaveData():

**Listing 2.7**

```
CREATE PROCEDURE [dbo].[SaveData]
        @symbol NVARCHAR(30),
        @ticks NVARCHAR(MAX)
AS
BEGIN
        EXEC('
                DECLARE @xmlId INT,
                        @xmlTicks XML = ''' + @ticks + ''';

                EXEC sp_xml_preparedocument
                        @xmlId OUTPUT,
                        @xmlTicks;

                -- read data from xml to table
                INSERT INTO ' + @symbol + '
                SELECT *
                FROM OPENXML( @xmlId, N''*/*'', 0)
                WITH
                (
                        Time DATETIME N''Time'',
                        Price REAL N''Price'',
                        Volume REAL N''Volume''
                );

                EXEC sp_xml_removedocument @xmlId;
        ');
END
```

Please note, that the xml with stored ticks data should be passed as @ticks parameter into the procedure.This decision was taken due to performance reasons - it's easier to call the procedure one time and send there 20 ticks, than to call it 20 times, passing there one tick.Let's see how the xml string should be formed in the following listing:

**Listing 2.8**

```
//+------------------------------------------------------------------+
// Saving data to database                                           |
//+------------------------------------------------------------------+
CBsvSqlServer::DbSaveData(CAdoTable *table)
  {
// if there is nothing to write, then return
   if(table.Records().Total()==0) return;

// forming the xml with data to pass into the stored procedure
   string xml;
   StringAdd(xml,"<Ticks>");

   for(int i=0; i<table.Records().Total(); i++)
     {
      CAdoRecord *row=table.Records().GetRecord(i);

      StringAdd(xml,"<Tick>");

      StringAdd(xml,"<Time>");
      MqlDateTime mdt;
      mdt=row.GetValue(0).ToDatetime();
      StringAdd(xml,StringFormat("%04u%02u%02u %02u:%02u:%02u",mdt.year,mdt.mon,mdt.day,mdt.hour,mdt.min,mdt.sec));
      StringAdd(xml,"</Time>");

      StringAdd(xml,"<Price>");
      StringAdd(xml,DoubleToString(row.GetValue(1).ToDouble()));
      StringAdd(xml,"</Price>");

      StringAdd(xml,"<Volume>");
      StringAdd(xml,DoubleToString(row.GetValue(2).ToDouble()));
      StringAdd(xml,"</Volume>");

      StringAdd(xml,"</Tick>");
     }

   StringAdd(xml,"</Ticks>");

// working with ms sql via Oledb provider
   COleDbConnection *conn=new COleDbConnection();
   conn.ConnectionString(DbConnectionString());

// using stored procedure to write data
   COleDbCommand *cmd=new COleDbCommand();
   cmd.CommandText("SaveData");
   cmd.CommandType(CMDTYPE_STOREDPROCEDURE);
   cmd.Connection(conn);

   CAdoValue *vSymbol=new CAdoValue();
   vSymbol.SetValue(Symbol());
   cmd.Parameters().Add("@symbol",vSymbol);

   CAdoValue *vTicks=new CAdoValue();
   vTicks.SetValue(xml);
   cmd.Parameters().Add("@ticks",vTicks);

   conn.Open();

// executing stored procedure
   cmd.ExecuteNonQuery();

   conn.Close();

   delete cmd;
   delete conn;

   ResetAdoError();
  }
```

Good half of this function takes the formation of this very string with xml. Further, this string is passed to stored procedure and there it is parsed.

For now the implementation of interaction with SQL Server 2008 is finished, and we can implement the **BuySellVolume SqlServer.mq5** indicator.

As you will see, the implementation of this version is similar to the implementation of the last, except for some changes that will discuss further.

**Listing 2.9**

```
// including file with the indicator class
#include "BsvSqlServer.mqh"

//+------------------------------------------------------------------+
//| Indicator Properties                                             |
//+------------------------------------------------------------------+
#property indicator_separate_window

#property indicator_buffers 2
#property indicator_plots   2

#property indicator_type1   DRAW_HISTOGRAM
#property indicator_color1  Red
#property indicator_width1  2

#property indicator_type2   DRAW_HISTOGRAM
#property indicator_color2  SteelBlue
#property indicator_width2  2

//+------------------------------------------------------------------+
//| Input parameters of indicator                                    |
//+------------------------------------------------------------------+
input datetime StartTime=D'2010.04.04'; // start calculations from this date

//+------------------------------------------------------------------+
//| Data Buffers                                                     |
//+------------------------------------------------------------------+
double ExtBuyBuffer[];
double ExtSellBuffer[];

//+------------------------------------------------------------------+
//| Variables                                                        |
//+------------------------------------------------------------------+
// declaring indicator class
CBsvSqlServer bsv;
//+------------------------------------------------------------------+
//| OnInit
//+------------------------------------------------------------------+
int OnInit()
  {
// setting indicator properties
   IndicatorSetString(INDICATOR_SHORTNAME,"BuySellVolume");
   IndicatorSetInteger(INDICATOR_DIGITS,2);
// buffer for 'buy'
   SetIndexBuffer(0,ExtBuyBuffer,INDICATOR_DATA);
   PlotIndexSetString(0,PLOT_LABEL,"Buy");
// buffer for 'sell'
   SetIndexBuffer(1,ExtSellBuffer,INDICATOR_DATA);
   PlotIndexSetString(1,PLOT_LABEL,"Sell");

// calling the Init function of indicator class
   bsv.Init();

// setting the timer to load ticks into database
   EventSetTimer(60);

   return(0);
  }
//+------------------------------------------------------------------+
//| OnDeinit
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   EventKillTimer();
// if there are unsaved data left, then save them
   bsv.SaveData();
  }
//+------------------------------------------------------------------+
//| OnCalculate
//+------------------------------------------------------------------+
int OnCalculate(const int rates_total,
                const int prev_calculated,
                const datetime &time[],
                const double &open[],
                const double &high[],
                const double &low[],
                const double &close[],
                const long &tick_volume[],
                const long &volume[],
                const int &spread[])
  {
   if(prev_calculated==0)
     {
      // calculating the time of the nearest bar
      datetime st[];
      CopyTime(Symbol(),Period(),StartTime,1,st);
      // loading data
      bsv.LoadData(st[0],time,ExtBuyBuffer,ExtSellBuffer);
     }

// processing incoming tick
   bsv.ProcessTick(ExtBuyBuffer,ExtSellBuffer);

   return(rates_total);
  }
//+------------------------------------------------------------------+
//| OnTimer
//+------------------------------------------------------------------+
void OnTimer()
  {
// saving data
   bsv.SaveData();
  }
```

The first difference that strikes the eye - the presence of the StartTime input parameter. This parameter is intended to limit the interval of loading data for the indicator. The fact is that large amount of data may take a long calculation time, although in fact obsolete data do not interest us.

The second difference - the type of the bsv variable is changed to another.

The third difference - loading data on the first calculation of indicator data has been added, as well as the Init() function in the OnInit(), and SaveData() function in the OnDeinit().

Now, let's try to compile the indicator and see the result:

![](https://c.mql5.com/2/1/EURUSDM15.png)

Figure 2. The BuySellVolume indicator linked to SQL Server 2008 database on EURUSD M15

Done! Now our data are saved, and we can freely switch among timeframes.

### 3\. Linking to SQLite 3.6

"To break a fly on the wheel" - I think you understand what I mean.For this task deploying SQL Server is rather ridiculously.Of course, if you have already this DBMS installed and you are actively using it, it may be the preferred option.But what if you want to give indicator to someone who is far from all these technologies and wants a minimum of efforts for solution to work?

Here is the third version of indicator, that, unlike the previous ones, works with a database that has a file-server architecture.In this approach, in most cases you'll only need a couple of DLLs with the database kernel.

Although I had never worked with SQLite earlier, I chose it for its simplicity, speed and lightweight.Initially, we have only API to work from programs, written in C++ and TCL, but I've also found the ODBC driver and ADO.NET provider of third-party developers.Since [AdoSuite](https://www.mql5.com/en/code/93) allows to work with data sources via ODBC, it would seem better to download and install the ODBC driver. But as I understand, its support was discontinued over a year ago, and besides, ADO.NET theoretically should work faster.

So let's look at what needs to be done so that we can work with SQLite via ADO.NET provider from our indicator.

Two actions will bring us to our goal:

- First, you must download and install provider. Here's official website [http://sqlite.phxsoftware.com/](https://www.mql5.com/go?link=http://sqlite.phxsoftware.com/ "/go?link=http://sqlite.phxsoftware.com/"), where download link is available. From all these files we are interested in the **System.** **Data.** **SQLite.dll.** assembly. It includes the SQLite kernel itself and ADO.NET provider. For convenience, I've attached this library to the article. After downloading, open Windows\\assembly folder in Windows Explorer (!). You should see a list of assemblies, as shown in Figure 3:

![](https://c.mql5.com/2/1/gac1.png)

Figure 3. Explorer can display the GAC (global assembly cache) as a list

Now, drag-and-drop (!) **System.** **Data.** **SQLite.dll** to this folder.

As a result, the assembly is placed in the global assembly cache (GAC), and we can work with it:

![](https://c.mql5.com/2/1/gac2.png)

Figure 4. System.Data.SQLite.dll installed in the GAC

For now provider setup is complete.

- The second preparatory action we must to - is to write AdoSuite provider to work with SQLite. It is written quickly and easily (for me it took about 15 minutes). I will not post its code here for the article not to become more huge. You can see the code in files, attached to this article.

Now - when everything is done - you can start writing an indicator. For SQLite database let's create a new empty file in the MQL5\\Files folder. SQLite is not choosy for file extension, so let's call it simply - **BuySellVolume.sqlite**.

In fact, it's not necessary to create the file: it will be automatically created when you first query the database, specified in connection string (see Listing 3.2).Here we create it explicitly only in order to make it clear, where it came from.

Create a new file called BsvSqlite.mqh, include our base class and provider for SQLite:

```
#include "BsvEngine.mqh"
#include <Ado\Providers\SQLite.mqh>
```

The derived class has the same form as the previous one, except the name:

**Listing 3.1**

```
//+------------------------------------------------------------------+
// Class of the BuySellVolume indicator, linked with SQLite database |
//+------------------------------------------------------------------+
class CBsvSqlite : public CBsvEngine
  {
protected:

   virtual string    DbConnectionString();
   virtual bool      DbCheckAvailable();
   virtual CAdoTable *DbLoadData(const datetime startTime,const datetime endTime);
   virtual void      DbSaveData(CAdoTable *table);

  };
```

Now let's proceed with methods implementation.

The DbConnectionString() looks as follows:

**Listing****3.2**

```
//+------------------------------------------------------------------+
// Returns the string for connection to database                     |
//+------------------------------------------------------------------+
string CBsvSqlite::DbConnectionString(void)
  {
   return "Data Source=MQL5\Files\BuySellVolume.sqlite";
  }
```

As you see, the connection string looks much simpler and only indicates the location of our base.

Here the relative path is indicated, but also absolute path is allowed: "DataSource = c:\\Program Files\\Metatrader 5\\MQL 5\\Files\\BuySellVolume.sqlite".

Listing 3.3 shows the DbCheckAvailable() code. Since SQLite does not offer anything like stored procedures to us, now all queries are written directly in the code:

**Listing 3.3**

```
//+------------------------------------------------------------------+
// Checks whether it's possible to connect to database               |
//+------------------------------------------------------------------+
bool CBsvSqlite::DbCheckAvailable(void)
  {
// working with SQLite via written SQLite provider
   CSQLiteConnection *conn=new CSQLiteConnection();
   conn.ConnectionString(DbConnectionString());

// command, that checks the availability of table for the instrument
   CSQLiteCommand *cmdCheck=new CSQLiteCommand();
   cmdCheck.Connection(conn);
   cmdCheck.CommandText(StringFormat("SELECT EXISTS(SELECT name FROM sqlite_master WHERE name = '%s')", Symbol()));

// command, that creates a table for the instrument
   CSQLiteCommand *cmdTable=new CSQLiteCommand();
   cmdTable.Connection(conn);
   cmdTable.CommandText(StringFormat("CREATE TABLE %s(Time DATETIME NOT NULL, " +
                        "Price DOUBLE NOT NULL, "+
                        "Volume DOUBLE NOT NULL)",Symbol()));

// command, that creates an index for the time
   CSQLiteCommand *cmdIndex=new CSQLiteCommand();
   cmdIndex.Connection(conn);
   cmdIndex.CommandText(StringFormat("CREATE INDEX Ind%s ON %s(Time)", Symbol(), Symbol()));

   conn.Open();

   if(CheckAdoError())
     {
      ResetAdoError();
      delete cmdCheck;
      delete cmdTable;
      delete cmdIndex;
      delete conn;
      return false;
     }

   CSQLiteTransaction *tran=conn.BeginTransaction();

   CAdoValue *vExists=cmdCheck.ExecuteScalar();

// if there is no table, we create it
   if(vExists.ToLong()==0)
     {
      cmdTable.ExecuteNonQuery();
      cmdIndex.ExecuteNonQuery();
     }

   if(!CheckAdoError()) tran.Commit();
   else tran.Rollback();

   conn.Close();

   delete vExists;
   delete cmdCheck;
   delete cmdTable;
   delete cmdIndex;
   delete tran;
   delete conn;

   if(CheckAdoError())
     {
      ResetAdoError();
      return false;
     }

   return true;
  }
```

The result of this function is identical to the equivalent for SQL Server. One thing I would like to note - it is types of fields for the table. The funny thing is that fields types have little meaning to SQLite. Moreover, the there are no DOUBLE and DATETIME data types there (at least, they are not included in the standard ones). All values are stored in string form, and then dynamically casted into needed type.

So what's the point in declaring columns as DOUBLE and DATETIME? Do not know the intricacies of the operation, but on query ADO.NET converts them to DOUBLE and DATETIME types automatically. But this is not always true, as there are some moments, one of which will emerge in the following listing.

So, let's consider the listing of the following DbLoadData() function:

**Listing 3.4**

```
//+------------------------------------------------------------------+
// Loading data from the database                                    |
//+------------------------------------------------------------------+
CAdoTable *CBsvSqlite::DbLoadData(const datetime startTime,const datetime endTime)
  {
// working with SQLite via written SQLite provider
   CSQLiteConnection *conn=new CSQLiteConnection();
   conn.ConnectionString(DbConnectionString());

   CSQLiteCommand *cmd=new CSQLiteCommand();
   cmd.Connection(conn);
   cmd.CommandText(StringFormat(
                   "SELECT DATETIME(@startTime, '+' || CAST(Bar*@period AS TEXT) || ' minutes') AS BarTime, "+
                   "  SUM(CASE WHEN Volume > 0 THEN Volume ELSE 0 END) as Buy, "+
                   "  SUM(CASE WHEN Volume < 0 THEN Volume ELSE 0 END) as Sell "+
                   "FROM "+
                   "("+
                   "  SELECT CAST(strftime('%%s', julianday(Time)) - strftime('%%s', julianday(@startTime)) AS INTEGER)/ (60*@period) AS Bar, "+
                   "     Volume "+
                   "  FROM %s "+
                   "  WHERE Time >= @startTime AND Time <= @endTime "+
                   ") x "+
                   "GROUP BY Bar "+
                   "ORDER BY 1",Symbol()));

// substituting parameters
   CAdoValue *vStartTime=new CAdoValue();
   vStartTime.SetValue(startTime);
   cmd.Parameters().Add("@startTime",vStartTime);

   CAdoValue *vEndTime=new CAdoValue();
   vEndTime.SetValue(endTime);
   cmd.Parameters().Add("@endTime",vEndTime);

   CAdoValue *vPeriod=new CAdoValue();
   vPeriod.SetValue(PeriodSeconds()/60);
   cmd.Parameters().Add("@period",vPeriod);

   CSQLiteDataAdapter *adapter=new CSQLiteDataAdapter();
   adapter.SelectCommand(cmd);

// creating table and filling it with data
   CAdoTable *table=new CAdoTable();
   adapter.Fill(table);

   delete adapter;
   delete conn;

   if(CheckAdoError())
     {
      delete table;
      ResetAdoError();
      return NULL;
     }

// as we get the string with date, but not the date itself, it is necessary to convert it
   for(int i=0; i<table.Records().Total(); i++)
     {
      CAdoRecord* row= table.Records().GetRecord(i);
      string strDate = row.GetValue(0).AnyToString();
      StringSetCharacter(strDate,4,'.');
      StringSetCharacter(strDate,7,'.');
      row.GetValue(0).SetValue(StringToTime(strDate));
     }

   return table;
  }
```

This function works the same way as its implementation for MS SQL. But why there is the loop at the end of the function? Yes, in this magic query all my attempts to return the DATETIME were unsuccessful. Absence of DATETIME type in SQLite is evident - instead of the date the string in the YYYY-MM-DD hh:mm:ss format is returned. But it can easily be cast into a form, that is understandable for the StringToTime function, and we used that advantage.

And, finally, the DbSaveData() function:

**Listing 3.5**

```
//+------------------------------------------------------------------+
// Saving data to database                                           |
//+------------------------------------------------------------------+
CBsvSqlite::DbSaveData(CAdoTable *table)
  {
// if there is nothing to write, then return
   if(table.Records().Total()==0) return;

// working with SQLite via SQLite provider
   CSQLiteConnection *conn=new CSQLiteConnection();
   conn.ConnectionString(DbConnectionString());

// using stored procedure to write data
   CSQLiteCommand *cmd=new CSQLiteCommand();
   cmd.CommandText(StringFormat("INSERT INTO %s VALUES(@time, @price, @volume)", Symbol()));
   cmd.Connection(conn);

// adding parameters
   CSQLiteParameter *pTime=new CSQLiteParameter();
   pTime.ParameterName("@time");
   cmd.Parameters().Add(pTime);

   CSQLiteParameter *pPrice=new CSQLiteParameter();
   pPrice.ParameterName("@price");
   cmd.Parameters().Add(pPrice);

   CSQLiteParameter *pVolume=new CSQLiteParameter();
   pVolume.ParameterName("@volume");
   cmd.Parameters().Add(pVolume);

   conn.Open();

   if(CheckAdoError())
     {
      ResetAdoError();
      delete cmd;
      delete conn;
      return;
     }

// ! explicitly starting transaction
   CSQLiteTransaction *tran=conn.BeginTransaction();

   for(int i=0; i<table.Records().Total(); i++)
     {
      CAdoRecord *row=table.Records().GetRecord(i);

      // filling parameters with values
      CAdoValue *vTime=new CAdoValue();
      MqlDateTime mdt;
      mdt=row.GetValue(0).ToDatetime();
      vTime.SetValue(mdt);
      pTime.Value(vTime);

      CAdoValue *vPrice=new CAdoValue();
      vPrice.SetValue(row.GetValue(1).ToDouble());
      pPrice.Value(vPrice);

      CAdoValue *vVolume=new CAdoValue();
      vVolume.SetValue(row.GetValue(2).ToDouble());
      pVolume.Value(vVolume);

      // adding record
      cmd.ExecuteNonQuery();
     }

// completing transaction
   if(!CheckAdoError())
      tran.Commit();
   else tran.Rollback();

   conn.Close();

   delete tran;
   delete cmd;
   delete conn;

   ResetAdoError();
  }
```

I want cover the details of this function implementation.

First, everything is done in the transaction, although it is logical. But this was done not due to data safety reasons - it was done due to performance reasons: if an entry is added without explicit transaction, server creates a transaction implicitly, inserts a record into table and removes a transaction. And this is done for each tick! Moreover, the entire database is locked when entry is being recorded! It is worth noting, that commands do not necessarily need transaction. Again, I have not fully understood, why it is happening. I suppose that this is due to the lack of multiple transactions.

Secondly, we create a command once, and then in a loop we assign parameters and execute it. This, again, is the issue of productivity, as the command is compiled (optimized) once, and then work is done with a compiled version.

Well, let's get to the point. Let's look at the **BuySellVolume SQLite.mq5** indicator itself:

**Listing 3.6**

```
// including file with the indicator class
#include "BsvSqlite.mqh"

//+------------------------------------------------------------------+
//| Indicator Properties                                             |
//+------------------------------------------------------------------+
#property indicator_separate_window

#property indicator_buffers 2
#property indicator_plots   2

#property indicator_type1   DRAW_HISTOGRAM
#property indicator_color1  Red
#property indicator_width1  2

#property indicator_type2   DRAW_HISTOGRAM
#property indicator_color2  SteelBlue
#property indicator_width2  2

//+------------------------------------------------------------------+
//| Input parameters of indicator                                    |
//+------------------------------------------------------------------+
input datetime StartTime=D'2010.04.04';   // start calculations from this date

//+------------------------------------------------------------------+
//| Data Buffers
//+------------------------------------------------------------------+
double ExtBuyBuffer[];
double ExtSellBuffer[];

//+------------------------------------------------------------------+
//| Variables
//+------------------------------------------------------------------+
// declaring indicator class
CBsvSqlite bsv;
//+------------------------------------------------------------------+
//| OnInit
//+------------------------------------------------------------------+
int OnInit()
  {
// setting indicator properties
   IndicatorSetString(INDICATOR_SHORTNAME,"BuySellVolume");
   IndicatorSetInteger(INDICATOR_DIGITS,2);
// buffer for 'buy'
   SetIndexBuffer(0,ExtBuyBuffer,INDICATOR_DATA);
   PlotIndexSetString(0,PLOT_LABEL,"Buy");
// buffer for 'sell'
   SetIndexBuffer(1,ExtSellBuffer,INDICATOR_DATA);
   PlotIndexSetString(1,PLOT_LABEL,"Sell");

// calling the Init function of indicator class
   bsv.Init();

// setting the timer to load ticks into database
   EventSetTimer(60);

   return(0);
  }
//+------------------------------------------------------------------+
//| OnDeinit
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   EventKillTimer();
// if there are unsaved data left, then save them
   bsv.SaveData();
  }
//+------------------------------------------------------------------+
//| OnCalculate
//+------------------------------------------------------------------+
int OnCalculate(const int rates_total,
                const int prev_calculated,
                const datetime &time[],
                const double &open[],
                const double &high[],
                const double &low[],
                const double &close[],
                const long &tick_volume[],
                const long &volume[],
                const int &spread[])
  {
   if(prev_calculated==0)
     {
      // calculating the time of the nearest bar
      datetime st[];
      CopyTime(Symbol(),Period(),StartTime,1,st);
      // loading data
      bsv.LoadData(st[0],time,ExtBuyBuffer,ExtSellBuffer);
     }

// processing incoming tick
   bsv.ProcessTick(ExtBuyBuffer,ExtSellBuffer);

   return(rates_total);
  }
//+------------------------------------------------------------------+
//| OnTimer
//+------------------------------------------------------------------+
void OnTimer()
  {
// saving data
   bsv.SaveData();
  }
```

Only function class has changed, the rest of the code remained unchanged.

For now implementation of the third version of the indicator is over - you can view the result.

![](https://c.mql5.com/2/1/EURUSDM5.png)

Figure 5. The BuySellVolume indicator linked to SQLite 3.6 database on EURUSD M5

By the way, unlike Sql Server Management Studio in SQLite there are no standard utilities to work with databases. Therefore, in order not to work with "black box", you can download the appropriate utility from third-party developers. Personally, I like SQLiteMan - it is easy to use and at the same time have all the necessary functionality. You can download it from here: [http://sourceforge.net/projects/sqliteman/](https://www.mql5.com/go?link=https://sourceforge.net/projects/sqliteman/ "/go?link=https://sourceforge.net/projects/sqliteman/").

### Conclusion

If you read these lines, then all is over;). I must confess, I didn't expect this article to be so huge. Therefore, the questions, that I will certainly answer, are inevitable.

As we see, every solution has its pros and cons. The first variant differs by its independence, the second - by its performance, and the third - by its portability. Which one to choose - is up to you.

Is implemented indicator useful? Same up to you to decide. As for me - it is very interesting specimen.

In doing so, let me say goodbye. See ya!

**Description of archives contents:**

| # | Filename | Description |
| --- | --- | --- |
| 1 | Sources\_en.zip | Contains the source codes of all indicators and [AdoSuite](https://www.mql5.com/en/code/93) library. It should be unpacked into appropriate folder of your terminal. Purpose of indicators: without use of database **(BuySellVolume.mq5)**, working with SQL Server 2008 database **(BuySellVolume SqlServer.mq5)** and working with SQLite database **(BuySellVolume SQLite.mq5).** |
| 2 | BuySellVolume-DB-SqlServer.zip | SQL Server 2008 database archive\* |
| 3 | BuySellVolume-DB-SQLite.zip | SQLite database archive\* |
| 4 | System.Data.SQLite.zip | System.Data.SQLite.dll archive, necessary to work with SQLite database |
| 5 | Databases\_MQL5\_doc\_en.zip | Source codes, indicators and the AdoSuite library documentation archive |

\\* Both databases contain tick indicator data from 5 to 9 April inclusive for the following instruments: AUDNZD, EURUSD, GBPUSD, USDCAD, USDCHF, USDJPY.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/69](https://www.mql5.com/ru/articles/69)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/69.zip "Download all attachments in the single ZIP archive")

[buysellvolume-db-sqlserver.zip](https://www.mql5.com/en/articles/download/69/buysellvolume-db-sqlserver.zip "Download buysellvolume-db-sqlserver.zip")(13372.03 KB)

[buysellvolume-db-sqlite.zip](https://www.mql5.com/en/articles/download/69/buysellvolume-db-sqlite.zip "Download buysellvolume-db-sqlite.zip")(12875.03 KB)

[system\_data\_sqlite.zip](https://www.mql5.com/en/articles/download/69/system_data_sqlite.zip "Download system_data_sqlite.zip")(423.17 KB)

[sources\_en.zip](https://www.mql5.com/en/articles/download/69/sources_en.zip "Download sources_en.zip")(55.16 KB)

[databases\_mql5\_doc\_en.zip](https://www.mql5.com/en/articles/download/69/databases_mql5_doc_en.zip "Download databases_mql5_doc_en.zip")(435.85 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [How to Export Quotes from МetaTrader 5 to .NET Applications Using WCF Services](https://www.mql5.com/en/articles/27)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/999)**
(8)


![parliament718](https://c.mql5.com/avatar/avatar_na2.png)

**[parliament718](https://www.mql5.com/en/users/parliament718)**
\|
28 Dec 2011 at 07:02

Great article! Really just the solution I was looking for. Unfortunately I'm having the same issue as Denkir commented on in the Russian version of this article.

It does not compile giving error:

'Values' - cannot call protected member [function](https://www.mql5.com/en/docs/constants/namedconstants/compilemacros "MQL5 documentation: Predefined Macrosubstitutions").

Associated with these lines in CBsvEngine::SaveData(void)

```
row.Values().GetValue(0).SetValue(TickBuffer[i].time);
row.Values().GetValue(1).SetValue(TickBuffer[i].bid);
row.Values().GetValue(2).SetValue(VolumeBuffer[i]);
```

And also the same error associated with numerous calls to 'Values()' in CDbDataAdapter::Fill(CAdoTable \*table)   in the file DbDataAdapter.mqh

I would really love to get this working. Much appreciated!


![zephyrrr](https://c.mql5.com/avatar/avatar_na2.png)

**[zephyrrr](https://www.mql5.com/en/users/zephyrrr)**
\|
14 Feb 2012 at 08:41

can it be run in [strategy tester](https://www.mql5.com/en/articles/239 "Article \"The Fundamentals of Testing in MetaTrader 5\"")? when i use adoSuite in strategy tester, it always terminate the program.


![Automated-Trading](https://c.mql5.com/avatar/2021/6/60C759A6-5565.png)

**[Automated-Trading](https://www.mql5.com/en/users/automated-trading)**
\|
14 Feb 2012 at 09:45

**zephyrrr:**

can it be run in strategy tester? when i use adoSuite in strategy tester, it always terminate the program.

Unfortunately the build 586 has an error in calling of 32-bit DLLs [functions](https://www.mql5.com/en/docs/constants/namedconstants/compilemacros "MQL5 documentation: Predefined Macrosubstitutions"). It will be fixed in next build.


![Francisco Jose Castro Cabal](https://c.mql5.com/avatar/2020/5/5ECA3A32-1EC3.jpg)

**[Francisco Jose Castro Cabal](https://www.mql5.com/en/users/fjccpm)**
\|
24 Feb 2012 at 15:37

**Automated-Trading:**

Unfortunately the build 586 has an error in calling of 32-bit DLLs functions. It will be fixed in next build.

Unfortunately this haven't been solved after the last update. Any solutions or recommendations? how long this could take? I was told to wait until the next build and it came without solving the problem.


![Cen Chen](https://c.mql5.com/avatar/2021/3/6059C66E-2924.gif)

**[Cen Chen](https://www.mql5.com/en/users/anlycn)**
\|
18 Jul 2021 at 08:52

![Creating Active Control Panels in MQL5 for Trading](https://c.mql5.com/2/0/panel__2.png)[Creating Active Control Panels in MQL5 for Trading](https://www.mql5.com/en/articles/62)

The article covers the problem of development of active control panels in MQL5. Interface elements are managed by the event handling mechanism. Besides, the option of a flexible setup of control elements properties is available. The active control panel allows working with positions, as well setting, modifying and deleting market and pending orders.

![Migrating from MQL4 to MQL5](https://c.mql5.com/2/0/logo__4.png)[Migrating from MQL4 to MQL5](https://www.mql5.com/en/articles/81)

This article is a quick guide to MQL4 language functions, it will help you to migrate your programs from MQL4 to MQL5. For each MQL4 function (except trading functions) the description and MQL5 implementation are presented, it allows you to reduce the conversion time significantly. For convenience, the MQL4 functions are divided into groups, similar to MQL4 Reference.

![MQL for "Dummies": How to Design and Construct Object Classes](https://c.mql5.com/2/0/cube__1.png)[MQL for "Dummies": How to Design and Construct Object Classes](https://www.mql5.com/en/articles/53)

By creating a sample program of visual design, we demonstrate how to design and construct classes in MQL5. The article is written for beginner programmers, who are working on MT5 applications. We propose a simple and easy grasping technology for creating classes, without the need to deeply immerse into the theory of object-oriented programming.

![A Virtual Order Manager to track orders within the position-centric MetaTrader 5 environment](https://c.mql5.com/2/0/virtual__1.png)[A Virtual Order Manager to track orders within the position-centric MetaTrader 5 environment](https://www.mql5.com/en/articles/88)

This class library can be added to an MetaTrader 5 Expert Advisor to enable it to be written with an order-centric approach broadly similar to MetaTrader 4, in comparison to the position-based approach of MetaTrader 5. It does this by keeping track of virtual orders at the MetaTrader 5 client terminal, while maintaining a protective broker stop for each position for disaster protection.

[![](https://www.mql5.com/ff/sh/wm94j0jmkwd29943z2/ddfa713cb3cdd580c3e81e0e13b5b1b8.jpg)\\
Revised MetaTrader 5 Web Terminal\\
\\
Trade with no restrictions from any mobile device, OS and web browser\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=fkjlpstbxdmrrwpblfatcsdjyrxbizyj&s=f462f051eb7aaec36d6b31792d312d60d3f5a50c83b12d0d66e85d5d61bd941b&uid=&ref=https://www.mql5.com/en/articles/69&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5083368007352195665)

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