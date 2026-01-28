---
title: Developing a multi-currency Expert Advisor (Part 17): Further preparation for real trading
url: https://www.mql5.com/en/articles/15360
categories: Trading, Trading Systems, Expert Advisors
relevance_score: 9
scraped_at: 2026-01-22T17:27:11.761964
---

[![](https://www.mql5.com/ff/sh/0hvxp984jjj79943z2/6373d9e5710a718ffa6a7d50a5db9dd1.jpg)\\
Web terminal on your iPhone or Android\\
\\
Full-featured MetaTrader 5 platform for any devices and web browsers\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=uyigsjnbfcdvysiynusmriwvhincciwd&s=c95531ae2fd8a81b0fac3def2e4cf820a67584bbf4b02f76ec75f808942dbbd2&uid=&ref=https://www.mql5.com/en/articles/15360&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5049143556158891453)

MetaTrader 5 / Tester


### Introduction

In one of the previous articles, we already turned our attention to the EA improvements necessary for working on real accounts. Until now, our efforts have been focused mainly on getting acceptable EA results in the strategy tester. Real trading requires much more preparations.

In addition to restoring the EA operation after restarting the terminal, the ability to use slightly different names of trading instruments and auto completion of trading when the specified indicators are reached, we also face the following issue: in order to form the initialization string, we use information obtained directly from the database, which stores all the results of optimizations of trading strategy instances and their groups.

To run the EA, we must have a file with the database in the shared terminal folder. The size of the database is already several gigabytes, and it will only grow in the future. So making the database an integral part of the EA is not rational - only a very small part of the information stored there is needed for launch. Therefore, it is necessary to implement a mechanism for extracting and using this information in the EA.

### Mapping out the path

Let us recall that we have considered and implemented automation of two stages of testing. At the first stage, the parameters of a single instance of the trading strategy are optimized ( [part 11](https://www.mql5.com/en/articles/14741)). The model trading strategy under study uses only one trading instrument (symbol) and one timeframe. Therefore, we consistently ran it through the optimizer, changing symbols and timeframes. For each combination of symbol and timeframe, optimization was carried out in turn according to different optimization criteria. All results of optimization passes were set in the 'passes' table of our database.

At the second stage, we optimized the selection of a group of parameter sets obtained in the first stage that yielded the best results when used together ( [part 6](https://www.mql5.com/en/articles/14478) and [part 13](https://www.mql5.com/en/articles/14892)). As in the first stage, we included sets of parameters using the same symbol-timeframe pair into one group. Information about the results of all groups reviewed during optimization was also saved in our database.

At the third stage, we no longer used the standard strategy tester optimizer, so we are not talking about its automation yet. The third stage consisted of selecting one of the best groups found in the second stage for each available combination of symbol and timeframe. We used optimization on three symbols (EURGBP, EURUSD, GBPUSD) and three timeframes (H1, M30, M15). Thus, the result of the third stage will be nine selected groups. But to simplify and accelerate calculations in the tester, we limited ourselves in the last articles to only the three best groups (with three different symbols and the H1 timeframe).

The result of the third stage was a set of row identifiers from the 'passes' table, which we passed through the input parameter to our final _SimpleVolumesExpert.mq5_ EA:

```
input string     passes_ = "734469,"
                           "736121,"
                           "776928";    // - Comma-separated pass IDs
```

We could change this parameter before launching the EA test. Thus, it was possible to run the final EA with any desired subset of groups from the set of groups available in the database in the 'passes' table, or, to be more precise, with a subset, which does not exceed 247 characters in length. This is a limitation imposed by the MQL5 language on the values of input string parameters. According to the [documentation](https://www.mql5.com/en/docs/basis/variables/inputvariables), the maximum length of a string parameter value can be from 191 to 253 characters, depending on the length of the parameter name.

Therefore, if we want to include more than, roughly speaking, 40 groups into the work, then it will not be possible to do it this way. For example, we might have to make the _passes\__ variable a simple stirng variable rather than an input string parameter by removing the word _input_ from the code. In this case, we can specify the required set of groups only in the source code. However, we do not need to use such large sets yet. Moreover, according to the experiments conducted in [part 5](https://www.mql5.com/en/articles/14336), it is more profitable for us not to make one group from a large number of single copies of trading strategies or groups of trading strategies. It is more profitable to split the initial number of single copies of trading strategies into several subgroups, from which a smaller number of new groups can be assembled. These new groups can either be combined into one final group, or the grouping process can be repeated by division into new subgroups. Thus, at each level of unification, we will have to take a relatively small number of strategies or groups as a single group.

When the EA has access to the database with the results of all optimization passes, it is sufficient to pass a list of IDs of the required optimization passes via the input. The EA receives the initialization strings of those groups of trading strategies, that participated in the listed passes, from the database on its own. Based on the initialization strings received from the database, it will construct an initialization string for an EA object that includes all trading strategies from the listed groups. This EA will trade using all the trading strategy instances included in it.

If there is no access to the database, the EA still needs to somehow generate an initialization string for the EA object, containing the required composition of single instances of trading strategies or groups of trading strategies. For example, we can save it to a file and pass the name of the file to the EA it will load the initialization string from. Or we can insert the contents of the initialization string into the source code of the EA via an additional mqh library file. We can even combine the two methods by saving the initialization string to a file, then importing it using the file import facilities in MetaEditor (Edit → Insert → File).

However, if we want to provide the ability to work with different selected groups in one EA, choosing the desired one in the inputs, then this approach will quickly show its weak scalability. We will need to do a lot of manual, repetitive work. Therefore, let's try to formulate the problem a little differently: we want to form a library of good initialization strings, from which we can choose one for the current EA launch. The library should be an integral part of the EA, so that we do not have to use another separate file along with it.

Taking into account the above, the upcoming work can be divided into the following stages:

- **Selection and saving**. At this stage, we should have a tool that allows us to select groups and save their initialization strings for later use. It would probably be a good idea to provide the ability to save some additional information about the selected groups (name, brief description, approximate composition, creation date, etc.)

- **Forming the library**. From the groups selected in the previous stage, a final selection is made of those that will be used in the library for a specific release of the EA, and an include file with all the necessary information is formed.

- **Creating the final EA**. By modifying the EA from the previous part, we will turn it into a new final EA using the created group library. This EA will no longer need access to our optimization database, as all the necessary information about the trading strategy groups used will be included in it.

Let's start implementing our plans.

### Revisiting previous accomplishments

The steps mentioned are a prototype of the implementation of Stage 8 described in [Part 9](https://www.mql5.com/en/articles/14680). Let us recall that in that article we listed a set of stages, the completion of which can allow us to get a ready-made EA with good trading performance. Stage 8 implied that we collect all the best groups of groups found for different trading strategies, symbols, timeframes and other parameters into one final EA. However, we have not yet considered in detail the question "How exactly should the best groups be selected?"

On the one hand, the answer to the question may turn out to be pretty simple. For example, we might simply select the best results from all the groups according to some parameter (total profit, Sharpe ratio, normalized average annual profit). But on the other hand, the answer may turn out to be much more complicated. For example, what if better test results are achieved if a complex criterion is used to select the best groups? Or what if some of the best groups should not be included in the final EA at all, since their inclusion will worsen the results achieved without them? This topic will most likely require its own detailed study.

Another issue that will also require separate study is the optimal division of groups into subgroups with normalization of subgroups. I have already touched upon this issue in [part 5](https://www.mql5.com/en/articles/14336) even before we started implementing any automation of test stages. We then manually selected nine single instances of trading strategies, three instances for each of the three trading instruments (symbols) used.

It turned out that if you first make three normalized groups of three strategies for each symbol, and then combine them into one final normalized group, then the results in the tests will be somewhat better compared to combining nine single copies of trading strategies into a final normalized group. But we cannot say for sure whether this method of grouping will be optimal. And would it be more preferable for other trading strategies than simply combining them into one group? In general, there is room for further research here too.

Fortunately, we can postpone these two questions until later. To explore them, we would need auxiliary tools that have not yet been implemented. Without them, the work will be much less efficient and will take much more time.

### Selecting and saving groups

It might seem we already have everything we need. Simply take the _SimpleVolumesExpert.mq5_ EA from the previous part, set comma-separated IDs of passes in the _passes\__ input, launch a single tester pass and get the required initialization string saved to the database. Seemingly, the only thing missing is some additional data. But it turned out that information about the pass does not enter the database.

The point is that we have only the results of optimization passes uploaded to the database. Single pass results are not uploaded. As you might remember, uploading is performed inside the _CTesterHandler::ProcessFrames()_ method called from the _OnTesterPass()_ handler in the upper level:

```
//+------------------------------------------------------------------+
//| Handling incoming frames                                         |
//+------------------------------------------------------------------+
void CTesterHandler::ProcessFrames(void) {
// Open the database
   DB::Connect();

// Variables for reading data from frames
   ...

// Go through frames and read data from them
   while(FrameNext(pass, name, id, value, data)) {
      // Convert the array of characters read from the frame into a string
      values = CharArrayToString(data);

      // Form a string with names and values of the pass parameters
      inputs = GetFrameInputs(pass);

      // Form an SQL query from the received data
      query = StringFormat("INSERT INTO passes "
                           "VALUES (NULL, %d, %d, %s,\n'%s',\n'%s');",
                           s_idTask, pass, values, inputs,
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

When a single pass is launched, the handler is not called, as this is not provided for by the single pass event model. This handler is called only in an Expert Advisor running in data frame collection mode. Launching an EA instance of the EA in this mode occurs automatically when optimization starts, but does not occur when a single pass starts. Therefore, it turns out that the existing implementation does not save information about single passes to the database.

We can, of course, leave everything as is and develop an EA that will need to be launched for optimization according to some unnecessary parameter. The goal of such optimization will be to obtain the results of the first pass, after which the optimization will stop. This way, the results of the pass will be entered into the database. But this seems too ugly, so we will go another way.

When running a single pass in the EA, the OnTester() handler will be called upon completion. Therefore, we will have to insert the code for saving the results of a single pass either directly into the handler or into one of the methods called from the handler. Probably, the most appropriate place to insert the method is _CTesterHandler::Tester()_. However, it is worth considering that this method will also be called when the EA completes the optimization pass. This method now contains code that generates and sends the results of the optimization pass through the data frame mechanism.

When a single pass is started, the data for the frame is still generated, but the data frame itself, even if created, cannot be used. If we try to use the _FrameNext()_ function for getting a frame, after creating the frame by the _FrameAdd()_ function in the EA launched in single pass mode, _FrameNext()_ will not read the created frame. It will behave as if no frames were created.

Therefoe, let's do the following. In the _CTesterHandler::Tester()_ handler, we will check whether this pass is a single one or performed as part of optimization. Depending on the result, we will either immediately save the pass results to the database (for a single pass), or create a data frame to send to the main EA (for optimization). Let's add a new method called to save a single pass and another auxiliary method that generates an SQL query to insert the required data into the _passes_ table. We will need the latter because now such an action will be performed in two places of the code, and not in one. Therefore, we will move it to a separate method.

```
//+------------------------------------------------------------------+
//| Optimization event handling class                                |
//+------------------------------------------------------------------+
class CTesterHandler {

    ...

   static void       ProcessFrame(string values);  // Handle single pass data

   // Generate SQL query to insert pass results
   static string     GetInsertQuery(string values, string inputs, ulong pass = 0);
public:
   ...
};
```

We already have the _GetInsertQuery()_ implementation. All we have to do is move the code block from the _ProcessFrames()_ method and call it at the right place in the _ProcessFrames()_ method:

```
//+------------------------------------------------------------------+
//| Generate SQL query to insert pass results                        |
//+------------------------------------------------------------------+
string CTesterHandler::GetInsertQuery(string values, string inputs, ulong pass) {
   return StringFormat("INSERT INTO passes "
                       "VALUES (NULL, %d, %d, %s,\n'%s',\n'%s');",
                       s_idTask, pass, values, inputs,
                       TimeToString(TimeLocal(), TIME_DATE | TIME_SECONDS));
}

//+------------------------------------------------------------------+
//| Handling incoming frames                                         |
//+------------------------------------------------------------------+
void CTesterHandler::ProcessFrames(void) {
   ...

// Go through frames and read data from them
   while(FrameNext(pass, name, id, value, data)) {
      // Convert the array of characters read from the frame into a string
      values = CharArrayToString(data);

      // Form a string with names and values of the pass parameters
      inputs = GetFrameInputs(pass);

      // Form an SQL query from the received data
      query = GetInsertQuery(values, inputs, pass);

      // Add it to the SQL query array
      APPEND(queries, query);
   }

   ...
}
```

To save the data of a single pass we will call a new method _ProcessFrame()_ accepting a string, that is part of an SQL query and contains data about the pass for insertion into the _passes_ table, as a parameter. Within the method itself, we simply connect to the database, form the final SQL query and execute it:

```
//+------------------------------------------------------------------+
//| Handle single pass data                                          |
//+------------------------------------------------------------------+
void CTesterHandler::ProcessFrame(string values) {
// Open the database
   DB::Connect();

// Form an SQL query from the received data
   string query = GetInsertQuery(values, "", 0);

// Execute the request
   DB::Execute(query);

// Close the database
   DB::Close();
}
```

Taking into account the added methods, the pass completion event handler can be modified as follows:

```
//+------------------------------------------------------------------+
//| Handling completion of tester pass for agent                     |
//+------------------------------------------------------------------+
void CTesterHandler::Tester(double custom,   // Custom criteria
                            string params    // Description of EA parameters in the current pass
                           ) {

    ...

// Generate a string with pass data
   data = StringFormat("%s,'%s'", data, params);

// If this is a pass within the optimization,
   if(MQLInfoInteger(MQL_OPTIMIZATION)) {
      // Open a file to write a frame data
      int f = FileOpen(s_fileName, FILE_WRITE | FILE_TXT | FILE_ANSI);

      // Write a description of the EA parameters
      FileWriteString(f, data);

      // Close the file
      FileClose(f);

      // Create a frame with data from the recorded file and send it to the main terminal
      if(!FrameAdd("", 0, 0, s_fileName)) {
         PrintFormat(__FUNCTION__" | ERROR: Frame add error: %d", GetLastError());
      }
   } else {
      // Otherwise, it is a single pass, call the method to add its results to the database
      CTesterHandler::ProcessFrame(data);
   }
}
```

Save the changes made to the _TesterHandler.mqh_ file in the current folder.

Now, after each single pass, information about its results is entered into our database. We are not too interested in various statistical parameters of the pass in terms of the current task. The most important thing for us is the saved initialization string of the normalized strategy group used in the pass. The saved string is what we need the most here.

But the presence of the required initialization strings in one of the _passes_ table columns is not sufficient for their further comfortable use. We also wanted to attach some information to the initialization string. However, it is not worth expanding the set of the _passes_ table columns, since the vast majority of rows in this table will store information about the results of optimization passes, for which additional information is not needed.

Therefore, let's make a new table that will be used to store the selected results. This can already be attributed to the library formation stage.

### Forming the library

Let's not overload the new table with redundant fields containing information that can be obtained from other database tables. For example, if an entry in the new table has a relationship with an entry in the passes table ( _passes_) via an external key, then there is already a creation date. Also, using the pass ID, we can build a chain of connections and determine which project this pass belongs to, and therefore the group of strategies used in the pass.

Considering this, let's create the _strategy\_groups_ table with the following set of fields:

- **id\_pass**. Pass ID from the _passes_ table (external key)
- **name**. The name of the strategy group that will be used to generate enumerations for the strategy group selection input.

The SQL code to create the required table could be as follows:

```
-- Table: strategy_groups
DROP TABLE IF EXISTS strategy_groups;

CREATE TABLE strategy_groups (
    id_pass INTEGER REFERENCES passes (id_pass) ON DELETE CASCADE
                                                ON UPDATE CASCADE
                    PRIMARY KEY,
    name    TEXT
);
```

Let's create the _CGroupsLibrary_ auxiliary class to perform most of the further actions. Its tasks include inserting and retrieving information about strategy groups from the database and forming an mqh file with the actual library of good groups that will be used by the final EA. We will get back to it a bit later. For now, let's make an EA that we will use to form the library.

The existing _SimpleVolumesExpert.mq5_ EA does almost everything it needs to but it still needs some improvement. We planned to use it as the final version of the final EA. So let's save it under a new name _SimpleVolumesStage3.mq5_. Now we should make the necessary additions to the new file. We are missing two things: the ability to specify the name of the group formed for the currently selected passes (in the _passes\__ parameter) and saving the initialization string of this group to the new _strategy\_groups_ table.

The former is quite simple to implement. Let's add a new EA input to be used as the group name later. If the parameter is empty, no saving to the library occurs.

```
input group "::: Saving to library"
input string groupName_  = "";         // - Group name (if empty - no saving)
```

But in case of the former one, we will have to work a little harder. To insert data into the _strategy\_groups_ table, we need to know the ID assigned to the current pass record when inserted into the _passes_ table. Since its value is automatically allocated by the database itself (in the query we simply pass NULL instead of its value), it does not exist in the code as the value of any variable. Therefore, we cannot currently use it in another place where it is needed. We need to somehow define this value.

This can be done in different ways. For example, knowing that the identifiers assigned to new rows form an increasing sequence, you can simply select the value of the currently largest ID after insertion. This can be done if we know for sure that no new strings are currently passed to the _passes_ table. But if another first or second stage optimization is currently underway in parallel, its results may end up in the same database. In this case, we can no longer be sure that the last ID is the one that corresponds to the pass we launched to form the library. In general, this can be done only if we are ready to put up with some limitations and remember them.

A much more reliable method, free from the possible errors described above, is the following one. We can slightly modify the SQL query for inserting data, turning it into a query that will return the generated ID of the new table row as its result. To do this, simply add the "RETURNING rowid" operator to the end of the SQL query. Let's do this in the _GetInsertQuery()_ method, which generates an SQL query to insert a new row into the _passes_ table. Even though the ID column in the _passes_ table is named _id\_pass_, we can name it _rowid_, since it has the appropriate type (INTEGER PRIMARY KEY AUTOINCREMENT) and replaces the hidden _rowid_ column automatically present in SQLite tables .

```
//+------------------------------------------------------------------+
//| Generate SQL query to insert pass results                        |
//+------------------------------------------------------------------+
string CTesterHandler::GetInsertQuery(string values, string inputs, ulong pass) {
   return StringFormat("INSERT INTO passes "
                       "VALUES (NULL, %d, %d, %s,\n'%s',\n'%s') RETURNING rowid;",
                       s_idTask, pass, values, inputs,
                       TimeToString(TimeLocal(), TIME_DATE | TIME_SECONDS));
}
```

We will also need to modify the MQL5 code that sends this request. Currently, we use the _DB::Execute(query)_ method for that. It implies that the _query_ passed to it is not a query that returns any data.

Therefore, the _CDatabase_ class receives the new method _Insert()_, which will execute the passed insert query and return a single read result value. Inside, instead of the _DatabaseExecute()_ function, we will use the _DatabasePrepare()_ function, which then allows us to access the query results:

```
//+------------------------------------------------------------------+
//| Class for handling the database                                  |
//+------------------------------------------------------------------+
class CDatabase {
   ...
public:
   ...
   // Execute a query to the database for insertion with return of the new entry ID
   static ulong      Insert(string query);
};

...

//+------------------------------------------------------------------+
//| Execute a query to the database for insertion returning the      |
//| new entry ID                                                     |
//+------------------------------------------------------------------+
ulong CDatabase::Insert(string query) {
   ulong res = 0;

// Execute the request
   int request = DatabasePrepare(s_db, query);

// If there is no error
   if(request != INVALID_HANDLE) {
      // Data structure for reading a single string of a query result
      struct Row {
         int         rowid;
      } row;

      // Read data from the first result string
      if(DatabaseReadBind(request, row)) {
         res = row.rowid;
      } else {
         // Report an error if necessary
         PrintFormat(__FUNCTION__" | ERROR: Reading row for request \n%s\nfailed with code %d",
                     query, GetLastError());
      }
   } else {
      // Report an error if necessary
      PrintFormat(__FUNCTION__" | ERROR: Request \n%s\nfailed with code %d",
                  query, GetLastError());
   }
   return res;
}
//+------------------------------------------------------------------+
```

I decided to not complicate this method with additional checks that the submitted query is indeed an INSERT query, that it contains a command to return an ID, and that the returned value is not composite. Deviation from these conditions will lead to errors when executing this code, but since this method will be used in only one place in the project, we will try to be able to pass a correct request to it.

Save the changes in the _Database.mqh_ file of the current folder.

The next issue that arose during implementation was how to pass the ID value to the higher level of code, since processing it at the point of receipt led to the need to endow existing methods with external functionality and additional passed parameters. Therefore, we decided to do the following way: the _CTesterHandler_ class received the _s\_idPass_ static property. The ID of the current pass was written into it. From here, we can get the value at any point in the program:

```
//+------------------------------------------------------------------+
//| Optimization event handling class                                |
//+------------------------------------------------------------------+
class CTesterHandler {
   ...
public:
   ...
   static ulong      s_idPass;
};

...
ulong CTesterHandler::s_idPass = 0;

...

//+------------------------------------------------------------------+
//| Handle single pass data                                          |
//+------------------------------------------------------------------+
void CTesterHandler::ProcessFrame(string values) {
// Open the database
   DB::Connect();

// Form an SQL query from the received data
   string query = GetInsertQuery(values, "", 0);

// Execute the request
   s_idPass = DB::Insert(query);

// Close the database
   DB::Close();
}
```

Save the changes made to the _TesterHandler.mqh_ file in the current folder.

Now it is time to return to the declared _CGroupsLibrary_ auxiliary class. We ended up with the need to declare two public methods in it - one private method and one static array:

```
//+------------------------------------------------------------------+
//| Class for working with a library of selected strategy groups     |
//+------------------------------------------------------------------+
class CGroupsLibrary {
private:
   // Exporting group names and initialization strings extracted from the database as MQL5 code
   static void       ExportParams(string &p_names[], string &p_params[]);

public:
   // Add the pass name and ID to the database
   static void       Add(ulong p_idPass, string p_name);

   // Export passes to mqh file
   static void       Export(string p_idPasses);

   // Array to fill with initialization strings from mqh file
   static string     s_params[];
};
```

In the library-forming EA, only the _Add()_ method will be used. It will receive the pass ID and group name to save to be saved to the library. The method code itself is very simple: form an SQL query for inserting a new entry to the _strategy\_groups_ table out of the input data and execute it.

```
//+------------------------------------------------------------------+
//| Add the pass name and ID to the database                         |
//+------------------------------------------------------------------+
void CGroupsLibrary::Add(ulong p_idPass, string p_name) {
   string query = StringFormat("INSERT INTO strategy_groups VALUES(%d, '%s')",
                               p_idPass, p_name);

// Open the database
   if(DB::Connect()) {
      // Execute the request
      DB::Execute(query);

      // Close the database
      DB::Close();
   }
}
```

Now, to complete the development of the library formation tool, we only need to add calling the Add() method to the _SimpleVolumesStage3.mq5_ EA after the tester pass is complete:

```
//+------------------------------------------------------------------+
//| Test results                                                     |
//+------------------------------------------------------------------+
double OnTester(void) {
   // Handle the completion of the pass in the EA object
   double res = expert.Tester();

   // If the group name is not empty, save the pass to the library
   if(groupName_ != "") {
      CGroupsLibrary::Add(CTesterHandler::s_idPass, groupName_);
   }
   return res;
}
```

Let's save the changes made to the _SimpleVolumesStage3.mq5_ and _GroupsLibrary.mqh_ files in the current folder. If we add stubs for the rest of the _CGroupsLibrary_ class methods, then we can already use the compiled _SimpleVolumesStage3.mq5_ EA.

### Filling in the library

Let's try to form a library from the nine good pass IDs selected earlier. To do this, launch the _SimpleVolumesStage3.ex5_ EA in the tester specifying various combinations selected from nine IDs in the _passes\__ input. In the _groupName\__ input, we will set a clear name that reflects the composition of the current group of single instances of trading strategies combined into one group.

After several runs, let's look at the results that appear in the _strategy\_groups_ table adding some parameters for the passes made with different groups for informational purposes. For example, the following SQL query will help us with this:

```
SELECT sg.id_pass,
       sg.name,
       p.custom_ontester,
       p.sharpe_ratio,
       p.profit,
       p.profit_factor,
       p.equity_dd_relative
  FROM strategy_groups sg
       JOIN
       passes p ON sg.id_pass = p.id_pass;
```

The query resulted in the following table:

![](https://c.mql5.com/2/124/6297885518900__1.png)

Fig. 1. Group library composition

In the _name_ column, we see the names of the groups, which reflect the trading instruments (symbols), timeframes and the number of instances of trading strategies used in this group. For example, the presence of "EUR-GBP-USD" means that this group includes instances of trading strategies that work on three symbols: EURGBP, EURUSD and GBPUSD. If the group name starts with "Only EURGBP", then it includes instances of strategies only for the EURGBP symbol. The timeframes used are denoted in a similar way. The number of instances of trading strategies is specified at the end of the name. For example, "3x16 items" indicates that this group combines three standardized groups of 16 strategies each.

The _custom\_ontester_ column displays the normalized average annual profit for each group. It should be noted that the range of values for this parameter exceeded the expected value, so in the future it would be necessary to understand the reasons for this phenomenon. For example, the results of groups where only GBPUSD was used were significantly higher than those of groups with several symbols. The best result was saved last in line 20. In this group, we have included subgroups that yield the best results for each symbol and one or more timeframes.

### Exporting the library

The next step is to transfer the group library from the database to an mqh file that can be connected to the final EA. To do this, let's implement the methods in the _CGroupsLibrary_ class responsible for export, and one more auxiliary EA, which will be used to run these methods.

In the _Export()_ method, we will get from the database and add to the corresponding arrays the names of library groups and their initialization strings. The generated arrays will be passed to the next method _ExportParams()_:

```
//+------------------------------------------------------------------+
//| Exporting passes to mqh file                                     |
//+------------------------------------------------------------------+
void CGroupsLibrary::Export(string p_idPasses) {
// Array of group names
   string names[];

// Array of group initialization strings
   string params[];

// If the connection to the main database is established,
   if(DB::Connect()) {
      // Form a request to receive passes with the specified IDs
      string query = "SELECT sg.id_pass,"
                     "       sg.name,"
                     "       p.params"
                     "  FROM strategy_groups sg"
                     "       JOIN"
                     "       passes p ON sg.id_pass = p.id_pass";

      query = StringFormat("%s "
                           "WHERE p.id_pass IN (%s);",
                           query, p_idPasses);

      // Prepare and execute the request
      int request = DatabasePrepare(DB::Id(), query);

      // If the request is successful
      if(request != INVALID_HANDLE) {
         // Structure for reading results
         struct Row {
            ulong          idPass;
            string         name;
            string         params;
         } row;

         // For all query results, add the name and initialization string to the arrays
         while(DatabaseReadBind(request, row)) {
            APPEND(names, row.name);
            APPEND(params, row.params);
         }
      }

      DB::Close();

      // Export to mqh file
      ExportParams(names, params);
   }
}
```

In the _ExportParams()_ method, form a string with MQL5 code, which will create an enumeration (enum) with a given name _ENUM\_GROUPS\_LIBRARY_ and fill it with elements. Each element will have a comment containing the group name. Next, the code will declare a static string array _CGroupsLibrary::s\_params\[\]_, which will be filled with initialization strings for groups from the library. Each initialization string will be preprocessed: all line feeds will be replaced with spaces, and a backslash will be added before double quotes. This is necessary in order to place the initialization string inside double quotes in the generated code.

Once the code is fully formed in the _data_ variable, we create the file named _ExportedGroupsLibrary.mqh_ and save the received code in it.

```
//+------------------------------------------------------------------+
//| Export group names extracted from the database and               |
//| initialization strings in the form of MQL5 code                  |
//+------------------------------------------------------------------+
void CGroupsLibrary::ExportParams(string &p_names[], string &p_params[]) {
   // ENUM_GROUPS_LIBRARY enumeration header
   string data = "enum ENUM_GROUPS_LIBRARY {\n";

   // Fill the enumeration with group names
   FOREACH(p_names, { data += StringFormat("   GL_PARAMS_%d, // %s\n", i, p_names[i]); });

   // Close the enumeration
   data += "};\n\n";

   // Group initialization string array header and its opening bracket
   data += "string CGroupsLibrary::s_params[] = {";

   // Fill the array by replacing invalid characters in the initialization strings
   string param;
   FOREACH(p_names, {
      param = p_params[i];
      StringReplace(param, "\r", "");
      StringReplace(param, "\n", " ");
      StringReplace(param, "\"", "\\\"");
      data += StringFormat("\"%s\",\n", param);
   });

   // Close the array
   data += "};\n";

// Open the file to write data
   int f = FileOpen("ExportedGroupsLibrary.mqh", FILE_WRITE | FILE_TXT | FILE_ANSI);

// Write the generated code
   FileWriteString(f, data);

// Close the file
   FileClose(f);
}
```

Next comes the very important part:

```
// Connecting the exported mqh file.
// It will initialize the CGroupsLibrary::s_params[] static variable
// and ENUM_GROUPS_LIBRARY enumeration
#include "ExportedGroupsLibrary.mqh"
```

We include the file that will be received after the export directly into the _GroupsLibrary.mqh_ file. In this case, the final EA will only need to include this file in order to be able to use the exported library. This approach creates a small inconvenience: in order to be able to compile the EA that will handle the library export, the _ExportedGroupsLibrary.mqh_ file, which appears only after export, should already exist. However, only the presence of this file is important, not its contents. Therefore, we should simply create an empty file with this name in the current folder, and compilation will proceed without errors.

To run the EA method, we need a script or EA, in which this will happen. It might look like this:

```
//+------------------------------------------------------------------+
//| Inputs                                                           |
//+------------------------------------------------------------------+
input group "::: Exporting from library"
input string     passes_ = "802150,802151,802152,802153,802154,"
                           "802155,802156,802157,802158,802159,"
                           "802160,802161,802162,802164,802165,"
                           "802166,802167,802168,802169,802173";    // - Comma-separated IDs of the saved passes

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit() {
// Call the group library export method
   CGroupsLibrary::Export(passes_);

// Successful initialization
   return(INIT_SUCCEEDED);
}

void OnTick() {
   ExpertRemove();
}
```

By changing the _passes\__ parameter, we can choose the composition and order, in which the groups will be exported from the library to the database. After running the EA once on the chart, the ExportedGroupsLibrary.mqh, file will appear in the terminal data folder. It should be transferred to the current folder containing the project code.

### Creating the final EA

We have finally reached the final phase. All that remains is to make some minor changes to the _SimpleVolumesExpert.mq5_ EA. First, we need to include the _GroupsLibrary.mqh_ file:

```
#include "GroupsLibrary.mqh"
```

Next, replace the _passes\__ input with a new one allowing us to select a group from the library:

```
input group "::: Selection for the group"
input ENUM_GROUPS_LIBRARY       groupId_     = -1;    // - Group from the library
```

In the _OnInit()_ function, instead of getting initialization strings from the database by pass IDs (as before), we will now simply take the initialization string from the _CGroupsLibrary::s\_params\[\]_ array with an index corresponding to the selected value of the _groupId\__ input:

```
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit() {
   ...

// Initialization string with strategy parameter sets
   string strategiesParams = NULL;

// If the selected strategy group index from the library is valid, then
   if(groupId_ >= 0 && groupId_ < ArraySize(CGroupsLibrary::s_params)) {
      // Take the initialization string from the library for the selected group
      strategiesParams = CGroupsLibrary::s_params[groupId_];
   }

// If the strategy group from the library is not specified, then we interrupt the operation
   if(strategiesParams == NULL) {
      return INIT_FAILED;
   }

   ...

// Successful initialization
   return(INIT_SUCCEEDED);
}
```

Save the changes made to the _SimpleVolumesExpert.mq5_ file in the current folder.

Since we have added comments with names to the _ENUM\_GROUPS\_LIBRARY_ enumeration elements, then we will be able to see understandable names, and not just a sequence of numbers, in the dialog for selecting the EA parameters:

![](https://c.mql5.com/2/124/2025-03-12_18_42_24.png)

Fig. 2. Selecting a group from the library by name in the EA parameters

Let's run the EA with the last group from the list and look at the result:

![](https://c.mql5.com/2/124/2560163814658__1.png)

![](https://c.mql5.com/2/124/6408298749178__1.png)

Fig. 3. Results of testing the final EA with the most attractive group from the library

It is clear that the results for the average annual normalized profit indicator were close to those stored in the database. Small differences are primarily due to the fact that the final EA used a standardized group (this can be verified by looking at the value of the maximum relative drawdown, which is approximately 10% of the deposit used). When generating the initialization string for this group in the _SimpleVolumesStage3.ex5_ EA, the group was not yet standardized during the pass, so the drawdown there was approximately 5.4%.

### Conclusion

We have received the final EA, which can work independently of the database filled in the optimization process. Perhaps, we will return to this issue again, since practice can make its own adjustments, and the method proposed in this article may turn out to be less convenient than some other. But in any case, achieving the set goal is a step forward.

While working on the code for this article, new circumstances were discovered that require further investigation. For example, it turned out that the results of testing this EA are sensitive not only to the quote server, but also to the symbol selected as the main one in the strategy tester settings. We may need to make some adjustments to the optimization automation in the first and second stages. But more about that next time.

Finally, I want to make a warning that was implicitly present before. I never said in the previous parts that following the proposed direction will allow you to get a guaranteed profit. On the contrary, we received disappointing test results at some points. Also, despite the efforts expended to prepare the EA for real trading, we are unlikely to be able to say at some point that we have done everything possible and impossible to ensure the correct operation of the EA on real accounts. This is a perfect outcome that can and should be strived for, but achieving it always seems like a matter of the foggy future. This, however, does not prevent us from approaching it.

All results presented in this article and all previous articles in the series are based only on historical testing data and are not a guarantee of any profit in the future. The work within this project is of a research nature. All published results can be used by anyone at their own risk.

Thank you for your attention! See you soon!

### Archive contents

| # | Name | Version | Description | Recent changes |
| --- | --- | --- | --- | --- |
|  | MQL5/Experts/Article.15360 |
| --- | --- |
| 1 | Advisor.mqh | 1.04 | EA base class | [Part 10](https://www.mql5.com/en/articles/14739) |
| --- | --- | --- | --- | --- |
| 2 | Database.mqh | 1.04 | Class for handling the database | [Part 17](https://www.mql5.com/en/articles/15360) |
| --- | --- | --- | --- | --- |
| 3 | ExpertHistory.mqh | 1.00 | Class for exporting trade history to file | [Part 16](https://www.mql5.com/en/articles/15330) |
| --- | --- | --- | --- | --- |
| 4 | ExportedGroupsLibrary.mqh | — | Generated file listing strategy group names and the array of their initialization strings | [Part 17](https://www.mql5.com/en/articles/15360) |
| --- | --- | --- | --- | --- |
| 5 | Factorable.mqh | 1.01 | Base class of objects created from a string | [Part 10](https://www.mql5.com/en/articles/14739) |
| --- | --- | --- | --- | --- |
| 6 | GroupsLibrary.mqh | 1.00 | Class for working with a library of selected strategy groups | [Part 17](https://www.mql5.com/en/articles/15360) |
| --- | --- | --- | --- | --- |
| 7 | HistoryReceiverExpert.mq5 | 1.00 | EA for replaying the history of deals with the risk manager | [Part 16](https://www.mql5.com/en/articles/15330) |
| --- | --- | --- | --- | --- |
| 8 | HistoryStrategy.mqh | 1.00 | Class of the trading strategy for replaying the history of deals | [Part 16](https://www.mql5.com/en/articles/15330) |
| --- | --- | --- | --- | --- |
| 9 | Interface.mqh | 1.00 | Basic class for visualizing various objects | [Part 4](https://www.mql5.com/en/articles/14246) |
| --- | --- | --- | --- | --- |
| 10 | LibraryExport.mq5 | 1.00 | EA that saves initialization strings of selected passes from the library to the ExportedGroupsLibrary.mqh file | [Part 17](https://www.mql5.com/en/articles/15360) |
| --- | --- | --- | --- | --- |
| 11 | Macros.mqh | 1.02 | Useful macros for array operations | [Part 16](https://www.mql5.com/en/articles/15330) |
| --- | --- | --- | --- | --- |
| 12 | Money.mqh | 1.01 | Basic money management class | [Part 12](https://www.mql5.com/en/articles/14764) |
| --- | --- | --- | --- | --- |
| 13 | NewBarEvent.mqh | 1.00 | Class for defining a new bar for a specific symbol | [Part 8](https://www.mql5.com/en/articles/14574) |
| --- | --- | --- | --- | --- |
| 14 | Receiver.mqh | 1.04 | Base class for converting open volumes into market positions | [Part 12](https://www.mql5.com/en/articles/14764) |
| --- | --- | --- | --- | --- |
| 15 | SimpleHistoryReceiverExpert.mq5 | 1.00 | Simplified EA for replaying the history of deals | [Part 16](https://www.mql5.com/en/articles/15330) |
| --- | --- | --- | --- | --- |
| 16 | SimpleVolumesExpert.mq5 | 1.20 | EA for parallel operation of several groups of model strategies. The parameters will be taken from the built-in group library. | [Part 17](https://www.mql5.com/en/articles/15330) |
| --- | --- | --- | --- | --- |
| 17 | SimpleVolumesStage3.mq5 | 1.00 | The EA that saves a generated standardized group of strategies to a library of groups with a given name. | [Part 17](https://www.mql5.com/en/articles/15330) |
| --- | --- | --- | --- | --- |
| 18 | SimpleVolumesStrategy.mqh | 1.09 | Class of trading strategy using tick volumes | [Part 15](https://www.mql5.com/en/articles/15294) |
| --- | --- | --- | --- | --- |
| 19 | Strategy.mqh | 1.04 | Trading strategy base class | [Part 10](https://www.mql5.com/en/articles/14739) |
| --- | --- | --- | --- | --- |
| 20 | TesterHandler.mqh | 1.03 | Optimization event handling class | [Part 17](https://www.mql5.com/en/articles/15360) |
| --- | --- | --- | --- | --- |
| 21 | VirtualAdvisor.mqh | 1.06 | Class of the EA handling virtual positions (orders) | [Part 15](https://www.mql5.com/en/articles/15294) |
| --- | --- | --- | --- | --- |
| 22 | VirtualChartOrder.mqh | 1.00 | Graphical virtual position class | [Part 4](https://www.mql5.com/en/articles/14246) |
| --- | --- | --- | --- | --- |
| 23 | VirtualFactory.mqh | 1.04 | Object factory class | [Part 16](https://www.mql5.com/en/articles/15330) |
| --- | --- | --- | --- | --- |
| 24 | VirtualHistoryAdvisor.mqh | 1.00 | Trade history replay EA class | [Part 16](https://www.mql5.com/en/articles/15330) |
| --- | --- | --- | --- | --- |
| 25 | VirtualInterface.mqh | 1.00 | EA GUI class | [Part 4](https://www.mql5.com/en/articles/14246) |
| --- | --- | --- | --- | --- |
| 26 | VirtualOrder.mqh | 1.04 | Class of virtual orders and positions | [Part 8](https://www.mql5.com/en/articles/14574) |
| --- | --- | --- | --- | --- |
| 27 | VirtualReceiver.mqh | 1.03 | Class for converting open volumes to market positions (receiver) | [Part 12](https://www.mql5.com/en/articles/14764) |
| --- | --- | --- | --- | --- |
| 28 | VirtualRiskManager.mqh | 1.02 | Risk management class (risk manager) | [Part 15](https://www.mql5.com/en/articles/15294) |
| --- | --- | --- | --- | --- |
| 29 | VirtualStrategy.mqh | 1.05 | Class of a trading strategy with virtual positions | [Part 15](https://www.mql5.com/en/articles/15294) |
| --- | --- | --- | --- | --- |
| 30 | VirtualStrategyGroup.mqh | 1.00 | Class of trading strategies group(s) | [Part 11](https://www.mql5.com/en/articles/14741) |
| --- | --- | --- | --- | --- |
| 31 | VirtualSymbolReceiver.mqh | 1.00 | Symbol receiver class | [Part 3](https://www.mql5.com/en/articles/14148) |
| --- | --- | --- | --- | --- |

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/15360](https://www.mql5.com/ru/articles/15360)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/15360.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/15360/mql5.zip "Download MQL5.zip")(72.95 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/482834)**
(7)


![Viktor Kudriavtsev](https://c.mql5.com/avatar/2020/8/5F496EFA-E815.jpg)

**[Viktor Kudriavtsev](https://www.mql5.com/en/users/ale5312)**
\|
6 Sep 2024 at 07:00

**Yuriy Bykov [#](https://www.mql5.com/ru/forum/471894#comment_54499751):**

The work of the Expert Advisor consists of two parts: opening virtual positions and synchronisation of open virtual positions with real ones. The set TF is used only in the first part to determine the opening signal. And synchronisation should ideally be performed on each tick or at least on each new bar of the minimum timeframe M1, because at any moment the virtual position may reach TP or SL.

In the VirtualAdvisor::Tick() method, there is a check at the beginning for the occurrence of a new bar on all monitored symbols and timeframes, including M1. If it has not occurred, the Expert Advisor does not perform any more actions. It will do something else only when a new bar occurs on M1. In this case, you can optimise in OHLC mode on M1 and get almost the same results when the EA works on the chart (where there are all ticks). And optimisation is much faster this way. The line of code you mentioned is just a safety net in case we don't need to track a new bar on M1 in the strategy. This way it is guaranteed to be tracked at least on one symbol.

If you want, you can, of course, disable this mode of operation through the variable useOnlyNewBars\_ = false. Then the Expert Advisor will check and synchronise positions on every available tick.

I see. But for example, can we make synchronisation of positions work on every tick, and opening of virtual (new) positions occurs when a new bar occurs on the TF specified in the strategy (m15,m30,h1)?

**Yuriy Bykov [#](https://www.mql5.com/ru/forum/471894#comment_54499751):**

Opening of a new M1 bar can occur inside a bar of a higher timeframe. Note that SignalForOpen() uses the current timeframe, which is usually H1, M30 or M15. Therefore, there will no longer be a coincidence of the opening and closing prices of the current timeframe. In addition, this check comes only when the tick volume of the current bar on the current timeframe has significantly exceeded the typical tick volume of one bar. This cannot happen on the first tick, when the tick volume is only 1.

I don't understand you a little bit here. Yes, SignalForOpen() uses the TF set in the settings of the current virtual strategy instance, I can see that. But for example, if I want the EA to work strictly on the closed last bars, then here I have to specify units instead of zeros.

```
      if(m_volumes[0] > avrVolume * (1 + m_signalDeviation + m_ordersTotal * m_signaAddlDeviation)) {
         // если цена открытия свечи меньше текущей цены (закрытия), то
         if(iOpen(m_symbol, m_timeframe, 0) < iClose(m_symbol, m_timeframe, 0)) {
```

I should specify units instead of zeros ? Do I understand correctly ?

![Yuriy Bykov](https://c.mql5.com/avatar/avatar_na2.png)

**[Yuriy Bykov](https://www.mql5.com/en/users/antekov)**
\|
6 Sep 2024 at 10:16

**Viktor Kudriavtsev [#](https://www.mql5.com/ru/forum/471894#comment_54501754):**

For example, can we make position synchronisation work on every tick, and opening of virtual (new) positions occurs when a new bar occurs on the TF specified in the strategy (m15,m30,h1)?

Yes, this will be the case if useOnlyNewBars\_ = false. This variable is not used by strategies, they themselves determine when to check for an opening signal and when to open positions when a signal has been received earlier. For example, only when a new bar occurs on H1. In this case, you must then modify the code so that the signal received in the middle of the bar survives until the beginning of the next bar. Now the received signal is used immediately (leads to the opening of virtual positions), so it is not saved anywhere.

I don't understand you a bit here. Yes, SignalForOpen() uses the TF set in the settings of the current instance of the virtual strategy, I can see that. But for example, if I want the EA to work strictly on the closed last bars, then here  I should specify units instead of zeros ? Do I understand correctly ?

If by the words"EA worked strictly on closed last bars" you mean that when the tick volume exceeds the threshold value on the current bar to determine the direction of the signal to open, we will take the previous bar and look at its direction, then you have understood everything correctly.

![Viktor Kudriavtsev](https://c.mql5.com/avatar/2020/8/5F496EFA-E815.jpg)

**[Viktor Kudriavtsev](https://www.mql5.com/en/users/ale5312)**
\|
11 Sep 2024 at 16:18

Yuri hello. I have an error when executing the [Expert Advisor](https://www.mql5.com/en/market/mt5/ "A Market of Applications for the MetaTrader 5 and MetaTrader 4") SimpleVolumesStage3.mq5 and saving information to the database:

```
2024.09.11 21:02:09.909 Core 1  2024.09.06 23:54:59
2024.09.11 21:02:09.909 Core 1  2024.09.06 23:54:59   database error, FOREIGN KEY constraint failed
2024.09.11 21:02:09.909 Core 1  2024.09.06 23:54:59   CDatabase::Execute | ERROR: 5619 in query
2024.09.11 21:02:09.909 Core 1  2024.09.06 23:54:59   INSERT INTO strategy_groups VALUES(0, 'EA_EG_EU (H1, M30, M15, 9x16 items)')
2024.09.11 21:02:09.909 Core 1  final balance 24603.99 USD
```

What does it mean and how to fix it? The table was added to the database using your query from the article.

![Viktor Kudriavtsev](https://c.mql5.com/avatar/2020/8/5F496EFA-E815.jpg)

**[Viktor Kudriavtsev](https://www.mql5.com/en/users/ale5312)**
\|
12 Sep 2024 at 15:29

Yuri, I have been looking through your code and I see that the error occurs a little earlier in the CDatabase::Insert function, the log writes this:

```
2024.09.12 20:14:11.248 Core 1  2024.09.06 23:54:59   CDatabase::Insert | ERROR: Reading row for request
2024.09.12 20:14:11.248 Core 1  2024.09.06 23:54:59   INSERT INTO passes VALUES (NULL, 0, 0, 10000.00,0.00,11096.20,21542.31,-10446.11,92.51,-63.35,630.89,39.00,444.04,53.00,-376.27,52.00,-376.27,52.00,9430.69,569.31,5.69,5.69,569.31,9325.11,683.96,6.83,6.83,683.96,2.15,2.06,16.22,3.44,3736.76,8435.00,5170.00,3042.00,2128.00,2766.00,2404.00,1706.00,1336.00,6.00,4.00,99.11,8122.90,'class CVirtualStrategyGroup([\
2024.09.12 20:14:11.248 Core 1  2024.09.06 23:54:59           class CVirtualStrategyGroup([\
2024.09.12 20:14:11.248 Core 1  2024.09.06 23:54:59           class CVirtualStrategyGroup([\
2024.09.12 20:14:11.248 Core 1  2024.09.06 23:54:59           class CSimpleVolumesStrategy("CADCHF",16385,220,1.40,1.70,150,2200.00,200.00,46000,24)\
2024.09.12 20:14:11.248 Core 1  2024.09.06 23:54:59          ],66.401062),class CVirtualStrategyGroup([\
\
.....\
\
2024.09.12 20:38:26.905	Core 1	2024.09.06 23:54:59          ],13.365410),class CVirtualStrategyGroup([\
2024.09.12 20:38:26.905	Core 1	2024.09.06 23:54:59           class CSimpleVolumesStrategy("CADJPY",15,132,0.40,1.90,0,7200.00,600.00,45000,27)\
2024.09.12 20:38:26.905	Core 1	2024.09.06 23:54:59          ],13.365410),\
2024.09.12 20:38:26.905	Core 1	2024.09.06 23:54:59          ],2.970797),\
2024.09.12 20:38:26.905	Core 1	2024.09.06 23:54:59          ],1.462074)',
2024.09.12 20:38:26.905	Core 1	2024.09.06 23:54:59   '',
2024.09.12 20:38:26.905	Core 1	2024.09.06 23:54:59   '2024.09.06 23:54:59') RETURNING rowid;
2024.09.12 20:38:26.905	Core 1	2024.09.06 23:54:59   failed with code 5039
```

Cannot execute

```
      if(DatabaseReadBind(request, row)) {
```

What can this be related to? The second stage is passed and the test itself passes (the [Expert Advisor](https://www.mql5.com/en/market/mt5/ "A Market of Applications for the MetaTrader 5 and MetaTrader 4") trades and passes from the database are loaded).

![Yuriy Bykov](https://c.mql5.com/avatar/avatar_na2.png)

**[Yuriy Bykov](https://www.mql5.com/en/users/antekov)**
\|
12 Sep 2024 at 18:35

Hello Victor.

I will be back soon to continue working on this [project](https://www.mql5.com/en/articles/7863 "Article: Projects allow you to create profitable trading robots! But it's not for sure ") and will try to sort out the errors I found. Thank you for finding them. I managed to reproduce some of the errors you wrote about earlier. They turned out to be related to the fact that in later parts, edits were made that were aimed at one thing, but in addition had an impact on other things that were not considered in the next article. This influence created errors. In the next article we will go through all the steps of automated optimisation again, eliminating all the errors that were detected.

![Creating a Trading Administrator Panel in MQL5 (Part IX): Code Organization (III): Communication Module](https://c.mql5.com/2/124/Creating_a_Trading_Administrator_Panel_in_MQL5_Part_X___LOGO.png)[Creating a Trading Administrator Panel in MQL5 (Part IX): Code Organization (III): Communication Module](https://www.mql5.com/en/articles/17044)

Join us for an in-depth discussion on the latest advancements in MQL5 interface design as we unveil the redesigned Communications Panel and continue our series on building the New Admin Panel using modularization principles. We'll develop the CommunicationsDialog class step by step, thoroughly explaining how to inherit it from the Dialog class. Additionally, we'll leverage arrays and ListView class in our development. Gain actionable insights to elevate your MQL5 development skills—read through the article and join the discussion in the comments section!

![Neural Networks in Trading: A Complex Trajectory Prediction Method (Traj-LLM)](https://c.mql5.com/2/89/logo-midjourney_image_15595_398_3845__1.png)[Neural Networks in Trading: A Complex Trajectory Prediction Method (Traj-LLM)](https://www.mql5.com/en/articles/15595)

In this article, I would like to introduce you to an interesting trajectory prediction method developed to solve problems in the field of autonomous vehicle movements. The authors of the method combined the best elements of various architectural solutions.

![Price Action Analysis Toolkit Development (Part 17): TrendLoom EA Tool](https://c.mql5.com/2/125/Price_Action_Analysis_Toolkit_Development_Part_17.png)[Price Action Analysis Toolkit Development (Part 17): TrendLoom EA Tool](https://www.mql5.com/en/articles/17329)

As a price action observer and trader, I've noticed that when a trend is confirmed by multiple timeframes, it usually continues in that direction. What may vary is how long the trend lasts, and this depends on the type of trader you are, whether you hold positions for the long term or engage in scalping. The timeframes you choose for confirmation play a crucial role. Check out this article for a quick, automated system that helps you analyze the overall trend across different timeframes with just a button click or regular updates.

![From Basic to Intermediate: Passing by Value or by Reference](https://c.mql5.com/2/90/logo-15345.png)[From Basic to Intermediate: Passing by Value or by Reference](https://www.mql5.com/en/articles/15345)

In this article, we will practically understand the difference between passing by value and passing by reference. Although this seems like something simple and common and not causing any problems, many experienced programmers often face real failures in working on the code precisely because of this small detail. Knowing when, how, and why to use pass by value or pass by reference will make a huge difference in our lives as programmers. The content presented here is intended solely for educational purposes. Under no circumstances should the application be viewed for any purpose other than to learn and master the concepts presented.

[![](https://www.mql5.com/ff/si/q0vxp9pq0887p07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fvps%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Duse.vps%26utm_content%3Drent.vps%26utm_campaign%3D0622.MQL5.com.Internal&a=rktadgjlwhobyedohbrepzshvpcqrlpo&s=a93cef75a53eb5da24c98e0068b3c2b96015191a0af0d1857f5b4dd22e55e7bf&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=afypxqxyxipbvnivniefaubzsbhgiqnu&ssn=1769092030926447023&ssn_dr=0&ssn_sr=0&fv_date=1769092030&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F15360&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Developing%20a%20multi-currency%20Expert%20Advisor%20(Part%2017)%3A%20Further%20preparation%20for%20real%20trading%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176909203011498592&fz_uniq=5049143556158891453&sv=2552)

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