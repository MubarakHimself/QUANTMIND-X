---
title: Integrating MQL-based Expert Advisors and databases (SQL Server, .NET and C#)
url: https://www.mql5.com/en/articles/2895
categories: Integration, Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T21:15:49.974185
---

[![](https://www.mql5.com/ff/si/3fgkjn78mkxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=ocndbzpeklfncxysjbwfhhbalbrsdbtv&s=a4309643278437a00bdd33c5809fc6b4b4032749c00fccd07b3b84e7b8b45126&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=ujsliqkmvojuzhthmoclxzbxatkgwpbu&ssn=1769192148266793505&ssn_dr=0&ssn_sr=0&fv_date=1769192148&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F2895&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Integrating%20MQL-based%20Expert%20Advisors%20and%20databases%20(SQL%20Server%2C%20.NET%20and%20C%23)%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176919214849812095&fz_uniq=5071701755189734611&sv=2552)

MetaTrader 5 / Examples


### Introduction. MQL-based experts and databases

Questions related to integrating the work with databased into Expert Advisors written in MQL5 often appear on the forums. Interest in this topic is not surprising. Databases are very good as a means of saving data. Unlike the terminal logs, the data do not disappear from the databases. They are easy to sort and filter, choosing only the required ones. A database can be used to pass the necessary information to an expert — for example, certain commands. And most importantly — the obtained data can be analyzed from different perspectives and processed statistically. For example, writing a one-line query is enough to find out the average and total profit for a specified time for each currency pair. And now imagine how long it takes to manually calculate this for the account history in the trading terminal.

Unfortunately, MetaTrader does not provide built-in tools for interacting with database servers. The problem can only be solved by importing functions from DLL files. The task is not simple, but feasible.

I have done this multiple times, and I decided to share my experience in this article. As an example, this is how the interaction of MQL5 experts with the Microsoft SQL Server database server is organized. To create a DLL file for the experts to import functions for working with the database, the Microsoft.NET platform and the C# language were used. The article describes the process of creating and preparing a DLL file, and then the import of its functions into an expert written in MQL5. The expert code provided as an example is very simple. It requires minimal changes to be made in order to be compiled in MQL4.

### Preparation for work

The following will be needed for work.

1. An installed MetaTrader 5 terminal with an active trading account. You can use not only demo accounts, but real accounts as well — using the test expert poses no risk to the deposit.

2. Installed instance of the Microsoft SQL Server database server. You can use a database on another computer and connect to it over the network. The free Express Edition can be downloaded from the Microsoft site, its limitations are not significant for most users. You can download it here: [https://www.microsoft.com/en-us/sql-server/sql-server-editions-express](https://www.mql5.com/go?link=https://www.microsoft.com/en-us/sql-server/sql-server-editions-express "https://www.microsoft.com/en-us/sql-server/sql-server-editions-express"). Microsoft sometimes changes the links on their site. Therefore, if the direct link does not work, simply type a phrase like "SQL Server Express download" in any search engine. If you are installing SQL Server for the first time, there may be some difficulties with installation. In particular, it may require additional components to be installed (specifically, PowerShell and .NET 4.5) on older versions of the OS. Also, SQL Server and VS C++ 2017 may sometimes conflict, with the installer asking to restore С++. You can do this through "Control Panel", "Programs", "Programs and Features", "VS C++ 2017", "Change", "Repair". These problems do not always occur and can be easily solved.

3. Integrated Development Environment using .NET and C#. I personally use Microsoft Visual Studio (which also has a free version), therefore, the examples will be given for it. You can use a different development environment and even another programming language. But then you will have to think of a way to implement the given examples in your environment and in the chosen language.
4. Tool for exporting functions from DLL files written in .NET to unmanaged code. MQL-based experts cannot work with managed .NET code. Therefore, the resulting DLL file should be specifically prepared, providing the ability to export functions. Various ways to achieve this are described on the web. I used the "UnmanagedExports" package, created by Robert Giesecke. If you use Microsoft Visual Studio version 2012 or higher, you can add it to the project directly from the IDE menu. The way to do this will be discussed further.

Apart from installing the required programs, it is also necessary to perform another preparatory operation. For a number of reasons, the "UnmanagedExports" cannot work if "Russian (Russia)" is selected as the language for non-Unicode programs in the language settings of your computer. There may be problems with other languages as well, unless it is "English (US)". To install it, open the control panel. Navigate to the "Regional and Language Options" tab, then go to the "Advanced" tab. On the tab "language for non-Unicode programs", press "Change system locale...". If "English (US)" is set there, everything is fine. If there is something else, change it into "English (US)" and restart the computer.

If this is not done, then syntax error will occur in the ".il" files when compiling the project at the execution stage of the "UnmanagedExports" scripts. They cannot be fixed. Even if your project is very simple, and there are certainly no errors in the C# code, errors will still occur in ".il" files, and you will be unable to export the functions from the project to unmanaged code.

This applies only to 64-bit applications. 32-bit applications can be handled by other means, which do not require changing the system locale. For example, you can use the DllExporter.exe program, which can be downloaded here: [https://www.codeproject.com/Articles/37675/Simple-Method-of-DLL-Export-without-C-CLI](https://www.mql5.com/go?link=https://www.codeproject.com/Articles/37675/Simple-Method-of-DLL-Export-without-C-CLI "https://www.codeproject.com/Articles/37675/Simple-Method-of-DLL-Export-without-C-CLI").

Changing the system locale will make some applications inoperable. Unfortunately, these inconveniences will have to be reconciled, but it is not for long. Changing the locale is only necessary when you compile the project. After the compilation succeeds, the system locale can be switched back.

### Creating a DLL file

Open Visual Studio and create a new Visual C# project, selecting "Class Library" as its type. Let us name it MqlSqlDemo. In the project properties, in the "Build" section, it is necessary to configure the "Platform target". There, it is necessary to change "Any CPU" to "x64" (both in the Debug configuration and in the Release configuration). This is due to the peculiarities of exporting functions to unmanaged code — specifying the processor type is mandatory.

Install the .NET framework version 4.5. It is usually selected by default.

When creating a project, the "Class1.cs" file containing the "Class1" class is automatically added to the project. Rename both the file and the class to "MqlSqlDemo.cs" and "MqlSqlDemo". Functions to be exported from the DLL file can only be static — this is again required for export to unmanaged code.

_Strictly speaking, you can also export non-static functions. But to do this, it is necessary to refer to C++/CLI tools, which are not considered in this article._

Since all functions in our class must be static, it is logical to make the class itself static. In this case, if the "static" modifier is missing for some function, this will be immediately found out when compiling the project. We get the following class definition:

publicstaticclass MqlSqlDemo

{

// ...

}

It is now necessary to configure the project's dependencies ("References" section in "Solution Explorer"). Remove all redundant options, leaving only "System" and "System.Data".

Now add the "UnmanagedExports" package.

The package description is available on its author's site: [https://sites.google.com/site/robertgiesecke/Home/uploads/unmanagedexports](https://www.mql5.com/go?link=https://sites.google.com/site/robertgiesecke/Home/uploads/unmanagedexports "https://sites.google.com/site/robertgiesecke/Home/uploads/unmanagedexports").

The most convenient way to add it is by using the NuGet package manager. Instructions for adding can be found on the NuGet site: [https://www.nuget.org/packages/UnmanagedExports](https://www.mql5.com/go?link=https://www.nuget.org/packages/UnmanagedExports "https://www.nuget.org/packages/UnmanagedExports")

Only one of these instructions is needed in our case:

```
Install-Package UnmanagedExports -Version 1.2.7
```

In the Visual Studio menu, select the "Tools" sections, then "NuGet Package Manager", and "Package Manager Console". Command line will appear at the bottom. Insert the copied "Install-Package UnmanagedExports -Version 1.2.7" instruction there and press "Enter". The package manager will connect to the Internet for a while and download the package, then will add it to the project and print the following:

```
PM> Install-Package UnmanagedExports -Version 1.2.7
Installing 'UnmanagedExports 1.2.7'.
Successfully installed 'UnmanagedExports 1.2.7'.
Adding 'UnmanagedExports 1.2.7' to MqlSqlDemo.
Successfully added 'UnmanagedExports 1.2.7' to MqlSqlDemo.

PM>
```

This means that the package has been added successfully.

After that, we can proceed to directly writing the code in the class definition file MqlSqlDemo.cs.

Configure the used namespaces.

- Visual Studio adds a lot of extra ones. From the "using" section, remove everything except " **using System;**".

- Then add " **using System.Data;**" — this is where the classes for working with the databases will be taken from.

- Add " **using System.Data.SqlClient;**": here are the classes for working with the SQL Server database specifically.

- Add " **using System.Runtime.InteropServices;**": here are the attributes for interaction with the unmanaged code.
- Add " **using RGiesecke.DllExport;**": the attribute for marking the exported functions will be taken from here.

Here is the resulting set:

using System;

using System.Data;

using System.Data.SqlClient;

using System.Runtime.InteropServices;

using RGiesecke.DllExport;

Add the necessary variables. Variables in a static class can also be static only. We will need objects for working with the database — the connection object and the command object:

privatestatic SqlConnection conn = null;

privatestatic SqlCommand com = null;

We also need a string that will be used to send detailed messages about the function execution results:

privatestaticstring sMessage = string.Empty;

Declare two constants with values 0 and 1 — they will serve as the return values for most functions. If a function ran successfully, it will return 0, otherwise — 1. This will make the code more understandable.

publicconstint iResSuccess = 0;

publicconstint iResError = 1;

And now, to the functions.

There are limitations for functions that are to be exported for use in MQL5.

1. As previously mentioned, the functions should be static.

2. Template collection classes (the **System.Collections.Generic** namespace) are forbidden. The code including them will compile normally, but unexplainable errors may occur at runtime. These classes can be used in other functions that will not be imported, but it is better to do without them at all. You can use regular arrays. Our project is written only for informational purposes, so it will not contain such classes (as well as arrays).

This demo project will work using only simple data types — numbers or strings. In theory, it would also be possible to pass **Boolean** values, which are internally represented as integers too. But the values of these numbers can be interpreted different by different systems (MQL and .NET). This leads to errors. Therefore, we restrict ourselves to three types of data — **int**, **string** and **double**. If necessary, Boolean values should be passed as int.

In real projects, it is possible to pass complex data structures, but you can do without it to organize the work with SQL Server.

To work with the database, we first need to establish a connection. This is done by the **CreateConnection** function. The function will take one parameter — a string with the parameters to connect to the SQL Server database. It will return an integer, showing if the connection was successful. If the connection is successful, **iResSuccess** will be returned, i.e. 0. If it fails — **iResError**, i.e. 1. More detailed information will be put into the message string — **sMessage**.

Here is the result:

```
[DllExport("CreateConnection", CallingConvention = CallingConvention.StdCall)]
    public static int CreateConnection(
            [MarshalAs(UnmanagedType.LPWStr)] string sConnStr)
    {
        // Clear the message string:
        sMessage = string.Empty;
        // If a connection exists - close it and change the
        // connection string to a new one, if not -
        // re-create the connection and command objects:
        if (conn != null)
        {
                conn.Close();
                conn.ConnectionString = sConnStr;
        }
        else
        {
                conn = new SqlConnection(sConnStr);
                com = new SqlCommand();
                com.Connection = conn;
        }
        // Try to open the connection:
        try
        {
                conn.Open();
        }
        catch (Exception ex)
        {
                // The connection was not opened for some reason.
                // Write the error information to the message string:
                sMessage = ex.Message;
                // Release the resources and reset the objects:
                com.Dispose();
                conn.Dispose();
                conn = null;
                com = null;
                // Error:
                return iResError;
        }
        // Everything went well, the connection is open:
        return iResSuccess;
}
```

Each function to be exported is marked by the **DllExport** attribute before the function definition. It is located in the RGiesecke.DllExport namespace, imported from the RGiesecke.DllExport.Metadata assembly. The assembly is automatically added to the project when the NuGet manager installs the UnmanagedExports package. Two parameters should be passed to this attribute:

- function name under which it will be exported. This name will be used by external programs (including MetaTrader 5) to call it from the DLL. The name of the exported function can be the same as the function name in the code — CreateConnection;
- the second parameter indicates which function calling mechanism will be used. **CallingConvention.StdCall** is suitable for all our functions.

Pay attention to the attribute **\[MarshalAs(UnmanagedType.LPWStr)\]**. It stands before the connection string parameter ConnStringIn, taken by the function. This attribute shows how the string should be sent. At the time of writing this article, MetaTrader 5 and MetaTrader 4 work with Unicode strings — UnmanagedType.LPWStr.

At the time of the function call, the text describing the error in the previous connection attempt can remain in the message string, so the string is cleared at the beginning of the function. Also, the function can be called when the previous connection is not yet closed. Therefore, first check if the connection and command objects exist. If they do, the connection can be closed and the objects can be reused. If not, then it is necessary to create new objects.

The **Open** used for connecting does not return any result. Therefore, finding out if the connection was successful is only possible by catching exceptions. In case of an error, release the resources, zero the objects, write the information to the message string and return iResError. If everything is fine — return iResSuccess.

If the connection fails to open, to find out the cause of failure, the robot needs to read the message contained in the sMessage string. To do this, add the **GetLastMessage** function. It will return a string with the message:

```
[DllExport("GetLastMessage", CallingConvention = CallingConvention.StdCall)]
[return: MarshalAs(UnmanagedType.LPWStr)]
public static string GetLastMessage()
{
        return sMessage;
}
```

Just like the function for establishing a connection, this function is also marked with the DllExport export attribute. The **\[return: MarshalAs(UnmanagedType.LPWStr)\]** attribute indicates the way the returned result is to be passed. Since the result is a string, it should be passed to MetaTrader 5 in Unicode as well. Therefore, **UnmanagedType.LPWStr** is used here as well.

Once the connection is opened, you can start working with the database. Let us add the ability to execute queries to the database. This will be done by the **ExecuteSql** function:

```
[DllExport("ExecuteSql", CallingConvention = CallingConvention.StdCall)]
public static int ExecuteSql(
        [MarshalAs(UnmanagedType.LPWStr)] string sSql)
{
        // Clear the message string:
        sMessage = string.Empty;
        // First, check if the connection is established.
        if (conn == null)
        {
                // The connection is not open yet.
                // Report the error and return the error flag:
                sMessage = "Connection is null, call CreateConnection first.";
                return iResError;
        }
        // The connection is ready, try to execute the command.
        try
        {
                com.CommandText = sSql;
                com.ExecuteNonQuery();
        }
        catch (Exception ex)
        {
                // Error while executing the command.
                // Write the error information to the message string:
                sMessage = ex.Message;
                // Return the error flag:
                return iResError;
        }
        // Everything went well - return the flag of the successful execution:
        return iResSuccess;
}
```

The query text is passed to the function by a parameter. Before executing the query, check if the connection is open. As with the function for opening a connection, this function returns iResSuccess if it is successful and iResError in case of an error. To get more detailed information on what caused the error, it is necessary to use the GetLastMessage function. The ExecuteSql function can be used to execute any queries — write, delete or modify data. It is also possible to work with the database structure. Unfortunately, it does not allow reading data — the function does not return a result and does not store the read data anywhere. The query will be executed, but you will be unable to see what was read. Therefore, we add two more functions for reading data.

The first function is designed to read one integer from the database table.

```
[DllExport("ReadInt", CallingConvention = CallingConvention.StdCall)]
public static int ReadInt(
        [MarshalAs(UnmanagedType.LPWStr)] string sSql)
{
        // Clear the message string:
        sMessage = string.Empty;
        // First, check if the connection is established.
        if (conn == null)
        {
                // The connection is not open yet.
                // Report the error and return the error flag:
                sMessage = "Connection is null, call CreateConnection first.";
                return iResError;
        }
        // Variable to receive the returned result:
        int iResult = 0;
        // The connection is ready, try to execute the command.
        try
        {
                com.CommandText = sSql;
                iResult = (int)com.ExecuteScalar();
        }
        catch (Exception ex)
        {
                // Error while executing the command.
                // Write the error information to the message string:
                sMessage = ex.Message;
        }
        // Return the obtained result:
        return iResult;
}
```

Data reading is much more difficult to implement than simple execution of commands. This function is greatly simplified and utilizes the **ExecuteScalar** function of the **SqlCommand** class. It returns the value of the first column of the first row returned by the query. Therefore, the SQL query passed by the parameter should be formed in such a way that the returned set of data has rows, and the first column contains an integer. In addition, the function should somehow return the number read. Therefore, its result will no longer be a message of execution success. To understand if the query was successful and to read the data, it will be necessary to analyze the last in any case, by calling GetLastMessage. If the last message is empty, then there was no error and the data were read. If something is written there, it means that an error has occurred and the data could not be read.

The second function also reads one value from the database, but of another type — not an integer, but a string. Strings can be read just like numbers; the difference is only in the type of the returned result. Since the function returns a string, it should be marked with the **\[return: MarshalAs(UnmanagedType.LPWStr)\]** attribute. Here is the code for this function:

```
[DllExport("ReadString", CallingConvention = CallingConvention.StdCall)]
[return: MarshalAs(UnmanagedType.LPWStr)]
public static string ReadString(
        [MarshalAs(UnmanagedType.LPWStr)] string sSql)
{
        // Clear the message string:
        sMessage = string.Empty;
        // First, check if the connection is established.
        if (conn == null)
        {
                // The connection is not open yet.
                // Report the error and return the error flag:
                sMessage = "Connection is null, call CreateConnection first.";
                return string.Empty;
        }
        // Variable to receive the returned result:
        string sResult = string.Empty;
        // The connection is ready, try to execute the command.
        try
        {
                com.CommandText = sSql;
                sResult = com.ExecuteScalar().ToString();
        }
        catch (Exception ex)
        {
                // Error while executing the command.
                // Write the error information to the message string:
                sMessage = ex.Message;
        }
        // Return the obtained result:
        return sResult;
}
```

Such data reading capabilities are sufficient for a demo project. For a real expert, it may be unnecessary at all — it is more important for the experts to write data to the database for further analysis. If it is still necessary to read the data, these functions can be used — they are quite fit for work. However, sometimes you need to read a lot of rows from the table, containing several columns.

This can be done in two ways. You can return complex data structures from the function (this path is not suitable for MQL4). We can also declare a static variable of the **DataSet** class in our class. When reading, it will be necessary to load the data from the database to this DataSet, and then use other functions to read the data from it, one cell for one function call. This approach is implemented in the **HerdOfRobots** project mentioned below. It can be studied in detail in the project code. In order not to inflate the article, data reading from multiple rows will not be considered here.

After the work with the database is complete, the connection will need to be closed, releasing the resources used. This is done by the **CloseConnection** function:

```
[DllExport("CloseConnection", CallingConvention = CallingConvention.StdCall)]
public static void CloseConnection()
{
        // First, check if the connection is established.
        if (conn == null)
                // The connection is not open yet - meaning it does not need to be closed either:
                return;
        // The connection is open - is should be closed:
        com.Dispose();
        com = null;
        conn.Close();
        conn.Dispose();
        conn = null;
}
```

This simple function does not take any parameters and does not return a result.

All the necessary functions are ready. Compile the project.

Since the function is to be used not from other .NET applications, but from MetaTrader (which does not use .NET), the compilation will take place in two stages. At the first stage, everything is done in the same way as for all .NET projects. A normal build is created, which is later processed by the UnmanagedExports package. The package starts working after the build is compiled. First, the IL decompiler starts, which parses the resulting build into an IL code. The IL code is modified — references to the DllExport attributes are removed from it and instructions for exporting the functions marked by this attribute are added. After that, the file with the IL code is recompiled, overwriting the original DLL.

All these actions are performed automatically. But, as mentioned above, if Russian is selected for non-Unicode programs in the operating system settings, attempts to compile the file with the modified IL code are likely to make the UnmanagedExports give an error and, unable to do anything.

If no error messages are received during the compilation, then everything went well and the obtained DLL can be used in experts. In addition, when UnmanagedExports successfully processes a DLL, it adds two more files with extensions ".exp" and ".lib" (in this case, "MqlSqlDemo.exp" and "MqlSqlDemo.lib"). We have no use for them, but their presence can indicate that UnmanagedExports completed successfully.

It should be noted that the demo project has a very significant limitation: it allows running only one expert working with the database in one MetaTrader terminal. All experts use one instance of the loaded DLL. Since our class is made static, it will be the only one for all running experts. The variables will also be common. If you run several experts, they will all use the same connection and the same command object. If multiple experts attempt to address these objects simultaneously, problems may arise.

But such a project is sufficient to explain the operation principles and to test the connection to the database. Now we have a DLL file with the functions. We can proceed to writing an expert in MQL5.

### Creating an Expert Advisor in MQL5

Let us create a simple expert in MQL5. Its code can also be complied in the MQL4 editor, by changing the extension from "mq5" to "mq4". This expert is only for demonstrating the successful work with the database, so it will not perform any trading operations.

Run MetaEditor, press the "New" button. Select "Expert Advisor (template)" and press "Next". Specify the name "MqlSqlDemo". Also add one parameter — "ConnectionString" of type "string". This will be the connection string indicating how to connect to your database server. For example, you can set this initial value for the parameter:

_Server=localhost;Database=master;Integrated Security=True_

This connection string allows connecting to an unnamed ("Default Instance") database server installed on the same computer, where the MetaTrader terminal is running. There is no need to specify login and password — authorization by Windows account is used.

If you downloaded SQL Server Express and installed it on your computer without changing the parameters, then your SQL Server will be a "named instance". It will receive the name "SQLEXPRESS". It will have a different connection string:

_Server=localhost\\\SQLEXPRESS;Database=master;Integrated Security=True_

When adding a string parameter to the Expert Advisor template, there is a limitation on the string size. A longer connection string (for example, to a named server "SQLEXPRESS") may not fit. But this is not a problem — the parameter value can be left blank at this stage. It can later be changed to any value when editing the expert code. It is also possible to specify the required connection string when launching the expert.

Click "Next". No more functions need to be added, so leave all the checkboxes on the next screen unchecked. Press "Next" again and receive the generated initial code for the expert.

The purpose of the expert is only to demonstrate the connection to the database and work with it. To do this, it is sufficient to use only the initialization function — OnInit. Drafts for other functions — OnDeinit and OnTick — can be removed right away.

As a result, we obtain the following:

//+------------------------------------------------------------------+

//\|                                                   MqlSqlDemo.mq5 \|

//\|                        Copyright 2018, MetaQuotes Software Corp. \|

//\|                                             https://www.mql5.com \|

//+------------------------------------------------------------------+

#property copyright"Copyright 2018, MetaQuotes Software Corp."

#property link"https://www.mql5.com"

#property version"1.00"

#property strict

//\-\-\- input parameters

inputstring   ConnectionString = "Server=localhost\\\SQLEXPRESS;Database=master;Integrated Security=True";

//+------------------------------------------------------------------+

//\| Expert initialization function                                   \|

//+------------------------------------------------------------------+

intOnInit()

{

//---

//---

return(INIT\_SUCCEEDED);

}

Please note: when connecting to a named instance (in this case, "SQLEXPRESS") it is necessary to repeat the "\\" character twice: " **localhost\\\SQLEXPRESS**". This is required both when adding the parameter to the expert template and in the code. If the character is specified only once, the compiler treats it as if the escape sequence (special character) "\\S" is specified in the string, and reports that it was not recognized during compilation.

However, when attaching a compiled robot to a chart, its parameters will have only one "\\" character, despite that two of them are specified in the code. This happens because all Escape sequences in strings are converted into corresponding characters during compilation. The sequence "\\\" is converted into a single "\\" character, and users (who do not need to work with the code) see a normal string. Therefore, if you specify the connection string not in the code, but when starting the Expert Advisor, only a single "\\" character should be specified in the connection string.

_Server=localhost\\SQLEXPRESS;Database=master;Integrated Security=True_

Now let us add functionality to the draft expert. First, it is necessary to import the functions for working with the database from the created DLL. Add the import section before the OnInit function. The imported functions are described almost in the same way as they are declared in the C# code. It is only necessary to remove all the modifiers and attributes:

```
// Description of the imported functions.
#import "MqlSqlDemo.dll"

// Function for opening a connection:
int CreateConnection(string sConnStr);
// Function for reading the last message:
string GetLastMessage();
// Function for executing the SQL command:
int ExecuteSql(string sSql);
// Function for reading an integer:
int ReadInt(string sSql);
// Function for reading a string:
string ReadString(string sSql);
// Function for closing a connection:
void CloseConnection();

// End of import:
#import
```

For greater clarity of the code, declare constants for the results of the function execution. As in the DLL, these will be 0 on successful execution and 1 on error:

```
// Successful execution of the function:
#define iResSuccess  0
// Error while executing the function:
#define iResError 1
```

Now we can add calls to the functions for working with the database to the OnInit initialization function. Here is how it will look:

```
int OnInit()
  {
   // Try to open a connection:
   if (CreateConnection(ConnectionString) != iResSuccess)
   {
      // Failed to establish the connection.
      // Print the message and exit:
      Print("Error when opening connection. ", GetLastMessage());
      return(INIT_FAILED);
   }
   Print("Connected to database.");
   // The connection was established successfully.
   // Try to execute queries.
   // Create a table and write the data into it:
   if (ExecuteSql(
      "create table DemoTest(DemoInt int, DemoString nvarchar(10));")
      == iResSuccess)
      Print("Created table in database.");
   else
      Print("Failed to create table. ", GetLastMessage());
   if (ExecuteSql(
      "insert into DemoTest(DemoInt, DemoString) values(1, N'Test');")
      == iResSuccess)
      Print("Data written to table.");
   else
      Print("Failed to write data to table. ", GetLastMessage());
   // Proceed to reading the data. Read an integer from the database:
   int iTestInt = ReadInt("select top 1 DemoInt from DemoTest;");
   string sMessage = GetLastMessage();
   if (StringLen(sMessage) == 0)
      Print("Number read from database: ", iTestInt);
   else // Failed to read number.
      Print("Failed to read number from database. ", GetLastMessage());
   // Now read a string:
   string sTestString = ReadString("select top 1 DemoString from DemoTest;");
   sMessage = GetLastMessage();
   if (StringLen(sMessage) == 0)
      Print("String read from database: ", sTestString);
   else // Failed to read string.
      Print("Failed to read string from database. ", GetLastMessage());
   // The table is no longer needed - it can be deleted.
   if (ExecuteSql("drop table DemoTest;") != iResSuccess)
      Print("Failed to delete table. ", GetLastMessage());
   // Completed the work - close the connection:
   CloseConnection();
   // Complete initialization:
   return(INIT_SUCCEEDED);
  }
```

Compile the expert. That is it, the test expert is ready. You can run it. Before running the expert, it is necessary to add the DLL to the libraries folder of the MetaTrader profile you use. Start MetaTrader, in the "File" menu, select "Open Data Folder". Open the "MQL5" folder (for MetaTrader 4, the "MQL4" folder), then the "Libraries" folder. Place the created DLL file (MqlSqlDemo.dll) to this folder. The expert should already be compiled and ready for use by this time. Naturally, running Expert Advisors and importing functions from DLL should be allowed in the MetaTrader 5 settings, otherwise it will immediately fail with an error at startup.

Start the expert, changing the connection string values to your database server access parameters. If everything is done correctly, the expert will output the following to the log:

_2018.07.10 20:36:21.428    MqlSqlDemo (EURUSD,H1)    Connected to database._

_2018.07.10 20:36:22.187    MqlSqlDemo (EURUSD,H1)    Created table in database._

_2018.07.10 20:36:22.427    MqlSqlDemo (EURUSD,H1)    Data written to table._

_2018.07.10 20:36:22.569    MqlSqlDemo (EURUSD,H1)    Number read from database: 1_

_2018.07.10 20:36:22.586    MqlSqlDemo (EURUSD,H1)    String read from database: Test_

Connecting to the database, executing SQL commands, writing and reading data — everything is executed successfully.

### Conclusion

A complete solution for Visual Studio — the archive containing all the necessary files is attached to the article under the name " **MqlSqlDemo.zip**". The "UnmanagedExports" package is already installed. The test expert **MqlSqlDemo.mq5** and its variant for MQL4 are located in the " **MQL**" subfolder.

The approach described in this article is fully operational. Based on the above principles, applications have been created that allow working with thousands and even tens of thousands of experts launched simultaneously. Everything has been tested repeatedly and works to date.

Both the DLL file and the expert created within this article are intended for educational and evaluation purposes only. Of course, you can use the DLL in real projects. However, it is quite likely to become unfitting soon due to its significant limitations. If you want to add the ability to work with databases to your experts, you will most likely need more features. In that case, it will be necessary to add the code yourself, using this article as an example. If you have any difficulties, ask for help in the comments to the article, contact me on Skype ( **cansee378**) or through the contact page on my site: [http://life-warrior.org/contact](https://www.mql5.com/go?link=http://life-warrior.org/contact "http://life-warrior.org/contact").

If you do not have the time or the desire to go through the C# code, you can download the finished project. The free open-source program named HerdOfRobots is implemented using the same principles that are described in the article. The installation pack contains the complete files with the imported functions both for MetaTrader 5 and MetaTrader 4, along with the program. These libraries have much more powerful features. For example, they allow running up to 63 experts in one terminal (which can be connected to different databases), reading data from tables row by row, writing date/time values to the database.

The HerdOfRobots program provides convenient features for controlling the experts connecting to the database, as well as analyzing the data written by them. The pack contains a manual which describes all aspects of the work in great detail. Archive with the program installer — SetupHerdOfRobots.zip — is also attached to the article. If you want to see the code of the program used for connecting to the database of the MqlToSql64 project (MqlToSql for MT4), in order to use the mentioned advanced features in your projects later, the code can be freely downloaded from open repositories:

[https://bitbucket.org/CanSeeThePain/herdofrobots](https://www.mql5.com/go?link=https://bitbucket.org/CanSeeThePain/herdofrobots "https://bitbucket.org/CanSeeThePain/herdofrobots")

[https://bitbucket.org/CanSeeThePain/mqltosql64](https://www.mql5.com/go?link=https://bitbucket.org/CanSeeThePain/mqltosql64 "https://bitbucket.org/CanSeeThePain/mqltosql64")

[https://bitbucket.org/CanSeeThePain/mqltosql](https://www.mql5.com/go?link=https://bitbucket.org/CanSeeThePain/mqltosql "https://bitbucket.org/CanSeeThePain/mqltosql")

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/2895](https://www.mql5.com/ru/articles/2895)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/2895.zip "Download all attachments in the single ZIP archive")

[MqlSqlDemo.zip](https://www.mql5.com/en/articles/download/2895/mqlsqldemo.zip "Download MqlSqlDemo.zip")(565.98 KB)

[SetupHerdOfRobots.zip](https://www.mql5.com/en/articles/download/2895/setupherdofrobots.zip "Download SetupHerdOfRobots.zip")(6741.3 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/278353)**
(14)


![ratracesurvivor](https://c.mql5.com/avatar/avatar_na2.png)

**[ratracesurvivor](https://www.mql5.com/en/users/ratracesurvivor)**
\|
11 Feb 2019 at 09:37

Love the article - thanks very much for the in-depth discussion.

Hoping someone can help me with a snag during Build time.

```
C:\Users\user\source\repos\mql\MqlSqlDemo\packages\UnmanagedExports.1.2.7\tools\RGiesecke.DllExport.targets(58,3): error : Microsoft.Build.Utilities.ToolLocationHelper could not find ildasm.exe
```

Essentially, the UnmanagedExports utility is unable to find the disassembler executable required to perform it's work.

I know for a fact that ildasm.exe exists, and it's various locations... but not sure how to get DllExport to recognize the proper path.


![Dmitry Melnichenko](https://c.mql5.com/avatar/2017/6/5941A584-597E.gif)

**[Dmitry Melnichenko](https://www.mql5.com/en/users/melnik)**
\|
11 Feb 2019 at 19:18

This is the error:

2019.02.11 21:03:17.627 Cannot find 'CreateConnection' in 'MqlSqlDemo.dll'

![Spec Trance](https://c.mql5.com/avatar/2019/4/5CB24373-1A4C.png)

**[Spec Trance](https://www.mql5.com/en/users/spectrance)**
\|
24 Apr 2019 at 23:14

Superb Article !!! Thanks for explaining the nitty-gritty details as well as for providing the downloadable files. Will try to implement this myself. Thanks a lot !!! Well wishes & Kudos!


![leonerd](https://c.mql5.com/avatar/2017/5/5919A02E-9AEB.jpg)

**[leonerd](https://www.mql5.com/en/users/leonerd)**
\|
17 Oct 2019 at 18:45

Hi. Is there an example with [parsing](https://www.mql5.com/en/articles/5638 "Article: Parsing MQL Using MQL ") and exporting logs to DB?


![gmartin86](https://c.mql5.com/avatar/avatar_na2.png)

**[gmartin86](https://www.mql5.com/en/users/gmartin86)**
\|
2 Dec 2021 at 17:53

Hello I have tried the dll. And works!

I would like to test the complete library. That you could upload it again? (https://bitbucket.org links are broken?)

Thank you!

![Expert Advisor featuring GUI: Adding functionality (part II)](https://c.mql5.com/2/32/avatar_expert_Graph_panel71p__1.png)[Expert Advisor featuring GUI: Adding functionality (part II)](https://www.mql5.com/en/articles/4727)

This is the second part of the article showing the development of a multi-symbol signal Expert Advisor for manual trading. We have already created the graphical interface. It is now time to connect it with the program's functionality.

![Visualizing optimization results using a selected criterion](https://c.mql5.com/2/32/VisualizeBest100.png)[Visualizing optimization results using a selected criterion](https://www.mql5.com/en/articles/4636)

In the article, we continue to develop the MQL application for working with optimization results. This time, we will show how to form the table of the best results after optimizing the parameters by specifying another criterion via the graphical interface.

![Trading account monitoring is an indispensable trader's tool](https://c.mql5.com/2/34/monitoring_logo.png)[Trading account monitoring is an indispensable trader's tool](https://www.mql5.com/en/articles/5178)

Trading account monitoring provides a detailed report on all completed deals. All trading statistics are collected automatically and provided to you as easy-to-understand diagrams and graphs.

![Testing currency pair patterns: Practical application and real trading perspectives. Part IV](https://c.mql5.com/2/31/LOGO.png)[Testing currency pair patterns: Practical application and real trading perspectives. Part IV](https://www.mql5.com/en/articles/4543)

This article concludes the series devoted to trading currency pair baskets. Here we test the remaining pattern and discuss applying the entire method in real trading. Market entries and exits, searching for patterns and analyzing them, complex use of combined indicators are considered.

[![](https://www.mql5.com/ff/si/5k7a2kbftss6k97n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F1171%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dbest.vps%26utm_content%3Drent.vps%26utm_campaign%3D0622.MQL5.com.Internal&a=nwegcasiojnqcoyrdlgofmjtfardztwf&s=d64d6f3c87f2458cba81f6d7b6694dd9e89dd354d4abc1d0584e405285806c9f&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=azibpxzallnkchgqdtwjkttbcxtndddb&ssn=1769192148266793505&ssn_dr=0&ssn_sr=0&fv_date=1769192148&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F2895&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Integrating%20MQL-based%20Expert%20Advisors%20and%20databases%20(SQL%20Server%2C%20.NET%20and%20C%23)%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176919214849846681&fz_uniq=5071701755189734611&sv=2552)

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