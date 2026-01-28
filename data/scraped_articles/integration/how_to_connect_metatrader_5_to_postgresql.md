---
title: How to connect MetaTrader 5 to PostgreSQL
url: https://www.mql5.com/en/articles/12308
categories: Integration
relevance_score: 3
scraped_at: 2026-01-23T21:11:22.665006
---

[![](https://www.mql5.com/ff/si/h2ryn394uwcpxwmxc2.jpg)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Feconomic-calendar%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dopen.calendar%26utm_content%3Deconomic.calendar%26utm_campaign%3Den.0009.desktop.default&a=qdeulxgvibgwytgewnvfatbocjnnninc&s=5c0c60f00ff5f5bedb0fdf65d9d79eb820442eb43ffac2b85aa003224f9dba14&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=xlrbssidgbgbpyhwrbrmamoefclukzyd&ssn=1769191881098742759&ssn_dr=0&ssn_sr=0&fv_date=1769191881&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F12308&back_ref=https%3A%2F%2Fwww.google.com%2F&title=How%20to%20connect%20MetaTrader%205%20to%20PostgreSQL%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176919188125531662&fz_uniq=5071641419489160137&sv=2552)

MetaTrader 5 / Integration


### Introduction

The landscape of software development changed a lot in the last ten years. We saw the popularization of cloud computing and acronyms like IASS, PASS, and SAAS are now key tools that must be under consideration in any software project. Things became easier for both the end user and the software developer.

The MetaQuotes team was aware of these changes and since 2014 we have a native [WebRequest](https://www.mql5.com/en/articles/1392 "WebRequest article") for MetaTrader 5.

One of the areas that changed the most was database management. Solutions that used to be complex or even "weird" from a practical perspective became not only doable but the preferred solution for many use cases. This was the case for database access via REST API.

A proposal for database access via REST API would look like over-engineering a couple of years ago. Today, a quick search for "managed database with access via rest API" will return dozen of providers, ranging from a few dollars/month in basic plans to customized enterprise solutions. Many of these providers offer generous free tiers for prototyping, testing or even deploying small production workloads.

This article analyses five standard alternatives to connecting a Postgres database to MetaTrader 5, their requirements, Pros, and Cons. Also, we will set up a development environment, install a Postgres database as a remote one, connect to it, and insert and retrieve data to be consumed by an MQL5 script or EA.

This development environment setup and respective procedures can be easily replicated with any RDBMS since the REST API stand as a layer of abstraction between the DB system and the client code.

### MetaTrader 5 and Databases

MetaTrader 5 already has the functions you may need to work with a [database](https://www.mql5.com/en/docs/database "Database functions on reference documentation") and the functions you may need to connect to a database via [network](https://www.mql5.com/en/docs/network "Network functions on reference documentation").

Since 2020 the platform provides native integration with [SQLite](https://www.mql5.com/en/articles/7463). You can use those database functions mentioned above to interact with it from code. Besides that you can interact with your databases via a dedicated GUI in MetaEditor, making it easy to create tables, alter tables, and perform CRUD operations without the need for additional software.

That was a great improvement in the end-user experience and a key addition to the MQL5 developer arsenal.

Among dozens of RDBMS available, many of them with open-source licenses, SQLite seems to have been a smart choice by MetaTrader 5 developers. Despite being a [full-featured](https://www.mql5.com/go?link=https://sqlite.org/fullsql.html "https://sqlite.org/fullsql.html") SQL database, with multi-column indexes, triggers, views, acid transactions, full-text search, aggregate functions, and more, it is lightweight, file-based, scalable, and requires zero maintenance. According to its website, "it seems likely that there are over one trillion (1e12) SQLite databases in active use".

Regardless of its impressive features, SQLite is limited by design to a single user and is not aimed at concurrent access in web deployments. The large number of forum posts and articles on the MQL5 website about how to connect MetaTrader 5 to MySQL reveals that there is a demand for a more robust solution for other use cases.

This article is focused on setting up a development environment for these use cases using Postgres.

**Why Postgres**

First of all, I chose Postgres because the other popular open-source alternative, MySQL, was already extensively covered here.

Second, Postgres is a mature open-source project, multiplatform, is very well maintained, with consistent documentation. It is popular and you can find a plethora of sample code, guides, and tutorials around the web. Likewise, there are a lot of cloud providers available for all needs and budgets.

Postgres is enterprise-grade, and at the same time can be easily managed by a single user working alone on a home machine.

And of course, I chose Postgres because I trust it. For more than a decade, I am a happy user of Postgres on different projects.

And last but not least, I am currently implementing the solution I'm sharing here for my personal trading environment. So it is a kind of "skin in the game". I'm eating my dog food.

### Four Ways To Interact With Postgres From MQL5

As far as I can see, there are four main approaches to calling Postgres from MQL5:

1. a dedicated MQL5 library/driver
2. a .dll from the C++ client interface
3. via .NET Npgsql driver
4. a REST API

Let's take a look at the requirements, pros, and cons of each one. I'm pretty sure that the large community of seasoned developers from the MQL5 community will be offering easy solutions for the cons, as well as pointing out the disadvantages that I was not able to see in the pros. This feedback from the MQL5 community is expected because both the less experienced developer and the non-developer trader who arrives here will benefit from this related discussion.

**1\. a dedicated MQL5 library/driver**

This library does not exist yet. It needs to be developed and would require many hours of hard work from a senior MQL5 developer. That would not be cheap. We need to take into account the maintenance costs too. Postgres is mature, but it isn't in any sense static. It is an active project with regular releases and some of these releases, if not many, will require updates on client code.

For example: right now, at the time of writing, the last Postgres version (15) requires that regular users of a database must be granted some privileges in the "public schema". This requirement did not exist in previous versions. Probably, maintenance is required in several codebases out there.

The advantage of commissioning the development of a dedicated MQL5 driver for Postgres is that, if shared, it could be useful for a lot of MQL5 users. The disadvantage is pretty obvious: the cost in time/money.

Where to start if you choose this way:

A [generic search](https://www.mql5.com/en/search#!keyword=mysql&module=mql5_module_articles "Search for MySQL on MQL5 website") for MySQL articles on this site will return some useful references.

The open-source C++ client library [libpqxx](https://www.mql5.com/go?link=https://pqxx.org/libpqxx/ "https://pqxx.org/libpqxx/")

The official C client library for Postgres [libpq](https://www.mql5.com/go?link=https://www.postgresql.org/docs/current/libpq.html "https://www.postgresql.org/docs/current/libpq.html")

**2\. a .dll from the C++ client interface**

This is an externally maintained official C++ library, libpqxx, that is built on top of the internally maintained official C library, libpq, that is shipped with the Postgres distribution.

Personally, I've never used it, and all that I can say is that it is there for a long time and seems to be well-maintained. The disadvantage of this method is that the MQL5 Market does not allow DLL's. If this is not an issue for your project and you are at home working with .dll's from MetaTrader, it might be your solution.

Where to start if you choose this way:

The open-source C++ client library[libpqxx](https://www.mql5.com/go?link=https://pqxx.org/libpqxx/ "https://pqxx.org/libpqxx/")

[https://pqxx.org/libpqxx/](https://www.mql5.com/go?link=https://pqxx.org/libpqxx/ "https://pqxx.org/libpqxx/")

**3\. via .NET Npgsql driver**

Since 2018, MetaTrader 5 added native support for .NET libraries with 'smart' functions import. With the release of platform build 1930 .NET libraries can be used without writing special wrappers —  MetaEditor does it on its own".

All that you need to use the .NET Npgsql driver is to import the .dll itself. There are some limitations that you can check on the official release notes (https://www.mql5.com/en/forum/285632).

Where to start if you choose this way:

The open-source Postgres [driver](https://www.mql5.com/go?link=https://www.npgsql.org/doc/index.html "https://www.npgsql.org/doc/index.html") for .NET

**4\. a REST API**

If you choose the path "without .dll", this should be the faster and cheaper method. The API can be written in any language and you can have a working prototype in a day or even some hours.

Besides that, some cloud providers offer built-in REST API for Postgres for free. All that you need to get started is a good development environment for your MQL5 code.

By using this method your MQL5 code can consume your Postgres responses as JSON.

Where to start if you choose this way:

Here! Just keep reading, follow the steps below, download the sample code, and start storing and querying your deals and trades in a Postgres database.

### Setting Up The Development Environment

Choose whichever method you choose, you will need a development environment in a Windows machine with a running Postgres server. As the saying goes, there is more than one way to do it. I remember these three paths, from the most complex and time-consuming to the simplest:

1. compilation from source code
2. docker container
3. third-party msi installer

All of them are good ways of having Postgres on Windows but believe me, a compilation from source code on Windows should be your last option, except if you are willing to learn theory and practice of intermittent resilience in software development.

The docker container is a very good option, a robust and flexible installation in which your database server will be running in a "remote" machine, not "localhost" (see below). After all, it is easy. You just need Docker installed, and two to three command lines and you are ready to go.

Taking apart the relative inconvenience of "third party" software, the third-party msi installer is a good alternative to avoid the adventurous compilation from source or the docker installation and container management.

But I would not recommend a development environment for a database server, or any kind of server for this matter, as a "localhost", if it is possible to develop against a server located in a remote machine. That is because it is always good practice to develop, test, and debug a server in a remote environment, and not in “localhost”, in order to troubleshoot connection settings and authentication issues as soon as possible.

Enters WSL.

**What is WSL**

WSL stands for Windows Subsystem For Linux.

In case you missed it, since 2016 you can run a Linux distribution on a Windows machine as a subsystem. No worries! No hacks here. WSL is developed by Microsoft and is built-in on Windows. You just need to enable it, as we will see below.

**Why WSL**

Why not simply install Postgres on another Windows machine, eventually a virtual machine?

Because Postgres is a Unix native system, created and developed in \*nix systems. By installing it on Linux you have easy installation, easy updates, and easy maintenance. All the official documentation is targeted to a Unix system. And most of the sample code, snippets and general help that you can find on the web reflect this fact.

Thus you will have an easy time developing in a Linux system. And WSL was developed by Microsoft for this exact purpose.

**Install WSL**

Prerequisites from Microsoft documentation:

“You must be running Windows 10 version 2004 and higher (Build 19041 and higher) or Windows 11 to use the commands below. If you are on earlier versions please see the manual install page.”

If your system meets this prerequisite, just open a Power Shell as administrator and enter the command below to install/enable WSL:

wsl –install

![wsl install command on Power Shell](https://c.mql5.com/2/52/wsl_install_1_v37.PNG)

This command will install Ubuntu on WSL, as it is the default distro.

Restart Windows.

This should be a straightforward process. If it is not, you will find a section with the most common installation issues in the MS official documentation linked above.

After rebooting, you should see something like this. Go on, create a new UNIX username and password.

![wsl install after first reboot](https://c.mql5.com/2/52/wsl_install_4_after_reboot_65g.PNG)

Now that you have WSL/Ubuntu installed and running, let's install Postgres on it.

**On WSL, install Postgres**

Enter the command below.

sudo apt install postgresql postgresql-contrib

This command will install the last stable version of PostgreSQL package available in Ubuntu repositories. It includes the server, the pgsql client, convenient binaries, and some utilities. All that you need to start.

If you want to install the latest stable Postgres version - usually different than the last stable version in Ubuntu repositories - you may include the official Postgres repository in the sources list of your package manager. You can find [detailed instructions](https://www.mql5.com/go?link=https://www.postgresql.org/download/linux/ubuntu/ "https://www.postgresql.org/download/linux/ubuntu/") on the official Postgres documentation.

If everything was ok, entering the command

psql --version

should return the installed version of your Postgres database.

**Start the server**

Type this command to start the server.

sudo service postgresql start

By default,  a new Postgres installation only accepts connections from "localhost". Let's change that.

Find the Postgres config file.

sudo -u postgres psql -c "SHOW config\_file"

![postgres install show config file](https://c.mql5.com/2/52/pg_install_9_z3k.PNG)

Edit the config file to accept connections outside localhost. Change the listen\_addresses line.

![postgres configuration listen_addresses](https://c.mql5.com/2/52/pg_conf_listen_addresses_33x.PNG)

Find the pg\_hba configuration file.

sudo -u postgres psql -c "SHOW hba\_file"

![postgres configuration show hba file](https://c.mql5.com/2/52/pg_hba_show_hba_file_y3k.PNG)

Edit pg\_hba.conf file to allow authentication by password on both IPv4 and IPv6.

![postgres configuration pg_hba auth by password](https://c.mql5.com/2/52/pg_hba_scram_q2n.PNG)

Now access the psql utility as the default Postgres user created by install. It is named 'postgres'.

sudo -u postgres psql

Create a database regular user with CREATEDB privilege. Untill now there is only user 'postgres' created by install.

Grant all privileges on schema public to mt5\_user. This is not necessary if your Postgres version is below 15.

CREATE USER mt5\_user PASSWORD '123' CREATEDB;

GRANT ALL ON SCHEMA public TO mt5\_user;

![postgres create user grant all privileges](https://c.mql5.com/2/52/pg_create_user_grant_all_t3g.PNG)

Create a database my\_remote\_db and grant all privileges on it to mt5\_user.

GRANT ALL PRIVILEGES ON DATABASE my\_remote\_db TO mt5\_user;

![psql create db grant all privileges](https://c.mql5.com/2/52/create_db_grant_all_r26.PNG)

**Connect to Postgres**

By now, you should have a database server running in a remote machine, with a different IP from your Windows localhost and ready to accept connections via the network. We can connect via socket or HTTP. Since we will be interacting with a REST API, in this example we will use the latter.

Let's see if we can connect to my\_remote\_db as mt5\_user with password 123 at WSL host.

Enter this command to get your WSL hostname (IP).

hostname -I

![ubuntu terminal command hostname](https://c.mql5.com/2/52/hostname_I_22r.PNG)

Check the Postgres server status. Start it if down. You can use this command to start, restart or stop the server.

sudo service postgresql {status, start, stop}

On MetaTrader 5 terminal, go to Tools > Options > Expert Advisors tab and include the WSL host IP on the list of allowed ones.

![MT5 terminal tools options menu](https://c.mql5.com/2/52/mt5_tools_options_url.PNG)

The MetaTrader 5 terminal accepts HTTP and HTTPS connections only on ports 80 and 443, respectively. **Only ports 80 and 443.** You should take this security feature into account if you are developing your API. Usually, before moving to a real server in production, the development server will be listening on an unprivileged port, like 3000 or 5000. Thus, to be able to send requests to the IP you declared in your terminal settings above, you need to redirect the traffic arriving at the development server port to port 80 for HTTP requests and/or 443 for HTTPS requests.

To keep things simple, you will find instructions on the README of the attached Python app about how to perform this redirection on WSL.

**Starting the Demo App**

Since this article is about MQL5, I will not discuss the details of the API implementation. Instead, I did a demo app that you can download and install as a Python package to test your MQL5 code interactions with the API.

To start the demo app you just need Python installed on WSL. It should already be there.

It is highly recommended that you create a Python virtual environment ('venv') to install the app. This will ensure that you system Python installation will not be messed. After playing with the app, you can simply delete the virtual environment.

You can create the virtual environment with this command.

python3 -m venv 'venv'

Thus, after you have installed the demo app, to start developing your MQL5 code, you will:

1. start WSL
2. start Postgres server
3. start the demo app
4. get the hostname IP from demo app output
5. add hostname IP to allowed addresses in the terminal

Both, WSL and the Postgres server can be configured to start at Windows startup.

**Inserting data from MQL5**

Let's try inserting some data. First, our account info. On your MT5 terminal, create a new script and add the following code.

```
//+------------------------------------------------------------------+
//|                                                post_acc_info.mq5 |
//|                                  Copyright 2023, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2023, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"
#include <JAson.mqh> //--- include the JSON library
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
//--- gathering the data - Account Info
   CJAVal data;
   CJAVal acc_info;
//--- doubles
   data["balance"] =          AccountInfoDouble(ACCOUNT_BALANCE);
   data["credit"] =           AccountInfoDouble(ACCOUNT_CREDIT);
   data["profit"] =           AccountInfoDouble(ACCOUNT_PROFIT);
   data["equity"] =           AccountInfoDouble(ACCOUNT_EQUITY);
   data["margin"] =           AccountInfoDouble(ACCOUNT_MARGIN);
   data["margin_free"] =      AccountInfoDouble(ACCOUNT_MARGIN_FREE);
   data["margin_level"] =     AccountInfoDouble(ACCOUNT_MARGIN_LEVEL);
   data["margin_so_call"] =   AccountInfoDouble(ACCOUNT_MARGIN_SO_CALL);
   data["margin_so_so"] =     AccountInfoDouble(ACCOUNT_MARGIN_SO_SO);
//--- integers
   data["login"] =            AccountInfoInteger(ACCOUNT_LOGIN);
   data["leverage"] =         AccountInfoInteger(ACCOUNT_LEVERAGE);
   data["trade_allowed"] =    AccountInfoInteger(ACCOUNT_TRADE_ALLOWED);
   data["ea_allowed"] =       AccountInfoInteger(ACCOUNT_TRADE_EXPERT);
   data["trade_mode"] =       AccountInfoInteger(ACCOUNT_TRADE_MODE);
   data["margin_so_mode"] =   AccountInfoInteger(ACCOUNT_MARGIN_SO_MODE);
//-- strings
   data["company"] =          AccountInfoString(ACCOUNT_COMPANY);
   data["currency"] =         AccountInfoString(ACCOUNT_CURRENCY);
   data["name"] =             AccountInfoString(ACCOUNT_NAME);
   data["server"] =           AccountInfoString(ACCOUNT_SERVER);

//--- fill in the acc_info array with Account Info data
   acc_info["acc_info"].Add(data);

//--- WebRequest arguments
   string method = "POST";
   string url = "http://172.22.18.235/accs";
   string headers = "Content-Type: application/json";
   int timeout = 500;
   char post[], result[];
   string result_headers;

//--- prepare JSON data to send
   string json = acc_info.Serialize();
   ArrayResize(post, json.Length(), 0);
   StringToCharArray(json, post, 0, StringLen(json), CP_UTF8);
   ResetLastError();

//--- send the request
   int res = WebRequest(method, url, headers, timeout, post, result, result_headers);
   if(res == -1)
     {
      Print("Error in WebRequest  =", GetLastError());
      MessageBox("Add " + url + " to allowed URLs on MT5 terminal", "Unknown URL", MB_ICONINFORMATION);     }
   else
     {
      Print("Starting post...");

      if(res == 201)// HTTP result code 201 (created)
        {
         Print("posted accs");
        }
     }
  }
```

As you can see from the beginning of the file, we are using a helper library to serialize/deserialize our JSON data. It was developed by a member of the MQL5 community and you can find the library on [his repository](https://www.mql5.com/go?link=https://github.com/vivazzi/JAson "https://github.com/vivazzi/JAson") at GitHub.

Now, let's insert our deals from MetaTrader 5 history. Create a new script and add the following code.

```
//+------------------------------------------------------------------+
//|                                                   post_deals.mq5 |
//|                                  Copyright 2023, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2023, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"
#include <JAson.mqh> //--- include the JSON library
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
//--- gathering the data - Deals
   CJAVal data;
   CJAVal deals;
//--- request trade history
   HistorySelect(0, TimeCurrent());
   int deals_total = HistoryDealsTotal();
//--- iterate over all deals to get data
//--- of each deal from its ticket number
   for(int i = 0; i < deals_total; i++)
     {
      //-- integers
      ulong deal_ticket =   HistoryDealGetTicket(i);
      data["ticket"] =     (int) deal_ticket;
      data["order"] =      (int) HistoryDealGetInteger(deal_ticket, DEAL_ORDER);
      data["position"] =   (int) HistoryDealGetInteger(deal_ticket, DEAL_POSITION_ID);
      data["time"] =       (int) HistoryDealGetInteger(deal_ticket, DEAL_TIME);
      data["type"] =       (int) HistoryDealGetInteger(deal_ticket, DEAL_TYPE);
      data["entry"] =      (int) HistoryDealGetInteger(deal_ticket, DEAL_ENTRY);
      data["magic"] =      (int) HistoryDealGetInteger(deal_ticket, DEAL_MAGIC);
      data["reason"] =     (int) HistoryDealGetInteger(deal_ticket, DEAL_REASON);
      //--- strings
      data["symbol"] =     (string) HistoryDealGetString(deal_ticket, DEAL_SYMBOL);
      //--- doubles
      data["volume"] =     (double) HistoryDealGetDouble(deal_ticket, DEAL_VOLUME);
      data["price"] =      (double) HistoryDealGetDouble(deal_ticket, DEAL_PRICE);
      data["profit"] =     (double) HistoryDealGetDouble(deal_ticket, DEAL_PROFIT);
      data["swap"] =       (double) HistoryDealGetDouble(deal_ticket, DEAL_SWAP);
      data["comission"] =  (double) HistoryDealGetDouble(deal_ticket, DEAL_COMMISSION);
 //--- fill in the deals array with each deal data
      deals["deals"].Add(data);
     }
 //--- WebRequest arguments
   string method = "POST";
   string url = "http://172.22.18.235/deals";
   string headers = "Content-Type: application/json";
   int timeout = 500;
   char post[], result[];
   string result_headers;

 //--- prepare JSON data to send
   string json = deals.Serialize();
   ArrayResize(post, json.Length(), 0);
   StringToCharArray(json, post, 0, StringLen(json), CP_UTF8);
   ResetLastError();
//--- send the request
   int res = WebRequest(method, url, headers, timeout, post, result, result_headers);

   if(res == -1)
     {
      Print("Error in WebRequest  =", GetLastError());
      MessageBox("Add " + url + " to allowed URLs on MT5 terminal", "Unknown URL", MB_ICONINFORMATION);     }
   else
     {
      Print("Starting post...");

      if(res == 201)// HTTP result code 201 (created)
        {
         Print("posted deals");
        }
     }
  }
```

**Querying data from MQL5**

Now let's query our recently inserted data. On your MetaTrader 5 terminal, create a new script and add the following code.

```
//+------------------------------------------------------------------+
//|                                                 get_endpoint.mq5 |
//|                                  Copyright 2023, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2023, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"
#include <JAson.mqh> //--- include the JSON library
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
//--- choose the testing endpoint
   string endpoint = "accs"; // or "deals"
//--- WebRequest arguments
   string method = "GET";
   string url = "http://172.22.18.235/" + endpoint;
   string cookie = NULL, headers;
   int timeout = 500;
   char post[], result[];
   ResetLastError();
//--- send the request
   int res = WebRequest(method, url, cookie, NULL, timeout, post, 0, result, headers);
   if(res == -1)
     {
      Print("Error in WebRequest  =", GetLastError());
      MessageBox("Add " + url + " to allowed URLs on MT5 terminal", "Unknown URL", MB_ICONINFORMATION);
     }
   else
     {
      Print("Starting get...");
      if(res == 200)// HTTP result code 200 (OK)
        {
         PrintFormat("Server headers: %s", headers);
         ResetLastError();
         // save the returned JSON in a file
         string terminal_data_path = TerminalInfoString(TERMINAL_DATA_PATH);
         string subfolder = "TutoPostgres";
         string filename = endpoint + "_fromserver.json";
         int filehandle = FileOpen(subfolder + "\\" + filename, FILE_WRITE | FILE_BIN);
         if(filehandle != INVALID_HANDLE)
           {
            FileWriteArray(filehandle, result, 0, ArraySize(result));
            FileClose(filehandle);
            Print(filename + " created at " + terminal_data_path + "\\" + subfolder);
           }
         else
            Print("File open failed with error ", GetLastError());
        }
      else
         PrintFormat("Request to '%s' failed with error code %d", url, res);
     }
  }
```

By changing the endpoint from "accs" to "deals" you can query your just-inserted deals. Check your <MT5 terminal path>\\Files\\TutoPostgres. If everything went fine, there should be at least two files there: accs\_fromserver.json and deals\_fromserver.json.

**Consuming JSON data in Expert Advisors**

To consume the server returned JSON data you need to deserialize it. The helper library mentioned above can do that.

If you looked at the JSON files saved after querying the database with the example code above, you may have seen a JSON string like this:

```
[\
  {\
    "a_balance": "10005.93",\
    "a_company": "MetaQuotes Software Corp.",\
    "a_credit": "0.0",\
    "a_currency": "USD",\
    "a_ea_allowed": true,\
    "a_equity": "10005.93",\
    "a_id": 3,\
    "a_leverage": 100,\
    "a_login": 66744794,\
    "a_margin": "0.0",\
    "a_margin_free": "10005.93",\
    "a_margin_level": "0.0",\
    "a_margin_so_call": "50.0",\
    "a_margin_so_mode": "0",\
    "a_margin_so_so": "30.0",\
    "a_name": "MetaTrader 5 Desktop Demo",\
    "a_profit": "0.0",\
    "a_server": "MetaQuotes-Demo",\
    "a_trade_allowed": true,\
    "a_trade_mode": "0"\
  },\
  {\
(...)\
```\
\
We will be using this keys to access the deserialized data. The '0' (zero) in the array index is accessing the first account returned. If you have more than one account, this endpoint ("accs") will return all accounts and you can access each of them iterating over the array by this index.\
\
```\
//+------------------------------------------------------------------+\
//|                                                 consume_json.mq5 |\
//|                                  Copyright 2023, MetaQuotes Ltd. |\
//|                                             https://www.mql5.com |\
//+------------------------------------------------------------------+\
#property copyright "Copyright 2023, MetaQuotes Ltd."\
#property link      "https://www.mql5.com"\
#property version   "1.00"\
#include <JAson.mqh> //--- include the JSON library\
//+------------------------------------------------------------------+\
//| Script program start function                                    |\
//+------------------------------------------------------------------+\
void OnStart()\
  {\
//--- choose the testing endpoint\
   string endpoint = "accs"; // or "deals"\
//--- WebRequest arguments\
   string method = "GET";\
   string url = "http://172.22.18.235/" + endpoint;\
   string cookie = NULL, headers;\
   int timeout = 500;\
   char post[], result[];\
   ResetLastError();\
//--- send the request\
   int res = WebRequest(method, url, cookie, NULL, timeout, post, 0, result, headers);\
   if(res == -1)\
     {\
      Print("Error in WebRequest  =", GetLastError());\
      MessageBox("Add " + url + " to allowed URLs on MT5 terminal", "Unknown URL", MB_ICONINFORMATION);\
     }\
   else\
     {\
      Print("Starting get...");\
      if(res == 200)// HTTP result code 200 (OK)\
        {\
         CJAVal data;\
         data.Deserialize(result);\
         //--- doubles\
         double a_balance =         data[0]["a_balance"].ToDbl();\
         double a_credit =          data[0]["a_credit"].ToDbl();\
         double a_profit =          data[0]["a_profit"].ToDbl();\
         double a_equity =          data[0]["a_equity"].ToDbl();\
         double a_margin =          data[0]["a_margin"].ToDbl();\
         double a_margin_free =     data[0]["a_margin_free"].ToDbl();\
         double a_margin_level =    data[0]["a_margin_level"].ToDbl();\
         double a_margin_so_call =  data[0]["a_margin_so_call"].ToDbl();\
         double a_margin_so_so =    data[0]["a_margin_so_so"].ToDbl();\
         //--- longs\
         long a_login =             data[0]["a_login"].ToInt();\
         long a_leverage =          data[0]["a_leverage"].ToInt();\
         long a_trade_mode =        data[0]["a_trade_mode"].ToInt();\
         long a_margin_so_mode =    data[0]["a_margin_so_mode"].ToInt();\
         long a_id =                data[0]["a_id"].ToInt(); //--- database generated ID\
         //--- strings\
         string a_company =         data[0]["a_company"].ToStr();\
         string a_currency =        data[0]["a_currency"].ToStr();\
         string a_name =            data[0]["a_name"].ToStr();\
         string a_server =          data[0]["a_server"].ToStr();\
         //--- booleans\
         bool a_ea_allowed =        data[0]["a_ea_allowed"].ToBool();\
         bool a_trade_allowed =     data[0]["a_trade_allowed"].ToBool();\
         //printf("Server headers: %s", headers);\
         //--- doubles\
         printf("Balance: %d", a_balance);\
         printf("Credit: %d", a_credit);\
         printf("Profit: %d", a_profit);\
         printf("Equity: %d", a_equity);\
         printf("Margin: %d", a_margin);\
         printf("Margin Free: %d", a_margin_free);\
         printf("Margin Level: %d", a_margin_level);\
         printf("Margin Call Level: %d", a_margin_so_call);\
         printf("Margin Stop Out Level: %d", a_margin_so_so);\
         //--- longs\
         printf("Login: %d", a_login);\
         printf("Leverage: %d", a_leverage);\
         printf("Trade Mode: %d", a_trade_mode);\
         printf("Margin Stop Out Mode: %d", a_margin_so_mode);\
         printf("Database ID: %d", a_id);\
         //--- strings\
         printf("Company: %s", a_company);\
         printf("Currency: %s", a_currency);\
         printf("Platform Name: %s", a_name);\
         printf("Server: %s", a_server);\
         //--- booleans\
         printf("Expert Advisor Allowed: %d", a_ea_allowed);\
         printf("Trade Allowed: %d", a_trade_allowed);\
         Print("Done!");\
        }\
      else\
         PrintFormat("Request to '%s' failed with error code %d", url, res);\
     }\
  }\
```\
\
**SQLite as a Postgres mirror**\
\
It is also possible to leverage the existing MQL5 infrastructure by using the remote data as a local SQLite database. To implement this feature, we need to synchronize the databases. This synchronization would be almost real-time, with only a few seconds of delay. But it would improve performance, avoid network latency, and allow data access via standard MetaEditor GUI and the use of MQL5 Database Functions in your MQL5 code.\
\
If you think this feature would be useful, please let me know. I'll be glad to write a detailed tutorial with sample code for this synchronization between the remote Postgres and the local SQLite databases.\
\
### Conclusion\
\
In these notes we have reviewed some currently available methods to connect an MQL5 code instance to a Postgres database. We chose a REST API as a viable and fast alternative to the more expensive dedicated driver development or the use of .dll's. Also, we developed a basic demo app as an example of how to set up a development environment for MQL5/Postgres on Windows Subsystem For Linux.\
\
Now you can start developing! Choose a good cloud provider and leverage all the power of Postgres analytics, automation, web scalability, and machine learning extensions to power up your trading.\
\
**Attached files** \|\
\
\
[Download ZIP](https://www.mql5.com/en/articles/download/12308.zip "Download all attachments in the single ZIP archive")\
\
[tuto-postgres-mql5-files.zip](https://www.mql5.com/en/articles/download/12308/tuto-postgres-mql5-files.zip "Download tuto-postgres-mql5-files.zip")(5.99 KB)\
\
[tuto-mql5-postgres-rest-demo-app-main.zip](https://www.mql5.com/en/articles/download/12308/tuto-mql5-postgres-rest-demo-app-main.zip "Download tuto-mql5-postgres-rest-demo-app-main.zip")(12.38 KB)\
\
**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.\
\
This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.\
\
#### Other articles by this author\
\
- [Statistical Arbitrage Through Cointegrated Stocks (Part 10): Detecting Structural Breaks](https://www.mql5.com/en/articles/20946)\
- [Statistical Arbitrage Through Cointegrated Stocks (Part 9): Backtesting Portfolio Weights Updates](https://www.mql5.com/en/articles/20657)\
- [Statistical Arbitrage Through Cointegrated Stocks (Part 8): Rolling Windows Eigenvector Comparison for Portfolio Rebalancing](https://www.mql5.com/en/articles/20485)\
- [Statistical Arbitrage Through Cointegrated Stocks (Part 7): Scoring System 2](https://www.mql5.com/en/articles/20173)\
- [Statistical Arbitrage Through Cointegrated Stocks (Part 6): Scoring System](https://www.mql5.com/en/articles/20026)\
- [Statistical Arbitrage Through Cointegrated Stocks (Part 5): Screening](https://www.mql5.com/en/articles/19626)\
- [Statistical Arbitrage Through Cointegrated Stocks (Part 4): Real-time Model Updating](https://www.mql5.com/en/articles/19428)\
\
**Last comments \|**\
**[Go to discussion](https://www.mql5.com/en/forum/446717)**\
(3)\
\
\
![Dominik Egert](https://c.mql5.com/avatar/2024/2/65db1363-e44b.jpg)\
\
**[Dominik Egert](https://www.mql5.com/en/users/dominik_egert)**\
\|\
14 Jul 2023 at 10:03\
\
**Hay Day [#](https://www.mql5.com/en/forum/446717#comment_48115930):**\
\
Playing CarX Street with the mod APK has made the game even more addictive. The modded version unlocks exclusive events, rewards, and achievements, making it hard to put the game down.\
\
[https://carxstreet.pro](https://www.mql5.com/go?link=https://carxstreet.pro/ "https://carxstreet.pro/")\
\
SPAM - This is spam...\
\
\
![abimael Silva](https://c.mql5.com/avatar/2020/2/5E382B8D-A1E6.jpg)\
\
**[abimael Silva](https://www.mql5.com/en/users/abimael)**\
\|\
7 Jun 2024 at 03:29\
\
Very interesting, I'm having problems with Mysql, I'll try to use your approach.\
\
\
![Jocimar Lopes](https://c.mql5.com/avatar/2023/2/63de1090-f297.jpg)\
\
**[Jocimar Lopes](https://www.mql5.com/en/users/jslopes)**\
\|\
7 Jun 2024 at 13:50\
\
**abimael Silva [#](https://www.mql5.com/en/forum/446717#comment_53613292):**\
\
Very interesting, I'm having problems with Mysql, I'll try to use your approach.\
\
It is a very simple and well-known approach, Abimael. It is only a REST app between your client and your server.\
\
Depending on your needs, I would suggest that you look for ready-made open-source API generators. Here you have an [open-source example on GitHub](https://www.mql5.com/go?link=https://github.com/blocknotes/sinatra-rest-api "https://github.com/blocknotes/sinatra-rest-api"). It is in Ruby (waaaay better :)) but the concept is the same.\
\
Good luck!\
\
![Category Theory in MQL5 (Part 7): Multi, Relative and Indexed Domains](https://c.mql5.com/2/54/Category-Theory-p7-avatar.png)[Category Theory in MQL5 (Part 7): Multi, Relative and Indexed Domains](https://www.mql5.com/en/articles/12470)\
\
Category Theory is a diverse and expanding branch of Mathematics which is only recently getting some coverage in the MQL5 community. These series of articles look to explore and examine some of its concepts & axioms with the overall goal of establishing an open library that provides insight while also hopefully furthering the use of this remarkable field in Traders' strategy development.\
\
![Creating an EA that works automatically (Part 11): Automation (III)](https://c.mql5.com/2/50/aprendendo_construindo_011_avatar.png)[Creating an EA that works automatically (Part 11): Automation (III)](https://www.mql5.com/en/articles/11293)\
\
An automated system will not be successful without proper security. However, security will not be ensured without a good understanding of certain things. In this article, we will explore why achieving maximum security in automated systems is such a challenge.\
\
![How to create a custom True Strength Index indicator using MQL5](https://c.mql5.com/2/54/true_strength_index_avatar.png)[How to create a custom True Strength Index indicator using MQL5](https://www.mql5.com/en/articles/12570)\
\
Here is a new article about how to create a custom indicator. This time we will work with the True Strength Index (TSI) and will create an Expert Advisor based on it.\
\
![Population optimization algorithms: Saplings Sowing and Growing up (SSG)](https://c.mql5.com/2/52/growing-tree-avatar.png)[Population optimization algorithms: Saplings Sowing and Growing up (SSG)](https://www.mql5.com/en/articles/12268)\
\
Saplings Sowing and Growing up (SSG) algorithm is inspired by one of the most resilient organisms on the planet demonstrating outstanding capability for survival in a wide variety of conditions.\
\
[![](https://www.mql5.com/ff/sh/0wxx5f0vuwq7xh89z2/01.png)VPS for 24/7 tradingContact your broker and find out how to get a free hosting subscriptionLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=nhetzvgituppcfrhndpblbihmzziogdh&s=d00c975c8bda3d8c1b29f042ad33ac81952ccea2f130a8f1ffa9015bab8ade87&uid=&ref=https://www.mql5.com/en/articles/12308&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5071641419489160137)\
\
This website uses cookies. Learn more about our [Cookies Policy](https://www.mql5.com/en/about/cookies).\
\
![close](https://c.mql5.com/i/close.png)\
\
![MQL5 - Language of trade strategies built-in the MetaTrader 5 client terminal](https://c.mql5.com/i/registerlandings/logo-2.png)\
\
You are missing trading opportunities:\
\
- Free trading apps\
- Over 8,000 signals for copying\
- Economic news for exploring financial markets\
\
RegistrationLog in\
\
latin characters without spaces\
\
a password will be sent to this email\
\
An error occurred\
\
\
- [Log in With Google](https://www.mql5.com/en/auth_oauth2?provider=Google&amp;return=popup&amp;reg=1)\
\
You agree to [website policy](https://www.mql5.com/en/about/privacy) and [terms of use](https://www.mql5.com/en/about/terms)\
\
If you do not have an account, please [register](https://www.mql5.com/en/auth_register)\
\
Allow the use of cookies to log in to the MQL5.com website.\
\
Please enable the necessary setting in your browser, otherwise you will not be able to log in.\
\
[Forgot your login/password?](https://www.mql5.com/en/auth_forgotten?return=popup)\
\
- [Log in With Google](https://www.mql5.com/en/auth_oauth2?provider=Google&amp;return=popup)