---
title: Using MetaTrader 5 as a Signal Provider for MetaTrader 4
url: https://www.mql5.com/en/articles/344
categories: Trading, Trading Systems, Expert Advisors
relevance_score: 4
scraped_at: 2026-01-23T17:41:08.327031
---

[![](https://www.mql5.com/ff/sh/0hvxp984jjj79943z2/6373d9e5710a718ffa6a7d50a5db9dd1.jpg)\\
Web terminal on your iPhone or Android\\
\\
Full-featured MetaTrader 5 platform for any devices and web browsers\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=uyigsjnbfcdvysiynusmriwvhincciwd&s=c95531ae2fd8a81b0fac3def2e4cf820a67584bbf4b02f76ec75f808942dbbd2&uid=&ref=https://www.mql5.com/en/articles/344&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5068525107183286914)

MetaTrader 5 / Examples


### Introduction

There has been multiple reasons for me, why I have chosen to write this article and to investigate if it's doable.

First, [MetaTrader 5](https://www.metatrader5.com/ "https://www.metatrader5.com/") has been out and available for a long time, but we all are still waiting for our favourite brokers to allow us trade in real. Some has made strategies by using [MQL5](https://www.mql5.com/en/docs) and have a good performance, and want to run them on real accounts now. Others, maybe, likes how trading is organised and wants to trade manually, but by using MetaTrader 5, instead of MetaTrader 4.

Second reason, during the [Automated Trading Championship](https://championship.mql5.com/2011/en "https://championship.mql5.com/2011/en") everyone has been thinking about following leaders in their own real accounts. Some has created their own way of following the trades, but some are still searching how to do it, how to get results as close as possible to traders in championship, how to apply the same money management.

Third, some people have good strategies and they want to provide their trading signals not only to themselves, but also to their friends or others. They need possibility to accept multiple connections without loosing performance and distribute signals real time.

These are the questions which have been in my mind all the time and I will try to find a solution which would cover these requirements.

### 1\. How to follow MQL5 championship activities?

Lately I have found multiple articles in [MQL5.community](https://www.mql5.com/en) which was in my knowledge level and made me think that I could build it. I will also tell to you that I have been using application which was following activities in championship homepage and was trading in my real account (luckily, with profit). Problem was - data is updated each 5 minutes and you can miss the right moment to open and close.

From [championship forum](https://championship.mql5.com/2011/en/comments "https://championship.mql5.com/2011/en/comments") I understood that there are other people which is doing the same thing, and it's not effective and also it gives huge traffic for championship homepage and organisers might not like it. So, is there a solution? I looked at all solutions and I liked the possibility to access every participant's account in 'investor' (trading disabled) mode through MetaTrader 5.

Can we use it to receive information of every trade activity in real time and to transfer it in real time? To find it, I created Expert Advisor and tried to run it on account which had only 'investor' mode access. For my surprise, it was possible to attach it and also, it was possible to get information about Positions, Orders and Deals - those where doors to possible solution!

### 2\. What to follow - Positions, Orders or Deals?

If we are about to transfer information from MetaTrader 5 to MetaTrader 4, then we need to take in consideration all order types which are possible in MetaTrader 4. Also, when we follow, we want to know about every action performed in account related to trading, therefore 'Positions' will not give us full information unless we compare status of 'Positions' on every tick or second.

Therefore, it would be better to follow 'Orders' or 'Deals'.

I started to looks at Orders:

![Orders](https://c.mql5.com/2/3/Orders__2.PNG)

I liked that they are executed before 'Deal' is and also they contain information about pending (limit) orders, but they lacking one important thing compared to 'Deals' - entry type ( [ENUM\_DEAL\_ENTRY](https://www.mql5.com/en/docs/constants/tradingconstants/dealproperties#enum_deal_entry)):

![Deals](https://c.mql5.com/2/3/Deals__2.PNG)

DEAL\_ENTRY\_TYPE helps to understand what happened in traders account while 'Orders' require calculation in parallel. The best would be to merge 'Deals' with 'Orders', then we could have pending orders and also follow every action in trade account. Since price movements differs between different broker companies, then pending orders could actually lead to mistakes and incorrect results.

In case if we follow 'Deals' only, we will still execute pending orders, but with small delay (up to network connection). Between speed(pending orders) and performance(deals) I choosed to go for performance('Deals').

### 3\. How to provide 'signals'?

There have been different articles and discussions how to communicate and transfer data from MetaTrader 5 to other applications and computers. Since I want other clients to be able to connect to us and they most likely will be located on other computers, then I choose TCP socket connection.

Since MQL5 does not allow to do it with API functions, then we need to use external library. There are multiple articles about involving "WinInet.dll" library (e.g. " [Using WinInet.dll for Data Exchange between Terminals via the Internet](https://www.mql5.com/en/articles/73)" and other) but none of them really satisfy our needs.

Since I'm a little bit familiar with C#, then I decided to create my own library. For this, I used article " [Exposing C# code to MQL5 using unmanaged exports](https://www.mql5.com/en/articles/249)" to help me out with compatibility issues. I created server with very simple interface and possibility to accept up to 500 clients in the same time (requires .NET framework 3.5 or later in your PC. Already installed in most of computers. " [Microsoft .NET Framework 3.5](https://www.mql5.com/go?link=https://www.microsoft.com/en-us/download/details.aspx?id=21 "http://www.microsoft.com/download/en/details.aspx?id=21")").

```
#import "SocketServer.dll"    // Library created on C# (created by using information available on https://www.mql5.com/en/articles/249)
string About();            // Information about library.
int SendToAll(string msg);  // Sends one text message to all clients.
bool Stop();               // Stops the server.
bool StartListen(int port); // Starts the server. Server will listen from incomming connections (max 500 clients).
                               // All clients are built on Assync threads.
string ReadLogLine();       // Retrieve one log line from server (can contain erros and other information).
                               // Reading is optional. Server stores only last 100 lines.
#import
```

Server itself is running in background on separate threads and will not block or slow down work of MetaTrader 5 or your strategy, no matter how many clients will be connected.

C# source code:

```
         internal static void WaitForClients()
        {
            if (server != null)
            {
                Debug("Cant start lisening! Server not disposed.");
                return;
            }
            try
            {

                IPEndPoint localEndPoint = new IPEndPoint(IPAddress.Any, iPort);
                server = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);

                server.Bind(localEndPoint);
                server.Listen(500);
                isServerClosed = false;
                isServerClosedOrClosing = false;

                while (!isServerClosedOrClosing)
                {
                    allDone.Reset();
                    server.BeginAccept(new AsyncCallback(AcceptCallback), server);
                    allDone.WaitOne();
                }

            }
            catch (ThreadAbortException)
            {
            }
            catch (Exception e)
            {
                Debug("WaitForClients() Error: " + e.Message);
            }
            finally
            {
                if (server != null)
                {
                    server.Close();
                    server = null;
                }
                isServerClosed = true;
                isServerClosedOrClosing = true;
            }
        }

        internal static void AcceptCallback(IAsyncResult ar)
        {
            try
            {
                allDone.Set();
                if (isServerClosedOrClosing)
                    return;
                Socket listener = (Socket)ar.AsyncState;
                Socket client = listener.EndAccept(ar);

                if (clients != null)
                {
                    lock (clients)
                    {
                        Array.Resize(ref clients, clients.Length + 1);
                        clients[clients.Length - 1].socket = client;
                        clients[clients.Length - 1].ip = client.RemoteEndPoint.ToString();
                        clients[clients.Length - 1].alive = true;
                    }
                    Debug("Client connected: " + clients[clients.Length - 1].ip);
                }
            }
            catch (Exception ex)
            {
                Debug("AcceptCallback() Error: " + ex.Message);
            }
        }
```

To find out more about Asynchronous Server Sockets in C# I recommend you to read [Microsoft MSDN](https://www.mql5.com/go?link=https://docs.microsoft.com/en-us/dotnet/framework/network-programming/using-an-asynchronous-server-socket "http://msdn.microsoft.com/en-us/library/5w7b7x5f.aspx") or some articles which you can find with [Google](https://www.mql5.com/go?link=http://www.google.com/ "http://www.google.com/").

### 4\. How to collect 'signals'?

On MetaTrader 4 we would like to receive information all the time and not only when new tick is generated, therefore we create 'Script' for it, instead of Expert Advisor. Also, we need to be able to open socket connection with our signal provider - MetaTrader 5.

For this I choose to get help from MQL4 codebase: " [https://www.mql5.com/en/code/9296](https://www.mql5.com/en/code/9296)". There I found quite good include file ( [WinSock.mqh](https://www.mql5.com/en/code/9296)) which allows to work with sockets in very simple way. Even some people has been complaining about stability, I found it good enough for my purpose and haven't experienced any problems during my testing.

```
#include <winsock.mqh>  // Downloaded from MQ4 homepage
                     // DOWNLOAD:   http://codebase.mql4.com/download/18644
                     // ARTICLE:    http://codebase.mql4.com/6122
```

### 5\. Data processing

Now we have our concept and all we need to do is to make sure deals are processed and transferred one by one to all clients in format which they can understand and execute.

**5.1. Server side**

As we clarified, it will be Expert Advisor but it does not care about currency on which it has been added.

During start-up it will also start listening thread which will be waiting for incoming connections:

```
int OnInit()
  {
   string str="";
   Print(UTF8_to_ASCII(About()));
//--- start the server
   Print("Starting server on port ",InpPort,"...");
   if(!StartListen(InpPort))
     {
      PrintLogs();
      Print("OnInit() - FAILED");
      return -1;
     }
```

In this version, Expert Advisor will not care about connected clients. Every time there is a trade - it will send notification to all clients, even there are none. Since we need to know only about trades then we will use function [OnTrade()](https://www.mql5.com/en/docs/basis/function/events#ontrade) and will remove [OnTick()](https://www.mql5.com/en/docs/basis/function/events#ontick). In this function we look at latest history and decide if this is a deal we need to inform about or not.

See my comments in code to understand it better:

```
//+------------------------------------------------------------------+
//| OnTrade() - every time when there is an activity related to      |
//|             traiding.                                            |
//+------------------------------------------------------------------+
void OnTrade()
  {
//--- find all new deals and report them to all connected clients
//--- 24 hours back.
   datetime dtStart=TimeCurrent()-60*60*24;
//--- 24 hours front (in case if you live in GMT-<hours>)
   datetime dtEnd=TimeCurrent()+60*60*24;
//--- select history from last 24 hours.
   if(HistorySelect(dtStart,dtEnd))
     {
      //--- go through all deals (from oldest to newest).
      for(int i=0;i<HistoryDealsTotal();i++)
        {
         //--- get deal ticket.
         ulong ticket=HistoryDealGetTicket(i);
         //--- if this deal is interesting for us.
         if(HistoryDealGetInteger(ticket,DEAL_ENTRY)!=DEAL_ENTRY_STATE)
           {
            //Print("Entry type ok.");
            //--- check if this deal is newer than previously reported one.
            if(HistoryDealGetInteger(ticket,DEAL_TIME)>g_dtLastDealTime)
              {
               //--- if some part of position has been closed then check if we need to enable it
               if(HistoryDealGetInteger(ticket,DEAL_ENTRY)==DEAL_ENTRY_OUT)
                 {
                  vUpdateEnabledSymbols();
                 }
               //--- if opposite position is opened, then we need to enable disabled symbol.
               else if(HistoryDealGetInteger(ticket,DEAL_ENTRY)==DEAL_ENTRY_INOUT)
                 {
                  //--- enable this specific symbol.
                  vEnableSymbol(HistoryDealGetString(ticket,DEAL_SYMBOL));
                 }
               //--- check if symbol is enabled.
               if(bIsThisSymbolEnabled(HistoryDealGetString(ticket,DEAL_SYMBOL)))
                 {
                  //--- build deal-string and send to all connected clients
                  int cnt=SendToAll(sBuildDealString(ticket));
                  //--- technical error with server.
                  if(cnt<0)
                    {
                     Print("Failed to send new deals!");
                    }
                  //--- if sent to no one (cnt==0) or if sent to someone (cnt>0)
                  else
                    {
                     //--- update datetime for last sucessfully transfered deal
                     g_dtLastDealTime=(datetime)HistoryDealGetInteger(ticket,DEAL_TIME);
                    }
                 }
               //--- do not notify becayse symbol is disabled.
               else
                 {
                  //--- update datetime for last deal, we will not notify about.
                  g_dtLastDealTime=(datetime)HistoryDealGetInteger(ticket,DEAL_TIME);
                 }
              }
           }
        }
     }
  }
```

As you noticed, when there is new deal found, we call function BuildDealString() to prepare data for transfer. All data are transferred in text format and each deal starts with '<' and ends with '>'.

This will help us to separate multiple deals since it is possible to receive more than one deal at the time due to TCP/IP protocol.

```
//+------------------------------------------------------------------+
//| This function builds deal string                                 |
//| Examples:                                                        |
//| EURUSD;BUY;IN;0.01;1.37294                                       |
//| EURUSD;SELL;OUT;0.01;1.37310                                     |
//| EURUSD;SELL;IN;0.01;1.37320                                      |
//| EURUSD;BUY;INOUT;0.02;1.37294                                    |
//+------------------------------------------------------------------+
string sBuildDealString(ulong ticket)
  {
   string deal="";
   double volume=0;
   bool bFirstInOut=true;
//--- find deal volume.
//--- if this is INOUT then volume must contain ONLY volume of 'IN'.
   if(HistoryDealGetInteger(ticket,DEAL_ENTRY)==DEAL_ENTRY_INOUT)
     {
      if(PositionSelect(HistoryDealGetString(ticket,DEAL_SYMBOL)))
        {
         volume=PositionGetDouble(POSITION_VOLUME);
        }
      else
        {
         Print("Failed to get volume!");
        }
     }
//--- if it's 'IN' or 'OUT' deal then use it's volume as is.
   else
     {
      volume=HistoryDealGetDouble(ticket,DEAL_VOLUME);
     }
//--- build deal string(format sample: "<EURUSD;BUY;IN;0.01;1.37294>").
   int iDealEntry=(int)HistoryDealGetInteger(ticket,DEAL_ENTRY);
//--- if this is OUT deal, and there are no open positions left.
   if(iDealEntry==DEAL_ENTRY_OUT && !PositionSelect(HistoryDealGetString(ticket,DEAL_SYMBOL)))
     {
      //--- For safety reasons, we check if there is any position left with current symbol. If NO, then let's use
      //--- new deal type - OUTALL. This will guarante that there are no open orders left on or account when all
      //--- position has been closed on 'remote' MetaTrader 5 side. This can happen due to fact, that volume is
      //--- is mapped to new values on client side, therefor there can be some very small difference which leaves
      //--- order open with very small lot size.
      iDealEntry=DEAL_ENTRY_OUTALL;  // My own predefined value (this value should not colide with EMUN_DEAL_ENTRY values).
     }
   StringConcatenate(deal,"<",AccountInfoInteger(ACCOUNT_LOGIN),";",
                   HistoryDealGetString(ticket,DEAL_SYMBOL),";",
                   Type2String((ENUM_DEAL_TYPE)HistoryDealGetInteger(ticket,DEAL_TYPE)),";",
                   Entry2String(iDealEntry),";",DoubleToString(volume,2),";",
                      DoubleToString(HistoryDealGetDouble(ticket,DEAL_PRICE),
                   (int)SymbolInfoInteger(HistoryDealGetString(ticket,DEAL_SYMBOL),SYMBOL_DIGITS)),">");
   Print("DEAL:",deal);
   return deal;
  }
```

When looking at code, you might be surprised about new DEAL\_ENTRY type - DEAL\_ENTRY\_OUTALL. It is created by me and you will understand more about it when I will explain about volume handling in MetaTrader 4 side.

One more thing which might be interesting is [OnTimer()](https://www.mql5.com/en/docs/basis/function/events#ontimer) function. During initialization I call [EventSetTimer(1)](https://www.mql5.com/en/docs/eventfunctions/eventsettimer) to get OnTimer() call every second. Inside if that function is one line which prints out information (logs) from server library:

```
//+------------------------------------------------------------------+
//| Print logs from Server every second (if there are any)           |
//+------------------------------------------------------------------+
void OnTimer()
  {
   PrintLogs();
  }
```

Call this function (PrintLogs) after every function which you execute from server library, to print out status and error information.

In server side you will also find an input parameter StartupType:

```
enum ENUM_STARTUP_TYPE
  {
   STARTUP_TYPE_CLEAR,    // CLEAR - Send every new DEAL wich appears on account.
   STARTUP_TYPE_CONTINUE  // CONTINUE - Do not send DEAL before existing POSITION has been closed.
  };
//--- input parameters
input ENUM_STARTUP_TYPE InpStartupType=STARTUP_TYPE_CONTINUE; // Startup type
```

This is added do to fact that signal provider can be added on account which already have opened positions (e.g if following championship) and therefore information about them might be misleading on client side. By this parameter you can choose, if you want to receive information from existing trades or only from newly opened positions.

It is also important if you apply to account for first time or you reapply for account on which you have been running it before and you have just restarted your PC, program or made a modification in your code.

**5.2. Client**

On client side we have script which is looping in socket receive function for infinity (recv). Since this function in 'blocking' then script is locked for the time till something is received from server, so no worries about processor time.

```
//--- server up and running. Start data collection and processing
   while(!IsStopped())
     {
      Print("Client: Waiting for DEAL...");
      ArrayInitialize(iBuffer,0);
      iRetVal=recv(iSocketHandle,iBuffer,ArraySize(iBuffer)<<2,0);
      if(iRetVal>0)
        {
         string sRawData=struct2str(iBuffer,iRetVal<<18);
         Print("Received("+iRetVal+"): "+sRawData);
```

This causes a problem to stop the client. When you will click "Remove Script", script will not be removed. You need to click it twice and then script will be removed by time-out. This could be fixed if time-out for receive function could be applied, but since I'm using sample already available in Codebase then I will leave it for original author.

Once data are received we do splitting and verification before deal is processed in real account:

```
         //--- split records
         string arrDeals[];
         //--- split raw data in multiple deals (in case if more than one is received).
         int iDealsReceived=Split(sRawData,"<",10,arrDeals);
         Print("Found ",iDealsReceived," deal orders.");
         //--- process each record
         //--- go through all DEALs received
         for(int j=0;j<iDealsReceived;j++)
           {
            //--- split each record to values
            string arrValues[];
            //--- split each DEAL in to values
            int iValuesInDeal=Split(arrDeals[j],";",10,arrValues);
            //--- verify if DEAL request received in correct format (with correct count of values)
            if(iValuesInDeal==6)
              {
                 if(ProcessOrderRaw(arrValues[0],arrValues[1],arrValues[2],
                                    arrValues[3],arrValues[4],
                                         StringSubstr(arrValues[5],0,StringLen(arrValues[5])-1)))
                 {
                  Print("Processing of order done sucessfully.");
                 }
               else
                 {
                  Print("Processing of order failed:\"",arrDeals[j],"\"");
                 }
              }
            else
              {
               Print("Invalid order received:\"",arrDeals[j],"\"");
               //--- this was last one in array
               if(j==iDealsReceived-1)
                 {
                  //--- it might be incompleate beginning of next deal.
                  sLeftOver=arrDeals[j];
                 }
              }
           }
```

```
//+------------------------------------------------------------------+
//| Processing received raw data (text format)                       |
//+------------------------------------------------------------------+
bool ProcessOrderRaw(string saccount,string ssymbol,string stype,string sentry,string svolume,string sprice)
  {
//--- clearing
   saccount= Trim(saccount);
   ssymbol = Trim(ssymbol);
   stype=Trim(stype);
   sentry=Trim(sentry);
   svolume= Trim(svolume);
   sprice = Trim(sprice);
//--- validations
   if(!ValidateAccountNumber(saccount)){Print("Invalid account:",saccount);return(false);}
   if(!ValidateSymbol(ssymbol)){Print("Invalid symbol:",ssymbol);return(false);}
   if(!ValidateType(stype)){Print("Invalid type:",stype);return(false);}
   if(!ValidateEntry(sentry)){Print("Invalid entry:",sentry);return(false);}
   if(!ValidateVolume(svolume)){Print("Invalid volume:",svolume);return(false);}
   if(!ValidatePrice(sprice)){Print("Invalid price:",sprice);return(false);}
//--- convertations
   int account=StrToInteger(saccount);
   string symbol=ssymbol;
   int type=String2Type(stype);
   int entry=String2Entry(sentry);
   double volume= GetLotSize(StrToDouble(svolume),symbol);
   double price = NormalizeDouble(StrToDouble(sprice),(int)MarketInfo(ssymbol,MODE_DIGITS));
   Print("DEAL[",account,"|",symbol,"|",Type2String(type),"|",\
        Entry2String(entry),"|",volume,"|",price,"]");
//--- execution
   ProcessOrder(account,symbol,type,entry,volume,price);
   return(true);
  }
```

Since, not everyone have 10 000$ on their account, then recalculation of Lot size is done on client side by function GetLotSize(). Strategy running on server side can also imply money management and therefore we need to do the same on client side.

I offer you "Lot mapping" - user of client can specify its lot size preferences (min and max) and then Client Script will do the mapping for you:

```
extern string _1 = "--- LOT MAPPING ---";
extern double  InpMinLocalLotSize  =  0.01;
extern double  InpMaxLocalLotSize  =  1.00; // Recomended bigger than
extern double  InpMinRemoteLotSize =  0.01;
extern double  InpMaxRemoteLotSize =  15.00;
```

```
//+------------------------------------------------------------------+
//| Calculate lot size                                               |
//+------------------------------------------------------------------+
double GetLotSize(string remote_lots, string symbol)
{
   double dRemoteLots = StrToDouble(remote_lots);
   double dLocalLotDifference = InpMaxLocalLotSize - InpMinLocalLotSize;
   double dRemoteLotDifference = InpMaxRemoteLotSize - InpMinRemoteLotSize;
   double dLots = dLocalLotDifference * (dRemoteLots / dRemoteLotDifference);
   double dMinLotSize = MarketInfo(symbol, MODE_MINLOT);
   if(dLots<dMinLotSize)
      dLots=dMinLotSize;
   return (NormalizeDouble(dLots,InpVolumePrecision));
}
```

Client side supports 4 and 5 digit brokers and it also has 'regular-lot' (0.1) and 'mini-lot' (0.01) support. For this reason I needed to create new DEAL\_ENTRY type - DEAL\_OUTALL.

Since client side is doing mapping, there can be some situation when small lot size leaves unclosed.

```
void ProcessOrder(int account, string symbol, int type, int entry, double volume, double price)
{
   if(entry==OP_IN)
   {
      DealIN(symbol,type,volume,price,0,0,account);
   }
   else if(entry==OP_OUT)
   {
      DealOUT(symbol, type, volume, price, 0, 0,account);
   }
   else if(entry==OP_INOUT)
   {
      DealOUT_ALL(symbol, type, account);
      DealIN(symbol,type,volume,price,0,0,account);
   }
   else if(entry==OP_OUTALL)
   {
      DealOUT_ALL(symbol, type, account);
   }
}
```

**5.3. MetaTrader 5 Positions vs MetaTrader 4 Orders**

During implementation, I found another problem - in MetaTrader 5  there is always only one position for each symbol while in MetaTrader 4 it's handled in totally different way. To get as close as possible, each new deal with the same entry and symbol, I cover by opening multiple orders on MetaTrader 4 side.

Each new 'IN' deal is a new order and when there is an 'OUT' deal, I implemented functionality which performs 3 step closing:

1. Go through all open orders and close the one which match requested size, if none, then

2. Go through all open orders and close those which are smaller than requested OUT volume size, if something is still left, then
3. Close order which size is bigger than requested size and open new order with size which should be left unclosed. In normal cases, third step should never be performed. Created for protection purposes.

```
//+------------------------------------------------------------------+
//| Process DEAL ENTRY OUT                                           |
//+------------------------------------------------------------------+
void DealOUT(string symbol, int cmd, double volume, double price, double stoploss, double takeprofit, int account)
{
   int type = -1;
   int i=0;

   if(cmd==OP_SELL)
      type = OP_BUY;
   else if(cmd==OP_BUY)
      type = OP_SELL;

   string comment = "OUT."+Type2String(cmd);
   //--- Search for orders with equal VOLUME size and with PROFIT > 0
   for(i=0;i<OrdersTotal();i++)
   {
      if(OrderSelect(i,SELECT_BY_POS))
      {
         if(OrderMagicNumber()==account)
         {
            if(OrderSymbol()==symbol)
            {
               if(OrderType()==type)
               {
                  if(OrderLots()==volume)
                  {
                     if(OrderProfit()>0)
                     {
                        if(CloseOneOrder(OrderTicket(), symbol, type, volume))
                        {
                           Print("Order with exact volume and profit>0 found and executed.");
                           return;
                        }
                     }
                  }
               }
            }
         }
      }
   }
   //--- Search for orders with equal VOLUME size and with ANY profit size
   for(i=0;i<OrdersTotal();i++)
   {
      if(OrderSelect(i,SELECT_BY_POS))
      {
         if(OrderMagicNumber()==account)
         {
            if(OrderSymbol()==symbol)
            {
               if(OrderType()==type)
               {
                  if(OrderLots()==volume)
                  {
                     if(CloseOneOrder(OrderTicket(), symbol, type, volume))
                     {
                        Print("Order with exact volume found and executed.");
                        return;
                     }
                  }
               }
            }
         }
      }
   }
   double volume_to_clear = volume;
   //--- Search for orders with smaller volume AND with PROFIT > 0
   int limit = OrdersTotal();
   for(i=0;i<limit;i++)
   {
      if(OrderSelect(i,SELECT_BY_POS))
      {
         if(OrderMagicNumber()==account)
         {
            if(OrderSymbol()==symbol)
            {
               if(OrderType()==type)
               {
                  if(OrderLots()<=volume_to_clear)
                  {
                     if(OrderProfit()>0)
                     {
                        if(CloseOneOrder(OrderTicket(), symbol, type, OrderLots()))
                        {
                           Print("Order with smaller volume and profit>0 found and executed.");
                           volume_to_clear-=OrderLots();
                           if(volume_to_clear==0)
                           {
                              Print("All necessary volume is closed.");
                              return;
                           }
                           limit = OrdersTotal();
                           i = -1; // Because it will be increased at end of cycle and will have value 0.
                        }
                     }
                  }
               }
            }
         }
      }
   }
   //--- Search for orders with smaller volume
   limit = OrdersTotal();
   for(i=0;i<limit;i++)
   {
      if(OrderSelect(i,SELECT_BY_POS))
      {
         if(OrderMagicNumber()==account)
         {
            if(OrderSymbol()==symbol)
            {
               if(OrderType()==type)
               {
                  if(OrderLots()<=volume_to_clear)
                  {
                     if(CloseOneOrder(OrderTicket(), symbol, type, OrderLots()))
                     {
                        Print("Order with smaller volume found and executed.");
                        volume_to_clear-=OrderLots();
                        if(volume_to_clear==0)
                        {
                           Print("All necessary volume is closed.");
                           return;
                        }
                        limit = OrdersTotal();
                        i = -1; // Because it will be increased at end of cycle and will have value 0.
                     }
                  }
               }
            }
         }
      }
   }
   //--- Search for orders with higher volume
   for(i=0;i<OrdersTotal();i++)
   {
      if(OrderSelect(i,SELECT_BY_POS))
      {
         if(OrderMagicNumber()==account)
         {
            if(OrderSymbol()==symbol)
            {
               if(OrderType()==type)
               {
                  if(OrderLots()>=volume_to_clear)
                  {
                     if(CloseOneOrder(OrderTicket(), symbol, type, OrderLots()))
                     {
                        Print("Order with smaller volume found and executed.");
                        volume_to_clear-=OrderLots();
                        if(volume_to_clear<0)//Closed too much
                        {
                           //Open new to compensate lose
                           DealIN(symbol,type,volume_to_clear,price,OrderStopLoss(),OrderTakeProfit(),account);
                        }
                        else if(volume_to_clear==0)
                        {
                           Print("All necessary volume is closed.");
                           return;
                        }
                     }
                  }
               }
            }
         }
      }
   }
   if(volume_to_clear!=0)
   {
      Print("Some volume left unclosed: ",volume_to_clear);
   }
}
```

### Conclusion

Files made and attached here can definitely be improved with better client server protocol, smarter communication and better execution, but my task was to verify if it is possible, and to build it with acceptable quality, so everyone could use it for their private needs.

It works good enough to follow your own strategies and strategies for all participants in MQL5 Championship. Performance and possibilities which are provided by MQL4 and MQL5 are good enough to even take it in professional and commercial way. I believe it is possible to make a very good signal provider for all MetaTrader 4 and MetaTrader 5 clients by just using your private computer and your own strategy.

I would like to see people to improving code which I have provided here and to come back with opinions and recommendations. I will also try to answer your questions in case if you will have any. Parallel, I'm running test where I follow my favourite championship participants. Now it has been running good for a week. If I will find any problems then I will provide you with updates.

**Tsaktuo**

**Please note, by applying described functionality and executables to your real account, you take full responsibility for all loses or damages which might be caused by it. Trade on real account ONLY after good testing and ONLY with a good understanding about functionality which is provided here.**

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/344.zip "Download all attachments in the single ZIP archive")

[dealclient\_mql4.zip](https://www.mql5.com/en/articles/download/344/dealclient_mql4.zip "Download dealclient_mql4.zip")(7.85 KB)

[dealserver\_mql5.zip](https://www.mql5.com/en/articles/download/344/dealserver_mql5.zip "Download dealserver_mql5.zip")(20.6 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/5249)**
(16)


![HongDi s& t development co.,ltd.](https://c.mql5.com/avatar/avatar_na2.png)

**[hongbin fei](https://www.mql5.com/en/users/codeidea)**
\|
24 Mar 2013 at 15:32

No. I think this error cannot be changed

error:129, mean [price changed](https://www.mql5.com/en/docs/constants/errorswarnings/enum_trade_return_codes "MQL5 Documentation: Prices have changed")

![enbo lu](https://c.mql5.com/avatar/2013/9/52326F50-93D0.jpg)

**[enbo lu](https://www.mql5.com/en/users/luenbo)**
\|
10 Sep 2013 at 07:06

I wonder if there will be an [Automated Trading](https://www.mql5.com/en/book "Book \"MQL5 Programming for Traders\"") Championship ATC2014 next year?


![Gyorgy Keczan](https://c.mql5.com/avatar/2013/11/5281025A-B6D4.jpg)

**[Gyorgy Keczan](https://www.mql5.com/en/users/qpack)**
\|
22 May 2014 at 12:48

Hello!

MT4 does not open the file, I can not be attached chart.

Not is TsaktuoDealClient.ex4, only TsaktuoDealClient.mql4 file.

Why is that?

2014.05.22 12:36:32.413Cannot [open file](https://www.mql5.com/en/docs/files/fileopen "MQL5 documentation: FileOpen function") 'C:\\Users\\gyurc\\AppData\\Roaming\\MetaQuotes\\Terminal\\F8B0CF1E1FEED3B00D2D7E193237B799\\MQL4\\Experts\\Scripts\\TsaktuoDealClient.ex4' \[2\]

![MetaCoderX](https://c.mql5.com/avatar/avatar_na2.png)

**[MetaCoderX](https://www.mql5.com/en/users/metacoderx)**
\|
22 Jan 2015 at 13:29

Excelent implementation. With the recent insolvency of Alpari Broker, there is only a few brokers with MT5 servers so, and only than 2 brokers with MT5, ECN, and 1:500 leverage. So this code is extremly useful.

Could you please update it to the last version fo MT5 and MT4?

Great work, well commented code!!

![Miguel Angel Alzate Lora](https://c.mql5.com/avatar/2020/2/5E4E8C02-459F.jpg)

**[Miguel Angel Alzate Lora](https://www.mql5.com/en/users/angelalzate)**
\|
13 Feb 2019 at 19:43

Thank you very much in advance for the article, very well explained. Secondly, I would like to know if I can use this same procedure to feed mt4 with mt5 data. Thank you and happy afternoon


![Interview with Ge Senlin (ATC 2011)](https://c.mql5.com/2/0/yyy999_avatar.png)[Interview with Ge Senlin (ATC 2011)](https://www.mql5.com/en/articles/549)

The Expert Advisor by Ge Senlin (yyy999) from China got featured in the top ten of the Automated Trading Championship 2011 in late October and hasn't left it since then. Not often participants from the PRC show good results in the Championship - Forex trading is not allowed in this country. After the poor results in the previous year ATC, Senlin has prepared a new multicurrency Expert Advisor that never closes loss positions and uses position increase instead. Let's see whether this EA will be able to rise even higher with such a risky strategy.

![Interview with Ilnur Khasanov (ATC 2011)](https://c.mql5.com/2/0/aharata.png)[Interview with Ilnur Khasanov (ATC 2011)](https://www.mql5.com/en/articles/548)

The Expert Advisor of Ilnur Khasanov (aharata) is holding its place in our TOP-10 chart of the Automated Trading Championship 2011 participants from the third week already, though Ilnur's acquaintance with Forex has started only a year ago. The idea that forms the basis of the Expert Advisor is simple but the trading robot contains self-optimization elements. Perhaps, that is the key to its survival? Besides, the author had to change the Expert Advisor planned to be submitted for the Championship...

![Dr. Tradelove or How I Stopped Worrying and Created a Self-Training Expert Advisor](https://c.mql5.com/2/0/smart_EA.png)[Dr. Tradelove or How I Stopped Worrying and Created a Self-Training Expert Advisor](https://www.mql5.com/en/articles/334)

Just over a year ago joo, in his article "Genetic Algorithms - It's Easy!", gave us a tool for implementation of the genetic algorithm in MQL5. Now utilizing the tool we will create an Expert Advisor that will genetically optimize its own parameters upon certain boundary conditions...

![Interview with Igor Korepin (ATC 2011)](https://c.mql5.com/2/0/Xupypr_ava.png)[Interview with Igor Korepin (ATC 2011)](https://www.mql5.com/en/articles/547)

Appearance of the Expert Advisor cs2011 by Igor Korepin (Xupypr) at the very top of the Automated Trading Championship 2011 was really impressive - its balance was almost twice that of the EA featured on the second place. However, despite such a sound breakaway, the Expert Advisor could not stay long on the first line. Igor frankly said that he relied much on a lucky start of his trading robot in the competition. We'll see if luck helps this simple EA to take the lead in the ATC 2011 race again.

[![](https://www.mql5.com/ff/sh/0wxx5f0vuwq7xh89z2/01.png)VPS for 24/7 tradingContact your broker and find out how to get a free hosting subscriptionLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=nhetzvgituppcfrhndpblbihmzziogdh&s=d00c975c8bda3d8c1b29f042ad33ac81952ccea2f130a8f1ffa9015bab8ade87&uid=&ref=https://www.mql5.com/en/articles/344&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5068525107183286914)

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