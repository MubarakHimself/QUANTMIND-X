---
title: How to Export Quotes from МetaTrader 5 to .NET Applications Using WCF Services
url: https://www.mql5.com/en/articles/27
categories: Integration
relevance_score: 0
scraped_at: 2026-01-24T14:09:16.173279
---

[![](https://www.mql5.com/ff/sh/bhdtjfb1zry09943z2/267b575d2182c180804d340af38ce02c.jpg)\\
Trade from your iPhone or Android device\\
\\
You only need an internet connection to use the new powerful MetaTrader 5 Web terminal\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=wtigumvtenarnsocpyfoqnanxrilnbxx&s=ec8c539e52b83881ff2d16eaff6913b25803952eb277cac55f670a102b2edc1f&uid=&ref=https://www.mql5.com/en/articles/27&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5083369690979375706)

MetaTrader 5 / Examples


### Introduction

Programmers who use the DDE service in MetaTrader 4 probably have heard that in the fifth version it is no longer supported. And there is no standard solution for exporting quotes. As a solution of this problem, the MQL5 developers suggest using your own dll, that implements it. So if we have to write the implementation, let's do it smart!

Why .NET?

For me with my long experience of programming in .NET, it would be more reasonable, interesting and simple to implement the export of quotes using this platform. Unfortunately, there isn't any native support of .NET in MQL5 in fifth version. I am sure that developers have some reasons for it. Therefore, we will use the win32 dll as a wrapper for .NET support.

Why WCF?

The Windows Communication Foundation Technology (WCF) has been chosen by me because of several reasons: on the one hand, it's easy-to-extend and adapt, on the other hand, I wanted to check it under a hard work. Moreover, according to Microsoft, WCF has a little more performance as compared to .NET Remoting.

### System Requirements

Let's think, what we want from our system. I think, there are two main requirements:

1. Of course, we need to export ticks, better using the native structure [MqlTick](https://www.mql5.com/en/docs/constants/structures/mqltick);
2. It's preferable to know the list of currently exported symbols.

Let's start...

### 1\. General classes and contracts

First of all, let's create a new class library and name it **QExport.** **dll.** We define the MqlTick structure as a DataContract:

```
    [StructLayout(LayoutKind.Sequential)]
    [DataContract]
    public struct MqlTick
    {
        [DataMember]
        public Int64 Time { get; set; }
        [DataMember]
        public Double Bid { get; set; }
        [DataMember]
        public Double Ask { get; set; }
        [DataMember]
        public Double Last { get; set; }
        [DataMember]
        public UInt64 Volume { get; set; }
    }
```

Then we'll define the contracts of the service. I don't like to use configuration classes and generated proxy-classes, so you won't meet such features here.

Let's define the first server contract according to the requirements, described above:

```
    [ServiceContract(CallbackContract = typeof(IExportClient))]
    public interface IExportService
    {
        [OperationContract]
        void Subscribe();

        [OperationContract]
        void Unsubscribe();

        [OperationContract]
        String[] GetActiveSymbols();
    }
```

As we see, there is a standard scheme of subscribing and unsubscribing from server notifications. The brief details of operations are described below:

| Opearation | Description |
| --- | --- |
| Subscribe() | Subscribe to ticks export |
| Unsubscribe() | Unsubscribe to ticks export |
| GetActiveSymbols() | Returns list of exported symbols |

And the following information should be sent to the client callback: the quote itself and notification about changes of the lits of exported symbols. Let's define the operations required as "One Way operations" to increase the performance:

```
    [ServiceContract]
    public interface IExportClient
    {
        [OperationContract(IsOneWay = true)]
        void SendTick(String symbol, MqlTick tick);

        [OperationContract(IsOneWay = true)]
        void ReportSymbolsChanged();
    }
```

| Operation | Description |
| --- | --- |
| SendTick(String, MqlTick) | Sends tick |
| ReportSymbolsChanged() | Notify the client about the changes in the exported symbols list |

### 2\. Server implementation

Let's create a new build with name Qexport.Service.dll for the service with the server contract implementation.

Let's choose the NetNamedPipesBinding for a binding, because it has
the largest performance as compared to standard bindings. If we need to broadcast quotes, over a network for example, the NetTcpBinding
should be used.

Here are some details of the server contract implementation:

The class definition. First of all, it should be marked with the ServiceBehavior attribute with the following modifiers:

- _InstanceContextMode__=__InstanceContextMode__.__Single_
\- to provide the use of one service instance for all the requests processed, it will increase the performance of solution. In addition, we will get the possibility to serve and manage the list of exported symbols;

- _ConcurrencyMode__=__ConcurrencyMode__.__Multiple_-means the parallel processing for all of the client's requests;
- _UseSynchronizationContext__=__false_– means that we don't attach to the GUI thread to prevent hang situations. It isn't necessary here for our task, but it's necessary if we want to host the service using the Windows applications.

- _IncludeExceptionDetailInFaults__=__true_– to include the exception details to the object FaultException
when passed to the client.

The ExportService itself contains two interfaces:  IExportService,
IDisposable. The first one implements all service functions, the second one implements the standard model of .NET resources release.

```
    [ServiceBehavior(InstanceContextMode = InstanceContextMode.Single,\
        ConcurrencyMode = ConcurrencyMode.Multiple,\
        UseSynchronizationContext = false,\
        IncludeExceptionDetailInFaults = true)]
    public class ExportService : IExportService, IDisposable
    {
```

Let's describe the variables of the service:

```
        // full address of service in format net.pipe://localhost/server_name
        private readonly String _ServiceAddress;

        // service host
        private ServiceHost _ExportHost;

        // active clients callbacks collection
        private Collection<IExportClient> _Clients = new Collection<IExportClient>();

        // active symbols list
        private List<String> _ActiveSymbols = new List<string>();

        // object for locking
        private object lockClients = new object();
```

Let's define the Open() and Close() methods, which open and close our service:

```
        public void Open()
        {
            _ExportHost = new ServiceHost(this);

            // point with service
            _ExportHost.AddServiceEndpoint(typeof(IExportService),  // contract
                new NetNamedPipeBinding(),                          // binding
                new Uri(_ServiceAddress));                          // address

            // remove the restriction of 16 requests in queue
            ServiceThrottlingBehavior bhvThrot = new ServiceThrottlingBehavior();
            bhvThrot.MaxConcurrentCalls = Int32.MaxValue;
            _ExportHost.Description.Behaviors.Add(bhvThrot);

            _ExportHost.Open();
        }

        public void Close()
        {
            Dispose(true);
        }

        private void Dispose(bool disposing)
        {
            try
            {
                // closing channel for each client
                // ...

                // closing host
                _ExportHost.Close();
            }
            finally
            {
                _ExportHost = null;
            }

            // ...
        }
```

Next, the implementation of IExportService methods:

```
        public void Subscribe()
        {
            // get the callback channel
            IExportClient cl = OperationContext.Current.GetCallbackChannel<IExportClient>();
            lock (lockClients)
                _Clients.Add(cl);
        }

        public void Unsubscribe()
        {
            // get the callback chanell
            IExportClient cl = OperationContext.Current.GetCallbackChannel<IExportClient>();
            lock (lockClients)
                _Clients.Remove(cl);
        }

        public String[] GetActiveSymbols()
        {
            return _ActiveSymbols.ToArray();
        }
```

Now we need to add methods to send ticks and to register and delete the exported symbols.

```
     public void RegisterSymbol(String symbol)
        {
            if (!_ActiveSymbols.Contains(symbol))
                _ActiveSymbols.Add(symbol);

              // sending notification to all clients about changes in the list of active symbols
              //...
        }

        public void UnregisterSymbol(String symbol)
        {
            _ActiveSymbols.Remove(symbol);

             // sending notification to all clients about the changes in the list of active symbols
             //...
        }

        public void SendTick(String symbol, MqlTick tick)
        {
            lock (lockClients)
                for (int i = 0; i < _Clients.Count; i++)
                    try
                    {
                        _Clients[i].SendTick(symbol, tick);
                    }
                    catch (CommunicationException)
                    {
                        // it seems that connection with client has lost - we just remove the client
                        _Clients.RemoveAt(i);
                        i--;
                    }
        }
```

Let's summarize the list of main server functions (only those that we need):

| Methods | Description |
| --- | --- |
| Open() | Runs server |
| Close() | Stops server |
| RegisterSymbol(String) | Adds symbol to the list of exported symbols |
| UnregisterSymbol(String) | Deletes symbol from the list of exported symbols |
| GetActiveSymbols() | Returns number of exported symbols |
| SendTick(String, MqlTick) | Sends tick to clients |

**3\. Client implementation**

We have considered the server, I think its clear, so it's time to consider the client. Let's build the **Qexport.Client.dll.** The client contract will be implemented there. First, it should be marked with CallbackBehavior attrubute, that defines its behaviour. It has the following modifiers:

- ConcurrencyMode = ConcurrencyMode.Multiple -
means the parallel processing for all callbacks and server responses. This modifier is very important. Imagine, that server wants to notify the client about the changes in list of exported symbols by calling callback ReportSymbolsChanged(). And the client (in its callback) wants to receive the new list of the exported symbols by calling of server method GetActiveSymbols(). So it turns out that client can't receive response from the server because it proceeding callback with waiting for the server response. As a result the client will fall because of timeout.

- _UseSynchronizationContext = false_\- specifies that we don't attach to the GUI to prevent hang situations. By default, the wcf callbacks are attached to the parent thread. If the parent thread has GUI, the situation is possible when callback waits for the completion of the method it has been called by, but the method cannot finish because it waits for the callback finish. It's something similar to the previous case, although these are two different things.


As for the server case, the client also implements two interfaces: IExportClient and IDisposable:

```
 [CallbackBehavior(ConcurrencyMode = ConcurrencyMode.Multiple,\
        UseSynchronizationContext = false)]
    public class ExportClient : IExportClient, IDisposable
    {
```

Let's describe the service variables:

```
        // full service address
        private readonly String _ServiceAddress;

        // service object
        private IExportService _ExportService;

        // Returns service instance
        public IExportService Service
        {
            get
            {
                return _ExportService;
            }
        }

        // Returns communication channel
        public IClientChannel Channel
        {
            get
            {
                return (IClientChannel)_ExportService;
            }
        }
```

Now we will create events for our callback methods. It's required for the client application to be able to subscribe to the events and get notifications about the changes of the client state.

```
        // calls when tick received
        public event EventHandler<TickRecievedEventArgs> TickRecieved;

        // call when symbol list has changed
        public event EventHandler ActiveSymbolsChanged;
```

Also define the Open() and Close() methods for the client:

```
        public void Open()
        {
            // creating channel factory
            var factory = new DuplexChannelFactory<IExportService>(
                new InstanceContext(this),
                new NetNamedPipeBinding());

            // creating server channel
            _ExportService = factory.CreateChannel(new EndpointAddress(_ServiceAddress));

            IClientChannel channel = (IClientChannel)_ExportService;
            channel.Open();

            // connecting to feeds
            _ExportService.Subscribe();
        }

        public void Close()
        {
            Dispose(true);
        }

        private void Dispose(bool disposing)
        {
            try
            {
                // unsubscribe feeds
                _ExportService.Unsubscribe();
                Channel.Close();

            }
            finally
            {
                _ExportService = null;
            }
            // ...
        }
```

Note that connection and disconnection from feeds are called when a client is opened or closed, so it isn't necessary to call them directly.

And now, let's write the client contract. Its implementation leads to generation of the following events:

```
        public void SendTick(string symbol, MqlTick tick)
        {
            // firing event TickRecieved
        }

        public void ReportSymbolsChanged()
        {
            // firing event ActiveSymbolsChanged
        }
```

Finally, the main properties and methods of the client are defined as follows:

| Property | Description |
| --- | --- |
| Service | Service communication channel |
| Channel | Instance of service contract IExportService |

| Method | Description |
| --- | --- |
| Open() | Connects to server |
| Close() | Disconnects from server |

| Event | Description |
| --- | --- |
| TickRecieved | Generated after the new quote receiving |
| ActiveSymbolsChanged | Generated after the changes in the list of active symbols |

### 4\. Transfer speed between two .NET applications

It was interesting to me to measure the transfer speed between two .NET applications, in fact, it's throughput, which is measured in ticks per second. I wrote several console applications to measure the service performance: the first on is for the server, the second one is for the client. I wrote the following code in the Main() function of the server:

```
            ExportService host = new ExportService("mt5");
            host.Open();

            Console.WriteLine("Press any key to begin tick export");
            Console.ReadKey();

            int total = 0;

            Stopwatch sw = new Stopwatch();

            for (int c = 0; c < 10; c++)
            {
                int counter = 0;
                sw.Reset();
                sw.Start();

                while (sw.ElapsedMilliseconds < 1000)
                {
                    for (int i = 0; i < 100; i++)
                    {
                        MqlTick tick = new MqlTick { Time = 640000, Bid = 1.2345 };
                        host.SendTick("GBPUSD", tick);
                    }
                    counter++;
                }

                sw.Stop();
                total += counter * 100;

                Console.WriteLine("{0} ticks per second", counter * 100);
            }

            Console.WriteLine("Average {0:F2} ticks per second", total / 10);

            host.Close();
```

As we see, the code performs ten throughput measurements.  I have got the following test results on my Athlon 3000+:

```
2600 ticks per second
3400 ticks per second
3300 ticks per second
2500 ticks per second
2500 ticks per second
2500 ticks per second
2400 ticks per second
2500 ticks per second
2500 ticks per second
2500 ticks per second
Average 2670,00 ticks per second
```

2500 ticks per second - I think it's sufficent to export quotes for 100 symbols (of course, virtually, because it seems that nobody wants to open so many charts and attach experts =)) Moreover, with increasing number of clients, the maximal number of exported symbols for each client is reduced.

### **5\. Creating a "stratum"**

Now it's time to think how to connect it with the client terminal. Let's see what we have at the first call of the function in MetaTrader 5: the .NET runtime environment (CLR) is loaded to the process and application domain is created by default. It's interesting, that it is not unloaded after the code execution.

The only way to unload CLR from the process is to terminate it (close
the client terminal), that will force Windows to clear all the process
resources. So, we can create our objects and they will exist until the
application domain is unoaded, or until it is destroyed by Garbage
Collector.

You can say that it seems good, but even
if we prevent the object destroying by Garbage Collector, we can't be
able to access the objects from MQL5. Fortunately, such access can be organized easily. The trick is the following: for each application domain there is a table of Garbage Collector handles (GC handle table), which is used by application to track the object lifetime and allows to manage it manually.

The application adds and deletes elements from the table by using the type _System.Runtime.InteropServices.GCHandle._ All we need is to wrap our object with such a descriptor and we have an access to it throughout the property _GCHandle.Target._ Thus we can get the reference to the object _GCHandle,_ which is in the table of handles and it's guaranteed that it will not be moved or deleted by Garbage Collector. The wrapped object will also aviod recycling, because of the reference by descriptor.

Now it's time to test the theory in practice. To do it, let's create a new win32 dll with name **QExpertWrapper.** **dll** and add the CLR support, System.dll, QExport.dll,
Qexport.Service.dll to the build reference. Also we create an auxiliary class ServiceManaged for management purposes – to carry out marshalling, to receive objects by handles, etc.

```
ref class ServiceManaged
{
        public:
                static IntPtr CreateExportService(String^);
                static void DestroyExportService(IntPtr);
                static void RegisterSymbol(IntPtr, String^);
                static void UnregisterSymbol(IntPtr, String^);
                static void SendTick(IntPtr, String^, IntPtr);
};
```

Let's consider the implementation of these methods. The CreateExportService method creates the service, wraps it into GCHandle by using GCHandle.Alloc and returns its reference. If something goes wrong, it shows a MessageBox with an error. I have used it for the debug purpose, so I am not sure that it's really necessary, but I've left it here just in case.

```
IntPtr ServiceManaged::CreateExportService(String^ serverName)
{
        try
        {
                ExportService^ service = gcnew ExportService(serverName);
                service->Open();

                GCHandle handle = GCHandle::Alloc(service);
                return GCHandle::ToIntPtr(handle);
        }
        catch (Exception^ ex)
        {
                MessageBox::Show(ex->Message, "CreateExportService");
        }
}
```

The DestroyExportServicemethod gets the pointer to the GCHandle of service, gets the service from the Target property and calls its method Close(). It's important to release the service object by calling its method Free(). Overwise it will remain in memory, the Garbage Collector doesn't remove it.

```
void ServiceManaged::DestroyExportService(IntPtr hService)
{
        try
        {
                GCHandle handle = GCHandle::FromIntPtr(hService);

                ExportService^ service = (ExportService^)handle.Target;
                service->Close();

                handle.Free();
        }
        catch (Exception^ ex)
        {
                MessageBox::Show(ex->Message, "DestroyExportService");
        }
}
```

The RegisterSymbol method adds a symbol to the list of exported symbols:

```
void ServiceManaged::RegisterSymbol(IntPtr hService, String^ symbol)
{
        try
        {
                GCHandle handle = GCHandle::FromIntPtr(hService);
                ExportService^ service = (ExportService^)handle.Target;

                service->RegisterSymbol(symbol);
        }
        catch (Exception^ ex)
        {
                MessageBox::Show(ex->Message, "RegisterSymbol");
        }
}
```

The UnregisterSymbol method deletes a symbol from the list:

```
void ServiceManaged::UnregisterSymbol(IntPtr hService, String^ symbol)
{
        try
        {
                GCHandle handle = GCHandle::FromIntPtr(hService);
                ExportService^ service = (ExportService^)handle.Target;

                service->UnregisterSymbol(symbol);
        }
        catch (Exception^ ex)
        {
                MessageBox::Show(ex->Message, "UnregisterSymbol");
        }
}
```

And now the SendTick method. As we see, the pointer is transformed to the MqlTick structure using the Marshal class. Another point: there isn't any code in catch block - it is done to avoid the lags of the general tick queue in the case of error.

```
void ServiceManaged::SendTick(IntPtr hService, String^ symbol, IntPtr hTick)
{
        try
        {
                GCHandle handle = GCHandle::FromIntPtr(hService);
                ExportService^ service = (ExportService^)handle.Target;

                MqlTick tick = (MqlTick)Marshal::PtrToStructure(hTick, MqlTick::typeid);

                service->SendTick(symbol, tick);
        }
        catch (...)
        {
        }
}
```

Let's consider the implementation of functions, which will be called from our ex5 programs:

```
#define _DLLAPI extern "C" __declspec(dllexport)

// ---------------------------------------------------------------
// Creates and opens service
// Returns its pointer
// ---------------------------------------------------------------
_DLLAPI long long __stdcall CreateExportService(const wchar_t* serverName)
{
        IntPtr hService = ServiceManaged::CreateExportService(gcnew String(serverName));

        return (long long)hService.ToPointer();
}

// ----------------------------------------- ----------------------
// Closes service
// ---------------------------------------------------------------
_DLLAPI void __stdcall DestroyExportService(const long long hService)
{
        ServiceManaged::DestroyExportService(IntPtr((HANDLE)hService));
}

// ---------------------------------------------------------------
// Sends tick
// ---------------------------------------------------------------
_DLLAPI void __stdcall SendTick(const long long hService, const wchar_t* symbol, const HANDLE hTick)
{
        ServiceManaged::SendTick(IntPtr((HANDLE)hService), gcnew String(symbol), IntPtr((HANDLE)hTick));
}

// ---------------------------------------------------------------
// Registers symbol to export
// ---------------------------------------------------------------
_DLLAPI void __stdcall RegisterSymbol(const long long hService, const wchar_t* symbol)
{
        ServiceManaged::RegisterSymbol(IntPtr((HANDLE)hService), gcnew String(symbol));
}

// ---------------------------------------------------------------
// Removes symbol from list of exported symbols
// ---------------------------------------------------------------
_DLLAPI void __stdcall UnregisterSymbol(const long long hService, const wchar_t* symbol)
{
        ServiceManaged::UnregisterSymbol(IntPtr((HANDLE)hService), gcnew String(symbol));
}
```

The code is ready, now we need to compile and build it. Let's specify the output directory as "C:\\Program Files\\MetaTrader 5\\MQL5\\Libraries" in the project options. After the compilation three libraries will appear in the specified folder.

The mql5 program uses only one of them, namely QExportWrapper.dll, two other libraries are used by it. Because of this reason we need to put the libraries Qexport.dll and Qexport.Service.dll into root folder of MetaTrader. It isn't convenient.

The solution is to create configuration file and specify the path for the libraries there. Let's create the file with name **terminal.exe.config** in the root folder of MetaTrader and write the following strings there:

```
<?xml version="1.0" encoding="UTF-8" ?>
<configuration>
   <runtime>
      <assemblyBinding xmlns="urn:schemas-microsoft-com:asm.v1">
         <probing privatePath="mql5\libraries" />
      </assemblyBinding>
   </runtime>
</configuration>
```

It's ready. Now CLR will search for the libraries in the folder we have specified.

### **6\. Server part implementation in** **MQL5**

Finally, we have reached the programming of  server part in mql5. Let's create a new file **QService.** **mqh** and define the imported functions of **QExpertWrapper.** **dll**:

```
#import "QExportWrapper.dll"
   long  CreateExportService(string);
   void DestroyExportService(long);
   void RegisterSymbol(long, string);
   void UnregisterSymbol(long, string);
   void SendTick(long, string, MqlTick&);
#import

```

It's great that mql5 has classes because it's an ideal feature to incapsulate all the logics inside, that significantly simplifies the work and understanding of code. Therefore let's design a class that will be a shell for the library methods.

Moreover, to avoid the situation with creation of service for every symbol, let's organize the checking of the working service with such name, and we will work through it in such a case. An ideal method to serve this information are global variables, because of the following reasons:

- the global variables disappear after the client terminal close. The same is with the service;
- we can serve the number of objects
Qservice, that uses the service. It allows to close the physical service only after the last object is closed.

So, let's create a class Qservice:

```
class QService
{
   private:
      // service pointer
      long hService;
      // service name
      string serverName;
      // name of the global variable of the service
      string gvName;
      // flag that indicates is service closed or not
      bool wasDestroyed;

      // enters the critical section
      void EnterCriticalSection();
      // leaves the critical section
      void LeaveCriticalSection();

   public:

      QService();
      ~QService();

      // opens service
      void Create(const string);
      // closes service
      void Close();
      // sends tick
      void SendTick(const string, MqlTick&);
};

//--------------------------------------------------------------------
QService::QService()
{
   wasDestroyed = false;
}

//--------------------------------------------------------------------
QService::~QService()
{
   // close if it hasn't been destroyed
   if (!wasDestroyed)
      Close();
}

//--------------------------------------------------------------------
QService::Create(const string serviceName)
{
   EnterCriticalSection();

   serverName = serviceName;

   bool exists = false;
   string name;

   // check for the active service with such name
   for (int i = 0; i < GlobalVariablesTotal(); i++)
   {
      name = GlobalVariableName(i);
      if (StringFind(name, "QService|" + serverName) == 0)
      {
         exists = true;
         break;
      }
   }

   if (!exists)   // if not exists
   {
      // starting service
      hService = CreateExportService(serverName);
      // adding a global variable
      gvName = "QService|" + serverName + ">" + (string)hService;
      GlobalVariableTemp(gvName);
      GlobalVariableSet(gvName, 1);
   }
   else          // the service is exists
   {
      gvName = name;
      // service handle
      hService = (int)StringSubstr(gvName, StringFind(gvName, ">") + 1);
      // notify the fact of using the service by this script
      // by increase of its counter
      GlobalVariableSet(gvName, NormalizeDouble(GlobalVariableGet(gvName), 0) + 1);
   }

   // register the chart symbol
   RegisterSymbol(hService, Symbol());

   LeaveCriticalSection();
}

//--------------------------------------------------------------------
QService::Close()
{
   EnterCriticalSection();

   // notifying that this script doen't uses the service
   // by decreasing of its counter
   GlobalVariableSet(gvName, NormalizeDouble(GlobalVariableGet(gvName), 0) - 1);

   // close service if there isn't any scripts that uses it
   if (NormalizeDouble(GlobalVariableGet(gvName), 0) < 1.0)
   {
      GlobalVariableDel(gvName);
      DestroyExportService(hService);
   }
   else UnregisterSymbol(hService, Symbol()); // unregistering symbol

   wasDestroyed = true;

   LeaveCriticalSection();
}

//--------------------------------------------------------------------
QService::SendTick(const string symbol, MqlTick& tick)
{
   if (!wasDestroyed)
      SendTick(hService, symbol, tick);
}

//--------------------------------------------------------------------
QService::EnterCriticalSection()
{
   while (GlobalVariableCheck("QService_CriticalSection") > 0)
      Sleep(1);
   GlobalVariableTemp("QService_CriticalSection");
}

//--------------------------------------------------------------------
QService::LeaveCriticalSection()
{
   GlobalVariableDel("QService_CriticalSection");
}
```

The class contains the following methods:

| Method | Description |
| --- | --- |
| Create(const string) | Starts service |
| Close() | Closes service |
| SendTick(const string, MqlTick&) | Sends quote |

Also note that the private methods EnterCriticalSection() and LeaveCriticalSection() allow you to run the critical code sections between them.

It will relieve us from the cases of the simultaneous
calls of function Create() and creation of new services for each
QService.

So, we have described the class for working with service, now let's write an Expert Advisor for the quotes broadcasting. The Expert Advisor has been chosen because of its possibility to process all the ticks arrived.

```
//+------------------------------------------------------------------+
//|                                                    QExporter.mq5 |
//|                                             Copyright GF1D, 2010 |
//|                                             garf1eldhome@mail.ru |
//+------------------------------------------------------------------+
#property copyright "GF1D, 2010"
#property link      "garf1eldhome@mail.ru"
#property version   "1.00"

#include "QService.mqh"
//--- input parameters
input string  ServerName = "mt5";

QService* service;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   service = new QService();
   service.Create(ServerName);
   return(0);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   service.Close();
   delete service;
   service = NULL;
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
   MqlTick tick;
   SymbolInfoTick(Symbol(), tick);

   service.SendTick(Symbol(), tick);
}
//+------------------------------------------------------------------+
```

### **7\. Testing communication performance between ex5 and .NET client**

It's evident, that the total performance of service will decreased if the quotes will arrive directly from the client terminal, so I have interested to measure it. I was sure that it should have decreased because of the inevitable loss of CPU time for marshalling and typecasting.

For this purpose I wrote a simple script that is the same as for the first test. The Start() function looks as follows:

```
   QService* serv = new QService();
   serv.Create("mt5");

   MqlTick tick;
   SymbolInfoTick("GBPUSD", tick);

   int total = 0;

   for(int c = 0; c < 10; c++)
   {
      int calls = 0;

      int ticks = GetTickCount();

      while(GetTickCount() - ticks < 1000)
      {
         for(int i = 0; i < 100; i++) serv.SendTick("GBPUSD", tick);
         calls++;
      }

      Print(calls * 100," calls per second");

      total += calls * 100;
   }

   Print("Average ", total / 10," calls per second");

   serv.Close();
   delete serv;
```

I have got the following results:

```
1900  calls per second
2400  calls per second
2100  calls per second
2300  calls per second
2000  calls per second
2100  calls per second
2000  calls per second
2100  calls per second
2100  calls per second
2100  calls per second
Average  2110  calls per second
```

2500 ticks/sec vs 1900 ticks/sec. 25% is the price that should be paid for the use of services from MT5, but anyway it's sufficient. It's interesting to note that performance can be increased by using the threads pool and static method **System** **.Threading.ThreadPool.QueueUserWorkItem**.

Using this method, I have got the transfer speed up to 10000 ticks per second. But its work in a hard testing was unstable because of the fact that the Garbage Collector has no time to delete objects - as a result the memory, allocated by MetaTrader grows rapidly and finally it crashes. But it was a hard testing, far from the real, so there is nothing dangerous in using the threads pool.

### **8\. Realtime testing**

I have created an example of ticks table using the service. The project is attached in the archive and named **WindowsClient**. The result of its work is presented below:

![](https://c.mql5.com/2/0/pic1.gif)

Fig 1. Main window of the WindowsClient application with quotes table

### Conclusion

I this article I have described one of the methods of exporting quotes to .NET applications. All the required has been implemented and now we have ready classes that can be used in your own applications. The  only one thing that it isn't convenient to attach scripts to each of the necessary chart.

At present time I think that this problem can be solved using the MetaTrader profiles. From the other side, if you don't need all quotes, you can organize it with a script that broadcasts quotes for the necessary symbols. As you understand, the market depth broadcasting or even two-side access can be organized the same way.

Description of archives:

**Bin.rar** \- archive with a ready solution. For users, who want to see
how it works. Still note that .NET Framework 3.5 (maybe
it also will work with version 3.0) should be installed on your
computer.

**Src.rar**-
full source code of the project. To work with it you'll need MetaEditor
and Visual Studio
2008.

**QExportDemoProfile.rar** -  Metatrader profile, that attaches script to 10 charts, as shown at Fig. 1.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/27](https://www.mql5.com/ru/articles/27)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/27.zip "Download all attachments in the single ZIP archive")

[qexportdemoprofile.rar](https://www.mql5.com/en/articles/download/27/qexportdemoprofile.rar "Download qexportdemoprofile.rar")(7.37 KB)

[bin.rar](https://www.mql5.com/en/articles/download/27/bin.rar "Download bin.rar")(33.23 KB)

[src.rar](https://www.mql5.com/en/articles/download/27/src.rar "Download src.rar")(137.41 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Practical Application Of Databases For Markets Analysis](https://www.mql5.com/en/articles/69)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/541)**
(28)


![Douglas Mendes](https://c.mql5.com/avatar/2015/10/5611D2CD-918B.jpg)

**[Douglas Mendes](https://www.mql5.com/en/users/douglasmendes)**
\|
18 Oct 2015 at 17:32

Just posted a new job based on this article: [https://www.mql5.com/en/job/34392](https://www.mql5.com/en/job/34392) .

It's not working in my MT5 64 bits environment...

Great article!

Thanks

![Douglas Mendes](https://c.mql5.com/avatar/2015/10/5611D2CD-918B.jpg)

**[Douglas Mendes](https://www.mql5.com/en/users/douglasmendes)**
\|
20 Oct 2015 at 12:23

Just for knowledge, I discovered what happened in my 64 bits machine.

After hours and hours of researching and debugging, discovered that one referenced assembly was not loading, generating the exception "System.IO.FileNotFoundException: Unable to load file or assembly 'QExport.Service, Version=1.0.5771.13857, Culture=neutral, PublicKeyToken=56996a45dd1e337b'".

Maybe because the dll has no config file, don't know yet, MT 5 did not know where to find the assembly. So it was trying to get it in the base path (path where metaeditor64.exe is located). After changing the output directory of the referenced [projects](https://www.mql5.com/en/articles/7863 "Article: Projects allow you to create profitable trading robots! But it is not exactly") to that path, worked as a charm.

![Douglas Mendes](https://c.mql5.com/avatar/2015/10/5611D2CD-918B.jpg)

**[Douglas Mendes](https://www.mql5.com/en/users/douglasmendes)**
\|
20 Oct 2015 at 12:25

**sabe:**

Hi Joe,

Was there any special trick to get it working on x64? I've just compiled it for x64, but the dll crashes with weird errors on startup.

Sabe, see my answer below.

\[\]'s

![MAWO](https://c.mql5.com/avatar/avatar_na2.png)

**[MAWO](https://www.mql5.com/en/users/mawo)**
\|
14 Jun 2019 at 13:12

Hello,

I have tested your finished file, everything looks good in MT5. But there is no data in the WinClient.

![](https://c.mql5.com/3/282/image__44.png)

[![](https://c.mql5.com/3/282/image__46.png)](https://c.mql5.com/3/282/image__45.png "https://c.mql5.com/3/282/image__45.png")

![Rain Kambar](https://c.mql5.com/avatar/avatar_na2.png)

**[Rain Kambar](https://www.mql5.com/en/users/nightdev)**
\|
11 Dec 2024 at 15:40

I wonder if there is a similar [project](https://www.mql5.com/en/articles/7863 "Article: Projects allow you to create profitable trading robots! But it's not for sure ") with pre-loading of historical bars? I would like to add it all to Lightweight-Chart and use the chart conveniently.

![MQL5.community - User Memo](https://c.mql5.com/2/0/helpButton__1.png)[MQL5.community - User Memo](https://www.mql5.com/en/articles/24)

You have just registered and most likely you have questions such as, "How do I insert a picture to my a message?" "How do I format my MQL5 source code?" "Where are my personal messages kept?" You may have many other questions. In this article, we have prepared some hands-on tips that will help you get accustomed in MQL5.community and take full advantage of its available features.

![Easy Stock Market Trading with MetaTrader](https://c.mql5.com/2/16/779_23.gif)[Easy Stock Market Trading with MetaTrader](https://www.mql5.com/en/articles/1566)

This article raises the issues of automated trading on the stock market. Examples of MetaTrader 4 and QUIK integration are provided for your information. In addition to that, you can familiarize yourself with MetaTrader advantages aimed at solving this issue, and see how a trading robot can perform operations on MICEX.

![The Order of Object Creation and Destruction in MQL5](https://c.mql5.com/2/0/recycle_ava__1.png)[The Order of Object Creation and Destruction in MQL5](https://www.mql5.com/en/articles/28)

Every object, whether it is a custom object, a dynamic array or an array of objects, is created and deleted in MQL5-program in its particular way. Often, some objects are part of other objects, and the order of object deleting at deinitialization becomes especially important. This article provides some examples that cover the mechanisms of working with objects.

![Checking the Myth: The Whole Day Trading Depends on How the Asian Session Is Traded](https://c.mql5.com/2/17/872_36.png)[Checking the Myth: The Whole Day Trading Depends on How the Asian Session Is Traded](https://www.mql5.com/en/articles/1575)

In this article we will check the well-known statement that "The whole day trading depends on how the Asian session is traded".

[Running robots on virtual hosting is easyFollow our step-by-step MetaTrader VPS guide for beginnersRead![](https://www.mql5.com/ff/sh/au4fqg4kms7s9mq1z2/01.png)![](https://www.mql5.com/ff/sh/au4fqg4kms7s9mq1z2/02.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/articles/13586&a=uzpprdshbcrtxvjxpmescehprypbymxc&s=516438f25b531570d9b7d49dcfb29c82fa1021f5ede6571df8026dbfbafcd13f&uid=&ref=https://www.mql5.com/en/articles/27&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5083369690979375706)

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