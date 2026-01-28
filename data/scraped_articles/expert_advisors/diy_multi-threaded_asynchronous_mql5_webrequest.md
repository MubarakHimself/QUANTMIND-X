---
title: DIY multi-threaded asynchronous MQL5 WebRequest
url: https://www.mql5.com/en/articles/5337
categories: Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T21:28:44.540199
---

[![](https://www.mql5.com/ff/si/h2ryn394uwcpxwmxc2.jpg)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Feconomic-calendar%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dopen.calendar%26utm_content%3Deconomic.calendar%26utm_campaign%3Den.0009.desktop.default&a=qdeulxgvibgwytgewnvfatbocjnnninc&s=5c0c60f00ff5f5bedb0fdf65d9d79eb820442eb43ffac2b85aa003224f9dba14&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=pykcdanrkggxiphpjpgosskjyznsagab&ssn=1769192922992173265&ssn_dr=0&ssn_sr=0&fv_date=1769192922&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F5337&back_ref=https%3A%2F%2Fwww.google.com%2F&title=DIY%20multi-threaded%20asynchronous%20MQL5%20WebRequest%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176919292287242605&fz_uniq=5071861815735955418&sv=2552)

MetaTrader 5 / Expert Advisors


Implementation of trading algorithms often requires analyzing data from various external sources, including Internet. MQL5 provides the [WebRequest](https://www.mql5.com/en/docs/network/webrequest) function for sending HTTP requests to the "outside world", but, unfortunately, it has one noticeable drawback. The function is synchronous meaning it blocks the EA operation for the entire duration of a request execution. For each EA, MetaTrader 5 allocates a single thread that sequentially executes the existing API function calls in the code, as well as incoming event handlers (such as ticks, depth of market changes in BookEvent, timer, trading operations, chart events, etc.). Only one code fragment is executed at a time, while all remaining "tasks" wait for their turn in queues, till the current fragment returns control to the kernel.

For example, if an EA should process new ticks in real time and periodically check economic news on one or several websites, it is impossible to fulfill both requirements without them interfering with one other. As soon as WebRequest is executed in the code, the EA remains "frozen" on the function call string, while new tick events are skipped. Even with the ability to read skipped ticks using the CopyTicks function, the moment for making a trading decision may be missed. Here is how this situation is illustrated using the UML sequence diagram:

![Event handling sequence diagram featuring the blocking code in one thread](https://c.mql5.com/2/34/multiweb1mql.png)

**Fig.1. Event handling sequence diagram featuring the blocking code in one thread**

In this regard, it would be good to create a tool for asynchronous non-blocking execution of HTTP requests, a kind of WebRequestAsync. Obviously, we need to get hold of additional threads for that. The easiest way to do this in MetaTrader 5 is to run additional EAs, to which you can send additional HTTP requests. Besides, you can call WebRequest there and obtain the results after a while. While the request is being processed in such an auxiliary EA, our main EA remains available for prompt and interactive actions. The UML sequence diagram may look like this for that case:

![The sequence diagram delegating asynchronous event handling to other threads](https://c.mql5.com/2/34/multiweb2mql.png)

**Fig. 2. The sequence diagram delegating asynchronous event handling to other threads**

### 1\. Planning

As you know, each EA should work on a separate chart in MetaTrader. Thus, the creation of auxiliary EAs requires dedicated charts for them. Doing it manually is inconvenient. Therefore, it makes sense to delegate all routine operations to a special manager - an EA that would manage a pool of auxiliary charts and EAs, and also provide a single entry point for registering new requests from client programs. In a sense, this architecture can be called a 3-level one similar to the client-server architecture, where the EA manager acts as a server:

![The multiweb library architecture: client MQL code - server (assistant pool manager) - helper EAs](https://c.mql5.com/2/34/multiweb4mql.png)

**Fig. 3 The multiweb library architecture: client MQL code <-> server (assistant pool manager) <-> helper EAs**

But for the sake of simplification, the manager and the auxiliary EA can be implemented in the form of the same code (program). One of the two roles of such a "universal" EA - a manager or an assistant - will be determined by the priority law. The first instance launched declares itself a manager, opens auxiliary charts and launches a specified number of itself in the role of assistants.

What exactly and how should the client, the manager and the assistants pass to each other? To understand this, let's analyze the WebRequest function.

As you know, MetaTrader 5 features two options of the WebRequest function. We will consider the second one to be the most universal.

```
int WebRequest
(
  const string      method,           // HTTP method
  const string      url,              // url address
  const string      headers,          // headers
  int               timeout,          // timeout
  const char        &data[],          // HTTP message body array
  char              &result[],        // array with server response data
  string            &result_headers   // server response headers
);
```

The first five parameters are input ones. They are passed from the calling code to the kernel and define the request contents. The last two parameters are output ones. They are passed from the kernel to the calling code and contain the query result. Obviously, turning this function into an asynchronous one actually requires dividing it into two components: initializing the query and getting the results:

```
int WebRequestAsync
(
  const string      method,           // HTTP method
  const string      url,              // url address
  const string      headers,          // headers
  int               timeout,          // timeout
  const char        &data[],          // HTTP message body array
);

int WebRequestAsyncResult
(
  char              &result[],        // array with server response data
  string            &result_headers   // server response headers
);
```

The names and prototypes of the functions are conditional. In fact, we need to pass this information between different MQL programs. Normal function calls are not suitable for this. To let MQL programs "communicate" with each other, MetaTrader 5 has the [custom events](https://www.mql5.com/en/docs/eventfunctions/eventchartcustom) exchange system we are going to use. Event exchange is performed based on a receiver ID using [ChartID](https://www.mql5.com/en/docs/chart_operations/chartid) — it is unique for each chart. There may only be one EA on a chart, but there is no such limitation in case of indicators. This means a user should make sure that each chart contains no more than one indicator communicating with the manager.

In order for the data exchange to work, you need to pack all "function" parameters into the user event parameters. Both request parameters and results can contain fairly large amounts of information that do not physically fit into into the limited scope of events. For example, even if we decide to pass the HTTP method and the URL in the sparam string event parameter, limiting the length to 63 characters would be an obstacle in most working cases. This means that an event exchange system needs to be supplemented with some kind of shared data repository, and only links to records in this repository should be sent in the event parameters. Fortunately, MetaTrader 5 provides such storage in the form of [custom resources](https://www.mql5.com/en/docs/common/resourcecreate). In fact, resources dynamically created from MQL are always images. But an image is a container of binary information, where you can write anything you want.

To simplify the task, we will use a ready-made solution for writing and reading arbitrary data into user resources — classes from [Resource.mqh and ResourceData.mqh](https://www.mql5.com/en/code/22166) developed by a member of the MQL5 community [fxsaber](https://www.mql5.com/en/users/fxsaber).

The provided link leads to a source — the TradeTransactions library is not related to the current article's subject, but the discussion (in Russian) contains an [example of data storage and exchange via the resources](https://www.mql5.com/ru/forum/277866#comment_8753037). Since the library can change, and also for the convenience of readers, all the files used in the article are attached below, but their versions correspond to the time of writing the article and may differ from the current versions provided via the link above. Besides, mentioned resource classes use yet another library in their work — [TypeToBytes](https://www.mql5.com/en/code/16280). Its version is also attached to the article.

We do not need to delve into the internal structure of these auxiliary classes. The main thing is that we can rely on the ready-made RESOURCEDATA class as a “black box” and use its constructor and a couple of functions suitable for us. We will look at this in more detail later. Now, let's elaborate on the overall concept.

The sequence of the interaction of our architecture parts looks as follows:

1. To perform an asynchronous web request, the client MQL program should use the classes we develop to pack the request parameters into a local resource and send a custom event to the manager with a link to the resource; the resource is created within the client program and is not deleted until the results are obtained (when it becomes unnecessary);
2. The manager finds an unoccupied assistant EA in the pool and sends it a link to the resource; however, this instance is marked as temporarily occupied and cannot be selected for subsequent requests until the current request has been processed;
3. Parameters of a web request from the client external resource are unpacked in the assistant EA that received a custom event;
4. The assistant EA calls the standard blocking WebRequest and waits for an answer (header and/or web document);
5. The assistant EA packs the request results into its local resource and sends a custom event to the manager with a link to this resource;
6. The manager forwards the event to the client and marks the appropriate assistant as free again;
7. The client receives a message from the manager and unpacks the result of the request from the external assistant resource;
8. The client and the assistant can delete their local resources.

Results can be passed more efficiently on steps 5 and 6 due to the fact that the assistant EA sends the result directly to the client window bypassing the manager.

The steps described above are related to the main stage of processing HTTP requests. Now, it is time to describe linking disparate parts into a single architecture. It also partially relies on user events.

The central link of the architecture — the manager — is supposed to be launched manually. You should do it only once. Like any other running EA, it automatically recovers together with the chart after the terminal restarts. The terminal allows only one web request manager.

The manager creates the required number of auxiliary windows (to be set in the settings) and launches instances of themselves in them that “find out” about their assistant status thanks to the special “protocol” (details are in the implementation section).

Any assistant informs the manager of its closing with the help of a special event. This is necessary to maintain a relevant list of available assistants in the manager. Similarly, the manager notifies assistants of its closing. In turn, the assistants stop working and close their windows. The assistants are of no use without the manager, while re-launching the manager inevitably re-creates the assistants (for example, if you change the number of assistants in the settings).

Windows for assistants, like the auxiliary EAs themselves, are always supposed to be created automatically from the manager, and therefore our program should “clean them up”. Do not launch the assistant EA manually — inputs that do not correspond with the manager status are considered an error by the program.

During its launch, the client MQL program should survey the terminal window for the presence of the manager using bulk messaging and specifying its ChartID in the parameter. The manager (if found) should return the ID of its window to the client. After that, the client and the manager can exchange messages.

These are the main features. It is time to move on to implementation.

### 2\. Implementation

To simplify the development, create a single multiweb.mqh header file where we describe all the classes: some of them are common for the client and "servers", while others are inherited and specific for each of these roles.

### 2.1. Base classes (start)

Let's start from the class storing resources, IDs and variables of each element. Instances of classes derived from it will be used in the manager, in the assistants and in the client. In the client and in the assistants, such objects are needed primarily to store the resources "passed by the link". Beside this, note that several instances were created in the client to execute multiple web requests simultaneously. Therefore, the analysis of the current requests' status (at least of whether an object is already busy or not) should be used on the clients to the full extent. In the manager, these objects are used to implement identification and tracking the assistants' status. Below is the base class.

```
class WebWorker
{
  protected:
    long chartID;
    bool busy;
    const RESOURCEDATA<uchar> *resource;
    const string prefix;

    const RESOURCEDATA<uchar> *allocate()
    {
      release();
      resource = new RESOURCEDATA<uchar>(prefix + (string)chartID);
      return resource;
    }

  public:
    WebWorker(const long id, const string p = "WRP_"): chartID(id), busy(false), resource(NULL), prefix("::" + p)
    {
    }

    ~WebWorker()
    {
      release();
    }

    long getChartID() const
    {
      return chartID;
    }

    bool isBusy() const
    {
      return busy;
    }

    string getFullName() const
    {
      return StringSubstr(MQLInfoString(MQL_PROGRAM_PATH), StringLen(TerminalInfoString(TERMINAL_PATH)) + 5) + prefix + (string)chartID;
    }

    virtual void release()
    {
      busy = false;
      if(CheckPointer(resource) == POINTER_DYNAMIC) delete resource;
      resource = NULL;
    }

    static void broadcastEvent(ushort msg, long lparam = 0, double dparam = 0.0, string sparam = NULL)
    {
      long currChart = ChartFirst();
      while(currChart != -1)
      {
        if(currChart != ChartID())
        {
          EventChartCustom(currChart, msg, lparam, dparam, sparam);
        }
        currChart = ChartNext(currChart);
      }
    }
};
```

The variables:

- chartID — ID of the chart an MQL program has been launched at;
- busy — if the current instance is busy processing a web request;
- resource — resource of an object (random data storage); the RESOURCEDATA class is taken from ResourceData.mqh;
- prefix — unique prefix for each status; a prefix is used in the names of resources. In a particular client, it is recommended to make a unique settings as shown below. Assistant EAs use the "WRR\_" (abbreviated from Web Request Result) prefix by default.

The 'allocate' method to be used in derived classes. It creates an object of the RESOURCEDATA<uchar> type resource in the 'resource' variable. The chart ID is also used in naming the resource, together with the prefix. The resource can be released using the 'release' method.

The getFullName method should be mentioned in particular, since it returns the full resource name, which includes the current MQL program name and directory path. The full name is used to access third-party program resources (for reading only). For example, if the multiweb.mq5 EA is located in MQL5\\Experts and launched on the chart with the ID 129912254742671346, the resource in it receives the full name "\\Experts\\multiweb.ex5::WRR\_129912254742671346". We will pass such strings to resources as a link using the sparam string parameter of custom events.

The broadcastEvent static method, which sends messages to all windows, will be used in the future to find the manager.

To work with a request and an associated resource in the client program, we define the ClientWebWorker class derived from WebWorker (hereinafter the code is abbreviated, the full versions are in the attached files).

```
class ClientWebWorker : public WebWorker
{
  protected:
    string _method;
    string _url;

  public:
    ClientWebWorker(const long id, const string p = "WRP_"): WebWorker(id, p)
    {
    }

    string getMethod() const
    {
      return _method;
    }

    string getURL() const
    {
      return _url;
    }

    bool request(const string method, const string url, const string headers, const int timeout, const uchar &body[], const long managerChartID)
    {
      _method = method;
      _url = url;

      // allocate()? and what's next?
      ...
    }

    static void receiveResult(const string resname, uchar &initiator[], uchar &headers[], uchar &text[])
    {
      Print(ChartID(), ": Reading result ", resname);

      ...
    }
};
```

First of all, note that the 'request' method is an actual implementation of step 1 described above. Here a web request is sent to the manager. The method declaration follows the prototype of hypothetical WebRequestAsync. The receiveResult static method performs the reverse action from step 7. As the 'resname' first input, it receives the full name of the external resource in which the request results are stored, while the 'initiator', 'headers' and 'text' byte arrays are to be filled within the method with data unpacked from the resource.

What is 'initiator'? The answer is very simple. Since all our "calls" are now asynchronous (and the order of their execution is not guaranteed), we should be able to match the result with the previously sent request. Therefore, the assistance EAs pack the full name of the source client resource used to initiate the request into their response resource together with data obtained from the Internet. After unpacking, the name gets into the 'initiator' parameter and can be used to associate the result with the corresponding request.

The receiveResult method is static, since it uses no object variables — all results are returned to the calling code via the parameters.

Both methods contain ellipses where packing and unpacking data to and from resources are required. This will be considered in the next section.

### 2.2. Packing requests and request results into resources

As we remember, resources are supposed to be processed at the lower level using the RESOURCEDATA class. This is a [template](https://www.mql5.com/en/docs/basis/oop/class_templates) class, meaning it accepts a parameter with a data type that we write and read to or from a resource. Since our data also contain strings, it is reasonable to choose the smallest uchar type as a storage unit. Thus, the object of the RESOURCEDATA<uchar> class is used as a data container. When creating a resource, a unique (for the program) 'name' is created in its constructor:

```
RESOURCEDATA<uchar>(const string name)
```

We can pass this name (supplemented by the program name as a prefix) in custom events, so that other MQL programs are able to access the same resource. Please note that all other programs, except the one within which the resource was created, have read-only access.

Data is written to the resource using the overloaded assignment operator:

```
void operator=(const uchar &array[]) const
```

where 'array' is a kind of an array we have to prepare.

Reading data from the resource is performed using the function:

```
int Get(uchar &array[]) const
```

Here, 'array' is an output parameter where the original array contents is placed.

Now let's turn to the application aspect of using resources to pass data about HTTP requests and their results. We are going to create a layer class between resources and the main code - ResourceMediator. The class is to pack the 'method', 'url', 'headers', 'timeout' and 'data' parameters to the 'array' byte array and then write to the resource on the client's side. On the server side, it is to unpack the parameters from the resource. Similarly, this class will package the server-side 'result' and 'result\_headers' parameters into the 'array' byte array and write to the resource to read it as an array and unpack it on the client side.

The ResourceMediator constructor accepts the pointer to the RESOURCEDATA resource, which will then be processed inside the methods. In addition, ResourceMediator contains supporting structures for storing meta information about data. Indeed, when packing and unpacking resources, we need a certain header containing the sizes of all the fields in addition to the data itself.

For example, if we simply use the StringToCharArray function to convert a URL into an array of bytes, then when performing the inverse operation using CharArrayToString, we need to set the array length. Otherwise, not only URL bytes but also the header field following them will be read from the array. As you may remember, we store all data in a single array before witing to the resource. Meta info about the length of the fields should also be converted into a sequence of bytes. We apply unions for that.

```
#define LEADSIZE (sizeof(int)*5) // 5 fields in web-request

class ResourceMediator
{
  private:
    const RESOURCEDATA<uchar> *resource; // underlying asset

    // meta-data in header is represented as 5 ints `lengths` and/or byte array `sizes`
    union lead
    {
      struct _l
      {
        int m; // method
        int u; // url
        int h; // headers
        int t; // timeout
        int b; // body
      }
      lengths;

      uchar sizes[LEADSIZE];

      int total()
      {
        return lengths.m + lengths.u + lengths.h + lengths.t + lengths.b;
      }
    }
    metadata;

    // represent int as byte array and vice versa
    union _s
    {
      int x;
      uchar b[sizeof(int)];
    }
    int2chars;


  public:
    ResourceMediator(const RESOURCEDATA<uchar> *r): resource(r)
    {
    }

    void packRequest(const string method, const string url, const string headers, const int timeout, const uchar &body[])
    {
      // fill metadata with parameters data lengths
      metadata.lengths.m = StringLen(method) + 1;
      metadata.lengths.u = StringLen(url) + 1;
      metadata.lengths.h = StringLen(headers) + 1;
      metadata.lengths.t = sizeof(int);
      metadata.lengths.b = ArraySize(body);

      // allocate resulting array to fit metadata plus parameters data
      uchar data[];
      ArrayResize(data, LEADSIZE + metadata.total());

      // put metadata as byte array at the beginning of the array
      ArrayCopy(data, metadata.sizes);

      // put all data fields into the array, one by one
      int cursor = LEADSIZE;
      uchar temp[];
      StringToCharArray(method, temp);
      ArrayCopy(data, temp, cursor);
      ArrayResize(temp, 0);
      cursor += metadata.lengths.m;

      StringToCharArray(url, temp);
      ArrayCopy(data, temp, cursor);
      ArrayResize(temp, 0);
      cursor += metadata.lengths.u;

      StringToCharArray(headers, temp);
      ArrayCopy(data, temp, cursor);
      ArrayResize(temp, 0);
      cursor += metadata.lengths.h;

      int2chars.x = timeout;
      ArrayCopy(data, int2chars.b, cursor);
      cursor += metadata.lengths.t;

      ArrayCopy(data, body, cursor);

      // store the array in the resource
      resource = data;
    }

    ...
```

First, the packRequest method writes the sizes of all fields to the 'metadata' structure. Then the contents of this structure is copied to the beginning of the 'data' array in the form of an array of bytes. The 'data' array is subsequently placed to the resource. The 'data' array size is reserved based on the total length of all fields and the size of the structure with meta data. String type parameters are converted into arrays using StringToCharArray and copied to the resulting array with a corresponding shift, which is kept up to date in the 'cursor' variable. The 'timeout' parameter is converted into a symbol array using the int2chars union. The 'body' parameter is copied to the array "as is" since it is already an array of the required type. Finally, moving the contents of the common array into the resource is performed in a string (as you may remember, '=' operator is overloaded in the RESOURCEDATA class):

```
      resource = data;
```

The reverse operation of retrieving request parameters from the resource is performed in the unpackRequest method.

```
    void unpackRequest(string &method, string &url, string &headers, int &timeout, uchar &body[])
    {
      uchar array[];
      // fill array with data from resource
      int n = resource.Get(array);
      Print(ChartID(), ": Got ", n, " bytes in request");

      // read metadata from the array
      ArrayCopy(metadata.sizes, array, 0, 0, LEADSIZE);
      int cursor = LEADSIZE;

      // read all data fields, one by one
      method = CharArrayToString(array, cursor, metadata.lengths.m);
      cursor += metadata.lengths.m;
      url = CharArrayToString(array, cursor, metadata.lengths.u);
      cursor += metadata.lengths.u;
      headers = CharArrayToString(array, cursor, metadata.lengths.h);
      cursor += metadata.lengths.h;

      ArrayCopy(int2chars.b, array, 0, cursor, metadata.lengths.t);
      timeout = int2chars.x;
      cursor += metadata.lengths.t;

      if(metadata.lengths.b > 0)
      {
        ArrayCopy(body, array, 0, cursor, metadata.lengths.b);
      }
    }

    ...
```

Here the main work is performed by the string calling resource.Get(array). Then, the meta data bytes, as well as all subsequent fields based on them, are read from the 'array' step by step.

Request execution results are packed and unpacked a similar way using the packResponse and unpackResponse methods (the full code is attached below).

```
    void packResponse(const string source, const uchar &result[], const string &result_headers);
    void unpackResponse(uchar &initiator[], uchar &headers[], uchar &text[]);
```

Now we can go back to the ClientWebWorker source code and complete the 'request' and 'receiveResult' methods.

```
class ClientWebWorker : public WebWorker
{
    ...

    bool request(const string method, const string url, const string headers, const int timeout, const uchar &body[], const long managerChartID)
    {
      _method = method;
      _url = url;

      ResourceMediator mediator(allocate());
      mediator.packRequest(method, url, headers, timeout, body);

      busy = EventChartCustom(managerChartID, 0 /* TODO: specific message */, chartID, 0.0, getFullName());
      return busy;
    }

    static void receiveResult(const string resname, uchar &initiator[], uchar &headers[], uchar &text[])
    {
      Print(ChartID(), ": Reading result ", resname);
      const RESOURCEDATA<uchar> resource(resname);
      ResourceMediator mediator(&resource);
      mediator.unpackResponse(initiator, headers, text);
    }
};
```

They are quite simple due to the ResourceMediator class taking over all the routine work.

The remaining questions are who and when calls the WebWorker methods, as well as how we can get the values of some utility parameters, such as managerChartID, in the 'request' method. Although I am slightly running ahead, I recommend allocating the management of all WebWorker classes objects to more high-level classes that would support actual object lists and exchange messages between programs "on behalf" of the objects including the manager search messages. But before we move to this new level, it is necessary to complete a similar preparation for the "server" part.

### 2.3. Base classes (continued)

Let's declare the custom derivative from WebWorker to handle asynchronous requests on the "server" (manager) side, just like the ClientWebWorker class does that on the client side.

```
class ServerWebWorker : public WebWorker
{
  public:
    ServerWebWorker(const long id, const string p = "WRP_"): WebWorker(id, p)
    {
    }

    bool transfer(const string resname, const long clientChartID)
    {
      // respond to the client with `clientChartID` that the task in `resname` was accepted
      // and pass the task to this specific worker identified by `chartID`
      busy = EventChartCustom(clientChartID, TO_MSG(MSG_ACCEPTED), chartID, 0.0, resname)
          && EventChartCustom(chartID, TO_MSG(MSG_WEB), clientChartID, 0.0, resname);
      return busy;
    }

    void receive(const string source, const uchar &result[], const string &result_headers)
    {
      ResourceMediator mediator(allocate());
      mediator.packResponse(source, result, result_headers);
    }
};
```

The 'transfer' method delegates handling a request to a certain instance of an assistant EA according to step 2 in the overall interaction sequence. The resname parameter is a resource name obtained from a client, while clientChartID is a client window ID. We obtain all these parameters from custom events. The custom events themselves, including MSG\_WEB, are described below.

The 'receive' method creates a local resource in the WebWorker current object ('allocate' call) and writes the name of an original request initiator resource there, as well as data obtained from the Internet (result) and HTTP headers (result\_headers) using the 'mediator' object of the ResourceMediator class. This is a part of step 5 of the overall sequence.

So, we have defined the WebWorker classes for both the client and the "server". In both cases, these objects will most likely be created in large quantities. For example, one client can download several documents at once, while on the manager’s side, it is initially desirable to distribute a sufficient number of assistants, since requests may come from many clients simultaneously. Let's define the WebWorkersPool base class for arranging the object array. Let's make it a template, because the type of stored objects will differ on the client and on the “server” (ClientWebWorker and ServerWebWorker, respectively).

```
template<typename T>
class WebWorkersPool
{
  protected:
    T *workers[];

  public:
    WebWorkersPool() {}

    WebWorkersPool(const uint size)
    {
      // allocate workers; in clients they are used to store request parameters in resources
      ArrayResize(workers, size);
      for(int i = 0; i < ArraySize(workers); i++)
      {
        workers[i] = NULL;
      }
    }

    ~WebWorkersPool()
    {
      for(int i = 0; i < ArraySize(workers); i++)
      {
        if(CheckPointer(workers[i]) == POINTER_DYNAMIC) delete workers[i];
      }
    }

    int size() const
    {
      return ArraySize(workers);
    }

    void operator<<(T *worker)
    {
      const int n = ArraySize(workers);
      ArrayResize(workers, n + 1);
      workers[n] = worker;
    }

    T *findWorker(const string resname) const
    {
      for(int i = 0; i < ArraySize(workers); i++)
      {
        if(workers[i] != NULL)
        {
          if(workers[i].getFullName() == resname)
          {
            return workers[i];
          }
        }
      }
      return NULL;
    }

    T *getIdleWorker() const
    {
      for(int i = 0; i < ArraySize(workers); i++)
      {
        if(workers[i] != NULL)
        {
          if(ChartPeriod(workers[i].getChartID()) > 0) // check if exist
          {
            if(!workers[i].isBusy())
            {
              return workers[i];
            }
          }
        }
      }
      return NULL;
    }

    T *findWorker(const long id) const
    {
      for(int i = 0; i < ArraySize(workers); i++)
      {
        if(workers[i] != NULL)
        {
          if(workers[i].getChartID() == id)
          {
            return workers[i];
          }
        }
      }
      return NULL;
    }

    bool revoke(const long id)
    {
      for(int i = 0; i < ArraySize(workers); i++)
      {
        if(workers[i] != NULL)
        {
          if(workers[i].getChartID() == id)
          {
            if(CheckPointer(workers[i]) == POINTER_DYNAMIC) delete workers[i];
            workers[i] = NULL;
            return true;
          }
        }
      }
      return false;
    }

    int available() const
    {
      int count = 0;
      for(int i = 0; i < ArraySize(workers); i++)
      {
        if(workers[i] != NULL)
        {
          count++;
        }
      }
      return count;
    }

    T *operator[](int i) const
    {
      return workers[i];
    }

};
```

The idea behind the methods is simple. The constructor and destructor allocate and free the array of specified size handlers. The group of findWorker and getIdleWorker methods searches for objects in the array by various criteria. The 'operator<<' operator allows adding objects dynamically, while the 'revoke' method allows removing them dynamically.

The pool of handlers on the client side should have some specificity (in particular, with regard to event handling). Therefore, we extend the base class using the derived ClientWebWorkersPool one.

```
template<typename T>
class ClientWebWorkersPool: public WebWorkersPool<T>
{
  protected:
    long   managerChartID;
    short  managerPoolSize;
    string name;

  public:
    ClientWebWorkersPool(const uint size, const string prefix): WebWorkersPool(size)
    {
      name = prefix;
      // try to find WebRequest manager chart
      WebWorker::broadcastEvent(TO_MSG(MSG_DISCOVER), ChartID());
    }

    bool WebRequestAsync(const string method, const string url, const string headers, int timeout, const char &data[])
    {
      T *worker = getIdleWorker();
      if(worker != NULL)
      {
        return worker.request(method, url, headers, timeout, data, managerChartID);
      }
      return false;
    }

    void onChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
    {
      if(MSG(id) == MSG_DONE) // async request is completed with result or error
      {
        Print(ChartID(), ": Result code ", (long)dparam);

        if(sparam != NULL)
        {
          // read data from the resource with name in sparam
          uchar initiator[], headers[], text[];
          ClientWebWorker::receiveResult(sparam, initiator, headers, text);
          string resname = CharArrayToString(initiator);

          T *worker = findWorker(resname);
          if(worker != NULL)
          {
            worker.onResult((long)dparam, headers, text);
            worker.release();
          }
        }
      }

      ...

      else
      if(MSG(id) == MSG_HELLO) // manager is found as a result of MSG_DISCOVER broadcast
      {
        if(managerChartID == 0 && lparam != 0)
        {
          if(ChartPeriod(lparam) > 0)
          {
            managerChartID = lparam;
            managerPoolSize = (short)dparam;
            for(int i = 0; i < ArraySize(workers); i++)
            {
              workers[i] = new T(ChartID(), name + (string)(i + 1) + "_");
            }
          }
        }
      }
    }

    bool isManagerBound() const
    {
      return managerChartID != 0;
    }
};
```

The variables:

- managerChartID — ID of a window where the working manager is found;
- managerPoolSize — initial size of the handler object array;
- name — common prefix for resources in all pool objects.

### 2.4. Exchanging messages

In the ClientWebWorkersPool constructor, we see the call of WebWorker::broadcastEvent(TO\_MSG(MSG\_DISCOVER), ChartID()) that sends the MSG\_DISCOVER event to all windows passing the ID of the current window in the event parameter. MSG\_DISCOVER is a reserved value: it should be defined at the beginning of the same header file together with other types of messages the programs are to exchange.

```
#define MSG_DEINIT   1 // tear down (manager <-> worker)
#define MSG_WEB      2 // start request (client -> manager -> worker)
#define MSG_DONE     3 // request is completed (worker -> client, worker -> manager)
#define MSG_ERROR    4 // request has failed (manager -> client, worker -> client)
#define MSG_DISCOVER 5 // find the manager (client -> manager)
#define MSG_ACCEPTED 6 // request is in progress (manager -> client)
#define MSG_HELLO    7 // the manager is found (manager -> client)
```

The comments mark the direction a message is sent in.

The TO\_MSG macro is designed for transforming the listed IDs into real event codes relative to a random user-selected base value. We will receive it via the MessageBroadcast input.

```
sinput uint MessageBroadcast = 1;

#define TO_MSG(X) ((ushort)(MessageBroadcast + X))
```

This approach allows moving all events into any free range by changing the base value. Note that custom events can be used in the terminal by other programs as well. Therefore, it is important to avoid collisions.

The MessageBroadcast input will appear in all of our MQL programs featuring the multiweb.mqh file, i.e. in the clients and in the manager. Specify the same MessageBroadcast value when launching the manager and the clients.

Let's get back to the ClientWebWorkersPool class. The onChartEvent method takes a special place. It is to be called from the standard [OnChartEvent](https://www.mql5.com/en/docs/event_handlers/onchartevent) event handler. An event type is passed in the 'id' parameter. Since we receive codes from the system based on the selected base value, we should use the "mirrored" MSG macro to convert it back into the MSG\_\*\*\* range:

```
#define MSG(x) (x - MessageBroadcast - CHARTEVENT_CUSTOM)
```

Here CHARTEVENT\_CUSTOM is a beginning of the range for all custom events in the terminal.

As we can see, the onChartEvent method in ClientWebWorkersPool handles some of the messages mentioned above. For example, the manager should respond with the message MSG\_HELLO to the MSG\_DISCOVER bulk messaging. In this case, the manager window ID is passed in the lparam parameter, while the number of available assistants is passed in the dparam parameter. When the manager is detected, the pool fills the empty 'workers' array with real objects of the required type. The current window ID, as well as the unique resource name in each object is passed to the object constructor. The latter consists of the common 'name' prefix and the serial number in the array.

After the managerChartID field receives a meaningful value, it becomes possible to send requests to the manager. The 'request' method is reserved for that in the ClientWebWorker class, while its usage is demonstrated in the WebRequestAsync method from the pool. First, WebRequestAsync finds a free handler object using getIdleWorker and then calls worker.request(method, url, headers, timeout, data, managerChartID) for it. Inside the 'request' method, we have a comment regarding the selection of a special message code for sending an event. Now, after considering the event subsystem, we can form the final version of the ClientWebWorker::request method:

```
class ClientWebWorker : public WebWorker
{
    ...

    bool request(const string method, const string url, const string headers, const int timeout, const uchar &body[], const long managerChartID)
    {
      _method = method;
      _url = url;

      ResourceMediator mediator(allocate());
      mediator.packRequest(method, url, headers, timeout, body);

      busy = EventChartCustom(managerChartID, TO_MSG(MSG_WEB), chartID, 0.0, getFullName());
      return busy;
    }

    ...
};
```

MSG\_WEB is a message about executing a web request. After receiving it, the manager should find a free assistant EA and pass the client resource name (sparam) to it with the request parameters, as well as the chartID (lparam) client window ID.

The assistant executes the request and returns results to the client using the MSG\_DONE event (if successful) or an error code using MSG\_ERROR (in case of problems). The result (or error) code is passed to dparam, while the result itself is packed into a resource located in the assistant EA under the name passed to sparam. In the MSG\_DONE branch, we see how the data is retrieved from the resource by calling the previously considered ClientWebWorker::receiveResult(sparam, initiator, headers, text) function. Then, the search for the client handler object (findWorker) is performed by the request initiator resource name and a couple of methods are called on a detected object:

```
    T *worker = findWorker(resname);
    if(worker != NULL)
    {
      worker.onResult((long)dparam, headers, text);
      worker.release();
    }
```

We already know the 'release' method — it releases the resource that is not needed already. What is onResult? If we look at the full source code, we will see that the ClientWebWorker class features two virtual functions without implementation: onResult and onError. This makes the class abstract. The client code should describe its derived class from ClientWebWorker and provide implementation. The names of the methods imply that onResult is called if the results are successfully received, while onError is called in case of an error. This provides feedback between the working classes of asynchronous requests and the client program code that uses them. In other words, the client program does not need to know anything about the messages the kernel uses internally: all interactions of the client code with the developed API are performed by the MQL5 OOP built-in tools.

Let's look at the client source code (multiwebclient.mq5).

### 2.5. Client EA

The test EA is to send several requests via multiweb API based on data entered by a user. To achieve this, we need to include the header file and add the inputs:

```
sinput string Method = "GET";
sinput string URL = "https://google.com/,https://ya.ru,https://www.startpage.com/";
sinput string Headers = "User-Agent: n/a";
sinput int Timeout = 5000;

#include <multiweb.mqh>
```

Ultimately, all parameters are intended for configuring performed HTTP requests. In the URL list, we can list several comma-separated addresses in order to evaluate the parallelism and speed of request execution. The URL parameter is divided into addresses using the StringSplit function in OnInit, like this:

```
int urlsnum;
string urls[];

void OnInit()
{
  // get URLs for test requests
  urlsnum = StringSplit(URL, ',', urls);
  ...
}
```

Besides, we need to create a pool of request handler objects (ClientWebWorkersPool) in OnInit. But in order to do this, we need to describe our class derived from ClientWebWorker.

```
class MyClientWebWorker : public ClientWebWorker
{
  public:
    MyClientWebWorker(const long id, const string p = "WRP_"): ClientWebWorker(id, p)
    {
    }

    virtual void onResult(const long code, const uchar &headers[], const uchar &text[]) override
    {
      Print(getMethod(), " ", getURL(), "\nReceived ", ArraySize(headers), " bytes in header, ", ArraySize(text), " bytes in document");
      // uncommenting this leads to potentially bulky logs
      // Print(CharArrayToString(headers));
      // Print(CharArrayToString(text));
    }

    virtual void onError(const long code) override
    {
      Print("WebRequest error code ", code);
    }
};
```

Its only objective is to log status and obtained data. Now we can create a pool of such objects in OnInit.

```
ClientWebWorkersPool<MyClientWebWorker> *pool = NULL;

void OnInit()
{
  ...
  pool = new ClientWebWorkersPool<MyClientWebWorker>(urlsnum, _Symbol + "_" + EnumToString(_Period) + "_");
  Comment("Click the chart to start downloads");
}
```

As you can see, the pool is parametrized by the MyClientWebWorker class which makes it possible to create our objects from the library code. The array size is selected equal to the number of entered addresses. This is reasonable for demonstration purposes: a smaller number would mean a processing queue and discrediting the idea of parallel execution, while a larger number would be a waste of resources. In real projects, the pool size does not have to be equal to the number of tasks, but this requires additional algorithmic binding.

The prefix for resources is set as a combination of the name of the working symbol and the chart period.

The final touch on initialization is searching for the manager window. As you remember, the search is performed by the pool itself (the ClientWebWorkersPool class). The client code only needs to make sure that the manager is found. For these purposes, let us set some reasonable time, within which the message about the search manager and the "response" should be guaranteed to achieve the goals. Let it be 5 seconds. Create a timer for this time:

```
void OnInit()
{
  ...
  // wait for manager negotiation for 5 seconds maximum
  EventSetTimer(5);
}
```

Check if the manager is present in the timer handler. Display an alert if connection is not established.

```
void OnTimer()
{
  // if the manager did not respond during 5 seconds, it seems missing
  EventKillTimer();
  if(!pool.isManagerBound())
  {
    Alert("WebRequest Pool Manager (multiweb) is not running");
  }
}
```

Do not forget to remove the pool object in the OnDeinit handler.

```
void OnDeinit(const int reason)
{
  delete pool;
  Comment("");
}
```

To let the pool handle all service messages without our involvement, including, first of all, searching for the manager, use the standard OnChartEvent chart event handler:

```
void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
{
  if(id == CHARTEVENT_CLICK) // initiate test requests by simple user action
  {
    ...
  }
  else
  {
    // this handler manages all important messaging behind the scene
    pool.onChartEvent(id, lparam, dparam, sparam);
  }
}
```

All events, except for CHARTEVENT\_CLICK, are sent to the pool where the appropriate actions are performed based on the analysis of the applied events' codes (the onChartEvent fragment was provided above).

The CHARTEVENT\_CLICK event is interactive and is used directly to launch the download. In the simplest case, it may look as follows:

```
void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
{
  if(id == CHARTEVENT_CLICK) // initiate test requests by simple user action
  {
    if(pool.isManagerBound())
    {
      uchar Body[];

      for(int i = 0; i < urlsnum; i++)
      {
        pool.WebRequestAsync(Method, urls[i], Headers, Timeout, Body);
      }
    }
    ...
```

The full code of the example is a bit lengthier since it also features the logic for calculating the execution time and comparing it with a sequential call of the standard WebRequest for the same set of addresses.

### 2.6. Manager EA and assistant EA

We have finally reached the "server" part. Since the basic mechanisms have already been implemented inthe header file, the code of managers and assistants is not as cumbersome as one might imagine.

As you may remember, we have only one EA working as a manager or as an assistant (the multiweb.mq5 file). As in the case of the client, we include the header file and declare the input parameters:

```
sinput uint WebRequestPoolSize = 3;
sinput ulong ManagerChartID = 0;

#include <multiweb.mqh>
```

WebRequestPoolSize is a number of auxiliary windows the manager should create to launch assistants on them.

ManagerChartID is a manager window ID. This parameter is usable only as an assistant and is filled with the manager when assistants are launched from the source code automatically. Filling ManagerChartID manually when launching the manager is treated as an error.

The algorithm is built around two global variables:

```
bool manager;
WebWorkersPool<ServerWebWorker> pool;
```

The 'manager' logical flag indicates the role of the current EA instance. The 'pool' variable is an array of handler objects of incoming tasks. WebWorkersPool is typified by the ServerWebWorker class described above. The array is not initialized in advance because its filling depends on the role.

The first launched instance (defined in OnInit) receives the manager role.

```
const string GVTEMP = "WRP_GV_TEMP";

int OnInit()
{
  manager = false;

  if(!GlobalVariableCheck(GVTEMP))
  {
    // when first instance of multiweb is started, it's treated as manager
    // the global variable is a flag that the manager is present
    if(!GlobalVariableTemp(GVTEMP))
    {
      FAILED(GlobalVariableTemp);
      return INIT_FAILED;
    }

    manager = true;
    GlobalVariableSet(GVTEMP, 1);
    Print("WebRequest Pool Manager started in ", ChartID());
  }
  else
  {
    // all next instances of multiweb are workers/helpers
    Print("WebRequest Worker started in ", ChartID(), "; manager in ", ManagerChartID);
  }

  // use the timer for delayed instantiation of workers
  EventSetTimer(1);
  return INIT_SUCCEEDED;
}
```

The EA checks the presence of a special global variable of the terminal. If it is absent, the EA assigns itself the manager and creates such a global variable. If the variable is already present, then so is the manager, and therefore this instance becomes an assistant. Please note that the global variable is temporary, which means it is not saved when the terminal is restarted. But if the manager is left on any chart, it creates the variable again.

The timer is then set to one second, since initialization of auxiliary charts usually takes a couple of seconds and doing it from OnInit is not the best solution. Fill in the pool in the timer event handler:

```
void OnTimer()
{
  EventKillTimer();
  if(manager)
  {
    if(!instantiateWorkers())
    {
      Alert("Workers not initialized");
    }
    else
    {
      Comment("WebRequest Pool Manager ", ChartID(), "\nWorkers available: ", pool.available());
    }
  }
  else // worker
  {
    // this is used as a host of resource storing response headers and data
    pool << new ServerWebWorker(ChartID(), "WRR_");
  }
}
```

In case of an assistant role, yet another ServerWebWorker handler object is simply added to the array. The manager case is more complicated and is arranged in the separate instantiateWorkers function. Let's have a look at it.

```
bool instantiateWorkers()
{
  MqlParam Params[4];

  const string path = MQLInfoString(MQL_PROGRAM_PATH);
  const string experts = "\\MQL5\\";
  const int pos = StringFind(path, experts);

  // start itself again (in another role as helper EA)
  Params[0].string_value = StringSubstr(path, pos + StringLen(experts));

  Params[1].type = TYPE_UINT;
  Params[1].integer_value = 1; // 1 worker inside new helper EA instance for returning results to the manager or client

  Params[2].type = TYPE_LONG;
  Params[2].integer_value = ChartID(); // this chart is the manager

  Params[3].type = TYPE_UINT;
  Params[3].integer_value = MessageBroadcast; // use the same custom event base number

  for(uint i = 0; i < WebRequestPoolSize; ++i)
  {
    long chart = ChartOpen(_Symbol, _Period);
    if(chart == 0)
    {
      FAILED(ChartOpen);
      return false;
    }
    if(!EXPERT::Run(chart, Params))
    {
      FAILED(EXPERT::Run);
      return false;
    }
    pool << new ServerWebWorker(chart);
  }
  return true;
}
```

This function uses the [Expert](https://www.mql5.com/en/code/19003) third-party library developed by our old friend - member of MQL5 community [fxsaber](https://www.mql5.com/en/users/fxsaber), therefore a corresponding header file has been added at the beginning of the source code.

```
#include <fxsaber\Expert.mqh>
```

The Expert library allows you to dynamically generate tpl templates with specified EAs' parameters and apply them to specified charts, which leads to the launch of EAs. In our case, the parameters of all assistant EAs are the same, so their list is generated once before creating a specified number of windows.

The parameter 0 specifies the path to the executable EA file, i.e. to itself. Parameter 1 is WebRequestPoolSize. It is equal to 1 at each assistant. As I have already mentioned, the handler object is needed in the assistant only for storing a resource with HTTP request results. Each assistant handles the request by a blocking WebRequest, i.e. only one handler object is used at most. Parameter 2 — ManagerChartID manager window ID. Parameter 3 — basic value for message codes (MessageBroadcast parameter is taken from multiweb.mqh).

Further on, empty charts are created in the loop with the help of ChartOpen and assistant EAs are launched in them using EXPERT::Run (chart, Params). The ServerWebWorker(chart) handler object is created for each new window and added to the pool. In the manager, the handler objects are nothing more than links to assistants' window IDs and their status, since HTTP requests are not executed in the manager itself and no resources are created for them.

Incoming tasks are handled based on user events in OnChartEvent.

```
void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
{
  if(MSG(id) == MSG_DISCOVER) // a worker EA on new client chart is initialized and wants to bind to this manager
  {
    if(manager && (lparam != 0))
    {
      // only manager responds with its chart ID, lparam is the client chart ID
      EventChartCustom(lparam, TO_MSG(MSG_HELLO), ChartID(), pool.available(), NULL);
    }
  }
  else
  if(MSG(id) == MSG_WEB) // a client has requested a web download
  {
    if(lparam != 0)
    {
      if(manager)
      {
        // the manager delegates the work to an idle worker
        // lparam is the client chart ID, sparam is the client resource
        if(!transfer(lparam, sparam))
        {
          EventChartCustom(lparam, TO_MSG(MSG_ERROR), ERROR_NO_IDLE_WORKER, 0.0, sparam);
        }
      }
      else
      {
        // the worker does actually process the web request
        startWebRequest(lparam, sparam);
      }
    }
  }
  else
  if(MSG(id) == MSG_DONE) // a worker identified by chart ID in lparam has finished its job
  {
    WebWorker *worker = pool.findWorker(lparam);
    if(worker != NULL)
    {
      // here we're in the manager, and the pool hold stub workers without resources
      // so this release is intended solely to clean up busy state
      worker.release();
    }
  }
}
```

First of all, as a response to MSG\_DISCOVER obtained from the client with the lparam ID, the manager returns the MSG\_HELLO message containing its window ID.

Upon receiving MSG\_WEB, lparam should contain the window ID of the client that sent the request, while sparam should contain a name of the resource with packed request parameters. Working as the manager, the code tries to pass the task with these parameters to an idle assistant by calling the 'transfer' function (described below) and thereby change the status of the selected object to "busy". If there are no idle assistants, the MSG\_ERROR event is sent to the client with the ERROR\_NO\_IDLE\_WORKER code. Assistant executes the HTTP request in the startWebRequest function.

The MSG\_DONE event arrives to the manager from the assistant when the latter uploads the requested document. The manager finds the appropriate object by assistant ID in lparam and disables its "busy" status by calling the 'release' method. As already mentioned, the assistant sends the results of its operation directly to the client.

The full source code also contains the MSG\_DEINIT event closely related to OnDeinit handling. The idea is that the assistants are notified of the manager removal and, in response, unload themselves and close their window, while the manager is notified of the removal of the assistant and deletes it from the manager's pool. I believe, you can get an understanding of this mechanism on your own.

The 'transfer' function searches for a free object and calls its 'transfer' method (discussed above).

```
bool transfer(const long returnChartID, const string resname)
{
  ServerWebWorker *worker = pool.getIdleWorker();
  if(worker == NULL)
  {
    return false;
  }
  return worker.transfer(resname, returnChartID);
}
```

The startWebRequest function is defined as follows:

```
void startWebRequest(const long returnChartID, const string resname)
{
  const RESOURCEDATA<uchar> resource(resname);
  ResourceMediator mediator(&resource);

  string method, url, headers;
  int timeout;
  uchar body[];

  mediator.unpackRequest(method, url, headers, timeout, body);

  char result[];
  string result_headers;

  int code = WebRequest(method, url, headers, timeout, body, result, result_headers);
  if(code != -1)
  {
    // create resource with results to pass back to the client via custom event
    ((ServerWebWorker *)pool[0]).receive(resname, result, result_headers);
    // first, send MSG_DONE to the client with resulting resource
    EventChartCustom(returnChartID, TO_MSG(MSG_DONE), ChartID(), (double)code, pool[0].getFullName());
    // second, send MSG_DONE to the manager to set corresponding worker to idle state
    EventChartCustom(ManagerChartID, TO_MSG(MSG_DONE), ChartID(), (double)code, NULL);
  }
  else
  {
    // error code in dparam
    EventChartCustom(returnChartID, TO_MSG(MSG_ERROR), ERROR_MQL_WEB_REQUEST, (double)GetLastError(), resname);
    EventChartCustom(ManagerChartID, TO_MSG(MSG_DONE), ChartID(), (double)GetLastError(), NULL);
  }
}
```

By using ResourceMediator, the function unpacks request parameters and calls the standard MQL WebRequest function. If the function is executed without MQL errors, the results are sent to the client. To do that, they are packed into a local resource using the 'receive' method (described above), and its name is passed with the MSG\_DONE message in the sparam parameter of the EventChartCustom function. Note that HTTP errors (for example, invalid page 404 or web server error 501) fall here as well — the client receives the HTTP code in the dparam parameter and response HTTP headers in the resource allowing you to analyze the situation.

If WebRequest call ends with an MQL error, the client receives MSG\_ERROR message with the ERROR\_MQL\_WEB\_REQUEST code, while GetLastError result is placed to dparam. Since the local resource is not filled in this case, the name of a source resource is passed directly in the sparam parameter, so that a certain instance of a handler object with a resource can still be identified on the client side.

[![Diagram of the multiweb library classes for asynchronous and parallel WebRequest call](https://c.mql5.com/2/34/multiweb3__1.png)](https://c.mql5.com/2/34/multiweb3.png "https://c.mql5.com/2/34/multiweb3.png")

**Fig. 4. Diagram of the multiweb library classes for asynchronous and parallel WebRequest call**

### 3\. Testing

Testing the implemented software complex can be performed as follows.

First, open the terminal settings and specify all servers to be accessed in the list of allowed URLs on the Experts tab.

Next, launch the multiweb EA and set 3 assistants in the inputs. As a result, 3 new windows are opened featuring the same multiweb EA launched in a different role. The EA role is displayed in the comment in the upper left corner of the window.

Now, let's launch the multiwebclient client EA on another chart and click on the chart once. With the default settings, it initiates 3 parallel web requests and writes diagnostics to the log, including the size of the obtained data and the running time. If the TestSyncRequests special parameter is left 'true', sequential requests of the same pages are executed using the standard WebRequest in addition to parallel web requests via the manager. This is done to compare execution speeds of the two options. As a rule, the parallel processing is several times faster than the sequential one - from sqrt(N) to N, where N is a number of available assistants.

The sample log is displayed below.

```
01:16:50.587    multiweb (EURUSD,H1)    OnInit 129912254742671339
01:16:50.587    multiweb (EURUSD,H1)    WebRequest Pool Manager started in 129912254742671339
01:16:52.345    multiweb (EURUSD,H1)    OnInit 129912254742671345
01:16:52.345    multiweb (EURUSD,H1)    WebRequest Worker started in 129912254742671345; manager in 129912254742671339
01:16:52.757    multiweb (EURUSD,H1)    OnInit 129912254742671346
01:16:52.757    multiweb (EURUSD,H1)    WebRequest Worker started in 129912254742671346; manager in 129912254742671339
01:16:53.247    multiweb (EURUSD,H1)    OnInit 129912254742671347
01:16:53.247    multiweb (EURUSD,H1)    WebRequest Worker started in 129912254742671347; manager in 129912254742671339
01:17:16.029    multiweb (EURUSD,H1)    Pool manager transfers \Experts\multiwebclient.ex5::GBPJPY_PERIOD_M5_1_129560567193673862
01:17:16.029    multiweb (EURUSD,H1)    129912254742671345: Reading request \Experts\multiwebclient.ex5::GBPJPY_PERIOD_M5_1_129560567193673862
01:17:16.029    multiweb (EURUSD,H1)    129912254742671345: Got 64 bytes in request
01:17:16.029    multiweb (EURUSD,H1)    129912254742671345: GET https://google.com/ User-Agent: n/a 5000
01:17:16.030    multiweb (EURUSD,H1)    Pool manager transfers \Experts\multiwebclient.ex5::GBPJPY_PERIOD_M5_2_129560567193673862
01:17:16.030    multiweb (EURUSD,H1)    129912254742671346: Reading request \Experts\multiwebclient.ex5::GBPJPY_PERIOD_M5_2_129560567193673862
01:17:16.030    multiwebclient (GBPJPY,M5)      Accepted: \Experts\multiwebclient.ex5::GBPJPY_PERIOD_M5_1_129560567193673862 after 0 retries
01:17:16.031    multiweb (EURUSD,H1)    129912254742671346: Got 60 bytes in request
01:17:16.031    multiweb (EURUSD,H1)    129912254742671346: GET https://ya.ru User-Agent: n/a 5000
01:17:16.031    multiweb (EURUSD,H1)    Pool manager transfers \Experts\multiwebclient.ex5::GBPJPY_PERIOD_M5_3_129560567193673862
01:17:16.031    multiwebclient (GBPJPY,M5)      Accepted: \Experts\multiwebclient.ex5::GBPJPY_PERIOD_M5_2_129560567193673862 after 0 retries
01:17:16.031    multiwebclient (GBPJPY,M5)      Accepted: \Experts\multiwebclient.ex5::GBPJPY_PERIOD_M5_3_129560567193673862 after 0 retries
01:17:16.031    multiweb (EURUSD,H1)    129912254742671347: Reading request \Experts\multiwebclient.ex5::GBPJPY_PERIOD_M5_3_129560567193673862
01:17:16.032    multiweb (EURUSD,H1)    129912254742671347: Got 72 bytes in request
01:17:16.032    multiweb (EURUSD,H1)    129912254742671347: GET https://www.startpage.com/ User-Agent: n/a 5000
01:17:16.296    multiwebclient (GBPJPY,M5)      129560567193673862: Result code 200
01:17:16.296    multiweb (EURUSD,H1)    Result code from 129912254742671346: 200, now idle
01:17:16.297    multiweb (EURUSD,H1)    129912254742671346: Done in 265ms
01:17:16.297    multiwebclient (GBPJPY,M5)      129560567193673862: Reading result \Experts\multiweb.ex5::WRR_129912254742671346
01:17:16.300    multiwebclient (GBPJPY,M5)      129560567193673862: Got 16568 bytes in response
01:17:16.300    multiwebclient (GBPJPY,M5)      GET https://ya.ru
01:17:16.300    multiwebclient (GBPJPY,M5)      Received 3704 bytes in header, 12775 bytes in document
01:17:16.715    multiwebclient (GBPJPY,M5)      129560567193673862: Result code 200
01:17:16.715    multiwebclient (GBPJPY,M5)      129560567193673862: Reading result \Experts\multiweb.ex5::WRR_129912254742671347
01:17:16.715    multiweb (EURUSD,H1)    129912254742671347: Done in 686ms
01:17:16.715    multiweb (EURUSD,H1)    Result code from 129912254742671347: 200, now idle
01:17:16.725    multiwebclient (GBPJPY,M5)      129560567193673862: Got 45236 bytes in response
01:17:16.725    multiwebclient (GBPJPY,M5)      GET https://www.startpage.com/
01:17:16.725    multiwebclient (GBPJPY,M5)      Received 822 bytes in header, 44325 bytes in document
01:17:16.900    multiwebclient (GBPJPY,M5)      129560567193673862: Result code 200
01:17:16.900    multiweb (EURUSD,H1)    Result code from 129912254742671345: 200, now idle
01:17:16.900    multiweb (EURUSD,H1)    129912254742671345: Done in 873ms
01:17:16.900    multiwebclient (GBPJPY,M5)      129560567193673862: Reading result \Experts\multiweb.ex5::WRR_129912254742671345
01:17:16.903    multiwebclient (GBPJPY,M5)      129560567193673862: Got 13628 bytes in response
01:17:16.903    multiwebclient (GBPJPY,M5)      GET https://google.com/
01:17:16.903    multiwebclient (GBPJPY,M5)      Received 790 bytes in header, 12747 bytes in document
01:17:16.903    multiwebclient (GBPJPY,M5)      > > > Async WebRequest workers [3] finished 3 tasks in 873ms
```

Note that the total execution time of all requests is equal to the execution time of the slowest one.

If we set the number of assistants to one in the manager, requests are handled sequentially.

### Conclusion

In this article, we have considered a number of classes and ready-made EAs for executing HTTP requests in non-blocking mode. This allows obtaining data from the Internet in several parallel threads and increasing the efficiency of EAs which, in addition to HTTP requests, should perform analytical calculations in real time. In addition, this library can also be used in indicators where the standard WebRequest is prohibited. To implement the entire architecture, we had to use a wide range of MQL features: passing user events, creating resources, opening windows dynamically and running EAs on them.

At the time of writing, the creation of auxiliary windows for launching assistant EAs is the only option for paralleling HTTP requests, but MetaQuotes plans to develop special background MQL programs. The MQL5/Services folder is already reserved for such services. When this technology appears in the terminal, this library can probably be improved by replacing auxiliary windows with services.

Attached files:

- MQL5/Include/multiweb.mqh — library
- MQL5/Experts/multiweb.mq5 — manager EA and assistant EA
- MQL5/Experts/multiwebclient.mq5 — demo client EA
- MQL5/Include/fxsaber/Resource.mqh — auxiliary class for working with resources
- MQL5/Include/fxsaber/ResourceData.mqh — auxiliary class for working with resources
- MQL5/Include/fxsaber/Expert.mqh — auxiliary class for launching EAs
- MQL5/Include/TypeToBytes.mqh — data conversion library

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/5337](https://www.mql5.com/ru/articles/5337)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/5337.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/5337/mql5.zip "Download MQL5.zip")(17.48 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Backpropagation Neural Networks using MQL5 Matrices](https://www.mql5.com/en/articles/12187)
- [Parallel Particle Swarm Optimization](https://www.mql5.com/en/articles/8321)
- [Custom symbols: Practical basics](https://www.mql5.com/en/articles/8226)
- [Calculating mathematical expressions (Part 2). Pratt and shunting yard parsers](https://www.mql5.com/en/articles/8028)
- [Calculating mathematical expressions (Part 1). Recursive descent parsers](https://www.mql5.com/en/articles/8027)
- [MQL as a Markup Tool for the Graphical Interface of MQL Programs (Part 3). Form Designer](https://www.mql5.com/en/articles/7795)
- [MQL as a Markup Tool for the Graphical Interface of MQL Programs. Part 2](https://www.mql5.com/en/articles/7739)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/298396)**
(62)


![magnomilk](https://c.mql5.com/avatar/avatar_na2.png)

**[magnomilk](https://www.mql5.com/en/users/magnomilk)**
\|
23 Feb 2022 at 11:10

Really nice article.

However I get issues when trying to compile with metatrader 5.

Initialise sequence for array expected:

in template 'const TYPETOBYTES::STRUCT\_TYPE<T> TYPETOBYTES::FillBytes(const uchar)' specified with \[T=uchar\] TypeToBytes.mqh 314 31

I understand that it is an issue initilising the array. I could try to fix it. However, I dont see any report of this issue, wondering whether it is only my self facing the issue.

Thanks for the article anywell, wonderful anyway!

![Stanislav Korotky](https://c.mql5.com/avatar/2010/10/4CA7CFA0-1F0C.jpg)

**[Stanislav Korotky](https://www.mql5.com/en/users/marketeer)**
\|
24 Feb 2022 at 10:49

**magnomilk [#](https://www.mql5.com/pt/forum/300413#comment_27898414):**

Really nice article.

However I get issues when trying to compile with metatrader 5.

Initialise sequence for array expected:

in template 'const TYPETOBYTES::STRUCT\_TYPE<T> TYPETOBYTES::FillBytes(const uchar)' specified with \[T=uchar\] TypeToBytes.mqh 314 31

I understand that it is an issue initilising the array. I could try to fix it. However, I dont see any report of this issue, wondering whether it is only my self facing the issue.

Thanks for the article anywell, wonderful anyway!

Make sure you're using the latest TypeToBytes library.

![Sergei Poliukhov](https://c.mql5.com/avatar/avatar_na2.png)

**[Sergei Poliukhov](https://www.mql5.com/en/users/operlay)**
\|
30 Jul 2024 at 00:21

**Igor K "WebRequest error code 4002"....**

**MetaTrader 5**

**Version: 5.00 build 2093**

**02 Jul 2019**

**===cut here===**

**2019.07.23 00:47:37.182 multiwebclient (USDJPY,H1) Accepted: aQuotes\\Terminal\\D0E8209F77C8CF37AD8BF550E51FF075\\MQL5\\Experts\\multiwebclient.ex5::USDJPY\_PERIOD\_H1\_2\_128968169154443359 after 0 retries**

**2019.07.23 00:47:37.182 multiwebclient (USDJPY,H1) WebRequest error code 4002**

**===cut here===**

Only ports 80 (http) and 443 (https) are allowed for WebRequest.

![Sergei Poliukhov](https://c.mql5.com/avatar/avatar_na2.png)

**[Sergei Poliukhov](https://www.mql5.com/en/users/operlay)**
\|
31 Jul 2024 at 08:50

The file in the example multiwebclient sends only once when clicking on the chart. in case of several clicks the request to the http server is not sent. (I will dig into your sources) Perhaps it is necessary to pre-create not in INIT but each time to create on CLICK on the chart.

I thought that the object is deleted after closing the chart or [closing the terminal](https://www.mql5.com/en/docs/common/terminalclose "MQL5 documentation: TerminalClose function").

Do you have an example to send a json message? Also it is not superfluous for someone to have a check for closing charts.

JSON SOFTWARE

For example, create a json class and fill it with setters. Adding arrays, adding elements, data types (its not simple).

And an example, a working EA, so that you can make sending from your function, for example, when a new bar appears, a JSON request is sent.

Many people would find it useful.

![Stanislav Korotky](https://c.mql5.com/avatar/2010/10/4CA7CFA0-1F0C.jpg)

**[Stanislav Korotky](https://www.mql5.com/en/users/marketeer)**
\|
31 Jul 2024 at 20:27

**Sergei Poliukhov closing the terminal.**
**Do you have an example to send a json message? Also it is not superfluous for someone to have a check for closing charts.**

**JSON SOFTWARE**

**For example, create a json class and fill it with setters. Adding arrays, adding elements, data types (its not simple).**

**And an example, a direct working advisor, so that you can make sending from your function, for example, when a new bar appears, a JSON request is sent.**

**Many people would find it useful.**

All 100% worked as described, at the time of writing the article, i.e. on click 3 requests were sent in parallel to the default 3 servers. You need settings and logs to find problems. Chart closures are processed and a notification is sent to the manager - look at the code.

```
void OnDeinit(const int reason)
{
  if(manager)
  {
    Print("WebRequest Pool Manager closed, ", ChartID());
    for(int i = 0; i < pool.size(); i++)
    {
      if(CheckPointer(pool[i]) == POINTER_DYNAMIC)
      {
        // let workers know they are not needed anymore
        EventChartCustom(pool[i].getChartID(), TO_MSG(MSG_DEINIT), ChartID(), 0.0, NULL);
      }
    }
    GlobalVariableDel(GVTEMP);
  }
  else
  {
    Print("WebRequest Worker closed, ", ChartID());
    // let know the manager that this worker is not available anymore
    EventChartCustom(ManagerChartID, TO_MSG(MSG_DEINIT), ChartID(), 0.0, NULL);
    ObjectDelete(0, OBJ_LABEL_NAME);
    ChartClose(0);
  }
  Comment("");
}
```

After the article was released, an improved version of launching via chart objects was posted in the [discussion](https://www.mql5.com/ru/forum/288985/page3#comment_9362458).

There are a lot of materials on json on the site, but I have not published my work on "pure" json and I don't have time to do it now.

![Using OpenCL to test candlestick patterns](https://c.mql5.com/2/34/OpenCL_for_candle_patterns.png)[Using OpenCL to test candlestick patterns](https://www.mql5.com/en/articles/4236)

The article describes the algorithm for implementing the OpenCL candlestick patterns tester in the "1 minute OHLC" mode. We will also compare its speed with the built-in strategy tester launched in the fast and slow optimization modes.

![Reversing: Formalizing the entry point and developing a manual trading algorithm](https://c.mql5.com/2/34/Reverse_trade.png)[Reversing: Formalizing the entry point and developing a manual trading algorithm](https://www.mql5.com/en/articles/5268)

This is the last article within the series devoted to the Reversing trading strategy. Here we will try to solve the problem, which caused the testing results instability in previous articles. We will also develop and test our own algorithm for manual trading in any market using the reversing strategy.

![Developing the symbol selection and navigation utility in MQL5 and MQL4](https://c.mql5.com/2/34/Select_Symbols_Utility_MQL5.png)[Developing the symbol selection and navigation utility in MQL5 and MQL4](https://www.mql5.com/en/articles/5348)

Experienced traders are well aware of the fact that most time-consuming things in trading are not opening and tracking positions but selecting symbols and looking for entry points. In this article, we will develop an EA simplifying the search for entry points on trading instruments provided by your broker.

![Reversing: Reducing maximum drawdown and testing other markets](https://c.mql5.com/2/34/Graal.png)[Reversing: Reducing maximum drawdown and testing other markets](https://www.mql5.com/en/articles/5111)

In this article, we continue to dwell on reversing techniques. We will try to reduce the maximum balance drawdown till an acceptable level for the instruments considered earlier. We will see if the measures will reduce the profit. We will also check how the reversing method performs on other markets, including stock, commodity, index, ETF and agricultural markets. Attention, the article contains a lot of images!

[![](https://www.mql5.com/ff/sh/0wxx5f0vuwq7xh89z2/01.png)VPS for 24/7 tradingContact your broker and find out how to get a free hosting subscriptionLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=nhetzvgituppcfrhndpblbihmzziogdh&s=d00c975c8bda3d8c1b29f042ad33ac81952ccea2f130a8f1ffa9015bab8ade87&uid=&ref=https://www.mql5.com/en/articles/5337&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5071861815735955418)

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