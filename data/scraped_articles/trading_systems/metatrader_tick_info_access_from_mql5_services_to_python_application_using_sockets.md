---
title: MetaTrader tick info access from MQL5 services to Python application using sockets
url: https://www.mql5.com/en/articles/18680
categories: Trading Systems, Integration
relevance_score: 6
scraped_at: 2026-01-23T11:34:46.527515
---

[![](https://www.mql5.com/ff/si/h2ryn394uwcpxwmxc2.jpg)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Feconomic-calendar%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dopen.calendar%26utm_content%3Deconomic.calendar%26utm_campaign%3Den.0009.desktop.default&a=qdeulxgvibgwytgewnvfatbocjnnninc&s=5c0c60f00ff5f5bedb0fdf65d9d79eb820442eb43ffac2b85aa003224f9dba14&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=btqegimwjdwxyvricmvjcsikbzxbcqxv&ssn=1769157285768703704&ssn_dr=0&ssn_sr=0&fv_date=1769157285&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F18680&back_ref=https%3A%2F%2Fwww.google.com%2F&title=MetaTrader%20tick%20info%20access%20from%20MQL5%20services%20to%20Python%20application%20using%20sockets%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176915728546510369&fz_uniq=5062566385520911524&sv=2552)

MetaTrader 5 / Trading systems


### Introduction

Sometimes everything is not programmable in the MQL5 language. And even if it is possible to convert existing advanced libraries in MQL5, it would be time-consuming. A better option is to integrate or utilize the existing libraries to achieve a task. For example, there are numerous machine-learning-libraries in python. It's not the best option for us to create identical copies in MQL5 form just to achieve a machine learning task for trading. It is better to export the data needed for that python machine learning library, do the necessary process in the python environment and import the result back into the MQL5 program. This article is about transporting such data from the MetaTrader terminal to the python environment.

There is a python package called MetaTrader 5 which can be used to access MetaTrader’s charts data such as bars info, tick info, user info, trade info etc. But this package is only available for Windows, meaning that if you want to use the package, then you need to have Windows OS. In this article, I am going to show you how you can access tick data from MetaTrader to a python program using sockets.

With the introduction of Berkeley sockets in the 4.2BSD Unix operating system in 1983, communication between machines became easy and widespread. More advanced application layer libraries and protocols all depend on transport level socket protocols, which we are going to use in this article.

For demo purpose, I am going to transport only tick data (bid, ask and time of tick data) to the python application through socket ports to be utilized in the python application. We can export all kinds of chart data such as bar info, user info, trade info etc. using the idea presented in this article. And we can use not only services but also scripts, indicators or expert advisors to do the task.

### Program Flow

This article focuses on utilizing MetaTrader’s 5 services program to send tick info such as bid, ask and time to a python server and the python server will broadcast the info to all the client sockets which are connected to the server. This is more clear in the following figure.

![Socket data flow](https://c.mql5.com/2/154/Pic-1.png)

As you can see from the figure, the MetaTrader service program is connected to a python server listening on port 9070. All the tick data of the charts that are open in MetaTrader 5 terminal, will be sent to the python server at port 9070. The Python server then analyzes the data received from MetaTrader 5, does the necessary analysis on the data and distributes or better say broadcasts the tick info to the connected clients. The clients can then use the received data to perform necessary tasks or apply algorithms to achieve the desired result, which can be transmitted back to the MetaTrader services program for further processing.

### Why services and python?

There is no specific reason to choose MetaTrader 5 services to send tick information. Scripts, Indicator and Expert Advisor can also be used to do such tasks and I shall do that in my next articles. I just wanted to show that tick and other chart related information can be transported to other applications using MetaTrader services using socket programming. And instead of python server and clients, we can use other platforms and frameworks to develop the server and clients as well. Python was convenient for me.

### Services Program

As we all know that services in MetaTrader 5 have only one function, OnStart and everything has to be done within that function. Services are not tied to any windows and the built-in functions like \_Symbol, \_Period, \_Point and others are inaccessible in services like they can be accessed in scripts, indicators and expert advisors. And running and managing services can only be done through the navigator window. Services cannot be dragged to open chart windows for running. You can see how to run services in the demo section below, where I have included a GIF file.

The services program starts by defining socket, server and port variables.

```
int socket;
string server = "localhost";
int port = 9070;
```

The _SocketInit()_ function creates a socket handle using the _SocketCreate()_ method. The socket handle along with server address and port is passed to _SocketConnect()_ function for connecting to the server, which return true in case connection is successful.

```
void SocketInit() {
   socket=SocketCreate();
   bool connect = SocketConnect(socket, server, port, 1000);
   if(connect) {
      Print("socket is connected", " ", server, " port ", port);
   }
}
```

When we add and run the services program from the navigator window in MetaTrader terminal, _OnStart_ function is called where we call the _SocketInit()_ function.

```
void OnStart() {
   SocketInit();
```

Then three variables are defined, first one to store tick info of opened chart, second to store the chart ID and third to store tick info to be sent to the server.

```
MqlTick latestTick;
long next;
string payload;
```

As we need to constantly send tick info to the server, while loop with true is defined and then check if the server is connected. If the socket connection wasn't successful, then SocketIsConnected(socket handle) would return false and the service is stopped.

```
while(true) {
   if(!SocketIsConnected(socket)){
      Print("socket is not initialized yet so stopping the service");
      break;
   }
```

The next and payload variables are initialized with the first chart ID and empty string.

```
next = ChartFirst();
payload = "";
```

Another while loop for looping all the chart windows and then getting the chart symbol is done next.

```
while (next != -1) {
   string chartSymbol = ChartSymbol(next);
```

Latest tick info such as bid, ask and time for the chart are extracted.

```
SymbolInfoTick(chartSymbol, latestTick);
double bid = latestTick.bid;
double ask = latestTick.ask;
string tickTime = TimeToString(latestTick.time, TIME_SECONDS);
```

A payload is generated to be sent to the server, using symbol, time, bid and ask values. I have tried to create a JSON string value for the payload so that I can use the JSON library in python server to decode the JSON string into a JSON object for data extraction. You can use whatever format is convenient for you.

```
bool stringAdded = StringAdd(payload, StringFormat("{\"pair\": \"%s\", \"time\": \"%s\", \"bid\": %f, \"ask\": %f}", chartSymbol, tickTime, bid, ask));
```

If there are more chart windows open, then to append the tick info of next chart window by using #@# as a separator is done.

```
next = ChartNext(next);
if (next != -1 && stringAdded) {
   stringAdded = StringAdd(payload, "#@#");
}
```

When all the open chart windows are scanned for tick data and the payload is populated, it is ready to be sent to the server. SocketSend method sends the data to the socket port that was defined socket handle. Make sure the value in the length field is correct else it will send extra characters with the data which would be an extra burden on the server side for parsing the data.

```
uchar data[];
int len = StringToCharArray(payload, data);
SocketSend(socket, data, len-1);
```

And the loop is continued restarting from the first chart window open until the connection to the server is open.

### Python Server Program

As described in the program flow, that server should be listening to receive tick information at 9070 from MetaTrader terminal and also listening to clients for sending the received tick information on port 9071. So two socket connections are bound to those ports and kept listening at those ports.

```
host = '127.0.0.1'
MT5_RECIEVING_PORT = 9070
CLIENT_SENDING_PORT = 9071
mt5Socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
mt5Socket.bind((host, MT5_RECIEVING_PORT))
mt5Socket.listen()
clientSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
clientSocket.bind((host, CLIENT_SENDING_PORT))
clientSocket.listen()
```

I programmed the server in such a way that only one MetaTrader 5 is allowed to connect and communicate. This is done to avoid mismanagement and havoc conditions that may arise for multiple MetaTrader 5 connections. On the other hand, clients are to receive the tick data for further processing, the same data can be utilized by different clients for different algorithms to be applied to achieve the desired result. Thus, there is a need to manage the client connections.

The following dict variable is used for storing live connected clients for managing the connected clients on the server side.

```
connectedClients = {}
```

MetaTrader and clients flow are started as separate threads so that they can run independently of each other.

```
if __name__ == "__main__":

   thread = threading.Thread(target=acceptFromMt5)
   thread.start()
   thread = threading.Thread(target=acceptFromClients)
   thread.start()
```

MetaTrader 5 thread continues to wait until MetaTrader service connects to it. Once the connection is accepted, the processMt5Data function is called where data is received, processed and sent to respective clients. I will discuss more on this later.

If there is an error in accepting a connection from MetaTrader service, then it will wait for the next connection.

```
def acceptFromMt5():
   try:
       print("Server is listening for mt5")
       mt5Client, address = mt5Socket.accept()
       print(f"mt5 service is connected at {str(address)}")
       processMt5Data(mt5Client)

   except Exception as ex:
       print(f"error in accepting mt5 client {ex}")
       acceptFromMt5()
```

Before discussing the tick info analysis part, it is crucial to understand how clients are connected and managed. As with the MetaTrader 5 service, the server waits for a new client connection. A client is required to send its identification so that it can be managed on the server. The identification is the symbol pair of the chart window. This identification is used as the key to the dict object, and the client is added to that dictionary. A while loop is used so that many clients can connect.

```
def acceptFromClients():
   try:
       while True:

           print("Server is listening for clients")
           client, address = clientSocket.accept()
           pair = client.recv(7).decode("ascii")
           print(f"{pair} client service is connected at {str(address)}")
           clients = connectedClients[pair] if connectedClients and pair in connectedClients.keys() else []
           clients.append(client)
           connectedClients[pair] = clients

   except Exception as ex:
       print(f"error in accepting other clients {ex}")
       acceptFromClients()
```

Now that we have understood how newly connected clients are added, it would be easy to understand how the tick data received is broadcasted to those clients. Tick data is converted to JSON object and symbol is extracted from it. And tick info is broadcasted to clients based on the clients’ identification key sent to the server while connecting.

Tick data is split using the same separator used in MetaTrader 5 services program. Additional splitting is needed if multiple tick data for a chart is sent all at once. The second splitting was done as the python library for JSON parsing was unable to parse the tick info if multiple tick information is sent all at once from MetaTrader service to the server.

```
def processMt5Data(mt5Client):
   data = "mt5 client connected"
   repeatativeEmpty = 0
   while(len(data) > 0):

       try:
          data = mt5Client.recv(1024000).decode("ascii")

           if len(data) > 0:
               for jsn in data.split("#@#"):

                   if "}{" in jsn:
                       splittedTickData = jsn.split("}{")
                       jsn = splittedTickData[0] + "}"

                   jsonTickData = json.loads(jsn)
                   pair = jsonTickData["pair"]

                   if pair in connectedClients.keys():
                       broadcastToClients(pair, jsonTickData)
           repeatativeEmpty = repeatativeEmpty + 1 if len(data) == 0 else 0

           if repeatativeEmpty > 10:
               print(f"data is not recieved for 10 times in a row {data}")
               break

       except Exception as ex:
           print(f"error in processing mt5 data {ex}")
           break

       time.sleep(0.1)
   acceptFromMt5()
```

Sending data to clients is straightforward. If an error occurs while sending data, then the client is removed from the connected clients list assuming the client is not connected and the dictionary is updated.

```
def broadcastToClients(pair, message):
   for client in connectedClients[pair]:

       try:
           client.send(str(message).encode("ascii"))

       except Exception as ex:
           print(f"error while sending {message} to {client} for {pair}")
           client.close()
           clients = connectedClients[pair]
           clients.remove(client)
           connectedClients[pair] = clients
```

And that completes the server program.

### Python Client Program

The client program connects with the server on port 9071 as already discussed in program flow. And as already discussed above that for proper management of clients on the server, a key representing the symbol/currency pair is needed in the server. Thus, we need options to show available key list to be chosen from the list for identification ID which is to be sent to the server.

```
CURRENCY_PAIRS = [\
\
   "AUDUSD",\
   "AUDJPY",\
   "AUDCAD",\
   "AUDNZD",\
   "AUDCHF",\
   "CADJPY",\
   "CADCHF",\
   "CHFJPY",\
   "EURUSD",\
   "EURJPY",\
   "EURGBP",\
   "EURCAD",\
   "EURAUD",\
   "EURNZD",\
   "EURCHF",\
   "GBPUSD",\
   "GBPJPY",\
   "GBPCAD",\
   "GBPAUD",\
   "GBPNZD",\
   "GBPCHF",\
   "NZDUSD",\
   "NZDJPY",\
   "NZDCAD",\
   "USDCHF",\
   "USDJPY",\
   "USDCAD",\
   "NZDCHF"\
]
server = "127.0.0.1"
port = 9071
```

As you can see, one option from the above list is chosen and sent to the server once connected, and then it waits for data to arrive from the server.

```
if __name__ == "__main__":
   print(f"please choose from the currency pairs given below as name \n {', '.join(CURRENCY_PAIRS)}")
   name = input("enter the name for the client : ")

   if name in CURRENCY_PAIRS:
       client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

       if client.connect_ex((server, port)) == 0:
           client.send(name.encode("ascii"))
           receiveData(client)

       else:
           print("server could not be connected.")

   else:
       print("you didn't choose from the above list")
```

The client program I wrote just prints the tick data, but we can do anything with it according to our requirements, such as further processing of data using available frameworks and libraries, applying machine learning libraries etc.

```
def receiveData(client):
   repeatativeEmpty = 0

   while True:
       data = client.recv(1024).decode("ascii")
       print("data received ", data)
       repeatativeEmpty = repeatativeEmpty + 1 if len(data) == 0 else 0

       if repeatativeEmpty > 10:
           print(f"data is not recieved for 10 times in a row {data}")

           break
```

And that completes the client program.

### Demo of Data Flow

The server needs to be started first for the smooth data transformation, otherwise none of the services would run. For testing purpose, the python server is run on virtual environment, which can be seen in the following figure.

![server started](https://c.mql5.com/2/154/Untitled_drawing-2.png)

We need the tick data to be received by the server and then broadcasted to clients, so the second task is to start the MetaTrader services program, as seen in the following screenshot.

![MetaTrader service started](https://c.mql5.com/2/154/Screenshot_2025-07-02_at_10.15.202PM.png)

If MetaTrader service is not run, then connected clients would wait for data to be received. As soon as a client sends identification keys to the server and if there is tick data sent from MetaTrader services, the data is printed in the client terminal for now. This is done for EURUSD and GBPUSD as both chart windows are open in the MetaTrader 5 terminal at that moment. This is evident in the following screenshots.

![EURUSD client started](https://c.mql5.com/2/154/Screenshot_2025-07-02_at_10.16.575PM.png)

![GBPUSD client started](https://c.mql5.com/2/154/Screenshot_2025-07-02_at_10.20.31xPM.png)

I have tried to capture all of the above screenshotted process in a GIF so that the whole process is more evident.

![tick data transport from MetaTrader to python complete process](https://c.mql5.com/2/154/ScreenRecording2025-07-02at11.26.37PM-ezgif.com-video-to-gif-converter__2.gif)

### Conclusion

This article discusses using socket programming to send tick data from MetaTrader 5 to a python application using MetaTrader services and python server and client. As we can transport tick data as done in this article, we can transport other information about charts as well. I have used services, but we can use scripts, indicators or expert advisors too. And like python, we can use other programming languages too.

Moreover, this article also tries to eradicate the Windows OS dependency. MetaTrader and python library such as MetaTrader 5 all depend on Windows OS. Instead of using the MetaTrader 5 python library, we can use the socket protocol for data transfer from MetaTrader to a more convincing library for easy and advanced processing of MetaTrader data.

| File Name | Description |
| --- | --- |
| TickSocketService.mq5 | MQL file containing codes to connect to socket server at 9070 and then send tick data |
| tick\_server.py | python server socket opening port 9070 for MetaTrader and port 9071 for other clients |
| tick\_client.py | python client socket connecting to port 9071 and receives data sent by the server |

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/18680.zip "Download all attachments in the single ZIP archive")

[TickSocketService.mq5](https://www.mql5.com/en/articles/download/18680/ticksocketservice.mq5 "Download TickSocketService.mq5")(2 KB)

[tick\_server.py](https://www.mql5.com/en/articles/download/18680/tick_server.py "Download tick_server.py")(2.86 KB)

[tick\_client.py](https://www.mql5.com/en/articles/download/18680/tick_client.py "Download tick_client.py")(1.32 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [MetaTrader Meets Google Sheets with Pythonanywhere: A Guide to Secure Data Flow](https://www.mql5.com/en/articles/19175)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/492029)**
(5)


![Yevgeniy Koshtenko](https://c.mql5.com/avatar/2025/3/67e0f73d-60b1.jpg)

**[Yevgeniy Koshtenko](https://www.mql5.com/en/users/koshtenko)**
\|
5 Aug 2025 at 16:51

**MetaQuotes:**

Published article [Transferring tick data from MetaTrader to Python via sockets using MQL5 services](https://www.mql5.com/en/articles/18680):

Author: [lazymesh](https://www.mql5.com/en/users/lazymesh "lazymesh")

Interesting work. Is it possible to create in this way a risk management server for a network of terminals listening to it?


![Alain Verleyen](https://c.mql5.com/avatar/2024/5/663a6cdf-e866.jpg)

**[Alain Verleyen](https://www.mql5.com/en/users/angevoyageur)**
\|
5 Aug 2025 at 23:02

Good and useful article, congratulations.


![Ramesh Maharjan](https://c.mql5.com/avatar/avatar_na2.png)

**[Ramesh Maharjan](https://www.mql5.com/en/users/lazymesh)**
\|
8 Aug 2025 at 17:23

**Alain Verleyen [#](https://www.mql5.com/ru/forum/492594#comment_57737590):**

Nice and useful article, congratulations.

Thanks

![Ramesh Maharjan](https://c.mql5.com/avatar/avatar_na2.png)

**[Ramesh Maharjan](https://www.mql5.com/en/users/lazymesh)**
\|
8 Aug 2025 at 17:26

**Yevgeniy Koshtenko [#](https://www.mql5.com/ru/forum/492594#comment_57735955):**

Interesting work. Is it possible to create a risk management server in this way for a network of terminals listening to it?

yes, it is possible

![Delane Tendai Nyaruni](https://c.mql5.com/avatar/2023/8/64f0a879-0536.jpg)

**[Delane Tendai Nyaruni](https://www.mql5.com/en/users/mr_nyaruni)**
\|
8 Aug 2025 at 21:32

**Yevgeniy Koshtenko [#](https://www.mql5.com/en/forum/492029#comment_57735956):**

Interesting work. Is it possible to create in this way a risk management server for a network of terminals listening to it?

Yes it is possible, depends on the latency you can tolerate


![MQL5 Wizard Techniques you should know (Part 77): Using Gator Oscillator and the Accumulation/Distribution Oscillator](https://c.mql5.com/2/160/18946-mql5-wizard-techniques-you-logo.png)[MQL5 Wizard Techniques you should know (Part 77): Using Gator Oscillator and the Accumulation/Distribution Oscillator](https://www.mql5.com/en/articles/18946)

The Gator Oscillator by Bill Williams and the Accumulation/Distribution Oscillator are another indicator pairing that could be used harmoniously within an MQL5 Expert Advisor. We use the Gator Oscillator for its ability to affirm trends, while the A/D is used to provide confirmation of the trends via checks on volume. In exploring this indicator pairing, as always, we use the MQL5 wizard to build and test out their potential.

![Building a Trading System (Part 1): A Quantitative Approach](https://c.mql5.com/2/159/18587-building-a-profitable-trading-logo__1.png)[Building a Trading System (Part 1): A Quantitative Approach](https://www.mql5.com/en/articles/18587)

Many traders evaluate strategies based on short-term performance, often abandoning profitable systems too early. Long-term profitability, however, depends on positive expectancy through optimized win rate and risk-reward ratio, along with disciplined position sizing. These principles can be validated using Monte Carlo simulation in Python with back-tested metrics to assess whether a strategy is robust or likely to fail over time.

![Market Profile indicator (Part 2): Optimization and rendering on canvas](https://c.mql5.com/2/106/Market_Profile_Indicator_Part2_LOGO.png)[Market Profile indicator (Part 2): Optimization and rendering on canvas](https://www.mql5.com/en/articles/16579)

The article considers an optimized version of the Market Profile indicator, where rendering with multiple graphical objects is replaced with rendering on a canvas - an object of the CCanvas class.

![Implementing Practical Modules from Other Languages in MQL5 (Part 03): Schedule Module from Python, the OnTimer Event on Steroids](https://c.mql5.com/2/159/18913-implementing-practical-modules-logo.png)[Implementing Practical Modules from Other Languages in MQL5 (Part 03): Schedule Module from Python, the OnTimer Event on Steroids](https://www.mql5.com/en/articles/18913)

The schedule module in Python offers a simple way to schedule repeated tasks. While MQL5 lacks a built-in equivalent, in this article we’ll implement a similar library to make it easier to set up timed events in MetaTrader 5.

[![](https://www.mql5.com/ff/sh/5z040u47jcv59943z2/6c76c03a8b37e08b8655a1a085770b7a.jpg)\\
MetaTrader 5 for iOS and Android\\
\\
Fully featured platform for any devices and web browsers\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=ddonqpipxfqlnsvzlwuowsuwlejpyjxk&s=9daba65b69f40afc3c35f95b1f84ef5824d68c47f29ce96a6dc5b164a2727baa&uid=&ref=https://www.mql5.com/en/articles/18680&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5062566385520911524)

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