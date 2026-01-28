---
title: MetaTrader 5 and Python integration: receiving and sending data
url: https://www.mql5.com/en/articles/5691
categories: Trading, Integration, Expert Advisors
relevance_score: 6
scraped_at: 2026-01-23T11:28:12.547633
---

[![](https://www.mql5.com/ff/si/h2ryn394uwcpxwmxc2.jpg)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Feconomic-calendar%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dopen.calendar%26utm_content%3Deconomic.calendar%26utm_campaign%3Den.0009.desktop.default&a=qdeulxgvibgwytgewnvfatbocjnnninc&s=5c0c60f00ff5f5bedb0fdf65d9d79eb820442eb43ffac2b85aa003224f9dba14&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=ltsreylewqqayirlpcvroqhgvxbihfut&ssn=1769156891166127683&ssn_dr=0&ssn_sr=0&fv_date=1769156891&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F5691&back_ref=https%3A%2F%2Fwww.google.com%2F&title=MetaTrader%205%20and%20Python%20integration%3A%20receiving%20and%20sending%20data%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176915689172457605&fz_uniq=5062491361032184603&sv=2552)

MetaTrader 5 / Trading


### Why integrate MQL5 and Python?

Comprehensive data processing requires extensive tools and is often beyond the sandbox of one single application. Specialized programming languages are used for processing and analyzing data, statistics and machine learning. One of the leading programming languages for data processing is Python. A very effective solution is to use the power of the language and included libraries for the development of trading systems.

There are different solutions for implementing the interaction of two or more programs. Sockets are one the fastest and most flexible solutions.

A network socket is the endpoint of interprocess communication over a computer network. The MQL5 Standard Library includes a group of [Socket](https://www.mql5.com/en/docs/network/socketcreate) functions, which provide a low-level interface for working on the Internet. This is a common interface for different programming languages, as it uses system calls at the operating system level.

Data exchange between the prices is implemented over TCP/IP (Transmission Control Protocol/Internet Protocol). Thus, processes can interact within a single computer and over a local network or the Internet.

To establish a connection, it is necessary to create and initialize a TCP server to which the client process will connect. Once the interaction of processes is completed, the connection must be forcibly closed. Data in a TCP exchange is a stream of bytes.

When creating a server, we need to associate a socket with one or more hosts (IP addresses) and an unused port. If the list of hosts is not set or is specified as "0.0.0.0", the socket will listen to all hosts. If you specify "127.0.0.1" or 'localhost', connection will be possible only within the "internal loop", i.e. only within one computer.

Since only the client is available in MQL5, we will create a server in Python.

### Creating a socket server in Python

The purpose of the article is not to teach the Python programming basics. It is therefore assumed that the reader is familiar with this language.

We will use version 3.7.2 and the built-in [socket](https://www.mql5.com/go?link=https://docs.python.org/3/library/socket.html "https://docs.python.org/3/library/socket.html") package. Please read related documentation for more details.

We will write a simple program which will create a socket server and receive the necessary information from the client (the MQL5 program), handle it and send back the result. This seems to be the most efficient interaction method. Suppose we need to use a machine learning library, such as for example [scikit learn](https://www.mql5.com/go?link=https://scikit-learn.org/stable/ "https://scikit-learn.org/stable/"), which will calculate linear regression using prices and return coordinates, based on which a line can be drawn in the MetaTrader 5 terminal. This is the basic example. However, such interaction can also be used for training a neural network, for sending to it data from the terminal (quotes), learning and returning the result to the terminal.

Let us create the socketserver.py program and import the libraries described above:

```
import socket, numpy as np
from sklearn.linear_model import LinearRegression
```

Now we can proceed to creating a class responsible for socket manipulation:

```
class socketserver:
    def __init__(self, address = '', port = 9090):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.address = address
        self.port = port
        self.sock.bind((self.address, self.port))
        self.cummdata = ''

    def recvmsg(self):
        self.sock.listen(1)
        self.conn, self.addr = self.sock.accept()
        print('connected to', self.addr)
        self.cummdata = ''

        while True:
            data = self.conn.recv(10000)
            self.cummdata+=data.decode("utf-8")
            if not data:
                break
            self.conn.send(bytes(calcregr(self.cummdata), "utf-8"))
            return self.cummdata

    def __del__(self):
        self.sock.close()
```

When creating a class object, the constructor gets the host name (IP address) and port number. Then the _sock_ object is created, which is associated with the address and port _sock.bind()_.

The _recvmsg_ method listens for the incoming connection _sock.listen(1)_. When an incoming client connection arrives, the server accepts it _self.sock.accept()._

Then the server waits in an infinite loop for an incoming client message, which arrives as a stream of bytes. Since the message length is not known in advance, the server receives this message in parts, say 1 Kbytes at a time, until the whole message is read _self.conn.recv(10000)_. Received piece of data is converted to a string _data.decode("utf-8")_ and is added to the rest of the string _summdata_.

Once all the data have been received ( _if not data:)_, the server sends to the client a string containing the rightmost and leftmost coordinates of the calculated regression line. The string is preliminary converted to a byte array _conn.send(bytes(calcregr(self.cummdata), "utf-8"))_.

At the end, the method returns the string received from the client. It can be used for the visualization of received quotes, among others.

A destructor closes the socket once the Python program execution is completed.

Please note that this is not the only possible implementation of the class. Alternatively, you can separate the methods for receiving and sending messages and use it in different ways at different points in time. I have only described the basic technology for creating a connection. You can implement your own solutions.

Let us consider in more detail the linear regression learning method within the current implementation:

```
def calcregr(msg = ''):
    chartdata = np.fromstring(msg, dtype=float, sep= ' ')
    Y = np.array(chartdata).reshape(-1,1)
    X = np.array(np.arange(len(chartdata))).reshape(-1,1)

    lr = LinearRegression()
    lr.fit(X, Y)
    Y_pred = lr.predict(X)
    type(Y_pred)
    P = Y_pred.astype(str).item(-1) + ' ' + Y_pred.astype(str).item(0)
    print(P)
    return str(P)
```

The received byte stream is converted to a utf-8 string, which is then accepted by the _calcregr(msg = ' ')_ method. Since the string contains a sequence of prices separated by spaces (as implemented in the client), it is converted to a [NumPy](https://www.mql5.com/go?link=http://www.numpy.org/ "http://www.numpy.org/") array of the float type. After that the price array is converted to a column (the data receiving format is sclearn) _Y = np.array(chartdata).reshape(-1,1)__._The predictor for the model is the linear time (a sequence of values; its size is equal to the length of the training sample) _X = np.array(np.arange(len(chartdata))).reshape(-1,1)_

This is followed by training and model prediction, while the first and the last values of the line (the edges of the segment) are written to the "P" variable, converted to a string and passed to the client in byte form.

Now, we only need to create the class object and call the recvmsg() method in a loop:

```
serv = socketserver('127.0.0.1', 9090)

while True:
    msg = serv.recvmsg()
```

### Creating a socket client in MQL5

Let us create a simple Expert Advisor, which can connect to the server, pass the specified number of recent Close prices, get back the coordinates of the regression line and draw it on the chart.

The socksend() function will pass data to the server:

```
bool socksend(int sock,string request)
  {
   char req[];
   int  len=StringToCharArray(request,req)-1;
   if(len<0) return(false);
   return(SocketSend(sock,req,len)==len);
  }
```

It receives the string, converts to a byte array and sends to a server.

The socketreceive() function listens on the port. Once a server response is received, the function returns it as a string:

```
string socketreceive(int sock,int timeout)
  {
   char rsp[];
   string result="";
   uint len;
   uint timeout_check=GetTickCount()+timeout;
   do
     {
      len=SocketIsReadable(sock);
      if(len)
        {
         int rsp_len;
         rsp_len=SocketRead(sock,rsp,len,timeout);
         if(rsp_len>0)
           {
            result+=CharArrayToString(rsp,0,rsp_len);
           }
        }
     }
   while((GetTickCount()<timeout_check) && !IsStopped());
   return result;
  }
```

The last function drawlr() receives a string, in which the left and right line coordinates are written, then parses the string to a string array and draws the linear regression line on a chart:

```
void drawlr(string points)
  {
   string res[];
   StringSplit(points,' ',res);

   if(ArraySize(res)==2)
     {
      Print(StringToDouble(res[0]));
      Print(StringToDouble(res[1]));
      datetime temp[];
      CopyTime(Symbol(),Period(),TimeCurrent(),lrlenght,temp);
      ObjectCreate(0,"regrline",OBJ_TREND,0,TimeCurrent(),NormalizeDouble(StringToDouble(res[0]),_Digits),temp[0],NormalizeDouble(StringToDouble(res[1]),_Digits));
     }

```

The function is implemented in the OnTick() handler.

```
void OnTick() {
 socket=SocketCreate();
 if(socket!=INVALID_HANDLE) {
  if(SocketConnect(socket,"localhost",9090,1000)) {
   Print("Connected to "," localhost",":",9090);

   double clpr[];
   int copyed = CopyClose(_Symbol,PERIOD_CURRENT,0,lrlenght,clpr);

   string tosend;
   for(int i=0;i<ArraySize(clpr);i++) tosend+=(string)clpr[i]+" ";
   string received = socksend(socket, tosend) ? socketreceive(socket, 10) : "";
   drawlr(recieved); }

  else Print("Connection ","localhost",":",9090," error ",GetLastError());
  SocketClose(socket); }
 else Print("Socket creation error ",GetLastError()); }
```

### Testing the MQL5-Python Client-Server application

To run the application, you need to have the Python interpreter installed. You can download it from the [official website](https://www.mql5.com/go?link=https://www.python.org/ "https://www.python.org/").

Then run the server application socketserver.py. It creates a socket and listens for new connections from the MQL5 program socketclientEA.mq5.

After a successful connection, the connection process and regression line prices are displayed in the program window. The prices are sent back to the client:

![](https://c.mql5.com/2/35/snip_20190226154343__1.png)

The connection activity and regression line prices are also displayed in the MetaTrader 5 terminal. Also the regression line is drawn on a chart and is further updated at each new tick:

![](https://c.mql5.com/2/35/snip_20190226154835.png)

We have considered the implementation of direct interaction of two programs via a socket connection. At the same time, MetaQuotes has developed a Python package, which allows receiving data directly from the terminal. For more details, please see the forum discussion related to [the use of Python in MetaTrader](https://www.mql5.com/ru/forum/306688) (in Russian, so use the auto translation option).

Let us create a script to demonstrate how to receive quotes from the terminal.

### Getting and analyzing quotes using MetaTrader 5 Python API

First you need to install the MetaTrader5 Python module (the summary of Python discussions is available [here](https://www.mql5.com/en/forum/306742)).

```
pip install MetaTrader5
```

Import it to the program and initialize connection to the terminal:

```
from MetaTrader5 import *
from datetime import date
import pandas as pd
import matplotlib.pyplot as plt

# Initializing MT5 connection
MT5Initialize()
MT5WaitForTerminal()

print(MT5TerminalInfo())
print(MT5Version())
```

After that create the list of desired symbols and successively request Close prices for each currency pair from the terminal to pandas dataframe:

```
# Create currency watchlist for which correlation matrix is to be plotted
sym = ['EURUSD','GBPUSD','USDJPY','USDCHF','AUDUSD','GBPJPY']

# Copying data to dataframe
d = pd.DataFrame()
for i in sym:
     rates = MT5CopyRatesFromPos(i, MT5_TIMEFRAME_M1, 0, 1000)
     d[i] = [y.close for y in rates]
```

Now we can disconnect from the terminal and then represent currency pair prices as percentage changes by calculating the correlation matrix and displaying it on the screen:

```
# Deinitializing MT5 connection
MT5Shutdown()

# Compute Percentage Change
rets = d.pct_change()

# Compute Correlation
corr = rets.corr()

# Plot correlation matrix
plt.figure(figsize=(10, 10))
plt.imshow(corr, cmap='RdYlGn', interpolation='none', aspect='auto')
plt.colorbar()
plt.xticks(range(len(corr)), corr.columns, rotation='vertical')
plt.yticks(range(len(corr)), corr.columns);
plt.suptitle('FOREX Correlations Heat Map', fontsize=15, fontweight='bold')
plt.show()
```

![](https://c.mql5.com/2/35/snip_20190314074100.png)

We can see a good correlation between GBPUSD and GBPJPY in the above heat map. Then we can test the co-integration by importing the statmodels library:

```
# Importing statmodels for cointegration test
import statsmodels
from statsmodels.tsa.stattools import coint

x = d['GBPUSD']
y = d['GBPJPY']
x = (x-min(x))/(max(x)-min(x))
y = (y-min(y))/(max(y)-min(y))

score = coint(x, y)
print('t-statistic: ', score[0], ' p-value: ', score[1])
```

The relationship between two currency pairs can be displayed as a Z score:

```
# Plotting z-score transformation
diff_series = (x - y)
zscore = (diff_series - diff_series.mean()) / diff_series.std()

plt.plot(zscore)
plt.axhline(2.0, color='red', linestyle='--')
plt.axhline(-2.0, color='green', linestyle='--')

plt.show()
```

![](https://c.mql5.com/2/35/snip_20190314085641.png)

### Visualizing market data using the Plotly library

It is often needed to visualize quotes in a convenient form. This can be implemented using the [Plotly](https://www.mql5.com/go?link=https://plot.ly/ "https://plot.ly/") library, which also allows saving charts in the interactive .html format.

Let us download EURUSD quotes and display them in a candlestick chart:

```
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 16:13:03 2019

@author: dmitrievsky
"""
from MetaTrader5 import *
from datetime import datetime
import pandas as pd
# Initializing MT5 connection
MT5Initialize()
MT5WaitForTerminal()

print(MT5TerminalInfo())
print(MT5Version())

# Copying data to pandas data frame
stockdata = pd.DataFrame()
rates = MT5CopyRatesFromPos("EURUSD", MT5_TIMEFRAME_M1, 0, 5000)
# Deinitializing MT5 connection
MT5Shutdown()

stockdata['Open'] = [y.open for y in rates]
stockdata['Close'] = [y.close for y in rates]
stockdata['High'] = [y.high for y in rates]
stockdata['Low'] = [y.low for y in rates]
stockdata['Date'] = [y.time for y in rates]

import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

trace = go.Ohlc(x=stockdata['Date'],
                open=stockdata['Open'],
                high=stockdata['High'],
                low=stockdata['Low'],
                close=stockdata['Close'])

data = [trace]
plot(data)
```

![](https://c.mql5.com/2/35/snip_20190315085137.png)

It is also possible to download and display any depth of the Bid and Ask history:

```
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 16:13:03 2019

@author: dmitrievsky
"""
from MetaTrader5 import *
from datetime import datetime

# Initializing MT5 connection
MT5Initialize()
MT5WaitForTerminal()

print(MT5TerminalInfo())
print(MT5Version())

# Copying data to list
rates = MT5CopyTicksFrom("EURUSD", datetime(2019,3,14,13), 1000, MT5_COPY_TICKS_ALL)
bid = [x.bid for x in rates]
ask = [x.ask for x in rates]
time = [x.time for x in rates]

# Deinitializing MT5 connection
MT5Shutdown()

import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
data = [go.Scatter(x=time, y=bid), go.Scatter(x=time, y=ask)]

plot(data)
```

![](https://c.mql5.com/2/35/snip_20190315085427.png)

### Conclusion

In this article, we considered options for implementing communication between the terminal and a program written in Python, via sockets and directly using MetaQuotes' specialized library. Unfortunately, the current implementation of the socket client in MetaTrader 5 is not suitable for running in the Strategy Tester, so no full testing and measurement of the solution performance were performed. Let us wait for further updates of the socket functionality.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/5691](https://www.mql5.com/ru/articles/5691)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/5691.zip "Download all attachments in the single ZIP archive")

[Socket\_client-server.zip](https://www.mql5.com/en/articles/download/5691/socket_client-server.zip "Download Socket_client-server.zip")(4.33 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Creating a mean-reversion strategy based on machine learning](https://www.mql5.com/en/articles/16457)
- [Fast trading strategy tester in Python using Numba](https://www.mql5.com/en/articles/14895)
- [Time series clustering in causal inference](https://www.mql5.com/en/articles/14548)
- [Propensity score in causal inference](https://www.mql5.com/en/articles/14360)
- [Causal inference in time series classification problems](https://www.mql5.com/en/articles/13957)
- [Cross-validation and basics of causal inference in CatBoost models, export to ONNX format](https://www.mql5.com/en/articles/11147)
- [Metamodels in machine learning and trading: Original timing of trading orders](https://www.mql5.com/en/articles/9138)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/308679)**
(107)


![Alex FXPIP](https://c.mql5.com/avatar/2020/10/5F9DC28E-F0D0.PNG)

**[Alex FXPIP](https://www.mql5.com/en/users/rufxstrategy)**
\|
14 Feb 2025 at 10:21

**Ernesto Che data to draw fibs and channels. In your example python returns two values, but I have to return data for up to 12 different structures with three coordinate points each.**

At first I got stuck with the fact that mt5 did not return a long string describing all structures at once. I made the exchange through several shorter queries. Everything seems to work, but from time to time.

Something tells me that it's about the timeout setting. Could you please suggest a direction for finding a solution?

Thanks in advance

Hi! I was raising a socket too. I have a problem - I write an indicator in python. It takes data from mt5 a lot and as much as you want, but to return data to mt5 is not so easy. On the socket it is possible to transmit only a string up to 100 small strings, but I need more. What solutions are there besides socket? And besides web/internet requests of [data exchange](https://www.mql5.com/en/articles/3331 "Article: Using cloud storage to exchange data between terminals ") between Python->MT5? I don't want to raise MySQL DB for this. We are talking about transferring from python to MT5 about 40 currency pairs with a history of 1000 recalculated readings and further construction of indicator lines in MT5... I want to get away from calculations in MT5 as python does it much faster. So I would like to see all lines of the indicator and not only the last bar transmitted by a string once a second.

Can someone advise me something useful?

TCP/IP Socket - MT5 will receive one string anyway and it will not fit 30 000 data.... Limitations of one string variable even in jason format is up to 100 strings. What is the use of such a Socket ?

It turns out that for big data one way out is Python-->MySQL-->MT5

I also came up with this idea:

Using multiprocessing.shared\_memory in Python and WinAPI in MQL5 allows you to read data directly from memory.


![Maxim Kuznetsov](https://c.mql5.com/avatar/2016/1/56935A91-AF51.png)

**[Maxim Kuznetsov](https://www.mql5.com/en/users/nektomk)**
\|
14 Feb 2025 at 10:52

**Alex Renko [#](https://www.mql5.com/ru/forum/307118/page5#comment_55913286):**

other than web/internet requesters for data exchange between Python->MT5? I don't want to bring up MySQL DB.

"bring up SQLite" \- MT5 has it built in, python obviously has it too...

You can use the socket to report what you need promptly (signals, alerts, etc.), and put big data into SQLite. That is, in python you put everything you need into the database and blow the whistle on the "data update" socket. And in MQL you can reread the database on the whistle.

The database itself can be stored on a frame disc

![Alex FXPIP](https://c.mql5.com/avatar/2020/10/5F9DC28E-F0D0.PNG)

**[Alex FXPIP](https://www.mql5.com/en/users/rufxstrategy)**
\|
18 Feb 2025 at 03:19

**Maxim Kuznetsov [#](https://www.mql5.com/ru/forum/307118/page6#comment_55913501):**

"bring up SQLite" \- MT5 has it built in, python obviously has it as well.

You use the socket to report what you need promptly (signals, alerts, etc.), and put big data into SQLite. That is, in python you put everything you need into the database and blow the whistle on the "data update" socket. And in MQL you can reread the database on the whistle

The database itself can be stored on a frame disc

I see your point, thanks for the answer... I thought so

MetaTrader 5 uses its own programming language **MQL5**, which runs in an isolated process. This does not allow direct interaction with the shared memory of the operating system, as other applications that use low-level APIs can do.

Shell within a shell.... It's been 20 years since they can't make a native version for mac...

![Maxim Kuznetsov](https://c.mql5.com/avatar/2016/1/56935A91-AF51.png)

**[Maxim Kuznetsov](https://www.mql5.com/en/users/nektomk)**
\|
18 Feb 2025 at 04:50

**Alex Renko [#](https://www.mql5.com/ru/forum/307118/page6#comment_55939733):**

I see your point, thanks for the reply.... well I thought so

MetaTrader 5 uses its own programming language **MQL5**, which runs in an isolated process. This does not allow direct interaction with the shared memory of the operating system, as other applications that use low-level APIs can do.

Shell within a shell.... It's been 20 years since they can't make a native version for mac...

nobody forbids to use DLL - and there you can do what you want.

You can search memory directly (probably no faster :-) ).

Or efficient intermediate solutions, a la in-memory key-value db.

Only in most cases, the brakes and bottle-neck are on Python's side

![Stefano Cerbioni](https://c.mql5.com/avatar/2016/6/5755B8A8-5894.png)

**[Stefano Cerbioni](https://www.mql5.com/en/users/faustf)**
\|
9 Oct 2025 at 08:42

for me not  work example anyone  know  why ? i sett in option 127.0.0.1:9090 ![](https://c.mql5.com/3/476/1798824087676.png)  i run [python socketserver](https://www.mql5.com/en/articles/5691 "Article: Connecting MetaTrader 5 and Python: Receiving and Sending Data ").py e  run EA socketclientEA.ex5 ( after  fix  char   to uchar )run it ,  but retrun me this error

```
2025.10.09 10:41:44.665 socketclientEA (#GDAXIm,H1)     Connection localhost:9090 error 4014
```

![Color optimization of trading strategies](https://c.mql5.com/2/35/avatar-colorful.png)[Color optimization of trading strategies](https://www.mql5.com/en/articles/5437)

In this article we will perform an experiment: we will color optimization results. The color is determined by three parameters: the levels of red, green and blue (RGB). There are other color coding methods, which also use three parameters. Thus, three testing parameters can be converted to one color, which visually represents the values. Read this article to find out if such a representation can be useful.

![Studying candlestick analysis techniques (Part II): Auto search for new patterns](https://c.mql5.com/2/35/Pattern_I__3.png)[Studying candlestick analysis techniques (Part II): Auto search for new patterns](https://www.mql5.com/en/articles/5630)

In the previous article, we analyzed 14 patterns selected from a large variety of existing candlestick formations. It is impossible to analyze all the patterns one by one, therefore another solution was found. The new system searches and tests new candlestick patterns based on known candlestick types.

![Developing graphical interfaces for Expert Advisors and indicators based on .Net Framework and C#](https://c.mql5.com/2/35/icon__3.png)[Developing graphical interfaces for Expert Advisors and indicators based on .Net Framework and C#](https://www.mql5.com/en/articles/5563)

The article presents a simple and fast method of creating graphical windows using Visual Studio with subsequent integration into the Expert Advisor's MQL code. The article is meant for non-specialist audiences and does not require any knowledge of C# and .Net technology.

![MQL Parsing by Means of MQL](https://c.mql5.com/2/35/MQL5-avatar-analysis.png)[MQL Parsing by Means of MQL](https://www.mql5.com/en/articles/5638)

The article describes a preprocessor, a scanner, and a parser to be used in parsing the MQL-based source codes. MQL implementation is attached.

[![](https://www.mql5.com/ff/sh/592yc11u3j4rs5z9z2/01.png)How AI helps create robots for MetaTrader 5Learn from our book "Neural Networks in Algo Trading with MQL5"Read](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/neurobook%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.neurobook%26utm_content=visit.page%26utm_campaign=neurobook.promo.04.2024&a=ghrobswocqgvhztzjldphupateyllpro&s=9929cb0b8629585b5a42fabc06c525e41f6c0ebdf3045d044a5413b93ea88b47&uid=&ref=https://www.mql5.com/en/articles/5691&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5062491361032184603)

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