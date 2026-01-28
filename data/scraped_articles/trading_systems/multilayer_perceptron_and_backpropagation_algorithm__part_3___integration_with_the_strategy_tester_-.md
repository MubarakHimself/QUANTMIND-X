---
title: Multilayer perceptron and backpropagation algorithm (Part 3): Integration with the Strategy Tester - Overview (I).
url: https://www.mql5.com/en/articles/9875
categories: Trading Systems, Integration
relevance_score: 6
scraped_at: 2026-01-23T11:44:25.604640
---

[![](https://www.mql5.com/ff/sh/9nb0c8df2rmwfn89z2/01.png) MetaTrader VPS vs regular cloud hosting services8 reasons why our solution is the best option for automated tradingRead](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/450486&a=dgmsfszgoedimaicrqqmagvqzpwuxkur&s=c59e3617ccf44fd54d4c50a03b44fd689ff7507b8fe4990c83772cc5419e627d&uid=&ref=https://www.mql5.com/en/articles/9875&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5062681237241374439)

MetaTrader 5 / Tester


### Introduction

In previous articles, we discussed the construction and use of machine learning models in a simplified way using a client-server connection. However, these models only work in production environments because the tester does not execute network functions. Therefore, when starting this study, I tested models in the Python environment. This is not that bad, but this implies making the model (or models) deterministic when deciding whether to buy or sell a particular asset, or when implementing technical indicators in Python. The latter is not always applicable due to custom or closed source code. The absence of the Strategy Tester is especially important when using strategies that utilize custom indicators as filters or when testing strategies with take profit and stop loss, trailing stop or breakeven. Building your own tester, even in more accessible languages like Python, is quite a challenge.

### Overview

I needed something adaptable for regression and classification models, which can be easily used in Python, so I decided to build a messaging system. The system will work synchronously while exchanging messages. Python will represent the server side, while MQL5 will represent the client side.

![](https://c.mql5.com/2/51/Captura_de_tela_20221027_174440.png)

### Organizing the development

While trying to find a way to integrate the system, I first thought about using REST API which is rather simple in terms of building and managing. However, after reviewing the documentation for the WebRequest function, I realized that this option is not applicable since the documentation explicitly states that it cannot be used.

"WebRequest() cannot run in [the Strategy Tester](https://www.mql5.com/en/docs/runtime/testing#alert_etc)".

I found this limitation on network functions rather frustrating, but I continued to explore other ways to share information. I thought about using a named pipe to send messages in binary files, but in my case it would serve only for experimentation and was not necessary at that moment. However, I have shelved this idea for future updates.

Moving further in my research, I came across some messages that gave me a new solution:

"Many developers face the same problem - how to get to the trading terminal sandbox without using unsafe DLLs..

One of the easiest and safest methods is to use standard Named Pipes that work as normal file operations. They allow you to organize interprocess client-server communication between programs."

"Protection system of the MetaTrader 5 trading platform does not allow MQL5 programs to run outside their sandbox, guarding traders against threats when using untrusted Expert Advisors. Using named pipes, you can easily create integrations with third-party software and manage EAs from outside."

Thinking further, I realized that I could leverage message exchanges using CSV files. This is because in Python there will be no problems with processing CSV data, and the standard MQL5 classes that work with files (CFile, CFileTxt, etc.) allow writing data of all types and arrays, but they do not include the option to write the header to a CSV file. But this limitation can be easily resolved.

So, I decided to develop an architecture that would allow sharing files before developing a solution on the MQL5 side. Inter-Process-Communication (IPC) is a group of mechanisms that allow processes to transfer information between themselves.

After thinking about how to implement the required controls, I designed the architecture to deploy. While this process may seem unnecessary or even absurd, it is very important for the development as it gives an idea of what will be done, and which activities need to be done first.

Using [Figma](https://www.mql5.com/go?link=https://www.figma.com/file/hSxLOdFiLTYRoAXRvCpS6v/Untitled?node-id=0%253A1 "https://www.figma.com/file/hSxLOdFiLTYRoAXRvCpS6v/Untitled?node-id=0%3A1"), I designed what I would later use as documentation and reference.

![](https://c.mql5.com/2/51/892398002008.png)

In order to better understand the context of the previous topic, I will leave an explanation about the message flow that we will establish in order to create stable and secure communication. The idea is to not go into some technical issues initially in order to make the architecture easier to understand.

Whenever the server (Python) is initialized, it will wait for an initialization message to be sent, which is "1 - Waiting for initialization" flow. The message exchange process starts only after the Expert Advisor is attached to the chart. The task of MetaTrader is to send a message to Python about which host, port and environment it is running on.

The following macros are responsible for generating the initialization message header.

```
#define HEADER_FILE_INIT {"host","port","typerun"}
#define LINES_FILE_INT(HOST, PORT, TYPE) {{string(HOST), string(PORT), string(TYPE)}}
```

When I speak about the environment, I mean the place where the EA is running, either the Strategy Tester or the live account. So, we will use "Test" for the test environment and "Live" for the live one.

You can see below that the EA receives the "Host" and "Port" parameters.

```
sinput group   "General Configuration"
sinput string            InpHost           = "127.0.0.1";
sinput int               InpPort           = 8081;
```

During initialization, the server collects environment, host and port entries and stores them for later reading.

```
static EtypeRun typerun= (MQLInfoInteger(MQL_TESTER) || MQLInfoInteger(MQL_VISUAL_MODE))?TEST:LIVE;
```

```
if(!monitor.OnInit(typerun, InpHost, InpPort))
   return(INIT_FAILED);
bool CMonitor::OnInit(EtypeRun type_run, string host, int port)
  {
   ...
   File.SetCommon(true);
   File.Open("TransferML/init.csv", FILE_WRITE|FILE_SHARE_READ|FILE_ANSI);
   string header[3]   = HEADER_FILE_INIT;
   string lines[1][3] = LINES_FILE_INT(host,port,type_run);

   if((File.WriteHeader(header)<1&File.WriteLine(lines)<1&!Strategy.Config(m_params))!=0)
      res=false;

   File.Close();

...
  }
```

In the code above, we read the "init" file and pass the environment and host data. Steps "1- Startup Initialization" and "2- Send Initialization" are performed here.

Below is a Python code that will receive the initialization, will process the data, will set the environment to be used and will confirm the initialization to the client. Steps "2- Collect data input", "3 -Startup process data input", "4 - Set Env" and "5 - Confirm initialization" are performed here.

```
host, port, typerun = file.check_init_param(PATH_COMMON.format(INIT_ARCHIVE))
```

```
file.save_file_csv(PATH_COMMON.format(INIT_OK_ARCHIVE))
```

After all these steps, MetaTrader should be waiting to receive a server startup confirmation. This step is "3 - Waiting for confirmation".

```
bool CMonitor::OnInit(EtypeRun type_run, string host, int port)
  {
   ...
   while(!File.IsExist("TransferML/init_checked.csv", FILE_COMMON))
     {
      //waiting for startup
      Comment("waiting for startup");
     }
...

  }
```

After this step, Python selects which thread to use. If it is the production thread, we will use a connection via sockets, leveraging part of what was done in the previous article. If this is a test thread, then we will use messaging with CSV files. The server side waits for instructions from the client, which sends commands like "START", "STOP" and "BREAK" (the names were chosen randomly) to start the process of sending the value generated by some model.

### Pros and cons

Our communication is clear and efficient, because the standardization of data at every stage ensures system stability. In addition, data understanding is simplified both on the Python side using of the pandas library, and on the MQL5 side using matrices and vectors.

![](https://c.mql5.com/2/50/1555669240839.png)

Message exchange is the heart of the problem, so I chose to standardize data sending and receiving in the CSV format. To simplify this task, I've developed a class that abstracts the effort of creating strings and headers. It is used as the basis for data exchange between environments. Next is the class header with the main methods and attributes.

```
class CFileCSV : public CFile
  {
private:
   template<typename T>
   string            ToString(const int, const T &[][]);
   template<typename T>
   string            ToString(const T &[]);
   short             m_delimiter;

public:
                     CFileCSV(void);
                    ~CFileCSV(void);
   //--- methods for working with files
   int               Open(const string,const int, const short);
   template<typename T>
   uint              WriteHeader(const T &values[]);
   template<typename T>
   uint              WriteLine(const T &values[][]);
   string            Read(void);
  };
```

As you can see, the methods that write lines and headers accept dynamic vectors and matrices, which allows you to build a file at runtime, without the need to concatenate the text using the "StringAdd()" or "StringConcatenate()" functions in the main code. This work is done by the "ToString" functions, which receive a vector or matrix and convert it to the CSV format.

For example:

Imagine that we have a model that receives the value of the last 4 candlesticks, and the information that we consider necessary to be transmitted is something like this:

data;close;val\_ma

10202022;10.55;10.49

10212022;10.95;11.09

10222022;11.55;11.29

10232022;11.15;11.29

This example illustrates the use of static data stored in global variables. However, in a real system, this data will be collected according to the needs of each strategy, as can be seen in the image that shows the integration architecture. It is important to emphasize that the strategy is the main element of the system, since it determines what information is needed for the model to work correctly. For example, if we need to add information about prices or indicators, this would be one of the options. However, keep in mind that changing the format of the data being sent or received will require appropriate code support. While this problem can be easily fixed, it is important to plan for system development. As mentioned earlier, this is just a proof-of-concept (POC) example, and if it looks promising, it could be improved in the future.

To manually create the above example, we need an array with three values that will represent the header and an array \[4\]\[3\] that will contain the data. As you can see, writing and reading this CSV file is simple.

```
#include "FileCSV.mqh"

#define PATH(path) "Test/"+path+".csv"

string H[3] = { "data", "close", "val_ma" };
string L[4][3]  = {{"10202022", "10.55", "10.49"},{"10212022", "10.95", "11.09"},{"10222022", "11.55", "11.29"},{"10232022", "11.15", "11.29"}};

CFileCSV              File;
ulong start=0,time=0;
void OnStart()
  {
   start=0;
   time=0;
   start=GetTickCount();
   for(int i=0; i<100; i++)
     {
      File.Open(PATH("init"), FILE_WRITE|FILE_SHARE_READ|FILE_ANSI);
      ResetLastError();
      if((File.WriteHeader(H)<1&File.WriteLine(L)<1)!=0)
         Print("Error : ", GetLastError());
      File.Close();

      while(!File.IsExist(PATH("init_checked")))
        {
         //waiting for startup
         Comment("waiting for startup");
        }
      File.Delete(PATH("init"));
      File.Delete(PATH("init_checked"));
     }
   time=GetTickCount()-start;
   Print("Time send 100 archives with transfer message [ms]: ",time);
  }
```

The disadvantage of this approach is that the data is written to disk, which can affect the average processing speed. However, if you compare it with the processing speed of a system that uses sockets, the performance seems reasonable.

Implementing a test sending:

We will send 100 files containing 3 data columns and 4 data rows and then measure the data transfer speed.

```
from Services import File

PATH_COMMON     = r'C:\Users\letha\AppData\Roaming\MetaQuotes\Terminal\B8C209507DCA35B09B2C3483BD67B706\MQL5\Files\Test\{}.csv'
INIT_ARCHIVE    = 'init'
INIT_OK_ARCHIVE = 'init_checked'

if __name__ == "__main__":
    file = File()
    file.delete_file(PATH_COMMON.format(INIT_ARCHIVE))
    file.delete_file(PATH_COMMON.format(INIT_OK_ARCHIVE))

    while True:

        receive = file.check_open_file(PATH_COMMON.format(INIT_ARCHIVE))
        file.delete_file(PATH_COMMON.format(INIT_ARCHIVE))
        file.save_file_csv(PATH_COMMON.format(INIT_OK_ARCHIVE))
void OnStart()
  {

   start=0;
   time=0;

   start=GetTickCount();

   for(int i=0; i<100; i++)
     {
      File.Open(PATH("init"), FILE_WRITE|FILE_SHARE_READ|FILE_ANSI);

      ResetLastError();
      if((File.WriteHeader(H)<1&File.WriteLine(L)<1)!=0)
         Print("Error : ", GetLastError());
      File.Close();

      while(!File.IsExist(PATH("init_checked")))
        {
         //waiting for startup
         Comment("waiting for startup");
        }

      File.Delete(PATH("init"));
      File.Delete(PATH("init_checked"));

     }

   time=GetTickCount()-start;
   Print("Time send 100 archives with transfer message [ms]: ",time);

  }
```

Here is the result:

testeCSV (EURUSD,M1)Time to send 100 files, transfer message \[ms\]: 5578

This system is not exceptional, but it has its value, as we will send small amounts of data to the server. And the data will be sent once for each new candlestick opening, so we don't need to worry about that. But if you are going to create a system for streaming quotes, order book data, or anything else, then this architecture is not recommended. There is the possibility to evolve the system into something more elaborate in the future.

Also, the process is limited to one model/strategy, but this can be improved to provide better scalability in the future.

Using linear regression:

What is linear regression?

Linear regression is a statistical technique widely used in financial analysis to predict the behavior of financial assets such as stocks, bonds and currencies. This technique allows financial analysts to identify the relationship between different variables and thus "predict" the future performance of an asset.

To use linear regression for financial assets, first we need to collect the relevant historical data. This includes information about the asset's closing price, trading volume, profit and other relevant economic variables. This data can be obtained from sources such as the stock exchange or financial websites.

Once the data has been collected, it is necessary to choose which dependent and independent variable we will use in the analysis. The dependent variable is the one that needs to be predicted, while the independent variables are those that are used to explain the behavior of the dependent variable. For example, if the goal is to predict the price of a stock, the dependent variable would be the price of the stock, while the independent variables could be the trading volume, the profit, etc.

Then, it is necessary to apply a statistical technique to find the equation of the regression line, which represents the relationship between the independent and dependent variables. This equation is used to predict the future behavior of an asset.

After applying the linear regression technique, it is important to assess the quality of the prediction made. To do this, we can compare predicted results with actual historical data. If the prediction accuracy is low, it may be necessary to make adjustments in the methodology or the selection of different independent variables.

Linear regression is a statistical technique widely used in financial analysis to predict the behavior of financial assets such as stocks, bonds and currencies. This technique allows financial analysts to identify the relationship between different variables and thus predict the future performance of an asset. Implementing linear regression in Python is easy to do using the scikit-learn library and can be a valuable tool for forecasting financial asset prices. However, it is important to remember that linear regression is a basic technique and may not be appropriate for all types of financial assets or specific situations. It is always important to evaluate the prediction quality and consider other financial analysis techniques.

So, you can consider other techniques, such as time series analysis or forecasting models based on artificial intelligence, which can also be used to predict the behavior of financial assets.

### Implementation in Python:

```
import random
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder

random.seed(42)

encoder = OneHotEncoder()

# Create an empty dataframe
data = pd.DataFrame(columns=['ticker', 'price', 'volume', 'economic_indicator'])

# Fill the dataframe with random values
for i in range(500):
    row = {
        'ticker': "FAKE3",
        'price': round(random.uniform(100, 200), 2),
        'volume': round(random.uniform(10000, 100000), 2),
        'economic_indicator': round(random.uniform(1, 100), 2)
    }
    data = data.append(row, ignore_index=True)

print(data)

# apply one-hot encoding of the column "ticker"
onehot_encoded = encoder.fit_transform(data[['ticker']])

# add a new one-hot encoded column to the original dataframe
data['tiker_encoder'] = onehot_encoded.toarray()

# Selecting independent and dependent variables
X = data[['tiker_encoder', 'volume', 'economic_indicator']]
y = data['price']

# Creating the linear regression model
model = LinearRegression()

# Training the model on historical data
model.fit(X, y)

# Making predictions with the trained model
y_pred = model.predict(X)

# Evaluating prediction quality
r2 = r2_score(y, y_pred)
print("Determination coefficient:", r2)

# Making predictions for new data
new_data = [[1, 23228.17, 61.21]]
new_price_pred = model.predict(new_data)
print("Price prediction for new data:", new_price_pred)
```

This code uses the scikit-learn library to generate a linear regression model based on historical price data, trading volume and one indicator. The model is trained on historical data and is used to predict prices. The coefficient of determination (R²) is also calculated as a measure of the prediction quality. In addition, the model is also used to make predictions using the newly provided data.

Please note that this code is static and serves only as an example. It can be easily adapted to work with live and dynamic data and used in a production environment. In addition, it is necessary to obtain market data and economic indicators to train the model.

It is important to note that this is just a basic implementation of the linear regression model used to predict the stock price and it may be necessary to adjust the model and data according to your specific needs. The use of the model on a live account is not recommended.

The provided example is for illustrative purposes only and should not be considered as a complete implementation. A detailed demonstration of the complete implementation will be presented in the next article.

### Conclusion

The proposed architecture was effective in overcoming restrictions on testing Python models, providing a variety of testing options and assisting in validating and evaluating the efficiency of ML models. In the next article, we will discuss in more depth the implementation of the CFileCSV Class, which will be used as the basis for data transfer used in MQL5.

It is important to note that the implementation of the CFileCSV Class will be fundamental for the data exchange between MQL5 and Python, enabling the use of advanced data analysis and modeling features on both platforms, and will be a fundamental component for taking the full advantage of this architecture.

Translated from Portuguese by MetaQuotes Ltd.

Original article: [https://www.mql5.com/pt/articles/9875](https://www.mql5.com/pt/articles/9875)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/9875.zip "Download all attachments in the single ZIP archive")

[Publicacao\_Parte\_III\_I.zip](https://www.mql5.com/en/articles/download/9875/publicacao_parte_iii_i.zip "Download Publicacao_Parte_III_I.zip")(115.74 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Developing an MQL5 RL agent with RestAPI integration (Part 4): Organizing functions in classes in MQL5](https://www.mql5.com/en/articles/13863)
- [Developing an MQL5 RL agent with RestAPI integration (Part 3): Creating automatic moves and test scripts in MQL5](https://www.mql5.com/en/articles/13813)
- [Developing an MQL5 RL agent with RestAPI integration (Part 2): MQL5 functions for HTTP interaction with the tic-tac-toe game REST API](https://www.mql5.com/en/articles/13714)
- [Developing an MQL5 Reinforcement Learning agent with RestAPI integration (Part 1): How to use RestAPIs in MQL5](https://www.mql5.com/en/articles/13661)
- [Integrating ML models with the Strategy Tester (Conclusion): Implementing a regression model for price prediction](https://www.mql5.com/en/articles/12471)
- [Integrating ML models with the Strategy Tester (Part 3): Managing CSV files (II)](https://www.mql5.com/en/articles/12069)

**[Go to discussion](https://www.mql5.com/en/forum/449263)**

![How to Become a Successful Signal Provider on MQL5.com](https://c.mql5.com/2/55/How_to_Become_a_Successful_Signal_Provider_Avatar.png)[How to Become a Successful Signal Provider on MQL5.com](https://www.mql5.com/en/articles/12814)

My main goal in this article is to provide you with a simple and accurate account of the steps that will help you become a top signal provider on MQL5.com. Drawing upon my knowledge and experience, I will explain what it takes to become a successful signal provider, including how to find, test, and optimize a good strategy. Additionally, I will provide tips on publishing your signal, writing a compelling description and effectively promoting and managing it.

![Improve Your Trading Charts With Interactive GUI's in MQL5 (Part I): Movable GUI (I)](https://c.mql5.com/2/55/Revolutionize_Your_Trading_Charts_Part_I_avatar.png)[Improve Your Trading Charts With Interactive GUI's in MQL5 (Part I): Movable GUI (I)](https://www.mql5.com/en/articles/12751)

Unleash the power of dynamic data representation in your trading strategies or utilities with our comprehensive guide on creating movable GUI in MQL5. Dive into the core concept of chart events and learn how to design and implement simple and multiple movable GUI on the same chart. This article also explores the process of adding elements to your GUI, enhancing their functionality and aesthetic appeal.

![Creating an EA that works automatically (Part 14): Automation (VI)](https://c.mql5.com/2/51/aprendendo_construindo_014_avatar.png)[Creating an EA that works automatically (Part 14): Automation (VI)](https://www.mql5.com/en/articles/11318)

In this article, we will put into practice all the knowledge from this series. We will finally build a 100% automated and functional system. But before that, we still have to learn one last detail.

![Automated exchange grid trading using stop pending orders on Moscow Exchange (MOEX)](https://c.mql5.com/2/54/moex-mesh-trading-avatar.png)[Automated exchange grid trading using stop pending orders on Moscow Exchange (MOEX)](https://www.mql5.com/en/articles/10671)

The article considers the grid trading approach based on stop pending orders and implemented in an MQL5 Expert Advisor on the Moscow Exchange (MOEX). When trading in the market, one of the simplest strategies is a grid of orders designed to "catch" the market price.

[![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/01.png)![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/02.png)Powerful analytics for traders of any levelAll the necessary trading reports for beginners and professionals](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/454106&a=muccpajyfystoakuukdobwigjejzmpqn&s=52daad60fa795e635264e6f94898f05493bca3b5124d4cca8eb7e82333c2ef12&uid=&ref=https://www.mql5.com/en/articles/9875&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5062681237241374439)

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