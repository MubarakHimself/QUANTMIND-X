---
title: MQL5 Integration: Python
url: https://www.mql5.com/en/articles/14135
categories: Trading Systems, Integration, Indicators, Expert Advisors
relevance_score: 6
scraped_at: 2026-01-23T11:40:11.855349
---

[![](https://www.mql5.com/ff/sh/zf7a2k61x98jzh89z2/01.png)Speed up your tradingUse our high-speed VPS for MetaTrader 4 and 5Learn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=qtrrsuiwuicrscmckjynyanztbditglq&s=c617dc80d90cfd3783ec1345eec2b419b281f10fec6eac77b3218984ac337259&uid=&ref=https://www.mql5.com/en/articles/14135&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5062632115700409834)

MetaTrader 5 / Trading systems


### Introduction

In this new article, I'm going to give you an ultimate introduction to an important tool that will add value to your programming skills. We'll look at Python Integration. When it comes to how this can be useful for us as developers it will depend on your objectives of software as Python is a high-level programming language that is easy to read, and it is simple as well. Python is a programming language that provides extensive libraries for areas such as data analysis, statistical computing, and machine learning. So, the integration of Python and MQL5 can provide better insights that can be helpful for financial market participants to improve their results through data processing and predictive analytics.

In this article, I will explain how to use Python with MQL5 by giving you simple Python basics and some simple examples after we set up our environment. I'll cover that through the following topics:

- [Python Overview](https://www.mql5.com/en/articles/14135#overview)
- [Python Integration benefits](https://www.mql5.com/en/articles/14135#benefits)
- [Setting up the environment](https://www.mql5.com/en/articles/14135#environment)
- [Simple applications](https://www.mql5.com/en/articles/14135#applications)
- [Conclusion](https://www.mql5.com/en/articles/14135#conclusion)

Let's go deep into this interesting topic to understand how we can improve our trading results through using Python with MQL5.

### Python Overview

Python was developed by Guido van Rossum and released in 1991. It is a high-level programming language known for its readability and simplicity, making it ideal for beginners as well as experienced developers. It is suitable for a wide range of applications, from simple scripts to complex systems, and its concise syntax allows for clean, maintainable code.

Python's key features include:

- Readability: Making code easier to read and write, Python's syntax is intuitive and mirrors natural language.
- Interpretive language: Python executes code on a line-by-line basis, making debugging and development easier.
- Dynamically typed: The types of variables are determined at run time, providing flexibility in coding.
- Extensive standard library: Python's comprehensive library provides support for common tasks such as file I/O, system calls, and data manipulation.
- Versatility: Python has support for a wide range of programming paradigms, including object-oriented, procedural, and functional programming.
- Cross-platform compatibility: Python runs on a variety of operating systems, including Windows, MacOS, and Linux, without requiring any code changes.
- Strong community and ecosystem: The large community and ecosystem of Python offers numerous libraries and tools that extend the possibilities of Python.

Areas of application for Python include:

- Data science: Data analysis and visualization libraries such as Pandas and Matplotlib.
- Artificial Intelligence: Machine learning tools such as scikit-learn, TensorFlow, and Keras.
- Automation: Selenium and Beautiful Soup for automation of repetitive tasks.
- Web development: Web application development frameworks such as Django and Flask.
- Scientific research: Ideal for simulating, statistical analysis, and model building.

Pandas, NumPy, SciPy, scikit-learn, TensorFlow, Matplotlib, Selenium, Django, and Flask are some of the most popular Python libraries and frameworks. In summary, Python's continued popularity and relevance in the programming world are due to its combination of readability, simplicity, and powerful libraries.

### Python Integration benefits

As previously stated, the integration of Python with MQL5 offers a multitude of advantageous features, making it a valuable addition to the system. This section is an attempt to present the most significant advantages associated with integrating Python into a given MQL5 system.

So the integration of these two systems offers the following advantages:

- Facilitate sophisticated data manipulation and analysis through the use of advanced data analysis techniques by using the extensive libraries available in Python, including Pandas and NumPy.
- The incorporation of machine learning libraries, such as scikit-learn, TensorFlow, and Keras, allows for the development and implementation of machine learning models for predictive analytics.
- Automate complex trading strategies and optimize them using Python's robust ecosystem, which helps apply automation and efficiency.
- Enhance your trading algorithms using Python's vast collection of libraries and frameworks.

For more details:

#### Data Analysis:

It is possible to utilize Python's libraries to great effect. Python offers a multitude of libraries tailored for data analysis, including Pandas and NumPy. These libraries facilitate sophisticated data manipulation and statistical analysis, thereby enhancing the quality and depth of analysis in trading strategies.

- The Pandas library provides high-level data structures and methods that facilitate rapid and straightforward data analysis. The Pandas library enables users to efficiently handle large datasets and perform operations such as filtering, grouping, and aggregating data with minimal effort.
- NumPy is a computational library that supports efficient numerical computations, making it well-suited for handling large arrays and matrices of numeric data. It offers a comprehensive range of mathematical operations and is frequently utilized as the foundational library upon which other libraries like Pandas and scikit-learn are built.

#### Machine Learning:

In the field of machine learning, Python language is the preferred choice due to its simplicity and the availability of robust libraries, including scikit-learn, TensorFlow, and Keras. When integrated with MQL5, traders can utilize these libraries to build, develop, and deploy predictive models that employ historical data to predict market movements.

- Scikit-learn: This library provides users with simple yet effective tools for machine learning, enabling them to conduct in-depth data analysis and mining. It boasts a diverse range of algorithms, catering to various needs including classification, clustering, regression, and more.
- TensorFlow and Keras: They are highly popular and widely used tools among developers for deep learning purposes, offering a comprehensive suite of tools for building and training neural networks to construct sophisticated models.

#### Automation and Efficiency:

In the context of repetitive tasks and sophisticated trading strategies, automation can be an effective solution. By automating processes, trading can streamline their operations and reduce the potential for human error. This is particularly beneficial when working with complex strategies, as having a clear and accurate code for the strategy in question can help mitigate risks.

- This automation can be applied to a variety of data-related tasks, including collection, processing, and analysis to free up traders' time, it allows them to focus on developing and executing strategies.
- Furthermore, this approach can be applied to backtesting and optimization by utilizing historical data to assess the strategy's performance and identify areas for improvement through parameter optimization, ultimately leading to enhanced results.

#### Access to a comprehensive range of libraries and frameworks:

MQL5 functionality can be enhanced through the use of a comprehensive ecosystem of libraries and frameworks. These include advanced statistical tools, APIs for external data sources, and complex visualizations, which can be leveraged to expand the capabilities of your MQL5 applications.

- Data Visualization: Libraries such as Matplotlib and Seaborn provide the tools to create informative charts, from simple to sophisticated, which can be used to visualize available data, such as trading performance and other metrics.
- APIs and Data Sources: Python libraries offer a convenient solution for retrieving financial data, performing web scraping, and accessing data sources. These libraries can interface with numerous APIs, which is advantageous for those looking to enhance their trading strategies.

There are many other benefits that can considered when talking about this topic but I think that we mentioned the most important benefits of integrating Python into our systems.

We will provide straightforward applications for trading-related areas that can be applied in practice. This will help to illustrate how we use trading concepts when integrating Python to MQL5. We will examine how these concepts and techniques can be used for different fields demonstrating how they can be integrated into trading processes.

### Setting up the environment

In this section, we will set up the needed software to be able to use Python with MQL5, and the following are steps to take that.

- Install MetaTrader 5 by downloading the installer file by visiting [https://www.metatrader5.com/en](https://www.metatrader5.com/en "https://www.metatrader5.com/en") then you can install it to your device.
- Download the latest version of Python from [https://www.python.org/downloads/windows](https://www.mql5.com/go?link=https://www.python.org/downloads/windows/ "https://www.python.org/downloads/windows/")
- When installing Python, check "Add Python to PATH%" to be able to run Python scripts from the command line.
- It is of great importance to create a distinct environment for each project to maintain a clean, isolated, and reproducible setup. The following steps illustrate how this can be achieved through the use of the command line:

- Navigate the project directory

```
cd /path/to/your/project
```

- Use venv to create the mytestenv environment (the Built-in Virtual Environment Tool)

```
python -m venv mytestenv
```

- It is now necessary to activate the environment that has been created

```
mytestenv\Scripts\activate
```

- The MetaTrader 5 module should be installed from the command line, you can visit the MetaTrader5 Python's package through the [https://pypi.org/project/MetaTrader5/](https://www.mql5.com/go?link=https://pypi.org/project/MetaTrader5/ "https://pypi.org/project/MetaTrader5/") link

```
pip install MetaTrader5
```

- Show the MetaTrader installation details

```
pip show MetaTrader5
```

- In order to facilitate the utilization of the aforementioned functions, it is necessary to add the matplotlib and pandas packages

```
pip install matplotlib
pip install pandas
```

- In case of deactivation, we can use the following command in the command line

```
deactivate
```

At this point in the process, the necessary software, namely MetaTrader5, Python, and the requisite libraries, has been installed on the device, thus enabling the commencement of work.

### Simple applications

As previously stated, Python and MQL5 are invaluable tools that can be utilized in a multitude of tasks and domains, including data analysis, machine learning, and automation. This section endeavors to present straightforward trading-related applications that elucidate the utilization of Python scripts with MetaTrader 5 to obtain an overview of some of the fundamental tasks that can be accomplished.

#### Application one: Open MT5 using a Python script:

In this application, the objective is to create a Python script that will open the MetaTrader 5 terminal and print a message indicating whether or not the terminal has been initialized. The following code represents the complete script. It is necessary to replace the (xxxxx) with your relevant account details, including the account number, login, password, and the broker server.

```
import MetaTrader5 as mt5
print ("MetaTrader5 PKG version: ",mt5.__version__)
if mt5.initialize(login=xxxxx, server="xxxxx",password="xxxxx"):
    print ("MT5 initialized Successfully")
else: print ("MT5 initialization Failed, Error code ",mt5.last_error())

```

#### After running this code, you'll find the MetaTrader 5 terminal initialized the same as you click the executable file then you can use it for your trading normally. In addition to that, you will find the following result in your console:

- MetaTrader5 PKG version:  5.0.4424
- MetaTrader5 PKG author:  MetaQuotes Ltd.
- MT5 initialized Successfully

#### Application two: Open positions using a Python script:

In this application, the objective is to create a Python script that will enable the opening of a buy position on the MetaTrader 5 terminal. The following steps will be undertaken to achieve this: a script will be created to open a 0.01 buy position on the XAUUSD at the Ask price, with a stop loss level and take profit.

The MetaTrader5 model is to be imported as mt5.

```
import MetaTrader5 as mt5
```

Printing the version of the MetaTrader5 as a package information.

```
print ("MetaTrader5 PKG version: ",mt5.__version__)
```

Printing the author of the MetaTrader5 as a package information.

```
print ("MetaTrader5 PKG author: ",mt5.__author__)
```

The MetaTrader5 connection is initialized with a message indicating whether the initialization was successful or unsuccessful. If unsuccessful, the message provides the error code. The account details, including the login, the broker server, and the password, should be replaced with the actual account details.

```
if mt5.initialize(login=xxxxx, server="xxxxx",password="xxxxx"):
    print ("MT5 initialized Successfully")
else: print ("MT5 initialization Failed, Error code ",mt5.last_error())
```

The following variables are declared: symbol, lot, point, order\_type, price, sl, tp, deviation, magic, comment, type\_time, and type\_filling.

```
symbol="XAUUSD"
lot=0.01
point=mt5.symbol_info(symbol).point
order_type=mt5.ORDER_TYPE_BUY
price=mt5.symbol_info_tick(symbol).ask
sl=mt5.symbol_info_tick(symbol).ask-100
tp=mt5.symbol_info_tick(symbol).ask+150
deviation=10
magic=2222222
comment="python order"
type_time=mt5.ORDER_TIME_GTC
type_filling=mt5.ORDER_FILLING_IOC
```

Sending the order by declaring the request to be equivalent to the order details as per what we declared before.

```
request={
    "action":mt5.TRADE_ACTION_DEAL,
    "symbol":symbol,
    "volume":lot,
    "type":order_type,
    "price":price,
    "sl":sl,
    "tp":tp,
    "deviation":deviation,
    "magic":magic,
    "comment":comment,
    "type_time":type_time,
    "type_filling":type_filling,
    }
```

Check that there are sufficient funds to perform a requested trade operation by using (order\_check) to be the equivalent of the result value. The check result is returned as a MqlTradeCheckResult structure.

```
result=mt5.order_check(request)
```

The execution of trade operations is accomplished by transmitting a request via the "order\_send" function, which serves as an equivalent to the result value as an update.

```
result=mt5.order_send(request)
```

It is now necessary to terminate the connection to the MetaTrader 5 terminal that was previously established using the shutdown() function.

```
mt5.shutdown()
```

Consequently, the complete code can be located in the same manner as demonstrated by the following block of code.

```
import MetaTrader5 as mt5
print ("MetaTrader5 PKG version: ",mt5.__version__)
print ("MetaTrader5 PKG author: ",mt5.__author__)
if mt5.initialize(login=xxxxx, server="xxxxx",password="xxxxx"):
    print ("MT5 initialized Successfully")
else: print ("MT5 initialization Failed, Error code ",mt5.last_error())
symbol="XAUUSD"
lot=0.01
point=mt5.symbol_info(symbol).point
order_type=mt5.ORDER_TYPE_BUY
price=mt5.symbol_info_tick(symbol).ask
sl=mt5.symbol_info_tick(symbol).ask-100
tp=mt5.symbol_info_tick(symbol).ask+150
deviation=10
magic=2222222
comment="python order"
type_time=mt5.ORDER_TIME_GTC
type_filling=mt5.ORDER_FILLING_IOC
request={
    "action":mt5.TRADE_ACTION_DEAL,
    "symbol":symbol,
    "volume":lot,
    "type":order_type,
    "price":price,
    "sl":sl,
    "tp":tp,
    "deviation":deviation,
    "magic":magic,
    "comment":comment,
    "type_time":type_time,
    "type_filling":type_filling,
    }
result=mt5.order_check(request)
result=mt5.order_send(request)
mt5.shutdown()
```

The result obtained upon executing the code will be identical to the following:

- MetaTrader5 PKG version:  5.0.4424
- MetaTrader5 PKG author:  MetaQuotes Ltd.
- MT5 initialized Successfully
- A buy trade on the gold (XAUUSD) was initiated at the Ask price with a lot size of 0.01, a stop-loss order at the Ask price minus 100, and a take-profit at the Ask price plus 150.

![buyTrade](https://c.mql5.com/2/88/buyTrade.png)

#### Application three: Using MT5 Python API to get data:

A multitude of tasks can be accomplished through the utilization of MQL5 and Python in the context of data management. This section will present a simple example related to trading, in which financial data is obtained using the MT5 Python API. In addition to plotting the data into a graph for visualization purposes, the price of gold (XAUUSD) from August 1st, 2023 until now (The time of writing the article was 12 August 2024) will be obtained, printed, and visualized as a line graph. The following steps will be presented for this process.

The requisite libraries must be imported:

The MetaTrader5 module, designated "mt5," is to be imported for future use as an object for interaction with the MetaTrader 5 trading terminal, as previously determined.

```
import MetaTrader5 as mt5
```

The Pandas' library is imported as pd for the purpose of data manipulation and analysis.

```
import pandas as pd
```

Importing plotly.express as px to be used for data visualization to present the XAUUSD data.

```
import plotly.express as px
```

Importing plot from plotly.offline to facilitate the generation of plots without the necessity of an internet connection.

```
from plotly.offline import plot
```

In order to facilitate the manipulation of dates and times, it is necessary to import the datetime module from datetime.

```
from datetime import datetime
```

Printing the MetaTrader5 package information (version and author).

```
print ("MetaTrader5 PKG version: ",mt5.__version__)
print ("MetaTrader5 PKG author: ",mt5.__author__)
```

The MetaTrader 5 terminal is initialized, and a console message is printed to indicate whether the initialization was successful. If the initialization is successful, the message "MT5 initialized successfully" is displayed. Conversely, if the initialization is unsuccessful, the message "MT5 initialization failed, error code " is displayed, along with the error code and the last error message.

```
if mt5.initialize(login=xxxxx, server="xxxxx",password="xxxxx"):
    print ("MT5 initialized Successfully")
else: print ("MT5 initialization Failed, Error code ",mt5.last_error())
```

The historical data (open, high, low, close, volume, spread, and real volume) of the XAUUSD was obtained from the MetaTrader 5 platform. The following command was used for this purpose:

- pd.DataFrame: A pd.dataFrame can be created with specific criteria, resulting in a two-dimensional, labeled data structure.
- mt5.copy\_rates\_range: is used to determine the type of data in terms of symbol (XAUUSD), timeframe (mt5.TIMEFRAME\_D1), starting date (datetime(2023,8,1)), and ending date (datetime.now()).

```
xauusd_data=pd.DataFrame(mt5.copy_rates_range("XAUUSD", mt5.TIMEFRAME_D1, datetime(2023,8,1), datetime.now()))
```

The time column is converted from Unix timestamps to a readable datetime format using the pandas.to\_datetime function.

```
xauusd_data['time'] = pd.to_datetime(xauusd_data['time'],unit='s')
```

The retrieved data for the XAUUSD should be printed as per the following line of code.

```
print(xauusd_data)
```

The retrieved data was plotted using the px.line function, which enables the creation of a line plot using Plotly Express. This plot represents the price of XAUUSD over time, from the first of August 2023 until the now.

```
fig = px.line(xauusd_data, x=xauusd_data['time'], y=xauusd_data['close'])
plot(fig)
```

The full code is the same as this block of code.

```
import MetaTrader5 as mt5
import pandas as pd
import plotly.express as px
from plotly.offline import plot
from datetime import datetime
print ("MetaTrader5 PKG version: ",mt5.__version__)
print ("MetaTrader5 PKG author: ",mt5.__author__)
if mt5.initialize(login=xxxxx, server="xxxxx",password="xxxxx"):
    print ("MT5 initialized Successfully")
else: print ("MT5 initialization Failed, Error code ",mt5.last_error())
xauusd_data=pd.DataFrame(mt5.copy_rates_range("XAUUSD", mt5.TIMEFRAME_D1, datetime(2023,8,1), datetime.now()))
xauusd_data['time'] = pd.to_datetime(xauusd_data['time'],unit='s')
print(xauusd_data)
fig = px.line(xauusd_data, x=xauusd_data['time'], y=xauusd_data['close'])
plot(fig)
```

Upon executing the code, the following results are obtained.

- The message of (MetaTrader5 PKG version: 5.0.4424) was printed to the console
- The message of (MetaTrader5 PKG author:  MetaQuotes Ltd.) was printed to the console
- The message of (MT5 initialized Successfully) printed to the console
- The XAUUSD data printed to the console is the same as the following figure

![XAUUSD_data](https://c.mql5.com/2/88/XAUUSD_data.png)

- Opening the graph of XAUUSD in a browser as a result of plotting the same as the following

![XAUUSD_plot](https://c.mql5.com/2/88/XAUUSD_plot.png)

As illustrated in the preceding figure, a line graph depicts the closing prices of gold (XAUUSD) over time, spanning from August 2023 to the present. The aforementioned straightforward applications demonstrate the potential for utilizing Python with MQL5 in various aspects of automated trading, data analysis, and other related tasks.

### Conclusion

Python is a highly versatile and powerful programming language that can be used in a number of different fields, including trading and financial markets. The use of Python enables the automation, data analysis, and execution of strategies, which in turn facilitates more informed trading decisions. The combination of Python and MQL5 enables the creation of sophisticated trading systems that utilize data and machine learning. It represents a significant advancement in the field of algorithmic trading. It enables the creation of more adaptable and data-driven trading systems. This combination has the potential to enhance trading outcomes in financial markets.

This article demonstrated the use of Python in conjunction with MQL5. Furthermore, the article outlined the steps required to set up a Python environment for MetaTrader 5. By establishing an effective workflow, developers can more efficiently manage dependencies and enhance the scalability of their trading systems. Python is capable of automating tasks and analyzing market trends. This is demonstrated by practical applications, including opening MetaTrader 5, executing trades, retrieving asset data, and visualizing it.

It is my hope that this article will assist you in getting started with using Python with MQL5. Experimenting with various applications can be highly beneficial for both traders and developers. For further information, please refer to resources such as:

- Python for Dummies by Stef Maruch
- Python Crash Course by Eric Matthes
- Python All-In-One by John Shovic and Alan Simpson

Additionally, you can access further documentation on the MQL5 website through [https://www.mql5.com/en/docs/python\_metatrader5](https://www.mql5.com/en/docs/python_metatrader5) link.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/14135.zip "Download all attachments in the single ZIP archive")

[openMT5.py](https://www.mql5.com/en/articles/download/14135/openmt5.py "Download openMT5.py")(0.39 KB)

[openPosition.py](https://www.mql5.com/en/articles/download/14135/openposition.py "Download openPosition.py")(1.22 KB)

[getData.py](https://www.mql5.com/en/articles/download/14135/getdata.py "Download getData.py")(0.84 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [How to build and optimize a cycle-based trading system (Detrended Price Oscillator - DPO)](https://www.mql5.com/en/articles/19547)
- [How to build and optimize a volume-based trading system (Chaikin Money Flow - CMF)](https://www.mql5.com/en/articles/16469)
- [How to build and optimize a volatility-based trading system (Chaikin Volatility - CHV)](https://www.mql5.com/en/articles/14775)
- [Advanced Variables and Data Types in MQL5](https://www.mql5.com/en/articles/14186)
- [Building and testing Keltner Channel trading systems](https://www.mql5.com/en/articles/14169)
- [Building and testing Aroon Trading Systems](https://www.mql5.com/en/articles/14006)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/471455)**
(4)


![Dibea Koffi Badjo](https://c.mql5.com/avatar/avatar_na2.png)

**[Dibea Koffi Badjo](https://www.mql5.com/en/users/dibeakoffi)**
\|
31 Aug 2024 at 21:44

Very interesting, unfortunately  [metaTrader python](https://www.mql5.com/en/docs/python_metatrader5/mt5login_py "MQL5 Documentation: login function") library is not available yet on MacOs.

![](https://c.mql5.com/avatar/avatar_na2.png)

**\[Deleted\]**
\|
30 Nov 2024 at 14:31

Thanks,

just find an eror in the 3rd application

import plotly.express aspxcould not be resolved

from plotly.offline import plotcould not be resolved

i resolved this by reinstalling : pandas and  matplotlib

c:\\pip install MetaTrader5 pandas matplotlib

then modify code like this :

```
import MetaTrader5 as mt5
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Initialisation de MetaTrader 5
if not mt5.initialize():
    print("Erreur d'initialisation :", mt5.last_error())
    quit()
print("MT5 initialized Successfully")

# Définir les paramètres
symbol = "XAUUSD"
start_date = datetime(2023, 8, 1)
end_date = datetime(2024, 8, 12)

# Vérifier si le symbole est disponible
if not mt5.symbol_select(symbol, True):
    print(f"Le symbole {symbol} n'est pas disponible.")
    mt5.shutdown()
    quit()

# Récupérer les données historiques
rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_D1, start_date, end_date)
if rates is None:
    print("Erreur lors de la récupération des données :", mt5.last_error())
    mt5.shutdown()
    quit()

# Convertir les données en DataFrame pandas
data = pd.DataFrame(rates)
data['time'] = pd.to_datetime(data['time'], unit='s')  # Convertir les timestamps en datetime

# Afficher les premières lignes des données
print(data.head())

# Visualiser les données sous forme de graphique
plt.figure(figsize=(10, 6))
plt.plot(data['time'], data['close'], label=f"Prix de clôture {symbol}", color="blue")
plt.title(f"Prix de l'or ({symbol}) du {start_date.date()} au {end_date.date()}")
plt.xlabel("Date")
plt.ylabel("Prix")
plt.legend()
plt.grid()
plt.show()

# Déconnexion de MetaTrader 5
mt5.shutdown()
```

Hope it will help

![Gustavo Hennemann](https://c.mql5.com/avatar/avatar_na2.png)

**[Gustavo Hennemann](https://www.mql5.com/en/users/chuckero)**
\|
13 Jan 2025 at 06:15

**Omar Saghir [#](https://www.mql5.com/en/forum/471455#comment_55263437):**

Thanks,

just find an eror in the 3rd application

import plotly.express aspxcould not be resolved

from plotly.offline import plotcould not be resolved

i resolved this by reinstalling : pandas and  matplotlib

c:\\pip install MetaTrader5 pandas matplotlib

...

It is also possible to install Plotly library:

```
pip install plotly
```

More information:

[https://plotly.com/python/getting-started/](https://www.mql5.com/go?link=https://plotly.com/python/getting-started/ "https://plotly.com/python/getting-started/")

![Alexey Volchanskiy](https://c.mql5.com/avatar/2018/8/5B70B603-444A.png)

**[Alexey Volchanskiy](https://www.mql5.com/en/users/vdev)**
\|
22 Feb 2025 at 09:37

Thanks to the author, good example. It would be interesting to read an article about using machine learning tools such as scikit-learn, TensorFlow and Keras through MQL5+Python.

![Pattern Recognition Using Dynamic Time Warping in MQL5](https://c.mql5.com/2/89/logo-midjourney_image_15572_396_3823.png)[Pattern Recognition Using Dynamic Time Warping in MQL5](https://www.mql5.com/en/articles/15572)

In this article, we discuss the concept of dynamic time warping as a means of identifying predictive patterns in financial time series. We will look into how it works as well as present its implementation in pure MQL5.

![Building a Candlestick Trend Constraint Model (Part 8): Expert Advisor Development (I)](https://c.mql5.com/2/88/logo-midjourney_image_15321_390_3753__3.png)[Building a Candlestick Trend Constraint Model (Part 8): Expert Advisor Development (I)](https://www.mql5.com/en/articles/15321)

In this discussion, we will create our first Expert Advisor in MQL5 based on the indicator we made in the prior article. We will cover all the features required to make the process automatic, including risk management. This will extensively benefit the users to advance from manual execution of trades to automated systems.

![MQL5 Wizard Techniques you should know (Part 32): Regularization](https://c.mql5.com/2/90/logo-15576.png)[MQL5 Wizard Techniques you should know (Part 32): Regularization](https://www.mql5.com/en/articles/15576)

Regularization is a form of penalizing the loss function in proportion to the discrete weighting applied throughout the various layers of a neural network. We look at the significance, for some of the various regularization forms, this can have in test runs with a wizard assembled Expert Advisor.

![Reimagining Classic Strategies (Part IV): SP500 and US Treasury Notes](https://c.mql5.com/2/90/logo-15531_385_3705.png)[Reimagining Classic Strategies (Part IV): SP500 and US Treasury Notes](https://www.mql5.com/en/articles/15531)

In this series of articles, we analyze classical trading strategies using modern algorithms to determine whether we can improve the strategy using AI. In today's article, we revisit a classical approach for trading the SP500 using the relationship it has with US Treasury Notes.

[We've created a channel for MQL5 developersFollow MQL5.community on social media and be the first to receive important updatesLearn more![](https://www.mql5.com/ff/sh/a83xrgctr82w45z9z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=pwgdbvtemvkwsfltqonysypfvtrtufji&s=e99a66a1660cd810b1edbac65597df695e2c2220d1e937834f402f9aeabd4289&uid=&ref=https://www.mql5.com/en/articles/14135&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5062632115700409834)

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