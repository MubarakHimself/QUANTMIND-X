---
title: Price Driven CGI Model: Theoretical Foundation
url: https://www.mql5.com/en/articles/14964
categories: Integration
relevance_score: 0
scraped_at: 2026-01-24T14:03:49.514703
---

[![](https://www.mql5.com/ff/sh/rvgkjnsrvj1mzh89z2/01.png)Best VPS for tradersTwo-click launch from MetaTrader, minimum ping to broker, 15 USD/monthLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=wpjhvzsogglsviotmypjoyhhtuxlrzhi&s=aa6c5782a1658c2f617954d478dea9989a27ae26ecabc09d0ab1204277fdf8e3&uid=&ref=https://www.mql5.com/en/articles/14964&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5083310974481471764)

MetaTrader 5 / Examples


- [Introduction](https://www.mql5.com/en/articles/14964#para1)
- [Mathematical Aspect](https://www.mql5.com/en/articles/14964#para2)

  - [Linear Relationship](https://www.mql5.com/en/articles/14964#subpara1)
  - [Exponential Relationship](https://www.mql5.com/en/articles/14964#subpara2)

- [What do we need?](https://www.mql5.com/en/articles/14964#para3)
- [Exporting price data from MetaTrader 5.](https://www.mql5.com/en/articles/14964#para4)
- [Normalizing price data exported from MetaTrader 5](https://www.mql5.com/en/articles/14964#para5)

  - [Installing Pandas](https://www.mql5.com/en/articles/14964#subpara3)
  - [Python script for data manipulation](https://www.mql5.com/en/articles/14964#subpara4)
  - [Normalized Data](https://www.mql5.com/en/articles/14964#subpara5)

- [Visualizing the idea.](https://www.mql5.com/en/articles/14964#para6)
- [Significance of the Price Man project.](https://www.mql5.com/en/articles/14964#para7)
- [Conclusion](https://www.mql5.com/en/articles/14964#para8)

### Introduction

Studying various MQL5 community articles has sparked my creativity in using MQL5 for algorithmic trading systems. Combining MQL5 with Python, C++, and Java enables innovative financial solutions. This project involves creating a price-driven CGI character, visualizing market emotions like optimism and fear through CGI object size changes.

Behavioral Finance examines how psychological factors impact market behaviors. Key concepts include Prospect Theory, which notes investors' tendency to feel losses more intensely than gains. The project integrates MQL5 with Python and creative software for CGI animation, utilizing open-source programs to keep it simple and engaging. We will explore the potential of MetaTrader 5 price data, aiming to inspire data scientists and algorithmic traders.

Below is an animated CGI character, created with Inkscape and animated in Blender 3D, whose size changes in response to market data-driven algorithms.

![Price Man CGI](https://c.mql5.com/2/83/blender_iT4WqG19Zh.gif)

### Mathematical Aspect

Before delving deep in the topic, I have considered formulating the mathematical functions that guide us through the project. This will aid us with a calculated reference even when we start writing our algorithm. The factors that change in our project are the size of our sim object and price. Price change is the effector of the object size.

To relate price changes to the size of a CGI object, we use mathematical functions that map price movements to object scale.

Let's define variables useful for the project:

> 1. Price Change (Δ𝑃): Change in asset price over time.
> 2. Initial Size (𝑆0): Starting size of the CGI object.
> 3. New Size (𝑆𝑡): Size of the CGI object at time 𝑡.
> 4. Scaling Factor (𝛼): Controls sensitivity of size to price changes.

There are two possible mathematical relationships to lay for the project. These are linear relationship and exponential relationship. Each of these has got its advantages and suitable application.

1. Linear relationship is simple and direct, suitable for small or moderate price changes.
2. Exponential relationship captures significant changes more dramatically, thus suitable for complexity.

Linear relationship:

![Linear Relationship](https://c.mql5.com/2/83/Linear_Relationship.PNG)

Exponential relationship:

![Exponential Relationship](https://c.mql5.com/2/83/Exponential_Relationship.PNG)

By substituting the given values into the formula, we can calculate the size of the object whenever price changes.

### What do we need?

At this stage where we are conceptualizing the idea, it is important to be familiar with tools that will make the project possible. I have listed the tools that are essentials for this project and description:

| Software Tool | Description |
| --- | --- |
| [MetaTrader 5](https://www.metatrader5.com/en/download "https://www.metatrader5.com/en/download") | Powerful platform for trading, and obtaining price data, and it also comes with the Strategy Tester and a professional Meta Editor for writing programs required in this project. |
| [Notepad++](https://www.mql5.com/go?link=https://notepad-plus-plus.org/downloads/ "https://notepad-plus-plus.org/downloads/")(Optional) | Another optional code editor for writing program of different syntax, it is open source, you can choose a different editor that suits from the internet. |
| [Blender 3D](https://www.mql5.com/go?link=https://www.blender.org/download/ "https://www.blender.org/download/") (Optional, but recommended) | For 3D modelling, Animation and Simulation. Advantage it contains Blender Python which can handle pythons scripts that we shall use to drive our Price Man character. |
| [Inkscape](https://www.mql5.com/go?link=https://inkscape.org/ "https://inkscape.org/") (Optional) | Powerful 2D Vector Image editor. Open-source software. |

### Exporting price data from MetaTrader 5

It is possible to export data from MetaTrader 5 manually by pressing key combination Ctrl + U at the same time on Windows to bring the symbols list. Select the desired symbol from specification and highlight it, then go to bars tabs set the timeframe and the dates which you want data from and click the request button on the same window. Data of price will be listed from start to end according to the settings made. Below the window is an option for exporting data, and the format will be CSV. The data sheet can be read using Google Docs or an Excel software. Below is an image showing the process done on Volatility 75(1s) Index by [Deriv.com](https://www.mql5.com/go?link=https://track.deriv.com/_r6xDODPy3Ly2vdm9PpHVCmNd7ZgqdRLk/1/ "visit Deriv") website.

![Exporting Price Data From MetaTrader 5](https://c.mql5.com/2/83/Exporting_Price_Data_on_MetaTrade_5.gif)

The data exported need further manipulation to make it suit our desired usage, for example we only need OPEN and CLOSE price to be used for keyframing the growth of our object. The method above has got limited customization to the data being exported. We only managed to set the timeframe and dates for the data, while everything else remained default. Some parts of the data are not necessary in our project, in the next segment will delve into setting up Pandas for normalizing the data file we exported using python.

### Normalizing the Price data exported from Mt5

The problem presented above by manually exporting data can be addressed by developing a specialized script for the job. This script must be capable to focus on required data and represent it hastily when being used in other software programs. Scripts can be customized to include specific parameters and settings tailored to your needs, such as time frames, symbols, and data formats. Scripts can be designed to adapt to different data sets or conditions dynamically, providing more flexibility to the exported data. At this segment, we want to create a Python script for manipulating the exported _CSV_ price data file and saving it in a specific location in the computer.

Make sure you install python in your computer as a priority because the processes below requires it installed. It is free to download from [python.org](https://www.mql5.com/go?link=https://www.python.org/downloads/ "https://www.python.org/downloads/") website.

### Installing Pandas:

Pandas, is a powerful open-source data analysis and manipulation library for Python. It provides easy-to-use data structures and data analysis tools that are essential for working with structured data, such as CSV files, Excel spreadsheets, databases, and more. I am using a Windows computer, but I will explain how to set up on other platforms.

We use Python's package manager, _pip,_ to automatically install libraries, provided the computer is on a network with internet access.

- On Windows, using command prompt or powershell, run the command below.

```
pip install pandas
```

- On macOS, run the same command in the Terminal.
- Using Linux (Ubuntu\\Debian) the command works is executed in its Terminal.

### Python script for data manipulation:

Now, let's look at the program script for modifying our data. The program reads a CSV file exported from MetaTrader 5, which contains tab-separated columns. It identifies and normalizes the 'Open' and 'Close' prices found in columns '<OPEN>' and '<CLOSE>', respectively. After normalization, it saves these normalized prices to a new CSV file, making the data more suitable for further analysis, visualization, or integration into other applications in this case we are going to use Blender for 3D visualization tasks.

The mathematical formula governing the program is:

![Formula for normalizing data](https://c.mql5.com/2/83/Normalizing_exported_Data.PNG)

If both _'<OPEN>'_ and _'<CLOSE>'_ columns are found, the program proceeds to normalize these prices. Normalization involves finding the minimum _(min\_price)_ and maximum _(max\_price)_ values of both _'Open'_ and _'Close'_ prices across the entire dataset. Scaling the _'Open'_ and _'Close'_ prices to a normalized range between 0 and 1 using the above formula. This normalization step ensures that the prices are adjusted to a consistent scale, which can be useful for various analytical or visualization purposes.

After normalization, the program creates a new DataFrame _(df\[\['Open\_norm', 'Close\_norm'\]\])_ containing only the normalized _'Open'_ and _'Close_' prices. It then saves this normalized data to a new CSV file _(normalized\_data.csv)_ in the specified directory _(r"C:\\Users\path to where you want to save the file"\\")_. Finally, the program prints a message confirming where the normalized data CSV file has been saved _(f"Normalized data saved to {output\_csv\_file\_path}")._

I have put together the code here, and you can refer to the summary and formula above to:

```
import pandas as pd

# Path to the CSV file exported from MetaTrader 5. Note that you can customize the paths to match your end
csv_file_path = r"C:\Users\path to the storage location of your CSV file\Volatility 75 (1s) Index_M1_202407050000_202407060000.csv"

# Read the CSV file into a DataFrame with tab delimiter
df = pd.read_csv(csv_file_path, delimiter='\t')

# Print the column names to debug
print("Column names in CSV:", df.columns)

# Check if 'OPEN' and 'CLOSE' columns exist
if '<OPEN>' in df.columns and '<CLOSE>' in df.columns:
    # Normalize the Open and Close prices
    min_price = min(df['<OPEN>'].min(), df['<CLOSE>'].min())
    max_price = max(df['<OPEN>'].max(), df['<CLOSE>'].max())

    df['Open_norm'] = (df['<OPEN>'] - min_price) / (max_price - min_price)
    df['Close_norm'] = (df['<CLOSE>'] - min_price) / (max_price - min_price)

    # Save the normalized data to a new CSV file
    output_csv_file_path = r"C:\Users\paths to where the normalized file will be saved\normalized_data.csv"
    df[['Open_norm', 'Close_norm']].to_csv(output_csv_file_path, index=False)

    print(f"Normalized data saved to {output_csv_file_path}")
else:
    print("'OPEN' and/or 'CLOSE' columns not found in CSV file. Available columns are:", df.columns)
```

To run the program, open the command prompt on Windows and make the sure it is open to the storage location where the script is saved. Run the command below.

```
python ModifyPriceData.py
```

### Normalized Data:

Let's look at the presentation of  data normalization as view in Excel.

Before Normalization:

![Original data CSV file from MetaTrader5](https://c.mql5.com/2/83/Original_data_CSV_file_from_MetaTrader5.PNG)

### **After Normalizing:**

The processed output file presented differently with fewer specified details according to the script we created.

![Normalized data](https://c.mql5.com/2/83/normalized_data.PNG)

### Visualizing the idea:

I started by creating my character Price Man in Inkscape. I decided to use simple shape geometry, which makes it easy for calculations at the current stage of the project.

![Realtime price changing the scale of Price Man](https://c.mql5.com/2/83/blender_MGJdxgM8Cd.gif)

The future of MetaTrader 5 can have a panel with the character responding to price change in real time, as demonstrated in the imaginary image below.

![The PriceMan floating on MetaTrader 5 interface.](https://c.mql5.com/2/83/blender_CuGz64IHjg.gif)

### Significance of the Price Man project

By linking market price changes to the size of CGI objects, we can create a visual representation of market emotions. This approach helps bridge the gap between abstract financial data and human behavior, making market movements more accessible and engaging for various audiences. I have outline practical applications of a successful development below and designated institutions:

1. Educational Tools: Visualize market movements to teach financial concepts.
2. Investor Tools: Help investors quickly gauge market sentiment.
3. Entertainment and Media: Use visualizations in news programs or social media to engage audiences.
4. Art and Creativity: Create dynamic digital art that evolves with market changes.

### Conclusion

It is possible to develop data manipulation algorithm tools. Price data can be optimized for use in other projects that respond to price changes. Replay systems can be modeled to re-analyze past market behavior, which is useful for institutions studying financial markets. With the algorithm mentioned earlier, we were able to edit and perform mathematical calculations on over a thousand values, a task that is difficult to do manually. The same algorithm can be adapted for more complex price data performance based on the specific projects you are pursuing.

We will continue refining the Price Man concept and utilize collaborative software to bring the project to fruition. Our goal is to integrate the Price Man character panel into the MetaTrader 5 platform, making it responsive to real-time price fluctuations. This sets the stage for the next phase of development beyond our current progress. I have attached the files listed in the table below if you are keen to got through them to see how it works.

In the next article, we will utilize the Blender Python library to animate simulations for the Price Man CGI character using the normalized data. Familiarity with the software programs used is essential, along with the skill of exporting from one software and importing into another. Feel free to share additional ideas and contributions in the comments below.

| Attachment | Description |
| --- | --- |
| Volatility 75 (1s) Index\_M1\_202407050000\_202407060000.csv | Price data file exported from MetaTrader 5. |
| Normalized\_data.csv | The file with data intended for use in further exploration of the idea. |
| ModifyPriceData.py | Algorithm designed to normalize price data. |
| Price Man.png | The image file of our CGI model |

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/14964.zip "Download all attachments in the single ZIP archive")

[Volatility\_75\_n1s6\_Index\_M1\_202407050000\_202407060000.csv](https://www.mql5.com/en/articles/download/14964/volatility_75_n1s6_index_m1_202407050000_202407060000.csv "Download Volatility_75_n1s6_Index_M1_202407050000_202407060000.csv")(87.31 KB)

[normalized\_data.csv](https://www.mql5.com/en/articles/download/14964/normalized_data.csv "Download normalized_data.csv")(55.56 KB)

[ModifyPriceData.py](https://www.mql5.com/en/articles/download/14964/modifypricedata.py "Download ModifyPriceData.py")(1.22 KB)

[Price\_Man.png](https://www.mql5.com/en/articles/download/14964/price_man.png "Download Price_Man.png")(6.66 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [From Novice to Expert: Higher Probability Signals](https://www.mql5.com/en/articles/20658)
- [From Novice to Expert: Navigating Market Irregularities](https://www.mql5.com/en/articles/20645)
- [From Novice to Expert: Automating Trade Discipline with an MQL5 Risk Enforcement EA](https://www.mql5.com/en/articles/20587)
- [From Novice to Expert: Trading the RSI with Market Structure Awareness](https://www.mql5.com/en/articles/20554)
- [From Novice to Expert: Developing a Geographic Market Awareness with MQL5 Visualization](https://www.mql5.com/en/articles/20417)
- [The MQL5 Standard Library Explorer (Part 5): Multiple Signal Expert](https://www.mql5.com/en/articles/20289)
- [The MQL5 Standard Library Explorer (Part 4): Custom Signal Library](https://www.mql5.com/en/articles/20266)

**[Go to discussion](https://www.mql5.com/en/forum/469867)**

![Cascade Order Trading Strategy Based on EMA Crossovers for MetaTrader 5](https://c.mql5.com/2/84/Cascade_Order_Trading_Strategy_Based_on_EMA_Crossovers___LOGO.png)[Cascade Order Trading Strategy Based on EMA Crossovers for MetaTrader 5](https://www.mql5.com/en/articles/15250)

The article guides in demonstrating an automated algorithm based on EMA Crossovers for MetaTrader 5. Detailed information on all aspects of demonstrating an Expert Advisor in MQL5 and testing it in MetaTrader 5 - from analyzing price range behaviors to risk management.

![Creating an Interactive Graphical User Interface in MQL5 (Part 2): Adding Controls and Responsiveness](https://c.mql5.com/2/84/Creating_an_Interactive_Graphical_User_Interface_in_MQL5_0Part_2v___LOGO.png)[Creating an Interactive Graphical User Interface in MQL5 (Part 2): Adding Controls and Responsiveness](https://www.mql5.com/en/articles/15263)

Enhancing the MQL5 GUI panel with dynamic features can significantly improve the trading experience for users. By incorporating interactive elements, hover effects, and real-time data updates, the panel becomes a powerful tool for modern traders.

![Portfolio Optimization in Python and MQL5](https://c.mql5.com/2/84/Portfolio_Optimization_in_Python_and_MQL5__LOGO.png)[Portfolio Optimization in Python and MQL5](https://www.mql5.com/en/articles/15288)

This article explores advanced portfolio optimization techniques using Python and MQL5 with MetaTrader 5. It demonstrates how to develop algorithms for data analysis, asset allocation, and trading signal generation, emphasizing the importance of data-driven decision-making in modern financial management and risk mitigation.

![Using JSON Data API in your MQL projects](https://c.mql5.com/2/83/Using_Json_Data_API_in_your_MQL_projects__LOGO.png)[Using JSON Data API in your MQL projects](https://www.mql5.com/en/articles/14108)

Imagine that you can use data that is not found in MetaTrader, you only get data from indicators by price analysis and technical analysis. Now imagine that you can access data that will take your trading power steps higher. You can multiply the power of the MetaTrader software if you mix the output of other software, macro analysis methods, and ultra-advanced tools through the ​​API data. In this article, we will teach you how to use APIs and introduce useful and valuable API data services.

[![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/01.png)![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/02.png)Powerful analytics for traders of any levelAll the necessary trading reports for beginners and professionals](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/454106&a=muccpajyfystoakuukdobwigjejzmpqn&s=52daad60fa795e635264e6f94898f05493bca3b5124d4cca8eb7e82333c2ef12&uid=&ref=https://www.mql5.com/en/articles/14964&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5083310974481471764)

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