---
title: Price Action Analysis Toolkit Development (Part 9): External Flow
url: https://www.mql5.com/en/articles/16967
categories: Trading Systems, Integration
relevance_score: 3
scraped_at: 2026-01-23T18:42:27.296076
---

[![](https://www.mql5.com/ff/sh/qv94j0cd8n2n55z9z2/01.png)![](https://www.mql5.com/ff/sh/qv94j0cd8n2n55z9z2/02.png)Boost your trading experienceRead our book "MQL5 Programming for Traders"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/book%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.algobook%26utm_content=visit.page%26utm_campaign=algobook.promo.04.2024&a=heclgjpfbvfghpmyaciuaesdtswflupo&s=4255fbe1b8cbc4d1b40afbaebf4235e5ace8b5103cba60d996897a03d588556f&uid=&ref=https://www.mql5.com/en/articles/16967&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5069674302992746721)

MetaTrader 5 / Trading systems


### Introduction

Financial markets can be complex, often presenting challenges for traders and analysts in processing and interpreting data accurately. To overcome these challenges, various libraries has been developed to simplify market analysis, each designed to handle data in ways that align with specific goals. While MQL5 libraries are commonly used to create trading strategies and indicators, external libraries, such as Python’s data analytics tools, provide additional resources for more advanced and in-depth analysis.

In previous [articles](https://www.mql5.com/en/articles/16434), I focused on price action and data analysis, primarily exploring signal generation through calculations and metrics processed by MQL5 Expert Advisors (EAs). These [articles](https://www.mql5.com/en/articles/15927), however, were confined to MQL5-based analysis. This article marks a shift by introducing the integration of external libraries specifically designed for advanced data analysis. By leveraging the Python library, Pandas, we can explore new avenues for traders, providing a broader range of analytical options to cater to various needs.

This method does not replace MQL5 but positions it as the core of this system. The MQL5 EA serves as a bridge, enabling interaction with external libraries and servers to build a more robust and flexible analytical framework. I’d like to emphasize that this article is only the beginning of exploring advanced analytics, and I look forward to presenting more comprehensive insights and techniques in future discussions.

Let’s take a look at how we will approach this task:

- [Strategy overview](https://www.mql5.com/en/articles/16967#para2)
- [Python](https://www.mql5.com/en/articles/16967#para3)
- [Main Functions Of The MQL5 EA](https://www.mql5.com/en/articles/16967#para4)
- [Outcomes](https://www.mql5.com/en/articles/16967#para5)
- [Conclusion](https://www.mql5.com/en/articles/16967#para6)

### Strategy Overview

The main goal of this system is to generate a trading signal using Python’s Pandas library, leveraging its data interpretation features. In this section, we will take a closer look at how the core logic is executed by the system. The overall exchange occurs between the MQL5 Expert Advisor (EA) and Python, with various processes happening in between before the signal is ultimately displayed on the chart. Let's follow the steps outlined below to understand how the system works.

1\. MQL5 EA to Python:

- The MQL5 Expert Advisor gather previous day's high, low, open, close, and volume of 10 days.
- Sends data to Python Server.
- Format the data into a CSV string and send it via an HTTP POST request to the Python server.

2\. Python:

- Python receives data and process the received data using libraries like Pandas.
- Analyze the data to generate a trading signal, average price, and volume, then create an explanation.

3\. Python to EA:

- Python sends the generated signal, average price, volume, and explanation back to the MQL5 EA via HTTP response.

4\. MQL5 Expert Advisor:

- The MQL5 EA receives and parse response, extract signal, average price, average volume, and explanation from the response.
- If the signal is different, update the chart with the new signal and display it.

Let's refer to the following diagram for more insight.

![Intergration Flow](https://c.mql5.com/2/112/MQL5_Python_Intergration.png)

Fig 1. Logic Chart

### Python

Python is a versatile programming language recognized for its simplicity and wide range of applications. It is commonly used for data analysis, machine learning, web development, and automation, among other tasks. In this system, Python is crucial for performing advanced data analysis, particularly through libraries such as Pandas, which provide efficient tools for manipulating and analyzing data. Python also creates a server for interaction between MQL5 and its libraries. By utilizing Python to process complex datasets, the system generates trading signals based on historical market data, which are then passed to the MQL5 Expert Advisor (EA) to aid in live trading decisions.

I will guide you through the process of installing Python, creating a script, and running it step by step

- Download Python Installer by going to the official [Python website](https://www.mql5.com/go?link=https://www.python.org/downloads/ "https://www.python.org/downloads/")
-  Run the Installer

Important: Make sure to check the box that says, "Add Python to PATH" before clicking on Install Now. This makes it easier to run Python from the command line.

-  Complete the Installation

The installer will begin the installation process. Wait for it to finish. Once the installation is complete, click Close.

-  Verify the Installation

Open the Command Prompt (press Windows + R, type cmd, and press Enter). Type _python --version_ and press Enter. If Python is installed correctly, you should see the version of Python displayed (e.g., Python 3.x.x).

After successful installation of python, now consider installing flask and pandas. Flask is a lightweight, web framework for Python used to build web applications. It is designed to be simple, flexible, and easy to use. Flask is being used to set up a simple web server that allows communication between the MQL5 Expert Advisor (EA) and the Python data analysis tools. The EA sends data to Flask (via HTTP requests), which processes the data using Python libraries, in this case Pandas and returns a trading signal to the EA.

To install Flask and Pandas using the Command Prompt, follow these steps:

Open Command Prompt

- Press Windows + R, type cmd, and press Enter.

Ensure pip  (Python Package Installer) is Installed

- First, check if pip (Python’s package manager) is installed by running:

```
pip --version
```

- If pip is installed, you will see the version number. If it's not installed, you can follow the steps below to install it. PIP (Python Package Installer) is usually installed automatically with Python. If PIP is not installed, download [get-pip.py](https://www.mql5.com/go?link=https://bootstrap.pypa.io/get-pip.py "https://bootstrap.pypa.io/get-pip.py"), and run it using the command python get-pip.py.

Install Flask

- To install Flask, run the following command in Command Prompt:

```
pip install Flask
```

- Wait for the installation to complete. Flask will be installed and ready to use.

Install Pandas

- To install Pandas, run this command:

```
pip install pandas
```

- Similarly, wait for the installation to complete.

Verify Installation

- After installing both packages, you can verify the installation by running these commands:

```
python -c "import flask; print(flask.__version__)"

python -c "import pandas; print(pandas.__version__)"
```

- This should print the installed versions of Flask and Pandas, confirming that they have been installed successfully.

You can now proceed to create a Python script. Personally, I prefer using Notepad++ for this task. To create the script, open Notepad++, start a new file, and set the language to Python by selecting it from the Language menu. Once you’ve written your script, save it in a directory that is easy to locate. Ensure you save the file with a .py extension, which identifies it as a Python script.

Python Script

```
import pandas as pd
import flask
from flask import request, jsonify

app = flask.Flask(__name__)
app.config["DEBUG"] = True

@app.route('/analyze', methods=['POST'])
def analyze_csv():
    try:
        # Read CSV data from the POST request
        csv_data = request.data.decode('utf-8')

        # Write the CSV data to a file (optional, for debugging)
        with open('received_data.csv', 'w') as file:
            file.write(csv_data)

        # Load the CSV data into a DataFrame
        from io import StringIO
        data = StringIO(csv_data)
        df = pd.read_csv(data)

        # Ensure the CSV has the correct columns
        required_columns = ['date', 'prev_high', 'prev_low', 'prev_open', 'prev_close', 'prev_volume']
        for column in required_columns:
            if column not in df.columns:
                return jsonify({"error": f"Missing column: {column}"}), 400

        # Print the received metrics for debugging
        print("Received metrics:")
        print(df)

        # Perform analysis (Example: Calculate average price and volume)
        df['average_price'] = (df['prev_high'] + df['prev_low'] + df['prev_open'] + df['prev_close']) / 4
        average_price = df['average_price'].mean()  # Average of all the average prices
        average_volume = df['prev_volume'].mean()  # Average volume

        # Print the computed averages
        print(f"Average Price: {average_price}")
        print(f"Average Volume: {average_volume}")

        # Create a trading signal based on a simple rule
        last_close = df['prev_close'].iloc[-1]
        if last_close > average_price:
            signal = "BUY"
            signal_explanation = f"The last close price ({last_close}) is higher than the average price ({average_price})."
        else:
            signal = "SELL"
            signal_explanation = f"The last close price ({last_close}) is lower than the average price ({average_price})."

        # Print the signal and explanation
        print(f"Generated Signal: {signal}")
        print(f"Signal Explanation: {signal_explanation}")

        # Return the signal as JSON
        return jsonify({
            "signal": signal,
            "average_price": average_price,
            "average_volume": average_volume,
            "signal_explanation": signal_explanation
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='189.7.6.8', port=5877)
```

To run the script, open the Command Prompt on your computer.

Run the command:

```
cd C:\Users\pathway to your python script folder
```

Followed by a command:

```
python filename.py
```

Use the exact name you assigned to your script when executing it. Once the script runs, it will indicate that the port is actively listening.

```
 Running on http://189.7.6.8:5877
```

### Main Functions

Initialization ( _OnInit_)

This function runs when the EA is first initialized in the MetaTrader platform. It is used to set up any resources or configurations needed by the EA. In this case, it simply prints a message to the log that indicates the EA is ready to start interacting with the Python server.

```
int OnInit()
{
   Print("Expert initialized. Ready to send data to Python.");
   return(INIT_SUCCEEDED);
}
```

The _INIT\_SUCCEEDED_ return value signifies that the initialization was successful.

Deinitialization ( _OnDeinit)_

This function is triggered when the EA is removed or the MetaTrader platform is closed. It's typically used for cleanup operations, such as releasing resources or closing open files. Here, it simply prints a message indicating that the EA has been _deinitialized_ and is no longer active.

```
void OnDeinit(const int reason)
{
   Print("Expert deinitialized.");
}
```

_OnTick_ (Core Functionality)

This is the main function of the EA that gets executed every time the market conditions change (on every new tick).

```
void OnTick()
{
   // Check if enough time has passed since the last signal update
   if(TimeCurrent() - lastSignalTime < signalInterval)
   {
      return;  // Skip if it's too soon to update
   }

   // Collect data and prepare CSV for Python
   string csvData = "date,prev_high,prev_low,prev_open,prev_close,prev_volume\n";

   // Get the previous trend data for the last `trendDays`
   for(int i = 1; i <= 10; i++)  // You can adjust the trendDays here
   {
      datetime prevDate = iTime(Symbol(), PERIOD_D1, i);
      double prevHigh = iHigh(Symbol(), PERIOD_D1, i);
      double prevLow = iLow(Symbol(), PERIOD_D1, i);
      double prevOpen = iOpen(Symbol(), PERIOD_D1, i);
      double prevClose = iClose(Symbol(), PERIOD_D1, i);
      long prevVolume = iVolume(Symbol(), PERIOD_D1, i);

      csvData += StringFormat("%s,%.5f,%.5f,%.5f,%.5f,%ld\n",
                              TimeToString(prevDate, TIME_DATE | TIME_MINUTES),
                              prevHigh, prevLow, prevOpen, prevClose, prevVolume);
   }

   // Save data to CSV file
   string fileName = StringFormat("%s_analytics.csv", Symbol());
   int fileHandle = FileOpen(fileName, FILE_WRITE | FILE_CSV | FILE_ANSI);
   if(fileHandle != INVALID_HANDLE)
   {
      FileWriteString(fileHandle, csvData);
      FileClose(fileHandle);
      Print("CSV file created: ", fileName);
   }
}
```

It also executes the following operations:

- Signal Interval Check: The EA first checks if enough time has passed since the last signal update using _TimeCurrent_(). If not, it skips processing.
- Data Collection: The EA collects market data for the last 10 days (or more, if you change the value in the loop) including: Previous day’s high, low, open, close, and volume.
- Data Formatting: It then formats this data into a CSV format for easy transmission to the Python server.
- CSV File Saving: The data is saved as a .csv file with the name <symbol>\_analytics.csv to the disk. If the file creation and writing succeed, a success message is printed.

HTTP Request to Python Server ( _WebRequest_)

```
string headers = "Content-Type: application/json\r\n";
char result[];
string resultHeaders;
int responseCode = WebRequest(
    "POST",                  // HTTP method
    pythonUrl,               // URL
    headers,                 // Custom headers
    timeout,                 // Timeout in milliseconds
    data,                    // Data to send
    result,                  // Response content
    resultHeaders            // Response headers
);

if(responseCode == 200)
{
    string response = CharArrayToString(result);
    Print("Received response: ", response);
}
else
{
    Print("Error: HTTP request failed with code ", responseCode);
}
```

After preparing the data in CSV format, the EA sends this data to a Python server via HTTP POST.

- Headers: The Content-Type header is set to application/ _json_, which tells the server that the data being sent is in JSON format.
- WebRequest: The _WebRequest_ function is used to send the HTTP POST request to the Python server. The function returns:

> 1) _responseCode_: The HTTP response code (e.g., 200 for success).

> 2) result: The server's response content (which is typically the analysis results).
>
> ![WebRequest](https://c.mql5.com/2/112/WebRequest_Flow.png)
>
> Fig 2. WebRequest Flow Chart

- If the request is successful ( _responseCode_ == 200), the response content is converted from a char array to a string, and a success message is printed. If the request fails, an error message is displayed.

Parsing and Displaying Response

```
if(responseCode == 200)
{
    string signal = "";
    string avgPrice = "";
    string avgVolume = "";
    string explanation = "";

    // Extract signal, avgPrice, avgVolume, and explanation from the response
    int signalStart = StringFind(response, "\"signal\":");
    int signalEnd = StringFind(response, "\"average_price\":");
    int explanationStart = StringFind(response, "\"signal_explanation\":");
    int avgPriceStart = StringFind(response, "\"average_price\":");
    int avgVolumeStart = StringFind(response, "\"average_volume\":");

    if(signalStart != -1 && signalEnd != -1)
    {
        signal = StringSubstr(response, signalStart + 10, signalEnd - signalStart - 12);
    }

    if(explanationStart != -1)
    {
        explanation = StringSubstr(response, explanationStart + 23, StringFind(response, "\"", explanationStart + 23) - (explanationStart + 23));
    }

    if(avgPriceStart != -1)
    {
        avgPrice = StringSubstr(response, avgPriceStart + 16, StringFind(response, "\"", avgPriceStart + 16) - (avgPriceStart + 16));
    }

    if(avgVolumeStart != -1)
    {
        avgVolume = StringSubstr(response, avgVolumeStart + 18, StringFind(response, "\"", avgVolumeStart + 18) - (avgVolumeStart + 18));
    }

    // Update the chart if the signal has changed
    if(signal != lastSignal)
    {
        lastSignal = signal;
        lastSignalTime = TimeCurrent();  // Update last signal time
        string receivedSummary = "Signal: " + signal + "\n" +
                                 "Avg Price: " + avgPrice + "\n" +
                                 "Avg Volume: " + avgVolume + "\n" +
                                 "Explanation: " + explanation;
        Print("Received metrics and signal: ", receivedSummary);
        Comment(receivedSummary);  // Display it on the chart
    }
}
```

After receiving the response from the Python server, the EA parses the response to extract key data, such as:

- Signal: The trading signal (e.g., "buy" or "sell").
- Avg Price: The average price derived from the analysis.
- Avg Volume: The average volume of trades.
- Explanation: An explanation of why the signal was generated.

The function uses _StringFind_ and _StringSubstr_ to extract these values from the response string.

If the signal has changed since the last update (signal != _lastSignal_), it:

> 1) Updates the _lastSignal_ and _lastSignalTime_ variables.

> 2) Displays the new signal, average price, volume, and explanation as a comment on the chart using the Comment() function.

Signal Update and Display

This part is integrated into the previous step, where the signal is updated and displayed on the chart.

```
if(signal != lastSignal)
{
    lastSignal = signal;
    lastSignalTime = TimeCurrent();  // Update last signal time
    string receivedSummary = "Signal: " + signal + "\n" +
                             "Avg Price: " + avgPrice + "\n" +
                             "Avg Volume: " + avgVolume + "\n" +
                             "Explanation: " + explanation;
    Print("Received metrics and signal: ", receivedSummary);
    Comment(receivedSummary);  // Display it on the chart
}
```

If the signal has changed (i.e., it differs from the previous one), the EA:

- Updates the _lastSigna_ l and _lastSignalTime_ variables.
- Creates a string summary containing the signal, average price, average volume, and explanation.
- Displays the summary on the chart as a comment and prints it to the log.

_CharArrayToString_ (Utility Function)

```
string CharArrayToString(char &arr[])
{
    string result = "";
    for(int i = 0; i < ArraySize(arr); i++)
    {
        result += StringFormat("%c", arr[i]);
    }
    return(result);
}
```

This utility function is used to convert a char array (received from the HTTP response) into a string. It loops through each element of the char array and appends the corresponding character to the result string.

Each of these steps is designed to handle a specific part of the process: collecting data, sending it to the Python server, receiving the analysis, and updating the chart with the trading signal. The approach ensures that the EA can operate autonomously, gather relevant market data, and make decisions based on the Python-powered analytics.

### Outcomes

The first step is to confirm that your Python script is running and actively listening on the required server. For detailed instructions on setting up and running the script, refer to the Python section above. If it's actively listening, it should write:

```
Running on http://189.7.6.8:5877
```

Please note that the API and host mentioned above are not the actual ones used but were generated for educational purposes. Next, we proceed to initiate the MQL5 EA. If the connection between MQL5 and the Python server is successfully established, you will see logging messages in the 'Experts' tab of the chart. Additionally, the Python script running in the command prompt will display the received metrics.

The Command Prompt will display the following:

```
189.7.6.8 - - [21/Jan/2025 10:53:44] "POST /analyze HTTP/1.1" 200 -
Received metrics:
                date  prev_high  prev_low  prev_open  prev_close  prev_volume
0   2025.01.20 00:00    868.761   811.734    826.389     863.078      83086.0
1   2025.01.19 00:00    856.104   763.531    785.527     826.394      82805.0
2   2025.01.18 00:00    807.400   752.820    795.523     785.531      82942.0
3   2025.01.17 00:00    886.055   790.732    868.390     795.546      83004.0
4   2025.01.16 00:00    941.334   864.202    932.870     868.393      83326.0
5   2025.01.15 00:00    943.354   870.546    890.620     932.876      83447.0
6   2025.01.14 00:00    902.248   848.496    875.473     890.622      83164.0
7   2025.01.13 00:00    941.634   838.520    932.868     875.473      82516.0
8   2025.01.12 00:00    951.350   868.223    896.455     932.883      83377.0
9   2025.01.11 00:00    920.043   857.814    879.103     896.466      83287.0
10               NaN        NaN       NaN        NaN         NaN          NaN
```

The information above will be used by pandas for analysis and signal generation. The tenth day's data shows 'NaN' because the day has not yet closed, meaning the analysis primarily relies on values obtained from previous days. However, it also incorporates the current price levels of the tenth day, which are incomplete. Below, you can find the logging and analysis results displayed by pandas in the command prompt (CMD).

```
Average Price: 865.884525
Average Volume: 83095.4
Generated Signal: SELL
Signal Explanation: The last close price (nan) is lower than the average price (865.884525).
```

The MetaTrader 5 will display the following:

Let's begin by checking the logging in the 'Experts' tab. Refer below to view the results obtained.

```
2025.01.21 10:50:28.106 External Flow (Boom 300 Index,D1)       CSV file created: Boom 300 Index_analytics.csv
2025.01.21 10:50:28.161 External Flow (Boom 300 Index,D1)       Received response: {
2025.01.21 10:50:28.161 External Flow (Boom 300 Index,D1)         "average_price": 865.884525,
2025.01.21 10:50:28.161 External Flow (Boom 300 Index,D1)         "average_volume": 83095.4,
2025.01.21 10:50:28.161 External Flow (Boom 300 Index,D1)         "signal": "SELL",
2025.01.21 10:50:28.161 External Flow (Boom 300 Index,D1)         "signal_explanation": "The last close price (nan) is lower than the average price (865.884525)."
```

The result will also be displayed on the chart, and the process will repeat based on the timeout settings specified in the input parameters.

![Chart Display](https://c.mql5.com/2/112/result.PNG)

Fig 3. Displayed Result

Below is a diagram of a profitable trade I placed based on the generated signal and additional analysis. The trade is displayed on the M1 (1-minute) timeframe for improved clarity.

![Win Trade](https://c.mql5.com/2/112/External_Flow_Win_Trade.png)

Fig 4. Placed Trade

### Conclusion

Having outlined the steps to implement advanced analysis using external libraries such as pandas for this project, I believe we have established a strong foundation for developing more advanced tools for price action and market analytics. I encourage every trader to view this as an overall guide to understanding anticipated market movements. However, for optimal trade execution, please incorporate other strategies you are familiar with. Your feedback and comments are most welcome as we continue working toward creating more professional tools for advanced market analytics.

| Date | Tool Name | Description | Version | Updates | Notes |
| --- | --- | --- | --- | --- | --- |
| 01/10/24 | [Chart Projector](https://www.mql5.com/en/articles/16014) | Script to overlay the previous day's price action with ghost effect. | 1.0 | Initial Release | First tool in Lynnchris Tool Chest |
| 18/11/24 | [Analytical Comment](https://www.mql5.com/en/articles/15927) | It provides previous day's information in a tabular format, as well as anticipates the future direction of the market. | 1.0 | Initial Release | Second tool in the Lynnchris Tool Chest |
| 27/11/24 | [Analytics Master](https://www.mql5.com/en/articles/16434) | Regular Update of market metrics after every two hours | 1.01 | Second Release | Third tool in the Lynnchris Tool Chest |
| 02/12/24 | [Analytics Forecaster](https://www.mql5.com/en/articles/16559) | Regular Update of market metrics after every two hours with telegram integration | 1.1 | Third Edition | Tool number 4 |
| 09/12/24 | [Volatility Navigator](https://www.mql5.com/en/articles/16560) | The EA analyzes market conditions using the Bollinger Bands, RSI and ATR indicators | 1.0 | Initial Release | Tool Number 5 |
| 19/12/24 | [Mean Reversion Signal Reaper](https://www.mql5.com/en/articles/16700) | Analyzes market using mean reversion strategy and provides signal | 1.0 | Initial Release | Tool number 6 |
| 9/01/2025 | [Signal Pulse](https://www.mql5.com/en/articles/16861) | Multiple timeframe analyzer | 1.0 | Initial Release | Tool number 7 |
| 17/01/2025 | Metrics Board | Panel with button for analysis | 1.0 | Initial Release | Tool number 8 |
| 21/01/2025 | External Flow | Analytics through external libraries | 1.0 | Initial Release | Tool number 9 |

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/16967.zip "Download all attachments in the single ZIP archive")

[External\_Flow.mq5](https://www.mql5.com/en/articles/download/16967/external_flow.mq5 "Download External_Flow.mq5")(6.72 KB)

[DeepANALYTICS.py](https://www.mql5.com/en/articles/download/16967/deepanalytics.py "Download DeepANALYTICS.py")(2.48 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Price Action Analysis Toolkit (Part 55): Designing a CPI Mini-Candle Overlay for Intra-bar Pressure](https://www.mql5.com/en/articles/20949)
- [Price Action Analysis Toolkit Development (Part 54): Filtering Trends with EMA and Smoothed Price Action](https://www.mql5.com/en/articles/20851)
- [Price Action Analysis Toolkit Development (Part 53): Pattern Density Heatmap for Support and Resistance Zone Discovery](https://www.mql5.com/en/articles/20390)
- [Price Action Analysis Toolkit Development (Part 52): Master Market Structure with Multi-Timeframe Visual Analysis](https://www.mql5.com/en/articles/20387)
- [Price Action Analysis Toolkit Development (Part 51): Revolutionary Chart Search Technology for Candlestick Pattern Discovery](https://www.mql5.com/en/articles/20313)
- [Price Action Analysis Toolkit Development (Part 50): Developing the RVGI, CCI and SMA Confluence Engine in MQL5](https://www.mql5.com/en/articles/20262)
- [Price Action Analysis Toolkit Development (Part 49): Integrating Trend, Momentum, and Volatility Indicators into One MQL5 System](https://www.mql5.com/en/articles/20168)

**[Go to discussion](https://www.mql5.com/en/forum/480254)**

![Redefining MQL5 and MetaTrader 5 Indicators](https://c.mql5.com/2/113/Redefining_MQL5_and_MetaTrader_5_Indicators___LOGO.png)[Redefining MQL5 and MetaTrader 5 Indicators](https://www.mql5.com/en/articles/16931)

An innovative approach to collecting indicator information in MQL5 enables more flexible and streamlined data analysis by allowing developers to pass custom inputs to indicators for immediate calculations. This approach is particularly useful for algorithmic trading, as it provides enhanced control over the information processed by indicators, moving beyond traditional constraints.

![From Basic to Intermediate: Variables (I)](https://c.mql5.com/2/84/Do_brsico_ao_intermediirio__Variiveis_I.png)[From Basic to Intermediate: Variables (I)](https://www.mql5.com/en/articles/15301)

Many beginning programmers have a hard time understanding why their code doesn't work as they expect. There are many things that make code truly functional. It's not just a bunch of different functions and operations that make the code work. Today I invite you to learn how to properly create real code, rather than copy and paste fragments of it. The materials presented here are for didactic purposes only. Under no circumstances should the application be viewed for any purpose other than to learn and master the concepts presented.

![Build Self Optimizing Expert Advisors in MQL5 (Part 4): Dynamic Position Sizing](https://c.mql5.com/2/113/Build_Self_Optimizing_Expert_Advisors_in_MQL5__4__LOGO.png)[Build Self Optimizing Expert Advisors in MQL5 (Part 4): Dynamic Position Sizing](https://www.mql5.com/en/articles/16925)

Successfully employing algorithmic trading requires continuous, interdisciplinary learning. However, the infinite range of possibilities can consume years of effort without yielding tangible results. To address this, we propose a framework that gradually introduces complexity, allowing traders to refine their strategies iteratively rather than committing indefinite time to uncertain outcomes.

![Monitoring trading with push notifications — example of a MetaTrader 5 service](https://c.mql5.com/2/85/Monitoring_Trade_Using_Push_Notifications___LOGO.png)[Monitoring trading with push notifications — example of a MetaTrader 5 service](https://www.mql5.com/en/articles/15346)

In this article, we will look at creating a service app for sending notifications to a smartphone about trading results. We will learn how to handle lists of Standard Library objects to organize a selection of objects by required properties.

[![](https://www.mql5.com/ff/sh/6xjc81sb5f2g45z9z2/01.png)Follow MQL5.community on social mediaWe publish the best technical materials from experts – free from advertising and irrelevant contentLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=yexgeaiatphxecqagtoxizolvboismyb&s=4e531fd1f983c26570e2dac7588b735354f2f9e0aea561427c030e4a1d2f060b&uid=&ref=https://www.mql5.com/en/articles/16967&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5069674302992746721)

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