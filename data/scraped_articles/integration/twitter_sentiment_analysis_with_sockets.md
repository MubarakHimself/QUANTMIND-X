---
title: Twitter Sentiment Analysis with Sockets
url: https://www.mql5.com/en/articles/15407
categories: Integration
relevance_score: 0
scraped_at: 2026-01-24T14:03:40.502220
---

[Running robots on virtual hosting is easyFollow our step-by-step MetaTrader VPS guide for beginnersRead![](https://www.mql5.com/ff/sh/au4fqg4kms7s9mq1z2/01.png)![](https://www.mql5.com/ff/sh/au4fqg4kms7s9mq1z2/02.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/articles/13586&a=uzpprdshbcrtxvjxpmescehprypbymxc&s=516438f25b531570d9b7d49dcfb29c82fa1021f5ede6571df8026dbfbafcd13f&uid=&ref=https://www.mql5.com/en/articles/15407&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5083309552847296779)

MetaTrader 5 / Integration


### Introduction

This article introduces a sophisticated trading bot that leverages real-time sentiment analysis from social media platforms to inform its trading decisions. By integrating MetaTrader 5 (MT5) with a Python-based sentiment analysis engine, this bot represents a cutting-edge fusion of quantitative finance and natural language processing.

The bot's architecture is built upon a client-server model, utilizing socket communication to bridge the gap between MT5's trading capabilities and Python's data processing prowess. At its core, the system analyzes Twitter sentiment related to specific financial instruments, translating social media buzz into actionable trading signals.

This innovative approach not only demonstrates the potential of interdisciplinary technologies in finance but also highlights the growing importance of alternative data sources in modern trading strategies. As we delve deeper into the bot's functionality and code structure, we'll explore how it processes social media data, manages network communications, and executes trades based on sentiment scores.

The following analysis will provide insights into the bot's components, discussing both its MetaTrader 5 Expert Advisor (EA) written in MQL5 and its Python server counterpart. We'll examine the intricacies of their interaction, the sentiment analysis methodology, and the trading logic implemented. This exploration will offer valuable perspectives for traders, developers, and researchers interested in the intersection of social media analytics and algorithmic trading.

### Break down.

I'll break down the code for both the MetaTrader 5 Expert Advisor (EA) and the Python server, explaining their key components and functionality.

### MetaTrader 5 Expert Advisor (MQL5)

1\. Initialization and Inputs:

The EA begins by defining input parameters for the trading symbol, Twitter API credentials, and trading parameters like stop loss and take profit levels. It also includes necessary libraries for trade execution and position management.

```
//+------------------------------------------------------------------+
//|                               Twitter_Sentiment_with_soquets.mq5 |
//|       Copyright 2024, Javier Santiago Gaston de Iriarte Cabrera. |
//|                      https://www.mql5.com/en/users/jsgaston/news |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, Javier Santiago Gaston de Iriarte Cabrera."
#property link      "https://www.mql5.com/en/users/jsgaston/news"
#property version   "1.001"
#property description "This EA sends data to a Python server and receives sentiment analysis"

input group                "---- Symbol to work with ----"
input string symbol1 = "BTCUSD";                                       // Symbol

input group                "---- Passwords ----"
input string twitter_api_key = "TwitterApiKey";                        // Twitter API key
input string twitter_api_secret = "TwitterApiSecret";                  // Twitter API secret
input string twitter_access_token = "TwitterAccessToken";              // Twitter Access Token
input string twitter_access_token_secret = "TwitterAccessTokenSecret"; // Twitter Access Token Secret
input string twitter_bearer_token = "TwitterBearerToken";              // Twitter Bearer Token
input string client_id = "TwitterClientID";                            // Twitter Client ID
input string client_secret = "TwitterClientSecret";                    // Twitter Client Secret

input group                "---- Stops ----"
input bool   InpUseStops   = false;    // Use stops in trading
input int    InpTakeProfit = 1000;      // TakeProfit level
input int    InpStopLoss   = 500;      // StopLoss level
input double InpLot = 0.1;             // Lot size

#include <Trade\Trade.mqh> // Instatiate Trades Execution Library
#include <Trade\OrderInfo.mqh> // Instatiate Library for Orders Information
#include <Trade\PositionInfo.mqh> // Library for all position features and information

// Create a trade object
CTrade trade;

// Last request time
datetime lastRequestTime = 0;
int requestInterval = 30 * 60; // 30 minutes in seconds
```

2\. OnInit() function:

\- Checks if push notifications are enabled.

\- Initiates communication with the Python server during initialization.

```
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   if(!TerminalInfoInteger(TERMINAL_NOTIFICATIONS_ENABLED))
     {
      Print("Push notifications are not enabled. Please enable them in terminal settings.");
      return INIT_FAILED;
     }

   // Call the function to communicate with the Python server during initialization
   string data = StringFormat("%s,%s,%s,%s,%s,%s,%s,%s", symbol1, twitter_api_key, twitter_api_secret, twitter_access_token, twitter_access_token_secret, twitter_bearer_token, client_id, client_secret);
   string result = CommunicateWithPython(data);
   Print("Result received from Python server during initialization: ", result);

   return(INIT_SUCCEEDED);
  }
```

3\. OnTick() function:

\- Implements a 30-minute interval between requests to the Python server.

\- Formats and sends data to the Python server using the CommunicateWithPython() function.

\- Processes the received sentiment data and executes trades based on the sentiment score.

```
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
   // Check if 30 minutes have passed since the last request
   if(TimeCurrent() - lastRequestTime < requestInterval)
     {
      return; // Exit if the interval has not passed
     }

   // Update the last request time
   lastRequestTime = TimeCurrent();

   // Call the function to communicate with the Python server
   string data = StringFormat("%s,%s,%s,%s,%s,%s,%s,%s", symbol1, twitter_api_key, twitter_api_secret, twitter_access_token, twitter_access_token_secret, twitter_bearer_token, client_id, client_secret);
   string result = CommunicateWithPython(data);

   if(result == "")
     {
      Print("No data received from Python");
      return;
     }

   // Process the sentiment value
   Print("Raw result: ", result);  // Debug line
   string sentiment_values[];
   int num_elements = StringSplit(result, ',', sentiment_values);
   Print("Number of elements: ", num_elements);  // Debug line

   if(num_elements > 0)
     {
      double tweet_sentiment = StringToDouble(sentiment_values[0]);
      Print("Twitter sentiment: ", tweet_sentiment);  // Debug line
      double price = SymbolInfoDouble(symbol1, SYMBOL_BID);
      double take_profit = InpTakeProfit * _Point;
      double stop_loss = InpStopLoss * _Point;

      if(tweet_sentiment > 0)
        {
         // Buy if sentiment is positive
         if(PositionSelect(symbol1))
           {
            Print("Position already open. Skipping buy.");
           }
         else
           {
            if(trade.Buy(InpLot, symbol1, price, price - stop_loss, price + take_profit))
              Print("Buying ", InpLot, " lots of ", symbol1);
            else
              Print("Failed to place buy order. Error: ", GetLastError());
           }
        }
      else if(tweet_sentiment < 0)
        {
         // Sell if sentiment is negative
         if(PositionSelect(symbol1))
           {
            Print("Position already open. Skipping sell.");
           }
         else
           {
            if(trade.Sell(InpLot, symbol1, price, price + stop_loss, price - take_profit))
              Print("Selling ", InpLot, " lots of ", symbol1);
            else
              Print("Failed to place sell order. Error: ", GetLastError());
           }
        }
     }
   else if(StringFind(result, "ERROR,") == 0)
     {
      Print("Error received from Python server: ", result);
     }
   else
     {
      Print("Unexpected response format: ", result);
     }
  }
```

4\. Socket Communication:

\- InitSocket() function creates a socket and connects to the Python server.

\- CommunicateWithPython() function handles sending data to and receiving responses from the Python server.

```
//+------------------------------------------------------------------+
//| Initialize socket                                                |
//+------------------------------------------------------------------+
int InitSocket()
  {
   int socket_handle = SocketCreate();
   if(socket_handle < 0)
     {
      Print("Error creating socket");
      return -1;
     }

   Print("Socket created successfully.");

   // Connect to Python server
   bool isConnected = SocketConnect(socket_handle, "127.0.0.1", 65432, 5000);
   if(!isConnected)
     {
      int error = GetLastError();
      Print("Error connecting to Python server. Error code: ", error);
      SocketClose(socket_handle);
      return -1;
     }

   Print("Connection to Python server established.");
   return socket_handle;
  }

//+------------------------------------------------------------------+
//| Function to send and receive data                                |
//+------------------------------------------------------------------+
string CommunicateWithPython(string data)
  {
   int socket_handle = InitSocket();
   if(socket_handle < 0)
      return "";

   // Ensure data is encoded in UTF-8
   uchar send_buffer[];
   StringToCharArray(data, send_buffer);
   int bytesSent = SocketSend(socket_handle, send_buffer, ArraySize(send_buffer));
   if(bytesSent < 0)
     {
      Print("Error sending data!");
      SocketClose(socket_handle);
      return "";
     }

   Print("Data sent: ", bytesSent);

   uint timeout = 5000; // 5 seconds timeout

   uchar rsp[];
   string result;
   uint timeout_check = GetTickCount() + timeout;
   do
     {
      uint len = SocketIsReadable(socket_handle);
      if(len)
        {
         int rsp_len;
         rsp_len = SocketRead(socket_handle, rsp, len, timeout);
         if(rsp_len > 0)
           {
            result += CharArrayToString(rsp, 0, rsp_len);
           }
        }
     }
   while(GetTickCount() < timeout_check && !IsStopped());
   SocketClose(socket_handle);

   if(result == "")
     {
      Print("No data received from Python");
      return "";
     }

   Print("Data received from Python: ", result);
   return result;
  }

//+------------------------------------------------------------------+
//| Helper function to convert uchar array to string                 |
//+------------------------------------------------------------------+
string CharArrayToString(const uchar &arr[], int start, int length)
  {
   string result;
   char temp[];
   ArrayResize(temp, length);
   ArrayCopy(temp, arr, 0, start, length);
   result = CharArrayToString(temp);
   return result;
  }
//+------------------------------------------------------------------+
```

5\. Trading Logic:

\- If sentiment is positive (> 0), it attempts to open a buy position.

\- If sentiment is negative (< 0), it attempts to open a sell position.

\- Uses the CTrade class for trade execution.

### Python Server:

1\. Server Setup:

\- The start\_server() function initializes a socket server that listens for incoming connections.

```
def start_server():
    """Starts the server that waits for incoming connections."""
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('127.0.0.1', 65432))
    server_socket.listen(1)
    print("Python server started and waiting for connections...")

    while True:
        client_socket, addr = server_socket.accept()
        print(f"Connection from {addr}")

        try:
            data = client_socket.recv(1024)
            data = data.decode('utf-8', errors='ignore')
            print(f"Received data: {data}")

            inputs = data.split(',')
            if len(inputs) != 8:
                raise ValueError("Eight inputs were expected")

            symbol, twitter_api_key, twitter_api_secret, twitter_access_token, twitter_access_token_secret, twitter_bearer_token, client_id, client_secret = inputs

            result = process_data(symbol, twitter_bearer_token)
            result_string = f"{result['tweet_sentiment']}"
            client_socket.sendall(result_string.encode('utf-8'))
            print(f"Response sent to client: {result_string}")
        except Exception as e:
            print(f"Communication error: {e}")
            error_message = f"ERROR,{str(e)}"
            client_socket.sendall(error_message.encode('utf-8'))
        finally:
            client_socket.shutdown(socket.SHUT_RDWR)
            client_socket.close()
            print("Connection closed")
```

2\. Data Processing:

\- When a connection is received, it decodes the data and splits it into separate inputs.

\- Calls the process\_data() function with the symbol and Twitter bearer token.

```
def process_data(symbol, bearer_token):
    """Processes the data obtained from news and tweets."""
    result = { "tweet_sentiment": 0}

    try:
        result["tweet_sentiment"] = analyze_tweets(bearer_token, symbol)
    except Exception as e:
        raise Exception(f"Error processing data: {e}")

    print(f"Data processed. Result: {result}")
    return result
```

3\. Tweet Analysis:

\- The analyze\_tweets() function uses the Twitter API to fetch recent tweets about the given symbol.

\- It uses TextBlob to perform sentiment analysis on each tweet.

\- Calculates the average sentiment score across all retrieved tweets.

```
def analyze_tweets(bearer_token, symbol):
    """Analyzes recent tweets related to the given symbol."""
    try:
        headers = {
            'Authorization': f'Bearer {bearer_token}',
        }
        query = f"{symbol} lang:en -is:retweet"

        # Get the current time and subtract an hour
        end_time = datetime.now(timezone.utc) - timedelta(seconds=10)  # Subtract 10 seconds from the current time
        start_time = end_time - timedelta(hours=4)

        # Convert to RFC 3339 (ISO 8601) format with second precision and 'Z' at the end
        start_time_str = start_time.strftime('%Y-%m-%dT%H:%M:%SZ')
        end_time_str = end_time.strftime('%Y-%m-%dT%H:%M:%SZ')

        search_url = f"https://api.twitter.com/2/tweets/search/recent?query={query}&max_results=100&start_time={start_time_str}&end_time={end_time_str}&sort_order=relevancy"

        print(f"Performing tweet search with query: {query}")
        print(f"Search URL: {search_url}")

        response = requests.get(search_url, headers=headers)
        print(f"Response status code: {response.status_code}")
        print(f"Response text: {response.text}")

        if response.status_code != 200:
            raise Exception(f"Error searching tweets: {response.status_code} - {response.text}")

        tweets = response.json().get('data', [])
        if not tweets:
            print("No tweets found")
            return 0

        sentiments = [TextBlob(tweet['text']).sentiment.polarity for tweet in tweets]
        if not sentiments:
            print("No sentiments found")
            return 0

        average_sentiment = sum(sentiments) / len(sentiments)
        return average_sentiment
    except Exception as e:
        print(f"Error: {e}")
        raise Exception(f"Error analyzing tweets: {e}")
```

4\. Error Handling:

\- Implements try-except blocks to catch and handle potential errors during data processing and API requests.

5\. Response:

\- Sends the calculated sentiment score back to the MT5 client.

Key Points:

1\. Real-time Integration: The system provides near real-time sentiment analysis by fetching and analyzing recent tweets every 30 minutes.

2\. Sentiment-based Trading: The EA uses the sentiment score to make trading decisions, opening buy positions for positive sentiment and sell positions for negative sentiment.

3\. Error Handling: Both the EA and Python server implement error handling to manage potential issues with API requests, data processing, or communication.

4\. Scalability: The socket-based communication allows for potential expansion, such as adding more data sources or more complex analysis in the Python server without significantly altering the MT5 EA.

5\. Security Considerations: The system passes API credentials with each request. In a production environment, this would need to be made more secure.

6\. Limitations: The current implementation only opens new positions and doesn't manage existing ones based on changing sentiment.

This bot demonstrates an interesting approach to integrating external data analysis with MT5 trading. However, it would require thorough testing and possibly refinement of its trading logic before being used in a live trading environment.

### How to Proceed?

1st. Insert the URL (in Tools -> Expert Advisor)

![URL](https://c.mql5.com/2/159/url.png)

2nd. Start Python server

```
Python server started and waiting for connections...
```

3rd. Start connection with EA and wait for receiving data

```
2024.07.24 23:29:45.087 Twitter_Sentiment_with_soquets_v4 (EURUSD,H1)   Socket created successfully.
2024.07.24 23:29:45.090 Twitter_Sentiment_with_soquets_v4 (EURUSD,H1)   Connection to Python server established.
2024.07.24 23:29:45.090 Twitter_Sentiment_with_soquets_v4 (EURUSD,H1)   Data sent: 380
```

```
Data processed. Result: {'tweet_sentiment': 0.20970252525252508}
Response sent to client: 0.20970252525252508
Connection closed
```

```
2024.07.24 23:29:50.082 Twitter_Sentiment_with_soquets_v4 (EURUSD,H1)   Data received from Python: 0.20970252525252508
2024.07.24 23:29:50.082 Twitter_Sentiment_with_soquets_v4 (EURUSD,H1)   Result received from Python server during initialization: 0.20970252525252508
2024.07.24 23:29:50.082 Twitter_Sentiment_with_soquets_v4 (EURUSD,H1)   Socket created successfully.
2024.07.24 23:29:50.084 Twitter_Sentiment_with_soquets_v4 (EURUSD,H1)   Connection to Python server established.
2024.07.24 23:29:50.084 Twitter_Sentiment_with_soquets_v4 (EURUSD,H1)   Data sent: 380
2024.07.24 23:29:55.083 Twitter_Sentiment_with_soquets_v4 (EURUSD,H1)   Data received from Python: 0.20970252525252508
2024.07.24 23:29:55.083 Twitter_Sentiment_with_soquets_v4 (EURUSD,H1)   Raw result: 0.20970252525252508
2024.07.24 23:29:55.083 Twitter_Sentiment_with_soquets_v4 (EURUSD,H1)   Number of elements: 1
2024.07.24 23:29:55.084 Twitter_Sentiment_with_soquets_v4 (EURUSD,H1)   Twitter sentiment: 0.20970252525252508
2024.07.24 23:29:55.201 Twitter_Sentiment_with_soquets_v4 (EURUSD,H1)   Buying 0.1 lots of EURUSD
```

### How to obtain good results?

To improve the strategy and sentiment analysis for better results, consider the following modifications:

1\. Refine Sentiment Analysis:

\- Implement more advanced NLP techniques:

     \\* Use pre-trained language models like BERT or GPT for more nuanced sentiment analysis.

     \\* Incorporate aspect-based sentiment analysis to focus on specific attributes of the asset.

\- Expand data sources:

     \\* Include financial news articles, SEC filings, and other relevant text sources.

     \\* Analyze data from multiple social media platforms, not just Twitter.

\- Implement sentiment trend analysis:

     \\* Track changes in sentiment over time rather than just absolute values.

     \\* Use moving averages of sentiment scores to smooth out short-term fluctuations.

2\. Enhance Trading Logic:

\- Implement sentiment thresholds:

     \\* Define specific sentiment levels for opening long or short positions.

     \\* Use different thresholds for entering and exiting trades.

\- Combine sentiment with technical analysis:

     \\* Use sentiment as a confirmation tool for technical indicators.

     \\* For example, only go long if both sentiment is positive and price is above a moving average.

\- Incorporate volume analysis:

     \\* Consider the volume of social media posts alongside sentiment.

     \\* Higher volume with strong sentiment could indicate a more reliable signal.

3\. Position Sizing and Risk Management:

\- Implement dynamic position sizing:

     \\* Adjust position size based on the strength of the sentiment signal.

     \\* Consider account balance and overall market volatility when sizing positions.

\- Use sentiment for stop-loss and take-profit levels:

     \\* Adjust stop-loss levels based on sentiment volatility.

     \\* Set take-profit targets that account for potential sentiment shifts.

4\. Time Frame Considerations:

\- Analyze sentiment across multiple time frames:

     \\* Short-term sentiment for entry/exit timing.

     \\* Long-term sentiment trends for overall market direction.

\- Implement time-based filters:

     \\* Consider the time of day or week when analyzing sentiment and making trades.

     \\* Some assets might have more reliable sentiment signals during specific market hours.

5\. Asset-Specific Tuning:

\- Customize the strategy for different asset classes:

     \\* Cryptocurrencies might react differently to sentiment than traditional stocks.

     \\* Develop asset-specific sentiment dictionaries or models.

6\. Machine Learning Integration:

\- Develop a machine learning model to predict price movements:

     \\* Use sentiment scores as features alongside traditional market data.

     \\* Implement reinforcement learning for continuous strategy improvement.

7\. Backtesting and Optimization:

\- Conduct extensive backtesting:

     \\* Test the strategy across different market conditions and time periods.

     \\* Use walk-forward optimization to avoid overfitting.

\- Implement parameter optimization:

     \\* Use genetic algorithms or other optimization techniques to fine-tune strategy parameters.

8\. Sentiment Validation:

\- Implement a system to validate the accuracy of sentiment analysis:

     \\* Regularly check a sample of analyzed texts manually.

     \\* Track the correlation between sentiment scores and subsequent price movements.

9\. Adaptive Strategy:

\- Develop a system that can adapt to changing market conditions:

     \\* Adjust sentiment thresholds based on overall market volatility.

     \\* Implement regime detection to switch between different sub-strategies.

10\. Sentiment Decay:

    \- Implement a time decay factor for sentiment:

      \\* Give more weight to recent sentiment data.

      \\* Gradually reduce the impact of older sentiment scores.

11\. Contrarian Approach:

    \- Consider implementing a contrarian strategy in certain conditions:

      \\* Extreme sentiment in one direction might indicate a potential reversal.

Remember, while these modifications can potentially improve the strategy, they also increase complexity. It's crucial to thoroughly test each change and understand its impact on the overall system. Start with simpler modifications and gradually increase complexity as you validate each step. Additionally, always be aware that past performance doesn't guarantee future results, especially in the dynamic world of financial markets.

Also, to get twitter tweets, I have used X for Developers account.

### Conclusion

The trading bot presented in this article represents a significant step forward in the integration of social media sentiment analysis with algorithmic trading. By leveraging the power of MetaTrader 5 and Python, this system demonstrates the potential for real-time, data-driven decision-making in the financial markets.

The bot's innovative approach, combining socket-based communication, sentiment analysis of Twitter data, and automated trading execution, showcases the possibilities that arise from interdisciplinary technological integration. It highlights the growing importance of alternative data sources in modern trading strategies and the potential for natural language processing to provide valuable insights for financial decision-making.

However, as with any trading system, there is room for improvement and refinement. The suggestions provided for enhancing the strategy – from implementing more sophisticated sentiment analysis techniques to incorporating machine learning models – offer a roadmap for future development. These improvements could potentially lead to more robust and reliable trading signals, better risk management, and improved overall performance.

It's crucial to note that while this bot presents an exciting approach to sentiment-based trading, it should be thoroughly tested and validated before being deployed in live trading environments. The complexities of financial markets, the potential for rapid sentiment shifts, and the inherent risks of algorithmic trading all necessitate a cautious and methodical approach to implementation and optimization.

As the fields of natural language processing, machine learning, and algorithmic trading continue to evolve, systems like this bot will likely play an increasingly important role in the financial landscape. They represent not just a new tool for traders, but a new paradigm in how we understand and interact with financial markets.

The journey from this initial implementation to a fully optimized, market-ready system is one of continuous learning, testing, and refinement. It's a journey that reflects the broader evolution of financial technology – always pushing boundaries, always seeking new insights, and always striving to make more informed decisions in an ever-changing market environment.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/15407.zip "Download all attachments in the single ZIP archive")

[Server\_english.py](https://www.mql5.com/en/articles/download/15407/server_english.py "Download Server_english.py")(3.81 KB)

[Twitter\_Sentiment\_with\_soquets\_v4.mq5](https://www.mql5.com/en/articles/download/15407/twitter_sentiment_with_soquets_v4.mq5 "Download Twitter_Sentiment_with_soquets_v4.mq5")(17.35 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Integrating Discord with MetaTrader 5: Building a Trading Bot with Real-Time Notifications](https://www.mql5.com/en/articles/16682)
- [Trading Insights Through Volume: Trend Confirmation](https://www.mql5.com/en/articles/16573)
- [Trading Insights Through Volume: Moving Beyond OHLC Charts](https://www.mql5.com/en/articles/16445)
- [From Python to MQL5: A Journey into Quantum-Inspired Trading Systems](https://www.mql5.com/en/articles/16300)
- [Example of new Indicator and Conditional LSTM](https://www.mql5.com/en/articles/15956)
- [Scalping Orderflow for MQL5](https://www.mql5.com/en/articles/15895)
- [Using PSAR, Heiken Ashi, and Deep Learning Together for Trading](https://www.mql5.com/en/articles/15868)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/470539)**
(1)


![Stanislav Korotky](https://c.mql5.com/avatar/2010/10/4CA7CFA0-1F0C.jpg)

**[Stanislav Korotky](https://www.mql5.com/en/users/marketeer)**
\|
23 Apr 2025 at 16:35

Incredibly incomplete stuff, looks like a draft.

Not a single mention of imports, not a single word/grounding about used sentiment analisys selection for which numerous libs exist, no results.

![Causal analysis of time series using transfer entropy](https://c.mql5.com/2/86/Causal_analysis_of_time_series_using_transfer_entropy___LOGO.png)[Causal analysis of time series using transfer entropy](https://www.mql5.com/en/articles/15393)

In this article, we discuss how statistical causality can be applied to identify predictive variables. We will explore the link between causality and transfer entropy, as well as present MQL5 code for detecting directional transfers of information between two variables.

![MQL5 Wizard Techniques you should know (Part 29): Continuation on Learning Rates with MLPs](https://c.mql5.com/2/86/MQL5_Wizard_Techniques_you_should_know_Part_29___LOGO.png)[MQL5 Wizard Techniques you should know (Part 29): Continuation on Learning Rates with MLPs](https://www.mql5.com/en/articles/15405)

We wrap up our look at learning rate sensitivity to the performance of Expert Advisors by primarily examining the Adaptive Learning Rates. These learning rates aim to be customized for each parameter in a layer during the training process and so we assess potential benefits vs the expected performance toll.

![Neural Networks Made Easy (Part 81): Context-Guided Motion Analysis (CCMR)](https://c.mql5.com/2/73/Neural_networks_are_easy_Part_81___LOGO.png)[Neural Networks Made Easy (Part 81): Context-Guided Motion Analysis (CCMR)](https://www.mql5.com/en/articles/14505)

In previous works, we always assessed the current state of the environment. At the same time, the dynamics of changes in indicators always remained "behind the scenes". In this article I want to introduce you to an algorithm that allows you to evaluate the direct change in data between 2 successive environmental states.

![Building A Candlestick Trend Constraint Model (Part 7): Refining our model for EA development](https://c.mql5.com/2/86/Building_A_Candlestick_Trend_Constraint_Model_Part_7___LOGO.png)[Building A Candlestick Trend Constraint Model (Part 7): Refining our model for EA development](https://www.mql5.com/en/articles/15154)

In this article, we will delve into the detailed preparation of our indicator for Expert Advisor (EA) development. Our discussion will encompass further refinements to the current version of the indicator to enhance its accuracy and functionality. Additionally, we will introduce new features that mark exit points, addressing a limitation of the previous version, which only identified entry points.

[![](https://www.mql5.com/ff/sh/0wxx5f0vuwq7xh89z2/01.png)VPS for 24/7 tradingContact your broker and find out how to get a free hosting subscriptionLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=nhetzvgituppcfrhndpblbihmzziogdh&s=d00c975c8bda3d8c1b29f042ad33ac81952ccea2f130a8f1ffa9015bab8ade87&uid=&ref=https://www.mql5.com/en/articles/15407&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5083309552847296779)

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