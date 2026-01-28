---
title: Sentiment Analysis and Deep Learning for Trading with EA and Backtesting with Python
url: https://www.mql5.com/en/articles/15225
categories: Trading Systems
relevance_score: 6
scraped_at: 2026-01-23T11:41:32.551472
---

[Running robots on virtual hosting is easyFollow our step-by-step MetaTrader VPS guide for beginnersRead![](https://www.mql5.com/ff/sh/au4fqg4kms7s9mq1z2/01.png)![](https://www.mql5.com/ff/sh/au4fqg4kms7s9mq1z2/02.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/articles/13586&a=uzpprdshbcrtxvjxpmescehprypbymxc&s=516438f25b531570d9b7d49dcfb29c82fa1021f5ede6571df8026dbfbafcd13f&uid=&ref=https://www.mql5.com/en/articles/15225&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5062646825963398714)

MetaTrader 5 / Examples


### Introduction

Integrating deep learning and sentiment analysis into trading strategies in MetaTrader 5 (MQL5) represents a sophisticated advancement in algorithmic trading. Deep learning, a subset of machine learning, involves neural networks with multiple layers that can learn and make predictions from vast and complex datasets. Sentiment analysis, on the other hand, is a natural language processing (NLP) technique used to determine the sentiment or emotional tone behind a body of text. By leveraging these technologies, traders can enhance their decision-making processes and improve trading outcomes.

For this article, we will integrate Python into MQL5 using a DLL shell32.dll, which executes what we need for Windows. By installing Python and running it through shell32.dll, we will be able to launch Python scripts from the MQL5 Expert Advisor (EA). There are two Python scripts: one to run the trained ONNX model from TensorFlow, and another script that uses libraries to fetch news from the internet, read the headlines, and quantify media sentiment using AI. This is one possible solution, but there are many ways and different sources to obtain the sentiment of a stock or symbol. Once the model and sentiment are obtained, if both values are in agreement, the order is executed by the EA.

Can we perform a test in Python to understand the results of combining sentiment analysis and deep learning? The answer is yes, and we will proceed to study the code.

### Backtesting Sentiment Analysis with Deep Learning using Python

To perform the backtesting of this strategy, we will use the following libraries. I will use my other article as a starting point. Anyway, here I will also provide the required explanations.

We will use the following libraries:

```
import ccxt
import pandas as pd
import numpy as np
import onnx
import onnxruntime as ort
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
import requests
from datetime import datetime, timedelta
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from newsapi import NewsApiClient
```

First of all, we ensure that nltk is updated.

```
nltk.download('vader_lexicon')
```

nltk (Natural Language Toolkit) is a library used for working with human language data (text). It provides easy-to-use interfaces to over 50 corpora and lexical resources, such as WordNet, along with a suite of text processing libraries for classification, tokenization, stemming, tagging, parsing, and semantic reasoning, as well as wrappers for industrial-strength NLP libraries.

Readers must adapt the python backtesting script to specify where to obtain data, news feed and data for ONNX models.

We will use the following to obtain the sentiment analysis:

```
def get_news_sentiment(symbol, api_key, date):
    try:
        newsapi = NewsApiClient(api_key=api_key)

        # Obtener noticias relacionadas con el símbolo para la fecha específica
        end_date = date + timedelta(days=1)
        articles = newsapi.get_everything(q=symbol,
                                          from_param=date.strftime('%Y-%m-%d'),
                                          to=end_date.strftime('%Y-%m-%d'),
                                          language='en',
                                          sort_by='relevancy',
                                          page_size=10)

        sia = SentimentIntensityAnalyzer()

        sentiments = []
        for article in articles['articles']:
            text = article.get('title', '')
            if article.get('description'):
                text += ' ' + article['description']

            if text:
                sentiment = sia.polarity_scores(text)
                sentiments.append(sentiment['compound'])

        avg_sentiment = np.mean(sentiments) if sentiments else 0
        return avg_sentiment
    except Exception as e:
        print(f"Error al obtener el sentimiento para {symbol} en la fecha {date}: {e}")
        return 0
```

For the backtest, we will use news-api as a feed, because their free API lets us get 1 month look-back of news. If you need more, you can buy a subscription.

The rest of the code will be to obtain the predictions from the ONNX model to predict next close prices. We will just compare the sentiment with the deep learning predictions, and if both conclude with same results, an order will be created. It looks like this:

```
investment_df = comparison_df.copy()
investment_df['price_direction'] = np.where(investment_df['prediction'].shift(-1) > investment_df['prediction'], 1, -1)
investment_df['sentiment_direction'] = np.where(investment_df['sentiment'] > 0, 1, -1)
investment_df['position'] = np.where(investment_df['price_direction'] == investment_df['sentiment_direction'], investment_df['price_direction'], 0)
investment_df['strategy_returns'] = investment_df['position'] * (investment_df['actual'].shift(-1) - investment_df['actual']) / investment_df['actual']
investment_df['buy_and_hold_returns'] = (investment_df['actual'].shift(-1) - investment_df['actual']) / investment_df['actual']
```

The code first creates a copy of \`comparison\_df\` and names it \`investment\_df\`. Then it adds a new column called \`price\_direction\` which takes the value of 1 if the next prediction is higher than the current prediction and -1 otherwise. Next it adds another column called \`sentiment\_direction\` which takes the value of 1 if the sentiment is positive and -1 if it's negative. Then it adds a column named \`position\` which takes the value of \`price\_direction\` if it matches \`sentiment\_direction\` and 0 otherwise. The code then calculates \`strategy\_returns\` by multiplying \`position\` with the relative change in the actual values from one row to the next. Finally it calculates \`buy\_and\_hold\_returns\` as the relative change in the actual values from one row to the next without considering the positions.

Results from this backtest look like this:

```
Datos normalizados guardados en 'binance_data_normalized.csv'
Sentimientos diarios guardados en 'daily_sentiments.csv'
Predicciones y sentimiento guardados en 'predicted_data_with_sentiment.csv'
Mean Absolute Error (MAE): 30.66908467315391
Root Mean Squared Error (RMSE): 36.99641752814565
R-squared (R2): 0.9257591918098058
Mean Absolute Percentage Error (MAPE): 0.00870572230484879
Gráfica guardada como 'ETH_USDT_price_prediction.png'
Gráfica de residuales guardada como 'ETH_USDT_residuals.png'
Correlation between actual and predicted prices: 0.9752007459642241
Gráfica de estrategia de inversión guardada como 'ETH_USDT_investment_strategy.png'
Gráfica de drawdown guardada como 'ETH_USDT_drawdown.png'
Sharpe Ratio: 9.41431958149606
Sortino Ratio: 11800588386323879936.0000
Número de rendimientos totales: 28
Número de rendimientos en exceso: 28
Número de rendimientos negativos: 19
Media de rendimientos en exceso: 0.005037
Desviación estándar de rendimientos negativos: 0.000000
Sortino Ratio: nan
Beta: 0.33875104783408166
Alpha: 0.006981197358213854
Cross-Validation MAE: 1270.7809910146143 ± 527.5746657573876
SMA Mean Absolute Error (MAE): 344.3737716856061
SMA Mean Absolute Error (MAE): 344.3737716856061
SMA Root Mean Squared Error (RMSE): 483.0396130996611
SMA R-squared (R2): 0.5813550203375846
Gráfica de predicción SMA guardada como 'ETH_USDT_sma_price_prediction.png'
Gráfica de precio, predicción y sentimiento guardada como 'ETH_USDT_price_prediction_sentiment.png'
Gráfica de drawdown guardada como 'ETH_USDT_drawdown.png'
Maximum Drawdown: 0.00%
```

As results say, the correlation between the predicted prices and the real prices are very good. R2, that is a metric to measure how good the predictions of the model, also looks good. Sharpe ratio is higher than 5, which is excellent, as well as Sortino. Also, other results are shown in graphs.

The graph that compares the strategy vs hold looks like this:

![Strategy vs hold](https://c.mql5.com/2/82/ethusdt_graph_startegyvshold.png)

Other graphs like price prediction vs actual price

![price prediction vs actual price](https://c.mql5.com/2/82/ethusd_price_prediction_vs_actual_price.png)

and, actual price, price prediction and sentiment

![Price Prediction and Sentiment](https://c.mql5.com/2/82/price_prediction_and_sentiment.png)

The results show that this strategy is very profitable, so we are now using this argument to create an EA.

This EA should have two Python scripts that make the sentiment analysis and the Deep Learning Model, and should be all merged to function in the EA.

### ONNX Model

The code for the data acquisition, training, and ONNX model remains the same as we used in previous articles. Therefore, I will proceed to discuss the Python code for sentiment analysis.

### Sentiment Analysis with Python

We will use the libraries \`requests\` and \`TextBlob\` to fetch forex news and perform sentiment analysis, along with the \`csv\` library for reading and writing data. Additionally, the \`datetime\` and \`time\` libraries will be utilized.

```
import requests
from textblob import TextBlob
import csv
from datetime import datetime
import time
from time import sleep
```

The idea for this script is first to delay for a few seconds upon starting (to ensure that the next part of the script can function properly). The second part of the script will read the API key we want to use. For this case, we will use the Marketaux API, which offers a series of free news and free calls. There are more options such as News API, Alpha Vantage, or Finhub, some of which are paid but provide more news, including historical news, allowing a backtesting of the strategy in MT5. As mentioned earlier, we will use Marketaux for now since it has a free API to obtain daily news. If we want to use other sources, we will need to adapt the code.

Here is a draft of how the script could be structured:

Here's the function to read the api key from the input of the EA:

```
api_file_path = 'C:/Users/jsgas/AppData/Roaming/MetaQuotes/Terminal/24F345EB9F291441AFE537834F9D8A19/MQL5/Files/Files/api.txt'
print(api_file_path)

def read_api_from_file():
    try:
        with open(api_file_path, 'r', encoding='utf-16') as file:
            raw_data = file.read()
            print(f"Raw data from file: {repr(raw_data)}")  # Print raw data
            api = raw_data.strip()  # Lee el contenido y elimina espacios en blanco adicionales
            api = api.replace('\ufeff', '')  # Remove BOM character if present
            print(f"API after stripping whitespace: {api}")
            time.sleep(5)
            return api
    except FileNotFoundError:
        print(f"El archivo {api_file_path} no existe.")
        time.sleep(5)
        return None

# Configuración de la API de Marketaux
api=read_api_from_file()
MARKETAUX_API_KEY = api
```

Before reading the news, we need to know what to read, and for that, we will have this Python script read from a text file created by the EA, so that the Python script knows what to read or which symbol to study and obtain news about, and, what api key is input in the EA, what date is today so the model gets done and for the news to arrive for this date.

It must also be capable of writing a txt or csv so it serves as input to the EA, with the results of the Sentiment.

```
def read_symbol_from_file():
    try:
        with open(symbol_file_path, 'r', encoding='utf-16') as file:
            raw_data = file.read()
            print(f"Raw data from file: {repr(raw_data)}")  # Print raw data
            symbol = raw_data.strip()  # Lee el contenido y elimina espacios en blanco adicionales
            symbol = symbol.replace('\ufeff', '')  # Remove BOM character if present
            print(f"Symbol after stripping whitespace: {symbol}")
            return symbol
    except FileNotFoundError:
        print(f"El archivo {symbol_file_path} no existe.")
        return None
```

```
def save_sentiment_to_txt(average_sentiment, file_path='C:/Users/jsgas/AppData/Roaming/MetaQuotes/Terminal/24F345EB9F291441AFE537834F9D8A19/MQL5/Files/Files/'+str(symbol)+'sentiment.txt'):
    with open(file_path, 'w') as f:
        f.write(f"{average_sentiment:.2f}")
```

```
if symbol:
    news, current_rate = get_forex_news(symbol)

    if news:
        print(f"Noticias para {symbol}:")
        for i, (title, description) in enumerate(news, 1):
            print(f"{i}. {title}")
            print(f"   {description[:100]}...")  # Primeros 100 caracteres de la descripción

        print(f"\nTipo de cambio actual: {current_rate if current_rate else 'No disponible'}")

        # Calcular el sentimiento promedio
        sentiment_scores = [TextBlob(title + " " + description).sentiment.polarity for title, description in news]
        average_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
        print(f"Sentimiento promedio: {average_sentiment:.2f}")

        # Guardar resultados en CSV
        #save_to_csv(symbol, current_rate, average_sentiment)

        # Guardar sentimiento promedio en un archivo de texto
        save_sentiment_to_txt(average_sentiment)
        print("Sentimiento promedio guardado en 'sentiment.txt'")
    else:
        print("No se pudieron obtener noticias de Forex.")
else:
    print("No se pudo obtener el símbolo del archivo.")
```

Readers must adapt the whole script depending on what the study, forex, stocks or crypto.

### The Expert Advisor

We must include shell32.dll as here to run the python scripts

```
#include <WinUser32.mqh>

#import "shell32.dll"
int ShellExecuteW(int hwnd, string lpOperation, string lpFile, string lpParameters, string lpDirectory, int nShowCmd);
#import
```

We must add the python scripts to the File folder

```
string script1 = "C:\\Users\\jsgas\\AppData\\Roaming\\MetaQuotes\\Terminal\\24F345EB9F291441AFE537834F9D8A19\\MQL5\\Files\\Files\\dl model for mql5 v6 Final EURUSD_bien.py";
string script2 = "C:\\Users\\jsgas\\AppData\\Roaming\\MetaQuotes\\Terminal\\24F345EB9F291441AFE537834F9D8A19\\MQL5\\Files\\Files\\sentiment analysis marketaux v6 Final EURUSD_bien.py";
```

And all the paths to the inputs and outputs of the python scripts,

```
// Ruta del archivo donde se escribirá el símbolo
string filePathSymbol = "//Files//symbol.txt";
// Ruta del archivo donde se escribirá el timeframe
string filePathTimeframe = "//Files//timeframe.txt";
string filePathTime = "//Files//time.txt";
string filePathApi = "//Files//api.txt";

string fileToSentiment = "//Files//"+Symbol()+"sentiment.txt";

string file_add = "C://Users//jsgas//AppData//Roaming//MetaQuotes//Terminal//24F345EB9F291441AFE537834F9D8A19//MQL5//Files";
string file_str = "//Files//model_";
string file_str_final = ".onnx";
string file_str_nexo = "_";

string file_add2 = "C:\\Users\\jsgas\\AppData\\Roaming\\MetaQuotes\\Terminal\\24F345EB9F291441AFE537834F9D8A19\\MQL5\\Files";
string file_str2 = "\\Files\\model_";
string file_str_final2 = ".onnx";
string file_str_nexo2 = "_";
```

We must input the Marketaux api key

```
input string api_key      = "mWpORHgs3GdjqNZkxZwnXmrFLYmG5jhAbVrF";           // MARKETAUX_API_KEY www.marketaux.com
```

We can obtain that from [here](https://www.mql5.com/go?link=https://www.marketaux.com/pricing "https://www.marketaux.com/pricing"), and it will look as this:

![api key](https://c.mql5.com/2/82/api_key.png)

I don't work for marketaux, so you can use any other news feed, or subscription you want/need.

You will have to setup a Magic Number, so orders don't get mixed up

```
int OnInit()
  {
   ExtTrade.SetExpertMagicNumber(Magic_Number);
```

You can also add it here

```
void OpenBuyOrder(double lotSize, double slippage, double stopLoss, double takeProfit)
  {
// Definir la estructura MqlTradeRequest
   MqlTradeRequest request;
   MqlTradeResult result;

// Inicializar la estructura de la solicitud
   ZeroMemory(request);

// Establecer los parámetros de la orden
   request.action   = TRADE_ACTION_DEAL;
   request.symbol   = _Symbol;
   request.volume   = lotSize;
   request.type     = ORDER_TYPE_BUY;
   request.price    = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   request.deviation= slippage;
   request.sl       = stopLoss;
   request.tp       = takeProfit;
   request.magic    = Magic_Number;
   request.comment  = "Buy Order";

// Enviar la solicitud de comercio
   if(!OrderSend(request,result))
     {
      Print("Error al abrir orden de compra: ", result.retcode);
```

That last snippet of the code is how the order is made, you can also use trade from CTrade for making orders.

This will write a file (to use as input in the .py scritps):

```
void WriteToFile(string filePath, string data)
  {
   Print("Intentando abrir el archivo: ", filePath);
// Abre el archivo en modo de escritura, crea el archivo si no existe
   int fileHandle = FileOpen(filePath, FILE_WRITE | FILE_TXT);
   if(fileHandle != INVALID_HANDLE)
     {
      // Escribe los datos en el archivo
      FileWriteString(fileHandle, data);
      FileClose(fileHandle);  // Cierra el archivo
      Print("Archivo escrito exitosamente: ", filePath);
     }
   else
     {
      Print("Error al abrir el archivo ", filePath, ". Código de error: ", GetLastError());
     }
  }
```

This will write the symbol, timeframe and current date in the file:

```
void WriteSymbolAndTimeframe()
  {
// Obtén el símbolo actual
   currentSymbol = Symbol();
// Obtén el período de tiempo del gráfico actual
   string currentTimeframe = GetTimeframeString(Period());
   currentTime = TimeToString(TimeCurrent(), TIME_DATE);

// Escribe cada dato en su respectivo archivo
   WriteToFile(filePathSymbol, currentSymbol);
   WriteToFile(filePathTimeframe, currentTimeframe);
   WriteToFile(filePathTime, currentTime);
   WriteToFile(filePathApi,api_key);

   Sleep(10000); // Puedes ajustar o eliminar esto según sea necesario
  }
```

The function WriteSymbolAndTimeframe performs the following tasks:

1. First, it retrieves the current trading symbol and stores it in currentSymbol
2. Then, it gets the current chart's timeframe as a string using GetTimeframeString(Period()) and stores it in currentTimeframe
3. It also gets the current time in a specific format using TimeToString(TimeCurrent(), TIME\_DATE) and stores it in currentTime
4. Next, it writes each of these values to their respective files:

   - currentSymbol is written to filePathSymbol
   - currentTimeframe is written to filePathTimeframe
   - currentTime is written to filePathTime
   - api\_key is written to filePathApi

6. Finally, the function pauses for 10 seconds using Sleep(10000) which can be adjusted or removed as needed.

We can launch the scripts with this:

```
void OnTimer()
  {
   datetime currentTime2 = TimeCurrent();

// Verifica si ha pasado el intervalo para el primer script
   if(currentTime2 - lastExecutionTime1 >= interval1)
     {
      // Escribe los datos necesarios antes de ejecutar el script
      WriteSymbolAndTimeframe();

      // Ejecuta el primer script de Python
      int result = ShellExecuteW(0, "open", "cmd.exe", "/c python \"" + script1 + "\"", "", 1);
      if(result > 32)
         Print("Script 1 iniciado exitosamente");
      else
         Print("Error al iniciar Script 1. Código de error: ", result);
      lastExecutionTime1 = currentTime2;
     }
```

The function \`OnTimer\` is executed periodically and performs the following tasks:

1. First, it retrieves the current time and stores it in \`currentTime2\`.
2. It then checks if the time elapsed since the last execution of the first script (\`lastExecutionTime1\`) is greater than or equal to a predefined interval (\`interval1\`).
3. If the condition is met, it writes the necessary data by calling \`WriteSymbolAndTimeframe\`.
4. Next, it executes the first Python script by running a command via \`ShellExecuteW\` which opens \`cmd.exe\` and runs the Python script specified by \`script1\`.
5. If the script execution is successful (indicated by a result greater than 32), it prints a success message; otherwise, it prints an error message with the corresponding error code.
6. Finally, it updates \`lastExecutionTime1\` to the current time (\`currentTime2\`).

We can read the file with this function:

```
string ReadFile(string file_name)
  {
   string result = "";
   int handle = FileOpen(file_name, FILE_READ|FILE_TXT|FILE_ANSI); // Use FILE_ANSI for plain text

   if(handle != INVALID_HANDLE)
     {
      int file_size = FileSize(handle); // Get the size of the file
      result = FileReadString(handle, file_size); // Read the whole file content
      FileClose(handle);
     }
   else
     {
      Print("Error opening file: ", file_name);
     }

   return result;
  }
```

The code defines a function named ReadFile which takes a file name as an argument and returns the file content as a string first it initializes an empty string result then it attempts to open the file with read permissions and in plain text mode using FileOpen if the file handle is valid it gets the file size using FileSize reads the entire file content into result using FileReadString and then closes the file using FileClose if the file handle is invalid it prints an error message with the file name finally it returns the result containing the file content.

By changing this condition, we can add the sentiment as one more:

```
   if(ExtPredictedClass==PRICE_DOWN && Sentiment_number<0)
      signal=ORDER_TYPE_SELL;    // sell condition
   else
     {
      if(ExtPredictedClass==PRICE_UP && Sentiment_number>0)
         signal=ORDER_TYPE_BUY;  // buy condition
      else
         Print("No order possible");
     }
```

The sentiment in this case goes from 10 to -10, being 0 a neutral signal. You can modify as you want this strategy.

The rest of the code is the simple EA used from the article [How to use ONNX models in MQL5](https://www.mql5.com/en/articles/12373) with a few modifications.

This is not a complete finished EA, this is just a simple example of how to use python and mql5 to create a sentiment & deep learning Expert Advisor. As more time you invest in this EA, you will get less errors and problems. This is a cutting edge case study, and backtesting shows promising results. I hope you find this article helpful, and if someone can manage to get a good sample of news or makes it work for some time, please share results. In order to test the strategy, you should use a demo account.

### Conclusion

In conclusion, the integration of deep learning and sentiment analysis into MetaTrader 5 (MQL5) trading strategies exemplifies the advanced capabilities of modern algorithmic trading. By leveraging Python scripts through a DLL shell32.dll interface, we can seamlessly execute complex models and obtain valuable sentiment data, thereby enhancing trading decisions and outcomes. The process outlined includes using Python to fetch and analyze news sentiment, running ONNX models for price predictions, and executing trades when both indicators align.

The backtesting results demonstrate the strategy's potential profitability, as indicated by strong correlation metrics, high R-squared values, and excellent Sharpe and Sortino ratios. These findings suggest that combining sentiment analysis with deep learning can significantly improve the accuracy of trading signals and overall strategy performance.

Moving forward, the development of a fully functional Expert Advisor (EA) involves meticulous integration of various components, including Python scripts for sentiment analysis and ONNX models for price prediction. By continually refining these elements and adapting the strategy to different markets and data sources, traders can build a robust and effective trading tool.

This study serves as a foundation for those interested in exploring the convergence of machine learning, sentiment analysis, and algorithmic trading, offering a pathway to more informed and potentially profitable trading decisions.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/15225.zip "Download all attachments in the single ZIP archive")

[Scripts.zip](https://www.mql5.com/en/articles/download/15225/scripts.zip "Download Scripts.zip")(1640.86 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/469585)**
(14)


![Javier Santiago Gaston De Iriarte Cabrera](https://c.mql5.com/avatar/2024/11/672e255d-38a9.jpg)

**[Javier Santiago Gaston De Iriarte Cabrera](https://www.mql5.com/en/users/jsgaston)**
\|
27 Jul 2024 at 23:55

**WillowTrader [#](https://www.mql5.com/en/forum/469585#comment_54115680):**

hi Javier, exactly. I went on to review and experiment with the code. Did the dl model found in backtesting folder first. in this model we create a neural network to predict price of etherium based on the close history right? -> i adapted it to see if we can get the direction right. that model itself does perform little better than the toss of a coin it seems, but i look forward to adding the sentiment data to it. Or did i misunderstand the purpose of that model?

I am currently handling the issues i get saving it as ONNX model. Helpful I guess, for learning.

Thank you for this. I will share with you once i managed to get an implementation of this going.

I just added sentiment to the trading logic.

![iwetago247](https://c.mql5.com/avatar/avatar_na2.png)

**[iwetago247](https://www.mql5.com/en/users/iwetago247)**
\|
10 Aug 2024 at 08:38

**Javier Santiago Gaston De Iriarte Cabrera [#](https://www.mql5.com/en/forum/469585#comment_54118114):**

I just added sentiment to the trading logic.

Can I please have your model please it can change my life, I don't have the resources and ability you possess,not the money too, but please help me with your model


![Javier Santiago Gaston De Iriarte Cabrera](https://c.mql5.com/avatar/2024/11/672e255d-38a9.jpg)

**[Javier Santiago Gaston De Iriarte Cabrera](https://www.mql5.com/en/users/jsgaston)**
\|
16 Aug 2024 at 04:07

**iwetago247 [#](https://www.mql5.com/en/forum/469585/page2#comment_54259618):**

Can I please have your model please it can change my life, I don't have the resources and ability you possess,not the money too, but please help me with your model

Hi, you should aim to get a cpu or gpu to compute models, you could use the ones from articles (but search for the timeframe validity) (one 1day time frame model for one symbol holds on for 3 -6 months) ... or you could post a freelance to someone make the models for you. (Each symbol must have its own model for the correct time frame).

![Alasdair](https://c.mql5.com/avatar/2024/3/65EF7E60-2070.png)

**[Alasdair](https://www.mql5.com/en/users/alasdairkite)**
\|
17 Aug 2024 at 19:19

Hello,

Are you trading with this model? If so please can you share with me it's performance.

Thank you.

![yehaichang](https://c.mql5.com/avatar/2025/4/67f6d33b-abb3.jpg)

**[yehaichang](https://www.mql5.com/en/users/yehaichang)**
\|
9 Apr 2025 at 20:03

Prepare to buy


![MQL5 Wizard Techniques you should know (Part 26): Moving Averages and the Hurst Exponent](https://c.mql5.com/2/83/MQL5_Wizard_Techniques_you_should_know_Part_26__LOGO2.png)[MQL5 Wizard Techniques you should know (Part 26): Moving Averages and the Hurst Exponent](https://www.mql5.com/en/articles/15222)

The Hurst Exponent is a measure of how much a time series auto-correlates over the long term. It is understood to be capturing the long-term properties of a time series and therefore carries some weight in time series analysis even outside of economic/ financial time series. We however, focus on its potential benefit to traders by examining how this metric could be paired with moving averages to build a potentially robust signal.

![Reimagining Classic Strategies in Python: MA Crossovers](https://c.mql5.com/2/83/Reimagining_Classic_Strategies_in_Python___LOGO.png)[Reimagining Classic Strategies in Python: MA Crossovers](https://www.mql5.com/en/articles/15160)

In this article, we revisit the classic moving average crossover strategy to assess its current effectiveness. Given the amount of time since its inception, we explore the potential enhancements that AI can bring to this traditional trading strategy. By incorporating AI techniques, we aim to leverage advanced predictive capabilities to potentially optimize trade entry and exit points, adapt to varying market conditions, and enhance overall performance compared to conventional approaches.

![Creating an Interactive Graphical User Interface in MQL5 (Part 1): Making the Panel](https://c.mql5.com/2/83/Creation_of_an_Interactive_Graphical_User_Interface_in_MQL5.png)[Creating an Interactive Graphical User Interface in MQL5 (Part 1): Making the Panel](https://www.mql5.com/en/articles/15205)

This article explores the fundamental steps in crafting and implementing a Graphical User Interface (GUI) panel using MetaQuotes Language 5 (MQL5). Custom utility panels enhance user interaction in trading by simplifying common tasks and visualizing essential trading information. By creating custom panels, traders can streamline their workflow and save time during trading operations.

![Neural networks made easy (Part 77): Cross-Covariance Transformer (XCiT)](https://c.mql5.com/2/70/Neural_networks_made_easy_pPart_77c__Cross-Covariance_Transformer_tXCiTl____LOGO.png)[Neural networks made easy (Part 77): Cross-Covariance Transformer (XCiT)](https://www.mql5.com/en/articles/14276)

In our models, we often use various attention algorithms. And, probably, most often we use Transformers. Their main disadvantage is the resource requirement. In this article, we will consider a new algorithm that can help reduce computing costs without losing quality.

[![](https://www.mql5.com/ff/si/s2n3m9ymjh52n07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F523%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dchoose.signals%26utm_content%3Dsubscribe.signal%26utm_campaign%3D0622.MQL5.com.Internal&a=fyznzyduwsltgnhlftytumasbfgbwlqw&s=91bc0eca8f132d3df7d14cdb1baebac753aef179403d60dc83856af55a4d6769&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=ahyocwtpebfllqybcbimfwniofmjiibx&ssn=1769157691457265216&ssn_dr=0&ssn_sr=0&fv_date=1769157691&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F15225&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Sentiment%20Analysis%20and%20Deep%20Learning%20for%20Trading%20with%20EA%20and%20Backtesting%20with%20Python%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176915769130245210&fz_uniq=5062646825963398714&sv=2552)

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