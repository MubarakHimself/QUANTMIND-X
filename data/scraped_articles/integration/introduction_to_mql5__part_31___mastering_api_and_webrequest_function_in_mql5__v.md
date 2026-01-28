---
title: Introduction to MQL5 (Part 31): Mastering API and WebRequest Function in MQL5 (V)
url: https://www.mql5.com/en/articles/20546
categories: Integration, Indicators, Expert Advisors
relevance_score: 6
scraped_at: 2026-01-23T11:54:51.483531
---

[![](https://www.mql5.com/ff/sh/dcfwvnr2j2662m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
Trading chats in MQL5 Channels\\
\\
Dozens of channels with market analytics in different languages.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=fbkqsrihzrcaspjwpzqwvwhuwytvekmw&s=58ba7bd7d20708f42b52a0a9fb72b3cddf13cbc212e4450461952955dfcc433c&uid=&ref=https://www.mql5.com/en/articles/20546&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5062810734800316761)

MetaTrader 5 / Integration


### **Introduction**

Welcome back to Part 31 of the Introduction to MQL5 series! In the previous articles, we covered the basics of the API and WebRequest function in MQL5. I showed you how to send a request to a server, receive a response, and sort the server response to retrieve important information. Specifically, in the [previous article](https://www.mql5.com/en/articles/20425/211686#!tab=article), we retrieved the candle data for the last 5 daily candles of BTCUSDT using the Binance API. We also discussed how to classify related data into separate arrays, such as open, high, low, and close prices. With this organized data, you can build both Expert Advisors and indicators.

In this article, we'll go one step further and work on a more complex project. After sorting the server response, we will extract the crucial candle data from the last 10 thirty-minute candles of BTCUSDT. But we are not just retrieving the data; this article will lay the foundation on how to create an indicator that visualizes the data in candle format. We will first collect the candle information, sort it, and save it into a file before creating a custom indicator that reads this file and uses the saved candle details to display the candles immediately on the chart because indicators cannot use the WebRequest function directly in real time. With this method, we may see the API data inside MetaTrader 5 as a true chart.

### **Request Candle Data from API**

To obtain candle data for our project, we must first send a GET request to the Binance API. We wish to obtain the open, high, low, and closing prices, the same as in the previous article. This time, however, we are concentrating on the previous ten BTCUSDT thirty-minute candles. We will use MQL5's OnTick event handler to improve the efficiency of our software. This enables our code to execute each time a new tick is received. But since it would be superfluous and might go over the API limits, we don't want to submit the API request on every tick. Rather, we will only submit the request once for each new 30-minute candle that forms on MetaTrader 5.

Example:

```
string method = "GET";
string url = "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=30m&limit=10";
string headers = "";
int time_out = 5000;
char   data[];
char   result[];
string result_headers;

datetime last_bar_time = 0;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---

//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---

  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//---
   datetime current_m30_time = iTime(_Symbol,PERIOD_M30,0);

   if(current_m30_time != last_bar_time)
     {

      WebRequest(method, url, headers, time_out, data, result, result_headers);
      string server_result = CharArrayToString(result);

      Print(server_result);

      last_bar_time =  current_m30_time;

     }

  }
```

Output:

![Figure 1. Server Response](https://c.mql5.com/2/185/figure_1.png)

Explanation:

To store our data and server response, we first specify the method, URL, headers, timeout, and variables. We wish to ask the Binance API for information; thus, the method is set to "GET." The endpoint for retrieving candle data for BTCUSDT is specified by the URL; the interval is set to 30 minutes and is restricted to the previous ten candles. Because Binance does not require additional headers for this public request, headers are kept empty.  The request will wait for a response from the server up to the five-second timeout. The data to be transmitted, the server answer, and any headers returned are all stored in variables. A variable is also utilized to monitor the timing of the most recent 30-minute candle.

An initialization function prepares the program when the expert advisor launches. This phase guarantees that the expert begins correctly and is prepared to process incoming data, although in this case, no extra preparation is needed. In a similar vein, a deinitialization function is executed when the expert is eliminated or the chart is closed. Although it is empty here, it is provided so that any future cleanup can be taken care of.

Every time a new price tick appears, the primary logic takes place. First, the EA determines the current 30-minute candle's opening time on the chart. To ascertain whether a new candle has begun, it then compares this time with the previously noted candle time. The EA requests the most recent candle data from the Binance API if it is a new candle. After that, the server response is produced after being transformed into a readable format for verification.

The EA adjusts the recorded candle time to the current one after the data has been retrieved. It guarantees that, instead of being sent on each tick, the request will only be sent once every new 30-minute candle. This method increases the expert advisor's efficiency by enabling it to obtain the most recent candle data only when required. This is ideal for developing an indicator or EA that depends on external API data without making needless repetitive queries.

Analogy:

Imagine the Binance API as a vast collection of books. The most recent candles, each of which is a snapshot of market prices, are described in each book. Our objective is to obtain the precise books we require. In this case, the previous ten thirty-minute candles for BTCUSDT, and arrange the data so we can utilize it at a later time. Our "request" to the library is first prepared. Telling the library exactly the books we want, along with the symbol (BTCUSDT), the interval (30 minutes), and the quantity of volumes (10), is similar to this. In a manner similar to setting a timeout for our request, we also choose how long we are prepared to wait for the library to find the books.

It's like going to the library and getting ready to take notes when the EA begins. Before we begin retrieving books, there is a preparation process to make sure everything is in order. Similarly, even if there isn't anything particular to accomplish this time, we might tidy up our desk or return any resources as we leave the library. Now visualize the market as an ever-changing clock. Every tick is like a librarian giving us a fresh page of knowledge. We wish to avoid always requesting the same books. Rather, we look at the clock to see if a new half-hour segment has begun. If so, we ask the librarian to provide us with the newest titles. After receiving the books from the library, we jot down any pertinent information in a notepad for future use.

Once the data is recorded, we use a sticky note to indicate when we last retrieved books from the shelf. This guarantees that we won't request the same books again until the start of the subsequent thirty-minute segment.

### **Sorting Candle Data for Each Candle**

The next step is to divide the server response into distinct candles for each 30-minute bar. The open, high, low, and close prices of the first candle are all combined. The same process is carried out for the remaining candles to guarantee that each 30-minute bar has its own systematic data collection. Recall that I stated in the last piece that you must first understand the JSON format. We can correctly arrange the candle data now that we know how Binance does it. The candle data is returned by Binance as an array of arrays, such as this:

```
[\
[array 1],\
[array 2],\
[array 3],\
[array 4],\
[array 5],\
[array 6],\
[array 7],\
[array 8],\
[array 9],\
[array 10]\
]
```

A single 30-minute candle with all of its information is represented by each inner array. Sorting this server answer into separate candles is the next stage. All the information for the first candle is grouped together, followed by the second, and so on, until the final candle. The format makes it obvious that the best character to use to divide each candle data set is the square bracket sign that closes the array. We are unable to utilize commas because they are part of the candle data itself, which would cause errors and mix up the numbers.

Therefore, using the closing bracket is the safest course of action. The next step is to treat each candle individually after the server response has been divided into distinct sections for each candle. Both the beginning square bracket and extraneous characters like double quote marks will be eliminated. Each candle is made cleaner and simpler to transform into precise values that we may store inside our arrays by performing this cleanup.

Example:

```
string candle_data[];
string first_bar_data;
string first_bar_data_array[];
string second_bar_data;
string second_bar_data_array[];
string third_bar_data;
string third_bar_data_array[];
string forth_bar_data;
string fourth_bar_data_array[];
string fifth_bar_data;
string fifth_bar_data_array[];
string sixth_bar_data;
string sixth_bar_data_array[];
string senventh_bar_data;
string seventh_bar_data_array[];
string eighth_bar_data;
string eighth_bar_data_array[];
string nineth_bar_data;
string nineth_bar_data_array[];
string tenth_bar_data;
string tenth_bar_data_array[];
```

```
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//---
   datetime current_m30_time = iTime(_Symbol,PERIOD_M30,0);

   if(current_m30_time != last_bar_time)
     {

      WebRequest(method, url, headers, time_out, data, result, result_headers);
      server_result = CharArrayToString(result);

      Print(server_result);

      last_bar_time =  current_m30_time;

      int array_count = StringSplit(server_result,']', candle_data);

      //FIRST CANDLE
      first_bar_data = candle_data[0];
      StringReplace(first_bar_data,"[[","");\
      StringReplace(first_bar_data,"\"","");\
\
      StringSplit(first_bar_data,',',first_bar_data_array);\
\
      //SECOND CANDLE\
      second_bar_data = candle_data[1];\
      StringReplace(second_bar_data,",[","");\
      StringReplace(second_bar_data,"\"","");\
\
      StringSplit(second_bar_data,',',second_bar_data_array);\
\
      //THIRD CANDLE\
      third_bar_data = candle_data[2];\
      StringReplace(third_bar_data,",[","");\
      StringReplace(third_bar_data,"\"","");\
\
      StringSplit(third_bar_data,',',third_bar_data_array);\
\
      //FORTH CANDLE\
      forth_bar_data = candle_data[3];\
      StringReplace(forth_bar_data,",[","");\
      StringReplace(forth_bar_data,"\"","");\
\
      StringSplit(forth_bar_data,',',fourth_bar_data_array);\
\
      //FIFTH CANDLE\
      fifth_bar_data = candle_data[4];\
      StringReplace(fifth_bar_data,",[","");\
      StringReplace(fifth_bar_data,"\"","");\
\
      StringSplit(fifth_bar_data,',',fifth_bar_data_array);\
\
      //SIXTH CANDLE\
      sixth_bar_data = candle_data[5];\
      StringReplace(sixth_bar_data,",[","");\
      StringReplace(sixth_bar_data,"\"","");\
\
      StringSplit(sixth_bar_data,',',sixth_bar_data_array);\
\
      //SEVENTH CANDLE\
      senventh_bar_data = candle_data[6];\
      StringReplace(senventh_bar_data,",[","");\
      StringReplace(senventh_bar_data,"\"","");\
\
      StringSplit(senventh_bar_data,',',seventh_bar_data_array);\
\
      //EIGHTH CANDLE\
      eighth_bar_data = candle_data[7];\
      StringReplace(eighth_bar_data,",[","");\
      StringReplace(eighth_bar_data,"\"","");\
\
      StringSplit(eighth_bar_data,',',eighth_bar_data_array);\
\
      //NINETH CANDLE\
      nineth_bar_data = candle_data[8];\
      StringReplace(nineth_bar_data,",[","");\
      StringReplace(nineth_bar_data,"\"","");\
\
      StringSplit(nineth_bar_data,',',nineth_bar_data_array);\
\
      //TENTH CANDLE\
      tenth_bar_data = candle_data[9];\
      StringReplace(tenth_bar_data,",[","");\
      StringReplace(tenth_bar_data,"\"","");\
\
      StringSplit(tenth_bar_data,',',tenth_bar_data_array);\
\
     }\
  }\
```\
\
Explanation:\
\
We start by simply accepting whatever the server provides, which is frequently a single, massive text file with all the candle data in it. We must first identify the boundaries between each candle to make sense of it. We use a splitting tool to divide that lengthy text block whenever we see a closing square bracket, which acts as a separator. This instantly gives us a tidy list (an array) with all the raw data for a single candle represented by each item. We now concentrate on the first candle in our updated list. There are still some unnecessary characters in the raw data, such as quote marks and opening square brackets from the server's formatting. Thus, we take the data from that initial candle and eliminate all those unnecessary characters. After doing so, we are left with a text that is flawlessly clear and only includes the crucial numbers.\
\
To differentiate the real price points, we must utilize such clear terminology. Once more, the text is divided, but this time, a comma is used, and it is crucial because it distinguishes the candle data. Each item in the resulting short, ready-to-use list is one of those exact values. We repeated the same process for the remaining candles. For each candle, we remove any unwanted quotation marks and brackets before dividing the remaining data by commas. By the time we're finished, we've successfully created ten unique, organized lists. Each one only relates to a single candle and has the neatly arranged values for the candle data. The data is now ready for analysis or chart development because it is clean and well-organized.\
\
Analogy:\
\
Imagine the entire server response as a single, long bookshelf with few spaces between the numerous volumes that are crammed onto it. Each book represents one 30-minute candle. You began by splitting this long bookcase into ten smaller books, which meant that the massive string for candles was split into ten halves. Just like ten books are arranged side by side on a shelf, ten distinct candle pieces are now arranged in an array.\
\
You then selected the first book on the shelf. The plastic and store-bought stickers on the front cover stand in for the additional characters that come with the JSON format, like the leading double brackets and quote marks. You took off the stickers and plastic to make the book clean before reading it. After cleaning the cover, you opened the book and found chapters within. These chapters, which stand for the open, high, low, and close values, are divided by commas. As a result, you divided those chapters into smaller sections and arranged them neatly on a different, smaller shelf that was only used for the first candle.\
\
For the second book, you took it off the shelf, took off the stickers, opened it, divided the chapters, and arranged everything neatly in a different tiny part. The third, fourth, fifth, and tenth books all followed the same process. Each book's excess wrappers were taken off before its contents were divided into manageable chunks. In the end, you get ten extremely tidy sections, each of which belongs to a single candle, rather than one big, confusing bookcase. The values within each candle are easily accessible and segregated throughout each segment. Instead of working with a single, massive text block, this makes it easy to interact with each candle separately.\
\
### **Converting Candle Values into Their Correct Data Types**\
\
By now, every candle detail in our candle arrays has already been divided into distinct text elements. Since we gathered all the data from the API as raw strings, all values are still in text format. Before we can utilize them for calculations or display them inside an indicator, we must convert them to the appropriate data types, as they represent dates and numbers.\
\
Example:\
\
```\
//DATETIME\
long bar1_time_s;\
datetime bar1_time;\
long bar2_time_s;\
datetime bar2_time;\
long bar3_time_s;\
datetime bar3_time;\
long bar4_time_s;\
datetime bar4_time;\
long bar5_time_s;\
datetime bar5_time;\
long bar6_time_s;\
datetime bar6_time;\
long bar7_time_s;\
datetime bar7_time;\
long bar8_time_s;\
datetime bar8_time;\
long bar9_time_s;\
datetime bar9_time;\
long bar10_time_s;\
datetime bar10_time;\
\
//OPEN\
double bar1_open;\
double bar2_open;\
double bar3_open;\
double bar4_open;\
double bar5_open;\
double bar6_open;\
double bar7_open;\
double bar8_open;\
double bar9_open;\
double bar10_open;\
\
//HIGH\
double bar1_high;\
double bar2_high;\
double bar3_high;\
double bar4_high;\
double bar5_high;\
double bar6_high;\
double bar7_high;\
double bar8_high;\
double bar9_high;\
double bar10_high;\
\
//LOW\
double bar1_low;\
double bar2_low;\
double bar3_low;\
double bar4_low;\
double bar5_low;\
double bar6_low;\
double bar7_low;\
double bar8_low;\
double bar9_low;\
double bar10_low;\
\
//CLOSE\
double bar1_close;\
double bar2_close;\
double bar3_close;\
double bar4_close;\
double bar5_close;\
double bar6_close;\
double bar7_close;\
double bar8_close;\
double bar9_close;\
double bar10_close;\
```\
\
```\
//TIME\
bar1_time_s = (long)StringToInteger(first_bar_data_array[0])/1000;\
bar1_time = (datetime)bar1_time_s;\
\
bar2_time_s = (long)StringToInteger(second_bar_data_array[0])/1000;\
bar2_time = (datetime)bar2_time_s;\
\
bar3_time_s = (long)StringToInteger(third_bar_data_array[0])/1000;\
bar3_time = (datetime)bar3_time_s;\
\
bar4_time_s = (long)StringToInteger(fourth_bar_data_array[0])/1000;\
bar4_time = (datetime)bar4_time_s;\
\
bar5_time_s = (long)StringToInteger(fifth_bar_data_array[0])/1000;\
bar5_time = (datetime)bar5_time_s;\
\
bar6_time_s = (long)StringToInteger(sixth_bar_data_array[0])/1000;\
bar6_time = (datetime)bar6_time_s;\
\
bar7_time_s = (long)StringToInteger(seventh_bar_data_array[0])/1000;\
bar7_time = (datetime)bar7_time_s;\
\
bar8_time_s = (long)StringToInteger(eighth_bar_data_array[0])/1000;\
bar8_time = (datetime)bar8_time_s;\
\
bar9_time_s = (long)StringToInteger(nineth_bar_data_array[0])/1000;\
bar9_time = (datetime)bar9_time_s;\
\
bar10_time_s = (long)StringToInteger(tenth_bar_data_array[0])/1000;\
bar10_time = (datetime)bar10_time_s;\
\
//OPEN\
bar1_open = StringToDouble(first_bar_data_array[1]);\
bar2_open = StringToDouble(second_bar_data_array[1]);\
bar3_open = StringToDouble(third_bar_data_array[1]);\
bar4_open = StringToDouble(fourth_bar_data_array[1]);\
bar5_open = StringToDouble(fifth_bar_data_array[1]);\
bar6_open = StringToDouble(sixth_bar_data_array[1]);\
bar7_open = StringToDouble(seventh_bar_data_array[1]);\
bar8_open = StringToDouble(eighth_bar_data_array[1]);\
bar9_open = StringToDouble(nineth_bar_data_array[1]);\
bar10_open = StringToDouble(tenth_bar_data_array[1]);\
\
//HIGH\
bar1_high = StringToDouble(first_bar_data_array[2]);\
bar2_high = StringToDouble(second_bar_data_array[2]);\
bar3_high = StringToDouble(third_bar_data_array[2]);\
bar4_high = StringToDouble(fourth_bar_data_array[2]);\
bar5_high = StringToDouble(fifth_bar_data_array[2]);\
bar6_high = StringToDouble(sixth_bar_data_array[2]);\
bar7_high = StringToDouble(seventh_bar_data_array[2]);\
bar8_high = StringToDouble(eighth_bar_data_array[2]);\
bar9_high = StringToDouble(nineth_bar_data_array[2]);\
bar10_high = StringToDouble(tenth_bar_data_array[2]);\
\
//LOW\
bar1_low = StringToDouble(first_bar_data_array[3]);\
bar2_low = StringToDouble(second_bar_data_array[3]);\
bar3_low = StringToDouble(third_bar_data_array[3]);\
bar4_low = StringToDouble(fourth_bar_data_array[3]);\
bar5_low = StringToDouble(fifth_bar_data_array[3]);\
bar6_low = StringToDouble(sixth_bar_data_array[3]);\
bar7_low = StringToDouble(seventh_bar_data_array[3]);\
bar8_low = StringToDouble(eighth_bar_data_array[3]);\
bar9_low = StringToDouble(nineth_bar_data_array[3]);\
bar10_low = StringToDouble(tenth_bar_data_array[3]);\
\
//CLOSE\
bar1_close = StringToDouble(first_bar_data_array[4]);\
bar2_close = StringToDouble(second_bar_data_array[4]);\
bar3_close = StringToDouble(third_bar_data_array[4]);\
bar4_close = StringToDouble(fourth_bar_data_array[4]);\
bar5_close = StringToDouble(fifth_bar_data_array[4]);\
bar6_close = StringToDouble(sixth_bar_data_array[4]);\
bar7_close = StringToDouble(seventh_bar_data_array[4]);\
bar8_close = StringToDouble(eighth_bar_data_array[4]);\
bar9_close = StringToDouble(nineth_bar_data_array[4]);\
bar10_close = StringToDouble(tenth_bar_data_array[4]);\
```\
\
Explanation:\
\
We generate variables for the candle values for each candle. To create an appropriate date-time value in MQL5, the time values must first be transformed from text numbers that represent milliseconds to long numbers. It is necessary to transform the numerical price values from text to double numbers. For this reason, a number of variables are established beforehand. Later on, the converted value for each of these variables will be obtained. The string value must first be transformed to a long number and then split by 1,000. This is because the time of each candle is received from the API in milliseconds. The outcome is then transformed into a correct date-time value that MQL5 can understand and stored in a long variable. This provides us with the appropriate candle time for the chart. Each of the ten candles goes through the same procedure.\
\
Converting the open price values comes next after the time values are set. Every open price is extracted from its string position and converted into a double number using a conversion function. To ensure that each candle has its real numerical open value, this process is performed for each candle. The high values then undergo the same process. Every high value is transferred from its textual location into the conversion function and then into the relevant double variable. After that, the procedure is repeated for the low values and, at last, for the close values. In each instance, the objective is to ensure that every price is converted from plain text into an appropriate numeric format so that it can be utilized for computations, candle drawing, and price movement comparison.\
\
Analogy:\
\
Imagine each candle as a book with its chapters already sorted and cleansed. The book is written in a language that the library cannot understand, which is the only issue. The book is finished, but unless we translate the words into the language that our system understands, it is useless. We must now proceed with that translation. There are five key pages in every candle book. The candle's opening time appears on the first page, followed by the open price on the second, the high price on the third, the low price on the fourth, and the close on the fifth. However, instead of using correct numerical values, each of these pages is written in language. Imagine that the number is written in words rather than numbers when you open the book. You must rewrite each of these numbers into the appropriate format before your library system can store or calculate with them. The conversion stage is that rewriting procedure.\
\
Every book has time written in milliseconds on the first page, which is similar to printing the date using a calendar system that your library does not understand. Rewriting the date into seconds is the first step; this is similar to translating the foreign calendar into your local calendar. The date is then eventually written in the official date style of your library when you convert the result once more into the datetime format. For every candle book, you repeat this process, entering the appropriate datetime into your library system for books one through ten.\
\
After that, you proceed to the remaining pages. The open, high, low, and close values are stored on each pricing page, but they are all still strings. Imagine them as handwritten pages that the library scanner is unable to read. For the algorithm to completely understand each pricing, you rewrite it using clean numerical digits. The library system can now compare prices, save values, and perform computations without difficulty after conversion. For each page in each candle book, you do this. Each candle book is no longer a handwritten foreign document by the end of this process. Everything has been converted into accurate dates and numbers that your library's indexing system can understand. You can now build an indication, draw candles, or perform any necessary computations using the fully readable books on your candle shelf.\
\
### **Grouping Candle Values into Separate Arrays**\
\
At this point, the values of each candle have been correctly transformed into the appropriate data types, with prices being doubles and timings being datetime. Organizing these values so that all comparable data kinds are grouped together is the next stage. In other words, we put the open prices from each of the ten candles into one array, the close prices into another, and so on for the high, low, and time values. When building indicators or carrying out computations, we can more easily work with the structure that is created by grouping the variables in this manner. We can now work with the complete set of data at once by simply referring to the array of open prices, the array of close prices, or the array of times rather than accessing each candle separately each time.\
\
Additionally, this approach increases efficiency and clarity. For instance, the software can depict each candle's open, high, low, and close by progressively reading the arrays while displaying candles on a chart. Because all values of the same type are already grouped together, it also simplifies computations such as moving averages, candle comparisons, and other technical analysis operations.\
\
Example:\
\
```\
datetime OpenTime[10] = {bar1_time, bar2_time, bar3_time, bar4_time, bar5_time,bar6_time, bar7_time, bar8_time, bar9_time, bar10_time};\
double   OpenPrice[10] = {bar1_open, bar2_open, bar3_open, bar4_open, bar5_open,bar6_open, bar7_open, bar8_open, bar9_open, bar10_open};\
double   ClosePrice[10] = {bar1_close, bar2_close, bar3_close, bar4_close, bar5_close,bar6_close, bar7_close, bar8_close, bar9_close, bar10_close};\
double   LowPrice[10] = {bar1_low, bar2_low, bar3_low, bar4_low, bar5_low,bar6_low, bar7_low, bar8_low, bar9_low, bar10_low};\
double   HighPrice[10] = {bar1_high, bar2_high, bar3_high, bar4_high, bar5_high,bar6_high, bar7_high, bar8_high, bar9_high, bar10_high};\
```\
\
Explanation:\
\
This step groups all the individual candle values into arrays so that all similar data are stored together. We now have a single array for every kind of data, rather than distinct variables for every candle. Working with the candle information later in the program is made much simpler as a result. We maintain the 10 candles' open times in the same order that they were received in the initial array. Every candle time has its own slot. Instead of managing ten different time variables, this configuration makes it easier to refer to any candle time by simply pointing to its index.\
\
The technique remains unchanged for the price values. From candle one to candle ten, the open values are arranged in a single array. The same method is used to store the closes, highs, and lows. The first candle is always referred to by the first index, and the remaining indices follow the same order. This ensures that each pricing array's elements all correspond to the same candle number. We can handle candle data in an organized and reliable manner by using arrays. We can process each candle automatically by iterating through the arrays in loops rather than managing each one individually. Likewise, we prevent confusion and maintain proper alignment because every price value shares the same index for the same candle. When creating indicators, calculating values, or creating charts, this arrangement is crucial.\
\
Analogy:\
\
Imagine a collection of books, where each book stands in for a candle. Every book has all the candle's details, including the time and price range. You have been reviewing the information page by page and book by book up until now. It functions, but as the collection grows, handling each one individually becomes challenging and causes a slowdown. You divide the data by category, rather than having everything in one book. One shelf is constructed for all open hours, another for open prices, and still another for closes, highs, and lows. The candle number on the shelves is represented by each row. Therefore, candle one belongs to row one on each shelf, candle two to row two, and so forth.\
\
This eliminates the need for you to open each book separately to compare or read similar content. Simply visit the open price shelf to view all the available prices. You visit the time shelf if you require all the closure times. This arrangement facilitates the efficient computation, comparison, and display of the data.\
\
### **Storing Candle Information for Visualization**\
\
The next step is to create a file to keep all the related candle data that has been grouped into arrays. The file's function is to save all the candle data in an orderly manner so that the candles can subsequently be shown on a chart. The indicator or program can readily access the data by saving it to a file, eliminating the need to repeatedly request it from the server.\
\
The file is set up as a straightforward table. Time, Open, High, Low, and Close are among the headings on the first row that make it obvious what each column stands for. Each row contains the details for a single candle beneath the headings. All candle times are stored in the first column, followed by open prices in the second, high prices in the third, and so on. Candle one is represented by the first data row, candle two by the next, and each subsequent row successively represents the subsequent candle. The candle information is kept orderly and easily accessible with this table-like arrangement. The indicator can accurately replicate the candles on the chart by reading each column. Future chores like adding new candles or updating the data are also made easier by using this framework.\
\
Example:\
\
```\
string filename;\
int handle;\
```\
\
```\
filename = "BTCUSDTM30_MQL5.csv";\
handle = FileOpen(filename, FILE_WRITE|FILE_CSV|FILE_SHARE_READ|FILE_ANSI, ',');\
\
if(handle != INVALID_HANDLE)\
  {\
// Write a header row\
   FileWrite(handle, "Time", "Open", "High", "Low", "Close");\
\
// Write the 5 days of data row by row\
   for(int i = 0; i < 10; i++)\
     {\
      FileWrite(handle, OpenTime[i], OpenPrice[i], HighPrice[i], LowPrice[i], ClosePrice[i]);\
     }\
\
   FileClose(handle);\
   Print("EA successfully wrote the data to " + filename);\
  }\
else\
  {\
   Print("Error opening file for writing. Error code: ", + GetLastError());\
  }\
```\
\
Explanation:\
\
Selecting a filename for the file is the first step in storing the candle data. A concise filename makes it easier to handle several files by providing a quick overview of the data contained within, such as the candles' symbol and timeline. The file is then attempted to be opened for writing by the program. Certain settings are used to guarantee proper operation. When the file is opened in Comma Separated Values (CSV) format, the data is arranged in a table structure that is simple for both programs and people to understand. Additionally, sharing and encoding parameters are configured to enable safe access and appropriate interpretation by various programs. Next, the program writes a header row after the file is opened. This row gives a clear indication of the kind of data recorded in the file and identifies each column. The header facilitates later reading and maintains the file's organization.\
\
The program fills the file with candle data row by row after the header. With all pertinent values neatly organized in the columns, each row represents a single candle. To capture all ten candles in the correct order, the program iterates through the arrays in the same order as the chart. Then the file is closed once all the candle data has been added to ensure that no resources are retained unnecessarily and that everything is saved. A success message is displayed by the application to show that the file was processed properly.\
\
Compiling and executing the code comes after creating it. The Expert Advisor will immediately start the file creation process after it has been initialized on your chart. All the candle data will be saved in a structured CSV file created by the software. On your computer, the file will be generated within the MQL5 directory's File folder. MQL5 keeps program-generated files in this default directory, making it simple to find them later. When the file arrives, you can view it to confirm that all the candle data has been accurately stored before using it to create your indication or perform other analyses.\
\
Analogy:\
\
Imagine 10 books arranged on a shelf, each containing pages of information on each candle. Depending on their type, the pages have been divided into various piles. Imagine now putting all of these stacks in a new binder so that you can easily access the data whenever you need it. To make it clear what kind of information it contains, such as the symbol and time period of the candles, you first label the binder clearly. After that, you open the binder and get it ready to receive the pages in an orderly fashion. To serve as a guide for anyone looking inside, a title page is placed at the front that explains what each column signifies.\
\
After that, you arrange the pages in the binder one after the other, with each row representing a candle. The first row represents the first candle, the second row represents the second, and so on. This arrangement of the pages ensures that all the candle information is organized and readily available when needed. To make sure all the data is securely stored, you close the binder after sorting everything. This is comparable to the software completing the file so that the information is stored on your computer permanently.\
\
You must compile and execute the procedure in order for your library system to use this binder. This translates to "compiling the EA and initializing it on your chart" in programming terms. The binder, which is the file containing all the candle data, is immediately created on your computer once the EA begins operating. It is located in your MQL5 directory's file folder. The binder can now be opened and used, giving your indicator or program access to the arranged candle data for analysis or visualization.\
\
Path:\
\
```\
C:\Users\Dell\AppData\Roaming\MetaQuotes\Terminal\D0E8209F77C8CF37AD8BF550E51FF075\MQL5\Files\
```\
\
![Figure 2. Candle Data in File](https://c.mql5.com/2/185/figure_2.png)\
\
To avoid giving you too much information at once, we will pause here for now. The next article will take the next step and develop an indicator that visualizes the candles on the chart in the correct candle format using the data from the file. This will enable you to view the data that you have stored and arranged as real candlesticks.\
\
### **Conclusion**\
\
To improve our handling of Binance API data, we retrieved the last 10 30-minute candles in this article and divided the server response into individual candles. We grouped related values into arrays and transformed them into the appropriate data types to make the data easier to handle. To keep the candle data in an orderly table that would be easy to access in the future, we finally prepared a file. This process prepares the data for visualization and provides a structured foundation for building indicators or EAs. In the next article, we will use this file to create an indicator that displays the candles on the chart in proper format, allowing you to analyze market behavior visually and efficiently.\
\
**Attached files** \|\
\
\
[Download ZIP](https://www.mql5.com/en/articles/download/20546.zip "Download all attachments in the single ZIP archive")\
\
[Project\_22\_API\_and\_WebRequest.mq5](https://www.mql5.com/en/articles/download/20546/Project_22_API_and_WebRequest.mq5 "Download Project_22_API_and_WebRequest.mq5")(11 KB)\
\
**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.\
\
This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.\
\
#### Other articles by this author\
\
- [Introduction to MQL5 (Part 36): Mastering API and WebRequest Function in MQL5 (X)](https://www.mql5.com/en/articles/20938)\
- [Introduction to MQL5 (Part 35): Mastering API and WebRequest Function in MQL5 (IX)](https://www.mql5.com/en/articles/20859)\
- [Introduction to MQL5 (Part 34): Mastering API and WebRequest Function in MQL5 (VIII)](https://www.mql5.com/en/articles/20802)\
- [Introduction to MQL5 (Part 33): Mastering API and WebRequest Function in MQL5 (VII)](https://www.mql5.com/en/articles/20700)\
- [Introduction to MQL5 (Part 32): Mastering API and WebRequest Function in MQL5 (VI)](https://www.mql5.com/en/articles/20591)\
- [Introduction to MQL5 (Part 30): Mastering API and WebRequest Function in MQL5 (IV)](https://www.mql5.com/en/articles/20425)\
\
**[Go to discussion](https://www.mql5.com/en/forum/501802)**\
\
![From Novice to Expert: Trading the RSI with Market Structure Awareness](https://c.mql5.com/2/185/20554-from-novice-to-expert-trading-logo__1.png)[From Novice to Expert: Trading the RSI with Market Structure Awareness](https://www.mql5.com/en/articles/20554)\
\
In this article, we will explore practical techniques for trading the Relative Strength Index (RSI) oscillator with market structure. Our focus will be on channel price action patterns, how they are typically traded, and how MQL5 can be leveraged to enhance this process. By the end, you will have a rule-based, automated channel-trading system designed to capture trend continuation opportunities with greater precision and consistency.\
\
![Automating Trading Strategies in MQL5 (Part 46): Liquidity Sweep on Break of Structure (BoS)](https://c.mql5.com/2/185/20569-automating-trading-strategies-logo__1.png)[Automating Trading Strategies in MQL5 (Part 46): Liquidity Sweep on Break of Structure (BoS)](https://www.mql5.com/en/articles/20569)\
\
In this article, we build a Liquidity Sweep on Break of Structure (BoS) system in MQL5 that detects swing highs/lows over a user-defined length, labels them as HH/HL/LH/LL to identify BOS (HH in uptrend or LL in downtrend), and spots liquidity sweeps when price wicks beyond the swing but closes back inside on a bullish/bearish candle.\
\
![The View and Controller components for tables in the MQL5 MVC paradigm: Resizable elements](https://c.mql5.com/2/160/18941-komponenti-view-i-controller-logo__2.png)[The View and Controller components for tables in the MQL5 MVC paradigm: Resizable elements](https://www.mql5.com/en/articles/18941)\
\
In the article, we will add the functionality of resizing controls by dragging edges and corners of the element with the mouse.\
\
![Automated Risk Management for Passing Prop Firm Challenges](https://c.mql5.com/2/185/19655-automated-risk-management-for-logo.png)[Automated Risk Management for Passing Prop Firm Challenges](https://www.mql5.com/en/articles/19655)\
\
This article explains the design of a prop-firm Expert Advisor for GOLD, featuring breakout filters, multi-timeframe analysis, robust risk management, and strict drawdown protection. The EA helps traders pass prop-firm challenges by avoiding rule breaches and stabilizing trade execution under volatile market conditions.\
\
[What's wrong with regular VPS?Here are the 8 most common problems that algorithmic traders may encounterRead![](https://www.mql5.com/ff/sh/hzatb686qjqxwtr4z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/450486&a=drhremihlwuaqyvgpzfddbtmgciejpba&s=c37d25bcceb93ed153b814e6ba4d4839461a9b2d68dd82b95b142be06d310f3f&uid=&ref=https://www.mql5.com/en/articles/20546&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5062810734800316761)\
\
This website uses cookies. Learn more about our [Cookies Policy](https://www.mql5.com/en/about/cookies).\
\
![close](https://c.mql5.com/i/close.png)\
\
![MQL5 - Language of trade strategies built-in the MetaTrader 5 client terminal](https://c.mql5.com/i/registerlandings/logo-2.png)\
\
You are missing trading opportunities:\
\
- Free trading apps\
- Over 8,000 signals for copying\
- Economic news for exploring financial markets\
\
RegistrationLog in\
\
latin characters without spaces\
\
a password will be sent to this email\
\
An error occurred\
\
\
- [Log in With Google](https://www.mql5.com/en/auth_oauth2?provider=Google&amp;return=popup&amp;reg=1)\
\
You agree to [website policy](https://www.mql5.com/en/about/privacy) and [terms of use](https://www.mql5.com/en/about/terms)\
\
If you do not have an account, please [register](https://www.mql5.com/en/auth_register)\
\
Allow the use of cookies to log in to the MQL5.com website.\
\
Please enable the necessary setting in your browser, otherwise you will not be able to log in.\
\
[Forgot your login/password?](https://www.mql5.com/en/auth_forgotten?return=popup)\
\
- [Log in With Google](https://www.mql5.com/en/auth_oauth2?provider=Google&amp;return=popup)