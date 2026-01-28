---
title: From Novice to Expert: Animated News Headline Using MQL5 (II)
url: https://www.mql5.com/en/articles/18465
categories: Trading, Integration
relevance_score: 3
scraped_at: 2026-01-23T17:54:59.749772
---

[![](https://www.mql5.com/ff/si/h2ryn394uwcpxwmxc2.jpg)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Feconomic-calendar%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dopen.calendar%26utm_content%3Deconomic.calendar%26utm_campaign%3Den.0009.desktop.default&a=qdeulxgvibgwytgewnvfatbocjnnninc&s=5c0c60f00ff5f5bedb0fdf65d9d79eb820442eb43ffac2b85aa003224f9dba14&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=frtfjprvhwarsfyhqwjiesutbogucgbd&ssn=1769180098418507526&ssn_dr=0&ssn_sr=0&fv_date=1769180098&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F18465&back_ref=https%3A%2F%2Fwww.google.com%2F&title=From%20Novice%20to%20Expert%3A%20Animated%20News%20Headline%20Using%20MQL5%20(II)%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918009835220212&fz_uniq=5068794105280003521&sv=2552)

MetaTrader 5 / Examples


### Contents:

- [Introduction](https://www.mql5.com/en/articles/18465#para1)
- [Overview](https://www.mql5.com/en/articles/18465#para2)
- [Integration of the external news API](https://www.mql5.com/en/articles/18465#para3)
- [Testing](https://www.mql5.com/en/articles/18465#para4)
- [Conclusion](https://www.mql5.com/en/articles/18465#para5)
- [Key Lessons](https://www.mql5.com/en/articles/18465#para6)
- [Attachments](https://www.mql5.com/en/articles/18465#para7)

### Introduction

In this discussion, we build on the foundation laid in the [previous article](https://www.mql5.com/en/articles/18299), where we created a scrolling horizontal bar on the chart to display upcoming news events along with countdown timers indicating how much time remains before each release in order of importance. We concluded that article with a placeholder for live news feed—and now, we’re ready to bring that feature to life.

![News headline EA with with new feed placeholder.](https://c.mql5.com/2/148/terminal64_C0XBnAFKIn.gif)

The first version of the News Headline EA (No API integration, MQL5 Economic Calendar Events only.)

Our focus today is on integrating news headlines from external API sources to solve the challenge of news accessibility within the MetaTrader 5 platform. We will concentrate specifically on headlines—concise news titles that give traders a quick snapshot of what's happening in the financial world. This empowers them to stay informed and decide whether to dive deeper via a web browser or the terminal’s built-in News tab.

Key Takeaways from This Article:

- Explore various of financial market news sources
- Learn how to access external news APIs
- Retrieve and parse news data for usability in MQL5
- Master the [WebRequest()](https://www.mql5.com/en/docs/network/webrequest) function
- Integrate external APIs seamlessly into MQL5-based tools

To support this integration, I went the extra mile to research available news APIs and compiled a list for your convenience. Most of these services offer free access tiers with usage limitations, which are still sufficient for educational and testing purposes. The table below includes five recommended news sources, though there are many more you can explore.

| Provider | Rate Limits | Special Features | Data Scope |
| --- | --- | --- | --- |
| [NewsAPI.org](https://www.mql5.com/go?link=https://newsapi.org/ "https://newsapi.org/") | 500 requests/day | Search filters (keywords, dates, publishers, languages), Boolean operators, multi-language support (14 languages) | Global general news (150,000+ sources across 55 countries) |
| [Marketaux](https://www.mql5.com/go?link=https://www.marketaux.com/ "https://www.marketaux.com/") | 5 API calls/request | Entity extraction (stocks/companies), sentiment scores per entity, AI-powered financial news analysis | Financial news with company-specific metrics |
| [Alpha Vantage](https://www.mql5.com/go?link=https://www.alphavantage.co/ "https://www.alphavantage.co/") | 500 requests/day, 5 requests/minute | 60 plus economic indicators (GDP, inflation), technical/fundamental data, market sentiment analysis | Stocks, forex, crypto, ETFs |
| [Finnhub](https://www.mql5.com/go?link=https://finnhub.io/docs/api/market-news "https://finnhub.io/docs/api/market-news") | 60 calls/minute | Real-time WebSocket streaming, economic calendar, earnings reports, AI sentiment analysis | Market news, economic indicators, corporate filings |
| [EODHD](https://www.mql5.com/go?link=https://eodhd.com/register "https://eodhd.com/register") | 20 calls/day (free plan) | Daily sentiment scores (-1 to 1), 50+ topic tags (earnings, IPOs), full-text articles with symbol filtering | Stocks, ETFs, Forex, Crypto news |

For this project, we’ll be working with [Alpha Vantage](https://www.mql5.com/go?link=https://www.alphavantage.co/ "https://www.alphavantage.co/"), a popular and user-friendly API service known for its comprehensive financial data and straightforward access methods. Its well-documented endpoints and reliable free tier make it an excellent choice for integrating real-time news headlines into our EA.

In the next section, we will outline the roadmap of our discussion, covering key steps such as obtaining API keys, making web requests, parsing JSON responses, and displaying the news data on the chart. After that, we’ll move on to the hands-on implementation to bring this functionality to life.

### Overview

To make this discussion easier to follow, let’s begin with a quick revision of some key terms we’ll be using throughout this project:

- API (Application Programming Interface): A set of rules that allows one software application to interact with another. In our case, it enables MetaTrader 5 to fetch data from external sources like Alpha Vantage.
- API Documentation: The official guide provided by the API provider that explains available endpoints, request formats, parameters, and response structures.
- API Key: A unique identifier provided by the API provider that allows access to their services. It helps manage usage and ensures secure communication.
- JSON (JavaScript Object Notation): It's a lightweight data-interchange format that's easy for humans to read and write, and easy for machines to parse and generate. JSON is commonly used to transmit data between a server and a web or software application, especially in web APIs.
- Parse: The process of analyzing structured data (like JSON) and extracting useful information from it.
- WebRequest: An MQL5 function used to send HTTP requests from the terminal to external servers.
- Integration: The process of connecting and combining the API data with our Expert Advisor (EA) to work seamlessly on the chart.

With our foundational code—News Headline EA—already in place, the first step is to obtain an API key from our chosen provider, Alpha Vantage. To get your API key, visit the Alpha Vantage website and sign up for a free account. You’ll be asked to provide basic information such as your name, email address, and intended usage. After submitting the form, your API key will be displayed immediately.

The next step involves studying the API documentation provided by Alpha Vantage. This will help us understand the request format, available news endpoints, and the structure of the returned JSON data. Once we’re familiar with this, we can design how the data will be parsed and integrated into our EA.

**JSON code structure from Alpha Vantage**

I’ve already obtained my API key for this project, and Vantage offers a wide range of features. However, for now, we’ll focus specifically on financial news.

To view the JSON response from the API, you can use [this link](https://www.mql5.com/go?link=https://www.alphavantage.co/query?function=NEWS_SENTIMENT%26tickers=AAPL%26apikey=demo "https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers=AAPL&apikey=demo").

Before implementing the logic for parsing news headlines from the Alpha Vantage API, we can recognize that each news item in the JSON response contains multiple fields, such as "title", "summary", "url", and "time\_published". However, in the early stages of this EA's design, we are prioritizing simplicity and visual clarity by focusing only on extracting the "title" field. Titles are succinct, high-level descriptors that offer a snapshot of the news content, which aligns well with the EA's goal of providing a fast, low-clutter scrolling ticker. By focusing on the "title" alone, we reduce the overhead of parsing larger blocks of text and maintain a clean, compact visual on the chart, helping traders stay informed without distraction. Below, I’ve shared a JSON code snippet to help you understand the structure of the data returned.

```
 {
    "items": "50",
    "sentiment_score_definition": "x <= -0.35: Bearish; -0.35 < x <= -0.15: Somewhat-Bearish; -0.15 < x < 0.15: Neutral; 0.15 <= x < 0.35:
	Somewhat_Bullish; x >= 0.35: Bullish",
    "relevance_score_definition": "0 < x <= 1, with a higher score indicating higher relevance.",
    "feed": [\
        {\
            "title": "How To Trade Tesla Today Using Technical Analysis - Tesla  ( NASDAQ:TSLA ) ",\
            "url": "https://www.benzinga.com/markets/equities/25/06/45857951/how-to-trade-tesla-today-using-technical-analysis",\
            "time_published": "20250610T121929",\
            "authors": [\
                "RIPS"\
            ],\
            "summary": "Good Morning Traders! In today's Market Clubhouse Morning Memo, we will discuss SPY, QQQ, AAPL, MSFT, NVDA, GOOGL, META, and TSLA.\
	Our proprietary formula, exclusive to Market Clubhouse, dictates these price levels. This dynamic equation takes into account price, volume, and options flow.",\
            "banner_image": "https://cdnwp-s3.benzinga.com/wp-content/uploads/2024/10/22181427/Paolo-and-Gali.jpg?optimize=medium&dpr=2&auto=webp&width=230",\
            "source": "Benzinga",\
            "category_within_source": "Trading",\
            "source_domain": "www.benzinga.com",\
            "topics": [\
                {\
                    "topic": "Technology",\
                    "relevance_score": "0.5"\
                },\
                {\
                    "topic": "Financial Markets",\
                    "relevance_score": "0.5855"\
                },\
                {\
                    "topic": "Manufacturing",\
                    "relevance_score": "0.5"\
                }\
            ],\
            "overall_sentiment_score": 0.216021,\
            "overall_sentiment_label": "Somewhat-Bullish",\
            "ticker_sentiment": [\
                {\
                    "ticker": "MSFT",\
                    "relevance_score": "0.101012",\
                    "ticker_sentiment_score": "0.303818",\
                    "ticker_sentiment_label": "Somewhat-Bullish"\
                },\
                {\
                    "ticker": "GOOG",\
                    "relevance_score": "0.033751",\
                    "ticker_sentiment_score": "0.075535",\
                    "ticker_sentiment_label": "Neutral"\
                },\
                {\
                    "ticker": "NVDA",\
                    "relevance_score": "0.101012",\
                    "ticker_sentiment_score": "0.346995",\
                    "ticker_sentiment_label": "Somewhat-Bullish"\
                },\
                {\
                    "ticker": "AAPL",\
                    "relevance_score": "0.134402",\
                    "ticker_sentiment_score": "0.077776",\
                    "ticker_sentiment_label": "Neutral"\
                },\
                {\
                    "ticker": "TSLA",\
                    "relevance_score": "0.134402",\
                    "ticker_sentiment_score": "0.111086",\
                    "ticker_sentiment_label": "Neutral"\
                }\
            ]\
        },\
```\
\
The planned EA will fetch news headlines once per trading day and scroll them in a continuous ticker bar using the canvas object. The "title" field will be parsed and trimmed from the raw JSON string, bypassing the need for full JSON deserialization, which is inefficient in MQL5. This allows the EA to handle real-time updates lightly and reliably. Since these titles are short and eye-catching by nature, they are ideal for scanning quickly. Eventually, additional fields like "summary" or "url" could be parsed for detailed views or tooltips, but using "title" at this stage provides a fast, low-latency solution that fits directly into the existing visual and performance structure.\
\
In the upcoming sections, we’ll guide you through each of these steps—accessing the API, handling the data, and displaying it on the MetaTrader 5 chart using the News Headline EA.\
\
### Integration of the external news API with News Headline EA\
\
The goal is to dynamically fetch relevant financial headlines and stream them across the chart in a smooth, real-time ticker. This significantly boosts the value of the EA by keeping traders informed of the latest sentiment-moving headlines right on their charts without switching platforms. In this walkthrough, we explain exactly how each part of the implementation works, what it does, and why we structure it this way.\
\
**Step 1:** Adding the API Key Input Field for User Configuration\
\
```\
input string InpAlphaVantageKey = "";  // your Alpha Vantage API key\
```\
\
Here, we define a user-configurable input parameter named InpAlphaVantageKey. This allows users to paste their personal Alpha Vantage API key directly into the EA's settings panel. Without a valid key, no news data can be retrieved. Alpha Vantage issues free API keys with usage limits (typically 500 requests/day), so this field gives users control over authentication without hardcoding credentials. We design the rest of the system to gracefully skip News fetching when this value is empty, ensuring stability even if users forget to enter a key.\
\
**Step 2:** Declaring State Variables for Headline Storage and Control Logic\
\
```\
string   newsHeadlines[];     // holds the extracted headline titles\
int      totalNews = 0;       // keeps track of how many headlines we’ve stored\
datetime lastNewsReload = 0;  // helps ensure we fetch news only once per day\
```\
\
These variables form the core of our news management system.\
\
- _newsHeadlines\[\]_ is a dynamic string array that will hold parsed news titles from the API response. Each string represents one news headline.\
- _totalNews_ is an integer counter that records how many headlines we’ve stored—this is important for rendering the correct number of headlines on the chart.\
- _lastNewsReload_is used to enforce a daily update policy. Alpha Vantage may penalize overuse, so this variable ensures we only fetch headlines once per day unless we explicitly change this logic. It also avoids redundant network requests during the same trading session.\
\
**Step 3:** Fetching News Data from Alpha Vantage with WebRequest\
\
```\
void FetchAlphaVantageNews()\
{\
  if(StringLen(InpAlphaVantageKey) == 0) return;\
```\
\
We begin by verifying whether the API key was entered by the user. If it’s empty, the function exits immediately. This check prevents unnecessary network activity and ensures that the system remains stable and quiet until it’s properly configured. It also avoids returning invalid results or triggering rate limit errors from the API server due to malformed requests.\
\
Here, we construct a normalized datetime value for the start of the current trading day (midnight server time). This is used to compare against lastNewsReload so we know whether we’ve already fetched headlines for today. By using this method instead of raw timestamps, we ensure that news is fetched just once every new day, regardless of how frequently the EA is ticking. This approach keeps our API usage efficient and predictable.\
\
```\
MqlDateTime tm; TimeToStruct(TimeTradeServer(), tm);\
tm.hour = tm.min = tm.sec = 0;\
datetime today = StructToTime(tm);\
\
if(lastNewsReload == today) return;\
lastNewsReload = today;\
```\
\
We dynamically build the API request URL by appending the user’s API key to the base endpoint. This URL is what we’ll pass into WebRequest. The NEWS\_SENTIMENT function is designed to return JSON-formatted metadata and sentiment-rich headlines from multiple high-credibility news sources. By constructing this string at runtime, we make our EA modular and API-key agnostic.\
\
```\
  string url = "https://www.alphavantage.co/query?function=NEWS_SENTIMENT&apikey=" + InpAlphaVantageKey;\
```\
\
This section initializes the parameters required for the WebRequest function:\
\
```\
char post[];\
char response_data[];\
string headers;\
int timeout = 5000;\
ResetLastError();\
```\
\
- post\[\] is kept empty since we’re making a GET request, not sending any body content.\
- response\_data\[\] is where the raw byte response from the server will be saved.\
- headers is left as an empty string because the Alpha Vantage API does not require custom HTTP headers for this endpoint.\
- timeout is set to 5000 milliseconds to ensure we don’t hang for too long on slow connections.\
- ResetLastError() clears out any prior error codes to ensure that any failure we catch next is actually from this request and not residual from something else.\
\
```\
int result = WebRequest("GET", url, headers, timeout, post, response_data, headers);\
if(result != 200)\
{\
  Print("WebRequest failed with result: ", result, ", error: ", GetLastError());\
  return;\
}\
```\
\
This block executes the actual HTTP request. WebRequest returns the HTTP status code (e.g., 200 for success, 403 for forbidden, etc.). If the result is not 200, it means the request failed or was blocked, so we print a diagnostic message with the error code and exit early. This helps us diagnose connectivity or authentication issues without crashing the EA or corrupting data.\
\
Once we confirm the request succeeded, we convert the byte array response\_data\[\] into a human-readable string (resultStr). This string now contains the full JSON response from Alpha Vantage. Before we begin extracting headlines, we clear the newsHeadlines\[\] array to discard any old data and make room for the new day’s headlines.\
\
```\
string resultStr = CharArrayToString(response_data, 0, WHOLE_ARRAY);\
ArrayResize(newsHeadlines, 0);\
```\
\
We set up our parsing logic. The variable _pos_ is our scanning pointer in the JSON string. The array keys\[\] contains two common variations of the key that precedes each headline title in the response JSON. We include both spaced and unspaced versions to be robust against any minor formatting changes in the API response.\
\
```\
  int pos = 0;\
  const string keys[2] = { "\"title\": \"", "\"title\":\"" };\
```\
\
This loop scans through the JSON string to find and extract each news title.\
\
```\
  while(true)\
  {\
    int start = -1;\
    for(int k=0;k<ArraySize(keys);k++)\
    {\
      start = StringFind(resultStr, keys[k], pos);\
      if(start >= 0)\
      {\
        start += StringLen(keys[k]);\
        break;\
      }\
    }\
    if(start < 0) break;\
\
    int end = StringFind(resultStr, "\"", start);\
    if(end < 0) break;\
\
    string title = StringSubstr(resultStr, start, end - start);\
    title = Trim(title);\
```\
\
We:\
\
- Look for the next "title" key,\
- Adjust start to the beginning of the actual text content,\
- Find the end of the quote-enclosed title string,\
- Extract and clean up the headline with Trim() to remove any surrounding whitespace.\
\
This simple pattern-matching approach avoids needing a full JSON parser and is efficient enough for our limited and specific use case.\
\
Each time we find a headline, we dynamically expand the newsHeadlines\[\] array and append the new title. Once all titles have been processed, we update totalNews to reflect how many entries we’ve captured. This gives us a clear count to work with later when drawing the headlines.\
\
```\
    int idx = ArraySize(newsHeadlines);\
    ArrayResize(newsHeadlines, idx + 1);\
    newsHeadlines[idx] = title;\
  }\
\
  totalNews = ArraySize(newsHeadlines);\
```\
\
**Step 4:** Drawing the Scrolling News Ticker on the Chart\
\
Here we construct the visual ticker by concatenating all fetched headlines into a single scrolling string.\
\
- We add a "\|" separator between entries to improve readability.\
- If no headlines are available, we show a placeholder message to keep the ticker active.\
- TextOut() renders this string on the canvas at a specific offNews offset to create a scrolling effect.\
- The line offNews -= InpNewsSpeed; causes the text to shift left over time, creating continuous movement. Once it scrolls out of view, we reset it in the main draw loop.\
\
```\
string ticker = "";\
for(int i=0;i<totalNews;i++)\
{\
  ticker += newsHeadlines[i];\
  if(i < totalNews - 1) ticker += "   |   ";\
}\
if(totalNews == 0) ticker = placeholder;\
\
newsCanvas.TextOut(offNews, yOff, ticker, XRGB(255,255,255), ALIGN_LEFT);\
offNews -= InpNewsSpeed;\
```\
\
**Step 5:** Scheduling the News Fetch with the Timer\
\
We place our FetchAlphaVantageNews() call inside the OnTimer() function to make sure it runs periodically. The internal date logic ensures we only hit the API once per day. This design gives the EA the ability to refresh itself without requiring chart reloads or user action. DrawAll() is also called here to update the canvas on each timer tick, ensuring the ticker scrolls smoothly across the screen.\
\
```\
void OnTimer()\
{\
  //...for the other code\
  FetchAlphaVantageNews();  // only updates once daily\
  //...for the rest of the code\
  DrawAll();  // redraws canvas each timer tick\
}\
```\
\
**Step 6:** Cleaning Up When the EA Is Removed\
\
We include a cleanup routine in the OnDeinit() function to clear the newsHeadlines array. This helps ensure that no memory is retained after the EA is unloaded, which is good practice for preventing resource leaks and ensuring a clean restart next time the EA is loaded.\
\
```\
void OnDeinit(const int reason)\
{\
  //...our preivous code here\
  ArrayResize(newsHeadlines,0);  // clear stored headlines\
}\
```\
\
Now that everything is in place, we have an upgraded version of the News Headline EA that successfully integrates an external API (Alpha Vantage) and leverages the built-in MQL5 Economic Calendar. Together, these features enable real-time news and economic events to be displayed directly on the chart, offering traders seamless and immediate access to critical information without leaving the platform.\
\
At the end of this article, I’ll attach the full source code that includes everything we’ve discussed so far. The next step will involve thoroughly testing these new features to verify that the concept works as expected. Based on our observations, we’ll provide a final evaluation and outline what’s coming next in future updates.\
\
### Testing\
\
In the MetaTrader 5 terminal, ensure that WebRequest access is enabled and the Alpha Vantage server URL is added to the list of allowed URLs. You can access these settings by navigating to Tools > Options > Expert Advisors, or by pressing Ctrl + O on your keyboard. Refer to the image below.\
\
![Allow WebRequest options](https://c.mql5.com/2/149/terminal64_kdzN9iOINg.png)\
\
Allow WebRequest and add the Alpha Vantage link\
\
You can access the compiled Expert Advisor file from the “Expert Advisors” section in the Navigator window (shortcut: Ctrl + N). To load it onto a chart, simply right-click on the EA and select “Attach to Chart,” or drag and drop it directly onto the desired chart. Refer to the image below for a quick visual guide.\
\
![Accessing the compiled file in the terminal](https://c.mql5.com/2/148/terminal64_GkyDmF9a3p.png)\
\
Accessing the News Headline\
\
With the recent integration, our input settings have been enhanced to include new features—most notably, the ability to fetch live news headlines via an external API. To activate the news scrolling feature on the chart, you’ll need to obtain an API key from Alpha Vantage and enter it in the appropriate input field. In the image below, you can see how the input settings interface now looks. If no API key is provided, the EA will display placeholder text in the news section instead of real-time headlines.\
\
![Input Settings (News Headline EA)](https://c.mql5.com/2/148/News_Headline_EA.png)\
\
Input Settings (News Headline EA)\
\
Here is an illustration showing the EA in action after successful deployment. All economic events for June 11 are scrolling smoothly, alongside global market news headlines. For a clearer view and more details, refer to the image below.\
\
![Attaching the News Headline EA to the chart](https://c.mql5.com/2/148/terminal64_avFjGUQaas.gif)\
\
Attaching the News Headline EA to the chart\
\
I also managed to capture a clear presentation of the events displayed in the terminal for June 11. This highlights just how effective the News Headline EA is in pinpointing crucial information directly on the chart—helping traders stay updated without interrupting their chart analysis. Moreover, the EA is designed with efficient pixel space management in mind, allowing the full chart window to remain visible while seamlessly integrating news and events without shrinking or cluttering the view. See the animated illustration below demonstrating how the News Headline EA streamlines access to both economic events and market news headlines.\
\
![The advantage of the News Headline EA in simplifying news accessibility by a single row.](https://c.mql5.com/2/148/ShareX_NdthMw6cMY.gif)\
\
The News Headline EA presents events and news in a clean, non-intrusive manner, ensuring your chart remains clear and fully visible.\
\
With great excitement, we experimented with different scrolling speeds, and the results were truly impressive. It highlighted the potential of creating smooth, resource-efficient visualizations directly on the chart. This experiment also showcased how powerful and user-friendly the MQL5 libraries can be. One standout feature was the use of the CCanvas class, which allowed us to render a semi-transparent background for the news and events lane—keeping the underlying chart grid visible. This reinforces the idea that users can stay informed without sacrificing full chart visibility.\
\
While the current implementation lacks interactivity with the visual elements, it opens the door to exciting future enhancements. Below is an example of our high-speed scrolling setup, where the text blurs slightly due to the rapid movement, but still demonstrates the performance and visual potential.\
\
![High speed scroll setting for calendar events and news](https://c.mql5.com/2/148/terminal64_VzocwpCKmC.gif)\
\
High-speed scroll setting for News and Calendar Events\
\
### Conclusion\
\
I’m genuinely thrilled with the successful implementation of the CCanvas class, which effectively addresses a clear challenge we faced. We’ve managed to develop an alternative method for accessing news and economic events directly on the chart. While MetaTrader 5 does offer access to these updates, their visibility and accessibility while actively trading can be cumbersome. By streamlining the display through scrolling headlines, we've introduced a more user-friendly and visually integrated solution.\
\
Of course, there are some limitations in the current version. The headline news is limited to brief titles, unlike the native terminal version, where you can click and read full articles. However, our solution shines when it comes to economic calendar events, allowing traders to stay updated as they work. For headlines, the benefit lies in being aware of trending topics, which can then be explored further outside the terminal. It’s common to only read headlines that capture interest—so having them scroll across the chart ensures you don't miss out on significant global financial news, all without interrupting your regular trading workflow.\
\
One of the standout advantages is having all this information overlaid directly on the chart in a non-intrusive way, enabling informed trading decisions without sacrificing chart clarity.\
\
Looking ahead, potential improvements could include date labels for each headline, dynamic resizing to adapt to different chart scales, and an alert system that notifies traders before scheduled events. I’m excited for the third phase of this project, where we plan to introduce another information lane tailored to trader needs.\
\
Ultimately, one of the broader goals of this project is to make MQL5 programming more approachable—especially for beginners. To support this, I intend to conclude each development phase with a summary of key programming concepts used, and I aim to continue this practice in future chapters of the series.\
\
### Key Lessons\
\
Below is a summary table of the key programming concepts demonstrated throughout this discussion.\
\
| Concept | Details |\
| --- | --- |\
| Object-Oriented Design | Encapsulation of calendar events into a dedicated class with properties for time, symbol, name, and importance, improving modularity and readability |\
| Dynamic Memory & Arrays | Use of resizable arrays and dynamic allocation to manage varying numbers of event objects in separate collections for high, medium, and low importance |\
| Event-Driven Architecture | Lifecycle functions for initialization, timed updates, and cleanup that drive the EA’s behavior on startup, at each timer tick, and on shutdown |\
| API Integration | Connecting to an external news service via HTTP requests, handling responses and errors, and extracting headline text from the returned data |\
| Graphics Programming | Rendering text and shapes on a transparent overlay using a canvas object to create smooth, scrolling lanes for events and headlines |\
| Time & Date Logic | Converting server time into date boundaries to load only today’s future events and ensure news is fetched once per day |\
| Modular Function Design | Separation of tasks into focused helper functions for positioning, loading data, drawing lanes, and trimming text to keep code organized and maintainable |\
| Resource Cleanup | Properly destroying visual objects and deleting dynamically created event objects to prevent memory leaks when the EA is removed |\
| User Input Handling | Exposing parameters for scroll speeds, display toggles, timer frequency, offset settings, and API key entry to allow user customization |\
| Efficient Redraw Techniques | Minimizing redraw work by erasing and updating only necessary parts of the canvas and adjusting to chart size changes |\
| MQL5 Economic Calendar | Retrieving calendar events, filtering by date, categorizing by importance, and sorting by time to display a prioritized list of upcoming events |\
\
For reference, here’s an image illustrating the meaning of the three colored rectangles used to indicate News priority. Color serves as the primary indicator, though shapes—such as circles—could be used as well.\
\
![](https://c.mql5.com/2/148/PriorityLabels.png)\
\
Event importance labels\
\
### Attachments\
\
Find the EA’s source code attached below—compile it, test it, and then share your feedback and expansion ideas in the comments. Stay tuned for our next article!\
\
| Filename | Version | Description |\
| --- | --- | --- |\
| News Headline EA.mq5 | 1.03 | Expert Advisor that displays economic calendar events and real-time market news headlines directly on the chart using the built-in MQL5 Canvas and Alpha Vantage API |\
\
[Back to contents](https://www.mql5.com/en/articles/18465#para0)\
\
**Attached files** \|\
\
\
[Download ZIP](https://www.mql5.com/en/articles/download/18465.zip "Download all attachments in the single ZIP archive")\
\
[News\_Headline\_EA.mq5](https://www.mql5.com/en/articles/download/18465/news_headline_ea.mq5 "Download News_Headline_EA.mq5")(24.38 KB)\
\
**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.\
\
This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.\
\
#### Other articles by this author\
\
- [From Novice to Expert: Higher Probability Signals](https://www.mql5.com/en/articles/20658)\
- [From Novice to Expert: Navigating Market Irregularities](https://www.mql5.com/en/articles/20645)\
- [From Novice to Expert: Automating Trade Discipline with an MQL5 Risk Enforcement EA](https://www.mql5.com/en/articles/20587)\
- [From Novice to Expert: Trading the RSI with Market Structure Awareness](https://www.mql5.com/en/articles/20554)\
- [From Novice to Expert: Developing a Geographic Market Awareness with MQL5 Visualization](https://www.mql5.com/en/articles/20417)\
- [The MQL5 Standard Library Explorer (Part 5): Multiple Signal Expert](https://www.mql5.com/en/articles/20289)\
- [The MQL5 Standard Library Explorer (Part 4): Custom Signal Library](https://www.mql5.com/en/articles/20266)\
\
**[Go to discussion](https://www.mql5.com/en/forum/489452)**\
\
![Price Action Analysis Toolkit Development (Part 28): Opening Range Breakout Tool](https://c.mql5.com/2/150/18486-price-action-analysis-toolkit-logo.png)[Price Action Analysis Toolkit Development (Part 28): Opening Range Breakout Tool](https://www.mql5.com/en/articles/18486)\
\
At the start of each trading session, the market’s directional bias often becomes clear only after price moves beyond the opening range. In this article, we explore how to build an MQL5 Expert Advisor that automatically detects and analyzes Opening Range Breakouts, providing you with timely, data‑driven signals for confident intraday entries.\
\
![MQL5 Wizard Techniques you should know (Part 71): Using Patterns of MACD and the OBV](https://c.mql5.com/2/150/18462-mql5-wizard-techniques-you-logo__1.png)[MQL5 Wizard Techniques you should know (Part 71): Using Patterns of MACD and the OBV](https://www.mql5.com/en/articles/18462)\
\
The Moving-Average-Convergence-Divergence (MACD) oscillator and the On-Balance-Volume (OBV) oscillator are another pair of indicators that could be used in conjunction within an MQL5 Expert Advisor. This pairing, as is practice in these article series, is complementary with the MACD affirming trends while OBV checks volume. As usual, we use the MQL5 wizard to build and test any potential these two may possess.\
\
![Data Science and ML (Part 44): Forex OHLC Time series Forecasting using Vector Autoregression (VAR)](https://c.mql5.com/2/151/18371-data-science-and-ml-part-44-logo.png)[Data Science and ML (Part 44): Forex OHLC Time series Forecasting using Vector Autoregression (VAR)](https://www.mql5.com/en/articles/18371)\
\
Explore how Vector Autoregression (VAR) models can forecast Forex OHLC (Open, High, Low, and Close) time series data. This article covers VAR implementation, model training, and real-time forecasting in MetaTrader 5, helping traders analyze interdependent currency movements and improve their trading strategies.\
\
![Training a multilayer perceptron using the Levenberg-Marquardt algorithm](https://c.mql5.com/2/100/Training_a_Multilayer_Perceptron_Using_the_Levenberg-Marquardt_Algorithm___LOGO.png)[Training a multilayer perceptron using the Levenberg-Marquardt algorithm](https://www.mql5.com/en/articles/16296)\
\
The article presents an implementation of the Levenberg-Marquardt algorithm for training feedforward neural networks. A comparative analysis of performance with algorithms from the scikit-learn Python library has been conducted. Simpler learning methods, such as gradient descent, gradient descent with momentum, and stochastic gradient descent are preliminarily discussed.\
\
[Best articles and CodeBase updates in MQL5.community channelsFollow us to ensure you never miss out on important updates![](https://www.mql5.com/ff/sh/n9yf51p2srwzfqh5z2/02.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=dgazvhktsxqakdvarucjbvmvzenwlyje&s=98a038fe082e458df8c4a1d8e116e3a6646fd5517f06e48b2356b7ee005817d6&uid=&ref=https://www.mql5.com/en/articles/18465&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5068794105280003521)\
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