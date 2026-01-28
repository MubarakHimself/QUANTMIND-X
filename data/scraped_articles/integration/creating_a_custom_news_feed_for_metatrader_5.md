---
title: Creating a custom news feed for MetaTrader 5
url: https://www.mql5.com/en/articles/4149
categories: Integration, Expert Advisors
relevance_score: 6
scraped_at: 2026-01-23T17:25:42.429414
---

[![](https://www.mql5.com/ff/si/fx5m8s6u6uxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=ysducdhemkrdsdtzzbfkclolrllnhezk&s=33f180a31db6c3b846d77732b0bc78169421a47b8cf9f076ca717f4e4846d1c7&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=tpmunbisfkfepbcahecfowrvivnbcctu&ssn=1769178340071248436&ssn_dr=0&ssn_sr=0&fv_date=1769178340&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F4149&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Creating%20a%20custom%20news%20feed%20for%20MetaTrader%205%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176917834056614025&fz_uniq=5068215843768170212&sv=2552)

MetaTrader 5 / Examples


### Introduction

MetaTrader 5 has many useful features that a trader would need, regardless of their trading style, including a possible access to a live news feed. It provides traders with invaluable context that may have some effect on the markets. The only limitation is the scope of the news provided. I believe, traders could benefit from having access to a more flexible news feed that allows the ability to not only choose the kind of news but also its source.

![Built-in news feed](https://c.mql5.com/2/30/builtinnewsfeed.png)

Depending on which broker you trade with, the live news feed can either be very useful or even completely useless. You have to rely on how it has been implemented in the terminal by the broker. Some brokers provide access to relevent news feeds from reputable news outlets. Others though simply do not. Thereby presenting an opportunity to explore the possibility of building an application that meets these demands.

### The News API

Since all the news one could possibly need are freely available on the Internet, all we require is a convenient way of gaining direct access to the news that we want. One way to achieve this is by using RSS tools. I wrote an article on how to code an [RSS reader for MetaTrader 5](https://www.mql5.com/en/articles/1589 "Rss reader for MT5"). The biggest drawback of going down this route is having to manually input each URL, which combined with the extra setup needed for enabling web requests in the terminal, can become tedious.

I believe, the best solution is to use a web [API](https://en.wikipedia.org/wiki/Application_programming_interface "API wikipedia page"). After an extensive search, I stumbled upon a free to use API that provides access to multiple news outlets. Simply called the NewsAPI, it is accessed through HTTP Get requests that return json metadata. It allows one to retrieve headlines of news that has been publised by a chosen news outlet. There are a variety of news outlets to choose from and they claim more will be added. Users are also encouraged to suggest new sources. The service seems to only lack language variety and is dominated by mostly United States and Europe-based news outlets. Still, I think, this is acceptable. Another plus is the fact that all this is accessible from a single web address.

### NewsAPI access

To work with the NewsAPI you need to be authorized to access the full service. To do this, visit the [official website](https://www.mql5.com/go?link=https://newsapi.org/ "NewsAPI.org") and click on Get API Key. Register your email address and you will receive an API key that you need to access all the features.

![NewsAPI website](https://c.mql5.com/2/30/napi3.png)

### Using the API

The entire API is accessed by making requests using two main [URLs](https://en.wikipedia.org/wiki/URL "Url wikipedia page"):

1. https://newsapi.org/v1/sources? - a request made as is will return a list of all news sources available on the API.


    This list also includes identifier information for each source that should be specified when requesting the latest news headlines.The sources URL can be qualified by optional parameters that specify what kind of list of sources is returned.

2. https://newsapi.org/v1/articles? - the articles URL returns news headlines and snippets from a specific source. The URL should contain two compulsory parameters. The first is an identifier which uniquely identifies the required source. The second is the API key for authorization.


### | Parameters | Description | Possible parameter values | Sources/Articles | Example | | --- | --- | --- | --- | --- | | category (optional) | category you would like to get sources for | **business, entertainment, gaming, general, music, politics, science-and-nature, sport, technology**. | sources | https://newsapi.org/v1/sources?category=business | | language (optional) | language you would like to news sources to in | **en, de, fr** | sources | https://newsapi.org/v1/sources?language=en | | country (optional) | country the source is based in | **au, de, gb, in, it, us** | sources | https://newsapi.org/v1/sources?country=us | | source (compulsory) | identifier for the news source | pick any one from list returned by a request using the sources URL | articles | https://newsapi.org/v1/articles?source=cnbc&apiKey=API\_KEY | | api key (compulsory) | user authentication token |  | articles | see example above | | sortby | manner in which the news headlines will be sorted, i.e. popularity,  by the order they appear on the website and by chronological order | **top**, **latest**, **popular** | articles | https://newsapi.org/v1/articles?source=cnbc&sortBy=top&apiKey=API\_KEY |

The table above shows the main parameters that can be used along with the two URLs of the API. For a complete list, please refer to the [documentation](https://www.mql5.com/go?link=https://newsapi.org/ "/go?link=https://newsapi.org/") available on the website.

### A script for tesing the API

Now that we have a general idea of how to use the API, it is time to apply it. The script we will create serves the purpose of familiarizing ourselves with the responses given by making API requests. To achieve this, the script will need to be able to make web requests and also save the responses to a text file. Using the script, we should be able to test any API request and observe the response.

Here is the code of the script NewsAPI\_test.

```
//+------------------------------------------------------------------+
//|                                                 NewsAPI_test.mq5 |
//|                                          Copyright 2017, ufranco |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2017, ufranco"
#property link      "https://www.mql5.com"
#property version   "1.00"
#property script_show_inputs

#define BASE_URL "https://newsapi.org/v1/"
#define SRCE "sources?"
#define ATCLE "articles?source="
#define API_KEY "&apiKey=484c84eb9765418fb58ea936908a47ac"
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
enum mode
  {
   sources,
   articles
  };

input string sFilename="sorce.txt";
input mode Mode=sources;
input string parameters="";
int timeout=5000;
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
//---
   TestWebRequest();
  }
//+------------------------------------------------------------------+
//| TestWebRequest                                                   |
//+------------------------------------------------------------------+
void TestWebRequest(void)
  {
   string cookie=NULL,headers;
   char post[],result[];
   int res;
   string _url;
//---
   switch(Mode)
     {
      case sources:
         _url=BASE_URL+SRCE+parameters;
         break;
      case articles:
         _url=BASE_URL+ATCLE+parameters+API_KEY;
         break;
     }
//---
   ResetLastError();
   res=WebRequest("GET",_url,cookie,NULL,timeout,post,0,result,headers);

   if(res==-1)
     {
      Alert("Could not download file");
      return;
     }
   else Print("Download success");

   string pStream=CharArrayToString(result,0,-1,CP_UTF8);

   int hFile=FileOpen(sFilename,FILE_BIN|FILE_WRITE);

   if(hFile==INVALID_HANDLE)
     {
      Print("Invalid file handle");
      return;
     }

   FileWriteString(hFile,pStream);
   FileClose(hFile);

   return;
  }
```

First off there are the define directives, which represent the different components of a URL that constitute an API request. As the first of user inputs we have sFilename, where one would input a file name that will be used to cache the API responses. The Mode parameter is an enumeration which allows switching between the two main URLs of the API.

The user input parameter named parameters is for entry of additional compulsory or optional URL parameters to be included in an API request. Our script only has one function which builds the URL API call depending on the selected parameter settings. If the webrequest function is successful, the metadata will be saved to file.

We can now conduct some testing and study the responses.The initial test will be a run of the script using the default parameters. When we open the file, the API response can be read. It is important to note the structure of the json object, as it will be useful when we need to extract specific infomation from the data.

```
{
   "status":"ok","sources":
   [\
     {"id":"abc-news-au","name":"ABC News (AU)","description":"Australia's most trusted source of local, national and world news. Comprehensive, independent, in-depth analysis, the latest business, sport, weather and more.","url":"http://www.abc.net.au/news","category":"general","language":"en","country":"au","urlsToLogos":{"small":"","medium":"","large":""},"sortBysAvailable":["top"]},\
     {"id":"al-jazeera-english","name":"Al Jazeera English","description":"News, analysis from the Middle East and worldwide, multimedia and interactives, opinions, documentaries, podcasts, long reads and broadcast schedule.","url":"http://www.aljazeera.com","category":"general","language":"en","country":"us","urlsToLogos":{"small":"","medium":"","large":""},"sortBysAvailable":["top","latest"]},\
     {"id":"ars-technica","name":"Ars Technica","description":"The PC enthusiast's resource. Power users and the tools they love, without computing religion.","url":"http://arstechnica.com","category":"technology","language":"en","country":"us","urlsToLogos":{"small":"","medium":"","large":""},"sortBysAvailable":["top","latest"]},\
     {"id":"associated-press","name":"Associated Press","description":"The AP delivers in-depth coverage on the international, politics, lifestyle, business, and entertainment news.","url":"https://apnews.com/","category":"general","language":"en","country":"us","urlsToLogos":{"small":"","medium":"","large":""},"sortBysAvailable":["top"]},\
     {"id":"bbc-news","name":"BBC News","description":"Use BBC News for up-to-the-minute news, breaking news, video, audio and feature stories. BBC News provides trusted World and UK news as well as local and regional perspectives. Also entertainment, business, science, technology and health news.","url":"http://www.bbc.co.uk/news","category":"general","language":"en","country":"gb","urlsToLogos":{"small":"","medium":"","large":""},"sortBysAvailable":["top"]},\
     {"id":"bbc-sport","name":"BBC Sport","description":"The home of BBC Sport online. Includes live sports coverage, breaking news, results, video, audio and analysis on Football, F1, Cricket, Rugby Union, Rugby League, Golf, Tennis and all the main world sports, plus major events such as the Olympic Games.","url":"http://www.bbc.co.uk/sport","category":"sport","language":"en","country":"gb","urlsToLogos":{"small":"","medium":"","large":""},"sortBysAvailable":["top"]},\
     {"id":"bild","name":"Bild","description":"Die Seite 1 für aktuelle Nachrichten und Themen, Bilder und Videos aus den Bereichen News, Wirtschaft, Politik, Show, Sport, und Promis.","url":"http://www.bild.de","category":"general","language":"de","country":"de","urlsToLogos":{"small":"","medium":"","large":""},"sortBysAvailable":["top","latest"]},\
     {"id":"bloomberg","name":"Bloomberg","description":"Bloomberg delivers business and markets news, data, analysis, and video to the world, featuring stories from Businessweek and Bloomberg News.","url":"http://www.bloomberg.com","category":"business","language":"en","country":"us","urlsToLogos":{"small":"","medium":"","large":""},"sortBysAvailable":["top"]},\
     {"id":"breitbart-news","name":"Breitbart News","description":"Syndicated news and opinion website providing continuously updated headlines to top news and analysis sources.","url":"http://www.breitbart.com","category":"politics","language":"en","country":"us","urlsToLogos":{"small":"","medium":"","large":""},"sortBysAvailable":["top","latest"]},\
     {"id":"business-insider","name":"Business Insider","description":"Business Insider is a fast-growing business site with deep financial, media, tech, and other industry verticals. Launched in 2007, the site is now the largest business news site on the web.","url":"http://www.businessinsider.com","category":"business","language":"en","country":"us","urlsToLogos":{"small":"","medium":"","large":""},"sortBysAvailable":["top","latest"]},\
     {"id":"business-insider-uk","name":"Business Insider (UK)","description":"Business Insider is a fast-growing business site with deep financial, media, tech, and other industry verticals. Launched in 2007, the site is now the largest business news site on the web.","url":"http://uk.businessinsider.com","category":"business","language":"en","country":"gb","urlsToLogos":{"small":"","medium":"","large":""},"sortBysAvailable":["top","latest"]},\
     {"id":"buzzfeed","name":"Buzzfeed","description":"BuzzFeed is a cross-platform, global network for news and entertainment that generates seven billion views each month.","url":"https://www.buzzfeed.com","category":"entertainment","language":"en","country":"us","urlsToLogos":{"small":"","medium":"","large":""},"sortBysAvailable":["top","latest"]},\
     {"id":"cnbc","name":"CNBC","description":"Get latest business news on stock markets, financial & earnings on CNBC. View world markets streaming charts & video; check stock tickers and quotes.","url":"http://www.cnbc.com","category":"business","language":"en","country":"us","urlsToLogos":{"small":"","medium":"","large":""},"sortBysAvailable":["top"]},\
     {"id":"cnn","name":"CNN","description":"View the latest news and breaking news today for U.S., world, weather, entertainment, politics and health at CNN","url":"http://us.cnn.com","category":"general","language":"en","country":"us","urlsToLogos":{"small":"","medium":"","large":""},"sortBysAvailable":["top"]},\
     {"id":"daily-mail","name":"Daily Mail","description":"All the latest news, sport, showbiz, science and health stories from around the world from the Daily Mail and Mail on Sunday newspapers.","url":"http://www.dailymail.co.uk/home/index.html","category":"entertainment","language":"en","country":"gb","urlsToLogos":{"small":"","medium":"","large":""},"sortBysAvailable":["top","latest"]},\
     {"id":"der-tagesspiegel","name":"Der Tagesspiegel","description":"Nachrichten, News und neueste Meldungen aus dem Inland und dem Ausland - aktuell präsentiert von tagesspiegel.de.","url":"http://www.tagesspiegel.de","category":"general","language":"de","country":"de","urlsToLogos":{"small":"","medium":"","large":""},"sortBysAvailable":["latest"]},\
     {"id":"die-zeit","name":"Die Zeit","description":"Aktuelle Nachrichten, Kommentare, Analysen und Hintergrundberichte aus Politik, Wirtschaft, Gesellschaft, Wissen, Kultur und Sport lesen Sie auf ZEIT ONLINE.","url":"http://www.zeit.de/index","category":"business","language":"de","country":"de","urlsToLogos":{"small":"","medium":"","large":""},"sortBysAvailable":["latest"]},\
     {"id":"engadget","name":"Engadget","description":"Engadget is a web magazine with obsessive daily coverage of everything new in gadgets and consumer electronics.","url":"https://www.engadget.com","category":"technology","language":"en","country":"us","urlsToLogos":{"small":"","medium":"","large":""},"sortBysAvailable":["top","latest"]},\
     {"id":"entertainment-weekly","name":"Entertainment Weekly","description":"Online version of the print magazine includes entertainment news, interviews, reviews of music, film, TV and books, and a special area for magazine subscribers.","url":"http://www.ew.com","category":"entertainment","language":"en","country":"us","urlsToLogos":{"small":"","medium":"","large":""},"sortBysAvailable":["top"]},\
     {"id":"espn","name":"ESPN","description":"ESPN has up-to-the-minute sports news coverage, scores, highlights and commentary for NFL, MLB, NBA, College Football, NCAA Basketball and more.","url":"http://espn.go.com","category":"sport","language":"en","country":"us","urlsToLogos":{"small":"","medium":"","large":""},"sortBysAvailable":["top"]},\
     {"id":"espn-cric-info","name":"ESPN Cric Info","description":"ESPN Cricinfo provides the most comprehensive cricket coverage available including live ball-by-ball commentary, news, unparalleled statistics, quality editorial comment and analysis.","url":"http://www.espncricinfo.com/","category":"sport","language":"en","country":"us","urlsToLogos":{"small":"","medium":"","large":""},"sortBysAvailable":["top","latest"]},\
     {"id":"financial-times","name":"Financial Times","description":"The latest UK and international business, finance, economic and political news, comment and analysis from the Financial Times on FT.com.","url":"http://www.ft.com/home/uk","category":"business","language":"en","country":"gb","urlsToLogos":{"small":"","medium":"","large":""},"sortBysAvailable":["top","latest"]},\
     {"id":"focus","name":"Focus","description":"Minutenaktuelle Nachrichten und Service-Informationen von Deutschlands modernem Nachrichtenmagazin.","url":"http://www.focus.de","category":"general","language":"de","country":"de","urlsToLogos":{"small":"","medium":"","large":""},"sortBysAvailable":["top"]},\
     {"id":"football-italia","name":"Football Italia","description":"Italian football news, analysis, fixtures and results for the latest from Serie A, Serie B and the Azzurri.","url":"http://www.football-italia.net","category":"sport","language":"en","country":"it","urlsToLogos":{"small":"","medium":"","large":""},"sortBysAvailable":["top","latest"]},\
     {"id":"fortune","name":"Fortune","description":"Fortune 500 Daily and Breaking Business News","url":"http://fortune.com","category":"business","language":"en","country":"us","urlsToLogos":{"small":"","medium":"","large":""},"sortBysAvailable":["top","latest"]},\
     {"id":"four-four-two","name":"FourFourTwo","description":"The latest football news, in-depth features, tactical and statistical analysis from FourFourTwo, the UK&#039;s favourite football monthly.","url":"http://www.fourfourtwo.com/news","category":"sport","language":"en","country":"gb","urlsToLogos":{"small":"","medium":"","large":""},"sortBysAvailable":["top","latest"]},\
     {"id":"fox-sports","name":"Fox Sports","description":"Find live scores, player and team news, videos, rumors, stats, standings, schedules and fantasy games on FOX Sports.","url":"http://www.foxsports.com","category":"sport","language":"en","country":"us","urlsToLogos":{"small":"","medium":"","large":""},"sortBysAvailable":["top","latest"]},\
     {"id":"google-news","name":"Google News","description":"Comprehensive, up-to-date news coverage, aggregated from sources all over the world by Google News.","url":"https://news.google.com","category":"general","language":"en","country":"us","urlsToLogos":{"small":"","medium":"","large":""},"sortBysAvailable":["top"]},\
     {"id":"gruenderszene","name":"Gruenderszene","description":"Online-Magazin für Startups und die digitale Wirtschaft. News und Hintergründe zu Investment, VC und Gründungen.","url":"http://www.gruenderszene.de","category":"technology","language":"de","country":"de","urlsToLogos":{"small":"","medium":"","large":""},"sortBysAvailable":["top","latest"]},\
     {"id":"hacker-news","name":"Hacker News","description":"Hacker News is a social news website focusing on computer science and entrepreneurship. It is run by Paul Graham's investment fund and startup incubator, Y Combinator. In general, content that can be submitted is defined as \"anything that gratifies one's intellectual curiosity\".","url":"https://news.ycombinator.com","category":"technology","language":"en","country":"us","urlsToLogos":{"small":"","medium":"","large":""},"sortBysAvailable":["top","latest"]},\
     {"id":"handelsblatt","name":"Handelsblatt","description":"Auf Handelsblatt lesen sie Nachrichten über Unternehmen, Finanzen, Politik und Technik. Verwalten Sie Ihre Finanzanlagen mit Hilfe unserer Börsenkurse.","url":"http://www.handelsblatt.com","category":"business","language":"de","country":"de","urlsToLogos":{"small":"","medium":"","large":""},"sortBysAvailable":["latest"]},\
     {"id":"ign","name":"IGN","description":"IGN is your site for Xbox One, PS4, PC, Wii-U, Xbox 360, PS3, Wii, 3DS, PS Vita and iPhone games with expert reviews, news, previews, trailers, cheat codes, wiki guides and walkthroughs.","url":"http://www.ign.com","category":"gaming","language":"en","country":"us","urlsToLogos":{"small":"","medium":"","large":""},"sortBysAvailable":["top","latest"]},\
     {"id":"independent","name":"Independent","description":"National morning quality (tabloid) includes free online access to news and supplements. Insight by Robert Fisk and various other columnists.","url":"http://www.independent.co.uk","category":"general","language":"en","country":"gb","urlsToLogos":{"small":"","medium":"","large":""},"sortBysAvailable":["top"]},\
     {"id":"mashable","name":"Mashable","description":"Mashable is a global, multi-platform media and entertainment company.","url":"http://mashable.com","category":"entertainment","language":"en","country":"us","urlsToLogos":{"small":"","medium":"","large":""},"sortBysAvailable":["top","latest"]},\
     {"id":"metro","name":"Metro","description":"News, Sport, Showbiz, Celebrities from Metro - a free British newspaper.","url":"http://metro.co.uk","category":"general","language":"en","country":"gb","urlsToLogos":{"small":"","medium":"","large":""},"sortBysAvailable":["top","latest"]},\
     {"id":"mirror","name":"Mirror","description":"All the latest news, sport and celebrity gossip at Mirror.co.uk. Get all the big headlines, pictures, analysis, opinion and video on the stories that matter to you.","url":"http://www.mirror.co.uk/","category":"general","language":"en","country":"gb","urlsToLogos":{"small":"","medium":"","large":""},"sortBysAvailable":["top","latest"]},\
     {"id":"mtv-news","name":"MTV News","description":"The ultimate news source for music, celebrity, entertainment, movies, and current events on the web. It's pop culture on steroids.","url":"http://www.mtv.com/news","category":"music","language":"en","country":"us","urlsToLogos":{"small":"","medium":"","large":""},"sortBysAvailable":["top","latest"]},\
     {"id":"mtv-news-uk","name":"MTV News (UK)","description":"All the latest celebrity news, gossip, exclusive interviews and pictures from the world of music and entertainment.","url":"http://www.mtv.co.uk/news","category":"music","language":"en","country":"gb","urlsToLogos":{"small":"","medium":"","large":""},"sortBysAvailable":["top"]},\
     {"id":"national-geographic","name":"National Geographic","description":"Reporting our world daily: original nature and science news from National Geographic.","url":"http://news.nationalgeographic.com","category":"science-and-nature","language":"en","country":"us","urlsToLogos":{"small":"","medium":"","large":""},"sortBysAvailable":["top"]},\
     {"id":"new-scientist","name":"New Scientist","description":"Breaking science and technology news from around the world. Exclusive stories and expert analysis on space, technology, health, physics, life and Earth.","url":"https://www.newscientist.com/section/news","category":"science-and-nature","language":"en","country":"us","urlsToLogos":{"small":"","medium":"","large":""},"sortBysAvailable":["top"]},\
     {"id":"newsweek","name":"Newsweek","description":"Newsweek provides in-depth analysis, news and opinion about international issues, technology, business, culture and politics.","url":"http://www.newsweek.com","category":"general","language":"en","country":"us","urlsToLogos":{"small":"","medium":"","large":""},"sortBysAvailable":["top","latest"]},\
     {"id":"new-york-magazine","name":"New York Magazine","description":"NYMAG and New York magazine cover the new, the undiscovered, the next in politics, culture, food, fashion, and behavior nationally, through a New York lens.","url":"http://nymag.com","category":"general","language":"en","country":"us","urlsToLogos":{"small":"","medium":"","large":""},"sortBysAvailable":["top","latest"]},\
     {"id":"nfl-news","name":"NFL News","description":"The official source for NFL news, schedules, stats, scores and more.","url":"http://www.nfl.com/news","category":"sport","language":"en","country":"us","urlsToLogos":{"small":"","medium":"","large":""},"sortBysAvailable":["top","latest"]},\
     {"id":"polygon","name":"Polygon","description":"Polygon is a gaming website in partnership with Vox Media. Our culture focused site covers games, their creators, the fans, trending stories and entertainment news.","url":"http://www.polygon.com","category":"gaming","language":"en","country":"us","urlsToLogos":{"small":"","medium":"","large":""},"sortBysAvailable":["top"]},\
     {"id":"recode","name":"Recode","description":"Get the latest independent tech news, reviews and analysis from Recode with the most informed and respected journalists in technology and media.","url":"http://www.recode.net","category":"technology","language":"en","country":"us","urlsToLogos":{"small":"","medium":"","large":""},"sortBysAvailable":["top"]},\
     {"id":"reddit-r-all","name":"Reddit /r/all","description":"Reddit is an entertainment, social news networking service, and news website. Reddit's registered community members can submit content, such as text posts or direct links.","url":"https://www.reddit.com/r/all","category":"general","language":"en","country":"us","urlsToLogos":{"small":"","medium":"","large":""},"sortBysAvailable":["top","latest"]},\
     {"id":"reuters","name":"Reuters","description":"Reuters.com brings you the latest news from around the world, covering breaking news in business, politics, entertainment, technology, video and pictures.","url":"http://www.reuters.com","category":"general","language":"en","country":"us","urlsToLogos":{"small":"","medium":"","large":""},"sortBysAvailable":["top","latest"]},\
     {"id":"spiegel-online","name":"Spiegel Online","description":"Deutschlands führende Nachrichtenseite. Alles Wichtige aus Politik, Wirtschaft, Sport, Kultur, Wissenschaft, Technik und mehr.","url":"http://www.spiegel.de","category":"general","language":"de","country":"de","urlsToLogos":{"small":"","medium":"","large":""},"sortBysAvailable":["top"]},\
     {"id":"t3n","name":"T3n","description":"Das Online-Magazin bietet Artikel zu den Themen E-Business, Social Media, Startups und Webdesign.","url":"http://t3n.de","category":"technology","language":"de","country":"de","urlsToLogos":{"small":"","medium":"","large":""},"sortBysAvailable":["top"]},\
     {"id":"talksport","name":"TalkSport","description":"Tune in to the world's biggest sports radio station - Live Premier League football coverage, breaking sports news, transfer rumours &amp; exclusive interviews.","url":"http://talksport.com","category":"sport","language":"en","country":"gb","urlsToLogos":{"small":"","medium":"","large":""},"sortBysAvailable":["top","latest"]},\
     {"id":"techcrunch","name":"TechCrunch","description":"TechCrunch is a leading technology media property, dedicated to obsessively profiling startups, reviewing new Internet products, and breaking tech news.","url":"https://techcrunch.com","category":"technology","language":"en","country":"us","urlsToLogos":{"small":"","medium":"","large":""},"sortBysAvailable":["top","latest"]},\
     {"id":"techradar","name":"TechRadar","description":"The latest technology news and reviews, covering computing, home entertainment systems, gadgets and more.","url":"http://www.techradar.com","category":"technology","language":"en","country":"us","urlsToLogos":{"small":"","medium":"","large":""},"sortBysAvailable":["top","latest"]},\
     {"id":"the-economist","name":"The Economist","description":"The Economist offers authoritative insight and opinion on international news, politics, business, finance, science, technology and the connections between them.","url":"http://www.economist.com","category":"business","language":"en","country":"gb","urlsToLogos":{"small":"","medium":"","large":""},"sortBysAvailable":["top","latest"]},\
     {"id":"the-guardian-au","name":"The Guardian (AU)","description":"Latest news, sport, comment, analysis and reviews from Guardian Australia","url":"https://www.theguardian.com/au","category":"general","language":"en","country":"au","urlsToLogos":{"small":"","medium":"","large":""},"sortBysAvailable":["top"]},\
     {"id":"the-guardian-uk","name":"The Guardian (UK)","description":"Latest news, sport, business, comment, analysis and reviews from the Guardian, the world's leading liberal voice.","url":"https://www.theguardian.com/uk","category":"general","language":"en","country":"gb","urlsToLogos":{"small":"","medium":"","large":""},"sortBysAvailable":["top","latest"]},\
     {"id":"the-hindu","name":"The Hindu","description":"The Hindu. latest news, analysis, comment, in-depth coverage of politics, business, sport, environment, cinema and arts from India's national newspaper.","url":"http://www.thehindu.com","category":"general","language":"en","country":"in","urlsToLogos":{"small":"","medium":"","large":""},"sortBysAvailable":["top","latest"]},\
     {"id":"the-huffington-post","name":"The Huffington Post","description":"The Huffington Post is a politically liberal American online news aggregator and blog that has both localized and international editions founded by Arianna Huffington, Kenneth Lerer, Andrew Breitbart, and Jonah Peretti, featuring columnists.","url":"http://www.huffingtonpost.com","category":"general","language":"en","country":"us","urlsToLogos":{"small":"","medium":"","large":""},"sortBysAvailable":["top"]},\
     {"id":"the-lad-bible","name":"The Lad Bible","description":"The LAD Bible is one of the largest community for guys aged 16-30 in the world. Send us your funniest pictures and videos!","url":"http://www.theladbible.com","category":"entertainment","language":"en","country":"gb","urlsToLogos":{"small":"","medium":"","large":""},"sortBysAvailable":["top","latest"]},\
     {"id":"the-new-york-times","name":"The New York Times","description":"The New York Times: Find breaking news, multimedia, reviews & opinion on Washington, business, sports, movies, travel, books, jobs, education, real estate, cars & more at nytimes.com.","url":"http://www.nytimes.com","category":"general","language":"en","country":"us","urlsToLogos":{"small":"","medium":"","large":""},"sortBysAvailable":["top"]},\
     {"id":"the-next-web","name":"The Next Web","description":"The Next Web is one of the world’s largest online publications that delivers an international perspective on the latest news about Internet technology, business and culture.","url":"http://thenextweb.com","category":"technology","language":"en","country":"us","urlsToLogos":{"small":"","medium":"","large":""},"sortBysAvailable":["latest"]},\
     {"id":"the-sport-bible","name":"The Sport Bible","description":"TheSPORTbible is one of the largest communities for sports fans across the world. Send us your sporting pictures and videos!","url":"http://www.thesportbible.com","category":"sport","language":"en","country":"gb","urlsToLogos":{"small":"","medium":"","large":""},"sortBysAvailable":["top","latest"]},\
     {"id":"the-telegraph","name":"The Telegraph","description":"Latest news, business, sport, comment, lifestyle and culture from the Daily Telegraph and Sunday Telegraph newspapers and video from Telegraph TV.","url":"http://www.telegraph.co.uk","category":"general","language":"en","country":"gb","urlsToLogos":{"small":"","medium":"","large":""},"sortBysAvailable":["top","latest"]},\
     {"id":"the-times-of-india","name":"The Times of India","description":"Times of India brings the Latest News and Top Breaking headlines on Politics and Current Affairs in India and around the World, Sports, Business, Bollywood News and Entertainment, Science, Technology, Health and Fitness news, Cricket and opinions from leading columnists.","url":"http://timesofindia.indiatimes.com","category":"general","language":"en","country":"in","urlsToLogos":{"small":"","medium":"","large":""},"sortBysAvailable":["top","latest"]},\
     {"id":"the-verge","name":"The Verge","description":"The Verge covers the intersection of technology, science, art, and culture.","url":"http://www.theverge.com","category":"technology","language":"en","country":"us","urlsToLogos":{"small":"","medium":"","large":""},"sortBysAvailable":["top","latest"]},\
     {"id":"the-wall-street-journal","name":"The Wall Street Journal","description":"WSJ online coverage of breaking news and current headlines from the US and around the world. Top stories, photos, videos, detailed analysis and in-depth reporting.","url":"http://www.wsj.com","category":"business","language":"en","country":"us","urlsToLogos":{"small":"","medium":"","large":""},"sortBysAvailable":["top"]},\
     {"id":"the-washington-post","name":"The Washington Post","description":"Breaking news and analysis on politics, business, world national news, entertainment more. In-depth DC, Virginia, Maryland news coverage including traffic, weather, crime, education, restaurant reviews and more.","url":"https://www.washingtonpost.com","category":"general","language":"en","country":"us","urlsToLogos":{"small":"","medium":"","large":""},"sortBysAvailable":["top"]},\
     {"id":"time","name":"Time","description":"Breaking news and analysis from TIME.com. Politics, world news, photos, video, tech reviews, health, science and entertainment news.","url":"http://time.com","category":"general","language":"en","country":"us","urlsToLogos":{"small":"","medium":"","large":""},"sortBysAvailable":["top","latest"]},\
     {"id":"usa-today","name":"USA Today","description":"Get the latest national, international, and political news at USATODAY.com.","url":"http://www.usatoday.com/news","category":"general","language":"en","country":"us","urlsToLogos":{"small":"","medium":"","large":""},"sortBysAvailable":["top","latest"]},\
     {"id":"wired-de","name":"Wired.de","description":"Wired reports on how emerging technologies affect culture, the economy and politics.","url":"https://www.wired.de","category":"technology","language":"de","country":"de","urlsToLogos":{"small":"","medium":"","large":""},"sortBysAvailable":["top","latest"]},\
     {"id":"wirtschafts-woche","name":"Wirtschafts Woche","description":"Das Online-Portal des führenden Wirtschaftsmagazins in Deutschland. Das Entscheidende zu Unternehmen, Finanzen, Erfolg und Technik.","url":"http://www.wiwo.de","category":"business","language":"de","country":"de","urlsToLogos":{"small":"","medium":"","large":""},"sortBysAvailable":["latest"]}\
   ]
  }
```

On opening the file, we can see that the response contains an array of json objects that represent all the news outlets that one can request news headlines from.

Next is a test of an API call requesting news from a selected source. Again we observe the response by opening the file.

```
{
   "status":"ok","source":"cnbc","sortBy":"top","articles":
   [\
     {"author":"Reuters","title":"'Singles Day' China shopping festival smashes record at the halfway mark","description":"Alibaba said its Singles Day sales surged past last year's total just after midday Saturday, hitting a record $18 billion.","url":"https://www.cnbc.com/2017/11/11/singles-day-china-shopping-festival-smashes-record-at-the-halfway-mark.html","urlToImage":"https://fm.cnbc.com/applications/cnbc.com/resources/img/editorial/2017/11/10/104835461-RTS1JDT7-singles-day.1910x1000.jpg","publishedAt":"2017-11-11T10:50:08Z"},\
     {"author":"The Associated Press","title":"Trump: Putin again denies meddling in 2016 election","description":"President Donald Trump said Saturday that Russia's Vladimir Putin again denied interfering in the 2016 U.S. elections.","url":"https://www.cnbc.com/2017/11/11/trump-putin-again-denies-meddling-in-2016-election.html","urlToImage":"https://fm.cnbc.com/applications/cnbc.com/resources/img/editorial/2017/08/18/104661874-RTS19PQA-vladimir-putin.1910x1000.jpg","publishedAt":"2017-11-11T12:07:32Z"},\
     {"author":"Jeff Cox","title":"GE limps into investor day with shareholders demanding answers on dividend and turnaround plan","description":"As General Electric limps into its investor day presentation Monday, it has gone from a paradigm of success to a morass of excess.","url":"https://www.cnbc.com/2017/11/10/ge-faces-investor-day-with-questions-about-its-past-and-future.html","urlToImage":"https://fm.cnbc.com/applications/cnbc.com/resources/img/editorial/2017/10/20/104786151-JohnFlannery2.1910x1000.jpg","publishedAt":"2017-11-10T16:16:05Z"},\
     {"author":"Sarah Whitten","title":"Here's where military service members can get freebies on Veterans Day","description":"Businesses across the country are saying \"thank you\" to Veterans on Friday by offering freebies to active and retired military members.","url":"https://www.cnbc.com/2016/11/10/heres-where-military-service-members-can-get-freebies-on-veterans-day.html","urlToImage":"https://fm.cnbc.com/applications/cnbc.com/resources/img/editorial/2016/11/10/104097817-GettyImages-496717392.1910x1000.jpg","publishedAt":"2016-11-10T18:30:41Z"},\
     {"author":"Morgan Brennan","title":"With an eye toward the North Korean threat, a 'missile renaissance' blooms in the US","description":"Raytheon is cranking out about 20 Standard Missile variants per month, as part of the effort to help repel a possible attack from North Korea.","url":"https://www.cnbc.com/2017/11/11/north-korea-threat-leads-to-a-us-missile-renaissance.html","urlToImage":"https://fm.cnbc.com/applications/cnbc.com/resources/img/editorial/2017/08/03/104631031-RTS1A3SU.1910x1000.jpg","publishedAt":"2017-11-11T14:00:56Z"},\
     {"author":"Larry Kudlow","title":"Larry Kudlow: A pro-growth GOP tax cut is on the way — this year","description":"One way or another, Congress will come up with a significant pro-growth bill, writes Larry Kudlow.","url":"https://www.cnbc.com/2017/11/11/larry-kudlow-pro-growth-gop-tax-cut-is-on-the-way--this-year.html","urlToImage":"https://fm.cnbc.com/applications/cnbc.com/resources/img/editorial/2017/11/02/104816986-GettyImages-869498942.1910x1000.jpg","publishedAt":"2017-11-11T14:22:58Z"},\
     {"author":"Reuters","title":"Trans-Pacific trade deal advances without United States","description":"Last-minute resistance from Canada had raised new doubts about its survival.","url":"https://www.cnbc.com/2017/11/11/trans-pacific-trade-deal-advances-without-united-states.html","urlToImage":"https://fm.cnbc.com/applications/cnbc.com/resources/img/editorial/2016/11/22/104123278-GettyImages-624177112.1910x1000.jpg","publishedAt":"2017-11-11T10:23:03Z"},\
     {"author":"Jacob Pramuk","title":"McConnell says he 'misspoke' about middle-class tax hikes","description":"Mitch McConnell told The New York Times that \"you can't guarantee that no one sees a tax increase.\"","url":"https://www.cnbc.com/2017/11/10/mitch-mcconnell-says-he-misspoke-about-republican-tax-plan.html","urlToImage":"https://fm.cnbc.com/applications/cnbc.com/resources/img/editorial/2017/09/22/104726784-RTX3FY40-mcconnell.1910x1000.jpg","publishedAt":"2017-11-10T23:04:47Z"},\
     {"author":"Erin Barry","title":"Start-up Dia&Co is catering the 70 percent of US women the fashion industry ignores","description":"There are more than 100 million plus-size women in the U.S., but finding fashionable clothes in their size can be a challenge.","url":"https://www.cnbc.com/2017/11/10/diaco-caters-to-the-70-percent-of-us-women-fashion-ignores.html","urlToImage":"https://fm.cnbc.com/applications/cnbc.com/resources/img/editorial/2017/02/22/104298276-Lydia-Gilbert--Nadia-Boujarwah-2_credit-DiaCo_r.1910x1000.jpg","publishedAt":"2017-11-11T14:01:01Z"},\
     {"author":"Elizabeth Gurdus","title":"Cramer shares a little-known investing concept critical to buying stocks","description":"Jim Cramer explained why the idea of suitability is crucial when it comes to individual investing.","url":"https://www.cnbc.com/2017/11/10/cramer-shares-an-investing-concept-critical-to-buying-stocks.html","urlToImage":"https://fm.cnbc.com/applications/cnbc.com/resources/img/editorial/2017/11/10/104835669-GettyImages-825493934.1910x1000.jpg","publishedAt":"2017-11-10T23:10:58Z"}\
   ]
  }
```

In this instance, the API response contains an array of json objects, where each is a news headline.

Lastly, let us investigate what response we get when an incorrect API call is made. Here are the results when an invalid API key is used.

```
{
  "status":"error",
  "code":"apiKeyInvalid",
  "message":"Your API key is invalid or incorrect. Check your key, or go to https://newsapi.org to create a free API
  key."
}
```

Another example of an error response, this time a request was made with incorrect source id.

```
{
 "status":"error",
 "code":"sourceDoesntExist",
 "message":"The news source you've entered doesn't exist. Check your spelling, or see /v1/sources for a list of valid sources."
}
```

After playing around with the API, we can now have a look at how this data can be quickly accessed through the use of a json parser.

### Parsing JSON meta data

In the MQL5.com codebase, there are two json libraries available. The first we will try is [json.mqh](https://www.mql5.com/en/code/11134). Compilation of a test script using this library revealed a number of errors and warnings. These errors were located in the library file itself. On a visit to the codebase webpage of the library, the author stipulates that the code is actively maintained on github. The author has not updated the code file available through the codebase.

Fixing the compilation errors was easy enough, all that was needed is the inclusion of the hash file directly in the json.mqh file. The warnings were all caused by implicit type casting. The corrected json.mqh file is given below.

```
// $Id: json.mqh 102 2014-02-24 03:39:28Z ydrol $
#include "hash.mqh"
#ifndef YDROL_JSON_MQH
#define YDROL_JSON_MQH

// (C)2014 Andrew Lord forex@NICKNAME@lordy.org.uk
// Parse a JSON String - Adapted for mql4++ from my gawk implementation
// ( https://code.google.com/p/oversight/source/browse/trunk/bin/catalog/json.awk )

/*
   TODO the constants true|false|null could be represented as fixed objects.
      To do this the deleting of _hash and _array must skip these objects.

   TODO test null

   TODO Parse Unicode Escape
*/

/*
   See json_demo for examples.

 This requires the hash.mqh ( http://codebase.mql4.com/9238 , http://lordy.co.nf/hash )

 */

enum ENUM_JSON_TYPE { JSON_NULL,JSON_OBJECT,JSON_ARRAY,JSON_NUMBER,JSON_STRING,JSON_BOOL };

class JSONString;
// Generic class for all JSON types (Number, String, Bool, Array, Object )
class JSONValue : public HashValue
  {
private:
   ENUM_JSON_TYPE    _type;

public:
                     JSONValue() {}
                    ~JSONValue() {}
   ENUM_JSON_TYPE getType() { return _type; }
   void setType(ENUM_JSON_TYPE t) { _type=t; }

   // Type methods
   bool isString() { return _type==JSON_STRING; }
   bool isNull() { return _type==JSON_NULL; }
   bool isObject() { return _type==JSON_OBJECT; }
   bool isArray() { return _type==JSON_ARRAY; }
   bool isNumber() { return _type==JSON_NUMBER; }
   bool isBool() { return _type==JSON_BOOL; }

   // Override in child classes
   virtual string toString()
     {
      return "";
     }

   // Some convenience getters to cast to the subtype.
   string getString()
     {
      return ((JSONString *)GetPointer(this)).getString();
     }
   double getDouble()
     {
      return ((JSONNumber *)GetPointer(this)).getDouble();
     }
   long getLong()
     {
      return ((JSONNumber *)GetPointer(this)).getLong();
     }
   int getInt()
     {
      return ((JSONNumber *)GetPointer(this)).getInt();
     }
   bool getBool()
     {
      return ((JSONBool *)GetPointer(this)).getBool();
     }

   // Static getters call by Array and Object. when returning child results.
   // They allow application to check value will be retrieved without halting program
   // (sometimes program halt is desired - rather an EA stop then continue working with faulty data)
   static bool getString(JSONValue *val,string &out)
     {
      if(val!=NULL && val.isString())
        {
         out = val.getString();
         return true;
        }
      return false;
     }
   static bool getBool(JSONValue *val,bool &out)
     {
      if(val!=NULL && val.isBool())
        {
         out = val.getBool();
         return true;
        }
      return false;
     }
   static bool getDouble(JSONValue *val,double &out)
     {
      if(val!=NULL && val.isNumber())
        {
         out = val.getDouble();
         return true;
        }
      return false;
     }
   static bool getLong(JSONValue *val,long &out)
     {
      if(val!=NULL && val.isNumber())
        {
         out = val.getLong();
         return true;
        }
      return false;
     }
   static bool getInt(JSONValue *val,int &out)
     {
      if(val!=NULL && val.isNumber())
        {
         out = val.getInt();
         return true;
        }
      return false;
     }
  };
// -----------------------------------------
....
```

The code of the test script newstest\_json is detailed below. The script simply reads a file that contains json meta data saved from an API call made from earlier testing. When the script is run, all the available news outlets contained in the data will be output to the terminal.

```
//+------------------------------------------------------------------+
//|                                                newstest_json.mq5 |
//|                                          Copyright 2017, ufranco |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2017, ufranco"
#property link      "https://www.mql5.com"
#property version   "1.00"
#include <json.mqh>
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
//---
   Test();
  }
//+------------------------------------------------------------------+
//| Test                                                             |
//+------------------------------------------------------------------+
bool Test()
  {

   string pStream;
   string sources_filename="sources.txt";

   int hFile,iStringSize;

// read file contents
   hFile=::FileOpen(sources_filename,FILE_TXT|FILE_READ|FILE_UNICODE);
   if(hFile==INVALID_HANDLE)
     {
      ::Print("error opening file "+sources_filename);
      return(false);
     }

   while(!::FileIsEnding(hFile))
     {
      iStringSize = ::FileReadInteger(hFile, INT_VALUE);
      pStream    += ::FileReadString(hFile, iStringSize);
     }

   ::FileClose(hFile);

   Print("success opening and reading file");

   JSONParser *parser=new JSONParser();

   JSONValue *jv=parser.parse(pStream);

   if(jv==NULL)
     {
      Print("error:"+(string)parser.getErrorCode()+parser.getErrorMessage());
        } else {

      if(jv.isObject())
        {
         JSONObject *jo = jv;
         JSONArray  *jd =  jo.getArray("sources");

         for(int i=0;i<jd.size();i++)
           {
            Print(jd.getObject(i).getString("id"));
           }
        }
      delete jv;
     }
   delete parser;

   return(true);
  }
```

With the corrected file, the library functions well.

![library test result](https://c.mql5.com/2/30/jsontest.png)

Now we can take a look at the second json library [JAson.mqh](https://www.mql5.com/en/code/13663). A test using this library did not turn up any errors. It worked perfectly the first time. The script newstest\_JAson was used for testing this library.

```
//+------------------------------------------------------------------+
//|                                               newstest_JAson.mq5 |
//|                                          Copyright 2017, ufranco |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2017, ufranco"
#property link      "https://www.mql5.com"
#property version   "1.00"
#include <JAson.mqh>
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
//---
   Test();
  }
//+------------------------------------------------------------------+
//| Test                                                             |
//+------------------------------------------------------------------+
bool Test()
  {

   string pStream;
   string sources_filename="sources.txt";

   int hFile,iStringSize;

// read file contents
   hFile=::FileOpen(sources_filename,FILE_TXT|FILE_READ|FILE_UNICODE);
   if(hFile==INVALID_HANDLE)
     {
      ::Print("error opening file "+sources_filename);
      return(false);
     }

   while(!::FileIsEnding(hFile))
     {
      iStringSize = ::FileReadInteger(hFile, INT_VALUE);
      pStream    += ::FileReadString(hFile, iStringSize);
     }

   ::FileClose(hFile);

   Print("success opening and reading file");

   CJAVal  srce;

   if(!srce.Deserialize(pStream))
     {
      ::Print("Json deserialize error");
      return(false);
     }

   CJAVal *json_array=new CJAVal(srce["sources"]);

   for(int i=0;i<ArraySize(json_array.m_e);i++)
     {
      Print(json_array[i]["id"].ToStr());
     }

   delete json_array;

   return(true);
  }
```

And here are the results of the test.

![JAson.mqh library test results](https://c.mql5.com/2/30/jasontest.png)

Comparing this library to the first one, json.mqh, both have support for all json data types. The major difference is that json.mqh implements each json data type as a class, and defines several classes. Whereas in JAson.mqh, json data types are defined by a publicly accessible class property, therefore the library defines a single class.

Moving on, we can now program an application for MetaTrader 5 that displays the news. The application will use the JAson.mqh library. The application will be implemented as an Expert Advisor that will display a list of news sources. When an item in the list is selected, an adjacent textbox will display the latest news snippets available from the outlet.

### The CNewsFeed class

In a previous article, I used the standard library to build a graphical user interface. This time around I wish to explore the use of [Anatoli Kazharski](https://www.mql5.com/en/users/tol64 "Anatoli Kazharski (tol64)")'s library. With the  library seemingly in constant beta with the author providing regular updates and adding new features, there exists a number of versions of the extensive library. I chose to use the version provided in the article [Graphical Interfaces XI:Refactoring the library code](https://www.mql5.com/en/articles/3365 "GUI library"). I feel, it provides everything that we need and nothing extra that we do not need.

Our application will be pretty basic, there will be no need for a menu bar or any tabs. Let us begin with the include file which will contain the main class for creating the application.

```
//+------------------------------------------------------------------+
//|                                              NewsFeedProgram.mqh |
//|                                          Copyright 2017, ufranco |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2017, ufranco"
#property link      "https://www.mql5.com"
#include <EasyAndFastGUI\WndEvents.mqh>
#include <EasyAndFastGUI\TimeCounter.mqh>
#include <JAson.mqh>

#define BASE_URL "https://newsapi.org/v1/"
#define SRCE "sources?"
#define ATCLE "articles?source="
#define LATEST "&sortBy=latest"
#define API_KEY "&apiKey=484c84eb9765418fb58ea936908a47ac"
//+------------------------------------------------------------------+
//| Class for creating an application                                |
//+------------------------------------------------------------------+
class CNewsFeed : public CWndEvents

```

NewsFeedprogram.mqh will include GUI library and also the json library. Just as in the script, the directives store components of a URL.

```
//+------------------------------------------------------------------+
//|                                              NewsFeedProgram.mqh |
//|                                          Copyright 2017, ufranco |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2017, ufranco"
#property link      "https://www.mql5.com"
#include <EasyAndFastGUI\WndEvents.mqh>
#include <EasyAndFastGUI\TimeCounter.mqh>
#include <JAson.mqh>

#define BASE_URL "https://newsapi.org/v1/"
#define SRCE "sources?"
#define ATCLE "articles?source="
#define LATEST "&sortBy=latest"
#define API_KEY "&apiKey=484c84eb9765418fb58ea936908a47ac"
//+------------------------------------------------------------------+
//| Class for creating an application                                |
//+------------------------------------------------------------------+
class CNewsFeed : public CWndEvents
  {
private:
   //--- Time counters
   CTimeCounter      m_counter; // for updating the items in the status bar
   //--- Main window
   CWindow           m_window;
   //--- Status bar
   CStatusBar        m_status_bar;
   //--- List views
   CListView         m_listview;
   //--- Edits
   CTextBox          m_text_box;
   //--- Main Json objects
   CJAVal            srce;
   CJAVal            js;
   //--- Json pointers to reference nested elements
   CJAVal           *articles;
   CJAVal           *articlesArrayElement;
   CJAVal           *sources;
   CJAVal           *sourcesArrayElement;
```

The main class CNewsFeed is a descendent of CWndEvents. Its private properties are control components that make up the application, i.e. the main window form, to which the list view, textbox and status bar are attached. A time counter is included for updating the status bar. The rest of the private properties of type CJAVal enable the Json parser, srce and js properties which will contain json objects returned from a call for news sources and specific news from a certain news outlet respectively. The remaining properties are pointers which refer to nested json objects.

The rest of the class is shown below.

```
//+------------------------------------------------------------------+
//|                                              NewsFeedProgram.mqh |
//|                                          Copyright 2017, ufranco |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2017, ufranco"
#property link      "https://www.mql5.com"
#include <EasyAndFastGUI\WndEvents.mqh>
#include <EasyAndFastGUI\TimeCounter.mqh>
#include <JAson.mqh>

#define BASE_URL "https://newsapi.org/v1/"
#define SRCE "sources?"
#define ATCLE "articles?source="
#define LATEST "&sortBy=latest"
#define API_KEY "&apiKey=484c84eb9765418fb58ea936908a47ac"
//+------------------------------------------------------------------+
//| Class for creating an application                                |
//+------------------------------------------------------------------+
class CNewsFeed : public CWndEvents
  {
private:
   //--- Time counters
   CTimeCounter      m_counter; // for updating the items in the status bar
   //--- Main window
   CWindow           m_window;
   //--- Status bar
   CStatusBar        m_status_bar;
   //--- List views
   CListView         m_listview;
   //--- Edits
   CTextBox          m_text_box;
   //--- Main Json objects
   CJAVal            srce;
   CJAVal            js;
   //--- Json pointers to reference nested elements
   CJAVal           *articles;
   CJAVal           *articlesArrayElement;
   CJAVal           *sources;
   CJAVal           *sourcesArrayElement;
public:
                     CNewsFeed(void);
                    ~CNewsFeed(void);
   //--- Initialization/deinitialization
   bool              OnInitEvent(void);
   void              OnDeinitEvent(const int reason);
   //--- Timer
   void              OnTimerEvent(void);
   //--- Chart event handler
   virtual void      OnEvent(const int id,const long &lparam,const double &dparam,const string &sparam);

   //--- Create the graphical interface of the program
   bool              CreateGUI(void);
private:
   //--- Main window
   bool              CreateWindow(const string text);
   //--- Status bar
   bool              CreateStatusBar(const int x_gap,const int y_gap);
   //--- List views
   bool              CreateListView(const int x_gap,const int y_gap);
   //--- Edits
   bool              CreateTextBox(const int x_gap,const int y_gap);
   //--- Fill sources Json object
   bool              PrepareSources(void);
   //--- Fill articles Json object based on selected element from sources Json object
   bool              PrepareArticles(int listview_index);
   //--- Download data
   string            DownLoad(string url,string file_name);
  };
```

Detailed descriptions of methods related to the graphical user interface can be found in articles written by [Anatoli Kazharski](https://www.mql5.com/en/users/tol64 "Anatoli Kazharski (tol64)").

### Modifying the CTextBox class

There is only one part of the library i want to discuss, because i had to make some modifications to enable the functionality i needed. This relates specifically to the textbox control. During testing i noticed that the update() method of textbox did not react as expected when the textbox was refreshed with new content. This problem was solved by adding a few private methods  from the same class to the update method as shown below.

Refreshing the content in the textbox brought on other problems as well.These problems come about when minimizing and maximizing the application. Again the culprit was the textbox. These problems mostly relate to autoresizing.

```
//+------------------------------------------------------------------+
//| Updating the control                                             |
//+------------------------------------------------------------------+
void CTextBox::Update(const bool redraw=false)
  {
//--- Redraw the table, if specified
   if(redraw)
     {
      //--- Draw
      //ChangeTextBoxSize();
      WordWrap();
      Draw();
      CalculateTextBoxSize();
      ChangeScrollsSize();
      //--- Apply
      m_canvas.Update();
      m_textbox.Update();
      return;
     }
//--- Apply
   m_canvas.Update();
   m_textbox.Update();
  }
```

First the issues with the scroll bars. Horizontal and vertical scrolling seems to partially malfunction after refreshing the textbox, i say partially because when a certain point was reached while scrolling content would suddenly disappear. I noticed the problems occured whenever  new content written to the textbox had larger " size " parameters ( ie maximum line width and the number of lines that make up the text)  relative to the initial text displayed during initialization. To avoid these problems i enabled word wrap mode so there would be no need for a horizontal scroll bar and initialized the text box with a large number of empty lines. Autoresizing of the listview and textbox in the vertical plane was disabled.

Therefore if you want to run the application and you have downloaded the Gui library just replace the TextBox.mqh file with the one shown at the end of the aricle.

### Methods for processing Json Objects

```
//+------------------------------------------------------------------+
//| method for web requests and caching data                         |
//+------------------------------------------------------------------+
string CNewsFeed::DownLoad(string url,string file_name="")
  {

// if terminal is not connected ea assumes there is no connectivity
   if(!(bool)::TerminalInfoInteger(TERMINAL_CONNECTED))return(NULL);

   string cookie=NULL,headers,pStream;
   char post[],result[];
   int res,hFile;

   ::ResetLastError();
   int timeout=5000;
// web request
   res=::WebRequest("GET",url,cookie,NULL,timeout,post,0,result,headers);

   if(res==-1)
     {
      ::Print("WebRequest failure");
      return(NULL);
     }

// downloaded data stream
   pStream=::CharArrayToString(result,0,-1,CP_UTF8);

   if(file_name!="")
     {

      hFile=::FileOpen(file_name,FILE_BIN|FILE_WRITE);

      if(hFile==INVALID_HANDLE)
        {
         return(pStream);
         ::Print("Invalid file handle - "+file_name+" - could not save data to file");
        }
      // write downloaded data to a file
      ::FileWriteString(hFile,pStream);
      ::FileClose(hFile);
     }
//Print("download success");
   return(pStream);
  }
```

Download() - Used to make API calls via webrequest function and returns the response as a string value. If a second string parameter is specified, the string response will be saved to a file. If an error occurs NULL value is returned.

```
//+------------------------------------------------------------------+
//| downloads data to fill sources json object                       |
//+------------------------------------------------------------------+
bool CNewsFeed::PrepareSources(void)
  {
   string sStream;
   int    iStringSize,hFile;

   string sources_filename="sources.txt";
// download data
   sStream=DownLoad(BASE_URL+SRCE,sources_filename);

   if(sStream==NULL)
     {
      if(!::FileIsExist(sources_filename))
        {
         ::Print("error : required file does not exit");
         return(false);
        }
      // read file contents
      hFile=::FileOpen(sources_filename,FILE_TXT|FILE_READ|FILE_UNICODE);
      if(hFile==INVALID_HANDLE)
        {
         ::Print("error opening file "+sources_filename);
         return(false);
        }

      while(!::FileIsEnding(hFile))
        {
         iStringSize = ::FileReadInteger(hFile, INT_VALUE);
         sStream    += ::FileReadString(hFile, iStringSize);
        }

      ::FileClose(hFile);
     }
// parse json data
   if(!srce.Deserialize(sStream))
     {
      ::Print("Json deserialize error");
      return(false);
     }
// assign json object to sources
   if(srce["status"].ToStr()=="ok")
     {
      sources=srce["sources"];
      return(true);
     }
   else
     {
      Print("error json api access denied");
      return(false);
     }
  }
```

PrepareSources() is called during EA initialization once to make an API request for the available news sources. The response is saved to a file and parsed using Deserialize method of the Json parser. From here, the sources array of json objects is assigned to the sources pointer. If there are no connectivity and the sources the .txt file has not been created yet, the EA will not initialize successfully.

```
//+------------------------------------------------------------------+
//| downloads data to fill articles json object                      |
//+------------------------------------------------------------------+
bool CNewsFeed::PrepareArticles(int listview_index)
  {
   string sStream,id;
   int iStringSize,hFile;
// check sources json object
   if(sources==NULL)
     {
      ::Print("Invalid pointer access");
      return(false);
     }

// check index
   if(listview_index>=::ArraySize(sources.m_e))
     {
      Print("invalid array index reference");
      return(false);
     }
// json objects assignment to reference sources array elements
   sourcesArrayElement=sources[listview_index];
// get name of news source
   id=sourcesArrayElement["id"].ToStr();
// reset sourcesArrayElement json object
   sourcesArrayElement=NULL;

// download data for specific news source
   sStream=DownLoad(BASE_URL+ATCLE+id+API_KEY,id+".txt");
   if(sStream==NULL)
     {
      if(!::FileIsExist(id+".txt"))
        {
         ::Print("error : required file does not exit");
         return(false);
        }

      // read json data file
      hFile=::FileOpen(id+".txt",FILE_TXT|FILE_READ|FILE_UNICODE);
      if(hFile==INVALID_HANDLE)
        {
         ::Print("error opening file "+id+".txt");
         return(false);
        }

      while(!::FileIsEnding(hFile))
        {
         iStringSize = ::FileReadInteger(hFile, INT_VALUE);
         sStream    += ::FileReadString(hFile, iStringSize);
        }

      ::FileClose(hFile);
     }

// parse json file
   if(!js.Deserialize(sStream))
     {
      ::Print("Json deserialize error");
      return(false);
     }
// assign json object to articles pointer
   if(js["status"].ToStr()=="ok")
     {
      articles=js["articles"];
      return(true);
     }
   else
     {
      Print("error json api access denied");
      return(false);
     }
  }
```

PrepareArticles() - this function is used in the main chart event handler method. To make an API request for news from a specified news outlet before it can be displayed in the text box. An integer value passed to function represents the index of a selected list view item. This index is used to identify the chosen news outlet, so the proper API request URL can be constructed. The API response is handled in the same manner it is processed in the PrepareSources method.

Also note that both of the methods just described can function without connectivity if and only if an API request corresponds with an existing file.

### Refreshing the text box

Next, let us take a look at the OnchartEvent method. When a list view item is clicked, first the text box is auto scrolled all the way to the top. This is necessary to avoid new content from being displayed incorrectly. Methods of the list view object SelectedItemText() and SelectedItemIndex() are used to get the name and index of the clicked list view item which in this case defines the name of the chosen news outlet and its position in the sources array of the srce json object. From this infomation, the proper URL can be built to make an API request for news headlines using PrepareArtilces() method. If successful, the text box is refreshed with the latest headlines, otherwise the text box will display an error message.

```
//+------------------------------------------------------------------+
//| Chart event handler                                              |
//+------------------------------------------------------------------+
void CNewsFeed::OnEvent(const int id,const long &lparam,const double &dparam,const string &sparam)
  {
// if one of the news sources is selected (clicked)
   if(id==CHARTEVENT_CUSTOM+ON_CLICK_LIST_ITEM)
     {
      string nameofobject="";        //name of selected news source
      int selected,asize,g;          //selected - index of selected list item , asize - array size of articles json object array , g -array index
      bool ar=false;                 // ar - return value of prepareArticles method
                                     // first autoscroll the text box to the begining , only if the vertical scroll is needed
      if(m_text_box.GetScrollVPointer().IsScroll())
         m_text_box.VerticalScrolling(0);
      //---
      nameofobject=m_listview.SelectedItemText();
      //---
      selected=m_listview.SelectedItemIndex();
      //---
      ar=PrepareArticles(selected);
      //---
      asize=(articles!=NULL)? ::ArraySize(articles.m_e):0;
      //---delete current contents of the text box
      m_text_box.ClearTextBox();
      //--- add heading for new text box contents
      m_text_box.AddText(0,nameofobject+" Top HeadLines:");
      //--- depending on the success of PrepareArticles method text box contents are filled
      if(asize>0 && ar)// if PrepareArticles is successful
        {
         string descrip,des;
         for(g=0; g<asize;g++)
           {
            // set json object to array element of articles array of json objects
            articlesArrayElement=articles[g];
            // get value
            des=articlesArrayElement["description"].ToStr();
            // set additional text to be displayed depending on its availability
            descrip=(des!="null" && des!="")? " -> "+des:".";
            // add new text to text box
            m_text_box.AddLine(string(g+1)+". "+articlesArrayElement["title"].ToStr()+descrip);
           }
        }
      else // if PrepareArticles is not successful
        {
         asize=1; // set asize to one
         for(g=0; g<asize;g++)
           {
            // display error message on the text box
            m_text_box.AddLine("Error retrieving data from feed.");
           }
        }
      //-- redraw the text box
      m_text_box.Update(true);
      //-- reset value of articles object
      articles=NULL;
      //Print("clicked listview item is "+nameofobject);
      return;
     }
  }
```

That concludes the definition of the CNewsFeed class, so it can be included in an Expert Advisor.

### The NewsFeedProgram Expert Advisor

The code is shown below along with some screenshots of how the application will look like when running.

```
//+------------------------------------------------------------------+
//|                                               NewsFeedExpert.mq5 |
//|                                          Copyright 2017, ufranco |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2017, ufranco"
#property link      "https://www.mql5.com"
#property version   "1.00"
#include <NewsFeedProgram.mqh>
CNewsFeed program;
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {

   if(!program.OnInitEvent())
     {
      ::Alert("Check your internet connection and set up the terminal \n"+
              "for Web requests");
      return(INIT_FAILED);
     }

//--- Set up the trading panel
   if(!program.CreateGUI())
     {
      ::Print("Failed to create graphical interface!");
      return(INIT_FAILED);
     }
//--- Initialization successful
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//--- destroy timer
   program.OnDeinitEvent(reason);

  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//---

  }
//+------------------------------------------------------------------+
//| Timer function                                                   |
//+------------------------------------------------------------------+
void OnTimer()
  {
//---
   program.OnTimerEvent();
  }
//+------------------------------------------------------------------+
//| ChartEvent function                                              |
//+------------------------------------------------------------------+
void OnChartEvent(const int id,
                  const long &lparam,
                  const double &dparam,
                  const string &sparam)
  {
//---
   program.ChartEvent(id,lparam,dparam,sparam);
  }
```

![Initial look of application](https://c.mql5.com/2/30/api1.png)

![News display](https://c.mql5.com/2/30/api2.png)

### Conclusion

In this article, we have explored  the possibility of creating a custom news feed using a news web API. The demonstrated EA is very basic and can probably be expanded by enabling an auto update feature so that the app can display the latest news as it becomes available. I hope, it will be useful to someone.

_Please note that for the EA to work correctly, please download the GUI library. Then replace the TextBox.mqh file with the one attached to this article._

### Programs and Files used in the article

| Name | Type | Description |
| --- | --- | --- |
| JAson.mqh | Header file | Json serialization and deserialization Native MQL class |
| json.mqh | Header file | Json parser class |
| TextBox.mqh | Header file | Modified text box class for displaying formated text on the chart |
| NewsFeedProgram.mqh | Header file | Main class for the news feed EA |
| NewsFeedExpert.mq5 | Expert Advisor file | Expert Advisor implementing the main news feed class |
| NewsAPI\_test.mq5 | Script file | Script for testing API calls |
| newstest\_JAson.mq5 | Script file | Test script for accessing capabilities of the JAson.mqh library |
| newstest\_json.mq5 | Script file | Test script for accessing capabilites of the json.mqh library |

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/4149.zip "Download all attachments in the single ZIP archive")

[NewsFeedExpert.mq5](https://www.mql5.com/en/articles/download/4149/newsfeedexpert.mq5 "Download NewsFeedExpert.mq5")(2.48 KB)

[NewsAPI\_test.mq5](https://www.mql5.com/en/articles/download/4149/newsapi_test.mq5 "Download NewsAPI_test.mq5")(2.52 KB)

[NewsFeedProgram.mqh](https://www.mql5.com/en/articles/download/4149/newsfeedprogram.mqh "Download NewsFeedProgram.mqh")(35.64 KB)

[TextBox.mqh](https://www.mql5.com/en/articles/download/4149/textbox.mqh "Download TextBox.mqh")(282.87 KB)

[newstest\_JAson.mq5](https://www.mql5.com/en/articles/download/4149/newstest_jason.mq5 "Download newstest_JAson.mq5")(2.03 KB)

[newstest\_json.mq5](https://www.mql5.com/en/articles/download/4149/newstest_json.mq5 "Download newstest_json.mq5")(2.18 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Building Volatility models in MQL5 (Part I): The Initial Implementation](https://www.mql5.com/en/articles/20589)
- [Bivariate Copulae in MQL5 (Part 2): Implementing Archimedean copulae in MQL5](https://www.mql5.com/en/articles/19931)
- [Bivariate Copulae in MQL5 (Part 1): Implementing Gaussian and Student's t-Copulae for Dependency Modeling](https://www.mql5.com/en/articles/18361)
- [Dynamic mode decomposition applied to univariate time series in MQL5](https://www.mql5.com/en/articles/19188)
- [Singular Spectrum Analysis in MQL5](https://www.mql5.com/en/articles/18777)
- [Websockets for MetaTrader 5: Asynchronous client connections with the Windows API](https://www.mql5.com/en/articles/17877)
- [Resampling techniques for prediction and classification assessment in MQL5](https://www.mql5.com/en/articles/17446)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/223595)**
(3)


![Реter Konow](https://c.mql5.com/avatar/avatar_na2.png)

**[Реter Konow](https://www.mql5.com/en/users/peterkonow)**
\|
11 Jul 2018 at 20:40

Useful and interesting topic. Thank you for the informative article.


![Sergey Genikhov](https://c.mql5.com/avatar/2014/11/545DE5ED-8BD1.jpg)

**[Sergey Genikhov](https://www.mql5.com/en/users/seriygena)**
\|
18 Jan 2019 at 16:31

Hash file, not compiled. We'll look for the error.

[![Complilation errors](https://c.mql5.com/3/264/Hash_compilation_errors__1.PNG)](https://c.mql5.com/3/264/Hash_compilation_errors.PNG "https://c.mql5.com/3/264/Hash_compilation_errors.PNG")

[![1st error, '''' variable expected](https://c.mql5.com/3/264/1st__error__1.PNG)](https://c.mql5.com/3/264/1st__error.PNG "https://c.mql5.com/3/264/1st__error.PNG")

![Sergey Genikhov](https://c.mql5.com/avatar/2014/11/545DE5ED-8BD1.jpg)

**[Sergey Genikhov](https://www.mql5.com/en/users/seriygena)**
\|
18 Jan 2019 at 18:21

**Sergey Genikhov:**

Hash file, not compiled. We'll look for the error.

The first function has been fixed, we need to fix the next...

![fixed function](https://c.mql5.com/3/264/1st_func_fixed.PNG)

![Momentum Pinball trading strategy](https://c.mql5.com/2/30/gejnwlva_uo6trie37_Momentum_Pinball.png)[Momentum Pinball trading strategy](https://www.mql5.com/en/articles/2825)

In this article, we continue to consider writing the code to trading systems described in a book by Linda B. Raschke and Laurence A. Connors “Street Smarts: High Probability Short-Term Trading Strategies”. This time we study Momentum Pinball system: there is described creation of two indicators, trade robot and signal block on it.

![The NRTR indicator and trading modules based on NRTR for the MQL5 Wizard](https://c.mql5.com/2/30/qatis21ft_NRTR_2.png)[The NRTR indicator and trading modules based on NRTR for the MQL5 Wizard](https://www.mql5.com/en/articles/3690)

In this article we are going to analyze the NRTR indicator and create a trading system based on this indicator. We are going to develop a module of trading signals that can be used in creating strategies based on a combination of NRTR with additional trend confirmation indicators.

![Night trading during the Asian session: How to stay profitable](https://c.mql5.com/2/30/timezone.png)[Night trading during the Asian session: How to stay profitable](https://www.mql5.com/en/articles/4102)

The article deals with the concept of night trading, as well as trading strategies and their implementation in MQL5. We perform tests and make appropriate conclusions.

![Testing patterns that arise when trading currency pair baskets. Part II](https://c.mql5.com/2/29/LOGO__1.png)[Testing patterns that arise when trading currency pair baskets. Part II](https://www.mql5.com/en/articles/3818)

We continue testing the patterns and trying the methods described in the articles about trading currency pair baskets. Let's consider in practice, whether it is possible to use the patterns of the combined WPR graph crossing the moving average. If the answer is yes, we should consider the appropriate usage methods.

[Running robots on virtual hosting is easyFollow our step-by-step MetaTrader VPS guide for beginnersRead![](https://www.mql5.com/ff/sh/au4fqg4kms7s9mq1z2/01.png)![](https://www.mql5.com/ff/sh/au4fqg4kms7s9mq1z2/02.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/articles/13586&a=uzpprdshbcrtxvjxpmescehprypbymxc&s=516438f25b531570d9b7d49dcfb29c82fa1021f5ede6571df8026dbfbafcd13f&uid=&ref=https://www.mql5.com/en/articles/4149&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5068215843768170212)

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