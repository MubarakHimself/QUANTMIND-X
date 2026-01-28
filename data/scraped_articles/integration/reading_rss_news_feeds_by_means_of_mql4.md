---
title: Reading RSS News Feeds by Means of MQL4
url: https://www.mql5.com/en/articles/1366
categories: Integration
relevance_score: 0
scraped_at: 2026-01-24T14:07:26.362675
---

[![](https://www.mql5.com/ff/si/fx5m8s6u6uxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=ysducdhemkrdsdtzzbfkclolrllnhezk&s=33f180a31db6c3b846d77732b0bc78169421a47b8cf9f076ca717f4e4846d1c7&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=oazsyrdefgdxngurqlraablvbqwbveua&ssn=1769252845587426391&ssn_dr=0&ssn_sr=0&fv_date=1769252845&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F1366&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Reading%20RSS%20News%20Feeds%20by%20Means%20of%20MQL4%20-%20MQL4%20Articles&scr_res=1920x1080&ac=176925284554542778&fz_uniq=5083351596282157549&sv=2552)

MetaTrader 4 / Examples


### Introduction

This article deals with an example of reading RSS markup by means of MQL4 using the functions from the article [HTML Walkthrough Using MQL4](https://www.mql5.com/en/articles/1544). It is assumed that the reader has read the article or at least has the general understanding of the ideas described there.

### What Is RSS and Why Do We Need It in MQL4?

RSS is an XML format for transferring various data from one source to another.

RSS is actively used by news agencies, companies, as well as various news web sites.

RSS can be aggregated (or read) by a variety of special applications (readers) and delivered to users in a convenient form. In this article, we will try to make a work piece which can then be turned into a news indicator or just an RSS reader on MQL4 language. What kind of information are we interested in RSS? It is the news, of course.

As mentioned above, RSS is an XML document. So, what is XML?

Xml (eXtensible Markup Language) is a text format for storing structured data. The structure can be visually represented as a tree of elements. XML elements are described by the tags.

Below is an example of a simple XML document:

```
<!--?xml version="1.0" encoding="windows-1252"?-->
<weeklyevents>
        <event>
                <title>Rightmove HPI m/m</title>
                <country>GBP</country>
                <date><!--[CDATA[05-15-2011]]--></date>
                <time><!--[CDATA[23:01]]--></time>
                <impact><!--[CDATA[Medium]]--></impact>
                <forecast>
                <previous><!--[CDATA[1.7%]]--></previous>
        </forecast></event>
</weeklyevents>
```

### Implementation

As we can see from the above example, XML is somewhat similar to HTML. Therefore, in order not to "reinvent the wheel", we will use the code from the article [HTML Walkthrough Using MQL4](https://www.mql5.com/en/articles/1544 "The article on mql4 forum on html walkthrough").

The first thing we need to do is connect HTML walkthrough functions to our project (indicator). To do this, download ReportHTMLtoCSV-2.mq4 file and put it to experts/include folder. Since we are going to use the file as a function library, start() function should be commented out in it.

I would also suggest to rename the file (for example, into HTMLTagsLib.mq4) for more clarity.

The file is ready. Now, connect it to the indicator (the work piece file for the indicator is attached below):

```
#include <htmltagslib.mq4>
```

Now we need to include wininet.dll Windows standard library to work with the links:

```
#include <winuser32.mqh>
#import "wininet.dll"
  int InternetAttemptConnect(int x);
  int InternetOpenA(string sAgent, int lAccessType,
                    string sProxyName = "", string sProxyBypass = "",
                    int lFlags = 0);
  int InternetOpenUrlA(int hInternetSession, string sUrl,
                       string sHeaders = "", int lHeadersLength = 0,
                       int lFlags = 0, int lContext = 0);
  int InternetReadFile(int hFile, int& sBuffer[], int lNumBytesToRead,
                       int& lNumberOfBytesRead[]);
  int InternetCloseHandle(int hInet);
#import
```

We will use ReadWebResource(string url) function for reading URL. The function's operation is not a topic of this article. Therefore, we will not dwell on it.

We are only interested in the input and output arguments. The function receives a link to be read and returns the resource content as a string.

In order to analyze the tags, we will use two functions from HTMLTagsLib.mq4 file - FillTagStructure() and GetContent(). These functions are described in details in the article [HTML Walkthrough Using MQL4](https://www.mql5.com/en/articles/1544). It should be noted that input data for analysis is passed as an array. Therefore, after the data has been received, it should be converted into array using ReadWebResource(string url) function.

ArrayFromString() function will help us in that:

```
//+------------------------------------------------------------------+
int ArrayFromString(string & webResData[], string inputStr, string divider)
{
   if (inputStr == "")
   {
     Print ("Input string is not set");
     return(0);
   }
   if (divider == "")
   {
      Print ("Separator is not set");
      return(0);
   }
   int i, stringCounter = 0;

   string tmpChar, tmpString, tmpArr[64000];
   int inputStringLen = StringLen(inputStr);
   for (i = 0; i < inputStringLen; i++ )
   {
      tmpChar = StringSubstr(inputStr, i, 1);
      tmpString = tmpString + tmpChar;
      tmpArr[stringCounter] = tmpString;
      if (tmpChar == divider)
      {
          stringCounter++;
          tmpString = "";
      }
   }
   if (stringCounter > 0)
   {
      ArrayResize(webResData, stringCounter);
      for (i = 0; i < stringCounter; i++) webResData[i] = tmpArr[i];
   }
   return (stringCounter);
}
```

Three arguments are passed to the function's input. The first one is the link to the array where the function's operation result is stored, the second one is a string that should be converted into an array and the third one is a separator, by which the string is divided. The function returns the number of rows in the resulting array.

Now our data is ready for analysis.

In the next fragment, we analyze data and display the values of title and country tags in the terminal's console:

```
   string webRss = ReadWebResource(rssUrl);
   int i, stringsCount = ArrayFromString(webResData, webRss, ">");

   string tags[];    // array for storing the tags
   int startPos[][2];// tag start coordinates
   int endPos[][2];  // tag end coordinates

   FillTagStructure(tags, startPos, endPos, webResData);
   int tagsNumber = ArraySize(tags);

   string text = "";
   string currTag;
   int start[1][2];
   int end[1][2];

   for (i = 0; i < tagsNumber; i++)
      {
      currTag = tags[i];

      if (currTag == "<weeklyevents>")
         {
            Print("News block start;");
         }

      if (currTag == "<event>")
         {
            text = "";
            start[0][0] = -1;
            start[0][1] = -1;
         }

      if (currTag == "<title>")
         {// coordinates of the initial position for selecting the content between the tags
            start[0][0] = endPos[i][0];
            start[0][1] = endPos[i][1];
         }

      if (currTag == "</title>")
         {// coordinates of the end position for selecting the contents between the tags
            end[0][0] = startPos[i][0];
            end[0][1] = startPos[i][1];
            text = text + GetContent(webResData, start, end) + ";";
         }

      if (currTag == "<country>")
         {// coordinates of the initial position for selecting the content between the tags
            start[0][0] = endPos[i][0];
            start[0][1] = endPos[i][1];
         }

      if (currTag == "</country>")
         {// coordinates of the end position for selecting the contents between the tags
            end[0][0] = startPos[i][0];
            end[0][1] = startPos[i][1];
            text = text + GetContent(webResData, start, end) + " ;";
         }

      if (currTag == "</event>")
         {
            Print(text);
         }

      if (currTag == "</weeklyevents>")
         {
            Print("end of the news;");
         }

      }
```

Using FillTagStructure() function, we receive the number and the structure of the tags, while GetContent() function provides us with their value.

Script operation results:

![Script operation results](https://c.mql5.com/2/13/imagez4.png)

Fig. 1. NewsRss script operation results

In the results, we can see the news title and the currency symbol of the country the news is related to.

### Conclusions

We have examined the way of reading RSS by means of MQL4 using the functions for HTML tags analysis. The drawbacks of this method are described in details in the article [HTML Walkthrough Using MQL4](https://www.mql5.com/en/articles/1544). I would also like to add that one of the drawbacks of the method is an "inconvenience" of using the functions in the code in contrast to other standard libraries for reading XML.

Now that the article and the script have been completed, I am going to consider connection of the external library for working with XML. As for the advantages, I would name implementation speed as one of them.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/1366](https://www.mql5.com/ru/articles/1366)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/1366.zip "Download all attachments in the single ZIP archive")

[HTMLTagsLib.mqh](https://www.mql5.com/en/articles/download/1366/HTMLTagsLib.mqh "Download HTMLTagsLib.mqh")(10.02 KB)

[NewsRSS.mq4](https://www.mql5.com/en/articles/download/1366/NewsRSS.mq4 "Download NewsRSS.mq4")(6.03 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/39127)**
(4)


![Zsolt Haromszeki](https://c.mql5.com/avatar/2015/11/565B6CBD-2A42.jpg)

**[Zsolt Haromszeki](https://www.mql5.com/en/users/relative)**
\|
25 Jun 2015 at 09:42

Does anybody know a Build 600+ compatible RSS reader?

Update:

I'm looking for an MT4 version.


![Mohammad Hanif Ansari](https://c.mql5.com/avatar/2018/3/5AA7ABD0-121F.jpg)

**[Mohammad Hanif Ansari](https://www.mql5.com/en/users/ifctrader)**
\|
23 Sep 2015 at 07:44

**Lime:**

Does anybody know a Build 600+ compatible RSS reader?

Update:

I'm looking for an MT4 version.

You can order in freelance section.


![iJSmile](https://c.mql5.com/avatar/2020/7/5F1D903B-12DC.jpg)

**[iJSmile](https://www.mql5.com/en/users/ijsmile)**
\|
3 Jun 2016 at 21:49

Great article. It works. Thank you.


![agu2a](https://c.mql5.com/avatar/2016/11/5839125B-950A.jpg)

**[agu2a](https://www.mql5.com/en/users/agu2a)**
\|
4 Jan 2017 at 08:35

This is a really good article, but I noticed that ForexFactory feed doesn't include actual values, only previous and consensus. Is there a feed out there with actual data?


![How to Install and Use OpenCL for Calculations](https://c.mql5.com/2/0/avatar__4.png)[How to Install and Use OpenCL for Calculations](https://www.mql5.com/en/articles/690)

It has been over a year since MQL5 started providing native support for OpenCL. However not many users have seen the true value of using parallel computing in their Expert Advisors, indicators or scripts. This article serves to help you install and set up OpenCL on your computer so that you can try to use this technology in the MetaTrader 5 trading terminal.

![LibMatrix: Library of Matrix Algebra (Part One)](https://c.mql5.com/2/17/843_42.png)[LibMatrix: Library of Matrix Algebra (Part One)](https://www.mql5.com/en/articles/1365)

The author familiarizes the readers with a simple library of matrix algebra and provides descriptions and peculiarities of the main functions.

![Testing Expert Advisors on Non-Standard Time Frames](https://c.mql5.com/2/13/1099_8.gif)[Testing Expert Advisors on Non-Standard Time Frames](https://www.mql5.com/en/articles/1368)

It's not just simple; it's super simple. Testing Expert Advisors on non-standard time frames is possible! All we need to do is to replace standard time frame data with non-standard time frame data. Furthermore, we can even test Expert Advisors that use data from several non-standard time frames.

![Mechanical Trading System "Chuvashov's Triangle"](https://c.mql5.com/2/17/985_56.png)[Mechanical Trading System "Chuvashov's Triangle"](https://www.mql5.com/en/articles/1364)

Let me offer you an overview and the program code of the mechanical trading system based on ideas of Stanislav Chuvashov. Triangle's construction is based on the intersection of two trend lines built by the upper and lower fractals.

[![](https://www.mql5.com/ff/si/d9hnbkyp2d47h07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fsignals%2Fmt5%2Fpage1%3Fpreset%3D2%26utm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dmax.profit.signals%26utm_content%3Dsubscribe.signal%26utm_campaign%3D0622.MQL5.com.Internal&a=hgyovyikvykcdukcncnktswvlctghemf&s=545653d14172edfb3c9c02ca8e948778c29f9c1b70be9a587e8d4b040fb23539&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=vaxgzmurajxqnrkckgxoaibcreqjvtbc&ssn=1769252845587426391&ssn_dr=0&ssn_sr=0&fv_date=1769252845&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F1366&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Reading%20RSS%20News%20Feeds%20by%20Means%20of%20MQL4%20-%20MQL4%20Articles&scr_res=1920x1080&ac=17692528455456217&fz_uniq=5083351596282157549&sv=2552)

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