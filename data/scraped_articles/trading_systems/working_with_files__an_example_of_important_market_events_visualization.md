---
title: Working with Files. An Example of Important Market Events Visualization
url: https://www.mql5.com/en/articles/1382
categories: Trading Systems, Expert Advisors
relevance_score: 0
scraped_at: 2026-01-24T13:59:50.507620
---

[![](https://www.mql5.com/ff/sh/a27a2kwmtszm2m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
MQL5 Channels - Messenger for traders\\
\\
Subscribe to traders' channels or create your own.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=vpcudokyepxfrcxrpjcktglhsjlemtza&s=f08ad2c1289e29bd5630f1ef977aef297d5cdbfcb686faed4a4b0f1e276d3c4a&uid=&ref=https://www.mql5.com/en/articles/1382&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5083268359815960613)

MetaTrader 4 / Examples


Every trader knows that is is rather risky to open a position using only technical analysis ( **TA**). It would be more proper to open position using both fundamental and technical analysis. Most often, traders who work on TA use an events schedule in order to consider possibly volatility of the market and not to expose their positions to undue risks. It would be very convenient if all key events of the trade day were displayed in the price chart. For example:

15.05.2006;9:00; Industrial Production M.M ;Japan;March;3.40%;1.00%;

15.05.2006;16:30; New York Fed Survey ;USA;May;15.8;15;

15.05.2006;17:00; Netto Capital Flow ;USA;March;86.9Bln;80.0Bln;

16.05.2006;12:30; HICP Y.Y ;U.K.;April;1.80%;2.00%;

16.05.2006;13:00; ZEW Economic Sentiment;Germany;May;62.7;65;

16.05.2006;16:00; Wal-Mart Stores Report;USA;1qtr.;-0.61;-;

16.05.2006;16:30; PPI M.M ;USA;April;0.50%;0.70%;

16.05.2006;16:30; PPI ex. Mortgage and Energy M.M;USA;April;0.10%;0.20%;

16.05.2006;16:30; Housing Starts ;USA;April;1.960Mln;1.960Mln;

16.05.2006;16:30; Building Permits ;USA;April;2.094Mln;2.050Mln;

16.05.2006;17:15; Industrial Production M.M;USA;April;0.60%;0.40%;

16.05.2006;17:15; Capacity Utilization ;USA;April;81.30%;81.50%;

17.05.2006;8:30; Industrial Production (rev.) M.M ;Japan;March;0.20%;0.20%;

17.05.2006;12:30; Unemployment M.M ;U.K.;April;+12,600;+5,000;

17.05.2006;12:30; BoE Minutes 4.05 ;U.K.;-;-;-;

17.05.2006;13:00; Industrial Production M.M; Eurozone;March;0.00%;-0.20%;

17.05.2006;13:00; Industrial Production Y.Y ;Eurozone;March;3.20%;2.80%;

17.05.2006;13:00; HICP (rev.) Y.Y ;Eurozone;April;2.40%;2.40%;

17.05.2006;16:30; CPI M.M;USA;April;0.40%;0.50%;

17.05.2006;16:30; CPI ex. Mortgage and Energy M.M ;USA;April;0.30%;0.20%;

18.05.2006;12:30; Retail Sales M.M ;U.K.;April;0.70%;0.20%;

18.05.2006;12:30; Retail Sales Y.Y;U.K.;April;2.60%;2.60%;

18.05.2006;16:30; Initial Jobless Claims ;USA;8-14.05;324,000;320,000;

18.05.2006;17:30; Fed Chairman Bernanke Speaks ;USA;-;-;-;

18.05.2006;8:00; Leading Economic Indicators M.M ;USA;April;-0.10%;0.10%;

18.05.2006;20:00; Philadelphia Fed Survey ;USA;May;13.2;12;

19.05.2006;1:30; Alan Greenspen Speaks;USA;-;-;-;

19.05.2006;3:50; Gross Domestic Product (GDP) Q.Q ;Japan;1qrt.;1.30%;0.20%;

19.05.2006;8:00; Bank of Japan Meeting ;Japan;-;-;-;

19.05.2006;13:00; Current Account ;Eurozone;March;-4.5Bln;-3.0Bln;

To read the data from the file, it is necessary to connect to it first, i.e., to open it for reading. There is an operator used for this purpose in MQL4:

```
int FileOpen( string filename, int mode, int delimiter=';')
```

the parameters of which are: file name, type (a binary FILE\_BIN or a line-oriented FILE\_CSV with separators), access method (reading FILE\_READ or writing FILE\_WRITE), and delimiter character between line data. If the file is opened successfully, the identifier will be assigned with a unique value, otherwise the file identifier will be assigned with the value of -1. To refine the error data, one can use the GetLastError() function. The FileName variable is placed in the script heading.

```
#property show_inputs
extern string FileName = "week.txt";

…

   int handle;
   handle=FileOpen(FileName,FILE_CSV|FILE_READ,';');
   if(handle<1)
    {
     Print("File was not found: ", GetLastError());
     return(false);
    }
```

This is the way in which we connected to the file. The next stage will be reading of all data. We will read lines, then convert them into necessary types. Then we will try to write the filter of events displaying. To do so, we just need to put the variables into the script heading. The variables will show whether the event from this country should be displayed in the price chart.

```
    while(!FileIsEnding(handle))
   {
    string sDate=FileReadString(handle); // Date
    string sTime=FileReadString(handle); // Time
    string sCountry=FileReadString(handle); // Country
    string sPeriod=FileReadString(handle); // Period
    string sDescription=FileReadString(handle); // Description
    string sPrev=FileReadString(handle); // Prev
    string sForecast=FileReadString(handle); // Expected
    string sCurrent=FileReadString(handle); // Current value

    FileReadString(handle); // null


    Print(
      sDate+" "
      ,sTime+" "
      ,sCountry+" "
      ,sPeriod+" "
      ,sDescription+" "
      ,sForecast+" "
      ,sCurrent+" ");

    i++;
    datetime dt = StrToTime(sDate+" "+sTime);

         color c = Red;

         if (sCountry == "Japan") c = Yellow;
         if (sCountry == "USA") c = Brown;
         if (sCountry == "Germany") c = Green;
         if (sCountry == "Eurozone") c = Blue;
         if (sCountry == "U.K.") c = Orange;
         if (sCountry == "Canada") c = Gray;
         if (sCountry == "Australia") c = DarkViolet;
         if (sCountry == "Sweden") c = FireBrick;
         if (sCountry == "South African Republic")
             c = DodgerBlue;
         if (sCountry == "Denmark") c = RosyBrown;
         if (sCountry == "Norway") c = HotPink;



         if ((sCountry == "Japan") && (!Japan)) continue;
         if ((sCountry == "USA") && (!USA)) continue;
         if ((sCountry == "Germany") && (!Germany)) continue;
         if ((sCountry == "Eurozone") && (!ES)) continue;
         if ((sCountry == "U.K.") && (!GB)) continue;
         if ((sCountry == "Canada") && (!Canada)) continue;
         if ((sCountry == "Australia") && (!Australia))
              continue;
         if ((sCountry == "Sweden") && (!Shweden)) continue;
         if ((sCountry == "South African Republic")&& (!UAR))
               continue;
         if ((sCountry == "Denmark") && (!Dania)) continue;
         if ((sCountry == "Norway") && (!Norvegia)) continue;




          if (DisplayText)
          {
          ObjectCreate("x"+i, OBJ_TEXT, 0, dt, Close[0]);
          ObjectSet("x"+i, OBJPROP_COLOR, c);
          ObjectSetText("x"+i,
            sDescription + " "
            + sCountry + " "
            + sPeriod + " "
            + sCurrent + " "
            + sForecast,
            8);
          ObjectSet("x"+i, OBJPROP_ANGLE, 90);
          }


          ObjectCreate(sCountry+" "+i, OBJ_VLINE, 0, dt,
                       Close[0]);
          ObjectSet(sCountry+" "+i, OBJPROP_COLOR, c);
          ObjectSet(sCountry+" "+i, OBJPROP_STYLE, STYLE_DOT);
          ObjectSet(sCountry+" "+i, OBJPROP_BACK, true);
          ObjectSetText(sCountry+" "+i,
            sDescription + " · "
            + sPeriod + " · "
            + sCurrent + " · "
            + sForecast,
            8);

   }
```

Now, you can always see the forthcoming events and their impact on the market. The ready script code and data files are attached to this article. Please do not forget that scripts should be placed in the Experts/Scripts directory, and data files should be in Experts/Files. The date format (YYYY.MM.DD HH:MM) and delimiters may not be forgotten. The work of the script is illustrated below.

![](https://c.mql5.com/2/13/11283e29.gif)

![](https://c.mql5.com/2/13/27283t29.gif)

![](https://c.mql5.com/2/13/main.jpg)

Working with files in MQL4 gives users a multitude of opportunities: both connection of the terminal to external data feeds and simplifying or optimization of work with the trading terminal. Examples are both supporting of logs and possibility to display support/resistance levels getting data directly from WorldWide Web. It means that the user is unrestricted to choose methods of working with files. Automated opening and managing positions should contribute to decreasing of traders' stresses and, in their turn, allow to analyze more factors influencing trading. Actually, this is what MQL4 is intended for.

Below are all MQL4 functions used to work with files. More details can be found in MQL4 documentation.

```
FileClose
FileDelete
FileFlush
FileIsEnding
FileIsLineEnding
FileOpen
FileOpenHistory
FileReadArray
FileReadDouble
FileReadInteger
FileReadNumber
FileReadString
FileSeek
FileSize
FileTell
FileWrite
FileWriteArray
FileWriteDouble
FileWriteInteger
FileWriteString

                                                                                                     Translated from Russian by MetaQuotes Software Corp.
                                                                                                     Original article: /ru/articles/1382
```

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/1382](https://www.mql5.com/ru/articles/1382)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/1382.zip "Download all attachments in the single ZIP archive")

[ReadFromFile.mq4](https://www.mql5.com/en/articles/download/1382/ReadFromFile.mq4 "Download ReadFromFile.mq4")(3.12 KB)

[week.txt](https://www.mql5.com/en/articles/download/1382/week.txt "Download week.txt")(2.22 KB)

[week1.txt](https://www.mql5.com/en/articles/download/1382/week1.txt "Download week1.txt")(1.86 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [MQL4 as a Trader's Tool, or The Advanced Technical Analysis](https://www.mql5.com/en/articles/1410)
- [Expert System 'Commentator'. Practical Use of Embedded Indicators in an MQL4 Program](https://www.mql5.com/en/articles/1406)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/39204)**
(9)


![](https://c.mql5.com/avatar/avatar_na2.png)

**\[Deleted\]**
\|
17 Oct 2006 at 18:53

```
Hello, I am trying to use your script for news. I have created
very short txt file with news events.  It is:
```

```
16.10.2006; 3:00am  ;USA; 0       ; 0    ;0
16.10.2006; 5:30am  ;USA; 0.008 ; 0.005 ;-0.003
16.10.2006; 5:30am  ;USA; 13.8 ; 11 ;22.9
16.10.2006; 10:00am ;USA; 0 ; 0 ;0
16.10.2006; 10:30am ;USA; 0 ; 0 ;0
16.10.2006; 12:40pm ;USA; 0 ; 0 ;0
However when, I run the script nothing happens.
I have a 15 minute chart up for the
EUR USD.
Thanks so much
Steve
```

```
//+------------------------------------------------------------------+
//|                                                 ReadFromFile.mq4 |
//|                                                       ·•°njel°•· |
//|                                                     iamnotlinked |
//+------------------------------------------------------------------+
#property copyright "·•°njel°•·"
#property link      "iamnotlinked"
#property show_inputs

extern bool DisplayText = true;

extern bool Japan = true;
extern bool USA = true;
extern bool Germany = true;
extern bool ES = true;
extern bool GB = true;
extern bool Canada = true;

extern string FileName = "\\experts\week.txt";

int start()
  {
  Alert("hello");  // *********** this also does not fire off
  ObjectsDeleteAll();

   int handle;
   handle=FileOpen(FileName,FILE_CSV|FILE_READ,';');
   if(handle<1)
    {
     /////// ************* neither of these two alerts fires either.
	Alert("File not found, the last error is ", GetLastError());
     return(false);
    }

   if(handle>=1)
    {
     Alert("File found ", FileName);
     return(false);
    }


   int i= 0;
   while(!FileIsEnding(handle))
   {
    string sDate=FileReadString(handle); // Date
    string sTime=FileReadString(handle); // Time
    string sDescription=FileReadString(handle); // Description
    string sCountry=FileReadString(handle); // Country
    string sPeriod=FileReadString(handle); // Period
    string sCurrent=FileReadString(handle); // Current value
    string sForecast=FileReadString(handle); // Expected
    FileReadString(handle); // null

    i++;
    datetime dt = StrToTime(sDate+" "+sTime);

         color c = Red;

         if (sCountry == "ßïîíèÿ") c = Yellow;
         if (sCountry == "ÑØÀ") c = White;
         if (sCountry == "Ãåðìàíèÿ") c = Green;
         if (sCountry == "ÅÑ") c = Blue;
         if (sCountry == "Áðèòàíèÿ") c = Orange;
         if (sCountry == "Êàíàäà") c = Gray;


         if ((sCountry == "ßïîíèÿ") && (!Japan)) continue;
         if ((sCountry == "ÑØÀ") && (!USA)) continue;
         if ((sCountry == "Ãåðìàíèÿ") && (!Germany)) continue;
         if ((sCountry == "ÅÑ") && (!ES)) continue;
         if ((sCountry == "Áðèòàíèÿ") && (!GB)) continue;
         if ((sCountry == "Êàíàäà") && (!Canada)) continue;


          if (DisplayText)
          {
          ObjectCreate("x"+i, OBJ_TEXT, 0, dt, Close[0]);
          ObjectSet("x"+i, OBJPROP_COLOR, c);
          ObjectSetText("x"+i, sDescription + " "+ sCountry + " " +
                        sPeriod + " " + sCurrent + " " + sForecast, 8);
          ObjectSet("x"+i, OBJPROP_ANGLE, 90);
          }


          ObjectCreate("y"+i, OBJ_VLINE, 0, dt, Close[0]);
          ObjectSet("y"+i, OBJPROP_COLOR, c);
          ObjectSet("y"+i, OBJPROP_STYLE, STYLE_DOT);
          ObjectSet("y"+i, OBJPROP_BACK, true);
          ObjectSetText("y"+i, sDescription + " "+ sCountry + " " +
                        sPeriod + " " + sCurrent + " " + sForecast, 8);
   }
   return(0);
  }
//+------------------------------------------------------------------+
```

![okwh](https://c.mql5.com/avatar/2011/9/4E7F67FD-3C19.jpg)

**[okwh](https://www.mql5.com/en/users/dxdcn)**
\|
7 Dec 2006 at 14:58

Ugly for so vis.

Could you mind to set a \* symble as a mark only,

then when one click that \* mark, then popup a small window to show details
like tootip ????


![Methasit Witayasumpunt](https://c.mql5.com/avatar/2018/10/5BB25458-DBEC.jpg)

**[Methasit Witayasumpunt](https://www.mql5.com/en/users/methasitwi)**
\|
19 Mar 2013 at 12:58

Thanks you very much. It's work for me.

Could anyone tell me what's indicator as below picture, I like it !

Thanks you everybody again.

Methasit W. / New programmer & [Gold](https://www.mql5.com/en/quotes/metals/xauusd "XAUUSD chart: technical analysis") Trader.


![](https://c.mql5.com/avatar/avatar_na2.png)

**\[Deleted\]**
\|
22 Jul 2013 at 23:51

any one got it working ? I put it in scripts folder, it starts but then returns File not found error ! week.txt is in expert/Files/ folder and also expert/, and test folder !


![](https://c.mql5.com/avatar/avatar_na2.png)

**\[Deleted\]**
\|
27 Jul 2013 at 13:37

Ok the readfromfile.mq4 attachement has a mistyping, now works very well, thanks a lot


![How to Use Crashlogs to Debug Your Own DLLs](https://c.mql5.com/2/13/153_6.gif)[How to Use Crashlogs to Debug Your Own DLLs](https://www.mql5.com/en/articles/1414)

25 to 30% of all crashlogs received from users appear due to errors occurring when functions imported from custom dlls are executed.

![Error 146 ("Trade context busy") and How to Deal with It](https://c.mql5.com/2/17/94_1.gif)[Error 146 ("Trade context busy") and How to Deal with It](https://www.mql5.com/en/articles/1412)

The article deals with conflict-free trading of several experts on one МТ 4 Client Terminal. It will be useful for those who have basic command of working with the terminal and programming in MQL 4.

![MagicNumber: "Magic" Identifier of the Order](https://c.mql5.com/2/13/105_2.gif)[MagicNumber: "Magic" Identifier of the Order](https://www.mql5.com/en/articles/1359)

The article deals with the problem of conflict-free trading of several experts on the same МТ 4 Client Terminal. It "teaches" the expert to manage only "its own" orders without modifying or closing "someone else's" positions (opened manually or by other experts). The article was written for users who have basic skills of working with the terminal and programming in MQL 4.

![Information Storage and View](https://c.mql5.com/2/13/128_4.gif)[Information Storage and View](https://www.mql5.com/en/articles/1405)

The article deals with convenient and efficient methods of information storage and viewing. Alternatives to the terminal standard log file and the Comment() function are considered here.

[Best articles and CodeBase updates in MQL5.community channelsFollow us to ensure you never miss out on important updates![](https://www.mql5.com/ff/sh/n9yf51p2srwzfqh5z2/02.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=dgazvhktsxqakdvarucjbvmvzenwlyje&s=98a038fe082e458df8c4a1d8e116e3a6646fd5517f06e48b2356b7ee005817d6&uid=&ref=https://www.mql5.com/en/articles/1382&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5083268359815960613)

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