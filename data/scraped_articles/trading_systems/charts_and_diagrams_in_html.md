---
title: Charts and diagrams in HTML
url: https://www.mql5.com/en/articles/244
categories: Trading Systems, Integration
relevance_score: 0
scraped_at: 2026-01-24T13:56:03.694659
---

[![](https://www.mql5.com/ff/si/3fgkjn78mkxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=ocndbzpeklfncxysjbwfhhbalbrsdbtv&s=a4309643278437a00bdd33c5809fc6b4b4032749c00fccd07b3b84e7b8b45126&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=xaffilgaguvhkcrnzudepyafxrodvgey&ssn=1769252162163906882&ssn_dr=0&ssn_sr=0&fv_date=1769252162&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F244&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Charts%20and%20diagrams%20in%20HTML%20-%20MQL5%20Articles&scr_res=1920x1080&ac=17692521622307871&fz_uniq=5083222068658444097&sv=2552)

MetaTrader 5 / Examples


### Introduction

Most likely, [MetaTrader 5](https://www.metatrader5.com/ "https://www.metatrader5.com/") is a fully self-sufficient product, and does not need additional extensions. MetaTrader 5 provides connection with the broker, displays quotes, allows us to use a variety of indicators for market analysis, and, of course, gives the trader an opportunity to make [trade operations.](https://www.mql5.com/en/docs/constants/tradingconstants/enum_trade_request_actions) It is quite clear that since MetaTrader 5 is focused primarily on making trade comfortable, it can not, and technically should not, be an absolutely universal tool, designed for research, analysis of mathematical methods, creation of multimedia content, and so on.

Moreover, the excessive universality of a software product ultimately leads to a decrease in its efficiency, reliability, and security. On the other hand, in certain cases, the user may need some additional features, especially traders are people with various areas of expertise and educational backgrounds. Therefore, any additional features may increase the attractiveness of the trading platform, if they, of course, are achieved in a fairly simple way, and not at the expense of its reliability and safety.

In this article we will consider one of the such supplements, which provide the opportunity to create and show the [charts and diagrams](https://www.mql5.com/en/articles/102) based on the data, obtained from the terminal.

Each program must do what it does best. If we adhere to this principle, then let us make MetaTrader 5 responsible for trading with the broker, collecting and processing incoming information, and use a different, intended for these purposes program, for the graphical display of this information.

### WEB-browser

Today it is difficult to find a computer that does not have an installed WEB-browser. For a long time browsers have been evolving and improving. Modern browsers are quite reliable, stable in their work, and most importantly, free. Taking into account that a WEB-browser is practically the basic tool for accessing the Internet, the most of the users are familiar with it, and experience little difficulties when using it.

The capabilities of modern browsers are so wide that we have gotten used to watching videos, listening to music, playing games, and doing a number of other activities via a WEB-browser. Thus, today a WEB-browser is a well-developed tool for displaying different types of information that can be presented in various formats.

It can not be left unmentioned, that there are currently several popular WEB-browsers: [InternetExplorer](https://www.mql5.com/go?link=http://www.microsoft.com/windows/internet-explorer/default.aspx "http://www.microsoft.com/windows/internet-explorer/default.aspx"), [Mozilla Firefox](https://www.mql5.com/go?link=https://www.mozilla.org/en-US/firefox/ "http://www.firefox.com/"), [Google Chrome](https://www.mql5.com/go?link=http://www.google.com/chrome/ "http://www.google.com/chrome/"), and [Opera](https://www.mql5.com/go?link=http://www.opera.com/ "http://www.opera.com/"). These browsers may differ significantly from each other in the aspect of software implementation and user interfaces. However, theoretically, they should fully support the basic standards adopted in the network for exchanging information, which primarily concerns the standards of the language of HTML.

In practice, despite the efforts of developers, browsers still have some individual characteristics in terms of implementation of certain protocols or technologies. If we decide that a particular browser, due to its individual features does not suit us, then this problem is easily fixed by installing one or several other WEB-browsers to our computer. Even ardent supporters of such browsers as Firefox at the same have at least Internet Explorer installed in their systems.

Despite the fact that WEB-browsers were developed as the client part, providing interaction with a remote server, they can also be used to display local information stored on your computer. An example of this can be viewing of WEB-pages, previously saved on your computer. The browser doesn't need the access to the Internet for working with local pages.

Thus, a WEB-browser, running in an offline-mode, is a very attractive candidate for the role of a program used to expand the graphics capabilities of the [MetaTrader 5](https://www.metatrader5.com/ "https://www.metatrader5.com/") client terminal. To use it you do not need to make expensive purchases, cumbersome and lengthy installations, nor learn how to use a new software product.

Therefore, further in this article we will consider the possibilities of using WEB-browsers for constructing charts and diagrams, based on the data, obtained in MetaTrader 5.

### HTML and JavaScript

By choosing to use a WEB-browser as our extension, let's define for ourselves the basic rule, which we will henceforth strictly adhere to - the display of the created HTML-pages must be carried out without the local or remote WEB-server. That is, we will not install on our computer any server software, and the displaying of our pages will not require an access to the network. The HTML-pages that we create should be displayed only by the means of the WEB-browser, and should be located on our computer. This rule will minimize the risk associated with the possible reduction in security due to accessing the outside network.

Using only the features of HTML 4 for information display, we can create WEB-pages with tables, formatted text, and images, but these opportunities can not fully satisfy us, since our goal is to build full-fledged charts and diagrams, based on data received from MetaTrader 5.

In the majority of the cases, what we see in the browser when traveling to different sites, is created using the extensions of HTML. In general, these extensions are executed on the server side, and for this reason is unfit for our purposes. Technologies that are able to work on the browser side and do not require server software, for example, Macromedia Flash, JavaScript, and Java, may be of interest to us.

If for execution, on the browser side, of applications Macromedia Flash and Java, we will, as a minimum, need the installation of additional plug-ins, then the user programs, written in JavaScript, are executed directly by the browser. All common WEB-browsers have their own built-in JavaScript interpreters. In order to avoid having to install any additional software or plug-ins, let's chose [JavaScript](https://en.wikipedia.org/wiki/JavaScript "https://en.wikipedia.org/wiki/JavaScript").

Thus, in what follows, we will only use MetaTrader 5 with [MQL5](https://www.mql5.com/) and a WEB-browser with [HTML](https://ru.wikipedia.org/wiki/HTML "https://ru.wikipedia.org/wiki/HTML") and [JavaScript](https://ru.wikipedia.org/wiki/JavaScript "https://ru.wikipedia.org/wiki/JavaScript"). No additional software will be needed. It should be recalled that an HTML-page is nothing more than a text file. Therefore, to create an HTML document, we can use any text editor. For example, we can create and edit HTML-code in [MetaEditor 5](https://www.metatrader5.com/en/automated-trading/metaeditor "https://www.metatrader5.com/en/automated-trading/metaeditor"). When writing this article, the editing of the HTML-code was done in the browser [Opera @ USB](https://www.mql5.com/go?link=http://www.opera-usb.com/ "http://www.opera-usb.com/") v10.63, which allows you to edit the page content, save the modified page, and preview the way it will be displayed.

A person, not familiar with the languages of HTML and JavaScript, may be reasonably apprehensive of the possible difficulties associated with mastering them. In order to facilitate our task, and avoid an in-depth study of HTML and JavaScript, we will try to use ready solutions based on this technology. Since in the scope of this article, our goal is limited only to the construction of charts and diagrams, we will use ready, written specially for this purpose JavaScript-libraries.

The [Emprise JavaScript Charts](https://www.mql5.com/go?link=http://www.ejschart.com/index.php "http://www.ejschart.com/index.php") is a quiet advanced graphics library. Perhaps the reader will be interested to get better acquainted with it through the provided link, however, this library is not quite free. Therefore, let's turn to free libraries, for example, [Dygraphs JavaScript Visualization Library](https://www.mql5.com/go?link=http://dygraphs.com/ "http://dygraphs.com/") and [Highcharts charting library](https://www.mql5.com/go?link=https://www.highcharts.com/ "http://www.highcharts.com/"). Dygraphs is attractive due to its compactness and simplicity, and the Highcharts library, in turn, includes a greater amount of features and looks more universal. Despite the fact that the Highcharts library is approximately 75 KB, and requires an additional [jQuery](https://www.mql5.com/go?link=http://jquery.com/ "http://jquery.com/") library, which is approximately another 70 KB, we will still make pick it as the library of our choice.

You can get acquainted with the Highcharts library on our website [http://www.highcharts.com/](https://www.mql5.com/go?link=https://www.highcharts.com/ "http://www.highcharts.com/") in the section "Demo Gallery". For each of the examples, by clicking "View options" you can see its source JavaScript-code. Detailed documentation about the library is located in the section "Documentation/Options Reference", in this section you can also find many examples of the use of different option. At the first glance, because of the abundance of the JavaScript-code, and the, unusual to an MQL-programmer syntax, the use of this library may seem quite complicated. But this is not quite so. Consider the first example of a simple HTML-file, which, through the means of the library, will display the chart.

As an example, let's create a text file named Test\_01.htm in the Notepad editor, and copy the following simple example of use of the library.

```
<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01//EN" "http://www.w3.org/TR/html4/strict.dtd">
<html>
<head>
<meta http-equiv="Content-Type" content="text/html; charset=utf-8">
<title>Example</title>
<!-- - -->
<script src="http://ajax.googleapis.com/ajax/libs/jquery/1.4.4/jquery.min.js"
             type="text/javascript"></script>
<script src="/js/highcharts.js" type="text/javascript"></script>
<!-- - -->
<script type="text/javascript">
var chart1;
$(document).ready(function(){
  chart1 = new Highcharts.Chart({
    chart: {renderTo: 'container1'},
    series: [{data: [29.9, 71.5, 106.4, 129.2, 144.0, 176.0, 135.6, 148.5, 216.4, 194.1, 95.6, 54.4]}]
  });
});
</script>
<!-- - -->
</head>
<body>
<div id="container1" style="width: 700px; height: 400px "></div>
</body>
</html>
```

The sample code is separated into four sections by the comments.

The first, upper part of the code contains the usual HTML-page tags. This part of the code is of no special interest for us right now.

It is followed by another part, which contains two tags <script>. In the first case, we give the browser a command to download from the ajax.googleapis.com website the library code jquery.min.js. The second case assumes that on the server side, the catalog /js/ contains the library highcharts.js, which the browser must download. Having previously decided that in the process of displaying our pages there should not be any access made to external sources, this part of the code will have to be changed.

After making the changes, this part of the code will look like this

```
<script src="jquery.min.js" type="text/javascript"></script>
<script src="highcharts.js" type="text/javascript"></script>
```

In this case, we give the command to download both libraries from the catalog that holds our HTML-file, that is, from the current catalog. In order for the libraries to be downloaded by the browser, they must first be downloaded from ajax.googleapis.com and [http://www.highcharts.com](https://www.mql5.com/go?link=https://www.highcharts.com/ "http://www.highcharts.com/") respectively, and copied into the same catalog where our HTML-file is located. Both of these libraries can also be found at the end of this article, in the attachments.

In the next section of the code an object of class Highcharts.Chart is created. The parameter "renderTo: 'container1'" indicates that the chart will be displayed in the HTML-element called "container1", and the parameter "data" defines the data that will be displayed on the chart. As we can see in this example, the data is defined in the same way as the parameters, - during the creation of an object of Highcharts.Chart class. By making simple changes, we locate the definition of the displayed data into a separate part of the code, this will allow us, in a case where we need to display multiple charts, to group their data.

In the last part of our example, the tag <div> declares an HTML-element called "container1", and the dimensions of this item are indicated. As mentioned earlier, this is the HTML-element that will be used to construct the chart, the size of which will be determined by the, specified in the tag <div>, size of the element "container1".

Taking into account the made changes, the code of our example will look as follows:

```
<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01//EN" "http://www.w3.org/TR/html4/strict.dtd">
<html>
<head>
<meta http-equiv="Content-Type" content="text/html; charset=utf-8">
<title>Example</title>
<!-- - -->
<script src="jquery.min.js" type="text/javascript"></script>
<script src="highcharts.js" type="text/javascript"></script>
<!-- - -->
<script type="text/javascript">
var dat1 = [29.9, 71.5, 106.4, 129.2, 144.0, 176.0, 135.6, 148.5, 216.4, 194.1, 95.6, 54.4];
</script>
<!-- - -->
<script type="text/javascript">
var chart1;
$(document).ready(function(){
  chart1 = new Highcharts.Chart({
    chart: {renderTo: 'container1'},
    series: [{data: dat1}]
  });
});
</script>
<!-- - -->
</head>
<body>
<div id="container1" style="width: 700px; height: 400px "></div>
</body>
</html>
```

This test case and all the libraries can be copied from the attachments at the end of this article. The Test\_01.htm example file and the files of the libraries are located in the same \\Test folder, therefore, we simply double-click on the HTML-file Test\_01.htm to see the results of our work.

It must be kept in mind that for a normal display of this test page, the execution of JavaScript should be allowed in WEB-browser. Since the browsers, for security purposes, allows you to disable this option, it may happen that it is turned off. As a result, we should see the following:

![Test_01.htm](https://c.mql5.com/2/2/Fig01.png)

Figure 1. Test\_01.htm

This is our first test chart, and despite the apparent complexity of this technology, its creation did not take long.

We should note some features of the displayed charts, created in this way. In the copied catalog, open the file Test\_01.htm, and if the WEB-browser allows you to zoom in to the viewed pages, you'll notice that even with a substantial enlargement, **the quality of the chart is not worsen**.

This is due to the fact that this chart is not a static image, such as PNG or JPEG-files, and is re-sketched after a zooming in or out of the area allotted for its drawing. Therefore, such an image can not be saved to a disk, the way we usually save a picture we liked. Since the chart was constructed by the means of JavaScript, we must not fail to mention the fact that different browsers, having their own built-in interpreters of this language, may not always executу it in the same way.

The charts created using JavaScript, may sometimes look differ when using different browsers. Most often, these differences, compared with other browsers, occur most often in Internet Explorer.

But we'll hope that the creators of JavaScript-libraries will take care of the maximum possible compatibility of their code with the most popular WEB-browsers.

### MetaTrader 5 and MQL5

In the above example, the data, intended to be displayed on the chart, was set manually during the creation of the HTML-page. To arrange the transfer of data from [MetaTrader 5](https://www.metatrader5.com/en "https://www.metatrader5.com/en") into the created chart, we will use the simplest method. Let [MetaTrader 5](https://www.metatrader5.com/en "https://www.metatrader5.com/en") record data to a separate file from which it will be loaded into the browser, when displaying the chart. Let's write an example that includes an HTML-page, which will display the chart, by downloading data from a file and script on MQL5, which will create this file.

As the HTML-file, we will use the previously created file Test\_01.htm, after making some small changes to it. We called the modified file as example1.htm. All of the made changes will be reduced to the fact that lines:

```
<script type="text/javascript">
var dat1 = [29.9, 71.5, 106.4, 129.2, 144.0, 176.0, 135.6, 148.5, 216.4, 194.1, 95.6, 54.4];
</script>
```

will be replaced by

```
<script type="text/javascript">
var dat1=[0];
</script>
<script src="exdat.txt" type="text/javascript"></script>
```

Now the browser, when downloading the HTML-page, will need to also load the exdat.txt text file, in which the values, intended to be displayed on the chart will be assigned to the dat1 array. This file should contain a fragment of JavaScript-code. This file can be easily created in MetaTrader 5, using the corresponding script.

An example of such script is provided below.

```
//+------------------------------------------------------------------+
//|                                                     Example1.mq5 |
//|                        Copyright 2010, MetaQuotes Software Corp. |
//|                                              https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2010, MetaQuotes Software Corp."
#property link      "https://www.mql5.com"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
  int i,n,fhandle;
  double gr[25];
  string str;

  n=ArraySize(gr);
  for(i=0;i<n;i++)
    {
    gr[i]=NormalizeDouble(MathSin(i*3*2*M_PI/n),4);
    }

  str=DoubleToString(gr[0],4);
  for(i=1;i<n;i++)
    {
    str+=","+DoubleToString(gr[i],4);
    }

  ResetLastError();
  fhandle=FileOpen("exdat.txt",FILE_WRITE|FILE_TXT|FILE_ANSI);
  if(fhandle<0){Print("File open failed, error ",GetLastError());return;}

  FileWriteString(fhandle,"dat1=["+str+"];\n");

  FileClose(fhandle);
  }
//+------------------------------------------------------------------+
```

To store the displayed data this script uses the gr\[\] array, which holds 25 items. This array, as an example, is filled with values of sinus function, with rounding being done up to four decimal places. This array, of course, can be filled with any other, more useful data.

Further, this data is formatted and combined into a single text string. In order to reduce the volume of the generated text file, the values of the gr\[\] array elements with only four decimal points are placed into the string. For this purpose we used the [DoubleToString()](https://www.mql5.com/en/docs/convert/doubletostring) function.

After the text string str is formed, it is stored in the exdat.txt file. In case of a successful execution of the script, the texdat.txt text file will be created in the \\MQL5\\Files subfolder of the client terminal; if the file already exists, it will be overwritten.

The jquery.min.js, highcharts.js, Example1.mq5, Example1.htm and exdat.txt files are presented at the end of this article in the attachments section. These five files are located in the catalog \\Example1. In order to simply view the results, just copy this example and in the catalog \\Example1 open the file Example1.htm. The chart will be built according to the data from the file exdat.txt.

![Example1.htm](https://c.mql5.com/2/2/Fig02.png)

Figure 2. Example1.htm

Of course, to run the Example1.mq5 script it should be located in the \\MQL5\\Scripts folder of the client terminal and be compiled.

As mentioned earlier, after the launch of the script, the exdat.txt file will be created in the \\MQL5\\Files folder, but in our example, the HTML-file, the files from the libraries, and the data file must be all located in the same folder. Therefore, we have to copy the files jquery.min.js, highcharts.js and Example1.htm into the \\MQL5\\Files folder or copy the exdat.txt file to the folder where these files are located.

In this example, the HTML-page and the libraries are stored in different files. At the design stage, it may be useful that different parts of the project are located in separate files. This helps to avoid, for example, random changes of the code of the libraries when editing the HTML-file. But after the HTML-page is completely edited and no further changes are expected to be made, the libraries can be integrated directly into the HTML-code file.

This is possible because JavaScript-libraries are nothing more than simple text files. If we open the jquery.min.js or the highcharts.js with a text editor, we won't see anything intelligible, because the source code of the libraries was compressed to the maximum capacity.

Compression is performed by removing the service symbols, for example a line feed or a series of spaces. After such compression any formatting is lost, but the text remains as text, since the type of the file does not change. Therefore, it makes no difference whether the browser connects to the library code from an external file with the extension .js, or whether it reads it from the current HTML-file, which in turn is also in a text format.

In order to combine the files, replace in Example1.htm the lines

```
<script src="jquery.min.js" type="text/javascript"></script>
<script src="highcharts.js" type="text/javascript"></script>
```

with

```
<script type="text/javascript">

</script>
```

Next, using a text editor such as Notepad, we open the file of the library jquery.min.js, and by choosing the command "Select all", copy the contents of the file. Next, open the file Example1.htm, paste the copied text of the library between the tags <script type=\\"text/javascript\\"> and </script>. Save the obtained file as Example2.htm. In the same manner, copy the contents of the library highcharts.js into this file, placing it between the text of the previously copied library and the tag </script>.

As the result of copying, the HTML-file increases in size, however, now we do not need separate files of the libraries for its correct display. It is sufficent to have the exdat.txt data file in the Folder\\Example2, which includes the files Example2.htm and exdat.txt is located at the end of this article in the attachments section.

### A report on the trading history in a graphical form

For a more complete demonstration of the proposed method of displaying graphical information, we will create a report that shows the history of the trading account at a specified time interval. The HTML-based report, which is created in the MetaTrader 5 when you select the "Report" command in the context menu of the "History" tab, will serve as the prototype. This report includes a large number of different characteristics, summarized in one table. Assuming that these characteristics will be more visual when presented in the form of charts and diagrams, let's display them using the highcharts.js graphical library.

In the examples above, for the construction of the chart, we used the default display parameters, set in this version of the highcharts.js library.

For practical purposes, this option will not succeed, since in each case we will have to adjust the view of the chart to fit individual specific requirements. For this purpose, the highcharts.js library provides a wide range of opportunities, having a large number of options that can be applied to the chart or diagram. As already mentioned, the list of options, along with their detailed descriptions and examples, can be found at [http://www.highcharts.com](https://www.mql5.com/go?link=https://www.highcharts.com/ "http://www.highcharts.com/").

We won't dwell on the description of options of the graphics library and on the specifics of its use because this article is intended only to suggest and demonstrate the ability to use a
WEB-browser for displaying information received from MetaTrader 5. Especially since depending on the specific requirements for the creation of a WEB-page, some other JavaScript-library may be used. The reader can independently select the most suitable library, and study it as in-depth as the practice of its use requires.

To display the history of the trading account we created the ProfitReport.htm file. It can be found in the attachments.  The \\Report folder contains the data.txt with the data to display. The data.txt file is placed in the folder as an example.

When we copy the \\Report folder and open the ProfitReport.htm, we see the trading characteristics of the test account, created for this example, in graphical form.

![ProfitReport.htm](https://c.mql5.com/2/2/Fig03.png)

Figure 3. ProfitReport.htm

When creating the ProfitReport.htm, we first made a rough page layout, and determined approximately where, and what type of information will be located.

Then we placed the charts, with their default options, on the page.

After creating this template, we chose the most fitting options for each individual chart. After completing the editing, we simply copied the texts of the libraries into the page. As already mentioned, for a correct page display, it should be located in the same catalog as the file data.txt, which contains the data, intended for its display.

The data.txt file was created in MetaTrader 5, using the ProfitReport.mq5 script. In the case of a successful execution of this script, the data.txt file is created in the \\MQL5\\Files folder, containing the trade characteristics of the currently active account.

We must not forget that the script should be placed in the \\MQL5\\Scripts folder and compiled.

```
//-----------------------------------------------------------------------------------
//                                                                   ProfitReport.mq5
//                                          Copyright 2011, MetaQuotes Software Corp.
//                                                                https://www.mql5.com
//-----------------------------------------------------------------------------------
#property copyright   "Copyright 2011, MetaQuotes Software Corp."
#property link        "https://www.mql5.com"
#property version     "1.00"
#property script_show_inputs

#include <Arrays\ArrayLong.mqh>
#include <Arrays\ArrayDouble.mqh>
#include <Arrays\ArrayString.mqh>
#include <Arrays\ArrayInt.mqh>

//--- input parameters
input int nD=30;               // Number of days
//--- global
double   balabce_cur=0;        // balance
double   initbalance_cur=0;    // Initial balance (not including deposits to the account)
int      days_num;             // number of days in the report (including the current day)
datetime tfrom_tim;            // Date from
datetime tend_tim;             // Date to
double   netprofit_cur=0;      // Total Net Profit
double   grossprofit_cur=0;    // Gross Profit
double   grossloss_cur=0;      // Gross Loss
int      totaltrades_num=0;    // Total Trades
int      longtrades_num=0;     // Number of Long Trades
double   longtrades_perc=0;    // % of Long Trades
int      shorttrades_num=0;    // Number of Short Trades
double   shorttrades_perc=0;   // % of Short Trades
int      proftrad_num=0;       // Number of All Profit Trades
double   proftrad_perc=0;      // % of All Profit Trades
int      losstrad_num=0;       // Number of All Loss Trades
double   losstrad_perc=0;      // % of All Loss Trades
int      shortprof_num=0;      // Number of Short Profit Trades
double   shortprof_perc=0;     // % of Short Profit Trades
double   shortloss_perc=0;     // % of Short Loss Trades
int      longprof_num=0;       // Number of Long Profit Trades
double   longprof_perc=0;      // % of Long Profit Trades
double   longloss_perc=0;      // % of Long Loss Trades
int      maxconswins_num=0;    // Number of Maximum consecutive wins
double   maxconswins_cur=0;    // Maximum consecutive wins ($)
int      maxconsloss_num=0;    // Number of Maximum consecutive losses
double   maxconsloss_cur=0;    // Maximum consecutive losses ($)
int      aveconswins_num=0;    // Number of Average consecutive wins
double   aveconswins_cur=0;    // Average consecutive wins ($)
int      aveconsloss_num=0;    // Number of Average consecutive losses
double   aveconsloss_cur=0;    // Average consecutive losses ($)
double   largproftrad_cur=0;   // Largest profit trade
double   averproftrad_cur=0;   // Average profit trade
double   larglosstrad_cur=0;   // Largest loss trade
double   averlosstrad_cur=0;   // Average loss trade
double   profitfactor=0;       // Profit Factor
double   expectpayoff=0;       // Expected Payoff
double   recovfactor=0;        // Recovery Factor
double   sharperatio=0;        // Sharpe Ratio
double   ddownabs_cur=0;       // Balance Drawdown Absolute
double   ddownmax_cur=0;       // Balance Drawdown Maximal
double   ddownmax_perc=0;      // % of Balance Drawdown Maximal
int      symbols_num=0;        // Numbre of Symbols

string       Band="";
double       Probab[33],Normal[33];
CArrayLong   TimTrad;
CArrayDouble ValTrad;
CArrayString SymNam;
CArrayInt    nSymb;

//-----------------------------------------------------------------------------------
// Script program start function
//-----------------------------------------------------------------------------------
void OnStart()
  {
  int         i,n,m,k,nwins=0,nloss=0,naverw=0,naverl=0,nw=0,nl=0;
  double      bal,sum,val,p,stdev,vwins=0,vloss=0,averwin=0,averlos=0,pmax=0;
  MqlDateTime dt;
  datetime    ttmp,it;
  string      symb,br;
  ulong       ticket;
  long        dtype,entry;

  if(!TerminalInfoInteger(TERMINAL_CONNECTED)){printf("Terminal not connected.");return;}
  days_num=nD;
  if(days_num<1)days_num=1;             // number of days in the report (including the current day)
  tend_tim=TimeCurrent();                                                // date to
  tfrom_tim=tend_tim-(days_num-1)*86400;
  TimeToStruct(tfrom_tim,dt);
  dt.sec=0; dt.min=0; dt.hour=0;
  tfrom_tim=StructToTime(dt);                                            // date from
//---------------------------------------- Bands
  ttmp=tfrom_tim;
  br="";
  if(dt.day_of_week==6||dt.day_of_week==0)
    {
    Band+=(string)(ulong)(ttmp*1000)+",";
    br=",";ttmp+=86400;
    }
  for(it=ttmp;it<tend_tim;it+=86400)
    {
    TimeToStruct(it,dt);
    if(dt.day_of_week==6){Band+=br+(string)(ulong)(it*1000)+","; br=",";}
    if(dt.day_of_week==1&&br==",") Band+=(string)(ulong)(it*1000);
    }
  if(dt.day_of_week==6||dt.day_of_week==0) Band+=(string)(ulong)(tend_tim*1000);

//----------------------------------------
  balabce_cur=AccountInfoDouble(ACCOUNT_BALANCE);                          // Balance

  if(!HistorySelect(tfrom_tim,tend_tim)){Print("HistorySelect failed");return;}
  n=HistoryDealsTotal();                                           // Number of Deals
  for(i=0;i<n;i++)
    {
    ticket=HistoryDealGetTicket(i);
    entry=HistoryDealGetInteger(ticket,DEAL_ENTRY);
    if(ticket>=0&&(entry==DEAL_ENTRY_OUT||entry==DEAL_ENTRY_INOUT))
      {
      dtype=HistoryDealGetInteger(ticket,DEAL_TYPE);
      if(dtype==DEAL_TYPE_BUY||dtype==DEAL_TYPE_SELL)
        {
        totaltrades_num++;                                          // Total Trades
        val=HistoryDealGetDouble(ticket,DEAL_PROFIT);
        val+=HistoryDealGetDouble(ticket,DEAL_COMMISSION);
        val+=HistoryDealGetDouble(ticket,DEAL_SWAP);
        netprofit_cur+=val;                                         // Total Net Profit
        if(-netprofit_cur>ddownabs_cur)ddownabs_cur=-netprofit_cur; // Balance Drawdown Absolute
        if(netprofit_cur>pmax)pmax=netprofit_cur;
        p=pmax-netprofit_cur;
        if(p>ddownmax_cur)
          {
          ddownmax_cur=p;                                 // Balance Drawdown Maximal
          ddownmax_perc=pmax;
          }
        if(val>=0)              //win
          {
          grossprofit_cur+=val;                            // Gross Profit
          proftrad_num++;                                  // Number of Profit Trades
          if(val>largproftrad_cur)largproftrad_cur=val;    // Largest profit trade
          nwins++;vwins+=val;
          if(nwins>=maxconswins_num)
            {
            maxconswins_num=nwins;
            if(vwins>maxconswins_cur)maxconswins_cur=vwins;
            }
          if(vloss>0){averlos+=vloss; nl+=nloss; naverl++;}
          nloss=0;vloss=0;
          }
        else                    //loss
          {
          grossloss_cur-=val;                                   // Gross Loss
          if(-val>larglosstrad_cur)larglosstrad_cur=-val;       // Largest loss trade
          nloss++;vloss-=val;
          if(nloss>=maxconsloss_num)
            {
            maxconsloss_num=nloss;
            if(vloss>maxconsloss_cur)maxconsloss_cur=vloss;
            }
          if(vwins>0){averwin+=vwins; nw+=nwins; naverw++;}
          nwins=0;vwins=0;
          }
        if(dtype==DEAL_TYPE_SELL)
          {
          longtrades_num++;                          // Number of Long Trades
          if(val>=0)longprof_num++;                  // Number of Long Profit Trades
          }
        else if(val>=0)shortprof_num++;               // Number of Short Profit Trades

        symb=HistoryDealGetString(ticket,DEAL_SYMBOL);   // Symbols
        k=1;
        for(m=0;m<SymNam.Total();m++)
          {
          if(SymNam.At(m)==symb)
            {
            k=0;
            nSymb.Update(m,nSymb.At(m)+1);
            }
          }
        if(k==1)
          {
          SymNam.Add(symb);
          nSymb.Add(1);
          }

        ValTrad.Add(val);
        TimTrad.Add(HistoryDealGetInteger(ticket,DEAL_TIME));
        }
      }
    }
  if(vloss>0){averlos+=vloss; nl+=nloss; naverl++;}
  if(vwins>0){averwin+=vwins; nw+=nwins; naverw++;}
  initbalance_cur=balabce_cur-netprofit_cur;
  if(totaltrades_num>0)
    {
    longtrades_perc=NormalizeDouble((double)longtrades_num/totaltrades_num*100,1);     // % of Long Trades
    shorttrades_num=totaltrades_num-longtrades_num;                                 // Number of Short Trades
    shorttrades_perc=100-longtrades_perc;                                           // % of Short Trades
    proftrad_perc=NormalizeDouble((double)proftrad_num/totaltrades_num*100,1);         // % of Profit Trades
    losstrad_num=totaltrades_num-proftrad_num;                                      // Number of Loss Trades
    losstrad_perc=100-proftrad_perc;                                                // % of All Loss Trades
    if(shorttrades_num>0)
      {
      shortprof_perc=NormalizeDouble((double)shortprof_num/shorttrades_num*100,1);     // % of Short Profit Trades
      shortloss_perc=100-shortprof_perc;                                            // % of Short Loss Trades
      }
    if(longtrades_num>0)
      {
      longprof_perc=NormalizeDouble((double)longprof_num/longtrades_num*100,1);        // % of Long Profit Trades
      longloss_perc=100-longprof_perc;                                              // % of Long Loss Trades
      }
    if(grossloss_cur>0)profitfactor=NormalizeDouble(grossprofit_cur/grossloss_cur,2);  // Profit Factor
    if(proftrad_num>0)averproftrad_cur=NormalizeDouble(grossprofit_cur/proftrad_num,2);// Average profit trade
    if(losstrad_num>0)averlosstrad_cur=NormalizeDouble(grossloss_cur/losstrad_num,2);  // Average loss trade
    if(naverw>0)
      {
      aveconswins_num=(int)NormalizeDouble((double)nw/naverw,0);
      aveconswins_cur=NormalizeDouble(averwin/naverw,2);
      }
    if(naverl>0)
      {
      aveconsloss_num=(int)NormalizeDouble((double)nl/naverl,0);
      aveconsloss_cur=NormalizeDouble(averlos/naverl,2);
      }
    p=initbalance_cur+ddownmax_perc;
    if(p!=0)
      {
      ddownmax_perc=NormalizeDouble(ddownmax_cur/p*100,1); // % of Balance Drawdown Maximal
      }
    if(ddownmax_cur>0)recovfactor=NormalizeDouble(netprofit_cur/ddownmax_cur,2); // Recovery Factor

    expectpayoff=netprofit_cur/totaltrades_num;                    // Expected Payoff

    sum=0;
    val=balabce_cur;
    for(m=ValTrad.Total()-1;m>=0;m--)
      {
      bal=val-ValTrad.At(m);
      p=val/bal;
      sum+=p;
      val=bal;
      }
    sum=sum/ValTrad.Total();
    stdev=0;
    val=balabce_cur;
    for(m=ValTrad.Total()-1;m>=0;m--)
      {
      bal=val-ValTrad.At(m);
      p=val/bal-sum;
      stdev+=p*p;
      val=bal;
      }
    stdev=MathSqrt(stdev/ValTrad.Total());
    if(stdev>0)sharperatio=NormalizeDouble((sum-1)/stdev,2);    // Sharpe Ratio

    stdev=0;
    for(m=0;m<ValTrad.Total();m++)
      {
      p=ValTrad.At(m)-expectpayoff;
      stdev+=p*p;
      }
    stdev=MathSqrt(stdev/ValTrad.Total());                      // Standard deviation
    if(stdev>0)
      {
      ArrayInitialize(Probab,0.0);
      for(m=0;m<ValTrad.Total();m++)                           // Histogram
        {
        i=16+(int)NormalizeDouble((ValTrad.At(m)-expectpayoff)/stdev,0);
        if(i>=0 && i<ArraySize(Probab))Probab[i]++;
        }
      for(m=0;m<ArraySize(Probab);m++) Probab[m]=NormalizeDouble(Probab[m]/totaltrades_num,5);
      }
    expectpayoff=NormalizeDouble(expectpayoff,2);                  // Expected Payoff
    k=0;
    symbols_num=SymNam.Total();                                  // Symbols
    for(m=0;m<(6-symbols_num);m++)
      {
      if(k==0)
        {
        k=1;
        SymNam.Insert("",0);
        nSymb.Insert(0,0);
        }
      else
        {
        k=1;
        SymNam.Add("");
        nSymb.Add(0);
        }
      }
    }
  p=1.0/MathSqrt(2*M_PI)/4.0;
  for(m=0;m<ArraySize(Normal);m++)                             // Normal distribution
    {
    val=(double)m/4.0-4;
    Normal[m]=NormalizeDouble(p*MathExp(-val*val/2),5);
    }

  filesave();
  }
//-----------------------------------------------------------------------------------
// Save file
//-----------------------------------------------------------------------------------
void filesave()
  {
  int n,fhandle;
  string loginame,str="",br="";
  double sum;

  ResetLastError();
  fhandle=FileOpen("data.txt",FILE_WRITE|FILE_TXT|FILE_ANSI);
  if(fhandle<0){Print("File open failed, error ",GetLastError());return;}

  loginame="\""+(string)AccountInfoInteger(ACCOUNT_LOGIN)+", "+
                        TerminalInfoString(TERMINAL_COMPANY)+"\"";
  str+="var PName="+loginame+";\n";
  str+="var Currency=\""+AccountInfoString(ACCOUNT_CURRENCY)+"\";\n";
  str+="var Balance="+(string)balabce_cur+";\n";
  str+="var IniBalance="+(string)initbalance_cur+";\n";
  str+="var nDays="+(string)days_num+";\n";
  str+="var T1="+(string)(ulong)(tfrom_tim*1000)+";\n";
  str+="var T2="+(string)(ulong)(tend_tim*1000)+";\n";
  str+="var NetProf="+DoubleToString(netprofit_cur,2)+";\n";
  str+="var GrossProf="+DoubleToString(grossprofit_cur,2)+";\n";
  str+="var GrossLoss="+DoubleToString(grossloss_cur,2)+";\n";
  str+="var TotalTrad="+(string)totaltrades_num+";\n";
  str+="var NProfTrad="+(string)proftrad_num+";\n";
  str+="var ProfTrad="+DoubleToString(proftrad_perc,1)+";\n";
  str+="var NLossTrad="+(string)losstrad_num+";\n";
  str+="var LossTrad="+DoubleToString(losstrad_perc,1)+";\n";
  str+="var NLongTrad="+(string)longtrades_num+";\n";
  str+="var LongTrad="+DoubleToString(longtrades_perc,1)+";\n";
  str+="var NShortTrad="+(string)shorttrades_num+";\n";
  str+="var ShortTrad="+DoubleToString(shorttrades_perc,1)+";\n";
  str+="var ProfLong ="+DoubleToString(longprof_perc,1)+";\n";
  str+="var LossLong ="+DoubleToString(longloss_perc,1)+";\n";
  FileWriteString(fhandle,str); str="";
  str+="var ProfShort="+DoubleToString(shortprof_perc,1)+";\n";
  str+="var LossShort="+DoubleToString(shortloss_perc,1)+";\n";
  str+="var ProfFact="+DoubleToString(profitfactor,2)+";\n";
  str+="var LargProfTrad="+DoubleToString(largproftrad_cur,2)+";\n";
  str+="var AverProfTrad="+DoubleToString(averproftrad_cur,2)+";\n";
  str+="var LargLosTrad="+DoubleToString(larglosstrad_cur,2)+";\n";
  str+="var AverLosTrad="+DoubleToString(averlosstrad_cur,2)+";\n";
  str+="var NMaxConsWin="+(string)maxconswins_num+";\n";
  str+="var MaxConsWin="+DoubleToString(maxconswins_cur,2)+";\n";
  str+="var NMaxConsLos="+(string)maxconsloss_num+";\n";
  str+="var MaxConsLos="+DoubleToString(maxconsloss_cur,2)+";\n";
  str+="var NAveConsWin="+(string)aveconswins_num+";\n";
  str+="var AveConsWin="+DoubleToString(aveconswins_cur,2)+";\n";
  str+="var NAveConsLos="+(string)aveconsloss_num+";\n";
  str+="var AveConsLos="+DoubleToString(aveconsloss_cur,2)+";\n";
  str+="var ExpPayoff="+DoubleToString(expectpayoff,2)+";\n";
  str+="var AbsDD="+DoubleToString(ddownabs_cur,2)+";\n";
  str+="var MaxDD="+DoubleToString(ddownmax_cur,2)+";\n";
  str+="var RelDD="+DoubleToString(ddownmax_perc,1)+";\n";
  str+="var RecFact="+DoubleToString(recovfactor,2)+";\n";
  str+="var Sharpe="+DoubleToString(sharperatio,2)+";\n";
  str+="var nSymbols="+(string)symbols_num+";\n";
  FileWriteString(fhandle,str);

  str="";br="";
  for(n=0;n<ArraySize(Normal);n++)
    {
    str+=br+"["+DoubleToString(((double)n-16)/4.0,2)+","+DoubleToString(Normal[n],5)+"]";
    br=",";
    }
  FileWriteString(fhandle,"var Normal=["+str+"];\n");

  str="";
  str="[-4.25,0]";
  for(n=0;n<ArraySize(Probab);n++)
    {
    if(Probab[n]>0)
      {
      str+=",["+DoubleToString(((double)n-16)/4.0,2)+","+DoubleToString(Probab[n],5)+"]";
      }
    }
  str+=",[4.25,0]";
  FileWriteString(fhandle,"var Probab=["+str+"];\n");

  str=""; sum=0;
  if(ValTrad.Total()>0)
    {
    sum+=ValTrad.At(0);
    str+="["+(string)(ulong)(TimTrad.At(0)*1000)+","+DoubleToString(sum,2)+"]";
    for(n=1;n<ValTrad.Total();n++)
      {
      sum+=ValTrad.At(n);
      str+=",["+(string)(ulong)(TimTrad.At(n)*1000)+","+DoubleToString(sum,2)+"]";
      }
    }
  FileWriteString(fhandle,"var Prof=["+str+"];\n");
  FileWriteString(fhandle,"var Band=["+Band+"];\n");

  str="";br="";
  for(n=0;n<SymNam.Total();n++)
    {
    str+=br+"{name:\'"+SymNam.At(n)+"\',data:["+(string)nSymb.At(n)+"]}";
    br=",";
    }
  FileWriteString(fhandle,"var Sym=["+str+"];\n");

  FileClose(fhandle);
  }
```

As we can see, the script code is rather cumbersome, but this is not due of the complexity of the task, rather because of the large number of trading characteristics, the value of which need to be determined. For the storage of these values, the beginning of the script declares the [global variables](https://www.mql5.com/en/docs/basis/variables/global), provided with relevant commentaries.

The [OnStart()](https://www.mql5.com/en/docs/basis/function/events#onstart) function verifies whether the terminal is connected to the trading server, and if not, the script finish its work. In the absence of connection to the server, we won't be able to define an active account and obtain information about it.

The next step is the calculation of the date, from which the trading data for the current active account will be included in the report. As the end date, we use the value of the current date and the current time at the time of the execution of the script. The number of days, included in the report, can be set when loading the script by changing the input parameter "Number of days", which is by default equal to 30 days. Once we have defined the beginning and ending time of the report, in the string variable Band, a pair of time values, corresponding to the beginning and end of the weekend, is formed. This information is used so that on the balance chart the time intervals, corresponding to Saturday and Sunday, could be marked yellow.

Next, using the [HistorySelect()](https://www.mql5.com/en/docs/trading/historyselect) function, the history of deals and orders for a specified interval becomes available, and by calling the [HistoryDealsTotal()](https://www.mql5.com/en/docs/trading/historydealstotal) function, we determine the number of deals in the history. After this, based on the number of deals, a cycle is arranged, which gathers the statistics, necessary for the calculation of trading characteristics, and at the end of the cycle their values are determined.

When we created the script, our task was to preserve the meaning of the trading characteristics in accordance with the report generated in MetaTrader 5. It is assumed that the characteristics, calculated by the script, must correspond to the description, which is given in the Help file of the terminal.

The information on access to the account history and the calculations of trading characteristics can be found in the following articles:

- [Orders, Positions, and Deals in MetaTrader 5;](https://www.mql5.com/en/articles/211)
- [Creating an Information Board using the Standard Library Classes and Google Chart API;](https://www.mql5.com/en/articles/102)
- [What the Numbers in the Expert Testing Report Mean;](https://www.mql5.com/en/articles/1486)

- [Mathematics in Trading: How to Estimate Trade Results](https://www.mql5.com/en/articles/1492).


The majority of characteristics are calculated quite easily, so in this article we won't consider the operations associated with the calculation of each characteristic, and will further consider only the existing differences from the standard report and its supplements.

In the report, which is generated by the terminal, the balance chart is constructed by a sequential display of values for each time it changes, and the X scale reflects the number of such changes. In our case, for the construction of the chart we use a time scale.

Therefore, the profit chart is very different from the chart, generated by the terminal. We chose this option of a chart construction in order to display the time of positions closings on a real time scale. Therefore, we can see when the trading activity increased or decreased over the reporting period.

When constructing a chart, it must be kept in mind that MQL5 operates with the date value [presented as the number of seconds elapsed since January 1, 1970](https://www.mql5.com/en/docs/basis/types/integer/datetime), while the graphics library requires this value as the number of milliseconds since January 1, 1970. Therefore the received date values in the script must be multiplied by a thousand in order to be displayed correctly.

To store the value of the profit and the time of closing the deal, the script uses the [CArrayDouble](https://www.mql5.com/en/docs/standardlibrary/datastructures/carraydouble) and [CArrayLong](https://www.mql5.com/en/docs/standardlibrary/datastructures/carraylong) classes from the [Standard Library](https://www.mql5.com/en/docs/standardlibrary). Every time a resultant deal is detected in the loop, information about it is placed, using the [Add()](https://www.mql5.com/en/docs/standardlibrary/datastructures/carraydouble/carraydoubleadd) method, into the element, which is added to the end of the array. This allows us to bypass the need to determine in advance the required number of elements. The size of the array simply increases with the number of found deals in the deals history.

For each deal a check is performed on which symbol it has been executed, while retaining the name of the symbol and the number of deals performed on it. Just like for the profit chart, this data, when viewing the history, accumulates by recording it into an element that is added to the end of the array. To store the name of the symbol and the number of deals, we use the [CArrayString](https://www.mql5.com/en/docs/standardlibrary/datastructures/carraystring) and [CArrayInt](https://www.mql5.com/en/docs/standardlibrary/datastructures/carrayint) classes of the standard library.

The single column on the chart will be too wide for the case if the deals were executed on the one symbol. In order to avoid this, the data array always contains at least 7 elements. Unused elements are not displayed on the diagram, since they have zero values, and thus do not allow the column to become too wide. To make sure that when there is a small number of symbols, the columns are placed approximately in the middle of the X axis, the insignificant elements of the array are sequentially inserted in the beginning or at the end of the array.

The next difference from the standard report is the attempt to construct the chart of [probability distribution](https://en.wikipedia.org/wiki/Probability_distribution "https://en.wikipedia.org/wiki/Probability_distribution") for the sequence of profit values for each deal.

![Probability density](https://c.mql5.com/2/2/Fig04.png)

Figure 4. Probability density

Most often this type of chart is presented in the form of a histograms. In our case, the probability chart is created by constructing a spline, based on the existing column values of such histogram. The calculated values of the probability density are complemented by on the left and on the right, outside of the chart, with zero values. This is necessary so that the constructed by the spline chart, is not interrupted at the last known value, and continued beyond the chart, declining to zero.

For comparison, on the probability density chart, gray color is used to highlight the chart of the [normal distribution](http://en.wikipedia.org/wiki/Normal_distribution "http://en.wikipedia.org/wiki/Normal_distribution"), normalized in such a way that the sum of its readings is equal to one, just like the chart, which was built on the values of the histogram. In the provided example report, the number of deals is not enough give a more or less reliable estimate of the probability distribution of the values of profit trades. We can assume that when there is a large number of deals in the history, this chart will look more authentic.

Once all of the trading characteristics are calculated, the filesave() function is called at the end of the script. This function opens the data.txt file, into which the names and values of the variables are recorded in a text format. The values of these variables correspond to the calculated parameters, and their names correspond to the names, which are used in the HTML-file during the transfer of parameters to the functions of the graphics library.

In order to reduce the number of disk accesses during the writing of the file, short lines are merged into one longer line, and only then it is recorded into the file. The data.txt file, as is customary in MetaTrader 5, is created in the catalog MQL5\\Files; if this file already exists, it is overwritten. For convenience, you can copy the ProfitReport.htm file into this catalog, and run it from there.

In the MetaTrader 5 terminal, when saving a report in the HTML format, it is automatically opened by a browser, which is registered as the default browser. This possibility was not implemented in the example provided in this article.

In order to add an autoplay, insert the following lines to the beginning of the ProfitReport.mq5 script

```
#import "shell32.dll"
int ShellExecuteW(int hwnd,string lpOperation,string lpFile,string lpParameters,
                  string lpDirectory,int nShowCmd);
#import
```

and in the end, after a call to the filesave() function, add

```
string path=TerminalInfoString(TERMINAL_DATA_PATH)+"\\MQL5\\Files\\ProfitReport.htm";
ShellExecuteW(NULL,"open",path,NULL,NULL,1);
```

If the file ProfitReport.htm exists in the specified in the variable path, then when the function ShellExecuteW() is called, it will be opened by a browser. The ShellExecuteW() function is located in the shell32.dll system library, the declaring of this function is added at the beginning of the file to provide access to it.

### Conclusion

The use of WEB-browsers allows us to show a lot of different information at the same time which may be helpful in, for example, the organization of visual control over
the internal state of individual modules of the
Expert Advisor, running in the client terminal.

The capital management data, the trading signals, the trailing stop and other modules data can be conveniently and simultaneously displayed. The multi-page-HTML-reports can be used if it's neccessary to display too much information.

It should be noted that the capabilities of the JavaScript language are much wider than just drawing charts. Using this language, we can make a truly interactive WEB-pages. In the Internet you can find a large number of ready JavaScript-Codes, included in the WEB-page, and various examples of the use of this language.

For example, the terminal can be managed directly from the browser window if two-way data exchange between the terminal and the browser is organized.

We hope that the method, described in this article, will be useful.

### Files

JS\_Lib.zip            - highcharts.js and jquery.min.js

librariesTest.zip   - highcharts.js, jquery.min.js and Test\_01.htm

Example1.zip       - highcharts.js, jquery.min.js, Example1.htm, Example1.mq5 and exdat.txt

Example2.zip       - Example2.htm and exdat.txt

Report.zip           - ProfitReport.htm and data.txt

ProfitReport.mq5 - Script for the collection of statistics and the creation of the data.txt file

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/244](https://www.mql5.com/ru/articles/244)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/244.zip "Download all attachments in the single ZIP archive")

[js\_lib.zip](https://www.mql5.com/en/articles/download/244/js_lib.zip "Download js_lib.zip")(53.99 KB)

[test.zip](https://www.mql5.com/en/articles/download/244/test.zip "Download test.zip")(54.53 KB)

[example1.zip](https://www.mql5.com/en/articles/download/244/example1.zip "Download example1.zip")(55.31 KB)

[example2.zip](https://www.mql5.com/en/articles/download/244/example2.zip "Download example2.zip")(54.3 KB)

[report.zip](https://www.mql5.com/en/articles/download/244/report.zip "Download report.zip")(57.25 KB)

[profitreport.mq5](https://www.mql5.com/en/articles/download/244/profitreport.mq5 "Download profitreport.mq5")(17.12 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Introduction to the Empirical Mode Decomposition Method](https://www.mql5.com/en/articles/439)
- [Kernel Density Estimation of the Unknown Probability Density Function](https://www.mql5.com/en/articles/396)
- [The Box-Cox Transformation](https://www.mql5.com/en/articles/363)
- [Time Series Forecasting Using Exponential Smoothing (continued)](https://www.mql5.com/en/articles/346)
- [Time Series Forecasting Using Exponential Smoothing](https://www.mql5.com/en/articles/318)
- [Analysis of the Main Characteristics of Time Series](https://www.mql5.com/en/articles/292)
- [Statistical Estimations](https://www.mql5.com/en/articles/273)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/3192)**
(16)


![Denis Kirichenko](https://c.mql5.com/avatar/2019/5/5CEDB8D2-7CB7.jpg)

**[Denis Kirichenko](https://www.mql5.com/en/users/denkir)**
\|
31 Jan 2011 at 17:59

Thanks for the "help": as usual, nobody helped, I figured it out myself :-)

Question to the author, is there something similar for 3D charts?

![Гребенев Вячеслав](https://c.mql5.com/avatar/avatar_na2.png)

**[Гребенев Вячеслав](https://www.mql5.com/en/users/virty)**
\|
21 Feb 2011 at 14:29

Yeah. The graphical representation of information in MT5 is a bit weak. And how many interesting things can be drawn!: graphs of correlations of everything and anything, dynamic display of the current rate through different wavelet transformations, 3D and who knows what else. I would like to make some bright screen seiver on semi-chaotic correlations of residuals of series of different rates!

Graphics in the [strategy tester](https://www.metatrader5.com/en/terminal/help/algotrading/testing "Help: Strategy Tester in MetaTrader 5 Client Terminal") are especially bad. Even exporting the tester results to a text file is not done at all.

By the way, if you want to export data to a file anyway, it is better not to invent your own graphics, but to use ready-made packages such as vinnigraphics, sigmaplot, origine or excel.

I was especially pleased with the crown of the article - the probability distribution function. People are trying to make a reasonable study of the course. It is good to see.

![gino](https://c.mql5.com/avatar/avatar_na2.png)

**[gino](https://www.mql5.com/en/users/gino)**
\|
23 Jan 2012 at 23:34

Thank you for this excellent script !

Would it be possible to have this script work on the test results of the [strategy tester](https://www.mql5.com/en/articles/239 "Article \"The Fundamentals of Testing in MetaTrader 5\"") (instead of the deal history on the trade server) ?

This would be a great help in interpreting test results.

Thanks

Gino.

![Иван](https://c.mql5.com/avatar/avatar_na2.png)

**[Иван](https://www.mql5.com/en/users/solandr)**
\|
30 Oct 2012 at 08:45

I can't figure out where I can download the files that are supposedly attached to the article? The file names are there, but there are no links in them:

### Files

JS\_Lib.zip - files of highcharts.js and jquery.min.js libraries

Test.zip - files highcharts.js, jquery.min.js and Test\_01.htm

Example1.zip - files highcharts.js, jquery.min.js, Example1.htm, Example1.mq5 and exdat.txt

Example2.zip - files Example2.htm and exdat.txt

Report.zip - files ProfitReport.htm and data.txt

ProfitReport.mq5 - Script for collecting statistics and creating data.txt file.

![Алексей Ванин](https://c.mql5.com/avatar/2020/9/5F622D31-6397.jpg)

**[Алексей Ванин](https://www.mql5.com/en/users/luk1a)**
\|
16 Sep 2020 at 15:21

In my work as a web programmer I use the FusionCharts library.

I like the simple installation of charts and graphs on a template.

1\. I connect Java Script library FusionCharts.

2\. I create a DIV to display the chart (id="chart\_container").

3\. I insert chart data into the chartData array.

4\. I edit the chartConfig array. This is the design and parameters of the chart.

5\. [I call](https://www.mql5.com/en/docs/basis/function/call "MQL5 Documentation: Function Call") FusionCharts.ready(function(){ var fusioncharts = new FusionCharts(chartConfig);fusioncharts.render(); });

Detailed code and demonstration of the example: http://profi.spage.me/jquery/creation-of-graphs-chart-and-diagrams-on-java-script

![Drawing Channels - Inside and Outside View](https://c.mql5.com/2/0/channels_MQL5.png)[Drawing Channels - Inside and Outside View](https://www.mql5.com/en/articles/200)

I guess it won't be an exaggeration, if I say the channels are the most popular tool for the analysis of market and making trade decisions after the moving averages. Without diving deeply into the mass of trade strategies that use channels and their components, we are going to discuss the mathematical basis and the practical implementation of an indicator, which draws a channel determined by three extremums on the screen of the client terminal.

![Connecting NeuroSolutions Neuronets](https://c.mql5.com/2/0/neural_DLL.png)[Connecting NeuroSolutions Neuronets](https://www.mql5.com/en/articles/236)

In addition to creation of neuronets, the NeuroSolutions software suite allows exporting them as DLLs. This article describes the process of creating a neuronet, generating a DLL and connecting it to an Expert Advisor for trading in MetaTrader 5.

![The Implementation of a Multi-currency Mode in MetaTrader 5](https://c.mql5.com/2/0/Multicurrency_Expert_Advisor.png)[The Implementation of a Multi-currency Mode in MetaTrader 5](https://www.mql5.com/en/articles/234)

For a long time multi-currency analysis and multi-currency trading has been of interest to people. The opportunity to implement a full fledged multi-currency regime became possible only with the public release of MetaTrader 5 and the MQL5 programming language. In this article we propose a way to analyze and process all incoming ticks for several symbols. As an illustration, let's consider a multi-currency RSI indicator of the USDx dollar index.

![Trade Events in MetaTrader 5](https://c.mql5.com/2/0/trade_events.png)[Trade Events in MetaTrader 5](https://www.mql5.com/en/articles/232)

A monitoring of the current state of a trade account implies controlling open positions and orders. Before a trade signal becomes a deal, it should be sent from the client terminal as a request to the trade server, where it will be placed in the order queue awaiting to be processed. Accepting of a request by the trade server, deleting it as it expires or conducting a deal on its basis - all those actions are followed by trade events; and the trade server informs the terminal about them.

[Best articles and CodeBase updates in MQL5.community channelsFollow us to ensure you never miss out on important updates![](https://www.mql5.com/ff/sh/n9yf51p2srwzfqh5z2/02.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=dgazvhktsxqakdvarucjbvmvzenwlyje&s=98a038fe082e458df8c4a1d8e116e3a6646fd5517f06e48b2356b7ee005817d6&uid=&ref=https://www.mql5.com/en/articles/244&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5083222068658444097)

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