---
title: The Player of Trading Based on Deal History
url: https://www.mql5.com/en/articles/242
categories: Trading, Expert Advisors
relevance_score: 0
scraped_at: 2026-01-24T13:40:05.847047
---

[![](https://www.mql5.com/ff/sh/0wxx5f0vuwq7xh89z2/01.png)VPS for 24/7 tradingContact your broker and find out how to get a free hosting subscriptionLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=nhetzvgituppcfrhndpblbihmzziogdh&s=d00c975c8bda3d8c1b29f042ad33ac81952ccea2f130a8f1ffa9015bab8ade87&uid=&ref=https://www.mql5.com/en/articles/242&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5083027115797910404)

MetaTrader 5 / Trading


### Seeing Once Is Better Than Hearing Twice

Visual analysis of trade history is a significant part of analytical work of a trader. If it wasn't so, there would be no technical analysis that turns the world of digits into the world of pictures. Well, it's clear, since 80% of human perception is done with eyes. Statistics, which generalizes information, cannot tell many nuances. And only visualization with its intuitive perception of the world of digits can cross the t's. They say, seeing once is better than hearing twice.

In this article, we are not going to consider how to write an Expert Advisor intended for automation of visual displaying of trade history. We are going to discuss the questions of passing of information between objects, planning huge applications, managing charts, synchronizing of information of different symbols, etc.

Traditionally, at first I want to tell you about benefits of the player application and associated script, and then we'll move to the code analysis.

### Running Deals in the MetaTrader 5 Strategy Tester

Operation of the player is based on the MetaTrader 5 HTML report. Thus the history of Automated Trading Championship 2010 can be obtained by logging in to a necessary account of ATC-2010 and saving the trade history as a HTML report.

Since the server of the [Automated Trading Championship 2008](https://championship.mql5.com/ "https://championship.mql5.com/") is stopped, we are not able to make it in the same way. The site contains the general report of all contestants packed as a single zip-archive. [Automated\_Trading\_Championship\_2008\_All\_Trades.zip](https://c.mql5.com/2/28/Automated_Trading_Championship_2008_All_Trades__5.zip "https://c.mql5.com/2/28/Automated_Trading_Championship_2008_All_Trades__5.zip")

The "Automated Trading Championship 2008 All Trades.zip" archive should be unpacked to the \\Files folder of MetaTrader 5 installation directory.

To analyze the history of the [Automated Trading Championship 2008](https://championship.mql5.com/ "https://championship.mql5.com/"), you need to run the Report Parser MT4 script that will parse the history, make a selection for the specified login and save it in a binary file. This binary file is read by the Player Report Expert Advisor.

The Player Report EA should be run in the strategy tester with the necessary login specified. Once the testing is over, save a report in the HTML format. The specified login doesn't affect the result of testing, but it will be displayed in the report as the input parameter "login". It allows further discerning of the reports. Since the reports are created by the same Expert Advisor, it's recommended to give them names that differ from the default one.

The Report Parser MT4 script also has the "login" input parameter where you should specify the login of a contestant whose history you want to see. If you don't know the login of a contestant, but you know the nickname, start the script with zero (default) value of the login. In this case, the script won't make a selection by a login; it will just create a csv-file where all the logins are listed in alphabetical order. Name of the file is **"Automated Trading Championship 2008 All Trades\_plus"**. As soon as you find a necessary participant in this file, run the script once again with the login specified.

Thus the tandem of the Report Parser MT4 script and the Player Report EA creates the standard html-report of [MetaTrader 5 Strategy Tester](https://www.metatrader5.com/en/automated-trading/strategy-tester "https://www.metatrader5.com/en/automated-trading/strategy-tester") on the basis of trade history in the MetaTrader 4 format.

The Player Report Expert Advisor does not execute the trades exactly as they were executed in reality, it makes it approximately. The reasons for that are different quotes, rounding of time to minutes in the report and slippages during execution. In most cases, the difference is several points, at that it happens in 10% of trades. But it is sufficient to decrease the profit in the strategy tester from ~170 thousands to ~160 thousands, for example. Everything depends on the volume of deals with slippage.

### Operation of the Player

As I previously mentioned, the player can be used for watching the trade history of the [Automated Trading Championship 2008](https://championship.mql5.com/ "https://championship.mql5.com/") using additional applications, and the [Automated Trading Championship 2010](https://championship.mql5.com/2010/en "https://championship.mql5.com/2010/en") can be watched directly.

In addition, the player supports all MetaTrader 5 reports, thus you can watch the trade history of any Expert Advisor run in the strategy tester or the history of manual trading that is not formatted by the tester, but saved as a report from the "History" tab of the "Toolbox" window.

Parameters of the Player History Trades exp v5 Expert Advisor:

- [name of HTML file of the strategy tester report](https://www.mql5.com/en/articles/242#report_filename);
- [list of required charts](https://www.mql5.com/en/articles/242#charts_list);
- [delete charts when deleting the EA](https://www.mql5.com/en/articles/242#close_charts_for_deleting_EA);
- start and end of history;
- [period of the generator](https://www.mql5.com/en/articles/242#generator_period);
- font of the comments to deals;
- [font size for the comment to deals](https://www.mql5.com/en/articles/242#comment_size);
- [color](https://www.mql5.com/en/articles/242#buy_color) of buy and sell operations;
- [number of speeds](https://www.mql5.com/en/articles/242#speed_number);
- [vertical size of the progress button](https://www.mql5.com/en/articles/242#progress_bar_size).


A report of the MetaTrader 5 Strategy tester is used as input file for the player of deal history. It is the name of a report file that should be specified as the input parameter of the Player History Trades exp v5 EA **"name of html file of the strategy tester report"**. When starting the player, a user is able to specify a period of playing in the input variables "start of history" and "end of history".

If those variables are not set, the player will take them from the trade history starting from the first deal and ending with the last deal. The number of symbols used for trading makes no difference. Only the time of the first and the last deal at the account is considered.

In addition, a user is able to set the names of symbols whose charts should be analyzed. The names should be specified as an enumeration in the **"list of required charts"** variable. Analysis of this variable is not sensitive to case and type of separator. If the variable is not set, all the symbols traded at the account are opened. And sometimes there are a lot of them.

For example, [Manov](https://championship.mql5.com/2010/en/users/Manov "https://championship.mql5.com/2010/en/users/Manov") used 12 currency pairs in his trading. I recommend setting not more than four symbols at a time. Firstly, it is convenient to arrange them; secondly, a lot of charts slow down the speed of playing. Since each symbol is processed in the general loop, increasing the number of symbols leads to slowing down of generation of ticks.

The player will also work even if you specify a symbol that was not used in trading. In this case, the chart will not display any deals, it will be just like the other charts. In addition, it will have the balance indicator attached; however, it will display only the history of the general balance in any of its variants.

I intentionally skipped description of the **"Delete chart when deleting the EA"** parameter. It concerns behavior of the Expert Advisor, not its management. The point is the Expert Advisor analyses a lot of information for its operation. I decided that some information that the EA has will be useful for analysis in the form of files. The Expert Advisor creates csv files that contain trade operations for each symbol and the file with synchronized balances of all symbols, what can be useful for detecting a symbol in a multi-currency basket.

The same variable is also used for deletion of charts automatically opened by the Expert Advisor. Traditionally, the EA should clean its work place at the end of operation. But if a user wants to closely analyze a chart without the EA control, they should start the EA with the "delete charts when deleting the EA" parameter set to 'false'.

The following parameters are not so important.

Period of the generator sets the initial parameter of the tick generator. The "tick" term here is not used in its classic meaning; it means the variation of level. In the Expert Advisor, ticks are generated according to four points of bars. The **"Period of the generator"** parameter sets up the initial state of the generator. Further, you'll be able to change this parameter in the player while it works.

Why don't we generate all the periods starting from M1? Why do we need to change the period of the generator? The matter is the bars of greater timeframes contain a lot of M1 bars, thus we might need to speed up the process of generation. That is why the possibility of changing the period is implemented. Not all timeframes are implemented in the generator, only some of them. The way to change it in the code will be described later.

The **"font of the comments to deals"** parameter may be useful, for example, when comments to deals impede viewing the deals themselves. If you set the size to 1, the inscription will look as a thin line and won't impede viewing. At that you'll be able to see the volume of the deal and position at the "List of objects" tab as you find out the object name from the tooltip.

The trade history is drawn with separate deals, but the line that is drawn depends on the type of position.

Using the **"color of buy operations"** and **"color of sell operations"** you can set the colors you want.

![Displaying the trade history](https://c.mql5.com/2/2/33v826jbkj1_29udrp0_pfa3czrz__1.png)

At the screenshot above, you can see that the level of position often differs from the level of deal.

But the profit is calculated on the basis of the level of position. So I decided to display the position with a trend line and connect the levels of position and deal using a vertical line. The comment near the level of position displays the following information:

```
[deal volume|position volume]
```

If the type of deal doesn't correspond to the type of position (for example, there is a partial closing), then volumes will be displayed with additional signs:

```
[<deal volume>|position volume]
```

At the first place you can see the volume of deal as it is displayed in the trade report; the volume of position is calculated on the basis of the previous state of position and the changes made by the deal.

The **"number of speeds"** parameter regulates the number of steps of decreasing the speed of playback. The player is started at maximum speed. Further, you'll be able to decrease and increase it within the value of the "number of speeds" parameter. Thus the speed button and the period of the generator compose a full range of tools for managing the speed of playback of the trade history.

And the last parameter is the **"vert. size of the progress button"**. I made it for users that prefer big buttons of the progress bar. Generally, my aim was to avoid hiding of the chart behind the controls. That is why the "vert. size of the progress button" parameter is set to 8.

Now let's move to the controls of the player.

![Controls of the player](https://c.mql5.com/2/2/oqfg7pamo7_voda855__1.png)

**The speed** is controlled using the left and right arrows. The mode of controlling depends on the state of the middle (square) button. In the unpressed state it changes the speed, in the pressed state it changes the period of the generator.

The object of controlling the **balance indicator** is displayed as a full oval, but actually it is made of two big buttons that greatly exceed the borders of its visual size. The left button is intended for adding and deleting of the balance indicator from the chart, and the right button controls the data content.

In the "All" state, the information about the total balance of the account is displayed; the "Sum" state is intended for displaying a sampling of balance for the chart symbol the indicator is run on. Controlling of the indicator is asynchronous, what means that the indicator can be run on one chart and not run on another.

![The indicator of balance](https://c.mql5.com/2/2/4t4zq3dqk_ikqlkpu__1.png)

The object of controlling the indicator of balance is the only exception in the synchronization of charts; all the other objects of controls are synchronized. In other words, changes made for a symbol are automatically made to the other symbols.

The **plays/stop** displays what operation will be performed as soon as you press it. During the playback, two small lines are displayed showing that the operation will be paused if you press it. And vice versa, the triangle is displayed in the paused state. Thus if you press it, the player will start its operation.

The **progress line** consists of 100 control buttons made as triggers - if a button is pressed, all the other buttons become unpressed. Since there are 100 buttons, the playback period is divided into 100 parts. If the number of bars is not divisible by 100, the remainder is added to the last part. That is why the settings include the "start of history" and "end of history" parameters. Changing these parameters, you can navigate to the necessary period of history.

By pressing a button, a user changes the date of the internal generator of ticks and navigates the zero bar. If it is not pressed but the internal time of the generator has already moved outside the active button, the player will perform the corresponding switching on its own.

Thus the "progress line" object is both the indicator of the progress and the active control of navigation. The control objects of the player are automatically hidden and expanded to the middle of chart; thus if you need to press a certain button of the progress line, expand the chart to the full screen.

Now let's talk about the behavior of charts that are managed by the player. The player performs synchronization of all charts, but it doesn't mean that any change of scale, chart type, color scheme, shifting of the zero bar, etc. performed at the main chart is repeated at the other charts.

The changes include changing of timeframe. Here we should note that the chart considered as the main one by the player is the one where the controls are displayed, not the one with the blue line of activity. Usually, it is one and the same chart, but it is not always so. To activate a chart, click on it in the chart field.

There is a feature of using the player. If two objects are in the same field, the buttons stop working. That is why sometimes when the bid line crosses the player field, to press a button you'll need to switch to another chart or change the vertical scale of the chart.

Trade Player - YouTube

[Photo image of MQL5.community](https://www.youtube.com/channel/UC8bwZMk2yh5WDToFos49OjQ?embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F242)

MQL5.community

1.91K subscribers

[Trade Player](https://www.youtube.com/watch?v=c33bdUtYPAg)

MQL5.community

Search

Watch later

Share

Copy link

Info

Shopping

Tap to unmute

If playback doesn't begin shortly, try restarting your device.

Share

Include playlist

An error occurred while retrieving sharing information. Please try again later.

0:00

0:00 / 2:04

•Live

•

The video demonstrates the playback of trading of [Manov](https://championship.mql5.com/2010/en/users/Manov "https://championship.mql5.com/2010/en/users/Manov"), one of the participants of [ATC 2010](https://championship.mql5.com/2010/en "https://championship.mql5.com/2010/en"). To make it, I connected to his account in the client terminal using the login= **630165** and password= **MetaTrader** parameters. The trade report was saved under the ReportHistory-630165.html name in the folder **terminal\_data\_folder**\\MQL5\\Files. You can download this file as an archive and unpack it to the specified folder.

### Preparing to Start

1. To make everything work, download player\_history\_trades.zip and unpack it to the folder **terminal\_data\_folder**/MQL5/Indicators.

2. Open the copied folder Player History Trades and [compile](https://www.metatrader5.com/en/metaeditor/help/development/compile "https://www.metatrader5.com/en/metaeditor/help/development/compile") four files in its root directory in MetaEditor. The sequence of compilation of files does not matter.

3. Make sure that the required period of history for all symbols in the trade report is available on the [М1 timeframe](https://www.metatrader5.com/en/terminal/help/charts_analysis/charts#operations "https://www.metatrader5.com/en/terminal/help/charts_analysis/charts#operations"). To do it, manually open the necessary chart with the M1 timeframe, place a vertical line and open the [List of objects](https://www.metatrader5.com/en/terminal/help/charts_advanced/charts_objects_list "https://www.metatrader5.com/en/terminal/help/charts_advanced/charts_objects_list") using the **Ctrl+B** key combination of the context menu. Then change the date of the vertical line to the date of start of trading.

4. Then press the "Show" button. If there are no quotes, there can be two reasons of it. Either they are not downloaded, or the "Max bars in chart" parameter is too small. To see it, go to [Tools->Options->Charts](https://www.metatrader5.com/en/terminal/help/startworking/settings "https://www.metatrader5.com/en/terminal/help/startworking/settings").

Now everything should work.

### Start of the Development

To develop an application, you should have a plan, which further turns into a block diagram as you study it and then it turns into the code. But the project itself starts earlier. The start point of any project is the application properties required by the user. So what properties should the player of trade history have?

01. Being multi-currency.
02. Automatic opening of required charts.
03. Convenient navigation interface and a possibility of scrolling the history in both directions.
04. Synchronous displaying on all charts.
05. Start/Pause of playback.
06. A possibility to choose (and the default mode) the number and symbols of charts that are displayed.
07. A possibility to choose (and the default mode) a period the player will work at.
08. Displaying of history of deals on a chart.
09. Displaying of history of balance and equity.
10. Separate displaying of balance (equity) of a symbol and the total balance (equity) of an account.


The first four items determine the general concept. The other properties determine the direction of implementation of methods.

The general plan of operation of the player:

1. Load a HTML report;
2. Parse it to deals and restore the history of positions;
3. Prepare deals as a queue of orders for opening/closing;
4. At a user command, start displaying the dynamics of deal history with calculation of necessary rates in the form of indicators (charts of equity, drawdowns, etc.);
5. Organize displaying of an information panel on the chart with the other rates.


In addition, a special Expert Advisor is required for trading in the strategy tester according to a MetaTrader 4 report:

1. Parsed deals should be written as a binary data file for the Expert Advisor;
2. Create the report of the MetaTrader 5 Strategy Tester.


This is the general scheme for starting the development, a specification of requirements. If you have it, you can plan writing of the code from top downwards, from the concept to implementation of the functionality.

Not to extend the article, further I'm going to describe only the most significant parts of the code. You shouldn't meet any problems when reading the code since it is well commented.

### Orders and Deals

Currently, there are two trade concepts. The old concept that is used in MetaTrader 4, and the one used for real bidding and used in MetaTrader 5, so called "netting" concept. The detailed description of difference between them is given in the [Orders, Positions, and Deals in MetaTrader 5](https://www.mql5.com/en/articles/211) article.

I'm going to describe only one significant difference. In MetaTrader 4, an order can be represented as a container that stores the information about time of opening, open price and trade volume. And while the door of the container is open, it is in the active trade state. As soon as you close the container, all the information from it is moved in the history.

In MetaTrader 5, positions are used as such containers. But the significant difference is the absence of history of positions. There is only the common history of orders and deals. And though the history contains all the necessary information for restoring the history of positions, you need to spend some time on reorganization of thinking.

You can find out which position a selected order or deal belongs to using the [ORDER\_POSITION\_ID](https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties#enum_order_property_integer) and [DEAL\_POSITION\_ID](https://www.mql5.com/en/docs/constants/tradingconstants/dealproperties#enum_deal_property_integer) identifiers respectively. Thus to convert the history in a format suitable for MetaTrader 5, I will divide the orders of the MetaTrader 4 history into two separate trades - opening and closing trades.

### The HTML Parser

For the ones who are not acquainted with computer slang, I will describe what the word parse means. Parsing means the syntactical (grammatical or lexical) analysis of a text or any sequence of lexemes (symbols, words, bytes, etc.) that checks the correspondence of the input text to a specified grammar and composes a parse tree, according to which you can perform further calculations or transformations.

Two big classes CTable and CHTML are used in the parser. Usage of the CTable class is described in details in the [Electronic Tables in MQL5](https://www.mql5.com/en/articles/228) article, that's why I'm not going to describe it again.

For parsing HTML, I have developed the CHTML class. According to my primary idea, its description should have become an article. But the class is too simple for writing an article, thus I will give a concise description of it.

The general concept of the class can be described by the term 'tag'. A tag can be represented as a function with enclosures. For example, Tag(header,casket), where 'header' is the tag title (tag variables that control the appearance of page are usually specified there), and 'casket' is the content of the tag container. Such tags consist the entire HTML language.

The general structure of the class can be represented as a three-stage call of objects. The instance of the CHTML class creates the objects of all possible tags in its body. The functions of tags are created by a template, and they differ from each other only with their names and settings of two flags.

One flag determines presence of header and the other one determines presence of casket. Those flags allow presenting all the tags with a common structure. Each tag instance creates an instance of the CTegs class in its body. This class contains common methods for all tags, and it performs main operations of searching a necessary tag in a document body.

So, this is how the three-stage call looks like:

```
h.td.base.casket
```

This inscription means that the 'h' object calls the value of the 'casket' variable through the 'td' object (that is an instance of the <td header >casket</td> tag) through the nested object 'base' (that is an object of the CTegs class).

The class also includes methods of searching the tags, they are combined in the public method

```
h.td.Search(text,start);
```

that returns the search point of the end of tag and fills the 'header' and 'casket' variables of the tag.

The other functions prepared in the class are not used, so I won't describe them, there are a lot of other interesting things to tell about.

At the end of description of working with HTML documents, I want to mention that two types of parsers are used in the article; they differ only with the type of saving of information obtained from the file. The first type uses saving of the entire document in a single variable of the 'string' type, it is used in the player. The second type uses line-by-line parsing of the report. It is used in the script for preparing the history of the Championship 2008.

Why do I use two approaches? The matter is for the correct operation of functions of the CTegs class, the whole tag should be placed in the analyzed string. And it is not always possible. For example, in case of tags like table, html, body (they are multi-line). A variable of the string type allows storing (according to my calculations) 32750 symbols without the tabulation symbols. And with '\\r' (after each 32748-th symbol) I managed to store up to 2 000 000 symbols; after reaching this value I stopped my attempts. Most probably, more symbols can be stored.

So why do we use two approaches? The point is for a universal parser of the player you need to find a proper table. The required tables for the tester report and for the report of deal history are located in different places. To keep the versatility (for the parser to understand both reports), I use the scheme of searching the table with the 'td' tag containing "deals".

Structure of the report of the Championship 2008 is known and there is no need to search the necessary table. However, the report document is huge (35 MB), and placing the whole report to a single variable would take a lot of time. This situation stipulates the second approach to parsing.

### The Player

10 requirements to the player are described in the "Starting of the Development" section. Since the multi-currency is at the first place, the Expert Advisor should manage charts. It will be logical if each chart is processed by a separate object that has all the functionality required for the player to work.

Since we work with history, we need a separate example of history for uninterrupted operation instead of hoping that we can get it anytime we want. In addition, repeated getting of the same history would be wasteful comparing to keeping it in the player. In the end, the following scheme comes out:

![The general scheme of the player of trade history](https://c.mql5.com/2/2/vvlfd_p7wec_qjuk0_og3bokn_ofnadn__1.png)

[Object oriented programming](https://www.mql5.com/en/docs/basis/oop) (OOP) allows writing pretty huge applications using block systems. Developed part of the Expert Advisor code can be previously written in a script, debugged and then connected to the Expert Advisor with a minimum adaptation.

Such scheme of development is convenient, because you are sure that the connected code doesn't contain errors (because it works in the script without errors), and any found bugs are the errors of adaptation. There is no such advantage when a code is written bottom-up, when you describe everything in one place as a procedure. And a new bug may appear in any place of the application.

Thus programming from top downward has the advantages in simplicity and speed of writing an application. You may ask "what is simple here?", I would answer with an allegory - it's hard to learn riding a bicycle, but once you learn it, you won't even notice that process. You'll just enjoy a fast riding. Once you learn the OOP syntax, you'll get a great advantage.

To continue the narration, I need to describe three terms of OOP: Association, Aggregation and Composition.

- Association means a connection between objects. Aggregation and Composition are particular cases of the association.

- Aggregation implies that objects are connected with the "part-of" relationship. Aggregation can be multiple, i.e. one object can be aggregated in several classes or objects.

- Composition is a stricter variant of aggregation. In addition to the "part-of" requirement, this "part" cannot belong to different "owners" simultaneously, and it is deleted when the owner is deleted.

Since association includes aggregation and composition, during a detailed analysis, all the cases that cannot be described as aggregation or composition are called association. Generally, all three idioms are called association.

```
class Base
  {
public:
                     Base(void){};
                    ~Base(void){};
   int               a;
  };
//+------------------------------------------------------------------+

class A_Association
  {
public:
                     A_Association(void){};
                    ~A_Association(void){};
   void              Association(Base *a_){};
   // At association, data of the bound object
   // will be available through the object pointer only in the method,
   // where the pointer is passed.
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class A_Aggregation
  {
   Base             *a;
public:
                     A_Aggregation(void){};
                    ~A_Aggregation(void){};
   void              Aggregation(Base *a_){a=a_;};
   // At aggregation, data of the bound object
   // will be available through the object pointer in any method of the class.
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class A_Composition
  {
   Base             *a;
public:
                     A_Composition(void){ a=new Base;};
                    ~A_Composition(void){delete a;};
   // At composition, the object becomes the class member.
  };
```

There is a function in MQL5 for passing a pointer through a parameter:

```
GetPointer(pointer)
```

Its parameter is the object pointer.

For example:

```
void OnStart()
  {
   Base a;
   A_Association b;
   b.Association(GetPointer(a));
  }
```

Functions that are called in [OnInit()](https://www.mql5.com/en/docs/basis/function/events#oninit) of my code often use association. Composition is applied in the CHTML class. And I use aggregation and composition together for binding objects within the CPlayer class. For instance, using aggregation, the objects of the CChartData and SBase classes create a common field of data for all the objects created using composition in the player.

Visually, it can be represented as following:

![Binding of data](https://c.mql5.com/2/2/vx9rf22q1r_cu6t0o__1.png)

Classes, whose objects are compositely created in the CPlayer class, have a template structure with further expanding of functionality. Usage of templates is described in the [Using Pseudo-Templates as Alternative to C++ Templates](https://www.mql5.com/en/articles/253) article, so I'm not going to give its detailed description here.

A template for the class looks as following:

```
//this_is_the_start_point
//+******************************************************************+
class _XXX_
  {
private:
   long              chart_id;
   string            name;
   SBase            *s;
   CChartData       *d;
public:
   bool              state;
                     _XXX_(){state=0;};
                    ~_XXX_(){};
   void              Create(long Chart_id, SBase *base, CChartData *data);
   void              Update();
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void _XXX_::Create(long Chart_id, SBase *base, CChartData *data)
  {
   chart_id=Chart_id;
   s=base; // binding data to the player structure
   d=data; // binding data to the chart structure
   name=" "+ChartSymbol(chart_id);

   if(ObjectFind(chart_id,name)<0)// if there is no object yet
     {//--- try to create the object
      if(ObjectCreate(chart_id,name,OBJ_TREND,0,0,0,0,0))
        {//---
        }
      else
        {//--- failed to create the object, tell about it
         Print("Failed to create the object"+name+". Error code ",GetLastError());
         ResetLastError();
        }
     }
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void _XXX_::Update()
  {
  };
//+******************************************************************+
//this_is_the_end_point
```

So I created the empty classes by the template, connected them, checked whether they process all the requests correctly, and only after it I started filling the composite classes with necessary functionality. This is what is called programming top-down. In case of any failure, you know where to look for its reason.

Now, as the general concept of building is clear, we can proceed with specifics.

First of all, let's take a look at operation of functions, declared in the Player History Trades exp v5 Expert Advisor.

The [OnInit()](https://www.mql5.com/en/docs/basis/function/events#oninit) function as usually prepares information. It creates an object of the CParser\_Tester class that parses a report of the strategy tester, gets the list of all traded financial instruments, processes deals, calculates volumes and levels of positions and then draws the history on the chart. The last item describes the reason why the object is not deleted right after the data is passed. The matter is the information is prepared earlier than the charts are opened. And the graphical objects require a chart ID for drawing. That is why the object of the CParser\_Tester class is deleted later.

Further, as we have the names of symbols used for trading, the Balance\_Process() function is called, it calculates the balances and equities of all symbols passed to it as well as the total balance and equity on the basis of M1 history.

In this part; the application is especially sensitive to a lack of information; that is why I've implemented interruption of running of the EA in case the information for one of the symbols is not downloaded. When the application stops running, it displays an alert with the symbol information for which needs to be downloaded.

The result of working of the Balance\_Process() function are binary files of the history of balance and equity on M1 that are further cut into necessary periods by the balance indicator. Well, I'm running slightly ahead, the operation of the balance indicator will be described later.

The next step of starting the Expert Advisor is selection of symbols. In this place, we analyze the input parameter "list of required charts"; in case the necessary symbol is in the " [Market Watch](https://www.metatrader5.com/en/terminal/help/trading/market_watch "https://www.metatrader5.com/en/terminal/help/trading/market_watch")" list, add it to the array of symbols. In this way, we protect ourselves from "fools", as a user may specify an abracadabra instead of a symbol name or misprint it.

As we have the verified list of symbols requested by a user for opening, we can [open the charts](https://www.mql5.com/en/docs/chart_operations/chartopen). It is done using the following function:

```
ChartOpen(symbol,period)
```

This function opens a chart with the symbol and period passed to it in parameters. In theory, this function returns ID of the opened chart but it doesn't happen always.

As a result of loosing ID, the application malfunctions. To avoid it, I've created two functions:

```
ChartTotal(arrayID);   // get the list of charts before opening additional charts
CurrentChart(arrayID); // get the list of chart for operation
```

One function is executed before opening the charts, and the other one is executed after. The ChartTotal() function obtains the list of charts that were opened before the start of the EA (including the chart the EA is run on) and save their IDs in the input array.

The CurrentChart() function obtains that information, creates a new list considering already opened charts; then according to the difference of lists, it passes the IDs of charts created by the EA to the parametric array. This scheme is reliable, because it works according to the fact of chart opening.

Now as we have IDs of the necessary charts, we can take them under control. To do it, go over all charts in a loop, and using the object of the CParser\_Tester (as you remember, previously I said that we would need it) draw the history of deals and create the objects for managing the chart.

The last action in OnInit() - create the timer and call it for working. All the other actions will be performed in [OnTimer()](https://www.mql5.com/en/docs/basis/function/events#ontimer).

The first problem of creating the player appears on the initial stage of development. It is the problem of creation of a timer. The [EventSetTimer(timer)](https://www.mql5.com/en/docs/eventfunctions/eventsettimer) function allows creating a timer with no less than 1 second frequency. With this variant, ticks would be generated once per second. Even with the limitation of human vision, one second is too long. I need at least 100 milliseconds.

That is why I've implemented a loop inside the timer; it exits several milliseconds before coming of a new Timer event. But this implementation made a lot of technical solutions impossible. For example, the possibility of receiving events disappears, because they constantly wait for the timer to exit from the cycle. And the impossibility of receiving events melts away the possibility of placing the objects of players in indicator and perform parallel calculations of all charts simultaneously. But even with consequent processing of charts the Expert Advisor works pretty fast.

The event of chart activation is replaced with the composite class CClick, whose objects create a signal of changing of the active chart being processed in the cycle of the Click(n) function. The Click() function is a trigger that tracks changes of the chart activation button. If it detects that the button is pressed, it switches all the other objects to the passive state. The button of chart activation is always near the user, but it is not visible, because it has the size of the entire chart, it is colored as the background and it is in the background. When a chart is activated, the button is moved behind the visible borders of the chart, what allows seeing the graphical objects of the player controls, which are hidden by the button of chart activation in the passive mode.

Further, as we have detected the main chart using the Click() function, go to calculation of time motion, call the Progress(Time) function of the active player. This function performs the following calculations: checks if a user performs navigation actions - if not, checks whether it is time to go to the next bar; if it's time then it checks whether the progress should be moved to the next section.

In the end, as we exit from the Progress(Time) function, the cycle has the information about the current time, which is used for further calculations. Then the settings of the active chart are copied to the slave charts. It is done using a loop in the CopyPlayer(n) function. After it, in the Play(Time) function go to execution of all changes that should be made to the chart to make a user think that the time moves, quotes come and trading is performed.

Composite Classes of the Player.

1. CArrayRuler\* -  stores and searches information for quick moving between bars of a current timeframe.
2. CRuler\*         -  stores and searches M1 history information for generation of ticks.
3. CSpeed          -  controls the speed settings and the period of the generator of ticks.
4. CProgress      -  combines all progress buttons into a single object, watches that only one buttons is pressed, changes colors of the buttons.
5. CPlay             -  it is in charge of starting and stopping the player, also it controls the balance indicator.
6. CClick            -  it is in charge of the signals of chart activation.
7. CBackGround  -  the object hides the zero bar from users, as well as it hides future bars when the chart shift from the right border state is enabled.
8. CBarDraw      -  draws the zero bar depending on the scale and type of the chart (bars, candlesticks or line).
9. CHistoryDraw -  makes an illusion for user that the last deal changes in real time.


\\* \- the classes do not include graphical objects.

As I already mentioned, the objects of the CChartData and SBase classes create a common field of data for all the objects inside the player using aggregation. The object of the CChartData class is used for storing and updating of information about the chart as well as managing it. Under managing the chart we mean changing its setting by copying the settings of the main chart. This is how the synchronization of charts is performed. A user just makes an initial signal by changing settings of the active chart, and then several functions of the player make the rest of synchronization operations.

This is how it is done:

The CopyPlayer(n) function, described in the Expert Advisor, calls the CPlayer::Copy(CPlayer \*base) function in a loop associatively passing the pointer to the player of the active chart. Inside CPlayer::Copy(CPlayer \*base) from the player pointer, the pointer of the CChartData object of the active player is associatively passed. Thus the information about the state of the active chart is placed in the object of the CChartData class of the slave chart for copying. After it, information is updated in the CPlayer::Update() function, where all the necessary checks are performed and all the objects are switched to necessary states.

Previously, I promised to tell you how to add periods in the list of available periods of the generator. To do it, open the "Player5\_1.mqh" include file. The static array TFarray\[\] is declared at the beginning of the file. A necessary period should be added to its place in the enumeration that fills the array, and don't forget to change the size of the array and the CountTF variable. After that, compile the Player History Trades exp v5 Expert Advisor.

### The Balance and Drawdown Charts

The balance indicator is managed from the object of the CPlay class. It contains controlling methods and buttons.

The methods of controlling of the indicator are:

```
   Ind_Balance_Create();                 // add the indicator
   IndicatorDelete(int ind_total);     // delete the indicator
   EventIndicators(bool &prev_state);   // send an event to the indicator
   StateIndicators();                  // state of the indicator, state checks
```

The methods of adding/deleting work depending on the state of the name\_on\_balance button. They use the standard MQL5 functions IndicatorCreate() and ChartIndicatorDelete().

The indicator receives an event and performs calculations located in the OnChartEvent() function of the indicator depending on the event code. Events are divided into three types.

They are "update the indicator", "calculate the total balance" and "calculate balance for a symbol". Thus when sending events, depending on the state of the name\_all\_balance button, the user controls the type of calculation. But the indicator code itself doesn't contain any parsing of trade history, calculation of position or recalculation of profit. The indicator doesn't need it.

The balance indicator is intended for displaying of history data, so there's no point in redoing everything again each time you change the type or add/remove the indicator. The indicator reads the binary file of data calculated for the M1 timeframe, and then, depending on the current timeframe of the chart, divides the data.

This binary file is prepared by the Balance\_Process() function called in OnInit(). If the user adds a symbol that was not used for trading, and there is no corresponding binary file, then the indicator will display the history of total balance in both variants.

Now let's talk about the format of data passed to the indicator. To divide the information correctly, it is not enough to know four points of a bar (Open, High, Low and Close).

Besides it, you need to know what is the first - high or low. For restoring the information, the Balance\_Process() function uses the same principle as the "1 minute OHLC" mode of the strategy tester - if the close price of a bar is lower than the open price, then the second point is the maximum, otherwise it is minimum.

The same scheme is used for the third point. As a result, we obtain the format of data (open, 2-nd point, 3-rd point, close) where everything is consistent and definite. This format is used for dividing the M1 history of quotes. And the result is used for calculation of the history of balance and equity, depending on parsed trade history (in the same format).

### Conclusion

In conclusion, I want to say that this development does not pretend to be a tester visualizer, even though we can use it in this way. However, if the ideas implemented in it appear to be useful in the real visualizer, I will be glad. The development of the player is aimed at helping traders and EA writers with preparing to the forthcoming championship and with the hard work of analyzing trade strategies.

In addition, I want to say that the MQL5 language is a powerful tool for programming that allows implementing pretty huge projects. If you're still reading this article then you've probably noticed that the "player" project consist of nearly 8000 lines of code. I cannot imagine writing such a code in MQL4, and the matter is not in describing all of it with procedures. If there is a ready development, it can be remade in the procedure style. But developing such projects from a scratch is really hard.

Good luck!

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/242](https://www.mql5.com/ru/articles/242)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/242.zip "Download all attachments in the single ZIP archive")

[reporthistory-630165.zip](https://www.mql5.com/en/articles/download/242/reporthistory-630165.zip "Download reporthistory-630165.zip")(61.18 KB)

[player\_history\_trades.zip](https://www.mql5.com/en/articles/download/242/player_history_trades.zip "Download player_history_trades.zip")(113.51 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Self-organizing feature maps (Kohonen maps) - revisiting the subject](https://www.mql5.com/en/articles/2043)
- [Debugging MQL5 Programs](https://www.mql5.com/en/articles/654)
- [Electronic Tables in MQL5](https://www.mql5.com/en/articles/228)
- [Using Pseudo-Templates as Alternative to C++ Templates](https://www.mql5.com/en/articles/253)
- [How to Copy Trading from MetaTrader 5 to MetaTrader 4](https://www.mql5.com/en/articles/189)
- [Evaluation of Trade Systems - the Effectiveness of Entering, Exiting and Trades in General](https://www.mql5.com/en/articles/137)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/4274)**
(25)


![](https://c.mql5.com/avatar/avatar_na2.png)

**\[Deleted\]**
\|
6 Nov 2016 at 23:18

**Dennis Kirichenko:**

Nicholas, huge **respect** for the idea and the work done!!! Way to go!

The idea is as old as Moscow ...


![Aleksei Kuznetsov](https://c.mql5.com/avatar/2013/10/52601B64-7C6E.jpg)

**[Aleksei Kuznetsov](https://www.mql5.com/en/users/elibrarius)**
\|
9 Nov 2016 at 19:51

Does anyone have this player working?


![Kevin Tello](https://c.mql5.com/avatar/avatar_na2.png)

**[Kevin Tello](https://www.mql5.com/en/users/kevintello1996-gmail)**
\|
2 Oct 2021 at 15:22

hello i don't know how to install it, could you help me?


![Amos Tsopotsa](https://c.mql5.com/avatar/2019/9/5D71C18E-200D.JPG)

**[Amos Tsopotsa](https://www.mql5.com/en/users/51a6ab68)**
\|
1 May 2022 at 14:13

**MetaQuotes:**

New article [The Player of Trading Based on Deal History](https://www.mql5.com/en/articles/242) is published:

Author: [Николай](https://www.mql5.com/en/users/Urain)

a very interesting good read  are there any further improvements you made n these codes i hope your account is still active i would want to have a chat with you on this

![Vaal Rog](https://c.mql5.com/avatar/avatar_na2.png)

**[Vaal Rog](https://www.mql5.com/en/users/20773231)**
\|
11 Apr 2023 at 01:37

Hello, I have been trying to install this tool for several days, but I have not been able to, is there a manual or can you help me. Thank you very much.

I would like to install the tool for [backtesting](https://www.mql5.com/en/articles/2612 "Article: Testing trading strategies on real ticks "), as it is indispensable in this wonderful world of trading.

![Enhancing the Quality of the Code with the Help of Unit Test](https://c.mql5.com/2/17/936_17.png)[Enhancing the Quality of the Code with the Help of Unit Test](https://www.mql5.com/en/articles/1579)

Even simple programs may often have errors that seem to be unbelievable. "How could I create that?" is our first thought when such an error is revealed. "How can I avoid that?" is the second question which comes to our mind less frequently. It is impossible to create absolutely faultless code, especially in big projects, but it is possible to use technologies for their timely detection. The article describes how the MQL4 code quality can be enhanced with the help of the popular Unit Testing method.

![How to Add New UI Languages to the MetaTrader 5 Platform](https://c.mql5.com/2/0/language.png)[How to Add New UI Languages to the MetaTrader 5 Platform](https://www.mql5.com/en/articles/311)

User interface of the MetaTrader 5 platform is translated into several languages. Don't worry if your native language is not among the supported ones. You can easily complete the translation using the special MetaTrader 5 MultiLanguage Pack utility, offered by MetaQuotes Software Corp. for free to all comers. In this article we will show some examples of how to add a new user interface languages to the MetaTrader 5 platform.

![William Blau's Indicators and Trading Systems in MQL5. Part 1: Indicators](https://c.mql5.com/2/0/MQL5_Willam_Blau_1.png)[William Blau's Indicators and Trading Systems in MQL5. Part 1: Indicators](https://www.mql5.com/en/articles/190)

The article presents the indicators, described in the book by William Blau "Momentum, Direction, and Divergence". William Blau's approach allows us to promptly and accurately approximate the fluctuations of the price curve, to determine the trend of the price movements and the turning points, and eliminate the price noise. Meanwhile, we are also able to detect the overbought/oversold states of the market, and signals, indicating the end of a trend and reversal of the price movement.

![Payments and payment methods](https://c.mql5.com/2/0/mql5_payment__1.png)[Payments and payment methods](https://www.mql5.com/en/articles/302)

MQL5.community Services offer great opportunities for traders as well as for the developers of applications for the MetaTrader terminal. In this article, we explain how payments for MQL5 services are performed, how the earned money can be withdraw, and how the operation security is ensured.

[![](https://www.mql5.com/ff/si/0nfwvn6yhmgzf97n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F117%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dorder.expert%26utm_content%3Dorder.freelance%26utm_campaign%3D0622.MQL5.com.Internal&a=tunpwtbhegzufrqocbwiszessdutnobs&s=d9e7484e15300021b4066b1df77a94a1352f9e7c326d5113006bb4f6476bafeb&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=ieidhmapxhqfwbdibzeibvpxqladdefw&ssn=1769251204193028417&ssn_dr=0&ssn_sr=0&fv_date=1769251204&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F242&back_ref=https%3A%2F%2Fwww.google.com%2F&title=The%20Player%20of%20Trading%20Based%20on%20Deal%20History%20-%20MQL5%20Articles&scr_res=1920x1080&ac=17692512046411211&fz_uniq=5083027115797910404&sv=2552)

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