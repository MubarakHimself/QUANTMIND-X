---
title: Marvel Your MQL5 Customers with a Usable Cocktail of Technologies!
url: https://www.mql5.com/en/articles/728
categories: Integration
relevance_score: 3
scraped_at: 2026-01-23T21:20:00.026707
---

[![](https://www.mql5.com/ff/sh/a27a2kwmtszm2m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
MQL5 Channels - Messenger for traders\\
\\
Subscribe to traders' channels or create your own.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=vpcudokyepxfrcxrpjcktglhsjlemtza&s=f08ad2c1289e29bd5630f1ef977aef297d5cdbfcb686faed4a4b0f1e276d3c4a&uid=&ref=https://www.mql5.com/en/articles/728&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5071752818055917006)

MetaTrader 5 / Examples


### Introduction

MQL5 provides programmers with a very complete set of functions and object-oriented API thanks to which they can do everything they want within the MetaTrader environment. However, Web Technology is an extremely versatile tool nowadays that may come to the rescue in some situations when you need to do something very specific, want to marvel your customers with something different or simply you do not have enough time to master a specific part of MT5 Standard Library. Today's exercise walks you through a practical example about how you can manage your development time at the same time as you also create an amazing tech cocktail.

This tutorial is showing you how you can create a CSV file from an awesome web-driven GUI ( [Graphical User Interface](https://en.wikipedia.org/wiki/Graphical_user_interface "https://en.wikipedia.org/wiki/Graphical_user_interface")). Specifically, we will create the news calendar used by the EA explained in [Building an Automatic News Trader](https://www.mql5.com/en/articles/719) article. The web technologies we are going to work with are HTML5, CSS and JQuery. This exercise is especially interesting for MQL5 developers who already have some Web knowledge or want to learn some of those technologies in order to combine them with their MQL5 developments. Personally I had the opportunity to work in recent years with JQuery, HTML5 and CSS, so I am very familiar to it all. All this is known as the client side of a web application.

![Cocktails. Picture distributed by mountainhiker under a Creative Commons License on Flickr](https://c.mql5.com/2/6/cocktails.jpg)

**Figure 1. Cocktails. Picture distributed by mountainhiker under a Creative Commons License on Flickr**

This month, I have no material time to study [Classes for Creation of Control Panels and Dialogs](https://www.mql5.com/en/docs/standardlibrary/controls), so I have preferred to take the approach explained in [Charts and diagrams in HTML](https://www.mql5.com/en/articles/244). That's why I opened the thread [EA's GUIs to enter data](https://www.mql5.com/en/forum/13188).

### 1\. The Cocktail of Technologies

The HTML language and the Web were born in 1989, not a long time ago, to describe and share some scientific documents in the [CERN](https://www.mql5.com/go?link=http://home.cern/ "http://home.web.cern.ch/") (European Organization for Nuclear Research). HTML was originally conceived as a communication tool for the scientific community. Since then, the HyperText Markup Language has been constantly evolving in order to share information among people all over the world. I don't want to bore you with some history of science and technology, but just remember this information.

So, in the beginning, as developers took HTML and made it their own to make cool things, they used to write the HTML code, the CSS style code and the JavaScript all in one single document. They mixed everything and soon realized that they had to adopt a philosophy of separating things so that web applications worked with less errors. That is why today we always separate the structure (HTML), the visual presentation (CSS) and the behavior (JavaScript) when working on the client side of a web app. Whenever you want to do an exercise like today's, you should know these three interdependent technologies.

### 2\. And What About Usability?

[Usability](https://en.wikipedia.org/wiki/Usability "https://en.wikipedia.org/wiki/Usability") means easy to use. It means how easy it is to learn to use something, how efficient it feels to use it, how easily you can go wrong when using something, or how much users like to use that thing. Let's briefly recall. We are developing a web-driven GUI so that our customers can make a news calendar for our MQL5 product, the EA explained in [Building an Automatic News Trader](https://www.mql5.com/en/articles/719) article. The aspect of usability is important because we are integrating a number of technologies. But we have to be careful, all this has to be transparent to the user! We will succeed when our customers use our web-driven solution and they end up saying, 'How cool has been using this product!", "How I love it!"

Our GUI design follows a number of known usability guidelines:

- It has a light background color and a dark font color in order to make the text readable.
- The font size is large enough so that the text can be read by a normal audience.
- The icons are not alone, but go along with its corresponding text description.
- The links are underlined and are blue. This makes them visible.

### 3\. The Web GUI for Creating a CSV of News

This exercise's entire code is available in **news-watcher-csv.html**. This is what we will deliver to our customer together with our EA. By the way, I recommend you first take a look at that file which I will explain in detail below.

**3.1. Loading the Behavior Layer (JS) and the Display Layer (CSS)**

The [Google Hosted Libraries](https://www.mql5.com/go?link=https://developers.google.com/speed/libraries/ "https://developers.google.com/speed/libraries/devguide") is a content distribution network for the most popular open-source JavaScript libraries. You can take from Google's servers the JS libraries you need so that you don't have to copy them on your machine for working with them, your web app will request Google in the first HTTP request the libraries you specify. This is the approach taken by this exercise in order for the browser to load the JQuery library and its corresponding CSS.

We load all the necessary stuff in the document's header tag:

```
<link rel="stylesheet" href="http://ajax.googleapis.com/ajax/libs/jqueryui/1.10.3/themes/smoothness/jquery-ui.css" />
<link rel="stylesheet" href="jquery.timepicker.css" />
<script src="http://ajax.googleapis.com/ajax/libs/jquery/1.10.2/jquery.min.js"></script>
<script src="http://ajax.googleapis.com/ajax/libs/jqueryui/1.10.3/jquery-ui.min.js"></script>
<script src="jquery.timepicker.min.js"></script>
```

Nevertheless, we also include the CSS and the JavaScript of [jquery.timepicker](https://www.mql5.com/go?link=http://jonthornton.github.io/jquery-timepicker/ "http://jonthornton.github.io/jquery-timepicker/"), the widget that allows customers to select a specific time. This is because there are no silver bullets! [JQuery UI](https://www.mql5.com/go?link=http://jqueryui.com/ "http://jqueryui.com/") does not come with any visual widget for selecting times of a day out of the box, so we have to resort to this third party component. The files **jquery.timepicker.css** and **jquery.timepicker.min.js** are not available on Google's servers, so we must copy them on our local machine and reference them in the HTML document.

This may be uncomfortable for our customers, so in the context of this exercise I recommend you first upload these two files to a reliable public server and then point them from your HTML document, just as we do with the JQuery library which is hosted on Google. The fewer files see our customers, the better. Remember what we discussed earlier regarding usability! This is left to you as an improvement exercise. In any case, keep in mind that all this assumes that our customer has an Internet connection in order to use this GUI.

**3.2. JQuery Widgets Used in Our GUI** Our GUI contains the following JQuery visual controls. Remember that all this is already programmed in this JavaScript framework! Please, have a look at the official documentation to deepen your knowledge in [JQuery](https://www.mql5.com/go?link=http://jquery.com/ "http://jquery.com/") and [JQuery UI](https://www.mql5.com/go?link=http://jqueryui.com/ "http://jqueryui.com/"). I will explain a little later how the main jQuery program works. **3.2.1. Datepicker** This control allows users to easily enter a particular date.

**![Figure 2. jQuery datepicker](https://c.mql5.com/2/6/news-watcher-datepicker__1.jpg)**

**Figure 2. jQuery datepicker**

**3.2.2. Timepicker** This plugin helps users to enter a specific time. It is inspired by Google Calendar.

**![Figure 3. jQuery timepicker. This extension is written by Jon Thornton](https://c.mql5.com/2/6/news-watcher-timepicker__1.jpg)**

**Figure 3. [jQuery timepicker](https://www.mql5.com/go?link=http://jonthornton.github.io/jquery-timepicker/ "http://jonthornton.github.io/jquery-timepicker/"). This extension is written by Jon Thornton**

**3.2.3. Sortable** This is for reordering the news in a list or grid using the mouse.

![Figure 4. jQuery sortable](https://c.mql5.com/2/6/news-watcher-sortable__1.jpg)

**Figure 4. jQuery sortable**

**3.2.4. Dialog** This control is for showing some content in an interactive overlay. Usability purists do not recommend it because it is a little intrusive to the user. It also forces users to carefully move their hands to interact with it, so people with motor problems may feel uncomfortable interacting with a jQuery dialog, however, this widget can be used in some contexts. I am aware of this. Improving this Graphical User Interface so that the CSV content is displayed in a somewhat less intrusive way is left as an exercise for you.

**![Figure 5. jQuery dialog](https://c.mql5.com/2/6/news-watcher-dialog__1.jpg)**

**Figure 5. jQuery dialog**

**3.3. The Marvel, Adding Custom Value to Your Product**

Maybe you, either as a freelancer or as a company, are offering solutions based on MT5 and have a database storing your customers' preferences for marketing issues. In that case, you can take advantage of that information to customize your web-driven GUI:

![Figure 6. You can incorporate a Scarlatti clip so that Bob can create his calendar in a relaxed environment](https://c.mql5.com/2/6/news-watcher-gui-700__3.jpg)

**Figure 6. You can incorporate a Scarlatti clip so that Bob can create his calendar in a relaxed environment**

In this example your customer Bob loves Domenico Scarlatti, the famous Italian/Spanish Baroque composer, so you incorporate a Scarlatti clip so that Bob can create his calendar in a relaxed environment.

**3.4. Our GUI's JQuery code**

Please, now open the file **news-watcher-csv.html** and observe the three parts that we discussed earlier in this tutorial. Remember that a client web app consists of presentation (CSS), structure (HTML) and behavior (JavaScript/jQuery). With all this in view you will easily understand what the jQuery program does.

The main jQuery program is a snap. First of all, it makes a small fix in YouTube's iframe so that the document's HTML5 code can properly validate. Then, the app initializes the widgets that we discussed earlier, and finally programs the behaviour of both the button for adding news and the link for creating the CSV content. That's it!

```
$(function() {

    // fix YouTube's zIndex iframes issue

    $('iframe').each(function(){
        var url = $(this).attr("src");
        $(this).attr("src",url+"?wmode=transparent");
    });

    // Init GUI elements

    $("#datepicker").datepicker({ 'dateFormat': 'yy.m.dd' });
    $("#timepicker").timepicker({ 'timeFormat': 'H:i:s' });
    $("#sortable").sortable().disableSelection();
    $("#news-calendar").hide();

    // Buttons behavior

    $("#add-new").click(function() {

        $("#news-calendar").fadeIn("500");

        $("#sortable").append('<li class="ui-state-default"><span class="ui-icon ui-icon-arrowthick-2-n-s"></span>' +
            $('#base-currency').val() + ';' +
            $('#datepicker').val() + ';' +
            $('#timepicker').val() + ';' +
            $('#description').val() + '</li>');

    });

    $("#get-csv").click(function() {

        var csv = "Country;Time;Event\n";

        $("#sortable li").each(function(){

            csv += $(this).text() + "\n";

        });

        $("#dialog").empty().dialog({
            modal: true,
            width: 650,
            show: {
                effect: "blind",
                duration: 400
            },
            hide: {
                effect: "explode",
                duration: 400
            },
            buttons: {
                Ok: function() {
                $( this ).dialog( "close" );
                }
            }
        }).append(csv.replace(/\n/g, '<br/>'));

        return false;

    });

});
```

The jQuery code above dynamically manipulates the HTML structure below:

```
<div id="csv-generator">
    <p>Dear Bob, please, generate the CSV file for NewsWatcher EA while you listen to your favourite music:</p>
    <iframe id="video" width="400" height="280" src="https://www.youtube.com/embed/4pSh8kHKuYw" allowfullscreen></iframe>
    <div id="form">
        <p>Enter news in your CSV calendar:</p>
        <div class="form-field">
            <label>Base currency:</label>
            <select id="base-currency">
                <option>AUD</option>
                <option>CAD</option>
                <option>CHF</option>
                <option>CNY</option>
                <option>EUR</option>
                <option>GBP</option>
                <option>JPY</option>
                <option>NZD</option>
                <option>USD</option>
            </select>
        </div>
        <div class="form-field">
            <label>Date:</label>
            <input type="text" id="datepicker"/>
        </div>
        <div class="form-field">
            <label>Time:</label>
            <input type="text" id="timepicker"/>
        </div>
        <div class="form-field">
            <label>Description:</label>
            <input id="description" type="text"/>
        </div>
        <div id="add-new-button" class="form-field">
            <label></label>
            <button id="add-new">+ Add new</button>
        </div>
    </div>
</div><!-- end of csv-generator -->
<div id="news-calendar">
    <p><a id="get-csv" href="#">Get CSV</a></p>
    <ul id="sortable"></ul>
</div>
<div id="dialog" title="Please, copy and paste in data_folder\MQL5\FILES\news_watcher.csv"></div>
```

Finally, it goes without saying that the presentation layer is the following CSS which is clearly separated from the other two layers:

```
<style>

    /* Simple CSS reset */

    * {
        margin: 0;
        padding: 0
    }

    /* HTML tags */

    body {
        font-family: Arial, Helvetica, sans-serif;
        padding: 0em 2em
    }

    header { border-bottom: 1px solid #cccccc }
    footer { border-top: 1px solid #cccccc; margin-top: 2em; padding-top: 1em }

    h1, h2, h3, h4, h5, p { padding: 1em 0em }

    label {
        width: 150px;
        display: inline-block;
        margin: 5px;
        text-align: right
    }

    input { border: 1px solid #cccccc }
    button, option, input { padding: 4px }
    select { padding: 3px }

    /* Custom ids */

    #csv-generator {
        filter: progid:DXImageTransform.Microsoft.gradient(startColorstr='#ffffff', endColorstr='#fafafa');
        background: -webkit-gradient(linear, left top, left bottom, from(#ffffff), to(#fafafa));
        background: -moz-linear-gradient(top, #ffffff, #fafafa);
        width: 100%;
        clear: both;
        overflow: hidden
    }

    #video {
        width: 30%;
        float: left;
        margin-right: 1em;
        -moz-box-shadow: 0px 3px 5px #888;
        -webkit-box-shadow: 0px 3px 5px #888;
        box-shadow: 0px 3px 5px #888;
        border:0
    }

    #dialog { font-size: 12px }

    /* Form for adding the news */

    #form { width: 65%; float: right }
    input#datepicker, input#timepicker { width: 100px }
    input#description { width: 300px }
    #add-new-button { float: left; margin-right: 1em }

    /* Sortable */

    #news-calendar { clear: both; margin-top: 2em; padding: 0em 2em 2em 1em; background: #fafafa; border: 1px solid #cccccc }
    #sortable { list-style-type: none; margin: 0; padding: 0em; width: auto; clear: both }
    #sortable li { margin: 3px; padding: 0.4em; padding-left: 1.5em; }
    #sortable li span { position: absolute; margin-left: -1.3em; }

    a#get-csv {
        background: url('http://icons.iconarchive.com/icons/deleket/soft-scraps/24/File-New-icon.png') no-repeat 0px 8px;
        padding: 10px 58px 20px 30px;
    }

    /* Custom classes */

    .form-field { margin-bottom: 0.5em }

    /* Overwrites */

    .ui-dialog-title { font-size: 12px }

</style>
```

This simple presentation layer can also be improved, for example, by writing a CSS code that takes into account all mobile devices. Currently this is possible with CSS3 media queries and [responsive web design](https://en.wikipedia.org/wiki/Responsive_web_design "https://en.wikipedia.org/wiki/Responsive_web_design") but there is no space enough in this article to explore this technique, so it is left as an exercise for the reader.

### Conclusion

Web Technology is an extremely versatile tool nowadays that may come to the rescue in some situations. Today we have created a web-based [Graphical User Interface](https://en.wikipedia.org/wiki/Graphical_user_interface "https://en.wikipedia.org/wiki/Graphical_user_interface") for creating a news calendar in CSV format to be used by the Expert Advisor that we already developed in [Building an Automatic News Trader](https://www.mql5.com/en/articles/719) article.

HTML5, CSS and JQuery are the main web technologies we have worked with. All this is known as the client side of a web application. We also briefly discussed the need to always think of the person who will use the interface, making a brief note on issues of [usability](https://en.wikipedia.org/wiki/Usability "https://en.wikipedia.org/wiki/Usability").

**\*Very important notes**: The HTML5 code of this tutorial has been validated through the [W3C Markup Validation Service](https://www.mql5.com/go?link=http://validator.w3.org/ "http://validator.w3.org/") to guarantee a quality product, and has been tested in recent versions of Chrome and Firefox browsers. IE8 is becoming obsolete, please do not run this exercise in that browser. jQuery 2.0 doesn’t support IE 6, 7 or 8. The three files of this tutorial are sent in txt format because MQL5 does not allow sending HTML, CSS and JavaScript. Please, download them and rename **news-watcher-csv.txt** to **news-watcher-csv.html**, **jquery.timepicker.txt** to **jquery.timepicker.css** and **jquery.timepicker.min.txt** to **jquery.timepicker.min.js**.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/728.zip "Download all attachments in the single ZIP archive")

[news-watcher-csv.txt](https://www.mql5.com/en/articles/download/728/news-watcher-csv.txt "Download news-watcher-csv.txt")(8.39 KB)

[jquery7timepicker.txt](https://www.mql5.com/en/articles/download/728/jquery7timepicker.txt "Download jquery7timepicker.txt")(1.42 KB)

[jquerygtimepickerbmin.txt](https://www.mql5.com/en/articles/download/728/jquerygtimepickerbmin.txt "Download jquerygtimepickerbmin.txt")(9.95 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Building a Social Technology Startup, Part II: Programming an MQL5 REST Client](https://www.mql5.com/en/articles/1044)
- [Building a Social Technology Startup, Part I: Tweet Your MetaTrader 5 Signals](https://www.mql5.com/en/articles/925)
- [Raise Your Linear Trading Systems to the Power](https://www.mql5.com/en/articles/734)
- [Extending MQL5 Standard Library and Reusing Code](https://www.mql5.com/en/articles/741)
- [Building an Automatic News Trader](https://www.mql5.com/en/articles/719)
- [Another MQL5 OOP Class](https://www.mql5.com/en/articles/703)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/14665)**
(10)


![](https://c.mql5.com/avatar/avatar_na2.png)

**\[Deleted\]**
\|
31 Oct 2013 at 22:27

Thank you for your improvement,

Any extensionis welcome, that's theidea! We can usewebtechnologyin our MQL5 developments to create cross-platform software. I take this opportunity tomake a comment, however.

In general,JavaScript codeshould notuse ActiveX controlsfor security issues, becausemalicious software writers could write ActiveX programs to get into the user's windows system. Your customers should trust you in order for you to do something like that, and you should clearly explain them that you are using ActiveX to run certain functions, etc.

Of course, you can useActiveXfor yourown use, interacting withyour Windowsas you want.

More info here, as an example:

[http://entertainment.howstuffworks.com/activex-for-animation3.htm](https://www.mql5.com/go?link=https://entertainment.howstuffworks.com/activex-for-animation.htm "http://entertainment.howstuffworks.com/activex-for-animation3.htm")

[http://articles.winferno.com/web-browser-security/dangers-of-activex/](https://www.mql5.com/go?link=http://articles.winferno.com/web-browser-security/dangers-of-activex/ "http://articles.winferno.com/web-browser-security/dangers-of-activex/")

![Ivan Negreshniy](https://c.mql5.com/avatar/2013/7/51E51A58-4224.jpg)

**[Ivan Negreshniy](https://www.mql5.com/en/users/hlaiman)**
\|
1 Nov 2013 at 21:18

Thank you for the remark. Indeed, ActiveX, as well as other Microsoft internet technologies, may be used for viruses distribution. Same situation is observed in other solutions for communication, such as Google etc. Of course, the main distributor of viruses today is the Internet.

From this point of view, most of the similar tasks would be much safer to solve in a local text editor. And it would be safer to 100% on a typewriter. And instead of a computer would be better to use calculator ;). But this, unfortunately, would affect on usability.

But I think we should not worry too much about this, because the fight against viruses carried by special anti-virus programs and sites. It only remains to protect our clients in MetaTrader environment, where the risk to a lesser extent regard to viruses, but to a greater extent regard to Forex trading.


![](https://c.mql5.com/avatar/avatar_na2.png)

**\[Deleted\]**
\|
3 Nov 2013 at 00:38

**hlaiman:**

Thank you for the remark. Indeed, ActiveX, as well as other Microsoft internet technologies, may be used for viruses distribution. Same situation is observed in other solutions for communication, such as Google etc. Of course, the main distributor of viruses today is the Internet.

From this point of view, most of the similar tasks would be much safer to solve in a local text editor. And it would be safer to 100% on a typewriter. And instead of a computer would be better to use calculator ;). But this, unfortunately, would affect on usability.

But I think we should not worry too much about this, because the fight against viruses carried by special anti-virus programs and sites. It only remains to protect our clients in MetaTrader environment, where the risk to a lesser extent regard to viruses, but to a greater extent regard to Forex trading.

Well, I think that the remark about ActiveX must be done. I like the idea of cross-platform EAs, but Web users must be aware that ActiveX may be dangerous.

That said, if your customers trust you, you can use ActiveX. You can put a dialog box explaining users very well what your app does, asking them to accept that your app needs to run certain [functions](https://www.mql5.com/en/docs/constants/namedconstants/compilemacros "MQL5 documentation: Predefined Macrosubstitutions") to properly work, etc.

IMHO, I think that web technology can help you to complement some of your MQL5 developments without using ActiveX, in a very secure way.


![Ivan Negreshniy](https://c.mql5.com/avatar/2013/7/51E51A58-4224.jpg)

**[Ivan Negreshniy](https://www.mql5.com/en/users/hlaiman)**
\|
4 Nov 2013 at 18:39

**laplacianlab:**

Well, I think that the remark about ActiveX must be done. I like the idea of cross-platform EAs, but Web users must be aware that ActiveX may be dangerous.

That said, if your customers trust you, you can use ActiveX. You can put a dialog box explaining users very well what your app does, asking them to accept that your app needs to run certain functions to properly work, etc.

IMHO, I think that web technology can help you to complement some of your MQL5 developments without using ActiveX, in a very secure way.

I want to remind that the DDE/OLE/ActiveX/COM/DCOM is an evolution of Microsoft technologies of inter-program interaction. These technologies are supported by almost all Windows applications and services. These technologies are the basis of .Net

Therefore, complete abandonment of ActiveX technology is equivalent to the complete abandonment of Windows OS, and consequently of all programs written for Windows, such as MetaTrader Terminals and MetaEditor IDE.

To better explain the subject of our discussion, I will give an example of life. We know that viruses infect people by [moving](https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/ma "MetaTrader 5 Help: Moving Average indicator") through airborne droplets. To fight infection, we can use the anti-virus vaccines, tablets, masks, etc. But we can also fight more radically. By analogy with your offer, complete abandonment of ActiveX, completely abandon the air. But who needs this security, if by taking away from viruses their carrier - the air, we can destroy target of infection - people? )


![Sergey Golubev](https://c.mql5.com/avatar/2012/12/50D09349-3D90.jpg)

**[Sergey Golubev](https://www.mql5.com/en/users/newdigital)**
\|
6 Jun 2014 at 09:28

[Forum on trading, automated trading systems and testing trading strategies](https://www.mql5.com/en/forum)

[Press review](https://www.mql5.com/en/forum/12423/page172#comment_938992)

[newdigital](https://www.mql5.com/en/users/newdigital "https://www.mql5.com/en/users/newdigital"), 2014.06.06 09:25

3 Steps to Trade Major News Events(based on [dailyfx article](https://www.mql5.com/go?link=https://www.dailyfx.com/forex/education/trading_tips/post_of_the_day/2014/06/06/3-Steps-to-Trade-Major-News-Events.html "http://www.dailyfx.com/forex/education/trading_tips/post_of_the_day/2014/06/06/3-Steps-to-Trade-Major-News-Events.html"))

Talking Points:

- News releases can be stressful on traders
- Develop a plan before the event arrives

Major news releases can be stressful on traders. That stress can show up for a variety of trading styles.

Perhaps you are already in a good position with a good entry and you are afraid the news release may take a bite out of your good entry.

Perhaps you want to enter into a new position as prices are near a technically sound entry point, but you are uncertain if the technical picture will hold up through the volatile release. Therefore, you agonize over the decision of whether to enter now or after the news event.

Maybe, you like to be in the action and initiating new positions during the release. The fast paced volatility during the news release still gets makes your palms sweat as you place trades.

As you can see, news events stress traders in a variety of ways.

Today, we are going to cover three steps to trade news events.

![](https://c.mql5.com/3/40/news1.png)

**Step 1 - Have a Strategy**

It sounds simple, yet the emotion of the release can easily draw us off course. We see prices moving quickly in a straight line and are afraid to miss out or afraid to lose the gains we have been sitting on. Therefore, we make an emotional decision and act.

Having a strategy doesn’t have to be complicated. Remember, staying out of the market during news and doing nothing is a strategy.

A strategy for the trader with a floating profit entering the news event could be as simple as “I am going to close off half my position and move my stop loss to better than break even.”

For the trader wanting to initiate a new position that is technically based, they may decide to wait until at least 15 minutes after the release, then decide if the set-up is still valid.

The active news trader may realize they need a plan of buy and sell rules because they trade based on what ‘feels good.’

**Step 2 - Use Conservative Leverage**

If you are in the market when the news is released, make sure you are implementing conservative amounts of leverage. We don’t know where the prices may go and during releases, prices tend to move fast. Therefore, de-emphasize the influence of each trade on your account equity by using low amounts of leverage.

Our Traits of Successful Traders research found that traders who implement less than ten times effective leverage tend to be more profitable on average.

![](https://c.mql5.com/3/40/news2.png)

**3 - Don’t Deviate from the Strategy**

If you have taken the time to think about a strategy from step number one and if you have realized the importance of being conservatively levered, then you are 90% of the way there! However, this last 10% can arguably be the most difficult. Whatever your plan is, stick to it!

If I put together a plan to lose 20 pounds of body weight that includes eating healthier and exercising, but I continue to eat high fat and sugar foods with limited exercise, then I am only setting myself up for frustration.

You don’t have to be stressed or frustrated through fundamental news releases.

![Technical Indicators and Digital Filters](https://c.mql5.com/2/0/Indicators_as_digital_filters_MQL5.png)[Technical Indicators and Digital Filters](https://www.mql5.com/en/articles/736)

In this article, technical indicators are treated as digital filters. Operation principles and basic characteristics of digital filters are explained. Also, some practical ways of receiving the filter kernel in MetaTrader 5 terminal and integration with a ready-made spectrum analyzer proposed in the article "Building a Spectrum Analyzer" are considered. Pulse and spectrum characteristics of the typical digital filters are used as examples.

![MQL5 Cookbook: Saving Optimization Results of an Expert Advisor Based on Specified Criteria](https://c.mql5.com/2/0/avatar__7.png)[MQL5 Cookbook: Saving Optimization Results of an Expert Advisor Based on Specified Criteria](https://www.mql5.com/en/articles/746)

We continue the series of articles on MQL5 programming. This time we will see how to get results of each optimization pass right during the Expert Advisor parameter optimization. The implementation will be done so as to ensure that if the conditions specified in the external parameters are met, the corresponding pass values will be written to a file. In addition to test values, we will also save the parameters that brought about such results.

![MQL5 Cookbook: Sound Notifications for MetaTrader 5 Trade Events](https://c.mql5.com/2/0/avatar__8.png)[MQL5 Cookbook: Sound Notifications for MetaTrader 5 Trade Events](https://www.mql5.com/en/articles/748)

In this article, we will consider such issues as including sound files in the file of the Expert Advisor, and thus adding sound notifications to trade events. The fact that the files will be included means that the sound files will be located inside the Expert Advisor. So when giving the compiled version of the Expert Advisor (\*.ex5) to another user, you will not have to also provide the sound files and explain where they need to be saved.

![Lite_EXPERT2.mqh: Functional Kit for Developers of Expert Advisors](https://c.mql5.com/2/17/812_123.gif)[Lite\_EXPERT2.mqh: Functional Kit for Developers of Expert Advisors](https://www.mql5.com/en/articles/1380)

This article continues the series of articles "Expert Advisors Based on Popular Trading Systems and Alchemy of Trading Robot Optimization". It familiarizes the readers with a more universal function library of the Lite\_EXPERT2.mqh file.

[![](https://www.mql5.com/ff/sh/9nb0c8df2rmwfn89z2/01.png) MetaTrader VPS vs regular cloud hosting services8 reasons why our solution is the best option for automated tradingRead](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/450486&a=dgmsfszgoedimaicrqqmagvqzpwuxkur&s=c59e3617ccf44fd54d4c50a03b44fd689ff7507b8fe4990c83772cc5419e627d&uid=&ref=https://www.mql5.com/en/articles/728&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5071752818055917006)

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