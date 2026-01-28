---
title: Building an Interactive Application to Display RSS Feeds in MetaTrader 5
url: https://www.mql5.com/en/articles/1589
categories: Expert Advisors
relevance_score: 6
scraped_at: 2026-01-23T17:30:44.597897
---

[We've created a channel for MQL5 developersFollow MQL5.community on social media and be the first to receive important updatesLearn more![](https://www.mql5.com/ff/sh/a83xrgctr82w45z9z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=pwgdbvtemvkwsfltqonysypfvtrtufji&s=e99a66a1660cd810b1edbac65597df695e2c2220d1e937834f402f9aeabd4289&uid=&ref=https://www.mql5.com/en/articles/1589&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5068321542913325073)

MetaTrader 5 / Expert Advisors


### Contents

- [Introduction](https://www.mql5.com/en/articles/1589#intro)
- [1\. RSS documents in general](https://www.mql5.com/en/articles/1589#chapter1)
- [2\. Overall structure of the application](https://www.mql5.com/en/articles/1589#chapter2)

  - [2.1. The user interface](https://www.mql5.com/en/articles/1589#c2_1)

  - [2.2. Code implementation](https://www.mql5.com/en/articles/1589#c2_2)

  - [2.3. The easy XML parser](https://www.mql5.com/en/articles/1589#c2_3)

  - [2.4. Expert Advisor code](https://www.mql5.com/en/articles/1589#c2_4)

  - [2.5. Methods for initialization of controls](https://www.mql5.com/en/articles/1589#c2_5)

  - [2.6. Methods for RSS document processing](https://www.mql5.com/en/articles/1589#c2_6)

  - [2.6.1. LoadDocument()](https://www.mql5.com/en/articles/1589#c2_6_1)

  - [2.6.2. ItemNodesTotal()](https://www.mql5.com/en/articles/1589#c2_6_2)

  - [2.6.3. FreeDocumentTree()](https://www.mql5.com/en/articles/1589#c2_6_3)

  - [2.7. Methods for extracting information from the document tree](https://www.mql5.com/en/articles/1589#c2_7)

  - [2.7.1. getChannelTitle()](https://www.mql5.com/en/articles/1589#c2_7_1)

  - [2.7.2. getTitle()](https://www.mql5.com/en/articles/1589#c2_7_2)

  - [2.8. Methods for formatting text](https://www.mql5.com/en/articles/1589#c2_8)

  - [2.8.1. FormatString()](https://www.mql5.com/en/articles/1589#c2_8_1)
  - [2.8.2. removeTags()](https://www.mql5.com/en/articles/1589#c2_8_2)
  - [2.8.3. removeSpecialCharacters()](https://www.mql5.com/en/articles/1589#c2_8_3)
  - [2.8.4. tagPosition()](https://www.mql5.com/en/articles/1589#c2_8_4)

  - [2.9. Methods for handling events of independent controls](https://www.mql5.com/en/articles/1589#c2_9)

  - [2.9.1. OnChangeListView()](https://www.mql5.com/en/articles/1589#c2_9_1)

  - [2.9.2. OnObjectEdit()](https://www.mql5.com/en/articles/1589#c2_9_2)

  - [2.9.3. OnClickButton1/2()](https://www.mql5.com/en/articles/1589#c2_9_3)

  - [2.10. Implementation of CRssReader class](https://www.mql5.com/en/articles/1589#c2_10)
  - [2.11. The Expert Advisor code](https://www.mql5.com/en/articles/1589#c2_11)

- [Conclusion](https://www.mql5.com/en/articles/1589#conclusion)

### Introduction

The article " [Reading RSS News Feeds by Means of MQL4](https://www.mql5.com/en/articles/1366 "Link to MQL4 article")" described a rather rudimentary script that could be used to display RSS feeds in the terminal's console by means of a simple library that was originally built for parsing HTML-documents.

With the advent of MetaTrader 5 and the MQL5 programming language I thought it possible to create an interactive application that would be able to display RSS content better. This article describes how to produce this application using the extensive [MQL5 Standard library](https://www.mql5.com/en/docs/standardlibrary) and some other tools developed by MQL5 community contributors.

### 1\. RSS documents in general

Before we get to grips with the specifics of the application, I think it necessary to give an overview of the general structure of an [RSS document.](https://www.mql5.com/go?link=https://www.xul.fr/en-xml-rss.html "Rss tutorial")

In order to understand the description to follow you need be familiar with the extensible markup language and related concepts. Please refer to [XML Tutorial](https://www.mql5.com/go?link=http://www.tutorialspoint.com/xml/index.htm "Xml tutorial") if you are not familiar with XML-documents. Note that in this article, node refers to a tag in an XML-document. As mentioned in the MQL4 article referenced above, RSS-files are simply XML-documents with a specific tag structure.

Each RSS document has a global container, the RSS tag. This is common to all RSS documents. The channel tag is always a direct descendant of the RSS tag. This contains information about the website the feed describes. From here on RSS-documents can vary in terms of the specific tags they contain, but there are some tags that all documents should contain in order to be verified RSS-files.

The required tags are:

- title - The title of the channel. Should contain the name of the website;
- link - URL of the website that provides this channel;
- description - Summary of what the website is about;
- item - one item tag at least, for the content.

The tags shown above should all be child nodes of the channel tag. The item node is the one that contains data relating to specific content.

Each item node in turn, must also contain the following tags:

- title - Title of the content;
- link - The URL link to the content;
- description - Summary of the content;
- date - Date the content was published on website.

All RSS documents contain the described tags and follow the same structure.

An example of a complete RSS document is given below.

```
<?xml version="1.0"?>
<rss version="2.0">
  <channel>
    <title>Xul.fr: Tutorials and Applications of the Web 2.0</title>
    <link>http://www.xul.fr/</link>
    <description>Ajax, JavaScript, XUL, RSS, PHP and all technologies of the Web 2.0. Building a CMS, tutorial and application.</description>
    <pubDate>Wed, 07 Feb 2007 14:20:24 GMT</pubDate>
    <item>
    <title>News on interfaces of the Web in 2010</title>
    <link>http://www.xul.fr/en/2010.php</link>
    <description>Steve Jobs explains why iPad does not support Adobe Flash:&lt;em&gt;At Adobe they are lazy.
    They have the potential to make  interesting things, but they refuse to do so.
    Apple does not support Flash because it is too buggy.
     Each time a Mac crashes, most often it is because of Flash. Nobody will use Flash.
     The world is moving  to &lt;a href="http://www.xul.fr/en/html5/" target="_parent"&gt;HTML 5&lt;/a&gt;</description>
     <pubDate>Sat, 11 Dec 10 09:41:06 +0100</pubDate>
    </item>
    <item>
      <title>Textured Border in CSS</title>
      <link>http://www.xul.fr/en/css/textured-border.php</link>
      <description>   The border attribute of the style sheets can vary in color and width, but it was not expected to give it a texture. However, only a CSS rule is required to add this graphic effect...   The principle is to assign a texture to the whole &lt;em&gt;fieldset&lt;/em&gt; and insert into it another &lt;em&gt;fieldset&lt;/em&gt; (for rounded edges) or a &lt;em&gt;div&lt;/em&gt;, whose background is the same as that of the page</description>
      <pubDate>Wed, 29 Jul 09 15:56:54  0200</pubDate>
    </item>
    <item>
      <title>Create an RSS feed from SQL, example with Wordpress</title>
      <link>http://www.xul.fr/feed/rss-sql-wordpress.html</link>
      <description>Articles contain at least the following items: And possibly, author's name, or an image. This produces the following table: The returned value is true if the database is found, false otherwise. It remains to retrieve the data from the array</description>
      <pubDate>Wed, 29 Jul 09 15:56:50  0200</pubDate>
    </item>
    <item>
      <title>Firefox 3.5</title>
      <link>http://www.xul.fr/gecko/firefox35.php</link>
      <description>Les balises audio et vid&#xE9;o sont impl&#xE9;ment&#xE9;es. Le format de donn&#xE9;e JSON est reconnu nativement par Firefox. L'avantage est d'&#xE9;viter l'utilisation de la fonction eval() qui n'est pas s&#xFB;r, ou d'employer des librairies additionnelles, qui est nettement plus lent</description>
      <pubDate>Wed, 24 Jun 09 15:18:47  0200</pubDate>
    </item>
    <item>
      <title>Contestation about HTML 5</title>
      <link>http://www.xul.fr/en/html5/contestation.php</link>
      <description>  Nobody seemed to be worried so far, but the definition of HTML 5 that is intended to be the format of billions of Web pages in coming years, is conducted and decided by a single person! &lt;em&gt;Hey, wait! Pay no attention to the multi-billions dollar Internet corporation behind the curtain. It's me Ian Hickson! I am my own man</description>
      <pubDate>Wed, 24 Jun 09 15:18:29  0200</pubDate>
    </item>
    <item>
      <title>Form Objects in HTML 4</title>
      <link>http://www.xul.fr/javascript/form-objects.php</link>
      <description>   It is created by the HTML &lt;em&gt;form&lt;/em&gt; tag:   The name or id attribute can access by script to its content. It is best to use both attributes with the same identifier, for the sake of compatibility.   The &lt;em&gt;action&lt;/em&gt; attribute indicates the page to which send the form data. If this attribute is empty, the page that contains the form that will be charged the data as parameters</description>
      <pubDate>Wed, 24 Jun 09 15:17:49  0200</pubDate>
    </item>
    <item>
      <title>DOM Tutorial</title>
      <link>http://www.xul.fr/en/dom/</link>
      <description>  The Document Object Model describes the structure of an XML or HTML document, a web page and allows access to each individual element.</description>
      <pubDate>Wed, 06 May 2009 18:30:11 GMT</pubDate>
    </item>
  </channel>
</rss>
```

### 2\. Overall structure of the application

Here I will give a description of the information that the RSS Reader should display and an over view of the graphical user interface of the application.

The first aspect the application should display is the channel title, which is contained in the title tag. This information will serve as an indication of the website the feed references.

The application should also display a snapshot of all the content the feed describes, this relates to all the item tags in the document.For each item tag, the title of the content will be displayed. Lastly, I want the RSS Reader to be able to show the description of the content, this will be the data contained in the description tag of each item node.

**2.1. The user interface**

The user interface is a function of the information to be displayed by the application.

The idea I had for a user interface is best depicted by the diagram below.

![Sketch of the application dialogue](https://c.mql5.com/2/17/app_dialogue_diagram.png)

Fig. 1. Sketch of the application dialogue

The diagram shows the different sections that make up the user interface.

- First is the title bar. This is where the channel title will be displayed;
- Input area. It is here that users will input the web address of an RSS feed;
- Title area. The title for specific content will be displayed here;
- Text area. The description of  the content is shown here;
- List view area. This scrollable list will display the titles of all the content the feed contains;
- The button on the left resets and clears text displayed in the Title, text and list view areas;

- The update current feed button  retrieves new updates for a currently loaded feed.


The RSS Reader will work in the following manner - when the program is loaded on to the chart, the empty application dialogue is displayed, the user then has to enter the web address of a desired RSS feed in the input area then press enter. This will load all the content titles, i.e the title tag values for each item tag onto the list view area. The list will be numbered from 1, this representing the most recently published content.

Each list item will be click-able, on clicking a list item, it will be highlighted and the corresponding description of the title content will be displayed in the text area. At the same time the content title will be shown more clearly in the title area section. If an error occurs during loading of the feed for what ever reason, an error message will be displayed in the text area section.

The reset button can then be used to clear any text in the text area, list view area, title area area sections.

Update current feed simply checks for any updates for the current feed.

**2.2. Code implementation**

The RSS Reader will be implemented as an Expert Advisor and the [MQL5 Standard library](https://www.mql5.com/en/docs/standardlibrary/controls) will be used.

The code will be contained in a class CRssReader which will be a descendant of the [CAppDialog](https://www.mql5.com/en/docs/standardlibrary/controls/cappdialog) class. The CAppDialog class, given in the Dialog.mqh include file, will enable the implementation of the application dialogue, which provides functionality for a title bar, and application controls for minimizing, maximizing and closing. This will be the foundation of the user interface, on top of which other sections will be added. For the sections to be added, namely being: the title area, text area, list view area and buttons. Each will be a control. The buttons will be implemented as a button control described in the Button.mqh include file.

The list view control defined in the ListViewArea.mqh include file will be used to construct the list view area section of the RSS Reader.

The edit control will obviously suffice for constructing the input area section, this control is defined in the Edit.mqh file.

The title area and text area sections provide a unique challenge when it comes to their implementation. The problem is that both need to have support for text that can be displayed on several lines. Text objects in MQL5 do not recognize new line feed characters. Another issue is that only a limited number of string characters can be displayed in one line of a text object. This means if you create a text object with a long enough description, the object will be displayed with the text cut off, only a certain number of characters will be shown. By trial and error I found out that the character limit is 63, inclusive of spaces and punctuation marks.

In order to overcome these problems I decided to implement both sections as modified list view controls. For the title area section the modified list view control will be non scrollable and will have a fixed number of list items (2). Each list item will not be click-able or selectable and the physical appearance of the control will be such that it does not look like a list. These 2 list items will represent two lines of text. If the text is too long to fit in one line, it will be split accordingly and displayed as 2 lines of text. The control for the title area section will be defined in the TitleArea.mqh file.

For the text area section, a similar approach will be applied, only this time the number of list items will be dynamic and the modified list view control will be vertically scrollable.This control will be given in the TextArea.mqh file.

The libraries mentioned so far take care of  the user interface. There is still one more important library crucial to this application that needs to be discussed. This is the library used to parse XML documents.

**2.3. The easy XML parser**

Since an RSS document is an XML file, the [EasyXML - XML Parser](https://www.mql5.com/en/code/1998) library developed by [liquinaut](https://www.mql5.com/en/users/liquinaut "https://www.mql5.com/en/users/liquinaut") and found in the Code Base is applied.

The library is quite extensive and contains almost all of the functionality needed for our RSS Reader. I made some modifications to the original library to add some extra features I felt were necessary.

These were minor additions. The first of which was the addition of an extra method called loadXmlFromUrlWebReq(). This method simply provides an alternative to using loadXmlFromUrl(), which relies on the WinInet library for processing web requests, loadXmlFromUrlWebReq() uses the built-in [WebRequest()](https://www.mql5.com/en/docs/network/webrequest) function for enabling downloads from the Internet.

```
//+------------------------------------------------------------------+
//| load xml by given url using MQL5 webrequest function             |
//+------------------------------------------------------------------+
bool CEasyXml::loadXmlFromUrlWebReq(string pUrl)
  {
//---
   string cookie=NULL,headers;
   char post[],result[];
   int res;
//---
   string _url=pUrl;
   string sStream;
   res=WebRequest("GET",_url,cookie,NULL,5000,post,0,result,headers);
//--- check error
   if(res==-1)
     {
      Err=EASYXML_ERR_WEBREQUEST_URL;
      return(Error());
     }
//---success downloading file
   sStream=CharArrayToString(result,0,-1,CP_UTF8);
//---set up cach file
   if(blSaveToCache)
     {
      bool bResult=writeStreamToCacheFile(sStream);
      if(!bResult) Error(-1,false);
     }
//---
   return(loadXmlFromString(sStream));
  }
```

The second addition was the GetErrorMsg() method, this allows one to retrieve the error message output by the parser when ever an error occurs.

```
string            GetErrorMsg(void){   return(ErrMsg);}
```

The last addition was made to correct a rather serious flaw I found when testing the easyxml parser.

I found that the library was not able to recognize XML style sheet declarations. The code mistakes a style sheet declaration for an attribute. This caused the program to get stuck in an infinite loop, as the code continuously searched for the corresponding attribute value, which never existed.

This was easily rectified, with a little modification of the skipProlog() method.

```
//+------------------------------------------------------------------+
//| skip xml prolog                                                  |
//+------------------------------------------------------------------+
bool CEasyXml::skipProlog(string &pText,int &pPos)
  {
//--- skip xml declaration
   if(StringCompare(EASYXML_PROLOG_OPEN,StringSubstr(pText,pPos,StringLen(EASYXML_PROLOG_OPEN)))==0)
     {
      int iClose=StringFind(pText,EASYXML_PROLOG_CLOSE,pPos+StringLen(EASYXML_PROLOG_OPEN));

      if(blDebug) Print("### Prolog ###    ",StringSubstr(pText,pPos,(iClose-pPos)+StringLen(EASYXML_PROLOG_CLOSE)));

      if(iClose>0)
        {
         pPos=iClose+StringLen(EASYXML_PROLOG_CLOSE);
           } else {
         Err=EASYXML_INVALID_PROLOG;
         return(false);
        }
     }
//--- skip stylesheet declarations
   if(StringCompare(EASYXML_STYLESHEET_OPEN,StringSubstr(pText,pPos,StringLen(EASYXML_STYLESHEET_OPEN)))==0)
     {
      int iClose=StringFind(pText,EASYXML_STYLESHEET_CLOSE,pPos+StringLen(EASYXML_STYLESHEET_OPEN));

      if(blDebug) Print("### Prolog ###    ",StringSubstr(pText,pPos,(iClose-pPos)+StringLen(EASYXML_STYLESHEET_CLOSE)));

      if(iClose>0)
        {
         pPos=iClose+StringLen(EASYXML_STYLESHEET_CLOSE);
           } else {
         Err=EASYXML_INVALID_PROLOG;
         return(false);
        }
     }
//--- skip comments
   if(!skipWhitespaceAndComments(pText,pPos,"")) return(false);

//--- skip doctype
   if(StringCompare(EASYXML_DOCTYPE_OPEN,StringSubstr(pText,pPos,StringLen(EASYXML_DOCTYPE_OPEN)))==0)
     {
      int iClose=StringFind(pText,EASYXML_DOCTYPE_CLOSE,pPos+StringLen(EASYXML_DOCTYPE_OPEN));

      if(blDebug) Print("### DOCTYPE ###    ",StringSubstr(pText,pPos,(iClose-pPos)+StringLen(EASYXML_DOCTYPE_CLOSE)));

      if(iClose>0)
        {
         pPos=iClose+StringLen(EASYXML_DOCTYPE_CLOSE);
           } else {
         Err=EASYXML_INVALID_DOCTYPE;
         return(false);
        }
     }

//--- skip comments
   if(!skipWhitespaceAndComments(pText,pPos,"")) return(false);

   return(true);
  }
```

Despite this problem, take nothing away from liquinaut, easyxml.mqh is an excellent tool.

**2.4. Expert Advisor code**

Now that all the libraries needed for the application have been described it is time to bring these components together to define the CRssReader class.

Please note that the RSS Reader Expert Advisor code will begin with the definition of the CRssReader class.

```
//+------------------------------------------------------------------+
//|                                                    RssReader.mq5 |
//|                                                          Ufranco |
//|                                              https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Ufranco"
#property link      "https://www.mql5.com"
#property version   "1.00"
#include <Controls\Dialog.mqh>
#include <Controls\Edit.mqh>
#include <Controls\Button.mqh>
#include <TitleArea.mqh>
#include <TextArea.mqh>
#include <ListViewArea.mqh>
#include <easyxml.mqh>
//+------------------------------------------------------------------+
//| defines                                                          |
//+------------------------------------------------------------------+
//--- indents and gaps
#define INDENT_LEFT                         (11)      // indent from left (with allowance for border width)
#define INDENT_TOP                          (11)      // indent from top (with allowance for border width)
#define INDENT_RIGHT                        (11)      // indent from right (with allowance for border width)
#define INDENT_BOTTOM                       (11)      // indent from bottom (with allowance for border width)
#define CONTROLS_GAP_X                      (5)       // gap by X coordinate
#define CONTROLS_GAP_Y                      (5)       // gap by Y coordinate

#define EDIT_HEIGHT                         (20)      // size by Y coordinate
#define BUTTON_WIDTH                        (150)     // size by X coordinate
#define BUTTON_HEIGHT                       (20)      // size by Y coordinate
#define TEXTAREA_HEIGHT                     (131)     // size by Y coordinate
#define LIST_HEIGHT                         (93)      // size by Y coordinate
```

We begin by including the necessary files. The define directives are used to set the physical parameters of the controls in pixels.

The INDENT\_LEFT, INDENT\_RIGHT, INDENT\_TOP and INDENT\_DOWN set the distance between a control and the application dialogue's edge.

- CONTROLS\_GAP\_Y is the vertical distance between two controls;
- EDIT\_HEIGHT sets the height of the Edit control which makes up the input area;
- BUTTON\_WIDTH and BUTTON\_HEIGHT define the width and height of all the button controls;
- TEXTAREA\_HEIGHT is the height of the text area section;
- LIST\_HEIGHT sets the height of the list view control.


After the defines we begin the definition of the CRssReader class.

```
//+------------------------------------------------------------------+
//| Class CRssReader                                                 |
//| Usage: main class for the RSS application                        |
//+------------------------------------------------------------------+
class CRssReader : public CAppDialog
  {
private:
   int               m_shift;                    // index of first item tag
   string            m_rssurl;                   // copy of web address of last feed
   string            m_textareaoutput[];         // array of strings prepared for output to the text area panel
   string            m_titleareaoutput[];        // array of strings prepared for output to title area panel
   CButton           m_button1;                  // the button object
   CButton           m_button2;                  // the button object
   CEdit             m_edit;                     // input panel
   CTitleArea        m_titleview;                // the display field object
   CListViewArea     m_listview;                 // the list object
   CTextArea         m_textview;                 // text area object
   CEasyXml          m_xmldocument;              // xml document object
   CEasyXmlNode     *RssNode;                    // root node object
   CEasyXmlNode     *ChannelNode;                // channel node object
   CEasyXmlNode     *ChannelChildNodes[];        // array of channel child node objects
```

As previously mentioned CRssReader inherits from the [CAppDialog](https://www.mql5.com/en/docs/standardlibrary/controls/cappdialog) class.

The class has several private properties given below:

- m\_shift - this integer type variable stores the index of the first item node, in the ChannelChildnodes array;
- m\_rssurl - is a string value which keeps a copy of the last URL that was input;
- m\_textareaoutput\[\] -is an array of strings, each element corresponds to a line of text with a certain number of characters;
- m\_titleareaoutput\[\] - this also an array of strings serves the same purpose as the previous string array;
- m\_button1 and m\_button2 are objects of the [CButton](https://www.mql5.com/en/docs/standardlibrary/controls/cbutton) type;
- m\_listview is an object representing a list control;
- m\_edit property a [CEdit](https://www.mql5.com/en/docs/standardlibrary/controls/cedit) object and implements the input area;
- m\_titleview is an object for title area display field;
- m\_textview - the object for the text area section;
- m\_xmldocument is an xml document object;
- RssNode is the root Node object;
- ChannelNode is the object of the channel node;
- ChannelChildNodes array is a set of pointers to the direct descendants of the Channel tag.

Our class will only have two publicly exposed methods.

```
public:
                     CRssReader(void);
                    ~CRssReader(void);
   //--- create
   virtual bool      Create(const long chart,const string name,const int subwin,const int x1,const int y1,const int x2,const int y2);
   //--- chart event handler
   virtual bool      OnEvent(const int id,const long &lparam,const double &dparam,const string &sparam);
```

The first method Create() sets the size and initial position of the application dialogue.

It also initializes all the controls of the RSS Reader app (remember that our class inherits from the [CAppDialog](https://www.mql5.com/en/docs/standardlibrary/controls/cappdialog) class and therefore public methods of the parent class and its ancestors can be called by instances of CRssReader).

```
//+------------------------------------------------------------------+
//| Create                                                           |
//+------------------------------------------------------------------+
bool CRssReader::Create(const long chart,const string name,const int subwin,const int x1,const int y1,const int x2,const int y2)
  {
   if(!CAppDialog::Create(chart,name,subwin,x1,y1,x2,y2))
      return(false);
//--- create dependent controls
   if(!CreateEdit())
      return(false);
   if(!CreateButton1())
      return(false);
   if(!CreateButton2())
      return(false);
   if(!CreateTitleView())
      return(false);
   if(!CreateListView())
      return(false);
   if(!CreateTextView())
      return(false);
//--- succeed
   return(true);
  }
```

The second, is the OnEvent() method, the function enables interactivity by assigning specific events to a corresponding control and a handler function.

```
//+------------------------------------------------------------------+
//| Event Handling                                                   |
//+------------------------------------------------------------------+
EVENT_MAP_BEGIN(CRssReader)
ON_EVENT(ON_CHANGE,m_listview,OnChangeListView)
ON_EVENT(ON_END_EDIT,m_edit,OnObjectEdit)
ON_EVENT(ON_CLICK,m_button1,OnClickButton1)
ON_EVENT(ON_CLICK,m_button2,OnClickButton2)
EVENT_MAP_END(CAppDialog)
```

**2.5. Methods for initialization of controls**

The CreateEdit(), CreateButton1() ,CreateButton2(), CreateTitleView(), CreateListView() and CreateTextView() protected methods are called by the main Create() function for initializing a corresponding control.

```
protected:
   // --- creating controls
   bool              CreateEdit(void);
   bool              CreateButton1(void);
   bool              CreateButton2(void);
   bool              CreateTitleView(void);
   bool              CreateListView(void);
   bool              CreateTextView(void);
```

It is in each of these functions that the size, position and properties (i.e font, font size, color, border color, border type) of a control are set.

```
//+------------------------------------------------------------------+
//| Create the display field                                         |
//+------------------------------------------------------------------+
bool CRssReader::CreateEdit(void)
  {
//--- coordinates
   int x1=INDENT_LEFT;
   int y1=INDENT_TOP;
   int x2=ClientAreaWidth()-INDENT_RIGHT;
   int y2=y1+EDIT_HEIGHT;
//--- create
   if(!m_edit.Create(m_chart_id,m_name+"Edit",m_subwin,x1,y1,x2,y2))
      return(false);
   if(!m_edit.Text("Please enter the web address of an Rss feed"))
      return(false);
   if(!m_edit.ReadOnly(false))
      return(false);
   if(!Add(m_edit))
      return(false);
//--- succeed
   return(true);
  }
//+------------------------------------------------------------------+
//| Create button 1                                                  |
//+------------------------------------------------------------------+
bool CRssReader::CreateButton1(void)
  {
//--- coordinates
   int x1=INDENT_LEFT;
   int y1=INDENT_TOP+(EDIT_HEIGHT+CONTROLS_GAP_Y);
   int x2=x1+BUTTON_WIDTH;
   int y2=y1+BUTTON_HEIGHT;
//--- create
   if(!m_button1.Create(m_chart_id,m_name+"Button1",m_subwin,x1,y1,x2,y2))
      return(false);
   if(!m_button1.Text("Reset"))
      return(false);
   if(!m_button1.Font("Comic Sans MS"))
      return(false);
   if(!m_button1.FontSize(8))
      return(false);
   if(!m_button1.Color(clrWhite))
      return(false);
   if(!m_button1.ColorBackground(clrBlack))
      return(false);
   if(!m_button1.ColorBorder(clrBlack))
      return(false);
   if(!m_button1.Pressed(true))
      return(false);
   if(!Add(m_button1))
      return(false);
//--- succeed
   return(true);
  }
//+------------------------------------------------------------------+
//| Create button 2                                                  |
//+------------------------------------------------------------------+
bool CRssReader::CreateButton2(void)
  {
//--- coordinates
   int x1=(ClientAreaWidth()-INDENT_RIGHT)-BUTTON_WIDTH;
   int y1=INDENT_TOP+(EDIT_HEIGHT+CONTROLS_GAP_Y);
   int x2=ClientAreaWidth()-INDENT_RIGHT;
   int y2=y1+BUTTON_HEIGHT;
//--- create
   if(!m_button2.Create(m_chart_id,m_name+"Button2",m_subwin,x1,y1,x2,y2))
      return(false);
   if(!m_button2.Text("Update current feed"))
      return(false);
   if(!m_button2.Font("Comic Sans MS"))
      return(false);
   if(!m_button2.FontSize(8))
      return(false);
   if(!m_button2.Color(clrWhite))
      return(false);
   if(!m_button2.ColorBackground(clrBlack))
      return(false);
   if(!m_button2.ColorBorder(clrBlack))
      return(false);
   if(!m_button2.Pressed(true))
      return(false);
   if(!Add(m_button2))
      return(false);
//--- succeed
   return(true);
  }
//+------------------------------------------------------------------+
//| Create the display field                                         |
//+------------------------------------------------------------------+
bool CRssReader::CreateTitleView(void)
  {
//--- coordinates
   int x1=INDENT_LEFT;
   int y1=INDENT_TOP+(EDIT_HEIGHT+CONTROLS_GAP_Y)+BUTTON_HEIGHT+CONTROLS_GAP_Y;
   int x2=ClientAreaWidth()-INDENT_RIGHT;
   int y2=y1+(EDIT_HEIGHT*2);
   m_titleview.Current();
//--- create
   if(!m_titleview.Create(m_chart_id,m_name+"TitleView",m_subwin,x1,y1,x2,y2))
     {
      Print("error creating title view");
      return(false);
     }
   else
     {
      for(int i=0;i<2;i++)
        {
         m_titleview.AddItem(" ");
        }
     }
   if(!Add(m_titleview))
     {
      Print("error adding title view");
      return(false);
     }
//--- succeed
   return(true);
  }
//+------------------------------------------------------------------+
//| Create the "ListView" element                                    |
//+------------------------------------------------------------------+
bool CRssReader::CreateListView(void)
  {

//--- coordinates
   int x1=INDENT_LEFT;
   int y1=INDENT_TOP+((EDIT_HEIGHT+CONTROLS_GAP_Y)*2)+20+TEXTAREA_HEIGHT+CONTROLS_GAP_Y+BUTTON_HEIGHT+CONTROLS_GAP_Y;
   int x2=ClientAreaWidth()-INDENT_RIGHT;
   int y2=y1+LIST_HEIGHT;
//--- create
   if(!m_listview.Create(m_chart_id,m_name+"ListView",m_subwin,x1,y1,x2,y2))
      return(false);
   if(!Add(m_listview))
      return(false);
//--- fill out with strings
   for(int i=0;i<20;i++)
      if(!m_listview.AddItem(" "))
         return(false);
//--- succeed
   return(true);
  }
//+------------------------------------------------------------------+
//| Create the display field                                         |
//+------------------------------------------------------------------+
bool CRssReader::CreateTextView(void)
  {
//--- coordinates
   int x1=INDENT_LEFT;
   int y1=INDENT_TOP+((EDIT_HEIGHT+CONTROLS_GAP_Y)*2)+20+BUTTON_HEIGHT+CONTROLS_GAP_Y;
   int x2=ClientAreaWidth()-INDENT_RIGHT;
   int y2=y1+TEXTAREA_HEIGHT;
   m_textview.Current();
//--- create
   if(!m_textview.Create(m_chart_id,m_name+"TextArea",m_subwin,x1,y1,x2,y2))
     {
      Print("error creating text area view");
      return(false);
     }
   else
     {
      for(int i=0;i<1;i++)
        {
         m_textview.AddItem(" ");
        }
      m_textview.VScrolled(true);
      ChartRedraw();
     }
   if(!Add(m_textview))
     {
      Print("error adding text area view");
      return(false);
     }
//----success
   return(true);
  }
```

**2.6. Methods for RSS document processing**

```
// --- rss document processing
   bool              LoadDocument(string filename);
   int               ItemNodesTotal(void);
   void              FreeDocumentTree(void);
```

**2.6.1. LoadDocument()**

This function has a couple of important roles to play. The main one being the processing of web requests. The loadXmlFromUrlWebReq() is called to download the RSS file.

If this is successfully completed, the function moves on to its second task of initializing the pointers RssNode, ChannelNode and also filling the array ChannelChildnodes. It is here that m\_rssurl and m\_shift properties are set. Once all this is done, the function returns true.

If the RSS file cannot be downloaded, the title area, list view area and text area sections are cleared of any text and a status message is displayed on the title bar. This is followed by the output of an error message in the text area section. Then the function returns false.

```
//+------------------------------------------------------------------+
//|   load document                                                  |
//+------------------------------------------------------------------+
bool CRssReader::LoadDocument(string filename)
  {
   if(!m_xmldocument.loadXmlFromUrlWebReq(filename))
     {
      m_textview.ItemsClear();
      m_listview.ItemsClear();
      m_titleview.ItemsClear();
      CDialog::Caption("Failed to load Feed");
      if(!m_textview.AddItem(m_xmldocument.GetErrorMsg()))
         Print("error displaying error message");
      return(false);
     }
   else
     {
      m_rssurl=filename;
      RssNode=m_xmldocument.getDocumentRoot();
      ChannelNode=RssNode.FirstChild();
      if(CheckPointer(RssNode)==POINTER_INVALID || CheckPointer(ChannelNode)==POINTER_INVALID)
         return(false);
     }
   ArrayResize(ChannelChildNodes,ChannelNode.Children().Total());
   for(int i=0;i<ChannelNode.Children().Total();i++)
     {
      ChannelChildNodes[i]=ChannelNode.Children().At(i);
     }
   m_shift=ChannelNode.Children().Total()-ItemNodesTotal();
   return(true);
  }
```

**2.6.2. ItemNodesTotal()**

This helper function is used in the LoadDocument() method. It returns an integer value which is the number of item nodes that are descendants of the channel tag.

If there are no item nodes, the document will be an invalid RSS document and the function will return 0.

```
//+------------------------------------------------------------------+
//| function counts the number of item tags in document              |
//+------------------------------------------------------------------+
int CRssReader::ItemNodesTotal(void)
  {
   int t=0;
   for(int i=0;i<ChannelNode.Children().Total();i++)
     {
      if(ChannelChildNodes[i].getName()=="item")
        {
         t++;
        }
      else continue;
     }
   return(t);
  }
```

**2.6.3. FreeDocumentTree()**

This function resets all the CEasyXmlNode pointers.

First the elements of the ChannelChildnodes array are deleted by calling the Shutdown() method of the [CArrayObj](https://www.mql5.com/en/docs/standardlibrary/datastructures/carrayobj) class. The array is then freed with a single call of ArrayFree().

Next the pointer to the channel node is deleted and easyxml parser's document tree is cleared. These actions result in RssNode and ChannelNode pointers becoming bad pointers, which is why they are both assigned the NULL value.

```
//+------------------------------------------------------------------+
//| free document tree and reset pointer values                      |
//+------------------------------------------------------------------+
void CRssReader::FreeDocumentTree(void)
  {
   ChannelNode.Children().Shutdown();
   ArrayFree(ChannelChildNodes);
   RssNode.Children().Shutdown();
   m_xmldocument.Clear();
   m_shift=0;
   RssNode=NULL;
   ChannelNode=NULL;
  }
```

**2.7. Methods for extracting infomation from the document tree**

These functions are for getting text from an RSS document.

```
//--- getters
   string            getChannelTitle(void);
   string            getTitle(CEasyXmlNode *Node);
   string            getDescription(CEasyXmlNode *Node);
   string            getDate(CEasyXmlNode *Node);
```

**2.7.1. getChannelTitle()**

This function retrieves the current channel title of the RSS document.

It starts off by checking the validity of the channel node pointer. If the pointer is valid, it loops through all the direct descendants of the channel node looking for the title tag.

The for loop uses the m\_shift property to limit the number of channel node descendants to search from. If the function is unsuccessful it returns NULL.

```
//+------------------------------------------------------------------+
//| get channel title                                                |
//+------------------------------------------------------------------+
string CRssReader::getChannelTitle(void)
  {
   string ret=NULL;
   if(!CheckPointer(ChannelNode)==POINTER_INVALID)
     {
      for(int i=0;i<m_shift;i++)
        {
         if(ChannelChildNodes[i].getName()=="title")
           {
            ret=ChannelChildNodes[i].getValue();
            break;
           }
         else continue;
        }
     }
//---return value
   return(ret);
  }
```

**2.7.2. getTitle()**

The function takes as input a pointer to an item tag, and traverses the descentants of that tag looking for a title tag and returns its value.

The getDescription() and getDate() functions follow the same format and work similarly. A successful call of the function returns a string value, otherwise NULL is returned as output.

```
//+------------------------------------------------------------------+
//| display title                                                    |
//+------------------------------------------------------------------+
string CRssReader::getTitle(CEasyXmlNode *Node)
  {
   int k=Node.Children().Total();
   string n=NULL;
   for(int i=0;i<k;i++)
     {
      CEasyXmlNode*subNode=Node.Children().At(i);
      if(subNode.getName()=="title")
        {
         n=subNode.getValue();
         break;
        }
      else continue;
     }
   return(n);
  }
//+------------------------------------------------------------------+
//| display description                                              |
//+------------------------------------------------------------------+
string CRssReader::getDescription(CEasyXmlNode *Node)
  {
   int k=Node.Children().Total();
   string n=NULL;
   for(int i=0;i<k;i++)
     {
      CEasyXmlNode*subNode=Node.Children().At(i);
      if(subNode.getName()=="description")
        {
         n=subNode.getValue();
         break;
        }
      else continue;
     }
   return(n);
  }
//+------------------------------------------------------------------+
//| display date                                                     |
//+------------------------------------------------------------------+
string CRssReader::getDate(CEasyXmlNode *Node)
  {
   int k=Node.Children().Total();
   string n=NULL;
   for(int i=0;i<k;i++)
     {
      CEasyXmlNode*subNode=Node.Children().At(i);
      if(subNode.getName()=="pubDate")
        {
         n=subNode.getValue();
         break;
        }
      else continue;
     }
   return(n);
  }
```

**2.8. Methods for formatting text**

These functions are for preparing text for output as text objects in order to overcome some of the limitations that text objects have.

```
 //--- text formating
   bool              FormatString(string v,string &array[],int n);
   string            removeTags(string _string);
   string            removeSpecialCharacters(string s_tring);
   int               tagPosition(string _string,int w);
```

**2.8.1. FormatString()**

This is the main function that prepares text extracted from an RSS document for output to the application.

Basically it takes a string input value, and splits the text into lines of "n" characters. "n" being an integer value of the number of characters in a single line of text. After every "n" characters in the text, the code searches for a suitable place to insert a new line feed character. Then the entire string value is processed and the new line feed characters are inserted into the original text.

The [StringSplit()](https://www.mql5.com/en/docs/strings/stringsplit) function is used to create an array of strings, each no more that "n" characters long. The function returns a boolean value and also an array of string values that are ready for output.

```
//+------------------------------------------------------------------+
//| format string for output to text area panel                      |
//+------------------------------------------------------------------+
bool CRssReader::FormatString(string v,string &array[],int n)
  {
   ushort ch[],space,fullstop,comma,semicolon,newlinefeed;
   string _s,_k;
   space=StringGetCharacter(" ",0);
   fullstop=StringGetCharacter(".",0);
   comma=StringGetCharacter(",",0);
   semicolon=StringGetCharacter(";",0);
   newlinefeed=StringGetCharacter("\n",0);
   _k=removeTags(v);
   _s=removeSpecialCharacters(_k);
   int p=StringLen(_s);
   ArrayResize(ch,p+1);
   int d=StringToShortArray(_s,ch,0,-1);
   for(int i=1;i<d;i++)
     {
      int t=i%n;
      if(!t== 0)continue;
      else
        {
         if(ch[(i/n)*n]==fullstop || ch[(i/n)*n]==semicolon || ch[(i/n)*n]==comma)
           {
            ArrayFill(ch,((i/n)*n)+1,1,newlinefeed);
           }
         else
           {
            for(int k=i;k>=0;k--)
              {
               if(ch[k]==space)
                 {
                  ArrayFill(ch,k,1,newlinefeed);
                  break;
                 }
               else continue;
              }
           }
        }
     }
   _s=ShortArrayToString(ch,0,-1);
   int s=StringSplit(_s,newlinefeed,array);
   if(!s>0)
     {return(false);}
// success
   return(true);
  }
```

**2.8.2. removeTags()**

This function became a necessity after I noticed that a good number of RSS documents contain HTML tags within the XML nodes.

Some RSS documents are published in this manner, as many RSS aggregating applications work in the browser.

The function takes a string value and searches for tags within the text. If any tags are found the function loops through every character of the text and stores the position of each opening and closing tag character in the 2 dimensional array a\[\]\[\]. This array is used to extract the text between the tags and the extracted string is returned. If no tags are found, the input string is returned as is.

```
//+------------------------------------------------------------------+
//| remove tags                                                      |
//+------------------------------------------------------------------+
string CRssReader::removeTags(string _string)
  {
   string now=NULL;
   if(StringFind(_string,"<",0)>-1)
     {
      int v=0,a[][2];
      ArrayResize(a,2024);
      for(int i=0;i<StringLen(_string);i++)
        {
         int t=tagPosition(_string,i);
         if(t>0)
           {
            v++;
            a[v-1][0]=i;
            a[v-1][1]=t;
           }
         else continue;
        }
      ArrayResize(a,v);
      for(int i=0;i<v-1;i++)
        {
         now+=StringSubstr(_string,(a[i][1]+1),(a[i+1][0]-(a[i][1]+1)));
        }
     }
   else
     {
      now=_string;
     }
   return(now);
  }
```

A partial example of such a document is shown below.

```
<item>
    <title>GIGABYTE X99-Gaming G1 WIFI Motherboard Review</title>
    <author>Ian Cutress</author>
    <description><![CDATA[ <p>The gaming motherboard range from a manufacturer is one with a lot of focus in terms of design and function due to the increase in gaming related PC sales. On the Haswell-E side of gaming, GIGABYTE is putting forward the X99-Gaming G1 WIFI at the top of its stack, and this is what we are reviewing today.&nbsp;</p>\
<p align="center"><a href='http://dynamic1.anandtech.com/www/delivery/ck.php?n=a1f2f01f&amp;cb=582254849' target='_blank'><img src='http://dynamic1.anandtech.com/www/delivery/avw.php?zoneid=24&amp;cb=582254849&amp;n=a1f2f01f' border='0' alt='' /></a><img src="http://toptenreviews.122.2o7.net/b/ss/tmn-test/1/H.27.3--NS/0" height="1" width="1" border="0" alt="" /></p>]]></description>
    <link>http://www.anandtech.com/show/8788/gigabyte-x99-gaming-g1-wifi-motherboard-review</link>
        <pubDate>Thu, 18 Dec 2014 10:00:00 EDT</pubDate>
        <guid isPermaLink="false">tag:www.anandtech.com,8788:news</guid>
        <category><![CDATA[ Motherboards]]></category>
</item>
```

**2.8.3. removeSpecialCharacters()**

This function simply replaces certain string constants with the correct character.

For example as ampersand character in some xml documents can be represented as "&amp". This function uses the built-in [StringReplace()](https://www.mql5.com/en/docs/strings/stringreplace) function to replace these kinds of occurances.

```
//+------------------------------------------------------------------+
//| remove special characters                                        |
//+------------------------------------------------------------------+
string CRssReader::removeSpecialCharacters(string s_tring)
  {
   string n=s_tring;
   StringReplace(n,"&amp;","&");
   StringReplace(n,"&#39;","'");
   StringReplace(n,"&nbsp;"," ");
   StringReplace(n,"&ldquo;","\'");
   StringReplace(n,"&rdquo;","\'");
   StringReplace(n,"&quot;","\"");
   StringReplace(n,"&ndash;","-");
   StringReplace(n,"&rsquo;","'");
   StringReplace(n,"&gt;","");
   return(n);
  }
```

**2.8.4. tagPosition()**

This is a helper function called in the removeTags() function. It takes as input a string and an integer value.

The input integer value represents the position of a character in the string, from which the function will begin to search for an opening tag character ie, "<".If an opening tag is found then the function begins to search for a closing tag and returns as output the postion of the corresponding closing tag character ">". If there are no tags found the function returns -1.

```
//+------------------------------------------------------------------+
//| tag positions                                                    |
//+------------------------------------------------------------------+
int CRssReader::tagPosition(string _string,int w)
  {
   int iClose=-1;
   if(StringCompare("<",StringSubstr(_string,w,StringLen("<")))==0)
     {
      iClose=StringFind(_string,">",w+StringLen("<"));
     }

   return(iClose);
  }
```

**2.9. Methods for handling events of independent controls**

These functions handle the captured events of a specific control.

```
//--- handlers of the dependent controls events
   void              OnChangeListView(void);
   void              OnObjectEdit(void);
   void              OnClickButton1(void);
   void              OnClickButton2(void);
  };
```

**2.9.1. OnChangeListView()**

This is an event handler function and is called whenever one of the list items in the list view area section of the application are clicked.

The function is responsible for enabling the viewing of the description summary for some content referenced in the RSS document.

The function clears the text area and title area sections of any text, retrieves new data from the document tree and prepares it for output. All this only occurs if the ChannelChildnodes array is not empty.

```
//+------------------------------------------------------------------+
//| Event handler                                                    |
//+------------------------------------------------------------------+
void CRssReader::OnChangeListView(void)
  {
   int a=0,k=0,l=0;
   a=m_listview.Current()+m_shift;
   if(ArraySize(ChannelChildNodes)>a)
     {
      if(m_titleview.ItemsClear())
        {
         if(!FormatString(getTitle(ChannelChildNodes[a]),m_titleareaoutput,55))
           {
            return;
           }
         else
         if(ArraySize(m_titleareaoutput)>0)
           {
            for(l=0;l<ArraySize(m_titleareaoutput);l++)
              {
               m_titleview.AddItem(removeSpecialCharacters(m_titleareaoutput[l]));
              }
           }
        }
      if(m_textview.ItemsClear())
        {
         if(!FormatString(getDescription(ChannelChildNodes[a]),m_textareaoutput,35))
            return;
         else
         if(ArraySize(m_textareaoutput)>0)
           {
            for(k=0;k<ArraySize(m_textareaoutput);k++)
              {
               m_textview.AddItem(m_textareaoutput[k]);
              }
            m_textview.AddItem(" ");
            m_textview.AddItem("Date|"+getDate(ChannelChildNodes[a]));
           }
         else return;
        }
     }
  }
```

**2.9.2. OnObjectEdit()**

The handler function is called whenever a user finishes entering some text into the input area.

The function calls the LoadDocument() method. If the download is successful, the text is cleared from the entire application. Next, the caption is changed and the new content is output to the list view area section.

```
//+------------------------------------------------------------------+
//| Event handler                                                    |
//+------------------------------------------------------------------+
void CRssReader::OnObjectEdit(void)
  {
   string f=m_edit.Text();
   if(StringLen(f)>0)
     {
      if(ArraySize(ChannelChildNodes)<1)
        {
         CDialog::Caption("Loading...");
         if(LoadDocument(f))
           {
            if(!CDialog::Caption(removeSpecialCharacters(getChannelTitle())))
               Print("error changing caption");
            if(m_textview.ItemsClear() && m_listview.ItemsClear() && m_titleview.ItemsClear())
              {
               for(int i=0;i<ItemNodesTotal()-1;i++)
                 {
                  if(!m_listview.AddItem(removeSpecialCharacters(IntegerToString(i+1)+"."+getTitle(ChannelChildNodes[i+m_shift]))))
                    {
                     Print("can not add item to listview area");
                     return;
                    }
                 }
              }
            else
              {
               Print("text area/listview area not cleared");
               return;
              }
           }
         else return;
        }
      else
        {
         FreeDocumentTree();
         CDialog::Caption("Loading new RSS Feed...");
         if(LoadDocument(f))
           {
            if(!CDialog::Caption(removeSpecialCharacters(getChannelTitle())))
               Print("error changing caption");
            if(m_textview.ItemsClear() && m_listview.ItemsClear() && m_titleview.ItemsClear())
              {
               for(int i=0;i<ItemNodesTotal()-1;i++)
                 {
                  if(!m_listview.AddItem(removeSpecialCharacters(IntegerToString(i+1)+"."+getTitle(ChannelChildNodes[i+m_shift]))))
                    {
                     Print("can not add item to listview area");
                     return;
                    }
                 }
              }
            else
              {
               Print("text area/listview area not cleared");
               return;
              }
           }
         else return;
        }
     }
   else return;
  }
```

**2.9.3. OnClickButton1/2()**

These handlers are called whenever a user clicks either the reset or check for feed updates buttons.

Clicking the reset button refreshes the app dialogue to the state it was in when the Expert advisor was first launched.

Clicking the "check for feed update" button causes a recall of the load LoadDocument() method and the RSS feed data will be downloaded, refreshing the list view area section.

```
//+------------------------------------------------------------------+
//| Event handler  refresh the app dialogue                          |
//+------------------------------------------------------------------+
void CRssReader::OnClickButton1(void)
  {
   if(ArraySize(ChannelChildNodes)<1)
     {
      if(!m_edit.Text("Enter the web address of an Rss feed"))
         Print("error changing edit text");
      if(!CDialog::Caption("RSSReader"))
         Print("error changing caption");
      if(m_textview.ItemsClear() && m_listview.ItemsClear() && m_titleview.ItemsClear())
        {
         for(int i=0;i<20;i++)
           {
            if(!m_listview.AddItem(" "))
               Print("error adding to listview");
           }
         m_listview.VScrolled(true);
         for(int i=0;i<1;i++)
           {
            m_textview.AddItem(" ");
           }
         m_textview.VScrolled(true);
         for(int i=0;i<2;i++)
           {
            m_titleview.AddItem(" ");
           }
         return;
        }
     }
   else
     {
      FreeDocumentTree();
      if(!m_edit.Text("Enter the web address of an Rss feed"))
         Print("error changing edit text");
      if(!CDialog::Caption("RSSReader"))
         Print("error changing caption");
      if(m_textview.ItemsClear() && m_listview.ItemsClear() && m_titleview.ItemsClear())
        {
         for(int i=0;i<20;i++)
           {
            if(!m_listview.AddItem(" "))
               Print("error adding to listview");
           }
         m_listview.VScrolled(true);
         for(int i=0;i<1;i++)
           {
            m_textview.AddItem(" ");
           }
         m_textview.VScrolled(true);
         for(int i=0;i<2;i++)
           {
            m_titleview.AddItem(" ");
           }
         return;
        }
     }
  }
//+------------------------------------------------------------------+
//| Event handler  update current feed                               |
//+------------------------------------------------------------------+
void CRssReader::OnClickButton2(void)
  {
   string f=m_rssurl;
   if(ArraySize(ChannelChildNodes)<1)
      return;
   else
     {
      FreeDocumentTree();
      CDialog::Caption("Checking for RSS Feed update...");
      if(LoadDocument(f))
        {
         if(!CDialog::Caption(removeSpecialCharacters(getChannelTitle())))
            Print("error changing caption");
         if(m_textview.ItemsClear() && m_listview.ItemsClear() && m_titleview.ItemsClear())
           {
            for(int i=0;i<ItemNodesTotal()-1;i++)
              {
               if(!m_listview.AddItem(removeSpecialCharacters(IntegerToString(i+1)+"."+getTitle(ChannelChildNodes[i+m_shift]))))
                 {
                  Print("can not add item to listview area");
                  return;
                 }
              }
           }
         else
           {
            Print("text area/listview area not cleared");
            return;
           }
        }
      else return;
     }
  }
```

That concludes the definition of the CRssReader class.

**2.10. The CRssReader class implementation**

```
//+------------------------------------------------------------------+
//| Class CRssReader                                                 |
//| Usage: main class for the RSS application                        |
//+------------------------------------------------------------------+
class CRssReader : public CAppDialog
  {
private:
   int               m_shift;                   // index of first item tag
   string            m_rssurl;                  // copy of web address of last feed
   string            m_textareaoutput[];        // array of strings prepared for output to the text area panel
   string            m_titleareaoutput[];       // array of strings prepared for output to title area panel
   CButton           m_button1;                 // the button object
   CButton           m_button2;                 // the button object
   CEdit             m_edit;                    // input panel
   CTitleArea        m_titleview;               // the display field object
   CListViewArea     m_listview;                // the list object
   CTextArea         m_textview;                // text area object
   CEasyXml          m_xmldocument;             // xml document object
   CEasyXmlNode     *RssNode;                   // root node object
   CEasyXmlNode     *ChannelNode;               // channel node object
   CEasyXmlNode     *ChannelChildNodes[];       // array of channel child node objects

public:
                     CRssReader(void);
                    ~CRssReader(void);
   //--- create
   virtual bool      Create(const long chart,const string name,const int subwin,const int x1,const int y1,const int x2,const int y2);
   //--- chart event handler
   virtual bool      OnEvent(const int id,const long &lparam,const double &dparam,const string &sparam);

protected:
   // --- creating controls
   bool              CreateEdit(void);
   bool              CreateButton1(void);
   bool              CreateButton2(void);
   bool              CreateTitleView(void);
   bool              CreateListView(void);
   bool              CreateTextView(void);
   // --- rss document processing
   bool              LoadDocument(string filename);
   int               ItemNodesTotal(void);
   void              FreeDocumentTree(void);
   //--- getters
   string            getChannelTitle(void);
   string            getTitle(CEasyXmlNode *Node);
   string            getDescription(CEasyXmlNode *Node);
   string            getDate(CEasyXmlNode *Node);
   //--- text formating
   bool              FormatString(string v,string &array[],int n);
   string            removeTags(string _string);
   string            removeSpecialCharacters(string s_tring);
   int               tagPosition(string _string,int w);
   //--- handlers of the dependent controls events
   void              OnChangeListView(void);
   void              OnObjectEdit(void);
   void              OnClickButton1(void);
   void              OnClickButton2(void);
  };
//+------------------------------------------------------------------+
//| Event Handling                                                   |
//+------------------------------------------------------------------+
EVENT_MAP_BEGIN(CRssReader)
ON_EVENT(ON_CHANGE,m_listview,OnChangeListView)
ON_EVENT(ON_END_EDIT,m_edit,OnObjectEdit)
ON_EVENT(ON_CLICK,m_button1,OnClickButton1)
ON_EVENT(ON_CLICK,m_button2,OnClickButton2)
EVENT_MAP_END(CAppDialog)
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CRssReader::CRssReader(void)
  {

  }
//+------------------------------------------------------------------+
//| Destructor                                                       |
//+------------------------------------------------------------------+
CRssReader::~CRssReader(void)
  {
  }
//+------------------------------------------------------------------+
//| Create                                                           |
//+------------------------------------------------------------------+
bool CRssReader::Create(const long chart,const string name,const int subwin,const int x1,const int y1,const int x2,const int y2)
  {
   if(!CAppDialog::Create(chart,name,subwin,x1,y1,x2,y2))
      return(false);
//--- create dependent controls
   if(!CreateEdit())
      return(false);
   if(!CreateButton1())
      return(false);
   if(!CreateButton2())
      return(false);
   if(!CreateTitleView())
      return(false);
   if(!CreateListView())
      return(false);
   if(!CreateTextView())
      return(false);
//--- succeed
   return(true);
  }
//+------------------------------------------------------------------+
//| Create the display field                                         |
//+------------------------------------------------------------------+
bool CRssReader::CreateEdit(void)
  {
//--- coordinates
   int x1=INDENT_LEFT;
   int y1=INDENT_TOP;
   int x2=ClientAreaWidth()-INDENT_RIGHT;
   int y2=y1+EDIT_HEIGHT;
//--- create
   if(!m_edit.Create(m_chart_id,m_name+"Edit",m_subwin,x1,y1,x2,y2))
      return(false);
   if(!m_edit.Text("Please enter the web address of an Rss feed"))
      return(false);
   if(!m_edit.ReadOnly(false))
      return(false);
   if(!Add(m_edit))
      return(false);
//--- succeed
   return(true);
  }
//+------------------------------------------------------------------+
//| Create button 1                                                  |
//+------------------------------------------------------------------+
bool CRssReader::CreateButton1(void)
  {
//--- coordinates
   int x1=INDENT_LEFT;
   int y1=INDENT_TOP+(EDIT_HEIGHT+CONTROLS_GAP_Y);
   int x2=x1+BUTTON_WIDTH;
   int y2=y1+BUTTON_HEIGHT;
//--- create
   if(!m_button1.Create(m_chart_id,m_name+"Button1",m_subwin,x1,y1,x2,y2))
      return(false);
   if(!m_button1.Text("Reset"))
      return(false);
   if(!m_button1.Font("Comic Sans MS"))
      return(false);
   if(!m_button1.FontSize(8))
      return(false);
   if(!m_button1.Color(clrWhite))
      return(false);
   if(!m_button1.ColorBackground(clrBlack))
      return(false);
   if(!m_button1.ColorBorder(clrBlack))
      return(false);
   if(!m_button1.Pressed(true))
      return(false);
   if(!Add(m_button1))
      return(false);
//--- succeed
   return(true);
  }
//+------------------------------------------------------------------+
//| Create button 2                                                  |
//+------------------------------------------------------------------+
bool CRssReader::CreateButton2(void)
  {
//--- coordinates
   int x1=(ClientAreaWidth()-INDENT_RIGHT)-BUTTON_WIDTH;
   int y1=INDENT_TOP+(EDIT_HEIGHT+CONTROLS_GAP_Y);
   int x2=ClientAreaWidth()-INDENT_RIGHT;
   int y2=y1+BUTTON_HEIGHT;
//--- create
   if(!m_button2.Create(m_chart_id,m_name+"Button2",m_subwin,x1,y1,x2,y2))
      return(false);
   if(!m_button2.Text("Update current feed"))
      return(false);
   if(!m_button2.Font("Comic Sans MS"))
      return(false);
   if(!m_button2.FontSize(8))
      return(false);
   if(!m_button2.Color(clrWhite))
      return(false);
   if(!m_button2.ColorBackground(clrBlack))
      return(false);
   if(!m_button2.ColorBorder(clrBlack))
      return(false);
   if(!m_button2.Pressed(true))
      return(false);
   if(!Add(m_button2))
      return(false);
//--- succeed
   return(true);
  }
//+------------------------------------------------------------------+
//| Create the display field                                         |
//+------------------------------------------------------------------+
bool CRssReader::CreateTitleView(void)
  {
//--- coordinates
   int x1=INDENT_LEFT;
   int y1=INDENT_TOP+(EDIT_HEIGHT+CONTROLS_GAP_Y)+BUTTON_HEIGHT+CONTROLS_GAP_Y;
   int x2=ClientAreaWidth()-INDENT_RIGHT;
   int y2=y1+(EDIT_HEIGHT*2);
   m_titleview.Current();
//--- create
   if(!m_titleview.Create(m_chart_id,m_name+"TitleView",m_subwin,x1,y1,x2,y2))
     {
      Print("error creating title view");
      return(false);
     }
   else
     {
      for(int i=0;i<2;i++)
        {
         m_titleview.AddItem(" ");
        }
     }
   if(!Add(m_titleview))
     {
      Print("error adding title view");
      return(false);
     }
//--- succeed
   return(true);
  }
//+------------------------------------------------------------------+
//| Create the "ListView" element                                    |
//+------------------------------------------------------------------+
bool CRssReader::CreateListView(void)
  {

//--- coordinates
   int x1=INDENT_LEFT;
   int y1=INDENT_TOP+((EDIT_HEIGHT+CONTROLS_GAP_Y)*2)+20+TEXTAREA_HEIGHT+CONTROLS_GAP_Y+BUTTON_HEIGHT+CONTROLS_GAP_Y;
   int x2=ClientAreaWidth()-INDENT_RIGHT;
   int y2=y1+LIST_HEIGHT;
//--- create
   if(!m_listview.Create(m_chart_id,m_name+"ListView",m_subwin,x1,y1,x2,y2))
      return(false);
   if(!Add(m_listview))
      return(false);
//--- fill out with strings
   for(int i=0;i<20;i++)
      if(!m_listview.AddItem(" "))
         return(false);
//--- succeed
   return(true);
  }
//+------------------------------------------------------------------+
//| Create the display field                                         |
//+------------------------------------------------------------------+
bool CRssReader::CreateTextView(void)
  {
//--- coordinates
   int x1=INDENT_LEFT;
   int y1=INDENT_TOP+((EDIT_HEIGHT+CONTROLS_GAP_Y)*2)+20+BUTTON_HEIGHT+CONTROLS_GAP_Y;
   int x2=ClientAreaWidth()-INDENT_RIGHT;
   int y2=y1+TEXTAREA_HEIGHT;
   m_textview.Current();
//--- create
   if(!m_textview.Create(m_chart_id,m_name+"TextArea",m_subwin,x1,y1,x2,y2))
     {
      Print("error creating text area view");
      return(false);
     }
   else
     {
      for(int i=0;i<1;i++)
        {
         m_textview.AddItem(" ");
        }
      m_textview.VScrolled(true);
      ChartRedraw();
     }
   if(!Add(m_textview))
     {
      Print("error adding text area view");
      return(false);
     }
//----success
   return(true);
  }
//+------------------------------------------------------------------+
//|   load document                                                  |
//+------------------------------------------------------------------+
bool CRssReader::LoadDocument(string filename)
  {
   if(!m_xmldocument.loadXmlFromUrlWebReq(filename))
     {
      m_textview.ItemsClear();
      m_listview.ItemsClear();
      m_titleview.ItemsClear();
      CDialog::Caption("Failed to load Feed");
      if(!m_textview.AddItem(m_xmldocument.GetErrorMsg()))
         Print("error displaying error message");
      return(false);
     }
   else
     {
      m_rssurl=filename;
      RssNode=m_xmldocument.getDocumentRoot();
      ChannelNode=RssNode.FirstChild();
      if(CheckPointer(RssNode)==POINTER_INVALID || CheckPointer(ChannelNode)==POINTER_INVALID)
         return(false);
     }
   ArrayResize(ChannelChildNodes,ChannelNode.Children().Total());
   for(int i=0;i<ChannelNode.Children().Total();i++)
     {
      ChannelChildNodes[i]=ChannelNode.Children().At(i);
     }
   m_shift=ChannelNode.Children().Total()-ItemNodesTotal();
   return(true);
  }
//+------------------------------------------------------------------+
//| function counts the number of item tags in document              |
//+------------------------------------------------------------------+
int CRssReader::ItemNodesTotal(void)
  {
   int t=0;
   for(int i=0;i<ChannelNode.Children().Total();i++)
     {
      if(ChannelChildNodes[i].getName()=="item")
        {
         t++;
        }
      else continue;
     }
   return(t);
  }
//+------------------------------------------------------------------+
//| free document tree and reset pointer values                      |
//+------------------------------------------------------------------+
void CRssReader::FreeDocumentTree(void)
  {
   ChannelNode.Children().Shutdown();
   ArrayFree(ChannelChildNodes);
   RssNode.Children().Shutdown();
   m_xmldocument.Clear();
   m_shift=0;
   RssNode=NULL;
   ChannelNode=NULL;
  }
//+------------------------------------------------------------------+
//| get channel title                                                |
//+------------------------------------------------------------------+
string CRssReader::getChannelTitle(void)
  {
   string ret=NULL;
   if(!CheckPointer(ChannelNode)==POINTER_INVALID)
     {
      for(int i=0;i<m_shift;i++)
        {
         if(ChannelChildNodes[i].getName()=="title")
           {
            ret=ChannelChildNodes[i].getValue();
            break;
           }
         else continue;
        }
     }
//---return value
   return(ret);
  }
//+------------------------------------------------------------------+
//| display title                                                    |
//+------------------------------------------------------------------+
string CRssReader::getTitle(CEasyXmlNode *Node)
  {
   int k=Node.Children().Total();
   string n=NULL;
   for(int i=0;i<k;i++)
     {
      CEasyXmlNode*subNode=Node.Children().At(i);
      if(subNode.getName()=="title")
        {
         n=subNode.getValue();
         break;
        }
      else continue;
     }
   return(n);
  }
//+------------------------------------------------------------------+
//| display description                                              |
//+------------------------------------------------------------------+
string CRssReader::getDescription(CEasyXmlNode *Node)
  {
   int k=Node.Children().Total();
   string n=NULL;
   for(int i=0;i<k;i++)
     {
      CEasyXmlNode*subNode=Node.Children().At(i);
      if(subNode.getName()=="description")
        {
         n=subNode.getValue();
         break;
        }
      else continue;
     }
   return(n);
  }
//+------------------------------------------------------------------+
//| display date                                                     |
//+------------------------------------------------------------------+
string CRssReader::getDate(CEasyXmlNode *Node)
  {
   int k=Node.Children().Total();
   string n=NULL;
   for(int i=0;i<k;i++)
     {
      CEasyXmlNode*subNode=Node.Children().At(i);
      if(subNode.getName()=="pubDate")
        {
         n=subNode.getValue();
         break;
        }
      else continue;
     }
   return(n);
  }
//+------------------------------------------------------------------+
//| format string for output to text area panel                      |
//+------------------------------------------------------------------+
bool CRssReader::FormatString(string v,string &array[],int n)
  {
   ushort ch[],space,fullstop,comma,semicolon,newlinefeed;
   string _s,_k;
   space=StringGetCharacter(" ",0);
   fullstop=StringGetCharacter(".",0);
   comma=StringGetCharacter(",",0);
   semicolon=StringGetCharacter(";",0);
   newlinefeed=StringGetCharacter("\n",0);
   _k=removeTags(v);
   _s=removeSpecialCharacters(_k);
   int p=StringLen(_s);
   ArrayResize(ch,p+1);
   int d=StringToShortArray(_s,ch,0,-1);
   for(int i=1;i<d;i++)
     {
      int t=i%n;
      if(!t== 0)continue;
      else
        {
         if(ch[(i/n)*n]==fullstop || ch[(i/n)*n]==semicolon || ch[(i/n)*n]==comma)
           {
            ArrayFill(ch,((i/n)*n)+1,1,newlinefeed);
           }
         else
           {
            for(int k=i;k>=0;k--)
              {
               if(ch[k]==space)
                 {
                  ArrayFill(ch,k,1,newlinefeed);
                  break;
                 }
               else continue;
              }
           }
        }
     }
   _s=ShortArrayToString(ch,0,-1);
   int s=StringSplit(_s,newlinefeed,array);
   if(!s>0)
     {return(false);}
// success
   return(true);
  }
//+------------------------------------------------------------------+
//| remove special characters                                        |
//+------------------------------------------------------------------+
string CRssReader::removeSpecialCharacters(string s_tring)
  {
   string n=s_tring;
   StringReplace(n,"&amp;","&");
   StringReplace(n,"&#39;","'");
   StringReplace(n,"&nbsp;"," ");
   StringReplace(n,"&ldquo;","\'");
   StringReplace(n,"&rdquo;","\'");
   StringReplace(n,"&quot;","\"");
   StringReplace(n,"&ndash;","-");
   StringReplace(n,"&rsquo;","'");
   StringReplace(n,"&gt;","");
   return(n);
  }
//+------------------------------------------------------------------+
//| remove tags                                                      |
//+------------------------------------------------------------------+
string CRssReader::removeTags(string _string)
  {
   string now=NULL;
   if(StringFind(_string,"<",0)>-1)
     {
      int v=0,a[][2];
      ArrayResize(a,2024);
      for(int i=0;i<StringLen(_string);i++)
        {
         int t=tagPosition(_string,i);
         if(t>0)
           {
            v++;
            a[v-1][0]=i;
            a[v-1][1]=t;
           }
         else continue;
        }
      ArrayResize(a,v);
      for(int i=0;i<v-1;i++)
        {
         now+=StringSubstr(_string,(a[i][1]+1),(a[i+1][0]-(a[i][1]+1)));
        }
     }
   else
     {
      now=_string;
     }
   return(now);
  }
//+------------------------------------------------------------------+
//| tag positions                                                    |
//+------------------------------------------------------------------+
int CRssReader::tagPosition(string _string,int w)
  {
   int iClose=-1;
   if(StringCompare("<",StringSubstr(_string,w,StringLen("<")))==0)
     {
      iClose=StringFind(_string,">",w+StringLen("<"));
     }

   return(iClose);
  }
//+------------------------------------------------------------------+
//| Event handler                                                    |
//+------------------------------------------------------------------+
void CRssReader::OnChangeListView(void)
  {
   int a=0,k=0,l=0;
   a=m_listview.Current()+m_shift;
   if(ArraySize(ChannelChildNodes)>a)
     {
      if(m_titleview.ItemsClear())
        {
         if(!FormatString(getTitle(ChannelChildNodes[a]),m_titleareaoutput,55))
           {
            return;
           }
         else
         if(ArraySize(m_titleareaoutput)>0)
           {
            for(l=0;l<ArraySize(m_titleareaoutput);l++)
              {
               m_titleview.AddItem(removeSpecialCharacters(m_titleareaoutput[l]));
              }
           }
        }
      if(m_textview.ItemsClear())
        {
         if(!FormatString(getDescription(ChannelChildNodes[a]),m_textareaoutput,35))
            return;
         else
         if(ArraySize(m_textareaoutput)>0)
           {
            for(k=0;k<ArraySize(m_textareaoutput);k++)
              {
               m_textview.AddItem(m_textareaoutput[k]);
              }
            m_textview.AddItem(" ");
            m_textview.AddItem("Date|"+getDate(ChannelChildNodes[a]));
           }
         else return;
        }
     }
  }
//+------------------------------------------------------------------+
//| Event handler                                                    |
//+------------------------------------------------------------------+
void CRssReader::OnObjectEdit(void)
  {
   string f=m_edit.Text();
   if(StringLen(f)>0)
     {
      if(ArraySize(ChannelChildNodes)<1)
        {
         CDialog::Caption("Loading...");
         if(LoadDocument(f))
           {
            if(!CDialog::Caption(removeSpecialCharacters(getChannelTitle())))
               Print("error changing caption");
            if(m_textview.ItemsClear() && m_listview.ItemsClear() && m_titleview.ItemsClear())
              {
               for(int i=0;i<ItemNodesTotal()-1;i++)
                 {
                  if(!m_listview.AddItem(removeSpecialCharacters(IntegerToString(i+1)+"."+getTitle(ChannelChildNodes[i+m_shift]))))
                    {
                     Print("can not add item to listview area");
                     return;
                    }
                 }
              }
            else
              {
               Print("text area/listview area not cleared");
               return;
              }
           }
         else return;
        }
      else
        {
         FreeDocumentTree();
         CDialog::Caption("Loading new RSS Feed...");
         if(LoadDocument(f))
           {
            if(!CDialog::Caption(removeSpecialCharacters(getChannelTitle())))
               Print("error changing caption");
            if(m_textview.ItemsClear() && m_listview.ItemsClear() && m_titleview.ItemsClear())
              {
               for(int i=0;i<ItemNodesTotal()-1;i++)
                 {
                  if(!m_listview.AddItem(removeSpecialCharacters(IntegerToString(i+1)+"."+getTitle(ChannelChildNodes[i+m_shift]))))
                    {
                     Print("can not add item to listview area");
                     return;
                    }
                 }
              }
            else
              {
               Print("text area/listview area not cleared");
               return;
              }
           }
         else return;
        }
     }
   else return;
  }
//+------------------------------------------------------------------+
//| Event handler  refresh the app dialogue                          |
//+------------------------------------------------------------------+
void CRssReader::OnClickButton1(void)
  {
   if(ArraySize(ChannelChildNodes)<1)
     {
      if(!m_edit.Text("Enter the web address of an Rss feed"))
         Print("error changing edit text");
      if(!CDialog::Caption("RSSReader"))
         Print("error changing caption");
      if(m_textview.ItemsClear() && m_listview.ItemsClear() && m_titleview.ItemsClear())
        {
         for(int i=0;i<20;i++)
           {
            if(!m_listview.AddItem(" "))
               Print("error adding to listview");
           }
         m_listview.VScrolled(true);
         for(int i=0;i<1;i++)
           {
            m_textview.AddItem(" ");
           }
         m_textview.VScrolled(true);
         for(int i=0;i<2;i++)
           {
            m_titleview.AddItem(" ");
           }
         return;
        }
     }
   else
     {
      FreeDocumentTree();
      if(!m_edit.Text("Enter the web address of an Rss feed"))
         Print("error changing edit text");
      if(!CDialog::Caption("RSSReader"))
         Print("error changing caption");
      if(m_textview.ItemsClear() && m_listview.ItemsClear() && m_titleview.ItemsClear())
        {
         for(int i=0;i<20;i++)
           {
            if(!m_listview.AddItem(" "))
               Print("error adding to listview");
           }
         m_listview.VScrolled(true);
         for(int i=0;i<1;i++)
           {
            m_textview.AddItem(" ");
           }
         m_textview.VScrolled(true);
         for(int i=0;i<2;i++)
           {
            m_titleview.AddItem(" ");
           }
         return;
        }
     }
  }
//+------------------------------------------------------------------+
//| Event handler  update current feed                               |
//+------------------------------------------------------------------+
void CRssReader::OnClickButton2(void)
  {
   string f=m_rssurl;
   if(ArraySize(ChannelChildNodes)<1)
      return;
   else
     {
      FreeDocumentTree();
      CDialog::Caption("Checking for RSS Feed update...");
      if(LoadDocument(f))
        {
         if(!CDialog::Caption(removeSpecialCharacters(getChannelTitle())))
            Print("error changing caption");
         if(m_textview.ItemsClear() && m_listview.ItemsClear() && m_titleview.ItemsClear())
           {
            for(int i=0;i<ItemNodesTotal()-1;i++)
              {
               if(!m_listview.AddItem(removeSpecialCharacters(IntegerToString(i+1)+"."+getTitle(ChannelChildNodes[i+m_shift]))))
                 {
                  Print("can not add item to listview area");
                  return;
                 }
              }
           }
         else
           {
            Print("text area/listview area not cleared");
            return;
           }
        }
      else return;
     }
  }
```

It can now be used in the Expert Advisor code.

**2.11. The Expert Advisor code**

The Expert Advisor has no input variables since the application is meant to be entirely interactive.

First we declare a global variable which is an instance of the CRssReader class. In the [OnInit()](https://www.mql5.com/en/docs/basis/function/events#oninit) function we initialize the application dialogue with a call to the main Create() method. If this is successful, the Run() method of an ancestor class is called.

In the [OnDeinit()](https://www.mql5.com/en/docs/basis/function/events#ondeinit) function the Destroy() method of the parent class is called to delete the entire application and remove the Expert Advisor from the chart.

The [OnChartEvent()](https://www.mql5.com/en/docs/basis/function/events#onchartevent) function contains a call to an ancestor method of the CRssReader class, which will handle the processing of all events.

```
//Expert Advisor code begins here
//+------------------------------------------------------------------+
//| Global Variables                                                 |
//+------------------------------------------------------------------+
CRssReader ExtDialog;
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//--- create application dialog
   if(!ExtDialog.Create(0,"RSSReader",0,20,20,518,394))
      return(INIT_FAILED);
//--- run application
   ExtDialog.Run();
//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---
   ExtDialog.Destroy(reason);
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
   ExtDialog.ChartEvent(id,lparam,dparam,sparam);
  }
```

The code then needs to be compiled and the program will be ready for use.

When the RssReader.mq5 Expert Advisor is loaded onto a chart an empty application dialogue appears as follows:

![Fig. 2. ScreenShot of the empty app dialogue of the RssReader Expert Advisor](https://c.mql5.com/2/17/Figure2_RSS_Reader_MetaTrader5.png)

Fig. 2. ScreenShot of the empty app dialogue of the RssReader Expert Advisor

Enter a web address and the RSS content will be loaded into the application dialogue, as depicted by the image below:

![Fig. 3. RssReader EA working in the terminal](https://c.mql5.com/2/17/Figure3_RSS_Reader_MetaTrader5.png)

Fig. 3. RssReader EA working in the terminal

I tested the program with a wide range of RSS feeds. The only problem I observed was related to the display of some unwanted characters, mostly the result of RSS documents containing characters usually found in HTML-documents.

I also noticed that changing the period of a chart whilst the application is running, causes the EA to reinitialize and can result in the application controls not being drawn properly.

I was not able to correct this behaviour, so my advice is to avoid changing the chart period when the RSS Reader program is running.

### Conclusion

We have completed the creation of an entirely interactive RSS Reader application for MetaTrader 5, using [object-oriented](https://www.mql5.com/en/docs/basis/oop) programming techniques.

There are a lot more features that could be added to the application and I sure there are many more ways that the user interface can be arranged. I hope those with possibly better application GUI design skills will improve the application and share their creations.

P.S. Please note that the easyxml.mqh file available for download here is not the same as the one available in the [Code Base](https://www.mql5.com/en/code/1998), it contains modifications already mentioned in the article. All the necessary includes are in the RssReader.zip file.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/1589.zip "Download all attachments in the single ZIP archive")

[easyxml.mqh](https://www.mql5.com/en/articles/download/1589/easyxml.mqh "Download easyxml.mqh")(25.66 KB)

[easyxmlattribute.mqh](https://www.mql5.com/en/articles/download/1589/easyxmlattribute.mqh "Download easyxmlattribute.mqh")(3.03 KB)

[easyxmlerrordescription.mqh](https://www.mql5.com/en/articles/download/1589/easyxmlerrordescription.mqh "Download easyxmlerrordescription.mqh")(3.48 KB)

[easyxmlnode.mqh](https://www.mql5.com/en/articles/download/1589/easyxmlnode.mqh "Download easyxmlnode.mqh")(7.67 KB)

[ListViewArea.mqh](https://www.mql5.com/en/articles/download/1589/listviewarea.mqh "Download ListViewArea.mqh")(19.89 KB)

[TextArea.mqh](https://www.mql5.com/en/articles/download/1589/textarea.mqh "Download TextArea.mqh")(13.47 KB)

[TitleArea.mqh](https://www.mql5.com/en/articles/download/1589/titlearea.mqh "Download TitleArea.mqh")(13.56 KB)

[RssReader.mq5](https://www.mql5.com/en/articles/download/1589/rssreader.mq5 "Download RssReader.mq5")(27.83 KB)

[RssReader.zip](https://www.mql5.com/en/articles/download/1589/rssreader.zip "Download RssReader.zip")(20.49 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/40689)**
(4)


![Heroix](https://c.mql5.com/avatar/2013/7/51D5B71D-6867.jpg)

**[Heroix](https://www.mql5.com/en/users/heroix)**
\|
18 Sep 2016 at 22:49

Hmm. Here's a question. I don't know if anyone has encountered it.

How much RSS feed slows down at news sources?

p.s. let's say here: http: [//www.bls.gov/feed/bls\_latest.rss](https://www.mql5.com/go?link=http://www.bls.gov/feed/bls_latest.rss "http://www.bls.gov/feed/bls_latest.rss").

p.p.s. if you have your own example, it would be interesting to know too.

![Maxim Kuznetsov](https://c.mql5.com/avatar/2016/1/56935A91-AF51.png)

**[Maxim Kuznetsov](https://www.mql5.com/en/users/nektomk)**
\|
18 Sep 2016 at 23:18

just a question:

Where else is RSS used ?

![Heroix](https://c.mql5.com/avatar/2013/7/51D5B71D-6867.jpg)

**[Heroix](https://www.mql5.com/en/users/heroix)**
\|
18 Sep 2016 at 23:48

**Maxim Kuznetsov:**

just a question:

Where else is RSS used ?

In announcements, etc. - in the promotion of new information, in general.

![Andres Moreno Abrego](https://c.mql5.com/avatar/2017/1/588D679D-B74B.jpg)

**[Andres Moreno Abrego](https://www.mql5.com/en/users/andresmorenoabr)**
\|
10 Feb 2017 at 20:11

Hi Francis,

Thank you for the great example on building an RSS reader.

I tried using the reader but when I enter the RSS URL and click on the "update current feed" button, it just shows the "Loading..." message but doesn´t do anything else.

Do you have any clues of why this might be happening?

I'm using this feed:

http://rss.cnn.com/rss/edition.rss

from this page:

http://edition.cnn.com/services/rss/

Regards,

Andres

![MQL5 Cookbook: ОСО Orders](https://c.mql5.com/2/17/OCO-Orders-MetaTrader5.png)[MQL5 Cookbook: ОСО Orders](https://www.mql5.com/en/articles/1582)

Any trader's trading activity involves various mechanisms and interrelationships including relations among orders. This article suggests a solution of OCO orders processing. Standard library classes are extensively involved, as well as new data types are created herein.

![Trader's Statistical Cookbook: Hypotheses](https://c.mql5.com/2/12/Trader_Statistics_Recipes_MetaTrader5_Alglib_MQL5__1.png)[Trader's Statistical Cookbook: Hypotheses](https://www.mql5.com/en/articles/1240)

This article considers hypothesis - one of the basic ideas of mathematical statistics. Various hypotheses are examined and verified through examples using methods of mathematical statistics. The actual data is generalized using nonparametric methods. The Statistica package and the ported ALGLIB MQL5 numerical analysis library are used for processing data.

![Principles of Exchange Pricing through the Example of Moscow Exchange's Derivatives Market](https://c.mql5.com/2/12/MOEX.png)[Principles of Exchange Pricing through the Example of Moscow Exchange's Derivatives Market](https://www.mql5.com/en/articles/1284)

This article describes the theory of exchange pricing and clearing specifics of Moscow Exchange's Derivatives Market. This is a comprehensive article for beginners who want to get their first exchange experience on derivatives trading, as well as for experienced forex traders who are considering trading on a centralized exchange platform.

![Third Generation Neural Networks: Deep Networks](https://c.mql5.com/2/12/Deep_neural_network_MetaTrader5__2.png)[Third Generation Neural Networks: Deep Networks](https://www.mql5.com/en/articles/1103)

This article is dedicated to a new and perspective direction in machine learning - deep learning or, to be precise, deep neural networks. This is a brief review of second generation neural networks, the architecture of their connections and main types, methods and rules of learning and their main disadvantages followed by the history of the third generation neural network development, their main types, peculiarities and training methods. Conducted are practical experiments on building and training a deep neural network initiated by the weights of a stacked autoencoder with real data. All the stages from selecting input data to metric derivation are discussed in detail. The last part of the article contains a software implementation of a deep neural network in an Expert Advisor with a built-in indicator based on MQL4/R.

[![](https://www.mql5.com/ff/sh/592yc11u3j4rs5z9z2/01.png)How AI helps create robots for MetaTrader 5Learn from our book "Neural Networks in Algo Trading with MQL5"Read](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/neurobook%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.neurobook%26utm_content=visit.page%26utm_campaign=neurobook.promo.04.2024&a=ghrobswocqgvhztzjldphupateyllpro&s=9929cb0b8629585b5a42fabc06c525e41f6c0ebdf3045d044a5413b93ea88b47&uid=&ref=https://www.mql5.com/en/articles/1589&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5068321542913325073)

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