---
title: Extracting structured data from HTML pages using CSS selectors
url: https://www.mql5.com/en/articles/5706
categories: Integration, Expert Advisors
relevance_score: 0
scraped_at: 2026-01-24T14:06:07.444379
---

[![](https://www.mql5.com/ff/sh/x8fwvn495ta7y774z2/01.png)Does your broker offer sponsored hosting for trading?Now it's even easier to get MetaTrader VPS for free – contact your broker for details](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=xscnzeyhifcgygpwvysykhqydcmmbgpp&s=f87b748147e376d34c8f0fdb9737b1766f20cc2174769a0e6b9975b5c2e8ddae&uid=&ref=https://www.mql5.com/en/articles/5706&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5083336387802962333)

MetaTrader 5 / Integration


The MetaTrader development environment enables the integration of applications with external data, in particular with the data obtained from the Internet using the WebRequest function. HTML is the most universal and the most frequently used data format on the web. If a public service does not provide an open API for requests or its protocol is difficult to implement in MQL, the desired HTML pages can be parsed. In particular, traders often use various economic calendars. Although the task is not so relevant now, since the platform features the built-in Calendar, some traders may need specific news from specific sites. Also, we sometimes need to analyze deals from a trading HTML report received from third parties.

The MQL5 ecosystem provides various solutions to the problem, which however are usually specific and have their limitations. On the other hand, there is kind of "native" and universal method to search and parse data from HTML. This method is connected with the use of CSS selectors. In this article we will consider the MQL5 implementation of this method, as well as examples of their practical use.

To analyze HTML, we need to create a parser which can convert internal page text into a hierarchy of some objects called [Document Object Model](https://www.mql5.com/go?link=https://www.w3.org/TR/WD-DOM/introduction.html "https://www.w3.org/TR/WD-DOM/introduction.html") or DOM. From this hierarchy, we will be able to find objects with specified parameters. This approach is based on the use of service information about the document structure, which is not available in the external page view.

For example, we can select rows of a specific table in a document, read the required columns from them and get an array with values, which can be easily saved into a csv file, displayed on a chart or used in Expert Advisor calculations.

### Overview of HTML/CSS and DOM technology

HTML is a popular format which is familiar to almost everyone. Therefore, I will not describe in detail the syntax of this hypertext markup language.

The primary source of related technical information is [IETF](https://www.mql5.com/go?link=https://www.ietf.org/ "https://www.ietf.org/") (Internet Engineering Task Force) and its specifications, the so-called RFC (Request For Comments). There are a lot of HTML specifications (here is [an example](https://www.mql5.com/go?link=https://tools.ietf.org/html/rfc1866 "https://tools.ietf.org/html/rfc1866")). Standards are also available on the website of the related organization, W3C ( [World Wide Web Consortium](https://www.mql5.com/go?link=https://www.w3.org/ "https://www.w3.org/"), [HTML5.2](https://www.mql5.com/go?link=https://www.w3.org/TR/html52/ "https://www.w3.org/TR/html52/")).

These organizations have developed the CSS ( [Cascading Style Sheets](https://www.mql5.com/go?link=https://www.w3.org/standards/webdesign/htmlcss "https://www.w3.org/standards/webdesign/htmlcss")) technology and they regulate it. However we are interested in this technology not because it describes information representation styles on web pages, but because of [CSS selectors](https://www.mql5.com/go?link=https://www.w3.org/TR/selectors-3/ "https://www.w3.org/TR/selectors-3/") contained therein, i.e. a special query language which enables the search of elements inside html pages.

Both HTML and CSS keep constantly evolving, while new versions are being created. For example, the currently relevant versions are HTML5.2 and CSS4. However the update and expansion is always accompanied with the inheritance of old version features. The web is so large, heterogeneous and is often inert, and thus new versions exist along old ones. As a result, when writing algorithms which imply the use of web technologies, you should carefully use the specifications: on the one hand, you should take into account possible traditional deviations and on the other hand you should add some simplifications which will help in avoiding issues with multiple variations.

In this project, we will consider the simplified HTML syntax.

An html document consists of tags inside characters '<' and '>'. The tag name and optional attributes are specified inside the tag. Optional attributes are string pairs of name="value", while the sign '=' can sometimes be omitted. Here is a tag example:

<a href="https://www.w3.org/standards/webdesign/htmlcss" target="\_blank">HTML and CSS</a>

— this is a tag named 'a' (which is interpreted by web browsers as a hyperlink), with two parameters: 'href' for the website address at the specified hyperlink and 'target' for the website opening option (in this case it is equal to "\_blank", i.e. the site should open in a new browser tab).

This first tag is the opening tag. It is followed by the text which is actually visible on the web page: "HTML and CSS", and the matching closing tag, having the same name as the opening tag and an additional slash '/' after the angle bracket '<' (all characters together make up the tag '</a>'). In other words, opening and closing tags are used in pairs and may include other tags, but only whole tags, without overlapping. Here is an example of a correct nesting:

<group attribute1="value1">

<name>text1</name>

<name>text2</name>

</group>

the following "overlapping" is not allowed:

<group id="id1">

<name>text1

</group>

</name>

However, the use is not allowed only in theory. In practice, tags may often be opened or closed by mistake in the wrong place of the document. The parser should be able to handle this situation.

Some tags may be empty, i.e. this can be an empty line:

<p></p>

In accordance with the standards, some tags may (or rather must) have no content at all. For example, the tag describing an image:

<img src="/ico20190101.jpg">

It looks like an opening tag, but it does not have a matching closing one. Such tags are called empty. Please note that the attributes belonging to the tag are not the tag contents.

It is not always easy to determine whether a tag is empty and whether there should be a closing tag further. Although the names of valid empty tags are defined in specifications, some other tags may remain unclosed. Also because HTML and XML formats are close (and there is another variety XHTML), some web page designers create empty tags as follows:

<img src="/ico20190101.jpg" />

Pay attention to the slash '/' before the angle bracket '>'. This slash is considered excessive in terms of strict HTML5 rules. All these specific cases can be met in normal web pages, so the parser must be able to handle them.

Tag and attribute names which are interpreted by web browsers are standard, but HTML can contain customized elements. Such elements are skipped by browsers unless the developers "connects" them to DOM using the specialized script API. We should keep in mind that every tag may contain useful information.

A parser can be considered a [finite-state machine](https://en.wikipedia.org/wiki/Finite-state_machine "https://en.wikipedia.org/wiki/Finite-state_machine"), which advances letter by letter and changes its state in accordance with the context. It is clear from the above tag structure description that initially the parser is outside of any tag (let us call this state "blank"). Then, after encountering the opening angle bracket '<' we get into an opening tag (the "insideTagOpen" state), which lasts until the closing angle bracket '>'. The combination of characters '</' suggests that we are in a closing tag (the "insideTagClose" state), and so on. Other states will be considered in the parser implementation section.

When switching between states, we can select structured information from the current position in the document, because we know the meaning of the state. For example, if the current position is inside an opening tag, the tag name can be selected as a line between the last '<' and the subsequent space or '>' (depending on whether the tag contains attributes). The parser will extract data and create objects of a certain DomElement class. In addition to the name, attributes and contents, the hierarchy of the objects will be preserved based on the tags nesting structure. In other words, each object will have a parent (except the root element which describes the entire document) and an optional array of child objects.

The parser will output the full tree of objects, in which one object will corresponds to one tag in the source document.

CSS selectors describe standard notations for the conditional selection of objects based on their parameters and position in the hierarchy. The full list of selectors is quite extensive. We will provide support for some of them, which are included in the CSS1, CSS2 and CSS3 standards.

Here is a the list of the main selector components:

- \\* \- any object (universal selector);
- .value — an object with the 'class' attribute and a "value"; example: <div class="example"></div>; appropriate selector: .example;
- #id — an object with the 'id' attribute and a "value"; for the tag <div id="unique"></div> it is the selector #unique;
- tag — an object with the 'tag' name; to find all 'div's as the above one or <div>text</div>, use selector: div;

They can be accompanied by the so-called pseudo classes which are added on the right:


- :first-child — the object is the first child class inside a parent;
- :last-child — the object is the last child class i9nside the parent;
- :nth-child(n) — the object has the specified position number in the list of child nods of its parent;
- :nth-last-child(n) — the object has the specified position number in the list of child nods of its parent with reverse numbering;

A single selector can be supplemented by the condition related to attributes:


- \[attr\] — the object has the 'attr' attribute (it does not matter whether the attribute has any value or not);
- \[attr=value\] — the object has the 'attr' attribute with the 'value';
- \[attr\*=text\] — the object has the 'attr' attribute with the value containing the substring 'text';
- \[attr^=start\] — the object has the 'attr' attribute with the value beginning with the 'start' string;
- \[attr$=end\] — the object has the 'attr' attribute with the value ending with the 'end' substring;

If needed, it is possible to specify several pairs of brackets with different attributes.

**Simple selector** is the name selector or a universal selector which can be optionally followed by a class, an identifier, zero or more attributes or a pseudo class, in any order. A simple selector selects an element when all components of the selector match the element properties.

**CSS selector** (or full selector) is a chain of one or more simple selectors joined by combining characters (' ' (space), '>', '+', '~'):


- container element — the 'element' object is nested in the 'container' object at an arbitrary level;
- parent > element — the 'element' object has a direct parent 'parent' (the nesting level is equal to 1);
- e1 + element — the 'element' object has a common parent with 'e1' and immediately follows it;
- e1 ~ element — the 'element' object has a common parent with 'e1' and follows it at any distance;

So far, we have been studying pure theory. Let us view how the above ideas work in practice.

Any modern web browser allows viewing HTML of the currently open page. For example, in Chrome you can run the 'View page source' command from the context menu or open the developer window (Developer tools, Ctrl+Shift+I). The developer window has the Console tab, in which we can try to find elements using CSS selectors. To apply a selector, simply call the document.querySelectorAll function from the console (it is included in the software API of all browsers).

For example, in the start forum page [https://www.mql5.com/en/forum](https://www.mql5.com/en/forum), we can run the following command (JavaScript code):

document.querySelectorAll("div.widgetHeader")

As a result of this we will receive a list of 'div' elements (tags), in which the "widgetHeader" class is specified. I decided to use this selector after viewing the source page code, based on which it is clear that the forum topics are designed in this way.

The selector can be expanded as follows:

document.querySelectorAll("div.widgetHeader a:first-child")

to receive the list of forum topic discussion headers: they are available as hyperlinks 'a', which are first child elements in each 'div' block selected at the first stage. Here is how this might look (depends on the browser version):

![The MQL5 web page and result of selection of HTML elements using CSS selectors](https://c.mql5.com/2/35/chrome.png)

**The MQL5 web page and result of selection of HTML elements using CSS selectors**

You should similarly analyze the HTML code of desired sites, spot the elements of interest and pick up appropriate CSS selectors. The developer window features the Elements (or similar) tab, in which you can select any tag in the document (this tag will be highlighted) and find appropriate CSS selectors for this tag. Thus you will gradually practice the use of selectors and learn to create selector chains manually. Further we will consider how to select appropriate selectors for a specific web page.

### Designing

Let us view the classes which we may need, at a global level. The initial HTML text processing will be performed by the the HtmlParser class. The class will scan the text for markup characters '<', '/', '>' and some others, and it will create DomElement class objects according to the above described finite-state machine rules: one object will be created for each empty tag or a pair of opening and closing tags. The opening tag may have attributes, which we need to read and save in the current DomElement object. This will be performed by the AttributesParser class. The class will also operate following the principle of the finite-state machine.

The parser will create DomElement objects taking into account the hierarchy, which is identical to tag nesting order. For example, if the text contains the 'div' tag, within which several paragraphs are placed (which means the presence of 'p' tags), such paragraphs will be converted into child objects of the object which describes 'div'.

The initial root object will contain the entire document. Similarly to the browser (which provides the document.querySelectorAll method), we should provide in DomElement a method for requesting objects corresponding to passed CSS selectors. The selectors should also be pre-analyzed and converted from the string representation to objects: a single selector component will be stored in the SubSelector class and the entire simple selector will be stored in SubSelectorArray.

Once we have the ready DOM tree as a result of the parser operation, we can request from the root DomElement object (or any other object) all its child elements matching selector parameters. All selected elements will be placed in the iterable DomIterator list. For simplicity, let us implement the list as a child of DomElement, in which an array of child nodes is used for storing the found elements.

Settings with specific site or HTML files processing rules and the algorithm execution result can be conveniently stored in a class, which combines map properties (i.e. provides access to values based on the names of appropriate attributes) and array properties (i.e. access to elements by index). Let us call this class IndexMap.

Let us provide the possibility to nest IndexMap one into another: when collecting tabular data from a web page, we get a list of rows each containing the list of columns. For both of data types we can save the names of source elements. This can be especially useful in cases where some of the required elements are missing in the source document (which may happen quite often) - in such cases simple indexing ignores important information about which data is missing. As a bonus, let us "train" IndexMap to get serialized into a multiline text, including CSV format. This feature is useful when converting HTML pages into tabular data. If necessary, you can replace the IndexMap class with your own while preserving the main functionality.

The following UML diagram displays the described classes.

![UML diagram of classes implementing CSS selectors in MQL](https://c.mql5.com/2/35/htmlcss2.png)

**UML diagram of classes implementing CSS selectors in MQL**

### Implementation

**HtmlParser**

In the HtmlParser class, we describe the variables which are necessary to scan the source text and to generate the object tree, as well as to arrange the finite-state machine algorithm.

The current position in the text is stored in the 'offset' variable. The resulting tree root and the current object (scanning is performed in this object context) are represented by the 'root' and 'cursor' pointers. Their DomElement type will be considered later. The list of tags, which may be empty according to the HTML specification, will be loaded into the 'empties' map (which is initialized in the constructor, see below). Finally, we provide the 'state' variable for the description of finite-state machine states. The variable is an enumeration of the StateBit type.

```
enum StateBit
{
  blank,
  insideTagOpen,
  insideTagClose,
  insideComment,
  insideScript
};

class HtmlParser
{
  private:

    StateBit state;

    int offset;
    DomElement *root;
    DomElement *cursor;
    IndexMap empties;
    ...
```

The StateBit enumeration contains elements describing the following parser states depending on the current position in the text:

- blank — outside of a tag;
- insideTagOpen — inside an opening tag;
- insideTagClose — inside a closing tag;
- insideComment — inside a comment (comments in HTML code are written in tags <!-- comment -->); no objects are generated as long as the parser is inside a comment, no matter which tags are contained in the comment;
- insideScript — inside a script; this state should be highlighted because javascript code often contains substrings which can be interpreted as HTML tags, although they are not DOM elements but are parts of a script);

In addition, let us describe constant strings which will be used to search for markup:

```
    const string TAG_OPEN_START;
    const string TAG_OPEN_STOP;
    const string TAG_OPENCLOSE_STOP;
    const string TAG_CLOSE_START;
    const string TAG_CLOSE_STOP;
    const string COMMENT_START;
    const string COMMENT_STOP;
    const string SCRIPT_STOP;
```

The parser constructor initializes all these variables:

```
  public:
    HtmlParser():
      TAG_OPEN_START("<"),
      TAG_OPEN_STOP(">"),
      TAG_OPENCLOSE_STOP("/>"),
      TAG_CLOSE_START("</"),
      TAG_CLOSE_STOP(">"),
      COMMENT_START("<!--"),
      COMMENT_STOP("-->"),
      SCRIPT_STOP("/script>"),
      state(blank)
    {
      for(int i = 0; i < ArraySize(empty_tags); i++)
      {
        empties.set(empty_tags[i]);
      }
    }
```

An array of empty\_tags strings is used here. This array is preliminary connected from an external text file:

```
string empty_tags[] =
{
  #include <empty_strings.h>
};
```

See the contents below (valid empty tags, but the list is not complete):

```
//  header
"isindex",
"base",
"meta",
"link",
"nextid",
"range",
// body
"img",
"br",
"hr",
"frame",
"wbr",
"basefont",
"spacer",
"area",
"param",
"keygen",
"col",
"limittext"
```

Do not forget to delete the DOM tree:

```
    ~HtmlParser()
    {
      if(root != NULL)
      {
        delete root;
      }
    }
```

The main operations are performed by the parse method:

```
    DomElement *parse(const string &html)
    {
      if(root != NULL)
      {
        delete root;
      }
      root = new DomElement("root");
      cursor = root;
      offset = 0;

      while(processText(html));

      return root;
    }
```

The web page is input, an empty root DomElement is created, the cursor is set to it, while the current position in the text (offset) is set to the very beginning. Then the processText helper method is called in a loop until the entire text is successfully read. The finite-state machine is then executed in this method. The default state of the machine is blank.

```
    bool processText(const string &html)
    {
      int p;
      if(state == blank)
      {
        p = StringFind(html, "<", offset);
        if(p == -1) // no more tags
        {
          return(false);
        }
        else if(p > 0)
        {
          if(p > offset)
          {
            string text = StringSubstr(html, offset, p - offset);
            StringTrimLeft(text);
            StringTrimRight(text);
            StringReplace(text, "&nbsp;", "");
            if(StringLen(text) > 0)
            {
              cursor.setText(text);
            }
          }
        }

        offset = p;

        if(IsString(html, COMMENT_START)) state = insideComment;
        else
        if(IsString(html, TAG_CLOSE_START)) state = insideTagClose;
        else
        if(IsString(html, TAG_OPEN_START)) state = insideTagOpen;

        return(true);
      }
```

The algorithm searches the text for the angle bracket '<'. If it is not found, then there are no more tags so processing should be interrupted (false is returned). If the bracket is found and there is a fragment of text between the new found tag and the previous position (offset), the fragment is considered to be the contents of the current tag (the object is available at the 'cursor' pointer) - so this text is added to the object using the call of cursor.setText().

Then position in the text is moved to the beginning of the new found tag and depending on the signature which follows '<' (COMMENT\_START, TAG\_CLOSE\_START, TAG\_OPEN\_START) the parser is switched to the appropriate new state. The IsString function is a small helper string comparison method, which uses StringSubstr.

In any case true is returned from the processText method, which means that the method will be called again in the loop, but the parser state will be different now. If the current position is in the opening tag, the following code is executed.

```
      else
      if(state == insideTagOpen)
      {
        offset++;
        int pspace = StringFind(html, " ", offset);
        int pright = StringFind(html, ">", offset);
        p = MathMin(pspace, pright);
        if(p == -1)
        {
          p = MathMax(pspace, pright);
        }

        if(p == -1 || pright == -1) // no tag closing
        {
          return(false);
        }
```

If the text has neither space nor '>', the HTML syntax is broken, so false is returned. Further steps select the tag name.

```
        if(pspace > pright)
        {
          pspace = -1; // outer space, disregard
        }

        bool selfclose = false;
        if(IsString(html, TAG_OPENCLOSE_STOP, pright - StringLen(TAG_OPENCLOSE_STOP) + 1))
        {
          selfclose = true;
          if(p == pright) p--;
          pright--;
        }

        string name = StringSubstr(html, offset, p - offset);

        StringToLower(name);
        StringTrimRight(name);
        DomElement *e = new DomElement(cursor, name);
```

Here we have created a new object with the found name. The current object (cursor) is used as the object parent.

Now we need to process the attributes, if there are any.

```
        if(pspace != -1)
        {
          string txt;
          if(pright - pspace > 1)
          {
            txt = StringSubstr(html, pspace + 1, pright - (pspace + 1));
            e.parseAttributes(txt);
          }
        }
```

The parseAttributes method "lives" directly in the DomElement class, which we will consider later.

If the tag is not closed, you should check if it is not the one which can be empty. If it is, it should be "closed" implicitly.

```
        bool softSelfClose = false;
        if(!selfclose)
        {
          if(empties.isKeyExisting(name))
          {
            selfclose = true;
            softSelfClose = true;
          }
        }
```

Depending on whether the tag is closed or not, we either move deeper along the object hierarchy, setting the newly created object as the current one (e), or we remain within the context of the previous object. In any case, position in the text (offset) is moved to the last read character, i.e. beyond '>'.

```
        pright++;
        if(!selfclose)
        {
          cursor = e;
        }
        else
        {
          if(!softSelfClose) pright++;
        }

        offset = pright;
```

A special case is the script. If we meed the <script> tag, the parser switches to the insideScript state, otherwise it switches to the blank state.

```
        if((name == "script") && !selfclose)
        {
          state = insideScript;
        }
        else
        {
          state = blank;
        }

        return(true);

      }
```

The following code is executed in the closing tag state.

```
      else
      if(state == insideTagClose)
      {
        offset += StringLen(TAG_CLOSE_START);
        p = StringFind(html, ">", offset);
        if(p == -1)
        {
          return(false);
        }
```

Again search for '>',which must be available according to the HTML syntax. Iа the bracket is not found, the process should be interrupted. The tag name is highlighted in case of success. This is done to check if the closing tag matches the opening one. And if the matching is broken, it is necessary to somehow overcome this layout error and try to continue parsing.

```
        string tag = StringSubstr(html, offset, p - offset);
        StringToLower(tag);

        DomElement *rewind = cursor;

        while(StringCompare(cursor.getName(), tag) != 0)
        {
          string previous = cursor.getName();
          cursor = cursor.getParent();
          if(cursor == NULL)
          {
            // orphan closing tag
            cursor = rewind;
            state = blank;
            offset = p + 1;
            return(true);
          }
        }
```

We are processing the closing tag, which means that the context of the current object has ended and so the parser switches back to the parent DomElement:

```
        cursor = cursor.getParent();
        if(cursor == NULL) return(false);

        state = blank;
        offset = p + 1;

        return(true);
      }
```

If successful, the parser state again becomes 'blank'.

When the parser is inside a comment, it is obviously looking for the end of the comment.

```
      else
      if(state == insideComment)
      {
        offset += StringLen(COMMENT_START);
        p = StringFind(html, COMMENT_STOP, offset);
        if(p == -1)
        {
          return(false);
        }

        offset = p + StringLen(COMMENT_STOP);
        state = blank;

        return(true);
      }
```

When the parser is inside a script, it searches for the end of the script.

```
      else
      if(state == insideScript)
      {
        p = StringFind(html, SCRIPT_STOP, offset);
        if(p == -1)
        {
          return(false);
        }

        offset = p + StringLen(SCRIPT_STOP);
        state = blank;

        cursor = cursor.getParent();
        if(cursor == NULL) return(false);

        return(true);
      }
      return(false);
    }
```

This was actually the entire HtmlParser class. Now let us consider DomElement.

**DomElement. Beginning**

The DomElement class has variables for storing the name (mandatory), contents, attributes, links to parent and child elements (created as 'protected' because it will be used in the derived class DomIterator).

```
class DomElement
{
  private:
    string name;
    string content;
    IndexMap attributes;
    DomElement *parent;

  protected:
    DomElement *children[];
```

A set of constructors does not require explanations:

```
  public:
    DomElement(): parent(NULL) {}
    DomElement(const string n): parent(NULL)
    {
      name = n;
    }

    DomElement(DomElement *p, const string &n, const string text = "")
    {
      p.addChild(&this);
      parent = p;
      name = n;
      if(text != "") content = text;
    }
```

Of course, the class has "setter" and "getter" field methods (they are omitted in the article), as well as a set of methods for operations with child elements (only prototypes are shown in the article):

```
    void addChild(DomElement *child)
    int getChildrenCount() const;
    DomElement *getChild(const int i) const;
    void addChildren(DomElement *p)
    int getChildIndex(DomElement *e) const;
```

The parseAttributes method which was used at the parsing stage, delegates further work to the AttributesParser helper class.

```
    void parseAttributes(const string &data)
    {
      AttributesParser p;
      p.parseAll(data, attributes);
    }
```

A simple 'data' string is output, based on which the method fills the 'attributes' map with the properties found.

The full AttributesParser code is available in attachments below. The class is not large and operates by finite-state machine principle, similarly to HtmlParser. But it has only two states:

```
enum AttrBit
{
  name,
  value
};
```

Since the list of attributes is a string consisting of name="value" pairs, AttributesParser is always either at the name or at the value. This parser could be implemented using the StringSplit function, but because of possible formatting deviations (such as the presence or absence of quotes, the use of spaces inside the quotes, etc.), the machine approach was chosen.

As for the DomElement class, most of the work in it should be performed by methods which select child elements corresponding to given CSS selectors. Before we proceed to this feature, it is necessary to describe the selector classes.

**SubSelector and SubSelectorArray**

The SubSelector class describes one component of a simple selector. For example the simple selector "td\[align=left\]\[width=325\]" has three components:

- tag name - td
- align attribute condition - \[align=left\]
- width attribute condition - \[width=325\]

The simple selector "td:first-child" has two components:


- tag name - td
- child index condition using the pseudo class - :first-child

The simple selector "span.main\[id^=calendarTip\]" again has three components:


- tag name — span
- class — main
- the id attribute must start with the calendarTip string

Here is the class:

```
class SubSelector
{
  enum PseudoClassModifier
  {
    none,
    firstChild,
    lastChild,
    nthChild,
    nthLastChild
  };

  public:
    ushort type;
    string value;
    PseudoClassModifier modifier;
    string param;
};
```

The 'type' variable contains the first character of the selector ('.', '#', '\[') or the default 0, which corresponds to the name selector. The value variable stores the substring which follows the character. i.e. the actual searched element. If the selector string has a pseudo class, its id is written to the 'modifier' field. In the description of selectors ":nth-child" and ":nth-last-child", the index of the searched element is specified in brackets. This will be saved in the 'param' field (it can only be a number in the current implementation, but special formulas are also allowed and therefore the field is declared as string).\
\
The SubSelectorArray class provides a bunch of components, therefore let us declare the 'selectors' array in it:\
\
```\
class SubSelectorArray\
{\
  private:\
    SubSelector *selectors[];\
```\
\
SubSelectorArray is one simple selector as a whole. No class is needed for full CSS selectors since they are processed sequentially, step by step, i.e. one selector at each hierarchy level.\
\
Let us add the supported pseudo class selectors to the 'mod' map. This enables the immediate getting of the appropriate modifier from PseudoClassModifier for that string:\
\
```\
    IndexMap mod;\
\
    static TypeContainer<PseudoClassModifier> first;\
    static TypeContainer<PseudoClassModifier> last;\
    static TypeContainer<PseudoClassModifier> nth;\
    static TypeContainer<PseudoClassModifier> nthLast;\
\
    void init()\
    {\
      mod.add(":first-child", &first);\
      mod.add(":last-child", &last);\
      mod.add(":nth-child", &nth);\
      mod.add(":nth-last-child", &nthLast);\
    }\
```\
\
The TypeContainer class is a template wrapper for the values which are added to IndexMap.\
\
Note that static members (in this case objects for the map) must be initialized after the class description:\
\
```\
TypeContainer<PseudoClassModifier> SubSelectorArray::first(PseudoClassModifier::firstChild);\
TypeContainer<PseudoClassModifier> SubSelectorArray::last(PseudoClassModifier::lastChild);\
TypeContainer<PseudoClassModifier> SubSelectorArray::nth(PseudoClassModifier::nthChild);\
TypeContainer<PseudoClassModifier> SubSelectorArray::nthLast(PseudoClassModifier::nthLastChild);\
```\
\
Let us get back to the SubSelectorArray class.\
\
When it is necessary to add a simple selector component to the array, the add function is called:\
\
```\
    void add(const ushort t, string v)\
    {\
      int n = ArraySize(selectors);\
      ArrayResize(selectors, n + 1);\
\
      PseudoClassModifier m = PseudoClassModifier::none;\
      string param;\
\
      for(int j = 0; j < mod.getSize(); j++)\
      {\
        int p = StringFind(v, mod.getKey(j));\
        if(p > -1)\
        {\
          if(p + StringLen(mod.getKey(j)) < StringLen(v))\
          {\
            param = StringSubstr(v, p + StringLen(mod.getKey(j)));\
            if(StringGetCharacter(param, 0) == '(' && StringGetCharacter(param, StringLen(param) - 1) == ')')\
            {\
              param = StringSubstr(param, 1, StringLen(param) - 2);\
            }\
            else\
            {\
              param = "";\
            }\
          }\
\
          m = mod[j].get<PseudoClassModifier>();\
          v = StringSubstr(v, 0, p);\
\
          break;\
        }\
      }\
\
      if(StringLen(param) == 0)\
      {\
        selectors[n] = new SubSelector(t, v, m);\
      }\
      else\
      {\
        selectors[n] = new SubSelector(t, v, m, param);\
      }\
    }\
```\
\
The first character (type) and the next string is passed to it. The string is parsed to the searched object name, optionally a pseudo class and a parameter. All this is then passed to the SubSelector constructor, while a new selector component is added to the 'selectors' array.\
\
The add function is used indirectly from the simple selector constructor:\
\
```\
  private:\
    void createFromString(const string &selector)\
    {\
      ushort p = 0; // previous/pending type\
      int ppos = 0;\
      int i, n = StringLen(selector);\
      for(i = 0; i < n; i++)\
      {\
        ushort t = StringGetCharacter(selector, i);\
        if(t == '.' || t == '#' || t == '[' || t == ']')\
        {\
          string v = StringSubstr(selector, ppos, i - ppos);\
          if(i == 0) v = "*";\
          if(p == '[' && StringLen(v) > 0 && StringGetCharacter(v, StringLen(v) - 1) == ']')\
          {\
            v = StringSubstr(v, 0, StringLen(v) - 1);\
          }\
          add(p, v);\
          p = t;\
          if(p == ']') p = 0;
          ppos = i + 1;
        }
      }

      if(ppos < n)
      {
        string v = StringSubstr(selector, ppos, n - ppos);
        if(p == '[' && StringLen(v) > 0 && StringGetCharacter(v, StringLen(v) - 1) == ']')
        {
          v = StringSubstr(v, 0, StringLen(v) - 1);
        }
        add(p, v);
      }
    }

  public:
    SubSelectorArray(const string selector)
    {
      init();
      createFromString(selector);
    }
```

The createFromString function receives a text representation of the CSS selector and views it in a loop to find special beginning characters '.', '#' or '\[', then determines where the component ends and calls the 'add' method for the selected information. The loop continues as long as the chain of components continues.\
\
The full code of the SubSelectorArray is attached below.\
\
Now it is time to get back to the DomElement class. This is the most difficult part.\
\
**DomElement. Continued**\
\
The querySelect method is used to search for elements matching the specified selectors (in the textual representation). In this method, the full CSS selector is divided into simple selectors, which are then converted to the SubSelectorArray object. The list of matching elements us searched for each simple selector. Other elements matching the next simple selector are searched relative to the found elements. This is continued until the last simple selector is met or until the list of found elements becomes empty.\
\
```\
    DomIterator *querySelect(const string q)\
    {\
      DomIterator *result = new DomIterator();\
```\
\
Here the return value is the unfamiliar DomIterator class, which is the child of DomElement. It provides auxiliary functionality in addition to DomElement (in particular, it allows "scrolling" child elements), so we will not analyze DomIterator in details now. There is another complicated part.\
\
The selector string is analyzed character by character. For this purpose several local variables are used. The current character is stored in the **_c_** variable (abbr. of 'character'). The previous character is stored in the **_p_** variable (abbr. of 'previous'). If a character is one of combinator characters (' ', '+', '>', '~'), it is saved in a variable ( **_a_**), but is not used until the next simple selector is determined.\
\
Combinators are located between simple selectors, while the operation defined by the combinators can only be performed after reading the entire selector on the right. Therefore, the last read combinator ( **_a_**) first passes through the "waiting" state: the **_a_** variable is not used until the next combinator appears or the string end is reached, while both cases mean that the selector has been fully formed. Only at this moment the "old" combinator ( **_b_**) is applied and is replaced by a new one ( **_a_**). The code itself is clearer than its description:\
\
```\
      int cursor = 0; // where selector string started\
      int i, n = StringLen(q);\
      ushort p = 0;   // previous character\
      ushort a = 0;   // next/pending operator\
      ushort b = '/'; // current operator, 'root' notation from the start\
      string selector = "*"; // current simple selector, 'any' by default\
      int index = 0;  // position in the resulting array of objects\
\
      for(i = 0; i < n; i++)\
      {\
        ushort c = StringGetCharacter(q, i);\
        if(isCombinator(c))\
        {\
          a = c;\
          if(!isCombinator(p))\
          {\
            selector = StringSubstr(q, cursor, i - cursor);\
          }\
          else\
          {\
            // suppress blanks around other combinators\
            a = MathMax(c, p);\
          }\
          cursor = i + 1;\
        }\
        else\
        {\
          if(isCombinator(p)) // action\
          {\
            index = result.getChildrenCount();\
\
            SubSelectorArray selectors(selector);\
            find(b, &selectors, result);\
            b = a;\
\
            // now we can delete outdated results in positions up to 'index'\
            result.removeFirst(index);\
          }\
        }\
        p = c;\
      }\
\
      if(cursor < i) // action\
      {\
        selector = StringSubstr(q, cursor, i - cursor);\
\
        index = result.getChildrenCount();\
\
        SubSelectorArray selectors(selector);\
        find(b, &selectors, result);\
        result.removeFirst(index);\
      }\
\
      return result;\
    }\
```\
\
The 'cursor' variable always points at the first character, from which the string with the simple selector begins (i.e. at the character which immediately follows the previous combinator or at the string beginning). When another combinator is found, copy the substring from 'cursor' to the current character (i) into the 'selector' variable.\
\
Sometimes there are several combinators in succession: this may usually happen when other combinator characters surround spaces, while the space itself is also a combinator. For example, the entries "td>span" and "td > span" are equivalent, but spaces are inserted in the second case to improve readability. Such situations are handled by the line:\
\
```\
a = MathMax(c, p);\
```\
\
It compares the current and previous characters if both are combinators. Then, based on the fact that the space has the smallest code, select an "older" combinator. The combinator array is obviously defined as follows:\
\
```\
ushort combinators[] =\
{\
  ' ', '+', '>', '~'\
};\
```\
\
Check of whether the character is included into this array is performed by a simple isCombinator helper function.\
\
If there are two combinators in a row, other than a space, then the selector is erroneous and behavior is not defined in specifications. However, our code does not lose performance and suggests consistent behavior.\
\
If the current character is not a combinator while the previous character was a combinator, the execution falls into a branch marked with an 'action' comment. Now memorize the current size of the array of DomElements selected to this moment by calling:\
\
```\
index = result.getChildrenCount();\
```\
\
The array is initially empty and index = 0.\
\
Create an array of selector objects corresponding to the current simple selector, i.e. the 'selector' string:\
\
```\
SubSelectorArray selectors(selector);\
```\
\
Then call the 'find' method, which will be considered further.\
\
```\
find(b, &selectors, result);\
```\
\
Pass the combinator character into it (this should be the preceding combinator, i.e. from the b variable), as well as the simple selector and an array to output results to.\
\
After that move the queue of combinators forward, copy the last found combinator (which has not yet been processed) from the **_a_** variable to **_b_** and delete from results everything which was available before the call of 'find' using:\
\
```\
result.removeFirst(index);\
```\
\
The removeFirst method is defined in DomIterator. It performs a simple task: it deletes from an array all first elements up to the specified number. This is done because in the process of each successive simple selector processing, we narrow the element selection conditions and everything selected earlier is no longer valid, while the newly added elements (which meet these narrow conditions) start with 'index'.\
\
Similar processing (marked with the 'action' comment) is also performed after reaching the end of the input string. In this case, the last pending combinator should be processed in conjunction with the rest of the line (from the 'cursor' position).\
\
Now let us consider the 'find' method.\
\
```\
    bool find(const ushort op, const SubSelectorArray *selectors, DomIterator *output)\
    {\
      bool found = false;\
      int i, n;\
```\
\
If one of the combinators setting tag nesting conditions (' ', '>') is input, checks should be recursively called for all child elements. In this branch we also need to take into account the special combinator '/', which is used at search beginning in the calling method.\
\
```\
      if(op == ' ' || op == '>' || op == '/')\
      {\
        n = ArraySize(children);\
        for(i = 0; i < n; i++)\
        {\
          if(children[i].match(selectors))\
          {\
            if(op == '/')\
            {\
              found = true;\
              output.addChild(GetPointer(children[i]));\
            }\
```\
\
The 'match' method will be considered later. It returns true, if the object corresponds to passed selector or false if otherwise. At the very beginning of search (combinator op = '/'), there are no combinations yet, so all tags meeting selector rules are added to the result (output.addChild).\
\
```\
            else\
            if(op == ' ')\
            {\
              DomElement *p = &this;\
              while(p != NULL)\
              {\
                if(output.getChildIndex(p) != -1)\
                {\
                  found = true;\
                  output.addChild(GetPointer(children[i]));\
                  break;\
                }\
                p = p.parent;\
              }\
            }\
```\
\
For the combinator ' ', a check is performed of whether the current DomElement or any its parent already exists in 'output'. This means that the new child elements which satisfies search conditions is already nested into the parent. This is exactly the task of the combinator.\
\
The combinator '>' operates in a similar way, but it needs to track only immediate "relatives" and thus only check if the current DomElement is available in interim results. If it is, then it has earlier been selected to 'output' by conditions of the selector on the left of the combinator and its i-th child element has just met the conditions of the selector to the right of the combinator.\
\
```\
            else // op == '>'\
            {\
              if(output.getChildIndex(&this) != -1)\
              {\
                found = true;\
                output.addChild(GetPointer(children[i]));\
              }\
            }\
          }\
```\
\
Then similar checks need to be performed deep in the DOM tree, therefore 'find' should be recursively called for child elements.\
\
```\
          children[i].find(op, selectors, output);\
        }\
      }\
```\
\
Combinators '+' and '~' set conditions of whether two elements refer to the same parent.\
\
```\
      else\
      if(op == '+' || op == '~')\
      {\
        if(CheckPointer(parent) == POINTER_DYNAMIC)\
        {\
          if(output.getChildIndex(&this) != -1)\
          {\
```\
\
One of the elements must be already selected by a selector on the left. If this condition is met, check the "siblings" for the selector on the right ("siblings" are the children of the current node parent).\
\
```\
            int q = parent.getChildIndex(&this);\
            if(q != -1)\
            {\
              n = (op == '+') ? (q + 2) : parent.getChildrenCount();\
              if(n > parent.getChildrenCount()) n = parent.getChildrenCount();\
              for(i = q + 1; i < n; i++)\
              {\
                DomElement *m = parent.getChild(i);\
                if(m.match(selectors))\
                {\
                  found = true;\
                  output.addChild(m);\
                }\
              }\
            }\
```\
\
The difference between handling of '+' and '~' is as follows: with '+' elements must be immediate neighbors while with '~' there can be any number of other "siblings" between the elements. Therefore the loop is only performed once for '+', i.e. for the next element in the array of child elements. The 'match' function is called again inside the loop (see details later).\
\
```\
          }\
        }\
        for(i = 0; i < ArraySize(children); i++)\
        {\
          found = children[i].find(op, selectors, output) || found;\
        }\
      }\
      return found;\
    }\
```\
\
After all checks, move to the next DOM element tree hierarchy level and call 'find' for child nods.\
\
That is all about the 'find' method. Now let us view the 'match' function. This is the last point in the description of the selector implementation.\
\
The function checks in the current object the entire chain of components of a simple selector passed through an input parameter. If at least one component in the loop does not match the element properties, the check fails.\
\
```\
    bool match(const SubSelectorArray *u)\
    {\
      bool matched = true;\
      int i, n = u.size();\
      for(i = 0; i < n && matched; i++)\
      {\
        if(u[i].type == 0) // tag name and pseudo-classes\
        {\
          if(u[i].value == "*")\
          {\
            // any tag\
          }\
```\
\
The 0 type selector is the tag name or a pseudo class. Any tag is suitable for a selector containing an asterisk. Otherwise the selector string should be compared with the tag name:\
\
```\
          else\
          if(StringCompare(name, u[i].value) != 0)\
          {\
            matched = false;\
          }\
```\
\
The currently implemented pseudo-classes set limitations on the number of the current element in the array of a parent's child elements, so we analyze the indexes:\
\
```\
          else\
          if(u[i].modifier == PseudoClassModifier::firstChild)\
          {\
            if(parent != NULL && parent.getChildIndex(&this) != 0)\
            {\
              matched = false;\
            }\
          }\
          else\
          if(u[i].modifier == PseudoClassModifier::lastChild)\
          {\
            if(parent != NULL && parent.getChildIndex(&this) != parent.getChildrenCount() - 1)\
            {\
              matched = false;\
            }\
          }\
          else\
          if(u[i].modifier == PseudoClassModifier::nthChild)\
          {\
            int x = (int)StringToInteger(u[i].param);\
            if(parent != NULL && parent.getChildIndex(&this) != x - 1) // children are counted starting from 1\
            {\
              matched = false;\
            }\
          }\
          else\
          if(u[i].modifier == PseudoClassModifier::nthLastChild)\
          {\
            int x = (int)StringToInteger(u[i].param);\
            if(parent != NULL && parent.getChildrenCount() - parent.getChildIndex(&this) - 1 != x - 1)\
            {\
              matched = false;\
            }\
          }\
        }\
```\
\
Selector '.' imposes a restriction on the "class" attribute:\
\
```\
        else\
        if(u[i].type == '.')\
        {\
          if(attributes.isKeyExisting("class"))\
          {\
            Container *c = attributes["class"];\
            if(c == NULL || StringFind(" " + c.get<string>() + " ", " " + u[i].value + " ") == -1)\
            {\
              matched = false;\
            }\
          }\
          else\
          {\
            matched = false;\
          }\
        }\
```\
\
Selector '#' imposes a restriction on the "id" attribute:\
\
```\
        else\
        if(u[i].type == '#')\
        {\
          if(attributes.isKeyExisting("id"))\
          {\
            Container *c = attributes["id"];\
            if(c == NULL || StringCompare(c.get<string>(), u[i].value) != 0)\
            {\
              matched = false;\
            }\
          }\
          else\
          {\
            matched = false;\
          }\
        }\
```\
\
The selector '\[' enables the specification of an arbitrary set of required attributes. Also, in addition to strict comparison of values, it is possible to check the occurrence of a substring (suffix '\*'), beginning ('^') and end ('$').\
\
```\
        else\
        if(u[i].type == '[')\
        {\
          AttributesParser p;\
          IndexMap hm;\
          p.parseAll(u[i].value, hm);\
          // attributes are selected one by one: element[attr1=value][attr2=value]\
          // the map should contain only 1 valid pair at a time\
          if(hm.getSize() > 0)\
          {\
            string key = hm.getKey(0);\
            ushort suffix = StringGetCharacter(key, StringLen(key) - 1);\
\
            if(suffix == '*' || suffix == '^' || suffix == '$') // contains, starts with, or ends with\
            {\
              key = StringSubstr(key, 0, StringLen(key) - 1);\
            }\
            else\
            {\
              suffix = 0;\
            }\
\
            if(hasAttribute(key) && attributes[key] != NULL)\
            {\
              string v = hm[0] != NULL ? hm[0].get<string>() : "";\
              if(StringLen(v) > 0)\
              {\
                if(suffix == 0)\
                {\
                  if(key == "class")\
                  {\
                    matched &= (StringFind(" " + attributes[key].get<string>() + " ", " " + v + " ") > -1);\
                  }\
                  else\
                  {\
                    matched &= (StringCompare(v, attributes[key].get<string>()) == 0);\
                  }\
                }\
                else\
                if(suffix == '*')\
                {\
                  matched &= (StringFind(attributes[key].get<string>(), v) != -1);\
                }\
                else\
                if(suffix == '^')\
                {\
                  matched &= (StringFind(attributes[key].get<string>(), v) == 0);\
                }\
                else\
                if(suffix == '$')\
                {\
                  string x = attributes[key].get<string>();\
                  if(StringLen(x) > StringLen(v))\
                  {\
                    matched &= (StringFind(x, v, StringLen(x) - StringLen(v)) == StringLen(v));\
                  }\
                }\
              }\
            }\
            else\
            {\
              matched = false;\
            }\
          }\
        }\
      }\
\
      return matched;\
\
    }\
```\
\
Please note that the "class" attribute is also supported and processed here. Moreover, similarly to '.', not strict matching is checked, but the availability of the class among a set of other classes. Often in HTML multiple classes are assigned to one element. In this case classes are specified in the 'class' attributed separated with a space.\
\
Let's sum up the intermediate results. We have implemented in the DomElement class the querySelect method, which accepts a string with the full CSS selector as a parameter and returns the DomIterator object, i.e. an array of found matching elements. Inside querySelect, the CSS selector string is divided into a sequence of simple selectors and combinator characters between them. For each simple selector, the 'find' method with the specified combinator is called. This method updates the list of results, while recursively calling itself for child elements. Comparison of simple selector components with the properties of a particular element is performed in the 'match' method.\
\
For example, using the querySelect method we can select rows from a table using one CSS selector and then we can call querySelect for each row with another CSS selector to isolate specific cells. Since operations with tables are required very often, let us create the tableSelect method in the DomElement class, which will implement the above described approach. Its code is provided in a simplified form.\
\
```\
    IndexMap *tableSelect(const string rowSelector, const string &columSelectors[], const string &dataSelectors[])\
    {\
```\
\
The row selector is specified in the rowSelector parameter, while cell selectors are specified in the columSelectors array.\
\
Once all the elements are selected, we will need to take some information from them, such as text or attribute value. Let us use the dataSelectors to determine the position of the required information within an element, while an individual data extraction method can be used for each table column.\
\
If dataSelectors\[i\] is an empty row, read the textual contents of the tag (between the opening and closing parts, for example "100%" from tag "<p>100%</p>"). If dataSelectors\[i\] is a row, consider this the attribute name and use this value.\
\
Let us view the full implementation in detail:\
\
```\
      DomIterator *r = querySelect(rowSelector);\
```\
\
Here we get the resulting list of elements by row selector.\
\
```\
      IndexMap *data = new IndexMap('\n');\
      int counter = 0;\
      r.rewind();\
```\
\
Here we create an empty map to which table data will be added, and prepare for a loop through row objects. Here is the loop:\
\
```\
      while(r.hasNext())\
      {\
        DomElement *e = r.next();\
\
        string id = IntegerToString(counter);\
\
        IndexMap *row = new IndexMap();\
```\
\
Thus we get the next row, (e), create a container map for it (row), to which cells will be added, and run loop through columns:\
\
```\
        for(int i = 0; i < ArraySize(columSelectors); i++)\
        {\
          DomIterator *d = e.querySelect(columSelectors[i]);\
```\
\
In each row object, select the list of cell objects (d) using the appropriate selector. Select data from each found cell and save it to the 'row' map:\
\
```\
          string value;\
\
          if(d.getChildrenCount() > 0)\
          {\
            if(dataSelectors[i] == "")\
            {\
              value = d[0].getText();\
            }\
            else\
            {\
              value = d[0].getAttribute(dataSelectors[i]);\
            }\
\
            StringTrimLeft(value);\
            StringTrimRight(value);\
\
            row.setValue(IntegerToString(i), value);\
          }\
```\
\
Integer keys are used here for code simplicity, while the full source code supports the use of element identifiers for the keys.\
\
If a matching cell is not found, mark it as empty.\
\
```\
          else // field not found\
          {\
            row.set(IntegerToString(i));\
          }\
          delete d;\
        }\
```\
\
Add the field 'row' to the 'data' table.\
\
```\
        if(row.getSize() > 0)\
        {\
          data.set(id, row);\
          counter++;\
        }\
        else\
        {\
          delete row;\
        }\
      }\
\
      delete r;\
\
      return data;\
    }\
```\
\
Thus, at the output we get a map of maps, i.e. a table with row numbers along the first dimension and column numbers along the second. If necessary, the tableSelect function can be adjusted to other data containers.\
\
A non-trading utility Expert Advisor was created to apply all the above classes.\
\
### The WebDataExtractor utility Expert Advisor\
\
The Expert Advisor is used to convert data from web pages into a tabular structure with the possibility to save the result to a CSV file.\
\
The Expert Advisor received the following input parameters: a link to the source data (a local file or a web page which can be downloaded using WebRequest), row and column selectors and the CSV file name. The main input parameters are shown below:\
\
```\
input string URL = "";\
input string SaveName = "";\
input string RowSelector = "";\
input string ColumnSettingsFile = "";\
input string TestQuery = "";\
input string TestSubQuery = "";\
```\
\
In URL, specify the web page address (beginning with http:// or https://) or the local html file name.\
\
In SaveName, the name of the CSV file with results is specified in normal mode. But it can also be used for other purpose: to save the downloaded page for subsequent debugging of selectors. In this mode the next parameter should be left empty: RowSelector, in which the CSS row selector is usually specified.\
\
Since there are several column selectors, they are set in a separate CSV set file, which name is specified in the ColumnSettingsFile parameter. The file format is as follows.\
\
The first line is the header, each subsequent line describes a separate field (a data column in the table row).\
\
The file should have three columns: name, CSS selector, data locator:\
\
- name — the i-th column in the output CSV file;\
- CSS selector — for selecting an element, from which data for the i-th column of the output CSV file will be use. This selector is applied inside each DOM-element, previously selected from a web page using RowSelector. To directly select a row element specify here '.';\
- data "locator" — determines which element part data will be used from; the attribute name can be specified or it can be left empty to take the text contents of a tag.\
\
TestQuery and TestSubQuery parameters allow testing selectors for a row and one column, while outputting the result to log but not saving to CSV and not using settings files for all columns.\
\
Here is the main operating function of the Expert Advisor in a brief form.\
\
```\
int process()\
{\
  string xml;\
\
  if(StringFind(URL, "http://") == 0 || StringFind(URL, "https://") == 0)\
  {\
    xml = ReadWebPageWR(URL);\
  }\
  else\
  {\
    Print("Reading html-file ", URL);\
    int h = FileOpen(URL, FILE_READ|FILE_TXT|FILE_SHARE_WRITE|FILE_SHARE_READ|FILE_ANSI, 0, CP_UTF8);\
    if(h == INVALID_HANDLE)\
    {\
      Print("Error reading file '", URL, "': ", GetLastError());\
      return -1;\
    }\
    StringInit(xml, (int)FileSize(h));\
    while(!FileIsEnding(h))\
    {\
      xml += FileReadString(h) + "\n";\
    }\
    // xml = FileReadString(h, (int)FileSize(h)); - has 4095 bytes limit in binary files!\
    FileClose(h);\
  }\
  ...\
```\
\
Thus we have read an HTML page from a file or downloaded from the Internet. Now, in order to convert the document to the hierarchy of DOM objects, let us create the HtmlParser object and start parsing:\
\
```\
  HtmlParser p;\
  DomElement *document = p.parse(xml);\
```\
\
If testing selectors are specified, handle them by querySelect calls:\
\
```\
  if(TestQuery != "")\
  {\
    Print("Testing query, subquery: '", TestQuery, "', '", TestSubQuery, "'");\
    DomIterator *r = document.querySelect(TestQuery);\
    r.printAll();\
\
    if(TestSubQuery != "")\
    {\
      r.rewind();\
      while(r.hasNext())\
      {\
        DomElement *e = r.next();\
        DomIterator *d = e.querySelect(TestSubQuery);\
        d.printAll();\
        delete d;\
      }\
    }\
\
    delete r;\
    return(0);\
  }\
```\
\
In a normal operation mode, read the column setting file and call the tableSelect function:\
\
```\
  string columnSelectors[];\
  string dataSelectors[];\
  string headers[];\
\
  if(!loadColumnConfig(columnSelectors, dataSelectors, headers)) return(-1);\
\
  IndexMap *data = document.tableSelect(RowSelector, columnSelectors, dataSelectors);\
```\
\
If a CSV file for saving results is specified, let the 'data' map perform this task.\
\
```\
  if(StringLen(SaveName) > 0)\
  {\
    Print("Saving data as CSV to ", SaveName);\
    int h = FileOpen(SaveName, FILE_WRITE|FILE_CSV|FILE_ANSI, '\t', CP_UTF8);\
    if(h == INVALID_HANDLE)\
    {\
      Print("Error writing ", data.getSize() ," rows to file '", SaveName, "': ", GetLastError());\
    }\
    else\
    {\
      FileWriteString(h, StringImplodeExt(headers, ",") + "\n");\
\
      FileWriteString(h, data.asCSVString());\
      FileClose(h);\
      Print((string)data.getSize() + " rows written");\
    }\
  }\
  else\
  {\
    Print("\n" + data.asCSVString());\
  }\
\
  delete data;\
\
  return(0);\
}\
```\
\
Let us proceed to practical application of the Expert Advisor.\
\
### Practical use\
\
Traders often deal with some standard HTML files, such as testing reports and trading reports generated by MetaTrader. We sometimes receive such files from other traders or download from the Internet and want to visualize the data on a chart for further analysis. For this purpose data from HTML should be converted to a tabular view (to the CSV format in a simple case).\
\
CSS selector in our utility can automate this process.\
\
Let us have a look inside the HTML files. Below is the appearance and part of HTML code of the MetaTrader 5 trading report (the ReportHistory.html file is attached below).\
\
![Trading report appearance and part of HTML code](https://c.mql5.com/2/35/chrome-report.png)\
\
**Trading report appearance and part of HTML code**\
\
And now here is the appearance and part of HTML code of the MetaTrader 5 testing report (the Tester.html file is attached below).\
\
![Tester report appearance and part of HTML code](https://c.mql5.com/2/35/chrome-tester.png)\
\
**Tester report appearance and part of HTML code**\
\
According to the appearance in the above figure, the trading report has 2 tables: Orders and Deals. However, from the internal layout we can see that this is a single table. All visible headers and the dividing line are formed by the styles of table cells. We need to learn to distinguish between orders and deals and save each of the sub-tables to a separate CSV file.\
\
The difference between the first part and the second is in the number of columns: 11 columns for orders and 13 columns for deals. Unfortunately, the CSS standard does not allow setting conditions for selecting parent elements (in our case, the table rows, 'tr' tag) based on the number or content of children (in our case, table cells, 'td' tag). So, in some cases, required elements cannot be selected using standard means. But we are developing our own implementation of selectors and thus we can add a special non-standard selector for the number of child elements. This will be a new pseudo class. Let us set it as ":has-n-children(n)", by analogy with ":nth-child(n)".\
\
The following selector can be used for selecting order rows:\
\
tr:has-n-children(11)\
\
However, this is not the entire solution to the problem, because this selector selects the table header in addition to the data rows. Let us remove it. Pay attention to coloring of data rows - the bgcolor attribute is set for them, and the color value alternates for even and odd rows (#FFFFFF and #F7F7F7). A color, i.e. the bgcolor attribute is also used for the header, but its value is equal to #E5F0FC. Thus, the data rows have light colors with bgcolor starting with "#F". Let us add this condition to the selector:\
\
tr:has-n-children(11)\[bgcolor^="#F"\]\
\
The selector correctly determines all rows with orders.\
\
Parameters of each order can be read from the row cells. To do this, let us write the configuration file ReportHistoryOrders.cfg.csv:\
\
Name,Selector,Data\
\
Time,td:nth-child(1),\
\
Order,td:nth-child(2),\
\
Symbol,td:nth-child(3),\
\
Type,td:nth-child(4),\
\
Volume,td:nth-child(5),\
\
Price,td:nth-child(6),\
\
S/L,td:nth-child(7),\
\
T/P,td:nth-child(8),\
\
Time,td:nth-child(9),\
\
State,td:nth-child(10),\
\
Comment,td:nth-child(11),\
\
All fields in this file are simply identified by the sequence number. In other cases you may need smarter selectors with attributes and classes.\
\
To get a table of deals, simply replace the number of child elements to 13 in the row selector:\
\
tr:has-n-children(13)\[bgcolor^="#F"\]\
\
The configuration file ReportHistoryDeals.cfg.csv is attached below.\
\
Now, by launching WebDataExtractor with the following input parameters (the webdataex-report1.set file is attached):\
\
URL=ReportHistory.html\
\
SaveName=ReportOrders.csv\
\
RowSelector=tr:has-n-children(11)\[bgcolor^="#F"\]\
\
ColumnSettingsFile=ReportHistoryOrders.cfg.csv\
\
we will receive the resulting ReportOrders.csv file which corresponds to the source HTML report:\
\
![CSV file resulting from the application of CSS selectors to a trading report](https://c.mql5.com/2/35/report-csv.png)\
\
**CSV file resulting from the application of CSS selectors to a trading report**\
\
To get the table of deals, use the attached settings from webdataex-report2.set.\
\
The selectors which we created are also suitable for tester reports. The attached webdataex-tester1.set and webdataex-tester2.set allow you to convert a sample HTML report Tester.html into CSV files.\
\
Important! The layout of many web pages as well as of the the generated HTML files in MetaTrader can be changed from time to time. Due to this the some of selectors will no longer be applicable, even if the external presentation is almost the same. In this case you should re-analyze the HTML code and modify the CSS selectors accordingly.\
\
Now let us view the conversion for the MetaTrader 4 tester report; this allows demonstration of some interesting techniques in selecting CSS selectors. For the check we will use the attached StrategyTester-ecn-1.htm.\
\
These files have two tables: one with the testing results and the other one with trading trading operations. To select the second table we will use the selector "table ~ table". Omit the first row in the operations table, because it contains a header. This can be done using the selector "tr + tr".\
\
Having combined them, we obtain a selector for selecting working rows:\
\
table ~ table tr + tr\
\
This actually means the following: select a table after the table (i.e. the second one, inside the table select each line having a previous row, i.e. all except the first one).\
\
Settings for extracting deal parameters from cells are available in the file test-report-mt4.cfg.csv. The date field is processed by the class selector:\
\
DateTime,td.msdate,\
\
i.e. it searches for td tags with the class="msdate" attribute.\
\
The full settings file for the utility is webdataex-tester-mt4.set.\
\
Additional CSS selector use and setup examples are provided in the [WebDataExtractor discussion](https://www.mql5.com/en/market/product/12635#!tab=tab_p_comments) page.\
\
The utility can do much more:\
\
\
- automatically substitute strings (for example, change country names to currency symbols or replace verbal descriptions of news priority with a number)\
- output the DOM tree to log and find suitable selectors without a browser;\
- download and convert web pages by timer or by request from a global variable;\
\
If you need help in the setting up of CSS selectors for a specific web page, you can purchase WebDataExtractor ( [for MetaTrader 4](https://www.mql5.com/en/market/product/12635), [for MetaTrader 5](https://www.mql5.com/en/market/product/36301)) and receive recommendations as part of product support. However, the availability of source codes allows you to use the entire functionality and expand it if necessary. This is absolutely free.\
\
### Conclusions\
\
We have considered the technology of CSS selectors, which is one of the main standards in the interpretation of web documents. The implementation of the most commonly used CSS selectors in MQL allows the flexible setup and conversion of any HTML page, including standard MetaTrader documents, into structured data without using third-party software.\
\
We have not considered some other technologies which can also provide versatile tools for processing web documents. Such tools can be useful, because MetaTrader uses not only HTML, but also the XML format. Trader can be especially interested in XPath and XSLT. These formats can serve as further steps in developing the idea of automating trading systems based on web standards. Support for CSS selectors in MQL is just the first step towards this goal.\
\
Translated from Russian by MetaQuotes Ltd.\
\
Original article: [https://www.mql5.com/ru/articles/5706](https://www.mql5.com/ru/articles/5706)\
\
**Attached files** \|\
\
\
[Download ZIP](https://www.mql5.com/en/articles/download/5706.zip "Download all attachments in the single ZIP archive")\
\
[html2css.zip](https://www.mql5.com/en/articles/download/5706/html2css.zip "Download html2css.zip")(35.94 KB)\
\
**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.\
\
This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.\
\
#### Other articles by this author\
\
- [Backpropagation Neural Networks using MQL5 Matrices](https://www.mql5.com/en/articles/12187)\
- [Parallel Particle Swarm Optimization](https://www.mql5.com/en/articles/8321)\
- [Custom symbols: Practical basics](https://www.mql5.com/en/articles/8226)\
- [Calculating mathematical expressions (Part 2). Pratt and shunting yard parsers](https://www.mql5.com/en/articles/8028)\
- [Calculating mathematical expressions (Part 1). Recursive descent parsers](https://www.mql5.com/en/articles/8027)\
- [MQL as a Markup Tool for the Graphical Interface of MQL Programs (Part 3). Form Designer](https://www.mql5.com/en/articles/7795)\
- [MQL as a Markup Tool for the Graphical Interface of MQL Programs. Part 2](https://www.mql5.com/en/articles/7739)\
\
**Last comments \|**\
**[Go to discussion](https://www.mql5.com/en/forum/312790)**\
(18)\
\
\
![Mahdi Ebrahimzadeh](https://c.mql5.com/avatar/2023/12/658fcccd-b8d5.png)\
\
**[Mahdi Ebrahimzadeh](https://www.mql5.com/en/users/pipcrop)**\
\|\
4 Mar 2025 at 00:56\
\
**MetaQuotes:**\
\
New article [Extracting structured data from HTML pages using CSS selectors](https://www.mql5.com/en/articles/5706) has been published:\
\
Author: [Stanislav Korotky](https://www.mql5.com/en/users/marketeer "marketeer")\
\
I downloaded this EA and tried to compile it and there are so many errors!\
\
[![](https://c.mql5.com/3/457/5327663318507__1.png)](https://c.mql5.com/3/457/5327663318507.png "https://c.mql5.com/3/457/5327663318507.png")\
\
for sure i am missing something, what it can be?!\
\
EDITED/UPDATED:\
\
I find-out that it is not FREE!! then why it is an article? a good way for advertising?\
\
for sure, owner has rights to sell or publish for free but looks like it is a marketing trick or I am just making some easy thing bold?!\
\
![Fernando Carreiro](https://c.mql5.com/avatar/2025/9/68d40cf8-38fb.png)\
\
**[Fernando Carreiro](https://www.mql5.com/en/users/fmic)**\
\|\
4 Mar 2025 at 01:04\
\
**[@Mahdi Ebrahimzadeh](https://www.mql5.com/en/users/pipcrop) [#](https://www.mql5.com/en/forum/312790#comment_56061012):** I downloaded this EA and tried to compile it and there are so many errors! for sure i am missing something, what it can be?!\
\
Given that MQL5 has changed a over the years, maybe the article's code is no longer completely valid and may need some adjustments.\
\
\
![Mahdi Ebrahimzadeh](https://c.mql5.com/avatar/2023/12/658fcccd-b8d5.png)\
\
**[Mahdi Ebrahimzadeh](https://www.mql5.com/en/users/pipcrop)**\
\|\
4 Mar 2025 at 01:11\
\
**Fernando Carreiro [#](https://www.mql5.com/en/forum/312790#comment_56061029):**\
\
Given that MQL5 has changed a over the years, maybe the article's code is no longer completely valid and may need some adjustments.\
\
Thanks Fernando for response, I find it for SALE! :)\
\
the reason it is not working as main part of code including classes is not there.\
\
![Lorentzos Roussos](https://c.mql5.com/avatar/2025/3/67c6d936-d959.jpg)\
\
**[Lorentzos Roussos](https://www.mql5.com/en/users/lorio)**\
\|\
4 Mar 2025 at 10:56\
\
**Mahdi Ebrahimzadeh [#](https://www.mql5.com/en/forum/312790#comment_56061190):**\
\
Thanks Fernando for response, I find it for SALE! :)\
\
the reason it is not working as main part of code including classes is not there.\
\
Also the site's code (that's served on the front end) must have changed too .\
\
What are you trying to extract?\
\
![Stanislav Korotky](https://c.mql5.com/avatar/2010/10/4CA7CFA0-1F0C.jpg)\
\
**[Stanislav Korotky](https://www.mql5.com/en/users/marketeer)**\
\|\
4 Mar 2025 at 13:14\
\
**Mahdi Ebrahimzadeh [#](https://www.mql5.com/en/forum/312790#comment_56061012):**\
\
I downloaded this EA and tried to compile it and there are so many errors!\
\
for sure i am missing something, what it can be?!\
\
EDITED/UPDATED:\
\
I find-out that it is not FREE!! then why it is an article? a good way for advertising?\
\
for sure, owner has rights to sell or publish for free but looks like it is a marketing trick or I am just making some easy thing bold?!\
\
Did you read the article?\
\
**_"If you need help in the setting up of CSS selectors for a specific web page, you can purchase WebDataExtractor (for MetaTrader 4, for MetaTrader 5) and receive recommendations as part of product support. However, the availability of source codes allows you to use the entire functionality and expand it if necessary. This is absolutely free."_**\
\
As for compilation - it works for me without an error. I'm attaching the source codes.\
\
![Library for easy and quick development of MetaTrader programs (part III). Collection of market orders and positions, search and sorting](https://c.mql5.com/2/35/MQL5-avatar-doeasy__2.png)[Library for easy and quick development of MetaTrader programs (part III). Collection of market orders and positions, search and sorting](https://www.mql5.com/en/articles/5687)\
\
In the first part, we started creating a large cross-platform library simplifying the development of programs for MetaTrader 5 and MetaTrader 4 platforms. Further on, we implemented the collection of history orders and deals. Our next step is creating a class for a convenient selection and sorting of orders, deals and positions in collection lists. We are going to implement the base library object called Engine and add collection of market orders and positions to the library.\
\
![Library for easy and quick development of MetaTrader programs (part II). Collection of historical orders and deals](https://c.mql5.com/2/35/MQL5-avatar-doeasy__1.png)[Library for easy and quick development of MetaTrader programs (part II). Collection of historical orders and deals](https://www.mql5.com/en/articles/5669)\
\
In the first part, we started creating a large cross-platform library simplifying the development of programs for MetaTrader 5 and MetaTrader 4 platforms. We created the COrder abstract object which is a base object for storing data on history orders and deals, as well as on market orders and positions. Now we will develop all the necessary objects for storing account history data in collections.\
\
![MTF indicators as the technical analysis tool](https://c.mql5.com/2/35/mtf-avatar.png)[MTF indicators as the technical analysis tool](https://www.mql5.com/en/articles/2837)\
\
Most of traders agree that the current market state analysis starts with the evaluation of higher chart timeframes. The analysis is performed downwards to lower timeframes until the one, at which deals are performed. This analysis method seems to be a mandatory part of professional approach for successful trading. In this article, we will discuss multi-timeframe indicators and their creation ways, as well as we will provide MQL5 code examples. In addition to the general evaluation of advantages and disadvantages, we will propose a new indicator approach using the MTF mode.\
\
![Studying candlestick analysis techniques (part III): Library for pattern operations](https://c.mql5.com/2/35/Pattern_I__4.png)[Studying candlestick analysis techniques (part III): Library for pattern operations](https://www.mql5.com/en/articles/5751)\
\
The purpose of this article is to create a custom tool, which would enable users to receive and use the entire array of information about patterns discussed earlier. We will create a library of pattern related functions which you will be able to use in your own indicators, trading panels, Expert Advisors, etc.\
\
[Need a reliable hosting solution for your robots?Contact your broker and find out about available Sponsored MetaTrader VPS offeringsLearn more![](https://www.mql5.com/ff/sh/0pw0dk81s56qy774z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=vljwvezfjkfbvviocwskggexlvgykvob&s=70cf8e354b9a125332533ffb65d7365abe8dde5b5c1ede9caac479a9e9df4f25&uid=&ref=https://www.mql5.com/en/articles/5706&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5083336387802962333)\
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