---
title: Parsing HTML with curl
url: https://www.mql5.com/en/articles/7144
categories: Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T19:39:52.287128
---

[![](https://www.mql5.com/ff/sh/9nb0c8df2rmwfn89z2/01.png) MetaTrader VPS vs regular cloud hosting services8 reasons why our solution is the best option for automated tradingRead](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/450486&a=dgmsfszgoedimaicrqqmagvqzpwuxkur&s=c59e3617ccf44fd54d4c50a03b44fd689ff7507b8fe4990c83772cc5419e627d&uid=&ref=https://www.mql5.com/en/articles/7144&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5070465032241747542)

MetaTrader 5 / Trading systems


### Introduction

Let us discuss the case, when data from a website cannot be obtained using ordinary requests. What can be done in this case? The very first
possible idea is to find a resource which can be accessed using GET or POST requests. Sometimes, such resources do not exist. For example,
this can concern the operation of some unique indicator or access to rarely updated statistics.

One may ask "What's the point?" A simple solution is to access the site page directly from an MQL script and read the already known number of
positions at the known page position. Then the received string can be further processed. This is one of the possible methods. But in this case
the MQL script code will be tightly bound to the HTML code of the page. What if the HTML code changes? That is why we need a parser which enables a
tree-like operation with an HTML document (the details will be discussed in a separate section). If we implement the parser in MQL, will this
be convenient and efficient in terms of performance? Can such a code be properly maintained? That is why the parsing functionality will be
implemented in a separate library. However, the parser will not solve all problems. It will perform the desired functionality. But what if
the site design changes radically and will use other class names and attributes? In this case we will need to change the search object or event
multiple objects. Therefore, one of our goals is to create the necessary code as quickly as possible and with the least effort. It will be
better if we use ready-made parts. This will enable the developer to easily maintain the code and quickly edit it in case of the above
situation.

We will select a website with not too large pages and will try to obtain interesting data from this site. The kind of data is not important in
this case, however let us try to create a useful tool. Of course, this data must be available to MQL scripts in the terminal. The program code
will be created as a standard DLL.

In this article we will implement the tool without asynchronous calls and multi-threading.

### Existing Solutions

The possibility to obtain data from the Internet and process it has always been interesting for developers. The mql5.com website features
several articles, which describe interesting and diverse approaches:

- [MQL5-RPC. Remote Procedure Calls from MQL5](https://www.mql5.com/en/articles/342). The article provides a
detailed analysis of how to receive data from the resources which can be accessed using the XML-RPC technology. The request is sent
directly from the MQL script.



- [Extracting structured data from HTML pages using CSS selectors](https://www.mql5.com/en/articles/5706).
The example shows how to obtain data and create an HTML code parser using MQL.
- [DIY multi-threaded asynchronous MQL5 WebRequest](https://www.mql5.com/en/articles/5337). The article
provides an interesting version of creating asynchronous requests without further analysis using MQL means.

I highly recommend that you read these articles.

### Defining the Task

We will experiment with the following site: [https://www.mataf.net/en/forex/tools/volatility](https://www.mql5.com/go?link=https://www.mataf.net/en/forex/tools/volatility "https://www.mataf.net/en/forex/tools/volatility").
As is clear from the name, the site features data on the volatility of currency pairs. The volatility is shown in three different units: pips,
US dollars and percent. The website page is not too bulky, thus it can be efficiently accepted and parsed to obtain the required values. The
preliminary study of source text shows that we will have to obtain values stored in separate table cells. Thus, let us break the main task into
two subtasks:

1. Getting and storing the page.
2. Parsing the obtained page to receive the document structure and to search for the required information in this structure. Data processing
    and passing to the client.

Let us start with the implementation of the first part. Do we need to save the obtained page as a file? In a really working version, it is
obvious that there is no need to save page. We need an adjustable cache which will be updated at certain intervals. There should be a
possibility to disable the use of the cache in special cases. For example, if an MQL indicator sends queries to the source page at every
tick, this indicator is likely to be banned from this site. The ban will happen even quicker if the data requesting script is running on
multiple currency pairs. In any case, the correctly designed tool will not send requests too often. Instead, it will send a request
once, will save the result in a file and will later request data from the file. Upon the expiration of the cache validity, the file will be
updated using a new request. This avoids too frequent requests.

In our case, we will not create this cache. By sending several training requests, we will not affect the site operation. Instead we can
focus on more important points. Further comments will be provided regarding saving of files to a disk, but in this case the data will be
saved in memory and will be passed to the second program block, i.e. to the parser. We will use simplified code whenever applicable,
making it understandable to beginners and still reflecting the essence of the main idea.

### Getting the HTML page of a third-party site

As mentioned earlier, one of the ideas is to utilize existing ready-to-use components and ready-made libraries. However, we still
need to ensure reliability and security of the entire system. The components will be selected based on their reputation. To obtain the
desired page, we will use the well-known open project

[**curl**](https://www.mql5.com/go?link=https://curl.haxx.se/libcurl/c/ "https://curl.haxx.se/libcurl/c/").

This project enables receiving and sending of files to almost any sources: http, https, ftp servers and many others. It supports setting
of login and password for authorization on the server, processing of redirects and timeouts. The project is provided with
comprehensive documentation describing all features of the project. Furthermore, it is an open-source cross-platform project,
which is definitely an advantage. There is another project which can implement the same functionality. It is the 'wget' project.
However, in this case curl is used for the following two reasons:

- curl can receive and send files, while wget only received files.
- wget is only available as the wget.exe console application.

The inability to send files by wget is not relevant for our task, because we only need to receive an HTML page. However, if we get
acquainted with curl, later we will be able to use it for other task, which may require data sending.

A more serious disadvantage relates to the fact, that it is only available as wget.exe utility without any libraries like
wget.dll, wget.lib.

- In this case, in order to use wget from a dll connected to MetaTrader, we will need to create a separate process, which is time
and effort consuming.
- Data obtained via wget can only be passed as a file, which is inconvenient for us because we decided to use cache instead.


In these terms, curl is more convenient. In addition to the console application curl.exe, it provides libraries:
libcurl-x64.dll and libcurl-x64.lib. Thus, we can include curl to our program without any additional development process
and work with the memory buffer rather than create a separate file with curl operation results. Curl is also available as
source code, but creation of libraries based on the source code can be time consuming. Therefore, the attached archive
includes the created libraries, dependencies and all include files required for operation.

### Creating a Library

Open Visual Studio (I used Visual Studio 2017) and create a simple dll. Let us call the project GetAndParse — the resulting
library will have the same name. Create two folders in the project folder: "lib" and "include". These two folders will be used
for connecting third-party libraries. Copy libcurl-x64.lib to the 'lib' folder and create the 'curl' folder under the
'include' folder. Copy all include files to 'curl'. Open the menu: "Project -> GetAndParse Properties". In the left part
of the dialog box, open "C/C++" and select "General". In the right part, select "Additional Include Directories", click on
down arrow and select "Edit". In the new dialog box, open the leftmost button in the upper row "New Line". This command adds an
editable line in the below list. By clicking the button on the right, select the newly created "include" folder and click "OK".

Unwrap "Linker", select General, and then click "Additional Library Directories" on the right. By repeating the same actions,
add the created "lib" folder.

From the same list, select the "input" line and click "Additional Dependencies" on the right. Add the "libcurl-x64.lib" name in
the upper box.

We also need to add libcurl-x64.dll. Copy this file along with the encryption support libraries to the "debug" and "release"
folders.



The attached archive includes the required files, which are located in appropriate folders. The attached project properties
have also been modified, thus you will not need to perform any additional actions.

### Class for Obtaining HTML Pages

In the project, create the CCurlExec class, which will implement the main task. It will interact with curl, therefore connect
it as follows:

```
#include <curl\curl.h>
```

This can be done in the CCurlExec.cpp file, but I preferred to include it in stdafx.h

Define a type alias for the callback function, which is used for saving the received data:

```
typedef size_t (*callback)(void*, size_t, size_t, void*);
```

Create simple structures to save the received data in memory:

```
typedef struct MemoryStruct {
        vector<char> membuff;
        size_t size = 0;
} MSTRUCT, *PMSTRUCT;
```

... and in a file:

```
typedef struct FileStruct {
        std::string CalcName() {
                char cd[MAX_PATH];
                char fl[MAX_PATH];
                srand(unsigned(std::time(0)));
                ::GetCurrentDirectoryA(MAX_PATH, cd);
                ::GetTempFileNameA(cd, "_cUrl", std::rand(), fl);
                return std::string(fl);
        }
        std::string filename;
        FILE* stream = nullptr;
} FSTRUCT, *PFSTRUCT;
```

I think these structures do not need explanation. The tool should be able to store information in memory. For this purpose, we
provide a buffer in the MSTRUCT structure and its size.

To store information as a file (we will implement this possibility in the project, though in our case we will only use storing in
memory), add the file name getting function to the FSTRUCT structure. For this purpose, use Windows API to work with temporary
files.

Now create a couple of callback functions to populate the described structures. Method to fill the MSTRUCT type structure:

```
size_t CCurlExec::WriteMemoryCallback(void * contents, size_t size, size_t nmemb, void * userp)
{
        size_t realsize = size * nmemb;
        PMSTRUCT mem = (PMSTRUCT)userp;
        vector<char>tmp;
        char* data = (char*)contents;
        tmp.insert(tmp.end(), data, data + realsize);
        if (tmp.size() <= 0) return 0;
        mem->membuff.insert(mem->membuff.end(), tmp.begin(), tmp.end() );
        mem->size += realsize;
        return realsize;
}
```

We will not provide here the second method for saving data in a file, which is similar to the first one. The function signatures
are taken from the documentation on the curl project website.

These two methods will be used as "default functions". They will be used in case the developer will not provide his own methods for
these purposes.

The idea of these methods is very simple. The following is passed in method parameters: information about received data size, a
pointer to source, i.e. internal curl buffer, and recipient, the MSTRUCT structure. After some preliminary
conversions, the recipient structure fields are filled.

And finally, the method that performs the main actions: it receives an HTML page and fills out a structure of the MSTRUCT type
using the received data:

```
bool CCurlExec::GetFiletoMem(const char* pUri)
{
        CURL *curl;
        CURLcode res;
        res  = curl_global_init(CURL_GLOBAL_ALL);
        if (res == CURLE_OK) {
                curl = curl_easy_init();
                if (curl) {
                        curl_easy_setopt(curl, CURLOPT_URL, pUri);
                        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);
                        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0L);
                        curl_easy_setopt(curl, CURLOPT_ERRORBUFFER, m_errbuf);
                        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 20L);
                        curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 60L);
                        curl_easy_setopt(curl, CURLOPT_USERAGENT, "libcurl-agent/1.0");
                        curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L); //redirects
#ifdef __DEBUG__
                        curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L);
#endif
                        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteMemoryCallback);
                        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &m_mchunk);
                        res = curl_easy_perform(curl);
                        if (res != CURLE_OK) PrintCurlErr(m_errbuf, res);
                        curl_easy_cleanup(curl);
                }// if (curl)
                curl_global_cleanup();
        } else PrintCurlErr(m_errbuf, res);
        return (res == CURLE_OK)? true: false;
}
```

Pay attention to important aspects of curl operation. Firstly, two initializations are performed, as a result of which the user receives a
pointer to the curl "core" and to its "handle", which is used in further calls. Further connection is configured, which can involve a lot of
settings. In this case we determine the connection address, the need to check the certificates, specify the buffer into which errors will be
written, determine the timeout duration, "user-agent" header, the need to handle redirects, specify the function that will be called to
process received data (the above described default method) and the object to store the data. Setting of the CURLOPT\_VERBOSE option

enables the display of detailed information about
operations being performed, which can be useful for debugging purposes. Once all the options are specified, the curl function
curl\_easy\_perform is called. It performs the main operation. After that data is cleared.

Let us add one more general method:

```
bool CCurlExec::GetFile(const char * pUri, callback pFunc, void * pTarget)
{
        CURL *curl;
        CURLcode res;
        res = curl_global_init(CURL_GLOBAL_ALL);
        if (res == CURLE_OK) {
                curl = curl_easy_init();
                if (curl) {
                        curl_easy_setopt(curl, CURLOPT_URL, pUri);
                        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);
                        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0L);
                        curl_easy_setopt(curl, CURLOPT_ERRORBUFFER, m_errbuf);
                        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 20L);
                        curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 60L);
                        curl_easy_setopt(curl, CURLOPT_USERAGENT, "libcurl-agent/1.0");
                        curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
#ifdef __DEBUG__
                        curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L);
#endif
                        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, pFunc);
                        curl_easy_setopt(curl, CURLOPT_WRITEDATA, pTarget);
                        res = curl_easy_perform(curl);
                        if (res != CURLE_OK) PrintCurlErr(m_errbuf, res);
                        curl_easy_cleanup(curl);
                }// if (curl)
                curl_global_cleanup();
        }       else PrintCurlErr(m_errbuf, res);

        return (res == CURLE_OK) ? true : false;
}
```

This method enables the developer to use a custom callback function to process the received data (the pFunc parameter) and a custom object to
store this data (the pTarget parameter). Thus, an HTML page can be easily saved, for example, as a csv file.

Let us see how information is saved to a file without going into detail. The appropriate callback function was mentioned earlier, along with
the helper object FSTRUCT with a code for selecting the file name. However, in most cases the work does not end there. To obtain the file name,
it can either be set in advance (in this case you should check whether the file with this name exists, before writing), or the library can be
allowed to get a readable and meaningful file name. Such name should be obtained from the actual address, at which data was read after
processing of redirects. The following method shows how an actual address is obtained

### ``` bool CCurlExec::GetFiletoFile(const char * pUri) ```

The full method code is available in the archive. The tools provided in 'curl' are used for address parsing:

```
std::string CCurlExec::ParseUri(const char* pUri) {
#if !CURL_AT_LEAST_VERSION(7, 62, 0)
#error "this example requires curl 7.62.0 or later"
        return  std::string();
#endif
        CURLU *h  = curl_url();
        if (!h) {
                cerr << "curl_url(): out of memory" << endl;
                return std::string();
        }
        std::string szres{};
        if (pUri == nullptr) return  szres;
        char* path;
        CURLUcode res;
        res = curl_url_set(h, CURLUPART_URL, pUri, 0);
        if ( res == CURLE_OK) {
                res = curl_url_get(h, CURLUPART_PATH, &path, 0);
                if ( res == CURLE_OK) {
                        std::vector <string> elem;
                        std::string pth = path;
                        if (pth[pth.length() - 1] == '/') {
                                szres = "index.html";
                        }
                        else {
                                Split(pth, elem);
                                cout << elem[elem.size() - 1] << endl;
                                szres =  elem[elem.size() - 1];
                        }
                        curl_free(path);
                }// if (curl_url_get(h, CURLUPART_PATH, &path, 0) == CURLE_OK)
        }// if (curl_url_set(h, CURLUPART_URL, pUri, 0) == CURLE_OK)
        return szres;
}
```

You see that curl correctly extracts "PATH" from the uri and checks if the PATH ends with the '/' character. If it does, the file name should be
"index.html". If not, "PATH" is split into separate elements, while the file name will be the last element in the resulting list.

Both of the above methods are implemented in the project, though generally the task of data saving to a file is beyond the scope of this article.

In addition to the described methods, the CCurlExec class provides two elementary methods for receiving access to the memory buffer, to
which data received from the network was saved. The data can be presented as

```
std::vector<char>
```

or in the following form:

```
std::string
```

depending on further html parser selection. There is no need to study other methods and properties of the CCurlExec class as they are not applicable
within our project.

To conclude this chapter, I would like to add a few remarks. The curl library is thread secure. In this case it is used synchronously, for which
methods of curl\_easy\_init type are utilized. All curl functions having "easy" in their name are only used synchronously. The asynchronous
use of the library is provided via functions which include "multi" in their names. For example, curl\_easy\_init has an asynchronous
analogous function curl\_multi\_init. Work with asynchronous functions in curl is not very complicated, but it involves lengthy calling
code. Therefore, we will not consider asynchronous operation now but we may get back to it later.

### HTML Parsing Class

Let us try to find a ready component to perform this task. There are many different components available. When selecting a component, use the
same criteria as in the previous chapter. In this case, the preferred option is the Gumbo project from Google. It is available on github. The
appropriate link is available in the attached project archive. You can compile the project for yourself, which may be easier than using
curl. Anyway the attached project includes all necessary files:

- gumbo.lib is available in the lib project folder
- gumbo.dll is included in debug and release

Once again open the menu "Project -> GetAndParse Properties". Unwrap "Linker", select "input", and then select "Additional
Dependencies" on the right. Add the "gumbo.lib" name in the upper box.

In addition, in the earlier created "include" folder create the "gumbo" folder and copy all include files to it. Make an entry in the
stdafx.h file:

```
#include <gumbo\gumbo.h>
```

Two words about gumbo. This is the html5 code parser in C++. Pros:

- Full correspondence to HTML5 specification.
- Resistance to incorrect input data.
- Simple API, which can be called from other languages.
- Passes all html5lib-0.95 tests.
- Tested on more than two and a half billion pages from the Google index.

Cons:

- The performance is not very high.

The parser only builds the page tree and does nothing else. This can also be treated as a disadvantage. The developer can then
work with the tree using preferred methods. There are resources that provide wrappers for this parser, but we will not use
them. Our aim is not to "improve" the parser, so we will use it as it is. It will build a tree, in which we will search for desired
data. Work with a component is simple:

```
        GumboOutput* output = gumbo_parse(input);
//      ... do something with output ...
        gumbo_destroy_output(&options, output);
```

We call a function, passing to it a pointer to a buffer with the source HTML data. The function creates a parser with which the
developer works. The developer calls the function and frees resources.



Let us proceed to this task and start with the examining of html code of the desired page. The purpose is obvious - we need to
understand what to look for and where the necessary data is located. Open the link
\_https://www.mataf.net/en/forex/tools/volatility and look at the source code of the page. Volatility data are contained
in the table <table id="volTable" ... This data is enough to be able to find the table in the tree. Obviously, we need to
receive volatility for a particular currency pair. The attributes of table rows contain currency symbol names: <tr
id="row\_AUDCHF" class="data\_volat" name="AUDCHF"... Using this data, the desired row can be easily found. Each row
consists of five columns. We do not need the first two columns, while the three others contain the required data. Let us choose a
column, receive text data, convert them to double and return to the client. To make the process clearer, let us split the data
search into three stages:

1. Find the table by its identifier ("volTable").
2. Find the row using its identifier ("row\_Currency Pair Name").
3. Find the volatility value in one of the last three columns and return the found value.

Let's start writing the code. Create the **CVolatility** class in the project. The parser library has already
been connected, so no additional actions are required here. As you remember, volatility in the desired table was shown in
three columns, in three different ways. Therefore, let us create the appropriate enumeration for selecting one of them:


```
typedef enum {
        byPips = 2,
        byCurr = 3,
        byPerc = 4
} VOLTYPE;
```

I think this part is clear and does not need any additional explanation. It is implemented as the selection of the column
number.

Next, create a method that returns the volatility value:

```
double CVolatility::FindData(const std::string& szHtml, const std::string& pair, VOLTYPE vtype)
{
        if (pair.empty()) return -1;
        m_pair = pair;
        TOUPPER(m_pair);
        m_column = vtype;
        GumboOutput* output = gumbo_parse(szHtml.c_str() );
        double res = FindTable(output->root);
        const GumboOptions mGumboDefaultOptions = { &malloc_wrapper, &free_wrapper, NULL, 8, false, -1, GUMBO_TAG_LAST, GUMBO_NAMESPACE_HTML };
        gumbo_destroy_output(&mGumboDefaultOptions, output);
        return res;
}// void CVolatility::FindData(char * pHtml)
```

Call the method with the following arguments:

- szHtml — reference to a buffer with received data in html format.
- pair — name of the currency pair for which volatility is searched
- vtype — volatility type, the table column number

The method returns the volatility value or -1 in case of error.

Thus, the operation starts with table search, from the very beginning of the tree. This search is implemented by the
following closed method:

```
double CVolatility::FindTable(GumboNode * node) {
        double res = -1;
        if (node->type != GUMBO_NODE_ELEMENT) {
                return res;
        }
        GumboAttribute* ptable;
        if ( (node->v.element.tag == GUMBO_TAG_TABLE)                          && \
                (ptable = gumbo_get_attribute(&node->v.element.attributes, "id") ) && \
                (m_idtable.compare(ptable->value) == 0) )                          {
                GumboVector* children = &node->v.element.children;
                GumboNode*   pchild = nullptr;
                for (unsigned i = 0; i < children->length; ++i) {
                        pchild = static_cast<GumboNode*>(children->data[i]);
                        if (pchild && pchild->v.element.tag == GUMBO_TAG_TBODY) {
                                return FindTableRow(pchild);
                        }
                }//for (int i = 0; i < children->length; ++i)
        }
        else {
                for (unsigned int i = 0; i < node->v.element.children.length; ++i) {
                        res = FindTable(static_cast<GumboNode*>(node->v.element.children.data[i]));
                        if ( res != -1) return res;
                }// for (unsigned int i = 0; i < node->v.element.children.length; ++i)
        }
        return res;
}//void CVolatility::FindData(GumboNode * node, const std::string & szHtml)
```

The method is recursively called until an element satisfying the following two requirements is found:

1. This must be a table.
2. Its "id" must be "volTable".

If such an element is not found, the method will return -1. Otherwise, the method will return the value, which will return a
similar method that searches for a table row:


```
double CVolatility::FindTableRow(GumboNode* node) {
        std::string szRow = "row_" + m_pair;
        GumboAttribute* prow       = nullptr;
        GumboNode*      child_node = nullptr;
        GumboVector* children = &node->v.element.children;
        for (unsigned int i = 0; i < children->length; ++i) {
                child_node = static_cast<GumboNode*>(node->v.element.children.data[i]);
                if ( (child_node->v.element.tag == GUMBO_TAG_TR) && \
                         (prow = gumbo_get_attribute(&child_node->v.element.attributes, "id")) && \
                        (szRow.compare(prow->value) == 0)) {
                        return GetVolatility(child_node);
                }
        }// for (unsigned int i = 0; i < node->v.element.children.length; ++i)
        return -1;
}// double CVolatility::FindVolatility(GumboNode * node)
```

Once a table row with the id = "row\_PairName" is found, search is completed by calling the method, which reads the cell value in a
certain column of the found row:


```
double CVolatility::GetVolatility(GumboNode* node)
{
        double res = -1;
        GumboNode*      child_node = nullptr;
        GumboVector* children = &node->v.element.children;
        int j = 0;
        for (unsigned int i = 0; i < children->length; ++i) {
                child_node = static_cast<GumboNode*>(node->v.element.children.data[i]);
                if (child_node->v.element.tag == GUMBO_TAG_TD && j++ == (int)m_column) {
                        GumboNode* ch = static_cast<GumboNode*>(child_node->v.element.children.data[0]);
                        std::string t{ ch->v.text.text };
                        std::replace(t.begin(), t.end(), ',', '.');
                        res = std::stod(t);
                        break;
                }// if (child_node->v.element.tag == GUMBO_TAG_TD && j++ == (int)m_column)
        }// for (unsigned int i = 0; i < children->length; ++i) {
        return res;
}
```

Please note that comma is used for data separator in the table, instead of the point. Therefore the code has a few lines to solve
this issue. Similar to previous cases, the method returns -1 in case of an error or the volatility value in case of success.

However, this approach has a disadvantage. The code is still strongly tied to data, which the user cannot affect, though the
parser releases this dependence to some extent. Therefore, if the website design changes significantly, the developer
will have to rewrite the whole part related to tree search. Anyway the search procedure is simple, and the related several
functions can be easily edited.

Other CVolatility class members are available in the attached archive. We will not consider them within this article.

### Combining all together

The main code is ready. Now we need to put everything together and design a function, which will create objects and perform calls
in the proper sequence. The following code is inserted into the GetAndParse.h file:

```
#ifdef GETANDPARSE_EXPORTS
#define GETANDPARSE_API extern "C" __declspec(dllexport)
#else
#define GETANDPARSE_API __declspec(dllimport)
#endif

GETANDPARSE_API double GetVolatility(const wchar_t* wszPair, UINT vtype);
```

It already contains macro definition, which we have slightly modified to enable this function call by mql. See the explanation of
why this is done at this

[link](https://www.mql5.com/en/articles/5798).

The function code is written in the GetAndParse.cpp file:

```
const static char vol_url[] = "https://www.mataf.net/ru/forex/tools/volatility";

GETANDPARSE_API double GetVolatility(const wchar_t*  wszPair, UINT vtype) {
        if (!wszPair) return -1;
        if (vtype < 2 || vtype > 4) return -1;

        std::wstring w{ wszPair };
        std::string s(w.begin(), w.end());

        CCurlExec cc;
        cc.GetFiletoMem(vol_url);
        CVolatility cv;
        return cv.FindData(cc.GetBufferAsString(), s, (VOLTYPE)vtype);
}
```

Is it a good idea to hard code the page address? Why can't it be implemented as an argument of GetVolatility function call? It makes no
sense, because the information search algorithm returned by the parser is forcedly bound to the HTML page elements. Therefore it
is an address-specific library. This method should not be always used, it is an appropriate application in our case.

### Library Compilation and Installation

Build the library in a usual way. Take all dlls from the Release folder, including: GETANDPARSE.dll, gumbo.dll,
libcrypto-1\_1-x64.dll, libcurl-x64.dll, and libssl-1\_1-x64.dll, and copy them to the 'Libraries' folder of the terminal.
Thus the library has been installed.



### Library Usage Tutorial Script

This is a simple script:

```
#property copyright "Copyright 2019, MetaQuotes Software Corp."
#property link      "https://www.mql5.com"
#property version   "1.00"
#property script_show_inputs

#import "GETANDPARSE.dll"
double GetVolatility(string wszPair,uint vtype);
#import
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
enum ReqType
  {
   byPips    = 2, //Volatility by Pips
   byCurr    = 3, //Volatility by Currency
   byPercent = 4  //Volatility by Percent
  };

input string  PairName="EURUSD";
input ReqType tpe=byPips;
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+

void OnStart()
  {
   double res=GetVolatility(PairName,tpe);
   PrintFormat("Volatility for %s is %.3f",PairName,res);
  }
```

The script seems to need no further explanation. The script code is attached below.

### Conclusion

We have discussed a method for parsing the page HTML in the most simplified form. The library is made of ready components. The code has been
greatly simplified to help beginners in understanding the idea. The main disadvantage of this solution is the synchronous execution. The
script will not take control until the library receives the HTML page and processes it. This may take time, which is unacceptable for
indicators and Expert Advisors. Another approach is required for use in such applications. We will try to find better solutions in further
articles.

### Programs used in the article

| # | Name | Type | Description |
| --- | --- | --- | --- |
| 1 | GetVolat.mq5 | Script | The script which receives volatility data. |
| 2 | GetAndParse.zip | Archive | The source code of the library and of the test console application |

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/7144](https://www.mql5.com/ru/articles/7144)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/7144.zip "Download all attachments in the single ZIP archive")

[GetVolat.mq5](https://www.mql5.com/en/articles/download/7144/getvolat.mq5 "Download GetVolat.mq5")(1.45 KB)

[GetAndParse.zip](https://www.mql5.com/en/articles/download/7144/getandparse.zip "Download GetAndParse.zip")(4605.35 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [MVC design pattern and its application (Part 2): Diagram of interaction between the three components](https://www.mql5.com/en/articles/10249)
- [MVC design pattern and its possible application](https://www.mql5.com/en/articles/9168)
- [Using cryptography with external applications](https://www.mql5.com/en/articles/8093)
- [Building an Expert Advisor using separate modules](https://www.mql5.com/en/articles/7318)
- [Arranging a mailing campaign by means of Google services](https://www.mql5.com/en/articles/6975)
- [A DLL for MQL5 in 10 Minutes (Part II): Creating with Visual Studio 2017](https://www.mql5.com/en/articles/5798)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/324923)**
(16)


![Denis Sartakov](https://c.mql5.com/avatar/2020/8/5F49477B-C850.jpg)

**[Denis Sartakov](https://www.mql5.com/en/users/denissergeev)**
\|
12 Dec 2019 at 21:23

**Andrei Novichkov:**

Unfortunately I don't remember where I got \*.lib in my project, I must have compiled it myself. Try to search on their site. Here, I have it in stock, but again, where it came from is a big question. It should be all from their site, but you should check it anyway

Yeah, thanks, yeah, it seems to be all there, but it's weird,

I downloaded the sources with this version - the [project](https://www.mql5.com/en/articles/7863 "Article: Projects allow you to create profitable trading robots! But it's not accurate ") in VC 2010 does not compile, a lot of errors - some .h files are not found....

![Andrei Novichkov](https://c.mql5.com/avatar/2016/11/58342001-4AC3.png)

**[Andrei Novichkov](https://www.mql5.com/en/users/andreifx60)**
\|
13 Dec 2019 at 12:32

**Denis Sartakov:**

yeah, thanks, yeah, it seems to be all there, but it's weird,

I downloaded the sources with this version - the project in VC 2010 does not compile, a lot of errors - some .h files are not found....

This thing is not easy to build, so don't give up ) There are instructions on the web like "How to build libcurl for windows".

![Denis Sartakov](https://c.mql5.com/avatar/2020/8/5F49477B-C850.jpg)

**[Denis Sartakov](https://www.mql5.com/en/users/denissergeev)**
\|
13 Dec 2019 at 19:42

**Andrei Novichkov:**

This thing is not easy to build, so don't give up ) There are tutorials on the web like "How to build libcurl under windows".

ha, ha, here is a complete instruction how to build this monster from source.

the task is really not easy !

[https://curl.haxx.se/libcurl/c/Using-libcurl-with-SSH-support-in-Visual-Studio-2010.pdf](https://www.mql5.com/go?link=https://curl.haxx.se/libcurl/c/Using-libcurl-with-SSH-support-in-Visual-Studio-2010.pdf "https://curl.haxx.se/libcurl/c/Using-libcurl-with-SSH-support-in-Visual-Studio-2010.pdf")

at the first step there is an ambush ! ActivePerl for 32 bit can't be downloaded so easily !

![Andrei Novichkov](https://c.mql5.com/avatar/2016/11/58342001-4AC3.png)

**[Andrei Novichkov](https://www.mql5.com/en/users/andreifx60)**
\|
14 Dec 2019 at 10:36

**Denis Sartakov:**

ha, ha, here's the full instructions on how to build this monster from source.

the task is really hard !

[https://curl.haxx.se/libcurl/c/Using-libcurl-with-SSH-support-in-Visual-Studio-2010.pdf](https://www.mql5.com/go?link=https://curl.haxx.se/libcurl/c/Using-libcurl-with-SSH-support-in-Visual-Studio-2010.pdf "https://curl.haxx.se/libcurl/c/Using-libcurl-with-SSH-support-in-Visual-Studio-2010.pdf")

at the first step there is an ambush ! ActivePerl for 32 bit is not so easy to download

Look for one without all this Pel stuff. You should have it. And you will need all kinds of oupenssl libraries, as far as I remember. Nothing, it's not easy, but you'll learn a lot while trying ).


![Maxim Kuznetsov](https://c.mql5.com/avatar/2016/1/56935A91-AF51.png)

**[Maxim Kuznetsov](https://www.mql5.com/en/users/nektomk)**
\|
14 Dec 2019 at 11:32

**Denis Sartakov:**

ha, ha, here's the full instructions on how to build this monster from source.

the task is really hard !

[https://curl.haxx.se/libcurl/c/Using-libcurl-with-SSH-support-in-Visual-Studio-2010.pdf](https://www.mql5.com/go?link=https://curl.haxx.se/libcurl/c/Using-libcurl-with-SSH-support-in-Visual-Studio-2010.pdf "https://curl.haxx.se/libcurl/c/Using-libcurl-with-SSH-support-in-Visual-Studio-2010.pdf")

at the first step there is an ambush ! ActivePerl for 32 bit is not so easy to download

Why build it? Without programming experience it's a lot of fun...

take a ready DLL, there are 32 bit and 64 bit DLLs on the site.

![Developing Pivot Mean Oscillator: a novel Indicator for the Cumulative Moving Average](https://c.mql5.com/2/37/PMO_200x200.png)[Developing Pivot Mean Oscillator: a novel Indicator for the Cumulative Moving Average](https://www.mql5.com/en/articles/7265)

This article presents Pivot Mean Oscillator (PMO), an implementation of the cumulative moving average (CMA) as a trading indicator for the MetaTrader platforms. In particular, we first introduce Pivot Mean (PM) as a normalization index for timeseries that computes the fraction between any data point and the CMA. We then build PMO as the difference between the moving averages applied to two PM signals. Some preliminary experiments carried out on the EURUSD symbol to test the efficacy of the proposed indicator are also reported, leaving ample space for further considerations and improvements.

![Library for easy and quick development of MetaTrader programs (part XV): Collection of symbol objects](https://c.mql5.com/2/36/MQL5-avatar-doeasy__10.png)[Library for easy and quick development of MetaTrader programs (part XV): Collection of symbol objects](https://www.mql5.com/en/articles/7041)

In this article, we will consider creation of a symbol collection based on the abstract symbol object developed in the previous article. The abstract symbol descendants are to clarify a symbol data and define the availability of the basic symbol object properties in a program. Such symbol objects are to be distinguished by their affiliation with groups.

![MQL5 Cookbook: Trading strategy stress testing using custom symbols](https://c.mql5.com/2/37/custom_stress_test.png)[MQL5 Cookbook: Trading strategy stress testing using custom symbols](https://www.mql5.com/en/articles/7166)

The article considers an approach to stress testing of a trading strategy using custom symbols. A custom symbol class is created for this purpose. This class is used to receive tick data from third-party sources, as well as to change symbol properties. Based on the results of the work done, we will consider several options for changing trading conditions, under which a trading strategy is being tested.

![A New Approach to Interpreting Classic and Hidden Divergence. Part II](https://c.mql5.com/2/37/new_approach_divergence.png)[A New Approach to Interpreting Classic and Hidden Divergence. Part II](https://www.mql5.com/en/articles/5703)

The article provides a critical examination of regular divergence and efficiency of various indicators. In addition, it contains filtering options for an increased analysis accuracy and features description of non-standard solutions. As a result, we will create a new tool for solving the technical task.

[![](https://www.mql5.com/ff/sh/ub4fqgrk4rkv8gz9z2/01.png)![](https://www.mql5.com/ff/sh/ub4fqgrk4rkv8gz9z2/02.png)Explore your trading for freeUpdated statistics in MetaTrader 5 will help you to thoroughly evaluate results and reduce risksLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/454106&a=bkbqgaxtrafeuegfvjisjjwjohagrvnr&s=25c5856d7857fc6b6db7cffb15ae4ce40fd19d1ab594d8a900ad65673d9ffa0e&uid=&ref=https://www.mql5.com/en/articles/7144&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5070465032241747542)

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