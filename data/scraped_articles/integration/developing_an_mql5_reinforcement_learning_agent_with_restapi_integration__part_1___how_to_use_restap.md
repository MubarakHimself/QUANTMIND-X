---
title: Developing an MQL5 Reinforcement Learning agent with RestAPI integration (Part 1): How to use RestAPIs in MQL5
url: https://www.mql5.com/en/articles/13661
categories: Integration, Machine Learning
relevance_score: 15
scraped_at: 2026-01-22T17:10:12.969899
---

[![](https://www.mql5.com/ff/sh/qv94j0cd8n2n55z9z2/01.png)![](https://www.mql5.com/ff/sh/qv94j0cd8n2n55z9z2/02.png)Boost your trading experienceRead our book "MQL5 Programming for Traders"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/book%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.algobook%26utm_content=visit.page%26utm_campaign=algobook.promo.04.2024&a=heclgjpfbvfghpmyaciuaesdtswflupo&s=4255fbe1b8cbc4d1b40afbaebf4235e5ace8b5103cba60d996897a03d588556f&uid=&ref=https://www.mql5.com/en/articles/13661&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5048940661903827395)

MetaTrader 5 / Integration


### Introduction

In programming and system development, communication between different applications is crucial. APIs (Application Programming Interfaces) play an important role in this context as they allow systems to communicate and exchange data efficiently. In this article, we will specifically talk about RestAPI, which has changed the way systems interact on the web.

RestAPI (Representational State Transfer Application Programming Interface) is a set of rules that define how systems should communicate on the Internet. It is based on simple and scalable architectural principles and uses concepts such as resources and identifiers to manage data. Actually, it is a method of communication that allows applications to request and send information in an organized manner.

The REST architecture emerged in the 2000s thanks to an article by Roy Fielding that outlined principles for building reliable and scalable web systems. Since then, RestAPIs have gained a lot of popularity. This is due to their simplicity compared to earlier protocols such as SOAP (Simple Object Access Protocol).

SOAP, based on XML, was very popular but was known to be complex and require large amounts of data for simple transactions. RestAPIs have changed the game significantly by becoming an easier and more versatile alternative for system-to-system communication.

RestAPIs are versatile and widely used in various systems and applications. Its simplicity and adherence to REST principles allow the creation of scalable, integrable, and easy to maintain systems. They are widely used in various industries, including the financial field.

If you compare them with the SOAP protocol, it becomes clear why RestAPIs have gained a strong position. While SOAP was inflexible and cumbersome, RestAPIs are lightweight, easy to use, and ideal for modern systems that need efficient communication. Moreover, they can be implemented in almost any programming language, making them an ideal choice for developers around the world.

### **What is an API?**

![whats is an api](https://c.mql5.com/2/59/what-is-an-api.png)

In an increasingly digital world, the term "API" is widely used, but what exactly does it mean? The abbreviation "API" stands for "Application Programming Interface". It is a fundamental concept in software development and systems integration.

In essence, an API is a set of rules and protocols that allow different software to communicate with each other. It acts as a bridge, allowing one program to use the functionality of another without having to understand all the internal details of that program. APIs are essential for building applications because they allow developers to use third-party resources or even internal systems.

Let's explore the main aspects of the API:

1.Communication between applications:

One of the most important functions of an API is the facilitation of communication between different applications or systems. Imagine that you are developing a weather forecasting app and want to include real-time weather forecast in it. Instead of building an entire weather forecasting system from scratch, you can use the API of an existing weather service. The API will provide a set of instructions and methods that your app can use to request up-to-date weather data.

2\. Abstraction of complexity:

APIs allow you to abstract the underlying complexity of the system. This means that when using an API, you don't need to understand all the technical details of the system or service you're interacting with. You just need to know how to use the API. This makes development more efficient and allows developers to focus on their tasks rather than reinventing the wheel every time.

3\. API types:

As mentioned above, there are several types of APIs. Web APIs are widely used on the internet and are based on web protocols such as HTTP. Local APIs provide access to operating system or middleware resources. Program APIs are used to access remote program components. There are several technologies and standards to implement them, such as gRPC and REST. RESTful APIs follow the principles of REST, while SOAP APIs are XML-based and more complex. For example, the GraphQL API is known for its flexibility in allowing clients to query specific data.

4\. Real world examples:

Let look at some real world examples to understand how the APIs are used. Social networks such as Facebook or Twitter offer APIs that allow developers to integrate content sharing or authentication functionality into their own applications. Payment services such as PayPal offer APIs that enable financial transactions. Even some map services, such as Google Maps, have APIs that allow interactive maps to be included in applications.

5\. Security and Authentication:

APIs often include security mechanisms to ensure that only authorized applications can access resources. This may include API key generation, token authentication, or other sensitive data protection methods.

### **What is RestAPI?**

**![api diagram](https://c.mql5.com/2/59/apidiagram.png)**

Before we entered the world of RestAPIs, software systems typically communicated through stricter and more complicated protocols. Communication between systems was not so fluid, making exchange of information between different applications a challenging task.

Then came the era of REST, which brought with a revolutionary approach to communication between systems. REST which stands for _Representational State Transfer_ was first introduced by Roy Fielding in 2000. This innovative approach to software architecture became the basis of RestAPI. REST defines a set of principles that guide how systems should interact, including the use of URLs (Uniform Resource Locators) to identify resources and HTTP methods to interact with them.

RestAPIs play a fundamental role in how applications and systems communicate in the digital age. They allow a system to request information or actions from another system, making data the exchange and operations more efficient and flexible.

Let's consider a simple example. Suppose you use a weather forecast app on our smartphone. This app should show you the latest information about temperature and weather conditions in your current location. However, it does not generate this information alone, but receives it from a remote server that collects weather information from various sources.

This is where RestAPI is helpful. When we open the weather app, it sends a request to the remote server via RestAPI. This request, usually a GET request, queries information about the weather conditions at your location. The server processes the request and produces a response, usually in JSON or XML format, with the requested weather data. The application then uses this data to display it in the appropriate interface. What makes RestAPIs so powerful is the standardization of interaction. Systems can communicate reliably: in other words, they know that if they follow REST principles, information will be transmitted in a reliable manner.

Examples of use:

In social media applications. When you upload a photo to your favorite social platform, RestAPI takes care of transmitting data to the server and then makes that photo available for other users. This interaction is based on requests and responses via RestAPI.

**Key differences between API and REST API**

The API simplifies the process of integrating many applications by providing readily available code and information channels to help developers build robust digital solutions. The API acts as a mediator between applications, facilitating interaction between them. However, due to different application architectures, APIs can be of different types, such as programmatic, local, web, or REST APIs.

The API makes it possible to connect computers or computer programs. Basically, it is a software interface that provides services to other programs to enhance the necessary functionality. In recent years, APIs have gained popular in the market as almost all web applications use them. For example, whenever you check the weather or book a travel ticket on your smartphone, an API is called behind the scenes.

Because APIs allow companies to open the data and functionality of their applications to external third-party developers, this eventually generates business partnerships, increasing revenue.

### **Types of API**

API Web

1. Web-based architecture, basically following the REST style.
2. Communication through the HTTP protocol.
3. It can be used to provide resources and services over the internet.

API Local

1. Specific architecture for accessing local services or system resources.
2. It can provide access to operating system or middleware resources.
3. It may be available on different operating systems and is not limited to a specific platform.

Program API

1. Based on the Remote Procedure Calling (RPC) technology, and includes variations such as gRPC, JSON-RPC, among others.
2. It allows access to remote software components as if they were local, facilitating communication between distributed systems.
3. There are various technologies and standards for implementing program APIs.

RESTful API

4. RESTful architecture based on REST principles, including resources, HTTP methods, and data presentation.
5. It uses standard HTTP methods (GET, POST, PUT, DELETE) to perform operations on resources.
6. Usually returns data in JSON or XML format.

API SOAP

4. Based on XML and follows the Simple Object Access Protocol (SOAP).
5. Provides highly compliant services with advanced features such as security and transactions.
6. More complex than RESTful APIs.

API GraphQL

1. Uses a flexible query language that allows clients to only request data they need.
2. It allows clients to specify the structure of the desired responses, making it popular in modern applications.
3. It does not necessarily follow a RESTful or SOAP architecture and is more flexible in terms of queries.

### **Protocols and API architecture**

1. XML-RPC: This protocol was created to exchange information between two or more networks. The client performs RPC using XML to encode their calls and HTTP requests for data transfer.


1. JSON-RPC: A lightweight JSON-encoded RPC, similar to XML-RPC, that allows multiple notifications and server calls that can be responded to asynchronously.

2. SOAP: A Web API protocol designed to exchange structured information. It uses XML to authenticate, authorize, and communicate between processes running on operating systems. Because web protocols such as HTTP run on most operating systems, SOAP allows clients to call web services and receive responses regardless of the language and platform used.

3. REST: An architectural style for providing standards between systems on the internet. Because it is not a protocol, library or tool, it makec communications between systems easy. The REST architecture makes the Client and Server implementations independent, without affecting each other's functioning.


### **Understanding the need for an API**

**![](https://c.mql5.com/2/59/Difference-Between-API-and-REST-API__1.png)**

1. **Automation** accelerates API testing and thereby increases efficiency. APIs not only connect the digital world with its dynamism, but also enable companies to become more agile by automating workflows.

2. **Integration** of platforms and applications can be implemented via APIs, allowing you to take advantage of continuous communication. Without APIs, companies lacked connectivity, which proportionately reduced productivity and performance. System integration allows you to move data, making it easier to automate your workflow and improve collaboration at work.

3. **Efficiency** increases as human intervention decreases. Providing access to APIs avoids duplication of content, giving companies more flexibility so they can spend time on quality innovation.

4. **Security** is an added benefit as the API adds an extra layer of security between your data and the server. Developers can further enhance security using tokens, signatures, and Transport Layer Security (TLS) encryption.


**Introduction to REST API:**

![](https://c.mql5.com/2/59/rest-api-model__1.png)

REST API is a software architecture model that establishes the rules and standards for interaction between applications and services on the internet. When an API strictly follows the principles of REST architecture, it is often referred to as the RESTful API.

One of the main advantages of the RESTful API is its flexible approach to accessing web services. This is because they use standard HTTP protocols and methods (such as GET, POST, PUT, and DELETE) to perform operations on the resources represented by URLs. This simplicity of design makes interacting with web services more accessible, even on devices or environments with limited computing resources.

**REST architecture**

1. **Nonstationarity**: systems operating in the REST paradigm must be non-stationary. When communicating between a client and a server, the stateless restriction makes the servers immune to the client's state and vice versa. Constraints are applied using resources instead of commands. They are web nouns that describe any object, document, or thing that can be stored/sent to other resources.

2. **Cache compatibility**. Caching helps servers mitigate some of the limitations associated with not being static. This is a critical factor that improves the performance of modern web applications. Caching not only improves performance on the client side, but also produces significant benefits on the server side. A well-functioning caching mechanism will significantly reduce the average response time of our server.

3. **Distributed nature**. REST uses a distributed approach in which client and server applications are separated from each other. Regardless of where requests are launched, the only information known to the client application is the URI (Uniform Resource Identifier) of the requested resource. The server application should transmit the requested data over HTTP but should not attempt to modify the client application.

4. **Multi-layering**. Multi-layer system makes REST architecture scalable. In a RESTful architecture, client and server applications are separated, so REST API calls and responses pass through different layers. Since the REST API is multi-layered, it must be designed in such a way that neither the client nor the server identifies their interaction with the end or intermediate applications.


### **Key benefits of REST API**

1. The REST API responds flexibly to different types of calls, for example, returning different data formats and changes structurally with the correct implementation of hypermedia. This allows users to interact bidirectionally with clients and servers, even if they are on different servers.

2. REST API adapts to any changes made to the data stored in the database, even if they are located on different internal and external servers. Since it relies to some extent on code, it helps to synchronize data across different sites.

3. Data Flow uses HTTP (GET, POST, PUT or DELETE) methods for communication, these methods are self-explanatory. Additionally, REST architecture helps improve developer productivity by allowing them to display information on the client side and store or manipulate data on the server side.


### **API vs REST API**

1. **Structure**. While most APIs structurally follow an application-to-application format, REST APIs work strictly on a web-based client and server concept. The client and server are separated from each other, which provides greater flexibility.

2. **Design**. APIs are usually lightweight architectures designed for limited devices. In contrast, REST APIs interact at the level of individual systems, which makes their architecture more complex.

3. **Protocol**. The main purpose of the API is to standardize data exchange between web services. Depending on the type of API, the choice of protocol varies. On the other hand, REST API is an architectural style for creating web services that communicate over HTTP. Although the REST API was formulated back in 2000 by computer scientist Roy Fielding, it remains the gold standard for public APIs.

4. **Support**. Most APIs are easy to implement because they do not face statelessness. In turn, the REST API executes even if the user does not know the names of the functions and parameters in a particular order.

5. **Unified Interface**. Not many APIs separate client and server or one application from another. The REST API adheres to the principles of a single interface and prohibits the use of multiple or separate interfaces within an API. Hypermedia connections should be used ideally to distribute a single interface. This should also ensure that a similar piece of data (such as username or email) belongs to only one URI. As a result, regardless of the initial request, all API requests for the same resources must appear the same

6. **Scalability**. While scalability is an issue for generic APIs, REST API has a layered structure, making it modular and more flexible to achieve scalability.


**Examples: Using WebRequest function in MQL5**

Now that we have entered the world of APIs and REST APIs, you are probably curious: "How does MQL5 fit into this story?" So, it's time for us to roll up our sleeves and start working with practical examples in MQL5. After all, there's nothing better than a good practical example.

To make the examples more complete and interesting, we will work with jsonplaceholder.typicode.com. We will also add some endpoints from Coinbase.

The idea is simple: let's start with basic CRUD operations using jsonplaceholder because it's great for understanding the basics and doing testing. But then we'll take it to a higher level and apply this knowledge to a more complex and challenging scenario of the Coinbase API. It has everything: quotes, historical data and even the ability to enter the world of trading. It all depends on which endpoints we access.

Let's start with CRUD operations.

New Post - POST

```
int CreateNewPost(string title, string body, int userId)
  {
   uchar result[];
   string result_headers;
   string url = "https://jsonplaceholder.typicode.com/posts";

   char post_data[];
   StringToCharArray(StringFormat("{\"title\": \"%s\", \"body\": \"%s\", \"userId\": %d}", title, body, userId), post_data);

   string headers = "Content-Type: application/json\r\n";
   int timeout = 5000;

   int res = WebRequest("POST", url, headers, timeout, post_data, result, result_headers);

   if(res > 0)
     {
      Print("Post created successfully.");
     }
   else
     {
      Print("Error: Failed to create post.");
     }

   return -1;
  }
```

Update Post - PUT

```
bool UpdatePost(int postId, string newTitle, string newBody)
  {
   uchar result[];
   string result_headers;
   string url = StringFormat("https://jsonplaceholder.typicode.com/posts/%d", postId);

   char put_data[]; // Declare post_data as char[]
   StringToCharArray(StringFormat("{\"title\": \"%s\", \"body\": \"%s\"}", newTitle, newBody), put_data);

   string headers = "Content-Type: application/json\r\n"; // Declare headers as char[]
   int timeout = 5000;

   int res = WebRequest("PUT", url, headers, timeout, put_data, result, result_headers);

   if(res > 0)
     {
      Print("Post updated successfully.");
      return true;
     }
   else
     {
      Print("Error: Failed to update post.");
      return false;
     }
  }
```

Delete Post - DELETE

```
bool DeletePost(int postId)
  {
   char data[];
   uchar result[];
   string result_headers;
   string url = StringFormat("https://jsonplaceholder.typicode.com/posts/%d", postId);
   int timeout = 5000;

   int res = WebRequest("DELETE", url, NULL, timeout, data, result, result_headers);

   if(res > 0)
     {
      Print("Post deleted successfully.");
      return true;
     }
   else
     {
      Print("Error: Failed to delete post.");
      return false;
     }
  }
```

Get Post - GET

```
string GetPostById(int postId)
  {
   char data[];
   uchar result[];
   string result_headers;
   string url = StringFormat("https://jsonplaceholder.typicode.com/posts/%d", postId);
   int timeout = 5000;

   int res = WebRequest("GET", url, NULL, timeout, data, result, result_headers);

   if(res > 0)
     {
      CJAVal jv;
      if(jv.Deserialize(result))
        {
         string postTitle = jv["title"].ToStr();
         Print("Post title: ", postTitle);
         return postTitle;
        }
      else
        {
         Print("Error: Unable to parse the response.");
        }
     }
   else
     {
      Print("Error: Failed to fetch post.");
     }

   return "";
  }
```

Now, imagine that you want to check the current price of Bitcoin (who wouldn't?). In addition to the general price, you may be interested in bid, ask, and spot price of BTC. To get this information via the API, we will use the WebRequest function in MQL5:

```
string GetBitcoinPrice(string priceType)
  {
   char data[];
   uchar result[];
   string result_headers;

   string baseURL = "https://api.coinbase.com/v2/prices/";
   if(priceType == "buy")
      baseURL += "buy";
   else
      if(priceType == "sell")
         baseURL += "sell";
      else
         baseURL += "spot";

   string url = baseURL + "?currency=USD";
   char headers[];
   int timeout = 5000;
   int res;

   Print("Fetching Bitcoin price for type: ", priceType);
   res = WebRequest("GET", url, NULL, NULL, timeout, data, 0, result, result_headers);

   string price = "";
   if(res > 0)
     {
      Print("Response received from Coinbase API");

      CJAVal jv;
      if(jv.Deserialize(result))
        {
         price = jv["data"]["amount"].ToStr();
         Print("Price fetched: ", price);
        }
      else
        {
         Print("Error: Unable to parse the response.");
        }
     }
   else
     {
      Print("Error: No response from Coinbase API or an error occurred.");
     }

   return price;
  }
```

We can also list all currencies.

```
string GetAvailableCurrencies()
  {
   char data[];
   uchar result[];
   string result_headers;
   string url = "https://api.coinbase.com/v2/currencies";
   char headers[];
   int timeout = 5000;
   int res;

   Print("Fetching list of available currencies from Coinbase API");
   res = WebRequest("GET", url, NULL, NULL, timeout, data, 0, result, result_headers);

   string currencies = "";
   if(res > 0)
     {
      Print("Response received from Coinbase API");

      CJAVal jv;
      if(jv.Deserialize(result))
        {
         // Considerando que a resposta é uma lista de moedas
         for(int i = 0; i < jv["data"].Size(); i++)
           {
            currencies += "Currency: " + jv["data"][i]["id"].ToStr();
            currencies += ", Name: " + jv["data"][i]["name"].ToStr();
            currencies += ", Min Size: " + jv["data"][i]["min_size"].ToStr();
            currencies += "\n";
           }
         Print(currencies);
        }
      else
        {
         Print("Error: Unable to parse the response.");
        }
     }
   else
     {
      Print("Error: No response from Coinbase API or an error occurred.");
     }

   return currencies;
  }
```

Now that we've seen all the functions created, let's run them and see the result.

```
#include "libraries/RESTFunctions.mqh"

int OnStart()
  {
//--- CRUD Operations on Posts ---//

// 1. Create a new post
   int userId = 1; // exemplo de userID, você pode ajustar conforme necessário
   string title = "Exemplo de Título";
   string body = "Este é o conteúdo do post para demonstração.";
   int newPostId = CreateNewPost(title, body, userId);
   if(newPostId != -1)
      Print("New Post ID: ", newPostId);

// 2. Update the created post
   string updatedTitle = "Título Atualizado";
   string updatedBody = "Conteúdo do post atualizado.";
   if(UpdatePost(newPostId, updatedTitle, updatedBody))
      Print("Post atualizado com sucesso.");

// 3. Get the updated post
   string fetchedTitle = GetPostById(newPostId);
   if(StringLen(fetchedTitle) > 0)
      Print("Título do Post Obtido: ", fetchedTitle);

// 4. Delete the post
   if(DeletePost(newPostId))
      Print("Post excluído com sucesso.");

//--- Coinbase Operations ---//

   const string buyPrice = GetBitcoinPrice("buy");
   const string sellPrice = GetBitcoinPrice("sell");
   const string spotPrice = GetBitcoinPrice("spot");

   Print("Buy Price: ", buyPrice);
   Print("Sell Price: ", sellPrice);
   Print("Spot Price: ", spotPrice);

   const string currencies = GetAvailableCurrencies();
   Print("Available Currencies: ", currencies);

//---
   return(INIT_SUCCEEDED);
  }
```

As a result, we will see the following:

![](https://c.mql5.com/2/59/2544471261206.png)

### **Conclusion:**

In short, APIs play a fundamental role in the world of programming and systems development because they allow different applications and systems to communicate with each other efficiently and flexibly. RestAPIs, based on the REST architecture, have become a popular choice as they offer greater simplicity and versatility compared to older protocols such as SOAP. They are widely used in various fields, including finance, as they enable the creation of scalable, integrated, and easy to maintain systems.

APIs are needed to facilitate communication between different applications, abstract away underlying complexity, and allow developers to leverage third-party resources. There are several types such as Web API, Local, Programmatic, RESTful, SOAP, and GraphQL, each with its own characteristics and uses.

The REST API, in turn, follows a set of architectural principles that make it flexible, adaptable, and easy to understand. The REST API is particularly suitable for web communications, uses the HTTP protocol to perform operations on resources, and often returns data in JSON or XML format.

If we compare APIs with REST APIs, we can note that REST APIs have a stricter structure, separate the client from the server, and operate primarily in a web context. Both adhere to a single interface and promote scalability thanks to a multi-level structure.

Finally, by applying these concepts to practical examples using MQL5, we can understand how APIs are used in the real world, from basic CRUD operations to retrieving financial data using the Coinbase API.

Well, guys, we have come to the end of this conversation about APIs, REST, MQL5 and everything else. Please note that everything that has been said here is the result of my professional vision: always with attention to detail - because you never know when a bug may appear out of nowhere!

It's important to remember that in the world of technology, things can change quickly, so it's always a good idea to stay on top of the latest trends and updates. And, of course, it never hurts to check if everything works as it should.

I hope this conversation was useful and instructive for you. Don't forget that even with all the technology, a good dose of curiosity always goes a long way in the world of programming.

Now I hope you'll explore the power of APIs, create your own integrations, and develop amazing systems. Remember, if things don't go according to plan, don't worry - that's how we gain experience.

Feel free to contact me if you have any further questions or need assistance. I am ready to help you in any way I can.

See you later, guys!

Translated from Portuguese by MetaQuotes Ltd.

Original article: [https://www.mql5.com/pt/articles/13661](https://www.mql5.com/pt/articles/13661)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/13661.zip "Download all attachments in the single ZIP archive")

[Parte\_01.zip](https://www.mql5.com/en/articles/download/13661/parte_01.zip "Download Parte_01.zip")(431.78 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Developing an MQL5 RL agent with RestAPI integration (Part 4): Organizing functions in classes in MQL5](https://www.mql5.com/en/articles/13863)
- [Developing an MQL5 RL agent with RestAPI integration (Part 3): Creating automatic moves and test scripts in MQL5](https://www.mql5.com/en/articles/13813)
- [Developing an MQL5 RL agent with RestAPI integration (Part 2): MQL5 functions for HTTP interaction with the tic-tac-toe game REST API](https://www.mql5.com/en/articles/13714)
- [Integrating ML models with the Strategy Tester (Conclusion): Implementing a regression model for price prediction](https://www.mql5.com/en/articles/12471)
- [Integrating ML models with the Strategy Tester (Part 3): Managing CSV files (II)](https://www.mql5.com/en/articles/12069)
- [Multilayer perceptron and backpropagation algorithm (Part 3): Integration with the Strategy Tester - Overview (I).](https://www.mql5.com/en/articles/9875)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/466047)**
(3)


![Khunpol Chumpol](https://c.mql5.com/avatar/2024/5/663D00B7-05BA.jpg)

**[Khunpol Chumpol](https://www.mql5.com/en/users/khunpolchumpol)**
\|
3 Jun 2024 at 23:16

Thanks you Sir!, i will add your to be my friend in MQL5 :)


![Wen Feng Lin](https://c.mql5.com/avatar/avatar_na2.png)

**[Wen Feng Lin](https://www.mql5.com/en/users/ken138888)**
\|
18 Jun 2024 at 00:26

powerful


![Israel Pelumi Abioye](https://c.mql5.com/avatar/2023/11/6554a830-8858.png)

**[Israel Pelumi Abioye](https://www.mql5.com/en/users/13467913)**
\|
13 Jul 2024 at 22:42

**MetaQuotes:**

Check out the new article: [Developing an MQL5 Reinforcement Learning agent with RestAPI integration (Part 1): How to use RestAPIs in MQL5](https://www.mql5.com/en/articles/13661).

Author: [Jonathan Pereira](https://www.mql5.com/en/users/14134597 "14134597")

Great article.


![Developing a Replay System (Part 34): Order System (III)](https://c.mql5.com/2/59/sistema_de_Replay_bParte_34x_logo.png)[Developing a Replay System (Part 34): Order System (III)](https://www.mql5.com/en/articles/11484)

In this article, we will complete the first phase of construction. Although this part is fairly quick to complete, I will cover details that were not discussed previously. I will explain some points that many do not understand. Do you know why you have to press the Shift or Ctrl key?

![Population optimization algorithms: Evolution Strategies, (μ,λ)-ES and (μ+λ)-ES](https://c.mql5.com/2/63/midjourney_image_13923_53_472__2-logo.png)[Population optimization algorithms: Evolution Strategies, (μ,λ)-ES and (μ+λ)-ES](https://www.mql5.com/en/articles/13923)

The article considers a group of optimization algorithms known as Evolution Strategies (ES). They are among the very first population algorithms to use evolutionary principles for finding optimal solutions. We will implement changes to the conventional ES variants and revise the test function and test stand methodology for the algorithms.

![Population optimization algorithms: Bacterial Foraging Optimization - Genetic Algorithm (BFO-GA)](https://c.mql5.com/2/64/Bacterial_Foraging_Optimization_-_Genetic_Algorithmz_BFO-GA____LOGO.png)[Population optimization algorithms: Bacterial Foraging Optimization - Genetic Algorithm (BFO-GA)](https://www.mql5.com/en/articles/14011)

The article presents a new approach to solving optimization problems by combining ideas from bacterial foraging optimization (BFO) algorithms and techniques used in the genetic algorithm (GA) into a hybrid BFO-GA algorithm. It uses bacterial swarming to globally search for an optimal solution and genetic operators to refine local optima. Unlike the original BFO, bacteria can now mutate and inherit genes.

![Overcoming ONNX Integration Challenges](https://c.mql5.com/2/75/Overcoming_ONNX_Integration_Challenges____LOGO.png)[Overcoming ONNX Integration Challenges](https://www.mql5.com/en/articles/14703)

ONNX is a great tool for integrating complex AI code between different platforms, it is a great tool that comes with some challenges that one must address to get the most out of it, In this article we discuss the common issues you might face and how to mitigate them.

[What's wrong with regular VPS?Here are the 8 most common problems that algorithmic traders may encounterRead![](https://www.mql5.com/ff/sh/hzatb686qjqxwtr4z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/450486&a=drhremihlwuaqyvgpzfddbtmgciejpba&s=c37d25bcceb93ed153b814e6ba4d4839461a9b2d68dd82b95b142be06d310f3f&uid=&ref=https://www.mql5.com/en/articles/13661&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5048940661903827395)

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