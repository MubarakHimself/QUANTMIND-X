---
title: Developing an MQL5 RL agent with RestAPI integration (Part 2): MQL5 functions for HTTP interaction with the tic-tac-toe game REST API
url: https://www.mql5.com/en/articles/13714
categories: Integration
relevance_score: 12
scraped_at: 2026-01-22T17:17:11.690620
---

[![](https://www.mql5.com/ff/si/3fgkjn78mkxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=ocndbzpeklfncxysjbwfhhbalbrsdbtv&s=a4309643278437a00bdd33c5809fc6b4b4032749c00fccd07b3b84e7b8b45126&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=ndxsyjtcrkxnaotfjrwgknkmhxyjanwt&ssn=1769091429577768369&ssn_dr=0&ssn_sr=0&fv_date=1769091429&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F13714&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Developing%20an%20MQL5%20RL%20agent%20with%20RestAPI%20integration%20(Part%202)%3A%20MQL5%20functions%20for%20HTTP%20interaction%20with%20the%20tic-tac-toe%20game%20REST%20API%20-%20MQL5%20Articles&scr_res=1920x1080&ac=17690914299181111&fz_uniq=5049020324957234021&sv=2552)

MetaTrader 5 / Examples


### Introduction

In the previous article, we talked about APIs and RestAPIs, highlighting how these critical technologies facilitate communication and data exchange between different systems. We analyzed the evolution of RestAPIs from the perspective of Roy Fielding's principles and how they have replaced older protocols such as SOAP with more efficient and flexible alternatives. We also emphasized the importance of RestAPI simplicity, scalability, and versatility, as well as their role in the development of advanced interconnected systems.

In this new article, we will expand on these concepts and apply them to a practical example. We will focus on developing a set of functions in MQL5 for working with HTTP calls and integrating with RestAPI capabilities for effective interaction with the external environment. For this, we'll consider the creation of the tic-tac-toe game in Python as a practical example.

We will start with developing functions in MQL5. These functions are necessary to establish effective communication with the external environment, in this case, with the tic-tac-toe game developed in Python. They enable the sending of HTTP requests and receiving of responses, acting as communication bridges between MQL5 and the game API.

In parallel, we will look at API development using FastAPI, which has been chosen for its characteristics such as high performance, ease of development and strong support for asynchronous APIs. FastAPI also integrates well with modern API development tools, making it easy to create efficient and scalable endpoints that our MQL5 feature set can use to interact with tic-tac-toe games.

After learning the basic MQL5 functions, we will create a test script designed to interact with the Python API. This script will be the most important element in demonstrating the practical applicability of our MQL5 functions in the context of the tic-tac-toe game.

This example will not only illustrate the practical application of the concepts discussed above but will also provide valuable insight into how these technologies can be used together to create innovative solutions.

In this article, we will look at both the technical development of functions in MQL5 and the API in FastAPI, and their role in building interconnected systems.

This article is divided into four main parts:

1. **Introduction and contextualization**. We will briefly review the API and RestAPI concepts discussed in the previous article and highlight the evolution and importance of these technologies for the interconnection of systems.

2. **Development of functions in MQL5 for HTTP calls**. Here we will focus on developing specific functions in MQL5. These functions will be designed to establish and manage HTTP communication, which is a fundamental step in integrating MQL5 with external environments.

3. **Creation and integration of API in FastAPI for tic-tac-toe**. In this section, we will look at developing a reliable API using FastAPI. The API will serve as the basis for our practical example - tic-tac-toe in Python - and will be integrated with functions developed in MQL5.

4. **Practical application and testing**. In the final part, we will implement a test script in MQL5 for interaction with the tic-tac-toe API. In this step, we will demonstrate the practical integration of the tools and concepts discussed, showing how the following function can be applied in a real context:


```
       def make_move(self, row, col):
           if self.board[row][col] == ' ':
               if self.player_turn:
                   self.board[row][col] = 'X'
               else:
                   self.board[row][col] = 'O'
               self.player_turn = not self.player_turn
           else:
               print("Invalid game. Try again.")
```


Each of these parts contributes to a thorough understanding of how RestAPI and MQL5 can be used together to develop interconnected solutions.

![](https://c.mql5.com/2/61/6337888553925.png)

The diagram above illustrates the sequence of interactions between the MQL5 script and tic-tac-toe, with the external API acting as an intermediary. We will start with a request to launch a game from an MQL5 script, which is processed by an external API. This in turn will call the tic-tac-toe implemented in Python. Moves are sent and received using HTTP requests, and an external API manages the input and output logic. Using FastAPI is key to the efficiency of this process, ensuring asynchronous and high-performance interactions. This flow provides a visual representation of the practicality and effectiveness of our approach to integration between different systems.

### **1\. Introduction and contextualization**

Continuing our series, today's article returns to the discussion of APIs and RestAPIs, focusing on their evolution and growing importance for the interconnection of digital systems. APIs, which have become fundamental in the era of digitalization, play a critical role in the integration of different programs, ensuring communication and harmonious functioning. With an architecture built around simplicity and scalability, the emergence of RestAPIs represents a significant leap forward, overcoming the limitations of older protocols such as SOAP and offering a more efficient option for interoperability between systems. The [first article](https://www.mql5.com/en/articles/13661) is devoted to deepening the understanding of these interfaces, describing their functionality, diversity and basic architecture, as well as considering examples of practical application. We set the stage for more complex discussions and more specific applications in future articles.

![](https://c.mql5.com/2/61/4090478434186.png)

RestAPIs, which emerged around the turn of the millennium, quickly gained popularity due to their simplified approach compared to protocols such as SOAP, which are based on XML and are known for being complex and requiring large amounts of data for basic operations. Emerging as a more flexible and versatile alternative, RestAPIs have established themselves in various sectors, including finance. Their ease of use and implementation in virtually all programming languages makes them particularly attractive for modern systems that require efficient communication.

The widespread acceptance of RestAPIs is largely due to their adaptability and support for various data formats (such as JSON and XML), which facilitates interoperability between different systems and platforms. Additionally, an important aspect of RestAPI is security: strong mechanisms such as token authentication and cryptography ensure that sensitive information is protected during data exchange. This level of security and flexibility is essential, especially in applications that handle critical information such as financial transactions and personal data.

### **2 - Developing functions in MQL5 for HTTP calls**

In this article, we will begin a detailed study of how to develop functions in MQL5 in order to make HTTP calls and manipulate JSON data. These functions play a vital role in the integration of MQL5 programs with the external environment, allowing you to effectively interact with web services and manipulate information in JSON format.

Additionally, when working with HTTP calls, it is very important to understand the meaning of HTTP status codes. These codes are returned by web servers to indicate the result of a request. For example, a status code "200 OK" indicates that the request was successful, while a status code "404 Not Found" means that the requested resource was not found on the server. Knowing these codes is essential for developing applications that interact with web services because they provide important information about the status of an HTTP request.

For complete and detailed information about HTTP status codes, you can refer to the complete documentation available at this [link](https://www.mql5.com/go?link=https://developer.mozilla.org/en-US/docs/Web/HTTP/Status "https://developer.mozilla.org/en-US/docs/Web/HTTP/Status"). Thus, you can understand these codes even better and confidently use them in your projects.

**SendGetRequest function - Performing GET requests**

We will begin with the SendGetRequest function, which is a central part of this process. This function enables the implementation of HTTP GET requests. It accepts a number of important parameters that provide a high degree of flexibility and control over the application.

```
int SendGetRequest(const string url, const string query_param, string &out, string headers = "", const int timeout = 5000)
```

- url: URL to which the GET request will be sent.
- query\_param: Optional query parameters that can be added to the URL.
- out: Output variable that will store the response to the request.
- headers: Optional HTTP headers that you can provide.
- timeout: Specifies the request timeout in milliseconds.

The SendGetRequest function is key in the GET request processing process, from request construction to HTTP error handling. It is important to note that although error handling is present in the function, in its current form it is relatively basic. I think it may be improved in the future.

After a successful response (response code between 200 and 204), the function decodes the data into UTF-8 and saves it in the "out" variable. This functionality is necessary to ensure that the collected data is processed correctly and used effectively in your projects.

**SendPostRequest function - Performing POST requests**

Now let's move on to the SendPostRequest function, which passes data to external web services, where the payload is often in JSON format. This function is used when creating or updating resources via the API.

```
int SendPostRequest(const string url, const string payload, string &out, string headers = "", const int timeout = 5000)
```

- url: URL to which the POST request will be sent.
- payload: Payload data that is sent along with the request, often in JSON format.
- out: Output variable that will store the response to the request.
- headers: Optional HTTP headers that you can provide.
- timeout: Specifies the request timeout in milliseconds.

The SendPostRequest function simplifies the process of creating a POST request, adding headers, and handling HTTP errors, although it is important to note that the current error handling may not be complete and may be improved in the future, just like with the SendGetRequest function. If the response is successful (response code 200 to 204), the data is decoded to UTF-8 and stored in the "out" variable.

**Request Function - Centralized Approach**

The Request function acts as a centralized entry point for GET and POST requests, providing a convenient and flexible approach.

```
int Request(string method,
            string &out,
            const string url,
            const string payload = "",
            const string query_param = "",
            string headers = "",
            const int timeout = 5000)
```

- method: Specifies the HTTP method to be used: "GET" or "POST".
- url: URL to which the request will be sent.
- payload: Data that is sent in the POST request (optional).
- query\_param: Optional query parameters for GET requests.
- out: Output variable that will store the response to the request.
- headers: Optional HTTP headers that you can provide.
- timeout: Specifies the request timeout in milliseconds.

The Request function acts as an abstraction layer, making it easy to choose between GET and POST without having to worry about implementation details. This not only simplifies the code, but also improves readability, making development for HTTP calls more efficient and accessible.

![](https://c.mql5.com/2/61/2034596175700.png)

The aim of this explanation is to assist the reader in learning how to effectively work with HTTP calls. With this knowledge, you will be able to integrate your own MQL5 projects with a wide range of web services, allowing you to send and receive data using the capabilities of HTTP calls and efficient work with data in JSON format. We will continue to explore these resources, guiding you step by step so that you can master your MQL5 programming skills and become an expert in interacting with web services over HTTP.

In addition to mastering HTTP calls in MQL5, it is important to understand the JSON (JavaScript Object Notation) format. JSON is an efficient and widespread way of structuring data, often used to exchange information between systems. Here are just a few reasons why it's important to learn how to work with JSON:

> #### 1\. Standard Web Format: JSON is the format of choice for many web APIs, making it fundamental to integrating external applications and services. By understanding JSON, you can easily send and receive data from web services in a structure that is widely used and understood.

> #### 2\. Ease of reading and writing: JSON is known for its readability. It uses simple, human-readable syntax to make it easier for developers to understand and debug data, which is especially important when working with API responses and debugging requests.

> #### 3\. Flexibility and nesting: JSON allows nesting of objects and arrays, which is ideal for representing complex and hierarchical data. This is very important when working with detailed and structured information.

> #### 4\. Compatible with different languages: JSON is compatible with a wide range of programming languages, making it a versatile choice. Regardless of the language used, understanding JSON makes working with data easier.

**Relevance of JSON in HTTP Calls**

Now that we understand the importance of JSON, we need to see how it integrates with HTTP calls. For both incoming requests (sending data) and outgoing responses (receiving data), JSON plays a fundamental role.

When sending data to a web service, you often need to format it as JSON and include it in the request. This allows the service to understand and process this data properly. On the other hand, when receiving responses from web services, the data is often returned in JSON format. Therefore, knowing how to parse and extract information from JSON becomes critical.

To further simplify working with JSON in MQL5, I decided to use the [**JAson**](https://www.mql5.com/en/code/13663) library, which is written in native MQL5. This eliminates the need to use external DLLs. This library is remarkably easy to ease and has proven its effectiveness. Now let's take a closer look at why the **JAson** library is an excellent choice for working with JSON in our MQL5 projects:

1. **Ease of use:** One of the main advantages of the **JAson** library is its ease of use. It was designed to simplify working with JSON, making reading, writing, and manipulating JSON objects accessible even to novice programmers. Intuitive syntax and thoughtful functionality will allow you to save time and effort when working with JSON.

2. **Convenient abstraction:** The **JAson** provides a convenient abstraction for JSON, which means we don't have to manually deal with the parsing and serialization details. Instead, we can use methods and functions provided by the library to perform common tasks such as accessing properties of JSON objects, adding elements to JSON arrays, and creating complex JSON structures.

3. **Improved code readability:** Working directly with JSON in our MQL5 code can be complex and result in confusing code. However, by using the JAson library, we get more readable and organized code. The operations associated with JSON are simplified, making our code more understandable and maintainable.

4. **Development efficiency:** Efficiency is a key point when developing a project. The **JAson** library can help speed up the development process by allowing us to focus on business logic rather than detailed JSON processing. This means we can deliver functionality faster and with fewer errors.


### **3\. Creation and integration of an API in FastAPI for the tic-tac-toe game**

Let's start our hands-on example by learning the tic-tac-toe code in Python. This is a fundamental step in the development of interaction between MQL5 and Python, since tic-tac-toe will serve as the environment with which MQL5 will interact.

**_TicTacToe_ class:**

```
import random

class TicTacToe:
```

Here we import the 'random' module and create a class called _TicTacToe_. A class is a software structure that organizes related functions and variables. In this case, the _TicTacToe_ class presents the tic-tac-toe game.

**Initialize the game board:**

```
    def __init__(self):
        self.board = [[' ' for _ in range(3)] for _ in range(3)]
        self.player_turn = True
```

In this part, we will define the _\_\_init\_\__ method which is a special method called when an instance of a class is created. It initializes the game state.

- _self.board_ is a two-dimensional list (3x3) representing the game board. Initially, all cells are empty (represented by empty cells).
- _self.player\_turn_ is a Boolean variable that keeps track of whose turn it is in the game. Initially, it is the first player's turn (True).

**Method _print\_board:_**

```
    def print_board(self):
        for row in self.board:
            print("|".join(row))
            print("-" * 5)
```

The _print\_board_ method is responsible for displaying the current state of the game board. It runs through the lines of the game board and prints elements separated by vertical bars ('\|') and horizontal lines ('-----') to create a visual representation of the game board.

**Method _check\_winner:_**

```
    def check_winner(self):
        for i in range(3):
            if self.board[i][0] == self.board[i][1] == self.board[i][2] != ' ':
                return self.board[i][0]
            if self.board[0][i] == self.board[1][i] == self.board[2][i] != ' ':
                return self.board[0][i]
        if self.board[0][0] == self.board[1][1] == self.board[2][2] != ' ':
            return self.board[0][0]
        if self.board[0][2] == self.board[1][1] == self.board[2][0] != ' ':
            return self.board[0][2]
        return None
```

The _check\_winner_ method checks if there is a winner in the game. It checks all rows, columns and diagonals for three consecutive single player symbols ("X" or "O"). If a player wins, that player's symbol is returned (for example, "X" if player "X" wins). If no winner is found, the method will return None.

**Method _make\_move:_**

```
    def make_move(self, row, col):
        if self.board[row][col] == ' ':
            if self.player_turn:
                self.board[row][col] = 'X'
            else:
                self.board[row][col] = 'O'
            self.player_turn = not self.player_turn
        else:
            print("Invalid game. Try again.")
```

The make\_move method is responsible for making a move in the game. It receives the row and column coordinates where the player intends to make a move. Before making a move, it checks whether the cell on the game board (' ') is empty. If it is empty, a move is made, and the player symbol ("X" or "O") is placed in the cell. The player's turn then changes. If the cell is not empty, the "Invalid move" message is displayed.

The above code represents the process of integration between MQL5 and Python. Developing tic-tac-toe in Python is an important step, as it will serve as the environment in which our interactions will take place.

By examining every aspect of this code, you can see the ease and efficiency with which Python allows us to create a functional game. However, the true potential of this code will be revealed when we connect it to MQL5, which will allow these two different technologies to work together synergistically.

Now that we've laid a solid foundation with tic-tac-toe, we're ready to take the next step: develop a Python API using FastAPI. This API will be the link between MQL5 and the tic-tac-toe game, allowing MQL5 to execute queries and receive responses to play the game. Along with the development process, we will see the aspects of this integration.

Now that you are familiar with the _TicTacToe_ class and with the way it works, let's take a closer look at the FastAPI API code above. This API will be used in the interaction between MQL5 and the tic-tac-toe game we previously developed in Python. Let's understand each part of this code:

```
# Import of Libraries
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tic_tac_toe import TicTacToe

# Instantiate FastAPI
app = FastAPI()

# Running game storage
games = {}
```

Let's start by importing the two required libraries: FastAPI and Pydantic. FastAPI is a framework for rapidly developing web interfaces in Python. Pydantic is used to define data models that describe the structure of the data expected from the API.

```
# Defining data models
class GameBoard(BaseModel):
    board: list
    player_turn: str

class PlayerMove(BaseModel):
    row: int
    col: int
```

Next, we will define two data models using Pydantic. GameBoard represents the current state of the game board and contains a 'board' list to store the board cells and a string player\_turn to indicate the progress of the game. PlayerMove represents the move the player wants to perform and includes the row and column coordinates of the move.

```
# Definition of API Endpoints

# Endpoint to start a new game
@app.get("/start-game/")
def start_game():
    game_id = len(games) + 1
    game = TicTacToe()
    games[game_id] = game
    return {"game_id": game_id}

# Endpoint to make a play in the game
@app.post("/play/{game_id}/")
def play(game_id: int, move: PlayerMove):
    game = games.get(game_id)
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")

    try:
        game.make_move(move.row, move.col)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Use the print_board method to print the current game do
    game.print_board()

    return {
        "board": game.board,
        "player_turn": game.player_turn,
    }
```

The API has two main endpoints. First, _/start-game/_, allows you to start a new game of tic-tac-toe. It creates an instance of the _TicTacToe_ class to represent a game and associates it with a unique ID in the games dictionary. The game ID is returned as the response.

The second endpoint _/play/{game\_id}/_ allows players to make moves in an existing game. As a parameter, it receives the game ID and game data of the player. The API checks if a game with the specified ID exists in the _games_ dictionary. If it does not exist, error 404 is returned. Then it tries to make a move in the game and update the state of the game board. The current state of the game board is returned as a response.

```
# Execução da API
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

We complete the code by executing the API using Uvicorn. This allows other applications to access the API.

![](https://c.mql5.com/2/61/6219770609749.png)

Automatically generated API documentation

[FastAPI](https://www.mql5.com/go?link=https://fastapi.tiangolo.com/ "https://fastapi.tiangolo.com/") is a modern and fast framework for creating APIs in Python 3.6+, based on the standards from Python type hints. Its simplicity lies in its ability to provide fast, efficient, and reliable coding, focusing on the following capabilities:

1. **Speed and performance**: FastAPI is one of the fastest frameworks for Python, using Starlette for routing and Uvicorn for ASGI server execution. This makes it ideal for applications requiring high performance.

2. **Automatic documentation generation**: With FastAPI, API documentation is generated automatically and is also updated as the code changes. This is made possible by using type hints from Python and integrated documentation systems such as Swagger UI and ReDoc.

3. **Data validation and serialization**: Using Pydantic, FastAPI offers a robust data validation system. Requests and responses are automatically validated and serialized, reducing the amount of code required and avoiding many common pitfalls.

4. **Ease of use**: FastAPI is designed to be easy to use and intuitive to learn. This makes it easier for developers to create and maintain efficient APIs, with a less steep learning curve compared to other frameworks.

5. **Support for asynchronous operations**: With built-in support for asynchronous operations, FastAPI is suitable for applications that need to handle concurrent requests, such as real-time services.

6. **Security and authentication**: The framework also provides tools for implementing API security, including support for OAuth2 with JWT tokens and other authentication methods.


Using FastAPI to create an API clearly demonstrates its efficiency and simplicity. This frame provides seamless integration between various technologies, such as MQL5 and Python, and allows you to quickly develop high-performance web applications. Using FastAPI, developers can focus more on business logic and less on the technical details of API implementation, speeding up the development process and ensuring the quality of the final product.

### **4\. Practical implementation and testing**

#### The MQL5 test script under analysis is a practical example of integration of this programming language with an external API built in FastAPI, applied in the tic-tac-toe game context. The process is divided into several phases, each of which is responsible for certain interaction with the API:

The script starts by establishing the FastAPI API address:

```
input string apiUrl = "http://localhost:8000";
```

This line defines the base API URL that we will use to send requests to the FastAPI server. This setting is necessary to ensure that the script can interact correctly with the API.

#### Function to print API responses

```
void PrintResponse(string response) {
    Print("Resposta da API: ", response);
}
```

This auxiliary function is designed to output responses received from the API to the MQL5 log. It provides useful information about the outcome of each request submitted.

#### Initialization and first request

```
void OnStart() {
    string response;
    int startGameRes = Request("GET", response, apiUrl + "/start-game/");
    // ...
}
```

_OnStart_ is the main function of the script. By sending a GET request to the API, we will start a new game. The startGameRes variable stores the result of this request, which indicates whether it was successful.

#### Processing the response and starting the game

```
string gameIdStr = StringSubstr(response, StringFind(response, ":") + 1);
uint gameId = StringToInteger(gameIdStr);
```

Having received a response from the API, the script extracts the game ID from it. This ID is very important for subsequent games as it identifies the current gaming session.

#### Making moves

```
string jsonMove1 = "{\"row\": 0, \"col\": 0}";
// ...
int playRes1 = Request("POST", response, apiUrl + "/play/" + IntegerToString(gameId) + "/", jsonMove1);
```

The script then performs a series of moves, each of which is represented by a JSON object that contains the coordinates of the move on the game board. These moves are sent to the API using POST requests. The process is repeated for each subsequent move.

![](https://c.mql5.com/2/61/601178842900__2.png)

#### Gameplay analysis

- **Start**: The game starts, we get a game ID.
- **Moves**: Three moves are made, alternating between players.
- **Feedback**: After each game, the script processes the API response to check for updated game state.

#### Perceptions:

- **Effective integration**: The script demonstrates the effective integration between MQL5 and FastAPI API, showing how MQL5 can be used to interact with an external application.
- **Flexible model**: This example serves as a flexible model for other types of applications, demonstrating MQL5's ability to integrate with external APIs.
- **Practical applicability**: This script is a practical example of combining MQL5 programming and the flexibility of the Python API using FastAPI, opening the way to new innovations in interaction between different systems.

This MQL5 test script not only illustrates theoretical concepts of system integration, but also provides practical application, highlighting the interconnectedness and adaptability in programming modern systems.

### Conclusion

This article shows that a journey through the world of programming is always full of discoveries. At this stage, we consider the development of MQL5 functions for processing HTTP calls, which, at first glance, may seem a little complex from a technical point of view, but in fact opens up a whole universe of possibilities.

When thinking about MQL5, we usually associate it directly with trading, but who would have thought that it could be used for something completely different, for example, to run a tic-tac-toe game made in Python? This shows that a programming language can find applications far beyond our usual understanding.

Speaking of Python, creating a tic-tac-toe game was a fun way to see programming in action. This is a practical example that helps to better understand how different languages and technologies can be related.

The choice of FastAPI to create the API was strategic. Fast, efficient and easy to use, FastAPI has proven to be an excellent tool for building a bridge between our game and MQL5. It's interesting to see how an API, which may seem like just a technical intermediary, actually plays a key role in connecting the different worlds of programming.

Finally, we created a test script in MQL5. We tested everything in practice, to see how theory could become reality. This is where we became convinced of the potential of integrating these technologies. The script shows that with a little creativity and technical knowledge, we can create amazing things.

So, what have we learned from all this? Programming is a vast field with many surprises and opportunities. MQL5 and Python are only some of the tools at our disposal that, when used together, can create unexpected and innovative solutions. And perhaps most importantly, in the world of technology there is always something new to learn and explore.

Translated from Portuguese by MetaQuotes Ltd.

Original article: [https://www.mql5.com/pt/articles/13714](https://www.mql5.com/pt/articles/13714)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/13714.zip "Download all attachments in the single ZIP archive")

[Parte\_02.zip](https://www.mql5.com/en/articles/download/13714/parte_02.zip "Download Parte_02.zip")(35.95 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Developing an MQL5 RL agent with RestAPI integration (Part 4): Organizing functions in classes in MQL5](https://www.mql5.com/en/articles/13863)
- [Developing an MQL5 RL agent with RestAPI integration (Part 3): Creating automatic moves and test scripts in MQL5](https://www.mql5.com/en/articles/13813)
- [Developing an MQL5 Reinforcement Learning agent with RestAPI integration (Part 1): How to use RestAPIs in MQL5](https://www.mql5.com/en/articles/13661)
- [Integrating ML models with the Strategy Tester (Conclusion): Implementing a regression model for price prediction](https://www.mql5.com/en/articles/12471)
- [Integrating ML models with the Strategy Tester (Part 3): Managing CSV files (II)](https://www.mql5.com/en/articles/12069)
- [Multilayer perceptron and backpropagation algorithm (Part 3): Integration with the Strategy Tester - Overview (I).](https://www.mql5.com/en/articles/9875)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/466426)**
(4)


![Stanislav Korotky](https://c.mql5.com/avatar/2010/10/4CA7CFA0-1F0C.jpg)

**[Stanislav Korotky](https://www.mql5.com/en/users/marketeer)**
\|
27 Apr 2024 at 12:56

Between part 1 and part 2 you obviously missed another whole article linking RestAPI with python and FastAPI. This is a site for traders after all.


![Jonathan Pereira](https://c.mql5.com/avatar/2020/3/5E5F1716-E757.jpg)

**[Jonathan Pereira](https://www.mql5.com/en/users/14134597)**
\|
27 Apr 2024 at 14:44

**Stanislav Korotky [#](https://www.mql5.com/pt/forum/458157#comment_53194834):**

Between part 1 and part 2, you obviously missed another complete article linking RestAPI with python and FastAPI. After all, this is a site for traders.

If you had a modicum of ability, you'd realise that between part 1 and part 2, you already have enough information to adapt to your trading environment. Use your head and take advantage of sites like [https://brapi.dev/docs](https://www.mql5.com/go?link=https://brapi.dev/docs "https://brapi.dev/docs") to enrich your analysis and implementation in MetaTrader. Stop expecting everything to be chewed over for you. After all, this is a site for traders, not amateurs.

![Stanislav Korotky](https://c.mql5.com/avatar/2010/10/4CA7CFA0-1F0C.jpg)

**[Stanislav Korotky](https://www.mql5.com/en/users/marketeer)**
\|
28 Apr 2024 at 16:06

**Jonathan Pereira [#](https://www.mql5.com/ru/forum/466244#comment_53195447):**

If you had any ability at all, you would have realised that between the first and second part you already have enough information to adapt to your trading environment. Use your head and use sites like [https://brapi.dev/docs](https://www.mql5.com/go?link=https://brapi.dev/docs "https://brapi.dev/docs") to enrich your analysis and implementation in MetaTrader. Stop expecting everything to be chewed up for you. After all, this is a site for traders, not amateurs.

If you would moderate your arrogance in favour of a more methodical presentation of the material and would not send to the Internet for information that you missed in the articles (which, with this approach, you can not write at all, but send everyone to Google at once), it would be much easier for the readers. And insulting them does not honour you as an author or as a moderator.

![Denis Kirichenko](https://c.mql5.com/avatar/2019/5/5CEDB8D2-7CB7.jpg)

**[Denis Kirichenko](https://www.mql5.com/en/users/denkir)**
\|
28 Apr 2024 at 18:12

**Jonathan Pereira [#](https://www.mql5.com/ru/forum/466244#comment_53195447):**

If you had any ability at all, you would have realised that between the first and second part you already have enough information to adapt to your trading environment. Use your head and use sites like [https://brapi.dev/docs](https://www.mql5.com/go?link=https://brapi.dev/docs "https://brapi.dev/docs") to enrich your analysis and implementation in MetaTrader. Stop expecting everything to be chewed up for you. After all, this is a site for traders, not amateurs.

Imho, Señor Pereira is burning somehow unconstructively, in the style of "you are a fool yourself".... Apparently Senor Pereira has not seen Stanislav's textbook, so he is so categorical....


![Custom Indicators (Part 1): A Step-by-Step Introductory Guide to Developing Simple Custom Indicators in MQL5](https://c.mql5.com/2/76/Indicators_Article_Thumbnail_Artwork.png)[Custom Indicators (Part 1): A Step-by-Step Introductory Guide to Developing Simple Custom Indicators in MQL5](https://www.mql5.com/en/articles/14481)

Learn how to create custom indicators using MQL5. This introductory article will guide you through the fundamentals of building simple custom indicators and demonstrate a hands-on approach to coding different custom indicators for any MQL5 programmer new to this interesting topic.

![MQL5 Wizard Techniques you should know (Part 17): Multicurrency Trading](https://c.mql5.com/2/76/MQL5_Wizard_Techniques_you_should_know_wPart_17m_Multicurrency_Trading___LOGO.png)[MQL5 Wizard Techniques you should know (Part 17): Multicurrency Trading](https://www.mql5.com/en/articles/14806)

Trading across multiple currencies is not available by default when an expert advisor is assembled via the wizard. We examine 2 possible hacks traders can make when looking to test their ideas off more than one symbol at a time.

![Developing a Replay System (Part 38): Paving the Path (II)](https://c.mql5.com/2/61/Replay_Parte_38_Pavimentando_o_Terreno_LOGO.png)[Developing a Replay System (Part 38): Paving the Path (II)](https://www.mql5.com/en/articles/11591)

Many people who consider themselves MQL5 programmers do not have the basic knowledge that I will outline in this article. Many people consider MQL5 to be a limited tool, but the actual reason is that they do not have the required knowledge. So, if you don't know something, don't be ashamed of it. It's better to feel ashamed for not asking. Simply forcing MetaTrader 5 to disable indicator duplication in no way ensures two-way communication between the indicator and the Expert Advisor. We are still very far from this, but the fact that the indicator is not duplicated on the chart gives us some confidence.

![The Group Method of Data Handling: Implementing the Combinatorial Algorithm in MQL5](https://c.mql5.com/2/76/The_Group_Method_of_Data_Handling___LOGO.png)[The Group Method of Data Handling: Implementing the Combinatorial Algorithm in MQL5](https://www.mql5.com/en/articles/14804)

In this article we continue our exploration of the Group Method of Data Handling family of algorithms, with the implementation of the Combinatorial Algorithm along with its refined incarnation, the Combinatorial Selective Algorithm in MQL5.

[![](https://www.mql5.com/ff/si/3p2yc19r7qvs297n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F618%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dsignal.advantage%26utm_content%3Dsubscribe.signal%26utm_campaign%3D0622.MQL5.com.Internal&a=bewozmaxwejekdopjicjtsbzmjgfjyvt&s=e49ac7e84b713650e3af82ec3c6b4d02fdf06617c5821011b1e499af5edd01f4&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=anuqlputahddxaomfpmzognpqmtfmqvk&ssn=1769091429577768369&ssn_dr=0&ssn_sr=0&fv_date=1769091429&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F13714&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Developing%20an%20MQL5%20RL%20agent%20with%20RestAPI%20integration%20(Part%202)%3A%20MQL5%20functions%20for%20HTTP%20interaction%20with%20the%20tic-tac-toe%20game%20REST%20API%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176909142991888114&fz_uniq=5049020324957234021&sv=2552)

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