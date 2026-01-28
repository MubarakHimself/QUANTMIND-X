---
title: Integration of Broker APIs with Expert Advisors using MQL5 and Python
url: https://www.mql5.com/en/articles/16012
categories: Integration
relevance_score: 12
scraped_at: 2026-01-22T17:16:51.118424
---

[![](https://www.mql5.com/ff/sh/592yc11u3j4rs5z9z2/01.png)How AI helps create robots for MetaTrader 5Learn from our book "Neural Networks in Algo Trading with MQL5"Read](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/neurobook%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.neurobook%26utm_content=visit.page%26utm_campaign=neurobook.promo.04.2024&a=ghrobswocqgvhztzjldphupateyllpro&s=9929cb0b8629585b5a42fabc06c525e41f6c0ebdf3045d044a5413b93ea88b47&uid=&ref=https://www.mql5.com/en/articles/16012&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5049016111594316624)

MetaTrader 5 / Examples


### Introduction

Today, we will explore how to establish a seamless connection between our MetaTrader 5 Expert Advisors and external brokers using API integration. The primary objective is to address the challenge of insufficient funds in trading accounts by enabling automated top-ups when balances fall below a set threshold.  This approach effectively tackles critical fund management issues, enhancing both efficiency and security in trading operations.

Typically, we follow a routine of logging into our broker account portal to perform various transactions and operations. While this is the traditional approach, there exists a powerful feature called an API (Application Programming Interface) that allows us to do much more and optimize our approach. For some of you, this may be a familiar term. However, for others, I will break it down into easy-to-understand sections to ensure everyone is on the same page:

1. What is an API?
2. Usage of APIs
3. Accessing APIs
4. API Documentation

Let’s explore these in detail:

1\. What is an API?

An API (Application Programming Interface) enables external interactions with a software system. In this context, it provides us with the ability to perform operations on our broker accounts directly on the server—without the need to log in manually.

Interestingly, [APIs](https://en.wikipedia.org/wiki/API "https://en.wikipedia.org/wiki/API") have been around since the 1940s but gained traction in the 1970s, achieving widespread adoption after the year 2000.

2\. Usage of APIs

APIs facilitate communication between software applications and systems. Here are some of the common application areas:

- Website servers
- Mobile applications

APIs allow seamless data access from various sources, such as:

- Social media platforms
- Cloud storage
- Payment gateways
- Weather stations
- Stocks and Financial Markets

![API](https://c.mql5.com/2/110/API.png)

API as a bridge between external computers and cloud servers.

3\. Accessing APIs

To use an API, you generally need to:

- Obtain an API key from the provider.
- Study the API documentation to integrate it correctly.
- Include the API key as a parameter in your requests for authentication and authorization.

The API key is crucial for identifying your application or user while ensuring secure access.

4\. API Documentation

Each API comes with its own user guide and specifications. These detail how to interact with the API effectively. For instance, the documentation for [Telegram's API](https://www.mql5.com/go?link=https://core.telegram.org/api "https://core.telegram.org/api") differs significantly from that of a broker’s API, such as [Deriv Broker's API](https://www.mql5.com/go?link=https://api.deriv.com/ "https://api.deriv.com/").

By understanding and utilizing APIs, we can streamline our operations, automate processes, and enhance efficiency in managing our broker accounts. Let’s dive deeper into each aspect and explore its practical implementation.

Here is a list of major sections in our discussion:

- [Overview of this discussion](https://www.mql5.com/en/articles/16012#para2)
- [Deriv API](https://www.mql5.com/en/articles/16012#para3)
- [Python Library for Web Socket API Communications](https://www.mql5.com/en/articles/16012#para4)
- [Python Script for Bridging the Communication Between MetaTrader 5 and Broker API](https://www.mql5.com/en/articles/16012#para5)
- [Funds Manager EA](https://www.mql5.com/en/articles/16012#para6)
- [Demo Testing and Results](https://www.mql5.com/en/articles/16012#para7)
- [Conclusion](https://www.mql5.com/en/articles/16012#para8)

### Overview of this discussion

In this article, we will explore the use of MQL5 with a broker's API to facilitate seamless operations in fund management. Our objective is to uncover the vast potential that MQL5 Expert Advisors (EAs) offer. Since MQL5 alone cannot directly interact with web servers, we will utilize external language libraries, such as Python or Node.js, depending on the broker’s API capabilities. For this discussion, we will focus on Deriv.com as our chosen broker.

Benefits of This Approach

- Automated Fund Transfers: The EA will automate the transfer of trading gains from the trading account to a secure account.
- Trading Account Top-Up: It will replenish the trading account balance if equity falls below a predefined threshold.
- Continuous Operations: By hosting the EA on a virtual server, the system can operate 24/7 with virtually unlimited fund management capabilities.

Challenges and Considerations

- Risk of Fund Depletion: Reserved funds may be exhausted if the EA continues to top up without generating consistent profits.
- Need for Robust Fund Management: A sophisticated fund management system must be implemented to halt operations if specific loss or equity thresholds are breached.

In the next sections, we will delve into the features of the Deriv broker API, develop Python code to enable seamless interaction with the API, and integrate this solution with an MQL5 EA designed for efficient and reliable fund management, all presented in straightforward, step-by-step instructions for easy comprehension and implementation.

![Basic Bi-directional Flow EA to Broker API](https://c.mql5.com/2/110/EA_Python_Broker.png)

Bi-Directional Flow Between MQL5 and Broker

### Deriv API

[Deriv.com](https://www.mql5.com/go?link=https://track.deriv.com/_r6xDODPy3Ly2vdm9PpHVCmNd7ZgqdRLk/1/ "https://track.deriv.com/_r6xDODPy3Ly2vdm9PpHVCmNd7ZgqdRLk/1/") is a popular brokerage that supports MetaTrader 5 and offers API access for automated operations. Detailed information can be found in the [Deriv API documentation](https://www.mql5.com/go?link=https://developers.deriv.com/docs/getting-started "https://developers.deriv.com/docs/getting-started"). To begin, follow the steps outlined on their website to sign up for API access.

For educational purposes, you’ll need to create an application in the [Deriv API dashboard.](https://www.mql5.com/go?link=https://api.deriv.com/dashboard/ "https://api.deriv.com/dashboard/") During this process, select a Demo account and generate an API token with the appropriate access level. For this presentation, I recommend choosing the Trade API access level, as it allows operations such as topping up your demo account funds. Avoid selecting the Admin Access level unless you’re certain you can manage it securely, as this level grants full control over your account. Always keep your API token private and secure to prevent unauthorized access.

Once the API token is created, the next steps involve setting up a Python WebSocket client and integrating it with MQL5 to develop a program capable of managing account operations.

Key Operations

The primary functions we aim to achieve with our program are:

- Depositing funds from your Deriv account into your Deriv MetaTrader 5 account.
- Interfacing with the Deriv API for seamless fund management.

The Deriv API documentation provides detailed instructions for performing these operations under the MT5 APIs section. As part of this process, you will obtain two critical values:

- App ID: Identifies your application on the Deriv platform.
- API Token: Grants access to perform the specified actions.

With these values, we can proceed to develop the Python WebSocket client and integrate it with MQL5 to implement our fund management system.

You and Your API Security

Security is a critical topic that cannot be overlooked. Just as you secure your phone from unauthorized access, it is equally essential to safeguard your API keys from others. An API token acts as a gateway to your account, and if it falls into the hands of cybercriminals, it could leave you vulnerable to malicious activities. They could exploit your account and carry out unauthorized actions without your knowledge.

In the context of Deriv, the platform provides API token access levels, allowing you to choose the permissions granted to the token. It is highly recommended to use the demo access features and avoid selecting Admin Access unless absolutely necessary. The Admin Access level grants full control over your account, posing a significant risk if it gets into the wrong hands.

This advice isn’t limited to Deriv; it applies to API tokens from any service. Always treat API keys like passwords—keep them private, store them securely, and never share them with anyone you don’t trust. By doing so, you minimize the risk of unauthorized access and ensure your account remains protected.

### Python Library for Web Socket API Communications

First and foremost, we need to have [Python](https://www.mql5.com/go?link=https://www.python.org/ "https://www.python.org/") installed on our computer to proceed with this project. Next, we need to install the Python WebSocket library to enable web communication with the broker's API. The [Deriv API](https://www.mql5.com/go?link=https://api.deriv.com/?link=https://track.deriv.com/_r6xDODPy3Ly2vdm9PpHVCmNd7ZgqdRLk/1/ "https://api.deriv.com/?link=https://track.deriv.com/_r6xDODPy3Ly2vdm9PpHVCmNd7ZgqdRLk/1/") uses [WebSocket](https://www.mql5.com/en/book/advanced/project/project_websocket_mql5) for high-speed communication, so it is essential to set up our computers accordingly. Below is the command to install the Python library on Windows using the Command Prompt.

```
pip install websocket-client
```

To install the library, open the _Command Prompt_ on Windows and enter the command above. Once the installation is complete, you'll be ready to start running and testing. However, before proceeding, we will take the next few steps to develop the Python script that will handle communication between MetaTrader 5 and the Broker API ( [Deriv's API](https://www.mql5.com/go?link=https://api.deriv.com/ "https://api.deriv.com/") as an example).

### Python Script for Bridging the Communication Between MetaTrader 5 and Broker API

I’ve outlined the steps for developing the Python script. Let’s dive into the construction of this program before we proceed with creating our Fund Manager EA. Open your code editor ( [Notepad++](https://www.mql5.com/go?link=https://notepad-plus-plus.org/downloads/ "https://notepad-plus-plus.org/downloads/") is preferred) and create a new file. Follow along as we develop the script, which we'll name _deriv\_api\_handler.py_. Make sure to note the location of your script, as it will be needed later in the EA code.

1\. Initial Setup and Configuration

Firstly, we want to establish the necessary imports, configure connection settings, and set up the file paths where communication with MQL5 will happen. We start by importing the essential libraries. We know that the _json_ library is crucial for parsing and generating JSON data, which is the format to be used for sending messages through _WebSocket_ and processing commands. The _websocket_ library is chosen to handle the _WebSocket_ connection to the Deriv API, and _time_ helps the program introduce pauses in the script when necessary. Lastly, _os_ allows it to interact with the file system, like checking for command files.

```
import json
import websocket
import time
import os

# Configuration
API_URL = "wss://ws.binaryws.com/websockets/v3?app_id= Your app ID"   #Replace with your App ID created in Deriv API dashboard
API_TOKEN = "Your API token"   #Replace with your actual token

# File paths (Replace YourComputerName with the actual name of your computer)
MQL5_TO_PYTHON = "C:/Users/YourComputerName/AppData/Roaming/MetaQuotes/Terminal/Common/Files/mql5_to_python.txt"
PYTHON_TO_MQL5 = "C:/Users/YourComputerName/AppData/Roaming/MetaQuotes/Terminal/Common/Files/python_to_mql5.txt"
```

- Next, we define our API URL and token, as these are the keys needed to authorize our _WebSocket_ connection with the Deriv server. This URL points to the _WebSocket_ endpoint, and the API token is what proves our identity to the server.

- Finally, we specify file paths for interacting with MQL5. These files will act as the bridge between our Python script and MQL5, allowing for the exchange of commands and responses. We should be careful to choose paths that we know are accessible to both systems (MQL5 and Python).

2\. WebSocket Connection

In this code section of our script we establish a secure connection to the Deriv WebSocket API and authorize the script to make requests.

```
def connect_to_deriv():
    """Connects to Deriv's WebSocket API."""
    try:
        ws.connect(API_URL)
        ws.send(json.dumps({"authorize": API_TOKEN}))
        response = json.loads(ws.recv())
        print(f"Authorization Response: {response}")
        if response.get("error"):
            print("Authorization failed:", response["error"]["message"])
            return False
        return True
    except Exception as e:
        print(f"Error during authorization: {e}")
        return False
```

- We define a function to manage the _WebSocket_ connection. First, we try to connect to the API using the URL we set up earlier. We initiate the connection with _ws.connect(API\_URL)_ and then send the authorization message containing our API token. This is necessary to authenticate our script with the Deriv server.

- Immediately, the script listens for a response from the server. The server will return a JSON object that confirms whether the connection was successful. If it contains an error, it knows the token was invalid or there was another issue. This error handling is essential to ensure the script fails gracefully.

- Our decision to use _try-except_ blocks ensures that we don’t crash the script if something goes wrong with the connection or message exchange. It’s a safety measure that gives the program flexibility to debug and handle issues without disrupting the whole process.

3\. Command Processing

At this stage, the program interprets the received commands and takes appropriate actions based on the command type. Once successfully connected and authenticated, the script will process commands from MQL5 sent in the form of a JSON string, so it will be ready to parse and handle that.

- In the _process\_command_ function, the script tries to parse the command into a Python dictionary first. This allows it to easily access different fields of the command (like the deposit amount). Then it checks for specific keys in the parsed JSON (like "mt5\_deposit"), which tells what kind of action MQL5 is asking to be performed.

- If the command requests a deposit (" _mt5\_deposit_"), a separate function _mt5\_deposit_ is called to handle it. This modular approach makes the script flexible and maintainable because we can easily add other command types in the future.

- If the command is unknown or formatted incorrectly, it returns an error message, which helps the MQL5 system understand what went wrong.

```
def process_command(command):
    """Processes a command from MQL5."""
    try:
        command_data = json.loads(command)  # Parse the JSON command
        if "mt5_deposit" in command_data:
            return mt5_deposit(
                command_data["amount"],
                command_data["from_binary"],
                command_data["to_mt5"]
            )
        else:
            return {"error": "Unknown command"}
    except json.JSONDecodeError:
        return {"error": "Invalid command format"}
    except Exception as e:
        return {"error": f"Unexpected error: {e}"}
```

4\. Deposit Operation

This section performs the deposit operation as requested by MQL5, sending the required data to the Deriv API. We created a dedicated function (mt5\_deposit) for performing the deposit. This keeps our code organized and isolates the logic for handling deposits, making it easier to maintain or extend.

- Inside this function, a _JSON_ message that tells Deriv to execute a deposit is crafted. This message includes all necessary details such as the amount, the source account, and the target MT5 account. Sending the right data in the correct format is crucial to ensure the transaction is processed successfully.

- After sending the deposit request, it waits for a response from the server. Once the response is received, it is parsed into JSON and returned. This lets the MQL5 system know if the operation succeeded or failed.

- Exceptions are handled in case something goes wrong with the communication or the deposit operation, ensuring that errors are captured and reported back to MQL5.

```
def mt5_deposit(amount, from_binary, to_mt5):
    """Performs a deposit operation to the MT5 account."""
    try:
        ws.send(json.dumps({
            "mt5_deposit": 1,
            "amount": amount,
            "from_binary": from_binary,
            "to_mt5": to_mt5
        }))
        response = ws.recv()
        return json.loads(response)
    except Exception as e:
        return {"error": f"Error during deposit operation: {e}"}
```

5\. Reading Commands from MQL5

This is a section in the code where a command from MQL5 is checked, and processed if available, and ensure the same command is not processed repeatedly. To read commands from MQL5, it checks if the file _mql5\_to\_python.txt_ exists. This is where MQL5 writes commands that need to be processed. If the file exists, it reads its content. By stripping unnecessary whitespace and checking for a BOM (Byte Order Mark), to ensure that the data is correctly handled, regardless of formatting inconsistencies.

- Once the command is read, it is printed for debugging purposes, so it can verify that the content was retrieved as expected. After reading the command, it deletes the file, ensuring that the same command won’t be processed again in the future.
- If the file doesn’t exist, it returns None, signaling that there is no command to process. This helps prevent unnecessary checks if there’s no command available.

```
def read_command():
    """Reads a command from the MQL5 file and deletes the file after reading."""
    print(f"Checking for command file at: {MQL5_TO_PYTHON}")
    if os.path.exists(MQL5_TO_PYTHON):
        print(f"Command file found: {MQL5_TO_PYTHON}")
        with open(MQL5_TO_PYTHON, "r", encoding="utf-8") as file:
            command = file.read().strip()
        print(f"Raw Command read: {repr(command)}")

        # Strip potential BOM and whitespace
        if command.startswith("\ufeff"):
            command = command[1:]

        print(f"Processed Command: {repr(command)}")
        os.remove(MQL5_TO_PYTHON)  # Remove file after reading
        return command
    print(f"Command file not found at: {MQL5_TO_PYTHON}")
    return None
```

6\. Writing Responses to MQL5

This part sends the response back to MQL5 so that it can take further actions based on our result.  After processing a command and obtaining a response, it is sent back to MQL5. This is done by writing the response to the _python\_to\_mql5.txt_ file, where MQL5 can read it.

- To ensure the response is properly formatted, the Python dictionary is converted into a JSON string using _json.dumps()_. Writing this JSON string to the file ensures that the MQL5 system can interpret the response correctly.

- This step is critical because it completes the communication loop between Python and MQL5, allowing MQL5 to know whether the operation succeeded or failed and take appropriate action.

```
def write_response(response):
    """Writes a response to the MQL5 file."""
    with open(PYTHON_TO_MQL5, "w", encoding="utf-8") as file:
        file.write(json.dumps(response))
```

7\. Main Loop

We create a continuous loop to read and process commands from MQL5, ensuring the system operates in real-time. The main loop is the heart of the script, where everything comes together. After successfully connecting and authorizing with the Deriv API, it enters a loop where it continuously checks for new commands. The _read\_command_ function is called to check if there’s a new command to process.

- If a command is found, it is processed and the result is written back to MQL5. If the command file doesn’t exist or an error occurs, it is handled gracefully by printing an error message and exiting the loop.
- The loop is crucial for maintaining a responsive system. We ensured that the script doesn’t run endlessly or fail without providing useful feedback. By implementing try-except blocks, we protect the loop from unexpected errors and ensure that it doesn’t crash the script, but instead exits cleanly

```
if __name__ == "__main__":
    if not connect_to_deriv():
        print("Failed to authorize. Exiting.")
        exit(1)

    print("Connected and authorized. Waiting for commands...")
    while True:
        try:
            command = read_command()
            if command:
                print(f"Processing command: {command}")
                response = process_command(command)
                print(f"Response: {response}")
                write_response(response)
                print("Response written. Exiting loop.")
                break  # Exit the loop after processing one command
            else:
                print("No command file found. Exiting.")
                break  # Exit the loop if the command file is not found
        except Exception as e:
            print(f"Error in main loop: {e}")
            break  # Exit the loop on unexpected error
```

### Funds Manager EA

At this stage, we will continue exploring the development process of the Fund Manager EA. We will design this EA to monitor the account balance and propose a top-up if the balance falls below a specified threshold. The top-up operation will be triggered through the Python script we developed earlier. While this EA does not encompass all the functionalities of a fully-fledged trading Expert Advisor, we will focus on the specific code segment that integrates Broker API interactions. The discussion aims to demonstrate the potential for integrating Broker APIs with any EA.

Overview of the Development Plan:

One of the key components of this EA is the implementation of the _ShellExecuteW_ function, which will be used to launch the _deriv\_api\_handler.py_ script from within the EA code. The EA will monitor the current account balance and, if it detects that the balance is below the defined threshold, it will issue a command to initiate a deposit.

Typically, such operations are handled in the _[OnTimer()](https://www.mql5.com/en/docs/event_handlers/ontimer)_ function to allow periodic checks and automation over an extended period. However, for the purposes of testing and immediate feedback, I have opted to place these operations within the _[OnInit()](https://www.mql5.com/en/docs/event_handlers/oninit)_ function. This approach ensures that the API is tested immediately after the EA is launched. In the long term, the _OnTimer()_ function will be better suited for continuously monitoring the account balance and requesting a top-up as needed.

1\. Meta Information and Library Import

This section sets up metadata for the Expert Advisor (EA) and imports a system library to execute external commands. The _ShellExecuteW_ function from the _shell32.dll_ library is used to run external applications like a Python script, enabling communication with external systems or APIs.

```
//+------------------------------------------------------------------+
//|                                             Fund Manager EA.mq5  |
//|                                Copyright 2024, Clemence Benjamin |
//|             https://www.mql5.com/en/users/billionaire2024/seller |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, Clemence Benjamin"
#property link      "https://www.mql5.com/en/users/billionaire2024/seller"
#property version   "1.01"
#property description "Deposit and Withdraw funds between broker and trading account within the EA"

#import "shell32.dll"
int ShellExecuteW(int hwnd, string lpOperation, string lpFile, string lpParameters, string lpDirectory, int nShowCmd);
#import
```

2\. Input Parameters for Customization

Here, customizable input parameters allow users to adapt the EA to their specific needs without modifying the core code. The parameters include the balance threshold, the top-up amount, account IDs for fund transfers, and paths for the Python script and executable.

```
input double balance_threshold = 100000.0;  // Threshold balance for top-up
input double top_up_amount = 100000.0;      // Amount to top-up if balance is below threshold
input string from_binary = "CR0000000";     // Binary account ID to withdraw funds from. Replace zero with real one
input string to_mt5 = "MTRReplace";     // MT5 account ID to deposit funds into. Replace with your MT5 acc from Deriv Broker
input string python_script_path = "C:\\Users\\YourComputerName\\PathTo\\deriv_api_handler.py"; // Python script path
input string python_exe = "python";         // Python executable command (ensure Python is in PATH)
```

3\. Initialization: Balance Check and Deposit

The _OnInit_ function is the entry point of the EA. It first retrieves the current account balance using _AccountInfoDouble(ACCOUNT\_BALANCE)_. If the balance is below the specified threshold, the EA proceeds to initiate a deposit.

```
int OnInit()
  {
   double current_balance = AccountInfoDouble(ACCOUNT_BALANCE);
   Print("Current Account Balance: ", current_balance);

   if(current_balance < balance_threshold)
     {
      Print("Balance is below the threshold. Attempting a deposit...");
```

4\. Writing Deposit Command to a File

The EA creates a file to communicate with the Python script. The file contains a JSON-formatted string specifying the deposit command, including the amount, source, and destination account details. This file acts as the interface between MQL5 and the Python system.

```
string command_file = "mql5_to_python.txt";
string command_path = TerminalInfoString(TERMINAL_COMMONDATA_PATH) + "\\Files\\" + command_file;
int handle = FileOpen(command_file, FILE_WRITE | FILE_COMMON | FILE_TXT | FILE_ANSI);

string deposit_command = StringFormat("{\"mt5_deposit\": 1, \"amount\": %.2f, \"from_binary\": \"%s\", \"req_id\": 1, \"to_mt5\": \"%s\"}", top_up_amount, from_binary, to_mt5);
FileWrite(handle, deposit_command);
FileClose(handle);
Print("Deposit command written to file: ", command_path);
```

5\. Launching the Python Script

Using the ShellExecuteW function, the EA runs the Python script specified in the input parameters. The script processes the deposit request and interacts with external systems.

```
int result = ShellExecuteW(0, "open", python_exe, python_script_path, NULL, 1);
if(result <= 32)
  {
   Print("Failed to launch Python script. Error code: ", result);
   return(INIT_FAILED);
  }
else
  {
   Print("Python script launched successfully.");
  }
```

6\. Checking the Python Response

After running the Python script, the EA checks for a response file. If the file exists, the EA reads it to confirm whether the deposit was successful. The response file must contain a success message for the EA to proceed accordingly.

```
string response_file = "python_to_mql5.txt";
if(FileIsExist(response_file, FILE_COMMON))
  {
   handle = FileOpen(response_file, FILE_READ | FILE_COMMON | FILE_TXT | FILE_ANSI);
   string response = FileReadString(handle);
   FileClose(handle);
   Print("Response from Python: ", response);

   if(StringFind(response, "\"status\":\"success\"") >= 0)
     {
      Print("Deposit was successful.");
     }
   else
     {
      Print("Deposit failed. Response: ", response);
     }
  }
else
  {
   Print("Response file not found. Ensure the Python script is running and processing the command.");
  }
```

7\. Finalizing Initialization

If the balance is above the threshold, the EA skips the deposit process. The initialization ends successfully, and the EA enters its operational phase.

```
else
  {
   Print("Balance is above the threshold. No deposit attempt needed.");
  }

return(INIT_SUCCEEDED);
```

8\. Deinitialization

The _OnDeinit_ function logs when the EA is removed or deinitialized.

```
void OnDeinit(const int reason)
  {
   Print("Expert deinitialized.");
  }
```

### Demo Testing and Results

We will discuss two tests here;

1. Testing the Python script (deriv\_api\_handler.py) in Command Prompt
2. Testing the Fund Manager EA

Let's get into detail:

1\. Testing the Python script (deriv\_api\_handler.py)

I worked on my Python script using a free tool called Notepad++. Below is an animated clip demonstrating how to launch the command prompt directly from the editor. This approach is essential as it ensures the command prompt is directed to the folder containing the script, making it convenient to execute the script directly from its location.

![Launching cmd from Notepad++](https://c.mql5.com/2/110/ShareX_VAeTvUPHd1.gif)

Launching cmd from Notepad++

Once the command prompt window is open, you can type the command to execute the script. The goal is to verify successful authorization and access to the Deriv API.

Enter the following command into the command prompt and press Enter:

```
python deriv_api_handler.py
```

Response with default API credentials:

With the default script that lacks valid credentials, you will receive the response shown below. It is essential to ensure that you provide your working credentials (API Token and App ID) to establish proper authorization and functionality

```
Error during authorization: Handshake status 401 Unauthorized -+-+- {'date': 'Thu, 15 Jan 2025 08:43:53 GMT', 'content-type':
'application/json;charset=UTF-8', 'content-length': '24', 'connection': 'keep-alive', 'content-language': 'en', 'upgrade':
'websocket', 'sec-websocket-accept': 'yfwlFELh2d3KczdgV3OT8Nolp0Q=', 'cf-cache-status': 'DYNAMIC', 'server': 'cloudflare', 'cf-ray':
'902cd20129b638df-HRE', 'alt-svc': 'h3=":443"; ma=86400'} -+-+- b'{"error":"InvalidAppID"}'
Failed to authorize. Exiting.
```

Response with correct credentials:

In the result snippet below, using the correct credentials, we successfully establish a connection and authorization. The program then checks for the _mql5\_to\_python.txt_ file in the common file folder, as shown below. This text file contains the command information that the script processes and forwards through a WebSocket connection. The API sends back a response, and in this example, random account details are used to protect my credentials. You will need to use your correct details to achieve positive results.

Our goal was met successfully, as we received a response, and a _python\_to\_mql5.txt_ file was generated with the API's response, which is then communicated back to MetaTrader 5.

```
Connected and authorized. Waiting for commands...
Checking for command file at: C:/Users/YourComputerName/AppData/Roaming/MetaQuotes/Terminal/Common/Files/mql5_to_python.txt
Command file found: C:/Users/YourComputerName/AppData/Roaming/MetaQuotes/Terminal/Common/Files/mql5_to_python.txt
Raw Command read: '{"mt5_deposit": 1, "amount": 100000.00, "from_binary": "CR4000128", "req_id": 1, "to_mt5": "MTR130002534"}'
Processed Command: '{"mt5_deposit": 1, "amount": 100000.00, "from_binary": "CR4000128", "req_id": 1, "to_mt5": "MTR130002534"}'
Processing command: {"mt5_deposit": 1, "amount": 100000.00, "from_binary": "CR4000128", "req_id": 1, "to_mt5": "MTR130002534"}
Response: {'echo_req': {'amount': 100000, 'from_binary': 'CR4000128', 'mt5_deposit': 1, 'to_mt5': 'MTR130002534'}, 'error': {'code': 'PermissionDenied',
'message': 'Permission denied, requires payments scope(s).'}, 'msg_type': 'mt5_deposit'}
Response written. Exiting loop.
```

2\. Testing the Fund Manager EA

To test our Expert Advisor (EA), we launched it in MetaTrader 5. As mentioned earlier, we designed it to attempt a deposit during initialization. Since we are working with a demo account that has an equity of 10,000, I’ve set the balance threshold to 100,000 to trigger a deposit attempt if the current balance is below this value. Below is an animated screenshot showcasing the launch, along with input settings that allow for customization of these values.

![Launching the Fund Manager EA](https://c.mql5.com/2/110/ShareX_hqlmfvHBvq.gif)

Launching the Fund Manager EA

In the Experts tab of the Toolbox window in MetaTrader 5, you can view the log of all operations performed by the EA. Below, you'll find the log results displayed.

```
2025.01.14 11:49:56.012 Fund Manager EA1 (EURUSD,M1)    Current Account Balance: 10000.22
2025.01.14 11:49:56.012 Fund Manager EA1 (EURUSD,M1)    Balance is below the threshold. Attempting a deposit...
2025.01.14 11:49:56.013 Fund Manager EA1 (EURUSD,M1)    Deposit command written to file: C:\Users\BTA24\AppData\Roaming\MetaQuotes\Terminal\Common\Files\mql5_to_python.txt
2025.01.14 11:49:56.097 Fund Manager EA1 (EURUSD,M1)    Python script launched successfully.
2025.01.14 11:50:01.132 Fund Manager EA1 (EURUSD,M1)    Response file path: C:\Users\BTA24\AppData\Roaming\MetaQuotes\Terminal\Common\Files\python_to_mql5.txt
2025.01.14 11:50:01.133 Fund Manager EA1 (EURUSD,M1)    Response from Python: {"echo_req": {"amount": 100000, "from_binary": "CR4000128", "mt5_deposit": 1,
 "to_mt5": "MTR130002534"}, "error": {"code": "PermissionDenied", "message": "Permission denied, requires payments scope(s)."}, "msg_type": "mt5_deposit"}
2025.01.14 11:50:01.133 Fund Manager EA1 (EURUSD,M1)    Deposit failed. Response: {"echo_req": {"amount": 100000, "from_binary": "CR4000128", "mt5_deposit": 1,
"to_mt5": "MTR130002534"}, "error": {"code": "PermissionDenied", "message": "Permission denied, requires payments scope(s)."}, "msg_type": "mt5_deposit"}
```

The result above demonstrates a successful interaction between the Fund Manager EA, the Python script, and the Deriv API. Privately, I have successfully performed top-up operations using this setup. In this example, however, we encountered a "Permission Denied" error because random credentials were used to protect my personal API credentials.

### Conclusion

We have successfully integrated MQL5 and Python to interact with an external broker server through its API. This solution addresses the challenge of running out of funds during automated trading, particularly for accounts hosted on a VPS. The EA automatically replenishes the account when its balance falls below a set threshold. While our focus was on the Deriv API, similar integrations can be achieved with other broker APIs, many of which offer advanced features and varying levels of access. For demonstration purposes, we tested the EA using an empty account and verified its functionality through API responses. You can extend this system to include withdrawal capabilities and other advanced features.

The Deriv API application is simple and primarily facilitates the management of API tokens needed for EA communication. You can explore the API further to unlock additional possibilities. Attached are the Python script and the Fund Manager EA files for testing and expansion. Please don't hesitate to share your thoughts or ask questions in the comment section below to keep the discussion alive.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/16012.zip "Download all attachments in the single ZIP archive")

[deriv\_api\_handler.py](https://www.mql5.com/en/articles/download/16012/deriv_api_handler.py "Download deriv_api_handler.py")(3.82 KB)

[Fund\_Manager\_EA.mq5](https://www.mql5.com/en/articles/download/16012/fund_manager_ea.mq5 "Download Fund_Manager_EA.mq5")(5.13 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [From Novice to Expert: Higher Probability Signals](https://www.mql5.com/en/articles/20658)
- [From Novice to Expert: Navigating Market Irregularities](https://www.mql5.com/en/articles/20645)
- [From Novice to Expert: Automating Trade Discipline with an MQL5 Risk Enforcement EA](https://www.mql5.com/en/articles/20587)
- [From Novice to Expert: Trading the RSI with Market Structure Awareness](https://www.mql5.com/en/articles/20554)
- [From Novice to Expert: Developing a Geographic Market Awareness with MQL5 Visualization](https://www.mql5.com/en/articles/20417)
- [The MQL5 Standard Library Explorer (Part 5): Multiple Signal Expert](https://www.mql5.com/en/articles/20289)
- [The MQL5 Standard Library Explorer (Part 4): Custom Signal Library](https://www.mql5.com/en/articles/20266)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/479791)**
(6)


![Fernando Carreiro](https://c.mql5.com/avatar/2025/9/68d40cf8-38fb.png)

**[Fernando Carreiro](https://www.mql5.com/en/users/fmic)**
\|
17 Jan 2025 at 13:54

**[@Stanislav Korotky](https://www.mql5.com/en/users/marketeer) [#](https://www.mql5.com/en/forum/479791#comment_55662801):** AFAIK, MT5 allows for running Python scripts directly from Navigator, right on regular charts.

It is true that you can launch a Python script from the terminal via the **_Navigator_**, but it is not true that they operate "on the chart". They run externally and may use the **_Python API_**, but they will not directly interact in any way with the chart or any other visual component of the _MetaTrader 5_ terminal.

![Clemence Benjamin](https://c.mql5.com/avatar/2025/3/67df27c6-2936.png)

**[Clemence Benjamin](https://www.mql5.com/en/users/billionaire2024)**
\|
19 Jan 2025 at 14:21

**Fernando Carreiro [#](https://www.mql5.com/en/forum/479791#comment_55662899):**

It is true that you can launch a Python script from the terminal via the **_Navigator_**, but it is not true that they operate "on the chart". They run externally and may use the **_Python API_**, but they will not directly interact in any way with the chart or any other visual component of the _MetaTrader 5_ terminal.

I agree with you, esteemed sir.

![Fernando Carreiro](https://c.mql5.com/avatar/2025/9/68d40cf8-38fb.png)

**[Fernando Carreiro](https://www.mql5.com/en/users/fmic)**
\|
19 Jan 2025 at 14:34

**[@Clemence Benjamin](https://www.mql5.com/en/users/billionaire2024) [#](https://www.mql5.com/en/forum/479791#comment_55676095):** I agree with you, esteemed sir.

However, why do you need the "Python" interface to handle the broker's API?

In the case where a broker does not offer MetaTrader 5, then you can use MQL5 to directly communicate with the broker's API. There is no need for the Python interface at all.

MQL5 even has network sockets, and can easily implement web sockets. You can easily also implement calling REST API's too. And if need be, it can also make use of DLL calls.

Not to mention that MQL5 is way faster than Python. **In essence, there is no need to use Python for accessing the API.**

![Clemence Benjamin](https://c.mql5.com/avatar/2025/3/67df27c6-2936.png)

**[Clemence Benjamin](https://www.mql5.com/en/users/billionaire2024)**
\|
19 Jan 2025 at 19:58

**@Fernando Carreiro [#](https://www.mql5.com/en/forum/479791#comment_55676166):**

However, why do you need the "Python" interface to handle the broker's API?

In the case where a broker does not offer MetaTrader 5, then you can use MQL5 to directly communicate with the broker's API. There is no need for the Python interface at all.

MQL5 even has network sockets, and can easily implement web sockets. You can easily also implement calling REST API's too. And if need be, it can also make use of DLL calls.

Not to mention that MQL5 is way faster than Python. **In essence, there is no need to use Python for accessing the API.**

Yes, sir. I appreciate that you're highlighting straightforward and effective approaches—thank you for that!

While I wouldn't necessarily emphasize Python as a pressing need, I believe it all comes down to exploring how these languages can collaborate on the subject matter.

At some point, the need for integration might naturally arise

![Herman Makmur](https://c.mql5.com/avatar/avatar_na2.png)

**[Herman Makmur](https://www.mql5.com/en/users/spottypegasus)**
\|
6 Aug 2025 at 07:06

Hi everyone,

I need to make RISE/FALL trabsactions on DERIV through MQL5 with websocket connection....

I found this [https://www.mql5.com/en/articles/10275](https://www.mql5.com/en/articles/10275 "https://www.mql5.com/en/articles/10275") to retrieve the history ticks but NOT to do the CALL/PUT (placing order)

Can anybody help me on this?

Thanks and regards,

Herman

![Neural Network in Practice: Pseudoinverse (II)](https://c.mql5.com/2/84/Rede_neural_na_prstica__Pseudo_Inversa__LOGO_.png)[Neural Network in Practice: Pseudoinverse (II)](https://www.mql5.com/en/articles/13733)

Since these articles are educational in nature and are not intended to show the implementation of specific functionality, we will do things a little differently in this article. Instead of showing how to apply factorization to obtain the inverse of a matrix, we will focus on factorization of the pseudoinverse. The reason is that there is no point in showing how to get the general coefficient if we can do it in a special way. Even better, the reader can gain a deeper understanding of why things happen the way they do. So, let's now figure out why hardware is replacing software over time.

![Mastering Log Records (Part 3): Exploring Handlers to Save Logs](https://c.mql5.com/2/108/logify60x60__1.png)[Mastering Log Records (Part 3): Exploring Handlers to Save Logs](https://www.mql5.com/en/articles/16866)

In this article, we will explore the concept of handlers in the logging library, understand how they work, and create three initial implementations: Console, Database, and File. We will cover everything from the basic structure of handlers to practical testing, preparing the ground for their full functionality in future articles.

![Neural Networks in Trading: Spatio-Temporal Neural Network (STNN)](https://c.mql5.com/2/84/Neural_networks_in_trading_STNN___LOGO.png)[Neural Networks in Trading: Spatio-Temporal Neural Network (STNN)](https://www.mql5.com/en/articles/15290)

In this article we will talk about using space-time transformations to effectively predict upcoming price movement. To improve the numerical prediction accuracy in STNN, a continuous attention mechanism is proposed that allows the model to better consider important aspects of the data.

![Neural Networks in Trading: Dual-Attention-Based Trend Prediction Model](https://c.mql5.com/2/83/Neural_networks_made_easy__A_dual_attention_model_for_trend_forecasting___LOGO.png)[Neural Networks in Trading: Dual-Attention-Based Trend Prediction Model](https://www.mql5.com/en/articles/15255)

We continue the discussion about the use of piecewise linear representation of time series, which was started in the previous article. Today we will see how to combine this method with other approaches to time series analysis to improve the price trend prediction quality.

[![](https://www.mql5.com/ff/si/s2n3m9ymjh52n07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F523%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dchoose.signals%26utm_content%3Dsubscribe.signal%26utm_campaign%3D0622.MQL5.com.Internal&a=fyznzyduwsltgnhlftytumasbfgbwlqw&s=91bc0eca8f132d3df7d14cdb1baebac753aef179403d60dc83856af55a4d6769&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=fydqrwkkmdirpitwvintmennhcqdqzqu&ssn=1769091409267273036&ssn_dr=0&ssn_sr=0&fv_date=1769091409&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F16012&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Integration%20of%20Broker%20APIs%20with%20Expert%20Advisors%20using%20MQL5%20and%20Python%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176909140945948036&fz_uniq=5049016111594316624&sv=2552)

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