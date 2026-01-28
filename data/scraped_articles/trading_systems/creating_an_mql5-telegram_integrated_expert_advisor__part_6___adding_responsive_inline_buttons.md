---
title: Creating an MQL5-Telegram Integrated Expert Advisor (Part 6): Adding Responsive Inline Buttons
url: https://www.mql5.com/en/articles/15823
categories: Trading Systems, Integration, Expert Advisors
relevance_score: 9
scraped_at: 2026-01-22T17:36:27.908556
---

[![](https://www.mql5.com/ff/sh/bhdtjfb1zry09943z2/267b575d2182c180804d340af38ce02c.jpg)\\
Trade from your iPhone or Android device\\
\\
You only need an internet connection to use the new powerful MetaTrader 5 Web terminal\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=wtigumvtenarnsocpyfoqnanxrilnbxx&s=ec8c539e52b83881ff2d16eaff6913b25803952eb277cac55f670a102b2edc1f&uid=&ref=https://www.mql5.com/en/articles/15823&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5049247455712749545)

MetaTrader 5 / Trading systems


### Introduction

This article dives into making our [MetaQuotes Language 5](https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5 "https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5") (MQL5) Expert Advisor more responsive and interactive for users through Telegram. In the [fifth installment of this series](https://www.mql5.com/en/articles/15750), we laid the groundwork for our bot by implementing the ability to respond to commands and messages from Telegram and by creating custom keyboard buttons. In this segment, we’re upping the interactivity of our bot by integrating inline buttons that trigger various actions and respond dynamically to user inputs.

The article is organized to address a few key components. First, we will introduce inline buttons in [Telegram](https://www.mql5.com/go?link=https://telegram.org/ "https://telegram.org/") bots, including what they are, their usefulness, and the benefits they provide over other methods of creating a bot interface. Then, we will transition to discussing how to use these inline buttons in MQL5, so that they can be part of our Expert Advisor's user interface.

From there, we will demonstrate how to handle the callback queries sent from [Telegram](https://www.mql5.com/go?link=https://telegram.org/ "https://telegram.org/") when a user presses a button. This will involve processing the user's action and determining what is the next appropriate step for the bot to take in its conversation with the user. Finally, we will test the built-in functionality of the bot to ensure that everything works flawlessly. Here are the topics we will cover in the article through:

1. Introduction to Inline Buttons in Telegram Bots
2. Integrating Inline Buttons into MQL5
3. Handling Callback Queries for Button Actions
4. Testing the Implementation of the Inline Button States
5. Conclusion

By the end of this article, you'll gain a clear understanding of how to integrate and manage inline buttons within your [MQL5](https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5 "https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5") Expert Advisor for [Telegram](https://www.mql5.com/go?link=https://telegram.org/ "https://telegram.org/"), enhancing the functionality of your trading bot, and making it more responsive and interactive for users. Let's get started then.

### Introduction to Inline Buttons in Telegram Bots

Inline buttons are interactive elements that appear directly within Telegram bot messages, allowing users to perform actions with a single tap. These buttons utilize [JavaScript Object Notation](https://en.wikipedia.org/wiki/JSON "https://en.wikipedia.org/wiki/JSON") (JSON)-formatted inline keyboard markup to define their appearance and behavior, offering a more integrated and responsive interface compared to traditional methods. By embedding these buttons directly in messages, bots can provide users with a streamlined experience and immediate interaction without requiring additional text commands or messages. To provide an insight into what exactly we are talking about, we have provided a visual illustration of inline buttons as below:

![INLINE BUTTONS ILLUSTRATION](https://c.mql5.com/2/92/Screenshot_2024-09-09_010929.png)

The primary advantage of inline buttons over traditional reply keyboards lies in their ability to remain within the message itself, making interactions more seamless and contextually relevant. Inline buttons, defined using [JSON](https://en.wikipedia.org/wiki/JSON "https://en.wikipedia.org/wiki/JSON") structures, allow for complex user interactions and dynamic responses. This approach eliminates the need for separate menus or additional messages, thereby reducing clutter and enhancing user engagement by providing instant feedback and actions. With this insight, we can now begin their implementation in MQL5 for [MetaTrader 5](https://www.metatrader5.com/en "https://www.metatrader5.com/en") as in the next section.

### Integrating Inline Buttons into MQL5

Incorporating inline buttons into our [MQL5](https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5 "https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5") Expert Advisor (EA) requires a framework that can handle the interactions of messages and the various states of the buttons. This is achieved by creating and managing several classes that are each responsible for a different part of the processing of calls and messages. We will explain the integration in detail, including the classes and functions used, and, more importantly, why they are used and how they contribute to the functioning of the bot with inline buttons. The first thing we have to do is design a class that can encapsulate all the particulars of a message received from Telegram.

```
//+------------------------------------------------------------------+
//|        Class_Message                                             |
//+------------------------------------------------------------------+
class Class_Message : public CObject {
public:
    Class_Message(); // Constructor
    ~Class_Message(){}; // Destructor

    bool              done; //--- Indicates if a message has been processed.
    long              update_id; //--- Stores the update ID from Telegram.
    long              message_id; //--- Stores the message ID.
    long              from_id; //--- Stores the sender’s ID.
    string            from_first_name; //--- Stores the sender’s first name.
    string            from_last_name; //--- Stores the sender’s last name.
    string            from_username; //--- Stores the sender’s username.
    long              chat_id; //--- Stores the chat ID.
    string            chat_first_name; //--- Stores the chat’s first name.
    string            chat_last_name; //--- Stores the chat’s last name.
    string            chat_username; //--- Stores the chat’s username.
    string            chat_type; //--- Stores the chat type.
    datetime          message_date; //--- Stores the date of the message.
    string            message_text; //--- Stores the text of the message.
};
```

Here, we define the "Class\_Message" class, which serves as a container for all relevant details about messages received from [Telegram](https://www.mql5.com/go?link=https://telegram.org/ "https://telegram.org/"). This class is essential for managing and processing message data within our MQL5 Expert Advisor (EA).

In this class, we include several public attributes that capture specific aspects of a message. The "done" attribute indicates whether the message has been processed. The "update\_id" and "message\_id" attributes store unique identifiers for the update and the message, respectively. The "from\_id", "from\_first\_name", "from\_last\_name", and "from\_username" attributes hold information about the sender of the message. Similarly, "chat\_id", "chat\_first\_name", "chat\_last\_name", "chat\_username", and "chat\_type" capture details about the chat where the message was sent. The "message\_date" attribute records the date and time of the message, and "message\_text" stores the actual content of the message. After defining the class members, we can then proceed to initialize our class members.

```
//+------------------------------------------------------------------+
//|      Constructor to initialize class members                     |
//+------------------------------------------------------------------+
Class_Message::Class_Message(void) {
    done = false; //--- Sets default value indicating message not yet processed.
    update_id = 0; //--- Initializes update ID to zero.
    message_id = 0; //--- Initializes message ID to zero.
    from_id = 0; //--- Initializes sender ID to zero.
    from_first_name = NULL; //--- Sets sender's first name to NULL (empty).
    from_last_name = NULL; //--- Sets sender's last name to NULL (empty).
    from_username = NULL; //--- Sets sender's username to NULL (empty).
    chat_id = 0; //--- Initializes chat ID to zero.
    chat_first_name = NULL; //--- Sets chat’s first name to NULL (empty).
    chat_last_name = NULL; //--- Sets chat’s last name to NULL (empty).
    chat_username = NULL; //--- Sets chat’s username to NULL (empty).
    chat_type = NULL; //--- Sets chat type to NULL (empty).
    message_date = 0; //--- Initializes message date to zero.
    message_text = NULL; //--- Sets message text to NULL (empty).
}
```

Here, we initialize the "Class\_Message" constructor to set default values for all attributes of the class. The "done" attribute is set to false to indicate that the message has not been processed. We initialize "update\_id", "message\_id", "from\_id", and "chat\_id" to 0, with "from\_first\_name", "from\_last\_name", "from\_username", "chat\_first\_name", "chat\_last\_name", "chat\_username", and "chat\_type" set to [NULL](https://www.mql5.com/en/docs/basis/types/void) to indicate that these fields are empty. Finally, "message\_date" is set to 0, and "message\_text" is initialized to [NULL](https://www.mql5.com/en/docs/basis/types/void), ensuring that each new instance of "Class\_Message" starts with default values before being populated with actual data. Via the same logic, we define the chat class where we will store our chat update details as below:

```
//+------------------------------------------------------------------+
//|        Class_Chat                                                |
//+------------------------------------------------------------------+
class Class_Chat : public CObject {
public:
    Class_Chat(){}; //--- Declares an empty constructor.
    ~Class_Chat(){}; //--- Declares an empty destructor.
    long              member_id; //--- Stores the chat ID.
    int               member_state; //--- Stores the state of the chat.
    datetime          member_time; //--- Stores the time of chat activities.
    Class_Message     member_last; //--- Instance of Class_Message to store the last message.
    Class_Message     member_new_one; //--- Instance of Class_Message to store the new message.
};
```

After defining the chats class, we will need to define an extra class that will handle call-back queries. It will be essential for handling the specific data associated with callback queries, which differ from regular messages. [Callback](https://en.wikipedia.org/wiki/Callback_(computer_programming) "https://en.wikipedia.org/wiki/Callback_(computer_programming)") queries provide unique data, such as the callback data and the interaction that triggered the query, which is not present in standard messages. Thus, the class will enable us to capture and manage this specialized data effectively. Furthermore, it will allow us to handle user interactions with inline buttons in a distinct manner. This separation will ensure that we can accurately process and respond to button presses, distinguishing them from other types of messages and interactions. The implementation will be as follows:

```
//+------------------------------------------------------------------+
//|        Class_CallbackQuery                                       |
//+------------------------------------------------------------------+
class Class_CallbackQuery : public CObject {
public:
    string            id; //--- Stores the callback query ID.
    long              from_id; //--- Stores the sender’s ID.
    string            from_first_name; //--- Stores the sender’s first name.
    string            from_last_name; //--- Stores the sender’s last name.
    string            from_username; //--- Stores the sender’s username.
    long              message_id; //--- Stores the message ID related to the callback.
    string            message_text; //--- Stores the message text.
    string            data; //--- Stores the callback data.
    long              chat_id; //--- Stores the chat ID to send responses.
};
```

Here, we define a class named "Class\_CallbackQuery" to manage the data associated with callback queries from Telegram. This class is crucial for handling interactions with inline buttons. In the class, we declare various variables to store information specific to callback queries. The variable "id" holds the unique identifier for the callback query, allowing us to distinguish between different queries. "from\_id" stores the ID of the sender, which helps in identifying the user who triggered the callback. We use "from\_first\_name", "from\_last\_name", and "from\_username" to keep track of the sender's name details.

The "message\_id" variable captures the ID of the message related to the callback, while "message\_text" contains the text of that message. "data" holds the callback data that was sent with the inline button, which is crucial for determining the action to take based on the button pressed. Finally, "chat\_id" stores the chat ID where responses should be sent, ensuring that the reply reaches the correct chat context. The rest of the Expert's class definition and initialization remain the same other than we now need to include an extra custom function for processing callback queries.

```
//+------------------------------------------------------------------+
//|        Class_Bot_EA                                              |
//+------------------------------------------------------------------+
class Class_Bot_EA {
private:
    string            member_token; //--- Stores the bot’s token.
    string            member_name; //--- Stores the bot’s name.
    long              member_update_id; //--- Stores the last update ID processed by the bot.
    CArrayString      member_users_filter; //--- Array to filter messages from specific users.
    bool              member_first_remove; //--- Indicates if the first message should be removed.

protected:
    CList             member_chats; //--- List to store chat objects.

public:
    Class_Bot_EA(); //--- Constructor.
    ~Class_Bot_EA(){}; //--- Destructor.
    int getChatUpdates(); //--- Function to get updates from Telegram.
    void ProcessMessages(); //--- Function to process incoming messages.
    void ProcessCallbackQuery(Class_CallbackQuery &cb_query); //--- Function to process callback queries.
};

//+------------------------------------------------------------------+
//|   Constructor for Class_Bot_EA                                   |
//+------------------------------------------------------------------+
Class_Bot_EA::Class_Bot_EA(void) {
    member_token = NULL; //--- Initialize bot token to NULL (empty).
    member_token = getTrimmedToken(InpToken); //--- Assign bot token by trimming input token.
    member_name = NULL; //--- Initialize bot name to NULL.
    member_update_id = 0; //--- Initialize last update ID to zero.
    member_first_remove = true; //--- Set first remove flag to true.
    member_chats.Clear(); //--- Clear the list of chat objects.
    member_users_filter.Clear(); //--- Clear the user filter array.
}
```

After defining all the necessary classes, we can proceed to get the chat update details.

```
//+------------------------------------------------------------------+
//|   Function to get chat updates from Telegram                     |
//+------------------------------------------------------------------+
int Class_Bot_EA::getChatUpdates(void) {

    //...

    return 0; //--- Return 0 to indicate successful processing of updates.
}
```

Here, we define the function "getChatUpdates" as a method of the class "Class\_Bot\_EA". This function intends to pull updates from the Telegram API: updates that consist of either new messages or callback queries that the bot has not yet handled. The current implementation of "getChatUpdates" returns an integer value of 0, which conventionally signifies that the operation completed successfully. By returning 0, we signal that we have pulled the updates and processed them without running into any trouble. The next step for us is to fill in this function so that it does what it is intended to do: pull updates from the API.

```
    if (member_token == NULL) { //--- If bot token is empty
        Print("ERR: TOKEN EMPTY"); //--- Print error message indicating empty token.
        return (-1); //--- Return -1 to indicate error.
    }
```

We determine whether the "member\_token" variable is empty. If "member\_token" is [NULL](https://www.mql5.com/en/docs/basis/types/void), that means we haven't been given a bot token. So we let the user know by printing "ERR: TOKEN EMPTY" that there's a necessary piece of information that hasn't been provided, and we return -1 to signal an error condition that stops the function from going any further. If we pass this step, we can proceed to post the request to get the chat updates.

```
    string out; //--- String to hold response data.
    string url = TELEGRAM_BASE_URL + "/bot" + member_token + "/getUpdates"; //--- Construct URL for Telegram API.
    string params = "offset=" + IntegerToString(member_update_id); //--- Set parameters including the offset based on the last update ID.

    int res = postRequest(out, url, params, WEB_TIMEOUT); //--- Send a POST request to Telegram with a timeout.
```

Firstly, we define a variable called "out" that is of type string. We set up this "out" variable to hold the response data that we will get back from the Telegram API. We then construct the API URL needed to get the updates. To do this, we combine a "TELEGRAM\_BASE\_URL" with several other components: "/bot" and the token for the bot, which is held in "member\_token"; and "/getUpdates", which is the endpoint we hit to get updates from Telegram. The Telegram API is a part of the other main platform that our application uses, and the "getUpdates" method is how we pull new data in from that platform. We then go ahead and make the call to the API and allow the API to return new data to us from our application. We can then use the output to make further amendments.

```
    if (res == 0) { //--- If request succeeds (res = 0)
        CJSONValue obj_json(NULL, jv_UNDEF); //--- Create a JSON object to parse the response.
        bool done = obj_json.Deserialize(out); //--- Deserialize the response.
        if (!done) { //--- If deserialization fails
            Print("ERR: JSON PARSING"); //--- Print error message indicating JSON parsing error.
            return (-1); //--- Return -1 to indicate error.
        }

        bool ok = obj_json["ok"].ToBool(); //--- Check if the response has "ok" field set to true.
        if (!ok) { //--- If "ok" field is false
            Print("ERR: JSON NOT OK"); //--- Print error message indicating that JSON response is not okay.
            return (-1); //--- Return -1 to indicate error.
        }
    }
```

We begin by determining whether the request was successful by checking the value of the "res" variable. If "res" equals 0, we know the request succeeded, and we can proceed to deal with the response. We create a "CJSONValue" object, "obj\_json," which we use to parse the response. The object is initialized in the "NULL" state and with "jv\_UNDEF," which denotes an undefined state or an object prepared to receive some data. After parsing with "out," we have an object that contains the parsed data, or we've encountered an error during parsing.

If deserialization fails—it is indicated by the variable named "done" being false—we print out the error message "ERR: JSON PARSING" and return "-1" to signal a problem. If we successfully deserialize the data, we check to see if the response contains a field named "ok". We convert this to a boolean, using the "ToBool" method, and store the result in the variable "ok". If "ok" is false—meaning the request was not successful on the server's side—we print "ERR: JSON NOT OK" and return "-1". In this way, we ensure that we properly handle both the deserialization of the response and its content. We then continue to iterate over each response by employing the following logic.

```
        int total = ArraySize(obj_json["result"].m_elements); //--- Get the total number of update elements.
        for (int i = 0; i < total; i++) { //--- Iterate through each update element.
            CJSONValue obj_item = obj_json["result"].m_elements[i]; //--- Access individual update element.

            if (obj_item["message"].m_type != jv_UNDEF) { //--- Check if the update has a message.
                Class_Message obj_msg; //--- Create an instance of Class_Message to store the message details.
                obj_msg.update_id = obj_item["update_id"].ToInt(); //--- Extract and store update ID.
                obj_msg.message_id = obj_item["message"]["message_id"].ToInt(); //--- Extract and store message ID.
                obj_msg.message_date = (datetime)obj_item["message"]["date"].ToInt(); //--- Extract and store message date.
                obj_msg.message_text = obj_item["message"]["text"].ToStr(); //--- Extract and store message text.
                obj_msg.message_text = decodeStringCharacters(obj_msg.message_text); //--- Decode any special characters in the message text.
            }
        }
```

To begin our examination of the total number of update elements in the response, we employ the [ArraySize](https://www.mql5.com/en/docs/array/arraysize) function to count the elements within the "m\_elements" array of the "result" object in "obj\_json". We store the count in the variable "total". Next, we set up a loop that repetitively processes each update element from the "m\_elements" array. There are "total" elements to process; thus, the loop control variable ranges from 0 to "total" minus 1. During each iteration, the loop control variable's current value "i" indicates which element of the "m\_elements" array we are accessing. We assign the "i-th" element to the variable "obj\_item". We now check to see if the current update (the "obj\_item") contains a valid "message".

Next, we instantiate an object of the "Class\_Message" called "obj\_msg", which will hold the particulars of the message in question. The first field we populate in "obj\_msg" is its "update\_id" field. To do this, we extract the "update\_id" from the "obj\_item," convert it to an integer, and place it in "obj\_msg.update\_id." The next field we access in "obj\_msg" is its "message\_id" field. For this value, we again extract information from the "message" field of "obj\_item." We convert the value in the "message\_id" field of "obj\_item" to an integer and place it in "obj\_msg.message\_id." After this, we populate the "datetime" field of "obj\_msg" with the "date" value of "item." Following this, we populate the "message\_text" field in "obj\_msg." We extract the "text" value from "message," convert it to a string, and place it in "obj\_msg.message\_text." Finally, we use the "decodeStringCharacters" function to ensure that any special characters in the "message\_text" will render correctly. A similar approach is used to get the other response details.

```
                obj_msg.from_id = obj_item["message"]["from"]["id"].ToInt(); //--- Extract and store the sender's ID.
                obj_msg.from_first_name = obj_item["message"]["from"]["first_name"].ToStr(); //--- Extract and store the sender's first name.
                obj_msg.from_first_name = decodeStringCharacters(obj_msg.from_first_name); //--- Decode any special characters in the sender's first name.
                obj_msg.from_last_name = obj_item["message"]["from"]["last_name"].ToStr(); //--- Extract and store the sender's last name.
                obj_msg.from_last_name = decodeStringCharacters(obj_msg.from_last_name); //--- Decode any special characters in the sender's last name.
                obj_msg.from_username = obj_item["message"]["from"]["username"].ToStr(); //--- Extract and store the sender's username.
                obj_msg.from_username = decodeStringCharacters(obj_msg.from_username); //--- Decode any special characters in the sender's username.

                obj_msg.chat_id = obj_item["message"]["chat"]["id"].ToInt(); //--- Extract and store the chat ID.
                obj_msg.chat_first_name = obj_item["message"]["chat"]["first_name"].ToStr(); //--- Extract and store the chat's first name.
                obj_msg.chat_first_name = decodeStringCharacters(obj_msg.chat_first_name); //--- Decode any special characters in the chat's first name.
                obj_msg.chat_last_name = obj_item["message"]["chat"]["last_name"].ToStr(); //--- Extract and store the chat's last name.
                obj_msg.chat_last_name = decodeStringCharacters(obj_msg.chat_last_name); //--- Decode any special characters in the chat's last name.
                obj_msg.chat_username = obj_item["message"]["chat"]["username"].ToStr(); //--- Extract and store the chat's username.
                obj_msg.chat_username = decodeStringCharacters(obj_msg.chat_username); //--- Decode any special characters in the chat's username.
                obj_msg.chat_type = obj_item["message"]["chat"]["type"].ToStr(); //--- Extract and store the chat type.
```

After getting the chat details, we proceed to process the message based on its associated chat ID.

```
                //--- Process the message based on chat ID.
                member_update_id = obj_msg.update_id + 1; //--- Update the last processed update ID.
```

After extracting and storing the necessary message details, we update the last processed update ID. We achieve this by assigning the value of "obj\_msg.update\_id" plus 1 to the variable "member\_update\_id". This ensures that the next time we process updates, we can skip over this update and continue from the next one. Finally, we need to apply a filter check on the user messages.

```
                //--- Check if we need to filter messages based on user or if no filter is applied.
                if (member_users_filter.Total() == 0 ||
                    (member_users_filter.Total() > 0 &&
                    member_users_filter.SearchLinear(obj_msg.from_username) >= 0)) {

                    int index = -1; //--- Initialize index to -1 (indicating no chat found).
                    for (int j = 0; j < member_chats.Total(); j++) { //--- Iterate through all chat objects.
                        Class_Chat *chat = member_chats.GetNodeAtIndex(j); //--- Get chat object by index.
                        if (chat.member_id == obj_msg.chat_id) { //--- If chat ID matches
                            index = j; //--- Store the index.
                            break; //--- Break the loop since we found the chat.
                        }
                    }

                    if (index == -1) { //--- If no matching chat was found
                        member_chats.Add(new Class_Chat); //--- Create a new chat object and add it to the list.
                        Class_Chat *chat = member_chats.GetLastNode(); //--- Get the last (newly added) chat object.
                        chat.member_id = obj_msg.chat_id; //--- Assign the chat ID.
                        chat.member_time = TimeLocal(); //--- Record the current time for the chat.
                        chat.member_state = 0; //--- Initialize the chat state to 0.
                        chat.member_new_one.message_text = obj_msg.message_text; //--- Store the new message in the chat.
                        chat.member_new_one.done = false; //--- Mark the new message as not processed.
                    } else { //--- If matching chat was found
                        Class_Chat *chat = member_chats.GetNodeAtIndex(index); //--- Get the chat object by index.
                        chat.member_time = TimeLocal(); //--- Update the time for the chat.
                        chat.member_new_one.message_text = obj_msg.message_text; //--- Store the new message.
                        chat.member_new_one.done = false; //--- Mark the new message as not processed.
                    }
                }
```

To filter messages based on the user or to allow all messages to pass through without filtering, we first check whether "member\_users\_filter" contains any elements. If the filter is empty ("Total == 0"), we let all messages go through. If the filter contains elements ("Total > 0"), we check whether the username of the sender ("obj\_msg.from\_username") is present in the filter. We use a sequential search method, "SearchLinear", in which the sender's username is checked against the filter to see if it is present. If the username is found (the method returns an index of 0 or more), we proceed to process the message normally. After this filtering step, we look up the chat of the message. We search the sender's username in the filter so that only certain usernames (those that are the sender's above in the filter) can pass through.

When the "chat.member\_id" is the same as the message's chat ID ("obj\_msg.chat\_id"), we first record the current index in the variable "index" during a loop and subsequently break out of that loop since we have located the correct chat. When we find no matches for the chat and "index" furthermore stays -1, we whip up a fresh chat object and plop it into the "member\_chats" using the "Add" method. "GetLastNode" then helps us to harvest the newly minted chat object, which we keep in the [pointer](https://www.mql5.com/en/docs/basis/types/object_pointers) "chat". We bestow the chat ID from "obj\_msg.chat\_id" on "chat.member\_id" and employ the [TimeLocal](https://www.mql5.com/en/docs/dateandtime/timelocal) function to peg the current time onto "chat.member\_time". We set the chat's "member\_state" at the very beginning to 0 and stash the new message in "chat.member\_new\_one.message\_text".

We also indicate that the message is unprocessed by setting "chat.member\_new\_one.done" to false. If we find a matching chat (i.e., "index" is not -1), we retrieve the corresponding chat object with "GetNodeAtIndex" and update its "member\_time" with the current time. We then stick the new message in "chat.member\_new\_one.message\_text" and again mark it as unprocessed, by setting "chat.member\_new\_one.done" to false. This ensures that the chat gets to be updated with the most recent message and that the system is aware that the message has not yet been processed. Next, we need to handle callback queries from Telegram chats.

```
            //--- Handle callback queries from Telegram.
            if (obj_item["callback_query"].m_type != jv_UNDEF) { //--- Check if there is a callback query in the update.
                Class_CallbackQuery obj_cb_query; //--- Create an instance of Class_CallbackQuery.

                //...

            }
```

We start by checking if the current update ("obj\_item") contains a callback query by verifying whether the "callback\_query" field's type ("m\_type") is not equal to "jv\_UNDEF". This ensures that a callback query exists within the update. If this condition is met, we proceed by creating an instance of the "Class\_CallbackQuery" object, named "obj\_cb\_query". This object will be used to store and manage the details of the callback query. We can then use the object to get and store the callback query data.

```
                obj_cb_query.id = obj_item["callback_query"]["id"].ToStr(); //--- Extract and store the callback query ID.
                obj_cb_query.from_id = obj_item["callback_query"]["from"]["id"].ToInt(); //--- Extract and store the sender's ID.
                obj_cb_query.from_first_name = obj_item["callback_query"]["from"]["first_name"].ToStr(); //--- Extract and store the sender's first name.
                obj_cb_query.from_first_name = decodeStringCharacters(obj_cb_query.from_first_name); //--- Decode any special characters in the sender's first name.
                obj_cb_query.from_last_name = obj_item["callback_query"]["from"]["last_name"].ToStr(); //--- Extract and store the sender's last name.
                obj_cb_query.from_last_name = decodeStringCharacters(obj_cb_query.from_last_name); //--- Decode any special characters in the sender's last name.
                obj_cb_query.from_username = obj_item["callback_query"]["from"]["username"].ToStr(); //--- Extract and store the sender's username.
                obj_cb_query.from_username = decodeStringCharacters(obj_cb_query.from_username); //--- Decode any special characters in the sender's username.
                obj_cb_query.message_id = obj_item["callback_query"]["message"]["message_id"].ToInt(); //--- Extract and store the message ID related to the callback.
                obj_cb_query.message_text = obj_item["callback_query"]["message"]["text"].ToStr(); //--- Extract and store the message text related to the callback.
                obj_cb_query.message_text = decodeStringCharacters(obj_cb_query.message_text); //--- Decode any special characters in the message text.
                obj_cb_query.data = obj_item["callback_query"]["data"].ToStr(); //--- Extract and store the callback data.
                obj_cb_query.data = decodeStringCharacters(obj_cb_query.data); //--- Decode any special characters in the callback data.

                obj_cb_query.chat_id = obj_item["callback_query"]["message"]["chat"]["id"].ToInt(); //--- Extract and store the chat ID.
```

We start with the details of the callback query itself. The callback query ID is taken from the "callback\_query" field. We use the "ToStr" method to convert it to string format, and we store it in "obj\_cb\_query.id". The next piece of information we extract is the sender's ID, which is taken from the "from" field. Again, we use the "ToInt" method, and we store the converted number in "obj\_cb\_query.from\_id". After that, we take the sender's first name, which is in the "from" field, and we convert it to string format. The sender's first name is stored in "obj\_cb\_query.from\_first\_name". The last thing we do with the first name is use the "decodeStringCharacters" function to decode any special characters that might be in the first name.

In parallel, we obtain the last name of the person sending the message, transform it into a string, and place it in "obj\_cb\_query.from\_last\_name." As before, we call on "decodeStringCharacters" to unmask any special characters in the last name. The process for obtaining the sender's username is the same: we extract the username, store it in "obj\_cb\_query.from\_username," and employ "decodeStringCharacters" to work on any special characters that might inhibit the proper function of the username in the future.

Next, we focus on the callback query's associated message. We take the message ID from the "message" field, convert it to an integer, and store it in "obj\_cb\_query.message\_id". Meanwhile, the message text is also extracted and converted to a string, which is stored in "obj\_cb\_query.message\_text". Any special characters in the text are decoded. We then turn our attention to the callback data. We extract it, convert it to a string, and store it in "obj\_cb\_query.data". Like any other data, the callback data is special character-encoded.

At last, we get from the callback query the ID of the chat to which the message was sent, convert that to an integer, and put it in "obj\_cb\_query.chat\_id." This gives us the complete set of information about the callback query, including what user was in the chat, what the message was, and what the callback data was. We then proceed to process the data and update the iteration.

```
                ProcessCallbackQuery(obj_cb_query); //--- Call function to process the callback query.

                member_update_id = obj_item["update_id"].ToInt() + 1; //--- Update the last processed update ID for callback queries.
```

Here, we call the "ProcessCallbackQuery" function, passing the "obj\_cb\_query" object as an argument. This function is responsible for handling the callback query and processing the extracted details we gathered earlier, such as the user information, chat ID, message text, and callback data. By calling this function, we ensure that the callback query is handled appropriately based on its specific contents.

After processing the callback query, we update the last processed update ID by retrieving the "update\_id" from the "obj\_item" field, converting it to an integer, and then adding 1 to it. This value is stored in "member\_update\_id", which tracks the most recent update processed. This step ensures that we do not reprocess the same callback query in future iterations, keeping track of the update progress efficiently. Finally, after processing the first message, we need to mark the message as already handled to avoid reprocessing.

```
        member_first_remove = false; //--- After processing the first message, mark that the first message has been handled.
```

We assign the variable "member\_first\_remove" to "false" after we handle the first message. This means we take care of the first message, and if anything special needs to be done for it, we have now done it. Why do we do this step? Its purpose is to mark that the first message has been handled and that it will not be handled again. By doing this, we ensure that any future logic that depends on the first message being unprocessed does not run because it does not need to.

The full source code snippet responsible for getting and storing both chat messages and callback queries is as below:

```
//+------------------------------------------------------------------+
//|   Function to get chat updates from Telegram                     |
//+------------------------------------------------------------------+
int Class_Bot_EA::getChatUpdates(void) {
    if (member_token == NULL) { //--- If bot token is empty
        Print("ERR: TOKEN EMPTY"); //--- Print error message indicating empty token.
        return (-1); //--- Return -1 to indicate error.
    }

    string out; //--- String to hold response data.
    string url = TELEGRAM_BASE_URL + "/bot" + member_token + "/getUpdates"; //--- Construct URL for Telegram API.
    string params = "offset=" + IntegerToString(member_update_id); //--- Set parameters including the offset based on the last update ID.

    int res = postRequest(out, url, params, WEB_TIMEOUT); //--- Send a POST request to Telegram with a timeout.

    if (res == 0) { //--- If request succeeds (res = 0)
        CJSONValue obj_json(NULL, jv_UNDEF); //--- Create a JSON object to parse the response.
        bool done = obj_json.Deserialize(out); //--- Deserialize the response.
        if (!done) { //--- If deserialization fails
            Print("ERR: JSON PARSING"); //--- Print error message indicating JSON parsing error.
            return (-1); //--- Return -1 to indicate error.
        }

        bool ok = obj_json["ok"].ToBool(); //--- Check if the response has "ok" field set to true.
        if (!ok) { //--- If "ok" field is false
            Print("ERR: JSON NOT OK"); //--- Print error message indicating that JSON response is not okay.
            return (-1); //--- Return -1 to indicate error.
        }

        int total = ArraySize(obj_json["result"].m_elements); //--- Get the total number of update elements.
        for (int i = 0; i < total; i++) { //--- Iterate through each update element.
            CJSONValue obj_item = obj_json["result"].m_elements[i]; //--- Access individual update element.

            if (obj_item["message"].m_type != jv_UNDEF) { //--- Check if the update has a message.
                Class_Message obj_msg; //--- Create an instance of Class_Message to store the message details.
                obj_msg.update_id = obj_item["update_id"].ToInt(); //--- Extract and store update ID.
                obj_msg.message_id = obj_item["message"]["message_id"].ToInt(); //--- Extract and store message ID.
                obj_msg.message_date = (datetime)obj_item["message"]["date"].ToInt(); //--- Extract and store message date.
                obj_msg.message_text = obj_item["message"]["text"].ToStr(); //--- Extract and store message text.
                obj_msg.message_text = decodeStringCharacters(obj_msg.message_text); //--- Decode any special characters in the message text.

                obj_msg.from_id = obj_item["message"]["from"]["id"].ToInt(); //--- Extract and store the sender's ID.
                obj_msg.from_first_name = obj_item["message"]["from"]["first_name"].ToStr(); //--- Extract and store the sender's first name.
                obj_msg.from_first_name = decodeStringCharacters(obj_msg.from_first_name); //--- Decode any special characters in the sender's first name.
                obj_msg.from_last_name = obj_item["message"]["from"]["last_name"].ToStr(); //--- Extract and store the sender's last name.
                obj_msg.from_last_name = decodeStringCharacters(obj_msg.from_last_name); //--- Decode any special characters in the sender's last name.
                obj_msg.from_username = obj_item["message"]["from"]["username"].ToStr(); //--- Extract and store the sender's username.
                obj_msg.from_username = decodeStringCharacters(obj_msg.from_username); //--- Decode any special characters in the sender's username.

                obj_msg.chat_id = obj_item["message"]["chat"]["id"].ToInt(); //--- Extract and store the chat ID.
                obj_msg.chat_first_name = obj_item["message"]["chat"]["first_name"].ToStr(); //--- Extract and store the chat's first name.
                obj_msg.chat_first_name = decodeStringCharacters(obj_msg.chat_first_name); //--- Decode any special characters in the chat's first name.
                obj_msg.chat_last_name = obj_item["message"]["chat"]["last_name"].ToStr(); //--- Extract and store the chat's last name.
                obj_msg.chat_last_name = decodeStringCharacters(obj_msg.chat_last_name); //--- Decode any special characters in the chat's last name.
                obj_msg.chat_username = obj_item["message"]["chat"]["username"].ToStr(); //--- Extract and store the chat's username.
                obj_msg.chat_username = decodeStringCharacters(obj_msg.chat_username); //--- Decode any special characters in the chat's username.
                obj_msg.chat_type = obj_item["message"]["chat"]["type"].ToStr(); //--- Extract and store the chat type.

                //--- Process the message based on chat ID.
                member_update_id = obj_msg.update_id + 1; //--- Update the last processed update ID.

                if (member_first_remove) { //--- If it's the first message after starting the bot
                    continue; //--- Skip processing it.
                }

                //--- Check if we need to filter messages based on user or if no filter is applied.
                if (member_users_filter.Total() == 0 ||
                    (member_users_filter.Total() > 0 &&
                    member_users_filter.SearchLinear(obj_msg.from_username) >= 0)) {

                    int index = -1; //--- Initialize index to -1 (indicating no chat found).
                    for (int j = 0; j < member_chats.Total(); j++) { //--- Iterate through all chat objects.
                        Class_Chat *chat = member_chats.GetNodeAtIndex(j); //--- Get chat object by index.
                        if (chat.member_id == obj_msg.chat_id) { //--- If chat ID matches
                            index = j; //--- Store the index.
                            break; //--- Break the loop since we found the chat.
                        }
                    }

                    if (index == -1) { //--- If no matching chat was found
                        member_chats.Add(new Class_Chat); //--- Create a new chat object and add it to the list.
                        Class_Chat *chat = member_chats.GetLastNode(); //--- Get the last (newly added) chat object.
                        chat.member_id = obj_msg.chat_id; //--- Assign the chat ID.
                        chat.member_time = TimeLocal(); //--- Record the current time for the chat.
                        chat.member_state = 0; //--- Initialize the chat state to 0.
                        chat.member_new_one.message_text = obj_msg.message_text; //--- Store the new message in the chat.
                        chat.member_new_one.done = false; //--- Mark the new message as not processed.
                    } else { //--- If matching chat was found
                        Class_Chat *chat = member_chats.GetNodeAtIndex(index); //--- Get the chat object by index.
                        chat.member_time = TimeLocal(); //--- Update the time for the chat.
                        chat.member_new_one.message_text = obj_msg.message_text; //--- Store the new message.
                        chat.member_new_one.done = false; //--- Mark the new message as not processed.
                    }
                }
            }


            //--- Handle callback queries from Telegram.
            if (obj_item["callback_query"].m_type != jv_UNDEF) { //--- Check if there is a callback query in the update.
                Class_CallbackQuery obj_cb_query; //--- Create an instance of Class_CallbackQuery.
                obj_cb_query.id = obj_item["callback_query"]["id"].ToStr(); //--- Extract and store the callback query ID.
                obj_cb_query.from_id = obj_item["callback_query"]["from"]["id"].ToInt(); //--- Extract and store the sender's ID.
                obj_cb_query.from_first_name = obj_item["callback_query"]["from"]["first_name"].ToStr(); //--- Extract and store the sender's first name.
                obj_cb_query.from_first_name = decodeStringCharacters(obj_cb_query.from_first_name); //--- Decode any special characters in the sender's first name.
                obj_cb_query.from_last_name = obj_item["callback_query"]["from"]["last_name"].ToStr(); //--- Extract and store the sender's last name.
                obj_cb_query.from_last_name = decodeStringCharacters(obj_cb_query.from_last_name); //--- Decode any special characters in the sender's last name.
                obj_cb_query.from_username = obj_item["callback_query"]["from"]["username"].ToStr(); //--- Extract and store the sender's username.
                obj_cb_query.from_username = decodeStringCharacters(obj_cb_query.from_username); //--- Decode any special characters in the sender's username.
                obj_cb_query.message_id = obj_item["callback_query"]["message"]["message_id"].ToInt(); //--- Extract and store the message ID related to the callback.
                obj_cb_query.message_text = obj_item["callback_query"]["message"]["text"].ToStr(); //--- Extract and store the message text related to the callback.
                obj_cb_query.message_text = decodeStringCharacters(obj_cb_query.message_text); //--- Decode any special characters in the message text.
                obj_cb_query.data = obj_item["callback_query"]["data"].ToStr(); //--- Extract and store the callback data.
                obj_cb_query.data = decodeStringCharacters(obj_cb_query.data); //--- Decode any special characters in the callback data.

                obj_cb_query.chat_id = obj_item["callback_query"]["message"]["chat"]["id"].ToInt(); //--- Extract and store the chat ID.

                ProcessCallbackQuery(obj_cb_query); //--- Call function to process the callback query.

                member_update_id = obj_item["update_id"].ToInt() + 1; //--- Update the last processed update ID for callback queries.
            }
        }

        member_first_remove = false; //--- After processing the first message, mark that the first message has been handled.
    }

    return 0; //--- Return 0 to indicate successful processing of updates.
}
```

After getting the chat updates, we now need to proceed to process the responses. This is handled in the following section.

### Handling Callback Queries for Button Actions

In this section, we handle incoming messages and respond with inline buttons based on specific commands. The first thing we need to do is process the initialization command or message sent by the user, and from there, we can instantiate the inline buttons and then proceed to get the callback queries. We can't skip this step because we can't just provide an inline keyboard without first having to get a command from the user. This is the logic we employ.

```
#define BTN_MENU "BTN_MENU" //--- Identifier for menu button

//+------------------------------------------------------------------+
//| Process new messages                                             |
//+------------------------------------------------------------------+
void Class_Bot_EA::ProcessMessages(void){
   //--- Loop through all chats
   for(int i=0; i<member_chats.Total(); i++){
      Class_Chat *chat = member_chats.GetNodeAtIndex(i); //--- Get the current chat
      if(!chat.member_new_one.done){ //--- Check if the message has not been processed yet
         chat.member_new_one.done = true; //--- Mark the message as processed
         string text = chat.member_new_one.message_text; //--- Get the message text

         //--- Example of sending a message with inline buttons
         if (text == "Start" || text == "/start" || text == "Help" || text == "/help"){
            string message = "Welcome! You can control me via inline buttons!"; //--- Welcome message
            //--- Define inline button to provide menu
            string buttons = "[[{\"text\": \"Provide Menu\", \"callback_data\": \""+BTN_MENU+"\"}]]";
            sendMessageToTelegram(chat.member_id, message, customInlineKeyboardMarkup(buttons)); //--- Send the inline keyboard markup
         }
      }
   }
}
```

In this instance, we set up a function called "ProcessMessages" to take care of user messages coming into our system. The very first thing this function does is iterate over the set of all chats we've stored in "member\_chats." For each of these chats, we get the chat object corresponding to the current chat by calling "GetNodeAtIndex(i)." Now that we have a handle on the current chat, we check if the message in "member\_new\_one" has already been processed. If it hasn't, we mark it as processed.

Next, we extract the actual content of the message using "chat.member\_new\_one.message\_text". We evaluate this content to determine whether the user has sent any commands, like "Start", "/start", "Help", or "/help". When we receive a command along these lines, we return a message that welcomes the user and tells them, "You can control me via inline buttons!" We then define an inline callback button that we want to serve as a menu option for the user. We use the "callback\_data" field of the button to indicate that it is related to "BTN\_MENU". We format the button as a JSON object and store it in the "buttons" variable.

In conclusion, the "sendMessageToTelegram" function is called to send the welcome message and our custom inline keyboard to the user. This function takes three parameters: the "chat.member\_id", the "message", and the button markup that is generated by the "customInlineKeyboardMarkup" function. The message along with our inline buttons is sent to the user. They can now interact with the inline buttons in the way that a typical user interacts with a Telegram bot. Since we are freshmen to the inline keyboard stuff, let us concentrate on its logic.

```
            string buttons = "[[{\"text\": \"Provide Menu\", \"callback_data\": \""+BTN_MENU+"\"}]]";
```

**Detailed Breakdown:**

> **Outer Brackets:** The entire string is enclosed in double quotes (" "), which is typical for defining string literals in many programming languages. Inside this string, we see the characters "\[\[ ... \]\]". These brackets are used to define the structure of the inline keyboard:

1. The first set of brackets \[ ... \] denotes an array of rows in the keyboard.
2. The second set of brackets \[ ... \] represents a row within that array. In this case, there is only one row.

**Button Definition:**

> Inside the second set of brackets, we have an object {"text": "Provide Menu", "callback\_data": " + BTN\_MENU + "}. This object defines a single button:

> 1. **"text":** This key specifies the label of the button, which is "Provide Menu". This is the text that will appear on the button when the user sees it.
> 2. **"callback\_data":** This key specifies the data that will be sent back to the bot when the button is clicked. In this case, the value is "BTN\_MENU", which is a constant we defined elsewhere in the code. This allows the bot to recognize which button was clicked and respond accordingly.

**Combining Elements:**

> The "BTN\_MENU" constant is inserted into the JSON string using string concatenation. This allows the dynamic inclusion of the button's callback data. For example, if "BTN\_MENU" is "BTN\_MENU", the resulting JSON would look like this: \[{"text": "Provide Menu", "callback\_data": "BTN\_MENU"}\].

**Final Format:**

> The final format of the buttons [string](https://www.mql5.com/en/docs/basis/types/stringconst), when used in the code, is: "\[ \[{ "text": "Provide Menu", "callback\_data": "BTN\_MENU" }\] \]". This format specifies that there is one row on the keyboard, and that row contains one button.

When the Telegram API receives this JSON structure, it interprets it as an inline keyboard with a single button. When a user clicks this button, the bot will receive the callback data "BTN\_MENU" in the callback query, which it can then use to determine the appropriate response. In the structure, we have used a custom function to create the inline button. Its logic is as below:

```
//+------------------------------------------------------------------+
//| Create a custom inline keyboard markup for Telegram              |
//+------------------------------------------------------------------+
string customInlineKeyboardMarkup(const string buttons){
   //--- Construct the JSON string for the inline keyboard markup
   string result = "{\"inline_keyboard\": " + UrlEncode(buttons) + "}"; //--- Encode buttons as JSON
   return(result);
}
```

The "customInlineKeyboardMarkup" function creates a custom inline keyboard markup for Telegram messages. To do this, we start with a string parameter, "buttons", which contains the JSON structure defining the inline buttons. Our job is to construct a JSON object that Telegram can use to render the inline keyboard. We begin by forming the JSON structure with the key "inline\_keyboard". Next, we use the "UrlEncode" function to handle any special characters that might be present in the "buttons" string. This encoding step is crucial because, without it, we might run into issues with special characters in the button definitions. After appending the encoded buttons string, we close the JSON object. The result string is a valid JSON representation of the inline keyboard markup. We return this string so that it can be sent to the Telegram API, which will then interactively render the inline keyboard in the message. Upon running the program, we have the following output.

![INITIALIZATION MESSAGE](https://c.mql5.com/2/92/Screenshot_2024-09-09_134848.png)

We can see that was a success. We did create the inline button. However, we can't yet respond to its clicks. Thus, we need to capture the callback query that is received and respond to the click respectively. To achieve this, we will need to create a function that gets the query data.

```
//+------------------------------------------------------------------+
//|   Function to process callback queries                           |
//+------------------------------------------------------------------+
void Class_Bot_EA::ProcessCallbackQuery(Class_CallbackQuery &cb_query) {
   Print("Callback Query ID: ", cb_query.id); //--- Log the callback query ID
   Print("Chat Token: ", member_token); //--- Log the member token
   Print("From First Name: ", cb_query.from_first_name); //--- Log the sender's first name
   Print("From Last Name: ", cb_query.from_last_name); //--- Log the sender's last name
   Print("From Username: ", cb_query.from_username); //--- Log the sender's username
   Print("Message ID: ", cb_query.message_id); //--- Log the message ID
   Print("Message Text: ", cb_query.message_text); //--- Log the message text
   Print("Callback Data: ", cb_query.data); //--- Log the callback data
}
```

The "ProcessCallbackQuery" function manages the details of a callback query that comes from [Telegram](https://www.mql5.com/go?link=https://telegram.org/ "https://telegram.org/"). It works on an instance of "Class\_CallbackQuery" that holds all the information associated with the callback. First, it logs the ID of the callback query, which is a unique identifier for the query and is essential for tracking and managing it. Next, the function logs the "member\_token." The role of this token is to indicate which bot or member is processing the callback and thus to ensure that the correct bot and only one bot are handling the query.

We then record the first and last names of the sender by using "cb\_query.from\_first\_name" and "cb\_query.from\_last\_name", respectively. These allow us to know the identity of the user who pressed the inline button and provide a personal touch should we ever need to address the user in the future. And speaking of identity, we also record the sender's username using "cb\_query.from\_username". This gives us another way to directly address the user in the future, should the need arise. After recording the sender's identity, we then log the ID of the message that was associated with the callback using "cb\_query.message\_id". Knowing this ID lets us know which specific message the button press is about.

Additionally, we log the message text using "cb\_query.message\_text". This provides context about the message when the button was clicked. We also log the callback data with "cb\_query.data". This data is what was sent back by the button click and is used to determine what action to take based on the user's interaction. By logging these details, we gain a comprehensive view of the callback query. This is handy for debugging and provides a better understanding of user interactions with the bot. Once we run the program, these are the outputs we get in the trading terminal.

![MT5 MESSAGES](https://c.mql5.com/2/92/Screenshot_2024-09-09_140945.png)

Since we know get the information, we can check the button clicked and generate a response accordingly. In our case, let us use the callback data from the action of the menu button. First, we will define the button constants. We have added detailed comments for easier understanding.

```
#define BTN_NAME "BTN_NAME" //--- Identifier for name button
#define BTN_INFO "BTN_INFO" //--- Identifier for info button
#define BTN_QUOTES "BTN_QUOTES" //--- Identifier for quotes button
#define BTN_MORE "BTN_MORE" //--- Identifier for more options button
#define BTN_SCREENSHOT "BTN_SCREENSHOT" //--- Identifier for screenshot button
#define EMOJI_CANCEL "\x274C" //--- Cross mark emoji

#define EMOJI_UP "\x2B06" //--- Upwards arrow emoji
#define BTN_BUY "BTN_BUY" //--- Identifier for buy button
#define BTN_CLOSE "BTN_CLOSE" //--- Identifier for close button
#define BTN_NEXT "BTN_NEXT" //--- Identifier for next button

#define EMOJI_PISTOL "\xF52B" //--- Pistol emoji
#define BTN_CONTACT "BTN_CONTACT" //--- Identifier for contact button
#define BTN_JOIN "BTN_JOIN" //--- Identifier for join button
```

After defining the function, we can then proceed to have the responses.

```
   //--- Respond based on the callback data
   string response_text;
   if (cb_query.data == BTN_MENU) {
      response_text = "You clicked "+BTN_MENU+"!"; //--- Prepare response text for BTN_MENU
      Print("RESPONSE = ", response_text); //--- Log the response
      //--- Send the response message to the correct group/channel chat ID
      sendMessageToTelegram(cb_query.chat_id, response_text, NULL);
      string message = "Information"; //--- Message to display options
      //--- Define inline buttons with callback data
      string buttons = "[[{\"text\": \"Get Expert's Name\", \"callback_data\": \""+BTN_NAME+"\"}],"\
                        "[{\"text\": \"Get Account Information\", \"callback_data\": \""+BTN_INFO+"\"}],"\
                        "[{\"text\": \"Get Current Market Quotes\", \"callback_data\": \""+BTN_QUOTES+"\"}],"\
                        "[{\"text\": \"More\", \"callback_data\": \""+BTN_MORE+"\"}, {\"text\": \"Screenshots\", \"callback_data\": \""+BTN_SCREENSHOT+"\"}, {\"text\": \""+EMOJI_CANCEL+"\", \"callback_data\": \""+EMOJI_CANCEL+"\"}]]";
      sendMessageToTelegram(cb_query.chat_id, message, customInlineKeyboardMarkup(buttons)); //--- Send the inline keyboard markup
   }
```

Here, we manage the response to a callback query based on its callback data. We start by initializing a string variable, "response\_text", to hold the message we want to send back to the user. We then check if the "callback\_data" from the callback query ("cb\_query.data") matches the constant "BTN\_MENU". If it does, we set "response\_text" to "You clicked "+BTN\_MENU+"!", which acknowledges the button press and includes the identifier for the clicked button. We log this response using the "Print" function to track what is being sent.

Next, we use the "sendMessageToTelegram" function to send the "response\_text" message to the chat identified by "cb\_query.chat\_id". Since we are sending a simple text message without an inline keyboard at this stage, the third parameter is [NULL](https://www.mql5.com/en/docs/basis/types/void), indicating that no additional keyboard markup is included.

After sending the initial message, we prepare a new message with the text "Information", which will provide the user with various options. We then define the inline buttons using a JSON-like structure in the "buttons" string. This structure includes buttons with labels such as "Get Expert's Name", "Get Account Information", "Get Current Market Quotes", "More", "Screenshots", and "Cancel". Each button is assigned specific "callback\_data" values, like "BTN\_NAME", "BTN\_INFO", "BTN\_QUOTES", "BTN\_MORE", "BTN\_SCREENSHOT", and "EMOJI\_CANCEL", which help identify which button was pressed.

Finally, we send this new message along with the inline keyboard using the "sendMessageToTelegram" function. The inline keyboard is formatted into JSON by the "customInlineKeyboardMarkup" function, ensuring that Telegram can correctly display the buttons. This approach allows us to engage users interactively by providing them with various options directly within the Telegram interface. Upon compilation, we get the following results.

![MENU INLINE RESPONSE](https://c.mql5.com/2/92/Screenshot_2024-09-09_143053.png)

That was a success. We now need to respond to the respective callback query data that is received from the specific inline buttons provided. We will first start with the one responsible for getting the program's name.

```
   else if (cb_query.data == BTN_NAME) {
      response_text = "You clicked "+BTN_NAME+"!"; //--- Prepare response text for BTN_NAME
      Print("RESPONSE = ", response_text); //--- Log the response
      string message = "The file name of the EA that I control is:\n"; //--- Message with EA file name
      message += "\xF50B"+__FILE__+" Enjoy.\n"; //--- Append the file name and a friendly message
      sendMessageToTelegram(cb_query.chat_id, message, NULL); //--- Send the message
   }
```

This section manages a particular callback query in which the callback\_data equals "BTN\_NAME". We start by setting up a response text in the "response\_text" variable. If the "callback\_data" matches "BTN\_NAME," we set "response\_text" to "You clicked " + BTN\_NAME + "!" This acknowledges the button press and includes the identifier of the clicked button. We then output this response using the "Print" function to keep an eye on what is being sent to the user.

We then create a novel message that conveys details about the EA (Expert Advisor) file over which the bot has control. This bot-generated missive opens with the words "The file name of the EA that I control is:\\n" and goes on to append the current source file's name, represented by [\_\_FILE\_\_](https://www.mql5.com/en/docs/constants/namedconstants/compilemacros), to the message, topped off with a friendly "Enjoy." One odd touch is that the missive opens with the character "\\xF50B," which represents an icon, a typographical flourish, or just a way to dazzle the reader on the bot's behalf.

In conclusion, we call the function "sendMessageToTelegram" to send the message to the chat that corresponds to "cb\_query.chat\_id". [NULL](https://www.mql5.com/en/docs/basis/types/void) is passed in for the third parameter, which means that no inline keyboard will accompany this message. When we click on the button, we have the following response.

![NAME GIF](https://c.mql5.com/2/92/NAME_GIF.gif)

That was a success. Now to get the responsiveness of the other buttons, that is account information and market price quotes, a similar approach is used.

```
   else if (cb_query.data == BTN_INFO) {
      response_text = "You clicked "+BTN_INFO+"!"; //--- Prepare response text for BTN_INFO
      Print("RESPONSE = ", response_text); //--- Log the response
      ushort MONEYBAG = 0xF4B0; //--- Define money bag emoji
      string MONEYBAGcode = ShortToString(MONEYBAG); //--- Convert emoji code to string
      string currency = AccountInfoString(ACCOUNT_CURRENCY); //--- Get the account currency
      //--- Construct the account information message
      string message = "\x2733\Account No: "+(string)AccountInfoInteger(ACCOUNT_LOGIN)+"\n";
      message += "\x23F0\Account Server: "+AccountInfoString(ACCOUNT_SERVER)+"\n";
      message += MONEYBAGcode+"Balance: "+(string)AccountInfoDouble(ACCOUNT_BALANCE)+" "+currency+"\n";
      message += "\x2705\Profit: "+(string)AccountInfoDouble(ACCOUNT_PROFIT)+" "+currency+"\n";
      sendMessageToTelegram(cb_query.chat_id, message, NULL); //--- Send the message
   }
   else if (cb_query.data == BTN_QUOTES) {
      response_text = "You clicked "+BTN_QUOTES+"!"; //--- Prepare response text for BTN_QUOTES
      Print("RESPONSE = ", response_text); //--- Log the response
      double Ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK); //--- Get the current ask price
      double Bid = SymbolInfoDouble(_Symbol, SYMBOL_BID); //--- Get the current bid price
      //--- Construct the market quotes message
      string message = "\xF170 Ask: "+(string)Ask+"\n";
      message += "\xF171 Bid: "+(string)Bid+"\n";
      sendMessageToTelegram(cb_query.chat_id, message, NULL); //--- Send the message
   }
```

Upon compilation, we get the following results:

![INFO AND QUOTES GIF](https://c.mql5.com/2/92/INFO_AND_QUOTES_GIF.gif)

That was a success. We now proceed to handle the "More" inline button as well. Up to this extent, you can see that we don't clutter the interface or chat field with messages. It is clean and we re-use the inline buttons efficiently.

```
   else if (cb_query.data == BTN_MORE) {
      response_text = "You clicked "+BTN_MORE+"!"; //--- Prepare response text for BTN_MORE
      Print("RESPONSE = ", response_text); //--- Log the response

      string message = "Choose More Options Below:\n"; //--- Message to prompt for additional options
      message += "Trading Operations"; //--- Title for trading operations
      //--- Define inline buttons for additional options
      string buttons = "[[{\"text\": \""+EMOJI_UP+"\", \"callback_data\": \""+EMOJI_UP+"\"}],"\
                        "[{\"text\": \"Buy\", \"callback_data\": \""+BTN_BUY+"\"}, {\"text\": \"Close\", \"callback_data\": \""+BTN_CLOSE+"\"}, {\"text\": \"Next\", \"callback_data\": \""+BTN_NEXT+"\"}]]";
      sendMessageToTelegram(cb_query.chat_id, message, customInlineKeyboardMarkup(buttons)); //--- Send the inline keyboard markup
   }
```

Here, we handle a callback query where the callback data is "BTN\_MORE". We start by preparing a response message stored in the "response\_text" variable. If the callback data matches "BTN\_MORE", we set "response\_text" to "You clicked "+BTN\_MORE+"!", which acknowledges the button press and includes the identifier for the clicked button. This response is logged using the "Print" function to keep track of what is being sent.

Next, we construct a new message that prompts the user to choose from additional options. The "message" variable begins with "Choose More Options Below:\\n", followed by "Trading Operations", which acts as a title for the set of options related to trading. We then define the inline buttons for these additional options using a JSON-like structure in the "buttons" string. This structure includes:

- A button with an emoji "EMOJI\_UP" and its corresponding "callback\_data" as "EMOJI\_UP".
- A row of buttons for various trading operations: "Buy", "Close", and "Next", each with their respective "callback\_data" values of "BTN\_BUY", "BTN\_CLOSE", and "BTN\_NEXT".

Finally, we use the "sendMessageToTelegram" function to send this message along with the inline keyboard to the chat identified by "cb\_query.chat\_id". The inline keyboard markup is formatted into JSON by the "customInlineKeyboardMarkup" function. If we click on this button, we should receive another extended button. This is as illustrated below:

![MORE BUTTON GIF](https://c.mql5.com/2/92/MORE_BTN_GIF.gif)

That went as expected. We now just work on the new buttons that emerge. First, it is the upward emoji button. If it is clicked, we want to go back to the previous menu, which is the main menu.

```
   else if (cb_query.data == EMOJI_UP) {
      response_text = "You clicked "+EMOJI_UP+"!"; //--- Prepare response text for EMOJI_UP
      Print("RESPONSE = ", response_text); //--- Log the response
      string message = "Choose a menu item:\n"; //--- Message to prompt for menu selection
      message += "Information"; //--- Title for information options
      //--- Define inline buttons for menu options
      string buttons = "[[{\"text\": \"Get Expert's Name\", \"callback_data\": \""+BTN_NAME+"\"}],"\
                        "[{\"text\": \"Get Account Information\", \"callback_data\": \""+BTN_INFO+"\"}],"\
                        "[{\"text\": \"Get Current Market Quotes\", \"callback_data\": \""+BTN_QUOTES+"\"}],"\
                        "[{\"text\": \"More\", \"callback_data\": \""+BTN_MORE+"\"}, {\"text\": \"Screenshots\", \"callback_data\": \""+BTN_SCREENSHOT+"\"}, {\"text\": \""+EMOJI_CANCEL+"\", \"callback_data\": \""+EMOJI_CANCEL+"\"}]]";
      sendMessageToTelegram(cb_query.chat_id, message, customInlineKeyboardMarkup(buttons)); //--- Send the inline keyboard markup
   }
```

Here, we just default and send the main menu's inline keyboard. In the same logic, we respond to the other buttons for position opening and closure operations as below.

```
   else if (cb_query.data == BTN_BUY) {
      response_text = "You clicked "+BTN_BUY+"!"; //--- Prepare response text for BTN_BUY
      Print("RESPONSE = ", response_text); //--- Log the response

      CTrade obj_trade; //--- Create a trade object
      double Ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK); //--- Get the current ask price
      double Bid = SymbolInfoDouble(_Symbol, SYMBOL_BID); //--- Get the current bid price
      //--- Open a buy position
      obj_trade.Buy(0.01, NULL, 0, Bid - 300 * _Point, Bid + 300 * _Point);
      double entry = 0, sl = 0, tp = 0, vol = 0;
      ulong ticket = obj_trade.ResultOrder(); //--- Get the ticket number of the new order
      if (ticket > 0) {
         if (PositionSelectByTicket(ticket)) { //--- Select the position by ticket
            entry = PositionGetDouble(POSITION_PRICE_OPEN); //--- Get the entry price
            sl = PositionGetDouble(POSITION_SL); //--- Get the stop loss price
            tp = PositionGetDouble(POSITION_TP); //--- Get the take profit price
            vol = PositionGetDouble(POSITION_VOLUME); //--- Get the volume
         }
      }
      //--- Construct the message with position details
      string message = "\xF340\Opened BUY Position:\n";
      message += "Ticket: "+(string)ticket+"\n";
      message += "Open Price: "+(string)entry+"\n";
      message += "Lots: "+(string)vol+"\n";
      message += "SL: "+(string)sl+"\n";
      message += "TP: "+(string)tp+"\n";
      sendMessageToTelegram(cb_query.chat_id, message, NULL); //--- Send the message
   }
   else if (cb_query.data == BTN_CLOSE) {
      response_text = "You clicked "+BTN_CLOSE+"!"; //--- Prepare response text for BTN_CLOSE
      Print("RESPONSE = ", response_text); //--- Log the response
      CTrade obj_trade; //--- Create a trade object
      int totalOpenBefore = PositionsTotal(); //--- Get the total number of open positions before closing
      obj_trade.PositionClose(_Symbol); //--- Close the position for the symbol
      int totalOpenAfter = PositionsTotal(); //--- Get the total number of open positions after closing
      //--- Construct the message with position closure details
      string message = "\xF62F\Closed Position:\n";
      message += "Total Positions (Before): "+(string)totalOpenBefore+"\n";
      message += "Total Positions (After): "+(string)totalOpenAfter+"\n";
      sendMessageToTelegram(cb_query.chat_id, message, NULL); //--- Send the message
   }
```

Upon running the program, here are the results we get.

![MORE OPERATIONS GIF](https://c.mql5.com/2/92/MORE_OPERATIONS_GIF.gif)

Fantastic. Similarly, we add the other control segments as below.

```
   else if (cb_query.data == BTN_NEXT) {
      response_text = "You clicked "+BTN_NEXT+"!"; //--- Prepare response text for BTN_NEXT
      Print("RESPONSE = ", response_text); //--- Log the response

      string message = "Choose Still More Options Below:\n"; //--- Message to prompt for further options
      message += "More Options"; //--- Title for more options
      //--- Define inline buttons for additional options
      string buttons = "[[{\"text\": \""+EMOJI_UP+"\", \"callback_data\": \""+EMOJI_UP+"\"}, {\"text\": \"Contact\", \"callback_data\": \""+BTN_CONTACT+"\"}, {\"text\": \"Join\", \"callback_data\": \""+BTN_JOIN+"\"},{\"text\": \""+EMOJI_PISTOL+"\", \"callback_data\": \""+EMOJI_PISTOL+"\"}]]";
      sendMessageToTelegram(cb_query.chat_id, message, customInlineKeyboardMarkup(buttons)); //--- Send the inline keyboard markup
   }
   else if (cb_query.data == BTN_CONTACT) {
      response_text = "You clicked "+BTN_CONTACT+"!"; //--- Prepare response text for BTN_CONTACT
      Print("RESPONSE = ", response_text); //--- Log the response
      string message = "Contact the developer via link below:\n"; //--- Message with contact link
      message += "https://t.me/Forex_Algo_Trader";
      sendMessageToTelegram(cb_query.chat_id, message, NULL); //--- Send the message
   }
   else if (cb_query.data == BTN_JOIN) {
      response_text = "You clicked "+BTN_JOIN+"!"; //--- Prepare response text for BTN_JOIN
      Print("RESPONSE = ", response_text); //--- Log the response
      string message = "You want to be part of our MQL5 Community?\n"; //--- Message inviting to join the community
      message += "Welcome! <a href=\"https://t.me/forexalgo_trading\">Click me</a> to join.\n";
      message += "<s>Civil Engineering</s> Forex AlgoTrading\n"; //--- Strikethrough text
      message += "<pre>This is a sample of our MQL5 code</pre>\n"; //--- Preformatted text
      message += "<u><i>Remember to follow community guidelines!\xF64F\</i></u>\n"; //--- Italic and underline text
      message += "<b>Happy Trading!</b>\n"; //--- Bold text
      sendMessageToTelegram(cb_query.chat_id, message, NULL); //--- Send the message
   }
   else if (cb_query.data == EMOJI_PISTOL) {
      response_text = "You clicked "+EMOJI_PISTOL+"!"; //--- Prepare response text for EMOJI_PISTOL
      Print("RESPONSE = ", response_text); //--- Log the response
      string message = "Choose More Options Below:\n"; //--- Message to prompt for more options
      message += "Trading Operations"; //--- Title for trading operations
      //--- Define inline buttons for additional trading options
      string buttons = "[[{\"text\": \""+EMOJI_UP+"\", \"callback_data\": \""+EMOJI_UP+"\"}],"\
                        "[{\"text\": \"Buy\", \"callback_data\": \""+BTN_BUY+"\"}, {\"text\": \"Close\", \"callback_data\": \""+BTN_CLOSE+"\"}, {\"text\": \"Next\", \"callback_data\": \""+BTN_NEXT+"\"}]]";
      sendMessageToTelegram(cb_query.chat_id, message, customInlineKeyboardMarkup(buttons)); //--- Send the inline keyboard markup
   }
```

This takes care of the buttons generated in the "Next" field as well as their responsiveness. We then need to take care of the screenshot button that is on the main inline button's menu.

```
   else if (cb_query.data == BTN_SCREENSHOT) {
      response_text = "You clicked "+BTN_SCREENSHOT+"!"; //--- Prepare response text for BTN_SCREENSHOT
      Print("RESPONSE = ", response_text); //--- Log the response

      string message = "Okay. Command 'get Current Chart Screenshot' received.\n"; //--- Message acknowledging screenshot command
      message += "Screenshot sending process initiated \xF60E"; //--- Emoji indicating process initiation
      sendMessageToTelegram(cb_query.chat_id, message, NULL); //--- Send the message
      string caption = "Screenshot of Symbol: "+_Symbol+ //--- Caption for screenshot
                       " ("+EnumToString(ENUM_TIMEFRAMES(_Period))+ //--- Timeframe
                       ") @ Time: "+TimeToString(TimeCurrent()); //--- Current time
      //--- Send the screenshot to Telegram
      sendScreenshotToTelegram(cb_query.chat_id, _Symbol, _Period, caption);
   }
```

Finally, we need to take care of the "Cancel" button by removing the current inline buttons, ready to start again.

```
   else if (cb_query.data == EMOJI_CANCEL) {
      response_text = "You clicked "+EMOJI_CANCEL+"!"; //--- Prepare response text for EMOJI_CANCEL
      Print("RESPONSE = ", response_text); //--- Log the response

      string message = "Choose /start or /help to begin."; //--- Message for user guidance
      sendMessageToTelegram(cb_query.chat_id, message, NULL); //--- Send the message
      //--- Reset the inline button state by removing the keyboard
      removeInlineButtons(member_token, cb_query.chat_id, cb_query.message_id);
   }
```

Here, a similar approach to the logic we have been using is implemented only that we have added another function to remove the inline buttons.

```
//+------------------------------------------------------------------+
//| Remove inline buttons by editing message reply markup            |
//+------------------------------------------------------------------+
void removeInlineButtons(string memberToken, long chatID, long messageID){
   //--- Reset the inline button state by removing the keyboard
   string url = TELEGRAM_BASE_URL + "/bot" + memberToken + "/editMessageReplyMarkup"; //--- API URL to edit message
   string params = "chat_id=" + IntegerToString(chatID) + //--- Chat ID parameter
                 "&message_id=" + IntegerToString(messageID) + //--- Message ID parameter
                 "&reply_markup=" + UrlEncode("{\"inline_keyboard\":[]}"); //--- Empty inline keyboard
   string response;
   int res = postRequest(response, url, params, WEB_TIMEOUT); //--- Send request to Telegram API
}
```

Here, we define the "removeInlineButtons" function. Its purpose is to get rid of inline buttons from a previously sent message by changing the message's reply markup. The function has three parameters: the "memberToken" (the bot's authentication token), the "chatID" (the ID of the chat where the message was sent), and the "messageID" (the ID of the message that contains the inline buttons). First, we construct the API endpoint URL for Telegram's "editMessageReplyMarkup" method. We do this by combining the "TELEGRAM\_BASE\_URL" with "/bot" and the "memberToken". That forms the URL we will use to communicate with Telegram's servers.

Then we specify the "params" string, which contains the required parameters for the API call. We include the "chat\_id" parameter. To get the value for it, we convert the variable "chatID" from an integer to a string. We do the same for the "message\_id" parameter. Finally, we tell the API to remove the inline buttons by sending an empty "reply\_markup" field. The value for this field is an empty JSON string, which we obtain by "UrlEncoding" the value of the variable "emptyInlineKeyboard".

Once we have set up the parameters, we declare the "response" variable to hold whatever the server sends back and call "postRequest" to send the API request to Telegram. The "postRequest" function sends the request using the provided URL and parameters along with a timeout ("WEB\_TIMEOUT") just in case things go wrong. If the request succeeds, we have our desired outcome—a message with no inline buttons, effectively resetting their state. If the callback data is unrecognized, we return a printout stating that the clicked button is unknown, meaning that the button is not recognized.

```
   else {
      response_text = "Unknown button!"; //--- Prepare response text for unknown buttons
      Print("RESPONSE = ", response_text); //--- Log the response
   }
```

When we click on the cancel button, we get the following output.

![CANCEL BUTTON GIF](https://c.mql5.com/2/92/CANCEL_GIF.gif)

That was a success. The full source code responsible for processing callback queries is as follows.

```
#define BTN_MENU "BTN_MENU" //--- Identifier for menu button

//+------------------------------------------------------------------+
//| Process new messages                                             |
//+------------------------------------------------------------------+
void Class_Bot_EA::ProcessMessages(void){
   //--- Loop through all chats
   for(int i=0; i<member_chats.Total(); i++){
      Class_Chat *chat = member_chats.GetNodeAtIndex(i); //--- Get the current chat
      if(!chat.member_new_one.done){ //--- Check if the message has not been processed yet
         chat.member_new_one.done = true; //--- Mark the message as processed
         string text = chat.member_new_one.message_text; //--- Get the message text

         //--- Example of sending a message with inline buttons
         if (text == "Start" || text == "/start" || text == "Help" || text == "/help"){
            string message = "Welcome! You can control me via inline buttons!"; //--- Welcome message
            //--- Define inline button to provide menu
            string buttons = "[[{\"text\": \"Provide Menu\", \"callback_data\": \""+BTN_MENU+"\"}]]";
            sendMessageToTelegram(chat.member_id, message, customInlineKeyboardMarkup(buttons)); //--- Send the inline keyboard markup
         }
      }
   }
}

#define BTN_NAME "BTN_NAME" //--- Identifier for name button
#define BTN_INFO "BTN_INFO" //--- Identifier for info button
#define BTN_QUOTES "BTN_QUOTES" //--- Identifier for quotes button
#define BTN_MORE "BTN_MORE" //--- Identifier for more options button
#define BTN_SCREENSHOT "BTN_SCREENSHOT" //--- Identifier for screenshot button
#define EMOJI_CANCEL "\x274C" //--- Cross mark emoji

#define EMOJI_UP "\x2B06" //--- Upwards arrow emoji
#define BTN_BUY "BTN_BUY" //--- Identifier for buy button
#define BTN_CLOSE "BTN_CLOSE" //--- Identifier for close button
#define BTN_NEXT "BTN_NEXT" //--- Identifier for next button

#define EMOJI_PISTOL "\xF52B" //--- Pistol emoji
#define BTN_CONTACT "BTN_CONTACT" //--- Identifier for contact button
#define BTN_JOIN "BTN_JOIN" //--- Identifier for join button

//+------------------------------------------------------------------+
//|   Function to process callback queries                           |
//+------------------------------------------------------------------+
void Class_Bot_EA::ProcessCallbackQuery(Class_CallbackQuery &cb_query) {
   Print("Callback Query ID: ", cb_query.id); //--- Log the callback query ID
   Print("Chat Token: ", member_token); //--- Log the member token
   Print("From First Name: ", cb_query.from_first_name); //--- Log the sender's first name
   Print("From Last Name: ", cb_query.from_last_name); //--- Log the sender's last name
   Print("From Username: ", cb_query.from_username); //--- Log the sender's username
   Print("Message ID: ", cb_query.message_id); //--- Log the message ID
   Print("Message Text: ", cb_query.message_text); //--- Log the message text
   Print("Callback Data: ", cb_query.data); //--- Log the callback data

   //--- Respond based on the callback data
   string response_text;
   if (cb_query.data == BTN_MENU) {
      response_text = "You clicked "+BTN_MENU+"!"; //--- Prepare response text for BTN_MENU
      Print("RESPONSE = ", response_text); //--- Log the response
      //--- Send the response message to the correct group/channel chat ID
      sendMessageToTelegram(cb_query.chat_id, response_text, NULL);
      string message = "Information"; //--- Message to display options
      //--- Define inline buttons with callback data
      string buttons = "[[{\"text\": \"Get Expert's Name\", \"callback_data\": \""+BTN_NAME+"\"}],"\
                        "[{\"text\": \"Get Account Information\", \"callback_data\": \""+BTN_INFO+"\"}],"\
                        "[{\"text\": \"Get Current Market Quotes\", \"callback_data\": \""+BTN_QUOTES+"\"}],"\
                        "[{\"text\": \"More\", \"callback_data\": \""+BTN_MORE+"\"}, {\"text\": \"Screenshots\", \"callback_data\": \""+BTN_SCREENSHOT+"\"}, {\"text\": \""+EMOJI_CANCEL+"\", \"callback_data\": \""+EMOJI_CANCEL+"\"}]]";
      sendMessageToTelegram(cb_query.chat_id, message, customInlineKeyboardMarkup(buttons)); //--- Send the inline keyboard markup
   }
   else if (cb_query.data == BTN_NAME) {
      response_text = "You clicked "+BTN_NAME+"!"; //--- Prepare response text for BTN_NAME
      Print("RESPONSE = ", response_text); //--- Log the response
      string message = "The file name of the EA that I control is:\n"; //--- Message with EA file name
      message += "\xF50B"+__FILE__+" Enjoy.\n"; //--- Append the file name and a friendly message
      sendMessageToTelegram(cb_query.chat_id, message, NULL); //--- Send the message
   }
   else if (cb_query.data == BTN_INFO) {
      response_text = "You clicked "+BTN_INFO+"!"; //--- Prepare response text for BTN_INFO
      Print("RESPONSE = ", response_text); //--- Log the response
      ushort MONEYBAG = 0xF4B0; //--- Define money bag emoji
      string MONEYBAGcode = ShortToString(MONEYBAG); //--- Convert emoji code to string
      string currency = AccountInfoString(ACCOUNT_CURRENCY); //--- Get the account currency
      //--- Construct the account information message
      string message = "\x2733\Account No: "+(string)AccountInfoInteger(ACCOUNT_LOGIN)+"\n";
      message += "\x23F0\Account Server: "+AccountInfoString(ACCOUNT_SERVER)+"\n";
      message += MONEYBAGcode+"Balance: "+(string)AccountInfoDouble(ACCOUNT_BALANCE)+" "+currency+"\n";
      message += "\x2705\Profit: "+(string)AccountInfoDouble(ACCOUNT_PROFIT)+" "+currency+"\n";
      sendMessageToTelegram(cb_query.chat_id, message, NULL); //--- Send the message
   }
   else if (cb_query.data == BTN_QUOTES) {
      response_text = "You clicked "+BTN_QUOTES+"!"; //--- Prepare response text for BTN_QUOTES
      Print("RESPONSE = ", response_text); //--- Log the response
      double Ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK); //--- Get the current ask price
      double Bid = SymbolInfoDouble(_Symbol, SYMBOL_BID); //--- Get the current bid price
      //--- Construct the market quotes message
      string message = "\xF170 Ask: "+(string)Ask+"\n";
      message += "\xF171 Bid: "+(string)Bid+"\n";
      sendMessageToTelegram(cb_query.chat_id, message, NULL); //--- Send the message
   }
   else if (cb_query.data == BTN_MORE) {
      response_text = "You clicked "+BTN_MORE+"!"; //--- Prepare response text for BTN_MORE
      Print("RESPONSE = ", response_text); //--- Log the response

      string message = "Choose More Options Below:\n"; //--- Message to prompt for additional options
      message += "Trading Operations"; //--- Title for trading operations
      //--- Define inline buttons for additional options
      string buttons = "[[{\"text\": \""+EMOJI_UP+"\", \"callback_data\": \""+EMOJI_UP+"\"}],"\
                        "[{\"text\": \"Buy\", \"callback_data\": \""+BTN_BUY+"\"}, {\"text\": \"Close\", \"callback_data\": \""+BTN_CLOSE+"\"}, {\"text\": \"Next\", \"callback_data\": \""+BTN_NEXT+"\"}]]";
      sendMessageToTelegram(cb_query.chat_id, message, customInlineKeyboardMarkup(buttons)); //--- Send the inline keyboard markup
   }
   else if (cb_query.data == EMOJI_CANCEL) {
      response_text = "You clicked "+EMOJI_CANCEL+"!"; //--- Prepare response text for EMOJI_CANCEL
      Print("RESPONSE = ", response_text); //--- Log the response

      string message = "Choose /start or /help to begin."; //--- Message for user guidance
      sendMessageToTelegram(cb_query.chat_id, message, NULL); //--- Send the message
      //--- Reset the inline button state by removing the keyboard
      removeInlineButtons(member_token, cb_query.chat_id, cb_query.message_id);
   }
   else if (cb_query.data == EMOJI_UP) {
      response_text = "You clicked "+EMOJI_UP+"!"; //--- Prepare response text for EMOJI_UP
      Print("RESPONSE = ", response_text); //--- Log the response
      string message = "Choose a menu item:\n"; //--- Message to prompt for menu selection
      message += "Information"; //--- Title for information options
      //--- Define inline buttons for menu options
      string buttons = "[[{\"text\": \"Get Expert's Name\", \"callback_data\": \""+BTN_NAME+"\"}],"\
                        "[{\"text\": \"Get Account Information\", \"callback_data\": \""+BTN_INFO+"\"}],"\
                        "[{\"text\": \"Get Current Market Quotes\", \"callback_data\": \""+BTN_QUOTES+"\"}],"\
                        "[{\"text\": \"More\", \"callback_data\": \""+BTN_MORE+"\"}, {\"text\": \"Screenshots\", \"callback_data\": \""+BTN_SCREENSHOT+"\"}, {\"text\": \""+EMOJI_CANCEL+"\", \"callback_data\": \""+EMOJI_CANCEL+"\"}]]";
      sendMessageToTelegram(cb_query.chat_id, message, customInlineKeyboardMarkup(buttons)); //--- Send the inline keyboard markup
   }
   else if (cb_query.data == BTN_BUY) {
      response_text = "You clicked "+BTN_BUY+"!"; //--- Prepare response text for BTN_BUY
      Print("RESPONSE = ", response_text); //--- Log the response

      CTrade obj_trade; //--- Create a trade object
      double Ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK); //--- Get the current ask price
      double Bid = SymbolInfoDouble(_Symbol, SYMBOL_BID); //--- Get the current bid price
      //--- Open a buy position
      obj_trade.Buy(0.01, NULL, 0, Bid - 300 * _Point, Bid + 300 * _Point);
      double entry = 0, sl = 0, tp = 0, vol = 0;
      ulong ticket = obj_trade.ResultOrder(); //--- Get the ticket number of the new order
      if (ticket > 0) {
         if (PositionSelectByTicket(ticket)) { //--- Select the position by ticket
            entry = PositionGetDouble(POSITION_PRICE_OPEN); //--- Get the entry price
            sl = PositionGetDouble(POSITION_SL); //--- Get the stop loss price
            tp = PositionGetDouble(POSITION_TP); //--- Get the take profit price
            vol = PositionGetDouble(POSITION_VOLUME); //--- Get the volume
         }
      }
      //--- Construct the message with position details
      string message = "\xF340\Opened BUY Position:\n";
      message += "Ticket: "+(string)ticket+"\n";
      message += "Open Price: "+(string)entry+"\n";
      message += "Lots: "+(string)vol+"\n";
      message += "SL: "+(string)sl+"\n";
      message += "TP: "+(string)tp+"\n";
      sendMessageToTelegram(cb_query.chat_id, message, NULL); //--- Send the message
   }
   else if (cb_query.data == BTN_CLOSE) {
      response_text = "You clicked "+BTN_CLOSE+"!"; //--- Prepare response text for BTN_CLOSE
      Print("RESPONSE = ", response_text); //--- Log the response
      CTrade obj_trade; //--- Create a trade object
      int totalOpenBefore = PositionsTotal(); //--- Get the total number of open positions before closing
      obj_trade.PositionClose(_Symbol); //--- Close the position for the symbol
      int totalOpenAfter = PositionsTotal(); //--- Get the total number of open positions after closing
      //--- Construct the message with position closure details
      string message = "\xF62F\Closed Position:\n";
      message += "Total Positions (Before): "+(string)totalOpenBefore+"\n";
      message += "Total Positions (After): "+(string)totalOpenAfter+"\n";
      sendMessageToTelegram(cb_query.chat_id, message, NULL); //--- Send the message
   }
   else if (cb_query.data == BTN_NEXT) {
      response_text = "You clicked "+BTN_NEXT+"!"; //--- Prepare response text for BTN_NEXT
      Print("RESPONSE = ", response_text); //--- Log the response

      string message = "Choose Still More Options Below:\n"; //--- Message to prompt for further options
      message += "More Options"; //--- Title for more options
      //--- Define inline buttons for additional options
      string buttons = "[[{\"text\": \""+EMOJI_UP+"\", \"callback_data\": \""+EMOJI_UP+"\"}, {\"text\": \"Contact\", \"callback_data\": \""+BTN_CONTACT+"\"}, {\"text\": \"Join\", \"callback_data\": \""+BTN_JOIN+"\"},{\"text\": \""+EMOJI_PISTOL+"\", \"callback_data\": \""+EMOJI_PISTOL+"\"}]]";
      sendMessageToTelegram(cb_query.chat_id, message, customInlineKeyboardMarkup(buttons)); //--- Send the inline keyboard markup
   }
   else if (cb_query.data == BTN_CONTACT) {
      response_text = "You clicked "+BTN_CONTACT+"!"; //--- Prepare response text for BTN_CONTACT
      Print("RESPONSE = ", response_text); //--- Log the response
      string message = "Contact the developer via link below:\n"; //--- Message with contact link
      message += "https://t.me/Forex_Algo_Trader";
      sendMessageToTelegram(cb_query.chat_id, message, NULL); //--- Send the message
   }
   else if (cb_query.data == BTN_JOIN) {
      response_text = "You clicked "+BTN_JOIN+"!"; //--- Prepare response text for BTN_JOIN
      Print("RESPONSE = ", response_text); //--- Log the response
      string message = "You want to be part of our MQL5 Community?\n"; //--- Message inviting to join the community
      message += "Welcome! <a href=\"https://t.me/forexalgo_trading\">Click me</a> to join.\n";
      message += "<s>Civil Engineering</s> Forex AlgoTrading\n"; //--- Strikethrough text
      message += "<pre>This is a sample of our MQL5 code</pre>\n"; //--- Preformatted text
      message += "<u><i>Remember to follow community guidelines!\xF64F\</i></u>\n"; //--- Italic and underline text
      message += "<b>Happy Trading!</b>\n"; //--- Bold text
      sendMessageToTelegram(cb_query.chat_id, message, NULL); //--- Send the message
   }
   else if (cb_query.data == EMOJI_PISTOL) {
      response_text = "You clicked "+EMOJI_PISTOL+"!"; //--- Prepare response text for EMOJI_PISTOL
      Print("RESPONSE = ", response_text); //--- Log the response
      string message = "Choose More Options Below:\n"; //--- Message to prompt for more options
      message += "Trading Operations"; //--- Title for trading operations
      //--- Define inline buttons for additional trading options
      string buttons = "[[{\"text\": \""+EMOJI_UP+"\", \"callback_data\": \""+EMOJI_UP+"\"}],"\
                        "[{\"text\": \"Buy\", \"callback_data\": \""+BTN_BUY+"\"}, {\"text\": \"Close\", \"callback_data\": \""+BTN_CLOSE+"\"}, {\"text\": \"Next\", \"callback_data\": \""+BTN_NEXT+"\"}]]";
      sendMessageToTelegram(cb_query.chat_id, message, customInlineKeyboardMarkup(buttons)); //--- Send the inline keyboard markup
   }
   else if (cb_query.data == BTN_SCREENSHOT) {
      response_text = "You clicked "+BTN_SCREENSHOT+"!"; //--- Prepare response text for BTN_SCREENSHOT
      Print("RESPONSE = ", response_text); //--- Log the response

      string message = "Okay. Command 'get Current Chart Screenshot' received.\n"; //--- Message acknowledging screenshot command
      message += "Screenshot sending process initiated \xF60E"; //--- Emoji indicating process initiation
      sendMessageToTelegram(cb_query.chat_id, message, NULL); //--- Send the message
      string caption = "Screenshot of Symbol: "+_Symbol+ //--- Caption for screenshot
                       " ("+EnumToString(ENUM_TIMEFRAMES(_Period))+ //--- Timeframe
                       ") @ Time: "+TimeToString(TimeCurrent()); //--- Current time
      //--- Send the screenshot to Telegram
      sendScreenshotToTelegram(cb_query.chat_id, _Symbol, _Period, caption);
   }
   else {
      response_text = "Unknown button!"; //--- Prepare response text for unknown buttons
      Print("RESPONSE = ", response_text); //--- Log the response
   }

   //--- Optionally, reset the inline button state by removing the keyboard
   // removeInlineButtons(member_token, cb_query.chat_id, cb_query.message_id);
}

//+------------------------------------------------------------------+
//| Create a custom inline keyboard markup for Telegram              |
//+------------------------------------------------------------------+
string customInlineKeyboardMarkup(const string buttons){
   //--- Construct the JSON string for the inline keyboard markup
   string result = "{\"inline_keyboard\": " + UrlEncode(buttons) + "}"; //--- Encode buttons as JSON
   return(result);
}

//+------------------------------------------------------------------+
//| Remove inline buttons by editing message reply markup            |
//+------------------------------------------------------------------+
void removeInlineButtons(string memberToken, long chatID, long messageID){
   //--- Reset the inline button state by removing the keyboard
   string url = TELEGRAM_BASE_URL + "/bot" + memberToken + "/editMessageReplyMarkup"; //--- API URL to edit message
   string params = "chat_id=" + IntegerToString(chatID) + //--- Chat ID parameter
                 "&message_id=" + IntegerToString(messageID) + //--- Message ID parameter
                 "&reply_markup=" + UrlEncode("{\"inline_keyboard\":[]}"); //--- Empty inline keyboard
   string response;
   int res = postRequest(response, url, params, WEB_TIMEOUT); //--- Send request to Telegram API
}
```

To briefly summarize, we manage [callback](https://en.wikipedia.org/wiki/Callback_(computer_programming) "https://en.wikipedia.org/wiki/Callback_(computer_programming)") queries by answering certain button presses and directing users to appropriate inline keyboard options. This enhances interaction by giving contextually appropriate messages and options. Next, we will test the integration to make sure that these functions work as they should and that interactions are processed correctly.

### Testing the Implementation of the Inline Button States

In this section, we will verify how well the inline buttons interact with the [Telegram](https://www.mql5.com/go?link=https://telegram.org/ "https://telegram.org/") bot and [MQL5](https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5 "https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5") Expert Advisor. This process will involve simulating user actions, like pressing buttons, and ensuring that the bot handles [callback](https://en.wikipedia.org/wiki/Callback_(computer_programming) "https://en.wikipedia.org/wiki/Callback_(computer_programming)") queries correctly. We'll assess the proper display, removal, or update of inline buttons based on user interaction. To provide further clarity, we created a video demonstrating how the integration functions, showcasing the step-by-step behavior of the bot when responding to inline button presses. This ensures that the setup works as expected in real time. Below is the illustration.

YouTube

We have tested the button interactions and callback queries to ensure the bot works accurately with user inputs and that inline button states are updated or reset as needed. This offers a non-linear interaction style, which enhances engagement and provides a more efficient experience when controlling the bot through Telegram.

### Conclusion

To conclude, we have put in place and tested the passage of [callback](https://en.wikipedia.org/wiki/Callback_(computer_programming) "https://en.wikipedia.org/wiki/Callback_(computer_programming)") queries and inline buttons in the [Telegram](https://www.mql5.com/go?link=https://telegram.org/ "https://telegram.org/") bot. Now, the bot can respond to user inputs with tailored messages and offer interactive options through inline keyboards. The user experience has been enhanced by the addition of real-time, easy-to-use buttons for actions like accessing menus, getting expert information, or executing commands related to trading.

We've tested the system, and we can confirm that it works as intended, processing each [callback](https://en.wikipedia.org/wiki/Callback_(computer_programming) "https://en.wikipedia.org/wiki/Callback_(computer_programming)") query correctly and giving users the relevant feedback they need. In doing these things, the bot still retains a conversational quality, which helps in maintaining the user's interest and usability. We can see that the inline buttons are more efficient in that they do not clutter the chat field, just as intended. We hope that you found the article detailed and easy to understand.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/15823.zip "Download all attachments in the single ZIP archive")

[TELEGRAM\_MQL5\_INLINE\_BUTTONS\_PART6.mq5](https://www.mql5.com/en/articles/download/15823/telegram_mql5_inline_buttons_part6.mq5 "Download TELEGRAM_MQL5_INLINE_BUTTONS_PART6.mq5")(191.11 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [MQL5 Trading Tools (Part 12): Enhancing the Correlation Matrix Dashboard with Interactivity](https://www.mql5.com/en/articles/20962)
- [Creating Custom Indicators in MQL5 (Part 5): WaveTrend Crossover Evolution Using Canvas for Fog Gradients, Signal Bubbles, and Risk Management](https://www.mql5.com/en/articles/20815)
- [MQL5 Trading Tools (Part 11): Correlation Matrix Dashboard (Pearson, Spearman, Kendall) with Heatmap and Standard Modes](https://www.mql5.com/en/articles/20945)
- [Creating Custom Indicators in MQL5 (Part 4): Smart WaveTrend Crossover with Dual Oscillators](https://www.mql5.com/en/articles/20811)
- [Building AI-Powered Trading Systems in MQL5 (Part 8): UI Polish with Animations, Timing Metrics, and Response Management Tools](https://www.mql5.com/en/articles/20722)
- [Creating Custom Indicators in MQL5 (Part 3): Multi-Gauge Enhancements with Sector and Round Styles](https://www.mql5.com/en/articles/20719)
- [Creating Custom Indicators in MQL5 (Part 2): Building a Gauge-Style RSI Display with Canvas and Needle Mechanics](https://www.mql5.com/en/articles/20632)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/472994)**
(4)


![Clemence Benjamin](https://c.mql5.com/avatar/2025/3/67df27c6-2936.png)

**[Clemence Benjamin](https://www.mql5.com/en/users/billionaire2024)**
\|
12 Sep 2024 at 23:04

**MetaQuotes:**

Check out the new article: [Creating an MQL5-Telegram Integrated Expert Advisor (Part 6): Adding Responsive Inline Buttons](https://www.mql5.com/en/articles/15823).

Author: [Allan Munene Mutiiria](https://www.mql5.com/en/users/29210372 "29210372")

This is fantastic! You consistently provide valuable insights, and I truly appreciate it. Thank you, esteemed Sir [Allan](https://www.mql5.com/en/users/29210372).

![Allan Munene Mutiiria](https://c.mql5.com/avatar/2022/11/637df59b-9551.jpg)

**[Allan Munene Mutiiria](https://www.mql5.com/en/users/29210372)**
\|
13 Sep 2024 at 02:44

**Clemence Benjamin [#](https://www.mql5.com/en/forum/472994#comment_54560242):**

This is fantastic! You consistently provide valuable insights, and I truly appreciate it. Thank you, esteemed Sir [Allan](https://www.mql5.com/en/users/29210372).

[@Clemence Benjamin](https://www.mql5.com/en/users/billionaire2024) that's so nice of you. Thank you for the kind feedback and recognition. You're most welcomed.


![Javier Santiago Gaston De Iriarte Cabrera](https://c.mql5.com/avatar/2024/11/672e255d-38a9.jpg)

**[Javier Santiago Gaston De Iriarte Cabrera](https://www.mql5.com/en/users/jsgaston)**
\|
13 Sep 2024 at 22:39

Great job! Thanks!


![Allan Munene Mutiiria](https://c.mql5.com/avatar/2022/11/637df59b-9551.jpg)

**[Allan Munene Mutiiria](https://www.mql5.com/en/users/29210372)**
\|
15 Sep 2024 at 22:38

**Javier Santiago Gaston De Iriarte Cabrera [#](https://www.mql5.com/en/forum/472994#comment_54570170):**

Great job! Thanks!

[@Javier Santiago Gaston De Iriarte Cabrera](https://www.mql5.com/en/users/jsgaston) thanks for the kind feedback and recognition. You're much welcomed.

![How to Implement Auto Optimization in MQL5 Expert Advisors](https://c.mql5.com/2/93/Implementing_Auto_Optimization_in_MQL5_Expert_Advisors__LOGO.png)[How to Implement Auto Optimization in MQL5 Expert Advisors](https://www.mql5.com/en/articles/15837)

Step by step guide for auto optimization in MQL5 for Expert Advisors. We will cover robust optimization logic, best practices for parameter selection, and how to reconstruct strategies with back-testing. Additionally, higher-level methods like walk-forward optimization will be discussed to enhance your trading approach.

![MQL5 Wizard Techniques you should know (Part 38): Bollinger Bands](https://c.mql5.com/2/93/MQL5_Wizard_Techniques_you_should_know_Part_38____LOGO__2.png)[MQL5 Wizard Techniques you should know (Part 38): Bollinger Bands](https://www.mql5.com/en/articles/15803)

Bollinger Bands are a very common Envelope Indicator used by a lot of traders to manually place and close trades. We examine this indicator by considering as many of the different possible signals it does generate, and see how they could be put to use in a wizard assembled Expert Advisor.

![Developing a multi-currency Expert Advisor (Part 10): Creating objects from a string](https://c.mql5.com/2/77/Developing_a_multi-currency_advisor_2Part_101___LOGO.png)[Developing a multi-currency Expert Advisor (Part 10): Creating objects from a string](https://www.mql5.com/en/articles/14739)

The EA development plan includes several stages with intermediate results being saved in the database. They can only be retrieved from there again as strings or numbers, not objects. So we need a way to recreate the desired objects in the EA from the strings read from the database.

![Two-sample Kolmogorov-Smirnov test as an indicator of time series non-stationarity](https://c.mql5.com/2/76/Smirnovs_homogeneity_criterion_as_an_indicator_of_non-stationarity_of_a_time_series___LOGO.png)[Two-sample Kolmogorov-Smirnov test as an indicator of time series non-stationarity](https://www.mql5.com/en/articles/14813)

The article considers one of the most famous non-parametric homogeneity tests – the two-sample Kolmogorov-Smirnov test. Both model data and real quotes are analyzed. The article also provides an example of constructing a non-stationarity indicator (iSmirnovDistance).

[![](https://www.mql5.com/ff/sh/wm94j0jmkwd29943z2/ddfa713cb3cdd580c3e81e0e13b5b1b8.jpg)\\
Revised MetaTrader 5 Web Terminal\\
\\
Trade with no restrictions from any mobile device, OS and web browser\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=fkjlpstbxdmrrwpblfatcsdjyrxbizyj&s=f462f051eb7aaec36d6b31792d312d60d3f5a50c83b12d0d66e85d5d61bd941b&uid=&ref=https://www.mql5.com/en/articles/15823&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5049247455712749545)

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