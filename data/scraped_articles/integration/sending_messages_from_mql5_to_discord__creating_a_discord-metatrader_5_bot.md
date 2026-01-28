---
title: Sending Messages from MQL5 to Discord, Creating a Discord-MetaTrader 5 Bot
url: https://www.mql5.com/en/articles/18550
categories: Integration, Expert Advisors
relevance_score: 9
scraped_at: 2026-01-22T17:40:53.329842
---

[![](https://www.mql5.com/ff/si/3fgkjn78mkxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=ocndbzpeklfncxysjbwfhhbalbrsdbtv&s=a4309643278437a00bdd33c5809fc6b4b4032749c00fccd07b3b84e7b8b45126&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=xkeozoeeplfivqrngxjhcroxnrkwbwpu&ssn=1769092851770545745&ssn_dr=0&ssn_sr=0&fv_date=1769092851&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F18550&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Sending%20Messages%20from%20MQL5%20to%20Discord%2C%20Creating%20a%20Discord-MetaTrader%205%20Bot%20-%20MQL5%20Articles&scr_res=1920x1080&ac=17690928514237593&fz_uniq=5049299257313306866&sv=2552)

MetaTrader 5 / Integration


**Contents**

- [Introduction](https://www.mql5.com/en/articles/18550#para1)
- [Creating a Discord Webhook](https://www.mql5.com/en/articles/18550#creating-discord-webhook)
- [Sending the first message from MetaTrader 5 to Discord](https://www.mql5.com/en/articles/18550#first-message-from-MT5-discord)
- [Creating a Discord class in MQL5](https://www.mql5.com/en/articles/18550#discord-class-mql5)
- [Discord bot's identity](https://www.mql5.com/en/articles/18550#discord-bot-identity)
- [Working with embeds and image files](https://www.mql5.com/en/articles/18550#embeds-and-imagefiles)
- [Sending trade notifications to Discord](https://www.mql5.com/en/articles/18550#trade-notifications-mt5-to-discord)
- [Conclusion](https://www.mql5.com/en/articles/18550#para2)

### Introduction

We are no longer living in the early stages of the internet and the primal digital era; nowadays, almost everything can be connected with anything on the internet; it only depends on one's willingness to do the work required.

The presence of API's (Application Programming Interface) which is a set of rules and protocols that allows different software applications to communicate with each other, have made it easier for more than one software or application to connect to another and hence it has made the most connected internet that we see today.

To use any API provided, you have to adhere to the rules and protocols and not to mention you must be granted a secure way _(if there is any)_ to access it by the API provider.

> ![](https://c.mql5.com/2/150/article_image__1.png)

Making a communication between MetaTrader 5 and external applications is not a new thing; it has been done before on several applications that offer reliable API(s) that MQL5 developers use to send and receive information using Web Requests. The most common app that [MQL5 traders use for such communications is Telegram](https://www.mql5.com/en/articles/2355).

In this article, we are going to discuss how we can send messages and trading information (signals) from MetaTrader 5 to Discord, using the MQL5 programming language.

### Creating a Discord Webhook

For those who aren't yet familiar with [Discord](https://www.mql5.com/go?link=https://discord.com/ "https://discord.com/"):

Discord is a free communication platform that allows users to chat via text, voice, and video, and share screens. It's popular for connecting with friends and communities, particularly within online gaming, but is also used for various other groups like book clubs or sewing circles.

Similarly to Telegram, this is a communication platform that allows users to communicate with each other. Both these platforms provide API's which allow communications even outside their platforms, but _Discord if far ahead of telegram when it comes to the integration, automation aspects._

This platform allows users to create flexible communities and manage apps (also known as bots) of different kinds.

There are various ways you can create bots and automate communications and the process of sharing information to Discord, but the simplest way is using a _Webhook_ in one of the channels from your community. Here is how you create one.

> ![](https://c.mql5.com/2/150/bandicam_2025-06-18_11-03-42-600.png)
>
> Figure 01.

From your community, go to _server settings_.

> ![](https://c.mql5.com/2/150/community_settings.gif)
>
> Figure 02.

From **server settings** go to _Integrations > Webhooks._

> ![](https://c.mql5.com/2/150/webhooks.gif)
>
> Figure 03.

Click the button _New Webhook_ to create a Webhook. _A Webhook with the default name Captain Hook will be created_. Feel free to modify this name to whatever name you want.

> ![](https://c.mql5.com/2/150/new_webhook.gif)
>
> Figure 04.

After modifying its name, you can select the channel where you want to apply this webhook to. In this server, I have a channel named trading signals, which is where I want to apply this Webhook to. _See figure 01._

Finally, we have to copy _the Webhook URL_. This _is our API gateway_ that we can _use with [Web Requests in MQL5](https://www.mql5.com/en/docs/network/webrequest)_ **.**

> ![](https://c.mql5.com/2/150/copy_webhook_url.gif)
>
> Figure 05.

### Sending the First Message from MetaTrader 5 to Discord

The first thing we have to do is add [https://discord.com](https://www.mql5.com/go?link=https://discord.com/ "https://discord.com/") to the list of allowed URLs in MetaTrader 5 otherwise, everything we are going to discuss afterward won't work.

In MetaTrader 5 go to _Tools > Options>Expert Advisors > Ensure Allow WebRequest for listed URL is checked_ **,** then proceed to _add a URL on the form below._

> ![](https://c.mql5.com/2/150/allowed_urls_mt5.gif)

Using the Webhook URL, we can send a POST request to this API endpoint given a JSON-formatted data.

Filename: Discord EA.mq5

```
#include <jason.mqh> //https://www.mql5.com/en/code/13663

string discord_webhook_url = "https://discord.com/api/webhooks/1384809399767269527/105Kp27yKnQDpKD01VdEb01GS5P-KH5o5rYKuJb_xD_D8O23GPkGLXGn9pHBB1aOt4wR";
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---

   CJAVal js(NULL, jtUNDEF); //Creating a json object

   string raw_json = "{\"content\": \"Hello world from MetaTrader5!\"}";

   bool b = js.Deserialize(raw_json); //Parse JSON string into a structured object

   string json;
   js.Serialize(json); //Convert the object to a valid JSON string

//--- Sending a Post webrequest

   char data[];
   ArrayResize(data, StringToCharArray(json, data, 0, StringLen(json))); //--- serialize to string

//--- send data

   char res_data[]; //For storing the body of the response
   string res_headers=NULL; //For storing response headers (if needed)

   string headers="Content-Type: application/json; charset=utf-8";
   uint timeout=10000; //10 seconds timeout for the HTTP request

   int ret = WebRequest("POST",
                         discord_webhook_url,
                         headers,
                         timeout,
                         data,
                         res_data,
                         res_headers); //Send a post request

   if (ret==-1)
     {
       printf("func=%s line=%d, Failed to send a webrequest. Error = %s",__FUNCTION__,__LINE__,ErrorDescription(GetLastError()));
       return false;
     }

//--- Check if the post request was successful or not

   if (ret==204)
     {
       if (MQLInfoInteger(MQL_DEBUG))
         Print("Message sent to discord successfully");
     }
   else
     {
       printf("Failed to send message to discord. Json response Error = %s",CharArrayToString(res_data));
     }

//---
   return(INIT_SUCCEEDED);
  }
```

All information and messages that you want to send to Discord must always be in a JSON format.

_Anything under the **content** key in a JSON object represents the **main text body of the message sent to Discord**._

Running the above code sends a simple message to Discord.

> ![](https://c.mql5.com/2/150/bandicam_2025-06-18_12-07-10-864.png)

Awesome! However, that was a tiresome process to handle the JSON data and everything else just to send a simple message to Discord, let's put everything in a class to make it easier to send messages without worrying about the technicalities every time.

### Creating a Discord Class in MQL5

To ensure we have a standard class of functionality, let's add logging to it for tracking the errors produced by the API endpoint and the whole sending requests process.

Filename: Discord.mqh.

```
#include <errordescription.mqh>
#include <logging.mqh>
#include <jason.mqh>

class CDiscord
  {
protected:
   string m_webhook_url;
   string m_headers;
   uint m_timeout;
   CLogging logging;

   bool PostRequest(const string json);
   string GetFormattedJson(const string raw_json);

public:
                     CDiscord(const string webhook_url, const string headers="Content-Type: application/json; charset=utf-8", const uint timeout=10000);
                    ~CDiscord(void);

                     bool SendMessage(const string message);
                     bool SendEmbeds(const string title, const string description_, color clr, const string footer_text, const datetime time);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CDiscord::CDiscord(const string webhook_url, string headers="Content-Type: application/json; charset=utf-8", uint timeout=10000):
 m_webhook_url(webhook_url),
 m_headers(headers),
 m_timeout(timeout)
 {
//--- Initialize the logger

   logging.Config("Discord server");
   if (!logging.init())
      return;

   logging.info("Initialized",MQLInfoString(MQL_PROGRAM_NAME),__LINE__);
 }
```

We wrap the repetitive processes for sending POST requests and for converting texts to JSON format.

```
string CDiscord::GetFormattedJson(const string raw_json)
 {
   CJAVal js(NULL, jtUNDEF);
   bool b = js.Deserialize(raw_json);

   string json;
   js.Serialize(json);

   return json;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CDiscord::PostRequest(const string json)
 {
   char res[];

//--- serialize to string

   char data[];
   ArrayResize(data, StringToCharArray(json, data, 0, StringLen(json)));

//--- send data

   char res_data[];
   string res_headers=NULL;

   int ret = WebRequest("POST", m_webhook_url, m_headers, m_timeout, data, res_data, res_headers); //Send a post request

   if (ret==-1)
     {
       printf("func=%s line=%d, Failed to send a webrequest. Error = %s",__FUNCTION__,__LINE__,ErrorDescription(GetLastError()));
       return false;
     }

//--- Check if the post request was successful or not

   if (ret==204)
     {
       if (MQLInfoInteger(MQL_DEBUG))
         Print("Message sent to discord successfully");
         logging.info("Message sent to discord successfully",MQLInfoString(MQL_PROGRAM_NAME), __LINE__);
     }
   else
     {
       printf("Failed to send message to discord. Json response Error = %s",CharArrayToString(res_data));
       logging.error(CharArrayToString(res_data), MQLInfoString(MQL_PROGRAM_NAME), __LINE__);
     }

  return true;
 }
```

Below is a simple function for sending messages to Discord.

```
bool CDiscord::SendMessage(const string message)
 {
   string raw_json = StringFormat("{\"content\": \"%s\"}",message);
   string json = GetFormattedJson(raw_json); //Deserialize & Serialize the message in JSON format

   return PostRequest(json);
 }
```

Here is how to use it.

Filename: Discord EA.mq5

```
#include <Discord.mqh>
CDiscord *discord;

string discord_webhook_url = "https://discord.com/api/webhooks/1384809399767269527/105Kp27yKnQDpKD01VdEb01GS5P-KH5o5rYKuJb_xD_D8O23GPkGLXGn9pHBB1aOt4wR";
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---

    discord = new CDiscord(discord_webhook_url);
    discord.SendMessage("Hello from MetaTrader5!");
  }
```

This simple and powerful function and can be used to send messages of different kinds, since [Discord uses a Markdown formatted text](https://www.mql5.com/go?link=https://support.discord.com/hc/en-us/articles/210298617-Markdown-Text-101-Chat-Formatting-Bold-Italic-Underline "https://support.discord.com/hc/en-us/articles/210298617-Markdown-Text-101-Chat-Formatting-Bold-Italic-Underline").

**Example Markdown message**

```
    discord.SendMessage(
                         "# h1 header\n"
                         "## h2 header\n"
                         "### h3 header\n\n"
                         "**bold text** normal text\n"
                         "[MQL5](https://www.mql5.com)\n"
                         "![image](https://i.imgur.com/4M34hi2.png)"
                       );
```

Outputs.

> ![](https://c.mql5.com/2/150/6303844690924.png)

And that's not even the coolest thing you can do with the Markdown text editor provided by Discord. Let's share some lines of code with Discord.

````
    discord.SendMessage("```MQL5\n" //we put the type of language for markdown highlighters after three consecutive backticks
                        "if (CheckPointer(discord)!=POINTER_INVALID)\n"
                        "  delete discord;\n"
                        "```");

    discord.SendMessage("```Python\n"
                        "while(True):\n"
                        "  break\n"
                        "```");
````

Outputs.

> ![](https://c.mql5.com/2/150/code_in_markdown.gif)

_This was to show you how powerful Markdown text editors are._

**Adding emoji's to your messages**

Emojis are quite useful in text messages; they add elegance, improve readability, not to mention, they inject a bit of humour.

Unlike other communication platforms with emojis that are crafted from specific numeric codes, Discord uses a unique syntax to identify and render emojis directly in raw text messages.

All you have to do is to put the name of the emoji between two colons — one colon at the beginning and the other at the end of the emoji's name.

> ![](https://c.mql5.com/2/150/emojis.gif)

When these emojis' codes are inserted into a text message, they will be rendered as emojis in Discord.

```
discord.SendMessage(":rotating_light: Trade Alert!");
```

Message.

> ![](https://c.mql5.com/2/150/4250031103335.png)

There are thousands of emojis, and it is challenging to keep track of them all. Here is the cheat sheet — [https://github.com/ikatyang/emoji-cheat-sheet/blob/master/README.md](https://www.mql5.com/go?link=https://github.com/ikatyang/emoji-cheat-sheet/blob/master/README.md "https://github.com/ikatyang/emoji-cheat-sheet/blob/master/README.md"). I have created a simple library file with some of the emoji's codes and syntax, _please don't hesitate to add more emoji's codes as it pleases you_.

Filename: discord emojis.mqh

```
#define DISCORD_EMOJI_ROCKET                   ":rocket:"
#define DISCORD_EMOJI_CHART_UP                 ":chart_with_upwards_trend:"
#define DISCORD_EMOJI_CHART_DOWN               ":chart_with_downwards_trend:"
#define DISCORD_EMOJI_BAR_CHART                ":bar_chart:"
#define DISCORD_EMOJI_BOMB                     ":bomb:"
#define DISCORD_EMOJI_THUMBS_UP                ":thumbsup:"
#define DISCORD_EMOJI_THUMBS_DOWN              ":thumbsdown:"
#define DISCORD_EMOJI_WARNING                  ":warning:"
#define DISCORD_EMOJI_JOY                      ":joy:"
#define DISCORD_EMOJI_SOB                      ":sob:"
#define DISCORD_EMOJI_SMILE                    ":smile:"
#define DISCORD_EMOJI_FIRE                     ":fire:"
#define DISCORD_EMOJI_STAR                     ":star:"
#define DISCORD_EMOJI_BLUSH                    ":blush:"
#define DISCORD_EMOJI_THINKING                 ":thinking:"
#define DISCORD_EMOJI_ROTATING_LIGHT           ":rotating_light:"
#define DISCORD_EMOJI_X                        ":x:"
#define DISCORD_EMOJI_WHITE_CHECK_MARK         ":white_check_mark:"
#define DISCORD_EMOJI_BALLOT_BOX_WITH_CHECK    ":ballot_box_with_check:"

#define DISCORD_EMOJI_HASH                     ":hash:"
#define DISCORD_EMOJI_ASTERISK                 ":asterisk:"

#define DISCORD_EMOJI_ZERO                     ":zero:"
#define DISCORD_EMOJI_ONE                      ":one:"
#define DISCORD_EMOJI_TWO                      ":two:"
#define DISCORD_EMOJI_THREE                    ":three:"
#define DISCORD_EMOJI_FOUR                     ":four:"
#define DISCORD_EMOJI_FIVE                     ":five:"
#define DISCORD_EMOJI_SIX                      ":six:"
#define DISCORD_EMOJI_SEVEN                    ":seven:"
#define DISCORD_EMOJI_EIGHT                    ":eight:"
#define DISCORD_EMOJI_NINE                     ":nine:"
#define DISCORD_EMOJI_TEN                      ":keycap_ten:"

#define DISCORD_EMOJI_RED_CIRCLE               ":red_circle:"
#define DISCORD_EMOJI_GREEN_CIRCLE             ":green_circle:"
```

Sending a message.

```
discord.SendMessage(DISCORD_EMOJI_ROTATING_LIGHT " Trade Alert! " DISCORD_EMOJI_JOY);
```

Message outcome.

> ![](https://c.mql5.com/2/150/4261772875123.png)

**Mentioning users & roles**

This simple message can handle mentions and roles, which are crucial for effective message delivery.

- @everyone — This notifies all users in the channel even if they are offline
- @here — This notifies all non-idle users in the chat. _All users who are online._
- <@user\_id> — This notifies a specific user or users.
- <@&role\_id> **—** This notifies all users assigned to a specific role. For example, sending trading signals to all users who have been assigned the role of manual traders.

Example usage.

```
discord.SendMessage("@everyone This is an information about a trading account");
discord.SendMessage("@here This is a quick trading signals for all of you that are active");
```

A message outcome.

> ![](https://c.mql5.com/2/150/5874195879177.png)

### Discord Bot's Identity

The good thing about a webhook is that it is not restricted to the default appearance (name and avatar); you can change these values anytime you send a post request.

This ability is quite handy as you can have multiple trading robots that send some kind of information to a community or two, all sharing one webhook, having the ability to use a different identity helps to distinguish the senders and received messages.

We can make this identity a necessity every time the discord class is called (initiated) to make the identity global.

```
class CDiscord
  {
protected:

//...
//...

public:
                     string m_name;
                     string m_avatar_url;

                     CDiscord(const string webhook_url,
                              const string name,
                              const string avatar_url,
                              const string headers="Content-Type: application/json; charset=utf-8",
                              const uint timeout=10000);
 }
```

We declare these variables publicly as a user might want to modify or access them along the way (after class initialization).

This makes it even easier to track the logs for each username that gets assigned to this Discord messenger.

```
CDiscord::CDiscord(const string webhook_url,
                   const string name,
                   const string avatar_url,
                   const string headers="Content-Type: application/json; charset=utf-8",
                   const uint timeout=10000):

 m_webhook_url(webhook_url),
 m_headers(headers),
 m_timeout(timeout),
 m_name(name),
 m_avatar_url(avatar_url)
 {
//--- Initialize the logger

   logging.Config(m_name);
   if (!logging.init())
      return;

   logging.info("Initialized",MQLInfoString(MQL_PROGRAM_NAME),__LINE__);
 }
```

Now, we have to append this identity (name and avatar) information everywhere where a JSON string is found.

```
bool CDiscord::SendMessage(const string message)
 {
   string raw_json = StringFormat("{"
                                    "\"username\": \"%s\","
                                    "\"avatar_url\": \"%s\","
                                    "\"content\": \"%s\""
                                  "}",
                                  m_name, m_avatar_url, message);

   string json = GetFormattedJson(raw_json); //Deserialize & Serialize the message in JSON format

   return PostRequest(json);
 }
```

Inside the SendJSON() function, we need to make this identity optional because a user might have a different idea in the first place, that's why they chose to go with a function that takes a raw JSON string that gives them control over what information they want to send.

```
bool CDiscord::SendJSON(const string raw_json, bool use_id=true)
 {
   CJAVal js;

   js.Deserialize(raw_json);

   if (use_id) //if a decides to use the ids assigned to the class constructor, we append that information to the json object
     {
       js["username"] = m_name;
       js["avatar_url"] = m_avatar_url;
     }

   string json;
   js.Serialize(json);

   return PostRequest(json);
 }
```

Let us set a different identity for our bot.

```
#include <Discord.mqh>
CDiscord *discord;

input string discord_webhook_url = "https://discord.com/api/webhooks/1384809399767269527/105Kp27yKnQDpKD01VdEb01GS5P-KH5o5rYKuJb_xD_D8O23GPkGLXGn9pHBB1aOt4wR";
input string avatar_url_ = "https://imgur.com/m7sVf51.jpeg";
input string bots_name = "Signals Bot";
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---

    discord = new CDiscord(discord_webhook_url,
                           bots_name,
                           avatar_url_);

    discord.SendMessage("Same webhook but, different Identity :smile: cheers!");

//---
   return(INIT_SUCCEEDED);
  }
```

Outcome.

> ![](https://c.mql5.com/2/150/3192566328259.png)

### Working with Embeds and Image files

Despite having attached the image to the message discussed in the previous section when creating a Discord class, there is a proper way to attach files and embed items to a message.

We can accomplish this with an independent JSON request.

```
discord.SendJSON(
    "{"
        "\"content\": \"Here's the latest chart update:\","
        "\"embeds\": ["\
            "{"\
                "\"title\": \"Chart Update\","\
                "\"image\": {"\
                    "\"url\": \"https://imgur.com/SBsomI7.png\""\
                "}"\
            "}"\
        "]"
    "}"
);
```

Results.

> ![](https://c.mql5.com/2/150/chart_embed.gif)

Unfortunately, you cannot send an image directly from MQL5 using Web Request _, despite Discord being capable of receiving the files._  Currently, the best you can do is to share an image from some link where the file is hosted online.

I had to use [imgur.com](https://www.mql5.com/go?link=https://imgur.com/ "https://imgur.com/") in the previous example.

### Sending Trade Notifications from MetaTrader 5 to Discord

Traders often use the ability to send information from MetaTrader 5 to external platforms to send updates about their trading activities. Let's use this bot (Expert Advisor) for such a task.

Starting with notifications for opening a trade.

**01: Sending trades opening notifications**

The [OnTradeTransaction](https://www.mql5.com/en/docs/event_handlers/ontradetransaction) function is handy for this task.

Below are condition checkers for checking the state of a recent position and deal.

```
#define IS_TRANSACTION_POSITION_OPENED         (trans.type == TRADE_TRANSACTION_DEAL_ADD && HistoryDealSelect(trans.deal) && (ENUM_DEAL_ENTRY)HistoryDealGetInteger(trans.deal, DEAL_ENTRY) == DEAL_ENTRY_IN)
#define IS_TRANSACTION_POSITION_CLOSED         (trans.type == TRADE_TRANSACTION_DEAL_ADD && HistoryDealSelect(trans.deal) && (ENUM_DEAL_ENTRY)HistoryDealGetInteger(trans.deal, DEAL_ENTRY) == DEAL_ENTRY_OUT && ((ENUM_DEAL_REASON)HistoryDealGetInteger(trans.deal, DEAL_REASON) != DEAL_REASON_SL && (ENUM_DEAL_REASON)HistoryDealGetInteger(trans.deal, DEAL_REASON) != DEAL_REASON_TP))
#define IS_TRANSACTION_POSITION_MODIFIED       (trans.type == TRADE_TRANSACTION_POSITION)
```

```
void OnTradeTransaction(const MqlTradeTransaction& trans,
                        const MqlTradeRequest& request,
                        const MqlTradeResult& result)
  {
     ulong deal_ticket = trans.deal;
     ulong position_ticket = trans.position;

//---

     m_deal.Ticket(deal_ticket); //select a deal by it's ticket

     if (IS_TRANSACTION_POSITION_OPENED)
       {
         if (deal_ticket==0)
           return;

         if (!m_position.SelectByTicket(position_ticket)) //select a position by ticket
            return;

         if (!m_symbol.Name(m_deal.Symbol())) //select a symbol from a position
           {
             printf("line=%d Failed to select symbol %s. Error = %s",__LINE__,m_deal.Symbol(),ErrorDescription(GetLastError()));
             return;
           }

         string message =
            DISCORD_EMOJI_ROTATING_LIGHT " **TRADE OPENED ALERT** \n\n"
            "- **Symbol:**"+m_position.Symbol()+"\n"
            "- **Trade Type:** "+ (m_position.PositionType()==POSITION_TYPE_BUY ? DISCORD_EMOJI_GREEN_CIRCLE: DISCORD_EMOJI_RED_CIRCLE) +" "+m_position.TypeDescription()+"\n"
            "- **Entry Price:** `"+DoubleToString(m_deal.Price(), m_symbol.Digits())+"`\n"
            "- **Stop Loss:** `"+DoubleToString(m_position.StopLoss(), m_symbol.Digits())+"`\n"
            "- **Take Profit:** `"+DoubleToString(m_position.TakeProfit(), m_symbol.Digits())+"`\n"
            "- **Time UTC:** `"+TimeToString(TimeGMT())+"`\n"
            "- **Position Ticket:** `"+(string)position_ticket+"`\n"
            "> "DISCORD_EMOJI_WARNING" *Risk what you can afford to lose.*";

         discord.SendMessage(message);
       }

  //...
  //... Other lines of code
  //...
 }
```

Note:

When formatting market values like entry price, stop loss, and take profit, etc., I wrap each one with a single backtick (\`). This creates an inline code block in Discord, which helps visually separate the numbers and makes them easier to copy for users.

Discord uses.

- A single backtick (\`) to format inline code — great for short values like prices.
- Triple backticks (\`\`\`) to format multi-line code blocks — used for larger blocks of code or messages.

I opened two opposite trades manually using **One-Click Trading** in MetaTrader 5, below is the outcome.

> ![](https://c.mql5.com/2/150/5488022703974.png)

**02: Sending trades modification notifications**

```
if (IS_TRANSACTION_POSITION_MODIFIED)
  {
    if (!m_position.SelectByTicket(position_ticket))
      {
        printf("Failed to modify a position. Erorr = %s",ErrorDescription(GetLastError()));
        return;
      }

    if (!m_symbol.Name(m_deal.Symbol()))
      {
        printf("line=%d Failed to select symbol %s. Error = %s",__LINE__,m_deal.Symbol(),ErrorDescription(GetLastError()));
        return;
      }

    string message =
       DISCORD_EMOJI_BAR_CHART " **TRADE MODIFIED ALERT** \n\n"
       "- **Symbol:** `"+m_position.Symbol()+"`\n"
       "- **Trade Type:** "+ (m_position.PositionType()==POSITION_TYPE_BUY ? DISCORD_EMOJI_GREEN_CIRCLE: DISCORD_EMOJI_RED_CIRCLE) +" "+m_position.TypeDescription()+"\n"
       "- **Entry Price:** `"+DoubleToString(m_position.PriceOpen(), m_symbol.Digits())+"`\n"
       "- **New Stop Loss:** `"+DoubleToString(m_position.StopLoss(), m_symbol.Digits())+"`\n"
       "- **New Take Profit:** `"+DoubleToString(m_position.TakeProfit(), m_symbol.Digits())+"`\n"
       "- **Time UTC:** `"+TimeToString(TimeGMT())+"`\n"
       "- **Position Ticket:** `"+(string)position_ticket+"`\n"
       "> "DISCORD_EMOJI_WARNING" *Risk what you can afford to lose.*";

    discord.SendMessage(message);
  }
```

_Everything is the same in this message as it was for sending trade notifications, only stop loss and take profit field names were changed._

> Video from the article "Sending Messages from MQL5 to Discord, Creating a Discord-MetaTrader5 Bot" - YouTube
>
> [Photo image of MQL5.community](https://www.youtube.com/channel/UC8bwZMk2yh5WDToFos49OjQ?embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F18550)
>
> MQL5.community
>
> 1.91K subscribers
>
> [Video from the article "Sending Messages from MQL5 to Discord, Creating a Discord-MetaTrader5 Bot"](https://www.youtube.com/watch?v=VJm2mfWr9P0)
>
> MQL5.community
>
> Search
>
> Watch later
>
> Share
>
> Copy link
>
> Info
>
> Shopping
>
> Tap to unmute
>
> If playback doesn't begin shortly, try restarting your device.
>
> More videos
>
> ## More videos
>
> You're signed out
>
> Videos you watch may be added to the TV's watch history and influence TV recommendations. To avoid this, cancel and sign in to YouTube on your computer.
>
> CancelConfirm
>
> Share
>
> Include playlist
>
> An error occurred while retrieving sharing information. Please try again later.
>
> [Watch on](https://www.youtube.com/watch?v=VJm2mfWr9P0&embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F18550)
>
> 0:00
>
> 0:00 / 1:07
>
> •Live
>
> •

**03: Sending closing trades notifications**

```
if (IS_TRANSACTION_POSITION_CLOSED)
   {
    if (!m_symbol.Name(m_deal.Symbol()))
      {
        printf("line=%d Failed to select symbol %s. Error = %s",__LINE__,m_deal.Symbol(),ErrorDescription(GetLastError()));
        return;
      }

    m_symbol.RefreshRates(); //Get recent ask and bid prices

    long reason_integer;
    m_deal.InfoInteger(DEAL_REASON, reason_integer);

    string reason_text = GetDealReasonText(reason_integer);

    string message =
          DISCORD_EMOJI_X " **TRADE CLOSED ALERT**\n\n"
          "- **Symbol:** `" + m_deal.Symbol() + "`\n"
          "- **Trade Type:** " + (m_position.PositionType() == POSITION_TYPE_BUY ? DISCORD_EMOJI_GREEN_CIRCLE : DISCORD_EMOJI_RED_CIRCLE) + " " + m_position.TypeDescription() + "\n"
          "- **Entry Price:** `" + DoubleToString(m_deal.Price(), m_symbol.Digits()) + "`\n"
          "- **Exit Price:** `" + (m_position.PositionType() == POSITION_TYPE_BUY ? (DoubleToString(m_symbol.Bid(), m_symbol.Digits())) : (DoubleToString(m_symbol.Ask(), m_symbol.Digits()))) + "`\n"
          "- **Profit:** " + (m_deal.Profit() >= 0 ? DISCORD_EMOJI_THUMBS_UP : DISCORD_EMOJI_THUMBS_DOWN) + " `" + DoubleToString(m_deal.Profit(), 2) + "`\n"
          "- **Close Reason:** `" + reason_text+ "`\n"
          "- **Commission:** `" + DoubleToString(m_position.Commission(), 2) + "`\n"
          "- **Swap:** `" + DoubleToString(m_position.Swap(), 2)+ "`\n"
          "- **Time (UTC):** `" + TimeToString(TimeGMT()) + "`\n"
          "- **Deal Ticket:** `" + string(deal_ticket) + "`\n"
          "> "DISCORD_EMOJI_WARNING" *Risk what you can afford to lose.*";

    discord.SendMessage(message);
  }
```

On a closing trade message, we put a thumbs-up emoji when a trade was closed in profit, and a thumbs-down emoji when it was closed in a loss. Unlike opening and modifying the position, where we had a position ticket number. A closed position is no longer a position — It is a deal so, we use the deal ticket number for it.

> ![](https://c.mql5.com/2/150/2055515316991.png)

This deal was closed manually so its reason becomes _Client (Desktop Terminal)_, this is according to [ENUM\_DEAL\_REASON](https://www.mql5.com/en/docs/constants/tradingconstants/dealproperties#:~:text=as%20the%20reason.-,%20ENUM_DEAL_REASON,-Identifier).

```
string GetDealReasonText(long reason)
{
   switch((int)reason)
   {
      case DEAL_REASON_CLIENT:            return "Client (Desktop Terminal)";
      case DEAL_REASON_MOBILE:            return "Mobile App";
      case DEAL_REASON_WEB:               return "Web Platform";
      case DEAL_REASON_EXPERT:            return "Expert Advisor";
      case DEAL_REASON_SL:                return "Stop Loss Hit";
      case DEAL_REASON_TP:                return "Take Profit Hit";
      case DEAL_REASON_SO:                return "Stop Out (Margin)";
      case DEAL_REASON_ROLLOVER:          return "Rollover Execution";
      case DEAL_REASON_VMARGIN:           return "Variation Margin Charged";
      case DEAL_REASON_SPLIT:             return "Stock Split Adjustment";
      case DEAL_REASON_CORPORATE_ACTION:  return "Corporate Action";
      default:                            return "Unknown Reason";
   }
}
```

### The Bottom LIne

Discord webhooks are a good starting point for MetaTrader 5 and Discord integration, enabling communications between these two powerful platforms. However, as I said earlier, all API's have rules and limitations — Discord Webhook API is not different because Webhook requests are rate-limited.

- You can only send 5 messages every 2 seconds using a single Webhook
- A maximum of 30 messages per minute per channel is allowed for a single Webhook.

Exceeding this limitation, Discord will return HTTP 429 (Too Many Requests). To tackle this, you can add delays or queues in your MQL5 code.

Also, throughout the article, I talk about Discord bot(s), but a Webhook is very different from a Discord bot. _The entire workflow between MQL5 and Discord is what I refer to as a bot._

Bots are essentially programs that run within Discord, offering a wide range of functionalities, from basic commands to complex automations. Webhooks, on the other hand, are simple URLs that allow external applications to send automated messages to a specific Discord channel

Below is the tabulated difference between a Discord Webhook and a Discord bot.

|  | Webhooks | Bots |
| --- | --- | --- |
| Function | - Can only send messages to a set channel<br>- They can only send messages, not view any.<br>- Can send up to 10 embeds per message. | - Much more flexible as _they can do more complex actions similar to what a regular user can do._<br>- Bots can view and send messages.<br>- Only one embed per message is allowed. |
| Customization | - Can create 10 webhooks per server  with the ability to customize each avatar and name.<br>- Able to hyperlink any text outside of an embed | - Public bots often have a preset avatar and name, which cannot be modified by end users.<br>- Cannot hyperlink any text in a normal message, must use an embed |
| Load and security | - Just an endpoint to send data to, no actual hosting is required.<br>- No authentication that the data sent to the webhook is from a trusted source.<br>- No authentication that the data sent to the webhook is from a trusted source. If the webhook URL is leaked, only non-permanent problems may occur (e.g., spamming)<br>- Easy to change webhook URL if needed. | - Bots have to be hosted in a secure environment that will need to be kept online all the time, which costs more resources.<br>- Bots are authenticated via a token; a compromised token can cause severe damage due to their capabilities if they have permissions granted to them by the server owner.<br>- However, you can reset the bot token if needed. |

A Discord bot is quite complex and not very practical to MQL5 programmers, as we often want a functionality to alert our fellow traders about the trading progress only. Not to mention, to create a Discord bot, you need a couple of [Python modules](https://www.mql5.com/go?link=https://discord.com/developers/docs/intro "https://discord.com/developers/docs/intro").

However, you can develop this fully fledged Discord bot that works with MetaTrader 5 if you'd like to.

Since we have a MetaTrader 5-Python package, from the Python environment, you can work your way out to a full integration between these two platforms.

_Best regards._

**Attachments Table**

| Filename | Description & Usage |
| --- | --- |
| Include\\discord emojis.mqh | Contain emoji codes and syntax compatible with Discord. |
| Include\\discord.mqh | It has the CDiscord class for sending messages in JSON format to the Discord platform. |
| Include\\errordescription.mqh | Contains descriptions of all error codes produced by MetaTrader 5 in MQL5 language. |
| [Include\\jason.mqh](https://www.mql5.com/en/code/13663) | A library for serialization and deserialization of JSON protocol. |
| Include\\logging.mqh | A library for logging all the information and errors produced by the CDiscord class (Webhook sender). |
| Experts\\Discord EA.mq5 | An Expert Advisor (EA) for testing the Discord webhook and for sending trade alerts to the assigned Discord webhook URL. |

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/18550.zip "Download all attachments in the single ZIP archive")

[Attachments.zip](https://www.mql5.com/en/articles/download/18550/attachments.zip "Download Attachments.zip")(19.26 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Python-MetaTrader 5 Strategy Tester (Part 03): MT5-Like Trading Operations — Handling and Managing](https://www.mql5.com/en/articles/20782)
- [Python-MetaTrader 5 Strategy Tester (Part 02): Dealing with Bars, Ticks, and Overloading Built-in Functions in a Simulator](https://www.mql5.com/en/articles/20455)
- [Implementing Practical Modules from Other Languages in MQL5 (Part 06): Python-Like File IO operations in MQL5](https://www.mql5.com/en/articles/20695)
- [Data Science and ML (Part 47): Forecasting the Market Using the DeepAR model in Python](https://www.mql5.com/en/articles/20571)
- [Implementing Practical Modules from Other Languages in MQL5 (Part 05): The Logging module from Python, Log Like a Pro](https://www.mql5.com/en/articles/20458)
- [Implementing Practical Modules from Other Languages in MQL5 (Part 04): time, date, and datetime modules from Python](https://www.mql5.com/en/articles/19035)
- [Python-MetaTrader 5 Strategy Tester (Part 01): Trade Simulator](https://www.mql5.com/en/articles/18971)

**[Go to discussion](https://www.mql5.com/en/forum/489798)**

![MetaTrader 5 Machine Learning Blueprint (Part 1): Data Leakage and Timestamp Fixes](https://c.mql5.com/2/164/17520-metatrader-5-machine-learning-logo.png)[MetaTrader 5 Machine Learning Blueprint (Part 1): Data Leakage and Timestamp Fixes](https://www.mql5.com/en/articles/17520)

Before we can even begin to make use of ML in our trading on MetaTrader 5, it’s crucial to address one of the most overlooked pitfalls—data leakage. This article unpacks how data leakage, particularly the MetaTrader 5 timestamp trap, can distort our model's performance and lead to unreliable trading signals. By diving into the mechanics of this issue and presenting strategies to prevent it, we pave the way for building robust machine learning models that deliver trustworthy predictions in live trading environments.

![Developing a Replay System (Part 73): An Unusual Communication (II)](https://c.mql5.com/2/100/Desenvolvendo_um_sistema_de_Replay_Parte_73_Uma_comunicaimo_inusitada_II___LOGO.png)[Developing a Replay System (Part 73): An Unusual Communication (II)](https://www.mql5.com/en/articles/12363)

In this article, we will look at how to transmit information in real time between the indicator and the service, and also understand why problems may arise when changing the timeframe and how to solve them. As a bonus, you will get access to the latest version of the replay /simulation app.

![Fast trading strategy tester in Python using Numba](https://c.mql5.com/2/101/Fast_Trading_Strategy_Tester_in_Python_Using_Numba__LOGO.png)[Fast trading strategy tester in Python using Numba](https://www.mql5.com/en/articles/14895)

The article implements a fast strategy tester for machine learning models using Numba. It is 50 times faster than the pure Python strategy tester. The author recommends using this library to speed up mathematical calculations, especially the ones involving loops.

![Developing Advanced ICT Trading Systems: Implementing Order Blocks in an Indicator](https://c.mql5.com/2/99/Desarrollo_de_Sistemas_Avanzados_de_Trading_ICT___LOGO.png)[Developing Advanced ICT Trading Systems: Implementing Order Blocks in an Indicator](https://www.mql5.com/en/articles/15899)

In this article, we will learn how to create an indicator that detects, draws, and alerts on the mitigation of order blocks. We will also take a detailed look at how to identify these blocks on the chart, set accurate alerts, and visualize their position using rectangles to better understand the price action. This indicator will serve as a key tool for traders who follow the Smart Money Concepts and the Inner Circle Trader methodology.

[![](https://www.mql5.com/ff/si/0nfwvn6yhmgzf97n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F117%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dorder.expert%26utm_content%3Dorder.freelance%26utm_campaign%3D0622.MQL5.com.Internal&a=tunpwtbhegzufrqocbwiszessdutnobs&s=d9e7484e15300021b4066b1df77a94a1352f9e7c326d5113006bb4f6476bafeb&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=voidttzzlpmvwvxckyzyibpyyulpfcrv&ssn=1769092851770545745&ssn_dr=0&ssn_sr=0&fv_date=1769092851&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F18550&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Sending%20Messages%20from%20MQL5%20to%20Discord%2C%20Creating%20a%20Discord-MetaTrader%205%20Bot%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176909285142343866&fz_uniq=5049299257313306866&sv=2552)

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