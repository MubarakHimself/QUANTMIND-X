---
title: Integrating Discord with MetaTrader 5: Building a Trading Bot with Real-Time Notifications
url: https://www.mql5.com/en/articles/16682
categories: Integration
relevance_score: 6
scraped_at: 2026-01-23T17:19:39.187346
---

[![](https://www.mql5.com/ff/sh/wm94j0jmkwd29943z2/ddfa713cb3cdd580c3e81e0e13b5b1b8.jpg)\\
Revised MetaTrader 5 Web Terminal\\
\\
Trade with no restrictions from any mobile device, OS and web browser\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=fkjlpstbxdmrrwpblfatcsdjyrxbizyj&s=f462f051eb7aaec36d6b31792d312d60d3f5a50c83b12d0d66e85d5d61bd941b&uid=&ref=https://www.mql5.com/en/articles/16682&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5068103371459589496)

MetaTrader 5 / Integration


### Introduction

The capacity to remotely monitor and control trading activity has become more important in today's fast-paced trading market. One of the possible solutions to implementing such a remote monitoring features is integrating Discord notifications with MetaTrader 5. By receiving notifications to your Discord app, you can monitor your activities in real time from any location. In this article, we will use a real-world example of a random trading bot to illustrate ideas and implementation steps. So, we will establish a reliable communication between MetaTrader 5 and the Discord platform, through which you can get real-time notifications for trade executions, market changes, and other alerts.

In this article, we will see which settings you need to make on the platform side to enable this kind of integration. In particular, we will see WebRequest settings, which allow you to connect the platform with other resources, such as Discord and other instant messengers. We will also see how to configure your Discord server in order to be able to receive notifications from MetaTrader 5.

The material covered in this article requires some prior knowledge of MQL5 programming and a good knowledge of how the platform operates.

### Discord and MetaTrader 5 configuration

You need to add these two https and verify that the specified URL permits Web requests:

![Allow WebRequests](https://c.mql5.com/2/106/allow_webrequest.jpg)

Along with creating and copying a webhook, you also need to setup a server (or utilize one you currently have in Discord).

WebHook - YouTube

[Photo image of MQL5.community](https://www.youtube.com/channel/UC8bwZMk2yh5WDToFos49OjQ?embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F16682)

MQL5.community

1.91K subscribers

[WebHook](https://www.youtube.com/watch?v=jNut3L414-E)

MQL5.community

Search

Watch later

Share

Copy link

Info

Shopping

Tap to unmute

If playback doesn't begin shortly, try restarting your device.

More videos

## More videos

Share

Include playlist

An error occurred while retrieving sharing information. Please try again later.

[Watch on](https://www.youtube.com/watch?v=jNut3L414-E&embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F16682)

0:00

0:00 / 0:40

•Live

•

### EA Code example

In the Expert Advisor, don't forget to paste the WebHook link.

```
#include <Trade/Trade.mqh>

CTrade trade;
// Discord webhook URL - Replace with your webhook URL
string discord_webhook = "https://discord.com/api/webhooks/XXXXXXXXXXXXXXXXXXXXXXXXX";

// Strategy Parameters
```

\\*\\*\\* I don't know why, but the webhook needs to be changed from discordapp.com to discord.com.

Here's how we set up our MetaTrader 5 Expert Advisor's basic Discord settings:

```
string discord_webhook = "https://discord.com/api/webhooks/your-webhook-url";

input group "Discord Settings"
input string DiscordBotName = "MT5 Trading Bot";
input color MessageColor = clrBlue;
input bool SendPriceUpdates = true;
```

The most important link between Discord and your MetaTrader 5 Expert Advisor is the webhook URL. Additionally, we've included inputs that users may customize to change the bot's look and behavior. These options provide the integration flexibility, enabling traders to modify the system to suit their own requirements.

A crucial step in guaranteeing dependable connection between MetaTrader 5 and Discord is the activation procedure. In this stage, we must confirm that the Discord webhook is operating properly and that the Expert Advisor has the required authorization to submit web requests. Several crucial tests are part of this verification process:

```
int OnInit() {
    Print("Initialization step 1: Checking WebRequest permissions...");

    if(!TerminalInfoInteger(TERMINAL_TRADE_ALLOWED)) {
        Print("Error: WebRequest is not allowed. Please allow in Tool -> Options -> Expert Advisors");
        return INIT_FAILED;
    }

    string test_message = "{\"content\":\"Test message from MT5\"}";
    string headers = "Content-Type: application/json\r\n";
    char data[], result[];
    ArrayResize(data, StringToCharArray(test_message, data, 0, WHOLE_ARRAY, CP_UTF8) - 1);

    int res = WebRequest(
        "POST",
        discord_webhook,
        headers,
        5000,
        data,
        result,
        headers
    );

    if(res == -1) {
        Print("Make sure these URLs are allowed: https://discord.com/*, https://discordapp.com/*");
        return INIT_FAILED;
    }

    return(INIT_SUCCEEDED);
}
```

In order to make the Discord alerts both educational and aesthetically pleasing, message formatting is essential. Because markdown formatting is supported by Discord, we may organize our messages to highlight crucial information. We've put in place a thorough method for formatting trade data that makes it simple to comprehend each deal's specifics quickly:

```
string FormatTradeMessage(TradeInfo& tradeInfo) {
    string message = "**New " + tradeInfo.type + " Signal Alert!**\n";
    message += "Symbol: " + tradeInfo.symbol + "\n";
    message += "Type: " + tradeInfo.type + "\n";
    message += "Price: " + DoubleToString(tradeInfo.price, _Digits) + "\n";
    message += "Lots: " + DoubleToString(tradeInfo.lots, 2) + "\n";
    message += "Stop Loss: " + DoubleToString(tradeInfo.sl, _Digits) + "\n";
    message += "Take Profit: " + DoubleToString(tradeInfo.tp, _Digits) + "\n";
    message += "Time: " + TimeToString(TimeCurrent());
    return message;
}
```

Correct handling of special characters is crucial when working with JSON data in MetaTrader 5 in order to avoid incorrect JSON strings that might result in unsuccessful message delivery. A strong JSON escaping method that can handle all common special characters is part of our implementation:

```
string EscapeJSON(string text) {
    string escaped = text;
    StringReplace(escaped, "\\", "\\\\");
    StringReplace(escaped, "\"", "\\\"");
    StringReplace(escaped, "\n", "\\n");
    StringReplace(escaped, "\r", "\\r");
    StringReplace(escaped, "\t", "\\t");
    return escaped;
}
```

The SendDiscordMessage function handles the reliable delivery of messages from MetaTrader 5 to Discord through Discord's webhook API. At its core, this function takes a message string and an optional error flag as parameters, transforming them into a properly formatted HTTP request that Discord's servers can understand and process.

The function begins with a safety check by verifying if web requests are enabled through the isWebRequestEnabled flag. This verification step prevents any attempted communications when the MetaTrader 5 platform lacks the necessary permissions, avoiding potential system hangs or crashes. If web requests aren't enabled, the function immediately returns false, indicating the message couldn't be sent.

When constructing the message, the function uses visual marks for improved readability. It prepends either a red X emoji (❌) for error messages or a green checkmark emoji (✅) for successful operations. These visual hints allow traders to quickly understand the nature of notifications at a glance in their Discord channel.

The message is then wrapped into a JSON payload, which is the format Discord's API expects. The EscapeJSON function (called within the payload construction) plays a vital role here by ensuring special characters in the message won't break the JSON structure. This includes handling quotation marks, newlines, and other special characters that could otherwise cause parsing errors.

The function sets up the proper HTTP headers, specifically indicating that the content type is JSON. This header information is crucial as it tells Discord's servers how to interpret the incoming data. The Content-Type header is set to "application/json", which is standard for REST API communications.

One of the more technical aspects involves the conversion of the string payload into a char array. This conversion is necessary because MetaTrader 5's WebRequest function expects binary data rather than plain strings. The ArrayResize function ensures the char array is properly sized to hold the converted message, accounting for UTF-8 encoding which is essential for handling special characters and emojis correctly.

The actual communication happens through the WebRequest function call, which sends a POST request to the Discord webhook URL. The function includes several important parameters:

- A timeout value of 5000 milliseconds (5 seconds) to prevent the system from hanging if Discord's servers are slow to respond
- The previously prepared headers and data
- Arrays to store the response data and headers

The function monitors the success of the message delivery through HTTP response codes. A response code of 200 or 204 indicates successful delivery (200 means success with content returned, 204 means success with no content). When success is detected, the function updates the lastMessageTime timestamp, which can be used for rate limiting purposes, and returns true to indicate successful delivery.

In cases where the message fails to send (indicated by any response code other than 200 or 204), the function returns false, allowing the calling code to handle the failure appropriately. This error handling mechanism enables the implementation of retry logic or alternative notification methods when Discord communication fails.

This implementation creates a robust and reliable communication channel between MetaTrader 5 and Discord, handling all the complexities of cross-platform communication while providing clear success/failure feedback to the calling code. The attention to proper error handling, character encoding, and API compliance makes this function a dependable core component of the Discord integration system.

```
bool SendDiscordMessage(string message, bool isError = false) {
    if(!isWebRequestEnabled) return false;

    message = (isError ? "❌ " : "✅ ") + message;
    string payload = "{\"content\":\"" + EscapeJSON(message) + "\"}";
    string headers = "Content-Type: application/json\r\n";

    char post[], result[];
    ArrayResize(post, StringToCharArray(payload, post, 0, WHOLE_ARRAY, CP_UTF8) - 1);

    int res = WebRequest(
        "POST",
        discord_webhook,
        headers,
        5000,
        post,
        result,
        headers
    );

    if(res == 200 || res == 204) {
        lastMessageTime = TimeCurrent();
        return true;
    }

    return false;
}
```

We've put our Discord integration into practice by putting a basic random trading method into place. This tactic demonstrates how to successfully combine trade logic with Discord alerts, even if its main purpose is educational:

```
void PlaceRandomTrade() {
    bool isBuy = (MathRand() % 2) == 1;

    double price = isBuy ? SymbolInfoDouble(_Symbol, SYMBOL_ASK)
                        : SymbolInfoDouble(_Symbol, SYMBOL_BID);

    int slPoints = 50 + (MathRand() % 100);
    int tpPoints = 50 + (MathRand() % 100);

    double sl = isBuy ? price - slPoints * _Point : price + slPoints * _Point;
    double tp = isBuy ? price + tpPoints * _Point : price - tpPoints * _Point;

    TradeInfo tradeInfo;
    tradeInfo.symbol = _Symbol;
    tradeInfo.type = isBuy ? "BUY" : "SELL";
    tradeInfo.price = price;
    tradeInfo.lots = LotSize;
    tradeInfo.sl = sl;
    tradeInfo.tp = tp;

    string message = FormatTradeMessage(tradeInfo);
    if(SendDiscordMessage(message)) {
        trade.SetExpertMagicNumber(magicNumber);

        bool success = isBuy ?
            trade.PositionOpen(_Symbol, ORDER_TYPE_BUY, LotSize, price, sl, tp, "Random Strategy Trade") :
            trade.PositionOpen(_Symbol, ORDER_TYPE_SELL, LotSize, price, sl, tp, "Random Strategy Trade");

        if(success) {
            SendDiscordMessage("✅ Trade executed successfully! Ticket: " + IntegerToString(trade.ResultOrder()));
        }
    }
}
```

For traders who wish to keep an eye on price moves without continuously monitoring their trading platform, regular market updates may be quite helpful. We've put in place a pricing update feature that updates Discord on a regular basis:

````
void SendPriceUpdate() {
    if(!SendPriceUpdates) return;
    if(TimeCurrent() - lastMessageTime < 300) return;

    string message = "```\n";
    message += "Price Update for " + _Symbol + "\n";
    message += "Bid: " + DoubleToString(SymbolInfoDouble(_Symbol, SYMBOL_BID), _Digits) + "\n";
    message += "Ask: " + DoubleToString(SymbolInfoDouble(_Symbol, SYMBOL_ASK), _Digits) + "\n";
    message += "Spread: " + DoubleToString(SymbolInfoInteger(_Symbol, SYMBOL_SPREAD), 0) + " points\n";
    message += "```";

    SendDiscordMessage(message);
}
````

When the Expert Advisor is terminated, the implementation contains the following appropriate cleaning procedures:

````
void OnDeinit(const int reason) {
    SendDiscordMessage("```\nEA stopped. Reason code: " +
                      IntegerToString(reason) + "```");
}
````

#### Example Result

With this integration example, you should observe the following:

Example of integration of Discord with MQL5 (MT5) - YouTube

[Photo image of MQL5.community](https://www.youtube.com/channel/UC8bwZMk2yh5WDToFos49OjQ?embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F16682)

MQL5.community

1.91K subscribers

[Example of integration of Discord with MQL5 (MT5)](https://www.youtube.com/watch?v=v2D8pRdxBUE)

MQL5.community

Search

Watch later

Share

Copy link

Info

Shopping

Tap to unmute

If playback doesn't begin shortly, try restarting your device.

More videos

## More videos

You're signed out

Videos you watch may be added to the TV's watch history and influence TV recommendations. To avoid this, cancel and sign in to YouTube on your computer.

CancelConfirm

Share

Include playlist

An error occurred while retrieving sharing information. Please try again later.

[Watch on](https://www.youtube.com/watch?v=v2D8pRdxBUE&embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F16682)

0:00

0:00 / 0:25

•Live

•

#### Security and Webhook Management

A number of crucial factors need to be taken into mind while integrating Discord in a production setting. Security comes first. Since anybody with access to the webhook URL can post messages to your Discord channel, it should be handled as a sensitive piece of data. To safeguard the webhook URL, it is advised to either employ encryption techniques or put it in a secure configuration file.

Another important factor to take into account is network dependability. Discord's API may occasionally encounter outages or rate limitations, and internet connections can be erratic. To deal with these circumstances politely, your solution should include strong error handling and retry features. This might entail keeping a message queue for retrying unsuccessful efforts and putting in place an exponential backoff technique for failed messages.

#### Performance Considerations

Additionally crucial is performance improvement, especially for high-frequency trading systems. Discord alerts are helpful, but they shouldn't affect how well your trade logic works. To manage Discord interactions without interfering with your primary trading activities, think about putting in place a separate thread or employing asynchronous message queuing.

There are several methods to expand the capabilities of the Discord integration. For instance, you may wish to include features like real-time performance indicators, such as trading statistics and profit/loss computations Alerts from risk management that let you know when certain exposure levels are met Market analysis is updated according to the indicators of your trading plan. You should also monitor the health of the system, including error rates and connection Discord-based custom commands that let you change parameters or check the status of your trading system.

To put these additions into practice, you'll need to think carefully about how to properly arrange various kinds of information and how to structure your messages. It is possible to use Discord's markdown style to produce visually unique messages for various notification kinds, which facilitates the rapid identification of crucial information.

Anyway, it is always good to find a balance between offering helpful information and preventing information overload when adding new features. Think about the most important notifications. You may want to disable particular notification kinds or distinct webhook URLs for various message categories.

#### Practical Use Cases

The following are some possible real-world uses for this Discord integration:

- Portfolio Management: Tracking several trading techniques across different accounts
- Danger management: Being alerted right away when your preset levels are crossed
- Obtaining frequent updates on strategy performance and market conditions
- Sharing market research and trading signals with a trading team


The approach we've spoken about offers these apps a strong base while also being adaptable enough to meet certain requirements. Understanding the integration's technical facets as well as the real-world requirements of the traders who will be utilizing the system is essential for effective deployment.

Maintaining dependable functionality of the Discord integration requires periodic testing and observation. This entails keeping an eye on message delivery rates, tracing down unsuccessful communications, and making sure that all important alerts are sent out on time. Think about putting in place logging systems that keep track of all Discord conversations and any mistakes that happen, enabling you to promptly find and fix any problems that may come up.

Your creativity and trading requirements are the only restrictions on the many options for expanding and improving this connection. You may build a strong communication system that improves your trading operations and enables you to keep updated about your trading activities from any location in the globe by beginning with this strong foundation and progressively adding features based on feedback and real-world usage.

#### Monitoring and Maintenance

Keep in mind that having the appropriate resources and knowledge at your disposal is just as important to effective trading as having sound techniques. This Discord integration is a vital tool for contemporary traders as it establishes a vital connection between your automated trading system and your capacity to track and react to market conditions instantly.

### Conclusion

In conclusion, Discord's integration with MetaTrader 5 is another useful solution offering traders a way to track and react to market circumstances. The advantages of real-time notifications, team communication, and remote monitoring capabilities make it very useful and convenient. However, you will need some time and knowledge to implement it. Also, you need to be very careful and pay attention attention to security, network dependability, and performance optimization. The possibility of improved features like voice channel notifications, interactive instructions, and machine learning-based alert filtering provides even more opportunities for trading efficiency as both platforms develop further. No matter where they are, traders can keep greater control over their trading activity and make sure they never miss important market moves or trading signals by creating this crucial connection between automated trading systems and real-time communication.

| File | Save Path |
| --- | --- |
| Discord\_examples.mq5 | Save this in the following folder: MQL5/Experts/ |

Best regards, Javier S. Gastón de Iriarte Cabrera

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/16682.zip "Download all attachments in the single ZIP archive")

[Discord\_example.mq5](https://www.mql5.com/en/articles/download/16682/discord_example.mq5 "Download Discord_example.mq5")(20.38 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Trading Insights Through Volume: Trend Confirmation](https://www.mql5.com/en/articles/16573)
- [Trading Insights Through Volume: Moving Beyond OHLC Charts](https://www.mql5.com/en/articles/16445)
- [From Python to MQL5: A Journey into Quantum-Inspired Trading Systems](https://www.mql5.com/en/articles/16300)
- [Example of new Indicator and Conditional LSTM](https://www.mql5.com/en/articles/15956)
- [Scalping Orderflow for MQL5](https://www.mql5.com/en/articles/15895)
- [Using PSAR, Heiken Ashi, and Deep Learning Together for Trading](https://www.mql5.com/en/articles/15868)

**[Go to discussion](https://www.mql5.com/en/forum/478479)**

![Integrate Your Own LLM into EA (Part 5): Develop and Test Trading Strategy with LLMs (III) – Adapter-Tuning](https://c.mql5.com/2/106/Integrate_Your_Own_LLM_into_EA_Part_5___LOGO__1.png)[Integrate Your Own LLM into EA (Part 5): Develop and Test Trading Strategy with LLMs (III) – Adapter-Tuning](https://www.mql5.com/en/articles/13500)

With the rapid development of artificial intelligence today, language models (LLMs) are an important part of artificial intelligence, so we should think about how to integrate powerful LLMs into our algorithmic trading. For most people, it is difficult to fine-tune these powerful models according to their needs, deploy them locally, and then apply them to algorithmic trading. This series of articles will take a step-by-step approach to achieve this goal.

![Mastering File Operations in MQL5: From Basic I/O to Building a Custom CSV Reader](https://c.mql5.com/2/106/Mastering_File_Operations_in_MQL5_LOGO.png)[Mastering File Operations in MQL5: From Basic I/O to Building a Custom CSV Reader](https://www.mql5.com/en/articles/16614)

This article focuses on essential MQL5 file-handling techniques, spanning trade logs, CSV processing, and external data integration. It offers both conceptual understanding and hands-on coding guidance. Readers will learn to build a custom CSV importer class step-by-step, gaining practical skills for real-world applications.

![MQL5 Wizard Techniques you should know (Part 51): Reinforcement Learning with SAC](https://c.mql5.com/2/107/MQL_Wizard_Techniques_you_should_know_Part_51_LOGO.png)[MQL5 Wizard Techniques you should know (Part 51): Reinforcement Learning with SAC](https://www.mql5.com/en/articles/16695)

Soft Actor Critic is a Reinforcement Learning algorithm that utilizes 3 neural networks. An actor network and 2 critic networks. These machine learning models are paired in a master slave partnership where the critics are modelled to improve the forecast accuracy of the actor network. While also introducing ONNX in these series, we explore how these ideas could be put to test as a custom signal of a wizard assembled Expert Advisor.

![Integrating MQL5 with data processing packages (Part 4): Big Data Handling](https://c.mql5.com/2/106/Integrating_MQL5_with_data_processing_packages_Part_4_Big_Data_Handling_Logo.png)[Integrating MQL5 with data processing packages (Part 4): Big Data Handling](https://www.mql5.com/en/articles/16446)

Exploring advanced techniques to integrate MQL5 with powerful data processing tools, this part focuses on efficient handling of big data to enhance trading analysis and decision-making.

[![](https://www.mql5.com/ff/sh/x8fwvn495ta7y774z2/01.png)Does your broker offer sponsored hosting for trading?Now it's even easier to get MetaTrader VPS for free – contact your broker for details](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=xscnzeyhifcgygpwvysykhqydcmmbgpp&s=f87b748147e376d34c8f0fdb9737b1766f20cc2174769a0e6b9975b5c2e8ddae&uid=&ref=https://www.mql5.com/en/articles/16682&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5068103371459589496)

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