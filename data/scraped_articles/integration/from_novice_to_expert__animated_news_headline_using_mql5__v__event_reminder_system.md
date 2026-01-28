---
title: From Novice to Expert: Animated News Headline Using MQL5 (V)—Event Reminder System
url: https://www.mql5.com/en/articles/18750
categories: Integration
relevance_score: 3
scraped_at: 2026-01-23T21:06:17.322339
---

[![](https://www.mql5.com/ff/sh/6xjc81sb5f2g45z9z2/01.png)Follow MQL5.community on social mediaWe publish the best technical materials from experts – free from advertising and irrelevant contentLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=yexgeaiatphxecqagtoxizolvboismyb&s=4e531fd1f983c26570e2dac7588b735354f2f9e0aea561427c030e4a1d2f060b&uid=&ref=https://www.mql5.com/en/articles/18750&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5071575831043582615)

MetaTrader 5 / Examples


### Contents:

- [Introduction](https://www.mql5.com/en/articles/18750#1)
- [Concept](https://www.mql5.com/en/articles/18750#2)
- [Implementation](https://www.mql5.com/en/articles/18750#3)
- [Testing](https://www.mql5.com/en/articles/18750#4)
- [Conclusion](https://www.mql5.com/en/articles/18750#5)
- [Key Lessons](https://www.mql5.com/en/articles/18750#para6)
- [Attachments](https://www.mql5.com/en/articles/18750#para7)

### Introduction

High-impact economic news releases can trigger sharp price swings, sometimes delivering big rewards and other times catching traders off guard with sudden losses. Our News Headline EA already gives you an edge by displaying upcoming events, technical indicator insights, and AI-generated commentary right on the chart, so you don’t have to switch between windows. But let’s face it: no one wants to sit glued to the screen all day waiting for the next data release.

To make this workflow truly practical, we’re taking the next step by integrating an intelligent alerting system that will proactively notify you moments before each major event. This way, you can stay focused on your strategy, confident you won’t miss critical news when it matters most.

Because this series is designed for both experienced developers and those new to MetaTrader 5, we’ve organized all notification-related material into a dedicated section. Here, we’ll explore the platform’s built-in alerting capabilities in detail, from on-screen pop-ups and sounds to push notifications on your phone, so you can tailor timely reminders to suit your trading style.

In this discussion, we’ll tackle the current limitations of our News Headline EA while also equipping you with the skills to integrate MQL5 alerts into any project you’re working on. In the next section, we’ll break down exactly how MQL5 alerts function and walk through our plan for bringing them seamlessly into the EA. With this foundation in place, you’ll be ready to move confidently into the code implementation.

### Concept

Under this section we will first explore the offerings of MQL5 when it comes to alerts, and we will discuss the integration plan. In the next paragraph, get to understand the alerts when in MetaTrader 5.

Understanding alerts in MOL5

Alerts are an essential feature in MetaTrader 5, designed to grab the trader’s attention when important market events or conditions occur—such as reaching specific price levels, triggering indicator signals, or executing trades. By default, alerts in the MetaTrader 5 terminal are managed via the Toolbox > Alerts tab, where users can configure basic alert conditions.

However, MQL5 provides powerful functions for creating customized alerts, including _Alert(), PlaySound(), SendMail()_, and _SendNotification()_. These functions generate pop-up dialog boxes, audible cues, and messages that can be received on both desktop and mobile MetaTrader 5 terminals. Even without using an explicit alert function, you can still log important events using Print() to output messages to the Journal tab.

Further integration is possible by extending the native _Alert()_ mechanism to work with external notification services such as Telegram, Discord, and others through API calls. While today's discussion does not delve into those external integrations, they have been covered in some of our previous sessions.

Instead, today’s focus is on understanding how to use alerts effectively in the context of news trading, where timing and visibility of information are critical for decision-making.

The table below provides a description of the alerting functions available in MQL5, along with how each can be effectively adapted and applied within the context of this project.

| Function | Description | Adaptability to the News Headline EA |
| --- | --- | --- |
| [Alert()](https://www.mql5.com/en/docs/common/alert) | Displays a pop-up dialog box with a message on the MetaTrader 5 terminal. It interrupts the user with immediate on-screen information and also logs the message in the Experts log. | Highly suitable for notifying traders of imminent high-impact events or sudden changes detected by the EA, ensuring the user’s attention is captured even during live trading sessions. |
| [PlaySound()](https://www.mql5.com/en/docs/common/playsound) | Plays a custom sound file (WAV format) on the desktop terminal. Useful for providing an audible signal without requiring the user to look at the screen. | Ideal for adding an audible cue when new news headlines, indicator signals, or AI insights appear in the EA, helping traders stay informed even when multitasking. |
| [SendMail()](https://www.mql5.com/en/docs/network/sendmail) | Sends an email message via the SMTP server configured in MetaTrader 5’s options. The email can include a subject and body text with relevant trading details. | Useful for delivering detailed news summaries or alerts about high-impact events to traders who prefer receiving updates outside the terminal, especially when they’re away from their trading desk. |
| [SendNotification()](https://www.mql5.com/en/docs/network/sendnotification) | Sends a push notification to the MetaTrader 5 mobile app, allowing traders to receive concise messages on their smartphones or tablets in real-time. | Perfect for notifying traders instantly about critical events, major news updates, or significant AI insights detected by the EA, ensuring they stay connected regardless of their physical location. |

Referring to the table above, options that deliver immediate attention, such as terminal pop-up alerts and sounds, are particularly valuable because we want to be notified ahead of upcoming events to prepare in time. In this context, terminal-based alerts will take higher priority, while _SendMail()_ and _SendNotification()_ will be used as secondary options. Although email and push notifications introduce a slight delay due to message transmission, the delay is generally minimal—but it’s still important to consider when timing is critical.

Expandability

As mentioned earlier, we can broaden the EA’s capabilities to support external notifications on other platforms by integrating APIs. This involves creating custom functions that gather alert details and transmit them via _WebRequest_ to external servers, enabling communication and notifications through services like Slack, Telegram, SMS gateways, or custom web apps. This approach makes the News Headline EA highly flexible and able to fit seamlessly into diverse trading workflows and notification ecosystems.

Integration plan

Today, our primary focus will be on implementing alerts for upcoming news events, designed to trigger at a preset time before each event occurs. There are three critical categories of news alerts, corresponding to the news impact levels: high, medium, and low. For now, we’ll concentrate exclusively on managing alerts for these news events and set aside other features of the EA. It’s important to note that not every news item needs to generate an alert—instead, we want to provide traders with the flexibility to select which impact levels they wish to be notified about,  and the time in minutes to get notified  before each event takes place.

### Implementation

Step 1: Inputs Configuration

The integration begins by defining all alert‑related settings in one place: global toggles for alerts and push notifications, separate switches for high/medium/low-impact levels, and the “minutes before” parameter. This centralized configuration not only simplifies the user interface—traders can tailor exactly which events trigger pop‑ups or pushes—but also keeps the EA’s conditional logic lean. On initialization, the EA reads these inputs and then simply checks each one before firing any alert, avoiding hard‑coded values or buried flags throughout the code.

```
//--- ALERT INPUTS ---------------------------------------------------
input bool   InpEnableAlerts        = true; //Enable Alerts
input bool   InpAlertHigh           = true; //High Impact Alerts
input bool   InpAlertMed            = false; // Medium Impact Alerts
input bool   InpAlertLow            = false; // Low Impact Alerts
input int    InpAlertMinutesBefore  = 5; //Alert Minutes Before Event

//--- PUSH NOTIFICATIONS INPUTS -------------------------------------
input bool   InpEnablePush          = false; //Enable Push Notifications
input bool   InpPushHigh            = true;  //High Impact Push
input bool   InpPushMed             = false;  //Medium Impact Push
input bool   InpPushLow             = false;  //Low impact Push
```

Step 2: Event State Tracking

Each calendar event is wrapped in a CEvent object that carries its own alerted boolean, initialized to false. When the program determines it’s time to notify, the EA itself calls _Alert()_ (for the on‑screen popup) and, if enabled, _SendNotification()_ (for mobile push), then immediately sets that event’s alerted flag to true. This pattern prevents duplicates without any external lookup structures. A word of caution: always update your event’s state immediately after invoking the notification to guarantee exactly one alert per event.

```
// Event storage
class CEvent : public CObject
{
public:
  datetime time;
  string   sym, name;
  int      imp;
  bool     alerted;

  CEvent(datetime t,const string &S,const string &N,int I)
  {
    time    = t;
    sym     = S;
    name    = N;
    imp     = I;
    alerted = false;
  }
};
```

Step 3: Notification Dispatch Logic

All notification dispatch happens inside _CheckAndAlertEvents()_, executed every timer tick. The routine first verifies the master alert switch, computes one deadline timestamp ( _now_ \+ _minutesBefore \* 60)_, and then iterates the high, medium, and low-impact arrays in turn. Upon finding an event that hasn’t yet been alerted, falls within the deadline, and matches the user’s chosen impact levels, the EA constructs a concise message and fires the notification calls. Centralizing this logic makes it straightforward to add alternate channels—such as email or SMS—by extending this single function rather than modifying multiple parts of the EA.

```
//+------------------------------------------------------------------+
//| CheckAndAlertEvents: popups and optional push                    |
//+------------------------------------------------------------------+
void CheckAndAlertEvents()
{
  if(!InpEnableAlerts) return;
  datetime now = TimeTradeServer();
  datetime threshold = now + InpAlertMinutesBefore * 60;
  string msg;

  for(int i=0;i<ArraySize(highArr);i++)
  {
    CEvent *e=highArr[i];
    if(!e.alerted && e.time<=threshold && InpAlertHigh)
    {
      msg = "In "+IntegerToString(InpAlertMinutesBefore)+"m: "+e.sym+" "+e.name;
      Alert(msg);
      if(InpEnablePush && InpPushHigh) SendNotification(msg);
      e.alerted = true;
    }
  }
  for(int i=0;i<ArraySize(medArr);i++)
  {
    CEvent *e=medArr[i];
    if(!e.alerted && e.time<=threshold && InpAlertMed)
    {
      msg = "In "+IntegerToString(InpAlertMinutesBefore)+"m: "+e.sym+" "+e.name;
      Alert(msg);
      if(InpEnablePush && InpPushMed) SendNotification(msg);
      e.alerted = true;
    }
  }
  for(int i=0;i<ArraySize(lowArr);i++)
  {
    CEvent *e=lowArr[i];
    if(!e.alerted && e.time<=threshold && InpAlertLow)
    {
      msg = "In "+IntegerToString(InpAlertMinutesBefore)+"m: "+e.sym+" "+e.name;
      Alert(msg);
      if(InpEnablePush && InpPushLow) SendNotification(msg);
      e.alerted = true;
    }
  }
}
```

Step 4: OnTimer

To ensure smooth chart animations, the EA invokes _CheckAndAlertEvents()_ at the start of OnTimer(), before any data fetching or drawing. Since both _Alert()_ and _SendNotification()_ are non‑blocking in MQL5, the subsequent 20 ms redraws and scrolling operations proceed uninterrupted. It’s a useful tactic: run your side‑effect routines first in the loop so that any downstream rendering remains fluid.

```
void OnTimer()
{
  CheckAndAlertEvents();   // fire alerts and pushes first
  ReloadEvents();
  FetchAlphaVantageNews();
  FetchAIInsights();
  DrawAll();
  // … remaining drawing and scrolling …
}
```

Beneath all of this lies a modular design: inputs handle configuration, _ReloadEvents_ and _FetchAlphaVantageNews_ manage data retrieval, _CheckAndAlertEvents_ takes care of notifications, and _DrawAll_ plus its helpers manage on‑chart rendering. This clear separation of concerns allows you to swap out or enhance the notification mechanism—adding SMS, email, webhooks, etc.—by altering just one function, leaving the rest of the EA untouched. Building your Expert Advisor in this loosely coupled, well‑documented fashion makes it far easier to maintain, extend, and debug over time.

Testing

To test the new alerting capabilities, we loaded the updated News Headline EA onto a live chart and configured the “minutes before” input to target the next scheduled events. The input panel lets you specify exactly how many minutes in advance you wish to receive each alert, ensuring you’re notified at just the right moment. Below are screenshots from my hands‑on testing: one showing the EA’s input settings, another capturing the on‑screen pop‑up appearing minutes before the event, and a final image of the matching push notification arriving on my mobile terminal.

![Configuring alerts News Headline EA](https://c.mql5.com/2/155/Alert.gif)

Setting up the News Headline EA

Testing Push Notification

Whenever MetaTrader 5 is installed on a supported mobile device, it is assigned a unique MetaQuotes ID (MQID). This ID is crucial for enabling communication between the MQL5 platform and the desktop MetaTrader 5 terminal. To allow the mobile terminal to receive push notifications, you need to add your MQID to the desktop terminal and enable push notifications.

This can be done by opening the Options dialog, either through the Tools menu or by using the shortcut Ctrl + O. Within these settings, locate the section for Notifications, find your MQID in the mobile app, and enter it into the list of IDs you wish to use for alerts. See the image below for reference. This setting is essential—without it, you won’t be able to receive any notifications on your mobile device.

In turn, make sure that notifications are allowed for MetaTrader 5 on your mobile device, and consider setting a unique notification tone to catch your attention. This type of alert is especially valuable for traders, as people typically keep their mobile devices close by, ensuring they never miss an important alarm from the News Headline EA.

![Setting PushNotifications](https://c.mql5.com/2/155/Setting_PushNotifications.png)

Setting the Push Notifications

On my mobile MetaTrader 5 terminal, the push notifications arrived as expected. Below is an excerpt from the screenshot showing those alerts, corresponding to the on‑chart examples above.

![Push notifications receve on Android Mobile MetaTrader 5](https://c.mql5.com/2/155/Mobile_Terminal.png)

Push notifications received on Android Mobile MetaTrader 5

### Conclusion

It’s essential to incorporate alerting features into our Indicators and Expert Advisors to enhance their practical value for traders. In this project, we successfully integrated a notification system, significantly transforming the News Headline EA into something closer to a true trading companion. Expert Advisors have broad capabilities—not only can they perform trading operations, but they can also function like indicators or scripts to provide insights, visualizations, or automated responses.

In this particular case, the EA focuses on insights and the display of upcoming economic events rather than executing trades. Practically speaking, no trader can sit and watch the chart all day waiting for news and calendar updates. This makes the integration of alerts a perfect solution to notify users proactively about significant events. The addition of push notifications elevates the tool even further, enabling traders to receive timely updates on mobile devices. This brings true mobility and flexibility, especially when the EA is hosted online yet can still communicate with a remote mobile terminal.

Initially, I had planned for this to be the final update for the News Headline EA. However, reaching this stage has revealed new possibilities and improvements that could transform it into an even more comprehensive solution. For example, integrating trading logic directly linked to news events, designing a trading dashboard for news-driven strategies, and other advanced features remain exciting opportunities for future development. It’s often the case that fresh ideas emerge after each publication, and if I have the chance, I’d love to expand this project further.

I hope you’ve gained valuable insights from this discussion. Below, I’ve attached the full source code with all the details we’ve explored together and the Python files from the [previous article](https://www.mql5.com/en/articles/18685). You are welcome to build upon it, add new features, or refine the EA as you see fit. Your thoughts and feedback are very welcome in the comment section below!

### Key Lessons

| Lesson | Description |
| --- | --- |
| Centralized Configuration | Group all user‐adjustable parameters—toggles, thresholds, API keys—at the top of your code so that settings are easy to find, document, and modify without digging into the logic. |
| Timer‑Driven Architecture: | Use millisecond timers (OnTimer) to drive periodic tasks such as data fetching, alert checks, and canvas updates, balancing responsiveness against CPU load. |
| Canvas Double‐Buffering | Render all drawing operations to an off‐screen bitmap (Canvas) before calling Update, preventing flicker and ensuring smooth on‑chart animations. |
| Non‑Blocking Notifications | Invoke Alert() and SendNotification() early in your loop; because they are non‑blocking, they won’t pause your redraws or data processing. |
| Stateful Event Objects | Embed an “alerted” flag directly in each event object to track which events have already fired notifications, eliminating duplicate alerts without external maps. |
| Modular Separation of Concerns | Divide your EA into clear sections—configuration, data retrieval, alert logic, and rendering—to make maintenance and future extensions straightforward. |
| Rate‑Limiting Logic | Implement simple time‑based checks (e.g., “minutes before” thresholds) to prevent excessive or premature alerts and to control external API call frequency. |
| WebRequest Integration | Leverage MQL5’s WebRequest to call external services (news APIs, AI servers), handling headers, timeouts, and response parsing within your EA. |
| JSON Parsing Techniques | Extract only the necessary fields from returned JSON strings (e.g., titles or insight text) using StringFind and substring operations, keeping parsing logic robust but simple. |
| Cleanup and Resource Management | Always destroy created objects (Canvas, timers) and delete dynamic memory in OnDeinit to prevent memory leaks and to keep the platform stable. |

### Attachments

| Filename | Version | Description |
| --- | --- | --- |
| News Headline EA.mq5 | 1.07 | Economic Calendar, Alpha Vantage news, on‑chart indicator insights (RSI, Stoch, MACD, CCI), AI‑driven commentary lane, plus event alerts and optional push notifications. |
| download\_model.py | 1.00 | Simple Python script using the Hugging Face Hub client to download and cache the quantized GGUF model, printing its local file path. |
| serve\_insights.py | 1.00 | FastAPI application loading the GGUF model via llama‑cpp, exposing a POST /insights endpoint that generates and returns AI insights. |

[Back to contents](https://www.mql5.com/en/articles/18750#para0)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/18750.zip "Download all attachments in the single ZIP archive")

[\_News\_Headline\_EA.mq5](https://www.mql5.com/en/articles/download/18750/_news_headline_ea.mq5 "Download _News_Headline_EA.mq5")(40.41 KB)

[download\_model.py](https://www.mql5.com/en/articles/download/18750/download_model.py "Download download_model.py")(0.28 KB)

[serve\_insights.py](https://www.mql5.com/en/articles/download/18750/serve_insights.py "Download serve_insights.py")(1.77 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/490849)**
(1)


![Jason Smith](https://c.mql5.com/avatar/2025/5/682db193-4634.png)

**[Jason Smith](https://www.mql5.com/en/users/eddymorrow)**
\|
11 Jul 2025 at 08:41

Very nice. Thanx


![From Basic to Intermediate: Union (II)](https://c.mql5.com/2/101/Do_bwsico_ao_intermedisrio_Uniho_II.png)[From Basic to Intermediate: Union (II)](https://www.mql5.com/en/articles/15503)

Today we have a very funny and quite interesting article. We will look at Union and will try to solve the problem discussed earlier. We'll also explore some unusual situations that can arise when using union in applications. The materials presented here are intended for didactic purposes only. Under no circumstances should the application be viewed for any purpose other than to learn and master the concepts presented.

![Introduction to MQL5 (Part 18): Introduction to Wolfe Wave Pattern](https://c.mql5.com/2/155/18555-introduction-to-mql5-part-18-logo.png)[Introduction to MQL5 (Part 18): Introduction to Wolfe Wave Pattern](https://www.mql5.com/en/articles/18555)

This article explains the Wolfe Wave pattern in detail, covering both the bearish and bullish variations. It also breaks down the step-by-step logic used to identify valid buy and sell setups based on this advanced chart pattern.

![Non-linear regression models on the stock exchange](https://c.mql5.com/2/103/Nonlinear_regression_models_on_the_stock_exchange___LOGO.png)[Non-linear regression models on the stock exchange](https://www.mql5.com/en/articles/16473)

Non-linear regression models on the stock exchange: Is it possible to predict financial markets? Let's consider creating a model for forecasting prices for EURUSD, and make two robots based on it - in Python and MQL5.

![Self Optimizing Expert Advisors in MQL5 (Part 8): Multiple Strategy Analysis (3) — Weighted Voting Policy](https://c.mql5.com/2/155/18770-self-optimizing-expert-advisors-logo.png)[Self Optimizing Expert Advisors in MQL5 (Part 8): Multiple Strategy Analysis (3) — Weighted Voting Policy](https://www.mql5.com/en/articles/18770)

This article explores how determining the optimal number of strategies in an ensemble can be a complex task that is easier to solve through the use of the MetaTrader 5 genetic optimizer. The MQL5 Cloud is also employed as a key resource for accelerating backtesting and optimization. All in all, our discussion here sets the stage for developing statistical models to evaluate and improve trading strategies based on our initial ensemble results.

[Need a reliable hosting solution for your robots?Contact your broker and find out about available Sponsored MetaTrader VPS offeringsLearn more![](https://www.mql5.com/ff/sh/0pw0dk81s56qy774z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=vljwvezfjkfbvviocwskggexlvgykvob&s=70cf8e354b9a125332533ffb65d7365abe8dde5b5c1ede9caac479a9e9df4f25&uid=&ref=https://www.mql5.com/en/articles/18750&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5071575831043582615)

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