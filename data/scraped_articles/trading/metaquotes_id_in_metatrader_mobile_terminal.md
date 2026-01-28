---
title: MetaQuotes ID in MetaTrader Mobile Terminal
url: https://www.mql5.com/en/articles/476
categories: Trading
relevance_score: 3
scraped_at: 2026-01-23T18:21:47.785636
---

[![](https://www.mql5.com/ff/sh/6xjc81sb5f2g45z9z2/01.png)Follow MQL5.community on social mediaWe publish the best technical materials from experts – free from advertising and irrelevant contentLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=yexgeaiatphxecqagtoxizolvboismyb&s=4e531fd1f983c26570e2dac7588b735354f2f9e0aea561427c030e4a1d2f060b&uid=&ref=https://www.mql5.com/en/articles/476&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5069375656736785418)

MetaTrader 5 / Examples


Android and iOS powered devices offer us many features we do not even know about. One of these features is push notifications allowing us to receive personal messages, regardless of our phone number or mobile network operator. MetaTrader mobile terminal already can receive such messages right from your trading robot. You should only know MetaQuotes ID of your device. More than 9 000 000 mobile terminals have already received it.

The world around us is constantly changing. Few people remember paging, though it was extremely popular at the time. GSM phones granted us the ability to send SMS messages to any cellular network user and paging was soon forgotten.

Can we long for more? Yes, we can! We can expand our opportunities even further with push notifications - the new service provided by modern smartphones.

### What is MetaQuotes ID

Push technology is widely used in mobile devices powered by iOS and Android OS allowing their users to receive instant notifications from various services in one place.

![MetaQuotes ID in MetaTarder 5 for iPhone](https://c.mql5.com/2/25/my_id__2.png)[Push notifications](https://www.metatrader5.com/en/mobile-trading/iphone/help/push "https://www.metatrader5.com/en/mobile-trading/iphone/help/push") are noted for their instant delivery. Besides, there is no need to launch third-party applications and keep them running. Also, push notifications cannot be lost in delivery and users do not depend on a specific mobile network operator. Only the appropriate device and Internet access are needed.

MetaQuotes ID is a unique user ID allowing to receive push notifications from MetaQuotes Software Corp. services and applications right on a mobile device. MetaQuotes ID is submitted to a user when installing the mobile version of the terminal:

| ![](https://c.mql5.com/2/4/mt4.png) | ![](https://c.mql5.com/2/4/mt5.png) |
| --- | --- |
| [MetaTrader 4 for iPhone](https://download.mql5.com/cdn/mobile/mt4/ios "https://download.mql5.com/cdn/mobile/mt4/ios") | [MetaTrader 5 for iPhone](https://download.mql5.com/cdn/mobile/mt5/ios "https://download.mql5.com/cdn/mobile/mt5/ios") |
| [MetaTrader 4 for Android](https://download.mql5.com/cdn/mobile/mt4/android?utm_campaign=MQL5.community "https://download.mql5.com/cdn/mobile/mt4/android?utm_campaign=MQL5.community") | [MetaTrader 5 for Android](https://download.mql5.com/cdn/mobile/mt5/android?utm_campaign=MQL5.community "https://download.mql5.com/cdn/mobile/mt5/android") |

A separate ID is submitted in each case preventing users from getting lost in a great amount of notifications. You can find your MetaQuotes ID in "Messages" section after installing the application. The screenshot on the left shows a user ID in MetaTrader 5 for iPhone.

Compared to unreliable SMS messages, you are not bound to a specific phone number and the messages are absolutely free. You can receive almost unlimited number of messages. You just need to ensure that your tariff plan includes Internet access.

Before the advent of push notifications, traders could use their mobile phones to receive messages about their trading account status, trading signals and other relevant information. To do this, an email was usually sent to a specific address, from which the appropriate message was then sent to a device via paid SMS gateways. But not all mobile network operators provide such a possibility. Besides, the system has several drawbacks.

With push notifications in MetaTrader mobile terminals, you can not only trade from everywhere but also use one more convenient way of working with your client terminal providing you with trading signals and important notifications on your account status. MetaQuotes Software Corp. took a step further and integrated new technologies in MQL5.community services.

### How it works?

Notifications received by users via MetaQuotes ID can be of two types: notifications from the desktop version of the client terminal and from [MQL5.community](https://www.mql5.com/) services.

To subscribe for the client terminal's notifications, specify MetaQuotes ID in the terminal's settings. To receive push notifications from MQL5.community, specify MetaQuotes ID in your profile. In both cases notifications are sent to a mobile device with a specified MetaQuotes ID via a special server when a certain event occurs. They are delivered instantly.

### Delivering messages from the client terminal

The main advantage of push notifications is the ability to quickly react to various events in the trading terminal. Specify your MetaQuotes ID and enable push notifications in [settings](https://www.metatrader5.com/en/terminal/help/startworking/settings "https://www.metatrader5.com/en/terminal/help/startworking/settings") to receive them on your mobile device:

![Configuring push notifications in the client terminal](https://c.mql5.com/2/4/options_notifications__1.png)

**Sending via MQL5 and MQL4**

The most interesting feature of sending notifications is adding an appropriate functionality to a trading robot. Special [SendNotification()](https://www.mql5.com/en/docs/network/sendnotification) function is provided in MQL4 and [MQL5](https://www.metatrader5.com/ "https://www.metatrader5.com/") languages. The function is easy to use:

```
//+------------------------------------------------------------------+
//|                                                 Notification.mq5 |
//|                        Copyright 2012, MetaQuotes Software Corp. |
//|                                              https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2012, MetaQuotes Software Corp."
#property link      "https://www.mql5.com"
#property version   "1.00"
#property script_show_inputs;
//+------------------------------------------------------------------+
//| Text message to send                                             |
//+------------------------------------------------------------------+
input string message="Enter message text";
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
//--- Send the message
   res=SendNotification(message);
   if(!res)
     {
      Print("Message sending failed");
     }
   else
     {
      Print("Message sent");
     }
//---
  }
//+------------------------------------------------------------------+
```

This sample MQL5 script sends a message to a mobile device with MetaQuotes ID specified in the terminal settings. The only parameter of SendNotification() function is a message text which should not exceed 255 characters.

With this function, you will always be in touch with your trading account and Expert Advisor. Forward-looking developers already introduce this feature in their [Market](https://www.mql5.com/en/market) products to provide additional convenient functionality.

**Sending via alerts**

You do not have to know MQL4 or MQL5 to work with push notifications. Messages sending can be configured via "Alerts" function in [MetaTrader 4](https://www.metatrader4.com/ "https://www.metatrader4.com/") and [MetaTrader 5](https://www.metatrader5.com/en/terminal/help/startworking/interface "https://www.metatrader5.com/en/terminal/help/startworking/interface") terminals:

![Sending push notifications via Alerts in MetaTrader 5](https://c.mql5.com/2/4/toolbox_en.png)

Select "Notification" in "Action" and enter the text that should be sent when a specified event occurs in "Source" field.

With this feature you will not miss a single important event.

### Integration with MQL5.community services

MetaTrader 5 is closely integrated with MQL5.community providing traders with unique opportunities including direct access to [Code Base](https://www.mql5.com/en/code), [Articles](https://www.mql5.com/en/articles) and [Market](https://www.mql5.com/en/market), [MQL5 Cloud Network](https://cloud.mql5.com/en "https://cloud.mql5.com/en"), [MQL5 source codes Storage](https://www.metatrader5.com/en/metaeditor/help/mql5storage "https://www.metatrader5.com/en/metaeditor/help/mql5storage") and more. Working in MetaTrader 5 is closely connected with activity on MQL5.community.

**Community events**

How can push notifications be of any use here? They facilitate convenient working with your MQL5.community account. After specifying MetaQuotes ID in your profile, you will receive notifications on the following important events:

- Personal messages including their texts
- Messages from customers or developers from Freelance service
- Changes in the status of your publications in Code Base, Articles, Market and Signals
- Moderator comments to your publications in Code Base, Articles, Market and Signals
- Information messages about your rented [virtual hosting](https://www.mql5.com/en/vps)
- Comments to forum topics on blog posts

- Announcements of new publications in Code Base and Articles
- New orders in Freelance service
- Notifications on registering as a Seller in Market

Enter your profile and specify MetaQuotes ID:

![MetaQuotes ID in MQL5.community member's profile](https://c.mql5.com/2/25/mql5community_profile__2.png)

Then choose events you want to receive notifications about.

![Setting up notifications about about events on MQL5.community](https://c.mql5.com/2/25/mql5_send_pm__2.png)

**MQL5 account security**

For additional protection of your account, you can enable two-step authentication which is also based on push notifications and MetaQuotes ID. If you open the site from an unknown IP address, then in addition to the login and password you'll need to enter a special one-time code which will be sent to your mobile device using MetaQuotes ID. Enable "Authorize from allowed static IP addresses only" option at Profile — Settings — Security:

![Enabling two-step authentication for an MQL5 account](https://c.mql5.com/2/25/twostep__2.png)

If you use a static IP address, add it to the list. You won't need to enter a one-time code when visiting the site from that address. In all other cases, a code to login to your MQL5 account will be sent to your MetaTrader 4/5 mobile terminal.

**Chat with MQL5.community friends and colleagues**

MetaTrader 4/5 mobile terminals include a chat allowing you to communicate with your MQL5.community friends and colleagues right from your smartphone.

![Chats in MetaTrader 4/5 mobile terminals](https://c.mql5.com/2/25/message_history_en__1.png)

### Be mobile with MetaTrader terminals

Being alert to market changes is a key ability of a successful trader. [MetaTrader 4](https://www.metatrader4.com/ "https://www.metatrader4.com/") and [MetaTrader 5](https://www.metatrader5.com/ "https://www.metatrader5.com/") trading platforms have everything to be up-to-date. Mobile terminals for the most popular [iOS](https://download.mql5.com/cdn/mobile/mt5/ios "https://download.mql5.com/cdn/mobile/mt5/ios") and [Android OS](https://download.mql5.com/cdn/mobile/mt5/android?utm_campaign=MQL5.community "https://download.mql5.com/cdn/mobile/mt5/android") platforms are available to all traders for free.

Sample MQL5 Expert Advisors for creating a message box on a chart are attached below. Place them to the \[terminal dara folder\]\\MQL5\\Experts, compile in [MetaEditor](https://www.metatrader5.com/en/automated-trading/metaeditor "https://www.metatrader5.com/en/automated-trading/metaeditor") and launch on any chart. Try to send notifications to your mobile device. You will surely find that easy and convenient.

Use push notifications to receive data instantly and securely. More than 9 000 000 unique MetaQuotes IDs have already been registered as of December 2016.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/476](https://www.mql5.com/ru/articles/476)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/476.zip "Download all attachments in the single ZIP archive")

[sendnotificationsample.mq5](https://www.mql5.com/en/articles/download/476/sendnotificationsample.mq5 "Download sendnotificationsample.mq5")(4.55 KB)

[message\_pane.mq5](https://www.mql5.com/en/articles/download/476/message_pane.mq5 "Download message_pane.mq5")(4.32 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

#### Other articles by this author

- [Getting Started with MQL5 Algo Forge](https://www.mql5.com/en/articles/18518)
- [Installing MetaTrader 5 and Other MetaQuotes Apps on HarmonyOS NEXT](https://www.mql5.com/en/articles/18612)
- [MetaTrader 5 on macOS](https://www.mql5.com/en/articles/619)
- [How to earn money by fulfilling traders' orders in the Freelance service](https://www.mql5.com/en/articles/1019)
- [MetaTrader 4 on macOS](https://www.mql5.com/en/articles/1356)
- [Working with ONNX models in float16 and float8 formats](https://www.mql5.com/en/articles/14330)
- [Regression models of the Scikit-learn Library and their export to ONNX](https://www.mql5.com/en/articles/13538)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/7599)**
(183)


![Jun Ping Fan](https://c.mql5.com/avatar/2025/2/67b9a7db-249b.jpg)

**[Jun Ping Fan](https://www.mql5.com/en/users/wanjunzhiwang)**
\|
22 Feb 2025 at 10:42

You need to log in to your MOL5 account on your mobile first, turn on notifications (Settings on your phone return to MT4) and then the ID's will appear underneath, the ones I've tried are already bound.


![emlynjarode](https://c.mql5.com/avatar/avatar_na2.png)

**[emlynjarode](https://www.mql5.com/en/users/emlynjarode)**
\|
3 Apr 2025 at 09:38

My android app is not showing the Meta quotes ID ,  how do I fix this am feeling really frustrated 🥴


![Michael Charles Schefe](https://c.mql5.com/avatar/2021/5/60ADC6A5-6810.gif)

**[Michael Charles Schefe](https://www.mql5.com/en/users/ausimike)**
\|
3 Apr 2025 at 11:49

**emlynjarode [#](https://www.mql5.com/en/forum/7599/page9#comment_56343081):**

My android app is not showing the Meta quotes ID ,  how do I fix this am feeling really frustrated 🥴

contact support at bottom of every page on this website called "Contacts and requests". They will give you advice or they may be able to reset your id or send you new one.

However, I do remember reading a thread that not every device is compatible with the service. But, good luck to you.

![Manuel Ricardo Davila Dena](https://c.mql5.com/avatar/2025/6/683de2d4-4d45.jpg)

**[Manuel Ricardo Davila Dena](https://www.mql5.com/en/users/390351624)**
\|
13 Apr 2025 at 22:40

[https://c.mql5.com/3/461/WhatsApp_Image_2025-04-13_at_15.15.58.jpeg](https://c.mql5.com/3/461/WhatsApp_Image_2025-04-13_at_15.15.58.jpeg "https://c.mql5.com/3/461/WhatsApp_Image_2025-04-13_at_15.15.58.jpeg")

Hi, has anyone else had a Metaquotes ID NUll on Android?

[![METAQUOTES ID NULL](https://c.mql5.com/3/461/WhatsApp_Image_2025-04-13_at_15.15.58__1.jpeg)](https://c.mql5.com/3/461/WhatsApp_Image_2025-04-13_at_15.15.58.jpeg "https://c.mql5.com/3/461/WhatsApp_Image_2025-04-13_at_15.15.58.jpeg")

![Miguel Angel Vico Alba](https://c.mql5.com/avatar/2025/10/68e99f33-714e.jpg)

**[Miguel Angel Vico Alba](https://www.mql5.com/en/users/mike_explosion)**
\|
13 Apr 2025 at 23:08

**Ricardo Davila [#](https://www.mql5.com/es/forum/19655#comment_56444933) :** Hi, has anyone else had NUll of Metaquotes ID on Android happen to them?

[Forum about trading, automated trading systems and testing of trading strategies](https://www.mql5.com/en/forum)

[MT5 does not show MQID](https://www.mql5.com/en/forum/378377)

[somethingtrading01](https://www.mql5.com/en/users/algotrading01), 24/09/2021 20:29

When I connect to WI-FI and click the MQID button in the message bar, I get the code; however, when I restart the app and click the MQID button when using MOBILE DATA, it says it is null.

[Forum for trading, automated trading systems and trading strategy testing](https://www.mql5.com/en/forum)

[MT5 does not show MQID](https://www.mql5.com/en/forum/378377#comment_26511681)

[Alexey Petrov](https://www.mql5.com/en/users/Alexx), 16/12/2021, 09:21am

Your mobile device can only register MetaQuotes ID if a valid Google account is used on it.

![How to purchase a trading robot from the MetaTrader Market and to install it?](https://c.mql5.com/2/0/MQL5_market__1.png)[How to purchase a trading robot from the MetaTrader Market and to install it?](https://www.mql5.com/en/articles/498)

A product from the MetaTrader Market can be purchased on the MQL5.com website or straight from the MetaTrader 4 and MetaTrader 5 trading platforms. Choose a desired product that suits your trading style, pay for it using your preferred payment method, and activate the product.

![Trade Operations in MQL5 - It's Easy](https://c.mql5.com/2/0/egg.png)[Trade Operations in MQL5 - It's Easy](https://www.mql5.com/en/articles/481)

Almost all traders come to market to make money but some traders also enjoy the process itself. However, it is not only manual trading that can provide you with an exciting experience. Automated trading systems development can also be quite absorbing. Creating a trading robot can be as interesting as reading a good mystery novel.

![Fundamentals of Statistics](https://c.mql5.com/2/0/statistic.png)[Fundamentals of Statistics](https://www.mql5.com/en/articles/387)

Every trader works using certain statistical calculations, even if being a supporter of fundamental analysis. This article walks you through the fundamentals of statistics, its basic elements and shows the importance of statistics in decision making.

![Interview with Irina Korobeinikova (irishka.rf)](https://c.mql5.com/2/0/zh0ku.png)[Interview with Irina Korobeinikova (irishka.rf)](https://www.mql5.com/en/articles/465)

Having a female member on the MQL5.community is rare. This interview was inspired by a one of a kind case. Irina Korobeinikova (irishka.rf) is a fifteen-year-old programmer from Izhevsk. She is currently the only girl who actively participates in the "Jobs" service and is featured on the Top Developers list.

[![](https://www.mql5.com/ff/sh/5z040u47jcv59943z2/6c76c03a8b37e08b8655a1a085770b7a.jpg)\\
MetaTrader 5 for iOS and Android\\
\\
Fully featured platform for any devices and web browsers\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=ddonqpipxfqlnsvzlwuowsuwlejpyjxk&s=9daba65b69f40afc3c35f95b1f84ef5824d68c47f29ce96a6dc5b164a2727baa&uid=&ref=https://www.mql5.com/en/articles/476&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5069375656736785418)

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