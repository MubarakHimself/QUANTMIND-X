---
title: Developing an MQTT client for MetaTrader 5: a TDD approach — Final
url: https://www.mql5.com/en/articles/14677
categories: Integration
relevance_score: 6
scraped_at: 2026-01-23T17:21:28.409911
---

[![](https://www.mql5.com/ff/sh/x8fwvn495ta7y774z2/01.png)Does your broker offer sponsored hosting for trading?Now it's even easier to get MetaTrader VPS for free – contact your broker for details](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=xscnzeyhifcgygpwvysykhqydcmmbgpp&s=f87b748147e376d34c8f0fdb9737b1766f20cc2174769a0e6b9975b5c2e8ddae&uid=&ref=https://www.mql5.com/en/articles/14677&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5068136142060058085)

MetaTrader 5 / Integration


> > > > "Our aim is always to work at the highest level of abstraction that is possible given a problem and the constraints on its solution." (Bjarne Stroustrup, Programming Principles and Practice Using C++)

### Introduction

Because this is the last article of this series, maybe a quick recap would be useful, or at least convenient.

In the [first article of this series](https://www.mql5.com/en/articles/12857), we saw that MQTT is a message-sharing protocol based on the publish/subscribe interaction model (pub/sub) that can be useful in a trading environment by allowing the user to share any kind of data in real-time: trade transactions, account info, statistical data for ingestion by a machine learning pipeline, in plain text, XML, JSON, or binary data, including images. MQTT is lightweight, resilient to network instabilities or interruptions, and content-agnostic. Besides that, the protocol is mature, battle-tested, and an open standard maintained by OASIS. The two most used versions of it, the previous 3.1.1 and the current 5.0 are among the most used protocols for connecting a virtually unlimited number of devices in the so-called Internet of Things. MQTT can be used in any scenario where real-time data-sharing between decoupled machines is required.

There are many MQTT brokers available for development, testing, and production environments, both open-source and commercial, and a lot of MQTT clients for virtually any modern programming language. You can check them in this [list of MQTT software that includes brokers, libraries, and tools](https://www.mql5.com/go?link=https://mqtt.org/software/ "https://mqtt.org/software/").

In the [second article of this series](https://www.mql5.com/en/articles/13334), we described our code organization for this client development and commented on some initial design choices, like the Object-Oriented paradigm. Most of that code changed in our first major refactoring, but the functionality remains the same. In that article, we also described some connections we made with a local broker running on the Windows Subsystem for Linux (WSL) just to realize that our CONNECT class was generating bad packets. We improved it and reported it in the next article.

The [third article of this series](https://www.mql5.com/en/articles/13388) was dedicated to taking some notes about the Operational Behavior part of the protocol and its relationship with the CONNECT flags. We described the semantics of those flags and how we are setting them. We also made some notes about the Test-Driven Development practice that we are using for this project. Finally, we explained how we are testing our classes’ protected methods.

We deep dive in the importance of the MQTT 5.0 Properties in the [fourth article of this series](https://www.mql5.com/en/articles/13651). We commented on each of them and their respective data types, or data representation in MQTT parlance. There we took some notes about how MQTT 5.0 Properties, in particular the User Property(ies), can be used to extend the protocol.

The PUBLISH Control Packet and its unique (non-reserved) fixed header flags were the topic of the [fifth article of this series](https://www.mql5.com/en/articles/13998). We dedicated a lot of space trying to show how these PUBLISH fixed header flags operate at the bit level. We marked out the characteristics of the different MQTT Quality of Service levels (QoS 0, QoS 1, and QoS 2) with some illustrative diagrams showing the packet exchange between the client and the broker in each of these QoS.

An intermezzo was described in our [sixth article of this series](https://www.mql5.com/en/articles/14391). It was our first major code refactoring. We changed the blueprint of our Control Packet classes and removed duplicated functions and obsolete test code. That article is mainly a documentation of these changes with some notes about the PUBACK Control Packet. The semantics of the PUBACK Reason Codes with their respective Reason Strings are annotated in that article.

Finally, in this seventh and last part we want to share with you some working code that is intended to address a very common trader need when it comes to the building of indicators signals to be used in Expert Advisors: the lack of a required symbol for the indicator in the trading account.

We suggest one possible solution using custom symbols and a pair of MQTT clients running as services on the Metatrader 5 terminal. Even though the demo code is oversimplified and runs on a single terminal instance, due to the main characteristic of the MQTT protocol itself - which is the decoupling between the sender and the receiver by a “broker” mediation - this solution can be extended to accommodate any number of device instances and symbols.

At the end of the article, we indicate the current status of the library, our development priorities with a possible roadmap, and where you can follow and contribute to the project.

In the descriptions that follow, we are using the terms MUST and MAY as they are used by the OASIS Standard, which in turn uses them as described in [IETF RFC 2119](https://www.mql5.com/go?link=https://www.rfc-editor.org/info/rfc2119 "https://www.rfc-editor.org/info/rfc2119").

Also, unless otherwise stated, all quotes are from the [OASIS Standard](https://www.mql5.com/go?link=https://docs.oasis-open.org/mqtt/mqtt/v5.0/mqtt-v5.0.html "https://docs.oasis-open.org/mqtt/mqtt/v5.0/mqtt-v5.0.html").

### A Trader's Need

Let’s suppose you, as a trader, specialize in cryptos. You learned that there is a consistent negative correlation between the S&P 500 and an obscure, almost unknown memecoin named AncapCoin. You noted that when the S&P 500 goes up, AncapCoin goes down, and vice-versa. This knowledge gives you an edge on the markets and you have been making money trading AncapCoin following its negative correlation with S&P 500. Now, to maximize your gains, you want to automate your operations, eventually running your Expert Advisor 24/7 in a VPS. All that you need is a basic S&P 500 indicator to support your EA buying and selling decisions.

But AncapBroker specializes in crypto too. So they don’t have S&P 500 among their symbols. And the brokers that offer S&P 500 don’t offer your lovely AncapCoin. How could you have your S&P 500 indicator running in your trading account provided by AncapBroker?

Even though AncapBroker and AncapCoin are just made-up names, this fictional scenario is not fictional at all. Commonly, a trader who operates indices would appreciate having an indicator made of a composite of selected stocks that are not offered by the broker that offers the indices and vice-versa. Consider commodities that are available for trading in a centralized Exchange and their respective CFDs available for trading in CFD brokers and you will have a typical mismatch where the trader has legitimate access to the data but the data come from different providers. For manual, discretionary trading the issue is non-issue. With a separate monitor, or even with a separate window in the same monitor you can follow the missing symbol market. But when it comes to automation, even when both providers offer Metatrader 5, it is very difficult, if not practically impossible for the average retail trader, to have the quotes in real-time in the place where they are needed to make the trading decision: the trading account.

As a developer, if you take this scenario as a customer requirement, then you might as well take the publish/subscribe messaging pattern as a candidate to address this requirement. Among the available open and well-maintained pub/sub specifications, MQTT is probably one of the most simple, cheap, and robust. To be crystal clear here, MQTT was designed precisely to address the kind of requirement outlined above in our fictional scenario, that is, to collect and distribute data from several sources, possibly geographically dispersed, in real-time, and with minimal network overhead.

In the sections below we will see how you can implement this solution in Metatrader 5, using our client in its current state, to the point where you will have the quotes from your **data source account** flowing in realtime to your **trading account**. There it will be available for building that required indicator.

In the **trading account**, we will

1. Create a custom symbol to represent our missing S&P 500
2. Write a SUBSCRIBE MQTT service to receive the S&P 500 quotes from the MQTT broker
3. Update our custom symbol representing the S&P 500 using MQL5 functions

In the **data source account**, we will

1. Write a PUBLISH MQTT service to collect S&P 500 quotes and send them to an MQTT broker

Please, note that to keep things simple and short, we are talking about one trading account and one data source account, but the number of these accounts is theoretically unlimited. In practical terms, the number of accounts connected on both sides is limited only by the physical limits of the devices (memory, CPU, network bandwidth, etc.) and by the limits imposed by your MQTT broker.

Also, please, keep in mind that our goal here is not to provide you with a ready-made production-ready solution. Instead, our goal is to show you the main steps required for one possible implementation and, of course, to drive your interest in the potential uses of MQTT pub/sub pattern for your trading operations or for delivering customized applications for your customers. We are convinced that it goes far beyond copying quotes from a memecoin.

### Create a Custom Symbol

A custom symbol is a symbol you create with your desired or required specifications and update it with the quotes you provide. Because you have control over its specifications and quotes, custom symbols are useful for creating composite indices and for testing Expert Advisors and indicators.

You can create a custom symbol using the Metatrader 5 editor graphical UI, or programmatically via specific MQL5 functions. The documentation has detailed instructions about [how to create, customize, and use custom symbols](https://www.metatrader5.com/en/terminal/help/trading_advanced/custom_instruments "https://www.metatrader5.com/en/terminal/help/trading_advanced/custom_instruments") programmatically or via graphical UI. For our purposes here, the graphical UI is enough.

On the MetaTrader 5 that hosts your **trading account**, click on View > Symbols > create custom symbol.

![MetaTrader 5 - Creating a Custom Symbol Using the Graphical UI](https://c.mql5.com/2/74/create-custom-symbol-sp500-UI.PNG)

Fig. 01 - MetaTrader 5 - Creating a Custom Symbol Using the Graphical UI

In the windows that will pop up next, in the “Copy from:” field, choose the S&P 500 symbol to create a custom symbol based on the real index. Enter a name and a short description for your custom symbol.

![Setting a Custom Symbol Specifications Using the Graphical UI](https://c.mql5.com/2/74/create-custom-symbol-sp500-UI_1.PNG)

Fig. 02 - MetaTrader 5 - Setting a Custom Symbol Specifications Using the Graphical UI

WARNING: The attentive reader may have noted that we are creating our custom symbol in the **trading account**, but according to our example, this account doesn’t have the S&P 500 symbol to be used as a model for our custom symbol. We should be on the **data source account**. After all, what we are doing here is precisely because we don’t have the S&P 500 available in the trading account. And you are right!  In a real situation you will need to fill these symbol specifications according to your needs, probably manually copying the S&P 500 specifications. We are oversimplifying here because creating custom symbols is not our focus here. Instead, we are interested in connecting the accounts via MQTT. When faced with the need to customize the symbol for your real situation, please refer to the documentation linked above.

After clicking on OK, your newly created custom symbol should appear in the tree list at the left.

![Checking the Symbols Tree After a Custom Symbol Creation Using the Graphical UI](https://c.mql5.com/2/74/create-custom-symbol-sp500-UI_2.PNG)

Fig. 03 - MetaTrader 5 - Checking the Symbols Tree After a Custom Symbol Creation Using the Graphical UI

Then add it to your Market Watch so it will be available for chart visualizations. **This step is required for tick updates.**

_“The CustomTicksAdd function only works for custom symbols opened in the Market Watch window. If the symbol is not selected in Market Watch, then you should add ticks using CustomTicksReplace.” ( [MQL5 reference documentation](https://www.mql5.com/en/docs/customsymbols/customticksadd))_

![MetaTrader 5 - Checking the Market Watch Window After a Custom Symbol Creation Using the Graphical UI](https://c.mql5.com/2/74/create-custom-symbol-sp500-UI_3.PNG)

Fig. 04 - MetaTrader 5 - Checking the Market Watch Window After a Custom Symbol Creation Using the Graphical UI

On the Market Watch window, you will note that it has no quotes yet. So, let’s subscribe this account to the MQTT broker that will send real-time quotes to update our newly created MySPX500 custom symbol.

### Write a SUBSCRIBE MQTT Service

At the current stage, our client can subscribe with QoS 0 and QoS 1, but to update quotes/ticks we think QoS 0 is enough because an eventually lost tick is not critical in this context. We are sending one tick each 500ms, so if one or the other gets lost, its position will be immediately occupied by the next.

We start our code for the subscribe service with input parameters for the broker host and port. Remember to replace this with your actual broker data. See in the next section some notes about how to set up your dev/testing environment with a local broker.

```
#property service
//--- input parameters
input string   host = "172.20.106.92";
input int      port = 80;
```

Next, we declare some global vars for our objects from Connect and Subscribe classes. We will need these variables available for deletion (cleaning up) when stopping or returning from error.

```
//--- global vars
int skt;
CConnect *conn;
CSubscribe *sub;
```

_“All objects created by the expression of object\_pointer=new Class\_name must be then deleted by the delete(object\_pointer) operator.”_( [MQL5 reference documentation](https://www.mql5.com/en/docs/customsymbols/customticksadd))

Right on our service OnStart() method, we configure a connection object to be a Clean Start connection - meaning we have no broker session for this connection -, with a generous Keep Alive to avoid the need to send periodic ping request(s) while on development, setting our client identifier, and finally building our CONNECT packet. Be sure to set a different client identifier from that used in your publish service.

```
   uchar conn_pkt[];
   conn = new CConnect(host, port);
   conn.SetCleanStart(true);
   conn.SetKeepAlive(3600);
   conn.SetClientIdentifier("MT5_SUB");
   conn.Build(conn_pkt);
```

In the same OnStart() method we also configure and build our subscribe packet with its Topic Filter. To be clear and intuitive we are using the name we choose for our custom symbol as the Topic Filter. This is arbitrary, provided you use the same Topic Filter in your publish service, of course.

```
  uchar sub_pkt[];
  sub = new CSubscribe();
  sub.SetTopicFilter("MySPX500");
  sub.Build(sub_pkt);
```

Finally, we call the functions to send both packets in sequence, delegating the error handling for those functions and returning -1 if something goes wrong with our subscription request.

```
if(SendConnect(host, port, conn_pkt) == 0)
     {
      Print("Client connected ", host);
     }
   if(!SendSubscribe(sub_pkt))
     {
      return -1;
     }
```

The SendConnect function has only a few lines of MQTT-related code. Most of it is networking/socket-related and we will not be detailing it here. Instead, please refer to the [documentation about Network Functions](https://www.mql5.com/en/docs/network). If you want to go deeper in your understanding of the MQL5 Networking Functions, we strongly recommend the related chapters in the AlgoBook, where you will find detailed explanations and useful examples both for [plain and secure (TLS) networking with MQL5](https://www.mql5.com/en/book/advanced/network).

In the AlgoBook you will find information like the below excerpt, which helped us to identify and do a workaround for an intermittent and non-deterministic behavior in our function (see the commented code on the attached files).

_“Programmers familiar with the Windows/Linux socket system APIs know that a value of 0 can also be a normal state when there is no incoming data in the socket's internal buffer. However, this function behaves differently in MQL5. With an empty system socket buffer, it speculatively returns 1, deferring the actual check for data availability until the next call to one of the read functions. In particular, this situation with a dummy result of 1 byte occurs, as a rule, the first time a function is called on a socket when the receiving internal buffer is still empty.”_( [AlgoBook](https://www.mql5.com/en/book/advanced/network/network_socket_state))

For the MQTT side, all that we do in the SendConnect is to check for the CONNACK response from the broker, as well as for the value of the associated Reason Code.

```
if(rsp[0] >> 4 != CONNACK)
     {
      Print("Not Connect acknowledgment");
      CleanUp();
      return -1;
     }
   if(rsp[3] != MQTT_REASON_CODE_SUCCESS)  // Connect Return code (Connection accepted)
     {
      Print("Connection Refused");
      CleanUp();
      return -1;
     }
```

As you can see, in both we return -1 for error after cleaning up those dynamic pointers to our class objects.

The same proportion of network/MQTT-related code applies to the SendSubscribe function. After checking for the SUBACK response from the broker and its respective Reason Code, we perform the deletion of those class object dynamic pointers if any error occurs.

```
if(((rsp[0] >> 4) & SUBACK) != SUBACK)
     {
      Print("Not Subscribe acknowledgment");
     }
   else
      Print("Subscribed");
   if(rsp[5] > 2)  // Suback Reason Code (Granted QoS 2)
     {
      Print("Subscription Refused with error code %d ", rsp[4]);
      CleanUp();
      return false;
     }
```

In an infinite loop, we wait for broker messages and read them with the help of a Publish class static method.

```
msg += CPublish().ReadMessageRawBytes(inpkt);
               //printf("New quote arrived for MySPX500: %s", msg);
               //UpdateRates(msg);
               printf("New tick arrived for MySPX500: %s", msg);
               UpdateTicks(msg);
```

You will note that we left a commented development code for updating rates instead of ticks. You can uncomment these lines if you want to test this way. Keep in mind that when updating only rates, and not ticks, some information may not be present in the Market Watch window, and in the graph. But it is a reasonable alternative if you want to reduce the RAM, CPU, and bandwidth consumption and have more interest in the data for trading automation instead of in the visuals.

The UpdateRates function works in tandem with its counterpart on the publish service. We are paying the to/from string conversion price on both sides while we develop our MQTT User Property(ies) and have a more reliable binary data exchange. It is the top priority in our quasi-roadmap.

```
void UpdateTicks(string new_ticks)
  {
   string new_ticks_arr[];

   StringSplit(new_ticks, 45, new_ticks_arr);

   MqlTick last_tick[1];

   last_tick[0].time          = StringToTime(new_ticks_arr[0]);
   last_tick[0].bid           = StringToDouble(new_ticks_arr[1]);
   last_tick[0].ask           = StringToDouble(new_ticks_arr[2]);
   last_tick[0].last          = StringToDouble(new_ticks_arr[3]);
   last_tick[0].volume        = StringToInteger(new_ticks_arr[4]);
   last_tick[0].time_msc      = StringToInteger(new_ticks_arr[5]);
   last_tick[0].flags         = (uint)StringToInteger(new_ticks_arr[6]);
   last_tick[0].volume_real   = StringToDouble(new_ticks_arr[7]);

   if(CustomTicksAdd("MySPX500", last_tick) < 1)
     {
      Print("Update ticks failed: ", _LastError);
     }
  }
```

By starting the subscribe service you should see something like this in your log Experts tab.

![MetaTrader 5 - Log Output on the Experts Tab Showing 5270 Error](https://c.mql5.com/2/74/log-output-subscribe-service-started-no-connection.PNG)

Fig. 05 - MetaTrader 5 - Log Output on the Experts Tab Showing 5270 Error

Our subscribe service is talking alone in the desert. Let’s fix this by running our MQTT broker.

### Set Up a Local Broker for Development and Testing

The environment we are using for our development uses the Windows Subsystem For Linux (WSL) on a Windows machine. If all that you want is to run the examples, you can run both the client and the broker on a loopback in the same machine, provided that you use different client identifiers for the publish and subscribe services. But if more than running the examples you want to set up a development environment, we recommend that you set up a separate machine for them. As you probably know, when developing client/server applications - and the pub/sub pattern may be included in this architecture for this matter - it is considered good practice to have each side running on its host. By doing this you can troubleshoot connection, authentication, and other network issues earlier.

This setup with WSL is pretty simple. We detailed the [WSL installation, activation, and configuration](https://www.mql5.com/en/articles/12308) in another article published a year ago. Below are some tips specifically for the use of the Mosquitto broker on the WSL. These are minor details that made our life easier and maybe you may find them useful as well.

- If you activate the WSL with its defaults, and install the Mosquitto the easy and recommended way, you will probably install it using the package manager and run it as a Ubuntu service. That is, Mosquitto will be started automatically when you start the WSL shell. This is good and convenient for regular use, but for development, we recommend that you stop the Mosquitto service and relaunch it manually via the command line with the verbose (-v) flag. This will avoid the need to use the _tail_ command to follow the logs because Mosquitto will be running in the foreground and redirecting all the logs to STDOUT. Besides that, the logs don’t include all the information you will get if launching it with the verbose flag.

![Windows Subsystem For Linux - Stop Mosquitto Server and Relaunch With Verbose Flag](https://c.mql5.com/2/74/wsl-mosquitto-start-verbose.PNG)

Fig. 06 - Windows Subsystem For Linux - Stop Mosquitto Server and Relaunch With Verbose Flag

- Remember that you must include the WSL hostname in the allowed URLs for networking in the Metatrader 5 terminal.

![Windows Subsystem For Linux - Getting the WSL Hostname](https://c.mql5.com/2/74/wsl-hostname.PNG)

Fig. 07 - Windows Subsystem For Linux - Getting the WSL Hostname

![MetaTrader 5 - Include Allowed URLs in the Terminal Options Menu](https://c.mql5.com/2/74/mt5-terminal-options-allowed-urls.PNG)

Fig. 08 - MetaTrader 5 - Include Allowed URLs in the Terminal Options Menu

- As an effort to increase security, the most recent versions of Mosquitto only allow local connections by default, that is, connections from the same machine. To connect from another machine - and your Windows machine is considered another machine in this context - you have to include a one-line listener for port 1883 (or another port of your choice) in your Mosquitto.conf file.

![Windows Subsystem For Linux - Configuring Mosquitto Server to Listen on Port 1883](https://c.mql5.com/2/74/wsl-mosquitto-conf-listener.PNG)

Fig. 09 - Windows Subsystem For Linux - Configuring Mosquitto Server to Listen on Port 1883

- Finally, remember that Mosquitto will be running on port 1883 by default for non-TLS connections. The Metatrader 5 terminal only allows connections to ports 80 (HTTP) and 443 (HTTPS). So you need to redirect the traffic from port 80 to port 1883. This can be done with a short one-line command if you install a Linux utility named _redir_. You can install it using the package manager as well.

![Windows Subsystem For Linux - Performing Port Redirection Using the Redir Utility](https://c.mql5.com/2/74/wsl-redir-mosquitto-port-redirection.PNG)

Fig. 10 - Windows Subsystem For Linux - Performing Port Redirection Using the Redir Utility

- If you forget to include the WSL hostname in the list of allowed URLs or redirect the ports you will end with a refused connection and probably an error like this will pop up in your log Experts Tab

![MetaTrader 5 - Log Output Showing Error 5273](https://c.mql5.com/2/74/error-5273-sub-service.PNG)

Fig. 11 - MetaTrader 5 - Log Output Showing Error 5273 - Failed to send/receive data from socket

Even if you are not familiar with Linux, this setup should not take more than ten to fifteen minutes to be configured, if everything goes well. When it is done you can check if your subscribe service is working as intended.

![MetaTrader 5 Log Output Showing MQTT Subscribe Service Started and Subscribed](https://c.mql5.com/2/74/mt5-log-output-subscribed.PNG)

Fig. 12 - MetaTrader 5 Log Output Showing MQTT Subscribe Service Started and Subscribed

### Write a PUBLISH MQTT Service

The publish service follows the same structure as the subscribe service, so we will be saving you time and not repeating ourselves here. Except to remind you again to use a different client identifier for the publish and subscribe services. (We are emphasizing this client identifier because by ignoring this little detail in the usual lazy developer copy-paste routine we have wasted some time debugging a “non-bug” that caused Mosquitto not to deliver our messages appropriately.)

```
   uchar conn_pkt[];
   conn = new CConnect(host, port);
   conn.SetCleanStart(true);
   conn.SetKeepAlive(3600);
   conn.SetClientIdentifier("MT5_PUB");
   conn.Build(conn_pkt);
```

The publish runs in a continuous loop until it is stopped. Remember to set the same Topic Filter you are using on your publish service, of course, and if you want to update the quotes - instead of the ticks - uncommenting the payload assignment to GetRates (and commenting the assignment to GetLastTick) should be enough.

```
   do
     {
      uchar pub_pkt[];
      pub = new CPublish();
      pub.SetTopicName("MySPX500");
      //string payload = GetRates();
      string payload = GetLastTick();
      pub.SetPayload(payload);
      pub.Build(pub_pkt);
      delete(pub);
     //ArrayPrint(pub_pkt);
      if(!SendPublish(pub_pkt))
        {
         return -1;
         CleanUp();
        }
      ZeroMemory(pub_pkt);
      Sleep(500);
     }
   while(!IsStopped());
```

The remarks we made about the proportion of networking code and MQTT-specific code in the subscribe service also apply here. We don’t even need to check for the PUBACK because we will not receive one, since we are using QoS 0. So, it is only a matter of building the packets, connecting and sending them.

It is worth noting that in the publish service we are paying the string conversion price too, at least until our User Property(ies) are fully implemented, and we can exchange binary data with confidence.

```
string GetLastTick()
  {
   MqlTick last_tick;
   if(SymbolInfoTick("#USSPX500", last_tick))
     {
      string format = "%G-%G-%G-%d-%I64d-%d-%G";
      string out;
      out = TimeToString(last_tick.time, TIME_SECONDS);
      out += "-" + StringFormat(format,
                                last_tick.bid, //double
                                last_tick.ask, //double
                                last_tick.last, //double
                                last_tick.volume, //ulong
                                last_tick.time_msc, //long
                                last_tick.flags, //uint
                                last_tick.volume_real);//double
      Print(last_tick.time,
            ": Bid = ", last_tick.bid,
            " Ask = ", last_tick.ask,
            " Last = ", last_tick.last,
            " Volume = ", last_tick.volume,
            " Time msc = ", last_tick.time_msc,
            " Flags = ", last_tick.flags,
            " Vol Real = ", last_tick.volume_real
           );
      Print(out);
      return out;
     }
   else
      Print("Failed to get rates for #USSPX500");
   return "";
  }
```

By starting the publish service on the MetaTrader 5 terminal you should see something like the following in your log Experts tab.

![MetaTrader 5 Log Output Showing MQTT Publish Service Started and Connected](https://c.mql5.com/2/74/mt5-log-output-publish-started.PNG)

Fig. 13 - MetaTrader 5 Log Output Showing MQTT Publish Service Started and Connected

You can subscribe to the topic on the Mosquitto broker and check the broker's verbose output.

![Windows Subsystem For Linux Showing Mosquitto Server Logging with Verbose Flag](https://c.mql5.com/2/74/wsl-mosquitto-verbose-log-publish.PNG)

Fig. 14 - Windows Subsystem For Linux Showing Mosquitto Server Logging with Verbose Flag

If the message was successfully received you should see it on your subscription tab.

![Windows Subsystem For Linux Showing mosquitto_sub Utility Output](https://c.mql5.com/2/74/wsl-mosquitto_sub-output.PNG)

Fig. 15 - Windows Subsystem For Linux Showing mosquitto\_sub Utility Output

### Update the Custom Symbol

With both services in place and with each of them checked against your broker, it is time to run them in the Metatrader 5 terminal and see the results of your hard job.

![MetaTrader 5 Navigator With MQTT Publish and Subscribe Services Started](https://c.mql5.com/2/74/mt5-navigator-mqtt-services-running.PNG)

Fig. 16 - MetaTrader 5 Navigator With MQTT Publish and Subscribe Services Started

If everything goes well you should see something like the following on the ticks tab of your Market Watch windows.

![ MetaTrader 5 Market Watch Ticks Tab With Custom Symbol Tick Updates](https://c.mql5.com/2/74/mqtt-MarketWatch-MySPX500-tick-updates.gif)

Fig. 17 - MetaTrader 5 Market Watch Ticks Tab With Custom Symbol Tick Updates

On your custom symbol graph, the tick updates should be reflected too.

![MetaTrader 5 Market Graph with Custom Symbol Tick Updates](https://c.mql5.com/2/74/mqtt-graph-MySPX500-tick-updates.gif)

Fig. 18 - MetaTrader 5 Market Graph with Custom Symbol Tick Updates

On the log Experts tab you should find a bit verbose output. That is because we left some debug info for now. There you can see, among other information, the output of the PrintArray function for the packets being exchanged. This output may avoid the need to apply a packet analyzer, like Wireshark, to check the content of the packets.

![MetaTrader 5 Log Output on Expert Tab With MQTT Services Logging For Development and Debug](https://c.mql5.com/2/74/mqtt-log-output-MySPX500-tick-updates.gif)

Fig. 19 - MetaTrader 5 Log Output on Expert Tab With MQTT Services Logging For Development and Debug

### Conclusion

This article presents a functional code for sharing real-time quotes among brokers and accounts via MQTT. While the demo runs on the same Metatrader 5 instance and machine, it showcases MQTT's flexibility and robustness. Our native MQTT client is still in development during our limited free time. Our aim is a fully compliant version by June, addressing challenges like QoS\_2 implementation and User Properties. We plan to refine the code and make it public on GitHub by April's end.

We welcome contributors to our open-source project, regardless of MQL5 expertise. Our Test-Driven Development approach ensures error-free progress, even for non-experts. Starting from basic tests, we've steadily progressed through MQL5 documentation. We're committed to refining our client until it meets all MQTT standards.

Join us, even with basic skills. Your input is valuable. Welcome aboard!

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/14677.zip "Download all attachments in the single ZIP archive")

[mqtt-headers.zip](https://www.mql5.com/en/articles/download/14677/mqtt-headers.zip "Download mqtt-headers.zip")(23.78 KB)

[mqtt-services.zip](https://www.mql5.com/en/articles/download/14677/mqtt-services.zip "Download mqtt-services.zip")(3.51 KB)

[mqtt-tests.zip](https://www.mql5.com/en/articles/download/14677/mqtt-tests.zip "Download mqtt-tests.zip")(19.52 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Statistical Arbitrage Through Cointegrated Stocks (Part 10): Detecting Structural Breaks](https://www.mql5.com/en/articles/20946)
- [Statistical Arbitrage Through Cointegrated Stocks (Part 9): Backtesting Portfolio Weights Updates](https://www.mql5.com/en/articles/20657)
- [Statistical Arbitrage Through Cointegrated Stocks (Part 8): Rolling Windows Eigenvector Comparison for Portfolio Rebalancing](https://www.mql5.com/en/articles/20485)
- [Statistical Arbitrage Through Cointegrated Stocks (Part 7): Scoring System 2](https://www.mql5.com/en/articles/20173)
- [Statistical Arbitrage Through Cointegrated Stocks (Part 6): Scoring System](https://www.mql5.com/en/articles/20026)
- [Statistical Arbitrage Through Cointegrated Stocks (Part 5): Screening](https://www.mql5.com/en/articles/19626)
- [Statistical Arbitrage Through Cointegrated Stocks (Part 4): Real-time Model Updating](https://www.mql5.com/en/articles/19428)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/465429)**
(2)


![Реter Konow](https://c.mql5.com/avatar/avatar_na2.png)

**[Реter Konow](https://www.mql5.com/en/users/peterkonow)**
\|
27 Aug 2024 at 09:28

Very good and interesting article. I learnt a lot of new and useful things. Frankly speaking, this does not happen often, but this is the case. I will study your other articles carefully.


![Jocimar Lopes](https://c.mql5.com/avatar/2023/2/63de1090-f297.jpg)

**[Jocimar Lopes](https://www.mql5.com/en/users/jslopes)**
\|
7 Jan 2025 at 04:35

**Реter Konow [#](https://www.mql5.com/en/forum/465429#comment_54409336):**

Very good and interesting article. I learnt a lot of new and useful things. Frankly speaking, this does not happen often, but this is the case. I will study your other articles carefully.

Hey, Peter! I'm glad you found some useful information on the article.

By the way, the code is on [GitHub](https://www.mql5.com/go?link=https://github.com/gavranha/mql5-mqtt-cli/ "https://github.com/gavranha/mql5-mqtt-cli/"), freely available for use, study, and development.

From the README:

"Update on January 6th, 2025

As the saying goes, the best open-source code starts by itching your own itch. This was the case here.

But it turns out that we've eventually found a better solution for our itch, so we are no longer working on this code.

If you think it can be useful as a starting point — or want to learn from our mistakes — just fork it and use it at will."

Have a great new year!

![Quantitative analysis in MQL5: Implementing a promising algorithm](https://c.mql5.com/2/62/Quantitative_analysis_in_MQL5_-__implementing_a_promising_algorithm__LOGO.png)[Quantitative analysis in MQL5: Implementing a promising algorithm](https://www.mql5.com/en/articles/13835)

We will analyze the question of what quantitative analysis is and how it is used by major players. We will create one of the quantitative analysis algorithms in the MQL5 language.

![MQL5 Wizard Techniques You Should Know (Part 15): Support Vector Machines with Newton's Polynomial](https://c.mql5.com/2/75/MQL5_Wizard_Techniques_You_Should_Know_1Part_15c____LOGO.png)[MQL5 Wizard Techniques You Should Know (Part 15): Support Vector Machines with Newton's Polynomial](https://www.mql5.com/en/articles/14681)

Support Vector Machines classify data based on predefined classes by exploring the effects of increasing its dimensionality. It is a supervised learning method that is fairly complex given its potential to deal with multi-dimensioned data. For this article we consider how it’s very basic implementation of 2-dimensioned data can be done more efficiently with Newton’s Polynomial when classifying price-action.

![Population optimization algorithms: Simulated Annealing (SA) algorithm. Part I](https://c.mql5.com/2/62/Population_optimization_algorithms_Simulated_Annealing_algorithm_LOGO.png)[Population optimization algorithms: Simulated Annealing (SA) algorithm. Part I](https://www.mql5.com/en/articles/13851)

The Simulated Annealing algorithm is a metaheuristic inspired by the metal annealing process. In the article, we will conduct a thorough analysis of the algorithm and debunk a number of common beliefs and myths surrounding this widely known optimization method. The second part of the article will consider the custom Simulated Isotropic Annealing (SIA) algorithm.

![Neural networks made easy (Part 67): Using past experience to solve new tasks](https://c.mql5.com/2/62/Neural_networks_made_easy_Part_67__LOGO.png)[Neural networks made easy (Part 67): Using past experience to solve new tasks](https://www.mql5.com/en/articles/13854)

In this article, we continue discussing methods for collecting data into a training set. Obviously, the learning process requires constant interaction with the environment. However, situations can be different.

[![](https://www.mql5.com/ff/sh/20jc81m23z78s5z9z2/01.png)![](https://www.mql5.com/ff/sh/20jc81m23z78s5z9z2/02.png)Create your own AI for tradingRead our book "Neural Networks in Algo Trading with MQL5"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/neurobook%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.neurobook%26utm_content=visit.page%26utm_campaign=neurobook.promo.04.2024&a=elbyupbppbqpzzvzhxtydvlupfcbmnmb&s=0d2f8feb92df3772a11aca1f195d2996b59d6539e283cdf4a18ccff02e5ad43d&uid=&ref=https://www.mql5.com/en/articles/14677&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5068136142060058085)

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