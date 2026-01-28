---
title: General information on Trading Signals for MetaTrader 4 and MetaTrader 5
url: https://www.mql5.com/en/articles/618
categories: Trading
relevance_score: 4
scraped_at: 2026-01-23T17:40:19.946206
---

[![](https://www.mql5.com/ff/si/3fgkjn78mkxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=ocndbzpeklfncxysjbwfhhbalbrsdbtv&s=a4309643278437a00bdd33c5809fc6b4b4032749c00fccd07b3b84e7b8b45126&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=pnpwkbhueuyxgpelfcybfhmkqlshtxcp&ssn=1769179219451456256&ssn_dr=0&ssn_sr=0&fv_date=1769179219&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F618&back_ref=https%3A%2F%2Fwww.google.com%2F&title=General%20information%20on%20Trading%20Signals%20for%20MetaTrader%204%20and%20MetaTrader%205%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176917921907937181&fz_uniq=5068510061912848978&sv=2552)

MetaTrader 5 / Trading


MetaTrader 4 / MetaTrader 5 Trading Signals is a service allowing traders to copy trading operations of a Signals Provider.

Some traders do not have enough time for active trading, others do not possess enough self-confidence or knowledge to work in the market. Trading signals make traders' money work rather than merely collecting dust on the accounts.

### What Is So Special About Our Service?

Our goal was to develop the new massively used service protecting Subscribers and relieving them of unnecessary costs:

01. Full focus on the protection of Subscribers.
02. Very simple process of buying and selling trading signals.
03. Advanced and secure MQL5.community [payment system](https://www.mql5.com/en/articles/302) (PayPal, WebMoney, bank cards).
04. Full transparency of trading history.
05. Very reasonable prices, as there is only a fixed subscription fee and the ability to return the current month subscription fee in case of a disagreement with a Signals Provider.
06. No conflict of interest between our company and other participants. Our interest is limited by 20% of commission for subscription.
07. No need to sign paper contracts and arrange complex interactions between brokers and Signals Providers.
08. No commissions for deals, no increasing spreads and, as a result, no attempts by Signals Providers to obtain more profit performing frequent deals.
09. No personal data on Subscribers is collected, there is no access to their accounts and balances, as the password is not required.
10. Signals Providers know nothing about their Subscribers, except their amount.
11. Work with any MetaTrader-broker, including regulated-ones.
12. Each deal performed via the service is reliable, as it is provided with a unique digital signature when passing the execution queue. That protects against possible fraud and modifications.

### How It Works?

To implement the MetaTrader 4 / MetaTrader 5 Trading Signals service, we have developed a resilient cloud network consisting of multiple Signal Servers all over the world located near broker servers to reduce network latency.

[MQL5.com](https://www.mql5.com/en/signals) web site contains publicly available list of MetaTrader 4 and MetaTrader 5 signals, which is automatically updated in the client terminals.

![The list of MetaTrader 4 and MetaTrader 5 Trading Signals on MQL5.com](https://c.mql5.com/2/17/618_1.png)

![The list Trading Signals in MetaTrader 5](https://c.mql5.com/2/17/618_2.png)

Traders need only two things to subscribe to a signal:

1. **MQL5.com account**. There is no need to specify a real name. A nickname is sufficient here.
2. **Trade account.** You don't have to specify its number and password.

These are all necessary things required to subscribe to the Provider's signal. Select a signal, click "Copy Trades", and you'll be offered to open the MetaTrader platform to proceed with subscription.

![Subscribe for MetaTrader 4 or MetaTrader 5 Trading Signal](https://c.mql5.com/2/17/618_3.png)

Signals Providers need to provide more personal data and pass a probationary period. Generating signals for other traders imposes some responsibility on Providers. To ensure Subscriber's protection, Signals Providers should specify the following personal details: first and last names, address, contact phones and scanned copy of a passport or a similar document. This information may be necessary in case of any disagreements between a Subscriber and a Provider. This information will not be disclosed to any third parties.

When publishing a new signal, Providers should specify:

1. **Signal name**
2. **Trading terminal type - MetaTrader 4 or MetaTrader 5**
3. **Broker's trading server name**
4. **Number of the account that will transfer the signals**
5. **Investor password.** This password allows to connect to a trader's account in READ ONLY mode and view the current trading operations, as well as their history. This password is enough for "scanning" the current trading operations and their further distribution among Subscribers. At the same time, this password does not allow managing a trading account and performing trade operations. If a Master password is specified, the signal is not enabled.
6. **Subscription price**

If a signal is not free, the Provider should pass moderation and register as a Seller providing actual personal details.

![Register as a Signal Provider for MetaTrader 4 and MetaTrader 5](https://c.mql5.com/2/17/618_4.png)

Once all stages have been accomplished, the signal can be transferred by the Provider and accepted by Subscribers:

1. On the Provider's side, a trader or an Expert Advisor performs trading operations in his/her own terminal.
2. This information is sent to the broker's trading server where the Signals Provider is served.
3. MetaTrader Signal Server connects to the Provider's broker using specified Investor password and receives all trading operations in real time.
4. The Subscriber's terminal is constantly connected to the necessary MetaTrader Signal Server. Therefore, it receives all trading operations instantly.
5. The Subscriber's terminal performs a trading operation on its broker's server.


![How Trading Signals for MetaTrader 4 and MetaTrader 5 Works?](https://c.mql5.com/2/5/MetaTrader-Signal-Scheme__1.jpg)

Please note that Signals Providers and Subscribers operate without the knowledge of the broker negotiating directly between each other. Both participants can be served by completely different brokers.

**Note:** MetaQuotes Software Corp. does not interfere with relations between Subscribers and Signals Providers. Our company only provides the necessary infrastructure for arranging the interaction between all the participants.

### Initial Synchronization of Trading Signals

The client terminal protects traders against obvious errors to the maximum possible extent.

Suppose that we have an account subscribed to [a signal](https://www.mql5.com/en/signals/839). If all trading signals are allowed in the terminal, the trading account will be synchronized with the Provider's one during authorization. It's not recommended to have on your trade account positions and orders that are not based on the provider's signals. They increase the overall load on the account as compared with the signal provider.

![The confirmation of Trading Signal Initial Synchronization](https://c.mql5.com/2/17/618_5.png)

It is critically important to synchronize during the right market conditions to ensure the security of the Subscriber's account. Automatic synchronization works only in case the total floating profit of a Signals Source is negative or equals to zero. Therefore, it is guaranteed that Subscribers will enter the market at the price, which is not worse than the one, at which the Signals Source entered the market. This is an important psychological component of how traders evaluate the quality of copying a signal.

If a profit on the Provider's account is positive, the appropriate window will appear explaining the situation and offering to wait for better market conditions. Traders may accept the risk and synchronize immediately.

![The confirmation of Trading Signal Initial Synchronization](https://c.mql5.com/2/17/618_6.png)

Information about consent to use signals, as well as about forced synchronization will necessarily be fixed in the terminal's journal. Besides, each deal performed via the Signals service has special "signal" reason type allowing to easily identify such operations. All this has been done to protect subscribers and providers allowing them to manage any possible disagreements with greater accuracy.

Let's examine two examples of initial synchronization:

1. A Signals Provider opened a long position hoping to gain 100 points of profit. However, the price has gone down by 20 points at the moment.



![Trading Signals: operation will be mirrored as the provider has worse conditions](https://c.mql5.com/2/5/provider_trade_scheme1__1.png)




    That means that the Provider believes that the price will soon change its direction and the targeted profit will be received. In this case, the appropriate position will be opened in Subscriber's terminal and the Subscriber will receive 120 points of profit instead of 100 ones if the price actually changes its direction. If the Provider closes the position fixing the loss, the Subscriber's one will also be closed with a smaller loss. As a result, the quality of signals copying will always be better than 100% and Subscribers will be pleased since they have managed to enter the market at better prices.

2. Let's consider a different course of events. The price has moved upwards by 40 points and the Signals Provider has some profit already.



![Trading Signals: operation will not be copied as the provider has floating profit](https://c.mql5.com/2/5/provider_trade_scheme2__1.png)




    In this case, the appropriate position will not be opened at the Subscriber's terminal automatically, as he or she may receive a smaller profit or even a loss. The Subscriber may receive 60 points of profit, while the Provider will receive 100. The Subscriber may even suffer losses if the Provider will close the position having 30 points of profit, while the Subscriber will have 10 points of loss. In any case, the Subscriber will be disappointed.





    Unfortunately, some people do not consider trader's psychology and do not pay attention to the evaluation of results that may take the following forms: "I have gained smaller profit - the signals execution is poor" or "I have suffered losses, while the Provider has still gained some profit - the execution is completely bad". Rational arguments and mathematical proof cannot beat psychology. Therefore, we try to protect traders against errors at the initial stage.

In case of connection loss, order placing error, terminal shutdown etc., the account will be re-synchronized with the Signal Source. In this case, the entire Subscriber's and Provider's sets of orders will be checked. Deals closed by the Provider are also closed at the Subscriber's side, while new Provider's deals will be also opened at the Subscriber's side at the price, "which is not worse than the Provider's one".

### Managing Funds or How to Select a Deal Volume?

The question of how exactly the Subscriber's deposit will participate in trading via Signals service is one of the most critical ones. When solving this issue, we followed the already mentioned principle - providing maximum protection for each participant. As a result, we can offer a secure solution for Subscribers.

When enabling signals in the terminal and subscribing to one of them, Subscribers should select what part of the deposit is to be used when following the signals. There was an alternative solution of setting the ratio between Subscriber's and Provider's position volumes. But such a system could not guarantee the security of the Subscriber's deposit. For example, suppose that Provider's deposit is 30 000$, while Subscriber's one is 10 000$ and the ratio of 1:1 has been selected. In that case, the Signals Provider may just wait out temporary drawdown having a large volume order, while the Subscriber may lose all the funds with all his or her positions closed by Stop Out. The situation may get even worse if the Provider's balance suddenly changes (top up or withdraw), while previously specified volumes ratio remains intact.

To avoid such cases, we have decided to implement the system of percentage-based allocation of the part of a deposit, which is to be used in trading via the Signals service. This system is quite complicated as it considers deposit currencies, their conversion and leverages.

Let's consider a specific example of using the volumes management system:

1. Provider: balance 15 000 USD, leverage 1:100
2. Subscriber1: balance 40 000 EUR, leverage 1:200, deposit load percentage 50%

3. Subscriber2: balance 5 000 EUR, leverage 1:50, deposit load percentage 35%
4. EURUSD exchange rate = 1.2700


Calculation of Provider's and Subscriber's position volumes ratio:

1. Balances ratio considering specified part of the deposit in percentage terms:

Subscriber1: (40 000 \* 0,5) / 15 000 = 1,3333 (133.33%)

Subscriber2: (5 000 \* 0,35) / 15 000 = 0,1166 (11.66%)

2. After considering the leverages:

Subscriber1: the leverage of Subscriber1 (1:200) is greater than Provider's one (1:100), thus correction on leverages is not performed

Subscriber2: 0,1166 \\* (50 / 100) = 0,0583 (5.83%)
3. After considering currency rates of the deposits at the moment of calculation:

Subscriber1: 1,3333 \\* 1,2700 = 1,6933 (169.33%)

Subscriber2: 0,0583 \\* 1,2700 = 0,0741 (7.41%)
4. Total percentage value after the rounding (performed using a multistep algorithm):

Subscriber1: 160% or 1.6 ratio

Subscriber2: 7% or 0.07 ratio

Thus under the given conditions, Provider's deal with volume of 1 lot will be copied:

\- to Subscriber1 account in amount of 160% -  volume of 1.6 lots

\- to Subscriber2 account in amount of 7% -  volume of 0.07 lots

Be careful not to confuse the percentage value of the used part of the
deposit and the actual ratio of position volumes. The trading terminal
allows setting the part of the deposit in percentage value. This value
is used to calculate the ratio of position volumes. This data is always
fixed in the log and is shown in the following way:

Subscriber1:

2012.11.12 13:33:23    Signal    '1277190': percentage for volume
conversion selected according to the ratio of balances and leverages,
new value 160%

2012.11.12 13:27:55    Signal    '1277190': signal provider has
balance 15 000.00 USD, leverage 1:100; subscriber has balance 40 000.00
EUR, leverage 1:200

2012.11.12 13:27:54    Signal    '1277190': money management: use 50%
of deposit, equity limit: 0.00 EUR, deviation/slippage: 1.0 spreads

Subscriber2:

2012.11.12 13:33:23    Signal    '1277191': percentage for volume
conversion selected according to the ratio of balances and leverages,
new value 7%

2012.11.12 13:27:55    Signal    '1277191': signal provider has
balance 15 000.00 USD, leverage 1:50; subscriber has balance 5 000.00
EUR, leverage 1:50

2012.11.12 13:27:54    Signal    '1277191': money management: use 35%
of deposit, equity limit: 0.00 EUR, deviation/slippage: 1.0 spreads

### Features of MetaTrader 4 / MetaTrader 5 Trading Signals

Operation of trading signals on our platform has a number of features. Both subscribers and providers should be well aware of them.

A single trading account can be managed by the signals of only one Provider at a time. We have deliberately banned using several signals on a single account to protect traders from unexpected losses.

It should also be considered that a trading account can be subscribed to only one signal, whilst MQL5.com account may have several subscriptions for different accounts. This means that a trader may have several accounts opened at different brokerage companies and managed by different signals. All these subscriptions may be registered and paid up from a single MQL5.com-account.

As for the payment for trading signals, it is quite simple and transparent. Subscribers pay a fixed sum monthly or weekly and receive trading signals for this period. There are neither commissions for each deal, nor increasing spreads, nor additional commissions from gained profits.

The MQL5.com payment system is used for buying subscriptions via PayPal, WebMoney or bank cards. When buying a subscription to a signal, the funds are transferred from the Subscriber's account to the Provider's account (our commission is 20%) where they are automatically blocked for the subscription period. At the end of the period the funds are unblocked and deposited to the Provider's account. In case of any valid claims, the funds for the current period will be returned to the Subscriber.

One of the main features is that Providers and Subscribers do not need to have their accounts on the same trade server. Delays between executions of trading operations on the Provider's and Subscriber's accounts are minimized. This has been made possible thanks to new Signal Servers having cloud architecture and located all over the world. At the same time, the highest quality of execution will be achieved if a Provider and a Subscriber work on the same server.

[Subscribe for a trading signals](https://www.mql5.com/en/signals "Subscribe for a trading signals in MetaTrader 4 and MetaTrader 5")           OR           [Become a trading signals provider](https://www.mql5.com/en/articles/591 "Become a trading signals provider for  MetaTrader 4 and MetaTrader 5")

We also recommend following articles dedicated to signals:

- [How to Become a Signals Provider for MetaTrader 4 and MetaTrader 5](https://www.mql5.com/en/articles/591)
- [How to Subscribe to Trading Signals](https://www.mql5.com/en/articles/523)
- [How to Prepare a Trading Account for Migration to Virtual Hosting](https://www.mql5.com/en/articles/994)

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/618](https://www.mql5.com/ru/articles/618)

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
**[Go to discussion](https://www.mql5.com/en/forum/9963)**
(361)


![Juvenille Emperor Limited](https://c.mql5.com/avatar/2019/4/5CB0FE21-E283.jpg)

**[Eleni Anna Branou](https://www.mql5.com/en/users/eleanna74)**
\|
24 Feb 2023 at 12:04

**Sorsys [#](https://www.mql5.com/en/forum/9963/page10#comment_45209683):**

Max open lots should not exceed \_\_\_\_% of your balance.

Pls fill in the blank.

There is no definite answer to that question, it depends on your trading style, the undertaken risk and many other factors.

Some say that the ideal trade size is 0.01 per $1000 per trade, others say 0.01 per $3000, some others may open 0.10 per $1000.

It's all a matter or risk management really.

Also it depends on the number of trades that will be open simultaneously, some strategies have only 1 trade open at a time, some other grid strategies may have dozens.

Personally I feel comfortable with anything around 0.01-0.03 per $1000/trade and a deposit load under 50%, but this is all very objective.

![Luis Maria Baptista Pina Soares](https://c.mql5.com/avatar/2025/1/679b7c77-4f3f.jpg)

**[Luis Maria Baptista Pina Soares](https://www.mql5.com/en/users/lmbpsoares)**
\|
5 Feb 2025 at 19:17

Hello,

I have a doubt would like to change the percentage of my deposit to be used, where can I access to reconfigure this option of the signal I subscribed?

Best regards

![Juvenille Emperor Limited](https://c.mql5.com/avatar/2019/4/5CB0FE21-E283.jpg)

**[Eleni Anna Branou](https://www.mql5.com/en/users/eleanna74)**
\|
5 Feb 2025 at 19:42

**Luis Maria Baptista Pina Soares [#](https://www.mql5.com/en/forum/9963/page10#comment_55834657):**

Hello,

I have a doubt would like to change the percentage of my deposit to be used, where can I access to reconfigure this option of the signal I subscribed?

Best regards

You need to go to MT4/5 >> Tools >> Options >> Signals >> Use no more than: ... of deposit.

If you are using MQL5 VPS, you need to click the: Enable realtime signal subscritpion option and synchronize your new signal subscription settings to your MQL5 VPS again (right click in your MQL5 VPS >> Synchronize Signal only.

![Fernando Carreiro](https://c.mql5.com/avatar/2025/9/68d40cf8-38fb.png)

**[Fernando Carreiro](https://www.mql5.com/en/users/fmic)**
\|
5 Feb 2025 at 19:46

**[@Luis Maria Baptista Pina Soares](https://www.mql5.com/en/users/lmbpsoares) [#](https://www.mql5.com/en/forum/9963/page10#comment_55834657):** I have a doubt would like to change the percentage of my deposit to be used, where can I access to reconfigure this option of the signal I subscribed?

Read the following from the documentation ... [How to Subscribe to a Signal - Trading Signals and Copy Trading - MetaTrader 5 Help](https://www.metatrader5.com/en/terminal/help/signals/signal_subscriber#settings "Click to change text")

> ![](https://c.mql5.com/3/455/4595365521948.png)

![Diego Heras Garcia](https://c.mql5.com/avatar/avatar_na2.png)

**[Diego Heras Garcia](https://www.mql5.com/en/users/piimpam)**
\|
11 Dec 2025 at 11:04

I have published a signal that trades on Roboforex under the symbol .USTECHCash and a subscriber to the signal that tries to copy trades from ICMarkets. In the subscriber's broker it has the same symbol but with the name USTEC. Is there any way to modify the token in the copy or any way to map the tokens? Or the obvious question, is it possible to copy trades from one broker to another broker with the same symbols but [slightly](https://www.mql5.com/en/docs/constants/objectconstants/webcolors "MQL5 Documentation: Web Colour Set ") different names? Thank you very much.


![MetaTrader 5 on Linux](https://c.mql5.com/2/0/linux5.png)[MetaTrader 5 on Linux](https://www.mql5.com/en/articles/625)

In this article, we demonstrate an easy way to install MetaTrader 5 on popular Linux versions — Ubuntu and Debian. These systems are widely used on server hardware as well as on traders’ personal computers.

![Interview with Alexey Masterov (ATC 2012)](https://c.mql5.com/2/0/avatar_reinhardf17.png)[Interview with Alexey Masterov (ATC 2012)](https://www.mql5.com/en/articles/624)

We do our best to introduce all the leading Championship Participants to our audience in reasonable time. To achieve that, we closely monitor the most promising contestants in our TOP-10 and arrange interviews with them. However, the sharp rise of Alexey Masterov (reinhard) up to the third place has become a real surprise!

![Neural Networks: From Theory to Practice](https://c.mql5.com/2/0/ava_seti.png)[Neural Networks: From Theory to Practice](https://www.mql5.com/en/articles/497)

Nowadays, every trader must have heard of neural networks and knows how cool it is to use them. The majority believes that those who can deal with neural networks are some kind of superhuman. In this article, I will try to explain to you the neural network architecture, describe its applications and show examples of practical use.

![How to Prepare MetaTrader 5 Quotes for Other Applications](https://c.mql5.com/2/0/ava__1.png)[How to Prepare MetaTrader 5 Quotes for Other Applications](https://www.mql5.com/en/articles/502)

The article describes the examples of creating directories, copying data, filing, working with the symbols in Market Watch or the common list, as well as the examples of handling errors, etc. All these elements can eventually be gathered in a single script for filing the data in a user-defined format.

[![](https://www.mql5.com/ff/si/3p2yc19r7qvs297n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F618%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dsignal.advantage%26utm_content%3Dsubscribe.signal%26utm_campaign%3D0622.MQL5.com.Internal&a=bewozmaxwejekdopjicjtsbzmjgfjyvt&s=e49ac7e84b713650e3af82ec3c6b4d02fdf06617c5821011b1e499af5edd01f4&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=qqdoeylllobzkgccrrdkjraylovckbyd&ssn=1769179219451456256&ssn_dr=0&ssn_sr=0&fv_date=1769179219&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F618&back_ref=https%3A%2F%2Fwww.google.com%2F&title=General%20information%20on%20Trading%20Signals%20for%20MetaTrader%204%20and%20MetaTrader%205%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176917921907956530&fz_uniq=5068510061912848978&sv=2552)

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