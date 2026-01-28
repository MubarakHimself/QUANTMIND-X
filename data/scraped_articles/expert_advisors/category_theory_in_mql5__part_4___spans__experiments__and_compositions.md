---
title: Category Theory in MQL5 (Part 4): Spans, Experiments, and Compositions
url: https://www.mql5.com/en/articles/12394
categories: Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T21:26:42.751594
---

[Launching MetaTrader VPS for the first time?Read our comprehensive, step-by-step instructions![](https://www.mql5.com/ff/sh/0xb0c8bjq5sadh89z2/01.png)Read](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/articles/13586&a=gxygkojxdwrcfbbgfrchvjgelflsnelu&s=49eab2fb45d89f59a191e88145774dcd7f9533039acb10dd9c28061b04fa92fe&uid=&ref=https://www.mql5.com/en/articles/12394&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5071837154033741665)

MetaTrader 5 / Examples


### Introduction

In the previous article, we saw how category theory can be potent in complex systems via its concepts of products, coproducts, and the universal property, examples of applications of which in finance and algorithmic trading were shared. Here, we will delve deeper into spans, experiments, and compositions. We will see how these concepts provide a more nuanced and flexible way of reasoning about systems, and how they can be used to develop more sophisticated trading strategies. By understanding the underlying structure of financial markets in terms of category theory, traders can gain new insights into the behavior of financial instruments, construct more sophisticated portfolios, and develop more effective risk management strategies. Overall, the application of category theory in finance has the potential to revolutionize the way we think about financial markets and to enable traders to make more informed decisions.

### Spans, Experiments, and Compositions

In category theory, a [span](https://en.wikipedia.org/wiki/Span_(category_theory) "https://en.wikipedia.org/wiki/Span_(category_theory)") is a construction that relates three objects and two morphisms between them. Specifically, a span is a diagram of the form:

![](https://c.mql5.com/2/52/2279743275832.png)

This very basic diagram that can also be represented with the single line below:

A<--- f --- P --- g --->B

where A, B, and P, are domains in a category, and f: P to A, and g: P to B, are morphisms in the category. The morphisms f: P to A, and g: P to B, are called the legs of the span.

The span, P, can be thought of as a way of relating two different paths or perspectives between A and B, one via f and the other via g. The legs f and g connect these paths at A and B respectively, and allow for comparison and composition of the two paths.

Let's start by going over the theory first, before diving into possible application within MQL5.

Spans are important in category theory because they provide a way to compare two different morphisms in a category. Given two morphisms f: A → B and g: A → C, a span from B to C is a diagram of the form B ← A → C, where the two arrows represent the morphisms f and g. Spans are often used to define limits and colimits in category theory. For example, a limit of a diagram in a category can be defined as a universal cone over that diagram, where a cone is a span from the limit object to each object in the diagram that satisfies certain conditions.

Spans are also useful in defining pullbacks, which are a type of limit in which the objects involved are related by a pair of morphisms. Given two morphisms f: A → B and g: A → C in a category, a pullback of f and g is an object P together with two morphisms p1: P → B and p2: P → C such that f ∘ p1 = g ∘ p2 and P is universal with respect to this property. Pullbacks are important in many areas of mathematics and science, including algebraic geometry, topology, and computer science.

Another important concept in category theory is the experiment, which is a diagram consisting of two parallel morphisms and a third morphism connecting their codomains. An experiment can be thought of as a way of comparing two different ways of transforming an object in a category. For example, given two morphisms f: A → B and g: A → C, an experiment from B to C is a diagram of the form A → B ⟶ D ← C, where the arrows represent the morphisms f, g, and h, respectively. Experiments can be used to define limits and colimits in a similar way to spans, and they are also useful for defining coequalizers, which are a type of colimit that can be used to identify two different morphisms that have the same codomain.

Composites are a fundamental concept in category theory that arises from the composition of two or more morphisms. Given two morphisms f: A → B and g: B → C, their composite is a morphism g ∘ f: A → C, which is obtained by applying f followed by g. Composites are associative, meaning that (h ∘ g) ∘ f = h ∘ (g ∘ f) for any three morphisms f, g, and h. This property allows for the composition of many morphisms at once, and it is used to define the notion of a category, which is a collection of objects and morphisms that satisfy certain axioms.

Here are ten impromptu applications of spans, experiments, and composites in category theory in finance and trading:

01. Spans could model the underlying assets of a financial derivative and the hedging instruments used to replicate its payoffs. Universal-property of the span would help define the derivative’s price.
02. Efficient portfolio construction by composites through combining different asset classes in a way that minimizes risk and maximizes return. This can be guided by using the universal-property of a composition.
03. Spans could model the risk exposure of a financial institution to different market factors. This can be achieved by constructing a span that connects the institution's assets to the relevant market indices.
04. Experiments could test the performance of different trading algorithms under various market conditions. This can be simulated by constructing an experiment that mimics market behavior and measuring the performance of the algorithm on the simulated data.
05. Modelling the behavior of financial systems at a zoomed-out scale is something compositions of domains could help with. For instance looking at how various sectors of SP500 correlate in different longer cycles could be an application.
06. Spans can model the replication of a particular financial instrument using a combination of simpler instruments. This is useful in designing new financial instruments that have desirable properties like less correlations to existing ones.
07. Experiments could test trading strategies by comparing the performance of a particular strategy against a control group. This could be accomplished once again using the universal-property of the experiment.
08. Experiments can also test the efficiency of different market microstructure designs. By constructing an experiment that simulates the behavior of different types of market participants and measuring the resulting market outcomes this could be done.
09. Composites could model the overall risk exposure of a financial institution. This can be done by constructing a composite of the institution's various business lines and analyzing their interdependence.
10. Spans could model the relationship between different financial data sources. This could design machine learning algorithms that can extract useful features from disparate data sources.

To illustrate this concept’s application, we can describe a span ‘experiment’ that involves finding whether there is a relationship between the profit on a security’s _long_ open trade position and a pair of other variables namely the security's moving average and the security's average true range. This ‘experiment’ can be represented as a diagram that involves an apex, two domains, and morphisms between them.

Let’s say we have a category C that contains two domains, A and B, that represent the domains of the moving average and the domain of the average true range, respectively. And in this category, we also have a domain P that represents the long position profit and acts as our apex.

We can then go on to define two morphisms f: P -> A and g: P -> B that map the long position float P to its observables in domains A and B, respectively. These morphisms represent the logging of the observables.

This diagrammatic representation allows us to analyse the experiment in a more abstract and formal way, and to apply the concepts and tools of category theory to reason about it.

To perform the ‘experiment’, we open a buy order on the current chart symbol, say EURUSD, 0.1 lots and then at each new bar we log the current float, MA, and ATR. Based on the observed data, we can see if there are any correlations between the data and this can then be used in coming up with an ideal trailing stop system for _long_ positions. The indicators used here of MA and ATR can easily be substituted for what the reader deems more appropriate. I have only chosen these for illustration.

If we run this experiment on EURUSD on the hourly timeframe on the 1st of March, this will be some of our data.

|     |     |     |
| --- | --- | --- |
| P: (Float/ Profit) | A: ATR | B: MA |
| -6.60000 | 0.00203 | 1.12138 |
| -14.90000 | 0.00181 | 1.12136 |
| -18.80000 | 0.00175 | 1.12140 |
| -24.20000 | 0.00157 | 1.12125 |
| -29.00000 | 0.00146 | 1.12100 |
| -24.30000 | 0.00127 | 1.12078 |
|  |  |  |

If we do _lagging_ correlations between each of our two domain data sets A & B with the position profit, this can help set whether each of these domains could forecast negative drawdowns in a position. This information will help in setting or moving an existing stop-loss on a long position.

Another way these spans could help in defining a stop-loss could be if we hypothesise that each of the terminal domains A & B via their respective morphisms f & g do form a coproduct (sum) of how far the ideal stop-loss should be for long positions. These morphisms are in essence functions that take input and provide output. In this case each of the terminal domains would provide its indicator value as input and each of the functions f and g would provide an output double value.

Summing these double values, the equivalent of a coproduct, would give us the ideal stop-loss price. If we take it that the morphisms’ output (stop-loss price) has a linear relationship with the morphism’s indicator inputs, then these equations are implied.

![](https://c.mql5.com/2/52/3821654676641.png)

where xa ⊆A and ma and ca are slope and y-intercept coefficients of the linear relation. Similarly, for B domain

![](https://c.mql5.com/2/52/363550399846.png)

This hypothesis is for a linear relationship between the ideal stop-loss delta and the indicator values. If this relationship were a curve then the above equations would be quadratic, and with more coefficients and exponents. Our simpler option however can be coded as indicated below.

```
double _sl=((m_ma.Main(_index)*m_slope_ma)+(m_intercept_ma*m_symbol.Point()))+((m_atr.Main(_index)*m_slope_atr)+(m_intercept_atr*m_symbol.Point()));

```

If, using the MQL5 in-built expert trailing class we build our own trailing class that uses our ideal stop-loss delta, then the coefficients m and c for both the A domain and B domain could be inputs for this trailing class. Testing over the past year, for EURUSD on the hour with our signal as the built-in ‘signalRSI.mqh’ class, give us the report and curve represented below.

![](https://c.mql5.com/2/52/ct_4_report_fixed.png)

[![c_1](https://c.mql5.com/2/52/ct_4_curve_1.png)](https://c.mql5.com/2/52/ct_4_curve_1.png "https://c.mql5.com/2/52/ct_4_curve_1.png")

The ideas indicated here could be developed further if we look at composite spans.

A composite span is a span of spans. In our case we could make the argument that our indicator values are lagging, say slightly. This slight lag could be leading to less accurate prices for our trailing stop. In order to address this, we could re-constitute domains A and B as spans. A would be changed to a product of A’ and C, while B would be of C and B’.

![mc_1](https://c.mql5.com/2/52/multi_comp_1.png)

Composite spans that map to subdomains A', C, and B’ can help optimise the morphisms f: P to A, g: P to B, f’: A’ to A, f’’: C to A, g’’: C to B, and g’: B’ to B in fine tuning our trailing stop system by providing a more granular view of the data and relationships between observables in domains A and B.

In re-constituting A and B, we are now viewing them as products of A’ & C for A and C & B’ for B. Remember P is a sum of A & B but these will be products. We know ATR is the average true-range of price over a set period so it is mathematically equivalent to product between the inverse of the period length, and the sum of price-ranges over that length meaning we have A’ for our period length inverse and C for our prices sum. Conversely, the MA is equivalent to the product of an inverse of a period and the sum of recent-prices. So, in this case we also have the same product arrangement we have at span A.

![m_c_2](https://c.mql5.com/2/52/multi_comp_2.png)

The domain C is labelled as price and if we are to be a bit pedantic this seems contradictory in that that price availed to the ATR (A), is a sum of the true-range values over a period where as that given to the MA is simply the SUM current close prices.

This is where Category theory’s unwritten rule of focusing on morphisms between domains and not on unpacking what is in a domain comes in handy. Because with it we can, not only easily assemble our diagram as simply as shown above but can also easily identify possible [universal properties](https://en.wikipedia.org/wiki/Universal_property "https://en.wikipedia.org/wiki/Universal_property"). Universal properties are what set our approach here apart from other numerous math methods that are typically used in optimising or solving for missing values.

As far as we’re concerned domain C represents price. How that domain processes out price ranges versus close prices is not in the scope of this article and in fact for our purposes does not affect the end result.

However, before we look at applicable universal properties it is helpful to note that by breaking down the data into these subdomains, it becomes easier to identify patterns and correlations between the different observables, which can help optimise not just the morphisms f and g, but also f’, f’’, g’, and g’’.

Once again for brevity we can take the relationships between the added terminal domains A’, C, and B’ to be linear meaning the equation formats above will still be applicable.

If as above we do run tests now with more inputs as the number of morphisms has tripled, we do get the following report and test curve.

![](https://c.mql5.com/2/52/ct_4_report_2_fixed.png)

[![c_2](https://c.mql5.com/2/52/ct_4_curve_2.png)](https://c.mql5.com/2/52/ct_4_curve_2.png "https://c.mql5.com/2/52/ct_4_curve_2.png")

There is clearly improvement in the overall performance of the trailing system. However, this could be attributable to over fitting as the number of input parameters went up 3-fold from the previous test.

Returning to the concept of universal property, our composite span does present us with two candidates for this concept. Firstly, in the domain A span, since domain A’ and domain C, are terminal, a universal morphism is implied between A’ and C.

If we label these morphisms f’’’ and g’’’ for C to A’ and C to B’ respectively, we do imply there are relationships between price and the period chosen for ATR. Likewise, there would be a relationship between price and the period for MA indicator.

As has been the hypothesis in the morphisms above, the relationship between price and the indicator periods can be linear meaning we stick to the simple equation format above or it could be a curve meaning we pick our highest exponent and adopt a quadratic equation.

If we however stick to linear relationships, and building on our last version of the trailing class we keep the constants to all morphisms (with the exception of f’’’ and g’’’) constant (unaltered and always using default values), we could run comparative tests over the same period and see how it performs compared to our previous trailing classes.

This is the testing result.

![](https://c.mql5.com/2/52/ct_4_report_3_fixed.png)

[![c_3](https://c.mql5.com/2/52/ct_4_curve_3.png)](https://c.mql5.com/2/52/ct_4_curve_3.png "https://c.mql5.com/2/52/ct_4_curve_3.png")

The results are not the best of the three reports but the level of performance with fewer inputs and while using the universal-property principles does mean this could be an idea worth examining over longer periods? As always, all code posted here is not a grail or complete trading system so readers are urged to do their own research and diligence before even using any parts of it.

### Conclusion

In conclusion, we have seen how spans, experiments, and compositions in category theory could be used in setting exit trading strategies. Spans are a cell unit of a pairing of ideas/precepts or systems represented here as domains. This pairing does provide experiments which are in essence the universal property of this span. Composition takes the span and augments it with other spans in order to come up with more insightful systems and methods which in our case here were useful in fine tuning the exit strategy of a trading system.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/12394.zip "Download all attachments in the single ZIP archive")

[TrailingCT4.mqh](https://www.mql5.com/en/articles/download/12394/trailingct4.mqh "Download TrailingCT4.mqh")(9.4 KB)

[TrailingCT4\_r2.mqh](https://www.mql5.com/en/articles/download/12394/trailingct4_r2.mqh "Download TrailingCT4_r2.mqh")(10.7 KB)

[TrailingCT4\_r3.mqh](https://www.mql5.com/en/articles/download/12394/trailingct4_r3.mqh "Download TrailingCT4_r3.mqh")(12.07 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Codex Pipelines, from Python to MQL5, for Indicator Selection: A Multi-Quarter Analysis of the XLF ETF with Machine Learning](https://www.mql5.com/en/articles/20595)
- [Codex Pipelines: From Python to MQL5 for Indicator Selection — A Multi-Quarter Analysis of the FXI ETF](https://www.mql5.com/en/articles/20550)
- [Market Positioning Codex for VGT with Kendall's Tau and Distance Correlation](https://www.mql5.com/en/articles/20271)
- [Markets Positioning Codex in MQL5 (Part 2): Bitwise Learning, with Multi-Patterns for Nvidia](https://www.mql5.com/en/articles/20045)
- [Markets Positioning Codex in MQL5 (Part 1): Bitwise Learning for Nvidia](https://www.mql5.com/en/articles/20020)
- [MQL5 Wizard Techniques you should know (Part 85): Using Patterns of Stochastic-Oscillator and the FrAMA with Beta VAE Inference Learning](https://www.mql5.com/en/articles/19948)
- [MQL5 Wizard Techniques you should know (Part 84): Using Patterns of Stochastic Oscillator and the FrAMA - Conclusion](https://www.mql5.com/en/articles/19890)

**[Go to discussion](https://www.mql5.com/en/forum/444497)**

![Creating a comprehensive Owl trading strategy](https://c.mql5.com/2/0/Example_of_creating_Avatar.png)[Creating a comprehensive Owl trading strategy](https://www.mql5.com/en/articles/12026)

My strategy is based on the classic trading fundamentals and the refinement of indicators that are widely used in all types of markets. This is a ready-made tool allowing you to follow the proposed new profitable trading strategy.

![Creating an EA that works automatically (Part 08): OnTradeTransaction](https://c.mql5.com/2/50/aprendendo_construindo_008_avatar.png)[Creating an EA that works automatically (Part 08): OnTradeTransaction](https://www.mql5.com/en/articles/11248)

In this article, we will see how to use the event handling system to quickly and efficiently process issues related to the order system. With this system the EA will work faster, so that it will not have to constantly search for the required data.

![Neural networks made easy (Part 35): Intrinsic Curiosity Module](https://c.mql5.com/2/50/Neural_Networks_Made_035_avatar.png)[Neural networks made easy (Part 35): Intrinsic Curiosity Module](https://www.mql5.com/en/articles/11833)

We continue to study reinforcement learning algorithms. All the algorithms we have considered so far required the creation of a reward policy to enable the agent to evaluate each of its actions at each transition from one system state to another. However, this approach is rather artificial. In practice, there is some time lag between an action and a reward. In this article, we will get acquainted with a model training algorithm which can work with various time delays from the action to the reward.

![Data Science and Machine Learning(Part 14): Finding Your Way in the Markets with Kohonen Maps](https://c.mql5.com/2/52/data_science_ml_kohonen_maps_avatar.png)[Data Science and Machine Learning(Part 14): Finding Your Way in the Markets with Kohonen Maps](https://www.mql5.com/en/articles/12261)

Are you looking for a cutting-edge approach to trading that can help you navigate complex and ever-changing markets? Look no further than Kohonen maps, an innovative form of artificial neural networks that can help you uncover hidden patterns and trends in market data. In this article, we'll explore how Kohonen maps work, and how they can be used to develop smarter, more effective trading strategies. Whether you're a seasoned trader or just starting out, you won't want to miss this exciting new approach to trading.

[![](https://www.mql5.com/ff/sh/ub4fqgrk4rkv8gz9z2/01.png)![](https://www.mql5.com/ff/sh/ub4fqgrk4rkv8gz9z2/02.png)Explore your trading for freeUpdated statistics in MetaTrader 5 will help you to thoroughly evaluate results and reduce risksLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/454106&a=bkbqgaxtrafeuegfvjisjjwjohagrvnr&s=25c5856d7857fc6b6db7cffb15ae4ce40fd19d1ab594d8a900ad65673d9ffa0e&uid=&ref=https://www.mql5.com/en/articles/12394&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5071837154033741665)

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