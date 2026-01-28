---
title: Category Theory in MQL5 (Part 12): Orders
url: https://www.mql5.com/en/articles/12873
categories: Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T19:25:39.896020
---

[Best articles and CodeBase updates in MQL5.community channelsFollow us to ensure you never miss out on important updates![](https://www.mql5.com/ff/sh/n9yf51p2srwzfqh5z2/02.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=dgazvhktsxqakdvarucjbvmvzenwlyje&s=98a038fe082e458df8c4a1d8e116e3a6646fd5517f06e48b2356b7ee005817d6&uid=&ref=https://www.mql5.com/en/articles/12873&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5070274696471057153)

MetaTrader 5 / Trading systems


### **Introduction**

In the prior article we looked at graphs, the vertices and arrows of an interconnected system, within category theory and examined how various paths with their attributes could be used to define various trailing stop methods to a typical trading system.

For this article we will consider [orders](https://en.wikipedia.org/wiki/Order_theory "https://en.wikipedia.org/wiki/Order_theory") within category theory and how, like on our previous article, they could supplement trading setups via trailing stops. Orders, speaking plainly, are concerned with ranking the ‘magnitude’ of various elements, typically found in a set. They bring to light the idea that a particular set can have its elements ranked according to multiple criteria. Category theory adds dimension to this by introducing notion of set of sets and even set of set of sets (etc.). For this article therefore, we will be dwelling on set ranks.

Specifically, we will focus on patterns in ordered sets as a means of generating trade exits. There are many sets that could be considered in this effort since our list could range from price action patterns, to indicator patterns or even common multi-asset index patterns. However, keeping with the steps of a basic trading system we looked at in previous articles, we will reconstitute the three-intra steps between the first and last steps of our five-step process, into subsets in order to derive our ordered set patterns.

### **Understanding Orders in Category Theory**

As per order theory, there are mainly three types of orders. [Pre-orders](https://en.wikipedia.org/wiki/Preorder "https://en.wikipedia.org/wiki/Preorder"), [Partial-orders](https://en.wikipedia.org/wiki/Partially_ordered_set#Partial_order "https://en.wikipedia.org/wiki/Partially_ordered_set#Partial_order"), and [Linear-orders](https://en.wikipedia.org/wiki/Total_order "https://en.wikipedia.org/wiki/Total_order"). Linear-orders are also called Total-orders. Preorders are set elements rankings where every element of the set is compared to all the other elements (reflexivity) and the results of each comparison have logical bearing on other element comparisons (transitivity). Preorders do accommodate ambiguity where if any two elements have the same magnitude as per the binary operation, both can be included in the output set. Also preorder sets accommodate undefined results which happen in the event that two elements cannot be compared due to, say, fundamental dissimilarity. Partial orders are a type of preorder that introduce an additional concept of antisymmetry. What this means is if any two elements are found to have the same magnitude from the comparing binary operation then only one of them is included in the output set. Thus, the name ‘partial’ since only one of any equal elements outputted. Finally, linear orders, keeping with this trend, is a specialized form of partial order where there are no undefined results. This means all elements are comparable. Recall as mentioned above with pre-orders, some elements could be incomparable meaning the binary operation outputs an undefined result. This is not accommodated with linear-orders.

Thus, formally, given a set S with a binary relation R,

### R ⊆ S x S

R would be considered a pre-order if for all:

### s, s’, s” ∈ S

we have reflexivity which is formally represented as:

### s ≦ s

as well as transitivity which would be implied by:

### s ≦ s’ and s’ ≦ s” meaning s ≦ s”

Partial orders as mentioned above would add antisymmetry to pre-orders such that if:

### s ≦ s’ and s’ ≦ s then s = s’

implying only one of s, or s’ is represented in output.

Linear orders would add comparability to partial orders such that for any two elements there is a definite relation of either:

### s ≦ s’ or s’ ≦ s

meaning undefined outputs are not accommodated.

As mentioned these order formats are useful in marking patterns that can aid in making decisions in any trading system. For this article we will reconstitute the intra-step monoid sets, used in previous articles, of lookback period, applied price, and indicator while for simplicity preserving the end sets of timeframe and trade-action. The primary tested parameter will be decision order as was the case in previous article however because we will consider partial orders there is possibility for undefined results. This means our output order, of monoid sets that represent decision points, may be less than the input 3. Sometimes it may be 2 or even 1. This implies in certain situations we may have to make only 3 decisions or 4 decisions rather than the default 5 when assessing whether to modify our stop loss.

### **Applying Order Theory in MQL5**

For this article we will not look at preorders since such an article would be too long. Rather we will focus on partial orders and linear orders. Since we already have the data structures that define our five trade steps following the previous article, the main thing that is pending for implementation of order theory is the binary functions that process our data structures and output a set (typically a subset of the inputs). Since we are focusing on partial orders and linear orders, there will be one function for each.

It may be useful to emphasize the differences and relative benefits of the two ordering forms we are considering for this article. Partial orders as mentioned above allow undefined classifications in their output unlike linear orders. This can be useful in a number of scenarios. Let’s consider a simple case of classifying price bars on a chart as either bullish or bearish. In this process you are bound to come upon long legged Doji candle which strictly speaking is neither bullish nor bearish. With linear order classification this data point would have to be omitted since it would violate the comparability axiom.

![](https://c.mql5.com/2/55/3468670854337.png)

However, with partial ordering, by including this data point and thus its result, the output set would be more complete and representative of the dataset. To show why this is significant, long legged Doji candles, and other candles like it which include Gravestone Doji and the Dragonfly Doji, tend to feature in major price support and resistance regions. If your classification is therefore omitting any of these patterns your analysis and therefore forecasts are bound to be less accurate because for most trade systems, over the long-term price support and resistance regions play a critical role in defining trade setups. Addition of antisymmetry property in creating partial orders can thus be used in better filtering trade signals.

A case study on a trade system, that is different from our 5-step process looked at in previous articles, could have price bar patterns vectorized. By having each price bar formation as a vector which is simply an array of weights, we can compare how similar various patterns are. If we train over a decent period and identify eventual price formations for a considerable number of vectorized patterns we can compare any new pattern to what has been trained and based on the Euclidean distance from these trained patterns, the pattern closest to our new pattern can have its post price formation serve as the most likely outcome of our new pattern.

Linear ordering can be preferred to partial orders in instances of portfolio analysis. If we are faced with a wide array of assets that need to be appraised or included in a portfolio then linear-ordering with strict uniform weighting requirement (comparability) would be a no brainer. This is so for a number of reasons some of which are taken for granted. Linear orders allow evaluation of the assets in a sequential manner which provides a stream-lined process. Regardless of the number of assets, the priorities are already defined by the asset weighting which could be a value that quantifies anything from past realized value to future potential risk. This not only leads to efficiency but affords the trader to not consider every possible asset by adding focus.

This prioritization means the assets that matter the most have their investment decisions considered first which leads to the critical question of asset allocation. How will each asset be sized within the portfolio? With linear ordering the weighting of each asset can often act as a fair proxy to how much capital should be used in purchasing the asset something which would be harder to consistently do with partial ordering.

### **Case Study: Developing a Trading System with Orders**

The chosen trading strategy that builds on the system we looked at in previous articles and will involve selecting interim steps from 2 to 4 of our 5-step method. We will select these from partial order method in one trading system and also by linear order in another system.

For the partial order selection off the bat all our monoid sets are incomparable because we have look back period which is integer type, we have applied price which is a string enumeration, and finally indicator type which also strictly speaking is a string choice. So, in order to introduce some ability to use the binary operator:

### ≦

We will normalize just a pair of the sets, leaving he third to its default format. The inputs of the function that performs the partial order will inherently be price action. The price parameters we will consider as inputs to our partial order function will be autocorrelation indexing. We will simply assign indices to various autocorrelation patterns and for each index we will have a particular pair of sets normalized which will then inform our selected set order for our trade system. With partial orders as pointed out above the presence of undefined sets means only two sets will be selected meaning we will almost certainly always end up with a 4-step process rather than the 5-step we have been using.

For linear orders though all monoid sets will be normalized. This should give us the full 5-steps we have been considering and as with partial orders the input for the linear order function will be our autocorrelation index. The assignment of autocorrelation index will be rudimentary as can be seen from the listing below.

```
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CTrailingCT::Ordering(int Lookback)
   {
      m_low.Refresh(-1);
      m_high.Refresh(-1);
      m_close.Refresh(-1);

      double _c[],_l[],_h[];

      ArrayResize(_c,Lookback);ArrayInitialize(_c,0.0);
      ArrayResize(_l,Lookback);ArrayInitialize(_l,0.0);
      ArrayResize(_h,Lookback);ArrayInitialize(_h,0.0);

      for(int i=0;i<Lookback;i++)
      {
         _c[i]=m_close.GetData(i);
         _l[i]=m_low.GetData(Lookback+i);
         _h[i]=m_high.GetData(Lookback+i);
      }

      double _r_h=0.0,_r_l=0.0;

      if(MathCorrelationSpearman(_c,_l,_r_l) && MathCorrelationSpearman(_c,_h,_r_h))
      {
         if(_r_l>=__PHI){ LongIndex(5); }
         else if(_r_l>=1.0-__PHI){ LongIndex(4); }
         else if(_r_l>=0.0){ LongIndex(3); }
         else if(_r_l>=(-1.0+__PHI)){ LongIndex(2); }
         else if(_r_l>=(-__PHI)){ LongIndex(1); }
         else{ LongIndex(0);}

         if(_r_h>=__PHI){ ShortIndex(5); }
         else if(_r_h>=1.0-__PHI){ ShortIndex(4); }
         else if(_r_h>=0.0){ ShortIndex(3); }
         else if(_r_h>=(-1.0+__PHI)){ ShortIndex(2); }
         else if(_r_h>=(-__PHI)){ ShortIndex(1); }
         else{ ShortIndex(0);}

         return(true);
      }

      return(false);
   }
```

To begin our partial order processing code will be embedded in the long and short trailing stop processing functions as per attachment at the end of the article. What is noteworthy here is in essence each assigned index only maps two sets since the third is undefined. The order as defined by each possible index implies the first assigned set has a higher weighting than the last assigned set. The need to actually come up with a normalizing function that gives a physical weight to each is therefore negated since it will not meaningfully affect the end result.

Likewise, the linear order function will be listed as follows:

```
	ENUM_TIMEFRAMES _timeframe=GetTimeframe(m_timeframe,__TIMEFRAMES);

         int _lookback=m_default_lookback;
         ENUM_APPLIED_PRICE _appliedprice=__APPLIEDPRICES[m_default_appliedprice];
         double _indicator=m_default_indicator;

         if(m_long_index==0)
         {
            _lookback=GetLookback(m_lookback,__LOOKBACKS,_timeframe);
            _appliedprice=GetAppliedprice(m_appliedprice,__APPLIEDPRICES,_lookback,_timeframe);
            _indicator=GetIndicator(_lookback,_timeframe,_appliedprice);
         }
         else if(m_long_index==1)
         {
            _appliedprice=GetAppliedprice(m_appliedprice,__APPLIEDPRICES,__LOOKBACKS[m_default_lookback],_timeframe);
            _lookback=GetLookback(m_lookback,__LOOKBACKS,_timeframe);
            _indicator=GetIndicator(_lookback,_timeframe,_appliedprice);
         }
         else if(m_long_index==2)
         {
            _appliedprice=GetAppliedprice(m_appliedprice,__APPLIEDPRICES,__LOOKBACKS[m_default_lookback],_timeframe);
            _indicator=GetIndicator(m_default_lookback,_timeframe,_appliedprice);
            _lookback=GetLookback(m_lookback,__LOOKBACKS,_timeframe);
         }
         else if(m_long_index==3)
         {
            _indicator=GetIndicator(__LOOKBACKS[m_default_lookback],_timeframe,__APPLIEDPRICES[m_default_appliedprice]);
            _appliedprice=GetAppliedprice(m_appliedprice,__APPLIEDPRICES,m_default_lookback,_timeframe);
            _lookback=GetLookback(m_lookback,__LOOKBACKS,_timeframe);
         }
         else if(m_long_index==4)
         {
            _indicator=GetIndicator(__LOOKBACKS[m_default_lookback],_timeframe,__APPLIEDPRICES[m_default_appliedprice]);
            _lookback=GetLookback(m_lookback,__LOOKBACKS,_timeframe);
            _appliedprice=GetAppliedprice(m_appliedprice,__APPLIEDPRICES,_lookback,_timeframe);
         }
         else if(m_long_index==5)
         {
            _lookback=GetLookback(m_lookback,__LOOKBACKS,_timeframe);
            _indicator=GetIndicator(_lookback,_timeframe,__APPLIEDPRICES[m_default_appliedprice]);
            _appliedprice=GetAppliedprice(m_appliedprice,__APPLIEDPRICES,_lookback,_timeframe);
         }
         //
         int _trade_decision=GetTradeDecision(_timeframe,_lookback,_appliedprice,_indicator);
```

What is noteworthy here is each index provides an exhaustive enumeration of all the sets since comparability is a requirement for all sets. And as mentioned above, the exercise of assigning weight numbers to each set which then are used in sorting is eschewed since the indexing used implies a weighting order which is implemented as shown above.

If we run tests using our new trailing classes, on a signal class of the library awesome oscillator using fixed margin for the symbol USDJPY for the past 12 months we get the following reports for each of the ordering methods. First is the report on partial order method.

[![r1](https://c.mql5.com/2/55/ct_12_report_1__1.png)](https://c.mql5.com/2/55/ct_12_report_1.png "https://c.mql5.com/2/55/ct_12_report_1.png")

Our expert advisor clocks more than 100k in earnings on settings, as is always the case in these article series, that do not utilize price targets for take profit or stop loss but only hold positions until signal indicator deems they should be closed. Additionally, of course since we are honing our trailing stop most of the profitable closed positions were actually due to our trailing stop, which one can argue speaks to the merit of partial orders in generating reliable trailing stop set indications. We also tested linear order based trailing stops and this yielded the following report.

[![r2](https://c.mql5.com/2/55/ct_12_report_2__1.png)](https://c.mql5.com/2/55/ct_12_report_2.png "https://c.mql5.com/2/55/ct_12_report_2.png")

Surprisingly this result is not as profitable as what we had with partial (‘ambiguous’ accommodating) orders. Bizarrely enough the equity drawdown is even worse on fewer trades. Testing over longer periods and on multiple symbols is required for drawing any definitive conclusions but it could be safe to say partial orders are more promising than linear orders.

### **Conclusion**

This article has considered the effectiveness of partial orders versus linear orders in setting and modifying trailing stops to a typical expert advisor. Prior to that we considered the relative benefits of these two ordering principles and we did not look at the anchor ordering principle of both, namely pre-orders since that would have made the article too long without meaningfully adding to the content. Remember partial orders are a specialized form of pre-orders while linear orders are also a specialized form of partial orders. So, with plenty of overlap in the basic definition of these ordering types we focused on the last two.

The order theory methods of partial and linear have been exploited for their specific benefits. Partial orders allow more flexibility in classification of raw data sets which can lead to a richer and more accurate analysis. Conversely linear ordering by being stricter tends to require raw data normalization to ensure comparability which leads to overall prioritization and efficiency in decision making.

These ordering methods have potential for further exploration and refinement in these order series as we begin to integrate past covered concepts into some new ones that we’ll be looking at.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/12873.zip "Download all attachments in the single ZIP archive")

[TrailingCT\_12\_2.mqh](https://www.mql5.com/en/articles/download/12873/trailingct_12_2.mqh "Download TrailingCT_12_2.mqh")(50.71 KB)

[ct\_12.mqh](https://www.mql5.com/en/articles/download/12873/ct_12.mqh "Download ct_12.mqh")(27.33 KB)

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

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/450141)**
(1)


![Carl Schreiber](https://c.mql5.com/avatar/2018/2/5A745EEE-EB76.PNG)

**[Carl Schreiber](https://www.mql5.com/en/users/gooly)**
\|
9 Aug 2023 at 13:27

It's disappointing, again no EA that produces the shown [test results](https://www.mql5.com/en/docs/common/TesterStatistics "MQL5 Documentation: TesterStatistics function"). :(

Please provide the EA(s) here with the setup so that we can reproduce the results.

![Integrating ML models with the Strategy Tester (Part 3): Managing CSV files (II)](https://c.mql5.com/2/51/Avatar_Perceptron_Multicamadas_e_o-Algoritmo_Backpropagation_Parte_3_02.png)[Integrating ML models with the Strategy Tester (Part 3): Managing CSV files (II)](https://www.mql5.com/en/articles/12069)

This material provides a complete guide to creating a class in MQL5 for efficient management of CSV files. We will see the implementation of methods for opening, writing, reading, and transforming data. We will also consider how to use them to store and access information. In addition, we will discuss the limitations and the most important aspects of using such a class. This article ca be a valuable resource for those who want to learn how to process CSV files in MQL5.

![Rebuy algorithm: Multicurrency trading simulation](https://c.mql5.com/2/54/Multicurrency_Trading_Simulation_Avatar.png)[Rebuy algorithm: Multicurrency trading simulation](https://www.mql5.com/en/articles/12579)

In this article, we will create a mathematical model for simulating multicurrency pricing and complete the study of the diversification principle as part of the search for mechanisms to increase the trading efficiency, which I started in the previous article with theoretical calculations.

![Developing an MQTT client for MetaTrader 5: a TDD approach](https://c.mql5.com/2/56/mqtt-avatar.png)[Developing an MQTT client for MetaTrader 5: a TDD approach](https://www.mql5.com/en/articles/12857)

This article reports the first attempts in the development of a native MQTT client for MQL5. MQTT is a Client Server publish/subscribe messaging transport protocol. It is lightweight, open, simple, and designed to be easy to implement. These characteristics make it ideal for use in many situations.

![Understanding MQL5 Object-Oriented Programming (OOP)](https://c.mql5.com/2/56/object-oriented-programming-avatar.png)[Understanding MQL5 Object-Oriented Programming (OOP)](https://www.mql5.com/en/articles/12813)

As developers, we need to learn how to create and develop software that can be reusable and flexible without duplicated code especially if we have different objects with different behaviors. This can be smoothly done by using object-oriented programming techniques and principles. In this article, we will present the basics of MQL5 Object-Oriented programming to understand how we can use principles and practices of this critical topic in our software.

[![](https://www.mql5.com/ff/sh/zf7a2k61x98jzh89z2/01.png)Speed up your tradingUse our high-speed VPS for MetaTrader 4 and 5Learn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=qtrrsuiwuicrscmckjynyanztbditglq&s=c617dc80d90cfd3783ec1345eec2b419b281f10fec6eac77b3218984ac337259&uid=&ref=https://www.mql5.com/en/articles/12873&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5070274696471057153)

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