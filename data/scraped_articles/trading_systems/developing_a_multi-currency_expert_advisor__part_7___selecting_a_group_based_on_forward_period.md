---
title: Developing a multi-currency Expert Advisor (Part 7): Selecting a group based on forward period
url: https://www.mql5.com/en/articles/14549
categories: Trading Systems, Integration
relevance_score: 9
scraped_at: 2026-01-22T17:37:31.706496
---

[![](https://www.mql5.com/ff/sh/592yc11u3j4rs5z9z2/01.png)How AI helps create robots for MetaTrader 5Learn from our book "Neural Networks in Algo Trading with MQL5"Read](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/neurobook%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.neurobook%26utm_content=visit.page%26utm_campaign=neurobook.promo.04.2024&a=ghrobswocqgvhztzjldphupateyllpro&s=9929cb0b8629585b5a42fabc06c525e41f6c0ebdf3045d044a5413b93ea88b47&uid=&ref=https://www.mql5.com/en/articles/14549&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5049260387859277865)

MetaTrader 5 / Tester


### Introduction

In the previous [articles](https://www.mql5.com/ru/blogs/post/756958), I have been optimizing individual instances of trading strategies on the same time period - from 2018 to 2022. This is a fairly long period, which surely includes various events affecting the price dynamics. On the other hand, it is not too large and the time of one run remains small enough — within a few tens of seconds. The selected period is arranged so that there is still more than a year left until the current moment in time. This makes it possible to see how the strategy instances will behave in the section of history that was not used when optimizing their parameters.

The standard MetaTrader 5 tester can perform single passes and optimization taking into account the presence of the so-called forward period. When using it, the tester will split the entire specified test period into two parts — main period and forward period. At the end of the main period, all trades are closed and the balance of the trading account is returned to its initial state. Then the EA will work again on the forward period, and all statistics collected by the tester will be calculated separately for the main and forward periods.

In the field of machine learning, the terms In-Sample and Out-Of-Sample (IS and OOS) are often used to refer to the dataset models are trained and tested on. In our area, the main period will play the role of IS, and the forward periodwill be OOS. Keep in mind that OOS is a broader concept than the forward period. We can run EA tests without specifying the forward period in the tester, but if the test period is located outside the period the optimization was carried out on, then this will still be testing on OOS.

Since we have not used OOS testing much in the past (except for the end of this [article](https://www.mql5.com/en/articles/14148)), then it is time to see whether it is possible to maintain comparable results when working on the forward period, as shown by the EAs on the main period.

### Forward period for ready-made EAs

Let's look at the results shown by the EAs from the previous [article](https://www.mql5.com/en/articles/14478) for a group of manually selected strategies and the best groups of strategies selected automatically based on maximum profit. I will take the entire year of 2023 as the forward period. Three months of 2024 will be left in reserve for now. The result I would like to achieve is a profit in the forward period being approximately five times less than on the main period, since the forward is five times shorter than the main period. The drawdown on the forward should preferably be approximately the same or less than on the main period.

Let's start with a group of manually selected strategies ( _BaselineExpert.mq5_ EA). The vertical section of the blue line clearly separates the main period from the forward period on the graph. At this boundary, the account balance again becomes equal to USD 10,000. The part of the chart that relates to the forward period occupies only a small part of the entire graph. If we need to examine it in more detail, you can run a separate tester pass only on the time interval that refers here to the forward period.

![](https://c.mql5.com/2/74/6532823501696.png)

Main period

![](https://c.mql5.com/2/74/378673279494.png)

Forward period

![](https://c.mql5.com/2/74/65037398656.png)

Fig. 1. BaselineExpert.mq5 EA results on the main and forward periods

In this case, we clearly see that the results are significantly worse even without a separate pass for the forward period. This does not mean a complete breakdown of the strategy though. The drawdown increased from 10% to 12% and the recovery period lasted more than six months. But this is not yet a reversal of the balance curve trend. Or is it? No matter how you look at it, the results need improvement.

Let's now look at the best group selected without clustering sets ( _OptGroupExpert.mq5_ EA).

![](https://c.mql5.com/2/74/1544364330534.png)

Main period

![](https://c.mql5.com/2/74/24176346719.png)

Forward period

![](https://c.mql5.com/2/74/934353009686.png)

Fig. 2. OptGroupExpert.mq5 EA results on the main and forward periods

For this group of strategies, the results are also significantly worse than expected, despite the fact that a small profit was still made at the end of the forward period. The drawdown has increased by almost one and a half times. It seems that the best group in the main period is no longer than the best group in the forward.

Let's look at the results of the group selected with clustering of parameter sets ( _OptGroupClusterExpert.mq5_ EA), although there is already a strong suspicion that the results will be approximately the same as in the two previous cases.

![](https://c.mql5.com/2/74/2851336609942.png)

Main period

![](https://c.mql5.com/2/74/1067474196641.png)

Forward period

![](https://c.mql5.com/2/74/4908186340959.png)

Fig. 3. OptGroupClusterExpert.mq5 EA results on the main and forward periods

Indeed, my suspicion was justified. The results here are just as unclear, with an even greater drawdown. Therefore, we will not present the results for the best group selected from the file, where only one set is left in one cluster. They are about the same.

So, do we have any groups whose results in the forward period satisfy our expectations? To answer this question, we will conduct a new optimization, but with a forward period, so as not to manually launch the EA in the tester on the forward period for all groups of parameter sets obtained during the previous optimization.

### Optimization results with forward period

So, we got the optimization results with a forward period for the _OptGroupClusterExpert.mq5_ EA. The parameter set file used was a file with one set for each of the 64 clusters. For the initial analysis of optimization results, ordinary Excel will probably be sufficient for us. Let's export the optimization results for the main and forward periods to XML files and save them in Excel format:

![](https://c.mql5.com/2/74/756447678969.png)

Fig. 4. Source files with optimization results for the main (IS) and forward periods (OOS)

There is the _Back Result_ column in the file for the forward period, which contains the result obtained in the main period with the same set of optimized parameters. This is good, but we would like to see all the other characteristics from the main period next to it. Therefore, we will combine the data from these two tables into one by the _Pass_ key column. The same values in this column correspond to the same combinations of inputs in the pass.

After merging, we will color the data related to the main period and the forward period in different colors, temporarily hide some columns and sort the data in descending order of normalized profit in the main period:

![](https://c.mql5.com/2/74/1839461482355.png)

Fig. 5. Combined results for the main and forward periods

It is clear that for the best results in the main period, the results in the forward period are mostly negative. The results are by no means disastrous (losing 3-5% of the initial balance per year), but they are certainly not good.

Let's recall how we get the values in the _Forward Result_ and _Back Result_ columns. This is the result of the _OnTester()_ function returned by the EA after the pass. We call this value normalized profit and calculate it as follows:

_Result = Profit_ \\* (10% / _EquityDD_),

where _Profit is a_ profit obtained during the test period, _EquityDD is_ a maximum relative drawdown of funds during the test period.

The meaning of this calculated value is the estimated profit over the test period that could be obtained if the size of the opened positions were changed so that the maximum relative drawdown reached 10%.

The parameter results can be compared correctly if the same position scaling factor is used for the forward period and main periods: _coeff_ = (10% / _EquityDD_). It would be problematic for us to obtain the value of this ratio for the main period during the forward test, so let's make such an adjustment now. The conversion equation will look like this:

_ForwardResultCorrected = ForwardResult \*_( _coeff\_IS / coeff\_OOS_)

= _ForwardResult_ \\* ( _EquityDD\_OOS_ / _EquityDD\_IS_)

After applying the adjustment, we get the following results:

![](https://c.mql5.com/2/74/4936682818965.png)

Fig. 6. Results after re-calculating normalized profit in the forward period

We see that the results in the forward period have increased in absolute value. This is correct for the following reasons. Let's imagine that we took, for example, the second set of parameters on the main period. Based on the fact that the drawdown on it was 1.52%, we will increase the position size by 10 / 1.52 = 6.58 times to achieve the target drawdown of 10%. Then, if we do not know anything about the forward period yet, then we would also have to increase the size of positions by 6.58 times. But in this case, if the profit received in the forward period was -98, then the normalized profit should be calculated by multiplying the profit by the same ratio of 6.58. So, we get -635 instead of -240. The value of -240 was smaller because the drawdown in the forward period was almost three times larger (4.03% instead of 1.52%) and when calculating the normalized profit, the ratio was 10 / 4.03 = 2.48, that is, almost three times smaller.

The results are not very pleasant. Let's try to find something more encouraging now. First, let's see if we have any positive results in the forward period at all. Let's sort the data by the Forward Result Corrected column and see the following:

![](https://c.mql5.com/2/74/2208534192441.png)

Fig. 7. Results sorted by the forward period result

Still, there are groups of sets that have positive results even in the forward period. They correspond to those groups, for which the standardized profit of approximately 15,000–18,000 is achieved in the main period. We see that here the drawdown does not differ much between the main and forward periods, and the normalized profit in the forward period is approximately one fifth of the normalized profit in the main period.

So, is it possible to choose good groups?

### Philosophical question

Actually, this is a very difficult question. It can be formulated in different ways. For example:

- Do we have the right to use selection taking into account the forward period?
- Wouldn't we be fooling ourselves if we hoped that such a choice would allow us to continue to obtain similar results?
- How much can we trust such a choice?
- Will the results be obviously better in the new period for the forward if we selected the group taking into account the forward, compared to the selection without taking it into account?
- For what future period can the results be repeated? Is it comparable to the forward period?
- If we carry out a similar selection procedure, for example, every year, will this mean that we will always have a good group for the next year? What about six months? What about every week?

While they have a common basis, these questions touch upon different aspects.

In order to try to somehow answer these questions, let us recall the Bayesian approach. First, we formulate a series of hypotheses (or assumptions), then we evaluate their probabilities before receiving new data. After running the experiment, we update our estimates to take into account new data. Some hypotheses become more probable in our eyes, while others become less likely. Here we consider probability as the degree of our confidence in a certain event outcome.

Our main hypothesis will be the following: selecting a group taking into account the forward period improves the results obtained in the period after the forward. Alternative hypothesis: selecting a group taking into account the forward period does not improve results.

One possible experiment would be to select a number of groups taking into account the forward period and several groups without selecting the forward period. Then we test all selected groups in the period after the forward.

If the groups selected with the forward period in mind perform better than the groups selected without the forward period, then this will be weak evidence in favor of the main hypothesis. Our confidence in the validity of the main hypothesis will increase.

If the results for groups selected without taking into account the forward period are approximately the same or better, then this will be weak evidence in favor of the alternative hypothesis. At the same time, we cannot completely reject any of the hypotheses, since the results shown by any group of strategies in the period after the forward depend on many other factors besides the method of selection into groups. It could simply have been an overall bad period for the strategies being used, and therefore one or another method of selecting strategies into groups might not have produced a noticeable effect.

We probably cannot afford more here.

The phrase "taking into account the forward period" may be understood slightly incorrectly. If we apply selection taking into account the forward period, this means that the period that was previously a forward (OOS), now ceases to be an OOS period and becomes an IS period, although we continue to call it a forward period. This means that to evaluate the trading results we need to use a new forward period (forward followed by forward, pardon the tautology).

Let's describe in more detail the experiment we want to conduct to obtain additional information. Let's say we have historical data for the period from 2018 to 2023 inclusive. Based on these, we want to select a group of strategies that will show good results in 2024. Then we can do it in two ways:

- Carry out the optimization on the period 2018-2023 (IS) and select the best group based on its results.
- Perform optimization on 2018-2022 (IS) with simultaneous verification for the forward period 2023 (OOS). Select the best group that provides good and approximately similar results in both periods.

In the second method, we will most likely not select the same group as in the first. Its results will be somewhat worse, but it lasted a year in the OOS period that did not participate in the optimization. In the first method, it is impossible to say something like this about the selected group, since we did not check it outside the IS period. But in the first method, we optimized (trained) the group over a longer period, since in the second method we had to allocate 2022 for the forward period and not use it for the optimization.

Which of these methods will be better? Let's try to conduct such an experiment by comparing the results for groups selected in two ways when trading in 2024.

### Selecting using the first method

To select using the first method, we first need to optimize a single copy of the strategy for the period 2018-2023. Previously, we carried out such optimization for the period up to 2022 without including 2023. After the optimization, we will obtain sets of parameters, which we will divide into clusters, as described in the previous [article](https://www.mql5.com/en/articles/14478). Then we run optimization to select good groups from eight sets of parameters. Let's look at the results of the best found groups of sets for the period 2018-2023 and 2024:

![](https://c.mql5.com/2/74/1270845482241.png)

Fig. 8. Results of OptGroupClusterExpert.mq5 optimization to select a group on the main period of 2018-2023

![](https://c.mql5.com/2/74/2896583694752.png)

Fig. 9. Results of OptGroupClusterExpert.mq5 to selectgroups for a period of three months in 2024

We see that the best groups found in the main period for 2018-2023 have generally positive results in the forward period in 2024, but they differ quite significantly from each other. For a more thorough check, select the topmost group, assign it a scaling factor value _scale\__ = 10 / 2.04 = 5 and run it in the tester on the main period of 2023 and the forward period of 2024.

![](https://c.mql5.com/2/74/1227095619698.png)

2023

![](https://c.mql5.com/2/74/4419358810240.png)

2024 (3 months)

![](https://c.mql5.com/2/74/279989030232.png)

Fig. 10. OptGroupClusterExpert.mq5 EA results for the best group for 2023 and 2024

Based on these results, it is not possible to particularly assess the prospects for the further EA behavior with such a group of sets of strategy parameters, but at least we did not see the onset of a clearly expressed trend towards a decrease in the balance curve in 2024. Let us remember these results and return to them later to compare them with the results obtained using the second selection method.

### Selecting using the second method

Let's use the ready-made optimization results on 2018-2022, select the best group in terms of the received standardized profit and take a closer look at its results. We have already seen them in Fig. 3, but now let's look at its graph not from 2018, but only from 2023. Let's set the entire 2023 year as the main period in the tester, and the entire available 2024 year as the forward period. This is what we get:

![](https://c.mql5.com/2/74/1182424119678.png)

2023

![](https://c.mql5.com/2/74/5296698602376.png)

2024 (3 months)

![](https://c.mql5.com/2/74/2549480981170.png)

Fig. 11.  OptGroupClusterExpert.mq5 EA results for 2023 and 2024

Note that the drawdown for 2023 exceeded the calculated one for the main test period by almost two times: USD 1820 instead of USD 1000.

Use the following algorithm to select into groups, while taking into account 2023 as a forward period:

- In the combined table of optimization results for 2018-2022 (main period) and for 2023 (forward period), calculate the ratio of their values in the main and forward periods for all parameters. For example, in case of the number of trades:


_TradesRatio = OOS\_Trades / IS\_Trades_.
The closer these ratios are to 1, the more identical the values of these parameters are in the two periods. For the profit parameter, introduce a ratio that takes into account the different period lengths — in one year, the profit should be approximately 5 times less than in five years:


_ResultRatio = OOS\_ForwardResultCorrected_ \\* 5 _/ IS\_BackResult_.

- Let's calculate for all these relations the sum of their deviations from unity. This value will be our measure of the difference between the results of each group in the main and forward periods:


_SumDiff_ = \|1 - _ResultRatio_ \| \+ ... \+ \|1 - _TradesRatio_ \|.

- Also, take into account that the drawdown could be different for each pass in the main and forward periods. Select the maximum drawdown from two periods and use it to calculate the scaling factor for the sizes of positions opened to achieve the standardized drawdown of 10%:

_Scale_ = 10 / _MAX(OOS\_EquityDD, IS\_EquityDD)_.

- Now we want to select the sets with the prevalence of _Scale_ over _SumDiff_. To do this, calculate the last parameter:

_Res = Scale / SumDiff_.

- Sort all groups by the value calculated in the previous _Res_ step in descending order. In this case, the groups, whose results in the main and forward periods were more similar and the drawdown in both periods was smaller, find themselves at the top of the table.

- Let's take the group at the top as the first one. To select the next group, sort out all the groups that have the same cluster indices as the first one, and again take the one that ends up at the very top. Let's repeat this a couple of times, now sorting out all the indices that were included in the previously selected groups. We will take the resulting four groups for the new EA.

To test the collaboration of selected groups based on the _OptGroupClusterExpert.mq5_ EA, create a new one and make some minor changes to it. Since the EA will not be used for optimization, the _OnTesterInit()_ and _OnTesterDeinit()_ functions can be removed from it. We can also remove the inputs that specify the indices of the parameter sets to include in the group, since we will hardcode them in the code based on the selection procedure performed.

In the _OnInit()_ function, create two arrays — _strGroups_ for selected groups and _scales_\- for the group multipliers. _strGroups_ array elements are the strings containing the indices of the parameter sets, separated by commas.

```
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit() {
   // Load strategy parameter sets
   int totalParams = LoadParams(fileName_, params);

   // If nothing is loaded, report an error
   if(totalParams == 0) {
      PrintFormat(__FUNCTION__" | ERROR: Can't load data from file %s.\n"
                  "Check that it exists in data folder or in common data folder.",
                  fileName_);
      return(INIT_PARAMETERS_INCORRECT);
   }

   // Selected set groups
   string strGroups[] = {"55,12,3,35,48,54,16,40",
                         "11,54,33,30,62,6,10,23",
                         "50,15,8,34,2,36,4,9",
                         "26,42,25,22,36,51,53,0"
                        };

   // Scaling factors for selected set groups
   double scales[] = {4.16,
                      3.40,
                      3.33,
                      2.76
                     };

   // Set parameters in the money management class
   CMoney::DepoPart(expectedDrawdown_ / 10.0);
   CMoney::FixedBalance(fixedBalance_);

   // Create an EA handling virtual positions
   expert = new CVirtualAdvisor(magic_, "SimpleVolumes_OptGroupForwardCluster");

   CVirtualStrategyGroup *groups[ArraySize(strGroups)];

   FOREACH(strGroups, {
      // Form the string from the parameter set indices separated by commas
      string strIndexes = strGroups[i];

      // Turn the string into the array
      string indexes[];
      StringSplit(strIndexes, ',', indexes);

      // Create and fill the array of all strategy instances
      CVirtualStrategy *strategies[];

      FOREACH(indexes, {
         // Remove the cluster number from the parameter set string
         string param = CSVStringGet(params[StringToInteger(indexes[i])], 0, 11);
         // Add a strategy with a set of parameters with a given index
         APPEND(strategies, new CSimpleVolumesStrategy(param))
      });

      // Add the strategy to the next group of strategies
      groups[i] = new CVirtualStrategyGroup(strategies, scales[i]);
   });

   // Form and add the group of strategy groups to the EA
   expert.Add(CVirtualStrategyGroup(groups, scale_));

   return(INIT_SUCCEEDED);
}
```

Save this code in the _OptGroupForwardClusterExpert.mq5_ file of the current folder.

Let's look at the EA test results. Just like last time, we will combine two periods in one pass - 2023 and the first three months of 2024.

![](https://c.mql5.com/2/74/2104622082411.png)

2023

![](https://c.mql5.com/2/74/6270556384075.png)

2024 (3 months)

![](https://c.mql5.com/2/74/3699310013696.png)

Fig. 12. OptGroupClusterForwardExpert.mq5 EA results for 2023 and 2024

Here, the results for 2023 are clearly better: the rising trend is observed throughout the entire period, although there is also a period from March to July when the balance curve did not show any significant growth. The drawdown during this period also improved and is within the maximum expected limits.

The results for 2024 are also better, but not particularly amazing. Perhaps, three months is a too short period for the graph to look as beautiful as over a long period of several years.

If we compare these results with the results for the first method of selecting good groups, no clear advantage is yet visible for any of these methods. The results are generally quite similar, but the second method required more effort from us compared to the first one. However, since we have clearly outlined the algorithm of actions for the second selection method, it can be automated in the future.

### Conclusion

As we can see, the conducted experiment did not increase our confidence that it is better to allocate an additional period as a forward and select groups taking into account the best work in both periods. But that does not mean this approach should not be used. Besides, we only used three months from 2024 for comparison. This is too short a period, since we have seen that the trading strategy used can have periods of balance growth fluctuations around zero lasting up to several months. Therefore, it is not clear whether the first three months of 2024 represent the beginning of such a period, which will then be replaced by growth, or there may not be any growth at all.

We could try to conduct a similar experiment by moving the periods back one year. In this case, the main period will begin in 2017, the selection by the second method will be carried out in 2022, and for comparison of the two methods we will have all of 2023 and the beginning of 2024.

However, we will move further. Nothing stops us from selecting some of the groups using the first method, while selecting others using the second one, and combine them in one EA. But what is the maximum number of trading strategy instances we can combine in one EA so that its operation does not take up too many server resources and does not require abnormally high amounts of RAM? I will try to clarify this issue in one of the following articles.

Thank you for your attention and stay tuned!

P.S. I have not made any changes to the previously created files while preparing this article. I have added only one new file. So it is the only one attached below. You can find all other files mentioned in the previous [article](https://www.mql5.com/en/articles/14478).

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/14549](https://www.mql5.com/ru/articles/14549)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/14549.zip "Download all attachments in the single ZIP archive")

[OptGroupForwardClusterExpert.mq5](https://www.mql5.com/en/articles/download/14549/optgroupforwardclusterexpert.mq5 "Download OptGroupForwardClusterExpert.mq5")(14.94 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Moving to MQL5 Algo Forge (Part 4): Working with Versions and Releases](https://www.mql5.com/en/articles/19623)
- [Moving to MQL5 Algo Forge (Part 3): Using External Repositories in Your Own Projects](https://www.mql5.com/en/articles/19436)
- [Moving to MQL5 Algo Forge (Part 2): Working with Multiple Repositories](https://www.mql5.com/en/articles/17698)
- [Moving to MQL5 Algo Forge (Part 1): Creating the Main Repository](https://www.mql5.com/en/articles/17646)
- [Developing a multi-currency Expert Advisor (Part 24): Adding a new strategy (II)](https://www.mql5.com/en/articles/17328)
- [Developing a multi-currency Expert Advisor (Part 24): Adding a new strategy (I)](https://www.mql5.com/en/articles/17277)
- [Developing a multi-currency Expert Advisor (Part 23): Putting in order the conveyor of automatic project optimization stages (II)](https://www.mql5.com/en/articles/16913)

**[Go to discussion](https://www.mql5.com/en/forum/471771)**

![Creating a Trading Administrator Panel in MQL5 (Part I): Building a Messaging Interface](https://c.mql5.com/2/90/logo-midjourney_image_15417_409_3949__4.png)[Creating a Trading Administrator Panel in MQL5 (Part I): Building a Messaging Interface](https://www.mql5.com/en/articles/15417)

This article discusses the creation of a Messaging Interface for MetaTrader 5, aimed at System Administrators, to facilitate communication with other traders directly within the platform. Recent integrations of social platforms with MQL5 allow for quick signal broadcasting across different channels. Imagine being able to validate sent signals with just a click—either "YES" or "NO." Read on to learn more.

![Reimagining Classic Strategies (Part VI): Multiple Time-Frame Analysis](https://c.mql5.com/2/89/logo-midjourney_image_15610_407_3930__2.png)[Reimagining Classic Strategies (Part VI): Multiple Time-Frame Analysis](https://www.mql5.com/en/articles/15610)

In this series of articles, we revisit classic strategies to see if we can improve them using AI. In today's article, we will examine the popular strategy of multiple time-frame analysis to judge if the strategy would be enhanced with AI.

![Creating an MQL5-Telegram Integrated Expert Advisor (Part 3): Sending Chart Screenshots with Captions from MQL5 to Telegram](https://c.mql5.com/2/89/logo-Creating_an_MQL5-Telegram_Integrated_Expert_Advisor_lPart_1k.png)[Creating an MQL5-Telegram Integrated Expert Advisor (Part 3): Sending Chart Screenshots with Captions from MQL5 to Telegram](https://www.mql5.com/en/articles/15616)

In this article, we create an MQL5 Expert Advisor that encodes chart screenshots as image data and sends them to a Telegram chat via HTTP requests. By integrating photo encoding and transmission, we enhance the existing MQL5-Telegram system with visual trading insights directly within Telegram.

![Integrating MQL5 with data processing packages (Part 2): Machine Learning and Predictive Analytics](https://c.mql5.com/2/89/logo-midjourney_image_15578_406_3921__2.png)[Integrating MQL5 with data processing packages (Part 2): Machine Learning and Predictive Analytics](https://www.mql5.com/en/articles/15578)

In our series on integrating MQL5 with data processing packages, we delve in to the powerful combination of machine learning and predictive analysis. We will explore how to seamlessly connect MQL5 with popular machine learning libraries, to enable sophisticated predictive models for financial markets.

[Launching MetaTrader VPS for the first time?Read our comprehensive, step-by-step instructions![](https://www.mql5.com/ff/sh/0xb0c8bjq5sadh89z2/01.png)Read](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/articles/13586&a=gxygkojxdwrcfbbgfrchvjgelflsnelu&s=49eab2fb45d89f59a191e88145774dcd7f9533039acb10dd9c28061b04fa92fe&uid=&ref=https://www.mql5.com/en/articles/14549&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5049260387859277865)

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