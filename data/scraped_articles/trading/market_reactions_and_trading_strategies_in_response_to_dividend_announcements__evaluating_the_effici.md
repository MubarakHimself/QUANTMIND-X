---
title: Market Reactions and Trading Strategies in Response to Dividend Announcements: Evaluating the Efficient Market Hypothesis in Stock Trading
url: https://www.mql5.com/en/articles/13850
categories: Trading, Trading Systems, Integration
relevance_score: 0
scraped_at: 2026-01-24T13:30:00.814924
---

[![](https://www.mql5.com/ff/sh/zf7a2k61x98jzh89z2/01.png)Speed up your tradingUse our high-speed VPS for MetaTrader 4 and 5Learn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=qtrrsuiwuicrscmckjynyanztbditglq&s=c617dc80d90cfd3783ec1345eec2b419b281f10fec6eac77b3218984ac337259&uid=&ref=https://www.mql5.com/en/articles/13850&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5082906195288658215)

MetaTrader 5 / Trading


### Abstract

This paper intends to analyze the impact of dividend announcement on the stock market returns that investors earn. In doing so it also aims to check the validity of Efficient Market Hypothesis in the Indian Stock Market. The basic idea is that if efficient market hypothesis holds and there is information efficiency in the market then any information on dividend announcement whether positive or negative should be reflected in the stock prices and market returns, and they should adjust in such a way that investors earn market return and there is no possibility of earning abnormal returns around the date of dividend announcement. For checking this claim companies under Nifty 50 are taken into consideration and they are divided into 3 groups depending on whether they announced positive, negative or no change in the dividend during the year 2022. Then for each company Abnormal Returns are calculated for an event window of 30 days (+/- 15 days of announcement of dividend) by subtracting market rate of return from return on individual company. Then these Abnormal Returns are averaged for each group to get Average Abnormal Returns(AAR) for each group. Also to check if there is any trend present in Abnormal Returns, Cumulative Average Abnormal Returns (CAAR) are being calculated for each group.

So if the efficient market hypothesis holds then these AAR should be around 0 for each group and CAAR should show no trend around the date of dividend announcement. But in this paper it is being found using graphical analysis and t-test that there is mostly positive AAR and CAAR shows increasing trend around dividend announcement date and vice versa for negative dividend announcement. So in this paper the result is that the efficient market hypothesis does not hold in case of Indian Stock Market and that it takes some time for the market to incorporate any new information and investors can take benefit of this information inefficiency.

### Introduction

From time to time, the stock market experiences dramatic price changes. This volatility in the price of stocks incorporates many factors which can majorly be grouped into macro factors and micro factors. Macro factors mainly takes into consideration the events that are economy wide and affects decision making, like change in oil prices, changes in tax policy by government, Monetary Policy, Exchange rate changes, political unrest and wars. In addition to these, Micro factors are the ones which are company specific and determines the movement in price of stock due to more investment or acquisitions, better sales to total asset ratio, and prominently, the dividend policy of the firm.

Dividend is a type of Free Cash Flow (FCF) from the company. FCF is an important indicator looked after by investors while making investment decisions. Dividend Signaling theory throws light on the fact that the announcement of dividend by the company is an indicator of the future growth prospects of the company. And as the future growth prospect becomes clear, ceteris paribus, the stock price of the firm is expected to grow. But there have been unusual citing in the history and some research studies also contradicts this view. IDFC Ltd. announced an interim dividend on 14 February, 2023 and saw a massive fall of 14 percent in their stock price. The major reason cited by investors was that the firm did not find an investment opportunity and hence its future profitability is bleak. Similarly, Thaler, R. H. et.al (2005) shows that dividend changes are negatively correlated with future profitability. In light of this finding, it is important to observe how well the markets react to announcements or changes in dividend policy by the firm.

Efficient Market Hypothesis (EMH) states that share price imbed in itself all market information and hence there is no scope for investors to make any benefit by purchasing an undervalued asset and reselling it at a higher price. But there are nearly 4.8 million active traders in the stock market, which hints at the fact that there is still scope to make profit by trading in the stock market. Modern Portfolio theory sheds light on the fact that markets aren’t fully efficient. Researchers have also come out with the view that the Efficient Market Hypothesis doesn’t hold in every scenario and Behavioral choices are coming into play. Therefore, new variants of EMH have been introduced into the vocabulary.

Therefore, this paper tries to assess how the Indian stock market reacts to announcements of dividends by firms during the year 2022, and does Efficient Market Hypothesis hold in the Indian context. This research interest emerges due to the ambiguity shown by the previous researches in this field. The Indian stock market is studied because India is an emerging market and has two well developed exchanges, Bombay Stock Exchange (BSE) and National Stock Exchange (NSE) which is well looked after by SEBI, which ensures safety to investors. Year of interest is taken to be 2022 because the Indian economy started showing a revival trend in mid-2021, after COVID 19, and this revival is expected to be reflected in the firm's activities in the upcoming year.

### Literature Review

Previous research carried out in this area majorly focuses on the possible abnormal return from dividend announcement and how long does the impact stay. Studies have been carried out in different countries which incorporate stock markets which are in different stages of development.

Legenzova, R. et. al (2017) studied the effect of dividend announcement on the Baltic Stock market. They studied the effect of three types of dividends, regular dividend, dividend paid due to fall in nominal value of stock, and dividend paid due to reduction in book value of stock. The study was carried out for the period 2010-2015, and results showed that the Average Abnormal Returns (AARs) were positive around the dividend announcement period and AR can be realized for 3-7 days after the announcement, emphasizing on the fact that there is weak efficiency and this result is statistically significant.

Dharmarathne D. (2020) assessed the impact of dividend announcement on Colombo Stock Exchange. The study considers 27 stock dividends from 2004-2014 and this study enlightens the event study method by incorporating stock volatility clustering phenomenon to the Market Model along with time series modeling techniques. Results show that there is positive AR due to dividend announcement and this window is open for a few days, hence citing that the stock market doesn’t incorporate this information quickly and hence, semi strong form of EMH doesn’t hold in Colombo Stock Exchange. Study further puts light on the aspect of information leakage because the Cumulative Average Abnormal Returns (CAAR) shows a hike even a day before the announcement of dividend.

Ghada Abbass (2015) carried out a similar study in Damascus Securities Exchange for period 2011-2014, considering 18 dividend announcements from 11 companies. The study is carried out using event study methodology keeping an event window of 40 days. There have been more positive reactions than negative during the event window. The findings show that there is no abnormal return before the announcement day, implying no informational leakage and further that the market takes 6-15 days to adjust back to normal, opening up a window of 9 days to make abnormal returns after the announcement day. The main paper studying the effect of dividend announcement on the Indian stock market and the assessment of the semi strong form of EMH is by T. Manjunath and T. Mallikarjunappa. The study is carried out for the year 2002 by keeping an event window of 60 days, based on the dividend announcements of 149 companies which are part of the BSE-200 Index. The study shows that AAR does not approximate to zero and CAAR shows wide fluctuations, indicating the scope to make abnormal returns even for 24 days after the announcement of dividend. Paper concludes that the semi strong form of EMH doesn’t hold in the Indian market.

Another detailed study on the Indian market is carried out by I. Berezinets et. al. (2015). The study is carried out for Group A companies listed under BSE 200 for the period of 2010-2012. The event study window was 31 days and the research was bifurcated into three subparts studying the impact of positive, negative and neutral dividend announcement on the stock market. The analysis showed that absolute CAAR in case of decreased dividend is three times higher than increased dividend which implies that the market reacts more to negative announcements. Hence the paper cites the firms to carry out a policy of positive dividend as compared to the previous period so as to ensure investor interest.

### Methodology

The study of the Indian stock market’s reaction to dividend announcements by firms is studied using the event study methodology. An event study examines the impact of an event on the financial performance of a security, such as company stock. For this particular study, the date of dividend announcement is taken as the event day. An event window of 31 days surrounding the event day is considered for the study.

The sample consists of the firms included in NIFTY 50 who have announced dividends in the year 2022. A total of 43 firms out of the list of 50 have announced a dividend, furthermore, only final dividend announcement have been considered. The study caters to studying the differential effect to stock prices due to increased, decreased or neutral dividend as compared to the last dividend announcement by the firm. Hence the lot of firms have been grouped into 3, increased dividend consisting of 29 firms, decreased dividend consisting of 5 firms and neutral dividend consisting of 9 firms.

For all the groupings and, using closing prices for +15 & -15 days of the announcement of dividend, daily returns have been calculated. The procedure followed for the same is to subtract the previous day price from the current day price and then dividing the whole by the previous day price.

![](https://c.mql5.com/2/61/455030812267.png)

Here, P(t) represents the closing price of the stock at time t, and P(t-1) represents the closing price of the stock at time t-1. The fraction represents the change in price from the previous day to the current day, divided by the previous day's price, which gives the daily return as a proportion.

For the same event window (-15 to +15 days), the returns from the market index are calculated using the same procedure. NSE 500 is used as the market index. Then these daily returns are used to calculate abnormal returns by subtracting market rate of return from rate of return of individual stocks.

![](https://c.mql5.com/2/61/5530276703447.png)

where R\_i is the return on the individual stock and R\_m is market rate of return.

By using graphical and analytical methods we intend to see if there is pattern or trend in abnormal returns around dividend announcement date.

Now we intend to calculate Average Abnormal Returns (AAR), calculated for each day. For each day, starting T-15 to T+15, AAR has been calculated by summing the AR by each firm for the particular day and dividing by the number of firms in a particular group.

![](https://c.mql5.com/2/61/1960415025060.png)

Finally, Cumulative Average Abnormal Return (CAAR) will have been calculated as the running total of AARs. This calculation will be done differently for all 3 categories.

![](https://c.mql5.com/2/61/3916757353627.png)

If significant abnormal returns are present around the announcement date then we can conclude that there is information inefficiency in the market and that efficient market hypothesis does not hold.To test the significance of pattern followed by AAR and CAAR, t- test will be followed.

T - Statistic for checking significance of results:

![](https://c.mql5.com/2/61/5119719251907.png)

![](https://c.mql5.com/2/61/17062269584.png)

For graphical analysis we have used line charts which will clearly indicate the path followed by AAR and CAAR during the event window. Additionally, the next section tabulates the values of AAR and CAAR for all three groups.

The data have been taken from various sources:

- The data for the closing prices of each stock is drawn from the BSE website which is used in calculating returns for each stock of Nifty 50.
- NSE 500 acts as a proxy for market index.

**Results:**

1. **Negative Dividend Announcement:**

![](https://c.mql5.com/2/61/2280099227660.png)



![](https://c.mql5.com/2/61/3813987026492.png)

The graph of AAR for negative dividend announcements shows that around the announcement date, AAR is mostly negative implying that investors are earning less than the market return. This implies that a negative dividend announcement leads to investor’s loss of confidence and hence a fall in stock price as compared to the proxy market index - NSE 500.

This is further substantiated by the downward trend of Cumulative Average Abnormal Return (CAAR). The graph of CAAR presents that after the announcement of a decreased dividend, there is steep decline in CAAR in period t+1 and it continues till about period t+10. This implies that the effect of a negative dividend persists till a long time, atleast for 10 days and hence the market isn’t adapting back to its normal state by incorporating the dividend announcement information. After T+10, there is some increase in CAAR. This implies that there is information inefficiency in the market i.e. stock prices are not incorporating information of negative dividend announcement so that investors will earn market rate of return. Thus the efficient market hypothesis does not hold in this case.

![](https://c.mql5.com/2/61/4865155758248.png)

From the result of T -test we can see that most of the T ratios were insignificant. The critical T value is taken as +/- 1.96 which is approximate 2 tailed T critical value for 5% significant level. In table most T values are below this critical threshold implying our results are insignificant and that efficient market hypothesis does not hold.

2. **Neutral Dividend Announcement:**

The graph presented shows the AAR and CAAR due to a neutral dividend, that is, the same dividend being announced in 2022 by the firm as the dividend announced in the previous term. The trend of AAR is fluctuating, and shows a slight increase on the day of announcement and the day after. To further get a glance of further results, CAAR graph presents a better view. The CAAR fluctuates around zero level, by increasing in some period and decreasing in others, as a whole, not giving the CAAR to move in a particular direction, hence not showing a particular pattern. CAAR shows that there is an increase in returns starting from the date of announcement of dividend which starts falling after 3 days and again picks up in 4th day.

![](https://c.mql5.com/2/61/1508871825367.png)

![](https://c.mql5.com/2/61/2705745105925.png)

The graph of AAR for companies announcing no change in the dividend can clearly seen to be fluctuating around 0 within a band of +/- 0.01 with sometime being positive and sometime being negative. Correspondingly there is no upward or downward trend seen in the graph of CAAR for these companies. This implies that announcement of no dividend change does not lead to earning of positive or negative abnormal returns and that investors earn market return which is as expected.

3. **Increased Dividend Announcement:**

Increased dividend means a positive dividend as compared to the dividend announced in the previous period. The graphical analysis for this group presents a picture where AAR continues to show a fluctuating pattern as the previous groups. AAR shows a sudden increase on the day of announcement.

![](https://c.mql5.com/2/61/5614081812069.png)

![](https://c.mql5.com/2/61/121952025063.png)

CAAR presents a better picture with the returns showing a continues upward trend, majorly picking up from day T-5, followed by a peak on the day of announcement of dividend. There is continuous rise of abnormal returns showing that the market does not adjust quickly to information on dividend announcement, hence opening up an option for the investors to make profit out of the sudden increase in price of the traded stock, as opposed to the proxy market index (here taken as NSE 500). This implies the presence of inefficiency tin the market and further negates our study interest of presence of EMH in the Indian Stock market.

![](https://c.mql5.com/2/61/143226883106.png)





From the result of T -test we can see that most of the T ratios were insignificant.In table most T values are below this critical threshold of +/-1.96 implying our results are insignificant and that efficient market hypothesis does not hold in this case also.


So we saw from the above analysis that inverstors can earn abnormal returns around the date of dividend announcement for some days. In view of this the following strategy is useful, If an investor believes that any company is going to announce increase in dividend then he can buy stocks of that share before announcement and can expect to earn positive abnormal returns for some time as price of that stock is likely to increase relative to price of market index around the date of dividend announcement.

Similarly, if investors believes that company is going to announe decrease in dividend then he can sell stocks before announcement to avoid potential losses as price of stocks is likely to decrease relative to price of market index around the date of dividend announcement. Also we got to know that these abnormal returns are likely to be present for around +/- 10 days of dividend announcement. So, By following this strategy investors can earn better return than by investing in broad market indices.This is likely to be a profitable strategy for investors and they can earn abnormal returns for some time by following this strategy as shown by our analysis.

### Conclusion

This paper intended to analyze the impact of dividend announcement on the stock market returns that investors earn. In doing so we also aimed to check  the validity of Efficient Market Hypothesis in the Indian Stock Market.

The results show that all three groups of dividend announcements show that the market doesn't follow Efficient market Hypothesis since AAR doesnt follow a trend of being around the level of zero and CAAR of every group shows a pattern around the date of announcement of dividend. Additionally, there is a presence of inefficiency and investors had a chance to make abnormal profit even after 5 days of announcement of dividend. All these factors cater to the fact that the stock market does not immediately take into account the change in value due to micro shock (i.e) dividend announcement.

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Custom Debugging and Profiling Tools for MQL5 Development (Part I): Advanced Logging](https://www.mql5.com/en/articles/17933)
- [Building a Custom Market Regime Detection System in MQL5 (Part 2): Expert Advisor](https://www.mql5.com/en/articles/17781)
- [Building a Custom Market Regime Detection System in MQL5 (Part 1): Indicator](https://www.mql5.com/en/articles/17737)
- [Advanced Memory Management and Optimization Techniques in MQL5](https://www.mql5.com/en/articles/17693)
- [Mastering JSON: Create Your Own JSON Reader from Scratch in MQL5](https://www.mql5.com/en/articles/16791)
- [Mastering File Operations in MQL5: From Basic I/O to Building a Custom CSV Reader](https://www.mql5.com/en/articles/16614)
- [Modified Grid-Hedge EA in MQL5 (Part IV): Optimizing Simple Grid Strategy (I)](https://www.mql5.com/en/articles/14518)

**[Go to discussion](https://www.mql5.com/en/forum/458476)**

![Data label for time series mining (Part 4)：Interpretability Decomposition Using Label Data](https://c.mql5.com/2/61/Data_label_for_time_series_mining_nPart_45Interpretability_Decomposition_Using_Label_Data_LOGO.png)[Data label for time series mining (Part 4)：Interpretability Decomposition Using Label Data](https://www.mql5.com/en/articles/13218)

This series of articles introduces several time series labeling methods, which can create data that meets most artificial intelligence models, and targeted data labeling according to needs can make the trained artificial intelligence model more in line with the expected design, improve the accuracy of our model, and even help the model make a qualitative leap!

![MQL5 Wizard Techniques you should know (Part 08): Perceptrons](https://c.mql5.com/2/61/MQL5_Wizard_Techniques_you_should_know_xPart_08c_Perceptrons_LOGO.png)[MQL5 Wizard Techniques you should know (Part 08): Perceptrons](https://www.mql5.com/en/articles/13832)

Perceptrons, single hidden layer networks, can be a good segue for anyone familiar with basic automated trading and is looking to dip into neural networks. We take a step by step look at how this could be realized in a signal class assembly that is part of the MQL5 Wizard classes for expert advisors.

![Design Patterns in software development and MQL5 (Part 3): Behavioral Patterns 1](https://c.mql5.com/2/61/Design_Patterns_yPart_39_Behavioral_Patterns_1__LOGO.png)[Design Patterns in software development and MQL5 (Part 3): Behavioral Patterns 1](https://www.mql5.com/en/articles/13796)

A new article from Design Patterns articles and we will take a look at one of its types which is behavioral patterns to understand how we can build communication methods between created objects effectively. By completing these Behavior patterns we will be able to understand how we can create and build a reusable, extendable, tested software.

![Neural networks made easy (Part 53): Reward decomposition](https://c.mql5.com/2/57/decomposition_of_remuneration_053_avatar.png)[Neural networks made easy (Part 53): Reward decomposition](https://www.mql5.com/en/articles/13098)

We have already talked more than once about the importance of correctly selecting the reward function, which we use to stimulate the desired behavior of the Agent by adding rewards or penalties for individual actions. But the question remains open about the decryption of our signals by the Agent. In this article, we will talk about reward decomposition in terms of transmitting individual signals to the trained Agent.

[![](https://www.mql5.com/ff/si/x6w0dk14xy0tf97n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F586%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dhow.test.expert%26utm_content%3Dbuy.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=repptjucjbnrxhoeoqbekpbncvsnhylz&s=3da978a0c510a6306b46ee79cdf8418a5c0da5e081f296e18b262b00031a2310&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=qsmuuqyhkoznxhfkdhizilztekgxpigr&ssn=1769250599968349056&ssn_dr=0&ssn_sr=0&fv_date=1769250599&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F13850&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Market%20Reactions%20and%20Trading%20Strategies%20in%20Response%20to%20Dividend%20Announcements%3A%20Evaluating%20the%20Efficient%20Market%20Hypothesis%20in%20Stock%20Trading%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176925059996771234&fz_uniq=5082906195288658215&sv=2552)

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