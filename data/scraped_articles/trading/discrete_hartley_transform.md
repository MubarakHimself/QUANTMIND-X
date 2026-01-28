---
title: Discrete Hartley transform
url: https://www.mql5.com/en/articles/12984
categories: Trading, Indicators
relevance_score: 0
scraped_at: 2026-01-24T13:30:28.698058
---

[![](https://www.mql5.com/ff/sh/5z040u47jcv59943z2/6c76c03a8b37e08b8655a1a085770b7a.jpg)\\
MetaTrader 5 for iOS and Android\\
\\
Fully featured platform for any devices and web browsers\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=ddonqpipxfqlnsvzlwuowsuwlejpyjxk&s=9daba65b69f40afc3c35f95b1f84ef5824d68c47f29ce96a6dc5b164a2727baa&uid=&ref=https://www.mql5.com/en/articles/12984&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5082911405083988291)

MetaTrader 5 / Trading


### Introduction

In 1942, [Ralph Hartley](https://en.wikipedia.org/wiki/Ralph_Hartley "https://en.wikipedia.org/wiki/Ralph_Hartley") proposed an analogue of the Fourier transform in his article " [A More Symmetrical Fourier Analysis Applied to Transmission Problems](https://www.mql5.com/go?link=https://gwern.net/doc/math/1942-hartley.pdf "https://gwern.net/doc/math/1942-hartley.pdf")".

Just like Fourier transform ( **_FT_**), Hartley transform ( **_HT_**) turns the original signal into a sum of trigonometric functions. But there is one significant difference between them. **_FT_** converts real values to [complex numbers](https://www.mql5.com/en/docs/basis/types/complex), while **_HT_** provides only real results. Because of this difference, the Hartley transform did not become popular - scientists and technicians did not see any advantages in it and continued to use the usual Fourier transform. In 1983, [Ronald Bracewell](https://en.wikipedia.org/wiki/Ronald_N._Bracewell "https://en.wikipedia.org/wiki/Ronald_N._Bracewell") presented a discrete version of the Hartley transform.

### A bit of theory

[Discrete Hartley transform](https://en.wikipedia.org/wiki/Discrete_Hartley_transform "https://en.wikipedia.org/wiki/Discrete_Hartley_transform") ( **_DHT_**) can be used in the analysis and processing of discrete time series. It allows filtering signals, analyzing their spectrum and much more. The capabilities of **_DHT_** are no less than those of the discrete Fourier transform. However, unlike DFT, **_DHT_** uses only real numbers, which makes it more convenient for implementation in practice, and the results of its application are more visual.

Let us have **_N_** real numbers **_h\[0\] … h\[N-1\]_**. Using the discrete Hartley transform, we obtain **_N_** real numbers **_H\[0\]…H\[N-1\]_** from them.

> ![](https://c.mql5.com/2/56/0.png)

This transformation allows us to transfer a signal from the time domain to the frequency domain. With its help, we can estimate how great the influence of a particular harmonic in the original signal is. **_H\[0\]_** number contains basic information about the signal. **_H\[1\]…H\[N-1\]_** numbers provide additional data. These numbers show how strong a particular harmonic is in the original signal. The index of these numbers shows how many cycles of this harmonic will fit in the original signal. In other words, the higher the index, the higher the harmonic frequency.

Inverse Hartley transform is used to move from the frequency domain to the time domain. Its equation looks like this.

> ![](https://c.mql5.com/2/56/0__1.png)

In both equations, the **_cas_** ( [cos](https://www.mql5.com/en/docs/math/mathcos) and [sin](https://www.mql5.com/en/docs/math/mathsin)) function represents the sum of trigonometric functions.

> ![](https://c.mql5.com/2/56/0__2.png)

Although, it can be replaced with a difference. The essence of the transformation will not change. Now, let's see how we can put **_DHT_** into practice.

### Introducing discrete Hartley transform

So, **_DHT_** converts a signal from the time domain to the frequency domain. But is it possible to get any practical benefit from this?

Let's take 15 **_Open_** prices as a signal. This is how the spectrum of this time series looks like ( **_H\[0\]_** is not displayed due to scale differences).

> ![](https://c.mql5.com/2/56/1__5.png)

The figure clearly shows that different harmonics have different strengths. But what can be done with this spectrum?

Let's take [simple moving average](https://www.mql5.com/en/docs/indicators/ima) as an example. Its equation is very simple.

> ![](https://c.mql5.com/2/56/0__3.png)

What will happen to the indicator if we set one of the prices to zero? It is unlikely we will get anything good. But in the frequency domain this is possible. We can zero out any number of harmonics.

This is what the original signal looks like.

> ![](https://c.mql5.com/2/56/2__5.png)

Now let's set all **_H\[1\] – H\[14\]_** harmonics to zero. Now we only have basic information about the original signal. Let's apply the inverse Hartley transform to this spectrum.

> ![](https://c.mql5.com/2/56/3__5.png)

Now remove the harmonics with the highest frequencies **_H\[10\] – H\[14\]_**. The Hartley transform will provide the following result.

> ![](https://c.mql5.com/2/56/4__4.png)

We have smoothed the original signal. Here is the first way to apply the Hartley transform in practice. First, we can smooth the time series in the frequency domain. After that, the obtained values are transferred to the input of the usual indicators. Let's take two moving averages as an example. One of them is applied to the price as usual (red line). The second one is applied to **_DHT_** values (blue line).

> ![](https://c.mql5.com/2/56/5__4.png)

Nullifying some harmonics is not the only way to handle the signal spectrum. We can attenuate all harmonics at once, for example by dividing them by a given number.

> ![](https://c.mql5.com/2/56/0__4.png)

Alternatively, we can attenuate each harmonic according to its frequency - the higher the frequency, the greater the attenuation.

> ![](https://c.mql5.com/2/56/0__5.png)

In any case, we will get a smoothed time series. This is what a signal looks like with harmonics attenuated by 2 times.

> ![](https://c.mql5.com/2/56/6__5.png)

Another way to handle the spectrum is to leave only the strongest harmonics. To do this, we will have to find the average of all harmonics.

> ![](https://c.mql5.com/2/56/0__6.png)

We will leave only those of them that exceed this average by their [absolute value](https://www.mql5.com/en/docs/math/mathabs).

> ![](https://c.mql5.com/2/56/0__7.png)

Then we will only have the main signal plus the strongest harmonics.

> ![](https://c.mql5.com/2/56/7__4.png)

Another signal processing option is that we can reverse the harmonic values. Then the reconstructed signal will be in the opposite phase.

> ![](https://c.mql5.com/2/56/0__8.png)

In this case, we will receive a mirror reflection of the signal - the upward trend is replaced by a downward one and vice versa. This approach can be useful when calculating support and resistance levels.

> ![](https://c.mql5.com/2/56/8__2.png)

All options for processing the spectrum of the source signal can be used either individually or in combination with each other. For example, you can first leave only the strongest harmonics, and then change their sign to the opposite. In this case, the result will show a possible countertrend impulse.

### Indicator with optimal spectrum

Any linear indicator is a set of coefficients. If we apply the Hartley transform to these coefficients, we obtain the spectral characteristic of the indicator.

We know that [SMA](https://www.mql5.com/en/docs/indicators/ima) is a low pass filter. Let's check this statement. All coefficients of this indicator are equal to **_1/N_**. The Hartley transform provides this indicator spectrum.

> ![](https://c.mql5.com/2/56/9__3.png)

As we can see, **_SMA_** passes only the main signal **_H\[0\]_**, but all other harmonics are completely suppressed.

Spectra (frequency characteristics of indicators) can differ greatly from each other. For example, **_LWMA_** passes all harmonics of the input signal.

> ![](https://c.mql5.com/2/56/10__1.png)

On the other hand, **_SMMA_** allows only some harmonics to pass through and suppresses the rest.

> ![](https://c.mql5.com/2/56/11__1.png)

Each indicator has its own unique spectrum. It can be used to handle the price series. To do this, we need to first find the spectrum of the original signal **_H\[\]_**. Then we multiply it term by term by the spectrum of the indicator **_I\[\]_**.

> ![](https://c.mql5.com/2/56/0__9.png)

The inverse Hartley transform is applied to the obtained result. As a result, we obtain signal filtering in the frequency domain. This is what the operation of a frequency filter based on the **_LWMA_** indicator looks like.

> ![](https://c.mql5.com/2/56/12__1.png)

But we can go the other way by first setting the spectrum of the indicator, and then getting its coefficients. Let's try to make an indicator whose spectral characteristic will correspond to the spectrum of the signal.

The algorithm will be as follows. First, we get the signal spectrum with **_DHT_**. Then we need to normalize it. To do this, we need to divide all harmonic values by **_D = H\[0\]_**.

> ![](https://c.mql5.com/2/56/0__10.png)

Note that after the normalization, **_H\[0\] = 1_** is a mandatory condition when constructing an indicator.

After that, we need to apply the inverse transformation, which will provide the weighting coefficients of the indicator.

> ![](https://c.mql5.com/2/56/13__1.png)

These coefficients are not very different from **_SMA_**. But such an indicator will have a smaller lag compared to the moving average, which will make it possible to more accurately track market dynamics.

### Noise and color

When processing financial time series, the term "noise" most often refers to unwanted signal distortion. To eliminate such noise, a variety of filters can be used, including **_SMA_**.

A random or unpredictable signal can also be considered a noise. What happens if we represent market price movements as a sum of noises? To do this, we need to turn to the concept of colored noises. [Colored noises](https://en.wikipedia.org/wiki/Colors_of_noise "https://en.wikipedia.org/wiki/Colors_of_noise") are noises that have different energies in different frequency ranges. They got their name from the different colors of visible light: red is low-frequency noise, while violet is high-frequency noise.

Representing price movements as a sum of colored noises can give interesting results when analyzing financial time series. This approach allows us to take into account various frequency components of price movement.

Each colored noise has its own unique characteristic, which reflects the distribution of energy in the spectrum depending on the **_f_** frequency.

> ![](https://c.mql5.com/2/56/0__11.png)

There are five primary colors of noise.

| p parameter | noise color |
| --- | --- |
| -2 | red |
| -1 | pink |
| 0 | white |
| +1 | blue |
| +2 | violet |

Each noise is associated with a specific price movement. For example, red noise can indicate the presence of long-term trends or cycles in price movements. White noise indicates that the market is in a flat state. Violet noise can indicate random and unpredictable price behavior. The use of colored noise in the analysis of financial time series can help to identify hidden patterns, as well as patterns that are not always visible in conventional analysis.

Now let's see how different noises behave in the market. To do this, we need to take a few simple steps.

First we need to find the spectrum of the signal, with which we can estimate the energy of each harmonic. To do this, we need to square the value of each harmonic.

> ![](https://c.mql5.com/2/56/0__12.png)

Now, we can estimate the value of the scaling factor for noise with the **_p_** parameter. When using this coefficient, the total energies of the signal and noise will be equal.

> ![](https://c.mql5.com/2/56/0__13.png)

Knowing this coefficient, we can construct the energy spectrum of **_EN\[\]_** noise.

> ![](https://c.mql5.com/2/56/0__14.png)

Now we can find the **_HN\[\]_** noise spectrum. To do this, we need to take the square root of the energy spectrum.

> ![](https://c.mql5.com/2/56/0__15.png)

There is very little left to do - assign +/- signs to the harmonics of the noise spectrum, just like the harmonics of the original signal. In this case, the noise and the original signal will be in the same phase.

> ![](https://c.mql5.com/2/56/0__16.png)

After this, we need to perform an inverse Hartley transform to get the noise values on the price chart. This is what red noise looks like in the market.

> ![](https://c.mql5.com/2/56/14__1.png)

But we can also take the opposite signs of the harmonics. In this case, the noise will be in the opposite phase with respect to the original signal. In addition, we can take strictly positive or negative harmonic values. We can afford this in the frequency domain. Then we will be able to see the boundaries of noise movements in the market.

> ![](https://c.mql5.com/2/56/15__3.png)

The concept of colored noise can be used not only to describe market dynamics, but also to develop indicators.

To do this, we first need to set the indicator power **_E > 0_**. This parameter determines how sensitive the indicator will be. After this, we carry out the already familiar procedures. First we find the scaling factor.

> ![](https://c.mql5.com/2/56/0__17.png)

After that, we find the spectrum of the **_HI\[\]_** indicator. Do not forget that **_HI\[0\]_** should be equal to 1.

> ![](https://c.mql5.com/2/56/0__18.png)

All we have to do is assign +/- signs to the harmonics of the indicator spectrum if necessary. Then we need to apply the inverse Hartley transform and obtain the indicator coefficients. This is what a red noise indicator looks like with different variants of harmonic signs.

> ![](https://c.mql5.com/2/56/16__1.png)

When developing indicators, we can use not only pure noise, but also their various combinations. For example, this is what the spectrum of the sum of red and violet noise looks like.

> ![](https://c.mql5.com/2/56/17.png)

This is what the difference between the spectra of red and white noise looks like.

> ![](https://c.mql5.com/2/56/18.png)

After receiving the **_coefficient\[\]_** indicator coefficients, we should normalize them. To do this, we first need to find the sum of all coefficients.

> ![](https://c.mql5.com/2/56/0__19.png)

After this, we need to divide each coefficient by the resulting amount.

> ![](https://c.mql5.com/2/56/0__20.png)

In addition to the main noises listed, there are others. For example, one definition of black noise is noise with the parameter **_p < -2_**. Let's try a different approach. Let us assume that the **_p_** noise parameter can take fractional values. Let's see how it changes over time.

To calculate the noise parameter, we first need to find the energy spectrum of the signal **_E\[\]_**. After this, we need to calculate four coefficients.

> ![](https://c.mql5.com/2/56/0__21.png)

> ![](https://c.mql5.com/2/56/0__22.png)

> ![](https://c.mql5.com/2/56/0__23.png)

> ![](https://c.mql5.com/2/56/0__24.png)

In all equations, **_k = 1…N-1_**. The most suitable noise parameter can be calculated using the equation.

> ![](https://c.mql5.com/2/56/0__25.png)

This is how this parameter changes in market conditions.

> ![](https://c.mql5.com/2/56/19.png)

As we can see, the use of colored noise in financial time series analysis can be a useful tool for exploring and understanding market dynamics. This approach can help reveal hidden patterns and improve forecasting of price movements.

### Conclusion

The discrete Hartley transform has its own fast transformation algorithms. But if we allocate the array for **_cas_** values beforehand, we can significantly speed up the data handling speed. The size of this array should be equal to **_(N-1)^2+1_**, where **_N_** is an indicator period. Then the values of this array are set as follows:

> ![](https://c.mql5.com/2/56/0__26.png)

I used exactly this approach in this article.

The following indicators are attached to the article.

| Symbol | Description |
| --- | --- |
| DHT | The indicator demonstrates the capabilities of processing signal harmonics.<br>- _**iPeriod**_ \- indicator period;<br>- **_CutOff_** \- number of harmonics of the original signal to be left. If CutOff=0, then all harmonics remain; <br>- **_Constant_** \- weakening a signal harmonics by a constant; <br>- **_Hyperbolic_** \- suppression of harmonics according to the hyperbolic law; <br>- _**Strong**_ \- preservation of the original signal phase; <br>- **_Inverse_** \- signal phase change; |
| DHT LWMA | The indicator shows price processing by the LWMA indicator in the spectral region |
| Spectrum | The indicator whose coefficients give a spectral characteristic similar to that of the original signal |
| Noise Levels | The indicator shows the levels of colored noises.<br>Noise - noise color<br>Type - change the phase of the original signal |
| Noise Indicator | The indicator selects coefficients corresponding to the selected noise color.<br>E - noise power. |
| Fractional Noise | The indicator displaying the fractional noise parameter. |

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/12984](https://www.mql5.com/ru/articles/12984)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/12984.zip "Download all attachments in the single ZIP archive")

[DHT.mq5](https://www.mql5.com/en/articles/download/12984/dht.mq5 "Download DHT.mq5")(7.08 KB)

[DHT\_LWMA.mq5](https://www.mql5.com/en/articles/download/12984/dht_lwma.mq5 "Download DHT_LWMA.mq5")(6.35 KB)

[Spectrum.mq5](https://www.mql5.com/en/articles/download/12984/spectrum.mq5 "Download Spectrum.mq5")(5.81 KB)

[Noise\_Levels.mq5](https://www.mql5.com/en/articles/download/12984/noise_levels.mq5 "Download Noise_Levels.mq5")(7.17 KB)

[Noise\_Indicator.mq5](https://www.mql5.com/en/articles/download/12984/noise_indicator.mq5 "Download Noise_Indicator.mq5")(9.12 KB)

[Fractional\_Noise.mq5](https://www.mql5.com/en/articles/download/12984/fractional_noise.mq5 "Download Fractional_Noise.mq5")(6.15 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [How can century-old functions update your trading strategies?](https://www.mql5.com/en/articles/17252)
- [Polynomial models in trading](https://www.mql5.com/en/articles/16779)
- [Trend criteria in trading](https://www.mql5.com/en/articles/16678)
- [Cycles and trading](https://www.mql5.com/en/articles/16494)
- [Cycles and Forex](https://www.mql5.com/en/articles/15614)
- [Practicing the development of trading strategies](https://www.mql5.com/en/articles/14494)
- [Angle-based operations for traders](https://www.mql5.com/en/articles/14326)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/455899)**
(4)


![Verner999](https://c.mql5.com/avatar/avatar_na2.png)

**[Verner999](https://www.mql5.com/en/users/verner999)**
\|
22 Jul 2023 at 21:30

Thanks for the interesting article! Some things look very promising. :)


![Aleksandr Shirin](https://c.mql5.com/avatar/2020/8/5F29BD03-87FF.jpg)

**[Aleksandr Shirin](https://www.mql5.com/en/users/withoutthetime)**
\|
29 Aug 2023 at 19:55

Really enjoyed the article. Very similar to what can be applied in practice. I downloaded the indicators, used them on different timeframes. So far, I got the impression that they give effective signals. I will test more.


![Aleksey Vyazmikin](https://c.mql5.com/avatar/2024/6/6678986f-2caa.png)

**[Aleksey Vyazmikin](https://www.mql5.com/en/users/-aleks-)**
\|
30 Aug 2023 at 10:15

Thanks for the article!

Just yesterday I was thinking about methods of identifying market stages through statistical indicators in a subsample, and I saw similar ideas in the article.

Have you done any deeper research in this direction? I wonder how fast (with what lag) and with what error it was possible to classify trends\\flats?

![Aleksej Poljakov](https://c.mql5.com/avatar/2017/5/591B60CC-05D2.png)

**[Aleksej Poljakov](https://www.mql5.com/en/users/aleksej1966)**
\|
30 Aug 2023 at 12:59

**Aleksey Vyazmikin [#](https://www.mql5.com/ru/forum/451141#comment_49039106):**

Thanks for the article!

Just yesterday I was thinking about methods of identifying market stages through statistical indicators in a subsample, and I saw similar ideas in the article.

Have you done any more in-depth research in this direction? I wonder how fast (with what lag) and with what error it was possible to classify trends\\flats?

Identifying market stages is probably the easiest task. We take the price spectrum (without the main signal) and pass it through cluster analysis (Kohonen maps) - that's what we get market stages. But everything is very complicated with trends - the sign of change/beginning of a new trend is relative weakness of the low-frequency component and relative strengthening of high-frequency harmonics. But, unfortunately, it is possible to seriously miss the trend direction.

![Structures in MQL5 and methods for printing their data](https://c.mql5.com/2/57/formatte_series_mqlformat-avatar.png)[Structures in MQL5 and methods for printing their data](https://www.mql5.com/en/articles/12900)

In this article we will look at the MqlDateTime, MqlTick, MqlRates and MqlBookInfo strutures, as well as methods for printing data from them. In order to print all the fields of a structure, there is a standard ArrayPrint() function, which displays the data contained in the array with the type of the handled structure in a convenient tabular format.

![Integrate Your Own LLM into EA (Part 1): Hardware and Environment Deployment](https://c.mql5.com/2/59/Hardware_icon_up__1.png)[Integrate Your Own LLM into EA (Part 1): Hardware and Environment Deployment](https://www.mql5.com/en/articles/13495)

With the rapid development of artificial intelligence today, language models (LLMs) are an important part of artificial intelligence, so we should think about how to integrate powerful LLMs into our algorithmic trading. For most people, it is difficult to fine-tune these powerful models according to their needs, deploy them locally, and then apply them to algorithmic trading. This series of articles will take a step-by-step approach to achieve this goal.

![Neural networks made easy (Part 40): Using Go-Explore on large amounts of data](https://c.mql5.com/2/54/neural_networks_go_explore_040_avatar.png)[Neural networks made easy (Part 40): Using Go-Explore on large amounts of data](https://www.mql5.com/en/articles/12584)

This article discusses the use of the Go-Explore algorithm over a long training period, since the random action selection strategy may not lead to a profitable pass as training time increases.

![Learn how to deal with date and time in MQL5](https://c.mql5.com/2/59/date_and_time_in_MQL5_logo__1.png)[Learn how to deal with date and time in MQL5](https://www.mql5.com/en/articles/13466)

A new article about a new important topic which is dealing with date and time. As traders or programmers of trading tools, it is very crucial to understand how to deal with these two aspects date and time very well and effectively. So, I will share some important information about how we can deal with date and time to create effective trading tools smoothly and simply without any complicity as much as I can.

[Best articles and CodeBase updates in MQL5.community channelsFollow us to ensure you never miss out on important updates![](https://www.mql5.com/ff/sh/n9yf51p2srwzfqh5z2/02.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=dgazvhktsxqakdvarucjbvmvzenwlyje&s=98a038fe082e458df8c4a1d8e116e3a6646fd5517f06e48b2356b7ee005817d6&uid=&ref=https://www.mql5.com/en/articles/12984&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5082911405083988291)

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