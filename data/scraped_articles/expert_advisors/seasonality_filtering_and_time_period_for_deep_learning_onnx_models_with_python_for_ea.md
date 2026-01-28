---
title: Seasonality Filtering and time period for Deep Learning ONNX models with python for EA
url: https://www.mql5.com/en/articles/14424
categories: Expert Advisors, Machine Learning
relevance_score: 3
scraped_at: 2026-01-23T21:24:44.598376
---

[![](https://www.mql5.com/ff/sh/6xjc81sb5f2g45z9z2/01.png)Follow MQL5.community on social mediaWe publish the best technical materials from experts – free from advertising and irrelevant contentLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=yexgeaiatphxecqagtoxizolvboismyb&s=4e531fd1f983c26570e2dac7588b735354f2f9e0aea561427c030e4a1d2f060b&uid=&ref=https://www.mql5.com/en/articles/14424&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5071812608295644905)

MetaTrader 5 / Expert Advisors


### Introduction

After reading the article: [Benefiting from Forex market seasonality](https://www.mql5.com/en/articles/12996), I decided to create another paper to compare an EA with and without seasonality to see if it can benefit.

I already knew that markets were influenced by seasonal factors. This was clear when I learned that Mark Zuckerberg funded Facebook with money from an investor. This investor had previously invested money from his Bar Mitzvah in oil stocks, predicting a rise due to expected hurricanes in the Caribbean. He had analyzed weather forecasts that indicated upcoming severe weather during that period.

I'm very proud and interested in writing this article, which aims at exploiting the idea that market and seasonality are good companions. A good approach to making this come true would be to merge both EAs in one, but we already have an article about that, here's the link: [An example on how to ensemble ONNX models in mql5](https://www.mql5.com/en/articles/12433).

First of all, we will compare models with and without filtering with using an EA, to see how filtering data affects or not, and, after this, we will discuss seasoning with a graph, to end up with a real case of study, for February 2024, with and without seasoning. In the last part of the article (which I find very interesting), we will discuss other approaches to the EA we already have from the article: [How to use ONNX models in MQL5](https://www.mql5.com/en/articles/12373), and we will see if we can benefit from fine-tuning those EAs and ONNX models. I will tell you right now that the answer is yes, we can.

To make this happen, we will first download the data (all ticks) with this script: [Download all data from a symbol](https://www.mql5.com/en/market/product/111572?source=Site+Market+My+Products+Page). We just have to add the script to the symbol chart we need to study, and, in some time (less than an hour) we will have downloaded all the historic ticks from that symbol in our Files folder.

When we have all the ticks downloaded, we will work with that csv and extract only the data we need or want (in this case, the periods from January 2015 to February to 2023).

### Seasonality

Seasonality in trading is all about spotting the regular ebbs and flows in asset prices that happen predictably throughout the year. It's like recognizing that certain stocks tend to do better at certain times than others. Let's unpack this idea a bit.

Understanding Seasonality in Trading:

- Definition: Seasonality means noticing how prices tend to fluctuate in a recurring pattern based on the time of year. This could be tied to actual seasons (like summer or winter), commercial seasons (such as the holiday rush), or even specific months.
- Examples: Smart investors keep an eye on these patterns because they're often reliable and profitable. Here are a few examples:
  - Weather-Related Seasonality: Just like the weather affects farming seasons, it also impacts things like commodity prices and related stocks. For instance, a company selling beach gear might see a boost in sales during summer but a dip in colder months, affecting its stock.
  - Holiday Seasonality: Retail stocks often see a bump during holiday shopping frenzies. Companies that thrive on holiday sales, like gift shops, tend to shine during these times.
  - Quarterly Earnings Seasonality: Publicly traded companies report earnings every quarter, and their stock prices can react predictably during these seasons.
  - Tax Seasonality: Tax-related events can shake up the market, especially for sectors tied to finance.
  - Natural Cycles: Industries like tourism or energy have their own seasonal demand patterns, like summer vacations or winter heating needs.

Trading Strategies Based on Seasonality:

- Traders can leverage seasonality in a few ways:
  - Identifying Seasonal Patterns: Digging into past data to spot trends that repeat at certain times of the year.
  - Timing Trades: Making moves in and out of positions based on these seasonal trends.
  - Managing Risk: Adjusting how much risk you're taking on during volatile periods.
  - Sector Rotation: Switching up investments between sectors that tend to perform better at different times of the year.

**Filtering Data**

We will use the Low-pass filter. According to Wikipedia:

A low-pass filter is a filter that passes signals with a frequency lower than a selected cutoff frequency and attenuates signals with frequencies higher than the cutoff frequency. The exact frequency response of the filter depends on the filter design. The filter is sometimes called a high-cut filter, or treble-cut filter in audio applications. A low-pass filter is the complement of a high-pass filter.

Why do we choose low-pass filters over high-pass filters in algorithmic trading? In algorithmic trading, the preference for low-pass filters stems from several key advantages:

1. Signal Smoothing: Low-pass filters effectively smooth out noisy price movements, emphasizing longer-term trends over short-term fluctuations.
2. Reducing High-Frequency Noise: They help attenuate high-frequency noise, which may not provide meaningful information for trading strategies.
3. Lower Transaction Costs: By focusing on longer-term trends, low-pass filters can lead to fewer, more strategic trades, potentially reducing transaction expenses.
4. Better Risk Management: Low-pass filters contribute to a more stable and predictable trading strategy, reducing the impact of short-term market fluctuations.
5. Alignment with Investment Horizon: They are well-suited for strategies with longer-term investment horizons, capturing trends over extended periods effectively.

I personally use low-pass filter to filter high frequencies. It does not have much sense to use a high-pass filter here.

This is what we will use (note: I ended up changing the order and cutoff\_frequency parameters in the last part of the article to 0.1 cutoff and order equal to 1, because they ended up giving better results. Also, the correct .py to filtering are the ones of the last part of the article (there I used not only better parameters, I also used minmaxscaler to fit and inverse).

```
# Low-pass filter parameters
cutoff_frequency = 0.01  # Cutoff frequency as a proportion of the Nyquist frequency
order = 4

# Apply the low-pass filter
def butter_lowpass_filter(data, cutoff, fs, order):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    print("Filter coefficients - b:", b)
    print("Filter coefficients - a:", a)
    y = lfilter(b, a, data)
    return y

filtered_data_low = butter_lowpass_filter(df2['close'], cutoff_frequency, fs=1, order=order)
```

We will make use of "onnx\_LSTM\_simple\_filtered.py" and "onnx\_LSTM\_simple\_not\_filtered.py" to make the ONNX models and compare them.

Note: I've used model v1 and model v2 that have different Low-pass filter parameters.

Here are the results:

We will use the same inputs for the EAs

![inputs](https://c.mql5.com/2/72/Imagen2.jpg)

![inputs2](https://c.mql5.com/2/72/Imagen3.jpg)

The time period to study, will be from the first of February to the first of March.

For the non-filtered model:

```
RMSE         : 0.0010798043714784716
MSE          : 1.165977480664017e-06
R2 score     : 0.8799146678247277
```

![Not filtered](https://c.mql5.com/2/72/not_filtering.png)

Filtered v1

```
# Parámetros del filtro pasa bajo

cutoff_frequency = 0.01  # Frecuencia de corte en proporción a la frecuencia de Nyquist

order = 4
```

```
RMSE         : 0.0010228999869332884
MSE          : 1.0463243832681218e-06
R2 score     : 0.8922378749062259
```

![filtered v1](https://c.mql5.com/2/72/filtered_v1.png)

Filtered v2

```
cutoff_frequency = 0.1  # Frecuencia de corte en proporción a la frecuencia de Nyquist

order = 2
```

```
RMSE         : 0.0010899163515744447
MSE          : 1.1879176534293484e-06
R2 score     : 0.8775550550819025
```

![filtered v2](https://c.mql5.com/2/72/filtered_v2.png)

Conclusion over filtering

It yields better results only when using the correct parameters.

So, yes. It's convenient to use filtering.

Code used and models:

- [ONNX.eurusd.H1.120.Prediction\_FILTERED.mq5](https://www.mql5.com/en/articles/download/14424/146140/onnx.eurusd.h1.120.prediction_filtered.mq5 ".mq5")
- [ONNX.eurusd.H1.120.Prediction\_FILTERED\_v2.mq5](https://www.mql5.com/en/articles/download/14424/146162/onnx.eurusd.h1.120.prediction_filtered_v2.mq5 "Download ONNX.eurusd.H1.120.Prediction_FILTERED_v2.mq5") [ONNX.eurusd.H1.120.Prediction\_NOT\_FILTERED.mq5](https://www.mql5.com/en/articles/download/14424/146162/onnx.eurusd.h1.120.prediction_not_filtered.mq5 "Download ONNX.eurusd.H1.120.Prediction_NOT_FILTERED.mq5")
- [onnx\_LSTM\_simple\_EURUSD\_filtered.py](https://www.mql5.com/en/articles/download/14424/146140/onnx_lstm_simple_eurusd_filtered.py ".py") [onnx\_LSTM\_simple\_EURUSD\_not\_filtered.py](https://www.mql5.com/en/articles/download/14424/146140/onnx_lstm_simple_eurusd_not_filtered.py ".py")
- [Download all data from a symbol](https://www.mql5.com/en/market/product/111572?source=Site+Market+My+Products+Page) [EURUSD\_LSTM\_120\_1h\_not\_filtered.onnx](https://www.mql5.com/en/articles/download/14424/146159/eurusd_lstm_120_1h_not_filtered.onnx ".onnx") [EURUSD\_LSTM\_120\_1h\_filtered\_v1.onnx](https://www.mql5.com/en/articles/download/14424/146159/eurusd_lstm_120_1h_filtered_v1.onnx ".onnx") [EURUSD\_LSTM\_120\_1h\_filtered\_v2.onnx](https://www.mql5.com/en/articles/download/14424/146159/eurusd_lstm_120_1h_filtered_v2.onnx ".onnx")

### Are symbols seasonal?

For this part, we will first see this in a graph, we will get the data from February since 2015 to 2023 and we will add the data to see how it moves around those weeks.

This is what we can see from that period:

![Sum of febrary](https://c.mql5.com/2/72/Figure_1.png)

Conclusion:

We can see it has some tendencies, or at least we don't see a black (sum line) horizontal line. There are gaps between each line, because the symbol trades over the year and it prices fluctuate. This is why in the next part of the article, we will concatenate all the symbols by years for February and this is why we need to use a filter, so it doesn't pass the high frequency from, for example, the last date of 2022 to the first of February 2023 and when the AI gets trained of, for example, the close of a Friday and the open of a Monday, so, it doesn't study those changes and seeks a more smoothed data.

Scripts and data used:

- [Download all data from a symbol](https://www.mql5.com/en/market/product/111572?source=Site+Market+My+Products+Page) [Download\_seasonalities\_1h\_v2.py](https://www.mql5.com/en/articles/download/14424/146153/download_seasonalities_1h_v2.py "Download Download_seasonalities_1h_v2.py")
- [data febs.zip](https://www.mql5.com/en/articles/download/14424/data_febs.zip "Download data febs.zip")
- [draw\_with\_sum.py](https://www.mql5.com/en/articles/download/14424/draw_with_sum.py "Download draw_with_sum.py")

### Is this data from the symbol correlated?

The autocorrelation is a characteristic of the data that shows the degree of similarity between values in successive time intervals.

A value near 1 indicates that there is a big positive correlation.

Below are the results we obtained with [autocorrelation.py](https://www.mql5.com/en/articles/download/14424/autocorrelation.py "Download autocorrelation.py")

```
[1.         0.99736147 0.99472432 0.99206626 0.98937664 0.98671649\
 0.98405706 0.98144222 0.9787753  0.97615525 0.97356318 0.97099777\
 0.96848029 0.96602671 0.96360361 0.96113539 0.95865344 0.95615626\
 0.95362417 0.95108177 0.94854957 0.94599045 0.94346076 0.94091564\
 0.93837742 0.93583734 0.9332909  0.93074655 0.92826504 0.92579028\
 0.92330505 0.92084645 0.91834403 0.91581296 0.91328091 0.91076099\
 0.90826447]
```

### Creating a ONNX model for February seasons

For this task, we just have to concatenate all data into one single csv and make a model with it.

We will use the created concat seasonal.py to make one single csv, that is added to the zip seasonal\_feb\_contac. With the onnx\_LSTM\_...\_seasonals.py we will train and create the model.

Scripts and data used (all attached below):

- [seasonal\_feb\_concat.zip](https://www.mql5.com/en/articles/download/14424/seasonal_feb_concat.zip "Download seasonal_feb_concat.zip") [create\_concat\_seasonal\_data.py](https://www.mql5.com/en/articles/download/14424/create_concat_seasonal_data.py "Download create_concat_seasonal_data.py")
- [onnx\_LSTM\_simple\_EURUSD\_concat\_seasonals.py](https://www.mql5.com/en/articles/download/14424/onnx_lstm_simple_eurusd_concat_seasonals.py "Download onnx_LSTM_simple_EURUSD_concat_seasonals.py")

### Testing results for the seasonal model and comparing it to the 120 days (1h) filtered model

Seasonal model

```
RMSE         : 0.013137568368684325
MSE          : 0.00017259570264185493
R2 score     : 0.7166764010650979
```

Although not astonishing, the results look overall good (it doesn't have many negative Sharpe results)

![Sharpe](https://c.mql5.com/2/72/TesterOptgraphReport2024.03.10.3r1j.png)

![sharpe2](https://c.mql5.com/2/72/TesterOptgraphReport2024.03.10.2.png)

![seasonal optimization](https://c.mql5.com/2/72/seasonal.png)

If we compare it to the filtered model,

![Sharpe filtered](https://c.mql5.com/2/72/TesterOptgraphReport2024.03.10.3q2t.png)

![Sharpe filtered 2d](https://c.mql5.com/2/72/TesterOptgraphReport2024.03.10.2r2t.png)

![filtered optimization](https://c.mql5.com/2/72/filtered.png)

What I've found curious, is that the number of negative Sharpe values occupies half of the table from the optimization for the filtered model, while the negative ones for the seasonal model occupies around a fifth of the table. This is notable, because even having a lower r2, it seems to be a robust model giving profitable returns.

I could have also tested the EA without SL and TP, but I find a better practice to always use them in EAs.

Code and ONNX model used (all attached below):

[Download ONNX.eurusd.H1.120.Prediction_seasonal.mq5](https://www.mql5.com/en/articles/download/14424/146164/onnx.eurusd.h1.120.prediction_seasonal.mq5 "Download ONNX.eurusd.H1.120.Prediction_seasonal.mq5") [Download ONNX.eurusd.H1.120.Prediction_seasonal.mq5](https://www.mql5.com/en/articles/download/14424/146164/onnx.eurusd.h1.120.prediction_seasonal.mq5 "Download ONNX.eurusd.H1.120.Prediction_seasonal.mq5") [ONNX.eurusd.H1.120.Prediction\_seasonal.mq5](https://www.mql5.com/en/articles/download/14424/146171/onnx.eurusd.h1.120.prediction_seasonal.mq5 "Download ONNX.eurusd.H1.120.Prediction_seasonal.mq5") [EURUSD\_LSTM\_270\_1h\_filtered\_seasonal0.72.onnx](https://www.mql5.com/en/articles/download/14424/eurusd_lstm_270_1h_filtered_seasonal0.72.onnx "Download EURUSD_LSTM_270_1h_filtered_seasonal0.72.onnx") [onnx\_LSTM\_simple\_EURUSD\_concat\_seasonals.py](https://www.mql5.com/en/articles/download/14424/onnx_lstm_simple_eurusd_concat_seasonals.py "Download onnx_LSTM_simple_EURUSD_concat_seasonals.py")

### What time period to use?

For this part of the article, I've fine-tuned the filter to have better results for EURUSD

```
cutoff_frequency = 0.1  # Frecuencia de corte en proporción a la frecuencia de Nyquist
order = 1
```

I also modified the filter:

```
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import butter, lfilter

# Parámetros del filtro pasa bajo
cutoff_frequency = 0.1  # Frecuencia de corte en proporción a la frecuencia de Nyquist
order = 1

# Aplicar filtro pasa bajo
def butter_lowpass_filter(data, cutoff, fs, order):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    print("Coeficientes del filtro - b:", b)
    print("Coeficientes del filtro - a:", a)
    y = lfilter(b, a, data)
    return y

scaled = MinMaxScaler(feature_range=(0,1))

valores_df1 = df.filter(['close']).values
valores_df1 = pd.DataFrame(valores_df1)
x_features1 = valores_df1[[0]]
valores_df2 = x_features1.values

data_escalada = scaled.fit_transform(valores_df2)

print(data_escalada)

filtered_data_low = butter_lowpass_filter(data_escalada, cutoff_frequency, fs=1, order=order)
print("filtered_data_low",filtered_data_low)

filtered_data_low_unscaled = scaled.inverse_transform(filtered_data_low)
```

The order must be rounded to an integer price.

This strategy is tested over a month for the 1-hour interval, specifically in February 2024. For the 30-minute interval, testing is conducted from February 1st to February 15th, and so on.

For the 15-minute interval, I conducted tests with and without filters. The results indicate that using a filter with fine-tuning gives better results (at least overall).

15 min with filter (Sharpe)

![tests 15 min with filter](https://c.mql5.com/2/72/TesterOptgraphReport2024.03.12.png)

![tester 2d 15 min](https://c.mql5.com/2/72/TesterOptgraphReport2024.03.12.2.png)

![testing 15min with filter](https://c.mql5.com/2/72/testing.png)

[LTSM\_simple\_15m\_filtrado.py](https://www.mql5.com/en/articles/download/14424/ltsm_simple_15m_filtrado.py "Download LTSM_simple_15m_filtrado.py") [LSTM.15m.EURUSD.120.0.98.onnx](https://www.mql5.com/en/articles/download/14424/lstm.15m.eurusd.120.0.98.onnx "Download LSTM.15m.EURUSD.120.0.98.onnx") [ONNX.eurusd.H1.120.15m\_eurusd.mq5](https://www.mql5.com/en/articles/download/14424/onnx.eurusd.h1.120.15m_eurusd.mq5 "Download ONNX.eurusd.H1.120.15m_eurusd.mq5")

15 min without filter

![15 min without filter tests](https://c.mql5.com/2/72/TesterOptgraphReport2024.03.12__2.png)

![2d 15 min without filter](https://c.mql5.com/2/72/TesterOptgraphReport2024.03.12.2__2.png)

![tests 15 min with filter .](https://c.mql5.com/2/72/tests.png)

[LTSM\_simple\_15m\_filtrado\_sin.py](https://www.mql5.com/en/articles/download/14424/ltsm_simple_15m_filtrado_sin.py ".py") [LSTM.15m.EURUSD.120.0.98.onnx](https://www.mql5.com/en/articles/download/14424/lstm.15m.eurusd.120.0.98.onnx ".onnx") [ONNX.eurusd.H1.120.15m\_eurusd.mq5](https://www.mql5.com/en/articles/download/14424/onnx.eurusd.h1.120.15m_eurusd.mq5 ".mq5")

30 min with filter (now on I will always use filter)

![tester 30 min](https://c.mql5.com/2/72/TesterOptgraphReport2024.03.12__4.png)

![tester 30 min 2d](https://c.mql5.com/2/72/TesterOptgraphReport2024.03.12.2__4.png)

![testing 30 min](https://c.mql5.com/2/72/testing_1h.png)

[LSTM.30m.EURUSD.120.0.94.onnx](https://www.mql5.com/en/articles/download/14424/146446/lstm.30m.eurusd.120.0.94.onnx "Download LSTM.30m.EURUSD.120.0.94.onnx") [LTSM\_simple\_30m\_filtrado.py](https://www.mql5.com/en/articles/download/14424/ltsm_simple_30m_filtrado.py "Download LTSM_simple_30m_filtrado.py") [ONNX.eurusd.H1.120.30m\_eurusd.mq5](https://www.mql5.com/en/articles/download/14424/onnx.eurusd.h1.120.30m_eurusd.mq5 "Download ONNX.eurusd.H1.120.30m_eurusd.mq5")

1 hour

![1 hour tests](https://c.mql5.com/2/72/TesterOptgraphReport2024.03.12__6.png)

![1 hour tests heatmap](https://c.mql5.com/2/72/TesterOptgraphReport2024.03.12.2__6.png)

![1 hour tests ](https://c.mql5.com/2/72/testing_1h__2.png)

[1 hour files.zip](https://www.mql5.com/en/articles/download/14424/1_hour_files.zip "Download 1 hour files.zip") (815.64 KB)

2 hours

As I can't use a two-day period, I used one day period (for the next days value in the EA)

![tester 2h](https://c.mql5.com/2/72/TesterOptgraphReport2024.03.12__8.png)

![heat map 2h](https://c.mql5.com/2/72/TesterOptgraphReport2024.03.12.2__8.png)

![tests 2h](https://c.mql5.com/2/72/tests_2h.png)

![graph 2h](https://c.mql5.com/2/72/TesterGraphReport2024.03.12.png)

![Backtesting 2h](https://c.mql5.com/2/72/backtesting_2h.png)

[2h.zip](https://www.mql5.com/en/articles/download/14424/2h.zip "Download 2h.zip")

As I can't load more than 32 files, the next files will all be uploaded in a folder.

### Conclusion

It seems that with this strat, as the period goes higher, the results look more robust for all the TPs and SLs

### NextDay var

I've used the strategy from this article: [How to use ONNX models in mql5](https://www.mql5.com/en/articles/12373), but as I've had good results with the 2hours strat, and in that one I used 1Day period. We will study other values for the NextDays var

```
   if(TimeCurrent()>=ExtNextDay)
     {
      GetMinMax();
      //--- set next day time
      ExtNextDay=TimeCurrent();
      ExtNextDay-=ExtNextDay%PeriodSeconds(PERIOD_D1);
      ExtNextDay+=PeriodSeconds(PERIOD_D1);
     }
```

```
void GetMinMax(void)
  {
   vectorf close;
   close.CopyRates(_Symbol,PERIOD_D1,COPY_RATES_CLOSE,0,SAMPLE_SIZE);
   ExtMin=close.Min();
   ExtMax=close.Max();
  }
```

We will now proceed to study EURUSD with period 30min with filtered data, and different NextDay periods (We will use 1D, 12h and 6h) and discuss results.

1D with 30 min period

![1D 30min](https://c.mql5.com/2/72/TesterOptgraphReport2024.03.12_d1.png)

![1D heatmap 30min](https://c.mql5.com/2/72/TesterOptgraphReport2024.03.12.2_d1.png)

![1D tests](https://c.mql5.com/2/72/tests_d1.png)

12 H with 30 min period

![12h 30 min tests](https://c.mql5.com/2/72/TesterOptgraphReport2024.03.12_h12.png)

![12h heatmap](https://c.mql5.com/2/72/TesterOptgraphReport2024.03.12.2_h12.png)

![12h tests 30 min](https://c.mql5.com/2/72/tests_h12.png)

6h 30 min period

![6h 30 min period](https://c.mql5.com/2/72/TesterOptgraphReport2024.03.12_h6.png)

![6h heatmap 30 min period](https://c.mql5.com/2/72/TesterOptgraphReport2024.03.12.2_h6.png)

![h6 30 min tests](https://c.mql5.com/2/72/tests_h6.png)

The results seem better when fine-tuning the NextDay var. Let's see how it evolves in lower NextDay periods.

30 min with NextDay periods of 4h

![4h periods](https://c.mql5.com/2/72/TesterOptgraphReport2024.03.12_h4.png)

![4h period heatmaps](https://c.mql5.com/2/72/TesterOptgraphReport2024.03.12.2_h4.png)

![tests 4h periods](https://c.mql5.com/2/72/tests_h4.png)

30 min with NextDay periods of 2h

![30 min 2 h periods](https://c.mql5.com/2/72/TesterOptgraphReport2024.03.12_h2.png)

![30 min 2h nextday periods heatmap](https://c.mql5.com/2/72/TesterOptgraphReport2024.03.12.2_h2.png)

![2h period tests ](https://c.mql5.com/2/72/tests_h2.png)

30 min with NextDay periods of 1h

![h1 periods 30 min](https://c.mql5.com/2/72/TesterOptgraphReport2024.03.12_h1.png)

![h1 heatmap 30 min ](https://c.mql5.com/2/72/TesterOptgraphReport2024.03.12.2_h1.png)

![tests 1h 30 min](https://c.mql5.com/2/72/tests_h1.png)

It seems that for at least 30 minute periods, around 8 to 12 bars of 30 minutes give better results.

As this "game" is to win more money, and one way to do that is to have more winning trades and a solid strategy. Let's see if we can use this start to win with 5 minutes periods, so we will try this strat for 5 minutes and 1 h and 30 min for the NextDay var.

![Inputs](https://c.mql5.com/2/72/inputs__4.png)

### 5 minutes with 1 h NextDay var.

![tester 1 h (5 min period)](https://c.mql5.com/2/72/TesterOptgraphReport2024.03.12__10.png)

![heatmap 1h (5 min period)](https://c.mql5.com/2/72/TesterOptgraphReport2024.03.12.2__10.png)

![results test 1h (5 min)](https://c.mql5.com/2/72/testing_1_hour_NextDay.png)

### 5 minutes with 30 min NextDay var.

### ![tester 30 min (5 min)](https://c.mql5.com/2/72/TesterOptgraphReport2024.03.12__12.png)      ![heatmap 30 min (5 min)](https://c.mql5.com/2/72/TesterOptgraphReport2024.03.12.2__12.png)      ![testing 30 min (5 min)](https://c.mql5.com/2/72/testing_30_min_NextDay.png)

Files used:   [Last files.zip](https://www.mql5.com/en/articles/download/14424/146474/last_files.zip "Download Last files.zip")

Other time periods, seem to be more reliable, but, if you want another EA, you have more to use.

For last, we can make adjustments to the strategy, that could end up giving better results or more solid results, like establishing a limit of time or bars, for example, in 1 h periods, and using a 12h for next day, we could establish that once the order is made, it must stay opened no more than 12 hours.

```
   if(ExtPredictedClass>=0)
     {
      if(PositionSelect(_Symbol))
         CheckForClose();
      else
        {
         CheckForOpen();
         started_time1 = rates[0].time;
         string started_time1_str = TimeToString(started_time1);
```

```
int total = PositionsTotal();
   if(total>0)
     {
      datetime started_time2 = rates[0].time;
      string started_time2_str = TimeToString(started_time2);
      int segundos = started_time2 - started_time1;
      Print("Tiempo 1: ", started_time1_str);
      Print("Tiempo 2: ", started_time2_str);
      Print("Diferencia en segundos: ", segundos);
      if((segundos >= days*PeriodSeconds(PERIOD_H12)) && segundos != 0)
        {
         Print("closing position----------------");
         ExtTrade.PositionClose(_Symbol,3);

        }
     }
```

![tester](https://c.mql5.com/2/73/TesterOptgraphReport2024.03.14.png)

![heatmap](https://c.mql5.com/2/73/TesterOptgraphReport2024.03.14.2.png)

![tests](https://c.mql5.com/2/73/tests__2.png)

![graph](https://c.mql5.com/2/73/TesterGraphReport2024.03.14.png)

[.mq5](https://www.mql5.com/en/articles/download/14424/146660/onnx.eurusd.120.1h_h12_eurusd.v3.mq5 ".mq5") [ONNX.eurusd.120.1h\_H12\_eurusd.v3.mq5](https://www.mql5.com/en/articles/download/14424/onnx.eurusd.120.1h_h12_eurusd.v3.mq5 "Download ONNX.eurusd.120.1h_H12_eurusd.v3.mq5") (9.23 KB)

### Conclusion

We have seen seasoning, filtering and compared models, EAs and parameters, trying to achieve better results with fine-tuning. I hope you enjoyed reading this article as I enjoyed making it.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/14424.zip "Download all attachments in the single ZIP archive")

[ONNX.eurusd.H1.120.Prediction\_FILTERED.mq5](https://www.mql5.com/en/articles/download/14424/onnx.eurusd.h1.120.prediction_filtered.mq5 "Download ONNX.eurusd.H1.120.Prediction_FILTERED.mq5")(8.03 KB)

[onnx\_LSTM\_simple\_EURUSD\_filtered.py](https://www.mql5.com/en/articles/download/14424/onnx_lstm_simple_eurusd_filtered.py "Download onnx_LSTM_simple_EURUSD_filtered.py")(12.59 KB)

[onnx\_LSTM\_simple\_EURUSD\_not\_filtered.py](https://www.mql5.com/en/articles/download/14424/onnx_lstm_simple_eurusd_not_filtered.py "Download onnx_LSTM_simple_EURUSD_not_filtered.py")(11.17 KB)

[Download\_seasonalities\_1h\_v2.py](https://www.mql5.com/en/articles/download/14424/download_seasonalities_1h_v2.py "Download Download_seasonalities_1h_v2.py")(7.77 KB)

[data\_febs.zip](https://www.mql5.com/en/articles/download/14424/data_febs.zip "Download data_febs.zip")(6574.19 KB)

[draw\_with\_sum.py](https://www.mql5.com/en/articles/download/14424/draw_with_sum.py "Download draw_with_sum.py")(2.37 KB)

[seasonal\_feb\_concat.zip](https://www.mql5.com/en/articles/download/14424/seasonal_feb_concat.zip "Download seasonal_feb_concat.zip")(6548.13 KB)

[create\_concat\_seasonal\_data.py](https://www.mql5.com/en/articles/download/14424/create_concat_seasonal_data.py "Download create_concat_seasonal_data.py")(0.75 KB)

[onnx\_LSTM\_simple\_EURUSD\_concat\_seasonals.py](https://www.mql5.com/en/articles/download/14424/onnx_lstm_simple_eurusd_concat_seasonals.py "Download onnx_LSTM_simple_EURUSD_concat_seasonals.py")(7.53 KB)

[EURUSD\_LSTM\_120\_1h\_not\_filtered.onnx](https://www.mql5.com/en/articles/download/14424/eurusd_lstm_120_1h_not_filtered.onnx "Download EURUSD_LSTM_120_1h_not_filtered.onnx")(884.39 KB)

[EURUSD\_LSTM\_120\_1h\_filtered\_v1.onnx](https://www.mql5.com/en/articles/download/14424/eurusd_lstm_120_1h_filtered_v1.onnx "Download EURUSD_LSTM_120_1h_filtered_v1.onnx")(884.34 KB)

[EURUSD\_LSTM\_120\_1h\_filtered\_v2.onnx](https://www.mql5.com/en/articles/download/14424/eurusd_lstm_120_1h_filtered_v2.onnx "Download EURUSD_LSTM_120_1h_filtered_v2.onnx")(884.39 KB)

[ONNX.eurusd.H1.120.Prediction\_FILTERED\_v2.mq5](https://www.mql5.com/en/articles/download/14424/onnx.eurusd.h1.120.prediction_filtered_v2.mq5 "Download ONNX.eurusd.H1.120.Prediction_FILTERED_v2.mq5")(8.03 KB)

[ONNX.eurusd.H1.120.Prediction\_NOT\_FILTERED.mq5](https://www.mql5.com/en/articles/download/14424/onnx.eurusd.h1.120.prediction_not_filtered.mq5 "Download ONNX.eurusd.H1.120.Prediction_NOT_FILTERED.mq5")(8.04 KB)

[EURUSD\_LSTM\_270\_1h\_filtered\_seasonal0.72.onnx](https://www.mql5.com/en/articles/download/14424/eurusd_lstm_270_1h_filtered_seasonal0.72.onnx "Download EURUSD_LSTM_270_1h_filtered_seasonal0.72.onnx")(884.36 KB)

[onnx\_LSTM\_simple\_EURUSD\_concat\_seasonals.py](https://www.mql5.com/en/articles/download/14424/onnx_lstm_simple_eurusd_concat_seasonals.py "Download onnx_LSTM_simple_EURUSD_concat_seasonals.py")(7.53 KB)

[ONNX.eurusd.H1.120.Prediction\_seasonal.mq5](https://www.mql5.com/en/articles/download/14424/onnx.eurusd.h1.120.prediction_seasonal.mq5 "Download ONNX.eurusd.H1.120.Prediction_seasonal.mq5")(8.04 KB)

[onnx\_LSTM\_simple\_EURUSD\_filtered\_1hour.py](https://www.mql5.com/en/articles/download/14424/onnx_lstm_simple_eurusd_filtered_1hour.py "Download onnx_LSTM_simple_EURUSD_filtered_1hour.py")(12.75 KB)

[autocorrelation.py](https://www.mql5.com/en/articles/download/14424/autocorrelation.py "Download autocorrelation.py")(0.96 KB)

[LTSM\_simple\_15m\_filtrado.py](https://www.mql5.com/en/articles/download/14424/ltsm_simple_15m_filtrado.py "Download LTSM_simple_15m_filtrado.py")(6.95 KB)

[LSTM.15m.EURUSD.120.0.98.onnx](https://www.mql5.com/en/articles/download/14424/lstm.15m.eurusd.120.0.98.onnx "Download LSTM.15m.EURUSD.120.0.98.onnx")(884.4 KB)

[ONNX.eurusd.H1.120.15m\_eurusd.mq5](https://www.mql5.com/en/articles/download/14424/onnx.eurusd.h1.120.15m_eurusd.mq5 "Download ONNX.eurusd.H1.120.15m_eurusd.mq5")(8.14 KB)

[LTSM\_simple\_15m\_filtrado\_sin.py](https://www.mql5.com/en/articles/download/14424/ltsm_simple_15m_filtrado_sin.py "Download LTSM_simple_15m_filtrado_sin.py")(6.57 KB)

[LSTM.15m.EURUSD.120.0.98.onnx](https://www.mql5.com/en/articles/download/14424/lstm.15m.eurusd.120.0.98.onnx "Download LSTM.15m.EURUSD.120.0.98.onnx")(884.42 KB)

[ONNX.eurusd.H1.120.15m\_eurusd.mq5](https://www.mql5.com/en/articles/download/14424/onnx.eurusd.h1.120.15m_eurusd.mq5 "Download ONNX.eurusd.H1.120.15m_eurusd.mq5")(8.14 KB)

[LSTM.30m.EURUSD.120.0.94.onnx](https://www.mql5.com/en/articles/download/14424/lstm.30m.eurusd.120.0.94.onnx "Download LSTM.30m.EURUSD.120.0.94.onnx")(884.41 KB)

[LTSM\_simple\_30m\_filtrado.py](https://www.mql5.com/en/articles/download/14424/ltsm_simple_30m_filtrado.py "Download LTSM_simple_30m_filtrado.py")(6.96 KB)

[ONNX.eurusd.H1.120.30m\_eurusd.mq5](https://www.mql5.com/en/articles/download/14424/onnx.eurusd.h1.120.30m_eurusd.mq5 "Download ONNX.eurusd.H1.120.30m_eurusd.mq5")(8.15 KB)

[1\_hour\_files.zip](https://www.mql5.com/en/articles/download/14424/1_hour_files.zip "Download 1_hour_files.zip")(815.64 KB)

[2h.zip](https://www.mql5.com/en/articles/download/14424/2h.zip "Download 2h.zip")(814.85 KB)

[Last\_files.zip](https://www.mql5.com/en/articles/download/14424/last_files.zip "Download Last_files.zip")(5832.24 KB)

[ONNX.eurusd.120.1h\_H12\_eurusd.v3.mq5](https://www.mql5.com/en/articles/download/14424/onnx.eurusd.120.1h_h12_eurusd.v3.mq5 "Download ONNX.eurusd.120.1h_H12_eurusd.v3.mq5")(9.23 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Integrating Discord with MetaTrader 5: Building a Trading Bot with Real-Time Notifications](https://www.mql5.com/en/articles/16682)
- [Trading Insights Through Volume: Trend Confirmation](https://www.mql5.com/en/articles/16573)
- [Trading Insights Through Volume: Moving Beyond OHLC Charts](https://www.mql5.com/en/articles/16445)
- [From Python to MQL5: A Journey into Quantum-Inspired Trading Systems](https://www.mql5.com/en/articles/16300)
- [Example of new Indicator and Conditional LSTM](https://www.mql5.com/en/articles/15956)
- [Scalping Orderflow for MQL5](https://www.mql5.com/en/articles/15895)
- [Using PSAR, Heiken Ashi, and Deep Learning Together for Trading](https://www.mql5.com/en/articles/15868)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/464357)**
(2)


![Roman Shiredchenko](https://c.mql5.com/avatar/2012/3/4F76634B-9044.jpg)

**[Roman Shiredchenko](https://www.mql5.com/en/users/r0man)**
\|
25 Jul 2024 at 19:34

Thanks for the article on seasonality and its automation. I had a quick look at it. I will be at my computer at the weekend, I will read it more carefully and use the elements of automotisation proposed by you in my trading systems on the seasonality factor.


![Roman Shiredchenko](https://c.mql5.com/avatar/2012/3/4F76634B-9044.jpg)

**[Roman Shiredchenko](https://www.mql5.com/en/users/r0man)**
\|
25 Jul 2024 at 23:02

I would like to see a more extended conclusion in the form of whether more interesting (positive) results were achieved when using the seasonality factor...


![Data Science and Machine Learning (Part 21): Unlocking Neural Networks, Optimization algorithms demystified](https://c.mql5.com/2/73/Data_Science_and_Machine_Learning_Part_21___LOGO.png)[Data Science and Machine Learning (Part 21): Unlocking Neural Networks, Optimization algorithms demystified](https://www.mql5.com/en/articles/14435)

Dive into the heart of neural networks as we demystify the optimization algorithms used inside the neural network. In this article, discover the key techniques that unlock the full potential of neural networks, propelling your models to new heights of accuracy and efficiency.

![Master MQL5 from beginner to pro (Part I): Getting started with programming](https://c.mql5.com/2/60/Introduction_to_programming_using_the_MQL5_language_LOGO.png)[Master MQL5 from beginner to pro (Part I): Getting started with programming](https://www.mql5.com/en/articles/13594)

This article is an introduction to a series of articles about programming. It is assumed here that the reader has never dealt with programming before. So, this series starts from the very basics. Programming knowledge level: Absolute Beginner.

![Population optimization algorithms: Spiral Dynamics Optimization (SDO) algorithm](https://c.mql5.com/2/61/Spiral_Dynamics_Optimization_SDO_LOGO.png)[Population optimization algorithms: Spiral Dynamics Optimization (SDO) algorithm](https://www.mql5.com/en/articles/12252)

The article presents an optimization algorithm based on the patterns of constructing spiral trajectories in nature, such as mollusk shells - the spiral dynamics optimization (SDO) algorithm. I have thoroughly revised and modified the algorithm proposed by the authors. The article will consider the necessity of these changes.

![Population optimization algorithms: Intelligent Water Drops (IWD) algorithm](https://c.mql5.com/2/60/Intelligent_Water_Drops_IWD_LOGO.png)[Population optimization algorithms: Intelligent Water Drops (IWD) algorithm](https://www.mql5.com/en/articles/13730)

The article considers an interesting algorithm derived from inanimate nature - intelligent water drops (IWD) simulating the process of river bed formation. The ideas of this algorithm made it possible to significantly improve the previous leader of the rating - SDS. As usual, the new leader (modified SDSm) can be found in the attachment.

[![](https://www.mql5.com/ff/sh/zf7a2k61x98jzh89z2/01.png)Speed up your tradingUse our high-speed VPS for MetaTrader 4 and 5Learn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=qtrrsuiwuicrscmckjynyanztbditglq&s=c617dc80d90cfd3783ec1345eec2b419b281f10fec6eac77b3218984ac337259&uid=&ref=https://www.mql5.com/en/articles/14424&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5071812608295644905)

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