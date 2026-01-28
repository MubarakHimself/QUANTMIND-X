---
title: Alternative risk return metrics in MQL5
url: https://www.mql5.com/en/articles/13514
categories: Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T21:25:53.336909
---

[![](https://www.mql5.com/ff/si/3fgkjn78mkxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=ocndbzpeklfncxysjbwfhhbalbrsdbtv&s=a4309643278437a00bdd33c5809fc6b4b4032749c00fccd07b3b84e7b8b45126&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=sdfilgtisbmgkdltowticxvgqkwgowjd&ssn=1769192752315986645&ssn_dr=0&ssn_sr=0&fv_date=1769192752&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F13514&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Alternative%20risk%20return%20metrics%20in%20MQL5%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176919275210174853&fz_uniq=5071827241249222447&sv=2552)

MetaTrader 5 / Examples


### Introduction

All traders hope to maximize the percentage return on their investment by as much as possible, however higher returns usually come at a higher risk. This is the reason why risk adjusted returns are the main measure of performance in the investment industry. There are many different measures of risk adjusted return, each one with its own set of advantages and disadvantages. The Sharpe ratio is a popular risk return measure famous for imposing unrealistic preconditions on the distribution of returns being analyzed. This has inevitably lead to the development of alternative performance metrics that seek to provide the same ubiquity of the Sharpe ratio without its shortcommings. In this article we provide the implementation of alternative risk return metrics and generate hypothetical equity curves to analyze their characteristics.

### Simulated equity curves

In order to ensure  interpretability we will use SP 500 data as the basis for a simulated trading strategy. We will not use any specific trading rules, instead we employ random numbers to generate equity curves and corresponding  returns series. The initial capital will be standardized to a configurable amount. Random numbers will be defined by a  seed so that anyone who wants to reproduce the experiments can do so.

### Visualizing equity curves

The graphic below represents a Metatrader 5 (MT5) application implemented as an Expert Advisor (EA), that displays three equity curves. The red equity curve is the benchmark from which the blue and green equity curves are derived. The benchmark can be altered by configuring the initial capital. Adjustable from the application.

![Simulated Equity Curves EA](https://c.mql5.com/2/58/SimulatedEquityCurves.gif)

Equity curves will be created based on the benchmark series. Each one is defined by a random componet that can be controlled through two adjustable constants. The mean coefficient and a standard deviation coeffecient. These in combination with the standard deviation of benchmark returns, determine the parameters for the normaly distributed random numbers used to generate two hypothetical equity curves.

![Selected period ](https://c.mql5.com/2/58/US500IndexDailyPeriod.png)

We first calculate the returns using SP 500 daily close prices from the period between 20 March 2020 and 5 January 2022, inclusive. The equity curve is built from the series of returns. With the series that define the equity and returns we will compare the performance results calculated along with the appearance of a given equity curve.

The code for the application is attached to the article.

### Drawdown ratios

Drawdown represents the largest amount of equity that a strategy lost between any two points in time. This value gives an indication of the risk that a strategy took on to accomplish its gains, if any were attained. When  a series made up of an arbitrary number of the highest drawdowns is tallied and aggregated, the result can be used as a measure of variability.

### Burke Ratio

In 1994 Burke wrote a paper titled "A sharper sharpe ratio" , which introduced the Burke ratio as an alternative to the familiar Sharpe ratio. The Burke ratio substitutes the denominator in the Sharpe ratio formula for the squared sum of a specified number of the highest absolute drawdowns. The numerator can either be the mean return or the absolute monetary return of the strategy/portfolio, ie ,net profit. We will have a look at the two versions of the calculation. We distinguish the two versions as the net profit based Burke Ratio and the mean returns Burke Ratio. The formulae are given below.

![Net profit based Burke Ratio formula](https://c.mql5.com/2/58/Npburkeformula.png)

![Mean returns based Burke Ratio formula](https://c.mql5.com/2/58/meanBurkeformula.png)

MaxD is the series of the largest T absolute drawdowns calculated from the equity values. N represents the number of equity values used in the calculation.

The net profit Burke Ratio is implemented as the netprofit\_burke() function. The function requires an array of equity values describing the equity curve as well as an integer value denoting the number of the highest drawdowns to consider in the computation.

```
//+------------------------------------------------------------------+
//|Net profit based Burke ratio                                      |
//+------------------------------------------------------------------+
double netprofit_burke(double &in_ec[],int n_highestdrawdowns=0)
  {
   double outdd[];
   double sumdd=0;
   int insize=ArraySize(in_ec);

   if(n_highestdrawdowns<=0)
      n_highestdrawdowns=int(insize/20);

   if(MaxNDrawdowns(n_highestdrawdowns,in_ec,outdd))
     {
      for(int i=0; i<ArraySize(outdd); i++)
        {
         sumdd+=(outdd[i]*outdd[i]);
        }
      return (in_ec[insize-1]-in_ec[0])/(MathSqrt((1.0/double(insize)) * sumdd));
     }
   else
      return 0;
  }
```

When the default value of zero is specified, the function uses the formula N/20 to set the number of drawdowns to consider, where N is the size of the equity array.

To collect the specified number of drawdowns, MaxNDradowns() function is enlisted. It outputs a series of the highest absolute drawdowns arranged in ascending order.

```
//+------------------------------------------------------------------+
//|Maximum drawdowns function given equity curve                     |
//+------------------------------------------------------------------+
bool MaxNDrawdowns(const int num_drawdowns,double &in_ec[],double &out_dd[])
  {
   ZeroMemory(out_dd);

   ResetLastError();

   if(num_drawdowns<=0)
     {
      Print("Invalid function parameter for num_drawdowns ");
      return false;
     }
   double u[],v[];

   int size = ArraySize(in_ec);

   if((ArrayResize(v,(size*(size-1))/2)< int((size*(size-1))/2))||
      (ArraySize(out_dd)!=num_drawdowns && ArrayResize(out_dd,num_drawdowns)<num_drawdowns))
     {
      Print(__FUNCTION__, " resize error ", GetLastError());
      return false;
     }

   int k=0;
   for(int i=0; i<size-1; i++)
     {
      for(int j=i+1; j<size; j++)
        {
         v[k]=in_ec[i]-in_ec[j];
         k++;
        }
     }

   ArraySort(v);

   for(int i=0; i<k; i++)
     {
      if(v[i]>0)
        {
         if(i)
           {
            if(!ArrayRemove(v,0,i))
              {
               Print(__FUNCTION__, " error , ArrayRemove: ",GetLastError());
               return false;
              }
            else
               break;
           }
         else
            break;
        }
     }

   size=ArraySize(v);

   if(size && size<=num_drawdowns)
     {
      if(ArrayCopy(out_dd,v)<size)
        {
         Print(__FUNCTION__, " error ", GetLastError());
         return false;
        }
      else
         return (true);
     }

   if(ArrayCopy(out_dd,v,0,size-num_drawdowns,num_drawdowns)<num_drawdowns)
     {
      Print(__FUNCTION__, " error ", GetLastError());
      return false;
     }

   return(true);

  }
```

The Burke ratio computation that uses mean returns as the numerator is implemented as the meanreturns\_burke function and has similar input parameters.

```
//+------------------------------------------------------------------+
//|Mean return based Burke ratio                                     |
//+------------------------------------------------------------------+
double meanreturns_burke(double &in_ec[],int n_highestdrawdowns=0)
  {
   double outdd[];
   double rets[];

   double sumdd=0;
   int insize=ArraySize(in_ec);

   if(ArrayResize(rets,insize-1)<insize-1)
     {
      Print(__FUNCTION__," Memory allocation error ",GetLastError());
      return 0;
     }

   for(int i=1; i<insize; i++)
      rets[i-1] = (in_ec[i]/in_ec[i-1]) - 1.0;

   if(n_highestdrawdowns<=0)
      n_highestdrawdowns=int(insize/20);

   if(MaxNDrawdowns(n_highestdrawdowns,in_ec,outdd))
     {
      for(int i=0; i<ArraySize(outdd); i++)
         sumdd+=(outdd[i]*outdd[i]);
      return MathMean(rets)/(MathSqrt((1.0/double(insize)) * sumdd));
     }
   else
      return 0;
  }
```

### Net profit to maximum drawdown ratio

The Burke ratio formula that uses net profit as the numerator is similar to the Netprofit to Maximum Drawdown ratio (NPMD). The difference is in the use of the single highest drawdown in the calculation of the NPMD ratio.

![NPMD ratio formula](https://c.mql5.com/2/58/Npmdformula.PNG)

NPMD calculation is implemented as the netProfiMaxDD() function, requiring the array of equity values as input.

```
//+------------------------------------------------------------------+
//|Net profit to maximum drawdown ratio                              |
//+------------------------------------------------------------------+
double netProfiMaxDD(double &in_ec[])
  {
   double outdd[];
   int insize=ArraySize(in_ec);

   if(MaxNDrawdowns(1,in_ec,outdd))
      return ((in_ec[insize-1]-in_ec[0])/outdd[0]);
   else
      return 0;
  }
```

Drawdown based ratios were introduced to address some of the criticisms levied against the Sharpe ratio. The calculations do not penalize abnormal or large gains and most importantly are non-parametric. Although advantageous, the use of absolute drawdowns in the denominator makes both the Burke and NPMD ratios favour strategies with relatively mild downward spikes.

![Burk Ratio results](https://c.mql5.com/2/58/npBurkeCompare.PNG)

Refering to the equity curve visualization tool. The blue curve has the highest returns but it scores lower than the rest.

![Mean returns based Burke Ratio results](https://c.mql5.com/2/58/meanburkecompare.PNG)

The benchmark values for both ratios emphasize just how misleading the metrics can be when used to compare the performance of strategies. The benchmark ratios are significantly higher despite the other curves delivering more in real returns.

![NPMD ratio results](https://c.mql5.com/2/58/NPMDCompare.PNG)

Using absolute drawdowns can over emphasize risk, relative to using the distribution of negative returns as is the case with the Sharpe ratio.

![Sharpe ratio results](https://c.mql5.com/2/58/sharpeCompare.PNG)

Comparing the Sharpe scores, we see that the green curve has the highest, with much less of a difference between the simulated strategy results.

### Drawdown ratios interpretation

The higher the Burke or NPMD ratio, the better the risk-adjusted performance of the investment strategy. It means the strategy is generating higher returns compared to the risk taken.

\- If the Burke or NPMD ratio is greater than 0, it suggests that the investment strategy is providing excess return compared to the calculated risk.

\- If the Burke or NPMD ratio is less than 0, it indicates that the investment strategy is not generating sufficient excess return compared to the risk taken.

### Partial moment ratios

Partial moment based ratios are another attempt at an alternative to Sharpe Ratio. They are based on the statistical concept of semi variance and provide insights into how well a strategy is performing in terms of downside risk (negative returns) compared to upside potential (positive returns).To calculate partial moment ratios, we first need to determine the partial gain and partial loss. These values are derived by identifying a threshold return level, usually the minimum acceptable return or the risk-free rate, and calculating the difference between the actual return and the threshold return for each observation.

The calculation can ignore differences that are either above the threshold for the lower partial moment or below threshold for the higher partial moment (HPM). The lower partial moment (LPM) measures the squared deviations of returns that fall below the threshold, while the higher partial moment (HPM) measures the squared deviations of returns that exceed the threshold. Partial moments present an alternative perspective on risk compared to drawdown ratios, by focusing on the risk associated with returns that are either below or above a specific threshold.

Below are formulae for the LPM and HPM respectively:

![LPM formula](https://c.mql5.com/2/58/Lpmformula.PNG)

![HPM formula](https://c.mql5.com/2/58/Hpmformula.PNG)

Where thresh is the threshold , x is an observed return and max determines the maximum value between the resulting difference and zero before being raised to the power of n. n defines the degree of a partial moment. When n=0 , the LPM becomes the probability that an observation  is less than the threshold and HPM gives the probability it is above. N is the number of observed returns. Two partial moment ratios we will look at are the generalized Omega and the Upside Potential Ratio (UPR).

### Omega

The generalized Omega is defined by an single n degree term and a threshold value which defines the LPM used in the calculation.

![Omega formula](https://c.mql5.com/2/58/Omegaformula.PNG)

The omega() function implements the computation of the Omega ratio by taking as input an array of returns. The function uses a degree 2 lower partial moment, with the threshold returns taken as zero.

```
//+------------------------------------------------------------------+
//|omega ratio                                                       |
//+------------------------------------------------------------------+
double omega(double &rt[])
  {
   double rb[];

   if(ArrayResize(rb,ArraySize(rt))<0)
     {
      Print(__FUNCTION__, " Resize error ",GetLastError());
      return 0;
     }

   ArrayInitialize(rb,0.0);

   double pmomentl=MathPow(partialmoment(2,rt,rb),0.5);

   if(pmomentl)
      return MathMean(rt)/pmomentl;
   else
      return 0;
  }
```

### Upside potential ratio

The UPR uses two n degree terms (n1 and n2 in the formula) and a threshold. n1 determines the HPM degree in the numerator and n2 specifies the LPM degree of the denominator.

![UPR formula](https://c.mql5.com/2/58/UPRformula.PNG)

Similar to the implementation of the Omega performance metric,  the upsidePotentialRatio() function computes the UPR ratio. The computation also uses degree 2 partial moments, and threshold returns defined as zero.

```
//+------------------------------------------------------------------+
//|Upside potential ratio                                            |
//+------------------------------------------------------------------+
double upsidePotentialRatio(double &rt[])
  {
   double rb[];

   if(ArrayResize(rb,ArraySize(rt))<0)
     {
      Print(__FUNCTION__, " Resize error ",GetLastError());
      return 0;
     }

   ArrayInitialize(rb,0.0);

   double pmomentu=MathPow(partialmoment(2,rt,rb,true),0.5);
   double pmomentl=MathPow(partialmoment(2,rt,rb),0.5);
   if(pmomentl)
      return pmomentu/pmomentl;
   else
      return 0;
  }
```

Partial moment calculations are implemented in the partialmoment() function. It requires as input the degree of the moment as an unsigned integer,  two arrays of double type and boolean value. The first array should contain the observed returns and the second the threshold or benchmark returns used in the calculation. The boolean value determines the type of partial moment to be calculated, either true for the  higher partial moment and false for the lower partial moment.

```
//+------------------------------------------------------------------+
//|Partial Moments                                                   |
//+------------------------------------------------------------------+
double partialmoment(const uint n,double &rt[],double &rtb[],bool upper=false)
  {
   double pm[];
   int insize=ArraySize(rt);

   if(n)
     {
      if(ArrayResize(pm,insize)<insize)
        {
         Print(__FUNCTION__," resize error ", GetLastError());
         return 0;
        }

      for(int i=0; i<insize; i++)
         pm[i] = (!upper)?MathPow(MathMax(rtb[i]-rt[i],0),n):MathPow(MathMax(rt[i]-rtb[i],0),n);
      return MathMean(pm);
     }
   else
     {
      int k=0;
      for(int i=0; i<insize; i++)
        {
         if((!upper && rtb[i]>=rt[i]) || (upper && rt[i]>rtb[i]))
           {
            ArrayResize(pm,k+1,1);
            pm[k]=rt[i];
            ++k;
           }
         else
            continue;
        }

      return MathMean(pm);

     }

  }
```

![UPR results](https://c.mql5.com/2/58/UPRcompare.PNG)

Looking at the Omega and UPR ratios of our equity curves we notice the similarity in rating with that of the Sharpe ratio.

![Omega results](https://c.mql5.com/2/58/OmegaCompare.PNG)

Again consistency is favoured over the more volatile equity curves. Omega specifically looks like a viable alternative to the Sharpe ratio.

### Partial moment ratio interpretation

Omega is much more straight forward when it comes to its interpretation relative to the UPR. The higher omega is the better. Negative values indicate money losing strategies with positive values suggesting performance that delivers excess returns relative to the risk take.

The UPR on the other hand can be a bit of an oddball when it comes to interpretation , have a look at some simulated equity curves with diverging performance profiles.

![Odd UPR results](https://c.mql5.com/2/58/UprOddballresult.PNG)

The blue equity cure shows negative returns yet the UPR result is positive. Even more odd are the results from the green and red equity curves. The curves themselves are almost similar with the green equity curve producing the better returns yet the UPR value is less than that of the red curve.

### Regression analysis of returns - Jensen's Alpha

Linear regression enables construction of a line that best fits a data set. Regression based metrics therefore measure the linearity of equity curves. Jensen's Alpha computes the value of alpha in the standard regression equation. It quantifies the relationship between benchmark returns and the observed returns.

![Jensen's Alpha formula](https://c.mql5.com/2/58/regressionformula.PNG)

To calculate the value of alpha we can use the least squares fit method. The function leastsquarefit() function takes as input two arrays that define the response and the predictor. In this context the response would be the an array of observed returns and the predictor is the array of benchmark returns. The function outputs the alpha and beta values whose references should be supplied when calling the function.

```
//+------------------------------------------------------------------+
//|linear model using least squares fit y=a+bx                       |
//+------------------------------------------------------------------+
double leastsquaresfit(double &y[],double &x[], double &alpha,double &beta)
  {
   double esquared=0;

   int ysize=ArraySize(y);
   int xsize=ArraySize(x);

   double sumx=0,sumy=0,sumx2=0,sumxy=0;

   int insize=MathMin(ysize,xsize);

   for(int i=0; i<insize; i++)
     {
      sumx+=x[i];
      sumx2+=x[i]*x[i];
      sumy+=y[i];
      sumxy+=x[i]*y[i];
     }

   beta=((insize*sumxy)-(sumx*sumy))/((insize*sumx2)-(sumx*sumx));
   alpha=(sumy-(beta*sumx))/insize;

   double pred,error;

   for(int i=0; i<insize; i++)
     {
      pred=alpha+(beta*x[i]);
      error=pred-y[i];
      esquared+=(error*error);
     }

   return esquared;

  }
```

Applying Jensen's Alpha to our simulated equity curves, we get a measure of the linearity of the green and blue equity curves relative to the benchmark (red equity curve).

![Jensen's Alpha results](https://c.mql5.com/2/58/jensenAlphacompare.PNG)

This is the first metric that rates the blue equity curve as the best performing strategy. This metric usually rewards good performance when the bechmark returns are bad. It is possible for Jensen's alpha to indicate positive returns even if absolute returns are negative. This can happen when the absolute benchmark returns are simply worse than the returns being studied. So be careful when using this metric.

### Jensen's Alpha interpretation

If Jensen's Alpha is positive, it suggests that the portfolio/strategy has generated excess returns compared to a benchmark, indicating outperformance after considering its risk level. Otherwise, if Jensen's Alpha is negative, it implies that the strategy has underperformed relative to the benchmark returns.

Besides the alpha value, the beta value from the least squares computation provides a measure of the sensitivity of returns relative to the benchmark. A beta of 1 indicates that the strategy's returns move in sync with the benchmark. A beta less than 1 suggests the equity curve is less volatile than that of the  benchmark, while a beta greater than 1 indicates higher volatility.

### Conclusion

We have described the implementation of a few risk adjusted return metrics that could be used to as alternatives to the Sharpe Ratio. In my opinion, the best candidate is the omega metric. It provides the same advantages of the Sharpe ratio without any expectation of normality in the distribution of returns. Though, it should also be noted that its better to consider multiple strategy performance metrics when making investment decisions. One will never be able to provide a complete picture of expected risk or returns. AlsoRemember, most risk return metrics provide retrospective measures, and historical performance does not guarantee future results. Therefore, it's important to consider other factors like investment objectives, time horizon, and risk tolerance when assessing investment options.

The source code for all metrics are contained in PerformanceRatios.mqh. It should be noted that none of the implementations produce annualized figures. The attached zip file also contains the code for the application used to visualize our simulated equity curves. It was implemented using the EasyAndFastGUI library which is available at the  mql5.com [codebase](https://www.mql5.com/en/code/19703 "/en/code/19703"). The library is not attached, with the article. What is attached is source code of the EA and a working compiled version.

| FileName | Description |
| --- | --- |
| Mql5\\Files\\sp500close.csv | an attached csv file of SP500 close prices used in the calculation of the equity curves |
| Mql5\\Include\\PerformanceRatios.mqh | include file that contains the definition of all the performance metrics described in the article |
| Mql5\\Experts\\EquityCurves.mq5 | Expert advisor source code for the visualization tool, for it to compile it requires the easy and fast gui available in the codebase |
| Mql5\\Experts\\EquityCurves.ext | This is the compiled version of the EA |

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/13514.zip "Download all attachments in the single ZIP archive")

[EquityCurves.ex5](https://www.mql5.com/en/articles/download/13514/equitycurves.ex5 "Download EquityCurves.ex5")(319.8 KB)

[EquityCurves.mq5](https://www.mql5.com/en/articles/download/13514/equitycurves.mq5 "Download EquityCurves.mq5")(17.84 KB)

[sp500close.csv](https://www.mql5.com/en/articles/download/13514/sp500close.csv "Download sp500close.csv")(5.91 KB)

[PerformanceRatios.mqh](https://www.mql5.com/en/articles/download/13514/performanceratios.mqh "Download PerformanceRatios.mqh")(9.33 KB)

[Mql5.zip](https://www.mql5.com/en/articles/download/13514/mql5.zip "Download Mql5.zip")(327.96 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Building Volatility models in MQL5 (Part I): The Initial Implementation](https://www.mql5.com/en/articles/20589)
- [Bivariate Copulae in MQL5 (Part 2): Implementing Archimedean copulae in MQL5](https://www.mql5.com/en/articles/19931)
- [Bivariate Copulae in MQL5 (Part 1): Implementing Gaussian and Student's t-Copulae for Dependency Modeling](https://www.mql5.com/en/articles/18361)
- [Dynamic mode decomposition applied to univariate time series in MQL5](https://www.mql5.com/en/articles/19188)
- [Singular Spectrum Analysis in MQL5](https://www.mql5.com/en/articles/18777)
- [Websockets for MetaTrader 5: Asynchronous client connections with the Windows API](https://www.mql5.com/en/articles/17877)
- [Resampling techniques for prediction and classification assessment in MQL5](https://www.mql5.com/en/articles/17446)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/455645)**
(1)


![fxsaber](https://c.mql5.com/avatar/2019/8/5D67260D-44C9.png)

**[fxsaber](https://www.mql5.com/en/users/fxsaber)**
\|
13 Feb 2024 at 15:18

I like the idea of demonstrating the criterion through visualisation. For completeness, the equity generator for a given criterion value is missing.

I would like to move the criterion value slider with the mouse and get several equity curves corresponding to the changed value at once.

![Category Theory in MQL5 (Part 23): A different look at the Double Exponential Moving Average](https://c.mql5.com/2/58/category-theory-p18-avatar.png)[Category Theory in MQL5 (Part 23): A different look at the Double Exponential Moving Average](https://www.mql5.com/en/articles/13456)

In this article we continue with our theme in the last of tackling everyday trading indicators viewed in a ‘new’ light. We are handling horizontal composition of natural transformations for this piece and the best indicator for this, that expands on what we just covered, is the double exponential moving average (DEMA).

![Classification models in the Scikit-Learn library and their export to ONNX](https://c.mql5.com/2/58/Scikit_learn_to-ONNX_avatar.png)[Classification models in the Scikit-Learn library and their export to ONNX](https://www.mql5.com/en/articles/13451)

In this article, we will explore the application of all classification models available in the Scikit-Learn library to solve the classification task of Fisher's Iris dataset. We will attempt to convert these models into ONNX format and utilize the resulting models in MQL5 programs. Additionally, we will compare the accuracy of the original models with their ONNX versions on the full Iris dataset.

![Data label for time series mining (Part 3)：Example for using label data](https://c.mql5.com/2/58/data-label-for-time-series-mining-avatar.png)[Data label for time series mining (Part 3)：Example for using label data](https://www.mql5.com/en/articles/13255)

This series of articles introduces several time series labeling methods, which can create data that meets most artificial intelligence models, and targeted data labeling according to needs can make the trained artificial intelligence model more in line with the expected design, improve the accuracy of our model, and even help the model make a qualitative leap!

![Studying PrintFormat() and applying ready-made examples](https://c.mql5.com/2/56/printformat-avatar.png)[Studying PrintFormat() and applying ready-made examples](https://www.mql5.com/en/articles/12905)

The article will be useful for both beginners and experienced developers. We will look at the PrintFormat() function, analyze examples of string formatting and write templates for displaying various information in the terminal log.

[![](https://www.mql5.com/ff/sh/x8fwvn495ta7y774z2/01.png)Does your broker offer sponsored hosting for trading?Now it's even easier to get MetaTrader VPS for free – contact your broker for details](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=xscnzeyhifcgygpwvysykhqydcmmbgpp&s=f87b748147e376d34c8f0fdb9737b1766f20cc2174769a0e6b9975b5c2e8ddae&uid=&ref=https://www.mql5.com/en/articles/13514&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5071827241249222447)

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