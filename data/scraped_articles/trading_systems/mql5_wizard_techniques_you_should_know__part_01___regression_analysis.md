---
title: MQL5 Wizard techniques you should know (Part 01): Regression Analysis
url: https://www.mql5.com/en/articles/11066
categories: Trading Systems, Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T19:30:46.488273
---

[![](https://www.mql5.com/ff/sh/6zw0dkux8bqt7m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
MQL5 Channels - Messenger for traders\\
\\
Install the app and receive market analytics and trading tips.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=iuciwacmrxvmiibwyujliagqikizpsoo&s=268cbb13914c54b6c5c875db99b154944f6e0122b3400b54c9ac0d4f69f0f0d6&uid=&ref=https://www.mql5.com/en/articles/11066&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5070345614971049011)

MetaTrader 5 / Tester


### 1\. Introduction

**MQL5 wizard** allows the rapid construction and deployment of expert advisors by having most of the menial aspects of trading pre-coded in the MQL5 library. This allows traders to focus on their custom aspects of their trading such as special entry and exit conditions. Included in the library are some entry and exit signal classes like signals of 'Accelerator Oscillator' indicator, or signals of 'Adaptive Moving Average' indicator and many others. Besides being based on lagging indicators for most traders they may not be convertible to successful strategies. This is why the ability to create your own custom signal is essential. For this article we will explore how this can be done with regression analysis.

### 2\. **Creating the class**

2.1 **Regression analysis**, per Wikipedia, is a set of statistical processes for estimating the relationships between a dependent variable and one or more independent variables. It can be useful to a trader’s expert signal since price data is a time series. This, therefore, allows one to test for the ability of prior prices and price changes to influence future prices and price changes. Regression analysis can be represented by the equation ![eqn_1](https://c.mql5.com/2/47/equation_1.png)

Where **_y_** the dependent and therefore predicted variable depends on prior **_x_** values each with their own coefficient **_β_**, and an error **_ε_.** We can think of **_x_** values and the **_y_** value to be previous and projected price levels respectively. Besides working with price levels, price changes can also be examined in similar fashion. The unknown **_y_** is dependent on **_x_** s, **_β_** s, and **_ε_**. Of these though only the **_x_** s and **_β_** 0 (the y-intercept) are known. The y-intercept is known because it is the price immediately before  **_x_** _i_ 1\. We therefore need to find the respective **_β_** s for each **_x_** and then the **_ε_**. Because each **_x_** _i_ 1was a **_y_** _I_ at the prior time point of the time series, we can solve for the **_β_** values using simultaneous equations. If the next change in price is dependent on only two prior changes, for example, our current equation could be: ![eqn_2](https://c.mql5.com/2/47/equation_2.png)

And the previous equations would be: ![eqn_3](https://c.mql5.com/2/47/equation_3.png)

Since we estimate error **_ε_** separately, we can solve the two simultaneous equations for the **_β_** values. The numbering of the x values in the Wikipedia formula is not in the MQL5 'series' format meaning the highest numbered x is the most recent. I have thus renumbered the x values in the above 2 equations to show how they can be simultaneous. Again we start with the y-intercepts xi1 and xi0 to represent **_β_** 0 in equation 1. The solving of simultaneous equations is better handled with matrices for efficiency. The tools for this are in the MQL5 Library.

2.2  **MQL5 library** has an extensive collection of classes on statistics, and common algorithms that clearly negate the need for one to have to code them from scratch. Its code is also open to the public meaning it can be independently checked. For our purposes we’ll use the ‘RMatrixSolve’ function, under class ‘CDenseSolver’ in the 'solvers.mqh' file. At the heart of this function is the use of matrix LU decomposition to quickly and efficiently solve for **_β_** values. Articles have been written on this in the MetaQuotes archive and Wikipedia also has an explanation [here](https://en.wikipedia.org/wiki/LU_decomposition "https://en.wikipedia.org/wiki/Projection_matrix").

Before we delve into solving for **_β_** values it would be helpful to look at how the ‘CExpertSignal’ class is structured as it is the basis for our class. In almost all expert signal classes that can be assembled in the wizard, there is a ‘LongCondition’ function and a ‘ShortCondition’ function. As you would expect, the two return a value that sets whether you should go long or short respectively. This value needs to be an integer in the range of 0 to 100 in order to map with the wizard’s input parameters of ‘Signal\_ThresholdOpen’ and ‘Signal\_ThresholdClose’. Typically, when trading you want your conditions for closing a position to be less conservative than your conditions for opening. This means the threshold for opening will be higher than the threshold for closing. In developing our signal therefore, we are going to have input parameters for computing the close threshold and separate but similar input parameters for the opening threshold. The selection of inputs to use when computing a condition will be determined by whether or not we have open positions. If we have open positions we will use the close parameters. If no positions are present we will use open parameters. The listing of our expert signal class interface that shows these two sets of parameters is below.

```
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CSignalDUAL_RA : public CExpertSignal
  {
protected:
   CiMA              m_h_ma;             // highs MA handle
   CiMA              m_l_ma;             // lows MA handle
   CiATR             m_ATR;
   //--- adjusted parameters
   int               m_size;
   double            m_open_determination,m_close_determination;
   int               m_open_collinearity,m_open_data,m_open_error;
   int               m_close_collinearity,m_close_data,m_close_error;
public:
                     CSignalDUAL_RA();
                    ~CSignalDUAL_RA();
   //--- methods of setting adjustable parameters

   //--- PARAMETER FOR SETTING THE NUMBER OF INDEPENDENT VARIABLES
   void              Size(int value)                  { m_size=value;                  }

   //--- PARAMETERS FOR SETTING THE OPEN 'THRESHOLD' FOR THE EXPERTSIGNAL CLASS
   void              OpenCollinearity(int value)      { m_open_collinearity=value;     }
   void              OpenDetermination(double value)  { m_open_determination=value;    }
   void              OpenError(int value)             { m_open_error=value;            }
   void              OpenData(int value)              { m_open_data=value;             }

   //--- PARAMETERS FOR SETTING THE CLOSE 'THRESHOLD' FOR THE EXPERTSIGNAL CLASS
   void              CloseCollinearity(int value)     { m_close_collinearity=value;    }
   void              CloseDetermination(double value) { m_close_determination=value;   }
   void              CloseError(int value)            { m_close_error=value;           }
   void              CloseData(int value)             { m_close_data=value;            }

   //--- method of verification of settings
   virtual bool      ValidationSettings(void);
   //--- method of creating the indicator and timeseries
   virtual bool      InitIndicators(CIndicators *indicators);
   //--- methods for detection of levels of entering the market
   virtual bool      OpenLongParams(double &price,double &sl,double &tp,datetime &expiration);
   virtual bool      OpenShortParams(double &price,double &sl,double &tp,datetime &expiration);
   //--- methods of checking if the market models are formed
   virtual int       LongCondition(void);
   virtual int       ShortCondition(void);
protected:
   //--- method of initialization of the oscillator
   bool              InitRA(CIndicators *indicators);
   //--- methods of getting data
   int               CheckDetermination(int ind,bool close);
   double            CheckCollinearity(int ind,bool close);
   //
   double            GetY(int ind,bool close);
   double            GetE(int ind,bool close);

   double            Data(int ind,bool close);
   //
  };
```

Also, here is a listing of our ‘LongCondition’ and ‘ShortCondition’ functions.

```
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int CSignalDUAL_RA::LongCondition(void)
   {
      int _check=CheckDetermination(0,PositionSelect(m_symbol.Name()));
      if(_check>0){ return(_check); }

      return(0);
   }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int CSignalDUAL_RA::ShortCondition(void)
   {
      int _check=CheckDetermination(0,PositionSelect(m_symbol.Name()));
      if(_check<0){ return((int)fabs(_check)); }

      return(0);
   }
```

To continue though, in order to solve for **_β_** values, we will use the ‘GetY’ function. This is listed below.

```
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double CSignalDUAL_RA::GetY(int ind,bool close)
  {
      double _y=0.0;

      CMatrixDouble _a;_a.Resize(m_size,m_size);
      double _b[];ArrayResize(_b,m_size);ArrayInitialize(_b,0.0);

      for(int r=0;r<m_size;r++)
      {
         _b[r]=Data(r,close);

         for(int c=0;c<m_size;c++)
         {
            _a[r].Set(c,Data(r+c+1, close));
         }
      }

      int _info=0;
      CDenseSolver _S;
      CDenseSolverReport _r;
      double _x[];ArrayResize(_x,m_size);ArrayInitialize(_x,0.0);

      _S.RMatrixSolve(_a,m_size,_b,_info,_r,_x);

      for(int r=0;r<m_size;r++)
      {
         _y+=(Data(r,close)*_x[r]);
      }
      //---
      return(_y);
  }
```

The ‘Data’ function referred to will switch between changes in the close price of the symbol being traded or changes in the moving average of the same close price. The option used will be defined by either the ‘m\_open\_data’ input parameter or the ‘m\_close\_data’ input parameter depending on whether we are computing the open threshold or the close threshold. The listing for selecting data is shown in the enumeration below.

```
enum Edata
  {
      DATA_TREND=0,        // changes in moving average close
      DATA_RANGE=1         // changes in close
  };
```

And the ‘Data’ function that selects this is listed below.

```
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double CSignalDUAL_RA::Data(int ind,bool close)
   {
      if(!close)
      {
         if(Edata(m_open_data)==DATA_TREND)
         {
            m_h_ma.Refresh(-1);
            return((m_l_ma.Main(StartIndex()+ind)-m_l_ma.Main(StartIndex()+ind+1))-(m_h_ma.Main(StartIndex()+ind)-m_h_ma.Main(StartIndex()+ind+1)));
         }
         else if(Edata(m_open_data)==DATA_RANGE)
         {
            return((Low(StartIndex()+ind)-Low(StartIndex()+ind+1))-(High(StartIndex()+ind)-High(StartIndex()+ind+1)));
         }
      }
      else if(close)
      {
         if(Edata(m_close_data)==DATA_TREND)
         {
            m_h_ma.Refresh(-1);
            return((m_l_ma.Main(StartIndex()+ind)-m_l_ma.Main(StartIndex()+ind+1))-(m_h_ma.Main(StartIndex()+ind)-m_h_ma.Main(StartIndex()+ind+1)));
         }
         else if(Edata(m_close_data)==DATA_RANGE)
         {
            return((Low(StartIndex()+ind)-Low(StartIndex()+ind+1))-(High(StartIndex()+ind)-High(StartIndex()+ind+1)));
         }
      }

      return(0.0);
   }
```

Once we have the **_β_** values we can then proceed to estimate the error.

2.3  **Standard Error** according to [Wikipedia](https://en.wikipedia.org/wiki/Standard_error "https://en.wikipedia.org/wiki/Standard_error") can be estimated with the formula below.

![eqn_4](https://c.mql5.com/2/47/equation_4.png)

With s as the standard deviation and n the sample size, the error serves as a sobering reminder that not all projections no matter how diligent, will be 100% accurate all the time. We should always factor in and expect some error on our part. The standard deviation, shown in the formula, will be measured between our predicted values and the actual values. For comparison purposes we can also look at a raw error such as the last difference between our forecast and the actual. These two options can be selected from the enumeration below.

```
enum Eerror
  {
      ERROR_LAST=0,        // use the last error
      ERROR_STANDARD=1     // use standard error
  }
```

The ’GetE’ function will then return our error estimate depending on the input parameters ‘m\_open\_error’ or ‘m\_close\_error’ while using the formula above. This is listed below.

```
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double CSignalDUAL_RA::GetE(int ind,bool close)
  {
      if(!close)
      {
         if(Eerror(m_open_error)==ERROR_STANDARD)
         {
            double _se=0.0;
            for(int r=0;r<m_size;r++) { _se+=pow(Data(r,close)-GetY(r+1,close),2.0); }
            _se=sqrt(_se/(m_size-1)); _se=_se/sqrt(m_size); return(_se);
         }
         else if(Eerror(m_open_error)==ERROR_LAST){ return(Data(ind,close)-GetY(ind+1,close)); }
      }
      else if(close)
      {
         if(Eerror(m_close_error)==ERROR_STANDARD)
         {
            double _se=0.0;
            for(int r=0;r<m_size;r++){  _se+=pow(Data(r,close)-GetY(r+1,close),2.0); }
            _se=sqrt(_se/(m_size-1)); _se=_se/sqrt(m_size); return(_se);
         }
         else if(Eerror(m_close_error)==ERROR_LAST){ return(Data(ind,close)-GetY(ind+1,close)); }
      }
//---
      return(Data(ind,close)-GetY(ind+1,close));
  }
```

Once again, the use of ‘m\_open\_error’ or ‘m\_close\_error’ will be determined by whether or not we have open positions. Once we have our error estimate we should be able to make a ball park prediction for y. Regression analysis however has a number of pitfalls. One such pitfall is the ability of the independent variables to be too similar and therefore over inflate the predicted value. This phenomenon is called collinearity and is worth checking for.

2.4  **Collinearity** which Wikipedia defines [here](https://en.wikipedia.org/wiki/Multicollinearity "https://en.wikipedia.org/wiki/Multicollinearity"), can be surmised as the occurrence of high intercorrelations among two or more independent variables in a multiple regression model as per [Investopedia.](https://www.mql5.com/go?link=https://www.investopedia.com/terms/m/multicollinearity.asp "https://www.investopedia.com/terms/m/multicollinearity.asp") It does not have a formula per se and is detected by the variance inflation factor (VIF). This factor is measured across all the independent variables ( **_x_**) to help get a sense of how each of these variables is unique in predicting **_y_**. It is given by the formula below where R is the regression of each independent variable against the others.

![eqn_5](https://c.mql5.com/2/47/equation_5.png)

For our purposes though, in taking account of collinearity, we will take the inverse of the spearman correlation between two recent data sets of independent variables and normalise it. Our data sets length will be set by the input parameter ‘m\_size’ whose minimum length is 3. By normalization we will simply subtract it from two and invert the result. This normalized _weight_ can then be multiplied either to the error estimate, or the predicted value, or both, or be unused. These options are listed in the enumeration below.

```
enum Echeck
  {
      CHECK_Y=0,           // check for y only
      CHECK_E=1,           // check for the error only
      CHECK_ALL=2,         // check for both the y and the error
      CHECK_NONE=-1        // do not use collinearity checks
  };
```

The choice of the applied weight is also set by either input parameter ‘m\_open\_collinearity’ or ‘m\_close\_collinearity’. Again, depending on if positions are open. The ‘CheckCollinearity’ listing is given below.

```
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double CSignalDUAL_RA::CheckCollinearity(int ind,bool close)
  {
      double _check=0.0;
      double _c=0.0,_array_1[],_array_2[],_r=0.0;
      ArrayResize(_array_1,m_size);ArrayResize(_array_2,m_size);
      ArrayInitialize(_array_1,0.0);ArrayInitialize(_array_2,0.0);
      for(int s=0; s<m_size; s++)
      {
         _array_1[s]=Data(ind+s,close);
         _array_2[s]=Data(m_size+ind+s,close);
      }
      _c=1.0/(2.0+fmin(-1.0,MathCorrelationSpearman(_array_1,_array_2,_r)));

      double   _i=Data(m_size+ind,close),    //y intercept
               _y=GetY(ind,close),           //product sum of x and its B coefficients
               _e=GetE(ind,close);           //error



      if(!close)
      {
         if(Echeck(m_open_collinearity)==CHECK_Y){ _check=_i+(_c*_y)+_e;          }
         else if(Echeck(m_open_collinearity)==CHECK_E){ _check=_i+_y+(_c*_e);     }
         else if(Echeck(m_open_collinearity)==CHECK_ALL){ _check=_i+(_c*(_y+_e)); }
         else if(Echeck(m_open_collinearity)==CHECK_NONE){ _check=_i+(_y+_e);     }
      }
      else if(close)
      {
         if(Echeck(m_close_collinearity)==CHECK_Y){ _check=_i+(_c*_y)+_e;          }
         else if(Echeck(m_close_collinearity)==CHECK_E){ _check=_i+_y+(_c*_e);     }
         else if(Echeck(m_close_collinearity)==CHECK_ALL){ _check=_i+(_c*(_y+_e)); }
         else if(Echeck(m_close_collinearity)==CHECK_NONE){ _check=_i+(_y+_e);     }
      }

//---
      return(_check);
  }
```

Besides checking for collinearity, there are times when regression analysis is not as predictive as it could be due to exogeneous changes in the market. To keep track of this, and measure the ability of our signal’s independent variables to influence our dependent variable (the forecast), we use the coefficient of determination.

2.5  **Coefficient of determination** is a statistical measurement that examines how differences in one variable can be explained by the difference in a second variable, when predicting the outcome of a given event as per [Investopedia](https://www.mql5.com/go?link=https://www.investopedia.com/terms/c/coefficient-of-determination.asp "https://www.investopedia.com/terms/c/coefficient-of-determination.asp"). Wikipedia also provides a more exhaustive definition and our formulae shown below are adopted from [there](https://en.wikipedia.org/wiki/Coefficient_of_determination "https://en.wikipedia.org/wiki/Coefficient_of_determination").

![eqn_6](https://c.mql5.com/2/47/equation_6.png)

The formula for sum of squares (with y be the actual value and f the forecast value),

![eqn_7](https://c.mql5.com/2/47/equation_7.png)

The formula for sum of totals (with y being an actual value and ÿ being the moving average of these values),

### ![eqn_8](https://c.mql5.com/2/47/equation_8.png)

And finally, that for the coefficient itself also referred to as R squared.

What this coefficient does is measure the extent to which our **_x_** s are influencing the **_y_**. This is important because, as mentioned, there are periods when regression ebbs meaning it is safer to stay away from the markets. By monitoring this through a filter we are more likely to trade when the system is dependable. Typically, you want this coefficient to be above 0 with 1 being the ideal. The input parameter used in defining our threshold will be 'm\_open\_determination’ or ‘m\_close\_determination’, once again subject to number of open positions. If the coefficient of determination as computed by the 'CheckDetermination' function, listed below, is less than this parameter then the long or short conditions will return zero.

```
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int CSignalDUAL_RA::CheckDetermination(int ind,bool close)
  {
      int _check=0;
      m_h_ma.Refresh(-1);m_l_ma.Refresh(-1);
      double _det=0.0,_ss_res=0.0,_ss_tot=0.0;
      for(int r=0;r<m_size;r++)
      {
         _ss_res+=pow(Data(r,close)-GetY(r+1,close),2.0);
         _ss_tot+=pow(Data(r,close)-((m_l_ma.Main(r)-m_l_ma.Main(r+1))-(m_h_ma.Main(r)-m_h_ma.Main(r+1))),2.0);
      }

      if(_ss_tot!=0.0)
      {
         _det=(1.0-(_ss_res/_ss_tot));
         if(_det>=m_open_determination)
         {
            double _threshold=0.0;
            for(int r=0; r<m_size; r++){ _threshold=fmax(_threshold,fabs(Data(r,close))); }

            double _y=CheckCollinearity(ind,close);

            _check=int(round(100.0*_y/fmax(fabs(_y),fabs(_threshold))));
         }
      }
//---
      return(_check);
  }
```

Once we can check for the coefficient of determination we would have a workable signal. What follows next would be assembling this signal in the MQL5 wizard into an expert advisor.

### 3.  Assembling with MQL5 Wizard

3.1  **Custom ancillary code listing** can be used together with the code from the MQL5 wizard in assembling an expert advisor. This is entirely optional and is to the style of the trader. For the purposes of this article we are going to look at custom pending order opening that is based on the symbol’s prevailing ATR, as well as a system of trailing open positions that is based on the same indicator. We will not use take profit targets.

3.1.1  _ATR based pending orders_ can be set by overloading the functions ‘OpenLongParams’ and ‘OpenShortParams’ and customizing them in our signal class as shown below.

```
//+------------------------------------------------------------------+
//| Detecting the levels for buying                                  |
//+------------------------------------------------------------------+
bool CSignalDUAL_RA::OpenLongParams(double &price,double &sl,double &tp,datetime &expiration)
  {
   CExpertSignal *general=(m_general!=-1) ? m_filters.At(m_general) : NULL;
//---
   if(general==NULL)
     {
      m_ATR.Refresh(-1);
      //--- if a base price is not specified explicitly, take the current market price
      double base_price=(m_base_price==0.0) ? m_symbol.Ask() : m_base_price;

      //--- price overload that sets entry price to be based on ATR
      price      =m_symbol.NormalizePrice(base_price-(m_price_level*(m_ATR.Main(0)/m_symbol.Point()))*PriceLevelUnit());

      sl         =0.0;
      tp         =0.0;
      expiration+=m_expiration*PeriodSeconds(m_period);
      return(true);
     }
//---
   return(general.OpenLongParams(price,sl,tp,expiration));
  }
//+------------------------------------------------------------------+
//| Detecting the levels for selling                                 |
//+------------------------------------------------------------------+
bool CSignalDUAL_RA::OpenShortParams(double &price,double &sl,double &tp,datetime &expiration)
  {
   CExpertSignal *general=(m_general!=-1) ? m_filters.At(m_general) : NULL;
//---
   if(general==NULL)
     {
      m_ATR.Refresh(-1);
      //--- if a base price is not specified explicitly, take the current market price
      double base_price=(m_base_price==0.0) ? m_symbol.Bid() : m_base_price;

      //--- price overload that sets entry price to be based on ATR
      price      =m_symbol.NormalizePrice(base_price+(m_price_level*(m_ATR.Main(0)/m_symbol.Point()))*PriceLevelUnit());

      sl         =0.0;
      tp         =0.0;
      expiration+=m_expiration*PeriodSeconds(m_period);
      return(true);
     }
//---
   return(general.OpenShortParams(price,sl,tp,expiration));
  }
```

The MQL5 wizard generated expert advisor has an input parameter ‘Signal\_PriceLevel’. By default, it is zero but if assigned a value it represents the distance, in price points of the traded symbol, from the current price at which a market order will be placed. When this input is negative stop orders are placed. When positive limit orders are placed.It is a double data type. For our purposes this input will be a fraction or multiple of the current price points in the ATR.

3.1.2   _ATR trailing class_ is also a customised ‘CExpertTrailing’ class that also uses the ATR to set and move the stop loss. The implementation of its key functions is in the listing below.

```
//+------------------------------------------------------------------+
//| Checking trailing stop and/or profit for long position.          |
//+------------------------------------------------------------------+
bool CTrailingATR::CheckTrailingStopLong(CPositionInfo *position,double &sl,double &tp)
  {
//--- check
   if(position==NULL)
      return(false);
//---
   m_ATR.Refresh(-1);
   double level =NormalizeDouble(m_symbol.Bid()-m_symbol.StopsLevel()*m_symbol.Point(),m_symbol.Digits());

   //--- sl adjustment to be based on ATR
   double new_sl=NormalizeDouble(level-(m_atr_weight*(m_ATR.Main(0)/m_symbol.Point())),m_symbol.Digits());

   double pos_sl=position.StopLoss();
   double base  =(pos_sl==0.0) ? position.PriceOpen() : pos_sl;
//---
   sl=EMPTY_VALUE;
   tp=EMPTY_VALUE;
   if(new_sl>base && new_sl<level)
      sl=new_sl;
//---
   return(sl!=EMPTY_VALUE);
  }
//+------------------------------------------------------------------+
//| Checking trailing stop and/or profit for short position.         |
//+------------------------------------------------------------------+
bool CTrailingATR::CheckTrailingStopShort(CPositionInfo *position,double &sl,double &tp)
  {
//--- check
   if(position==NULL)
      return(false);
//---
   m_ATR.Refresh(-1);
   double level =NormalizeDouble(m_symbol.Ask()+m_symbol.StopsLevel()*m_symbol.Point(),m_symbol.Digits());

   //--- sl adjustment to be based on ATR
   double new_sl=NormalizeDouble(level+(m_atr_weight*(m_ATR.Main(0)/m_symbol.Point())),m_symbol.Digits());

   double pos_sl=position.StopLoss();
   double base  =(pos_sl==0.0) ? position.PriceOpen() : pos_sl;
//---
   sl=EMPTY_VALUE;
   tp=EMPTY_VALUE;
   if(new_sl<base && new_sl>level)
      sl=new_sl;
//---
   return(sl!=EMPTY_VALUE);
  }
```

Once again the 'm\_atr\_weight' will be an optimisable parameter, like with 'm\_price\_level', that sets how close we can trail open positions.

3.2  **Wizard assembly** will then be done in straightforward fashion with the only notable stages being the selection of our signal as shown below.

![wizard_1_crop](https://c.mql5.com/2/47/wizard_1_crop.png)

And the addition of our custom trailing method shown below.

![wizard_2_crop](https://c.mql5.com/2/47/wizard_2_crop.png)

### 4\. Testing in Strategy Tester

4.1  **Compilation** is what would follow assembly in the MQL5 wizard in order to create the expert advisor file and also confirm that there are no errors in our code.

4.2  **Default inputs** of the expert advisor would also need to be set in the strategy tester inputs tab. The key here is to ensure ‘Signal\_TakeLevel’ and ‘Signal\_StopLevel’ are set to zero. This is because as mentioned, for the purpose of this article, the exit is defined only by the trailing stop or the ‘Signal\_ThresholdClose’ input parameter.

4.3  **Optimisation** should be performed ideally with the real ticks of the broker you intend to trade with. For this article we will optimize EURUSD on the 4-hour timeframe across its V shaped period of 2018.01.01 to 2021.01.01. For comparison purposes we will run two optimizations: the first one will only use market orders, while the second will be open to pending orders. I use ‘open to’ pending orders because we will still consider the option of using only market orders given that ‘Signal\_PriceLevel’ can be zero since optimization is from a negative value to a positive value. Optimization [can be set](https://c.mql5.com/2/47/tester_1.jpg "https://c.mql5.com/2/47/tester_1.jpg") up prior for the option that uses pending orders as shown below. The only difference between this and the option that does not use pending orders is the later will leave 'Signal\_PriceLevel' input parameter at 0 not part of the optimised inputs.

![](https://c.mql5.com/2/47/tester_1.jpg)

4.4  **Results** from our optimisation are presented below. First is the report and equity curve of the best results from trading only with market orders.

![](https://c.mql5.com/2/47/tester_2.jpg)

![](https://c.mql5.com/2/47/curve_1__3.png)

Part of report 1,

Then a likewise report and curve from using pending orders.

![](https://c.mql5.com/2/47/tester_3.jpg)

![](https://c.mql5.com/2/47/curve_2__6.jpg)

Part of report 2.

It appears our regression analysis signal benefits from using pending orders by having less drawdowns at the sacrifice of some profits. Other modifications could be made as well to enhance this system such as changing the trailing stop class, or the money management type. For our testing purposes we used fixed margin percent and we optimised with our criteria set to 'complex criterion'. It is desirable however to test as extensively as possible on historical tick data and do sufficient forward walks before deploying things which are beyond this article's scope.

### 5\. Conclusion

5.1  **MQL5 wizard** is clearly a resourceful tool that should be in every trader’s arsenal. What we have considered here is how to take some statistical concepts of regression analysis like collinearity and coefficient of determination, and use them as a foundation to a robust trading system. Next steps would be extensive testing on historical tick data and exploring if this signal can be paired with other unique signals based on a trader's experience or on built-in signals in the MQL5 library in order to come up with a more comprehensive trading system. As will be the case in these series of articles, this article is not meant to provide a grail but rather a process which can be adjusted to better fit a trader's approach to the markets. Thanks for reading!

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/11066.zip "Download all attachments in the single ZIP archive")

[TrailingATR.mqh](https://www.mql5.com/en/articles/download/11066/trailingatr.mqh "Download TrailingATR.mqh")(6.24 KB)

[SignalDUAL\_RA.mqh](https://www.mql5.com/en/articles/download/11066/signaldual_ra.mqh "Download SignalDUAL_RA.mqh")(18.75 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/427351)**
(5)


![Guilherme Mendonca](https://c.mql5.com/avatar/2018/9/5B98163A-29AC.jpg)

**[Guilherme Mendonca](https://www.mql5.com/en/users/billy-gui)**
\|
9 Sep 2022 at 16:49

Hi Stephen,

Very good article. I made some tests and it returned good results. Looks like we have an encouraging trade system.

I would like to know how can I use input with non-fixed timeframes for optimization.

I've tried to change these lines, but OnInit "returned non-zero code 1" with no error message.

```
//--- inputs for expert
input string Expert_Title                     ="Regr2"; // Document name
ulong        Expert_MagicNumber               =26034;   //
bool         Expert_EveryTick                 =false;   //
input ENUM_TIMEFRAMES   timeframe             =PERIOD_M5;      //TimeFrame
//--- inputs for main signal
.
.
.
.
int OnInit()
  {
//--- Initializing expert
   if(!ExtExpert.Init(Symbol(),timeframe,Expert_EveryTick,Expert_MagicNumber))
     {
      //--- failed
      printf(__FUNCTION__+": error initializing expert");
      ExtExpert.Deinit();
      return(INIT_FAILED);
     }
//--- Creating signal
.
.
.
.
```

[![](https://c.mql5.com/3/393/327956708489__1.png)](https://c.mql5.com/3/393/327956708489.png "https://c.mql5.com/3/393/327956708489.png")

Can you help me?

![Gamuchirai Zororo Ndawana](https://c.mql5.com/avatar/2024/3/6607f08d-4cae.jpg)

**[Gamuchirai Zororo Ndawana](https://www.mql5.com/en/users/gamuchiraindawa)**
\|
10 Apr 2024 at 15:00

**Guilherme Mendonca [#](https://www.mql5.com/en/forum/427351#comment_41956991):**

Hi Stephen,

Very good article. I made some tests and it returned good results. Looks like we have an encouraging trade system.

I would like to know how can I use input with non-fixed timeframes for optimization.

I've tried to change these lines, but OnInit "returned non-zero code 1" with no error message.

Can you help me?

Hi, have you received help from Stephen?


![Stephen Njuki](https://c.mql5.com/avatar/avatar_na2.png)

**[Stephen Njuki](https://www.mql5.com/en/users/ssn)**
\|
20 Jun 2024 at 11:58

**Guilherme Mendonca [#](https://www.mql5.com/en/forum/427351#comment_41956991):**

Hi Stephen,

Very good article. I made some tests and it returned good results. Looks like we have an encouraging trade system.

I would like to know how can I use input with non-fixed timeframes for optimization.

I've tried to change these lines, but OnInit "returned non-zero code 1" with no error message.

Can you help me?

Hello,

Just seeing this. Sorry. This can be a bit tricky because wizard assembled Expert Advisors tend to use and stick to the timeframe of the chart they are attached to. And this is referenced in quite a few different places, not just in the OnInit() function that it seems you're trying to modify. So if your input time frame does not match up with the time frame used and expected in other places (to be the chart time frame), then this is bound to generate errors.

Big picture though, there is a case for reading data and price buffers in more than one time frame so I think in the near future I will look to do an article on how this could be achieved. Thx for your feedback.

![gardee005](https://c.mql5.com/avatar/avatar_na2.png)

**[gardee005](https://www.mql5.com/en/users/gardee005)**
\|
6 Nov 2024 at 16:09

I get an error when compiling an expert made using the SignalDUAL\_RA. it traces back to the [matrix](https://www.mql5.com/en/docs/matrix/matrix_types " MQL5 Documentation: Matrix and vector types").mqh

The CMatrixDouble is a function call to the MetaEditor built in classes, matrix.mqh, so i presume the error does not lie there.

Can you help me solve this?

Can any one help?

![gardee005](https://c.mql5.com/avatar/avatar_na2.png)

**[gardee005](https://www.mql5.com/en/users/gardee005)**
\|
7 Nov 2024 at 13:32

it would not compile for me with the previously mentioned issue (see screenshot). I solved the issue  as

\_a\[r\].Set(c,Data(r+c,close));         should be              \_a.Set(r,c,Data(r+c,close));

with the use of \_a\[r\] we are not editing the r-th row directly but passing it as a reference. This fix worked.

![How to master Machine Learning](https://c.mql5.com/2/47/machine-learning.png)[How to master Machine Learning](https://www.mql5.com/en/articles/10431)

Check out this selection of useful materials which can assist traders in improving their algorithmic trading knowledge. The era of simple algorithms is passing, and it is becoming harder to succeed without the use of Machine Learning techniques and Neural Networks.

![Developing a trading Expert Advisor from scratch (Part 8): A conceptual leap](https://c.mql5.com/2/45/development__1.png)[Developing a trading Expert Advisor from scratch (Part 8): A conceptual leap](https://www.mql5.com/en/articles/10353)

What is the easiest way to implement new functionality? In this article, we will take one step back and then two steps forward.

![DoEasy. Controls (Part 4): Panel control, Padding and Dock parameters](https://c.mql5.com/2/46/MQL5-avatar-doeasy-library-2__3.png)[DoEasy. Controls (Part 4): Panel control, Padding and Dock parameters](https://www.mql5.com/en/articles/10756)

In this article, I will implement handling Padding (internal indents/margin on all sides of an element) and Dock parameters (the way an object is located inside its container).

![DoEasy. Controls (Part 3): Creating bound controls](https://c.mql5.com/2/46/MQL5-avatar-doeasy-library-2__2.png)[DoEasy. Controls (Part 3): Creating bound controls](https://www.mql5.com/en/articles/10733)

In this article, I will create subordinate controls bound to the base element. The development will be performed using the base control functionality. In addition, I will tinker with the graphical element shadow object a bit since it still suffers from some logic errors when applied to any of the objects capable of having a shadow.

[![](https://www.mql5.com/ff/si/q0vxp9pq0887p07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fvps%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Duse.vps%26utm_content%3Drent.vps%26utm_campaign%3D0622.MQL5.com.Internal&a=rktadgjlwhobyedohbrepzshvpcqrlpo&s=a93cef75a53eb5da24c98e0068b3c2b96015191a0af0d1857f5b4dd22e55e7bf&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=zehdbsigostxwtdbjbmuwxxemfpxodxs&ssn=1769185844195983586&ssn_dr=0&ssn_sr=0&fv_date=1769185844&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F11066&back_ref=https%3A%2F%2Fwww.google.com%2F&title=MQL5%20Wizard%20techniques%20you%20should%20know%20(Part%2001)%3A%20Regression%20Analysis%20-%20MQL5%20Articles&scr_res=1920x1080&ac=17691858446855711&fz_uniq=5070345614971049011&sv=2552)

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