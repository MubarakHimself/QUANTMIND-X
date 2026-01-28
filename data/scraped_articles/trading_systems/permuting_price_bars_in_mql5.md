---
title: Permuting price bars in MQL5
url: https://www.mql5.com/en/articles/13591
categories: Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T19:23:08.853796
---

[![](https://www.mql5.com/ff/sh/a27a2kwmtszm2m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
MQL5 Channels - Messenger for traders\\
\\
Subscribe to traders' channels or create your own.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=vpcudokyepxfrcxrpjcktglhsjlemtza&s=f08ad2c1289e29bd5630f1ef977aef297d5cdbfcb686faed4a4b0f1e276d3c4a&uid=&ref=https://www.mql5.com/en/articles/13591&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5070242814928818794)

MetaTrader 5 / Examples


### Introduction

MetaTrader 5's Strategy Tester is the primary tool used by many to assess the potential of Expert Advisors (EA). Whilst its features are adequate, experienced developers can use it to produce "trick" EA's that are able to feign extraordinary performance. We have all seen those screen shots of  equity curves showing unbelievable performance from EA sellers. It all looks impressive at first glance but, often when the strategy is applied in the real world, it will produce a totally different equity curve. How can we spare ourselves the aggravation of falling for these cheap tricks? In this article we will look at  such a system, and demonstrate how permutation testing can be used to cut through the smoke screen of misleading equity curves to get a more accurate take on strategy performance. Also, in a previous article we saw the implementation of an algorithm to permute tick data. This time around we will describe a method for permuting price bars.

### Permuting OHLC data

Permuting price bars is little harder to accomplish because of the multiple series involved. Similar to permuting tick data, when handling price bars we strive to preserve the general trend of the original price series. Its also essential that we never allow the open or close of a bar go beyond or below the boundaries of the high or low respectively. The goal is to get a series of bars with a distribution of features exactly the same as the original data.

Besides the trend, we have to maintain the dispersion of price changes as the series progresses from the open to the close. The spread of price changes between the open and close should be the same in the permuted bars as the original. Outside the bars themselves we must make sure that the distribution of bar to bar price changes are the same as well. Specifically, the difference between the close of one bar and the open of the next.

This is important so as to not disadvantage the strategy being tested. The general characteristics of the series should be similar, the only difference should be the the absolute values of each open, high , low , close (OHLC) between the first and last bar. The code to implement  this is quite similar to that used in the CPermuteTicks class introduced in the article [Monte Carlo Permutation testing in MetaTrader 5](https://www.mql5.com/en/articles/13162 "/en/articles/13162"). The price bars permutation code will be encapsulated in the CPermuteRates class contained in PermuteRates.mqh.

### The CPermuteRates class

```
//+------------------------------------------------------------------+
//| struct to handle relative values of rate data to be worked on    |
//+------------------------------------------------------------------+
struct CRelRates
  {
   double            rel_open;
   double            rel_high;
   double            rel_low;
   double            rel_close;
  };

//+------------------------------------------------------------------+
//| Class to enable permuation of a collection of ticks in an array  |
//+------------------------------------------------------------------+
class CPermuteRates
  {
private :
   MqlRates          m_rates[];        //original rates to be shuffled
   CRelRates         m_differenced[];  //log difference of rates
   bool              m_initialized;    //flag to signal state of random number object
   CUniFrand         *m_random;        //random number generator

public :
   //constructor
                     CPermuteRates(void);
   //desctructor
                    ~CPermuteRates(void);
   bool              Initialize(MqlRates &in_rates[]);
   bool              Permute(MqlRates &out_rates[]);
  };
```

PermuteRate.mqh begins with the definition of a simple structure that will store the  log differences of raw prices.

-  rel\_open will hold the log difference between the current open and the previous bar's close

- rel\_high represents the log difference between the current bar high and open.

-  rel\_low refers to the log difference between the current bar low and  open

-  rel\_close is again the log difference between the current bar close and open

The custom CRelRates structure represents data extracted from MqlRates that will be permuted. Other struct members of MqlRates will not be altered. The final result of permuted rates will have these struct members copied from the original price series. As aready mentioned what will change are the OHLC values only.

```
//+------------------------------------------------------------------+
//| Permute the bars                                                 |
//+------------------------------------------------------------------+
bool CPermuteRates::Permute(MqlRates &out_rates[])
  {
//---
   if(!m_initialized)
     {
      Print("Initialization error");
      ZeroMemory(out_rates);
      return false;
     }
//---
   int i,j;
   double temp=0.0;
//---
   i=ArraySize(m_rates)-2;
//---
   while(i > 1 && !IsStopped())
     {
      j = (int)(m_random.RandomDouble() * i) ;
      if(j >= i)
         j = i - 1 ;
      --i ;
      temp = m_differenced[i+1].rel_open ;
      m_differenced[i+1].rel_open = m_differenced[j+1].rel_open ;
      m_differenced[j+1].rel_open = temp ;
     }
//---
   i =ArraySize(m_rates)-2;
//---
   while(i > 1  && !IsStopped())
     {
      j = (int)(m_random.RandomDouble() * i) ;
      if(j >= i)
         j = i - 1 ;
      --i ;
      temp = m_differenced[i].rel_high;
      m_differenced[i].rel_high = m_differenced[j].rel_high ;
      m_differenced[j].rel_high = temp ;
      temp = m_differenced[i].rel_low ;
      m_differenced[i].rel_low = m_differenced[j].rel_low ;
      m_differenced[j].rel_low = temp ;
      temp = m_differenced[i].rel_close ;
      m_differenced[i].rel_close = m_differenced[j].rel_close ;
      m_differenced[j].rel_close = temp ;
     }
//---
   if(ArrayCopy(out_rates,m_rates)!=int(m_rates.Size()))
     {
      ZeroMemory(out_rates);
      Print("Copy error ", GetLastError());
      return false;
     }
//---
   for(i=1 ; i<ArraySize(out_rates) && !IsStopped() ; i++)
     {
      out_rates[i].open  = MathExp(((MathLog(out_rates[i-1].close)) + m_differenced[i-1].rel_open)) ;
      out_rates[i].high  = MathExp(((MathLog(out_rates[i].open)) + m_differenced[i-1].rel_high)) ;
      out_rates[i].low   = MathExp(((MathLog(out_rates[i].open)) + m_differenced[i-1].rel_low)) ;
      out_rates[i].close = MathExp(((MathLog(out_rates[i].open)) + m_differenced[i-1].rel_close)) ;
     }
//---
   if(IsStopped())
      return false;
//---
   return true;
//---
  }
```

The permutation is done in the Permute() method. The CRelRates structure separates bar data into two types of descriptors. rel\_open series of values represents changes from one bar to the next whilst the rel\_high, rel\_low and rel\_close represent the changes within a bar. To permute the bars we first shuffle the series of rel\_open prices, these are the inter bar differences. From there the inner bar changes are shuffled. The new OHLC series are constructed from the shuffled inter bar data to get the new open values with corresponding high , low and close prices constructed from shuffled inner bar changes.

### Changes to CPermuteTicks

There are a number of differences between CPermuteRates and the old CPermuteTicks class. One of which is the use of a custom random number generator, which i found to be a little faster than using the MQL5's built in functions.

```
//+------------------------------------------------------------------+
//|                                                UniformRandom.mqh |
//|                        Copyright 2023, MetaQuotes Software Corp. |
//|                                             https://www.MQL5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2023, MetaQuotes Software Corp."
#property link      "https://www.MQL5.com"
//+----------------------------------------------------------------------+
//|  CUniFrand class: Uniformly distributed random 0 - 1 number generator|
//+----------------------------------------------------------------------+
class CUniFrand
  {
private :
   uint              m_m[256];
   int               m_mwc_initialized;
   int               m_mwc_seed;
   uint              m_carry;

   uint              random(void);

public :
   //constructor
                     CUniFrand(void);
   //desctructor
                    ~CUniFrand(void);
   //optionally set a seed for number generator
   void              SetSeed(const int iseed);
   //get random number between 0 and 1
   double            RandomDouble(void);
  };
//+------------------------------------------------------------------+
//|  Default constructor                                             |
//+------------------------------------------------------------------+
CUniFrand::CUniFrand(void)
  {
   m_mwc_initialized=0;
   m_mwc_seed=123456789;
   m_carry=362436;
  }
//+------------------------------------------------------------------+
//|   Destructor                                                     |
//+------------------------------------------------------------------+
CUniFrand::~CUniFrand(void)
  {
  }
//+------------------------------------------------------------------+
//| creates and returns random integer number                        |
//+------------------------------------------------------------------+
uint CUniFrand::random(void)
  {
   uint t,a=809430660;
   static uchar i;
   if(!m_mwc_initialized)
     {
      uint k,j=m_mwc_seed;
      m_mwc_initialized=1;
      for(k=0; k<256; k++)
        {
         j = 69069 * j + 12345;
         m_m[k]=j;
        }
     }

   t=a*m_m[++i] + m_carry;
   m_carry = (uint)(t>>32);
   m_m[i]  = (uint)(t&UINT_MAX);

   return m_m[i];
  }
//+------------------------------------------------------------------+
//| Optionally set the seed for random number generator              |
//+------------------------------------------------------------------+
void CUniFrand::SetSeed(const int iseed)
  {
   m_mwc_seed=iseed;
   m_mwc_initialized=0;
  }
//+------------------------------------------------------------------+
//| returns a random number between 0 and 1                          |
//+------------------------------------------------------------------+
double CUniFrand::RandomDouble(void)
  {
   double mult =1.0/UINT_MAX;
   return mult * random();
  }
//+------------------------------------------------------------------+
```

It is also applied to the  new CPermuteTicks class. Unnecessary intermediary operations have been eliminated for the sake of efficiency. Only the bid prices are shuffled. With other tick properties being copied from the original tick series, this solves a problem that would sometimes result in permuted ticks with unrealistic spreads. The new CPermuteTick series is shown below.

```
//+------------------------------------------------------------------+
//|                                                 PermuteTicks.mqh |
//|                                  Copyright 2023, MetaQuotes Ltd. |
//|                                             https://www.MQL5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2023, MetaQuotes Ltd."
#property link      "https://www.MQL5.com"
#include<UniformRandom.mqh>
//+------------------------------------------------------------------+
//| Class to enable permuation of a collection of ticks in an array  |
//+------------------------------------------------------------------+
class CPermuteTicks
  {
private :
   MqlTick           m_ticks[];        //original tick data to be shuffled
   double            m_differenced[];  //log difference of tick data
   bool              m_initialized;    //flag representing proper preparation of a dataset
   CUniFrand         *m_random;
public :
   //constructor
                     CPermuteTicks(void);
   //desctrucotr
                    ~CPermuteTicks(void);
   bool              Initialize(MqlTick &in_ticks[]);
   bool              Permute(MqlTick &out_ticks[]);
  };
//+------------------------------------------------------------------+
//| constructor                                                      |
//+------------------------------------------------------------------+
CPermuteTicks::CPermuteTicks(void):m_initialized(false)
  {
   m_random = new CUniFrand();
   m_random.SetSeed(MathRand());
  }
//+------------------------------------------------------------------+
//| destructor                                                       |
//+------------------------------------------------------------------+
CPermuteTicks::~CPermuteTicks(void)
  {
   delete m_random;
//---clean up
   ArrayFree(m_ticks);
//---
   ArrayFree(m_differenced);
//---
  }

//+--------------------------------------------------------------------+
//|Initialize the permutation process by supplying ticks to be permuted|
//+--------------------------------------------------------------------+
bool CPermuteTicks::Initialize(MqlTick &in_ticks[])
  {
//---check the random number object
   if(m_random==NULL)
     {
      Print("Critical internal error, failed to initialize random number generator");
      return false;
     }
//---set or reset initialization flag
   m_initialized=false;
//---check arraysize
   if(in_ticks.Size()<5)
     {
      Print("Insufficient amount of data supplied ");
      return false;
     }
//---copy ticks to local array
   if(ArrayCopy(m_ticks,in_ticks)!=int(in_ticks.Size()))
     {
      Print("Error copying ticks ", GetLastError());
      return false;
     }
//---ensure the size of m_differenced array
   if(m_differenced.Size()!=m_ticks.Size()-1)
      ArrayResize(m_differenced,m_ticks.Size()-1);
//---fill m_differenced with differenced values, excluding the first tick
   for(uint i=1; i<m_ticks.Size() && !IsStopped(); i++)
     {
      m_differenced[i-1]=MathLog(m_ticks[i].bid/m_ticks[i-1].bid);//(m_logticks[i])-(m_logticks[i-1]);
     }
//---set the initilization flag
   m_initialized=true;
//---
   return true;
  }

//+------------------------------------------------------------------+
//|Public method which applies permutation and gets permuted ticks   |
//+------------------------------------------------------------------+
bool CPermuteTicks::Permute(MqlTick &out_ticks[])
  {
//---ensure required data already supplied through initialization
   if(!m_initialized)
     {
      Print("not initialized");
      return false;
     }
//---
   int i,j;
   double tempvalue;

   i=(int)m_ticks.Size()-1;

   while(i>1 && !IsStopped())
     {
      j=(int)(m_random.RandomDouble()*i);
      if(j>=i)
         j=i-1;
      --i;
      //---swap tick data randomly
      tempvalue=m_differenced[i];
      m_differenced[i]=m_differenced[j];
      m_differenced[j]=tempvalue;

     }
//----
   if(IsStopped())
      return false;
//---copy the first tick
   if(ArrayCopy(out_ticks,m_ticks)!=int(m_ticks.Size()))
     {
      Print(__FUNCTION__," array copy failure ", GetLastError());
      return false;
     }
//---apply exponential transform to data and copy original tick data member info
//---not involved in permutation operations
   for(uint k = 1; k<m_ticks.Size() && !IsStopped(); k++)
     {
      out_ticks[k].bid=MathExp((MathLog(out_ticks[k-1].bid) + m_differenced[k-1]));//MathExp(m_logticks[k]);
      out_ticks[k].ask=out_ticks[k].bid + (m_ticks[k].ask - m_ticks[k].bid);
     }
//---
   if(IsStopped())
      return false;
   else
      return true;
  }
//+------------------------------------------------------------------+
```

CPermuteTicks still works in the same manner as previous version and CPermuteRates works similarly. The difference between the two is  one works with ticks whilst the other works with rates

### The CPermutedSymbolData class

The script PrepareSymbolsForPermutationTest has been updated to reflect the changes made to CPermuteTicks and the introduction of CPermuteRates. The functionality of the script is enclosed in the class CPermutedSymbolData, it enables the production of custom symbols with either permuted ticks or rates based on an existing symbol.

```
//+------------------------------------------------------------------+
//|Permute rates or ticks of symbol                                  |
//+------------------------------------------------------------------+
enum ENUM_RATES_TICKS
  {
   ENUM_USE_RATES=0,//Use rates
   ENUM_USE_TICKS//Use ticks
  };
//+------------------------------------------------------------------+
//| defines:max number of data download attempts and array resize    |
//+------------------------------------------------------------------+
#define MAX_DOWNLOAD_ATTEMPTS 10
#define RESIZE_RESERVE 100
//+------------------------------------------------------------------+
//|CPermuteSymbolData class                                          |
//| creates custom symbols from an existing base symbol's  data      |
//|  symbols represent permutations of base symbol's data            |
//+------------------------------------------------------------------+
class CPermuteSymbolData
  {
private:
   ENUM_RATES_TICKS  m_use_rates_or_ticks;//permute either ticks or rates
   string            m_basesymbol;        //base symbol
   string            m_symbols_id;        //common identifier added to names of new symbols
   datetime          m_datarangestart;    //beginning date for range of base symbol's data
   datetime          m_datarangestop;     //ending date for range of base symbol's data
   uint              m_permutations;      //number of permutations and ultimately the number of new symbols to create
   MqlTick           m_baseticks[];       //base symbol's tick
   MqlTick           m_permutedticks[];   //permuted ticks;
   MqlRates          m_baserates[];       //base symbol's rates
   MqlRates          m_permutedrates[];   //permuted rates;
   CPermuteRates     *m_rates_shuffler;    //object used to shuffle rates
   CPermuteTicks     *m_ticks_shuffler;    //object used to shuffle ticks
   CNewSymbol        *m_csymbols[];        //array of created symbols

public:
                     CPermuteSymbolData(const ENUM_RATES_TICKS mode);
                    ~CPermuteSymbolData(void);
   bool              Initiate(const string base_symbol,const string symbols_id,const datetime start_date,const datetime stop_date);
   uint              Generate(const uint permutations);
  };
```

This is achieved by specifying the type of data to be shuffled , either ticks or rates, in the constructor call. The enumeration ENUM\_RATES\_TICKS describes the options available to the constructor's single parameter.

```
//+-----------------------------------------------------------------------------------------+
//|set and check parameters for symbol creation, download data and initialize data shuffler |
//+-----------------------------------------------------------------------------------------+
bool CPermuteSymbolData::Initiate(const string base_symbol,const string symbols_id,const datetime start_date,const datetime stop_date)
  {
//---reset number of permutations previously done
   m_permutations=0;
//---set base symbol
   m_basesymbol=base_symbol;
//---make sure base symbol is selected, ie, visible in WatchList
   if(!SymbolSelect(m_basesymbol,true))
     {
      Print("Failed to select ", m_basesymbol," error ", GetLastError());
      return false;
     }
//---set symbols id
   m_symbols_id=symbols_id;
//---check, set data date range
   if(start_date>=stop_date)
     {
      Print("Invalid date range ");
      return false;
     }
   else
     {
      m_datarangestart= start_date;
      m_datarangestop = stop_date;
     }
//---download data
   Comment("Downloading data");
   uint attempts=0;
   int downloaded=-1;
   while(attempts<MAX_DOWNLOAD_ATTEMPTS && !IsStopped())
     {
      downloaded=(m_use_rates_or_ticks==ENUM_USE_TICKS)?CopyTicksRange(m_basesymbol,m_baseticks,COPY_TICKS_ALL,long(m_datarangestart)*1000,long(m_datarangestop)*1000):CopyRates(m_basesymbol,PERIOD_M1,m_datarangestart,m_datarangestop,m_baserates);
      if(downloaded<=0)
        {
         Sleep(500);
         ++attempts;
        }
      else
         break;
     }
//---check download result
   if(downloaded<=0)
     {
      Print("Failed to download data for ",m_basesymbol," error ", GetLastError());
      Comment("");
      return false;
     }

//Print(downloaded," Ticks downloaded ", " data start ",m_basedata[0].time, " data end ", m_basedata[m_basedata.Size()-1].time);
//---return shuffler initialization result
   switch(m_use_rates_or_ticks)
     {
      case ENUM_USE_TICKS:
        {
         if(m_ticks_shuffler==NULL)
            m_ticks_shuffler=new CPermuteTicks();
         return m_ticks_shuffler.Initialize(m_baseticks);
        }
      case ENUM_USE_RATES:
        {
         if(m_rates_shuffler==NULL)
            m_rates_shuffler=new CPermuteRates();
         return m_rates_shuffler.Initialize(m_baserates);
        }
      default:
         return false;
     }
  }
```

Once an instance of CPermutedSymbolData is created, the Initiate() method should be called to specify the  symbol and the date period defining the ticks or rates that permutations will be based on.

```
//+------------------------------------------------------------------+
//| generate symbols return newly created or refreshed symbols       |
//+------------------------------------------------------------------+
uint CPermuteSymbolData::Generate(const uint permutations)
  {
//---check permutations
   if(!permutations)
     {
      Print("Invalid parameter value for Permutations ");
      Comment("");
      return 0;
     }
//---resize m_csymbols
   if(m_csymbols.Size()!=m_permutations+permutations)
      ArrayResize(m_csymbols,m_permutations+permutations,RESIZE_RESERVE);
//---
   string symspath=m_basesymbol+m_symbols_id+"_PermutedData";
//int exists;
//---do more permutations
   for(uint i=m_permutations; i<m_csymbols.Size() && !IsStopped(); i++)
     {
      if(CheckPointer(m_csymbols[i])==POINTER_INVALID)
         m_csymbols[i]=new CNewSymbol();

      if(m_csymbols[i].Create(m_basesymbol+m_symbols_id+"_"+string(i+1),symspath,m_basesymbol)<0)
         continue;

      Comment("Processing Symbol "+m_basesymbol+m_symbols_id+"_"+string(i+1));
      if(!m_csymbols[i].Clone(m_basesymbol) ||
         (m_use_rates_or_ticks==ENUM_USE_TICKS && !m_ticks_shuffler.Permute(m_permutedticks)) ||
         (m_use_rates_or_ticks==ENUM_USE_RATES && !m_rates_shuffler.Permute(m_permutedrates)))
         break;
      else
        {
         m_csymbols[i].Select(true);
         Comment("Adding permuted data");
         if(m_use_rates_or_ticks==ENUM_USE_TICKS)
            m_permutations+=(m_csymbols[i].TicksReplace(m_permutedticks)>0)?1:0;
         else
            m_permutations+=(m_csymbols[i].RatesUpdate(m_permutedrates)>0)?1:0;
        }
     }
//---return successfull number of permutated symbols
   Comment("");
//---
   if(IsStopped())
      return 0;
//---
   return m_permutations;
  }
//+------------------------------------------------------------------+
```

If Initiate() returns true, the Generate() method can be called with the number of required permutations. The method will return the count of custom symbols whose data has been successfully replenished with permuted ticks or rates.

```
//+------------------------------------------------------------------+
//|                            PrepareSymbolsForPermutationTests.mq5 |
//|                                  Copyright 2023, MetaQuotes Ltd. |
//|                                             https://www.MQL5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2023, MetaQuotes Ltd."
#property link      "https://www.MQL5.com"
#property version   "1.00"
#include<PermutedSymbolData.mqh>
#property script_show_inputs

//--- input parameters
input string   BaseSymbol="EURUSD";
input ENUM_RATES_TICKS PermuteRatesOrTicks=ENUM_USE_RATES;
input datetime StartDate=D'2022.01.01 00:00';
input datetime EndDate=D'2023.01.01 00:00';
input uint     Permutations=100;
input string   CustomID="_p";//SymID to be added to symbol permutation names
//---
CPermuteSymbolData *symdata;
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
   ulong startime = GetTickCount64();
   uint permutations_completed=0; // number of successfully added permuted data
//---intialize the permuted symbol object
   symdata = new CPermuteSymbolData(PermuteRatesOrTicks);
//---set the properties of the permuted symbol object
   if(symdata.Initiate(BaseSymbol,CustomID,StartDate,EndDate))
      permutations_completed = symdata.Generate(Permutations);   // do the permutations
//---print number of symbols whose bar or tick data has been replenished.
   Print("Number of permuted symbols is ", permutations_completed, ", Runtime ",NormalizeDouble(double(GetTickCount64()-startime)/double(60000),1),"mins");
//---clean up
   delete symdata;
  }
//+------------------------------------------------------------------+
```

Above is the code of the script. All source code is attached to the article.

### Applying permutation testing

In the introduction of the article we talked of a common problem that is faced by many looking to purchase Expert Advisors. There is the possibility that unscrupulous sellers may employ deceptive tactics to market their products. Often sellers will display screeshots of attractive equity curves to represent the potential profits. Many have fallen victim to these tactics and learnt the hard way that these screenshots were produced from induced strategies. In this section we will have a look at an infamous EA available in the code base that can be used to produce misleading equity curves. And apply a permutation test to uncover the deception.

![Equity Curve](https://c.mql5.com/2/59/Curve.PNG)

### Overview of the permutation test

It should be noted that this kind of testing is quite laborious and requires significant time and compute resources. Though, in my opinion, the results are well worth the effort and can save one from making a terrible decision. The method employed entails selecting an appropriate sample for testing. Separating this sample into  in-sample and out of sample data sets. The EA will be optimized on the in-sample data and  final performance recorded on tests conducted on  out of sample data  using the optimized  parameters. This is done using the original data series and as well as on at least 100 permuted data sets. This is exactly what was done to test the EA used in our demonstration.

### Testing the grr-al EA

Anyone who has studied the MQL5 documentation or explored the code base must have come across this EA. In the MQL5 documentation it is described as a "Testing Grail". When it is run in the Strategy Tester in either 1 Minute OHLC or Open prices only tick generation mode an equity curve with an impressive profile is produced. This is the EA we will use in our demonstration. We have modified the code slightly to expose  a few of the global variables for optimization. Two of the the 3 parameters were selected for optimization namely SL, the stoploss and DELTA.

```
#define MAGIC_NUMBER 12937
#define DEV 20
#define RISK 0.0
#define BASELOT 0.1

input double DELTA =30;
input double SL =700;
input double TP =100;
```

The settings used for optimization are shown in the graphic.

![Optimization inputs](https://c.mql5.com/2/59/Inputs.PNG)

The data set selected was that of EURUSD, for the entire 2022 year on the hourly timeframe. The first six months of the year 2022 were used for optimization, and the last half used as the out of sample period for testing the optimal parameters.

![Optimization Settings](https://c.mql5.com/2/59/Settings.PNG)

First the script PrepareSymbolsForPermutationsTests was used to generate the custom symbols of permutated data. The program run was timed and noted as shown below. The error code is because i ran out of drive space, on my first attempt and only 99 custom symbols were successfully added.

```
PR      0       11:53:04.548    PrepareSymbolsForPermutationTests (EURUSD,MN1)  CNewSymbol::TicksReplace: failed to replace ticks! Error code: 5310
EL      0       11:53:04.702    PrepareSymbolsForPermutationTests (EURUSD,MN1)  Number of permuted symbols is 99, Runtime 48.9mins
```

The amount of data generated was almost 40 gigabytes of tick data for one year of data permuted 100 times!

![Tick Data Folder size](https://c.mql5.com/2/59/TickDataFileSize.gif)

For interest sake, using rates was much faster and took up much less space.

```
NK      0       12:51:23.166    PrepareSymbolsForPermutationTests (EURUSD,M1)   Number of permuted symbols is 100, Runtime 1.4mins
```

Using this data each symbol was optimized on the in-sample sets.

![Optimization results snapshot](https://c.mql5.com/2/59/Optimization.gif)

The parameters that produced the largest absolute profit were used in the out of sample test. Optimization and out of sample testing was conducted using open prices only tick mode. This means the EA had every advantage to shine.

The results from all tests are shown below presented in a csv file. IS Profit and OOS PROFIT are the in sample and out of sample profit respectively

```
<SYMBOL> <OPTIMAL DELTA> <OPTIMAL SL> <IS PROFIT> <OOS PROFIT>
EURUSD 3.00 250.00 31995.60 32347.20
EURUSD_p_1 3.00 50.00 29283.40 34168.20
EURUSD_p_2 5.00 50.00 32283.50 21047.60
EURUSD_p_3 3.00 20.00 33696.20 34915.30
EURUSD_p_4 3.00 20.00 32589.30 38693.20
EURUSD_p_5 3.00 230.00 33771.10 40458.20
EURUSD_p_6 3.00 40.00 30899.10 34061.50
EURUSD_p_7 3.00 250.00 34309.10 31861.20
EURUSD_p_8 3.00 40.00 33729.00 35359.90
EURUSD_p_9 3.00 300.00 36027.90 38174.50
EURUSD_p_10 3.00 30.00 33405.90 35693.70
EURUSD_p_11 3.00 30.00 32723.30 36453.00
EURUSD_p_12 11.00 300.00 34191.20 34277.80
EURUSD_p_13 3.00 130.00 35029.70 33930.00
EURUSD_p_14 11.00 290.00 33924.40 34851.70
EURUSD_p_15 3.00 140.00 33920.50 32263.20
EURUSD_p_16 3.00 20.00 34388.00 33694.40
EURUSD_p_17 3.00 60.00 35081.70 35612.20
EURUSD_p_18 5.00 70.00 36830.00 40442.30
EURUSD_p_19 3.00 170.00 37693.70 37404.90
EURUSD_p_20 3.00 50.00 31265.30 34875.10
EURUSD_p_21 3.00 20.00 30248.10 38426.00
EURUSD_p_22 5.00 250.00 32369.80 37263.80
EURUSD_p_23 7.00 50.00 31197.50 35466.40
EURUSD_p_24 7.00 30.00 26252.20 34963.10
EURUSD_p_25 3.00 20.00 31343.90 37156.00
EURUSD_p_26 25.00 280.00 29762.10 27336.10
EURUSD_p_27 3.00 60.00 33775.10 37034.60
EURUSD_p_28 3.00 260.00 35341.70 36744.20
EURUSD_p_29 5.00 50.00 31775.80 34673.60
EURUSD_p_30 3.00 20.00 32520.30 37907.10
EURUSD_p_31 3.00 230.00 35481.40 42938.20
EURUSD_p_32 3.00 100.00 32862.70 38291.70
EURUSD_p_33 3.00 190.00 36511.70 26714.30
EURUSD_p_34 3.00 290.00 29809.10 35312.40
EURUSD_p_35 3.00 290.00 34044.60 33460.00
EURUSD_p_36 3.00 90.00 32203.10 35730.90
EURUSD_p_37 3.00 180.00 39506.50 30947.30
EURUSD_p_38 3.00 180.00 35844.90 41717.30
EURUSD_p_39 3.00 90.00 30602.30 35390.10
EURUSD_p_40 3.00 250.00 29592.20 33025.90
EURUSD_p_41 3.00 140.00 34281.80 31501.40
EURUSD_p_42 3.00 30.00 34235.70 39422.40
EURUSD_p_43 3.00 170.00 35580.10 35994.20
EURUSD_p_44 3.00 20.00 34400.60 36250.50
EURUSD_p_45 5.00 190.00 35942.70 31068.30
EURUSD_p_46 3.00 20.00 32560.60 37114.70
EURUSD_p_47 3.00 200.00 36837.30 40843.10
EURUSD_p_48 3.00 20.00 29188.30 33418.10
EURUSD_p_49 3.00 40.00 33985.60 29720.50
EURUSD_p_50 3.00 250.00 36849.00 38007.00
EURUSD_p_51 3.00 50.00 33867.90 39323.30
EURUSD_p_52 3.00 120.00 33066.30 39852.40
EURUSD_p_53 3.00 60.00 36977.30 37284.40
EURUSD_p_54 3.00 20.00 29990.30 35975.70
EURUSD_p_55 15.00 70.00 29872.80 34179.40
EURUSD_p_56 3.00 250.00 35909.60 35911.50
EURUSD_p_57 3.00 200.00 37642.70 34849.80
EURUSD_p_58 3.00 290.00 39164.00 35440.90
EURUSD_p_59 3.00 100.00 28312.70 33917.80
EURUSD_p_60 3.00 60.00 28141.60 38826.00
EURUSD_p_61 3.00 50.00 29670.90 34973.70
EURUSD_p_62 3.00 40.00 32170.80 31062.60
EURUSD_p_63 3.00 260.00 28312.80 29236.50
EURUSD_p_64 3.00 20.00 31632.50 35458.30
EURUSD_p_65 3.00 260.00 35345.20 38522.70
EURUSD_p_66 7.00 270.00 31077.60 34531.10
EURUSD_p_67 3.00 90.00 33893.70 30969.00
EURUSD_p_68 3.00 170.00 34118.70 37280.50
EURUSD_p_69 3.00 40.00 33867.50 35256.20
EURUSD_p_70 3.00 180.00 37710.60 30337.20
EURUSD_p_71 5.00 200.00 40851.10 40985.60
EURUSD_p_72 3.00 20.00 29258.40 31194.70
EURUSD_p_73 3.00 20.00 30956.50 38021.40
EURUSD_p_74 3.00 90.00 35807.40 32625.70
EURUSD_p_75 3.00 260.00 32801.10 36161.70
EURUSD_p_76 3.00 260.00 34825.40 28957.70
EURUSD_p_77 3.00 90.00 39725.80 35923.00
EURUSD_p_78 3.00 180.00 37880.80 37090.90
EURUSD_p_79 3.00 180.00 34191.50 38190.70
EURUSD_p_80 3.00 40.00 29235.30 33207.70
EURUSD_p_81 3.00 20.00 29923.50 34291.00
EURUSD_p_82 3.00 90.00 35077.80 37203.40
EURUSD_p_83 3.00 40.00 32901.50 32182.40
EURUSD_p_84 3.00 50.00 31302.60 34339.00
EURUSD_p_85 3.00 60.00 30336.90 37948.10
EURUSD_p_86 5.00 50.00 35166.10 37898.60
EURUSD_p_87 5.00 290.00 33005.20 32648.30
EURUSD_p_88 7.00 140.00 34349.70 31435.50
EURUSD_p_89 3.00 20.00 30680.20 37002.30
EURUSD_p_90 3.00 100.00 35382.50 37643.80
EURUSD_p_91 3.00 50.00 35187.20 36392.00
EURUSD_p_92 3.00 120.00 32423.10 35943.20
EURUSD_p_93 3.00 100.00 31722.70 39913.30
EURUSD_p_94 11.00 300.00 31548.40 32684.70
EURUSD_p_95 3.00 100.00 30094.00 38929.70
EURUSD_p_96 3.00 170.00 35400.30 29260.30
EURUSD_p_97 3.00 300.00 35696.50 35772.20
EURUSD_p_98 3.00 20.00 31336.20 35935.70
EURUSD_p_99 3.00 20.00 32466.30 39986.40
EURUSD_p_100 3.00 20.00 32082.40 33625.10
```

The calculated p-value is given as 0,8217821782178217.

```
MO      0       09:49:57.991    ProcessOptFiles (EURUSD,MN1)    P-value is 0.8217821782178217
```

This asserts that the probability of observing the performance attained on the original dataset by luck is over 80 percent. This clearly indicates that this EA is worthless.

### Why does this work?

The premise of permutation testing within the context of strategy development is that, a EA strategy is a description of a pattern or a set of rules used to gain an edge in trading. When the data that it works on is permuted the original patterns that it would otherwise profit from would have been disrupted. If the EA does in fact trade based on some pattern its performance on permuted data will suffer. When performance from permuted and unpermuted tests are compared it becomes clear that even after optimization the EA does in fact rely on some unique pattern or rule. The performance from the unpermuted data set should stand out from the permuted tests.

As we saw from the demonstrated test, the EA in question is known to exploit the method of tick generation and does not employ any real strategy (patterns or rules). The permutation test was able to reveal this.

Permutation tests can also be used to give an indication of the extent of overfitting after optimization. To test for overfitting we would need to test and compare the in-sample performance from the permuted and unpermuted data sets. The extent to which the unpermuted performance figures differ from the permuted  results can be used to quantify overfitting. When overfitting is prevalent there will be little difference between permuted and unpermuted performance results. We would see fairly large p-values.

### Conclusion

We saw the implementation of an algorithm for permuting price bars. As well as updated code for generating custom symbols with permuted ticks or bars. The progams described were used to demonstrate a permutation test on an EA with induced positive performance results. Permutation testing is an essential tool for anyone interested in automated trading. So essential, i think it should be added as a feature of Mt5's strategy tester .

| File | Description |
| --- | --- |
| MQL5\\Experts\\grr-al.mq5 | This a slightly modified version of the EA available in MQL5.com's codebase, it trades by exploiting the tick generation method of the strategy tester in 1 minute OHLC mode. |
| MQL5\\Include\\NewSymbol.mqh | contains CNewSymbol class definition for creating custom symbols |
| MQL5\\Include\ PermutedSymbolData.mqh | defines the CPermutedSymbolData class for creating custom symbols with permuted rates or ticks |
| MQL5\\Include\\PermuteRates.mqh | contains the CPermuteRates class for creating permutations of an array of MqlRates data |
| MQL5\\Include\\PermuteTicks.mqh | defines the CPermuteTicks class for creating permutations of an array of MqlTick data |
| MQL5\\Include\\UniformRandom.mqh | CUniFrand encapsulates a uniformly distributed random number generator |
| MQL5\\Scripts\\PrepareSymbolsForPermutationTests.mq5 | this is script that times all the code utilities together to generate custom symbols in MetaTrader 5 |

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/13591.zip "Download all attachments in the single ZIP archive")

[grr-al.mq5](https://www.mql5.com/en/articles/download/13591/grr-al.mq5 "Download grr-al.mq5")(4.94 KB)

[NewSymbol.mqh](https://www.mql5.com/en/articles/download/13591/newsymbol.mqh "Download NewSymbol.mqh")(29.34 KB)

[PermutedSymbolData.mqh](https://www.mql5.com/en/articles/download/13591/permutedsymboldata.mqh "Download PermutedSymbolData.mqh")(8.18 KB)

[PermuteRates.mqh](https://www.mql5.com/en/articles/download/13591/permuterates.mqh "Download PermuteRates.mqh")(6.04 KB)

[PermuteTicks.mqh](https://www.mql5.com/en/articles/download/13591/permuteticks.mqh "Download PermuteTicks.mqh")(4.79 KB)

[UniformRandom.mqh](https://www.mql5.com/en/articles/download/13591/uniformrandom.mqh "Download UniformRandom.mqh")(2.84 KB)

[PrepareSymbolsForPermutationTests.mq5](https://www.mql5.com/en/articles/download/13591/preparesymbolsforpermutationtests.mq5 "Download PrepareSymbolsForPermutationTests.mq5")(1.91 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/456290)**
(2)


![Aleksandr Slavskii](https://c.mql5.com/avatar/2017/4/58E88E5E-2732.jpg)

**[Aleksandr Slavskii](https://www.mql5.com/en/users/s22aa)**
\|
29 Feb 2024 at 13:40

Theworld's fastest speed in a car or any land guided vehicle \- **1,228 km/h** \-  was demonstrated on a Thrust SSC jet car by Englishman Andy Green on 15 October 1997 .

The 21 kilometre long track was marked on the bottom of a dried up lake in the Black Rock Desert, Nevada, USA.

//---

Let's see if this was fabricated to deceive the public.

We'll make the exact same track, but put rocks on it and try to drive it.

So we find out that the probability of getting the same speed as Andy Green tends to zero per cent. This clearly indicates that the ̶c̶о̶о̶в̶е̶t̶n̶и̶k̶k̶ speed record is fictitious.

:)

//---

The example of using this method of getting custom quotes, is not correct.

![Aleksey Vyazmikin](https://c.mql5.com/avatar/2024/6/6678986f-2caa.png)

**[Aleksey Vyazmikin](https://www.mql5.com/en/users/-aleks-)**
\|
15 Sep 2024 at 20:22

Permutation tests can also be used для определения степени переобучения after optimisation. To test for overtraining, we will need to test and compare performance in a sample of permutation and non permutation datasets. The extent to which the non-rearrangement performance differs from the permutation-dominated results can be used to quantify overtraining. When overlearning dominates, the difference between permutation and non permutation performance results will be small. We will see fairly large p values.

It is not clear why an overtrained model should perform well at random? As far as I know, on the contrary, if one is testing on generated plausible data, the model is considered more robust by generalising rather than memorising (overtraining).


![Neural networks made easy (Part 42): Model procrastination, reasons and solutions](https://c.mql5.com/2/54/NN_Simple_Part_42_procrastination_avatar.png)[Neural networks made easy (Part 42): Model procrastination, reasons and solutions](https://www.mql5.com/en/articles/12638)

In the context of reinforcement learning, model procrastination can be caused by several reasons. The article considers some of the possible causes of model procrastination and methods for overcoming them.

![Integrate Your Own LLM into EA (Part 2): Example of Environment Deployment](https://c.mql5.com/2/59/penguin-image.png)[Integrate Your Own LLM into EA (Part 2): Example of Environment Deployment](https://www.mql5.com/en/articles/13496)

With the rapid development of artificial intelligence today, language models (LLMs) are an important part of artificial intelligence, so we should think about how to integrate powerful LLMs into our algorithmic trading. For most people, it is difficult to fine-tune these powerful models according to their needs, deploy them locally, and then apply them to algorithmic trading. This series of articles will take a step-by-step approach to achieve this goal.

![Neural networks made easy (Part 43): Mastering skills without the reward function](https://c.mql5.com/2/54/NN_Simple_Part_43_avatar.png)[Neural networks made easy (Part 43): Mastering skills without the reward function](https://www.mql5.com/en/articles/12698)

The problem of reinforcement learning lies in the need to define a reward function. It can be complex or difficult to formalize. To address this problem, activity-based and environment-based approaches are being explored to learn skills without an explicit reward function.

![Neural networks made easy (Part 41): Hierarchical models](https://c.mql5.com/2/54/NN_Simple_Part_41_Hierarchical_Models_Avatars.png)[Neural networks made easy (Part 41): Hierarchical models](https://www.mql5.com/en/articles/12605)

The article describes hierarchical training models that offer an effective approach to solving complex machine learning problems. Hierarchical models consist of several levels, each of which is responsible for different aspects of the task.

[![](https://www.mql5.com/ff/si/q0vxp9pq0887p07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fvps%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Duse.vps%26utm_content%3Drent.vps%26utm_campaign%3D0622.MQL5.com.Internal&a=rktadgjlwhobyedohbrepzshvpcqrlpo&s=a93cef75a53eb5da24c98e0068b3c2b96015191a0af0d1857f5b4dd22e55e7bf&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=swxsvlmscswdugnfoowvoneejovcbjip&ssn=1769185387104845729&ssn_dr=0&ssn_sr=0&fv_date=1769185387&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F13591&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Permuting%20price%20bars%20in%20MQL5%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918538711954190&fz_uniq=5070242814928818794&sv=2552)

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