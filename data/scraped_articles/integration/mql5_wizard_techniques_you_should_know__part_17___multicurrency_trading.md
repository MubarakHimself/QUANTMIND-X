---
title: MQL5 Wizard Techniques you should know (Part 17): Multicurrency Trading
url: https://www.mql5.com/en/articles/14806
categories: Integration, Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T21:10:08.545453
---

[![](https://www.mql5.com/ff/sh/qv94j0cd8n2n55z9z2/01.png)![](https://www.mql5.com/ff/sh/qv94j0cd8n2n55z9z2/02.png)Boost your trading experienceRead our book "MQL5 Programming for Traders"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/book%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.algobook%26utm_content=visit.page%26utm_campaign=algobook.promo.04.2024&a=heclgjpfbvfghpmyaciuaesdtswflupo&s=4255fbe1b8cbc4d1b40afbaebf4235e5ace8b5103cba60d996897a03d588556f&uid=&ref=https://www.mql5.com/en/articles/14806&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5071625979081730943)

MetaTrader 5 / Examples


### Preface

This article continues the series on how the MQL5 wizard is ideal for rapid testing and prototyping ideas for traders. For a lot of people developing expert advisers and trade systems, the need to keep learning and be abreast with trends in not just machine learning but trade & risk management in general is important. We therefore consider within these series how the MQL5 IDE is useful in this regard by not only saving time but also minimizing coding errors.

I had intended to look at [neural architecture search](https://en.wikipedia.org/wiki/Neural_architecture_search "https://en.wikipedia.org/wiki/Neural_architecture_search") (NAS) for this piece but realized there are some basic precepts that I have not yet covered which are worth considering, and chief among those being trading multiple securities in a wizard assembled expert advisor. So, we'll take a detour from looking at new trade setups and consider some of the basics for this article. NAS will be considered in the next article.

### Introduction

Multi-currency trading unlike trading a single currency reduces risk concentration. Because each account has a set leverage level and therefore definite free margin amount, when faced with a situation where you have to trade more than one symbol the amount of free margin to be allocated has to be split among all the currencies available. If these currencies have no correlation or are correlated inversely, then the risk of over reliance on just one of them can greatly be mitigated.

In addition, if a trade system where the opening of positions across multiple currencies is not done in parallel to minimize risk as mentioned but is sequential where each currency pair is considered at a distinct time, then cross-market opportunities can be utilized. What this would involve is opening sequential or follow on positions whose currency pairs correlate inversely to already opened positions if opened positions are drawing down, or doing the opposite if already opened positions have a float. The pairing of currencies as either margin or profit in any pair makes this process particularly interesting when one is selecting a cross-market opportunity, besides considering correlations alone.

The margin profit pairing in each currency pair also presents an advantage of multi-currency trading, which is hedging. Bordering on arbitrage, a practice not only frowned upon by most brokers but also very difficult to implement in practice, hedging with multiple currency pairs based on their margin or profit currencies can be engaged with pending orders especially in situations where a trader could be looking to hold positions over high impact news events or even the weekend.

More than these strategies, though, multi-currency trading allows the trader or expert advisor to view and capitalize on currency specific trends across many of the tradeable pairs in market watch. So for instance if a particular currency is earning favourable currency interest you could buy and hold a number of pairs where it has a superior rate under [uncovered interest arbitrage](https://en.wikipedia.org/wiki/Uncovered_interest_arbitrage "https://en.wikipedia.org/wiki/Uncovered_interest_arbitrage")and earn not just from any price changes but also from the interest. The ability to do an analysis on multiple currency pairs and placing trades that take advantage of these cross-currency setups is only feasible if the expert advisor is built for multi-currency trading, something which the wizard assembled expert advisors we’ve looked at are not capable of doing.

So, for this article, we will look to build templates that modify the classes used in the MQL5 wizard to assemble an expert advisor in broadly 2 scenarios. First, we will look to have a wizard assembled expert that simply analyses and opens parallel positions across multiple currency pairs with the goal of mitigating risk exposure. This approach will probably take the most customization, as we will modify a major file used in the wizard assembly. Finally, in the second approach, we will consider having wizard assembled experts that consider relative currency pair correlations and can open positions in sequence based on independent analysis.

### Template for the MQL5 Wizard (Option-1):

Expert Advisors assembled with the wizard have 3 main anchor classes, with each in a separate file. These are ‘ExpertBase’, ‘ExpertTrade’, and ‘Expert’. Besides these 3 anchor classes there are 3 other ancillary classes namely ‘ExpertSignal’, ‘ExpertMoney’, and ‘ExpertTrailling’ that are respectively inherited from when building an expert’s signal, money management, or trailing classes. In all these classes, it is the ‘ExpertBase’ class that defines the symbol class object that is used in accessing market information of the traded symbol by default (‘m\_symbol’). As mentioned above, by default for all wizard assembled expert advisors only trade one symbol, so the symbol class initialized within ‘ExpertBase’ is meant to handle just one symbol.

To make this multi-symbol, a possible solution could be to convert this class instance into an array. While this is possible, the classes that access this symbol class instance, use many functions to do so and this connectedness is always expecting a none array meaning a lot of unfeasible changes would have to be made to the code. So, this is not a good solution in making the wizard assembled experts’ multi-currency.

A more feasible and similar approach that we’ll consider is making the instance of the umbrella ‘Expert’ class an array whose size matches the number of trade symbols we’d be testing on. An instance of this class is always declared in the ‘\*.mq5’ file of the wizard assembled expert advisor, and simply converting this into an array in essence marks all the major downstream customization we have to do.

```
//+------------------------------------------------------------------+
//| Global expert object                                             |
//+------------------------------------------------------------------+
CExpert ExtExpert[__FOLIO];
CExpertSignal *signals[__FOLIO];
```

Converting the expert class instance into an array will require that we have predefined an array of currency pairs to trade. This we do in the code below:

```
//+------------------------------------------------------------------+
//| 'Commodity' Currency folio                                       |
//+------------------------------------------------------------------+
#define                          __FOLIO 9
string                           __F[__FOLIO]

                                 =

                                 {
                                 "AUDNZD","AUDUSD","AUDCAD","AUDJPY",
                                 "NZDUSD","NZDCAD","NZDJPY",
                                 "USDCAD",
                                 "CADJPY"
                                 };
input string                     __prefix="";
input string                     __suffix="";
```

The above code gets amended to a customized version of the Expert class. Since we are modifying this class, we then have to save an instance of it under a new name to maintain the default settings for typical assemblies. The name we use is ‘ExpertFolio.mqh’ from the original name ‘Expert.mqh’. And besides the name change and modified code in the header, we also need to change the ‘Init’ function’s listing to be more accommodative of symbol’s that do not match the symbol of the chart to which the expert is attached. This we do in the code below:

```
//+------------------------------------------------------------------+
//| Initialization and checking for input parameters                 |
//+------------------------------------------------------------------+
bool CExpert::Init(string symbol,ENUM_TIMEFRAMES period,bool every_tick,ulong magic)
  {
//--- returns false if the EA is initialized on a timeframe different from the current one
   if(period!=::Period())
     {
      PrintFormat(__FUNCTION__+": wrong timeframe (must be: %s)",EnumToString(period));
      return(false);
     }

   if(m_on_timer_process && !EventSetTimer(PeriodSeconds(period)))
      {
      PrintFormat(__FUNCTION__+": cannot set timer at: ",EnumToString(period));
      return(false);
      }
//--- initialize common information
   if(m_symbol==NULL)
     {
      if((m_symbol=new CSymbolInfo)==NULL)
         return(false);
     }
   if(!m_symbol.Name(symbol))
      return(false);

....
....

//--- ok
   return(true);
  }
```

The above changes can be made to a replica of the expert class file which we then go on to rename as indicated above, or we can create a new class that inherits from the expert class in ‘Expert.mqh’ and then add an ‘override’ ‘Init’ function in this new class. This new class and file will what will be referenced in the main expert file. Below is a listing of this:

```
#include "Expert.mqh"
//+------------------------------------------------------------------+
//| Class CExfolio.                                                  |
//| Purpose: Base class expert advisor.                              |
//| Derives from class CExpertBase.                                  |
//+------------------------------------------------------------------+
class CExfolio : public CExpert
  {
protected:


public:

                     CExfolio(void);
                    ~CExfolio(void);

   //--- initialization
   virtual bool      Init(string symbol,ENUM_TIMEFRAMES period,bool every_tick,ulong magic=0) override;

   //...

  };
```

As can be seen from the code above the override command was added at the end of the interface declaration, but because the Init function is not ‘virtual’ by default the ‘Init’ function in the original expert class needs to be modified by having this added. So that line only would look like this:

```
//+------------------------------------------------------------------+
//| Initialization and checking for input parameters                 |
//+------------------------------------------------------------------+
bool CExfolio::Init(string symbol,ENUM_TIMEFRAMES period,bool every_tick,ulong magic)
  {
//--- returns false if the EA is initialized on a symbol/timeframe different from the current one

      bool _init=true;

      if(!CExpert::Init(symbol,period,every_tick,magic))
      {
         _init=false;
         //
         if(symbol!=_Symbol)
         {
            if(Reinit(symbol,period,every_tick,magic)){ _init=true; }
         }
      }

      CExpert::OnTimerProcess(true);

      if(CExpert::m_on_timer_process && !EventSetTimer(PeriodSeconds(_Period)))
      {
         printf(__FUNCTION__+": cannot set timer at: ",EnumToString(period));
         _init=false;
      }

//--- ok
      return(_init);
  }
```

The main changes here ensure initialisation does not fail if the chart symbol does not match the portfolio symbol. This would certainly be a ‘neater’ option since our created class file is considerably smaller meaning we have less duplicity than in the first case however we would need to modify the inbuilt expert file and this means for each terminal update we’d have to add the virtual function specifier to the init function.

Besides these expert class changes, the expert advisor file that is generated by the wizard would also need to have each of the OnInit, OnTick, OnTimer, OnDeInit, and Trade functions modified. What would be added will be a for loop that iterates through all the currencies pre-declared in the customized expert class. Their declaration is accompanied by prefix and suffix string input parameters that ensure their names are well-formed and are in market watch. Rather than explicitly name the symbols and manage for prefixed/ suffixed names the proper approach at accessing available symbols is highlighted [here](https://www.mql5.com/en/book/automation/symbols/symbols_list), but even then you would need to have a predefined list of the symbols that you are interested in, in order to properly filter the often very long list of symbols that available in market watch.

The symbols selected in the custom expert class are what used to be referred to as ‘commodities’ especially in the bygone Cold War era since their momentum was overly influenced with the price of commodities. So, our list uses the currencies AUD, NZD, and CAD as the main commodities. The addition of USD, and JPY is for volatility exposure, but they were not members of the ‘commodities’ group.

The signal class used for the expert advisor in the wizard can be any of the inbuilt classes. This is because for the first implementation, the symbols are executing in parallel to minimize risk exposure. The trade decisions on anyone is not influenced by what is happening to another symbol, so the settings for the signal selected will be applied across all symbols. From testing, this implies that our test runs will have better chances of doing well in forward walks or cross validation since different symbols share the same settings and there is therefore less curve fitting. We use the RSI signal class that requires relatively few input parameters, namely: indicator period and applied price.

### Template for the MQL5 Wizard (Option-2):

For the second implementation of multi-currency trading, we’ll look to modify the wizard assembled expert advisor file by adding to it a ‘Select’ function. This function, by relying on the correlations of the symbols within the trade portfolio, seeks to capitalize on cross-market opportunities by only trading more than one symbol if these opportunities exist. It essentially is a filter from the approach we considered in the first option above.

In order to achieve this firstly we should enable on-timer trading which surprisingly is not on by default. We do this with one line in the ‘OnInit’ function, as indicated below:

```
//
ExtExpert[f].OnTimerProcess(true);
```

With this set we can then focus on the ‘Select’ function which simply features two major sections. The first handles situations where no positions are open, and a correlation confirming signal needs to be selected among all the trade symbols in the portfolio. This signal is confirming because remember we still rely on the signal class RSI signal as the primary for all symbols. So, the confirmation in our case will be got via [auto-correlation](https://en.wikipedia.org/wiki/Autocorrelation "https://en.wikipedia.org/wiki/Autocorrelation")of the close price buffer for each of the symbols. The symbol with the largest positive value from the portfolio will be selected.

We use the largest positive value and not simply the magnitude because we are targeting trending symbols or the symbol with the strongest trend from the portfolio. Once we have this, we determine its trending direction by looking at the close buffer price change between the latest value and the last value. A positive change would be bullish, while a negative change would be bearish. This is processed in the listing below:

```
//+------------------------------------------------------------------+
//| Symbol Selector via Correlation                                  |
//+------------------------------------------------------------------+
bool  Select(double Direction, int &Index, ENUM_POSITION_TYPE &Type)
{  if(PositionsTotal() == 0)
   {  double _max = 0.0;
      int _index = -1;
      Type = INVALID_HANDLE;
      for(int f = 0; f < __FOLIO; f++)
      {  vector _v_0, _v_1;
         _v_0.CopyRates(__F[f], Period(), 8, 0, 30);
         _v_1.CopyRates(__F[f], Period(), 8, 30, 30);
         double _corr = _v_0.CorrCoef(_v_1);
         if(_max < _corr && ((Direction > 0.0 && _v_0[0] > _v_0[29]) || (Direction < 0.0 && _v_0[0] < _v_0[29])))
         {  _max = _corr;
            _index = f;
            if(_v_0[0] > _v_0[29])
            {  Type = POSITION_TYPE_BUY;
            }
            else if(_v_0[0] < _v_0[29])
            {  Type = POSITION_TYPE_SELL;
            }
         }
      }
      Index = _index;
      return(true);
   }
   else if(PositionsTotal() == 1)
   {

//...
//...

   }
   return(false);
}
```

So, the above listing is the ‘default’ portion of the select function, and it handles situations where no position has been opened yet. Once a position has been opened, then we’d proceed to look for cross-market opportunities depending on the performance of the opened position. If the initial opened position is in drawdown, then we sift through the remaining portfolio symbols and try to find one with an effective ‘inverse’ correlation. ‘Inverse’ in the sense that we are only interested in magnitude, and we’ll follow the trend of this symbol as long as its correlation magnitude to the current open symbol is highest when compared to the already opened position. The code that handles this is below:

```
//+------------------------------------------------------------------+
//| Symbol Selector via Correlation                                  |
//+------------------------------------------------------------------+
bool  Select(double Direction, int &Index, ENUM_POSITION_TYPE &Type)
{  if(PositionsTotal() == 0)
   {
//...
//...

   }
   else if(PositionsTotal() == 1)
   {  ulong _ticket = PositionGetTicket(0);
      if(PositionSelectByTicket(_ticket))
      {  double _float = PositionGetDouble(POSITION_PROFIT);
         ENUM_POSITION_TYPE _type = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
         int _index = ArrayBsearch(__F, PositionGetString(POSITION_SYMBOL));
         double _max = 0.0;
         Type = INVALID_HANDLE;
         for(int f = 0; f < __FOLIO; f++)
         {  if(f == _index)
            {  continue;
            }
            else
            {  vector _v_0, _v_1;
               _v_0.CopyRates(__F[_index], Period(), 8, 0, 30);
               _v_1.CopyRates(__F[f], Period(), 8, 0, 30);
               double _corr = fabs(_v_0.CorrCoef(_v_1));
               if(_float < 0.0 && _max < _corr)
               {  _max = _corr;
                  Index = f;
                  if(_v_1[0] > _v_1[29])
                  {  Type = POSITION_TYPE_BUY;
                  }
                  else  if(_v_1[0] < _v_1[29])
                  {  Type = POSITION_TYPE_SELL;
                  }
               }
            }
         }
      }
      return(true);
   }
   return(false);
}
```

In order to use this ‘Select’ function within the environment of a wizard assembled expert advisor, we do have to make some more changes to the expert class. This is why creating a duplicate instance of the expert class and renaming it, as bulky as it seems, would be more pragmatic. The first change we need to make is to make the function ‘Refresh’ accessible publicly. This means moving it from the ‘Protected’ section of the class interface to the ‘Public’ portion. Similarly, the function ‘Process’ should also be given public access. This change would appear as indicated below:

```
//+------------------------------------------------------------------+
//| Class CExpert.                                                   |
//| Purpose: Base class expert advisor.                              |
//| Derives from class CExpertBase.                                  |
//+------------------------------------------------------------------+
class CExpert : public CExpertBase
  {
protected:

   //...
   //...

public:
                     CExpert(void);
                    ~CExpert(void);

   //...

   //--- refreshing
   virtual bool      Refresh(void);
   //--- processing (main method)
   virtual bool      Processing(void);

protected:

   //...
   //...

  };
```

These two key functions need to be moved to public because we have to manually call them within our altered expert advisor. Typically, they get called internally by the ‘OnTick’ or ‘OnTimer’ functions, depending on which of the two is enabled. And when they are called they do a full signal, trailing and money management processing of the symbol they are called on. In our case, we want to process only certain symbols depending on a) their relative correlations, and b) whether we already have a position open. This clearly would necessitate we ‘take command’ of how and when they are called. We would do so in the ‘OnTimer’ function as indicated below:

```
//+------------------------------------------------------------------+
//| "Timer" event handler function                                   |
//+------------------------------------------------------------------+
void OnTimer()
{  for(int f = 0; f < __FOLIO; f++)
   {  ExtExpert[f].Refresh();
   }
   for(int f = 0; f < __FOLIO; f++)
   {  int _index = -1;
      ENUM_POSITION_TYPE _type = INVALID_HANDLE;
      ExtExpert[f].Magic(f);
      if(Select(signals[f].Direction(), _index, _type))
      {  ExtExpert[f].OnTimer();
         ExtExpert[f].Processing();
      }
   }
}
```

In selecting the symbol to trade, we track it by assigning its index within the source portfolio array as its magic number. This means on each for loop within all the default processing functions of ‘OnTick’, ‘OnTrade’, and ‘OnTimer’ we’d need to update the magic number used by the trade class before processing. In addition, because custom symbols were used for this testing, for data quality reasons, the name of the symbols in the portfolio array needed to be included with their suffix (or prefix if applicable). For some reason, strategy tester cannot synchronize custom symbols even though all the data is present on the test computer if you’re adding the prefix and suffix when initializing.

Also, we want to trade ‘OnTimer’ not on the default ‘OnTick’ and this should in theory be set by easily adjusting the ‘OnProcessTimer’ and ‘OnProcessTick’ parameters, however changing them either results in no trades or trading on tick. This has meant that more invasive changes within the OnTick function and OnTimer function of the expert class were necessary such that the ‘OnProcess’ function was disabled within ‘OnTick’ and the ‘OnProcess’ function that was made public above, is now called independently within the expert advisor’s ‘OnTimer’ function as is shown in the ‘OnTimer’ code above. This is necessary again because of an unknown reason, at the time of this writing, the ‘Process’ function within ‘OnTimer’ in ‘ExpertFolio’ class does not execute. We are able to refresh prices and indicator values across all symbols, but the ‘Process’ function needs to be called independently. And also the timer needs to be declared and killed manually like in ordinary expert advisors.

### Testing and Optimization

We perform test runs on the ‘commodity’ cross AUDJPY on the 4-hour time frame from 2023 to 2024. Our test expert advisors do not use stop loss because in the second option as outlined above, losing positions are managed by cross-market opportunities. We optimize only for RSI period, since it is the signal class indicator, and open & close thresholds, expiration of limit orders and entry gap for the limit orders. The money management used was for fixed lots to the minimum size. The reports and equity curves of both options are presented below:

![r1](https://c.mql5.com/2/76/r1.png)

![c1](https://c.mql5.com/2/76/c1.png)

![r2](https://c.mql5.com/2/76/r2.png)

![c2](https://c.mql5.com/2/76/c2.png)

As can be seen from the reports above, a lot more trades are placed than would be the case with a single currency setup, as one would expect. Interestingly the settings used to perform these multiple trades while managing drawdowns to reasonable limits which points to extended application and potential. Also, as expected the second option placed fewer trades than the first because a) there was no secondary filter for the entry signal and b) cross-market (hedging) rules are used in the second option to minimize drawdowns and manage risk.

Besides cross-market opportunities, uncovered interest arbitrage opportunities can also be explored as mentioned in the intro and in this case economic news data would need to be extracted from the mql5 economic calendar. Strategy testing with these values is still limited, but recently an interesting article on storing economic news data in a database all the while using the MQL5 IDE was posted [here](https://www.mql5.com/en/articles/14324), andit is worth a read if this is an avenue you would like to explore.

### Conclusion

To conclude, we have explored how multi-currency trading can be introduced to expert advisors assembled via the MQL5 wizard. These can be easily coded without the wizard, however the MQL5 wizard not only allows rapid development of experts with less repetition, it also allows more than a single idea to be tested concurrently via a weighting system. As always, new readers can refer to guides [here](https://www.mql5.com/en/articles/171) and [here](https://www.mql5.com/en/articles/275) on how to use the wizard.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/14806.zip "Download all attachments in the single ZIP archive")

[ExpertFolio\_1.mqh](https://www.mql5.com/en/articles/download/14806/expertfolio_1.mqh "Download ExpertFolio_1.mqh")(121.5 KB)

[ExpertFolio\_2.mqh](https://www.mql5.com/en/articles/download/14806/expertfolio_2.mqh "Download ExpertFolio_2.mqh")(121.49 KB)

[opt\_17\_1.mq5](https://www.mql5.com/en/articles/download/14806/opt_17_1.mq5 "Download opt_17_1.mq5")(7.04 KB)

[opt\_17\_2.mq5](https://www.mql5.com/en/articles/download/14806/opt_17_2.mq5 "Download opt_17_2.mq5")(10.18 KB)

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

**[Go to discussion](https://www.mql5.com/en/forum/466413)**

![Developing an MQL5 RL agent with RestAPI integration (Part 2): MQL5 functions for HTTP interaction with the tic-tac-toe game REST API](https://c.mql5.com/2/61/DALLvE_2023-11-26_00.52.08_-_A_digital_artwork_illustrating_the_integration_of_MQL55_Pythono_and_Fas.png)[Developing an MQL5 RL agent with RestAPI integration (Part 2): MQL5 functions for HTTP interaction with the tic-tac-toe game REST API](https://www.mql5.com/en/articles/13714)

In this article we will talk about how MQL5 can interact with Python and FastAPI, using HTTP calls in MQL5 to interact with the tic-tac-toe game in Python. The article discusses the creation of an API using FastAPI for this integration and provides a test script in MQL5, highlighting the versatility of MQL5, the simplicity of Python, and the effectiveness of FastAPI in connecting different technologies to create innovative solutions.

![The Group Method of Data Handling: Implementing the Combinatorial Algorithm in MQL5](https://c.mql5.com/2/76/The_Group_Method_of_Data_Handling___LOGO.png)[The Group Method of Data Handling: Implementing the Combinatorial Algorithm in MQL5](https://www.mql5.com/en/articles/14804)

In this article we continue our exploration of the Group Method of Data Handling family of algorithms, with the implementation of the Combinatorial Algorithm along with its refined incarnation, the Combinatorial Selective Algorithm in MQL5.

![Custom Indicators (Part 1): A Step-by-Step Introductory Guide to Developing Simple Custom Indicators in MQL5](https://c.mql5.com/2/76/Indicators_Article_Thumbnail_Artwork.png)[Custom Indicators (Part 1): A Step-by-Step Introductory Guide to Developing Simple Custom Indicators in MQL5](https://www.mql5.com/en/articles/14481)

Learn how to create custom indicators using MQL5. This introductory article will guide you through the fundamentals of building simple custom indicators and demonstrate a hands-on approach to coding different custom indicators for any MQL5 programmer new to this interesting topic.

![Developing a Replay System (Part 37): Paving the Path (I)](https://c.mql5.com/2/61/Desenvolvendo_um_sistema_de_Replay__Parte_37__LOGO.png)[Developing a Replay System (Part 37): Paving the Path (I)](https://www.mql5.com/en/articles/11585)

In this article, we will finally begin to do what we wanted to do much earlier. However, due to the lack of "solid ground", I did not feel confident to present this part publicly. Now I have the basis to do this. I suggest that you focus as much as possible on understanding the content of this article. I mean not simply reading it. I want to emphasize that if you do not understand this article, you can completely give up hope of understanding the content of the following ones.

[![](https://www.mql5.com/ff/si/dwquj7nmuxsb297n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F994%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dhow.use.vps%26utm_content%3Drent.vps%26utm_campaign%3D0622.MQL5.com.Internal&a=enhudadyvnrfwcvutcjazdvrxjyrzhyf&s=8f8a773cbff7e7ca26346dfb885f4f329a8b1f2c99472f858f32c0b06b662998&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=gwwrbibxhzsnyhesfsaowprzfttkswvx&ssn=1769191807424142619&ssn_dr=0&ssn_sr=0&fv_date=1769191807&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F14806&back_ref=https%3A%2F%2Fwww.google.com%2F&title=MQL5%20Wizard%20Techniques%20you%20should%20know%20(Part%2017)%3A%20Multicurrency%20Trading%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176919180753146948&fz_uniq=5071625979081730943&sv=2552)

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