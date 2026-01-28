---
title: News Trading Made Easy (Part 3): Performing Trades
url: https://www.mql5.com/en/articles/15359
categories: Trading, Integration
relevance_score: -5
scraped_at: 2026-01-24T14:17:51.607523
---

[![](https://www.mql5.com/ff/sh/9nb0c8df2rmwfn89z2/01.png) MetaTrader VPS vs regular cloud hosting services8 reasons why our solution is the best option for automated tradingRead](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/450486&a=dgmsfszgoedimaicrqqmagvqzpwuxkur&s=c59e3617ccf44fd54d4c50a03b44fd689ff7507b8fe4990c83772cc5419e627d&uid=&ref=https://www.mql5.com/en/articles/15359&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5083470957718281308)

MetaTrader 5 / Trading


### Introduction

Previously, we created an Expert Advisor to store economic data in our news calendar database. We also developed many classes to build a foundation for our expert to perform adequately. In this article, we will expand on these classes in our project to finally get to our goal of trading from economic data. Our next goal will be profitability which we will address in the next upcoming articles. For this article, we will add a new view to our database to display all unique events from the [MQL5 economic calendar](https://www.mql5.com/en/economic-calendar) to provide information about the different events. We will also add new inputs to the expert to filter economic data when trading to provide flexibility. You can check out the [previous article](https://www.mql5.com/en/articles/14912 "News Trading Made Easy (Part 2): Risk Management") in the News Trading Made Easy series where we created a risk management class to manage risk for trading and more useful information if you haven't already.

**What can you expect?**

Improved graphics that are concise, modern, and responsive for the current chart. The image below is a representation of these graphics in Light mode.

![AUDUSD in Light Mode(illustration)](https://c.mql5.com/2/85/AUDUSDH1l2p.png)

Sections: 1,2,3,4,5,6,7,8 and 9 will automatically be shown every time the expert is on a chart.

Sections: 10, (11 & 12 & 13& 14 & 15 are a group) and 16 are optional and will update every new 1-minute candle (this is to improve performance when back-testing).

Section

- 10: Displays the terminal Date and time. The time text will be shown in red when a news event occurs while the chart is in light mode.
- 11: Displays the current/next news event date and time. The text will be shown in red when the date and time are equal to the terminal time.
- 12: Displays the name of the news event. The text color will change depending on the event's Importance, e.g. High Importance is shown in red.
- 13: Displays the country name of the news event. The text color will change depending on the event's Importance and chart color mode e.g. Light mode.
- 14: Displays the currency name of the news event. The text color will vary.
- 15: Displays the news event's importance. The text color will vary.
- 16: Displays the spread for the current symbol and the rating which is calculated with 2 weeks' worth of 1-minute candle spread data and categorized into groups of excellent, good, normal, bad, and terrible with different colors for each category with variation for dark mode and light mode.

The image below is a representation of how dark mode is implemented.

Section

- 17: Displays the event time of all events which will or have occurred on the current terminal day.

![AUDUSD in Dark Mode(Illustration)](https://c.mql5.com/2/85/AUDUSDH1_DarkModey2d.png)

**DISPLAY Inputs**

- CHART COLOUR MODE: This option's purpose is to change between Dark or Light Mode.
- DISPLAY NEWS INFO: This option's purpose is to show or not show sections (11 & 12 & 13& 14 & 15) on the chart.
- DISPLAY EVENT OBJ: This option's purpose is to show or not show section 17 on the chart.
- DISPLAY SPREAD RATING: This option's purpose is to show or not show section 16 on the chart.
- DISPLAY DATE: This option's purpose is to show or not show section 10 on the chart.

![Display Input Options](https://c.mql5.com/2/85/Display_Options.png)

**DST SCHEDULE Inputs**

- SELECT DST OPTION: This option's purpose is to allow the user/trader to select there custom DST schedule or allow the expert to automatically select the recommended DST schedule to correctly configure the event times when back-testing in the strategy tester.
- SELECT CUSTOM DST: This option's purpose is to allow the user/trader to manually configure the DST schedule.

![DST SCHEDULE Input Options](https://c.mql5.com/2/85/DST_Options__1.png)

**RISK MANAGEMENT Inputs**

- SELECT RISK OPTION: This option's purpose is to allow the user/trader to select different Risk Management profiles e.g. MINIMUM LOTSIZE, MAXIMUM LOTSIZE etc.
- RISK FLOOR: This option's purpose is to set a minimum risk for all risk profiles. Example if there is not enough money for a lot-size of 1 lot, but there is enough money for the minimum lot-size of 0.01 lot, then 0.01 will be used to open the trade instead of not opening any trades because there wasn't enough money. This is just a safety net for if the risk profiles were not configured appropriately.
- MAX-RISK: This option's purpose is to open a trade with the percentage of free-margin in the account if there wasn't enough money for a normal trade to be opened. This option is only operational when RISK FLOOR is set to MAX-RISK.
- RISK CEILING: This option's purpose is to set a lot-size cap/limit when the account is large enough to open the max lot for a specific symbol. The cap/limit varies from MAX LOTSIZE which means the maximum lot-size possible is the one set by the specific symbol, whereas MAX LOTSIZE(x2) will open two trades with the maximum lot-size depending on if the volume limit allows for this.
- PERCENTAGE OF \[BALANCE \| FREE-MARGIN\]: This option's purpose is to risk a certain percentage of the available Balance or Free-margin.
- AMOUNT PER \[BALANCE \| FREE-MARGIN\]: This option's purpose is to risk a certain value amount of the Balance or Free-margin e.g. if \[BALANCE \| FREE-MARGIN\] is set to 1000 and EACH AMOUNT is set to 10, this means that for every 1000 in Balance or Free-margin currency value risk 10 currency value for each trade. So if your balance/free-margin is 1000 USD, then risk 10 USD for every trade.
- LOTSIZE PER \[BALANCE \| FREE-MARGIN\]: This option's purpose is to risk a certain lot-size for a Balance or Free-margin value e.g. if \[BALANCE \| FREE-MARGIN\] is set to 1000 and EACH LOTS(VOLUME) is set to 0.1, this means that for every 1000 in Balance or Free-margin currency value risk 0.1 in lot-size for each trade. So if your balance/free-margin is 1000 USD, then risk 0.1 for every trade.
- CUSTOM LOTSIZE: This option's purpose is to risk a predetermined lot-size for every trade opened.
- PERCENTAGE OF MAX-RISK: This option's purpose is to risk a percentage of the Maximum risk volume for a symbol with the account's available Free-margin e.g. if the Maximum risk volume for AUDUSD with an account's Free-margin at 10,000 USD is 100 Lots then if we set PERCENTAGE OF MAX-RISK to 25% then the lot-size used will be 25% of 100 Lots which is 25 Lots.

![RISK MANAGEMENT Input Options](https://c.mql5.com/2/86/Risk_Management_Options.png)

**NEWS SETTINGS Inputs**

News Settings consists of varies input options namely:

1. CALENDAR IMPORTANCE
2. EVENT FREQUENCY
3. EVENT SECTOR
4. EVENT TYPE
5. EVENT CURRENCY

These options are shown in the image below.

![NEWS SETTINGS Input Options](https://c.mql5.com/2/86/News_Options_Overview.png)

- CALENDAR IMPORTANCE: This option's purpose is to filter the news data to a specified news Importance.

![CALENDAR IMPORTANCE Input parameter](https://c.mql5.com/2/86/News_Options_Part1.png)

- EVENT FREQUENCY: This option's purpose is to filter the news data based of its frequency of occurrence.

![EVENT FREQUENCY Input parameter](https://c.mql5.com/2/86/News_Options_Part2.png)

- EVENT SECTOR: This option's purpose is to filter the news data based on the sector.

![EVENT SECTOR Input parameter](https://c.mql5.com/2/86/News_Options_Part3.png)

- EVENT TYPE: This option's purpose is to filter the news data of its type, e.g. EVENT is typically used for Speeches and meetings, whereas INDICATOR is for interest rates, Employment data etc. and HOLIDAY is for News years and various other holidays.

![EVENT TYPE Input parameter](https://c.mql5.com/2/86/News_Options_Part4.png)

- EVENT CURRENCY: This option's purpose is to filter the news data based of the selected currency options available. SYMBOL CURRENCIES will consider all the currencies from SYMBOL MARGIN, SYMBOL BASE and SYMBOL PROFIT.

![EVENT CURRENCY Input parameter](https://c.mql5.com/2/86/News_Options_Part5.png)

**TRADE SETTINGS Inputs**

- STOPLOSS\[0=NONE\]: This option's purpose is to set a fixed stoploss for all trades. When the stoploss is set to zero all trades will not have a stoploss value.
- TAKEPROFIT\[0=NONE\]: This option's purpose is to set a fixed take-profit for all trades. When the take-profit is set to zero, all trades will not have a take-profit value.
- PRE-ENTRY SEC: This option's purpose is to allow the user/trader to configure the number of seconds before a trade is open before the event time. So if the PRE-ENTRY SEC is set to 5 this means that 5 seconds before the event occurs is the timespan for trades to be opened before the event time. Ex. if the event time is 15:00pm then trades will be allowed from 5 seconds before the event at 14:59:45-14:59:59.
- TRADING DAY OF WEEK: This option's purpose is to filter any business day of the week e.g. Monday, Tuesday etc.

![TRADE SETTINGS Input Options](https://c.mql5.com/2/86/Trade_Options.png)

We will now dive into the code that will make our expert functional.

### Symbol Properties Class

**Added changes made from Part 2:**

- Declaration of enumeration for spread rating

```
//Enumeration for Spread rating
enum SpreadRating
  {
   SpreadRating_Terrible,//Terrible
   SpreadRating_Bad,//Bad
   SpreadRating_Normal,//Normal
   SpreadRating_Good,//Good
   SpreadRating_Excellent//Excellent
  };
```

- Declaration of Boolean variable to configure Chart color mode

```
bool isLightMode;//Variable to configure Chart color mode
```

- Declaration of Boolean function to retrieve spread float

```
bool              SpreadFloat(string SYMBOL=NULL);//Retrieve Spread Float
```

```
//+------------------------------------------------------------------+
//|Retrieve Spread Float                                             |
//+------------------------------------------------------------------+
bool CSymbolProperties::SpreadFloat(string SYMBOL=NULL)
  {
   if(SetSymbolName(SYMBOL))//Set Symbol
     {
      return CSymbol.SpreadFloat();
     }
   Print("Unable to retrieve Symbol's Spread Float");
   return false;//Retrieve false when failed.
  }
```

- Declaration of Spread rating function to retrieve spread rating

```
SpreadRating      SpreadValue(string SYMBOL=NULL);//Retrieve Spread Rating
```

This function has to return an enumeration value from SpreadRating.

```
//+------------------------------------------------------------------+
//|Retrieve Spread Rating                                            |
//+------------------------------------------------------------------+
SpreadRating CSymbolProperties::SpreadValue(string SYMBOL=NULL)
  {
   if(SetSymbolName(SYMBOL))//Set Symbol
     {
      if(SpreadFloat(SYMBOL))//Check if Symbol has a floating Spread
        {

         //Declarations
         vector Spreads;
         int SpreadArray[],SpreadAvg=0,SpreadMax=0,SpreadMin=0,
                           SpreadUpper=0,SpreadLower=0,SpreadAvgUpper=0,
                           SpreadAvgLower=0,SpreadMidUpper=0,SpreadMidLower=0;

         //Get Spread data from CopySpread built-in function for 2 weeks using M1 timeframe.
         if(CopySpread(GetSymbolName(),PERIOD_M1,iTime(GetSymbolName(),PERIOD_W1,2),
                       iTime(GetSymbolName(),PERIOD_M1,0),SpreadArray)==-1)
           {
            Print("Error trying to retrieve spread values");
            return SpreadRating_Normal;//Retrieve default value when failed.
           }
         else
           {
            Spreads.Assign(SpreadArray);//Assign spread array into Spreads vector variable

            SpreadMax = int(Spreads.Max());//Assign max spread
            SpreadMin = int(Spreads.Min());//Assign min spread
            SpreadAvg = int(Spreads.Median());//Assign average spread

            //Divide Spread into sectors based of different averages.
            SpreadMidUpper = int((SpreadAvg+SpreadMax)/2);
            SpreadMidLower = int((SpreadAvg+SpreadMin)/2);
            SpreadAvgUpper = int((SpreadAvg+SpreadMidUpper)/2);
            SpreadAvgLower = int((SpreadAvg+SpreadMidLower)/2);
            SpreadUpper = int((SpreadMidUpper+SpreadMax)/2);
            SpreadLower = int((SpreadMidLower+SpreadMin)/2);

            int Spread = Spread(SYMBOL);//Assign Symbol's Spread

            if(Spread<SpreadLower||Spread==SpreadMin)//Excellent
              {
               return SpreadRating_Excellent;
              }
            else
               if(Spread>=SpreadLower&&Spread<SpreadAvgLower)//Good
                 {
                  return SpreadRating_Good;
                 }
               else
                  if(Spread>=SpreadAvgLower&&Spread<=SpreadAvgUpper)//Normal
                    {
                     return SpreadRating_Normal;
                    }
                  else
                     if(Spread>SpreadAvgUpper&&Spread<=SpreadUpper)//Bad
                       {
                        return SpreadRating_Bad;
                       }
                     else//Terrible
                       {
                        return SpreadRating_Terrible;
                       }
           }
        }
      else
        {
         return SpreadRating_Normal;//Retrieve default value when spread is fixed.
        }
     }
   Print("Unable to retrieve Symbol's Spread Rating");
   return SpreadRating_Normal;//Retrieve default value when failed.
  }
```

We first set the symbol, then we check if the symbol has a floating spread before we perform simple calculations to rate the spread based on its averages. If the setting of the symbol failed or the symbol doesn't have a floating spread, we return SpreadRating\_Normal as a default value.

```
   if(SetSymbolName(SYMBOL))//Set Symbol
     {
      if(SpreadFloat(SYMBOL))//Check if Symbol has a floating Spread
        {
```

If we successfully set the symbol and the symbol has a floating spread, then we will declare a Spreads vector variable, and int variables to store spread values. After declaring our variables, we will use the CopySpread function to store the spread values into our SpreadArray variable from the 1-minute candle starting from 2 weeks back to the current 1-minute candle time. If the CopySpread function fails for some reason, we will return SpreadRating\_Normal as a default value.

```
 //Declarations
         vector Spreads;
         int SpreadArray[],SpreadAvg=0,SpreadMax=0,SpreadMin=0,
                           SpreadUpper=0,SpreadLower=0,SpreadAvgUpper=0,
                           SpreadAvgLower=0,SpreadMidUpper=0,SpreadMidLower=0;

         //Get Spread data from CopySpread built-in function for 2 weeks using M1 timeframe.
         if(CopySpread(GetSymbolName(),PERIOD_M1,iTime(GetSymbolName(),PERIOD_W1,2),
                       iTime(GetSymbolName(),PERIOD_M1,0),SpreadArray)==-1)
           {
            Print("Error trying to retrieve spread values");
            return SpreadRating_Normal;//Retrieve default value when failed.
           }
```

Once CopySpread is successful, we will assign the Spreads vector the integer values from our SpreadArray. Then we need to get basic information from these array values, such as the maximum spread throughout the 2-week period as well as the minimum spread and the average spread, we will store these values into SpreadMax, SpreadMin and SpreadAvg accordingly. We now want to get different averages from these three previous values.

For SpreadMidUpper variable we want the average between the SpreadAvg and SpreadMax, for SpreadMidLower variable we want the average between SpreadAvg and SpreadMin, for SpreadAvgUpper variable we want the average between SpreadAvg and SpreadMidUpper, for SpreadAvgLower variable we want the average between SpreadAvg and SpreadMidLower, for SpreadUpper variable we want the average between SpreadMidUpper and SpreadMax, for SpreadLower variable we want the average between SpreadMidLower and SpreadMin. We will also need the current symobl's spread to compare to classify the spread.

```
Spreads.Assign(SpreadArray);//Assign spread array into Spreads vector variable

            SpreadMax = int(Spreads.Max());//Assign max spread
            SpreadMin = int(Spreads.Min());//Assign min spread
            SpreadAvg = int(Spreads.Median());//Assign average spread

            //Divide Spread into sectors based of different averages.
            SpreadMidUpper = int((SpreadAvg+SpreadMax)/2);
            SpreadMidLower = int((SpreadAvg+SpreadMin)/2);
            SpreadAvgUpper = int((SpreadAvg+SpreadMidUpper)/2);
            SpreadAvgLower = int((SpreadAvg+SpreadMidLower)/2);
            SpreadUpper = int((SpreadMidUpper+SpreadMax)/2);
            SpreadLower = int((SpreadMidLower+SpreadMin)/2);

            int Spread = Spread(SYMBOL);//Assign Symbol's Spread
```

> ![Spread Values](https://c.mql5.com/2/86/SpreadRatingIllustration.png)

We have 5 spread classifications namely:

- Excellent: When the current spread is less than SpreadLower variable or equal to SpreadMin.
- Good: When the current spread is more than or equal to SpreadLower variable and the current spread is less than SpreadAvgLower variable.
- Normal: When the current spread is less than or equal to SpreadAvgLower variable and the current spread is less than or equal to SpreadAvgUpper variable.
- Bad: When the current spread is more than SpreadAvgUpper and the current spread is less than or equal to SpreadUpper variable.
- Terrible: When the current spread is more than SpreadUpper variable

```
            if(Spread<SpreadLower||Spread==SpreadMin)//Excellent
              {
               return SpreadRating_Excellent;
              }
            else
               if(Spread>=SpreadLower&&Spread<SpreadAvgLower)//Good
                 {
                  return SpreadRating_Good;
                 }
               else
                  if(Spread>=SpreadAvgLower&&Spread<=SpreadAvgUpper)//Normal
                    {
                     return SpreadRating_Normal;
                    }
                  else
                     if(Spread>SpreadAvgUpper&&Spread<=SpreadUpper)//Bad
                       {
                        return SpreadRating_Bad;
                       }
                     else//Terrible
                       {
                        return SpreadRating_Terrible;
                       }
```

- Declaration of function to retrieve spread color based of its rating

```
color             SpreadColor(string SYMBOL=NULL);//Retrieve Spread Color
```

In order to get the spread color for each spread enumeration value, we will consider using a switch statement as the enumeration values are constant. Colors for each rating are as shown:

- Excellent: if in light mode then clrBlue

> > ![Excellent Light Mode](https://c.mql5.com/2/86/SpreadExcellent_lightMode.png)
> >
> > else clrLightCyan
> >
> > ![Excellent Dark Mode](https://c.mql5.com/2/86/SpreadExcellent_DarkMode.png)

- Good: if in light mode then clrCornflowerBlue

> > ![Good Light Mode](https://c.mql5.com/2/86/SpreadGood_lightMode.png)
> >
> >  else clrLightGreen
> >
> > ![Good Dark Mode](https://c.mql5.com/2/86/SpreadGood_DarkMode.png)

- Normal: if in light mode then clrBlack

> > ![Normal Light Mode](https://c.mql5.com/2/86/SpreadNormal_lightMode.png)
> >
> >   else clrWheat
> >
> > ![Normal Dark Mode](https://c.mql5.com/2/86/SpreadNormal_DarkMode.png)

- Bad: clrOrange
- Terrible: clrRed
- Default:  if in light mode then clrBlack, else clrWheat

```
//+------------------------------------------------------------------+
//|Retrieve Spread Color                                             |
//+------------------------------------------------------------------+
color CSymbolProperties::SpreadColor(string SYMBOL=NULL)
  {
   switch(SpreadValue(SYMBOL))//Get Spread Rating value
     {
      case SpreadRating_Excellent://Excellent Spread
         return (isLightMode)?clrBlue:clrLightCyan;
         break;
      case SpreadRating_Good://Good Spread
         return (isLightMode)?clrCornflowerBlue:clrLightGreen;
         break;
      case SpreadRating_Normal://Normal Spread
         return (isLightMode)?clrBlack:clrWheat;
         break;
      case SpreadRating_Bad://Bad Spread
         return clrOrange;
         break;
      case SpreadRating_Terrible://Terrible Spread
         return clrRed;
         break;
      default://failed to be identified
         return (isLightMode)?clrBlack:clrWheat;//Retrieve default color when failed.
         break;
     }
  }
```

- Declaration of string function to retrieve spread's description

```
string            SpreadDesc(string SYMBOL=NULL);//Retrieve Spread Description
```

```
//+------------------------------------------------------------------+
//|Retrieve Spread Description                                       |
//+------------------------------------------------------------------+
string CSymbolProperties::SpreadDesc(string SYMBOL=NULL)
  {
   switch(SpreadValue(SYMBOL))//Get Spread Rating value
     {
      case SpreadRating_Excellent://Excellent Spread
         return "Excellent";
         break;
      case SpreadRating_Good://Good Spread
         return "Good";
         break;
      case SpreadRating_Normal://Normal Spread
         return "Normal";
         break;
      case SpreadRating_Bad://Bad Spread
         return "Bad";
         break;
      case SpreadRating_Terrible://Terrible Spread
         return "Terrible";
         break;
      default://failed to be identified
         return "Unknown";//Retrieve default value when failed.
         break;
     }
  }
```

- Declaration of string function to retrieve symbol's description

```
string            Description(string SYMBOL=NULL);//Retrieve Symbol's Description
```

```
//+------------------------------------------------------------------+
//|Retrieve Symbol's Description                                     |
//+------------------------------------------------------------------+
string CSymbolProperties::Description(string SYMBOL=NULL)
  {
   if(SetSymbolName(SYMBOL))//Set Symbol
     {
      return CSymbol.Description();
     }
   Print("Unable to retrieve Symbol's Description");
   return "";//Retrieve an empty string when failed.
  }
```

![Symbol Description](https://c.mql5.com/2/86/Symbol_Description.png)

### Chart Properties Class

This class has been restructured from Part 2. Chart properties class will now inherit from the chart class from MQL5 include classes. ChartProp structure will store all the chart properties that we will introduce changes to. Our public ChartRefresh function will call our ChartGet funtion that will initialize the chart properties, and then we will call the function ChartSet that will configure the chart with our chart property values from ChartGet.

```
#include "SymbolProperties.mqh"
#include <Charts/Chart.mqh>
CSymbolProperties CSymbol;//Symbol Properties object
//+------------------------------------------------------------------+
//|ChartProperties class                                             |
//+------------------------------------------------------------------+
class CChartProperties : public CChart
  {
private:
//Structure for chart properties
   struct ChartProp
     {
      ENUM_CHART_MODE mode;//Chart Mode
      color          clrBackground;//Chart Background Color
      color          clrForeground;//Chart Foreground Color
      color          clrLineLast;//Chart Line Color
      color          clrCandleBear;//Chart Bear Candle Color
      color          clrBarDown;//Chart Down Candle Color
      color          clrCandleBull;//Chart Bull Candle Color
      color          clrBarUp;//Chart Up Candle Color
      color          clrLineAsk;//Chart Ask Color
      color          clrLineBid;//Chart Bid Color
      color          clrChartLine;//Chart Line Color
      color          clrStopLevels;//Chart Stop Level Color
      color          clrVolumes;//Chart Volumes Color
      bool           Foreground;//Chart Foreground Visibility
      bool           ShowLineAsk;//Chart Ask Line Visibility
      bool           ShowLineBid;//Chart Bid Line Visibility
      bool           ShowPeriodSep;//Chart Period Separator Visibility
      bool           ShowOHLC;//Chart Open-High-Low-Close Visibility
      bool           ShowGrid;//Chart Grid Visibility
      ENUM_CHART_VOLUME_MODE ShowVolumes;//Chart Volumes Visibility
      bool           AutoScroll;//Chart Auto Scroll Option
      bool           Shift;//Chart Shift Option
      double         ShiftSize;//Chart Shift Size
      bool           ShowObjectDescr;//Chart Object Descriptions
      ulong          CHART_SHOW_TRADE_LEVELS;//Chart Trade Levels Visibility
      ulong          CHART_SHOW_ONE_CLICK;//Chart One Click Trading Visibility
      ulong          CHART_SHOW_TICKER;//Chart Ticker Visibility
      ulong          CHART_DRAG_TRADE_LEVELS;//Chart Drag Trade levels
      ENUM_CHART_POSITION Navigate;//Chart Navigate
     };
   ChartProp         DefaultChart,MyChart;//Used to store chart properties
   void              ChartSet(ChartProp &Prop);//Apply Chart format
   void              ChartGet();//Assign Chart property values
public:
                     CChartProperties();//Constructor
                    ~CChartProperties(void);//Destructor
                     //Configure the chart
   void              ChartRefresh() {ChartGet();ChartSet(MyChart);}
   string            GetChartPeriodName();//Retrieve Period name
  };
```

In the constructor we assign the inherit variable m\_chart\_id the current chart ID.

```
//+------------------------------------------------------------------+
//|Constructor                                                       |
//+------------------------------------------------------------------+
CChartProperties::CChartProperties()
  {
   m_chart_id=ChartID();//Set chart id
   ChartGet();//Get chart values
   ChartSet(MyChart);//customize chart
  }
```

For the function ChartGet we assign values to our variables DefaultChart and MyChart, where DefaultChart will store the current chart properties before we modify the chart and where MyChart will store our custom values.

```
//+------------------------------------------------------------------+
//|Assign Chart property values                                      |
//+------------------------------------------------------------------+
void CChartProperties::ChartGet()
  {
   DefaultChart.mode = Mode();//assign chart mode
   MyChart.mode = CHART_CANDLES;//assign custom chart mode
   DefaultChart.clrBackground = ColorBackground();//assign Background color
   MyChart.clrBackground = (isLightMode)?clrWhite:clrBlack;//assign custom Background color
   DefaultChart.clrForeground = ColorForeground();//assign foreground color
   MyChart.clrForeground = (isLightMode)?clrBlack:clrWhite;//assign custom foreground color
   DefaultChart.clrLineLast = ColorLineLast();//assign Chart Line Color
   MyChart.clrLineLast = clrWhite;//assign custom Chart Line Color
   DefaultChart.clrCandleBear = ColorCandleBear();//assign Chart Bear Candle Color
   MyChart.clrCandleBear = clrBlack;//assign custom Chart Bear Candle Color
   DefaultChart.clrBarDown = ColorBarDown();//assign Chart Down Candle Color
   MyChart.clrBarDown = (isLightMode)?clrBlack:CSymbol.Background();//assign custom Chart Down Candle Color
   DefaultChart.clrCandleBull = ColorCandleBull();//assign Chart Bull Candle Color
   MyChart.clrCandleBull = CSymbol.Background();//assign custom Chart Bull Candle Color
   DefaultChart.clrBarUp = ColorBarUp();//assign Chart Up Candle Color
   MyChart.clrBarUp = (isLightMode)?clrBlack:CSymbol.Background();//assign custom Chart Up Candle Color
   DefaultChart.clrLineAsk = ColorLineAsk();//assign Chart Ask Color
   MyChart.clrLineAsk = (isLightMode)?clrBlack:clrWhite;//assign custom Chart Ask Color
   DefaultChart.clrLineBid = ColorLineBid();//assign Chart Bid Color
   MyChart.clrLineBid = (isLightMode)?clrBlack:CSymbol.Background();//assign custom Chart Bid Color
   DefaultChart.clrChartLine = ColorChartLine();//assign Chart Line Color
   MyChart.clrChartLine = (isLightMode)?clrBlack:clrWhite;//assign custom Chart Line Color
   DefaultChart.clrStopLevels = ColorStopLevels();//assign Chart Stop Level Color
   MyChart.clrStopLevels = clrRed;//assign custom Chart Stop Level Color
   DefaultChart.clrVolumes = ColorVolumes();//assign Chart Volumes Color
   MyChart.clrVolumes = clrGreen;//assign custom Chart Volumes Color
   DefaultChart.Foreground = Foreground();//assign Chart Foreground Visibility
   MyChart.Foreground = false;//assign custom Chart Foreground Visibility
   DefaultChart.ShowLineAsk = ShowLineAsk();//assign Chart Ask Line Visibility
   MyChart.ShowLineAsk = true;//assign custom Chart Ask Line Visibility
   DefaultChart.ShowLineBid = ShowLineBid();//assign Chart Bid Line Visibility
   MyChart.ShowLineBid = true;//assign custom Chart Bid Line Visibility
   DefaultChart.ShowPeriodSep = ShowPeriodSep();//assign Chart Period Separator Visibility
   MyChart.ShowPeriodSep = true;//assign custom Chart Period Separator Visibility
   DefaultChart.ShowOHLC = ShowOHLC();//assign Chart Open-High-Low-Close Visibility
   MyChart.ShowOHLC = false;//assign custom Chart Open-High-Low-Close Visibility
   DefaultChart.ShowGrid = ShowGrid();//assign Chart Grid Visibility
   MyChart.ShowGrid = false;//assign custom Chart Grid Visibility
   DefaultChart.ShowVolumes = ShowVolumes();//assign Chart Volumes Visibility
   MyChart.ShowVolumes = CHART_VOLUME_HIDE;//assign custom Chart Volumes Visibility
   DefaultChart.AutoScroll = AutoScroll();//assign Chart Auto Scroll Option
   MyChart.AutoScroll = true;//assign custom Chart Auto Scroll Option
   DefaultChart.Shift = Shift();//assign Chart Shift Option
   MyChart.Shift = true;//assign custom Chart Shift Option
   DefaultChart.ShiftSize = ShiftSize();//assign Chart Shift Size
   MyChart.ShiftSize = 15;//assign custom Chart Shift Size
   DefaultChart.ShowObjectDescr = ShowObjectDescr();//assign Chart Object Descriptions
   MyChart.ShowObjectDescr = false;//assign custom Chart Object Descriptions
   DefaultChart.Navigate = CHART_END;//assign Chart Navigate
   MyChart.Navigate = CHART_END;//assign custom Chart Navigate
   //---assign Chart Trade Levels Visibility
   DefaultChart.CHART_SHOW_TRADE_LEVELS = ChartGetInteger(ChartId(),CHART_SHOW_TRADE_LEVELS);
   //---assign custom Chart Trade Levels Visibility
   MyChart.CHART_SHOW_TRADE_LEVELS = ulong(true);
   //---assign Chart One Click Trading Visibility
   DefaultChart.CHART_SHOW_ONE_CLICK = ChartGetInteger(ChartId(),CHART_SHOW_ONE_CLICK);
   //---assign custom Chart One Click Trading Visibility
   MyChart.CHART_SHOW_ONE_CLICK = ulong(false);
   //---assign Chart Ticker Visibility
   DefaultChart.CHART_SHOW_TICKER = ChartGetInteger(ChartId(),CHART_SHOW_TICKER);
   //---assign custom Chart Ticker Visibility
   MyChart.CHART_SHOW_TICKER = ulong(false);
   //---assign Chart Drag Trade levels
   DefaultChart.CHART_DRAG_TRADE_LEVELS = ChartGetInteger(ChartId(),CHART_DRAG_TRADE_LEVELS);
   //---assign custom Chart Drag Trade levels
   MyChart.CHART_DRAG_TRADE_LEVELS = ulong(false);
  }
```

Our ChartSet function will take our ChartProp structure as an argument to configure the current chart.

```
//+------------------------------------------------------------------+
//|Apply Chart format                                                |
//+------------------------------------------------------------------+
void CChartProperties::ChartSet(ChartProp &Prop)
  {
   Mode(Prop.mode);//Set Chart Candle Mode
   ColorBackground(Prop.clrBackground);//Set Chart Background Color
   ColorForeground(Prop.clrForeground);//Set Chart Foreground Color
   ColorLineLast(Prop.clrLineLast);//Set Chart Line Color
   ColorCandleBear(Prop.clrCandleBear);//Set Chart Bear Candle Color
   ColorBarDown(Prop.clrBarDown);//Set Chart Down Candle Color
   ColorCandleBull(Prop.clrCandleBull);//Set Chart Bull Candle Color
   ColorBarUp(Prop.clrBarUp);//Set Chart Up Candle Color
   ColorLineAsk(Prop.clrLineAsk);//Set Chart Ask Color
   ColorLineBid(Prop.clrLineBid);//Set Chart Bid Color
   ColorChartLine(Prop.clrChartLine);//Set Chart Line Color
   ColorStopLevels(Prop.clrStopLevels);//Set Chart Stop Level Color
   ColorVolumes(Prop.clrVolumes);//Set Chart Volumes Color
   Foreground(Prop.Foreground);//Set if Chart is in Foreground Visibility
   ShowLineAsk(Prop.ShowLineAsk);//Set Chart Ask Line Visibility
   ShowLineBid(Prop.ShowLineBid);//Set Chart Bid Line Visibility
   ShowPeriodSep(Prop.ShowPeriodSep);//Set Chart Period Separator Visibility
   ShowOHLC(Prop.ShowOHLC);//Set Chart Open-High-Low-Close Visibility
   ShowGrid(Prop.ShowGrid);//Set Chart Grid Visibility
   ShowVolumes(Prop.ShowVolumes);//Set Chart Volumes Visibility
   AutoScroll(Prop.AutoScroll);//Set Chart Auto Scroll Option
   Shift(Prop.Shift);//Set Chart Shift Option
   ShiftSize(Prop.ShiftSize);//Set Chart Shift Size Value
   ShowObjectDescr(Prop.ShowObjectDescr);//Set Chart Show Object Descriptions
   ChartSetInteger(ChartId(),CHART_SHOW_TRADE_LEVELS,Prop.CHART_SHOW_TRADE_LEVELS);//Set Chart Trade Levels Visibility
   ChartSetInteger(ChartId(),CHART_SHOW_ONE_CLICK,Prop.CHART_SHOW_ONE_CLICK);//Set Chart One Click Trading Visibility
   ChartSetInteger(ChartId(),CHART_SHOW_TICKER,Prop.CHART_SHOW_TICKER);//Set Chart Ticker Visibility
   ChartSetInteger(ChartId(),CHART_DRAG_TRADE_LEVELS,Prop.CHART_DRAG_TRADE_LEVELS);//Set Chart Drag Trade levels
   Navigate(Prop.Navigate);//Set Chart Navigate
  }
```

As for the function GetChartPeriodName we will retrieve the chart period name for the current chart using a switch statement.

```
//+------------------------------------------------------------------+
//|Retrieve Period name                                              |
//+------------------------------------------------------------------+
string CChartProperties::GetChartPeriodName()
  {
   switch(ChartPeriod(ChartId()))//Get chart Period with chart id
     {
      case PERIOD_M1:
         return("M1");
      case PERIOD_M2:
         return("M2");
      case PERIOD_M3:
         return("M3");
      case PERIOD_M4:
         return("M4");
      case PERIOD_M5:
         return("M5");
      case PERIOD_M6:
         return("M6");
      case PERIOD_M10:
         return("M10");
      case PERIOD_M12:
         return("M12");
      case PERIOD_M15:
         return("M15");
      case PERIOD_M20:
         return("M20");
      case PERIOD_M30:
         return("M30");
      case PERIOD_H1:
         return("H1");
      case PERIOD_H2:
         return("H2");
      case PERIOD_H3:
         return("H3");
      case PERIOD_H4:
         return("H4");
      case PERIOD_H6:
         return("H6");
      case PERIOD_H8:
         return("H8");
      case PERIOD_H12:
         return("H12");
      case PERIOD_D1:
         return("Daily");
      case PERIOD_W1:
         return("Weekly");
      case PERIOD_MN1:
         return("Monthly");
     }
   return("unknown period");
  }
```

Our destructor will restore the previous chart configuration before we made any changes to the chart.

```
//+------------------------------------------------------------------+
//|Destructor                                                        |
//+------------------------------------------------------------------+
CChartProperties::~CChartProperties()
  {
   ChartSet(DefaultChart);//restore chart default configuration
   m_chart_id=-1;//reset chart id
  }
```

### Object Properties Class

In this class a few changes were made to all for custom object text color. In part 2 you could only use one object text color for all text objects. Our solution is to declare a color variable outside the class called TextObj\_color.

```
#include "ChartProperties.mqh"
color TextObj_color;
//+------------------------------------------------------------------+
//|ObjectProperties class                                            |
//+------------------------------------------------------------------+
class CObjectProperties:public CChartProperties
  {
private:
   //Simple  chart objects structure
   struct ObjStruct
     {
      long           ChartId;
      string         Name;
     } Objects[];//ObjStruct variable array

   //-- Add chart object to Objects array
   void              AddObj(long chart_id,string name)
     {
      ArrayResize(Objects,Objects.Size()+1,Objects.Size()+2);
      Objects[Objects.Size()-1].ChartId=chart_id;
      Objects[Objects.Size()-1].Name=name;
     }

protected:
   void              DeleteObj()
     {
      for(uint i=0;i<Objects.Size();i++)
        {
         ObjectDelete(Objects[i].ChartId,Objects[i].Name);
        }
     }

public:
                     CObjectProperties(void) {}//Class constructor

   //-- Create Rectangle chart object
   void              Square(long chart_ID,string name,int x_coord,int y_coord,int width,int height,ENUM_ANCHOR_POINT Anchor);

   //-- Create text chart object
   void              TextObj(long chartID,string name,string text,int x_coord,int y_coord,
                             ENUM_BASE_CORNER Corner=CORNER_LEFT_UPPER,int fontsize=10);

   //-- Create Event object
   void               EventObj(long chartID,string name,string description,datetime eventdate);

   //-- Class destructor removes all chart objects created previously
                    ~CObjectProperties(void)
     {
      DeleteObj();
     }
  };
```

As we can see below the parameters for the Textobj function are many, to avoid making the parameters longer we will just use Textobj\_color to change the text object's color.

```
//+------------------------------------------------------------------+
//|Create text chart object                                          |
//+------------------------------------------------------------------+
void CObjectProperties::TextObj(long chartID,string name,string text,int x_coord,int y_coord,
                                ENUM_BASE_CORNER Corner=CORNER_LEFT_UPPER,int fontsize=10)
  {
   ObjectDelete(chartID,name);//Delete previous object with the same name and chart id
   if(ObjectCreate(chartID,name,OBJ_LABEL,0,0,0))//Create object label
     {
      AddObj(chartID,name);//Add object to array
      ObjectSetInteger(chartID,name,OBJPROP_XDISTANCE,x_coord);//Set x Distance/coordinate
      ObjectSetInteger(chartID,name,OBJPROP_YDISTANCE,y_coord);//Set y Distance/coordinate
      ObjectSetInteger(chartID,name,OBJPROP_CORNER,Corner);//Set object's corner anchor
      ObjectSetString(chartID,name,OBJPROP_TEXT,text);//Set object's text
      ObjectSetInteger(chartID,name,OBJPROP_COLOR,TextObj_color);//Set object's color
      ObjectSetInteger(chartID,name,OBJPROP_FONTSIZE,fontsize);//Set object's font-size
     }
   else
     {
      Print("Failed to create object: ",name);
     }
  }
```

A small change was made to our function Square to allow for different background colors depending on the chart color mode.

```
//+------------------------------------------------------------------+
//|Create Rectangle chart object                                     |
//+------------------------------------------------------------------+
void CObjectProperties::Square(long chart_ID,string name,int x_coord,int y_coord,int width,int height,ENUM_ANCHOR_POINT Anchor)
  {
   const int              sub_window=0;             // subwindow index
   const int              x=x_coord;                // X coordinate
   const int              y=y_coord;                // Y coordinate
   const color            back_clr=(isLightMode)?clrWhite:clrBlack;// background color
   const ENUM_BORDER_TYPE border=BORDER_SUNKEN;     // border type
   const color            clr=clrRed;               // flat border color (Flat)
   const ENUM_LINE_STYLE  style=STYLE_SOLID;        // flat border style
   const int              line_width=0;             // flat border width
   const bool             back=false;               // in the background
   const bool             selection=false;          // highlight to move
   const bool             hidden=true;              // hidden in the object list

   ObjectDelete(chart_ID,name);//Delete previous object with the same name and chart id
   if(ObjectCreate(chart_ID,name,OBJ_RECTANGLE_LABEL,sub_window,0,0))//create rectangle object label
     {
      AddObj(chart_ID,name);//Add object to array
      ObjectSetInteger(chart_ID,name,OBJPROP_XDISTANCE,x);//Set x Distance/coordinate
      ObjectSetInteger(chart_ID,name,OBJPROP_YDISTANCE,y);//Set y Distance/coordinate
      ObjectSetInteger(chart_ID,name,OBJPROP_XSIZE,width);//Set object's width/x-size
      ObjectSetInteger(chart_ID,name,OBJPROP_YSIZE,height);//Set object's height/y-size
      ObjectSetInteger(chart_ID,name,OBJPROP_BGCOLOR,back_clr);//Set object's background color
      ObjectSetInteger(chart_ID,name,OBJPROP_BORDER_TYPE,border);//Set object's border type
      ObjectSetInteger(chart_ID,name,OBJPROP_ANCHOR,Anchor);//Set objects anchor point
      ObjectSetInteger(chart_ID,name,OBJPROP_COLOR,clr);//Set object's color
      ObjectSetInteger(chart_ID,name,OBJPROP_STYLE,style);//Set object's style
      ObjectSetInteger(chart_ID,name,OBJPROP_WIDTH,line_width);//Set object's flat border width
      ObjectSetInteger(chart_ID,name,OBJPROP_BACK,back);//Set if object is in foreground or not
      ObjectSetInteger(chart_ID,name,OBJPROP_SELECTABLE,selection);//Set if object is selectable/dragable
      ObjectSetInteger(chart_ID,name,OBJPROP_SELECTED,selection);//Set if object is Selected
      ObjectSetInteger(chart_ID,name,OBJPROP_HIDDEN,hidden);//Set if object is hidden in object list
      ChartRedraw(chart_ID);
     }
   else
     {
      Print("Failed to create object: ",name);
     }
  }
```

### CommonVariables Header File

For trading purposes I decided to create a database in memory, this database needs a name and the name will take in consideration the broker's name, the current chart ID, and whether the expert is in the strategy tester or not.

```
#define NEWS_DATABASE_MEMORY           StringFormat("Calendar_%s_%d_%s.sqlite",broker,ChartID(),(MQLInfoInteger(MQL_TESTER)?"TESTER":"REAL"))
```

The enumeration Choice is for personalization and will be used for the expert's input, this will replace the Boolean datatype. The DayOfTheWeek enumeration will be used to select the trading day of the week without Saturday and Sunday. Whereas the Boolean function Answer will convert the Choice enumeration into a Boolean datatype.

```
enum Choice
  {
   Yes,//YES
   No//NO
  };

enum DayOfTheWeek
  {
   Monday,//MONDAY
   Tuesday,//TUESDAY
   Wednesday,//WEDNESDAY
   Thursday,//THURSDAY
   Friday,//FRIDAY
   AllDays//ALL DAYS
  };

//+------------------------------------------------------------------+
//|Convert enumeration Choice into a boolean value                   |
//+------------------------------------------------------------------+
bool Answer(Choice choose)
  {
   return (choose==Yes)?true:false;
  }
```

### Time Variables Class

This class's purpose is to store candlestick's time data, this data will be used to check if a new candle has formed.

```
//+------------------------------------------------------------------+
//|TimeVariables class                                               |
//+------------------------------------------------------------------+
class CTimeVariables
  {
private:
   //---Array to store candlestick times
   datetime          CandleTime[2000];
public:
                     CTimeVariables(void);
   //---Set Array index time
   void              SetTime(uint index,datetime time);
   //---Get Array index time
   datetime          GetTime(uint index);
  };
```

In the constructor, we will set a default time for all the indexes within the CandleTime array.

```
//+------------------------------------------------------------------+
//|Constructor                                                       |
//+------------------------------------------------------------------+
CTimeVariables::CTimeVariables()
  {
   for(uint i=0; i<CandleTime.Size(); i++)
     {
      CandleTime[i]=D'1970.01.01';
     }
  }
```

In the function SetTime we have two parameters, one for the array index and the other for datetime. If the index argument is more than or equal to zero and less than the CandleTime size, then we will assign the array index with the time argument.

```
//+------------------------------------------------------------------+
//|Set Array index time                                              |
//+------------------------------------------------------------------+
void CTimeVariables::SetTime(uint index,datetime time)
  {
   if(index>=0&&index<CandleTime.Size())
     {
      CandleTime[index] = time;
     }
  }
```

The function GetTime will accept one positive integer argument, to retrieve the datetime from the array index value from CandleTime if the index argument is valid.

```
//+------------------------------------------------------------------+
//|Get Array index time                                              |
//+------------------------------------------------------------------+
datetime CTimeVariables::GetTime(uint index)
  {
   return (index>=0&&index<CandleTime.Size())?CandleTime[index]:datetime(0);
  }
```

### Time Management Class

We will declare DSTSchedule enumeration for the user/trader to select between Auto DST or Custom DST for the expert's input. MySchedule variable will be used to store the custom DST.

```
//-- Enumeration for DST schedule
enum DSTSchedule
  {
   AutoDst_Selection,//AUTO DST
   CustomDst_Selection//CUSTOM DST
  } MyDST;

DST_type MySchedule;//Variable for custom DST schedule
```

The function below will return the hour for a specific date in an integer datatype.

```
int               ReturnHour(datetime time);//Returns the Hour for a specific date
```

```
//+------------------------------------------------------------------+
//|Returns the Hour for a specific date                              |
//+------------------------------------------------------------------+
int CTimeManagement::ReturnHour(datetime time)
  {
   return Time(time).hour;
  }
```

The function below will return the minute for a specific date in an integer datatype.

```
int               ReturnMinute(datetime time);//Returns the Minute for a specific date
```

```
//+------------------------------------------------------------------+
//|Returns the Minute for a specific date                            |
//+------------------------------------------------------------------+
int CTimeManagement::ReturnMinute(datetime time)
  {
   return Time(time).min;
  }
```

The function below will return the second for a specific date in an integer datatype.

```
int               ReturnSecond(datetime time);//Returns the Second for s specific date
```

```
//+------------------------------------------------------------------+
//|Returns the Second for s specific date                            |
//+------------------------------------------------------------------+
int CTimeManagement::ReturnSecond(datetime time)
  {
   return Time(time).sec;
  }
```

The function below will return the MqlDateTime for the datetime argument.

```
//-- Will convert datetime to MqlDateTime
   MqlDateTime       Time(datetime Timetoformat);
```

```
//+------------------------------------------------------------------+
//|Will convert datetime to MqlDateTime                              |
//+------------------------------------------------------------------+
MqlDateTime CTimeManagement::Time(datetime Timetoformat)
  {
   TimeToStruct(Timetoformat,timeFormat);
   return timeFormat;
  }
```

The function below will return the datetime for the datetime argument time with modification to the hour, minute and second.

```
//-- Will return a datetime with changes to the hour,minute and second
   datetime          Time(datetime time,int Hour,int Minute,int Second);
```

```
//+------------------------------------------------------------------+
//|Will return a datetime with changes to the hour,minute and second |
//+------------------------------------------------------------------+
datetime CTimeManagement::Time(datetime time,int Hour,int Minute,int Second)
  {
   timeFormat=Time(time);
   timeFormat.hour=Hour;
   timeFormat.min=Minute;
   timeFormat.sec=Second;
   return StructToTime(timeFormat);
  }
```

The function below will return the datetime for the datetime argument time with modification to the hour and minute.

```
//-- Will return a datetime with changes to the hour and minute
   datetime          Time(datetime time,int Hour,int Minute);
```

```
//+------------------------------------------------------------------+
//|Will return a datetime with changes to the hour and minute        |
//+------------------------------------------------------------------+
datetime CTimeManagement::Time(datetime time,int Hour,int Minute)
  {
   timeFormat=Time(time);
   timeFormat.hour=Hour;
   timeFormat.min=Minute;
   return StructToTime(timeFormat);
  }
```

The function below will return true if the TimeTradeServer time is within the BeginTime and EndTime arguments.

```
//-- Check current time is within a time range
   bool              TimeIsInRange(datetime BeginTime,datetime EndTime);
```

```
//+------------------------------------------------------------------+
//|Check current time is within a time range                         |
//+------------------------------------------------------------------+
bool CTimeManagement::TimeIsInRange(datetime BeginTime,datetime EndTime)
  {
   if(BeginTime<=TimeTradeServer()&&EndTime>=TimeTradeServer())
     {
      return true;
     }
   return false;
  }
```

The function below will return true if the PreEvent datetime is less than or equal to TimeTradeServer and EventTime is more than TimeTradeServer.

```
//-- Check if current time is within preEvent time and Event time
   bool              TimePreEvent(datetime PreEvent,datetime Event);
```

```
//+------------------------------------------------------------------+
//|Check if current time is within preEvent time and Event time      |
//+------------------------------------------------------------------+
bool CTimeManagement::TimePreEvent(datetime PreEventTime,datetime EventTime)
  {
   if(PreEventTime<=TimeTradeServer()&&EventTime>TimeTradeServer())
     {
      return true;
     }
   return false;
  }
```

The function below will return the MqlDateTime for the current time with modification to the hour and minute.

```
//-- Return MqlDateTime for current date time with custom hour and minute
   MqlDateTime       Today(int Hour,int Minute);
```

```
//+------------------------------------------------------------------+
//|Return MqlDateTime for current date time with custom hour and     |
//|minute                                                            |
//+------------------------------------------------------------------+
MqlDateTime CTimeManagement::Today(int Hour,int Minute)
  {
   TimeTradeServer(today);
   today.hour=Hour;
   today.min=Minute;
   return today;
  }
```

The function below will return the MqlDateTime for the current time with modification to the hour, minute and second.

```
//-- Return MqlDateTime for current date time with custom hour, minute and second
   MqlDateTime       Today(int Hour,int Minute,int Second);
```

```
//+------------------------------------------------------------------+
//|Return MqlDateTime for current date time with custom hour, minute |
//|and second                                                        |
//+------------------------------------------------------------------+
MqlDateTime CTimeManagement::Today(int Hour,int Minute,int Second)
  {
   TimeTradeServer(today);
   today.hour=Hour;
   today.min=Minute;
   today.sec=Second;
   return today;
  }
```

The function below will return true if the current day is equal to the corresponding day of the week, or the enumeration DayOfTheWeek is equal to AllDays.

```
//-- Check current day of the week
   bool              isDayOfTheWeek(DayOfTheWeek Day);
```

```
//+------------------------------------------------------------------+
//|Check current day of the week                                     |
//+------------------------------------------------------------------+
bool CTimeManagement::isDayOfTheWeek(DayOfTheWeek Day)
  {
   switch(Day)
     {
      case  Monday://Monday
         if(DayOfWeek(TimeTradeServer())==MONDAY)
           {
            return true;
           }
         break;
      case Tuesday://Tuesday
         if(DayOfWeek(TimeTradeServer())==TUESDAY)
           {
            return true;
           }
         break;
      case Wednesday://Wednesday
         if(DayOfWeek(TimeTradeServer())==WEDNESDAY)
           {
            return true;
           }
         break;
      case Thursday://Thursday
         if(DayOfWeek(TimeTradeServer())==THURSDAY)
           {
            return true;
           }
         break;
      case Friday://Friday
         if(DayOfWeek(TimeTradeServer())==FRIDAY)
           {
            return true;
           }
         break;
      case AllDays://All days
         return true;
         break;
      default://Unknown
         break;
     }
   return false;
  }
```

The function below will return the Day of Week for a specific date.

```
//-- Return enumeration Day of week for a certain date
   ENUM_DAY_OF_WEEK  DayOfWeek(datetime time);
```

```
//+------------------------------------------------------------------+
//|Return enumeration Day of week for a certain date                 |
//+------------------------------------------------------------------+
ENUM_DAY_OF_WEEK CTimeManagement::DayOfWeek(datetime time)
  {
   return (ENUM_DAY_OF_WEEK)Time(time).day_of_week;
  }
```

### Candle Properties Class

A new function has been added to this class.

```
//+------------------------------------------------------------------+
//|CandleProperties class                                            |
//+------------------------------------------------------------------+
class CCandleProperties : public CChartProperties
  {
private:
   CTimeManagement   Time;//TimeManagement object
   CTimeVariables    CTV;//Timevariables object
public:
   double            Open(int CandleIndex,ENUM_TIMEFRAMES Period=PERIOD_CURRENT,string SYMBOL=NULL);//Retrieve Candle Open-Price
   double            Close(int CandleIndex,ENUM_TIMEFRAMES Period=PERIOD_CURRENT,string SYMBOL=NULL);//Retrieve Candle Close-Price
   double            High(int CandleIndex,ENUM_TIMEFRAMES Period=PERIOD_CURRENT,string SYMBOL=NULL);//Retrieve Candle High-Price
   double            Low(int CandleIndex,ENUM_TIMEFRAMES Period=PERIOD_CURRENT,string SYMBOL=NULL);//Retrieve Candle Low-Price
   bool              IsLargerThanPreviousAndNext(datetime CandleTime,int Offset,string SYMBOL);//Determine if one candle is larger than two others
   bool              NewCandle(int index,ENUM_TIMEFRAMES period=PERIOD_CURRENT,string SYMBOL=NULL);//Check if a new candle is present
  };
```

The function NewCandle will return true when a new candle has formed, and will then save the current candle's open time in the class Timevariables using the function SetTime. The previously saved time will be compared with the current candle's open time to check if the times are different, if the times are different, then we assume a new candle has formed.

```
//+------------------------------------------------------------------+
//|Check if a new candle is present                                  |
//+------------------------------------------------------------------+
bool CCandleProperties::NewCandle(int index,ENUM_TIMEFRAMES period=PERIOD_CURRENT,string SYMBOL=NULL)
  {
   if(CTV.GetTime(index) == iTime(((SYMBOL==NULL)?Symbol():SYMBOL),period,0))
     {
      return false;//Candle time are equal no new candles have formed
     }
   else
     {
     //--- Candle time has changed set the new time
      CTV.SetTime(index,iTime(((SYMBOL==NULL)?Symbol():SYMBOL),period,0));
      return true;
     }
  }
```

### Sessions Class

This class's purpose is to deal with session trading times. We won't make use of these trading sessions times in this article, but we will have a use for this class later on. This class will inherit from the TimeManagement class as this class will utilize the functions in TimeManagement.

> ![Session Times](https://c.mql5.com/2/86/TradeSession_Times.png)

```
#include "Timemanagement.mqh"
//+------------------------------------------------------------------+
//|Sessions Class                                                    |
//+------------------------------------------------------------------+
class CSessions:CTimeManagement
  {
public:
                     CSessions(void) {}
                    ~CSessions(void) {}
   //--- Check if trading Session has began
   bool              isSessionStart(int offsethour=0,int offsetmin=0);
   //--- Check if trading Session has ended
   bool              isSessionEnd(int offsethour=0,int offsetmin=45);
   //--- Get Session End datetime
   datetime          SessionEnd(int offsethour=0,int offsetmin=45);
  };
```

The function below will check all valid trading Sessions for the current Symbol and Day Of Week, Once the earliest trading session is found we will add and offset to this time ex. if the earliest trading session is from 01:00-05:00 then our offset hour is 1 and the offset minute is 0, our trading session will start at 02:00-05:00. The purpose of knowing the starting session time is to avoid large spreads that usually occur at the beginning of the trading session. If our trading session is currently active, the function will return true.

```
//+------------------------------------------------------------------+
//|Check if trading Session has started                              |
//+------------------------------------------------------------------+
bool CSessions::isSessionStart(int offsethour=0,int offsetmin=0)
  {
//--- Declarations
   datetime datefrom,dateto,DateFrom[],DateTo[];

//--- Find all session times
   for(int i=0; i<10; i++)
     {
      //--- Get the session dates for the current symbol and Day of week
      if(SymbolInfoSessionTrade(Symbol(),DayOfWeek(TimeTradeServer()),i,datefrom,dateto))
        {
         //--- Check if the end date's hour is at midnight
         if(ReturnHour(dateto)==00||ReturnHour(dateto)==24)
           {
            //--- Adjust the date to one minute before midnight
            dateto = Time(TimeTradeServer(),23,59);
           }
         //--- Re-adjust DateFrom Array size
         ArrayResize(DateFrom,int(ArraySize(DateFrom))+1,int(ArraySize(DateFrom))+2);
         //--- Assign the last array index datefrom value
         DateFrom[int(ArraySize(DateFrom))-1] = datefrom;
         //--- Re-adjust DateTo Array size
         ArrayResize(DateTo,int(ArraySize(DateTo))+1,int(ArraySize(DateTo))+2);
         //--- Assign the last array index dateto value
         DateTo[int(ArraySize(DateTo))-1] = dateto;
        }
     }

//--- Check if there are session times
   if(DateFrom.Size()>0)
     {
      /* Adjust DateFrom index zero date as the first index date will be the earliest date
       from the whole array, we add the offset to this date only*/
      DateFrom[0] = TimePlusOffset(DateFrom[0],MinutesS(startoffsetmin));
      DateFrom[0] = TimePlusOffset(DateFrom[0],HoursS(startoffsethour));
      //--- Iterate through the whole array
      for(uint i=0; i<DateFrom.Size(); i++)
        {
         //--- Check if the current time is within the trading session
         if(TimeIsInRange(DateFrom[i],DateTo[i]))
           {
            return true;
           }
        }
     }
   else
     {
      //--- If there are no trading session times
      return true;
     }
   return false;
  }
```

The function below will return the true if the session has ended, on some brokers an hour before the trading session ends spreads are huge, so this function can help us avoid trading during those times.

```
//+------------------------------------------------------------------+
//|Check if trading Session has ended                                |
//+------------------------------------------------------------------+
bool CSessions::isSessionEnd(int offsethour=0,int offsetmin=45)
  {
//--- Declarations
   datetime datefrom,dateto,DateTo[],lastdate=0,sessionend;

//--- Find all session times
   for(int i=0; i<10; i++)
     {
      //--- Get the session dates for the current symbol and Day of week
      if(SymbolInfoSessionTrade(Symbol(),DayOfWeek(TimeTradeServer()),i,datefrom,dateto))
        {
         //--- Check if the end date's hour is at midnight
         if(ReturnHour(dateto)==00||ReturnHour(dateto)==24)
           {
            //--- Adjust the date to one minute before midnight
            dateto = Time(TimeTradeServer(),23,59);
           }
         //--- Re-adjust DateTo Array size
         ArrayResize(DateTo,int(ArraySize(DateTo))+1,int(ArraySize(DateTo))+2);
         //--- Assign the last array index dateto value
         DateTo[int(ArraySize(DateTo))-1] = dateto;
        }
     }

//--- Check if there are session times
   if(DateTo.Size()>0)
     {
      //--- Assign lastdate a default value
      lastdate = DateTo[0];
      //--- Iterate through the whole array
      for(uint i=0; i<DateTo.Size(); i++)
        {
         //--- Check for the latest date in the array
         if(DateTo[i]>lastdate)
           {
            lastdate = DateTo[i];
           }
        }
     }
   else
     {
      //--- If there are no trading session times
      return false;
     }
   /* get the current time and modify the hour and minute time to the lastdate variable
   and assign the new datetime to sessionend variable*/
   sessionend = Today(ReturnHour(lastdate),ReturnMinute(lastdate));
//--- Re-adjust the sessionend dates with the minute and hour offsets
   sessionend = TimeMinusOffset(sessionend,MinutesS(offsetmin));
   sessionend = TimeMinusOffset(sessionend,HoursS(offsethour));

//--- Check if sessionend date is more than the current time
   if(TimeTradeServer()<sessionend)
     {
      return false;
     }
   return true;
  }
```

The function below will return the trading session ending date for the current day, there are some limitations to trading session times in MQL5, I've noticed that during a holiday some symbols can close a lot earlier than the trading session time would indicate so that is something to keep in mind.

```
//+------------------------------------------------------------------+
//|Get Session End datetime                                          |
//+------------------------------------------------------------------+
datetime CSessions::SessionEnd(int offsethour=0,int offsetmin=45)
  {
//--- Declarations
   datetime datefrom,dateto,DateTo[],lastdate=0,sessionend;

//--- Find all session times
   for(int i=0; i<10; i++)
     {
      //--- Get the session dates for the current symbol and Day of week
      if(SymbolInfoSessionTrade(Symbol(),DayOfWeek(TimeTradeServer()),i,datefrom,dateto))
        {
         //--- Check if the end date's hour is at midnight
         if(CTV.ReturnHour(dateto)==00||CTV.ReturnHour(dateto)==24)
           {
            //--- Adjust the date to one minute before midnight
            dateto = Time(TimeTradeServer(),23,59);
           }
         //--- Re-adjust DateTo Array size
         ArrayResize(DateTo,int(ArraySize(DateTo))+1,int(ArraySize(DateTo))+2);
         //--- Assign the last array index dateto value
         DateTo[int(ArraySize(DateTo))-1] = dateto;
        }
     }

//--- Check if there are session times
   if(DateTo.Size()>0)
     {
      //--- Assign lastdate a default value
      lastdate = DateTo[0];
      //--- Iterate through the whole array
      for(uint i=0; i<DateTo.Size(); i++)
        {
         //--- Check for the latest date in the array
         if(DateTo[i]>lastdate)
           {
            lastdate = DateTo[i];
           }
        }
     }
   else
     {
      //--- If there are no trading session times
      return 0;
     }
   /* get the current time and modify the hour and minute time to the lastdate variable
   and assign the new datetime to sessionend variable*/
   sessionend = Today(ReturnHour(lastdate),ReturnMinute(lastdate));
//--- Re-adjust the sessionend dates with the minute and hour offsets
   sessionend = TimeMinusOffset(sessionend,MinutesS(offsetmin));
   sessionend = TimeMinusOffset(sessionend,HoursS(offsethour));
//--- return sessionend date
   return sessionend;
  }
```

### News Class

This class is the most crucial in the whole project and the largest, spanning over 1000 lines of code alone. From part 2 we made major improvements to this class and the code that interacts with the calendar database in the common folder. In this article we will create another database but this time, the database will be in memory.

**Benefits of DBs in memory compared to storage**

- Speed: In-memory databases store data directly in RAM, leading to significantly faster read and write operations. This is especially beneficial for applications requiring real-time data processing and analytics. Since RAM is much faster than disk storage, accessing data in an in-memory database is quicker. This reduces response times and improves overall performance.
- Performance: The reduced latency and faster data access translate into higher throughput, meaning the database can handle more transactions per second. In-memory databases can process large volumes of data more efficiently, making them suitable for big data applications, analytics, and other intensive computing tasks.

We will still use our DB in storage, we will collect the data from the DB in storage and transfer this data into our new DB in memory for a balanced performance when back-testing as only using the DB in storage will affect before drastically depending on your computer's specs.

We will start off with the declarations outside the class for use in the expert's input. The DBMemoryConnection integer variable will hold the integer connection handle for our database in memory. Calendar\_Importance enumeration will be used to select different event importance in the expert's input parameter. Event\_Sector enumeration will be used to select different event sectors in the expert's input parameter. Event\_Frequency enumeration will be used to select different event frequencies for the expert's input parameter. Event\_Type enumeration will be used to select different event types for the expert's input parameter. Event\_Currency enumeration will be used to select different event currency options in the expert's input parameter. UpcomingNews Calendar structure variable will store the next economic event details for easy access in other class/files, we declared it outside the news class.

```
int       DBMemoryConnection;//In memory database handle

//--- Enumeration for Calendar Importance
enum Calendar_Importance
  {
   Calendar_Importance_None,//NONE
   Calendar_Importance_Low,//LOW
   Calendar_Importance_Moderate,//MODERATE
   Calendar_Importance_High,//HIGH
   Calendar_Importance_All//ALL
  } myImportance;

//--- Enumeration for Calendar Sector
enum Event_Sector
  {
   Event_Sector_None,//NONE
   Event_Sector_Market,//MARKET
   Event_Sector_Gdp,//GDP
   Event_Sector_Jobs,//JOBS
   Event_Sector_Prices,//PRICES
   Event_Sector_Money,//MONEY
   Event_Sector_Trade,//TRADE
   Event_Sector_Government,//GOVERNMENT
   Event_Sector_Business,//BUSINESS
   Event_Sector_Consumer,//CONSUMER
   Event_Sector_Housing,//HOUSING
   Event_Sector_Taxes,//TAXES
   Event_Sector_Holidays,//HOLIDAYS
   Event_Sector_ALL//ALL
  } mySector;

//--- Enumeration for Calendar Event Frequency
enum Event_Frequency
  {
   Event_Frequency_None,//NONE
   Event_Frequency_Week,//WEEK
   Event_Frequency_Month,//MONTH
   Event_Frequency_Quarter,//QUARTER
   Event_Frequency_Year,//YEAR
   Event_Frequency_Day,//DAY
   Event_Frequency_ALL//ALL
  } myFrequency;

//--- Enumeration for Calendar Event type
enum Event_Type
  {
   Event_Type_Event,//EVENT
   Event_Type_Indicator,//INDICATOR
   Event_Type_Holiday,//HOLIDAY
   Event_Type_All//ALL
  } myType;

//--- Enumeration for Calendar Event Currency
enum Event_Currency
  {
   Event_Currency_Symbol,//SYMBOL CURRENCIES
   Event_Currency_Margin,//SYMBOL MARGIN
   Event_Currency_Base,//SYMBOL BASE
   Event_Currency_Profit,//SYMBOL PROFIT
   Event_Currency_ALL,//ALL CURRENCIES
   Event_Currency_NZD_NZ,//NZD -> NZ
   Event_Currency_EUR_EU,//EUR -> EU
   Event_Currency_JPY_JP,//JPY -> JP
   Event_Currency_CAD_CA,//CAD -> CA
   Event_Currency_AUD_AU,//AUD -> AU
   Event_Currency_CNY_CN,//CNY -> CN
   Event_Currency_EUR_IT,//EUR -> IT
   Event_Currency_SGD_SG,//SGD -> SG
   Event_Currency_EUR_DE,//EUR -> DE
   Event_Currency_EUR_FR,//EUR -> FR
   Event_Currency_BRL_BR,//BRL -> BR
   Event_Currency_MXN_MX,//MXN -> MX
   Event_Currency_ZAR_ZA,//ZAR -> ZA
   Event_Currency_HKD_HK,//HKD -> HK
   Event_Currency_INR_IN,//INR -> IN
   Event_Currency_NOK_NO,//NOK -> NO
   Event_Currency_USD_US,//USD -> US
   Event_Currency_GBP_GB,//GBP -> GB
   Event_Currency_CHF_CH,//CHF -> CH
   Event_Currency_KRW_KR,//KRW -> KW
   Event_Currency_EUR_ES,//EUR -> ES
   Event_Currency_SEK_SE,//SEK -> SE
   Event_Currency_ALL_WW//ALL -> WW
  } myCurrency;

//--- Structure variable to store Calendar next Event data
Calendar UpcomingNews;
```

Additional functionality to the news class:

- Expansion to the Enumeration CalendarComponents: EventInfo\_View was added to show event details in the calendar storage database, Currencies\_View was added to show all currencies available by the MQL5 economic calendar.
- Structure Array CalendarContents has increased by size from 10 to 12 to accommodate the two new views EventInfo\_View and Currencies\_View.
- Declaration of DBMemory variable of MQL5CalendarContents structure type to store DB properties.
- Declaration of CalendarData structure and variables to store all data from the Calendar DB in the common folder.
- Declaration of the function GetCalendar which will request all the filtered data from the calendar DB in the common folder and store this data into the CalendarData structure array Data this array data will be inserted into our new calendar DB in memory once it is created.
- Declaration of the function GetAutoDST to retrieve the enumeration DST\_type from the table AutoDST in the Calendar DB in the common folder.
- Declaration of the function Request\_Importance to retrieve the string request for the event importance based on the Calendar\_Importance enumeration.
- Declaration of the function Request\_Frequency to retrieve the string request for the event frequency based on the Event\_Frequency enumeration.
- Declaration of the function Request\_Sector to retrieve the string request for the event sector based on the Event\_Sector enumeration.
- Declaration of the function Request\_Type to retrieve the string request for the event type based on the Event\_Type enumeration.
- Declaration of the function Request\_Currency to retrieve the string request for the event currency based on the Event\_Currency enumeration.
- Declaration of the function EconomicDetailsMemory will populate the NewsTime Calendar structure array with the events for a specific date.
- Declaration of the function CreateEconomicDatabaseMemory will create the calendar database in memory.
- Declaration of the function EconomicNextEvent will update UpcomingNews structure variable with the next event data.
- Declaration of the function GetImpact will retrieve Upcoming Event Impact data.
- Declaration of the function IMPORTANCE will convert the Importance string into Calendar Event Importance Enumeration.
- Declaration of the function IMPORTANCE will convert Calendar\_Importance Enumeration into Calendar Event Importance Enumeration.
- Declaration of the function GetImportance will convert Calendar Event Importance Enumeration into string Importance Rating.
- Declaration of the function GetImportance\_color will retrieve color for each Calendar Event Importance Enumeration.
- Declaration of the function SECTOR will convert Event\_Sector Enumeration into Calendar Event Sector Enumeration.
- Declaration of the function FREQUENCY will convert Event\_Frequency Enumeration into Calendar Event Frequency Enumeration.
- Declaration of the function TYPE will convert Event\_Type Enumeration into Calendar Event Type Enumeration.

```
//+------------------------------------------------------------------+
//|News class                                                        |
//+------------------------------------------------------------------+
class CNews : private CCandleProperties
  {
   //Private Declarations Only accessable by this class/header file
private:

   //-- To keep track of what is in our database
   enum CalendarComponents
     {
      AutoDST_Table,//AutoDST Table
      CalendarAU_View,//View for DST_AU
      CalendarNONE_View,//View for DST_NONE
      CalendarUK_View,//View for DST_UK
      CalendarUS_View,//View for DST_US
      EventInfo_View,//View for Event Information
      Currencies_View,//View for Currencies
      Record_Table,// Record Table
      TimeSchedule_Table,//TimeSchedule Table
      MQL5Calendar_Table,//MQL5Calendar Table
      AutoDST_Trigger,//Table Trigger for AutoDST
      Record_Trigger//Table Trigger for Record
     };

   //-- structure to retrieve all the objects in the database
   struct SQLiteMaster
     {
      string         type;//will store object's type
      string         name;//will store object's name
      string         tbl_name;//will store table name
      int            rootpage;//will store rootpage
      string         sql;//Will store the sql create statement
     } DBContents[];//Array of type SQLiteMaster

   //--  MQL5CalendarContents inherits from SQLiteMaster structure
   struct MQL5CalendarContents:SQLiteMaster
     {
      CalendarComponents  Content;
      string         insert;//Will store the sql insert statement
     } CalendarContents[12],DBMemory;//Array to Store objects in our database

   CTimeManagement   CTime;//TimeManagement Object declaration
   CDaylightSavings_UK  Savings_UK;//DaylightSavings Object for the UK and EU
   CDaylightSavings_US  Savings_US;//DaylightSavings Object for the US
   CDaylightSavings_AU  Savings_AU;//DaylightSavings Object for the AU

   bool              AutoDetectDST(DST_type &dstType);//Function will determine Broker DST
   DST_type          DSTType;//variable of DST_type enumeration declared in the CommonVariables class/header file
   bool              InsertIntoTables(int db,Calendar &Evalues[]);//Function for inserting Economic Data in to a database's table
   void              CreateAutoDST(int db);//Function for creating and inserting Recommend DST for the Broker into a table
   bool              CreateCalendarTable(int db,bool &tableExists);//Function for creating a table in a database
   bool              CreateTimeTable(int db,bool &tableExists);//Function for creating a table in a database
   void              CreateCalendarViews(int db);//Function for creating views in a database
   void              CreateRecordTable(int db);//Creates a table to store the record of when last the Calendar database was updated/created
   string            DropRequest;//Variable for dropping tables in the database

   //-- Function for retrieving the MQL5CalendarContents structure for the enumartion type CalendarComponents
   MQL5CalendarContents CalendarStruct(CalendarComponents Content)
     {
      MQL5CalendarContents Calendar;
      for(uint i=0;i<CalendarContents.Size();i++)
        {
         if(CalendarContents[i].Content==Content)
           {
            return CalendarContents[i];
           }
        }
      return Calendar;
     }

   //--- To Store Calendar DB Data
   struct CalendarData
     {
      int            EventId;//Event Id
      string         Country;//Event Country
      string         EventName;//Event Name
      string         EventType;//Event Type
      string         EventImportance;//Event Importance
      string         EventCurrency;//Event Currency
      string         EventCode;//Event Code
      string         EventSector;//Event Sector
      string         EventForecast;//Event Forecast Value
      string         EventPreval;//Event Previous Value
      string         EventImpact;//Event Impact
      string         EventFrequency;//Event Frequency
      string         DST_UK;//DST UK
      string         DST_US;//DST US
      string         DST_AU;//DST AU
      string         DST_NONE;//DST NONE
     } DB_Data[],DB_Cal;//Structure variables

   //--- Will Retrieve all relevant Calendar data for DB in Memory from DB in Storage
   void              GetCalendar(CalendarData &Data[])
     {
      //--- Open calendar DB in Storage
      int db=DatabaseOpen(NEWS_DATABASE_FILE, DATABASE_OPEN_READWRITE | DATABASE_OPEN_CREATE| DATABASE_OPEN_COMMON);
      if(db==INVALID_HANDLE)//Checks if the database was able to be opened
        {
         //if opening the database failed
         if(!FileIsExist(NEWS_DATABASE_FILE,FILE_COMMON))//Checks if the database Calendar exists in the common folder
           {
            return;//Returns true when the database was failed to be opened and the file doesn't exist in the common folder
           }
        }
      //--- Get filtered calendar DB data
      string SqlRequest = StringFormat("Select MQ.EventId,MQ.Country,MQ.EventName,MQ.EventType,MQ.EventImportance,MQ.EventCurrency,"
                                       "MQ.EventCode,MQ.EventSector,MQ.EventForecast,MQ.EventPreValue,MQ.EventImpact,MQ.EventFrequency,"
                                       "TS.DST_UK,TS.DST_US,TS.DST_AU,TS.DST_NONE from %s MQ "
                                       "Inner Join %s TS on TS.ID=MQ.ID "
                                       "Where %s and %s and %s and %s and %s;",
                                       CalendarStruct(MQL5Calendar_Table).name,CalendarStruct(TimeSchedule_Table).name,
                                       Request_Importance(myImportance),Request_Frequency(myFrequency),
                                       Request_Sector(mySector),Request_Type(myType),Request_Currency(myCurrency));
      //--- Process Sql request
      int Request = DatabasePrepare(db,SqlRequest);
      if(Request==INVALID_HANDLE)
        {
         //--- Print details if request failed.
         Print("DB: ",NEWS_DATABASE_FILE, " request failed with code ", GetLastError());
         Print("SQL");
         Print(SqlRequest);
        }
      else
        {
         //--- Clear data from whole array
         ArrayRemove(Data,0,WHOLE_ARRAY);
         //--- create structure variable to get data from request
         CalendarData data;
         //Assigning values from the sql query into Data structure array
         for(int i=0; DatabaseReadBind(Request,data); i++)
           {
            //--- Resize Data Array
            ArrayResize(Data,i+1,i+2);
            Data[i]  = data;
           }
        }
      DatabaseFinalize(Request);//Finalize request
      //--- Close Calendar database
      DatabaseClose(db);
     }

   //--- Retrieve the AutoDST enumeration data from calendar DB in storage
   DST_type          GetAutoDST()
     {
      string Sch_Dst;
      //--- open the database 'Calendar' in the common folder
      int db=DatabaseOpen(NEWS_DATABASE_FILE, DATABASE_OPEN_READONLY|DATABASE_OPEN_COMMON);

      if(db==INVALID_HANDLE)//Checks if 'Calendar' failed to be opened
        {
         if(!FileIsExist(NEWS_DATABASE_FILE,FILE_COMMON))//Checks if 'Calendar' database exists
           {
            Print("Could not find Database!");
            return DST_NONE;//return default value when failed.
           }
        }

      //--- Sql query to get AutoDST value
      string request_text="SELECT DST FROM 'AutoDST'";
      //--- Process sql request
      int request=DatabasePrepare(db,request_text);
      if(request==INVALID_HANDLE)
        {
         Print("DB: ",NEWS_DATABASE_FILE, " request failed with code ", GetLastError());
         DatabaseClose(db);//Close Database
         return DST_NONE;//return default value when failed.
        }

      //--- Read Sql request output data
      if(DatabaseRead(request))
        {
         //-- Store the first column data into string variable Sch_Dst
         if(!DatabaseColumnText(request,0,Sch_Dst))
           {
            Print("DatabaseRead() failed with code ", GetLastError());
            DatabaseFinalize(request);//Finalize request
            DatabaseClose(db);//Closes the database 'Calendar'
            return DST_NONE;//return default value when failed.
           }
        }
      DatabaseFinalize(request);//Finalize request
      DatabaseClose(db);//Closes the database 'Calendar'
      return (Sch_Dst=="DST_UK")?DST_UK:(Sch_Dst=="DST_US")?DST_US:
             (Sch_Dst=="DST_AU")?DST_AU:DST_NONE;//Returns the enumeration value for each corresponding string
     }

   //--- Retrieve Sql request string for calendar event Importance
   string            Request_Importance(Calendar_Importance Importance)
     {
      //--- Constant request prefix string
      const string constant="MQ.EventImportance";
      //--- switch statement for Calendar_Importance enumeration
      switch(Importance)
        {
         case Calendar_Importance_All://String Request for all event Importance
            return constant+"<>'"+EnumToString(myImportance)+"'";
            break;
         default://String Request for any event Importance
            return constant+"='"+EnumToString(IMPORTANCE(myImportance))+"'";
            break;
        }
     }

   //--- Retrieve Sql request string for calendar event Frequency
   string            Request_Frequency(Event_Frequency Frequency)
     {
      //--- Constant request prefix string
      const string constant="MQ.EventFrequency";
      //--- switch statement for Event_Frequency enumeration
      switch(Frequency)
        {
         case Event_Frequency_ALL://String Request for all event frequencies
            return constant+"<>'"+EnumToString(myFrequency)+"'";
            break;
         default://String Request for any event frequency
            return constant+"='"+EnumToString(FREQUENCY(myFrequency))+"'";
            break;
        }
     }

   //--- Retrieve Sql request string for calendar event Sector
   string            Request_Sector(Event_Sector Sector)
     {
      //--- Constant request prefix string
      const string constant="MQ.EventSector";
      //--- switch statement for Event_Sector enumeration
      switch(Sector)
        {
         case Event_Sector_ALL://String Request for all event sectors
            return constant+"<>'"+EnumToString(mySector)+"'";
            break;
         default://String Request for any event sector
            return constant+"='"+EnumToString(SECTOR(mySector))+"'";
            break;
        }
     }

   //--- Retrieve Sql request string for calendar event type
   string            Request_Type(Event_Type Type)
     {
      //--- Constant request prefix string
      const string constant="MQ.EventType";
      //--- switch statement for Event_Type enumeration
      switch(Type)
        {
         case Event_Type_All://String Request for all event types
            return constant+"<>'"+EnumToString(myType)+"'";
            break;
         default://String request for any event type
            return constant+"='"+EnumToString(TYPE(myType))+"'";
            break;
        }
     }

   //--- Retrieve Sql request string for calendar event Currency
   string            Request_Currency(Event_Currency Currency)
     {
      //--- Constant request prefix string and request suffix
      const string constant_prefix="(MQ.EventCurrency",constant_suffix="')";
      //--- switch statement for Event_Currency enumeration
      switch(Currency)
        {
         case Event_Currency_ALL://String Request for all currencies
            return constant_prefix+"<>'"+EnumToString(myCurrency)+constant_suffix;
            break;
         case Event_Currency_Symbol://String Request for all symbol currencies
            return constant_prefix+"='"+CSymbol.CurrencyBase()+"' or MQ.EventCurrency='"+
                   CSymbol.CurrencyMargin()+"' or MQ.EventCurrency='"+CSymbol.CurrencyProfit()+constant_suffix;
            break;
         case Event_Currency_Margin://String Request for Margin currency
            return constant_prefix+"='"+CSymbol.CurrencyMargin()+constant_suffix;
            break;
         case Event_Currency_Base://String Request for Base currency
            return constant_prefix+"='"+CSymbol.CurrencyBase()+constant_suffix;
            break;
         case Event_Currency_Profit://String Request for Profit currency
            return constant_prefix+"='"+CSymbol.CurrencyProfit()+constant_suffix;
            break;
         case Event_Currency_NZD_NZ://String Request for NZD currency
            return constant_prefix+"='NZD' and MQ.EventCode='NZ"+constant_suffix;
            break;
         case Event_Currency_EUR_EU://String Request for EUR currency and EU code
            return constant_prefix+"='EUR' and MQ.EventCode='EU"+constant_suffix;
            break;
         case Event_Currency_JPY_JP://String Request for JPY currency
            return constant_prefix+"='JPY' and MQ.EventCode='JP"+constant_suffix;
            break;
         case Event_Currency_CAD_CA://String Request for CAD currency
            return constant_prefix+"='CAD' and MQ.EventCode='CA"+constant_suffix;
            break;
         case Event_Currency_AUD_AU://String Request for AUD currency
            return constant_prefix+"='AUD' and MQ.EventCode='AU"+constant_suffix;
            break;
         case Event_Currency_CNY_CN://String Request for CNY currency
            return constant_prefix+"='CNY' and MQ.EventCode='CN"+constant_suffix;
            break;
         case Event_Currency_EUR_IT://String Request for EUR currency and IT code
            return constant_prefix+"='EUR' and MQ.EventCode='IT"+constant_suffix;
            break;
         case Event_Currency_SGD_SG://String Request for SGD currency
            return constant_prefix+"='SGD' and MQ.EventCode='SG"+constant_suffix;
            break;
         case Event_Currency_EUR_DE://String Request for EUR currency and DE code
            return constant_prefix+"='EUR' and MQ.EventCode='DE"+constant_suffix;
            break;
         case Event_Currency_EUR_FR://String Request for EUR currency and FR code
            return constant_prefix+"='EUR' and MQ.EventCode='FR"+constant_suffix;
            break;
         case Event_Currency_BRL_BR://String Request for BRL currency
            return constant_prefix+"='BRL' and MQ.EventCode='BR"+constant_suffix;
            break;
         case Event_Currency_MXN_MX://String Request for MXN currency
            return constant_prefix+"='MXN' and MQ.EventCode='MX"+constant_suffix;
            break;
         case Event_Currency_ZAR_ZA://String Request for ZAR currency
            return constant_prefix+"='ZAR' and MQ.EventCode='ZA"+constant_suffix;
            break;
         case Event_Currency_HKD_HK://String Request for HKD currency
            return constant_prefix+"='HKD' and MQ.EventCode='HK"+constant_suffix;
            break;
         case Event_Currency_INR_IN://String Request for INR currency
            return constant_prefix+"='INR' and MQ.EventCode='IN"+constant_suffix;
            break;
         case Event_Currency_NOK_NO://String Request for NOK currency
            return constant_prefix+"='NOK' and MQ.EventCode='NO"+constant_suffix;
            break;
         case Event_Currency_USD_US://String Request for USD currency
            return constant_prefix+"='USD' and MQ.EventCode='US"+constant_suffix;
            break;
         case Event_Currency_GBP_GB://String Request for GBP currency
            return constant_prefix+"='GBP' and MQ.EventCode='GB"+constant_suffix;
            break;
         case Event_Currency_CHF_CH://String Request for CHF currency
            return constant_prefix+"='CHF' and MQ.EventCode='CH"+constant_suffix;
            break;
         case Event_Currency_KRW_KR://String Request for KRW currency
            return constant_prefix+"='KRW' and MQ.EventCode='KR"+constant_suffix;
            break;
         case Event_Currency_EUR_ES://String Request for EUR currency and ES code
            return constant_prefix+"='EUR' and MQ.EventCode='ES"+constant_suffix;
            break;
         case Event_Currency_SEK_SE://String Request for SEK currency
            return constant_prefix+"='SEK' and MQ.EventCode='SE"+constant_suffix;
            break;
         case Event_Currency_ALL_WW://String Request for ALL currency
            return constant_prefix+"='ALL' and MQ.EventCode='WW"+constant_suffix;
            break;
         default://String Request for no currencies
            return constant_prefix+"='"+constant_suffix;
            break;
        }
     }

   //Public declarations accessable via a class's Object
public:
                     CNews(void);//Constructor
                    ~CNews(void);//Destructor
   void              CreateEconomicDatabase();//Creates the Calendar database for a specific Broker
   datetime          GetLatestNewsDate();//Gets the latest/newest date in the Calendar database
   void              EconomicDetails(Calendar &NewsTime[],datetime date_from=0,datetime date_to=0);//Gets values from the MQL5 economic Calendar
   void              EconomicDetailsMemory(Calendar &NewsTime[],datetime date);//Gets values from the MQL5 DB Calendar in Memory
   void              CreateEconomicDatabaseMemory();//Create calendar database in memory
   void              EconomicNextEvent(datetime date=0);//Will update UpcomingNews structure variable with the next event data
   bool              UpdateRecords();//Checks if the main Calendar database needs an update or not
   ENUM_CALENDAR_EVENT_IMPACT GetImpact();//Will retrieve Upcoming Event Impact data

   //--- Convert Importance string into Calendar Event Importance Enumeration
   ENUM_CALENDAR_EVENT_IMPORTANCE IMPORTANCE(string Importance)
     {
      //--- Calendar Importance is High
      if(Importance==EnumToString(CALENDAR_IMPORTANCE_HIGH))
        {
         return CALENDAR_IMPORTANCE_HIGH;
        }
      else
         //--- Calendar Importance is Moderate
         if(Importance==EnumToString(CALENDAR_IMPORTANCE_MODERATE))
           {
            return CALENDAR_IMPORTANCE_MODERATE;
           }
         else
            //--- Calendar Importance is Low
            if(Importance==EnumToString(CALENDAR_IMPORTANCE_LOW))
              {
               return CALENDAR_IMPORTANCE_LOW;
              }
            else
               //--- Calendar Importance is None
              {
               return CALENDAR_IMPORTANCE_NONE;
              }
     }

   //--- Convert Calendar_Importance Enumeration into Calendar Event Importance Enumeration
   ENUM_CALENDAR_EVENT_IMPORTANCE IMPORTANCE(Calendar_Importance Importance)
     {
      //--- switch statement for Calendar_Importance enumeration
      switch(Importance)
        {
         case Calendar_Importance_None://None
            return CALENDAR_IMPORTANCE_NONE;
            break;
         case Calendar_Importance_Low://Low
            return CALENDAR_IMPORTANCE_LOW;
            break;
         case Calendar_Importance_Moderate://Moderate
            return CALENDAR_IMPORTANCE_MODERATE;
            break;
         case Calendar_Importance_High://High
            return CALENDAR_IMPORTANCE_HIGH;
            break;
         default://None
            return CALENDAR_IMPORTANCE_NONE;
            break;
        }
     }

   //--- Convert Calendar Event Importance Enumeration into string Importance Rating
   string            GetImportance(ENUM_CALENDAR_EVENT_IMPORTANCE Importance)
     {
      //--- switch statement for ENUM_CALENDAR_EVENT_IMPORTANCE enumeration
      switch(Importance)
        {
         case  CALENDAR_IMPORTANCE_HIGH://High
            return "HIGH";
            break;
         case CALENDAR_IMPORTANCE_MODERATE://Moderate
            return "MODERATE";
            break;
         case CALENDAR_IMPORTANCE_LOW://Low
            return "LOW";
            break;
         default://None
            return "NONE";
            break;
        }
     }

   //--- Retrieve color for each Calendar Event Importance Enumeration
   color             GetImportance_color(ENUM_CALENDAR_EVENT_IMPORTANCE Importance)
     {
      //--- switch statement for ENUM_CALENDAR_EVENT_IMPORTANCE enumeration
      switch(Importance)
        {
         case CALENDAR_IMPORTANCE_HIGH://High
            return clrRed;
            break;
         case CALENDAR_IMPORTANCE_MODERATE://Moderate
            return clrOrange;
            break;
         case CALENDAR_IMPORTANCE_LOW://Low
            return (isLightMode)?clrBlue:clrLightBlue;
            break;
         default://None
            return (isLightMode)?clrBlack:clrWheat;
            break;
        }
     }

   //--- Convert Event_Sector Enumeration into Calendar Event Sector Enumeration
   ENUM_CALENDAR_EVENT_SECTOR SECTOR(Event_Sector Sector)
     {
      //--- switch statement for Event_Sector enumeration
      switch(Sector)
        {
         case Event_Sector_None://NONE
            return CALENDAR_SECTOR_NONE;
            break;
         case Event_Sector_Market://MARKET
            return CALENDAR_SECTOR_MARKET;
            break;
         case Event_Sector_Gdp://GDP
            return CALENDAR_SECTOR_GDP;
            break;
         case Event_Sector_Jobs://JOBS
            return CALENDAR_SECTOR_JOBS;
            break;
         case Event_Sector_Prices://PRICES
            return CALENDAR_SECTOR_PRICES;
            break;
         case Event_Sector_Money://MONEY
            return CALENDAR_SECTOR_MONEY;
            break;
         case Event_Sector_Trade://TRADE
            return CALENDAR_SECTOR_TRADE;
            break;
         case Event_Sector_Government://GOVERNMENT
            return CALENDAR_SECTOR_GOVERNMENT;
            break;
         case Event_Sector_Business://BUSINESS
            return CALENDAR_SECTOR_BUSINESS;
            break;
         case Event_Sector_Consumer://CONSUMER
            return CALENDAR_SECTOR_CONSUMER;
            break;
         case Event_Sector_Housing://HOUSING
            return CALENDAR_SECTOR_HOUSING;
            break;
         case Event_Sector_Taxes://TAXES
            return CALENDAR_SECTOR_TAXES;
            break;
         case Event_Sector_Holidays://HOLIDAYS
            return CALENDAR_SECTOR_HOLIDAYS;
            break;
         default://Unknown
            return CALENDAR_SECTOR_NONE;
            break;
        }
     }

   //--- Convert Event_Frequency Enumeration into Calendar Event Frequency Enumeration
   ENUM_CALENDAR_EVENT_FREQUENCY FREQUENCY(Event_Frequency Frequency)
     {
      //--- switch statement for Event_Frequency enumeration
      switch(Frequency)
        {
         case  Event_Frequency_None://NONE
            return CALENDAR_FREQUENCY_NONE;
            break;
         case Event_Frequency_Day://DAY
            return CALENDAR_FREQUENCY_DAY;
            break;
         case Event_Frequency_Week://WEEK
            return CALENDAR_FREQUENCY_WEEK;
            break;
         case Event_Frequency_Month://MONTH
            return CALENDAR_FREQUENCY_MONTH;
            break;
         case Event_Frequency_Quarter://QUARTER
            return CALENDAR_FREQUENCY_QUARTER;
            break;
         case Event_Frequency_Year://YEAR
            return CALENDAR_FREQUENCY_YEAR;
            break;
         default://Unknown
            return CALENDAR_FREQUENCY_NONE;
            break;
        }
     }

   //--- Convert Event_Type Enumeration into Calendar Event Type Enumeration
   ENUM_CALENDAR_EVENT_TYPE TYPE(Event_Type Type)
     {
      //--- switch statement for Event_Type enumeration
      switch(Type)
        {
         case Event_Type_Event://EVENT
            return CALENDAR_TYPE_EVENT;
            break;
         case Event_Type_Indicator://INDICATOR
            return CALENDAR_TYPE_INDICATOR;
            break;
         case Event_Type_Holiday://HOLIDAY
            return CALENDAR_TYPE_HOLIDAY;
            break;
         default://Unknown
            return CALENDAR_TYPE_EVENT;
            break;
        }
     }
  };
```

We will first go through the SQL statement in the function GetCalendar that will request all the data from the calendar DB in the common folder for our calendar DB in memory. In this request, we select all the columns in the MQL5Calendar table and the TimeSchedule table we join the tables where their IDs are the same. We then filter the data based on the enumerations selected by the trader/user in the expert's input parameters for news settings.

```
//--- Get filtered calendar DB data
      string SqlRequest = StringFormat("Select MQ.EventId,MQ.Country,MQ.EventName,MQ.EventType,MQ.EventImportance,MQ.EventCurrency,"
                                       "MQ.EventCode,MQ.EventSector,MQ.EventForecast,MQ.EventPreValue,MQ.EventImpact,MQ.EventFrequency,"
                                       "TS.DST_UK,TS.DST_US,TS.DST_AU,TS.DST_NONE from %s MQ "
                                       "Inner Join %s TS on TS.ID=MQ.ID "
                                       "Where %s and %s and %s and %s and %s;",
                                       CalendarStruct(MQL5Calendar_Table).name,CalendarStruct(TimeSchedule_Table).name,
                                       Request_Importance(myImportance),Request_Frequency(myFrequency),
                                       Request_Sector(mySector),Request_Type(myType),Request_Currency(myCurrency));
```

We will take a look at the SQL request for the following News Settings configuration on the EURUSD symbol.

> ![News Settings Configuration 1](https://c.mql5.com/2/86/News_settings_config_1.png)

As shown below we can see when we select all for calendar importance, we purposely don't convert our Calendar\_Importance Enumeration variable myImportance because there isn't any event importance with the value 'Calendar\_Importance\_All' so we can easily select all the events where the EventImportance is not equal to 'Calendar\_Importance\_All'. The same can be said for all the News Settings input parameters that are selected to ALL.

```
case Calendar_Importance_All://String Request for all event Importance
            return constant+"<>'"+EnumToString(myImportance)+"'";
```

```
Select MQ.EventId,MQ.Country,MQ.EventName,MQ.EventType,MQ.EventImportance,MQ.EventCurrency,MQ.EventCode,MQ.EventSector,
MQ.EventForecast,MQ.EventPreValue,MQ.EventImpact,MQ.EventFrequency,TS.DST_UK,TS.DST_US,TS.DST_AU,TS.DST_NONE from MQL5Calendar MQ
Inner Join TimeSchedule TS on TS.ID=MQ.ID Where MQ.EventImportance<>'Calendar_Importance_All' and MQ.EventFrequency<>'Event_Frequency_ALL'
and MQ.EventSector<>'Event_Sector_ALL' and MQ.EventType<>'Event_Type_All' and (MQ.EventCurrency='EUR' or MQ.EventCurrency='EUR' or
MQ.EventCurrency='USD');
```

We will take another look at one more SQL request from the function GetCalendar for the following News Settings configuration on the EURUSD symbol.

> ![News Settings Configuration 2](https://c.mql5.com/2/86/News_settings_config_2.png)

As shown below when we select any other option other than all for calendar importance, we will convert our Calendar\_Importance Enumeration variable myImportance into the enumeration type ENUM\_CALENDAR\_EVENT\_IMPORTANCE, so the string can match the one stored in the MQL5Calendar table to correct get the event importance of a certain type.

```
default://String Request for any event Importance
            return constant+"='"+EnumToString(IMPORTANCE(myImportance))+"'";
```

```
Select MQ.EventId,MQ.Country,MQ.EventName,MQ.EventType,MQ.EventImportance,MQ.EventCurrency,MQ.EventCode,MQ.EventSector,MQ.EventForecast,
MQ.EventPreValue,MQ.EventImpact,MQ.EventFrequency,TS.DST_UK,TS.DST_US,TS.DST_AU,TS.DST_NONE from MQL5Calendar MQ Inner Join TimeSchedule TS
on TS.ID=MQ.ID Where MQ.EventImportance='CALENDAR_IMPORTANCE_HIGH' and MQ.EventFrequency='CALENDAR_FREQUENCY_MONTH' and
MQ.EventSector='CALENDAR_SECTOR_JOBS' and MQ.EventType='CALENDAR_TYPE_INDICATOR' and (MQ.EventCurrency<>'Event_Currency_ALL');
```

In the news class constructor, we have to initialize the array indexes for the new views for our calendar DB in the common folder. For the Event Info view this is how we will initialize the array index below.

```
//--- initializing properties for the EventInfo view
   CalendarContents[5].Content = EventInfo_View;
   CalendarContents[5].name = "Event Info";
   CalendarContents[5].sql = "CREATE VIEW IF NOT EXISTS 'Event Info' "
                             "AS SELECT EVENTID as 'ID',COUNTRY as 'Country',EVENTNAME as 'Name',"
                             "REPLACE(EVENTTYPE,'CALENDAR_TYPE_','') as 'Type',REPLACE(EVENTSECTOR,'CALENDAR_SECTOR_','') as 'Sector',"
                             "REPLACE(EVENTIMPORTANCE,'CALENDAR_IMPORTANCE_','') as 'Importance',EVENTCURRENCY as 'Currency' "
                             "FROM MQL5Calendar GROUP BY \"Name\" ORDER BY \"Country\" Asc,"
                             "CASE \"Importance\" WHEN 'HIGH' THEN 1 WHEN 'MODERATE' THEN 2 WHEN 'LOW' THEN 3 ELSE 4 END,\"Sector\" Desc;";
   CalendarContents[5].tbl_name = "Event Info";
   CalendarContents[5].type = "view";
```

Let's go through the SQL statement to create the view 'Event Info'. Firstly, we only create the view if it doesn't already exist, then we select the columns 'EVENTID' and rename it to 'ID', 'COUNTRY' and rename it to 'Country', 'EVENTNAME' and rename it to 'Name', 'EVENTTYPE' we replace the text 'CALENDAR\_TYPE\_' with an empty string and rename the column to 'Type', 'EVENTSECTOR' we replace the text 'CALENDAR\_SECTOR\_' with an empty string and rename the column to 'Sector', 'EVENTIMPORTANCE' we replace the text 'CALENDAR\_IMPORTANCE\_' with an empty string and rename the column 'Importance', 'EVENTCURRENCY' and rename it to 'Currency' from the MQL5Calendar table. We then group the query by the 'EVENTNAME' which is now 'Name' so the events with the same event name will not be shown in the view multiple times. We then give the query an order sequence, we first order the result by the 'Country' ascending so the country with the starting alphabet A like Australia will be shown first, we then order the result by the 'EVENTIMPORTANCE' which is now called 'Importance' so what I wanted to do is show the events with the highest importance, to do this we have to give the importance string/text values a ranking so in this case when Importance is 'HIGH' it gets first priority then 'MODERATE' gets second priority and 'LOW' gets third priority and finally any other value gets last priority. Furthermore, we then order the query result by the 'EVENTSECTOR' which is now named 'Sector' in descending order. Views sample will be shown below.

```
CREATE VIEW IF NOT EXISTS 'Event Info' AS SELECT MQ.EVENTID as 'ID',MQ.COUNTRY as 'Country',MQ.EVENTNAME as 'Name',REPLACE(MQ.EVENTTYPE,
'CALENDAR_TYPE_','') as 'Type',REPLACE(MQ.EVENTSECTOR,'CALENDAR_SECTOR_','') as 'Sector',REPLACE(MQ.EVENTIMPORTANCE,'CALENDAR_IMPORTANCE_','')
as 'Importance',MQ.EVENTCURRENCY as 'Currency' FROM MQL5Calendar MQ INNER JOIN TimeSchedule TS on TS.ID=MQ.ID GROUP BY "Name" ORDER BY
"Country" Asc,CASE "Importance" WHEN 'HIGH' THEN 1 WHEN 'MODERATE' THEN 2 WHEN 'LOW' THEN 3 ELSE 4 END,"Sector" Desc;
```

> ![Event Info view](https://c.mql5.com/2/86/Event_Info.png)

```
ID       Country  Name                                          Type            Sector        Importance   Currency
36030006 Australia  RBA Governor Lowe Speech                    EVENT           MONEY         HIGH         AUD
36030008 Australia  RBA Interest Rate Decision                  INDICATOR       MONEY         HIGH         AUD
36010029 Australia  PPI q/q                                     INDICATOR       PRICES        MODERATE     AUD
36030014 Australia  RBA Trimmed Mean CPI q/q                    INDICATOR       PRICES        MODERATE     AUD
36030009 Australia  RBA Weighted Median CPI q/q                 INDICATOR       PRICES        MODERATE     AUD
36010031 Australia  Wage Price Index q/q                        INDICATOR       PRICES        MODERATE     AUD
36030026 Australia  RBA Assistant Governor Boulton Speech       EVENT           MONEY         MODERATE     AUD
36030024 Australia  RBA Assistant Governor Bullock Speech       EVENT           MONEY         MODERATE     AUD
36030025 Australia  RBA Assistant Governor Ellis Speech         EVENT           MONEY         MODERATE     AUD

62 lines later...

76020002 Brazil  BCB Interest Rate Decision                     INDICATOR       MONEY         HIGH         BRL
76020004 Brazil  BCB Inflation Report                           EVENT           PRICES        MODERATE     BRL
76050001 Brazil  FIPE CPI m/m                                   INDICATOR       PRICES        MODERATE     BRL
76010005 Brazil  Mid-Month CPI m/m                              INDICATOR       PRICES        MODERATE     BRL
76020001 Brazil  BCB Focus Market Report                        EVENT           MONEY         MODERATE     BRL
76020003 Brazil  BCB MPC (Copom) Minutes                        EVENT           MONEY         MODERATE     BRL
76020005 Brazil  BCB National Monetary Council Meeting          EVENT           MONEY         MODERATE     BRL
76010009 Brazil  Unemployment Rate 3-months                     INDICATOR       JOBS          MODERATE     BRL
76020010 Brazil  Nominal Budget Balance                         INDICATOR       GOVERNMENT    MODERATE     BRL
76020011 Brazil  Primary Budget balance                         INDICATOR       GOVERNMENT    MODERATE     BRL
76010014 Brazil  Services Volume m/m                            INDICATOR       BUSINESS      MODERATE     BRL

98 lines later...

124040017 Canada  BoC Governor Macklem Speech                   EVENT           MONEY          HIGH         CAD
124040003 Canada  BoC Governor Poloz Speech                     EVENT           MONEY          HIGH         CAD
124040006 Canada  BoC Interest Rate Decision                    INDICATOR       MONEY          HIGH         CAD
124040009 Canada  BoC Monetary Policy Report Press Conference   EVENT           MONEY          HIGH         CAD
124010011 Canada  Employment Change                             INDICATOR       JOBS           HIGH         CAD
124010021 Canada  GDP m/m                                       INDICATOR       GDP            HIGH         CAD
124010008 Canada  Core Retail Sales m/m                         INDICATOR       CONSUMER       HIGH         CAD
124020001 Canada  Ivey PMI                                      INDICATOR       BUSINESS       HIGH         CAD
124010024 Canada  IPPI m/m                                      INDICATOR       PRICES         MODERATE     CAD
124010026 Canada  RMPI m/m                                      INDICATOR       PRICES         MODERATE     CAD
124040001 Canada  BoC Business Outlook Survey                   EVENT           MONEY          MODERATE     CAD
```

For our currencies view, we select the unique/Distinct EventCurrency and EventCode from the table MQL5Calendar.

```
//--- initializing properties for the Currencies view
   CalendarContents[6].Content = Currencies_View;
   CalendarContents[6].name = "Currencies";
   CalendarContents[6].sql = "CREATE VIEW IF NOT EXISTS Currencies AS "
                             "SELECT Distinct EventCurrency as 'Currency',EventCode as 'Code' FROM 'MQL5Calendar';";
   CalendarContents[6].tbl_name = "Currencies";
   CalendarContents[6].type = "view";
```

```
SELECT * FROM 'Currencies';
```

```
Currency  	Code
NZD  		NZ
EUR 		EU
JPY  		JP
CAD  		CA
AUD  		AU
CNY  		CN
EUR  		IT
SGD  		SG
EUR  		DE
EUR  		FR
BRL  		BR
MXN  		MX
ZAR  		ZA
HKD  		HK
INR  		IN
NOK  		NO
USD  		US
GBP  		GB
CHF  		CH
KRW  		KR
EUR  		ES
SEK  		SE
ALL  		WW
```

Now we will initialize the properties for our table MQL5Calendar that will be in our DB in memory. This table will be one big table that is essentially a combination of the tables in our DB in storage, these tables are MQL5Calendar and TimeSchedule.

```
//-- initializing properties for the MQL5Calendar table for DB in System Memory
   DBMemory.Content = MQL5Calendar_Table;
   DBMemory.name = "MQL5Calendar";
   DBMemory.sql = "CREATE TABLE IF NOT EXISTS MQL5Calendar(EVENTID  INT   NOT NULL,COUNTRY  TEXT   NOT NULL,"
                  "EVENTNAME   TEXT   NOT NULL,EVENTTYPE   TEXT   NOT NULL,EVENTIMPORTANCE   TEXT   NOT NULL,"
                  "EVENTCURRENCY  TEXT   NOT NULL,EVENTCODE   TEXT   NOT NULL,EVENTSECTOR TEXT   NOT NULL,"
                  "EVENTFORECAST  TEXT   NOT NULL,EVENTPREVALUE  TEXT   NOT NULL,EVENTIMPACT TEXT   NOT NULL,"
                  "EVENTFREQUENCY TEXT   NOT NULL,DST_UK   TEXT   NOT NULL,DST_US   TEXT   NOT NULL,"
                  "DST_AU   TEXT   NOT NULL,DST_NONE   TEXT   NOT NULL)STRICT;";
   DBMemory.tbl_name="MQL5Calendar";
   DBMemory.type = "table";
   DBMemory.insert = "INSERT INTO 'MQL5Calendar'(EVENTID,COUNTRY,EVENTNAME,EVENTTYPE,EVENTIMPORTANCE,EVENTCURRENCY,EVENTCODE,"
                     "EVENTSECTOR,EVENTFORECAST,EVENTPREVALUE,EVENTIMPACT,EVENTFREQUENCY,DST_UK,DST_US,DST_AU,DST_NONE) "
                     "VALUES (%d,'%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s', '%s', '%s');";
```

In our destructor we close the connection to the database in Memory, closing the connection to the database in memory will drop the whole database. So we only close the connection when we no longer need the database any longer.

```
//+------------------------------------------------------------------+
//|Destructor                                                        |
//+------------------------------------------------------------------+
CNews::~CNews(void)
  {
   if(FileIsExist(NEWS_TEXT_FILE,FILE_COMMON))//Check if the news database open text file exists
     {
      FileDelete(NEWS_TEXT_FILE,FILE_COMMON);
     }
   DatabaseClose(DBMemoryConnection);//Close DB in memory
  }
```

Previously in our function UpdateRecords we went through a sequence to check if we should update the calendar database in storage. The sequence went as follows:

1. We would check if the Calendar database exists in the common folder, if the database doesn't exist we perform an update.
2. We would check if all the database objects exist in the database and that their SQL statements are what we expect, if not, we perform an update.
3. We would check if the date in the Records table is equal to the current date, if not, we perform an update.

For this article, we will add one more step in the sequence. The purpose of this added step is to check if the news data in the view Calendar\_NONE is accurate. I've noticed that sometimes the news data from the MQL5 calendar can change over time and if we have stored the news that hasn't changed, we need to be able to check for inconsistences between what we have stored and what has been updated from the MQL5 calendar if anything.

In the code below we retrieve the news data for the current day using the function EconomicDetails and store this data into the array TodayNews. Once the news data is within the TodayNews array we iterate through each news event in the array and check if there is a match in our view Calendar\_NONE, if we can't find a match, then we will perform an update. If all the news data has a match in the Calendar\_NONE view we don't perform any updates to the database in storage.

```
      Calendar TodaysNews[];
      datetime Today = CTime.Time(TimeTradeServer(),0,0,0);
      EconomicDetails(TodaysNews,Today,Today+CTime.DaysS());

      for(uint i=0;i<TodaysNews.Size();i++)
        {
         request_text=StringFormat("SELECT ID FROM %s where Replace(Date,'.','-')=Replace('%s','.','-') and ID=%d;",
                                   CalendarStruct(CalendarNONE_View).name,TodaysNews[i].EventDate,TodaysNews[i].EventId);
         request=DatabasePrepare(db,request_text);//Creates a handle of a request, which can then be executed using DatabaseRead()
         if(request==INVALID_HANDLE)//Checks if the request failed to be completed
           {
            Print("DB: ",NEWS_DATABASE_FILE, " request failed with code ", GetLastError());
            DatabaseFinalize(request);
            DatabaseClose(db);
            return perform_update;
           }
         //PrintFormat(request_text);
         if(!DatabaseRead(request))//Will be true if there are results from the sql query/request
           {
            DatabaseFinalize(request);
            DatabaseClose(db);
            return perform_update;
           }
         DatabaseFinalize(request);
        }
      DatabaseClose(db);//Closes the database
      perform_update=false;
      return perform_update;
```

The code below will be responsible for creating the database in memory. Firstly, we open the connection to the database, we then drop the table MQL5Calendar if it already exists, we then create the table MQL5Calendar. Once the table is created we get all the relevant data from the function GetCalendar, we then insert all the data retrieved from the function into our MQL5Calendar table in our database in memory. Furthermore, we clear the whole array DB\_Data, and set our DST schedule in the variable MySchedule. For trading outside the strategy tester the user/trader won't be able to manually change the DST schedule, this is to prevent configuring the wrong DST schedule as configuring the DST schedule is only necessary for the strategy tester only.

```
//+------------------------------------------------------------------+
//|Create calendar database in memory                                |
//+------------------------------------------------------------------+
void CNews::CreateEconomicDatabaseMemory()
  {
//--- Open/create the database in memory
   DBMemoryConnection=DatabaseOpen(NEWS_DATABASE_MEMORY,DATABASE_OPEN_MEMORY);
   if(DBMemoryConnection==INVALID_HANDLE)//Checks if the database failed to open/create
     {
      Print("DB: ",NEWS_DATABASE_MEMORY, " open failed with code ", GetLastError());
      return;//will terminate execution of the rest of the code below
     }
//--- Drop the table if it already exists
   DatabaseExecute(DBMemoryConnection,StringFormat("Drop table IF EXISTS %s",DBMemory.name));
//--- Attempt to create the table
   if(!DatabaseExecute(DBMemoryConnection,DBMemory.sql))
     {
      Print("DB: create the Calendar table failed with code ", GetLastError());
      return;
     }
//--- Check if the table exists
   if(DatabaseTableExists(DBMemoryConnection,DBMemory.tbl_name))
     {
      //--- Get all news data and time from the database in storage
      GetCalendar(DB_Data);
      //--- Insert all the news data and times into the table
      for(uint i=0;i<DB_Data.Size();i++)
        {
         string request_text=StringFormat(DBMemory.insert,DB_Data[i].EventId,DB_Data[i].Country,
                                          DB_Data[i].EventName,DB_Data[i].EventType,DB_Data[i].EventImportance,
                                          DB_Data[i].EventCurrency,DB_Data[i].EventCode,DB_Data[i].EventSector,
                                          DB_Data[i].EventForecast,DB_Data[i].EventPreval,DB_Data[i].EventImpact,
                                          DB_Data[i].EventFrequency,DB_Data[i].DST_UK,DB_Data[i].DST_US,
                                          DB_Data[i].DST_AU,DB_Data[i].DST_NONE);

         if(!DatabaseExecute(DBMemoryConnection, request_text))//Will attempt to run this sql request/query
           {
            //--- failed to run sql request/query
            Print(GetLastError());
            PrintFormat(request_text);
            return;
           }
        }
     }
//--- Remove all data from the array
   ArrayRemove(DB_Data,0,WHOLE_ARRAY);
//--- Assign the DST schedule
   MySchedule = (MQLInfoInteger(MQL_TESTER))?(MyDST==AutoDst_Selection)?GetAutoDST():MySchedule:DST_NONE;
  }
```

Ok, now the following function called EconomicDetailsMemory retrieves all news events that take place on a specific date and then stores the news data in the array NewsTime, which is passed by reference.

```
//+------------------------------------------------------------------+
//|Gets values from the MQL5 DB Calendar in Memory                   |
//+------------------------------------------------------------------+
void CNews::EconomicDetailsMemory(Calendar &NewsTime[],datetime date)
  {
//--- SQL query to retrieve news data for a certain date
   string request_text=StringFormat("WITH MySubQuery AS (SELECT EventId as 'Id',Country,EventName as 'Name',EventType as 'Type'"
                                    ",EventImportance as 'Importance',%s as 'CTime',EventCurrency as 'Currency',EventCode as 'Code',"
                                    "EventSector as 'Sector',EventForecast as 'Forecast',EventPrevalue as 'Prevalue',EventImpact as'Impact',"
                                    "EventFrequency as 'Freq',RANK() OVER (PARTITION BY %s Order BY CASE EventPrevalue WHEN 'None' "
                                    "THEN 2 ELSE 1 END,CASE EventForecast WHEN 'None' THEN 2 ELSE 1 END,CASE EventImportance WHEN "
                                    "'CALENDAR_IMPORTANCE_HIGH' THEN 1 WHEN 'CALENDAR_IMPORTANCE_MODERATE' THEN 2 WHEN 'CALENDAR_IMPORTANCE_LOW'"
                                    " THEN 3 ELSE 4 END) Ranking FROM %s) SELECT Id,Country,Name,Type,Importance,CTime,Currency,Code,Sector,"
                                    "Forecast,Prevalue,Impact,Freq FROM MySubQuery where Date(Replace(CTime,'.','-'))=Date(Replace('%s','.','-')) and "
                                    "Ranking<2 Group by CTime;",EnumToString(MySchedule),EnumToString(MySchedule),DBMemory.name,
                                    TimeToString(date));
   int request=DatabasePrepare(DBMemoryConnection,request_text);//Creates a handle of a request, which can then be executed using DatabaseRead()
   if(request==INVALID_HANDLE)//Checks if the request failed to be completed
     {
      Print("DB: ",NEWS_DATABASE_MEMORY, " request failed with code ", GetLastError());
      PrintFormat(request_text);
     }
//--- Calendar structure variable
   Calendar ReadDB_Data;
//--- Remove any data in the array
   ArrayRemove(NewsTime,0,WHOLE_ARRAY);
   for(int i=0; DatabaseReadBind(request,ReadDB_Data); i++)//Will read all the results from the sql query/request
     {
      //--- Resize array NewsTime
      ArrayResize(NewsTime,i+1,i+2);
      //--- Assign calendar structure values into NewsTime array index
      NewsTime[i]  = ReadDB_Data;
     }
//--- Removes a request created in DatabasePrepare()
   DatabaseFinalize(request);
  }
```

Lets break down the SQL query, as it is more complex than any of our past queries. So in this query we make use of a [WITH clause](https://www.mql5.com/go?link=https://www.sqlitetutor.com/with/ "SQLite tutor") and a [RANK() function](https://www.mql5.com/go?link=https://www.sqlitetutorial.net/sqlite-window-functions/sqlite-rank/ "https://www.sqlitetutorial.net/sqlite-window-functions/sqlite-rank/").

**What is a WITH clause, and how does it work?**

The WITH clause, also known as a Common Table Expression (CTE), is used to define temporary result sets that you can reference within a SELECT, INSERT, UPDATE, or DELETE statement. The WITH clause makes complex queries easier to understand and maintain by breaking them into simpler parts. It can also help improve performance by reusing the results of expensive subqueries. CTEs are temporary and only exist for the duration of the query. They do not create any permanent objects in the database.

**What is a RANK() function, and how does it work?**

The RANK() function is used to assign a unique rank to each row within a result set based on the values in one or more columns. The rows are ordered based on the specified criteria, and the rank is assigned accordingly. The RANK() function is part of the window functions in SQL, which means it operates over a window (or subset) of rows and can return multiple rows for each row in the input set. The RANK() function assigns a rank to each row based on the order specified in the ORDER BY clause within the window function's definition. If multiple rows have the same values in the columns specified for ordering, they are given the same rank, and the next rank is skipped by the number of identical ranks. PARTITION BY (optional) splits the result set into partitions, and the RANK() function is applied to each partition independently.

In this WITH clause highlighted below we SELECT the EventId, Country, EventName, EventType, EventImportance, DST\_NONE(DST Schedule set by the user/trader in the expert's inputs), EventCurrency, EventCode, EventSector, EventForecast, EventPrevalue, EventImpact, EventFrequency, then we use a rank function the purpose of this function is to give a ranking to each SELECT result, this ranking will always be 1 for a unique event time(DST\_NONE), if we have multiple events with the same event time, then we apply the ranking, so if the event has an EventPrevalue equal to 'None' it gets a ranking of 2 else it gets a ranking of 1 and so on for the rest of the Order BY clause.

```
WITH MySubQuery AS (SELECT EventId as 'Id',Country,EventName as 'Name',EventType as 'Type',EventImportance as 'Importance',
DST_NONE as 'CTime',EventCurrency as 'Currency',EventCode as 'Code',EventSector as 'Sector',EventForecast as 'Forecast',
EventPrevalue as 'Prevalue',EventImpact as'Impact',EventFrequency as 'Freq',RANK() OVER (PARTITION BY DST_NONE
Order BY CASE EventPrevalue WHEN 'None' THEN 2 ELSE 1 END,CASE EventForecast WHEN 'None' THEN 2 ELSE 1 END,
CASE EventImportance WHEN 'CALENDAR_IMPORTANCE_HIGH' THEN 1 WHEN 'CALENDAR_IMPORTANCE_MODERATE' THEN 2
WHEN 'CALENDAR_IMPORTANCE_LOW' THEN 3 ELSE 4 END) Ranking FROM MQL5Calendar) SELECT Id,Country,Name,Type,
Importance,CTime,Currency,Code,Sector,Forecast,Prevalue,Impact,Freq FROM MySubQuery where
Date(Replace(CTime,'.','-'))=Date(Replace('2024.07.30 00:00','.','-')) and Ranking<2 Group by CTime;
```

Below is an example of the CTE(MySubQuery) results, I highlighted the events that will be eliminated from the results as their ranking exceeds 1. We only consider events with a ranking less than 2 and then group the result by their time, as when we plot the event objects on the chart we don't want to create multiple event objects with the same time, we only want the event with the most importance to be shown.

```
Id 		Country 	Name 					Type 			Importance 			CTime 		Currency Code Sector 			Forecast  Prevalue   Impact 		Freq 			  Ranking
392030007 	Japan 		Unemployment Rate 			CALENDAR_TYPE_INDICATOR CALENDAR_IMPORTANCE_MODERATE 	2024.07.30 01:30 JPY 	 JP   CALENDAR_SECTOR_JOBS 	2500000   2600000    CALENDAR_IMPACT_NA CALENDAR_FREQUENCY_MONTH   1
392050002 	Japan 		Jobs to Applicants Ratio 		CALENDAR_TYPE_INDICATOR CALENDAR_IMPORTANCE_MODERATE 	2024.07.30 01:30 JPY 	 JP   CALENDAR_SECTOR_JOBS 	1240000   1240000    CALENDAR_IMPACT_NA CALENDAR_FREQUENCY_MONTH   1
36010001 	Australia 	Building Approvals m/m 			CALENDAR_TYPE_INDICATOR CALENDAR_IMPORTANCE_MODERATE 	2024.07.30 03:30 AUD 	 AU   CALENDAR_SECTOR_BUSINESS 	-900000   5500000    CALENDAR_IMPACT_NA CALENDAR_FREQUENCY_MONTH   1
36010002 	Australia 	Private House Approvals m/m 		CALENDAR_TYPE_INDICATOR CALENDAR_IMPORTANCE_LOW 	2024.07.30 03:30 AUD 	 AU   CALENDAR_SECTOR_BUSINESS 	None 	  2100000    CALENDAR_IMPACT_NA CALENDAR_FREQUENCY_MONTH   2
250010005 	France 		GDP q/q 				CALENDAR_TYPE_INDICATOR CALENDAR_IMPORTANCE_HIGH 	2024.07.30 07:30 EUR 	 FR   CALENDAR_SECTOR_GDP 	200000 	  200000     CALENDAR_IMPACT_NA CALENDAR_FREQUENCY_QUARTER 1
250010006 	France 		GDP y/y 				CALENDAR_TYPE_INDICATOR CALENDAR_IMPORTANCE_MODERATE 	2024.07.30 07:30 EUR 	 FR   CALENDAR_SECTOR_GDP 	1000000	  1100000    CALENDAR_IMPACT_NA CALENDAR_FREQUENCY_QUARTER 2
250010008 	France 		Consumer Spending m/m 			CALENDAR_TYPE_INDICATOR CALENDAR_IMPORTANCE_MODERATE 	2024.07.30 07:30 EUR 	 FR   CALENDAR_SECTOR_CONSUMER 	800000 	  1500000    CALENDAR_IMPACT_NA CALENDAR_FREQUENCY_MONTH   2
756050001 	Switzerland 	KOF Economic Barometer 			CALENDAR_TYPE_INDICATOR CALENDAR_IMPORTANCE_MODERATE 	2024.07.30 09:00 CHF 	 CH   CALENDAR_SECTOR_BUSINESS 	101500000 102700000  CALENDAR_IMPACT_NA CALENDAR_FREQUENCY_MONTH   1
724010001 	Spain 		CPI m/m 				CALENDAR_TYPE_INDICATOR CALENDAR_IMPORTANCE_MODERATE 	2024.07.30 09:00 EUR 	 ES   CALENDAR_SECTOR_PRICES 	200000 	  400000     CALENDAR_IMPACT_NA CALENDAR_FREQUENCY_MONTH   1
724010003 	Spain 		HICP m/m 				CALENDAR_TYPE_INDICATOR CALENDAR_IMPORTANCE_MODERATE 	2024.07.30 09:00 EUR 	 ES   CALENDAR_SECTOR_PRICES 	300000 	  400000     CALENDAR_IMPACT_NA CALENDAR_FREQUENCY_MONTH   1
724010005 	Spain 		GDP q/q 				CALENDAR_TYPE_INDICATOR CALENDAR_IMPORTANCE_MODERATE 	2024.07.30 09:00 EUR 	 ES   CALENDAR_SECTOR_GDP 	300000 	  800000     CALENDAR_IMPACT_NA CALENDAR_FREQUENCY_QUARTER 1
724010002 	Spain 		CPI y/y 				CALENDAR_TYPE_INDICATOR CALENDAR_IMPORTANCE_LOW 	2024.07.30 09:00 EUR  	 ES   CALENDAR_SECTOR_PRICES 	3000000   3400000    CALENDAR_IMPACT_NA CALENDAR_FREQUENCY_MONTH   5
724010004 	Spain 		HICP y/y 				CALENDAR_TYPE_INDICATOR CALENDAR_IMPORTANCE_LOW 	2024.07.30 09:00 EUR 	 ES   CALENDAR_SECTOR_PRICES 	3500000   3600000    CALENDAR_IMPACT_NA CALENDAR_FREQUENCY_MONTH   5
724010006 	Spain 		GDP y/y 				CALENDAR_TYPE_INDICATOR CALENDAR_IMPORTANCE_LOW 	2024.07.30 09:00 EUR 	 ES   CALENDAR_SECTOR_GDP 	1900000   2500000    CALENDAR_IMPACT_NA CALENDAR_FREQUENCY_QUARTER 5
752020001 	Sweden 		Business Confidence 			CALENDAR_TYPE_INDICATOR CALENDAR_IMPORTANCE_LOW 	2024.07.30 09:00 SEK 	 SE   CALENDAR_SECTOR_BUSINESS 	95500000  97300000   CALENDAR_IMPACT_NA CALENDAR_FREQUENCY_MONTH   5
752020002 	Sweden 		Manufacturing Confidence 		CALENDAR_TYPE_INDICATOR CALENDAR_IMPORTANCE_LOW 	2024.07.30 09:00 SEK 	 SE   CALENDAR_SECTOR_BUSINESS 	99000000  99200000   CALENDAR_IMPACT_NA CALENDAR_FREQUENCY_MONTH   5
752020003 	Sweden 		Consumer Confidence 			CALENDAR_TYPE_INDICATOR CALENDAR_IMPORTANCE_LOW 	2024.07.30 09:00 SEK 	 SE   CALENDAR_SECTOR_CONSUMER 	95300000  93300000   CALENDAR_IMPACT_NA CALENDAR_FREQUENCY_MONTH   5
752020004 	Sweden 		Inflation Expectations 			CALENDAR_TYPE_INDICATOR CALENDAR_IMPORTANCE_LOW 	2024.07.30 09:00 SEK 	 SE   CALENDAR_SECTOR_CONSUMER 	6300000   6200000    CALENDAR_IMPACT_NA CALENDAR_FREQUENCY_MONTH   5
752020005 	Sweden 		Economic Tendency Indicator 		CALENDAR_TYPE_INDICATOR CALENDAR_IMPORTANCE_LOW 	2024.07.30 09:00 SEK 	 SE   CALENDAR_SECTOR_BUSINESS 	94800000  96300000   CALENDAR_IMPACT_NA CALENDAR_FREQUENCY_MONTH   5
276010008 	Germany 	GDP q/q 				CALENDAR_TYPE_INDICATOR CALENDAR_IMPORTANCE_HIGH 	2024.07.30 10:00 EUR 	 DE   CALENDAR_SECTOR_GDP 	100000 	  200000     CALENDAR_IMPACT_NA CALENDAR_FREQUENCY_QUARTER 1
380010020 	Italy 		GDP q/q 				CALENDAR_TYPE_INDICATOR CALENDAR_IMPORTANCE_MODERATE 	2024.07.30 10:00 EUR 	 IT   CALENDAR_SECTOR_GDP 	200000 	  300000     CALENDAR_IMPACT_NA CALENDAR_FREQUENCY_QUARTER 2
276010009 	Germany 	GDP y/y 				CALENDAR_TYPE_INDICATOR CALENDAR_IMPORTANCE_MODERATE 	2024.07.30 10:00 EUR 	 DE   CALENDAR_SECTOR_GDP 	-400000   -900000    CALENDAR_IMPACT_NA CALENDAR_FREQUENCY_QUARTER 2
380010021 	Italy 		GDP y/y 				CALENDAR_TYPE_INDICATOR CALENDAR_IMPORTANCE_LOW 	2024.07.30 10:00 EUR 	 IT   CALENDAR_SECTOR_GDP 	400000 	  700000     CALENDAR_IMPACT_NA CALENDAR_FREQUENCY_QUARTER 4
999030016 	European Union 	GDP q/q 				CALENDAR_TYPE_INDICATOR CALENDAR_IMPORTANCE_HIGH 	2024.07.30 11:00 EUR 	 EU   CALENDAR_SECTOR_GDP 	200000 	  300000     CALENDAR_IMPACT_NA CALENDAR_FREQUENCY_QUARTER 1
999030017 	European Union 	GDP y/y 				CALENDAR_TYPE_INDICATOR CALENDAR_IMPORTANCE_MODERATE 	2024.07.30 11:00 EUR 	 EU   CALENDAR_SECTOR_GDP 	400000 	  400000     CALENDAR_IMPACT_NA CALENDAR_FREQUENCY_QUARTER 2
999040003 	European Union 	Industrial Confidence Indicator 	CALENDAR_TYPE_INDICATOR CALENDAR_IMPORTANCE_LOW 	2024.07.30 11:00 EUR 	 EU   CALENDAR_SECTOR_BUSINESS 	-9700000  -10100000  CALENDAR_IMPACT_NA CALENDAR_FREQUENCY_MONTH   3
999040004 	European Union 	Services Sentiment Indicator 		CALENDAR_TYPE_INDICATOR CALENDAR_IMPORTANCE_LOW 	2024.07.30 11:00 EUR 	 EU   CALENDAR_SECTOR_BUSINESS 	8300000   6500000    CALENDAR_IMPACT_NA CALENDAR_FREQUENCY_MONTH   3
999040005 	European Union 	Economic Sentiment Indicator 		CALENDAR_TYPE_INDICATOR CALENDAR_IMPORTANCE_LOW 	2024.07.30 11:00 EUR 	 EU   CALENDAR_SECTOR_BUSINESS 	95300000  95900000   CALENDAR_IMPACT_NA CALENDAR_FREQUENCY_MONTH   3
999040006 	European Union 	Consumer Confidence Index 		CALENDAR_TYPE_INDICATOR CALENDAR_IMPORTANCE_LOW 	2024.07.30 11:00 EUR 	 EU   CALENDAR_SECTOR_CONSUMER 	-13000000 -13000000  CALENDAR_IMPACT_NA CALENDAR_FREQUENCY_MONTH   3
999040007 	European Union 	Consumer Price Expectations 		CALENDAR_TYPE_INDICATOR CALENDAR_IMPORTANCE_LOW 	2024.07.30 11:00 EUR  	 EU   CALENDAR_SECTOR_CONSUMER 	14400000  13100000   CALENDAR_IMPACT_NA CALENDAR_FREQUENCY_MONTH   3
999040008 	European Union 	Industry Selling Price Expectations 	CALENDAR_TYPE_INDICATOR CALENDAR_IMPORTANCE_LOW 	2024.07.30 11:00 EUR 	 EU   CALENDAR_SECTOR_BUSINESS 	6300000   6100000    CALENDAR_IMPACT_NA CALENDAR_FREQUENCY_MONTH   3
380020007 	Italy 		10-Year BTP Auction 			CALENDAR_TYPE_INDICATOR CALENDAR_IMPORTANCE_MODERATE 	2024.07.30 11:30 EUR 	 IT   CALENDAR_SECTOR_MARKET 	None 	  4010000    CALENDAR_IMPACT_NA CALENDAR_FREQUENCY_NONE    1
380020005 	Italy 		5-Year BTP Auction 			CALENDAR_TYPE_INDICATOR CALENDAR_IMPORTANCE_LOW 	2024.07.30 11:30 EUR 	 IT   CALENDAR_SECTOR_MARKET 	None 	  3550000    CALENDAR_IMPACT_NA CALENDAR_FREQUENCY_NONE    2
826130002 	United Kingdom 	10-Year Treasury Gilt Auction 		CALENDAR_TYPE_INDICATOR CALENDAR_IMPORTANCE_LOW 	2024.07.30 11:30 GBP 	 GB   CALENDAR_SECTOR_MARKET 	None 	  4371000    CALENDAR_IMPACT_NA CALENDAR_FREQUENCY_NONE    2
724080001 	Spain 		Business Confidence 			CALENDAR_TYPE_INDICATOR CALENDAR_IMPORTANCE_LOW 	2024.07.30 12:00 EUR 	 ES   CALENDAR_SECTOR_BUSINESS 	-4100000  -5700000   CALENDAR_IMPACT_NA CALENDAR_FREQUENCY_MONTH   1
76030002 	Brazil 		FGV IGP-M Inflation Index m/m 		CALENDAR_TYPE_INDICATOR CALENDAR_IMPORTANCE_LOW 	2024.07.30 13:00 BRL 	 BR   CALENDAR_SECTOR_PRICES 	1000000   810000     CALENDAR_IMPACT_NA CALENDAR_FREQUENCY_MONTH   1
484020016 	Mexico 		GDP q/q 				CALENDAR_TYPE_INDICATOR CALENDAR_IMPORTANCE_HIGH 	2024.07.30 14:00 MXN 	 MX   CALENDAR_SECTOR_GDP 	0 	  200000     CALENDAR_IMPACT_NA CALENDAR_FREQUENCY_QUARTER 1
276010020 	Germany 	CPI m/m 				CALENDAR_TYPE_INDICATOR CALENDAR_IMPORTANCE_MODERATE 	2024.07.30 14:00 EUR 	 DE   CALENDAR_SECTOR_PRICES 	0 	  100000     CALENDAR_IMPACT_NA CALENDAR_FREQUENCY_MONTH   2
276010022 	Germany 	HICP m/m 				CALENDAR_TYPE_INDICATOR CALENDAR_IMPORTANCE_MODERATE 	2024.07.30 14:00 EUR 	 DE   CALENDAR_SECTOR_PRICES 	-100000   200000     CALENDAR_IMPACT_NA CALENDAR_FREQUENCY_MONTH   2
76010012 	Brazil 		PPI m/m 				CALENDAR_TYPE_INDICATOR CALENDAR_IMPORTANCE_MODERATE 	2024.07.30 14:00 BRL 	 BR   CALENDAR_SECTOR_PRICES 	700000 	  450000     CALENDAR_IMPACT_NA CALENDAR_FREQUENCY_MONTH   2
484020017 	Mexico 		GDP n.s.a. y/y 				CALENDAR_TYPE_INDICATOR CALENDAR_IMPORTANCE_MODERATE 	2024.07.30 14:00 MXN 	 MX   CALENDAR_SECTOR_GDP 	2000000   1600000    CALENDAR_IMPACT_NA CALENDAR_FREQUENCY_QUARTER 2
276010021 	Germany 	CPI y/y 				CALENDAR_TYPE_INDICATOR CALENDAR_IMPORTANCE_LOW 	2024.07.30 14:00 EUR 	 DE   CALENDAR_SECTOR_PRICES 	2200000   2200000    CALENDAR_IMPACT_NA CALENDAR_FREQUENCY_MONTH   6
276010023 	Germany 	HICP y/y 				CALENDAR_TYPE_INDICATOR CALENDAR_IMPORTANCE_LOW 	2024.07.30 14:00 EUR 	 DE   CALENDAR_SECTOR_PRICES 	2500000   2500000    CALENDAR_IMPACT_NA CALENDAR_FREQUENCY_MONTH   6
76010013 	Brazil 		PPI y/y 				CALENDAR_TYPE_INDICATOR CALENDAR_IMPORTANCE_LOW 	2024.07.30 14:00 BRL 	 BR   CALENDAR_SECTOR_PRICES 	2200000   170000     CALENDAR_IMPACT_NA CALENDAR_FREQUENCY_MONTH   6
840170001 	United States 	S&P/CS HPI Composite-20 y/y 		CALENDAR_TYPE_INDICATOR CALENDAR_IMPORTANCE_MODERATE 	2024.07.30 15:00 USD 	 US   CALENDAR_SECTOR_HOUSING 	6800000   7200000    CALENDAR_IMPACT_NA CALENDAR_FREQUENCY_MONTH   1
840110001 	United States 	HPI m/m 				CALENDAR_TYPE_INDICATOR CALENDAR_IMPORTANCE_LOW 	2024.07.30 15:00 USD 	 US   CALENDAR_SECTOR_HOUSING 	300000 	  200000     CALENDAR_IMPACT_NA CALENDAR_FREQUENCY_MONTH   2
840110002 	United States 	HPI y/y 				CALENDAR_TYPE_INDICATOR CALENDAR_IMPORTANCE_LOW 	2024.07.30 15:00 USD 	 US   CALENDAR_SECTOR_HOUSING 	5700000   6300000    CALENDAR_IMPACT_NA CALENDAR_FREQUENCY_MONTH   2
840110003 	United States 	HPI 					CALENDAR_TYPE_INDICATOR CALENDAR_IMPORTANCE_LOW 	2024.07.30 15:00 USD 	 US   CALENDAR_SECTOR_HOUSING 	427700000 424300000  CALENDAR_IMPACT_NA CALENDAR_FREQUENCY_MONTH   2
840170002 	United States 	S&P/CS HPI Composite-20 n.s.a. m/m 	CALENDAR_TYPE_INDICATOR CALENDAR_IMPORTANCE_LOW 	2024.07.30 15:00 USD 	 US   CALENDAR_SECTOR_HOUSING 	700000 	  1400000    CALENDAR_IMPACT_NA CALENDAR_FREQUENCY_MONTH   2
840170003 	United States 	S&P/CS HPI Composite-20 s.a. m/m 	CALENDAR_TYPE_INDICATOR CALENDAR_IMPORTANCE_LOW 	2024.07.30 15:00 USD 	 US   CALENDAR_SECTOR_HOUSING 	200000 	  400000     CALENDAR_IMPACT_NA CALENDAR_FREQUENCY_MONTH   2
840030021 	United States 	JOLTS Job Openings 			CALENDAR_TYPE_INDICATOR CALENDAR_IMPORTANCE_HIGH 	2024.07.30 16:00 USD 	 US   CALENDAR_SECTOR_JOBS 	7979000   8140000    CALENDAR_IMPACT_NA CALENDAR_FREQUENCY_MONTH   1
840180002 	United States 	CB Consumer Confidence Index 		CALENDAR_TYPE_INDICATOR CALENDAR_IMPORTANCE_HIGH 	2024.07.30 16:00 USD 	 US   CALENDAR_SECTOR_CONSUMER 	108000000 100400000  CALENDAR_IMPACT_NA CALENDAR_FREQUENCY_MONTH   1
840060003 	United States 	Dallas Fed Services Revenues 		CALENDAR_TYPE_INDICATOR CALENDAR_IMPORTANCE_LOW 	2024.07.30 16:30 USD 	 US   CALENDAR_SECTOR_BUSINESS 	3900000   1900000    CALENDAR_IMPACT_NA CALENDAR_FREQUENCY_MONTH   1
840060004 	United States 	Dallas Fed Services Business Activity 	CALENDAR_TYPE_INDICATOR CALENDAR_IMPORTANCE_LOW 	2024.07.30 16:30 USD 	 US   CALENDAR_SECTOR_BUSINESS 	-6700000  -4100000   CALENDAR_IMPACT_NA CALENDAR_FREQUENCY_MONTH   1
484010001 	Mexico Fiscal 	Balance 				CALENDAR_TYPE_INDICATOR CALENDAR_IMPORTANCE_LOW 	2024.07.30 22:30 MXN 	 MX   CALENDAR_SECTOR_TRADE 	-900000   -174071000 CALENDAR_IMPACT_NA CALENDAR_FREQUENCY_MONTH   1
```

Final Results are shown below.

```
Id 		Country 	Name 					Type 			Importance 			CTime 		Currency Code Sector 			Forecast  Prevalue   Impact 		Freq 			  Ranking
392030007 	Japan 		Unemployment Rate 			CALENDAR_TYPE_INDICATOR CALENDAR_IMPORTANCE_MODERATE 	2024.07.30 01:30 JPY 	 JP   CALENDAR_SECTOR_JOBS 	2500000   2600000    CALENDAR_IMPACT_NA CALENDAR_FREQUENCY_MONTH   1
36010001 	Australia 	Building Approvals m/m 			CALENDAR_TYPE_INDICATOR CALENDAR_IMPORTANCE_MODERATE 	2024.07.30 03:30 AUD 	 AU   CALENDAR_SECTOR_BUSINESS 	-900000   5500000    CALENDAR_IMPACT_NA CALENDAR_FREQUENCY_MONTH   1
250010005 	France 		GDP q/q 				CALENDAR_TYPE_INDICATOR CALENDAR_IMPORTANCE_HIGH 	2024.07.30 07:30 EUR 	 FR   CALENDAR_SECTOR_GDP 	200000 	  200000     CALENDAR_IMPACT_NA CALENDAR_FREQUENCY_QUARTER 1
756050001 	Switzerland 	KOF Economic Barometer 			CALENDAR_TYPE_INDICATOR CALENDAR_IMPORTANCE_MODERATE 	2024.07.30 09:00 CHF 	 CH   CALENDAR_SECTOR_BUSINESS 	101500000 102700000  CALENDAR_IMPACT_NA CALENDAR_FREQUENCY_MONTH   1
276010008 	Germany 	GDP q/q 				CALENDAR_TYPE_INDICATOR CALENDAR_IMPORTANCE_HIGH 	2024.07.30 10:00 EUR 	 DE   CALENDAR_SECTOR_GDP 	100000 	  200000     CALENDAR_IMPACT_NA CALENDAR_FREQUENCY_QUARTER 1
999030016 	European Union 	GDP q/q 				CALENDAR_TYPE_INDICATOR CALENDAR_IMPORTANCE_HIGH 	2024.07.30 11:00 EUR 	 EU   CALENDAR_SECTOR_GDP 	200000 	  300000     CALENDAR_IMPACT_NA CALENDAR_FREQUENCY_QUARTER 1
380020007 	Italy 		10-Year BTP Auction 			CALENDAR_TYPE_INDICATOR CALENDAR_IMPORTANCE_MODERATE 	2024.07.30 11:30 EUR 	 IT   CALENDAR_SECTOR_MARKET 	None 	  4010000    CALENDAR_IMPACT_NA CALENDAR_FREQUENCY_NONE    1
724080001 	Spain 		Business Confidence 			CALENDAR_TYPE_INDICATOR CALENDAR_IMPORTANCE_LOW 	2024.07.30 12:00 EUR 	 ES   CALENDAR_SECTOR_BUSINESS 	-4100000  -5700000   CALENDAR_IMPACT_NA CALENDAR_FREQUENCY_MONTH   1
76030002 	Brazil 		FGV IGP-M Inflation Index m/m 		CALENDAR_TYPE_INDICATOR CALENDAR_IMPORTANCE_LOW 	2024.07.30 13:00 BRL 	 BR   CALENDAR_SECTOR_PRICES 	1000000   810000     CALENDAR_IMPACT_NA CALENDAR_FREQUENCY_MONTH   1
484020016 	Mexico 		GDP q/q 				CALENDAR_TYPE_INDICATOR CALENDAR_IMPORTANCE_HIGH 	2024.07.30 14:00 MXN 	 MX   CALENDAR_SECTOR_GDP 	0 	  200000     CALENDAR_IMPACT_NA CALENDAR_FREQUENCY_QUARTER 1
840170001 	United States 	S&P/CS HPI Composite-20 y/y 		CALENDAR_TYPE_INDICATOR CALENDAR_IMPORTANCE_MODERATE 	2024.07.30 15:00 USD 	 US   CALENDAR_SECTOR_HOUSING 	6800000   7200000    CALENDAR_IMPACT_NA CALENDAR_FREQUENCY_MONTH   1
840030021 	United States 	JOLTS Job Openings 			CALENDAR_TYPE_INDICATOR CALENDAR_IMPORTANCE_HIGH 	2024.07.30 16:00 USD 	 US   CALENDAR_SECTOR_JOBS 	7979000   8140000    CALENDAR_IMPACT_NA CALENDAR_FREQUENCY_MONTH   1
840060003 	United States 	Dallas Fed Services Revenues 		CALENDAR_TYPE_INDICATOR CALENDAR_IMPORTANCE_LOW 	2024.07.30 16:30 USD 	 US   CALENDAR_SECTOR_BUSINESS 	3900000   1900000    CALENDAR_IMPACT_NA CALENDAR_FREQUENCY_MONTH   1
484010001 	Mexico Fiscal 	Balance 				CALENDAR_TYPE_INDICATOR CALENDAR_IMPORTANCE_LOW 	2024.07.30 22:30 MXN 	 MX   CALENDAR_SECTOR_TRADE 	-900000   -174071000 CALENDAR_IMPACT_NA CALENDAR_FREQUENCY_MONTH   1
```

These result will be stored in variable below:

```
Calendar &NewsTime[]
```

The function below EconomicNextEvent is similar the previous function mentioned which is EconomicDetailsMemory, the main difference is that the function below purpose is to get the next event instead of retrieve all the events for a specific date.

```
//+------------------------------------------------------------------+
//|Will update UpcomingNews structure variable with the next event   |
//|data                                                              |
//+------------------------------------------------------------------+
void CNews::EconomicNextEvent(datetime date=0)
  {
//--- Declare unassigned Calendar structure variable Empty
   Calendar Empty;
//--- assign empty values to Calendar structure variable UpcomingNews
   UpcomingNews = Empty;
//--- If date variable is zero then assign current date
   date = (date==0)?TimeTradeServer():date;
//--- Query to retrieve next upcoming event.
   string request_text=StringFormat("WITH MySubQuery AS (SELECT EventId as 'Id',Country,EventName as 'Name',"
                                    "EventType as 'Type',EventImportance as 'Importance',%s as 'CTime',EventCurrency as 'Currency',"
                                    "EventCode as 'Code',EventSector as 'Sector',EventForecast as 'Forecast',"
                                    "EventPrevalue as 'Prevalue',EventImpact as 'Impact',EventFrequency as 'Freq',"
                                    "RANK() OVER (PARTITION BY %s ORDER BY CASE EventPrevalue WHEN 'None' THEN 2 ELSE 1 END"
                                    ", CASE EventForecast WHEN 'None' THEN 2 ELSE 1 END,CASE EventImportance "
                                    "WHEN 'CALENDAR_IMPORTANCE_HIGH' THEN 1 WHEN 'CALENDAR_IMPORTANCE_MODERATE' THEN 2 WHEN"
                                    " 'CALENDAR_IMPORTANCE_LOW' THEN 3 ELSE 4 END) Ranking FROM %s) SELECT Id,Country,Name,"
                                    "Type,Importance,CTime,Currency,Code,Sector,Forecast,Prevalue,Impact,Freq FROM MySubQuery "
                                    "where Replace(CTime,'.','-')>=Replace('%s','.','-') AND Ranking<2 Group by CTime LIMIT 1;",
                                    EnumToString(MySchedule),EnumToString(MySchedule),DBMemory.name,TimeToString(date));

   int request=DatabasePrepare(DBMemoryConnection,request_text);//Creates a handle of a request, which can then be executed using DatabaseRead()
   if(request==INVALID_HANDLE)//Checks if the request failed to be completed
     {
      Print("DB: ",NEWS_DATABASE_MEMORY, " request failed with code ", GetLastError());
      PrintFormat(request_text);
     }
//--- Assign query results to Calendar structure variable UpcomingNews
   DatabaseReadBind(request,UpcomingNews);
//--- Removes a request created in DatabasePrepare()
   DatabaseFinalize(request);
  }
```

The query below for the function EconomicNextEvent is similar to the query in the function EconomicDetailsMemory which we previously explained the only difference is that the query below will limit the results by 1.

```
WITH MySubQuery AS (SELECT EventId as 'Id',Country,EventName as 'Name',EventType as 'Type',EventImportance as 'Importance',
DST_NONE as 'CTime',EventCurrency as 'Currency',EventCode as 'Code',EventSector as 'Sector',EventForecast as 'Forecast',
EventPrevalue as 'Prevalue',EventImpact as'Impact',EventFrequency as 'Freq',RANK() OVER (PARTITION BY DST_NONE
Order BY CASE EventPrevalue WHEN 'None' THEN 2 ELSE 1 END,CASE EventForecast WHEN 'None' THEN 2 ELSE 1 END,
CASE EventImportance WHEN 'CALENDAR_IMPORTANCE_HIGH' THEN 1 WHEN 'CALENDAR_IMPORTANCE_MODERATE' THEN 2
WHEN 'CALENDAR_IMPORTANCE_LOW' THEN 3 ELSE 4 END) Ranking FROM MQL5Calendar) SELECT Id,Country,Name,Type,
Importance,CTime,Currency,Code,Sector,Forecast,Prevalue,Impact,Freq FROM MySubQuery where
Date(Replace(CTime,'.','-'))=Date(Replace('2024.07.30 00:00','.','-')) and Ranking<2 Group by CTime LIMIT 1;
```

Sample results for the query above is shown below.

```
Id 		Country 	Name 					Type 			Importance 			CTime 		Currency Code Sector 			Forecast  Prevalue   Impact 		Freq 			  Ranking
392030007 	Japan 		Unemployment Rate 			CALENDAR_TYPE_INDICATOR CALENDAR_IMPORTANCE_MODERATE 	2024.07.30 01:30 JPY 	 JP   CALENDAR_SECTOR_JOBS 	2500000   2600000    CALENDAR_IMPACT_NA CALENDAR_FREQUENCY_MONTH   1
```

The sample result above will be stored into the variable we declared outside the news class called UpcomingNews.

```
//--- Assign query results to Calendar structure variable UpcomingNews
   DatabaseReadBind(request,UpcomingNews);
```

The final newly added function in the news class from part 2 is called GetImpact which is shown below. The purpose for this function is to return the enumeration value for the enumeration called ENUM\_CALENDAR\_EVENT\_IMPACT for the upcoming event based of previous events with similar properties or values to the upcoming event. Let me explain further.

```
//+------------------------------------------------------------------+
//|Will retrieve Upcoming Event Impact data                          |
//+------------------------------------------------------------------+
ENUM_CALENDAR_EVENT_IMPACT CNews::GetImpact()
  {
//--- Declaration of string variable
   string impact=NULL;
//--- Query to get impact data from previous event with the same event id and matching EventPrevalue and EventForecast scenarios.
   string request_text=StringFormat("SELECT EventImpact FROM %s where Replace(%s,'.','-')<Replace('%s','.','-') AND EventId=%d"
                                    " %s ORDER BY %s DESC LIMIT 1;",DBMemory.name,EnumToString(MySchedule),UpcomingNews.EventDate,
                                    UpcomingNews.EventId,((UpcomingNews.EventPreval=="None"||UpcomingNews.EventForecast=="None")?
                                          "AND EventImpact='CALENDAR_IMPACT_NA'":(int(UpcomingNews.EventPreval)<int(UpcomingNews.EventForecast))?
                                          "AND EventPrevalue<EventForecast AND EventImpact<>'CALENDAR_IMPACT_NA'":
                                          (int(UpcomingNews.EventPreval)>int(UpcomingNews.EventForecast))?
                                          "AND EventPrevalue>EventForecast AND EventImpact<>'CALENDAR_IMPACT_NA'":
                                          "AND EventPrevalue=EventForecast AND EventImpact<>'CALENDAR_IMPACT_NA'"),
                                    EnumToString(MySchedule));

   int request=DatabasePrepare(DBMemoryConnection,request_text);//Creates a handle of a request, which can then be executed using DatabaseRead()

   if(request==INVALID_HANDLE)//Checks if the request failed to be completed
     {
      Print("DB: ",NEWS_DATABASE_MEMORY, " request failed with code ", GetLastError());
      PrintFormat(request_text);
     }
   if(DatabaseRead(request))//Will read the one record in the 'Record' table
     {
      //--- assign first column result into impact variable
      DatabaseColumnText(request,0,impact);
     }
//--- Removes a request created in DatabasePrepare()
   DatabaseFinalize(request);
//--- Return equivalent Event impact in the enumeration ENUM_CALENDAR_EVENT_IMPACT
   if(impact=="CALENDAR_IMPACT_POSITIVE")
     {
      return CALENDAR_IMPACT_POSITIVE;
     }
   else
      if(impact=="CALENDAR_IMPACT_NEGATIVE")
        {
         return CALENDAR_IMPACT_NEGATIVE;
        }
      else
        {
         return CALENDAR_IMPACT_NA;
        }
  }
```

According to my observations, the event impact basically works in such a way that upcoming events or events that occur on the current day all had the enumeration value CALENDAR\_IMPACT\_NA of the impact, which means that the impact is unavailable for the event. The event impact is only updated after the day of the event has passed and only if the event had both the event previous value and event forecast values available before the event occurred, there are some instances where the event impact will still be registered as unavailable even when all these requirements are meet.

**Why do we need the previous event impact value, and how do we make use of it?**

Firstly, the reason we require the previous event impact is to predict what the next event's impact outcome could be. So when the upcoming event's previous value is not 'None' and as well as the forecast value, we will then check if these values are equal  or which value is greater than the other. Once we found out which value is greater than the other or if the values are the same, we then look for previous events that had the same configuration and retrieve the event impact for that previous event to predict what the upcoming event's impact could be. Ex. if the upcoming event's previous value is 3000 and the forecast is 10,000, we look for the last event with the same event ID that had a previous value less than the forecast value and use this event's impact for the upcoming event.

What is the event impact, and what is the impact based of?

The event impact is the measurement of the perceived effect a specific event had on a currency, the impact is based of the event's currency. So if the unemployment rate is the upcoming event and the previous event with the similar previous value and forecast value configuration has an event impact of CALENDAR\_IMPACT\_NEGATIVE and the event currency is USD, this essentially means that the US dollar was negatively affected in the previous unemployment rate, so if you were on the symbol EURUSD right after the previous unemployment rate occurred we would ideally see the EURUSD strengthen meaning EUR would've gained value against the US dollar.

In the query below we SELECT the event impact from the MQL5Calendar table in the database in memory, in the WHERE clause we look for event dates before the upcoming event date and filter for the same event ID. We then filter for the same relationship scenario between the event previous value(EventPreval) and the event forecast value(EventForecast), if the UpcomingNews.EventPreval is equal to 'None' or UpcomingNews.EventForecast is equal to 'None' then we already know that the event impact will be 'None',if the UpcomingNews.EventPreval is less than UpcomingNews. EventForecast then we look for previous events where EventPrevalue<EventForecast and the event impact is not NA, if the UpcomingNews.EventPreval is more than UpcomingNews.EventForecast then we look for previous events where EventPreval>EventForecast and the event impact is not NA, else we look for where the EventPrevalue=EventForecast and the event impact is not NA. All results are filtered for the most recent date as the ORDER BY clause is in descending order for the event date and the LIMIT clause will allow for only one record to be returned/outputted.

```
//--- Query to get impact data from previous event with the same event id and matching EventPrevalue and EventForecast scenarios.
   string request_text=StringFormat("SELECT EventImpact FROM %s where Replace(%s,'.','-')<Replace('%s','.','-') AND EventId=%d"
                                    " %s ORDER BY %s DESC LIMIT 1;",DBMemory.name,EnumToString(MySchedule),UpcomingNews.EventDate,
                                    UpcomingNews.EventId,((UpcomingNews.EventPreval=="None"||UpcomingNews.EventForecast=="None")?
                                          "AND EventImpact='CALENDAR_IMPACT_NA'":(int(UpcomingNews.EventPreval)<int(UpcomingNews.EventForecast))?
                                          "AND EventPrevalue<EventForecast AND EventImpact<>'CALENDAR_IMPACT_NA'":
                                          (int(UpcomingNews.EventPreval)>int(UpcomingNews.EventForecast))?
                                          "AND EventPrevalue>EventForecast AND EventImpact<>'CALENDAR_IMPACT_NA'":
                                          "AND EventPrevalue=EventForecast AND EventImpact<>'CALENDAR_IMPACT_NA'"),
                                    EnumToString(MySchedule));
```

An example of how the query above looks when all the variables are filled with data. So as you can see below, we SELECT the EventImpact FROM MQL5Calendar WHERE DST\_NONE is less than '2024.08.01 16:30' and the EventId is equal to 124500001 and EventPrevalue is less than EventForecast and EventImpact is not equal to 'CALENDAR\_IMPACT\_NA' we then set the results to descending order in relation to DST\_NONE and finally limit the results to one result.

```
SELECT EventImpact FROM MQL5Calendar where Replace(DST_NONE,'.','-')<Replace('2024.08.01 16:30','.','-') AND EventId=124500001
AND EventPrevalue<EventForecast AND EventImpact<>'CALENDAR_IMPACT_NA' ORDER BY DST_NONE DESC LIMIT 1;
```

### Common Graphics Class

This class is responsible for showing the visual elements of the expert on the chart, so by now you have noticed that the visuals for Part 3 are a bigger step up from Part 2 and so will be the code that creates these visual elements. Lets take a look at the code. So this class will need access to the news information, risk management information, account information, so these classes are included.

```
#include "ObjectProperties.mqh"
#include "RiskManagement.mqh"
#include "CommonVariables.mqh"
#include "News.mqh"
//+------------------------------------------------------------------+
//|CommonGraphics class                                              |
//+------------------------------------------------------------------+
class CCommonGraphics:CObjectProperties
  {
private:
   //--- GraphicText structure this structure is responsible for managing the graphical text
   struct GraphicText
     {
      //--- private declaration for struct GraphicText
   private:
      //--- this structure will store properties for the subtext
      struct subtextformat
        {
         string      Label;//Store text label
         string      Text;//Store text value
        };
      //--- this structure inherits from subtextformat and is responsible for finding text
      struct found:subtextformat
        {
         bool        isFound;//Check if text is found
         int         index;//Get index for the text
        };
      //--- structure array for subtexts
      subtextformat  sub_text[];
      //--- function to find text properties from text's label
      found          FoundText(string label)
        {
         found find;
         find.Label="";
         find.Text="";
         find.isFound=false;
         find.index=-1;
         for(uint i=0;i<sub_text.Size();i++)
           {
            //--- If text label is found in array
            if(label==sub_text[i].Label)
              {
               //--- Assign text properties
               find.Label=sub_text[i].Label;
               find.Text=sub_text[i].Text;
               find.isFound=true;
               find.index=int(i);
               return find;//return found text properties
              }
           }
         return find;//return text properties
        }
      //--- public declaration for struct GraphicText
   public:
      //--- string variable
      string         text;
      //--- function to set/add text properties
      void           subtext(string label,string value)
        {
         //--- Get text properties from label
         found result = FoundText(label);
         //--- Check if text label was found/exists in array sub_text
         if(!result.isFound)
           {
            //--- Resize array sub_text
            ArrayResize(sub_text,sub_text.Size()+1,sub_text.Size()+2);
            //--- Add text properties for new array index
            sub_text[sub_text.Size()-1].Label = label;
            sub_text[sub_text.Size()-1].Text = value;
           }
         else
           {
            /* Set new text/override text from text label that exists
            in the array sub_text array */
            sub_text[result.index].Text = value;
           }
        }
      //--- function to retrieve text from text label
      string         subtext(string label)
        {
         return FoundText(label).Text;
        }
     };// End of struct GraphicText

   //--- AccountInfo object declaration
   CAccountInfo      CAccount;
   //--- News object declaration
   CNews             NewsObj;
   //--- TimeManagement object declaration
   CTimeManagement   CTime;
   //--- Calendar structure array declaration
   Calendar          CalendarArray[];
   //--- color variable declaration
   color             EventColor;
   //--- unit variable declarations
   uint              Fontsize,X_start,Y_start;
   //--- void function declarations for Graphical blocks
   void              Block_1();
   void              Block_2(uint SecondsPreEvent=5);
   CRiskManagement   CRisk;//Risk management class object
   //--- GraphicText structure array declarations
   GraphicText       Texts_Block1[9],Texts_Block2[7];
   //--- structure to store text height and width
   struct Text_Prop_Size
     {
      uint           Height;//store text height
      uint           Width;//store text width
     };

   //--- void function to retrieve sum of the texts height and the maximum width of texts from GraphicText array Texts
   void              GetTextMaxWidthAndHeight(GraphicText &Texts[],uint &Max_Height,uint &Max_Width,uint FontSize)
     {
      //--- set fontsize properties to get accurate text height and width sizes
      TextSetFont("Arial",(-1*FontSize)-100);
      //--- set variables to default value of zero
      Max_Height=0;
      Max_Width=0;
      //--- loop through all texts in the GraphicText array Texts
      for(uint i=0;i<Texts.Size();i++)
        {
         //--- temporary declarations for height and width
         uint Height=0,Width=0;
         //--- retrieve text height and width from index in Texts array
         TextGetSize(Texts[i].text,Width,Height);
         //--- sum texts height to variable Max_Height
         Max_Height+=Height;
         //--- assign width if text width is more than variable Max_Width value
         Max_Width=(Width>Max_Width)?Width:Max_Width;
        }
     }
   //--- function to retrieve text height and width properties in the structure Text_Prop_Size format
   Text_Prop_Size    GetText(string Text,uint FontSize)
     {
      //--- structure Text_Prop_Size variable
      Text_Prop_Size Size;
      //--- set fontsize properties to get accurate text height and width sizes
      TextSetFont("Arial",(-1*FontSize)-100);
      //--- retrieve text height and width from Text string variable
      TextGetSize(Text,Size.Width,Size.Height);
      //--- return structure Text_Prop_Size variable
      return Size;
     }
   //--- Function to get texts height sum and max width in the structure Text_Prop_Size format
   Text_Prop_Size    GetTextMax(GraphicText &Texts[],uint FontSize)
     {
      //--- structure Text_Prop_Size variable
      Text_Prop_Size Size;
      //--- uint variable declarations for text properties
      uint Max_Height;
      uint Max_Width;
      //--- Retrieve sum of texts height and maximum texts width into Max_Height,Max_Width
      GetTextMaxWidthAndHeight(Texts,Max_Height,Max_Width,FontSize);
      //--- assign values into structure Text_Prop_Size variable
      Size.Height = Max_Height;
      Size.Width = Max_Width;
      //--- return structure Text_Prop_Size variable
      return Size;
     }
   //--- Boolean declarations
   bool              is_date,is_spread,is_news,is_events;

public:
   //--- class constructor
                     CCommonGraphics(bool display_date,bool display_spread,bool display_news,bool display_events);
                    ~CCommonGraphics(void) {}//class destructor
   void              GraphicsRefresh(uint SecondsPreEvent=5);//will create/refresh the chart objects
   //--- will update certain graphics at an interval
   void              Block_2_Realtime(uint SecondsPreEvent=5);
   //--- will create chart event objects
   void              NewsEvent();
  };
```

The structure below called GraphicText is not your ordinary structure in MQL5, this structure contains private and/or public declarations of structures, structure arrays, functions and a string variable. In this case, the structure GraphicText behaves like a whole separate class. This structures purpose is to manage and store graphical texts, the structure array sub\_text will store all the text properties and the function FoundText will search for any text label match within the sub\_text array, if there is a match we will return the details of this match within the structure called found which inherits from the structure subtextformat. In the public declarations the string variable called text will store the full length of the text in the sub\_text structure array. The void function subtext will add the text label and text/subtext into the array sub\_text or will override the text/subtext stored in the array if the string variable label is found within the array. The string function called subtext will retrieve the text for the label associated with the labels in the sub\_text structure array.

```
   //--- GraphicText structure this structure is responsible for managing the graphical text
   struct GraphicText
     {
      //--- private declaration for struct GraphicText
   private:
      //--- this structure will store properties for the subtext
      struct subtextformat
        {
         string      Label;//Store text label
         string      Text;//Store text value
        };
      //--- this structure inherits from subtextformat and is responsible for finding text
      struct found:subtextformat
        {
         bool        isFound;//Check if text is found
         int         index;//Get index for the text
        };
      //--- structure array for subtexts
      subtextformat  sub_text[];
      //--- function to find text properties from text's label
      found          FoundText(string label)
        {
         found find;
         find.Label="";
         find.Text="";
         find.isFound=false;
         find.index=-1;
         for(uint i=0;i<sub_text.Size();i++)
           {
            //--- If text label is found in array
            if(label==sub_text[i].Label)
              {
               //--- Assign text properties
               find.Label=sub_text[i].Label;
               find.Text=sub_text[i].Text;
               find.isFound=true;
               find.index=int(i);
               return find;//return found text properties
              }
           }
         return find;//return text properties
        }
      //--- public declaration for struct GraphicText
   public:
      //--- string variable
      string         text;
      //--- function to set/add text properties
      void           subtext(string label,string value)
        {
         //--- Get text properties from label
         found result = FoundText(label);
         //--- Check if text label was found/exists in array sub_text
         if(!result.isFound)
           {
            //--- Resize array sub_text
            ArrayResize(sub_text,sub_text.Size()+1,sub_text.Size()+2);
            //--- Add text properties for new array index
            sub_text[sub_text.Size()-1].Label = label;
            sub_text[sub_text.Size()-1].Text = value;
           }
         else
           {
            /* Set new text/override text from text label that exists
            in the array sub_text array */
            sub_text[result.index].Text = value;
           }
        }
      //--- function to retrieve text from text label
      string         subtext(string label)
        {
         return FoundText(label).Text;
        }
     };// End of struct GraphicText
```

The function below is created to retrieve the sum of text heights in the structure array Texts as well as the maximum width from the texts in the array.

```
 //--- void function to retrieve sum of the texts height and the maximum width of texts from GraphicText array Texts
   void              GetTextMaxWidthAndHeight(GraphicText &Texts[],uint &Max_Height,uint &Max_Width,uint FontSize)
     {
      //--- set fontsize properties to get accurate text height and width sizes
      TextSetFont("Arial",(-1*FontSize)-100);
      //--- set variables to default value of zero
      Max_Height=0;
      Max_Width=0;
      //--- loop through all texts in the GraphicText array Texts
      for(uint i=0;i<Texts.Size();i++)
        {
         //--- temporary declarations for height and width
         uint Height=0,Width=0;
         //--- retrieve text height and width from index in Texts array
         TextGetSize(Texts[i].text,Width,Height);
         //--- sum texts height to variable Max_Height
         Max_Height+=Height;
         //--- assign width if text width is more than variable Max_Width value
         Max_Width=(Width>Max_Width)?Width:Max_Width;
        }
     }
```

The function below is created to retrieve the height and width properties in the structure Text\_Prop\_Size format from the string variable Text.

```
//--- function to retrieve text height and width properties in the structure Text_Prop_Size format
   Text_Prop_Size    GetText(string Text,uint FontSize)
     {
      //--- structure Text_Prop_Size variable
      Text_Prop_Size Size;
      //--- set fontsize properties to get accurate text height and width sizes
      TextSetFont("Arial",(-1*FontSize)-100);
      //--- retrieve text height and width from Text string variable
      TextGetSize(Text,Size.Width,Size.Height);
      //--- return structure Text_Prop_Size variable
      return Size;
     }
```

The function below is created to retrieve the sum of heights and maximum width properties in the structure Text\_Prop\_Size format from the structure array Texts.

```
//--- Function to get texts height sum and max width in the structure Text_Prop_Size format
   Text_Prop_Size    GetTextMax(GraphicText &Texts[],uint FontSize)
     {
      //--- structure Text_Prop_Size variable
      Text_Prop_Size Size;
      //--- uint variable declarations for text properties
      uint Max_Height;
      uint Max_Width;
      //--- Retrieve sum of texts height and maximum texts width into Max_Height,Max_Width
      GetTextMaxWidthAndHeight(Texts,Max_Height,Max_Width,FontSize);
      //--- assign values into structure Text_Prop_Size variable
      Size.Height = Max_Height;
      Size.Width = Max_Width;
      //--- return structure Text_Prop_Size variable
      return Size;
     }
```

In the class's constructor we will initialize our previously declared variables which are is\_date(which will be used to decide whether to display the date information on the chart), is\_spread(which will be used to decide whether to display the spread information on the chart), is\_news(which will be used to decide whether to display the news information on the chart) and is\_events(which will be used to decide whether to display the event object information on the chart). NewsObject. EconomicNextEvent funtion will update the UpcomingNews variable with the next news details.

```
//+------------------------------------------------------------------+
//|Constructor                                                       |
//+------------------------------------------------------------------+
CCommonGraphics::CCommonGraphics(bool display_date,bool display_spread,bool display_news,bool display_events):
//--- Assign variables
   is_date(display_date),is_spread(display_spread),is_news(display_news),is_events(display_events)
  {
//--- get next news event
   NewsObject.EconomicNextEvent();
  }
```

GraphicsRefresh will be responsible for first clearing the chart of objects previously created if any, then will be calling other functions that will create chart objects to display various information on the current chart.

```
//+------------------------------------------------------------------+
//|will create/refresh the chart objects                             |
//+------------------------------------------------------------------+
void CCommonGraphics::GraphicsRefresh(uint SecondsPreEvent=5)
  {
//--- create graphics if outside the strategy tester or in the strategy tester and visual mode is enabled
   if((!MQLInfoInteger(MQL_TESTER))||(MQLInfoInteger(MQL_TESTER)&&MQLInfoInteger(MQL_VISUAL_MODE)))
     {
      //--- Delete chart objects
      DeleteObj();//function from Object properties class
      Block_1();//Create graphics for block 1
      //--- Check whether to create graphics for block 2
      if(is_date||is_news||is_spread)
        {
         Block_2(SecondsPreEvent);//Create graphics for block 2
        }
      //--- creates event objects
      NewsEvent();
     }
  }
```

The function below called Block\_1 will be responsible for creating the graphical elements for the graphical block 1 which is indicated below and consists of:

- Symbol Name
- Symbol Period
- Symbol Description
- Symbol Contract size
- Symbol Minimum Lot-size
- Symbol Maximum Lot-size
- Symbol Volume Step
- Symbol Volume Limit
- Risk Option
- Risk Floor
- Risk Ceiling

> ![Graphical block 1](https://c.mql5.com/2/87/GBPUSDM1_Block1.png)

```
//+------------------------------------------------------------------+
//|Graphical Block 1                                                 |
//+------------------------------------------------------------------+
void CCommonGraphics::Block_1()
  {
//--- Set text object color depending if the chart color mode is LightMode or not
   TextObj_color = (isLightMode)?clrBlack:clrWheat;
//--- Set text properties for Symbol name,Symbol period and Symbol description # section 1
   Texts_Block1[0].text = Symbol()+", "+GetChartPeriodName()+": "+CSymbol.Description();//set main text
   Texts_Block1[0].subtext("Symbol Name",Symbol()+",");//set subtext - label,value
   Texts_Block1[0].subtext("Symbol Period",GetChartPeriodName());//set subtext - label,value
   Texts_Block1[0].subtext("Symbol Desc",": "+CSymbol.Description());//set subtext - label,value
//--- Set text properties for Contract size # section 2
   Texts_Block1[1].text = "Contract Size: "+string(CSymbol.ContractSize());//set main text
   Texts_Block1[1].subtext("Contract Size Text","Contract Size:");//set subtext - label,value
   Texts_Block1[1].subtext("Contract Size",string(CSymbol.ContractSize()));//set subtext - label,value
//--- Set text properties for Minimum lot # section 3
   Texts_Block1[2].text = "Minimum Lot: "+string(CSymbol.LotsMin());//set main text
   Texts_Block1[2].subtext("Minimum Lot Text","Minimum Lot:");//set subtext - label,value
   Texts_Block1[2].subtext("Minimum Lot",string(CSymbol.LotsMin()));//set subtext - label,value
//--- Set text properties for Max lot # section 4
   Texts_Block1[3].text = "Max Lot: "+string(CSymbol.LotsMax());//set main text
   Texts_Block1[3].subtext("Max Lot Text","Max Lot:");//set subtext - label,value
   Texts_Block1[3].subtext("Max Lot",string(CSymbol.LotsMax()));//set subtext - label,value
//--- Set text properties for Volume step # section 5
   Texts_Block1[4].text = "Volume Step: "+string(CSymbol.LotsStep());//set main text
   Texts_Block1[4].subtext("Volume Step Text","Volume Step:");//set subtext - label,value
   Texts_Block1[4].subtext("Volume Step",string(CSymbol.LotsStep()));//set subtext - label,value
//--- Set text properties for Volume limit # section 6
   Texts_Block1[5].text = "Volume Limit: "+string(CSymbol.LotsLimit());//set main text
   Texts_Block1[5].subtext("Volume Limit Text","Volume Limit:");//set subtext - label,value
   Texts_Block1[5].subtext("Volume Limit",string(CSymbol.LotsLimit()));//set subtext - label,value
//--- Set text properties for Risk option # section 7
   Texts_Block1[6].text = "Risk Option: "+CRisk.GetRiskOption();//set main text
   Texts_Block1[6].subtext("Risk Option Text","Risk Option:");//set subtext - label,value
   Texts_Block1[6].subtext("Risk Option",CRisk.GetRiskOption());//set subtext - label,value
//--- Set text properties for Risk floor # section 8
   Texts_Block1[7].text = "Risk Floor: "+CRisk.GetRiskFloor();//set main text
   Texts_Block1[7].subtext("Risk Floor Text","Risk Floor:");//set subtext - label,value
   Texts_Block1[7].subtext("Risk Floor",CRisk.GetRiskFloor());//set subtext - label,value
//--- Set text properties for Risk ceiling # section 9
   Texts_Block1[8].text = "Risk Ceiling: "+CRisk.GetRiskCeil();//set main text
   Texts_Block1[8].subtext("Risk Ceiling Text","Risk Ceiling:");//set subtext - label,value
   Texts_Block1[8].subtext("Risk Ceiling",CRisk.GetRiskCeil());//set subtext - label,value

//--- Set basic properties
   Fontsize=10;//Set Fontsize
   X_start=2;//Set X distance
   Y_start=2;//Set Y distance

/* Create objects # section 1*/
//-- Create the background object for width and height+3 of section 1 text
   Square(0,"Symbol Name background",X_start,Y_start,GetText(Texts_Block1[0].text,Fontsize).Width,
          GetText(Texts_Block1[0].text,Fontsize).Height+3,ANCHOR_LEFT_UPPER);
   Y_start+=3;//Re-adjust Y distance
   X_start+=2;//Re-adjust X distance
//-- Will create the text objects for section 1
   TextObj(0,"Symbol Name",Texts_Block1[0].subtext("Symbol Name"),X_start,Y_start,CORNER_LEFT_UPPER,Fontsize);
   X_start+=GetText(Texts_Block1[0].subtext("Symbol Name"),Fontsize).Width;//Re-adjust X distance
   TextObj(0,"Symbol Period",Texts_Block1[0].subtext("Symbol Period"),X_start,Y_start,CORNER_LEFT_UPPER,Fontsize);
   X_start+=GetText(Texts_Block1[0].subtext("Symbol Period"),Fontsize).Width;//Re-adjust X distance
   TextObj(0,"Symbol Desc",Texts_Block1[0].subtext("Symbol Desc"),X_start,Y_start,CORNER_LEFT_UPPER,Fontsize);
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Create objects # section 2*/
   X_start=2;//Reset X distance
   Y_start+=GetText(Texts_Block1[0].text,Fontsize).Height;//Re-adjust Y distance, add height from section 1
   //-- Create the background object for width and height+3 of section 2 text
   Square(0,"Symbol Contract Size background",X_start,Y_start,GetText(Texts_Block1[1].text,Fontsize).Width,
          GetText(Texts_Block1[1].text,Fontsize).Height+3,ANCHOR_LEFT_UPPER);
   Y_start+=3;//Re-adjust Y distance
   X_start+=2;//Re-adjust X distance
//-- Will create the text objects for section 2
   TextObj(0,"Symbol Contract Size Text",Texts_Block1[1].subtext("Contract Size Text"),X_start,Y_start,CORNER_LEFT_UPPER,Fontsize);
   X_start+=GetText(Texts_Block1[1].subtext("Contract Size Text"),Fontsize).Width;//Re-adjust X distance
   TextObj(0,"Symbol Contract Size",Texts_Block1[1].subtext("Contract Size"),X_start,Y_start,CORNER_LEFT_UPPER,Fontsize);
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Create objects # section 3*/
   X_start=2;//Reset X distance
   Y_start+=GetText(Texts_Block1[1].text,Fontsize).Height;//Re-adjust Y distance, add height from section 2
   //-- Create the background object for width and height+3 of section 3 text
   Square(0,"Symbol MinLot background",X_start,Y_start,GetText(Texts_Block1[2].text,Fontsize).Width,
          GetText(Texts_Block1[2].text,Fontsize).Height+3,ANCHOR_LEFT_UPPER);
   Y_start+=3;//Re-adjust Y distance
   X_start+=2;//Re-adjust X distance
//-- Will create the text objects for section 3
   TextObj(0,"Symbol MinLot Text",Texts_Block1[2].subtext("Minimum Lot Text"),X_start,Y_start,CORNER_LEFT_UPPER,Fontsize);
   X_start+=GetText(Texts_Block1[2].subtext("Minimum Lot Text"),Fontsize).Width;//Re-adjust X distance
   TextObj(0,"Symbol MinLot",Texts_Block1[2].subtext("Minimum Lot"),X_start,Y_start,CORNER_LEFT_UPPER,Fontsize);
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Create objects # section 4*/
   X_start=2;//Reset X distance
   Y_start+=GetText(Texts_Block1[2].text,Fontsize).Height;//Re-adjust Y distance, add height from section 3
   //-- Create the background object for width and height+3 of section 4 text
   Square(0,"Symbol MaxLot background",X_start,Y_start,GetText(Texts_Block1[3].text,Fontsize).Width,
          GetText(Texts_Block1[3].text,Fontsize).Height+3,ANCHOR_LEFT_UPPER);
   Y_start+=3;//Re-adjust Y distance
   X_start+=2;//Re-adjust X distance
//-- Will create the text objects for section 4
   TextObj(0,"Symbol MaxLot Text",Texts_Block1[3].subtext("Max Lot Text"),X_start,Y_start,CORNER_LEFT_UPPER,Fontsize);
   X_start+=GetText(Texts_Block1[3].subtext("Max Lot Text"),Fontsize).Width;//Re-adjust X distance
   TextObj(0,"Symbol MaxLot",Texts_Block1[3].subtext("Max Lot"),X_start,Y_start,CORNER_LEFT_UPPER,Fontsize);
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Create objects # section 5*/
   X_start=2;//Reset X distance
   Y_start+=GetText(Texts_Block1[3].text,Fontsize).Height;//Re-adjust Y distance, add height from section 4
   //-- Create the background object for width and height+3 of section 5 text
   Square(0,"Symbol Volume Step background",X_start,Y_start,GetText(Texts_Block1[4].text,Fontsize).Width,
          GetText(Texts_Block1[4].text,Fontsize).Height+3,ANCHOR_LEFT_UPPER);
   Y_start+=3;//Re-adjust Y distance
   X_start+=2;//Re-adjust X distance
//-- Will create the text objects for section 5
   TextObj(0,"Symbol Volume Step Text",Texts_Block1[4].subtext("Volume Step Text"),X_start,Y_start,CORNER_LEFT_UPPER,Fontsize);
   X_start+=GetText(Texts_Block1[4].subtext("Volume Step Text"),Fontsize).Width;//Re-adjust X distance
   TextObj(0,"Symbol Volume Step",Texts_Block1[4].subtext("Volume Step"),X_start,Y_start,CORNER_LEFT_UPPER,Fontsize);
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Create objects # section 6*/
   X_start=2;//Reset X distance
   Y_start+=GetText(Texts_Block1[4].text,Fontsize).Height;//Re-adjust Y distance, add height from section 5
   //-- Create the background object for width and height+3 of section 6 text
   Square(0,"Symbol Volume Limit background",X_start,Y_start,GetText(Texts_Block1[5].text,Fontsize).Width,
          GetText(Texts_Block1[5].text,Fontsize).Height+3,ANCHOR_LEFT_UPPER);
   Y_start+=3;//Re-adjust Y distance
   X_start+=2;//Re-adjust X distance
//-- Will create the text objects for section 6
   TextObj(0,"Symbol Volume Limit Text",Texts_Block1[5].subtext("Volume Limit Text"),X_start,Y_start,CORNER_LEFT_UPPER,Fontsize);
   X_start+=GetText(Texts_Block1[5].subtext("Volume Limit Text"),Fontsize).Width;//Re-adjust X distance
   TextObj(0,"Symbol Volume Limit",Texts_Block1[5].subtext("Volume Limit"),X_start,Y_start,CORNER_LEFT_UPPER,Fontsize);
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Create objects # section 7*/
   X_start=2;//Reset X distance
   Y_start+=GetText(Texts_Block1[5].text,Fontsize).Height;//Re-adjust Y distance, add height from section 6
   //-- Create the background object for width and height+3 of section 7 text
   Square(0,"Risk Option background",X_start,Y_start,GetText(Texts_Block1[6].text,Fontsize).Width,
          GetText(Texts_Block1[6].text,Fontsize).Height+3,ANCHOR_LEFT_UPPER);
   Y_start+=3;//Re-adjust Y distance
   X_start+=2;//Re-adjust X distance
//-- Will create the text objects for section 7
   TextObj(0,"Risk Option Text",Texts_Block1[6].subtext("Risk Option Text"),X_start,Y_start,CORNER_LEFT_UPPER,Fontsize);
   X_start+=GetText(Texts_Block1[6].subtext("Risk Option Text"),Fontsize).Width;//Re-adjust X distance
   TextObj(0,"Risk Option",Texts_Block1[6].subtext("Risk Option"),X_start,Y_start,CORNER_LEFT_UPPER,Fontsize);
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Create objects # section 8*/
   X_start=2;//Reset X distance
   Y_start+=GetText(Texts_Block1[6].text,Fontsize).Height;//Re-adjust Y distance, add height from section 7
   //-- Create the background object for width and height+3 of section 8 text
   Square(0,"Risk Floor background",X_start,Y_start,GetText(Texts_Block1[7].text,Fontsize).Width,
          GetText(Texts_Block1[7].text,Fontsize).Height+3,ANCHOR_LEFT_UPPER);
   Y_start+=3;//Re-adjust Y distance
   X_start+=2;//Re-adjust X distance
//-- Will create the text objects for section 8
   TextObj(0,"Risk Floor Text",Texts_Block1[7].subtext("Risk Floor Text"),X_start,Y_start,CORNER_LEFT_UPPER,Fontsize);
   X_start+=GetText(Texts_Block1[7].subtext("Risk Floor Text"),Fontsize).Width;//Re-adjust X distance
   TextObj(0,"Risk Floor",Texts_Block1[7].subtext("Risk Floor"),X_start,Y_start,CORNER_LEFT_UPPER,Fontsize);
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Create objects # section 9*/
   X_start=2;//Reset X distance
   Y_start+=GetText(Texts_Block1[7].text,Fontsize).Height;//Re-adjust Y distance, add height from section 8
   //-- Create the background object for width and height+3 of section 9 text
   Square(0,"Risk Ceil background",X_start,Y_start,GetText(Texts_Block1[8].text,Fontsize).Width,
          GetText(Texts_Block1[8].text,Fontsize).Height+3,ANCHOR_LEFT_UPPER);
   Y_start+=3;//Re-adjust Y distance
   X_start+=2;//Re-adjust X distance
//-- Will create the text objects for section 9
   TextObj(0,"Risk Ceil Text",Texts_Block1[8].subtext("Risk Ceiling Text"),X_start,Y_start,CORNER_LEFT_UPPER,Fontsize);
   X_start+=GetText(Texts_Block1[8].subtext("Risk Ceiling Text"),Fontsize).Width;//Re-adjust X distance
   TextObj(0,"Risk Ceil",Texts_Block1[8].subtext("Risk Ceiling"),X_start,Y_start,CORNER_LEFT_UPPER,Fontsize);
  }
```

The function below called Block\_2 will be responsible for creating the graphical elements for the graphical block 2 which is indicated below and consists of:

- Current Date and Time
- Event Date
- Event Name
- Event Country
- Event Currency
- Event Importance
- Spread Rating

> ![Graphical block 2](https://c.mql5.com/2/87/aUS30M1_Block2.png)

```
//+------------------------------------------------------------------+
//|Graphical Block 2                                                 |
//+------------------------------------------------------------------+
void CCommonGraphics::Block_2(uint SecondsPreEvent=5)
  {
//--- Set text object color depending if the chart color mode is LightMode or not
   TextObj_color=(isLightMode)?clrBlack:clrWheat;
   if(is_date)//Check whether to display date information
     {
      //--- Set text properties for Date and Time # section 10
      Texts_Block2[0].text = "Date:"+TimeToString(TimeTradeServer(),TIME_DATE)+"|| Time:"+TimeToString(TimeTradeServer(),TIME_MINUTES)
                             +"   ";//set main text
      Texts_Block2[0].subtext("Date Text","Date:");//set subtext - label,value
      Texts_Block2[0].subtext("Date",TimeToString(TimeTradeServer(),TIME_DATE));//set subtext - label,value
      Texts_Block2[0].subtext("Time Text","|| Time:");//set subtext - label,value
      Texts_Block2[0].subtext("Time",TimeToString(TimeTradeServer(),TIME_MINUTES));//set subtext - label,value
     }
   if(is_news)//Check whether to display news information
     {
      //--- Set text object color depending on upcoming news event's Importance
      EventColor = NewsObj.GetImportance_color(NewsObj.IMPORTANCE(UpcomingNews.EventImportance));
      //--- Set text properties for Event Date # section 11
      Texts_Block2[1].text = "Event: @"+UpcomingNews.EventDate+" ";//set main text
      Texts_Block2[1].subtext("Event Date Text","Event: @");//set subtext - label,value
      Texts_Block2[1].subtext("Event Date",UpcomingNews.EventDate);//set subtext - label,value
      //--- Set text properties for Event Name # section 12
      Texts_Block2[2].text = "Name: "+UpcomingNews.EventName+" ";//set main text
      Texts_Block2[2].subtext("Event Name Text","Name: ");//set subtext - label,value
      Texts_Block2[2].subtext("Event Name",UpcomingNews.EventName);//set subtext - label,value
      //--- Set text properties for Event Country # section 13
      Texts_Block2[3].text = "Country: "+UpcomingNews.CountryName+" ";//set main text
      Texts_Block2[3].subtext("Event Country Text","Country: ");//set subtext - label,value
      Texts_Block2[3].subtext("Event Country",UpcomingNews.CountryName);//set subtext - label,value
      //--- Set text properties for Event Currency # section 14
      Texts_Block2[4].text = "Currency: "+UpcomingNews.EventCurrency+" ";//set main text
      Texts_Block2[4].subtext("Event Currency Text","Currency: ");//set subtext - label,value
      Texts_Block2[4].subtext("Event Currency",UpcomingNews.EventCurrency);//set subtext - label,value
      //--- Set text properties for Event Importance # section 15
      Texts_Block2[5].text = "Importance: "+NewsObj.GetImportance(NewsObj.IMPORTANCE(UpcomingNews.EventImportance))+" ";//set main text
      Texts_Block2[5].subtext("Importance Text","Importance: ");//set subtext - label,value
      Texts_Block2[5].subtext("Importance",NewsObj.GetImportance(NewsObj.IMPORTANCE(UpcomingNews.EventImportance)));//set subtext - label,value
     }
   if(is_spread)//Check whether to display spread information
     {
      //--- Set text properties for Spread # section 16
      Texts_Block2[6].text = "Spread: "+string(CSymbol.Spread())+" Rating: "+CSymbol.SpreadDesc()+" ";//set main text
      Texts_Block2[6].subtext("Spread Text","Spread:");//set subtext - label,value
      Texts_Block2[6].subtext("Spread",string(CSymbol.Spread()));//set subtext - label,value
      Texts_Block2[6].subtext("Rating Text"," Rating:");//set subtext - label,value
      Texts_Block2[6].subtext("Rating Desc",CSymbol.SpreadDesc());//set subtext - label,value
     }

//--- Set basic properties
   Fontsize=10;
   X_start=2;//Reset X distance
   Y_start=GetTextMax(Texts_Block1,Fontsize).Height+29;//Re-adjust Y distance from graphical block 1 Height

   /* Create objects # section 10*/
   if(is_date)//Check whether to display date information
     {
      //-- Create the background object for width and height+3 of section 10 text
      Square(0,"Datetime background",X_start,Y_start,GetText(Texts_Block2[0].text,Fontsize).Width,
             GetText(Texts_Block2[0].text,Fontsize).Height+3,ANCHOR_LEFT_UPPER);
      Y_start+=3;//Re-adjust Y distance
      X_start+=2;//Re-adjust X distance
      //-- Will create the text objects for section 10
      TextObj(0,"Date Text",Texts_Block2[0].subtext("Date Text"),X_start,Y_start,CORNER_LEFT_UPPER,Fontsize);
      X_start+=GetText(Texts_Block2[0].subtext("Date Text"),Fontsize).Width;//Re-adjust X distance
      TextObj(0,"Date",Texts_Block2[0].subtext("Date"),X_start,Y_start,CORNER_LEFT_UPPER,Fontsize);
      X_start+=GetText(Texts_Block2[0].subtext("Date"),Fontsize).Width;//Re-adjust X distance
      TextObj(0,"Time Text",Texts_Block2[0].subtext("Time Text"),X_start,Y_start,CORNER_LEFT_UPPER,Fontsize);
      X_start+=GetText(Texts_Block2[0].subtext("Time Text"),Fontsize).Width;//Re-adjust X distance
      //--- Adjust text color depending if chart color mode is LightMode and if a news event is occurring
      TextObj_color = CTime.TimeIsInRange(CTime.TimeMinusOffset(datetime(UpcomingNews.EventDate),SecondsPreEvent),
                                          CTime.TimePlusOffset(datetime(UpcomingNews.EventDate),59))?clrRed:(isLightMode)?clrBlack:clrWheat;
      TextObj(0,"Time",Texts_Block2[0].subtext("Time"),X_start,Y_start,CORNER_LEFT_UPPER,Fontsize);
     }
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   if(UpcomingNews.CountryName!=NULL&&is_news)//Check whether to display news information and if upcoming news is available
     {
      /* Create objects # section 11*/
      Y_start+=(is_date)?GetText(Texts_Block2[0].text,Fontsize).Height:0;//Re-adjust Y distance depending if section 10 is shown
      X_start=2;//Reset X distance
      //-- Create the background object for width and height+3 of section 11 text
      Square(0,"Event Date background",X_start,Y_start,GetText(Texts_Block2[1].text,Fontsize).Width,
             GetText(Texts_Block2[1].text,Fontsize).Height+3,ANCHOR_LEFT_UPPER);
      Y_start+=3;//Re-adjust Y distance
      X_start+=2;//Re-adjust X distance
      //--- Set text object color depending if the chart color mode is LightMode or not
      TextObj_color=(isLightMode)?clrBlack:clrWheat;
      //-- Will create the text objects for section 11
      TextObj(0,"Event Date Text",Texts_Block2[1].subtext("Event Date Text"),X_start,Y_start,CORNER_LEFT_UPPER,Fontsize);
      X_start+=GetText(Texts_Block2[1].subtext("Event Date Text"),Fontsize).Width;//Re-adjust X distance
      //--- Adjust text color depending if chart color mode is LightMode and if a news event is occurring
      TextObj_color = CTime.TimeIsInRange(CTime.TimeMinusOffset(datetime(UpcomingNews.EventDate),SecondsPreEvent),
                                          CTime.TimePlusOffset(datetime(UpcomingNews.EventDate),59))?clrRed:(isLightMode)?clrBlack:clrWheat;
      TextObj(0,"Event Date",Texts_Block2[1].subtext("Event Date"),X_start,Y_start,CORNER_LEFT_UPPER,Fontsize);
      ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      /* Create objects # section 12*/
      Y_start+=GetText(Texts_Block2[1].text,Fontsize).Height;//Re-adjust Y distance
      X_start=2;//Reset X distance
      //-- Create the background object for width and height+3 of section 12 text
      Square(0,"Event Name background",X_start,Y_start,GetText(Texts_Block2[2].text,Fontsize).Width,
             GetText(Texts_Block2[2].text,Fontsize).Height+3,ANCHOR_LEFT_UPPER);
      Y_start+=3;//Re-adjust Y distance
      X_start+=2;//Re-adjust X distance
      //--- Set text object color depending if the chart color mode is LightMode or not
      TextObj_color=(isLightMode)?clrBlack:clrWheat;
      //-- Will create the text objects for section 12
      TextObj(0,"Event Name Text",Texts_Block2[2].subtext("Event Name Text"),X_start,Y_start,CORNER_LEFT_UPPER,Fontsize);
      X_start+=GetText(Texts_Block2[2].subtext("Event Name Text"),Fontsize).Width;//Re-adjust X distance
      //--- Set text object color depending on upcoming news event's Importance
      TextObj_color=EventColor;
      TextObj(0,"Event Name",Texts_Block2[2].subtext("Event Name"),X_start,Y_start,CORNER_LEFT_UPPER,Fontsize);
      ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      /* Create objects # section 13*/
      Y_start+=GetText(Texts_Block2[2].text,Fontsize).Height;//Re-adjust Y distance
      X_start=2;//Reset X distance
      //-- Create the background object for width and height+3 of section 13 text
      Square(0,"Event Country background",X_start,Y_start,GetText(Texts_Block2[3].text,Fontsize).Width,
             GetText(Texts_Block2[3].text,Fontsize).Height+3,ANCHOR_LEFT_UPPER);
      Y_start+=3;//Re-adjust Y distance
      X_start+=2;//Re-adjust X distance
      //--- Set text object color depending if the chart color mode is LightMode or not
      TextObj_color=(isLightMode)?clrBlack:clrWheat;
      //-- Will create the text objects for section 13
      TextObj(0,"Event Country Text",Texts_Block2[3].subtext("Event Country Text"),X_start,Y_start,CORNER_LEFT_UPPER,Fontsize);
      X_start+=GetText(Texts_Block2[3].subtext("Event Country Text"),Fontsize).Width;//Re-adjust X distance
      //--- Set text object color depending on upcoming news event's Importance
      TextObj_color=EventColor;
      TextObj(0,"Event Country",Texts_Block2[3].subtext("Event Country"),X_start,Y_start,CORNER_LEFT_UPPER,Fontsize);
      ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      /* Create objects # section 14*/
      Y_start+=GetText(Texts_Block2[3].text,Fontsize).Height;//Re-adjust Y distance
      X_start=2;//Reset X distance
      //-- Create the background object for width and height+3 of section 14 text
      Square(0,"Event Currency background",X_start,Y_start,GetText(Texts_Block2[4].text,Fontsize).Width,
             GetText(Texts_Block2[4].text,Fontsize).Height+3,ANCHOR_LEFT_UPPER);
      Y_start+=3;//Re-adjust Y distance
      X_start+=2;//Re-adjust X distance
      //--- Set text object color depending if the chart color mode is LightMode or not
      TextObj_color=(isLightMode)?clrBlack:clrWheat;
      //-- Will create the text objects for section 14
      TextObj(0,"Event Currency Text",Texts_Block2[4].subtext("Event Currency Text"),X_start,Y_start,CORNER_LEFT_UPPER,Fontsize);
      X_start+=GetText(Texts_Block2[4].subtext("Event Currency Text"),Fontsize).Width;//Re-adjust X distance
      //--- Set text object color depending on upcoming news event's Importance
      TextObj_color=EventColor;
      TextObj(0,"Event Currency",Texts_Block2[4].subtext("Event Currency"),X_start,Y_start,CORNER_LEFT_UPPER,Fontsize);
      ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      /* Create objects # section 15*/
      Y_start+=GetText(Texts_Block2[4].text,Fontsize).Height;//Re-adjust Y distance
      X_start=2;//Reset X distance
      //-- Create the background object for width and height+3 of section 15 text
      Square(0,"Event Importance background",X_start,Y_start,GetText(Texts_Block2[5].text,Fontsize).Width,
             GetText(Texts_Block2[5].text,Fontsize).Height+3,ANCHOR_LEFT_UPPER);
      Y_start+=3;//Re-adjust Y distance
      X_start+=2;//Re-adjust X distance
      //--- Set text object color depending if the chart color mode is LightMode or not
      TextObj_color=(isLightMode)?clrBlack:clrWheat;
      //-- Will create the text objects for section 15
      TextObj(0,"Importance Text",Texts_Block2[5].subtext("Importance Text"),X_start,Y_start,CORNER_LEFT_UPPER,Fontsize);
      X_start+=GetText(Texts_Block2[5].subtext("Importance Text"),Fontsize).Width;//Re-adjust X distance
      //--- Set text object color depending on upcoming news event's Importance
      TextObj_color=EventColor;
      TextObj(0,"Importance",Texts_Block2[5].subtext("Importance"),X_start,Y_start,CORNER_LEFT_UPPER,Fontsize);
      ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      /* Create objects # section 16*/
      if(is_spread)//Check whether to display spread information
        {
         Y_start+=GetText(Texts_Block2[5].text,Fontsize).Height;//Re-adjust Y distance
         X_start=2;//Reset X distance
         //-- Create the background object for width and height+3 of section 16 text
         Square(0,"Spread background",X_start,Y_start,GetText(Texts_Block2[6].text,Fontsize).Width,
                GetText(Texts_Block2[6].text,Fontsize).Height+3,ANCHOR_LEFT_UPPER);
         Y_start+=3;//Re-adjust Y distance
         X_start+=2;//Re-adjust X distance
         //--- Set text object color depending on spread rating
         color Spread_clr=CSymbol.SpreadColor();
         //--- Set text object color depending if the chart color mode is LightMode or not
         TextObj_color=(isLightMode)?clrBlack:clrWheat;
         //-- Will create the text objects for section 16
         TextObj(0,"Symbol Spread Text",Texts_Block2[6].subtext("Spread Text"),X_start,Y_start,CORNER_LEFT_UPPER,Fontsize);
         X_start+=GetText(Texts_Block2[6].subtext("Spread Text"),Fontsize).Width;//Re-adjust X distance
         //--- Set text object color depending on spread rating
         TextObj_color=Spread_clr;
         TextObj(0,"Symbol Spread",Texts_Block2[6].subtext("Spread"),X_start,Y_start,CORNER_LEFT_UPPER,Fontsize);
         X_start+=GetText(Texts_Block2[6].subtext("Spread"),Fontsize).Width;//Re-adjust X distance
         //--- Set text object color depending if the chart color mode is LightMode or not
         TextObj_color=(isLightMode)?clrBlack:clrWheat;
         TextObj(0,"Symbol Rating Text",Texts_Block2[6].subtext("Rating Text"),X_start,Y_start,CORNER_LEFT_UPPER,Fontsize);
         X_start+=GetText(Texts_Block2[6].subtext("Rating Text"),Fontsize).Width;//Re-adjust X distance
         //--- Set text object color depending on spread rating
         TextObj_color=Spread_clr;
         TextObj(0,"Symbol Rating",Texts_Block2[6].subtext("Rating Desc"),X_start,Y_start,CORNER_LEFT_UPPER,Fontsize);
        }
     }
   else
      /* Create objects # section 16*/
      if(is_spread)//Check whether to display spread information
        {
         Y_start+=(is_date)?GetText(Texts_Block2[0].text,Fontsize).Height:0;//Re-adjust Y distance depending if section 10 is shown
         X_start=2;//Reset X distance
         //-- Create the background object for width and height+3 of section 16 text
         Square(0,"Spread background",X_start,Y_start,GetText(Texts_Block2[6].text,Fontsize).Width,
                GetText(Texts_Block2[6].text,Fontsize).Height+3,ANCHOR_LEFT_UPPER);
         Y_start+=3;//Re-adjust Y distance
         X_start+=2;//Re-adjust X distance
         //--- Set text object color depending on spread rating
         color Spread_clr=CSymbol.SpreadColor();
         //--- Set text object color depending if the chart color mode is LightMode or not
         TextObj_color=(isLightMode)?clrBlack:clrWheat;
         //-- Will create the text objects for section 16
         TextObj(0,"Symbol Spread Text",Texts_Block2[6].subtext("Spread Text"),X_start,Y_start,CORNER_LEFT_UPPER,Fontsize);
         X_start+=GetText(Texts_Block2[6].subtext("Spread Text"),Fontsize).Width;//Re-adjust X distance
         //--- Set text object color depending on spread rating
         TextObj_color=Spread_clr;
         TextObj(0,"Symbol Spread",Texts_Block2[6].subtext("Spread"),X_start,Y_start,CORNER_LEFT_UPPER,Fontsize);
         X_start+=GetText(Texts_Block2[6].subtext("Spread"),Fontsize).Width;//Re-adjust X distance
         //--- Set text object color depending if the chart color mode is LightMode or not
         TextObj_color=(isLightMode)?clrBlack:clrWheat;
         TextObj(0,"Symbol Rating Text",Texts_Block2[6].subtext("Rating Text"),X_start,Y_start,CORNER_LEFT_UPPER,Fontsize);
         X_start+=GetText(Texts_Block2[6].subtext("Rating Text"),Fontsize).Width;//Re-adjust X distance
         //--- Set text object color depending on spread rating
         TextObj_color=Spread_clr;
         TextObj(0,"Symbol Rating",Texts_Block2[6].subtext("Rating Desc"),X_start,Y_start,CORNER_LEFT_UPPER,Fontsize);
        }
  }
```

The function below will update any changes to the graphical Block 2 sections. And will not re-create all the elements from the function Block\_2.

```
//+------------------------------------------------------------------+
//|will update certain graphics at an interval                       |
//+------------------------------------------------------------------+
void CCommonGraphics::Block_2_Realtime(uint SecondsPreEvent=5)
  {
   if(MQLInfoInteger(MQL_TESTER)&&!MQLInfoInteger(MQL_VISUAL_MODE))
     {
      return;//exit if in strategy tester and not in visual mode
     }

   if(is_date)//Check whether to display date information
     {
      //--- Set text properties for Date and Time # section 10
      Texts_Block2[0].text = "Date:"+TimeToString(TimeTradeServer(),TIME_DATE)+"|| Time:"+TimeToString(TimeTradeServer(),TIME_SECONDS)
                             +"   ";//set main text
      Texts_Block2[0].subtext("Date Text","Date:");//set subtext - label,value
      Texts_Block2[0].subtext("Date",TimeToString(TimeTradeServer(),TIME_DATE));//set subtext - label,value
      Texts_Block2[0].subtext("Time Text","|| Time:");//set subtext - label,value
      Texts_Block2[0].subtext("Time",TimeToString(TimeTradeServer(),TIME_SECONDS));//set subtext - label,value
     }
   if(is_news)//Check whether to display news information
     {
      //--- Set text object color depending on upcoming news event's Importance
      EventColor = NewsObj.GetImportance_color(NewsObj.IMPORTANCE(UpcomingNews.EventImportance));
      //--- Set text properties for Event Date # section 11
      Texts_Block2[1].text = "Event: @"+UpcomingNews.EventDate+" ";//set main text
      Texts_Block2[1].subtext("Event Date Text","Event: @");//set subtext - label,value
      Texts_Block2[1].subtext("Event Date",UpcomingNews.EventDate);//set subtext - label,value
      //--- Set text properties for Event Name # section 12
      Texts_Block2[2].text = "Name: "+UpcomingNews.EventName+" ";//set main text
      Texts_Block2[2].subtext("Event Name Text","Name: ");//set subtext - label,value
      Texts_Block2[2].subtext("Event Name",UpcomingNews.EventName);//set subtext - label,value
      //--- Set text properties for Event Country # section 13
      Texts_Block2[3].text = "Country: "+UpcomingNews.CountryName+" ";//set main text
      Texts_Block2[3].subtext("Event Country Text","Country: ");//set subtext - label,value
      Texts_Block2[3].subtext("Event Country",UpcomingNews.CountryName);//set subtext - label,value
      //--- Set text properties for Event Currency # section 14
      Texts_Block2[4].text = "Currency: "+UpcomingNews.EventCurrency+" ";//set main text
      Texts_Block2[4].subtext("Event Currency Text","Currency: ");//set subtext - label,value
      Texts_Block2[4].subtext("Event Currency",UpcomingNews.EventCurrency);//set subtext - label,value
      //--- Set text properties for Event Importance # section 15
      Texts_Block2[5].text = "Importance: "+NewsObj.GetImportance(NewsObj.IMPORTANCE(UpcomingNews.EventImportance))+" ";//set main text
      Texts_Block2[5].subtext("Importance Text","Importance: ");//set subtext - label,value
      Texts_Block2[5].subtext("Importance",NewsObj.GetImportance(NewsObj.IMPORTANCE(UpcomingNews.EventImportance)));//set subtext - label,value
     }
   if(is_spread)//Check whether to display spread information
     {
      //--- Set text properties for Spread # section 16
      Texts_Block2[6].text = "Spread: "+string(CSymbol.Spread())+" Rating: "+CSymbol.SpreadDesc()+" ";//set main text
      Texts_Block2[6].subtext("Spread Text","Spread:");//set subtext - label,value
      Texts_Block2[6].subtext("Spread",string(CSymbol.Spread()));//set subtext - label,value
      Texts_Block2[6].subtext("Rating Text"," Rating:");//set subtext - label,value
      Texts_Block2[6].subtext("Rating Desc",CSymbol.SpreadDesc());//set subtext - label,value
     }

//--- Set basic properties
   Fontsize=10;
   X_start=2;//Reset X distance
   Y_start=GetTextMax(Texts_Block1,Fontsize).Height+29;//Re-adjust Y distance from section block 1

   /* Create objects # section 10*/
   if(is_date)//Check whether to display date information
     {
      //-- Check if the background object x-size for section 10 is the same size as section 10 text width
      if(ObjectGetInteger(0,"Datetime background",OBJPROP_XSIZE)!=GetText(Texts_Block2[0].text,Fontsize).Width)
        {
         //-- Will re-adjust background object to any changes of section 10 text width
         ObjectSetInteger(0,"Datetime background",OBJPROP_XSIZE,long(GetText(Texts_Block2[0].text,Fontsize).Width));
        }
      Y_start+=3;//Re-adjust Y distance
      X_start+=2;//Re-adjust X distance
      //--- Set text object color depending if the chart color mode is LightMode or not
      TextObj_color=(isLightMode)?clrBlack:clrWheat;
      //-- Will update the text objects for section 10
      TextObj(0,"Date Text",Texts_Block2[0].subtext("Date Text"),X_start,Y_start,CORNER_LEFT_UPPER,Fontsize);
      X_start+=GetText(Texts_Block2[0].subtext("Date Text"),Fontsize).Width;//Re-adjust X distance
      //--- Set text object color depending if the chart color mode is LightMode or not
      TextObj_color=(isLightMode)?clrBlack:clrWheat;
      TextObj(0,"Date",Texts_Block2[0].subtext("Date"),X_start,Y_start,CORNER_LEFT_UPPER,Fontsize);
      X_start+=GetText(Texts_Block2[0].subtext("Date"),Fontsize).Width;//Re-adjust X distance
      //--- Set text object color depending if the chart color mode is LightMode or not
      TextObj_color=(isLightMode)?clrBlack:clrWheat;
      TextObj(0,"Time Text",Texts_Block2[0].subtext("Time Text"),X_start,Y_start,CORNER_LEFT_UPPER,Fontsize);
      X_start+=GetText(Texts_Block2[0].subtext("Time Text"),Fontsize).Width;//Re-adjust X distance
      //--- Adjust text color depending if chart color mode is LightMode and if a news event is occurring
      TextObj_color = CTime.TimeIsInRange(CTime.TimeMinusOffset(datetime(UpcomingNews.EventDate),SecondsPreEvent),
                                          CTime.TimePlusOffset(datetime(UpcomingNews.EventDate),59))?clrRed:(isLightMode)?clrBlack:clrWheat;
      TextObj(0,"Time",Texts_Block2[0].subtext("Time"),X_start,Y_start,CORNER_LEFT_UPPER,Fontsize);
     }
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   if(UpcomingNews.CountryName!=NULL&&is_news)//Check whether to display news information
     {
      /* Create objects # section 11*/
      Y_start+=(is_date)?GetText(Texts_Block2[0].text,Fontsize).Height:0;
      X_start=2;//Reset X distance
      //-- Check if the background object x-size for section 11 is the same size as section 11 text width
      if(ObjectGetInteger(0,"Event Date background",OBJPROP_XSIZE)!=GetText(Texts_Block2[1].text,Fontsize).Width)
        {
         //-- Will re-adjust background object to any changes of section 11 text width
         ObjectSetInteger(0,"Event Date background",OBJPROP_XSIZE,long(GetText(Texts_Block2[1].text,Fontsize).Width));
        }
      Y_start+=3;//Re-adjust Y distance
      X_start+=2;//Re-adjust X distance
      //--- Set text object color depending if the chart color mode is LightMode or not
      TextObj_color=(isLightMode)?clrBlack:clrWheat;
      //-- Will update the text objects for section 11
      TextObj(0,"Event Date Text",Texts_Block2[1].subtext("Event Date Text"),X_start,Y_start,CORNER_LEFT_UPPER,Fontsize);
      X_start+=GetText(Texts_Block2[1].subtext("Event Date Text"),Fontsize).Width;//Re-adjust X distance
      //--- Adjust text color depending if chart color mode is LightMode and if a news event is occurring
      TextObj_color = CTime.TimeIsInRange(CTime.TimeMinusOffset(datetime(UpcomingNews.EventDate),SecondsPreEvent),
                                          CTime.TimePlusOffset(datetime(UpcomingNews.EventDate),59))?clrRed:(isLightMode)?clrBlack:clrWheat;
      TextObj(0,"Event Date",Texts_Block2[1].subtext("Event Date"),X_start,Y_start,CORNER_LEFT_UPPER,Fontsize);
      ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      /* Create objects # section 12*/
      Y_start+=GetText(Texts_Block2[1].text,Fontsize).Height;//Re-adjust Y distance
      X_start=2;//Reset X distance
      //-- Check if the background object x-size for section 12 is the same size as section 12 text width
      if(ObjectGetInteger(0,"Event Name background",OBJPROP_XSIZE)!=GetText(Texts_Block2[2].text,Fontsize).Width)
        {
         //-- Will re-adjust background object to any changes of section 12 text width
         ObjectSetInteger(0,"Event Name background",OBJPROP_XSIZE,long(GetText(Texts_Block2[2].text,Fontsize).Width));
        }
      Y_start+=3;//Re-adjust Y distance
      X_start+=2;//Re-adjust X distance
      //--- Set text object color depending if the chart color mode is LightMode or not
      TextObj_color=(isLightMode)?clrBlack:clrWheat;
      //-- Will update the text objects for section 12
      TextObj(0,"Event Name Text",Texts_Block2[2].subtext("Event Name Text"),X_start,Y_start,CORNER_LEFT_UPPER,Fontsize);
      X_start+=GetText(Texts_Block2[2].subtext("Event Name Text"),Fontsize).Width;//Re-adjust X distance
      //--- Set text object color depending on upcoming news event's Importance
      TextObj_color=EventColor;
      TextObj(0,"Event Name",Texts_Block2[2].subtext("Event Name"),X_start,Y_start,CORNER_LEFT_UPPER,Fontsize);
      ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      /* Create objects # section 13*/
      Y_start+=GetText(Texts_Block2[2].text,Fontsize).Height;//Re-adjust Y distance
      X_start=2;//Reset X distance
      //-- Check if the background object x-size for section 13 is the same size as section 13 text width
      if(ObjectGetInteger(0,"Event Country background",OBJPROP_XSIZE)!=GetText(Texts_Block2[3].text,Fontsize).Width)
        {
         //-- Will re-adjust background object to any changes of section 13 text width
         ObjectSetInteger(0,"Event Country background",OBJPROP_XSIZE,long(GetText(Texts_Block2[3].text,Fontsize).Width));
        }
      Y_start+=3;//Re-adjust Y distance
      X_start+=2;//Re-adjust X distance
      //--- Set text object color depending if the chart color mode is LightMode or not
      TextObj_color=(isLightMode)?clrBlack:clrWheat;
      //-- Will update the text objects for section 13
      TextObj(0,"Event Country Text",Texts_Block2[3].subtext("Event Country Text"),X_start,Y_start,CORNER_LEFT_UPPER,Fontsize);
      X_start+=GetText(Texts_Block2[3].subtext("Event Country Text"),Fontsize).Width;//Re-adjust X distance
      //--- Set text object color depending on upcoming news event's Importance
      TextObj_color=EventColor;
      TextObj(0,"Event Country",Texts_Block2[3].subtext("Event Country"),X_start,Y_start,CORNER_LEFT_UPPER,Fontsize);
      ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      /* Create objects # section 14*/
      Y_start+=GetText(Texts_Block2[3].text,Fontsize).Height;//Re-adjust Y distance
      X_start=2;//Reset X distance
      //-- Check if the background object x-size for section 14 is the same size as section 14 text width
      if(ObjectGetInteger(0,"Event Currency background",OBJPROP_XSIZE)!=GetText(Texts_Block2[4].text,Fontsize).Width)
        {
         //-- Will re-adjust background object to any changes of section 14 text width
         ObjectSetInteger(0,"Event Currency background",OBJPROP_XSIZE,long(GetText(Texts_Block2[4].text,Fontsize).Width));
        }
      Y_start+=3;//Re-adjust Y distance
      X_start+=2;//Re-adjust X distance
      //--- Set text object color depending if the chart color mode is LightMode or not
      TextObj_color=(isLightMode)?clrBlack:clrWheat;
      //-- Will update the text objects for section 14
      TextObj(0,"Event Currency Text",Texts_Block2[4].subtext("Event Currency Text"),X_start,Y_start,CORNER_LEFT_UPPER,Fontsize);
      X_start+=GetText(Texts_Block2[4].subtext("Event Currency Text"),Fontsize).Width;//Re-adjust X distance
      //--- Set text object color depending on upcoming news event's Importance
      TextObj_color=EventColor;
      TextObj(0,"Event Currency",Texts_Block2[4].subtext("Event Currency"),X_start,Y_start,CORNER_LEFT_UPPER,Fontsize);
      ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      /* Create objects # section 15*/
      Y_start+=GetText(Texts_Block2[4].text,Fontsize).Height;//Re-adjust Y distance
      X_start=2;//Reset X distance
      //-- Check if the background object x-size for section 15 is the same size as section 15 text width
      if(ObjectGetInteger(0,"Event Importance background",OBJPROP_XSIZE)!=GetText(Texts_Block2[5].text,Fontsize).Width)
        {
         //-- Will re-adjust background object to any changes of section 15 text width
         ObjectSetInteger(0,"Event Importance background",OBJPROP_XSIZE,long(GetText(Texts_Block2[5].text,Fontsize).Width));
        }
      Y_start+=3;//Re-adjust Y distance
      X_start+=2;//Re-adjust X distance
      //--- Set text object color depending if the chart color mode is LightMode or not
      TextObj_color=(isLightMode)?clrBlack:clrWheat;
      //-- Will update the text objects for section 15
      TextObj(0,"Importance Text",Texts_Block2[5].subtext("Importance Text"),X_start,Y_start,CORNER_LEFT_UPPER,Fontsize);
      X_start+=GetText(Texts_Block2[5].subtext("Importance Text"),Fontsize).Width;//Re-adjust X distance
      //--- Set text object color depending on upcoming news event's Importance
      TextObj_color=EventColor;
      TextObj(0,"Importance",Texts_Block2[5].subtext("Importance"),X_start,Y_start,CORNER_LEFT_UPPER,Fontsize);
      ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      /* Create objects # section 16*/
      if(is_spread)//Check whether to display spread information
        {
         Y_start+=GetText(Texts_Block2[5].text,Fontsize).Height;//Re-adjust Y distance
         X_start=2;//Reset X distance
         //-- Check if the background object x-size for section 16 is the same size as section 16 text width
         if(ObjectGetInteger(0,"Spread background",OBJPROP_XSIZE)!=GetText(Texts_Block2[6].text,Fontsize).Width)
           {
            //-- Will re-adjust background object to any changes of section 16 text width
            ObjectSetInteger(0,"Spread background",OBJPROP_XSIZE,long(GetText(Texts_Block2[6].text,Fontsize).Width));
           }
         Y_start+=3;//Re-adjust Y distance
         X_start+=2;//Re-adjust X distance
         //--- Set text object color depending on spread rating
         color Spread_clr=CSymbol.SpreadColor();
         //-- Will create the text object for the Symbol's name
         X_start+=GetText(Texts_Block2[6].subtext("Spread Text"),Fontsize).Width;//Re-adjust X distance
         //--- Set text object color depending on spread rating
         TextObj_color=Spread_clr;
         //-- Will update the text objects for section 16
         TextObj(0,"Symbol Spread",Texts_Block2[6].subtext("Spread"),X_start,Y_start,CORNER_LEFT_UPPER,Fontsize);
         X_start+=GetText(Texts_Block2[6].subtext("Spread"),Fontsize).Width;//Re-adjust X distance
         //--- Set text object color depending if the chart color mode is LightMode or not
         TextObj_color=(isLightMode)?clrBlack:clrWheat;
         TextObj(0,"Symbol Rating Text",Texts_Block2[6].subtext("Rating Text"),X_start,Y_start,CORNER_LEFT_UPPER,Fontsize);
         X_start+=GetText(Texts_Block2[6].subtext("Rating Text"),Fontsize).Width;//Re-adjust X distance
         //--- Set text object color depending on spread rating
         TextObj_color=Spread_clr;
         TextObj(0,"Symbol Rating",Texts_Block2[6].subtext("Rating Desc"),X_start,Y_start,CORNER_LEFT_UPPER,Fontsize);
        }
     }
   else
      /* Create objects # section 16*/
      if(is_spread)//Check whether to display spread information
        {
         Y_start+=(is_date)?GetText(Texts_Block2[0].text,Fontsize).Height:0;//Re-adjust Y distance
         X_start=2;//Reset X distance
         //-- Check if the background object x-size for section 16 is the same size as section 16 text width
         if(ObjectGetInteger(0,"Spread background",OBJPROP_XSIZE)!=GetText(Texts_Block2[6].text,Fontsize).Width)
           {
            //-- Will re-adjust background object to any changes of section 16 text width
            ObjectSetInteger(0,"Spread background",OBJPROP_XSIZE,long(GetText(Texts_Block2[6].text,Fontsize).Width));
           }
         Y_start+=3;//Re-adjust Y distance
         X_start+=2;//Re-adjust X distance
         //--- Set text object color depending on spread rating
         color Spread_clr=CSymbol.SpreadColor();
         //-- Will create the text object for the Symbol's name
         X_start+=GetText(Texts_Block2[6].subtext("Spread Text"),Fontsize).Width;//Re-adjust X distance
         //--- Set text object color depending on spread rating
         TextObj_color=Spread_clr;
         //-- Will update the text objects for section 16
         TextObj(0,"Symbol Spread",Texts_Block2[6].subtext("Spread"),X_start,Y_start,CORNER_LEFT_UPPER,Fontsize);
         X_start+=GetText(Texts_Block2[6].subtext("Spread"),Fontsize).Width;//Re-adjust X distance
         //--- Set text object color depending if the chart color mode is LightMode or not
         TextObj_color=(isLightMode)?clrBlack:clrWheat;
         TextObj(0,"Symbol Rating Text",Texts_Block2[6].subtext("Rating Text"),X_start,Y_start,CORNER_LEFT_UPPER,Fontsize);
         X_start+=GetText(Texts_Block2[6].subtext("Rating Text"),Fontsize).Width;//Re-adjust X distance
         //--- Set text object color depending on spread rating
         TextObj_color=Spread_clr;
         TextObj(0,"Symbol Rating",Texts_Block2[6].subtext("Rating Desc"),X_start,Y_start,CORNER_LEFT_UPPER,Fontsize);
        }
  }
```

The function below will create the event object on the chart with all the available news events for the current day.

> ![Section 17](https://c.mql5.com/2/87/xUS30M1_Section17.png)

```
//+------------------------------------------------------------------+
//|will create chart event objects                                   |
//+------------------------------------------------------------------+
void CCommonGraphics::NewsEvent()
  {
   if(!is_events||(MQLInfoInteger(MQL_TESTER)&&!MQLInfoInteger(MQL_VISUAL_MODE)))
     {return;}//exit if in strategy tester and not in visual mode or is_events variable is false
//--- Retrieve news events for the current Daily period into array CalendarArray
   NewsObj.EconomicDetailsMemory(CalendarArray,iTime(Symbol(),PERIOD_D1,0));
//--- Iterate through all events in CalendarArray
   for(uint i=0;i<CalendarArray.Size();i++)
     {
      //--- Create event object with the news properties
      EventObj(0,CalendarArray[i].EventName+" "+CalendarArray[i].CountryName+" "+CalendarArray[i].EventDate,
               CalendarArray[i].EventName+"["+CalendarArray[i].CountryName+"]",StringToTime(CalendarArray[i].EventDate));
     }
//--- Refresh the chart/ update the chart
   ChartRefresh();
  }
```

### Trade Management Class

This class will be responsible for opening trades for our expert. The functionality for this class will likely expand in later articles. Trade management class will inherit from risk management class to configure the lot-sizes for each trade.

```
#include <Trade\Trade.mqh>
#include <Trade\OrderInfo.mqh>
#include <Trade\SymbolInfo.mqh>
#include "RiskManagement.mqh"
#include "TimeManagement.mqh"
//+------------------------------------------------------------------+
//|TradeManagement class                                             |
//+------------------------------------------------------------------+
class CTradeManagement:CRiskManagement
  {
private:
   CTrade            Trade;//Trade class object
   CSymbolProperties CSymbol;//SymbolProperties class object
   CTimeManagement   CTime;//TimeManagement class object
   bool              TradeResult;//boolean to store trade result
   double            mySL;//double variable to store Stoploss
   double            myTP;//double variable to store Takeprofit
public:
   //--- Class constructor
                     CTradeManagement(string SYMBOL=NULL)
     {
      //--- Set symbol name
      CSymbol.SetSymbolName(SYMBOL);
     }
   //--- Class destructor
                    ~CTradeManagement(void) {}
   //--- Will retrieve if there are any open trades
   bool              OpenTrade(ENUM_POSITION_TYPE Type,ulong Magic,string COMMENT=NULL);
   //--- Will retrieve if there are any deals
   bool              OpenedDeal(ENUM_DEAL_TYPE Type,ulong Magic,string COMMENT=NULL);
   //--- Will attempt open buy trade
   bool              Buy(double SL,double TP,ulong Magic,string COMMENT=NULL);
   //--- Will attempt open buy trade with integer SL
   bool              Buy(int SL,double TP,ulong Magic,string COMMENT=NULL);
   //--- Will attempt open buy trade with integer TP
   bool              Buy(double SL,int TP,ulong Magic,string COMMENT=NULL);
   //--- Will attempt open buy trade with integer SL & TP
   bool              Buy(int SL,int TP,ulong Magic,string COMMENT=NULL);
   //--- Will attempt open sell trade
   bool              Sell(double SL,double TP,ulong Magic,string COMMENT=NULL);
   //--- Will attempt open sell trade with integer SL
   bool              Sell(int SL,double TP,ulong Magic,string COMMENT=NULL);
   //--- Will attempt open sell trade with integer TP
   bool              Sell(double SL,int TP,ulong Magic,string COMMENT=NULL);
   //--- Will attempt open sell trade with integer SL & TP
   bool              Sell(int SL,int TP,ulong Magic,string COMMENT=NULL);
  };
```

The function below will check for any open positions with a specific Position's type (Buy/Sell), Magic (Unique footprint), Comment (Position details). We will use this function to check if we already opened a trade when the news event is occurring, so we avoid opening extra/unnecessary trades.

```
//+------------------------------------------------------------------+
//|Will retrieve if there are any open trades                        |
//+------------------------------------------------------------------+
bool CTradeManagement::OpenTrade(ENUM_POSITION_TYPE Type,ulong Magic,string COMMENT=NULL)
  {
//--- Iterate through all open positions
   for(int i=0; i<PositionsTotal(); i++)
     {
      //--- Check if Position ticket is above zero
      if(PositionGetTicket(i)>0)
        {
         //--- Check if the Position's Symbol,Magic,Type,Comment is correct
         if(PositionGetString(POSITION_SYMBOL)==CSymbol.GetSymbolName()&&PositionGetInteger(POSITION_MAGIC)==Magic
            &&PositionGetInteger(POSITION_TYPE)==Type&&PositionGetString(POSITION_COMMENT)==COMMENT)
           {
            //--- Return true when there is an open position
            return true;
           }
        }
     }
//--- No open positions found.
   return false;
  }
```

The function OpenedDeal will check for any opened deals this could be opened buy/sell deals/trades. We need this function to prevent us/the expert from opening new trades when a trade was closed during a news event. Without this function, the expert will open a buy trade for example when NFP is about to happen, if the open trade is closed because of volatility we don't want the expert to open another, this is because we already traded the event no need for more trades that could result in unnecessary losses.

```
//+------------------------------------------------------------------+
//|Will retrieve if there are any deals                              |
//+------------------------------------------------------------------+
bool CTradeManagement::OpenedDeal(ENUM_DEAL_TYPE Type,ulong Magic,string COMMENT=NULL)
  {
//--- Check History starting from 2 minutes ago
   if(HistorySelect(CTime.TimeMinusOffset(TimeTradeServer(),CTime.MinutesS(2)),TimeTradeServer()))
     {
      //--- Iterate through all history deals
      for(int i=0; i<HistoryDealsTotal(); i++)
        {
         //--- Assign history deal ticket
         ulong ticket = HistoryDealGetTicket(i);
         //--- Check if ticket is more than zero
         if(ticket>0)
           {
            //--- Check if the Deal's Symbol,Magic,Type,Comment is correct
            if(HistoryDealGetString(ticket,DEAL_SYMBOL)==CSymbol.GetSymbolName()&&
               HistoryDealGetInteger(ticket,DEAL_MAGIC)==Magic&&HistoryDealGetInteger(ticket,DEAL_TYPE)==Type
               &&HistoryDealGetString(ticket,DEAL_COMMENT)==COMMENT)
              {
               //--- Return true when there are any deals
               return true;
              }
           }
        }
     }
//--- No deals found.
   return false;
  }
```

The function below will open all the Market orders for buy/long trades.

```
//+------------------------------------------------------------------+
//|Will attempt open buy trade                                       |
//+------------------------------------------------------------------+
bool CTradeManagement::Buy(double SL,double TP,ulong Magic,string COMMENT=NULL)
  {
//--- Normalize the SL Price
   CSymbol.NormalizePrice(SL);
//--- Normalize the TP Price
   CSymbol.NormalizePrice(TP);
//--- Set the order type for Risk management calculation
   SetOrderType(ORDER_TYPE_BUY);
//--- Set open price for Risk management calculation
   OpenPrice = CSymbol.Ask();
//--- Set close price for Risk management calculation
   ClosePrice = SL;
//--- Set Trade magic number
   Trade.SetExpertMagicNumber(Magic);
//--- Check if there are any open trades or opened deals already
   if(!OpenTrade(POSITION_TYPE_BUY,Magic,COMMENT)&&!OpenedDeal(DEAL_TYPE_BUY,Magic,COMMENT))
     {
      //--- Iterate through the Lot-sizes if they're more than max-lot
      for(double i=Volume();i>=CSymbol.LotsMin();i-=CSymbol.LotsMax())
        {
         //--- normalize Lot-size
         NormalizeLotsize(i);
         //--- Open trade with a Lot-size not more than max-lot
         TradeResult = Trade.Buy((i>CSymbol.LotsMax())?CSymbol.LotsMax():i,CSymbol.GetSymbolName(),CSymbol.Ask(),SL,TP,COMMENT);
         //--- Check if trade failed.
         if(!TradeResult)
           {
            return TradeResult;
           }
        }
     }
   else
     {
      //--- Trade failed because there is an open trade or opened deal
      return false;
     }
//--- Return trade result.
   return TradeResult;
  }
```

The function below will open all the Market orders for sell/short trades.

```
//+------------------------------------------------------------------+
//|Will attempt open sell trade                                      |
//+------------------------------------------------------------------+
bool CTradeManagement::Sell(double SL,double TP,ulong Magic,string COMMENT=NULL)
  {
//--- Normalize the SL Price
   CSymbol.NormalizePrice(SL);
//--- Normalize the TP Price
   CSymbol.NormalizePrice(TP);
//--- Set the order type for Risk management calculation
   SetOrderType(ORDER_TYPE_SELL);
//--- Set open price for Risk management calculation
   OpenPrice = CSymbol.Bid();
//--- Set close price for Risk management calculation
   ClosePrice = SL;
//--- Set Trade magic number
   Trade.SetExpertMagicNumber(Magic);
//--- Check if there are any open trades or opened deals already
   if(!OpenTrade(POSITION_TYPE_SELL,Magic,COMMENT)&&!OpenedDeal(DEAL_TYPE_SELL,Magic,COMMENT))
     {
      //--- Iterate through the Lot-sizes if they're more than max-lot
      for(double i=Volume();i>=CSymbol.LotsMin();i-=CSymbol.LotsMax())
        {
         //--- normalize Lot-size
         NormalizeLotsize(i);
         //--- Open trade with a Lot-size not more than max-lot
         TradeResult = Trade.Sell((i>CSymbol.LotsMax())?CSymbol.LotsMax():i,CSymbol.GetSymbolName(),CSymbol.Bid(),SL,TP,COMMENT);
         //--- Check if trade failed.
         if(!TradeResult)
           {
            return TradeResult;
           }
        }
     }
   else
     {
      //--- Trade failed because there is an open trade or opened deal
      return false;
     }
//--- Return trade result.
   return TradeResult;
  }
```

### News Trading Expert

We have new inputs for the expert, we have already seen the explanations in the introduction.

```
//--- width and height of the canvas (used for drawing)
#define IMG_WIDTH  200
#define IMG_HEIGHT 100
//--- enable to set color format
ENUM_COLOR_FORMAT clr_format=COLOR_FORMAT_XRGB_NOALPHA;
//--- drawing array (buffer)
uint ExtImg[IMG_WIDTH*IMG_HEIGHT];

#include "News.mqh"
CNews NewsObject;//Class object for News
#include "TimeManagement.mqh"
CTimeManagement CTM;//Class object for Time Management
#include "WorkingWithFolders.mqh"
CFolders Folder;//Class object for Folders
#include "ChartProperties.mqh"
CChartProperties Chart;//Class object for Chart Properties
#include "RiskManagement.mqh"
CRiskManagement CRisk;//Class object for Risk Management
#include "CommonGraphics.mqh"
CCommonGraphics *CGraphics;//Class pointer object for Common Graphics
CCandleProperties *CP;//Class pointer object for Candle Properties
#include "TradeManagement.mqh"
CTradeManagement Trade;//Class object for Trade Management

//--- used to separate Input Menu
enum iSeparator
  {
   Delimiter//__________________________
  };

//--- for chart color Mode selection
enum DisplayMode
  {
   Display_LightMode,//LIGHT MODE
   Display_DarkMode//DARK MODE
  };

sinput group "+--------|   DISPLAY   |--------+";
sinput DisplayMode iDisplayMode=Display_LightMode;//CHART COLOUR MODE
sinput Choice iDisplay_NewsInfo=Yes;//DISPLAY NEWS INFO
sinput Choice iDisplay_EventObj=Yes;//DISPLAY EVENT OBJ
sinput Choice iDisplay_Spread=Yes;//DISPLAY SPREAD RATING
sinput Choice iDisplay_Date=Yes;//DISPLAY DATE
sinput group "";
sinput group "+--------|   DST SCHEDULE   |--------+";
input DSTSchedule ScheduleDST=AutoDst_Selection;//SELECT DST OPTION
sinput iSeparator iCustomSchedule=Delimiter;//__________________________
sinput iSeparator iCustomScheduleL=Delimiter;//CUSTOM DST
input DST_type CustomSchedule=DST_NONE;//SELECT CUSTOM DST
sinput group "";
sinput group "+--------| RISK MANAGEMENT |--------+";
input RiskOptions RISK_Type=MINIMUM_LOT;//SELECT RISK OPTION
input RiskFloor RISK_Mini=RiskFloorMin;//RISK FLOOR
input double RISK_Mini_Percent=75;//MAX-RISK [100<-->0.01]%
input RiskCeil  RISK_Maxi=RiskCeilMax;//RISK CEILING
sinput iSeparator iRisk_1=Delimiter;//__________________________
sinput iSeparator iRisk_1L=Delimiter;//PERCENTAGE OF [BALANCE | FREE-MARGIN]
input double Risk_1_PERCENTAGE=3;//[100<-->0.01]%
sinput iSeparator iRisk_2=Delimiter;//__________________________
sinput iSeparator iRisk_2L=Delimiter;//AMOUNT PER [BALANCE | FREE-MARGIN]
input double Risk_2_VALUE=1000;//[BALANCE | FREE-MARGIN]
input double Risk_2_AMOUNT=10;//EACH AMOUNT
sinput iSeparator iRisk_3=Delimiter;//__________________________
sinput iSeparator iRisk_3L=Delimiter;//LOTSIZE PER [BALANCE | FREE-MARGIN]
input double Risk_3_VALUE=1000;//[BALANCE | FREE-MARGIN]
input double Risk_3_LOTSIZE=0.1;//EACH LOTS(VOLUME)
sinput iSeparator iRisk_4=Delimiter;//__________________________
sinput iSeparator iRisk_4L=Delimiter;//CUSTOM LOTSIZE
input double Risk_4_LOTSIZE=0.01;//LOTS(VOLUME)
sinput iSeparator iRisk_5=Delimiter;//__________________________
sinput iSeparator iRisk_5L=Delimiter;//PERCENTAGE OF MAX-RISK
input double Risk_5_PERCENTAGE=1;//[100<-->0.01]%
sinput group "";
sinput group "+--------| NEWS SETTINGS |--------+";
input Calendar_Importance iImportance=Calendar_Importance_High;//CALENDAR IMPORTANCE
input Event_Frequency iFrequency=Event_Frequency_ALL;//EVENT FREQUENCY
input Event_Sector iSector=Event_Sector_ALL;//EVENT SECTOR
input Event_Type iType=Event_Type_Indicator;//EVENT TYPE
input Event_Currency iCurrency=Event_Currency_Symbol;//EVENT CURRENCY
sinput group "";
sinput group "+--------| TRADE SETTINGS |--------+";
input uint iStoploss=500;//STOPLOSS [0=NONE]
input uint iTakeprofit=500;//TAKEPROFIT [0=NONE]
input uint iSecondsPreEvent=5;//PRE-ENTRY SEC
input DayOfTheWeek TradingDay=AllDays;//TRADING DAY OF WEEK
sinput group "";
//--- to keep track of start-up time
datetime Startup_date;
```

In the OnInit Integer function below we go through different procedures when set up the expert to trade whether in the strategy tester or not.

```
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//--- Assign if in LightMode or not
   isLightMode=(iDisplayMode==Display_LightMode)?true:false;
//--- call function for common initialization procedure
   InitCommon();
//--- store Init result
   int InitResult;
   if(!MQLInfoInteger(MQL_TESTER))//Checks whether the program is in the strategy tester
     {
      //--- initialization procedure outside strategy tester
      InitResult=InitNonTester();
     }
   else
     {
      //--- initialization procedure inside strategy tester
      InitResult=InitTester();
     }
//--- Create DB in memory
   NewsObject.CreateEconomicDatabaseMemory();
//--- Initialize Common graphics class pointer object
   CGraphics = new CCommonGraphics(Answer(iDisplay_Date),Answer(iDisplay_Spread),Answer(iDisplay_NewsInfo),Answer(iDisplay_EventObj));
   CGraphics.GraphicsRefresh(iSecondsPreEvent);//-- Create chart objects
//--- Initialize Candle properties pointer object
   CP = new CCandleProperties();
//--- Store start-up time.
   Startup_date = TimeTradeServer();
//--- return Init result
   return InitResult;
  }
```

In the function below, we initialize properties for both the strategy tester and the normal trading environment.

```
//+------------------------------------------------------------------+
//|function for common initialization procedure                      |
//+------------------------------------------------------------------+
void InitCommon()
  {
//Initializing CRiskManagement variable for Risk options
   RiskProfileOption = RISK_Type;
//Initializing CRiskManagement variable for Risk floor
   RiskFloorOption = RISK_Mini;
//Initializing CRiskManagement variable for RiskFloorMax
   RiskFloorPercentage = (RISK_Mini_Percent>100)?100:
                         (RISK_Mini_Percent<0.01)?0.01:RISK_Mini_Percent;//Percentage cannot be more than 100% or less than 0.01%
//Initializing CRiskManagement variable for Risk ceiling
   RiskCeilOption = RISK_Maxi;
//Initializing CRiskManagement variable for Risk options (PERCENTAGE OF BALANCE and PERCENTAGE OF FREE-MARGIN)
   Risk_Profile_1 = (Risk_1_PERCENTAGE>100)?100:
                    (Risk_1_PERCENTAGE<0.01)?0.01:Risk_1_PERCENTAGE;//Percentage cannot be more than 100% or less than 0.01%
//Initializing CRiskManagement variables for Risk options (AMOUNT PER BALANCE and AMOUNT PER FREE-MARGIN)
   Risk_Profile_2.RiskAmountBoF = Risk_2_VALUE;
   Risk_Profile_2.RiskAmount = Risk_2_AMOUNT;
//Initializing CRiskManagement variables for Risk options (LOTSIZE PER BALANCE and LOTSIZE PER FREE-MARGIN)
   Risk_Profile_3.RiskLotBoF = Risk_3_VALUE;
   Risk_Profile_3.RiskLot = Risk_3_LOTSIZE;
//Initializing CRiskManagement variable for Risk option (CUSTOM LOTSIZE)
   Risk_Profile_4 = Risk_4_LOTSIZE;
//Initializing CRiskManagement variable for Risk option (PERCENTAGE OF MAX-RISK)
   Risk_Profile_5 = (Risk_5_PERCENTAGE>100)?100:
                    (Risk_5_PERCENTAGE<0.01)?0.01:Risk_5_PERCENTAGE;//Percentage cannot be more than 100% or less than 0.01%
//--- Initializing DST Schedule variables
   MyDST = ScheduleDST;
   MySchedule = CustomSchedule;
//--- Initializing News filter variables
   myFrequency=iFrequency;
   myImportance=iImportance;
   mySector=iSector;
   myType=iType;
   myCurrency=iCurrency;
   Chart.ChartRefresh();//Load chart configurations
  }
```

The function below will initialize for the normal trading environment only.

```
//+------------------------------------------------------------------+
//|function for initialization procedure outside strategy tester     |
//+------------------------------------------------------------------+
int InitNonTester()
  {
//--- Check if in Strategy tester!
   if(MQLInfoInteger(MQL_TESTER))
     {
      //--- Initialization failed.
      return(INIT_SUCCEEDED);
     }
//--- create OBJ_BITMAP_LABEL object for drawing
   ObjectCreate(0,"STATUS",OBJ_BITMAP_LABEL,0,0,0);
   ObjectSetInteger(0,"STATUS",OBJPROP_XDISTANCE,5);
   ObjectSetInteger(0,"STATUS",OBJPROP_YDISTANCE,22);
//--- specify the name of the graphical resource
   ObjectSetString(0,"STATUS",OBJPROP_BMPFILE,"::PROGRESS");
   uint   w,h;          // variables for receiving text string sizes
   uint    x,y;          // variables for calculation of the current coordinates of text string anchor points
   /*
   In the Do while loop below, the code will check if the terminal is connected to the internet.
   If the the program is stopped the loop will break, if the program is not stopped and the terminal
   is connected to the internet the function CreateEconomicDatabase will be called from the News.mqh header file's
   object called NewsObject and the loop will break once called.
   */
   bool done=false;
   do
     {
      //--- clear the drawing buffer array
      ArrayFill(ExtImg,0,IMG_WIDTH*IMG_HEIGHT,0);

      if(!TerminalInfoInteger(TERMINAL_CONNECTED))
        {
         //-- integer dots used as a loading animation
         static int dots=0;
         //--- set the font
         TextSetFont("Arial",-150,FW_EXTRABOLD,0);
         TextGetSize("Waiting",w,h);//get text width and height values
         //--- calculate the coordinates of the 'Waiting' text
         x=10;//horizontal alignment
         y=IMG_HEIGHT/2-(h/2);//alignment for the text to be centered vertically
         //--- output the 'Waiting' text to ExtImg[] buffer
         TextOut("Waiting",x,y,TA_LEFT|TA_TOP,ExtImg,IMG_WIDTH,IMG_HEIGHT,ColorToARGB(CSymbol.Background()),clr_format);
         //--- calculate the coordinates for the dots after the 'Waiting' text
         x=w+13;//horizontal alignment
         y=IMG_HEIGHT/2-(h/2);//alignment for the text to be centered vertically
         TextSetFont("Arial",-160,FW_EXTRABOLD,0);
         //--- output of dots to ExtImg[] buffer
         TextOut(StringSubstr("...",0,dots),x,y,TA_LEFT|TA_TOP,ExtImg,IMG_WIDTH,IMG_HEIGHT,ColorToARGB(CSymbol.Background()),clr_format);
         //--- update the graphical resource
         ResourceCreate("::PROGRESS",ExtImg,IMG_WIDTH,IMG_HEIGHT,0,0,IMG_WIDTH,clr_format);
         //--- force chart update
         Chart.Redraw();
         dots=(dots==3)?0:dots+1;
         //-- Notify user that program is waiting for connection
         Print("Waiting for connection...");
         Sleep(500);
         continue;
        }
      else
        {
         //--- set the font
         TextSetFont("Arial",-120,FW_EXTRABOLD,0);
         TextGetSize("Getting Ready",w,h);//get text width and height values
         x=20;//horizontal alignment
         y=IMG_HEIGHT/2-(h/2);//alignment for the text to be centered vertically
         //--- output the text 'Getting Ready...' to ExtImg[] buffer
         TextOut("Getting Ready...",x,y,TA_LEFT|TA_TOP,ExtImg,IMG_WIDTH,IMG_HEIGHT,ColorToARGB(CSymbol.Background()),clr_format);
         //--- update the graphical resource
         ResourceCreate("::PROGRESS",ExtImg,IMG_WIDTH,IMG_HEIGHT,0,0,IMG_WIDTH,clr_format);
         //--- force chart update
         Chart.Redraw();
         //-- Notify user that connection is successful
         Print("Connection Successful!");
         NewsObject.CreateEconomicDatabase();//calling the database create function
         done=true;
        }
     }
   while(!done&&!IsStopped());
//-- Delete chart object
   ObjectDelete(0,"STATUS");
//-- force chart to update
   Chart.Redraw();
//--- Initialization succeeded.
   return(INIT_SUCCEEDED);
  }
```

The function below will initialize for the strategy tester environment only.

```
//+------------------------------------------------------------------+
//|function for initialization procedure inside strategy tester      |
//+------------------------------------------------------------------+
int InitTester()
  {
//--- Check if not in Strategy tester!
   if(!MQLInfoInteger(MQL_TESTER))
     {
      //--- Initialization failed.
      return(INIT_SUCCEEDED);
     }
//Checks whether the database file exists
   if(!FileIsExist(NEWS_DATABASE_FILE,FILE_COMMON))
     {
      //--- Warning messages
      Print("Necessary Files Do not Exist!");
      Print("Run Program outside of the Strategy Tester");
      Print("Necessary Files Should be Created First");
      //--- Initialization failed.
      return(INIT_FAILED);
     }
   else
     {
      //Checks whether the latest database date includes the time and date being tested
      datetime latestdate = CTM.TimeMinusOffset(NewsObject.GetLatestNewsDate(),CTM.DaysS());//Day before the latest recorded time in the database
      if(latestdate<TimeTradeServer())
        {
         Print("Necessary Files outdated!");
         Print("To Update Files: Run Program outside of the Strategy Tester");
        }
      Print("Database Dates End at: ",latestdate);
      PrintFormat("Dates after %s will not be available for backtest",TimeToString(latestdate));
     }
//--- Initialization succeeded.
   return(INIT_SUCCEEDED);
  }
```

The function below will run on every new tick.

```
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//--- Run procedures
   Execution();
  }
```

In this function below, we call all the functions that will make the expert functional. Not every function needs to run on every tick so we only call certain functions on every new certain candle for example every daily candle. This helps with performance and reduces unnecessary function calls and use of resources.

```
//+------------------------------------------------------------------+
//|Execute program procedures                                        |
//+------------------------------------------------------------------+
void Execution()
  {
//--- Update realtime Graphic every 1 min
   if(CP.NewCandle(1,PERIOD_M1))
     {
      CGraphics.Block_2_Realtime(iSecondsPreEvent);
     }
//--- function to open trades
   EnterTrade();
//--- Check if not start-up date
   if(!CTM.DateisToday(Startup_date))
     {
      //--- Run every New Daily Candle
      if(CP.NewCandle(2,PERIOD_D1))
        {
         //--- Check if not in strategy tester
         if(!MQLInfoInteger(MQL_TESTER))
           {
            //--- Update/Create DB in Memory
            NewsObject.CreateEconomicDatabaseMemory();
           }
         CGraphics.GraphicsRefresh(iSecondsPreEvent);//-- Create/Re-create chart objects
         //--- Update Realtime Graphics
         CGraphics.Block_2_Realtime(iSecondsPreEvent);
        }
      //--- Check if not in strategy tester
      if(!MQLInfoInteger(MQL_TESTER))
        {
         //--- Run every New Hourly Candle
         if(CP.NewCandle(3,PERIOD_H1))
           {
            //--- Check if DB in Storage needs an update
            if(NewsObject.UpdateRecords())
              {
               //--- initialization procedure outside strategy tester
               InitNonTester();
              }
           }
        }
     }
   else
     {
      //--- Run every New Daily Candle
      if(CP.NewCandle(4,PERIOD_D1))
        {
         //--- Update Event objects on chart
         CGraphics.NewsEvent();
        }
     }
  }
```

The function below will be responsible for opening trades for Market orders based of the event impact and event currency. If the event currency is equal to the Profit currency and the impact type is CALENDAR\_IMPACT\_NEGATIVE we open a buy trade as we assume that the Profit currency will weaken during the news event, if the event currency is equal to the Profit currency and the impact type is CALENDAR\_IMPACT\_POSITIVE we open a sell trade as we assume that the Profit currency will strengthen during the news event.

```
//+------------------------------------------------------------------+
//|function to open trades                                           |
//+------------------------------------------------------------------+
void EnterTrade()
  {
//--- static variable for storing upcoming event Impact value
   static ENUM_CALENDAR_EVENT_IMPACT Impact=CALENDAR_IMPACT_NA;
//--- Check if Upcoming news date has passed and if upcoming news is not null and if new minute candle has formed.
   if(datetime(UpcomingNews.EventDate)<TimeTradeServer()&&UpcomingNews.CountryName!=NULL&&CP.NewCandle(5,PERIOD_M1))
     {
      //--- Update for next upcoming news
      NewsObject.EconomicNextEvent();
      //--- Get impact value for upcoming news
      Impact=NewsObject.GetImpact();
     }
//--- Check if upcoming news date is about to occur and if it is the trading day of week
   if(CTM.TimePreEvent(CTM.TimeMinusOffset(datetime(UpcomingNews.EventDate),(iSecondsPreEvent==0)?1:iSecondsPreEvent)
                       ,datetime(UpcomingNews.EventDate))
      &&CTM.isDayOfTheWeek(TradingDay))
     {
      //--- Check each Impact value type
      switch(Impact)
        {
         //--- When Impact news is negative
         case CALENDAR_IMPACT_NEGATIVE:
            //--- Check if profit currency is news event currency
            if(UpcomingNews.EventCurrency==CSymbol.CurrencyProfit())
              {
               //--- Open buy trade with Event id as Magic number
               Trade.Buy(iStoploss,iTakeprofit,ulong(UpcomingNews.EventId),"NewsTrading");
              }
            else
              {
               //--- Open sell trade with Event id as Magic number
               Trade.Sell(iStoploss,iTakeprofit,ulong(UpcomingNews.EventId),"NewsTrading");
              }
            break;
         //--- When Impact news is positive
         case CALENDAR_IMPACT_POSITIVE:
            //--- Check if profit currency is news event currency
            if(UpcomingNews.EventCurrency==CSymbol.CurrencyProfit())
              {
               //--- Open sell trade with Event id as Magic number
               Trade.Sell(iStoploss,iTakeprofit,ulong(UpcomingNews.EventId),"NewsTrading");
              }
            else
              {
               //--- Open buy trade with Event id as Magic number
               Trade.Buy(iStoploss,iTakeprofit,ulong(UpcomingNews.EventId),"NewsTrading");
              }
            break;
         //--- Unknown
         default:
            break;
        }
     }
  }
```

### Conclusion

In this article, we went through adding a database in memory and creating additional views to provide more information on the events in the MQL5 economic calendar. We created additional graphical objects on the chart to display information about the upcoming event, as well as implemented a dark mode feature. Furthermore, we added relevant input options for the user/trader to filter news data for their preferences as well as Expert Advisor inputs for trading. Also, the article provides an explanation of how we open market orders based on the event impact and how the event impact is relevant to our trading strategy.

I'm open to hearing from you and any shared opinions are appreciated. In the next article, we will add more functionality to the news inputs to cater for individual economic events and trading using pending orders for more flexibility, as well as trading that doesn't require event impact. Thanks for reading!

Video

NewsTradingPart3 - YouTube

[Photo image of MQL5.community](https://www.youtube.com/channel/UC8bwZMk2yh5WDToFos49OjQ?embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F15359)

MQL5.community

1.91K subscribers

[NewsTradingPart3](https://www.youtube.com/watch?v=pysrqm-HbnY)

MQL5.community

Search

Watch later

Share

Copy link

Info

Shopping

Tap to unmute

If playback doesn't begin shortly, try restarting your device.

More videos

## More videos

You're signed out

Videos you watch may be added to the TV's watch history and influence TV recommendations. To avoid this, cancel and sign in to YouTube on your computer.

CancelConfirm

Share

Include playlist

An error occurred while retrieving sharing information. Please try again later.

[Watch on](https://www.youtube.com/watch?v=pysrqm-HbnY&embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F15359)

0:00

0:00 / 17:46

•Live

•

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/15359.zip "Download all attachments in the single ZIP archive")

[NewsTrading\_Part3.zip](https://www.mql5.com/en/articles/download/15359/newstrading_part3.zip "Download NewsTrading_Part3.zip")(567.27 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [News Trading Made Easy (Part 6): Performing Trades (III)](https://www.mql5.com/en/articles/16170)
- [News Trading Made Easy (Part 5): Performing Trades (II)](https://www.mql5.com/en/articles/16169)
- [News Trading Made Easy (Part 4): Performance Enhancement](https://www.mql5.com/en/articles/15878)
- [News Trading Made Easy (Part 2): Risk Management](https://www.mql5.com/en/articles/14912)
- [News Trading Made Easy (Part 1): Creating a Database](https://www.mql5.com/en/articles/14324)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/471336)**
(3)


![laurhaq](https://c.mql5.com/avatar/avatar_na2.png)

**[laurhaq](https://www.mql5.com/en/users/laurhaq)**
\|
13 Sep 2024 at 21:44

Thank you very much

Laurent

![Jefferson Judge Metha](https://c.mql5.com/avatar/2021/2/6017F3F5-4887.jpg)

**[Jefferson Judge Metha](https://www.mql5.com/en/users/jeffiq)**
\|
23 Oct 2024 at 20:33

Thank you for the Article very Good.

However, I do not want to Visualise this

![](https://c.mql5.com/3/446/6301960240973.png)

The Comments I do not want to Visualise them.


![Kabelo Frans Mampa](https://c.mql5.com/avatar/2023/1/63bd510f-63d8.png)

**[Kabelo Frans Mampa](https://www.mql5.com/en/users/kaaiblo)**
\|
24 Oct 2024 at 08:07

**Jefferson Judge Metha [#](https://www.mql5.com/en/forum/471336#comment_54917758):**

Thank you for the Article very Good.

However, I do not want to Visualise this

The Comments I do not want to Visualise them.

Hi, feel free to modify the code as you please. If you need assistance I'm here to help.


![Developing a robot in Python and MQL5 (Part 1): Data preprocessing](https://c.mql5.com/2/74/Robot_development_in_Python_and_MQL5_oPart_1z_Data_preprocessing____LOGO.png)[Developing a robot in Python and MQL5 (Part 1): Data preprocessing](https://www.mql5.com/en/articles/14350)

Developing a trading robot based on machine learning: A detailed guide. The first article in the series deals with collecting and preparing data and features. The project is implemented using the Python programming language and libraries, as well as the MetaTrader 5 platform.

![Creating an MQL5-Telegram Integrated Expert Advisor (Part 2): Sending Signals from MQL5 to Telegram](https://c.mql5.com/2/88/logo-Creating_an_MQL5-Telegram_Integrated_Expert_Advisor_sPart_1u.png)[Creating an MQL5-Telegram Integrated Expert Advisor (Part 2): Sending Signals from MQL5 to Telegram](https://www.mql5.com/en/articles/15495)

In this article, we create an MQL5-Telegram integrated Expert Advisor that sends moving average crossover signals to Telegram. We detail the process of generating trading signals from moving average crossovers, implementing the necessary code in MQL5, and ensuring the integration works seamlessly. The result is a system that provides real-time trading alerts directly to your Telegram group chat.

![Population optimization algorithms: Boids Algorithm](https://c.mql5.com/2/74/Population_optimization_algorithms_Boyd_algorithmp_or_flock_algorithm___LOGO.png)[Population optimization algorithms: Boids Algorithm](https://www.mql5.com/en/articles/14576)

The article considers Boids algorithm based on unique examples of animal flocking behavior. In turn, the Boids algorithm serves as the basis for the creation of the whole class of algorithms united under the name "Swarm Intelligence".

![Developing a multi-currency Expert Advisor (Part 6): Automating the selection of an instance group](https://c.mql5.com/2/74/Developing_a_multi-currency_advisor_Part_1___LOGO__4.png)[Developing a multi-currency Expert Advisor (Part 6): Automating the selection of an instance group](https://www.mql5.com/en/articles/14478)

After optimizing the trading strategy, we receive sets of parameters. We can use them to create several instances of trading strategies combined in one EA. Previously, we did this manually. Here we will try to automate this process.

[What's wrong with regular VPS?Here are the 8 most common problems that algorithmic traders may encounterRead![](https://www.mql5.com/ff/sh/hzatb686qjqxwtr4z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/450486&a=drhremihlwuaqyvgpzfddbtmgciejpba&s=c37d25bcceb93ed153b814e6ba4d4839461a9b2d68dd82b95b142be06d310f3f&uid=&ref=https://www.mql5.com/en/articles/15359&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5083470957718281308)

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