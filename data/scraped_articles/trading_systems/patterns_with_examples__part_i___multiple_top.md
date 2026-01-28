---
title: Patterns with Examples (Part I): Multiple Top
url: https://www.mql5.com/en/articles/9394
categories: Trading Systems, Indicators, Expert Advisors
relevance_score: 1
scraped_at: 2026-01-23T21:38:28.306545
---

[![](https://www.mql5.com/ff/sh/6xjc81sb5f2g45z9z2/01.png)Follow MQL5.community on social mediaWe publish the best technical materials from experts – free from advertising and irrelevant contentLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=yexgeaiatphxecqagtoxizolvboismyb&s=4e531fd1f983c26570e2dac7588b735354f2f9e0aea561427c030e4a1d2f060b&uid=&ref=https://www.mql5.com/en/articles/9394&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5071985240211141154)

MetaTrader 5 / Trading systems


### Contents

- [Abstract](https://www.mql5.com/en/articles/9394#para1)
- [About the reversal patterns](https://www.mql5.com/en/articles/9394#para2)
- [Why Multiple Top — its specific features](https://www.mql5.com/en/articles/9394#para3)
- [Can the Double Top concept be extended?](https://www.mql5.com/en/articles/9394#para4)
- [Writing code to render Multiple Top](https://www.mql5.com/en/articles/9394#para5)
  - [Class map](https://www.mql5.com/en/articles/9394#u1)
  - [Identifying tops and bottoms](https://www.mql5.com/en/articles/9394#u2)
  - [Selecting tops to work with](https://www.mql5.com/en/articles/9394#u3)
  - [Determining the pattern direction](https://www.mql5.com/en/articles/9394#u4)
  - [Filters for discarding invalid patterns](https://www.mql5.com/en/articles/9394#u5)
  - [Checking the result in the MetaTrader 5 Strategy Tester visualizer](https://www.mql5.com/en/articles/9394#u6)
- [Further development ideas](https://www.mql5.com/en/articles/9394#para6)
- [Conclusion](https://www.mql5.com/en/articles/9394#para7)

### Abstract

Patterns are often discussed on the internet, because they are used by many traders. Patterns can be referred to as visual analysis criteria for determining the direction of the pricing that follows. Algo trading is different from that. There cannot be visual criteria for algorithmic trading. Expert Advisors and indicators have individual methods for working with the price series. There are advantages and disadvantages at both ends. The code lacks the breadth of human thinking and the quality of human analysis, but the code has other valuable advantages: incomparable speed and incomparable amount of numerical or logical data processed per unit of time. It is not easy to instruct the machine what to do. This takes some practice. Over time, the programmer begins to understand the machine, and the machine begins to understand the programmer. This series of articles will be useful for beginners, who will learn how to structure their thoughts and to split complex tasks into simpler steps.

### About the reversal patterns

For me personally, reversal patterns have a too vague definition. Furthermore, they do not have any underlying mathematics. To be honest, any pattern has no underlaying mathematics and thus the only mathematics which can be considered here is statistics. Statistics are the only criteria for truth, but statistics are compiled based on the real trading. Obviously, there are no sources which can provide very precise statistics. There is even no point in providing such data for one specific research problem. The only solution here is backtesting and visualization in the strategy tester. Although the approach offers lower data quality, it has an undeniable advantage which is speed along with data amount.

Of course, reversal patterns do not serve as a sufficient tool for determining trend reversals, but in combination with other methods of analysis, such as levels or candlestick analysis, they can produce the desired result. Within this series of article, the patterns are not considered as a specifically interesting method of analysis, but they can be used for practicing algorithmic trading skills. In addition to practicing, you will obtain an interesting and useful auxiliary tool - if not for algo-trading, then for the trader eye. Useful indicators are greatly valued.

### Why Multiple Top — its specific features

This pattern has become quite popular on the internet due to its simplicity. The pattern is quite common on different trading instruments and on various chart timeframes, simply because there is nothing complicated about it. Furthermore, if you look closer at the pattern, you can see that the method concept can be expanded by using algo-trading and MQL5 language capabilities. We can try to create some general code which will not be limited only by a double top. A wisely created prototype can be used for exploring all pattern hybrids and successors.

The classic successor to the multiple top is the very popular "Head and Shoulders" pattern. Unfortunately, there is no structured information on how to trade this pattern. This problem is common for a lot of popular strategies - because there are many beautiful words but no statistics. I will try to understand in this article whether it is possible to use them in the framework of algorithmic trading. The only method to collect statistics without trading on a demo or real account is to use the capabilities of the strategy tester. Without this tool, you will not be able to draw any complex conclusions regarding a particular strategy.

### Can the Double Top concept be extended?

Regarding the topic of the article, I will try to draw a diagram as a tree of patterns that starts from a double top. This will assist in understanding how broad the possibilities of this concept are:

![Tree](https://c.mql5.com/2/42/yj84lt7jt1_vs6jwn5_0o0h40p.png)

I decided to combine the concept of several patterns with the assumption that they are based on approximately the same idea. This idea has a simple beginning - find a good movement in any direction and correctly determine the place where it is supposed to reverse. After visual contact with the proposed pattern, the trader should correctly draw some auxiliary lines, which should assist in evaluating whether the pattern meets certain criteria as well as in determining the market entry point along with the target and stop loss level. Take profit can be used here instead of the target.

Patterns can have some common construction principles, based on which the concept of these patterns can be combined. Such clear definition is what differs algorithmic traders from manual traders. Uncertainty and multiple interpretation of the same principles may lead to disappointing consequences.

The basic patters are as follows:

1. Double Top
2. Triple Top
3. Head and Shoulders

These patterns have similar structures and usage principles. All of them are aimed at identifying reversals. All three patterns have a similar logic regarding auxiliary lines. Please consider an example of the Double Top:

![Double extremum](https://c.mql5.com/2/42/9wtl6sy_2o2jchk.png)

In the above figure, all the required lines are numbered and mean the following:

1. Trend resistance
2. Auxiliary line for defining a pessimistic peak (someone thinks it is a neck)
3. Neck line
4. Optimistic target (it is also a take profit level for trading)
5. The maximum allowable stop-loss level (it is set at the far top)
6. Optimistic forecast line (equal to the previous trend movement)

A pessimistic target is determined relative to the point of the neck line intersection from the edge which is nearest to the market - we take the distance between "1" and "2", which is indicated as **"t"**, and measure the same distance in the direction of the proposed reversal. The minimum of the optimistic target is determined in a similar way, but the distance is measured between "5" and "3", which is indicated as **"s"**.

### Writing code to render Multiple Top

Let us begin by defining the reasoning logic to define these patterns. In order to find a pattern, we should stick to the bar-by-bar logic, that is, we will work not by ticks, but by bars. In this case, it will greatly reduce the load on the terminal as this will avoid unnecessary calculations. First, let us determine a class symbolizing some independent observer who will look for the pattern. All operations required for a correct pattern detection will be part of the instance, so search will be performed inside it. I have chosen this solution in order to enable further code modifications, for example, when we need to expand the functionality or to modify existing features.

**Class map**

Let's start with considering the class contents:

```
class ExtremumsPatternFamilySearcher// class simulating an independent pattern search
   {
   private:
   int BarsM;// how many bars on chart to use
   int MinimumSeriesBarsM;// the minimum number of bars in a row to detect a top
   int TopsM;// number of tops in the pattern
   int PointsPessimistM;// minimum distance in points to the nearest target
   double RelativeUnstabilityM;// maximum excess of the head size relative to the minimum shoulder
   double RelativeUnstabilityMinM;// minimum excess of the head size relative to the minimum shoulder
   double RelativeUnstabilityTimeM;// maximum excess of head and shoulders sizes
   bool bAbsolutelyHeadM;// whether a pronounced head is required
   bool bRandomExtremumsM;// random selection of extrema


   struct Top// top data
      {
      datetime Datetime0;// time of the candlestick closest to the market
      datetime Datetime1;// time of the next candlestick
      int Index0;// index of the candlestick closest to the market
      int Index1;// index of the next candlestick
      datetime DatetimeExtremum;// time of the top
      int IndexExtremum;// index of the top
      double Price;// price of the top
      bool bActive;// if the top is active (if not, then it does not exist)
      };

   struct Line// line
      {
      double Price0;// price of the candlestick closest to the market, to which the line is bound
      datetime Time0;// time of the candlestick closest to the market, to which the line is bound
      double Price1;// price of the farthest candlestick to which the line is bound
      datetime Time1;// time of the farthest candlestick to which the line is bound
      datetime TimeX;// time of the X point
      int Index1;// index of the left edge
      bool DirectionOfFormation;// direction
      double C;// free coefficient in the equation
      double K;// aspect ratio

      void CalculateKC()// find unknowns in the equation
         {
         if ( Time0 != Time1 ) K=double(Price0-Price1)/double(Time0-Time1);
         else K=0.0;
         C=double(Price1)-K*double(Time1);
         }

      double Price(datetime T)// function of line depending on time
         {
         return K*T+C;
         }
      };

   public:

   ExtremumsPatternFamilySearcher(int BarsI,int MinimumSeriesBarsI,int TopsI,int PointsPessimistI, double RelativeUnstabilityI,
   double RelativeUnstabilityMinI,double RelativeUnstabilityTimeI,bool bAbsolutelyHeadI,bool bRandomExtremumsI)// parametric constructor
      {
      BarsM=BarsI;
      MinimumSeriesBarsM=MinimumSeriesBarsI;
      TopsM=TopsI;
      PointsPessimistM=PointsPessimistI;
      RelativeUnstabilityM=RelativeUnstabilityI;
      RelativeUnstabilityMinM=RelativeUnstabilityMinI;
      RelativeUnstabilityTimeM=RelativeUnstabilityTimeI;
      bAbsolutelyHeadM=bAbsolutelyHeadI;
      bRandomExtremumsM=bRandomExtremumsI;
      bPatternFinded=bFindPattern();
      }

   int FormationDirection;// direction of the formation (multiple top or bottom, or none at all) ( -1,1,0 )
   bool bPatternFinded;// if the pattern was found during formation
   Top TopsUp[];// required upper extrema
   Top TopsDown[];// required lower extrema
   Top TopsUpAll[];// all upper extrema
   Top TopsDownAll[];// all lower extrema
   int RandomIndexUp[];// array for the random selection of the tops index
   int RandomIndexDown[];// array for the random selection of the bottoms index
   Top StartTop;// where the formation starts (top farthest from the market)
   Top EndTop;// where the formation ends (top closest to the market)
   Line Neck;// neck
   Top FarestTop;// top farthest from the neck (will be used to determine the head or the formation size) or the same as the head
   Line OptimistLine;// line of optimistic forecast
   Line PessimistLine;// line of pessimistic forecast
   Line BorderLine;// line at the edge of the pattern
   Line ParallelLine;// line parallel to the trend resistance


   private:
   void SetTopsSize();// setting sizes for arrays with tops
   bool SearchFirstUps();// search for tops
   bool SearchFirstDowns();// search for bottoms
   void CalculateMaximum(Top &T,int Index0,int Index1);// calculate the maximum price between two bars
   void CalculateMinimum(Top &T,int Index0,int Index1);// calculate the minimum price between two bars
   bool PrepareExtremums();// prepare extrema
   bool IsExtremumsAbsolutely();// control the priority of tops
   void DirectionOfFormation();// determine the direction of the formation
   void FindNeckUp(Top &TStart,Top &TEnd);// find neck for the bullish pattern
   void FindNeckDown(Top &TStart,Top &TEnd);// find neck for the bearish pattern
   void SearchFarestTop();// find top farthest from the neck
   bool bBalancedExtremums();// initial balancing of extrema (so that they do not differ much)
   bool bBalancedExtremumsHead();// if a pattern has more than 2 tops, we can check for a pronounced head
   bool bBalancedExtremumsTime();// require that the extrema be not very far in time relative to the minimum distance
   bool bBalancedHead();// balance the head (in other words, require that it be neither the first nor the last one on the list of tops, if there are more than three of them)
   bool CorrectNeckUpLeft();// adjust the neck so as to find the intersection of price and neck (this creates prerequisites for the previous trend)
   bool CorrectNeckDownLeft();// similarly for the bottom
   int CorrectNeckUpRight();// adjust the neck so as to find the intersection of price and neck on the right or at the current price position, which is the same (to determine the entry point)
   int CorrectNeckDownRight();// similarly for the bottom
   void SearchLineOptimist();// calculate the optimistic forecast line
   bool bWasTrend();// determine whether a trend preceded the pattern definition (in this case the optimistic target line is considered as the trend beginning)
   void SearchLineBorder();// determine trend resistance or support (usually a sloping line)
   void CalculateParallel();// determine a line parallel to support or resistance (crosses the neck at the pattern low or high)
   bool bCalculatePessimistic();// calculate the line of the pessimistic target
   bool bFindPattern();// perform all the above actions
   int iFindEnter();// find intersection with the neck
   public:
   void CleanAll();// clean up objects
   void DrawPoints();// draw points
   void DrawNeck();// draw the neck
   void DrawLineBorder();// line at the border
   void DrawParallel();// line parallel to the border
   void DrawOptimist();// line of optimistic forecast
   void DrawPessimist();// line of pessimistic forecast
   };
```

A class represents sequential operations which a person would perform if the person were in the place of a machine. Anyway, the detection of any formation can be split into a set of simple operations that follow one another. There is a rule in mathematics: if you don't know how to solve an equation, simplify it. This rule applies not only to mathematics, but also to any algorithm. The detection logic is not clear first. But if you know where to start detection, the task becomes much simpler. In this case, in order to find the whole pattern, we search for either tops or bottoms, or actually both.

**Determining tops and bottoms**

Without tops and bottoms, the whole pattern is meaningless, since the presence of tops and bottoms is a required condition for the pattern, although this condition alone is not enough. There are different ways to determining tops. The most important condition is the presence of a pronounced half-wave, while the half-wave is determined by two pronounced opposite movements, which in our case should be several bars in a row, in one direction. For this purpose, we need to determine the minimum number of bars in one direction, which indicate the presence of movement. For this, let's provide an input variable.

```
bool ExtremumsPatternFamilySearcher::SearchFirstUps()// find tops
   {
   int NumUp=0;// the number of found tops
   int NumDown=0;// the number of found bottoms
   bool bDown=false;// an auxiliary boolean which shows if a segment of bearish candlesticks has been found
   bool bUp=false;// an auxiliary boolean which shows if a segment of bullish candlesticks has been found
   bool bNextUp=true;// can we move on to searching for the next top
   bool bNextDown=true;// can we move on to searching for the next bottom

   for(int i=0;i<ArraySize(TopsUp);i++)// before search, set all necessary tops to an inactive state
      {
      TopsUp[i].bActive=false;
      }
   for(int i=0;i<ArraySize(TopsUpAll);i++)// before search, set all tops to an inactive state
      {
      if (!TopsUpAll[i].bActive) break;
      TopsUpAll[i].bActive=false;
      }


   for(int i=0;i<BarsM;i++)
      {
      if ( i+MinimumSeriesBarsM-1 < BarsM )// if remaining bars are enough to determine the extremum and we can start searching for the next top
         {
         if ( bNextUp )// if it is allowed to search for the next top
            {
            bDown=true;
            for(int j=i;j<i+MinimumSeriesBarsM;j++)// determine the first extrema for upper tops
               {
               if ( Open[j]-Close[j] < 0 )// if at least one of the selected candlesticks was upward
                  {
                  bDown=false;
                  break;
                  }
               }
            if ( bDown )
               {
               TopsUpAll[NumUp].Datetime0=Time[i+MinimumSeriesBarsM-1];
               TopsUpAll[NumUp].Index0=i+MinimumSeriesBarsM-1;
               bNextUp=false;
               }
            }
         }

      if ( MinimumSeriesBarsM+i < BarsM && bDown )// if the remaining bars are enough to determine the second half of the extremum and the previous half has been found
         {
         bUp=true;
         for(int j=i;j<MinimumSeriesBarsM+i;j++)//determine further candlesticks in the opposite direction
            {
            if ( Open[j]-Close[j] > 0 )//if at least one of the selected candlesticks was downward
               {
               bUp=false;
               break;
               }
            }
         if ( bUp )
            {
            TopsUpAll[NumUp].Datetime1=Time[i];
            TopsUpAll[NumUp].Index1=i;
            TopsUpAll[NumUp].bActive=true;
            bNextUp=false;
            }
         }
      // after that, register the found formation as a top, if it is a top
      if ( bDown && bUp )
         {
         CalculateMaximum(TopsUpAll[NumUp],TopsUpAll[NumUp].Index0,TopsUpAll[NumUp].Index1);// calculate extremum between two bars
         bNextUp=true;
         bDown=false;
         bUp=false;
         NumUp++;
         }
      }
   if ( NumUp >= TopsM ) return true;// if the required number of tops have been found
   else return false;
   }
```

Bottoms are defined in the opposite way:

```
bool ExtremumsPatternFamilySearcher::SearchFirstDowns()// find bottoms
   {
   int NumUp=0;
   int NumDown=0;
   bool bDown=false;// an auxiliary boolean which shows if a segment of bearish candlesticks has been found
   bool bUp=false;// an auxiliary boolean which shows if a segment of bullish candlesticks has been found
   bool bNextUp=true;// can we move on to searching for the next top
   bool bNextDown=true;// can we move on to searching for the next bottom

   for(int i=0;i<ArraySize(TopsDown);i++)// before search, set all necessary bottoms to an inactive state
      {
      TopsDown[i].bActive=false;
      }
   for(int i=0;i<ArraySize(TopsDownAll);i++)// before search, set all bottoms to an inactive state
      {
      if (!TopsDownAll[i].bActive) break;
      TopsDownAll[i].bActive=false;
      }

   for(int i=0;i<BarsM;i++)
      {
      if ( i+MinimumSeriesBarsM-1 < BarsM )// if remaining bars are enough to determine the extremum and we can start searching for the next top
         {
         if ( bNextDown )// if it is allowed to search for the next bottom
            {
            bUp=true;
            for(int j=i;j<i+MinimumSeriesBarsM;j++)// determine the first extrema for upper tops
               {
               if ( Open[j]-Close[j] > 0 )//if at least one of the selected candlesticks was downward
                  {
                  bUp=false;
                  break;
                  }
               }
            if ( bUp )
               {
               TopsDownAll[NumDown].Datetime0=Time[i+MinimumSeriesBarsM-1];
               TopsDownAll[NumDown].Index0=i+MinimumSeriesBarsM-1;
               bNextDown=false;
               }
            }
         }

      if ( MinimumSeriesBarsM+i < BarsM && bUp )// if the remaining bars are enough to determine the second half of the extremum and the previous half has been found
         {
         bDown=true;
         for(int j=i;j<MinimumSeriesBarsM+i;j++)//determine further candlesticks in the opposite direction
            {
            if ( Open[j]-Close[j] < 0 )// if at least one of the selected candlesticks was upward
               {
               bDown=false;
               break;
               }
            }
         if ( bDown )
            {
            TopsDownAll[NumDown].Datetime1=Time[i];
            TopsDownAll[NumDown].Index1=i;
            TopsDownAll[NumDown].bActive=true;
            bNextDown=false;
            }
         }
      // after that, register the found formation as a bottom, if it is a bottom
      if ( bDown && bUp )
         {
         CalculateMinimum(TopsDownAll[NumDown],TopsDownAll[NumDown].Index0,TopsDownAll[NumDown].Index1);// calculate extremum between two bars
         bNextDown=true;
         bDown=false;
         bUp=false;
         NumDown++;
         }
      }

   if ( NumDown == TopsM ) return true;//if the required number of bottoms have been found
   else return false;
   }
```

In this case I didn't use the logic of fractals. Instead, I created my own logic for determining tops and bottoms. I don't think it's better or worse than fractals, but at least there is no need to use any external functionality. Furthermore, there is no need to use unnecessary built-in language functions, which sometimes are not necessary. These functions might be good, but in this case they are redundant. The function determines all tops and bottoms, with which we will work in the future. The following image provides a visual representation of what is happening in this function:

![Searching for tops & bottoms](https://c.mql5.com/2/42/iz3ordjgzl_zcobj5.png)

First, it searches for movements 1; then it searches for movement 2, and finally 3 implies determining of the top or bottom. Logic for 3 is implemented in two separate functions that look like this:

```
void ExtremumsPatternFamilySearcher::CalculateMaximum(Top &T,int Index0,int Index1)// if 2 intermediate points are found, find High between them
   {
   double MaxValue=High[Index0];
   datetime MaxTime=Time[Index0];
   int MaxIndex=Index0;
   for(int i=Index0;i<=Index1;i++)
      {
      if ( High[i] >  MaxValue )
         {
         MaxValue=High[i];
         MaxTime=Time[i];
         MaxIndex=i;
         }
      }
   T.DatetimeExtremum=MaxTime;
   T.IndexExtremum=MaxIndex;
   T.Price=MaxValue;
   }

void ExtremumsPatternFamilySearcher::CalculateMinimum(Top &T,int Index0,int Index1)//if 2 intermediate points are found, find Low between them
   {
   double MinValue=Low[Index0];
   datetime MinTime=Time[Index0];
   int MinIndex=Index0;
   for(int i=Index0;i<=Index1;i++)
      {
      if ( Low[i] <  MinValue )
         {
         MinValue=Low[i];
         MinTime=Time[i];
         MinIndex=i;
         }
      }
   T.DatetimeExtremum=MinTime;
   T.IndexExtremum=MinIndex;
   T.Price=MinValue;
   }
```

Then, put all this into a pre-prepared container. The logic is as follows: all structures used within the class require gradual addition of data. After passing all the steps and stages, the required data is output. Using this data, the pattern can be graphically displayed on the chart. Of course, top and bottom determining logic can be different. My purpose is only to show a simple detection logic for complex things.

**Selecting tops to work with**

The tops and bottoms which we have found are only intermediate. After finding them, we need to select the tops which we consider as most appropriate to act as shoulders. We can't determine this for sure because the code does not have machine vision (in general, the usage of such complex techniques is unlikely to benefit the performance). For now, let's select the tops that are closest to the market:

```
bool ExtremumsPatternFamilySearcher::PrepareExtremums()// assign the tops with which we will work
   {
   int Quantity;// an auxiliary counter for random tops
   int PrevIndex;// an auxiliary index for maintaining the order of indexes (increment only)

   for(int i=0;i<TopsM;i++)// simply select the tops that are closest to the market
      {
      TopsUp[i]=TopsUpAll[i];
      TopsDown[i]=TopsDownAll[i];
      }
   return true;
   }
```

Visually on the symbol chart, the logic will be equivalent to the variant in the purple frame. I will draw some more variants for selection:

![Choose tops & bottoms](https://c.mql5.com/2/42/n5rc2_ke17ge.png)

In this case, the selection logic is very simple. The selected variants are 0 and 1 because they are closest to the market. Here everything applies to a double top. But the same logic will be used for triple or greater multiple top, the only difference being in the number of selected tops.

This function will be expanded in the future, to enable the ability to select tops randomly, as shown in blue in the image above. This will simulate multiple instances of pattern finders. This allows a more efficient and more frequent finding of all patterns in the automated mode.

**Determining the pattern direction**

Once we have identified the tops and bottoms, we must determine the direction of the formation, if such a formation exists at a given point in the market. At this stage, I consider assigning greater priority to the direction whose extremum type is closest to the market. Based on this logic, let's use variant 0 from the figure, because the closest to the market is the bottom, not the top (provided that the situation on the market is exactly the same as in the figure). This part is simple in the code:

```
void ExtremumsPatternFamilySearcher::DirectionOfFormation()// determine whether it is a double top (1) or double bottom (-1) (only if all tops and bottoms are found - if not found, then 0)
   {
   if ( TopsDown[0].DatetimeExtremum > TopsUp[0].DatetimeExtremum && TopsDown[ArraySize(TopsDown)-1].bActive )
      {
      StartTop=TopsDown[ArraySize(TopsDown)-1];
      EndTop=TopsDown[0];
      FormationDirection=-1;
      }
   else if ( TopsDown[0].DatetimeExtremum < TopsUp[0].DatetimeExtremum && TopsUp[ArraySize(TopsUp)-1].bActive )
      {
      StartTop=TopsUp[ArraySize(TopsUp)-1];
      EndTop=TopsUp[0];
      FormationDirection=1;
      }
   else FormationDirection=0;
   }
```

Further actions require a clearly determined direction. The direction is equivalent to the pattern type:

1. Multiple top
2. Multiple bottom

These rules also apply for the Head and Shoulders pattern and all other hybrid formations. The class was supposed to be common for all patterns of this family — this generality is already working in part.

**Filters to discard incorrect patterns:**

Now let's go further. Knowing that we have a direction and one of the ways to select tops and bottoms, we must provide the following for a multiple top: the tops that are between the selected ones should be lower than the lowest of the selected ones. For a multiple bottom, such bottoms should be higher than the highest of the selected ones. In this case, if tops are selected randomly, all the selected tops would be clearly distinguished. Otherwise, this check is not required:

```
bool ExtremumsPatternFamilySearcher::IsExtremumsAbsolutely()// require the selected extrema to be the most extreme ones
   {
   if ( bRandomExtremumsM )// check only if we have a random selection of tops (in other case the check should be considered completed)
      {
      if ( FormationDirection == 1 )
         {
         int StartIndex=RandomIndexUp[0];
         int EndIndex=RandomIndexUp[ArraySize(RandomIndexUp)-1];
         for(int i=StartIndex+1;i<EndIndex;i++)// check all tops between the selected ones
            {
            for(int j=0;j<ArraySize(TopsUp);j++)
               {
               if ( TopsUpAll[i].Price >= TopsUp[j].Price )
                  {
                  for(int k=0;k<ArraySize(RandomIndexUp);k++)
                     {
                     if ( i != RandomIndexUp[k] ) return false;
                     }
                  }
               }
            }
         return true;
         }
      else if ( FormationDirection == -1 )
         {
         int StartIndex=RandomIndexDown[0];
         int EndIndex=RandomIndexDown[ArraySize(RandomIndexDown)-1];
         for(int i=StartIndex+1;i<EndIndex;i++)// check all tops between the selected ones
            {
            for(int j=0;j<ArraySize(TopsDown);j++)
               {
               if ( TopsDownAll[i].Price <= TopsDown[j].Price )
                  {
                  for(int k=0;k<ArraySize(RandomIndexDown);k++)
                     {
                     if ( i != RandomIndexDown[k] ) return false;
                     }
                  }
               }
            }
         return true;
         }
      else return false;
      }
   else
      {
      return true;
      }
   }
```

If we visually display the correct and incorrect variant of random top selection, which is performed by the last predicate function, it will look like this:

![Control of unaccounted tops](https://c.mql5.com/2/42/alohey14_x9hcph.png)

These criteria are mirrored for the bullish and bearish patterns. The figure shows a bullish pattern as an example. The second case can be easily imagined.

After completing all preparatory procedures, we can proceed to searching for the neck. Different traders plot the neck in different ways. I have conditionally determined several types of construction:

1. Visually tilted (not by shadows)
2. Visually, horizontal (not by shadows)
3. Highest or lowest point, tilted (by shadows)
4. Highest or lowest point, horizontal (by shadows)

For safety reasons and to increase the chances of profit, I believe that the optimal variant is 4. I have chosen this due to the following:

- The beginning of a reversal movement is found more clearly
- This approach is easier to implement in code
- The slope is determined unambiguously (horizontally)

Perhaps, this is not entirely correct from the point of view of construction, but I haven't found any clear rules. This is not critical from the point of view of algo-trading. If we find something rational in this pattern, the tester or visualization will definitely show us something. Further task implies strengthening of trading results, which is however an absolutely different task.

I have created two mirror functions for the bullish and bearish patterns that define all the necessary parameters of the neck:

```
void ExtremumsPatternFamilySearcher::FindNeckUp(Top &TStart,Top &TEnd)// find the neck line based on the two extreme tops (for the classic multiple top)
   {
   double PriceMin=Low[TStart.IndexExtremum];
   datetime TimeMin=Time[TStart.IndexExtremum];
   for(int i=TStart.IndexExtremum;i>=TEnd.IndexExtremum;i--)// define the lowest point
      {
      if ( Low[i] < PriceMin )
         {
         PriceMin=Low[i];
         TimeMin=Time[i];
         }
      }
   // define the parameters of the anchor point and all parameters of the line equation
   Neck.Price0=PriceMin;
   Neck.TimeX=TimeMin;
   Neck.Time0=Time[0];
   Neck.Price1=PriceMin;
   Neck.Time1=TStart.DatetimeExtremum;
   Neck.DirectionOfFormation=true;
   Neck.CalculateKC();
   }

void ExtremumsPatternFamilySearcher::FindNeckDown(Top &TStart,Top &TEnd)// find the neck line based on two extreme bottoms (for the classic multiple bottom)
   {
   double PriceMax=High[TStart.IndexExtremum];
   datetime TimeMax=Time[TStart.IndexExtremum];
   for(int i=TStart.IndexExtremum;i>=TEnd.IndexExtremum;i--)// define the lowest point
      {
      if ( High[i] > PriceMax )
         {
         PriceMax=High[i];
         TimeMax=Time[i];
         }
      }
   // define the parameters of the anchor point and all parameters of the line equation
   Neck.Price0=PriceMax;
   Neck.TimeX=TimeMax;
   Neck.Time0=Time[0];
   Neck.Price1=PriceMax;
   Neck.Time1=TStart.DatetimeExtremum;
   Neck.DirectionOfFormation=false;
   Neck.CalculateKC();
   }
```

For correct and simple plotting of the neck, it's better to use the same rules for neck construction for all patterns of the selected family. On the one hand, this eliminates unnecessary details, which in our case will give nothing. To build a neck for a multiple top of any complexity, it is better to use two extreme tops of the pattern. The indices of these peaks will be the indices between which we will search for the lowest or highest price in the selected segment of the market. The neck will be a regular horizontal line. The first anchor points should be exactly at this level, while the anchor time should better be exactly equal to the time of the extreme tops or bottoms (depending on which pattern we are considering). This is how it will look in the picture:

![Neck](https://c.mql5.com/2/42/3gg.png)

The window to search for low or high is exactly between the first and the last top. This rule is valid for any pattern of this family, for any number of tops and bottoms.

To determine the optimistic target, first you should define the pattern size. The pattern size is the vertical distance from head to neck in points. To determine the distance, we first need to find the top which is farthest from the neck. This top will be the border of the pattern:

```
void ExtremumsPatternFamilySearcher::SearchFarestTop()// define the farthest top
   {
   double MaxTranslation;// temporary variable to determine the highest top
   if ( FormationDirection == 1 )// if we deal with a multiple top
      {
      MaxTranslation=TopsUp[0].Price-Neck.Price0;// temporary variable to determine the highest top
      FarestTop=TopsUp[0];
      for(int i=1;i<ArraySize(TopsUp);i++)
         {
         if ( TopsUp[i].Price-Neck.Price0 > MaxTranslation )
            {
            MaxTranslation=TopsUp[i].Price-Neck.Price0;
            FarestTop=TopsUp[i];
            }
         }
      }
   if ( FormationDirection == -1 )// if we deal with a multiple bottom
      {
      MaxTranslation=Neck.Price0-TopsDown[0].Price;// temporary variable to determine the lowest bottom
      FarestTop=TopsDown[0];
      for(int i=1;i<ArraySize(TopsDown);i++)
         {
         if ( Neck.Price0-TopsDown[i].Price > MaxTranslation )
            {
            MaxTranslation=Neck.Price0-TopsDown[0].Price;
            FarestTop=TopsDown[i];
            }
         }
      }
   }
```

An additional check is needed to make sure the tops do not differ too much. We can proceed to further steps only if the check is successful. More precisely, there should be two checks: one for the vertical size of the extrema, the other for the horizontal (time). If tops are too distant in time, such a variant does not suit either. Here is a check for the vertical size:

```
bool ExtremumsPatternFamilySearcher::bBalancedExtremums()// balance the tops
   {
   double Lowest;// the lowest top for the multiple top
   double Highest;// the highest bottom for the multiple bottom
   double AbsMin;// distance from the neck to the nearest top
   if ( FormationDirection == 1 )// for the multiple top
      {
      Lowest=TopsUp[0].Price;
      for(int i=1;i<ArraySize(TopsUp);i++)// find the lowest top
         {
         if ( TopsUp[i].Price < Lowest ) Lowest=TopsUp[i].Price;
         }
      AbsMin=Lowest-Neck.Price0;// determine distance from the lowest top to the neck
      if ( AbsMin == 0.0 ) return false;
      if ( ((FarestTop.Price - Neck.Price0)-AbsMin)/AbsMin >= RelativeUnstabilityM ) return false;// if the head is too much bigger than the lowest leverage
      }
   else if ( FormationDirection == -1 )// for the multiple bottom
      {
      Highest=TopsDown[0].Price;
      for(int i=1;i<ArraySize(TopsDown);i++)// find the highest top
         {
         if ( TopsDown[i].Price > Highest ) Highest=TopsDown[i].Price;
         }
      AbsMin=Neck.Price0-Highest;// determine distance from the highest top to the neck
      if ( AbsMin == 0.0 ) return false;
      if ( ((Neck.Price0-FarestTop.Price)-AbsMin)/AbsMin >= RelativeUnstabilityM ) return false;// if the head is too much bigger than the lowest leverage
      }
   else return false;
   return true;
   }
```

To determine the correct vertical size of the tops, we need two tops. The first one is the farthest one from the neck, and the second one if the closest to it. If these sizes differ greatly, then this formation may turn out to be invalid, and it is better not to risk and mark it as invalid. Similarly to the previous predicate, all this can be accompanied by an appropriate graphics of what is right and what is wrong:

![Control of vertical size](https://c.mql5.com/2/42/8yh.png)

They are easy to determine visually, but the code needs a quantitative metric. In this case, it is as simple as follows:

- K = ( **Max** \- **Min**)/ **Min**
- K <= **RelativeUnstabilityM**

This metric is quite efficient to filter out quite a large number of false patterns. Well, even the most sophisticated code cannot be more efficient than our eye. The only thing we can do is make the logic as close to reality as we can — but here we must know where to stop.

The horizontal check will look similar. The only difference is that we use bar indices as sizes (you can use time, there is no fundamental difference):

```
bool ExtremumsPatternFamilySearcher::bBalancedExtremumsTime()// balance the sizes of shoulders and head along the horizontal axis
   {
   double Lowest;// minimum distance between the tops
   double Highest;// maximum distance between the tops
   if ( FormationDirection == 1 )// for the multiple top
      {
      Lowest=TopsUp[1].IndexExtremum-TopsUp[0].IndexExtremum;
      Highest=TopsUp[1].IndexExtremum-TopsUp[0].IndexExtremum;
      for(int i=1;i<ArraySize(TopsUp)-1;i++)// find the lowest top
         {
         if ( TopsUp[i+1].IndexExtremum-TopsUp[i].IndexExtremum < Lowest ) Lowest=TopsUp[i+1].IndexExtremum-TopsUp[i].IndexExtremum;
         if ( TopsUp[i+1].IndexExtremum-TopsUp[i].IndexExtremum > Highest ) Highest=TopsUp[i+1].IndexExtremum-TopsUp[i].IndexExtremum;
         }
      if ( double(Highest-Lowest)/double(Lowest) > RelativeUnstabilityTimeM ) return false;// if the width of one of the waves differs much
      }
   else if ( FormationDirection == -1 )// for the multiple bottom
      {
      Lowest=TopsDown[1].IndexExtremum-TopsDown[0].IndexExtremum;
      Highest=TopsDown[1].IndexExtremum-TopsDown[0].IndexExtremum;
      for(int i=1;i<ArraySize(TopsDown)-1;i++)// find the lowest top
         {
         if ( TopsDown[i+1].IndexExtremum-TopsDown[i].IndexExtremum < Lowest ) Lowest=TopsDown[i+1].IndexExtremum-TopsDown[i].IndexExtremum;
         if ( TopsDown[i+1].IndexExtremum-TopsDown[i].IndexExtremum > Highest ) Highest=TopsDown[i+1].IndexExtremum-TopsDown[i].IndexExtremum;
         }
      if ( double(Highest-Lowest)/double(Lowest) > RelativeUnstabilityTimeM ) return false;// if the width of one of the waves differs much
      }
   else return false;
   return true;
   }
```

For this check, we can use a similar metric. Visually, it can be expressed as follows:

![Control of horizontal size ](https://c.mql5.com/2/42/o3cb48fk_i6izjf4zyhcbyo8_4k6bb8i.png)

In this case, the quantitative criteria will be the same. However, this time we use indices or time instead of points. It might be better to implement the number, with which we are comparing, separately, which would give room for flexible adjustment:

- K = ( **Max**- **Min**)/ **Min**
- K <= **RelativeUnstabilityTimeM**

The neck line must cross the price on the left — this means that the pattern was preceded by a trend:

```
bool ExtremumsPatternFamilySearcher::CorrectNeckUpLeft()// next the neck line must be corrected so that it finds an intersection with the price on the left
   {
   bool bCrossNeck=false;// indicates if the neck was crossed
   if ( Neck.DirectionOfFormation )// if the neck is found for a double top
      {
      for(int i=StartTop.Index1;i<BarsM;i++)// define the intersection point
         {
         if ( High[i] >= FarestTop.Price )// if the movement goes beyond the formation, then the formation is fake
            {
            return false;
            }
         if ( Close[i] < Neck.Price0 && Open[i] < Neck.Price0 && High[i] < Neck.Price0 && Low[i] < Neck.Price0   )
            {
            Neck.Time1=Time[i];
            Neck.Index1=i;
            return true;
            }
         }
      }
   return false;
   }

bool ExtremumsPatternFamilySearcher::CorrectNeckDownLeft()// next the neck line must be corrected so that it finds an intersection with the price on the left
   {
   bool bCrossNeck=false;// indicates if the neck was crossed
   if ( !Neck.DirectionOfFormation )// if the neck is found for a double bottom
      {
      for(int i=StartTop.Index1;i<BarsM;i++)// define the intersection point
         {
         if ( Low[i] <= FarestTop.Price )//  if the movement goes beyond the formation, then the formation is fake
            {
            return false;
            }
         if ( Close[i] > Neck.Price0 && Open[i] > Neck.Price0 && High[i] > Neck.Price0 && Low[i] > Neck.Price0 )
            {
            Neck.Time1=Time[i];
            Neck.Index1=i;
            return true;
            }
         }
      }
   return false;
   }
```

Again, there are two mirror functions for the bullish and bearish patterns. Below is a graphical illustration of this predicate and the next one:

![Left and right intersection control](https://c.mql5.com/2/42/al2y3s0m_z2wbja6a4e8_3c4ke_s_nt44ip.png)

The blue boxes mark the market segments where we control the intersection. Both segments are behind the pattern, to the left and to the right of the extreme tops.

There are only two checks left:

1. We need a pattern that crosses the neck line at the current moment (at the zero candlestick)
2. The pattern must be preceded by a movement greater than or equal to the pattern itself

The first point is needed for algorithmic trading. I don't think it's worth detecting formations only for viewing them, although this function is also provided. We need both detection and finding exactly at the point from which we can trade — where we can immediately open a position, knowing that we are at the entry point. The second point is one of the necessary conditions, because the pattern itself is useless without a good preceding movement.

Zero candlestick cross (checking the intersection on the right) is determined as follows:

```
int ExtremumsPatternFamilySearcher::CorrectNeckUpRight()// next the neck line must be corrected so that it finds an intersection with the price on the right
   bool bCrossNeck=false;// indicates if the neck was crossed
   if ( Neck.DirectionOfFormation )// if the neck is found for a double top
      {
      for(int i=EndTop.IndexExtremum;i>1;i--)// define the intersection point
         {
         if ( High[i] > FarestTop.Price || Low[i] < Neck.Price0 )// if the movement goes beyond the formation, then the formation is fake
            {
            return -1;
            }
         }
      }

   if ( Close[0] <= Neck.Price0 )
      {
      Neck.Time0=Time[0];
      return 1;
      }
   return 0;
   }

int ExtremumsPatternFamilySearcher::CorrectNeckDownRight()// next the neck line must be corrected so that it finds an intersection with the price on the right
   {
   bool bCrossNeck=false;// indicates if the neck was crossed
   if ( !Neck.DirectionOfFormation )// if the neck is found for a double bottom
      {
      for(int i=EndTop.IndexExtremum;i>1;i--)// define the intersection point
         {
         if ( Low[i] < FarestTop.Price || High[i] > Neck.Price0  )// if the movement goes beyond the formation, then the formation is fake
            {
            return -1;
            }
         }
      }

   if ( Close[0] >= Neck.Price0 )
      {
      Neck.Time0=Time[0];
      return 1;
      }

   return 0;
   }
```

<

Again, we have two mirror functions. Please note that the intersection on the right is not considered valid if the price has moved beyond the pattern and then returned back - this behavior is covered here and is shown in the previous figure.

Now, let's determine how to find the preceding trend. SO far I am using the optimistic forecast line for this purpose. If there is a market segment between the neck and the line of the optimistic forecast, then this is the desired movement. This movement must not be too extended in time, otherwise it is obviously not a movement:

```
bool ExtremumsPatternFamilySearcher::bWasTrend()// did we find the movement preceding the formation (also move here the anchor point to the intersection)
   {
   bool bCrossOptimist=false;// denotes if the neck is crossed
   if ( FormationDirection == 1 )// if the optimistic forecast is at the double top
      {
      for(int i=Neck.Index1;i<BarsM;i++)// define the intersection point
         {
         if ( High[i] > Neck.Price0 )// if the movement goes beyond the neck, then the formation is fake
            {
            return false;
            }
         if ( Low[i] < OptimistLine.Price0 )
            {
            OptimistLine.Time1=Time[i];
            return true;
            }
         }
      }
   else if ( FormationDirection == -1 )// if the optimistic forecast is at the double bottom
      {
      for(int i=Neck.Index1;i<BarsM;i++)// define the intersection point
         {
         if ( Low[i] < Neck.Price0 )//  if the movement goes beyond the neck, then the formation is fake
            {
            return false;
            }
         if ( High[i] > OptimistLine.Price0 )
            {
            OptimistLine.Time1=Time[i];
            return true;
            }
         }
      }
   return false;
   }
```

The last predicate can also be represented graphically as follows:

![Prior movement](https://c.mql5.com/2/42/ik7dgne0n97f4f_otof5u1k.png)

Let's finish reviewing the code here and move on to visual assessments. I think the main ideas of the method have been sufficiently described in this article. Further ideas will be considered in the next article of this series.

**Let's check the result in the MetaTrader 5 visual tester:**

I always use line drawing on the chart, as it is fast, simple, and clear. The MQL5 Help provides examples of using any graphical objects, including lines. I will not provide the drawing code here, but you can see its execution result. Of course, everything could be done better, but we only have a prototype. So, I believe here we can use the "necessity and sufficiency" principle:

![Triple top in the MetaTrader 5 strategy tester visualizer](https://c.mql5.com/2/42/6pnm42f_e5qxh8z_6_2q3bf994jqvmu.png)

Here is an example with a triple top. This example seemed more interesting to me. Double tops are detected in a similar way — you only need to set the desired number of tops in parameters. The code does not find such formations often, but it is only a demonstration. The code can be further refined (which I am planning to do later).

### Further development ideas

Later, we will consider what was left unsaid in this article, and will improve search quality for all formations. We will also refine the class to enable it to detect the Head and Shoulders formations. We will also try to find possible hybrid functions of these formations; one of them might be "N tops and multiple shoulders". The series is not devoted to only this family of patterns and will include new interesting and useful material. There are different approaches to pattern search, and the idea of this series is to show as many patterns as possible using different examples and thus to cover different possible ways to breaking down a complex task into a set of simpler ones. The series will include:

1. Other interesting patterns
2. Other methods for detecting different formation types
3. Trading using historical data and collecting statistics for different instruments and timeframes
4. There are a lot of patterns, and I don't know them all (so I can potentially consider your pattern)
5. We will also consider levels (as levels are often used to detect reversals)

### Conclusion

I tried to make the material simple and understandable for everyone. I hope anyone can find something useful here. The conclusion of this particular article is that, as can be seen from the visual strategy tester, a simple code is able to find complex formations. We so not necessarily need to use neural networks or write/use some complex machine vision algorithms. The MQL5 language has rich functionality to implement even the most complex algorithms. The possibilities are only limited by your imagination and diligence.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/9394](https://www.mql5.com/ru/articles/9394)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/9394.zip "Download all attachments in the single ZIP archive")

[Prototype.zip](https://www.mql5.com/en/articles/download/9394/prototype.zip "Download Prototype.zip")(309.42 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Multibot in MetaTrader (Part II): Improved dynamic template](https://www.mql5.com/en/articles/14251)
- [Brute force approach to patterns search (Part VI): Cyclic optimization](https://www.mql5.com/en/articles/9305)
- [Brute force approach to patterns search (Part V): Fresh angle](https://www.mql5.com/en/articles/12446)
- [OpenAI's ChatGPT features within the framework of MQL4 and MQL5 development](https://www.mql5.com/en/articles/12475)
- [Rebuy algorithm: Multicurrency trading simulation](https://www.mql5.com/en/articles/12579)
- [Rebuy algorithm: Math model for increasing efficiency](https://www.mql5.com/en/articles/12445)
- [Multibot in MetaTrader: Launching multiple robots from a single chart](https://www.mql5.com/en/articles/12434)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/375727)**
(31)


![Carlos Albert Barbero Marcos](https://c.mql5.com/avatar/2021/10/616EB0B9-1ED9.jpg)

**[Carlos Albert Barbero Marcos](https://www.mql5.com/en/users/galafron)**
\|
3 Nov 2021 at 18:44

Blurred logic like that leads to facial recognition, just as a sketch drawn with a pencil leads to a portrait.  A few drawing lets appear a pattern evidence  that translate to a trade..


![Evgeniy Ilin](https://c.mql5.com/avatar/2021/1/60171FA1-EC8E.png)

**[Evgeniy Ilin](https://www.mql5.com/en/users/w.hudson)**
\|
4 Nov 2021 at 11:41

**v3nen0 [#](https://www.mql5.com/en/forum/375727#comment_24485737):**

I find it too confusing, maybe less code and more explanations. Understanding code is hard writing is easy once we understand the problem.. So more explanations of the problem than code, plz.

in the next article I will conduct a complete analysis of the entire family of these patterns in order to issue a conclusion based on trade statistics (we will conduct backtests and summarize everything in tables).I hope to draw unambiguous conclusions about the applicability of a particular pattern on a particular timeframe, based on statistics.Most likely, the code will not be.There will be only facts confirmed by statistics.I think it will be useful to everyone.

![andrik377](https://c.mql5.com/avatar/avatar_na2.png)

**[andrik377](https://www.mql5.com/en/users/andrik377)**
\|
2 Dec 2021 at 07:53

[@Evgeniy Ilin](https://www.mql5.com/en/users/w.hudson) The maths of all patterns is here "Bulkowski - Encyclopedia of Price Patterns". It is available on torrents.


![Koorapetse Lesotlho](https://c.mql5.com/avatar/2022/7/62D2C53C-BAD4.jpg)

**[Koorapetse Lesotlho](https://www.mql5.com/en/users/phenyo101)**
\|
25 Mar 2022 at 07:15

**MetaQuotes:**

New article [Patterns with Examples (Part I): Multiple Top](https://www.mql5.com/en/articles/9394) has been published:

Author: [Evgeniy Ilin](https://www.mql5.com/en/users/W.HUDSON "W.HUDSON")

amazing article, i have been trying to formulate my own logic without using fractals. and you helped a lot in getting everything sorted. thanks

![Christophe Pa Trouillas](https://c.mql5.com/avatar/2023/3/6422fa6d-7451.png)

**[Christophe Pa Trouillas](https://www.mql5.com/en/users/metasignalspro)**
\|
5 Jul 2024 at 13:50

Hi Evgeniy,

This is heavy contribution. Thanks a lot.

I am **unable though to attach the indicator to any chart** for its MT4 version.

It won't load.

Do you see why?

![Better Programmer (Part 03): Give Up doing these 5 things to become a successful MQL5 Programmer](https://c.mql5.com/2/43/Article_image__1.png)[Better Programmer (Part 03): Give Up doing these 5 things to become a successful MQL5 Programmer](https://www.mql5.com/en/articles/9746)

This is the must-read article for anyone wanting to improve their programming career. This article series is aimed at making you the best programmer you can possibly be, no matter how experienced you are. The discussed ideas work for MQL5 programming newbies as well as professionals.

![Better Programmer (Part 02): Stop doing these 5 things to become a successful MQL5 programmer](https://c.mql5.com/2/43/Article_image.png)[Better Programmer (Part 02): Stop doing these 5 things to become a successful MQL5 programmer](https://www.mql5.com/en/articles/9711)

This is the must read article for anyone wanting to improve their programming career. This article series is aimed at making you the best programmer you can possibly be, no matter how experienced you are. The discussed ideas work for MQL5 programming newbies as well as professionals.

![Better Programmer (Part 04): How to become a faster developer](https://c.mql5.com/2/43/speed.png)[Better Programmer (Part 04): How to become a faster developer](https://www.mql5.com/en/articles/9752)

Every developer wants to be able to write code faster, and being able to code faster and effective is not some kind of special ability that only a few people are born with. It's a skill that can be learned by every coder, regardless of years of experience on the keyboard.

![Graphics in DoEasy library (Part 78): Animation principles in the library. Image slicing](https://c.mql5.com/2/42/MQL5-avatar-doeasy-library3-2__5.png)[Graphics in DoEasy library (Part 78): Animation principles in the library. Image slicing](https://www.mql5.com/en/articles/9612)

In this article, I will define the animation principles to be used in some parts of the library. I will also develop a class for copying a part of the image and pasting it to a specified form object location while preserving and restoring the part of the form background the image is to be superimposed on.

[![](https://www.mql5.com/ff/si/x6w0dk14xy0tf97n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F586%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dhow.test.expert%26utm_content%3Dbuy.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=repptjucjbnrxhoeoqbekpbncvsnhylz&s=3da978a0c510a6306b46ee79cdf8418a5c0da5e081f296e18b262b00031a2310&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=zaivwnwnxdnxzkwqhcuxxjfvhhchzeum&ssn=1769193506138179037&ssn_dr=0&ssn_sr=0&fv_date=1769193506&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F9394&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Patterns%20with%20Examples%20(Part%20I)%3A%20Multiple%20Top%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176919350669767039&fz_uniq=5071985240211141154&sv=2552)

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