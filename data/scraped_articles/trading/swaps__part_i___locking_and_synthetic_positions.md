---
title: Swaps (Part I): Locking and Synthetic Positions
url: https://www.mql5.com/en/articles/9198
categories: Trading, Trading Systems, Expert Advisors
relevance_score: 0
scraped_at: 2026-01-24T13:33:20.965834
---

[Best articles and CodeBase updates in MQL5.community channelsFollow us to ensure you never miss out on important updates![](https://www.mql5.com/ff/sh/n9yf51p2srwzfqh5z2/02.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=dgazvhktsxqakdvarucjbvmvzenwlyje&s=98a038fe082e458df8c4a1d8e116e3a6646fd5517f06e48b2356b7ee005817d6&uid=&ref=https://www.mql5.com/en/articles/9198&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5082946993183003119)

MetaTrader 5 / Trading


### Table Of Contents

- [Introduction](https://www.mql5.com/en/articles/9198#para1)
- [About Swaps](https://www.mql5.com/en/articles/9198#para2)
- [Locking Using Two Trading Accounts](https://www.mql5.com/en/articles/9198#para3)
- [About Exchange Rates](https://www.mql5.com/en/articles/9198#para4)
- [Locking Using Synthetic Positions](https://www.mql5.com/en/articles/9198#para5)
- [Writing a Utility to Use Swap Polygons](https://www.mql5.com/en/articles/9198#para6)
- [Conclusions from the First Tests](https://www.mql5.com/en/articles/9198#para7)
- [Conclusion](https://www.mql5.com/en/articles/9198#para8)

### Introduction

I was thinking about the topic of this article for a long time, but I did not have time to conduct a detailed research. The topic of swaps is quite widespread on the web, mainly among trading professionals who count every pip, which is actually the best approach to trading. From this article, you will find out how to use swaps for good and will see that the swaps must always be taken into account. Also, the article features a very complex but interesting idea on how to modernize swap trading methods. Such methods (when properly prepared) can be used within one account or as a profit enhancement tool for classic locking using two accounts.

### About Swaps

I will not explain the idea of swaps and their theory. I am only interested in the practical application of swaps. The most important question is whether it is possible to generate profit via swaps. From a trader's point of view, a swap is a profit or a loss. Furthermore, a lot of traders simply ignore it as they stick to intraday trading. Others try not to pay attention to it, thinking that it is so insignificant that it can hardly affect trading. In fact, almost half of the spread can be hidden in the swap. This spread is taken not at the time of buying or selling, but when the day changes on the server.

The swap is charged relative to the open position volume. This happens at the following moments:

1. Monday to Tuesday
2. Tuesday to Wednesday
3. From Wednesday to Thursday (almost all brokers charge a triple swap this night)
4. Thursday to Friday

Usually, the swap value is indicated in the trading instrument specification in points or as a percentage. There can be other calculation method, but I have managed to understand only two of them, which is quite enough. There is very little structured information about swaps. However, if you study the question, you may even find some efficient swap-based strategies. They generate minimal profit percent, but they have a great advantage — the profit is absolutely guaranteed. The main difficulty of this approach is the fact that the most popular brokers have very few instruments with a positive swap, so it is really hard to earn money from this idea. Even the possible potential profit is extremely low. Nevertheless, this is better than completely losing the deposit. And if you are using any other trading system, you will most likely lose it.

The bottom line of my conclusions regarding Forex trading is that nothing can guarantee a profit except positive swaps. Of course, there are some systems which are capable of generating a profit. However, when using them, we agree to pay our money to the broker for any trading operation, and we hope that the price will go in the right direction. A positive swap is the reverse process. I see the following statements as the markers of trading in the positive swap direction:

- A positive swap is equivalent to a partial price movement towards our open position (profit every day)
- After some time, a swap can cover losses on spreads and commissions; after a while, the swap will add money
- In order for the swap to work, the position should be held as long as possible, then the profit factor of this positions will be maximal
- If developed thoroughly, the profit will be absolutely predictable and guaranteed

Of course, the biggest disadvantage of this approach is the dependence on the deposit size, but no other concept is capable of so confidently guaranteeing profit in Forex. This dependence can be reduced by reducing the volume of open positions or risk (which is the same). Risk is the ratio of position volume to deposit: an increase in position volume increases our risks that the price may go in the losing direction and the deposit may not be enough for us to wait for the profit from swaps to compensate the losses from spreads and commissions. A locking mechanism was invented to minimize the influence of all possible negative effects.

### Locking Using Two Trading Accounts

This swap trading method is the most popular among traders. To implement this strategy, you will need two accounts with different swaps for the same currency pairs or other assets. Opening two opposite positions within the same account is meaningless — it is equivalent to simply losing the deposit. Even if a symbol has a positive swap, this swap will be negative when trading in the opposite direction. The following diagram reflects the concept of this method:

![Classic Swap Trading](https://c.mql5.com/2/42/xw8cl7ys_dpd8mx9o.png)

As you can see from the diagram, there are only 10 trading scenarios for a specifically selected instrument for which we want to trade swaps, 6 of which are actively used. The last four options can be chosen as a last resort, if it is impossible to find a currency pair that matches conditions "1-6", since one of the swaps here is negative. Profiting is possible from the positive swap which is greater than the negative one. You can find all of the above mentioned cases if you analyze different brokers and their swap tables. But the best options for this strategy are "2" and "5". These options have positive swaps at both ends. So, profit is earned from both brokers. Furthermore, you do not have to move funds between accounts that often.

The main disadvantage of this strategy is that you still need to move money between accounts, because when opening opposite positions you will have loss with one broker and profit with another broker. However, if you correctly calculate the trading volumes in relation to the existing deposit, you will not need to move funds too often. But there is one indisputable advantage: in any case there will be profit, while the exact size of this profit can be predicted. I think many users would prefer to avoid this routine and somehow perform these manipulations within one account (which is impossible). But there is one method of how to increase the profit of the classic swap trading method, even though it does not allow trading within one account. Let us discuss the main features of this method.

### About Exchange Rates

Let us start with the very basis relative to which all logic is built. Mathematical equations can be built on this basis. For example, consider EURUSD, USDJPY, EURJPY. All these 3 pairs are correlated. In order to understand the relationship, let us present these symbols in a slightly different form:

- 1/P = EUR/USD
- 1/P = USD/JPY
- 1/P = EUR/JPY
- P is the rate of the selected currency

Any trading instrument has a currency (or some equivalent asset) which we acquire and another currency which we give in return. For example, if you take the first ratio (EURUSD pair), then when opening a 1-lot Buy position, you acquire 100,000 units of the base currency. These are the Forex trading rules: one lot is always equal to 100,000 units of the base currency. The base currency of this pair is EUR and thus we buy EUR for USD. The currency rate "P" in this case means how many units of USD are contained in 1 EUR. The same is applicable to all other symbols: the base currency is contained in the numerator, while the denominator is the " **main currency**" (if you do not agree with this naming, please add a comment below). The amount of the main currency is calculated simply by multiplying the price by the EUR value:

- 1/P = EUR/USD --->  USD/P = EUR ---> USD = P\*EUR
- EUR = Lots\*100000

When opening a Sell position, the currencies change places. The base currency starts acting as the main one, and the main one becomes a base currency. In other words, we buy USD for EUR, but the amount of money of both currencies is calculated in the same way — relative to EUR. This is correct, because otherwise there would be quite a lot of confusion. Calculations are the same for other currencies. So, let us use in further calculation sign "+" for the base currency and sign "-" for the main currency. As a result, any trade has a set of two corresponding numbers which symbolize what and for what we buy. Another interpretation of this is that there is always a currency which acts as product and another currency which acts as a currency and which we pay to buy the product.

If we open several positions for several instruments, then there will be more main and additional currencies, and thus we have a kind of synthetic position. From the point of view of using swaps, such a synthetic position is absolutely useless. But we can create such a synthetic position that will be very useful. I will show it a bit later. I have determined the calculation of volume expressed by two currencies. Based on this, we can conclude that we can create a complex synthetic position which will be equivalent to some simpler one:

- EUR/JPY = EUR/USD \* USD/JPY — currency rate composed of two derivatives

In reality, there are an infinite number of such ratios, which are composed of several currencies, such as:

- EUR - European Union Euro
- USD - US dollar
- JPY - Japanese Yen
- GBP - Great Britain Pound
- CHF - Swiss Franc
- CAD - Canadian Dollar
- NZD - New Zealand Dollar
- AUD - Australian Dollar
- CNY - Chinese Yuan
- SGD - Singapore dollar
- NOK - Norwegian Krone
- SEK - Swedish Krona

This is not the complete list of currencies. What we need to know is that an arbitrary trading instrument can be composed of any currencies from this list. Some of these trading instruments are offered by brokers, while others can be obtained as a combination of positions of other instruments. A typical example is the EURJPY pair. This is just the simplest example of composing derivative exchange rates, but based on these ideas we can conclude that any position can be presented as a set of positions for other instruments. According to the above, it turns out that:

- Value1 is the base symbol currency expressed by an absolute value
- Value2 is an additional symbol currency expressed by an absolute value
- A is the lot volume of the position's base currency
- B is the lot volume of the position's main currency
- Contract is the amount of purchased or sold currency in absolute value (corresponds to 1 lot)
- A = 1/P = Value1/Value2 - it is the equation of any trading instrument (including those which are not presented in the market watch window)
- Value1 = Contract\*A
- Value2 = Contract\*B

We will need these ratios later to calculate lots. As for now please remember them. These ratios describe the ratio of the number of currencies being bought or sold. More serious code logic can be built on this basis.

### Locking Using Synthetic Positions

In this article, a synthetic position is a position which can be composed from several other positions, while these other positions must necessarily be composed of other instruments. This position must be equivalent to one open position for any instrument. Seems complicated? Actually, it is all very simple. Such a position may be needed in order to:

1. Lock the original position on a simulated trading instrument
2. Try to create the equivalent of a position with completely different swap rates
3. Other purposes

Originally, I came up with this idea in connection with point 2. Brokers set swap values for different purposes, the main of them being the desire to generate additional profits. I think brokers also take into account the swaps of their competitors to prevent traders from excessively trading the swaps. The below diagrams explain this concept. Perhaps you can enhance this diagram.

Here is the general scheme of this method:

![Method diagram](https://c.mql5.com/2/42/w06me_ksefuh.png)

Even this scheme does not cover the full data about how to open a synthetic position. This diagram only shows how to determine trading direction for a specific component of a synthetic position, which must necessarily be represented by one of the available instruments of the selected broker.

Now, we need to determine how to calculate the volumes of these positions. Logically, the volumes should be calculated based on the consideration that the position should be equivalent to a 1-lot the position for the resulting instrument, to which the selected variant of the equation is reduced. The following values are required for volume calculation:

- ContractB - the contract size of the pair to which the equation is reduced (in most cases it is equal to 100,000 units of the base currency)
- Contract\[1\] - the contract size of the pair for which you want to determine the lot
- A\[1\] - the amount of the base currency expressed in lots of the previous balanced pair (or the first in the chain)
- B\[1\] - the amount of the main currency expressed in lots of theprevious balanced pair (or the first in the chain)
- A\[2\] - the amount of the base currency expressed in lots of thecurrent pair being balanced
- B\[2\] - the amount of the main currency expressed in lots of thecurrent pair being balanced
- C\[1\] - contract size of the previous balanced pair (or the first one in the chain)
- C\[2\] - contract size of the current pair being balanced

Please note that it is not always possible to determine "ContractB", as the instrument resulting from the combination may not be provided by the broker. In this case the contract can be set arbitrary, for example, equal to the basic constant "100000".

First, the first pair in the chain is determined, which contains the base currency of the resulting instrument in the desired position. Then, other pairs are searched, which compensate for the extra currencies that are not included in the resulting equivalent. Balancing ends when the main currency is in the right position in the current pair. I have created a diagram to show how this is done:

![Normalization](https://c.mql5.com/2/42/34tj68g3wfsm.png)

Now let us implement these techniques in the code and analyze the results. The first prototype will be very simple, as its only purpose is to evaluate the correctness of the ideas. I hope the above diagrams will help you to understand all the details of the idea.

### Writing a Utility to Examine Swap Polygons

**Market Watch sorting and data preparation:**

In order to use this technique, it is necessary to select only those pairs whose names are exactly 6 characters long and consist only of uppercase letters. I think all brokers adhere to this naming rule. Some brokers add prefixes or postfixes, which should also be taken into account when writing algorithms for working with string data. In order to store symbol information in a convenient format, I have created two structures (the second one will be used later):

```
struct Pair// required symbol information
   {
   string Name;// currency pair
   double SwapBuy;// buy swap
   double SwapSell;// sell swap
   double TickValue;// profit from 1 movement tick of a 1-lot position
   double TickSize;// tick size in the price
   double PointX;// point size in the price
   double ContractSize;// contract size in the base deposit currency
   double Margin;// margin for opening 1 lot
   };

struct PairAdvanced : Pair// extended container
   {
   string Side;// in numerator or denominator
   double LotK;// lot coefficient
   double Lot;// lot
   };
```

Some fields will not be used when sorting pairs. In order not to produce unnecessary containers, I have expanded it a little so that the structure could also be used for other purposes. I had a prototype of a similar algorithm, but with very limited capabilities: it could only consider those pairs that were in the main terminal window. Now, everything is simpler. What is more important, all operations are automated in the algorithm. The following function is needed in order to set the size of the array with instruments:

```
Pair Pairs[];// data of currency pairs
void SetSizePairsArray()// set size of the array of pairs
   {
   ArrayResize(Pairs,MaxSymbols);
   ArrayResize(BasicPairsLeft,MaxPairs*2); // since each pair has 2 currencies, there can be a maximum of 2 times more base currencies
   ArrayResize(BasicPairsRight,MaxPairs*2);// since each pair has 2 currencies, there can be a maximum of 2 times more base currencies
   }
```

The first line sets the maximum number of pairs from the Market Watch window which we can use. The other two lines set the size of the arrays that will be used. The remaining 2 arrays play an auxiliary role — they allow splitting a currency pair into 2 parts (2 compound currencies). The variables highlighted in yellow are the input parameters of the EA.

- MaxSymbols - maximum pairs storage size (I have implemented manual specification)
- MaxPairs - the maximum number of pairs in both parts of the formula that we generate (formulas longer than this number will not be searched by the Expert Advisor)

In order to check whether a trading instrument meets the criteria (signs of two different currencies which can be potential present in other instruments), I have created the following predicate function:

```
bool IsValid(string s)// checking the instrument validity (its name must consist of upper-case letters)
   {
   string Mask="abcdefghijklmnopqrstuvwxyz1234567890";// mask of unsupported characters (lowercase letters and numbers)
   for ( int i=0; i<StringLen(s); i++ )// reset symbols
      {
      for ( int j=0; j<StringLen(Mask); j++ )
         {
         if ( s[i] == Mask[j] ) return false;
         }
      }
   return true;
   }
```

This function is not the only condition for future checks of instruments. But this condition cannot be written inside a logical expression, so it is easier to implement it as a predicate. Now, let us move on to the main function which fills the array with the necessary data:

```
void FillPairsArray()// fill the array with required information about the instruments
   {
   int iterator=0;
   double correction;
   int TempSwapMode;

   for ( int i=0; i<ArraySize(Pairs); i++ )// reset symbols
      {
      Pairs[iterator].Name="";
      }

   for ( int i=0; i<SymbolsTotal(false); i++ )// check symbols from the MarketWatch window
      {
      TempSwapMode=int(SymbolInfoInteger(Pairs[iterator].Name,SYMBOL_SWAP_MODE));
      if ( StringLen(SymbolName(i,false)) == 6+PrefixE+PostfixE && IsValid(SymbolName(i,false)) && SymbolInfoInteger(SymbolName(i,false),SYMBOL_TRADE_MODE) == SYMBOL_TRADE_MODE_FULL
      && ( ( TempSwapMode  == 1 )  ||  ( ( TempSwapMode == 5 || TempSwapMode == 6 ) && CorrectedValue(Pairs[iterator].Name,correction) )) )
         {
         if ( iterator >= ArraySize(Pairs) ) break;
         Pairs[iterator].Name=SymbolName(i,false);
         Pairs[iterator].TickSize=SymbolInfoDouble(Pairs[iterator].Name,SYMBOL_TRADE_TICK_SIZE);
         Pairs[iterator].PointX=SymbolInfoDouble(Pairs[iterator].Name, SYMBOL_POINT);
         Pairs[iterator].ContractSize=SymbolInfoDouble(Pairs[iterator].Name,SYMBOL_TRADE_CONTRACT_SIZE);
         switch(TempSwapMode)
           {
            case  1:// in points
              Pairs[iterator].SwapBuy=SymbolInfoDouble(Pairs[iterator].Name,SYMBOL_SWAP_LONG)*Pairs[iterator].TickValue*(Pairs[iterator].PointX/Pairs[iterator].TickSize);
              Pairs[iterator].SwapSell=SymbolInfoDouble(Pairs[iterator].Name,SYMBOL_SWAP_SHORT)*Pairs[iterator].TickValue*(Pairs[iterator].PointX/Pairs[iterator].TickSize);
              break;
            case  5:// in percent
              Pairs[iterator].SwapBuy=correction*SymbolInfoDouble(Pairs[iterator].Name,SYMBOL_SWAP_LONG)*SymbolInfoDouble(Pairs[iterator].Name,SYMBOL_BID)*Pairs[iterator].ContractSize/(360.0*100.0);
              Pairs[iterator].SwapSell=correction*SymbolInfoDouble(Pairs[iterator].Name,SYMBOL_SWAP_SHORT)*SymbolInfoDouble(Pairs[iterator].Name,SYMBOL_BID)*Pairs[iterator].ContractSize/(360.0*100.0);
              break;
            case  6:// in percent
              Pairs[iterator].SwapBuy=correction*SymbolInfoDouble(Pairs[iterator].Name,SYMBOL_SWAP_LONG)*SymbolInfoDouble(Pairs[iterator].Name,SYMBOL_BID)*Pairs[iterator].ContractSize/(360.0*100.0);
              Pairs[iterator].SwapSell=correction*SymbolInfoDouble(Pairs[iterator].Name,SYMBOL_SWAP_SHORT)*SymbolInfoDouble(Pairs[iterator].Name,SYMBOL_BID)*Pairs[iterator].ContractSize/(360.0*100.0);
              break;
           }
         Pairs[iterator].Margin=SymbolInfoDouble(Pairs[iterator].Name,SYMBOL_MARGIN_INITIAL);
         Pairs[iterator].TickValue=SymbolInfoDouble(Pairs[iterator].Name,SYMBOL_TRADE_TICK_VALUE);
         iterator++;
         }
      }
   }
```

This function provides a simple iteration of all symbols and filtering by a complex compound condition which checks both the compliance with the string name length requirement and the possibility to trade this symbol, as well as other parameters, which relate to those symbols for which the swap calculation method differs from the most commonly used on "in points". One of the swap calculation methods is selected in the "switch" block. Currently, two methods are implemented: in points and in percent. Proper sorting is primarily important to avoid unnecessary computations. Also, please pay attention to the function highlighted in red. When the main currency (not the basic one) is represented by a currency which is does not match the deposit currency, a certain adjustment factor should be added to covert swap into the deposit currency. This function calculates the relevant values. Here is its code:

```
bool CorrectedValue(string Pair0,double &rez)// adjustment factor to convert to deposit currency for the percentage swap calculation method
   {
   string OurValue=AccountInfoString(ACCOUNT_CURRENCY);// deposit currency
   string Half2Source=StringSubstr(Pair0,PrefixE+3,3);// lower currency of the pair to be adjusted
   if ( Half2Source == OurValue )
      {
      rez=1.0;
      return true;
      }

   for ( int i=0; i<SymbolsTotal(false); i++ )// check symbols from the MarketWatch window
      {
      if ( StringLen(SymbolName(i,false)) == 6+PrefixE+PostfixE && IsValid(SymbolName(i,false)) )//find the currency rate to convert to the account currency
         {
         string Half1=StringSubstr(SymbolName(i,false),PrefixE,3);
         string Half2=StringSubstr(SymbolName(i,false),PrefixE+3,3);

         if ( Half2 == OurValue && Half1 == Half2Source )
            {
            rez=SymbolInfoDouble(SymbolName(i,false),SYMBOL_BID);
            return true;
            }
         if ( Half1 == OurValue && Half2 == Half2Source )
            {
            rez=1.0/SymbolInfoDouble(SymbolName(i,false),SYMBOL_BID);
            return true;
            }
         }
      }
   return false;
   }
```

This function serves as a predicate as well as returns the adjustment factor value to the variable that has been passed from outside by reference. The adjustment factor is calculated based on the rate of the desired currency, which includes our deposit currency.

**Randomly Generated Formulas**

Suppose the array has been filled with the necessary data. Now, we somehow need to iterate over these symbols and try to create on the fly all possible combinations of formulas which can be created from these pairs. First, it is necessary to decide in which form the formula will be stored. The structure that stores all the elements of this formula should be very simple and clear for users in case there is a need to view logs (while there will definitely be such a need, otherwise it will be impossible to identify errors).

Our formula is a set of factors both to the left and to the right of the "=" sign. The factor can be the currency rate in the power of 1 or -1 (which is equivalent to an inverted fraction, or a unit referred to the current instrument rate). I decided to use the following structure:

```
struct EquationBasic // structure containing the basic formula
   {
   string LeftSide;// currency pairs participating in the formula on the left side of the "=" sign
   string LeftSideStructure;// structure of the left side of the formula
   string RightSide;// currency pairs participating in the right side of the formula
   string RightSideStructure;// structure of the right side of the formula
   };
```

All the data will be stored in string format. To study the formula, these strings will be parsed to extract all the necessary information we need. Also, they can be printed whenever needed. The generated formulas will be printed in the following form:

![Random Equations](https://c.mql5.com/2/42/94x2xoky3hx8_o6hdvea_feglkrs9o.png)

For me personally, such a record is absolutely clear and readable. Character **"^"** is used as a separator between pairs. No separators are needed in the formula structure, since it consists of single characters **"u"** and **"d"**, which indicate the degree of the multiplier:

1. "u" is the currency rate
2. "d" is 1/currency rate

As you can see, the resulting formulas have a floating length and a floating size of both sides of the equation, but this size has its limitations. This approach provides the maximum variability of the generated formulas. This, in turn, provides the highest possible quality of the variants found within the trading conditions of the selected broker. Brokers provide completely different conditions. To ensure the successful generation of these formulas, we need additional random functions which can generate numbers in the required range. For this purpose, let us create the relevant functionality using the capabilities of the built-in MathRand function:

```
int GenerateRandomQuantityLeftSide()// generate a random number of pairs on the left side of the formula
   {
   int RandomQuantityLeftSide=1+int(MathFloor((double(MathRand())/32767.0)*(MaxPairs-1)));
   if ( RandomQuantityLeftSide >= MaxPairs ) return MaxPairs-1;
   return RandomQuantityLeftSide;
   }

int GenerateRandomQuantityRightSide(int LeftLenght)// generate a random number of pairs on the right side of the formula (taking into account the number of pairs on the left side)
   {
   int RandomQuantityRightSide=1+int(MathFloor((double(MathRand())/32767.0)*(MaxPairs-LeftLenght)));
   if ( RandomQuantityRightSide < 2 && LeftLenght == 1 ) return 2;// there must be at least 2 pairs in one of the sides, otherwise it will be equivalent to opening two opposite positions
   if ( RandomQuantityRightSide > (MaxPairs-LeftLenght) ) return (MaxPairs-LeftLenght);
   return RandomQuantityRightSide;
   }

int GenerateRandomIndex()// generate a random index of a symbol from the MarketWatch window
   {
   int RandomIndex=0;

   while(true)
      {
      RandomIndex=int(MathFloor((double(MathRand())/32767.0) * double(MaxSymbols)) );
      if ( RandomIndex >= MaxSymbols ) RandomIndex=MaxSymbols-1;
      if ( StringLen(Pairs[RandomIndex].Name) > 0 ) return RandomIndex;
      }

   return RandomIndex;
   }
```

All the three functions will be needed at a certain stage. Now, we can write the function which will generate these formulas. The code will become more and more complex, but I will not use an object-oriented approach, since the task is not standard. I decided to use a procedural approach. The resulting procedures are quite large and cumbersome, but there is no extra functionality, and each function implements a specific task without using any intermediate functions, in order to avoid code duplication. Otherwise, the code would be even more difficult to understand due to the task specifics. The function will look like as follows:

```
EquationBasic GenerateBasicEquation()// generate both parts of the random equation
   {
   int RandomQuantityLeft=GenerateRandomQuantityLeftSide();
   int RandomQuantityRight=GenerateRandomQuantityRightSide(RandomQuantityLeft);
   string TempLeft="";
   string TempRight="";
   string TempLeftStructure="";
   string TempRightStructure="";

   for ( int i=0; i<RandomQuantityLeft; i++ )
      {
      int RandomIndex=GenerateRandomIndex();
      if ( i == 0 && RandomQuantityLeft > 1 ) TempLeft+=Pairs[RandomIndex].Name+"^";
      if ( i != 0 && (RandomQuantityLeft-i) > 1 ) TempLeft+=Pairs[RandomIndex].Name+"^";
      if ( i == RandomQuantityLeft-1 ) TempLeft+=Pairs[RandomIndex].Name;

      if ( double(MathRand())/32767.0 > 0.5 ) TempLeftStructure+="u";
      else TempLeftStructure+="d";
      }

   for ( int i=RandomQuantityLeft; i<RandomQuantityLeft+RandomQuantityRight; i++ )
      {
      int RandomIndex=GenerateRandomIndex();

      if ( i == RandomQuantityLeft && RandomQuantityRight > 1 ) TempRight+=Pairs[RandomIndex].Name+"^";
      if ( i != RandomQuantityLeft && (RandomQuantityLeft+RandomQuantityRight-i) > 1 ) TempRight+=Pairs[RandomIndex].Name+"^";
      if ( i == RandomQuantityLeft+RandomQuantityRight-1 ) TempRight+=Pairs[RandomIndex].Name;

      if ( double(MathRand())/32767.0 > 0.5 ) TempRightStructure+="u";
      else TempRightStructure+="d";
      }

   EquationBasic result;
   result.LeftSide=TempLeft;
   result.LeftSideStructure=TempLeftStructure;
   result.RightSide=TempRight;
   result.RightSideStructure=TempRightStructure;

   return result;
   }
```

As you can see, all the three previously considered functions are used here to generate a random formula. These functions are not used anywhere else in the code. As soon as the formula is ready, we can proceed to a step-by-step analysis of this formula. All incorrect formulas will be discarded by the next extremely important complex filter. First of all, check for equality. If the parts are not equal, then this formula is incorrect. All complying formulas proceed to the next analysis step.

**Formula Balancing**

This step covers several analysis criteria at once:

1. Counting all extra factors in the numerator and denominator and removing them
2. Checking the availability of 1 currency in the numerator and 1 currency in the denominator
3. Checking the correspondence of the resulting fractions on the left and right sides
4. If the right side is the reciprocal of the left side, we simply reverse the right structure of the formula (which is similar to raising to power "-1")
5. If all stages are completed successfully, the result is written into a new variable.

This is how these steps appear in the code:

```
BasicValue BasicPairsLeft[];// array of base pairs to the left
BasicValue BasicPairsRight[];// array of base pairs to the right
bool bBalanced(EquationBasic &CheckedPair,EquationCorrected &r)// if the current formula is balanced (if yes, return the corrected version to the "r" variable)
   {
   bool bEnd=false;
   string SubPair;// the full name of the currency pair
   string Half1;// the first currency of the pair
   string Half2;// the second currency of the pair
   string SubSide;// the currency pair in the numerator or denominator
   string Divider;// separator
   int ReadStartIterator=0;// reading start index
   int quantityiterator=0;// quantity
   bool bNew;
   BasicValue b0;

   for ( int i=0; i<ArraySize(BasicPairsLeft); i++ )//reset the array of base pairs
      {
      BasicPairsLeft[i].Value = "";
      BasicPairsLeft[i].Quantity = 0;
      }
   for ( int i=0; i<ArraySize(BasicPairsRight); i++ )// resetting the array of base pairs
      {
      BasicPairsRight[i].Value = "";
      BasicPairsRight[i].Quantity = 0;
      }
   //// Calculate balance values for the left side
   quantityiterator=0;
   ReadStartIterator=0;
   for ( int i=ReadStartIterator; i<StringLen(CheckedPair.LeftSide); i++ )// extract base currencies from the left side of the equation
      {
      Divider=StringSubstr(CheckedPair.LeftSide,i,1);
      if ( Divider == "^" || i == StringLen(CheckedPair.LeftSide) - 1 )
         {
         SubPair=StringSubstr(CheckedPair.LeftSide,ReadStartIterator+PrefixE,6);
         SubSide=StringSubstr(CheckedPair.LeftSideStructure,quantityiterator,1);
         Half1=StringSubstr(CheckedPair.LeftSide,ReadStartIterator+PrefixE,3);
         Half2=StringSubstr(CheckedPair.LeftSide,ReadStartIterator+PrefixE+3,3);

         bNew=true;
         for ( int j=0; j<ArraySize(BasicPairsLeft); j++ )// if the currency is not found in the list, add it
            {
            if ( BasicPairsLeft[j].Value == Half1 )
               {
               if ( SubSide == "u" ) BasicPairsLeft[j].Quantity++;
               if ( SubSide == "d" ) BasicPairsLeft[j].Quantity--;
               bNew = false;
               break;
               }
            }
         if ( bNew )
            {
            for ( int j=0; j<ArraySize(BasicPairsLeft); j++ )// if the currency is not found in the list, add it
               {
               if ( StringLen(BasicPairsLeft[j].Value) == 0 )
                  {
                  if ( SubSide == "u" ) BasicPairsLeft[j].Quantity++;
                  if ( SubSide == "d" ) BasicPairsLeft[j].Quantity--;
                  BasicPairsLeft[j].Value=Half1;
                  break;
                  }
               }
            }

         bNew=true;
         for ( int j=0; j<ArraySize(BasicPairsLeft); j++ )// if the currency is not found in the list, add it
            {
            if ( BasicPairsLeft[j].Value == Half2 )
               {
               if ( SubSide == "u" ) BasicPairsLeft[j].Quantity--;
               if ( SubSide == "d" ) BasicPairsLeft[j].Quantity++;
               bNew = false;
               break;
               }
            }
         if ( bNew )
            {
            for ( int j=0; j<ArraySize(BasicPairsLeft); j++ )// if the currency is not found in the list, add it
               {
               if (  StringLen(BasicPairsLeft[j].Value) == 0 )
                  {
                  if ( SubSide == "u" ) BasicPairsLeft[j].Quantity--;
                  if ( SubSide == "d" ) BasicPairsLeft[j].Quantity++;
                  BasicPairsLeft[j].Value=Half2;
                  break;
                  }
               }
            }

         ReadStartIterator=i+1;
         quantityiterator++;
         }
      }
   /// end of left-side balance calculation

   //// Calculate balance values for the right side
   quantityiterator=0;
   ReadStartIterator=0;
   for ( int i=ReadStartIterator; i<StringLen(CheckedPair.RightSide); i++ )// extract base currencies from the right side of the equation
      {
      Divider=StringSubstr(CheckedPair.RightSide,i,1);

      if ( Divider == "^"|| i == StringLen(CheckedPair.RightSide) - 1 )
         {
         SubPair=StringSubstr(CheckedPair.RightSide,ReadStartIterator+PrefixE,6);
         SubSide=StringSubstr(CheckedPair.RightSideStructure,quantityiterator,1);
         Half1=StringSubstr(CheckedPair.RightSide,ReadStartIterator+PrefixE,3);
         Half2=StringSubstr(CheckedPair.RightSide,ReadStartIterator+PrefixE+3,3);

         bNew=true;
         for ( int j=0; j<ArraySize(BasicPairsRight); j++ )// if the currency is not found in the list, add it
            {
            if ( BasicPairsRight[j].Value == Half1 )
               {
               if ( SubSide == "u" ) BasicPairsRight[j].Quantity++;
               if ( SubSide == "d" ) BasicPairsRight[j].Quantity--;
               bNew = false;
               break;
               }
            }
         if ( bNew )
            {
            for ( int j=0; j<ArraySize(BasicPairsRight); j++ )// if the currency is not found in the list, add it
               {
               if (  StringLen(BasicPairsRight[j].Value) == 0 )
                  {
                  if ( SubSide == "u" ) BasicPairsRight[j].Quantity++;
                  if ( SubSide == "d" ) BasicPairsRight[j].Quantity--;
                  BasicPairsRight[j].Value=Half1;
                  break;
                  }
               }
            }

         bNew=true;
         for ( int j=0; j<ArraySize(BasicPairsRight); j++ )// if the currency is not found in the list, add it
            {
            if ( BasicPairsRight[j].Value == Half2 )
               {
               if ( SubSide == "u" ) BasicPairsRight[j].Quantity--;
               if ( SubSide == "d" ) BasicPairsRight[j].Quantity++;
               bNew = false;
               break;
               }
            }
         if ( bNew )
            {
            for ( int j=0; j<ArraySize(BasicPairsRight); j++ )// if the currency is not found in the list, add it
               {
               if (  StringLen(BasicPairsRight[j].Value) == 0 )
                  {
                  if ( SubSide == "u" ) BasicPairsRight[j].Quantity--;
                  if ( SubSide == "d" ) BasicPairsRight[j].Quantity++;
                  BasicPairsRight[j].Value=Half2;
                  break;
                  }
               }
            }

         ReadStartIterator=i+1;
         quantityiterator++;
         }
      }
   /// end of right-side balance calculation

   /// calculate the number of lower and upper currencies based on the received data from the previous block
   int LeftUpTotal=0;// the number of upper elements in the left part
   int LeftDownTotal=0;// the number of lower elements in the left part
   int RightUpTotal=0;// the number of upper elements in the right part
   int RightDownTotal=0;// the number of lower elements in the right part


   string LastUpLeft;
   string LastDownLeft;
   string LastUpRight;
   string LastDownRight;
   for ( int i=0; i<ArraySize(BasicPairsLeft); i++ )
      {
      if ( BasicPairsLeft[i].Quantity > 0 && StringLen(BasicPairsLeft[i].Value) > 0 ) LeftUpTotal+=BasicPairsLeft[i].Quantity;
      if ( BasicPairsLeft[i].Quantity < 0 && StringLen(BasicPairsLeft[i].Value) > 0 ) LeftDownTotal-=BasicPairsLeft[i].Quantity;
      }
   for ( int i=0; i<ArraySize(BasicPairsRight); i++ )
      {
      if ( BasicPairsRight[i].Quantity > 0 && StringLen(BasicPairsRight[i].Value) > 0 ) RightUpTotal+=BasicPairsRight[i].Quantity;
      if ( BasicPairsRight[i].Quantity < 0 && StringLen(BasicPairsRight[i].Value) > 0 ) RightDownTotal-=BasicPairsRight[i].Quantity;
      }
   ///
   /// check if both sides are equal
   if ( LeftUpTotal == 1 && LeftDownTotal == 1 && RightUpTotal == 1 && RightDownTotal == 1 )// there must be one pair in the upper and in the lower part of both sides of the equality, otherwise the formula is invalid
      {
      for ( int i=0; i<ArraySize(BasicPairsLeft); i++ )
         {
         if ( BasicPairsLeft[i].Quantity == 1 && StringLen(BasicPairsLeft[i].Value) > 0 ) LastUpLeft=BasicPairsLeft[i].Value;
         if ( BasicPairsLeft[i].Quantity == -1 && StringLen(BasicPairsLeft[i].Value) > 0 ) LastDownLeft=BasicPairsLeft[i].Value;
         }
      for ( int i=0; i<ArraySize(BasicPairsRight); i++ )
         {
         if ( BasicPairsRight[i].Quantity == 1 && StringLen(BasicPairsRight[i].Value) > 0 ) LastUpRight=BasicPairsRight[i].Value;
         if ( BasicPairsRight[i].Quantity == -1 && StringLen(BasicPairsRight[i].Value) > 0 ) LastDownRight=BasicPairsRight[i].Value;
         }
      }
   else return false;
   if ( (LastUpLeft == LastUpRight && LastDownLeft == LastDownRight) || (LastUpLeft == LastDownRight && LastDownLeft == LastUpRight) )
      {
      if ( LastUpLeft == LastDownRight && LastDownLeft == LastUpRight )// If the formula is cross-equivalent, then invert the structure of the right part of the equation (it is the same as raising to the power of -1)
         {
         string NewStructure;// the new structure that will be built from the previous one
         for ( int i=0; i<StringLen(CheckedPair.RightSideStructure); i++ )
            {
            if ( CheckedPair.RightSideStructure[i] == 'u' ) NewStructure+="d";
            if ( CheckedPair.RightSideStructure[i] == 'd' ) NewStructure+="u";
            }
         CheckedPair.RightSideStructure=NewStructure;
         }
      }
   else return false;// if the resulting fractions on both sides are not equivalent, then the formula is invalid
   if ( LastUpLeft == LastDownLeft ) return false;// if result in one, then the formula is invalid

  /// Now it is necessary to write all the above into a corrected and more convenient structure
   string TempResult=CorrectedResultInstrument(LastUpLeft+LastDownLeft,r.IsResultInstrument);
   if ( r.IsResultInstrument && LastUpLeft+LastDownLeft != TempResult )
      {
      string NewStructure="";// the new structure that will be built from the previous one
      for ( int i=0; i<StringLen(CheckedPair.RightSideStructure); i++ )
         {
         if ( CheckedPair.RightSideStructure[i] == 'u' ) NewStructure+="d";
         if ( CheckedPair.RightSideStructure[i] == 'd' ) NewStructure+="u";
         }
      CheckedPair.RightSideStructure=NewStructure;
      NewStructure="";// the new structure that will be built from the previous one
      for ( int i=0; i<StringLen(CheckedPair.LeftSideStructure); i++ )
         {
         if ( CheckedPair.LeftSideStructure[i] == 'u' ) NewStructure+="d";
         if ( CheckedPair.LeftSideStructure[i] == 'd' ) NewStructure+="u";
         }
      CheckedPair.LeftSideStructure=NewStructure;

      r.ResultInstrument=LastDownLeft+LastUpLeft;
      r.UpPair=LastDownLeft;
      r.DownPair=LastUpLeft;
      }
   else
      {
      r.ResultInstrument=LastUpLeft+LastDownLeft;
      r.UpPair=LastUpLeft;
      r.DownPair=LastDownLeft;
      }

   r.LeftSide=CheckedPair.LeftSide;
   r.RightSide=CheckedPair.RightSide;
   r.LeftSideStructure=CheckedPair.LeftSideStructure;
   r.RightSideStructure=CheckedPair.RightSideStructure;
   ///

   /// if code has reached this point, it is considered that we have found the formula meeting the criteria, and the next step is normalization

   return true;
   }
```

The function highlighted in green is needed in order to determine whether the list of symbols contains the one to which the formula was reduced. It may turn out that the formula has been reduced, for example, not to "USDJPY", but to "JPYUSD". Such a symbol obviously does not exist, even though it can be created. But our task is to amend the formula so that it produces a correct trading instrument. In this case, both parts of the formula should be risen to the power of -1, which is equivalent to reversing the structure of the formula (change "d" to "u" and vice versa). If there is no such a symbol in the Market Watch window, then leave it as is:

```
string CorrectedResultInstrument(string instrument, bool &bResult)// if any equivalent symbol corresponds to the generated formula, return this symbol (or leave as is)
   {
   string Half1="";
   string Half2="";
   string Half1input=StringSubstr(instrument,0,3);//input upper currency
   string Half2input=StringSubstr(instrument,3,3);//input lower currency
   bResult=false;
   for ( int j=0; j<ArraySize(Pairs); j++ )
      {
      Half1=StringSubstr(Pairs[j].Name,PrefixE,3);
      Half2=StringSubstr(Pairs[j].Name,PrefixE+3,3);
      if ( (Half1==Half1input && Half2==Half2input) || (Half1==Half2input && Half2==Half1input) )// direct match or crossed match
         {
         bResult=true;
         return Pairs[j].Name;
         }
      }

   return instrument;
   }
```

I have prepared the following structure to store the formulas than have passed though the filter. The structure has some fields from the previous one and some new fields:

```
struct EquationCorrected // corrected structure of the basic formula
   {
   string LeftSide;// currency pairs participating in the formula on the left side of the "=" sign
   string LeftSideStructure;// structure of the left side of the formula
   string RightSide;// currency pairs participating in the right side of the formula
   string RightSideStructure;// structure of the right side of the formula

   string ResultInstrument;// the resulting instrument to which both parts of the formula come after transformation
   bool IsResultInstrument;// has the suitable equivalent symbol been found
   string UpPair;// the upper currency of the resulting instrument
   string DownPair;// the lower currency of the resulting instrument
   };
```

**Normalizing Formulas**

This procedure is the next step in filtering the results. It consists of the following sequential operations, which follow one after the other:

1. Based on the resulting symbol obtained from both sides of the equation, select a starting pair from the list for both sides of the equality.
2. Both pairs, according to their power in the equation, must provide the base currency in the fraction numerator
3. If such a pair is found, and the lower currency of the fraction does not contain the main currency of the resulting instrument go further
4. Further we go so that the upper currency of the next pair compensates for the lower currency of the previous one
5. Repeat these steps until the desired resulting pair is found
6. Once the resulting pair is found, all unused components of the formula are discarded (as their product is one)
7. In parallel to this process, "lot factors" are calculated sequentially from pair to pair (they show which lot you need to open positions for specific pairs, to ensure our resulting instrument)
8. The result is written into a new variable, which will be used in the next analysis stage.

The function code is as follows:

```
bool bNormalized(EquationCorrected &d,EquationNormalized &v)// formula normalization attempt (the normalized formula is returned in "v" )
   {
   double PreviousContract;// previous contract
   bool bWasPairs;// if any pairs have been found
   double BaseContract;// contract of the pair to which the equation is reduced
   double PreviousLotK=0.0;// previous LotK
   double LotK;// current LotK
   string PreviousSubSide;// in numerator or denominator (previous factor)
   string PreviousPair;// previous pair
   string PreviousHalf1;// upper currency of the previous pair
   string PreviousHalf2;// lower currency of the previous pair
   string SubPair;// the full name of the currency pair
   string Half1;// the first currency of the pair
   string Half2;// the second currency of the pair
   string SubSide;// the currency pair in the numerator or denominator
   string Divider;// separator
   int ReadStartIterator=0;// reading start index
   int quantityiterator=0;// quantity
   int tryiterator=0;// the number of balancing attempts
   int quantityleft=0;// the number of pairs on the left after normalization
   int quantityright=0;//the number of pairs on the right after normalization
   bool bNew;
   BasicValue b0;

   for ( int i=0; i<ArraySize(BasicPairsLeft); i++ )//reset the array of base pairs
      {
      BasicPairsLeft[i].Value = "";
      BasicPairsLeft[i].Quantity = 0;
      }
   for ( int i=0; i<ArraySize(BasicPairsRight); i++ )// resetting the array of base pairs
      {
      BasicPairsRight[i].Value = "";
      BasicPairsRight[i].Quantity = 0;
      }

   if ( d.IsResultInstrument ) BaseContract=SymbolInfoDouble(d.ResultInstrument, SYMBOL_TRADE_CONTRACT_SIZE);// define the contract of the equivalent pair based on the instrument
   else BaseContract=100000.0;

   //// Calculate the number of pairs for the left side
   tryiterator=0;
   ReadStartIterator=0;
   for ( int i=ReadStartIterator; i<StringLen(d.LeftSide); i++ )// extract base currencies from the left side of the equation
      {
      Divider=StringSubstr(d.LeftSide,i,1);
      if ( Divider == "^" )
         {
         ReadStartIterator=i+1;
         tryiterator++;
         }

      if ( i == StringLen(d.LeftSide) - 1 )
         {
         ReadStartIterator=i+1;
         tryiterator++;
         }
      }
   /// end of quantity calculation for the left part

   ArrayResize(v.PairLeft,tryiterator);
   /// calculate the lot coefficients for the left side

   bool bBalanced=false;// is the formula balanced
   bool bUsed[];
   ArrayResize(bUsed,tryiterator);
   ArrayFill(bUsed,0,tryiterator,false);
   int balancediterator=0;
   PreviousHalf1="";
   PreviousHalf2="";
   PreviousLotK=0.0;
   PreviousSubSide="";
   PreviousPair="";
   PreviousContract=0.0;
   bWasPairs=false;// have there been pairs
   for ( int k=0; k<tryiterator; k++ )// try to normalize the left side
      {
      if( !bBalanced )
         {
         quantityiterator=0;
         ReadStartIterator=0;
         for ( int i=ReadStartIterator; i<StringLen(d.LeftSide); i++ )// extract base currencies from the left side of the equation
            {
            Divider=StringSubstr(d.LeftSide,i,1);
            if ( Divider == "^" || i == StringLen(d.LeftSide) - 1 )
               {
               SubPair=StringSubstr(d.LeftSide,ReadStartIterator+PrefixE,6);
               SubSide=StringSubstr(d.LeftSideStructure,quantityiterator,1);
               Half1=StringSubstr(d.LeftSide,ReadStartIterator+PrefixE,3);
               Half2=StringSubstr(d.LeftSide,ReadStartIterator+PrefixE+3,3);

               if ( ! bUsed[quantityiterator] && (( PreviousHalf1 == "" && ((Half1 == d.UpPair && SubSide == "u") || (Half2 == d.UpPair && SubSide == "d")) ) // if it is the first pair in the list
               || ( (( PreviousHalf2 == Half1 && PreviousSubSide == "u" ) || ( PreviousHalf1 == Half1 && PreviousSubSide == "d" )) && SubSide == "u" ) // if the current pair is in the numerator
               || ( (( PreviousHalf2 == Half2 && PreviousSubSide == "u" ) || ( PreviousHalf1 == Half2 && PreviousSubSide == "d" )) && SubSide == "d" )) )// if the current pair is in the denominator
                  {// find the entry point(pair) of the chain
                  if( PreviousHalf1 == "" )// define the lot coefficient of the first pair
                     {
                     if ( SubSide == "u" )
                        {
                        LotK=BaseContract/SymbolInfoDouble(SubPair, SYMBOL_TRADE_CONTRACT_SIZE);// (1 start)
                        v.PairLeft[balancediterator].LotK=LotK;
                        PreviousLotK=LotK;
                        bWasPairs=true;
                        PreviousContract=SymbolInfoDouble(SubPair, SYMBOL_TRADE_CONTRACT_SIZE);
                        }
                     if ( SubSide == "d" )
                        {
                        double Pt=SymbolInfoDouble(SubPair,SYMBOL_BID);
                        if ( Pt == 0.0 ) return false;
                        LotK=(BaseContract/SymbolInfoDouble(SubPair, SYMBOL_TRADE_CONTRACT_SIZE))/Pt;// (2 start)
                        v.PairLeft[balancediterator].LotK=LotK;
                        PreviousLotK=LotK;
                        bWasPairs=true;
                        PreviousContract=SymbolInfoDouble(SubPair, SYMBOL_TRADE_CONTRACT_SIZE);
                        }
                     }
                  else
                     {
                     if( PreviousSubSide == "u" )// define the lot coefficient of further pairs
                        {
                        if ( SubSide == "u" )
                           {
                           double Pp=SymbolInfoDouble(PreviousPair,SYMBOL_BID);
                           if ( Pp == 0.0 ) return false;
                           if ( PreviousContract <= 0.0 ) return false;
                           LotK=PreviousLotK*Pp*(PreviousContract/SymbolInfoDouble(SubPair, SYMBOL_TRADE_CONTRACT_SIZE));// ( 1 )
                           v.PairLeft[balancediterator].LotK=LotK;
                           PreviousLotK=LotK;
                           PreviousContract=SymbolInfoDouble(SubPair, SYMBOL_TRADE_CONTRACT_SIZE);
                           }
                        if ( SubSide == "d" )
                           {
                           double Pt=SymbolInfoDouble(SubPair,SYMBOL_BID);
                           double Pp=SymbolInfoDouble(PreviousPair,SYMBOL_BID);
                           if ( Pt == 0.0 ) return false;
                           if ( Pp == 0.0 ) return false;
                           if ( PreviousContract <= 0.0 ) return false;
                           LotK=PreviousLotK*(Pp/Pt)*(PreviousContract/SymbolInfoDouble(SubPair, SYMBOL_TRADE_CONTRACT_SIZE));// ( 2 )
                           v.PairLeft[balancediterator].LotK=LotK;
                           PreviousLotK=LotK;
                           PreviousContract=SymbolInfoDouble(SubPair, SYMBOL_TRADE_CONTRACT_SIZE);
                           }
                        }
                     if( PreviousSubSide == "d" )// define the lot coefficient of further pairs
                        {
                        if ( SubSide == "u" )
                           {
                           if ( PreviousContract <= 0.0 ) return false;
                           LotK=PreviousLotK*(PreviousContract/SymbolInfoDouble(SubPair, SYMBOL_TRADE_CONTRACT_SIZE));// ( 3 )
                           v.PairLeft[balancediterator].LotK=LotK;
                           PreviousLotK=LotK;
                           PreviousContract=SymbolInfoDouble(SubPair, SYMBOL_TRADE_CONTRACT_SIZE);
                           }
                        if ( SubSide == "d" )
                           {
                           double Pt=SymbolInfoDouble(SubPair,SYMBOL_BID);
                           if ( Pt == 0.0 ) return false;
                           if ( PreviousContract <= 0.0 ) return false;
                           LotK=(PreviousLotK/Pt)*(PreviousContract/SymbolInfoDouble(SubPair, SYMBOL_TRADE_CONTRACT_SIZE));// ( 4 )
                           v.PairLeft[balancediterator].LotK=LotK;
                           PreviousLotK=LotK;
                           PreviousContract=SymbolInfoDouble(SubPair, SYMBOL_TRADE_CONTRACT_SIZE);
                           }
                        }
                     }

                  bNew=true;
                  for ( int j=0; j<ArraySize(BasicPairsLeft); j++ )// if the currency is not found in the list, add it
                     {
                     if ( BasicPairsLeft[j].Value == Half1 )
                        {
                        if ( SubSide == "u" ) BasicPairsLeft[j].Quantity++;
                        if ( SubSide == "d" ) BasicPairsLeft[j].Quantity--;
                        bNew = false;
                        break;
                        }
                     }
                  if ( bNew )
                     {
                     for ( int j=0; j<ArraySize(BasicPairsLeft); j++ )// if the currency is not found in the list, add it
                        {
                        if ( StringLen(BasicPairsLeft[j].Value) == 0 )
                           {
                           if ( SubSide == "u" ) BasicPairsLeft[j].Quantity++;
                           if ( SubSide == "d" ) BasicPairsLeft[j].Quantity--;
                           BasicPairsLeft[j].Value=Half1;
                           break;
                           }
                        }
                     }

                  bNew=true;
                  for ( int j=0; j<ArraySize(BasicPairsLeft); j++ )// if the currency is not found in the list, add it
                     {
                     if ( BasicPairsLeft[j].Value == Half2 )
                        {
                        if ( SubSide == "u" ) BasicPairsLeft[j].Quantity--;
                        if ( SubSide == "d" ) BasicPairsLeft[j].Quantity++;
                        bNew = false;
                        break;
                        }
                     }
                  if ( bNew )
                     {
                     for ( int j=0; j<ArraySize(BasicPairsLeft); j++ )// if the currency is not found in the list, add it
                        {
                        if (  StringLen(BasicPairsLeft[j].Value) == 0 )
                           {
                           if ( SubSide == "u" ) BasicPairsLeft[j].Quantity--;
                           if ( SubSide == "d" ) BasicPairsLeft[j].Quantity++;
                           BasicPairsLeft[j].Value=Half2;
                           break;
                           }
                        }
                     }

                  v.PairLeft[balancediterator].Name=SubPair;
                  v.PairLeft[balancediterator].Side=SubSide;
                  v.PairLeft[balancediterator].ContractSize=SymbolInfoDouble(v.PairLeft[balancediterator].Name, SYMBOL_TRADE_CONTRACT_SIZE);


                  balancediterator++;
                  PreviousHalf1=Half1;
                  PreviousHalf2=Half2;
                  PreviousSubSide=SubSide;
                  PreviousPair=SubPair;

                  quantityleft++;
                  if ( SubSide == "u" && Half2 == d.DownPair )// if the fraction is not inverted
                     {
                     bBalanced=true;// if the missing part is in the denominator, then we have balanced the formula
                     break;// since the formula is balanced, we don't need the rest
                     }
                  if ( SubSide == "d" && Half1 == d.DownPair )// if the fraction is inverted
                     {
                     bBalanced=true;// if the missing part is in the numerator, then we have balanced the formula
                     break;// since the formula is balanced, we don't need the rest
                     }

                  int LeftUpTotal=0;// the number of upper elements in the left part
                  int LeftDownTotal=0;// the number of lower elements in the left part
                  string LastUpLeft;
                  string LastDownLeft;
                  for ( int z=0; z<ArraySize(BasicPairsLeft); z++ )
                     {
                     if ( BasicPairsLeft[z].Quantity > 0 && StringLen(BasicPairsLeft[z].Value) > 0 ) LeftUpTotal+=BasicPairsLeft[z].Quantity;
                     if ( BasicPairsLeft[z].Quantity < 0 && StringLen(BasicPairsLeft[z].Value) > 0 ) LeftDownTotal-=BasicPairsLeft[z].Quantity;
                     }
                  if ( bWasPairs && LeftUpTotal == 0 && LeftDownTotal == 0 ) return false;
                  }

               ReadStartIterator=i+1;
               bUsed[quantityiterator]=true;
               quantityiterator++;
               }
            }
         }
         else break;
      }
   /// end of coefficient calculation for the left part

   if ( !bBalanced ) return false;// if the left side is not balanced, then there is no point in balancing the right side

   //// Calculate the number of pairs for the right side
   tryiterator=0;
   ReadStartIterator=0;
   for ( int i=ReadStartIterator; i<StringLen(d.RightSide); i++ )// extract base currencies from the right side of the equation
      {
      Divider=StringSubstr(d.RightSide,i,1);
      if ( Divider == "^" )
         {
         ReadStartIterator=i+1;
         tryiterator++;
         }

      if ( i == StringLen(d.RightSide) - 1 )
         {
         ReadStartIterator=i+1;
         tryiterator++;
         }
      }
   ArrayResize(v.PairRight,tryiterator);
   /// end of calculation of the number of pairs for the right side

   bBalanced=false;// is the formula balanced
   ArrayResize(bUsed,tryiterator);
   ArrayFill(bUsed,0,tryiterator,false);
   balancediterator=0;
   PreviousHalf1="";
   PreviousHalf2="";
   PreviousLotK=0.0;
   PreviousSubSide="";
   PreviousPair="";
   PreviousContract=0.0;
   bWasPairs=false;
   for ( int k=0; k<tryiterator; k++ )// try to normalize the right side
      {
      if ( !bBalanced )
         {
         quantityiterator=0;
         ReadStartIterator=0;
         for ( int i=ReadStartIterator; i<StringLen(d.RightSide); i++ )// extract base currencies from the right side of the equation
            {
            Divider=StringSubstr(d.RightSide,i,1);
            if ( Divider == "^" || i == StringLen(d.RightSide) - 1 )
               {
               SubPair=StringSubstr(d.RightSide,ReadStartIterator+PrefixE,6);
               SubSide=StringSubstr(d.RightSideStructure,quantityiterator,1);
               Half1=StringSubstr(d.RightSide,ReadStartIterator+PrefixE,3);
               Half2=StringSubstr(d.RightSide,ReadStartIterator+PrefixE+3,3);

               if ( ! bUsed[quantityiterator] && (( PreviousHalf1 == "" && ((Half1 == d.UpPair && SubSide == "u") || (Half2 == d.UpPair && SubSide == "d")) ) // if it is the first pair in the list
               || ( (( PreviousHalf2 == Half1 && PreviousSubSide == "u" ) || ( PreviousHalf1 == Half1 && PreviousSubSide == "d" )) && SubSide == "u" ) // if the current pair is in the numerator
               || ( (( PreviousHalf2 == Half2 && PreviousSubSide == "u" ) || ( PreviousHalf1 == Half2 && PreviousSubSide == "d" )) && SubSide == "d" )) )// if the current pair is in the denominator
                  {// find the entry point(pair) of the chain
                  if( PreviousHalf1 == "" )// define the lot coefficient of the first pair
                     {
                     if ( SubSide == "u" )
                        {
                        LotK=BaseContract/SymbolInfoDouble(SubPair, SYMBOL_TRADE_CONTRACT_SIZE);// (1 start)
                        v.PairRight[balancediterator].LotK=LotK;
                        PreviousLotK=LotK;
                        bWasPairs=true;
                        PreviousContract=SymbolInfoDouble(SubPair, SYMBOL_TRADE_CONTRACT_SIZE);
                        }
                     if ( SubSide == "d" )
                        {
                        double Pt=SymbolInfoDouble(SubPair,SYMBOL_BID);
                        if ( Pt == 0.0 ) return false;
                        LotK=(BaseContract/SymbolInfoDouble(SubPair, SYMBOL_TRADE_CONTRACT_SIZE))/Pt;// (2 start)
                        v.PairRight[balancediterator].LotK=LotK;
                        PreviousLotK=LotK;
                        bWasPairs=true;
                        PreviousContract=SymbolInfoDouble(SubPair, SYMBOL_TRADE_CONTRACT_SIZE);
                        }
                     }
                  else
                     {
                     if( PreviousSubSide == "u" )// define the lot coefficient of further pairs
                        {
                        if ( SubSide == "u" )
                           {
                           double Pp=SymbolInfoDouble(PreviousPair,SYMBOL_BID);
                           if ( Pp == 0.0 ) return false;
                           if ( PreviousContract <= 0.0 ) return false;
                           LotK=PreviousLotK*Pp*(PreviousContract/SymbolInfoDouble(SubPair, SYMBOL_TRADE_CONTRACT_SIZE));// (1)
                           v.PairRight[balancediterator].LotK=LotK;
                           PreviousLotK=LotK;
                           PreviousContract=SymbolInfoDouble(SubPair, SYMBOL_TRADE_CONTRACT_SIZE);
                           }
                        if ( SubSide == "d" )
                           {
                           double Pt=SymbolInfoDouble(SubPair,SYMBOL_BID);
                           double Pp=SymbolInfoDouble(PreviousPair,SYMBOL_BID);
                           if ( Pt == 0.0 ) return false;
                           if ( Pp == 0.0 ) return false;
                           if ( PreviousContract <= 0.0 ) return false;
                           LotK=PreviousLotK*(Pp/Pt)*(PreviousContract/SymbolInfoDouble(SubPair, SYMBOL_TRADE_CONTRACT_SIZE));// (2)
                           v.PairRight[balancediterator].LotK=LotK;
                           PreviousLotK=LotK;
                           PreviousContract=SymbolInfoDouble(SubPair, SYMBOL_TRADE_CONTRACT_SIZE);
                           }
                        }
                     if( PreviousSubSide == "d" )// define the lot coefficient of further pairs
                        {
                        if ( SubSide == "u" )
                           {
                           if ( PreviousContract <= 0.0 ) return false;
                           LotK=PreviousLotK*(PreviousContract/SymbolInfoDouble(SubPair, SYMBOL_TRADE_CONTRACT_SIZE));// (3)
                           v.PairRight[balancediterator].LotK=LotK;
                           PreviousLotK=LotK;
                           PreviousContract=SymbolInfoDouble(SubPair, SYMBOL_TRADE_CONTRACT_SIZE);
                           }
                        if ( SubSide == "d" )
                           {
                           double Pt=SymbolInfoDouble(SubPair,SYMBOL_BID);
                           if ( Pt == 0.0 ) return false;
                           if ( PreviousContract <= 0.0 ) return false;
                           LotK=(PreviousLotK/Pt)*(PreviousContract/SymbolInfoDouble(SubPair, SYMBOL_TRADE_CONTRACT_SIZE));// (4)
                           v.PairRight[balancediterator].LotK=LotK;
                           PreviousLotK=LotK;
                           PreviousContract=SymbolInfoDouble(SubPair, SYMBOL_TRADE_CONTRACT_SIZE);
                           }
                        }
                     }
                  bNew=true;
                  for ( int j=0; j<ArraySize(BasicPairsRight); j++ )// if the currency is not found in the list, add it
                     {
                     if ( BasicPairsRight[j].Value == Half1 )
                        {
                        if ( SubSide == "u" ) BasicPairsRight[j].Quantity++;
                        if ( SubSide == "d" ) BasicPairsRight[j].Quantity--;
                        bNew = false;
                        break;
                        }
                     }
                  if ( bNew )
                     {
                     for ( int j=0; j<ArraySize(BasicPairsLeft); j++ )// if the currency is not found in the list, add it
                        {
                        if ( StringLen(BasicPairsLeft[j].Value) == 0 )
                           {
                           if ( SubSide == "u" ) BasicPairsRight[j].Quantity++;
                           if ( SubSide == "d" ) BasicPairsRight[j].Quantity--;
                           BasicPairsRight[j].Value=Half1;
                           break;
                           }
                        }
                     }

                  bNew=true;
                  for ( int j=0; j<ArraySize(BasicPairsRight); j++ )// if the currency is not found in the list, add it
                     {
                     if ( BasicPairsRight[j].Value == Half2 )
                        {
                        if ( SubSide == "u" ) BasicPairsRight[j].Quantity--;
                        if ( SubSide == "d" ) BasicPairsRight[j].Quantity++;
                        bNew = false;
                        break;
                        }
                     }
                  if ( bNew )
                     {
                     for ( int j=0; j<ArraySize(BasicPairsRight); j++ )// if the currency is not found in the list, add it
                        {
                        if (  StringLen(BasicPairsRight[j].Value) == 0 )
                           {
                           if ( SubSide == "u" ) BasicPairsRight[j].Quantity--;
                           if ( SubSide == "d" ) BasicPairsRight[j].Quantity++;
                           BasicPairsRight[j].Value=Half2;
                           break;
                           }
                        }
                     }

                  v.PairRight[balancediterator].Name=SubPair;
                  v.PairRight[balancediterator].Side=SubSide;
                  v.PairRight[balancediterator].ContractSize=SymbolInfoDouble(v.PairRight[balancediterator].Name, SYMBOL_TRADE_CONTRACT_SIZE);

                  balancediterator++;
                  PreviousHalf1=Half1;
                  PreviousHalf2=Half2;
                  PreviousSubSide=SubSide;
                  PreviousPair=SubPair;

                  quantityright++;
                  if ( SubSide == "u" && Half2 == d.DownPair )// if the fraction is not inverted
                     {
                     bBalanced=true;// if the missing part is in the denominator, then we have balanced the formula
                     break;// since the formula is balanced, we don't need the rest
                     }
                  if ( SubSide == "d" && Half1 == d.DownPair )// if the fraction is inverted
                     {
                     bBalanced=true;// if the missing part is in the numerator, then we have balanced the formula
                     break;// since the formula is balanced, we don't need the rest
                     }

                  int RightUpTotal=0;// the number of upper elements in the right part
                  int RightDownTotal=0;// the number of lower elements in the right part
                  string LastUpRight;
                  string LastDownRight;

                  for ( int z=0; z<ArraySize(BasicPairsRight); z++ )
                     {
                     if ( BasicPairsRight[z].Quantity > 0 && StringLen(BasicPairsRight[z].Value) > 0 ) RightUpTotal+=BasicPairsRight[z].Quantity;
                     if ( BasicPairsRight[z].Quantity < 0 && StringLen(BasicPairsRight[z].Value) > 0 ) RightDownTotal-=BasicPairsRight[z].Quantity;
                     }
                  if ( bWasPairs && RightUpTotal == 0 && RightDownTotal == 0 ) return false;
                  }

               ReadStartIterator=i+1;
               bUsed[quantityiterator]=true;
               quantityiterator++;
               }
            }
         }
         else break;
      }

   if ( quantityleft == 1 && quantityright == 1 ) return false;// if the equation has been normalized to only 2 pairs, it is not valid (at least 3 pairs are required)

   return bBalanced;
   }
```

This is a very healthy and complex procedure, but it seems to me that in such cases it is better not to produce intermediate states, because this will lead to significant code duplication. Furthermore, all stages are very compact and they are logically divided into blocks. The set of these functions give us absolutely identical results comparable with the results which we could obtain from performing all conversions in handwriting. Here, all these mathematical manipulations are performed by a set of complex but necessary methods.

We need to perform a special analysis in order to understand how profitable the found formula is. Do not forget that for each pair it is possible to open a position both up and down. Accordingly, there can be two intermediate variants of the circuits for each formula - direct and reverse. The one with the higher profitability will be accepted as the result.

To assess the profitability, I have created a metric similar to the profit factor, which consists of the profits and losses resulting from swaps. If the accrued positive swap of the existing circuit is greater than the negative modulus, then such a circuit is considered profitable. In other cases, such circuits are unprofitable — in other words, the swap factor of our contour will be positive only when it is greater than one.

The returned result is written into a completely different container, which has been created as a self-sufficient command package for trading and for further development of the trading logic. It contains everything needed to quickly and easily open the entire circuit:

```
struct EquationNormalized // the final structure with the formula in normalized form
   {
   Pair PairLeft[];// currency pairs on the left side
   Pair PairRight[];// currency pairs on the right side
   double SwapPlusRelative;// relative equivalent of the positive swap
   double SwapMinusRelative;// relative equivalent of the negative swap
   double SwapFactor;// resulting swap factor
   };
```

I have also added two methods which enable the convenient display of information about the contents, but they are not relevant for this article and thus I will not provide them here. You can view them in the attached source code. Now, information about each component of the equation is contained separately as elements of arrays. This provides easier work with the data later without the need to constantly parse them from strings. Perhaps, this solution could be used from the very beginning, but this would have spoiled the readability.

**Calculation of the swap factor and final adjustment of the equation structure**

This is the last stage, in which the most important variable of this system is calculated — the variants will be compared according to this value. The one with the highest value is the best.

```
void CalculateBestVariation(EquationNormalized &ii)// calculation of the best swap factor of the formula and final structure adjustment if needed
   {
   double SwapMinus=0.0;// total negative swap
   double SwapPlus=0.0;// total positive swap
   double SwapMinusReverse=0.0;// total negative swap
   double SwapPlusReverse=0.0;// total positive swap

   double SwapFactor=0.0;// swap factor of the direct pass
   double SwapFactorReverse=0.0;// swap factor of the reverse pass

   for ( int i=0; i<ArraySize(ii.PairLeft); i++ )// define the missing parameters for calculating the left side
      {
      for ( int j=0; j<ArraySize(Pairs); j++ )
         {
         if ( Pairs[j].Name == ii.PairLeft[i].Name )
            {
            ii.PairLeft[i].Margin=Pairs[j].Margin;
            ii.PairLeft[i].TickValue=Pairs[j].TickValue;
            ii.PairLeft[i].SwapBuy=Pairs[j].SwapBuy;
            ii.PairLeft[i].SwapSell=Pairs[j].SwapSell;
            break;
            }
         }
      }

   for ( int i=0; i<ArraySize(ii.PairRight); i++ )// define the missing parameters for calculating the right side
      {
      for ( int j=0; j<ArraySize(Pairs); j++ )
         {
         if ( Pairs[j].Name == ii.PairRight[i].Name )
            {
            ii.PairRight[i].Margin=Pairs[j].Margin;
            ii.PairRight[i].TickValue=Pairs[j].TickValue;
            ii.PairRight[i].SwapBuy=Pairs[j].SwapBuy;
            ii.PairRight[i].SwapSell=Pairs[j].SwapSell;
            break;
            }
         }
      }

   double TempSwap;
   // calculate all components taking into account a change in the structure
   for ( int i=0; i<ArraySize(ii.PairLeft); i++ )// for left parts
      {
      if ( ii.PairLeft[i].Side == "u" )
         {// for direct trading
         TempSwap=ii.PairLeft[i].SwapBuy*ii.LotKLeft[i];
         if ( TempSwap >= 0 ) SwapPlus+=TempSwap;
         else SwapMinus-=TempSwap;
         // for reverse trading
         TempSwap=ii.PairLeft[i].SwapSell*ii.LotKLeft[i];
         if ( TempSwap >= 0 ) SwapPlusReverse+=TempSwap;
         else SwapMinusReverse-=TempSwap;
         }
      if ( ii.PairLeft[i].Side == "d" )
         {// for direct trading
         TempSwap=ii.PairLeft[i].SwapSell*ii.LotKLeft[i];
         if ( TempSwap >= 0 ) SwapPlus+=TempSwap;
         else SwapMinus-=TempSwap;
         // for reverse trading
         TempSwap=ii.PairLeft[i].SwapBuy*ii.LotKLeft[i];
         if ( TempSwap >= 0 ) SwapPlusReverse+=TempSwap;
         else SwapMinusReverse-=TempSwap;
         }
      }

   for ( int i=0; i<ArraySize(ii.PairRight); i++ )// for right parts
      {
      if ( ii.PairRight[i].Side == "d" )
         {// for direct trading
         TempSwap=ii.PairRight[i].SwapBuy*ii.LotKRight[i];
         if ( TempSwap >= 0 ) SwapPlus+=TempSwap;
         else SwapMinus-=TempSwap;
         // for reverse trading
         TempSwap=ii.PairRight[i].SwapSell*ii.LotKRight[i];
         if ( TempSwap >= 0 ) SwapPlusReverse+=TempSwap;
         else SwapMinusReverse-=TempSwap;
         }
      if ( ii.PairRight[i].Side == "u" )
         {// for direct trading
         TempSwap=ii.PairRight[i].SwapSell*ii.LotKRight[i];
         if ( TempSwap >= 0 ) SwapPlus+=TempSwap;
         else SwapMinus-=TempSwap;
         // for reverse trading
         TempSwap=ii.PairRight[i].SwapBuy*ii.LotKRight[i];
         if ( TempSwap >= 0 ) SwapPlusReverse+=TempSwap;
         else SwapMinusReverse-=TempSwap;
         }
      }
   // calculate the swap factor for the direct pass
   if ( SwapMinus > 0.0 && SwapPlus > 0.0 ) SwapFactor=SwapPlus/SwapMinus;
   if ( SwapMinus == 0.0 && SwapPlus == 0.0 ) SwapFactor=1.0;
   if ( SwapMinus == 0.0 && SwapPlus > 0.0 ) SwapFactor=1000001.0;
   if ( SwapMinus > 0.0 && SwapPlus == 0.0 ) SwapFactor=0.0;
   // calculate the swap factor for the reverse pass
   if ( SwapMinusReverse > 0.0 && SwapPlusReverse > 0.0 ) SwapFactorReverse=SwapPlusReverse/SwapMinusReverse;
   if ( SwapMinusReverse == 0.0 && SwapPlusReverse == 0.0 ) SwapFactorReverse=1.0;
   if ( SwapMinusReverse == 0.0 && SwapPlusReverse > 0.0 ) SwapFactorReverse=1000001.0;
   if ( SwapMinusReverse > 0.0 && SwapPlusReverse == 0.0 ) SwapFactorReverse=0.0;
   // select the best approach and calculate the missing values in the structure
   if ( SwapFactor > SwapFactorReverse )
      {
      ii.SwapPlusRelative=SwapPlus;
      ii.SwapMinusRelative=SwapMinus;
      ii.SwapFactor=SwapFactor;
      }
   else
      {
      ii.SwapPlusRelative=SwapPlusReverse;
      ii.SwapMinusRelative=SwapMinusReverse;
      ii.SwapFactor=SwapFactorReverse;
      bool bSigned;
      for ( int i=0; i<ArraySize(ii.PairRight); i++ )// if it is a reverse pass, then reverse the right structure of the formula
         {
         bSigned=false;
         if ( !bSigned && ii.PairRight[i].Side == "u" )
            {
            ii.PairRight[i].Side="d";
            bSigned=true;
            }
         if ( !bSigned && ii.PairRight[i].Side == "d" )
            {
            ii.PairRight[i].Side="u";
            bSigned=true;
            }
         }
      bSigned=false;
      for ( int i=0; i<ArraySize(ii.PairLeft); i++ )// if it is a reverse pass, then reverse the left structure of the formula
         {
         bSigned=false;
         if ( !bSigned && ii.PairLeft[i].Side == "u" )
            {
            ii.PairLeft[i].Side="d";
            bSigned=true;
            }
         if ( !bSigned && ii.PairLeft[i].Side == "d" )
            {
            ii.PairLeft[i].Side="u";
            bSigned=true;
            }
         }
      }

   bool bSigned;
   for ( int i=0; i<ArraySize(ii.PairRight); i++ )// reverse the right side anyway
      {
      bSigned=false;
      if ( !bSigned && ii.PairRight[i].Side == "u" )
         {
         ii.PairRight[i].Side="d";
         bSigned=true;
         }
      if ( !bSigned && ii.PairRight[i].Side == "d" )
         {
         ii.PairRight[i].Side="u";
         bSigned=true;
         }
      }
   }
```

To enable the sequential output of the result, I have implemented the log which is only written if the successfully filtered formula variant is found. The log is as follows:

[![Log](https://c.mql5.com/2/42/56t2.png)](https://c.mql5.com/2/42/6md2.png "https://c.mql5.com/2/42/6md2.png")

Red color is used for the resulting symbol, to which both sides of the equation are reduced. The next line shows the normalized variant with lot coefficients. The third line shows the variant with the calculated swap factor. The fourth line is the best of the variants found during the brute-force session, which is also plotted on the chart plotted by the Comment function. This prototype is attached below, so you can test it. Actually, it can serve as a prototype of a trading assistant for swap trading. As for now, it has little functionality, but I will try to expand it in the next article. The prototype is presented in two versions: for MetaTrader 4 and MetaTrader 5.

### Conclusions from the First Tests

It is quite difficult to draw any conclusions regarding such a complex topic alone. Nevertheless, I managed to understand something useful, although I have not been able to find a swap factor larger than one so far. These are the first conclusions which I came to when analyzing the work of this prototype:

- For some currency pairs, you can increase positive swaps or reduce negative ones (due to the presentation of the position as a synthetic equivalent)
- Even if a profitable circuit is not found, one of its parts can always be used as an alternative position - for locking on two different trading accounts.
- Locking with such a synthetic position eliminates the need to use Swap Free accounts, since it allows having an opposite positive swap at both ends.
- It is necessary to perform better in-depth analysis with the most popular brokers, for which the functionality expansion is needed.
- I hope I will be able to proof that a profitable swap factor can be achieved (which is only a guess so far)
- Swaps can provide small but steady profit if used wisely

### Conclusion

I hope this approach is interesting for you and that it can provide food for thought. The method is very difficult to understand, but it actually implements the simple principle: opening two opposite positions with the same volume. Mere opening of such two opposite positions always generates a loss. There is no broker providing a positive one-way swap greater in modulus than a negative one-way swap. Of course, you will never find positive swaps in both directions, because it is mathematically impossible.

I will not provide the details of underlying mathematics, as it is a very wide topic. It is better to utilize the manifestations of this mathematics. By applying the described method, it is possible to reduce the loss caused by the swap in position locking. You can also try to find a gap in brokers' swap tables and enjoy locking with a positive profit factor (a profitable total swap of all positions) — it is risk-free trading which is not dependent on price fluctuations.

I think that swap trading methods are really underestimated, since a positive swap provides potential profit. The described method is just one of the possible variations of swap trading methods, but I like this task and I will try to continue it in the next articles, developing the idea, modernizing the code and creating new additional functionality. I will also describe some ideas regarding profit forecasting and trading functionality.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/9198](https://www.mql5.com/ru/articles/9198)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/9198.zip "Download all attachments in the single ZIP archive")

[Prototype.zip](https://www.mql5.com/en/articles/download/9198/prototype.zip "Download Prototype.zip")(20.52 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/370528)**
(12)


![pi.xero](https://c.mql5.com/avatar/2020/12/5FE402A7-1902.jpeg)

**[pi.xero](https://www.mql5.com/en/users/pi.xero)**
\|
28 Nov 2021 at 13:33

**MetaQuotes:**

New article [Swaps (Part I): Locking and Synthetic Positions](https://www.mql5.com/en/articles/9198) has been published:

Author: [Evgeniy Ilin](https://www.mql5.com/en/users/W.HUDSON "W.HUDSON")

Thank you for your articles.

![Evgeniy Ilin](https://c.mql5.com/avatar/2021/1/60171FA1-EC8E.png)

**[Evgeniy Ilin](https://www.mql5.com/en/users/w.hudson)**
\|
28 Nov 2021 at 14:57

**pi.xero [#](https://www.mql5.com/en/forum/370528#comment_26147840):**

Thank you for your articles.

You're the welcome ).

![Cesar Afif rezende Oaquim](https://c.mql5.com/avatar/2019/5/5CEEA9EF-F672.jpg)

**[Cesar Afif rezende Oaquim](https://www.mql5.com/en/users/albedo)**
\|
21 Feb 2023 at 20:28

I loved your article on swap, I've been testing some theories on the subject for months, and your article, which I found yesterday, is very complete. I will read the article again more calmly, but the idea is really viable and I have made a lot of progress in its implementation.


![Firas Taji](https://c.mql5.com/avatar/2015/9/55F327B4-D45C.jpg)

**[Firas Taji](https://www.mql5.com/en/users/firastaji)**
\|
1 May 2024 at 12:46

Hi,

I ran the expert on both MT4 and MT5, when there are a PostFix the expert will not run and won't get any results.

can you find out why or fix it?

Many Thanks

![Yutaka Okamoto](https://c.mql5.com/avatar/2017/10/59EC2879-5228.jpg)

**[Yutaka Okamoto](https://www.mql5.com/en/users/kagen.jp)**
\|
9 May 2024 at 01:52

```
void FillPairsArray()// fill the array with required information about the instruments
   {
   int iterator=0;
   double correction;
   int TempSwapMode;

   for ( int i=0; i<ArraySize(Pairs); i++ )// reset symbols
      {
      Pairs[iterator].Name="";
      }

   for ( int i=0; i<SymbolsTotal(false); i++ )// check symbols from the MarketWatch window
      {
      TempSwapMode=int(SymbolInfoInteger(Pairs[iterator].Name,SYMBOL_SWAP_MODE));
      if ( StringLen(SymbolName(i,false)) == 6+PrefixE+PostfixE && IsValid(SymbolName(i,false)) && SymbolInfoInteger(SymbolName(i,false),SYMBOL_TRADE_MODE) == SYMBOL_TRADE_MODE_FULL
      && ( ( TempSwapMode  == 1 )  ||  ( ( TempSwapMode == 5 || TempSwapMode == 6 ) && CorrectedValue(Pairs[iterator].Name,correction) )) )
         {
         if ( iterator >= ArraySize(Pairs) ) break;
         Pairs[iterator].Name=SymbolName(i,false);
         Pairs[iterator].TickSize=SymbolInfoDouble(Pairs[iterator].Name,SYMBOL_TRADE_TICK_SIZE);
         Pairs[iterator].PointX=SymbolInfoDouble(Pairs[iterator].Name, SYMBOL_POINT);
         Pairs[iterator].ContractSize=SymbolInfoDouble(Pairs[iterator].Name,SYMBOL_TRADE_CONTRACT_SIZE);
         switch(TempSwapMode)
           {
            case  1:// in points
              Pairs[iterator].SwapBuy=SymbolInfoDouble(Pairs[iterator].Name,SYMBOL_SWAP_LONG)*Pairs[iterator].TickValue*(Pairs[iterator].PointX/Pairs[iterator].TickSize);
              Pairs[iterator].SwapSell=SymbolInfoDouble(Pairs[iterator].Name,SYMBOL_SWAP_SHORT)*Pairs[iterator].TickValue*(Pairs[iterator].PointX/Pairs[iterator].TickSize);
              break;
            case  5:// in percent
              Pairs[iterator].SwapBuy=correction*SymbolInfoDouble(Pairs[iterator].Name,SYMBOL_SWAP_LONG)*SymbolInfoDouble(Pairs[iterator].Name,SYMBOL_BID)*Pairs[iterator].ContractSize/(360.0*100.0);
              Pairs[iterator].SwapSell=correction*SymbolInfoDouble(Pairs[iterator].Name,SYMBOL_SWAP_SHORT)*SymbolInfoDouble(Pairs[iterator].Name,SYMBOL_BID)*Pairs[iterator].ContractSize/(360.0*100.0);
              break;
            case  6:// in percent
              Pairs[iterator].SwapBuy=correction*SymbolInfoDouble(Pairs[iterator].Name,SYMBOL_SWAP_LONG)*SymbolInfoDouble(Pairs[iterator].Name,SYMBOL_BID)*Pairs[iterator].ContractSize/(360.0*100.0);
              Pairs[iterator].SwapSell=correction*SymbolInfoDouble(Pairs[iterator].Name,SYMBOL_SWAP_SHORT)*SymbolInfoDouble(Pairs[iterator].Name,SYMBOL_BID)*Pairs[iterator].ContractSize/(360.0*100.0);
              break;
           }
         Pairs[iterator].Margin=SymbolInfoDouble(Pairs[iterator].Name,SYMBOL_MARGIN_INITIAL);
         Pairs[iterator].TickValue=SymbolInfoDouble(Pairs[iterator].Name,SYMBOL_TRADE_TICK_VALUE);         // <= this
         iterator++;
         }
      }
   }
```

Hello,

I have a question about the FillPairsArray() method.

In the FillPairsArray() method, isn't the place where the value of [SYMBOL\_TRADE\_TICK\_VALUE](https://www.mql5.com/en/docs/constants/environment_state/marketinfoconstants#enum_symbol_info_double "MQL5 documentation: Symbol properties") is set to Pairs\[iterator\].TickValue before SWAP is calculated?

It appears to be set after the SWAP calculation.

Thank you.

![Other classes in DoEasy library (Part 71): Chart object collection events](https://c.mql5.com/2/42/MQL5-avatar-doeasy-library__9.png)[Other classes in DoEasy library (Part 71): Chart object collection events](https://www.mql5.com/en/articles/9360)

In this article, I will create the functionality for tracking some chart object events — adding/removing symbol charts and chart subwindows, as well as adding/removing/changing indicators in chart windows.

![Other classes in DoEasy library (Part 70): Expanding functionality and auto updating the chart object collection](https://c.mql5.com/2/42/MQL5-avatar-doeasy-library__8.png)[Other classes in DoEasy library (Part 70): Expanding functionality and auto updating the chart object collection](https://www.mql5.com/en/articles/9293)

In this article, I will expand the functionality of chart objects and arrange navigation through charts, creation of screenshots, as well as saving and applying templates to charts. Also, I will implement auto update of the collection of chart objects, their windows and indicators within them.

![Tips from a professional programmer (Part II): Storing and exchanging parameters between an Expert Advisor, scripts and external programs](https://c.mql5.com/2/42/tipstricks__1.png)[Tips from a professional programmer (Part II): Storing and exchanging parameters between an Expert Advisor, scripts and external programs](https://www.mql5.com/en/articles/9327)

These are some tips from a professional programmer about methods, techniques and auxiliary tools which can make programming easier. We will discuss parameters which can be restored after terminal restart (shutdown). All examples are real working code segments from my Cayman project.

![Tips from a professional programmer (Part I): Code storing, debugging and compiling. Working with projects and logs](https://c.mql5.com/2/42/tipstricks.png)[Tips from a professional programmer (Part I): Code storing, debugging and compiling. Working with projects and logs](https://www.mql5.com/en/articles/9266)

These are some tips from a professional programmer about methods, techniques and auxiliary tools which can make programming easier.

[![](https://www.mql5.com/ff/si/0nfwvn6yhmgzf97n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F117%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dorder.expert%26utm_content%3Dorder.freelance%26utm_campaign%3D0622.MQL5.com.Internal&a=tunpwtbhegzufrqocbwiszessdutnobs&s=d9e7484e15300021b4066b1df77a94a1352f9e7c326d5113006bb4f6476bafeb&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=qtyxjyapwiwvzsdielqkbxjdjqpqmqtf&ssn=1769250799602489626&ssn_dr=0&ssn_sr=0&fv_date=1769250799&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F9198&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Swaps%20(Part%20I)%3A%20Locking%20and%20Synthetic%20Positions%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176925079939058323&fz_uniq=5082946993183003119&sv=2552)

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