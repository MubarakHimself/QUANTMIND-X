---
title: Optimization. A Few Simple Ideas
url: https://www.mql5.com/en/articles/1052
categories: Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T21:29:25.342376
---

[![](https://www.mql5.com/ff/sh/6zw0dkux8bqt7m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
MQL5 Channels - Messenger for traders\\
\\
Install the app and receive market analytics and trading tips.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=iuciwacmrxvmiibwyujliagqikizpsoo&s=268cbb13914c54b6c5c875db99b154944f6e0122b3400b54c9ac0d4f69f0f0d6&uid=&ref=https://www.mql5.com/en/articles/1052&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5071869933224144899)

MetaTrader 5 / Examples


### Introduction

Having found a consistent strategy for the EA to trade, we launch it on the EURUSD chart, right? Can the strategy be more profitable on other currency pairs? Are there any other currency pairs on which the strategy will give better results without necessity to increase the lot in geometric progression?

What will happen if we choose EURUSD and the standard H1 timeframe and then, if we are not happy with the result, change it for EURJPY on H4?

Besides, if we have a 64 bit operation system, which allows us to stop worrying about the speed of testing the system, do we forget about ludicrous combinations of the trading system entry parameters, which take part in the complete enumeration during optimization and which results we have to neglect in the final reports?

I have solved these "minor issues" myself and in this article share the effective solutions. I appreciate though that there may be other more optimal solutions.

### Optimization by Timeframes

MQL5 provides a complete set of timeframes: from M1, M2 , M3, M4,... H1, H2,... to monthly charts. In total, there are 21 timeframes. In the process of optimization, however, we want to know what timeframes suit our strategy most of all - short ones like М1 and М5, medium ones - like H2 and H4 or long ones - D1 and W1.

Initially we do not need this diversity of options. In any case, if we can see that the strategy proves effective on the М5 timeframe, then at the next stage of optimization we can check if it is going to work on М3 or М6.

If we use a variable of the ENUM\_TIMEFRAMES type as an input parameter:

```
input ENUM_TIMEFRAMES marcoTF= PERIOD_M5;
```

then the Optimizer will offer 21 optimization variations. Do we really need this amount?

![Standard options of a timeframe](https://c.mql5.com/2/10/SeleccTmpLargo-640-303.png)

Initially we do not. How can we simplify the optimization? At first we can define the enumeration:

```
enum mis_MarcosTMP
{
   _M1= PERIOD_M1,
   _M5= PERIOD_M5,
   _M15=PERIOD_M15,
//   _M20=PERIOD_M20,
   _M30=PERIOD_M30,
   _H1= PERIOD_H1,
   _H2= PERIOD_H2,
   _H4= PERIOD_H4,
//   _H8= PERIOD_H8,
   _D1= PERIOD_D1,
   _W1= PERIOD_W1,
   _MN1=PERIOD_MN1
};
```

where we can add or delete the timeframes of interest. For optimization, define the input variable in the beginning of the code:

```
input mis_MarcosTMP timeframe= _H1;
```

and define a new function in the library .mqh:

```
//----------------------------------------- DEFINE THE TIMEFRAME ----------------------------------------------------------
ENUM_TIMEFRAMES defMarcoTiempo(mi_MARCOTMP_CORTO marco)
{
   ENUM_TIMEFRAMES resp= _Period;
   switch(marco)
   {
      case _M1: resp= PERIOD_M1; break;
      case _M5: resp= PERIOD_M5; break;
      case _M15: resp= PERIOD_M15; break;
      //case _M20: resp= PERIOD_M20; break;
      case _M30: resp= PERIOD_M30; break;
      case _H1: resp= PERIOD_H1; break;
      case _H2: resp= PERIOD_H2; break;
      case _H4: resp= PERIOD_H4; break;
      //case _H8: resp= PERIOD_H8; break;
      case _D1: resp= PERIOD_D1; break;
      case _W1: resp= PERIOD_W1; break;
      case _MN1: resp= PERIOD_MN1;
   }
return(resp);
}
```

Declare a new variable in the scope of global variables:

```
ENUM_TIMEFRAMES marcoTmp= defMarcoTiempo(marcoTiempo);          //timeframe is defined as a global variable
```

"marcoTmp" is a global variable, which is going to be used by the EA to define a required chart timeframe. In the table of the Optimizer parameters, the interval of launching the "marcoTiempo" variable can be defined. This will cover only the steps of our interest without spending time and resources on analyzing М6 or М12. This way we can analyze results of the EA's work on different timeframes.

![User-specific options of a timeframe](https://c.mql5.com/2/10/SeleccTmpCorto-640-436.png)

Surely, it can be done with

```
ENUM_TIMEFRAMES marcoTmp= (ENUM_TIMEFRAMES)marcoTiempo;
```

This gets obvious after months, even years of programming if you are a perfectionist and go through the code a lot trying to simplify it. Or if you are using a VPS and trying to keep your bill down by optimizing computer performance.

### Optimizing a Symbol or a Set of Symbols

In the MetaTrader 5 Strategy Tester there is an optimization mode, which facilitates running the EA on all symbols selected in the MarketWatch window. This function, however, does not allow to arrange optimization as if the selected symbol is another parameter. So if there are 15 selected symbols, then the Tester will arrange 15 runs. How can we find out which symbol is the best for our EA? If this is a multi-currency EA, then what group of symbols give the best result and with what parameter set? String variables do not get optimized in MQL5. How can it be done?

Code a symbol or a couple of symbols by the input parameter value the following way:

```
input int selecDePar= 0;

string cadParesFX= selecPares(selecDePar);
```

The "selecDePar" parameter is used as an optimization parameter, which is converted into a string variable. Use the "cadParesFX" variable in the EA. The currency pair/pairs (for this code it is irrelevant whether the strategy is multi-currency or not) are going to be stored in this variable together with other regular optimization parameters.

```
//------------------------------------- SELECT THE SET OF PAIRS -------------------------------------
string selecPares(int combina= 0)
{
   string resp="EURUSD";
   switch(combina)
      {
         case 1: resp= "EURJPY"; break;
         case 2: resp= "USDJPY"; break;
         case 3: resp= "USDCHF"; break;
         case 4: resp= "GBPJPY"; break;
         case 5: resp= "GBPCHF"; break;
         case 6: resp= "GBPUSD"; break;
         case 7: resp= "USDCAD"; break;
         case 8: resp= "CADJPY"; break;
         case 9: resp= "XAUUSD"; break;

         case 10: resp= "EURJPY;USDJPY"; break;
         case 11: resp= "EURJPY;GBPJPY"; break;
         case 12: resp= "GBPCHF;GBPJPY"; break;
         case 13: resp= "EURJPY;GBPCHF"; break;
         case 14: resp= "USDJPY;GBPCHF"; break;

         case 15: resp= "EURUSD;EURJPY;GBPJPY"; break;
         case 16: resp= "EURUSD;EURJPY;GBPCHF"; break;
         case 17: resp= "EURUSD;EURJPY;USDJPY"; break;
         case 18: resp= "EURJPY;GBPCHF;USDJPY"; break;
         case 19: resp= "EURJPY;GBPUSD;GBPJPY"; break;
         case 20: resp= "EURJPY;GBPCHF;GBPJPY"; break;
         case 21: resp= "USDJPY;GBPCHF;GBPJPY"; break;
         case 22: resp= "EURUSD;USDJPY;GBPJPY"; break;

         case 23: resp= "EURUSD;EURJPY;USDJPY;GBPUSD;USDCHF;USDCAD"; break;
         case 24: resp= "EURUSD;EURJPY;USDJPY;GBPUSD;USDCHF;USDCAD;AUDUSD"; break;
      }
   return(resp);
}
```

Depending on what our goal is, define pair combinations and inform the Tester of the interval to be analyzed. Give the Strategy Tester a command to optimize the "selecDePar" parameter on the interval from 15 to 22 (see the picture below). What do we do when we want to compare results for a single currency? In that case we run optimization on the interval from 0 to 9.

![Optimization of a Set of Pairs](https://c.mql5.com/2/10/SeleccSimb-640-268.png)

For example, the EA receives the value of the cadParesFX= "EURUSD;EURJPY;GBPCHF" parameter. In OnInit() call the "cargaPares()" function, which fills the arrayPares\[\] dynamic array with strings, divided by the ";" symbol in the cadParesFX parameter. All global variables have to be loaded into dynamic arrays, which save the values of every symbol including the control of opening a new bar on a symbol if possible. In the case we are working with one symbol, the dimensions of array will be equal to one.

```
//-------------------------------- STRING CONVERSION FROM CURRENCY PAIRS INTO AN ARRAY  -----------------------------------------------
int cargaPares(string cadPares, string &arrayPares[])
{            //convierte "EURUSD;GBPUSD;USDJPY" a {"EURUSD", "GBPUSD", "USDJPY"}; devuelve el número de paresFX
   string caract= "";
   int i= 0, k= 0, contPares= 1, longCad= StringLen(cadPares);
   if(cadPares=="")
   {
      ArrayResize(arrayPares, contPares);
      arrayPares[0]= _Symbol;
   }
   else
   {
      for (k= 0; k<longCad; k++) if (StringSubstr(cadPares, k, 1)==";") contPares++;
      ArrayResize(arrayPares, contPares);
      ZeroMemory(arrayPares);
      for(k=0; k<longCad; k++)
      {
         caract= StringSubstr(cadPares, k, 1);
         if (caract!=";") arrayPares[i]= arrayPares[i]+caract;
         else i++;
      }
    }
   return(contPares);
}
```

In OnInit() this function is implemented the following way:

```
string ar_ParesFX[];    //array, containing names of the pairs for the EA to work with
int numSimbs= 1;        //variable, containing information about the number of symbols it works with

int OnInit()
{

   //...
   numSimbs= cargaPares(cadParesFX, ar_ParesFX);     //returns the ar_ParesFX array with pairs for work in the EA
   //...

}
```

If numSimbs>1, the OnChartEvent() function is called. This works with a multi-currency system. Otherwise, the OnTick() function is used:

```
void OnTick()
{
   string simb="";
   bool entrar= (nSimbs==1);
   if(entrar)
   {
      .../...
      simb= ar_ParesFX[0];
      gestionOrdenes(simb);
      .../...
   }
   return;
}

//+------------------------------------------------------------------+
//| EVENT HANDLER                                                   |
//+-----------------------------------------------------------------+
void OnChartEvent(const int idEvento, const long& lPeriodo, const double& dPrecio, const string &simbTick)
{
   bool entrar= nSimbs>1 && (idEvento>=CHARTEVENT_CUSTOM);
   if(entrar)
   {
      .../...
      gestionOrdenes(simbTick);
      .../...
   }
}

```

This means that all functions in the role of the input parameter must contain at least the symbol under inquiry. For instance, instead of the Digits() function, we must use the following:

```
//--------------------------------- SYMBOLS OF A SYMBOL ---------------------------------------
int digitosSimb(string simb= NULL)
{
   int numDig= (int)SymbolInfoInteger(simb, SYMBOL_DIGITS);
   return(numDig);
}
```

In other words, we must forget about the functions Symbol() or Point(), as well as other variables traditional for МetaТtarder 4 such as Ask or Bid.

```
//----------------------------------- POINT VALUE in price (Point())---------------------------------
double valorPunto(string simb= NULL)
{
   double resp= SymbolInfoDouble(simb, SYMBOL_POINT);
   return(resp);
}
```

```
//--------------------------- precio ASK-BID  -----------------------------------------
double precioAskBid(string simb= NULL, bool ask= true)
{
   ENUM_SYMBOL_INFO_DOUBLE precioSolic= ask? SYMBOL_ASK: SYMBOL_BID;
   double precio= SymbolInfoDouble(simb, precioSolic);
   return(precio);
}
```

We also forgot about the function of control over the bar opening, which is present is such codes. If the ticks received in EURUSD are telling about opening of a new bar,  then the USDJPY ticks might not be received in the next 2 sec. As a result, on the next USDJPY tick the EA is to discover that a new bar is opening for this symbol even if for EURUSD this event took place 2 sec ago.

```
//------------------------------------- NEW MULTI-CURRENCY CANDLESTICK -------------------------------------
bool nuevaVelaMD(string simb= NULL, int numSimbs= 1, ENUM_TIMEFRAMES marcoTmp= PERIOD_CURRENT)
{
        static datetime arrayHoraNV[];
        static bool primVez= true;
        datetime horaVela= iTime(simb, marcoTmp, 0);    //received opening time of the current candlestick
        bool esNueva= false;
        int codS= buscaCadArray(simb, nombreParesFX);
        if(primVez)
        {
           ArrayResize(arrayHoraNV, numSimbs);
           ArrayInitialize(arrayHoraNV, 0);
           primVez= false;
        }
        esNueva= codS>=0? arrayHoraNV[codS]!= horaVela: false;
        if(esNueva) arrayHoraNV[codS]= horaVela;
        return(esNueva);
}
```

This method allowed me to discover during one optimization pass that:

- the EA works well in EURUSD,
- works very badly in EURJPY,
- and works satisfactory in USDJPY
- On the EURUSD, GBPCHF, EURJPY pairs it works very well (real case).

This is true for the period of М5 and a certain combination of other optimization parameters, but not for Н1 or Н2.

There is an awkward moment here. I asked technical support about it. I don't know why this is happening but the optimization result differs depending on the symbol we select in the Strategy Tester. That is why for checking the result I keep this pair fixed through the strategy development and make sure that this is one of those pairs that can be analyzed in the Optimizer.

### Optimization of a Parameter Combination

Sometimes, some illogical combinations out of all parameter combinations that take part in the optimization turn out to be suitable. Some of them make the strategy unreasonable. For example, if the variable of the entry "maxSpread" defines the value of the spread set for a trading operation, we optimize this variable for various pairs where the average broker's spread is less than 30 and XAUUSD is 400. It is absurd to analyze those pairs if they exceed 50 and XAUUSD is less than 200. Having passed the data to the optimizer, set "evalua maxSpread between 0 and 600 with the interval 20", but such a set together with other parameters produces numerous combinations that do not make sense.

Following the pattern described in the previous section, we have defined pairs for optimization in the "selecPares()" function. EURUSD is assigned option 0 and XAUUSD is assigned option 9. Then define a global variable of the bool "paramCorrect" type.

```
bool paramCorrect= (selecDePar<9 && maxSpread<50) ||
                   (selecDePar==9 && maxSpread>200);
```

Carry out action in OnInit() only if paramCorrect is in the correct position true.

```
int OnInit()
{
   ENUM_INIT_RETCODE resp= paramCorrect? INIT_SUCCEEDED: INIT_PARAMETERS_INCORRECT;
   if (paramCorrect)
   {
      //...
      nSimbs= cargaPares(cadParesFX, nombreParesFX);     //return the array nombreParesFX containing pairs for work in the EA
      //... function of the EA initialization
   }
   return(resp);
}
```

If the paramCorrect is in the incorrect position false, then the EA does not perform any action in OnInit() and returns the INIT\_PARAMETERS\_INCORRECT to the Strategy Tester, which means an incorrect input data set. When the Strategy Tester receives the INIT\_PARAMETERS\_INCORRECT value from OnInit(), then this parameter set does not get passed to other testing agents for implementation and the line in the table with optimization results is filled with zeros and highlighted in red (see the picture below).

![Results of using incorrect parameters](https://c.mql5.com/2/10/resultCeroOpt-640-345.png)

The reason of the program shutdown is passed to OnDeinit() as an input variable and it helps to understand the reason of the EA closing. This is a different matter though.

```
void OnDeinit(const int motivo)
{
   if(paramCorrect)
   {

      //functions of the program shutdown

   }
   infoDeInit(motivo);
   return;
}

//+-------------------------------------- INFORMATION ABOUT THE PROGRAM SHUTDOWN----------------------------
string infoDeInit(int codDeInit)
{                       //informs of the reason of the program shutdown
   string texto= "program initialization...", text1= "CIERRE por: ";
   switch(codDeInit)
   {
      case REASON_PROGRAM:     texto= text1+"The EA finished its work with the ExpertRemove() function"; break;  //0
      case REASON_ACCOUNT:     texto= text1+"The account was changed"; break;                                    //6
      case REASON_CHARTCHANGE: texto= text1+"Symbol or timeframe change"; break;                                 //3
      case REASON_CHARTCLOSE:  texto= text1+"The chart was closed"; break;                                       //4
      case REASON_PARAMETERS:  texto= text1+"Input parameters changed by the user"; break;                       //5
      case REASON_RECOMPILE:   texto= text1+"The program was recompiled"; break;                                 //2
      case REASON_REMOVE:      texto= text1+"The program was deleted from the chart"; break;                     //1
      case REASON_TEMPLATE:    texto= text1+"Another chart template was used"; break;                            //7
      case REASON_CLOSE:       texto= text1+"The terminal was closed"; break;                                    //9
      case REASON_INITFAILED:  texto= text1+"The OnInit() handler returned non-zero value"; break;               //8
      default:                 texto= text1+"Other reason";
   }
   Print(texto);
   return(texto);
}
```

The thing is that if the parameter set received by the Optimizer at the specified stage sets "paramCorrect" to false (for example if the EURUSD spread was set to 100 points), then we do not run the EA and this optimization step becomes zero without unnecessary use of your computer resources and expenses of renting agents for your MQL5.сommunity account.

Surely, all said above can be implemented with OnTesterInit() and the ParameterGetRange() and ParameterSetRange() functions, but the described pattern seems to be simpler. This is guaranteed to work whereas the pattern with OnTesterInit() is not as consistent.

### Conclusion

We have discussed speeding up a search for optimal timeframes in МetaТrader 5, optimizing the "symbol" parameter, when МetaТrader 5 does not allow to optimize string variables and making it indifferent to the number of symbols the EA is using. We have also seen an illustration of how to reduce the number of optimization steps dropping illogical sets of input parameters maintaining your computer performance and saving the funds.

The above ideas are not new and they can be implemented by a novice or a programmer with some experience. These ideas were a result of a long search for information and usage of the debugging program. These are very simple but efficient ideas. You could ask me why I am sharing them if I want MQL5 to generate profit? The answer is to overcome the "solitude" of a programmer.

Thank you for your attention. If you read it up to the end and if you are an experienced programmer, please don't judge me too harshly.

Translated from Spanish by MetaQuotes Ltd.

Original article: [https://www.mql5.com/es/articles/1052](https://www.mql5.com/es/articles/1052)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Neural network: Self-optimizing Expert Advisor](https://www.mql5.com/en/articles/2279)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/41343)**
(13)


![Jose](https://c.mql5.com/avatar/2013/10/524FCB40-0FED.jpg)

**[Jose](https://www.mql5.com/en/users/jlwarrior)**
\|
27 Oct 2015 at 23:02

A good article. I find it curious that I have also faced this problem with solutions very similar to yours. Another interesting line is the "virtual" optimisation presented in this article:

https://www.mql5.com/en/articles/143

In any case, thanks for fighting against programmer's loneliness :)

![Jose Miguel Soriano](https://c.mql5.com/avatar/2014/4/5349356B-89D7.jpg)

**[Jose Miguel Soriano](https://www.mql5.com/en/users/josemiguel1812)**
\|
29 Oct 2015 at 19:52

**Jose:**

A good article. I find it curious that I have also faced this problem with solutions very similar to yours. Another interesting line is the "virtual" optimisation presented in this article:

https://www.mql5.com/en/articles/143

In any case, thanks for fighting against programmer's loneliness :)

I do structured programming and I don't "read" object-oriented programming well.

I don't understand the strategy itself because I don't see how it gets the virtual result of all strategies on candle 1 to choose what I do when opening candle 0. On candle 100 (numbering from present to past), for [example](https://www.mql5.com/en/docs/check/mqlinfostring "MQL5 Documentation: MQL5InfoString function "), if I can estimate the result on candle 98 "moving" into the future with the known history and the priceBID that the MT5 tester system will give me.... but on candle1 how do I estimate the virtual result?

![Jose Miguel Soriano](https://c.mql5.com/avatar/2014/4/5349356B-89D7.jpg)

**[Jose Miguel Soriano](https://www.mql5.com/en/users/josemiguel1812)**
\|
15 Jun 2016 at 09:43

**MetaQuotes Software Corp.:**

New article [Optimisation. Some simple ideas](https://www.mql5.com/en/articles/1052):

Author: [Jose Miguel Soriano](https://www.mql5.com/en/users/josemiguel1812 "josemiguel1812")

Please, in spanish or english.


![Carl Schreiber](https://c.mql5.com/avatar/2018/2/5A745EEE-EB76.PNG)

**[Carl Schreiber](https://www.mql5.com/en/users/gooly)**
\|
15 Jun 2016 at 11:03

**Jose Miguel Soriano:**

Please, in spanish or english.

1) Take the url: ["https://www.mql5.com/en/articles/1052"](https://www.mql5.com/en/articles/1052 "https://www.mql5.com/en/articles/1052") and replace . **./en/..** by **../en/..**

2) Top-Right of the article you find a button in the colours of the flag. Put the mous above it and choose the language you want.

![Michael Jimenez](https://c.mql5.com/avatar/2022/3/623E2C35-8A5C.jpg)

**[Michael Jimenez](https://www.mql5.com/en/users/mjimenez18)**
\|
10 Dec 2023 at 01:50

**MetaQuotes:**

Published article [Optimising optimisation: some simple ideas](https://www.mql5.com/en/articles/1052):

Author: [Jose Miguel Soriano Trujillo](https://www.mql5.com/en/users/josemiguel1812 "josemiguel1812")

Hi Jose,

Excellent publication!

I would like to get in touch with you regarding this publication which I am trying to replicate as I see it super useful and necessary hehe.

Greetings.

![Studying the CCanvas Class. How to Draw Transparent Objects](https://c.mql5.com/2/17/CCanvas_class_Standard_library_MetaTrader5.png)[Studying the CCanvas Class. How to Draw Transparent Objects](https://www.mql5.com/en/articles/1341)

Do you need more than awkward graphics of moving averages? Do you want to draw something more beautiful than a simple filled rectangle in your terminal? Attractive graphics can be drawn in the terminal. This can be implemented through the CСanvas class, which is used for creating custom graphics. With this class you can implement transparency, blend colors and produce the illusion of transparency by means of overlapping and blending colors.

![Bi-Directional Trading and Hedging of Positions in MetaTrader 5 Using the HedgeTerminal Panel, Part 1](https://c.mql5.com/2/17/HedgeTerminalaArticle200x200_2.png)[Bi-Directional Trading and Hedging of Positions in MetaTrader 5 Using the HedgeTerminal Panel, Part 1](https://www.mql5.com/en/articles/1297)

This article describes a new approach to hedging of positions and draws the line in the debates between users of MetaTrader 4 and MetaTrader 5 about this matter. The algorithms making such hedging reliable are described in layman's terms and illustrated with simple charts and diagrams. This article is dedicated to the new panel HedgeTerminal, which is essentially a fully featured trading terminal within MetaTrader 5. Using HedgeTerminal and the virtualization of the trade it offers, positions can be managed in the way similar to MetaTrader 4.

![Bi-Directional Trading and Hedging of Positions in MetaTrader 5 Using the HedgeTerminal API, Part 2](https://c.mql5.com/2/17/HedgeTerminalaArticle200x200_2p2.png)[Bi-Directional Trading and Hedging of Positions in MetaTrader 5 Using the HedgeTerminal API, Part 2](https://www.mql5.com/en/articles/1316)

This article describes a new approach to hedging of positions and draws the line in the debates between users of MetaTrader 4 and MetaTrader 5 about this matter. It is a continuation of the first part: "Bi-Directional Trading and Hedging of Positions in MetaTrader 5 Using the HedgeTerminal Panel, Part 1". In the second part, we discuss integration of custom Expert Advisors with HedgeTerminalAPI, which is a special visualization library designed for bi-directional trading in a comfortable software environment providing tools for convenient position management.

![Principles of Exchange Pricing through the Example of Moscow Exchange's Derivatives Market](https://c.mql5.com/2/12/MOEX.png)[Principles of Exchange Pricing through the Example of Moscow Exchange's Derivatives Market](https://www.mql5.com/en/articles/1284)

This article describes the theory of exchange pricing and clearing specifics of Moscow Exchange's Derivatives Market. This is a comprehensive article for beginners who want to get their first exchange experience on derivatives trading, as well as for experienced forex traders who are considering trading on a centralized exchange platform.

[![](https://www.mql5.com/ff/sh/wm94j0jmkwd29943z2/ddfa713cb3cdd580c3e81e0e13b5b1b8.jpg)\\
Revised MetaTrader 5 Web Terminal\\
\\
Trade with no restrictions from any mobile device, OS and web browser\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=fkjlpstbxdmrrwpblfatcsdjyrxbizyj&s=f462f051eb7aaec36d6b31792d312d60d3f5a50c83b12d0d66e85d5d61bd941b&uid=&ref=https://www.mql5.com/en/articles/1052&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5071869933224144899)

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