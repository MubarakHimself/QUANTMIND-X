---
title: Developing a cross-platform grider EA
url: https://www.mql5.com/en/articles/5596
categories: Trading Systems, Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T19:42:33.671046
---

[What's wrong with regular VPS?Here are the 8 most common problems that algorithmic traders may encounterRead![](https://www.mql5.com/ff/sh/hzatb686qjqxwtr4z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/450486&a=drhremihlwuaqyvgpzfddbtmgciejpba&s=c37d25bcceb93ed153b814e6ba4d4839461a9b2d68dd82b95b142be06d310f3f&uid=&ref=https://www.mql5.com/en/articles/5596&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5070499546598938359)

MetaTrader 5 / Trading systems


### Introduction

Most frequent users of this website know pretty well that MQL5 is the best option for developing custom EAs. Unfortunately, not all brokers allow creating accounts available in MetaTrader 5. Even if you currently work with a broker that allows that, you may switch to a broker offering only MetaTrader 4 in the future. What are you going to do with all the MQL5 EAs you have developed in that case? Are you going to spend a huge amount of time to rework them to fit into MQL4? Perhaps, it would be more reasonable to develop an EA able to work both in MetaTrader 5 and MetaTrader 4?

In this article, we will try to develop such an EA and check if a trading system based on an order grid is usable.

### A few words about conditional compilation

[Conditional compilation](https://www.mql5.com/en/docs/basis/preprosessor/conditional_compilation) will allow us to develop an EA working both in MetaTrader 4 and MetaTrader 5. The applied syntax is as follows:

```
   #ifdef __MQL5__
      // MQL5 code
   #else
      // MQL4 code
   #endif
```

Conditional compilation allows us to specify that a certain block should be compiled only in case compilation is done in an MQL5 EA. When compiling in MQL4 and other language versions, this code block is simply discarded. The code block following the _#else_ operator is used instead (if set).

Thus, if some functionality is implemented differently in MQL4 and MQL5, we are to implement it in both ways, while conditional compilation allows selecting the option that is necessary for a certain language.

In other cases, we will use the syntax working both in MQL4 and MQL5.

### Grid trading systems

Before starting the EA development, let's describe the basics of grid trading strategies.

Griders are EAs that place several limit orders above the current price and the same number of limit orders below it simultaneously.

Limit orders are set with a certain step, rather than at a single price. In other words, the first limit order is set at a certain distance above the current price. The second limit order is set above the first one at the same distance. And so forth. The number of orders and the applied step vary.

Orders in one direction are placed above the current price, while orders in another direction are placed below the current price. It is considered that:

- during a trend, Long orders should be placed above the current price, while Short orders should be placed below it;
- during a flat, Short orders should be placed above the current price, while Long orders should be placed below it.


You can either apply stop levels, or work without them.

If you do not use stop loss and take profit, all open positions, both profitable and loss-making ones, exist till the overall profit reaches a certain level. After that, all open positions, as well as limit orders not affected by the price, are closed, and a new grid is set.

The screenshot below shows an open grid:

![Grid sample](https://c.mql5.com/2/35/grid__4.png)

Thus, in theory, grid trading systems allow you to make a profit in any market without waiting for any unique entry points, as well as without using any indicators.

If stop loss and take profit are used, then the profit is obtained due to the fact that the loss on one position is covered by the overall profit on the rest if the price moves in one direction.

Without stop levels, the profit is obtained due to opening a greater number of orders in the right direction. Even if at first the price touches the positions in one direction, and then turns around, new positions in the right direction will cover the loss on the previously opened ones, as there will be more of them in the end.

### Our grider EA's working principles

We have described the working principle of the simplest grider above. You can come up with your own options for grids changing the direction of opening orders, adding the ability to open multiple orders at the same price, adding indicators, etc.

In this article, we will try to implement the simplest grider version without stop losses, since the idea it is based on is very tempting.

Indeed, the idea that the price sooner or later reaches the profit when moving in one direction even if positions were initially opened in the wrong direction seems reasonable. Suppose that at the very beginning the price experienced correction and touched two orders. After that, the price began to move in the opposite (main trend) direction. In this case, sooner or later more than two orders will be opened in the right direction, and our initial loss will turn into a profit after some time. Why wouldn't this trading system work?

It seems that the only case, in which the trading system can cause a loss is when the price first touches one order, then goes back and touches the opposite, then again changes direction and touches another order, and changes its direction over and over again touching more and more distant orders. But is such price behavior is possible in real conditions at all?

### EA termplate

We will start developing the EA from the template. This will allow us to immediately see, which standard MQL functions are to be involved.

```
#property copyright "Klymenko Roman (needtome@icloud.com)"
#property link      "https://www.mql5.com/en/users/needtome"
#property version   "1.00"
#property strict

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---

//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
  }

void OnChartEvent(const int id,         // event ID
                  const long& lparam,   // event parameter of the long type
                  const double& dparam, // event parameter of the double type
                  const string& sparam) // event parameter of the string type
   {
   }
```

Its only difference from the standard template generated when creating the EA using the MQL5 Wizard is _#property strict_ string. We add it so that the EA works in MQL4 as well.

We need the [_OnChartEvent()_](https://www.mql5.com/en/docs/event_handlers/onchartevent) function to be able to respond to clicking the buttons. Next, we will implement the _Close All_ button to be able to manually close all symbol positions and orders if we reached the desired equity or simply want to stop the EA.

### Position opening function

Probably, the most important functionality of any EA is the ability to place an order. Here is where the first issues await us. In MQL5 and MQL4, orders are placed quite differently. In order to somehow unify this functionality, we will have to develop a custom function for placing orders.

Each order has its own type: buy order, sell order, limit buy or sell order. The variable, in which this type is set when placing an order, is also different in MQL5 and MQL4.

In MQL4, an order type is specified by an int type variable, while in MQL5, the [ENUM\_ORDER\_TYPE](https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties#enum_order_type) enumeration is used. There is no such enumeration in MQL4. Therefore, in order to combine both methods, we should create a custom enumeration for setting an order type. Due to this, the function we are to create in the future will not depend on the MQL version:

```
enum TypeOfPos{
   MY_BUY,
   MY_SELL,
   MY_BUYSTOP,
   MY_BUYLIMIT,
   MY_SELLSTOP,
   MY_SELLLIMIT,
};
```

Now we can create a custom function for placing an order. Let's name it _pdxSendOrder()_. We will pass to it all that is needed for placing an order: order type, open price, stop loss (0 if not set), take profit (0 if not set), volume, open position ticket (if an open position should be modified in MQL5), comment and symbol (if you need to open an order for a symbol other than the currently opened one):

```
// order sending function
bool pdxSendOrder(TypeOfPos mytype, double price, double sl, double tp, double volume, ulong position=0, string comment="", string sym=""){
   // check passed values
   if( !StringLen(sym) ){
      sym=_Symbol;
   }
   int curDigits=(int) SymbolInfoInteger(sym, SYMBOL_DIGITS);
   if(sl>0){
      sl=NormalizeDouble(sl,curDigits);
   }
   if(tp>0){
      tp=NormalizeDouble(tp,curDigits);
   }
   if(price>0){
      price=NormalizeDouble(price,curDigits);
   }

   #ifdef __MQL5__
      ENUM_TRADE_REQUEST_ACTIONS action=TRADE_ACTION_DEAL;
      ENUM_ORDER_TYPE type=ORDER_TYPE_BUY;
      switch(mytype){
         case MY_BUY:
            action=TRADE_ACTION_DEAL;
            type=ORDER_TYPE_BUY;
            break;
         case MY_BUYSTOP:
            action=TRADE_ACTION_PENDING;
            type=ORDER_TYPE_BUY_STOP;
            break;
         case MY_BUYLIMIT:
            action=TRADE_ACTION_PENDING;
            type=ORDER_TYPE_BUY_LIMIT;
            break;
         case MY_SELL:
            action=TRADE_ACTION_DEAL;
            type=ORDER_TYPE_SELL;
            break;
         case MY_SELLSTOP:
            action=TRADE_ACTION_PENDING;
            type=ORDER_TYPE_SELL_STOP;
            break;
         case MY_SELLLIMIT:
            action=TRADE_ACTION_PENDING;
            type=ORDER_TYPE_SELL_LIMIT;
            break;
      }

      MqlTradeRequest mrequest;
      MqlTradeResult mresult;
      ZeroMemory(mrequest);

      mrequest.action = action;
      mrequest.sl = sl;
      mrequest.tp = tp;
      mrequest.symbol = sym;
      if(position>0){
         mrequest.position = position;
      }
      if(StringLen(comment)){
         mrequest.comment=comment;
      }
      if(action!=TRADE_ACTION_SLTP){
         if(price>0){
            mrequest.price = price;
         }
         if(volume>0){
            mrequest.volume = volume;
         }
         mrequest.type = type;
         mrequest.magic = EA_Magic;
         switch(useORDER_FILLING_RETURN){
            case FOK:
               mrequest.type_filling = ORDER_FILLING_FOK;
               break;
            case RETURN:
               mrequest.type_filling = ORDER_FILLING_RETURN;
               break;
            case IOC:
               mrequest.type_filling = ORDER_FILLING_IOC;
               break;
         }
         mrequest.deviation=100;
      }
      if(OrderSend(mrequest,mresult)){
         if(mresult.retcode==10009 || mresult.retcode==10008){
            return true;
         }else{
            msgErr(GetLastError(), mresult.retcode);
         }
      }
   #else
      int type=OP_BUY;
      switch(mytype){
         case MY_BUY:
            type=OP_BUY;
            break;
         case MY_BUYSTOP:
            type=OP_BUYSTOP;
            break;
         case MY_BUYLIMIT:
            type=OP_BUYLIMIT;
            break;
         case MY_SELL:
            type=OP_SELL;
            break;
         case MY_SELLSTOP:
            type=OP_SELLSTOP;
            break;
         case MY_SELLLIMIT:
            type=OP_SELLLIMIT;
            break;
      }
      if(OrderSend(sym, type, volume, price, 100, sl, tp, comment, EA_Magic, 0)<0){
         msgErr(GetLastError());
      }else{
         return true;
      }

   #endif
   return false;
}
```

First, check the values passed to the function and normalize the prices.

**Inputs**. After that, use conditional compilation to define the current MQL version and set an order according to its rules. The additional _useORDER\_FILLING\_RETURN_ input parameter is used for MQL5. With its help, we configure the order execution mode in accordance with modes supported by the broker. Since the _useORDER\_FILLING\_RETURN_ input parameter is necessary only for the MQL5 EA, use conditional compilation again to add it:

```
#ifdef __MQL5__
   enum TypeOfFilling //Filling Mode
     {
      FOK,//ORDER_FILLING_FOK
      RETURN,// ORDER_FILLING_RETURN
      IOC,//ORDER_FILLING_IOC
     };
   input TypeOfFilling  useORDER_FILLING_RETURN=FOK; //Filling Mode
#endif
```

Also, when placing an order, the _EA\_Magic_ incoming parameter containing the EA's magic number is used.

If this parameter is not set in the EA settings, any positions on a symbol the EA has been launched at are considered owned by the EA. Thus, the EA takes full control over them.

If the magic number is set, the EA considers only positions having this magic number in its work.

**Displaying errors**. If an order is set successfully, _true_ is returned. Otherwise, the appropriate error code is passed to the _msgErr_() function for further analysis and displaying a comprehensible error message. The function displays a localized message containing a detailed error description. There is no point in providing its full code here. So I will show only a part of it:

```
void msgErr(int err, int retcode=0){
   string curErr="";
   switch(err){
      case 1:
         curErr=langs.err1;
         break;
//      case N:
//         curErr=langs.errN;
//         break;
      default:
         curErr=langs.err0+": "+(string) err;
   }
   if(retcode>0){
      curErr+=" ";
      switch(retcode){
         case 10004:
            curErr+=langs.retcode10004;
            break;
//         case N:
//            curErr+=langs.retcodeN;
//            break;
      }
   }

   Alert(curErr);
}
```

We will dwell more on localization in the next section.

### EA localization

Before resuming the EA development, let's make it bilingual. Let's add the ability to choose the language of EA messages. We will provide two languages: English and Russian.

Create the enumeration with possible language options and add a suitable parameter for selecting a language:

```
enum TypeOfLang{
   MY_ENG, // English
   MY_RUS, // Русский
};

input TypeOfLang  LANG=MY_RUS; // Language
```

Next, create a structure that will be used to store all text strings used in the EA. After that, declare the variable of the type we created:

```
struct translate{
   string err1;
   string err2;
//   ... other strings
};
translate langs;
```

We already have the variable containing the strings. Although, there are no strings there yet. Create the function that fills it with strings in the language selected in the _Language_ input. Let's name the function _init\_lang()_. Part of its code is displayed below:

```
void init_lang(){
   switch(LANG){
      case MY_ENG:
         langs.err1="No error, but unknown result. (1)";
         langs.err2="General error (2)";
         langs.err3="Incorrect parameters (3)";
//         ... other strings
         break;
      case MY_RUS:
         langs.err0="Во время выполнения запроса произошла ошибка";
         langs.err1="Нет ошибки, но результат неизвестен (1)";
         langs.err2="Общая ошибка (2)";
         langs.err3="Неправильные параметры (3)";
//         ... other strings
         break;
   }
}
```

The only thing left to do is call the _init\_lang ()_ function so that the strings are filled with the necessary values. The perfect place for calling it is a standard [_OnInit()_](https://www.mql5.com/en/docs/event_handlers/oninit) function since it is called during the EA launch, which is exactly what we need.

### Main inputs

It is time to add the main inputs to our EA. Apart from the already described _EA\_Magic_ and _LANG_, these are:

```
input double      Lot=0.01;     //Lot size
input uint        maxLimits=7;  //Number of limit orders in the grid in one direction
input int         Step=10;      //Grid step in points
input double      takeProfit=1; //Close deals when reaching the specified profit, $
```

In other words, we will open _maxLimits_ orders in one direction and the same number of orders in the opposite one. The first order is located at _Step_ points from the current price. While the second one is placed at _Step_ points from the first order and so forth.

A profit is fixed as soon as it reaches _takeProfit_ value (in $). In this case, all open positions are closed, as well as all placed orders are canceled. After that, the EA resets its grid.

We do not consider the possibility of losing at all, so take profit is the only condition for closing positions.

### Filling in the OnInit function

As already mentioned, the _OnInit()_ function is called once during the first EA launch. We have already added the _init\_lang()_ function call to it. Let's fill it up to the end, so as not to return to it anymore.

Within the frame of our EA, the only objective of the _OnInit()_ function is correction of the _Step_ input if the price has 3 or 5 digital places. In other words, if a single additional digital place is used by the broker for the symbol:

```
   ST=Step;
   if(_Digits==5 || _Digits==3){
      ST*=10;
   }
```

Thus, we are going to use the corrected ST parameter instead of the _Step_ input in the EA itself. Declare it before calling any functions by specifying the double type.

Since we will need the distance in the symbol price rather than in points to form a grid, let's perform the conversion right away:

```
   ST*=SymbolInfoDouble(_Symbol, SYMBOL_POINT);
```

Also in this function, we can check whether trading is allowed for our EA. If trading is disabled, it is better to immediately inform users about it so that they can improve that.

The check can be done using this small code:

```
   if(!MQLInfoInteger(MQL_TRADE_ALLOWED)){
      Alert(langs.noBuy+" ("+(string) EA_Magic+")");
      ExpertRemove();
   }
```

If trading is disabled, we inform of that in the language chosen by a user. After that, the EA operation is complete.

As a result, the final look of the _OnInit()_ function is as follows:

```
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   init_lang();

   if(!MQLInfoInteger(MQL_TRADE_ALLOWED)){
      Alert(langs.noBuy+" ("+(string) EA_Magic+")");
      ExpertRemove();
   }

   ST=Step;
   if(_Digits==5 || _Digits==3){
      ST*=10;
   }
   ST*=SymbolInfoDouble(_Symbol, SYMBOL_POINT);

   return(INIT_SUCCEEDED);
  }
```

### Adding the Close All button

Convenience of working with an EA is as important as its adherence to a selected trading strategy.

In our case, convenience is expressed in the ability to see at a glance how many positions in Long and Short are already open, and also find out the total profit for all currently open positions.

We should also be able to quickly close all open orders and positions if we are satisfied with profit or something goes wrong.

Therefore, let's add the button displaying all the necessary data and closing all positions and orders when clicked.

**Graphical object prefix**. Every graphical object in MetaTrader should have a name. The names of objects created by one EA should not coincide with the names of objects created on a chart manually or by other EAs. Therefore, first of all, let's define the prefix to be added to the names of all graphical objects:

```
string prefix_graph="grider_";
```

**Calculate positions and profit**. Now we can create a function that will calculate the number of open Long and Short positions, as well as their total profit, and display the button with obtained data or update the text on it if such a button already exists. Let's name the function _getmeinfo\_btn()_:

```
void getmeinfo_btn(string symname){
   double posPlus=0;
   double posMinus=0;
   double profit=0;
   double positionExist=false;

   // count the number of open Long and Short positions,
   // and total profit on them
   #ifdef __MQL5__
      int cntMyPos=PositionsTotal();
      for(int ti=cntMyPos-1; ti>=0; ti--){
         if(PositionGetSymbol(ti)!=symname) continue;
         if(EA_Magic>0 && PositionGetInteger(POSITION_MAGIC)!=EA_Magic) continue;

         positionExist=true;

         profit+=PositionGetDouble(POSITION_PROFIT);
         profit+=PositionGetDouble(POSITION_SWAP);

         if(PositionGetInteger(POSITION_TYPE)==POSITION_TYPE_BUY){
            posPlus+=PositionGetDouble(POSITION_VOLUME);
         }else{
            posMinus+=PositionGetDouble(POSITION_VOLUME);
         }
      }
   #else
      int cntMyPos=OrdersTotal();
      if(cntMyPos>0){
         for(int ti=cntMyPos-1; ti>=0; ti--){
            if(OrderSelect(ti,SELECT_BY_POS,MODE_TRADES)==false) continue;
            if( OrderType()==OP_BUY || OrderType()==OP_SELL ){}else{ continue; }
            if(OrderSymbol()!=symname) continue;
            if(EA_Magic>0 && OrderMagicNumber()!=EA_Magic) continue;

            positionExist=true;

            profit+=OrderCommission();
            profit+=OrderProfit();
            profit+=OrderSwap();

            if(OrderType()==OP_BUY){
               posPlus+=OrderLots();
            }else{
               posMinus+=OrderLots();
            }
         }
      }
   #endif

   // if there are open positions,
   // add the close button
   if(positionExist){
      createObject(prefix_graph+"delall", 233, langs.closeAll+" ("+DoubleToString(profit, 2)+") L: "+(string) posPlus+" S: "+(string) posMinus);
   }else{
      // otherwise, delete the button for closing positions
      if(ObjectFind(0, prefix_graph+"delall")>0){
         ObjectDelete(0, prefix_graph+"delall");
      }
   }

   // update the current chart to display
   // the implemented changes
   ChartRedraw(0);
}
```

Here we used conditional compilation for the second time since the functionality of working with open positions in MQL5 is different from that of MQL4. For the same reason, we will use conditional compilation more than once later in the article.

**Displaying the button**. Also note that in order to display the button on a chart, we use the _createObject()_ custom function. The function checks if the button with the name passed as the first function argument is present on the chart.

If the button has already been created, simply update the text on it according to the text passed in the third function argument.

If there is no button, create it in the upper right corner of the chart. In this case, the second function argument sets the button width:

```
void createObject(string name, int weight, string title){
   // if there is no 'name' button on the chart, create it
   if(ObjectFind(0, name)<0){
      // define the shift relative to the chart right angle where the button is to be displayed
      long offset= ChartGetInteger(0, CHART_WIDTH_IN_PIXELS)-87;
      long offsetY=0;
      for(int ti=0; ti<ObjectsTotal((long) 0); ti++){
         string objName= ObjectName(0, ti);
         if( StringFind(objName, prefix_graph)<0 ){
            continue;
         }
         long tmpOffset=ObjectGetInteger(0, objName, OBJPROP_YDISTANCE);
         if( tmpOffset>offsetY){
            offsetY=tmpOffset;
         }
      }

      for(int ti=0; ti<ObjectsTotal((long) 0); ti++){
         string objName= ObjectName(0, ti);
         if( StringFind(objName, prefix_graph)<0 ){
            continue;
         }
         long tmpOffset=ObjectGetInteger(0, objName, OBJPROP_YDISTANCE);
         if( tmpOffset!=offsetY ){
            continue;
         }

         tmpOffset=ObjectGetInteger(0, objName, OBJPROP_XDISTANCE);
         if( tmpOffset>0 && tmpOffset<offset){
            offset=tmpOffset;
         }
      }
      offset-=(weight+1);
      if(offset<0){
         offset=ChartGetInteger(0, CHART_WIDTH_IN_PIXELS)-87;
         offsetY+=25;
         offset-=(weight+1);
      }

     ObjectCreate(0, name, OBJ_BUTTON, 0, 0, 0);
     ObjectSetInteger(0,name,OBJPROP_XDISTANCE,offset);
     ObjectSetInteger(0,name,OBJPROP_YDISTANCE,offsetY);
     ObjectSetString(0,name,OBJPROP_TEXT, title);
     ObjectSetInteger(0,name,OBJPROP_XSIZE,weight);
     ObjectSetInteger(0,name,OBJPROP_FONTSIZE, 8);
     ObjectSetInteger(0,name,OBJPROP_COLOR, clrBlack);
     ObjectSetInteger(0,name,OBJPROP_YSIZE,25);
     ObjectSetInteger(0,name,OBJPROP_BGCOLOR, clrLightGray);
     ChartRedraw(0);
  }else{
     ObjectSetString(0,name,OBJPROP_TEXT, title);
  }
}
```

**Response to clicking the button**. Now if we call the _getmeinfo\_btn()_ function, the _Close All_ button appears on the chart (if we have open positions). However, nothing happens yet when clicking this button.

To add a response to clicking the button, we need to intercept clicking in the _OnChartEvent()_ standard function. Since this is the only objective of the _OnChartEvent()_ function, we can provide its final code:

```
void OnChartEvent(const int id,         // event ID
                  const long& lparam,   // event parameter of the long type
                  const double& dparam, // event parameter of the double type
                  const string& sparam) // event parameter of the string type
{
   string text="";
   switch(id){
      case CHARTEVENT_OBJECT_CLICK:
         // if the name of the clicked button is prefix_graph+"delall", then
         if (sparam==prefix_graph+"delall"){
            closeAllPos();
         }
         break;
   }
}
```

Now when clicking the position closing button, the _closeAllPos()_ function is called. This function is not implemented yet. We will do that in the next section.

**Additional actions**. We already have the _getmeinfo\_btn()_ function calculating the necessary data and displaying the position closing button. Moreover, we have implemented the action occurring when clicking the button. However, the _getmeinfo\_btn()_ function itself is not called anywhere in the EA yet. Therefore, it is not displayed on the chart for now.

We will use the _getmeinfo\_btn()_ function when dealing the code of the standard _OnTick()_ function.

In the meantime, let's switch our attention to the _OnDeInit()_ standard function. Since our EA creates a graphical object, ensure that all graphical objects created by it are removed from the chart when closing the EA. This is why we need the _OnDeInit()_ function. It is called automatically when closing an EA.

As a result, the _OnDeInit()_ function body looks as follows:

```
void OnDeinit(const int reason)
  {
      ObjectsDeleteAll(0, prefix_graph);
  }
```

This string removes all graphical objects containing the specified prefix in their names when closing the EA. We have only one such object so far.

### Implementing the function for closing all positions

Since we have already started using the _closeAllPos ()_ function, let's implement its code.

The _closeAllPos()_ function closes all currently open positions and removes all placed orders.

But it is not so simple. The function does not just delete all currently open positions. If we have an open Long position and the same Short one, we will try to close one of these positions by an opposite one. If your broker supports this operation on the current instrument, we get back the spread we paid for opening two positions. This improves the profitability of our EA. When closing all positions by take profit, we will actually have a profit slightly exceeding the one specified in the _takeProfit_ input parameter.

Thus, the first string of the _closeAllPos()_ function contains calling yet another function: _closeByPos()_.

The _closeByPos()_ function attempts to close positions by opposite ones. After all opposite positions are closed, the _closeAllPos()_ function closes the remaining positions in the usual way. After that, it closes placed orders.

I usually use the [CTrade](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade) object to close positions in MQL5. Therefore, before implementing the two custom functions, let's include the class and create its object right away:

```
#ifdef __MQL5__
   #include <Trade\Trade.mqh>
   CTrade Trade;
#endif
```

Now we can start developing the function closing all positions by opposite ones:

```
void closeByPos(){
   bool repeatOpen=false;
   #ifdef __MQL5__
      int cntMyPos=PositionsTotal();
      for(int ti=cntMyPos-1; ti>=0; ti--){
         if(PositionGetSymbol(ti)!=_Symbol) continue;
         if(EA_Magic>0 && PositionGetInteger(POSITION_MAGIC)!=EA_Magic) continue;

         if( PositionGetInteger(POSITION_TYPE)==POSITION_TYPE_BUY ){
            long closefirst=PositionGetInteger(POSITION_TICKET);
            double closeLots=PositionGetDouble(POSITION_VOLUME);

            for(int ti2=cntMyPos-1; ti2>=0; ti2--){
               if(PositionGetSymbol(ti2)!=_Symbol) continue;
               if(EA_Magic>0 && PositionGetInteger(POSITION_MAGIC)!=EA_Magic) continue;
               if( PositionGetInteger(POSITION_TYPE)!=POSITION_TYPE_SELL ) continue;
               if( PositionGetDouble(POSITION_VOLUME)!=closeLots ) continue;

               MqlTradeRequest request;
               MqlTradeResult  result;
               ZeroMemory(request);
               ZeroMemory(result);
               request.action=TRADE_ACTION_CLOSE_BY;
               request.position=closefirst;
               request.position_by=PositionGetInteger(POSITION_TICKET);
               if(EA_Magic>0) request.magic=EA_Magic;
               if(OrderSend(request,result)){
                  repeatOpen=true;
                  break;
               }
            }
            if(repeatOpen){
               break;
            }
         }
      }
   #else
      int cntMyPos=OrdersTotal();
      if(cntMyPos>0){
         for(int ti=cntMyPos-1; ti>=0; ti--){
            if(OrderSelect(ti,SELECT_BY_POS,MODE_TRADES)==false) continue;
            if( OrderSymbol()!=_Symbol ) continue;
            if(EA_Magic>0 && OrderMagicNumber()!=EA_Magic) continue;

            if( OrderType()==OP_BUY ){
               int closefirst=OrderTicket();
               double closeLots=OrderLots();

               for(int ti2=cntMyPos-1; ti2>=0; ti2--){
                  if(OrderSelect(ti2,SELECT_BY_POS,MODE_TRADES)==false) continue;
                  if( OrderSymbol()!=_Symbol ) continue;
                  if(EA_Magic>0 && OrderMagicNumber()!=EA_Magic) continue;
                  if( OrderType()!=OP_SELL ) continue;
                  if( OrderLots()<closeLots ) continue;

                  if( OrderCloseBy(closefirst, OrderTicket()) ){
                     repeatOpen=true;
                     break;
                  }
               }
               if(repeatOpen){
                  break;
               }
            }

         }
      }
   #endif
   // if we closed a position by an opposite one,
   // launch the closeByPos function again
   if(repeatOpen){
      closeByPos();
   }
}
```

The function calls itself if a close by operation was successful. This is necessary since positions may have different volumes, which means closing two positions may not always yield the necessary results. If the volumes are different, one of the positions' volumes simply decreases making it available for being closed by an opposite position during the next function launch.

After closing all opposite positions, the _closeAllPos()_ function closes the remaining ones:

```
void closeAllPos(){
   closeByPos();
   #ifdef __MQL5__
      int cntMyPos=PositionsTotal();
      for(int ti=cntMyPos-1; ti>=0; ti--){
         if(PositionGetSymbol(ti)!=_Symbol) continue;
         if(EA_Magic>0 && PositionGetInteger(POSITION_MAGIC)!=EA_Magic) continue;

         Trade.PositionClose(PositionGetInteger(POSITION_TICKET));
      }
      int cntMyPosO=OrdersTotal();
      for(int ti=cntMyPosO-1; ti>=0; ti--){
         ulong orderTicket=OrderGetTicket(ti);
         if(OrderGetString(ORDER_SYMBOL)!=_Symbol) continue;
         if(EA_Magic>0 && OrderGetInteger(ORDER_MAGIC)!=EA_Magic) continue;

         Trade.OrderDelete(orderTicket);
      }
   #else
      int cntMyPos=OrdersTotal();
      if(cntMyPos>0){
         for(int ti=cntMyPos-1; ti>=0; ti--){
            if(OrderSelect(ti,SELECT_BY_POS,MODE_TRADES)==false) continue;
            if( OrderSymbol()!=_Symbol ) continue;
            if(EA_Magic>0 && OrderMagicNumber()!=EA_Magic) continue;

            if( OrderType()==OP_BUY ){
               MqlTick latest_price;
               if(!SymbolInfoTick(OrderSymbol(),latest_price)){
                  Alert(GetLastError());
                  return;
               }
               if(!OrderClose(OrderTicket(), OrderLots(),latest_price.bid,100)){
               }
            }else if(OrderType()==OP_SELL){
               MqlTick latest_price;
               if(!SymbolInfoTick(OrderSymbol(),latest_price)){
                  Alert(GetLastError());
                  return;
               }
               if(!OrderClose(OrderTicket(), OrderLots(),latest_price.ask,100)){
               }
            }else{
               if(!OrderDelete(OrderTicket())){
               }
            }

         }
      }
   #endif
   // delete the position closing button
   if(ObjectFind(0, prefix_graph+"delall")>0){
      ObjectDelete(0, prefix_graph+"delall");
   }

}
```

### Implementing the OnTick function

We have already implemented almost all EA functionality. Now it is time to develop the most important part — placing an order grid.

The standard _OnTick()_ function is called upon arrival of each symbol tick. We will use the function to check if the grid order is present and create it if is not.

**Bar start check**. However, performing a check at every tick is redundant. It would be sufficient to check the presence of the grid, for example, once every 5 minutes. To do this, add the code checking the bar start to the _OnTick()_ function. If this is not the first tick from the bar start, complete the function operation without doing anything:

```
   if( !pdxIsNewBar() ){
      return;
   }
```

The _pdxIsNewBar()_ function looks as follows:

```
bool pdxIsNewBar(){
   static datetime Old_Time;
   datetime New_Time[1];

   if(CopyTime(_Symbol,_Period,0,1,New_Time)>0){
      if(Old_Time!=New_Time[0]){
         Old_Time=New_Time[0];
         return true;
      }
   }
   return false;
}
```

In order for the EA to check our conditions once every five minutes, it should be launched on M5 timeframe.

**Checking take profit**. Before checking the grid availability, we should check whether a take profit on all currently open grid positions is reached. If the take profit has been reached, then call the _closeAllPos()_ function described above.

```
   if(checkTakeProfit()){
      closeAllPos();
   }
```

To check for a take profit, call the _checkTakeProfit()_ function. It calculates profit on all currently open positions and compares it with the value of the _takeProfit_ input parameter:

```
bool checkTakeProfit(){
   if( takeProfit<=0 ) return false;
   double curProfit=0;
   double profit=0;

   #ifdef __MQL5__
      int cntMyPos=PositionsTotal();
      for(int ti=cntMyPos-1; ti>=0; ti--){
         if(PositionGetSymbol(ti)!=_Symbol) continue;
         if(EA_Magic>0 && PositionGetInteger(POSITION_MAGIC)!=EA_Magic) continue;

         profit+=PositionGetDouble(POSITION_PROFIT);
         profit+=PositionGetDouble(POSITION_SWAP);
      }
   #else
      int cntMyPos=OrdersTotal();
      if(cntMyPos>0){
         for(int ti=cntMyPos-1; ti>=0; ti--){
            if(OrderSelect(ti,SELECT_BY_POS,MODE_TRADES)==false) continue;
            if( OrderType()==OP_BUY || OrderType()==OP_SELL ){}else{ continue; }
            if(OrderSymbol()!=_Symbol) continue;
            if(EA_Magic>0 && OrderMagicNumber()!=EA_Magic) continue;

            profit+=OrderCommission();
            profit+=OrderProfit();
            profit+=OrderSwap();
         }
      }
   #endif
   if(profit>takeProfit){
      return true;
   }
   return false;
}
```

**Displaying the Close All button**. Do not forget about the _Close All_ button we have implemented but have not displayed yet. It is time to add its function call:

```
getmeinfo_btn(_Symbol);
```

It will look like this:

![Close All button](https://c.mql5.com/2/36/close_all.png)

**Placing a grid**. Finally, we approach the most important part of our EA. It looks quite simple since all the code is once again hidden behind the functions:

```
   // if a symbol features open positions or placed orders, then
   if( existLimits() ){
   }else{
   // otherwise, place the grid
      initLimits();
   }
```

The _existLimits()_ function returns 'true' if the symbol features open positions or placed orders:

```
bool existLimits(){
   #ifdef __MQL5__
      int cntMyPos=PositionsTotal();
      for(int ti=cntMyPos-1; ti>=0; ti--){
         if(PositionGetSymbol(ti)!=_Symbol) continue;
         if(EA_Magic>0 && PositionGetInteger(POSITION_MAGIC)!=EA_Magic) continue;
         return true;
      }
      int cntMyPosO=OrdersTotal();
      for(int ti=cntMyPosO-1; ti>=0; ti--){
         ulong orderTicket=OrderGetTicket(ti);
         if(OrderGetString(ORDER_SYMBOL)!=_Symbol) continue;
         if(EA_Magic>0 && OrderGetInteger(ORDER_MAGIC)!=EA_Magic) continue;
         return true;
      }
   #else
      int cntMyPos=OrdersTotal();
      if(cntMyPos>0){
         for(int ti=cntMyPos-1; ti>=0; ti--){
            if(OrderSelect(ti,SELECT_BY_POS,MODE_TRADES)==false) continue;
            if(OrderSymbol()!=_Symbol) continue;
            if(EA_Magic>0 && OrderMagicNumber()!=EA_Magic) continue;
            return true;
         }
      }
   #endif

   return false;
}
```

If the function returns 'true', do nothing. Otherwise, place a new order grid using the _initLimits()_ function:

```
void initLimits(){
   // price for setting grid orders
   double curPrice;
   // current symbol price
   MqlTick lastme;
   SymbolInfoTick(_Symbol, lastme);
   // if no current price is obtained, cancel placing the grid
   if( lastme.bid==0 ){
      return;
   }

   // minimum distance from the price available for placing stop losses and,
   // most probably, pending orders
   double minStop=SymbolInfoDouble(_Symbol, SYMBOL_POINT)*SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL);

   // place Long orders
   curPrice=lastme.bid;
   for(uint i=0; i<maxLimits; i++){
      curPrice+=ST;

      if( curPrice-lastme.ask < minStop ) continue;
      if(!pdxSendOrder(MY_BUYSTOP, curPrice, 0, 0, Lot, 0, "", _Symbol)){
      }
   }
   // place Short orders
   curPrice=lastme.ask;
   for(uint i=0; i<maxLimits; i++){
      curPrice-=ST;
      if( lastme.bid-curPrice < minStop ) continue;
      if(!pdxSendOrder(MY_SELLSTOP, curPrice, 0, 0, Lot, 0, "", _Symbol)){
      }
   }
}
```

### Testing the EA

Our EA is ready. Now we should test it and draw conclusions about the trading strategy's performance.

Since our EA works both in MetaTrader 4 and MetaTrader 5, we are able to select the terminal version, in which to perform the test. Although the choice is quite obvious here. MetaTrader 5 is considered to be more comprehensible and better.

First, let's perform testing without any optimization. Our EA should not fully depend on the inputs' values when using reasonable values. Let's take:

- EURUSD symbol;
- M5 timeframe;
- period from August 1, 2018 to January 1, 2019;
- tets mode _1 Minute OHLC_.

Inputs' default values remain intact (lot 0.01, step 10 points, 7 orders per grid, take profit $1).

The result is shown below:

![Balance chart during the first EA test](https://c.mql5.com/2/36/eurusd_test1.png)

As can be seen from the chart, everything went well the entire month and one week. We managed to earn almost $100 with a drawdown of $30. Then a seemingly impossible event happened. Have a look at the visualizer to see how the price moved in September:

![Visualization result](https://c.mql5.com/2/35/visual_test1.png)

It started on September 13, a bit after 16:15. First, the price touched a Long order. Then it activated 2 Short orders, 2 more Long orders and finally the remaining 5 Short orders. As a result, we have 3 Long orders and 7 Short ones.

This cannot be seen on the image but the price did not move further below. By September 20, it returned to the top point and activated the remaining 4 Long orders.

As a result, we have all 7 Short and 7 Long orders open. This means we will never achieve take profit any more.

If we have a look at the further price movement, it will go further up by about 80 points. If we had, say, 13 orders in our chain, then we probably could reverse the situation and gain profit.

Even if this were not enough, later the price would go down by 200 points, so with 30 orders in the chain, we could theoretically get a take profit. Although this would probably take months, and the drawdown would be huge.

**Test the new number of orders in the grid**. Let's check our assumptions. 13 orders in the grid changed nothing, while 20 orders allowed us to emerge unscathed:

![Testing EURUSD, 20 orders in the grid](https://c.mql5.com/2/36/eurusd_test2.png)

However, the drawdown comprised about $300, while the total profit is slightly over $100. Perhaps, our trading strategy is not a complete failure but it definitely needs dramatical improvements.

Therefore, there is no point in optimizing it now. But let's try to do that anyway.

**Optimization**. Optimization is performed using the following parameters:

- number of orders in a grid: 4-21;
- grid step: 10-20 points;
- take profit remains the same ($1).

The step of 13 points has turned out to be the best, while the number of orders in the grid is 16:

![Testing EURUSD, 16 orders in the grid, the step of 13 points](https://c.mql5.com/2/36/eurusd_test3.png)

This is the result of testing in the [_"Every tick based on real ticks"_](https://www.metatrader5.com/en/terminal/help/algotrading/tick_generation#tick_mode "https://www.metatrader5.com/en/terminal/help/algotrading/tick_generation#tick_mode") mode. Despite the fact that the result is positive, $119 for 5 months with the drawdown of $221 is not the best result. This means our trading strategy really needs improvements.

### Possible ways to improve the trading strategy

Apparently, using only one take profit for all positions is insufficient. From time to time, there are situations when the price hits all or most of the orders in both directions. In this case, we may wait for profit for months, if not infinitely.

Let's think about what we can do to solve the detected issue.

**Manual control**. Of course, the easiest way is to manually control the EA from time to time. If a potential issue is brewing, we may place additional orders or simply close all positions.

**Setting an additional grid**. We may try to set another grid if, for example, 70% of orders in one direction and 70% of orders in another direction are affected. Orders from the additional grid may allow for a quick increase in the number of open positions in one direction, thus reaching the take profit faster.

Apart from the number of open positions, we may check the last position open date. For instance, if more than a week has passed since opening the last position, a new grid is set.

With both options, there is a risk to further aggravate the situation increasing the already large drawdown.

**Close the entire grid and open a new one**. Apart from setting an additional grid, we may close all positions and placed orders belonging to the current grid admitting we lost the battle but not the war.

There are multiple cases when we can do that:

- if more than N% orders are opened in both directions,
- if N days have passed since opening the last position,
- if the loss on all open positions has reached $N.


As an example, let's try to implement the last item from the list. We will add an incoming parameter where we will set the size of the loss in $, at which we need to close positions on the current grid and open a new one. A number less than 0 is to be used for setting a loss:

```
input double      takeLoss=0; //Close in case of a loss, $
```

Now we have to re-write the _checkTakeProfit()_ function, so that it returns profit for all open positions rather than 'true' or 'false':

```
double checkTakeProfit(){
   double curProfit=0;
   double profit=0;

   #ifdef __MQL5__
      int cntMyPos=PositionsTotal();
      for(int ti=cntMyPos-1; ti>=0; ti--){
         if(PositionGetSymbol(ti)!=_Symbol) continue;
         if(EA_Magic>0 && PositionGetInteger(POSITION_MAGIC)!=EA_Magic) continue;

         profit+=PositionGetDouble(POSITION_PROFIT);
         profit+=PositionGetDouble(POSITION_SWAP);
      }
   #else
      int cntMyPos=OrdersTotal();
      if(cntMyPos>0){
         for(int ti=cntMyPos-1; ti>=0; ti--){
            if(OrderSelect(ti,SELECT_BY_POS,MODE_TRADES)==false) continue;
            if( OrderType()==OP_BUY || OrderType()==OP_SELL ){}else{ continue; }
            if(OrderSymbol()!=_Symbol) continue;
            if(EA_Magic>0 && OrderMagicNumber()!=EA_Magic) continue;

            profit+=OrderCommission();
            profit+=OrderProfit();
            profit+=OrderSwap();
         }
      }
   #endif
   return profit;
}
```

The changes are shown in yellow.

Now we are able to revise the _OnTick()_ function, so that it checks a stop loss on all positions in addition to a take profit:

```
   if(takeProfit>0 && checkTakeProfit()>takeProfit){
      closeAllPos();
   }else if(takeLoss<0 && checkTakeProfit()<takeLoss){
      closeAllPos();
   }
```

### Additional testing

Let's see if these improvements were of any use.

We are going to optimize only stop loss in $ within the range from -$5 to -$100. The remaining parameters remain at levels selected during the last test (the step of 13 points, 16 orders in the grid).

Most profit is received with the stop loss of -$56. The profit within 5 months comprises $156 with the maximum drawdown of $83:

![Testing EURUSD, the stop loss of -$56](https://c.mql5.com/2/36/eurusd_test4.png)

Analyzing the chart, we can see that stop loss was activated only once for five months. The result is, of course, better in terms of profit to drawdown ratio.

However, before making final conclusions, let's check whether our EA can yield at least some profit in the long term with the selected parameters. Let's try it on the period of the last five years:

![Testing EURUSD with the stop loss, 5 years](https://c.mql5.com/2/36/eurusd_test5.png)

The results are discouraging. Perhaps, additional optimization could improve it. In any case, the use of this grid trading strategy requires a radical improvement. The idea that additional open positions will sooner or later overcome the losses is incorrect in terms of long-term automated trading.

### Adding stop losses and take profits for orders

Unfortunately, other EA improvement options listed above do not lead to better results either. But what about stop losses for separate deals? Perhaps, adding stop losses will improve our EA for long-term automated trading.

Optimization on five-year history showed better results as compared to the above.

The stop loss of 140 points and the take profit of 50 points were most efficient. If not a single position is opened on the current grid within 30 days, it is closed and a new grid is opened.

The final result is shown below:

![Using stop loss and take profit for orders](https://c.mql5.com/2/36/grider_with_stops.png)

The profit is $351 with the drawdown of $226. Of course, this is better than trading results obtained without using a stop loss. However, we cannot help but notice that all results obtained when closing the current grid in less than 30 days after performing the last deal are loss-making. Besides, the number of days exceeding 30 mostly ends up in loss as well. So this result is more a coincidence than a rule.

### Conclusion

The main objective of this article was to write a trading EA working in both MetaTrader 4 and MetaTrader 5. We succeeded in that.

Also, once again, we saw that testing an EA on several months of history is insufficient unless you are ready to adjust its parameters every week.

Unfortunately, ideas based on simple griders are not viable. But maybe we missed something. If you know how to develop a basic grider that is actually profitable, please write your suggestions down in the comments.

Anyway, our findings do not mean that grid-based trading strategies cannot be profitable. For example, look at these signals:

- [EURUSD grider](https://www.mql5.com/en/signals/533422);

- [AUDUSD grider](https://www.mql5.com/en/signals/533425);
- [AUDCAD grider](https://www.mql5.com/en/signals/551229).


The signals are based on a single grider, which is more complex than the one described here. That grider can actually yield up to 100% of profit per month. We will dwell on it more in the next article about griders.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/5596](https://www.mql5.com/ru/articles/5596)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/5596.zip "Download all attachments in the single ZIP archive")

[griderEA.ex5](https://www.mql5.com/en/articles/download/5596/griderea.ex5 "Download griderEA.ex5")(112.85 KB)

[griderEA.mq5](https://www.mql5.com/en/articles/download/5596/griderea.mq5 "Download griderEA.mq5")(77.8 KB)

[griderEA.ex4](https://www.mql5.com/en/articles/download/5596/griderea.ex4 "Download griderEA.ex4")(36.54 KB)

[griderEA.mq4](https://www.mql5.com/en/articles/download/5596/griderea.mq4 "Download griderEA.mq4")(77.8 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Developing a cross-platform grid EA: testing a multi-currency EA](https://www.mql5.com/en/articles/7777)
- [Developing a cross-platform grid EA (Last part): Diversification as a way to increase profitability](https://www.mql5.com/en/articles/7219)
- [Developing a cross-platform grider EA (part III): Correction-based grid with martingale](https://www.mql5.com/en/articles/7013)
- [Developing a cross-platform Expert Advisor to set StopLoss and TakeProfit based on risk settings](https://www.mql5.com/en/articles/6986)
- [Developing a cross-platform grider EA (part II): Range-based grid in trend direction](https://www.mql5.com/en/articles/6954)
- [Selection and navigation utility in MQL5 and MQL4: Adding data to charts](https://www.mql5.com/en/articles/5614)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/313362)**
(27)


![Ken Zeng](https://c.mql5.com/avatar/2019/5/5CED2E26-7944.jpg)

**[Ken Zeng](https://www.mql5.com/en/users/zaifeng1984)**
\|
28 May 2019 at 14:41

If you have a ready-made class and don't use it, you have to build the wheel over and over again.


![Olusegun O  Enujowo ](https://c.mql5.com/avatar/avatar_na2.png)

**[Olusegun O Enujowo](https://www.mql5.com/en/users/shegz)**
\|
8 Jul 2019 at 09:05

I like it that someone can put a lot of working hours into developing something like this. Not because you're looking to [make profit](https://www.mql5.com/en/articles/401 "Article: Why Is MQL5 Market the Best Place for Selling Trading Strategies and Technical Indicators"), but because you wanted to achieve a purpose: which is to create an EA that will work on MT4 and MT5. Kudos!

Let's say I have a strategy that I feel can bring better profit(of cause, that's the reason we are all here- profit!), will you like to have it and as such, create or modify your EA on such strategy? It has nothing to do with grid system I promise.

![ArmenAlaverdyan](https://c.mql5.com/avatar/2020/11/5FC49449-E48D.jpg)

**[ArmenAlaverdyan](https://www.mql5.com/en/users/armenalaverdyan)**
\|
11 Jul 2019 at 10:24

I would like to give the author an important piece of advice. It is necessary to move the network of orders [evenly](https://www.mql5.com/en/articles/2742 "Article: Statistical Distributions in MQL5 - Taking the Best of R and Making it Faster ") as the price changes


![Ronghua Hu](https://c.mql5.com/avatar/avatar_na2.png)

**[Ronghua Hu](https://www.mql5.com/en/users/bonaccihu)**
\|
20 Aug 2019 at 13:36

It just so happens that recently I have been studying [grid trading](https://www.mql5.com/en/articles/6954 "Article: Creating a Cross-Platform Expert Advisor-Grid (Part II): Grid in the Range in the Direction of the Trend "), which is theoretically profitable, but when you are inside an oscillating trend, it is very easy to get caught in a trap with a big retracement, but instead, it is a good method to use at the upper and lower limits of an oscillating trend.


![honey bee](https://c.mql5.com/avatar/2024/3/65FFDFC7-C041.png)

**[honey bee](https://www.mql5.com/en/users/honeybee998)**
\|
7 Apr 2024 at 08:26

**Ken Zeng [#](https://www.mql5.com/zh/forum/313541#comment_11856470):**

If you have a ready-made class, don't use it, you have to build the wheel again and again.

What do you mean by ready-made classes?


![Studying candlestick analysis techniques (part IV): Updates and additions to Pattern Analyzer](https://c.mql5.com/2/35/Logo__3.png)[Studying candlestick analysis techniques (part IV): Updates and additions to Pattern Analyzer](https://www.mql5.com/en/articles/6301)

The article presents a new version of the Pattern Analyzer application. This version provides bug fixes and new features, as well as the revised user interface. Comments and suggestions from previous article were taken into account when developing the new version. The resulting application is described in this article.

![A DLL for MQL5 in 10 Minutes (Part II): Creating with Visual Studio 2017](https://c.mql5.com/2/35/logo__2.png)[A DLL for MQL5 in 10 Minutes (Part II): Creating with Visual Studio 2017](https://www.mql5.com/en/articles/5798)

The original basic article has not lost its relevance and thus if you are interested in this topic, be sure to read the first article. However much time has passed since then, so the current Visual Studio 2017 features an updated interface. The MetaTrader 5 platform has also acquired new features. The article provides a description of dll project development stages, as well as DLL setup and interaction with MetaTrader 5 tools.

![How to visualize multicurrency trading history based on HTML and CSV reports](https://c.mql5.com/2/35/mql5-article-html-csv.png)[How to visualize multicurrency trading history based on HTML and CSV reports](https://www.mql5.com/en/articles/5913)

Since its introduction, MetaTrader 5 provides multicurrency testing options. This possibility is often used by traders. However the function is not universal. The article presents several programs for drawing graphical objects on charts based on HTML and CSV trading history reports. Multicurrency trading can be analyzed in parallel, in several sub-windows, as well as in one window using the dynamic switching command.

![Using MATLAB 2018 computational capabilities in MetaTrader 5](https://c.mql5.com/2/35/ext_infin2.png)[Using MATLAB 2018 computational capabilities in MetaTrader 5](https://www.mql5.com/en/articles/5572)

After the upgrade of the MATLAB package in 2015, it is necessary to consider a modern way of creating DLL libraries. The article uses a sample predictive indicator to illustrate the peculiarities of linking MetaTrader 5 and MATLAB using modern 64-bit versions of the platforms, which are utilized nowadays. With the entire sequence of connecting MATLAB considered, MQL5 developers will be able to create applications with advanced computational capabilities much faster, avoiding «pitfalls».

[![](https://www.mql5.com/ff/sh/bhdtjfb1zry09943z2/267b575d2182c180804d340af38ce02c.jpg)\\
Trade from your iPhone or Android device\\
\\
You only need an internet connection to use the new powerful MetaTrader 5 Web terminal\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=wtigumvtenarnsocpyfoqnanxrilnbxx&s=ec8c539e52b83881ff2d16eaff6913b25803952eb277cac55f670a102b2edc1f&uid=&ref=https://www.mql5.com/en/articles/5596&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5070499546598938359)

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