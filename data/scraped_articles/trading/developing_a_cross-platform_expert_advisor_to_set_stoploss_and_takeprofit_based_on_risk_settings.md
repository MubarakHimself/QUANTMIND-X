---
title: Developing a cross-platform Expert Advisor to set StopLoss and TakeProfit based on risk settings
url: https://www.mql5.com/en/articles/6986
categories: Trading
relevance_score: 6
scraped_at: 2026-01-23T11:28:02.678906
---

[![](https://www.mql5.com/ff/sh/592yc11u3j4rs5z9z2/01.png)How AI helps create robots for MetaTrader 5Learn from our book "Neural Networks in Algo Trading with MQL5"Read](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/neurobook%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.neurobook%26utm_content=visit.page%26utm_campaign=neurobook.promo.04.2024&a=ghrobswocqgvhztzjldphupateyllpro&s=9929cb0b8629585b5a42fabc06c525e41f6c0ebdf3045d044a5413b93ea88b47&uid=&ref=https://www.mql5.com/en/articles/6986&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=6484715862083150609)

MetaTrader 5 / Trading


### Introduction

As you probably know, following of money management rules is highly recommended for any trading. It means that one is not recommended to
enter a trade in which more than N% of deposit can be lost.

N is chosen by the trader. In order to comply with this rule, one should correctly calculate the trading lot value.

At relevant master classes, presenters usually show a ready Excel file, which includes relevant lot calculation formulas for each symbol.
And thus they obtain the required lot value by "simply entering" their stop loss value.

Is this really that "simple"? The lot calculation operation can take a minute or more. So when you finally determine the lot size, the price
can move very far from the intended entry point. Moreover, this requires from you performing of extra operations. Another disadvantage of
this method, is that manual calculations often increase the chance of making an error.

So let's try make this process really simple. To do this, we will create an Expert Advisor for setting the opening and Stop Loss prices in the
visual mode. Based on these parameters and your risk value, the EA will set the appropriate lot value and will open a position in the relevant
direction.

### Task definition

We have outlined the first task.

Another task that will be performed by our Expert Advisor is setting a Take Profit price based on the preferred TP to SL ratio.

Gerchik and other successful traders recommend to use a Take Profit, which is at least 3 times your Stop Loss. That is, if you use Stop Loss of 40
points, then Take Profit should be at least 120 points. If there are not many chances for the price to reach this level, you should better
abstain from entering the trade.

For a more convenient statistics calculation, it is desirable to use the same SL to TP ratio. For example, enter trades with a TP to SL ratio of 3
to 1, 4 to 1, etc. You should choose the specific ratio for yourself based on your trading objectives.

The question is the following: how to place Take Profit without wasting time? Is Excel the only possible solution? Or should we try to guess the
approximate position on the chart?

This possibility will be implemented in Expert Advisor settings. A special parameter will be provided for entering the Take Profit to Stop
Loss ratio. For example, the value of 4 means that the ratio is 4 to 1. The EA then will set the Take Profit which is 4 times the Stop Loss value.

**Expert Advisor format**. Before proceeding to the EA development, we need to decide on its operation mode. This is probably the
most difficult task.

I have been using a similar EA for more than a year. Since its emergence, the EA has greatly changed.

In its first version, a dialog window was shown after the EA launch: it allowed to change any of the new position opening parameters. These
settings were automatically saved and were used during further EA launches. And this was the main advantage of this method. Very often you
configure desired settings once and keep using them in the future.

However the dialog window with the settings has a big disadvantage: it takes up almost the entire chart window space. Price movement cannot be
tracked due to this window. Since the dialog window is no longer used after the first setup, the window hinders the chart view without
providing any benefits.

Here is an example of a chart and a dialog window with the width of 640 pixels:

![The Expert Advisor version with a dialog window](https://c.mql5.com/2/36/dialog_version.png)

Fig.1. Expert Advisor version with a dialog window

As you can see, the window does not even completely fit on the screen.

In an attempt to solve this problem, I created 2 more EA versions.

In the first version the settings window was hidden by default and could be opened by clicking on the _Settings_
button. This EA is still available in the MetaTrader 5 Market.

The second version does not have any settings window. All EA settings are given in input parameters. This eliminates the use of a dialog
window. However this involves one of the following troubles:

- the need to configure the EA anew during each launch
- the need to upload settings from a SET file during each launch
- the need to change default settings in the EA code and to recompile it

Being programmers, I hope you would choose the third option. In this article, we will create a simplified clone of the second EA version. Here is
how the EA operation will look like:

![Expert Advisor version without a dialog box](https://c.mql5.com/2/36/nodialog_version.png)

Fig.2. Expert Advisor version without a dialog window

### Input parameters

For a netter understanding of the entire scope of work, let's look at the EA input parameters:

![Expert Advisor input parameters](https://c.mql5.com/2/36/input.png)

Fig.3. Expert Advisor input parameters

The trading activity will be governed by Stop Loss. Therefore let us pay attention to the first two parameters: " _Stop_
_Loss type_" and " _Stop Loss value in $ or %_".

The value is set in dollars by default, which is indicated in the _"Stop Loss type_" parameter. The Stop Loss can also
be set as percentage of your balance. If the Stop Loss as a percentage is selected, the specified value cannot exceed 5% of the deposit. This is
done to avoid errors when setting the EA parameters.

Parameter " _Stop Loss value in $ or %_" indicates the amount which you can afford losing in case of Stop Loss.

**Stop Loss value in cents (keys 7 and 8)**.Another Stop Loss related parameter is: " _Stop Loss value in cents_
_(keys 7 and 8)_".

Next, let us implement a set of shortcut keys, using which we can set the desired Stop Loss with one click. For example, if you always use a stop loss
of 7 cents in trading, then you only need to press the 7 key on the keyboard, and the Stop Loss will be set at the specified distance from the
current price. We will discuss this in detail later.

Please note that this parameter determines not the amount which you will lose in case of Stop Loss, but it is the distance between the open price and
the Stop Loss trigger price.

**No deal entry if with min. lot the risk is greater than specified.** Since the trade entry lot value is calculated
automatically based on the "

_Stop Loss value in $ or %_" parameter, a situation may occur when the minimum allowed lot would lead to a risk per trade higher than
the specified value.

In this case you can enter a trade with the minimum lot and ignore the risk, or you can cancel deal opening. The appropriate behavior is defined
in this parameter.

By default, the EA prevents from market entry in case if an increased risk.

**Cancel limit order after, hours**. In addition to market entry, the EA supports placing of limit orders based on the specified
price. This parameter allows limiting the order lifetime.

The lifetime is set in hours. For example, if the lifetime is set to 2 hours and the limit order does not trigger within 2 hours, such an order
shall be deleted.

**Tale Profit multiplier**. If you are trading with a specific Take Profit to Stop Loss ratio, this parameter can be used to configure
the automatic Take Profit setting in accordance with your rule.

The default value is 4. It means that the Take Profit will be set at such a price that the profit in case of Take Profit will be 4 times of possible
loss.

**Other parameters**. You can also:

- change the Magic number for the EA orders
- set a comment to orders
- select the interface language: English or Russian (default)


### Position opening function

Since we are writing a cross-platform Expert Advisor, it should work in both MetaTrader 4 and MetaTrader 5. However, different EA versions have
different position opening functions. To enable the EA operation in two platforms, we will use conditional compilation.

I already mentioned this type of compilation in my articles. For example, in the article [Developing \\
a cross-platform grider EA](https://www.mql5.com/en/articles/5596).

In short, the conditional compilation code is as follows:

```
#ifdef __MQL5__
   //MQL5 code
#else
   //MQL4 code
#endif
```

In this article, we will use the conditional compilation possibilities 3 times, to of which concern the position opening function. The rest
of the code can work both in MetaTrader 4 and MetaTrader 5.

The position opening function was already developed in the article [Developing a \\
cross-platform grider EA](https://www.mql5.com/en/articles/5596). Therefore let us use a ready solution. We need to add to it functionality for setting the limit order lifetime:

```
// Possible order types for the position opening function
enum TypeOfPos
  {
   MY_BUY,
   MY_SELL,
   MY_BUYSTOP,
   MY_BUYLIMIT,
   MY_SELLSTOP,
   MY_SELLLIMIT,
   MY_BUYSLTP,
   MY_SELLSLTP,
  };

// Selecting deal filling type for MT5
#ifdef __MQL5__
   enum TypeOfFilling //Deal filling type
     {
      FOK,//ORDER_FILLING_FOK
      RETURN,// ORDER_FILLING_RETURN
      IOC,//ORDER_FILLING_IOC
     };
   input TypeOfFilling  useORDER_FILLING_RETURN=FOK; //Order filling mode
#endif

/*
Position opening or limit order placing function
*/
bool pdxSendOrder(TypeOfPos mytype, double price, double sl, double tp, double volume, ulong position=0, string comment="", string sym="", datetime expiration=0){
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
      }else{
         #ifdef __MQL5__
         #else
            MqlTick latest_price;
            SymbolInfoTick(sym,latest_price);
            if( mytype == MY_SELL ){
               price=latest_price.ask;
            }else if( mytype == MY_BUY ){
               price=latest_price.bid;
            }
         #endif
      }
   #ifdef __MQL5__
      ENUM_TRADE_REQUEST_ACTIONS action=TRADE_ACTION_DEAL;
      ENUM_ORDER_TYPE type=ORDER_TYPE_BUY;
      switch(mytype){
         case MY_BUY:
            action=TRADE_ACTION_DEAL;
            type=ORDER_TYPE_BUY;
            break;
         case MY_BUYSLTP:
            action=TRADE_ACTION_SLTP;
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
         case MY_SELLSLTP:
            action=TRADE_ACTION_SLTP;
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
      if(expiration>0){
         mrequest.type_time = ORDER_TIME_SPECIFIED_DAY;
         mrequest.expiration = expiration;
      }
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
            if(action!=TRADE_ACTION_SLTP){
               switch(type){
                  case ORDER_TYPE_BUY:
//                     Alert("Order Buy #:",mresult.order," sl",sl," tp",tp," p",price," !!");
                     break;
                  case ORDER_TYPE_SELL:
//                     Alert("Order Sell #:",mresult.order," sl",sl," tp",tp," p",price," !!");
                     break;
               }
            }else{
//               Alert("Order Modify SL #:",mresult.order," sl",sl," tp",tp," !!");
            }
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

      if(OrderSend(sym, type, volume, price, 100, sl, tp, comment, EA_Magic, expiration)<0){
            msgErr(GetLastError());
      }else{
         switch(type){
            case OP_BUY:
               Alert("Order Buy sl",sl," tp",tp," p",price," !!");
               break;
            case OP_SELL:
               Alert("Order Sell sl",sl," tp",tp," p",price," !!");
               break;
            }
            return true;
      }

   #endif
   return false;
}
```

In MetaTrader 5, position fill type should be selected when opening a position. Therefore, add one more input for MetaTrader 5: " _Order fill_
_mode_".

Different brokers support different order filling types. The most popular one among brokers is ORDER\_FILLING\_FOK. Therefore it is selected by
default. If your broker does not support this mode, you can select any other desired mode.

### EA localization

Another mechanism, which was earlier described in [Developing a cross-platform grider \\
EA](https://www.mql5.com/en/articles/5596), is the possibility to localize text messages produced by the Expert Advisor. Therefore, we will not consider it again. Please read
the mentioned article for more details concerning the EA operation format.

### Programming the EA Interface

In this article, we will not consider EA development from scratch. It is also assumed that the reader has at least the basic MQL
knowledge.

In this article we will look at how the main EA parts are implemented. This will help you implement additional functionality if
necessary.



Let's start with the EA interface.

The following interface elements will be created at the EA launch:

- a comment will contain current spread in dollars, points and percent of price, as well as symbol session closing time;
- a horizontal line will be displayed at the spread distance from the current price, using which the stop loss will be placed;
- " _Show (0) open price line_" button will be added, which will show a green horizontal line at the order open price;
- another button is needed to open an order with specified parameters.

Thus, to create an order with the needed parameters (which are set using the EA inputs), it is necessary to move the red line to the price at which
the Stop Loss should be set.

If the red line is moved above the current price, a short position is opened. If the red line is moved below the current price, a long position is
opened.

The open position volume will be calculated automatically so that in case of Stop Loss you would lose the amount close to the one specified in EA
parameters. All you need to do is click the position opening button. A position at the market price will be opened after that.

If you want to place a limit order, you should additionally click " _Show (0) open price line_" and move the appeared
green line to the price at which you want to open a limit order. The limit order type and direction will be determined automatically based on
the positions of the Stop Loss and Take Profit lines (is Stop Loss is above or below the open price).

Basically, it is not necessary to click " _Show (0) open price line_" button. Once you move the red Stop Loss line to the
desired level, the button will be automatically pressed and a green open price line will appear. If you move this line, it will set a limit
order. If you leave the line where it is, a market position will be opened.

We have analyzed the operating principle. Now we can move on to programming.

**Working with the chart comment**. The standard _Comment_ function is used for working with chart comments.
Therefore, we need to prepare a string which will be shown on a chart via the

_Comment_ function. For this purpose, let us create a custom function _getmespread_:

```
/*
   Showing data on spread and session closing time in a chart comment
*/
void getmespread(){
   string msg="";

   // Get spread in the symbol currency
   curSpread=lastme.ask-lastme.bid;

   // If the market is not closed, show spread info
   if( !isClosed ){
      if(curSpread>0){
         StringAdd(msg, langs.Label1_spread+": "+(string) DoubleToString(curSpread, (int) SymbolInfoInteger(_Symbol, SYMBOL_DIGITS))+" "+currencyS+" ("+DoubleToString(curSpread/curPoint, 0)+langs.lbl_point+")");
         StringAdd(msg, "; "+DoubleToString(((curSpread)/lastme.bid)*100, 3)+"%");
      }else{
         StringAdd(msg, langs.Label1_spread+": "+langs.lblNo);
      }
      StringAdd(msg, "; ");
   }

   // Show market closing time if we could determine it
   if(StringLen(time_info)){
      StringAdd(msg, "   "+time_info);
   }

   Comment(msg);
}
```

The _getmespread_ function will be called during EA initialization ( _OnInit_) and at each new tick ( _OnTick_).

In _getmespread_, we use five global EA variables: _lastme_, _isClosed_, _time\_info_, _currencyS_, _curPoint_.

The _lastme_ variable stores data on the Ask, Bid and Last prices. The variable contents are updated in the _OnInit_
and _OnTick_ functions using the following command:

```
SymbolInfoTick(_Symbol,lastme);
```

Other variables are initialized in the OnInit function. _isClosed_ and _time\_info_ are initialized as follows:

```
  isClosed=false;
  // Get the current date
  TimeToStruct(TimeCurrent(), curDay);
  // Get symbol trading time for today
  if(SymbolInfoSessionTrade(_Symbol, (ENUM_DAY_OF_WEEK) curDay.day_of_week, 0, dfrom, dto)){
      time_info="";
      TimeToStruct(dto, curEndTime);
      TimeToStruct(dfrom, curStartTime);

         isEndTime=true;
         string tmpmsg="";
         tmp_val=curEndTime.hour;
         if(tmp_val<10){
            StringAdd(tmpmsg, "0");
         }
         StringAdd(tmpmsg, (string) tmp_val+":");
         tmp_val=curEndTime.min;
         if(tmp_val<10){
            StringAdd(tmpmsg, "0");
         }
         StringAdd(tmpmsg, (string) tmp_val);
         if(curEndTime.hour==curDay.hour){
            if(tmp_val>curDay.min){
            }else{
               isClosed=true;
            }
         }else{
            if(curEndTime.hour==0){
            }else{
               if( curEndTime.hour>1 && (curDay.hour>curEndTime.hour || curDay.hour==0)){
                  StringAdd(time_info, " ("+langs.lbl_close+")");
                  isClosed=true;
               }else if(curDay.hour<curStartTime.hour ){
                  StringAdd(time_info, " ("+langs.lbl_close+")");
                  isEndTime=false;
                  isClosed=true;
               }else if(curDay.hour==curStartTime.hour && curDay.min<curStartTime.min ){
                  StringAdd(time_info, " ("+langs.lbl_close+")");
                  isEndTime=false;
                  isClosed=true;
               }
            }
         }

         if(isEndTime){
            StringAdd(time_info, langs.lblshow_TIME+": "+tmpmsg+time_info);
         }else{
            StringAdd(time_info, langs.lblshow_TIME2+": "+tmpmsg+time_info);
         }
  }
```

The _currencyS_ variable will store the currency used for profit calculation for the current financial instrument. The currency
can be obtained using the following command:

```
currencyS=SymbolInfoString(_Symbol, SYMBOL_CURRENCY_PROFIT);
```

The symbol's point size will be stored in the _curPoint_ variable:

```
curPoint=SymbolInfoDouble(_Symbol, SYMBOL_POINT);
```

**Stop Loss line**. Only one line is shown on the chart upon EA launch: the red line for placing Stop Loss.

Similar to buttons, the line will be plotted in the _OnInit_ function. Before drawing the line, we need to check whether
there is already such a line on the chart. If there is a line, do not create a new line and other UI elements. Instead, add the following data to
global variables: the price of the current line level and the price of the deal open line (if there is any on the chart):

```
  // if there are Stop Loss and open price lines on the chart
  // add the appropriate prices to variables
  if(ObjectFind(0, exprefix+"_stop")>=0){
      draw_stop=ObjectGetDouble(0, exprefix+"_stop", OBJPROP_PRICE);
      if(ObjectFind(0, exprefix+"_open")>=0){
         draw_open=ObjectGetDouble(0, exprefix+"_open", OBJPROP_PRICE);
      }
  // otherwise create the entire Expert Advisor UI
  }else{
      draw_open=lastme.bid;
      draw_stop=draw_open-(SymbolInfoInteger(_Symbol, SYMBOL_SPREAD)*curPoint);
      ObjectCreate(0, exprefix+"_stop", OBJ_HLINE, 0, 0, draw_stop);
      ObjectSetInteger(0,exprefix+"_stop",OBJPROP_SELECTABLE,1);
      ObjectSetInteger(0,exprefix+"_stop",OBJPROP_SELECTED,1);
      ObjectSetInteger(0,exprefix+"_stop",OBJPROP_STYLE,STYLE_DASHDOTDOT);
      ObjectSetInteger(0,exprefix+"_stop",OBJPROP_ANCHOR,ANCHOR_TOP);

      // other UI elements
  }
```

Is it generally possible that a chart already has UI elements?

If you do not implement the code, which deletes all created user interface elements when closing the EA, appropriate elements will be
definitely left on the chart. Even if this code is implemented, an error in the EA operation can occur due to which the EA will be closed but the
created elements will be left on the chart. Therefore, always check whether such an element exists on the chart before creating it.

Thus we have created a red line. Moreover, we set it as selected by default. Therefore you do not need to double click on the line to select it. You
simply need to move it to the desired price. However, if you now move the red line, nothing will happen. The code which performs appropriate
actions has not yet been implemented.

Any interaction with the UI elements is performed in the standard _OnChartEvent_ function. Moving of interface
elements generates an event with the ID

_CHARTEVENT\_OBJECT\_DRAG_. Thus, in order to implement movement on the chart, the _OnChartEvent_ function must
intercept this event, check which element has called it and if the element belongs to the EA, the required code should be executed:

```
void OnChartEvent(const int id,         // event ID
                  const long& lparam,   // event parameter of the long type
                  const double& dparam, // event parameter of the double type
                  const string& sparam) // event parameter of the string type
  {
   switch(id){
      case CHARTEVENT_OBJECT_DRAG:
         if(sparam==exprefix+"_stop"){
            setstopbyline();
            showOpenLine();
            ObjectSetInteger(0,exprefix+"_openbtn",OBJPROP_STATE, true);
         }
         break;
   }
}
```

After moving the red line, the _setstopbyline_ function will be started, which remembers the Stop Loss level for the future
order:

```
/*
"Remembers" the Stop Loss levels for the future order
*/
void setstopbyline(){
   // Receive the price in which the Stop Loss line is located
   double curprice=ObjectGetDouble(0, exprefix+"_stop", OBJPROP_PRICE);
   // If the price is different from the one in which the Stop Loss line was positioned at the EA launch,
   if(  curprice>0 && curprice != draw_stop ){
      double tmp_double=SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
      if( tmp_double>0 && tmp_double!=1 ){
         if(tmp_double<1){
            resval=DoubleToString(curprice/tmp_double, 8);
            if( StringFind(resval, ".00000000")>0 ){}else{
               curprice=MathFloor(curprice)+MathFloor((curprice-MathFloor(curprice))/tmp_double)*tmp_double;
            }
         }else{
            if( MathMod(curprice,tmp_double) ){
               curprice= MathFloor(curprice/tmp_double)*tmp_double;
            }
         }
      }
      draw_stop=STOPLOSS_PRICE=curprice;

      updatebuttontext();
      ChartRedraw(0);
   }
}
```

In addition to the _setstopbyline_ function, moving of the red line causes the appearance of the open price line on the chart
(the

_showOpenLine_ function) and the change of the " _Show (0) open price line_" button state.

**Open line and button**. The " _Show (0) open price line_" button is also created during EA initialization:

```
      if(ObjectFind(0, exprefix+"_openbtn")<0){
         ObjectCreate(0, exprefix+"_openbtn", OBJ_BUTTON, 0, 0, 0);
         ObjectSetInteger(0,exprefix+"_openbtn",OBJPROP_XDISTANCE,0);
         ObjectSetInteger(0,exprefix+"_openbtn",OBJPROP_YDISTANCE,33);
         ObjectSetString(0,exprefix+"_openbtn",OBJPROP_TEXT, langs.btnShowOpenLine);
         ObjectSetInteger(0,exprefix+"_openbtn",OBJPROP_XSIZE,333);
         ObjectSetInteger(0,exprefix+"_openbtn",OBJPROP_FONTSIZE, 8);
         ObjectSetInteger(0,exprefix+"_openbtn",OBJPROP_YSIZE,25);
      }
```

As mentioned above, any interaction with the interface elements is processed within the standard _OnChartEvent_
function. This includes button pressing. The event with the ID

_CHARTEVENT\_OBJECT\_CLICK_ is responsible for that. We need to intercept this event, check the event source and perform appropriate
actions. For this, let's add the additional

_case_ in the _switch_ operator of the _OnChartEvent_ function:

```
      case CHARTEVENT_OBJECT_CLICK:
         if (sparam==exprefix+"_openbtn"){
            updateOpenLine();
         }
         break;
```

The _updateOpenLine_ function which is called upon a click on button " _Show (0) open price line_", is a small
wrapper for the main

_showOpenLine_ function call. This function simply shows the open price on the chart:

```
void showOpenLine(){
   if(ObjectFind(0, exprefix+"_open")<0){
      draw_open=lastme.bid;
      ObjectCreate(0, exprefix+"_open", OBJ_HLINE, 0, 0, draw_open);
      ObjectSetInteger(0,exprefix+"_open",OBJPROP_SELECTABLE,1);
      ObjectSetInteger(0,exprefix+"_open",OBJPROP_SELECTED,1);
      ObjectSetInteger(0,exprefix+"_open",OBJPROP_STYLE,STYLE_DASHDOTDOT);
      ObjectSetInteger(0,exprefix+"_open",OBJPROP_ANCHOR,ANCHOR_TOP);
      ObjectSetInteger(0,exprefix+"_open",OBJPROP_COLOR,clrGreen);
   }
}
```

Now we need to re-write the _CHARTEVENT\_OBJECT\_DRAG_ event handler so that it could react to the moving of the Stop Loss line
and of the open price line:

```
      case CHARTEVENT_OBJECT_DRAG:
         if(sparam==exprefix+"_stop"){
            setstopbyline();
            showOpenLine();
            ObjectSetInteger(0,exprefix+"_openbtn",OBJPROP_STATE, true);
         }else if(sparam==exprefix+"_open"){
               curprice=ObjectGetDouble(0, exprefix+"_open", OBJPROP_PRICE);
               if( curprice>0 && curprice != draw_open ){
                  double tmp_double=SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
                  if( tmp_double>0 && tmp_double!=1 ){
                     if(tmp_double<1){
                        resval=DoubleToString(curprice/tmp_double, 8);
                        if( StringFind(resval, ".00000000")>0 ){}else{
                           curprice=MathFloor(curprice)+MathFloor((curprice-MathFloor(curprice))/tmp_double)*tmp_double;
                        }
                     }else{
                        if( MathMod(curprice,tmp_double) ){
                           curprice= MathFloor(curprice/tmp_double)*tmp_double;
                        }
                     }
                  }
                  draw_open=open=OPEN_PRICE=curprice;

                  updatebuttontext();
                  ObjectSetString(0,exprefix+"Edit3",OBJPROP_TEXT,0, (string) NormalizeDouble(draw_open, _Digits));
                  ChartRedraw(0);
               }
         }
         break;
```

**Take Profit line**. In addition to the red and green lines, we need to implement a dotted line. It will appear after you move the red
Stop Loss line to the desired price level. The dotted line will show the price of the deal Take Profit level:

![Stop Loss, open price and Take Profit lines](https://c.mql5.com/2/36/interface.png)

Fig.4. Stop Loss, open price and Take Profit lines

**Position open button**. The position open button is drawn similarly to button " _Show (0) open price line_".

The _CHARTEVENT\_OBJECT\_CLICK_ event will be generated upon a click on this button. Processing of this event was considered
earlier. As a result of pressing of the position open button, the

_startPosition_ function will be performed:

```
      case CHARTEVENT_OBJECT_CLICK:
         if (sparam==exprefix+"_send"){
            startPosition();
         }else if (sparam==exprefix+"_openbtn"){
            updateOpenLine();
         }
         break;
```

**Deleting UI elements upon EA operation completion**. Do not forget about the proper deletion of interface elements after the
completion of the Expert Advisor operation. If you do not do this, all elements will be left on the chart.

To perform any commands during EA operation completion, add it inside the standard _OnDeinit_ function:

```
void OnDeinit(const int reason)
  {

     if(reason!=REASON_CHARTCHANGE){
        ObjectsDeleteAll(0, exprefix);
        Comment("");
     }

  }
```

The _reason_ variable contains information about reasons for completing the EA operation. Now the only important reason for us is
the change of the timeframe (

_REASON\_CHARTCHANGE_). By default, a change in the timeframe leads to EA operation completion and its relaunch. This is not the most
acceptable behavior for us. In case of timeframe change, all set Stop Loss and open price levels will be reset.

Therefore, in _OnDeinit_ we should check if the reason for the EA closing is the timeframe change. And only if the reason for
closing is different, we should delete all elements and clear the comment to the chart.

### Implementing Expert Advisor shortcuts

We considered order placing using a mouse. However, sometimes fast operation using keys can be really useful.

Pressing of keyboard keys also refers to the interaction with the Expert Advisor UI elements. Namely, it refers to the entire chart on which the EA is
running. Keystrokes should be intercepted in the

_OnChartEvent_ function.

The _CHARTEVENT\_KEYDOWN_ event is generated whenever a key is pressed. The code of the pressed key is added to the _sparam_
parameter. This data is quite enough to start keystroke processing:

```
void OnChartEvent(const int id,         // event ID
                  const long& lparam,   // event parameter of the long type
                  const double& dparam, // event parameter of the double type
                  const string& sparam) // event parameter of the string type
  {
   string text="";
   double curprice=0;
   switch(id){
      case CHARTEVENT_OBJECT_CLICK:
         // Pressing of buttons on the chart
         break;
      case CHARTEVENT_OBJECT_DRAG:
         // Moving lines
         break;
      case CHARTEVENT_KEYDOWN:
         switch((int) sparam){
            // Terminate EA operation without placing an order
            case 45: //x
               closeNotSave();
               break;
            // Place an order and complete EA operation
            case 31: //s
               startPosition();
               break;
            // Set minimum possible Stop Loss to open a Buy position
            case 22: //u
               setMinStopBuy();
               break;
            // Set minimum possible Stop Loss to open a Sell position
            case 38: //l
               setMinStopSell();
               break;
            // Cancel the set open price
            case 44: //z
               setZero();
               ChartRedraw();
               break;
            // Set Stop Loss at 0.2% from the current price to open a Long position
            case 3: //2
               set02StopBuy();
               break;
            // Set Stop Loss at 0.2% from the current price to open a Short position
            case 4: //3
               set02StopSell();
               break;
            // Set Stop Loss to 7 cents from the current price (the CENT_STOP parameter)
            // To open a Long position
            case 8: //7
               set7StopBuy();
               break;
            // Set Stop Loss to 7 cents from the current price (the CENT_STOP parameter)
            // To open a Short position
            case 9: //8
               set7StopSell();
               break;
         }
         break;
   }
}
```

Thus, if you set a fixed Stop Loss equal to the minimum possible size, i.e. 0.2% or in cents, then you do not even need to use the mouse. Start the EA,
press key "

_2_" to set Stop Loss at 0.2% from the price in the Long direction, press key " _S_" and an appropriate position
will be opened.

And if you are using MetaTrader 5, then you can even start the EA from the keyboard by using assigned hot keys. In the _Navigator_ window,
call your EA context menu, click

_Set hotkey_ and enjoy fast access:

![Assigning hotkeys for Expert Advisors](https://c.mql5.com/2/36/hotkeys.png)

Fig.5. Assigning hotkeys to Expert Advisors

### Calculating appropriate deal volume

Now let us consider the position opening function _startPosition_. There is almost nothing interesting in it. We simply
check the availability of all the data we need: Stop Loss price, position open price and EA settings. After that the EA calculates the entry
lot value in accordance with you risk settings. Then the earlier mentioned function

_pdxSendOrder_ is called.

The most interesting of all is the deal volume calculation mechanism.

First, we need to calculate potential loss in case of a Stop Loss, with the minimum possible volumes. This functionality implementation in MQL5
differs from that in MQL4.

MQL5 has a special _OrderCalcProfit_ function which allows calculating the size of potential profit that can be obtained if
the symbol price reaches the specified level. This function allows calculating potential profit as well as potential loss.

A more complicated loss calculation formula is used in MQL4.

Here is the resulting function:

```
double getMyProfit(double fPrice, double fSL, double fLot, bool forLong=true){
   double fProfit=0;

   fPrice=NormalizeDouble(fPrice,_Digits);
   fSL=NormalizeDouble(fSL,_Digits);
   #ifdef __MQL5__
      if( forLong ){
         if(OrderCalcProfit(ORDER_TYPE_BUY, _Symbol, fLot, fPrice, fSL, fProfit)){};
      }else{
         if(OrderCalcProfit(ORDER_TYPE_SELL, _Symbol, fLot, fPrice, fSL, fProfit)){};
      }
   #else
      if( forLong ){
         fProfit=(fPrice-fSL)*fLot* (1 / MarketInfo(_Symbol, MODE_POINT)) * MarketInfo(_Symbol, MODE_TICKVALUE);
      }else{
         fProfit=(fSL-fPrice)*fLot* (1 / MarketInfo(_Symbol, MODE_POINT)) * MarketInfo(_Symbol, MODE_TICKVALUE);
      }
   #endif
   if( fProfit!=0 ){
      fProfit=MathAbs(fProfit);
   }

   return fProfit;
}
```

Thus we have calculated loss amount with the minimum volume. Now we need to determine the deal volume with which the loss will not increase the
specified risk settings:

```
      profit=getMyProfit(open, STOPLOSS_PRICE, lot);
      if( profit!=0 ){
         // If loss with the minimum lot is less than your risks,
         // calculate appropriate deal volume
         if( profit<stopin_value ){
            // get the desired deal volume
            lot*=(stopin_value/profit);
            // adjust the volume if it does not correspond to the minimum allowed step
            // for this trading instrument
            if( SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP)==0.01 ){
               lot=(floor(lot*100))/100;
            }else if( SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP)==0.1 ){
               lot=(floor(lot*10))/10;
            }else{
               lot=floor(lot);
            }
         // If loss with the minimum lot is greater than your risks,
         // cancel position opening if this option is set in EA parameters
         }else if( profit>stopin_value && EXIT_IF_MORE ){
            Alert(langs.wrnEXIT_IF_MORE1+": "+(string) lot+" "+langs.wrnEXIT_IF_MORE2+": "+(string) profit+" "+AccountInfoString(ACCOUNT_CURRENCY)+" ("+(string) stopin_value+" "+AccountInfoString(ACCOUNT_CURRENCY)+")!");
            return;
         }
      }
```

### Entry restrictions

The Expert Advisor checks certain conditions in order not to open deals whee opening is not allowed.

For example, during initialization, the EA checks the minimum allowable lot volume for the current instrument. And if this value is 0, the EA
will not start. Because it will not be able to open a position for such a symbol:

```
   if(SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN)==0){
      Alert(langs.wrnMinVolume);
      ExpertRemove();
   }
```

It also checks of symbol trading is allowed. If symbol trading is prohibited or only closing of previously opened trades is allowed, the EA
will not be launched:

```
   if(SymbolInfoInteger(_Symbol, SYMBOL_TRADE_MODE)==SYMBOL_TRADE_MODE_DISABLED || SymbolInfoInteger(_Symbol, SYMBOL_TRADE_MODE)==SYMBOL_TRADE_MODE_CLOSEONLY ){
      Alert(langs.wrnOnlyClose);
      ExpertRemove();
   }
```

When opening a position, the EA checks correctness of the opening price and of the Stop Loss level. For example, if the minimum price step is 0.25
and your stop loss is set at 23.29, the broker will not accept your order. Generally the EA can automatically adjust the price to a proper value
(and the Stop Loss price will be set to 23.25 or 23.5). Thus indication of an invalid price is not possible. However, an additional check is
also performed:

```
   if( SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE)>0 && SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE)!=1 ){
      resval=DoubleToString(STOPLOSS_PRICE/SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE), 8);
      if( StringFind(resval, ".00000000")>0 ){}else{
         Alert(langs.wrnSYMBOL_TRADE_TICK_SIZE+" "+(string) SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE)+"! "+langs.wrnSYMBOL_TRADE_TICK_SIZE_end);
         return;
      }
   }
```

### Conclusions

In this article we have implemented only the basic order placing features. But even these possibilities can be of great assistance to those
who trade Gerchik levels or any other levels.

I hope this will save you from using Excel tables. This will increase your trading speed and accuracy. And thus your profit can ultimately
grow.

Any EA improvements are welcome.

If you do not have enough programming skills but you need any specific functionality, feel free to contact me. However, this might require
some fee.

Please check the functionality of the extended versions of this Expert Advisor in the Market:

- [Expert Advisor for MetaTrader 4](https://www.mql5.com/en/market/product/32839);
- [Expert Advisor for MetaTrader 5](https://www.mql5.com/en/market/product/29801).

Perhaps your required functionality is already available.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/6986](https://www.mql5.com/ru/articles/6986)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/6986.zip "Download all attachments in the single ZIP archive")

[openWithRisk.ex5](https://www.mql5.com/en/articles/download/6986/openwithrisk.ex5 "Download openWithRisk.ex5")(126.3 KB)

[openWithRisk.ex4](https://www.mql5.com/en/articles/download/6986/openwithrisk.ex4 "Download openWithRisk.ex4")(56.42 KB)

[openWithRisk.mq5](https://www.mql5.com/en/articles/download/6986/openwithrisk.mq5 "Download openWithRisk.mq5")(119.01 KB)

[openWithRisk.mq4](https://www.mql5.com/en/articles/download/6986/openwithrisk.mq4 "Download openWithRisk.mq4")(119.01 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Developing a cross-platform grid EA: testing a multi-currency EA](https://www.mql5.com/en/articles/7777)
- [Developing a cross-platform grid EA (Last part): Diversification as a way to increase profitability](https://www.mql5.com/en/articles/7219)
- [Developing a cross-platform grider EA (part III): Correction-based grid with martingale](https://www.mql5.com/en/articles/7013)
- [Developing a cross-platform grider EA (part II): Range-based grid in trend direction](https://www.mql5.com/en/articles/6954)
- [Selection and navigation utility in MQL5 and MQL4: Adding data to charts](https://www.mql5.com/en/articles/5614)
- [Developing a cross-platform grider EA](https://www.mql5.com/en/articles/5596)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/319029)**
(1)


![Yury Stukalov](https://c.mql5.com/avatar/2022/3/62360F01-8059.jpg)

**[Yury Stukalov](https://www.mql5.com/en/users/spectrumdvd)**
\|
24 Jun 2019 at 18:23

Guys this is my advisor I ordered it on Freelance back in February 2018 I gave a description and now you have stolen it this is called copyright infringement ayay not good to do so


![Arranging a mailing campaign by means of Google services](https://c.mql5.com/2/36/logo_Csharp.png)[Arranging a mailing campaign by means of Google services](https://www.mql5.com/en/articles/6975)

A trader may want to arrange a mailing campaign to maintain business relationships with other traders, subscribers, clients or friends. Besides, there may be a necessity to send screenshotas, logs or reports. These may not be the most frequently arising tasks but having such a feature is clearly an advantage. The article deals with using several Google services simultaneously, developing an appropriate assembly on C# and integrating it with MQL tools.

![Library for easy and quick development of MetaTrader programs (part IX): Compatibility with MQL4 - Preparing data](https://c.mql5.com/2/36/MQL5-avatar-doeasy__4.png)[Library for easy and quick development of MetaTrader programs (part IX): Compatibility with MQL4 - Preparing data](https://www.mql5.com/en/articles/6651)

In the previous articles, we started creating a large cross-platform library simplifying the development of programs for MetaTrader 5 and MetaTrader 4 platforms. In the eighth part, we implemented the class for tracking order and position modification events. Here, we will improve the library by making it fully compatible with MQL4.

![Extract profit down to the last pip](https://c.mql5.com/2/36/MQL5-avatar-profit_digging__1.png)[Extract profit down to the last pip](https://www.mql5.com/en/articles/7113)

The article describes an attempt to combine theory with practice in the algorithmic trading field. Most of discussions concerning the creation of Trading Systems is connected with the use of historic bars and various indicators applied thereon. This is the most well covered field and thus we will not consider it. Bars represent a very artificial entity; therefore we will work with something closer to proto-data, namely the price ticks.

![Developing a cross-platform grider EA (part II): Range-based grid in trend direction](https://c.mql5.com/2/36/mql5_ea_adviser_grid.png)[Developing a cross-platform grider EA (part II): Range-based grid in trend direction](https://www.mql5.com/en/articles/6954)

In this article, we will develop a grider EA for trading in a trend direction within a range. Thus, the EA is to be suited mostly for Forex and commodity markets. According to the tests, our grider showed profit since 2018. Unfortunately, this is not true for the period of 2014-2018.

[![](https://www.mql5.com/ff/sh/9nb0c8df2rmwfn89z2/01.png) MetaTrader VPS vs regular cloud hosting services8 reasons why our solution is the best option for automated tradingRead](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/450486&a=dgmsfszgoedimaicrqqmagvqzpwuxkur&s=c59e3617ccf44fd54d4c50a03b44fd689ff7507b8fe4990c83772cc5419e627d&uid=&ref=https://www.mql5.com/en/articles/6986&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=6484715862083150609)

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