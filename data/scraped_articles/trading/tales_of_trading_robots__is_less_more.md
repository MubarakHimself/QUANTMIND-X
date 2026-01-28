---
title: Tales of Trading Robots: Is Less More?
url: https://www.mql5.com/en/articles/910
categories: Trading, Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T18:20:26.919157
---

[![](https://www.mql5.com/ff/sh/0uquj7zv5pmx2m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
Market analytics in MQL5 Channels\\
\\
Tens of thousands of traders have chosen this messaging app to receive trading tips.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=epadtzgppsywkaeumqycnulasoijfbgz&s=9615c3e5c371aa0d7b34529539d05c10df73b35a1e2213e4ceee008933c7ede0&uid=&ref=https://www.mql5.com/en/articles/910&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5069357209852249016)

MetaTrader 5 / Trading


Before I can solve a problem I must state it to myself. When I think I have found the solution I must prove that I am right.

I know of only one way to prove this; and that is, with my own money.

Jesse Livermore

### Prologue

In [The Last Crusade](https://www.mql5.com/en/articles/368 "Article The Last Crusade") we reviewed quite an interesting yet currently not widely used method for displaying market information - point and figure charts. The offered script could plot a chart. However it didn't suggest trading automation. Now we can try to automate the trading process using the point and figure chart for analysis and for making decision on the direction and volume of trade.

I will not write here the basic drawing principles; just look at a typical chart:

```
Copyright (c) 2012-2014 Roman Rich
Euro vs US Dollar, Box-20, Reverse-3

    1.4588 | \.....\.................................................................... | 1.4588
    1.4521 | X\....X\................................................................... | 1.4521
    1.4454 | XO\.\.XO\.................................................................. | 1.4454
    1.4388 | XOX\X\XO.\................................................................. | 1.4388
    1.4322 | XOXOXOXO..\................................................................ | 1.4322
    1.4256 | XOXOXOXO...\....\.......................................................... | 1.4256
    1.4191 | XOXO/OXO....\...X\......................................................... | 1.4191
    1.4125 | XOX/.O/O.....\..XO\........................................................ | 1.4125
    1.4060 | XO/../.O......\.XO.\....................................................... | 1.4060
    1.3996 | ./.....O.......\XO..\...................................................... | 1.3996
    1.3932 | .......OX.......XO...\....................................................X | 1.3932
    1.3868 | .......OXO..X.X.XOX...\.................................................X.X | 1.3868
    1.3804 | .......OXO..XOXOXOXOX..\..............................................X.XOX | 1.3804
    1.3740 | .......OXO..XOXOXOXOXO..\.................................\...........XOXOX | 1.3740
    1.3677 | .......OXOX.XO.O.OXOXO...\................................X\..........XOXOX | 1.3677
    1.3614 | .......OXOXOX....O.OXO....\...............................XO\.........XOXOX | 1.3614
    1.3552 | .......O.OXOX...../OXO.....\..............................XO.\........XOXOX | 1.3552
    1.3490 | .........OXOX..../.O.OX.....\.............................XO..\.......XOXO. | 1.3490
    1.3428 | .........OXOX.../....OXO.....\X.\.........................XO...\\...X.XOX.. | 1.3428
    1.3366 | .........O.OX../.....OXO......XOX\........................XO....X\..XOXOX.. | 1.3366
    1.3305 | ...........OX./......OXO....X.XOXO\.....................X.XO....XO\.XOXO... | 1.3305
    1.3243 | ...........OX/.......O.O....XOXOXOX\....................XOXO....XO.\XOX.../ | 1.3243
    1.3183 | ...........O/..........OX...XOXOXOXO\...................XOXOX.X.XOX.XOX../. | 1.3183
    1.3122 | .........../...........OXO..XOXOXOXO.\..........X...X.X.XOXOXOXOXOXOXO../.. | 1.3122
    1.3062 | .......................OXOX.XOXO.OXO..\.........XOX.XOXOXOXOXOXOXOXOX../... | 1.3062
    1.3002 | .......................O.OXOXO...O/O...\........XOXOXOXOXO.OXO.OXOXO../.... | 1.3002
    1.2942 | .........................OXOX..../.O....\.......XOXOXOXOX..OX..OXOX../..... | 1.2942
    1.2882 | .........................O.OX.../..O.....\......XOXO.OXO...OX..OXOX./...... | 1.2882
    1.2823 | ...........................OX../...OX.....\.....XO...OX.../OX..O/OX/....... | 1.2823
    1.2764 | ...........................OX./....OXO.....\....X....OX../.O.../.O/........ | 1.2764
    1.2706 | ...........................OX/.....OXO..X...\...X....O../......../......... | 1.2706
    1.2647 | ...........................O/......O.OX.XOX..\..X....../................... | 1.2647
    1.2589 | .........................../.........OXOXOXO..\.X...../.................... | 1.2589
    1.2531 | .....................................OXOXOXO...\X..../..................... | 1.2531
    1.2474 | .....................................OXO.OXO....X.../...................... | 1.2474
    1.2417 | .....................................OX..O.O..X.X../....................... | 1.2417
    1.2359 | .....................................OX....OX.XOX./........................ | 1.2359
    1.2303 | .....................................O.....OXOXOX/......................... | 1.2303
    1.2246 | ...........................................OXOXO/.......................... | 1.2246
    1.2190 | ...........................................OXO./........................... | 1.2190
    1.2134 | ...........................................OX.............................. | 1.2134
    1.2078 | ...........................................O............................... | 1.2078
    1.2023 | ........................................................................... | 1.2023

             222222222222222222222222222222222222222222222222222222222222222222222222222
             000000000000000000000000000000000000000000000000000000000000000000000000000
             111111111111111111111111111111111111111111111111111111111111111111111111111
             111111111111111111111111112222222222222222222222222222222333333333333333344
             ...........................................................................
             000000000001111111111111110000000000000000000000011111111000000000000011100
             788888899990000001111112221122233445566666677888900001222123444567778901213
             ...........................................................................
             200011211220111220011231220101212121201112222001100010001002123110112020231
             658801925683489071404504193396436668111288937260415979579417630739120547713

             000100001012111111110111111100112010210001111101101101011111111101011101110
             910501876933613095500253237788652909250001557626626824655375907538165785367
             :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
             550433251023230204310404232105354323532031240033315125241340044324523153453
             000000000000000000000000000000000000000000000000000000000000000000000000000
```

I will not argue that trading opportunities are clearly visible in this chart; I only suggest that you check two trading hypothesis:

**Trade with the primary trend** — buy in a bull market and sell in a bear one.

**Use stop losses**, defined before you enter the market.

- buy above the support level using a stop-order above the high of the previous X column, sell below the resistance line using a sell stop order below the low of the previous O column; use a trailing stop at the pivot level;

**Let your profits grow.**

Close loss-making deals (good deals usually immediately show profit).

- buy when it breaks through the resistance line, sell when it breaks through the support line; set stop loss at the pivot level, set trailing stop at the trend line.

### How to Choose Volume

Pay attention to the above quotations of a master of stock speculation: use stop orders. I prefer to enter the market with a certain volume so that when the Stop Loss order triggers, loss of balance were not more than percent of the balance that I can accept (what Ralph Vince calls optimal F in his The Mathematics of Money Management). The risk of loss per deal is a well optimizable variable (opt\_f in the below code).

Thus, we have a function that places a buy/sell order with the volume calculation mechanism that depends on the acceptable risk per deal:

```
//+------------------------------------------------------------------+
//| The function places an order with a precalculated volume         |
//+------------------------------------------------------------------+
void PlaceOrder()
  {
//--- Variables for calculating the lot
   uint digits_2_lot=(uint)SymbolInfoInteger(symbol,SYMBOL_DIGITS);
   double trade_risk=AccountInfoDouble(ACCOUNT_EQUITY)*opt_f;
   double one_tick_loss_min_lot=SymbolInfoDouble(symbol,SYMBOL_TRADE_TICK_VALUE_LOSS)*SymbolInfoDouble(symbol,SYMBOL_VOLUME_STEP);
//--- Fill out the main fields of the request
   trade_request.magic=magic;
   trade_request.symbol=symbol;
   trade_request.action=TRADE_ACTION_PENDING;
   trade_request.tp=NULL;
   trade_request.comment=NULL;
   trade_request.type_filling=NULL;
   trade_request.stoplimit=NULL;
   trade_request.type_time=NULL;
   trade_request.expiration=NULL;
   if(is_const_lot==true)
     {
      order_vol=SymbolInfoDouble(symbol,SYMBOL_VOLUME_MIN);
     }
   else
     {
      order_vol=trade_risk/(MathAbs(trade_request.price-trade_request.sl)*MathPow(10,digits_2_lot)*one_tick_loss_min_lot)*SymbolInfoDouble(symbol,SYMBOL_VOLUME_STEP);
      order_vol=MathMax(order_vol,SymbolInfoDouble(symbol,SYMBOL_VOLUME_MIN));
      if(SymbolInfoDouble(symbol,SYMBOL_VOLUME_LIMIT)!=0) order_vol=MathMin(order_vol,SymbolInfoDouble(symbol,SYMBOL_VOLUME_LIMIT));
      order_vol=NormalizeDouble(order_vol,(int)MathAbs(MathLog10(SymbolInfoDouble(symbol,SYMBOL_VOLUME_STEP))));
     }
//--- Place an order
   while(order_vol>0)
     {
      trade_request.volume=MathMin(order_vol,SymbolInfoDouble(symbol,SYMBOL_VOLUME_MAX));
      if(!OrderSend(trade_request,trade_result)) Print("Failed to send order #",trade_request.order);
      order_vol=order_vol-SymbolInfoDouble(symbol,SYMBOL_VOLUME_MAX);
     };
   ticket=trade_result.order;
  };
```

### What Optimization Criterion to Use?

Actually there are only two optimization criteria: minimization or drawdown at a given level of return, or maximization of balance for a given level of drawdown. I prefer to optimize by the second criterion:

```
//+------------------------------------------------------------------+
//| Result of strategy run in the testing mode                       |
//+------------------------------------------------------------------+
double OnTester()
  {
   if(TesterStatistics(STAT_EQUITY_DDREL_PERCENT)>(risk*100))
      return(0);
   else
      return(NormalizeDouble(TesterStatistics(STAT_PROFIT),(uint)SymbolInfoInteger(symbol,SYMBOL_DIGITS)));
  };
```

where risk is the drawdown level above which the strategy is unacceptable for me. This is also one of the variables to be optimized.

### When to Re-optimize?

Someone reoptimizes strategies at time intervals (eg, once a week, a month), someone at deal intervals (after 50, 100 deals), someone starts new optimization when the market changes. In fact, the need for re-optimization depends only on the optimization criterion selected in the preceding paragraph. If the drawdown falls below the maximum allowed risk parameter, we reoptimize the strategy; if everything works ok, let it work as it is. Drawdown greater than 10% is unacceptable for me. Thus, if the system gets greater drawdown during operation I reoptimize its.

### Can I Optimize Everything at Once?

Unfortunately, in the strategy tester you cannot optimize external variables for mode "All symbols selected in the Market Watch". Therefore let us choose the market instruments along with other external optimizable variables as follows:

```
//+------------------------------------------------------------------+
//| Enumeration of symbols                                           |
//+------------------------------------------------------------------+
enum  SYMBOLS
  {
   AA=1,
   AIG,
   AXP,
   BA,
   C,
   CAT,
   DD,
   DIS,
   GE,
   HD,
   HON,
   HPQ,
   IBM,
   IP,
   INTC,
   JNJ,
   JPM,
   KO,
   MCD,
   MMM,
   MO,
   MRK,
   MSFT,
   PFE,
   PG,
   QQQ,
   T,
   SPY,
   UTX,
   VZ,
   WMT,
   XOM
  };
//+------------------------------------------------------------------+
//| Symbol selection function                                        |
//+------------------------------------------------------------------+
void  SelectSymbol()
  {
   switch(selected_symbol)
     {
      case  1: symbol="#AA";   break;
      case  2: symbol="#AIG";  break;
      case  3: symbol="#AXP";  break;
      case  4: symbol="#BA";   break;
      case  5: symbol="#C";    break;
      case  6: symbol="#CAT";  break;
      case  7: symbol="#DD";   break;
      case  8: symbol="#DIS";  break;
      case  9: symbol="#GE";   break;
      case 10: symbol="#HD";   break;
      case 11: symbol="#HON";  break;
      case 12: symbol="#HPQ";  break;
      case 13: symbol="#IBM";  break;
      case 14: symbol="#IP";   break;
      case 15: symbol="#INTC"; break;
      case 16: symbol="#JNJ";  break;
      case 17: symbol="#JPM";  break;
      case 18: symbol="#KO";   break;
      case 19: symbol="#MCD";  break;
      case 20: symbol="#MMM";  break;
      case 21: symbol="#MO";   break;
      case 22: symbol="#MRK";  break;
      case 23: symbol="#MSFT"; break;
      case 24: symbol="#PFE";  break;
      case 25: symbol="#PG";   break;
      case 26: symbol="#QQQ";  break;
      case 27: symbol="#T";    break;
      case 28: symbol="#SPY";  break;
      case 29: symbol="#UTX";  break;
      case 30: symbol="#VZ";   break;
      case 31: symbol="#WMT";  break;
      case 32: symbol="#XOM";  break;
      default: symbol="#SPY";  break;
     };
  };
```

If necessary, you can add required instruments in the symbol selection function and the enumeration (the function is called in [OnInit()](https://www.mql5.com/en/docs/basis/function/events#oninit)).

### What Robot Have We Created?

Tick Handler:

```
//+------------------------------------------------------------------+
//| A typical ticks handler OnTick()                                 |
//|     Draw the chart only based on complete bars but first         |
//|     check if it is a new bar.                                    |
//|     If the bar is new and there is a position, check             |
//|     whether we need to move the stop loss,                       |
//|     if the bar is new and no position, check                     |
//|     if we have conditions for opening a deal.                    |
//+------------------------------------------------------------------+
void OnTick()
  {
//--- If the bar is new
   if(IsNewBar()==true)
     {
      RecalcIndicators();
      //--- Tester/optimizer mode?
      if((MQLInfoInteger(MQL_TESTER)==true) || (MQLInfoInteger(MQL_OPTIMIZATION)==true))
        {
         //--- Is it the testing period?
         if(cur_bar_time_dig[0]>begin_of_test)
           {
            //--- If there is an open position on the symbol
            if(PositionSelect(symbol)==true)
               //--- check if we need to move SL; if need, move it
               TrailCondition();
            //--- If there are no positions
            else
            //--- check if we need to open a position; if we need, open it
               TradeCondition();
           }
        }
      else
        {
         //--- if there is an oprn position on the symbol
         if(PositionSelect(symbol)==true)
            //--- check if we need to move SL; if need, move it
            TrailCondition();
         //--- If there are no positions
         else
         //---  check if we need to open a position; if we need, open it
            TradeCondition();
        }

     };
  };
```

For Strategy 1: "buy above the support level using a stop-order above the high of the previous X column, sell below the resistance line using a sell stop order below the low of the previous O column; use a trailing stop at the pivot level\|":

```
//+------------------------------------------------------------------+
//| Function checks trade conditions for opening a deal              |
//+------------------------------------------------------------------+
void TradeCondition()
  {
   if(order_col_number!=column_count)
      //--- Are there any orders on the symbol?
     {
      if(OrdersTotal()>0)
        {
         //--- Delete them!
         for(int loc_count_1=0;loc_count_1<OrdersTotal();loc_count_1++)
           {
            ticket=OrderGetTicket(loc_count_1);
            if(!OrderSelect(ticket)) Print("Failed to select order #",ticket);
            if(OrderGetString(ORDER_SYMBOL)==symbol)
              {
               trade_request.order=ticket;
               trade_request.action=TRADE_ACTION_REMOVE;
               if(!OrderSend(trade_request,trade_result)) Print("Failed to send order #",trade_request.order);
              };
           };
         order_col_number=column_count;
         return;
        }
      else
        {
         order_col_number=column_count;
         return;
        }
     }
   else
      if((MathPow(10,pnf[column_count-1].resist_price)<SymbolInfoDouble(symbol,SYMBOL_ASK)) &&
         (pnf[column_count-1].column_type=='X') &&
         (pnf[column_count-1].max_column_price<=pnf[column_count-3].max_column_price))
        {
         //--- Conditions for BUY met; let's see if there are any pending Buy orders for the symbol with the price we need?
         trade_request.price=NormalizeDouble(MathPow(10,pnf[column_count-3].max_column_price+double_box),digit_2_orders);
         trade_request.sl=NormalizeDouble(MathPow(10,pnf[column_count-3].max_column_price-(reverse-1)*double_box),digit_2_orders);
         trade_request.type=ORDER_TYPE_BUY_STOP;
         if(OrderSelect(ticket)==false)
            //--- No pending orders - place an order
           {
            PlaceOrder();
            order_col_number=column_count;
           }
         else
         //--- If there is a pending order
           {
            //--- what is the type and price of the pending order?
            if((OrderGetInteger(ORDER_TYPE)==ORDER_TYPE_SELL_STOP) ||
               ((OrderGetInteger(ORDER_TYPE)==ORDER_TYPE_BUY_STOP) && (OrderGetDouble(ORDER_PRICE_OPEN)!=trade_request.price)))
              {
               //--- The wrong type or the price differs - close the order
               trade_request.order=ticket;
               trade_request.action=TRADE_ACTION_REMOVE;
               if(!OrderSend(trade_request,trade_result)) Print("Failed to send order #",trade_request.order);
               //--- open with the desired price
               PlaceOrder();
               order_col_number=column_count;
              };
           };
         return;
        }
   else
      if((MathPow(10,pnf[column_count-1].resist_price)>SymbolInfoDouble(symbol,SYMBOL_ASK)) &&
         (pnf[column_count-1].column_type=='O') &&
         (pnf[column_count-1].min_column_price>=pnf[column_count-3].min_column_price))
        {
         //--- Conditions for SELL met; let's see if there are any pending Sell orders for the symbol with the price we need?
         trade_request.price=NormalizeDouble(MathPow(10,pnf[column_count-3].min_column_price-double_box),digit_2_orders);
         trade_request.sl=NormalizeDouble(MathPow(10,pnf[column_count-3].min_column_price+(reverse-1)*double_box),digit_2_orders);
         trade_request.type=ORDER_TYPE_SELL_STOP;
         if(OrderSelect(ticket)==false)
            //--- No pending orders, place an order
           {
            PlaceOrder();
            order_col_number=column_count;
           }
         else
         //--- or there is a pending order
           {
            //--- what is the type and price of the pending order?
            if((OrderGetInteger(ORDER_TYPE)==ORDER_TYPE_BUY_STOP) ||
               ((OrderGetInteger(ORDER_TYPE)==ORDER_TYPE_SELL_STOP) && (OrderGetDouble(ORDER_PRICE_OPEN)!=trade_request.price)))
              {
               //--- The wrong type or the price differs - close the order
               trade_request.order=ticket;
               trade_request.action=TRADE_ACTION_REMOVE;
               if(!OrderSend(trade_request,trade_result)) Print("Failed to send order #",trade_request.order);
               //--- and open with the desired price
               PlaceOrder();
               order_col_number=column_count;
              };
           };
         return;
        }
   else
      return;
  };
//+------------------------------------------------------------------+
//| The function checks conditions for moving Stop Loss              |
//+------------------------------------------------------------------+
void TrailCondition()
  {
   if(PositionGetInteger(POSITION_TYPE)==POSITION_TYPE_BUY)
      trade_request.sl=NormalizeDouble(MathPow(10,pnf[column_count-1].max_column_price-reverse*double_box),digit_2_orders);
   else
      trade_request.sl=NormalizeDouble(MathPow(10,pnf[column_count-1].min_column_price+reverse*double_box),digit_2_orders);
   if(PositionGetDouble(POSITION_SL)!=trade_request.sl)
      PlaceTrailOrder();
  };
```

For Strategy 2 "buy at the breakthrough of the resistance line, sell at the breakthrough of the support line; set stop loss at the pivot level, set trailing stop at the trend line":

```
//+------------------------------------------------------------------+
//| The function checks trade conditions for opening a deal          |
//+------------------------------------------------------------------+
void TradeCondition()
  {
   if(order_col_number!=column_count)
      //--- Are there any orders for the symbol?
     {
      if(OrdersTotal()>0)
        {
         //--- Delete them!
         for(int loc_count_1=0;loc_count_1<OrdersTotal();loc_count_1++)
           {
            ticket=OrderGetTicket(loc_count_1);
            if(!OrderSelect(ticket)) Print("Failed to select order #",ticket);
            if(OrderGetString(ORDER_SYMBOL)==symbol)
              {
               trade_request.order=ticket;
               trade_request.action=TRADE_ACTION_REMOVE;
               if(!OrderSend(trade_request,trade_result)) Print("Failed to send order #",trade_request.order);
              };
           };
         order_col_number=column_count;
         return;
        }
      else
        {
         order_col_number=column_count;
         return;
        }
     }
   else
   if(MathPow(10,pnf[column_count-1].resist_price)>SymbolInfoDouble(symbol,SYMBOL_ASK))
     {
      //--- Conditions for BUY met; let's see if there are any pending Buy orders for the symbol with the price we need?
      trade_request.price=NormalizeDouble(MathPow(10,pnf[column_count-1].resist_price),digit_2_orders);
      trade_request.sl=NormalizeDouble(MathPow(10,pnf[column_count-1].resist_price-(reverse-1)*double_box),digit_2_orders);
      trade_request.type=ORDER_TYPE_BUY_STOP;
      if(OrderSelect(ticket)==false)
         //--- No pending orders - place an order
        {
         PlaceOrder();
         order_col_number=column_count;
        }
      else
      //--- or there is a pending order
        {
         //--- what is the type and price of the pending order?
         if((OrderGetInteger(ORDER_TYPE)==ORDER_TYPE_SELL_STOP) ||
            ((OrderGetInteger(ORDER_TYPE)==ORDER_TYPE_BUY_STOP) && (OrderGetDouble(ORDER_PRICE_OPEN)!=trade_request.price)))
           {
            //--- The wrong type or the price differs - close the order
            trade_request.order=ticket;
            trade_request.action=TRADE_ACTION_REMOVE;
            if(!OrderSend(trade_request,trade_result)) Print("Failed to send order #",trade_request.order);
            //--- open with the desired price
            PlaceOrder();
            order_col_number=column_count;
           };
        };
      return;
     }
   else
   if(MathPow(10,pnf[column_count-1].resist_price)<SymbolInfoDouble(symbol,SYMBOL_ASK))
     {
      //--- Conditions for SELL met; let's see if there are any pending Sell orders for the symbol with the price we need?
      trade_request.price=NormalizeDouble(MathPow(10,pnf[column_count-1].supp_price),digit_2_orders);
      trade_request.sl=NormalizeDouble(MathPow(10,pnf[column_count-1].supp_price+(reverse-1)*double_box),digit_2_orders);
      trade_request.type=ORDER_TYPE_SELL_STOP;
      if(OrderSelect(ticket)==false)
         //--- No pending orders - place an order
        {
         PlaceOrder();
         order_col_number=column_count;
        }
      else
      //--- If there is a pending order
        {
         //--- what is the type and price of the pending order?
         if((OrderGetInteger(ORDER_TYPE)==ORDER_TYPE_BUY_STOP) ||
            ((OrderGetInteger(ORDER_TYPE)==ORDER_TYPE_SELL_STOP) && (OrderGetDouble(ORDER_PRICE_OPEN)!=trade_request.price)))
           {
            //--- The wrong type or the price differs - close the order
            trade_request.order=ticket;
            trade_request.action=TRADE_ACTION_REMOVE;
            if(!OrderSend(trade_request,trade_result)) Print("Failed to send order #",trade_request.order);
            //--- open with the desired price
            PlaceOrder();
            order_col_number=column_count;
           };
        };
      return;
     }
   else
      return;
  };
//+------------------------------------------------------------------+
//| The function checks conditions for moving Stop Loss              |
//+------------------------------------------------------------------+
void TrailCondition()
  {
   if(PositionGetInteger(POSITION_TYPE)==POSITION_TYPE_BUY)
      trade_request.sl=NormalizeDouble(MathMax(SymbolInfoDouble(symbol,SYMBOL_ASK),MathPow(10,pnf[column_count-1].max_column_price-reverse*double_box)),digit_2_orders);
   else
      trade_request.sl=NormalizeDouble(MathMin(SymbolInfoDouble(symbol,SYMBOL_BID),MathPow(10,pnf[column_count-1].min_column_price+reverse*double_box)),digit_2_orders);
   if(PositionGetDouble(POSITION_SL)!=trade_request.sl)
      PlaceTrailOrder();
  };
```

Dear reader, please note a few things.

- Prices of market instruments vary in a fairly wide range, from cents to tens of thousands (example - stocks and CFDs on Japanese stock exchanges). Therefore, I use the price logarithm for the point and figure chart to avoid setting values of one to tens of thousands of pips for the box size.
- Indexing of the array of the point and figure chart starts from zero, so the index of the last column has is equal to the number of columns minus one.
- The value of the support line, if used, is greater than -10.0, but less than the price logarithm of the price. If no support (or resistance) line is used, its value in the chart array is -10.0. Therefore, the conditions of break through the support/resistance lines are written in the same form as in the code of Strategy 2 above.

See the code of auxiliary functions in attachments below. The code of the chart is too large, so i do not write it in the article; see it with comments in attachments below.

### Results of EA Trading

I have prepared two sets of market symbols in files symbol\_list\_1.mhq and symbol\_list\_2.mhq for optimization; they include currency pairs and stock CFDs from the Dow index.

Setup Window:

![Testing options](https://c.mql5.com/2/6/Figure1_Testing_Options.png)

In the first case, the setup window in the strategy tester looks as follows:

![Optimization parameters](https://c.mql5.com/2/6/Figure2_Testing_parameters.png)

Note the Testing Start line. The robot requires at least a few chart columns for analysis and decision making; and when you set box size equal to 50 pips or more, a one year history is often not enough even for one column. Therefore, for charts with box size of 50 pips or more use the interval of about three years or more from the robot operation start, an set the robot operation start in the Testing Start parameter in the Setup Window. In our example, for testing with the box size of 100 pips since 01.01.2012 specify interval since 01.01.2009 in the Settings tab; set interval since 01.01.2012 in the Parameters tab.

The false value of parameter "Trade minimum lot?" indicates that the lot size depends on the balance and the "Risk per Trade, %" variable (in this case 1% per trade, but it can also be optimized). "Max Drawdown, %" is the optimization criterion in the [OnTester()](https://www.mql5.com/en/docs/basis/function/events#ontester) function. In this article I use only two variables for optimization: Symbol and Box size in pips.

Optimization period: 2012-2013. The EA runs best on the EURUSD chart, since the symbol provides the best tick coverage. The table below contains a full report for testing on various currency pairs with box size 10 based on the first strategy:

| Pass | Result | Profit | Expected Payoff | Profit Factor | Recovery Factor | Sharpe Ratio | Custom | Equity DD % | Trades | selected\_symbol | box |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0,00 | 0,00 | -1 002,12 | -18,91 | 0,54 | -0,79 | -0,24 | 0,00 | 12,67 | 53,00 | AUDCAD | 10,00 |
| 1,00 | 886,56 | 886,56 | 14,53 | 1,40 | 1,52 | 0,13 | 886,56 | 5,76 | 61,00 | AUDCHF | 10,00 |
| 2,00 | 0,00 | -1 451,63 | -10,60 | 0,77 | -0,70 | -0,09 | 0,00 | 19,92 | 137,00 | AUDJPY | 10,00 |
| 3,00 | -647,66 | -647,66 | -17,50 | 0,57 | -0,68 | -0,24 | -647,66 | 9,46 | 37,00 | AUDNZD | 10,00 |
| 4,00 | -269,22 | -269,22 | -3,17 | 0,92 | -0,26 | -0,03 | -269,22 | 9,78 | 85,00 | AUDUSD | 10,00 |
| 5,00 | 0,00 | -811,44 | -13,52 | 0,72 | -0,64 | -0,14 | 0,00 | 12,20 | 60,00 | CADCHF | 10,00 |
| 6,00 | 0,00 | 1 686,34 | 16,53 | 1,36 | 1,17 | 0,12 | 0,00 | 11,78 | 102,00 | CHFJPY | 10,00 |
| 7,00 | 356,68 | 356,68 | 5,66 | 1,13 | 0,40 | 0,06 | 356,68 | 8,04 | 63,00 | EURAUD | 10,00 |
| 8,00 | 0,00 | -1 437,91 | -25,68 | 0,53 | -0,92 | -0,25 | 0,00 | 15,47 | 56,00 | EURCAD | 10,00 |
| 9,00 | 0,00 | -886,66 | -46,67 | 0,34 | -0,74 | -0,46 | 0,00 | 11,56 | 19,00 | EURCHF | 10,00 |
| 10,00 | 0,00 | -789,59 | -21,93 | 0,54 | -0,75 | -0,26 | 0,00 | 10,34 | 36,00 | EURGBP | 10,00 |
| 11,00 | 0,00 | 3 074,86 | 28,47 | 1,62 | 1,72 | 0,20 | 0,00 | 12,67 | 108,00 | EURJPY | 10,00 |
| 12,00 | 0,00 | -1 621,85 | -19,78 | 0,55 | -0,97 | -0,25 | 0,00 | 16,75 | 82,00 | EURNZD | 10,00 |
| 13,00 | 152,73 | 152,73 | 2,88 | 1,07 | 0,21 | 0,03 | 152,73 | 6,90 | 53,00 | EURUSD | 10,00 |
| 14,00 | 0,00 | -1 058,85 | -14,50 | 0,65 | -0,66 | -0,16 | 0,00 | 15,87 | 73,00 | GBPAUD | 10,00 |
| 15,00 | 0,00 | -1 343,47 | -25,35 | 0,43 | -0,64 | -0,34 | 0,00 | 20,90 | 53,00 | GBPCAD | 10,00 |
| 16,00 | 0,00 | -2 607,22 | -44,19 | 0,27 | -0,95 | -0,59 | 0,00 | 27,15 | 59,00 | GBPCHF | 10,00 |
| 17,00 | 0,00 | 1 160,54 | 11,72 | 1,27 | 0,81 | 0,10 | 0,00 | 12,30 | 99,00 | GBPJPY | 10,00 |
| 18,00 | 0,00 | -1 249,91 | -14,70 | 0,69 | -0,85 | -0,15 | 0,00 | 14,41 | 85,00 | GBPNZD | 10,00 |
| 19,00 | 208,94 | 208,94 | 5,36 | 1,12 | 0,25 | 0,05 | 208,94 | 7,81 | 39,00 | GBPUSD | 10,00 |
| 20,00 | 0,00 | -2 137,68 | -21,17 | 0,53 | -0,79 | -0,24 | 0,00 | 25,62 | 101,00 | NZDUSD | 10,00 |
| 21,00 | 0,00 | -1 766,80 | -38,41 | 0,30 | -0,97 | -0,53 | 0,00 | 18,10 | 46,00 | USDCAD | 10,00 |
| 22,00 | -824,69 | -824,69 | -11,95 | 0,73 | -0,90 | -0,13 | -824,69 | 9,11 | 69,00 | USDCHF | 10,00 |
| 23,00 | 2 166,53 | 2 166,53 | 26,10 | 1,58 | 2,40 | 0,18 | 2 166,53 | 7,13 | 83,00 | USDJPY | 10,00 |
|  | 2 029,87 | -10 213,52 |  |  |  |  |  | 13,40 | 1 659,00 |  |  |

Here is a summary table for various symbols and box size values:

| Strategy | Symbols | Box size | Trades | Equity DD % | Profit | Result | Expected Balance |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Currencies | 10 | 1 659 | September | -10 214 | 2 030 | 2 030 |
| 1 | Currencies | 20 | 400 | 5 | 1 638 | 2 484 | 2 484 |
| 1 | Stocks | 50 | 350 | 60 | 7 599 | 7 599 | 15 199 |
| 1 | Stocks | 100 | 81 | 2 | 4 415 | 4 415 | 17 659 |
| 2 | Currencies | 10 | 338 | 20 | -4 055 | 138 | 138 |
| 2 | Currencies | 20 | 116 | 8 | 4 687 | 3 986 | 3 986 |
| 2 | Stocks | 50 | 65 | 6 | 6 770 | 9 244 | 9 244 |
| 2 | Stocks | 100 | 12 | 1 | -332 | -332 | -5 315 |

What do we see?

There are fools, who are always doing wrong.

And there are the fools of Wall Street who believe you should always trade.

There is no one in the world who always has all the required information to buy or sell stocks and do it quite reasonably.

The conclusion may seem strange to you: your deposit is more likely to be higher with fewer trades. If two years ago we let our EA trade stocks with box size of 100 and the risk per trade of 1%, the EA would make only 81 deals since then (an average 1.25 deals per symbol per year) our deposit would have grown by 44%, while the average equity drawdown would have been slightly above 2%. Accepting possible drawdon of 10%, we could risk 4% per deal, and our deposit would have grown by 177% for two years, i.e. 90% per annum in U.S. dollars!

### Epilogue

The price is never too high to start buying, and is never too low to begin selling.

It is not thinking that makes the big money. It is sitting.

The above described strategies can be modified and can show an even greater return with the drawdown at no higher than 10%. Do not try to trade too often, better find a broker that provides not just a "standard set" of symbols of two dozen currency pairs and three dozen stocks, but at least three or four hundred symbols (stocks, futures). More likely the symbols will not be correlated and your deposit will be more secure. One more remark - stocks show better results than currency pairs.

### P.S. (an advertising remark)

My [PnF Chartist](https://www.mql5.com/en/market/product/3702 "Script \"PnF Chartist\"") script is available on the Market. It draws point and figure charts in text files using quotes provided by MT4, MT5 or Yahoo finance. Use it for visual search of price patterns, since there is no better tester/optimizer than your brain. As soon as you find patterns, use EA templates from this article to find proof of your ideas.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/910](https://www.mql5.com/ru/articles/910)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/910.zip "Download all attachments in the single ZIP archive")

[licence.txt](https://www.mql5.com/en/articles/download/910/licence.txt "Download licence.txt")(1.05 KB)

[strategy\_one.mq5](https://www.mql5.com/en/articles/download/910/strategy_one.mq5 "Download strategy_one.mq5")(37.87 KB)

[strategy\_two.mq5](https://www.mql5.com/en/articles/download/910/strategy_two.mq5 "Download strategy_two.mq5")(37.39 KB)

[symbol\_list\_1.mqh](https://www.mql5.com/en/articles/download/910/symbol_list_1.mqh "Download symbol_list_1.mqh")(2.43 KB)

[symbol\_list\_2.mqh](https://www.mql5.com/en/articles/download/910/symbol_list_2.mqh "Download symbol_list_2.mqh")(2.17 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [The Last Crusade](https://www.mql5.com/en/articles/368)
- [Trademinator 3: Rise of the Trading Machines](https://www.mql5.com/en/articles/350)
- [Dr. Tradelove or How I Stopped Worrying and Created a Self-Training Expert Advisor](https://www.mql5.com/en/articles/334)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/23449)**
(9)


![Roman Zamozhnyy](https://c.mql5.com/avatar/avatar_na2.png)

**[Roman Zamozhnyy](https://www.mql5.com/en/users/rich)**
\|
24 Apr 2014 at 15:43

Reworked for MT4 [Expert Advisor](https://www.mql5.com/en/market/mt5/ "A Market of Applications for the MetaTrader 5 and MetaTrader 4").


![Roman Zamozhnyy](https://c.mql5.com/avatar/avatar_na2.png)

**[Roman Zamozhnyy](https://www.mql5.com/en/users/rich)**
\|
24 Apr 2014 at 15:46

Redesigned for MT4 expert.


![Alexandr Gershkevich](https://c.mql5.com/avatar/2014/3/53203192-FF68.JPG)

**[Alexandr Gershkevich](https://www.mql5.com/en/users/avg660018)**
\|
29 Apr 2014 at 07:09

Enjoyed the thoughtfulness, detail and "probabilistic" hope for success. Thank you.


![Andriy Sydoruk](https://c.mql5.com/avatar/2020/4/5EA99895-3CC2.jpg)

**[Andriy Sydoruk](https://www.mql5.com/en/users/andreys)**
\|
7 Jul 2014 at 23:57

Very interesting!


![fiacko](https://c.mql5.com/avatar/avatar_na2.png)

**[fiacko](https://www.mql5.com/en/users/fiacko)**
\|
17 Mar 2015 at 09:42

**Rich:**

Reworked for MT4 Expert Advisor.

Good afternoon. In [testing mode](https://www.metatrader5.com/en/terminal/help/algotrading/testing "Tick generation modes in MetaTrader 5 Client Terminal"), all robots crash with errors. This applies to both mt4 and mt5

zero divide in 'strategy\_two.mq5' (348,99)

or

array out of range in 'strategy\_one.mq5' (438,32)

I can't understand what the reason is, please help me.

![MQL5 Cookbook: Development of a Multi-Symbol Indicator to Analyze Price Divergence](https://c.mql5.com/2/0/avatar__11.png)[MQL5 Cookbook: Development of a Multi-Symbol Indicator to Analyze Price Divergence](https://www.mql5.com/en/articles/754)

In this article, we will consider the development of a multi-symbol indicator to analyze price divergence in a specified period of time. The core topics have been already discussed in the previous article on the programming of multi-currency indicators "MQL5 Cookbook: Developing a Multi-Symbol Volatility Indicator in MQL5". So this time we will dwell only on those new features and functions that have been changed dramatically. If you are new to the programming of multi-currency indicators, I recommend you to first read the previous article.

![SQL and MQL5: Working with SQLite Database](https://c.mql5.com/2/0/MQL5_SQLite_avatar.png)[SQL and MQL5: Working with SQLite Database](https://www.mql5.com/en/articles/862)

This article is intended for developers who would be interested in using SQL in their projects. It explains the functionality and advantages of SQLite. The article does not require special knowledge of SQLite functions, yet minimum understanding of SQL would be beneficial.

![Why Is It Important to Update MetaTrader 4 to the Latest Build by August 1?](https://c.mql5.com/2/13/1176_14.png)[Why Is It Important to Update MetaTrader 4 to the Latest Build by August 1?](https://www.mql5.com/en/articles/1392)

From August 1, 2014, MetaTrader 4 desktop terminals older than build 600 will no longer be supported. However, many traders still work with outdated versions and are unaware of the updated platform's features. We have put a lot of effort into development and would like to move on with traders and abandon the older builds. In this article, we will describe the advantages of the new MetaTrader 4.

![Common Errors in MQL4 Programs and How to Avoid Them](https://c.mql5.com/2/13/1152_84.png)[Common Errors in MQL4 Programs and How to Avoid Them](https://www.mql5.com/en/articles/1391)

To avoid critical completion of programs, the previous version compiler handled many errors in the runtime environment. For example, division by zero or array out of range are critical errors and usually lead to program crash. The new compiler can detect actual or potential sources of errors and improve code quality. In this article, we discuss possible errors that can be detected during compilation of old programs and see how to fix them.

[![](https://www.mql5.com/ff/sh/qv94j0cd8n2n55z9z2/01.png)![](https://www.mql5.com/ff/sh/qv94j0cd8n2n55z9z2/02.png)Boost your trading experienceRead our book "MQL5 Programming for Traders"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/book%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.algobook%26utm_content=visit.page%26utm_campaign=algobook.promo.04.2024&a=heclgjpfbvfghpmyaciuaesdtswflupo&s=4255fbe1b8cbc4d1b40afbaebf4235e5ace8b5103cba60d996897a03d588556f&uid=&ref=https://www.mql5.com/en/articles/910&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5069357209852249016)

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