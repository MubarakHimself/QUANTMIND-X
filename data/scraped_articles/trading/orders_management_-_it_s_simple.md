---
title: Orders Management - It's Simple
url: https://www.mql5.com/en/articles/1404
categories: Trading, Trading Systems, Expert Advisors
relevance_score: 0
scraped_at: 2026-01-24T13:42:58.679969
---

[![](https://www.mql5.com/ff/sh/bhdtjfb1zry09943z2/267b575d2182c180804d340af38ce02c.jpg)\\
Trade from your iPhone or Android device\\
\\
You only need an internet connection to use the new powerful MetaTrader 5 Web terminal\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=wtigumvtenarnsocpyfoqnanxrilnbxx&s=ec8c539e52b83881ff2d16eaff6913b25803952eb277cac55f670a102b2edc1f&uid=&ref=https://www.mql5.com/en/articles/1404&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5083063017429537841)

MetaTrader 4 / Trading


### 1\. Introduction

Every trading expert has a block controlling open positions. It is the search among all orders within the loop, choosing of "its own" position by symbol and MagicNumber value, and then modifying or closing thereof. These blocks look very similarly and often have the same functions. This is why this repeating part of the code can be moved from the expert to the function - this will significantly simplify writing of experts and make the expert codes much more space-saving.

First of all let us divide the task into three stages that are different in complicity and functions – these three stages will correlate with three types of experts:

- Experts that can open only one position at a time
- Experts that can open one position of each type at a time (for example, one long and one short position)
- Expert that can open any amount of positions simultaneously

### 2\. One Position

There are many strategies that use only one open position. Their controlling blocks are rather simple, but writing of them still consumes time and attention.

Let us take a simple expert, the opening signal of which is
intersection of MACD lines (signal line and basic line), and simplify
its controlling block. This is how it looked before:

```
extern int  _MagicNumber = 1122;

int start()
  {
//---- Memorize indicator values for further analysis
   double MACD_1=iMACD(Symbol(),0,12,26,9,PRICE_CLOSE,
                       MODE_MAIN,1);
   double MACD_2=iMACD(Symbol(),0,12,26,9,PRICE_CLOSE,
                       MODE_MAIN,2);

   int _GetLastError=0,_OrdersTotal=OrdersTotal();
//---- search among all open positions
   for(int z=_OrdersTotal-1; z>=0; z --)
     {
      // if an error occurs at finding the position, go to the next one
      if(!OrderSelect(z,SELECT_BY_POS))
        {
         _GetLastError=GetLastError();
         Print("OrderSelect( ",z,", SELECT_BY_POS ) - Error #",
               _GetLastError);
         continue;
        }

      // if the position was opened not for the current symbol,
      // skip it
      if(OrderSymbol()!=Symbol()) continue;

      // if MagicNumber does not equal to _MagicNumber, skip
      // this position
      if(OrderMagicNumber()!=_MagicNumber) continue;

      //---- if BUY position is opened,
      if(OrderType()==OP_BUY)
        {
         //---- if MACD meets zero line top-down,
         if(NormalizeDouble(MACD_1,Digits +1)<  0.0 &&
            NormalizeDouble(MACD_2,Digits+1)>=0.0)
           {
            //---- close the position
            if(!OrderClose(OrderTicket(),OrderLots(),
               Bid,5,Green))
              {
               _GetLastError=GetLastError();
               Alert("Error OrderClose № ",_GetLastError);
               return(-1);
              }
           }
         // if signal has not changed, exit: it's too early for
         // opening a new position
         else
           { return(0); }
        }
      //---- if SELL position is opened,
      if(OrderType()==OP_SELL)
        {
         //---- if MACD meets zero line bottom-up,
         if(NormalizeDouble(MACD_1,Digits+1) >  0.0 &&
            NormalizeDouble(MACD_2,Digits+1)<=0.0)
           {
            //---- close the position
            if(!OrderClose(OrderTicket(),OrderLots(),
               Ask,5,Red))
              {
               _GetLastError=GetLastError();
               Alert("Error OrderClose № ",_GetLastError);
               return(-1);
              }
           }
         // if signal has not changed, exit: it's too early to open
         // a new position
         else return(0);
        }
     }

//+------------------------------------------------------------------+
//| if execution has reached this point, this means no open position |
//| check whether it is possible to open a position                  |
//+------------------------------------------------------------------+

//---- if MACD meets zero line bottom-up,
   if(NormalizeDouble(MACD_1,Digits+1)>0.0 &&
      NormalizeDouble(MACD_2,Digits+1)<=0.0)
     {
      //---- open a BUY position
      if(OrderSend(Symbol(),OP_BUY,0.1,Ask,5,0.0,0.0,
         "MACD_test",_MagicNumber,0,Green)<0)
        {
         _GetLastError=GetLastError();
         Alert("Error OrderSend № ",_GetLastError);
         return(-1);
        }
      return(0);
     }
//---- if MACD meets zero line top-down,
   if(NormalizeDouble(MACD_1,Digits+1)<0.0 &&
      NormalizeDouble(MACD_2,Digits+1)>=0.0)
     {
      //---- open a SELL position
      if(OrderSend(Symbol(),OP_SELL,0.1,Bid,5,0.0,0.0,
         "MACD_test",
         _MagicNumber,0,Red)<0)
        {
         _GetLastError=GetLastError();
         Alert("Error OrderSend № ",_GetLastError);
         return(-1);
        }
      return(0);
     }

   return(0);
  }
```

Now we have to write a function that would replace the block controlling positions. The function must search among all orders, find the necessary one, and memorize all its characteristics into global variables. It will be look like this:

```
int _Ticket = 0, _Type = 0; double _Lots = 0.0,
_OpenPrice=0.0,_StopLoss=0.0;
double _TakeProfit=0.0; datetime _OpenTime=-1;
double _Profit=0.0,_Swap=0.0;
double _Commission=0.0; string _Comment="";
datetime _Expiration=-1;

void OneOrderInit(int magic)
  {
   int _GetLastError,_OrdersTotal=OrdersTotal();

   _Ticket=0; _Type=0; _Lots=0.0; _OpenPrice=0.0;
   _StopLoss=0.0;
   _TakeProfit=0.0; _OpenTime=-1; _Profit=0.0;
   _Swap=0.0;
   _Commission=0.0; _Comment=""; _Expiration=-1;

   for(int z=_OrdersTotal-1; z>=0; z --)
     {
      if(!OrderSelect(z,SELECT_BY_POS))
        {
         _GetLastError=GetLastError();
         Print("OrderSelect( ",z,", SELECT_BY_POS ) - Error #",_GetLastError);
         continue;
        }
      if(OrderMagicNumber()==magic && OrderSymbol()==
         Symbol())
        {
         _Ticket     = OrderTicket();
         _Type       = OrderType();
         _Lots       = NormalizeDouble( OrderLots(), 1 );
         _OpenPrice  = NormalizeDouble( OrderOpenPrice(), Digits);
         _StopLoss   = NormalizeDouble(OrderStopLoss(),Digits);
         _TakeProfit = NormalizeDouble( OrderTakeProfit(), Digits);
         _OpenTime   = OrderOpenTime();
         _Profit     = NormalizeDouble( OrderProfit(), 2 );
         _Swap       = NormalizeDouble( OrderSwap(), 2 );
         _Commission = NormalizeDouble( OrderCommission(), 2 );
         _Comment    = OrderComment();
         _Expiration = OrderExpiration();
         return;
        }
     }
  }
```

As you can see, it's rather easy: there are 11 variables, each stores the value of one position characteristic (ticket #, type, lot size, etc.). These variables are zeroized when the function starts. This is necessary since the variables are declared at global level and are not zeroized at the function call, but we do not need information of the preceding tick, all data must be recent. Then all open positions are searched among in the standard way and, in case the symbol and the MagicNumber value coincide with those needed, the characteristics are memorized in the corresponding variables.

Now let us attach this function to our expert advisor:

```
extern int  _MagicNumber = 1122;

#include <OneOrderControl.mq4>

int start()
{
    int _GetLastError = 0;

// Memorize parameters of the open position (if available)
    OneOrderInit( _MagicNumber );

    //---- Memorize indicator values for further analysis
    double MACD_1 = iMACD(Symbol(), 0, 12, 26, 9, PRICE_CLOSE,
                          MODE_MAIN, 1 );
    double MACD_2 = iMACD(Symbol(), 0, 12, 26, 9, PRICE_CLOSE,
                          MODE_MAIN, 2 );

    // Now, instead of searching in positions, just see whether
    // there is an open position:
    if ( _Ticket > 0 )
    {
        //---- if a BUY position is opened,
        if ( _Type == OP_BUY )
        {
            //---- if MACD meets zero line top-down,
            if(NormalizeDouble( MACD_1, Digits + 1 ) <  0.0 &&
               NormalizeDouble( MACD_2, Digits + 1 ) >= 0.0)
            {
                //---- close position
                if(!OrderClose( _Ticket, _Lots, Bid, 5, Green))
                {
                  _GetLastError = GetLastError();
                  Alert( "Error OrderClose № ", _GetLastError);
                  return(-1);
                }
            }
            // if signal has not changed, exit: it's too early
            // to open a new position
            else return(0);
        }
        //---- if a SELL position is opened,
        if ( _Type == OP_SELL )
        {
            //---- if MACD meets zero line bottom-up,
            if(NormalizeDouble( MACD_1, Digits + 1 ) >  0.0 &&
               NormalizeDouble( MACD_2, Digits + 1 ) <= 0.0)
            {
                //---- close the position
                if(!OrderClose( _Ticket, _Lots, Ask, 5, Red))
                {
                    _GetLastError = GetLastError();
                    Alert( "Error OrderClose № ", _GetLastError);
                    return(-1);
                }
            }
            // if signal has not changed, exit: it's too early
            // to open a new position
            else return(0);
        }
    }
    // if there is no position opened by the expert
    // ( _Ticket == 0 )
    // if MACD meets zero line bottom-up,
    if(NormalizeDouble( MACD_1, Digits + 1 ) >  0.0 &&
       NormalizeDouble( MACD_2, Digits + 1 ) <= 0.0)
    {
        //---- open a BUY position
        if(OrderSend(Symbol(), OP_BUY, 0.1, Ask, 5, 0.0, 0.0,
           "CrossMACD", _MagicNumber, 0, Green ) < 0)
        {
            _GetLastError = GetLastError();
            Alert( "Error OrderSend № ", _GetLastError );
            return(-1);
        }
        return(0);
    }
    //---- if MACD meets zero line top-down,
    if ( NormalizeDouble( MACD_1, Digits + 1 ) <  0.0 &&
          NormalizeDouble( MACD_2, Digits + 1 ) >= 0.0    )
    {
        //---- open a SELL position
        if(OrderSend(Symbol(), OP_SELL, 0.1, Bid, 5, 0.0, 0.0,
           "CrossMACD",
              _MagicNumber, 0, Red ) < 0 )
        {
            _GetLastError = GetLastError();
            Alert( "Error OrderSend № ", _GetLastError );
            return(-1);
        }
        return(0);
    }

    return(0);
}
```

As you can see, the expert code is now much more compact and human-readable. This is the simplest case.

Now let us solve the next task.

### 3\. One Position of Each Type

We need a more complicated expert to implement the other function. It must open a number of positions of different types and work with them. Below is the expert's algorithm:

- when launched, the expert must place two pending orders: BuyStop at the level of Ask+20 points and a SellStop at the level of Bid+20 points;
- if one of the orders triggers, another must be deleted;
- the open position must be accompanied by the Trailing Stop; and

- after the position has been closed by StopLoss or TakeProfit, go to the start again, i.e., place two pending orders again.

The expert code is given below:

```
extern int    _MagicNumber = 1123;

extern double Lot          = 0.1;
extern int    StopLoss     = 60;
// distance to StopLoss in points (0 - disable)
extern int    TakeProfit=100;
// distance to TakeProfit in points (0 - disable)
extern int    TrailingStop=50;
// Trailing Stop in points (0 - disable)

extern int    Luft=20;
// distance to the level at which the pending order was placed

int start()
  {
// Variables, in which tickets of orders of each type will be
// memorized
   int BuyStopOrder=0,SellStopOrder=0,BuyOrder=0,SellOrder=0;
   int _GetLastError=0,_OrdersTotal=OrdersTotal();
// search in all open positions and memorize, positions of which
// type have already been opened:
   for(int z=_OrdersTotal-1; z>=0; z --)
     {
      // if an error occurs at searching for a position, go
      // to the next one
      if(!OrderSelect(z,SELECT_BY_POS))
        {
         _GetLastError=GetLastError();
         Print("OrderSelect(",z,", SELECT_BY_POS) - Error #",_GetLastError);
         continue;
        }

      // if the position was opened not for the current symbol, skip it
      if(OrderSymbol()!=Symbol()) continue;

      // if the MagicNumber is not equal to _MagicNumber, skip this
      // position
      if(OrderMagicNumber()!=_MagicNumber) continue;

      // depending on the position type, change value of the
      // variable:
      switch(OrderType())
        {
         case OP_BUY:      BuyOrder      = OrderTicket(); break;
         case OP_SELL:     SellOrder     = OrderTicket(); break;
         case OP_BUYSTOP:  BuyStopOrder  = OrderTicket(); break;
         case OP_SELLSTOP: SellStopOrder = OrderTicket(); break;
        }
     }

//---- If we have both pending orders, quit,
//---- we have to wait until one of them triggers
   if( BuyStopOrder > 0 && SellStopOrder > 0 ) return(0);

// search in all open positions for the second time - now
// we will work with them:
   _OrdersTotal=OrdersTotal();
   for(z=_OrdersTotal-1; z>=0; z --)
     {
      // if an error occurs in searching a position, go to
      // the next one
      if(!OrderSelect(z,SELECT_BY_POS))
        {
         _GetLastError=GetLastError();
         Print("OrderSelect(",z,", SELECT_BY_POS) - Error #",_GetLastError);
         continue;
        }

      // if the position was opened not for the current symbol,
      // skip it
      if(OrderSymbol()!=Symbol()) continue;

      // if the MagicNumber does not equal to _MagicNumber,
      // skip this position
      if(OrderMagicNumber()!=_MagicNumber) continue;

      // depending on the position type, change the variable
      // value:
      switch(OrderType())
        {
         //---- if there is an open BUY position,
         case OP_BUY:
           {
            // if the SellStop order has not been deleted
            // yet, delete it:
            if(SellStopOrder>0)
              {
               if(!OrderDelete(SellStopOrder))
                 {
                  Alert("OrderDelete Error #",GetLastError());
                  return(-1);
                 }
              }
            // check whether the StopLoss should be moved:
            // if the size of Trailing Stop is not very small,
            if(TrailingStop>MarketInfo(Symbol(),
               MODE_STOPLEVEL))
              {
               // if the profit exceeds the TrailingStop
               // points,
               if(NormalizeDouble(Bid-OrderOpenPrice(),Digits)>NormalizeDouble(TrailingStop*Point,Digits))
                 {
                  // if the new StopLoss level exceeds
                  // the current level of that for the
                  // position
                  // (or if the position does not have
                  // a StopLoss),
                  if(NormalizeDouble(Bid-TrailingStop*Point,Digits)>OrderStopLoss() || OrderStopLoss()<=0.0)
                    {
                     //---- modify the order
                     if(!OrderModify(OrderTicket(),OrderOpenPrice(),NormalizeDouble(Bid-TrailingStop*Point,Digits),OrderTakeProfit(),OrderExpiration()))
                       {
                        Alert("OrderModify Error #",GetLastError());
                        return(-1);
                       }
                    }
                 }
              }
            // if there is an open position, quit, there
            // is nothing to do
            return(0);
           }
         // The next block is absolutely the same as the
         // block processing a BUY position,
         // this is why we do not comment on it...
         case OP_SELL:
           {
            if(BuyStopOrder>0)
              {
               if(!OrderDelete(BuyStopOrder))
                 {
                  Alert("OrderDelete Error #",GetLastError());
                  return(-1);
                 }
              }
            if(TrailingStop>MarketInfo(Symbol(),
               MODE_STOPLEVEL))
              {
               if(NormalizeDouble(OrderOpenPrice()-Ask,Digits)>NormalizeDouble(TrailingStop*Point,Digits))
                 {
                  if(NormalizeDouble(Ask+TrailingStop*Point,Digits)<OrderStopLoss() || OrderStopLoss()<=0.0)
                    {
                     if(!OrderModify(OrderTicket(),OrderOpenPrice(),NormalizeDouble(Ask+TrailingStop*Point,Digits),OrderTakeProfit(),OrderExpiration()))
                       {
                        Alert("OrderModify Error #",GetLastError());
                        return(-1);
                       }
                    }
                 }
              }
            return(0);
           }
        }
     }

//+------------------------------------------------------------------+
//| If execution has reached this point, it means there are no       |
//| pending orders or open positions                                 |
//+------------------------------------------------------------------+
//---- Place BuyStop and SellStop:
   double _OpenPriceLevel,_StopLossLevel,_TakeProfitLevel;
   _OpenPriceLevel=NormalizeDouble(Ask+Luft*Point,Digits);

   if(StopLoss>0)
     { _StopLossLevel=NormalizeDouble(_OpenPriceLevel-StopLoss*Point,Digits); }
   else
     { _StopLossLevel=0.0; }

   if(TakeProfit>0)
     { _TakeProfitLevel=NormalizeDouble(_OpenPriceLevel+TakeProfit*Point,Digits); }
   else
     { _TakeProfitLevel=0.0; }

   if(OrderSend(Symbol(),OP_BUYSTOP,Lot,_OpenPriceLevel,5,_StopLossLevel,_TakeProfitLevel,"",_MagicNumber)<0)
     {
      Alert("OrderSend Error #",GetLastError());
      return(-1);
     }

   _OpenPriceLevel=NormalizeDouble(Bid-Luft*Point,Digits);

   if(StopLoss>0)
     { _StopLossLevel=NormalizeDouble(_OpenPriceLevel+StopLoss*Point,Digits); }
   else
     { _StopLossLevel=0.0; }

   if(TakeProfit>0)
     { _TakeProfitLevel=NormalizeDouble(_OpenPriceLevel-TakeProfit*Point,Digits); }
   else
     { _TakeProfitLevel=0.0; }

   if(OrderSend(Symbol(),OP_SELLSTOP,Lot,_OpenPriceLevel,
      5,_StopLossLevel,_TakeProfitLevel,"",_MagicNumber)<0)
     {
      Alert("OrderSend Error #",GetLastError());
      return(-1);
     }
   return(0);
  }
```

Let us now write the function that would simplify the block controlling open positions. It must find by one order of each type and store their characteristics in global variables. It will look like this:

```
// global variables in which order characteristics will be stored:
int _BuyTicket=0,_SellTicket=0,_BuyStopTicket=0;
int _SellStopTicket=0,_BuyLimitTicket=0,_SellLimitTicket=0;

double _BuyLots=0.0,_SellLots=0.0,_BuyStopLots=0.0;
double _SellStopLots=0.0,_BuyLimitLots=0.0,
_SellLimitLots=0.0;

double _BuyOpenPrice=0.0,_SellOpenPrice=0.0,
_BuyStopOpenPrice=0.0;
double _SellStopOpenPrice=0.0,_BuyLimitOpenPrice=0.0,
_SellLimitOpenPrice=0.0;

double _BuyStopLoss=0.0,_SellStopLoss=0.0,_BuyStopStopLoss=0.0;
double _SellStopStopLoss=0.0,_BuyLimitStopLoss=0.0,_SellLimitStopLoss=0.0;

double _BuyTakeProfit=0.0,_SellTakeProfit=0.0,
_BuyStopTakeProfit=0.0;
double _SellStopTakeProfit=0.0,_BuyLimitTakeProfit=0.0,
_SellLimitTakeProfit=0.0;

datetime _BuyOpenTime=-1,_SellOpenTime=-1,
_BuyStopOpenTime=-1;
datetime _SellStopOpenTime=-1,_BuyLimitOpenTime=-1,
_SellLimitOpenTime=-1;

double _BuyProfit=0.0,_SellProfit=0.0,_BuySwap=0.0,
_SellSwap=0.0;
double _BuyCommission=0.0,_SellCommission=0.0;

string _BuyComment="",_SellComment="",_BuyStopComment="";
string _SellStopComment="",_BuyLimitComment="",
_SellLimitComment="";

datetime _BuyStopExpiration=-1,_SellStopExpiration=-1;
datetime _BuyLimitExpiration=-1,_SellLimitExpiration=-1;
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void OneTypeOrdersInit(int magic)
  {
// zeroizing of variables:
   _BuyTicket=0; _SellTicket=0; _BuyStopTicket=0;
   _SellStopTicket=0; _BuyLimitTicket=0; _SellLimitTicket=0;

   _BuyLots=0.0; _SellLots=0.0; _BuyStopLots=0.0;
   _SellStopLots=0.0; _BuyLimitLots=0.0; _SellLimitLots=0.0;

   _BuyOpenPrice=0.0; _SellOpenPrice=0.0; _BuyStopOpenPrice=0.0;
   _SellStopOpenPrice=0.0; _BuyLimitOpenPrice=0.0;
   _SellLimitOpenPrice=0.0;

   _BuyStopLoss=0.0; _SellStopLoss=0.0; _BuyStopStopLoss=0.0;
   _SellStopStopLoss=0.0; _BuyLimitStopLoss=0.0;
   _SellLimitStopLoss=0.0;

   _BuyTakeProfit = 0.0; _SellTakeProfit = 0.0;
   _BuyStopTakeProfit = 0.0;
   _SellStopTakeProfit=0.0; _BuyLimitTakeProfit=0.0;
   _SellLimitTakeProfit=0.0;

   _BuyOpenTime=-1; _SellOpenTime=-1; _BuyStopOpenTime=-1;
   _SellStopOpenTime=-1; _BuyLimitOpenTime=-1;
   _SellLimitOpenTime=-1;

   _BuyProfit=0.0; _SellProfit=0.0; _BuySwap=0.0;
   _SellSwap=0.0;
   _BuyCommission=0.0; _SellCommission=0.0;

   _BuyComment=""; _SellComment=""; _BuyStopComment="";
   _SellStopComment=""; _BuyLimitComment="";
   _SellLimitComment="";

   _BuyStopExpiration=-1; _SellStopExpiration=-1;
   _BuyLimitExpiration=-1; _SellLimitExpiration=-1;

   int _GetLastError=0,_OrdersTotal=OrdersTotal();
   for(int z=_OrdersTotal-1; z>=0; z --)
     {
      if(!OrderSelect(z,SELECT_BY_POS))
        {
         _GetLastError=GetLastError();
         Print("OrderSelect(",z,",SELECT_BY_POS) - Error #",
               _GetLastError);
         continue;
        }
      if(OrderMagicNumber()==magic && OrderSymbol()==
         Symbol())
        {
         switch(OrderType())
           {
            case OP_BUY:
               _BuyTicket     = OrderTicket();
               _BuyLots       = NormalizeDouble( OrderLots(), 1 );
               _BuyOpenPrice  = NormalizeDouble( OrderOpenPrice(),
                                                Digits);
               _BuyStopLoss=NormalizeDouble(OrderStopLoss(),
                                            Digits);
               _BuyTakeProfit=NormalizeDouble(OrderTakeProfit(),
                                              Digits);
               _BuyOpenTime   = OrderOpenTime();
               _BuyProfit     = NormalizeDouble( OrderProfit(), 2 );
               _BuySwap       = NormalizeDouble( OrderSwap(), 2 );
               _BuyCommission = NormalizeDouble( OrderCommission(),
                                                2);
               _BuyComment=OrderComment();
               break;
            case OP_SELL:
               _SellTicket     = OrderTicket();
               _SellLots       = NormalizeDouble( OrderLots(), 1 );
               _SellOpenPrice  = NormalizeDouble( OrderOpenPrice(),
                                                 Digits);
               _SellStopLoss=NormalizeDouble(OrderStopLoss(),
                                             Digits);
               _SellTakeProfit=NormalizeDouble(OrderTakeProfit(),
                                               Digits);
               _SellOpenTime   = OrderOpenTime();
               _SellProfit     = NormalizeDouble( OrderProfit(), 2 );
               _SellSwap       = NormalizeDouble( OrderSwap(), 2 );
               _SellCommission = NormalizeDouble( OrderCommission(),
                                                 2);
               _SellComment=OrderComment();
               break;
            case OP_BUYSTOP:
               _BuyStopTicket     = OrderTicket();
               _BuyStopLots       = NormalizeDouble( OrderLots(), 1 );
               _BuyStopOpenPrice  = NormalizeDouble( OrderOpenPrice(),
                                                    Digits);
               _BuyStopStopLoss=NormalizeDouble(OrderStopLoss(),
                                                Digits);
               _BuyStopTakeProfit=NormalizeDouble(OrderTakeProfit(),
                                                  Digits);
               _BuyStopOpenTime   = OrderOpenTime();
               _BuyStopComment    = OrderComment();
               _BuyStopExpiration = OrderExpiration();
               break;
            case OP_SELLSTOP:
               _SellStopTicket     = OrderTicket();
               _SellStopLots       = NormalizeDouble( OrderLots(), 1 );
               _SellStopOpenPrice  = NormalizeDouble( OrderOpenPrice(),
                                                     Digits);
               _SellStopStopLoss=NormalizeDouble(OrderStopLoss(),
                                                 Digits);
               _SellStopTakeProfit=NormalizeDouble(OrderTakeProfit(),
                                                   Digits);
               _SellStopOpenTime   = OrderOpenTime();
               _SellStopComment    = OrderComment();
               _SellStopExpiration = OrderExpiration();
               break;
            case OP_BUYLIMIT:
               _BuyLimitTicket     = OrderTicket();
               _BuyLimitLots       = NormalizeDouble( OrderLots(), 1 );
               _BuyLimitOpenPrice  = NormalizeDouble( OrderOpenPrice(),
                                                     Digits);
               _BuyLimitStopLoss=NormalizeDouble(OrderStopLoss(),
                                                 Digits);
               _BuyLimitTakeProfit=NormalizeDouble(OrderTakeProfit(),
                                                   Digits);
               _BuyLimitOpenTime   = OrderOpenTime();
               _BuyLimitComment    = OrderComment();
               _BuyLimitExpiration = OrderExpiration();
               break;
            case OP_SELLLIMIT:
               _SellLimitTicket     = OrderTicket();
               _SellLimitLots       = NormalizeDouble( OrderLots(), 1 );
               _SellLimitOpenPrice  = NormalizeDouble( OrderOpenPrice(),
                                                      Digits);
               _SellLimitStopLoss=NormalizeDouble(OrderStopLoss(),
                                                  Digits);
               _SellLimitTakeProfit=NormalizeDouble(OrderTakeProfit(),
                                                    Digits);
               _SellLimitOpenTime   = OrderOpenTime();
               _SellLimitComment    = OrderComment();
               _SellLimitExpiration = OrderExpiration();
               break;
           }
        }
     }
  }
```

Now let us attach the function to the expert:

```
extern int    _MagicNumber = 1123;

extern double Lot          = 0.1;
extern int    StopLoss     = 60;
// distance to StopLoss in points (0 - disable)
extern int    TakeProfit=100;
// distance to TakeProfit in points (0 - disable)
extern int    TrailingStop=50;
// Trailing Stop in points (0 - disable)

extern int    Luft=20;
// distance to the placing level of the pending order

#include <OneTypeOrdersControl.mq4>
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int start()
  {
   int _GetLastError=0;

//---- Memorize parameters of open positions (if available)
   OneTypeOrdersInit(_MagicNumber);

//---- If we have both pending orders, quit,
//---- we have to wait until one of them triggers
   if( _BuyStopTicket > 0 && _SellStopTicket > 0 ) return(0);

//---- if there is an open BUY position,
   if(_BuyTicket>0)
     {
      //---- if the SellStop has not been deleted yet, delete it:
      if(_SellStopTicket>0)
        {
         if(!OrderDelete(_SellStopTicket))
           {
            Alert("OrderDelete Error #",GetLastError());
            return(-1);
           }
        }
      //---- check whether the StopLoss should be moved:
      //---- if the Trailing Stop size is not too small,
      if(TrailingStop>MarketInfo(Symbol(),
         MODE_STOPLEVEL))
        {
         //---- if the profit on the position exceeds TrailingStop points,
         if(NormalizeDouble(Bid-_BuyOpenPrice,Digits)>
            NormalizeDouble(TrailingStop*Point,Digits))
           {
            //---- if the new level of StopLoss exceeds that of the current
            //     position
            //---- (or if the position does not have a StopLoss),
            if(NormalizeDouble(Bid-TrailingStop*Point,
               Digits)>_BuyStopLoss
               || _BuyStopLoss<=0.0)
              {
               //---- modify the order
               if(!OrderModify(_BuyTicket,_BuyOpenPrice,
                  NormalizeDouble(Bid-TrailingStop*Point,
                  Digits),
                  _BuyTakeProfit,0))
                 {
                  Alert("OrderModify Error #",
                        GetLastError());
                  return(-1);
                 }
              }
           }
        }
      //---- if there is an open position, quit, there is nothing to do
      return(0);
     }

//---- The block below is absolutely similar to that processing
//     the BUY position,
//---- this is why we do not comment on it...
   if(_SellTicket>0)
     {
      if(_BuyStopTicket>0)
        {
         if(!OrderDelete(_BuyStopTicket))
           {
            Alert("OrderDelete Error #",GetLastError());
            return(-1);
           }
        }
      if(TrailingStop>MarketInfo(Symbol(),MODE_STOPLEVEL))
        {
         if(NormalizeDouble(_SellOpenPrice-Ask,Digits)>
            NormalizeDouble(TrailingStop*Point,Digits))
           {
            if(NormalizeDouble(Ask+TrailingStop*Point,
               Digits)<_SellStopLoss
               || _SellStopLoss<=0.0)
              {
               if(!OrderModify(_SellTicket,_SellOpenPrice,
                  NormalizeDouble(Ask+TrailingStop*Point,
                  Digits),
                  _SellTakeProfit,0))
                 {
                  Alert("OrderModify Error #",
                        GetLastError());
                  return(-1);
                 }
              }
           }
        }
      return(0);
     }

//+------------------------------------------------------------------+
//| If execution has reached this point, this means that there are no|
//| pending orders or open positions                                 |
//+------------------------------------------------------------------+
//---- Place BuyStop and SellStop:
   double _OpenPriceLevel,_StopLossLevel,_TakeProfitLevel;
   _OpenPriceLevel=NormalizeDouble(Ask+Luft*Point,Digits);

   if(StopLoss>0)
      _StopLossLevel=NormalizeDouble(_OpenPriceLevel -
                                     StopLoss*Point,Digits);
   else
      _StopLossLevel=0.0;

   if(TakeProfit>0)
      _TakeProfitLevel=NormalizeDouble(_OpenPriceLevel+
                                       TakeProfit*Point,Digits);
   else
      _TakeProfitLevel=0.0;

   if(OrderSend(Symbol(),OP_BUYSTOP,Lot,_OpenPriceLevel,
      5,_StopLossLevel,_TakeProfitLevel,"",_MagicNumber)<0)
     {
      Alert("OrderSend Error #",GetLastError());
      return(-1);
     }

   _OpenPriceLevel=NormalizeDouble(Bid-Luft*Point,Digits);

   if(StopLoss>0)
      _StopLossLevel=NormalizeDouble(_OpenPriceLevel+
                                     StopLoss*Point,Digits);
   else
      _StopLossLevel=0.0;

   if(TakeProfit>0)
      _TakeProfitLevel=NormalizeDouble(_OpenPriceLevel -
                                       TakeProfit*Point,Digits);
   else
      _TakeProfitLevel=0.0;

   if(OrderSend(Symbol(),OP_SELLSTOP,Lot,_OpenPriceLevel,
      5,_StopLossLevel,_TakeProfitLevel,"",
      _MagicNumber)<0)
     {
      Alert("OrderSend Error #",GetLastError());
      return(-1);
     }

   return(0);
  }
```

Here, the difference between the initial and the revised expert is much more remarkable – the block controlling positions is very simple and easy to understand.

Now it is the turn for the most complicated experts, those that do not have any limitations in amount of opened positions at a time.

### 4\. Control over All Positions

It was sufficient to use variables to store characteristics of one order. Now we have to create some arrays, one for each characteristic. Saving this, the function is practically the same:

- zeroize all arrays at startup;
- search in all orders and store in arrays characteristics of only those that have the necessary symbol and MagicNumber equal to the 'magic' function parameter;
- for better usability, add a global variable that will store the total count of orders of the expert – this will be helpful when accessing to arrays.

Let us immediately set about the function writing:

```
// the variable that will store the amount of all orders of the expert:
int _ExpertOrdersTotal=0;

// arrays where the order characteristics will be stored:
int _OrderTicket[],_OrderType[];
double _OrderLots[],_OrderOpenPrice[],_OrderStopLoss[],
_OrderTakeProfit[];
double _OrderProfit[],_OrderSwap[],_OrderCommission[];
datetime _OrderOpenTime[],_OrderExpiration[];
string _OrderComment[];
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void AllOrdersInit(int magic)
  {
   int _GetLastError=0,_OrdersTotal=OrdersTotal();

// change array sizes according to the current amount of
// positions
// (if _OrdersTotal = 0, change the arrays size for 1)
   int temp_value=MathMax(_OrdersTotal,1);
   ArrayResize(_OrderTicket,temp_value);
   ArrayResize(_OrderType,temp_value);
   ArrayResize(_OrderLots,temp_value);
   ArrayResize(_OrderOpenPrice,temp_value);
   ArrayResize(_OrderStopLoss,temp_value);
   ArrayResize(_OrderTakeProfit,temp_value);
   ArrayResize(_OrderOpenTime,temp_value);
   ArrayResize(_OrderProfit,temp_value);
   ArrayResize(_OrderSwap,temp_value);
   ArrayResize(_OrderCommission,temp_value);
   ArrayResize(_OrderComment,temp_value);
   ArrayResize(_OrderExpiration,temp_value);

// zeroize the arrays
   ArrayInitialize(_OrderTicket,0);
   ArrayInitialize(_OrderType,0);
   ArrayInitialize(_OrderLots,0);
   ArrayInitialize(_OrderOpenPrice,0);
   ArrayInitialize(_OrderStopLoss,0);
   ArrayInitialize(_OrderTakeProfit,0);
   ArrayInitialize(_OrderOpenTime,0);
   ArrayInitialize(_OrderProfit,0);
   ArrayInitialize(_OrderSwap,0);
   ArrayInitialize(_OrderCommission,0);
   ArrayInitialize(_OrderExpiration,0);

   _ExpertOrdersTotal=0;
   for(int z=_OrdersTotal-1; z>=0; z --)
     {
      if(!OrderSelect(z,SELECT_BY_POS))
        {
         _GetLastError=GetLastError();
         Print("OrderSelect(",z,",SELECT_BY_POS) - Error #",
               _GetLastError);
         continue;
        }
      if(OrderMagicNumber()==magic && OrderSymbol()==
         Symbol())
        {
         // fill the arrays
         _OrderTicket[_ExpertOrdersTotal]=OrderTicket();
         _OrderType[_ExpertOrdersTotal] = OrderType();
         _OrderLots[_ExpertOrdersTotal] =
                                         NormalizeDouble(OrderLots(),1);
         _OrderOpenPrice[_ExpertOrdersTotal]=
                                             NormalizeDouble(OrderOpenPrice(),Digits);
         _OrderStopLoss[_ExpertOrdersTotal]=
                                            NormalizeDouble(OrderStopLoss(),Digits);
         _OrderTakeProfit[_ExpertOrdersTotal]=
                                              NormalizeDouble(OrderTakeProfit(),Digits);
         _OrderOpenTime[_ExpertOrdersTotal]=OrderOpenTime();
         _OrderProfit[_ExpertOrdersTotal]=
                                          NormalizeDouble(OrderProfit(),2);
         _OrderSwap[_ExpertOrdersTotal]=
                                        NormalizeDouble(OrderSwap(),2);
         _OrderCommission[_ExpertOrdersTotal]=
                                              NormalizeDouble(OrderCommission(),2);
         _OrderComment[_ExpertOrdersTotal]=OrderComment();
         _OrderExpiration[_ExpertOrdersTotal]=
                                              OrderExpiration();
         _ExpertOrdersTotal++;
        }
     }

// change the arrays size according to the amount of
// positions that belong to the expert
// (if _ExpertOrdersTotal = 0, change the arrays size for 1)
   temp_value=MathMax(_ExpertOrdersTotal,1);
   ArrayResize(_OrderTicket,temp_value);
   ArrayResize(_OrderType,temp_value);
   ArrayResize(_OrderLots,temp_value);
   ArrayResize(_OrderOpenPrice,temp_value);
   ArrayResize(_OrderStopLoss,temp_value);
   ArrayResize(_OrderTakeProfit,temp_value);
   ArrayResize(_OrderOpenTime,temp_value);
   ArrayResize(_OrderProfit,temp_value);
   ArrayResize(_OrderSwap,temp_value);
   ArrayResize(_OrderCommission,temp_value);
   ArrayResize(_OrderComment,temp_value);
   ArrayResize(_OrderExpiration,temp_value);
  }
// the variable that will store the amount of all orders of the expert:
int _ExpertOrdersTotal=0;

// arrays where the order characteristics will be stored:
int _OrderTicket[],_OrderType[];
double _OrderLots[],_OrderOpenPrice[],_OrderStopLoss[],
_OrderTakeProfit[];
double _OrderProfit[],_OrderSwap[],_OrderCommission[];
datetime _OrderOpenTime[],_OrderExpiration[];
string _OrderComment[];
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void AllOrdersInit(int magic)
  {
   int _GetLastError=0,_OrdersTotal=OrdersTotal();

// change array sizes according to the current amount of
// positions
// (if _OrdersTotal = 0, change the arrays size for 1)
   int temp_value=MathMax(_OrdersTotal,1);
   ArrayResize(_OrderTicket,temp_value);
   ArrayResize(_OrderType,temp_value);
   ArrayResize(_OrderLots,temp_value);
   ArrayResize(_OrderOpenPrice,temp_value);
   ArrayResize(_OrderStopLoss,temp_value);
   ArrayResize(_OrderTakeProfit,temp_value);
   ArrayResize(_OrderOpenTime,temp_value);
   ArrayResize(_OrderProfit,temp_value);
   ArrayResize(_OrderSwap,temp_value);
   ArrayResize(_OrderCommission,temp_value);
   ArrayResize(_OrderComment,temp_value);
   ArrayResize(_OrderExpiration,temp_value);

// zeroize the arrays
   ArrayInitialize(_OrderTicket,0);
   ArrayInitialize(_OrderType,0);
   ArrayInitialize(_OrderLots,0);
   ArrayInitialize(_OrderOpenPrice,0);
   ArrayInitialize(_OrderStopLoss,0);
   ArrayInitialize(_OrderTakeProfit,0);
   ArrayInitialize(_OrderOpenTime,0);
   ArrayInitialize(_OrderProfit,0);
   ArrayInitialize(_OrderSwap,0);
   ArrayInitialize(_OrderCommission,0);
   ArrayInitialize(_OrderExpiration,0);

   _ExpertOrdersTotal=0;
   for(int z=_OrdersTotal-1; z>=0; z --)
     {
      if(!OrderSelect(z,SELECT_BY_POS))
        {
         _GetLastError=GetLastError();
         Print("OrderSelect(",z,",SELECT_BY_POS) - Error #",
               _GetLastError);
         continue;
        }
      if(OrderMagicNumber()==magic && OrderSymbol()==
         Symbol())
        {
         // fill the arrays
         _OrderTicket[_ExpertOrdersTotal]=OrderTicket();
         _OrderType[_ExpertOrdersTotal] = OrderType();
         _OrderLots[_ExpertOrdersTotal] =
                                         NormalizeDouble(OrderLots(),1);
         _OrderOpenPrice[_ExpertOrdersTotal]=
                                             NormalizeDouble(OrderOpenPrice(),Digits);
         _OrderStopLoss[_ExpertOrdersTotal]=
                                            NormalizeDouble(OrderStopLoss(),Digits);
         _OrderTakeProfit[_ExpertOrdersTotal]=
                                              NormalizeDouble(OrderTakeProfit(),Digits);
         _OrderOpenTime[_ExpertOrdersTotal]=OrderOpenTime();
         _OrderProfit[_ExpertOrdersTotal]=
                                          NormalizeDouble(OrderProfit(),2);
         _OrderSwap[_ExpertOrdersTotal]=
                                        NormalizeDouble(OrderSwap(),2);
         _OrderCommission[_ExpertOrdersTotal]=
                                              NormalizeDouble(OrderCommission(),2);
         _OrderComment[_ExpertOrdersTotal]=OrderComment();
         _OrderExpiration[_ExpertOrdersTotal]=
                                              OrderExpiration();
         _ExpertOrdersTotal++;
        }
     }

// change the arrays size according to the amount of
// positions that belong to the expert
// (if _ExpertOrdersTotal = 0, change the arrays size for 1)
   temp_value=MathMax(_ExpertOrdersTotal,1);
   ArrayResize(_OrderTicket,temp_value);
   ArrayResize(_OrderType,temp_value);
   ArrayResize(_OrderLots,temp_value);
   ArrayResize(_OrderOpenPrice,temp_value);
   ArrayResize(_OrderStopLoss,temp_value);
   ArrayResize(_OrderTakeProfit,temp_value);
   ArrayResize(_OrderOpenTime,temp_value);
   ArrayResize(_OrderProfit,temp_value);
   ArrayResize(_OrderSwap,temp_value);
   ArrayResize(_OrderCommission,temp_value);
   ArrayResize(_OrderComment,temp_value);
   ArrayResize(_OrderExpiration,temp_value);
  }
```

To get into details of how the function works, let us write a simple expert that would display information about all positions opened by the expert.

Its code is rather simple:

```
extern int _MagicNumber    = 0;

#include AllOrdersControl.mq4>

int start()
  {
   AllOrdersInit(_MagicNumber);

   if(_ExpertOrdersTotal>0)
     {
      string OrdersList=StringConcatenate(Symbol(),
                                          ", MagicNumber ",_MagicNumber,":\n");
      for(int n=0; n _ExpertOrdersTotal; n++)
        {
         OrdersList=StringConcatenate(OrdersList,
                                      "Order # ",_OrderTicket[n],
                                      ", profit/loss: ",
                                      DoubleToStr(_OrderProfit[n],2),
                                      " ",AccountCurrency(),"\n");
        }
      Comment(OrdersList);
     }

   return(0);
  }
```

If  \_MagicNumber is set as 0, the expert will display the list of positions opened manually:

![](https://c.mql5.com/2/14/allorderstest.gif)

### 5\. Consclusion

At the end, I would like to compare the speed of experts that search in their orders by themselves to that of experts that use functions. For this, both versions were tested in the "Every tick" mode 10 times consecutively (optimization by \_MagicNumber). Testing time was measured by MetaTrader itself – the time ellapsed is measured automatically.Thus, the results are as follows:

| **Expert Advisor** | **Time taken by 10 tests (mm:ss)** |
| --- | --- |
| CrossMACD\_beta <br>(without function) | 07:42 |
| CrossMACD | 11:37 |
| DoublePending\_beta<br>(without function) | 08:18 |
| DoublePending | 09:42 |

As you can see in the table, experts that use functions work a bit slower (at least, in testing). It is a reasonable cost of usability and simplicity of the expert source code.

In any case, everybody should decide for him or herself whether to use them or not.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/1404](https://www.mql5.com/ru/articles/1404)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/1404.zip "Download all attachments in the single ZIP archive")

[AllOrdersControl.mq4](https://www.mql5.com/en/articles/download/1404/AllOrdersControl.mq4 "Download AllOrdersControl.mq4")(4.05 KB)

[CrossMACD.mq4](https://www.mql5.com/en/articles/download/1404/CrossMACD.mq4 "Download CrossMACD.mq4")(3.01 KB)

[CrossMACD\_beta.mq4](https://www.mql5.com/en/articles/download/1404/CrossMACD_beta.mq4 "Download CrossMACD_beta.mq4")(3.63 KB)

[DoublePending.mq4](https://www.mql5.com/en/articles/download/1404/DoublePending.mq4 "Download DoublePending.mq4")(4.8 KB)

[DoublePending\_beta.mq4](https://www.mql5.com/en/articles/download/1404/DoublePending_beta.mq4 "Download DoublePending_beta.mq4")(6.74 KB)

[OneOrderControl.mq4](https://www.mql5.com/en/articles/download/1404/OneOrderControl.mq4 "Download OneOrderControl.mq4")(2.18 KB)

[OneTypeOrdersControl.mq4](https://www.mql5.com/en/articles/download/1404/OneTypeOrdersControl.mq4 "Download OneTypeOrdersControl.mq4")(6.51 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [How to Order an Expert Advisor and Obtain the Desired Result](https://www.mql5.com/en/articles/235)
- [Equivolume Charting Revisited](https://www.mql5.com/en/articles/1504)
- [Testing Visualization: Account State Charts](https://www.mql5.com/en/articles/1487)
- [An Expert Advisor Made to Order. Manual for a Trader](https://www.mql5.com/en/articles/1460)
- [Testing Visualization: Trade History](https://www.mql5.com/en/articles/1452)
- [Sound Alerts in Indicators](https://www.mql5.com/en/articles/1448)
- [Filtering by History](https://www.mql5.com/en/articles/1441)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/39233)**
(4)


![](https://c.mql5.com/avatar/avatar_na2.png)

**\[Deleted\]**
\|
27 Mar 2007 at 12:33

has how to get the total buy, sell, buystop, sellstop, buylimit and [sell limit](https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties#enum_order_type "MQL5 documentation:") the
script?

regard

egt520


![molanis](https://c.mql5.com/avatar/avatar_na2.png)

**[molanis](https://www.mql5.com/en/users/molanisfx)**
\|
11 Mar 2010 at 21:38

I added to the code in my strategy builder a script that handles errors so it resends the order n number of times (user defined) that way I make sure the order is opened. Take a look at [molanis.com](https://www.mql5.com/go?link=http://www.molanis.com/ "http://www.molanis.com/")

![](https://c.mql5.com/avatar/avatar_na2.png)

**\[Deleted\]**
\|
21 May 2010 at 20:31

is it missing ?

ArrayInitialize( \_OrderComment, 0);

![Parvin Masumov](https://c.mql5.com/avatar/avatar_na2.png)

**[Parvin Masumov](https://www.mql5.com/en/users/masumov)**
\|
11 May 2019 at 13:55

a new order does not work until I close a order.

i couldn't solve this problem

![Secrets of MetaTrader 4 Client Terminal: File Library in MetaEditor](https://c.mql5.com/2/14/211_2.gif)[Secrets of MetaTrader 4 Client Terminal: File Library in MetaEditor](https://www.mql5.com/en/articles/1430)

When creating custom programs, code editor is of great importance. The more functions are available in the editor, the faster and more convenient is creation of the program. Many programs are created on basis of an already existing code. Do you use an indicator or a script that does not fully suit your purposes? Download the code of this program from our website and customize it for yourselves.

![Trading Strategies](https://c.mql5.com/2/13/175_1.png)[Trading Strategies](https://www.mql5.com/en/articles/1419)

All categories classifying trading strategies are fully arbitrary. The classification below is to emphasize the basic differences between possible approaches to trading.

![Trading Tactics on Forex](https://c.mql5.com/2/14/205_1.png)[Trading Tactics on Forex](https://www.mql5.com/en/articles/1428)

The article will help a beginning trader to develop trading tactics on FOREX.

![Ten Basic Errors of a Newcomer in Trading](https://c.mql5.com/2/13/173_1.png)[Ten Basic Errors of a Newcomer in Trading](https://www.mql5.com/en/articles/1418)

There are ten basic errors of a newcomer intrading: trading at market opening, undue hurry in taking profit, adding of lots in a losing position, closing positions starting with the best one, revenge, the most preferable positions, trading by the principle of 'bought for ever', closing of a profitable strategic position on the first day, closing of a position when alerted to open an opposite position, doubts.

[![](https://www.mql5.com/ff/si/w766tj9vyj3g607n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fmarket%2Fmt5%2Fexpert%3FHasRent%3Don%26utm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Drent.expert%26utm_content%3Drent.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=sorsafcerhkgwrjzwwrpvelbicxjwzon&s=ae91b1eae8acb61167455495742e6cc8eb55ccedb33fd953f8256b68cbe9c3b4&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=zophrdsqvaeepassnqettcievfmxwopp&ssn=1769251377157542947&ssn_dr=0&ssn_sr=0&fv_date=1769251377&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F1404&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Orders%20Management%20-%20It%27s%20Simple%20-%20MQL4%20Articles&scr_res=1920x1080&ac=176925137751926480&fz_uniq=5083063017429537841&sv=2552)

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