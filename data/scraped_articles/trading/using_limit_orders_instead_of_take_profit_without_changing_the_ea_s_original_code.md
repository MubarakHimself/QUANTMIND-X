---
title: Using limit orders instead of Take Profit without changing the EA's original code
url: https://www.mql5.com/en/articles/5206
categories: Trading
relevance_score: 1
scraped_at: 2026-01-23T21:34:42.352919
---

[![](https://www.mql5.com/ff/sh/bhdtjfb1zry09943z2/267b575d2182c180804d340af38ce02c.jpg)\\
Trade from your iPhone or Android device\\
\\
You only need an internet connection to use the new powerful MetaTrader 5 Web terminal\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=wtigumvtenarnsocpyfoqnanxrilnbxx&s=ec8c539e52b83881ff2d16eaff6913b25803952eb277cac55f670a102b2edc1f&uid=&ref=https://www.mql5.com/en/articles/5206&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5071939563233947968)

MetaTrader 5 / Examples


### Contents

- [Introduction](https://www.mql5.com/en/articles/5206#para1)
- [1\. General aspects](https://www.mql5.com/en/articles/5206#para2)
- [2\. Principles of implementing "position - limit order" link](https://www.mql5.com/en/articles/5206#para3)
- [3\. Creating a limit take profit class](https://www.mql5.com/en/articles/5206#para4)

  - [3.1. Making changes to sending trade orders](https://www.mql5.com/en/articles/5206#para41)
  - [3.2. Processing trade operations](https://www.mql5.com/en/articles/5206#para42)

- [4\. Integrating the class to the EA](https://www.mql5.com/en/articles/5206#para5)
- [Conclusion](https://www.mql5.com/en/articles/5206#para6)

### Introduction

In various forums, users criticize MetaTrader 5 for its market performance of take profit levels. Such posts can be found on this website forum as well. Users write about the negative impact of a slippage on the financial result during take profit execution. As an alternative, some propose using limit orders for replacing a standard take profit.

On the other hand, the use of limit orders, in contrast to the standard take profit, allows traders to build an algorithm for partial and stage-by-stage closing of positions, since in the limit order, you can specify a volume different from the position's one. In this article, I want to offer one of the possible options for implementing such a take profit substitution.

### 1\. General aspects

I believe, there is no point in arguing about what is better — the built-in take profit or limit orders replacing it. Every trader should solve this issue on the basis of their strategy principles and requirements. This article simply offers one of possible solutions.

Before developing the system of limit orders, let's consider some aspects we should be aware of when designing the algorithm.

The main thing we should remember is that take profit is an order closing a position. This may seem self-evident, but everyone is accustomed to the fact that this task is performed by the terminal and the system. Since we decided to replace the system when setting a take profit, we should take full responsibility for its maintenance.

What exactly am I talking about? A position may be closed not only by take profit but also by stop loss and at trader's discretion (often involving some EAs to close at the market price). This means our system should track the presence of the accompanied position on the market and remove a limit order immediately in case it is absent for any reason. Otherwise, an undesirable position may be opened causing much greater losses as compared to a slippage during a standard take profit activation.

Besides, a position may be closed partially, as well as increased (on netting accounts). Therefore, it is important to track not only the availability of a position, but also its volume. If a position's volume is changed, a limit order should be immediately replaced.

Another aspect concerns the hedging system operation. This system performs a separate accounting of positions and allows opening several positions on a single symbol simultaneously. This means activation of a limit order does not close the existing position. Instead, it opens a new one. Thus, after a limit order is triggered, we need to perform closing by an opposite position.

Another possible issue is a take profit of pending orders. In this case, we should ensure that a take profit is not triggered before the order is processed. At first glance, it is possible to use stop-limit orders. For example, we can simultaneously place a sell stop order and a buy stop limit order. But the system does not allow us to perform a similar operation with a sell limit order. This arises an issue of tracking a pending order activation with the subsequent setting of a limit take profit. In its turn, tracking a pending order activation inside the program and setting a pending order without a take profit carries the risk of uncontrollable position opening. As a result, the price may reach the take profit level and reverse. Lack of control by the program does not allow closing the position, which eventually causes a loss.

My personal solution is setting pending orders while specifying a standard take profit. After a position is opened, a take profit is replaced with a limit order by placing the limit order and setting the take profit field to zero. This option insures us against loss of control over the situation. If the program loses connection to the server, the order take profit is activated by the system. In this case, possible losses caused by a negative slippage are lower than losses caused by lack of control.

Another issue is changing a previously set take profit. Often, when using different strategies, you have to track and adjust a take profit of an open position. We have two options here.

1. If we make changes to the code of such an EA, then in order not to look for all possible options for changing the take profit in the code, we simply replace the OrderSend function call with calling the method of our class, where we already check the presence of the previously set limit order and whether it corresponds to the new level. If necessary, change the previously placed order or ignore the command if the previously placed limit order meets the new requirements.
2. We use a purchased EA and we do not have access to its code; our program does not open a position, but only replaces the take profit. In this case, there is a high probability that a take profit is set for a position we have already set limit orders for. This means we should double-check existing limit orders for relevance and adjust them, while setting the take profit field to zero.

Besides, we should track the minimum distance for setting pending orders from the current price and the distance of freezing trades set by a broker. And if the former equally applies to setting a system take profit as well, the freezing distance may backfire when closing the tracked position near a set limit order making its removal or modification impossible. Unfortunately, such a risk should be taken into account not only when building a system, but also when using it, since it does not depend on the system's algorithm.

### 2\. Principles of implementing "position - limit order" link

As I have already mentioned before, tracking a position's state and looking for a matching limit take profit are necessary. Let's see how we can implement this. First of all, we need to determine at what point in time we need to make this control in order not to overload the terminal.

Potentially, a position can be changed at any moment when a trading session is open. However, this does not happen too often, while check on each tick significantly increases operations performed by the EA. Here we can use the events. According to MQL5 documentation, the Trade event is generated when completing a trading operation on a trade server. The [OnTrade](https://www.mql5.com/en/docs/event_handlers/ontrade) function is launched as a result of this event. Thus, this function allows launching the check of the match between open positions and placed limit take profits. This will allow us not to check the match on every tick and, at the same time, not to miss any changes.

The issue of identification comes next. At first glance, all is simple. We should simply check limit orders and open positions. However, we want to build a universal algorithm that works well on different types of accounts and with different strategies. Also keep in mind that limit orders can be used within the strategy. Therefore, we should allocate limit take profits. I offer using comments to identify them. Since our limit orders are used to replace a take profit, we will add "TP" at the beginning of the order comment to identify them. Next, we will add a stage number in case a multi-stage position closing is applied. That would be enough for the netting system, but let's not forget about the hedging system with the ability to open mutiple positions for one account. Therefore, we should add the appropriate position ID to the limit take profit comment.

### 3\. Creating a limit take profit class

Let's summarize the above. The functionality of our class can be divided into two logical processes:

1. Making changes to sending trading requests to the server.
2. Monitoring and correcting open positions and placed limit orders.

For ease of use, let's design our algorithm as the CLimitTakeProfit class and make all functions static in it. This allows us to use class methods without declaring its instance in the program code.

```
class CLimitTakeProfit : public CObject
  {
private:
   static CSymbolInfo       c_Symbol;
   static CArrayLong        i_TakeProfit; //fixed take profit
   static CArrayDouble      d_TakeProfit; //percent to close at take profit

public:
                     CLimitTakeProfit();
                    ~CLimitTakeProfit();
//---
   static void       Magic(int value)  {  i_Magic=value; }
   static int        Magic(void)       {  return i_Magic;}
//---
   static void       OnlyOneSymbol(bool value)  {  b_OnlyOneSymbol=value;  }
   static bool       OnlyOneSymbol(void)        {  return b_OnlyOneSymbol; }
//---
   static bool       OrderSend(const MqlTradeRequest &request, MqlTradeResult &result);
   static bool       OnTrade(void);
   static bool       AddTakeProfit(uint point, double percent);
   static bool       DeleteTakeProfit(uint point);

protected:
   static int        i_Magic;          //Magic number to control
   static bool       b_OnlyOneSymbol;  //Only position of one symbol under control
//---
   static bool       SetTakeProfits(ulong position_ticket, double new_tp=0);
   static bool       SetTakeProfits(string symbol, double new_tp=0);
   static bool       CheckLimitOrder(MqlTradeRequest &request);
   static void       CheckLimitOrder(void);
   static bool       CheckOrderInHistory(ulong position_id, string comment, ENUM_ORDER_TYPE type, double &volume, ulong call_position=0);
   static double     GetLimitOrderPriceByComment(string comment);
  };
```

Magic, OnlyOneSymbol, AddTakeProfit and DeleteTakeProfit methods are the ones for configuring the class operation. Magic — magic numbers to be used for tracking positions (hedge accounts). If -1, the class works with all positions. OnlyOneSymbol instructs the class to work only with positions of a symbol chart the EA is launched on. The AddTakeProfit and DeteleTakeProfit methods are used to add and delete fixed take profit levels with an indication of the volume to be closed as a percentage of the initial position volume.

Users may apply these methods if they want to, but they are optional. By default, the method works with all magic numbers and symbols without setting fixed take profits. A limit order is set only instead of a take profit specified in the position.

#### 3.1. Making changes to sending trade orders

The OrderSend method monitors orders sent by the EA. The name and form of the method call are similar to the standard function for sending orders to MQL5. This simplifies embedding the algorithm to the code of the previously written EA by replacing the standard function with our method.

We have already described an issue of replacing take profit for pending orders. For this reason, we will be able to replace a take profit for market orders only in this block. Keep in mind, however, that accepting an order by the server does not necessarily means it will be executed. Besides, after sending the order, we receive the order ticket but not the position ID. Therefore, we will replace the take profit in the monitoring block. Here, we will only track the moment the previously set take profit is changed.

At the beginning of the method code, check if the sent request corresponds to the filters set for the algorithm operation. In addition, we should check the type of a deal. It should correspond to the position's stop level modification request. Also, do not forget to check if a take profit is present in the request. If the request does not satisfy at least one of the requirements, it is immediately sent to the server unchanged.

After checking the requirements, the request is passed to the SetTakeProfit method, where limit orders are placed. Note that the class features two methods for working by a position ticket and a symbol. The second one is more applicable to netting accounts if the request does not contain a position ticket. If the method is successful, set the take profit field in the request to zero.

Since the request may change both take profit and stop loss, check if stop loss and take profit set in the position are appropriate. If necessary, send a request to the server and exit the function. The full method code is displayed below.

```
bool CLimitTakeProfit::OrderSend(MqlTradeRequest &request,MqlTradeResult &result)
  {
   if((b_OnlyOneSymbol && request.symbol!=_Symbol) ||
      (i_Magic>=0 && request.magic!=i_Magic) || !(request.action==TRADE_ACTION_SLTP && request.tp>0))
      return(::OrderSend(request,result));
//---
   if(((request.position>0 && SetTakeProfits(request.position,request.tp)) ||
       (request.position<=0 && SetTakeProfits(request.symbol,request.tp))) && request.tp>0)
      request.tp=0;
   if((request.position>0 && PositionSelectByTicket(request.position)) ||
      (request.position<=0 && PositionSelect(request.symbol)))
     {
      if(PositionGetDouble(POSITION_SL)!=request.sl || PositionGetDouble(POSITION_TP)!=request.tp)
         return(::OrderSend(request,result));
     }
//---
   return true;
  }
```

Now, let's analyze the SetTakeProfit method in details. At the beginning of the method, check if the specified position is present and if we work with the position symbol. Next, update data on the position instrument. After that, calculate the closest prices where limit orders are allowed. In case of any error, exit the method with the 'false' result.

```
bool CLimitTakeProfit::SetTakeProfits(ulong position_ticket, double new_tp=0)
  {
   if(!PositionSelectByTicket(position_ticket) || (b_OnlyOneSymbol && PositionGetString(POSITION_SYMBOL)!=_Symbol))
      return false;
   if(!c_Symbol.Name(PositionGetString(POSITION_SYMBOL)) || !c_Symbol.Select() || !c_Symbol.Refresh() || !c_Symbol.RefreshRates())
      return false;
//---
   double min_sell_limit=c_Symbol.NormalizePrice(c_Symbol.Ask()+c_Symbol.StopsLevel()*c_Symbol.Point());
   double max_buy_limit=c_Symbol.NormalizePrice(c_Symbol.Bid()-c_Symbol.StopsLevel()*c_Symbol.Point());
```

After that, prepare the structures templates for sending a trade request for placing a limit order. Calculate the take profit placed or specified in the position to use only fixed take profits that do not exceed the calculated distance.

```
   MqlTradeRequest tp_request={0};
   MqlTradeResult tp_result={0};
   tp_request.action =  TRADE_ACTION_PENDING;
   tp_request.magic  =  PositionGetInteger(POSITION_MAGIC);
   tp_request.type_filling =  ORDER_FILLING_RETURN;
   tp_request.position=position_ticket;
   tp_request.symbol=c_Symbol.Name();
   int total=i_TakeProfit.Total();
   double tp_price=(new_tp>0 ? new_tp : PositionGetDouble(POSITION_TP));
   if(tp_price<=0)
      tp_price=GetLimitOrderPriceByComment("TPP_"+IntegerToString(position_ticket));
   double open_price=PositionGetDouble(POSITION_PRICE_OPEN);
   int tp_int=(tp_price>0 ? (int)NormalizeDouble(MathAbs(open_price-tp_price)/c_Symbol.Point(),0) : INT_MAX);
   double position_volume=PositionGetDouble(POSITION_VOLUME);
   double closed=0;
   double closed_perc=0;
   double fix_closed_per=0;
```

Next, arrange the loop for checking and placing fixed take profits. First, set the order comment ( [the coding principle was discussed above](https://www.mql5.com/en/articles/5206#para31)). Then make sure the take profit specified in the position or the request does not exceed the fixed one. If it exceeds, go to the next take profit. Also, make sure the volume of the previously set limit orders does not overlap the position volume. If limit orders overlap the position volume, exit the loop.

```
   for(int i=0;i<total;i++)
     {
      tp_request.comment="TP"+IntegerToString(i)+"_"+IntegerToString(position_ticket);
      if(i_TakeProfit.At(i)<tp_int && d_TakeProfit.At(i)>0)
        {
         if(closed>=position_volume || fix_closed_perc>=100)
            break;
```

The next step is filling in the missing elements of the trade request structure. To do this, calculate the volume of a new limit order and specify the order type and open price.

```
//---
         double lot=position_volume*MathMin(d_TakeProfit.At(i),100-closed)/(100-fix_closed_perc);
         lot=MathMin(position_volume-closed,lot);
         lot=c_Symbol.LotsMin()+MathMax(0,NormalizeDouble((lot-c_Symbol.LotsMin())/c_Symbol.LotsStep(),0)*c_Symbol.LotsStep());
         lot=NormalizeDouble(lot,2);
         tp_request.volume=lot;
         switch((ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE))
           {
            case POSITION_TYPE_BUY:
              tp_request.type=ORDER_TYPE_SELL_LIMIT;
              tp_request.price=c_Symbol.NormalizePrice(open_price+i_TakeProfit.At(i)*c_Symbol.Point());
              break;
            case POSITION_TYPE_SELL:
              tp_request.type=ORDER_TYPE_BUY_STOP;
              tp_request.price=c_Symbol.NormalizePrice(open_price-i_TakeProfit.At(i)*c_Symbol.Point());
              break;
           }
```

After filling in the trade request, check whether a limit order with the same parameters was set before. To do this, use the CheckLimitOrder method ( [the method algorithm will be considered below](https://www.mql5.com/en/articles/5206#para412)) by passing the filled request structure to it. If the order was not set before, add the set order volume to the sum of the set volumes for the position. This is necessary to ensure that position and placed limit orders volumes correspond to each other.

```
         if(CheckLimitOrder(tp_request))
           {
            if(tp_request.volume>=0)
              {
               closed+=tp_request.volume;
               closed_perc=closed/position_volume*100;
              }
            else
              {
               fix_closed_per-=tp_request.volume/(position_volume-tp_request.volume)*100;
              }
            continue;
           }
```

If the order has not yet been placed, adjust its price in accordance with the broker’s requirements with respect to the current price and send a request to the server. If the request is sent successfully, we add the volume of the order placed to the sum of the previously set volumes for the position.

```
         switch(tp_request.type)
           {
            case ORDER_TYPE_BUY_LIMIT:
              tp_request.price=MathMin(tp_request.price,max_buy_limit);
              break;
            case  ORDER_TYPE_SELL_LIMIT:
              tp_request.price=MathMax(tp_request.price,min_sell_limit);
              break;
           }
         if(::OrderSend(tp_request,tp_result))
           {
            closed+=tp_result.volume;
            closed_perc=closed/position_volume*100;
            ZeroMemory(tp_result);
           }
        }
     }
```

After completing the loop, use the same algorithm to place a limit order for the missing volume at the price specified in a modifying request (or in a position). If the volume is less than the minimum allowed one, exit the function with the 'false' result.

```
   if(tp_price>0 && position_volume>closed)
     {
      tp_request.price=tp_price;
      tp_request.comment="TPP_"+IntegerToString(position_ticket);
      tp_request.volume=position_volume-closed;
      if(tp_request.volume<c_Symbol.LotsMin())
         return false;
      tp_request.volume=c_Symbol.LotsMin()+MathMax(0,NormalizeDouble((tp_request.volume-c_Symbol.LotsMin())/c_Symbol.LotsStep(),0)*c_Symbol.LotsStep());
      tp_request.volume=NormalizeDouble(tp_request.volume,2);
//---
      switch((ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE))
        {
         case POSITION_TYPE_BUY:
           tp_request.type=ORDER_TYPE_SELL_LIMIT;
           break;
         case POSITION_TYPE_SELL:
           tp_request.type=ORDER_TYPE_BUY_LIMIT;
           break;
        }
      if(CheckLimitOrder(tp_request) && tp_request.volume>=0)
        {
         closed+=tp_request.volume;
         closed_perc=closed/position_volume*100;
        }
      else
        {
         switch(tp_request.type)
           {
            case ORDER_TYPE_BUY_LIMIT:
              tp_request.price=MathMin(tp_request.price,max_buy_limit);
              break;
            case  ORDER_TYPE_SELL_LIMIT:
              tp_request.price=MathMax(tp_request.price,min_sell_limit);
              break;
           }
         if(tp_request.volume<=0)
           {
            tp_request.volume=position_volume-closed;
            tp_request.volume=c_Symbol.LotsMin()+MathMax(0,NormalizeDouble((tp_request.volume-c_Symbol.LotsMin())/c_Symbol.LotsStep(),0)*c_Symbol.LotsStep());
            tp_request.volume=NormalizeDouble(tp_request.volume,2);
           }
         if(::OrderSend(tp_request,tp_result))
           {
            closed+=tp_result.volume;
            closed_perc=closed/position_volume*100;
            ZeroMemory(tp_result);
           }
        }
     }
```

At the method completion, check whether the volume of placed limit orders covers the position volume. If it does, set position's take profit to zero and exit the function.

```
   if(closed>=position_volume && PositionGetDouble(POSITION_TP)>0)
     {
      ZeroMemory(tp_request);
      ZeroMemory(tp_result);
      tp_request.action=TRADE_ACTION_SLTP;
      tp_request.position=position_ticket;
      tp_request.symbol=c_Symbol.Name();
      tp_request.sl=PositionGetDouble(POSITION_SL);
      tp_request.tp=0;
      tp_request.magic=PositionGetInteger(POSITION_MAGIC);
      if(!OrderSend(tp_request,tp_result))
         return false;
     }
   return true;
  }
```

Let's have a look at the CheckLimitOrder method algorithm to make the picture complete. Functionally, this method checks the presence of a previously placed limit order for a prepared trade request. If an order is already set, the method returns 'true' and the new order is not set.

At the beginning of the method, determine the closest possible levels for placing limit orders. We will need them if it is necessary to modify a previously placed order.

```
bool CLimitTakeProfit::CheckLimitOrder(MqlTradeRequest &request)
  {
   double min_sell_limit=c_Symbol.NormalizePrice(c_Symbol.Ask()+c_Symbol.StopsLevel()*c_Symbol.Point());
   double max_buy_limit=c_Symbol.NormalizePrice(c_Symbol.Bid()-c_Symbol.StopsLevel()*c_Symbol.Point());
```

The next step is to arrange the loop for iterating over all open orders. A necessary order is identified by its comment.

```
   for(int i=0;i<total;i++)
     {
      ulong ticket=OrderGetTicket((uint)i);
      if(ticket<=0)
         continue;
      if(OrderGetString(ORDER_COMMENT)!=request.comment)
         continue;
```

When finding the order with the necessary comment, check its volume and order type. If one of the parameters does not match, delete existing pending order and exit the function with the 'false' result. In case of an order removal error, the volume of the existing order is displayed in the request volume field.

```
      if(OrderGetDouble(ORDER_VOLUME_INITIAL) != request.volume || OrderGetInteger(ORDER_TYPE)!=request.type)
        {
         MqlTradeRequest del_request={0};
         MqlTradeResult del_result={0};
         del_request.action=TRADE_ACTION_REMOVE;
         del_request.order=ticket;
         if(::OrderSend(del_request,del_result))
            return false;
         request.volume=OrderGetDouble(ORDER_VOLUME_INITIAL);
        }
```

At the next stage, check the open price of the detected order and the one specified in the parameters. If necessary, modify the current order and exit the method with the 'true' result.

```
      if(MathAbs(OrderGetDouble(ORDER_PRICE_OPEN)-request.price)>=c_Symbol.Point())
        {
         MqlTradeRequest mod_request={0};
         MqlTradeResult mod_result={0};
         mod_request.action=TRADE_ACTION_MODIFY;
         mod_request.price=request.price;
         mod_request.magic=request.magic;
         mod_request.symbol=request.symbol;
         switch(request.type)
           {
            case ORDER_TYPE_BUY_LIMIT:
              if(mod_request.price>max_buy_limit)
                 return true;
              break;
            case ORDER_TYPE_SELL_LIMIT:
              if(mod_request.price<min_sell_limit)
                 return true;
              break;
           }
         bool mod=::OrderSend(mod_request,mod_result);
        }
      return true;
     }
```

However, let's not forget that there may be cases when the limit order has already worked with that volume. Therefore, if the necessary order is not found among open ones, check the order history of the current position as well. This functionality is implemented in the CheckOrderInHistory method we call in the end.

```
   if(!PositionSelectByTicket(request.position))
      return true;
//---
   return CheckOrderInHistory(PositionGetInteger(POSITION_IDENTIFIER),request.comment, request.type, request.volume);
  }
```

Depending on the account type, we have two options for a limit order activation:

1. Direct activation in a position (netting accounts).
2. A limit order opens an opposite position and positions are closed by one another (hedging accounts).

When looking for such a possibility, note that such orders may not relate to this position, so we will carry out the search for deals and receive a ticket from one of them.

```
bool CLimitTakeProfit::CheckOrderInHistory(ulong position_id, string comment, ENUM_ORDER_TYPE type, double &volume, ulong call_position=0)
  {
   if(!HistorySelectByPosition(position_id))
      return true;
   int total=HistoryDealsTotal();
   bool hedging=(AccountInfoInteger(ACCOUNT_MARGIN_MODE)==ACCOUNT_MARGIN_MODE_RETAIL_HEDGING);
//---
   for(int i=0;i<total;i++)
     {
      ulong ticket=HistoryDealGetTicket((uint)i);
      ticket=HistoryDealGetInteger(ticket,DEAL_ORDER);
      if(!HistoryOrderSelect(ticket))
         continue;
      if(ticket<=0)
         continue;
```

For hedging accounts, we should first check whether the order is related to another position. If an order from another position is detected, search for the order with the necessary comment in that position. To do this, perform a recursive call of the CheckOrderInHistory function. To avoid looping, check, whether the method was called from this position, before calling the method. If the order is detected, exit the method with the 'true' result. Otherwise, reload the history of the position and move on to the next deal.

```
      if(hedging && HistoryOrderGetInteger(ticket,ORDER_POSITION_ID)!=position_id && HistoryOrderGetInteger(ticket,ORDER_POSITION_ID)!=call_position)
        {
         if(CheckOrderInHistory(HistoryOrderGetInteger(ticket,ORDER_POSITION_ID),comment,type,volume))
            return true;
         if(!HistorySelectByPosition(position_id))
            continue;
        }
```

Check comment and order type for the current position orders. If the order is detected, write its volume to the request with the minus sign and exit the method.

```
      if(HistoryOrderGetString(ticket,ORDER_COMMENT)!=comment)
         continue;
      if(HistoryOrderGetInteger(ticket,ORDER_TYPE)!=type)
         continue;
//---
      volume=-OrderGetDouble(ORDER_VOLUME_INITIAL);
      return true;
     }
   return false;
  }
```

The full code of all methods and functions is provided in the attachment.

#### 3.2. Processing trade operations

Monitoring and correcting existing positions and open limit orders form the second block of our algorithm.

Conducting trades on the account generates the Trade event, which in turn causes execution of the OnTrade function. Add the appropriate method to the class to handle trades.

The method algorithm starts some preparatory work: obtain the number of positions opened on an account and check the order type.

```
bool CLimitTakeProfit::OnTrade(void)
  {
   int total=PositionsTotal();
   bool result=true;
   bool hedhing=AccountInfoInteger(ACCOUNT_MARGIN_MODE)==ACCOUNT_MARGIN_MODE_RETAIL_HEDGING;
```

Next, arrange the loop for iterating over open positions. At the beginning of the loop, check if the position corresponds to the symbol and magic number sorting conditions (for hedging accounts).

```
   for(int i=0;i<total;i++)
     {
      ulong ticket=PositionGetTicket((uint)i);
      if(ticket<=0 || (b_OnlyOneSymbol && PositionGetString(POSITION_SYMBOL)!=_Symbol))
         continue;
//---
     if(i_Magic>0)
        {
         if(hedhing && PositionGetInteger(POSITION_MAGIC)!=i_Magic)
            continue;
        }
```

For hedging accounts, check if the position is opened to process our limit take profit. If yes, perform a close by operation. After positions are successfully closed, move on to the next position.

```
      if(hedhing)
        {
         string comment=PositionGetString(POSITION_COMMENT);
         if(StringFind(comment,"TP")==0)
           {
            int start=StringFind(comment,"_");
            if(start>0)
              {
               long ticket_by=StringToInteger(StringSubstr(comment,start+1));
               long type=PositionGetInteger(POSITION_TYPE);
               if(ticket_by>0 && PositionSelectByTicket(ticket_by) && type!=PositionGetInteger(POSITION_TYPE))
                 {
                  MqlTradeRequest   request  ={0};
                  MqlTradeResult    trade_result   ={0};
                  request.action=TRADE_ACTION_CLOSE_BY;
                  request.position=ticket;
                  request.position_by=ticket_by;
                  if(::OrderSend(request,trade_result))
                     continue;
                 }
              }
           }
        }
```

At the end of the loop, call the SetTakeProfits method to check and set limit orders for the position. [The method algorithm was described above.](https://www.mql5.com/en/articles/5206#para411)

```
      result=(SetTakeProfits(PositionGetInteger(POSITION_TICKET)) && result);
     }
```

After completing the open positions checking loop, make sure the active limit orders correspond to open positions and, if necessary, remove limit orders remaining after closing positions. To do this, call the CheckLimitOrder method. In this case, the function is called without parameters, in contrast to the function described above. This happens because we call a completely different method, while applying a similar name is possible due to the [function overload](https://www.mql5.com/en/docs/basis/function/functionoverload) property.

```
   CheckLimitOrder();
//---
   return result;
  }
```

The method algorithm is based on iterating over all placed orders. The necessary ones are selected using the comments.

```
void CLimitTakeProfit::CheckLimitOrder(void)
  {
   int total=OrdersTotal();
   bool res=false;
//---
   for(int i=0;(i<total && !res);i++)
     {
      ulong ticket=OrderGetTicket((uint)i);
      if(ticket<=0)
         continue;
      string comment=OrderGetString(ORDER_COMMENT);
      if(StringFind(comment,"TP")!=0)
         continue;
      int pos=StringFind(comment,"_",0);
      if(pos<0)
         continue;
```

After a limit take profit is detected, retrieve the opposite position ID from the comment. Use the ID to access the specified position. If no such position exists, remove the order.

```
      long pos_ticker=StringToInteger(StringSubstr(comment,pos+1));
      if(!PositionSelectByTicket(pos_ticker))
        {
         MqlTradeRequest del_request={0};
         MqlTradeResult del_result={0};
         del_request.action=TRADE_ACTION_REMOVE;
         del_request.order=ticket;
         if(::OrderSend(del_request,del_result))
           {
            i--;
            total--;
           }
         continue;
        }
```

If you manage to access the position, check if the order type corresponds to the position type. The check is necessary for netting accounts where a reversal position is possible during trades. If a mismatch is detected, remove the order and move on to checking the next one.

```
      switch((ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE))
        {
         case POSITION_TYPE_BUY:
           if(OrderGetInteger(ORDER_TYPE)==ORDER_TYPE_SELL_LIMIT)
              continue;
           break;
         case POSITION_TYPE_SELL:
           if(OrderGetInteger(ORDER_TYPE)==ORDER_TYPE_BUY_LIMIT)
              continue;
           break;
        }
      MqlTradeRequest del_request={0};
      MqlTradeResult del_result={0};
      del_request.action=TRADE_ACTION_REMOVE;
      del_request.order=ticket;
      if(::OrderSend(del_request,del_result))
        {
         i--;
         total--;
        }
     }
//---
   return;
  }
```

Find the entire code of all class methods in the attachment.

### 4\. Integrating the class to the EA

After completing working the class, let's see how it can be integrated into the already developed EA.

As you may remember, all methods of our class are static, which means we can use them without declaring the class instance. Such an approach was originally chosen to simplify class integration into the already developed EAs. In fact, this is the first step towards integrating a class into an EA.

Next, create the LimitOrderSend function having the call parameters similar to the OrderSend function. It is to be located below the class code and its only functionality is calling the CLimitTakeProfit::OrderSend method. Next, use the [#define directive](https://www.mql5.com/en/articles/4332) to replace the original OrderSend function into the custom one. Applying the method allows us to simultaneously embed the code into all EA functions sending trade requests, so that we do not have to waste time searching for such commands along the entire EA code.

```
bool LimitOrderSend(const MqlTradeRequest &request, MqlTradeResult &result)
 { return CLimitTakeProfit::OrderSend(request,result); }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
#define OrderSend(request,result)      LimitOrderSend(request,result)
```

Since many EAs do not feature the OnTrade function, we may include it in the class file. But if your EA features this function, you need to delete or comment out the code below and add the CLimitTakeProfit::OnTrade method call to your EA's function body.

```
void OnTrade()
  {
   CLimitTakeProfit::OnTrade();
  }
```

Next, we have to add the reference to the class file using the [#include directive](https://www.mql5.com/en/docs/basis/preprosessor/include) to integrate the class into the EA. Keep in mind that the class should be located before calling other libraries and EA code. Below is an example of adding the class to the MACD Sample.mq5 EA from the terminal standard delivery.

```
//+------------------------------------------------------------------+
//|                                          MACD Sample LimitTP.mq5 |
//|                   Copyright 2009-2017, MetaQuotes Software Corp. |
//|                                              http://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright   "Copyright 2009-2017, MetaQuotes Software Corp."
#property link        "http://www.mql5.com"
#property version     "5.50"
#property description "It is important to make sure that the expert works with a normal"
#property description "chart and the user did not make any mistakes setting input"
#property description "variables (Lots, TakeProfit, TrailingStop) in our case,"
#property description "we check TakeProfit on a chart of more than 2*trend_period bars"

#define MACD_MAGIC 1234502
//---
#include <Trade\LimitTakeProfit.mqh>
//---
#include <Trade\Trade.mqh>
#include <Trade\SymbolInfo.mqh>
#include <Trade\PositionInfo.mqh>
#include <Trade\AccountInfo.mqh>
//---
```

You can add partial position closing to the [OnInit](https://www.mql5.com/en/docs/event_handlers/oninit) function code. Our EA is ready to go.

**_Do not forget to test the EA before using it on real accounts._**

```
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit(void)
  {
//--- create all necessary objects
   if(!ExtExpert.Init())
      return(INIT_FAILED);
   CLimitTakeProfit::AddTakeProfit(100,50);
//--- secceed
   return(INIT_SUCCEEDED);
  }
```

![EA operation](https://c.mql5.com/2/34/Limit_TP_demo_n.gif.gif)

The full EA code can be found in the attachment.

### Conclusion

This article offers the mechanism of replacing a position's take profit with close by limit orders. We tried to simplify the method integration into any existing EA code for as much as possible. I hope, this article will be useful to you, and you will be able to evaluate all pros and cons of both methods.

### Programs used in the article

| # | Name | Type | Description |
| --- | --- | --- | --- |
| 1 | LimitTakeProfit.mqh | Class library | Class for replacing order take profit with limit orders |
| --- | --- | --- | --- |
| 2 | MACD Sample.mq5 | Expert Advisor | Original EA used in MetaTrader 5 examples |
| --- | --- | --- | --- |
| 3 | MACD Sample LimitTP.mq5 | Expert Advisor | Example of integrating the class into the EA used in MetaTrader 5 examples |
| --- | --- | --- | --- |

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/5206](https://www.mql5.com/ru/articles/5206)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/5206.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/5206/mql5.zip "Download MQL5.zip")(184.41 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Neural Networks in Trading: Hybrid Graph Sequence Models (GSM++)](https://www.mql5.com/en/articles/17279)
- [Neural Networks in Trading: Two-Dimensional Connection Space Models (Final Part)](https://www.mql5.com/en/articles/17241)
- [Neural Networks in Trading: Two-Dimensional Connection Space Models (Chimera)](https://www.mql5.com/en/articles/17210)
- [Neural Networks in Trading: Multi-Task Learning Based on the ResNeXt Model (Final Part)](https://www.mql5.com/en/articles/17157)
- [Neural Networks in Trading: Multi-Task Learning Based on the ResNeXt Model](https://www.mql5.com/en/articles/17142)
- [Neural Networks in Trading: Hierarchical Dual-Tower Transformer (Final Part)](https://www.mql5.com/en/articles/17104)
- [Neural Networks in Trading: Hierarchical Dual-Tower Transformer (Hidformer)](https://www.mql5.com/en/articles/17069)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/293889)**
(63)


![Marcel Cardoso](https://c.mql5.com/avatar/2022/3/6222700F-A6A5.png)

**[Marcel Cardoso](https://www.mql5.com/en/users/marcelcardoso7)**
\|
9 Aug 2023 at 16:01

Hello everyone!

I'm having a \[Invalid expiration\] problem, anyone knows how to fix it?

![Alison Rossete](https://c.mql5.com/avatar/2022/8/62EEBC4B-FB59.jpg)

**[Alison Rossete](https://www.mql5.com/en/users/rossete347)**
\|
26 Dec 2023 at 20:23

Hello.

When testing the EA with LimitTakeProfit, it returns the following error message: "Invalid expiry".

I've tried adding expiry along with the [trade request structure](https://www.mql5.com/en/docs/constants/structures/mqltraderequest "MQL5 documentation: Trade request structure (MqlTradeRequest)"), but to no avail.

Someone please help me.

Machine translation applied by moderator

![Vinicius Pereira De Oliveira](https://c.mql5.com/avatar/2025/4/6804f561-0038.png)

**[Vinicius Pereira De Oliveira](https://www.mql5.com/en/users/vinicius-fx)**
\|
26 Dec 2023 at 20:32

**Alison Rossete [#](https://www.mql5.com/pt/forum/296191/page2#comment_51331033):** Olá. Ao testar o EA com LimitTakeProfit, ele retorna a seguinte mensagem de erro: “Invalid expiration”. Tentei adicionar expiração junto com a estrutura de solicitação de negociação, mas sem sucesso. Alguém me ajude, por favor.

**_Machine translation used._**

Hi Alison, are you testing on Forex or B3?

**EDIT.1:** Some related topics (B3):

[How can I get the expiry date of an index? - General - MQL5 algorithmic trading forum](https://www.mql5.com/pt/forum/425085)

[Problem with Pending Order ( error = 4756) - Expert Advisors and Automated Trading - MQL5 Algorithmic Trading Forum](https://www.mql5.com/pt/forum/365069)

[Sending Order with BMF expiry date - Expert Advisors and Automated Trading - MQL5 Algorithmic Trading Forum](https://www.mql5.com/pt/forum/344432)

[How to programme Pending Orders for the mini-dollar - General - MQL5 Algorithmic Trading Forum](https://www.mql5.com/pt/forum/40174)

![Alison Rossete](https://c.mql5.com/avatar/2022/8/62EEBC4B-FB59.jpg)

**[Alison Rossete](https://www.mql5.com/en/users/rossete347)**
\|
26 Dec 2023 at 20:44

**Marcel Cardoso [#](https://www.mql5.com/en/forum/293889/page2#comment_48649415):**

Hello everyone!

I'm having a \[Invalid expiration\] problem, anyone knows how to fix it?

I have the same problem, did you manage to solve it?

![Fernando Carreiro](https://c.mql5.com/avatar/2025/9/68d40cf8-38fb.png)

**[Fernando Carreiro](https://www.mql5.com/en/users/fmic)**
\|
26 Dec 2023 at 22:14

**[@Alison Rossete](https://www.mql5.com/en/users/rossete347) [#](https://www.mql5.com/pt/forum/296191/page2#comment_51331033):** Someone help me, please.

Automatic translation applied by the moderator

In this forum, please comment in Portuguese. Use the automatic translation tool or comment in one of the forums in another language.

![Gap - a profitable strategy or 50/50?](https://c.mql5.com/2/34/GapDown.png)[Gap - a profitable strategy or 50/50?](https://www.mql5.com/en/articles/5220)

The article dwells on gaps — significant differences between a close price of a previous timeframe and an open price of the next one, as well as on forecasting a daily bar direction. Applying the GetOpenFileName function by the system DLL is considered as well.

![Movement continuation model - searching on the chart and execution statistics](https://c.mql5.com/2/34/wave_movie.png)[Movement continuation model - searching on the chart and execution statistics](https://www.mql5.com/en/articles/4222)

This article provides programmatic definition of one of the movement continuation models. The main idea is defining two waves — the main and the correction one. For extreme points, I apply fractals as well as "potential" fractals - extreme points that have not yet formed as fractals.

![Reversal patterns: Testing the Double top/bottom pattern](https://c.mql5.com/2/34/double_top.png)[Reversal patterns: Testing the Double top/bottom pattern](https://www.mql5.com/en/articles/5319)

Traders often look for trend reversal points since the price has the greatest potential for movement at the very beginning of a newly formed trend. Consequently, various reversal patterns are considered in the technical analysis. The Double top/bottom is one of the most well-known and frequently used ones. The article proposes the method of the pattern programmatic detection. It also tests the pattern's profitability on history data.

![EA remote control methods](https://c.mql5.com/2/34/RemoteControl_EA.png)[EA remote control methods](https://www.mql5.com/en/articles/5166)

The main advantage of trading robots lies in the ability to work 24 hours a day on a remote VPS server. But sometimes it is necessary to intervene in their work, while there may be no direct access to the server. Is it possible to manage EAs remotely? The article proposes one of the options for controlling EAs via external commands.

[![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/01.png)![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/02.png)Powerful analytics for traders of any levelAll the necessary trading reports for beginners and professionals](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/454106&a=muccpajyfystoakuukdobwigjejzmpqn&s=52daad60fa795e635264e6f94898f05493bca3b5124d4cca8eb7e82333c2ef12&uid=&ref=https://www.mql5.com/en/articles/5206&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5071939563233947968)

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