---
title: Library for easy and quick development of MetaTrader programs (part XXIII): Base trading class - verification of valid parameters
url: https://www.mql5.com/en/articles/7286
categories: Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T19:38:31.492189
---

[![](https://www.mql5.com/ff/sh/0uquj7zv5pmx2m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
Market analytics in MQL5 Channels\\
\\
Tens of thousands of traders have chosen this messaging app to receive trading tips.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=epadtzgppsywkaeumqycnulasoijfbgz&s=9615c3e5c371aa0d7b34529539d05c10df73b35a1e2213e4ceee008933c7ede0&uid=&ref=https://www.mql5.com/en/articles/7286&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5070448513797527043)

MetaTrader 5 / Examples


### Contents

- [Setting trading event sounds](https://www.mql5.com/en/articles/7286#node01)
- [Control over incorrect values, automating selection and use of trading method inputs](https://www.mql5.com/en/articles/7286#node02)
- [Testing](https://www.mql5.com/en/articles/7286#node03)
- [What's next?](https://www.mql5.com/en/articles/7286#node04)


We continue the development of the trading class.

We have ready-made trading methods working with "clean" conditions. We
are already able to check if a trading order can be sent to the server before actually doing that (server, terminal and program limitations
are checked). But this is not enough. We should also verify the validity of values passed by arguments to the methods of sending requests to
the server. For example, we need to make sure a stop order is not less than the minimum acceptable value set for a symbol (

[StopLevel](https://www.mql5.com/en/docs/constants/environment_state/marketinfoconstants)).

If it is indeed less,
there is no point in sending such an order to the server, as it will be rejected by the server. To avoid loading the server with deliberately
erroneous orders, we will check the validity of stop orders and return the error before sending the order to the server.

Besides, we will check the minimum distance, at which a position cannot be opened, a pending order cannot be removed, as well as stop orders and
pending order levels cannot be modified. That distance is defined by a freeze level set for a symbol in points (

[FreezeLevel](https://www.mql5.com/en/docs/constants/environment_state/marketinfoconstants)).

For now, if a level violation is detected, we simply inform of the error and return false
from the trading method.

In the current article, we will also set and play the sounds of sending a request to the server and the sounds
of an error detected when verifying trading order values or the one returned by the server after the order had already been sent.

Handling invalid values in a trading order, as well as processing errors returned by the server are to be implemented in the next
article.

First, we will improve the base trading object so that we are able to play sounds set for any trading event. To set sounds in the test EA more
conveniently (to avoid setting a sound for each event and symbol), we will set the same standard error and success sounds for all trading
actions and symbols.

We are able to set any sound for each trading event and symbol.

### Setting trading event sounds

In the [previous article](https://www.mql5.com/en/articles/7258), we implemented the methods for playing sounds of
various trading events:

```
//--- Play the sound of (1) opening/placing a specified position/order type,
//--- (2) closing/removal of a specified position/order type, (3) StopLoss modification for a specified position/order type,
//--- (4) TakeProfit modification for a specified position/order type, (5) placement price modification for a specified order type
   void                       PlaySoundOpen(const int action);
   void                       PlaySoundClose(const int action);
   void                       PlaySoundModifySL(const int action);
   void                       PlaySoundModifyTP(const int action);
   void                       PlaySoundModifyPrice(const int action);
//--- Play the error sound of (1) opening/placing a specified position/order type,
//--- (2) closing/removal of a specified position/order type, (3) StopLoss modification for a specified position/order type,
//--- (4) TakeProfit modification for a specified position/order type, (5) placement price modification for a specified order type
   void                       PlaySoundErrorOpen(const int action);
   void                       PlaySoundErrorClose(const int action);
   void                       PlaySoundErrorModifySL(const int action);
   void                       PlaySoundErrorModifyTP(const int action);
   void                       PlaySoundErrorModifyPrice(const int action);
```

These methods were located in the public section of the class. Here, we will implement only two methods for playing success and error sounds,
while the data on a trading event are to be passed to the new methods. This will simplify the implementation of sounds for various trading
events.

Let's move the methods to the private section of the class and **change their implementation a bit:**

```
//+------------------------------------------------------------------+
//| Play the sound of opening/placing                                |
//| a specified position/order type                                  |
//+------------------------------------------------------------------+
void CTradeObj::PlaySoundOpen(const int action)
  {
   switch(action)
     {
      case ORDER_TYPE_BUY              :   if(this.UseSoundOpen(action))   CMessage::PlaySound(this.m_datas.Buy.SoundOpen());             break;
      case ORDER_TYPE_BUY_STOP         :   if(this.UseSoundOpen(action))   CMessage::PlaySound(this.m_datas.BuyStop.SoundOpen());         break;
      case ORDER_TYPE_BUY_LIMIT        :   if(this.UseSoundOpen(action))   CMessage::PlaySound(this.m_datas.BuyLimit.SoundOpen());        break;
      case ORDER_TYPE_BUY_STOP_LIMIT   :   if(this.UseSoundOpen(action))   CMessage::PlaySound(this.m_datas.BuyStopLimit.SoundOpen());    break;
      case ORDER_TYPE_SELL             :   if(this.UseSoundOpen(action))   CMessage::PlaySound(this.m_datas.Sell.SoundOpen());            break;
      case ORDER_TYPE_SELL_STOP        :   if(this.UseSoundOpen(action))   CMessage::PlaySound(this.m_datas.SellStop.SoundOpen());        break;
      case ORDER_TYPE_SELL_LIMIT       :   if(this.UseSoundOpen(action))   CMessage::PlaySound(this.m_datas.SellLimit.SoundOpen());       break;
      case ORDER_TYPE_SELL_STOP_LIMIT  :   if(this.UseSoundOpen(action))   CMessage::PlaySound(this.m_datas.SellStopLimit.SoundOpen());   break;
      default: break;
     }
  }
//+------------------------------------------------------------------+
//| Play the sound of closing/removal of                             |
//| a specified position/order type                                  |
//+------------------------------------------------------------------+
void CTradeObj::PlaySoundClose(const int action)
  {
   switch(action)
     {
      case ORDER_TYPE_BUY              :   if(this.UseSoundClose(action))  CMessage::PlaySound(this.m_datas.Buy.SoundClose());            break;
      case ORDER_TYPE_BUY_STOP         :   if(this.UseSoundClose(action))  CMessage::PlaySound(this.m_datas.BuyStop.SoundClose());        break;
      case ORDER_TYPE_BUY_LIMIT        :   if(this.UseSoundClose(action))  CMessage::PlaySound(this.m_datas.BuyLimit.SoundClose());       break;
      case ORDER_TYPE_BUY_STOP_LIMIT   :   if(this.UseSoundClose(action))  CMessage::PlaySound(this.m_datas.BuyStopLimit.SoundClose());   break;
      case ORDER_TYPE_SELL             :   if(this.UseSoundClose(action))  CMessage::PlaySound(this.m_datas.Sell.SoundClose());           break;
      case ORDER_TYPE_SELL_STOP        :   if(this.UseSoundClose(action))  CMessage::PlaySound(this.m_datas.SellStop.SoundClose());       break;
      case ORDER_TYPE_SELL_LIMIT       :   if(this.UseSoundClose(action))  CMessage::PlaySound(this.m_datas.SellLimit.SoundClose());      break;
      case ORDER_TYPE_SELL_STOP_LIMIT  :   if(this.UseSoundClose(action))  CMessage::PlaySound(this.m_datas.SellStopLimit.SoundClose());  break;
      default: break;
     }
  }
//+------------------------------------------------------------------+
//| Play StopLoss modification sound of                              |
//| a specified position/order type                                  |
//+------------------------------------------------------------------+
void CTradeObj::PlaySoundModifySL(const int action)
  {
   switch(action)
     {
      case ORDER_TYPE_BUY              :   if(this.UseSoundModifySL(action))  CMessage::PlaySound(this.m_datas.Buy.SoundModifySL());            break;
      case ORDER_TYPE_BUY_STOP         :   if(this.UseSoundModifySL(action))  CMessage::PlaySound(this.m_datas.BuyStop.SoundModifySL());        break;
      case ORDER_TYPE_BUY_LIMIT        :   if(this.UseSoundModifySL(action))  CMessage::PlaySound(this.m_datas.BuyLimit.SoundModifySL());       break;
      case ORDER_TYPE_BUY_STOP_LIMIT   :   if(this.UseSoundModifySL(action))  CMessage::PlaySound(this.m_datas.BuyStopLimit.SoundModifySL());   break;
      case ORDER_TYPE_SELL             :   if(this.UseSoundModifySL(action))  CMessage::PlaySound(this.m_datas.Sell.SoundModifySL());           break;
      case ORDER_TYPE_SELL_STOP        :   if(this.UseSoundModifySL(action))  CMessage::PlaySound(this.m_datas.SellStop.SoundModifySL());       break;
      case ORDER_TYPE_SELL_LIMIT       :   if(this.UseSoundModifySL(action))  CMessage::PlaySound(this.m_datas.SellLimit.SoundModifySL());      break;
      case ORDER_TYPE_SELL_STOP_LIMIT  :   if(this.UseSoundModifySL(action))  CMessage::PlaySound(this.m_datas.SellStopLimit.SoundModifySL());  break;
      default: break;
     }
  }
//+------------------------------------------------------------------+
//| Play TakeProfit modification sound of                            |
//| a specified position/order type                                  |
//+------------------------------------------------------------------+
void CTradeObj::PlaySoundModifyTP(const int action)
  {
   switch(action)
     {
      case ORDER_TYPE_BUY              :   if(this.UseSoundModifyTP(action))  CMessage::PlaySound(this.m_datas.Buy.SoundModifyTP());            break;
      case ORDER_TYPE_BUY_STOP         :   if(this.UseSoundModifyTP(action))  CMessage::PlaySound(this.m_datas.BuyStop.SoundModifyTP());        break;
      case ORDER_TYPE_BUY_LIMIT        :   if(this.UseSoundModifyTP(action))  CMessage::PlaySound(this.m_datas.BuyLimit.SoundModifyTP());       break;
      case ORDER_TYPE_BUY_STOP_LIMIT   :   if(this.UseSoundModifyTP(action))  CMessage::PlaySound(this.m_datas.BuyStopLimit.SoundModifyTP());   break;
      case ORDER_TYPE_SELL             :   if(this.UseSoundModifyTP(action))  CMessage::PlaySound(this.m_datas.Sell.SoundModifyTP());           break;
      case ORDER_TYPE_SELL_STOP        :   if(this.UseSoundModifyTP(action))  CMessage::PlaySound(this.m_datas.SellStop.SoundModifyTP());       break;
      case ORDER_TYPE_SELL_LIMIT       :   if(this.UseSoundModifyTP(action))  CMessage::PlaySound(this.m_datas.SellLimit.SoundModifyTP());      break;
      case ORDER_TYPE_SELL_STOP_LIMIT  :   if(this.UseSoundModifyTP(action))  CMessage::PlaySound(this.m_datas.SellStopLimit.SoundModifyTP());  break;
      default: break;
     }
  }
//+------------------------------------------------------------------+
//| Play price modification sound                                    |
//| for a specified order type                                       |
//+------------------------------------------------------------------+
void CTradeObj::PlaySoundModifyPrice(const int action)
  {
   switch(action)
     {
      case ORDER_TYPE_BUY_STOP         :   if(this.UseSoundModifyPrice(action))  CMessage::PlaySound(this.m_datas.BuyStop.SoundModifyPrice());        break;
      case ORDER_TYPE_BUY_LIMIT        :   if(this.UseSoundModifyPrice(action))  CMessage::PlaySound(this.m_datas.BuyLimit.SoundModifyPrice());       break;
      case ORDER_TYPE_BUY_STOP_LIMIT   :   if(this.UseSoundModifyPrice(action))  CMessage::PlaySound(this.m_datas.BuyStopLimit.SoundModifyPrice());   break;
      case ORDER_TYPE_SELL_STOP        :   if(this.UseSoundModifyPrice(action))  CMessage::PlaySound(this.m_datas.SellStop.SoundModifyPrice());       break;
      case ORDER_TYPE_SELL_LIMIT       :   if(this.UseSoundModifyPrice(action))  CMessage::PlaySound(this.m_datas.SellLimit.SoundModifyPrice());      break;
      case ORDER_TYPE_SELL_STOP_LIMIT  :   if(this.UseSoundModifyPrice(action))  CMessage::PlaySound(this.m_datas.SellStopLimit.SoundModifyPrice());  break;
      default: break;
     }
  }
//+------------------------------------------------------------------+
//| Play the error sound of opening/placing                          |
//| a specified position/order type                                  |
//+------------------------------------------------------------------+
void CTradeObj::PlaySoundErrorOpen(const int action)
  {
   switch(action)
     {
      case ORDER_TYPE_BUY              :   if(this.UseSoundOpen(action))   CMessage::PlaySound(this.m_datas.Buy.SoundErrorOpen());              break;
      case ORDER_TYPE_BUY_STOP         :   if(this.UseSoundOpen(action))   CMessage::PlaySound(this.m_datas.BuyStop.SoundErrorOpen());          break;
      case ORDER_TYPE_BUY_LIMIT        :   if(this.UseSoundOpen(action))   CMessage::PlaySound(this.m_datas.BuyLimit.SoundErrorOpen());         break;
      case ORDER_TYPE_BUY_STOP_LIMIT   :   if(this.UseSoundOpen(action))   CMessage::PlaySound(this.m_datas.BuyStopLimit.SoundErrorOpen());     break;
      case ORDER_TYPE_SELL             :   if(this.UseSoundOpen(action))   CMessage::PlaySound(this.m_datas.Sell.SoundErrorOpen());             break;
      case ORDER_TYPE_SELL_STOP        :   if(this.UseSoundOpen(action))   CMessage::PlaySound(this.m_datas.SellStop.SoundErrorOpen());         break;
      case ORDER_TYPE_SELL_LIMIT       :   if(this.UseSoundOpen(action))   CMessage::PlaySound(this.m_datas.SellLimit.SoundErrorOpen());        break;
      case ORDER_TYPE_SELL_STOP_LIMIT  :   if(this.UseSoundOpen(action))   CMessage::PlaySound(this.m_datas.SellStopLimit.SoundErrorOpen());    break;
      default: break;
     }
  }
//+------------------------------------------------------------------+
//| Play the error sound of closing/removal of                       |
//| a specified position/order type                                  |
//+------------------------------------------------------------------+
void CTradeObj::PlaySoundErrorClose(const int action)
  {
   switch(action)
     {
      case ORDER_TYPE_BUY              :   if(this.UseSoundClose(action))  CMessage::PlaySound(this.m_datas.Buy.SoundErrorClose());             break;
      case ORDER_TYPE_BUY_STOP         :   if(this.UseSoundClose(action))  CMessage::PlaySound(this.m_datas.BuyStop.SoundErrorClose());         break;
      case ORDER_TYPE_BUY_LIMIT        :   if(this.UseSoundClose(action))  CMessage::PlaySound(this.m_datas.BuyLimit.SoundErrorClose());        break;
      case ORDER_TYPE_BUY_STOP_LIMIT   :   if(this.UseSoundClose(action))  CMessage::PlaySound(this.m_datas.BuyStopLimit.SoundErrorClose());    break;
      case ORDER_TYPE_SELL             :   if(this.UseSoundClose(action))  CMessage::PlaySound(this.m_datas.Sell.SoundErrorClose());            break;
      case ORDER_TYPE_SELL_STOP        :   if(this.UseSoundClose(action))  CMessage::PlaySound(this.m_datas.SellStop.SoundErrorClose());        break;
      case ORDER_TYPE_SELL_LIMIT       :   if(this.UseSoundClose(action))  CMessage::PlaySound(this.m_datas.SellLimit.SoundErrorClose());       break;
      case ORDER_TYPE_SELL_STOP_LIMIT  :   if(this.UseSoundClose(action))  CMessage::PlaySound(this.m_datas.SellStopLimit.SoundErrorClose());   break;
      default: break;
     }
  }
//+------------------------------------------------------------------+
//| Play StopLoss modification error sound of                        |
//| a specified position/order type                                  |
//+------------------------------------------------------------------+
void CTradeObj::PlaySoundErrorModifySL(const int action)
  {
   switch(action)
     {
      case ORDER_TYPE_BUY              :   if(this.UseSoundModifySL(action))  CMessage::PlaySound(this.m_datas.Buy.SoundErrorModifySL());             break;
      case ORDER_TYPE_BUY_STOP         :   if(this.UseSoundModifySL(action))  CMessage::PlaySound(this.m_datas.BuyStop.SoundErrorModifySL());         break;
      case ORDER_TYPE_BUY_LIMIT        :   if(this.UseSoundModifySL(action))  CMessage::PlaySound(this.m_datas.BuyLimit.SoundErrorModifySL());        break;
      case ORDER_TYPE_BUY_STOP_LIMIT   :   if(this.UseSoundModifySL(action))  CMessage::PlaySound(this.m_datas.BuyStopLimit.SoundErrorModifySL());    break;
      case ORDER_TYPE_SELL             :   if(this.UseSoundModifySL(action))  CMessage::PlaySound(this.m_datas.Sell.SoundErrorModifySL());            break;
      case ORDER_TYPE_SELL_STOP        :   if(this.UseSoundModifySL(action))  CMessage::PlaySound(this.m_datas.SellStop.SoundErrorModifySL());        break;
      case ORDER_TYPE_SELL_LIMIT       :   if(this.UseSoundModifySL(action))  CMessage::PlaySound(this.m_datas.SellLimit.SoundErrorModifySL());       break;
      case ORDER_TYPE_SELL_STOP_LIMIT  :   if(this.UseSoundModifySL(action))  CMessage::PlaySound(this.m_datas.SellStopLimit.SoundErrorModifySL());   break;
      default: break;
     }
  }
//+------------------------------------------------------------------+
//| Play TakeProfit modification error sound of                      |
//| a specified position/order type                                  |
//+------------------------------------------------------------------+
void CTradeObj::PlaySoundErrorModifyTP(const int action)
  {
   switch(action)
     {
      case ORDER_TYPE_BUY              :   if(this.UseSoundModifyTP(action))  CMessage::PlaySound(this.m_datas.Buy.SoundErrorModifyTP());             break;
      case ORDER_TYPE_BUY_STOP         :   if(this.UseSoundModifyTP(action))  CMessage::PlaySound(this.m_datas.BuyStop.SoundErrorModifyTP());         break;
      case ORDER_TYPE_BUY_LIMIT        :   if(this.UseSoundModifyTP(action))  CMessage::PlaySound(this.m_datas.BuyLimit.SoundErrorModifyTP());        break;
      case ORDER_TYPE_BUY_STOP_LIMIT   :   if(this.UseSoundModifyTP(action))  CMessage::PlaySound(this.m_datas.BuyStopLimit.SoundErrorModifyTP());    break;
      case ORDER_TYPE_SELL             :   if(this.UseSoundModifyTP(action))  CMessage::PlaySound(this.m_datas.Sell.SoundErrorModifyTP());            break;
      case ORDER_TYPE_SELL_STOP        :   if(this.UseSoundModifyTP(action))  CMessage::PlaySound(this.m_datas.SellStop.SoundErrorModifyTP());        break;
      case ORDER_TYPE_SELL_LIMIT       :   if(this.UseSoundModifyTP(action))  CMessage::PlaySound(this.m_datas.SellLimit.SoundErrorModifyTP());       break;
      case ORDER_TYPE_SELL_STOP_LIMIT  :   if(this.UseSoundModifyTP(action))  CMessage::PlaySound(this.m_datas.SellStopLimit.SoundErrorModifyTP());   break;
      default: break;
     }
  }
//+------------------------------------------------------------------+
//| Play price modification error sound                              |
//| for a specified order type                                       |
//+------------------------------------------------------------------+
void CTradeObj::PlaySoundErrorModifyPrice(const int action)
  {
   switch(action)
     {
      case ORDER_TYPE_BUY_STOP         :   if(this.UseSoundModifyPrice(action))  CMessage::PlaySound(this.m_datas.BuyStop.SoundErrorModifyPrice());         break;
      case ORDER_TYPE_BUY_LIMIT        :   if(this.UseSoundModifyPrice(action))  CMessage::PlaySound(this.m_datas.BuyLimit.SoundErrorModifyPrice());        break;
      case ORDER_TYPE_BUY_STOP_LIMIT   :   if(this.UseSoundModifyPrice(action))  CMessage::PlaySound(this.m_datas.BuyStopLimit.SoundErrorModifyPrice());    break;
      case ORDER_TYPE_SELL_STOP        :   if(this.UseSoundModifyPrice(action))  CMessage::PlaySound(this.m_datas.SellStop.SoundErrorModifyPrice());        break;
      case ORDER_TYPE_SELL_LIMIT       :   if(this.UseSoundModifyPrice(action))  CMessage::PlaySound(this.m_datas.SellLimit.SoundErrorModifyPrice());       break;
      case ORDER_TYPE_SELL_STOP_LIMIT  :   if(this.UseSoundModifyPrice(action))  CMessage::PlaySound(this.m_datas.SellStopLimit.SoundErrorModifyPrice());   break;
      default: break;
     }
  }
//+------------------------------------------------------------------+
```

Here we have added the check if a sound of a certain event is allowed to be played.
The sound is played only if the appropriate flag is set.

**In the public section of the class, declare the two methods** **for**
**playing success and error sounds:**

```
//--- Play a sound of a specified trading event for a set position/order type
   void                       PlaySoundSuccess(const ENUM_ACTION_TYPE action,const int order,bool sl=false,bool tp=false,bool pr=false);
//--- Play an error sound of a specified trading event for a set position/order type
   void                       PlaySoundError(const ENUM_ACTION_TYPE action,const int order,bool sl=false,bool tp=false,bool pr=false);

//--- Set/return the flag of using sounds
```

**Let's write their implementation outside the class body:**

```
//+------------------------------------------------------------------+
//| Play a sound of a specified trading event                        |
//| a specified position/order type                                  |
//+------------------------------------------------------------------+
void CTradeObj::PlaySoundSuccess(const ENUM_ACTION_TYPE action,const int order,bool sl=false,bool tp=false,bool pr=false)
  {
   if(!this.m_use_sound)
      return;
   switch((int)action)
     {
      //--- Open/set
      case ACTION_TYPE_BUY             :
      case ACTION_TYPE_BUY_LIMIT       :
      case ACTION_TYPE_BUY_STOP        :
      case ACTION_TYPE_BUY_STOP_LIMIT  :
      case ACTION_TYPE_SELL            :
      case ACTION_TYPE_SELL_LIMIT      :
      case ACTION_TYPE_SELL_STOP       :
      case ACTION_TYPE_SELL_STOP_LIMIT :
        this.PlaySoundOpen(order);
        break;
      //--- Close/remove
      case ACTION_TYPE_CLOSE           :
      case ACTION_TYPE_CLOSE_BY        :
        this.PlaySoundClose(order);
        break;
      //--- Modification
      case ACTION_TYPE_MODIFY          :
        if(sl) { this.PlaySoundModifySL(order);    return; }
        if(tp) { this.PlaySoundModifyTP(order);    return; }
        if(pr) { this.PlaySoundModifyPrice(order); return; }
        break;
      default:
        break;
     }
  }
//+------------------------------------------------------------------+
//| Play an error sound of a specified trading event                 |
//| a specified position/order type                                  |
//+------------------------------------------------------------------+
void CTradeObj::PlaySoundError(const ENUM_ACTION_TYPE action,const int order,bool sl=false,bool tp=false,bool pr=false)
  {
   if(!this.m_use_sound)
      return;
   switch((int)action)
     {
      //--- Open/set
      case ACTION_TYPE_BUY             :
      case ACTION_TYPE_BUY_LIMIT       :
      case ACTION_TYPE_BUY_STOP        :
      case ACTION_TYPE_BUY_STOP_LIMIT  :
      case ACTION_TYPE_SELL            :
      case ACTION_TYPE_SELL_LIMIT      :
      case ACTION_TYPE_SELL_STOP       :
      case ACTION_TYPE_SELL_STOP_LIMIT :
        this.PlaySoundErrorOpen(order);
        break;
      //--- Close/remove
      case ACTION_TYPE_CLOSE           :
      case ACTION_TYPE_CLOSE_BY        :
        this.PlaySoundErrorClose(order);
        break;
      //--- Modification
      case ACTION_TYPE_MODIFY          :
        if(sl) { this.PlaySoundErrorModifySL(order);    return; }
        if(tp) { this.PlaySoundErrorModifyTP(order);    return; }
        if(pr) { this.PlaySoundErrorModifyPrice(order); return; }
        break;
      default:
        break;
     }
  }
//+------------------------------------------------------------------+
```

The methods receive trading event and order
types, as well as StopLoss/TakeProfit
modification flags and order prices.


If the common flag allowing playing sounds for a trading object is not set, exit
the method — all sounds are disabled.

Next, depending on the trading
operation type, call the methods of playing the appropriate soundsfor a corresponding order.

If a trading event is a
modification, then

additionally check the flags indicating exactly what is being
modified. (If several parameters are modified at once, the sound of only the first one of them is played)

Also, the order of the arguments in trading methods has been changed: now a comment should immediately follow a magic number, which is in turn
followed by a deviation. The reason behind this is that the comment is more often set for different orders compared to a slippage. This is why a

comment has swapped places with a deviation:

```
//--- Open a position
   bool                       OpenPosition(const ENUM_POSITION_TYPE type,
                                           const double volume,
                                           const double sl=0,
                                           const double tp=0,
                                           const ulong magic=ULONG_MAX,
                                           const string comment=NULL,
                                           const ulong deviation=ULONG_MAX);
```

This has been done to all trading methods. All files are attached below.

**The method of setting standard sounds to all trading events, set**
**the flag allowing playing sounds:**

```
//+------------------------------------------------------------------+
//| Allow working with sounds and set standard sounds                |
//+------------------------------------------------------------------+
void CTradeObj::SetSoundsStandart(void)
  {
   this.SetUseSound(true);
   this.m_datas.Buy.UseSoundClose(true);
```

In addition, the class features some minor changes for simplifying compatibility with MQL4. I will not consider them here. See the attached
files for more info.

**This concludes the improvement of the trading object base class.**

When sending trading requests from the program, we need to set a pending order distance and define stop order values. To do this, we can pass a
certain order price in trading order parameters. Alternatively, we can pass a distance in price points for placing a pending order or a
distance in points from a position/pending order price stop orders are to be located at.

We can implement overloaded methods receiving the parameters in prices' real representation, as well as the methods receiving integer
distance values in points. This will allow us to perform passing to the methods of opening positions/placing orders and stop order
levels/modification.

However, this is not the best solution. First, we will have to implement at least two identical methods to receive real and integer values.


Second, this solution imposes limitations on the parameter combinations. If we pass real values, they should be real for each parameter —
order price, as well as StopLoss/TakeProfit level. The same limitation remains when passing distances in points to the methods — all passed
values should be integer.

The alternative is to implement multiple methods containing all possible combinations of prices and distances, which is not
practical.

Therefore, we will implement another solution: all trading methods will be made template. Types of the variables used to pass order
values will be defined inside the methods. This will allow us to pass the necessary values in any combinations to the methods, for example, an
order price and a distance from a stop order price in points or vice versa. This will give us much more flexibility when calculating order
levels.

Inside the trading methods, all incoming values are reduced to price values. The values in prices are sent to a trading order.

[Limitations for conducting trading operations](https://www.mql5.com/en/articles/2555) are to be checked in three
stages:

- Checking trading limitations

- Checking the funds sufficiency for opening positions/placing orders

- Checking parameter values by StopLevel and FreezeLevel


We have already implemented the first two stages. Now it is time to add checks by StopLevel and FreezeLevel. The two ready-made checks
(trading limitations and funds sufficiency) are currently launched directly from trading methods one by one, which is not practical.

Therefore, we are going to implement a unified method for checking all limitations. It will be launched from the trading methods, while
all three checks will be performed inside it in sequence. The list of detected limitations and errors will be created. If errors are
detected, the full list of errors and limitations is sent from the method to the journal, and the flag of the failure of passing all three
checks is returned. Otherwise, the success flag is returned.

Since almost all methods are ready and are only being finalized, I will not dwell on them in detail providing brief explanations instead.

### Control over incorrect values, automating selection and use of trading method inputs

The trading methods of the **CTrading** class receive real order price values. We will add the ability to pass a distance in
points as well. Supported parameter types that should be passed to trading classes for specifying prices or distances —

double, long, ulong,

int and uint. All other types are
considered invalid.

**The private section of the CTrading class receives the global flag**
**enabling sounds of trading events and the price structure**
**which is to contain prices/distances passed to the trading methods and converted into real values:**

```
//+------------------------------------------------------------------+
//| Trading class                                                    |
//+------------------------------------------------------------------+
class CTrading
  {
private:
   CAccount            *m_account;           // Pointer to the current account object
   CSymbolsCollection  *m_symbols;           // Pointer to the symbol collection list
   CMarketCollection   *m_market;            // Pointer to the list of the collection of market orders and positions
   CHistoryCollection  *m_history;           // Pointer to the list of the collection of historical orders and deals
   CArrayInt            m_list_errors;       // Error list
   bool                 m_is_trade_enable;   // Flag enabling trading
   bool                 m_use_sound;         // The flag of using sounds of the object trading events
   ENUM_LOG_LEVEL       m_log_level;         // Logging level
//---
   struct SDataPrices
     {
      double            open;                // Open price
      double            limit;               // Limit order price
      double            sl;                  // StopLoss price
      double            tp;                  // TakeProfit price
     };
   SDataPrices          m_req_price;         // Trade request prices
//--- Add the error code to the list
```

The global flag enabling sounds affects all trading events regardless of their sounds and irrespective of whether playing a sound is enabled
on each of the events.

For trading methods, we need to obtain an order object by its ticket. **Add the method declaration to the private**
**section of the class:**

```
//--- Return an order object by ticket
   COrder              *GetOrderObjByTicket(const ulong ticket);
//--- Return the number of (1) all positions, (2) buy, (3) sell positions
```

Since the methods of checking for trading limitations and funds sufficiency for opening positions/placing orders are to work within the
common method for checking errors, let's

move them from the public section of the class to the private one,
as well as

add the template method of placing trading request prices, the
methods returning flags enabling StopLevel and FreezeLevel
and

the method checking if trading operations are allowed by stop and freeze level
distances:

```
//--- Set trading request prices
   template <typename PR,typename SL,typename TP,typename PL>
   bool                 SetPrices(const ENUM_ORDER_TYPE action,const PR price,const SL sl,const TP tp,const PL limit,const string source_method,CSymbol *symbol_obj);
//--- Return the flag checking the permission to trade by (1) StopLoss, (2) TakeProfit distance, (3) order placement level by a StopLevel-based price
   bool                 CheckStopLossByStopLevel(const ENUM_ORDER_TYPE order_type,const double price,const double sl,const CSymbol *symbol_obj);
   bool                 CheckTakeProfitByStopLevel(const ENUM_ORDER_TYPE order_type,const double price,const double tp,const CSymbol *symbol_obj);
   bool                 CheckPriceByStopLevel(const ENUM_ORDER_TYPE order_type,const double price,const CSymbol *symbol_obj);
//--- Return the flag checking if a distance from a price to (1) StopLoss, (2) TakeProfit, (3) order placement level by FreezeLevel is acceptable
   bool                 CheckStopLossByFreezeLevel(const ENUM_ORDER_TYPE order_type,const double sl,const CSymbol *symbol_obj);
   bool                 CheckTakeProfitByFreezeLevel(const ENUM_ORDER_TYPE order_type,const double tp,const CSymbol *symbol_obj);
   bool                 CheckPriceByFreezeLevel(const ENUM_ORDER_TYPE order_type,const double price,const CSymbol *symbol_obj);
//--- Check trading limitations
   bool                 CheckTradeConstraints(const double volume,
                                              const ENUM_ACTION_TYPE action,
                                              const CSymbol *symbol_obj,
                                              const string source_method,
                                              double sl=0,
                                              double tp=0);
//--- Check if the funds are sufficient
   bool                 CheckMoneyFree(const double volume,const double price,const ENUM_ORDER_TYPE order_type,const CSymbol *symbol_obj,const string source_method);
//--- Check parameter values by StopLevel and FreezeLevel
   bool                 CheckLevels(const ENUM_ACTION_TYPE action,
                                    const ENUM_ORDER_TYPE order_type,
                                    double price,
                                    double limit,
                                    double sl,
                                    double tp,
                                    const CSymbol *symbol_obj,
                                    const string source_method);

public:
```

**In the public section of the class, declare the method checking**
**trading availability and trade request errors, as well as the methods**
**of setting and returning the flag enabling sounds:**

```
public:
//--- Constructor
                        CTrading();
//--- Get the pointers to the lists (make sure to call the method in program's OnInit() since the symbol collection list is created there)
   void                 OnInit(CAccount *account,CSymbolsCollection *symbols,CMarketCollection *market,CHistoryCollection *history)
                          {
                           this.m_account=account;
                           this.m_symbols=symbols;
                           this.m_market=market;
                           this.m_history=history;
                          }
//--- Return the error list
   CArrayInt           *GetListErrors(void)                                { return &this.m_list_errors; }
//--- Check for errors
   bool                 CheckErrors(const double volume,
                                    const double price,
                                    const ENUM_ACTION_TYPE action,
                                    const ENUM_ORDER_TYPE order_type,
                                    const CSymbol *symbol_obj,
                                    const string source_method,
                                    const double limit=0,
                                    double sl=0,
                                    double tp=0);

//--- Set the following for symbol trading objects:
//--- (1) correct filling policy, (2) filling policy,
//--- (3) correct order expiration type, (4) order expiration type,
//--- (5) magic number, (6) comment, (7) slippage, (8) volume, (9) order expiration date,
//--- (10) the flag of asynchronous sending of a trading request, (11) logging level
   void                 SetCorrectTypeFilling(const ENUM_ORDER_TYPE_FILLING type=ORDER_FILLING_FOK,const string symbol=NULL);
   void                 SetTypeFilling(const ENUM_ORDER_TYPE_FILLING type=ORDER_FILLING_FOK,const string symbol=NULL);
   void                 SetCorrectTypeExpiration(const ENUM_ORDER_TYPE_TIME type=ORDER_TIME_GTC,const string symbol=NULL);
   void                 SetTypeExpiration(const ENUM_ORDER_TYPE_TIME type=ORDER_TIME_GTC,const string symbol=NULL);
   void                 SetMagic(const ulong magic,const string symbol=NULL);
   void                 SetComment(const string comment,const string symbol=NULL);
   void                 SetDeviation(const ulong deviation,const string symbol=NULL);
   void                 SetVolume(const double volume=0,const string symbol=NULL);
   void                 SetExpiration(const datetime expiration=0,const string symbol=NULL);
   void                 SetAsyncMode(const bool mode=false,const string symbol=NULL);
   void                 SetLogLevel(const ENUM_LOG_LEVEL log_level=LOG_LEVEL_ERROR_MSG,const string symbol=NULL);

//--- Set standard sounds (1 symbol=NULL) for trading objects of all symbols, (2 symbol!=NULL) for a symbol trading object
   void                 SetSoundsStandart(const string symbol=NULL);
//--- Set a sound for a specified order/position type and symbol
//--- 'mode' specifies an event a sound is set for
//--- (symbol=NULL) for trading objects of all symbols,
//--- (symbol!=NULL) for a trading object of a specified symbol
   void                 SetSound(const ENUM_MODE_SET_SOUND mode,const ENUM_ORDER_TYPE action,const string sound,const string symbol=NULL);
//--- Set/return the flag enabling sounds
   void                 SetUseSounds(const bool flag);
   bool                 IsUseSounds(void)                            const { return this.m_use_sound; }
```

Let's implement all the methods declared above outside the class body.

**The method returning an order object by ticket:**

```
//+------------------------------------------------------------------+
//| Return an order object by ticket                                 |
//+------------------------------------------------------------------+
COrder *CTrading::GetOrderObjByTicket(const ulong ticket)
  {
   CArrayObj *list=this.m_market.GetList();
   list=CSelect::ByOrderProperty(list,ORDER_PROP_TICKET,ticket,EQUAL);
   if(list==NULL || list.Total()==0)
      return NULL;
   return list.At(0);
  }
//+------------------------------------------------------------------+
```

The method receives the desired ticket stored in the order
object properties.

Get the full list of all active orders and positions, sort
the list by

ticket. If
no order object with that ticket is found, return

NULL, otherwise return
the only order object from the list.

As you may remember, an order object can be represented either by a pending order, or by a
position.

The method returns an object, regardless of whether it is a pending order or a position.

**The template method calculating and writing trade request prices to the m\_req\_price structure:**

```
//+------------------------------------------------------------------+
//| Set trading request prices                                       |
//+------------------------------------------------------------------+
template <typename PR,typename SL,typename TP,typename PL>
bool CTrading::SetPrices(const ENUM_ORDER_TYPE action,const PR price,const SL sl,const TP tp,const PL limit,const string source_method,CSymbol *symbol_obj)
  {
//--- Reset the prices and check the order type. If it is invalid, inform of that and return 'false'
   ::ZeroMemory(this.m_req_price);
   if(action>ORDER_TYPE_SELL_STOP_LIMIT)
     {
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(source_method,CMessage::Text(4003));
      return false;
     }

//--- Open/close price
   if(price>0)
     {
      //--- price parameter type (double) - normalize the price up to Digits(), since the price has been passed
      if(typename(price)=="double")
         this.m_req_price.open=::NormalizeDouble(price,symbol_obj.Digits());
      //--- price parameter type (int) - the distance has been passed
      else if(typename(price)=="int" || typename(price)=="uint" || typename(price)=="long" || typename(price)=="ulong")
        {
         //--- Calculate the order price
         switch((int)action)
           {
            //--- Pending order
            case ORDER_TYPE_BUY_LIMIT       :  this.m_req_price.open=::NormalizeDouble(symbol_obj.Ask()-price*symbol_obj.Point(),symbol_obj.Digits());      break;
            case ORDER_TYPE_BUY_STOP        :
            case ORDER_TYPE_BUY_STOP_LIMIT  :  this.m_req_price.open=::NormalizeDouble(symbol_obj.Ask()+price*symbol_obj.Point(),symbol_obj.Digits());      break;

            case ORDER_TYPE_SELL_LIMIT      :  this.m_req_price.open=::NormalizeDouble(symbol_obj.BidLast()+price*symbol_obj.Point(),symbol_obj.Digits());  break;
            case ORDER_TYPE_SELL_STOP       :
            case ORDER_TYPE_SELL_STOP_LIMIT :  this.m_req_price.open=::NormalizeDouble(symbol_obj.BidLast()-price*symbol_obj.Point(),symbol_obj.Digits());  break;
            //--- Default - current position open prices
            default  :  this.m_req_price.open=
              (
               this.DirectionByActionType((ENUM_ACTION_TYPE)action)==ORDER_TYPE_BUY ? ::NormalizeDouble(symbol_obj.Ask(),symbol_obj.Digits()) :
               ::NormalizeDouble(symbol_obj.BidLast(),symbol_obj.Digits())
              ); break;
           }
        }
      //--- unsupported price types - display the message and return 'false'
      else
        {
         if(this.m_log_level>LOG_LEVEL_NO_MSG)
            ::Print(source_method,CMessage::Text(MSG_LIB_TEXT_UNSUPPORTED_PR_TYPE));
         return false;
        }
     }
   //--- If no price is specified, use the current prices
   else
     {
      this.m_req_price.open=
        (
         this.DirectionByActionType((ENUM_ACTION_TYPE)action)==ORDER_TYPE_BUY ?
         ::NormalizeDouble(symbol_obj.Ask(),symbol_obj.Digits())              :
         ::NormalizeDouble(symbol_obj.BidLast(),symbol_obj.Digits())
        );
     }

//--- StopLimit order price or distance
   if(limit>0)
     {
      //--- limit order price parameter type (double) - normalize the price up to Digits(), since the price has been passed
      if(typename(limit)=="double")
         this.m_req_price.limit=::NormalizeDouble(limit,symbol_obj.Digits());
      //--- limit order price parameter type (int) - the distance has been passed
      else if(typename(limit)=="int" || typename(limit)=="uint" || typename(limit)=="long" || typename(limit)=="ulong")
        {
         //--- Calculate a limit order price
         if(this.DirectionByActionType((ENUM_ACTION_TYPE)action)==ORDER_TYPE_BUY)
            this.m_req_price.limit=::NormalizeDouble(this.m_req_price.open-limit*symbol_obj.Point(),symbol_obj.Digits());
         else
            this.m_req_price.limit=::NormalizeDouble(this.m_req_price.open+limit*symbol_obj.Point(),symbol_obj.Digits());
        }
      //--- unsupported limit order price types - display the message and return 'false'
      else
        {
         if(this.m_log_level>LOG_LEVEL_NO_MSG)
            ::Print(source_method,CMessage::Text(MSG_LIB_TEXT_UNSUPPORTED_PL_TYPE));
         return false;
        }
     }

//--- Order price stop order prices are calculated from
   double price_open=
     (
      (action==ORDER_TYPE_BUY_STOP_LIMIT || action==ORDER_TYPE_SELL_STOP_LIMIT) && limit>0 ? this.m_req_price.limit : this.m_req_price.open
     );

//--- StopLoss
   if(sl>0)
     {
      //--- StopLoss parameter type (double) - normalize the price up to Digits(), since the price has been passed
      if(typename(sl)=="double")
         this.m_req_price.sl=::NormalizeDouble(sl,symbol_obj.Digits());
      //--- StopLoss parameter type (int) - calculate the placement distance
      else if(typename(sl)=="int" || typename(sl)=="uint" || typename(sl)=="long" || typename(sl)=="ulong")
        {
         //--- Calculate the StopLoss price
         if(this.DirectionByActionType((ENUM_ACTION_TYPE)action)==ORDER_TYPE_BUY)
            this.m_req_price.sl=::NormalizeDouble(price_open-sl*symbol_obj.Point(),symbol_obj.Digits());
         else
            this.m_req_price.sl=::NormalizeDouble(price_open+sl*symbol_obj.Point(),symbol_obj.Digits());
        }
      //--- unsupported StopLoss types - display the message and return 'false'
      else
        {
         if(this.m_log_level>LOG_LEVEL_NO_MSG)
            ::Print(source_method,CMessage::Text(MSG_LIB_TEXT_UNSUPPORTED_SL_TYPE));
         return false;
        }

     }

//--- TakeProfit
   if(tp>0)
     {
      //--- TakeProfit parameter type (double) - normalize the price up to Digits(), since the price has been passed
      if(typename(tp)=="double")
         this.m_req_price.tp=::NormalizeDouble(tp,symbol_obj.Digits());
      //--- TakeProfit parameter type (int) - calculate the placement distance
      else if(typename(tp)=="int" || typename(tp)=="uint" || typename(tp)=="long" || typename(tp)=="ulong")
        {
         if(this.DirectionByActionType((ENUM_ACTION_TYPE)action)==ORDER_TYPE_BUY)
            this.m_req_price.tp=::NormalizeDouble(price_open+tp*symbol_obj.Point(),symbol_obj.Digits());
         else
            this.m_req_price.tp=::NormalizeDouble(price_open-tp*symbol_obj.Point(),symbol_obj.Digits());
        }
      //--- unsupported TakeProfit types - display the message and return 'false'
      else
        {
         if(this.m_log_level>LOG_LEVEL_NO_MSG)
            ::Print(source_method,CMessage::Text(MSG_LIB_TEXT_UNSUPPORTED_TP_TYPE));
         return false;
        }
     }

//--- All prices are recorded
   return true;
  }
//+------------------------------------------------------------------+
```

Regardless of whether price levels are passed to the trading method in prices or distance, the method writes the calculated prices to the **m\_req\_price**
structure we declared in the private section. If the method passes a

double value, the price is normalized up to a Digits() symbol whose pointer to the object
is passed to the method. Integer values mean the distance has been passed. The method calculates a normalized price for this distance and
writes it to the structure.

All actions are described in the method code comments.

**The methods returning the validity of StopLoss, TakeProfit or order level distance relative to StopLevel:**

```
//+------------------------------------------------------------------+
//| Return the flag checking the validity of the distance            |
//| from the price to StopLoss by StopLevel                          |
//+------------------------------------------------------------------+
bool CTrading::CheckStopLossByStopLevel(const ENUM_ORDER_TYPE order_type,const double price,const double sl,const CSymbol *symbol_obj)
  {
   double lv=symbol_obj.TradeStopLevel()*symbol_obj.Point();
   double pr=(order_type==ORDER_TYPE_BUY ? symbol_obj.BidLast() : order_type==ORDER_TYPE_SELL ? symbol_obj.Ask() : price);
   return(this.DirectionByActionType((ENUM_ACTION_TYPE)order_type)==ORDER_TYPE_BUY ? sl<(pr-lv) : sl>(pr+lv));
  }
//+------------------------------------------------------------------+
//| Return the flag checking the validity of the distance            |
//| from the price to TakeProfit by StopLevel                        |
//+------------------------------------------------------------------+
bool CTrading::CheckTakeProfitByStopLevel(const ENUM_ORDER_TYPE order_type,const double price,const double tp,const CSymbol *symbol_obj)
  {
   double lv=symbol_obj.TradeStopLevel()*symbol_obj.Point();
   double pr=(order_type==ORDER_TYPE_BUY ? symbol_obj.BidLast() : order_type==ORDER_TYPE_SELL ? symbol_obj.Ask() : price);
   return(this.DirectionByActionType((ENUM_ACTION_TYPE)order_type)==ORDER_TYPE_BUY ? tp>(pr+lv) : tp<(pr-lv));
  }
//+------------------------------------------------------------------+
//| Return the flag checking the validity of the order distance      |
//| from the price to the placement level by StopLevel               |
//+------------------------------------------------------------------+
bool CTrading::CheckPriceByStopLevel(const ENUM_ORDER_TYPE order_type,const double price,const CSymbol *symbol_obj)
  {
   double lv=symbol_obj.TradeStopLevel()*symbol_obj.Point();
   double pr=(this.DirectionByActionType((ENUM_ACTION_TYPE)order_type)==ORDER_TYPE_BUY ? symbol_obj.Ask() : symbol_obj.BidLast());
   return
     (
      order_type==ORDER_TYPE_SELL_STOP       ||
      order_type==ORDER_TYPE_SELL_STOP_LIMIT ||
      order_type==ORDER_TYPE_BUY_LIMIT       ?  price<(pr-lv)  :
      order_type==ORDER_TYPE_BUY_STOP        ||
      order_type==ORDER_TYPE_BUY_STOP_LIMIT  ||
      order_type==ORDER_TYPE_SELL_LIMIT      ?  price>(pr+lv)  :
      true
     );
  }
//+------------------------------------------------------------------+
```

The reference price for checking an order distance and returning true if it
exceeds the minimum StopLevel is defined in the methods by the order type. Otherwise,

false is returned indicating the order price values are invalid.

**The methods returning the validity of StopLoss, TakeProfit or order level distance relative to FreezeLevel:**

```
//+------------------------------------------------------------------+
//| Return the flag checking the validity of the                     |
//| distance from the price to StopLoss by FreezeLevel               |
//+------------------------------------------------------------------+
bool CTrading::CheckStopLossByFreezeLevel(const ENUM_ORDER_TYPE order_type,const double sl,const CSymbol *symbol_obj)
  {
   if(symbol_obj.TradeFreezeLevel()==0 || order_type>ORDER_TYPE_SELL)
      return true;
   double lv=symbol_obj.TradeFreezeLevel()*symbol_obj.Point();
   double pr=(order_type==ORDER_TYPE_BUY ? symbol_obj.BidLast() : symbol_obj.Ask());
   return(this.DirectionByActionType((ENUM_ACTION_TYPE)order_type)==ORDER_TYPE_BUY ? sl<(pr-lv) : sl>(pr+lv));
  }
//+------------------------------------------------------------------+
//| Return the flag checking the distance validity of the            |
//| from the price to TakeProfit by FreezeLevel                      |
//+------------------------------------------------------------------+
bool CTrading::CheckTakeProfitByFreezeLevel(const ENUM_ORDER_TYPE order_type,const double tp,const CSymbol *symbol_obj)
  {
   if(symbol_obj.TradeFreezeLevel()==0 || order_type>ORDER_TYPE_SELL)
      return true;
   double lv=symbol_obj.TradeFreezeLevel()*symbol_obj.Point();
   double pr=(order_type==ORDER_TYPE_BUY ? symbol_obj.BidLast() : symbol_obj.Ask());
   return(this.DirectionByActionType((ENUM_ACTION_TYPE)order_type)==ORDER_TYPE_BUY ? tp>(pr+lv) : tp<(pr-lv));
  }
//+------------------------------------------------------------------+
//| Return the flag checking the validity of the distance            |
//| from the price to the order price by FreezeLevel                 |
//+------------------------------------------------------------------+
bool CTrading::CheckPriceByFreezeLevel(const ENUM_ORDER_TYPE order_type,const double price,const CSymbol *symbol_obj)
  {
   if(symbol_obj.TradeFreezeLevel()==0 || order_type<ORDER_TYPE_BUY_LIMIT)
      return true;
   double lv=symbol_obj.TradeFreezeLevel()*symbol_obj.Point();
   double pr=(this.DirectionByActionType((ENUM_ACTION_TYPE)order_type)==ORDER_TYPE_BUY ? symbol_obj.Ask() : symbol_obj.BidLast());
   return
     (
      order_type==ORDER_TYPE_SELL_STOP       ||
      order_type==ORDER_TYPE_SELL_STOP_LIMIT ||
      order_type==ORDER_TYPE_BUY_LIMIT       ?  price<(pr-lv)  :
      order_type==ORDER_TYPE_BUY_STOP        ||
      order_type==ORDER_TYPE_BUY_STOP_LIMIT  ||
      order_type==ORDER_TYPE_SELL_LIMIT      ?  price>(pr+lv)  :
      true
     );
  }
//+------------------------------------------------------------------+
```

Like when checking a distance by StopLevel, the distance from the current price for the order type to the order price is checked here.

The
zero freeze level means no freeze level for a symbol. Therefore, we first check the zero StopLevel and return

true if the absence of trading freeze level is confirmed.

Unlike FreezeLevel, the minimum StopLevel equal to zero means the level is floating and it should be managed as required. We will do this when
implementing handling errors returned by the trading server in subsequent articles.

**The method of checking parameter values by StopLevel and FreezeLevel:**

```
//+------------------------------------------------------------------+
//| Check parameter values by StopLevel and FreezeLevel              |
//+------------------------------------------------------------------+
bool CTrading::CheckLevels(const ENUM_ACTION_TYPE action,
                           const ENUM_ORDER_TYPE order_type,
                           double price,
                           double limit,
                           double sl,
                           double tp,
                           const CSymbol *symbol_obj,
                           const string source_method)
  {
//--- the result of conducting all checks
   bool res=true;
//--- StopLevel
//--- If this is not a position closure/order removal
   if(action!=ACTION_TYPE_CLOSE && action!=ACTION_TYPE_CLOSE_BY)
     {
      //--- When placing a pending order
      if(action>ACTION_TYPE_SELL)
        {
         //--- If the placement distance in points is less than StopLevel
         if(!this.CheckPriceByStopLevel(order_type,price,symbol_obj))
           {
            //--- add the error code to the list and write 'false' to the result
            this.AddErrorCodeToList(MSG_LIB_TEXT_PR_LESS_STOP_LEVEL);
            res &=false;
           }
        }
      //--- If StopLoss is present
      if(sl>0)
        {
         //--- If StopLoss distance in points from the open price is less than StopLevel
         double price_open=(action==ACTION_TYPE_BUY_STOP_LIMIT || action==ACTION_TYPE_SELL_STOP_LIMIT ? limit : price);
         if(!this.CheckStopLossByStopLevel(order_type,price_open,sl,symbol_obj))
           {
            //--- add the error code to the list and write 'false' to the result
            this.AddErrorCodeToList(MSG_LIB_TEXT_SL_LESS_STOP_LEVEL);
            res &=false;
           }
        }
      //--- If TakeProfit is present
      if(tp>0)
        {
         double price_open=(action==ACTION_TYPE_BUY_STOP_LIMIT || action==ACTION_TYPE_SELL_STOP_LIMIT ? limit : price);
         //--- If TakeProfit distance in points from the open price is less than StopLevel
         if(!this.CheckTakeProfitByStopLevel(order_type,price_open,tp,symbol_obj))
           {
            //--- add the error code to the list and write 'false' to the result
            this.AddErrorCodeToList(MSG_LIB_TEXT_TP_LESS_STOP_LEVEL);
            res &=false;
           }
        }
     }
//--- FreezeLevel
//--- If this is a position closure/order removal/modification
   if(action>ACTION_TYPE_SELL_STOP_LIMIT)
     {
      //--- If this is a position
      if(order_type<ORDER_TYPE_BUY_LIMIT)
        {
         //--- StopLoss modification
         if(sl>0)
           {
            //--- If the distance from the price to StopLoss is less than FreezeLevel
            if(!this.CheckStopLossByFreezeLevel(order_type,sl,symbol_obj))
              {
               //--- add the error code to the list and write 'false' to the result
               this.AddErrorCodeToList(MSG_LIB_TEXT_SL_LESS_FREEZE_LEVEL);
               res &=false;
              }
           }
         //--- TakeProfit modification
         if(tp>0)
           {
            //--- If the distance from the price to StopLoss is less than FreezeLevel
            if(!this.CheckTakeProfitByFreezeLevel(order_type,tp,symbol_obj))
              {
               //--- add the error code to the list and write 'false' to the result
               this.AddErrorCodeToList(MSG_LIB_TEXT_TP_LESS_FREEZE_LEVEL);
               res &=false;
              }
           }
        }
      //--- If this is a pending order
      else
        {
         //--- Placement price modification
         if(price>0)
           {
            //--- If the distance from the price to the order activation price is less than FreezeLevel
            if(!this.CheckPriceByFreezeLevel(order_type,price,symbol_obj))
              {
               //--- add the error code to the list and write 'false' to the result
               this.AddErrorCodeToList(MSG_LIB_TEXT_PR_LESS_FREEZE_LEVEL);
               res &=false;
              }
           }
        }
     }
   return res;
  }
//+------------------------------------------------------------------+
```

Depending on the type of a conducted trading operation and order/position type, price levels relative to StopLevel and FreezeLevel are checked. If
the prices are invalid, the error code is added to the error list and

false is added to the result. Upon completion of all checks, the final result is returned to
the calling method.

**The general method checking all limitations and errors:**

```
//+------------------------------------------------------------------+
//| Check limitations and errors                                     |
//+------------------------------------------------------------------+
bool CTrading::CheckErrors(const double volume,
                           const double price,
                           const ENUM_ACTION_TYPE action,
                           const ENUM_ORDER_TYPE order_type,
                           const CSymbol *symbol_obj,
                           const string source_method,
                           const double limit=0,
                           double sl=0,
                           double tp=0)
  {
//--- the result of conducting all checks
   bool res=true;
//--- Clear the error list
   this.m_list_errors.Clear();
   this.m_list_errors.Sort();

//--- Check trading limitations
   res &=this.CheckTradeConstraints(volume,action,symbol_obj,source_method,sl,tp);
//--- Check the funds sufficiency for opening positions/placing orders
   if(action<ACTION_TYPE_CLOSE_BY)
      res &=this.CheckMoneyFree(volume,price,order_type,symbol_obj,source_method);
//--- Check parameter values by StopLevel and FreezeLevel
   res &=this.CheckLevels(action,order_type,price,limit,sl,tp,symbol_obj,source_method);

//--- If there are limitations, display the header and the error list
   if(!res)
     {
      //--- Request was rejected before sending to the server due to:
      int total=this.m_list_errors.Total();
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
        {
         //--- For MQL5, first display the list header followed by the error list
         #ifdef __MQL5__
         ::Print(source_method,CMessage::Text(MSG_LIB_TEXT_REQUEST_REJECTED_DUE));
         for(int i=0;i<total;i++)
            ::Print((total>1 ? string(i+1)+". " : ""),CMessage::Text(m_list_errors.At(i)));
         //--- For MQL4, the journal messages are displayed in the reverse order: the error list in the reverse loop is followed by the list header
         #else
         for(int i=total-1;i>WRONG_VALUE;i--)
            ::Print((total>1 ? string(i+1)+". " : ""),CMessage::Text(m_list_errors.At(i)));
         ::Print(source_method,CMessage::Text(MSG_LIB_TEXT_REQUEST_REJECTED_DUE));
         #endif
        }
     }
   return res;
  }
//+------------------------------------------------------------------+
```

The method checking trading limitations, the
method checking the funds sufficiency, the one opening a position or
placing a pending order, as well as the method for checking minimum
stop order distances according to StopLevel and FreezeLevel are called in the method sequentially.


The operation result of each of the methods is added to the value returned from the method.


In case of any limitations and errors, the full list of detected errors is displayed in the journal.

Eventually, the
result of all checks is returned.

**The method setting the flag that enables sounds for all trading objects of all used symbols:**

```
//+------------------------------------------------------------------+
//| Set the flag enabling sounds                                     |
//+------------------------------------------------------------------+
void CTrading::SetUseSounds(const bool flag)
  {
   //--- Set the flag enabling sounds
   this.m_use_sound=flag;
   //--- Get the symbol list
   CArrayObj *list=this.m_symbols.GetList();
   if(list==NULL || list.Total()==0)
      return;
   //--- In a loop by the list of symbols
   int total=list.Total();
   for(int i=0;i<total;i++)
     {
      //--- get the next symbol object
      CSymbol *symbol_obj=list.At(i);
      if(symbol_obj==NULL)
         continue;
      //--- get a symbol trading object
      CTradeObj *trade_obj=symbol_obj.GetTradeObj();
      if(trade_obj==NULL)
         continue;
      //--- set the flag enabling sounds for a trading object
      trade_obj.SetUseSound(flag);
     }
  }
//+------------------------------------------------------------------+
```

The global flag enabling sounds by the trading class is set
in the method right away.

Similar flags are then set for each trading object of each used symbol in a loop by
all used symbols.

Since we have decided to use template trading methods to be able to pass price values to them as prices or as distances, **redefine previously defined**
**trading methods in the public section of the class —**

**set template data types for some parameters:**

```
//--- Open (1) Buy, (2) Sell position
   template<typename SL,typename TP>
   bool                 OpenBuy(const double volume,
                                const string symbol,
                                const ulong magic=ULONG_MAX,
                                const SL sl=0,
                                const TP tp=0,
                                const string comment=NULL,
                                const ulong deviation=ULONG_MAX);

   template<typename SL,typename TP>
   bool                 OpenSell(const double volume,
                                 const string symbol,
                                 const ulong magic=ULONG_MAX,
                                 const SL sl=0,
                                 const TP tp=0,
                                 const string comment=NULL,
                                 const ulong deviation=ULONG_MAX);

//--- Modify a position
   template<typename SL,typename TP>
   bool                 ModifyPosition(const ulong ticket,const SL sl=WRONG_VALUE,const TP tp=WRONG_VALUE);

//--- Close a position (1) fully, (2) partially, (3) by an opposite one
   bool                 ClosePosition(const ulong ticket,const string comment=NULL,const ulong deviation=ULONG_MAX);
   bool                 ClosePositionPartially(const ulong ticket,const double volume,const string comment=NULL,const ulong deviation=ULONG_MAX);
   bool                 ClosePositionBy(const ulong ticket,const ulong ticket_by);

//--- Set (1) BuyStop, (2) BuyLimit, (3) BuyStopLimit pending order
   template<typename PR,typename SL,typename TP>
   bool                 PlaceBuyStop(const double volume,
                                           const string symbol,
                                           const PR price,
                                           const SL sl=0,
                                           const TP tp=0,
                                           const ulong magic=ULONG_MAX,
                                           const string comment=NULL,
                                           const datetime expiration=0,
                                           const ENUM_ORDER_TYPE_TIME type_time=ORDER_TIME_GTC);
   template<typename PR,typename SL,typename TP>
   bool                 PlaceBuyLimit(const double volume,
                                           const string symbol,
                                           const PR price,
                                           const SL sl=0,
                                           const TP tp=0,
                                           const ulong magic=ULONG_MAX,
                                           const string comment=NULL,
                                           const datetime expiration=0,
                                           const ENUM_ORDER_TYPE_TIME type_time=ORDER_TIME_GTC);
   template<typename PR,typename PL,typename SL,typename TP>
   bool                 PlaceBuyStopLimit(const double volume,
                                           const string symbol,
                                           const PR price_stop,
                                           const PL price_limit,
                                           const SL sl=0,
                                           const TP tp=0,
                                           const ulong magic=ULONG_MAX,
                                           const string comment=NULL,
                                           const datetime expiration=0,
                                           const ENUM_ORDER_TYPE_TIME type_time=ORDER_TIME_GTC);

//--- Set (1) SellStop, (2) SellLimit, (3) SellStopLimit pending order
   template<typename PR,typename SL,typename TP>
   bool                 PlaceSellStop(const double volume,
                                           const string symbol,
                                           const PR price,
                                           const SL sl=0,
                                           const TP tp=0,
                                           const ulong magic=ULONG_MAX,
                                           const string comment=NULL,
                                           const datetime expiration=0,
                                           const ENUM_ORDER_TYPE_TIME type_time=ORDER_TIME_GTC);
   template<typename PR,typename SL,typename TP>
   bool                 PlaceSellLimit(const double volume,
                                           const string symbol,
                                           const PR price,
                                           const SL sl=0,
                                           const TP tp=0,
                                           const ulong magic=ULONG_MAX,
                                           const string comment=NULL,
                                           const datetime expiration=0,
                                           const ENUM_ORDER_TYPE_TIME type_time=ORDER_TIME_GTC);
   template<typename PR,typename PL,typename SL,typename TP>
   bool                 PlaceSellStopLimit(const double volume,
                                           const string symbol,
                                           const PR price_stop,
                                           const PL price_limit,
                                           const SL sl=0,
                                           const TP tp=0,
                                           const ulong magic=ULONG_MAX,
                                           const string comment=NULL,
                                           const datetime expiration=0,
                                           const ENUM_ORDER_TYPE_TIME type_time=ORDER_TIME_GTC);
//--- Modify a pending order
   template<typename PR,typename PL,typename SL,typename TP>
   bool                 ModifyOrder(const ulong ticket,
                                          const PR price=WRONG_VALUE,
                                          const SL sl=WRONG_VALUE,
                                          const TP tp=WRONG_VALUE,
                                          const PL limit=WRONG_VALUE,
                                          datetime expiration=WRONG_VALUE,
                                          ENUM_ORDER_TYPE_TIME type_time=WRONG_VALUE);
//--- Remove a pending order
   bool                 DeleteOrder(const ulong ticket);
  };
```

**Implement the Buy position opening method:**

```
//+------------------------------------------------------------------+
//| Open Buy position                                                |
//+------------------------------------------------------------------+
template<typename SL,typename TP>
bool CTrading::OpenBuy(const double volume,
                       const string symbol,
                       const ulong magic=ULONG_MAX,
                       const SL sl=0,
                       const TP tp=0,
                       const string comment=NULL,
                       const ulong deviation=ULONG_MAX)
  {
   ENUM_ACTION_TYPE action=ACTION_TYPE_BUY;
   ENUM_ORDER_TYPE order=ORDER_TYPE_BUY;
//--- Get a symbol object by a symbol name
   CSymbol *symbol_obj=this.m_symbols.GetSymbolObjByName(symbol);
   if(symbol_obj==NULL)
     {
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN,CMessage::Text(MSG_LIB_SYS_ERROR_FAILED_GET_SYM_OBJ));
      return false;
     }
//--- get a trading object from a symbol object
   CTradeObj *trade_obj=symbol_obj.GetTradeObj();
   if(trade_obj==NULL)
     {
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN,CMessage::Text(MSG_LIB_SYS_ERROR_FAILED_GET_TRADE_OBJ));
      return false;
     }
//--- Update symbol quotes
   symbol_obj.RefreshRates();
//--- Set the prices
   if(!this.SetPrices(order,0,sl,tp,0,DFUN,symbol_obj))
     {
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN,CMessage::Text(MSG_LIB_TEXT_UNSUPPORTED_PRICE_TYPE_IN_REQ));
      return false;
     }
//--- In case of trading limitations, funds insufficiency,
//--- StopLevel or FreezeLevel limitations, play the error sound and exit
   if(!this.CheckErrors(volume,symbol_obj.Ask(),action,ORDER_TYPE_BUY,symbol_obj,DFUN,0,this.m_req_price.sl,this.m_req_price.tp))
     {
      if(this.IsUseSounds())
         trade_obj.PlaySoundError(action,order);
      return false;
     }

//--- Send the request
   bool res=trade_obj.OpenPosition(POSITION_TYPE_BUY,volume,this.m_req_price.sl,this.m_req_price.tp,magic,comment,deviation);
//--- If the request is successful, play the success sound set for a symbol trading object for this type of trading operation
   if(res)
     {
      if(this.IsUseSounds())
         trade_obj.PlaySoundSuccess(action,order);
     }
//--- If the request is not successful, play the error sound set for a symbol trading object for this type of trading operation
   else
     {
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(CMessage::Text(MSG_LIB_SYS_ERROR),": ",CMessage::Text(trade_obj.GetResultRetcode()));
      if(this.IsUseSounds())
         trade_obj.PlaySoundError(action,order);
     }
//--- Return the result of sending a trading request in a symbol trading object
   return res;
  }
//+------------------------------------------------------------------+
```

Double, long, ulong,

int and uint values can be passed to
the method as StopLoss and TakeProfit parameter values. The method for setting prices writes correctly calculated and normalized prices
to the

**m\_req\_price** price structure. These calculated prices
are in turn passed to the

trading method of a symbol trading
object. The remaining actions are described in the code comments. I believe, they will not cause any questions or
misunderstandings. In any case, you are welcome to use the comments section.

The remaining trading methods are made in a similar way, so we are not going to dwell on them here. You can always study them in detail in the
library files attached below.

**This concludes the improvement of the CTrading class.**

We have examined the major changes made in the class.

We
have not considered minor changes and improvements since they are not necessary for understanding the main idea. Besides, they are all
present in the files attached below.

**Datas.mqh now features the constant of the error code**
**missed in the previous article:**

```
//--- CTrading
   MSG_LIB_TEXT_TERMINAL_NOT_TRADE_ENABLED,           // Trade operations are not allowed in the terminal (the AutoTrading button is disabled)
   MSG_LIB_TEXT_EA_NOT_TRADE_ENABLED,                 // EA is not allowed to trade (F7 --> Common --> Allow Automated Trading)
   MSG_LIB_TEXT_ACCOUNT_NOT_TRADE_ENABLED,            // Trading is disabled for the current account
   MSG_LIB_TEXT_ACCOUNT_EA_NOT_TRADE_ENABLED,         // Trading on the trading server side is disabled for EAs on the current account
   MSG_LIB_TEXT_TERMINAL_NOT_CONNECTED,               // No connection to the trade server
   MSG_LIB_TEXT_REQUEST_REJECTED_DUE,                 // Request was rejected before sending to the server due to:
   MSG_LIB_TEXT_NOT_ENOUTH_MONEY_FOR,                 // Insufficient funds for performing a trade
   MSG_LIB_TEXT_MAX_VOLUME_LIMIT_EXCEEDED,            // Exceeded maximum allowed aggregate volume of orders and positions in one direction
   MSG_LIB_TEXT_REQ_VOL_LESS_MIN_VOLUME,              // Request volume is less than the minimum acceptable one
   MSG_LIB_TEXT_REQ_VOL_MORE_MAX_VOLUME,              // Request volume exceeds the maximum acceptable one
   MSG_LIB_TEXT_CLOSE_BY_ORDERS_DISABLED,             // Close by is disabled
   MSG_LIB_TEXT_INVALID_VOLUME_STEP,                  // Request volume is not a multiple of the minimum lot change step gradation
   MSG_LIB_TEXT_CLOSE_BY_SYMBOLS_UNEQUAL,             // Symbols of opposite positions are not equal
   MSG_LIB_TEXT_SL_LESS_STOP_LEVEL,                   // StopLoss in points is less than a value allowed by symbol's StopLevel parameter
   MSG_LIB_TEXT_TP_LESS_STOP_LEVEL,                   // TakeProfit in points is less than a value allowed by symbol's StopLevel parameter
   MSG_LIB_TEXT_PR_LESS_STOP_LEVEL,                   // Order distance in points is less than a value allowed by symbol's StopLevel parameter
   MSG_LIB_TEXT_SL_LESS_FREEZE_LEVEL,                 // The distance from the price to StopLoss is less than a value allowed by symbol's FreezeLevel parameter
   MSG_LIB_TEXT_TP_LESS_FREEZE_LEVEL,                 // The distance from the price to TakeProfit is less than a value allowed by symbol's FreezeLevel parameter
   MSG_LIB_TEXT_PR_LESS_FREEZE_LEVEL,                 // The distance from the price to an order activation level is less than a value allowed by symbol's FreezeLevel parameter
   MSG_LIB_TEXT_UNSUPPORTED_SL_TYPE,                  // Unsupported StopLoss parameter type (should be 'int' or 'double')
   MSG_LIB_TEXT_UNSUPPORTED_TP_TYPE,                  // Unsupported TakeProfit parameter type (should be 'int' or 'double')
   MSG_LIB_TEXT_UNSUPPORTED_PR_TYPE,                  // Unsupported price parameter type (should be 'int' or 'double')
   MSG_LIB_TEXT_UNSUPPORTED_PL_TYPE,                  // Unsupported limit order price parameter type (should be 'int' or 'double')
   MSG_LIB_TEXT_UNSUPPORTED_PRICE_TYPE_IN_REQ,        // Unsupported price parameter type in a request

  };
```

Below is the text corresponding to the code:

```
//--- CEngine
   {"С момента последнего запуска ЕА торговых событий не было","No trade events since the last launch of EA"},
   {"Не удалось получить описание последнего торгового события","Failed to get description of the last trading event"},
   {"Не удалось получить список открытых позиций","Failed to get open positions list"},
   {"Не удалось получить список установленных ордеров","Failed to get pending orders list"},
   {"Нет открытых позиций","No open positions"},
   {"Нет установленных ордеров","No placed orders"},
   {"В терминале нет разрешения на проведение торговых операций (отключена кнопка \"Авто-торговля\")","No permission to conduct trading operations in terminal (\"AutoTrading\" button disabled)"},
   {"Для советника нет разрешения на проведение торговых операций (F7 --> Общие --> \"Разрешить автоматическую торговлю\")","EA does not have permission to conduct trading operations (F7 --> Common --> \"Allow Automatic Trading\")"},
   {"Для текущего счёта запрещена торговля","Trading prohibited for the current account"},
   {"Для советников на текущем счёте запрещена торговля на стороне торгового сервера","From the side of trade server, trading for EA on the current account prohibited"},
   {"Нет связи с торговым сервером","No connection to trading server"},
   {"Запрос отклонён до отправки на сервер по причине:","Request rejected before being sent to server due to:"},
   {"Недостаточно средств для совершения торговой операции","Not enough money to perform trading operation"},
   {"Превышен максимальный совокупный объём ордеров и позиций в одном направлении","Exceeded maximum total volume of orders and positions in one direction"},
   {"Объём в запросе меньше минимально-допустимого","Volume in request less than minimum allowable"},
   {"Объём в запросе больше максимально-допустимого","Volume in request greater than maximum allowable"},
   {"Закрытие встречным запрещено","CloseBy orders prohibited"},
   {"Объём в запросе не кратен минимальной градации шага изменения лота","Volume in request not a multiple of minimum gradation of step for changing lot"},
   {"Символы встречных позиций не равны","Symbols of two opposite positions not equal"},
   {"Размер StopLoss в пунктах меньше разрешённого параметром StopLevel символа","StopLoss in points less than allowed by symbol's StopLevel"},
   {"Размер TakeProfit в пунктах меньше разрешённого параметром StopLevel символа","TakeProfit in points less than allowed by symbol's StopLevel"},
   {"Дистанция установки ордера в пунктах меньше разрешённой параметром StopLevel символа","Distance to place order in points less than allowed by symbol's StopLevel"},
   {"Дистанция от цены до StopLoss меньше разрешённой параметром FreezeLevel символа","Distance from price to StopLoss less than allowed by symbol's FreezeLevel"},
   {"Дистанция от цены до TakeProfit меньше разрешённой параметром FreezeLevel символа","Distance from price to TakeProfit less than allowed by symbol's FreezeLevel"},
   {"Дистанция от цены до цены срабатывания ордера меньше разрешённой параметром FreezeLevel символа","Distance from price to order triggering price less than allowed by symbol's FreezeLevel"},
   {"Неподдерживаемый тип параметра StopLoss (необходимо int или double)","Unsupported StopLoss parameter type (int or double required)"},
   {"Неподдерживаемый тип параметра TakeProfit (необходимо int или double)","Unsupported TakeProfit parameter type (int or double required)"},
   {"Неподдерживаемый тип параметра цены (необходимо int или double)","Unsupported price parameter type (int or double required)"},
   {"Неподдерживаемый тип параметра цены limit-ордера (необходимо int или double)","Unsupported type of price parameter for limit order (int or double required)"},
   {"Неподдерживаемый тип параметра цены в запросе","Unsupported price parameter type in request"},

  };
```

I have received several user reports on the error detected when receiving the last [trading \\
event](https://www.mql5.com/en/articles/5724). The test EA, that relates to the articles and describes how to receive trading events, obtains data on the occurred trading event
by comparing the previous event value with the current one. This would be sufficient for the purpose of testing the tracking of trading
events by the library since I did not intend to use the unfinished library version in custom applications when writing articles on trading
events. But it turned out that obtaining info on trading events is in high demand and it is important to know the exact last event that
occurred.

The implemented method of obtaining a trading event may skip some events. For example, if you set a pending order two times in a row, the second
one is not tracked in the program (the library tracks all events) since it matches the penultimate one ("placing a pending order") though the
orders themselves may actually differ.

Therefore, we will fix this behavior. Today we will implement a simple flag informing the program of an event, so that we are able to view data on the
event in the program. In the next article, we will complete obtaining trading events in the program by creating a full list of all events
occurred simultaneously and sending it to the program. Thus, we will be able not only to find out about an occurred trading event but also view
all events occurred simultaneously as it is done for

[account](https://www.mql5.com/en/articles/6995) and [symbol collection \\
events](https://www.mql5.com/en/articles/7071).

Open the **EventsCollection.mqh** event collection file and make all the necessary improvements.

**Reset the last event flag in the class constructor:**

```
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CEventsCollection::CEventsCollection(void) : m_trade_event(TRADE_EVENT_NO_EVENT),m_trade_event_code(TRADE_EVENT_FLAG_NO_EVENT)
  {
   this.m_list_trade_events.Clear();
   this.m_list_trade_events.Sort(SORT_BY_EVENT_TIME_EVENT);
   this.m_list_trade_events.Type(COLLECTION_EVENTS_ID);
   this.m_is_hedge=#ifdef __MQL4__ true #else bool(::AccountInfoInteger(ACCOUNT_MARGIN_MODE)==ACCOUNT_MARGIN_MODE_RETAIL_HEDGING) #endif;
   this.m_chart_id=::ChartID();
   this.m_is_event=false;
   ::ZeroMemory(this.m_tick);
  }
//+------------------------------------------------------------------+
```

**Also, reset the last event flag at the very beginning of the**
**Refresh() method:**

```
//+------------------------------------------------------------------+
//| Update the event list                                            |
//+------------------------------------------------------------------+
void CEventsCollection::Refresh(CArrayObj* list_history,
                                CArrayObj* list_market,
                                CArrayObj* list_changes,
                                CArrayObj* list_control,
                                const bool is_history_event,
                                const bool is_market_event,
                                const int  new_history_orders,
                                const int  new_market_pendings,
                                const int  new_market_positions,
                                const int  new_deals,
                                const double changed_volume)
  {
//--- Exit if the lists are empty
   if(list_history==NULL || list_market==NULL)
      return;
//---
   this.m_is_event=false;
//--- If the event is in the market environment
   if(is_market_event)
     {
```

**In each** of the methods of creating a trading event in the blocks of adding a trading event to the list, sending an event to the
control program chart and setting the last event value:

```
      //--- Add the event object if it is not in the list
      if(!this.IsPresentEventInList(event))
        {
         this.m_list_trade_events.InsertSort(event);
         //--- Send a message about the event and set the value of the last trading event
         event.SendEvent();
         this.m_trade_event=event.TradeEvent();
        }
```

**complete setting the trading event flag:**

```
      if(!this.IsPresentEventInList(event))
        {
         this.m_list_trade_events.InsertSort(event);
         //--- Send a message about the event and set the value of the last trading event
         this.m_trade_event=event.TradeEvent();
         this.m_is_event=true;
         event.SendEvent();
        }
```

Use **Ctrl+F** to conveniently find all spots for setting the flag. Enter "event.SendEvent();" (without quotes) to the search bar.
Add

setting the event flag to each detected code spot as displayed in the
above listing.

Make some changes in the **Engine.mqh** file.

**In the public section of the CEngine class, add the method returning the flag of an occurred trading event:**

```
//--- Return the (1) hedge account, (2) working in the tester, (3) account event, (4) symbol event  and (5) trading event flag
   bool                 IsHedge(void)                             const { return this.m_is_hedge;                             }
   bool                 IsTester(void)                            const { return this.m_is_tester;                            }
   bool                 IsAccountsEvent(void)                     const { return this.m_accounts.IsEvent();                   }
   bool                 IsSymbolsEvent(void)                      const { return this.m_symbols.IsEvent();                    }
   bool                 IsTradeEvent(void)                        const { return this.m_events.IsEvent();                     }
```

The block containing the methods of working with sounds **features the**
**method setting the sound enabling flag:**

```
//--- Set standard sounds (symbol==NULL) for a symbol trading object, (symbol!=NULL) for trading objects of all symbols
   void                 SetSoundsStandart(const string symbol=NULL)
                          {
                           this.m_trading.SetSoundsStandart(symbol);
                          }
//--- Set the flag of using sounds
   void                 SetUseSounds(const bool flag) { this.m_trading.SetUseSounds(flag);   }
//--- Set a sound for a specified order/position type and symbol. 'mode' specifies an event a sound is set for
//--- (symbol=NULL) for trading objects of all symbols, (symbol!=NULL) for a trading object of a specified symbol
   void                 SetSound(const ENUM_MODE_SET_SOUND mode,const ENUM_ORDER_TYPE action,const string sound,const string symbol=NULL)
                          {
                           this.m_trading.SetSound(mode,action,sound,symbol);
                          }
//--- Play a sound by its description
   bool                 PlaySoundByDescription(const string sound_description);
```

The method simply calls the **CTrading** class method of the same name we have considered above.

**Reset the trading event flag at the beginning of the trade**
**events checking method:**

```
//+------------------------------------------------------------------+
//| Check trading events                                             |
//+------------------------------------------------------------------+
void CEngine::TradeEventsControl(void)
  {
//--- Initialize trading events' flags
   this.m_is_market_trade_event=false;
   this.m_is_history_trade_event=false;
   this.m_events.SetEvent(false);
//--- Update the lists
   this.m_market.Refresh();
   this.m_history.Refresh();
//--- First launch actions
   if(this.IsFirstStart())
     {
      this.m_last_trade_event=TRADE_EVENT_NO_EVENT;
      return;
     }
//--- Check the changes in the market status and account history
   this.m_is_market_trade_event=this.m_market.IsTradeEvent();
   this.m_is_history_trade_event=this.m_history.IsTradeEvent();

//--- If there is any event, send the lists, the flags and the number of new orders and deals to the event collection, and update it
   int change_total=0;
   CArrayObj* list_changes=this.m_market.GetListChanges();
   if(list_changes!=NULL)
      change_total=list_changes.Total();
   if(this.m_is_history_trade_event || this.m_is_market_trade_event || change_total>0)
     {
      this.m_events.Refresh(this.m_history.GetList(),this.m_market.GetList(),list_changes,this.m_market.GetListControl(),
                            this.m_is_history_trade_event,this.m_is_market_trade_event,
                            this.m_history.NewOrders(),this.m_market.NewPendingOrders(),
                            this.m_market.NewPositions(),this.m_history.NewDeals(),
                            this.m_market.ChangedVolumeValue());
      //--- Receive the last account trading event
      this.m_last_trade_event=this.m_events.GetLastTradeEvent();
     }
  }
//+------------------------------------------------------------------+
```

**The methods of working with a trading class have also changed. Now they also have template**
**parameters:**

```
//--- Open (1) Buy, (2) Sell position
   template<typename SL,typename TP>
   bool                 OpenBuy(const double volume,
                                const string symbol,
                                const ulong magic=ULONG_MAX,
                                SL sl=0,
                                TP tp=0,
                                const string comment=NULL,
                                const ulong deviation=ULONG_MAX);
   template<typename SL,typename TP>
   bool                 OpenSell(const double volume,
                                 const string symbol,
                                 const ulong magic=ULONG_MAX,
                                 SL sl=0,
                                 TP tp=0,
                                 const string comment=NULL,
                                 const ulong deviation=ULONG_MAX);
//--- Modify a position
   template<typename SL,typename TP>
   bool                 ModifyPosition(const ulong ticket,const SL sl=WRONG_VALUE,const TP tp=WRONG_VALUE);
//--- Close a position (1) fully, (2) partially, (3) by an opposite one
   bool                 ClosePosition(const ulong ticket,const string comment=NULL,const ulong deviation=ULONG_MAX);
   bool                 ClosePositionPartially(const ulong ticket,const double volume,const string comment=NULL,const ulong deviation=ULONG_MAX);
   bool                 ClosePositionBy(const ulong ticket,const ulong ticket_by);
//--- Set (1) BuyStop, (2) BuyLimit, (3) BuyStopLimit pending order
   template<typename PR,typename SL,typename TP>
   bool                 PlaceBuyStop(const double volume,
                                     const string symbol,
                                     const PR price,
                                     const SL sl=0,
                                     const TP tp=0,
                                     const ulong magic=ULONG_MAX,
                                     const string comment=NULL,
                                     const datetime expiration=0,
                                     const ENUM_ORDER_TYPE_TIME type_time=ORDER_TIME_GTC);
   template<typename PR,typename SL,typename TP>
   bool                 PlaceBuyLimit(const double volume,
                                     const string symbol,
                                     const PR price,
                                     const SL sl=0,
                                     const TP tp=0,
                                     const ulong magic=ULONG_MAX,
                                     const string comment=NULL,
                                     const datetime expiration=0,
                                     const ENUM_ORDER_TYPE_TIME type_time=ORDER_TIME_GTC);
   template<typename PR,typename PL,typename SL,typename TP>
   bool                 PlaceBuyStopLimit(const double volume,
                                     const string symbol,
                                     const PR price_stop,
                                     const PL price_limit,
                                     const SL sl=0,
                                     const TP tp=0,
                                     const ulong magic=ULONG_MAX,
                                     const string comment=NULL,
                                     const datetime expiration=0,
                                     const ENUM_ORDER_TYPE_TIME type_time=ORDER_TIME_GTC);
//--- Set (1) SellStop, (2) SellLimit, (3) SellStopLimit pending order
   template<typename PR,typename SL,typename TP>
   bool                 PlaceSellStop(const double volume,
                                     const string symbol,
                                     const PR price,
                                     const SL sl=0,
                                     const TP tp=0,
                                     const ulong magic=ULONG_MAX,
                                     const string comment=NULL,
                                     const datetime expiration=0,
                                     const ENUM_ORDER_TYPE_TIME type_time=ORDER_TIME_GTC);
   template<typename PR,typename SL,typename TP>
   bool                 PlaceSellLimit(const double volume,
                                     const string symbol,
                                     const PR price,
                                     const SL sl=0,
                                     const TP tp=0,
                                     const ulong magic=ULONG_MAX,
                                     const string comment=NULL,
                                     const datetime expiration=0,
                                     const ENUM_ORDER_TYPE_TIME type_time=ORDER_TIME_GTC);
   template<typename PR,typename PL,typename SL,typename TP>
   bool                 PlaceSellStopLimit(const double volume,
                                     const string symbol,
                                     const PR price_stop,
                                     const PL price_limit,
                                     const SL sl=0,
                                     const TP tp=0,
                                     const ulong magic=ULONG_MAX,
                                     const string comment=NULL,
                                     const datetime expiration=0,
                                     const ENUM_ORDER_TYPE_TIME type_time=ORDER_TIME_GTC);
//--- Modify a pending order
   template<typename PR,typename SL,typename TP,typename PL>
   bool                 ModifyOrder(const ulong ticket,
                                    const PR price=WRONG_VALUE,
                                    const SL sl=WRONG_VALUE,
                                    const TP tp=WRONG_VALUE,
                                    const PL stoplimit=WRONG_VALUE,
                                    datetime expiration=WRONG_VALUE,
                                    ENUM_ORDER_TYPE_TIME type_time=WRONG_VALUE);
//--- Remove a pending order
   bool                 DeleteOrder(const ulong ticket);
```

**Implementation of trading methods have also been changed according to template**
**data passed to the trading methods of the CTrading class:**

```
//+------------------------------------------------------------------+
//| Open Buy position                                                |
//+------------------------------------------------------------------+
template<typename SL,typename TP>
bool CEngine::OpenBuy(const double volume,const string symbol,const ulong magic=ULONG_MAX,SL sl=0,TP tp=0,const string comment=NULL,const ulong deviation=ULONG_MAX)
  {
   return this.m_trading.OpenBuy(volume,symbol,magic,sl,tp,comment,deviation);
  }
//+------------------------------------------------------------------+
//| Open a Sell position                                             |
//+------------------------------------------------------------------+
template<typename SL,typename TP>
bool CEngine::OpenSell(const double volume,const string symbol,const ulong magic=ULONG_MAX,SL sl=0,TP tp=0,const string comment=NULL,const ulong deviation=ULONG_MAX)
  {
   return this.m_trading.OpenSell(volume,symbol,magic,sl,tp,comment,deviation);
  }
//+------------------------------------------------------------------+
//| Modify a position                                                |
//+------------------------------------------------------------------+
template<typename SL,typename TP>
bool CEngine::ModifyPosition(const ulong ticket,const SL sl=WRONG_VALUE,const TP tp=WRONG_VALUE)
  {
   return this.m_trading.ModifyPosition(ticket,sl,tp);
  }
//+------------------------------------------------------------------+
//| Close a position in full                                         |
//+------------------------------------------------------------------+
bool CEngine::ClosePosition(const ulong ticket,const string comment=NULL,const ulong deviation=ULONG_MAX)
  {
   return this.m_trading.ClosePosition(ticket,comment,deviation);
  }
//+------------------------------------------------------------------+
//| Close a position partially                                       |
//+------------------------------------------------------------------+
bool CEngine::ClosePositionPartially(const ulong ticket,const double volume,const string comment=NULL,const ulong deviation=ULONG_MAX)
  {
   return this.m_trading.ClosePositionPartially(ticket,volume,comment,deviation);
  }
//+------------------------------------------------------------------+
//| Close a position by an opposite one                              |
//+------------------------------------------------------------------+
bool CEngine::ClosePositionBy(const ulong ticket,const ulong ticket_by)
  {
   return this.m_trading.ClosePositionBy(ticket,ticket_by);
  }
//+------------------------------------------------------------------+
//| Place BuyStop pending order                                      |
//+------------------------------------------------------------------+
template<typename PR,typename SL,typename TP>
bool CEngine::PlaceBuyStop(const double volume,
                           const string symbol,
                           const PR price,
                           const SL sl=0,
                           const TP tp=0,
                           const ulong magic=WRONG_VALUE,
                           const string comment=NULL,
                           const datetime expiration=0,
                           const ENUM_ORDER_TYPE_TIME type_time=ORDER_TIME_GTC)
  {
   return this.m_trading.PlaceBuyStop(volume,symbol,price,sl,tp,magic,comment,expiration,type_time);
  }
//+------------------------------------------------------------------+
//| Place BuyLimit pending order                                     |
//+------------------------------------------------------------------+
template<typename PR,typename SL,typename TP>
bool CEngine::PlaceBuyLimit(const double volume,
                            const string symbol,
                            const PR price,
                            const SL sl=0,
                            const TP tp=0,
                            const ulong magic=WRONG_VALUE,
                            const string comment=NULL,
                            const datetime expiration=0,
                            const ENUM_ORDER_TYPE_TIME type_time=ORDER_TIME_GTC)
  {
   return this.m_trading.PlaceBuyLimit(volume,symbol,price,sl,tp,magic,comment,expiration,type_time);
  }
//+------------------------------------------------------------------+
//| Place BuyStopLimit pending order                                 |
//+------------------------------------------------------------------+
template<typename PR,typename PL,typename SL,typename TP>
bool CEngine::PlaceBuyStopLimit(const double volume,
                                const string symbol,
                                const PR price_stop,
                                const PL price_limit,
                                const SL sl=0,
                                const TP tp=0,
                                const ulong magic=WRONG_VALUE,
                                const string comment=NULL,
                                const datetime expiration=0,
                                const ENUM_ORDER_TYPE_TIME type_time=ORDER_TIME_GTC)
  {
   return this.m_trading.PlaceBuyStopLimit(volume,symbol,price_stop,price_limit,sl,tp,magic,comment,expiration,type_time);
  }
//+------------------------------------------------------------------+
//| Place SellStop pending order                                     |
//+------------------------------------------------------------------+
template<typename PR,typename SL,typename TP>
bool CEngine::PlaceSellStop(const double volume,
                            const string symbol,
                            const PR price,
                            const SL sl=0,
                            const TP tp=0,
                            const ulong magic=WRONG_VALUE,
                            const string comment=NULL,
                            const datetime expiration=0,
                            const ENUM_ORDER_TYPE_TIME type_time=ORDER_TIME_GTC)
  {
   return this.m_trading.PlaceSellStop(volume,symbol,price,sl,tp,magic,comment,expiration,type_time);
  }
//+------------------------------------------------------------------+
//| Place SellLimit pending order                                    |
//+------------------------------------------------------------------+
template<typename PR,typename SL,typename TP>
bool CEngine::PlaceSellLimit(const double volume,
                             const string symbol,
                             const PR price,
                             const SL sl=0,
                             const TP tp=0,
                             const ulong magic=WRONG_VALUE,
                             const string comment=NULL,
                             const datetime expiration=0,
                             const ENUM_ORDER_TYPE_TIME type_time=ORDER_TIME_GTC)
  {
   return this.m_trading.PlaceSellLimit(volume,symbol,price,sl,tp,magic,comment,expiration,type_time);
  }
//+------------------------------------------------------------------+
//| Place SellStopLimit pending order                                |
//+------------------------------------------------------------------+
template<typename PR,typename PL,typename SL,typename TP>
bool CEngine::PlaceSellStopLimit(const double volume,
                                 const string symbol,
                                 const PR price_stop,
                                 const PL price_limit,
                                 const SL sl=0,
                                 const TP tp=0,
                                 const ulong magic=WRONG_VALUE,
                                 const string comment=NULL,
                                 const datetime expiration=0,
                                 const ENUM_ORDER_TYPE_TIME type_time=ORDER_TIME_GTC)
  {
   return this.m_trading.PlaceSellStopLimit(volume,symbol,price_stop,price_limit,sl,tp,magic,comment,expiration,type_time);
  }
//+------------------------------------------------------------------+
//| Modify a pending order                                           |
//+------------------------------------------------------------------+
template<typename PR,typename SL,typename TP,typename PL>
bool CEngine::ModifyOrder(const ulong ticket,
                          const PR price=WRONG_VALUE,
                          const SL sl=WRONG_VALUE,
                          const TP tp=WRONG_VALUE,
                          const PL stoplimit=WRONG_VALUE,
                          datetime expiration=WRONG_VALUE,
                          ENUM_ORDER_TYPE_TIME type_time=WRONG_VALUE)
  {
   return this.m_trading.ModifyOrder(ticket,price,sl,tp,stoplimit,expiration,type_time);
  }
//+------------------------------------------------------------------+
//| Remove a pending order                                           |
//+------------------------------------------------------------------+
bool CEngine::DeleteOrder(const ulong ticket)
  {
   return this.m_trading.DeleteOrder(ticket);
  }
//+------------------------------------------------------------------+
```

This concludes the improvement of trading classes and adjusting the obtaining of trading events.

### Testing

To test the current library version, let's use [the EA from the previous \\
article](https://www.mql5.com/en/articles/7258#node04) and save it to \\MQL5\\Experts\\TestDoEasy\ **Part23\** under the name **TestDoEasyPart23.mq5**.

Let's put things in order a bit. Move all actions related to the library initialization to the separate function OnInitDoEasy():

```
//+------------------------------------------------------------------+
//| Initializing DoEasy library                                      |
//+------------------------------------------------------------------+
void OnInitDoEasy()
  {
//--- Check if working with the full list is selected
   used_symbols_mode=InpModeUsedSymbols;
   if((ENUM_SYMBOLS_MODE)used_symbols_mode==SYMBOLS_MODE_ALL)
     {
      int total=SymbolsTotal(false);
      string ru_n="\nКоличество символов на сервере "+(string)total+".\nМаксимальное количество: "+(string)SYMBOLS_COMMON_TOTAL+" символов.";
      string en_n="\nNumber of symbols on server "+(string)total+".\nMaximum number: "+(string)SYMBOLS_COMMON_TOTAL+" symbols.";
      string caption=TextByLanguage("Внимание!","Attention!");
      string ru="Выбран режим работы с полным списком.\nВ этом режиме первичная подготовка списка коллекции символов может занять длительное время."+ru_n+"\nПродолжить?\n\"Нет\" - работа с текущим символом \""+Symbol()+"\"";
      string en="Full list mode selected.\nIn this mode, the initial preparation of the collection symbols list may take a long time."+en_n+"\nContinue?\n\"No\" - working with the current symbol \""+Symbol()+"\"";
      string message=TextByLanguage(ru,en);
      int flags=(MB_YESNO | MB_ICONWARNING | MB_DEFBUTTON2);
      int mb_res=MessageBox(message,caption,flags);
      switch(mb_res)
        {
         case IDNO :
           used_symbols_mode=SYMBOLS_MODE_CURRENT;
           break;
         default:
           break;
        }
     }
//--- Fill in the array of used symbols
   used_symbols=InpUsedSymbols;
   CreateUsedSymbolsArray((ENUM_SYMBOLS_MODE)used_symbols_mode,used_symbols,array_used_symbols);

//--- Set the type of the used symbol list in the symbol collection
   engine.SetUsedSymbols(array_used_symbols);
//--- Displaying the selected mode of working with the symbol object collection
   Print(engine.ModeSymbolsListDescription(),TextByLanguage(". Number of used symbols: ",". Number of symbols used: "),engine.GetSymbolsCollectionTotal());

//--- Create resource text files
   engine.CreateFile(FILE_TYPE_WAV,"sound_array_coin_01",TextByLanguage("Звук упавшей монетки 1","Falling coin 1"),sound_array_coin_01);
   engine.CreateFile(FILE_TYPE_WAV,"sound_array_coin_02",TextByLanguage("Звук упавших монеток","Falling coins"),sound_array_coin_02);
   engine.CreateFile(FILE_TYPE_WAV,"sound_array_coin_03",TextByLanguage("Звук монеток","Coins"),sound_array_coin_03);
   engine.CreateFile(FILE_TYPE_WAV,"sound_array_coin_04",TextByLanguage("Звук упавшей монетки 2","Falling coin 2"),sound_array_coin_04);
   engine.CreateFile(FILE_TYPE_WAV,"sound_array_click_01",TextByLanguage("Звук щелчка по кнопке 1","Button click 1"),sound_array_click_01);
   engine.CreateFile(FILE_TYPE_WAV,"sound_array_click_02",TextByLanguage("Звук щелчка по кнопке 2","Button click 2"),sound_array_click_02);
   engine.CreateFile(FILE_TYPE_WAV,"sound_array_click_03",TextByLanguage("Звук щелчка по кнопке 3","Button click 3"),sound_array_click_03);
   engine.CreateFile(FILE_TYPE_WAV,"sound_array_cash_machine_01",TextByLanguage("Звук кассового аппарата","Cash machine"),sound_array_cash_machine_01);
   engine.CreateFile(FILE_TYPE_BMP,"img_array_spot_green",TextByLanguage("Изображение \"Зелёный светодиод\"","Image \"Green Spot lamp\""),img_array_spot_green);
   engine.CreateFile(FILE_TYPE_BMP,"img_array_spot_red",TextByLanguage("Изображение \"Красный светодиод\"","Image \"Red Spot lamp\""),img_array_spot_red);

//--- Pass all existing collections to the trading class
   engine.TradingOnInit();
//--- Set synchronous passing of orders for all used symbols
   engine.TradingSetAsyncMode(false);
//--- Set standard sounds for trading objects of all used symbols
   engine.SetSoundsStandart();
//--- Set the general flag of using sounds
   engine.SetUseSounds(InpUseSounds);

//--- Set controlled values for symbols
   //--- Get the list of all collection symbols
   CArrayObj *list=engine.GetListAllUsedSymbols();
   if(list!=NULL && list.Total()!=0)
     {
      //--- In a loop by the list, set the necessary values for tracked symbol properties
      //--- By default, the LONG_MAX value is set to all properties, which means "Do not track this property"
      //--- It can be enabled or disabled (by setting the value less than LONG_MAX or vice versa - set the LONG_MAX value) at any time and anywhere in the program
      for(int i=0;i<list.Total();i++)
        {
         CSymbol* symbol=list.At(i);
         if(symbol==NULL)
            continue;
         //--- Set control of the symbol price increase by 100 points
         symbol.SetControlBidInc(100*symbol.Point());
         //--- Set control of the symbol price decrease by 100 points
         symbol.SetControlBidDec(100*symbol.Point());
         //--- Set control of the symbol spread increase by 40 points
         symbol.SetControlSpreadInc(40);
         //--- Set control of the symbol spread decrease by 40 points
         symbol.SetControlSpreadDec(40);
         //--- Set control of the current spread by the value of 40 points
         symbol.SetControlSpreadLevel(40);
        }
     }
//--- Set controlled values for the current account
   CAccount* account=engine.GetAccountCurrent();
   if(account!=NULL)
     {
      //--- Set control of the profit increase to 10
      account.SetControlledValueINC(ACCOUNT_PROP_PROFIT,10.0);
      //--- Set control of the funds increase to 15
      account.SetControlledValueINC(ACCOUNT_PROP_EQUITY,15.0);
      //--- Set profit control level to 20
      account.SetControlledValueLEVEL(ACCOUNT_PROP_PROFIT,20.0);
     }
  }
//+------------------------------------------------------------------+
```

Previously, the entire contents of the function was written in the EA's OnInit() handler. Now, when moving the library initialization actions to the
separate function (where you can set all you need for the EA operation), the OnInit() handler has become cleaner and looks more convenient:

```
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//--- Calling the function displays the list of enumeration constants in the journal
//--- (the list is set in the strings 22 and 25 of the DELib.mqh file) for checking the constants validity
   //EnumNumbersTest();

//--- Set EA global variables
   prefix=MQLInfoString(MQL_PROGRAM_NAME)+"_";
   for(int i=0;i<TOTAL_BUTT;i++)
     {
      butt_data[i].name=prefix+EnumToString((ENUM_BUTTONS)i);
      butt_data[i].text=EnumToButtText((ENUM_BUTTONS)i);
     }
   lot=NormalizeLot(Symbol(),fmax(InpLots,MinimumLots(Symbol())*2.0));
   magic_number=InpMagic;
   stoploss=InpStopLoss;
   takeprofit=InpTakeProfit;
   distance_pending=InpDistance;
   distance_stoplimit=InpDistanceSL;
   slippage=InpSlippage;
   trailing_stop=InpTrailingStop*Point();
   trailing_step=InpTrailingStep*Point();
   trailing_start=InpTrailingStart;
   stoploss_to_modify=InpStopLossModify;
   takeprofit_to_modify=InpTakeProfitModify;

//--- Initialize DoEasy library
   OnInitDoEasy();

//--- Check and remove remaining EA graphical objects
   if(IsPresentObects(prefix))
      ObjectsDeleteAll(0,prefix);

//--- Create the button panel
   if(!CreateButtons(InpButtShiftX,InpButtShiftY))
      return INIT_FAILED;
//--- Set trailing activation button status
   ButtonState(butt_data[TOTAL_BUTT-1].name,trailing_on);

//--- Check playing a standard sound by macro substitution and a custom sound by description
   engine.PlaySoundByDescription(SND_OK);
   Sleep(600);
   engine.PlaySoundByDescription(TextByLanguage("Звук упавшей монетки 2","Falling coin 2"));

//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
```

Working with all library events in the OnDoEasyEvent() function is now implemented in the same way:

```
//+------------------------------------------------------------------+
//| Handling DoEasy library events                                   |
//+------------------------------------------------------------------+
void OnDoEasyEvent(const int id,
                   const long &lparam,
                   const double &dparam,
                   const string &sparam)
  {
   int idx=id-CHARTEVENT_CUSTOM;
   string event="::"+string(idx);

//--- Retrieve (1) event time milliseconds, (2) reason and (3) source from lparam, as well as (4) set the exact event time
   ushort msc=engine.EventMSC(lparam);
   ushort reason=engine.EventReason(lparam);
   ushort source=engine.EventSource(lparam);
   long time=TimeCurrent()*1000+msc;

//--- Handling symbol events
   if(source==COLLECTION_SYMBOLS_ID)
     {
      CSymbol *symbol=engine.GetSymbolObjByName(sparam);
      if(symbol==NULL)
         return;
      //--- Number of decimal places in the event value - in case of a 'long' event, it is 0, otherwise - Digits() of a symbol
      int digits=(idx<SYMBOL_PROP_INTEGER_TOTAL ? 0 : symbol.Digits());
      //--- Event text description
      string id_descr=(idx<SYMBOL_PROP_INTEGER_TOTAL ? symbol.GetPropertyDescription((ENUM_SYMBOL_PROP_INTEGER)idx) : symbol.GetPropertyDescription((ENUM_SYMBOL_PROP_DOUBLE)idx));
      //--- Property change text value
      string value=DoubleToString(dparam,digits);

      //--- Check event reasons and display its description in the journal
      if(reason==BASE_EVENT_REASON_INC)
        {
         Print(symbol.EventDescription(idx,(ENUM_BASE_EVENT_REASON)reason,source,value,id_descr,digits));
        }
      if(reason==BASE_EVENT_REASON_DEC)
        {
         Print(symbol.EventDescription(idx,(ENUM_BASE_EVENT_REASON)reason,source,value,id_descr,digits));
        }
      if(reason==BASE_EVENT_REASON_MORE_THEN)
        {
         Print(symbol.EventDescription(idx,(ENUM_BASE_EVENT_REASON)reason,source,value,id_descr,digits));
        }
      if(reason==BASE_EVENT_REASON_LESS_THEN)
        {
         Print(symbol.EventDescription(idx,(ENUM_BASE_EVENT_REASON)reason,source,value,id_descr,digits));
        }
      if(reason==BASE_EVENT_REASON_EQUALS)
        {
         Print(symbol.EventDescription(idx,(ENUM_BASE_EVENT_REASON)reason,source,value,id_descr,digits));
        }
     }

//--- Handling account events
   else if(source==COLLECTION_ACCOUNT_ID)
     {
      CAccount *account=engine.GetAccountCurrent();
      if(account==NULL)
         return;
      //--- Number of decimal places in the event value - in case of a 'long' event, it is 0, otherwise - Digits() of a symbol
      int digits=int(idx<ACCOUNT_PROP_INTEGER_TOTAL ? 0 : account.CurrencyDigits());
      //--- Event text description
      string id_descr=(idx<ACCOUNT_PROP_INTEGER_TOTAL ? account.GetPropertyDescription((ENUM_ACCOUNT_PROP_INTEGER)idx) : account.GetPropertyDescription((ENUM_ACCOUNT_PROP_DOUBLE)idx));
      //--- Property change text value
      string value=DoubleToString(dparam,digits);

      //--- Checking event reasons and handling the increase of funds by a specified value,

      //--- In case of a property value increase
      if(reason==BASE_EVENT_REASON_INC)
        {
         //--- Display an event in the journal
         Print(account.EventDescription(idx,(ENUM_BASE_EVENT_REASON)reason,source,value,id_descr,digits));
         //--- if this is an equity increase
         if(idx==ACCOUNT_PROP_EQUITY)
           {
            //--- Get the list of all open positions
            CArrayObj* list_positions=engine.GetListMarketPosition();
            //--- Select positions with the profit exceeding zero
            list_positions=CSelect::ByOrderProperty(list_positions,ORDER_PROP_PROFIT_FULL,0,MORE);
            if(list_positions!=NULL)
              {
               //--- Sort the list by profit considering commission and swap
               list_positions.Sort(SORT_BY_ORDER_PROFIT_FULL);
               //--- Get the position index with the highest profit
               int index=CSelect::FindOrderMax(list_positions,ORDER_PROP_PROFIT_FULL);
               if(index>WRONG_VALUE)
                 {
                  COrder* position=list_positions.At(index);
                  if(position!=NULL)
                    {
                     //--- Get a ticket of a position with the highest profit and close the position by a ticket
                     engine.ClosePosition(position.Ticket());
                    }
                 }
              }
           }
        }
      //--- Other events are simply displayed in the journal
      if(reason==BASE_EVENT_REASON_DEC)
        {
         Print(account.EventDescription(idx,(ENUM_BASE_EVENT_REASON)reason,source,value,id_descr,digits));
        }
      if(reason==BASE_EVENT_REASON_MORE_THEN)
        {
         Print(account.EventDescription(idx,(ENUM_BASE_EVENT_REASON)reason,source,value,id_descr,digits));
        }
      if(reason==BASE_EVENT_REASON_LESS_THEN)
        {
         Print(account.EventDescription(idx,(ENUM_BASE_EVENT_REASON)reason,source,value,id_descr,digits));
        }
      if(reason==BASE_EVENT_REASON_EQUALS)
        {
         Print(account.EventDescription(idx,(ENUM_BASE_EVENT_REASON)reason,source,value,id_descr,digits));
        }
     }

//--- Handling market watch window events
   else if(idx>MARKET_WATCH_EVENT_NO_EVENT && idx<SYMBOL_EVENTS_NEXT_CODE)
     {
      //--- Market Watch window event
      string descr=engine.GetMWEventDescription((ENUM_MW_EVENT)idx);
      string name=(idx==MARKET_WATCH_EVENT_SYMBOL_SORT ? "" : ": "+sparam);
      Print(TimeMSCtoString(lparam)," ",descr,name);
     }

//--- Handling trading events
   else if(idx>TRADE_EVENT_NO_EVENT && idx<TRADE_EVENTS_NEXT_CODE)
     {
      Print(DFUN,engine.GetLastTradeEventDescription());
     }
  }
//+------------------------------------------------------------------+
```

Here all is divided into blocks according to an arrived event. Working
with trading events is represented by displaying the name of the last event in
the journal. If you need to handle a certain event, here you can find out what the event is and decide on how to handle it. You can do that
right here or define a separate flag for each trading event to reset trading event flags there, while handling them anywhere in the program.

The OnDoEasyEvent() function is called from the OnChartEvent() EA handler when working **outside** the tester.


When working in the tester, the EventsHandling() function is called from OnTick():

```
//+------------------------------------------------------------------+
//| Working with events in the tester                                |
//+------------------------------------------------------------------+
void EventsHandling(void)
  {
//--- If a trading event is present
   if(engine.IsTradeEvent())
     {
      long lparam=0;
      double dparam=0;
      string sparam="";
      OnDoEasyEvent(CHARTEVENT_CUSTOM+engine.LastTradeEvent(),lparam,dparam,sparam);
     }
//--- If there is an account event
   if(engine.IsAccountsEvent())
     {
      //--- Get the list of all account events occurred simultaneously
      CArrayObj* list=engine.GetListAccountEvents();
      if(list!=NULL)
        {
         //--- Get the next event in a loop
         int total=list.Total();
         for(int i=0;i<total;i++)
           {
            //--- take an event from the list
            CEventBaseObj *event=list.At(i);
            if(event==NULL)
               continue;
            //--- Send an event to the event handler
            long lparam=event.LParam();
            double dparam=event.DParam();
            string sparam=event.SParam();
            OnDoEasyEvent(CHARTEVENT_CUSTOM+event.ID(),lparam,dparam,sparam);
           }
        }
     }
//--- If there is a symbol collection event
   if(engine.IsSymbolsEvent())
     {
      //--- Get the list of all symbol events occurred simultaneously
      CArrayObj* list=engine.GetListSymbolsEvents();
      if(list!=NULL)
        {
         //--- Get the next event in a loop
         int total=list.Total();
         for(int i=0;i<total;i++)
           {
            //--- take an event from the list
            CEventBaseObj *event=list.At(i);
            if(event==NULL)
               continue;
            //--- Send an event to the event handler
            long lparam=event.LParam();
            double dparam=event.DParam();
            string sparam=event.SParam();

            OnDoEasyEvent(CHARTEVENT_CUSTOM+event.ID(),lparam,dparam,sparam);
           }
        }
     }
  }
//+------------------------------------------------------------------+
```

The function allows you to view the lists of appropriate events and fill in parameters of the event which is then sent to the OnDoEasyEvent()
function. Thus, we arrange the work with events in the OnDoEasyEvent() function regardless of where the EA is launched (in the tester or
not). If it is launched in the tester, events are handled from OnTick(), if it is launched outside the tester, events are handled from
OnChartEvent().

Thus, the OnTick() function now looks as follows:

```
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//--- If working in the tester
   if(MQLInfoInteger(MQL_TESTER))
     {
      engine.OnTimer();       // Working in the timer
      PressButtonsControl();  // Button pressing control
      EventsHandling();       // Working with events
     }
//--- If the trailing flag is set
   if(trailing_on)
     {
      TrailingPositions();    // Trailing positions
      TrailingOrders();       // Trailing of pending orders
     }
  }
//+------------------------------------------------------------------+
```

To check the operation of methods controlling the validity of values in the trading order parameters, we need to remove the auto correction
of invalid values from the EA.

**In the PressButtonEvents() function handling button pressing, remove everything related to correcting trading order parameter**
**values:**

```
//+------------------------------------------------------------------+
//| Handle pressing the buttons                                      |
//+------------------------------------------------------------------+
void PressButtonEvents(const string button_name)
  {
   string comment="";
   //--- Convert button name into its string ID
   string button=StringSubstr(button_name,StringLen(prefix));
   //--- If the button is pressed
   if(ButtonState(button_name))
     {
      //--- If the BUTT_BUY button is pressed: Open Buy position
      if(button==EnumToString(BUTT_BUY))
        {
         //--- Get the correct StopLoss and TakeProfit prices relative to StopLevel
         double sl=stoploss;//CorrectStopLoss(Symbol(),ORDER_TYPE_BUY,0,stoploss);
         double tp=takeprofit;//CorrectTakeProfit(Symbol(),ORDER_TYPE_BUY,0,takeprofit);
         //--- Open Buy position
         engine.OpenBuy(lot,Symbol(),magic_number,sl,tp);   // No comment - the default comment is to be set
        }
      //--- If the BUTT_BUY_LIMIT button is pressed: Place BuyLimit
      else if(button==EnumToString(BUTT_BUY_LIMIT))
        {
         //--- Get correct order placement relative to StopLevel
         double price_set=CorrectPricePending(Symbol(),ORDER_TYPE_BUY_LIMIT,distance_pending);
         //--- Get correct StopLoss and TakeProfit prices relative to the order placement level considering StopLevel
         double sl=stoploss;//CorrectStopLoss(Symbol(),ORDER_TYPE_BUY_LIMIT,price_set,stoploss);
         double tp=takeprofit;//CorrectTakeProfit(Symbol(),ORDER_TYPE_BUY_LIMIT,price_set,takeprofit);
         //--- Set BuyLimit order
         engine.PlaceBuyLimit(lot,Symbol(),price_set,sl,tp,magic_number,TextByLanguage("Отложенный BuyLimit","Pending BuyLimit order"));
        }
      //--- If the BUTT_BUY_STOP button is pressed: Set BuyStop
      else if(button==EnumToString(BUTT_BUY_STOP))
        {
         //--- Get correct order placement relative to StopLevel
         double price_set=CorrectPricePending(Symbol(),ORDER_TYPE_BUY_STOP,distance_pending);
         //--- Get correct StopLoss and TakeProfit prices relative to the order placement level considering StopLevel
         double sl=CorrectStopLoss(Symbol(),ORDER_TYPE_BUY_STOP,price_set,stoploss);
         double tp=CorrectTakeProfit(Symbol(),ORDER_TYPE_BUY_STOP,price_set,takeprofit);
         //--- Set BuyStop order
         engine.PlaceBuyStop(lot,Symbol(),price_set,sl,tp,magic_number,TextByLanguage("Отложенный BuyStop","Pending BuyStop order"));
        }
      //--- If the BUTT_BUY_STOP_LIMIT button is pressed: Set BuyStopLimit
      else if(button==EnumToString(BUTT_BUY_STOP_LIMIT))
        {
         //--- Get the correct BuyStop order placement price relative to StopLevel
         double price_set_stop=CorrectPricePending(Symbol(),ORDER_TYPE_BUY_STOP,distance_pending);
         //--- Calculate BuyLimit order price relative to BuyStop level considering StopLevel
         double price_set_limit=CorrectPricePending(Symbol(),ORDER_TYPE_BUY_LIMIT,distance_stoplimit,price_set_stop);
         //--- Get correct StopLoss and TakeProfit prices relative to the order placement level considering StopLevel
         double sl=stoploss;//CorrectStopLoss(Symbol(),ORDER_TYPE_BUY_STOP,price_set_limit,stoploss);
         double tp=takeprofit;//CorrectTakeProfit(Symbol(),ORDER_TYPE_BUY_STOP,price_set_limit,takeprofit);
         //--- Set BuyStopLimit order
         engine.PlaceBuyStopLimit(lot,Symbol(),price_set_stop,price_set_limit,sl,tp,magic_number,TextByLanguage("Отложенный BuyStopLimit","Pending BuyStopLimit order"));
        }
      //--- If the BUTT_SELL button is pressed: Open Sell position
      else if(button==EnumToString(BUTT_SELL))
        {
         //--- Get the correct StopLoss and TakeProfit prices relative to StopLevel
         double sl=stoploss;//CorrectStopLoss(Symbol(),ORDER_TYPE_SELL,0,stoploss);
         double tp=takeprofit;//CorrectTakeProfit(Symbol(),ORDER_TYPE_SELL,0,takeprofit);
         //--- Open Sell position
         engine.OpenSell(lot,Symbol(),magic_number,sl,tp);  // No comment - the default comment is to be set
        }
      //--- If the BUTT_SELL_LIMIT button is pressed: Set SellLimit
      else if(button==EnumToString(BUTT_SELL_LIMIT))
        {
         //--- Get correct order placement relative to StopLevel
         double price_set=CorrectPricePending(Symbol(),ORDER_TYPE_SELL_LIMIT,distance_pending);
         //--- Get correct StopLoss and TakeProfit prices relative to the order placement level considering StopLevel
         double sl=stoploss;//CorrectStopLoss(Symbol(),ORDER_TYPE_SELL_LIMIT,price_set,stoploss);
         double tp=takeprofit;//CorrectTakeProfit(Symbol(),ORDER_TYPE_SELL_LIMIT,price_set,takeprofit);
         //--- Set SellLimit order
         engine.PlaceSellLimit(lot,Symbol(),price_set,sl,tp,magic_number,TextByLanguage("Отложенный SellLimit","Pending order SellLimit"));
        }
      //--- If the BUTT_SELL_STOP button is pressed: Set SellStop
      else if(button==EnumToString(BUTT_SELL_STOP))
        {
         //--- Get correct order placement relative to StopLevel
         double price_set=CorrectPricePending(Symbol(),ORDER_TYPE_SELL_STOP,distance_pending);
         //--- Get correct StopLoss and TakeProfit prices relative to the order placement level considering StopLevel
         double sl=stoploss;//CorrectStopLoss(Symbol(),ORDER_TYPE_SELL_STOP,price_set,stoploss);
         double tp=takeprofit;//CorrectTakeProfit(Symbol(),ORDER_TYPE_SELL_STOP,price_set,takeprofit);
         //--- Set SellStop order
         engine.PlaceSellStop(lot,Symbol(),price_set,sl,tp,magic_number,TextByLanguage("Отложенный SellStop","Pending SellStop order"));
        }
      //--- If the BUTT_SELL_STOP_LIMIT button is pressed: Set SellStopLimit
      else if(button==EnumToString(BUTT_SELL_STOP_LIMIT))
        {
         //--- Get the correct SellStop order price relative to StopLevel
         double price_set_stop=CorrectPricePending(Symbol(),ORDER_TYPE_SELL_STOP,distance_pending);
         //--- Calculate SellLimit order price relative to SellStop level considering StopLevel
         double price_set_limit=CorrectPricePending(Symbol(),ORDER_TYPE_SELL_LIMIT,distance_stoplimit,price_set_stop);
         //--- Get correct StopLoss and TakeProfit prices relative to the order placement level considering StopLevel
         double sl=stoploss;//CorrectStopLoss(Symbol(),ORDER_TYPE_SELL_STOP,price_set_limit,stoploss);
         double tp=takeprofit;//CorrectTakeProfit(Symbol(),ORDER_TYPE_SELL_STOP,price_set_limit,takeprofit);
         //--- Set SellStopLimit order
         engine.PlaceSellStopLimit(lot,Symbol(),price_set_stop,price_set_limit,sl,tp,magic_number,TextByLanguage("Отложенный SellStopLimit","Pending SellStopLimit order"));
        }
      //--- If the BUTT_CLOSE_BUY button is pressed: Close Buy with the maximum profit
      else if(button==EnumToString(BUTT_CLOSE_BUY))
        {
```

In fact, we only need to leave the call of trading methods with the parameters set in the EA settings:

```
//+------------------------------------------------------------------+
//| Handle pressing the buttons                                      |
//+------------------------------------------------------------------+
void PressButtonEvents(const string button_name)
  {
   string comment="";
   //--- Convert button name into its string ID
   string button=StringSubstr(button_name,StringLen(prefix));
   //--- If the button is pressed
   if(ButtonState(button_name))
     {
      //--- If the BUTT_BUY button is pressed: Open Buy position
      if(button==EnumToString(BUTT_BUY))
        {
         //--- Open Buy position
         engine.OpenBuy(lot,Symbol(),magic_number,stoploss,takeprofit);   // No comment - the default comment is to be set
        }
      //--- If the BUTT_BUY_LIMIT button is pressed: Place BuyLimit
      else if(button==EnumToString(BUTT_BUY_LIMIT))
        {
         //--- Set BuyLimit order
         engine.PlaceBuyLimit(lot,Symbol(),distance_pending,stoploss,takeprofit,magic_number,TextByLanguage("Отложенный BuyLimit","Pending BuyLimit order"));
        }
      //--- If the BUTT_BUY_STOP button is pressed: Set BuyStop
      else if(button==EnumToString(BUTT_BUY_STOP))
        {
         //--- Set BuyStop order
         engine.PlaceBuyStop(lot,Symbol(),distance_pending,stoploss,takeprofit,magic_number,TextByLanguage("Отложенный BuyStop","Pending BuyStop order"));
        }
      //--- If the BUTT_BUY_STOP_LIMIT button is pressed: Set BuyStopLimit
      else if(button==EnumToString(BUTT_BUY_STOP_LIMIT))
        {
         //--- Set BuyStopLimit order
         engine.PlaceBuyStopLimit(lot,Symbol(),distance_pending,distance_stoplimit,stoploss,takeprofit,magic_number,TextByLanguage("Отложенный BuyStopLimit","Pending BuyStopLimit order"));
        }
      //--- If the BUTT_SELL button is pressed: Open Sell position
      else if(button==EnumToString(BUTT_SELL))
        {
         //--- Open Sell position
         engine.OpenSell(lot,Symbol(),magic_number,stoploss,takeprofit);  // No comment - the default comment is to be set
        }
      //--- If the BUTT_SELL_LIMIT button is pressed: Set SellLimit
      else if(button==EnumToString(BUTT_SELL_LIMIT))
        {
         //--- Set SellLimit order
         engine.PlaceSellLimit(lot,Symbol(),distance_pending,stoploss,takeprofit,magic_number,TextByLanguage("Отложенный SellLimit","Pending SellLimit order"));
        }
      //--- If the BUTT_SELL_STOP button is pressed: Set SellStop
      else if(button==EnumToString(BUTT_SELL_STOP))
        {
         //--- Set SellStop order
         engine.PlaceSellStop(lot,Symbol(),distance_pending,stoploss,takeprofit,magic_number,TextByLanguage("Отложенный SellStop","Pending SellStop order"));
        }
      //--- If the BUTT_SELL_STOP_LIMIT button is pressed: Set SellStopLimit
      else if(button==EnumToString(BUTT_SELL_STOP_LIMIT))
        {
         //--- Set SellStopLimit order
         engine.PlaceSellStopLimit(lot,Symbol(),distance_pending,distance_stoplimit,stoploss,takeprofit,magic_number,TextByLanguage("Отложенный SellStopLimit","Pending order SellStopLimit"));
        }
      //--- If the BUTT_CLOSE_BUY button is pressed: Close Buy with the maximum profit
      else if(button==EnumToString(BUTT_CLOSE_BUY))
        {
```

Thus, the task of checking the parameters validity is transferred to the library, while the necessary orders are sent in the EA. I will gradually
add all the necessary functionality for comfortable work.

The full EA code can be viewed in the files attached to the article.

Compile the EA and launch it in the tester, while preliminarily setting **Lots** to 10, as well as **StopLoss in points**
and

**TakeProfit in points** to 1 point each in the parameters:

![](https://c.mql5.com/2/37/terminal64_Nrn0401U8K.png)

Thus, we attempt to open a position with an invalid lot size, so that the funds for opening it are insufficient, and try violating the minimum stop
order distance regulated by symbol's StopLevel parameter:

![](https://c.mql5.com/2/37/0ohg5VoyZa.gif)

The EA displays two errors in the journal — "Not enough money to perform trading operation" and "StopLoss values violate the StopLevel
parameter requirements". We have also set TakeProfit to one point. Why does the EA display no info of that error as well? Actually, there is no
error here since

[placing TakeProfit and StopLoss levels performed within the \\
minimum SYMBOL\_TRADE\_STOPS\_LEVEL](https://www.mql5.com/en/articles/2555#invalid_SL_TP_for_position) does not violate the rule:

TakeProfit and StopLoss levels should be compared to the current price for performing the opposite operation

- Buying is done at the Ask price — the TakeProfit and StopLoss levels should
be compared to the Bid price.
- Selling is done at the Bid price — the TakeProfit and StopLoss levels should
be compared to the Ask price.

| Buying is done at the Ask price | Selling is done at the Bid price |
| --- | --- |
| TakeProfit >= Bid<br> StopLoss <= Bid | TakeProfit <= Ask<br> StopLoss >= Ask |

Since we pressed the Buy position button, the close price for it is Bid, while the open price is Ask. Stop order levels are set based on the open
price (here it is Ask). Thus, TakeProfit is at Ask+1 point, while StopLoss is at Ask-1 point. This means TakeProfit is above the Bid price the
allowed distance is calculated from, while StopLoss would be within the requirements only in case of a zero spread. Since the spread is
around ten points at the time of opening, we fall into the StopLoss minimum distance limitations.

### What's next?

In the next article, we will start implementing handling errors when sending incorrect trading orders, as well as errors returned by the
trade server.

All files of the current version of the library are attached below together with the test EA files for you to test and download.

Leave
your questions, comments and suggestions in the comments.

[Back to contents](https://www.mql5.com/en/articles/7286#node00)

**Previous articles within the series:**

[Part 1. Concept, data management](https://www.mql5.com/en/articles/5654)

[Part 2. Collection of historical orders and deals](https://www.mql5.com/en/articles/5669)

[Part \\
3\. Collection of market orders and positions, arranging the search](https://www.mql5.com/en/articles/5687)

[Part 4. \\
Trading events. Concept](https://www.mql5.com/en/articles/5724)

[Part 5. Classes and collection of trading events. \\
Sending events to the program](https://www.mql5.com/en/articles/6211)

[Part 6. Netting account events](https://www.mql5.com/en/articles/6383)

[Part \\
7\. StopLimit order activation events, preparing the functionality for order and position modification events](https://www.mql5.com/en/articles/6482)

[Part \\
8\. Order and position modification events](https://www.mql5.com/en/articles/6595)

[Part 9. Compatibility with MQL4 - \\
Preparing data](https://www.mql5.com/en/articles/6651)

[Part 10. Compatibility with MQL4 - Events of opening a position and \\
activating pending orders](https://www.mql5.com/en/articles/6767)

[Part 11. Compatibility with MQL4 - Position closure \\
events](https://www.mql5.com/en/articles/6921)

[Part 12. Account object class and account object collection](https://www.mql5.com/en/articles/6952)

[Part 13. Account object events](https://www.mql5.com/en/articles/6995)

[Part \\
14\. Symbol object](https://www.mql5.com/en/articles/7014)

[Part 15. Symbol object collection](https://www.mql5.com/en/articles/7041)

[Part \\
16\. Symbol collection events](https://www.mql5.com/en/articles/7071)

[Part 17. Interactivity of library objects](https://www.mql5.com/en/articles/7124)

[Part 18. Interactivity of account and any other library objects](https://www.mql5.com/en/articles/7149)

[Part \\
19\. Class of library messages](https://www.mql5.com/en/articles/7176)

[Part 20. Creating and storing program resources](https://www.mql5.com/en/articles/7195)

[Part 21. Trading classes - Base cross-platform trading object](https://www.mql5.com/en/articles/7229)

[Part \\
22\. Trading classes - Base trading class, verification of limitations](https://www.mql5.com/en/articles/7258)

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/7286](https://www.mql5.com/ru/articles/7286)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/7286.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/7286/mql5.zip "Download MQL5.zip")(3589.61 KB)

[MQL4.zip](https://www.mql5.com/en/articles/download/7286/mql4.zip "Download MQL4.zip")(3589.62 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Tables in the MVC Paradigm in MQL5: Customizable and sortable table columns](https://www.mql5.com/en/articles/19979)
- [How to publish code to CodeBase: A practical guide](https://www.mql5.com/en/articles/19441)
- [Tables in the MVC Paradigm in MQL5: Integrating the Model Component into the View Component](https://www.mql5.com/en/articles/19288)
- [The View and Controller components for tables in the MQL5 MVC paradigm: Resizable elements](https://www.mql5.com/en/articles/18941)
- [The View and Controller components for tables in the MQL5 MVC paradigm: Containers](https://www.mql5.com/en/articles/18658)
- [The View and Controller components for tables in the MQL5 MVC paradigm: Simple controls](https://www.mql5.com/en/articles/18221)
- [The View component for tables in the MQL5 MVC paradigm: Base graphical element](https://www.mql5.com/en/articles/17960)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/330468)**
(35)


![wolf1210](https://c.mql5.com/avatar/avatar_na2.png)

**[wolf1210](https://www.mql5.com/en/users/wolf1210)**
\|
20 Jan 2020 at 20:49

Hello Artiom

I write to you not knowing how to go to admin. My problem is this: I havean indicator which is giving fine Signals but only when I refresh the TF in use. If I do Nothing the siganls are not good. But this Refreshing should be done automatically. Do you have an idea? or how can I find it in the codebase? There is Nothing for a search, lol.

Thank you very much

Wolf1210

![Artyom Trishkin](https://c.mql5.com/avatar/2022/7/62C4775C-ABD6.jpg)

**[Artyom Trishkin](https://www.mql5.com/en/users/artmedia70)**
\|
20 Jan 2020 at 21:24

**wolf1210 :**

Hello Artiom

I write to you not knowing how to go to admin. My problem is this: I havean indicator which is giving fine Signals but only when I refresh the TF in use. If I do Nothing the siganls are not good. But this Refreshing should be done automatically. Do you have an idea? or how can I find it in the codebase? There is Nothing for a search, lol.

Thank you very much

Wolf1210

An incorrect data calculation has been made in the indicator. I do not think that it will become better if done correctly - most likely, it will be redrawn.

![Moon Domain - Unipessoal Lda](https://c.mql5.com/avatar/2020/8/5F384774-67B1.jpg)

**[Dmitri Diall](https://www.mql5.com/en/users/ddiall)**
\|
13 Oct 2020 at 16:55

Hi, what is the recommended way to get the price levels (entry, SL, TP) checked and, optionally corrected, by the library before submitting a [pending order](https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties#enum_order_type "MQL5 documentation:") or opening a position? The first time I read through your article series, I had the impression that this was possible via CEngine interface but now can't find how (or misunderstood the first time). The only methods I see are in Ctrading and are not public either -- **CTrading::CheckLevels()**, **CTrading::RequestErrorsCorrecting()**, etc...

![Moon Domain - Unipessoal Lda](https://c.mql5.com/avatar/2020/8/5F384774-67B1.jpg)

**[Dmitri Diall](https://www.mql5.com/en/users/ddiall)**
\|
3 Nov 2020 at 18:17

**Dima Diall:**

Hi, what is the recommended way to get the price levels (entry, SL, TP) checked and, optionally corrected, by the library before submitting a [pending order](https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties#enum_order_type "MQL5 documentation:") or opening a position? The first time I read through your article series, I had the impression that this was possible via CEngine interface but now can't find how (or misunderstood the first time). The only methods I see are in Ctrading and are not public either -- **CTrading::CheckLevels()**, **CTrading::RequestErrorsCorrecting()**, etc...

Hi Artyom - can you clarify this for me, please?

![Artyom Trishkin](https://c.mql5.com/avatar/2022/7/62C4775C-ABD6.jpg)

**[Artyom Trishkin](https://www.mql5.com/en/users/artmedia70)**
\|
3 Nov 2020 at 19:11

**Dima Diall :**

Hi Artyom - can you clarify this for me, please?

Please, a little later - very busy for now.

![Continuous Walk-Forward Optimization (Part 1): Working with Optimization Reports](https://c.mql5.com/2/37/MQL5-avatar-continuous_optimization.png)[Continuous Walk-Forward Optimization (Part 1): Working with Optimization Reports](https://www.mql5.com/en/articles/7290)

The first article is devoted to the creation of a toolkit for working with optimization reports, for importing them from the terminal, as well as for filtering and sorting the obtained data. MetaTrader 5 allows downloading optimization results, however our purpose is to add our own data to the optimization report.

![Library for easy and quick development of MetaTrader programs (part XXII): Trading classes - Base trading class, verification of limitations](https://c.mql5.com/2/37/MQL5-avatar-doeasy__4.png)[Library for easy and quick development of MetaTrader programs (part XXII): Trading classes - Base trading class, verification of limitations](https://www.mql5.com/en/articles/7258)

In this article, we will start the development of the library base trading class and add the initial verification of permissions to conduct trading operations to its first version. Besides, we will slightly expand the features and content of the base trading class.

![Library for easy and quick development of MetaTrader programs (part XXIV): Base trading class - auto correction of invalid parameters](https://c.mql5.com/2/37/MQL5-avatar-doeasy__6.png)[Library for easy and quick development of MetaTrader programs (part XXIV): Base trading class - auto correction of invalid parameters](https://www.mql5.com/en/articles/7326)

In this article, we will have a look at the handler of invalid trading order parameters and improve the trading event class. Now all trading events (both single ones and the ones occurred simultaneously within one tick) will be defined in programs correctly.

![Library for easy and quick development of MetaTrader programs (part XXI): Trading classes - Base cross-platform trading object](https://c.mql5.com/2/37/MQL5-avatar-doeasy__3.png)[Library for easy and quick development of MetaTrader programs (part XXI): Trading classes - Base cross-platform trading object](https://www.mql5.com/en/articles/7229)

In this article, we will start the development of the new library section - trading classes. Besides, we will consider the development of a unified base trading object for MetaTrader 5 and MetaTrader 4 platforms. When sending a request to the server, such a trading object implies that verified and correct trading request parameters are passed to it.

[Running robots on virtual hosting is easyFollow our step-by-step MetaTrader VPS guide for beginnersRead![](https://www.mql5.com/ff/sh/au4fqg4kms7s9mq1z2/01.png)![](https://www.mql5.com/ff/sh/au4fqg4kms7s9mq1z2/02.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/articles/13586&a=uzpprdshbcrtxvjxpmescehprypbymxc&s=516438f25b531570d9b7d49dcfb29c82fa1021f5ede6571df8026dbfbafcd13f&uid=&ref=https://www.mql5.com/en/articles/7286&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5070448513797527043)

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