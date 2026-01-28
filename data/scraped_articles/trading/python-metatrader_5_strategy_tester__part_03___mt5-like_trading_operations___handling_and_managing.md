---
title: Python-MetaTrader 5 Strategy Tester (Part 03): MT5-Like Trading Operations — Handling and Managing
url: https://www.mql5.com/en/articles/20782
categories: Trading, Trading Systems, Integration, Expert Advisors
relevance_score: 9
scraped_at: 2026-01-22T17:23:56.404312
---

[![](https://www.mql5.com/ff/sh/0hvxp984jjj79943z2/6373d9e5710a718ffa6a7d50a5db9dd1.jpg)\\
Web terminal on your iPhone or Android\\
\\
Full-featured MetaTrader 5 platform for any devices and web browsers\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=uyigsjnbfcdvysiynusmriwvhincciwd&s=c95531ae2fd8a81b0fac3def2e4cf820a67584bbf4b02f76ec75f808942dbbd2&uid=&ref=https://www.mql5.com/en/articles/20782&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5049101143356843246)

MetaTrader 5 / Trading


**Contents**

- [Introduction](https://www.mql5.com/en/articles/20782#intro)
- [The order\_send method](https://www.mql5.com/en/articles/20782#order_send)
- [Trade validation class](https://www.mql5.com/en/articles/20782#trade-val-class)
- [All validators inside order\_send](https://www.mql5.com/en/articles/20782#ordersend-vals)
- [The CTrade class inside a simulator](https://www.mql5.com/en/articles/20782#CTrade-class)
- [Performing trading actions in a simulator](https://www.mql5.com/en/articles/20782#trading-actions-in-simulator)
- [Managing orders and positions in a simulator](https://www.mql5.com/en/articles/20782#managing-orders-pos)
- [Conclusion](https://www.mql5.com/en/articles/20782#conclusion)

### Introduction

In the previous article, we implemented similar syntax and functions to those offered by the Python-MetaTrader module in our simulator. With similar orders, deals, positions, and structures. In this post, we will implement a very close to MetaTrader 5 approach of handling such structures (trading operations).

![](https://c.mql5.com/2/189/Article_image.png)

### The order\_send Method

All trading actions for MetaTrader 5, such as placing pending orders, opening buy or sell positions, modifying orders, and deleting them, are results obtained after calling a single boilerplate function.

In MQL5 that function is called [OrderSend](https://www.mql5.com/en/docs/trading/ordersend), in Python-MetaTrader 5 it is called [order\_send](https://www.mql5.com/en/docs/python_metatrader5/mt5ordersend_py) **.**

_According to the documentation._

The method order\_send sends a [request](https://www.mql5.com/en/docs/constants/structures/mqltraderequest)  to perform a [trading operation](https://www.mql5.com/en/docs/constants/tradingconstants/enum_trade_request_actions) from the terminal to the trade server. It is similar to OrderSend.

```
order_send(
   request      // request structure
   );
```

It takes a single parameter called request. The [MqlTradeRequest](https://www.mql5.com/en/docs/constants/structures/mqltraderequest)  type structure describes a required trading action.

The following table represents the fields the "request structure" must hold.

| Field | Description |
| --- | --- |
| action | Trading operation type. The value can be one of the values of the [TRADE\_REQUEST\_ACTIONS](https://www.mql5.com/en/docs/python_metatrader5/mt5ordercheck_py#trade_request_actions) enumeration |
| magic | EA ID. Allows arranging the analytical handling of trading orders. Each EA can set a unique ID when sending a trading request |
| order | Order ticket. Required for modifying pending orders |
| symbol | The name of the trading instrument, for which the order is placed. Not required when modifying orders and closing positions |
| volume | Requested volume of a deal in lots. A real volume when making a deal depends on an [order execution type](https://www.mql5.com/en/docs/python_metatrader5/mt5ordercheck_py#order_type_filling). |
| price | Price at which an order should be executed. The price is not set in case of market orders for instruments of the "Market Execution" ( [SYMBOL\_TRADE\_EXECUTION\_MARKET](https://www.mql5.com/en/docs/constants/environment_state/marketinfoconstants#enum_symbol_trade_execution)) type having the [TRADE\_ACTION\_DEAL](https://www.mql5.com/en/docs/python_metatrader5/mt5ordercheck_py#trade_request_actions) type |
| stoplimit | A price pending limit order is set when the price reaches the 'price' value (this condition is mandatory). The pending order is not passed to the trading system until that moment |
| sl | A price at which a stop loss order is activated when the price moves in an unfavorable direction |
| tp | A price at which a take profit order is activated when the price moves in a favorable direction |
| deviation | Maximum acceptable deviation from the requested price, specified in [points](https://www.mql5.com/en/docs/predefined/_point) |
| type | Order type. The value can be one of the values of the [ORDER\_TYPE](https://www.mql5.com/en/docs/python_metatrader5/mt5ordercalcmargin_py#order_type) enumeration |
| type\_filling | Order filling type. The value can be one of the [ORDER\_TYPE\_FILLING](https://www.mql5.com/en/docs/python_metatrader5/mt5ordercheck_py#order_type_filling) values |
| type\_time | Order type by expiration. The value can be one of the [ORDER\_TYPE\_TIME](https://www.mql5.com/en/docs/python_metatrader5/mt5ordercheck_py#order_type_time) values. |
| expiration | Pending order expiration time (for [TIME\_SPECIFIED](https://www.mql5.com/en/docs/python_metatrader5/mt5ordercheck_py#order_type_time) type orders) |
| comment | Comment to an order |
| position | Position ticket. Fill it when changing and closing a position for its clear identification. Usually, it is the same as the ticket of the order that opened the position. |
| position\_by | Opposite position ticket. It is used when closing a position by opening a position in the opposite direction (at the same symbol). |

We need a similar function in our class.

```
    def order_send(self, request: dict):
        """
        Sends a request to perform a trading operation from the terminal to the trade server. The function is similar to OrderSend in MQL5.
        """

        if not self.IS_TESTER:
            result = self.mt5_instance.order_send(request)
            if result is None or result.retcode != self.mt5_instance.TRADE_RETCODE_DONE:
                self.__GetLogger().warning(f"MT5 failed: {self.mt5_instance.last_error()}")
                return None
            return result
```

In the previous article, we introduced the strategy tester mode to the simulator class, i.e., when the variable IS\_TESTER = True. When not in this mode, the simulator relies on information from the MetaTrader 5 client, opens and manages all trading operations there.

The above code snippet sends an order request to MetaTrader 5 when a user is not in tester mode.

Otherwise, we extract the request's credentials.

```
        action     = request.get("action")
        order_type = request.get("type")
        symbol     = request.get("symbol")
        volume     = float(request.get("volume", 0))
        price      = float(request.get("price", 0))
        sl         = float(request.get("sl", 0))
        tp         = float(request.get("tp", 0))
        ticket     = int(request.get("ticket", -1))

        ticks_info = self.tick_cache[symbol]

        now = utils.ensure_utc(ticks_info.time)
        ts  = int(now.timestamp())
        msc = int(now.timestamp() * 1000)
```

We have to handle all operations manually inside this function, including writing (opening & closing of positions), deals to a container (basically doing MetaTrader 5's work).

**I: Placing Pending Orders**

In a simulator, a pending order is nothing but information about an order stored in a temporary orders container array (self.\_\_orders\_container\_\_).

```
 if action == self.mt5_instance.TRADE_ACTION_PENDING:
            order_ticket = self.__generate_order_ticket()

            order = self.TradeOrder(
                    ticket=order_ticket,
                    time_setup=ts,
                    time_setup_msc=msc,
                    time_done=0,
                    time_done_msc=0,
                    time_expiration=request.get("expiration", 0),
                    type=order_type,
                    type_time=request.get("type_time", 0),
                    type_filling=request.get("type_filling", 0),
                    state=self.mt5_instance.ORDER_STATE_PLACED,
                    magic=request.get("magic", 0),
                    position_id=0,
                    position_by_id=0,
                    reason=self.mt5_instance.DEAL_REASON_EXPERT,
                    volume_initial=volume,
                    volume_current=volume,
                    price_open=price,
                    sl=sl,
                    tp=tp,
                    price_current=price,
                    price_stoplimit=request.get("price_stoplimit", 0),
                    symbol=symbol,
                    comment=request.get("comment", ""),
                    external_id="",
                )
```

After creating an order, we add it to the orders container array and log it to the orders history array (container) as well.

```
            self.__orders_container__.append(order)
            self.__orders_history_container__.append(order)

            return {
                "retcode": self.mt5_instance.TRADE_RETCODE_DONE,
                "order": order_ticket,
            }
```

**II: Opening Positions**

In MetaTrader 5, [Positions](https://www.mql5.com/en/docs/constants/tradingconstants/positionproperties)  are contracts, bought or sold on a financial instrument. A long position (long) is formed as a result of buying anticipating a price increase; a short position (Short) is the result of the sale of an asset, in anticipation of a future price decrease.

In a simulator, a position is a bunch of "position-like" information stored in a container.

```
        if action == self.mt5_instance.TRADE_ACTION_DEAL:


            position_ticket = self.__generate_position_ticket()
            order_ticket    = self.__generate_order_ticket()
            deal_ticket     = self.__generate_deal_ticket()

            position = self.TradePosition(
                ticket=position_ticket,
                time=ts,
                time_msc=msc,
                time_update=ts,
                time_update_msc=msc,
                type=order_type,
                magic=request.get("magic", 0),
                identifier=position_ticket,
                reason=self.mt5_instance.DEAL_REASON_EXPERT,
                volume=volume,
                price_open=price,
                sl=sl,
                tp=tp,
                price_current=price,
                swap=0,
                profit=0,
                symbol=symbol,
                comment=request.get("comment", ""),
                external_id="",
            )

            self.__positions_container__.append(position)
```

The processes of opening and closing positions have the action TRADE\_ACTION\_DEAL.  The result of such an operation can be termed as a deal, so we have to log such a record into a deals container.

```
            self.__deals_history_container__.append(
                self.TradeDeal(
                    ticket=deal_ticket,
                    order=order_ticket,
                    time=ts,
                    time_msc=msc,
                    type=order_type,
                    entry=self.mt5_instance.DEAL_ENTRY_IN,
                    magic=request.get("magic", 0),
                    position_id=position_ticket,
                    reason=self.mt5_instance.DEAL_REASON_EXPERT,
                    volume=volume,
                    price=price,
                    commission=self.__calc_commission(),
                    swap=0,
                    profit=0,
                    fee=0,
                    symbol=symbol,
                    comment=request.get("comment", ""),
                    external_id="",
                )
            )

            return {
                "retcode": self.mt5_instance.TRADE_RETCODE_DONE,
                "deal": deal_ticket,
                "order": order_ticket,
                "position": position_ticket,
            }
```

A deal is simply a record (history) of the opening and closing of positions in the MetaTrader 5 terminal.

![](https://c.mql5.com/2/188/DEALS_IN_MT5.gif)

**III: Closing Positions**

The request for closing a position is very similar to the one for opening a new one. They are both deals, but with different entries.

To detect a request for closing a position, we check if it has a **position** key in its request (a ticket of an existing position).

```
        if action == self.mt5_instance.TRADE_ACTION_DEAL:

            # ---------- CLOSE POSITION ----------

            ticket = request.get("position", -1)
            if ticket != -1:
                pos = next(
                    (p for p in self.__positions_container__ if p.ticket == ticket),
                    None,
                )

                if not pos:
                    return {"retcode": self.mt5_instance.TRADE_RETCODE_INVALID}

                self.__positions_container__.remove(pos)

                deal_ticket = self.__generate_deal_ticket()
                self.__deals_history_container__.append(
                    self.TradeDeal(
                        ticket=deal_ticket,
                        order=0,
                        time=ts,
                        time_msc=msc,
                        type=order_type,
                        entry=self.mt5_instance.DEAL_ENTRY_OUT,
                        magic=request.get("magic", 0),
                        position_id=pos.ticket,
                        reason=self.mt5_instance.DEAL_REASON_EXPERT,
                        volume=volume,
                        price=price,
                        commission=self.__calc_commission(),
                        swap=0,
                        profit=0,
                        fee=0,
                        symbol=symbol,
                        comment=request.get("comment", ""),
                        external_id="",
                    )
                )

                return {
                    "retcode": self.mt5_instance.TRADE_RETCODE_DONE,
                    "deal": deal_ticket,
                }
```

However, we cannot accept every request for closing a position blindly; we have to validate them.

For now, two crucial details we have to check before writing a deal and removing a position from the container include.

1. Checking if a request has a valid price, In MetaTrader 5, buy positions are closed at the bid price while sell positions are closed at the ask price.

2. We check if a given order type in the request (position type) is the opposite of an existing order, i.e., if a request is sent for an existing buy order ORDER\_TYPE\_BUY, it should be eligible for closing when given ORDER\_TYPE\_SELL.

```
        if action == self.mt5_instance.TRADE_ACTION_DEAL:

            # ---------- CLOSE POSITION ----------

            ticket = request.get("position", -1)
            if ticket != -1:
                pos = next(
                    (p for p in self.__positions_container__ if p.ticket == ticket),
                    None,
                )

                if not pos:
                    return {"retcode": self.mt5_instance.TRADE_RETCODE_INVALID}

                # validate position close request

                if pos.type == order_type:
                    self.__GetLogger().critical("Failed to close an order. Order type must be the opposite")
                    return None

                if order_type == self.mt5_instance.ORDER_TYPE_BUY: # For a sell order/position

                    if not TradeValidators.price_equal(a=price, b=ticks_info.ask, eps=pow(10, -symbol_info.digits)):
                        self.__GetLogger().critical(f"Failed to close ORDER_TYPE_SELL. Price {price} is not equal to bid {ticks_info.bid}")
                        return None

                elif order_type == self.mt5_instance.ORDER_TYPE_SELL: # For a buy order/position
                    if not TradeValidators.price_equal(a=price, b=ticks_info.bid, eps=pow(10, -symbol_info.digits)):
                        self.__GetLogger().critical(f"Failed to close ORDER_TYPE_BUY. Price {price} is not equal to bid {ticks_info.bid}")
                        return None


                self.__positions_container__.remove(pos)

                deal_ticket = self.__generate_deal_ticket()
                self.__deals_history_container__.append(
                    self.TradeDeal(
                        ticket=deal_ticket,
                        order=0,
                        time=ts,
                        time_msc=msc,
                        type=order_type,
                        entry=self.mt5_instance.DEAL_ENTRY_OUT,
                        magic=request.get("magic", 0),
                        position_id=pos.ticket,
                        reason=self.mt5_instance.DEAL_REASON_EXPERT,
                        volume=volume,
                        price=price,
                        commission=self.__calc_commission(),
                        swap=0,
                        profit=0,
                        fee=0,
                        symbol=symbol,
                        comment=request.get("comment", ""),
                        external_id="",
                    )
                )

                return {
                    "retcode": self.mt5_instance.TRADE_RETCODE_DONE,
                    "deal": deal_ticket,
                }
```

**IV: Modifying Positions**

To modify a position, we focus on two details only. Stop loss and take profit values.

```
        elif action == self.mt5_instance.TRADE_ACTION_SLTP:

            ticket = request.get("position", -1)

            pos = next((p for p in self.__positions_container__ if p.ticket == ticket), None)
            if not pos:
                return {"retcode": self.mt5_instance.TRADE_RETCODE_INVALID}

            # --- Correct reference prices ---
            entry_price = pos.price_open
            market_price = ticks_info.bid if pos.type == self.mt5_instance.POSITION_TYPE_BUY else ticks_info.ask

            # --- Validate SL / TP relative to ENTRY ---
            if sl > 0:
                if not trade_validators.is_valid_sl(entry=entry_price, sl=sl, order_type=pos.type):
                    return None

            if tp > 0:
                if not trade_validators.is_valid_tp(entry=entry_price, tp=tp, order_type=pos.type):
                    return None

            # --- Validate freeze level against MARKET ---
            if sl > 0:
                if not trade_validators.is_valid_freeze_level(entry=market_price, stop_price=sl, order_type=pos.type):
                    return None

            if tp > 0:
                if not trade_validators.is_valid_freeze_level(entry=market_price, stop_price=tp, order_type=pos.type):
                    return None

            # --- APPLY MODIFICATION ---
            idx = self.__positions_container__.index(pos)

            updated_pos = pos._replace(
                sl=sl,
                tp=tp,
                time_update=ts,
                time_update_msc=msc
            )

            self.__positions_container__[idx] = updated_pos

            return {"retcode": self.mt5_instance.TRADE_RETCODE_DONE}
```

**V: Deleting Pending Orders**

This is a straightforward process of removing an order from its container array. No validations required.

```
        if action == self.mt5_instance.TRADE_ACTION_REMOVE:

            ticket = request.get("order", -1)

            self.__orders_container__ = [\
                o for o in self.__orders_container__ if o.ticket != ticket\
            ]

            return {"retcode": self.mt5_instance.TRADE_RETCODE_DONE}
```

**VI: Modifying Pending Orders**

To modify a pending order, we focus on five crucial details. Order's opening price, stop loss, take profit, time expiration, and stop limit.

```
        elif action == self.mt5_instance.TRADE_ACTION_SLTP:

            ticket = request.get("position", -1)

            pos = next((p for p in self.__positions_container__ if p.ticket == ticket), None)
            if not pos:
                return {"retcode": self.mt5_instance.TRADE_RETCODE_INVALID}

            # --- Correct reference prices ---
            entry_price = pos.price_open
            market_price = ticks_info.bid if pos.type == self.mt5_instance.POSITION_TYPE_BUY else ticks_info.ask

            # --- Validate SL / TP relative to ENTRY ---
            if sl > 0:
                if not trade_validators.is_valid_sl(entry=entry_price, sl=sl, order_type=pos.type):
                    return None

            if tp > 0:
                if not trade_validators.is_valid_tp(entry=entry_price, tp=tp, order_type=pos.type):
                    return None

            # --- Validate freeze level against MARKET ---
            if sl > 0:
                if not trade_validators.is_valid_freeze_level(entry=market_price, stop_price=sl, order_type=pos.type):
                    return None

            if tp > 0:
                if not trade_validators.is_valid_freeze_level(entry=market_price, stop_price=tp, order_type=pos.type):
                    return None

            # --- APPLY MODIFICATION ---
            idx = self.__positions_container__.index(pos)

            updated_pos = pos._replace(
                sl=sl,
                tp=tp,
                time_update=ts,
                time_update_msc=msc
            )

            self.__positions_container__[idx] = updated_pos

            return {"retcode": self.mt5_instance.TRADE_RETCODE_DONE}

```

But, all these actions are wrong without a core method(s) for validating the request's credentials in the first place.

In the first simulator we implemented in this article series, we had an unorganized way of validating orders' credentials. This time, we improve it by implementing a separate class for the task.

### The Trade Validation Class

As we know, MetaTrader 5 does not accept all requests passed. It checks for invalid credentials and throws an error, rejecting such orders when that happens.

These credentials are validated according to a given instrument specification, an account type, broker needs, and sometimes MetaTrader 5 client limits.

**I: Checking for the right Lot Size**

For a lot size to be accepted by MetaTrader 5:

- it has to be greater than the minimum allowed lot size (volume) for a particular instrument (symbol)
- It has to be smaller than the maximum allowed lot size for a particular instrument
- It has to be a multiple of the step size volume

_Inside validators.py_

```
class TradeValidators:
    def __init__(self, symbol_info: namedtuple, ticks_info: any, logger: any, mt5_instance: mt5=mt5):

        self.symbol_info = symbol_info
        self.ticks_info = ticks_info
        self.logger = logger
        self.mt5_instance = mt5_instance

    def is_valid_lotsize(self, lotsize: float) -> bool:

        # Validate lotsize

        if lotsize < self.symbol_info.volume_min: # check if the received lotsize is smaller than minimum accepted lot of a symbol
            self.logger.info(f"Trade validation failed: lotsize ({lotsize}) is less than minimum allowed ({self.symbol_info.volume_min})")
            return False

        if lotsize > self.symbol_info.volume_max: # check if the received lotsize is greater than the maximum accepted lot
            self.logger.info(f"Trade validation failed: lotsize ({lotsize}) is greater than maximum allowed ({self.symbol_info.volume_max})")
            return False

        step_count = lotsize / self.symbol_info.volume_step

        if abs(step_count - round(step_count)) > 1e-7: # check if the stoploss is a multiple of the step size
            self.logger.info(f"Trade validation failed: lotsize ({lotsize}) must be a multiple of step size ({self.symbol_info.volume_step})")
            return False

        return True
```

**II: Ensuring There is Enough Money for a New Position**

The MetaTrader 5 terminal checks if there is enough free margin on the account to accommodate a new position.

Below is a similar function for the task.

```
    def is_there_enough_money(self, margin_required: float, free_margin: float) -> bool:

        if margin_required < 0:
            self.logger.info("Trade validation failed: Cannot calculate margin requirements")
            return False

        # Check free margin
        if margin_required > free_margin:
            self.logger.info(f'Trade validation failed: Not enough money to open trade. '
                f'Required: {margin_required:.2f}, '
                f'Free margin: {free_margin:.2f}')

            return False

        return True
```

**III: Checking to Ensure A Valid Entry is Given**

For a buy position, its price must be equal to the ask price, and for a sell position, its price must be equal to the bid price. _This check is for positions only._

```
    def is_valid_entry(self, price: float, order_type: int) -> bool:

        eps = pow(10, -self.symbol_info.digits)
        if order_type == self.mt5_instance.ORDER_TYPE_BUY:  # BUY
            if not self.price_equal(a=price, b=self.ticks_info.ask, eps=eps):
                self.logger.info(f"Trade validation failed: Buy price {price} != ask {self.ticks_info.ask}")
                return False

        elif order_type == self.mt5_instance.ORDER_TYPE_SELL:  # SELL
            if not self.price_equal(a=price, b=self.ticks_info.bid, eps=eps):
                self.logger.info(f"Trade validation failed: Sell price {price} != bid {self.ticks_info.bid}")
                return False
        else:
            self.logger.error("Unknown MetaTrader 5 position type")
            return False

        return True
```

**IV: Ensuring Stop Loss and Take Profit Values Aren't Too Close to the Market**

All symbols come with a small threshold value indicating the minimum distance where stop loss and take profit values must be placed from the market.

This threshold value is called [SYMBOL\_TRADE\_STOPS\_LEVEL](https://www.mql5.com/en/articles/2555#invalid_SL_TP_for_position).

```
    def is_valid_stops_level(self, entry: float, stop_price: float, stops_type: str='') -> bool:

        point = self.symbol_info.point
        stop_level   = self.symbol_info.trade_stops_level * point

        distance = abs(entry-stop_price)

        if stop_price <= 0:
            return True

        if distance < stop_level:
            self.logger.info(f"{'Either SL or TP' if stops_type=='' else stops_type} is too close to the market. Min allowed distance = {stop_level}")
            return False

        return True
```

**V: Checking For Valid Stop Loss and Take Profit Values**

For a buy order, a stop loss must be below the entry price, while the takeprofit must be above the entry price.

For a sell order, a take profit value must be below the entry price, while the stop loss must be above the entry price.

```
    def is_valid_sl(self, entry: float, sl: float, order_type: int) -> bool:

        if not self.is_valid_stops_level(entry, sl, "Stoploss"): # check for stops levels
            return False

        if sl > 0:
            if order_type in self.BUY_ACTIONS: # buy action

                if sl >= entry:
                    self.logger.info(f"Trade validation failed: Buy-based order's stop loss ({sl}) must be below order opening price ({entry})")
                    return False

            elif order_type in self.SELL_ACTIONS: # sell action

                if sl <= entry:
                    self.logger.info(f"Trade validation failed: Sell-based order's stop loss ({sl}) must be above order opening price ({entry})")
                    return False

            else:
                self.logger.error("Unknown MetaTrader 5 order type")
                return False

        return True

    def is_valid_tp(self, entry: float, tp: float, order_type: int) -> bool:

        if not self.is_valid_stops_level(entry, tp, "Takeprofit"): # check for stops and freeze levels
            return False

        if tp > 0:
            if order_type in self.BUY_ACTIONS: # buy position
                if tp <= entry:
                    self.logger.info(f"Trade validation failed: {self.ORDER_TYPES_MAP[order_type]} take profit ({tp}) must be above order opening price ({entry})")
                    return False
            elif order_type in self.SELL_ACTIONS: # sell position
                if tp >= entry:
                    self.logger.info(f"Trade validation failed: {self.ORDER_TYPES_MAP[order_type]} take profit ({tp}) must be below order opening price ({entry})")
                    return False
            else:
                self.logger.error("Unknown MetaTrader 5 order type")
                return False

        return True
```

**VI: A Check To Ensure Maximum Lot Size Isn't Reached for an Instrument**

Some symbols in some brokers have a limit for the total lot size in individual opened orders and positions.

```
    def is_symbol_volume_reached(self, symbol_volume: float, volume_limit: float) -> bool:

        """Checks if the maximum allowed volume is reached for a particular instrument

        Returns:
            bool: True if the condition is reached and False when it is not.
        """

        if symbol_volume >= volume_limit and volume_limit > 0:
            self.logger.critical(f"Symbol Volume limit of {volume_limit} is reached!")
            return True

        return False
```

**VII: A Check To Ensure the Maximum Number of orders isn't reached**

Some accounts tend to have a limit in the number of pending orders that can be opened at a time. We have to check for that to ensure that we don't violate a simulated account, just like the MetaTrader 5 platform wouldn't let us violate a real one.

```
    def is_max_orders_reached(self, open_orders: int, ac_limit_orders: int) -> bool:
        """Checks whether the maximum number of orders for the account is reached

        Args:
            open_orders (int): The number of opened orders
            ac_limit_orders (int): Maximum number of orders allowed for the account

        Returns:
            bool: True if the threshold is reached, otherwise, it returns false.
        """

        if open_orders >= ac_limit_orders and ac_limit_orders > 0:
            self.logger.critical(f"Pending Orders limit of {ac_limit_orders} is reached!")
            return True

        return False
```

**VIII: Checking for the Freeze Level**

The [SYMBOL\_TRADE\_FREEZE\_LEVEL](https://www.mql5.com/en/docs/constants/environment_state/marketinfoconstants) parameter may be set in the symbol specification. It shows the distance of freezing the trade operations for pending orders and open positions in points. For example, if a trade on a financial instrument is redirected for processing to an external trading system, then a BuyLimit pending order may be currently too close to the current Ask price. And, if and a request to modify this order is sent at the moment when the opening price is close enough to the Ask price, it may happen so that the order will have been executed and modification will be impossible.

| Type of order/position | Activation price | Check |
| --- | --- | --- |
| Buy Limit order | Ask | Ask-OpenPrice >= SYMBOL\_TRADE\_FREEZE\_LEVEL |
| Buy Stop order | Ask | OpenPrice-Ask >= SYMBOL\_TRADE\_FREEZE\_LEVEL |
| Sell Limit order | Bid | OpenPrice-Bid >= SYMBOL\_TRADE\_FREEZE\_LEVEL |
| Sell Stop order | Bid | Bid-OpenPrice >= SYMBOL\_TRADE\_FREEZE\_LEVEL |
| Buy position | Bid | TakeProfit-Bid >= SYMBOL\_TRADE\_FREEZE\_LEVEL<br>Bid-StopLoss >= SYMBOL\_TRADE\_FREEZE\_LEVEL |
| Sell position | Ask | Ask-TakeProfit >= SYMBOL\_TRADE\_FREEZE\_LEVEL<br>StopLoss-Ask >= SYMBOL\_TRADE\_FREEZE\_LEVEL |

```
    def is_valid_freeze_level(self, entry: float, stop_price: float, order_type: int) -> bool:
        """
        Check SYMBOL_TRADE_FREEZE_LEVEL for pending orders and open positions.
        """

        freeze_level = self.symbol_info.trade_freeze_level
        if freeze_level <= 0:
            return True  # No freeze restriction

        point = self.symbol_info.point
        freeze_distance = freeze_level * point

        bid = self.ticks_info.bid
        ask = self.ticks_info.ask

        def log_fail(msg: str, dist: float):
            self.logger.info(
                f"{msg} | distance={dist/point:.1f} pts < "
                f"freeze_level={freeze_level} pts"
            )

        # ---------------- Pending Orders ----------------

        if order_type == self.mt5_instance.ORDER_TYPE_BUY_LIMIT:
            dist = ask - entry
            if dist < freeze_distance:
                log_fail("BuyLimit cannot be modified: Ask - OpenPrice", dist)
                return False
            return True

        if order_type == self.mt5_instance.ORDER_TYPE_SELL_LIMIT:
            dist = entry - bid
            if dist < freeze_distance:
                log_fail("SellLimit cannot be modified: OpenPrice - Bid", dist)
                return False
            return True

        if order_type == self.mt5_instance.ORDER_TYPE_BUY_STOP:
            dist = entry - ask
            if dist < freeze_distance:
                log_fail("BuyStop cannot be modified: OpenPrice - Ask", dist)
                return False
            return True

        if order_type == self.mt5_instance.ORDER_TYPE_SELL_STOP:
            dist = bid - entry
            if dist < freeze_distance:
                log_fail("SellStop cannot be modified: Bid - OpenPrice", dist)
                return False
            return True

        # ---------------- Open Positions (SL / TP modification) ----------------

        # Buy position
        if order_type == self.mt5_instance.ORDER_TYPE_BUY:
            if stop_price <= 0:
                return True

            if stop_price < entry:  # StopLoss
                dist = bid - stop_price
                if dist < freeze_distance:
                    log_fail("Buy position SL cannot be modified: Bid - SL", dist)
                    return False
            else:  # TakeProfit
                dist = stop_price - bid
                if dist < freeze_distance:
                    log_fail("Buy position TP cannot be modified: TP - Bid", dist)
                    return False

            return True

        # Sell position
        if order_type == self.mt5_instance.ORDER_TYPE_SELL:
            if stop_price <= 0:
                return True

            if stop_price > entry:  # StopLoss
                dist = stop_price - ask
                if dist < freeze_distance:
                    log_fail("Sell position SL cannot be modified: SL - Ask", dist)
                    return False
            else:  # TakeProfit
                dist = ask - stop_price
                if dist < freeze_distance:
                    log_fail("Sell position TP cannot be modified: Ask - TP", dist)
                    return False

            return True

        self.logger.error("Unknown MetaTrader 5 order type")
        return False
```

### All Validators inside order\_send (TL;DR)

With all these functions inside a class named TradeValidators within a file _validators.py_  applied to the function order\_send, below is how everything fits:

```
    def order_send(self, request: dict):
        """
        Sends a request to perform a trading operation from the terminal to the trade server. The function is similar to OrderSend in MQL5.
        """

        # -----------------------------------------------------

        if not self.IS_TESTER:
            result = self.mt5_instance.order_send(request)
            if result is None or result.retcode != self.mt5_instance.TRADE_RETCODE_DONE:
                self.__GetLogger().warning(f"MT5 failed: {self.mt5_instance.last_error()}")
                return None
            return result

        # -------------------- Extract request -----------------------------

        action     = request.get("action")
        order_type = request.get("type")
        symbol     = request.get("symbol")
        volume     = float(request.get("volume", 0))
        price      = float(request.get("price", 0))
        sl         = float(request.get("sl", 0))
        tp         = float(request.get("tp", 0))
        ticket     = int(request.get("ticket", -1))

        ticks_info = self.tick_cache[symbol]
        symbol_info = self.symbol_info(symbol)
        ac_info = self.account_info()

        now = ticks_info.time
        ts  = int(now)
        msc = int(now * 1000)

        if order_type not in self.ORDER_TYPES:
            return {"retcode": self.mt5_instance.TRADE_RETCODE_INVALID}


        trade_validators = TradeValidators(symbol_info=symbol_info,
                                           ticks_info=ticks_info,
                                           logger=self.__GetLogger(),
                                           mt5_instance=self.mt5_instance)

        # -------------------- REMOVE pending order ------------------------

        if action == self.mt5_instance.TRADE_ACTION_REMOVE:

            ticket = request.get("order", -1)

            self.__orders_container__ = [\
                o for o in self.__orders_container__ if o.ticket != ticket\
            ]

            return {"retcode": self.mt5_instance.TRADE_RETCODE_DONE}

        # --------------------- PENDING order --------------------------

        if action == self.mt5_instance.TRADE_ACTION_PENDING:

            if trade_validators.is_max_orders_reached(open_orders=len(self.__orders_container__),
                                                      ac_limit_orders=ac_info.limit_orders):
                return None

            if not trade_validators.is_valid_sl(entry=price, sl=sl, order_type=order_type) or not trade_validators.is_valid_tp(entry=price, tp=tp, order_type=order_type):
                return None

            total_volume = sum([pos.volume for pos in self.__positions_container__]) + sum([order.volume for order in self.__orders_container__])
            if trade_validators.is_symbol_volume_reached(symbol_volume=total_volume, volume_limit=symbol_info.volume_limit):
                return None

            order_ticket = self.__generate_order_ticket()

            order = self.TradeOrder(
                    ticket=order_ticket,
                    time_setup=ts,
                    time_setup_msc=msc,
                    time_done=0,
                    time_done_msc=0,
                    time_expiration=request.get("expiration", 0),
                    type=order_type,
                    type_time=request.get("type_time", 0),
                    type_filling=request.get("type_filling", 0),
                    state=self.mt5_instance.ORDER_STATE_PLACED,
                    magic=request.get("magic", 0),
                    position_id=0,
                    position_by_id=0,
                    reason=self.mt5_instance.DEAL_REASON_EXPERT,
                    volume_initial=volume,
                    volume_current=volume,
                    price_open=price,
                    sl=sl,
                    tp=tp,
                    price_current=price,
                    price_stoplimit=request.get("price_stoplimit", 0),
                    symbol=symbol,
                    comment=request.get("comment", ""),
                    external_id="",
                )

            self.__orders_container__.append(order)
            self.__orders_history_container__.append(order)

            return {
                "retcode": self.mt5_instance.TRADE_RETCODE_DONE,
                "order": order_ticket,
            }

        # ------------------ MARKET DEAL (open or close) ------------------

        if action == self.mt5_instance.TRADE_ACTION_DEAL:

            # ---------- CLOSE POSITION ----------

            ticket = request.get("position", -1)
            if ticket != -1:
                pos = next(
                    (p for p in self.__positions_container__ if p.ticket == ticket),
                    None,
                )

                if not pos:
                    return {"retcode": self.mt5_instance.TRADE_RETCODE_INVALID}

                # validate position close request

                if pos.type == order_type:
                    self.__GetLogger().critical("Failed to close an order. Order type must be the opposite")
                    return None

                if order_type == self.mt5_instance.ORDER_TYPE_BUY: # For a sell order/position

                    if not TradeValidators.price_equal(a=price, b=ticks_info.ask, eps=pow(10, -symbol_info.digits)):
                        self.__GetLogger().critical(f"Failed to close ORDER_TYPE_SELL. Price {price} is not equal to bid {ticks_info.bid}")
                        return None

                elif order_type == self.mt5_instance.ORDER_TYPE_SELL: # For a buy order/position
                    if not TradeValidators.price_equal(a=price, b=ticks_info.bid, eps=pow(10, -symbol_info.digits)):
                        self.__GetLogger().critical(f"Failed to close ORDER_TYPE_BUY. Price {price} is not equal to bid {ticks_info.bid}")
                        return None


                self.__positions_container__.remove(pos)

                deal_ticket = self.__generate_deal_ticket()
                self.__deals_history_container__.append(
                    self.TradeDeal(
                        ticket=deal_ticket,
                        order=0,
                        time=ts,
                        time_msc=msc,
                        type=order_type,
                        entry=self.mt5_instance.DEAL_ENTRY_OUT,
                        magic=request.get("magic", 0),
                        position_id=pos.ticket,
                        reason=self.mt5_instance.DEAL_REASON_EXPERT,
                        volume=volume,
                        price=price,
                        commission=self.__calc_commission(),
                        swap=0,
                        profit=0,
                        fee=0,
                        symbol=symbol,
                        comment=request.get("comment", ""),
                        external_id="",
                    )
                )

                return {
                    "retcode": self.mt5_instance.TRADE_RETCODE_DONE,
                    "deal": deal_ticket,
                }

            # ---------- OPEN POSITION ----------

            # validate new stops

            if not trade_validators.is_valid_sl(entry=price, sl=sl, order_type=order_type):
                return None
            if not trade_validators.is_valid_tp(entry=price, tp=tp, order_type=order_type):
                return None

            # validate the lotsize

            if not trade_validators.is_valid_lotsize(lotsize=volume):
                return None

            total_volume = sum([pos.volume for pos in self.__positions_container__]) + sum([order.volume for order in self.__orders_container__])
            if trade_validators.is_symbol_volume_reached(symbol_volume=total_volume, volume_limit=symbol_info.volume_limit):
                return None


            if not trade_validators.is_there_enough_money(margin_required=self.order_calc_margin(order_type=order_type,
                                                                                                 symbol=symbol,
                                                                                                 volume=volume,
                                                                                                 price=price),
                                                          free_margin=ac_info.margin_free):
                return None

            position_ticket = self.__generate_position_ticket()
            order_ticket    = self.__generate_order_ticket()
            deal_ticket     = self.__generate_deal_ticket()

            position = self.TradePosition(
                ticket=position_ticket,
                time=ts,
                time_msc=msc,
                time_update=ts,
                time_update_msc=msc,
                type=order_type,
                magic=request.get("magic", 0),
                identifier=position_ticket,
                reason=self.mt5_instance.DEAL_REASON_EXPERT,
                volume=volume,
                price_open=price,
                sl=sl,
                tp=tp,
                price_current=price,
                swap=0,
                profit=0,
                symbol=symbol,
                comment=request.get("comment", ""),
                external_id="",
            )

            self.__positions_container__.append(position)

            self.__deals_history_container__.append(
                self.TradeDeal(
                    ticket=deal_ticket,
                    order=order_ticket,
                    time=ts,
                    time_msc=msc,
                    type=order_type,
                    entry=self.mt5_instance.DEAL_ENTRY_IN,
                    magic=request.get("magic", 0),
                    position_id=position_ticket,
                    reason=self.mt5_instance.DEAL_REASON_EXPERT,
                    volume=volume,
                    price=price,
                    commission=self.__calc_commission(),
                    swap=0,
                    profit=0,
                    fee=0,
                    symbol=symbol,
                    comment=request.get("comment", ""),
                    external_id="",
                )
            )

            return {
                "retcode": self.mt5_instance.TRADE_RETCODE_DONE,
                "deal": deal_ticket,
                "order": order_ticket,
                "position": position_ticket,
            }

        elif action == self.mt5_instance.TRADE_ACTION_MODIFY: # Modifying pending orders

            ticket = request.get("order", -1)

            order = next(
                (o for o in self.__orders_container__ if o.ticket == ticket),
                None,
            )

            if not order:
                return {"retcode": self.mt5_instance.TRADE_RETCODE_INVALID}

            # validate new stops

            if not trade_validators.is_valid_freeze_level(entry=price, stop_price=sl, order_type=order_type):
                return None
            if not trade_validators.is_valid_freeze_level(entry=price, stop_price=tp, order_type=order_type):
                return None

            # Modify ONLY allowed fields
            order.price_open      = price
            order.sl              = sl
            order.tp              = tp
            order.time_expiration = request.get("expiration", order.time_expiration)
            order.price_stoplimit = request.get("price_stoplimit", order.price_stoplimit)

            return {"retcode": self.mt5_instance.TRADE_RETCODE_DONE}

        elif action == self.mt5_instance.TRADE_ACTION_SLTP: # Modifying an open position

            ticket = request.get("position", -1)

            pos = next(
                (p for p in self.__positions_container__ if p.ticket == ticket),
                None,
            )

            if not pos:
                return {"retcode": self.mt5_instance.TRADE_RETCODE_INVALID}

            # Check for valid stoplosses and TPs ensuring they are not too close to the market

            if pos.type == self.mt5_instance.ORDER_TYPE_BUY:
                if not trade_validators.is_valid_sl(entry=ticks_info.bid, sl=sl, order_type=order_type) or not trade_validators.is_valid_tp(entry=ticks_info.bid, tp=tp, order_type=order_type):
                    return None

            elif pos.type == self.mt5_instance.ORDER_TYPE_SELL:
                if not trade_validators.is_valid_sl(entry=ticks_info.ask, sl=sl, order_type=order_type) or not trade_validators.is_valid_tp(entry=ticks_info.ask, tp=tp, order_type=order_type):
                    return None

            if not trade_validators.is_valid_freeze_level(entry=price, stop_price=sl, order_type=order_type):
                return None
            if not trade_validators.is_valid_freeze_level(entry=price, stop_price=sl, order_type=order_type):
                return None

            pos.sl = sl
            pos.tp = tp
            pos.time_update = ts
            pos.time_update_msc = msc

            return {"retcode": self.mt5_instance.TRADE_RETCODE_DONE}

        return {
            "retcode": self.mt5_instance.TRADE_RETCODE_INVALID,
            "comment": "Unsupported trade action",
        }
```

### The CTrade Class Inside a Simulator

Just like MQL5, the Python-MetaTrader 5 module feels like a low-level module that lets us communicate with the MetaTrader 5 terminal. As we've just seen, it takes more than what's required to send a request for opening positions and orders _(a long, tiresome process)_.

In MQL5, we have the so-called [trade classes](https://www.mql5.com/en/docs/standardlibrary/tradeclasses), which provide us with a simple interface for opening and managing the trades. [In Python, we created a similar classes](https://www.mql5.com/en/articles/18208), let us adapt the [CTrade class](https://www.mql5.com/en/articles/18208#CTrade-Python) to fit our simulator needs.

```
import MetaTrader5 as mt5
from datetime import datetime, timezone
import config

class CTrade:

    def __init__(self, simulator, magic_number: int, filling_type_symbol: str, deviation_points: int):

        self.simulator = simulator
        self.mt5_instance = simulator.mt5_instance
        self.magic_number = magic_number
        self.deviation_points = deviation_points
        self.filling_type = self._get_type_filling(filling_type_symbol)

        if self.filling_type == -1:
            print("Failed to initialize the class, Invalid filling type. Check your symbol")
            return
```

We give our class a simulator instance for extracting some methods from it instead of directly from MetaTrader 5, which helps our class to use overridden methods discussed in [the prior article](https://www.mql5.com/en/articles/20455).

The CTrade class comes with plenty of methods; they all rely on a method called **order\_send**  that comes with the Python-MetaTrader 5 module.

Throughout the class, instead of calling methods from the Python-MetaTrader 5 module, we call overridden methods from a Simulator class.

```
class CTrade:

    def __init__(self, simulator, magic_number: int, filling_type_symbol: str, deviation_points: int):

        self.simulator = simulator
        self.mt5_instance = simulator.mt5_instance
        self.magic_number = magic_number
        self.deviation_points = deviation_points
        self.filling_type = self._get_type_filling(filling_type_symbol)

        if self.filling_type == -1:
            print("Failed to initialize the class, Invalid filling type. Check your symbol")
            return

    def _get_type_filling(self, symbol):

        symbol_info = self.simulator.symbol_info(symbol)
        if symbol_info is None:
            print(f"Failed to get symbol info for {symbol}")

        filling_map = {
            1: self.mt5_instance.ORDER_FILLING_FOK,
            2: self.mt5_instance.ORDER_FILLING_IOC,
            4: self.mt5_instance.ORDER_FILLING_BOC,
            8: self.mt5_instance.ORDER_FILLING_RETURN
        }

        return filling_map.get(symbol_info.filling_mode, f"Unknown Filling type")


    def __GetLogger(self):
        if self.simulator.IS_TESTER:
            return config.tester_logger

        return config.simulator_logger

    def position_open(self, symbol: str, volume: float, order_type: int, price: float, sl: float=0.0, tp: float=0.0, comment: str="") -> bool:

        """
        Open a market position (instant execution).

        Executes either a buy or sell order at the current market price. This is for immediate
        position opening, not pending orders.

        Args:
            symbol: Trading symbol (e.g., "EURUSD", "GBPUSD")
            volume: Trade volume in lots (e.g., 0.1 for micro lot)
            order_type: Trade direction (either ORDER_TYPE_BUY or ORDER_TYPE_SELL)
            price: Execution price. For market orders, this should be the current:
                - Ask price for BUY orders
                - Bid price for SELL orders
            sl: Stop loss price (set to 0.0 to disable)
            tp: Take profit price (set to 0.0 to disable)
            comment: Optional order comment (max 31 characters, will be truncated automatically)

        Returns:
            bool: True if position was opened successfully, False otherwise
        """

        request = {
            "action": self.mt5_instance.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "deviation": self.deviation_points,
            "magic": self.magic_number,
            "comment": comment,
            "type_time": self.mt5_instance.ORDER_TIME_GTC,
            "type_filling":  self.filling_type,
        }

        if sl > 0.0:
            request["sl"] = sl
        if tp > 0.0:
            request["tp"] = tp

        if self.simulator.order_send(request) is None:
            return False

        self.__GetLogger().info(f"Position Opened successfully!")

        return True

    def order_open(self, symbol: str, volume: float, order_type: int, price: float, sl: float = 0.0, tp: float = 0.0, type_time: int = mt5.ORDER_TIME_GTC, expiration: datetime = None, comment: str = "") -> bool:

        """
        Opens a pending order with full control over order parameters.

        Args:
            symbol: Trading symbol (e.g., "EURUSD")
            volume: Order volume in lots
            order_type: Order type (ORDER_TYPE_BUY_LIMIT, ORDER_TYPE_SELL_STOP, etc.)
            price: Activation price for pending order
            sl: Stop loss price (0 to disable)
            tp: Take profit price (0 to disable)
            type_time: Order expiration type (default: ORDER_TIME_GTC). Possible values:
                    - ORDER_TIME_GTC (Good-Til-Canceled)
                    - ORDER_TIME_DAY (Good for current day)
                    - ORDER_TIME_SPECIFIED (expires at specific datetime)
                    - ORDER_TIME_SPECIFIED_DAY (expires at end of specified day)
            expiration: Expiration datetime (required for ORDER_TIME_SPECIFIED types)
            comment: Optional order comment (max 31 characters)

        Returns:
            bool: True if order was placed successfully, False otherwise
        """

        # Validate expiration for time-specific orders
        if type_time in (self.mt5_instance.ORDER_TIME_SPECIFIED, self.mt5_instance.ORDER_TIME_SPECIFIED_DAY) and expiration is None:
            print(f"Expiration required for order type {type_time}")
            return False

        request = {
            "action": self.mt5_instance.TRADE_ACTION_PENDING,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": self.deviation_points,
            "magic": self.magic_number,
            "comment": comment[:31],  # MT5 comment max length is 31 chars
            "type_time": type_time,
            "type_filling": self.filling_type,
        }

        # Add expiration if required
        if type_time in (self.mt5_instance.ORDER_TIME_SPECIFIED, self.mt5_instance.ORDER_TIME_SPECIFIED_DAY) and expiration is not None:

            # Convert to broker's expected format (UTC timestamp in milliseconds)

            expiration_utc = expiration.astimezone(timezone.utc) if expiration.tzinfo else expiration.replace(tzinfo=timezone.utc)
            request["expiration"] = int(expiration_utc.timestamp() * 1000)

        # Send order

        if self.simulator.order_send(request) is None:
            return False

        self.__GetLogger().info(f"Order opened successfully!")
        return True


    def buy(self, volume: float, symbol: str, price: float, sl: float=0.0, tp: float=0.0, comment: str="") -> bool:

        """
        Opens a buy (market) position.

        Args:
            volume: Trade volume (lot size)
            symbol: Trading symbol (e.g., "EURUSD")
            price: Execution price
            sl: Stop loss price (optional, default=0.0)
            tp: Take profit price (optional, default=0.0)
            comment: Position comment (optional, default="")

        Returns:
            bool: True if order was sent successfully, False otherwise
        """

        return self.position_open(symbol=symbol, volume=volume, order_type=self.mt5_instance.ORDER_TYPE_BUY, price=price, sl=sl, tp=tp, comment=comment)
```

With this module adapted to fit our simulator needs, we now have an easy way of opening positions and pending orders. Before testing these methods, let us understand the changes made to our class and what should be done to run a simulator successfully.

In the [previous article](https://www.mql5.com/en/articles/20455), we introduced a method called Start, which sets our simulator class instance to a strategy tester mode that makes everything and almost all data in the class virtual.

```
    def Start(self, IS_TESTER: bool) -> bool: # simulator start

        self.IS_TESTER = IS_TESTER
```

With this mode chosen (set to True) the simulator imitates the MetaTrader 5 strategy tester behavior, set to False, a simulator is no longer as it relies on the MetaTrader 5 client directly  for all the information and opens the trades there, this mode was introduced in the class for testing purposes, ensuring what is conducted in a simulator resembles what's carried out in the MetaTrader 5 client.

With new changes implemented, this method is removed. By default, the class is called in strategy tester mode (simulator). To get into the MetaTrader 5 mode (usually for debugging purposes) a user must pass an argument --mt5 when calling the final script.

```
(venv) C:\Users\Omega Joctan\OneDrive\Documents\PyMetaTester>python test.py --mt5
```

### Performing Trading Actions in a Simulator

You must follow the steps below to test the current version of the simulator (files attached at the end of this article):

_Inside test.py_

**01: Initialize the MetaTrader 5 terminal**

```
import MetaTrader5 as mt5
from Trade.Trade import CTrade
from datetime import datetime, timedelta
import time
import pytz
from simulator import Simulator, CTrade

if not mt5.initialize(): # Initialize MetaTrader5 instance
    print(f"Failed to Initialize MetaTrader5. Error = {mt5.last_error()}")
    mt5.shutdown()
    quit()
```

**02: Calling the Simulator Instance**

We call the simulator class instance, giving it the MetaTrader 5 app instance, the account's balance labelled as deposit, and the leverage value.

```
sim = Simulator(simulator_name="MySimulator", mt5_instance=mt5, deposit=1078.30, leverage="1:500")
```

We then use a simulator instance for the CTrade class.

**03: (Optional) Instantiating the CTrade Class**

```
symbol = "EURUSD"
timeframe = mt5.TIMEFRAME_H1
```

```
m_trade = CTrade(simulator=sim, magic_number=112233, filling_type_symbol=symbol, deviation_points=100)
```

**04: We Assign Tick Information to the Simulator and Extract it Back**

```
mt5_ticks = mt5.symbol_info_tick(symbol) # tick source
sim.TickUpdate(symbol=symbol, tick=mt5_ticks) # very important

tick_from_sim = sim.symbol_info_tick(symbol=symbol) # we get ticks back from a class

ask = tick_from_sim.ask
bid = tick_from_sim.bid
```

**05: Finally, Some Trading Operations**

```
symbol_info = sim.symbol_info(symbol=symbol)
lotsize = symbol_info.volume_min

m_trade.buy(
    volume=lotsize,
    symbol=symbol,
    price=ask,
    sl=ask - 100 * symbol_info.point,
    tp=ask + 150 * symbol_info.point,
    comment="Market Buy"
)

m_trade.sell(
    volume=lotsize,
    symbol=symbol,
    price=bid,
    sl=bid + 100 * symbol_info.point,
    tp=bid - 150 * symbol_info.point,
    comment="Market Sell"
)

buy_limit_price = ask - 200 * symbol_info.point

m_trade.buy_limit(
    volume=lotsize,
    symbol=symbol,
    price=buy_limit_price,
    sl=buy_limit_price - 100 * symbol_info.point,
    tp=buy_limit_price + 200 * symbol_info.point,
    comment="Buy Limit"
)

sell_limit_price = bid + 200 * symbol_info.point

m_trade.sell_limit(
    volume=lotsize,
    symbol=symbol,
    price=sell_limit_price,
    sl=sell_limit_price + 100 * symbol_info.point,
    tp=sell_limit_price - 200 * symbol_info.point,
    comment="Sell Limit"
)

buy_stop_price = ask + 150 * symbol_info.point

m_trade.buy_stop(
    volume=lotsize,
    symbol=symbol,
    price=buy_stop_price,
    sl=buy_stop_price - 100 * symbol_info.point,
    tp=buy_stop_price + 300 * symbol_info.point,
    comment="Buy Stop"
)

sell_stop_price = bid - 150 * symbol_info.point

m_trade.sell_stop(
    volume=lotsize,
    symbol=symbol,
    price=sell_stop_price,
    sl=sell_stop_price + 100 * symbol_info.point,
    tp=sell_stop_price - 300 * symbol_info.point,
    comment="Sell Stop"
)
```

We can check to see if these opened positions and orders exist in our simulator.

```
print(f"positions in a simulator = {sim.positions_total()}:\n",sim.positions_get())
print(f"orders in a simulator = {sim.orders_total()}:\n", sim.orders_get())
```

Outputs (Tester Mode):

```
(venv) C:\Users\Omega Joctan\OneDrive\Documents\PyMetaTester>python test.py
2026-01-05 15:09:24,504 | INFO     | tester | [Trade.py:85 - position_open() ] => Position Opened successfully!
2026-01-05 15:09:24,513 | INFO     | tester | [Trade.py:85 - position_open() ] => Position Opened successfully!
2026-01-05 15:09:24,515 | INFO     | tester | [Trade.py:148 - order_open() ] => Order opened successfully!
2026-01-05 15:09:24,515 | INFO     | tester | [Trade.py:148 - order_open() ] => Order opened successfully!
2026-01-05 15:09:24,515 | INFO     | tester | [Trade.py:148 - order_open() ] => Order opened successfully!
2026-01-05 15:09:24,515 | INFO     | tester | [Trade.py:148 - order_open() ] => Order opened successfully!
positions in a simulator = 2:
 (TradePosition(ticket=113127357728313068862, time=1767622158, time_msc=1767622158000, time_update=1767622158, time_update_msc=1767622158000, type=0, magic=112233, identifier=113127357728313068862, reason=3, volume=0.01, price_open=1.16792, sl=1.1669200000000002, tp=1.1694200000000001, price_current=1.16792, swap=0, profit=0, symbol='EURUSD', comment='Market Buy', external_id=''),
TradePosition(ticket=113127357728890995262, time=1767622158, time_msc=1767622158000, time_update=1767622158, time_update_msc=1767622158000, type=1, magic=112233, identifier=113127357728890995262, reason=3, volume=0.01, price_open=1.16792, sl=1.16892, tp=1.16642, price_current=1.16792, swap=0, profit=0, symbol='EURUSD', comment='Market Sell', external_id=''))
orders in a simulator = 4:
 (TradeOrder(ticket=113127357729019468800, time_setup=1767622158, time_setup_msc=1767622158000, time_done=0, time_done_msc=0, time_expiration=0, type=2, type_time=0, type_filling=1, state=1, magic=112233, position_id=0, position_by_id=0, reason=3, volume_initial=0.01, volume_current=0.01, price_open=1.16592, sl=1.1649200000000002, tp=1.16792, price_current=1.16592, price_stoplimit=0, symbol='EURUSD', comment='Buy Limit', external_id=''),
TradeOrder(ticket=113127357729019468835, time_setup=1767622158, time_setup_msc=1767622158000, time_done=0, time_done_msc=0, time_expiration=0, type=3, type_time=0, type_filling=1, state=1, magic=112233, position_id=0, position_by_id=0, reason=3, volume_initial=0.01, volume_current=0.01, price_open=1.16992, sl=1.17092, tp=1.16792, price_current=1.16992, price_stoplimit=0, symbol='EURUSD', comment='Sell Limit', external_id=''),
TradeOrder(ticket=113127357729019468836, time_setup=1767622158, time_setup_msc=1767622158000, time_done=0, time_done_msc=0, time_expiration=0, type=4, type_time=0, type_filling=1, state=1, magic=112233, position_id=0, position_by_id=0, reason=3, volume_initial=0.01, volume_current=0.01, price_open=1.1694200000000001, sl=1.1684200000000002, tp=1.17242, price_current=1.1694200000000001, price_stoplimit=0, symbol='EURUSD', comment='Buy Stop', external_id=''),
TradeOrder(ticket=113127357729019468803, time_setup=1767622158, time_setup_msc=1767622158000, time_done=0, time_done_msc=0, time_expiration=0, type=5, type_time=0, type_filling=1, state=1, magic=112233, position_id=0, position_by_id=0, reason=3, volume_initial=0.01, volume_current=0.01, price_open=1.16642, sl=1.16742, tp=1.1634200000000001, price_current=1.16642, price_stoplimit=0, symbol='EURUSD', comment='Sell Stop', external_id=''))
```

Outputs (MetaTrader 5 Mode):

```
(venv) C:\Users\Omega Joctan\OneDrive\Documents\PyMetaTester>python test.py --mt5
2026-01-05 15:09:29,171 | INFO     | simulator | [Trade.py:85 - position_open() ] => Position Opened successfully!
2026-01-05 15:09:30,270 | INFO     | simulator | [Trade.py:85 - position_open() ] => Position Opened successfully!
2026-01-05 15:09:31,110 | INFO     | simulator | [Trade.py:148 - order_open() ] => Order opened successfully!
2026-01-05 15:09:31,711 | INFO     | simulator | [Trade.py:148 - order_open() ] => Order opened successfully!
2026-01-05 15:09:33,000 | INFO     | simulator | [Trade.py:148 - order_open() ] => Order opened successfully!
2026-01-05 15:09:33,952 | INFO     | simulator | [Trade.py:148 - order_open() ] => Order opened successfully!
positions in a simulator = 2:
 (TradePosition(ticket=1393244663, time=1767622166, time_msc=1767622166713, time_update=1767622166, time_update_msc=1767622166713, type=0, magic=112233, identifier=1393244663, reason=3, volume=0.01, price_open=1.16791, sl=1.1669100000000001, tp=1.16941, price_current=1.16791, swap=0.0, profit=0.0, symbol='EURUSD', comment='Market Buy', external_id=''),
TradePosition(ticket=1393244666, time=1767622167, time_msc=1767622167817, time_update=1767622167, time_update_msc=1767622167817, type=1, magic=112233, identifier=1393244666, reason=3, volume=0.01, price_open=1.16791, sl=1.16891, tp=1.16641, price_current=1.16791, swap=0.0, profit=0.0, symbol='EURUSD', comment='Market Sell', external_id=''))
orders in a simulator = 4:
 (TradeOrder(ticket=1393244672, time_setup=1767622168, time_setup_msc=1767622168661, time_done=0, time_done_msc=0, time_expiration=0, type=2, type_time=0, type_filling=2, state=1, magic=112233, position_id=0, position_by_id=0, reason=3, volume_initial=0.01, volume_current=0.01, price_open=1.16591, sl=1.16491, tp=1.16791, price_current=1.16791, price_stoplimit=0.0, symbol='EURUSD', comment='Buy Limit', external_id=''),
TradeOrder(ticket=1393244676, time_setup=1767622169, time_setup_msc=1767622169494, time_done=0, time_done_msc=0, time_expiration=0, type=3, type_time=0, type_filling=2, state=1, magic=112233, position_id=0, position_by_id=0, reason=3, volume_initial=0.01, volume_current=0.01, price_open=1.16991, sl=1.1709100000000001, tp=1.16791, price_current=1.16791, price_stoplimit=0.0, symbol='EURUSD', comment='Sell Limit', external_id=''),
TradeOrder(ticket=1393244679, time_setup=1767622170, time_setup_msc=1767622170093, time_done=0, time_done_msc=0, time_expiration=0, type=4, type_time=0, type_filling=2, state=1, magic=112233, position_id=0, position_by_id=0, reason=3, volume_initial=0.01, volume_current=0.01, price_open=1.16941, sl=1.16841, tp=1.17241, price_current=1.16791, price_stoplimit=0.0, symbol='EURUSD', comment='Buy Stop', external_id=''),
TradeOrder(ticket=1393244687, time_setup=1767622171, time_setup_msc=1767622171748, time_done=0, time_done_msc=0, time_expiration=0, type=5, type_time=0, type_filling=2, state=1, magic=112233, position_id=0, position_by_id=0, reason=3, volume_initial=0.01, volume_current=0.01, price_open=1.16641, sl=1.16741, tp=1.16341, price_current=1.16791, price_stoplimit=0.0, symbol='EURUSD', comment='Sell Stop', external_id=''))
```

### Managing Orders and Positions in a Simulator

With positions stored within arrays, containers in a simulator class, we can perform actions on them, such as detecting certain behaviors/conditions, modifying, or even closing them.

**01: Closing Positions**

Let's open two distinct positions and close one.

```
m_trade.buy(volume=lotsize, symbol=symbol, price=ask, comment="buy pos")
m_trade.sell(volume=lotsize, symbol=symbol, price=bid, comment="sell pos")

print(f"positions in a simulator = {sim.positions_total()}:\n",sim.positions_get())

positions = sim.positions_get()
for pos in positions:
    if pos.symbol == symbol and pos.type == sim.mt5_instance.POSITION_TYPE_BUY: # close a buy position
        m_trade.position_close(ticket=pos.ticket, deviation=10)

print("positions remaining: ", sim.positions_get())
```

Output:

```
2026-01-05 15:54:50,114 | INFO     | tester | [Trade.py:85 - position_open() ] => Position Opened successfully!
2026-01-05 15:54:50,114 | INFO     | tester | [Trade.py:85 - position_open() ] => Position Opened successfully!
positions in a simulator = 2:
 (TradePosition(ticket=113127532167305337632, time=1767624887, time_msc=1767624887000, time_update=1767624887, time_update_msc=1767624887000, type=0, magic=112233, identifier=113127532167305337632, reason=3, volume=0.01, price_open=1.16743, sl=0.0, tp=0.0, price_current=1.16743, swap=0, profit=0, symbol='EURUSD', comment='buy pos', external_id=''),
TradePosition(ticket=113127532167305337651, time=1767624887, time_msc=1767624887000, time_update=1767624887, time_update_msc=1767624887000, type=1, magic=112233, identifier=113127532167305337651, reason=3, volume=0.01, price_open=1.16743, sl=0.0, tp=0.0, price_current=1.16743, swap=0, profit=0, symbol='EURUSD', comment='sell pos', external_id=''))
2026-01-05 15:54:50,114 | INFO     | tester | [Trade.py:397 - position_close() ] => Position 113127532167305337632 closed successfully!
positions remaining:  (TradePosition(ticket=113127532167305337651, time=1767624887, time_msc=1767624887000, time_update=1767624887, time_update_msc=1767624887000, type=1, magic=112233, identifier=113127532167305337651, reason=3, volume=0.01, price_open=1.16743, sl=0.0, tp=0.0, price_current=1.16743, swap=0, profit=0, symbol='EURUSD', comment='sell pos', external_id=''),)
```

A buy position was closed exclusively!

**02: Position Modifications**

Similarly to closing positions, we have to select one before sending a modification request to it.

```
m_trade.buy(volume=lotsize, symbol=symbol, price=ask, comment="buy pos")
m_trade.sell(volume=lotsize, symbol=symbol, price=bid, comment="sell pos")

print(f"positions in a simulator = {sim.positions_total()}:\n",sim.positions_get())

positions = sim.positions_get()
for pos in positions:
    if pos.sl == 0:
        if pos.type == sim.mt5_instance.POSITION_TYPE_BUY:
            m_trade.position_modify(ticket=pos.ticket, sl=pos.price_open - 100 * symbol_info.point, tp=pos.tp)
        if pos.type == sim.mt5_instance.POSITION_TYPE_SELL:
            m_trade.position_modify(ticket=pos.ticket, sl=pos.price_open + 100 * symbol_info.point, tp=pos.tp)

print("positions after modification\n: ", sim.positions_get())
```

Outputs.

```
(venv) C:\Users\Omega Joctan\OneDrive\Documents\PyMetaTester>python test.py
2026-01-05 17:19:11,604 | INFO     | tester | [Trade.py:85 - position_open() ] => Position Opened successfully!
2026-01-05 17:19:11,604 | INFO     | tester | [Trade.py:85 - position_open() ] => Position Opened successfully!
positions in a simulator = 2:
 (TradePosition(ticket=113127856102580262436, time=1767629948, time_msc=1767629948000, time_update=1767629948, time_update_msc=1767629948000, type=0, magic=112233, identifier=113127856102580262436, reason=3, volume=0.01, price_open=1.16789, sl=0.0, tp=0.0, price_current=1.16789, swap=0, profit=0, symbol='EURUSD', comment='buy pos', external_id=''),
TradePosition(ticket=113127856102710534416, time=1767629948, time_msc=1767629948000, time_update=1767629948, time_update_msc=1767629948000, type=1, magic=112233, identifier=113127856102710534416, reason=3, volume=0.01, price_open=1.16789, sl=0.0, tp=0.0, price_current=1.16789, swap=0, profit=0, symbol='EURUSD', comment='sell pos', external_id=''))
2026-01-05 17:19:11,604 | INFO     | tester | [Trade.py:469 - position_modify() ] => Position 113127856102580262436 modified successfully!
2026-01-05 17:19:11,606 | INFO     | tester | [Trade.py:469 - position_modify() ] => Position 113127856102710534416 modified successfully!
positions after modification
:  (TradePosition(ticket=113127856102580262436, time=1767629948, time_msc=1767629948000, time_update=1767629948, time_update_msc=1767629948000, type=0, magic=112233, identifier=113127856102580262436, reason=3, volume=0.01, price_open=1.16789, sl=1.1668900000000002, tp=0.0, price_current=1.16789, swap=0, profit=0, symbol='EURUSD', comment='buy pos', external_id=''),
TradePosition(ticket=113127856102710534416, time=1767629948, time_msc=1767629948000, time_update=1767629948, time_update_msc=1767629948000, type=1, magic=112233, identifier=113127856102710534416, reason=3, volume=0.01, price_open=1.16789, sl=1.16889, tp=0.0, price_current=1.16789, swap=0, profit=0, symbol='EURUSD', comment='sell pos', external_id=''))
```

**03: Working With Pending Orders**

Below is how you can modify and delete pending orders.

```
m_trade.buy_stop(symbol=symbol, volume=symbol_info.volume_min, price=ask+500*symbol_info.point)

for order in sim.orders_get():

    print("order curr price: ", order.price_open)

    m_trade.order_modify(ticket=order.ticket, price=order.price_open+10*symbol_info.point, sl=order.sl, tp=order.tp)

    print("order moved 10 points upward", order.price_open)
    if m_trade.order_delete(ticket=order.ticket) is None:
        continue

    print("orders remaining: ", sim.orders_total())
```

Outputs.

```
(venv) C:\Users\Omega Joctan\OneDrive\Documents\PyMetaTester>python test.py
2026-01-05 17:43:15,677 | INFO     | tester | [Trade.py:148 - order_open() ] => Order opened successfully!
order curr price:  1.17281
2026-01-05 17:43:15,677 | INFO     | tester | [Trade.py:536 - order_modify() ] => Order 113127948523388723206 modified successfully!
order moved 10 points upward 1.17291
2026-01-05 17:43:15,677 | INFO     | tester | [Trade.py:431 - order_delete() ] => Order 113127948523388723206 deleted successfully!
orders remaining:  0
```

### What's Next?

The ability to use tick data from a particular period in the past and use it simulate a trading operation means that we are a few steps away from finalizing our custom simulator. Right now we have everything needed to create a loop that runs the simulation through all available ticks in specified time range. This is what we call strategy testing or a complete simulation.

In this article, we discussed the most important aspect of any trading simulator, that is, sending trading requests and managing them. In the next one, we are going to put it all together and run our very first strategy testing action in Python, Stay tuned!

Peace Out.

Share your thoughts and help to improve this project on GitHub: [https://github.com/MegaJoctan/PyMetaTester](https://www.mql5.com/go?link=https://github.com/MegaJoctan/StrategyTester5 "https://github.com/MegaJoctan/PyMetaTester")

**Attachments Table**

| Filename | Description & Usage |
| --- | --- |
| Trade/Trade.py | Contains the CTrade class, Class providing an easy way for trade operations execution. |
| config.py | A Python configuration file where the most useful variables for reusability throughout the project are defined. |
| utils.py | A utility Python file which contains simple functions to help with various tasks (helpers). |
| simulator.py | It has a class named Simulator. Our core simulator logic is in one place. |
| test.py | A file used for testing all the code and functions discussed in this post. |
| error\_description.py | It has functions for converting all MetaTrader 5 error codes into human-readable messages. |
| requirements.txt | Contains all Python dependencies and their versions, used in this project. |

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/20782.zip "Download all attachments in the single ZIP archive")

[Attachments.zip](https://www.mql5.com/en/articles/download/20782/Attachments.zip "Download Attachments.zip")(23.93 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Python-MetaTrader 5 Strategy Tester (Part 02): Dealing with Bars, Ticks, and Overloading Built-in Functions in a Simulator](https://www.mql5.com/en/articles/20455)
- [Implementing Practical Modules from Other Languages in MQL5 (Part 06): Python-Like File IO operations in MQL5](https://www.mql5.com/en/articles/20695)
- [Data Science and ML (Part 47): Forecasting the Market Using the DeepAR model in Python](https://www.mql5.com/en/articles/20571)
- [Implementing Practical Modules from Other Languages in MQL5 (Part 05): The Logging module from Python, Log Like a Pro](https://www.mql5.com/en/articles/20458)
- [Implementing Practical Modules from Other Languages in MQL5 (Part 04): time, date, and datetime modules from Python](https://www.mql5.com/en/articles/19035)
- [Python-MetaTrader 5 Strategy Tester (Part 01): Trade Simulator](https://www.mql5.com/en/articles/18971)

**[Go to discussion](https://www.mql5.com/en/forum/503807)**

![Developing a multi-currency Expert Advisor (Part 24): Adding a new strategy (II)](https://c.mql5.com/2/127/Developing_a_Multicurrency_Advisor_Part_25__LOGO.png)[Developing a multi-currency Expert Advisor (Part 24): Adding a new strategy (II)](https://www.mql5.com/en/articles/17328)

In this article, we will continue to connect the new strategy to the created auto optimization system. Let's look at what changes need to be made to the optimization project creation EA, as well as the second and third stage EAs.

![Creating Custom Indicators in MQL5 (Part 5): WaveTrend Crossover Evolution Using Canvas for Fog Gradients, Signal Bubbles, and Risk Management](https://c.mql5.com/2/190/20815-creating-custom-indicators-logo.png)[Creating Custom Indicators in MQL5 (Part 5): WaveTrend Crossover Evolution Using Canvas for Fog Gradients, Signal Bubbles, and Risk Management](https://www.mql5.com/en/articles/20815)

In this article, we enhance the Smart WaveTrend Crossover indicator in MQL5 by integrating canvas-based drawing for fog gradient overlays, signal boxes that detect breakouts, and customizable buy/sell bubbles or triangles for visual alerts. We incorporate risk management features with dynamic take-profit and stop-loss levels calculated via candle multipliers or percentages, displayed through lines and a table, alongside options for trend filtering and box extensions.

![Build a Remote Forex Risk Management System in Python](https://c.mql5.com/2/124/Remote_Professional_Forex_Risk_Manager_in_Python___LOGO.png)[Build a Remote Forex Risk Management System in Python](https://www.mql5.com/en/articles/17410)

We are making a remote professional risk manager for Forex in Python, deploying it on the server step by step. In the course of the article, we will understand how to programmatically manage Forex risks, and how not to waste a Forex deposit any more.

![Price Action Analysis Toolkit (Part 55): Designing a CPI Mini-Candle Overlay for Intra-bar Pressure](https://c.mql5.com/2/190/20949-price-action-analysis-toolkit-logo.png)[Price Action Analysis Toolkit (Part 55): Designing a CPI Mini-Candle Overlay for Intra-bar Pressure](https://www.mql5.com/en/articles/20949)

This article presents the design and MetaTrader 5 implementation of the Candle Pressure Index (CPI)—a CLV-based overlay that visualizes intra-Bar buying and selling pressure directly on price charts. The discussion focuses on candle structure, pressure classification, visualization mechanics, and a non-repainting, transition-based alert system designed for consistent behavior across timeframes and instruments.

[![](https://www.mql5.com/ff/sh/jup0jccfs9655z9z2/01.png)Learn to create your own robotsRead our book "MQL5 Programming for Traders"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/book%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.algobook%26utm_content=visit.page%26utm_campaign=algobook.promo.04.2024&a=rsxjstxkzbrlgjjrxaglpezpvrjflnvw&s=7224440013c3dbc50ba9cc078cd015fabca36df446b8e75028d6b30234663872&uid=&ref=https://www.mql5.com/en/articles/20782&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5049101143356843246)

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