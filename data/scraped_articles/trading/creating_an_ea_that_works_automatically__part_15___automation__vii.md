---
title: Creating an EA that works automatically (Part 15): Automation (VII)
url: https://www.mql5.com/en/articles/11438
categories: Trading, Trading Systems, Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T18:06:54.083807
---

[Running robots on virtual hosting is easyFollow our step-by-step MetaTrader VPS guide for beginnersRead![](https://www.mql5.com/ff/sh/au4fqg4kms7s9mq1z2/01.png)![](https://www.mql5.com/ff/sh/au4fqg4kms7s9mq1z2/02.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/articles/13586&a=uzpprdshbcrtxvjxpmescehprypbymxc&s=516438f25b531570d9b7d49dcfb29c82fa1021f5ede6571df8026dbfbafcd13f&uid=&ref=https://www.mql5.com/en/articles/11438&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5069114690228912268)

MetaTrader 5 / Examples


### Introduction

In the previous article [Creating an EA that works automatically (Part 14): Automation (VI)](https://www.mql5.com/en/articles/11318), we considered the C\_Automaton class and discussed its basics. But since using the C\_Automaton class to automate a system is not as easy as it might seem, in this article we will look at examples of how to do this.

Here we will see 3 different models. The article will focus solely on explaining how to adapt the C\_Automaton class to implement each of the models. For more details about the system, you should read the previous articles, since in this article we are only going to discuss adaptation, i.e. the dependent part.

The topic is actually quite long, so let's move on to practical examples.

### Automation example 1: 9-period exponential moving average

The example uses the EA code available in the EA\_v1.mq5 file attached below. Let's start with the class constructor:

```
                C_Automaton(const ulong magic, double FinanceStop, double FinanceTake, uint Leverage,
                            bool IsDayTrade, double Trailing, const ENUM_TIMEFRAMES iPeriod,
                            const double OverBought = 70, const double OverSold = 30, const int iShift = 1)
                        :C_Manager(magic, 0, 0, Leverage, IsDayTrade, 0, false, 10),
                         m_TF(iPeriod),
                         m_Handle(INVALID_HANDLE)
                        {
                                m_Infos.Shift      = iShift;
                                m_Infos.OverBought = OverBought;
                                m_Infos.OverSold   = OverSold;
                                ArraySetAsSeries(m_Buff, true);
                                m_nBars  = iBars(NULL, m_TF);
                                m_Handle = iMA(NULL, m_TF, 9, 0, MODE_EMA, PRICE_CLOSE);
                        }
```

Let's consider the differences compared to the default system:

- The system will have no stop loss or take profit, which means that there will be no profit or loss limit in the system, since the EA itself will generate these limits depending on the price movement.
- The system does not create trailing stop, breakeven or pending stop orders.
- We will use one handle, which will be the 9-period exponential moving average indicator based on the closing price.

The rest is already explained earlier, so we can move on to the mechanism calculation procedure. The relevant ways were discussed in the previous article, so please read it for more details. But once the calculation method is defined, the following code appears:

```
inline eTrigger CheckTrigger(void)
                        {
                                int iRet;

                                if (((iRet = iBars(NULL, m_TF)) > m_nBars) && (m_Handle != INVALID_HANDLE))
                                {
                                        if (CopyBuffer(m_Handle, 0, 0, m_Infos.Shift + 1, m_Buff) < m_Infos.Shift + 1) return TRIGGER_NONE;
                                        m_nBars = iRet;
                                        if (m_Buff[0] > m_Buff[m_Infos.Shift]) return TRIGGER_BUY;
                                        if (m_Buff[0] < m_Buff[m_Infos.Shift]) return TRIGGER_SELL;
                                };
                                return TRIGGER_NONE;
                        }
```

Sounds complicated? Actually, it is not. It is quite simple to do as long as you have created rules in a flowchart that we discussed in the previous article. First, we try to read the indicator content; if that's not possible, we return a null trigger, and the signal may be lost.

In some cases, especially when the lost signal was an exit one, the system can start to generate losses. However, let's not make hasty assumptions. I think that this explains why you should not leave the EA unsupervised. A possible solution would be to send a position closing signal to the C\_Manager class in case of any error. But as I mentioned earlier, we should not make assumptions, so it's up to you whether to add this signal or not.

Then we updated the bar counter so that the signal is only triggered at the next bar. However, this will only happen if there is a response from the indicator. Otherwise, the signal will be checked again on the next OnTime event. So, we really should not make assumptions about what is happening. Pay attention to the sequence in which everything happens. If the update of the number of bars occurred before, we would miss the next OnTime event. However, since we update it later, we can receive the OnTimer event and thus try to read the indicator again.

Now let's do the calculation to determine whether to buy or sell. To understand this calculation, it is necessary to understand how the exponential moving average is calculated. Unlike other moving averages, the exponential moving average reacts faster to price changes. So, as soon as it slopes, based on the position of the closing price which we defined in the constructor, we will actually know whether the bar closed above or below it.

But there is one detail in this calculation: it will only report this information quickly enough, if we are comparing the current average value with the immediate previous value. So, any slight change may cause the trigger to happen. If you want to reduce the sensitivity level, you must change the value contained in the **m\_Infos.Shift** variable from 1 to 2, or to a greater value. This will simulate the moving average shift in order to capture a certain type of movement, reducing or increasing the sensitivity of the system.

This type of shift is quite common to some settings such as _**Joe di Napoli**_. Many people think that it is necessary to look at the bar in relation to the moving average, but in reality it is only necessary to adjust the MA accordingly in order to understand if the bar is following the pattern. In the case of the Joe di Napoli setup, we should make a zigzag calculation on the moving average so that the trigger is activated at the appropriate point. Still, we don't need to look at the bars, while we only need the average values.

An important detail in the above calculation: the zero point in the calculation indicates the most recent value of the buffer, i.e. the last value calculated by the indicator.

If the last value is higher than the previous one, then the EA should immediately Buy. If it is lower, the EA should Sell.

Here is a curious fact: this system resembles the well-known Larry Williams' system. Some people use an additional element: Instead of buying or selling immediately, you can send a pending order to the high or low of the bar on which the moving average signal triggered. Since the C\_Manager class guarantees that the server has only one pending order, we will only have to change the **Triggers** function without changing the calculation. Thus, instead of a request for a market trade, the system will send a pending order with the data of the bar that generated the signal.

This will not be the code provided in the attached file, but it will look like something shown below:

```
inline virtual void Triggers(void) final
                        {
                                if (!CtrlTimeIsPassed()) ClosePosition(); else switch (CheckTrigger())
                                {
                                        case TRIGGER_BUY:
                                                if (m_Memory == TRIGGER_SELL) ClosePosition();
                                                if (GetVolumeInPosition() == 0)
                                                {
                                                        DestroyOrderPendent();
                                                        CreateOrder(ORDER_TYPE_BUY, iHigh(NULL, m_TF, 0));
                                                }
                                                m_Memory = TRIGGER_BUY;
                                                break;
                                        case TRIGGER_SELL:
                                                if (m_Memory == TRIGGER_BUY) ClosePosition();
                                                if (GetVolumeInPosition() == 0)
                                                {
                                                        DestroyOrderPendent();
                                                        CreateOrder(ORDER_TYPE_SELL, iLow(NULL, m_TF, 0));
                                                }
                                                m_Memory = TRIGGER_SELL;
                                                break;
                                }
                        };
```

We have added a new function to the C\_Manager class code; it is available in the attached file. Now pay attention to the fact that the order being created is different from a market entry. We now have an entry based on the price of one of the bar limits. This will change as the situation evolves. If the order was not executed on the bar where the trigger was created, then on the next bar the order will be reset automatically. Please note that this model is not provided in the attachment, I'm just demonstrating its potential.

I think you have already understood that the system is quite flexible. However, we have only scratched the surface of what can be achieved using this class system. To illustrate another application, let's look at the second example.

### Automation example 2: Using RSI or IFR

We have seen how the MA-based system works. Now, let's look at the use with another indicator. In this example we will use quite a popular indicator. However, the method is applicable to any other indicator or oscillator while the idea is quite universal.

The EA code for this example is available in the EA\_v2.mq5file attached to the article. Let's again start with the constructor:

```
                C_Automaton(const ulong magic, double FinanceStop, double FinanceTake, uint Leverage,
                            bool IsDayTrade, double Trailing, const ENUM_TIMEFRAMES iPeriod,
                            const double OverBought = 70, const double OverSold = 30, const int iShift = 1)
                        :C_Manager(magic, FinanceStop, 0, Leverage, IsDayTrade, Trailing, true, 10),
                         m_TF(iPeriod),
                         m_Handle(INVALID_HANDLE)
                        {
                                m_Infos.Shift      = iShift;
                                m_Infos.OverBought = OverBought;
                                m_Infos.OverSold   = OverSold;
                                ArraySetAsSeries(m_Buff, true);
                                m_nBars = iBars(NULL, m_TF);
                                m_Handle = iRSI(NULL, m_TF, 14, PRICE_CLOSE);

                        }
```

Let's look at the differences from the first example here. You can repeat the same process as in the previous example or adapt the previous example to look like this one:

- Here we have already defined a value to be used as a Stop Loss after opening a position;
- Indicate a value that will be used as Breakeven and Trailing stop;
- We will use a pending order as a stop point;
- Indicate the use of 14-period RSI based on the close price.
- Overbought and oversold values are entered here and stored for later use..

Do you see how simple the process is? You can simply use another indicator instead of iRSI. The next step is the following calculation:

```
inline eTrigger CheckTrigger(void)
                        {
                                int iRet;

                                if (((iRet = iBars(NULL, m_TF)) > m_nBars) && (m_Handle != INVALID_HANDLE))
                                {
                                        if (CopyBuffer(m_Handle, 0, 0, m_Infos.Shift + 1, m_Buff) < m_Infos.Shift + 1) return TRIGGER_NONE;
                                        m_nBars = iRet;
                                        if ((m_Buff[0] > m_Buff[m_Infos.Shift]) && (m_Buff[0] > m_Infos.OverSold) && (m_Buff[m_Infos.Shift] < m_Infos.OverSold)) return TRIGGER_BUY;
                                        if ((m_Buff[0] < m_Buff[m_Infos.Shift]) && (m_Buff[0] < m_Infos.OverBought) && (m_Buff[m_Infos.Shift] > m_Infos.OverBought)) return TRIGGER_SELL;
                                };
                                return TRIGGER_NONE;
                        }
```

Here, in the calculation, we continue doing the same tests as in the previous example, with some additional details. You might have noticed the following detail: we analyze whether the movement in the indicator has turned downwards or upwards. This factor is very important as it may indicate a correction. That is why we have a shift to notice this type of movement. But one way or another, we check if the indicator points to an oversold or overbought setup before we make a second decision based on the indicator's exit from that area. This gives a buy or sell trigger.

In the rest, we don't see any big changes. As you can see, the procedure is quite simple and it doesn't change much depending on the analysis. We simply define a trigger that fires an action to adapt it to whatever model and methodology is needed to automate the EA.

However, unlike the previous example, in this example we want to implement a breakeven movement and a trailing stop. The idea is that when there is a move that can bring profit, we can close the position in a more profitable way. This change is implemented with the following addition to the code:

```
inline virtual void Triggers(void) final
                        {
                                if (!CtrlTimeIsPassed()) ClosePosition(); else switch (CheckTrigger())
                                {
                                        case TRIGGER_BUY:
                                                if (m_Memory == TRIGGER_SELL) ClosePosition();
                                                if (m_Memory != TRIGGER_BUY) ToMarket(ORDER_TYPE_BUY);
                                                m_Memory = TRIGGER_BUY;
                                                break;
                                        case TRIGGER_SELL:
                                                if (m_Memory == TRIGGER_BUY) ClosePosition();
                                                if (m_Memory != TRIGGER_SELL) ToMarket(ORDER_TYPE_SELL);
                                                m_Memory = TRIGGER_SELL;
                                                break;
                                }
                                TriggerTrailingStop();
                        };
```

Note that the procedure has remained virtually unchanged from the original. However, we have added this call that will implement breakeven and trailing stop. As you can see, it all comes down to when and where to place calls, because the system already has all the details for later use and can adapt to our needs in each required case.

So, let's see another example to better understand the ideas.

### Automation example 3: Moving Average crossover

This example is available in the file EA\_v3.mq5. But we won't stop with just the most basic automation process: you can see some minor changes in the C\_Manager class. The first one is the creation of a routine to let the automation system know if there is a long or short position. It is shown below:

```
const bool IsBuyPosition(void) const
                        {
                                return m_Position.IsBuy;
                        }
```

This function, in fact, does not confirm if the position is open or not, but only returns if the variable is indicating buy or sell. The idea is not actually to check if there is an open position, but to return if it is bought or sold, hence the simplicity of the function. But if your automation system requires that verification, you can check if there is an open position. Anyway, this function is enough for our example. This is because of the function that comes below:

```
                void LockStopInPrice(const double Price)
                        {
                                if (m_InfosManager.IsOrderFinish)
                                {
                                        if (m_Pending.Ticket == 0) return;
                                        if ((m_Pending.PriceOpen > Price) && (m_Position.IsBuy)) return;
                                        if ((m_Pending.PriceOpen < Price) && (!m_Position.IsBuy)) return;
                                        ModifyPricePoints(m_Pending.Ticket, m_Pending.PriceOpen = Price, m_Pending.SL = 0, m_Pending.TP = 0);
                                }else
                                {
                                        if (m_Position.SL == 0) return;
                                        if ((m_Position.SL > Price) && (m_Position.IsBuy)) return;
                                        if ((m_Position.SL < Price) && (!m_Position.IsBuy)) return;
                                        ModifyPricePoints(m_Position.Ticket, m_Position.PriceOpen, Price, m_Position.TP);
                                }
                        }
```

This function is designed to set a stop price at a certain point. In fact, it performs the same action as the trailing stop code when breakeven has already been activated. However, in some types of trades, we do not break even on the position, but move the stop loss based on some criteria, such as the high or low of the previous bar, the price indicated by the moving average, or another automation mechanism. In this case we need a special function to perform this task. Note that there are mechanisms that prevent the value from moving in the specified direction that would increase the loss. This type of lock is very important especially if you want to send values based on the moving average.

Based on this, we now have the following code for the trailing stop function:

```
inline void TriggerTrailingStop(void)
                        {
                                double price, v1;

                                if ((m_Position.Ticket == 0) || (m_InfosManager.IsOrderFinish ? m_Pending.Ticket == 0 : m_Position.SL == 0)) return;
                                if (m_Position.EnableBreakEven) TriggerBreakeven(); else
                                {
                                        price = SymbolInfoDouble(_Symbol, (GetTerminalInfos().ChartMode == SYMBOL_CHART_MODE_LAST ? SYMBOL_LAST : (m_Position.IsBuy ? SYMBOL_ASK : SYMBOL_BID)));
                                        v1 = (m_InfosManager.IsOrderFinish ? m_Pending.PriceOpen : m_Position.SL);
                                        if (v1 > 0) if (MathAbs(price - v1) >= (m_Position.Gap * 2))
                                                LockStopInPrice(v1 + (m_Position.Gap * (m_Position.IsBuy ? 1 : -1)));
                                        {
                                                price = v1 + (m_Position.Gap * (m_Position.IsBuy ? 1 : -1));
                                                if (m_InfosManager.IsOrderFinish) ModifyPricePoints(m_Pending.Ticket, m_Pending.PriceOpen = price, m_Pending.SL = 0, m_Pending.TP = 0);
                                                else    ModifyPricePoints(m_Position.Ticket, m_Position.PriceOpen, price, m_Position.TP);
                                        }
                                }
                        }
```

The crossed-out parts have been removed because now have a dedicated call for better code reuse.

Now that we've implemented the changes to the C\_Manager class, let's look at how to create this third example, starting with the changes. These may differ from case to case. However, you should always pay attention to the planning done in order to create automation. And since the automation here needs a little more stuff than in the previous cases, let's see the required changes. These changes will be enough for any model that uses 2 indicators at the same time.

Let's start by declaring variables:

```
class C_Automaton : public C_Manager
{
        protected:
                enum eTrigger {TRIGGER_NONE, TRIGGER_BUY, TRIGGER_SELL};
        private :
                enum eSelectMedia {MEDIA_FAST, MEDIA_SLOW};
                struct st00
                {
                        int     Shift,
                                nBars;
                        double  OverBought,
                                OverSold;
                }m_Infos;
                struct st01
                {
                        double  Buff[];
                        int     Handle;
                }m_Op[sizeof(eSelectMedia) + 1];
                int     m_nBars;
                ENUM_TIMEFRAMES m_TF;
```

Here we have an enumeration that will help us to access the MA data in a high level language to avoid errors when programming the calculations.

The next thing we have is the structure that will allow us to access both the buffer and the handle of the indicator. But pay attention to the fact that the structure is declared as an array, and the size of this array is exactly the amount of data present in the enumeration plus 1. In other words, regardless of the number of elements we are going to use, all we need to do is add them to the enumeration. In this way, the array will adapt to the final model which we are going to build.

In a way, this is a better option than the default class model. But since we could implement a simpler model, it was implemented first. Thus, it seems to me that everything becomes clearer and easier to understand.

Now you know how to add several indicators to the C\_Automaton class in a very simple way. But let's see how to actually initialize things in the class constructor:

```
                C_Automaton(const ulong magic, double FinanceStop, double FinanceTake, uint Leverage,
                                                bool IsDayTrade, double Trailing, const ENUM_TIMEFRAMES iPeriod,
                                                const double OverBought = 70, const double OverSold = 30, const int iShift = 1)
                        :C_Manager(magic, FinanceStop, FinanceTake, Leverage, IsDayTrade, Trailing, true, 10),
                         m_TF(iPeriod)
                        {
                                for (int c0 = sizeof(eSelectMedia); c0 <= 0; c0--)
                                {
                                        m_Op[c0].Handle = INVALID_HANDLE;
                                        ArraySetAsSeries(m_Op[c0].Buff, true);
                                }
                                m_Infos.Shift      = (iShift < 3 ? 3 : iShift);
                                m_Infos.OverBought = OverBought;
                                m_Infos.OverSold   = OverSold;
                                m_nBars = iBars(NULL, m_TF);
                                m_Op[MEDIA_FAST].Handle = iMA(NULL, m_TF, 9, 0, MODE_EMA, PRICE_CLOSE);
                                m_Op[MEDIA_SLOW].Handle = iMA(NULL, m_TF, 20, 0, MODE_SMA, PRICE_CLOSE);
                        }
```

This is where the magic begins. See how I initialize all indicators by default, regardless of their number. The easiest way is to use this loop. You don't have to worry about the number of indicators in the enumeration. It doesn't really matter as this loop will be able to manage them all.

Now comes one point that really needs our attention. At this stage, all indicators will use the same timeframe, but you may need different timeframes for different indicators. In this case, you will need to adjust the constructor code. However, the changes will be minimal and will only apply to the specific model you have created.

But be careful with this when capturing the handles to be used later. You must capture them, so that each of the indicators is correctly instantiated. If this is not done correctly, you can have problems with your model. Here, I am indicating that I will use the 9-period exponential moving average, and the 20-period arithmetic moving average in the system. But you can make combinations between different indicators, provided, of course, that they are needed for your operating system.

**Important note!** There is one important thing to note here:_If you are using a custom indicator that you have created, it does not have to be on the asset chart. But since you won't find it among the standard indicators, to start the handle in order to get the data of this custom indicator, you should use the [iCustom](https://www.mql5.com/en/docs/indicators/icustom) function. Please see in the documentation how to use this function in order to be able to access your custom indicator. Again, it doesn't necessarily need to be on the chart_.

Pay attention to this moment as it is really important. When in the EA, we don't inform an offset value, we won't actually be able to use the default value. This is because if we use the default value, we will have difficulties actually checking the crossover of the averages. We have to indicate a minimum offset value, and this is the value of 3. We could even use 2 to make the trigger more sensitive. However, we cannot use the value of 1, as it does not allow us to do a proper analysis. To understand why, let's see how the calculation is performed.

Once the constructor correctly initializes the data that we are going to use, we will need to make the part responsible for the calculations, so that the trigger mechanism can make the EA operate automatically. If the mechanism has multiple indicators, the system should work a little differently than when using only one indicator. This can be seen from the following code:

```
inline eTrigger CheckTrigger(void)
                        {
                                int iRet;
                                bool bOk = false;

                                if (iRet = iBars(NULL, m_TF)) > m_nBars)
                                {
                                        for (int c0 = sizeof(eSelectMedia); c0 <= 0; c0--)
                                        {
                                                if (m_Op[c0].Handle == INVALID_HANDLE) return TRIGGER_NONE;
                                                if (CopyBuffer(m_Op[c0].Handle, 0, 0, m_Infos.Shift + 1, m_Op[c0].Buff) < m_Infos.Shift + 1) return TRIGGER_NONE;
                                                bOk = true;
                                        }
                                        if (!bOk) return TRIGGER_NONE; else m_nBars = iRet;
                                        if ((m_Op[MEDIA_FAST].Buff[1] > m_Op[MEDIA_SLOW].Buff[1]) && (m_Op[MEDIA_FAST].Buff[m_Infos.Shift] < m_Op[MEDIA_SLOW].Buff[m_Infos.Shift])) return TRIGGER_BUY;
                                        if ((m_Op[MEDIA_FAST].Buff[1] < m_Op[MEDIA_SLOW].Buff[1]) && (m_Op[MEDIA_FAST].Buff[m_Infos.Shift] > m_Op[MEDIA_SLOW].Buff[m_Infos.Shift])) return TRIGGER_SELL;
                                };
                                return TRIGGER_NONE;
                        }
```

It uses a loop that performs tasks without worrying about the number of indicators used. However, it is extremely important that all of them are correctly initialized in the constructor, because without this, the calculation stage will not be able to generate buy or sell triggers.

First, we check if the indicator ID is initialized correctly. If not, then we will not have a valid trigger. Once we have completed this check, we start capturing data from the indicator buffer. If you have any doubts about using this feature, I recommend reading about the [CopyBuffer](https://www.mql5.com/en/docs/series/copybuffer) function in the documentation. This feature will be especially useful when working with custom indicators.

Once we have all the indicators with their respective buffers, we can move on to the calculation part itself.But wait a minute...What code is this before the calculation?This code serves to avoid the situation when we have placed a null list of enumerators. In this case, the calculation system will not be triggered.. Without this code, the calculation could be triggered even if we had a null list of enumerators. This would completely destoy the robustness of the system. But let's go back to the calculation system, and now, because we're using MA crossover, we have to be very careful here.

Note that we are not recklessly checking the zero (most recent) value present in the buffer. The reason is that before the timeframe actually closes, we may have a false crossover of the averages, which may lead to an accidental trigger.

Will the system check buffers only when a new bar is generated? The answer is yes, but if the averages cross at the exact time of the bar formation, the system will trigger an order. Therefore, we ignore the most recent value and analyze the previous one. This is why we set the offset to at least 2 for maximum sensitivity, or 3 as is being done in this example, so that the crossover can happen a little further from the bar being formed. But you can try to use other calculation methods. This one is only provided for demonstration, in no way should it be used on a real account.

To complete this last model, let's see something else about the system:

```
inline virtual void Triggers(void) final
                        {
#define def_HILO 20
                                if (!CtrlTimeIsPassed()) ClosePosition(); else switch (CheckTrigger())
                                {
                                        case TRIGGER_BUY:
                                                if (m_Memory == TRIGGER_SELL) ClosePosition();
                                                if (m_Memory != TRIGGER_BUY) ToMarket(ORDER_TYPE_BUY);
                                                m_Memory = TRIGGER_BUY;
                                                break;
                                        case TRIGGER_SELL:
                                                if (m_Memory == TRIGGER_BUY) ClosePosition();
                                                if (m_Memory != TRIGGER_SELL) ToMarket(ORDER_TYPE_SELL);
                                                m_Memory = TRIGGER_SELL;
                                                break;
                                }
                                LockStopInPrice(IsBuyPosition() ?  iLow(NULL, m_TF, iLowest(NULL, m_TF, MODE_LOW, def_HILO, 0)) : iHigh(NULL, m_TF, iHighest(NULL, m_TF, MODE_HIGH, def_HILO, 0)));
#undef def_HILO
                        };
```

The big advantage of this function is precisely this code, in which we have a very curious trailing stop. Thus, the order or stop price, depending on the situation, will be at the high or at the low, depending on whether we are buying or selling. The value used is very similar to the **HILO** indicator known to many B3 traders. This indicator, for those who are not familiar with it, looks for a price high or low within a certain number of bars. This code is responsible for this:Here we are looking for the LO value and here for HI; in both cases HILO is 20.

This completes the third example.

### Concluding thoughts

In this small sequence, I showed how you can develop an EA that works automatically. I tried to demonstrate it in a playful and simple way. Even with this presentation, it will still take some study and some time, in order to actually learn how to develop the EA.

I showed the main failures, problems and difficulties involved in what governs the work of a programmer while creating an EA that works automatically. But I also showed you that this can bring a lot of knowledge and change the way you actually observe the market.

I tried to present things in such a way that you can actually create a system that is safe, reliable and robust. At the same time, it should be modular, compact and very light. And you should be able to use it in conjunction with many other things. It's no use having a system that doesn't allow you to operate a variety of things at the same time. Because you won't always be able to actually make a good profit if trading only one asset.

Perhaps most of the readers will be interested in this last article in the sequence, where I explain the idea using 3 practical examples. However, please note that knowledge of the entire sequence of articles is necessary in order to take advantage of this article. But in a very simple way, I believe I have managed to convey the idea that it is not necessary to be a genius in programming or to have several courses and graduations. You just need to really understand how the MetaTrader 5 platform and the MQL5 language work.

I also presented how you can create specific circumstances for an efficient working system, even when MQL5 or MetaTrader 5 do not provide the indicator that you want to use. This was done in example 3, where I showed how to create the HILO indicator internally. But regardless of this, the system should always be correctly implemented and tested. Because there is no point in creating a wonderful system, which in the end will not give you any profit.

Concluding this series of articles, I would like to emphasize that I have not covered all the possible options. The idea was not to delve into all the details up to the creation of a modeling library for creating automated EAs. I will return to this subject in a new series of articles, where we will talk about developing a useful tool for market beginners.

Remember that the real testing of any automated EA does not take place in the MetaTrader 5 Strategy Tester, but on a DEMO account with full-fledged market work. There, the EA is tested without settings that can hide its real performance. This concludes this series, see you next time. All the codes discussed in this series are attached, so you can study and analysis them in order to really understand how an automated EA works.

**IMPORTANT NOTE: Do not use the EA available in the attachment without proper knowledge. Such EAs are provided for demonstrative and educational use only.**

If you are going to use them on a LIVE account, then you will be doing so at your own risk, as they can cause significant losses to your deposit.

Translated from Portuguese by MetaQuotes Ltd.

Original article: [https://www.mql5.com/pt/articles/11438](https://www.mql5.com/pt/articles/11438)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/11438.zip "Download all attachments in the single ZIP archive")

[EA\_Automatico\_-\_15.zip](https://www.mql5.com/en/articles/download/11438/ea_automatico_-_15.zip "Download EA_Automatico_-_15.zip")(17.81 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Market Simulation (Part 09): Sockets (III)](https://www.mql5.com/en/articles/12673)
- [Market Simulation (Part 08): Sockets (II)](https://www.mql5.com/en/articles/12672)
- [Market Simulation (Part 07): Sockets (I)](https://www.mql5.com/en/articles/12621)
- [Market Simulation (Part 06): Transferring Information from MetaTrader 5 to Excel](https://www.mql5.com/en/articles/11794)
- [Market Simulation (Part 05): Creating the C\_Orders Class (II)](https://www.mql5.com/en/articles/12598)
- [Market Simulation (Part 04): Creating the C\_Orders Class (I)](https://www.mql5.com/en/articles/12589)
- [Market Simulation (Part 03): A Matter of Performance](https://www.mql5.com/en/articles/12580)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/449672)**
(9)


![FernandoN23](https://c.mql5.com/avatar/avatar_na2.png)

**[FernandoN23](https://www.mql5.com/en/users/fernandon23)**
\|
7 Jun 2023 at 14:33

**Daniel Jose [#](https://www.mql5.com/pt/forum/441864#comment_47336067):**

You're confusing things. The 4 that the documentation says is the size, in terms of bytes, used in the return of sizeof and not the number of maximum elements that will be returned.

Daniel, thanks for the quick reply.

Still on the subject of FOR with sizeof(enum), I'll add a test script, the result obtained and one more question below.

Thank you for your guidance.

test.mq5 script

> ```
> //+------------------------------------------------------------------+
> void OnStart()
>   {
>       enum eSelectMedia {MEDIA_FAST, MEDIA_SLOW};
>       enum eSelectMeses {JANEIRO, FEVEREIRO, MARCO, ABRIL, MAIO, JUNHO, JULHO, AGOSTO, SETEMBRO, OUTUBRO, NOVEMBRO, DEZEMBRO};
>
>       struct st01
>          {
>                double  Buff[];
>                int     Handle;
>          }m_Op[sizeof(eSelectMedia) + 1];
>
>       Print("Tamanho do eSelectMedia = ", sizeof(eSelectMedia));
>       Print("Tamanho do eSelectMeses = ", sizeof(eSelectMeses));
>       Print("Tamanho do m_Op = ", ArraySize(m_Op));
>
>       Print("========= Mostrar enum eSelectMedia =========");
>       for (int i = sizeof(eSelectMedia); i >= 0; i--)
>         {
>             Print("eSelectMedia - idx = ", i);
>         }
>
>       Print("========= Mostrar enum eSelectMeses =========");
>       for (int i = sizeof(eSelectMeses); i >= 0; i--)
>         {
>             Print("eSelectMeses - idx = ", i);
>         }
>
>       Print("========= Mostrar enum m_Op =========");
>       for (int i = ArraySize(m_Op); i >= 0; i--)
>         {
>             Print("m_Op - idx = ", i);
>         }
>   }
> // End OnStart()
> //+------------------------------------------------------------------+
> ```

Result:

> ```
> 2023.06.07 09:09:10.415    teste (EURUSD,M1)    Tamanho do eSelectMedia = 4
> 2023.06.07 09:09:10.415    teste (EURUSD,M1)    Tamanho do eSelectMeses = 4
> 2023.06.07 09:09:10.415    teste (EURUSD,M1)    Tamanho do m_Op = 5
> 2023.06.07 09:09:10.415    teste (EURUSD,M1)    ========= Mostrar enum eSelectMedia =========
> 2023.06.07 09:09:10.415    teste (EURUSD,M1)    eSelectMedia - idx = 4
> 2023.06.07 09:09:10.415    teste (EURUSD,M1)    eSelectMedia - idx = 3
> 2023.06.07 09:09:10.415    teste (EURUSD,M1)    eSelectMedia - idx = 2
> 2023.06.07 09:09:10.415    teste (EURUSD,M1)    eSelectMedia - idx = 1
> 2023.06.07 09:09:10.415    teste (EURUSD,M1)    eSelectMedia - idx = 0
> 2023.06.07 09:09:10.415    teste (EURUSD,M1)    ========= Mostrar enum eSelectMeses =========
> 2023.06.07 09:09:10.415    teste (EURUSD,M1)    eSelectMeses - idx = 4
> 2023.06.07 09:09:10.415    teste (EURUSD,M1)    eSelectMeses - idx = 3
> 2023.06.07 09:09:10.415    teste (EURUSD,M1)    eSelectMeses - idx = 2
> 2023.06.07 09:09:10.415    teste (EURUSD,M1)    eSelectMeses - idx = 1
> 2023.06.07 09:09:10.415    teste (EURUSD,M1)    eSelectMeses - idx = 0
> 2023.06.07 09:09:10.415    teste (EURUSD,M1)    ========= Mostrar enum m_Op =========
> 2023.06.07 09:09:10.415    teste (EURUSD,M1)    m_Op - idx = 5
> 2023.06.07 09:09:10.415    teste (EURUSD,M1)    m_Op - idx = 4
> 2023.06.07 09:09:10.415    teste (EURUSD,M1)    m_Op - idx = 3
> 2023.06.07 09:09:10.415    teste (EURUSD,M1)    m_Op - idx = 2
> 2023.06.07 09:09:10.415    teste (EURUSD,M1)    m_Op - idx = 1
> 2023.06.07 09:09:10.415    teste (EURUSD,M1)    m_Op - idx = 0
> ```

The eSelectMedia enum has 2 elements.

The eSelectMonths enum has 12 elements.

The return of sizeof() is 4, for any of them, according to the documentation.

The m\_Op has ArraySize() = 5, because it was defined based on sizeof(eSelectMedia) + 1, according to the C\_Automaton\_v3.mqh file.

When I used sizeof(enum) in the FOR loop, the number of elements in the corresponding enum was not taken into account. The interaction considered 4, which is the return of sizeof(enum), both for the enum with 2 elements and for the enum with 12 elements.

With this, how should I create a loop that considers the exact number of elements in an enumeration?

[Incorrectly formatted](https://www.mql5.com/en/articles/24#insert-code) code edited by the moderator.

![Fernando Carreiro](https://c.mql5.com/avatar/2025/9/68d40cf8-38fb.png)

**[Fernando Carreiro](https://www.mql5.com/en/users/fmic)**
\|
7 Jun 2023 at 14:52

**[@FernandoN23](https://www.mql5.com/en/users/fernandon23) [#](https://www.mql5.com/pt/forum/441864#comment_47355860):** script test.mq5

Please use the [use the **CODE** button **(Alt -S)**](https://www.mql5.com/pt/articles/24#insert-code) when entering your code.

[![Code button in the editor](https://c.mql5.com/3/171/MQL5_Editor_Code_Button__1.png)](https://c.mql5.com/3/171/MQL5_Editor_Code_Button.png "https://c.mql5.com/3/171/MQL5_Editor_Code_Button. png")

![Augustine Amunenwa](https://c.mql5.com/avatar/2023/7/64A060D6-3A30.png)

**[Augustine Amunenwa](https://www.mql5.com/en/users/augustineamunenwa)**
\|
17 Jul 2023 at 17:13

**MetaQuotes:**

New article [Creating an EA that works automatically (Part 15): Automation (VII)](https://www.mql5.com/en/articles/11438) has been published:

Author: [Daniel Jose](https://www.mql5.com/en/users/DJ_TLoG_831 "DJ_TLoG_831")

Good day Mr. Daniel please can you share the

EA\_v1.mq5, EA\_v2.mq5 and EA\_v3.mq5 file to me. Or I wonder where I can see the attached file.

![Liang Sun](https://c.mql5.com/avatar/avatar_na2.png)

**[Liang Sun](https://www.mql5.com/en/users/liangsunxp)**
\|
21 Jul 2023 at 12:41

Hi, I'm using the EA you provided on the EURUSD 1M chart and during the process, I'm experiencing that the ClosePosition function is not closing the position successfully.

I'm guessing it's because the account type is Hedging and the position must be closed by setting the action to TRADE\_ACTION\_CLOSE\_BY instead of [TRADE\_ACTION\_DEAL](https://www.mql5.com/en/docs/constants/tradingconstants/enum_trade_request_actions "MQL5 documentation: Trade Operation Types").

![K Marcos](https://c.mql5.com/avatar/2024/3/65EA0F8A-3465.jpg)

**[K Marcos](https://www.mql5.com/en/users/kmarcoscoder)**
\|
30 Apr 2024 at 05:44

Dear Daniel. Congratulations on your articles. Please help me with a question: when I replace the code block of the "inline virtual void Triggers(void) final" method to send a [pending order](https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties#enum_order_type "MQL5 documentation: Order properties") at the high or low of the bar that triggered the moving average signal, the EA already buys 2 contract volumes, even though it is set to buy 1 volume. If I go back to the original code, it works perfectly again. If you can help me, thank you!!!

![Category Theory in MQL5 (Part 11): Graphs](https://c.mql5.com/2/55/Category-Theory-p11-avatar.png)[Category Theory in MQL5 (Part 11): Graphs](https://www.mql5.com/en/articles/12844)

This article is a continuation in a series that look at Category Theory implementation in MQL5. In here we examine how Graph-Theory could be integrated with monoids and other data structures when developing a close-out strategy to a trading system.

![Forecasting with ARIMA models in MQL5](https://c.mql5.com/2/55/Forecasting_with_ARIMA_models_in_MQL5_avatar.png)[Forecasting with ARIMA models in MQL5](https://www.mql5.com/en/articles/12798)

In this article we continue the development of the CArima class for building ARIMA models by adding intuitive methods that enable forecasting.

![Simple Mean Reversion Trading Strategy](https://c.mql5.com/2/55/Mean_reversion_avatar.png)[Simple Mean Reversion Trading Strategy](https://www.mql5.com/en/articles/12830)

Mean reversion is a type of contrarian trading where the trader expects the price to return to some form of equilibrium which is generally measured by a mean or another central tendency statistic.

![Matrices and vectors in MQL5: Activation functions](https://c.mql5.com/2/54/matrix_vector_avatar.png)[Matrices and vectors in MQL5: Activation functions](https://www.mql5.com/en/articles/12627)

Here we will describe only one of the aspects of machine learning - activation functions. In artificial neural networks, a neuron activation function calculates an output signal value based on the values of an input signal or a set of input signals. We will delve into the inner workings of the process.

[![](https://www.mql5.com/ff/si/0nfwvn6yhmgzf97n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F117%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dorder.expert%26utm_content%3Dorder.freelance%26utm_campaign%3D0622.MQL5.com.Internal&a=tunpwtbhegzufrqocbwiszessdutnobs&s=d9e7484e15300021b4066b1df77a94a1352f9e7c326d5113006bb4f6476bafeb&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=maipsizjlbxfewotvopurhtwdlyvfzwf&ssn=1769180812694908018&ssn_dr=0&ssn_sr=0&fv_date=1769180812&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F11438&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Creating%20an%20EA%20that%20works%20automatically%20(Part%2015)%3A%20Automation%20(VII)%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918081280428854&fz_uniq=5069114690228912268&sv=2552)

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