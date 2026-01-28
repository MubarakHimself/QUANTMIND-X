---
title: Developing a Replay System — Market simulation (Part 04): adjusting the settings (II)
url: https://www.mql5.com/en/articles/10714
categories: Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T19:24:56.999104
---

[We've created a channel for MQL5 developersFollow MQL5.community on social media and be the first to receive important updatesLearn more![](https://www.mql5.com/ff/sh/a83xrgctr82w45z9z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=pwgdbvtemvkwsfltqonysypfvtrtufji&s=e99a66a1660cd810b1edbac65597df695e2c2220d1e937834f402f9aeabd4289&uid=&ref=https://www.mql5.com/en/articles/10714&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5070265672744768215)

MetaTrader 5 / Examples


### Introduction

In the previous article, " [Developing a Replay System — Market simulation (Part 03): adjusting the settings (I)](https://www.mql5.com/en/articles/10706)", we created an EA that can easily manage the market replay service. So far, we have managed to implement one important point: to pause and play the system. We have not created any type of control that allows you to select the desired replay start position. That is, it is not yet possible to start replay from the middle of the period or from another specific point. We always have to start from the beginning of the data, which is not practical for those who want to do training.

This time, we are going to implement the ability to select the replay start point in the simplest possible way. We will also make a small change in the strategy at the request of some friends who liked this system and would like to use it in their own EAs. So, we will make the appropriate changes to the system to allow this.

For now, we'll take this approach to demonstrate how a new application is actually created. Many people think that this comes out of nowhere, not understanding the whole process, from the moment the idea is born to the complete stabilization of the system and code, which allows the application to do exactly what we expect.

### Exchanging the EA for the indicator

This change is fairly easy to implement. After that, we will be able to use our own EA to do research using the market replay service or to trade on the live market. For example, we will be able to use the EA that I showed in previous articles. Read more in the series " [Developing a trading EA from scratch](https://www.mql5.com/en/articles/10085)". While not designed to be 100% automated, it can be adapted for use in a replay service. But let's leave that for the future. In addition, we will also be able to use some EAs from the series " [Creating an EA that works automatically (Part 01): Concepts and structures](https://www.mql5.com/en/articles/11216)", where we looked at how to create an EA Advisor that works in a fully automated mode.

However, our current focus is not on the EA (we will explore this in the future), but on something else.

The full indicator code can be seen below. It includes exactly the functionality that already existed in the EA, while implementing it as an indicator:

```
#property copyright "Daniel Jose"
#property indicator_chart_window
#property indicator_plots 0
//+------------------------------------------------------------------+
#include <Market Replay\C_Controls.mqh>
//+------------------------------------------------------------------+
C_Controls      Control;
//+------------------------------------------------------------------+
int OnInit()
{
        IndicatorSetString(INDICATOR_SHORTNAME, "Market Replay");
        Control.Init();

        return INIT_SUCCEEDED;
}
//+------------------------------------------------------------------+
int OnCalculate(const int rates_total, const int prev_calculated, const int begin, const double &price[])
{
        return rates_total;
}
//+------------------------------------------------------------------+
void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
{
        Control.DispatchMessage(id, lparam, dparam, sparam);
}
//+------------------------------------------------------------------+
```

The only difference is the addition of a short name, which should preferably be included in the indicators. This part is highlighted in the code above. By doing this, we get an additional advantage: we can use any EA to practice and train on the replay service. Before anyone asks a possible question, I will answer it: the market replay **IS NOT** a strategy tester. It is intended for those who want to practice reading the market and thus achieve stability by improving their perception of asset movements. The market replay does not replace the great MetaTrader 5 Strategy Tester. However, the strategy tester is not suitable for practicing the market replay.

Although the conversion seems to have no side effects at first glanced, this is not entirely true. When running the replay system so that the control is done by the indicator instead of Expert Advisor, you will notice a failure. When the chart timeframe is changed, the indicator is removed from the chart and then re-launched. This operation of removing and restarting it makes the button that indicates whether we are in pause or in play mode to be inconsistent with the actual status of the replay system. To fix this, we will need to make a small adjustment. So, we will have the following indicator start code:

```
int OnInit()
{
        u_Interprocess Info;

        IndicatorSetString(INDICATOR_SHORTNAME, "Market Replay");
        if (GlobalVariableCheck(def_GlobalVariableReplay)) Info.Value = GlobalVariableGet(def_GlobalVariableReplay); else Info.Value = 0;
        Control.Init(Info.s_Infos.isPlay);

        return INIT_SUCCEEDED;
}
```

The highlighted additions in the code ensure that the state of the replay service matches the button that we see on the chart. Changes in the control code are quite simple and do not require special attention.

Now, the template file, which previously was used by the EA, will now switch to using the indicator. This leaves us completely free to make other changes in the future.

### Position control implementation

Here we are going to implement a control to indicate where we want to go inside the replay file to start our market study. But this is not going to be a precise point. The starting position will be approximate. And this is not because it is impossible to do such a thing. On the contrary, it would be much easier to indicate an exact point. However, when talking and exchanging experiences with those who have more experience in the market, we found one consensus. The ideal option is not to go to an exact point, where we already expect a specific movement, but to start the replay at a point close to the desired movement. In other words, you need to understand what's going on before you can take action.

This idea seemed so good to me that I decided: the market replay should not jump to a specific point. Although it would be easier to implement, you need to go to the nearest point. Which point will be the nearest depends on the number of trades performed per day. The more deals we execute, the more difficult it is to hit the exact spot.

So, we will access a nearby point to understand what is actually happening in order to actually create a trade simulation. Again: **we are NOT creating a strategy tester**. By doing this, over time, you will learn to determine when a movement is safer or when the risk is too high, and you should not enter into the trade.

All work in this step will be done inside the C\_Control class. So, now let's get to work!

The first thing we'll do is define some definitions.

```
#define def_ButtonLeft  "Images\\Market Replay\\Left.bmp"
#define def_ButtonRight "Images\\Market Replay\\Right.bmp"
#define def_ButtonPin   "Images\\Market Replay\\Pin.bmp"
```

Now we need to create a set of variables to store the position system data. They are implemented as follows:

```
struct st_00
{
        string  szBtnLeft,
                szBtnRight,
                szBtnPin,
                szBarSlider;
        int     posPinSlider,
                posY;
}m_Slider;
```

This is exactly what you just noticed. We are going to use a slider to point to an approximate position where the replay system should start. We now have a generic function that will be used to create the play/pause buttons and slider buttons. This function is shown below. I don't think there will be any difficulty in understanding it as it is quite simple.

```
inline void CreateObjectBitMap(int x, int y, string szName, string Resource1, string Resource2 = NULL)
                        {
                                ObjectCreate(m_id, szName, OBJ_BITMAP_LABEL, 0, 0, 0);
                                ObjectSetInteger(m_id, szName, OBJPROP_XDISTANCE, x);
                                ObjectSetInteger(m_id, szName, OBJPROP_YDISTANCE, y);
                                ObjectSetString(m_id, szName, OBJPROP_BMPFILE, 0, "::" + Resource1);
                                ObjectSetString(m_id, szName, OBJPROP_BMPFILE, 1, "::" + (Resource2 == NULL ? Resource1 : Resource2));
                        }
```

Well, now every button will be created using this function. This makes things a lot easier and increases code reuse, thus making things more stable and faster. The next thing to create will be a function that will represent the channel to be used in the slider. It is created by the following function:

```
inline void CreteBarSlider(int x, int size)
                        {
                                ObjectCreate(m_id, m_Slider.szBarSlider, OBJ_RECTANGLE_LABEL, 0, 0, 0);
                                ObjectSetInteger(m_id, m_Slider.szBarSlider, OBJPROP_XDISTANCE, x);
                                ObjectSetInteger(m_id, m_Slider.szBarSlider, OBJPROP_YDISTANCE, m_Slider.posY - 4);
                                ObjectSetInteger(m_id, m_Slider.szBarSlider, OBJPROP_XSIZE, size);
                                ObjectSetInteger(m_id, m_Slider.szBarSlider, OBJPROP_YSIZE, 9);
                                ObjectSetInteger(m_id, m_Slider.szBarSlider, OBJPROP_BGCOLOR, clrLightSkyBlue);
                                ObjectSetInteger(m_id, m_Slider.szBarSlider, OBJPROP_BORDER_COLOR, clrBlack);
                                ObjectSetInteger(m_id, m_Slider.szBarSlider, OBJPROP_WIDTH, 3);
                                ObjectSetInteger(m_id, m_Slider.szBarSlider, OBJPROP_BORDER_TYPE, BORDER_FLAT);
                        }
```

The most curious thing here is the representation of the borders of the control channel. You can customize this as you wish, as well as the width of the channel, which is set in the **OBJPROP\_YSIZE** property. But when changing the value of this property, don't forget to adjust the value that subtracts **m\_Slider.posY** so that the channel is between the buttons.

The function that creates the play/pause buttons now looks like this:

```
void CreateBtnPlayPause(long id, bool state)
{
        m_szBtnPlay = def_PrefixObjectName + "Play";
        CreateObjectBitMap(5, 25, m_szBtnPlay, def_ButtonPause, def_ButtonPlay);
        ObjectSetInteger(id, m_szBtnPlay, OBJPROP_STATE, state);
}
```

Much easier, isn't it? Now let's look at the function that will create the sliders. It is shown below:

```
void CreteCtrlSlider(void)
{
        u_Interprocess Info;

        m_Slider.szBarSlider = def_PrefixObjectName + "Slider Bar";
        m_Slider.szBtnLeft   = def_PrefixObjectName + "Slider BtnL";
        m_Slider.szBtnRight  = def_PrefixObjectName + "Slider BtnR";
        m_Slider.szBtnPin    = def_PrefixObjectName + "Slider BtnP";
        m_Slider.posY = 40;
        CreteBarSlider(82, 436);
        CreateObjectBitMap(52, 25, m_Slider.szBtnLeft, def_ButtonLeft);
        CreateObjectBitMap(516, 25, m_Slider.szBtnRight, def_ButtonRight);
        CreateObjectBitMap(def_MinPosXPin, m_Slider.posY, m_Slider.szBtnPin, def_ButtonPin);
        ObjectSetInteger(m_id, m_Slider.szBtnPin, OBJPROP_ANCHOR, ANCHOR_CENTER);
        if (GlobalVariableCheck(def_GlobalVariableReplay)) Info.Value = GlobalVariableGet(def_GlobalVariableReplay); else Info.Value = 0;
        PositionPinSlider(Info.s_Infos.iPosShift);
}
```

Look carefully at the names of the controls, this is very important. These controls will not be available while the replay system is in the replay state. The function will be called every time we are in the paused state, creating the sliders. Note that because of this, we will be capturing the value of the terminal's global variable in order to correctly identify and position the slider.

Therefore, I recommend that you do not do anything to the global variable of the terminal manually. Please pay attention to another important detail, the pin. Unlike buttons, it's designed with an anchor point exactly in the center, making it easy to find. Here we have another function call:

```
inline void PositionPinSlider(int p)
{
        m_Slider.posPinSlider = (p < 0 ? 0 : (p > def_MaxPosSlider ? def_MaxPosSlider : p));
        ObjectSetInteger(m_id, m_Slider.szBtnPin, OBJPROP_XDISTANCE, m_Slider.posPinSlider + def_MinPosXPin);
        ChartRedraw();
}
```

It will place the slider in a certain region, ensuring that it stays within the limits set above.

As you can imagine, we still need to make small adjustments to the system. Each time the chart timeframe changes, the indicator is reset, causing us to lose the current point where we actually are in the replay system. One of the ways to avoid this is to make some additions to the initialization function. Taking advantage of these changes, we will also add some additional things. Let's see what the initialization function looks like now:

```
void Init(const bool state = false)
{
        if (m_szBtnPlay != NULL) return;
        m_id = ChartID();
        ChartSetInteger(m_id, CHART_EVENT_MOUSE_MOVE, true);
        CreateBtnPlayPause(m_id, state);
        GlobalVariableTemp(def_GlobalVariableReplay);
        if (!state) CreteCtrlSlider();
        ChartRedraw();
}
```

Now we also add the code to forward mouse movement events to the indicator. Without this addition, mouse events will be lost and will not be passed to the indicator by MetaTrader 5. To hide the slider when not needed, we've added a little check. If this check confirms that the slider should be displayed, it will be displayed on the screen.

With everything we've seen so far, you might be wondering: How is event handling performed now? Will we have some super complex additional code? Well, the way mouse events are handled doesn't change much. Adding a drag event is not something very complicated. All you really have to do is manage some limits so that things don't get out of control. The implementation itself is quite simple.

Let's look at the code of the function that handles all of these events: DispatchMessage. To make it easier to explain, let's look at the code in parts. The first part is responsible for handling object click events. Look at the following code:

```
void DispatchMessage(const int id, const long &lparam, const double &dparam, const string &sparam)
{
        u_Interprocess Info;

//... other local variables ....

        switch (id)
        {
                case CHARTEVENT_OBJECT_CLICK:
                        if (sparam == m_szBtnPlay)
                        {
                                Info.s_Infos.isPlay = (bool) ObjectGetInteger(m_id, m_szBtnPlay, OBJPROP_STATE);
                                if (!Info.s_Infos.isPlay) CreteCtrlSlider(); else
                                {
                                        ObjectsDeleteAll(m_id, def_PrefixObjectName + "Slider");
                                        m_Slider.szBtnPin = NULL;
                                }
                                Info.s_Infos.iPosShift = m_Slider.posPinSlider;
                                GlobalVariableSet(def_GlobalVariableReplay, Info.Value);
                                ChartRedraw();
                        }else   if (sparam == m_Slider.szBtnLeft) PositionPinSlider(m_Slider.posPinSlider - 1);
                        else if (sparam == m_Slider.szBtnRight) PositionPinSlider(m_Slider.posPinSlider + 1);
                break;

// ... The rest of the code...
```

When we press the play/pause button, we need to perform several actions. One of them is to create a slider if we are in a paused state. If we exit the pause state and enter the play state, then the controls must be hidden from the chart so that we can no longer access them. The current value of the slider should be sent to the global variable of the terminal. Thus, the replay service has access to the percentage position in which we are or want to place the replay system.

In addition to these issues related to the play/pause buttons, we also need to deal with the events that happen when we click on the scroll's point-wise shift buttons. If we click on the scroll's left button, the slider's current value should decrease by one. Similarly, if we press the scroll's right button, it should add one to the control up to the maximum set limit.

This is quite simple. At least in this part, it's not that hard to deal with object click messages. However, now there is a slightly more complex problem with dragging the slider. To understand this, let's look at the code that handles mouse movement events. It is shown below:

```
void DispatchMessage(const int id, const long &lparam, const double &dparam, const string &sparam)
{
        u_Interprocess Info;
        static int six = -1, sps;
        int x, y, px1, px2;

        switch (id)
        {

// ... Object click EVENT ...

                case CHARTEVENT_MOUSE_MOVE:
                        x = (int)lparam;
                        y = (int)dparam;
                        px1 = m_Slider.posPinSlider + def_MinPosXPin - 14;
                        px2 = m_Slider.posPinSlider + def_MinPosXPin + 14;
                        if ((((uint)sparam & 0x01) == 1) && (m_Slider.szBtnPin != NULL))
                        {
                                if ((y >= (m_Slider.posY - 14)) && (y <= (m_Slider.posY + 14)) && (x >= px1) && (x <= px2) && (six == -1))
                                {
                                        six = x;
                                        sps = m_Slider.posPinSlider;
                                        ChartSetInteger(m_id, CHART_MOUSE_SCROLL, false);
                                }
                                if (six > 0) PositionPinSlider(sps + x - six);
                        }else if (six > 0)
                        {
                                six = -1;
                                ChartSetInteger(m_id, CHART_MOUSE_SCROLL, true);
                        }
                        break;
        }
}
```

It seems a bit more complex, but it's actually as simple as handling object clicks. The only difference is that now we will have to use more variables, and some of them must be static so that the value is not lost between calls. When the mouse is moved, MetaTrader 5 sends a message to our system. We should use this message to find out what happened and find out where the mouse cursor is, which buttons are pressed, or some other information. All this information comes in the message sent by MetaTrader 5 to our application.

When the left button is pressed, there is something to be done. But to make sure the slider is on the screen, and we don't get a false positive, we provide an extra test to ensure the integrity of what we are doing.

If the test indicates that the event is valid, we run another test to check if we are clicking on the slider, i.e. in the region belonging to the slider. At the same time, we check if this position is still valid, because it can happen that the click has already been made but the position is not valid. In this case we should ignore it. If this check succeeds, we save both the click position and the control value. We also need to lock the chart dragging. This is necessary for the next step where we will calculate the position of the slider based on the previous values present in the control. Saving this data before any calculation is very important as it makes it easier to set up and understand how to proceed in this case. But the way it is done here, it is very easy to implement as the calculation will actually be the calculation of the deviation.

When the left button is released, the situation will return to the original mode. That is, the graph will be draggable again, and the static variable used to store the position of the mouse will have a value indicating that no position is being analyzed. The same method can be used to drag and drop anything on the chart, and that's another big plus. All this is done by clicking and dragging. Then all you need to do is analyze where region which can receive clicks is. Tweak this, and the rest will be done by the code. It will look like the code shown above.

After doing this, we already have a desired behavior in the controls. But we're not done yet. We have to force the service to use the value that we specify in the slider. We will implement this in the next topic.

### Adjusting the C\_Replay class

Things are never exactly the same as some people imagine. Just because we created a slider and set up something in the control class (C\_Control) doesn't mean everything works perfectly. We need to make some adjustments to the class which actually builds the replay.

These adjustments are not very complicated. In fact, there are very few of them and they are in very specific points. However, it is important to note that any changes made to one class will affect the other. But you won't necessarily have to make any changes in other points. I prefer to never touch unnecessary points, and always take the encapsulation to its maximum level, whenever possible, thus hiding the complexity of the whole system.

Let's get straight to the main point. The first thing to do is set up the Event\_OnTime function. It is responsible for adding traded ticks to the replication asset. Basically, we are going to add a little thing to this feature. Look at the code below:

```
#define macroGetMin(A)  (int)((A - (A - ((A % 3600) - (A % 60)))) / 60)
inline int Event_OnTime(void)
{
        bool isNew;
        int mili, test;
        static datetime _dt = 0;
        u_Interprocess Info;

        if (m_ReplayCount >= m_ArrayCount) return -1;
        if (m_dt == 0)
        {
                m_Rate[0].close = m_Rate[0].open =  m_Rate[0].high = m_Rate[0].low = m_ArrayInfoTicks[m_ReplayCount].Last;
                m_Rate[0].tick_volume = 0;
                m_Rate[0].time = m_ArrayInfoTicks[m_ReplayCount].dt - 60;
                CustomRatesUpdate(def_SymbolReplay, m_Rate, 1);
                _dt = TimeLocal();
        }
        isNew = m_dt != m_ArrayInfoTicks[m_ReplayCount].dt;
        m_dt = (isNew ? m_ArrayInfoTicks[m_ReplayCount].dt : m_dt);
        mili = m_ArrayInfoTicks[m_ReplayCount].milisec;
        do
        {
                while (mili == m_ArrayInfoTicks[m_ReplayCount].milisec)
                {
                        m_Rate[0].close = m_ArrayInfoTicks[m_ReplayCount].Last;
                        m_Rate[0].open = (isNew ? m_Rate[0].close : m_Rate[0].open);
                        m_Rate[0].high = (isNew || (m_Rate[0].close > m_Rate[0].high) ? m_Rate[0].close : m_Rate[0].high);
                        m_Rate[0].low = (isNew || (m_Rate[0].close < m_Rate[0].low) ? m_Rate[0].close : m_Rate[0].low);
                        m_Rate[0].tick_volume = (isNew ? m_ArrayInfoTicks[m_ReplayCount].Vol : m_Rate[0].tick_volume + m_ArrayInfoTicks[m_ReplayCount].Vol);
                        isNew = false;
                        m_ReplayCount++;
                }
                mili++;
        }while (mili == m_ArrayInfoTicks[m_ReplayCount].milisec);
        m_Rate[0].time = m_dt;
        CustomRatesUpdate(def_SymbolReplay, m_Rate, 1);
        mili = (m_ArrayInfoTicks[m_ReplayCount].milisec < mili ? m_ArrayInfoTicks[m_ReplayCount].milisec + (1000 - mili) : m_ArrayInfoTicks[m_ReplayCount].milisec - mili);
        test = (int)((m_ReplayCount * def_MaxPosSlider) / m_ArrayCount);
        GlobalVariableGet(def_GlobalVariableReplay, Info.Value);
        if (Info.s_Infos.iPosShift != test)
        {
                Info.s_Infos.iPosShift = test;
                GlobalVariableSet(def_GlobalVariableReplay, Info.Value);
        }

        return (mili < 0 ? 0 : mili);
};
#undef macroGetMin
```

In this function, we build 1-minute bars. Pay attention that we have added a variable in this part: this variable did not exist in the above code. We will now have a representation of the relative percentage position stored in the terminal's global variable. Therefore, we need this variable to decode the internal content stored in the terminal variable. Once the traded tick has been added to the 1-minute bar, we need to know at what percentage the current replay position is. This is done in this calculation, where we find out the relative position in relation to the total number of saved ticks.

This value is then compared with the value stored in the terminal's global variable. If they are different, we update the value so that the system indicates the correct relative position when it stops. This way, you won't have to do extra calculations or run into unnecessary problems.

This concludes the first stage. However, we have one more problem to solve. How to position the replay system in the desired relative position after adjusting the value during the pause?

This problem is a little more complicated. This is because we can have both addition, which is simpler to solve, and subtraction, which is a little more complicated. This subtraction is not the big problem, at least at this stage of development. But it will be such a problem in the next phase, which we will see in the next article in this series. But the first thing to do is to add an extra function in the C\_Replay class for adding or subtracting bars from the replay system. Let's see the preparation of this function:

```
int AdjustPositionReplay()
{
        u_Interprocess Info;
        int test = (int)((m_ReplayCount * def_MaxPosSlider * 1.0) / m_ArrayCount);

        Info.Value = GlobalVariableGet(def_GlobalVariableReplay);
        if (Info.s_Infos.iPosShift == test) return 0;
        test = (int)(m_ArrayCount * ((Info.s_Infos.iPosShift * 1.0) / def_MaxPosSlider));
        if (test < m_ReplayCount)
        {
                CustomRatesDelete(def_SymbolReplay, 0, LONG_MAX);
                CustomTicksDelete(def_SymbolReplay, 0, LONG_MAX);
                m_ReplayCount = 0;
                m_Rate[0].close = m_Rate[0].open =  m_Rate[0].high = m_Rate[0].low = m_ArrayInfoTicks[m_ReplayCount].Last;
                m_Rate[0].tick_volume = 0;
                m_Rate[0].time = m_ArrayInfoTicks[m_ReplayCount].dt - 60;
                CustomRatesUpdate(def_SymbolReplay, m_Rate, 1);
        };
        for (test = (test > 0 ? test - 1 : 0); m_ReplayCount < test; m_ReplayCount++)
                Event_OnTime();

        return Event_OnTime();
}
```

In the code above, we see a code that is the basis of this customization system. Let's understand what happens in this basic system. First we generate the percentage value of the current position. Then compare this value with the value found in the terminal's global variable. The control system records this value stored in a global variable. If the values are equal (it's not an absolute value, but a percentage value), the function exits because we're at the correct percentage point, or the user didn't change position during the system pause.

But if the values are different, the absolute value is generated based on the percentage value specified in the terminal's global variable. That is, now we will have an absolute point from which the replay system should start. This value is unlikely to be equal to the counter of trading ticks for a number of reasons. If it is less than the current value of the replay counter, all data present in the current resource will be deleted.

This is tricky, but not at this stage of development. It will be done in the next step. For now, there is no reason to be overly concerned. Now we can do something common for both situations: add new values until the position of the replay counter equals the absolute position minus 1. This minus 1 has the reason to allow this function to return a value that will be used later as a delay. This is achieved by the Event\_OnTime function.

Because of this type of change, it never comes without pain. Let's see what needs to be modified in the system. This is shown in the code below. It is the only place that has been changed:

```
#property service
#property copyright "Daniel Jose"
#property version   "1.00"
//+------------------------------------------------------------------+
#include <Market Replay\C_Replay.mqh>
//+------------------------------------------------------------------+
input string    user01 = "WINZ21_202110220900_202110221759"; //File with ticks
//+------------------------------------------------------------------+
C_Replay        Replay;
//+------------------------------------------------------------------+
void OnStart()
{
        ulong t1;
        int delay = 3;
        long id;
        u_Interprocess Info;
        bool bTest = false;

        if (!Replay.CreateSymbolReplay(user01)) return;
        id = Replay.ViewReplay();
        Print("Wait for permission to start replay ...");
        while (!GlobalVariableCheck(def_GlobalVariableReplay)) Sleep(750);
        Print("Replay system started ...");
        t1 = GetTickCount64();
        while ((ChartSymbol(id) != "") && (GlobalVariableGet(def_GlobalVariableReplay, Info.Value)))
        {
                if (!Info.s_Infos.isPlay)
                {
                        if (!bTest) bTest = (Replay.Event_OnTime() > 0);
                }else
                {
                        if (bTest)
                        {
                                delay = ((delay = Replay.AdjustPositionReplay()) == 0 ? 3 : delay);
                                bTest = false;
                                t1 = GetTickCount64();
                        }else if ((GetTickCount64() - t1) >= (uint)(delay))
                        {
                                if ((delay = Replay.Event_OnTime()) < 0) break;
                                t1 = GetTickCount64();
                        }
                }
        }
        Replay.CloseReplay();
        Print("Replay system stopped ...");
}
//+------------------------------------------------------------------+
```

While we are in the pause mode, we will run this test to see if we are changing the state of the service. When this happens, we will ask the C\_Replay class to perform the new positioning, which may or may not be executed.

If executed, we will have the value of the next delay which will be used after this adjustment has been made and the system has been positioned. If necessary, we will naturally continue the remaining time until we exit the replay state and enter the pause state. Then the whole procedure will be repeated again.

### Conclusion

The video shows the whole system in operation, you can see how everything happens. However, it is important to note that you will need to wait until things stabilize before using the replay system. When moving a position to the desired point, the movement may seem difficult to perform.

This situation will be corrected in the future. But we can take it for now, as we still have a lot to figure out.

YouTube

In the attachment, I have included two real market tick files so that you can experiment with the movement and positioning system on days with different numbers of traded ticks. So, you can see how the percentage system works. This complicates things for those who want to study a specific moment in the market, but this is precisely our intention, as explained at the beginning of the article.

With this replay system, which we are building here, you will really learn how to analyze the market. There will be no exact place where you will say, "HERE... this is where I should enter." Because it may happen so that the movement that you have observed actually occurs a few bars away. Therefore, you will have to learn how to analyze the market, otherwise you may not like this replay system presented in this series of articles.

Translated from Portuguese by MetaQuotes Ltd.

Original article: [https://www.mql5.com/pt/articles/10714](https://www.mql5.com/pt/articles/10714)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/10714.zip "Download all attachments in the single ZIP archive")

[Market\_Replay.zip](https://www.mql5.com/en/articles/download/10714/market_replay.zip "Download Market_Replay.zip")(10795.89 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/452038)**
(3)


![fernandomsoares](https://c.mql5.com/avatar/avatar_na2.png)

**[fernandomsoares](https://www.mql5.com/en/users/fernandomsoares)**
\|
11 Aug 2023 at 11:49

Hello Daniel!

Firstly, I would like to thank you very much for your willingness, enthusiasm and competence in teaching what you know.

I've been following your articles and I'd also like to say that the [projects](https://www.mql5.com/en/articles/7863 "Article: Projects let you create profitable trading robots! But it's not exactly") proposed in them are very useful for studying and using metatrader 5.

You're bringing together knowledge and showing how to apply it to really useful things, and that's extraordinary. Thank you!

Now, returning to the discussion about the article, you say that you can use Replay with your own EA, and that's great.

I'm developing an EA to trade the B3 mini-index market, and what it does is analyse the flow of tick data.

I'm having a lot of problems running the EA on past days via Metatrader 5 replay, because I need the tick by tick movement and it doesn't run all the ticks of the day's movement, it skips many seconds of data.

I've seen that the Replay service you're running and publishing can help me with this problem, since I download the tick data for the days from the brokers and store it.

Can I run the Replay service and run my EA reading the ticks from the service (downloaded file) and debug my EA? I ask this because I was wondering if, when I stopped debugging my EA, the service would also stop plotting the ticks on the chart. Would you have an example of how I could make the EA call in the Replay project?

![Daniel Jose](https://c.mql5.com/avatar/2021/1/5FF740FF-26B8.jpg)

**[Daniel Jose](https://www.mql5.com/en/users/dj_tlog_831)**
\|
11 Aug 2023 at 14:23

**fernandomsoares projects proposed in them are very useful for studying and using metatrader 5.**
**You're bringing together knowledge and showing how to apply it to really useful things, and that's extraordinary. Thank you!**

**Now, returning to the discussion about the article, you say that we can use Replay with our own EA, and this is great.**

**I'm developing an EA to trade the B3 mini-index market, and what it does is analyse the flow of tick data.**

**I'm having a lot of problems running the EA on past days via Metatrader 5 replay, because I need the tick-by-tick movement and it doesn't run all the ticks of the day's movement, it skips many seconds of data.**

**I have seen that the Replay that you are making and publishing can help me with this problem, since I download the data of the ticks of the days from the brokers and store it.**

**Can I run the Replay service and run my EA reading the ticks from the service (downloaded file) and debug my EA? I ask this because I was wondering if, when I stopped debugging my EA, the service would also stop plotting the ticks on the chart. Would you have an example of how I could make the EA call in the Replay project?**

Hum.... This is a curious question. I liked your question 😁👍. Since it could also be a question for other people. Thank you for raising it publicly.

And the answer is YES 😁👍. Yes, there is a way for you to use your [Expert Advisor](https://www.mql5.com/en/market/mt5/ "A Market of Applications for the MetaTrader 5 and MetaTrader 4") and tell the replay / simulator service to stop plotting ticks on the chart. To do this, you need to do something very specific. And since you're asking, I don't think you have any idea how it actually works. So I'd rather not tell you right now. But don't worry, I've already written the articles explaining in detail how to do what you want. They're going to take a while to come out, because before I explained how to do this. I had to explain something else that allows you to understand exactly what you want to do.

Have a little patience and keep following and studying the articles on the replay / simulator. And keep developing your Expert Advisor, and keep downloading and storing ticks. Because in order for your Expert Advisor to tell the service what you want it to do. **You only need to add one line of code**. That's what you've just read. You just need to add a single line of code and everything will work the way you want it to. But before you do that, you need to understand what this single line of code will do in MetaTrader 5. And explaining it in a comment is not so simple. 😁👍

![fernandomsoares](https://c.mql5.com/avatar/avatar_na2.png)

**[fernandomsoares](https://www.mql5.com/en/users/fernandomsoares)**
\|
17 Aug 2023 at 19:58

Thank you for your attention and reply. I'm going to keep following and I'm going to be very excited when you publish this. It will be very, very helpful in debugging and improving my EA. Thank you very much! May God continue to bless you!


![The RSI Deep Three Move Trading Technique](https://c.mql5.com/2/57/The_RSI_Deep_Three_Move_avatar.png)[The RSI Deep Three Move Trading Technique](https://www.mql5.com/en/articles/12846)

Presenting the RSI Deep Three Move Trading Technique in MetaTrader 5. This article is based on a new series of studies that showcase a few trading techniques based on the RSI, a technical analysis indicator used to measure the strength and momentum of a security, such as a stock, currency, or commodity.

![Everything you need to learn about the MQL5 program structure](https://c.mql5.com/2/57/about_mql5_program_structure_avatar.png)[Everything you need to learn about the MQL5 program structure](https://www.mql5.com/en/articles/13021)

Any Program in any programming language has a specific structure. In this article, you will learn essential parts of the MQL5 program structure by understanding the programming basics of every part of the MQL5 program structure that can be very helpful when creating our MQL5 trading system or trading tool that can be executable in the MetaTrader 5.

![Improve Your Trading Charts With Interactive GUI's in MQL5 (Part III): Simple Movable Trading GUI](https://c.mql5.com/2/57/movable_gui_003_avatar.png)[Improve Your Trading Charts With Interactive GUI's in MQL5 (Part III): Simple Movable Trading GUI](https://www.mql5.com/en/articles/12923)

Join us in Part III of the "Improve Your Trading Charts With Interactive GUIs in MQL5" series as we explore the integration of interactive GUIs into movable trading dashboards in MQL5. This article builds on the foundations set in Parts I and II, guiding readers to transform static trading dashboards into dynamic, movable ones.

![Trading strategy based on the improved Doji candlestick pattern recognition indicator](https://c.mql5.com/2/53/doji_candlestick_pattern_avatar.png)[Trading strategy based on the improved Doji candlestick pattern recognition indicator](https://www.mql5.com/en/articles/12355)

The metabar-based indicator detected more candles than the conventional one. Let's check if this provides real benefit in the automated trading.

[![](https://www.mql5.com/ff/si/x6w0dk14xy0tf97n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F586%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dhow.test.expert%26utm_content%3Dbuy.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=repptjucjbnrxhoeoqbekpbncvsnhylz&s=3da978a0c510a6306b46ee79cdf8418a5c0da5e081f296e18b262b00031a2310&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=hnbmqpqorxraapwdidnhkacetuirstan&ssn=1769185495108417682&ssn_dr=0&ssn_sr=0&fv_date=1769185495&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F10714&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Developing%20a%20Replay%20System%20%E2%80%94%20Market%20simulation%20(Part%2004)%3A%20adjusting%20the%20settings%20(II)%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918549553822681&fz_uniq=5070265672744768215&sv=2552)

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