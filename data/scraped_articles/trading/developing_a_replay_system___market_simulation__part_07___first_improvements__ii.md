---
title: Developing a Replay System — Market simulation (Part 07): First improvements (II)
url: https://www.mql5.com/en/articles/10784
categories: Trading
relevance_score: 3
scraped_at: 2026-01-23T18:05:54.572604
---

[Need a reliable hosting solution for your robots?Contact your broker and find out about available Sponsored MetaTrader VPS offeringsLearn more![](https://www.mql5.com/ff/sh/0pw0dk81s56qy774z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=vljwvezfjkfbvviocwskggexlvgykvob&s=70cf8e354b9a125332533ffb65d7365abe8dde5b5c1ede9caac479a9e9df4f25&uid=&ref=https://www.mql5.com/en/articles/10784&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5069078698402971728)

MetaTrader 5 / Examples


### Introduction

In the previous article [Developing a Replay System — Market simulation (Part 06): First improvements (I)](https://www.mql5.com/en/articles/10768), we fixed some things and added tests to our replay system. These tests are aimed at ensuring the highest possible stability. At the same time, we started creating and using a configuration file for the system.

Despite our best efforts to create an intuitive and stable replay system, we still face some unresolved issues. The solutions to some of them are simple, others are more complicated.

If you watched the video at the end of the previous article, you probably noticed some shortcomings in the system that still require improvement. I decided to leave these shortcomings visible so that everyone understands that there are still many areas for improvement and correction before starting more intensive use. This does not mean that the system cannot be used for practice.

However, I don't want to add new functionality without first eliminating any nuances that could lead to instability in the system or platform. I think someone will want to use replay and practice while the market is open, waiting for a real deal to appear or using an automated EA in the platform.

However, if the replay system is not stable enough, then it is not practical to keep it running while the market is open if there is a possibility that a real trade might occur. This warning is due to the fact that the replay system may cause failures or interfere with the normal operation of other platform functions.

Therefore, as we continue to work on improving the market replay stability and convenience, we will introduce some additional resources into the code to further improve the system.

### Ensuring the control indicator remains on the chart

The first improvement we will implement will be the control indicator. Currently, when replay starts, a template is loaded. This template file must contain the indicator used to control the replay service. Up to this point everything is fine: if this indicator is absent, then manipulations with the replay service are impossible. Therefore, this indicator is extremely important for the service.

However, after downloading and until now, we could not guarantee that this indicator would remain on the chart. Now the situation will change, and we will make sure that it remains on the chart. If for some reason the indicator is removed, the replay service must be stopped immediately.

But how can we do this? How can we make the indicator remain on the chart or how can we check if it is actually there? There are several ways, but the simplest and most elegant one, in my opinion, is to use the MetaTrader 5 platform itself for checking through the MQL5 language.

As a rule, there are no specific events in the indicator, but there is one that is especially useful for us: the [DeInit](https://www.mql5.com/en/docs/runtime/event_fire#deinit) event.

This event is fired if something happens and the indicator needs to be closed and then immediately triggered by the [OnInit](https://www.mql5.com/en/docs/event_handlers/oninit) event. Thus, when calling the [OnDeInit](https://www.mql5.com/en/docs/event_handlers/ondeinit) function, can we tell the replay service that the indicator has been removed? It is possible, but there is one important point. The OnDeInit event is called not only when an indicator is deleted or a chart is closed.

It also triggers when the chart period changes or the indicator parameters change. So it looks like things are getting complicated again. However, if we look at the documentation for the OnDeInit event, we will see what we can use [deinitialization reason codes](https://www.mql5.com/en/docs/constants/namedconstants/uninit).

Looking at these codes, we note that there are two of them that will be very useful in indicating that the control indicator has been removed from the chart. So let's create the following code:

```
void OnDeinit(const int reason)
{
        switch (reason)
        {
                case REASON_REMOVE:
                case REASON_CHARTCLOSE:
                        GlobalVariableDel(def_GlobalVariableReplay);
                        break;
        }
}
```

Here we check whether the removal of the indicator from the chart has caused the DeInit event to trigger (and, accordingly, the OnDeInit function). In this case, we must inform the replay service that the indicator is no longer present on the chart and that the service must be stopped immediately.

To do this, you need to remove the global variable from the terminal, which is the link between the control indicator and the replay service. Once we remove this variable, the replay service will understand that the activity has ended and it will close.

If you look at the service code, you will notice that when it is closed, the replay symbol chart is also closed. This happens exactly when the service executes the following code:

```
void CloseReplay(void)
{
        ArrayFree(m_Ticks.Info);
        ChartClose(m_IdReplay);
        SymbolSelect(def_SymbolReplay, false);
        CustomSymbolDelete(def_SymbolReplay);
        GlobalVariableDel(def_GlobalVariableReplay);
}
```

So, when you delete the control indicator, the chart will close along with the indicator, since the replay service will forcefully close it. However, the service may not have time to close the replay symbol chart. In this case, we need to make sure that this chart is closed, even if the symbol remains in the market watch window and for some reason the replay service cannot delete it, despite all attempts.

To do this, let's add one more line to the OnDeInit function of the control indicator.

```
void OnDeinit(const int reason)
{
        switch (reason)
        {
                case REASON_REMOVE:
                case REASON_CHARTCLOSE:
                        GlobalVariableDel(def_GlobalVariableReplay);
                        ChartClose(ChartID());
                        break;
        }
}
```

Now the following happens: as soon as the control indicator is removed from the chart for any reason but the replay service cannot close the chart, the indicator itself will attempt to close it. This may seem a little counterintuitive, but I want the chart, like the replay service, to free up the platform and not cause any inconvenience in case of any failure or error.

With this implementation, we have at least a guarantee that if the control indicator is removed from the chart, the service will be stopped and the chart will be closed. However, we have another problem related to the indicator.

### Preventing the control indicator from being deleted

This is a rather serious problem, since it may happen that the indicator will remain on the chart, but its constituent elements will simply be removed or destroyed, which will not allow the indicator to be used correctly.

Fortunately, this situation is quite easy to fix. However, this moment requires special care so that it does not become a source of problems in the future. By preventing the destruction of the indicator or the removal of its elements and objects, we can create an uncontrollable monster that causes a lot of problems. To solve this problem, we will intercept and process the event of object destruction on the chart. Let's see how this is done in practice.

Let's start by adding the following line of code to the C\_Control class:

```
void Init(const bool state = false)
{
        if (m_szBtnPlay != NULL) return;
        m_id = ChartID();
        ChartSetInteger(m_id, CHART_EVENT_MOUSE_MOVE, true);
        ChartSetInteger(m_id, CHART_EVENT_OBJECT_DELETE, true);
        CreateBtnPlayPause(m_id, state);
        GlobalVariableTemp(def_GlobalVariableReplay);
        if (!state) CreteCtrlSlider();
        ChartRedraw();
}
```

By adding this line of code, we will ask MetaTrader 5 to send us an event when the chart object is removed from the screen. Simply meeting this condition does not guarantee that objects will not be deleted, but it does at least guarantee that the MetaTrader 5 platform will notify us when this happens.

To ensure that the objects are deleted when the C\_Control class is deleted, we must tell MetaTrader 5 when not to send an object removal event. One of the points in which this type of function is used is shown below:

```
~C_Controls()
{
        m_id = (m_id > 0? m_id : ChartID());
        ChartSetInteger(m_id, CHART_EVENT_OBJECT_DELETE, false);
        ObjectsDeleteAll(m_id, def_PrefixObjectName);
}
```

This way we will tell MetaTrader 5 that we **DO NOT** want it to send us an event when an object is removed from the chart, and we can remove the necessary objects without further problems.

However, it's not that simple, and there's a potential problem here. Let's look at the code below:

```
void DispatchMessage(const int id, const long &lparam, const double &dparam, const string &sparam)
{
        u_Interprocess Info;
        static int six =-1, sps;
        int x, y, px1, px2;

        switch (id)
        {
                case CHARTEVENT_OBJECT_CLICK:
                        if (sparam == m_szBtnPlay)
                        {
                                Info.s_Infos.isPlay =(bool) ObjectGetInteger(m_id, m_szBtnPlay, OBJPROP_STATE);
                                if (!Info.s_Infos.isPlay) CreteCtrlSlider(); else
                                {
                                        ObjectsDeleteAll(m_id, def_PrefixObjectName + "Slider");
                                        m_Slider.szBtnPin = NULL;
                                }
                                Info.s_Infos.iPosShift = m_Slider.posPinSlider;
                                GlobalVariableSet(def_GlobalVariableReplay, Info.Value);
                                ChartRedraw();
                        }else if (sparam == m_Slider.szBtnLeft) PositionPinSlider(m_Slider.posPinSlider - 1);
                        else if (sparam == m_Slider.szBtnRight) PositionPinSlider(m_Slider.posPinSlider + 1);
                break;

// ... The rest of the code....
```

This line will have to remove the position control bar which will trigger the object removal event.

You might think that we can simply turn off the event and then remove the control panel and turn it back on. This is true, but keep in mind that as the amount of code increases, these on and off actions become much more common than they might seem at first glance. In addition, there is one more thing: for correct representation, you need to arrange objects in a certain order.

Therefore, simply turning a delete event on and off does not guarantee that the event will be handled correctly. We need to create a more elegant and robust solution that maintains the correct order of objects so that their presentation is always the same and the user does not notice the difference in the positioning system.

The simplest solution is to create a function that turns off the 'delete' event, deletes objects in the same chain, and then turns the delete event back on. This can be easily implemented using the following code, which will perform this task in the control bar.

```
inline void RemoveCtrlSlider(void)
{
        ChartSetInteger(m_id, CHART_EVENT_OBJECT_DELETE, false);
        ObjectsDeleteAll(m_id, def_NameObjectsSlider);
        ChartSetInteger(m_id, CHART_EVENT_OBJECT_DELETE, true);
}
```

Now every time we need to remove only the control panel, we will call this function and get the desired result.

Although this may seem trivial, at the current state this procedure is used not once, but twice in the same function, as shown below:

```
void DispatchMessage(const int id, const long &lparam, const double &dparam, const string &sparam)
        {
                u_Interprocess Info;
                static int six =-1, sps;
                int x, y, px1, px2;

                switch (id)
                {
                        case CHARTEVENT_OBJECT_DELETE:
                                if(StringSubstr(sparam, 0, StringLen(def_PrefixObjectName)) == def_PrefixObjectName)
                                {
                                        if(StringSubstr(sparam, 0, StringLen(def_NameObjectsSlider)) == def_NameObjectsSlider)
                                        {
                                                RemoveCtrlSlider();
                                                CreteCtrlSlider();
                                        }else
                                        {
                                                Info.Value = GlobalVariableGet(def_GlobalVariableReplay);
                                                CreateBtnPlayPause(Info.s_Infos.isPlay);
                                        }
                                        ChartRedraw();
                                }
                                break;
                        case CHARTEVENT_OBJECT_CLICK:
                                if (sparam == m_szBtnPlay)
                                {
                                        Info.s_Infos.isPlay =(bool) ObjectGetInteger(m_id, m_szBtnPlay, OBJPROP_STATE);
                                        if (!Info.s_Infos.isPlay) CreteCtrlSlider(); else
                                        {
                                                RemoveCtrlSlider();
                                                m_Slider.szBtnPin = NULL;
                                        }
                                        Info.s_Infos.iPosShift = m_Slider.posPinSlider;
                                        GlobalVariableSet(def_GlobalVariableReplay, Info.Value);
                                        ChartRedraw();
                                }else if (sparam == m_Slider.szBtnLeft) PositionPinSlider(m_Slider.posPinSlider - 1);
                                else if (sparam == m_Slider.szBtnRight) PositionPinSlider(m_Slider.posPinSlider + 1);
                                break;
                        case CHARTEVENT_MOUSE_MOVE:
                                x =(int)lparam;
                                y =(int)dparam;
                                px1 = m_Slider.posPinSlider + def_MinPosXPin - 14;
                                px2 = m_Slider.posPinSlider + def_MinPosXPin + 14;
                                if ((((uint)sparam & 0x01) == 1) && (m_Slider.szBtnPin != NULL))
                                {
                                        if ((y >= (m_Slider.posY - 14)) && (y <= (m_Slider.posY + 14)) && (x >= px1) && (x <= px2) && (six ==-1))
                                        {
                                                6 = x;
                                                sps = m_Slider.posPinSlider;
                                                ChartSetInteger(m_id, CHART_MOUSE_SCROLL, false);
                                        }
                                        if (six > 0) PositionPinSlider(sps + x - six);
                                }else if (6 > 0)
                                {
                                        6 =-1;
                                        ChartSetInteger(m_id, CHART_MOUSE_SCROLL, true);
                                }
                                break;
                }
        }
```

Let's take a closer look at the part that is responsible for processing object deletion events. When we tell the MetaTrader 5 platform that we want to receive events when an object is removed from the chart, it generates a delete event for each deleted object. We then capture this event and can check which object was deleted.

One important detail is that you will not see which object will be deleted, but which object was actually deleted. In our case, we check if it is one of those used by the control indicator. If so, we'll double check to see if it was one of the objects on the control panel or if it was a control button. If it was one of the objects that are part of the control panel, then the panel will be completely removed and immediately created again.

We don't need to inform anything to this creation function since it does all the work itself. Now, when it comes to the control button, we have a slightly different situation. In this case, we must read the terminal's global variable to find out the current state of the button before we can request its creation.

Finally, we force all objects to be placed on the chart so that the user will not even notice that they have been removed.

We do this to ensure that everything is in its place. Now let's look at something else that is also important for the operation of the replay service.

### Only one replay chart

When working with the system that automatically opens charts, it often happens that it starts opening charts of the same symbol, and after a while we no longer understand what exactly we are dealing with.

To avoid this, I implemented a small test that solves the problem when the replay system keeps opening one chart after another for the same purpose of replaying the market. The existence of this function also guarantees a certain stability with respect to the values contained in the global terminal variable.

If we have several charts that reflect the same idea, in this case market replay, then we may find that one of them has the specified value created by the control indicator and the other one has a completely different value. Although this problem is not yet completely solved, the simple fact that we will no longer have multiple charts referencing the same symbol at the same time will bring a lot of benefits.

The below code shows a way to ensure that only one chart is opened for a given instrument:

```
long ViewReplay(ENUM_TIMEFRAMES arg1)
{
        if ((m_IdReplay = ChartFirst()) > 0) do
        {
                if(ChartSymbol(m_IdReplay) == def_SymbolReplay) ChartClose(m_IdReplay);
        }while ((m_IdReplay = ChartNext(m_IdReplay)) > 0);
        m_IdReplay = ChartOpen(def_SymbolReplay, arg1);
        ChartApplyTemplate(m_IdReplay, "Market Replay.tpl");
        ChartRedraw(m_IdReplay);
        return m_IdReplay;
}
```

Here we check if there is an open chart in the MetaTrader 5 platform terminal. If there is one, then we use it as a starting point to check which symbol is open. If the symbol is the one used for market replay, then we will close this chart.

The problem is that if the replay symbol chart is open, then we have two options: the first is to close the chart and that's exactly what we do. The second option is to end the loop, but in this second case it may happen that more than one chart of the same asset is open. Therefore, I prefer to close all open charts; we will do this until the last chart is checked. As a result, we will not have an open market replay schedule.

Next we need to open the chart containing of the market replay system, apply the template so that the control indicator can be used, force the chart to be displayed, and return the index of the open chart.

But nothing prevents us from opening new chart for the replay symbol after the system is already loaded. We could add an additional test to the service so that only one chart remains open during the entire replay period. But I know that there are traders who like to use multiple charts of the same symbol at the same time. Each of the charts will use its own time interval.

For this reason, we will not add this additional test, but will do something else. We will not allow the presence and operation of the control indicator on any chart other than the one opened by the service. Well, we could end the indicator on the original chart by trying to replace it with a different chart. But the chart would close and the replay service would stop, preventing the change from being made.

In the next topic, we'll look at how to ensure that the control indicator does not open on any other chart than the original one.

### Only one control indicator per session

This part is quite interesting and in some cases can help you. Let's look at how to make sure that the indicator belongs to only one chart in one MetaTrader 5 working session.

To understand how to do this, we will look at the following code:

```
int OnInit()
{
        u_Interprocess Info;

        IndicatorSetString(INDICATOR_SHORTNAME, "Market Replay");
        if(GlobalVariableCheck(def_GlobalVariableReplay)) Info.Value = GlobalVariableGet(def_GlobalVariableReplay); else Info.Value = 0;
        Control.Init(Info.s_Infos.isPlay);

        return INIT_SUCCEEDED;
}
```

This code will check for the presence of a global terminal variable. If such a variable is present, we will capture it for later use. If there is no variable, then we initialize it.

But there is one important detail: the OnInit function is called whenever something happens, either on the chart or when updating the indicator parameters. In this case, the indicator does not contain any parameters and will not receive them. Thus, we only have chart events that will occur every time the chart time interval changes, i.e., if we go from 5 minutes to 4 minutes, OnInit will be called. In this case, if we simply block the initialization of the indicator if there is a global terminal variable, then we will have a problem. The reason is that the chart will be closed, which means that the service will be stopped. It's difficult, isn't it?

But the solution we will use will be quite simple and at the same time very elegant. We will use the global terminal variable to know whether a control indicator already exists on any chart or not. If it exists, it cannot be placed on another chart as long as it is present on any open chart in the current MetaTrader 5 session.

To implement this, we need to edit the code used for communication between processes.

```
#property copyright "Daniel Jose"
//+------------------------------------------------------------------+
#define def_GlobalVariableReplay "Replay Infos".
//+------------------------------------------------------------------+
#define def_MaxPosSlider 400
//+------------------------------------------------------------------+
union u_Interprocess
{
        double Value;
        struct st_0
        {
                bool isPlay; // Specifies in which mode we are - Play or Pause ...
                bool IsUsing; // Specifies if the indicator is running or not ...
                int iPosShift; // Value from 0 to 400 ...
        }s_Infos;
};
```

Remember that we can add internal variables to the structure as long as it does not exceed the 8 byte limit, which is the size that a double variable takes up in memory. But since the boolean type will only use 1 bit to exist, and we have 7 free bits left in the byte that the isPlay variable uses, we can easily add 7 more boolean numbers. Therefore, we will use one of these 7 free bits in order to find out whether or not there is a control indicator on any chart.

NOTE: Although this mechanism is quite adequate, there is one problem. But we will not talk about this for now. We will consider this problem in another article in the future, when it is necessary to make changes to the structure.

So, you might think that this is enough. But we need to add something to the code. However, we will not worry about the service code, but will only change the indicator code so that the added variable is actually useful to us.

The first thing we need to do is add a few additional lines to the indicator code.

```
void Init(const bool state = false)
{
        u_Interprocess Info;

        if (m_szBtnPlay != NULL) return;
        m_id = ChartID();
        ChartSetInteger(m_id, CHART_EVENT_MOUSE_MOVE, true);
        ChartSetInteger(m_id, CHART_EVENT_OBJECT_DELETE, true);
        CreateBtnPlayPause(state);
        GlobalVariableTemp(def_GlobalVariableReplay);
        if (!state) CreteCtrlSlider();
        ChartRedraw();
        Info.Value = GlobalVariableGet(def_GlobalVariableReplay);
        Info.s_Infos.IsUsing = true;
        GlobalVariableSet(def_GlobalVariableReplay, Info.Value);
}
```

Here we inform and record in the global terminal variable that the control indicator has been created. But why do we need the previous call to create a global terminal variable? Can't we skip it? This first call actually serves to notify the MetaTrader 5 platform that the global variable is temporary and should not be maintained. Even if you ask the platform to save the data of global terminal variables, the values of these variables considered temporary will not be saved and will be lost.

This is what we need, since if we need to save and then reset global terminal variables, it is not practical to have a variable that reports the presence of a control indicator when in fact there is none. For this reason, we have to do the things this way.

You should be careful about this matter. Because when the platform re-positions the indicator on the chart, the value of the global terminal variable may be different, because we have already advanced in replay. And if we do not have this line, the system will start the replay system from scratch.

There is something more we need to do.

```
case CHARTEVENT_OBJECT_CLICK:
        if (sparam == m_szBtnPlay)
        {
                Info.s_Infos.isPlay =(bool) ObjectGetInteger(m_id, m_szBtnPlay, OBJPROP_STATE);
                if (!Info.s_Infos.isPlay) CreteCtrlSlider(); else
                {
                        RemoveCtrlSlider();
                        m_Slider.szBtnPin = NULL;
                }
                Info.s_Infos.IsUsing = true;
                Info.s_Infos.iPosShift = m_Slider.posPinSlider;
                GlobalVariableSet(def_GlobalVariableReplay, Info.Value);
                ChartRedraw();
        }else if (sparam == m_Slider.szBtnLeft) PositionPinSlider(m_Slider.posPinSlider - 1);
        else if (sparam == m_Slider.szBtnRight) PositionPinSlider(m_Slider.posPinSlider + 1);
        break;
```

Each time the pause/play button is pressed, the value of the global terminal variable changes. But with the previous code, when the button is clicked, the saved value will no longer contain an the indication that the control indicator is present on the chart. Due to this, we need to add a code line. With it, we will have a correct indication, since switching from pose to play and vice versa will not create a false indication.

The part related to the C\_Replay class is complete, but we still have a little more work to do. Simply creating an indication does not guarantee anything other than the very fact of its existence. Let's move on to the indicator code. Here you should be a little more careful so that everything works correctly and does not turn into something with strange behavior.

So, let's pay attention to details. The first thing you should pay attention to is the OnInit code:

```
int OnInit()
{
#define def_ShortName "Market Replay"
        u_Interprocess Info;

        IndicatorSetString(INDICATOR_SHORTNAME, def_ShortName);
        if(GlobalVariableCheck(def_GlobalVariableReplay))
        {
                Info.Value = GlobalVariableGet(def_GlobalVariableReplay);
                if (Info.s_Infos.IsUsing)
                {
                        ChartIndicatorDelete(ChartID(), 0, def_ShortName);
                        return INIT_FAILED;
                }
        } else Info.Value = 0;
        Control.Init(Info.s_Infos.isPlay);

        return INIT_SUCCEEDED;
#undef def_ShortName
}
```

For practical reasons, we have created here a definition that specifies the name of the indicator: this name will allow us to remove the indicator from the list of indicators present in the window where we can see which indicators are actually on the chart. Even those that are not visualized are displayed in this window. We don't want any useless indicators left there.

Therefore, after removing the indicator form the chart, we need to find out if it is already present on any chart. For this, we check the value created in the terminal global variable. This makes checking very simple and efficient. There are other ways to perform this check, but since we are using a global terminal variable, it is easier to check through it.

The rest of the feature continues in the same way, but it will no longer be possible to add a control indicator to more than one chart in a single MetaTrader 5 session. Here and in the attached code we have not added a warning that the indicator already exists on another chart, but you can add this warning before the function returns an initialization error.

This might seem enough, but there's something else we need to fix. It should be remembered that every time MetaTrader 5 receives a request to change the timeframe, which is perhaps the most common event on the platform, all indicators, like many other things, will be removed and then reset.

Now think about the following. If the indicator informs via the global variable that there is a copy of it being executed on any chart, and you change the timeframe of this specific chart, the indicator will be removed. But when the MetaTrader 5 platform restores the indicator to the chart, it will not be able to be placed on the chart. The reason for this is precisely what we saw in the code of the OnInit function. We need to somehow edit the global terminal variable so that it no longer reports the existence of a control indicator.

There are quite exotic ways to solve this problem, but, again, the MetaTrader 5 platform together with the MQL5 language offers fairly simple means for solving it. Let's look at the following code:

```
void OnDeinit(const int reason)
{
        u_Interprocess Info;

        switch (reason)
        {
                case REASON_CHARTCHANGE:
                        if(GlobalVariableCheck(def_GlobalVariableReplay))
                        {
                                Info.Value = GlobalVariableGet(def_GlobalVariableReplay);
                                Info.s_Infos.IsUsing = false;
                                GlobalVariableSet(def_GlobalVariableReplay, Info.Value);
                        }
                        break;
                case REASON_REMOVE:
                case REASON_CHARTCLOSE:
                        GlobalVariableDel(def_GlobalVariableReplay);
                        ChartClose(ChartID());
                        break;
        }
}
```

As we remember, when an indicator is deleted, the DeInit event is generated, which calls the OnDeInit function. This function receives as a parameter a value indicating the reason for the call. This is the value we will use.

This value can be seen in the [deinitialization reason codes](https://www.mql5.com/en/docs/constants/namedconstants/uninit). Here we see that REASON\_CHARTCHANGE will indicated that the chart period has changed. So we do a check - **_it is always good to check things. Never imagine or assume, but always check_**, whether there is a terminal global variable with the expected name. If this is true, we will capture the value of the variable. Since the service may be doing something, and we don't want to disturb it, we modify the information here that the control indicator will no longer be present on the chart. Once this is done, we write the information back to the global terminal variable.

Here I must warn about a possible flaw of this system. Although the probability of something going wrong is low, you should always know that there is a flaw in the method, and thus prepare for possible problems.

The problem here is that there will be a small gap between reading and writing the variable. Although it is small, it exists when the service can write a value to a global terminal variable before the indicator does. When this kind of event occurs, the value expected by the service when accessing the global terminal variable will differ from what is actually in the variable.

There are ways to get around this flaw, but in this system that works with market replay, it is not critical. So, we can ignore this flaw. However, if you want to use this same mechanism in something more complex, where the stored values are critical, then I advise you to take find about more about how to lock and unlock reading and writing of the shared memory. Well, the terminal global variable is exactly the shared memory.

From the video below, you can get an idea of some of what has been fixed and what remains to be fixed. Now things are becoming more serious.

YouTube

### Conclusion

Although the system described here seems ideal for eliminating malfunctions caused by misuse of control indicators, it is still not a truly effective solution, since it only avoids some of the types of problems that we may actually have.

I think, after watching the video, you will notice that we have to solve another problem, which is much more complicated than it seems at first glance.

Translated from Portuguese by MetaQuotes Ltd.

Original article: [https://www.mql5.com/pt/articles/10784](https://www.mql5.com/pt/articles/10784)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/10784.zip "Download all attachments in the single ZIP archive")

[Market\_Replay.zip](https://www.mql5.com/en/articles/download/10784/market_replay.zip "Download Market_Replay.zip")(13057.92 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/455083)**
(1)


![Ayivi Svenn Auguste Emmanuel O Mivedor](https://c.mql5.com/avatar/2024/2/65c80478-3d12.jpg)

**[Ayivi Svenn Auguste Emmanuel O Mivedor](https://www.mql5.com/en/users/smivedor)**
\|
26 Oct 2023 at 01:49

I am really strunggling to get this up and running, most of my issues are with the script, can anyone help?


![Developing a Replay System — Market simulation (Part 08): Locking the indicator](https://c.mql5.com/2/54/replay-p8-avatar.png)[Developing a Replay System — Market simulation (Part 08): Locking the indicator](https://www.mql5.com/en/articles/10797)

In this article, we will look at how to lock the indicator while simply using the MQL5 language, and we will do it in a very interesting and amazing way.

![Neural networks made easy (Part 38): Self-Supervised Exploration via Disagreement](https://c.mql5.com/2/54/self_supervised_exploration_via_disagreement_038_avatar.png)[Neural networks made easy (Part 38): Self-Supervised Exploration via Disagreement](https://www.mql5.com/en/articles/12508)

One of the key problems within reinforcement learning is environmental exploration. Previously, we have already seen the research method based on Intrinsic Curiosity. Today I propose to look at another algorithm: Exploration via Disagreement.

![Neural networks made easy (Part 39): Go-Explore, a different approach to exploration](https://c.mql5.com/2/54/NN_39_Go_Explore_Avatar.png)[Neural networks made easy (Part 39): Go-Explore, a different approach to exploration](https://www.mql5.com/en/articles/12558)

We continue studying the environment in reinforcement learning models. And in this article we will look at another algorithm – Go-Explore, which allows you to effectively explore the environment at the model training stage.

![Developing an MQTT client for MetaTrader 5: a TDD approach — Part 3](https://c.mql5.com/2/58/mqtt_p3_avatar.png)[Developing an MQTT client for MetaTrader 5: a TDD approach — Part 3](https://www.mql5.com/en/articles/13388)

This article is the third part of a series describing our development steps of a native MQL5 client for the MQTT protocol. In this part, we describe in detail how we are using Test-Driven Development to implement the Operational Behavior part of the CONNECT/CONNACK packet exchange. At the end of this step, our client MUST be able to behave appropriately when dealing with any of the possible server outcomes from a connection attempt.

[Need a reliable hosting solution for your robots?Contact your broker and find out about available Sponsored MetaTrader VPS offeringsLearn more![](https://www.mql5.com/ff/sh/0pw0dk81s56qy774z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=vljwvezfjkfbvviocwskggexlvgykvob&s=70cf8e354b9a125332533ffb65d7365abe8dde5b5c1ede9caac479a9e9df4f25&uid=&ref=https://www.mql5.com/en/articles/10784&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5069078698402971728)

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