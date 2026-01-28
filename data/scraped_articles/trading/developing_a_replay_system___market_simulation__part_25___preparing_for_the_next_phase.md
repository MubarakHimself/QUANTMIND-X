---
title: Developing a Replay System — Market simulation (Part 25): Preparing for the next phase
url: https://www.mql5.com/en/articles/11203
categories: Trading, Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T18:02:39.031971
---

[![](https://www.mql5.com/ff/sh/jup0jccfs9655z9z2/01.png)Learn to create your own robotsRead our book "MQL5 Programming for Traders"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/book%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.algobook%26utm_content=visit.page%26utm_campaign=algobook.promo.04.2024&a=rsxjstxkzbrlgjjrxaglpezpvrjflnvw&s=7224440013c3dbc50ba9cc078cd015fabca36df446b8e75028d6b30234663872&uid=&ref=https://www.mql5.com/en/articles/11203&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5068973673567682445)

MetaTrader 5 / Tester


### Introduction

In the previous article " [Developing a Replay System — Market simulation (Part 24): FOREX (V)](https://www.mql5.com/en/articles/11189)" I demonstrated how we can harmoniously integrate two universes that at first glance seem different. On the one hand there is a market with a Bid price-based charting, and on the other hand there is the one based on Last prices. Our purpose was to create a method that would simulate, or more accurately generate, the likely price movement by counting only bars that ideally represent a 1 minute chart time. This was a very interesting challenge. The solution presented, although effective, is not the only way to achieve this specific goal. However, since the solution turned out to be effective, I consider this phase completed. Until it is unable to solve a particular model. In this case, we will again improve the proposed solution so that it can cover the unsolved model.

There are certain corrections that we have yet to make. Although in reality this is not really about making changes, but about eliminating some functionality that may significantly interfere with elements that we still intend to implement. One of these aspects is the ability to go back in time. The question of going back in time using the control is something that must be removed from the system. This functionality turns out to be impractical in the long term and, although it does not cause problems yet, it will certainly cause problems as we implement new features. You might find the idea of going back in time using a control interesting. I actually agree that it's an interesting concept, but in practice it's not that functional. The ability to go back in time in many situations can cause headaches when dealing with the problems it creates.

Fixing this function is not particularly difficult, just a little tedious as it requires adding tests and checks for the control indication. I'm also considering removing another element from the system. I'll make this decision in this article. Along with this change in the control indication, I will also focus on some other issues that need improvement for the service to function effectively. I invite you to follow the development presented in this article, which promises to be very informative. Today we'll cover a lot of interesting concepts that are sure to benefit your learning of programming and system development. Let's start with the first topic of this article.

### Limiting the use of the control indicator

We'll start by introducing some restrictions on the control indicator so that the user cannot go back in time. By 'going back in time' I mean that after a certain amount of progress, it will no longer be possible to use the control indicator to return to a previous position. To undo the actions, you will need to close the replay/simulation service and restart the process from the beginning. I understand that this limitation may seem daunting, but trust me, this approach will prevent many future problems that may arise when trying to use the go-back functionality.

Implementing this limitation is not difficult, but it does require some effort since you need to add specific tests to the system. These tests must be used with care so as not to create conflicts with other functions of the indicator, allowing it to work effectively. We will break this task down into several steps to make it easier to implement changes in an efficient manner.

### Getting started: Turning on and off the fine-tuning buttons

The task is relatively simple at this stage. It involves enabling or disabling access to the fine-tuning buttons located at the ends of the control panel. These buttons are shown in the image below:

![Figure 01](https://c.mql5.com/2/47/001__12.png)

Figure 01: Fine-tuning buttons

These buttons make it easier to fine-tune the desired advancement speed. With them, we you can move forward or backward for a certain time with great precision, which is very useful. However, to prevent the user from going back in time, it is important to hide or show these buttons as their presence is necessary. To better understand this step, think about this: why keep the button on the left active if the system hasn't advanced a single item? Or why so we need the button on the right when the system has reached its maximum advancement limit? So, we have the last generated tickets presents. So, why do we need the right button? So, there is no need to keep it? Therefore, the purpose of this stage is to inform the user that it is impossible to move forward or backward beyond the established limit.

This task is easy and simple because the main thing is to check the limits. If we reach the limits where further movement is not possible, we should disable the button display. However, I'll take a slightly different approach, which I think makes the result more interesting. First, we won't need to write a lot of code, just make small changes. The first step involves including the bitmaps that will represent the buttons when they are disabled as a control indicator resource. This is done as follows:

```
#define def_ButtonPlay          "Images\\Market Replay\\Play.bmp"
#define def_ButtonPause         "Images\\Market Replay\\Pause.bmp"
#define def_ButtonLeft          "Images\\Market Replay\\Left.bmp"
#define def_ButtonLeftBlock     "Images\\Market Replay\\Left_Block.bmp"
#define def_ButtonRight         "Images\\Market Replay\\Right.bmp"
#define def_ButtonRightBlock    "Images\\Market Replay\\Right_Block.bmp"
#define def_ButtonPin           "Images\\Market Replay\\Pin.bmp"
#define def_ButtonWait          "Images\\Market Replay\\Wait.bmp"
#resource "\\" + def_ButtonPlay
#resource "\\" + def_ButtonPause
#resource "\\" + def_ButtonLeft
#resource "\\" + def_ButtonLeftBlock
#resource "\\" + def_ButtonRight
#resource "\\" + def_ButtonRightBlock
#resource "\\" + def_ButtonPin
#resource "\\" + def_ButtonWait
```

These lines add bitmaps inside the control indicator that symbolize disabled buttons at the limits. This makes the interface more attractive, allowing you to create buttons whose appearance is more consistent with what you want to offer the user. Feel free to make changes. Once this step is done we need to refer to these values. The code is almost ready, we just need to refer these resources. This is done in the code below:

```
void CreteCtrlSlider(void)
   {
      u_Interprocess Info;

      m_Slider.szBarSlider = def_NameObjectsSlider + " Bar";
      m_Slider.szBtnLeft   = def_NameObjectsSlider + " BtnL";
      m_Slider.szBtnRight  = def_NameObjectsSlider + " BtnR";
      m_Slider.szBtnPin    = def_NameObjectsSlider + " BtnP";
      m_Slider.posY = 40;
      CreteBarSlider(82, 436);
      CreateObjectBitMap(52, 25, m_Slider.szBtnLeft, def_ButtonLeft, def_ButtonLeftBlock);
      CreateObjectBitMap(516, 25, m_Slider.szBtnRight, def_ButtonRight, def_ButtonRightBlock);
      CreateObjectBitMap(def_MinPosXPin, m_Slider.posY, m_Slider.szBtnPin, def_ButtonPin);
      ObjectSetInteger(m_id, m_Slider.szBtnPin, OBJPROP_ANCHOR, ANCHOR_CENTER);
      if (GlobalVariableCheck(def_GlobalVariableReplay)) Info.u_Value.df_Value = GlobalVariableGet(def_GlobalVariableReplay); else Info.u_Value.df_Value = 0;
      PositionPinSlider(Info.s_Infos.iPosShift);
```

Inserting these references allows the object responsible for the buttons to process them in a way that achieves the desired result. Please note that so far I have not added anything other than resource links and the system can now perform the expected function. However, to change the buttons when the adjustment limits are reached, we need to add a little more code. But don't worry, this is a pretty simple and straightforward task. The required code is shown below:

```
inline void PositionPinSlider(int p, const int minimal = 0)
   {
      m_Slider.posPinSlider = (p < 0 ? 0 : (p > def_MaxPosSlider ? def_MaxPosSlider : p));
      ObjectSetInteger(m_id, m_Slider.szBtnPin, OBJPROP_XDISTANCE, m_Slider.posPinSlider + def_MinPosXPin);
      ObjectSetInteger(m_id, m_Slider.szBtnLeft, OBJPROP_STATE, m_Slider.posPinSlider != minimal);
      ObjectSetInteger(m_id, m_Slider.szBtnRight, OBJPROP_STATE, m_Slider.posPinSlider < def_MaxPosSlider);
      ChartRedraw();
   }
```

I have introduced a new parameter to the call, but since we will initially be using the system in standard mode, this parameter starts with a value of zero. This means that there is no need for change at this time. After that, we can begin testing the limits in certain situations. To enable or disable the button to the left of a control, we will use one calculation. To turn off the button in the right corner of the control, we apply another calculation. In the case of a right-click, the calculation will only take into account whether the slider has reached the upper limit or not. However, the left button will work differently, initially based only on the zero value. After compiling the control indicator code and running the replay/simulation service, we will see the behavior demonstrated in the animation below:

![Animation 01](https://c.mql5.com/2/47/gif_001_h1t.gif)

Animation 01: Demonstration of the button on/off system

The solution was very easy to understand and implement and was a great starting point for what we really needed to develop. Now we are faced with a slightly more difficult task which is however necessary for the user to understand what is happening. We will consider this issue in detail in the next topic.

### Notifying the user about limit changes

We could make the process quite simple by simply toggling the left button on and off when the slider reaches the minimum point specified by the replay/simulation service. However, this may confuse the user, who will not be able to move the control backwards, i.e. to the zero point. To better understand, watch the animation below:

![Animation 02](https://c.mql5.com/2/47/Gif_002.gif)

Animation 02: Why can't I get to zero?

Animation 02 clearly shows the confusion the user may experience when the slider does not reach zero even though the left button indicates that movement is not possible. This situation shows that the current indications are not clear enough. Thus, we need to improve the notification about the existing restrictions or limits due to which the slider cannot be moved beyond a certain point. Now, before describing in detail how this indication will be implemented, you might be wondering what method is used to lock the control before it reaches the zero point. Curiosity is great! I didn't resort to any complicated software tricks; I just identified a stopping point. But where? The location can be seen below:

```
inline void PositionPinSlider(int p, const int minimal = 0)
   {
      m_Slider.posPinSlider = (p < minimal ? minimal : (p > def_MaxPosSlider ? def_MaxPosSlider : p));
      ObjectSetInteger(m_id, m_Slider.szBtnPin, OBJPROP_XDISTANCE, m_Slider.posPinSlider + def_MinPosXPin);
      ObjectSetInteger(m_id, m_Slider.szBtnLeft, OBJPROP_STATE, m_Slider.posPinSlider != minimal);
      ObjectSetInteger(m_id, m_Slider.szBtnRight, OBJPROP_STATE, m_Slider.posPinSlider < def_MaxPosSlider);
      ChartRedraw();
   }
```

You must be wondering, "What was done here?" Don't worry, there is a subtle but important detail to note: the variable **minimal** is set to zero. What happens if we change this value to, say, 100 or 80? Checking the value at this point will disable the button in the left corner. However, this will not prevent the system from decreasing the value if the user left-clicks or drags the slider to the left. It's right. However, now I set the slider to the position exactly defined by the **minimal** variable. Is this clear now? No matter how much the user tries to move the slider or press the left button, the specified point will not fall below the value set as the minimum possible.

Interesting, isn't it? Determining the minimum possible value is the task of the replay/simulation service, which automatically adjusts this value as replay or simulation progresses. However, the user can change this point if the service has not changed the minimum value that can be used. This may seem complicated, but it's easier than you think. We'll get into that later. For now, let's focus on the issue raised by animation 02, which shows the lack of clear indication to the user regarding the left limit. There are several ways to do this, some may seem aesthetically strange and others a little quirky. We can choose an intermediate solution. How about creating a wall notification? In my opinion, this is a smart choice as it can offer an interesting visual aspect. If you're a graphic artist, the results might be even better than the one presented here. We will use the following code:

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
//---
      ObjectCreate(m_id, m_Slider.szBarSliderBlock, OBJ_RECTANGLE_LABEL, 0, 0, 0);
      ObjectSetInteger(m_id, m_Slider.szBarSliderBlock, OBJPROP_XDISTANCE, x);
      ObjectSetInteger(m_id, m_Slider.szBarSliderBlock, OBJPROP_YDISTANCE, m_Slider.posY - 9);
      ObjectSetInteger(m_id, m_Slider.szBarSliderBlock, OBJPROP_YSIZE, 19);
      ObjectSetInteger(m_id, m_Slider.szBarSliderBlock, OBJPROP_BGCOLOR, clrRosyBrown);
      ObjectSetInteger(m_id, m_Slider.szBarSliderBlock, OBJPROP_BORDER_TYPE, BORDER_RAISED);
   }
```

The lines highlighted in green indicate the code that creates this lower limit indication. And yes, we use an object for this. If you prefer, you can use a bitmap to get more visually appealing results. I want to keep the code simple, given that many of the readers may have limited programming knowledge. This way, more accessible code makes it easier to understand how everything was implemented. Adding a bitmap or even a texture pattern is easy and the results can be quite interesting, especially if you're using DirectX programming. And yes, MQL5 allows to do this. But we'll leave that for another time. For now, let's keep things simple yet functional. The result is shown in animation 03 below:

![Animation 03](https://c.mql5.com/2/48/Gif_003.gif)

Animation 03: Now we have a left limit.

The introduction of the left limit indicator panel has made it much easier for users to understand why they cannot go back into replay/simulation. However, you may have noticed that the code above does not specify how left limit bar is resized. The code defining this size is shown below:

```
inline void PositionPinSlider(int p, const int minimal = 0)
   {
      m_Slider.posPinSlider = (p < minimal ? minimal : (p > def_MaxPosSlider ? def_MaxPosSlider : p));
      ObjectSetInteger(m_id, m_Slider.szBtnPin, OBJPROP_XDISTANCE, m_Slider.posPinSlider + def_MinPosXPin);
      ObjectSetInteger(m_id, m_Slider.szBtnLeft, OBJPROP_STATE, m_Slider.posPinSlider != minimal);
      ObjectSetInteger(m_id, m_Slider.szBtnRight, OBJPROP_STATE, m_Slider.posPinSlider < def_MaxPosSlider);
      ObjectSetInteger(m_id, m_Slider.szBarSliderBlock, OBJPROP_XSIZE, minimal + 2);
      ChartRedraw();
   }
```

The size of the indicator bar is determined by the **minimal** variable. As the replay/modeling service updates its data, the bar will be adjusted accordingly. Now the next step is to ensure that this limit is updated correctly by the replay/simulation service. The next topic will be devoted to this.

### Talking to the replay/simulation service

Now that the core of the system is set up to not allow the user to go back in time when the control prompt appears, we need the replay/simulation service to tell the control indicator at what point the user can no longer go back in time. This task is relatively easier compared to what we have already done. The main thing is to check the current position of the replay/simulation service at the time of the pause. This part is simple. Let's see how to implement the necessary functionality. Initially, you need to make a small change to the code, which will now look like this:

```
class C_Controls
{
   private :
//+------------------------------------------------------------------+
      string  m_szBtnPlay;
      long    m_id;
      bool    m_bWait;
      struct st_00
      {
         string  szBtnLeft,
                 szBtnRight,
                 szBtnPin,
                 szBarSlider,
                 szBarSliderBlock;
         int     posPinSlider,
                 posY,
                 Minimal;
      }m_Slider;
//+------------------------------------------------------------------+
      void CreteCtrlSlider(void)
         {
            u_Interprocess Info;

            m_Slider.szBarSlider      = def_NameObjectsSlider + " Bar";
            m_Slider.szBarSliderBlock = def_NameObjectsSlider + " Bar Block";
            m_Slider.szBtnLeft        = def_NameObjectsSlider + " BtnL";
            m_Slider.szBtnRight       = def_NameObjectsSlider + " BtnR";
            m_Slider.szBtnPin         = def_NameObjectsSlider + " BtnP";
            m_Slider.posY = 40;
            CreteBarSlider(82, 436);
            CreateObjectBitMap(52, 25, m_Slider.szBtnLeft, def_ButtonLeft, def_ButtonLeftBlock);
            CreateObjectBitMap(516, 25, m_Slider.szBtnRight, def_ButtonRight, def_ButtonRightBlock);
            CreateObjectBitMap(def_MinPosXPin, m_Slider.posY, m_Slider.szBtnPin, def_ButtonPin);
            ObjectSetInteger(m_id, m_Slider.szBtnPin, OBJPROP_ANCHOR, ANCHOR_CENTER);
            if (GlobalVariableCheck(def_GlobalVariableReplay)) Info.u_Value.df_Value = GlobalVariableGet(def_GlobalVariableReplay); else Info.u_Value.df_Value = 0;
            m_Slider.Minimal = Info.s_Infos.iPosShift;
            PositionPinSlider(Info.s_Infos.iPosShift);
         }
//+------------------------------------------------------------------+
inline void PositionPinSlider(int p, const int minimal = 0)
         {
            m_Slider.posPinSlider = (p < minimal ? minimal : (p > def_MaxPosSlider ? def_MaxPosSlider : p));
            m_Slider.posPinSlider = (p < m_Slider.Minimal ? m_Slider.Minimal : (p > def_MaxPosSlider ? def_MaxPosSlider : p));
            ObjectSetInteger(m_id, m_Slider.szBtnPin, OBJPROP_XDISTANCE, m_Slider.posPinSlider + def_MinPosXPin);
            ObjectSetInteger(m_id, m_Slider.szBtnLeft, OBJPROP_STATE, m_Slider.posPinSlider != minimal);
            ObjectSetInteger(m_id, m_Slider.szBtnLeft, OBJPROP_STATE, m_Slider.posPinSlider != m_Slider.Minimal);
            ObjectSetInteger(m_id, m_Slider.szBtnRight, OBJPROP_STATE, m_Slider.posPinSlider < def_MaxPosSlider);
            ObjectSetInteger(m_id, m_Slider.szBarSliderBlock, OBJPROP_XSIZE, minimal + 2);
            ObjectSetInteger(m_id, m_Slider.szBarSliderBlock, OBJPROP_XSIZE, m_Slider.Minimal + 2);
            ChartRedraw();
         }
//+------------------------------------------------------------------+
```

You will notice that the code has undergone some simple adjustments, which are enough to ensure that the limit bar is created and configured correctly, as well as the control buttons. To do this, we had to move the variable from the function call and place it inside the structure. It will be initialized in a certain place in the code, so that it can be accessed in the appropriate places in the future. Why did I choose this approach? This was done to avoid adjustments elsewhere in the code. Each time the replay/simulation service is suspended, the CreateCtrlSlider function is called. Even if some objects are destroyed, the call to this function will still occur, which will simplify the entire creation logic.

Now that we've solved the control indicator problem, it's time to focus on the replay/simulation service code and make some changes. While many of these changes are more aesthetic in nature, it is important to ensure the system is running smoothly before tackling more complex issues.

### Solving aesthetic problems in the replay/simulation service

The first problem we need to solve is not just an aesthetic one, but a technical one. This occurs when the replay/simulation service is asked to move to a future position before replay begins. In other words, if you have just opened the service and, instead of pressing play, you decide to move the chart forward a few positions and only then launch play, then there will be a problem with the correct display of the chart. To fix this, you need to force the system to perform a "false play" and then move to the position indicated by the slider. The required code modification is shown below:

```
void AdjustPositionToReplay(const bool bViewBuider)
   {
      u_Interprocess Info;
      MqlRates       Rate[def_BarsDiary];
      int            iPos, nCount;

      Info.u_Value.df_Value = GlobalVariableGet(def_GlobalVariableReplay);
      if ((m_ReplayCount == 0) && (m_Ticks.ModePlot == PRICE_EXCHANGE))
         for (; m_Ticks.Info[m_ReplayCount].volume_real == 0; m_ReplayCount++);
      if (Info.s_Infos.iPosShift == (int)((m_ReplayCount * def_MaxPosSlider * 1.0) / m_Ticks.nTicks)) return;
      iPos = (int)(m_Ticks.nTicks * ((Info.s_Infos.iPosShift * 1.0) / (def_MaxPosSlider + 1)));
      Rate[0].time = macroRemoveSec(m_Ticks.Info[iPos].time);
      if (iPos < m_ReplayCount)
      {
         CustomRatesDelete(def_SymbolReplay, Rate[0].time, LONG_MAX);
         CustomTicksDelete(def_SymbolReplay, m_Ticks.Info[iPos].time_msc, LONG_MAX);
         if ((m_dtPrevLoading == 0) && (iPos == 0)) FirstBarNULL(); else
         {
            for(Rate[0].time -= 60; (m_ReplayCount > 0) && (Rate[0].time <= macroRemoveSec(m_Ticks.Info[m_ReplayCount].time)); m_ReplayCount--);
            m_ReplayCount++;
         }
      }else if (iPos > m_ReplayCount)
      {
      CreateBarInReplay(true);
      if (bViewBuider)
      {
         Info.s_Infos.isWait = true;
         GlobalVariableSet(def_GlobalVariableReplay, Info.u_Value.df_Value);
      }else
      {
         for(; Rate[0].time > (m_Ticks.Info[m_ReplayCount].time); m_ReplayCount++);
         for (nCount = 0; m_Ticks.Rate[nCount].time < macroRemoveSec(m_Ticks.Info[iPos].time); nCount++);
         nCount = CustomRatesUpdate(def_SymbolReplay, m_Ticks.Rate, nCount);
      }
      for (iPos = (iPos > 0 ? iPos - 1 : 0); (m_ReplayCount < iPos) && (!_StopFlag);) CreateBarInReplay(false);
      CustomTicksAdd(def_SymbolReplay, m_Ticks.Info, m_ReplayCount);
      Info.u_Value.df_Value = GlobalVariableGet(def_GlobalVariableReplay);
      Info.s_Infos.isWait = false;
      GlobalVariableSet(def_GlobalVariableReplay, Info.u_Value.df_Value);
   }
```

This code call is critical to creating the required "false play". Without this call, a chart plotting error will appear. In addition, I have included an additional line in the code that adds the missing ticks to the market watch window, providing a more realistic and interesting replay. Other changes were also made to the code, as can be seen from the crossed-out lines. These checks will prevent us from entering the system if we are in the same movement position. This is a direct consequence of our decision to not allow the user to go back in time, so codes associated with this functionality can be removed.

Now that we've fixed this flaw, let's focus on an aesthetic issue that's been around for a long time, but we now have the opportunity to address it by making the user experience with the replay/simulation service more enjoyable. This aesthetic issue occurs when a file is selected to represent previous bars on the chart. When opening a chart through the replay/simulation service, price lines are not initially displayed. Although this does not affect the functionality of the system, from an aesthetic point of view, it is inconvenient to observe the chart without a price line. To fix or deal with this part, some changes are necessary. The first of these changes is presented below:

```
bool LoopEventOnTime(const bool bViewBuider)
   {
      u_Interprocess Info;
      int iPos, iTest;

      if (!m_Infos.bInit) ViewInfos();
      if (!m_Infos.bInit)
      {
         ChartSetInteger(m_IdReplay, CHART_SHOW_ASK_LINE, m_Ticks.ModePlot == PRICE_FOREX);
         ChartSetInteger(m_IdReplay, CHART_SHOW_BID_LINE, m_Ticks.ModePlot == PRICE_FOREX);
         ChartSetInteger(m_IdReplay, CHART_SHOW_LAST_LINE, m_Ticks.ModePlot == PRICE_EXCHANGE);
         m_Infos.PointsPerTick = SymbolInfoDouble(def_SymbolReplay, SYMBOL_TRADE_TICK_SIZE);
         m_MountBar.Rate[0].time = 0;
         m_Infos.bInit = true;
         ChartRedraw(m_IdReplay);
      }
      iTest = 0;
      while ((iTest == 0) && (!_StopFlag))
      {
         iTest = (ChartSymbol(m_IdReplay) != "" ? iTest : -1);
         iTest = (GlobalVariableGet(def_GlobalVariableReplay, Info.u_Value.df_Value) ? iTest : -1);
         iTest = (iTest == 0 ? (Info.s_Infos.isPlay ? 1 : iTest) : iTest);
         if (iTest == 0) Sleep(100);
      }
      if ((iTest < 0) || (_StopFlag)) return false;
      AdjustPositionToReplay(bViewBuider);
      iPos = 0;
      while ((m_ReplayCount < m_Ticks.nTicks) && (!_StopFlag))
      {
         iPos += (int)(m_ReplayCount < (m_Ticks.nTicks - 1) ? m_Ticks.Info[m_ReplayCount + 1].time_msc - m_Ticks.Info[m_ReplayCount].time_msc : 0);
         CreateBarInReplay(true);
         while ((iPos > 200) && (!_StopFlag))
         {
            if (ChartSymbol(m_IdReplay) == "") return false;
            GlobalVariableGet(def_GlobalVariableReplay, Info.u_Value.df_Value);
            if (!Info.s_Infos.isPlay) return true;
            Info.s_Infos.iPosShift = (ushort)((m_ReplayCount * def_MaxPosSlider) / m_Ticks.nTicks);
            GlobalVariableSet(def_GlobalVariableReplay, Info.u_Value.df_Value);
            Sleep(195);
            iPos -= 200;
         }
      }
      return (m_ReplayCount == m_Ticks.nTicks);
   }
```

We remove the crossed out parts of the code and add a highlighted new line. We could include the code for this call here, but in the future this code will most likely be moved to another function. Therefore, to facilitate future portability, I prefer to collect the required code elsewhere.

To solve the aesthetic problem of price lines not being displayed immediately when the chart is opened by the replay/simulation service, the following code is required:

```
void ViewInfos(void)
   {
      MqlRates Rate[1];

      ChartSetInteger(m_IdReplay, CHART_SHOW_ASK_LINE, m_Ticks.ModePlot == PRICE_FOREX);
      ChartSetInteger(m_IdReplay, CHART_SHOW_BID_LINE, m_Ticks.ModePlot == PRICE_FOREX);
      ChartSetInteger(m_IdReplay, CHART_SHOW_LAST_LINE, m_Ticks.ModePlot == PRICE_EXCHANGE);
      m_Infos.PointsPerTick = SymbolInfoDouble(def_SymbolReplay, SYMBOL_TRADE_TICK_SIZE);
      m_MountBar.Rate[0].time = 0;
      m_Infos.bInit = true;
      CopyRates(def_SymbolReplay, PERIOD_M1, 0, 1, Rate);
      if ((m_ReplayCount == 0) && (m_Ticks.ModePlot == PRICE_EXCHANGE))

         for (; m_Ticks.Info[m_ReplayCount].volume_real == 0; m_ReplayCount++);
      if (Rate[0].close > 0)
      {
         if (m_Ticks.ModePlot == PRICE_EXCHANGE) m_Infos.tick[0].last = Rate[0].close; else
         {
            m_Infos.tick[0].bid = Rate[0].close;
            m_Infos.tick[0].ask = Rate[0].close + (Rate[0].spread * m_Infos.PointsPerTick);
         }
         m_Infos.tick[0].time = Rate[0].time;
         m_Infos.tick[0].time_msc = Rate[0].time * 1000;
      }else
         m_Infos.tick[0] = m_Ticks.Info[m_ReplayCount];
      CustomTicksAdd(def_SymbolReplay, m_Infos.tick);
      ChartRedraw(m_IdReplay);
   }
```

These lines of code were crossed out in the previous function. But the attention should be paid to the additional lines required. What we do is identify the last bar placed on the chart by the replay/simulation service using a general function for working with indicators. If we manage to capture the bar, that is, if the closing value is greater than zero, we will set a special tick depending on the construction mode used. If the close value is zero, we will use the first valid tick from the list of loaded or simulated ticks. The function responsible for searching for a valid tick is precisely in the two mentioned lines. This function will be especially useful when working with the Last plotting mode, since in Bid mode the first tick is already valid. Ultimately, this specially created tick will be displayed in the market watch window, causing price lines to appear on the chart as soon as the service commands the MetaTrader 5 platform to open the chart.

We need yet another modification because we have an issue with the replay/modeling which, although somewhat unreliable, does not point to any bars before the data set to be presented. In this case, the construction of the first bar can be cut off. To finally solve this problem, we need to specify the bar that precedes the entire set that will be presented later. This code change will allow the system to work correctly on charts with different time intervals: from 1 minute to 1 day and even 1 week. Well, I guess 1 month is too much.

```
inline void FirstBarNULL(void)
   {
      MqlRates rate[1];
      int c0 = 0;

      for(; (m_Ticks.ModePlot == PRICE_EXCHANGE) && (m_Ticks.Info[c0].volume_real == 0); c0++);
      rate[0].close = (m_Ticks.ModePlot == PRICE_EXCHANGE ? m_Ticks.Info[c0].last : m_Ticks.Info[c0].bid);
      rate[0].open = rate[0].high = rate[0].low = rate[0].close;
      rate[0].tick_volume = 0;
      rate[0].real_volume = 0;
      rate[0].time = macroRemoveSec(m_Ticks.Info[c0].time) - 86400;
      CustomRatesUpdate(def_SymbolReplay, rate);
      m_ReplayCount = 0;
   }
```

The first step is to find a valid tick, especially if the charting system uses Last prices. Once this is done, we will create the previous bar using the first valid price of the tick series, which will be used in the replay or simulation. The important aspect here is the indication of the position in time, which is corrected by subtracting the value of 1 day in minutes. This ensures that the previous bar appears correctly on the chart, and is positioned to be fully visible even on a daily chart. This system is effective for both foreign exchange market data and stock markets.

### Conclusion

Using the provided attachments, you can test the current implementation of the replay/simulation service. The basic system is ready, but since there are some functions we are not yet using, additional changes and adjustments will be required to adapt the system to a more effective training mode. At this point, we consider the replay/simulation system to be complete. In the next few articles we will look at ways to further improve it, opening a new stage in the development of this system.

Translated from Portuguese by MetaQuotes Ltd.

Original article: [https://www.mql5.com/pt/articles/11203](https://www.mql5.com/pt/articles/11203)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/11203.zip "Download all attachments in the single ZIP archive")

[Files\_-\_BOLSA.zip](https://www.mql5.com/en/articles/download/11203/files_-_bolsa.zip "Download Files_-_BOLSA.zip")(1358.24 KB)

[Files\_-\_FOREX.zip](https://www.mql5.com/en/articles/download/11203/files_-_forex.zip "Download Files_-_FOREX.zip")(3743.96 KB)

[Files\_-\_FUTUROS.zip](https://www.mql5.com/en/articles/download/11203/files_-_futuros.zip "Download Files_-_FUTUROS.zip")(11397.51 KB)

[Market\_Replay\_-\_25.zip](https://www.mql5.com/en/articles/download/11203/market_replay_-_25.zip "Download Market_Replay_-_25.zip")(49.54 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/462813)**
(2)


![fernandomsoares](https://c.mql5.com/avatar/avatar_na2.png)

**[fernandomsoares](https://www.mql5.com/en/users/fernandomsoares)**
\|
10 Sep 2023 at 23:30

Hey Daniel, good evening!

Firstly thank you very much for this great contribution to everyone who accesses your content.

I can calmly say: "You're the man!".

Daniel, I'd like to raise a point that I don't know if I'm being hasty, but which is of the utmost importance for anyone developing a robot (EA) to operate on top of the Replay service.

Is it possible to make the service trigger the metatrader's OnTick event, so that the running EA can receive [each tick](https://www.mql5.com/en/articles/239 "Article \"Basic principles of testing in MetaTrader 5\"") processed?

And for the service to wait (not plotting another tick) until the ontick event (if it exists) is executed, so we could debug the robot and the service respects this stop (of the degug).

Thanks in advance!

![Daniel Jose](https://c.mql5.com/avatar/2021/1/5FF740FF-26B8.jpg)

**[Daniel Jose](https://www.mql5.com/en/users/dj_tlog_831)**
\|
11 Sep 2023 at 19:37

**fernandomsoares each tick processed?**
**And for the service to wait (not plotting another tick) until the ontick event (if it exists) is executed, so that we could debug the robot and the service respects this stop (of the degug).**

**Thanks in advance!**

OK, let's go in parts 😁. You're not the first person to ask me that. Which in a way makes me very happy. Because I can see that many people have the same idea. Each at a certain point in the implementation. But the answer is yes and no. But why the ambiguity? The reason is that although it's simple, I don't know exactly what your level of knowledge of MQL5 is. But regardless, you can continue building your [Expert Advisor](https://www.mql5.com/en/market/mt5/ "A Market of Applications for the MetaTrader 5 and MetaTrader 4") without any problems. All I ask is that you follow carefully and study each article that is posted. Because in order to do what you and everyone else is looking for, you only need to add a single line to your Expert Advisor. This line could already be added at this stage of development of the replay / simulator. But if you're asking this, it means that you don't yet know which line to add. Take it easy. Soon, the articles will begin to explore this functionality, where the use of this same line will be quite frequent. Then you and everyone else will understand how to do it. In other words, you'll understand how to create your own solutions. With a minimum of modifications to the system I'm showing you how to implement.😁👍

PS: Thanks for the compliment. I'm here to show you that MetaTrader 5 is much more than it seems. 😉👍

![MQL5 Wizard Techniques you should know (Part 12): Newton Polynomial](https://c.mql5.com/2/70/MQL5_Wizard_Techniques_you_should_know_Part_12_Newton_Polynomial___LOGO__1.png)[MQL5 Wizard Techniques you should know (Part 12): Newton Polynomial](https://www.mql5.com/en/articles/14273)

Newton’s polynomial, which creates quadratic equations from a set of a few points, is an archaic but interesting approach at looking at a time series. In this article we try to explore what aspects could be of use to traders from this approach as well as address its limitations.

![Developing a Replay System — Market simulation (Part 24): FOREX (V)](https://c.mql5.com/2/57/replay_p24_avatar.png)[Developing a Replay System — Market simulation (Part 24): FOREX (V)](https://www.mql5.com/en/articles/11189)

Today we will remove a limitation that has been preventing simulations based on the Last price and will introduce a new entry point specifically for this type of simulation. The entire operating mechanism will be based on the principles of the forex market. The main difference in this procedure is the separation of Bid and Last simulations. However, it is important to note that the methodology used to randomize the time and adjust it to be compatible with the C\_Replay class remains identical in both simulations. This is good because changes in one mode lead to automatic improvements in the other, especially when it comes to handling time between ticks.

![Data Science and Machine Learning (Part 20): Algorithmic Trading Insights, A Faceoff Between LDA and PCA in MQL5](https://c.mql5.com/2/70/Data_Science_and_Machine_Learning_Part_20__LOGO.png)[Data Science and Machine Learning (Part 20): Algorithmic Trading Insights, A Faceoff Between LDA and PCA in MQL5](https://www.mql5.com/en/articles/14128)

Uncover the secrets behind these powerful dimensionality reduction techniques as we dissect their applications within the MQL5 trading environment. Delve into the nuances of Linear Discriminant Analysis (LDA) and Principal Component Analysis (PCA), gaining a profound understanding of their impact on strategy development and market analysis.

![Benefiting from Forex market seasonality](https://c.mql5.com/2/59/Seasonal_analysis_logo_UP.png)[Benefiting from Forex market seasonality](https://www.mql5.com/en/articles/12996)

We are all familiar with the concept of seasonality, for example, we are all accustomed to rising prices for fresh vegetables in winter or rising fuel prices during severe frosts, but few people know that similar patterns exist in the Forex market.

[![](https://www.mql5.com/ff/sh/0wxx5f0vuwq7xh89z2/01.png)VPS for 24/7 tradingContact your broker and find out how to get a free hosting subscriptionLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=nhetzvgituppcfrhndpblbihmzziogdh&s=d00c975c8bda3d8c1b29f042ad33ac81952ccea2f130a8f1ffa9015bab8ade87&uid=&ref=https://www.mql5.com/en/articles/11203&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5068973673567682445)

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