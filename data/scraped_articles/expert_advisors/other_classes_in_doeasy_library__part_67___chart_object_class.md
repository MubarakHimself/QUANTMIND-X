---
title: Other classes in DoEasy library (Part 67): Chart object class
url: https://www.mql5.com/en/articles/9213
categories: Expert Advisors
relevance_score: 0
scraped_at: 2026-01-24T14:12:45.280446
---

[![](https://www.mql5.com/ff/sh/6zw0dkux8bqt7m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
MQL5 Channels - Messenger for traders\\
\\
Install the app and receive market analytics and trading tips.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=iuciwacmrxvmiibwyujliagqikizpsoo&s=268cbb13914c54b6c5c875db99b154944f6e0122b3400b54c9ac0d4f69f0f0d6&uid=&ref=https://www.mql5.com/en/articles/9213&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5083409342117452586)

MetaTrader 5 / Examples


### Contents

- [Concept](https://www.mql5.com/en/articles/9213#node01)
- [Improving library classes](https://www.mql5.com/en/articles/9213#node02)
- [Chart object class](https://www.mql5.com/en/articles/9213#node03)
- [Test](https://www.mql5.com/en/articles/9213#node04)
- [What's next?](https://www.mql5.com/en/articles/9213#node05)


### Concept

In this article, I will start the development of the library functionality aimed at working with symbol charts. This is the main working tool and I will try to make it very convenient. This will take a few articles. First, I will create a chart object which is to store all chart properties. We will be able to conveniently manage a chart by changing them. A set of chart object properties is to be [a set of integer, real and string chart parameters](https://www.mql5.com/en/docs/constants/chartconstants/enum_chart_property). Along with a gradual and sequential chart object refinement, its set of parameters will change — some parameters will be added, some will be moved to other objects, etc. But let's start simple.

Apart from creating a chart object, I will also slightly improve [the MQL5 signal object class](https://www.mql5.com/en/articles/9095#node04) and [the collection class of MQL5.com Signals](https://www.mql5.com/en/articles/9146). Currently, working with the signal collection is arranged so that signal properties no longer change during the first creation of the full list of all available signals. Even if we try to update the collection list anew, it will simply obtain a signal that has newly appeared in MQL5.com Signals database. The signals previously added to the collection list will remain unchanged. This behavior is incorrect since signal properties change as a result of trading on a signal provider's account. Therefore, I will make it so that new signals are added to the list, while the properties of existing signals are updated during any update of the signal collection list.

### Improving library classes

Currently, in the signal collection class of the library, all [MQL5.com Signals](https://www.mql5.com/en/signals) parameter values are immediately entered into the appropriate properties of the MQL5 signal object in its constructor when creating a new signal object. We need to add the method, in which all values of selected signal parameters are set to the object properties. Thus, we will be able to select a necessary signal and call the new method to update the properties of the already existing signal object. We will also need the method selecting the necessary signal in MQL5.com signal datavase by the signal ID (in MQL5, the signal is selected only by its index but storing a signal index in the object properties is unreliable since signal indices in the database may change).

In \\MQL5\\Include\\DoEasy\\Objects\\MQLSignalBase\ **MQLSignal.mqh**, add three new public methods:

the method for writing parameter values of a selected MQL5.com signal into the appropriate signal object properties,

the method returning a signal index in the MQL5.com signal database by its ID and

the method selecting a signal having a specified ID in the MQL5.com signal database for further work:

```
//--- Compare CMQLSignal objects by a specified property (to sort the list by an MQL5 signal object)
   virtual int       Compare(const CObject *node,const int mode=0) const;
//--- Compare CMQLSignal objects by all properties (to search for equal MQL5 signal objects)
   bool              IsEqual(CMQLSignal* compared_obj) const;

//--- Set signal object properties
   void              SetProperties(void);
//--- Look for a signal with a specified ID in the database, return the signal index
   int               IndexBase(const long signal_id);
//--- Select a signal in the signal database by its ID
   bool              SelectBase(const long signal_id);

//--- Constructors
                     CMQLSignal(){;}
                     CMQLSignal(const long signal_id);
```

Pass the signal parameters entry to the appropriate object properties in the new SetProperties() method from the parametric class constructor:

```
//+------------------------------------------------------------------+
//| Set object properties                                            |
//+------------------------------------------------------------------+
void CMQLSignal::SetProperties(void)
  {
   this.m_long_prop[SIGNAL_MQL5_PROP_SUBSCRIPTION_STATUS]            = (::SignalInfoGetInteger(SIGNAL_INFO_ID)==this.ID());
   this.m_long_prop[SIGNAL_MQL5_PROP_TRADE_MODE]                     = ::SignalBaseGetInteger(SIGNAL_BASE_TRADE_MODE);
   this.m_long_prop[SIGNAL_MQL5_PROP_DATE_PUBLISHED]                 = ::SignalBaseGetInteger(SIGNAL_BASE_DATE_PUBLISHED);
   this.m_long_prop[SIGNAL_MQL5_PROP_DATE_STARTED]                   = ::SignalBaseGetInteger(SIGNAL_BASE_DATE_STARTED);
   this.m_long_prop[SIGNAL_MQL5_PROP_DATE_UPDATED]                   = ::SignalBaseGetInteger(SIGNAL_BASE_DATE_UPDATED);
   this.m_long_prop[SIGNAL_MQL5_PROP_LEVERAGE]                       = ::SignalBaseGetInteger(SIGNAL_BASE_LEVERAGE);
   this.m_long_prop[SIGNAL_MQL5_PROP_PIPS]                           = ::SignalBaseGetInteger(SIGNAL_BASE_PIPS);
   this.m_long_prop[SIGNAL_MQL5_PROP_RATING]                         = ::SignalBaseGetInteger(SIGNAL_BASE_RATING);
   this.m_long_prop[SIGNAL_MQL5_PROP_SUBSCRIBERS]                    = ::SignalBaseGetInteger(SIGNAL_BASE_SUBSCRIBERS);
   this.m_long_prop[SIGNAL_MQL5_PROP_TRADES]                         = ::SignalBaseGetInteger(SIGNAL_BASE_TRADES);

   this.m_double_prop[this.IndexProp(SIGNAL_MQL5_PROP_BALANCE)]      = ::SignalBaseGetDouble(SIGNAL_BASE_BALANCE);
   this.m_double_prop[this.IndexProp(SIGNAL_MQL5_PROP_EQUITY)]       = ::SignalBaseGetDouble(SIGNAL_BASE_EQUITY);
   this.m_double_prop[this.IndexProp(SIGNAL_MQL5_PROP_GAIN)]         = ::SignalBaseGetDouble(SIGNAL_BASE_GAIN);
   this.m_double_prop[this.IndexProp(SIGNAL_MQL5_PROP_MAX_DRAWDOWN)] = ::SignalBaseGetDouble(SIGNAL_BASE_MAX_DRAWDOWN);
   this.m_double_prop[this.IndexProp(SIGNAL_MQL5_PROP_PRICE)]        = ::SignalBaseGetDouble(SIGNAL_BASE_PRICE);
   this.m_double_prop[this.IndexProp(SIGNAL_MQL5_PROP_ROI)]          = ::SignalBaseGetDouble(SIGNAL_BASE_ROI);

   this.m_string_prop[this.IndexProp(SIGNAL_MQL5_PROP_AUTHOR_LOGIN)] = ::SignalBaseGetString(SIGNAL_BASE_AUTHOR_LOGIN);
   this.m_string_prop[this.IndexProp(SIGNAL_MQL5_PROP_BROKER)]       = ::SignalBaseGetString(SIGNAL_BASE_BROKER);
   this.m_string_prop[this.IndexProp(SIGNAL_MQL5_PROP_BROKER_SERVER)]= ::SignalBaseGetString(SIGNAL_BASE_BROKER_SERVER);
   this.m_string_prop[this.IndexProp(SIGNAL_MQL5_PROP_NAME)]         = ::SignalBaseGetString(SIGNAL_BASE_NAME);
   this.m_string_prop[this.IndexProp(SIGNAL_MQL5_PROP_CURRENCY)]     = ::SignalBaseGetString(SIGNAL_BASE_CURRENCY);
  }
//+------------------------------------------------------------------+
```

Since a signal should be preliminarily selected in the MQL5.com Signals database, obtaining signal parameter values and writing them to the object properties implies that the signal is selected beforehand.

In the class constructor, introduce calling the method instead of strings passed to a new method:

```
//+------------------------------------------------------------------+
//| Parametric constructor                                           |
//+------------------------------------------------------------------+
CMQLSignal::CMQLSignal(const long signal_id)
  {
   this.m_long_prop[SIGNAL_MQL5_PROP_ID] = signal_id;
   this.SetProperties();
  }
//+------------------------------------------------------------------+
```

**The method returning the index of a signal specified by ID in the MQL5.com Signals database:**

```
//+------------------------------------------------------------------+
//| Look for a signal with a specified ID in the database,           |
//| return the index of a detected signal                            |
//+------------------------------------------------------------------+
int CMQLSignal::IndexBase(const long signal_id)
  {
   int total=::SignalBaseTotal();
   for(int i=0;i<total;i++)
     {
      if(::SignalBaseSelect(i) && ::SignalBaseGetInteger(SIGNAL_BASE_ID)==signal_id)
         return i;
     }
   return WRONG_VALUE;
  }
//+------------------------------------------------------------------+
```

Here, in the loop by the total number of signals, select the next signaland compare its ID to the one passed to the method. If IDs match, return the loop index (which is a signal index in the database). If no signal with such ID is found, return -1.

After the method operation, the found signal remains selected in the Signals database for further work.

**The following method selecting a signal specified by ID is based on that:**

```
//+------------------------------------------------------------------+
//| Select a signal by its ID in the signal database                 |
//+------------------------------------------------------------------+
bool CMQLSignal::SelectBase(const long signal_id)
  {
   return(this.IndexBase(signal_id)!=WRONG_VALUE);
  }
//+------------------------------------------------------------------+
```

The method returns the flag indicating that the search for a signal by a specified ID has no returned -1, i.e. if the signal index is found (it is not equal to -1), the signal is selected (true is returned). If searching for a signal returns -1, false is returned meaning there is no signal with such ID and it is not selected, accordingly.

In the file of the MQL5 signal collection class \\MQL5\\Include\\DoEasy\\Collections\ **MQLSignalsCollection.mqh**, make some minor changes to the method of updating the collection list:

```
//+------------------------------------------------------------------+
//| Update the collection list of MQL5 signal objects                |
//+------------------------------------------------------------------+
void CMQLSignalsCollection::Refresh(const bool messages=true)
  {
   this.m_signals_base_total=::SignalBaseTotal();
   //--- loop through all signals in the signal database
   for(int i=0;i<this.m_signals_base_total;i++)
     {
      //--- Select a signal from the signal database by the loop index
      if(!::SignalBaseSelect(i))
         continue;
      //--- Get the current signal ID and
      //--- create a new MQL5 signal object based on it
      long id=::SignalBaseGetInteger(SIGNAL_BASE_ID);
      CMQLSignal *signal=new CMQLSignal(id);
      if(signal==NULL)
         continue;
      //--- Set the sorting flag for the list by signal ID
      this.m_list.Sort(SORT_BY_SIGNAL_MQL5_ID);
      //--- Get the index of the MQL5 signal object in the list
      int index=this.m_list.Search(signal);
      //--- If such an object exists (the index exceeds -1)
      if(index!=WRONG_VALUE)
        {
         //--- Remove the newly created object,
         delete signal;
         //--- get the pointer to such an object in the list
         signal=this.m_list.At(index);
         //--- if the pointer is received, update all signal properties
         if(signal!=NULL)
            signal.SetProperties();
         //--- move on to the next loop iteration
         continue;
        }
      //--- No such object in the collection list yet
      //--- If failed to add a new signal object to the collection list,
      //--- remove the created object and go to the next loop iteration
      if(!this.m_list.InsertSort(signal))
        {
         delete signal;
         continue;
        }
      //--- If an MQL5 signal object is successfully added to the collection
      //--- and the new object message flag is set in the parameters passed to the method,
      //--- display a message about a newly found signal
      else if(messages)
        {
         ::Print(DFUN,CMessage::Text(MSG_MQLSIG_COLLECTION_TEXT_SIGNALS_NEW),":");
         signal.PrintShort(true);
        }
     }
  }
//+------------------------------------------------------------------+
```

Here, in the new code block, the entire logic is described in the comments. In short, objects present in the collection are not skipped. Instead, the new SetProperties() method described above is called for them. This method enters the values of the appropriate parameters of a signal selected in the database to all object properties.

This concludes the improvement of the classes for working with MQL5.com Signals.

**Let's start the development of the chart object class.**

A chart and, accordingly, a chart object feature quite a lot of parameters. First, we need to create new text messages related to the chart object.

In \\MQL5\\Include\\DoEasy\ **Data.mqh**, add new message indices:

```
//--- CChartObj
   MSG_CHART_OBJ_ID,                                  // Chart ID
   MSG_CHART_OBJ_SHOW,                                // Draw price chart attributes
   MSG_CHART_OBJ_IS_OBJECT,                           // Chart object
   MSG_CHART_OBJ_BRING_TO_TOP,                        // Chart above all others
   MSG_CHART_OBJ_CONTEXT_MENU,                        // Access the context menu using the right click
   MSG_CHART_OBJ_CROSSHAIR_TOOL,                      // Access the Crosshair tool using the middle click
   MSG_CHART_OBJ_MOUSE_SCROLL,                        // Scroll the chart horizontally using the left mouse button
   MSG_CHART_OBJ_EVENT_MOUSE_WHEEL,                   // Send messages about mouse wheel events to all MQL5 programs on a chart
   MSG_CHART_OBJ_EVENT_MOUSE_MOVE,                    // Send messages about mouse button click and movement events to all MQL5 programs on a chart
   MSG_CHART_OBJ_EVENT_OBJECT_CREATE,                 // Send messages about the graphical object creation event to all MQL5 programs on a chart
   MSG_CHART_OBJ_EVENT_OBJECT_DELETE,                 // Send messages about the graphical object destruction event to all MQL5 programs on a chart
   MSG_CHART_OBJ_MODE,                                // Chart type
   MSG_CHART_OBJ_FOREGROUND,                          // Price chart in the foreground
   MSG_CHART_OBJ_SHIFT,                               // Shift of the price chart from the right border
   MSG_CHART_OBJ_AUTOSCROLL,                          // Auto scroll to the right border of the chart
   MSG_CHART_OBJ_KEYBOARD_CONTROL,                    // Manage the chart using a keyboard
   MSG_CHART_OBJ_QUICK_NAVIGATION,                    // Allow the chart to intercept Space and Enter key strokes to activate the quick navigation bar
   MSG_CHART_OBJ_SCALE,                               // Scale
   MSG_CHART_OBJ_SCALEFIX,                            // Fixed scale
   MSG_CHART_OBJ_SCALEFIX_11,                         // Scale 1:1
   MSG_CHART_OBJ_SCALE_PT_PER_BAR,                    // Scale in points per bar
   MSG_CHART_OBJ_SHOW_TICKER,                         // Display a symbol ticker in the upper left corner
   MSG_CHART_OBJ_SHOW_OHLC,                           // Display OHLC values in the upper left corner
   MSG_CHART_OBJ_SHOW_BID_LINE,                       // Display Bid value as a horizontal line on the chart
   MSG_CHART_OBJ_SHOW_ASK_LINE,                       // Display Ask value as a horizontal line on a chart
   MSG_CHART_OBJ_SHOW_LAST_LINE,                      // Display Last value as a horizontal line on a chart
   MSG_CHART_OBJ_SHOW_PERIOD_SEP,                     // Display vertical separators between adjacent periods
   MSG_CHART_OBJ_SHOW_GRID,                           // Display a grid on the chart
   MSG_CHART_OBJ_SHOW_VOLUMES,                        // Display volumes on a chart
   MSG_CHART_OBJ_SHOW_OBJECT_DESCR,                   // Display text descriptions of objects
   MSG_CHART_OBJ_VISIBLE_BARS,                        // Number of bars on a chart that are available for display
   MSG_CHART_OBJ_WINDOWS_TOTAL,                       // Total number of chart windows including indicator subwindows
   MSG_CHART_OBJ_WINDOW_IS_VISIBLE,                   // Subwindow visibility
   MSG_CHART_OBJ_WINDOW_HANDLE,                       // Chart window handle
   MSG_CHART_OBJ_WINDOW_YDISTANCE,                    // Distance in Y axis pixels between the upper frame of the indicator subwindow and the upper frame of the chart main window
   MSG_CHART_OBJ_FIRST_VISIBLE_BAR,                   // Number of the first visible bar on the chart
   MSG_CHART_OBJ_WIDTH_IN_BARS,                       // Width of the chart in bars
   MSG_CHART_OBJ_WIDTH_IN_PIXELS,                     // Width of the chart in pixels
   MSG_CHART_OBJ_HEIGHT_IN_PIXELS,                    // Height of the chart in pixels
   MSG_CHART_OBJ_COLOR_BACKGROUND,                    // Color of background of the chart
   MSG_CHART_OBJ_COLOR_FOREGROUND,                    // Color of axes, scale and OHLC line
   MSG_CHART_OBJ_COLOR_GRID,                          // Grid color
   MSG_CHART_OBJ_COLOR_VOLUME,                        // Color of volumes and position opening levels
   MSG_CHART_OBJ_COLOR_CHART_UP,                      // Color for the up bar, shadows and body borders of bull candlesticks
   MSG_CHART_OBJ_COLOR_CHART_DOWN,                    // Color of down bar, its shadow and border of body of the bullish candlestick
   MSG_CHART_OBJ_COLOR_CHART_LINE,                    // Color of the chart line and the Doji candlesticks
   MSG_CHART_OBJ_COLOR_CANDLE_BULL,                   // Color of body of a bullish candlestick
   MSG_CHART_OBJ_COLOR_CANDLE_BEAR,                   // Color of body of a bearish candlestick
   MSG_CHART_OBJ_COLOR_BID,                           // Color of the Bid price line
   MSG_CHART_OBJ_COLOR_ASK,                           // Color of the Ask price line
   MSG_CHART_OBJ_COLOR_LAST,                          // Color of the last performed deal's price line (Last)
   MSG_CHART_OBJ_COLOR_STOP_LEVEL,                    // Color of stop order levels (Stop Loss and Take Profit)
   MSG_CHART_OBJ_SHOW_TRADE_LEVELS,                   // Display trade levels on the chart (levels of open positions, Stop Loss, Take Profit and pending orders)
   MSG_CHART_OBJ_DRAG_TRADE_LEVELS,                   // Drag trading levels on a chart using a mouse
   MSG_CHART_OBJ_SHOW_DATE_SCALE,                     // Display the time scale on a chart
   MSG_CHART_OBJ_SHOW_PRICE_SCALE,                    // Display a price scale on a chart
   MSG_CHART_OBJ_SHOW_ONE_CLICK,                      // Display the quick trading panel on the chart
   MSG_CHART_OBJ_IS_MAXIMIZED,                        // Chart window maximized
   MSG_CHART_OBJ_IS_MINIMIZED,                        // Chart window minimized
   MSG_CHART_OBJ_IS_DOCKED,                           // Chart window docked
   MSG_CHART_OBJ_FLOAT_LEFT,                          // Left coordinate of the undocked chart window relative to the virtual screen
   MSG_CHART_OBJ_FLOAT_TOP,                           // Upper coordinate of the undocked chart window relative to the virtual screen
   MSG_CHART_OBJ_FLOAT_RIGHT,                         // Right coordinate of the undocked chart window relative to the virtual screen
   MSG_CHART_OBJ_FLOAT_BOTTOM,                        // Bottom coordinate of the undocked chart window relative to the virtual screen

   MSG_CHART_OBJ_SHIFT_SIZE,                          // Shift size of the zero bar from the right border in %
   MSG_CHART_OBJ_FIXED_POSITION,                      // Chart fixed position from the left border in %
   MSG_CHART_OBJ_FIXED_MAX,                           // Chart fixed maximum
   MSG_CHART_OBJ_FIXED_MIN,                           // Chart fixed minimum
   MSG_CHART_OBJ_POINTS_PER_BAR,                      // Scale in points per bar
   MSG_CHART_OBJ_PRICE_MIN,                           // Chart minimum
   MSG_CHART_OBJ_PRICE_MAX,                           // Chart maximum

   MSG_CHART_OBJ_COMMENT,                             // Chart comment text
   MSG_CHART_OBJ_EXPERT_NAME,                         // Name of an EA launched on the chart
   MSG_CHART_OBJ_SCRIPT_NAME,                         // Name of a script launched on the chart

   MSG_CHART_OBJ_CHART_BARS,                          // Display as bars
   MSG_CHART_OBJ_CHART_CANDLES,                       // Display as Japaneses candlesticks
   MSG_CHART_OBJ_CHART_LINE,                          // Display as a line drawn at Close prices
   MSG_CHART_OBJ_CHART_VOLUME_HIDE,                   // Volumes not displayed
   MSG_CHART_OBJ_CHART_VOLUME_TICK,                   // Tick volumes
   MSG_CHART_OBJ_CHART_VOLUME_REAL,                   // Trading volumes

   MSG_CHART_OBJ_CHART_WINDOW,                        // Chart window

  };
//+------------------------------------------------------------------+
```

and text messages corresponding to newly added indices:

```
//--- CChartObj
   {"Идентификатор графика","Chart ID"},
   {"Отрисовка атрибутов ценового графика","Drawing attributes of a price chart"},
   {"Объект \"График\"","Object \"Chart\""},
   {"График поверх всех других","Chart on top of other charts"},
   {"Доступ к контекстному меню по нажатию правой клавиши мыши","Accessing the context menu by pressing the right mouse button"},
   {"Доступ к инструменту \"Перекрестие\" по нажатию средней клавиши мыши","Accessing the \"Crosshair tool\" by pressing the middle mouse button"},
   {"Прокрутка графика левой кнопкой мышки по горизонтали","Scrolling the chart horizontally using the left mouse button"},
   {"Отправка всем mql5-программам на графике сообщений о событиях колёсика мыши","Sending messages about mouse wheel events to all mql5 programs on a chart"},
   {"Отправка всем mql5-программам на графике сообщений о событиях перемещения и нажатия кнопок мыши","Send notifications of mouse move and mouse click events to all mql5 programs on a chart"},
   {"Отправка всем mql5-программам на графике сообщений о событии создания графического объекта","Send a notification of an event of new object creation to all mql5-programs on a chart"},
   {"Отправка всем mql5-программам на графике сообщений о событии уничтожения графического объекта","Send a notification of an event of object deletion to all mql5-programs on a chart"},
   {"Тип графика","Chart type"},
   {"Ценовой график на переднем плане","Price chart in the foreground"},
   {"Отступ ценового графика от правого края","Price chart indent from the right border"},
   {"Автоматический переход к правому краю графика","Automatic moving to the right border of the chart"},
   {"Управление графиком с помощью клавиатуры","Managing the chart using a keyboard"},
   {"Перехват графиком нажатий клавиш Space и Enter для активации строки быстрой навигации","Allowed to intercept Space and Enter key presses on the chart to activate the quick navigation bar"},
   {"Масштаб","Scale"},
   {"Фиксированный масштаб","Fixed scale mode"},
   {"Масштаб 1:1","Scale 1:1 mode"},
   {"Масштаб в пунктах на бар","Scale to be specified in points per bar"},
   {"Отображение в левом верхнем углу тикера символа","Display a symbol ticker in the upper left corner"},
   {"Отображение в левом верхнем углу значений OHLC","Display OHLC values in the upper left corner"},
   {"Отображение значения Bid горизонтальной линией на графике","Display Bid values as a horizontal line in a chart"},
   {"Отображение значения Ask горизонтальной линией на графике","Display Ask values as a horizontal line in a chart"},
   {"Отображение значения Last горизонтальной линией на графике","Display Last values as a horizontal line in a chart"},
   {"Отображение вертикальных разделителей между соседними периодами","Display vertical separators between adjacent periods"},
   {"Отображение сетки на графике","Display grid in the chart"},
   {"Отображение объемов на графике","Display volume in the chart"},
   {"Отображение текстовых описаний объектов","Display textual descriptions of objects"},
   {"Количество баров на графике, доступных для отображения","The number of bars on the chart that can be displayed"},
   {"Общее количество окон графика с подокнами индикаторов","The total number of chart windows, including indicator subwindows"},
   {"Видимость подокон","Visibility of subwindows"},
   {"Хэндл окна графика","Chart window handle"},
   {"Дистанция в пикселях по оси Y между верхней рамкой подокна индикатора и верхней рамкой главного окна графика","The distance between the upper frame of the indicator subwindow and the upper frame of the main chart window"},
   {"Номер первого видимого бара на графике","Number of the first visible bar in the chart"},
   {"Ширина графика в барах","Chart width in bars"},
   {"Ширина графика в пикселях","Chart width in pixels"},
   {"Высота графика в пикселях","Chart height in pixels"},
   {"Цвет фона графика","Chart background color"},
   {"Цвет осей, шкалы и строки OHLC","Color of axes, scales and OHLC line"},
   {"Цвет сетки","Grid color"},
   {"Цвет объемов и уровней открытия позиций","Color of volumes and position opening levels"},
   {"Цвет бара вверх, тени и окантовки тела бычьей свечи","Color for the up bar, shadows and body borders of bull candlesticks"},
   {"Цвет бара вниз, тени и окантовки тела медвежьей свечи","Color for the down bar, shadows and body borders of bear candlesticks"},
   {"Цвет линии графика и японских свечей \"Доджи\"","Line chart color and color of \"Doji\" Japanese candlesticks"},
   {"Цвет тела бычьей свечи","Body color of a bull candlestick"},
   {"Цвет тела медвежьей свечи","Body color of a bear candlestick"},
   {"Цвет линии Bid-цены","Bid price level color"},
   {"Цвет линии Ask-цены","Ask price level color"},
   {"Цвет линии цены последней совершенной сделки (Last)","Line color of the last executed deal price (Last)"},
   {"Цвет уровней стоп-ордеров (Stop Loss и Take Profit)","Color of stop order levels (Stop Loss and Take Profit)"},
   {"Отображение на графике торговых уровней (уровни открытых позиций, Stop Loss, Take Profit и отложенных ордеров)","Displaying trade levels in the chart (levels of open positions, Stop Loss, Take Profit and pending orders)"},
   {"Перетаскивание торговых уровней на графике с помощью мышки","Permission to drag trading levels on a chart with a mouse"},
   {"Отображение на графике шкалы времени","Showing the time scale on a chart"},
   {"Отображение на графике ценовой шкалы","Showing the price scale on a chart"},
   {"Отображение на графике панели быстрой торговли","Showing the \"One click trading\" panel on a chart"},
   {"Окно графика развернуто","Chart window is maximized"},
   {"Окно графика свернуто","Chart window is minimized"},
   {"Окно графика закреплено","The chart window is docked"},
   {"Левая координата открепленного графика относительно виртуального экрана","The left coordinate of the undocked chart window relative to the virtual screen"},
   {"Верхняя координата открепленного графика относительно виртуального экрана","The top coordinate of the undocked chart window relative to the virtual screen"},
   {"Правая координата открепленного графика  относительно виртуального экрана","The right coordinate of the undocked chart window relative to the virtual screen"},
   {"Нижняя координата открепленного графика  относительно виртуального экрана","The bottom coordinate of the undocked chart window relative to the virtual screen"},

   {"Размер отступа нулевого бара от правого края в процентах","The size of the zero bar indent from the right border in percents"},
   {"Положение фиксированной позиции графика от левого края в процентах","Chart fixed position from the left border in percent value"},
   {"Фиксированный максимум графика","Fixed  chart maximum"},
   {"Фиксированный минимум графика","Fixed  chart minimum "},
   {"Масштаб в пунктах на бар","Scale in points per bar"},
   {"Минимум графика","Chart minimum"},
   {"Максимум графика","Chart maximum"},

   {"Текст комментария на графике","Text of a comment in a chart"},
   {"Имя эксперта, запущенного на графике","The name of the Expert Advisor running on the chart"},
   {"Имя скрипта, запущенного на графике","The name of the script running on the chart"},

   {"Отображение в виде баров","Display as a sequence of bars"},
   {"Отображение в виде японских свечей","Display as Japanese candlesticks"},
   {"Отображение в виде линии, проведенной по ценам Close","Display as a line drawn by Close prices"},
   {"Объемы не показаны","Volumes are not shown"},
   {"Тиковые объемы","Tick volumes"},
   {"Торговые объемы","Trade volumes"},

   {"Окно графика","Chart window"},

  };
//+---------------------------------------------------------------------+
```

See [the article 19](https://www.mql5.com/en/articles/7176) for general understanding of messages, message index constants and lists. The article describes the development of the library message class in detail.

To display a description of some properties of a created chart object, I will need two functions — for returning the description of the chart display mode (bars, candles, line) and for returning the mode of displaying volumes on the chart (not displayed, tick, real). Add the functions to the service functions file \\MQL5\\Include\\DoEasy\\Services\ **DELib.mqh**, so that they always remain at hand:

```
//+------------------------------------------------------------------+
//| Return the description of the method of displaying a price chart |
//+------------------------------------------------------------------+
string ChartModeDescription(ENUM_CHART_MODE mode)
  {
   return
     (
      mode==CHART_BARS     ? CMessage::Text(MSG_CHART_OBJ_CHART_BARS)      :
      mode==CHART_CANDLES  ? CMessage::Text(MSG_CHART_OBJ_CHART_CANDLES)   :
      CMessage::Text(MSG_CHART_OBJ_CHART_LINE)
     );
  }
//+--------------------------------------------------------------------------+
//|Return the description of the mode of displaying volumes on a price chart |
//+--------------------------------------------------------------------------+
string ChartModeVolumeDescription(ENUM_CHART_VOLUME_MODE mode)
  {
   return
     (
      mode==CHART_VOLUME_TICK ? CMessage::Text(MSG_CHART_OBJ_CHART_VOLUME_TICK)  :
      mode==CHART_VOLUME_REAL ? CMessage::Text(MSG_CHART_OBJ_CHART_VOLUME_REAL)  :
      CMessage::Text(MSG_CHART_OBJ_CHART_VOLUME_HIDE)
     );
  }
//+------------------------------------------------------------------+
```

Both functions receive the appropriate chart property. Its compliance with one of the enumeration constants is checked and the string description of the chart/volume mode is returned.

Each of the library objects features the list of properties set in \\MQL5\\Include\\DoEasy\ **Defines.mqh**. The chart object is no exception. Let's set all the necessary properties for it in three enumerations of integer, real and string properties:

```
//+------------------------------------------------------------------+
//| Data for working with charts                                     |
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| List of possible chart events                                    |
//+------------------------------------------------------------------+
#define CHART_EVENTS_NEXT_CODE  (SIGNAL_MQL5_EVENTS_NEXT_CODE+1)   // The code of the next event after the last chart event code
//+------------------------------------------------------------------+
//| Chart integer property                                           |
//+------------------------------------------------------------------+
enum ENUM_CHART_PROP_INTEGER
  {
   CHART_PROP_ID = 0,                                 // Chart ID
   CHART_PROP_TIMEFRAME,                              // Chart timeframe
   CHART_PROP_SHOW,                                   // Price chart drawing
   CHART_PROP_IS_OBJECT,                              // Chart object (OBJ_CHART) identification attribute
   CHART_PROP_BRING_TO_TOP,                           // Show chart above all others
   CHART_PROP_CONTEXT_MENU,                           // Enable/disable access to the context menu using the right click
   CHART_PROP_CROSSHAIR_TOOL,                         // Enable/disable access to the Crosshair tool using the middle click
   CHART_PROP_MOUSE_SCROLL,                           // Scroll the chart horizontally using the left mouse button
   CHART_PROP_EVENT_MOUSE_WHEEL,                      // Send messages about mouse wheel events (CHARTEVENT_MOUSE_WHEEL) to all MQL5 programs on a chart
   CHART_PROP_EVENT_MOUSE_MOVE,                       // Send messages about mouse button click and movement events (CHARTEVENT_MOUSE_MOVE) to all MQL5 programs on a chart
   CHART_PROP_EVENT_OBJECT_CREATE,                    // Send messages about the graphical object creation event (CHARTEVENT_OBJECT_CREATE) to all MQL5 programs on a chart
   CHART_PROP_EVENT_OBJECT_DELETE,                    // Send messages about the graphical object destruction event (CHARTEVENT_OBJECT_DELETE) to all MQL5 programs on a chart
   CHART_PROP_MODE,                                   // Type of the chart (candlesticks, bars or line (ENUM_CHART_MODE))
   CHART_PROP_FOREGROUND,                             // Price chart in the foreground
   CHART_PROP_SHIFT,                                  // Mode of shift of the price chart from the right border
   CHART_PROP_AUTOSCROLL,                             // The mode of automatic shift to the right border of the chart
   CHART_PROP_KEYBOARD_CONTROL,                       // Allow managing the chart using a keyboard
   CHART_PROP_QUICK_NAVIGATION,                       // Allow the chart to intercept Space and Enter key strokes to activate the quick navigation bar
   CHART_PROP_SCALE,                                  // Scale
   CHART_PROP_SCALEFIX,                               // Fixed scale mode
   CHART_PROP_SCALEFIX_11,                            // 1:1 scale mode
   CHART_PROP_SCALE_PT_PER_BAR,                       // The mode of specifying the scale in points per bar
   CHART_PROP_SHOW_TICKER,                            // Display a symbol ticker in the upper left corner
   CHART_PROP_SHOW_OHLC,                              // Display OHLC values in the upper left corner
   CHART_PROP_SHOW_BID_LINE,                          // Display Bid value as a horizontal line on the chart
   CHART_PROP_SHOW_ASK_LINE,                          // Display Ask value as a horizontal line on a chart
   CHART_PROP_SHOW_LAST_LINE,                         // Display Last value as a horizontal line on a chart
   CHART_PROP_SHOW_PERIOD_SEP,                        // Display vertical separators between adjacent periods
   CHART_PROP_SHOW_GRID,                              // Display a grid on the chart
   CHART_PROP_SHOW_VOLUMES,                           // Display volumes on a chart
   CHART_PROP_SHOW_OBJECT_DESCR,                      // Display text descriptions of objects
   CHART_PROP_VISIBLE_BARS,                           // Number of bars on a chart that are available for display
   CHART_PROP_WINDOWS_TOTAL,                          // The total number of chart windows including indicator subwindows
   //CHART_PROP_WINDOW_IS_VISIBLE,                      // Subwindow visibility
   CHART_PROP_WINDOW_HANDLE,                          // Chart window handle
   //CHART_PROP_WINDOW_YDISTANCE,                       // Distance in Y axis pixels between the upper frame of the indicator subwindow and the upper frame of the chart main window
   CHART_PROP_FIRST_VISIBLE_BAR,                      // Number of the first visible bar on the chart
   CHART_PROP_WIDTH_IN_BARS,                          // Width of the chart in bars
   CHART_PROP_WIDTH_IN_PIXELS,                        // Width of the chart in pixels
   //CHART_PROP_HEIGHT_IN_PIXELS,                       // Height of the chart in pixels
   CHART_PROP_COLOR_BACKGROUND,                       // Color of background of the chart
   CHART_PROP_COLOR_FOREGROUND,                       // Color of axes, scale and OHLC line
   CHART_PROP_COLOR_GRID,                             // Grid color
   CHART_PROP_COLOR_VOLUME,                           // Color of volumes and position opening levels
   CHART_PROP_COLOR_CHART_UP,                         // Color for the up bar, shadows and body borders of bull candlesticks
   CHART_PROP_COLOR_CHART_DOWN,                       // Color of down bar, its shadow and border of body of the bullish candlestick
   CHART_PROP_COLOR_CHART_LINE,                       // Color of the chart line and the Doji candlesticks
   CHART_PROP_COLOR_CANDLE_BULL,                      // Color of body of a bullish candlestick
   CHART_PROP_COLOR_CANDLE_BEAR,                      // Color of body of a bearish candlestick
   CHART_PROP_COLOR_BID,                              // Color of the Bid price line
   CHART_PROP_COLOR_ASK,                              // Color of the Ask price line
   CHART_PROP_COLOR_LAST,                             // Color of the last performed deal's price line (Last)
   CHART_PROP_COLOR_STOP_LEVEL,                       // Color of stop order levels (Stop Loss and Take Profit)
   CHART_PROP_SHOW_TRADE_LEVELS,                      // Display trade levels on the chart (levels of open positions, Stop Loss, Take Profit and pending orders)
   CHART_PROP_DRAG_TRADE_LEVELS,                      // Enable the ability to drag trading levels on a chart using mouse
   CHART_PROP_SHOW_DATE_SCALE,                        // Display the time scale on a chart
   CHART_PROP_SHOW_PRICE_SCALE,                       // Display a price scale on a chart
   CHART_PROP_SHOW_ONE_CLICK,                         // Display the quick trading panel on the chart
   CHART_PROP_IS_MAXIMIZED,                           // Chart window maximized
   CHART_PROP_IS_MINIMIZED,                           // Chart window minimized
   CHART_PROP_IS_DOCKED,                              // Chart window docked
   CHART_PROP_FLOAT_LEFT,                             // Left coordinate of the undocked chart window relative to the virtual screen
   CHART_PROP_FLOAT_TOP,                              // Upper coordinate of the undocked chart window relative to the virtual screen
   CHART_PROP_FLOAT_RIGHT,                            // Right coordinate of the undocked chart window relative to the virtual screen
   CHART_PROP_FLOAT_BOTTOM,                           // Bottom coordinate of the undocked chart window relative to the virtual screen
  };
#define CHART_PROP_INTEGER_TOTAL (62)                 // Total number of integer properties
#define CHART_PROP_INTEGER_SKIP  (0)                  // Number of integer DOM properties not used in sorting
//+------------------------------------------------------------------+
//| Chart real properties                                            |
//+------------------------------------------------------------------+
enum ENUM_CHART_PROP_DOUBLE
  {
   CHART_PROP_SHIFT_SIZE = CHART_PROP_INTEGER_TOTAL,  // Shift size of the zero bar from the right border in %
   CHART_PROP_FIXED_POSITION,                         // Chart fixed position from the left border in %
   CHART_PROP_FIXED_MAX,                              // Chart fixed maximum
   CHART_PROP_FIXED_MIN,                              // Chart fixed minimum
   CHART_PROP_POINTS_PER_BAR,                         // Scale in points per bar
   CHART_PROP_PRICE_MIN,                              // Chart minimum
   CHART_PROP_PRICE_MAX,                              // Chart maximum
  };
#define CHART_PROP_DOUBLE_TOTAL  (7)                  // Total number of real properties
#define CHART_PROP_DOUBLE_SKIP   (0)                  // Number of real properties not used in sorting
//+------------------------------------------------------------------+
//| Chart string properties                                          |
//+------------------------------------------------------------------+
enum ENUM_CHART_PROP_STRING
  {
   CHART_PROP_COMMENT = (CHART_PROP_INTEGER_TOTAL+CHART_PROP_DOUBLE_TOTAL), // Chart comment text
   CHART_PROP_EXPERT_NAME,                            // Name of an EA launched on the chart
   CHART_PROP_SCRIPT_NAME,                            // Name of a script launched on the chart
   CHART_PROP_SYMBOL,                                 // Chart symbol
  };
#define CHART_PROP_STRING_TOTAL  (4)                  // Total number of string properties
//+------------------------------------------------------------------+
```

Three properties in the enumeration of integer object properties are not needed yet, so they are commented out. They have been added here because the chart features them but they are related not only to the chart object (chart main window) but also to the main window subwindows, so they cannot belong exclusively to a single chart object. These properties will be activated later.

As usual, after adding new object property enumerations, we need to add the enumeration of possible object sorting criteria:

```
//+------------------------------------------------------------------+
//| Possible chart sorting criteria                                  |
//+------------------------------------------------------------------+
#define FIRST_CHART_DBL_PROP  (CHART_PROP_INTEGER_TOTAL-CHART_PROP_INTEGER_SKIP)
#define FIRST_CHART_STR_PROP  (CHART_PROP_INTEGER_TOTAL-CHART_PROP_INTEGER_SKIP+CHART_PROP_DOUBLE_TOTAL-CHART_PROP_DOUBLE_SKIP)
enum ENUM_SORT_CHART_MODE
  {
//--- Sort by integer properties
   SORT_BY_CHART_SHOW = 0,                            // Sort by the price chart drawing attribute
   SORT_BY_CHART_IS_OBJECT,                           // Sort by chart object (OBJ_CHART) identification attribute
   SORT_BY_CHART_BRING_TO_TOP,                        // Sort by the flag of displaying a chart above all others
   SORT_BY_CHART_CONTEXT_MENU,                        // Sort by the flag of enabling/disabling access to the context menu using the right click
   SORT_BY_CHART_CROSSHAIR_TOO,                       // Sort by the flag of enabling/disabling access to the Crosshair tool using the middle click
   SORT_BY_CHART_MOUSE_SCROLL,                        // Sort by the flag of scrolling the chart horizontally using the left mouse button
   SORT_BY_CHART_EVENT_MOUSE_WHEEL,                   // Sort by the flag of sending messages about mouse wheel events to all MQL5 programs on a chart
   SORT_BY_CHART_EVENT_MOUSE_MOVE,                    // Sort by the flag of sending messages about mouse button click and movement events to all MQL5 programs on a chart
   SORT_BY_CHART_EVENT_OBJECT_CREATE,                 // Sort by the flag of sending messages about the graphical object creation event to all MQL5 programs on a chart
   SORT_BY_CHART_EVENT_OBJECT_DELETE,                 // Sort by the flag of sending messages about the graphical object destruction event to all MQL5 programs on a chart
   SORT_BY_CHART_MODE,                                // Sort by chart type
   SORT_BY_CHART_FOREGROUND,                          // Sort by the "Price chart in the foreground" flag
   SORT_BY_CHART_SHIFT,                               // Sort by the "Mode of shift of the price chart from the right border" flag
   SORT_BY_CHART_AUTOSCROLL,                          // Sort by the "The mode of automatic shift to the right border of the chart" flag
   SORT_BY_CHART_KEYBOARD_CONTROL,                    // Sort by the flag allowing the chart management using a keyboard
   SORT_BY_CHART_QUICK_NAVIGATION,                    // Sort by the flag allowing the chart to intercept Space and Enter key strokes to activate the quick navigation bar
   SORT_BY_CHART_SCALE,                               // Sort by scale
   SORT_BY_CHART_SCALEFIX,                            // Sort by the fixed scale flag
   SORT_BY_CHART_SCALEFIX_11,                         // Sort by the 1:1 scale flag
   SORT_BY_CHART_SCALE_PT_PER_BAR,                    // Sort by the flag of specifying the scale in points per bar
   SORT_BY_CHART_SHOW_TICKER,                         // Sort by the flag displaying a symbol ticker in the upper left corner
   SORT_BY_CHART_SHOW_OHLC,                           // Sort by the flag displaying OHLC values in the upper left corner
   SORT_BY_CHART_SHOW_BID_LINE,                       // Sort by the flag displaying Bid value as a horizontal line on the chart
   SORT_BY_CHART_SHOW_ASK_LINE,                       // Sort by the flag displaying Ask value as a horizontal line on the chart
   SORT_BY_CHART_SHOW_LAST_LINE,                      // Sort by the flag displaying Last value as a horizontal line on the chart
   SORT_BY_CHART_SHOW_PERIOD_SEP,                     // Sort by the flag displaying vertical separators between adjacent periods
   SORT_BY_CHART_SHOW_GRID,                           // Sort by the flag of displaying a grid on the chart
   SORT_BY_CHART_SHOW_VOLUMES,                        // Sort by the mode of displaying volumes on a chart
   SORT_BY_CHART_SHOW_OBJECT_DESCR,                   // Sort by the flag of displaying object text descriptions
   SORT_BY_CHART_VISIBLE_BARS,                        // Sort by the number of bars on a chart that are available for display
   SORT_BY_CHART_WINDOWS_TOTAL,                       // Sort by the total number of chart windows including indicator subwindows
   SORT_BY_CHART_WINDOW_IS_VISIBLE,                   // Sort by the subwindow visibility flag
   SORT_BY_CHART_WINDOW_HANDLE,                       // Sort by the chart handle
   SORT_BY_CHART_WINDOW_YDISTANCE,                    // Sort by the distance in Y axis pixels between the upper frame of the indicator subwindow and the upper frame of the chart main window
   SORT_BY_CHART_FIRST_VISIBLE_BAR,                   // Sort by the number of the first visible bar on the chart
   SORT_BY_CHART_WIDTH_IN_BARS,                       // Sort by the width of the chart in bars
   SORT_BY_CHART_WIDTH_IN_PIXELS,                     // Sort by the width of the chart in pixels
   SORT_BY_CHART_HEIGHT_IN_PIXELS,                    // Sort by the height of the chart in pixels
   SORT_BY_CHART_COLOR_BACKGROUND,                    // Sort by the color of the chart background
   SORT_BY_CHART_COLOR_FOREGROUND,                    // Sort by color of axes, scale and OHLC line
   SORT_BY_CHART_COLOR_GRID,                          // Sort by grid color
   SORT_BY_CHART_COLOR_VOLUME,                        // Sort by the color of volumes and position opening levels
   SORT_BY_CHART_COLOR_CHART_UP,                      // Sort by the color for the up bar, shadows and body borders of bull candlesticks
   SORT_BY_CHART_COLOR_CHART_DOWN,                    // Sort by the color of down bar, its shadow and border of body of the bullish candlestick
   SORT_BY_CHART_COLOR_CHART_LINE,                    // Sort by the color of the chart line and the Doji candlesticks
   SORT_BY_CHART_COLOR_CANDLE_BULL,                   // Sort by the color of a bullish candlestick body
   SORT_BY_CHART_COLOR_CANDLE_BEAR,                   // Sort by the color of a bearish candlestick body
   SORT_BY_CHART_COLOR_BID,                           // Sort by the color of the Bid price line
   SORT_BY_CHART_COLOR_ASK,                           // Sort by the color of the Ask price line
   SORT_BY_CHART_COLOR_LAST,                          // Sort by the color of the last performed deal's price line (Last)
   SORT_BY_CHART_COLOR_STOP_LEVEL,                    // Sort by the color of stop order levels (Stop Loss and Take Profit)
   SORT_BY_CHART_SHOW_TRADE_LEVELS,                   // Sort by the flag of displaying trading levels on the chart
   SORT_BY_CHART_DRAG_TRADE_LEVELS,                   // Sort by the flag enabling the ability to drag trading levels on a chart using mouse
   SORT_BY_CHART_SHOW_DATE_SCALE,                     // Sort by the flag of displaying the time scale on the chart
   SORT_BY_CHART_SHOW_PRICE_SCALE,                    // Sort by the flag of displaying the price scale on the chart
   SORT_BY_CHART_SHOW_ONE_CLICK,                      // Sort by the flag of displaying the quick trading panel on the chart
   SORT_BY_CHART_IS_MAXIMIZED,                        // Sort by the "Chart window maximized" flag
   SORT_BY_CHART_IS_MINIMIZED,                        // Sort by the "Chart window minimized" flag
   SORT_BY_CHART_IS_DOCKED,                           // Sort by the "Chart window docked" flag
   SORT_BY_CHART_FLOAT_LEFT,                          // Sort by the left coordinate of the undocked chart window relative to the virtual screen
   SORT_BY_CHART_FLOAT_TOP,                           // Sort by the upper coordinate of the undocked chart window relative to the virtual screen
   SORT_BY_CHART_FLOAT_RIGHT,                         // Sort by the right coordinate of the undocked chart window relative to the virtual screen
   SORT_BY_CHART_FLOAT_BOTTOM,                        // Sort by the bottom coordinate of the undocked chart window relative to the virtual screen
//--- Sort by real properties
   SORT_BY_CHART_SHIFT_SIZE = FIRST_CHART_DBL_PROP,   // Sort by the shift size of the zero bar from the right border in %
   SORT_BY_CHART_FIXED_POSITION,                      // Sort by the chart fixed position from the left border in %
   SORT_BY_CHART_FIXED_MAX,                           // Sort by the fixed chart maximum
   SORT_BY_CHART_FIXED_MIN,                           // Sort by the fixed chart minimum
   SORT_BY_CHART_POINTS_PER_BAR,                      // Sort by the scale value in points per bar
   SORT_BY_CHART_PRICE_MIN,                           // Sort by the chart minimum
   SORT_BY_CHART_PRICE_MAX,                           // Sort by the chart maximum
//--- Sort by string properties
   SORT_BY_CHART_COMMENT = FIRST_CHART_STR_PROP,      // Sort by a comment text on the chart
   SORT_BY_CHART_EXPERT_NAME,                         // Sort by a name of an EA launched on the chart
   SORT_BY_CHART_SCRIPT_NAME,                         // Sort by a name of a script launched on the chart
  };
//+------------------------------------------------------------------+
```

Learn more about the library objects [in the first](https://www.mql5.com/en/articles/5654) and several subsequent articles.

Now everything is ready for creating the chart object class.

### Chart object class

In the library directory's new folder \\MQL5\\Include\\DoEasy\ **Objects\Chart\**, create the new file **ChartObj.mqh** of the **CChartObj** class.

The base object of all library objects should be a base object. Its file should be connected to the class file:

```
//+------------------------------------------------------------------+
//|                                                     ChartObj.mqh |
//|                                  Copyright 2021, MetaQuotes Ltd. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2021, MetaQuotes Ltd."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
#property strict    // Necessary for mql4
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include "..\..\Objects\BaseObj.mqh"
//+------------------------------------------------------------------+
//| Chart object class                                               |
//+------------------------------------------------------------------+
class CChartObj : public CBaseObj
  {
  }
```

In the private class section, add the standard arrays for storing the library properties, the methods returning the actual property index in the array and the class member variable for storing the Digits() value of a symbol.

```
//+------------------------------------------------------------------+
//| Chart object class                                               |
//+------------------------------------------------------------------+
class CChartObj : public CBaseObj
  {
private:
   long              m_long_prop[CHART_PROP_INTEGER_TOTAL];       // Integer properties
   double            m_double_prop[CHART_PROP_DOUBLE_TOTAL];      // Real properties
   string            m_string_prop[CHART_PROP_STRING_TOTAL];      // String properties
   int               m_digits;                                    // Symbol's Digits()

//--- Return the index of the array the (1) double and (2) string properties are actually located at
   int               IndexProp(ENUM_CHART_PROP_DOUBLE property)   const { return(int)property-CHART_PROP_INTEGER_TOTAL;                         }
   int               IndexProp(ENUM_CHART_PROP_STRING property)   const { return(int)property-CHART_PROP_INTEGER_TOTAL-CHART_PROP_DOUBLE_TOTAL; }
```

In the public class section, write all methods that are standard for the library objects: the methods of installing and returning the object properties, the virtual methods returning the flags of supporting a property by the object (although I will not implement descendant objects, the methods should be virtual so that they can be changed in the descendant classes if needed), the methods returning the property descriptions, the methods of displaying in the journal and describing the object properties and names, the comparison methods and constructors.

```
public:
//--- Set object's (1) integer, (2) real and (3) string properties
   void              SetProperty(ENUM_CHART_PROP_INTEGER property,long value)    { this.m_long_prop[property]=value;                      }
   void              SetProperty(ENUM_CHART_PROP_DOUBLE property,double value)   { this.m_double_prop[this.IndexProp(property)]=value;    }
   void              SetProperty(ENUM_CHART_PROP_STRING property,string value)   { this.m_string_prop[this.IndexProp(property)]=value;    }
//--- Return object’s (1) integer, (2) real and (3) string property from the properties array
   long              GetProperty(ENUM_CHART_PROP_INTEGER property)         const { return this.m_long_prop[property];                     }
   double            GetProperty(ENUM_CHART_PROP_DOUBLE property)          const { return this.m_double_prop[this.IndexProp(property)];   }
   string            GetProperty(ENUM_CHART_PROP_STRING property)          const { return this.m_string_prop[this.IndexProp(property)];   }
//--- Return itself
   CChartObj        *GetObject(void)                                             { return &this;}

//--- Return the flag of the object supporting this property
   virtual bool      SupportProperty(ENUM_CHART_PROP_INTEGER property)           { return true; }
   virtual bool      SupportProperty(ENUM_CHART_PROP_DOUBLE property)            { return true; }
   virtual bool      SupportProperty(ENUM_CHART_PROP_STRING property)            { return true; }

//--- Get description of (1) integer, (2) real and (3) string properties
   string            GetPropertyDescription(ENUM_CHART_PROP_INTEGER property);
   string            GetPropertyDescription(ENUM_CHART_PROP_DOUBLE property);
   string            GetPropertyDescription(ENUM_CHART_PROP_STRING property);

//--- Display the description of object properties in the journal (full_prop=true - all properties, false - supported ones only)
   void              Print(const bool full_prop=false);
//--- Display a short description of the object in the journal
   virtual void      PrintShort(const bool dash=false);
//--- Return the object short name
   virtual string    Header(void);

//--- Compare CChartObj objects by a specified property (to sort the list by a specified chart object property)
   virtual int       Compare(const CObject *node,const int mode=0) const;
//--- Compare CChartObj objects by all properties (to search for equal chart objects)
   bool              IsEqual(CChartObj* compared_obj) const;

//--- Constructors
                     CChartObj(){;}
                     CChartObj(const long chart_id);
```

This is a standard set of library object methods. They were discussed and described many times already, so I will not dwell on them.

When creating a chart object, it is assumed that the appropriate chart is selected and its properties can be obtained using the [ChartGetInteger()](https://www.mql5.com/en/docs/chart_operations/chartgetinteger), [ChartGetDouble()](https://www.mql5.com/en/docs/chart_operations/chartgetdouble) and [ChartGetString()](https://www.mql5.com/en/docs/chart_operations/chartgetstring) functions. Therefore, in the class constructor, all chart object properties are simply filled with the values returned by the appropriate function properties and the symbol's Digits() value is set in the appropriate variable:

```
//+------------------------------------------------------------------+
//| Parametric constructor                                           |
//+------------------------------------------------------------------+
CChartObj::CChartObj(const long chart_id)
  {
//--- Set integer properties
   this.SetProperty(CHART_PROP_ID,chart_id);                                                             // Chart ID
   this.SetProperty(CHART_PROP_TIMEFRAME,::ChartPeriod(this.ID()));                                      // Chart timeframe
   this.SetProperty(CHART_PROP_SHOW,::ChartGetInteger(this.ID(),CHART_SHOW));                            // Price chart drawing attribute
   this.SetProperty(CHART_PROP_IS_OBJECT,::ChartGetInteger(this.ID(),CHART_IS_OBJECT));                  // Chart object identification attribute
   this.SetProperty(CHART_PROP_BRING_TO_TOP,false);                                                      // Show chart above all others
   this.SetProperty(CHART_PROP_CONTEXT_MENU,::ChartGetInteger(this.ID(),CHART_CONTEXT_MENU));            // Access to the context menu using the right click
   this.SetProperty(CHART_PROP_CROSSHAIR_TOOL,::ChartGetInteger(this.ID(),CHART_CROSSHAIR_TOOL));        // Access the Crosshair tool by pressing the middle mouse button
   this.SetProperty(CHART_PROP_MOUSE_SCROLL,::ChartGetInteger(this.ID(),CHART_MOUSE_SCROLL));            // Scroll the chart horizontally using the left mouse button
   this.SetProperty(CHART_PROP_EVENT_MOUSE_WHEEL,::ChartGetInteger(this.ID(),CHART_EVENT_MOUSE_WHEEL));  // Send messages about mouse wheel events to all MQL5 programs on a chart
   this.SetProperty(CHART_PROP_EVENT_MOUSE_MOVE,::ChartGetInteger(this.ID(),CHART_EVENT_MOUSE_MOVE));    // Send messages about mouse button click and movement events to all MQL5 programs on a chart
   this.SetProperty(CHART_PROP_EVENT_OBJECT_CREATE,::ChartGetInteger(this.ID(),CHART_EVENT_OBJECT_CREATE)); // Send messages about the graphical object creation event to all MQL5 programs on a chart
   this.SetProperty(CHART_PROP_EVENT_OBJECT_DELETE,::ChartGetInteger(this.ID(),CHART_EVENT_OBJECT_DELETE)); // Send messages about the graphical object destruction event to all MQL5 programs on a chart
   this.SetProperty(CHART_PROP_MODE,::ChartGetInteger(this.ID(),CHART_MODE));                            // Type of the chart (candlesticks, bars or line)
   this.SetProperty(CHART_PROP_FOREGROUND,::ChartGetInteger(this.ID(),CHART_FOREGROUND));                // Price chart in the foreground
   this.SetProperty(CHART_PROP_SHIFT,::ChartGetInteger(this.ID(),CHART_SHIFT));                          // Mode of shift of the price chart from the right border
   this.SetProperty(CHART_PROP_AUTOSCROLL,::ChartGetInteger(this.ID(),CHART_AUTOSCROLL));                // The mode of automatic shift to the right border of the chart
   this.SetProperty(CHART_PROP_KEYBOARD_CONTROL,::ChartGetInteger(this.ID(),CHART_KEYBOARD_CONTROL));    // Allow managing the chart using a keyboard
   this.SetProperty(CHART_PROP_QUICK_NAVIGATION,::ChartGetInteger(this.ID(),CHART_QUICK_NAVIGATION));    // Allow the chart to intercept Space and Enter key strokes to activate the quick navigation bar
   this.SetProperty(CHART_PROP_SCALE,::ChartGetInteger(this.ID(),CHART_SCALE));                          // Scale
   this.SetProperty(CHART_PROP_SCALEFIX,::ChartGetInteger(this.ID(),CHART_SCALEFIX));                    // Fixed scale mode
   this.SetProperty(CHART_PROP_SCALEFIX_11,::ChartGetInteger(this.ID(),CHART_SCALEFIX_11));              // 1:1 scale mode
   this.SetProperty(CHART_PROP_SCALE_PT_PER_BAR,::ChartGetInteger(this.ID(),CHART_SCALE_PT_PER_BAR));    // Mode for specifying the scale in points per bar
   this.SetProperty(CHART_PROP_SHOW_TICKER,::ChartGetInteger(this.ID(),CHART_SHOW_TICKER));              // Display a symbol ticker in the upper left corner
   this.SetProperty(CHART_PROP_SHOW_OHLC,::ChartGetInteger(this.ID(),CHART_SHOW_OHLC));                  // Display OHLC values in the upper left corner
   this.SetProperty(CHART_PROP_SHOW_BID_LINE,::ChartGetInteger(this.ID(),CHART_SHOW_BID_LINE));          // Display Bid value as a horizontal line on the chart
   this.SetProperty(CHART_PROP_SHOW_ASK_LINE,::ChartGetInteger(this.ID(),CHART_SHOW_ASK_LINE));          // Display Ask value as a horizontal line on the chart
   this.SetProperty(CHART_PROP_SHOW_LAST_LINE,::ChartGetInteger(this.ID(),CHART_SHOW_LAST_LINE));        // Display Last value as a horizontal line on the chart
   this.SetProperty(CHART_PROP_SHOW_PERIOD_SEP,::ChartGetInteger(this.ID(),CHART_SHOW_PERIOD_SEP));      // Display vertical separators between adjacent periods
   this.SetProperty(CHART_PROP_SHOW_GRID,::ChartGetInteger(this.ID(),CHART_SHOW_GRID));                  // Display the chart grid
   this.SetProperty(CHART_PROP_SHOW_VOLUMES,::ChartGetInteger(this.ID(),CHART_SHOW_VOLUMES));            // Display volumes on the chart
   this.SetProperty(CHART_PROP_SHOW_OBJECT_DESCR,::ChartGetInteger(this.ID(),CHART_SHOW_OBJECT_DESCR));  // Display text descriptions of the objects
   this.SetProperty(CHART_PROP_VISIBLE_BARS,::ChartGetInteger(this.ID(),CHART_VISIBLE_BARS));            // Number of bars on a chart that are available for display
   this.SetProperty(CHART_PROP_WINDOWS_TOTAL,::ChartGetInteger(this.ID(),CHART_WINDOWS_TOTAL));          // The total number of chart windows including indicator subwindows
   this.SetProperty(CHART_PROP_WINDOW_HANDLE,::ChartGetInteger(this.ID(),CHART_WINDOW_HANDLE));          // Chart window handle
   this.SetProperty(CHART_PROP_FIRST_VISIBLE_BAR,::ChartGetInteger(this.ID(),CHART_FIRST_VISIBLE_BAR));  // Number of the first visible bar on the chart
   this.SetProperty(CHART_PROP_WIDTH_IN_BARS,::ChartGetInteger(this.ID(),CHART_WIDTH_IN_BARS));          // Chart width in bars
   this.SetProperty(CHART_PROP_WIDTH_IN_PIXELS,::ChartGetInteger(this.ID(),CHART_WIDTH_IN_PIXELS));      // Chart width in pixels
   this.SetProperty(CHART_PROP_COLOR_BACKGROUND,::ChartGetInteger(this.ID(),CHART_COLOR_BACKGROUND));    // Chart background color
   this.SetProperty(CHART_PROP_COLOR_FOREGROUND,::ChartGetInteger(this.ID(),CHART_COLOR_FOREGROUND));    // Color of axes, scale and OHLC line
   this.SetProperty(CHART_PROP_COLOR_GRID,::ChartGetInteger(this.ID(),CHART_COLOR_GRID));                // Grid color
   this.SetProperty(CHART_PROP_COLOR_VOLUME,::ChartGetInteger(this.ID(),CHART_COLOR_VOLUME));            // Color of volumes and position opening levels
   this.SetProperty(CHART_PROP_COLOR_CHART_UP,::ChartGetInteger(this.ID(),CHART_COLOR_CHART_UP));        // Color for the up bar, shadows and body borders of bullish candlesticks
   this.SetProperty(CHART_PROP_COLOR_CHART_DOWN,::ChartGetInteger(this.ID(),CHART_COLOR_CHART_DOWN));    // Color for the down bar, shadows and body borders of bearish candlesticks
   this.SetProperty(CHART_PROP_COLOR_CHART_LINE,::ChartGetInteger(this.ID(),CHART_COLOR_CHART_LINE));    // Color of the chart line and the Doji candlesticks
   this.SetProperty(CHART_PROP_COLOR_CANDLE_BULL,::ChartGetInteger(this.ID(),CHART_COLOR_CANDLE_BULL));  // Color of the bullish candle body
   this.SetProperty(CHART_PROP_COLOR_CANDLE_BEAR,::ChartGetInteger(this.ID(),CHART_COLOR_CANDLE_BEAR));  // Color of the bearish candle body
   this.SetProperty(CHART_PROP_COLOR_BID,::ChartGetInteger(this.ID(),CHART_COLOR_BID));                  // Bid price line color
   this.SetProperty(CHART_PROP_COLOR_ASK,::ChartGetInteger(this.ID(),CHART_COLOR_ASK));                  // Ask price line color
   this.SetProperty(CHART_PROP_COLOR_LAST,::ChartGetInteger(this.ID(),CHART_COLOR_LAST));                // Color of the last performed deal's price line (Last)
   this.SetProperty(CHART_PROP_COLOR_STOP_LEVEL,::ChartGetInteger(this.ID(),CHART_COLOR_STOP_LEVEL));    // Color of stop order levels (Stop Loss and Take Profit)
   this.SetProperty(CHART_PROP_SHOW_TRADE_LEVELS,::ChartGetInteger(this.ID(),CHART_SHOW_TRADE_LEVELS));  // Display trade levels on the chart (levels of open positions, Stop Loss, Take Profit and pending orders)
   this.SetProperty(CHART_PROP_DRAG_TRADE_LEVELS,::ChartGetInteger(this.ID(),CHART_DRAG_TRADE_LEVELS));  // Enable the ability to drag trading levels on a chart using mouse
   this.SetProperty(CHART_PROP_SHOW_DATE_SCALE,::ChartGetInteger(this.ID(),CHART_SHOW_DATE_SCALE));      // Display the time scale on the chart
   this.SetProperty(CHART_PROP_SHOW_PRICE_SCALE,::ChartGetInteger(this.ID(),CHART_SHOW_PRICE_SCALE));    // Display the price scale on the chart
   this.SetProperty(CHART_PROP_SHOW_ONE_CLICK,::ChartGetInteger(this.ID(),CHART_SHOW_ONE_CLICK));        // Display the quick trading panel on the chart
   this.SetProperty(CHART_PROP_IS_MAXIMIZED,::ChartGetInteger(this.ID(),CHART_IS_MAXIMIZED));            // Chart window maximized
   this.SetProperty(CHART_PROP_IS_MINIMIZED,::ChartGetInteger(this.ID(),CHART_IS_MINIMIZED));            // Chart window minimized
   this.SetProperty(CHART_PROP_IS_DOCKED,::ChartGetInteger(this.ID(),CHART_IS_DOCKED));                  // Chart window docked
   this.SetProperty(CHART_PROP_FLOAT_LEFT,::ChartGetInteger(this.ID(),CHART_FLOAT_LEFT));                // Left coordinate of the undocked chart window relative to the virtual screen
   this.SetProperty(CHART_PROP_FLOAT_TOP,::ChartGetInteger(this.ID(),CHART_FLOAT_TOP));                  // Upper coordinate of the undocked chart window relative to the virtual screen
   this.SetProperty(CHART_PROP_FLOAT_RIGHT,::ChartGetInteger(this.ID(),CHART_FLOAT_RIGHT));              // Right coordinate of the undocked chart window relative to the virtual screen
   this.SetProperty(CHART_PROP_FLOAT_BOTTOM,::ChartGetInteger(this.ID(),CHART_FLOAT_BOTTOM));            // Bottom coordinate of the undocked chart window relative to the virtual screen
//--- Set real properties
   this.SetProperty(CHART_PROP_SHIFT_SIZE,::ChartGetDouble(this.ID(),CHART_SHIFT_SIZE));                 // Shift size of the zero bar from the right border in %
   this.SetProperty(CHART_PROP_FIXED_POSITION,::ChartGetDouble(this.ID(),CHART_FIXED_POSITION));         // Chart fixed position from the left border in %
   this.SetProperty(CHART_PROP_FIXED_MAX,::ChartGetDouble(this.ID(),CHART_FIXED_MAX));                   // Fixed chart maximum
   this.SetProperty(CHART_PROP_FIXED_MIN,::ChartGetDouble(this.ID(),CHART_FIXED_MIN));                   // Fixed chart minimum
   this.SetProperty(CHART_PROP_POINTS_PER_BAR,::ChartGetDouble(this.ID(),CHART_POINTS_PER_BAR));         // Scale in points per bar
   this.SetProperty(CHART_PROP_PRICE_MIN,::ChartGetDouble(this.ID(),CHART_PRICE_MIN));                   // Chart minimum
   this.SetProperty(CHART_PROP_PRICE_MAX,::ChartGetDouble(this.ID(),CHART_PRICE_MAX));                   // Chart maximum
//--- Set string properties
   this.SetProperty(CHART_PROP_COMMENT,::ChartGetString(this.ID(),CHART_COMMENT));                       // Comment text on the chart
   this.SetProperty(CHART_PROP_EXPERT_NAME,::ChartGetString(this.ID(),CHART_EXPERT_NAME));               // name of an EA launched on the chart
   this.SetProperty(CHART_PROP_SCRIPT_NAME,::ChartGetString(this.ID(),CHART_SCRIPT_NAME));               // name of a script launched on the chart
   this.SetProperty(CHART_PROP_SYMBOL,::ChartSymbol(this.ID()));                                         // Chart symbol

   this.m_digits=(int)::SymbolInfoInteger(this.Symbol(),SYMBOL_DIGITS);
  }
//+------------------------------------------------------------------+
```

Symbol's [Digits()](https://www.mql5.com/en/docs/check/digits) is required for the correct display of some chart property values.

**The method comparing the CChartObj objects by a specified property:**

```
//+------------------------------------------------------------------+
//| Compare the CChartObj objects by a specified property            |
//+------------------------------------------------------------------+
int CChartObj::Compare(const CObject *node,const int mode=0) const
  {
   const CChartObj *obj_compared=node;
//--- compare integer properties of two objects
   if(mode<CHART_PROP_INTEGER_TOTAL)
     {
      long value_compared=obj_compared.GetProperty((ENUM_CHART_PROP_INTEGER)mode);
      long value_current=this.GetProperty((ENUM_CHART_PROP_INTEGER)mode);
      return(value_current>value_compared ? 1 : value_current<value_compared ? -1 : 0);
     }
//--- compare real properties of two objects
   else if(mode<CHART_PROP_DOUBLE_TOTAL+CHART_PROP_INTEGER_TOTAL)
     {
      double value_compared=obj_compared.GetProperty((ENUM_CHART_PROP_DOUBLE)mode);
      double value_current=this.GetProperty((ENUM_CHART_PROP_DOUBLE)mode);
      return(value_current>value_compared ? 1 : value_current<value_compared ? -1 : 0);
     }
//--- compare string properties of two objects
   else if(mode<CHART_PROP_DOUBLE_TOTAL+CHART_PROP_INTEGER_TOTAL+CHART_PROP_STRING_TOTAL)
     {
      string value_compared=obj_compared.GetProperty((ENUM_CHART_PROP_STRING)mode);
      string value_current=this.GetProperty((ENUM_CHART_PROP_STRING)mode);
      return(value_current>value_compared ? 1 : value_current<value_compared ? -1 : 0);
     }
   return 0;
  }
//+------------------------------------------------------------------+
```

The passed property is checked for its belonging to integer, real or string properties. The property of the current object is compared to the property of the one passed to the method for comparison in the appropriate code block. The result of a comparison that is greater than (1), less than (-1), or equal to (0) is returned.

**The method comparing the CChartObj objects by all properties:**

```
//+------------------------------------------------------------------+
//| Compare the CChartObj objects by all properties                  |
//+------------------------------------------------------------------+
bool CChartObj::IsEqual(CChartObj *compared_obj) const
  {
   int beg=0, end=CHART_PROP_INTEGER_TOTAL;
   for(int i=beg; i<end; i++)
     {
      ENUM_CHART_PROP_INTEGER prop=(ENUM_CHART_PROP_INTEGER)i;
      if(this.GetProperty(prop)!=compared_obj.GetProperty(prop)) return false;
     }
   beg=end; end+=CHART_PROP_DOUBLE_TOTAL;
   for(int i=beg; i<end; i++)
     {
      ENUM_CHART_PROP_DOUBLE prop=(ENUM_CHART_PROP_DOUBLE)i;
      if(this.GetProperty(prop)!=compared_obj.GetProperty(prop)) return false;
     }
   beg=end; end+=CHART_PROP_STRING_TOTAL;
   for(int i=beg; i<end; i++)
     {
      ENUM_CHART_PROP_STRING prop=(ENUM_CHART_PROP_STRING)i;
      if(this.GetProperty(prop)!=compared_obj.GetProperty(prop)) return false;
     }
   return true;
  }
//+------------------------------------------------------------------+
```

All subsequent properties of two objects (the current one and the one passed to the method) are compared in three loops by all object properties. If at least one pair of the same properties of two compared objects is not identical, false is returned — the objects are not identical. Upon completion of all loops by all properties, true is returned — no different properties, meaning the objects are identical.

**The methods returning the descriptions of integer, real and string object properties:**

```
//+------------------------------------------------------------------+
//| Return description of object's integer property                  |
//+------------------------------------------------------------------+
string CChartObj::GetPropertyDescription(ENUM_CHART_PROP_INTEGER property)
  {
   return
     (
      property==CHART_PROP_ID     ?  CMessage::Text(MSG_CHART_OBJ_ID)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.GetProperty(property)
         )  :
      property==CHART_PROP_TIMEFRAME      ?  CMessage::Text(MSG_LIB_TEXT_BAR_PERIOD)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+TimeframeDescription((ENUM_TIMEFRAMES)this.GetProperty(property))
         )  :
      property==CHART_PROP_SHOW   ?  CMessage::Text(MSG_CHART_OBJ_SHOW)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(this.GetProperty(property) ? CMessage::Text(MSG_LIB_TEXT_YES) : CMessage::Text(MSG_LIB_TEXT_NO))
         )  :
      property==CHART_PROP_IS_OBJECT      ?  CMessage::Text(MSG_CHART_OBJ_IS_OBJECT)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(this.GetProperty(property) ? CMessage::Text(MSG_LIB_TEXT_YES) : CMessage::Text(MSG_LIB_TEXT_NO))
         )  :
      property==CHART_PROP_BRING_TO_TOP   ?  CMessage::Text(MSG_CHART_OBJ_BRING_TO_TOP)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(this.GetProperty(property) ? CMessage::Text(MSG_LIB_TEXT_YES) : CMessage::Text(MSG_LIB_TEXT_NO))
         )  :
      property==CHART_PROP_CONTEXT_MENU   ?  CMessage::Text(MSG_CHART_OBJ_CONTEXT_MENU)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(this.GetProperty(property) ? CMessage::Text(MSG_LIB_TEXT_YES) : CMessage::Text(MSG_LIB_TEXT_NO))
         )  :
      property==CHART_PROP_CROSSHAIR_TOOL ?  CMessage::Text(MSG_CHART_OBJ_CROSSHAIR_TOOL)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(this.GetProperty(property) ? CMessage::Text(MSG_LIB_TEXT_YES) : CMessage::Text(MSG_LIB_TEXT_NO))
         )  :
      property==CHART_PROP_MOUSE_SCROLL   ?  CMessage::Text(MSG_CHART_OBJ_MOUSE_SCROLL)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(this.GetProperty(property) ? CMessage::Text(MSG_LIB_TEXT_YES) : CMessage::Text(MSG_LIB_TEXT_NO))
         )  :
      property==CHART_PROP_EVENT_MOUSE_WHEEL ?  CMessage::Text(MSG_CHART_OBJ_EVENT_MOUSE_WHEEL)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(this.GetProperty(property) ? CMessage::Text(MSG_LIB_TEXT_YES) : CMessage::Text(MSG_LIB_TEXT_NO))
         )  :
      property==CHART_PROP_EVENT_MOUSE_MOVE  ?  CMessage::Text(MSG_CHART_OBJ_EVENT_MOUSE_MOVE)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(this.GetProperty(property) ? CMessage::Text(MSG_LIB_TEXT_YES) : CMessage::Text(MSG_LIB_TEXT_NO))
         )  :
      property==CHART_PROP_EVENT_OBJECT_CREATE  ?  CMessage::Text(MSG_CHART_OBJ_EVENT_OBJECT_CREATE)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(this.GetProperty(property) ? CMessage::Text(MSG_LIB_TEXT_YES) : CMessage::Text(MSG_LIB_TEXT_NO))
         )  :
      property==CHART_PROP_EVENT_OBJECT_DELETE  ?  CMessage::Text(MSG_CHART_OBJ_EVENT_OBJECT_DELETE)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(this.GetProperty(property) ? CMessage::Text(MSG_LIB_TEXT_YES) : CMessage::Text(MSG_LIB_TEXT_NO))
         )  :
      property==CHART_PROP_MODE           ?  CMessage::Text(MSG_CHART_OBJ_MODE)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+ChartModeDescription((ENUM_CHART_MODE)this.GetProperty(property))
         )  :
      property==CHART_PROP_FOREGROUND     ?  CMessage::Text(MSG_CHART_OBJ_FOREGROUND)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(this.GetProperty(property) ? CMessage::Text(MSG_LIB_TEXT_YES) : CMessage::Text(MSG_LIB_TEXT_NO))
         )  :
      property==CHART_PROP_SHIFT        ?  CMessage::Text(MSG_CHART_OBJ_SHIFT)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(this.GetProperty(property) ? CMessage::Text(MSG_LIB_TEXT_YES) : CMessage::Text(MSG_LIB_TEXT_NO))
         )  :
      property==CHART_PROP_AUTOSCROLL        ?  CMessage::Text(MSG_CHART_OBJ_AUTOSCROLL)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(this.GetProperty(property) ? CMessage::Text(MSG_LIB_TEXT_YES) : CMessage::Text(MSG_LIB_TEXT_NO))
         )  :
      property==CHART_PROP_KEYBOARD_CONTROL        ?  CMessage::Text(MSG_CHART_OBJ_KEYBOARD_CONTROL)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(this.GetProperty(property) ? CMessage::Text(MSG_LIB_TEXT_YES) : CMessage::Text(MSG_LIB_TEXT_NO))
         )  :
      property==CHART_PROP_QUICK_NAVIGATION        ?  CMessage::Text(MSG_CHART_OBJ_QUICK_NAVIGATION)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(this.GetProperty(property) ? CMessage::Text(MSG_LIB_TEXT_YES) : CMessage::Text(MSG_LIB_TEXT_NO))
         )  :
      property==CHART_PROP_SCALE        ?  CMessage::Text(MSG_CHART_OBJ_SCALE)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.GetProperty(property)
         )  :
      property==CHART_PROP_SCALEFIX        ?  CMessage::Text(MSG_CHART_OBJ_SCALEFIX)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(this.GetProperty(property) ? CMessage::Text(MSG_LIB_TEXT_YES) : CMessage::Text(MSG_LIB_TEXT_NO))
         )  :
      property==CHART_PROP_SCALEFIX_11        ?  CMessage::Text(MSG_CHART_OBJ_SCALEFIX_11)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(this.GetProperty(property) ? CMessage::Text(MSG_LIB_TEXT_YES) : CMessage::Text(MSG_LIB_TEXT_NO))
         )  :
      property==CHART_PROP_SCALE_PT_PER_BAR  ?  CMessage::Text(MSG_CHART_OBJ_SCALE_PT_PER_BAR)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(this.GetProperty(property) ? CMessage::Text(MSG_LIB_TEXT_YES) : CMessage::Text(MSG_LIB_TEXT_NO))
         )  :
      property==CHART_PROP_SHOW_TICKER        ?  CMessage::Text(MSG_CHART_OBJ_SHOW_TICKER)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(this.GetProperty(property) ? CMessage::Text(MSG_LIB_TEXT_YES) : CMessage::Text(MSG_LIB_TEXT_NO))
         )  :
      property==CHART_PROP_SHOW_OHLC        ?  CMessage::Text(MSG_CHART_OBJ_SHOW_OHLC)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(this.GetProperty(property) ? CMessage::Text(MSG_LIB_TEXT_YES) : CMessage::Text(MSG_LIB_TEXT_NO))
         )  :
      property==CHART_PROP_SHOW_BID_LINE  ?  CMessage::Text(MSG_CHART_OBJ_SHOW_BID_LINE)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(this.GetProperty(property) ? CMessage::Text(MSG_LIB_TEXT_YES) : CMessage::Text(MSG_LIB_TEXT_NO))
         )  :
      property==CHART_PROP_SHOW_ASK_LINE  ?  CMessage::Text(MSG_CHART_OBJ_SHOW_ASK_LINE)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(this.GetProperty(property) ? CMessage::Text(MSG_LIB_TEXT_YES) : CMessage::Text(MSG_LIB_TEXT_NO))
         )  :
      property==CHART_PROP_SHOW_LAST_LINE ?  CMessage::Text(MSG_CHART_OBJ_SHOW_LAST_LINE)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(this.GetProperty(property) ? CMessage::Text(MSG_LIB_TEXT_YES) : CMessage::Text(MSG_LIB_TEXT_NO))
         )  :
      property==CHART_PROP_SHOW_PERIOD_SEP   ?  CMessage::Text(MSG_CHART_OBJ_SHOW_PERIOD_SEP)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(this.GetProperty(property) ? CMessage::Text(MSG_LIB_TEXT_YES) : CMessage::Text(MSG_LIB_TEXT_NO))
         )  :
      property==CHART_PROP_SHOW_GRID        ?  CMessage::Text(MSG_CHART_OBJ_SHOW_GRID)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(this.GetProperty(property) ? CMessage::Text(MSG_LIB_TEXT_YES) : CMessage::Text(MSG_LIB_TEXT_NO))
         )  :
      property==CHART_PROP_SHOW_VOLUMES   ?  CMessage::Text(MSG_CHART_OBJ_SHOW_VOLUMES)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+ChartModeVolumeDescription((ENUM_CHART_VOLUME_MODE)this.GetProperty(property))
         )  :
      property==CHART_PROP_SHOW_OBJECT_DESCR ?  CMessage::Text(MSG_CHART_OBJ_SHOW_OBJECT_DESCR)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(this.GetProperty(property) ? CMessage::Text(MSG_LIB_TEXT_YES) : CMessage::Text(MSG_LIB_TEXT_NO))
         )  :
      property==CHART_PROP_VISIBLE_BARS   ?  CMessage::Text(MSG_CHART_OBJ_VISIBLE_BARS)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.GetProperty(property)
         )  :
      property==CHART_PROP_WINDOWS_TOTAL  ?  CMessage::Text(MSG_CHART_OBJ_WINDOWS_TOTAL)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.GetProperty(property)
         )  :
      property==CHART_PROP_WINDOW_HANDLE  ?  CMessage::Text(MSG_CHART_OBJ_WINDOW_HANDLE)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.GetProperty(property)
         )  :
      property==CHART_PROP_FIRST_VISIBLE_BAR ?  CMessage::Text(MSG_CHART_OBJ_FIRST_VISIBLE_BAR)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.GetProperty(property)
         )  :
      property==CHART_PROP_WIDTH_IN_BARS  ?  CMessage::Text(MSG_CHART_OBJ_WIDTH_IN_BARS)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.GetProperty(property)
         )  :
      property==CHART_PROP_WIDTH_IN_PIXELS   ?  CMessage::Text(MSG_CHART_OBJ_WIDTH_IN_PIXELS)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.GetProperty(property)
         )  :
      property==CHART_PROP_COLOR_BACKGROUND        ?  CMessage::Text(MSG_CHART_OBJ_COLOR_BACKGROUND)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+::ColorToString((color)this.GetProperty(property),true)
         )  :
      property==CHART_PROP_COLOR_FOREGROUND  ?  CMessage::Text(MSG_CHART_OBJ_COLOR_FOREGROUND)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+::ColorToString((color)this.GetProperty(property),true)
         )  :
      property==CHART_PROP_COLOR_GRID     ?  CMessage::Text(MSG_CHART_OBJ_COLOR_GRID)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+::ColorToString((color)this.GetProperty(property),true)
         )  :
      property==CHART_PROP_COLOR_VOLUME   ?  CMessage::Text(MSG_CHART_OBJ_COLOR_VOLUME)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+::ColorToString((color)this.GetProperty(property),true)
         )  :
      property==CHART_PROP_COLOR_CHART_UP ?  CMessage::Text(MSG_CHART_OBJ_COLOR_CHART_UP)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+::ColorToString((color)this.GetProperty(property),true)
         )  :
      property==CHART_PROP_COLOR_CHART_DOWN  ?  CMessage::Text(MSG_CHART_OBJ_COLOR_CHART_DOWN)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+::ColorToString((color)this.GetProperty(property),true)
         )  :
      property==CHART_PROP_COLOR_CHART_LINE  ?  CMessage::Text(MSG_CHART_OBJ_COLOR_CHART_LINE)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+::ColorToString((color)this.GetProperty(property),true)
         )  :
      property==CHART_PROP_COLOR_CANDLE_BULL ?  CMessage::Text(MSG_CHART_OBJ_COLOR_CANDLE_BULL)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+::ColorToString((color)this.GetProperty(property),true)
         )  :
      property==CHART_PROP_COLOR_CANDLE_BEAR ?  CMessage::Text(MSG_CHART_OBJ_COLOR_CANDLE_BEAR)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+::ColorToString((color)this.GetProperty(property),true)
         )  :
      property==CHART_PROP_COLOR_BID        ?  CMessage::Text(MSG_CHART_OBJ_COLOR_BID)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+::ColorToString((color)this.GetProperty(property),true)
         )  :
      property==CHART_PROP_COLOR_ASK        ?  CMessage::Text(MSG_CHART_OBJ_COLOR_ASK)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+::ColorToString((color)this.GetProperty(property),true)
         )  :
      property==CHART_PROP_COLOR_LAST        ?  CMessage::Text(MSG_CHART_OBJ_COLOR_LAST)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+::ColorToString((color)this.GetProperty(property),true)
         )  :
      property==CHART_PROP_COLOR_STOP_LEVEL  ?  CMessage::Text(MSG_CHART_OBJ_COLOR_STOP_LEVEL)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+::ColorToString((color)this.GetProperty(property),true)
         )  :
      property==CHART_PROP_SHOW_TRADE_LEVELS ?  CMessage::Text(MSG_CHART_OBJ_SHOW_TRADE_LEVELS)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(this.GetProperty(property) ? CMessage::Text(MSG_LIB_TEXT_YES) : CMessage::Text(MSG_LIB_TEXT_NO))
         )  :
      property==CHART_PROP_DRAG_TRADE_LEVELS ?  CMessage::Text(MSG_CHART_OBJ_DRAG_TRADE_LEVELS)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(this.GetProperty(property) ? CMessage::Text(MSG_LIB_TEXT_YES) : CMessage::Text(MSG_LIB_TEXT_NO))
         )  :
      property==CHART_PROP_SHOW_DATE_SCALE   ?  CMessage::Text(MSG_CHART_OBJ_SHOW_DATE_SCALE)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(this.GetProperty(property) ? CMessage::Text(MSG_LIB_TEXT_YES) : CMessage::Text(MSG_LIB_TEXT_NO))
         )  :
      property==CHART_PROP_SHOW_PRICE_SCALE  ?  CMessage::Text(MSG_CHART_OBJ_SHOW_PRICE_SCALE)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(this.GetProperty(property) ? CMessage::Text(MSG_LIB_TEXT_YES) : CMessage::Text(MSG_LIB_TEXT_NO))
         )  :
      property==CHART_PROP_SHOW_ONE_CLICK ?  CMessage::Text(MSG_CHART_OBJ_SHOW_ONE_CLICK)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(this.GetProperty(property) ? CMessage::Text(MSG_LIB_TEXT_YES) : CMessage::Text(MSG_LIB_TEXT_NO))
         )  :
      property==CHART_PROP_IS_MAXIMIZED   ?  CMessage::Text(MSG_CHART_OBJ_IS_MAXIMIZED)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(this.GetProperty(property) ? CMessage::Text(MSG_LIB_TEXT_YES) : CMessage::Text(MSG_LIB_TEXT_NO))
         )  :
      property==CHART_PROP_IS_MINIMIZED   ?  CMessage::Text(MSG_CHART_OBJ_IS_MINIMIZED)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(this.GetProperty(property) ? CMessage::Text(MSG_LIB_TEXT_YES) : CMessage::Text(MSG_LIB_TEXT_NO))
         )  :
      property==CHART_PROP_IS_DOCKED      ?  CMessage::Text(MSG_CHART_OBJ_IS_DOCKED)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(this.GetProperty(property) ? CMessage::Text(MSG_LIB_TEXT_YES) : CMessage::Text(MSG_LIB_TEXT_NO))
         )  :
      property==CHART_PROP_FLOAT_LEFT     ?  CMessage::Text(MSG_CHART_OBJ_FLOAT_LEFT)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.GetProperty(property)
         )  :
      property==CHART_PROP_FLOAT_TOP      ?  CMessage::Text(MSG_CHART_OBJ_FLOAT_TOP)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.GetProperty(property)
         )  :
      property==CHART_PROP_FLOAT_RIGHT    ?  CMessage::Text(MSG_CHART_OBJ_FLOAT_RIGHT)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.GetProperty(property)
         )  :
      property==CHART_PROP_FLOAT_BOTTOM   ?  CMessage::Text(MSG_CHART_OBJ_FLOAT_BOTTOM)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.GetProperty(property)
         )  :
      ""
     );
  }
//+------------------------------------------------------------------+
//| Return description of object's real property                     |
//+------------------------------------------------------------------+
string CChartObj::GetPropertyDescription(ENUM_CHART_PROP_DOUBLE property)
  {
   return
     (
      property==CHART_PROP_SHIFT_SIZE        ?  CMessage::Text(MSG_CHART_OBJ_SHIFT_SIZE)+
         (!this.SupportProperty(property)    ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+::DoubleToString(this.GetProperty(property),2)
         )  :
      property==CHART_PROP_FIXED_POSITION    ?  CMessage::Text(MSG_CHART_OBJ_FIXED_POSITION)+
         (!this.SupportProperty(property)    ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+::DoubleToString(this.GetProperty(property),2)
         )  :
      property==CHART_PROP_FIXED_MAX         ?  CMessage::Text(MSG_CHART_OBJ_FIXED_MAX)+
         (!this.SupportProperty(property)    ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+::DoubleToString(this.GetProperty(property),this.m_digits)
         )  :
      property==CHART_PROP_FIXED_MIN         ?  CMessage::Text(MSG_CHART_OBJ_FIXED_MIN)+
         (!this.SupportProperty(property)    ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+::DoubleToString(this.GetProperty(property),this.m_digits)
         )  :
      property==CHART_PROP_POINTS_PER_BAR    ?  CMessage::Text(MSG_CHART_OBJ_POINTS_PER_BAR)+
         (!this.SupportProperty(property)    ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+::DoubleToString(this.GetProperty(property),2)
         )  :
      property==CHART_PROP_PRICE_MIN         ?  CMessage::Text(MSG_CHART_OBJ_PRICE_MIN)+
         (!this.SupportProperty(property)    ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+::DoubleToString(this.GetProperty(property),this.m_digits)
         )  :
      property==CHART_PROP_PRICE_MAX         ?  CMessage::Text(MSG_CHART_OBJ_PRICE_MAX)+
         (!this.SupportProperty(property)    ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+::DoubleToString(this.GetProperty(property),this.m_digits)
         )  :
      ""
     );
  }
//+------------------------------------------------------------------+
//| Return description of object's string property                   |
//+------------------------------------------------------------------+
string CChartObj::GetPropertyDescription(ENUM_CHART_PROP_STRING property)
  {
   return
     (
      property==CHART_PROP_COMMENT     ?  CMessage::Text(MSG_CHART_OBJ_COMMENT)+": \""+this.GetProperty(property)+"\""     :
      property==CHART_PROP_EXPERT_NAME ?  CMessage::Text(MSG_CHART_OBJ_EXPERT_NAME)+": \""+this.GetProperty(property)+"\"" :
      property==CHART_PROP_SCRIPT_NAME ?  CMessage::Text(MSG_CHART_OBJ_SCRIPT_NAME)+": \""+this.GetProperty(property)+"\"" :
      property==CHART_PROP_SYMBOL      ?  CMessage::Text(MSG_LIB_PROP_SYMBOL)+": \""+this.GetProperty(property)+"\""       :
      ""
     );
  }
//+------------------------------------------------------------------+
```

A property passed to the method is checked and the string with the appropriate property description is returned.

**The method displaying the full description of the object properties:**

```
//+------------------------------------------------------------------+
//| Display object properties in the journal                         |
//+------------------------------------------------------------------+
void CChartObj::Print(const bool full_prop=false)
  {
   ::Print("============= ",CMessage::Text(MSG_LIB_PARAMS_LIST_BEG)," (",this.Header(),") =============");
   int beg=0, end=CHART_PROP_INTEGER_TOTAL;
   for(int i=beg; i<end; i++)
     {
      ENUM_CHART_PROP_INTEGER prop=(ENUM_CHART_PROP_INTEGER)i;
      if(!full_prop && !this.SupportProperty(prop)) continue;
      ::Print(this.GetPropertyDescription(prop));
     }
   ::Print("------");
   beg=end; end+=CHART_PROP_DOUBLE_TOTAL;
   for(int i=beg; i<end; i++)
     {
      ENUM_CHART_PROP_DOUBLE prop=(ENUM_CHART_PROP_DOUBLE)i;
      if(!full_prop && !this.SupportProperty(prop)) continue;
      ::Print(this.GetPropertyDescription(prop));
     }
   ::Print("------");
   beg=end; end+=CHART_PROP_STRING_TOTAL;
   for(int i=beg; i<end; i++)
     {
      ENUM_CHART_PROP_STRING prop=(ENUM_CHART_PROP_STRING)i;
      if(!full_prop && !this.SupportProperty(prop)) continue;
      ::Print(this.GetPropertyDescription(prop));
     }
   ::Print("============= ",CMessage::Text(MSG_LIB_PARAMS_LIST_END)," (",this.Header(),") =============\n");
  }
//+------------------------------------------------------------------+
```

In three loops by all object properties, receive the description of each subsequent property and print it to the journal.

**The method displaying the short object description in the journal:**

```
//+------------------------------------------------------------------+
//| Display a short description of the object in the journal         |
//+------------------------------------------------------------------+
void CChartObj::PrintShort(const bool dash=false)
  {
   ::Print((dash ? "- " : ""),this.Header()," ID: ",(string)this.ID(),", HWND: ",(string)this.Handle());
  }
//+------------------------------------------------------------------+
```

The method receives the flag indicating the necessity to display a hyphen before an object description. The hyphen is needed in the chart object collection class to display a short description of the entire collection. By default, there is no hyphen. The method creates the string featuring the object description and some of its additional properties — chart ID and window handle.

**The method returning a short object name:**

```
//+------------------------------------------------------------------+
//| Return the object short name                                     |
//+------------------------------------------------------------------+
string CChartObj::Header(void)
  {
   return(CMessage::Text(MSG_CHART_OBJ_CHART_WINDOW)+" "+this.Symbol()+" "+TimeframeDescription(this.Timeframe()));
  }
//+------------------------------------------------------------------+
```

A string consisting of a header and names of a chart symbol and timeframe is created in the method.

All these methods are standard for library objects and form the basis of each object. For the convenient use of these methods, I usually add methods for quick access to the object properties. I will have multiple methods since the chart object features many different properties.

Basically, all its properties are flags indicating the status of a chart parameter. They can be enabled or disabled, i.e. there are only two states. This means we need the methods for setting the flags of such object properties. These will be private methods. The methods enabling and disabling a property will be made public. This makes them more convenient to use.

In the private section of the class, declare the methods for setting the flags of some object properties, the methods for setting the properties having a boundary number of values (3 and 6), as well as the methods for setting only the chart object properties — for read-only chart parameters — to be able to set a value of an appropriate chart parameter to the chart object property:

```
//+------------------------------------------------------------------+
//|                                                     ChartObj.mqh |
//|                                  Copyright 2021, MetaQuotes Ltd. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2021, MetaQuotes Ltd."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
#property strict    // Necessary for mql4
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include "..\..\Objects\BaseObj.mqh"
//+------------------------------------------------------------------+
//| Chart object class                                               |
//+------------------------------------------------------------------+
class CChartObj : public CBaseObj
  {
private:
   long              m_long_prop[CHART_PROP_INTEGER_TOTAL];       // Integer properties
   double            m_double_prop[CHART_PROP_DOUBLE_TOTAL];      // Real properties
   string            m_string_prop[CHART_PROP_STRING_TOTAL];      // String properties
   int               m_digits;                                    // Symbol's Digits()

//--- Return the index of the array the (1) double and (2) string properties are actually located at
   int               IndexProp(ENUM_CHART_PROP_DOUBLE property)   const { return(int)property-CHART_PROP_INTEGER_TOTAL;                         }
   int               IndexProp(ENUM_CHART_PROP_STRING property)   const { return(int)property-CHART_PROP_INTEGER_TOTAL-CHART_PROP_DOUBLE_TOTAL; }

//--- The methods of setting parameter flags
   bool              SetShowFlag(const string source,const bool flag,const bool redraw=false);
   bool              SetBringToTopFlag(const string source,const bool flag,const bool redraw=false);
   bool              SetContextMenuFlag(const string source,const bool flag,const bool redraw=false);
   bool              SetCrosshairToolFlag(const string source,const bool flag,const bool redraw=false);
   bool              SetMouseScrollFlag(const string source,const bool flag,const bool redraw=false);
   bool              SetEventMouseWhellFlag(const string source,const bool flag,const bool redraw=false);
   bool              SetEventMouseMoveFlag(const string source,const bool flag,const bool redraw=false);
   bool              SetEventObjectCreateFlag(const string source,const bool flag,const bool redraw=false);
   bool              SetEventObjectDeleteFlag(const string source,const bool flag,const bool redraw=false);
   bool              SetForegroundFlag(const string source,const bool flag,const bool redraw=false);
   bool              SetShiftFlag(const string source,const bool flag,const bool redraw=false);
   bool              SetAutoscrollFlag(const string source,const bool flag,const bool redraw=false);
   bool              SetKeyboardControlFlag(const string source,const bool flag,const bool redraw=false);
   bool              SetQuickNavigationFlag(const string source,const bool flag,const bool redraw=false);
   bool              SetScaleFixFlag(const string source,const bool flag,const bool redraw=false);
   bool              SetScaleFix11Flag(const string source,const bool flag,const bool redraw=false);
   bool              SetScalePTPerBarFlag(const string source,const bool flag,const bool redraw=false);
   bool              SetShowTickerFlag(const string source,const bool flag,const bool redraw=false);
   bool              SetShowOHLCFlag(const string source,const bool flag,const bool redraw=false);
   bool              SetShowBidLineFlag(const string source,const bool flag,const bool redraw=false);
   bool              SetShowAskLineFlag(const string source,const bool flag,const bool redraw=false);
   bool              SetShowLastLineFlag(const string source,const bool flag,const bool redraw=false);
   bool              SetShowPeriodSeparatorsFlag(const string source,const bool flag,const bool redraw=false);
   bool              SetShowGridFlag(const string source,const bool flag,const bool redraw=false);
   bool              SetShowObjectDescriptionsFlag(const string source,const bool flag,const bool redraw=false);
   bool              SetShowTradeLevelsFlag(const string source,const bool flag,const bool redraw=false);
   bool              SetDragTradeLevelsFlag(const string source,const bool flag,const bool redraw=false);
   bool              SetShowDateScaleFlag(const string source,const bool flag,const bool redraw=false);
   bool              SetShowPriceScaleFlag(const string source,const bool flag,const bool redraw=false);
   bool              SetShowOneClickPanelFlag(const string source,const bool flag,const bool redraw=false);
   bool              SetDockedFlag(const string source,const bool flag,const bool redraw=false);

//--- The methods of setting property values
   bool              SetMode(const string source,const ENUM_CHART_MODE mode,const bool redraw=false);
   bool              SetScale(const string source,const int scale,const bool redraw=false);
   bool              SetModeVolume(const string source,const ENUM_CHART_VOLUME_MODE mode,const bool redraw=false);
   void              SetVisibleBars(void);
   void              SetWindowsTotal(void);
   void              SetFirstVisibleBars(void);
   void              SetWidthInBars(void);
   void              SetWidthInPixels(void);
   void              SetMaximizedFlag(void);
   void              SetMinimizedFlag(void);
   void              SetExpertName(void);
   void              SetScriptName(void);

public:
```

Implement the declared private methods outside the class body.

The methods of setting flags are identical to each other.

Therefore, let's consider the logic using the **method setting the flag of drawing a price chart**:

```
//+------------------------------------------------------------------+
//| Set the flag of drawing the price chart                          |
//+------------------------------------------------------------------+
bool CChartObj::SetShowFlag(const string source,const bool flag,const bool redraw=false)
  {
   ::ResetLastError();
   if(!::ChartSetInteger(this.ID(),CHART_SHOW,flag))
     {
      CMessage::ToLog(source,::GetLastError(),true);
      return false;
     }
   this.SetProperty(CHART_PROP_SHOW,flag);
   if(redraw)
      ::ChartRedraw(this.ID());
   return true;
  }
//+------------------------------------------------------------------+
```

The method receives the source ( **source**) — name of the method the private method was called from, the flag value ( **flag**) that should be set in the chart parameters and the chart object property, as well as the flag indicating the necessity to redraw the chart ( **redraw**).

The functions of working with ChartSet\* charts are asynchronous. This means they return only the result of placing the command to the chart event queue rather than the result of changing the parameter itself. Usually, the result of the function operation appears after a certain chart event — changing its size, new tick arrival, chart update, etc. To display the parameter change immediately, we need to forcibly update the chart. This is done by the [ChartRedraw()](https://www.mql5.com/en/docs/chart_operations/chartredraw) function.

However, the change usually affects several chart parameters at once, i.e. most often this is a batch change. In this case, we should first send all events of changes in all the necessary chart parameters and call the forced chart redraw afterwards to avoid redrawing the chart after each parameter change command. The **redraw** flag is used for that. By default, it is set to false.

The first thing we do here is send a chart parameter change command. If the event is not set to the queue, report an error and return false.

If the event has been successfully set into the queue, change the chart object property. If the redraw flag is activated, forcibly redraw the chart. Return true  — the method operation is successful.

**Implementation of the remaining private methods for setting the flags** is identical to the considered one. Let's provide their full listing:

```
//+------------------------------------------------------------------+
//| Set the chart flag above all others                              |
//+------------------------------------------------------------------+
bool CChartObj::SetBringToTopFlag(const string source,const bool flag,const bool redraw=false)
  {
   ::ResetLastError();
   if(!::ChartSetInteger(this.ID(),CHART_BRING_TO_TOP,flag))
     {
      CMessage::ToLog(source,::GetLastError(),true);
      return false;
     }
   this.SetProperty(CHART_PROP_BRING_TO_TOP,flag);
   if(redraw)
      ::ChartRedraw(this.ID());
   return true;
  }
//+------------------------------------------------------------------+
//| Set the flag of accessing the context menu                       |
//| upon pressing the right mouse button                             |
//+------------------------------------------------------------------+
bool CChartObj::SetContextMenuFlag(const string source,const bool flag,const bool redraw=false)
  {
   ::ResetLastError();
   if(!::ChartSetInteger(this.ID(),CHART_CONTEXT_MENU,flag))
     {
      CMessage::ToLog(source,::GetLastError(),true);
      return false;
     }
   this.SetProperty(CHART_PROP_CONTEXT_MENU,flag);
   if(redraw)
      ::ChartRedraw(this.ID());
   return true;
  }
//+------------------------------------------------------------------+
//| Set the flag of accessing the Crosshair                          |
//| upon pressing the middle mouse button                            |
//+------------------------------------------------------------------+
bool CChartObj::SetCrosshairToolFlag(const string source,const bool flag,const bool redraw=false)
  {
   ::ResetLastError();
   if(!::ChartSetInteger(this.ID(),CHART_CROSSHAIR_TOOL,flag))
     {
      CMessage::ToLog(source,::GetLastError(),true);
      return false;
     }
   this.SetProperty(CHART_PROP_CROSSHAIR_TOOL,flag);
   if(redraw)
      ::ChartRedraw(this.ID());
   return true;
  }
//+------------------------------------------------------------------+
//| Set the chart scroll flag                                        |
//| horizontally using the left mouse button                         |
//+------------------------------------------------------------------+
bool CChartObj::SetMouseScrollFlag(const string source,const bool flag,const bool redraw=false)
  {
   ::ResetLastError();
   if(!::ChartSetInteger(this.ID(),CHART_MOUSE_SCROLL,flag))
     {
      CMessage::ToLog(source,::GetLastError(),true);
      return false;
     }
   this.SetProperty(CHART_PROP_MOUSE_SCROLL,flag);
   if(redraw)
      ::ChartRedraw(this.ID());
   return true;
  }
//+------------------------------------------------------------------+
//| Set the flag of sending mouse wheel event messages               |
//| to all MQL5 programs on the chart                                |
//+------------------------------------------------------------------+
bool CChartObj::SetEventMouseWhellFlag(const string source,const bool flag,const bool redraw=false)
  {
   ::ResetLastError();
   if(!::ChartSetInteger(this.ID(),CHART_EVENT_MOUSE_WHEEL,flag))
     {
      CMessage::ToLog(source,::GetLastError(),true);
      return false;
     }
   this.SetProperty(CHART_PROP_EVENT_MOUSE_WHEEL,flag);
   if(redraw)
      ::ChartRedraw(this.ID());
   return true;
  }
//+----------------------------------------------------------------------+
//| Set the flag of sending mouse button movement and pressing messages  |
//| to all MQL5 programs on the chart                                    |
//+----------------------------------------------------------------------+
bool CChartObj::SetEventMouseMoveFlag(const string source,const bool flag,const bool redraw=false)
  {
   ::ResetLastError();
   if(!::ChartSetInteger(this.ID(),CHART_EVENT_MOUSE_MOVE,flag))
     {
      CMessage::ToLog(source,::GetLastError(),true);
      return false;
     }
   this.SetProperty(CHART_PROP_EVENT_MOUSE_MOVE,flag);
   if(redraw)
      ::ChartRedraw(this.ID());
   return true;
  }
//+------------------------------------------------------------------+
//| Set the flag of sending graphical object creation event messages |
//| to all MQL5 programs on the chart                                |
//+------------------------------------------------------------------+
bool CChartObj::SetEventObjectCreateFlag(const string source,const bool flag,const bool redraw=false)
  {
   ::ResetLastError();
   if(!::ChartSetInteger(this.ID(),CHART_EVENT_OBJECT_CREATE,flag))
     {
      CMessage::ToLog(source,::GetLastError(),true);
      return false;
     }
   this.SetProperty(CHART_PROP_EVENT_OBJECT_CREATE,flag);
   if(redraw)
      ::ChartRedraw(this.ID());
   return true;
  }
//+----------------------------------------------------------------------+
//| Set the flag of sending graphical object destruction event messages  |
//| to all MQL5 programs on the chart                                    |
//+----------------------------------------------------------------------+
bool CChartObj::SetEventObjectDeleteFlag(const string source,const bool flag,const bool redraw=false)
  {
   ::ResetLastError();
   if(!::ChartSetInteger(this.ID(),CHART_EVENT_OBJECT_DELETE,flag))
     {
      CMessage::ToLog(source,::GetLastError(),true);
      return false;
     }
   this.SetProperty(CHART_PROP_EVENT_OBJECT_DELETE,flag);
   if(redraw)
      ::ChartRedraw(this.ID());
   return true;
  }
//+------------------------------------------------------------------+
//| Set the price chart flag in the foreground                       |
//+------------------------------------------------------------------+
bool CChartObj::SetForegroundFlag(const string source,const bool flag,const bool redraw=false)
  {
   ::ResetLastError();
   if(!::ChartSetInteger(this.ID(),CHART_FOREGROUND,flag))
     {
      CMessage::ToLog(source,::GetLastError(),true);
      return false;
     }
   this.SetProperty(CHART_PROP_FOREGROUND,flag);
   if(redraw)
      ::ChartRedraw(this.ID());
   return true;
  }
//+------------------------------------------------------------------+
//| Set the flag of the price chart shift from the right border      |
//+------------------------------------------------------------------+
bool CChartObj::SetShiftFlag(const string source,const bool flag,const bool redraw=false)
  {
   ::ResetLastError();
   if(!::ChartSetInteger(this.ID(),CHART_SHIFT,flag))
     {
      CMessage::ToLog(source,::GetLastError(),true);
      return false;
     }
   this.SetProperty(CHART_PROP_SHIFT,flag);
   if(redraw)
      ::ChartRedraw(this.ID());
   return true;
  }
//+------------------------------------------------------------------+
//|Set the flag for automatically moving to the right chart border   |
//+------------------------------------------------------------------+
bool CChartObj::SetAutoscrollFlag(const string source,const bool flag,const bool redraw=false)
  {
   ::ResetLastError();
   if(!::ChartSetInteger(this.ID(),CHART_AUTOSCROLL,flag))
     {
      CMessage::ToLog(source,::GetLastError(),true);
      return false;
     }
   this.SetProperty(CHART_PROP_AUTOSCROLL,flag);
   if(redraw)
      ::ChartRedraw(this.ID());
   return true;
  }
//+------------------------------------------------------------------+
//| Set the flag of managing the chart using a keyboard              |
//+------------------------------------------------------------------+
bool CChartObj::SetKeyboardControlFlag(const string source,const bool flag,const bool redraw=false)
  {
   ::ResetLastError();
   if(!::ChartSetInteger(this.ID(),CHART_KEYBOARD_CONTROL,flag))
     {
      CMessage::ToLog(source,::GetLastError(),true);
      return false;
     }
   this.SetProperty(CHART_PROP_KEYBOARD_CONTROL,flag);
   if(redraw)
      ::ChartRedraw(this.ID());
   return true;
  }
//+---------------------------------------------------------------------+
//| Set the flag of intercepting Space and Enter keystrokes by the chart|
//| to activate the fast navigation bar                                 |
//+---------------------------------------------------------------------+
bool CChartObj::SetQuickNavigationFlag(const string source,const bool flag,const bool redraw=false)
  {
   ::ResetLastError();
   if(!::ChartSetInteger(this.ID(),CHART_QUICK_NAVIGATION,flag))
     {
      CMessage::ToLog(source,::GetLastError(),true);
      return false;
     }
   this.SetProperty(CHART_PROP_QUICK_NAVIGATION,flag);
   if(redraw)
      ::ChartRedraw(this.ID());
   return true;
  }
//+------------------------------------------------------------------+
//| Set the fixed scale flag                                         |
//+------------------------------------------------------------------+
bool CChartObj::SetScaleFixFlag(const string source,const bool flag,const bool redraw=false)
  {
   ::ResetLastError();
   if(!::ChartSetInteger(this.ID(),CHART_SCALEFIX,flag))
     {
      CMessage::ToLog(source,::GetLastError(),true);
      return false;
     }
   this.SetProperty(CHART_PROP_SCALEFIX,flag);
   if(redraw)
      ::ChartRedraw(this.ID());
   return true;
  }
//+------------------------------------------------------------------+
//| Set the 1:1 scale flag                                           |
//+------------------------------------------------------------------+
bool CChartObj::SetScaleFix11Flag(const string source,const bool flag,const bool redraw=false)
  {
   ::ResetLastError();
   if(!::ChartSetInteger(this.ID(),CHART_SCALEFIX_11,flag))
     {
      CMessage::ToLog(source,::GetLastError(),true);
      return false;
     }
   this.SetProperty(CHART_PROP_SCALEFIX_11,flag);
   if(redraw)
      ::ChartRedraw(this.ID());
   return true;
  }
//+------------------------------------------------------------------+
//| Set the flag of specifying a scale in points per bar             |
//+------------------------------------------------------------------+
bool CChartObj::SetScalePTPerBarFlag(const string source,const bool flag,const bool redraw=false)
  {
   ::ResetLastError();
   if(!::ChartSetInteger(this.ID(),CHART_SCALE_PT_PER_BAR,flag))
     {
      CMessage::ToLog(source,::GetLastError(),true);
      return false;
     }
   this.SetProperty(CHART_PROP_SCALE_PT_PER_BAR,flag);
   if(redraw)
      ::ChartRedraw(this.ID());
   return true;
  }
//+------------------------------------------------------------------+
//|Set the flag displaying a symbol ticker in the upper left corner  |
//+------------------------------------------------------------------+
bool CChartObj::SetShowTickerFlag(const string source,const bool flag,const bool redraw=false)
  {
   ::ResetLastError();
   if(!::ChartSetInteger(this.ID(),CHART_SHOW_TICKER,flag))
     {
      CMessage::ToLog(source,::GetLastError(),true);
      return false;
     }
   this.SetProperty(CHART_PROP_SHOW_TICKER,flag);
   if(redraw)
      ::ChartRedraw(this.ID());
   return true;
  }
//+------------------------------------------------------------------+
//|Set the flag displaying OHLC values in the upper left corner      |
//+------------------------------------------------------------------+
bool CChartObj::SetShowOHLCFlag(const string source,const bool flag,const bool redraw=false)
  {
   ::ResetLastError();
   if(!::ChartSetInteger(this.ID(),CHART_SHOW_OHLC,flag))
     {
      CMessage::ToLog(source,::GetLastError(),true);
      return false;
     }
   this.SetProperty(CHART_PROP_SHOW_OHLC,flag);
   if(redraw)
      ::ChartRedraw(this.ID());
   return true;
  }
//+------------------------------------------------------------------+
//| Set the flag of displaying the Bid value                         |
//| using a horizontal line on the chart                             |
//+------------------------------------------------------------------+
bool CChartObj::SetShowBidLineFlag(const string source,const bool flag,const bool redraw=false)
  {
   ::ResetLastError();
   if(!::ChartSetInteger(this.ID(),CHART_SHOW_BID_LINE,flag))
     {
      CMessage::ToLog(source,::GetLastError(),true);
      return false;
     }
   this.SetProperty(CHART_PROP_SHOW_BID_LINE,flag);
   if(redraw)
      ::ChartRedraw(this.ID());
   return true;
  }
//+------------------------------------------------------------------+
//| Set the flag of displaying the Ask value                         |
//| using a horizontal line on the chart                             |
//+------------------------------------------------------------------+
bool CChartObj::SetShowAskLineFlag(const string source,const bool flag,const bool redraw=false)
  {
   ::ResetLastError();
   if(!::ChartSetInteger(this.ID(),CHART_SHOW_ASK_LINE,flag))
     {
      CMessage::ToLog(source,::GetLastError(),flag);
      return false;
     }
   this.SetProperty(CHART_PROP_SHOW_ASK_LINE,flag);
   if(redraw)
      ::ChartRedraw(this.ID());
   return true;
  }
//+------------------------------------------------------------------+
//| Set the flag of displaying the Last value                        |
//| using a horizontal line on the chart                             |
//+------------------------------------------------------------------+
bool CChartObj::SetShowLastLineFlag(const string source,const bool flag,const bool redraw=false)
  {
   ::ResetLastError();
   if(!::ChartSetInteger(this.ID(),CHART_SHOW_LAST_LINE,flag))
     {
      CMessage::ToLog(source,::GetLastError(),true);
      return false;
     }
   this.SetProperty(CHART_PROP_SHOW_LAST_LINE,flag);
   if(redraw)
      ::ChartRedraw(this.ID());
   return true;
  }
//+------------------------------------------------------------------+
//| Set the flag of displaying vertical separators                   |
//| between adjacent periods                                         |
//+------------------------------------------------------------------+
bool CChartObj::SetShowPeriodSeparatorsFlag(const string source,const bool flag,const bool redraw=false)
  {
   ::ResetLastError();
   if(!::ChartSetInteger(this.ID(),CHART_SHOW_PERIOD_SEP,flag))
     {
      CMessage::ToLog(source,::GetLastError(),true);
      return false;
     }
   this.SetProperty(CHART_PROP_SHOW_PERIOD_SEP,flag);
   if(redraw)
      ::ChartRedraw(this.ID());
   return true;
  }
//+------------------------------------------------------------------+
//| Set the flag of displaying a grid on the chart                   |
//+------------------------------------------------------------------+
bool CChartObj::SetShowGridFlag(const string source,const bool flag,const bool redraw=false)
  {
   ::ResetLastError();
   if(!::ChartSetInteger(this.ID(),CHART_SHOW_GRID,flag))
     {
      CMessage::ToLog(source,::GetLastError(),true);
      return false;
     }
   this.SetProperty(CHART_PROP_SHOW_GRID,flag);
   if(redraw)
      ::ChartRedraw(this.ID());
   return true;
  }
//+------------------------------------------------------------------+
//| Set the flag of displaying object text descriptions              |
//+------------------------------------------------------------------+
bool CChartObj::SetShowObjectDescriptionsFlag(const string source,const bool flag,const bool redraw=false)
  {
   ::ResetLastError();
   if(!::ChartSetInteger(this.ID(),CHART_SHOW_OBJECT_DESCR,flag))
     {
      CMessage::ToLog(source,::GetLastError(),true);
      return false;
     }
   this.SetProperty(CHART_PROP_SHOW_OBJECT_DESCR,flag);
   if(redraw)
      ::ChartRedraw(this.ID());
   return true;
  }
//+------------------------------------------------------------------+
//| Set the flag of displaying trading levels on the chart           |
//+------------------------------------------------------------------+
bool CChartObj::SetShowTradeLevelsFlag(const string source,const bool flag,const bool redraw=false)
  {
   ::ResetLastError();
   if(!::ChartSetInteger(this.ID(),CHART_SHOW_TRADE_LEVELS,flag))
     {
      CMessage::ToLog(source,::GetLastError(),true);
      return false;
     }
   this.SetProperty(CHART_PROP_SHOW_TRADE_LEVELS,flag);
   if(redraw)
      ::ChartRedraw(this.ID());
   return true;
  }
//+------------------------------------------------------------------+
//| Set the flag enabling dragging                                   |
//| trading levels on a chart using a mouse                          |
//+------------------------------------------------------------------+
bool CChartObj::SetDragTradeLevelsFlag(const string source,const bool flag,const bool redraw=false)
  {
   ::ResetLastError();
   if(!::ChartSetInteger(this.ID(),CHART_DRAG_TRADE_LEVELS,flag))
     {
      CMessage::ToLog(source,::GetLastError(),true);
      return false;
     }
   this.SetProperty(CHART_PROP_DRAG_TRADE_LEVELS,flag);
   if(redraw)
      ::ChartRedraw(this.ID());
   return true;
  }
//+------------------------------------------------------------------+
//| Set the flag of displaying the time scale on the chart           |
//+------------------------------------------------------------------+
bool CChartObj::SetShowDateScaleFlag(const string source,const bool flag,const bool redraw=false)
  {
   ::ResetLastError();
   if(!::ChartSetInteger(this.ID(),CHART_SHOW_DATE_SCALE,flag))
     {
      CMessage::ToLog(source,::GetLastError(),true);
      return false;
     }
   this.SetProperty(CHART_PROP_SHOW_DATE_SCALE,flag);
   if(redraw)
      ::ChartRedraw(this.ID());
   return true;
  }
//+------------------------------------------------------------------+
//| Set the flag of displaying the price scale on the chart          |
//+------------------------------------------------------------------+
bool CChartObj::SetShowPriceScaleFlag(const string source,const bool flag,const bool redraw=false)
  {
   ::ResetLastError();
   if(!::ChartSetInteger(this.ID(),CHART_SHOW_PRICE_SCALE,flag))
     {
      CMessage::ToLog(source,::GetLastError(),true);
      return false;
     }
   this.SetProperty(CHART_PROP_SHOW_PRICE_SCALE,flag);
   if(redraw)
      ::ChartRedraw(this.ID());
   return true;
  }
//+------------------------------------------------------------------+
//|Set the flag of displaying the quick trading panel on the chart   |
//+------------------------------------------------------------------+
bool CChartObj::SetShowOneClickPanelFlag(const string source,const bool flag,const bool redraw=false)
  {
   ::ResetLastError();
   if(!::ChartSetInteger(this.ID(),CHART_SHOW_ONE_CLICK,flag))
     {
      CMessage::ToLog(source,::GetLastError(),true);
      return false;
     }
   this.SetProperty(CHART_PROP_SHOW_ONE_CLICK,flag);
   if(redraw)
      ::ChartRedraw(this.ID());
   return true;
  }
//+------------------------------------------------------------------+
//| Set the flag of docking a chart window                           |
//+------------------------------------------------------------------+
bool CChartObj::SetDockedFlag(const string source,const bool flag,const bool redraw=false)
  {
   ::ResetLastError();
   if(!::ChartSetInteger(this.ID(),CHART_IS_DOCKED,flag))
     {
      CMessage::ToLog(source,::GetLastError(),true);
      return false;
     }
   this.SetProperty(CHART_PROP_IS_DOCKED,flag);
   if(redraw)
      ::ChartRedraw(this.ID());
   return true;
  }
//+------------------------------------------------------------------+
```

As we can see, all these methods are identical to the one considered above except for the editable property (marked with color in the last method).

**Three private methods for setting chart, scale and chart volumes display modes:**

```
//+------------------------------------------------------------------+
//| Set the chart type                                               |
//+------------------------------------------------------------------+
bool CChartObj::SetMode(const string source,const ENUM_CHART_MODE mode,const bool redraw=false)
  {
   ::ResetLastError();
   if(!::ChartSetInteger(this.ID(),CHART_MODE,mode))
     {
      CMessage::ToLog(source,::GetLastError(),true);
      return false;
     }
   this.SetProperty(CHART_PROP_MODE,mode);
   if(redraw)
      ::ChartRedraw(this.ID());
   return true;
  }
//+------------------------------------------------------------------+
//| Set the chart scale                                              |
//+------------------------------------------------------------------+
bool CChartObj::SetScale(const string source,const int scale,const bool redraw=false)
  {
   int value=(scale<0 ? 0 : scale>5 ? 5 : scale);
   ::ResetLastError();
   if(!::ChartSetInteger(this.ID(),CHART_SCALE,value))
     {
      CMessage::ToLog(source,::GetLastError(),true);
      return false;
     }
   this.SetProperty(CHART_PROP_SCALE,value);
   if(redraw)
      ::ChartRedraw(this.ID());
   return true;
  }
//+------------------------------------------------------------------+
//| Set the mode of displaying the volumes on a chart                |
//+------------------------------------------------------------------+
bool CChartObj::SetModeVolume(const string source,ENUM_CHART_VOLUME_MODE mode,const bool redraw=false)
  {
   ::ResetLastError();
   if(!::ChartSetInteger(this.ID(),CHART_SHOW_VOLUMES,mode))
     {
      CMessage::ToLog(source,::GetLastError(),true);
      return false;
     }
   this.SetProperty(CHART_PROP_SHOW_VOLUMES,mode);
   if(redraw)
      ::ChartRedraw(this.ID());
   return true;
  }
//+------------------------------------------------------------------+
```

The methods are also identical to the ones considered above except for the inputs, in which the required modes are passed instead of the flags, while the limits of the passed scale value (0-5) are additionally checked in the method of setting the chart scale.

The charts feature the properties that are impossible to set. They can only be obtained. But in order for the chart object to have the same values as the chart, we need methods allowing us to enter the values of the chart parameters into the corresponding object properties. They are set when creating the chart object, but if they are changed by the terminal, then we will need to react to this change in time and make the appropriate edits to the chart object properties.

The following private methods are meant for that:

```
//+------------------------------------------------------------------+
//|  Set the property                                                |
//| "Number of bars on a chart that are available for display"       |
//+------------------------------------------------------------------+
void CChartObj::SetVisibleBars(void)
  {
   this.SetProperty(CHART_PROP_VISIBLE_BARS,::ChartGetInteger(this.ID(),CHART_VISIBLE_BARS));
  }
//+-------------------------------------------------------------------+
//|  Set the property                                                 |
//| "The total number of chart windows including indicator subwindows"|
//+-------------------------------------------------------------------+
void CChartObj::SetWindowsTotal(void)
  {
   this.SetProperty(CHART_PROP_WINDOWS_TOTAL,::ChartGetInteger(this.ID(),CHART_WINDOWS_TOTAL));
  }
//+----------------------------------------------------------------------+
//| Set the property "The number of the first visible bar on the chart"  |
//+----------------------------------------------------------------------+
void CChartObj::SetFirstVisibleBars(void)
  {
   this.SetProperty(CHART_PROP_FIRST_VISIBLE_BAR,::ChartGetInteger(this.ID(),CHART_FIRST_VISIBLE_BAR));
  }
//+------------------------------------------------------------------+
//| Set the property "Width of the chart in bars"                    |
//+------------------------------------------------------------------+
void CChartObj::SetWidthInBars(void)
  {
   this.SetProperty(CHART_PROP_WIDTH_IN_BARS,::ChartGetInteger(this.ID(),CHART_WIDTH_IN_BARS));
  }
//+------------------------------------------------------------------+
//| Set the property "Width of the chart in pixels"                  |
//+------------------------------------------------------------------+
void CChartObj::SetWidthInPixels(void)
  {
   this.SetProperty(CHART_PROP_WIDTH_IN_PIXELS,::ChartGetInteger(this.ID(),CHART_WIDTH_IN_PIXELS));
  }
//+------------------------------------------------------------------+
//| Set the property "Chart window maximized"                        |
//+------------------------------------------------------------------+
void CChartObj::SetMaximizedFlag(void)
  {
   this.SetProperty(CHART_PROP_IS_MAXIMIZED,::ChartGetInteger(this.ID(),CHART_IS_MAXIMIZED));
  }
//+------------------------------------------------------------------+
//| Set the property "Chart window minimized"                        |
//+------------------------------------------------------------------+
void CChartObj::SetMinimizedFlag(void)
  {
   this.SetProperty(CHART_PROP_IS_MINIMIZED,::ChartGetInteger(this.ID(),CHART_IS_MINIMIZED));
  }
//+------------------------------------------------------------------+
//| Set the property "Name of an EA launched on the chart"           |
//+------------------------------------------------------------------+
void CChartObj::SetExpertName(void)
  {
   this.SetProperty(CHART_PROP_EXPERT_NAME,::ChartGetString(this.ID(),CHART_EXPERT_NAME));
  }
//+------------------------------------------------------------------+
//| Set the property "Name of a script launched on the chart"        |
//+------------------------------------------------------------------+
void CChartObj::SetScriptName(void)
  {
   this.SetProperty(CHART_PROP_SCRIPT_NAME,::ChartGetString(this.ID(),CHART_SCRIPT_NAME));
  }
//+------------------------------------------------------------------+
```

The chart parameter is set to the appropriate object property here. Working with these methods will be implemented in the coming articles.

Now we need to add the methods for a simplified access to the chart object properties into the public class section. These will be the methods for enabling/disabling the chart parameters, setting the color, size and other functionality that is necessary for working with the chart.

The methods are quite numerous. So let's divide them into groups.

**The methods of returning/setting properties to Enabled/Disabled status:**

```
//--- Constructors
                     CChartObj(){;}
                     CChartObj(const long chart_id);
//+------------------------------------------------------------------+
//| Methods of simplified access to chart object properties          |
//+------------------------------------------------------------------+
//--- (1) Return, (2) enable, (3) disable drawing a price chart
   bool              IsShow(void)                                    const { return (bool)this.GetProperty(CHART_PROP_SHOW);              }
   bool              SetShowON(const bool redraw=false)                    { return this.SetShowFlag(DFUN,true,redraw);                   }
   bool              SetShowOFF(const bool redraw=false)                   { return this.SetShowFlag(DFUN,true,redraw);                   }

//--- (1) Return, (2) enable, (3) disable access to the context menu using the right click
   bool              IsAllowedContextMenu(void)                      const { return (bool)this.GetProperty(CHART_PROP_CONTEXT_MENU);      }
   bool              SetContextMenuON(const bool redraw=false)             { return this.SetContextMenuFlag(DFUN,true,redraw);            }
   bool              SetContextMenuOFF(const bool redraw=false)            { return this.SetContextMenuFlag(DFUN,false,redraw);           }

//--- (1) Return, (2) enable, (3) disable access to the Crosshair using the middle click
   bool              IsCrosshairTool(void)                           const { return (bool)this.GetProperty(CHART_PROP_CROSSHAIR_TOOL);    }
   bool              SetCrosshairToolON(const bool redraw=false);
   bool              SetCrosshairToolOFF(const bool redraw=false);

//--- (1) Return, (2) enable, (3) disable scrolling the chart horizontally using the left click
   bool              IsMouseScroll(void)                             const { return (bool)this.GetProperty(CHART_PROP_MOUSE_SCROLL);      }
   bool              SetMouseScrollON(const bool redraw=false)             { return this.SetMouseScrollFlag(DFUN,true,redraw);            }
   bool              SetMouseScrollOFF(const bool redraw=false)            { return this.SetMouseScrollFlag(DFUN,false,redraw);           }

//--- (1) Return, (2) enable, (3) disable sending messages about mouse wheel events to all MQL5 programs on a chart
   bool              IsEventMouseWhell(void)                         const { return (bool)this.GetProperty(CHART_PROP_EVENT_MOUSE_WHEEL); }
   bool              SetEventMouseWhellON(const bool redraw=false)         { return this.SetEventMouseWhellFlag(DFUN,true,redraw);        }
   bool              SetEventMouseWhellOFF(const bool redraw=false)        { return this.SetEventMouseWhellFlag(DFUN,false,redraw);       }

//--- (1) Return, (2) enable, (3) disable sending messages about mouse button movement and click events to all MQL5 programs on a chart
   bool              IsEventMouseMove(void)                          const { return (bool)this.GetProperty(CHART_PROP_EVENT_MOUSE_MOVE);  }
   bool              SetEventMouseMoveON(const bool redraw=false)          { return this.SetEventMouseMoveFlag(DFUN,true,redraw);         }
   bool              SetEventMouseMoveOFF(const bool redraw=false)         { return this.SetEventMouseMoveFlag(DFUN,false,redraw);        }

//--- (1) Return, (2) enable, (3) disable sending messages about the graphical object creation event to all MQL5 programs on a chart
   bool              IsEventObjectCreate(void)                       const { return (bool)this.GetProperty(CHART_PROP_EVENT_OBJECT_CREATE);}
   bool              SetEventObjectCreateON(const bool redraw=false)       { return this.SetEventObjectCreateFlag(DFUN,true,redraw);      }
   bool              SetEventObjectCreateOFF(const bool redraw=false)      { return this.SetEventObjectCreateFlag(DFUN,false,redraw);     }

//--- (1) Return, (2) enable, (3) disable sending messages about the graphical object destruction event to all MQL5 programs on a chart
   bool              IsEventObjectDelete(void)                       const { return (bool)this.GetProperty(CHART_PROP_EVENT_OBJECT_DELETE);}
   bool              SetEventObjectDeleteON(const bool redraw=false)       { return this.SetEventObjectDeleteFlag(DFUN,true,redraw);      }
   bool              SetEventObjectDeleteOFF(const bool redraw=false)      { return this.SetEventObjectDeleteFlag(DFUN,false,redraw);     }

//--- (1) Return, (2) enable, (3) disable price chart in the foreground
   bool              IsForeground(void)                              const { return (bool)this.GetProperty(CHART_PROP_FOREGROUND);        }
   bool              SetForegroundON(const bool redraw=false)              { return this.SetForegroundFlag(DFUN,true,redraw);             }
   bool              SetForegroundOFF(const bool redraw=false)             { return this.SetForegroundFlag(DFUN,false,redraw);            }

//--- (1) Return, (2) enable, (3) disable shift of the price chart from the right border
   bool              IsShift(void)                                   const { return (bool)this.GetProperty(CHART_PROP_SHIFT);             }
   bool              SetShiftON(const bool redraw=false)                   { return this.SetShiftFlag(DFUN,true,redraw);                  }
   bool              SetShiftOFF(const bool redraw=false)                  { return this.SetShiftFlag(DFUN,false,redraw);                 }

//--- (1) Return, (2) enable, (3) disable auto scroll to the right border of the chart
   bool              IsAutoscroll(void)                              const { return (bool)this.GetProperty(CHART_PROP_AUTOSCROLL);        }
   bool              SetAutoscrollON(const bool redraw=false)              { return this.SetAutoscrollFlag(DFUN,true,redraw);             }
   bool              SetAutoscrollOFF(const bool redraw=false)             { return this.SetAutoscrollFlag(DFUN,false,redraw);            }

//--- (1) Return, (2) enable, (3) disable the permission to manage the chart using a keyboard
   bool              IsKeyboardControl(void)                         const { return (bool)this.GetProperty(CHART_PROP_KEYBOARD_CONTROL);  }
   bool              SetKeyboardControlON(const bool redraw=false)         { return this.SetKeyboardControlFlag(DFUN,true,redraw);        }
   bool              SetKeyboardControlOFF(const bool redraw=false)        { return this.SetKeyboardControlFlag(DFUN,false,redraw);       }

//--- (1) Return, (2) enable, (3) disable the permission for the chart to intercept Space and Enter key strokes to activate the quick navigation bar
   bool              IsQuickNavigation(void)                         const { return (bool)this.GetProperty(CHART_PROP_QUICK_NAVIGATION);  }
   bool              SetQuickNavigationON(const bool redraw=false)         { return this.SetQuickNavigationFlag(DFUN,true,redraw);        }
   bool              SetQuickNavigationOFF(const bool redraw=false)        { return this.SetQuickNavigationFlag(DFUN,false,redraw);       }

//--- (1) Return, (2) enable, (3) disable a fixed scale
   bool              IsScaleFix(void)                                const { return (bool)this.GetProperty(CHART_PROP_SCALEFIX);          }
   bool              SetScaleFixON(const bool redraw=false)                { return this.SetScaleFixFlag(DFUN,true,redraw);               }
   bool              SetScaleFixOFF(const bool redraw=false)               { return this.SetScaleFixFlag(DFUN,false,redraw);              }

//--- (1) Return, (2) enable, (3) disable the 1:1 scale
   bool              IsScaleFix11(void)                              const { return (bool)this.GetProperty(CHART_PROP_SCALEFIX_11);       }
   bool              SetScaleFix11ON(const bool redraw=false)              { return this.SetScaleFix11Flag(DFUN,true,redraw);             }
   bool              SetScaleFix11OFF(const bool redraw=false)             { return this.SetScaleFix11Flag(DFUN,false,redraw);            }

//--- (1) Return, (2) enable, (3) disable the mode of specifying the chart scale in points per bar
   bool              IsScalePTPerBar(void)                           const { return (bool)this.GetProperty(CHART_PROP_SCALE_PT_PER_BAR);  }
   bool              SetScalePTPerBarON(const bool redraw=false)           { return this.SetScalePTPerBarFlag(DFUN,true,redraw);          }
   bool              SetScalePTPerBarOFF(const bool redraw=false)          { return this.SetScalePTPerBarFlag(DFUN,false,redraw);         }

//--- (1) Return, (2) enable, (3) disable a display of a symbol ticker in the upper left corner
   bool              IsShowTicker(void)                              const { return (bool)this.GetProperty(CHART_PROP_SHOW_TICKER);       }
   bool              SetShowTickerON(const bool redraw=false)              { return this.SetShowTickerFlag(DFUN,true,redraw);             }
   bool              SetShowTickerOFF(const bool redraw=false)             { return this.SetShowTickerFlag(DFUN,false,redraw);            }

//--- (1) Return, (2) enable, (3) disable a display of OHLC values in the upper left corner
   bool              IsShowOHLC(void)                                const { return (bool)this.GetProperty(CHART_PROP_SHOW_OHLC);         }
   bool              SetShowOHLCON(const bool redraw=false)                { return this.SetShowOHLCFlag(DFUN,true,redraw);               }
   bool              SetShowOHLCOFF(const bool redraw=false)               { return this.SetShowOHLCFlag(DFUN,false,redraw);              }

//--- (1) Return, (2) enable, (3) disable the display of a Bid value as a horizontal line on the chart
   bool              IsShowBidLine(void)                             const { return (bool)this.GetProperty(CHART_PROP_SHOW_BID_LINE);     }
   bool              SetShowBidLineON(const bool redraw=false)             { return this.SetShowBidLineFlag(DFUN,true,redraw);            }
   bool              SetShowBidLineOFF(const bool redraw=false)            { return this.SetShowBidLineFlag(DFUN,false,redraw);           }

//--- (1) Return, (2) enable, (3) disable the display of a Ask value as a horizontal line on the chart
   bool              IsShowAskLine(void)                             const { return (bool)this.GetProperty(CHART_PROP_SHOW_ASK_LINE);     }
   bool              SetShowAskLineON(const bool redraw=false)             { return this.SetShowAskLineFlag(DFUN,true,redraw);            }
   bool              SetShowAskLineOFF(const bool redraw=false)            { return this.SetShowAskLineFlag(DFUN,false,redraw);           }

//--- (1) Return, (2) enable, (3) disable the display of a Last value as a horizontal line on the chart
   bool              IsShowLastLine(void)                            const { return (bool)this.GetProperty(CHART_PROP_SHOW_LAST_LINE);    }
   bool              SetShowLastLineON(const bool redraw=false)            { return this.SetShowLastLineFlag(DFUN,true,redraw);           }
   bool              SetShowLastLineOFF(const bool redraw=false)           { return this.SetShowLastLineFlag(DFUN,false,redraw);          }

//--- (1) Return, (2) enable, (3) disable the display of vertical separators between adjacent periods
   bool              IsShowPeriodSeparators(void)                    const { return (bool)this.GetProperty(CHART_PROP_SHOW_PERIOD_SEP);   }
   bool              SetShowPeriodSeparatorsON(const bool redraw=false)    { return this.SetShowPeriodSeparatorsFlag(DFUN,true,redraw);   }
   bool              SetShowPeriodSeparatorsOFF(const bool redraw=false)   { return this.SetShowPeriodSeparatorsFlag(DFUN,false,redraw);  }

//--- (1) Return, (2) enable, (3) disable the display of the chart grid
   bool              IsShowGrid(void)                                const { return (bool)this.GetProperty(CHART_PROP_SHOW_GRID);         }
   bool              SetShowGridON(const bool redraw=false)                { return this.SetShowGridFlag(DFUN,true,redraw);               }
   bool              SetShowGridOFF(const bool redraw=false)               { return this.SetShowGridFlag(DFUN,false,redraw);              }

//--- (1) Return, (2) enable, (3) disable the display of text descriptions of objects
   bool              IsShowObjectDescriptions(void)   const { return (bool)this.GetProperty(CHART_PROP_SHOW_OBJECT_DESCR);                }
   bool              SetShowObjectDescriptionsON(const bool redraw=false);
   bool              SetShowObjectDescriptionsOFF(const bool redraw=false);

//--- (1) Return, (2) enable, (3) disable the display of trade levels on the chart (levels of open positions, Stop Loss, Take Profit and pending orders)
   bool              IsShowTradeLevels(void)                         const { return (bool)this.GetProperty(CHART_PROP_SHOW_TRADE_LEVELS); }
   bool              SetShowTradeLevelsON(const bool redraw=false)         { return this.SetShowTradeLevelsFlag(DFUN,true,redraw);        }
   bool              SetShowTradeLevelsOFF(const bool redraw=false)        { return this.SetShowTradeLevelsFlag(DFUN,false,redraw);       }

//--- (1) Return, (2) enable, (3) disable the ability to drag trading levels on a chart using mouse
   bool              IsDragTradeLevels(void)                         const { return (bool)this.GetProperty(CHART_PROP_DRAG_TRADE_LEVELS); }
   bool              SetDragTradeLevelsON(const bool redraw=false)         { return this.SetDragTradeLevelsFlag(DFUN,true,redraw);        }
   bool              SetDragTradeLevelsOFF(const bool redraw=false)        { return this.SetDragTradeLevelsFlag(DFUN,false,redraw);       }

//--- (1) Return, (2) enable, (3) disable the display of the time scale on the chart
   bool              IsShowDateScale(void)                           const { return (bool)this.GetProperty(CHART_PROP_SHOW_DATE_SCALE);   }
   bool              SetShowDateScaleON(const bool redraw=false)           { return this.SetShowDateScaleFlag(DFUN,true,redraw);          }
   bool              SetShowDateScaleOFF(const bool redraw=false)          { return this.SetShowDateScaleFlag(DFUN,false,redraw);         }

//--- (1) Return, (2) enable, (3) disable the display of the price scale on the chart
   bool              IsShowPriceScale(void)                          const { return (bool)this.GetProperty(CHART_PROP_SHOW_PRICE_SCALE);  }
   bool              SetShowPriceScaleON(const bool redraw=false)          { return this.SetShowPriceScaleFlag(DFUN,true,redraw);         }
   bool              SetShowPriceScaleOFF(const bool redraw=false)         { return this.SetShowPriceScaleFlag(DFUN,false,redraw);        }

//--- (1) Return, (2) enable, (3) disable the display of the quick trading panel on the chart
   bool              IsShowOneClickPanel(void)                       const { return (bool)this.GetProperty(CHART_PROP_SHOW_ONE_CLICK);    }
   bool              SetShowOneClickPanelON(const bool redraw=false)       { return this.SetShowOneClickPanelFlag(DFUN,true,redraw);      }
   bool              SetShowOneClickPanelOFF(const bool redraw=false)      { return this.SetShowOneClickPanelFlag(DFUN,false,redraw);     }

//--- (1) Return, (2) enable, (3) disable docking the chart window
   bool              IsDocked(void)                                  const { return (bool)this.GetProperty(CHART_PROP_IS_DOCKED);         }
   bool              SetDockedON(const bool redraw=false)                  { return this.SetDockedFlag(DFUN,true,redraw); }
   bool              SetDockedOFF(const bool redraw=false)                 { return this.SetDockedFlag(DFUN,false,redraw); }

//--- (1) Enable and (2) disable the display of the chart above all others
   bool              SetBringToTopON(const bool redraw=false)              { return this.SetBringToTopFlag(DFUN,true,redraw);             }
   bool              SetBringToTopOFF(const bool redraw=false)             { return this.SetBringToTopFlag(DFUN,false,redraw);            }

```

All methods of setting flags are identical and return the result of the appropriate private methods of setting flags discussed above. The methods of returning the properties return the value set in the appropriate object property. The last two methods only set flags since there is no way to return the "Above all others" chart property using the ChartGetInteger() function with the CHART\_BRING\_TO\_TOP ID since this chart property is meant only for writing the values.

**The methods of returning/setting the modes of the chart display, chart scale and chart volumes:**

```
//--- (1) Return, set the chart type (2) bars, (3) candles, (4) line
   ENUM_CHART_MODE   Mode(void)                                      const { return (ENUM_CHART_MODE)this.GetProperty(CHART_PROP_MODE);   }
   bool              SetModeBars(const bool redraw=false)                  { return this.SetMode(DFUN,CHART_BARS,redraw);                 }
   bool              SetModeCandles(const bool redraw=false)               { return this.SetMode(DFUN,CHART_CANDLES,redraw);              }
   bool              SetModeLine(const bool redraw=false)                  { return this.SetMode(DFUN,CHART_LINE,redraw);                 }

//--- (1) Return, (2 - 7) set the chart scale
   int               Scale(void)                                     const { return (int)this.GetProperty(CHART_PROP_SCALE);              }
   bool              SetScale0(const bool redraw=false)                    { return this.SetScale(DFUN,0,redraw);                         }
   bool              SetScale1(const bool redraw=false)                    { return this.SetScale(DFUN,1,redraw);                         }
   bool              SetScale2(const bool redraw=false)                    { return this.SetScale(DFUN,2,redraw);                         }
   bool              SetScale3(const bool redraw=false)                    { return this.SetScale(DFUN,3,redraw);                         }
   bool              SetScale4(const bool redraw=false)                    { return this.SetScale(DFUN,4,redraw);                         }
   bool              SetScale5(const bool redraw=false)                    { return this.SetScale(DFUN,5,redraw);                         }

//--- (1) Return, set the volume display modes (2) disabled, (3) tick volumes, (4) real volumes
   ENUM_CHART_VOLUME_MODE  ModeVolume(void)                          const { return (ENUM_CHART_VOLUME_MODE)this.GetProperty(CHART_PROP_SHOW_VOLUMES);}
   bool              SetModeVolumeHide(const bool redraw=false)            { return this.SetModeVolume(DFUN,CHART_VOLUME_HIDE,redraw);    }
   bool              SetModeVolumeTick(const bool redraw=false)            { return this.SetModeVolume(DFUN,CHART_VOLUME_TICK,redraw);    }
   bool              SetModeVolumeReal(const bool redraw=false)            { return this.SetModeVolume(DFUN,CHART_VOLUME_REAL,redraw);    }
```

The methods of returning the property values here are identical to the methods of returning flags — the value set in the appropriate object property is returned. The methods of setting values return the result of private methods with the specified value that is to be set to the object and chart property.

**The methods of returning and setting the colors of displaying various  chart elements and other editable chart parameters:**

```
//--- Return, (2) set the chart background color
   color             ColorBackground(void)                           const { return (color)this.GetProperty(CHART_PROP_COLOR_BACKGROUND); }
   bool              SetColorBackground(const color colour,const bool redraw=false);

//--- (1) Return, (2) set the color of axes, scale and OHLC line
   color             ColorForeground(void)                           const { return (color)this.GetProperty(CHART_PROP_COLOR_FOREGROUND); }
   bool              SetColorForeground(const color colour,const bool redraw=false);

//--- (1) Return and (2) set the grid color
   color             ColorGrid(void)                                 const { return (color)this.GetProperty(CHART_PROP_COLOR_GRID);       }
   bool              SetColorGrid(const color colour,const bool redraw=false);

//--- (1) Return and (2) set the volume color and position opening levels
   color             ColorVolume(void)                               const { return (color)this.GetProperty(CHART_PROP_COLOR_VOLUME);     }
   bool              SetColorVolume(const color colour,const bool redraw=false);

//--- (1) Return and (2) set the color of up bar, its shadow and border of bullish candle body
   color             ColorUp(void)                                   const { return (color)this.GetProperty(CHART_PROP_COLOR_CHART_UP);   }
   bool              SetColorUp(const color colour,const bool redraw=false);

//--- (1) Return and (2) set the color of down bar, its shadow and border of bearish candle body
   color             ColorDown(void)                                 const { return (color)this.GetProperty(CHART_PROP_COLOR_CHART_DOWN); }
   bool              SetColorDown(const color colour,const bool redraw=false);

//--- (1) Return and (2) set the color of the chart line and Doji candles
   color             ColorLine(void)                                 const { return (color)this.GetProperty(CHART_PROP_COLOR_CHART_LINE); }
   bool              SetColorLine(const color colour,const bool redraw=false);

//--- (1) Return and (2) set the color of the bullish candle body
   color             ColorCandleBull(void)                           const { return (color)this.GetProperty(CHART_PROP_COLOR_CANDLE_BULL);}
   bool              SetColorCandleBull(const color colour,const bool redraw=false);

//--- (1) Return and (2) set the color of the bearish candle body
   color             ColorCandleBear(void)                           const { return (color)this.GetProperty(CHART_PROP_COLOR_CANDLE_BEAR);}
   bool              SetColorCandleBear(const color colour,const bool redraw=false);

//--- (1) Return and (2) set the Bid price line color
   color             ColorBid(void)                                  const { return (color)this.GetProperty(CHART_PROP_COLOR_BID);        }
   bool              SetColorBid(const color colour,const bool redraw=false);

//--- (1) Return and (2) set the Ask price line color
   color             ColorAsk(void)                                  const { return (color)this.GetProperty(CHART_PROP_COLOR_ASK);        }
   bool              SetColorAsk(const color colour,const bool redraw=false);

//--- (1) Return and (2) set the color of the price line of the last performed deal (Last)
   color             ColorLast(void)                                 const { return (color)this.GetProperty(CHART_PROP_COLOR_LAST);       }
   bool              SetColorLast(const color colour,const bool redraw=false);

//--- (1) Return and (2) set the color of stop order levels (Stop Loss and Take Profit)
   color             ColorStops(void)                                const { return (color)this.GetProperty(CHART_PROP_COLOR_STOP_LEVEL); }
   bool              SetColorStops(const color colour,const bool redraw=false);

//--- (1) Return and (2) set the left coordinate of the undocked chart window relative to the virtual screen
   int               FloatLeft(void)                                 const { return (int)this.GetProperty(CHART_PROP_FLOAT_LEFT);         }
   bool              SetFloatLeft(const int value,const bool redraw=false);

//--- (1) Return and (2) set the top coordinate of the undocked chart window relative to the virtual screen
   int               FloatTop(void)                                  const { return (int)this.GetProperty(CHART_PROP_FLOAT_TOP);          }
   bool              SetFloatTop(const int value,const bool redraw=false);

//--- (1) Return and (2) set the right coordinate of the undocked chart window relative to the virtual screen
   int               FloatRight(void)                                const { return (int)this.GetProperty(CHART_PROP_FLOAT_RIGHT);        }
   bool              SetFloatRight(const int value,const bool redraw=false);

//--- (1) Return and (2) set the bottom coordinate of the undocked chart window relative to the virtual screen
   int               FloatBottom(void)                               const { return (int)this.GetProperty(CHART_PROP_FLOAT_BOTTOM);       }
   bool              SetFloatBottom(const int value,const bool redraw=false);

//--- (1) Return and (2) set the shift size of the zero bar from the right border in %
   double            ShiftSize(void)                                 const { return this.GetProperty(CHART_PROP_SHIFT_SIZE);              }
   bool              SetShiftSize(const double value,const bool redraw=false);

//--- (1) Return and (2) set the chart fixed position from the left border in %
   double            FixedPosition(void)                             const { return this.GetProperty(CHART_PROP_FIXED_POSITION);          }
   bool              SetFixedPosition(const double value,const bool redraw=false);

//--- (1) Return and (2) set the fixed chart maximum
   double            FixedMaximum(void)                              const { return this.GetProperty(CHART_PROP_FIXED_MAX);               }
   bool              SetFixedMaximum(const double value,const bool redraw=false);

//--- (1) Return and (2) set the fixed chart minimum
   double            FixedMinimum(void)                              const { return this.GetProperty(CHART_PROP_FIXED_MIN);               }
   bool              SetFixedMinimum(const double value,const bool redraw=false);

//--- (1) Return and (2) set the value of the scale in points per bar
   double            PointsPerBar(void)                              const { return this.GetProperty(CHART_PROP_POINTS_PER_BAR);          }
   bool              SetPointsPerBar(const double value,const bool redraw=false);

//--- (1) Return and (2) set the comment on the chart
   string            Comment(void)                                   const { return this.GetProperty(CHART_PROP_COMMENT);                 }
   bool              SetComment(const string comment,const bool redraw=false);

//--- (1) Return and (2) set the chart symbol
   string            Symbol(void)                                    const { return this.GetProperty(CHART_PROP_SYMBOL);                  }
   bool              SetSymbol(const string symbol);

//--- (1) Return and (2) set the chart period
   ENUM_TIMEFRAMES   Timeframe(void)                                 const { return (ENUM_TIMEFRAMES)this.GetProperty(CHART_PROP_TIMEFRAME); }
   bool              SetTimeframe(const ENUM_TIMEFRAMES timeframe);
```

The methods of returning the properties return the values set in the appropriate object property. The setting methods are only declared here. Their implementation will be considered later.

**The methods returning the chart object properties corresponding to the read-only chart parameters:**

```
//--- (1) Return the Chart object identification attribute
   bool              IsObject(void)                                  const { return (bool)this.GetProperty(CHART_PROP_IS_OBJECT);         }

//--- Return the chart ID
   long              ID(void)                                        const { return this.GetProperty(CHART_PROP_ID);                      }

//--- Return the number of bars on a chart that are available for display
   int               VisibleBars(void)                               const { return (int)this.GetProperty(CHART_PROP_VISIBLE_BARS);       }

//--- Return the total number of chart windows including indicator subwindows
   int               WindowsTotal(void)                              const { return (int)this.GetProperty(CHART_PROP_WINDOWS_TOTAL);      }

//--- Return the chart window handle
   int               Handle(void)                                    const { return (int)this.GetProperty(CHART_PROP_WINDOW_HANDLE);      }

//--- Return the number of the first visible bar on the chart
   int               FirstVisibleBars(void)                          const { return (int)this.GetProperty(CHART_PROP_FIRST_VISIBLE_BAR);  }

//--- Return the chart width in bars
   int               WidthInBars(void)                               const { return (int)this.GetProperty(CHART_PROP_WIDTH_IN_BARS);      }

//--- Return the chart width in pixels
   int               WidthInPixels(void)                             const { return (int)this.GetProperty(CHART_PROP_WIDTH_IN_PIXELS);    }

//--- Return the "Chart window maximized" property
   bool              IsMaximized(void)                               const { return (bool)this.GetProperty(CHART_PROP_IS_MAXIMIZED);      }

//--- Return the "Chart window minimized" property
   bool              IsMinimized(void)                               const { return (bool)this.GetProperty(CHART_PROP_IS_MINIMIZED);      }

//--- Return the name of an EA launched on the chart
   string            ExpertName(void)                                const { return this.GetProperty(CHART_PROP_EXPERT_NAME);             }

//--- Return the name of a script launched on the chart
   string            ScriptName(void)                                const { return this.GetProperty(CHART_PROP_SCRIPT_NAME);             }
```

The methods return the value set to the appropriate chart object property.

**The methods returning/setting chart parameter values the subwindow index should be specified for:**

```
//--- (1) Return and (2) set the height of the specified chart in pixels
   int               WindowHeightInPixels(const int sub_window)      const { return (int)::ChartGetInteger(this.ID(),CHART_HEIGHT_IN_PIXELS,sub_window);       }
   bool              SetWindowHeightInPixels(const int height,const int sub_window,const bool redraw=false);

//--- Return the distance in Y axis pixels between the upper frame of the indicator subwindow and the upper frame of the chart main window
   int               WindowYDistance(const int sub_window)           const { return (int)::ChartGetInteger(this.ID(),CHART_WINDOW_YDISTANCE,sub_window);       }

//--- Return the specified subwindow visibility
   bool              IsVisibleWindow(const int sub_window)           const { return (bool)::ChartGetInteger(this.Handle(),CHART_WINDOW_IS_VISIBLE,sub_window); }

//--- Return the minimum of the specified chart
   double            PriceMinimum(const int sub_window)              const { return ::ChartGetDouble(this.ID(),CHART_PRICE_MIN,sub_window);  }

//--- Return the maximum of the specified chart
   double            PriceMaximum(const int sub_window)              const { return ::ChartGetDouble(this.ID(),CHART_PRICE_MAX,sub_window);  }
```

These methods are set here temporarily. Later, the chart object will have the list of indicator subwindows belonging to the chart. The numbers of these subwindows should be specified in the functions of setting/returning the values, so the methods are to be revised. Therefore, the values are returned directly from the chart described by the chart object, rather than from the object properties.

**The method for tick emulation:**

```
//--- Emulate a tick (chart updates - similar to the terminal Refresh command)
   void              EmulateTick(void)                                     { ::ChartSetSymbolPeriod(this.ID(),this.Symbol(),this.Timeframe());}

  };
//+------------------------------------------------------------------+
//| Parametric constructor                                           |
//+------------------------------------------------------------------+
```

The call of the [ChartSetSymbolPeriod()](https://www.mql5.com/en/docs/chart_operations/chartsetsymbolperiod) function while specifying the symbol and timeframe similar to that of the current chart can be used for updating the chart (similar to the Refresh command in the terminal). In its turn, the chart update triggers re-calculation of the indicators attached to it. Thus, it is possible to calculate an indicator on the chart even if there are no ticks (e.g., on weekends).

Implement declared methods in the block of methods for a simplified access to the object properties outside the class body.

**The methods for setting the chart element color and other customized chart parameters:**

```
//+------------------------------------------------------------------+
//| Set the chart background color                                   |
//+------------------------------------------------------------------+
bool CChartObj::SetColorBackground(const color colour,const bool redraw=false)
  {
   ::ResetLastError();
   if(!::ChartSetInteger(this.ID(),CHART_COLOR_BACKGROUND,colour))
     {
      CMessage::ToLog(DFUN,::GetLastError(),true);
      return false;
     }
   this.SetProperty(CHART_PROP_COLOR_BACKGROUND,colour);
   if(redraw)
      ::ChartRedraw(this.ID());
   return true;
  }
//+------------------------------------------------------------------+
//| Set the color of axes, scale and OHLC line                       |
//+------------------------------------------------------------------+
bool CChartObj::SetColorForeground(const color colour,const bool redraw=false)
  {
   ::ResetLastError();
   if(!::ChartSetInteger(this.ID(),CHART_COLOR_FOREGROUND,colour))
     {
      CMessage::ToLog(DFUN,::GetLastError(),true);
      return false;
     }
   this.SetProperty(CHART_PROP_COLOR_FOREGROUND,colour);
   if(redraw)
      ::ChartRedraw(this.ID());
   return true;
  }
//+------------------------------------------------------------------+
//| Set the grid color                                               |
//+------------------------------------------------------------------+
bool CChartObj::SetColorGrid(const color colour,const bool redraw=false)
  {
   ::ResetLastError();
   if(!::ChartSetInteger(this.ID(),CHART_COLOR_GRID,colour))
     {
      CMessage::ToLog(DFUN,::GetLastError(),true);
      return false;
     }
   this.SetProperty(CHART_PROP_COLOR_GRID,colour);
   if(redraw)
      ::ChartRedraw(this.ID());
   return true;
  }
//+------------------------------------------------------------------+
//| Set the volume color and position opening levels                 |
//+------------------------------------------------------------------+
bool CChartObj::SetColorVolume(const color colour,const bool redraw=false)
  {
   ::ResetLastError();
   if(!::ChartSetInteger(this.ID(),CHART_COLOR_VOLUME,colour))
     {
      CMessage::ToLog(DFUN,::GetLastError(),true);
      return false;
     }
   this.SetProperty(CHART_PROP_COLOR_VOLUME,colour);
   if(redraw)
      ::ChartRedraw(this.ID());
   return true;
  }
//+----------------------------------------------------------------------+
//|Set the color of up bar, its shadow and border of bullish candle body |
//+----------------------------------------------------------------------+
bool CChartObj::SetColorUp(const color colour,const bool redraw=false)
  {
   ::ResetLastError();
   if(!::ChartSetInteger(this.ID(),CHART_COLOR_CHART_UP,colour))
     {
      CMessage::ToLog(DFUN,::GetLastError(),true);
      return false;
     }
   this.SetProperty(CHART_PROP_COLOR_CHART_UP,colour);
   if(redraw)
      ::ChartRedraw(this.ID());
   return true;
  }
//+-----------------------------------------------------------------------+
//|Set the color of down bar, its shadow and border of bearish candle body|
//+-----------------------------------------------------------------------+
bool CChartObj::SetColorDown(const color colour,const bool redraw=false)
  {
   ::ResetLastError();
   if(!::ChartSetInteger(this.ID(),CHART_COLOR_CHART_DOWN,colour))
     {
      CMessage::ToLog(DFUN,::GetLastError(),true);
      return false;
     }
   this.SetProperty(CHART_PROP_COLOR_CHART_DOWN,colour);
   if(redraw)
      ::ChartRedraw(this.ID());
   return true;
  }
//+------------------------------------------------------------------+
//| Set the color of the chart line and Doji candles                 |
//+------------------------------------------------------------------+
bool CChartObj::SetColorLine(const color colour,const bool redraw=false)
  {
   ::ResetLastError();
   if(!::ChartSetInteger(this.ID(),CHART_COLOR_CHART_LINE,colour))
     {
      CMessage::ToLog(DFUN,::GetLastError(),true);
      return false;
     }
   this.SetProperty(CHART_PROP_COLOR_CHART_LINE,colour);
   if(redraw)
      ::ChartRedraw(this.ID());
   return true;
  }
//+------------------------------------------------------------------+
//| Set the color of bullish candle body                             |
//+------------------------------------------------------------------+
bool CChartObj::SetColorCandleBull(const color colour,const bool redraw=false)
  {
   ::ResetLastError();
   if(!::ChartSetInteger(this.ID(),CHART_COLOR_CANDLE_BULL,colour))
     {
      CMessage::ToLog(DFUN,::GetLastError(),true);
      return false;
     }
   this.SetProperty(CHART_PROP_COLOR_CANDLE_BULL,colour);
   if(redraw)
      ::ChartRedraw(this.ID());
   return true;
  }
//+------------------------------------------------------------------+
//| Set the color of bearish candle body                             |
//+------------------------------------------------------------------+
bool CChartObj::SetColorCandleBear(const color colour,const bool redraw=false)
  {
   ::ResetLastError();
   if(!::ChartSetInteger(this.ID(),CHART_COLOR_CANDLE_BEAR,colour))
     {
      CMessage::ToLog(DFUN,::GetLastError(),true);
      return false;
     }
   this.SetProperty(CHART_PROP_COLOR_CANDLE_BEAR,colour);
   if(redraw)
      ::ChartRedraw(this.ID());
   return true;
  }
//+------------------------------------------------------------------+
//| Set the Bid price line color                                     |
//+------------------------------------------------------------------+
bool CChartObj::SetColorBid(const color colour,const bool redraw=false)
  {
   ::ResetLastError();
   if(!::ChartSetInteger(this.ID(),CHART_COLOR_BID,colour))
     {
      CMessage::ToLog(DFUN,::GetLastError(),true);
      return false;
     }
   this.SetProperty(CHART_PROP_COLOR_BID,colour);
   if(redraw)
      ::ChartRedraw(this.ID());
   return true;
  }
//+------------------------------------------------------------------+
//| Set the Ask price line color                                     |
//+------------------------------------------------------------------+
bool CChartObj::SetColorAsk(const color colour,const bool redraw=false)
  {
   ::ResetLastError();
   if(!::ChartSetInteger(this.ID(),CHART_COLOR_ASK,colour))
     {
      CMessage::ToLog(DFUN,::GetLastError(),true);
      return false;
     }
   this.SetProperty(CHART_PROP_COLOR_ASK,colour);
   if(redraw)
      ::ChartRedraw(this.ID());
   return true;
  }
//+------------------------------------------------------------------+
//|Set the color of the price line of the last performed deal (Last) |
//+------------------------------------------------------------------+
bool CChartObj::SetColorLast(const color colour,const bool redraw=false)
  {
   ::ResetLastError();
   if(!::ChartSetInteger(this.ID(),CHART_COLOR_LAST,colour))
     {
      CMessage::ToLog(DFUN,::GetLastError(),true);
      return false;
     }
   this.SetProperty(CHART_PROP_COLOR_LAST,colour);
   if(redraw)
      ::ChartRedraw(this.ID());
   return true;
  }
//+------------------------------------------------------------------+
//|Set the color of stop order levels (Stop Loss and Take Profit)    |
//+------------------------------------------------------------------+
bool CChartObj::SetColorStops(const color colour,const bool redraw=false)
  {
   ::ResetLastError();
   if(!::ChartSetInteger(this.ID(),CHART_COLOR_STOP_LEVEL,colour))
     {
      CMessage::ToLog(DFUN,::GetLastError(),true);
      return false;
     }
   this.SetProperty(CHART_PROP_COLOR_STOP_LEVEL,colour);
   if(redraw)
      ::ChartRedraw(this.ID());
   return true;
  }
//+------------------------------------------------------------------+
//| Set the left coordinate of the undocked chart                    |
//| relative to the virtual screen                                   |
//+------------------------------------------------------------------+
bool CChartObj::SetFloatLeft(const int value,const bool redraw=false)
  {
   ::ResetLastError();
   if(!::ChartSetInteger(this.ID(),CHART_FLOAT_LEFT,value))
     {
      CMessage::ToLog(DFUN,::GetLastError(),true);
      return false;
     }
   this.SetProperty(CHART_PROP_FLOAT_LEFT,value);
   if(redraw)
      ::ChartRedraw(this.ID());
   return true;
  }
//+------------------------------------------------------------------+
//| Set the top coordinate of the undocked chart                     |
//| relative to the virtual screen                                   |
//+------------------------------------------------------------------+
bool CChartObj::SetFloatTop(const int value,const bool redraw=false)
  {
   ::ResetLastError();
   if(!::ChartSetInteger(this.ID(),CHART_FLOAT_TOP,value))
     {
      CMessage::ToLog(DFUN,::GetLastError(),true);
      return false;
     }
   this.SetProperty(CHART_PROP_FLOAT_TOP,value);
   if(redraw)
      ::ChartRedraw(this.ID());
   return true;
  }
//+------------------------------------------------------------------+
//| Set the right coordinate of the undocked chart                   |
//| relative to the virtual screen                                   |
//+------------------------------------------------------------------+
bool CChartObj::SetFloatRight(const int value,const bool redraw=false)
  {
   ::ResetLastError();
   if(!::ChartSetInteger(this.ID(),CHART_FLOAT_RIGHT,value))
     {
      CMessage::ToLog(DFUN,::GetLastError(),true);
      return false;
     }
   this.SetProperty(CHART_PROP_FLOAT_RIGHT,value);
   if(redraw)
      ::ChartRedraw(this.ID());
   return true;
  }
//+------------------------------------------------------------------+
//| Set the bottom coordinate of the undocked chart                  |
//| relative to the virtual screen                                   |
//+------------------------------------------------------------------+
bool CChartObj::SetFloatBottom(const int value,const bool redraw=false)
  {
   ::ResetLastError();
   if(!::ChartSetInteger(this.ID(),CHART_FLOAT_BOTTOM,value))
     {
      CMessage::ToLog(DFUN,::GetLastError(),true);
      return false;
     }
   this.SetProperty(CHART_PROP_FLOAT_BOTTOM,value);
   if(redraw)
      ::ChartRedraw(this.ID());
   return true;
  }
//+------------------------------------------------------------------+
//| Set the value of zeroth bar shift                                |
//| from the right edge in %                                         |
//+------------------------------------------------------------------+
bool CChartObj::SetShiftSize(const double value,const bool redraw=false)
  {
   double size=(value<10.0 ? 10.0 : value>50.0 ? 50.0 : value);
   ::ResetLastError();
   if(!::ChartSetDouble(this.ID(),CHART_SHIFT_SIZE,size))
     {
      CMessage::ToLog(DFUN,::GetLastError(),true);
      return false;
     }
   this.SetProperty(CHART_PROP_SHIFT_SIZE,size);
   if(redraw)
      ::ChartRedraw(this.ID());
   return true;
  }
//+------------------------------------------------------------------+
//| Set the location of the chart fixed position                     |
//| from the left edge in %                                          |
//+------------------------------------------------------------------+
bool CChartObj::SetFixedPosition(const double value,const bool redraw=false)
  {
   double pos=(value<0 ? 0 : value>100.0 ? 100.0 : value);
   ::ResetLastError();
   if(!::ChartSetDouble(this.ID(),CHART_FIXED_POSITION,pos))
     {
      CMessage::ToLog(DFUN,::GetLastError(),true);
      return false;
     }
   this.SetProperty(CHART_PROP_FIXED_POSITION,pos);
   if(redraw)
      ::ChartRedraw(this.ID());
   return true;
  }
//+------------------------------------------------------------------+
//| Set the fixed chart maximum                                      |
//+------------------------------------------------------------------+
bool CChartObj::SetFixedMaximum(const double value,const bool redraw=false)
  {
   ::ResetLastError();
   if(!::ChartSetDouble(this.ID(),CHART_FIXED_MAX,value))
     {
      CMessage::ToLog(DFUN,::GetLastError(),true);
      return false;
     }
   this.SetProperty(CHART_PROP_FIXED_MAX,value);
   if(redraw)
      ::ChartRedraw(this.ID());
   return true;
  }
//+------------------------------------------------------------------+
//| Set the fixed chart minimum                                      |
//+------------------------------------------------------------------+
bool CChartObj::SetFixedMinimum(const double value,const bool redraw=false)
  {
   ::ResetLastError();
   if(!::ChartSetDouble(this.ID(),CHART_FIXED_MIN,value))
     {
      CMessage::ToLog(DFUN,::GetLastError(),true);
      return false;
     }
   this.SetProperty(CHART_PROP_FIXED_MIN,value);
   if(redraw)
      ::ChartRedraw(this.ID());
   return true;
  }
//+------------------------------------------------------------------+
//| Set the value of the scale in points per bar                     |
//+------------------------------------------------------------------+
bool CChartObj::SetPointsPerBar(const double value,const bool redraw=false)
  {
   ::ResetLastError();
   if(!::ChartSetDouble(this.ID(),CHART_POINTS_PER_BAR,value))
     {
      CMessage::ToLog(DFUN,::GetLastError(),true);
      return false;
     }
   this.SetProperty(CHART_PROP_POINTS_PER_BAR,value);
   if(redraw)
      ::ChartRedraw(this.ID());
   return true;
  }
//+------------------------------------------------------------------+
//| Set the comment on the chart                                     |
//+------------------------------------------------------------------+
bool CChartObj::SetComment(const string comment,const bool redraw=false)
  {
   ::ResetLastError();
   if(!::ChartSetString(this.ID(),CHART_COMMENT,comment))
     {
      CMessage::ToLog(DFUN,::GetLastError(),true);
      return false;
     }
   this.SetProperty(CHART_PROP_COMMENT,comment);
   if(redraw)
      ::ChartRedraw(this.ID());
   return true;
  }
//+------------------------------------------------------------------+
//| Set the height of the specified chart in pixels                  |
//+------------------------------------------------------------------+
bool CChartObj::SetWindowHeightInPixels(const int height,const int sub_window,const bool redraw=false)
  {
   ::ResetLastError();
   if(!::ChartSetInteger(this.ID(),CHART_HEIGHT_IN_PIXELS,sub_window,height))
     {
      CMessage::ToLog(DFUN,::GetLastError(),true);
      return false;
     }
   if(redraw)
      ::ChartRedraw(this.ID());
   return true;
  }
//+------------------------------------------------------------------+
//| Set the chart symbol                                             |
//+------------------------------------------------------------------+
bool CChartObj::SetSymbol(const string symbol)
  {
   ::ResetLastError();
   if(!::ChartSetSymbolPeriod(this.ID(),symbol,this.Timeframe()))
     {
      CMessage::ToLog(DFUN,::GetLastError(),true);
      return false;
     }
   this.SetProperty(CHART_PROP_SYMBOL,symbol);
   this.m_digits=(int)::SymbolInfoInteger(this.Symbol(),SYMBOL_DIGITS);
   return true;
  }
//+------------------------------------------------------------------+
//| Set the chart period                                             |
//+------------------------------------------------------------------+
bool CChartObj::SetTimeframe(const ENUM_TIMEFRAMES timeframe)
  {
   ::ResetLastError();
   if(!::ChartSetSymbolPeriod(this.ID(),this.Symbol(),timeframe))
     {
      CMessage::ToLog(DFUN,::GetLastError(),true);
      return false;
     }
   this.SetProperty(CHART_PROP_TIMEFRAME,timeframe);
   return true;
  }
//+------------------------------------------------------------------+
```

The methods are identical to the private methods considered above and have the same logic, so I will leave them for you to study. If you have any questions, feel free to ask them in the comments below.

This completes the creation of the chart object. Find its full listing in the library files attached below.

### Test

To perform the test, I will use the [EA from the previous article](https://www.mql5.com/en/articles/9146#node04) and save it in the new folder \\MQL5\\Experts\\TestDoEasy\ **Part67\** as **TestDoEasyPart67.mq5**.

Open three charts of different instruments. The EA is to work on the first one, while two others will simply remain open. During the first launch, the EA reads all charts, creates the appropriate chart objects, adds them to the temporarily created list and reads the created list displaying the short descriptions of chart objects from it. Display the full chart object description for the very first chart the EA works on.

Since the class signals are now included using the main object of the CEngine library, instead of the string for including the MQL5 signal object class

```
//|                                             TestDoEasyPart66.mq5 |
//|                        Copyright 2021, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2021, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//--- includes
#include <DoEasy\Engine.mqh>
#include <DoEasy\Objects\MQLSignalBase\MQLSignal.mqh>
//--- enums
```

include the chart object class file:

```
//|                                             TestDoEasyPart67.mq5 |
//|                        Copyright 2021, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2021, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//--- includes
#include <DoEasy\Engine.mqh>
#include <DoEasy\Objects\Chart\ChartObj.mqh>
//--- enums
```

In the OnTick() handler, instead of the code block for working with signals

```
//--- Search for available signals in the database and check the ability to subscribe to a signal by its name
   static bool done=false;
   //--- If the first launch and working with signals is enabled in EA custom settings
   if(InpUseMqlSignals && !done)
     {
      //--- Display the list of all free signals in the journal
      Print("");
      engine.GetSignalsMQL5Collection().PrintShort(true,false,true);
      //--- Get the list of free signals
      CArrayObj *list=engine.GetListSignalsMQL5Free();
      //--- If the list is obtained
      if(list!=NULL)
        {
         //--- Find a signal with the maximum growth in % in the list
         int index_max_gain=CSelect::FindMQLSignalMax(list,SIGNAL_MQL5_PROP_GAIN);
         CMQLSignal *signal_max_gain=list.At(index_max_gain);
         //--- If the signal is found
         if(signal_max_gain!=NULL)
           {
            //--- Display the full signal description in the journal
            signal_max_gain.Print();
            //--- If managed to subscribe to a signal
            if(engine.SignalsMQL5Subscribe(signal_max_gain.ID()))
              {
               //--- Set subscription parameters
               //--- Enable copying deals by subscription
               engine.SignalsMQL5CurrentSetSubscriptionEnableON();
               //--- Set synchronization without the confirmation dialog
               engine.SignalsMQL5CurrentSetConfirmationsDisableOFF();
               //--- Set copying Stop Loss and Take Profit
               engine.SignalsMQL5CurrentSetSLTPCopyON();
               //--- Set the market order slippage used when synchronizing positions and copying deals
               engine.SignalsMQL5CurrentSetSlippage(2);
               //--- Set the percentage for converting deal volume
               engine.SignalsMQL5CurrentSetEquityLimit(50);
               //--- Set deposit limitations (in %)
               engine.SignalsMQL5CurrentSetDepositPercent(70);
               //--- Display subscription parameters in the journal
               engine.SignalsMQL5CurrentSubscriptionParameters();
              }
           }
        }
      done=true;
      return;
     }
   //--- If a signal subscription is active,  unsubscribe
   if(engine.SignalsMQL5CurrentID()>0)
     {
      engine.SignalsMQL5Unsubscribe();
     }
//---
```

add the code block for working with chart objects:

```
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//--- Handle the NewTick event in the library
   engine.OnTick(rates_data);

//--- If working in the tester
   if(MQLInfoInteger(MQL_TESTER))
     {
      engine.OnTimer(rates_data);   // Working in the timer
      PressButtonsControl();        // Button pressing control
      engine.EventsHandling();      // Working with events
     }

//--- If the trailing flag is set
   if(trailing_on)
     {
      TrailingPositions();          // Trailing positions
      TrailingOrders();             // Trailing pending orders
     }

//--- If it is the first launch
   static bool done=false;
   if(!done)
     {
      //--- Create the list object for storing chart objects
      CArrayObj *list=new CArrayObj();
      if(list==NULL)
         return;
      //--- Declare the variables and get the first chart ID
      long currChart,prevChart=ChartFirst();
      int i=0;
      //--- Create the chart object and add it to the list
      CChartObj *chart_first=new CChartObj(prevChart);
      list.Add(chart_first);
      //--- In the loop by the total number of terminal charts (not more than 100)
      while(i<CHARTS_MAX)
        {
         //--- based on the previous one, get the new chart
         currChart=ChartNext(prevChart);
         //--- When reaching the end of the chart list, complete the loop
         if(currChart<0) break;
         //--- Create the chart object based on the current chart ID in the loop and add it to the list
         CChartObj *chart=new CChartObj(currChart);
         list.Add(chart);
         //--- remember the current chart ID for ChartNext() and increase the loop counter
         prevChart=currChart;
         i++;
        }
      Print("");
      //--- From the filled list in the loop, receive the next chart object and display its short description
      int total=list.Total();
      for(int j=0;j<total;j++)
        {
         CChartObj *chart_obj=list.At(j);
         chart_obj.PrintShort();
        }
      Print("");
      //--- Display the full description of the very first chart
      chart_first=list.At(0);
      chart_first.Print();
      //--- Destroy the list of chart objects
      delete list;
      done=true;
     }
//---
  }
//+------------------------------------------------------------------+
```

The entire logic is described in the code and requires no additional explanations. If you have any questions, feel free to ask them in the comments below.

This is all I wanted to do in the current article.

Compile the EA, open three charts in the terminal and launch the EA on the first of them, while preliminarily specifying "Work only with the current symbol and timeframe" in the parameters:

![](https://c.mql5.com/2/42/6cH1LapyMZ.png)

During the first tick, the EA creates three chart objects and displays short descriptions of the three created chart objects in addition to the messages about the initialization of various library classes:

```
Chart window EURUSD H4 ID: 131733844391938630, HWND: 918600
Chart window AUDUSD H1 ID: 131733844391938632, HWND: 1182638
Chart window GBPUSD H4 ID: 131733844391938633, HWND: 1705036
```

Next, display the full description of all properties of the chart object describing the first terminal chart:

```
============= The beginning of the parameter list (Chart window EURUSD H4) =============
Chart ID: 131733844391938630
Timeframe: H4
Drawing attributes of a price chart: Yes
Object "Chart": No
Chart on top of other charts: No
Accessing the context menu by pressing the right mouse button: Yes
Accessing the "Crosshair tool" by pressing the middle mouse button: Yes
Scrolling the chart horizontally using the left mouse button: Yes
Sending messages about mouse wheel events to all mql5 programs on a chart: No
Send notifications of mouse move and mouse click events to all mql5 programs on a chart: No
Send a notification of an event of new object creation to all mql5-programs on a chart: No
Send a notification of an event of object deletion to all mql5-programs on a chart: No
Chart type: Display as Japanese candlesticks
Price chart in the foreground: No
Price chart indent from the right border: Yes
Automatic moving to the right border of the chart: Yes
Managing the chart using a keyboard: Yes
Allowed to intercept Space and Enter key presses on the chart to activate the quick navigation bar: Yes
Scale: 2
Fixed scale mode: No
Scale 1:1 mode: No
Scale to be specified in points per bar: No
Display a symbol ticker in the upper left corner: Yes
Display OHLC values in the upper left corner: Yes
Display Bid values as a horizontal line in a chart: Yes
Display Ask values as a horizontal line in a chart: Yes
Display Last values as a horizontal line in a chart: No
Display vertical separators between adjacent periods: No
Display grid in the chart: No
Display volume in the chart: Tick volumes
Display textual descriptions of objects: Yes
The number of bars on the chart that can be displayed: 96
The total number of chart windows, including indicator subwindows: 1
Chart window handle: 918600
Number of the first visible bar in the chart: 95
Chart width in bars: 117
Chart width in pixels: 466
Chart background color: clrWhite
Color of axes, scales and OHLC line: clrBlack
Grid color: clrSilver
Color of volumes and position opening levels: clrGreen
Color for the up bar, shadows and body borders of bull candlesticks: clrBlack
Color for the down bar, shadows and body borders of bear candlesticks: clrBlack
Line chart color and color of "Doji" Japanese candlesticks: clrBlack
Body color of a bull candlestick: clrWhite
Body color of a bear candlestick: clrBlack
Bid price level color: clrLightSkyBlue
Ask price level color: clrCoral
Line color of the last executed deal price (Last): clrSilver
Color of stop order levels (Stop Loss and Take Profit): clrOrangeRed
Displaying trade levels in the chart (levels of open positions, Stop Loss, Take Profit and pending orders): Yes
Permission to drag trading levels on a chart with a mouse: Yes
Showing the time scale on a chart: Yes
Showing the price scale on a chart: Yes
Showing the "One click trading" panel on a chart: No
Chart window is maximized: No
Chart window is minimized: No
The chart window is docked: Yes
The left coordinate of the undocked chart window relative to the virtual screen: 0
The top coordinate of the undocked chart window relative to the virtual screen: 0
The right coordinate of the undocked chart window relative to the virtual screen: 0
The bottom coordinate of the undocked chart window relative to the virtual screen: 0
------
The size of the zero bar indent from the right border in percents: 18.63
Chart fixed position from the left border in percent value: 0.00
Fixed  chart maximum: 1.22650
Fixed  chart minimum : 1.17770
Scale in points per bar: 1.00
Chart minimum: 1.17770
Chart maximum: 1.22650
------
Text of a comment in a chart: ""
The name of the Expert Advisor running on the chart: "TestDoEasyPart67"
The name of the script running on the chart: ""
Symbol: "EURUSD"
============= End of the parameter list (Chart window EURUSD H4) =============
```

### What's next?

In the next article, I will expand the features of the chart object by creating subwindow objects for it and create the chart object collection.

All files of the current version of the library are attached below together with the test EA file for MQL5 for you to test and download.

I do not recommend using chart objects in your work in their current state since they are to be changed further.

Leave your questions and suggestions in the comments.

[Back to contents](https://www.mql5.com/en/articles/9213#node00)

**\*Previous articles within the series:**

[Prices in DoEasy library (Part 62): Updating tick series in real time, preparation for working with Depth of Market](https://www.mql5.com/en/articles/8988)

[Prices in DoEasy library (Part 63): Depth of Market and its abstract request class](https://www.mql5.com/en/articles/9010)

[Prices in DoEasy library (Part 64): Depth of Market, classes of DOM snapshot and snapshot series objects](https://www.mql5.com/en/articles/9044)

[Prices and Signals in DoEasy library (Part 65): Depth of Market collection and the class for working with MQL5.com Signals](https://www.mql5.com/en/articles/9095)

[Other classes in DoEasy library (Part 66): MQL5.com Signals collection class](https://www.mql5.com/en/articles/9146)

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/9213](https://www.mql5.com/ru/articles/9213)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/9213.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/9213/mql5.zip "Download MQL5.zip")(3940.41 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/369066)**
(1)


![Jessie Gold Trader](https://c.mql5.com/avatar/2022/9/631790f0-7023.jpg)

**[Jessie Gold Trader](https://www.mql5.com/en/users/jessicajessica)**
\|
5 Jun 2021 at 16:22

very important information!

![Neural networks made easy (Part 13): Batch Normalization](https://c.mql5.com/2/48/Neural_networks_made_easy_013.png)[Neural networks made easy (Part 13): Batch Normalization](https://www.mql5.com/en/articles/9207)

In the previous article, we started considering methods aimed at improving neural network training quality. In this article, we will continue this topic and will consider another approach — batch data normalization.

![Brute force approach to pattern search (Part IV): Minimal functionality](https://c.mql5.com/2/41/1560775468.png)[Brute force approach to pattern search (Part IV): Minimal functionality](https://www.mql5.com/en/articles/8845)

The article presents an improved brute force version, based on the goals set in the previous article. I will try to cover this topic as broadly as possible using Expert Advisors with settings obtained using this method. A new program version is attached to this article.

![Other classes in DoEasy library (Part 68): Chart window object class and indicator object classes in the chart window](https://c.mql5.com/2/42/MQL5-avatar-doeasy-library__6.png)[Other classes in DoEasy library (Part 68): Chart window object class and indicator object classes in the chart window](https://www.mql5.com/en/articles/9236)

In this article, I will continue the development of the chart object class. I will add the list of chart window objects featuring the lists of available indicators.

![Other classes in DoEasy library (Part 66): MQL5.com Signals collection class](https://c.mql5.com/2/42/MQL5-avatar-doeasy-library__4.png)[Other classes in DoEasy library (Part 66): MQL5.com Signals collection class](https://www.mql5.com/en/articles/9146)

In this article, I will create the signal collection class of the MQL5.com Signals service with the functions for managing signals. Besides, I will improve the Depth of Market snapshot object class for displaying the total DOM buy and sell volumes.

[![](https://www.mql5.com/ff/sh/0hvxp984jjj79943z2/6373d9e5710a718ffa6a7d50a5db9dd1.jpg)\\
Web terminal on your iPhone or Android\\
\\
Full-featured MetaTrader 5 platform for any devices and web browsers\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=uyigsjnbfcdvysiynusmriwvhincciwd&s=c95531ae2fd8a81b0fac3def2e4cf820a67584bbf4b02f76ec75f808942dbbd2&uid=&ref=https://www.mql5.com/en/articles/9213&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5083409342117452586)

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