---
title: Other classes in DoEasy library (Part 66): MQL5.com Signals collection class
url: https://www.mql5.com/en/articles/9146
categories: Trading Systems, Expert Advisors
relevance_score: 4
scraped_at: 2026-01-23T17:46:58.098622
---

[![](https://www.mql5.com/ff/si/h2ryn394uwcpxwmxc2.jpg)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Feconomic-calendar%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dopen.calendar%26utm_content%3Deconomic.calendar%26utm_campaign%3Den.0009.desktop.default&a=qdeulxgvibgwytgewnvfatbocjnnninc&s=5c0c60f00ff5f5bedb0fdf65d9d79eb820442eb43ffac2b85aa003224f9dba14&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=pzvmyqytxywysujjaiuecpezuchsobtb&ssn=1769179615299426518&ssn_dr=0&ssn_sr=0&fv_date=1769179615&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F9146&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Other%20classes%20in%20DoEasy%20library%20(Part%2066)%3A%20MQL5.com%20Signals%20collection%20class%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176917961587013292&fz_uniq=5068642518704258015&sv=2552)

MetaTrader 5 / Examples


### Contents

- [Concept](https://www.mql5.com/en/articles/9146#node01)
- [Improving library classes](https://www.mql5.com/en/articles/9146#node02)
- [Collection class of MQL5 signal objects](https://www.mql5.com/en/articles/9146#node03)
- [Test](https://www.mql5.com/en/articles/9146#node04)
- [What's next?](https://www.mql5.com/en/articles/9146#node05)


### Concept

[In the previous article, I created the signal object class](https://www.mql5.com/en/articles/9095#node04), which is one signal out of multiple ones broadcast [in the MQL5.com Signals service](https://www.mql5.com/en/signals).

Today I will create the collection class of signals available in the signal database that can be obtained using the [SignalBaseSelect()](https://www.mql5.com/en/docs/signals/signalbaseselect) function by specifying the index of the necessary signal.

The collection allows storing all signals existing in the database as a list that is convenient for search and sorting. We will be able to detect and return the lists of signals by their various properties. For instance, we can obtain the lists of only free or only paid signals, as well as sort them by one of the parameters (like a signal profitability) or immediately get the signal index in the list with the parameter equal, exceeding or less than the specified value. The knowledge of a necessary signal allows us to quickly find it in the collection.

In the collection class, implement the ability to subscribe to a signal selected in the collection or unsubscribe from a signal on the current account.

Apart from working with the MQL5.com Signals service objects, we will improve [the Depth of Market (DOM) snapshot object class](https://www.mql5.com/en/articles/9044#node03) by adding extra properties allowing us to immediately calculate buy and sell order volumes when creating a DOM snapshot object. This will save end users from the necessity to conduct additional calculations when working with the DOM. We will immediately know the total buy and sell volumes of each DOM snapshot. This means we do not have to additionally search for Buy and Sell orders in the DOM with the subsequent summing up of their volumes when creating strategies using the DOM and its volumes.

### Improving library classes

As usual, let's add all new library messages to \\MQL5\\Include\\DoEasy\ **Data.mqh** right away.

Add the new message indices first:

```
//--- CMarketBookSnapshot
   MSG_MBOOK_SNAP_TEXT_SNAPSHOT,                      // DOM snapshot
   MSG_MBOOK_SNAP_VOLUME_BUY,                         // Buy volume
   MSG_MBOOK_SNAP_VOLUME_SELL,                        // Sell volume

//--- CMBookSeries
   MSG_MBOOK_SERIES_TEXT_MBOOKSERIES,                 // DOM snapshot series
   MSG_MBOOK_SERIES_ERR_ADD_TO_LIST,                  // Error. Failed to add DOM snapshot series to the list
```

...

```
//--- CMQLSignal
   MSG_SIGNAL_MQL5_TEXT_SIGNAL,                       // Signal
   MSG_SIGNAL_MQL5_TEXT_SIGNAL_MQL5,                  // MQL5.com Signals service signal
   MSG_SIGNAL_MQL5_TRADE_MODE,                        // Account type
   MSG_SIGNAL_MQL5_DATE_PUBLISHED,                    // Publication date
   MSG_SIGNAL_MQL5_DATE_STARTED,                      // Monitoring start date
   MSG_SIGNAL_MQL5_DATE_UPDATED,                      // Date of the latest update of the trading statistics
   MSG_SIGNAL_MQL5_ID,                                // ID
   MSG_SIGNAL_MQL5_LEVERAGE,                          // Trading account leverage
   MSG_SIGNAL_MQL5_PIPS,                              // Trading result in pips
   MSG_SIGNAL_MQL5_RATING,                            // Position in the signal rating
   MSG_SIGNAL_MQL5_SUBSCRIBERS,                       // Number of subscribers
   MSG_SIGNAL_MQL5_TRADES,                            // Number of trades
   MSG_SIGNAL_MQL5_SUBSCRIPTION_STATUS,               // Status of account subscription to a signal
   MSG_SIGNAL_MQL5_EQUITY,                            // Account equity
   MSG_SIGNAL_MQL5_GAIN,                              // Account growth in %
   MSG_SIGNAL_MQL5_MAX_DRAWDOWN,                      // Maximum drawdown
   MSG_SIGNAL_MQL5_PRICE,                             // Signal subscription price
   MSG_SIGNAL_MQL5_ROI,                               // Signal ROI (Return on Investment) in %
   MSG_SIGNAL_MQL5_AUTHOR_LOGIN,                      // Author login
   MSG_SIGNAL_MQL5_BROKER,                            // Broker (company) name
   MSG_SIGNAL_MQL5_BROKER_SERVER,                     // Broker server
   MSG_SIGNAL_MQL5_NAME,                              // Name
   MSG_SIGNAL_MQL5_CURRENCY,                          // Account currency
   MSG_SIGNAL_MQL5_TEXT_GAIN,                         // Growth
   MSG_SIGNAL_MQL5_TEXT_DRAWDOWN,                     // Drawdown
   MSG_SIGNAL_MQL5_TEXT_SUBSCRIBERS,                  // Subscribers

//--- CMQLSignalsCollection
   MSG_MQLSIG_COLLECTION_TEXT_MQL5_SIGNAL_COLLECTION, // Collection of MQL5.com Signals service signals
   MSG_MQLSIG_COLLECTION_TEXT_SIGNALS_PAID,           // Paid signals
   MSG_MQLSIG_COLLECTION_TEXT_SIGNALS_FREE,           // Free signals
   MSG_MQLSIG_COLLECTION_TEXT_SIGNALS_NEW,            // New signal added to collection
   MSG_MQLSIG_COLLECTION_ERR_FAILED_GET_SIGNAL,       // Failed to receive signal from collection
   MSG_SIGNAL_INFO_PARAMETERS,                        // Signal copying parameters
   MSG_SIGNAL_INFO_EQUITY_LIMIT,                      // Percentage for converting deal volume
   MSG_SIGNAL_INFO_SLIPPAGE,                          // Market order slippage when synchronizing positions and copying deals
   MSG_SIGNAL_INFO_VOLUME_PERCENT,                    // Limitation on signal equity
   MSG_SIGNAL_INFO_CONFIRMATIONS_DISABLED,            // Enable synchronization without confirmation dialog
   MSG_SIGNAL_INFO_COPY_SLTP,                         // Copy Stop Loss and Take Profit
   MSG_SIGNAL_INFO_DEPOSIT_PERCENT,                   // Limit by deposit
   MSG_SIGNAL_INFO_ID,                                // Signal ID
   MSG_SIGNAL_INFO_SUBSCRIPTION_ENABLED,              // Enable copying deals by subscription
   MSG_SIGNAL_INFO_TERMS_AGREE,                       // Agree to the terms of use of the Signals service
   MSG_SIGNAL_INFO_NAME,                              // Signal name
   MSG_SIGNAL_INFO_SIGNALS_PERMISSION,                // Allow using signals for program
   MSG_SIGNAL_INFO_TEXT_SIGNAL_SUBSCRIBED,            // Subscribed to signal
   MSG_SIGNAL_INFO_TEXT_SIGNAL_UNSUBSCRIBED,          // Unsubscribed from signal
   MSG_SIGNAL_INFO_ERR_SIGNAL_NOT_ALLOWED,            // Signal service disabled for program
   MSG_SIGNAL_INFO_TEXT_CHECK_SETTINGS,               // Please check program settings (Common-->Allow modification of Signals settings)

  };
//+------------------------------------------------------------------+
```

Next, add text messages corresponding to the newly added indices:

```
//--- CMarketBookSnapshot
   {"Снимок стакана цен","Depth of Market Snapshot"},
   {"Объём на покупку","Buy Volume"},
   {"Объём на продажу","Sell Volume"},

//--- CMBookSeries
   {"Серия снимков стакана цен","Series of shots of the Depth of Market"},
   {"Ошибка. Не удалось добавить серию снимков стакана цен в список","Error. Failed to add a shots series of the Depth of Market to the list"},

//--- CMBookSeriesCollection
   {"Коллекция серий снимков стакана цен","Collection of series of the Depth of Market shot"},

//--- CMQLSignal
   {"Сигнал","Signal"},
   {"Сигнал сервиса сигналов mql5.com","Signal from mql5.com signal service"},
   {"Тип счета","Account type"},
   {"Дата публикации","Publication date"},
   {"Дата начала мониторинга","Monitoring starting date"},
   {"Дата последнего обновления торговой статистики","The date of the last update of the signal's trading statistics"},
   {"ID","ID"},
   {"Плечо торгового счета","Account leverage"},
   {"Результат торговли в пипсах","Profit in pips"},
   {"Позиция в рейтинге сигналов","Position in rating"},
   {"Количество подписчиков","Number of subscribers"},
   {"Количество трейдов","Number of trades"},
   {"Состояние подписки счёта на этот сигнал","Account subscription status for this signal"},
   {"Средства на счете","Account equity"},
   {"Прирост счета в процентах","Account gain"},
   {"Максимальная просадка","Account maximum drawdown"},
   {"Цена подписки на сигнал","Signal subscription price"},
   {"Значение ROI (Return on Investment) сигнала в %","Return on Investment (%)"},
   {"Логин автора","Author login"},
   {"Наименование брокера (компании)","Broker name (company)"},
   {"Сервер брокера","Broker server"},
   {"Имя","Name"},
   {"Валюта счета","Base currency"},
   {"Прирост","Gain"},
   {"Просадка","Drawdown"},
   {"Подписчиков","Subscribers"},

//--- CMQLSignalsCollection
   {"Коллекция сигналов сервиса сигналов mql5.com","Collection of signals from the mql5.com signal service"},
   {"Платных сигналов","Paid signals"},
   {"Бесплатных сигналов","Free signals"},
   {"Новый сигнал добавлен в коллекцию","New signal added to collection"},
   {"Не удалось получить сигнал из коллекции","Failed to get signal from collection"},
   {"Параметры копирования сигнала","Signal copying parameters"},
   {"Процент для конвертации объема сделки","Equity limit"},
   {"Проскальзывание, с которым выставляются рыночные ордера при синхронизации позиций и копировании сделок","Slippage (used when placing market orders in synchronization of positions and copying of trades)"},
   {"Ограничение по средствам для сигнала","Maximum percent of deposit used"},
   {"Разрешение синхронизации без показа диалога подтверждения","Allow synchronization without confirmation dialog"},
   {"Копирование Stop Loss и Take Profit","Copy Stop Loss and Take Profit"},
   {"Ограничение по депозиту","Deposit percent"},
   {"Идентификатор сигнала","Signal ID"},
   {"Разрешение на копирование сделок по подписке","Permission to copy trades by subscription"},
   {"Согласие с условиями использования сервиса \"Сигналы\"","Agree to the terms of use of the \"Signals\" service"},
   {"Имя сигнала","Signal name"},
   {"Разрешение на работу с сигналами для программы","Permission to work with signals for the program"},
   {"Осуществлена подписка на сигнал","Signal subscribed"},
   {"Осуществлена отписка от сигнала","Signal unsubscribed"},
   {"Работа с сервисом сигналов для программы не разрешена","Work with the \"Signals\" service is not allowed for the program"},
   {
    "Пожалуйста, проверьте настройки программы (Общие --> Разрешить изменение настроек Сигналов)",
    "Please check the program settings (Common --> Allow modification of Signals settings)"
   },

  };
//+---------------------------------------------------------------------+
```

When displaying messages in the journal (especially debugging ones), we often indicate the name of a method a message was sent from at the start of the message itself. [In the article 19, I developed the class of library messages](https://www.mql5.com/en/articles/7176). However, I currently use them only to specify indices of the messages that should be displayed in the journal via the standard [Print()](https://www.mql5.com/en/docs/common/print) function. Since I am going to start a new library section for working with graphics soon, I will gradually move on to working with this class for displaying library messages. Today, I will add the ToLog() method overload to it, so that we can additionally pass a message "source" to the method of a class or to the function of a program the method was called from. Thus, we will have two variants of the ToLog() method allowing us to display messages with specifying its source function or method and without it.

Open \\MQL5\\Include\\DoEasy\\Services\ **Message.mqh** and add the declaration of an overloaded method to it:

```
//--- (1,2) display a message in the journal by ID, (3) to e-mail, (4) to a mobile device
   static void       ToLog(const int msg_id,const bool code=false);
   static void       ToLog(const string source,const int msg_id,const bool code=false);
   static bool       ToMail(const string message,const string subject=NULL);
   static bool       Push(const string message);
//--- (1) send a file to FTP, (2) return an error code
   static bool       ToFTP(const string filename,const string ftp_path=NULL);
   static int        GetError(void)                { return CMessage::m_global_error;  }
```

Let's write its implementation beyond its class body:

```
//+------------------------------------------------------------------+
//| Display a message in the journal by a message ID                 |
//+------------------------------------------------------------------+
void CMessage::ToLog(const int msg_id,const bool code=false)
  {
   CMessage::GetTextByID(msg_id);
   ::Print(m_text,(!code || msg_id>ERR_USER_ERROR_FIRST-1 ? "" : " "+CMessage::Retcode(msg_id)));
  }
//+------------------------------------------------------------------+
//| Display a message in the journal by a message ID                 |
//+------------------------------------------------------------------+
void CMessage::ToLog(const string source,const int msg_id,const bool code=false)
  {
   CMessage::GetTextByID(msg_id);
   ::Print(source,m_text,(!code || msg_id>ERR_USER_ERROR_FIRST-1 ? "" : " "+CMessage::Retcode(msg_id)));
  }
//+------------------------------------------------------------------+
```

Unlike the first form of calling the method, its second form features yet another input, in which the name of a method or function the ToLog() method should be called from is passed and which is to be displayed in the journal before the message.

We will get back to this class in subsequent articles to make improvements to it when transferring all library classes to display messages using this class.

Let's improve the **CMBookSnapshot** class in \\MQL5\\Include\\DoEasy\\Objects\\Book\ **MarketBookSnapshot.mqh**.

In the private class section, add class member variables for storing total buy and sell volumes of a DOM snapshot:

```
//+------------------------------------------------------------------+
//| "DOM snapshot" class                                             |
//+------------------------------------------------------------------+
class CMBookSnapshot : public CBaseObj
  {
private:
   string            m_symbol;                  // Symbol
   long              m_time;                    // Snapshot time
   int               m_digits;                  // Symbol's Digits
   long              m_volume_buy;              // DOM buy volume
   long              m_volume_sell;             // DOM sell volume
   double            m_volume_buy_real;         // DOM buy volume with an increased accuracy
   double            m_volume_sell_real;        // DOM sell volume with an increased accuracy
   CArrayObj         m_list;                    // List of DOM order objects
public:
```

In the section of methods for simplified access to DOM snapshot object properties of the class section, add the methods returning these newly added class properties and the methods for displaying their description:

```
//+----------------------------------------------------------------------+
//|Methods of a simplified access to the DOM snapshot object properties  |
//+----------------------------------------------------------------------+
//--- Set (1) a symbol, (2) a DOM snapshot time and (3) the specified time for all DOM orders
   void              SetSymbol(const string symbol)   { this.m_symbol=(symbol==NULL || symbol=="" ? ::Symbol() : symbol); }
   void              SetTime(const long time_msc)     { this.m_time=time_msc; }
   void              SetTimeToOrders(const long time_msc);
//--- Return (1) a DOM symbol, (2) symbol's Digits and (3) a snapshot time
//--- (4) buy and (5) sell DOM snapshot volume
//--- with increased accuracy for (6) buying and (7) selling,
   string            Symbol(void)               const { return this.m_symbol;             }
   int               Digits(void)               const { return this.m_digits;             }
   long              Time(void)                 const { return this.m_time;               }
   long              VolumeBuy(void)            const { return this.m_volume_buy;         }
   long              VolumeSell(void)           const { return this.m_volume_sell;        }
   double            VolumeBuyReal(void)        const { return this.m_volume_buy_real;    }
   double            VolumeSellReal(void)       const { return this.m_volume_sell_real;   }

//--- Return the description of DOM (1) buy and (2) sell volume
   string            VolumeBuyDescription(void);
   string            VolumeSellDescription(void);

  };
//+------------------------------------------------------------------+
```

When creating a new DOM snapshot object, we see all its orders in a loop, create objects of these orders and send them to the list. Now we need to consider order types in the class constructor and immediately add the current order volume to the variables storing the total volume of buy and sell orders depending on the current order type. Thus, each variable will eventually store the total volume of either buy or sell orders immediately upon creating the DOM snapshot object.

**Let's add these improvements to the class parametric constructor:**

```
//+------------------------------------------------------------------+
//| Parametric constructor                                           |
//+------------------------------------------------------------------+
CMBookSnapshot::CMBookSnapshot(const string symbol,const long time,MqlBookInfo &book_array[]) : m_time(time)
  {
   //--- Set a symbol
   this.SetSymbol(symbol);
   //--- Clear the list
   this.m_list.Clear();
   //--- In the loop by the structure array
   int total=::ArraySize(book_array);
   this.m_volume_buy=this.m_volume_sell=0;
   this.m_volume_buy_real=this.m_volume_sell_real=0;
   for(int i=0;i<total;i++)
     {
      //--- Create order objects of the current DOM snapshot depending on the order type
      CMarketBookOrd *mbook_ord=NULL;
      switch(book_array[i].type)
        {
         case BOOK_TYPE_BUY         : mbook_ord=new CMarketBookBuy(this.m_symbol,book_array[i]);         break;
         case BOOK_TYPE_SELL        : mbook_ord=new CMarketBookSell(this.m_symbol,book_array[i]);        break;
         case BOOK_TYPE_BUY_MARKET  : mbook_ord=new CMarketBookBuyMarket(this.m_symbol,book_array[i]);   break;
         case BOOK_TYPE_SELL_MARKET : mbook_ord=new CMarketBookSellMarket(this.m_symbol,book_array[i]);  break;
         default: break;
        }
      if(mbook_ord==NULL)
         continue;
      //--- Set the DOM snapshot time for the order
      mbook_ord.SetTime(this.m_time);

      //--- Set the sorted list flag for the list (by the price value) and add the current order object to it
      //--- If failed to add the object to the DOM order list, remove the order object
      this.m_list.Sort(SORT_BY_MBOOK_ORD_PRICE);
      if(!this.m_list.InsertSort(mbook_ord))
         delete mbook_ord;
      //--- If the order object is successfully added to the DOM order list, supplement the total snapshot volumes
      else
        {
         switch(mbook_ord.TypeOrd())
           {
            case BOOK_TYPE_BUY         :
              this.m_volume_buy+=mbook_ord.Volume();
              this.m_volume_buy_real+=mbook_ord.VolumeReal();
              break;
            case BOOK_TYPE_SELL        :
              this.m_volume_sell+=mbook_ord.Volume();
              this.m_volume_sell_real+=mbook_ord.VolumeReal();
              break;
            case BOOK_TYPE_BUY_MARKET  :
              this.m_volume_buy+=mbook_ord.Volume();
              this.m_volume_buy_real+=mbook_ord.VolumeReal();
              break;
            case BOOK_TYPE_SELL_MARKET :
              this.m_volume_buy+=mbook_ord.Volume();
              this.m_volume_buy_real+=mbook_ord.VolumeReal();
              break;
            default: break;
           }
        }
     }
  }
//+------------------------------------------------------------------+
```

First, initialize the variables storing total volumes of all DOM snapshot buy and sell orders. Next, in the loop body by all DOM orders, depending on the order type, add the current order volume to the appropriate variables storing the total volumes. Thus, total buy and sell volumes will be stored in all variables upon the loop completion by all DOM snapshot orders.

**The methods displaying the object short descriptionand all object properties in the journal, receive the display of total buy and sell volumes:**

```
//+------------------------------------------------------------------+
//| Display a short description of the object in the journal         |
//+------------------------------------------------------------------+
void CMBookSnapshot::PrintShort(void)
  {
   string vol_buy="Buy vol: "+(this.VolumeBuyReal()>0 ? ::DoubleToString(this.VolumeBuyReal(),2) : (string)this.VolumeBuy());
   string vol_sell="Sell vol: "+(this.VolumeSellReal()>0 ? ::DoubleToString(this.VolumeSellReal(),2) : (string)this.VolumeSell());
   ::Print(this.Header()," ",vol_buy,", ",vol_sell," ("+TimeMSCtoString(this.m_time),")");
  }
//+------------------------------------------------------------------+
//| Display object properties in the journal                         |
//+------------------------------------------------------------------+
void CMBookSnapshot::Print(void)
  {
   string vol_buy=CMessage::Text(MSG_MBOOK_SNAP_VOLUME_BUY)+": "+(this.VolumeBuyReal()>0 ? ::DoubleToString(this.VolumeBuyReal(),2) : (string)this.VolumeBuy());
   string vol_sell=CMessage::Text(MSG_MBOOK_SNAP_VOLUME_SELL)+": "+(this.VolumeSellReal()>0 ? ::DoubleToString(this.VolumeSellReal(),2) : (string)this.VolumeSell());
   ::Print(this.Header(),": ",vol_buy,", ",vol_sell," ("+TimeMSCtoString(this.m_time),"):");
   this.m_list.Sort(SORT_BY_MBOOK_ORD_PRICE);
   for(int i=this.m_list.Total()-1;i>WRONG_VALUE;i--)
     {
      CMarketBookOrd *ord=this.m_list.At(i);
      if(ord==NULL)
         continue;
      ::Print("- ",ord.Header());
     }
  }
//+------------------------------------------------------------------+
```

**Beyond the class body, implement the two new methods returning descriptions of DOM buy and sell volumes:**

```
//+------------------------------------------------------------------+
//| Return the DOM buy volume description                            |
//+------------------------------------------------------------------+
string CMBookSnapshot::VolumeBuyDescription(void)
  {
   return(CMessage::Text(MSG_MBOOK_SNAP_VOLUME_BUY)+": "+(this.VolumeBuyReal()>0 ? ::DoubleToString(this.VolumeBuyReal(),2) : (string)this.VolumeBuy()));
  }
//+------------------------------------------------------------------+
//| Return the DOM sell volume description                           |
//+------------------------------------------------------------------+
string CMBookSnapshot::VolumeSellDescription(void)
  {
   return(CMessage::Text(MSG_MBOOK_SNAP_VOLUME_SELL)+": "+(this.VolumeSellReal()>0 ? ::DoubleToString(this.VolumeSellReal(),2) : (string)this.VolumeSell()));
  }
//+------------------------------------------------------------------+
```

Both methods check the increased accuracy volume. If it exceeds zero, the header + the volume value (real) is returned, otherwise — integer.

The **CMBookSeriesCollection** DOM snapshot series collection class in \\MQL5\\Include\\DoEasy\\Collections\ **BookSeriesCollection.mqh**, namely its public section, receives the methods returning the lists by the specified criteria of the list object properties:

```
public:
//--- Return (1) itself and (2) the DOM series collection list
   CMBookSeriesCollection *GetObject(void)                              { return &this;               }
   CArrayObj              *GetList(void)                                { return &this.m_list;        }
   //--- Return the list by selected (1) integer, (2) real and (3) string properties meeting the compared criterion
   CArrayObj              *GetList(ENUM_MBOOK_ORD_PROP_INTEGER property,long value,ENUM_COMPARER_TYPE mode=EQUAL)  { return CSelect::ByMBookProperty(this.GetList(),property,value,mode);  }
   CArrayObj              *GetList(ENUM_MBOOK_ORD_PROP_DOUBLE property,double value,ENUM_COMPARER_TYPE mode=EQUAL) { return CSelect::ByMBookProperty(this.GetList(),property,value,mode);  }
   CArrayObj              *GetList(ENUM_MBOOK_ORD_PROP_STRING property,string value,ENUM_COMPARER_TYPE mode=EQUAL) { return CSelect::ByMBookProperty(this.GetList(),property,value,mode);  }
//--- Return the number of DOM series in the list
   int                     DataTotal(void)                        const { return this.m_list.Total();    }
//--- Return the pointer to the DOM series object (1) by symbol and (2) by index in the list
```

In the **CMQLSignal** DOM order object class in \\MQL5\\Include\\DoEasy\\Objects\\MQLSignalBase\ **MQLSignal.mqh**, supplement the PrintShort() method with the flag value indicating the need to display a hyphen before the object description:

```
//--- Display the description of object properties in the journal (full_prop=true - all properties, false - supported ones only)
   void              Print(const bool full_prop=false);
//--- Display a short description of the object in the journal
   virtual void      PrintShort(const bool dash=false);
//--- Return the object short name
   virtual string    Header(const bool shrt=false);
```

Let's make changes in the method body:

```
//+------------------------------------------------------------------+
//| Display a short description of the object in the journal         |
//+------------------------------------------------------------------+
void CMQLSignal::PrintShort(const bool dash=false)
  {
   ::Print
     (
      (dash ? "- " : ""),this.Header(true),
      " \"",this.Name(),"\". ",
      CMessage::Text(MSG_SIGNAL_MQL5_AUTHOR_LOGIN),": ",this.AuthorLogin(),
      ", ID ",this.ID(),
      ", ",CMessage::Text(MSG_SIGNAL_MQL5_TEXT_GAIN),": ",::DoubleToString(this.Gain(),2),
      ", ",CMessage::Text(MSG_SIGNAL_MQL5_TEXT_DRAWDOWN),": ",::DoubleToString(this.MaxDrawdown(),2),
      ", ",CMessage::Text(MSG_LIB_TEXT_REQUEST_PRICE),": ",::DoubleToString(this.Price(),2),
      ", ",CMessage::Text(MSG_SIGNAL_MQL5_TEXT_SUBSCRIBERS),": ",this.Subscribers()
     );
  }
//+------------------------------------------------------------------+
```

Depending on a passed value, a hyphen is displayed or not displayed before the object description. A number of subscribers is set at the very end of the description.

**At the very end of the class body, add the new method performing a signal subscription described by the object:**

```
//--- Return the account type name
   string            TradeModeDescription(void);

//--- Subscribe to a signal
   bool              Subscribe(void)                        { return ::SignalSubscribe(this.ID()); }

  };
//+------------------------------------------------------------------+
```

**In the class constructor, fix the initialization of the variable storing the signal subscription status:**

```
//+------------------------------------------------------------------+
//| Parametric constructor                                           |
//+------------------------------------------------------------------+
CMQLSignal::CMQLSignal(const long signal_id)
  {
   this.m_long_prop[SIGNAL_MQL5_PROP_ID]                             = signal_id;
   this.m_long_prop[SIGNAL_MQL5_PROP_SUBSCRIPTION_STATUS]            = (::SignalInfoGetInteger(SIGNAL_INFO_ID)==signal_id);
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

Previously, it was initialized with false. Now we are going to initialize it with the result of comparing the signal object IDwith the ID of the current signal with active subscription. If a signal has an active subscription, it will be the "currently subscribed" one featuring the signal ID from the MQL5.com Signal database. If they are equal, this means the subscription is active on that signal — the comparison result is equal to true, otherwise — false.

Since I am developing a new collection here, I need to define a custom ID for it. In \\MQL5\\Include\\DoEasy\ **Defines.mqh**, **add the ID of the MQL5.com Signals service signals collection:**

```
//--- Collection list IDs
#define COLLECTION_HISTORY_ID          (0x777A)                   // Historical collection list ID
#define COLLECTION_MARKET_ID           (0x777B)                   // Market collection list ID
#define COLLECTION_EVENTS_ID           (0x777C)                   // Event collection list ID
#define COLLECTION_ACCOUNT_ID          (0x777D)                   // Account collection list ID
#define COLLECTION_SYMBOLS_ID          (0x777E)                   // Symbol collection list ID
#define COLLECTION_SERIES_ID           (0x777F)                   // Timeseries collection list ID
#define COLLECTION_BUFFERS_ID          (0x7780)                   // Indicator buffer collection list ID
#define COLLECTION_INDICATORS_ID       (0x7781)                   // Indicator collection list ID
#define COLLECTION_INDICATORS_DATA_ID  (0x7782)                   // Indicator data collection list ID
#define COLLECTION_TICKSERIES_ID       (0x7783)                   // Tick series collection list ID
#define COLLECTION_MBOOKSERIES_ID      (0x7784)                   // DOM series collection list ID
#define COLLECTION_MQL5_SIGNALS_ID     (0x7785)                   // MQL5 signals collection list ID
//--- Data parameters for file operations
```

To be able to work with the MQL5.com Signals service signals collection, we need to create the methods for searching and sorting by signal object properties. A separate search and sorting method is created for each collection. All methods are identical to each other. They were described in detail [in the third article](https://www.mql5.com/en/articles/5687).

In the **CSelect** class file in \\MQL5\\Include\\DoEasy\\Services\ **Select.mqh**, **include the MQL5 signal object class file and declare the new methods for working with the signal object collection:**

```
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include <Arrays\ArrayObj.mqh>
#include "..\Objects\Orders\Order.mqh"
#include "..\Objects\Events\Event.mqh"
#include "..\Objects\Accounts\Account.mqh"
#include "..\Objects\Symbols\Symbol.mqh"
#include "..\Objects\PendRequest\PendRequest.mqh"
#include "..\Objects\Series\SeriesDE.mqh"
#include "..\Objects\Indicators\Buffer.mqh"
#include "..\Objects\Indicators\IndicatorDE.mqh"
#include "..\Objects\Indicators\DataInd.mqh"
#include "..\Objects\Ticks\DataTick.mqh"
#include "..\Objects\Book\MarketBookOrd.mqh"
#include "..\Objects\MQLSignalBase\MQLSignal.mqh"
//+------------------------------------------------------------------+

//+------------------------------------------------------------------+
//| Methods of working with MQL5 signal data                         |
//+------------------------------------------------------------------+
   //--- Return the list of MQL5 signals with one of (1) integer, (2) real and (3) string properties meeting a specified criterion
   static CArrayObj *ByMQLSignalProperty(CArrayObj *list_source,ENUM_SIGNAL_MQL5_PROP_INTEGER property,long value,ENUM_COMPARER_TYPE mode);
   static CArrayObj *ByMQLSignalProperty(CArrayObj *list_source,ENUM_SIGNAL_MQL5_PROP_DOUBLE property,double value,ENUM_COMPARER_TYPE mode);
   static CArrayObj *ByMQLSignalProperty(CArrayObj *list_source,ENUM_SIGNAL_MQL5_PROP_STRING property,string value,ENUM_COMPARER_TYPE mode);
   //--- Return the MQL5 signal index in the list with the maximum value of the (1) integer, (2) real and (3) string properties
   static int        FindMQLSignalMax(CArrayObj *list_source,ENUM_SIGNAL_MQL5_PROP_INTEGER property);
   static int        FindMQLSignalMax(CArrayObj *list_source,ENUM_SIGNAL_MQL5_PROP_DOUBLE property);
   static int        FindMQLSignalMax(CArrayObj *list_source,ENUM_SIGNAL_MQL5_PROP_STRING property);
   //--- Return the MQL5 signal in the list with the minimum value of (1) integer, (2) real and (3) string property
   static int        FindMQLSignalMin(CArrayObj *list_source,ENUM_SIGNAL_MQL5_PROP_INTEGER property);
   static int        FindMQLSignalMin(CArrayObj *list_source,ENUM_SIGNAL_MQL5_PROP_DOUBLE property);
   static int        FindMQLSignalMin(CArrayObj *list_source,ENUM_SIGNAL_MQL5_PROP_STRING property);
//---
  };
//+------------------------------------------------------------------+
```

**Let's write their implementation outside the class body:**

```
//+------------------------------------------------------------------+
//| Methods of working with MQL5 signal data                         |
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| Return the list of MQL5 signals with one integer                 |
//| property meeting the specified criterion                         |
//+------------------------------------------------------------------+
CArrayObj *CSelect::ByMQLSignalProperty(CArrayObj *list_source,ENUM_SIGNAL_MQL5_PROP_INTEGER property,long value,ENUM_COMPARER_TYPE mode)
  {
   if(list_source==NULL) return NULL;
   CArrayObj *list=new CArrayObj();
   if(list==NULL) return NULL;
   list.FreeMode(false);
   ListStorage.Add(list);
   int total=list_source.Total();
   for(int i=0; i<total; i++)
     {
      CMQLSignal *obj=list_source.At(i);
      if(!obj.SupportProperty(property)) continue;
      long obj_prop=obj.GetProperty(property);
      if(CompareValues(obj_prop,value,mode)) list.Add(obj);
     }
   return list;
  }
//+------------------------------------------------------------------+
//| Return the list of MQL5 signals with one real                    |
//| property meeting the specified criterion                         |
//+------------------------------------------------------------------+
CArrayObj *CSelect::ByMQLSignalProperty(CArrayObj *list_source,ENUM_SIGNAL_MQL5_PROP_DOUBLE property,double value,ENUM_COMPARER_TYPE mode)
  {
   if(list_source==NULL) return NULL;
   CArrayObj *list=new CArrayObj();
   if(list==NULL) return NULL;
   list.FreeMode(false);
   ListStorage.Add(list);
   for(int i=0; i<list_source.Total(); i++)
     {
      CMQLSignal *obj=list_source.At(i);
      if(!obj.SupportProperty(property)) continue;
      double obj_prop=obj.GetProperty(property);
      if(CompareValues(obj_prop,value,mode)) list.Add(obj);
     }
   return list;
  }
//+------------------------------------------------------------------+
//| Return the list of MQL5 signals with one string                  |
//| property meeting the specified criterion                         |
//+------------------------------------------------------------------+
CArrayObj *CSelect::ByMQLSignalProperty(CArrayObj *list_source,ENUM_SIGNAL_MQL5_PROP_STRING property,string value,ENUM_COMPARER_TYPE mode)
  {
   if(list_source==NULL) return NULL;
   CArrayObj *list=new CArrayObj();
   if(list==NULL) return NULL;
   list.FreeMode(false);
   ListStorage.Add(list);
   for(int i=0; i<list_source.Total(); i++)
     {
      CMQLSignal *obj=list_source.At(i);
      if(!obj.SupportProperty(property)) continue;
      string obj_prop=obj.GetProperty(property);
      if(CompareValues(obj_prop,value,mode)) list.Add(obj);
     }
   return list;
  }
//+------------------------------------------------------------------+
//| Return the MQL5 signal index in the list                         |
//| with the maximum integer property value                          |
//+------------------------------------------------------------------+
int CSelect::FindMQLSignalMax(CArrayObj *list_source,ENUM_SIGNAL_MQL5_PROP_INTEGER property)
  {
   if(list_source==NULL) return WRONG_VALUE;
   int index=0;
   CMQLSignal *max_obj=NULL;
   int total=list_source.Total();
   if(total==0) return WRONG_VALUE;
   for(int i=1; i<total; i++)
     {
      CMQLSignal *obj=list_source.At(i);
      long obj1_prop=obj.GetProperty(property);
      max_obj=list_source.At(index);
      long obj2_prop=max_obj.GetProperty(property);
      if(CompareValues(obj1_prop,obj2_prop,MORE)) index=i;
     }
   return index;
  }
//+------------------------------------------------------------------+
//| Return the MQL5 signal index in the list                         |
//| with the maximum real property value                             |
//+------------------------------------------------------------------+
int CSelect::FindMQLSignalMax(CArrayObj *list_source,ENUM_SIGNAL_MQL5_PROP_DOUBLE property)
  {
   if(list_source==NULL) return WRONG_VALUE;
   int index=0;
   CMQLSignal *max_obj=NULL;
   int total=list_source.Total();
   if(total==0) return WRONG_VALUE;
   for(int i=1; i<total; i++)
     {
      CMQLSignal *obj=list_source.At(i);
      double obj1_prop=obj.GetProperty(property);
      max_obj=list_source.At(index);
      double obj2_prop=max_obj.GetProperty(property);
      if(CompareValues(obj1_prop,obj2_prop,MORE)) index=i;
     }
   return index;
  }
//+------------------------------------------------------------------+
//| Return the MQL5 signal index in the list                         |
//| with the maximum string property value                           |
//+------------------------------------------------------------------+
int CSelect::FindMQLSignalMax(CArrayObj *list_source,ENUM_SIGNAL_MQL5_PROP_STRING property)
  {
   if(list_source==NULL) return WRONG_VALUE;
   int index=0;
   CMQLSignal *max_obj=NULL;
   int total=list_source.Total();
   if(total==0) return WRONG_VALUE;
   for(int i=1; i<total; i++)
     {
      CMQLSignal *obj=list_source.At(i);
      string obj1_prop=obj.GetProperty(property);
      max_obj=list_source.At(index);
      string obj2_prop=max_obj.GetProperty(property);
      if(CompareValues(obj1_prop,obj2_prop,MORE)) index=i;
     }
   return index;
  }
//+------------------------------------------------------------------+
//| Return the MQL5 signal index in the list                         |
//| with the minimum integer property value                          |
//+------------------------------------------------------------------+
int CSelect::FindMQLSignalMin(CArrayObj* list_source,ENUM_SIGNAL_MQL5_PROP_INTEGER property)
  {
   int index=0;
   CMQLSignal *min_obj=NULL;
   int total=list_source.Total();
   if(total==0) return WRONG_VALUE;
   for(int i=1; i<total; i++)
     {
      CMQLSignal *obj=list_source.At(i);
      long obj1_prop=obj.GetProperty(property);
      min_obj=list_source.At(index);
      long obj2_prop=min_obj.GetProperty(property);
      if(CompareValues(obj1_prop,obj2_prop,LESS)) index=i;
     }
   return index;
  }
//+------------------------------------------------------------------+
//| Return the MQL5 signal index in the list                         |
//| with the minimum real property value                             |
//+------------------------------------------------------------------+
int CSelect::FindMQLSignalMin(CArrayObj* list_source,ENUM_SIGNAL_MQL5_PROP_DOUBLE property)
  {
   int index=0;
   CMQLSignal *min_obj=NULL;
   int total=list_source.Total();
   if(total== 0) return WRONG_VALUE;
   for(int i=1; i<total; i++)
     {
      CMQLSignal *obj=list_source.At(i);
      double obj1_prop=obj.GetProperty(property);
      min_obj=list_source.At(index);
      double obj2_prop=min_obj.GetProperty(property);
      if(CompareValues(obj1_prop,obj2_prop,LESS)) index=i;
     }
   return index;
  }
//+------------------------------------------------------------------+
//| Return the MQL5 signal index in the list                         |
//| with the minimum string property value                           |
//+------------------------------------------------------------------+
int CSelect::FindMQLSignalMin(CArrayObj* list_source,ENUM_SIGNAL_MQL5_PROP_STRING property)
  {
   int index=0;
   CMQLSignal *min_obj=NULL;
   int total=list_source.Total();
   if(total==0) return WRONG_VALUE;
   for(int i=1; i<total; i++)
     {
      CMQLSignal *obj=list_source.At(i);
      string obj1_prop=obj.GetProperty(property);
      min_obj=list_source.At(index);
      string obj2_prop=min_obj.GetProperty(property);
      if(CompareValues(obj1_prop,obj2_prop,LESS)) index=i;
     }
   return index;
  }
//+------------------------------------------------------------------+
```

As already mentioned, all these methods (that are identical for each of the library object classes) were considered many times. See the [article 3](https://www.mql5.com/en/articles/5687) for more details.

Now everything is ready for the development of the collection class of MQL5.com Signals service signals objects.

### Collection class of MQL5 signal objects

In \\MQL5\\Include\\DoEasy\ **Collections\** library folder, create the new class CMQLSignalsCollection in **MQLSignalsCollection.mqh**.

**In the class file, include all class files necessary for its work:**

```
//+------------------------------------------------------------------+
//|                                         MQLSignalsCollection.mqh |
//|                        Copyright 2021, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2021, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include "ListObj.mqh"
#include "..\Services\Select.mqh"
#include "..\Objects\MQLSignalBase\MQLSignal.mqh"
//+------------------------------------------------------------------+
```

**The class should be derived from the base object of all library objects:**

```
//+------------------------------------------------------------------+
//| MQL5 signal object collection                                    |
//+------------------------------------------------------------------+
class CMQLSignalsCollection : public CBaseObj
  {
  }
```

Let's have a look at the class body and analyze the methods it consists of:

```
//+------------------------------------------------------------------+
//|                                         MQLSignalsCollection.mqh |
//|                        Copyright 2021, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2021, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include "ListObj.mqh"
#include "..\Services\Select.mqh"
#include "..\Objects\MQLSignalBase\MQLSignal.mqh"
//+------------------------------------------------------------------+
//| MQL5 signal object collection                                    |
//+------------------------------------------------------------------+
class CMQLSignalsCollection : public CBaseObj
  {
private:
   CListObj                m_list;                                   // List of MQL5 signal objects
   int                     m_signals_base_total;                     // Number of signals in the MQL5 signal database
//--- Subscribe to a signal
   bool                    Subscribe(const long signal_id);
//--- Set the flag allowing synchronization without confirmation dialog
   bool                    CurrentSetConfirmationsDisableFlag(const bool flag);
//--- Set the flag of copying Stop Loss and Take Profit
   bool                    CurrentSetSLTPCopyFlag(const bool flag);
//--- Set the flag allowing the copying of signals by subscription
   bool                    CurrentSetSubscriptionEnabledFlag(const bool flag);
public:
//--- Return (1) itself and (2) the DOM series collection list
   CMQLSignalsCollection  *GetObject(void)                              { return &this;                  }
   CArrayObj              *GetList(void)                                { return &this.m_list;           }
   //--- Return the list by selected (1) integer, (2) real and (3) string properties meeting the compared criterion
   CArrayObj              *GetList(ENUM_SIGNAL_MQL5_PROP_INTEGER property,long value,ENUM_COMPARER_TYPE mode=EQUAL)  { return CSelect::ByMQLSignalProperty(this.GetList(),property,value,mode);  }
   CArrayObj              *GetList(ENUM_SIGNAL_MQL5_PROP_DOUBLE property,double value,ENUM_COMPARER_TYPE mode=EQUAL) { return CSelect::ByMQLSignalProperty(this.GetList(),property,value,mode);  }
   CArrayObj              *GetList(ENUM_SIGNAL_MQL5_PROP_STRING property,string value,ENUM_COMPARER_TYPE mode=EQUAL) { return CSelect::ByMQLSignalProperty(this.GetList(),property,value,mode);  }
//--- Return the number of MQL5 signal objects in the list
   int                     DataTotal(void)                        const { return this.m_list.Total();    }
//--- Return the pointer to the MQL5 signal object (1) by ID, (2) by name and (3) by index in the list
   CMQLSignal             *GetMQLSignal(const long id);
   CMQLSignal             *GetMQLSignal(const string name);
   CMQLSignal             *GetMQLSignal(const int index)                { return this.m_list.At(index);  }

//--- Create the collection list of MQL5 signal objects
   bool                    CreateCollection(void);
//--- Update the collection list of MQL5 signal objects
   void                    Refresh(const bool messages=true);

//--- Display (1) the complete and (2) short collection description in the journal
   void                    Print(void);
   void                    PrintShort(const bool list=false,const bool paid=true,const bool free=true);

//--- Constructor
                           CMQLSignalsCollection();
//--- Subscribe to a signal by (1) ID and (2) signal name
   bool                    SubscribeByID(const long signal_id);
   bool                    SubscribeByName(const string signal_name);

//+----------------------------------------------------------------------------+
//| Methods of working with the current signal the subscription is active for  |
//+----------------------------------------------------------------------------+
//--- Return the flag allowing working with the signal service
   bool                    ProgramIsAllowed(void)                 { return (bool)::MQLInfoInteger(MQL_SIGNALS_ALLOWED); }
//--- Unsubscribe from the current signal
   bool                    CurrentUnsubscribe(void);

//--- Set the percentage for converting deal volume
   bool                    CurrentSetEquityLimit(const double value);
//--- Set the market order slippage used when synchronizing positions and copying deals
   bool                    CurrentSetSlippage(const double value);
//--- Set deposit limitations (in %)
   bool                    CurrentSetDepositPercent(const int value);

//--- Return the percentage for converting deal volume
   double                  CurrentEquityLimit(void)               { return ::SignalInfoGetDouble(SIGNAL_INFO_EQUITY_LIMIT);                  }
//--- Return the market order slippage used when synchronizing positions and copying deals
   double                  CurrentSlippage(void)                  { return ::SignalInfoGetDouble(SIGNAL_INFO_SLIPPAGE);                      }
//--- Return the flag allowing synchronization without confirmation dialog
   bool                    CurrentConfirmationsDisableFlag(void)  { return (bool)::SignalInfoGetInteger(SIGNAL_INFO_CONFIRMATIONS_DISABLED); }
//--- Return the flag of copying Stop Loss and Take Profit
   bool                    CurrentSLTPCopyFlag(void)              { return (bool)::SignalInfoGetInteger(SIGNAL_INFO_COPY_SLTP);              }
//--- Return deposit limitations (in %)
   int                     CurrentDepositPercent(void)            { return (int)::SignalInfoGetInteger(SIGNAL_INFO_DEPOSIT_PERCENT);         }
//--- Return the flag allowing the copying of signals by subscription
   bool                    CurrentSubscriptionEnabledFlag(void)   { return (bool)::SignalInfoGetInteger(SIGNAL_INFO_SUBSCRIPTION_ENABLED);   }
//--- Return the limitation by funds for a signal
   double                  CurrentVolumePercent(void)             { return ::SignalInfoGetDouble(SIGNAL_INFO_VOLUME_PERCENT);                }
//--- Return the signal ID
   long                    CurrentID(void)                        { return ::SignalInfoGetInteger(SIGNAL_INFO_ID);                           }
//--- Return the flag of agreeing to the terms of use of the Signals service
   bool                    CurrentTermsAgreeFlag(void)            { return (bool)::SignalInfoGetInteger(SIGNAL_INFO_TERMS_AGREE);            }
//--- Return the signal name
   string                  CurrentName(void)                      { return ::SignalInfoGetString(SIGNAL_INFO_NAME);                          }

//--- Enable synchronization without the confirmation dialog
   bool                    CurrentSetConfirmationsDisableON(void) { return this.CurrentSetConfirmationsDisableFlag(true);                    }
//--- Disable synchronization without the confirmation dialog
   bool                    CurrentSetConfirmationsDisableOFF(void){ return this.CurrentSetConfirmationsDisableFlag(false);                   }
//--- Enable copying Stop Loss and Take Profit
   bool                    CurrentSetSLTPCopyON(void)             { return this.CurrentSetSLTPCopyFlag(true);                                }
//--- Disable copying Stop Loss and Take Profit
   bool                    CurrentSetSLTPCopyOFF(void)            { return this.CurrentSetSLTPCopyFlag(false);                               }
//--- Enable copying deals by subscription
   bool                    CurrentSetSubscriptionEnableON(void)   { return this.CurrentSetSubscriptionEnabledFlag(true);                     }
//--- Disable copying deals by subscription
   bool                    CurrentSetSubscriptionEnableOFF(void)  { return this.CurrentSetSubscriptionEnabledFlag(false);                    }

//--- Return the description of enabling working with signals for the launched program
   string                  ProgramIsAllowedDescription(void);
//--- Return the percentage description for converting the deal volume
   string                  CurrentEquityLimitDescription(void);
//--- Return the description of the market order slippage used when synchronizing positions and copying deals
   string                  CurrentSlippageDescription(void);
//--- Return the description of the limitation by funds for a signal
   string                  CurrentVolumePercentDescription(void);
//--- Return the description of the flag allowing synchronization without confirmation dialog
   string                  CurrentConfirmationsDisableFlagDescription(void);
//--- Return the description of the flag of copying Stop Loss and Take Profit
   string                  CurrentSLTPCopyFlagDescription(void);
//--- Return the description of the deposit limitations (in %)
   string                  CurrentDepositPercentDescription(void);
//--- Return the description of the flag allowing the copying of signals by subscription
   string                  CurrentSubscriptionEnabledFlagDescription(void);
//--- Return the description of the signal ID
   string                  CurrentIDDescription(void);
//--- Return the description of the flag of agreeing to the terms of use of the Signals service
   string                  CurrentTermsAgreeFlagDescription(void);
//--- Return the description of the signal name
   string                  CurrentNameDescription(void);
//--- Display the parameters of signal copying settings in the journal
   void                    CurrentSubscriptionParameters(void);
//---
  };
//+------------------------------------------------------------------+
```

The private class section features the list object, which is to store MQL5 signal objects, as well as auxiliary variables and methods.

The public class section features standard methods of working with the object collection list, as well as two methods for subscribing to a selected signal by its ID and name. Also, the public class section features the methods for working with the current signal the subscription is active for.

Let's have a look at the implementation of some methods.

**In the class constructor,** clear the collection list, set the sorted list flag for it, set the ID of the MQL5 signal object collection for the list, write the total number of signals in the MQL5.com Signals database and call the collection creation method.

```
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CMQLSignalsCollection::CMQLSignalsCollection()
  {
   this.m_list.Clear();
   this.m_list.Sort();
   this.m_list.Type(COLLECTION_MQL5_SIGNALS_ID);
   this.m_signals_base_total=::SignalBaseTotal();
   this.CreateCollection();
  }
//+------------------------------------------------------------------+
```

Since we are not going to auto update the signal list, nor manage it by means of the library, the list update method is sufficient. All signals present in the database are read and sent to the collection list in the method. Users will have to call the Refresh() update method on their own before receiving any data from the collection if they want to have an updated signal list from the MQL5.com Signals database. However, we will have the collection creation method anyway to provide compatibility with a set of typical library collection methods. The method itself will simply clear the list and call the collection update method. After the first call of the Refresh() method from the collection creation method, the collection list is filled in and can be handled. If the collection list should be updated in search of possible new signals, simply call the Refresh() method before accessing the collection list.

**The collection creation method:**

```
//+------------------------------------------------------------------+
//| Create the collection list of MQL5 signal objects                |
//+------------------------------------------------------------------+
bool CMQLSignalsCollection::CreateCollection(void)
  {
   this.m_list.Clear();
   this.Refresh(false);
   if(m_list.Total()>0)
     {
      ::Print(CMessage::Text(MSG_MQLSIG_COLLECTION_TEXT_MQL5_SIGNAL_COLLECTION)," ",CMessage::Text(MSG_LIB_TEXT_TS_TEXT_CREATED_OK));
      return true;
     }
   return false;
  }
//+------------------------------------------------------------------+
```

Here we clear the signal collection list and fill in the list with signals from the MQL5.com Signals database. If the number of signals in the collection list exceeds zero (the list is filled), display the message about the successful creation of the collection list and return true.

Otherwise, return false.

**The method of updating the collection list:**

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
      //--- Set the sorting flag for the list by signal ID and,
      //--- if such a signal is already present in the collection list,
      //--- remove the created object and go to the next loop iteration
      m_list.Sort(SORT_BY_SIGNAL_MQL5_ID);
      if(this.m_list.Search(signal)!=WRONG_VALUE)
        {
         delete signal;
         continue;
        }
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

The method logic is described in detail in the code comments. In short: the method receives the flag indicating the necessity to inform of a newly detected signal. Since the method does not clear the collection list, only a newly found signal can be added to it. If the message flag is set, the journal displays a message about a newly detected signal in case a new signal object is successfully added to the list.

Currently, this is the simplest method that does not feature updating the parameters of existing signals — you will be able to update them on your own in your programs using the access to the signal object by its ID and setting new values to its properties. Later, I will add the auto update of the existing signals parameters by time. If the MQL5.com Signals collection class is in demand, I will implement sending events about new signals and changing the parameters of tracked signals.

**The method returning the pointer to an MQL5 signal object by a signal ID:**

```
//+------------------------------------------------------------------+
//| Return the pointer to an MQL5 signal object by an ID             |
//+------------------------------------------------------------------+
CMQLSignal *CMQLSignalsCollection::GetMQLSignal(const long id)
  {
   CArrayObj *list=GetList(SIGNAL_MQL5_PROP_ID,id,EQUAL);
   return(list!=NULL ? list.At(0) : NULL);
  }
//+------------------------------------------------------------------+
```

Get the list of MQL5 signal objects by a signal ID and return either a single object from the obtained list, or NULL.

**The method returning the pointer to an MQL5 signal object by a signal name:**

```
//+------------------------------------------------------------------+
//| Return the pointer to an MQL5 signal object by a name            |
//+------------------------------------------------------------------+
CMQLSignal *CMQLSignalsCollection::GetMQLSignal(const string name)
  {
   CArrayObj *list=GetList(SIGNAL_MQL5_PROP_NAME,name,EQUAL);
   return(list!=NULL ? list.At(0) : NULL);
  }
//+------------------------------------------------------------------+
```

Get the list of MQL5 signal objects by a signal name and return either a single object from the obtained list, or NULL.

**The method returning the full collection list to the journal:**

```
//+------------------------------------------------------------------+
//| Display complete collection description to the journal           |
//+------------------------------------------------------------------+
void CMQLSignalsCollection::Print(void)
  {
   ::Print(CMessage::Text(MSG_MQLSIG_COLLECTION_TEXT_MQL5_SIGNAL_COLLECTION),":");
   for(int i=0;i<this.m_list.Total();i++)
     {
      CMQLSignal *signal=this.m_list.At(i);
      if(signal==NULL)
         continue;
      signal.Print();
     }
  }
//+------------------------------------------------------------------+
```

The header is displayed first. Then, in the loop by the collection list, get the next MQL signal object and display its full description.

**The method returning the short collection list to the journal:**

```
//+------------------------------------------------------------------+
//| Display the short collection description in the journal          |
//+------------------------------------------------------------------+
void CMQLSignalsCollection::PrintShort(const bool list=false,const bool paid=true,const bool free=true)
  {
   //--- Display the header in the journal
   ::Print(CMessage::Text(MSG_MQLSIG_COLLECTION_TEXT_MQL5_SIGNAL_COLLECTION),":");
   //--- If the list is full, display short descriptions of all signals in the collection
   //--- according to the flags indicating the necessity to display paid and free signals
   if(list)
      for(int i=0;i<this.m_list.Total();i++)
        {
         CMQLSignal *signal=this.m_list.At(i);
         if(signal==NULL || (signal.Price()>0 && !paid) || (signal.Price()==0 && !free))
            continue;
         signal.PrintShort(true);
        }
   //--- If not the signal list
   else
     {
      //--- Sort the list by signal price
      this.m_list.Sort(SORT_BY_SIGNAL_MQL5_PRICE);
      //--- Get the list of free signals and their number
      CArrayObj *list_free=this.GetList(SIGNAL_MQL5_PROP_PRICE,0,EQUAL);
      int num_free=(list_free==NULL ? 0 : list_free.Total());
      //--- Sort the list by signal price
      this.m_list.Sort(SORT_BY_SIGNAL_MQL5_PRICE);
      //--- Get the list of paid signals and their number
      CArrayObj *list_paid=this.GetList(SIGNAL_MQL5_PROP_PRICE,0,MORE);
      int num_paid=(list_paid==NULL ? 0 : list_paid.Total());
      //--- Display the number of free and paid signals in the collection in the journal
      ::Print
        (
         "- ",CMessage::Text(MSG_MQLSIG_COLLECTION_TEXT_SIGNALS_FREE),": ",(string)num_free,
         ", ",CMessage::Text(MSG_MQLSIG_COLLECTION_TEXT_SIGNALS_PAID),": ",(string)num_paid
        );
     }
  }
//+------------------------------------------------------------------+
```

Depending on the passed flags, the method displays various messages and lists in the journal.

The header comes first. If the list flag is set, the journal displays short descriptions of signals from the collection. The flags of paid and free signals are considered. Depending on their status, the journal display either all signals, or only paid ones, or only free ones.

If the description should be displayed not as a list, the header is followed by the total number of free and paid signals in the collection list.

**The methods subscribing to a signal (private method) and unsubscribing from it (public method):**

```
//+------------------------------------------------------------------+
//| Subscribe to a signal                                            |
//+------------------------------------------------------------------+
bool CMQLSignalsCollection::Subscribe(const long signal_id)
  {
//--- If working with signals is disabled for a program,
//--- display the appropriate message and the recommendation to check the program settings
   if(!this.ProgramIsAllowed())
     {
      ::Print(DFUN,CMessage::Text(MSG_SIGNAL_INFO_ERR_SIGNAL_NOT_ALLOWED));
      ::Print(DFUN,CMessage::Text(MSG_SIGNAL_INFO_TEXT_CHECK_SETTINGS));
      return false;
     }
//--- If failed to subscribe to a signal, display the error message and return 'false'
   ::ResetLastError();
   if(!::SignalSubscribe(signal_id))
     {
      CMessage::ToLog(DFUN,::GetLastError(),true);
      return false;
     }
//--- Subscription successful. Display the successful signal subscription message and return 'true'
   ::Print(CMessage::Text(MSG_SIGNAL_INFO_TEXT_SIGNAL_SUBSCRIBED)," ID ",(string)this.CurrentID()," \"",CurrentName(),"\"");
   return true;
  }
//+------------------------------------------------------------------+
//| Unsubscribe from a subscribed signal                             |
//+------------------------------------------------------------------+
bool CMQLSignalsCollection::CurrentUnsubscribe(void)
  {
//--- If working with signals is disabled for a program,
//--- display the appropriate message and the recommendation to check the program settings
   if(!this.ProgramIsAllowed())
     {
      ::Print(DFUN,CMessage::Text(MSG_SIGNAL_INFO_ERR_SIGNAL_NOT_ALLOWED));
      ::Print(DFUN,CMessage::Text(MSG_SIGNAL_INFO_TEXT_CHECK_SETTINGS));
      return false;
     }
//--- Remember an ID and a name of the current signal
   ::ResetLastError();
   long id=this.CurrentID();
   string name=this.CurrentName();
//--- If the ID is zero (no subscription), return 'true'
   if(id==0)
      return true;
//--- If failed to unsubscribe from a signal, display the error message and return 'false'
   if(!::SignalUnsubscribe())
     {
      CMessage::ToLog(DFUN,::GetLastError(),true);
      return false;
     }
//--- Unsubscribed from a signal successfully. Display the message about successfully unsubscribing from a signal and return 'true'
   ::Print(CMessage::Text(MSG_SIGNAL_INFO_TEXT_SIGNAL_UNSUBSCRIBED)," ID ",(string)id," \"",name,"\"");
   return true;
  }
//+------------------------------------------------------------------+
```

The method logic is described in detail in the method listing.

**The public method performing a subscription to a signal by signal ID:**

```
//+------------------------------------------------------------------+
//| Subscribe to a signal by ID                                      |
//+------------------------------------------------------------------+
bool CMQLSignalsCollection::SubscribeByID(const long signal_id)
  {
   CMQLSignal *signal=GetMQLSignal(signal_id);
   if(signal==NULL)
     {
      ::Print(DFUN,CMessage::Text(MSG_MQLSIG_COLLECTION_ERR_FAILED_GET_SIGNAL),": ",signal_id);
      return false;
     }
   return this.Subscribe(signal.ID());
  }
//+------------------------------------------------------------------+
```

Here we obtain the pointer to the MQL5 signal object in the collection list by the ID passed to the method and return the result of the private signal subscription method considered above.

**The public method performing a subscription to a signal by signal name:**

```
//+------------------------------------------------------------------+
//| Subscribe to a signal by signal name                             |
//+------------------------------------------------------------------+
bool CMQLSignalsCollection::SubscribeByName(const string signal_name)
  {
   CMQLSignal *signal=GetMQLSignal(signal_name);
   if(signal==NULL)
     {
      ::Print(DFUN,CMessage::Text(MSG_MQLSIG_COLLECTION_ERR_FAILED_GET_SIGNAL),": \"",signal_name,"\"");
      return false;
     }
   return this.Subscribe(signal.ID());
  }
//+------------------------------------------------------------------+
```

Here we obtain the pointer to the MQL5 signal object in the collection list by a signal name passed to the method (the name should be known in advance) and return the result of the private signal subscription method considered above.

**The methods of setting the values of copying trading signals:**

```
//+------------------------------------------------------------------+
//| Set the percentage for converting a deal volume                  |
//+------------------------------------------------------------------+
bool CMQLSignalsCollection::CurrentSetEquityLimit(const double value)
  {
   ::ResetLastError();
   if(!::SignalInfoSetDouble(SIGNAL_INFO_EQUITY_LIMIT,value))
     {
      CMessage::ToLog(DFUN,::GetLastError(),true);
      return false;
     }
   return true;
  }
//+------------------------------------------------------------------+
//| Define the slippage used to set                                  |
//| market orders when synchronizing positions and copying deals     |
//+------------------------------------------------------------------+
bool CMQLSignalsCollection::CurrentSetSlippage(const double value)
  {
   ::ResetLastError();
   if(!::SignalInfoSetDouble(SIGNAL_INFO_SLIPPAGE,value))
     {
      CMessage::ToLog(DFUN,::GetLastError(),true);
      return false;
     }
   return true;
  }
//+------------------------------------------------------------------+
//| Set the flag enabling synchronization                            |
//| without confirmation dialog                                      |
//+------------------------------------------------------------------+
bool CMQLSignalsCollection::CurrentSetConfirmationsDisableFlag(const bool flag)
  {
   ::ResetLastError();
   if(!::SignalInfoSetInteger(SIGNAL_INFO_CONFIRMATIONS_DISABLED,flag))
     {
      CMessage::ToLog(DFUN,::GetLastError(),true);
      return false;
     }
   return true;
  }
//+------------------------------------------------------------------+
//| Set the flag of copying Stop Loss and Take Profit                |
//+------------------------------------------------------------------+
bool CMQLSignalsCollection::CurrentSetSLTPCopyFlag(const bool flag)
  {
   ::ResetLastError();
   if(!::SignalInfoSetInteger(SIGNAL_INFO_COPY_SLTP,flag))
     {
      CMessage::ToLog(DFUN,::GetLastError(),true);
      return false;
     }
   return true;
  }
//+------------------------------------------------------------------+
//| Set deposit limitations (in %)                                   |
//+------------------------------------------------------------------+
bool CMQLSignalsCollection::CurrentSetDepositPercent(const int value)
  {
   ::ResetLastError();
   if(!::SignalInfoSetInteger(SIGNAL_INFO_DEPOSIT_PERCENT,value))
     {
      CMessage::ToLog(DFUN,::GetLastError(),true);
      return false;
     }
   return true;
  }
//+------------------------------------------------------------------+
//| Set the flag allowing the copying of signals by subscription     |
//+------------------------------------------------------------------+
bool CMQLSignalsCollection::CurrentSetSubscriptionEnabledFlag(const bool flag)
  {
   ::ResetLastError();
   if(!::SignalInfoSetInteger(SIGNAL_INFO_SUBSCRIPTION_ENABLED,flag))
     {
      CMessage::ToLog(DFUN,::GetLastError(),true);
      return false;
     }
   return true;
  }
//+------------------------------------------------------------------+
```

The functions for setting values [SignalInfoSetDouble()](https://www.mql5.com/en/docs/signals/signalinfosetdouble) and [SignalInfoSetInteger()](https://www.mql5.com/en/docs/signals/signalinfosetinteger) are used here in all methods. If setting a value is unsuccessful, the methods show an error description and return false. If setting is successful, the methods return true.

**The methods returning descriptions of the parameters for setting trading signals copying:**

```
//+------------------------------------------------------------------+
//| Return the description of the permission to work with signals    |
//| for the currently launched program                               |
//+------------------------------------------------------------------+
string CMQLSignalsCollection::ProgramIsAllowedDescription(void)
  {
   return
     (
      CMessage::Text(MSG_SIGNAL_INFO_SIGNALS_PERMISSION)+": "+
      (this.ProgramIsAllowed() ? CMessage::Text(MSG_LIB_TEXT_YES) : CMessage::Text(MSG_LIB_TEXT_NO))
     );
  }
//+------------------------------------------------------------------+
//| Return the percentage description for converting the deal volume |
//+------------------------------------------------------------------+
string CMQLSignalsCollection::CurrentEquityLimitDescription(void)
  {
   return CMessage::Text(MSG_SIGNAL_INFO_EQUITY_LIMIT)+": "+::DoubleToString(this.CurrentEquityLimit(),2)+"%";
  }
//+------------------------------------------------------------------+
//| Return the description of the slippage used to set               |
//| market orders when synchronizing positions and copying deals     |
//+------------------------------------------------------------------+
string CMQLSignalsCollection::CurrentSlippageDescription(void)
  {
   return CMessage::Text(MSG_SIGNAL_INFO_SLIPPAGE)+": "+CMessage::Text(MSG_LIB_TEXT_BAR_SPREAD)+" * "+::DoubleToString(this.CurrentSlippage(),2);
  }
//+------------------------------------------------------------------+
//| Return the description of the limitation by funds for a signal   |
//+------------------------------------------------------------------+
string CMQLSignalsCollection::CurrentVolumePercentDescription(void)
  {
   return CMessage::Text(MSG_SIGNAL_INFO_VOLUME_PERCENT)+": "+::DoubleToString(this.CurrentVolumePercent(),2)+"%";
  }
//+------------------------------------------------------------------+
//| Return the description of the flag enabling synchronization      |
//| without confirmation dialog                                      |
//+------------------------------------------------------------------+
string CMQLSignalsCollection::CurrentConfirmationsDisableFlagDescription(void)
  {
   return
     (
      CMessage::Text(MSG_SIGNAL_INFO_CONFIRMATIONS_DISABLED)+": "+
      (this.CurrentConfirmationsDisableFlag() ? CMessage::Text(MSG_LIB_TEXT_YES) : CMessage::Text(MSG_LIB_TEXT_NO))
     );
  }
//+----------------------------------------------------------------------------+
//| Return the description of the flag of copying Stop Loss and Take Profit    |
//+----------------------------------------------------------------------------+
string CMQLSignalsCollection::CurrentSLTPCopyFlagDescription(void)
  {
   return
     (
      CMessage::Text(MSG_SIGNAL_INFO_COPY_SLTP)+": "+
      (this.CurrentSLTPCopyFlag() ? CMessage::Text(MSG_LIB_TEXT_YES) : CMessage::Text(MSG_LIB_TEXT_NO))
     );
  }
//+------------------------------------------------------------------+
//| Return the description of the deposit limitations (in %)         |
//+------------------------------------------------------------------+
string CMQLSignalsCollection::CurrentDepositPercentDescription(void)
  {
   return CMessage::Text(MSG_SIGNAL_INFO_DEPOSIT_PERCENT)+": "+(string)this.CurrentDepositPercent()+"%";
  }
//+------------------------------------------------------------------+
//| Return the description of the flag enabling                      |
//| deal copying by subscription                                     |
//+------------------------------------------------------------------+
string CMQLSignalsCollection::CurrentSubscriptionEnabledFlagDescription(void)
  {
   return
     (
      CMessage::Text(MSG_SIGNAL_INFO_SUBSCRIPTION_ENABLED)+": "+
      (this.CurrentSubscriptionEnabledFlag() ? CMessage::Text(MSG_LIB_TEXT_YES) : CMessage::Text(MSG_LIB_TEXT_NO))
     );
  }
//+------------------------------------------------------------------+
//| Return the signal ID description                                 |
//+------------------------------------------------------------------+
string CMQLSignalsCollection::CurrentIDDescription(void)
  {
   return CMessage::Text(MSG_SIGNAL_INFO_ID)+": "+(this.CurrentID()>0 ? (string)this.CurrentID() : CMessage::Text(MSG_LIB_PROP_EMPTY));
  }
//+------------------------------------------------------------------+
//| Return the description of the flag indicating                    |
//| consent to the terms of use of the Signals service               |
//+------------------------------------------------------------------+
string CMQLSignalsCollection::CurrentTermsAgreeFlagDescription(void)
  {
   return
     (
      CMessage::Text(MSG_SIGNAL_INFO_TERMS_AGREE)+": "+
      (this.CurrentTermsAgreeFlag() ? CMessage::Text(MSG_LIB_TEXT_YES) : CMessage::Text(MSG_LIB_TEXT_NO))
     );
  }
//+------------------------------------------------------------------+
//| Return the description of the signal name                        |
//+------------------------------------------------------------------+
string CMQLSignalsCollection::CurrentNameDescription(void)
  {
   return CMessage::Text(MSG_SIGNAL_INFO_NAME)+": "+(this.CurrentName()!="" ? this.CurrentName() : CMessage::Text(MSG_LIB_PROP_EMPTY));
  }
//+------------------------------------------------------------------+
```

The string featuring the parameter description header and its current value is created in each method.

**The method displaying the parameters of trading signal copying settings in the journal:**

```
//+------------------------------------------------------------------+
//| Display the parameters of signal copying settings in the journal |
//+------------------------------------------------------------------+
void CMQLSignalsCollection::CurrentSubscriptionParameters(void)
  {
   ::Print("============= ",CMessage::Text(MSG_SIGNAL_INFO_PARAMETERS)," =============");
   ::Print(this.ProgramIsAllowedDescription());
   ::Print(this.CurrentTermsAgreeFlagDescription());
   ::Print(this.CurrentSubscriptionEnabledFlagDescription());
   ::Print(this.CurrentConfirmationsDisableFlagDescription());
   ::Print(this.CurrentSLTPCopyFlagDescription());
   ::Print(this.CurrentSlippageDescription());
   ::Print(this.CurrentEquityLimitDescription());
   ::Print(this.CurrentDepositPercentDescription());
   ::Print(this.CurrentVolumePercentDescription());
   ::Print(this.CurrentIDDescription());
   ::Print(this.CurrentNameDescription());
   ::Print("");
  }
//+------------------------------------------------------------------+
```

The header is displayed first followed by all trading signal copying parameters shown one by one. The parameters are returned by the appropriate methods considered above.

**This completes the creation of the MQL5 signal object collection class.**

I may improve it later. However, I will leave it this way for now till I am able to determine whether it is in high demand.

To connect the trading signal collection class with the "outside world", we need to complete the development of the methods for working with them to the **CEngine** library main object class in \\MQL5\\Include\\DoEasy\ **Engine.mqh**.

**Collect the file of the trading signal collection class to the CEngine object class file and declare the MQL5 signal collection class object:**

```
//+------------------------------------------------------------------+
//|                                                       Engine.mqh |
//|                        Copyright 2020, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2020, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include "Services\TimerCounter.mqh"
#include "Collections\HistoryCollection.mqh"
#include "Collections\MarketCollection.mqh"
#include "Collections\EventsCollection.mqh"
#include "Collections\AccountsCollection.mqh"
#include "Collections\SymbolsCollection.mqh"
#include "Collections\ResourceCollection.mqh"
#include "Collections\TimeSeriesCollection.mqh"
#include "Collections\BuffersCollection.mqh"
#include "Collections\IndicatorsCollection.mqh"
#include "Collections\TickSeriesCollection.mqh"
#include "Collections\BookSeriesCollection.mqh"
#include "Collections\MQLSignalsCollection.mqh"
#include "TradingControl.mqh"
//+------------------------------------------------------------------+
//| Library basis class                                              |
//+------------------------------------------------------------------+
class CEngine
  {
private:
   CHistoryCollection   m_history;                       // Collection of historical orders and deals
   CMarketCollection    m_market;                        // Collection of market orders and deals
   CEventsCollection    m_events;                        // Event collection
   CAccountsCollection  m_accounts;                      // Account collection
   CSymbolsCollection   m_symbols;                       // Symbol collection
   CTimeSeriesCollection m_time_series;                  // Timeseries collection
   CBuffersCollection   m_buffers;                       // Collection of indicator buffers
   CIndicatorsCollection m_indicators;                   // Indicator collection
   CTickSeriesCollection m_tick_series;                  // Collection of tick series
   CMBookSeriesCollection m_book_series;                 // Collection of DOM series
   CMQLSignalsCollection m_signals_mql5;                 // Collection of MQL5.com Signals service signals
   CResourceCollection  m_resource;                      // Resource list
   CTradingControl      m_trading;                       // Trading management object
   CPause               m_pause;                         // Pause object
   CArrayObj            m_list_counters;                 // List of timer counters
```

**In the public section of the class, declare and implement the new methods for working with the collection class of DOM snapshots and the methods for working with the trading signal collection:**

```
//--- Update the DOM series of a specified symbol
   void                 MBookSeriesRefresh(const string symbol,const long time_msc)    { this.m_book_series.Refresh(symbol,time_msc);        }

//--- Return (1) the DOM series of a specified symbol, the DOM (2) by index and (3) by time in milliseconds
   CMBookSeries        *GetMBookSeries(const string symbol)                            { return this.m_book_series.GetMBookseries(symbol);   }
   CMBookSnapshot      *GetMBook(const string symbol,const int index)                  { return this.m_book_series.GetMBook(symbol,index);   }
   CMBookSnapshot      *GetMBook(const string symbol,const long time_msc)              { return this.m_book_series.GetMBook(symbol,time_msc);}

//--- Return the volume of a (1) buy and (2) sell DOM specified by symbol and index
   long                 MBookVolumeBuy(const string symbol,const int index);
   long                 MBookVolumeSell(const string symbol,const int index);
//--- Return the increased precision volume of a (1) buy and (2) sell DOM specified by symbol and index
   double               MBookVolumeBuyReal(const string symbol,const int index);
   double               MBookVolumeSellReal(const string symbol,const int index);
//--- Return the volume of a (1) buy and (2) sell DOM specified by symbol and time in milliseconds
   long                 MBookVolumeBuy(const string symbol,const long time_msc);
   long                 MBookVolumeSell(const string symbol,const long time_msc);
//--- Return the increased precision volume of a (1) buy and (2) sell DOM specified by symbol and time in milliseconds
   double               MBookVolumeBuyReal(const string symbol,const long time_msc);
   double               MBookVolumeSellReal(const string symbol,const long time_msc);


//--- Return (1) the collection of mql5.com Signals service signals and (2) the list of signals from the mql5.com Signals service signal collection
   CMQLSignalsCollection *GetSignalsMQL5Collection(void)                               { return &this.m_signals_mql5;                        }
   CArrayObj           *GetListSignalsMQL5(void)                                       { return this.m_signals_mql5.GetList();               }
//--- Return the list of (1) paid and (2) free signals
   CArrayObj           *GetListSignalsMQL5Paid(void)                    { return this.m_signals_mql5.GetList(SIGNAL_MQL5_PROP_PRICE,0,MORE); }
   CArrayObj           *GetListSignalsMQL5Free(void)                    { return this.m_signals_mql5.GetList(SIGNAL_MQL5_PROP_PRICE,0,EQUAL);}

//--- (1) Create and (2) update the collection of mql5.com Signals service signals
   bool                 SignalsMQL5Create(void)                                        { return this.m_signals_mql5.CreateCollection();      }
   void                 SignalsMQL5Refresh(void)                                       { this.m_signals_mql5.Refresh();                      }
//--- Subscribe to a signal by (1) ID and (2) signal name
   bool                 SignalsMQL5Subscribe(const long signal_id)                     { return this.m_signals_mql5.SubscribeByID(signal_id);}
   bool                 SignalsMQL5Subscribe(const string signal_name)                 { return this.m_signals_mql5.SubscribeByName(signal_name);}
//--- Unsubscribe from the current signal
   bool                 SignalsMQL5Unsubscribe(void)                                   { return this.m_signals_mql5.CurrentUnsubscribe();    }
//--- Return (1) ID and (2) the name of the current signal subscription is performed to
   long                 SignalsMQL5CurrentID(void)                                     { return this.m_signals_mql5.CurrentID();             }
   string               SignalsMQL5CurrentName(void)                                   { return this.m_signals_mql5.CurrentName();           }

//--- Set the percentage for converting deal volume
   bool                 SignalsMQL5CurrentSetEquityLimit(const double value)  { return this.m_signals_mql5.CurrentSetEquityLimit(value);     }
//--- Set the market order slippage used when synchronizing positions and copying deals
   bool                 SignalsMQL5CurrentSetSlippage(const double value)     { return this.m_signals_mql5.CurrentSetSlippage(value);        }
//--- Set deposit limitations (in %)
   bool                 SignalsMQL5CurrentSetDepositPercent(const int value)  { return this.m_signals_mql5.CurrentSetDepositPercent(value);  }
//--- Enable synchronization without the confirmation dialog
   bool                 SignalsMQL5CurrentSetConfirmationsDisableON(void)     { return this.m_signals_mql5.CurrentSetConfirmationsDisableON();}
//--- Disable synchronization without the confirmation dialog
   bool                 SignalsMQL5CurrentSetConfirmationsDisableOFF(void)    { return this.m_signals_mql5.CurrentSetConfirmationsDisableOFF();}
//--- Enable copying Stop Loss and Take Profit
   bool                 SignalsMQL5CurrentSetSLTPCopyON(void)                 { return this.m_signals_mql5.CurrentSetSLTPCopyON();           }
//--- Disable copying Stop Loss and Take Profit
   bool                 SignalsMQL5CurrentSetSLTPCopyOFF(void)                { return this.m_signals_mql5.CurrentSetSLTPCopyOFF();          }
//--- Enable copying deals by subscription
   bool                 SignalsMQL5CurrentSetSubscriptionEnableON(void)       { return this.m_signals_mql5.CurrentSetSubscriptionEnableON(); }
//--- Disable copying deals by subscription
   bool                 SignalsMQL5CurrentSetSubscriptionEnableOFF(void)      { return this.m_signals_mql5.CurrentSetSubscriptionEnableOFF();}

//--- Display (1) the complete, (2) short collection description in the journal and (3) parameters of the signal copying settings
   void                 SignalsMQL5Print(void)                                         { m_signals_mql5.Print();                             }
   void                 SignalsMQL5PrintShort(const bool list=false,const bool paid=true,const bool free=true)
                                                                                       { m_signals_mql5.PrintShort(list,paid,free);          }
   void                 SignalsMQL5CurrentSubscriptionParameters(void)                 { this.m_signals_mql5.CurrentSubscriptionParameters();}

//--- Return (1) the buffer collection and (2) the buffer list from the collection
```

The implemented methods return the result of calling the methods of corresponding collections of the same name.

**Let's have a look at implementing the methods returning the specified volumes of specified DOM snapshots:**

```
//+------------------------------------------------------------------+
//| Return the buy volume of a DOM                                   |
//| specified by symbol and index                                    |
//+------------------------------------------------------------------+
long CEngine::MBookVolumeBuy(const string symbol,const int index)
  {
   CMBookSnapshot *mbook=this.GetMBook(symbol,index);
   return(mbook!=NULL ? mbook.VolumeBuy() : 0);
  }
//+------------------------------------------------------------------+
//| Return the sell volume of a DOM                                  |
//| specified by symbol and index                                    |
//+------------------------------------------------------------------+
long CEngine::MBookVolumeSell(const string symbol,const int index)
  {
   CMBookSnapshot *mbook=this.GetMBook(symbol,index);
   return(mbook!=NULL ? mbook.VolumeSell() : 0);
  }
//+------------------------------------------------------------------+
//| Return the extended accuracy buy volume                          |
//| of a DOM specified by symbol and index                           |
//+------------------------------------------------------------------+
double CEngine::MBookVolumeBuyReal(const string symbol,const int index)
  {
   CMBookSnapshot *mbook=this.GetMBook(symbol,index);
   return(mbook!=NULL ? mbook.VolumeBuyReal() : 0);
  }
//+------------------------------------------------------------------+
//| Return the extended accuracy sell volume                         |
//| of a DOM specified by symbol and index                           |
//+------------------------------------------------------------------+
double CEngine::MBookVolumeSellReal(const string symbol,const int index)
  {
   CMBookSnapshot *mbook=this.GetMBook(symbol,index);
   return(mbook!=NULL ? mbook.VolumeSellReal() : 0);
  }
//+------------------------------------------------------------------+
//| Return the buy volume of a DOM                                   |
//| specified by symbol and time in milliseconds                     |
//+------------------------------------------------------------------+
long CEngine::MBookVolumeBuy(const string symbol,const long time_msc)
  {
   CMBookSnapshot *mbook=this.GetMBook(symbol,time_msc);
   return(mbook!=NULL ? mbook.VolumeBuy() : 0);
  }
//+------------------------------------------------------------------+
//| Return the sell volume of a DOM                                  |
//| specified by symbol and time in milliseconds                     |
//+------------------------------------------------------------------+
long CEngine::MBookVolumeSell(const string symbol,const long time_msc)
  {
   CMBookSnapshot *mbook=this.GetMBook(symbol,time_msc);
   return(mbook!=NULL ? mbook.VolumeSell() : 0);
  }
//+------------------------------------------------------------------+
//| Return the extended accuracy buy volume                          |
//| of a DOM specified by symbol and time in milliseconds            |
//+------------------------------------------------------------------+
double CEngine::MBookVolumeBuyReal(const string symbol,const long time_msc)
  {
   CMBookSnapshot *mbook=this.GetMBook(symbol,time_msc);
   return(mbook!=NULL ? mbook.VolumeBuyReal() : 0);
  }
//+------------------------------------------------------------------+
//| Return the extended accuracy sell volume                         |
//| of a DOM specified by symbol and time in milliseconds            |
//+------------------------------------------------------------------+
double CEngine::MBookVolumeSellReal(const string symbol,const long time_msc)
  {
   CMBookSnapshot *mbook=this.GetMBook(symbol,time_msc);
   return(mbook!=NULL ? mbook.VolumeSellReal() : 0);
  }
//+------------------------------------------------------------------+
```

Everything is simple here. The logic behind all the methods is identical: first, we obtain the DOM snapshot object from the collection list by symbol and index or time in milliseconds using previously implemented GetMBook() methods, next return the DOM volume corresponding to the method from the obtained DOM snapshot object or zero if failed to obtain the object.

**These are all the improvements and changes for today.**

### Test

Let's test creation of the [MQL5.com Signals](https://www.mql5.com/en/signals) collection. The test is to be performed as follows:

Get the full list from the Signals database, display the list of free signals only, find the most profitable signal in the list and subscribe to it. If the subscription is successful, we will display the parameters of the current signal and the parameters for copying trading signals that were set when subscribing to it. Unsubscribe from the current signal on the next tick.

To perform the test, I will use the [EA from the previous article](https://www.mql5.com/en/articles/9095#node05) and save it in \\MQL5\\Experts\\TestDoEasy\ **Part66\** as **TestDoEasyPart66.mq5**.

The list of the EA inputs features the setting allowing users to select working with the MQL5.com Signals service in the EA:

```
//--- input variables
input    ushort            InpMagic             =  123;  // Magic number
input    double            InpLots              =  0.1;  // Lots
input    uint              InpStopLoss          =  150;  // StopLoss in points
input    uint              InpTakeProfit        =  150;  // TakeProfit in points
input    uint              InpDistance          =  50;   // Pending orders distance (points)
input    uint              InpDistanceSL        =  50;   // StopLimit orders distance (points)
input    uint              InpDistancePReq      =  50;   // Distance for Pending Request's activate (points)
input    uint              InpBarsDelayPReq     =  5;    // Bars delay for Pending Request's activate (current timeframe)
input    uint              InpSlippage          =  5;    // Slippage in points
input    uint              InpSpreadMultiplier  =  1;    // Spread multiplier for adjusting stop-orders by StopLevel
input    uchar             InpTotalAttempts     =  5;    // Number of trading attempts
sinput   double            InpWithdrawal        =  10;   // Withdrawal funds (in tester)

sinput   uint              InpButtShiftX        =  0;    // Buttons X shift
sinput   uint              InpButtShiftY        =  10;   // Buttons Y shift

input    uint              InpTrailingStop      =  50;   // Trailing Stop (points)
input    uint              InpTrailingStep      =  20;   // Trailing Step (points)
input    uint              InpTrailingStart     =  0;    // Trailing Start (points)
input    uint              InpStopLossModify    =  20;   // StopLoss for modification (points)
input    uint              InpTakeProfitModify  =  60;   // TakeProfit for modification (points)

sinput   ENUM_SYMBOLS_MODE InpModeUsedSymbols   =  SYMBOLS_MODE_CURRENT;            // Mode of used symbols list
sinput   string            InpUsedSymbols       =  "EURUSD,AUDUSD,EURAUD,EURCAD,EURGBP,EURJPY,EURUSD,GBPUSD,NZDUSD,USDCAD,USDJPY";  // List of used symbols (comma - separator)
sinput   ENUM_TIMEFRAMES_MODE InpModeUsedTFs    =  TIMEFRAMES_MODE_LIST;            // Mode of used timeframes list
sinput   string            InpUsedTFs           =  "M1,M5,M15,M30,H1,H4,D1,W1,MN1"; // List of used timeframes (comma - separator)

sinput   ENUM_INPUT_YES_NO InpUseBook           =  INPUT_YES;                       // Use Depth of Market
sinput   ENUM_INPUT_YES_NO InpUseMqlSignals     =  INPUT_YES;                       // Use signal service
sinput   ENUM_INPUT_YES_NO InpUseSounds         =  INPUT_YES;                       // Use sounds

//--- global variables
```

In the previous article, all checks of working with signals were done in the EA's OnInit() handler. Today, we will work in OnTick().

Therefore, **let's remove the unnecessary test code block from the OnInit() handler:**

```
   CArrayObj *list=new CArrayObj();
   if(list!=NULL)
     {
      //--- request the total number of signals in the signal database
      int total=SignalBaseTotal();
      //--- loop through all signals
      for(int i=0;i<total;i++)
        {
         //--- select a signal for further operation
         if(!SignalBaseSelect(i))
            continue;
         long id=SignalBaseGetInteger(SIGNAL_BASE_ID);
         CMQLSignal *signal=new CMQLSignal(id);
         if(signal==NULL)
            continue;
         if(!list.Add(signal))
           {
            delete signal;
            continue;
           }
        }
      //--- display all profitable free signals with a non-zero number of subscribers
      Print("");
      static bool done=false;
      for(int i=0;i<list.Total();i++)
        {
         CMQLSignal *signal=list.At(i);
         if(signal==NULL)
            continue;
         if(signal.Price()>0 || signal.Subscribers()==0)
            continue;
         //--- The very first suitable signal is fully displayed in the journal
         if(!done)
           {
            signal.Print();
            done=true;
           }
         //--- Short descriptions are displayed for the rest
         else
            signal.PrintShort();
        }
      delete list;
     }

//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
```

**The OnTick() handler receives a new test code block fulfilling all test conditions described at the very start of this section:**

```
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//--- Remove EA graphical objects by an object name prefix
   ObjectsDeleteAll(0,prefix);
   Comment("");
//--- Deinitialize library
   engine.OnDeinit();
  }
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
  }
//+------------------------------------------------------------------+
```

The entire new code block is commented in detail. If you have any questions, ask them in the comments.

**In the OnInitDoEasy() library initialization function, add the code block, in which the collection list of trading signals is created and set the flag enabling copying trading signals by subscription:**

```
//--- Create tick series of all used symbols
   engine.TickSeriesCreateAll();

//--- Check created tick series - display descriptions of all created tick series in the journal
   engine.GetTickSeriesCollection().Print();

//--- Check created DOM series - display descriptions of all created DOM series in the journal
   engine.GetMBookSeriesCollection().Print();

//--- Create the collection of mql5.com Signals service signals
//--- If working with signals is enabled and the signal collection is created
   if(InpUseMqlSignals && engine.SignalsMQL5Create())
     {
      //--- Enable copying deals by subscription
      engine.SignalsMQL5CurrentSetSubscriptionEnableON();
      //--- Check created MQL5 signal objects of the Signals service - display the short collection description in the journal
      engine.SignalsMQL5PrintShort();
     }
//--- If working with signals is not enabled or failed to create the signal collection,
//--- disable copying deals by subscription
   else
      engine.SignalsMQL5CurrentSetSubscriptionEnableOFF();

//--- Create resource text files
```

Compile the EA and launch it on a symbol chart, while preliminarily setting working on the current symbol/timeframe and activating the flag of working with trading signals of the MQL5.com Signals service:

![](https://c.mql5.com/2/42/1M7F4yaHm3.png)

In the Common tab of the EA settings window, check "Allow modification of Signals settings":

![](https://c.mql5.com/2/42/CzXSLVSuK1__1.png)

Otherwise, the EA will not be able to work with the MQL5.com Signals.

After launching the EA, the journal displays the message about the successful creation of the signals collection and its short description:

```
Collection of MQL5.com Signals service signals created successfully
Collection of MQL5.com Signals service signals:
- Free signals: 195, Paid signals: 805
```

The full list of free signals is then displayed. Since they are numerous, I will show only a part of them as an example:

```
Collection of MQL5.com Signals service signals:
- Signal "GBPUSD EXPERT 23233". Author login: mbt_trader, ID 919099, Growth: 3.30, Drawdown: 11.92, Price: 0.00, Subscribers: 0
- Signal "Willian". Author login: Desg, ID 917396, Growth: 12.69, Drawdown: 15.50, Price: 0.00, Subscribers: 0
- Signal "VahidVHZ1366". Author login: 39085485, ID 921427, Growth: 34.36, Drawdown: 12.84, Price: 0.00, Subscribers: 0
- Signal "Vikings". Author login: Myxx, ID 921040, Growth: 7.05, Drawdown: 2.22, Price: 0.00, Subscribers: 2
- Signal "VantageFX Sunphone Dragon". Author login: sunphone, ID 916421, Growth: 537.89, Drawdown: 39.06, Price: 0.00, Subscribers: 21
- Signal "Forex money maker free". Author login: Yggdrasills, ID 916328, Growth: 44.66, Drawdown: 61.15, Price: 0.00, Subscribers: 0
...
...
...
- Signal "Nine Pairs ST". Author login: ebi.pilehvar, ID 935603, Growth: 25.92, Drawdown: 26.41, Price: 0.00, Subscribers: 2
- Signal "FBS140". Author login: mohammeeeedali, ID 949720, Growth: 42.14, Drawdown: 23.11, Price: 0.00, Subscribers: 2
- Signal "StopTheFourthAddition". Author login: pinheirodps, ID 934990, Growth: 41.78, Drawdown: 28.03, Price: 0.00, Subscribers: 2
- Signal "The art of Forex". Author login: Myxx, ID 801685, Growth: 196.39, Drawdown: 40.95, Price: 0.00, Subscribers: 59
- Signal "Bongsanmaskdance1803". Author login: kim25801863, ID 936062, Growth: 12.53, Drawdown: 10.31, Price: 0.00, Subscribers: 0
- Signal "Prospector Scalper EA". Author login: robots4forex, ID 435626, Growth: 334.76, Drawdown: 43.93, Price: 0.00, Subscribers: 215
- Signal "ADS MT5". Author login: vluxus, ID 478235, Growth: 295.68, Drawdown: 40.26, Price: 0.00, Subscribers: 92
```

Next, we get the full description of a detected signal with a maximum growth in % and the successful subscription message:

```
============= Beginning of parameter list (Signal from the MQL5.com Signal service) =============
Account type: Demo
Publication date: 2020.07.02 16:29
Monitoring start date: 2020.07.02 16:29
Date of the latest update of the trading statistics: 2021.03.07 15:11
ID: 784584
Trading account leverage: 33
Trading result in pips: -19248988
Position in the Rating of Signals: 872
Number of subscribers: 6
Number of trades: 1825
Status of account subscription to a signal: No
------
Account balance: 12061.98
Account equity: 12590.32
Account growth in %: 1115.93
Maximum drawdown: 70.62
Signal subscription price: 0.00
Signal ROI (Return on Investment) in %: 1169.19
------
Author login: "tradewai.com"
Broker (company) name: "MetaQuotes Software Corp."
Broker server: "MetaQuotes-Demo"
Name: "Tradewai"
Account currency: "USD"
============= End of parameter list (Signal from the MQL5.com Signal service) =============

Subscribed to signal ID 784584 "Tradewai"
```

The subscription parameters are displayed afterwards:

```
============= Signal copying parameters =============
Allow using signals for program: Yes
Agree to the terms of use of the Signals service: Yes
Enable copying deals by subscription: Yes
Enable synchronization without confirmation dialog: No
Copying Stop Loss and Take Profit: Yes
Market order slippage when synchronizing positions and copying deals: Spread * 2.00
Percentage for converting deal volume: 50.00%
Limit by deposit: 70%
Limitation on signal equity: 7.00%
Signal ID: 784584
Signal name: Tradewai
```

During the next tick, we obtain the message about successful unsubscribing from a signal.

```
Unsubscribed from the signal ID 784584 "Tradewai"
```

### What's next?

In the next article, I will start developing the library functionality for working with symbol charts.

All files of the current version of the library are attached below together with the test EA file for MQL5 for you to test and download.

Note that working with signals in the test EA shown here is meant solely for test purposes.

The example provided in the EA only gives a general idea for implementing custom solutions based on the library and its classes for working with the MQL5.com Signals service.

Leave your questions and suggestions in the comments.

[Back to contents](https://www.mql5.com/en/articles/9146#node00)

**\*Previous articles within the series:**

[Prices in DoEasy library (Part 62): Updating tick series in real time, preparation for working with Depth of Market](https://www.mql5.com/en/articles/8988)

[Prices in DoEasy library (Part 63): Depth of Market and its abstract request class](https://www.mql5.com/en/articles/9010)

[Prices in DoEasy library (Part 64): Depth of Market, classes of DOM snapshot and snapshot series objects](https://www.mql5.com/en/articles/9044)

[Prices and Signals in DoEasy library (Part 65): Depth of Market collection and the class for working with MQL5.com Signals](https://www.mql5.com/en/articles/9095)

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/9146](https://www.mql5.com/ru/articles/9146)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/9146.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/9146/mql5.zip "Download MQL5.zip")(3917.08 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/368181)**
(10)


![43783869](https://c.mql5.com/avatar/2021/4/608AE11B-5489.jpg)

**[43783869](https://www.mql5.com/en/users/43783869)**
\|
10 May 2021 at 17:07

I'm very sorry that I've done something I shouldn't have done again, and I regret it.

My own ignorant

Forgive me for filling those around me with my actions, I am sorry.

![Lovely Raja](https://c.mql5.com/avatar/2021/5/60951346-27B7.png)

**[Lovely Raja](https://www.mql5.com/en/users/lovelyraja)**
\|
11 May 2021 at 09:27

Hi sir i chat with you on WhatsApp


![Artyom Trishkin](https://c.mql5.com/avatar/2022/7/62C4775C-ABD6.jpg)

**[Artyom Trishkin](https://www.mql5.com/en/users/artmedia70)**
\|
11 May 2021 at 09:35

**Lovely Raja :**

Hi sir i chat with you on WhatsApp

I do not communicate through WhatsApp

![Francisco Carlos Sobral Ribeiro](https://c.mql5.com/avatar/avatar_na2.png)

**[Francisco Carlos Sobral Ribeiro](https://www.mql5.com/en/users/fcsrcarlos)**
\|
16 Jun 2023 at 02:34

Boa noite!

Saudações aqui do Brasil.

Estou tendo impedimento na hora de compilar o código e como meu conhecimento de mql5 é muito básico, resta pedir sua ajuda para superar os problemas na compilação...não sei resolver.

Quero aproveitar e parabenizá-lo pelo trabalho brilhante, seus códigos me ajudam muito.

![Artyom Trishkin](https://c.mql5.com/avatar/2022/7/62C4775C-ABD6.jpg)

**[Artyom Trishkin](https://www.mql5.com/en/users/artmedia70)**
\|
16 Jun 2023 at 08:20

**Francisco Carlos Sobral Ribeiro [#](https://www.mql5.com/ru/forum/364733#comment_47557772):**

Boa noite!

Saudações aqui do Brasil.

Estou tendo impedimento na hora de compilar o código e como meu conhecimento de mql5 é muito básico, resta pedir sua ajuda para superar os problemas na compilação...não sei resolver.

Quero aproveitar e parabenizá-lo pelo trabalho brilhante, seus códigos me ajudam muito.

In Trading.mqh, you need to move some methods to the protected section so that they can be accessed from derived classes. Now they are in the private section of the class. I made this error due to inattention, but the old compiler missed it. After updating the terminal, this error became detectable.

In lines 84 - 89 of the Trading.mqh file you should make the following changes:

In the Trading.mqh file, some methods need to be moved to a protected section so that they can be accessed from derived classes. Now they are in the private section of the class. This mistake was made by me inadvertently, but the old compiler missed it. After updating the terminal, this error began to be detected.

In lines 84 - 89 of the Trading.mqh file, you need to make the following changes:

```
//--- Returns the order direction by operation type
   ENUM_ORDER_TYPE      DirectionByActionType(const ENUM_ACTION_TYPE action)  const;
//--- Sets the trade object to the desired sound
   void                 SetSoundByMode(const ENUM_MODE_SET_SOUND mode,const ENUM_ORDER_TYPE action,const string sound,CTradeObj *trade_obj);

protected:
//--- Sets trade request prices
   template <typename PR,typename SL,typename TP,typename PL>
   bool                 SetPrices(const ENUM_ORDER_TYPE action,const PR price,const SL sl,const TP tp,const PL limit,const string source_method,CSymbol *symbol_obj);

private:
//--- Returns the flag for checking the permissibility by the distance of (1) StopLoss, (2) TakeProfit, (3) the order setting price from the StopLevel price
   bool                 CheckStopLossByStopLevel(const ENUM_ORDER_TYPE order_type,const double price,const double sl,const CSymbol *symbol_obj);
   bool                 CheckTakeProfitByStopLevel(const ENUM_ORDER_TYPE order_type,const double price,const double tp,const CSymbol *symbol_obj);
   bool                 CheckPriceByStopLevel(const ENUM_ORDER_TYPE order_type,const double price,const CSymbol *symbol_obj,const double limit=0);
```

and in lines 155 - 181 make the following changes :

and in lines 155 - 181 make similar edits:

```
//--- Returns the error handling method
   ENUM_ERROR_CODE_PROCESSING_METHOD   ResultProccessingMethod(const uint result_code);
//--- Error correction
   ENUM_ERROR_CODE_PROCESSING_METHOD   RequestErrorsCorrecting(MqlTradeRequest &request,const ENUM_ORDER_TYPE order_type,const uint spread_multiplier,CSymbol *symbol_obj,CTradeObj *trade_obj);

protected:
//--- (1) Opens a position, (2) sets a pending order
   template<typename SL,typename TP>
   bool                 OpenPosition(const ENUM_POSITION_TYPE type,
                                    const double volume,
                                    const string symbol,
                                    const ulong magic=ULONG_MAX,
                                    const SL sl=0,
                                    const TP tp=0,
                                    const string comment=NULL,
                                    const ulong deviation=ULONG_MAX,
                                    const ENUM_ORDER_TYPE_FILLING type_filling=WRONG_VALUE);
   template<typename PR,typename PL,typename SL,typename TP>
   bool                 PlaceOrder( const ENUM_ORDER_TYPE order_type,
                                    const double volume,
                                    const string symbol,
                                    const PR price,
                                    const PL price_limit=0,
                                    const SL sl=0,
                                    const TP tp=0,
                                    const ulong magic=ULONG_MAX,
                                    const string comment=NULL,
                                    const datetime expiration=0,
                                    const ENUM_ORDER_TYPE_TIME type_time=WRONG_VALUE,
                                    const ENUM_ORDER_TYPE_FILLING type_filling=WRONG_VALUE);

private:
//--- Returns the index of the query object in the list by (1) identifier,
//--- (2) order ticket, (3) request position ticket
   int                  GetIndexPendingRequestByID(const uchar id);
   int                  GetIndexPendingRequestByOrder(const ulong ticket);
   int                  GetIndexPendingRequestByPosition(const ulong ticket);

public:
```

After that everything will compile.

The corrected file is attached to this post.

After that, everything will compile.

The corrected file is attached to this post.

![Brute force approach to pattern search (Part IV): Minimal functionality](https://c.mql5.com/2/41/1560775468.png)[Brute force approach to pattern search (Part IV): Minimal functionality](https://www.mql5.com/en/articles/8845)

The article presents an improved brute force version, based on the goals set in the previous article. I will try to cover this topic as broadly as possible using Expert Advisors with settings obtained using this method. A new program version is attached to this article.

![Neural networks made easy (Part 12): Dropout](https://c.mql5.com/2/48/Neural_networks_made_easy_012.png)[Neural networks made easy (Part 12): Dropout](https://www.mql5.com/en/articles/9112)

As the next step in studying neural networks, I suggest considering the methods of increasing convergence during neural network training. There are several such methods. In this article we will consider one of them entitled Dropout.

![Other classes in DoEasy library (Part 67): Chart object class](https://c.mql5.com/2/42/MQL5-avatar-doeasy-library__5.png)[Other classes in DoEasy library (Part 67): Chart object class](https://www.mql5.com/en/articles/9213)

In this article, I will create the chart object class (of a single trading instrument chart) and improve the collection class of MQL5 signal objects so that each signal object stored in the collection updates all its parameters when updating the list.

![Prices and Signals in DoEasy library (Part 65): Depth of Market collection and the class for working with MQL5.com Signals](https://c.mql5.com/2/42/MQL5-avatar-doeasy-library__3.png)[Prices and Signals in DoEasy library (Part 65): Depth of Market collection and the class for working with MQL5.com Signals](https://www.mql5.com/en/articles/9095)

In this article, I will create the collection class of Depths of Market of all symbols and start developing the functionality for working with the MQL5.com Signals service by creating the signal object class.

[What's wrong with regular VPS?Here are the 8 most common problems that algorithmic traders may encounterRead![](https://www.mql5.com/ff/sh/hzatb686qjqxwtr4z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/450486&a=drhremihlwuaqyvgpzfddbtmgciejpba&s=c37d25bcceb93ed153b814e6ba4d4839461a9b2d68dd82b95b142be06d310f3f&uid=&ref=https://www.mql5.com/en/articles/9146&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5068642518704258015)

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