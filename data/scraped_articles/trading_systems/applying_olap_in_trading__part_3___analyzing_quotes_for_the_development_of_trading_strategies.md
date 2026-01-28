---
title: Applying OLAP in trading (part 3): Analyzing quotes for the development of trading strategies
url: https://www.mql5.com/en/articles/7535
categories: Trading Systems, Expert Advisors
relevance_score: 0
scraped_at: 2026-01-24T13:52:36.638286
---

[![](https://www.mql5.com/ff/si/6pp0j40fqxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=luckhiizjxvmvgigcufevttapwwrwbld&s=08cd1d929f27358481aded3c1c5f4e75a9bd5f52c477127afef2a5c532aec5c5&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=jiievewpimjdywrgavaecpwlnvpvwwxy&ssn=1769251954563775723&ssn_dr=0&ssn_sr=0&fv_date=1769251954&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F7535&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Applying%20OLAP%20in%20trading%20(part%203)%3A%20Analyzing%20quotes%20for%20the%20development%20of%20trading%20strategies%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176925195447858837&fz_uniq=5083178195567515249&sv=2552)

MetaTrader 5 / Trading systems


In this article we will continue dealing with the OLAP (On-Line Analytical Processing) technology applied in trading. The first two
articles provided the description of general techniques for creating classes which enable the [accumulation \\
and analysis of multi-dimensional data](https://www.mql5.com/en/articles/6602), as well the [visualization of analysis results](https://www.mql5.com/en/articles/6603)
in the graphical interface. Both articles dealt with the processing of trading reports received in different ways: from the strategy
tester, from the online trading history, and from HTML and CSV files (including MQL5 trading signals). However, OLAP can be applied in other
areas. In particular, OLAP is a convenient technique for analyzing quotes and developing trading strategies.

### Introduction

Here is a brief summary of what was implemented in previous articles (if you haven't read them, it is strongly recommended that you start with
the first two articles). The core was in the OLAPcube.mqh file which contained:

- all basic classes of selectors and aggregators
- classes of working records with source data (the abstract basic 'Record' class and a few specialized 'TradeRecord' child classes with data
on deals)
- basic adapter for reading various (abstract) data sources and forming arrays of working records from them
- specific adapter for the account trading history HistoryDataAdapter
- basic class for displaying results and its simplest implementation which uses data logging (Display, LogDisplay)
- a single control panel in the form of the Analyst class which links together the adapter, the aggregator and the display

Specific HTML report related fields were implemented in the HTMLcube.mqh files, in which classes of trades from the HTML report HTMLTradeRecord
and the adapter that generates them HTMLReportAdapter are defined.

Similarly, CSVTradeRecord classes of trades from CSV reports and an adapter for them CSVReportAdapter were implemented separately in the
CSVcube.mqh file.

Finally, to simplify OLAP integration with MQL5 programs, the OLAPcore.mqh file was written. It contained the OLAPWrapper wrapper class for the
entire OLAP functionality used in demonstration projects.

Since the new OLAP processing task addresses a new area, we will need to perform refactoring of the existing code and select the parts of it which
are common not only for the trading history but also for quotes or for any data sources.

### Refactoring

A new file has been created based on OLAPcube.mqh: OLAPCommon.mqh with only the basic types. Firstly, the removed parts include
enumerations describing the applied meaning of data fields, for example SELECTORS and TRADE\_RECORD\_FIELDS. Also, selector class and
record classes related to trading have been excluded. Of course, all these parts were not deleted, but were moved to a new file
OLAPTrades.mqh which was created for working with trading history and reports.

The former wrapper class OLAPWrapper, which became a template and was renamed to OLAPEngine, has been moved to the OLAPCommon.mqh file.
Enumerations of working fields will be used as a parametrization parameter (for example, TRADE\_RECORD\_FIELDS will be used for the
adaptation of projects from article 1 and 2, see the details below).

The OLAPTrades.mqh file contains the following types (described in articles 1 and 2):

- enumerations TRADE\_SELECTORS (former SELECTORS), TRADE\_RECORD\_FIELDS
- selectors TradeSelector, TypeSelector, SymbolSelector, MagicSelector, ProfitableSelector, DaysRangeSelector
- record classes TradeRecord, CustomTradeRecord, HistoryTradeRecord
- adapter HistoryDataAdapter
- OLAPEngineTrade engine — OLAPEngine<TRADE\_RECORD\_FIELDS> specialization

Pay attention to the presence of DaysRangeSelector selector, which has become a standard selector for analyzing trading history. In
earlier versions, it was located in the OLAPcore.mqh file as a custom selector example.

A default adapter instance is created at the end of the file:

```
  HistoryDataAdapter<RECORD_CLASS> _defaultHistoryAdapter;
```

along with the OLAP engine instance:

```
  OLAPEngineTrade _defaultEngine;
```

These objects are convenient to use from the client source code. The approach of presenting ready objects will be applied in other application
areas (header files), in particular in the planned quote analyzer.

Files HTMLcube.mqh and CSVcube.mqh remain almost unchanged. All the previously existing trading history and report analyzing functions
have been preserved. A new testing Expert Advisor OLAPRPRT.mq5 is attached below for demonstration purposes; it is an analogue of
OLAPDEMO.mq5 from the first article.

Using OLAPTrades.mqh as an example, you can easily create specialized implementations of OLAP classes for other data types.

We are going to complicate the project by adding new features. Therefore, all aspects of OLAP integration with a graphical interface will
not be considered here. In this article, we will focus on data analysis without reference to visualization (furthermore, there can be
different visualization methods). After reading this article, you can combine the updated engine with the GUI part from article 2.

### Improvements

In the context of quote analysis, we may need new methods of logical dissection and data accumulation. The required classes will be added to
OLAPCommon.mqh, because they are basic in nature. Thus, they will be available to any application cubes, including the former ones from
OLAPTrades.mqh.

The following has been added:

- selector MonthSelector
- selector WorkWeekDaySelector
- aggregator VarianceAggregator

MonthSelector will enable data grouping by months. This selector was somehow omitted in previous implementations.

```
  template<typename E>
  class MonthSelector: public DateTimeSelector<E>
  {
    public:
      MonthSelector(const E f): DateTimeSelector(f, 12)
      {
        _typename = typename(this);
       }

      virtual bool select(const Record *r, int &index) const
      {
        double d = r.get(selector);
        datetime t = (datetime)d;
        index = TimeMonth(t) - 1;
        return true;
       }

      virtual string getLabel(const int index) const
      {
        static string months[12] = {"January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"};
        return months[index];
       }
  };
```

WorkWeekDaySelector is an analogue of WeekDaySelector, but it splits data by weekdays (from 1 to 5). It is a convenient solution for the analysis of markets, in
which trading is not performed on weekends: weekend values are always zero so there is no need to reserve hypercube cells for them.

VarianceAggregator allows calculating data variance and thus it complements AverageAggregator. The idea of the new aggregator can be compared to the value
of Average True Range (ATR), however the aggregator can be calculated for any data samples (for example, separately by the hours of the day or
by the days of the week), as well as for other data sources (for example, variance of revenue in the trading history).

```
  template<typename E>
  class VarianceAggregator: public Aggregator<E>
  {
    protected:
      int counters[];
      double sumx[];
      double sumx2[];

    public:
      VarianceAggregator(const E f, const Selector<E> *&s[], const Filter<E> *&t[]): Aggregator(f, s, t)
      {
        _typename = typename(this);
       }

      virtual void setSelectorBounds(const int length = 0) override
      {
        Aggregator<E>::setSelectorBounds();
        ArrayResize(counters, ArraySize(totals));
        ArrayResize(sumx, ArraySize(totals));
        ArrayResize(sumx2, ArraySize(totals));
        ArrayInitialize(counters, 0);
        ArrayInitialize(sumx, 0);
        ArrayInitialize(sumx2, 0);
       }

      virtual void update(const int index, const double value) override
      {
        counters[index]++;
        sumx[index] += value;
        sumx2[index] += value * value;

        const int n = counters[index];
        const double variance = (sumx2[index] - sumx[index] * sumx[index] / n) / MathMax(n - 1, 1);
        totals[index] = MathSqrt(variance);
       }
  };
```

![Fig.1 A diagram of aggregator classes](https://c.mql5.com/2/37/aggregators.png)

**Fig.1 A diagram of aggregator classes**

Selectors QuantizationSelector and SerialNumberSelector are derived from BaseSelector instead of a more specific TradeSelector.
QuantizationSelector has got a new constructor parameter which allows setting the granularity of the selector. It is equal to zero by
default, which means data is grouped by exact match of the appropriate field value (the field is specified in the selector). For example, in
the previous article we used quantization by lot size to obtain the report on profits broken down by lot sizes. The cube cells were lots, such
as 0.01, 0.1 etc. which were contained in the trading history. Sometimes it is more convenient to quantize with a specified step (cell size).
This step can be specified using the new constructor parameter. The newly added parts are marked with the + comment in the below source code.

```
  template<typename T>
  class QuantizationSelector: public BaseSelector<T>
  {
    protected:
      Vocabulary<double> quants;
      uint cell;                 // +

    public:
      QuantizationSelector(const T field, const uint granularity = 0 /* + */): BaseSelector<T>(field), cell(granularity)
      {
        _typename = typename(this);
       }

      virtual void prepare(const Record *r) override
      {
        double value = r.get(selector);
        if(cell != 0) value = MathSign(value) * MathFloor(MathAbs(value) / cell) * cell; // +
        quants.add(value);
       }

      virtual bool select(const Record *r, int &index) const override
      {
        double value = r.get(selector);
        if(cell != 0) value = MathSign(value) * MathFloor(MathAbs(value) / cell) * cell; // +
        index = quants.get(value);
        return (index >= 0);
       }

      virtual int getRange() const override
      {
        return quants.size();
       }

      virtual string getLabel(const int index) const override
      {
        return (string)(float)quants[index];
       }
  };
```

In addition, other improvements have been made to existing classes. Now, Filter and FilterRange filter classes support comparison by the
field value, not only by the index of the cell to which the value will be added. This is convenient from the user perspective, since the cell
index is not always known in advance. The new mode is enabled if the selector returns an index equal to -1 (the newly added code lines are marked
with + comments):

```
  template<typename E>
  class Filter
  {
    protected:
      Selector<E> *selector;
      double filter;

    public:
      Filter(Selector<E> &s, const double value): selector(&s), filter(value)
      {
       }

      virtual bool matches(const Record *r) const
      {
        int index;
        if(selector.select(r, index))
        {
          if(index == -1)                                             // +
          {                                                           // +
            if(dynamic_cast<FilterSelector<E> *>(selector) != NULL)   // +
            {                                                         // +
              return r.get(selector.getField()) == filter;            // +
            }                                                         // +
          }                                                           // +
          else                                                        // +
          {                                                           // +
            if(index == (int)filter) return true;
          }                                                           // +
         }
        return false;
       }

      Selector<E> *getSelector() const
      {
        return selector;
       }

      virtual string getTitle() const
      {
        return selector.getTitle() + "[" + (string)filter + "]";
       }
  };
```

Of course, we need a selector that can return -1 as an index. It is called FilterSelector.

```
  template<typename T>
  class FilterSelector: public BaseSelector<T>
  {
    public:
      FilterSelector(const T field): BaseSelector(field)
      {
        _typename = typename(this);
       }

      virtual bool select(const Record *r, int &index) const override
      {
        index = -1;
        return true;
       }

      virtual int getRange() const override
      {
        return 0;
       }

      virtual double getMin() const override
      {
        return 0;
       }

      virtual double getMax() const override
      {
        return 0;
       }

      virtual string getLabel(const int index) const override
      {
        return EnumToString(selector);
       }
  };
```

As you can see, the selector returns true for any record, which means that the record should be processed, and -1 is returned as an index. Based
on this value the filter knows that the user requests to filter data not by index, but by a field value. The example of its usage will be
considered below.

Also, the log display now supports sorting of a multidimensional cube by values. Previously, multidimensional cubes could not be sorted.
Sorting of a multidimensional cube is only partially available, i.e. it is possible only for those selectors which can uniformly format
labels by strings in a lexicographic order. In particular, the new weekday selector provides labels: "1\`Monday", "2\`Tuesday",
"3\`Wednesday", "4\`Thursday", "5\`Friday". The day number at the beginning enables proper sorting. Otherwise, label comparing functions
would be needed for a proper implementation. Furthermore, for some of the "sequential" aggregators, IdentityAggregator,
ProgressiveTotalAggregator, it may be necessary to set priorities of cube sides, because in these aggregators the X axis always shows the
sequence number of the record, which however should not be the first criterion in sorting (or even should be used as the last criterion).

These are only some of the modifications in the source code. You can check all of them by comparing source codes.

### Extending OLAP to the quotes application area

Let's use the basic classes from OLAPCommon.mqh as the basis and create a file with quote analysis classes similar to OLAPTrades.mqh:
OLAPQuotes.mqh. Firstly, let's describe the following types:

- enumerations QUOTE\_SELECTORS, QUOTE\_RECORD\_FIELDS
- selectors QuoteSelector, ShapeSelector
- record classes QuotesRecord, CustomQuotesBaseRecord
- adapter QuotesDataAdapter
- OLAPEngineQuotes — OLAPEngine<QUOTE\_RECORD\_FIELDS> specialization

The QUOTE\_SELECTORS enumeration is defined as follows:

```
  enum QUOTE_SELECTORS
  {
    SELECTOR_NONE,       // none
    SELECTOR_SHAPE,      // type
    SELECTOR_INDEX,      // ordinal number
    /* below datetime field assumed */
    SELECTOR_MONTH,      // month-of-year
    SELECTOR_WEEKDAY,    // day-of-week
    SELECTOR_DAYHOUR,    // hour-of-day
    SELECTOR_HOURMINUTE, // minute-of-hour
    /* the next require a field as parameter */
    SELECTOR_SCALAR,     // scalar(field)
    SELECTOR_QUANTS,     // quants(field)
    SELECTOR_FILTER      // filter(field)
  };
```

The shape selector differentiates bars by types: bullish, bearish and neutral depending on the price movement direction.

The index selector corresponds to the SerialNumberSelector class which is defined in base classes (file OLAPCommon.mqh). When dealing
with trading operations, these were the serial numbers of the deals. Now, bar numbers will be used for quotes.

The month selector was described above. Other selectors are inherited from previous articles.

Data fields in the quotes are described by the following enumeration:

```
  enum QUOTE_RECORD_FIELDS
  {
    FIELD_NONE,          // none
    FIELD_INDEX,         // index (bar number)
    FIELD_SHAPE,         // type (bearish/flat/bullish)
    FIELD_DATETIME,      // datetime
    FIELD_PRICE_OPEN,    // open price
    FIELD_PRICE_HIGH,    // high price
    FIELD_PRICE_LOW,     // low price
    FIELD_PRICE_CLOSE,   // close price
    FIELD_PRICE_RANGE_OC,// price range (OC)
    FIELD_PRICE_RANGE_HL,// price range (HL)
    FIELD_SPREAD,        // spread
    FIELD_TICK_VOLUME,   // tick volume
    FIELD_REAL_VOLUME,   // real volume
    FIELD_CUSTOM1,       // custom 1
    FIELD_CUSTOM2,       // custom 2
    FIELD_CUSTOM3,       // custom 3
    FIELD_CUSTOM4,       // custom 4
    QUOTE_RECORD_FIELDS_LAST
  };
```

The purpose of each of them should be clear from the names and comments.

Two of the above enumerations are implemented as macros:

```
  #define SELECTORS QUOTE_SELECTORS
  #define ENUM_FIELDS QUOTE_RECORD_FIELDS
```

Note that similar macro definitions — SELECTORS and ENUM\_FIELDS — are available in all "applied" header files. As for now, we have two files
(OLAPTrades.mqh, OLAPQuotes.mqh — for the history of trading operations and for quotes), but there can be more such files. Thus, in any
project which uses OLAP, it is now possible to analyze only one application area at a time (for example, either OLAPTrades.mqh or
OLAPQuotes.mqh, but not both at once). Another small refactoring can be needed to enable cross-analysis of different cubes. This part is
not covered in this article, because tasks requiring parallel analysis of multiple metacubes seem to be rare. Should you need this, you may
perform such a refactoring yourself.

The parent selector for quotes is a specialization of BaseSelector with the fields QUOTE\_RECORD\_FIELDS:

```
  class QuoteSelector: public BaseSelector<QUOTE_RECORD_FIELDS>
  {
    public:
      QuoteSelector(const QUOTE_RECORD_FIELDS field): BaseSelector(field)
      {
       }
  };
```

The bar type selector (bullish or bearish) is implemented as follows:

```
  class ShapeSelector: public QuoteSelector
  {
    public:
      ShapeSelector(): QuoteSelector(FIELD_SHAPE)
      {
        _typename = typename(this);
       }

      virtual bool select(const Record *r, int &index) const
      {
        index = (int)r.get(selector);
        index += 1; // shift from -1, 0, +1 to [0..2]
        return index >= getMin() && index <= getMax();
       }

      virtual int getRange() const
      {
        return 3; // 0 through 2
       }

      virtual string getLabel(const int index) const
      {
        const static string types[3] = {"bearish", "flat", "bullish"};
        return types[index];
       }
  };
```

There are 3 reserved values indicating types: -1 for the downward movement, 0 for flat, +1 for the upward movement. Therefore, cell indices are
in the range from 0 to 2 (inclusive). The below QuotesRecord class demonstrates the filling of the field with the relevant value
corresponding to the type of the specific bar.

[![Fig.2 Diagram of selector classes](https://c.mql5.com/2/37/selectors__1.png)](https://c.mql5.com/2/37/selectors.png "https://c.mql5.com/2/37/selectors.png")

**Fig.2 Diagram of selector classes**

Here is the class of the record storing information about a particular bar:

```
  class QuotesRecord: public Record
  {
    protected:
      static int counter; // number of bars

      void fillByQuotes(const MqlRates &rate)
      {
        set(FIELD_INDEX, counter++);
        set(FIELD_SHAPE, rate.close > rate.open ? +1 : (rate.close < rate.open ? -1 : 0));
        set(FIELD_DATETIME, (double)rate.time);
        set(FIELD_PRICE_OPEN, rate.open);
        set(FIELD_PRICE_HIGH, rate.high);
        set(FIELD_PRICE_LOW, rate.low);
        set(FIELD_PRICE_CLOSE, rate.close);
        set(FIELD_PRICE_RANGE_OC, (rate.close - rate.open) / _Point);
        set(FIELD_PRICE_RANGE_HL, (rate.high - rate.low) * MathSign(rate.close - rate.open) / _Point);
        set(FIELD_SPREAD, (double)rate.spread);
        set(FIELD_TICK_VOLUME, (double)rate.tick_volume);
        set(FIELD_REAL_VOLUME, (double)rate.real_volume);
       }

    public:
      QuotesRecord(): Record(QUOTE_RECORD_FIELDS_LAST)
      {
       }

      QuotesRecord(const MqlRates &rate): Record(QUOTE_RECORD_FIELDS_LAST)
      {
        fillByQuotes(rate);
       }

      static int getRecordCount()
      {
        return counter;
       }

      static void reset()
      {
        counter = 0;
       }

      virtual string legend(const int index) const override
      {
        if(index >= 0 && index < QUOTE_RECORD_FIELDS_LAST)
        {
          return EnumToString((QUOTE_RECORD_FIELDS)index);
         }
        return "unknown";
       }
  };
```

All information is received from the MqlRates structure. The creation of class instances will be shown further in the adapter
implementation.

The application of fields is defined in the same class (integer, real, date). We need this, because all record fields are technically stored
in a double-type array.

```
  class QuotesRecord: public Record
  {
    protected:
      const static char datatypes[QUOTE_RECORD_FIELDS_LAST];

    public:
      ...
      static char datatype(const int index)
      {
        return datatypes[index];
       }
  };

  const static char QuotesRecord::datatypes[QUOTE_RECORD_FIELDS_LAST] =
  {
    0,   // none
    'i', // index, serial number
    'i', // type (-1 down/0/+1 up)
    't', // datetime
    'd', // open price
    'd', // high price
    'd', // low price
    'd', // close price
    'd', // range OC
    'd', // range HL
    'i', // spread
    'i', // tick
    'i', // real
    'd',    // custom 1
    'd',    // custom 2
    'd',    // custom 3
    'd'     // custom 4
  };
```

The presence of such a flag for field specialization allows adjusting data input/output in the user interface, which will be demonstrated
below.

An intermediate class is available to enable the filling of custom fields. Its main purpose is to call fillCustomFields from the custom
class which is specified by the basic one using a template (thus, at the time of CustomQuotesBaseRecord constructor call, our custom object
is already created and filled with standard fields, which are often needed to calculate custom fields):

```
  template<typename T>
  class CustomQuotesBaseRecord: public T
  {
    public:
      CustomQuotesBaseRecord(const MqlRates &rate): T(rate)
      {
        fillCustomFields();
       }
  };
```

It is used in the quotes adapter:

```
  template<typename T>
  class QuotesDataAdapter: public DataAdapter
  {
    private:
      int size;
      int cursor;

    public:
      QuotesDataAdapter()
      {
        reset();
       }

      virtual void reset() override
      {
        size = MathMin(Bars(_Symbol, _Period), TerminalInfoInteger(TERMINAL_MAXBARS));
        cursor = size - 1;
        T::reset();
       }

      virtual int reservedSize()
      {
        return size;
       }

      virtual Record *getNext()
      {
        if(cursor >= 0)
        {
          MqlRates rate[1];
          if(CopyRates(_Symbol, _Period, cursor, 1, rate) > 0)
          {
            cursor--;
            return new CustomQuotesBaseRecord<T>(rate[0]);
           }

          Print(__FILE__, " ", __LINE__, " ", GetLastError());

          return NULL;
         }
        return NULL;
       }
  };
```

The class goes through the bars in the chronological order, from older to newer ones. This means that indexing (FIELD\_INDEX field) is
performed as in a regular array, not in timeseries order.

Finally, the OLAP quote engine is as follows:

```
  class OLAPEngineQuotes: public OLAPEngine<QUOTE_SELECTORS,QUOTE_RECORD_FIELDS>
  {
    protected:
      virtual Selector<QUOTE_RECORD_FIELDS> *createSelector(const QUOTE_SELECTORS selector, const QUOTE_RECORD_FIELDS field) override
      {
        switch(selector)
        {
          case SELECTOR_SHAPE:
            return new ShapeSelector();
          case SELECTOR_INDEX:
            return new SerialNumberSelector<QUOTE_RECORD_FIELDS,QuotesRecord>(FIELD_INDEX);
          case SELECTOR_MONTH:
            return new MonthSelector<QUOTE_RECORD_FIELDS>(FIELD_DATETIME);
          case SELECTOR_WEEKDAY:
            return new WorkWeekDaySelector<QUOTE_RECORD_FIELDS>(FIELD_DATETIME);
          case SELECTOR_DAYHOUR:
            return new DayHourSelector<QUOTE_RECORD_FIELDS>(FIELD_DATETIME);
          case SELECTOR_HOURMINUTE:
            return new DayHourSelector<QUOTE_RECORD_FIELDS>(FIELD_DATETIME);
          case SELECTOR_SCALAR:
            return field != FIELD_NONE ? new BaseSelector<QUOTE_RECORD_FIELDS>(field) : NULL;
          case SELECTOR_QUANTS:
            return field != FIELD_NONE ? new QuantizationSelector<QUOTE_RECORD_FIELDS>(field, QuantGranularity) : NULL;
          case SELECTOR_FILTER:
            return field != FIELD_NONE ? new FilterSelector<QUOTE_RECORD_FIELDS>(field) : NULL;
         }
        return NULL;
       }

      virtual void initialize() override
      {
        Print("Bars read: ", QuotesRecord::getRecordCount());
       }

    public:
      OLAPEngineQuotes(): OLAPEngine() {}
      OLAPEngineQuotes(DataAdapter *ptr): OLAPEngine(ptr) {}

  };
```

All the main functions are still available in the OLAPEngine basic class which was described in the first article (its name was OLAPWrapper).
Here, we only need to create quote-specific selectors.

The default adapter and OLAP engine instances will be presented as ready-made objects:

```
  QuotesDataAdapter<RECORD_CLASS> _defaultQuotesAdapter;
  OLAPEngineQuotes _defaultEngine;
```

Based on the created classes for two analysis application areas (OLAPTrades.mqh, OLAPQuotes.mqh), OLAP functionality can be easily
extended to other purposes, such as processing of optimization results or of data received from external resources.

![Fig.3 Diagram of OLAP control classes](https://c.mql5.com/2/37/analyst2.png)

**Fig.3 Diagram of OLAP control classes**

### Expert Advisor for OLAP-Analysis of Quotes

Everything is ready to start using the created classes. Let's develop a non-trading Expert Advisor OLAPQTS.mq5. Its structure will be similar to
that of OLAPRPRT.mq5, which is used for analyzing trading reports.

There is the CustomQuotesRecord class, which allows demonstrating the calculation/filling of custom fields. It is inherited from
QuotesRecord. Let's use some custom fields to determine patterns in quotes which can be used as the basis for constructing trading
strategies. All such fields are filled in the fillCustomFields method. They will be described in detail later.

```
  class CustomQuotesRecord: public QuotesRecord
  {
    public:
      CustomQuotesRecord(): QuotesRecord() {}
      CustomQuotesRecord(const MqlRates &rate): QuotesRecord(rate)
      {
       }

      virtual void fillCustomFields() override
      {

        // ...

       }

      virtual string legend(const int index) const override
      {
        // ...
        return QuotesRecord::legend(index);
       }
  };
```

For the adapter to "know" about our record class CustomQuotesRecord and to create its instances, the following macro should be defined
before including OLAPQuotes.mqh:

```
  // this line plugs our class into default adapter in OLAPQuotes.mqh
  #define RECORD_CLASS CustomQuotesRecord

  #include <OLAP/OLAPQuotes.mqh>
```

The Expert Advisor is managed through input parameters which are similar to those used in the trading history analysis project. Data can be
accumulated in three metacube dimensions, for which it is possible to choose selectors along the X, Y and Z axes. It is also possible to filter
by one value or by a range of values. And finally, the user should choose the aggregator type (some aggregators require the specification of
the aggregation field, others imply a specific field) and optionally the sorting type.

```
  sinput string X = "————— X axis —————"; // · X ·
  input SELECTORS SelectorX = DEFAULT_SELECTOR_TYPE; // · SelectorX
  input ENUM_FIELDS FieldX = DEFAULT_SELECTOR_FIELD /* field does matter only for some selectors */; // · FieldX

  sinput string Y = "————— Y axis —————"; // · Y ·
  input SELECTORS SelectorY = SELECTOR_NONE; // · SelectorY
  input ENUM_FIELDS FieldY = FIELD_NONE; // · FieldY

  sinput string Z = "————— Z axis —————"; // · Z ·
  input SELECTORS SelectorZ = SELECTOR_NONE; // · SelectorZ
  input ENUM_FIELDS FieldZ = FIELD_NONE; // · FieldZ

  sinput string F = "————— Filter —————"; // · F ·
  input SELECTORS _Filter1 = SELECTOR_NONE; // · Filter1
  input ENUM_FIELDS _Filter1Field = FIELD_NONE; // · Filter1Field
  input string _Filter1value1 = ""; // · Filter1value1
  input string _Filter1value2 = ""; // · Filter1value2

  sinput string A = "————— Aggregator —————"; // · A ·
  input AGGREGATORS _AggregatorType = DEFAULT_AGGREGATOR_TYPE; // · AggregatorType
  input ENUM_FIELDS _AggregatorField = DEFAULT_AGGREGATOR_FIELD; // · AggregatorField
  input SORT_BY _SortBy = SORT_BY_NONE; // · SortBy
```

All the selectors and their fields are implemented as arrays and can be easily passed to the engine:

```
  SELECTORS _selectorArray[4];
  ENUM_FIELDS _selectorField[4];

  int OnInit()
  {
    _selectorArray[0] = SelectorX;
    _selectorArray[1] = SelectorY;
    _selectorArray[2] = SelectorZ;
    _selectorArray[3] = _Filter1;
    _selectorField[0] = FieldX;
    _selectorField[1] = FieldY;
    _selectorField[2] = FieldZ;
    _selectorField[3] = _Filter1Field;

    _defaultEngine.setAdapter(&_defaultQuotesAdapter);

    EventSetTimer(1);
    return INIT_SUCCEEDED;
   }
```

As we can see, the EA uses the default instances of the engine and of the quotes adapter. According to application specifics, the EA should
process the data once for the entered parameters. For this purpose, as well as to enable operation on weekends when there are no ticks, a timer
starts in the OnInit handler.

The processing start in OnTimer is as follows:

```
  LogDisplay _display(11, _Digits);

  void OnTimer()
  {
    EventKillTimer();

    double Filter1value1 = 0, Filter1value2 = 0;
    if(CustomQuotesRecord::datatype(_Filter1Field) == 't')
    {
      Filter1value1 = (double)StringToTime(_Filter1value1);
      Filter1value2 = (double)StringToTime(_Filter1value2);
     }
    else
    {
      Filter1value1 = StringToDouble(_Filter1value1);
      Filter1value2 = StringToDouble(_Filter1value2);
     }

    _defaultQuotesAdapter.reset();
    _defaultEngine.process(_selectorArray, _selectorField,
          _AggregatorType, _AggregatorField,
          _display,
          _SortBy,
          Filter1value1, Filter1value2);
   }
```

When analyzing quotes, we will need a filter by dates. Therefore, values for the filters are set in input parameters in the form of strings.
Depending on the type of the field to which the filter is applied, the strings are interpreted as a number or a date (in the common format
YYYY.MM.DD). In the first article, we always had to enter numerical values, which is inconvenient for the end user in the case of dates.

All prepared input parameters are passed to the 'process' method of the OLAP engine. Further work is done without user intervention, after
which the results are displayed in the expert log using a LogDisplay instance.

### Testing OLAP analysis of quotes

Let us perform simple quote research using the above described functionality.

Open the EURUSD D1 chart and attach the OLAPQTS EA to it. Leave all parameters with default values. This means: 'type' selector along the X axis
and the COUNT aggregator. The following filter settings should be changed: in the Filter1 parameter, set "filter(field)", in
Filter1Field — datetime, in Filter1Value1 and Filter1Value2 — "2019.01.01" and "2020.01.01" respectively. Thus, the calculation range
is limited to year 2019.

The EA execution result will be as follows:

```
  OLAPQTS (EURUSD,D1)	Bars read: 12626
  OLAPQTS (EURUSD,D1)	CountAggregator<QUOTE_RECORD_FIELDS> FIELD_NONE [3]
  OLAPQTS (EURUSD,D1)	Filters: FilterRange::FilterSelector<QUOTE_RECORD_FIELDS>(FIELD_DATETIME)[1546300800.0 ... 1577836800.0];
  OLAPQTS (EURUSD,D1)	Selectors: 1
  OLAPQTS (EURUSD,D1)	X: ShapeSelector(FIELD_SHAPE) [3]
  OLAPQTS (EURUSD,D1)	Processed records: 259
  OLAPQTS (EURUSD,D1)	  134.00000: bearish
  OLAPQTS (EURUSD,D1)	    0.00000: flat
  OLAPQTS (EURUSD,D1)	  125.00000: bullish
```

It can be seen from the log that the EA analyzed 12626 bars (the entire available history of EURUSD D1), but only 259 of them match the filter
conditions. 134 of them were bearish, 125 — bullish.

By switching the timeframe to H1 we can obtain the evaluation of one-hour bars:

```
  OLAPQTS (EURUSD,H1)	Bars read: 137574
  OLAPQTS (EURUSD,H1)	CountAggregator<QUOTE_RECORD_FIELDS> FIELD_NONE [3]
  OLAPQTS (EURUSD,H1)	Filters: FilterRange::FilterSelector<QUOTE_RECORD_FIELDS>(FIELD_DATETIME)[1546300800.0 ... 1577836800.0];
  OLAPQTS (EURUSD,H1)	Selectors: 1
  OLAPQTS (EURUSD,H1)	X: ShapeSelector(FIELD_SHAPE) [3]
  OLAPQTS (EURUSD,H1)	Processed records: 6196
  OLAPQTS (EURUSD,H1)	 3051.00000: bearish
  OLAPQTS (EURUSD,H1)	   55.00000: flat
  OLAPQTS (EURUSD,H1)	 3090.00000: bullish
```

Next, let's try to analyze the spreads. One of the MetaTrader features is that MqlRates structures stores only the minimum spread. When testing
trading strategies, such an approach can be dangerous, because this may give falsely optimistic profit estimates. A better option would be
to have the history of both the minimum and the maximum spreads. Of course, if necessary, you can use the history of ticks, but the bar mode is
more resource efficient. Let's try to evaluate real spreads by the hours of the day.

Let's use the same EURUSD H1 chart with the same filter by 2019 and add the following EA settings. Selector X — "hour-of-day", aggregator —
"AVERAGE", aggregator field — "spread". Here are the results:

```
  OLAPQTS (EURUSD,H1)	Bars read: 137574
  OLAPQTS (EURUSD,H1)	AverageAggregator<QUOTE_RECORD_FIELDS> FIELD_SPREAD [24]
  OLAPQTS (EURUSD,H1)	Filters: FilterRange::FilterSelector<QUOTE_RECORD_FIELDS>(FIELD_DATETIME)[1546300800.0 ... 1577836800.0];
  OLAPQTS (EURUSD,H1)	Selectors: 1
  OLAPQTS (EURUSD,H1)	X: DayHourSelector<QUOTE_RECORD_FIELDS>(FIELD_DATETIME) [24]
  OLAPQTS (EURUSD,H1)	Processed records: 6196
  OLAPQTS (EURUSD,H1)	    4.71984: 00
  OLAPQTS (EURUSD,H1)	    3.19066: 01
  OLAPQTS (EURUSD,H1)	    3.72763: 02
  OLAPQTS (EURUSD,H1)	    4.19455: 03
  OLAPQTS (EURUSD,H1)	    4.38132: 04
  OLAPQTS (EURUSD,H1)	    4.28794: 05
  OLAPQTS (EURUSD,H1)	    3.93050: 06
  OLAPQTS (EURUSD,H1)	    4.01158: 07
  OLAPQTS (EURUSD,H1)	    4.39768: 08
  OLAPQTS (EURUSD,H1)	    4.68340: 09
  OLAPQTS (EURUSD,H1)	    4.68340: 10
  OLAPQTS (EURUSD,H1)	    4.64479: 11
  OLAPQTS (EURUSD,H1)	    4.57915: 12
  OLAPQTS (EURUSD,H1)	    4.62934: 13
  OLAPQTS (EURUSD,H1)	    4.64865: 14
  OLAPQTS (EURUSD,H1)	    4.61390: 15
  OLAPQTS (EURUSD,H1)	    4.62162: 16
  OLAPQTS (EURUSD,H1)	    4.50579: 17
  OLAPQTS (EURUSD,H1)	    4.56757: 18
  OLAPQTS (EURUSD,H1)	    4.61004: 19
  OLAPQTS (EURUSD,H1)	    4.59459: 20
  OLAPQTS (EURUSD,H1)	    4.67054: 21
  OLAPQTS (EURUSD,H1)	    4.50775: 22
  OLAPQTS (EURUSD,H1)	    3.57312: 23
```

The average spread value is specified for each hour of the day. But this is averaging by the minimum spread and therefore it is not a real spread.
To have a more real picture, let's switch to the M1 timeframe. Thus, we will analyze all the available historical details (available without
using ticks).

```
  OLAPQTS (EURUSD,M1)	Bars read: 1000000
  OLAPQTS (EURUSD,M1)	AverageAggregator<QUOTE_RECORD_FIELDS> FIELD_SPREAD [24]
  OLAPQTS (EURUSD,M1)	Filters: FilterRange::FilterSelector<QUOTE_RECORD_FIELDS>(FIELD_DATETIME)[1546300800.0 ... 1577836800.0];
  OLAPQTS (EURUSD,M1)	Selectors: 1
  OLAPQTS (EURUSD,M1)	X: DayHourSelector<QUOTE_RECORD_FIELDS>(FIELD_DATETIME) [24]
  OLAPQTS (EURUSD,M1)	Processed records: 371475
  OLAPQTS (EURUSD,M1)	   14.05653: 00
  OLAPQTS (EURUSD,M1)	    6.63397: 01
  OLAPQTS (EURUSD,M1)	    6.00707: 02
  OLAPQTS (EURUSD,M1)	    5.72516: 03
  OLAPQTS (EURUSD,M1)	    5.72575: 04
  OLAPQTS (EURUSD,M1)	    5.77588: 05
  OLAPQTS (EURUSD,M1)	    5.82541: 06
  OLAPQTS (EURUSD,M1)	    5.82560: 07
  OLAPQTS (EURUSD,M1)	    5.77979: 08
  OLAPQTS (EURUSD,M1)	    5.44876: 09
  OLAPQTS (EURUSD,M1)	    5.32619: 10
  OLAPQTS (EURUSD,M1)	    5.32966: 11
  OLAPQTS (EURUSD,M1)	    5.32096: 12
  OLAPQTS (EURUSD,M1)	    5.32117: 13
  OLAPQTS (EURUSD,M1)	    5.29633: 14
  OLAPQTS (EURUSD,M1)	    5.21140: 15
  OLAPQTS (EURUSD,M1)	    5.17084: 16
  OLAPQTS (EURUSD,M1)	    5.12794: 17
  OLAPQTS (EURUSD,M1)	    5.27576: 18
  OLAPQTS (EURUSD,M1)	    5.48078: 19
  OLAPQTS (EURUSD,M1)	    5.60175: 20
  OLAPQTS (EURUSD,M1)	    5.70999: 21
  OLAPQTS (EURUSD,M1)	    5.87404: 22
  OLAPQTS (EURUSD,M1)	    6.94555: 23
```

The result is closer to reality: in some hours the average minimum spread has increased by 2-3 times. To make the analysis even more rigorous,
we can create the highest value instead of the average using the "MAX" aggregator. Although the resulting values will be the highest of the
minimum values, don't forget that they are based on one-minute bars inside each hour and therefore describe entry and exit conditions
during short-term trading perfectly well.

```
  OLAPQTS (EURUSD,M1)	Bars read: 1000000
  OLAPQTS (EURUSD,M1)	MaxAggregator<QUOTE_RECORD_FIELDS> FIELD_SPREAD [24]
  OLAPQTS (EURUSD,M1)	Filters: FilterRange::FilterSelector<QUOTE_RECORD_FIELDS>(FIELD_DATETIME)[1546300800.0 ... 1577836800.0];
  OLAPQTS (EURUSD,M1)	Selectors: 1
  OLAPQTS (EURUSD,M1)	X: DayHourSelector<QUOTE_RECORD_FIELDS>(FIELD_DATETIME) [24]
  OLAPQTS (EURUSD,M1)	Processed records: 371475
  OLAPQTS (EURUSD,M1)	  157.00000: 00
  OLAPQTS (EURUSD,M1)	   31.00000: 01
  OLAPQTS (EURUSD,M1)	   12.00000: 02
  OLAPQTS (EURUSD,M1)	   12.00000: 03
  OLAPQTS (EURUSD,M1)	   13.00000: 04
  OLAPQTS (EURUSD,M1)	   11.00000: 05
  OLAPQTS (EURUSD,M1)	   12.00000: 06
  OLAPQTS (EURUSD,M1)	   12.00000: 07
  OLAPQTS (EURUSD,M1)	   11.00000: 08
  OLAPQTS (EURUSD,M1)	   11.00000: 09
  OLAPQTS (EURUSD,M1)	   12.00000: 10
  OLAPQTS (EURUSD,M1)	   13.00000: 11
  OLAPQTS (EURUSD,M1)	   12.00000: 12
  OLAPQTS (EURUSD,M1)	   13.00000: 13
  OLAPQTS (EURUSD,M1)	   12.00000: 14
  OLAPQTS (EURUSD,M1)	   14.00000: 15
  OLAPQTS (EURUSD,M1)	   16.00000: 16
  OLAPQTS (EURUSD,M1)	   14.00000: 17
  OLAPQTS (EURUSD,M1)	   15.00000: 18
  OLAPQTS (EURUSD,M1)	   21.00000: 19
  OLAPQTS (EURUSD,M1)	   17.00000: 20
  OLAPQTS (EURUSD,M1)	   25.00000: 21
  OLAPQTS (EURUSD,M1)	   31.00000: 22
  OLAPQTS (EURUSD,M1)	   70.00000: 23
```

See the difference: at the beginning we had spreads of 4 points; now there are tens and even a hundred at midnight.

Let us evaluate the variance of spread and check how the new aggregator works. Let's do it by choosing "DEVIATION".

```
  OLAPQTS (EURUSD,M1)	Bars read: 1000000
  OLAPQTS (EURUSD,M1)	VarianceAggregator<QUOTE_RECORD_FIELDS> FIELD_SPREAD [24]
  OLAPQTS (EURUSD,M1)	Filters: FilterRange::FilterSelector<QUOTE_RECORD_FIELDS>(FIELD_DATETIME)[1546300800.0 ... 1577836800.0];
  OLAPQTS (EURUSD,M1)	Selectors: 1
  OLAPQTS (EURUSD,M1)	X: DayHourSelector<QUOTE_RECORD_FIELDS>(FIELD_DATETIME) [24]
  OLAPQTS (EURUSD,M1)	Processed records: 371475
  OLAPQTS (EURUSD,M1)	    9.13767: 00
  OLAPQTS (EURUSD,M1)	    3.12974: 01
  OLAPQTS (EURUSD,M1)	    2.72293: 02
  OLAPQTS (EURUSD,M1)	    2.70965: 03
  OLAPQTS (EURUSD,M1)	    2.68758: 04
  OLAPQTS (EURUSD,M1)	    2.64350: 05
  OLAPQTS (EURUSD,M1)	    2.64158: 06
  OLAPQTS (EURUSD,M1)	    2.64934: 07
  OLAPQTS (EURUSD,M1)	    2.62854: 08
  OLAPQTS (EURUSD,M1)	    2.72117: 09
  OLAPQTS (EURUSD,M1)	    2.80259: 10
  OLAPQTS (EURUSD,M1)	    2.79681: 11
  OLAPQTS (EURUSD,M1)	    2.80850: 12
  OLAPQTS (EURUSD,M1)	    2.81435: 13
  OLAPQTS (EURUSD,M1)	    2.83489: 14
  OLAPQTS (EURUSD,M1)	    2.90745: 15
  OLAPQTS (EURUSD,M1)	    2.95804: 16
  OLAPQTS (EURUSD,M1)	    2.96799: 17
  OLAPQTS (EURUSD,M1)	    2.88021: 18
  OLAPQTS (EURUSD,M1)	    2.76605: 19
  OLAPQTS (EURUSD,M1)	    2.72036: 20
  OLAPQTS (EURUSD,M1)	    2.85615: 21
  OLAPQTS (EURUSD,M1)	    2.94224: 22
  OLAPQTS (EURUSD,M1)	    4.60560: 23
```

These values represent the single standard deviation, based on which it is possible to configure filters in scalping strategies or robots that
are based on volatility impulses.

Let's check the filling of the field with the range or price movements on a bar, operation of quantization with the specified cell size and
sorting.

For this purpose, switch back to EURUSD D1 and use the same filter by 2019. Also, set the following parameters:

- QuantGranularity=100 (5-digit points)
- SelectorX=quants
- FieldX=price range (OC)
- Aggregator=COUNT
- SortBy=value (descending)

Get the following result:

```
  OLAPQTS (EURUSD,D1)	Bars read: 12627
  OLAPQTS (EURUSD,D1)	CountAggregator<QUOTE_RECORD_FIELDS> FIELD_NONE [20]
  OLAPQTS (EURUSD,D1)	Filters: FilterRange::FilterSelector<QUOTE_RECORD_FIELDS>(FIELD_DATETIME)[1546300800.0 ... 1577836800.0];
  OLAPQTS (EURUSD,D1)	Selectors: 1
  OLAPQTS (EURUSD,D1)	X: QuantizationSelector<QUOTE_RECORD_FIELDS>(FIELD_PRICE_RANGE_OC) [20]
  OLAPQTS (EURUSD,D1)	Processed records: 259
  OLAPQTS (EURUSD,D1)	      [value]   [title]
  OLAPQTS (EURUSD,D1) [ 0] 72.00000 "0.0"
  OLAPQTS (EURUSD,D1) [ 1] 27.00000 "100.0"
  OLAPQTS (EURUSD,D1) [ 2] 24.00000 "-100.0"
  OLAPQTS (EURUSD,D1) [ 3] 24.00000 "-200.0"
  OLAPQTS (EURUSD,D1) [ 4] 21.00000 "200.0"
  OLAPQTS (EURUSD,D1) [ 5] 17.00000 "-300.0"
  OLAPQTS (EURUSD,D1) [ 6] 16.00000 "300.0"
  OLAPQTS (EURUSD,D1) [ 7] 12.00000 "-400.0"
  OLAPQTS (EURUSD,D1) [ 8]  8.00000 "500.0"
  OLAPQTS (EURUSD,D1) [ 9]  8.00000 "400.0"
  OLAPQTS (EURUSD,D1) [10]  6.00000 "-700.0"
  OLAPQTS (EURUSD,D1) [11]  6.00000 "-500.0"
  OLAPQTS (EURUSD,D1) [12]  6.00000 "700.0"
  OLAPQTS (EURUSD,D1) [13]  4.00000 "-600.0"
  OLAPQTS (EURUSD,D1) [14]  2.00000 "600.0"
  OLAPQTS (EURUSD,D1) [15]  2.00000 "1000.0"
  OLAPQTS (EURUSD,D1) [16]  1.00000 "-800.0"
  OLAPQTS (EURUSD,D1) [17]  1.00000 "-1100.0"
  OLAPQTS (EURUSD,D1) [18]  1.00000 "900.0"
  OLAPQTS (EURUSD,D1) [19]  1.00000 "-1000.0"
```

As it was expected, most of the bars (72) fall under the zero range, i.e. the price change at these bars did not exceed 100 points. Changes ±100
and ±200 points go further, and so on.

However, it is only the demonstration of OLAP possibilities in analyzing quotes. Now it is time to move to the next step and create trading
strategies using OLAP.

### Building trading strategies based on OLAP analysis of quotes. Part 1

Let's try to find out whether quotes have any patterns associated with connected intraday and intraweek cycles. If prevailing price movements
are not symmetric at some hours or on some days of the week, we can use this to open deals. In order to detect this cyclic patterns we'll need to
use the hour-of-day and day-of-week selectors. Selectors can be used sequentially one-by-one or simultaneously, each one at its own axis.
The second option is more preferable as it allows building more accurate data samples, taking into account two factors (cycles) at a time.
There is no difference for the program, which selector is set on X axis and which on Y. However, this affects the display of the results to the
user.

The ranges of these selectors are 24 (hours in a day) and 5 (weekdays), and therefore the cube size is 120. It is also possible to connect the
seasonal cyclic patterns within a year by selecting the "month-of-year" selector along the Z axis. For simplicity, we will work with a
two-dimensional cube now.

Price change inside the bar is presented in two fields: FIELD\_PRICE\_RANGE\_OC and FIELD\_PRICE\_RANGE\_HL. The first one provides the point
difference between the Open and Close prices, the second one shows the range between High and Low. Let's use the first one as the source of
statistics for potential deals. It should be decided now, which statistics will be calculated, i.e. which aggregator should be applied.

Oddly enough, the ProfitFactorAggregator aggregator may come in handy here. It was already described in previous articles. This aggregator
separately sums up positive and negative values of the specified field and returns their quotient: divides the positive and the negative
value taken modulo. Thus, if positive price increments prevail in some hypercube cell, the profit factor will be well above 1. If negative
values prevail, profit factor will be significantly below 1. In other words, all values which differ much from 1 indicate good conditions
for opening a long or a short deal. When the profit factor is above 1, buy deals can be profitable, while sells are more profitable with profit
factor below 1. The real profit factor of selling is the inverse for the calculated value.

Let's perform the analysis on EURUSD H1. Choose input parameters:

- SelectorX=hour-of-day
- SelectorY=day-of-week
- Filter1=field
- Filter1Field=datetime
- Filter1Value1=2019.01.01
- Filter1Value2=2020.01.01
- AggregatorType=Profit Factor
- AggregatorField=price range (OC)
- SortBy=value (descending)

The full list of results with 120 lines is not interesting to us. Here are the initial and final values denoting the most profitable buying and
selling options (they appear at the very beginning and the very end due to the enabled sorting).

```
  OLAPQTS (EURUSD,H1)	Bars read: 137597
  OLAPQTS (EURUSD,H1)	ProfitFactorAggregator<QUOTE_RECORD_FIELDS> FIELD_PRICE_RANGE_OC [120]
  OLAPQTS (EURUSD,H1)	Filters: FilterRange::FilterSelector<QUOTE_RECORD_FIELDS>(FIELD_DATETIME)[1546300800.0 ... 1577836800.0];
  OLAPQTS (EURUSD,H1)	Selectors: 2
  OLAPQTS (EURUSD,H1)	X: DayHourSelector<QUOTE_RECORD_FIELDS>(FIELD_DATETIME) [24]
  OLAPQTS (EURUSD,H1)	Y: WorkWeekDaySelector<QUOTE_RECORD_FIELDS>(FIELD_DATETIME) [5]
  OLAPQTS (EURUSD,H1)	Processed records: 6196
  OLAPQTS (EURUSD,H1)	      [value]           [title]
  OLAPQTS (EURUSD,H1) [  0] 5.85417 "00; 1`Monday"
  OLAPQTS (EURUSD,H1) [  1] 5.79204 "00; 5`Friday"
  OLAPQTS (EURUSD,H1) [  2] 5.25194 "00; 4`Thursday"
  OLAPQTS (EURUSD,H1) [  3] 4.10104 "01; 4`Thursday"
  OLAPQTS (EURUSD,H1) [  4] 4.00463 "01; 2`Tuesday"
  OLAPQTS (EURUSD,H1) [  5] 2.93725 "01; 3`Wednesday"
  OLAPQTS (EURUSD,H1) [  6] 2.50000 "00; 3`Wednesday"
  OLAPQTS (EURUSD,H1) [  7] 2.44557 "15; 1`Monday"
  OLAPQTS (EURUSD,H1) [  8] 2.43496 "04; 5`Friday"
  OLAPQTS (EURUSD,H1) [  9] 2.36278 "20; 3`Wednesday"
  OLAPQTS (EURUSD,H1) [ 10] 2.33917 "04; 4`Thursday"
  ...
  OLAPQTS (EURUSD,H1) [110] 0.49096 "09; 3`Wednesday"
  OLAPQTS (EURUSD,H1) [111] 0.48241 "13; 4`Thursday"
  OLAPQTS (EURUSD,H1) [112] 0.45891 "19; 4`Thursday"
  OLAPQTS (EURUSD,H1) [113] 0.45807 "19; 3`Wednesday"
  OLAPQTS (EURUSD,H1) [114] 0.44993 "14; 3`Wednesday"
  OLAPQTS (EURUSD,H1) [115] 0.44513 "23; 4`Thursday"
  OLAPQTS (EURUSD,H1) [116] 0.42693 "23; 1`Monday"
  OLAPQTS (EURUSD,H1) [117] 0.37026 "10; 1`Monday"
  OLAPQTS (EURUSD,H1) [118] 0.34662 "23; 3`Wednesday"
  OLAPQTS (EURUSD,H1) [119] 0.19705 "23; 5`Friday"
```

Please note that the labels of two dimensions X and Y (which are used for the hour and the day of the week) are shown for each value.

The received values are not entirely correct because they ignore spread. Here custom fields can be used to solve the problem. For example, to
assess the potential effect of spreads, let's save in the first custom field the bar range minus spread. For the second field, bar direction
minus spread will be calculated.

```
  virtual void fillCustomFields() override
  {
    const double newBarRange = get(FIELD_PRICE_RANGE_OC);
    const double spread = get(FIELD_SPREAD);

    set(FIELD_CUSTOM1, MathSign(newBarRange) * (MathAbs(newBarRange) - spread));
    set(FIELD_CUSTOM2, MathSign(newBarRange) * MathSign(MathAbs(newBarRange) - spread));

    // ...
   }
```

Select custom field 1 as the aggregator. Here is the result:

```
  OLAPQTS (EURUSD,H1)	Bars read: 137598
  OLAPQTS (EURUSD,H1)	ProfitFactorAggregator<QUOTE_RECORD_FIELDS> FIELD_CUSTOM1 [120]
  OLAPQTS (EURUSD,H1)	Filters: FilterRange::FilterSelector<QUOTE_RECORD_FIELDS>(FIELD_DATETIME)[1546300800.0 ... 1577836800.0];
  OLAPQTS (EURUSD,H1)	Selectors: 2
  OLAPQTS (EURUSD,H1)	X: DayHourSelector<QUOTE_RECORD_FIELDS>(FIELD_DATETIME) [24]
  OLAPQTS (EURUSD,H1)	Y: WorkWeekDaySelector<QUOTE_RECORD_FIELDS>(FIELD_DATETIME) [5]
  OLAPQTS (EURUSD,H1)	Processed records: 6196
  OLAPQTS (EURUSD,H1)	      [value]           [title]
  OLAPQTS (EURUSD,H1) [  0] 6.34239 "00; 5`Friday"
  OLAPQTS (EURUSD,H1) [  1] 5.63981 "00; 1`Monday"
  OLAPQTS (EURUSD,H1) [  2] 5.15044 "00; 4`Thursday"
  OLAPQTS (EURUSD,H1) [  3] 4.41176 "01; 2`Tuesday"
  OLAPQTS (EURUSD,H1) [  4] 4.18052 "01; 4`Thursday"
  OLAPQTS (EURUSD,H1) [  5] 3.04167 "01; 3`Wednesday"
  OLAPQTS (EURUSD,H1) [  6] 2.60000 "00; 3`Wednesday"
  OLAPQTS (EURUSD,H1) [  7] 2.53118 "15; 1`Monday"
  OLAPQTS (EURUSD,H1) [  8] 2.50118 "04; 5`Friday"
  OLAPQTS (EURUSD,H1) [  9] 2.47716 "04; 4`Thursday"
  OLAPQTS (EURUSD,H1) [ 10] 2.46208 "20; 3`Wednesday"
  OLAPQTS (EURUSD,H1) [ 11] 2.20858 "03; 5`Friday"
  OLAPQTS (EURUSD,H1) [ 12] 2.11964 "03; 1`Monday"
  OLAPQTS (EURUSD,H1) [ 13] 2.11123 "19; 2`Tuesday"
  OLAPQTS (EURUSD,H1) [ 14] 2.10998 "01; 1`Monday"
  OLAPQTS (EURUSD,H1) [ 15] 2.07638 "10; 4`Thursday"
  OLAPQTS (EURUSD,H1) [ 16] 1.95498 "09; 5`Friday"
  ...
  OLAPQTS (EURUSD,H1) [105] 0.59029 "11; 5`Friday"
  OLAPQTS (EURUSD,H1) [106] 0.55008 "14; 5`Friday"
  OLAPQTS (EURUSD,H1) [107] 0.54643 "13; 3`Wednesday"
  OLAPQTS (EURUSD,H1) [108] 0.50484 "09; 3`Wednesday"
  OLAPQTS (EURUSD,H1) [109] 0.50000 "22; 1`Monday"
  OLAPQTS (EURUSD,H1) [110] 0.49744 "06; 2`Tuesday"
  OLAPQTS (EURUSD,H1) [111] 0.46686 "13; 4`Thursday"
  OLAPQTS (EURUSD,H1) [112] 0.44753 "19; 3`Wednesday"
  OLAPQTS (EURUSD,H1) [113] 0.44499 "19; 4`Thursday"
  OLAPQTS (EURUSD,H1) [114] 0.43838 "14; 3`Wednesday"
  OLAPQTS (EURUSD,H1) [115] 0.41290 "23; 4`Thursday"
  OLAPQTS (EURUSD,H1) [116] 0.39770 "23; 1`Monday"
  OLAPQTS (EURUSD,H1) [117] 0.35586 "10; 1`Monday"
  OLAPQTS (EURUSD,H1) [118] 0.34721 "23; 3`Wednesday"
  OLAPQTS (EURUSD,H1) [119] 0.18769 "23; 5`Friday"
```

The values mean that profit should be obtained from trading operations performed on Thursday: buying at 0, 1 and 4 a.m. and selling at 7 and 11
p.m. (19 and 23). On Friday, it is recommended to buy at 0, 3, 4, 9 in the morning and to sell at 11, 14 and 23. However, selling on Friday at 23 can be
risky due to soon session closure and a potential unfavorable gap (btw, gap analysis can also be easily automated here using custom fields).
In this project, the acceptable level of profit factor is set to 2 or more (for selling, respectively, 0.5 or less). In practice, the values
are usually worse than theoretical result, therefore a certain safety margin should be provided.

Also, profit factor should be calculated not only by the bar range, but also by the number of bullish and bearish candlesticks. For this purpose,
select the bar type (form) as the aggregator. Sometimes, a profit amount can be formed of one or two candlesticks of extraordinary size. Such
spikes will become more noticeable if we compare profit factor by candlestick sizes and profit factor by the number of bars in different
directions.

Generally speaking, we do not necessarily need to analyze data on the same timeframe that is selected in the lower selector by the date field. This
time, we used "hour-of-day" on the H1 timeframe. Data can be analyzed on any timeframe lower than or equal to the lower selector by the date
field. For example, we can perform a similar analysis on M15 and preserve grouping by hours using the "hour-of-day" selector. This way we
will determine the profit factor for 15-minute bars. However, for the current strategy we would need to additionally specify the entry
moment within an hour. This can be done by analyzing the most likely candlestick formation ways at each hour (i.e. after which
counter-movements the main bar body is formed). An example of the "digitization" of bar tails is available in comments in the OLAPQTS source
code.

A more visual method to identify stable "buying" and "selling" bars in the hour-by-hour and day-by-day analysis is to use
ProgressiveTotalAggregator. In this case, the "ordinal number" selector (consecutive analysis of all bars) should be set for the X axis X,
and "hour-of-day" and "day-of-week" selectors should be set for Y and Z, as well as the previous aggregation field "custom 1" should be used.
This would produce actual trading balance curves for each specific one-hour bar. But logging and analysis of such data is not convenient and
thus this method is more suitable with a connected graphical display. This would make the implementation even more complicated, that is why
let's use logs.

Let's create the SingleBar trading Expert Advisor that performs deals in accordance with the cycles found using the OLAP analysis. The main
parameters allow configuring scheduled trading:

```
  input string BuyHours = "";
  input string SellHours = "";
  input uint ActiveDayOfWeek = 0;
```

The String parameters BuyHours and SellHours accept lists of hours, in which buy and sell deals should be opened, respectively. Hours in each
list are separated by commas. The weekday is set in ActiveDayOfWeek (values from 1 for Monday to 5 for Friday). At the testing stage, one
specific day is checked. However, in the future the Expert Advisor should support a schedule with all days of the week. If ActiveDayOfWeek is
set to 0, the EA will trade on all days using the same schedule. However, this requires a preliminary OLAP analysis with the variation of
"hour-of-day", while resetting "day-of-week" along Y. If you wish, you can test this strategy yourself.

Settings are read and checked in OnInit:

```
  int buyHours[], sellHours[];

  int parseHours(const string &data, int &result[])
  {
    string str[];
    const int n = StringSplit(data, ',', str);
    ArrayResize(result, n);
    for(int i = 0; i < n; i++)
    {
      result[i] = (int)StringToInteger(str[i]);
     }
    return n;
   }

  int OnInit()
  {
    const int trend = parseHours(BuyHours, buyHours);
    const int reverse = parseHours(SellHours, sellHours);

    return trend > 0 || reverse > 0 ? INIT_SUCCEEDED : INIT_PARAMETERS_INCORRECT;
   }
```

In OnTick handler, the lists of trading hours will be checked and the special 'mode' variable will be set to +1 or -1 if the current hour is found
in any of them. If the hour is not found, 'mode' will be equal to 0, which means that all existing positions should be closed without opening new
positions. If there are no orders and 'mode' is not equal to zero, a new position should be opened. If there is an open position in the same
direction as the schedule suggests, the position is preserved. If the signal direction is opposite to the open position, the position
should be reversed. Only one position can be open at a time.

```
  template<typename T>
  int ArrayFind(const T &array[], const T value)
  {
    const int n = ArraySize(array);
    for(int i = 0; i < n; i++)
    {
      if(array[i] == value) return i;
     }
    return -1;
   }

  void OnTick()
  {
    MqlTick tick;
    if(!SymbolInfoTick(_Symbol, tick)) return;

    const int h = TimeHour(TimeCurrent());

    int mode = 0;

    if(ArrayFind(buyHours, h) > -1)
    {
      mode = +1;
     }
    else
    if(ArrayFind(sellHours, h) > -1)
    {
      mode = -1;
     }

    if(ActiveDayOfWeek != 0 && ActiveDayOfWeek != _TimeDayOfWeek()) mode = 0; // skip all days except specified

    // pick up existing orders (if any)
    const int direction = CurrentOrderDirection();

    if(mode == 0)
    {
      if(direction != 0)
      {
        OrdersCloseAll();
       }
      return;
     }

    if(direction != 0) // there exist open orders
    {
      if(mode == direction) // keep direction
      {
        return; // existing trade goes on
       }
      OrdersCloseAll();
     }


    const int type = mode > 0 ? OP_BUY : OP_SELL;

    const double p = type == OP_BUY ? SymbolInfoDouble(_Symbol, SYMBOL_ASK) : SymbolInfoDouble(_Symbol, SYMBOL_BID);

    OrderSend(_Symbol, type, Lot, p, 100, 0, 0);
   }
```

Deals are performed only at bar opening, as it is set by the trading strategy. Additional functions ArrayFind, CurrentOrderDirection and
OrdersCloseAll are shown below. All these functions, as well as the EA use the [MT4Orders](https://www.mql5.com/en/code/16006)
library for an easier operation with the trading API. In addition, the attached MT4Bridge/MT4Time.mqh code is used for working with dates.

```
  int CurrentOrderDirection(const string symbol = NULL)
  {
    for(int i = OrdersTotal() - 1; i >= 0; i--)
    {
      if(OrderSelect(i, SELECT_BY_POS))
      {
        if(OrderType() <= OP_SELL && (symbol == NULL || symbol == OrderSymbol()))
        {
          return OrderType() == OP_BUY ? +1 : -1;
         }
       }
     }
    return 0;
   }

  void OrdersCloseAll(const string symbol = NULL, const int type = -1) // OP_BUY or OP_SELL
  {
    for(int i = OrdersTotal() - 1; i >= 0; i--)
    {
      if(OrderSelect(i, SELECT_BY_POS))
      {
        if(OrderType() <= OP_SELL && (type == -1 || OrderType() == type) && (symbol == NULL || symbol == OrderSymbol()))
        {
          OrderClose(OrderTicket(), OrderLots(), OrderType() == OP_BUY ? SymbolInfoDouble(OrderSymbol(), SYMBOL_BID) : SymbolInfoDouble(OrderSymbol(), SYMBOL_ASK), 100);
         }
       }
     }
   }
```

The full source code is attached below. One of the things that were skipped in this article is the theoretical calculation of profit factor
following the logic used in the OLAP engine. This allows comparing the theoretical value with the practical profit factor from testing
results. These two values are usually similar but do not match exactly. Of course, the theoretical profit factor makes sense only if the EA is
set to trade only in one direction, either buying (BuyHours) or selling (SellHours). Otherwise, the two modes overlap and the theoretical
PF tends to 1. Also, the theoretical profitable profit factor for sell deals is indicated by values less than 1, since it is the inverse of the
normal profit factor. For example, the theoretical selling PF of 0.5 is the same as the practical PF in the tester equal to 2. For the buying
direction, theoretical and practical PFs are similar: values above 1 mean profit, those below 1 mean loss.

Let's test the SingleBar EA in 2019, using EURUSD H1 data. Set the found trading hour values for Friday:

- BuyHours=0,4,3,9
- SellHours=23,14,11
- ActiveDayOfWeek=5

The order in which hours are specified is not important. Here they are specified in the descending order by expected profitability. The
testing result is as follows:

[![Fig.4 Report of SingleBar EA trading the found schedule of Fridays for year 2019, EURUSD H1](https://c.mql5.com/2/37/olap3friday2019__1.png)](https://c.mql5.com/2/37/olap3friday2019.png "https://c.mql5.com/2/37/olap3friday2019.png")

**Fig.4 Report of SingleBar EA trading the found schedule of Fridays for year 2019,**
**EURUSD H1**

The results are good. But this is not surprising, because initial analysis was also performed for this year. Let's shift the testing start
date to the beginning of 2018 to see the performance of the found patterns.

[![Fig.5 Report of SingleBar EA trading the found schedule of Fridays of 2019 in the interval of 2018-2019, EURUSD H1](https://c.mql5.com/2/37/olap3friday2018__1.png)](https://c.mql5.com/2/37/olap3friday2018.png "https://c.mql5.com/2/37/olap3friday2018.png")

**Fig.5 Report of SingleBar EA trading the found schedule of Fridays of 2019 in the**
**interval of 2018-2019, EURUSD H1**

Although the results are worse, you can see that the patterns worked well since mid-2018 and thus they could be found earlier using OLAP analysis to
trade "in the present future". However, search for an optimal analysis period and determining of the duration of the found patterns is
another big topic. In a sense, OLAP analysis requires the same optimization as Expert Advisors. Theoretically, it is possible to implement
an approach where OLAP is built into an EA that is run in the tester in different history intervals of different lengths and different start
dates; for each of them a forward test is then performed. This is the Cluster Walk-Forward technology, however it is not fully supported in
MetaTrader (at the time of writing, only the automated launch of forward tests is possible, but no shift in starting dates or period resizing
is possible, and thus one has to implement it themselves using MQL5 or other tools, such as shell scripts).

In general, OLAP should be seen as a research tool that helps identifying areas for more thorough analysis using other means, such as
traditional optimization of Expert Advisor and other. Further we will see how the OLAP engine can be built into an Expert Advisor and used on
the fly, both in the tester and on-line.

Let's check the current trading strategy for a few other days. Both good and bad days are deliberately shown here.

[![Fig.6.a Report of SingleBar EA trading on Tuesdays, 2018-2019, based on the 2019 analysis, EURUSD H1](https://c.mql5.com/2/37/olap3tuesday2018__1.png)](https://c.mql5.com/2/37/olap3tuesday2018.png "https://c.mql5.com/2/37/olap3tuesday2018.png")

**Fig.6.a Report of SingleBar EA trading on Tuesdays, 2018-2019, based on the 2019**
**analysis, EURUSD H1**

[![Fig.6.b Report of SingleBar EA trading on Wednesdays, 2018-2019, based on the 2019 analysis, EURUSD H1](https://c.mql5.com/2/38/olap3wednesday2018.png)](https://c.mql5.com/2/37/olap3wednesday2018.png "https://c.mql5.com/2/37/olap3wednesday2018.png")

**Fig.6.b Report of SingleBar EA trading on Wednesdays, 2018-2019, based on the 2019**
**analysis, EURUSD H1**

[![Fig.6.c Report of SingleBar EA trading on Thursdays, 2018-2019, based on the 2019 analysis, EURUSD H1](https://c.mql5.com/2/38/olap3thursday2018__1.png)](https://c.mql5.com/2/38/olap3thursday2018.png "https://c.mql5.com/2/38/olap3thursday2018.png")

**Fig.6.c Report of SingleBar EA trading on Thursdays, 2018-2019, based on the 2019**
**analysis, EURUSD H1**

As expected, the ambiguous trading behavior on different days of the week demonstrates that there are no universal solutions and this one
requires further improvement.

Let's see what trading schedules could be found if we analyzed quotes in a longer period, for example, from 2015 to 2019, and then traded in the
forward mode in 2019.

```
  OLAPQTS (EURUSD,H1)	Bars read: 137606
  OLAPQTS (EURUSD,H1)	ProfitFactorAggregator<QUOTE_RECORD_FIELDS> FIELD_CUSTOM3 [120]
  OLAPQTS (EURUSD,H1)	Filters: FilterRange::FilterSelector<QUOTE_RECORD_FIELDS>(FIELD_DATETIME)[1420070400.0 ... 1546300800.0];
  OLAPQTS (EURUSD,H1)	Selectors: 2
  OLAPQTS (EURUSD,H1)	X: DayHourSelector<QUOTE_RECORD_FIELDS>(FIELD_DATETIME) [24]
  OLAPQTS (EURUSD,H1)	Y: WorkWeekDaySelector<QUOTE_RECORD_FIELDS>(FIELD_DATETIME) [5]
  OLAPQTS (EURUSD,H1)	Processed records: 24832
  OLAPQTS (EURUSD,H1)	      [value]           [title]
  OLAPQTS (EURUSD,H1) [  0] 2.04053 "01; 3`Wednesday"
  OLAPQTS (EURUSD,H1) [  1] 1.78702 "01; 4`Thursday"
  OLAPQTS (EURUSD,H1) [  2] 1.75055 "15; 1`Monday"
  OLAPQTS (EURUSD,H1) [  3] 1.71793 "00; 1`Monday"
  OLAPQTS (EURUSD,H1) [  4] 1.69210 "00; 4`Thursday"
  OLAPQTS (EURUSD,H1) [  5] 1.64361 "04; 3`Wednesday"
  OLAPQTS (EURUSD,H1) [  6] 1.63956 "20; 3`Wednesday"
  OLAPQTS (EURUSD,H1) [  7] 1.62157 "05; 3`Wednesday"
  OLAPQTS (EURUSD,H1) [  8] 1.53032 "00; 3`Wednesday"
  OLAPQTS (EURUSD,H1) [  9] 1.49733 "16; 1`Monday"
  OLAPQTS (EURUSD,H1) [ 10] 1.48539 "01; 5`Friday"
  ...
  OLAPQTS (EURUSD,H1) [109] 0.74241 "16; 5`Friday"
  OLAPQTS (EURUSD,H1) [110] 0.70346 "13; 3`Wednesday"
  OLAPQTS (EURUSD,H1) [111] 0.68990 "23; 2`Tuesday"
  OLAPQTS (EURUSD,H1) [112] 0.66238 "23; 4`Thursday"
  OLAPQTS (EURUSD,H1) [113] 0.66176 "14; 4`Thursday"
  OLAPQTS (EURUSD,H1) [114] 0.62968 "13; 1`Monday"
  OLAPQTS (EURUSD,H1) [115] 0.62585 "23; 5`Friday"
  OLAPQTS (EURUSD,H1) [116] 0.60150 "14; 5`Friday"
  OLAPQTS (EURUSD,H1) [117] 0.55621 "11; 2`Tuesday"
  OLAPQTS (EURUSD,H1) [118] 0.54919 "23; 3`Wednesday"
  OLAPQTS (EURUSD,H1) [119] 0.49804 "11; 3`Wednesday"
```

As you can see, an increase in the period leads to a decrease in the profitability of each individual hour. The generalization begins to play
against the pattern search at some point. Wednesday seems to be the most profitable day. However, the behavior is not very stable in the
forward period. For example, consider the following settings:

- BuyHours=1,4,20,5,0
- SellHours=11,23,13
- ActiveDayOfWeek=3

The resulting report is as follows:

[![Fig.7 Report of SingleBar EA trading on Wednesdays, 2015-2020, based on the analysis excluding 2019, EURUSD H1](https://c.mql5.com/2/38/olap3wednesday2015__1.png)](https://c.mql5.com/2/38/olap3wednesday2015.png "https://c.mql5.com/2/38/olap3wednesday2015.png")

**Fig.7 Report of SingleBar EA trading on Wednesdays, 2015-2020, based on the**
**analysis excluding 2019, EURUSD H1**

A more versatile technique is needed to solve this problem, while OLAP is only one of multiple required tools. Furthermore, it makes sense
to look for more complex (multifactorial) patterns. Let's try to create another trading strategy taking into account not only the time
cycle, but also the previous bar direction.

### Building trading strategies based on OLAP analysis of quotes. Part 2

It can be assumed that the direction of each bar can depend on the direction of the previous one to some extent. This dependence is most likely
to have a similar cyclical character connected by intraday and intraweek fluctuations which were detected in the previous section. In
other words, in addition to accumulating bar sizes and directions by the hours and days of the week in an OLAP analysis, it is also necessary to
somehow take into account the characteristics of the previous bar. Let's use the remaining custom fields for this.

In the third custom field the "asymmetric" covariance of two adjacent bars will be calculated. The ordinary covariance which is calculated
as a product of the price movements ranges inside the bars, taking into account the direction (plus for increase and minus for decrease),
does not have special predictive value, since the previous and next bars are equivalent in the obtained covariance value. However, trading
decisions are only efficient for the next bar, although they are made based on the previous bar. In other words, high covariance due to the
large movements of the previous bar has already been performed since such a bar is in history. That is why we'll try to use the "asymmetric"
covariance formula, in which only the range of the next bar is taken into account, along with the sign of the product of multiplying with the
previous bar.

This field allows testing two strategies: trend and reversal. For example, if we use the profit factor aggregator in this field, then values
greater than 1 indicate that trading in the previous bar direction is profitable; values less than 1 indicate that the opposite direction is
profitable. As in previous calculations, extreme values (much greater than 1 or much lower than 1) mean that trend or reversal operations,
respectively, will be more profitable.

In the fourth custom field, we will save the sign of whether adjacent bars are in the same direction (+1) or in different directions (-1).
Thus, we well be able to determine the number of adjacent reversal bars using aggregators, as well as the efficiency of entries for the trend
and reversal strategies.

Since the bars are always analyzed in chronological order (this order is provided by the adapter), we can save the previous bar size and spread
required for calculations in static variables. Of course, this can be done as long as a single instance of the quotes adapter is used (its
instance is created in the header file by default). This is suitable for our example and is easier to understand. However, generally the
adapter should be passing to the custom record constructor (such as CustomQuotesBaseRecord) and further to the fillCustomFields method a
certain container which would allow saving and restoring the state, for example as reference to an array: fillCustomFields(double
&bundle\[\]).

```
  class CustomQuotesRecord: public QuotesRecord
  {
    private:
      static double previousBarRange;
      static double previousSpread;

    public:
      // ...

      virtual void fillCustomFields() override
      {
        const double newBarRange = get(FIELD_PRICE_RANGE_OC);
        const double spread = get(FIELD_SPREAD);

        // ...

        if(MathAbs(previousBarRange) > previousSpread)
        {
          double mult = newBarRange * previousBarRange;
          double value = MathSign(mult) * MathAbs(newBarRange);

          // this is an attempt to approximate average losses due to spreads
          value += MathSignNonZero(value) * -1 * MathMax(spread, previousSpread);

          set(FIELD_CUSTOM3, value);
          set(FIELD_CUSTOM4, MathSign(mult));
         }
        else
        {
          set(FIELD_CUSTOM3, 0);
          set(FIELD_CUSTOM4, 0);
         }

        previousBarRange = newBarRange;
        previousSpread = spread;
       }

  };
```

The values of OLAPQTS inputs should be modified. The main change concerns the selection of "custom 3" in AggregatorField. The following
parameters remain unchanged: selectors by X and Y, aggregator type (PF) and sorting. Also, the date filter is changed.

- SelectorX=hour-of-day
- SelectorY=day-of-week
- Filter1=field
- Filter1Field=datetime
- Filter1Value1=2018.01.01
- Filter1Value2=2019.01.01
- AggregatorType=Profit Factor
- AggregatorField=custom 3
- SortBy=value (descending)

As we have already seen when analyzing quotes starting from 2015, the choice of a longer period is more suitable for the systems which aim to
determine cyclicity — it would correspond to the month-of-year selector. In our example, where we use hour and day of the week selectors, we
will analyze only 2018 and then will perform a forward test for 2019.

```
  OLAPQTS (EURUSD,H1)	Bars read: 137642
  OLAPQTS (EURUSD,H1)	Aggregator: ProfitFactorAggregator<QUOTE_RECORD_FIELDS> FIELD_CUSTOM3 [120]
  OLAPQTS (EURUSD,H1)	Filters: FilterRange::FilterSelector<QUOTE_RECORD_FIELDS>(FIELD_DATETIME)[1514764800.0 ... 1546300800.0];
  OLAPQTS (EURUSD,H1)	Selectors: 2
  OLAPQTS (EURUSD,H1)	X: DayHourSelector<QUOTE_RECORD_FIELDS>(FIELD_DATETIME) [24]
  OLAPQTS (EURUSD,H1)	Y: WorkWeekDaySelector<QUOTE_RECORD_FIELDS>(FIELD_DATETIME) [5]
  OLAPQTS (EURUSD,H1)	Processed records: 6203
  OLAPQTS (EURUSD,H1)	      [value]           [title]
  OLAPQTS (EURUSD,H1) [  0] 2.65010 "23; 1`Monday"
  OLAPQTS (EURUSD,H1) [  1] 2.37966 "03; 1`Monday"
  OLAPQTS (EURUSD,H1) [  2] 2.33875 "04; 4`Thursday"
  OLAPQTS (EURUSD,H1) [  3] 1.96317 "20; 3`Wednesday"
  OLAPQTS (EURUSD,H1) [  4] 1.91188 "18; 2`Tuesday"
  OLAPQTS (EURUSD,H1) [  5] 1.89293 "23; 3`Wednesday"
  OLAPQTS (EURUSD,H1) [  6] 1.87159 "12; 1`Monday"
  OLAPQTS (EURUSD,H1) [  7] 1.78903 "15; 5`Friday"
  OLAPQTS (EURUSD,H1) [  8] 1.74461 "01; 4`Thursday"
  OLAPQTS (EURUSD,H1) [  9] 1.73821 "13; 2`Tuesday"
  OLAPQTS (EURUSD,H1) [ 10] 1.73244 "14; 2`Tuesday"
  ...
  OLAPQTS (EURUSD,H1) [110] 0.57331 "22; 4`Thursday"
  OLAPQTS (EURUSD,H1) [111] 0.51515 "07; 5`Friday"
  OLAPQTS (EURUSD,H1) [112] 0.50202 "05; 5`Friday"
  OLAPQTS (EURUSD,H1) [113] 0.48557 "04; 2`Tuesday"
  OLAPQTS (EURUSD,H1) [114] 0.46313 "23; 2`Tuesday"
  OLAPQTS (EURUSD,H1) [115] 0.44182 "00; 2`Tuesday"
  OLAPQTS (EURUSD,H1) [116] 0.40907 "13; 1`Monday"
  OLAPQTS (EURUSD,H1) [117] 0.38230 "10; 1`Monday"
  OLAPQTS (EURUSD,H1) [118] 0.36296 "22; 5`Friday"
  OLAPQTS (EURUSD,H1) [119] 0.29462 "17; 5`Friday"
```

Let's create another Expert Advisor, NextBar, in order to test the strategy implemented in the "custom 3" field. Using the EA, we can check the
found trading opportunities in the strategy tester. The general Expert Advisor structure is similar to SingleBar: the same parameters,
functions and code fragments are used. The trading logic is more complicated, you can view it in the attached source file.

Let's select the most attractive combinations of hours (with the PF 2 and above or 0.5 and below), for example for Monday:

- PositiveHours=23,3
- NegativeHours=10,13
- ActiveDayOfWeek=1

Run a test in the range 2018.01.01-2019.05.01:

[![Fig.8 Report of NextBar EA trading in the interval 01.01.2018-01.05.2019 after OLAP analysis for 2018, EURUSD H1](https://c.mql5.com/2/38/olap3next-monday2018-2019-5__1.png)](https://c.mql5.com/2/38/olap3next-monday2018-2019-5.png "https://c.mql5.com/2/38/olap3next-monday2018-2019-5.png")

**Fig.8 Report of NextBar EA trading in the interval 01.01.2018-01.05.2019 after**
**OLAP analysis for 2018, EURUSD H1**

The strategy still worked successfully in January 2019, after which a losing series began. We need to somehow find out the lifespan of the
patterns and learn how to change them on the go.

### Adaptive trading based on OLAP analysis of quotes

So far, we have been using a special non-trading EA OLAPQTS for OLAP analysis while testing separate hypothesis using individually
developed EAs. A more logical and convenient solution would be to have an OLAP engine built into an Expert Advisor. Thus, the robot would be
able to automatically analyze quotes at a given periodicity and adjust the trading schedule. In addition, by implementing the main
parameters in the EA, we can optimize them using a method, which can emulate the Walk-Forward technique mentioned above. The EA is called
OLAPQRWF, which is an abbreviation of OLAP of Quotes with Rolling Walk-Forward.

The main inputs of the Expert Advisor:

```
  input int BarNumberLookBack = 2880; // BarNumberLookBack (week: 120 H1, month: 480 H1, year: 5760 H1)
  input double Threshold = 2.0; // Threshold (PF >= Threshold && PF <= 1/Threshold)
  input int Strategy = 0; // Strategy (0 - single bar, 1 - adjacent bars)
```

- BarNumberLookBack sets the number of historical bars, in which the OLAP analysis will be performed (the H1 timeframe is assumed here).
- Threshold is the profit factor threshold which is enough to open a deal.
- Strategy is the number of the tested strategy (currently we have two strategies: 0 — statistics of direction of separate bars, 1 — statistics
of directions two adjacent bars).

In addition, we need to specify the frequency with which the OLAP cube will be recalculated.

```
  enum UPDATEPERIOD
  {
    monthly,
    weekly
  };

  input UPDATEPERIOD Update = monthly;
```

In addition to strategies, we can select custom fields by which the aggregator is calculated. Fields 1 and 3 are calculated taking into
account bar range (respectively for strategies 0 and 1), while fields 2 and 4 only take into account the number of bars in each direction.

```
  enum CUSTOMFIELD
  {
    range,
    count
  };

  input CUSTOMFIELD CustomField = range;
```

The CustomQuotesRecord class is inherited from OLAPQTS unchanged. All previously used parameters for configuring selectors, filters and
aggregators are set as constants or as global variables (if they should be changed depending on the strategy), without changing their
names.

```
  const SELECTORS SelectorX = SELECTOR_DAYHOUR;
  const ENUM_FIELDS FieldX = FIELD_DATETIME;

  const SELECTORS SelectorY = SELECTOR_WEEKDAY;
  const ENUM_FIELDS FieldY = FIELD_DATETIME;

  const SELECTORS SelectorZ = SELECTOR_NONE;
  const ENUM_FIELDS FieldZ = FIELD_NONE;

  const SELECTORS _Filter1 = SELECTOR_FILTER;
  const ENUM_FIELDS _Filter1Field = FIELD_INDEX;
        int _Filter1value1 = -1; // to be filled with index of first bar to process
  const int _Filter1value2 = -1;

  const AGGREGATORS _AggregatorType = AGGREGATOR_PROFITFACTOR;
        ENUM_FIELDS _AggregatorField = FIELD_CUSTOM1;
  const SORT_BY _SortBy = SORT_BY_NONE;
```

Please note that bars will be filtered not by time but by quantity using FIELD\_INDEX. The actual value for \_Filter1value1 will be calculated as
the difference between the total number of bars and the specified BarNumberLookBack. Thus, the EA will always calculate the last
BarNumberLookBack bars.

The EA will trade in bar mode from the OnTick handler.

```
  bool freshStart = true;

  void OnTick()
  {
    if(!isNewBar()) return;

    if(Bars(_Symbol, _Period) < BarNumberLookBack) return;

    const int m0 = TimeMonth(iTime(_Symbol, _Period, 0));
    const int w0 = _TimeDayOfWeek();
    const int m1 = TimeMonth(iTime(_Symbol, _Period, 1));
    const int w1 = _TimeDayOfWeek();

    static bool success = false;

    if((Update == monthly && m0 != m1)
    || (Update == weekly && w0 < w1)
    || freshStart)
    {
      success = calcolap();
      freshStart = !success;
     }

    //...
   }
```

Depending on the analysis frequency, wait for the month or week to change and run OLAP in the 'calcolap' function.

```
  bool calcolap()
  {
    _Filter1value1 = Bars(_Symbol, _Period) - BarNumberLookBack;
    _AggregatorField = Strategy == 0 ? (ENUM_FIELDS)(FIELD_CUSTOM1 + CustomField) : (ENUM_FIELDS)(FIELD_CUSTOM3 + CustomField);

    _defaultQuotesAdapter.reset();
    const int processed =
    _defaultEngine.process(_selectorArray, _selectorField,
          _AggregatorType, _AggregatorField,
          stats,                              // custom display object
          _SortBy,
          _Filter1value1, _Filter1value2);

    return processed == BarNumberLookBack;
   }
```

This code part is already familiar. A couple of modifications concern the selection of the aggregation field according to input parameters as
well as setting of the index of the first analyzed bar.

Another important change implies the use of the special display object (stats), which will be called by the OLAP engine after performing
analysis.

```
  class MyOLAPStats: public Display
  {
    // ...
    public:
      virtual void display(MetaCube *cube, const SORT_BY sortby = SORT_BY_NONE, const bool identity = false) override
      {
        // ...
       }

      void trade(const double threshold, const double lots, const int strategy = 0)
      {
        // ...
       }
  };

  MyOLAPStats stats;
```

Since this object will determine the best trading hours from the obtained statistics, it is convenient to entrust trading to the same object via
the reserved 'trade' method. Thus, the following is added to OnTick:

```
  void OnTick()
  {
    // ...

    if(success)
    {
      stats.trade(Threshold, Lot, Strategy);
     }
    else
    {
      OrdersCloseAll();
     }
   }
```

Now, let's consider the MyOLAPStats class in more detail. OLAP analysis results are processed by the 'display' methods (the main virtual
method of the display) and saveVector (auxiliary).

```
  #define N_HOURS   24
  #define N_DAYS     5
  #define AXIS_HOURS 0
  #define AXIS_DAYS  1

  class MyOLAPStats: public Display
  {
    private:
      bool filled;
      double index[][3]; // value, hour, day
      int cursor;

    protected:
      bool saveVector(MetaCube *cube, const int &consts[], const SORT_BY sortby = SORT_BY_NONE)
      {
        PairArray *result = NULL;
        cube.getVector(0, consts, result, sortby);
        if(CheckPointer(result) == POINTER_DYNAMIC)
        {
          const int n = ArraySize(result.array);

          if(n == N_HOURS)
          {
            for(int i = 0; i < n; i++)
            {
              index[cursor][0] = result.array[i].value;
              index[cursor][1] = i;
              index[cursor][2] = consts[AXIS_DAYS];
              cursor++;
             }
           }

          delete result;
          return n == N_HOURS;
         }
        return false;
       }

    public:
      virtual void display(MetaCube *cube, const SORT_BY sortby = SORT_BY_NONE, const bool identity = false) override
      {
        int consts[];
        const int n = cube.getDimension();
        ArrayResize(consts, n);
        ArrayInitialize(consts, 0);

        filled = false;

        ArrayResize(index, N_HOURS * N_DAYS);
        ArrayInitialize(index, 1);
        cursor = 0;

        if(n == 2)
        {
          const int i = AXIS_DAYS;
          int m = cube.getDimensionRange(i); // should be 5 work days
          for(int j = 0; j < m; j++)
          {
            consts[i] = j;

            if(!saveVector(cube, consts, sortby)) // 24 hours (values) per current day
            {
              Print("Bad data format");
              return;
             }

            consts[i] = 0;
           }
          filled = true;
          ArraySort(index);
          ArrayPrint(index);
         }
        else
        {
          Print("Incorrect cube structure");
         }
       }

      //...
  };
```

A two-dimensional array 'index' is described in this class. It allows storing performance values in relation to the schedule. In the
'display' method, this array is sequentially populated with vectors from OLAP cube. The auxiliary saveVector function copies numbers for
all 24 hours for a specific trading day. The value, the hour number and the workday number are sequentially written in the second dimension of
'index'. The values are located in the first (0) element, which allows sorting the array by profit factor. Basically, this enables a
convenient view in the log.

The trading mode is selected based on the values of the 'index' array. Accordingly, trading orders are sent for the appropriate time of the day
and day of the week, which have PF above the threshold.

```
    void trade(const double threshold, const double lots, const int strategy = 0)
    {
      const int h = TimeHour(lastBar);
      const int w = _TimeDayOfWeek() - 1;

      int mode = 0;

      for(int i = 0; i < N_HOURS * N_DAYS; i++)
      {
        if(index[i][1] == h && index[i][2] == w)
        {
          if(index[i][0] >= threshold)
          {
            mode = +1;
            Print("+ Rule ", i);
            break;
           }

          if(index[i][0] <= 1.0 / threshold)
          {
            mode = -1;
            Print("- Rule ", i);
            break;
           }
         }
       }

      // pick up existing orders (if any)
      const int direction = CurrentOrderDirection();

      if(mode == 0)
      {
        if(direction != 0)
        {
          OrdersCloseAll();
         }
        return;
       }

      if(strategy == 0)
      {
        if(direction != 0) // there exist open orders
        {
          if(mode == direction) // keep direction
          {
            return; // existing trade goes on
           }
          OrdersCloseAll();
         }

        const int type = mode > 0 ? OP_BUY : OP_SELL;

        const double p = type == OP_BUY ? SymbolInfoDouble(_Symbol, SYMBOL_ASK) : SymbolInfoDouble(_Symbol, SYMBOL_BID);
        const double sl = StopLoss > 0 ? (type == OP_BUY ? p - StopLoss * _Point : p + StopLoss * _Point) : 0;

        OrderSend(_Symbol, type, Lot, p, 100, sl, 0);
       }
      // ...
     }
```

Here I have shown only one trading strategy, the code of which was used in the first testing Expert Advisor. The full source code is attached
below.

Let's optimize OLAPQRWF in the time interval between 2015 and 2019, and then run a forward test for 2019. Please note that the idea of the
optimization is to find meta parameters for trading: the duration of the OLAP analysis, the frequency of OLAP cube rebuilding, selection of
the strategy and of the custom aggregation field. In each optimization run, the EA builds an OLAP cube based on \_historical data\_ and trades
in its virtual \_future\_ using settings from the \_past\_. Why do we need a forward test in this case? Here, trading efficiency directly depends
on the specified meta-parameters, that is why it is important to check the applicability of the selected settings in the out-of-sample
interval.

Let's optimize all parameters which affect the analysis except the Update period (keep it monthly):

- BarNumberLookBack=720\|\|720\|\|480\|\|5760\|\|Y
- Threshold=2.0\|\|2.0\|\|0.5\|\|5.0\|\|Y
- Strategy=0\|\|0\|\|1\|\|1\|\|Y
- Update=0\|\|0\|\|0\|\|1\|\|N
- CustomField=0\|\|0\|\|0\|\|1\|\|Y

The EA calculates a synthetic custom optimization value which is equal to the product of the Sharpe ratio and the number of deals. Based on this
value, the best forecast is generated with the following input parameters:

- BarNumberLookBack=2160
- Threshold=3.0
- Strategy=0
- Update=monthly
- CustomField=count

Let's run a separate test from 2015 to 2020 and mark the behavior in the forward period.

[![Fig.9 OLAPQRWF EA report from 01.01.2015 to 01.01.2020 after the optimization of the OLAP analysis window for 2018 inclusive, EURUSD H1](https://c.mql5.com/2/38/olapqrwf-2015-2019-2020__1.png)](https://c.mql5.com/2/38/olapqrwf-2015-2019-2020.png "https://c.mql5.com/2/38/olapqrwf-2015-2019-2020.png")

**Fig.9 OLAPQRWF EA report from 01.01.2015 to 01.01.2020 after the optimization of**
**the OLAP analysis window for 2018 inclusive, EURUSD H1**

It can be concluded that an Expert Advisor which automatically determines a profitable schedule, successfully trades in 2019 using the
aggregation window size found in previous years. Of course, this system requires further study and analysis. Nevertheless, the tool is
confirmed to be working.

### Conclusion

In this article we have improved and expanded the functionality of the OLAP library (for online data processing) and have implemented its
bundle through a special adapter and working record classes with the quotes area. Using the described programs, it is possible to analyze
the history and to determine patterns the provide profitable trading. At the first stage, when familiarizing yourself with OLAP analysis,
it is more convenient to use individual non-trading Expert Advisors which only process source data and present generalized statistics.
Also, such EAs allow developing and debugging algorithms for calculating custom fields that contain the basic elements of trading
strategies (hypotheses). In further OLAP study steps, the engine is integrated with new or existing trading robots. In this case, EA
optimization should take into account not only common operation parameters, but also new meta parameters which are connected with OLAP and
affect collection of statistics.

Of course, OLAP tools are not a panacea, especially for unpredictable market situations. Thus, they cannot offer an "out of the box" grail.
Nevertheless, the built-in quotes analysis undoubtedly expands possibilities, allowing traders to search for new strategies and create
new Expert Advisors.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/7535](https://www.mql5.com/ru/articles/7535)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/7535.zip "Download all attachments in the single ZIP archive")

[MQLOLAP3.zip](https://www.mql5.com/en/articles/download/7535/mqlolap3.zip "Download MQLOLAP3.zip")(66.21 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Backpropagation Neural Networks using MQL5 Matrices](https://www.mql5.com/en/articles/12187)
- [Parallel Particle Swarm Optimization](https://www.mql5.com/en/articles/8321)
- [Custom symbols: Practical basics](https://www.mql5.com/en/articles/8226)
- [Calculating mathematical expressions (Part 2). Pratt and shunting yard parsers](https://www.mql5.com/en/articles/8028)
- [Calculating mathematical expressions (Part 1). Recursive descent parsers](https://www.mql5.com/en/articles/8027)
- [MQL as a Markup Tool for the Graphical Interface of MQL Programs (Part 3). Form Designer](https://www.mql5.com/en/articles/7795)
- [MQL as a Markup Tool for the Graphical Interface of MQL Programs. Part 2](https://www.mql5.com/en/articles/7739)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/338167)**
(5)


![](https://c.mql5.com/avatar/avatar_na2.png)

**\[Deleted\]**
\|
23 Jan 2020 at 08:20

Developing a seasonal theme via olap, nice. You can also use the inbuilt light skl, I guess.


![Stanislav Korotky](https://c.mql5.com/avatar/2010/10/4CA7CFA0-1F0C.jpg)

**[Stanislav Korotky](https://www.mql5.com/en/users/marketeer)**
\|
23 Jan 2020 at 12:19

**Maxim Dmitrievsky:**

Developing a seasonal theme via olap, nice. You can also use the inbuilt light skl, I guess.

I guess you could, but when I started OLAP in 2016, SQL wasn't in MT yet.

![Stanislav Korotky](https://c.mql5.com/avatar/2010/10/4CA7CFA0-1F0C.jpg)

**[Stanislav Korotky](https://www.mql5.com/en/users/marketeer)**
\|
28 Jan 2020 at 20:26

I attach source codes of this article, adapted quickly for MT4. They should compile normally, but the full functionality has not been fully tested. Some things are missing in MQL4 and cannot be _adequately_ emulated, in particular, the ArrayPrint function with support for [multidimensional arrays](https://www.mql5.com/en/docs/basis/variables "MQL5 Documentation: Variables") and arrays of structures - it is implemented as a simple stub without a nice output with alignment in log lines. Those who wish can improve it. Also here, as well as in the article, the graphical interface was not considered or ported to MT4.


![fxsaber](https://c.mql5.com/avatar/2019/8/5D67260D-44C9.png)

**[fxsaber](https://www.mql5.com/en/users/fxsaber)**
\|
19 Feb 2020 at 09:17

Thanks for the article! Is it correct that OLAP is now fully overlapping in meaning with [SQLite capabilities](https://www.mql5.com/en/articles/7463#analysis_by_entries)?


![Stanislav Korotky](https://c.mql5.com/avatar/2010/10/4CA7CFA0-1F0C.jpg)

**[Stanislav Korotky](https://www.mql5.com/en/users/marketeer)**
\|
19 Feb 2020 at 10:24

**fxsaber:**

Thanks for the article! Is it correct that OLAP is now fully overlapping with [SQLite capabilities](https://www.mql5.com/en/articles/7463#analysis_by_entries)?

Not completely overlap, rather complement each other. OLAP is usually an add-on on top of the database and other data sources. Writing queries in SQL is a chore. The goal of OLAP is to provide a more human interface.

![Library for easy and quick development of MetaTrader programs (part XXXIII): Pending trading requests - closing positions under certain conditions](https://c.mql5.com/2/38/MQL5-avatar-doeasy__1.png)[Library for easy and quick development of MetaTrader programs (part XXXIII): Pending trading requests - closing positions under certain conditions](https://www.mql5.com/en/articles/7554)

We continue the development of the library functionality featuring trading using pending requests. We have already implemented sending conditional trading requests for opening positions and placing pending orders. In the current article, we will implement conditional position closure – full, partial and closing by an opposite position.

![Library for easy and quick development of MetaTrader programs (part XXXII): Pending trading requests - placing orders under certain conditions](https://c.mql5.com/2/38/MQL5-avatar-doeasy.png)[Library for easy and quick development of MetaTrader programs (part XXXII): Pending trading requests - placing orders under certain conditions](https://www.mql5.com/en/articles/7536)

We continue the development of the functionality allowing users to trade using pending requests. In this article, we are going to implement the ability to place pending orders under certain conditions.

![How to create 3D graphics using DirectX in MetaTrader 5](https://c.mql5.com/2/39/MQL5-avatar-directx_yellow.png)[How to create 3D graphics using DirectX in MetaTrader 5](https://www.mql5.com/en/articles/7708)

3D graphics provide excellent means for analyzing huge amounts of data as they enable the visualization of hidden patterns. These tasks can be solved directly in MQL5, while DireсtX functions allow creating three-dimensional object. Thus, it is even possible to create programs of any complexity, even 3D games for MetaTrader 5. Start learning 3D graphics by drawing simple three-dimensional shapes.

![Library for easy and quick development of MetaTrader programs (part XXXI): Pending trading requests - opening positions under certain conditions](https://c.mql5.com/2/37/MQL5-avatar-doeasy__19.png)[Library for easy and quick development of MetaTrader programs (part XXXI): Pending trading requests - opening positions under certain conditions](https://www.mql5.com/en/articles/7521)

Starting with this article, we are going to develop a functionality allowing users to trade using pending requests under certain conditions, for example, when reaching a certain time limit, exceeding a specified profit or closing a position by stop loss.

[![](https://www.mql5.com/ff/sh/jup0jccfs9655z9z2/01.png)Learn to create your own robotsRead our book "MQL5 Programming for Traders"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/book%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.algobook%26utm_content=visit.page%26utm_campaign=algobook.promo.04.2024&a=rsxjstxkzbrlgjjrxaglpezpvrjflnvw&s=7224440013c3dbc50ba9cc078cd015fabca36df446b8e75028d6b30234663872&uid=&ref=https://www.mql5.com/en/articles/7535&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5083178195567515249)

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