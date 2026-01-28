---
title: Continuous Walk-Forward Optimization (Part 1): Working with Optimization Reports
url: https://www.mql5.com/en/articles/7290
categories: Integration
relevance_score: 3
scraped_at: 2026-01-23T21:14:59.558564
---

[![](https://www.mql5.com/ff/sh/zf7a2k61x98jzh89z2/01.png)Speed up your tradingUse our high-speed VPS for MetaTrader 4 and 5Learn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=qtrrsuiwuicrscmckjynyanztbditglq&s=c617dc80d90cfd3783ec1345eec2b419b281f10fec6eac77b3218984ac337259&uid=&ref=https://www.mql5.com/en/articles/7290&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5071689699216534687)

MetaTrader 5 / Tester


### Introduction

In the previous articles ( [Optimization Management (Part I)](https://www.mql5.com/en/articles/7029) and [Optimization \\
Management (Part 2)](https://www.mql5.com/en/articles/7059)) we considered a mechanism for launching the optimization in the terminal through a third-party process. This
allows creating a certain Optimization Manager which can implement the process similarly to a trading algorithm implementing a specific
trading process, i.e. in a fully automated mode without user interference. The idea is to create an algorithm which manages the sliding
optimization process, in which forward and historical periods are shifted by a preset interval and overlap each other.

This approach to algorithm optimization can serve as strategy robustness testing rather than pure optimization, although it performs both
roles. As a result, we can find out whether a trading system is stable and can determine optimal combinations of indicators for the system.
Since the described process can involve different robot coefficient filtration and optimal combination selection methods, which we need
to check in each of the time intervals (which can be multiple) the process can hardly be implemented manually. Moreover, thus we can
encounter errors connected with data transfer or other errors related to the human factor. Therefore, some tools are needed that would
manage the optimization process from the outside without our intervention. The created program meets the set goals. For a more structured
presentation, the program creation process has been split into several articles, each of which covers a specific area of the program
creation process.

This part is devoted to the creation of a toolkit for working with optimization reports, for importing them from the terminal, as well as for
filtering and sorting the obtained data. To provide a better presentation structure, we will use the \*xml file format. Data from the file can
be read by both humans and programs. Moreover, data can be grouped in blocks inside the file and thus the required information can be accessed
faster and easier.

Our program is a third-party process written in C# and it needs to create and read created \*xml documents similarly to MQL5 programs.
Therefore, the report creation block will be implemented as a DLL which can be used both in MQL5 and C# code. Thus, in order to develop an MQL5
code, we will need a library. We will first describe the library creation process, while the next article will provide description of the
MQL5 code that works with the created library and generates optimization parameters. We will consider these parameters in the current
article.

### Report Structure and Required Ratios

As already shown in previous articles, MetaTrader 5 can independently download the report of optimization passes, however it does not
provide as much information as the report generated on the Backtest tab after completion of a test with a specific set of parameters. In order
to have greater scope in working with optimization data, the report should include many of the data displayed on this tab, as well as provide
for the possibility to add more custom data to the report. For these purposes, we will download our own generated reports instead of the
standard one. Let's start with the definition of three data types required for our program:

- Tester settings (the same settings for the whole report)
- Trading Robot settings (unique for each optimization pass)
- Coefficients describing the trading results (unique for each optimization pass)

```

<Optimisation_Report Created="06.10.2019 10:39:02">
        <Optimiser_Settings>
                <Item Name="Bot">StockFut\StockFut.ex5</Item>
                <Item Name="Deposit" Currency="RUR">100000</Item>
                <Item Name="Leverage">1</Item>
        </Optimiser_Settings>
```

Parameters are written to the "Item" block, each having its own "Name"
attribute. The deposit currency will be written to the "Currency"
attribute.

Based on this, the file structure should contain 2 main sections: tester settings and the description of optimization passes. We need to keep
three parameters for the first section:

1. Robot Path Relative to the Experts folder
2. Deposit Currency and Deposit
3. Account Leverage

The second section will contain a sequence of blocks with optimization results, each of which will contain a section with coefficients as
well as a set of robot parameters.

```
<Optimisation_Results>
                <Result Symbol="SBRF Splice" TF="1" Start_DT="1481327340" Finish_DT="1512776940">
                        <Coefficients>
                                <VaR>
                                        <Item Name="90">-1055,18214207419</Item>
                                        <Item Name="95">-1323,65133343373</Item>
                                        <Item Name="99">-1827,30841143882</Item>
                                        <Item Name="Mx">-107,03475</Item>
                                        <Item Name="Std">739,584549199836</Item>
                                </VaR>
                                <Max_PL_DD>
                                        <Item Name="Profit">1045,9305</Item>
                                        <Item Name="DD">-630</Item>
                                        <Item Name="Total Profit Trades">1</Item>
                                        <Item Name="Total Lose Trades">1</Item>
                                        <Item Name="Consecutive Wins">1</Item>
                                        <Item Name="Consecutive Lose">1</Item>
                                </Max_PL_DD>
                                <Trading_Days>
                                        <Mn>
                                                <Item Name="Profit">0</Item>
                                                <Item Name="DD">0</Item>
                                                <Item Name="Number Of Profit Trades">0</Item>
                                                <Item Name="Number Of Lose Trades">0</Item>
                                        </Mn>
                                        <Tu>
                                                <Item Name="Profit">0</Item>
                                                <Item Name="DD">0</Item>
                                                <Item Name="Number Of Profit Trades">0</Item>
                                                <Item Name="Number Of Lose Trades">0</Item>
                                        </Tu>
                                        <We>
                                                <Item Name="Profit">1045,9305</Item>
                                                <Item Name="DD">630</Item>
                                                <Item Name="Number Of Profit Trades">1</Item>
                                                <Item Name="Number Of Lose Trades">1</Item>
                                        </We>
                                        <Th>
                                                <Item Name="Profit">0</Item>
                                                <Item Name="DD">0</Item>
                                                <Item Name="Number Of Profit Trades">0</Item>
                                                <Item Name="Number Of Lose Trades">0</Item>
                                        </Th>
                                        <Fr>
                                                <Item Name="Profit">0</Item>
                                                <Item Name="DD">0</Item>
                                                <Item Name="Number Of Profit Trades">0</Item>
                                                <Item Name="Number Of Lose Trades">0</Item>
                                        </Fr>
                                </Trading_Days>
                                <Item Name="Payoff">1,66020714285714</Item>
                                <Item Name="Profit factor">1,66020714285714</Item>
                                <Item Name="Average Profit factor">0,830103571428571</Item>
                                <Item Name="Recovery factor">0,660207142857143</Item>
                                <Item Name="Average Recovery factor">-0,169896428571429</Item>
                                <Item Name="Total trades">2</Item>
                                <Item Name="PL">415,9305</Item>
                                <Item Name="DD">-630</Item>
                                <Item Name="Altman Z Score">0</Item>
                        </Coefficients>
                        <Item Name="_lot_">1</Item>
                        <Item Name="USymbol">SBER</Item>
                        <Item Name="Spread_in_percent">3.00000000</Item>
                        <Item Name="UseAutoLevle">false</Item>
                        <Item Name="max_per">174</Item>
                        <Item Name="comission_stock">0.05000000</Item>
                        <Item Name="shift_stock">0.00000000</Item>
                        <Item Name="comission_fut">4.00000000</Item>
                        <Item Name="shift_fut">0.00000000</Item>
                </Result>
        </Optimisation_Results>
</Optimisation_Report>
```

Inside the Optimisation\_Results block, we will have repeated Result
blocks, each of which contains the i-th optimization pass. Each of the Result
blocks contains 4 attributes:

- Symbol
- TF
- Start\_DT
- Finish\_DT

These are the tester settings which will vary depending on the time interval in which the optimization is performed. Each of the robot
parameters is written to the

Item block with the Name attribute as a unique value, based on which
it can be identified. Robot coefficients are written to the

Coefficients block. Coefficients which cannot be grouped
are enumerated directly in the

Item block. Other coefficients are divided into blocks:

- VaR

> 1. 90 - quantile 90
> 2. 95 - quantile 95
> 3. 99 - quantile 99
> 4. Mx - math expectation
> 5. Std - standard deviation

- Max\_PL\_DD

> 1. Profit - total profit
> 2. DD - total drawdown
> 3. Total Profit Trades - total number of profitable trades
> 4. Total Lose Trades - total number of losing trades
> 5. Consecutive Wins - winning trades in a row
> 6. Consecutive Lose - losing trades in a row

- Trading\_Days - trading reports by days

> 1. Profit - average profit per day
> 2. DD - average losses per day
> 3. Number Of Profit Trades - number of profitable trades
> 4. Number Of Lose Trades - number of losing trades

As a result, we receive a list with the coefficients of optimization results, which fully describe test results. Now, to filter and select
robot parameters, there is a complete list of required coefficients which enable us to efficiently evaluate the robot performance.

### The wrapper class of the optimizations report, the class storing optimization dates, as well as the structure of  optimizations results in C\#.

Let's start with the structure storing data for a specific optimization pass.

```
public struct ReportItem
{
    public Dictionary<string, string> BotParams; // List of robot parameters
    public Coefficients OptimisationCoefficients; // Robot coefficients
    public string Symbol; // Symbol
    public int TF; // Timeframe
    public DateBorders DateBorders; // Date range
}
```

All robot coefficients are stored in the dictionary in a string format. The file with robot parameters does not save the type of data,
therefore the string format suites best here. The list of robot coefficients is provided in a different structure, similarly to other
blocks which are grouped in the \*xml optimizations report. Trading reports by days are also stored in the dictionary.

```
public Dictionary<DayOfWeek, DailyData> TradingDays;
```

The DayOfWeek and the dictionary must always contain the enumeration of 5 days (from Monday to Friday) as a key, similarly to the \*xml file. The
most interesting class in the data storing structure is DateBorders. Similar to the data being grouped within a structure containing
fields which describe each of the date parameters, date ranges are also stored in the DateBorders structure.

```
public class DateBorders : IComparable
{
    /// <summary>
    /// Constructor
    /// </summary>
    /// <param name="from">Range beginning date</param>
    /// <param name="till">Range ending date</param>
    public DateBorders(DateTime from, DateTime till)
    {
        if (till <= from)
            throw new ArgumentException("Date 'Till' is less or equal to date 'From'");

        From = from;
        Till = till;
    }
    /// <summary>
    /// From
    /// </summary>
    public DateTime From { get; }
    /// <summary>
    /// To
    /// </summary>
    public DateTime Till { get; }
}
```

For a fully-featured operation with the date range, we need the possibility to create two date ranges. For this purpose, overwrite 2
operators "==" and "!=".

Equality criteria are determined by the equality of both dates in the two passed ranges, i.e. the beginning date matches the trading beginning of
the second range (while the same also applies to the trading end). However, since the object type is 'class', it can be equal to null, and thus
we first need to provide for the

ability to compare to null. Let's use the [is](https://www.mql5.com/go?link=https://docs.microsoft.com/en-us/dotnet/csharp/language-reference/keywords/is "https://docs.microsoft.com/en-us/dotnet/csharp/language-reference/keywords/is")
keyword for that purpose. After that we can compare parameters with each other,
otherwise, if we try to compare to null, "null reference exception" will be returned.

```
#region Equal
/// <summary>
/// The equality comparison operator
/// </summary>
/// <param name="b1">Element 1</param>
/// <param name="b2">Element 2</param>
/// <returns>Result</returns>
public static bool operator ==(DateBorders b1, DateBorders b2)
{
    bool ans;
    if (b2 is null && b1 is null) ans = true;
    else if (b2 is null || b1 is null) ans = false;
    else ans = b1.From == b2.From && b1.Till == b2.Till;

    return ans;
}
/// <summary>
/// The inequality comparison operator
/// </summary>
/// <param name="b1">Element 1</param>
/// <param name="b2">Element 2</param>
/// <returns>Comparison result</returns>
public static bool operator !=(DateBorders b1, DateBorders b2) => !(b1 == b2);
#endregion
```

To overload the inequality operator, we no longer need to write
the above described procedures, while all of them are already written in operator "==". The next feature we need to implement is data sorting
by time periods, that is why we need to overload operators ">", "<", ">=", "<=".

```
#region (Grater / Less) than
/// <summary>
/// Comparing: current element is greater than the previous one
/// </summary>
/// <param name="b1">Element 1</param>
/// <param name="b2">Element 2</param>
/// <returns>Result</returns>
public static bool operator >(DateBorders b1, DateBorders b2)
{
    if (b1 == null || b2 == null)
        return false;

    if (b1.From == b2.From)
        return (b1.Till > b2.Till);
    else
        return (b1.From > b2.From);
}
/// <summary>
/// Comparing: current element is less than the previous one
/// </summary>
/// <param name="b1">Element 1</param>
/// <param name="b2">Element 2</param>
/// <returns>Result</returns>
public static bool operator <(DateBorders b1, DateBorders b2)
{
    if (b1 == null || b2 == null)
        return false;

    if (b1.From == b2.From)
        return (b1.Till < b2.Till);
    else
        return (b1.From < b2.From);
}
#endregion
```

If any of the parameters passed to the operator is equal to null,
the comparison becomes impossible, therefore return False. Otherwise compare step by step. If the first time interval matches, compare by
the second time interval. If they are not equal, compare by the first interval. Thus, if we describe the comparison logic based om the
"Greater" operator example, the greater interval is the one which is older in time than the previous one, either by the start date or by the end
date (if start dates are equal). The "less" comparison logic is similar to the "greater" comparison.

The next operators to be overloaded to enable the sorting option are 'Greater Than Or Equal' and 'Less Than Or Equal'.

```
#region Equal or (Grater / Less) than
/// <summary>
/// Greater than or equal comparison
/// </summary>
/// <param name="b1">Element 1</param>
/// <param name="b2">Element 2</param>
/// <returns>Result</returns>
public static bool operator >=(DateBorders b1, DateBorders b2) => (b1 == b2 || b1 > b2);
/// <summary>
/// Less than or equal comparison
/// </summary>
/// <param name="b1">Element 1</param>
/// <param name="b2">Element 2</param>
/// <returns>Result</returns>
public static bool operator <=(DateBorders b1, DateBorders b2) => (b1 == b2 || b1 < b2);
#endregion
```

As can be seen, the operator overload does not require the description of the internal comparison logic. Instead, we use the already
overloaded operators == and >, <. However, as Visual Studio suggests during compilation, in addition to the overloading of these
operators we need to overload some functions inherited from the "object" base class.

```
#region override base methods (from object)
/// <summary>
/// Overloading of equality comparison
/// </summary>
/// <param name="obj">Element to compare to</param>
/// <returns></returns>
public override bool Equals(object obj)
{
    if (obj is DateBorders other)
        return this == other;
    else
        return base.Equals(obj);
}
/// <summary>
/// Cast the class to a string and return its hash code
/// </summary>
/// <returns>String hash code</returns>
public override int GetHashCode()
{
    return ToString().GetHashCode();
}
/// <summary>
/// Convert the current class to a string
/// </summary>
/// <returns>String From date - To date</returns>
public override string ToString()
{
    return $"{From}-{Till}";
}
#endregion
/// <summary>
/// Compare the current element with the passed one
/// </summary>
/// <param name="obj"></param>
/// <returns></returns>
public int CompareTo(object obj)
{
    if (obj == null) return 1;

    if (obj is DateBorders borders)
    {
        if (this == borders)
            return 0;
        else if (this < borders)
            return -1;
        else
            return 1;
    }
    else
    {
        throw new ArgumentException("object is not DateBorders");
    }
}
```

Equals method:
overload it either using the overloaded operator == (if the passed object has type DateBorders) or the basic implementation of the method.

ToString method: overload it as a string representation of two dates
separated by a hyphen. This will help us to overload the GetHashCode method.

GetHashCode method: overload it by first casting the object to a
string and then returning the hash code of this string. When a new class instance is created in C#, its hash code is unique regardless of the
class content. That is, if we do not overload the method and create two instances of the DateBorders class with the same From and To dates
inside, they will have different hash codes despite identical contents. This rule does not apply to strings, because C# provides a
mechanism which prevents from the creation of new instances of the String class if the string was previously created — thus their hash codes
for identical strings will match. Using the ToString method overloading and using the string hash code, we provide the behavior of our class
hash codes similar to those of String. Now, when using the IEnumerable.Distinct method, we can guarantee that the logic of receiving the
unique list of date ranges will be correct, as this method is based on the hash codes of the compared objects.

Implementing the IComparable interface, from which our class is
inherited, we implement the

CompareTo method which compares the current class instance with the
passed one. Its implementation is easy and it uses overloads of previously overloaded operators.

Having implemented the required overloads, we can work with this class more efficiently. We can:

- Compare two instances for equality
- Compare two instances for greater than/less than
- Compare two instances for greater than or equal/less than or equal
- Sort ascending/descending
- Get unique values from a list of date ranges
- Use the IEnumerable.Sort method which sorts the list in the descending order and uses the IComparable interface.

Since we are implementing a rolling optimization, which will have backtests and forward tests, we need to create a method to compare
historic and forward intervals.

```
/// <summary>
/// Method for comparing forward and historical optimizations
/// </summary>
/// <param name="History">Array of historical optimization</param>
/// <param name="Forward">Array of forward optimizations</param>
/// <returns>Sorted list historical - forward optimization</returns>
public static Dictionary<DateBorders, DateBorders> CompareHistoryToForward(List<DateBorders> History, List<DateBorders> Forward)
{
    // array of comparable optimizations
    Dictionary<DateBorders, DateBorders> ans = new Dictionary<DateBorders, DateBorders>();

    // Sort the passed parameters
    History.Sort();
    Forward.Sort();

    // Create a historical optimization loop
    int i = 0;
    foreach (var item in History)
    {
if(ans.ContainsKey(item))
       	    continue;

        ans.Add(item, null); // Add historical optimization
        if (Forward.Count <= i)
            continue; // If the array of forward optimization is less than the index, continue the loop

        // Forward optimization loop
        for (int j = i; j < Forward.Count; j++)
        {
            // If the current forward optimization is contained in the results array, skip
            if (ans.ContainsValue(Forward[j]) ||
                Forward[j].From < item.Till)
            {
                continue;
            }

            // Compare forward and historical optimization
            ans[item] = Forward[j];
            i = j + 1;
            break;
        }
    }

    return ans;
}
```

As you can see, the method is static. This is done to make it available as a regular function, without binding to a specific class
instance. First of all it

sorts the passed time intervals in the ascending order. Thus, in the
next loop we can know for sure that all the previously passed intervals are less than or equal to the next ones. Then implement two loops:

foreach for historical intervals, nested
loop for forward intervals.

At the beginning of the historical data loop, we always add
historical ranges (key)

to the collection with results, as well as temporarily set null in
place of forward intervals. Forward results loop starts with the

i-th parameter. This prevents from repeating the loop with already
used elements of the forward list. That is the forward interval should always follow the historical one, i.e. it should be > than the
historical. That is why we implement the loop by forward intervals, in case in the passed list there is a forward period for the very first
historical interval, which precedes the very first historical interval. It is better to visualize the idea in a table:

| Historical | Forward |
| --- | --- |
| From | To | From | To |
| 10.03.2016 | 09.03.2017 | 12.12.2016 | 09.03.2017 |
| 10.06.2016 | 09.06.2017 | 10.03.2017 | 09.06.2017 |
| 10.09.2016 | 09.09.2017 | 10.06.2017 | 09.09.2017 |

So the first historical interval ends on 09.03.2017, and the first forward interval starts on 12.12.2016, which is not correct. That is
why we skip it in the

forward intervals loop, due to the condition.
Also, skip the forward interval, which is contained in the resulting dictionary.
If the j-th forward data does not yet exists in the resulting dictionary and the forward interval beginning date is >= current
historical interval end date, save the received value and exit the forward intervals loop as the required value has already been found.
Before exiting, assign the value of the forward interval following the selected one to the i variable (the variable which means the
forward list iterations beginning. This is done because the current interval will no longer be needed (due to the initial data
sorting).

A check before the historical optimization ensures that
all historical optimizations are unique. Thus, the following list is obtained in the resulting dictionary:

| Key | Value |
| --- | --- |
| 10.03.2016-09.3.2017 | 10.03.2017-09.06.2017 |
| 10.06.2016-09.06.2017 | 10.06.2017-09.09.2017 |
| 10.09.2016-09.09.2017 | null |

As can be seen from the presented data, the first forward interval is discarded and no interval is found for the last historical one, as no
such interval has been passed. Based on this logic, the program will compare data of the historic and forward intervals and will
understand which of the historical intervals should provide optimization parameters for forward tests.

To enable efficient operation with a specific optimization result, I have created a wrapper structure for the ReportItem
structure which contains a number of additional methods and overloaded operators. Basically, the wrapper contains two fields:

```
/// <summary>
/// Optimization pass report
/// </summary>
public ReportItem report;
/// <summary>
/// Sorting factor
/// </summary>
public double SortBy;
```

The first field was described above. The second field is created to enable sorting by multiple values, for example profit and recovery
factor. The sorting mechanism will be described later, but the idea is to convert these values to one and to store it in this
variable.

The structure also contains type conversion overloads:

```
/// <summary>
/// The operator of implicit type conversion from optimization pass to the current type
/// </summary>
/// <param name="item">Optimization pass report</param>
public static implicit operator OptimisationResult(ReportItem item)
{
    return new OptimisationResult { report = item, SortBy = 0 };
}
/// <summary>
/// The operator of explicit type conversion from current to the optimization pass structure
/// </summary>
/// <param name="optimisationResult">current type</param>
public static explicit operator ReportItem(OptimisationResult optimisationResult)
{
    return optimisationResult.report;
}
```

As a result, we can implicitly cast ReportItem type to its wrapper, and then explicitly cast the ReportItem wrapper to the trading
report element. This can be more efficient than sequential filling of fields. Since all fields in the ReportItem structure are divided
into categories, we may sometimes need a lengthy code in order to receive a desired value. A special method has been created to save space
and to create a more universal getter. It receives the requested robot ratios data via the passed enum SourtBy from the above

GetResult(SortBy resultType) code. The
implementation is simple but too long and it is therefore not provided here. The method iterates over the passed enum in the switch
constuct and returns the value of the requested coefficient. Since most of the coefficients have type double and since this type can
contain all other numeric types, coefficient values are converted to double.

Comparison operator overloads have also been implemented for this wrapper type:

```
/// <summary>
/// Overloading of the equality comparison operator
/// </summary>
/// <param name="result1">Parameter 1 to compare</param>
/// <param name="result2">Parameter 2 to compare</param>
/// <returns>Comparison result</returns>
public static bool operator ==(OptimisationResult result1, OptimisationResult result2)
{
    foreach (var item in result1.report.BotParams)
    {
        if (!result2.report.BotParams.ContainsKey(item.Key))
            return false;
        if (result2.report.BotParams[item.Key] != item.Value)
            return false;
    }

    return true;
}
/// <summary>
 /// Overloading of the inequality comparison operator
/// </summary>
/// <param name="result1">Parameter 1 to compare</param>
/// <param name="result2">Parameter 2 to compare</param>
/// <returns>Comparison result</returns>
public static bool operator !=(OptimisationResult result1, OptimisationResult result2)
{
    return !(result1 == result2);
}
/// <summary>
/// Overloading of the basic type comparison operator
/// </summary>
/// <param name="obj"></param>
/// <returns></returns>
public override bool Equals(object obj)
{
    if (obj is OptimisationResult other)
    {
        return this == other;
    }
    else
        return base.Equals(obj);
}
```

The elements of optimizations containing the same names and values of robot parameters will be considered equal. Thus, if we need to
compare two optimization passes, we already have the ready-to-use overloaded operators. This structure also contains a method which
writes data to a file. If it exists, data is simply added to the file. Explanation of the data writing element and method implementation
will be provided below.

### Creating a File to Store the Optimization Report

We will work with the optimization reports and will write them not only in the terminal, but also in the created program. That is why let us add
the optimization report creating method to this DLL. Let us also provide several methods for data writing to a file, i.e. enable writing of a
data array to a file as well as enable the addition of a separate element to existing file (if the file does not exist, it should be created). The
last method will be imported to the terminal and will be used in C# classes. Let's start considering the implemented report file writing
methods with the functions connected with the addition of data to a file. The ReportWriter class was created for this purpose. The full class
implementation is available in the attached project file. Here I will only show the most interesting methods. Let's first describe how this
class works.

It contains only static methods: this enables exporting of its methods to MQL5. For the same purpose, the class is marked with a public access
modifier. This class contains a static field of ReportItem type and a number of methods which alternately add coefficients and EA
parameters to it.

```
/// <summary>
/// temporary data keeper
/// </summary>
private static ReportItem ReportItem;
/// <summary>
/// clearing the temporary data keeper
/// </summary>
public static void ClearReportItem()
{
    ReportItem = new ReportItem();
}
```

Another method is ClearReportItem(). It recreates the field instance. In this case we lose access to the previous instance of this object: it is
erased and data saving process starts again. Data adding methods are grouped by blocks. Here are the signatures of these
methods.

```
/// <summary>
/// Add robot parameters
/// </summary>
/// <param name="name">Parameter name</param>
/// <param name="value">Parameter value</param>
public static void AppendBotParam(string name, string value);

/// <summary>
/// Add the main list of coefficients
/// </summary>
/// <param name="payoff"></param>
/// <param name="profitFactor"></param>
/// <param name="averageProfitFactor"></param>
/// <param name="recoveryFactor"></param>
/// <param name="averageRecoveryFactor"></param>
/// <param name="totalTrades"></param>
/// <param name="pl"></param>
/// <param name="dd"></param>
/// <param name="altmanZScore"></param>
public static void AppendMainCoef(double payoff,
                                  double profitFactor,
                                  double averageProfitFactor,
                                  double recoveryFactor,
                                  double averageRecoveryFactor,
                                  int totalTrades,
                                  double pl,
                                  double dd,
                                  double altmanZScore);

/// <summary>
/// Add VaR
/// </summary>
/// <param name="Q_90"></param>
/// <param name="Q_95"></param>
/// <param name="Q_99"></param>
/// <param name="Mx"></param>
/// <param name="Std"></param>
public static void AppendVaR(double Q_90, double Q_95,
                             double Q_99, double Mx, double Std);

/// <summary>
/// Add total PL / DD and associated values
/// </summary>
/// <param name="profit"></param>
/// <param name="dd"></param>
/// <param name="totalProfitTrades"></param>
/// <param name="totalLoseTrades"></param>
/// <param name="consecutiveWins"></param>
/// <param name="consecutiveLose"></param>
public static void AppendMaxPLDD(double profit, double dd,
                                 int totalProfitTrades, int totalLoseTrades,
                                 int consecutiveWins, int consecutiveLose);

/// <summary>
/// Add a specific day
/// </summary>
/// <param name="day"></param>
/// <param name="profit"></param>
/// <param name="dd"></param>
/// <param name="numberOfProfitTrades"></param>
/// <param name="numberOfLoseTrades"></param>
public static void AppendDay(int day,
                             double profit, double dd,
                             int numberOfProfitTrades,
                             int numberOfLoseTrades);
```

The method adding trading statistics broken down by days should be
called for each of the 5 trading days. If we do not add it for one of the days, the written file will not be read in the future. Once data is added to
the data storage field, we can proceed to recording the field. Before this, check if the file exists and create it if necessary. A few methods
have been added for creating the file.

```
/// <summary>
/// The method creates the file if it has not been created
/// </summary>
/// <param name="pathToBot">Path to the robot</param>
/// <param name="currency">Deposit currency</param>
/// <param name="balance">Balance</param>
/// <param name="leverage">Leverage</param>
/// <param name="pathToFile">Path to file</param>
private static void CreateFileIfNotExists(string pathToBot, string currency, double balance, int leverage, string pathToFile)
{
    if (File.Exists(pathToFile))
        return;
    using (var xmlWriter = new XmlTextWriter(pathToFile, null))
    {
        // set document format
        xmlWriter.Formatting = Formatting.Indented;
        xmlWriter.IndentChar = '\t';
        xmlWriter.Indentation = 1;

        xmlWriter.WriteStartDocument();

        // Create document root
        #region Document root
        xmlWriter.WriteStartElement("Optimisation_Report");

        // Write the creation date
        xmlWriter.WriteStartAttribute("Created");
        xmlWriter.WriteString(DateTime.Now.ToString("dd.MM.yyyy HH:mm:ss"));
        xmlWriter.WriteEndAttribute();

        #region Optimiser settings section
        // Optimizer settings
        xmlWriter.WriteStartElement("Optimiser_Settings");

        // Path to the robot
        WriteItem(xmlWriter, "Bot", pathToBot);
        // Deposit
        WriteItem(xmlWriter, "Deposit", balance.ToString(), new Dictionary<string, string> { { "Currency", currency } });
        // Leverage
        WriteItem(xmlWriter, "Leverage", leverage.ToString());

        xmlWriter.WriteEndElement();
        #endregion

        #region Optimization results section
        // the root node of the optimization results list
        xmlWriter.WriteStartElement("Optimisation_Results");
        xmlWriter.WriteEndElement();
        #endregion

        xmlWriter.WriteEndElement();
        #endregion

        xmlWriter.WriteEndDocument();
        xmlWriter.Close();
    }
}

/// <summary>
/// Write element to a file
/// </summary>
/// <param name="writer">Writer</param>
/// <param name="Name">Element name</param>
/// <param name="Value">Element value</param>
/// <param name="Attributes">Attributes</param>
private static void WriteItem(XmlTextWriter writer, string Name, string Value, Dictionary<string, string> Attributes = null)
{
    writer.WriteStartElement("Item");

    writer.WriteStartAttribute("Name");
    writer.WriteString(Name);
    writer.WriteEndAttribute();

    if (Attributes != null)
    {
        foreach (var item in Attributes)
        {
            writer.WriteStartAttribute(item.Key);
            writer.WriteString(item.Value);
            writer.WriteEndAttribute();
        }
    }

    writer.WriteString(Value);

    writer.WriteEndElement();
}
```

I also provide here the implementation of the WriteItem
method which contains the repeating code for adding a final element with data and element-specific attributes to a file. The file creating
method CreateFileIfNotExists

checks whether the file exists, creates the file and starts forming the minimum
required file structure.

Firstly, it creates the file root, i.e. the
<Optimization\_Report/> tag, inside which all the child structures of the file are located. Then

file creation data is filled — this is implemented for further
convenient work with files. After that we create a

node with unchanged optimizer settings and specify them. Then
create a

section which will store optimization results and immediately
close it. As a result we have an empty file with the minimum required formatting.

```

<Optimisation_Report Created="24.10.2019 19:10:08">
        <Optimiser_Settings>
                <Item Name="Bot">Path to bot</Item>
                <Item Name="Deposit" Currency="Currency">1000</Item>
                <Item Name="Leverage">1</Item>
        </Optimiser_Settings>
        <Optimisation_Results />
</Optimisation_Report>
```

Thus we will be able to read this file using the XmlDocument class. This is the most useful class for reading and editing existing Xml documents.
We will use exactly this class to add data to existing documents. Repeated operations are implemented as separate methods and thus we will be
able to add data to an exiting document more efficiently:

```
/// <summary>
/// Writing attributes to a file
/// </summary>
/// <param name="item">Node</param>
/// <param name="xmlDoc">Document</param>
/// <param name="Attributes">Attributes</param>
private static void FillInAttributes(XmlNode item, XmlDocument xmlDoc, Dictionary<string, string> Attributes)
{
    if (Attributes != null)
    {
        foreach (var attr in Attributes)
        {
            XmlAttribute attribute = xmlDoc.CreateAttribute(attr.Key);
            attribute.Value = attr.Value;
            item.Attributes.Append(attribute);
        }
    }
}

/// <summary>
/// Add section
/// </summary>
/// <param name="xmlDoc">Document</param>
/// <param name="xpath_parentSection">xpath to select parent node</param>
/// <param name="sectionName">Section name</param>
/// <param name="Attributes">Attribute</param>
private static void AppendSection(XmlDocument xmlDoc, string xpath_parentSection,
                                  string sectionName, Dictionary<string, string> Attributes = null)
{
    XmlNode section = xmlDoc.SelectSingleNode(xpath_parentSection);
    XmlNode item = xmlDoc.CreateElement(sectionName);

    FillInAttributes(item, xmlDoc, Attributes);

    section.AppendChild(item);
}

/// <summary>
/// Write item
/// </summary>
/// <param name="xmlDoc">Document</param>
/// <param name="xpath_parentSection">xpath to select parent node</param>
/// <param name="name">Item name</param>
/// <param name="value">Value</param>
/// <param name="Attributes">Attributes</param>
private static void WriteItem(XmlDocument xmlDoc, string xpath_parentSection, string name,
                              string value, Dictionary<string, string> Attributes = null)
{
    XmlNode section = xmlDoc.SelectSingleNode(xpath_parentSection);
    XmlNode item = xmlDoc.CreateElement(name);
    item.InnerText = value;

    FillInAttributes(item, xmlDoc, Attributes);

    section.AppendChild(item);
}
```

The first method FillInAttributes fills attributes for the passed node, WriteItem writes an item to the section specified via XPath, while
AppendSection adds a section inside another section, which is specified via a path passed using Xpath. These code blocks are often used when
adding data to a file. The data writing method is quite lengthy and is divided into blocks.

```
/// <summary>
/// Write trading results to a file
/// </summary>
/// <param name="pathToBot">Path to the bot</param>
/// <param name="currency">Deposit currency</param>
/// <param name="balance">Balance</param>
/// <param name="leverage">Leverage</param>
/// <param name="pathToFile">Path to file</param>
/// <param name="symbol">Symbol</param>
/// <param name="tf">Timeframe</param>
/// <param name="StartDT">Trading start dare</param>
/// <param name="FinishDT">Trading end date</param>
public static void Write(string pathToBot, string currency, double balance,
                         int leverage, string pathToFile, string symbol, int tf,
                         ulong StartDT, ulong FinishDT)
{
    // Create the file if it does not yet exist
    CreateFileIfNotExists(pathToBot, currency, balance, leverage, pathToFile);

    ReportItem.Symbol = symbol;
    ReportItem.TF = tf;

    // Create a document and read the file using it
    XmlDocument xmlDoc = new XmlDocument();
    xmlDoc.Load(pathToFile);

    #region Append result section
    // Write a request to switch to the optimization results section
    string xpath = "Optimisation_Report/Optimisation_Results";
    // Add a new section with optimization results
    AppendSection(xmlDoc, xpath, "Result",
                  new Dictionary<string, string>
                  {
                      { "Symbol", symbol },
                      { "TF", tf.ToString() },
                      { "Start_DT", StartDT.ToString() },
                      { "Finish_DT", FinishDT.ToString() }
                  });
    // Add section with optimization results
    AppendSection(xmlDoc, $"{xpath}/Result[last()]", "Coefficients");
    // Add section with VaR
    AppendSection(xmlDoc, $"{xpath}/Result[last()]/Coefficients", "VaR");
    // Add section with total PL / DD
    AppendSection(xmlDoc, $"{xpath}/Result[last()]/Coefficients", "Max_PL_DD");
    // Add section with trading results by days
    AppendSection(xmlDoc, $"{xpath}/Result[last()]/Coefficients", "Trading_Days");
    // Add section with trading results on Monday
    AppendSection(xmlDoc, $"{xpath}/Result[last()]/Coefficients/Trading_Days", "Mn");
    // Add section with trading results on Tuesday
    AppendSection(xmlDoc, $"{xpath}/Result[last()]/Coefficients/Trading_Days", "Tu");
    // Add section with trading results on Wednesday
    AppendSection(xmlDoc, $"{xpath}/Result[last()]/Coefficients/Trading_Days", "We");
    // Add section with trading results on Thursday
    AppendSection(xmlDoc, $"{xpath}/Result[last()]/Coefficients/Trading_Days", "Th");
    // Add section with trading results on Friday
    AppendSection(xmlDoc, $"{xpath}/Result[last()]/Coefficients/Trading_Days", "Fr");
    #endregion

    #region Append Bot params
    // Iterate through bot parameters
    foreach (var item in ReportItem.BotParams)
    {
        // Write the selected robot parameter
        WriteItem(xmlDoc, "Optimisation_Report/Optimisation_Results/Result[last()]",
                  "Item", item.Value, new Dictionary<string, string> { { "Name", item.Key } });
    }
    #endregion

    #region Append main coef
    // Set path to node with coefficients
    xpath = "Optimisation_Report/Optimisation_Results/Result[last()]/Coefficients";

    // Save coefficients
    WriteItem(xmlDoc, xpath, "Item", ReportItem.OptimisationCoefficients.Payoff.ToString(), new Dictionary<string, string> { { "Name", "Payoff" } });
    WriteItem(xmlDoc, xpath, "Item", ReportItem.OptimisationCoefficients.ProfitFactor.ToString(), new Dictionary<string, string> { { "Name", "Profit factor" } });
    WriteItem(xmlDoc, xpath, "Item", ReportItem.OptimisationCoefficients.AverageProfitFactor.ToString(), new Dictionary<string, string> { { "Name", "Average Profit factor" } });
    WriteItem(xmlDoc, xpath, "Item", ReportItem.OptimisationCoefficients.RecoveryFactor.ToString(), new Dictionary<string, string> { { "Name", "Recovery factor" } });
    WriteItem(xmlDoc, xpath, "Item", ReportItem.OptimisationCoefficients.AverageRecoveryFactor.ToString(), new Dictionary<string, string> { { "Name", "Average Recovery factor" } });
    WriteItem(xmlDoc, xpath, "Item", ReportItem.OptimisationCoefficients.TotalTrades.ToString(), new Dictionary<string, string> { { "Name", "Total trades" } });
    WriteItem(xmlDoc, xpath, "Item", ReportItem.OptimisationCoefficients.PL.ToString(), new Dictionary<string, string> { { "Name", "PL" } });
    WriteItem(xmlDoc, xpath, "Item", ReportItem.OptimisationCoefficients.DD.ToString(), new Dictionary<string, string> { { "Name", "DD" } });
    WriteItem(xmlDoc, xpath, "Item", ReportItem.OptimisationCoefficients.AltmanZScore.ToString(), new Dictionary<string, string> { { "Name", "Altman Z Score" } });
    #endregion

    #region Append VaR
    // Set path to node with VaR
    xpath = "Optimisation_Report/Optimisation_Results/Result[last()]/Coefficients/VaR";

    // Save VaR results
    WriteItem(xmlDoc, xpath, "Item", ReportItem.OptimisationCoefficients.VaR.Q_90.ToString(), new Dictionary<string, string> { { "Name", "90" } });
    WriteItem(xmlDoc, xpath, "Item", ReportItem.OptimisationCoefficients.VaR.Q_95.ToString(), new Dictionary<string, string> { { "Name", "95" } });
    WriteItem(xmlDoc, xpath, "Item", ReportItem.OptimisationCoefficients.VaR.Q_99.ToString(), new Dictionary<string, string> { { "Name", "99" } });
    WriteItem(xmlDoc, xpath, "Item", ReportItem.OptimisationCoefficients.VaR.Mx.ToString(), new Dictionary<string, string> { { "Name", "Mx" } });
    WriteItem(xmlDoc, xpath, "Item", ReportItem.OptimisationCoefficients.VaR.Std.ToString(), new Dictionary<string, string> { { "Name", "Std" } });
    #endregion

    #region Append max PL and DD
    // Set path to node with total PL / DD
    xpath = "Optimisation_Report/Optimisation_Results/Result[last()]/Coefficients/Max_PL_DD";

    // Save coefficients
    WriteItem(xmlDoc, xpath, "Item", ReportItem.OptimisationCoefficients.MaxPLDD.Profit.Value.ToString(), new Dictionary<string, string> { { "Name", "Profit" } });
    WriteItem(xmlDoc, xpath, "Item", ReportItem.OptimisationCoefficients.MaxPLDD.DD.Value.ToString(), new Dictionary<string, string> { { "Name", "DD" } });
    WriteItem(xmlDoc, xpath, "Item", ReportItem.OptimisationCoefficients.MaxPLDD.Profit.TotalTrades.ToString(), new Dictionary<string, string> { { "Name", "Total Profit Trades" } });
    WriteItem(xmlDoc, xpath, "Item", ReportItem.OptimisationCoefficients.MaxPLDD.DD.TotalTrades.ToString(), new Dictionary<string, string> { { "Name", "Total Lose Trades" } });
    WriteItem(xmlDoc, xpath, "Item", ReportItem.OptimisationCoefficients.MaxPLDD.Profit.ConsecutivesTrades.ToString(), new Dictionary<string, string> { { "Name", "Consecutive Wins" } });
    WriteItem(xmlDoc, xpath, "Item", ReportItem.OptimisationCoefficients.MaxPLDD.DD.ConsecutivesTrades.ToString(), new Dictionary<string, string> { { "Name", "Consecutive Lose" } });
    #endregion

    #region Append Days
    foreach (var item in ReportItem.OptimisationCoefficients.TradingDays)
    {
        // Set path to specific day node
        xpath = "Optimisation_Report/Optimisation_Results/Result[last()]/Coefficients/Trading_Days";
        // Select day
        switch (item.Key)
        {
            case DayOfWeek.Monday: xpath += "/Mn"; break;
            case DayOfWeek.Tuesday: xpath += "/Tu"; break;

            case DayOfWeek.Wednesday: xpath += "/We"; break;
            case DayOfWeek.Thursday: xpath += "/Th"; break;
            case DayOfWeek.Friday: xpath += "/Fr"; break;
        }

        // Save results
        WriteItem(xmlDoc, xpath, "Item", item.Value.Profit.Value.ToString(), new Dictionary<string, string> { { "Name", "Profit" } });
        WriteItem(xmlDoc, xpath, "Item", item.Value.DD.Value.ToString(), new Dictionary<string, string> { { "Name", "DD" } });
        WriteItem(xmlDoc, xpath, "Item", item.Value.Profit.Trades.ToString(), new Dictionary<string, string> { { "Name", "Number Of Profit Trades" } });
        WriteItem(xmlDoc, xpath, "Item", item.Value.DD.Trades.ToString(), new Dictionary<string, string> { { "Name", "Number Of Lose Trades" } });
    }
    #endregion

    // Rewrite the file with the changes
    xmlDoc.Save(pathToFile);

    // Clear the variable which stored results written to a file
    ClearReportItem();
}
```

First we load the entire document to memory and then add
sections. Let us consider the Xpath request format which passes the path to the root node.

```
$"{xpath}/Result[last()]/Coefficients"
```

The xpath variable contains the path to the node in which the
optimization pass elements are stored. This node stores optimization results nodes which can be presented as an array of structures. The

Result\[last()\] construct selects the last element of
the array, after which the path is passed to the nested

/Coefficients node. Following the
described principle, we select the required node with the results of optimizations.

The next step is adding of robot parameters: in the loop we add
parameters directly to the results directory. Then add

a number of coefficients into the coefficients directory. This
addition is divided into blocks. As a result we

save results and clear
the temporary storage. As a result we get a file with the list of parameters and optimization results. To separate threads during asynchronous
operations launched from different processes (this is how optimization in the tester is performed when using multiple processors),
another writing method has been created, which separates threads using named mutexes.

```
/// <summary>
/// Write to file while locking using a named mutex
/// </summary>
/// <param name="mutexName">Mutex name</param>
/// <param name="pathToBot">Path to the bot</param>
/// <param name="currency">Deposit currency</param>
/// <param name="balance">Balance</param>
/// <param name="leverage">Leverage</param>
/// <param name="pathToFile">Path to file</param>
/// <param name="symbol">Symbol</param>
/// <param name="tf">Timeframe</param>
/// <param name="StartDT">Trading start dare</param>
/// <param name="FinishDT">Trading end date</param>
/// <returns></returns>
public static string MutexWriter(string mutexName, string pathToBot, string currency, double balance,
                                 int leverage, string pathToFile, string symbol, int tf,
                                 ulong StartDT, ulong FinishDT)
{
    string ans = "";
    // Mutex lock
    Mutex m = new Mutex(false, mutexName);
    m.WaitOne();
    try
    {
        // write to file
        Write(pathToBot, currency, balance, leverage, pathToFile, symbol, tf, StartDT, FinishDT);
    }
    catch (Exception e)
    {
        // Catch error if any
        ans = e.Message;
    }

    // Release the mutex
    m.ReleaseMutex();
    // Return error text
    return ans;
}
```

This method writes data using the previous method but the writing process is wrapped by a mutex and in a try-catch block. The last one enables
mutex release even on case of an error. Otherwise the process may freeze and optimization may fail to continue. These methods are also used in
the OptimisationResult structure in the WriteResult method.

```
/// <summary>
/// The method adds current parameter to the existing file or creates a new file with the current parameter
/// </summary>
/// <param name="pathToBot">Relative path to the robot from the Experts folder</param>
/// <param name="currency">Deposit currency</param>
/// <param name="balance">Balance</param>
/// <param name="leverage">Leverage</param>
/// <param name="pathToFile">Path to file</param>
public void WriteResult(string pathToBot,
                        string currency, double balance,
                        int leverage, string pathToFile)
{
    try
    {
        foreach (var param in report.BotParams)
        {
            ReportWriter.AppendBotParam(param.Key, param.Value);
        }
        ReportWriter.AppendMainCoef(GetResult(ReportManager.SortBy.Payoff),
                                    GetResult(ReportManager.SortBy.ProfitFactor),
                                    GetResult(ReportManager.SortBy.AverageProfitFactor),
                                    GetResult(ReportManager.SortBy.RecoveryFactor),
                                    GetResult(ReportManager.SortBy.AverageRecoveryFactor),
                                    (int)GetResult(ReportManager.SortBy.TotalTrades),
                                    GetResult(ReportManager.SortBy.PL),
                                    GetResult(ReportManager.SortBy.DD),
                                    GetResult(ReportManager.SortBy.AltmanZScore));

        ReportWriter.AppendVaR(GetResult(ReportManager.SortBy.Q_90), GetResult(ReportManager.SortBy.Q_95),
                               GetResult(ReportManager.SortBy.Q_99), GetResult(ReportManager.SortBy.Mx),
                               GetResult(ReportManager.SortBy.Std));

        ReportWriter.AppendMaxPLDD(GetResult(ReportManager.SortBy.ProfitFactor), GetResult(ReportManager.SortBy.MaxDD),
                                  (int)GetResult(ReportManager.SortBy.MaxProfitTotalTrades),
                                  (int)GetResult(ReportManager.SortBy.MaxDDTotalTrades),
                                  (int)GetResult(ReportManager.SortBy.MaxProfitConsecutivesTrades),
                                  (int)GetResult(ReportManager.SortBy.MaxDDConsecutivesTrades));

        foreach (var day in report.OptimisationCoefficients.TradingDays)
        {
            ReportWriter.AppendDay((int)day.Key, day.Value.Profit.Value, day.Value.Profit.Value,
                                   day.Value.Profit.Trades, day.Value.DD.Trades);
        }

        ReportWriter.Write(pathToBot, currency, balance, leverage, pathToFile, report.Symbol, report.TF,
                           report.DateBorders.From.DTToUnixDT(), report.DateBorders.Till.DTToUnixDT());
    }
    catch (Exception e)
    {
        ReportWriter.ClearReportItem();
        throw e;
    }
}
```

In this method, we alternately add optimization results to a temporary storage, then call the Write
method to save them in an existing file or create a new file if it has not yet been created.

The described method for writing obtained data is required for adding information to a prepared file. There is another method which is better
suitable when a data series needs to be written. The method has been developed as an extension for the
IEnumerable<OptimisationResult> interface. Now we can save data for all lists inherited from the appropriate interface.

```
public static void ReportWriter(this IEnumerable<OptimisationResult> results, string pathToBot,
                                string currency, double balance,
                                int leverage, string pathToFile)
{
    // Delete the file if it exists
    if (File.Exists(pathToFile))
        File.Delete(pathToFile);

    // Create writer
    using (var xmlWriter = new XmlTextWriter(pathToFile, null))
    {
        // Set document format
        xmlWriter.Formatting = Formatting.Indented;
        xmlWriter.IndentChar = '\t';
        xmlWriter.Indentation = 1;

        xmlWriter.WriteStartDocument();

        // The root node of the document
        xmlWriter.WriteStartElement("Optimisation_Report");

        // Write attributes
        WriteAttribute(xmlWriter, "Created", DateTime.Now.ToString("dd.MM.yyyy HH:mm:ss"));

        // Write optimizer settings to file
        #region Optimiser settings section
        xmlWriter.WriteStartElement("Optimiser_Settings");

        WriteItem(xmlWriter, "Bot", pathToBot); // path to the robot
        WriteItem(xmlWriter, "Deposit", balance.ToString(), new Dictionary<string, string> { { "Currency", currency } }); // Currency and deposit
        WriteItem(xmlWriter, "Leverage", leverage.ToString()); // Leverage

        xmlWriter.WriteEndElement();
        #endregion

        // Write optimization results to the file
        #region Optimisation result section
        xmlWriter.WriteStartElement("Optimisation_Results");

        // Loop through optimization results
        foreach (var item in results)
        {
            // Write specific result
            xmlWriter.WriteStartElement("Result");

            // Write attributes of this optimization pass
            WriteAttribute(xmlWriter, "Symbol", item.report.Symbol); // Symbol
            WriteAttribute(xmlWriter, "TF", item.report.TF.ToString()); // Timeframe
            WriteAttribute(xmlWriter, "Start_DT", item.report.DateBorders.From.DTToUnixDT().ToString()); // Optimization start date
            WriteAttribute(xmlWriter, "Finish_DT", item.report.DateBorders.Till.DTToUnixDT().ToString()); // Optimization end date

            // Write optimization result
            WriteResultItem(item, xmlWriter);

            xmlWriter.WriteEndElement();
        }

        xmlWriter.WriteEndElement();
        #endregion

        xmlWriter.WriteEndElement();

        xmlWriter.WriteEndDocument();
        xmlWriter.Close();
    }
}
```

The method writes optimization reports into a file one by one until the array has no more data. If the file already exists at the passed path, it
will be replaced with a new one. First we

create a file writer and configure it. Then, following the already known
file structure, we write

optimizer settings and optimization
results one by one. As can be seen from the above code extract, the results are written in a loop, which loops through the elements of
the collection, at the instance of which the described method was called. Inside the loop, data writing is delegated to the

method created for writing data of a specific element into the file.

```
/// <summary>
/// Write a specific optimization pass
/// </summary>
/// <param name="resultItem">Optimization pass value</param>
/// <param name="writer">Writer</param>
private static void WriteResultItem(OptimisationResult resultItem, XmlTextWriter writer)
{
    // Write coefficients
    #region Coefficients
    writer.WriteStartElement("Coefficients");

    // Write VaR
    #region VaR
    writer.WriteStartElement("VaR");

    WriteItem(writer, "90", resultItem.GetResult(SortBy.Q_90).ToString()); // Quantile 90
    WriteItem(writer, "95", resultItem.GetResult(SortBy.Q_95).ToString()); // Quantile 95
    WriteItem(writer, "99", resultItem.GetResult(SortBy.Q_99).ToString()); // Quantile 99
    WriteItem(writer, "Mx", resultItem.GetResult(SortBy.Mx).ToString()); // Average for PL
    WriteItem(writer, "Std", resultItem.GetResult(SortBy.Std).ToString()); // Standard deviation for PL

    writer.WriteEndElement();
    #endregion

    // Write PL / DD parameters - extreme points
    #region Max PL DD
    writer.WriteStartElement("Max_PL_DD");
    WriteItem(writer, "Profit", resultItem.GetResult(SortBy.MaxProfit).ToString()); // Total profit
    WriteItem(writer, "DD", resultItem.GetResult(SortBy.MaxDD).ToString()); // Total loss
    WriteItem(writer, "Total Profit Trades", ((int)resultItem.GetResult(SortBy.MaxProfitTotalTrades)).ToString()); // Total number of winning trades
    WriteItem(writer, "Total Lose Trades", ((int)resultItem.GetResult(SortBy.MaxDDTotalTrades)).ToString()); // Total number of losing trades
    WriteItem(writer, "Consecutive Wins", ((int)resultItem.GetResult(SortBy.MaxProfitConsecutivesTrades)).ToString()); // Winning trades in a row
    WriteItem(writer, "Consecutive Lose", ((int)resultItem.GetResult(SortBy.MaxDDConsecutivesTrades)).ToString()); // Losing trades in a row
    writer.WriteEndElement();
    #endregion

    // Write trading results by days
    #region Trading_Days

    // The method writing trading results
    void AddDay(string Day, double Profit, double DD, int ProfitTrades, int DDTrades)
    {
        writer.WriteStartElement(Day);

        WriteItem(writer, "Profit", Profit.ToString()); // Profits
        WriteItem(writer, "DD", DD.ToString()); // Losses
        WriteItem(writer, "Number Of Profit Trades", ProfitTrades.ToString()); // Number of profitable trades
        WriteItem(writer, "Number Of Lose Trades", DDTrades.ToString()); // Number of losing trades

        writer.WriteEndElement();
    }

    writer.WriteStartElement("Trading_Days");

    // Monday
    AddDay("Mn", resultItem.GetResult(SortBy.AverageDailyProfit_Mn),
                 resultItem.GetResult(SortBy.AverageDailyDD_Mn),
                 (int)resultItem.GetResult(SortBy.AverageDailyProfitTrades_Mn),
                 (int)resultItem.GetResult(SortBy.AverageDailyDDTrades_Mn));
    // Tuesday
    AddDay("Tu", resultItem.GetResult(SortBy.AverageDailyProfit_Tu),
                 resultItem.GetResult(SortBy.AverageDailyDD_Tu),
                 (int)resultItem.GetResult(SortBy.AverageDailyProfitTrades_Tu),
                 (int)resultItem.GetResult(SortBy.AverageDailyDDTrades_Tu));
    // Wednesday
    AddDay("We", resultItem.GetResult(SortBy.AverageDailyProfit_We),
                 resultItem.GetResult(SortBy.AverageDailyDD_We),
                 (int)resultItem.GetResult(SortBy.AverageDailyProfitTrades_We),
                 (int)resultItem.GetResult(SortBy.AverageDailyDDTrades_We));
    // Thursday
    AddDay("Th", resultItem.GetResult(SortBy.AverageDailyProfit_Th),
                 resultItem.GetResult(SortBy.AverageDailyDD_Th),
                 (int)resultItem.GetResult(SortBy.AverageDailyProfitTrades_Th),
                 (int)resultItem.GetResult(SortBy.AverageDailyDDTrades_Th));
    // Friday
    AddDay("Fr", resultItem.GetResult(SortBy.AverageDailyProfit_Fr),
                 resultItem.GetResult(SortBy.AverageDailyDD_Fr),
                 (int)resultItem.GetResult(SortBy.AverageDailyProfitTrades_Fr),
                 (int)resultItem.GetResult(SortBy.AverageDailyDDTrades_Fr));

    writer.WriteEndElement();
    #endregion

    // Write other coefficients
    WriteItem(writer, "Payoff", resultItem.GetResult(SortBy.Payoff).ToString());
    WriteItem(writer, "Profit factor", resultItem.GetResult(SortBy.ProfitFactor).ToString());
    WriteItem(writer, "Average Profit factor", resultItem.GetResult(SortBy.AverageProfitFactor).ToString());
    WriteItem(writer, "Recovery factor", resultItem.GetResult(SortBy.RecoveryFactor).ToString());
    WriteItem(writer, "Average Recovery factor", resultItem.GetResult(SortBy.AverageRecoveryFactor).ToString());
    WriteItem(writer, "Total trades", ((int)resultItem.GetResult(SortBy.TotalTrades)).ToString());
    WriteItem(writer, "PL", resultItem.GetResult(SortBy.PL).ToString());
    WriteItem(writer, "DD", resultItem.GetResult(SortBy.DD).ToString());
    WriteItem(writer, "Altman Z Score", resultItem.GetResult(SortBy.AltmanZScore).ToString());

    writer.WriteEndElement();
    #endregion

    // Write robot coefficients
    #region Bot params
    foreach (var item in resultItem.report.BotParams)
    {
        WriteItem(writer, item.Key, item.Value);
    }
    #endregion
}
```

The implementation of the method that writes data to a file is very simple, although it is quite long. After creating appropriate sections and
filling the attributes the method adds data on

VaR of the performed optimization pass and values
characterizing the maximum profit and drawdown. A nested function
has been created to write optimization results for a specific date, which is called 5 times, for each of the days. After that coefficients
without grouping and root parameters are added. Since the
described procedure is performed in one loop for each of the elements, the data are not written to the file until the

xmlWriter.Close() method is called (this is
done in the main writing method). Thus, this is the fastest extension method for writing a data array, as compared to previously considered
methods. We have considered procedures related to data writing into a file. Now let us move on to the next logical part of the description,
i.e. data reading from the resulting file.

### Reading the optimization report file

We need to read the files in order to process the received information and to display it. Therefore, an appropriate file reading mechanism is
required. It is implemented as a separate class:

```
public class ReportReader : IDisposable
    {
        /// <summary>
        /// Constructor
        /// </summary>
        /// <param name="path">Path to file</param>
        public ReportReader(string path);

        /// <summary>
        /// Binary number format provider
        /// </summary>
        private readonly NumberFormatInfo formatInfo = new NumberFormatInfo { NumberDecimalSeparator = "." };

        #region DataKeepers
        /// <summary>
        /// Presenting the report file in OOP format
        /// </summary>
        private readonly XmlDocument document = new XmlDocument();

        /// <summary>
        /// Collection of document nodes (rows in excel table)
        /// </summary>
        private readonly System.Collections.IEnumerator enumerator;
        #endregion

        /// <summary>
        /// The read current report item
        /// </summary>
        public ReportItem? ReportItem { get; private set; } = null;

        #region Optimiser settings
        /// <summary>
        /// Path to the robot
        /// </summary>
        public string RelativePathToBot { get; }

        /// <summary>
        /// Balance
        /// </summary>
        public double Balance { get; }

        /// <summary>
        /// Currency
        /// </summary>
        public string Currency { get; }

        /// <summary>
        /// Leverage
        /// </summary>
        public int Leverage { get; }
        #endregion

        /// <summary>
        /// File creation date
        /// </summary>
        public DateTime Created { get; }

        /// <summary>
        /// File reader method
        /// </summary>
        /// <returns></returns>
        public bool Read();

        /// <summary>
        /// The method receiving the item by its name (the Name attribute)
        /// </summary>
        /// <param name="Name"></param>
        /// <returns></returns>
        private string SelectItem(string Name) => $"Item[@Name='{Name}']";

        /// <summary>
        /// Get the trading result value for the selected day
        /// </summary>
        /// <param name="dailyNode">Node of this day</param>
        /// <returns></returns>
        private DailyData GetDay(XmlNode dailyNode);

        /// <summary>
        /// Reset the quote reader
        /// </summary>
        public void ResetReader();

        /// <summary>
        /// Clear the document
        /// </summary>
        public void Dispose() => document.RemoveAll();
    }
```

Let's view the structure in more detail. The class is inherited from the iDisposable
interface. This is not a required condition, but is done for precaution. Now the describe class contains the required Dispasable
method which clears the document object. The object stores the
optimization results file loaded to memory.

The approach is convenient because when creating an instance, the class inherited from the above mentioned interface should be wrapped to
the 'using' construct, which automatically calls the specified method when it goes beyond the 'using' structure block boundaries. It
means that the read document will not be kept long in memory and thus the loaded memory amount is reduced.

The row-wise document reader class uses Enumerator
received from the read document. The read values are written to the

special property and thus we provide access to data. Also, the
following data is filled during class instantiation: properties specifying the main

optimizer settings, file
creation date and time. To eliminate the influence of OS localization settings (both when writing and when reading the file) the
double-precision number

delimiter format is indicated. When reading the file for the first
time, the class should be reset to list beginning. For this purpose we use the

ResetReader method which resets Enumerator to
the list beginning. The class constructor is implemented so as to fill in all the required properties and to prepare the class for further
use.

```
public ReportReader(string path)
{
    // load the document
    document.Load(path);

    // Get file creation date
    Created = DateTime.ParseExact(document["Optimisation_Report"].Attributes["Created"].Value, "dd.MM.yyyy HH:mm:ss", null);
    // Get enumerator
    enumerator = document["Optimisation_Report"]["Optimisation_Results"].ChildNodes.GetEnumerator();

    // Parameter receiving function
    string xpath(string Name) { return $"/Optimisation_Report/Optimiser_Settings/Item[@Name='{Name}']"; }

    // Get path to the robot
    RelativePathToBot = document.SelectSingleNode(xpath("Bot")).InnerText;

    // Get balance and deposit currency
    XmlNode Deposit = document.SelectSingleNode(xpath("Deposit"));
    Balance = Convert.ToDouble(Deposit.InnerText.Replace(",", "."), formatInfo);
    Currency = Deposit.Attributes["Currency"].Value;

    // Get leverage
    Leverage = Convert.ToInt32(document.SelectSingleNode(xpath("Leverage")).InnerText);
}
```

First of all it loads the passed document and fills its creation date. Enumerator
obtained during class instantiation belongs to the document child nodes located under section Optimisation\_Report/Optimisation\_Results,
i.e. to the nodes having tag <Result/>. To get the desired optimizer configuration parameters, path to the required document node is
specified using

xpath markup. An analogue of this built-in function having a shorter
path is the SelectItem method, which indicates the path to an item among document nodes having tag <Item/> according to its Name
attribute. The GetDay method converts the passed document node to the appropriate structure of the daily trading report. The last method in
this class is the data reader method. Its implementation in brief form is shown below.

```
public bool Read()
{
    if (enumerator == null)
        return false;

    // Read the next item
    bool ans = enumerator.MoveNext();
    if (ans)
    {
        // Current node
        XmlNode result = (XmlNode)enumerator.Current;
        // current report item
        ReportItem = new ReportItem[...]

        // Fill the robot parameters
        foreach (XmlNode item in result.ChildNodes)
        {
            if (item.Name == "Item")
                ReportItem.Value.BotParams.Add(item.Attributes["Name"].Value, item.InnerText);
        }

    }
    return ans;
}
```

The hidden code part contains the optimization report
instantiation operation and filling of the report fields with the read data. This operation includes similar actions, which convert the
string format to the required one. Further loop fills in the robot parameters using data read row by row from the file. This operation is only
performed

if the completing file line has not been reached. The result of the
operation is the

returning of an indication of whether the row was read or not. It also
serves as an indication of reaching the end of the file.

### Multifactor Filtering and Sorting of the Optimization Report

To meet the objectives, I created two enumerations that indicated the sorting direction (SortMethd and OrderBy). They are similar and
probably only one of them could be enough. However, in order to separate filtering and sorting methods, two enumerations were created
instead of one. The purpose of the enumerations is to show the ascending or descending order. The ratio type of the coefficients with the
passed value is indicated by flags. The purpose is to set the comparison condition.

```
/// <summary>
/// Filtering type
/// </summary>
[Flags]
public enum CompareType
{
    GraterThan = 1, // greater than
    LessThan = 2, // less than
    EqualTo = 4 // equal
}
```

Type of coefficients by which the data can be filtered and sorted are described by the aforementioned enumeration OrderBy. Sorting and
filtering methods are implemented as methods expanding collections inherited from the IEnumerable<OptimisationResult>
interface. In the filtering method, we check each of the coefficients item by item, whether it meets the specified criteria, and reject the
optimization passes in which any of the coefficients does not meet the criteria. To filter data we use the Where loop contained in the
IEnumerable interface. The method is implemented as follows.

```
/// <summary>
/// Optimization filtering method
/// </summary>
/// <param name="results">Current collection</param>
/// <param name="compareData">Collection of coefficients and filtering types</param>
/// <returns>Filtered collection</returns>
public static IEnumerable<OptimisationResult> FiltreOptimisations(this IEnumerable<OptimisationResult> results,
                                                                  IDictionary<SortBy, KeyValuePair<CompareType, double>> compareData)
{
    // Result sorting function
    bool Compare(double _data, KeyValuePair<CompareType, double> compareParams)
    {
        // Comparison result
        bool ans = false;
        // Comparison for equality
        if (compareParams.Key.HasFlag(CompareType.EqualTo))
        {
            ans = compareParams.Value == _data;
        }
        // Comparison for 'greater than current'
        if (!ans && compareParams.Key.HasFlag(CompareType.GraterThan))
        {
            ans = _data > compareParams.Value;
        }
        // Comparison for 'less than current'
        if (!ans && compareParams.Key.HasFlag(CompareType.LessThan))
        {
            ans = _data < compareParams.Value;
        }

        return ans;
    }
    // Sorting condition
    bool Sort(OptimisationResult x)
    {
        // Loop through passed sorting parameters
        foreach (var item in compareData)
        {
            // Compare the passed parameter with the current one
            if (!Compare(x.GetResult(item.Key), item.Value))
                return false;
        }

        return true;
    }

    // Filtering
    return results.Where(x => Sort(x));
}
```

Two functions are implemented inside the method, each of them performs its own part of data filtering task. Let's view them, starting with the
final function:

- Compare — its purpose is to compare the passed value presented as KeyValuePair and the value specified in the method. In addition to
greater/less and equality comparison, we may need to check other conditions. For this purpose we will utilize flags. A flag is one bit,
while the int field stores 8 bits. Thus we can have up to 8 simultaneously set or removed flags for the int field. Flags can be checked
sequentially, without the need to create multiple loops or huge conditions, and thus we only have three conditions. Moreover, in the
graphical interface which we will consider later it is also convenient to use flags to set required comparison parameters. We
sequentially check the flags in this function and also check whether data correspond to these flags.
- Sort: unlike the previous method, this one is designed to check multiple written parameters instead of one. We run an item-wise loop
through all flags passed for filtering and use the previously described function to find out whether the selected parameter meets the
specified criteria. To enable access to a value of the specific selected item in the loop without using the "Switch case" operator, the
aforementioned OptimisationResult.GetResult(OrderBy item) method is used. If the passed value does not match the requested one,
return false and thus discard unsuitable values.

The 'Where' method is used to sort data. It automatically generates a list of suitable conditions, which is returned as the extension method
execution result.

Data filtering is quite easy to understand. Difficulties may occur with sorting. Let us consider the sorting mechanism using an example.
Suppose we have Profit Factor and Recovery Factor parameters. We need to sort data by these two parameters. If we perform two sorting
iterations one after another, we will still receive data sorted by the last parameter. We need to compare these values in some way.

| Profit | Profit factor | Recovery factor |
| --- | --- | --- |
| 5000 | 1 | 9 |
| 15000 | 1.2 | 5 |
| -11000 | 0.5 | -2 |
| 0 | 0 | 0 |
| 10000 | 2 | 5 |
| 7000 | 1 | 4 |

These two coefficients are not normalized within their boundary values. Also they have very wide range of values relative to each other.
Logically, we first need to normalize them while preserving their sequence. The standard way to bring data to a normalized form is to divide
each of them by the maximum value in the series: thus we will obtain a series of values that vary in the range \[0;1\]. But first, we need to find the
extreme points of this series of values presented in the table.

|  | Profit factor | Recovery factor |
| --- | --- | --- |
| Min | 0 | -2 |
| Max | 2 | 9 |

As can be seen from the table, Recovery factor has negative values and thus the above approach is not suitable here. In order to eliminate this
effect, we simply shift the entire series by a negative value taken modulo. Now we can calculate the normalized value of each of the
parameters.

| Profit | Profit <br> factor | Recovery <br> factor | Normalized sum |
| --- | --- | --- | --- |
| 5000 | 0.5 | 1 | 0.75 |
| 15000 | 0.6 | 0.64 | 0.62 |
| -11000 | 0.25 | 0 | 0.13 |
| 0 | 0 | 0.18 | 0.09 |
| 10000 | 1 | 0.64 | 0.82 |
| 7000 | 0.5 | 0.55 | 0.52 |

Now that we have all the coefficients in the normalized form, we can use the weighted sum, in which the weight is equal to one divided by n (here n
is the number of factor being weighted). As a result we obtain a normalized column which can be used as the sorting criteria. If any of the
coefficients should be sorted in the descending order, we need to subtract this parameter from one and thus to swap the largest and the
smallest coefficients.

The code implementing this mechanism is presented as two methods, the first one of which indicates the sorting order (ascending or
descending), and the second method implements the sorting mechanism. The first of the methods, SortMethod GetSortMethod(SortBy
sortBy), is quite simple, so let's move on to the second method.

```
public static IEnumerable<OptimisationResult> SortOptimisations(this IEnumerable<OptimisationResult> results,
                                                                OrderBy order, IEnumerable<SortBy> sortingFlags,
                                                                Func<SortBy, SortMethod> sortMethod = null)
{
    // Get the unique list of flags for sorting
    sortingFlags = sortingFlags.Distinct();
    // Check flags
    if (sortingFlags.Count() == 0)
        return null;
    // If there is one flag, sort by this parameter
    if (sortingFlags.Count() == 1)
    {
        if (order == OrderBy.Ascending)
            return results.OrderBy(x => x.GetResult(sortingFlags.ElementAt(0)));
        else
            return results.OrderByDescending(x => x.GetResult(sortingFlags.ElementAt(0)));
    }

    // Form minimum and maximum boundaries according to the passed optimization flags
    Dictionary<SortBy, MinMax> Borders = sortingFlags.ToDictionary(x => x, x => new MinMax { Max = double.MinValue, Min = double.MaxValue });

    #region create Borders min max dictionary
    // Loop through the list of optimization passes
    for (int i = 0; i < results.Count(); i++)
    {
        // Loop through sorting flags
        foreach (var item in sortingFlags)
        {
            // Get the value of the current coefficient
            double value = results.ElementAt(i).GetResult(item);
            MinMax mm = Borders[item];
            // Set the minimum and maximum values
            mm.Max = Math.Max(mm.Max, value);
            mm.Min = Math.Min(mm.Min, value);
            Borders[item] = mm;
        }
    }
    #endregion

    // The weight of the weighted sum of normalized coefficients
    double coef = (1.0 / Borders.Count);

    // Convert the list of optimization results to the List type array
    // Since it is faster to work with
    List<OptimisationResult> listOfResults = results.ToList();
    // Loop through optimization results
    for (int i = 0; i < listOfResults.Count; i++)
    {
        // Assign value to the current coefficient
        OptimisationResult data = listOfResults[i];
        // Zero the current sorting factor
        data.SortBy = 0;
        // Loop through the formed maximum and minimum borders
        foreach (var item in Borders)
        {
            // Get the current result value
            double value = listOfResults[i].GetResult(item.Key);
            MinMax mm = item.Value;

            // If the minimum is below zero, shift all data by the negative minimum value
            if (mm.Min < 0)
            {
                value += Math.Abs(mm.Min);
                mm.Max += Math.Abs(mm.Min);
            }

            // If the maximum is greater than zero, calculate
            if (mm.Max > 0)
            {
                // Calculate the coefficient according to the sorting method
                if ((sortMethod == null ? GetSortMethod(item.Key) : sortMethod(item.Key)) == SortMethod.Decreasing)
                {
                    // Calculate the coefficient to sort in descending order
                    data.SortBy += (1 - value / mm.Max) * coef;
                }
                else
                {
                    // Calculate the coefficient to sort in ascending order
                    data.SortBy += value / mm.Max * coef;
                }
            }
        }
        // Replace the value of the current coefficient with the sorting parameter
        listOfResults[i] = data;
    }

    // Sort according to the passed sorting type
    if (order == OrderBy.Ascending)
        return listOfResults.OrderBy(x => x.SortBy);
    else
        return listOfResults.OrderByDescending(x => x.SortBy);
}
```

If sorting is to be performed by one parameter, execute sorting
without resorting to the normalization of the series. Then immediately return the result. If sorting is to be performed by several parameters, we
first

generate a dictionary consisting of maximum and minimum values of the
considered series. This allows accelerating the calculations, since otherwise we would need to request parameters during each
iteration. This would generate much more loops than we have considered in this implementation.

Then, weight is formed for the weighted summation, and an
operation is performed to normalize a series to its sum. Here two loops are used again, the above described operations are performed in the
internal loop. The resulting weighted sum is added to the

SortBy variable of the appropriate array element. At the end of this
operation, when the resulting coefficient to be used for sorting has already been formed, use the previously described sorting method via
the standard

List<T>.OrderBy or List<T> array method. OrderByDescending   — when
descending sorting is needed. Sorting method for separate members of the weighted sum is set by a

delegate passed as one of the function parameters.
If this delegate is left as a default parametrized value, the earlier mentioned method is used; otherwise the passed delegate is
used.

### Conclusion

We have created a mechanism that will be actively used within our application in the future. In addition to
the unloading and reading of xml files of a custom format, which store structured information about performed tests, the mechanism
contains C# collection expanding methods, which are used to sort and filter data. We have implemented the multi-factor sorting mechanism,
which is not available in the standard terminal tester. One of the advantages of the sorting method is the ability to account for a series of
factors. However, its disadvantage is that the results can only be compared within the given series. It means that the weighted sum of the
selected time interval cannot be compared with other intervals, because each of them uses an individual series of coefficients. In the next
articles, we will consider the algorithm conversion method to enable the application or an automated optimizer for the algorithms, as well
as the creation of such an automated optimizer.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/7290](https://www.mql5.com/ru/articles/7290)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/7290.zip "Download all attachments in the single ZIP archive")

[Metatrader\_Auto\_Optimiser\_Part\_1.zip](https://www.mql5.com/en/articles/download/7290/metatrader_auto_optimiser_part_1.zip "Download Metatrader_Auto_Optimiser_Part_1.zip")(256.81 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Continuous walk-forward optimization (Part 8): Program improvements and fixes](https://www.mql5.com/en/articles/7891)
- [Continuous Walk-Forward Optimization (Part 7): Binding Auto Optimizer's logical part with graphics and controlling graphics from the program](https://www.mql5.com/en/articles/7747)
- [Continuous Walk-Forward Optimization (Part 6): Auto optimizer's logical part and structure](https://www.mql5.com/en/articles/7718)
- [Continuous Walk-Forward Optimization (Part 5): Auto Optimizer project overview and creation of a GUI](https://www.mql5.com/en/articles/7583)
- [Continuous Walk-Forward Optimization (Part 4): Optimization Manager (Auto Optimizer)](https://www.mql5.com/en/articles/7538)
- [Continuous Walk-Forward Optimization (Part 3): Adapting a Robot to Auto Optimizer](https://www.mql5.com/en/articles/7490)
- [Continuous Walk-Forward Optimization (Part 2): Mechanism for creating an optimization report for any robot](https://www.mql5.com/en/articles/7452)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/330722)**
(15)


![Irina Dymura](https://c.mql5.com/avatar/2015/4/55270820-86F6.gif)

**[Irina Dymura](https://www.mql5.com/en/users/irina111)**
\|
14 Feb 2020 at 22:08

The article is interesting. Everything is clearly described. I am just learning C# programming. Is it possible to write a robot in C#? And how to do it, so that it could trade in termenal? Thank you very much!


![Andrey Azatskiy](https://c.mql5.com/avatar/2018/6/5B127D58-708F.jpg)

**[Andrey Azatskiy](https://www.mql5.com/en/users/andreykrivcov)**
\|
14 Feb 2020 at 22:33

**Irina Dymura:**

The article is interesting. Everything is clearly described. I am just learning C# programming. Is it possible to write a robot in C#? And how to do it so that it could trade in termenal? Thank you very much!

Anything is possible, but it is much better to write it in MQL5. However, either for educational purposes or for other needs, you can try. There are several options, the simplest and most optimal of them is to write all the logic in C# in the project dll (dynamic library). Then declare a public class with a number of public static methods in it (they will be exported as C functions in MQL5). Approximately, the class should contain a method that initialises the robot, destroys the robot and is called on each tick (similar to OnInit, OnDeinit, OnTick). Then the robot is created in MQL5 where the mentioned static functions from the created dll are exported and after compilation the following docking will be obtained:

1\. MQL5 calls the initialising method from the dll in OnInit. The dll initialises the robot class into a static variable.

2\. MQL5 OnTick calls the OnTick method from dll, and in response receives a sign to sell / buy / do nothing. If it is necessary to enter a deal, we enter using the code written in MQL5.

3\. MQL5 OnDeinit deletes the robot, we call the OnDeinit method from the dll and do something. In C# you don't need to delete classes, Garbige Collector does it, everything with memory is practically automated there.

I.e. calculations are on the C# side, and trading is on the MQL5 side.

There are also some similar APIs for the terminal, where the code from C# directly interacts with MQL5 through pipes or other connections. I even came across such a project on github, but in my opinion it is easier to write everything through a dll.

In the last series of articles about optimisation management, I showed how to use a dll to connect WPF GUI with C#. You can use the same method to transfer the robot to C#. Before my article they also wrote about the GUI, but only WinForms and MQL5, I have adopted part of this mechanism, I don't remember the link to that article, but it is also quite useful. I think I referred to it somewhere in [this](https://www.mql5.com/en/articles/7029) article.

Also, in the 3rd article of this series of articles, it is described how to export a number of static functions to MQL5 from dll.

![Guilherme Mendonca](https://c.mql5.com/avatar/2018/9/5B98163A-29AC.jpg)

**[Guilherme Mendonca](https://www.mql5.com/en/users/billy-gui)**
\|
11 Mar 2020 at 13:42

Hello!

So before I can use this program, will I have to convert using [Visual Studio](https://www.mql5.com/en/articles/5798 "Article: How to write a DLL in MQL5 in 10 minutes (Part II): Writing in Visual Studio 2017 ")?


![Andrey Azatskiy](https://c.mql5.com/avatar/2018/6/5B127D58-708F.jpg)

**[Andrey Azatskiy](https://www.mql5.com/en/users/andreykrivcov)**
\|
11 Mar 2020 at 17:21

**Guilherme Mendonca:**

Hello!

So before I can use this program, will I have to convert using Visual Studio?

Hello. This is the first article from 5 parts that are already published. At the 4rth part - there is full program. And yes, you have to compile some code it [visual studio](https://www.mql5.com/en/articles/5798 "Article: How to write a DLL in MQL5 in 10 minutes (Part II): Writing in Visual Studio 2017 "). If say about code that where ateched to the current article - so yes, it must be compiled in visual studio.

![MUZIC Gaming](https://c.mql5.com/avatar/2020/8/5F2A0021-9E30.png)

**[MUZIC Gaming](https://www.mql5.com/en/users/muzicgaming)**
\|
5 Aug 2020 at 10:30

https://www.mql5.com/en/articles/7059


![Library for easy and quick development of MetaTrader programs (part XXIV): Base trading class - auto correction of invalid parameters](https://c.mql5.com/2/37/MQL5-avatar-doeasy__6.png)[Library for easy and quick development of MetaTrader programs (part XXIV): Base trading class - auto correction of invalid parameters](https://www.mql5.com/en/articles/7326)

In this article, we will have a look at the handler of invalid trading order parameters and improve the trading event class. Now all trading events (both single ones and the ones occurred simultaneously within one tick) will be defined in programs correctly.

![Library for easy and quick development of MetaTrader programs (part XXIII): Base trading class - verification of valid parameters](https://c.mql5.com/2/37/MQL5-avatar-doeasy__5.png)[Library for easy and quick development of MetaTrader programs (part XXIII): Base trading class - verification of valid parameters](https://www.mql5.com/en/articles/7286)

In the article, we continue the development of the trading class by implementing the control over incorrect trading order parameter values and voicing trading events.

![Extending Strategy Builder Functionality](https://c.mql5.com/2/37/Article_Logo__1.png)[Extending Strategy Builder Functionality](https://www.mql5.com/en/articles/7361)

In the previous two articles, we discussed the application of Merrill patterns to various data types. An application was developed to test the presented ideas. In this article, we will continue working with the Strategy Builder, to improve its efficiency and to implement new features and capabilities.

![Library for easy and quick development of MetaTrader programs (part XXII): Trading classes - Base trading class, verification of limitations](https://c.mql5.com/2/37/MQL5-avatar-doeasy__4.png)[Library for easy and quick development of MetaTrader programs (part XXII): Trading classes - Base trading class, verification of limitations](https://www.mql5.com/en/articles/7258)

In this article, we will start the development of the library base trading class and add the initial verification of permissions to conduct trading operations to its first version. Besides, we will slightly expand the features and content of the base trading class.

[![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/01.png)![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/02.png)Powerful analytics for traders of any levelAll the necessary trading reports for beginners and professionals](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/454106&a=muccpajyfystoakuukdobwigjejzmpqn&s=52daad60fa795e635264e6f94898f05493bca3b5124d4cca8eb7e82333c2ef12&uid=&ref=https://www.mql5.com/en/articles/7290&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5071689699216534687)

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