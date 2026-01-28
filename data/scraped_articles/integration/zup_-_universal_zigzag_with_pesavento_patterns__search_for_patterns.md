---
title: ZUP - Universal ZigZag with Pesavento patterns. Search for patterns
url: https://www.mql5.com/en/articles/2990
categories: Integration, Indicators
relevance_score: 1
scraped_at: 2026-01-23T21:42:37.164414
---

[Best articles and CodeBase updates in MQL5.community channelsFollow us to ensure you never miss out on important updates![](https://www.mql5.com/ff/sh/n9yf51p2srwzfqh5z2/02.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=dgazvhktsxqakdvarucjbvmvzenwlyje&s=98a038fe082e458df8c4a1d8e116e3a6646fd5517f06e48b2356b7ee005817d6&uid=&ref=https://www.mql5.com/en/articles/2990&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5072034898623017754)

MetaTrader 5 / Integration


### Introduction

Many traders are interested in searching for patterns on price charts. This search can be performed using the ZUP indicator platform. There are several options for working with the platform. First, you can search for known patterns with strictly set parameters. Second, you can adjust the parameters to your requirements. Third, you can create your own patterns using the ZUP graphical interface and save their parameters to a file. This capability allows you to quickly check, whether these new patterns can be found on charts.

Patterns were described by many authors. One of them was Harold Gartley. Gartley published his book "Profits in the Stock Markets" back in 1935. One of the book pages contained an image of a pattern and a brief description of how it can be used for trading. Active development of this idea started only after several decades. Many new patterns were described at that time. One of the most famous authors of those patterns is Scott Carney. Carney developed a plethora of patterns, including the Gartley, the Bat, the Butterfly, the Crab and others. Scott Carney registered the copyright on all these patterns in 2016.

There are many algorithms for finding patterns. For example, you can select extremums on the chart, "stretch" the Fibonacci grid between them and determine the emergence of a pattern when the market reaches specific levels. This algorithm is used both for manual searching, and in some programs. Another option is a purely software solution. A program algorithm finds extreme points, checks their mutual arrangement, and if it corresponds to a pattern, the program shows an appropriate message.

Multiple possibilities for finding patterns are implemented in the ZUP indicator platform. They can be built manually. But the main algorithm is the automated search for patterns on market extremums, which are determined through various ZigZags. In ZUP versions up to 151, pattern search modes were set through parameters. But there are more than 100 parameters which can influence the search for patterns. Such an abundance of parameters complicates the task. A new option was introduced in version 152: the possibility to use the graphical interface. Basic settings are set through parameters, while fast connection of patterns to search is performed using the graphical interface.

For convenience, all parameters are divided into groups. Below are the groups of parameters, which affect search for patterns.

1. ZigZag settings.
2. 1 - Parameters for fibo Levels.
3. 2 - Parameters for Pesavento Patterns.
4. 3 - Parameters for Gartley Patterns.
5. 15 - Common Parameters.

A large number of built-in graphical instruments can also help to find patterns.

In the first part of the article, I provide a description of how patterns can be connected to search through the graphical interface in ZUP version 152. Then I explain parameters for finding Gartley patterns. Other patterns are briefly considered in the end of the article.

### Part One. Graphical interface for working with patterns

Graphical interface elements for working with patterns were added in version 152. The graphical interface enables the following:

1. Connecting and disconnecting patterns from search;
2. Viewing images with the templates of five-dot patterns;
3. Creating new five-dot patterns;
4. Editing parameters of five-dot patterns;
5. Saving the edited patterns instead of the old one or in a separate \*.CSV file;
6. Saving a newly created pattern to a new or selected \*.CSV file;
7. Connecting patterns from the selected file to search;
8. Creating lists of patterns to search for.

The ExtPanel parameter at the end of parameters list of the indicator platform is provided for working with the ZUP graphical interface. ExtPanel=true allows working with the graphical interface.

If you set ExtPanel=true, but the graphical interface does not show up or it has been deleted by pressing ESC, the combination of SHIFT-Z will display it.

Then press Insert. In the next menu, click on the "butterfly icon" —  ![butterfly](https://c.mql5.com/2/31/13u.png).  This opens a menu for working with patterns:

![menu](https://c.mql5.com/2/31/menu.png)

Compared to version 151, this one contains the Select button. A click on the button opens the following menu with a list:

![menu viewing](https://c.mql5.com/2/31/menu_Vieving.png)

The menu allows connecting patterns from the list to search or disconnecting them without using the indicator's parameters panel. This can be done by simply clicking on the appropriate button. The values of parameters of patterns to connect will be set in the list of ZUP parameters.

Button Viewing of 5 dot patterns opens the panel for working with five-dot patterns. A click on it opens the following screen with a chart:

![chart](https://c.mql5.com/2/31/grafik__3.png)

Note that the image with the pattern template is displayed using the MQL4 language. The coordinates of pattern points XABCD use the time of conditional bars on the chart. The time that can be set is limited to around 2033. Therefore, it is not recommended to maximally compress the chart on the MN timeframe when working with this UI element. The time of conditional bars, to which the patterns are "tied", may become more than 2033 when you compress the chart. This causes an error. When the chart is compressed to the maximum, the pattern template will either be distorted or will not be displayed at all.

What can be done with the resulting window?

Line     ![](https://c.mql5.com/2/31/select_pattern.png) duplicates/replaces the value of the SelectPattern parameter, which sets groups of 5-dot patterns for search.

When you work with the graphical interface, the values of parameters set through this interface are used. If the graphical interface is disabled ExtPanel=false, values set on the indicator parameters editing tab are used. Individual parameters are available on each chart in this case.

The Bullish button in line ![Bullish/Bearish](https://c.mql5.com/2/31/bullishbearish.png) sets the display of an image with the bullish or bearish interpretation of the pattern template (by default, the bullish interpretation is shown first) and the name of the displayed pattern.

Line ![visible pattern](https://c.mql5.com/2/31/visiblePattern.png) corresponds to the visiblePattern parameter, which sets the list of patterns connected to the search.

Each position in this line corresponds to one of 33 built-in ZUP patterns. If 1 is set in the position of a selected pattern, it means the pattern is added to search. 0 means the pattern is not used.

You can change the list of patterns added to the search in several ways.

1. Move the cursor to the selected position and click the left mouse button. The first click shows a picture with a pattern template. The second click changes 1 to 0 or 0 to 1.
2. Using line ![select](https://c.mql5.com/2/31/select.png) . Numbers 1 of 33 show the position, where the pink cursor is set. The cursor also shows which pattern is selected for manipulation. Use arrows to set the new position of the cursor. Yes/No changes 1 to 0 or 0 to 1.

The Hide button sets all positions to 0, i.e. all patterns are removed from the search. The following should be noted here. You can add more than 33 patterns to the search. But the line of ones and zeros has only 33 positions. When the pink cursor is moved to the leftmost position and << is clicked, the previous 33 patterns are shown. When you bring the pink cursor to the rightmost (33th) position and click >>, the next 33 patterns are shown.

    For program debugging, I connected a file with about 265 patterns, which were created by one of ZUP users in earlier indicator versions and were only used for search. Earlier, patterns were created using external programs, which enabled generation of files in \*.CSV format. This is quite a difficult job. Now, patterns can be created by writing their parameters to a CSV file. But it is even easier to use the graphical interface.

3. Arrows in line  ![select patern](https://c.mql5.com/2/31/select_pattern__1.png) allow selectinggroups of patterns set in the ZUP code through the SelectPattern parameter and connecting them to search. But the operation of this parameter depends on whether only ZUP patterns are connected to the search, or patterns from an external file are additionally used. Arrows in  ![read from file list pattern](https://c.mql5.com/2/31/readFromFileListPatterns0.png) allow selecting the value of readFromFileListPatterns \- "Patterns from ZUP/"Patterns from external file"/"Patterns from ZUP and from file". This parameter sets the place to store patterns. Using arrows in ![file](https://c.mql5.com/2/31/file.png) you can select an external file with the parameters of patterns. This line shows that you can connect file 1 of two 1/2, which is called M. This is a file with the parameters of Merrill patterns.

The New button allows creating a new pattern. If patterns from an external file are connected, buttons Edit and Delete can appear depending on the value selected in line ![read from file list pattern](https://c.mql5.com/2/31/readFromFileListPatterns0__1.png) . If there are no external files, button Edit can appear for built-in ZUP patterns. This option allows creating custom patterns based on the available 33 patterns by editing their parameters.

Below, you can additionally see line ![add parameter](https://c.mql5.com/2/31/add_parametr.png) or multiple lines with the values of additional parameters, which are specific for the pattern in the right image.

A click on New opens a list of 8 lines with parameters, which can be set for 5-dot patterns.

![list of parameters](https://c.mql5.com/2/31/list_parametrs.png)

A repeated click on New returns the previous screen value. Action of any buttons in the ZUP GUI can be canceled by a repeated click on the same button or by pressing ESC.

A click on one of the lines with the list of parameters opens the following window:

![](https://c.mql5.com/2/31/kartinka.png)

In this case the fifth line was clicked. The image shows an example of a pattern, which can be created using the selected parameters. Values of the initial pattern parameters are used for the default values of parameters of this new pattern. Parameters can be changed. Jokingly, a new pattern was created based on this one Baba yaga:

![baba yaga](https://c.mql5.com/2/31/baba_yaga.png)

A repeated click on the fifth line shows the following:

![](https://c.mql5.com/2/31/nachalo.png)

Line  ![parameters pattern](https://c.mql5.com/2/31/parametrs_patterns.png) shows editable parameters.

Line ![parameter](https://c.mql5.com/2/31/parametr.png) shows the selected value of the parameter. The new value of the parameter is selected using arrows. It can be edited manually by clicking 0.382 and editing the parameter value. When a parameter value changes, the graphical presentation of the pattern also changes.

Button Next sets the selected value and brings you to setting the value of the next parameter.

The number of not yet set parameters in line ![parameters pattern](https://c.mql5.com/2/31/parametrs_patterns__1.png) decreases, and the next pattern element BC appears.

Once the first parameter value is fixed by a click on Next , the Back button appears allowing to revert changes and edit the previous values of parameters.

Two buttons ![](https://c.mql5.com/2/31/tu_buttons.png) allow you to connect the following parameters at any time: XB line filter and the ratio of the lengths of pattern wings. The ratio is set using arrows. A click on a value selected using arrows allows setting a custom ratio manually. ATTENTION! Set to 0 to disable the ratio.

Once you finish editing the parameters of the new pattern, you will be offered to set its name:

![edit name pattern](https://c.mql5.com/2/31/name_pattern.png)

Let's call it, for example, AA.

After that you will be prompted to select the name of the file to save the newly created pattern to. The file connected as an external file with patterns is selected by default. The file name can be edited; in this case the pattern will be saved to the file with the edited name. If such a file has already been created, the pattern will be added to the end of the list. If the file with this name does not exist, a new file will be created.

![Choose/edit file](https://c.mql5.com/2/31/kartinka_sfu.png)

Editing of pattern parameters is similar to the creation of a new parameter. At the end of the editing, you will be prompted to save the resulting pattern in place of the edited one, or save it as a new pattern in the specified \*.csv file. If you have edited a pattern from the built-in ZUP list, you will be offered to save the pattern as a new one in the selected file.

\*.csv files are saved to MQL4\\Files\\ZUP\\ListPatterns.

I omit the detailed description of some of the buttons. Their meaning is clear from the above descriptions.

The maximum allowable value can be shown for some parameters you create/edit. It will not be possible to change this value later. This is done in order to avoid setting of such parameter values, which would make the pattern meaningless with the selected ExtDeltaGartley and with the already set values ​​of the parameters. That is, patterns with such values ​​can never be found on the chart. The Ignore button appears. It allows setting any values for the parameter: later we may need to increase the value of ExtDeltaGartley, after which currently not allowed values will become valid.

### Part Two. Gartley patterns search parameters

Search for Gartley patterns is enabled using the ExtIndicator=11 parameter. For search, the chart interval of ExtMaxBar bars is scanned. This is the first search option. It is called "Search for patterns with a Scanner".

The search can also be performed with other values of ExtIndicator. In this case, ExtGartleyOnOff parameter is enabled and search is performed on one ZigZag set by the ExtIndicator parameter. This is the second search option: "Searching for patterns on a Zigzag".

When you switch to another timeframe, patterns found using the first two options will be deleted from the chart and a new search will be performed.

There is the third option for finding patterns. This option is described in the article " [ZUP - Universal ZigZag with Pesavento Patterns. Graphical interface](https://www.mql5.com/en/articles/2966)"

Patterns displayed on a chart using the third option are preserved when you switch to other timeframes. However, the fractal filter may disable their display on some timeframes.

Patterns created using the graphical interface can be used in all three search options.

Below is a complete list of parameters used when searching for Gartley patterns and when working with the graphical interface described in the first part of the article.

- AlgorithmSearchPatterns  — select the scanner algorithm for searching for patterns for ExtIndicator=11:

  - 0 - corresponds to ExtIndicator=0 – standard ZigZag
  - 1 - corresponds to ExtIndicator=1 - Alex ZigZag, ray size is set in points
  - 2 - corresponds to ExtIndicator=1 - Alex ZigZag, ray size is set in %
  - 3 - corresponds to ExtIndicator=2
  - 4 - corresponds to ExtIndicator=4
  - 5 - corresponds to ExtIndicator=5
  - 6 - corresponds to ExtIndicator=12

- PotencialsLevels\_retXD \- allow displaying XD retracement levels of the potential five-dot patterns. In case of ExtIndicator=11, it is applied only if a pattern is found; further levels are displayed from its C point:


  - 0 - disable the output of potential levels
  - 1 - display potential levels together with the patterns if ExtGartleyOnOff=true
  - 2 - display potential levels. Patterns are disabled.

- visibleLevelsABCD  \- set a method of displaying the potential D point of the potential five-dot patterns:

  - 0 - additional levels are not displayed
  - 1 - display all options of the BD retracement levels
  - 2 - display all levels of various AB=CD options
  - 3 - display BD retracement levels and AB=CD options together

- maxDepth \- Depth (minBars) maximum value, up to which the ZigZag's Depth can be changed when actively searching for Gartley patterns. Applied if AlgorithmSearchPatterns=0.
- minDepth  \- set the minimum Depth value for searching for the Gartley patterns.
- FiboStep  \- include the following Backstep calculation when searching for patterns: Backstep=Depth\*1.618.
- IterationStepDepth  \- Depth ZigZag parameter change step when searching for the Gartley patterns.
- maxSize\_ \- maximum ray size in points. It is used when scanning the patterns if AlgorithmSearchPatterns=1, AlgorithmSearchPatterns=3, AlgorithmSearchPatterns=4, AlgorithmSearchPatterns=6.
- minSize\_ \- minimum ray value in points.
- IterationStepSize  \- Size ZigZag parameter change step when searching for the Gartley patterns.
- maxPercent\_ \- maximum percentage for calculating the Alex ZigZag. It is used when scanning the patterns if AlgorithmSearchPatterns=2.
- minPercent\_ \- minimum percentage for calculating the Alex ZigZag.
- IterationStepPercent  \- Percent ZigZag parameter change step (%).
- DirectionOfSearchMaxMin  \- direction of searching for patterns:

  - false - from minDepth to maxDepth;
  - true - from maxDepth to minDepth;

- SelectPattern  \- select five-dot patterns for searching:

  - 0 - search for all patterns
  - 1 - search for conventional patterns only - Gartle, Butterfly, Bat, Crab, except TOTAL
  - 2 - search for conventional and non-conventional patterns, except TOTAL
  - 3 - search for exotic patterns and anti-patterns, except TOTAL
  - 4 - search for anti-patterns only, except TOTAL
  - 5 - search for all patterns, except TOTAL
  - 6 - search for TOTAL only
  - 7 - random selection of five-dot patterns in order to search using visiblePattern
  - 8 - disable the search for five-dot patterns
  - "Gartley";
  - "Bat";
  - "Alternate Bat";
  - "Butterfly";
  - "Crab";
  - "Deep Crab";
  - "Leonardo";
  - "Shark";
  - "Cypher";
  - "Nen STAR";
  - "5-0";
  - "A Gartley";
  - "A Bat";
  - "A Alternate Bat";
  - "A Butterfly";
  - "A Crab";
  - "A Deep Crab";
  - "A Leonardo";
  - "A Shark";
  - "A Cypher";
  - "A Nen STAR";
  - "A 5-0";
  - "Black Swan";
  - "White Swan";
  - "Navarro 200";
  - "max Bat";
  - "max Gartley";
  - "max Butterfly";
  - "TOTAL 1";
  - "TOTAL 2";
  - "TOTAL 3";
  - "TOTAL 4";
  - "TOTAL".

- visiblePattern  \- set what patterns are to be searched for. Search for all patterns is disabled by default.
- NumberPattern  \- index number of the pattern used for ZigZag's calibration. The pattern parameters are displayed via InfoTF. If NumberPattern = 0, a ZigZag with the parameters as in ExtIndicator=0 mode is displayed.
- ExtGartleyTypeSearch  \- mode of searching for patterns:

  - 0 - finish a search after the first detected pattern
  - 1 - display all patterns on the segment specified by maxBarToD. The search is repeated at each ZigZag recalculation. "Chinese toy 1" mode
  - 2 - display all patterns on the segment specified by maxBarToD. The search is only performed once. "Chinese toy 2" mode

- ExtHiddenPP  \- ZigZag display mode for ExtIndicator=11:

  - 0 - ZigZag is not displayed. Only ZigZag extreme points are displayed as dots. No Pesavento patterns.
  - 1 - display a ZigZag calibrated by the pattern set by NumberPattern. The Pesavento patterns are displayed the usual way.
  - 2 - ZigZag is not displayed. Only ZigZag extreme points are displayed as dots. The Pesavento patterns are displayed only for the Gartley patterns

- ExtGartleyOnOff  \- display the Gartley patterns. Not in the scanner mode.
- maxBarToD  \- set the maximum number of bars from a zero point to a D point of the pattern.
- patternInfluence:

  - 0 - display patterns having no more than maxBarToD bars from a zero bar to a D point bar
  - 1 - consider the effect of a pattern, cancel the effect of maxBarToD
  - 2 - searching for patterns is performed on the entire ZigZag marking

- patternTrue  = true - display patterns satisfying the condition:

for bearish patterns, no bars with a High exceeding a High of a D point development area frame should be present from the D point to a zero bar;

for bullish patterns, no bars with a Low exceeding a Low of a D point development area frame should be present from the D point to a zero bar.

- AllowedBandPatternInfluence  \- set the ratio of the distance between the pattern's X and D points. This ratio sets the distance from a D point to the point where the pattern influence supposedly ends
- RangeForPointD  \- allow display of a D point development area
- OldNewRangeForPointD  \- select the method of building the pattern point development area construction
- ExtColorRangeForPointD  \- color of a D point development area frame
- VectorOfAMirrorTrend   = 1 display a trend vector
- VectorOfAMirrorTrend   = 2 display a mirror trend vector
- VectorOfAMirrorTrendColor  \- reverse trend line color
- VectorOfAMirrorTrendStyle  \- reverse trend line style
- shortNamePatterns  \- allow display of short pattern names
- visibleLineOrTriangle  \- allow displaying patterns as lines or triangles, except AB=CD patterns
- PatternLineStyle  \- set a line style for five-dot patterns and AB=CD
- PatternLineWidth  \- set a line width for five-dot patterns and AB=CD
- ExtColorPatternsBullish  \- color of bullish patterns
- ExtColorPatternsBearish  \- color of bearish patterns
- ExtColorPatternList  \- set the list of colors for filling the Gartley patterns in the "Chinese toy" mode. Color names are separated by commas. If some of the colors have an error in their names, the red color is set for them by default

- ExtDeltaGartley  \- tolerance for price deviation for searching patterns, the default values are 9% - 0.09
- ExtDeltaGartleyPRZ   \- special tolerance for building the pattern's D point development frame
- levelD  \- display XD retracements levels of the possible accurate pattern versions for the current combination
- colorLevelD  \- set colors of XD retracements levels
- Equilibrium  \- display Equilibrium, Reaction1 and Reaction2 lines
- ReactionType  — set reaction line type
- EquilibriumStyle  — set line style
- EquilibriumWidth  — set line width
- ColorEquilibrium  — set Equilibrium color
- ColorReaction  — set Reaction1 and Reaction2 color
- Ext\_3Drives \- display the 3 Drives pattern
- Ext\_xO \- set a ratio for searching the 3 Drives 7-dot pattern
- Dragon  — display the Dragon pattern
- PeakZZDragon  — set a ZigZag extreme point index number, up to which the search for the Dragon pattern is looked for
- Ext\_4PointPattern \- allow searching for the 4-dot continuation pattern
- \_maxXB \- set the maximum XB retracement value for The 4 Point Pattern. Find out more about the pattern here: http://kanetrading.com/
- ABCD  — enable searching for the AB=CD pattern:

  - 0 - hide the AB=CD patterns
  - 1 - display any AB=CD
  - 2 - display only harmonic AB=CD with the ratios corresponding to a Fibo series within a tolerance

- searchABCDAlternate  — display the alternative AB=CD patterns
- ABCDAlternate  — set the list of the alternative AB=CD patterns. X ratios from the X\*AB=CD equation are listed there separated by comma
- visibleABCDrayZZ  — display AB=CD patterns as a line
- Ext\_noname \- allow searching for unknown five-dot patterns with all its four retracements equal to one of Fibos

- CustomPattern  — display custom patterns:

  - 0 - hide custom patterns
  - 1 - display together with other patterns
  - 2 - display custom patterns only

- NameCustomPattern  — custom pattern name
- minXB  — set the minimum XB retracement value
- maxXB  — set the maximum XB retracement value
- minAC  — set the minimum AC retracement value
- maxAC  — set the maximum AC retracement value
- minBD  — set the minimum BD retracement value
- maxBD  — set the maximum BD retracement value
- minXD  — set the minimum XD retracement value
- maxXD  — set the maximum Equilibrium value of XD retracement.

Retracement High and Low set the search range

- filtrEquilibrium  — enable a filter line passing through X and B pattern points. If the filter is enabled, the pattern is displayed when the price breaks through the line while moving from C to D. It works only with custom and noname patterns.
- readFromFileListPatterns  — set reading of the pattern list from the \\\ZUP\\ListPatterns\\listpatterns.csv file:

  - 0 - disable reading the list of patterns from the file
  - 1 - use when searching for five-dot patterns of only the patterns that are on the list downloaded from the file
  - 2 - add the list of patterns from the file to the list of five-dot patterns built into ZUP. In this case, a composite list of patterns is generated.

Files \*.csv are located in MQL4\\Files\\ZUP\\ListPatterns.

- NameFileListPatterns  — set a name of a csv file the list of pattern parameters is read from
- writeToFileListPatterns  — allow writing the list of five-dot patterns:

  - 0 - write to \\\ZUP\\ListPatterns\\listpatternsdefault.csv
  - 1 - write to \\\ZUP\\ListPatterns\\listpatternscustom.csv
  - 2 - write to \\\ZUP\\ListPatterns\\listpatternsmixt.csv

- picture  — send a chart screenshot with a pattern in a file

Files with the screenshots of the found patterns are save to MQL4\\Files\\ZUP\\PicturePatterns.

- writeInfoPatternsToFileXML  — send the parameters of the current five-dot pattern to an XML file:

  - 0 - disable sending parameters to a file
  - 1 - time parameters are sent in expanded form
  - 2 - time parameters are sent in a number of seconds

The \*.xml files with the parameters of the found pattern are saved to MQL4\\Files\\ZUP\\XML.

- writeInfoPatternsToFileCSV  — send the parameters of the current five-dot pattern to an CSV file:

  - 0 - disable sending parameters to a file
  - 1 - time parameters are sent in expanded form
  - 2 - time parameters are sent in a number of seconds

The \*.csv files with the parameters of the found pattern are saved to MQL4\\Files\\ZUP\\CSV.

- namefileSymbolPeriod  = true — set file names with pattern images and parameters Symbol()+"\_"+Period()+"\_Patterns
- InfoPointD  — display data on D point of the pattern in large font size
- MonitorPatterns  — enable the pattern monitor
- TextSize  — set font size in the pattern monitor
- ExtGlobalVariableSet  — enable writing data about patterns to the terminal global variables


Information about a found pattern can be output to a global variable named "ZUP"+\_Symbol+\_Period . An example for the global variable name: ZUPGBPUSD240. If a pattern is found, the digit 1 will be written to the variable.

#### Explanation of some ZUP features implemented in Gartley pattern search modes

**Pattern Monitor**

The operation of the Pattern Monitor is demonstrated below through the example of the "Chinese toy" mode. A lot of found patterns are displayed on a chart in this mode:

![monitor patterns](https://c.mql5.com/2/31/EURGBPH4__3.png)

The Monitor is the lines in the upper left corner of the chart. Each line corresponds to one of the patterns displayed on the chart. The color of the first three columns corresponds to the color of the appropriate pattern. The color of the fourth column: blue means a Bullish pattern set by the ExtColorPatternsBullish parameter, red means a Bearish pattern set by the ExtColorPatternsBearish parameter.

The first column contains the number of the bar on which the pattern's D point is found.

The second column displays minBars/ExtBackstep parameters of the ZigZag, on which the corresponding pattern is found.

The third column shows the conditional deviation of the XC-XB-AD-CD retracements from the ideal value. Let's take a closer look at the specific example of the Bullish Butterfly pattern \[1.414/.786/.786/2.0\] in the fifth line of the Pattern Monitor. The ideal retracement values ​​are shown in square brackets after the pattern name. How are conditional deviations formed? What do conditional deviations mean?

The larger the number, the greater the deviation from the ideal value.

- Value 0: the deviation must be less than the ExtDeltaGartleyPRZ parameter value. The default parameter value is 2%.
- Value 1: the deviation must be less than the ExtDelta parameter value. The default value is 4%.
- Value 2: the deviation must be less than the ExtDeltaGartley parameter value. The default value is 9%.
- Value \* : the deviation is greater than the one specified in Value 2. The pattern found will correspond to the pattern, which would have been found by the first versions of ZUP.

**Equilibrium line and the mirror trend vector line**

The Equilibrium parameter enables the display of the equilibrium line, which passes through the pattern's XB points and two reaction lines (red dashed lines):

![equilibrium / vector of a mirror trend](https://c.mql5.com/2/31/EURGBPH4_1__7.png)

Reaction line 1 is parallel to the equilibrium line. The distance between these lines is equal to the distance between the pattern's point C and the equilibrium line.

Reaction line 2 is parallel to the equilibrium line. The distance between these lines is equal to the distance between point A of the pattern and the equilibrium line.

When the price approaches the reaction line, one can expect either correction or end of pattern formation.

The vector of the mirror trend is enabled by the VectorOfAMirrorTrend parameter. After that the drawing of the dashed line begins. The line of the mirror trend vector is pale green in the above image. It is drawn through the diagonal of the potential reversal zone, which is shown as a red rectangle at point D of the pattern.

### Part Three. Briefly about other patterns

In addition to Gartley pattern, ZUP allows working with [Pesavento patterns](https://www.mql5.com/en/articles/1470) and with other types of patterns.

One of such types includes Merrill patterns. They are used in combination with Bollinger Bands. ZUP allows identifying such patterns on an already formed ZigZag. The patterns are not displayed on the chart, while only their names are shown. Dynamic Merrill patterns are formed on the first four ZigZag rays on the right. Patterns on rays 2-5 are static.

Information about Merrill patterns is enabled using parameter infoMerrillPattern=true. Pattern information is displayed in a line in the upper left corner of the chart, in small print. The pattern name can be shown in large print by setting bigText=true. Such large label is displayed in the upper right corner.

![Merrill pattern](https://c.mql5.com/2/31/USDJPYM1.png)

A dynamic Merrill pattern called Triangle has formed on the first four rays.

A static pattern called M13 has formed on rays from second to fifth.

Merrill patterns are formed of five dots. Their graphical representation can be created using the graphical interface. For example, this has already been done by the forum participant Stanislav Trivailo.

When bigText=true, information about a found Gartley pattern can also be displayed in large print. This information and Monitor can only be displayed for the first two pattern search options.

![](https://c.mql5.com/2/31/USDJPYM1_1.png)

### Conclusions

This brief description will help you understand how to use ZUP to search for patterns.

You can download the ZUP indicator platform on the Market: [https://www.mql5.com/en/market/product/19758](https://www.mql5.com/en/market/product/19758)

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/2990](https://www.mql5.com/ru/articles/2990)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [ZUP - universal ZigZag with Pesavento patterns. Graphical interface](https://www.mql5.com/en/articles/2966)
- [ZUP - Universal ZigZag with Pesavento Patterns. Part 2](https://www.mql5.com/en/articles/1470)
- [ZUP - Universal ZigZag with Pesavento Patterns. Part 1](https://www.mql5.com/en/articles/1468)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/252441)**
(10)


![Eugeni Neumoin](https://c.mql5.com/avatar/avatar_na2.png)

**[Eugeni Neumoin](https://www.mql5.com/en/users/nen)**
\|
6 Mar 2018 at 15:17

**ys\_mql5:**

I, as I think, and many other traders, who are just starting to get acquainted with your indicator, are very interested in which zigzag algorithm is the most optimal for work. Or so - could you describe the pros/cons of each available algorithm, as I am sure that they were all added to the indicator for a reason. I realise that in some cases it is preferable to use one zigzag and in other cases another.

Maybe you could give some recommendations? Perhaps there is some more universal algorithm and its basic settings (where everyone could start), which would be suitable for a start, so that you do not have to painfully and repeatedly "dig" through the parameters, changing them and examining the chart in search of changes in the work of the indicator with the next set of settings?

PS Yes, I mean exclusively working with the indicator in the pattern search mode (ExtIndicator = 11).

Thanks.

1\. Is there an option in the indicator, which displays alphabetic designations of pattern tops in the colour of the pattern (or even better directly with the name on the chart)? It seems to me that it would be more convenient to orientate on the formed or forming pattern.

2\. What is the difference between patterns with the same name without prefix and with prefixes "A" and "max", for example - "Gartley", "A Gartley" and "max Gartley"?

3\. What is the group of patterns "TOTAL" - are they user patterns?

4\. If you use the"ExtGartleyTypeSearch \- pattern search mode" parameter in the "Chinese Toy 1" mode, then the patterns - \_http://prntscr.com/inkjnc are present on the last zigzags [,](https://www.mql5.com/go?link=http://prntscr.com/inkjnc, "http://prntscr.com/inkjnc,") and if you switch to the "0 - search ends after the first found pattern" mode, then the patterns that were in the Chinese Toy mode will not be present - \_ [http://prntscr.com/inkkfx.](https://www.mql5.com/go?link=http://prntscr.com/inkkfx. "http://prntscr.com/inkkfx.") Most likely there will be 1-2 patterns that have long since ended. Question, why in the mode of searching for the first found pattern the closest pattern to the current bar is not displayed?

In the ExtIndicator = 11 mode, the standard zigzag algorithm is used. What are patterns. The XA ray is an impulse wave. And the ABCD structure is a corrective wave. That is, in theory, patterns are part of the Elliott wave structure. More than 10 years ago Putnik researched the parameters of the standard zigzag and suggested parameter settings that do not always but often enough reveal Elliott wave patterns. Putnik specialises in wave analysis. These are his default settings. In fact, there can be many variants of settings. Talk to Putnik on this topic. He specialises in patterns.

I, as I wrote above, more often use Andrews Forks. If you tie them to the tops of the waves, the pitchforks magically tell you where the market reversal will occur long before the market gets there.

Here's a current example on the Yen:

[![](https://c.mql5.com/3/178/hxz9cxtpe1r.png)](https://c.mql5.com/3/178/o1kzeahkf40.png "https://c.mql5.com/3/178/o1kzeahkf40.png")

Look at how accurately the market bounces off the lines presented in the Andrews pitchfork set. I can't imagine that this is possible with patterns.

It's just some kind of miracle. I have an explanation for why this happens. But let's leave that topic.

Re.

1) There is no way to output letter designations of pattern vertices (XABCD).

2 and 3) The letter "A" stands for antipatterns. Look at A Gartley and Gartley for example. These as patterns and TOTAL were invented by the lieutenant. The prefix max is used for patterns that were in the first versions of ZUP, which is where the pattern search was created. These patterns used the widest range of retracements. The current version has built in, shall we say, classic Gartley and max Gartley. They differ in parameters. The differences are small.

4) For the Chinese toy mode the values of ExtGartleyTypeSearch  parameter 1 or 2 are used. In one case a one-time search for patterns is performed. In the other case, the patterns are searched at each change of the zero bar. In the limit, the search is performed on each tick, when the zero bar changes its size on each tick. This is too expensive for a metatrader.

![ys_mql5](https://c.mql5.com/avatar/avatar_na2.png)

**[ys\_mql5](https://www.mql5.com/en/users/ys_mql5)**
\|
6 Mar 2018 at 19:08

**Eugeni Neumoin:**

The ExtIndicator = 11 mode uses the standard zigzag algorithm. What are patterns. The XA ray is an impulse wave. And the ABCD structure is a correction wave. That is, in theory, patterns are part of the Elliott wave structure. More than 10 years ago Putnik researched the parameters of the standard zigzag and suggested parameter settings that do not always but often enough reveal Elliott wave patterns. Putnik specialises in wave analysis. These are his default settings. In fact, there can be many variants of settings. Talk to Putnik on this topic. He specialises in patterns.

I, as I wrote above, more often apply Andrews Forks. If they are tied to the tops of waves, the pitchforks magically suggest the places where the market reversal will occur long before the market gets there.

Here is a current example on the Yen:

Look at how accurately the market bounces off the lines presented in the Andrews pitchfork set. I don't imagine this is possible with patterns.

It's just some kind of miracle. I have an explanation for why this happens. But let's leave that topic.

For questions.

1) There is no possibility to output alphabetic designations of pattern vertices (XABCD).

2 and 3) The letter "A" stands for antipatterns. Look at A Gartley and Gartley for example. These as patterns and TOTAL were invented by the lieutenant. The prefix max is used for patterns that were in the first versions of ZUP, which is where the pattern search was created. These patterns used the widest range of retracements. The current version has built in, shall we say, classic Gartley and max Gartley. They differ in parameters. The differences are small.

4) For the Chinese toy mode the values of ExtGartleyTypeSearch  parameter 1 or 2 are used. In one case a one-time search for patterns is performed. In the other case, the patterns are searched at each change of the zero bar. In the limit, the search is performed on each tick, when the zero bar changes its size on each tick. This is too expensive for a metatrader.

Eugene, thank you for your comment.

Do you build the pitchfork manually or do you use automatic indicator building? And if so, the pitchfork algorithm uses the extrema found by the zigzag. And still we come to the point that we need to understand which zigzag was implemented for what purpose. Can you give a recommendation, which zigzag is more optimal to start with, maybe there is some special one for better work of the pitchfork algorithm?

![Eugeni Neumoin](https://c.mql5.com/avatar/avatar_na2.png)

**[Eugeni Neumoin](https://www.mql5.com/en/users/nen)**
\|
6 Mar 2018 at 19:47

**ys\_mql5:**

Eugene, thank you for your comment.

Do you build pitchforks manually or do you use automatic indicator building? And if so, the pitchfork algorithm uses the extrema found by the zigzag. And still we come to the point that we need to understand which zigzag was implemented for what purpose. Can you give a recommendation, which zigzag is the best to start with, maybe there is some special one for better work of the pitchfork building algorithm?

The standard zigzag ExtIndicator=0 is used. Default.

You can do manual wave markup with the wave zigzag. Description in the article about version 151.

Pitchforks are bound either to the extrema of the standard zigzag or to the wave zigzag.

It takes some skill. You can do it the easy way. Switch on the static pitchfork. Pstroeniya will be carried out on automatic. On the forexdengi forum, that's what everyone does. They build pitchforks on automatic.

But these pitchforks are not always useful. Understanding comes with time. It is useful to study wave analysis. For starters, you can at least watch webinars from Roman Pavelko. His webinars are broadcasted periodically by FxPro. Search on YouTube.

You can also look for Putnika's webinars on YouTube. Type DML@EWA in the search. There is a lot of information on the application of the [Andrews Fork](https://www.mql5.com/en/docs/constants/objectconstants/enum_object "MQL5 Documentation: Object Types") from ZUP there

![ys_mql5](https://c.mql5.com/avatar/avatar_na2.png)

**[ys\_mql5](https://www.mql5.com/en/users/ys_mql5)**
\|
6 Mar 2018 at 20:12

**Eugeni Neumoin:**

The default zigzag ExtIndicator=0 is used. Default.

It is possible to make manual wave markup with wave zigzag. Description in the article about version 151.

Pitchforks are bound either to the extrema of the standard zigzag or to the wave zigzag.

It takes some skill. You can do it the easy way. Switch on the static pitchfork. Pstroeniya will be carried out on automatic. That's what everyone on the forexdengi forum does. They build pitchforks on automatic.

But these pitchforks are not always useful. Understanding comes with time. It is useful to study wave analysis. For a start, you can at least watch webinars by Roman Pavelko. His webinars are broadcasted periodically by FxPro. Look for them on YouTube.

You can also look for Putnika's webinars on YouTube. Type DML@EWA in the search. There is a lot of information there on the application of the [Andrews Fork](https://www.mql5.com/en/docs/constants/objectconstants/enum_object "MQL5 Documentation: Object Types") from ZUP

Great, thanks. This will be a good starting point to learn this analysis technique.

I have looked at the market analysis by Gelox on forexdengi site, of course, you can't understand it at once (even if you understand the principles of wave analysis in general), you need to start small - with studying the basics of this approach. Besides, as I understand, there are no specific training materials on DML&EWA, as Putnik, as far as I understand, holds (or has held?) paid seminars on this topic. So, if there is information in the public domain, it is scattered in bits and pieces or only in practical application when analysing the market.

Eugene, could you recommend some good literature and/or resources on the topic of market analysis with the help of Andrews Pitchfork? I found Patrick Mikula's book "Alan Andrews' Best Trend Line Methods Plus Five New Techniques" in my stash. I think you can get a basic understanding of working with VE from there.

One more thing - what configuration for working with ZUP in MT4 is the most relevant now:

1. 9 ZUPs with different display settings on different TFs (as Igor advises in his seminar from 2013).
2. Gelox's "ZigZag Kit" posted by you - [https://www.forexdengi.com/threads/10201-zup-i-vili-endryusa-termini-ponyatiya-parametri?p=12419002&amp;](https://www.mql5.com/go?link=https://www.forexdengi.com/threads/10201-zup-i-vili-endryusa-termini-ponyatiya-parametri?p=12419002%26amp "https://www.forexdengi.com/threads/10201-zup-i-vili-endryusa-termini-ponyatiya-parametri?p=12419002&amp;amp") viewfull=1#post12419002
3. Or, judging by Gelox's most relevant post from 03.03.2018 - setting up 2 ZUP sets in 3 windows with different TFs and different minBars and ExtBackstep settings - [https://www.forexdengi.com/threads/84088-nastroyka-vil-endryusa-v-indikatornoy-platforme-zupv150-s-nen-v-kartinkah-ot-sledopyt?p=18037607&amp;](https://www.mql5.com/go?link=https://www.forexdengi.com/threads/84088-nastroyka-vil-endryusa-v-indikatornoy-platforme-zupv150-s-nen-v-kartinkah-ot-sledopyt?p=18037607%26amp "https://www.forexdengi.com/threads/84088-nastroyka-vil-endryusa-v-indikatornoy-platforme-zupv150-s-nen-v-kartinkah-ot-sledopyt?p=18037607&amp;amp") viewfull=1#post18037607

![Warrior of the LORD](https://c.mql5.com/avatar/2017/6/59561592-340F.jpeg)

**[Warrior of the LORD](https://www.mql5.com/en/users/80147047)**
\|
30 Jun 2018 at 16:39

Mark!


![Developing multi-module Expert Advisors](https://c.mql5.com/2/26/4990806_hdlx-Gear-cgy8j3o-gb3o7dl-sbeht.png)[Developing multi-module Expert Advisors](https://www.mql5.com/en/articles/3133)

MQL programming language allows implementing the concept of modular development of trading strategies. The article shows an example of developing a multi-module Expert Advisor consisting of separately compiled file modules.

![Synchronizing several same-symbol charts on different timeframes](https://c.mql5.com/2/31/6cd68idtz6fac-lu770iwbwo-3ndzmpk7.png)[Synchronizing several same-symbol charts on different timeframes](https://www.mql5.com/en/articles/4465)

When making trading decisions, we often have to analyze charts on several timeframes. At the same time, these charts often contain graphical objects. Applying the same objects to all charts is inconvenient. In this article, I propose to automate cloning of objects to be displayed on charts.

![Random Decision Forest in Reinforcement learning](https://c.mql5.com/2/31/family-eco.png)[Random Decision Forest in Reinforcement learning](https://www.mql5.com/en/articles/3856)

Random Forest (RF) with the use of bagging is one of the most powerful machine learning methods, which is slightly inferior to gradient boosting. This article attempts to develop a self-learning trading system that makes decisions based on the experience gained from interaction with the market.

![Multi-symbol balance graph in MetaTrader 5](https://c.mql5.com/2/31/MultiSymbol.png)[Multi-symbol balance graph in MetaTrader 5](https://www.mql5.com/en/articles/4430)

The article provides an example of an MQL application with its graphical interface featuring multi-symbol balance and deposit drawdown graphs based on the last test results.

[![](https://www.mql5.com/ff/si/m0dtjf9x3brdz07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fmarket%2Fmt5%2Fexpert%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dtop.experts%26utm_content%3Dbuy.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=widauvjabtsckwovwaperzkotrcrttvb&s=25ef75d39331f608a319410bf27ff02c1bd7986622ecc1eec8968a650f044731&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=jnubeipkbtxqflpabhpkqadgefzpqifa&ssn=1769193755240402409&ssn_dr=0&ssn_sr=0&fv_date=1769193755&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F2990&back_ref=https%3A%2F%2Fwww.google.com%2F&title=ZUP%20-%20Universal%20ZigZag%20with%20Pesavento%20patterns.%20Search%20for%20patterns%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176919375535664896&fz_uniq=5072034898623017754&sv=2552)

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