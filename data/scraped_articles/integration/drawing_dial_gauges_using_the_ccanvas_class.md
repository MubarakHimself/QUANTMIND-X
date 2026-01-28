---
title: Drawing Dial Gauges Using the CCanvas Class
url: https://www.mql5.com/en/articles/1699
categories: Integration, Indicators
relevance_score: 0
scraped_at: 2026-01-24T14:06:56.904771
---

[![](https://www.mql5.com/ff/si/3fgkjn78mkxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=ocndbzpeklfncxysjbwfhhbalbrsdbtv&s=a4309643278437a00bdd33c5809fc6b4b4032749c00fccd07b3b84e7b8b45126&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=jtrfqxxkntjdfubrtaizwtcdimkmlwrs&ssn=1769252815787656121&ssn_dr=0&ssn_sr=0&fv_date=1769252815&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F1699&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Drawing%20Dial%20Gauges%20Using%20the%20CCanvas%20Class%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176925281521766433&fz_uniq=5083344870363371982&sv=2552)

MetaTrader 5 / Examples


### Table Of Contents

- [Introduction](https://www.mql5.com/en/articles/1699#ppi)
- [1\. Coordinates and Anchor](https://www.mql5.com/en/articles/1699#pp0100)
- [2\. Gauge Elements](https://www.mql5.com/en/articles/1699#pp0200)
  - [2.1. Sizes](https://www.mql5.com/en/articles/1699#pp0201)
  - [2.2. Body Shape](https://www.mql5.com/en/articles/1699#pp0202)
  - [2.3. Scale](https://www.mql5.com/en/articles/1699#pp0203)
  - [2.4. Graduation Marks](https://www.mql5.com/en/articles/1699#pp0204)
  - [2.5. Legends](https://www.mql5.com/en/articles/1699#pp0205)
  - [2.6. Highlighted Ranges](https://www.mql5.com/en/articles/1699#pp0206)
  - [2.7. Needle](https://www.mql5.com/en/articles/1699#pp0207)
- [3\. Functions](https://www.mql5.com/en/articles/1699#functions)


  - [3.1. GaugeCreate](https://www.mql5.com/en/articles/1699#fgc)
  - [3.2. GaugeSetCaseParameters](https://www.mql5.com/en/articles/1699#fgscp)
  - [3.3. GaugeSetScaleParameters](https://www.mql5.com/en/articles/1699#fgssp)
  - [3.4. GaugeSetMarkParameters](https://www.mql5.com/en/articles/1699#fgsmp)
  - [3.5. GaugeSetMarkLabelFont](https://www.mql5.com/en/articles/1699#fgsmlf)
  - [3.6. GaugeSetLegendParameters](https://www.mql5.com/en/articles/1699#fgslp)
  - [3.7. GaugeSetRangeParameters](https://www.mql5.com/en/articles/1699#fgsrp)
  - [3.8. GaugeSetNeedleParameters](https://www.mql5.com/en/articles/1699#fgsnp)
  - [3.9. GaugeRedraw](https://www.mql5.com/en/articles/1699#fgr)
  - [3.10. GaugeNewValue](https://www.mql5.com/en/articles/1699#fgnv)
  - [3.11. GaugeDelete](https://www.mql5.com/en/articles/1699#fgd)
  - [3.12. GaugeCalcLocation](https://www.mql5.com/en/articles/1699#fgcl)
  - [3.13. GaugeRelocation](https://www.mql5.com/en/articles/1699#fgrlc)

- [4\. Enumerations](https://www.mql5.com/en/articles/1699#enums)

  - [4.1. ENUM\_CASE\_BORDER\_STYLE](https://www.mql5.com/en/articles/1699#encbs)
  - [4.2. ENUM\_CASE\_STYLE](https://www.mql5.com/en/articles/1699#encs)
  - [4.3. ENUM\_GAUGE\_LEGEND](https://www.mql5.com/en/articles/1699#engl)
  - [4.4. ENUM\_MARK\_STYLE](https://www.mql5.com/en/articles/1699#enmst)
  - [4.5. ENUM\_MUL\_SCALE](https://www.mql5.com/en/articles/1699#enmsc)
  - [4.6. ENUM\_NCENTER\_STYLE](https://www.mql5.com/en/articles/1699#enns)
  - [4.7. ENUM\_NEEDLE\_FILL](https://www.mql5.com/en/articles/1699#ennf)
  - [4.8. ENUM\_REL\_MODE](https://www.mql5.com/en/articles/1699#enrm)
  - [4.9. ENUM\_SCALE\_STYLE](https://www.mql5.com/en/articles/1699#enss)
  - [4.10. ENUM\_SIZE](https://www.mql5.com/en/articles/1699#ensz)

- [5\. Macro Substitution](https://www.mql5.com/en/articles/1699#macro)
- [6\. Modifying the CCanvas Class](https://www.mql5.com/en/articles/1699#pp0600)

  - [6.1. Drawing a Segment Using Antialiasing Algorithm](https://www.mql5.com/en/articles/1699#pp0601)
  - [6.2. Filling an Area with Antialiased Edges](https://www.mql5.com/en/articles/1699#pp0602)

- [7\. Application Examples](https://www.mql5.com/en/articles/1699#pp0700)

  - [7.1. Indicator of Current Profit](https://www.mql5.com/en/articles/1699#pp0701)
  - [7.2. The Dashboard Indicator](https://www.mql5.com/en/articles/1699#pp0702)

- [8\. Resource Intensity Assessment](https://www.mql5.com/en/articles/1699#pp0800)
- [Conclusion](https://www.mql5.com/en/articles/1699#ppe)

### Introduction

It all started when I first acquainted myself with the [CCanvas](https://www.mql5.com/en/docs/standardlibrary/canvasgraphics/ccanvas) class. When it came to practice, I stumbled upon an idea to draw a gauge indicator. My first gauges were pretty crude, but eventually they have been supplemented by new elements and become visually pleasing. And as a result, I have a small library now which allows to add a dial gauge to an indicator or an EA in a simple and easy manner. In this article, we will give consideration to structure of gauges, get acquainted with functions necessary for drawing and setting visual appearance, and assess resource intensity.

![Dial gauges](https://c.mql5.com/2/18/dashboard_03.png)

Fig.1. Dial gauges

### 1\. Coordinates and Anchor

There are two types of positioning a gauge on a chart: absolute and relative.

In case of _absolute_ positioning, coordinates represent distances in pixels from an anchor corner along X and Y axis.

In case of _relative_ positioning, local origin of coordinates is created according to the specified type of the relative positioning. When the vertical type is selected, the origin is located below or above a reference object (if an upper or a lower anchor corner is selected respectively). When the vertical type is selected, it is located on the left or on the right in the direction from the anchor corner. In this case, specified coordinates represent an offset from their local origin. Positive offsets lead to moving an object away from the reference object. In case of negative offsets, the object will encroach on the reference one.

The reference object can be represented only by an object of another gauge. It is essential that both objects will have the same anchor corner.

Fig. 2 depicts an example of relative positioning.

![Relative positioning](https://c.mql5.com/2/18/gg_rel_location.png)

Fig.2. Relative positioning of gauges

Let's review settings of each gauge:

- The "gg01" gauge: relative positioning is disabled. Horizontal offset — 40, vertical offset — 40.
- The "gg02" gauge: relative positioning — horizontal, reference object — "gg01". Horizontal offset from the local origin of coordinates (point A) — 15, vertical offset — 0.
- The "gg03" gauge: relative positioning — vertical, reference object — "gg01". Horizontal offset from the local origin of coordinates (point B) — 0, vertical offset — 15.
- The "gg04" gauge: relative positioning — vertical, reference object — "gg02". Horizontal offset from the local origin of coordinates (point C) — 50, vertical offset — 15.

Relative positioning facilitates input setting if the chart has several indicators containing gauges. If you decide to change size of one gauge, other gauges' coordinates will be automatically recalculated.

The [GaugeCreate()](https://www.mql5.com/en/articles/1699#fgc) function sets the positioning type and coordinates.

### 2\. Gauge Elements

The dial gauge consists of two graphical objects. One of them is called a _scale layer_, another one is called a _needle layer_. Both graphical objects have the same coordinates. The needle layer is placed over the scale layer. The gauge name set in input parameters serves as a prefix for names of both objects. For example, if the gauge name is "Gauge01", the scale layer will be called "Gauge01\_s", and the needle layer will have name "Gauge01\_n".

Fig.3 depicts structure of the gauge.

![Fig.3. Gauge structure](https://c.mql5.com/2/19/fig.3.measurements.png)

Fig.3. Gauge structure

_**The scale layer**_ contains:

- border (1)

- scale graduation marks (5, 6, 7)
- scale graduation labels (4)
- highlighted ranges (2, 12)
- legends (3, 10, 11)

Legends are distinguished by purposes:

- gauge description (3)
- units of measure (11)
- current value (10)
- multiplier of scale labels (omitted)

Scale graduation is divided into:

- major (7)
- middle (5)
- minor (6)

Only major graduation points have labels. Graduation step is set as a numeric value. Middle graduation step is calculated depending on a specified number of middle marks between major ones. Minor graduation step is calculated depending on a specified number of minor marks between middle ones. Minor and middle graduation can be omitted.

_**The needle layer**_ contains:

- needle (8)
- needle center (9)

**2.1. Sizes**

Fig.3 depicts sizes of some gauge elements:

- d — gauge size which corresponds to the diameter of the external contour line of the gauge
- b — border size
- g — size of space between a border and scale elements
- c — size of the needle center.

**NB**. The gauge diameter is the only size set in pixels ("d" in fig.3). All other elements and fonts are set in conditional units and their sizes are calculated as percentage of the diameter. It is made to facilitate scaling. Change the diameter, and all other sizes will be proportionally recalculated. Calculation coefficients are listed in the [Macro Substitution](https://www.mql5.com/en/articles/1699#macro) section and can be modified by the user.

**2.2. Body Shape**

There are two types of gauge body shape: a circle and a sector. The sector shape is more convenient if the [scale range](https://www.mql5.com/en/articles/1699#determ_scrng) angle is less than 180 degrees.

![Gauge shape](https://c.mql5.com/2/18/g_cases__2.png)

Fig.4. Gauge shape

Fig.4 depicts one circle gauge (a) and two sector shape gauges (b, c). The [GaugeSetCaseParameters()](https://www.mql5.com/en/articles/1699#fgscp) function is used to set the desired body shape.

**2.3. Scale**

This is the most important element of the gauge. Data readability depends on its appearance. The scale should not be overcomplicated, but at the same time it must be informative enough. Selection of scale extreme values, as well as a step of major marks, call for special attention. The [GaugeSetScaleParameters()](https://www.mql5.com/en/articles/1699#fgssp) function allows to set the scale range, its rotation and extreme values (minimum and maximum). Minimum value can be on the left (direct order) or on the right (inverse order).

**_The scale range_** is an angle contained by two radius vectors of scale extreme values. It is demonstrated in Fig.5.

![Scale range](https://c.mql5.com/2/18/gg_scale_range2.png)

Fig.5. Scale range

_**The scale rotation**_ is an angle of deviation of the scale range angle bisector from the line which upwardly and vertically comes from the gauge center. It is demonstrated in Fig.6.

![Scale rotation angle](https://c.mql5.com/2/18/gg_scale_rotation2.png)

Fig.6. Scale rotation angle

Combining the scale range angle and the rotation angle can help you to set the gauge appearance in a quite flexible manner. Fig.4(c) demonstrates a gauge with 90 degree range and 45 degree rotation.

_**Maximum and minimum scale values**_ are important parameters which should be selected depending on the range of allowed values of the displayed variable. Zero mark can be omitted for the sake of convenience. There is no point in drawing the scale from zero if your variable changes in the range from 400 to 600. Fig.7 depicts some examples of maximum and minimum scale values.

![Maximum and minimum scale values](https://c.mql5.com/2/18/g_min_max__1.png)

Fig.7. Maximum and minimum scale values

- a) values from 0 to 500, direct order
- b) values from -200 to 400, direct order
- c) values from -400 to 0, direct order
- d) values from 500 to 0, inverse order
- e) values from 200 to 800, direct order
- f) values from 0 to -800, inverse order

**2.4. Graduation Marks**

Graduation mark setting lies in selecting size of marks and aligning method.

Alignment can be as follows:

- inner edge of the scale
- outer edge of the scale
- center

Fig.8 depicts examples of aligning scale graduation marks:

- a — center
- b — inner edge
- c — outer edge

The [GaugeSetMarkParameters()](https://www.mql5.com/en/articles/1699#fgsmp) function is used for setting.

Position of labels for marks is referred to scale settings and adjusted using the [GaugeSetScaleParameters()](https://www.mql5.com/en/articles/1699#fgssp) function.

Fig.8(a) depicts an example of positioning labels inside the scale, Fig.8(b) and 8(c) — outside the scale.

It is recommended to use a _**multiplier**_, a coefficient all displayed values will be divided by, so the labels won't occupy too much space on the scale. The multiplier can have values from 0.0001 to 10000. Fig.4(c) depicts an example of applying a multiplier equal to 100, which allowed to use one-digit numbers instead of three-digit numbers in labels. Fig.1 depicts a situation where we use a multiplier equal to 0.0001 for ATR, which allowed not to use the decimal point and zeros in labels. The [GaugeSetScaleParameters()](https://www.mql5.com/en/articles/1699#fgssp) function sets the multiplier.

![Positioning marks and labels](https://c.mql5.com/2/18/g_scale_io__1.png)

Fig.8. Positioning marks and labels

**2.5. Legends**

Legends are meant for displaying supplemental information and can be of four types:

- gauge description
- units of measure
- current value
- multiplier

Any legend can be hidden. Only the gauge description is displayed by default.

Legend positioning is set by the angle and the radius. The angle is set in degrees and its value is equal to the angle between the line, which upwardly and vertically comes from the gauge center, and an imaginary segment, which connects the gauge center and the legend center. The radius is set in conditional units. It can have values from 0 to 10, where 0 corresponds to the radius of the needle center and 10 corresponds to the outer radius of the scale.

Fig.9 depicts an example of legend positioning.

- The "Profit" legend (gauge description) has following coordinates: angle - 0, radius - 3.
- The "0.00" legend (current value) has following coordinates: angle - 225, radius - 4.
- The "USD" legend (units of measure) has following coordinates: angle - 215, radius - 8.

The [GaugeSetLegendParameters()](https://www.mql5.com/en/articles/1699#fgslp) function is used for setting the legend parameters.

![Legend coordinates](https://c.mql5.com/2/18/gauge_labels4.png)

Fig.9. Legend coordinates

**NB.** Legends are not fixed on the scale and their angles are not connected with the scale rotation angle.

**2.6. Highlighted Ranges**

Highlighted data ranges represent an inherent element of any gauge. They help to see that the variable has taken on an emergency value or entered some special range. The [GaugeSetRangeParameters()](https://www.mql5.com/en/articles/1699#fgsrp) function can set up to four highlighted ranges. To do so, you need to set extreme values and color for highlighting. Fig.1 depicts the Profit indicator with two highlighted ranges: from 200 to 400 is the green range, which indicates time for fixing profit, and from -200 to -400 is the orange range, warning about large drawdown.

**2.7. Needle**

The [GaugeSetNeedleParameters()](https://www.mql5.com/en/articles/1699#fgsnp) function sets the size of the needle center and the type of area fill. The type of area fill influences on resource intensity of the indicator, as the needle layer is fully redrawn every time after data update. Fig.10 depicts example of area fill.

- filled needle with the use of antialiasing algorithm (a)
- filled needle without the use of antialiasing algorithm (b)
- not filled needle with the antialiased contour line (c)

![Methods of needle area fill](https://c.mql5.com/2/18/g_needles__1.png)

Fig.10. Methods of needle area fill

Pros and cons of each method are described in sections devoted to modification of the [CCanvas](https://www.mql5.com/en/articles/1699#pp0602) class and [resource intensity assessment](https://www.mql5.com/en/articles/1699#pp0800).

### 3\. Functions

Table 1 lists functions for drawing gauges and setting their appearance.

| Function | Behavior |
| --- | --- |
| [GaugeCalcLocation](https://www.mql5.com/en/articles/1699#fgcl) | Calculates gauge center coordinates |
| [GaugeCreate](https://www.mql5.com/en/articles/1699#fgc) | Creates the gauge |
| [GaugeDelete](https://www.mql5.com/en/articles/1699#fgd) | Deletes the gauge |
| [GaugeNewValue](https://www.mql5.com/en/articles/1699#fgnv) | Updates position of the needle and displayed value |
| [GaugeRedraw](https://www.mql5.com/en/articles/1699#fgr) | Redraws the gauge |
| [GaugeRelocation](https://www.mql5.com/en/articles/1699#fgrlc) | Changes location of gauge objects on the chart |
| [GaugeSetCaseParameters](https://www.mql5.com/en/articles/1699#fgscp) | Sets gauge body parameters |
| [GaugeSetLegendParameters](https://www.mql5.com/en/articles/1699#fgslp) | Sets legend parameters |
| [GaugeSetMarkLabelFont](https://www.mql5.com/en/articles/1699#fgsmlf) | Sets font of graduation labels |
| [GaugeSetMarkParameters](https://www.mql5.com/en/articles/1699#fgsmp) | Sets scale graduation parameters |
| [GaugeSetNeedleParameters](https://www.mql5.com/en/articles/1699#fgsnp) | Sets needle parameters |
| [GaugeSetRangeParameters](https://www.mql5.com/en/articles/1699#fgsrp) | Sets range parameters |
| [GaugeSetScaleParameters](https://www.mql5.com/en/articles/1699#fgssp) | Sets scale parameters |

Table 1. List of functions

Let's consider each function in depth. They are represented in the order in which we recommend to call them when initializing.

**3.1. GaugeCreate**

Creates the gauge.

```
bool GaugeCreate(
   string name,              // gauge name
   GAUGE_STR &g,             // reference to the gauge structure
   int x,                    // horizontal indent from the anchor corner
   int y,                    // vertical indent from the anchor corner
   int size,                 // gauge size
   string ObjRel,            // name of a graphical object relatively to which the position is set
   ENUM_REL_MODE rel_mode,   // relative positioning
   ENUM_BASE_CORNER corner,  // anchor corner
   bool back,                // objects on the background
   uchar scale_transparency, // scale transparency
   uchar needle_transparency // needle transparency
 );
```

**Parameters**

_name_

\[in\]  Gauge name. Used as a prefix for names of graphical objects which compose the gauge.

_g_

\[out\]  Reference to the gauge structure.

_x_

\[in\]  Distance in pixels from the anchor corner along the X axis. In case of relative positioning — distance from the local origin of coordinates.

_y_

\[in\]  Distance in pixels from the anchor corner along the Y axis. In case of relative positioning — distance from the local origin of coordinates.

_size_

\[in\]  [Size](https://www.mql5.com/en/articles/1699#pp0201) of the gauge. Represented as the body diameter.

_ObjRel_

\[in\]  Name of another graphical object relatively to which the position is set. Remains pertinent only if relative positioning is set.

_rel\_mode_

\[in\]  Method of [relative positioning](https://www.mql5.com/en/articles/1699#pp0100). Can have any value listed in [ENUM\_REL\_MODE](https://www.mql5.com/en/articles/1699#enrm).

_corner_

\[in\]  Chart corner to anchor the graphical object. Can have any value listed in [ENUM\_BASE\_CORNER](https://www.mql5.com/en/docs/constants/objectconstants/enum_basecorner).

_back_

\[in\]  Objects on the background.

_scale\_transparency_

\[in\]  Scale transparency level. Can have values from 0 to 255.

_needle\_transparency_

\[in\]  Needle transparency level. Can have values from 0 to 255.

**Return value**

Returns true if objects of the scale layer and the needle layer have been created. Otherwise returns false.

**3.2. GaugeSetCaseParameters**

Sets gauge body parameters.

```
void GaugeSetCaseParameters(
   GAUGE_STR &g,                  // reference to the gauge structure
   ENUM_CASE_STYLE style,         // body style
   color ccol,                    // body color
   ENUM_CASE_BORDER_STYLE bstyle, // border style
   color bcol,                    // border color
   ENUM_SIZE border_gap_size      // size of space between a border and scale elements
);
```

**Parameters**

_g_

\[out\]  Reference to the gauge structure.

_style_

\[in\]  Body style. Can have any value listed in [ENUM\_CASE\_STYLE](https://www.mql5.com/en/articles/1699#encs).

_ccol_

\[in\]  Body color.

_bstyle_

\[in\]  Border style. Can have any value listed in [ENUM\_CASE\_BORDER\_STYLE](https://www.mql5.com/en/articles/1699#encbs).

_bcol_

\[in\]  Border color.

_gap\_size_

\[in\]  Area between the internal line of the border and the nearest scale element ("g" in fig.3). Can have any value listed in [ENUM\_SIZE](https://www.mql5.com/en/articles/1699#ensz).

**3.3. GaugeSetScaleParameters**

Sets scale parameters.

```
void GaugeSetScaleParameters(
   GAUGE_STR &g,           // reference to the gauge structure
   int range,              // scale range
   int rotation,           // angle of rotation
   double min,             // minimum value (left)
   double max,             // maximum value (right)
   ENUM_MUL_SCALE mul,     // multiplier of scale labels
   ENUM_SCALE_STYLE style, // scale style
   color col,              // scale color
   bool display_arc        // display scale line
);
```

**Parameters**

_g_

\[out\]  Reference to the gauge structure.

_range_

\[in\]  [Scale range](https://www.mql5.com/en/articles/1699#determ_scrng). Set as an angle contained by two radius vectors of scale extreme marks. Can have values from 30 to 320 degrees (Fig.5).

_rotation_

\[in\]  [Scale rotation angle](https://www.mql5.com/en/articles/1699#determ_scrot) (Fig.6).

_min_

\[in\]  Minimum scale value in case of direct number assignment.

_max_

\[in\]  Maximum scale value in case of direct number assignment.

_mul_

\[in\]  Multiplier of scale labels. Can have any value listed in [ENUM\_MUL\_SCALE](https://www.mql5.com/en/articles/1699#enmsc).

_style_

\[in\]  Scale style. Can have any value listed in [ENUM\_SCALE\_STYLE](https://www.mql5.com/en/articles/1699#enss).

_col_

\[in\]  Scale color.

_display\_arc=false_

\[in\]  Display scale line.

**3.4. GaugeSetMarkParameters**

Sets scale graduation parameters.

```
void GaugeSetMarkParameters(
   GAUGE_STR &g,          // reference to the gauge structure
   ENUM_MARK_STYLE style, // style of scale marks
   ENUM_SIZE size,        // size of marks
   double major_tmi,      // major mark interval
   int middle_tmarks,     // number of middle marks between major ones
   int minor_tmarks       // number of minor marks between middle ones
);
```

**Parameters**

_g_

\[out\]  Reference to the gauge structure.

_style_

\[in\]  Style of scale graduation. Can have any value listed in [ENUM\_MARK\_STYLE](https://www.mql5.com/en/articles/1699#enmst).

_size_

\[in\]  Mark size. Can have any value listed in [ENUM\_SIZE](https://www.mql5.com/en/articles/1699#ensz).

_major\_tmi_

\[in\]  Step of major graduation marks. Major marks are accompanied by labels with relevant values.

_middle\_tmarks_

\[in\]  Number of middle marks between major ones. Can have any positive value. No size constraints. If set to 0, middle marks are not displayed.

_minor\_tmarks_

\[in\]  Number of minor marks between middle ones (or between major marks if middle ones are not displayed). Can have any positive value. No size constraints. If set to 0, minor marks are not displayed.

**3.5. GaugeSetMarkLabelFont**

Sets font of graduation marks.

```
void GaugeSetMarkLabelFont(
   GAUGE_STR &g,        // reference to the gauge structure
   ENUM_SIZE font_size, // font size
   string font,         // font
   bool italic,         // italic
   bool bold,           // bold
   color col            // color
);
```

**Parameters**

_g_

\[out\]  Reference to the gauge structure.

_font\_size_

\[in\]  Font size of graduation labels. Can have any value listed in [ENUM\_SIZE](https://www.mql5.com/en/articles/1699#ensz).

_font_

\[in\]  Font.

_italic_

\[in\]  Italic.

_bold_

\[in\]  Bold.

_col_

\[in\]  Font color.

**3.6. GaugeSetLegendParameters**

Sets legend parameters.

```
void GaugeSetLegendParameters(
   GAUGE_STR &g,         // reference to the gauge structure
   ENUM_GAUGE_LEGEND gl, // legend type
   bool enable,          // display legend
   string str,           // string (or a complementary parameter)
   int radius,           // coordinates - radius
   double angle,         // coordinates - angle
   uint font_size,       // font size
   string font,          // font
   bool italic,          // italic
   bool bold,            // bold
   color col             // color
);
```

**Parameters**

_g_

\[out\]  Reference to the gauge structure

_gl_

\[in\]  Legend type. Can have any value listed in [ENUM\_GAUGE\_LEGEND](https://www.mql5.com/en/articles/1699#engl).

_enable_

\[in\]  Display the legend.

_str_

\[in\]  This is a displayed string for legends of LEGEND\_DESCRIPTION or LEGEND\_UNITS type. This parameter is ignored for a legend of the LEGEND\_MUL type. Number of decimal places for a legend of the LEGEND\_VALUE type. Can have values from "0" to "8". Any other values are perceived as "0". For example, the string "2" means two decimal places. The string "hello" means 0 decimal places.

_radius_

\[in\]  Radius. Distance from the gauge center to the legend center in conditional units (fig. 9).

_angle_

\[in\]  Angular coordinates. Its value is equal to the angle between the line, which upwardly and vertically comes from the gauge center, and an imaginary segment, which connects the gauge center and the legend center (Fig.9).

_font\_size_

\[in\]  Legend font size.

_font_

\[in\]  Font.

_italic_

\[in\]  Italic.

_bold_

\[in\]  Bold.

_col_

\[in\]  Font color.

**3.7. GaugeSetRangeParameters**

Sets highlighted range parameters.

```
void GaugeSetRangeParameters(
   GAUGE_STR &g, // reference to the gauge structure
   int index,    // range index
   bool enable,  // display range
   double start, // initial value
   double end,   // final value
   color col     // color
);
```

**Parameters**

_g_

\[out\]  Reference to the gauge structure.

_index_

\[in\]  Range index. Can have values from 0 to 3.

_enable_

\[in\]  Display range.

_start_

\[in\]  Initial value.

_end_

\[in\]  Final value.

_col_

\[in\]  Color to highlight the range.

**3.8. GaugeSetNeedleParameters**

Sets needle parameters.

```
void GaugeSetNeedleParameters(
   GAUGE_STR &g,                     // reference to the gauge structure
   ENUM_NCENTER_STYLE ncenter_style, // needle center style
   color ncenter_col,                // needle center color
   color needle_col,                 // needle color
   ENUM_NEEDLE_FILL needle_fill      // method of needle area fill
);
```

**Parameters**

_g_

\[out\]  Reference to the gauge structure.

_ncenter\_style_

\[in\]  Style of the needle center. Can have any value listed in [ENUM\_NCENTER\_STYLE](https://www.mql5.com/en/articles/1699#enns).

_ncenter\_col_

\[in\]  Needle center color.

_needle\_col_

\[in\]  Needle color.

_needle\_fill_

\[in\]  Method of needle area fill. Can have any value listed in [ENUM\_NEEDLE\_FILL](https://www.mql5.com/en/articles/1699#ennf).

**3.9. GaugeRedraw**

Redraws the gauge. The function has to be called after changing any parameters to apply these changes.

```
void GaugeRedraw(
   GAUGE_STR &g       // reference to the gauge structure
);
```

**Parameters**

_g_

\[in\]  Reference to the gauge structure.

**3.10. GaugeNewValue**

Updates position of the needle and displayed value.

```
void GaugeNewValue(
   GAUGE_STR &g,     // reference to the gauge structure
   double v          // variable value
);
```

**Parameters**

_g_

\[in\]  Reference to the gauge structure.

_v_

\[in\]  Current value of variable.

**3.11. GaugeDelete**

Deletes graphical objects which compose the gauge. Call the function from the [OnDeinit()](https://www.mql5.com/en/docs/basis/function/events#ondeinit) handler.

```
void GaugeDelete(
   GAUGE_STR &g      // reference to the gauge structure
);
```

**Parameters**

_g_

\[in\]  Reference to the gauge structure.

**3.12. GaugeCalcLocation**

Calculates coordinates of gauge objects. If [relative positioning](https://www.mql5.com/en/articles/1699#pp0100) is disabled, it will always return same coordinates. Otherwise coordinates may differ from previous values if the reference object has changed its location or size.

```
bool GaugeCalcLocation(
   GAUGE_STR& g         // reference to the gauge structure
);
```

**Parameters**

_g_

\[in\]  Reference to the gauge structure.

**Return value**

Returns true if coordinate values differ from previous ones. Otherwise returns false. If the function returns true, call the [GaugeRelocation()](https://www.mql5.com/en/articles/1699#fgrlc) function to apply calculated parameters.

**3.13. GaugeRelocation**

Locates graphical objects which compose the gauge on the specified spot of the chart. Necessary to call if relative positioning is set and the [GaugeCalcLocation()](https://www.mql5.com/en/articles/1699#fgcl) function has returned true.

```
void GaugeRelocation(
   GAUGE_STR &g       // reference to the gauge structure
);
```

**Parameters**

g

\[in\]  Reference to the gauge structure.

### 4\. Enumerations

Table 2 lists enumerations used as function parameters.

| Enumeration | Description |
| --- | --- |
| [ENUM\_CASE\_BORDER\_STYLE](https://www.mql5.com/en/articles/1699#encbs) | Border style |
| [ENUM\_CASE\_STYLE](https://www.mql5.com/en/articles/1699#encs) | Body style |
| [ENUM\_GAUGE\_LEGEND](https://www.mql5.com/en/articles/1699#engl) | Legend type |
| [ENUM\_MARK\_STYLE](https://www.mql5.com/en/articles/1699#enmst) | Style of scale graduation |
| [ENUM\_MUL\_SCALE](https://www.mql5.com/en/articles/1699#enmsc) | Multiplier of scale graduation labels |
| [ENUM\_NCENTER\_STYLE](https://www.mql5.com/en/articles/1699#enns) | Style of the needle center |
| [ENUM\_NEEDLE\_FILL](https://www.mql5.com/en/articles/1699#ennf) | Method of needle area fill |
| [ENUM\_REL\_MODE](https://www.mql5.com/en/articles/1699#enrm) | Method of relative positioning |
| [ENUM\_SCALE\_STYLE](https://www.mql5.com/en/articles/1699#enss) | Scale style |
| [ENUM\_SIZE](https://www.mql5.com/en/articles/1699#ensz) | Size |

Table 2. List of enumerations

**4.1. ENUM\_CASE\_BORDER\_STYLE**

Border style. Values are listed in table 3.

| Identifier | Description |
| --- | --- |
| CASE\_BORDER\_NONE | No border |
| CASE\_BORDER\_THIN | Thin border |
| CASE\_BORDER\_THICK | Thick border |

Table 3. Values of ENUM\_CASE\_BORDER\_STYLE

**4.2. ENUM\_CASE\_STYLE**

Body style. Values are listed in table 4.

| Identifier | Description |
| --- | --- |
| CASE\_ROUND | Circular body |
| CASE\_SECTOR | Sector-type body |

Table 4. Values of ENUM\_CASE\_STYLE

**4.3. ENUM\_GAUGE\_LEGEND**

Legend type. Values are listed in table 5.

| Identifier | Description |
| --- | --- |
| LEGEND\_DESCRIPTION | Gauge description |
| LEGEND\_UNITS | Units of measure |
| LEGEND\_MUL | Multiplier of scale labels |
| LEGEND\_VALUE | Current value of variable |

Table 5. Values of ENUM\_GAUGE\_LEGEND

**4.4. ENUM\_MARK\_STYLE**

Style of scale graduation. Values are listed in table 6.

| Identifier | Description |
| --- | --- |
| MARKS\_INNER | Aligning marks by the inner edge |
| MARKS\_MIDDLE | Aligning marks by the center |
| MARKS\_OUTER | Aligning marks by the outer edge |

Table 6. Values of ENUM\_MARK\_STYLE

**4.5. ENUM\_MUL\_SCALE**

Multiplier of scale graduation labels. Values are listed in table 7.

| Identifier | Meaning | Display |
| --- | --- | --- |
| MUL\_10000 | 10000 | х10k |
| MUL\_1000 | 1000 | х1k |
| MUL\_100 | 100 | х100 |
| MUL\_10 | 10 | х10 |
| MUL\_1 | 1 | Not displayed |
| MUL\_01 | 0.1 | /10 |
| MUL\_001 | 0.01 | /100 |
| MUL\_0001 | 0.001 | /1k |
| MUL\_00001 | 0.0001 | /10k |

Table 7. Values of ENUM\_MUL\_SCALE

**4.6. ENUM\_NCENTER\_STYLE**

Style of the needle center. Values are listed in table 8.

| Identifier | Description |
| --- | --- |
| NDCS\_NONE | Don't display the needle center |
| NDCS\_SMALL | Display small |
| NDCS\_LARGE | Display large |

Table 8. Values of ENUM\_NCENTER\_STYLE

**4.7. ENUM\_NEEDLE\_FILL**

Method of needle area fill. Values are listed in table 9.

| Identifier | Description |
| --- | --- |
| NEEDLE\_FILL | Fill the needle without antialiasing of edges |
| NEEDLE\_FILL\_AA | Fill the needle with antialiasing of edges |
| NEEDLE\_NOFILL\_AA | Don't fill the needle but apply antialiasing of edges |

Table 9. Values of ENUM\_NEEDLE\_FILL

**4.8. ENUM\_REL\_MODE**

Method of relative positioning. Values are listed in table 10.

| Identifier | Description |
| --- | --- |
| RELATIVE\_MODE\_NONE | Relative positioning is disabled |
| RELATIVE\_MODE\_HOR | Horizontally |
| RELATIVE\_MODE\_VERT | Vertically |
| RELATIVE\_MODE\_DIAG | Diagonally |

Table 10. Values of ENUM\_REL\_MODE

**4.9. ENUM\_SCALE\_STYLE**

Scale style. Values are listed in table 11.

| Identifier | Description |
| --- | --- |
| SCALE\_INNER | Graduation labels inside the scale |
| SCALE\_OUTER | Graduation labels outside the scale |

Table 11. Values of ENUM\_SCALE\_STYLE

**4.10. ENUM\_SIZE**

Size. Values are listed in table 12.

| Identifier | Description |
| --- | --- |
| SIZE\_SMALL | Small |
| SIZE\_MIDDLE | Middle |
| SIZE\_LARGE | Large |

Table 12. Values of ENUM\_SIZE

### 5\. Macro Substitution

Coefficients for sizes:

```
#define DIAM_TO_NDCSL_RATIO   5   //needle center diameter (small) as percentage of the body diameter
#define DIAM_TO_NDCSB_RATIO   7.5 //needle center diameter (large) as percentage of the body diameter
//---
#define DIAM_TO_BD_SIZE_S     2 //border width (small) as percentage of the body diameter
#define DIAM_TO_BD_SIZE_B     5 //border width (large) as percentage of the body diameter
//---
#define DIAM_TO_BD_GAP_S      2.0 //space from the body border to inner elements of the gauge (small) as percentage of the body diameter
#define DIAM_TO_BD_GAP_M      3.0 //space from the body border to inner elements of the gauge (middle) as percentage of the body diameter
#define DIAM_TO_BD_GAP_L      7.0 //space from the body border to inner elements of the gauge (large) as percentage of the body diameter
//---
#define DIAM_TO_MSZ_MJ_S      3.3 //size of major scale (small) graduation as percentage of the body diameter
#define DIAM_TO_MSZ_MD_S      2.3 //size of middle scale (small) graduation as percentage of the body diameter
#define DIAM_TO_MSZ_MN_S      1.3 //size of minor scale (small) graduation as percentage of the body diameter
//---
#define DIAM_TO_MSZ_MJ_M      6.5 //size of major scale (middle) graduation as percentage of the body diameter
#define DIAM_TO_MSZ_MD_M      4.8 //size of middle scale (middle) graduation as percentage of the body diameter
#define DIAM_TO_MSZ_MN_M      3.0 //size of minor scale (middle) graduation as percentage of the body diameter
//---
#define DIAM_TO_MSZ_MJ_L      10.0 //size of major scale (large) graduation as percentage of the body diameter
#define DIAM_TO_MSZ_MD_L      7.5  //size of middle scale (large) graduation as percentage of the body diameter
#define DIAM_TO_MSZ_MN_L      5.0  //size of minor scale (large) graduation as percentage of the body diameter
//---
#define DIAM_TO_MFONT_SZ_S    4   //font size of scale (small) graduation labels as percentage of the body diameter
#define DIAM_TO_MFONT_SZ_M    6.5 //font size of scale (middle) graduation labels as percentage of the body diameter
#define DIAM_TO_MFONT_SZ_L    10  //font size of scale (large) graduation labels as percentage of the body diameter
```

Default colors:

```
#define DEF_COL_SCALE      clrBlack
#define DEF_COL_MARK_FONT  clrBlack
#define DEF_COL_CASE       clrMintCream
#define DEF_COL_BORDER     clrLightSteelBlue
#define DEF_COL_LAB        clrDarkGray
#define DEF_COL_NCENTER    clrLightSteelBlue
#define DEF_COL_NEEDLE     clrDimGray
```

### 6\. Modifying the CCanvas Class

**6.1. Drawing a Segment Using Antialiasing Algorithm**

The [LineAA](https://www.mql5.com/en/docs/standardlibrary/canvasgraphics/ccanvas/ccanvaslineaa) method allows to draw a segment using the antialiasing algorithm. But one problem turns up when we draw circular scale marks. When we convert coordinates of the segment's initial and final points from polar to rectangular coordinate system, we have fractional numbers which round up to a whole number. Consequently marks look crooked, which is shown in fig.11(b).

That's why we have added the [LineAA2](https://www.mql5.com/en/articles/1699#lineaa2) method, which differs from [LineAA](https://www.mql5.com/en/docs/standardlibrary/canvasgraphics/ccanvas/ccanvaslineaa) by the fact that the type of x1, y1, x2, y2 input parameters has been changed to double. Thus we can deliver fractional values of coordinates and get rid of the mentioned problem, which is vividly seen in figure 11(c).

```
//+------------------------------------------------------------------+
//| Draw line with antialiasing (with style) v.2                     |
//+------------------------------------------------------------------+
void CCanvas::LineAA2(const double x1,const double y1,const double x2,const double y2,const uint clr,const uint style)
  {
//--- line is out of image boundaries
   if((x1<0 && x2<0) || (y1<0 && y2<0))
      return;
   if(x1>=m_width && x2>=m_width)
      return;
   if(y1>=m_height && y2>=m_height)
      return;
//--- check
   if(x1==x2 && y1==y2)
     {
      PixelSet(int(x1),int(y1),clr);
      return;
     }
//--- set the line style
   if(style!=UINT_MAX)
      LineStyleSet(style);
//--- preliminary calculations
   double dx=x2-x1;
   double dy=y2-y1;
   double xy=sqrt(dx*dx+dy*dy);
   double xx=x1;
   double yy=y1;
   uint   mask=1<<m_style_idx;
//--- set pixels
   dx/=xy;
   dy/=xy;
   do
     {
      if((m_style&mask)==mask)
         PixelSetAA(xx,yy,clr);
      xx+=dx;
      yy+=dy;
      mask<<=1;
      if(mask==0x1000000)
         mask=1;
     }
   while(fabs(x2-xx)>=fabs(dx) && fabs(y2-yy)>=fabs(dy));
  }
```

Fig.11 depicts examples of various methods of drawing scale marks:

- the [Line](https://www.mql5.com/en/docs/standardlibrary/canvasgraphics/ccanvas/ccanvasline) method (a)
- the [LineAA](https://www.mql5.com/en/docs/standardlibrary/canvasgraphics/ccanvas/ccanvaslineaa) method (b)
- the [LineAA2](https://www.mql5.com/en/articles/1699#lineaa2) method (c)


![Method of drawing scale marks](https://c.mql5.com/2/18/mark_line__1.png)

Fig. 11. Various methods of drawing scale marks (increased by 200%)

**6.2. Filling an Area with Antialiased Edges**

The [Fill](https://www.mql5.com/en/docs/standardlibrary/canvasgraphics/ccanvas/ccanvasfill) method is meant for filling an area bounded by segments drawn without use of antialiasing algorithm. If we use this method for filling an area bounded by segments drawn by the [LineAA](https://www.mql5.com/en/docs/standardlibrary/canvasgraphics/ccanvas/ccanvaslineaa) method, the area will be filled incompletely, which is seen in fig.12(a).

![Filling an Area with Antialiased Edges](https://c.mql5.com/2/18/g_fill3.png)

Fig.12. Filling an area with antialiased edges (increased by 200%)

So we have added the [Fill2](https://www.mql5.com/en/articles/1699#fill2) method. The difference is that it fills not the background color, but any color different from color of segments which bound the area. It allows to fill undertints, which cannot be done using the [Fill](https://www.mql5.com/en/docs/standardlibrary/canvasgraphics/ccanvas/ccanvasfill) method. Fig.12(b) depicts an example.

```
//+------------------------------------------------------------------+
//| Fill closed region with color (v.2)                              |
//+------------------------------------------------------------------+
void CCanvas::Fill2(int x,int y,const uint clr)
  {
//--- check
   if(x<0 || x>=m_width || y<0 || y>=m_height)
      return;
//---
   int  index=y*m_width+x;
   uint old_clr=m_pixels[index];
//--- check if replacement is necessary
   if(old_clr==clr)
      return;
//--- use pseudo stack to emulate deeply-nested recursive calls
   int  stack[];
   uint count=1;
   int  idx;
   int  total=ArraySize(m_pixels);
//--- allocate memory for stack
   if(ArrayResize(stack,total)==-1)
      return;
   stack[0]=index;
   m_pixels[index]=clr;
   for(uint i=0;i<count;i++)
     {
      index=stack[i];
      x=index%m_width;
      //--- left adjacent point
      idx=index-1;
      if(x>0 && m_pixels[idx]!=clr)
        {
         stack[count]=idx;
         if(m_pixels[idx]==old_clr)
            count++;
         m_pixels[idx]=clr;
        }
      //--- top adjacent point
      idx=index-m_width;
      if(idx>=0 && m_pixels[idx]!=clr)
        {
         stack[count]=idx;
         if(m_pixels[idx]==old_clr)
            count++;
         m_pixels[idx]=clr;
        }
      //--- right adjacent point
      idx=index+1;
      if(x<m_width-1 && m_pixels[idx]!=clr)
        {
         stack[count]=idx;
         if(m_pixels[idx]==old_clr)
            count++;
         m_pixels[idx]=clr;
        }
      //--- bottom adjacent point
      idx=index+m_width;
      if(idx<total && m_pixels[idx]!=clr)
        {
         stack[count]=idx;
         if(m_pixels[idx]==old_clr)
            count++;
         m_pixels[idx]=clr;
        }
     }
//--- deallocate memory
   ArrayFree(stack);
  }
```

Nevertheless, this method also has disadvantages. If there is a small acute angle, its part will remain unfilled, which is shown in fig.12(c). So we have found a way out of this problem.

1) First the whole canvas (needle layer) is filled with color meant for the needle:

```
n.canvas.Fill(10, 10, ColorToARGB(n.needle.c, n.transparency));
```

2) Then we draw the needle composed of three segments using the [LineAA2](https://www.mql5.com/en/articles/1699#lineaa2) method:

```
n.canvas.LineAA2(db_xbuf[0], db_ybuf[0], db_xbuf[1], db_ybuf[1], 0);
n.canvas.LineAA2(db_xbuf[1], db_ybuf[1], db_xbuf[2], db_ybuf[2], 0);
n.canvas.LineAA2(db_xbuf[2], db_ybuf[2], db_xbuf[0], db_ybuf[0], 0);
```

3) After this we fill the area around the needle with transparent color using the [Fill2](https://www.mql5.com/en/articles/1699#fill2) method:

```
n.canvas.Fill2(10, 10, 0);
```

This method is not the best one, but it allows to draw the proper needle.

![Methods of needle area fill](https://c.mql5.com/2/18/gg_needles__1.png)

Fig.13. Needles filled using different methods

Fig.13 depicts needles filled using different methods.

- a) The needle composed of three segments drawn using the [LineAA2](https://www.mql5.com/en/articles/1699#lineaa2) method and filled using the [Fill2](https://www.mql5.com/en/articles/1699#fill2) method.

- b) The needle drawn via the [FillTriangle](https://www.mql5.com/en/docs/standardlibrary/canvasgraphics/ccanvas/ccanvasfilltriangle) method.

- c) The unfilled needle composed of three segments drawn using the [LineAA2](https://www.mql5.com/en/articles/1699#lineaa2) method.

As we can see, the needle shown in fig.13(b) is craggy and has small deviation from angles divisible by 90 degrees. Besides, we can see that the needle is shifted from the center which is caused by rounding off values of coordinates when we convert them from polar to rectangular coordinate system. But at the same time this method is the most practical in the context of resource intensity (we will come back to this issue [later](https://www.mql5.com/en/articles/1699#pp0800)). The needle shown in fig.13(c) is a trade-off in two methods described above. It is composed of three segments drawn using the [LineAA2](https://www.mql5.com/en/articles/1699#lineaa2) method but without area fill.

### 7\. Application Examples

Let's review the application of the gauge library through several examples.

**7.1. Indicator of Current Profit**

We will start with the simplest one. The following example demonstrates the basic set for adding the gauge to an EA or an indicator.

```
//+------------------------------------------------------------------+
//|                                       profit_gauge_indicator.mq5 |
//|                                         Copyright 2015, Decanium |
//+------------------------------------------------------------------+
#property copyright "Copyright 2015, Decanium"
#property version   "1.00"
#property indicator_plots 0
#property indicator_chart_window

#include <Gauge\gauge_graph.mqh>

input string inp_gauge_name="gg01";                  // Indicator name
input int inp_x = 40;                                // Horizontal offset
input int inp_y = 40;                                // Vertical offset
input int inp_size=300;                              // Indicator size
input string inp_ObjRel="";                          // Name of the base indicator in case of relative positioning
input ENUM_REL_MODE inp_rel_mode=RELATIVE_MODE_NONE; // Relative positioning mode
input ENUM_BASE_CORNER inp_corner=CORNER_LEFT_UPPER; // Anchor corner
input bool inp_back=false;                           // Indicator is in the background
input uchar inp_scale_transparency=0;                // Scale transparency level, 0..255
input uchar inp_needle_transparency=0;               // Needle transparency level, 0..255

//--- declaration of the gauge structure
GAUGE_STR g0;
//+------------------------------------------------------------------+
//| Custom indicator initialization function                         |
//+------------------------------------------------------------------+
int OnInit()
  {
//--- creating the gauge
   if(GaugeCreate(inp_gauge_name,g0,inp_x,inp_y,inp_size,inp_ObjRel,inp_rel_mode,
      inp_corner,inp_back,inp_scale_transparency,inp_needle_transparency)==false)
      return(INIT_FAILED);
//--- drawing the gauge
   GaugeRedraw(g0);
//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Custom indicator deinitialization function                       |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//--- deleting the gauge
   GaugeDelete(g0);
   ChartRedraw();
  }
//+------------------------------------------------------------------+
//| Custom indicator iteration function                              |
//+------------------------------------------------------------------+
int OnCalculate(const int rates_total,
                const int prev_calculated,
                const int begin,
                const double &price[])
  {
//--- updating readings
   double profit=AccountInfoDouble(ACCOUNT_PROFIT);
   GaugeNewValue(g0,profit);
//---
   return(rates_total);
  }
//+------------------------------------------------------------------+
//| ChartEvent function                                              |
//+------------------------------------------------------------------+
void OnChartEvent(const int id,
                  const long &lparam,
                  const double &dparam,
                  const string &sparam)
  {
//---
   if(id==CHARTEVENT_CHART_CHANGE)
     {
      if(GaugeCalcLocation(g0)==true)
         GaugeRelocation(g0);
     }
  }
//+------------------------------------------------------------------+
```

First we need to declare the gauge structure. Then we continue with the initialization function where we create the gauge using the [GaugeCreate()](https://www.mql5.com/en/articles/1699#fgc) function and call the drawing function — [GaugeRedraw()](https://www.mql5.com/en/articles/1699#fgr). [GaugeNewValue()](https://www.mql5.com/en/articles/1699#fgnv) can be used to update readings. In this example, it is called from the [OnCalculate()](https://www.mql5.com/en/docs/basis/function/events#oncalculate) handler.

The gauge will look as shown in fig.14.

![Indicator appearance, default parameters](https://c.mql5.com/2/18/gg_def_view.png)

Fig.14. Default appearance of the gauge

Now we will add ability to set the scale range and the rotation angle. It will add two parameters to the list of inputs.

```
input int inp_scale_range=270;   // Scale range, 30..320 degrees
input int inp_rotation=45;       // Scale rotation, 0..359 degrees
```

Now we extent the initialization code with the call of the function for setting scale parameters.

```
//--- setting parameters of the scale and marks on the scale
   GaugeSetScaleParameters(g0,inp_scale_range,inp_rotation,-200,400,MUL_1,SCALE_INNER,clrBlack);
```

Complementary to new input parameters we will also set:

- new maximum and minimum values (-200 and 400 respectively)
- multiplier of scale graduation labels (MUL\_1)
- scale style (SCALE\_INNER — graduation labels are inside)
- color of labels (clrBlack)


As we have changed scale extreme values, it is desirable to correct a step of major [marks](https://www.mql5.com/en/articles/1699#pp0204). The best value is 100, as it excludes text abundance. At that we will place one middle mark between two major ones and 4 minor marks between two middle ones. Thus we will have a minimum step between marks equal to 10.

```
   GaugeSetMarkParameters(g0,MARKS_INNER,SIZE_MIDDLE,100,1,4);
```

Now we will highlight two data ranges. The range having index 0, which starts with 200 and ends with 400, will be highlighted with the _clrLimeGreen_ color. The range having index 1, which starts with -100 and ends with -200, will be highlighted with the _clrCoral_ color.

```
//--- highlighting ranges on the scale
   GaugeSetRangeParameters(g0,0,true,200,400,clrLimeGreen);
   GaugeSetRangeParameters(g0,1,true,-100,-200,clrCoral);
```

Now we are going to set [legends](https://www.mql5.com/en/articles/1699#pp0205). We determine the gauge description, units of measure and the current value with one decimal place. Let's review them one by one.

_Gauge description:_

```
   GaugeSetLegendParameters(g0,LEGEND_DESCRIPTION,true,"Profit",3,0,14,"Arial",false,false);
```

Displayed string is "Profit", radius is 3, angle is 0, font is 14 conditional units.

_Units of measure:_

```
   GaugeSetLegendParameters(g0,LEGEND_UNITS,true,"USD",8,215,10,"Arial",true,false);
```

Displayed string is "USD", radius is 8, angle is 215, font is 10 conditional units.

_Current value:_

```
   GaugeSetLegendParameters(g0,LEGEND_VALUE,true,"1",4,225,20,"Arial",true,false);
```

Here the string "1" means the format of displaying (one decimal place). Coordinates: radius is 4, angle is 255. Font size is 20 conditional units.

Thus, after we have performed some additional settings, the gauge will look as shown in fig.15.

![Indicator of Current Profit](https://c.mql5.com/2/18/gg_profit_view.png)

Fig.15. Appearance of the gauge after additional setting

**7.2. The Dashboard Indicator**

Now we are going to review more complicated example, namely the _Dashboard_ indicator. It is shown in fig.1. The indicator displays the current profit, spread, free margin as percentage and current values of [ATR](https://www.mql5.com/en/docs/indicators/iatr), [Force Index](https://www.mql5.com/en/docs/indicators/iforce) and [RSI](https://www.mql5.com/en/docs/indicators/irsi).

First we will declare the gauge structure array.

```
//--- declaration of the gauge structure array
GAUGE_STR gg[6];
```

Then we will create and adjust gauges.

The margin level indicator will be placed in the bottom left corner. It will have absolute coordinates, and all other indicators will be located regarding to this indicator or the neighboring one.

```
//--- building the gg00 gauge, margin level
   if(GaugeCreate("gg00",gg[0],5,-90,240,"",RELATIVE_MODE_NONE,
      CORNER_LEFT_LOWER,inp_back,inp_scale_transparency,inp_needle_transparency)==false)
      return(INIT_FAILED);
//--- setting body parameters
   GaugeSetCaseParameters(gg[0],CASE_SECTOR,DEF_COL_CASE,CASE_BORDER_THIN,DEF_COL_BORDER,SIZE_MIDDLE);
//--- setting parameters of the scale and marks on the scale
   GaugeSetScaleParameters(gg[0],120,35,800,2000,MUL_100,SCALE_INNER,clrBlack);
   GaugeSetMarkParameters(gg[0],MARKS_INNER,SIZE_MIDDLE,200,1,4);
   GaugeSetMarkLabelFont(gg[0],SIZE_MIDDLE,"Arial",false,false,DEF_COL_MARK_FONT);
//--- highlighting ranges on the scale
   GaugeSetRangeParameters(gg[0],0,true,1400,2000,clrLimeGreen);
   GaugeSetRangeParameters(gg[0],1,true,1000,800,clrCoral);
//--- setting text labels
   GaugeSetLegendParameters(gg[0],LEGEND_DESCRIPTION,true,"Margin lvl",4,15,12,"Arial",false,false);
   GaugeSetLegendParameters(gg[0],LEGEND_VALUE,true,"0",3,80,16,"Arial",true,false);
   GaugeSetLegendParameters(gg[0],LEGEND_MUL,true,"",4,55,8,"Arial",true,false);
//--- setting needle parameters
   GaugeSetNeedleParameters(gg[0],NDCS_SMALL,DEF_COL_NCENTER,DEF_COL_NEEDLE,NEEDLE_FILL_AA);
```

We continue arranging the bottom row. The next is the current profit indicator.

```
//--- building the gg01 gauge, current profit
   if(GaugeCreate("gg01",gg[1],-80,20,320,"gg00",RELATIVE_MODE_HOR,
      CORNER_LEFT_LOWER,inp_back,inp_scale_transparency,inp_needle_transparency)==false)
      return(INIT_FAILED);
//--- setting body parameters
   GaugeSetCaseParameters(gg[1],CASE_SECTOR,DEF_COL_CASE,CASE_BORDER_THIN,DEF_COL_BORDER,SIZE_MIDDLE);
//--- setting parameters of the scale and marks on the scale
   GaugeSetScaleParameters(gg[1],200,0,-400,400,MUL_1,SCALE_INNER,clrBlack);
   GaugeSetMarkParameters(gg[1],MARKS_INNER,SIZE_MIDDLE,100,1,4);
   GaugeSetMarkLabelFont(gg[1],SIZE_MIDDLE,"Arial",false,false,DEF_COL_MARK_FONT);
//--- highlighting ranges on the scale
   GaugeSetRangeParameters(gg[1],0,true,200,400,clrLimeGreen);
   GaugeSetRangeParameters(gg[1],1,true,-200,-400,clrCoral);
//--- setting text labels
   GaugeSetLegendParameters(gg[1],LEGEND_DESCRIPTION,true,"Profit",3,0,16,"Arial",false,false);
   GaugeSetLegendParameters(gg[1],LEGEND_UNITS,true,"USD",3,-90,10,"Arial",true,false);
   GaugeSetLegendParameters(gg[1],LEGEND_VALUE,true,"1",3,90,12,"Arial",true,false);
//--- setting needle parameters
   GaugeSetNeedleParameters(gg[1],NDCS_SMALL,DEF_COL_NCENTER,DEF_COL_NEEDLE,NEEDLE_FILL_AA);
```

The spread indicator closes the bottom row.

```
//--- building the gg02 gauge, spread
   if(GaugeCreate("gg02",gg[2],-80,-20,240,"gg01",RELATIVE_MODE_HOR,
      CORNER_LEFT_LOWER,inp_back,inp_scale_transparency,inp_needle_transparency)==false)
      return(INIT_FAILED);
//--- setting body parameters
   GaugeSetCaseParameters(gg[2],CASE_SECTOR,DEF_COL_CASE,CASE_BORDER_THIN,DEF_COL_BORDER,SIZE_MIDDLE);
//--- setting parameters of the scale and marks on the scale
   GaugeSetScaleParameters(gg[2],120,-35,60,0,MUL_1,SCALE_INNER,clrBlack);
   GaugeSetMarkParameters(gg[2],MARKS_INNER,SIZE_MIDDLE,10,1,4);
   GaugeSetMarkLabelFont(gg[2],SIZE_MIDDLE,"Arial",false,false,DEF_COL_MARK_FONT);
//--- highlighting ranges on the scale
   GaugeSetRangeParameters(gg[2],0,true,35,10,clrLimeGreen);
   GaugeSetRangeParameters(gg[2],1,true,50,60,clrCoral);
//--- setting text labels
   GaugeSetLegendParameters(gg[2],LEGEND_DESCRIPTION,true,"Spread",4,-15,14,"Arial",false,false);
   GaugeSetLegendParameters(gg[2],LEGEND_VALUE,true,"0",3,-80,16,"Arial",true,false);
//--- setting needle parameters
   GaugeSetNeedleParameters(gg[2],NDCS_SMALL,DEF_COL_NCENTER,DEF_COL_NEEDLE,NEEDLE_FILL_AA);
```

The [ATR](https://www.mql5.com/en/docs/indicators/iatr) indicator (the left one in the upper row) is placed relatively to the free margin indicator.

```
//--- building the gg03 gauge, ATR
   if(GaugeCreate("gg03",gg[3],30,0,180,"gg00",RELATIVE_MODE_VERT,
      CORNER_LEFT_LOWER,inp_back,inp_scale_transparency,inp_needle_transparency)==false)
      return(INIT_FAILED);
//--- setting body parameters
   GaugeSetCaseParameters(gg[3],CASE_ROUND,DEF_COL_CASE,CASE_BORDER_THIN,DEF_COL_BORDER,SIZE_MIDDLE);
//--- setting parameters of the scale and marks on the scale
   GaugeSetScaleParameters(gg[3],270,45,0.001,0.004,MUL_00001,SCALE_INNER,clrBlack);
   GaugeSetMarkParameters(gg[3],MARKS_INNER,SIZE_LARGE,0.001,9,3);
   GaugeSetMarkLabelFont(gg[3],SIZE_LARGE,"Arial",false,false,DEF_COL_MARK_FONT);
//--- highlighting ranges on the scale
   GaugeSetRangeParameters(gg[3],0,true,0.002,0.001,clrYellow);
//--- setting text labels
   GaugeSetLegendParameters(gg[3],LEGEND_DESCRIPTION,true,"ATR",7,-140,26,"Arial",false,false);
//GaugeSetLegendParameters(gg[3],LEGEND_UNITS,true,"USD",8,180,5,"Arial",true,false);
   GaugeSetLegendParameters(gg[3],LEGEND_VALUE,true,"5",2,180,14,"Arial",true,false);
   GaugeSetLegendParameters(gg[3],LEGEND_MUL,true,"",2,0,20,"Arial",true,false);
//--- setting needle parameters
   GaugeSetNeedleParameters(gg[3],NDCS_SMALL,DEF_COL_NCENTER,DEF_COL_NEEDLE,NEEDLE_FILL_AA);
```

The [RSI](https://www.mql5.com/en/docs/indicators/irsi) indicator is placed relatively to the spread indicator (above).

```
//--- building the gg04 gauge, RSI
   if(GaugeCreate("gg04",gg[4],-30,0,180,"gg02",RELATIVE_MODE_VERT,
      CORNER_LEFT_LOWER,inp_back,inp_scale_transparency,inp_needle_transparency)==false)
      return(INIT_FAILED);
//--- setting body parameters
   GaugeSetCaseParameters(gg[4],CASE_ROUND,DEF_COL_CASE,CASE_BORDER_THIN,DEF_COL_BORDER,SIZE_MIDDLE);
//--- setting parameters of the scale and marks on the scale
   GaugeSetScaleParameters(gg[4],270,45,0,100,MUL_10,SCALE_INNER,clrBlack);
   GaugeSetMarkParameters(gg[4],MARKS_INNER,SIZE_LARGE,10,1,4);
   GaugeSetMarkLabelFont(gg[4],SIZE_LARGE,"Arial",false,false,DEF_COL_MARK_FONT);
//--- setting text labels
   GaugeSetLegendParameters(gg[4],LEGEND_DESCRIPTION,true,"RSI",7,-140,26,"Arial",false,false);
   GaugeSetLegendParameters(gg[4],LEGEND_VALUE,true,"2",2,180,16,"Arial",true,false);
   GaugeSetLegendParameters(gg[4],LEGEND_MUL,true,"",2,0,20,"Arial",true,false);
//--- setting needle parameters
   GaugeSetNeedleParameters(gg[4],NDCS_SMALL,DEF_COL_NCENTER,DEF_COL_NEEDLE,NEEDLE_FILL_AA);
```

The [Force Index](https://www.mql5.com/en/docs/indicators/iforce) indicator is placed above the current profit indicator.

```
//--- building the gg05 gauge, Force
   if(GaugeCreate("gg05",gg[5],-10,60,180,"gg03",RELATIVE_MODE_HOR,
      CORNER_LEFT_LOWER,inp_back,inp_scale_transparency,inp_needle_transparency)==false)
      return(INIT_FAILED);
//--- setting body parameters
   GaugeSetCaseParameters(gg[5],CASE_ROUND,DEF_COL_CASE,CASE_BORDER_THIN,DEF_COL_BORDER,SIZE_MIDDLE);
//--- setting parameters of the scale and marks on the scale
   GaugeSetScaleParameters(gg[5],270,45,-4,4,MUL_1,SCALE_INNER,clrBlack);
   GaugeSetMarkParameters(gg[5],MARKS_INNER,SIZE_LARGE,1,1,4);
   GaugeSetMarkLabelFont(gg[5],SIZE_LARGE,"Arial",false,false,DEF_COL_MARK_FONT);
//--- highlighting ranges on the scale
   GaugeSetRangeParameters(gg[5],0,true,-1,-4,clrMediumSeaGreen);
   GaugeSetRangeParameters(gg[5],1,true,1,4,clrCrimson);
//--- setting text labels
   GaugeSetLegendParameters(gg[5],LEGEND_DESCRIPTION,true,"Force",7,-140,20,"Arial",false,false);
   GaugeSetLegendParameters(gg[5],LEGEND_VALUE,true,"5",2,180,14,"Arial",true,false);
   GaugeSetLegendParameters(gg[5],LEGEND_MUL,true,"",3,0,10,"Arial",true,false);
//--- setting needle parameters
   GaugeSetNeedleParameters(gg[5],NDCS_SMALL,DEF_COL_NCENTER,DEF_COL_NEEDLE,NEEDLE_FILL_AA);
```

Gauges can be drawn in a cyclic manner.

```
//--- drawing gauges
   for(int i=0; i<6;i++)
     {
      GaugeRedraw(gg[i]);
      GaugeNewValue(gg[i],0);
     }
```

When the [OnCalculate()](https://www.mql5.com/en/docs/basis/function/events#oncalculate) event occurs, we recalculate current values and call the [GaugeNewValue()](https://www.mql5.com/en/articles/1699#fgnv) function for each indicator.

```
//--- updating readings
//--- spread
   GaugeNewValue(gg[2],spread[rates_total-1]);
//--- current profit
   double profit=AccountInfoDouble(ACCOUNT_PROFIT);
   GaugeNewValue(gg[1],profit);
//--- margin level
   double margin_level=AccountInfoDouble(ACCOUNT_MARGIN_LEVEL);
   GaugeNewValue(gg[0],margin_level);
//--- the ATR indicator
   calculated=BarsCalculated(handle_ATR);
   if(calculated>0)
     {
      double ival[1];
      if(CopyBuffer(handle_ATR,0,0,1,ival)<0)
         Print("ATR CopyBuffer error");
      else
         GaugeNewValue(gg[3],ival[0]);
     }
//--- the RSI indicator
   calculated=BarsCalculated(handle_RSI);
   if(calculated>0)
     {
      double ival[1];
      if(CopyBuffer(handle_RSI,0,0,1,ival)<0)
         Print("RSI CopyBuffer error");
      else
         GaugeNewValue(gg[4],ival[0]);
     }
//--- the Force Index indicator
   calculated=BarsCalculated(handle_Force);
   if(calculated>0)
     {
      double ival[1];
      if(CopyBuffer(handle_Force,0,0,1,ival)<0)
         Print("Force Index CopyBuffer error");
      else
         GaugeNewValue(gg[5],ival[0]);
     }
```

Please note, that there is no point to call [GaugeRelocation()](https://www.mql5.com/en/articles/1699#fgrlc) from the [OnChartEvent()](https://www.mql5.com/en/docs/basis/function/events#onchartevent) event in the given example. Although [relative positioning](https://www.mql5.com/en/articles/1699#pp0100) is used here, we do not need to recalculate coordinates of gauges if position or size of one of them has changed, as gauges are initialized all at once.

### 8\. Resource Intensity Assessment

The needle layer is fully redrawn whenever readings update. It may happen quite often, even several times per second in some instances. That is why the problem of resource intensity of drawing the needle is quite acute. We will write a small script to assess the CPU overhead for drawing the needle using various area fill methods.

```
//+------------------------------------------------------------------+
//|                                                    test_fill.mq5 |
//|                                         Copyright 2015, Decanium |
//+------------------------------------------------------------------+
#property copyright "Copyright 2015, Decanium"
#property version   "1.00"

#include <Canvas/Canvas2.mqh>

CCanvas canvas;
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
   Print("***** start test *****");
//---
   string ObjName="test";
   ObjectDelete(0,ObjName);
   canvas.CreateBitmapLabel(ObjName,10,10,400,400,COLOR_FORMAT_ARGB_NORMALIZE);
//---
   int x[3]={200,185,215};
   int y[3]={70, 250,250};
   int cycles=1000;
   uint col=ColorToARGB(clrRed,255);
   uint c1,c2;
//--- testing the area fill with antialiased edges
   canvas.Erase();
   c1=GetTickCount();
   for(int i=0; i<cycles; i++)
     {
      canvas.Fill(10, 10, col);
      canvas.LineAA2(x[0], y[0], x[1], y[1], 0);
      canvas.LineAA2(x[1], y[1], x[2], y[2], 0);
      canvas.LineAA2(x[2], y[2], x[0], y[0], 0);
      canvas.Fill2(10, 10, 0);
     }
   c2=GetTickCount();
   canvas.Update(true);
   Print("Filled AA: ",c2-c1," ms, ",cycles," cycles, ",
         DoubleToString(double(c2-c1)/double(cycles),2)," ms per cycle");
//--- testing the antialiased contour without filling
   canvas.Erase();
   c1=GetTickCount();
   for(int i=0; i<cycles; i++)
     {
      canvas.LineAA2(x[0], y[0], x[1], y[1], col);
      canvas.LineAA2(x[1], y[1], x[2], y[2], col);
      canvas.LineAA2(x[2], y[2], x[0], y[0], col);
     }
   c2=GetTickCount();
   canvas.Update(true);
   Print("Not filled AA: ",c2-c1," ms, ",cycles," cycles, ",
         DoubleToString(double(c2-c1)/double(cycles),2)," ms per cycle");
//--- testing the area fill without antialiasing
   canvas.Erase();
   c1=GetTickCount();
   for(int i=0; i<cycles; i++)
     {
      canvas.FillTriangle(x[0],y[0],x[1],y[1],x[2],y[2],col);
      canvas.LineAA2(x[0], y[0], (x[1]+x[2])/2, y[1], col);
     }
   c2=GetTickCount();
   canvas.Update(true);
   Print("Filled: ",c2-c1," ms, ",cycles," cycles, ",
         DoubleToString(double(c2-c1)/double(cycles),2)," ms per cycle");
  }
//+------------------------------------------------------------------+
```

The script launches each method of drawing the needle 1000 times in a cycle and measures the time spent for this process in milliseconds.

![Resource Intensity Test](https://c.mql5.com/2/19/test_fill__1.png)

Fig.16. Results of the resource intensity test

As you can see by the results, the filled needle with antialiased edges is drawn hundreds of times longer than the filled needle without antialiasing and tens of times longer than just an antialiased contour line without filling. In this case, beauty really has its price.

### Conclusion

In this article, we have reviewed the set of functions for drawing gauges. The main target of the library creation was the simplicity of adding gauges to an EA or an indicator without delving into the details of drawing and geometry. Though, it's up to you to decide whether we have reached this target.

Special attention should be paid to the resource intensity. Time-consuming computations in the OnCalculate() handler can cause the terminal suspension. So we recommend applying the compromise method of drawing the needle (antialiasing without filling).

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/1699](https://www.mql5.com/ru/articles/1699)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/1699.zip "Download all attachments in the single ZIP archive")

[dashboard.mq5](https://www.mql5.com/en/articles/download/1699/dashboard.mq5 "Download dashboard.mq5")(22.69 KB)

[canvas2.mqh](https://www.mql5.com/en/articles/download/1699/canvas2.mqh "Download canvas2.mqh")(171.52 KB)

[profit\_gauge\_indicator.mq5](https://www.mql5.com/en/articles/download/1699/profit_gauge_indicator.mq5 "Download profit_gauge_indicator.mq5")(8.15 KB)

[gauge\_graph.mqh](https://www.mql5.com/en/articles/download/1699/gauge_graph.mqh "Download gauge_graph.mqh")(154.69 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Applying network functions, or MySQL without DLL: Part II - Program for monitoring changes in signal properties](https://www.mql5.com/en/articles/7495)
- [Applying network functions, or MySQL without DLL: Part I - Connector](https://www.mql5.com/en/articles/7117)
- [Using OpenCL to test candlestick patterns](https://www.mql5.com/en/articles/4236)
- [Liquid Chart](https://www.mql5.com/en/articles/1208)
- [Working with GSM Modem from an MQL5 Expert Advisor](https://www.mql5.com/en/articles/797)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/61725)**
(18)


![Little---Prince](https://c.mql5.com/avatar/avatar_na2.png)

**[Little---Prince](https://www.mql5.com/en/users/little---prince)**
\|
16 Nov 2015 at 08:22

WeChat free experience shouting, another profit more than a thousand points of QQ group: 375124107, plus group please note "77", thank you for cooperation!

![negin1384](https://c.mql5.com/avatar/avatar_na2.png)

**[negin1384](https://www.mql5.com/en/users/negin1384)**
\|
2 Jan 2017 at 11:38

Hi

can you create this indicator for me? (for mt5)

for display price position in daily price limit.

max daily limit and min daily limit set by user.

[![](https://c.mql5.com/3/113/1__2.JPG)](https://c.mql5.com/3/113/1__1.JPG "https://c.mql5.com/3/113/1__1.JPG")

![Mohammad Hanif Ansari](https://c.mql5.com/avatar/2018/3/5AA7ABD0-121F.jpg)

**[Mohammad Hanif Ansari](https://www.mql5.com/en/users/ifctrader)**
\|
12 Apr 2017 at 05:25

Very interesting article. It means that we can do every thing in mql5. Thanks for writer and also translator from Russian.


![Evgeniy Scherbina](https://c.mql5.com/avatar/2014/4/53426E3A-A025.jpg)

**[Evgeniy Scherbina](https://www.mql5.com/en/users/nume)**
\|
25 Dec 2018 at 16:59

In case anyone is having trouble figuring out how to specify a decimal for the main value, here's the function:

```
GaugeSetLegendParameters(gauge, LEGEND_VALUE, true, "1", value_radius, value_angle, 10, "Arial", false, false);
```

"1" is the [number of decimal places](https://www.mql5.com/en/docs/constants/environment_state/marketinfoconstants#enum_symbol_info_integer "MQL5 documentation: Information about the tool").

And the value itself is specified through the function:

```
GaugeNewValue(gauge, profit);
```

![Charles Antoine Dominique Julien Fournel](https://c.mql5.com/avatar/2025/8/689b041a-b59f.jpg)

**[Charles Antoine Dominique Julien Fournel](https://www.mql5.com/en/users/oytaub)**
\|
18 Oct 2025 at 14:50

in case someone is looking for an updatet version,  see enclosed its fully compatible mql5 ccanvas ( 2025 )

[@Serhii Shevchuk](https://www.mql5.com/en/users/decanium) may be you can update your article with it ( i replace all your variable declaration such a s by sls which avoid global declaration of s in other libraries)

![Managing the MetaTrader Terminal via DLL](https://c.mql5.com/2/19/MetaTrader-dll.png)[Managing the MetaTrader Terminal via DLL](https://www.mql5.com/en/articles/1903)

The article deals with managing MetaTrader user interface elements via an auxiliary DLL library using the example of changing push notification delivery settings. The library source code and the sample script are attached to the article.

![Statistical Verification of the Labouchere Money Management System](https://c.mql5.com/2/18/labouchere.png)[Statistical Verification of the Labouchere Money Management System](https://www.mql5.com/en/articles/1800)

In this article, we test the statistical properties of the Labouchere money management system. It is considered to be a less aggressive kind of Martingale, since bets are not doubled, but are raised by a certain amount instead.

![Price Action. Automating the Inside Bar Trading Strategy](https://c.mql5.com/2/19/PA.png)[Price Action. Automating the Inside Bar Trading Strategy](https://www.mql5.com/en/articles/1771)

The article describes the development of a MetaTrader 4 Expert Advisor based on the Inside Bar strategy, including Inside Bar detection principles, as well as pending and stop order setting rules. Test and optimization results are provided as well.

![Using Layouts and Containers for GUI Controls: The CBox Class](https://c.mql5.com/2/19/avatar__2.png)[Using Layouts and Containers for GUI Controls: The CBox Class](https://www.mql5.com/en/articles/1867)

This article presents an alternative method of GUI creation based on layouts and containers, using one layout manager — the CBox class. The CBox class is an auxiliary control that acts as a container for essential controls in a GUI panel. It can make designing graphical panels easier, and in some cases, reduce coding time.

[![](https://www.mql5.com/ff/sh/0hvxp984jjj79943z2/6373d9e5710a718ffa6a7d50a5db9dd1.jpg)\\
Web terminal on your iPhone or Android\\
\\
Full-featured MetaTrader 5 platform for any devices and web browsers\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=uyigsjnbfcdvysiynusmriwvhincciwd&s=c95531ae2fd8a81b0fac3def2e4cf820a67584bbf4b02f76ec75f808942dbbd2&uid=&ref=https://www.mql5.com/en/articles/1699&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5083344870363371982)

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