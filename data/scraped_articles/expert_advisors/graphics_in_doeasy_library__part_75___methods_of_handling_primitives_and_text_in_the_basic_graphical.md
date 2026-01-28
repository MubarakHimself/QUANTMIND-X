---
title: Graphics in DoEasy library (Part 75): Methods of handling primitives and text in the basic graphical element
url: https://www.mql5.com/en/articles/9515
categories: Expert Advisors
relevance_score: 0
scraped_at: 2026-01-24T14:11:41.130807
---

[![](https://www.mql5.com/ff/sh/0hvxp984jjj79943z2/6373d9e5710a718ffa6a7d50a5db9dd1.jpg)\\
Web terminal on your iPhone or Android\\
\\
Full-featured MetaTrader 5 platform for any devices and web browsers\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=uyigsjnbfcdvysiynusmriwvhincciwd&s=c95531ae2fd8a81b0fac3def2e4cf820a67584bbf4b02f76ec75f808942dbbd2&uid=&ref=https://www.mql5.com/en/articles/9515&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5083394773588384490)

MetaTrader 5 / Examples


### Contents

- [Concept](https://www.mql5.com/en/articles/9515#node01)
- [Improving library classes](https://www.mql5.com/en/articles/9515#node02)
- [Methods of working with primitives](https://www.mql5.com/en/articles/9515#node03)
- [Methods of working with a text](https://www.mql5.com/en/articles/9515#node04)
- [Test](https://www.mql5.com/en/articles/9515#node05)
- [What's next?](https://www.mql5.com/en/articles/9515#node06)


### Concept

I continue the development of the basic graphical element object class used as a basis for creating more complex library graphical objects. [In the previous article](https://www.mql5.com/en/articles/9493), I produced the concept of constructing the base graphical object, created the graphical element and endowed it with basic properties that can already be set, changed and received.

Since the [CCanvas class](https://www.mql5.com/en/docs/standardlibrary/canvasgraphics/ccanvas) is meant for drawing "on the canvas", it features the methods for working with graphical primitives and a text. In the current article, I will create the element object methods allowing us to access and handle the CCanvas class methods for drawing. These methods are to be simple. They will be used to create advanced drawing methods in the descendant objects of the element object class.

In addition to creating the methods for working with primitives, I will create the methods for working with files. Our graphical objects, which are the GUI elements of custom programs, should "remember" their properties, status and location on the chart, for example when switching to another timeframe. To achieve this, I will save object properties to a file. When constructing an object, the properties will be read from it.

However, since files should be handled in the graphical object collection class and I have not developed it yet, I will simply add the methods for saving and uploading graphical object properties. While creating the graphical object collection class, we will use the methods for saving and uploading the properties I am going to develop here for the graphical element object.

Also, I will need the class for working with color. I am going to add it to the library as well.

The class will be taken from the [MQL5.com code library](https://www.mql5.com/en/code/888) developed by [Dmitry Fedoseev](https://www.mql5.com/en/users/integer) and submitted to the community.

The final result will be a graphical element ready for further use as a basis for library graphical objects.

### Improving library classes

If we need to clear a CCanvas class object that has transparency, we should use the [Erase()](https://www.mql5.com/en/docs/standardlibrary/canvasgraphics/ccanvas/ccanvaserase) method that receives zero by default:

```
   //--- clear/fill color
   void              Erase(const uint clr=0);
```

In our case, this is an incorrect solution. While clearing the canvas using zero, we lose sight of its alpha channel (color transparency channel), which eventually leads to artifacts when drawing on the canvas with the alpha channel cleared in this way.

To clear the canvas with the alpha channel, use 0x00FFFFFF instead of zero.

This is a completely transparent black color in the ARGB format (Alpha = 0, Red = 255, Green = 255, Blue = 255).

In \\MQL5\\Include\\DoEasy\ **Defines.mqh**, add the macro substitution for specifying such a color:

```
//--- Canvas parameters
#define PAUSE_FOR_CANV_UPDATE          (16)                       // Canvas update frequency
#define NULL_COLOR                     (0x00FFFFFF)               // Zero for the canvas with the alpha channel
//+------------------------------------------------------------------+
```

When displaying a text on the canvas using the [TextOut()](https://www.mql5.com/en/docs/standardlibrary/canvasgraphics/ccanvas/ccanvastextout) method, we can set the anchor angle (text alignment point) for the text message — the bounding rectangle, relative to which the message is to be located. The anchor points are set using six flags — combinations of two flags, all of which are listed below:

**Horizontal text alignment flags:**

- TA\_LEFT — anchor point on the left side of the bounding rectangle
- TA\_CENTER  — horizontal anchor point in the middle of the bounding rectangle
- TA\_RIGHT  — anchor point on the right side of the bounding rectangle

**Vertical text alignment flags:**

- TA\_TOP  — anchor point on the top side of the bounding rectangle
- TA\_VCENTER  — vertical anchor point in the middle of the bounding rectangle
- TA\_BOTTOM  — anchor point on the bottom side of the bounding rectangle

The possible combinations of flags and anchoring methods set by them are shown below:

![](https://c.mql5.com/2/42/firefox_ayLPAHNLi1.png)

To avoid a confusion about which flag comes first, simply set a custom enumeration specifying all possible flag combinations for aligning a text relative to its anchor point:

```
//+------------------------------------------------------------------+
//| Data for handling graphical elements                             |
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| List of anchoring methods                                        |
//| (horizontal and vertical text alignment)                         |
//+------------------------------------------------------------------+
enum ENUM_TEXT_ANCHOR
  {
   TEXT_ANCHOR_LEFT_TOP       =  0,                   // Text anchor point at the upper left corner of the bounding rectangle
   TEXT_ANCHOR_CENTER_TOP     =  1,                   // Text anchor point at the top center side of the bounding rectangle
   TEXT_ANCHOR_RIGHT_TOP      =  2,                   // Text anchor point at the upper right corner of the bounding rectangle
   TEXT_ANCHOR_LEFT_CENTER    =  4,                   // Text anchor point at the left center side of the bounding rectangle
   TEXT_ANCHOR_CENTER         =  5,                   // Text anchor point at the center of the bounding rectangle
   TEXT_ANCHOR_RIGHT_CENTER   =  6,                   // Text anchor point at the right center side of the bounding rectangle
   TEXT_ANCHOR_LEFT_BOTTOM    =  8,                   // Text anchor point at the bottom left corner of the bounding rectangle
   TEXT_ANCHOR_CENTER_BOTTOM  =  9,                   // Text anchor point at the bottom center side of the bounding rectangle
   TEXT_ANCHOR_RIGHT_BOTTOM   =  10,                  // Text anchor point at the bottom right corner of the bounding rectangle
  };
//+------------------------------------------------------------------+
//| The list of graphical element types                              |
//+------------------------------------------------------------------+
```

Here we have set three flags for each text anchoring level:

1. **Top vertical anchor point (TA\_TOP) — 0:**


   - left horizontal anchor point (TA\_LEFT) — 0,

   - center horizontal anchor point (TA\_CENTER) — 1,

   - right horizontal anchor point (TA\_RIGHT) — 2.
2. **Center vertical anchor point (TA\_VCENTER) — 4:**


   - left horizontal anchor point (TA\_LEFT) — 0,

   - center horizontal anchor point (TA\_CENTER) — 1,

   - right horizontal anchor point (TA\_RIGHT) — 2\.
3. **Bottom vertical anchor point (TA\_BOTTOM) — 8:**


   - left horizontal anchor point (TA\_LEFT) — 0,

   - center horizontal anchor point (TA\_CENTER) — 1,

   - right horizontal anchor point (TA\_RIGHT) — 2\.

Each of the ENUM\_TEXT\_ANCHOR enumeration values corresponds to the combination of correctly set flags described above:

- TEXT\_ANCHOR\_LEFT\_TOP = (TA\_LEFT \| TA\_TOP) = 0,
- TEXT\_ANCHOR\_CENTER\_TOP = (TA\_CENTER \| TA\_TOP) = 1,
- TEXT\_ANCHOR\_RIGHT\_TOP = (TA\_RIGHT \| TA\_TOP) = 2,
- TEXT\_ANCHOR\_LEFT\_CENTER = (TA\_LEFT \| TA\_VCENTER) = 4,
- TEXT\_ANCHOR\_CENTER = (TA\_CENTER \| TA\_VCENTER) = 5,
- TEXT\_ANCHOR\_RIGHT\_CENTER = (TA\_RIGHT \| TA\_VCENTER) = 6,
- TEXT\_ANCHOR\_LEFT\_BOTTOM = (TA\_LEFT \| TA\_BOTTOM) = 8,

- TEXT\_ANCHOR\_CENTER\_BOTTOM = (TA\_CENTER \| TA\_BOTTOM) = 9,

- TEXT\_ANCHOR\_RIGHT\_BOTTOM = (TA\_RIGHT \| TA\_BOTTOM) = 10.


I will use this enumeration further on to specify the text alignment relative to its anchor point.

Since I am currently developing the graphical part of the library, I will need various methods for working with color.

[The library of MQL5.com source codes](https://www.mql5.com/en/code/) features the [remarkable library of functions for working with colors](https://www.mql5.com/en/code/888) kindly provided by [Dmitry Fedoseev](https://www.mql5.com/en/users/integer) for general use.

Let's slightly improve the **CColors** class — [make it static](https://www.mql5.com/en/docs/basis/oop/staticmembers) in order not to set the class object, but rather to directly access its methods using the context resolution operator (::), for example:

```
class_name::variable
```

Thus, the CColors class included into the library allows us to access the class methods anywhere in the code (including a custom program), for example to mix two colors — blue with the opacity of 128 and red with the opacity of 64:

```
CColors::BlendColors(ColorToARGB(clrBlue,128),ColorToARGB(clrRed,64));
```

Save the class file in the library directory \\MQL5\\Include\\DoEasy\ **Services\** in **Colors.mqh**.

```
//+------------------------------------------------------------------+
//|                                                       Colors.mqh |
//|                                  Copyright 2021, MetaQuotes Ltd. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2021, MetaQuotes Ltd."
#property link      "https://www.mql5.com/en/users/integer"
#property version   "1.00"
#property strict    // Necessary for mql4
//+------------------------------------------------------------------+
//| Class for working with color                                     |
//+------------------------------------------------------------------+
class CColors
  {
private:
   static double     Arctan2(const double x,const double y);
   static double     Hue_To_RGB(double v1,double v2,double vH);
public:
//+--------------------------------------------------------------------+
//| The list of functions from http://www.easyrgb.com/index.php?X=MATH |
//+--------------------------------------------------------------------+
   static void       RGBtoXYZ(const double aR,const double aG,const double aB,double &oX,double &oY,double &oZ);
   static void       XYZtoRGB(const double aX,const double aY,const double aZ,double &oR,double &oG,double &oB);
   static void       XYZtoYxy(const double aX,const double aY,const double aZ,double &oY,double &ox,double &oy);
   static void       YxyToXYZ(const double aY,const double ax,const double ay,double &oX,double &oY,double &oZ);
   static void       XYZtoHunterLab(const double aX,const double aY,const double aZ,double &oL,double &oa,double &ob);
   static void       HunterLabToXYZ(const double aL,const double aa,const double ab,double &oX,double &oY,double &oZ);
   static void       XYZtoCIELab(const double aX,const double aY,const double aZ,double &oCIEL,double &oCIEa,double &oCIEb);
   static void       CIELabToXYZ(const double aCIEL,const double aCIEa,const double aCIEb,double &oX,double &oY,double &oZ);
   static void       CIELabToCIELCH(const double aCIEL,const double aCIEa,const double aCIEb,double &oCIEL,double &oCIEC,double &oCIEH);
   static void       CIELCHtoCIELab(const double aCIEL,const double aCIEC,const double aCIEH,double &oCIEL,double &oCIEa,double &oCIEb);
   static void       XYZtoCIELuv(const double aX,const double aY,const double aZ,double &oCIEL,double &oCIEu,double &oCIEv);
   static void       CIELuvToXYZ(const double aCIEL,const double aCIEu,const double aCIEv,double &oX,double &oY,double &oZ);
   static void       RGBtoHSL(const double aR,const double aG,const double aB,double &oH,double &oS,double &oL);
   static void       HSLtoRGB(const double aH,const double aS,const double aL,double &oR,double &oG,double &oB);
   static void       RGBtoHSV(const double aR,const double aG,const double aB,double &oH,double &oS,double &oV);
   static void       HSVtoRGB(const double aH,const double aS,const double aV,double &oR,double &oG,double &oB);
   static void       RGBtoCMY(const double aR,const double aG,const double aB,double &oC,double &oM,double &oY);
   static void       CMYtoRGB(const double aC,const double aM,const double aY,double &oR,double &oG,double &oB);
   static void       CMYtoCMYK(const double aC,const double aM,const double aY,double &oC,double &oM,double &oY,double &oK);
   static void       CMYKtoCMY(const double aC,const double aM,const double aY,const double aK,double &oC,double &oM,double &oY);
   static void       RGBtoLab(const double aR,const double aG,const double aB,double &oL,double &oa,double &ob);
//+------------------------------------------------------------------+
//| Other functions for working with color                           |
//+------------------------------------------------------------------+
   static void       ColorToRGB(const color aColor,double &aR,double &aG,double &aB);
   static double     GetR(const color aColor);
   static double     GetG(const color aColor);
   static double     GetB(const color aColor);
   static double     GetA(const color aColor);
   static color      RGBToColor(const double aR,const double aG,const double aB);
   static color      MixColors(const color aCol1,const color aCol2,const double aK);
   static color      BlendColors(const uint lower_color,const uint upper_color);
   static void       Gradient(color &aColors[],color &aOut[],int aOutCount,bool aCycle=false);
   static void       RGBtoXYZsimple(double aR,double aG,double aB,double &oX,double &oY,double &oZ);
   static void       XYZtoRGBsimple(const double aX,const double aY,const double aZ,double &oR,double &oG,double &oB);
   static color      Negative(const color aColor);
   static color      StandardColor(const color aColor,int &aIndex);
   static double     RGBtoGray(double aR,double aG,double aB);
   static double     RGBtoGraySimple(double aR,double aG,double aB);
  };
//+------------------------------------------------------------------+
//| Class methods                                                    |
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| Arctan2                                                          |
//+------------------------------------------------------------------+
double CColors::Arctan2(const double x,const double y)
  {
   if(y==0)
      return(x<0 ? M_PI : 0);
   else
     {
      if(x>0)
         return(::atan(y/x));
      if(x<0)
         return(y>0 ? atan(y/x)+M_PI : atan(y/x)-M_PI);
      else
         return(y<0 ? -M_PI_2 : M_PI_2);
     }
  }
//+------------------------------------------------------------------+
//| Hue_To_RGB                                                       |
//+------------------------------------------------------------------+
double CColors::Hue_To_RGB(double v1,double v2,double vH)
  {
   if(vH<0)
      vH+=1.0;
   if(vH>1.0)
      vH-=1;
   if((6.0*vH)<1.0)
      return(v1+(v2-v1)*6.0*vH);
   if((2.0*vH)<1.0)
      return(v2);
   if((3.0*vH)<2.0)
      return(v1+(v2-v1)*((2.0/3.0)-vH)*6.0);
//---
   return(v1);
  }
//+------------------------------------------------------------------+
//| Conversion of RGB into XYZ                                       |
//+------------------------------------------------------------------+
void CColors::RGBtoXYZ(const double aR,const double aG,const double aB,double &oX,double &oY,double &oZ)
  {
   double var_R=aR/255;
   double var_G=aG/255;
   double var_B=aB/255;
//---
   if(var_R>0.04045)
      var_R=::pow((var_R+0.055)/1.055,2.4);
   else
      var_R=var_R/12.92;
//---
   if(var_G>0.04045)
      var_G=::pow((var_G+0.055)/1.055,2.4);
   else
      var_G=var_G/12.92;
//---
   if(var_B>0.04045)
      var_B=::pow((var_B+0.055)/1.055,2.4);
   else
      var_B=var_B/12.92;
//---
   var_R =var_R*100.0;
   var_G =var_G*100.0;
   var_B =var_B*100.0;
   oX    =var_R*0.4124+var_G*0.3576+var_B*0.1805;
   oY    =var_R*0.2126+var_G*0.7152+var_B*0.0722;
   oZ    =var_R*0.0193+var_G*0.1192+var_B*0.9505;
  }
//+------------------------------------------------------------------+
//| Conversion of XYZ into RGB                                       |
//+------------------------------------------------------------------+
void CColors::XYZtoRGB(const double aX,const double aY,const double aZ,double &oR,double &oG,double &oB)
  {
   double var_X =aX/100;
   double var_Y =aY/100;
   double var_Z =aZ/100;
   double var_R =var_X*3.2406+var_Y*-1.5372+var_Z*-0.4986;
   double var_G =var_X*(-0.9689)+var_Y*1.8758+var_Z*0.0415;
   double var_B =var_X*0.0557+var_Y*(-0.2040)+var_Z*1.0570;
//---
   if(var_R>0.0031308)
      var_R=1.055*(::pow(var_R,1.0/2.4))-0.055;
   else
      var_R=12.92*var_R;
//---
   if(var_G>0.0031308)
      var_G=1.055*(::pow(var_G,1.0/2.4))-0.055;
   else
      var_G=12.92*var_G;
//---
   if(var_B>0.0031308)
      var_B=1.055*(::pow(var_B,1.0/2.4))-0.055;
   else
      var_B=12.92*var_B;
//---
   oR =var_R*255.0;
   oG =var_G*255.0;
   oB =var_B*255.0;
  }
//+------------------------------------------------------------------+
//| Conversion of XYZ into Yxy                                       |
//+------------------------------------------------------------------+
void CColors::XYZtoYxy(const double aX,const double aY,const double aZ,double &oY,double &ox,double &oy)
  {
   oY =aY;
   ox =aX/(aX+aY+aZ);
   oy =aY/(aX+aY+aZ);
  }
//+------------------------------------------------------------------+
//| Conversion of Yxy into XYZ                                       |
//+------------------------------------------------------------------+
void CColors::YxyToXYZ(const double aY,const double ax,const double ay,double &oX,double &oY,double &oZ)
  {
   oX =ax*(aY/ay);
   oY =aY;
   oZ =(1.0-ax-ay)*(aY/ay);
  }
//+------------------------------------------------------------------+
//| Conversion of XYZ into HunterLab                                 |
//+------------------------------------------------------------------+
void CColors::XYZtoHunterLab(const double aX,const double aY,const double aZ,double &oL,double &oa,double &ob)
  {
   oL =10.0*::sqrt(aY);
   oa =17.5*(((1.02*aX)-aY)/::sqrt(aY));
   ob =7.0*((aY-(0.847*aZ))/::sqrt(aY));
  }
//+------------------------------------------------------------------+
//| Conversion of HunterLab into XYZ                                 |
//+------------------------------------------------------------------+
void CColors::HunterLabToXYZ(const double aL,const double aa,const double ab,double &oX,double &oY,double  &oZ)
  {
   double var_Y =aL/10.0;
   double var_X =aa/17.5*aL/10.0;
   double var_Z =ab/7.0*aL/10.0;
//---
   oY =::pow(var_Y,2);
   oX =(var_X+oY)/1.02;
   oZ =-(var_Z-oY)/0.847;
  }
//+------------------------------------------------------------------+
//| Conversion of XYZ into CIELab                                    |
//+------------------------------------------------------------------+
void CColors::XYZtoCIELab(const double aX,const double aY,const double aZ,double &oCIEL,double &oCIEa,double &oCIEb)
  {
   double ref_X =95.047;
   double ref_Y =100.0;
   double ref_Z =108.883;
   double var_X =aX/ref_X;
   double var_Y =aY/ref_Y;
   double var_Z =aZ/ref_Z;
//---
   if(var_X>0.008856)
      var_X=::pow(var_X,1.0/3.0);
   else
      var_X=(7.787*var_X)+(16.0/116.0);
//---
   if(var_Y>0.008856)
      var_Y=::pow(var_Y,1.0/3.0);
   else
      var_Y=(7.787*var_Y)+(16.0/116.0);
//---
   if(var_Z>0.008856)
      var_Z=::pow(var_Z,1.0/3.0);
   else
      var_Z=(7.787*var_Z)+(16.0/116.0);
//---
   oCIEL =(116.0*var_Y)-16.0;
   oCIEa =500.0*(var_X-var_Y);
   oCIEb =200*(var_Y-var_Z);
  }
//+------------------------------------------------------------------+
//| Conversion of CIELab into ToXYZ                                  |
//+------------------------------------------------------------------+
void CColors::CIELabToXYZ(const double aCIEL,const double aCIEa,const double aCIEb,double &oX,double &oY,double &oZ)
  {
   double var_Y =(aCIEL+16.0)/116.0;
   double var_X =aCIEa/500.0+var_Y;
   double var_Z =var_Y-aCIEb/200.0;
//---
   if(::pow(var_Y,3)>0.008856)
      var_Y=::pow(var_Y,3);
   else
      var_Y=(var_Y-16.0/116.0)/7.787;
//---
   if(::pow(var_X,3)>0.008856)
      var_X=::pow(var_X,3);
   else
      var_X=(var_X-16.0/116.0)/7.787;
//---
   if(::pow(var_Z,3)>0.008856)
      var_Z=::pow(var_Z,3);
   else
      var_Z=(var_Z-16.0/116.0)/7.787;
//---
   double ref_X =95.047;
   double ref_Y =100.0;
   double ref_Z =108.883;
//---
   oX =ref_X*var_X;
   oY =ref_Y*var_Y;
   oZ =ref_Z*var_Z;
  }
//+------------------------------------------------------------------+
//| Conversion of CIELab into CIELCH                                 |
//+------------------------------------------------------------------+
void CColors::CIELabToCIELCH(const double aCIEL,const double aCIEa,const double aCIEb,double &oCIEL,double &oCIEC,double &oCIEH)
  {
   double var_H=Arctan2(aCIEb,aCIEa);
//---
   if(var_H>0)
      var_H=(var_H/M_PI)*180.0;
   else
      var_H=360.0-(::fabs(var_H)/M_PI)*180.0;
//---
   oCIEL =aCIEL;
   oCIEC =::sqrt(::pow(aCIEa,2)+::pow(aCIEb,2));
   oCIEH =var_H;
  }
//+------------------------------------------------------------------+
//| Conversion of CIELCH into CIELab                                 |
//+------------------------------------------------------------------+
void CColors::CIELCHtoCIELab(const double aCIEL,const double aCIEC,const double aCIEH,double &oCIEL,double &oCIEa,double &oCIEb)
  {
//--- Arguments from 0 to 360°
   oCIEL =aCIEL;
   oCIEa =::cos(M_PI*aCIEH/180.0)*aCIEC;
   oCIEb =::sin(M_PI*aCIEH/180)*aCIEC;
  }
//+------------------------------------------------------------------+
//| Conversion of XYZ into CIELuv                                    |
//+------------------------------------------------------------------+
void CColors::XYZtoCIELuv(const double aX,const double aY,const double aZ,double &oCIEL,double &oCIEu,double &oCIEv)
  {
   double var_U =(4.0*aX)/(aX+(15.0*aY)+(3.0*aZ));
   double var_V =(9.0*aY)/(aX+(15.0*aY)+(3.0*aZ));
   double var_Y =aY/100.0;
//---
   if(var_Y>0.008856)
      var_Y=::pow(var_Y,1.0/3.0);
   else
      var_Y=(7.787*var_Y)+(16.0/116.0);
//---
   double ref_X =95.047;
   double ref_Y =100.000;
   double ref_Z =108.883;
   double ref_U =(4.0*ref_X)/(ref_X+(15.0*ref_Y)+(3.0*ref_Z));
   double ref_V =(9.0*ref_Y)/(ref_X+(15.0*ref_Y)+(3.0*ref_Z));
//---
   oCIEL =(116.0*var_Y)-16.0;
   oCIEu =13.0*oCIEL*(var_U-ref_U);
   oCIEv =13.0*oCIEL*(var_V-ref_V);
  }
//+------------------------------------------------------------------+
//| Conversion of CIELuv into XYZ                                    |
//+------------------------------------------------------------------+
void CColors::CIELuvToXYZ(const double aCIEL,const double aCIEu,const double aCIEv,double &oX,double &oY,double &oZ)
  {
   double var_Y=(aCIEL+16.0)/116.0;
//---
   if(::pow(var_Y,3)>0.008856)
      var_Y=::pow(var_Y,3);
   else
      var_Y=(var_Y-16.0/116.0)/7.787;
//---
   double ref_X =95.047;
   double ref_Y =100.000;
   double ref_Z =108.883;
   double ref_U =(4.0*ref_X)/(ref_X+(15.0*ref_Y)+(3.0*ref_Z));
   double ref_V =(9.0*ref_Y)/(ref_X+(15.0*ref_Y)+(3.0*ref_Z));
   double var_U =aCIEu/(13.0*aCIEL)+ref_U;
   double var_V =aCIEv/(13.0*aCIEL)+ref_V;
//---
   oY=var_Y*100.0;
   oX=-(9.0*oY*var_U)/((var_U-4.0)*var_V-var_U*var_V);
   oZ=(9.0*oY-(15.0*var_V*oY)-(var_V*oX))/(3.0*var_V);
  }
//+------------------------------------------------------------------+
//| Conversion of RGB into HSL                                       |
//+------------------------------------------------------------------+
void CColors::RGBtoHSL(const double aR,const double aG,const double aB,double &oH,double &oS,double &oL)
  {
   double var_R   =(aR/255);
   double var_G   =(aG/255);
   double var_B   =(aB/255);
   double var_Min =::fmin(var_R,::fmin(var_G,var_B));
   double var_Max =::fmax(var_R,::fmax(var_G,var_B));
   double del_Max =var_Max-var_Min;
//---
   oL=(var_Max+var_Min)/2;
//---
   if(del_Max==0)
     {
      oH=0;
      oS=0;
     }
   else
     {
      if(oL<0.5)
         oS=del_Max/(var_Max+var_Min);
      else
         oS=del_Max/(2.0-var_Max-var_Min);
      //---
      double del_R =(((var_Max-var_R)/6.0)+(del_Max/2.0))/del_Max;
      double del_G =(((var_Max-var_G)/6.0)+(del_Max/2.0))/del_Max;
      double del_B =(((var_Max-var_B)/6.0)+(del_Max/2.0))/del_Max;
      //---
      if(var_R==var_Max)
         oH=del_B-del_G;
      else if(var_G==var_Max)
         oH=(1.0/3.0)+del_R-del_B;
      else if(var_B==var_Max)
         oH=(2.0/3.0)+del_G-del_R;
      //---
      if(oH<0)
         oH+=1.0;
      //---
      if(oH>1)
         oH-=1.0;
     }
  }
//+------------------------------------------------------------------+
//| Conversion of HSL into RGB                                       |
//+------------------------------------------------------------------+
void CColors::HSLtoRGB(const double aH,const double aS,const double aL,double &oR,double &oG,double &oB)
  {
   if(aS==0)
     {
      oR=aL*255;
      oG=aL*255;
      oB=aL*255;
     }
   else
     {
      double var_2=0.0;
      //---
      if(aL<0.5)
         var_2=aL*(1.0+aS);
      else
         var_2=(aL+aS)-(aS*aL);
      //---
      double var_1=2.0*aL-var_2;
      oR =255.0*Hue_To_RGB(var_1,var_2,aH+(1.0/3.0));
      oG =255.0*Hue_To_RGB(var_1,var_2,aH);
      oB =255.0*Hue_To_RGB(var_1,var_2,aH-(1.0/3.0));
     }
  }
//+------------------------------------------------------------------+
//| Conversion of RGB into HSV                                       |
//+------------------------------------------------------------------+
void CColors::RGBtoHSV(const double aR,const double aG,const double aB,double &oH,double &oS,double &oV)
  {
   const double var_R   =(aR/255.0);
   const double var_G   =(aG/255.0);
   const double var_B   =(aB/255.0);

   const double var_Min =::fmin(var_R,::fmin(var_G, var_B));
   const double var_Max =::fmax(var_R,::fmax(var_G,var_B));
   const double del_Max =var_Max-var_Min;
//---
   oV=var_Max;
//---
   if(del_Max==0)
     {
      oH=0;
      oS=0;
     }
   else
     {
      oS=del_Max/var_Max;
      const double del_R =(((var_Max-var_R)/6.0)+(del_Max/2))/del_Max;
      const double del_G =(((var_Max-var_G)/6.0)+(del_Max/2))/del_Max;
      const double del_B =(((var_Max-var_B)/6.0)+(del_Max/2))/del_Max;
      //---
      if(var_R==var_Max)
         oH=del_B-del_G;
      else if(var_G==var_Max)
         oH=(1.0/3.0)+del_R-del_B;
      else if(var_B==var_Max)
         oH=(2.0/3.0)+del_G-del_R;
      //---
      if(oH<0)
         oH+=1.0;
      //---
      if(oH>1.0)
         oH-=1.0;
     }
  }
//+------------------------------------------------------------------+
//| Conversion of HSV into RGB                                       |
//+------------------------------------------------------------------+
void CColors::HSVtoRGB(const double aH,const double aS,const double aV,double &oR,double &oG,double &oB)
  {
   if(aS==0)
     {
      oR =aV*255.0;
      oG =aV*255.0;
      oB =aV*255.0;
     }
   else
     {
      double var_h=aH*6.0;
      //---
      if(var_h==6)
         var_h=0;
      //---
      int    var_i =int(var_h);
      double var_1 =aV*(1.0-aS);
      double var_2 =aV*(1.0-aS*(var_h-var_i));
      double var_3 =aV*(1.0-aS*(1.0-(var_h-var_i)));
      double var_r =0.0;
      double var_g =0.0;
      double var_b =0.0;
      //---
      if(var_i==0)
        {
         var_r =aV;
         var_g =var_3;
         var_b =var_1;
        }
      else if(var_i==1.0)
        {
         var_r=var_2;
         var_g=aV;
         var_b=var_1;
        }
      else if(var_i==2.0)
        {
         var_r=var_1;
         var_g=aV;
         var_b=var_3;
        }
      else if(var_i==3)
        {
         var_r=var_1;
         var_g=var_2;
         var_b=aV;
        }
      else if(var_i==4)
        {
         var_r=var_3;
         var_g=var_1;
         var_b=aV;
        }
      else
        {
         var_r=aV;
         var_g=var_1;
         var_b=var_2;
        }
      //---
      oR =var_r*255.0;
      oG =var_g*255.0;
      oB =var_b*255.0;
     }
  }
//+------------------------------------------------------------------+
//| Conversion of RGB into CMY                                       |
//+------------------------------------------------------------------+
void CColors::RGBtoCMY(const double aR,const double aG,const double aB,double &oC,double &oM,double &oY)
  {
   oC =1.0-(aR/255.0);
   oM =1.0-(aG/255.0);
   oY =1.0-(aB/255.0);
  }
//+------------------------------------------------------------------+
//| Conversion of CMY into RGB                                       |
//+------------------------------------------------------------------+
void CColors::CMYtoRGB(const double aC,const double aM,const double aY,double &oR,double &oG,double &oB)
  {
   oR =(1.0-aC)*255.0;
   oG =(1.0-aM)*255.0;
   oB =(1.0-aY)*255.0;
  }
//+------------------------------------------------------------------+
//| Conversion of CMY into CMYK                                      |
//+------------------------------------------------------------------+
void CColors::CMYtoCMYK(const double aC,const double aM,const double aY,double &oC,double &oM,double &oY,double &oK)
  {
   double var_K=1;
//---
   if(aC<var_K)
      var_K=aC;
   if(aM<var_K)
      var_K=aM;
   if(aY<var_K)
      var_K=aY;
//---
   if(var_K==1.0)
     {
      oC =0;
      oM =0;
      oY =0;
     }
   else
     {
      oC =(aC-var_K)/(1.0-var_K);
      oM =(aM-var_K)/(1.0-var_K);
      oY =(aY-var_K)/(1.0-var_K);
     }
//---
   oK=var_K;
  }
//+------------------------------------------------------------------+
//| Conversion of CMYK into CMY                                      |
//+------------------------------------------------------------------+
void CColors::CMYKtoCMY(const double aC,const double aM,const double aY,const double aK,double &oC,double &oM,double &oY)
  {
   oC =(aC*(1.0-aK)+aK);
   oM =(aM*(1.0-aK)+aK);
   oY =(aY*(1.0-aK)+aK);
  }
//+------------------------------------------------------------------+
//| Conversion of RGB into Lab                                       |
//+------------------------------------------------------------------+
void CColors::RGBtoLab(const double aR,const double aG,const double aB,double &oL,double &oa,double &ob)
  {
   double X=0,Y=0,Z=0;
   RGBtoXYZ(aR,aG,aB,X,Y,Z);
   XYZtoHunterLab(X,Y,Z,oL,oa,ob);
  }
//+------------------------------------------------------------------+
//| Getting values of the RGB components                             |
//+------------------------------------------------------------------+
void CColors::ColorToRGB(const color aColor,double &aR,double &aG,double &aB)
  {
   aR =GetR(aColor);
   aG =GetG(aColor);
   aB =GetB(aColor);
  }
//+------------------------------------------------------------------+
//| Getting the R component value                                    |
//+------------------------------------------------------------------+
double CColors::GetR(const color aColor)
  {
   return(aColor&0xff);
  }
//+------------------------------------------------------------------+
//| Getting the G component value                                    |
//+------------------------------------------------------------------+
double CColors::GetG(const color aColor)
  {
   return((aColor>>8)&0xff);
  }
//+------------------------------------------------------------------+
//| Getting the B component value                                    |
//+------------------------------------------------------------------+
double CColors::GetB(const color aColor)
  {
   return((aColor>>16)&0xff);
  }
//+------------------------------------------------------------------+
//| Getting the A component value                                    |
//+------------------------------------------------------------------+
double CColors::GetA(const color aColor)
  {
   return(double(uchar((aColor)>>24)));
  }
//+------------------------------------------------------------------+
//| Conversion of RGB into const color                               |
//+------------------------------------------------------------------+
color CColors::RGBToColor(const double aR,const double aG,const double aB)
  {
   int int_r =(int)::round(aR);
   int int_g =(int)::round(aG);
   int int_b =(int)::round(aB);
   int Color =0;
//---
   Color=int_b;
   Color<<=8;
   Color|=int_g;
   Color<<=8;
   Color|=int_r;
//---
   return((color)Color);
  }
//+------------------------------------------------------------------+
//| Getting the value of the intermediary color between two colors   |
//+------------------------------------------------------------------+
color CColors::MixColors(const color aCol1,const color aCol2,const double aK)
  {
//--- aK - from 0 to 1
   double R1=0.0,G1=0.0,B1=0.0,R2=0.0,G2=0.0,B2=0.0;
//---
   ColorToRGB(aCol1,R1,G1,B1);
   ColorToRGB(aCol2,R2,G2,B2);
//---
   R1+=(int)::round(aK*(R2-R1));
   G1+=(int)::round(aK*(G2-G1));
   B1+=(int)::round(aK*(B2-B1));
//---
   return(RGBToColor(R1,G1,B1));
  }
//+------------------------------------------------------------------+
//| Blending two colors considering the transparency of color on top |
//+------------------------------------------------------------------+
color CColors::BlendColors(const uint lower_color,const uint upper_color)
  {
   double r1=0,g1=0,b1=0;
   double r2=0,g2=0,b2=0,alpha=0;
   double r3=0,g3=0,b3=0;
//--- Convert the colors in ARGB format
   uint pixel_color=::ColorToARGB(upper_color);
//--- Get the components of the lower and upper colors
   ColorToRGB(lower_color,r1,g1,b1);
   ColorToRGB(pixel_color,r2,g2,b2);
//--- Get the transparency percentage from 0.00 to 1.00
   alpha=GetA(upper_color)/255.0;
//--- If there is transparency
   if(alpha<1.0)
     {
      //--- Blend the components taking the alpha channel into account
      r3=(r1*(1-alpha))+(r2*alpha);
      g3=(g1*(1-alpha))+(g2*alpha);
      b3=(b1*(1-alpha))+(b2*alpha);
      //--- Adjustment of the obtained values
      r3=(r3>255)? 255 : r3;
      g3=(g3>255)? 255 : g3;
      b3=(b3>255)? 255 : b3;
     }
   else
     {
      r3=r2;
      g3=g2;
      b3=b2;
     }
//--- Combine the obtained components and return the color
   return(RGBToColor(r3,g3,b3));
  }
//+------------------------------------------------------------------+
//| Getting an array of the specified size with a color gradient     |
//+------------------------------------------------------------------+
void CColors::Gradient(color &aColors[],   // List of colors
                       color &aOut[],      // Return array
                       int   aOutCount,    // Set the size of the return array
                       bool  aCycle=false) // Closed-loop cycle. Return array ends with the same color as it starts with
  {
   ::ArrayResize(aOut,aOutCount);
//---
   int    InCount =::ArraySize(aColors)+aCycle;
   int    PrevJ   =0;
   int    nci     =0;
   double K       =0.0;
//---
   for(int i=1; i<InCount; i++)
     {
      int J=(aOutCount-1)*i/(InCount-1);
      //---
      for(int j=PrevJ; j<=J; j++)
        {
         if(aCycle && i==InCount-1)
           {
            nci =0;
            K   =1.0*(j-PrevJ)/(J-PrevJ+1);
           }
         else
           {
            nci =i;
            K   =1.0*(j-PrevJ)/(J-PrevJ);
           }
         aOut[j]=MixColors(aColors[i-1],aColors[nci],K);
        }
      PrevJ=J;
     }
  }
//+------------------------------------------------------------------+
//| One more variant of conversion of RGB into XYZ and               |
//| corresponding conversion of XYZ into RGB                         |
//+------------------------------------------------------------------+
void CColors::RGBtoXYZsimple(double aR,double aG,double aB,double &oX,double &oY,double &oZ)
  {
   aR/=255;
   aG/=255;
   aB/=255;
   aR*=100;
   aG*=100;
   aB*=100;
//---
   oX=0.431*aR+0.342*aG+0.178*aB;
   oY=0.222*aR+0.707*aG+0.071*aB;
   oZ=0.020*aR+0.130*aG+0.939*aB;
  }
//+------------------------------------------------------------------+
//| XYZtoRGBsimple                                                   |
//+------------------------------------------------------------------+
void CColors::XYZtoRGBsimple(const double aX,const double aY,const double aZ,double &oR,double &oG,double &oB)
  {
   oR=3.063*aX-1.393*aY-0.476*aZ;
   oG=-0.969*aX+1.876*aY+0.042*aZ;
   oB=0.068*aX-0.229*aY+1.069*aZ;
  }
//+------------------------------------------------------------------+
//| Negative color                                                   |
//+------------------------------------------------------------------+
color CColors::Negative(const color aColor)
  {
   double R=0.0,G=0.0,B=0.0;
   ColorToRGB(aColor,R,G,B);
//---
   return(RGBToColor(255-R,255-G,255-B));
  }
//+------------------------------------------------------------------+
//| Search for the most similar color                                |
//| in the set of standard colors of the terminal                    |
//+------------------------------------------------------------------+
color CColors::StandardColor(const color aColor,int &aIndex)
  {
   color m_c[]=
     {
      clrBlack,clrDarkGreen,clrDarkSlateGray,clrOlive,clrGreen,clrTeal,clrNavy,clrPurple,clrMaroon,clrIndigo,
      clrMidnightBlue,clrDarkBlue,clrDarkOliveGreen,clrSaddleBrown,clrForestGreen,clrOliveDrab,clrSeaGreen,
      clrDarkGoldenrod,clrDarkSlateBlue,clrSienna,clrMediumBlue,clrBrown,clrDarkTurquoise,clrDimGray,
      clrLightSeaGreen,clrDarkViolet,clrFireBrick,clrMediumVioletRed,clrMediumSeaGreen,clrChocolate,clrCrimson,
      clrSteelBlue,clrGoldenrod,clrMediumSpringGreen,clrLawnGreen,clrCadetBlue,clrDarkOrchid,clrYellowGreen,
      clrLimeGreen,clrOrangeRed,clrDarkOrange,clrOrange,clrGold,clrYellow,clrChartreuse,clrLime,clrSpringGreen,
      clrAqua,clrDeepSkyBlue,clrBlue,clrFuchsia,clrRed,clrGray,clrSlateGray,clrPeru,clrBlueViolet,clrLightSlateGray,
      clrDeepPink,clrMediumTurquoise,clrDodgerBlue,clrTurquoise,clrRoyalBlue,clrSlateBlue,clrDarkKhaki,clrIndianRed,
      clrMediumOrchid,clrGreenYellow,clrMediumAquamarine,clrDarkSeaGreen,clrTomato,clrRosyBrown,clrOrchid,
      clrMediumPurple,clrPaleVioletRed,clrCoral,clrCornflowerBlue,clrDarkGray,clrSandyBrown,clrMediumSlateBlue,
      clrTan,clrDarkSalmon,clrBurlyWood,clrHotPink,clrSalmon,clrViolet,clrLightCoral,clrSkyBlue,clrLightSalmon,
      clrPlum,clrKhaki,clrLightGreen,clrAquamarine,clrSilver,clrLightSkyBlue,clrLightSteelBlue,clrLightBlue,
      clrPaleGreen,clrThistle,clrPowderBlue,clrPaleGoldenrod,clrPaleTurquoise,clrLightGray,clrWheat,clrNavajoWhite,
      clrMoccasin,clrLightPink,clrGainsboro,clrPeachPuff,clrPink,clrBisque,clrLightGoldenrod,clrBlanchedAlmond,
      clrLemonChiffon,clrBeige,clrAntiqueWhite,clrPapayaWhip,clrCornsilk,clrLightYellow,clrLightCyan,clrLinen,
      clrLavender,clrMistyRose,clrOldLace,clrWhiteSmoke,clrSeashell,clrIvory,clrHoneydew,clrAliceBlue,clrLavenderBlush,
      clrMintCream,clrSnow,clrWhite,clrDarkCyan,clrDarkRed,clrDarkMagenta,clrAzure,clrGhostWhite,clrFloralWhite
     };
//---
   double m_rv=0.0,m_gv=0.0,m_bv=0.0;
//---
   ColorToRGB(aColor,m_rv,m_gv,m_bv);
//---
   double m_md=0.3*::pow(255,2)+0.59*::pow(255,2)+0.11*::pow(255,2)+1;
   aIndex=0;
//---
   for(int i=0; i<138; i++)
     {
      double m_d=0.3*::pow(GetR(m_c[i])-m_rv,2)+0.59*::pow(GetG(m_c[i])-m_gv,2)+0.11*::pow(GetB(m_c[i])-m_bv,2);
      //---
      if(m_d<m_md)
        {
         m_md   =m_d;
         aIndex =i;
        }
     }
//---
   return(m_c[aIndex]);
  }
//+------------------------------------------------------------------+
//| Conversion into gray color                                       |
//+------------------------------------------------------------------+
double CColors::RGBtoGray(double aR,double aG,double aB)
  {
   aR/=255;
   aG/=255;
   aB/=255;
//---
   aR=::pow(aR,2.2);
   aG=::pow(aG,2.2);
   aB=::pow(aB,2.2);
//---
   double rY=0.21*aR+0.72*aG+0.07*aB;
   rY=::pow(rY,1.0/2.2);
//---
   return(rY);
  }
//+------------------------------------------------------------------+
//| Simple conversion into gray color                                |
//+------------------------------------------------------------------+
double CColors::RGBtoGraySimple(double aR,double aG,double aB)
  {
   aR/=255;
   aG/=255;
   aB/=255;
   double rY=0.3*aR+0.59*aG+0.11*aB;
//---
   return(rY);
  }
//+------------------------------------------------------------------+
```

All made changes boil down to setting the ['static' modifier](https://www.mql5.com/en/docs/basis/oop/staticmembers) to each of the methods, as well as some purely cosmetic changes (apart from variable names in the method arguments) that suit my coding style. Besides, I have added the RGBtoLab() method for converting the RGB color model into Lab. The method simply converts the RGB model into XYZ, which in turn is converted into the Lab color model. This matter was once considered by [Anatoli Kazharski](https://www.mql5.com/en/users/tol64) in his article " [Graphical Interfaces IX: The Color Picker Control (Chapter 1)](https://www.mql5.com/en/articles/2579)":

There is no suitable method in the **CColors** class for converting from the **RGB** format to **Lab**. Therefore, where the conversion **RGB**-> **Lab** is required, double correction through color master model **XYZ** will need to be applied: **RGB**-> **XYZ**-> **Lab**.

I have simply followed the author's advice.

To make the **CColors** class visible to the entire library and programs based on it, include the class file to the file of the library service functions in \\MQL5\\Include\\DoEasy\\Services\ **DELib.mqh**:

```
//+------------------------------------------------------------------+
//|                                                        DELib.mqh |
//|                        Copyright 2020, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2020, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property strict  // Necessary for mql4
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include "..\Defines.mqh"
#include "Message.mqh"
#include "TimerCounter.mqh"
#include "Pause.mqh"
#include "Colors.mqh"
//+------------------------------------------------------------------+
//| Service functions                                                |
//+------------------------------------------------------------------+
```

I will not need the class here but I will start using it later when creating the graphical element object descendant classes.

Each graphical object features at least its size and the coordinates of its location on the chart. In addition, our objects are endowed with many properties that can be changed while the program is running. But if we relaunch the program or switch the timeframe, all changes made to graphical objects while the program is running are reset. To let each object remember the state of its properties, we need to save them outside. In this case, after restarting the program, all graphical objects constructed and changed during its operation read their appropriate properties (relevant as of the reset moment) from the file and restore them. To achieve this, we need to add two methods to the graphical element object class — the method for writing the object properties to the file and the one for reading object properties from the file.

In order to read and write the object properties, I will save the object properties to the structure, while the structure can be both saved to the file and read from it using the [StructToCharArray()](https://www.mql5.com/en/docs/convert/structtochararray) and [CharArrayToStruct()](https://www.mql5.com/en/docs/convert/chararraytostruct) standard functions.

Each graphical object is to feature the methods for saving properties to the file and reading them from the file since each graphical object based on the canvas is to be a descendant from the graphical element object, in which the methods are to be set. Thus, if the object is composite (i.e. it consists of other objects based on the graphical element), we can restore the states of all of its subordinate objects one by one according to the object index in the list of subordinate objects (the index is stored in the CANV\_ELEMENT\_PROP\_NUM constant of the ENUM\_CANV\_ELEMENT\_PROP\_INTEGER enumeration of the element object properties).

In this article, I will not save the properties to the file/read them from the file since this should be done from the graphical object collection class. I will consider it later — after creating the graphical element. Anyway, the writing and reading methods will be added here.

Since the graphical element is a descendant of the base object of all CGBaseObj library graphical objects, first let's set the virtual method for creating the structure out of the object properties and the virtual method for restoring the object properties from the structure in the protected section of the object class file (\\MQL5\\Include\\DoEasy\\Objects\\Graph\ **GBaseObj.mqh**):

```
protected:
   string            m_name_prefix;                      // Object name prefix
   string            m_name;                             // Object name
   long              m_chart_id;                         // Chart ID
   int               m_subwindow;                        // Subwindow index
   int               m_shift_y;                          // Subwindow Y coordinate shift
   int               m_type;                             // Object type

//--- Create (1) the object structure and (2) the object from the structure
   virtual bool      ObjectToStruct(void)                      { return true; }
   virtual void      StructToObject(void){;}

public:
```

These methods do nothing here — they should be redefined in the class descendants. The closest descendant of the class is the class of the graphical element object in \\MQL5\\Include\\DoEasy\\Objects\\Graph\ **GCnvElement.mqh**. Declare the samevirtual methods in its protected section:

```
//+------------------------------------------------------------------+
//| Class of the base object of the library graphical objects        |
//+------------------------------------------------------------------+
class CGCnvElement : public CGBaseObj
  {
protected:
   CCanvas           m_canvas;                                 // CCanvas class object
   CPause            m_pause;                                  // Pause class object
//--- Return the cursor position relative to the (1) entire element and (2) the element active area
   bool              CursorInsideElement(const int x,const int y);
   bool              CursorInsideActiveArea(const int x,const int y);
//--- Create (1) the object structure and (2) the object from the structure
   virtual bool      ObjectToStruct(void);
   virtual void      StructToObject(void);

private:
```

In the private section, declare the structure for storing all object properties, the object with the structure type and the object structure array:

```
private:
   struct SData
     {
      //--- Object integer properties
      int            id;                                       // Element ID
      int            type;                                     // Graphical element type
      int            number;                                   // Element index in the list
      long           chart_id;                                 // Chart ID
      int            subwindow;                                // Chart subwindow index
      int            coord_x;                                  // Form's X coordinate on the chart
      int            coord_y;                                  // Form's Y coordinate on the chart
      int            width;                                    // Element width
      int            height;                                   // Element height
      int            edge_right;                               // Element right border
      int            edge_bottom;                              // Element bottom border
      int            act_shift_left;                           // Active area offset from the left edge of the element
      int            act_shift_top;                            // Active area offset from the top edge of the element
      int            act_shift_right;                          // Active area offset from the right edge of the element
      int            act_shift_bottom;                         // Active area offset from the bottom edge of the element
      uchar          opacity;                                  // Element opacity
      color          color_bg;                                 // Element background color
      bool           movable;                                  // Element moveability flag
      bool           active;                                   // Element activity flag
      int            coord_act_x;                              // X coordinate of the element active area
      int            coord_act_y;                              // Y coordinate of the element active area
      int            coord_act_right;                          // Right border of the element active area
      int            coord_act_bottom;                         // Bottom border of the element active area
      //--- Object real properties

      //--- Object string properties
      uchar          name_obj[64];                             // Graphical element object name
      uchar          name_res[64];                             // Graphical resource name
     };
   SData             m_struct_obj;                             // Object structure
   uchar             m_uchar_array[];                          // uchar array of the object structure

   long              m_long_prop[ORDER_PROP_INTEGER_TOTAL];    // Integer properties
   double            m_double_prop[ORDER_PROP_DOUBLE_TOTAL];   // Real properties
   string            m_string_prop[ORDER_PROP_STRING_TOTAL];   // String properties
```

In the public section of the class, declare the methods of writing and reading object properties from the file:

```
public:
//--- Set object (1) integer, (2) real and (3) string properties
   void              SetProperty(ENUM_CANV_ELEMENT_PROP_INTEGER property,long value)   { this.m_long_prop[property]=value;                   }
   void              SetProperty(ENUM_CANV_ELEMENT_PROP_DOUBLE property,double value)  { this.m_double_prop[this.IndexProp(property)]=value; }
   void              SetProperty(ENUM_CANV_ELEMENT_PROP_STRING property,string value)  { this.m_string_prop[this.IndexProp(property)]=value; }
//--- Return object (1) integer, (2) real and (3) string property from the properties array
   long              GetProperty(ENUM_CANV_ELEMENT_PROP_INTEGER property)        const { return this.m_long_prop[property];                  }
   double            GetProperty(ENUM_CANV_ELEMENT_PROP_DOUBLE property)         const { return this.m_double_prop[this.IndexProp(property)];}
   string            GetProperty(ENUM_CANV_ELEMENT_PROP_STRING property)         const { return this.m_string_prop[this.IndexProp(property)];}

//--- Return the flag of the object supporting this property
   virtual bool      SupportProperty(ENUM_CANV_ELEMENT_PROP_INTEGER property)          { return true; }
   virtual bool      SupportProperty(ENUM_CANV_ELEMENT_PROP_DOUBLE property)           { return false;}
   virtual bool      SupportProperty(ENUM_CANV_ELEMENT_PROP_STRING property)           { return true; }

//--- Compare CGCnvElement objects with each other by all possible properties (for sorting the lists by a specified object property)
   virtual int       Compare(const CObject *node,const int mode=0) const;
//--- Compare CGCnvElement objects with each other by all properties (to search equal objects)
   bool              IsEqual(CGCnvElement* compared_obj) const;

//--- (1) Save the object to file and (2) upload the object from the file
   virtual bool      Save(const int file_handle);
   virtual bool      Load(const int file_handle);

//--- Create the element
```

Since the object has no real properties, the virtual method returning the flag of supporting real properties by the object should return false.

Implement declared methods outside the class body.

**The method creating the object structure from its properties:**

```
//+------------------------------------------------------------------+
//| Create the object structure                                      |
//+------------------------------------------------------------------+
bool CGCnvElement::ObjectToStruct(void)
  {
//--- Save integer properties
   this.m_struct_obj.id=(int)this.GetProperty(CANV_ELEMENT_PROP_ID);                            // Element ID
   this.m_struct_obj.type=(int)this.GetProperty(CANV_ELEMENT_PROP_TYPE);                        // Graphical element type
   this.m_struct_obj.number=(int)this.GetProperty(CANV_ELEMENT_PROP_NUM);                       // Eleemnt ID in the list
   this.m_struct_obj.chart_id=this.GetProperty(CANV_ELEMENT_PROP_CHART_ID);                     // Chart ID
   this.m_struct_obj.subwindow=(int)this.GetProperty(CANV_ELEMENT_PROP_WND_NUM);                // Chart subwindow index
   this.m_struct_obj.coord_x=(int)this.GetProperty(CANV_ELEMENT_PROP_COORD_X);                  // Form's X coordinate on the chart
   this.m_struct_obj.coord_y=(int)this.GetProperty(CANV_ELEMENT_PROP_COORD_Y);                  // Form's Y coordinate on the chart
   this.m_struct_obj.width=(int)this.GetProperty(CANV_ELEMENT_PROP_WIDTH);                      // Element width
   this.m_struct_obj.height=(int)this.GetProperty(CANV_ELEMENT_PROP_HEIGHT);                    // Element height
   this.m_struct_obj.edge_right=(int)this.GetProperty(CANV_ELEMENT_PROP_RIGHT);                 // Element right edge
   this.m_struct_obj.edge_bottom=(int)this.GetProperty(CANV_ELEMENT_PROP_BOTTOM);               // Element bottom edge
   this.m_struct_obj.act_shift_left=(int)this.GetProperty(CANV_ELEMENT_PROP_ACT_SHIFT_LEFT);    // Active area offset from the left edge of the element
   this.m_struct_obj.act_shift_top=(int)this.GetProperty(CANV_ELEMENT_PROP_ACT_SHIFT_TOP);      // Active area offset from the top edge of the element
   this.m_struct_obj.act_shift_right=(int)this.GetProperty(CANV_ELEMENT_PROP_ACT_SHIFT_RIGHT);  // Active area offset from the right edge of the element
   this.m_struct_obj.act_shift_bottom=(int)this.GetProperty(CANV_ELEMENT_PROP_ACT_SHIFT_BOTTOM);// Active area offset from the bottom edge of the element
   this.m_struct_obj.opacity=(uchar)this.GetProperty(CANV_ELEMENT_PROP_OPACITY);                // Element opacity
   this.m_struct_obj.color_bg=(color)this.GetProperty(CANV_ELEMENT_PROP_COLOR_BG);              // Element background color
   this.m_struct_obj.movable=(bool)this.GetProperty(CANV_ELEMENT_PROP_MOVABLE);                 // Element moveability flag
   this.m_struct_obj.active=(bool)this.GetProperty(CANV_ELEMENT_PROP_ACTIVE);                   // Element activity flag
   this.m_struct_obj.coord_act_x=(int)this.GetProperty(CANV_ELEMENT_PROP_COORD_ACT_X);          // X coordinate of the element active area
   this.m_struct_obj.coord_act_y=(int)this.GetProperty(CANV_ELEMENT_PROP_COORD_ACT_Y);          // Y coordinate of the element active area
   this.m_struct_obj.coord_act_right=(int)this.GetProperty(CANV_ELEMENT_PROP_ACT_RIGHT);        // Right border of the element active area
   this.m_struct_obj.coord_act_bottom=(int)this.GetProperty(CANV_ELEMENT_PROP_ACT_BOTTOM);      // Bottom border of the element active area
//--- Save real properties

//--- Save string properties
   ::StringToCharArray(this.GetProperty(CANV_ELEMENT_PROP_NAME_OBJ),this.m_struct_obj.name_obj);// Graphical element object name
   ::StringToCharArray(this.GetProperty(CANV_ELEMENT_PROP_NAME_RES),this.m_struct_obj.name_res);// Graphical resource name
   //--- Save the structure to the uchar array
   ::ResetLastError();
   if(!::StructToCharArray(this.m_struct_obj,this.m_uchar_array))
     {
      ::Print(DFUN,CMessage::Text(MSG_LIB_SYS_FAILED_SAVE_OBJ_STRUCT_TO_UARRAY),(string)::GetLastError());
      return false;
     }
   return true;
  }
//+------------------------------------------------------------------+
```

Here all is simple: each integer structure field receives the appropriate object property, while object string properties are saved in the appropriate uchar array of the structure. Next, simply save the newly created object property structure to the uchar array using [StructToCharArray()](https://www.mql5.com/en/docs/convert/structtochararray).

If failed to save the structure to the array, inform of the error and return false. As a result, return true.

**The method restoring the object properties from the structure:**

```
//+------------------------------------------------------------------+
//| Create the object from the structure                             |
//+------------------------------------------------------------------+
void CGCnvElement::StructToObject(void)
  {
//--- Save integer properties
   this.SetProperty(CANV_ELEMENT_PROP_ID,this.m_struct_obj.id);                                 // Element ID
   this.SetProperty(CANV_ELEMENT_PROP_TYPE,this.m_struct_obj.type);                             // Graphical element type
   this.SetProperty(CANV_ELEMENT_PROP_NUM,this.m_struct_obj.number);                            // Element index in the list
   this.SetProperty(CANV_ELEMENT_PROP_CHART_ID,this.m_struct_obj.chart_id);                     // Chart ID
   this.SetProperty(CANV_ELEMENT_PROP_WND_NUM,this.m_struct_obj.subwindow);                     // Chart subwindow index
   this.SetProperty(CANV_ELEMENT_PROP_COORD_X,this.m_struct_obj.coord_x);                       // Form's X coordinate on the chart
   this.SetProperty(CANV_ELEMENT_PROP_COORD_Y,this.m_struct_obj.coord_y);                       // Form's Y coordinate on the chart
   this.SetProperty(CANV_ELEMENT_PROP_WIDTH,this.m_struct_obj.width);                           // Element width
   this.SetProperty(CANV_ELEMENT_PROP_HEIGHT,this.m_struct_obj.height);                         // Element height
   this.SetProperty(CANV_ELEMENT_PROP_RIGHT,this.m_struct_obj.edge_right);                      // Element right edge
   this.SetProperty(CANV_ELEMENT_PROP_BOTTOM,this.m_struct_obj.edge_bottom);                    // Element bottom edge
   this.SetProperty(CANV_ELEMENT_PROP_ACT_SHIFT_LEFT,this.m_struct_obj.act_shift_left);         // Active area offset from the left edge of the element
   this.SetProperty(CANV_ELEMENT_PROP_ACT_SHIFT_TOP,this.m_struct_obj.act_shift_top);           // Active area offset from the upper edge of the element
   this.SetProperty(CANV_ELEMENT_PROP_ACT_SHIFT_RIGHT,this.m_struct_obj.act_shift_right);       // Active area offset from the right edge of the element
   this.SetProperty(CANV_ELEMENT_PROP_ACT_SHIFT_BOTTOM,this.m_struct_obj.act_shift_bottom);     // Active area offset from the bottom edge of the element
   this.SetProperty(CANV_ELEMENT_PROP_OPACITY,this.m_struct_obj.opacity);                       // Element opacity
   this.SetProperty(CANV_ELEMENT_PROP_COLOR_BG,this.m_struct_obj.color_bg);                     // Element background color
   this.SetProperty(CANV_ELEMENT_PROP_MOVABLE,this.m_struct_obj.movable);                       // Element moveability flag
   this.SetProperty(CANV_ELEMENT_PROP_ACTIVE,this.m_struct_obj.active);                         // Element activity flag
   this.SetProperty(CANV_ELEMENT_PROP_COORD_ACT_X,this.m_struct_obj.coord_act_x);               // X coordinate of the element active area
   this.SetProperty(CANV_ELEMENT_PROP_COORD_ACT_Y,this.m_struct_obj.coord_act_y);               // Y coordinate of the element active area
   this.SetProperty(CANV_ELEMENT_PROP_ACT_RIGHT,this.m_struct_obj.coord_act_right);             // Right border of the element active area
   this.SetProperty(CANV_ELEMENT_PROP_ACT_BOTTOM,this.m_struct_obj.coord_act_bottom);           // Bottom border of the element active area
//--- Save real properties

//--- Save string properties
   this.SetProperty(CANV_ELEMENT_PROP_NAME_OBJ,::CharArrayToString(this.m_struct_obj.name_obj));// Graphical element object name
   this.SetProperty(CANV_ELEMENT_PROP_NAME_RES,::CharArrayToString(this.m_struct_obj.name_res));// Graphical resource name
  }
//+------------------------------------------------------------------+
```

Here, each integer object property receives the value from the appropriate structure field, while the contents of the appropriate uchar array structure is read to the object string properties using [CharArrayToString()](https://www.mql5.com/en/docs/convert/chararraytostring).

**The method saving the object to the file:**

```
//+------------------------------------------------------------------+
//| Save the object to the file                                      |
//+------------------------------------------------------------------+
bool CGCnvElement::Save(const int file_handle)
  {
   if(!this.ObjectToStruct())
     {
      ::Print(DFUN,CMessage::Text(MSG_LIB_SYS_FAILED_CREATE_OBJ_STRUCT));
      return false;
     }
   if(::FileWriteArray(file_handle,this.m_uchar_array)==0)
     {
      ::Print(DFUN,CMessage::Text(MSG_LIB_SYS_FAILED_WRITE_UARRAY_TO_FILE));
      return false;
     }
   return true;
  }
//+------------------------------------------------------------------+
```

The method receives the handle of the file the object properties should be saved to. The object properties are then saved in the structure using the ObjectToStruct() method considered above. The uchar array created when constructing the structure is written to the file using [FileWriteArray()](https://www.mql5.com/en/docs/files/filewritearray) and true is returned. In case of a failure, the method displays the error message in the journal and returns false.

**The method uploading the object properties from the file:**

```
//+------------------------------------------------------------------+
//| Upload the object from the file                                  |
//+------------------------------------------------------------------+
bool CGCnvElement::Load(const int file_handle)
  {
   if(::FileReadArray(file_handle,this.m_uchar_array)==0)
     {
      ::Print(DFUN,CMessage::Text(MSG_LIB_SYS_FAILED_LOAD_UARRAY_FROM_FILE));
      return false;
     }
   if(!::CharArrayToStruct(this.m_struct_obj,this.m_uchar_array))
     {
      ::Print(DFUN,CMessage::Text(MSG_LIB_SYS_FAILED_CREATE_OBJ_STRUCT_FROM_UARRAY));
      return false;
     }
   this.StructToObject();
   return true;
  }
//+------------------------------------------------------------------+
```

The method receives the file handle featuring saved object properties. Next, the object properties from the file are uploaded to the uchar array using [FileReadArray()](https://www.mql5.com/en/docs/files/filereadarray). The uploaded properties are copied to the structure using [CharArrayToStruct()](https://www.mql5.com/en/docs/convert/chararraytostruct). The structure filled in from the file is set in the object properties using the above mentioned StructToObject() method and true is returned. If reading from the file or copying the obtained array to the structure ends with an error, the method informs of that and returns false.

The block of methods for a simplified access to object properties receives the methods for returning the right and bottom element edge, the methods for setting and returning the element background color, as well as the methods for returning the element ID and its index in the list of elements in the composite object:

```
//+------------------------------------------------------------------+
//| The methods of simplified access to object properties            |
//+------------------------------------------------------------------+
//--- Set the (1) X, (2) Y coordinates, (3) element width, (4) height, (5) right (6) and bottom edge,
   bool              SetCoordX(const int coord_x);
   bool              SetCoordY(const int coord_y);
   bool              SetWidth(const int width);
   bool              SetHeight(const int height);
   void              SetRightEdge(void)                        { this.SetProperty(CANV_ELEMENT_PROP_RIGHT,this.RightEdge());           }
   void              SetBottomEdge(void)                       { this.SetProperty(CANV_ELEMENT_PROP_BOTTOM,this.BottomEdge());         }
//--- Set the shift of the (1) left, (2) top, (3) right, (4) bottom edge of the active area relative to the element,
//--- (5) all shifts of the active area edges relative to the element, (6) the element background color and (7) the element opacity
   void              SetActiveAreaLeftShift(const int value)   { this.SetProperty(CANV_ELEMENT_PROP_ACT_SHIFT_LEFT,fabs(value));       }
   void              SetActiveAreaRightShift(const int value)  { this.SetProperty(CANV_ELEMENT_PROP_ACT_SHIFT_RIGHT,fabs(value));      }
   void              SetActiveAreaTopShift(const int value)    { this.SetProperty(CANV_ELEMENT_PROP_ACT_SHIFT_TOP,fabs(value));        }
   void              SetActiveAreaBottomShift(const int value) { this.SetProperty(CANV_ELEMENT_PROP_ACT_SHIFT_BOTTOM,fabs(value));     }
   void              SetActiveAreaShift(const int left_shift,const int bottom_shift,const int right_shift,const int top_shift);
   void              SetColorBG(const color colour)            { this.SetProperty(CANV_ELEMENT_PROP_COLOR_BG,colour);                  }
   void              SetOpacity(const uchar value,const bool redraw=false);

//--- Return the shift (1) of the left, (2) right, (3) top and (4) bottom edge of the element active area
   int               ActiveAreaLeftShift(void)           const { return (int)this.GetProperty(CANV_ELEMENT_PROP_ACT_SHIFT_LEFT);       }
   int               ActiveAreaRightShift(void)          const { return (int)this.GetProperty(CANV_ELEMENT_PROP_ACT_SHIFT_RIGHT);      }
   int               ActiveAreaTopShift(void)            const { return (int)this.GetProperty(CANV_ELEMENT_PROP_ACT_SHIFT_TOP);        }
   int               ActiveAreaBottomShift(void)         const { return (int)this.GetProperty(CANV_ELEMENT_PROP_ACT_SHIFT_BOTTOM);     }
//--- Return the coordinate (1) of the left, (2) right, (3) top and (4) bottom edge of the element active area
   int               ActiveAreaLeft(void)                const { return int(this.CoordX()+this.ActiveAreaLeftShift());                 }
   int               ActiveAreaRight(void)               const { return int(this.RightEdge()-this.ActiveAreaRightShift());             }
   int               ActiveAreaTop(void)                 const { return int(this.CoordY()+this.ActiveAreaTopShift());                  }
   int               ActiveAreaBottom(void)              const { return int(this.BottomEdge()-this.ActiveAreaBottomShift());           }
//--- Return (1) the background color, (2) the opacity, coordinate (3) of the right and (4) bottom element edge
   color             ColorBG(void)                       const { return (color)this.GetProperty(CANV_ELEMENT_PROP_COLOR_BG);           }
   uchar             Opacity(void)                       const { return (uchar)this.GetProperty(CANV_ELEMENT_PROP_OPACITY);            }
   int               RightEdge(void)                     const { return this.CoordX()+this.m_canvas.Width();                           }
   int               BottomEdge(void)                    const { return this.CoordY()+this.m_canvas.Height();                          }
//--- Return the (1) X, (2) Y coordinates, (3) element width and (4) height,
   int               CoordX(void)                        const { return (int)this.GetProperty(CANV_ELEMENT_PROP_COORD_X);              }
   int               CoordY(void)                        const { return (int)this.GetProperty(CANV_ELEMENT_PROP_COORD_Y);              }
   int               Width(void)                         const { return (int)this.GetProperty(CANV_ELEMENT_PROP_WIDTH);                }
   int               Height(void)                        const { return (int)this.GetProperty(CANV_ELEMENT_PROP_HEIGHT);               }
//--- Return the element (1) moveability and (2) activity flag
   bool              Movable(void)                       const { return (bool)this.GetProperty(CANV_ELEMENT_PROP_MOVABLE);             }
   bool              Active(void)                        const { return (bool)this.GetProperty(CANV_ELEMENT_PROP_ACTIVE);              }
//--- Return (1) the object name, (2) the graphical resource name, (3) the chart ID and (4) the chart subwindow index
   string            NameObj(void)                       const { return this.GetProperty(CANV_ELEMENT_PROP_NAME_OBJ);                  }
   string            NameRes(void)                       const { return this.GetProperty(CANV_ELEMENT_PROP_NAME_RES);                  }
   long              ChartID(void)                       const { return this.GetProperty(CANV_ELEMENT_PROP_CHART_ID);                  }
   int               WindowNum(void)                     const { return (int)this.GetProperty(CANV_ELEMENT_PROP_WND_NUM);              }
//--- Return (1) the element ID and (2) index in the list
   int               ID(void)                            const { return (int)this.GetProperty(CANV_ELEMENT_PROP_ID);                   }
   int               Number(void)                        const { return (int)this.GetProperty(CANV_ELEMENT_PROP_NUM);                  }

//+------------------------------------------------------------------+
```

All these methods simply return the appropriate element object property.

### Methods of working with primitives

[The CCanvas class](https://www.mql5.com/en/docs/standardlibrary/canvasgraphics/ccanvas) provides ample opportunities for drawing various graphical primitives on the canvas. We are able to either read the color of each pixel or set the required color and transparency to it. In addition to simply setting a color to a pixel, the class provides the tools for drawing different figures either pixel by pixel (without smoothing) or with the help of various smoothing methods.

The graphical element object class is to provide users the access to the CCanvas class drawing methods. Our methods will only slightly simplify calling the CCanvas class methods. The simplification is in the fact that the color is set in the usual way — by specifying the necessary color in the color format and setting the color opacity degree (0 — transparent, 255 — completely opaque), whereas the CCanvas class methods "ask" to specify the color immediately in the uint ARGB format, which is just a number. Not everyone is comfortable with specifying the desired color in this format (semi-transparent gray: 0x7F7F7F7F). In subsequent classes that are to be inherited from the graphical element object, I will expand the range of drawing features by adding the convenient functionality inherent in each created class. In the same class, which is basic for creating the remaining graphical objects, the drawing methods should be simple and straightforward.

The block of methods for simplified access to object properties is followed by new code blocks. I tried to distribute them according to their purpose.

Data receiving methods starts with "Get", while data setting methods begin with "Set".

**The method receiving the color of the point with specified coordinates:**

```
//+------------------------------------------------------------------+
//| The methods of receiving raster data                             |
//+------------------------------------------------------------------+
//--- Get a color of the dot with the specified coordinates
   uint              GetPixel(const int x,const int y)   const { return this.m_canvas.PixelGet(x,y);                                   }

//+------------------------------------------------------------------+
```

The result of calling the [PixelGet()](https://www.mql5.com/en/docs/standardlibrary/canvasgraphics/ccanvas/ccanvaspixelget) method of the CCanvas class is returned here. The method returns the color in ARGB format.

**The methods of filling, clearing and updating raster data:**

```
//+------------------------------------------------------------------+
//| The methods of filling, clearing and updating raster data        |
//+------------------------------------------------------------------+
//--- Clear the element filling it with color and opacity
   void              Erase(const color colour,const uchar opacity,const bool redraw=false);
//--- Clear the element completely
   void              Erase(const bool redraw=false);
//--- Update the element
   void              Update(const bool redraw=false)           { this.m_canvas.Update(redraw);                                         }

//+------------------------------------------------------------------+
```

The **Update()** method simply updates the object and chart using the [Update()](https://www.mql5.com/en/docs/standardlibrary/canvasgraphics/ccanvas/ccanvasupdate) method of the CCanvas class.

**The Erase() methods are implemented outside the class body:**

```
//+------------------------------------------------------------------+
//| Clear the element filling it with color and opacity              |
//+------------------------------------------------------------------+
void CGCnvElement::Erase(const color colour,const uchar opacity,const bool redraw=false)
  {
   this.m_canvas.Erase(::ColorToARGB(colour,opacity));
   if(redraw)
      ::ChartRedraw(this.m_chart_id);
  }
//+------------------------------------------------------------------+
//| Clear the element completely                                     |
//+------------------------------------------------------------------+
void CGCnvElement::Erase(const bool redraw=false)
  {
   this.m_canvas.Erase(NULL_COLOR);
   if(redraw)
      ::ChartRedraw(this.m_chart_id);
  }
//+------------------------------------------------------------------+
```

These are two overloaded methods.

In the first one, we pass the required color and opacity used to fill the entire element with the help of the [Erase()](https://www.mql5.com/en/docs/standardlibrary/canvasgraphics/ccanvas/ccanvaserase) method of the CCanvas class. Please note that the method applies the color and opacity level passed to the Erase() method of the CCanvas class using conversion of their values to ARGB format by the [ColorToARGB()](https://www.mql5.com/en/docs/convert/colortoargb) function. I will do that in all drawing methods.

In the second method, we simply fill the entire background with a completely transparent black color. Its value was previously defined using the NULL\_COLOR macro substitution.

Each of the methods receives the flag indicating the necessity to redraw the chart. If the flag is set, the chart is re-drawn.

Next is the **block of methods for drawing primitives without smoothing**. All methods are identical and call the corresponding methods of the CCanvas class. These methods receive the parameters specified in the method arguments and the color converted into ARGB format from the color and transparency values passed to the methods:

```
//+------------------------------------------------------------------+
//| The methods of drawing primitives without smoothing              |
//+------------------------------------------------------------------+
//--- Set the color of the dot with the specified coordinates
   void              SetPixel(const int x,const int y,const color clr,const uchar opacity=255)
                       { this.m_canvas.PixelSet(x,y,::ColorToARGB(clr,opacity));                                                       }

//--- Draw a segment of a vertical line
   void              DrawLineVertical(const int x,                // X coordinate of the segment
                                      const int y1,               // Y coordinate of the segment first point
                                      const int y2,               // Y coordinate of the segment second point
                                      const color clr,            // Color
                                      const uchar opacity=255)    // Opacity
                       { this.m_canvas.LineVertical(x,y1,y2,::ColorToARGB(clr,opacity));                                               }

//--- Draw a segment of a horizontal line
   void              DrawLineHorizontal(const int x1,             // X coordinate of the segment first point
                                        const int x2,             // X coordinate of the segment second point
                                        const int y,              // Segment Y coordinate
                                        const color clr,          // Color
                                        const uchar opacity=255)  // Opacity
                       { this.m_canvas.LineHorizontal(x1,x2,y,::ColorToARGB(clr,opacity));                                             }

//--- Draw a segment of a freehand line
   void              DrawLine(const int x1,                       // X coordinate of the segment first point
                              const int y1,                       // Y coordinate of the segment first point
                              const int x2,                       // X coordinate of the segment second point
                              const int y2,                       // Y coordinate of the segment second point
                              const color clr,                    // Color
                              const uchar opacity=255)            // Opacity
                       { this.m_canvas.Line(x1,y1,x2,y2,::ColorToARGB(clr,opacity));                                                   }

//--- Draw a polyline
   void              DrawPolyline(int &array_x[],                 // Array with the X coordinates of polyline points
                                  int & array_y[],                // Array with the Y coordinates of polyline points
                                  const color clr,                // Color
                                  const uchar opacity=255)        // Opacity
                       { this.m_canvas.Polyline(array_x,array_y,::ColorToARGB(clr,opacity));                                           }

//--- Draw a polygon
   void              DrawPolygon(int &array_x[],                  // Array with the X coordinates of polygon points
                                 int &array_y[],                  // Array with the Y coordinates of polygon points
                                 const color clr,                 // Color
                                 const uchar opacity=255)         // Opacity
                       { this.m_canvas.Polygon(array_x,array_y,::ColorToARGB(clr,opacity));                                            }

//--- Draw a rectangle using two points
   void              DrawRectangle(const int x1,                  // X coordinate of the first point defining the rectangle
                                   const int y1,                  // Y coordinate of the first point defining the rectangle
                                   const int x2,                  // X coordinate of the second point defining the rectangle
                                   const int y2,                  // Y coordinate of the second point defining the rectangle
                                   const color clr,               // color
                                   const uchar opacity=255)       // Opacity
                       { this.m_canvas.Rectangle(x1,y1,x2,y2,::ColorToARGB(clr,opacity));                                              }

//--- Draw a circle
   void              DrawCircle(const int x,                      // X coordinate of the circle center
                                const int y,                      // Y coordinate of the circle center
                                const int r,                      // Circle radius
                                const color clr,                  // Color
                                const uchar opacity=255)          // Opacity
                       { this.m_canvas.Circle(x,y,r,::ColorToARGB(clr,opacity));                                                       }

//--- Draw a triangle
   void              DrawTriangle(const int x1,                   // X coordinate of the triangle first vertex
                                  const int y1,                   // Y coordinate of the triangle first vertex
                                  const int x2,                   // X coordinate of the triangle second vertex
                                  const int y2,                   // Y coordinate of the triangle second vertex
                                  const int x3,                   // X coordinate of the triangle third vertex
                                  const int y3,                   // Y coordinate of the triangle third vertex
                                  const color clr,                // Color
                                  const uchar opacity=255)        // Opacity
                       { m_canvas.Triangle(x1,y1,x2,y2,x3,y3,::ColorToARGB(clr,opacity));                                              }

//--- Draw an ellipse using two points
   void              DrawEllipse(const int x1,                    // X coordinate of the first point defining the ellipse
                                 const int y1,                    // Y coordinate of the first point defining the ellipse
                                 const int x2,                    // X coordinate of the second point defining the ellipse
                                 const int y2,                    // Y coordinate of the second point defining the ellipse
                                 const color clr,                 // Color
                                 const uchar opacity=255)         // Opacity
                       { this.m_canvas.Ellipse(x1,y1,x2,y2,::ColorToARGB(clr,opacity));                                                }

//--- Draw an arc of an ellipse inscribed in a rectangle with corners at (x1,y1) and (x2,y2).
//--- The arc boundaries are clipped by lines from the center of the ellipse, which extend to two points with coordinates (x3,y3) and (x4,y4)
   void              DrawArc(const int x1,                        // X coordinate of the top left corner forming the rectangle
                             const int y1,                        // Y coordinate of the top left corner forming the rectangle
                             const int x2,                        // X coordinate of the bottom right corner forming the rectangle
                             const int y2,                        // Y coordinate of the bottom right corner forming the rectangle
                             const int x3,                        // X coordinate of the first point, to which a line from the rectangle center is drawn in order to obtain the arc boundary
                             const int y3,                        // Y coordinate of the first point, to which a line from the rectangle center is drawn in order to obtain the arc boundary
                             const int x4,                        // X coordinate of the second point, to which a line from the rectangle center is drawn in order to obtain the arc boundary
                             const int y4,                        // Y coordinate of the second point, to which a line from the rectangle center is drawn in order to obtain the arc boundary
                             const color clr,                     // Color
                             const uchar opacity=255)             // Opacity
                       { m_canvas.Arc(x1,y1,x2,y2,x3,y3,x4,y4,::ColorToARGB(clr,opacity));                                             }

//--- Draw a filled sector of an ellipse inscribed in a rectangle with corners at (x1,y1) and (x2,y2).
//--- The sector boundaries are clipped by lines from the center of the ellipse, which extend to two points with coordinates (x3,y3) and (x4,y4)
   void              DrawPie(const int x1,                        // X coordinate of the upper left corner of the rectangle
                             const int y1,                        // Y coordinate of the upper left corner of the rectangle
                             const int x2,                        // X coordinate of the bottom right corner of the rectangle
                             const int y2,                        // Y coordinate of the bottom right corner of the rectangle
                             const int x3,                        // X coordinate of the first point to find the arc boundaries
                             const int y3,                        // Y coordinate of the first point to find the arc boundaries
                             const int x4,                        // X coordinate of the second point to find the arc boundaries
                             const int y4,                        // Y coordinate of the second point to find the arc boundaries
                             const color clr,                     // Line color
                             const color fill_clr,                // Fill color
                             const uchar opacity=255)             // Opacity
                       { this.m_canvas.Pie(x1,y1,x2,y2,x3,y3,x4,y4,::ColorToARGB(clr,opacity),ColorToARGB(fill_clr,opacity));          }

//+------------------------------------------------------------------+
```

**The block of methods for drawing filled primitives without smoothing** **:**

```
//+------------------------------------------------------------------+
//| The methods of drawing filled primitives without smoothing       |
//+------------------------------------------------------------------+
//--- Fill in the area
   void              Fill(const int x,                            // X coordinate of the filling start point
                          const int y,                            // Y coordinate of the filling start point
                          const color clr,                        // Color
                          const uchar opacity=255,                // Opacity
                          const uint threshould=0)                // Threshold
                       { this.m_canvas.Fill(x,y,::ColorToARGB(clr,opacity),threshould);                                                }

//--- Draw a filled rectangle
   void              DrawRectangleFill(const int x1,              // X coordinate of the first point defining the rectangle
                                       const int y1,              // Y coordinate of the first point defining the rectangle
                                       const int x2,              // X coordinate of the second point defining the rectangle
                                       const int y2,              // Y coordinate of the second point defining the rectangle
                                       const color clr,           // Color
                                       const uchar opacity=255)   // Opacity
                       { this.m_canvas.FillRectangle(x1,y1,x2,y2,::ColorToARGB(clr,opacity));                                          }

//--- Draw a filled circle
   void              DrawCircleFill(const int x,                  // X coordinate of the circle center
                                    const int y,                  // Y coordinate of the circle center
                                    const int r,                  // Circle radius
                                    const color clr,              // Color
                                    const uchar opacity=255)      // Opacity
                       { this.m_canvas.FillCircle(x,y,r,::ColorToARGB(clr,opacity));                                                   }

//--- Draw a filled triangle
   void              DrawTriangleFill(const int         x1,      // X coordinate of the triangle first vertex
                                      const int         y1,      // Y coordinate of the triangle first vertex
                                      const int         x2,      // X coordinate of the triangle second vertex
                                      const int         y2,      // Y coordinate of the triangle second vertex
                                      const int         x3,      // X coordinate of the triangle third vertex
                                      const int         y3,      // Y coordinate of the triangle third vertex
                                      const color clr,           // Color
                                      const uchar opacity=255)   // Opacity
                       { this.m_canvas.FillTriangle(x1,y1,x2,y2,x3,y3,::ColorToARGB(clr,opacity));                                     }

//--- Draw a filled polygon
   void              DrawPolygonFill(int &array_x[],              // Array with the X coordinates of polygon points
                                     int &array_y[],              // Array with the Y coordinates of polygon points
                                     const color clr,             // Color
                                     const uchar opacity=255)     // Opacity
                       { this.m_canvas.FillPolygon(array_x,array_y,::ColorToARGB(clr,opacity));                                        }

//--- Draw a filled ellipse inscribed in a rectangle with the specified coordinates
   void              DrawEllipseFill(const int x1,                // X coordinate of the top left corner forming the rectangle
                                     const int y1,                // Y coordinate of the top left corner forming the rectangle
                                     const int x2,                // X coordinate of the bottom right corner forming the rectangle
                                     const int y2,                // Y coordinate of the bottom right corner forming the rectangle
                                     const color clr,             // Color
                                     const uchar opacity=255)     // Opacity
                       { this.m_canvas.FillEllipse(x1,y1,x2,y2,::ColorToARGB(clr,opacity));                                            }

//+------------------------------------------------------------------+
```

**The methods of drawing primitives using smoothing:**

```
//+------------------------------------------------------------------+
//| The methods of drawing primitives using smoothing                |
//+------------------------------------------------------------------+
//--- Draw a point using AntiAliasing algorithm
   void              SetPixelAA(const double x,                   // Point X coordinate
                                const double y,                   // Point Y coordinate
                                const color clr,                  // Color
                                const uchar opacity=255)          // Opacity
                       { this.m_canvas.PixelSetAA(x,y,::ColorToARGB(clr,opacity));                                                     }

//--- Draw a segment of a freehand line using AntiAliasing algorithm
   void              DrawLineAA(const int   x1,                   // X coordinate of the segment first point
                                const int   y1,                   // Y coordinate of the segment first point
                                const int   x2,                   // X coordinate of the segment second point
                                const int   y2,                   // Y coordinate of the segment second point
                                const color clr,                  // Color
                                const uchar opacity=255,          // Opacity
                                const uint  style=UINT_MAX)       // Line style is one of the ENUM_LINE_STYLE enumeration values or a custom value
                       { this.m_canvas.LineAA(x1,y1,x2,y2,::ColorToARGB(clr,opacity),style);                                           }

//--- Draw a segment of a freehand line using Wu algorithm
   void              DrawLineWu(const int   x1,                   // X coordinate of the segment first point
                                const int   y1,                   // Y coordinate of the segment first point
                                const int   x2,                   // X coordinate of the segment second point
                                const int   y2,                   // Y coordinate of the segment second point
                                const color clr,                  // Color
                                const uchar opacity=255,          // Opacity
                                const uint  style=UINT_MAX)       // Line style is one of the ENUM_LINE_STYLE enumeration values or a custom value
                       { this.m_canvas.LineWu(x1,y1,x2,y2,::ColorToARGB(clr,opacity),style);                                           }

//--- Draws a segment of a freehand line having a specified width using smoothing algorithm with the preliminary filtration
   void              DrawLineThick(const int   x1,                // X coordinate of the segment first point
                                   const int   y1,                // Y coordinate of the segment first point
                                   const int   x2,                // X coordinate of the segment second point
                                   const int   y2,                // Y coordinate of the segment second point
                                   const int   size,              // Line width
                                   const color clr,               // Color
                                   const uchar opacity=255,       // Opacity
                                   const uint  style=STYLE_SOLID, // Line style is one of the ENUM_LINE_STYLE enumeration values or a custom value
                                   ENUM_LINE_END end_style=LINE_END_ROUND) // Line style is one of the ENUM_LINE_END enumeration values
                       { this.m_canvas.LineThick(x1,y1,x2,y2,::ColorToARGB(clr,opacity),size,style,end_style);                         }

//--- Draw a vertical segment of a freehand line having a specified width using smoothing algorithm with the preliminary filtration
   void              DrawLineThickVertical(const int   x,         // X coordinate of the segment
                                           const int   y1,        // Y coordinate of the segment first point
                                           const int   y2,        // Y coordinate of the segment second point
                                           const int   size,      // Line width
                                           const color clr,       // Color
                                           const uchar opacity=255,// Opacity
                                           const uint  style=STYLE_SOLID,  // Line style is one of the ENUM_LINE_STYLE enumeration values or a custom value
                                           const ENUM_LINE_END end_style=LINE_END_ROUND)  // Line style is one of the ENUM_LINE_END enumeration values
                       { this.m_canvas.LineThickVertical(x,y1,y2,::ColorToARGB(clr,opacity),size,style,end_style);                     }

//--- Draw a horizontal segment of a freehand line having a specified width using smoothing algorithm with the preliminary filtration
   void              DrawLineThickHorizontal(const int   x1,      // X coordinate of the segment first point
                                             const int   x2,      // X coordinate of the segment second point
                                             const int   y,       // Segment Y coordinate
                                             const int   size,    // Line width
                                             const color clr,     // Color
                                             const uchar opacity=255,// Opacity
                                             const uint  style=STYLE_SOLID,  // Line style is one of the ENUM_LINE_STYLE enumeration values or a custom value
                                             const ENUM_LINE_END end_style=LINE_END_ROUND)  // Line style is one of the ENUM_LINE_END enumeration values
                       { this.m_canvas.LineThickHorizontal(x1,x2,y,::ColorToARGB(clr,opacity),size,style,end_style);                   }

//--- Draws a polyline using AntiAliasing algorithm
   void              DrawPolylineAA(int        &array_x[],        // Array with the X coordinates of polyline points
                                    int        &array_y[],        // Array with the Y coordinates of polyline points
                                    const color clr,              // Color
                                    const uchar opacity=255,      // Opacity
                                    const uint  style=UINT_MAX)   // Line style is one of the ENUM_LINE_STYLE enumeration values or a custom value
                       { this.m_canvas.PolylineAA(array_x,array_y,::ColorToARGB(clr,opacity),style);                                   }

//--- Draws a polyline using Wu algorithm
   void              DrawPolylineWu(int        &array_x[],        // Array with the X coordinates of polyline points
                                    int        &array_y[],        // Array with the Y coordinates of polyline points
                                    const color clr,              // Color
                                    const uchar opacity=255,      // Opacity
                                    const uint  style=UINT_MAX)   // Line style is one of the ENUM_LINE_STYLE enumeration values or a custom value
                       { this.m_canvas.PolylineWu(array_x,array_y,::ColorToARGB(clr,opacity),style);                                   }

//--- Draw a polyline with a specified width consecutively using two antialiasing algorithms.
//--- First, individual line segments are smoothed based on Bezier curves.
//--- Then, the raster antialiasing algorithm is applied to the polyline built from these segments to improve the rendering quality
   void              DrawPolylineSmooth(const int   &array_x[],   // Array with the X coordinates of polyline points
                                        const int   &array_y[],   // Array with the Y coordinates of polyline points
                                        const int    size,        // Line width
                                        const color  clr,         // Color
                                        const uchar  opacity=255, // Opacity
                                        const double tension=0.5, // Smoothing parameter value
                                        const double step=10,     // Approximation step
                                        const ENUM_LINE_STYLE style=STYLE_SOLID,// Line style is one of the ENUM_LINE_STYLE enumeration values or a custom value
                                        const ENUM_LINE_END   end_style=LINE_END_ROUND)// Line style is one of the ENUM_LINE_END enumeration values
                       { this.m_canvas.PolylineSmooth(array_x,array_y,::ColorToARGB(clr,opacity),size,style,end_style,tension,step);   }

//--- Draw a polyline having a specified width using smoothing algorithm with the preliminary filtration
   void              DrawPolylineThick(const int     &array_x[],  // Array with the X coordinates of polyline points
                                       const int     &array_y[],  // Array with the Y coordinates of polyline points
                                       const int      size,       // Line width
                                       const color    clr,        // Color
                                       const uchar    opacity=255,// Opacity
                                       const uint     style=STYLE_SOLID,         // Line style is one of the ENUM_LINE_STYLE enumeration values or a custom value
                                       ENUM_LINE_END  end_style=LINE_END_ROUND)  // Line style is one of the ENUM_LINE_END enumeration values
                       { this.m_canvas.PolylineThick(array_x,array_y,::ColorToARGB(clr,opacity),size,style,end_style);                 }

//--- Draw a polygon using AntiAliasing algorithm
   void              DrawPolygonAA(int        &array_x[],         // Array with the X coordinates of polygon points
                                   int        &array_y[],         // Array with the Y coordinates of polygon points
                                   const color clr,               // Color
                                   const uchar opacity=255,       // Opacity
                                   const uint  style=UINT_MAX)    // Line style is one of the ENUM_LINE_STYLE enumeration values or a custom value
                       { this.m_canvas.PolygonAA(array_x,array_y,::ColorToARGB(clr,opacity),style);                                    }

//--- Draw a polygon using Wu algorithm
   void              DrawPolygonWu(int        &array_x[],         // Array with the X coordinates of polygon points
                                   int        &array_y[],         // Array with the Y coordinates of polygon points
                                   const color clr,               // Color
                                   const uchar opacity=255,       // Opacity
                                   const uint  style=UINT_MAX)    // Line style is one of the ENUM_LINE_STYLE enumeration values or a custom value
                       { this.m_canvas.PolygonWu(array_x,array_y,::ColorToARGB(clr,opacity),style);                                    }

//--- Draw a polygon with a specified width consecutively using two smoothing algorithms.
//--- First, individual segments are smoothed based on Bezier curves.
//--- Then, the raster smoothing algorithm is applied to the polygon built from these segments to improve the rendering quality.
   void              DrawPolygonSmooth(int         &array_x[],    // Array with the X coordinates of polyline points
                                       int         &array_y[],    // Array with the Y coordinates of polyline points
                                       const int    size,         // Line width
                                       const color  clr,          // Color
                                       const uchar  opacity=255,  // Opacity
                                       const double tension=0.5,  // Smoothing parameter value
                                       const double step=10,      // Approximation step
                                       const ENUM_LINE_STYLE style=STYLE_SOLID,// Line style is one of the ENUM_LINE_STYLE enumeration values or a custom value
                                       const ENUM_LINE_END   end_style=LINE_END_ROUND)// Line style is one of the ENUM_LINE_END enumeration values
                       { this.m_canvas.PolygonSmooth(array_x,array_y,::ColorToARGB(clr,opacity),size,style,end_style,tension,step);    }

//--- Draw a polygon having a specified width using smoothing algorithm with the preliminary filtration
   void              DrawPolygonThick(const int  &array_x[],      // array with the X coordinates of polygon points
                                      const int  &array_y[],      // array with the Y coordinates of polygon points
                                      const int   size,           // line width
                                      const color clr,            // Color
                                      const uchar opacity=255,    // Opacity
                                      const uint  style=STYLE_SOLID,// line style
                                      ENUM_LINE_END end_style=LINE_END_ROUND) // line ends style
                       { this.m_canvas.PolygonThick(array_x,array_y,::ColorToARGB(clr,opacity),size,style,end_style);                  }

//--- Draw a triangle using AntiAliasing algorithm
   void              DrawTriangleAA(const int   x1,               // X coordinate of the triangle first vertex
                                    const int   y1,               // Y coordinate of the triangle first vertex
                                    const int   x2,               // X coordinate of the triangle second vertex
                                    const int   y2,               // Y coordinate of the triangle second vertex
                                    const int   x3,               // X coordinate of the triangle third vertex
                                    const int   y3,               // Y coordinate of the triangle third vertex
                                    const color clr,              // Color
                                    const uchar opacity=255,      // Opacity
                                    const uint  style=UINT_MAX)   // Line style is one of the ENUM_LINE_STYLE enumeration values or a custom value
                       { this.m_canvas.TriangleAA(x1,y1,x2,y2,x3,y3,::ColorToARGB(clr,opacity),style);                                 }

//--- Draw a triangle using Wu algorithm
   void              DrawTriangleWu(const int   x1,               // X coordinate of the triangle first vertex
                                    const int   y1,               // Y coordinate of the triangle first vertex
                                    const int   x2,               // X coordinate of the triangle second vertex
                                    const int   y2,               // Y coordinate of the triangle second vertex
                                    const int   x3,               // X coordinate of the triangle third vertex
                                    const int   y3,               // Y coordinate of the triangle third vertex
                                    const color clr,              // Color
                                    const uchar opacity=255,      // Opacity
                                    const uint  style=UINT_MAX)   // Line style is one of the ENUM_LINE_STYLE enumeration values or a custom value
                       { this.m_canvas.TriangleWu(x1,y1,x2,y2,x3,y3,::ColorToARGB(clr,opacity),style);                                 }

//--- Draw a circle using AntiAliasing algorithm
   void              DrawCircleAA(const int    x,                 // X coordinate of the circle center
                                  const int    y,                 // Y coordinate of the circle center
                                  const double r,                 // Circle radius
                                  const color  clr,               // Color
                                  const uchar opacity=255,        // Opacity
                                  const uint  style=UINT_MAX)     // Line style is one of the ENUM_LINE_STYLE enumeration values or a custom value
                       { this.m_canvas.CircleAA(x,y,r,::ColorToARGB(clr,opacity),style);                                               }

//--- Draw a circle using Wu algorithm
   void              DrawCircleWu(const int    x,                 // X coordinate of the circle center
                                  const int    y,                 // Y coordinate of the circle center
                                  const double r,                 // Circle radius
                                  const color  clr,               // Color
                                  const uchar opacity=255,        // Opacity
                                  const uint  style=UINT_MAX)     // Line style is one of the ENUM_LINE_STYLE enumeration values or a custom value
                       { this.m_canvas.CircleWu(x,y,r,::ColorToARGB(clr,opacity),style);                                               }

//--- Draw an ellipse by two points using AntiAliasing algorithm
   void              DrawEllipseAA(const double x1,               // X coordinate of the first point defining the ellipse
                                   const double y1,               // Y coordinate of the first point defining the ellipse
                                   const double x2,               // X coordinate of the second point defining the ellipse
                                   const double y2,               // Y coordinate of the second point defining the ellipse
                                   const color  clr,              // Color
                                   const uchar opacity=255,       // Opacity
                                   const uint  style=UINT_MAX)    // Line style is one of the ENUM_LINE_STYLE enumeration values or a custom value
                       { this.m_canvas.EllipseAA(x1,y1,x2,y2,::ColorToARGB(clr,opacity),style);                                        }

//--- Draw an ellipse by two points using Wu algorithm
   void              DrawEllipseWu(const int   x1,                // X coordinate of the first point defining the ellipse
                                   const int   y1,                // Y coordinate of the first point defining the ellipse
                                   const int   x2,                // X coordinate of the second point defining the ellipse
                                   const int   y2,                // Y coordinate of the second point defining the ellipse
                                   const color clr,               // Color
                                   const uchar opacity=255,       // Opacity
                                   const uint  style=UINT_MAX)    // Line style is one of the ENUM_LINE_STYLE enumeration values or a custom value
                       { this.m_canvas.EllipseWu(x1,y1,x2,y2,::ColorToARGB(clr,opacity),style);                                        }

//+------------------------------------------------------------------+
```

The logic of all added methods is completely transparent. All parameters passed to the methods are signed. The purpose of each method is stated in the comments. So, I believe all is clear here. In any case, you are welcome to use the comments section.

### Methods of working with a text

The CCanvas class remembers the settings of the last displayed text, including its font, color, transparency, etc. To find out the text size, we are able to use the [TextSize()](https://www.mql5.com/en/docs/standardlibrary/canvasgraphics/ccanvas/ccanvastextsize) method applying the current font settings to measure the width and height of the text-bounding rectangle. We may need this on several occasions, like when overwriting the previous text using the background color and writing it using new coordinates — text offset. In this case, we need not only text coordinates but also text anchor points (left-top, center-top, right-top etc.). We need to know exactly which anchor angle of the bounding rectangle is given to the text, otherwise the coordinates of the eraser rectangle will be set incorrectly. To achieve this, we need to add the class member variable storing the last specified anchor point.

Declare the variable in the private section of the class:

```
   long              m_long_prop[ORDER_PROP_INTEGER_TOTAL];    // Integer properties
   double            m_double_prop[ORDER_PROP_DOUBLE_TOTAL];   // Real properties
   string            m_string_prop[ORDER_PROP_STRING_TOTAL];   // String properties

   ENUM_TEXT_ANCHOR  m_text_anchor;                            // Current text alignment

//--- Return the index of the array the order (1) double and (2) string properties are located at
   int               IndexProp(ENUM_CANV_ELEMENT_PROP_DOUBLE property)  const { return(int)property-CANV_ELEMENT_PROP_INTEGER_TOTAL;                                 }
   int               IndexProp(ENUM_CANV_ELEMENT_PROP_STRING property)  const { return(int)property-CANV_ELEMENT_PROP_INTEGER_TOTAL-CANV_ELEMENT_PROP_DOUBLE_TOTAL;  }
```

At the very start of the parametric class constructor, initialize the values of the object typeand the text anchor point:

```
//+------------------------------------------------------------------+
//| Parametric constructor                                           |
//+------------------------------------------------------------------+
CGCnvElement::CGCnvElement(const ENUM_GRAPH_ELEMENT_TYPE element_type,
                           const int      element_id,
                           const int      element_num,
                           const long     chart_id,
                           const int      wnd_num,
                           const string   name,
                           const int      x,
                           const int      y,
                           const int      w,
                           const int      h,
                           const color    colour,
                           const uchar    opacity,
                           const bool     movable=true,
                           const bool     activity=true,
                           const bool     redraw=false)

  {
   this.m_name=this.m_name_prefix+name;
   this.m_chart_id=chart_id;
   this.m_subwindow=wnd_num;
   this.m_type=element_type;
   this.m_text_anchor=0;
   if(this.Create(chart_id,wnd_num,this.m_name,x,y,w,h,colour,opacity,redraw))
     {
```

In the **m\_type** parent class variable returned by its Type() virtual method, add the object type passed in the constructor parameters, while the **m\_text\_anchor** variable is initialized using the default value — the upper left corner of the bounding rectangle.

At the very end of the class body (namely, after the code block for working with primitives), add the code block for working with text:

```
//+------------------------------------------------------------------+
//| The methods of working with text                                 |
//+------------------------------------------------------------------+
//--- Return text the alignment type (anchor method)
   ENUM_TEXT_ANCHOR  TextAnchor(void)                       const { return this.m_text_anchor;                                         }

//--- Set the current font
   bool              SetFont(const string name,                   // Font name. For example, "Arial"
                             const int    size,                   // Font size
                             const uint   flags=0,                // Font creation flags
                             const uint   angle=0,                // Font slope angle in tenths of a degree
                             const bool   relative=true)          // Relative font size flag
                       { return this.m_canvas.FontSet(name,(relative ? size*-10 : size),flags,angle);                                  }

//--- Set a font name
   bool              SetFontName(const string name)               // Font name. For example, "Arial"
                       { return this.m_canvas.FontNameSet(name);                                                                       }

//--- Set a font size
   bool              SetFontSize(const int size,                  // Font size
                                 const bool relative=true)        // Relative font size flag
                       { return this.m_canvas.FontSizeSet(relative ? size*-10 : size);                                                 }

//--- Set font flags
//--- FONT_ITALIC - Italic, FONT_UNDERLINE - Underline, FONT_STRIKEOUT - Strikeout
   bool              SetFontFlags(const uint flags)               // Font creation flags
                       { return this.m_canvas.FontFlagsSet(flags);                                                                     }

//--- Set a font slope angle
   bool              SetFontAngle(const float angle)              // Font slope angle in tenths of a degree
                       { return this.m_canvas.FontAngleSet(uint(angle*10));                                                            }

//--- Set the font anchor angle (alignment type)
   void              SetTextAnchor(const uint flags=0)      { this.m_text_anchor=(ENUM_TEXT_ANCHOR)flags;                              }

//--- Gets the current font parameters and write them to variables
   void              GetFont(string &name,                        // The reference to the variable for returning a font name
                             int    &size,                        // Reference to the variable for returning a font size
                             uint   &flags,                       // Reference to the variable for returning font flags
                             uint   &angle)                       // Reference to the variable for returning a font slope angle
                       { this.m_canvas.FontGet(name,size,flags,angle);                                                                 }

//--- Return (1) the font name, (2) size, (3) flags and (4) slope angle
   string            FontName(void)                         const { return this.m_canvas.FontNameGet();                                }
   int               FontSize(void)                         const { return this.m_canvas.FontSizeGet();                                }
   int               FontSizeRelative(void)                 const { return(this.FontSize()<0 ? -this.FontSize()/10 : this.FontSize()); }
   uint              FontFlags(void)                        const { return this.m_canvas.FontFlagsGet();                               }
   uint              FontAngle(void)                        const { return this.m_canvas.FontAngleGet();                               }

//--- Return the text (1) width, (2) height and (3) all sizes (the current font is used to measure the text)
   int               TextWidth(const string text)                 { return this.m_canvas.TextWidth(text);                              }
   int               TextHeight(const string text)                { return this.m_canvas.TextHeight(text);                             }
   void              TextSize(const string text,                  // Text for measurement
                              int         &width,                 // Reference to the variable for returning a text width
                              int         &height)                // Reference to the variable for returning a text height
                       { this.m_canvas.TextSize(text,width,height);                                                                    }

//--- Display the text in the current font
   void              Text(int         x,                          // X coordinate of the text anchor point
                          int         y,                          // Y coordinate of the text anchor point
                          string      text,                       // Display text
                          const color clr,                        // Color
                          const uchar opacity=255,                // Opacity
                          uint        alignment=0)                // Text anchoring method
                       {
                        this.m_text_anchor=(ENUM_TEXT_ANCHOR)alignment;
                        this.m_canvas.TextOut(x,y,text,::ColorToARGB(clr,opacity),alignment);
                       }

  };
//+------------------------------------------------------------------+
```

Here all is similar to handling primitives. All methods feature comments disclosing their purpose, inputs and outputs.

I would like to comment on the method of setting the current font parameters:

```
//--- Set the current font
   bool              SetFont(const string name,                   // Font name. For example, "Arial"
                             const int    size,                   // Font size
                             const uint   flags=0,                // Font creation flags
                             const uint   angle=0,                // Font slope angle in tenths of a degree
                             const bool   relative=true)          // Relative font size flag
                       { return this.m_canvas.FontSet(name,(relative ? size*-10 : size),flags,angle);
```

Here **size** specifies the font size and is always set according to the font size we would set when displaying a text with the help of the ordinary [OBJ\_LABEL](https://www.mql5.com/en/docs/constants/objectconstants/enum_object/obj_label) text label object. The size is set as a positive integer value in it. Contrary to that, when drawing the text on the canvas, the font size is set the same way as in the [TextSetFont()](https://www.mql5.com/en/docs/objects/textsetfont) function:

The font size is set using positive or negative values. The sign defines whether the text size depends on the operating system settings (font scale).

- If the size is positive, it is converted into device physical units (pixels) when displaying the logical font as a physical one. The size corresponds to the height of symbol cells from available fonts. It is not recommended in cases of shared usage of texts displayed using the [TextOut()](https://www.mql5.com/en/docs/objects/textout) function and texts displayed using the [OBJ\_LABEL](https://www.mql5.com/en/docs/constants/objectconstants/enum_object) ("Text label") graphical object.
- If the size is negative, it is assumed to be set in tenths of a logical point (the value -350 is equal to 35 logical points) and is divided by 10. The resulting value is converted into physical units of the device (pixels) and corresponds to the absolute value of the character height from available fonts. To obtain a text of the [OBJ\_LABEL](https://www.mql5.com/en/docs/constants/objectconstants/enum_object) object size on the screen, multiply the font size specified in the object properties by -10.


The font relative size flag is checked in the method. If it is set (by default), the **size** value is multiplied by -10 so that the font is specified using a correct value for passing the CCanvas class to the [FontSet()](https://www.mql5.com/en/docs/standardlibrary/canvasgraphics/ccanvas/ccanvasfontset) method.

In the parametric class constructor, add font initialization (set its default name and size):

```
//+------------------------------------------------------------------+
//| Parametric constructor                                           |
//+------------------------------------------------------------------+
CGCnvElement::CGCnvElement(const ENUM_GRAPH_ELEMENT_TYPE element_type,
                           const int      element_id,
                           const int      element_num,
                           const long     chart_id,
                           const int      wnd_num,
                           const string   name,
                           const int      x,
                           const int      y,
                           const int      w,
                           const int      h,
                           const color    colour,
                           const uchar    opacity,
                           const bool     movable=true,
                           const bool     activity=true,
                           const bool     redraw=false)

  {
   this.m_name=this.m_name_prefix+name;
   this.m_chart_id=chart_id;
   this.m_subwindow=wnd_num;
   this.m_type=element_type;
   this.SetFont("Calibri",8);
   this.m_text_anchor=0;
   if(this.Create(chart_id,wnd_num,this.m_name,x,y,w,h,colour,opacity,redraw))
     {
```

These are all the improvements I have planned for the current article. Let's test the results.

### Test

We have the [EA from the previous article](https://www.mql5.com/en/articles/9493#node04) displaying two graphical element objects on the chart. Let's take the same EA, save it in the new folder \\MQL5\\Experts\\TestDoEasy\ **Part75\** as **TestDoEasyPart75.mq5** and do the following:

When clicking on the first (top) object, we will alternately draw a rectangle and a circle on it. With each new click on the object, the size of the rectangle decreases by 2 pixels on each side and the radius of the circle also decreases by 2 pixels. The rectangle is drawn in the usual way, while the circle is drawn using smoothing. With each new click, the object opacity is increased from 0 to 255 in a loop.

When clicking on the second (bottom) object, display a text on it alternately changing its anchor point. Write the anchor point name in the text itself. The object transparency remains unchanged.

Specify the number of created elements:

```
//+------------------------------------------------------------------+
//|                                             TestDoEasyPart75.mq5 |
//|                                  Copyright 2021, MetaQuotes Ltd. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2021, MetaQuotes Ltd."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//--- includes
#include <Arrays\ArrayObj.mqh>
#include <DoEasy\Services\Select.mqh>
#include <DoEasy\Objects\Graph\GCnvElement.mqh>
//--- defines
#define        ELEMENTS_TOTAL (2)   // Number of created graphical elements
//--- input parameters
sinput   bool  InpMovable  = true;  // Movable flag
//--- global variables
CArrayObj      list_elements;
//+------------------------------------------------------------------+
```

To avoid creating identical objects each time the timeframe changes, clear the list of the already created objects in the OnInit() handler before creating new ones:

```
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//--- Set the permissions to send cursor movement and mouse scroll events
   ChartSetInteger(ChartID(),CHART_EVENT_MOUSE_MOVE,true);
   ChartSetInteger(ChartID(),CHART_EVENT_MOUSE_WHEEL,true);
//--- Set EA global variables

//--- Create the specified number of graphical elements on the canvas
   list_elements.Clear();
   int total=ELEMENTS_TOTAL;
   for(int i=0;i<total;i++)
     {
      //--- When creating an object, pass all the required parameters to it
      CGCnvElement *element=new CGCnvElement(GRAPH_ELEMENT_TYPE_ELEMENT,i,0,ChartID(),0,"Element_0"+(string)(i+1),300,40+(i*80),100,70,clrSilver,200,InpMovable,true,true);
      if(element==NULL)
         continue;
      //--- Add objects to the list
      if(!list_elements.Add(element))
        {
         delete element;
         continue;
        }
     }
//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
```

Since I do not have the graphical object collection class yet to check the need for creating a new object with a specified name, here I will simply recreate these objects anew clearing the list of the previously created objects beforehand.

The **OnChartEvent()** handler receives handling mouse clicks on two created objects:

```
//+------------------------------------------------------------------+
//| ChartEvent function                                              |
//+------------------------------------------------------------------+
void OnChartEvent(const int id,
                  const long &lparam,
                  const double &dparam,
                  const string &sparam)
  {
//--- If clicking on an object
   if(id==CHARTEVENT_OBJECT_CLICK)
     {
      //--- In the new list, get the element object with the name corresponding to the sparam string parameter value of the OnChartEvent() handler
      CArrayObj *obj_list=CSelect::ByGraphCanvElementProperty(GetPointer(list_elements),CANV_ELEMENT_PROP_NAME_OBJ,sparam,EQUAL);
      //--- If the object is received from the list
      if(obj_list!=NULL && obj_list.Total()>0)
        {
         static uchar try0=0, try1=0;
         //--- Get the pointer to the object in the list
         CGCnvElement *obj=obj_list.At(0);
         //--- If this is the first graphical element
         if(obj.ID()==0)
           {
            //--- Set a new opacity level for the object
            uchar opasity=obj.Opacity();
            if((opasity+5)>255)
               opasity=0;
            else
               opasity+=5;
            //--- Set a new opacity to the object
            obj.SetOpacity(opasity);
            //--- Set rectangle and circle coordinates
            int x1=2,x2=obj.Width()-3;
            int y1=2,y2=obj.Height()-3;
            int xC=(x1+x2)/2;
            int yC=(y1+y2)/2;
            int R=yC-y1;
            //--- Draw a rectangle at each first click
            if(try0%2==0)
               obj.DrawRectangle(x1+try0,y1+try0,x2-try0,y2-try0,clrDodgerBlue,obj.Opacity());
            //--- Display the circle smoothed using AntiAliasing at each second click
            else
               obj.DrawCircleAA(xC,yC,R-try0,clrGreen,obj.Opacity());
            //--- If the number of clicks on the object exceeds 30
            if(try0>30)
              {
               //--- Clear the object setting its current color and transparency
               obj.Erase(obj.ColorBG(),obj.Opacity());
               //--- Re-start the click number countdown
               try0=0;
              }
            //--- Update the chart and the object, and display the comment featuring color values and object opacity
            obj.Update(true); // 'true' is not needed here since the next Comment command redraws the chart anyway
            Comment("Object name: ",obj.NameObj(),", opasity=",obj.Opacity(),", Color BG: ",(string)obj.ColorBG());
            //--- Increase the counter of mouse clicks by object
            try0++;
           }
         //--- If this is the second object
         else if(obj.ID()==1)
           {
            //--- Set the font parameters for it ("Calibri" size 8)
            obj.SetFont("Calibri",8);
            //--- Set the text anchor angle corresponding to the click counter by object
            obj.SetTextAnchor((ENUM_TEXT_ANCHOR)try1);
            //--- Create the text out of the anchor angle name
            string text=StringSubstr(EnumToString(obj.TextAnchor()),12);
            //--- Set the text coordinates relative to the upper left corner of the graphical element
            int xT=2,yT=2;
            //--- Depending on the anchor angle, set the new coordinates of the displayed text
            //--- LEFT_TOP
            if(try1==0)       { xT=2; yT=2;                                   }
            //--- CENTER_TOP
            else if(try1==1)  { xT=obj.Width()/2; yT=2;                       }
            //--- RIGHT_TOP
            //--- since the ENUM_TEXT_ANCHOR enumeration features no 3, increase the counter of object clicks by 1
            else if(try1==2)  { xT=obj.Width()-2; yT=2; try1++;               }
            //--- LEFT_CENTER
            else if(try1==4)  { xT=2; yT=obj.Height()/2;                      }
            //--- CENTER
            else if(try1==5)  { xT=obj.Width()/2; yT=obj.Height()/2;          }
            //--- RIGHT_CENTER
            //--- since the ENUM_TEXT_ANCHOR enumeration features no 7, increase the counter of object clicks by 1
            else if(try1==6)  { xT=obj.Width()-2; yT=obj.Height()/2; try1++;  }
            //--- LEFT_BOTTOM
            else if(try1==8)  { xT=2; yT=obj.Height()-2;                      }
            //--- CENTER_BOTTOM
            else if(try1==9)  { xT=obj.Width()/2; yT=obj.Height()-2;          }
            //--- RIGHT_BOTTOM
            else if(try1==10)    { xT=obj.Width()-2; yT=obj.Height()-2;       }
            //--- Clear the graphical element filling it with the current color and transparency
            obj.Erase(obj.ColorBG(),obj.Opacity());
            //--- Display the text with the calculated coordinates in the cleared element
            obj.Text(xT,yT,text,clrDodgerBlue,255,obj.TextAnchor());
            //--- Update the object and chart
            obj.Update(true); // 'true' is not needed here since the next Comment command redraws the chart anyway
            Comment("Object name: ",obj.NameObj(),", opasity=",obj.Opacity(),", Color BG: ",(string)obj.ColorBG());
            //--- Increase the counter of object clicks
            try1++;
            if(try1>10)
               try1=0;
           }
        }
     }
  }
//+------------------------------------------------------------------+
```

The handler code features detailed comments. I believe, its logic is clear. If you have any questions related to the current article, feel free to ask them in the comments below.

Compile the EA and launch it on the chart. Click on the objects:

![](https://c.mql5.com/2/42/LPsL3d1mGv.gif)

As a result, I accidentally obtained a funny image on the upper object resembling a CD :)

### What's next?

In the next article, I will start the development of the objects that are descendants of the graphical element object created here.

All files of the current version of the library are attached below together with the test EA file for MQL5 for you to test and download.

Leave your questions and suggestions in the comments.

[Back to contents](https://www.mql5.com/en/articles/9515#node00)

**\*Previous articles within the series:**

[Graphics in DoEasy library (Part 73): Form object of a graphical element](https://www.mql5.com/en/articles/9442)

[Graphics in DoEasy library (Part 74): Basic graphical element powered by the CCanvas class](https://www.mql5.com/en/articles/9493)

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/9515](https://www.mql5.com/ru/articles/9515)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/9515.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/9515/mql5.zip "Download MQL5.zip")(3972.88 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/372825)**
(1)


![Keith Watford](https://c.mql5.com/avatar/avatar_na2.png)

**[Keith Watford](https://www.mql5.com/en/users/forexample)**
\|
6 Jul 2021 at 12:42

Comments that do not relate to this topic, have been moved to " [Off Topic Posts](https://www.mql5.com/en/forum/339471)".

![Graphics in DoEasy library (Part 76): Form object and predefined color themes](https://c.mql5.com/2/42/MQL5-avatar-doeasy-library3-2__3.png)[Graphics in DoEasy library (Part 76): Form object and predefined color themes](https://www.mql5.com/en/articles/9553)

In this article, I will describe the concept of building various library GUI design themes, create the Form object, which is a descendant of the graphical element class object, and prepare data for creating shadows of the library graphical objects, as well as for further development of the functionality.

![Graphics in DoEasy library (Part 74): Basic graphical element powered by the CCanvas class](https://c.mql5.com/2/42/MQL5-avatar-doeasy-library3-2__1.png)[Graphics in DoEasy library (Part 74): Basic graphical element powered by the CCanvas class](https://www.mql5.com/en/articles/9493)

In this article, I will rework the concept of building graphical objects from the previous article and prepare the base class of all graphical objects of the library powered by the Standard Library CCanvas class.

![Graphics in DoEasy library (Part 77): Shadow object class](https://c.mql5.com/2/42/MQL5-avatar-doeasy-library3-2__4.png)[Graphics in DoEasy library (Part 77): Shadow object class](https://www.mql5.com/en/articles/9575)

In this article, I will create a separate class for the shadow object, which is a descendant of the graphical element object, as well as add the ability to fill the object background with a gradient fill.

![Graphics in DoEasy library (Part 73): Form object of a graphical element](https://c.mql5.com/2/42/MQL5-avatar-doeasy-library3-2.png)[Graphics in DoEasy library (Part 73): Form object of a graphical element](https://www.mql5.com/en/articles/9442)

The article opens up a new large section of the library for working with graphics. In the current article, I will create the mouse status object, the base object of all graphical elements and the class of the form object of the library graphical elements.

[![](https://www.mql5.com/ff/si/m0dtjf9x3brdz07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fmarket%2Fmt5%2Fexpert%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dtop.experts%26utm_content%3Dbuy.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=widauvjabtsckwovwaperzkotrcrttvb&s=25ef75d39331f608a319410bf27ff02c1bd7986622ecc1eec8968a650f044731&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=zvskwfxamdqwhxxoakjtdwdvvxftjylt&ssn=1769253098861158876&ssn_dr=0&ssn_sr=0&fv_date=1769253098&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F9515&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Graphics%20in%20DoEasy%20library%20(Part%2075)%3A%20Methods%20of%20handling%20primitives%20and%20text%20in%20the%20basic%20graphical%20element%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176925309861387546&fz_uniq=5083394773588384490&sv=2552)

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