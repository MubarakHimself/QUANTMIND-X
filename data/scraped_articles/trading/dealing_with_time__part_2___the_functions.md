---
title: Dealing with Time (Part 2): The Functions
url: https://www.mql5.com/en/articles/9929
categories: Trading, Trading Systems
relevance_score: 0
scraped_at: 2026-01-24T13:32:32.591261
---

[![](https://www.mql5.com/ff/si/fx5m8s6u6uxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=ysducdhemkrdsdtzzbfkclolrllnhezk&s=33f180a31db6c3b846d77732b0bc78169421a47b8cf9f076ca717f4e4846d1c7&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=kkzbzrcxqckfhhofhzlichwnakkcfwxu&ssn=1769250751888605403&ssn_dr=0&ssn_sr=0&fv_date=1769250751&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F9929&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Dealing%20with%20Time%20(Part%202)%3A%20The%20Functions%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176925075126376605&fz_uniq=5082938368888672703&sv=2552)

MetaTrader 5 / Examples


### The Global Variables

Instead of asking the support of your broker, from whom you will probably receive an insufficient answer (who would be willing to explain a missing hour), we simply look ourselves how they time their prices in the weeks of the time changes — but not cumbersome by hand, we let a program do it  — why do we have a PC after all.

Before the functions in the include file _DealingWithTime.mqh_(and after the macro substitution) the required variables are declared as global variables:

```
//--- global variables for time switches
int      DST_USD=0,                             // act time shift USA
         DST_EUR=0,                             // act time shift EU
         DST_AUD=0,                             // act time shift Australia
         DST_RUS=0;                             // D'2014.10.26 02:00', -10800,
```

These variables DST\_USD, DST\_EUR,.. will have the actual time shift of the USA, the EU,.. They will be updated and set by our functions. In the winter time which is the normal time they are zero: the time is not shifted at that period.

After that we have the variables with the next time the time changeover will take place. They are mainly needed to know when a new calculation is required in order to save the CPU resources:

```
datetime nxtSwitch_USD,                         // date of next switch
         nxtSwitch_EUR,                         // date of next switch
         nxtSwitch_AUD,                         // date of next switch
         nxtSwitch_RUB = D'2014.10.26 02:00';   // Russia s different :(
```

We will consider the Russian situation later in this article.

This structure and its global variable is the heart of all. :)

```
struct _OffsetBroker
  {
   int   USwinEUwin,                            // US=Winter & EU=Winter
         USsumEUsum,                            // US=Summer & EU=Summer
         USsumEUwin,                            // US=Summer & EU=Winter
         actOffset,                             // actual time offset of the broker
         secFxWiWi,                             // duration of FX in sec
         secFxSuSu,                             // duration of FX in sec
         secFxSuWi,                             // duration of FX in sec
         actSecFX;                              // actual duration of FX in sec
   bool  set;                                   // are all set?
  };
_OffsetBroker OffsetBroker;
```

We will assign the broker offsets for the three relevant periods and duration of the forex market is open in these periods, for both the actual value and for an easy check set if the values have been assigned. The global variable is named _OffsetBroker_, we will meet it several times.

### The Central Function to Determine the Time Shifts of the Broker

By the call of this function:

```
setBokerOffset();
```

an EA, indicator or script can determine by itself when the broker handles time changes and how. It is placed at the beginning of code in the attached script after Start() and the function determines the relevant values of the broker for the relevant periods (summer time, winter time, and the intermediate time), which can then be used to determine all other required times via GMT. Like everything else, it is located in the included include file DealingWithTime.mqh which also includes the parts from the first article. After the variable declaration and the initializations and the zeroing of the relevant global variables:

```
   datetime dateEUR,dateUSD,dateAUD,dateNxt,                   // switch date for EU, AU, & US
            arrTme[];                                          // array to copy time
   int b;
   OffsetBroker.USsumEUwin =
      OffsetBroker.USsumEUsum =
         OffsetBroker.USwinEUwin = INT_MIN;
   nxtSwitch_USD = nxtSwitch_EUR = nxtSwitch_AUD = 0;          // reset variables
```

we find the changeover weekends:

```
//--- AU, EU & US switches to winter time in 2020
   if(IS_DEBUG_MODE)
      Print("\n2nd half-year 2020 for ",AccountInfoString(ACCOUNT_COMPANY), "DebugMode: ",IS_DEBUG_MODE);

   nextDST("EUR", D'2020.06.21 14:00');                        // EUR: get DST and set next change
   b = CopyTime("EURUSD",PERIOD_H1,nxtSwitch_EUR,1,arrTme);    // get time last 1h bar before switch in EU
   dateEUR = arrTme[0];                                        // last hour on Friday before the weekend

   nextDST("USD", D'2020.06.21 14:00');                        // USA: get DST and set next change
   b = CopyTime("EURUSD",PERIOD_H1,nxtSwitch_USD,1,arrTme);    // get time last 1h bar before switch in USA
   dateUSD = arrTme[0];                                        // last hour on Friday before the weekend

   nextDST("AUD", D'2020.06.21 14:00');                        // AUD: get DST and set next change
   b = CopyTime("EURUSD",PERIOD_H1,nxtSwitch_AUD,1,arrTme);    // get time last 1h bar before switch in AU
   dateAUD = arrTme[0];                                        // last hour on Friday before the weekend

   dateNxt = fmax(nxtSwitch_EUR,nxtSwitch_USD)+WeekInSec;      // get the next weekend
   b = CopyTime("EURUSD",PERIOD_H1,dateNxt,1,arrTme);          // get time last 1h bar before the weekend
   dateNxt = arrTme[0];                                        // last hour on Friday before the weekend
```

To make it easy most of the printouts are executed automatically in the debug mode: _if(IS\_DEBUG\_MODE)_. So when you start e.g. the attached script in the debugger (F5) you'll see all the details but if you start the same script right on a chart you'll get only the important things.

For all three time zones, the function call e.g. _nextDST("EUR", D'2020.06.21 14:00')_ is firstly used to calculate the applicable time differences for the EU and secondly the next switch. In June it is daylight saving time, and the next changeover weekend will be to winter time. In the line immediately after that, we get the opening time of the last h1 bar on the Friday before that weekend as this will be the anchor for our calculation. See point 4 of the assumptions from the end of the first article:

          4\. If there are missing hours between Fri. 17:00 and Sun. 17:00, then there will be missing quotes on Sunday until the first quote and not on Friday after the last quote received.

I decided to use the times of h1 and "EURUSD". This symbol is the one with the longest history probably not only at MQ. But this also means that if the forex market closes at 17:00 New York time, the last hour or the last 1h bar there starts at **16:00**and this is the hour we are particularly interested in further on. The first hour after a weekend is then at 17:00 on Sunday in New York. The changeover of Australia is also determined for the sake of completeness, but is not used in the further (see below). After that, the first weekend after both time zones have changed over is still determined in order to calculate the broker's time shift of the next time shift period.

Then for the three Friday hours the respective period-related time shifts of the broker are computed with the function _chckFriday(...)_ on basis of the time 16:00 in New York. This function is part of include file and we will discuss it below.

```
   chckFriday(dateEUR,"EUR");                                  // function to determine broker offset for the Friday given
   chckFriday(dateUSD,"USD");                                  // function to determine broker offset for the Friday given
   chckFriday(dateNxt,"NXT");                                  // function to determine broker offset for the Friday given
```

After that, the same principle is used to calculate the other time changes in the other half of the year - which (see above) we actually do not need and which could be commented out:

```
   if(IS_DEBUG_MODE)
      Print("\n1st half-year 2021 for ",AccountInfoString(ACCOUNT_COMPANY), "DebugMode: ",IS_DEBUG_MODE);
   nxtSwitch_USD = nxtSwitch_EUR = nxtSwitch_AUD = 0;

   nextDST("AUD", D'2021.01.21 14:00');                        // AUD: get DST and set next change
   b = CopyTime("EURUSD",PERIOD_H1,nxtSwitch_AUD,1,arrTme);    // get time last 1h bar before switch in EU
   dateAUD = arrTme[0];                                        // last hour on Friday before the weekend

...
   chckFriday(dateUSD,"USD");                                  // function to determine broker offset for the Friday given
   chckFriday(dateEUR,"EUR");                                  // function to determine broker offset for the Friday given
   chckFriday(dateNxt,"NXT");                                  // function to determine broker offset for the Friday given
```

Finally, after the the broker offsets has been detected and assigned to the corresponding fields the decisive changeover times ( _nxtSwitch\_USD = nxtSwitch\_EUR = nxtSwitch\_AUD = 0_) are reset to zero for a subsequent use. Since the recalculation is to take place only if in the course of the historical course a weekend with a time changeover was 'passed', a later time can prevent the correct calculations, therefore rather once more reset than once too little. Then it is checked whether all values were assigned and the result is printed into the Expert journal and the check is returned:

```
   nxtSwitch_USD = nxtSwitch_EUR = nxtSwitch_AUD = 0;          // reset variables for use by a user
   if(OffsetBroker.USsumEUwin != INT_MIN
      && OffsetBroker.USsumEUsum != INT_MIN
      && OffsetBroker.USwinEUwin != INT_MIN
     )
      OffsetBroker.set = true;
   else
      OffsetBroker.set = false;
   if(OffsetBroker.set)
      Print("\nTime Offset of ",AccountInfoString(ACCOUNT_COMPANY),": ",
            "\nUS=Winter & EU=Winter (USwinEUwin) = ",OffsetBroker.USwinEUwin,
            "\nUS=Summer & EU=Summer (USsumEUsum) = ",OffsetBroker.USsumEUsum,
            "\nUS=Summer & EU=Winter (USsumEUwin) = ",OffsetBroker.USsumEUwin,
            "\n");
   else
      Print(__FILE__,"[",__LINE__,"] Assigning the broker offset went wrong - somehow.");
   return(OffsetBroker.set);
```

If everything is ok one will see e.g these lines:

Time Offset of MetaQuotes Software Corp.:

US=Winter & EU=Winter (USwinEUwin)       =   -7200

US=Summer & EU=Summer (USsumEUsum) = -10800

US=Summer & EU=Winter (USsumEUwin)    =   -7200

This enables any user to use these values for input variables instead of the function to be executed before an EA starts working either in the strategy tester or live on a chart. An example is presented at the end of this article.

### Determine and Set the Time Shifts of the Broker

Let us now consider the function _chckFriday(...)_. It determines the time offset of the respective broker for the different periods and assigns it to the respective field of the global variable _OffsetBroker_ from the type of the structure _\_OffsetBroker_. The structure knows these three fields:

```
   int   USwinEUwin,                            // US=Winter & EU=Winter
         USsumEUsum,                            // US=Summer & EU=Summer
         USsumEUwin,                            // US=Summer & EU=Winter
```

They are assigned the respective time shift of the broker of the corresponding period. The periods are:

- both regions the same, either winter (or standard time) or summer, or
- USA, already(still) in summer and the EU still(already) in winter.


The reverse situation, that the USA is already(still) in winter, but the EU is still(already) in summer, does not exist. Here the question arises not only why the fourth category does not exist and what about Australia or AUD  — is it missing here?

Here are the rules for switching:

- EU: last Sunday in October and last Sunday in March
- US: 1st Sunday in November and 2nd Sunday in March and
- AU: 1st Sunday in November and last Sunday in March


To the first question. The EU switches to winter time one or two weeks before the US: there it is still summer in the US while it is already winter in the EU, so the value is assigned to the _USsumEUwin_ field of the _OffsetBroker_ variable. Then in spring the USA switches to summer time before the EU, then again it is summer in the USA for a week or two while it is still winter in the EU. Again, the value is assigned to the _USsumEUwin_ field of the _OffsetBroker_ variable. This makes it clear that the reverse case EU already(still) summer, but the USA still(already) winter does not occur at all. This actually eliminates the need to calculate the broker time offsets for both changeover periods in autumn and spring. It is nevertheless executed  — simply because of the completeness and the control.

On the second question. Australia switches over in November like the U.S. and in the spring like the EU. So there are no additional different weekend for the changeover. However, the clock in Australia is advanced by 1 hour when it is winter in the EU and the USA, because Christmas and New Year fall in summer there.

Now, if we already calculate the time shift for a special period, then we can also calculate the current duration that FX market is open in these weeks. These values are stored in the fields _secFxWiWi_, _secFxSuSu_, _secFxSuWi_ and the current valid value in _actSecFX_. At the end of the article, in the chapter Application, it is shown how to handle this.

But before the values can be assigned, they have to be determined. After the variable declaration and the resetting of the global variables, the time shifts for EU and USA are calculated for the given time _tB_ (time Broker):

```
//+------------------------------------------------------------------+
//| auxiliary function to determine time offset of the broker        |
//+------------------------------------------------------------------+
int chckFriday(
   datetime tB,                                                // time Broker: the last hour on Friday
   string cmt=""                                               // text to start the line
)
  {

   int hNY, hGMT, hTC, hDiff;
   nxtSwitch_AUD = nxtSwitch_USD = nxtSwitch_EUR = 0;          // reset to be save
   nextDST("EUR",tB);                                          // get the offset for EUR of the time tB given
   nextDST("USD",tB);                                          // get the offset for USA of the time tB given
```

Here, _tB_ is the beginning of the last hour of Friday, i.e. when it is 16:00 in New York. This assumption is the basis of the further calculation, because we can calculate GMT for this time:

tGMT = tNY + ( _NYShift_ \+ _DST\_USD_)

and therefore the offset of the broker to GMT. We determine this offset this way: From the last Friday hour of the broker _tB_ we subtract the past seconds of this day, _SoB(tB)_. We get the time 00:00 for the day and then add the seconds until 16:00 (16\*3600). Now we know the New York time, from which we get GMT by adding _NYShift + DST\_USD_. Now we can easily determine the broker's time offset from GMT and then assign it to the appropriate field of _OffsetBroker_.

In the function, this is all done in hours (instead of seconds) with the macro substitution _HoD()_ \- Hour of Day, for the sake of documentation and easier verifiability in the printout:

```
   hNY   = HoD(tB - SoD(tB) + 16*3600);                        // get the hour of New York time
   hGMT  = HoD(tB - SoD(tB) + 16*3600 + NYShift + DST_USD);    // get the hour of GMT
   hTC   = HoD(tB);                                            // get the hour of the time given
   hDiff = hGMT - HoD(tB);                                     // get the difference between GMT and the broker
```

It's not so hard after all. ;)

For the sake of security, the following is inserted. It checks whether the situation not to be expected, USA in summer and EU in winter, does not occur:

```
   if(DST_USD==0 && DST_EUR!=0)                                // this should not occur
      Alert(__LINE__," ",TOSTR(DST_USD),TOSTR(DST_EUR),"  USwin && EUsum");
```

Now we can assign the found difference and the duration of opening of the FX market:

```
//--- set the broker offset for the various time situations:
   if(DST_USD+DST_EUR==0)                                      // both in winter (normal) time
     {
      OffsetBroker.actOffset = OffsetBroker.USwinEUwin = hDiff*3600;
      OffsetBroker.actSecFX  = OffsetBroker.secFxWiWi = SoW(tB);
     }
   else
      if(DST_USD == DST_EUR)                                   // else both in summer time
        {
         OffsetBroker.actOffset = OffsetBroker.USsumEUsum = hDiff*3600;
         OffsetBroker.actSecFX  = OffsetBroker.secFxSuSu = SoW(tB);
        }
      else
         if(DST_USD!=0 && DST_EUR==0)                          // US:summer EU:winter
           {
            OffsetBroker.actOffset = OffsetBroker.USsumEUwin = hDiff*3600;
            OffsetBroker.actSecFX  = OffsetBroker.secFxSuWi = SoW(tB);
           }
```

Finally, we print out all the values found and return the last actual offset:

```
//--- calc the ring of times NY->GMT->Broker->GMT->NY <= the last NY must always be 16!!
   Print(cmt,": ",DoWs(tB),TimeToString(tB),": ",TOSTR(hNY),TOSTR(hGMT),TOSTR(hTC),TOSTR(hDiff),
         " BrokerTime => GMT: ",TimeToString(tB+OffsetBroker.actOffset),
         " => tNY: ",TimeToString((tB + OffsetBroker.actOffset)-(NYShift + DST_USD)),
         "  End-FX after: ",OffsetBroker.actSecFX/3600,"h"
        );
   return(OffsetBroker.actOffset);
```

This looks as follows:

EUR: Fr.2020.10.23 23:00: hNY:16  hGMT:20  hTC:23  hDiff:-3   BrokerTime => GMT: 2020.10.23 20:00 => tNY: 2020.10.23 16:00  End FX in: 143h

USD: Fr.2020.10.30 22:00: hNY:16  hGMT:20  hTC:22  hDiff:-2   BrokerTime => GMT: 2020.10.30 20:00 => tNY: 2020.10.30 16:00  End FX in: 142h

NXT: Fr.2020.11.06 23:00: hNY:16  hGMT:21  hTC:23  hDiff:-2   BrokerTime => GMT: 2020.11.06 21:00 => tNY: 2020.11.06 16:00  End FX in: 143h

There are some interesting things displayed here that are worth discussing. The EU switches first on 10/25. A week later, the US on 11/1. In this interim period, the last hour on Friday at MQ starts at 22:00instead of 23:00 and the week ends with this bar, after 142hours instead of normally after 143 hours. 143 or 142 hours? The FX week has only 120 hours: 5\*24=120? The seconds of the week ( _SoW())_ and the other comparable functions refer to the calendric week, which begins on Sunday at 00:00. But from Su. 00:00 to Fr. 23:00 there are now 6\* 24-1 = 143. This value is used in the following to calculate for any given moment in a week the remaining time the FX market stays open.

These three lines are also used to check the logic and calculation and as an example of how a user can determine a required local time from GMT. Starting from the left, the broker's time stamp is followed by the assumed time in New York, _hNY: 16_, then the hour of GMT based on New York time, the broker's hour and its time offset: -2 or -3. In the second part further to the right, GMT is calculated from the broker's time _(tB+OffsetBroker.actOffset)_ and then from GMT again the time in New York _((tB + OffsetBroker.actOffset)-(NYShift + DST\_USD))._ Here must always be 16:00 for tNY and it does. A second check for arbitrary times in the history and for other brokers is done below.

### Calculating the Weekends of the Time Changeover

But before we come to the check, where we go through the time history for "EURUSD", we have to discuss the heart of the calculation, the function _nextDST(...)_.

The function is called with the parameter _zone_ for the timezone "USD", "EUR" or "AUD" (where "AUD" is not really needed) and _t_, the parameter for a current time, usually the current broker time _TimeCurrent()._ First it is checked if a recalculation is necessary at all (here for "EUR"):

```
void nextDST(string zone, datetime t)
  {
   if((zone == "EUR") && t < nxtSwitch_EUR)
     {
      if(IS_DEBUG_MODE)
         Print("no change as time < nxtSwitch_EUR");
      return;
     }
...
```

This also shows why it is important to reset the values for nxtSwitch\_EUR to zero at the beginning of the test, because otherwise a recalculation might not be done for the whole test duration.

Then, after the variable declaration and initialization, we come to the heart of the function, which is not from me. A quite some time ago I found somewhere in the net an algorithm, which determines a certain day in the month. It is used to determine the summer or winter time for a given point in time. The algorithm is not that complicated:

1. Determine the day of the month, a Sunday, on which changeover takes place.
2. Create a date from it.
3. Find the nearest future changeover Sunday.
4. Set the time shift, either 0h or -1h, and the next changeover Sunday.


The magic of this algorithm is in the code line that determines the day in the month of the time change. For the EU it is the last Sunday in March and it is calculated like this (as I said, the idea of the formula is not mine):

```
d = (int)(31 - MathMod((4 + MathFloor(5*y/4)), 7));         // determine the last Sunday in March for the EU switch
```

For the year 2021 results for the last Sunday in March, here as EXCEL formula, as d=25:

31 - MOD(ROUNDDOWN(5\*2021/4);7) = 25.

From this then the timestamp is formed, at which the EU switches to summer time: March 25, 2021, the last Sunday in March:

```
spr = StringToTime(""+(string)y+".03."+(string)d+" 03:00"); // convert to datetime format
```

The procedure for all other dates is similar and needs no separate explanation.

Here now a larger section of the code, which determines the current summer or winter time shift and the next changeover date in the EU for a given date. We need three sections in one year: before the first, between the first and the second and after the second time changeover. The following time changeover is then already in the next year and this must be taken into account:

```
   if(zone == "EUR")
     {
      d = (int)(31 - MathMod((4 + MathFloor(5*y/4)), 7));         // determine the last Sunday in March for the EU switch
      spr = StringToTime(""+(string)y+".03."+(string)d+" 03:00"); // convert to datetime format
      if(t < spr)
        {
         DST_EUR = 0;                                             // no time offset
         nxtSwitch_EUR = spr;                                     // set the next time switch
         if(IS_DEBUG_MODE)
            Print(zone,"-DST for ",TimeToString(t)," DST: ",StringFormat("% 5i",DST_EUR),"  nxtSwitch: ",DoWs(nxtSwitch_EUR)," ",TimeToString(nxtSwitch_EUR));
         return;
        }
      d = (int)(31 - MathMod((1 + MathFloor(5*y/4)), 7));         // determine the last Sunday in October for the EU switch
      aut = StringToTime(""+(string)y+".10."+(string)d+" 03:00"); // convert to datetime format
      if(t < aut)
        {
         DST_EUR =-3600;                           // = +1h => 09:00 London time = GMT+05h+DST_EU = GMT+0+1 = GMT+1;
         nxtSwitch_EUR = aut;                                     // set the next time switch
         if(IS_DEBUG_MODE)
            Print(zone,"-DST for ",TimeToString(t)," DST: ",StringFormat("% 5i",DST_EUR),"  nxtSwitch: ",DoWs(nxtSwitch_EUR)," ",TimeToString(nxtSwitch_EUR));
         return;
        }
      y++;                                                        // re-calc the spring switch for the next year
      d = (int)(31 - MathMod((4 + MathFloor(5*y/4)), 7));         // determine the last Sunday in March for the EU switch
      spr = StringToTime(""+(string)y+".03."+(string)d+" 03:00"); // convert to datetime format
      if(t < spr)
        {
         DST_EUR = 0;                                             // no time offset
         nxtSwitch_EUR = spr;                                     // set the next time switch
         if(IS_DEBUG_MODE)
            Print(zone,"-DST for ",TimeToString(t)," DST: ",StringFormat("% 5i",DST_EUR),"  nxtSwitch: ",DoWs(nxtSwitch_EUR)," ",TimeToString(nxtSwitch_EUR));
         return;
        }
      Print("ERROR for ",zone," @ ",TimeToString(t)," DST: ",StringFormat("% 5i",DST_EUR),"  nxtSwitch: ",DoWs(nxtSwitch_EUR)," ",TimeToString(nxtSwitch_EUR),"  winter: ",TimeToString(aut),"  spring: ",TimeToString(spr));
      return;
     }
```

One can see three section within one year:

1. Before the switch to summer time in March.
2. Before the switch to winter time in October/November.
3. In the winter time the switch to the summer time in the next year.

This is repeated for EUR, USD and AUD.

Single calls of the function _nextDST(..)_ like

```
nextDST("EUR", D'2019.02.05 20:00');
nextDST("EUR", D'2019.06.05 20:00');
nextDST("EUR", D'2019.11.20 20:00');

nextDST("USD", D'2019.02.05 20:00');
nextDST("USD", D'2019.06.05 20:00');
nextDST("USD", D'2019.11.20 20:00');

nextDST("AUD", D'2019.02.05 20:00');
nextDST("AUD", D'2019.06.05 20:00');
nextDST("AUD", D'2019.11.20 20:00');
```

will test all the three relevant time points in the year for the three regions. This is the result:

EU: last Sunday in March and last Sunday in October:

    EUR-DST for 2019.02.05 20:00 DST:        0   nxtSwitch: Su. 2019.03.31 03:00

    EUR-DST for 2019.06.05 20:00 DST: -3600   nxtSwitch: Su. 2019.10.27 03:00

    EUR-DST for 2019.11.20 20:00 DST:        0   nxtSwitch: Su. 2020.03.29 03:00

US: 2nd Sunday in March and 1st Sunday in November:

    USD-DST for 2019.02.05 20:00 DST:        0   nxtSwitch: Su. 2019.03.10 03:00

    USD-DST for 2019.06.05 20:00 DST: -3600   nxtSwitch: Su. 2019.11.03 03:00

    USD-DST for 2019.11.20 20:00 DST:        0   nxtSwitch: Su. 2020.03.08 03:00

AU: 1st Sunday in November and last Sunday in March:

    AUD-DST for 2019.02.05 20:00 DST: -3600   nxtSwitch: Su. 2019.03.31 03:00

    AUD-DST for 2019.06.05 20:00 DST:        0   nxtSwitch: Su. 2019.11.03 03:00

    AUD-DST for 2019.11.20 20:00 DST: -3600   nxtSwitch: Su. 2020.03.29 03:00

Perhaps the time shift in Australia is confusing, but Australia, unlike the USA and Europe, is located in the southern hemisphere, where the turn of the year is in the middle of summer and therefore it is to be expected that their daylight saving time is in the European winter.

### The Changeover in Russia

One small note, since MQ has Russian roots and there are many Russian users, the Russian time changes are also included. However, because of the large number of changes, when and how the clocks were turned in Russia, I decided to use a two-dimensional array, in which the times and the respective shifts were entered and which can be queried with this function:

```
long RussiaTimeSwitch[][2] =
  {
   D'1970.01.00 00:00', -10800,
   D'1980.01.00 00:00', -10800,
   D'1981.04.01 00:00', -14400,
...
   D'2012.01.00 00:00', -14400,
   D'2014.10.26 02:00', -10800,
   D'3000.12.31 23:59', -10800
  };
int SzRussiaTimeSwitch = 67;                    // ArraySize of RussiaTimeSwitch

//+------------------------------------------------------------------+
//| Russian Time Switches                                            |
//+------------------------------------------------------------------+
void offsetRubGMT(const datetime t)
  {
   int i = SzRussiaTimeSwitch; //ArrayRange(RussiaTimeSwitch,0); 66
   while(i-->0 && t < RussiaTimeSwitch[i][0])
      continue;
// t >= RussiaTimeSwitch[i][0]
   nxtSwitch_RUB  = (datetime)RussiaTimeSwitch[fmin(SzRussiaTimeSwitch-1,i+1)][0];
   DST_RUS        = (int)RussiaTimeSwitch[fmin(SzRussiaTimeSwitch-1,i+1)][1];
   return;
  }
//+------------------------------------------------------------------+
```

### The Function that Keeps the Times Updated

Now we come to the last function of this project, the function that keeps the crucial values up to date:

```
//+------------------------------------------------------------------+
//| function to determine broker offset for the time tB given        |
//+------------------------------------------------------------------+
void checkTimeOffset(datetime tB)
  {
   if(tB < nxtSwitch_USD && tB < nxtSwitch_EUR && tB < nxtSwitch_AUD)
      return;                                                  // nothing has changed, return
```

It also first asks right in the beginning whether the time offset (and the next changeover date) must be set. If not, the function is exited immediately.

Otherwise, the values for "EUR", "USD", "AUD", and "RUB" are calculated with the function _nextDST()_ described above:

```
   if(tB>nxtSwitch_USD)
      nextDST("USD", tB);                                      // US has switched
   if(tB>nxtSwitch_EUR)
      nextDST("EUR", tB);                                      // EU has switched
   if(tB>nxtSwitch_AUD)
      nextDST("AUD", tB);                                      // AU has switched
   if(tB>nxtSwitch_RUB)
      nextDST("RUB", tB);                                      // RU has switched
```

"USD" and "EUR" are needed for to determine the broker offset. "AUD" and "RUB" are only needed if a user wants to know them, otherwise one can deactivate them simply by //.

Then, depending on the period, the OffsetBroker.actOffset field must be assigned the broker's time offset valid from that moment and OffsetBroker.actSecFX, the current forex market opening period:

```
   if(DST_USD+DST_EUR==0)                                      // both in winter (normal) time
     {
      OffsetBroker.actOffset = OffsetBroker.USwinEUwin;
      OffsetBroker.actSecFX  = OffsetBroker.secFxWiWi;
     }
   else
      if(DST_USD == DST_EUR)                                   // else both in summer time
        {
         OffsetBroker.actOffset = OffsetBroker.USsumEUsum;
         OffsetBroker.actSecFX  = OffsetBroker.secFxSuSu;
        }
      else
         if(DST_USD != DST_EUR)                                // US:summer EU:winter
           {
            OffsetBroker.actOffset = OffsetBroker.USsumEUwin;
            OffsetBroker.actSecFX  = OffsetBroker.secFxSuWi;
           }
```

That's it. These are all the functions that are necessary to use: Determining the broker's time offsets from a quote and the function that always determines the current time offset and from which GMT and thus any other local time can be easily determined, even in the strategy tester.

We now show two ways to use everything.

### The Script Demonstrating Setting and Use

At first we put it into a (attached) script, _DealingWithTimeScript.mq5_:

```
#include <DealingWithTime.mqh>
//+------------------------------------------------------------------+
//| Finding the broker offsets                                       |
//+------------------------------------------------------------------+
void OnStart()
  {
//--- step 1: set the broker time offsets in winter, summer and in between
   bool isTimeSet = setBokerOffset();
   if(!isTimeSet)
     {
      Alert("setBokerOffset failed");
      return;
     }
```

This will print to the Expert log:

EUR: Fr.2020.10.23 23:00: hNY:16  hGMT:20  hTC:23  hDiff:-3   BrokerTime => GMT: 2020.10.23 20:00 => tNY: 2020.10.23 16:00  End-FX after: 143h

USD: Fr.2020.10.30 22:00: hNY:16  hGMT:20  hTC:22  hDiff:-2   BrokerTime => GMT: 2020.10.30 20:00 => tNY: 2020.10.30 16:00  End-FX after: 142h

NXT: Fr.2020.11.06 23:00: hNY:16  hGMT:21  hTC:23  hDiff:-2   BrokerTime => GMT: 2020.11.06 21:00 => tNY: 2020.11.06 16:00  End-FX after: 143h

USD: Fr.2021.03.12 23:00: hNY:16  hGMT:21  hTC:23  hDiff:-2   BrokerTime => GMT: 2021.03.12 21:00 => tNY: 2021.03.12 16:00  End-FX after: 143h

EUR: Fr.2021.03.26 22:00: hNY:16  hGMT:20  hTC:22  hDiff:-2   BrokerTime => GMT: 2021.03.26 20:00 => tNY: 2021.03.26 16:00  End-FX after: 142h

NXT: Fr.2021.04.02 23:00: hNY:16  hGMT:20  hTC:23  hDiff:-3   BrokerTime => GMT: 2021.04.02 20:00 => tNY: 2021.04.02 16:00  End-FX after: 143h

Time Offset of MetaQuotes Software Corp.:

US=Winter & EU=Winter (USwinEUwin)      =  -7200

US=Summer & EU=Summer (USsumEUsum) = -10800

US=Summer & EU=Winter (USsumEUwin)   =   -7200

Already this enables the user to assign the offsets found to input variables. This is the example of the next chapter. What we see here is already explained so we advance to to the next step of this script. We simulate a pathway from the past to the actual hour of the broker and calculate and print the values of GMT New York and the remaining time the forex market stays open for randomly selected timestamps.

We get the whole history of the 1h timestamps of "EURUSD" with the function _CopyTime("EURUSD",PERIOD\_H1,datetime(0),TimeCurrent(),arr1h)_.

But in order not to drown in tons of data and a long expression for this long period, we only show the data of randomly determined bars. For this we choose how many data there should be. With the given 5 it will be about 10, because in the random average the jump distance is about half of sz/5:

```
//--- step 2: get the quotes (here only 1h time stamps)
   datetime arr1h[], tGMT, tNY, tLeft;
   CopyTime("EURUSD",PERIOD_H1,datetime(0),TimeCurrent(),arr1h);
   int b       = 0,
       sz      = ArraySize(arr1h)-1,                  // oldest time stamp
       nChecks = sz/5,                                // ~2*5+1 randomly chosen bars
       chckBar = MathRand()%nChecks;                  // index of the first bar to check
```

Now we go through all bars, from the oldest to the current one, like in a test or optimization in the strategy tester: _while(++b<=sz)_. The first thing we do is check the time situation for each new bar: checkTimeOffset(arr1h\[b\]). We remember, in this function the first thing that is checked is if a recalculation is necessary, so this call is not very resource intensive despite its frequency:

```
//---  step 3: simulate an EA or an indicator go through the time from the past to now
   while(++b<=sz)
     {
      //--- check the time situation, normally do it at the first bar after the weekend
      checkTimeOffset(arr1h[b]);
```

Now we (only) compute for the bar determined by the random value GMT ( _tGMT_), New York time ( _tNY_) and _tLeft_ the remaining time until the fx-market is closing. Then this is printed and the index of the next bar is calculated:

```
      //--- for a randomly selected bar calc. the times of GMT, NY & tLeft and print them
      if(b>=chckBar || b==sz)
        {
         tGMT  = arr1h[b] + OffsetBroker.actOffset;         // GMT
         tNY   = tGMT - (NYShift+DST_USD);                  // time in New York
         tLeft = OffsetBroker.actSecFX - SoW(arr1h[b]);     // time till FX closes
         PrintFormat("DST_EUR:%+ 6i  DST_EUR:%+ 6i  t[%6i]  tBrk: %s%s  "+
                     "GMT: %s%s  NY: %s%s  End-FX: %2ih => left: %2ih ",
                     DST_EUR,DST_USD,b,
                     DoWs(arr1h[b]),TimeToString(arr1h[b],TIME_DATE|TIME_MINUTES),
                     DoWs(tGMT),TimeToString(tGMT,TIME_DATE|TIME_MINUTES),
                     DoWs(tNY),TimeToString(tNY,TIME_DATE|TIME_MINUTES),
                     OffsetBroker.actSecFX/3600,tLeft/3600
                    );
         chckBar += MathRand()%nChecks;               // calc. the index of the next bar to check
        }
```

Here now the complete printout of this script for a demo account of Metaquotes:

EUR: Fr.2020.10.23 23:00: hNY:16  hGMT:20  hTC:23  hDiff:-3   BrokerTime => GMT: 2020.10.23 20:00 => tNY: 2020.10.23 16:00  End-FX after: 143h

USD: Fr.2020.10.30 22:00: hNY:16  hGMT:20  hTC:22  hDiff:-2   BrokerTime => GMT: 2020.10.30 20:00 => tNY: 2020.10.30 16:00  End-FX after: 142h

NXT: Fr.2020.11.06 23:00: hNY:16  hGMT:21  hTC:23  hDiff:-2   BrokerTime => GMT: 2020.11.06 21:00 => tNY: 2020.11.06 16:00  End-FX after: 143h

USD: Fr.2021.03.12 23:00: hNY:16  hGMT:21  hTC:23  hDiff:-2   BrokerTime => GMT: 2021.03.12 21:00 => tNY: 2021.03.12 16:00  End-FX after: 143h

EUR: Fr.2021.03.26 22:00: hNY:16  hGMT:20  hTC:22  hDiff:-2   BrokerTime => GMT: 2021.03.26 20:00 => tNY: 2021.03.26 16:00  End-FX after: 142h

NXT: Fr.2021.04.02 23:00: hNY:16  hGMT:20  hTC:23  hDiff:-3   BrokerTime => GMT: 2021.04.02 20:00 => tNY: 2021.04.02 16:00  End-FX after: 143h

Time Offset of MetaQuotes Software Corp.:

US=Winter & EU=Winter (USwinEUwin)      =  -7200

US=Summer & EU=Summer (USsumEUsum) = -10800

US=Summer & EU=Winter (USsumEUwin)   =   -7200

DST\_EUR: -3600  DST\_EUR: -3600  t\[ 28194\]  tBrk: Mo.2002.05.20 22:00  GMT: Mo.2002.05.20 19:00  NY: Mo.2002.05.20 15:00  End-FX: 143h => left: 97h

DST\_EUR: -3600  DST\_EUR: -3600  t\[ 40805\]  tBrk: We.2004.05.26 06:00  GMT: We.2004.05.26 03:00  NY: Tu.2004.05.25 23:00  End-FX: 143h => left: 65h

DST\_EUR: -3600  DST\_EUR: -3600  t\[ 42882\]  tBrk: Th.2004.09.23 19:00  GMT: Th.2004.09.23 16:00  NY: Th.2004.09.23 12:00  End-FX: 143h => left: 28h

DST\_EUR:      +0  DST\_EUR:     +0  t\[ 44752\]  tBrk: Tu.2005.01.11 17:00  GMT: Tu.2005.01.11 15:00  NY: Tu.2005.01.11 10:00  End-FX: 143h => left: 78h

DST\_EUR:      +0  DST\_EUR: -3600  t\[ 64593\]  tBrk: We.2008.03.26 03:00  GMT: We.2008.03.26 01:00  NY: Tu.2008.03.25 21:00  End-FX: 142h => left: 67h

DST\_EUR:      +0  DST\_EUR:     +0  t\[ 88533\]  tBrk: Tu.2012.02.07 13:00  GMT: Tu.2012.02.07 11:00  NY: Tu.2012.02.07 06:00  End-FX: 143h => left: 82h

DST\_EUR:      +0  DST\_EUR:     +0  t\[118058\]  tBrk: We.2016.11.16 06:00  GMT: We.2016.11.16 04:00  NY: Tu.2016.11.15 23:00  End-FX: 143h => left: 65h

DST\_EUR: -3600  DST\_EUR: -3600  t\[121841\]  tBrk: Mo.2017.06.26 05:00  GMT: Mo.2017.06.26 02:00  NY: Su.2017.06.25 22:00  End-FX: 143h => left: 114h

DST\_EUR:      +0  DST\_EUR: -3600  t\[144995\]  tBrk: Mo.2021.03.22 06:00  GMT: Mo.2021.03.22 04:00  NY: Mo.2021.03.22 00:00  End-FX: 142h => left: 112h

DST\_EUR: -3600  DST\_EUR: -3600  t\[148265\]  tBrk: Tu.2021.09.28 15:00  GMT: Tu.2021.09.28 12:00  NY: Tu.2021.09.28 08:00  End-FX: 143h => left: 80h

The first two blocks have already been discussed. The third and last part shows for the randomly chosen points of time the respective time differences of EU and USA, the index of the point of time followed by the times of the broker, GMT and New York followed by the opening time of the FX market at this point of time and the remaining time - for clarity converted in hours instead of seconds. This can be quickly checked now: On 5/20/2002, broker time (MQ) is 22:00, daylight saving time applies, GMT = broker-3h = 19:00 and NY = GMT - (5h-1h) = 15:00 and the FX market closes in 97 hours. 97 = 4\*24 (Mon.22:00-Fri.22:00 = 96h) +1h (Fri.22:00-23:00) - qed.

So an EA, indicator that just need the various offsets needs only two function calls:

```
   bool isTimeSet = setBokerOffset();
   if(!isTimeSet)
     {
      Alert("setBokerOffset failed");
      return;
     }
..
   checkTimeOffset(TimeCurrent());
```

### The Alternative using it via Input Variables

Finally, an example of how an EA can use this over its input variables. With the script from above you got the expression:

Time Offset of MetaQuotes Software Corp.:

US=Winter & EU=Winter (USwinEUwin)      =  -7200

US=Summer & EU=Summer (USsumEUsum) = -10800

US=Summer & EU=Winter (USsumEUwin)   =   -7200

Knowing this the EA (not attached, you can copy it from here) would look like:

```
#include <DealingWithTime.mqh>
// offsets of MetaQuotes demo account: DO NOT USE THEM FOR DIFFERENT BROKERS!!
input int   USwinEUwin=  -7200;    // US=Winter & EU=Winter
input int   USsumEUsum= -10800;    // US=Summer & EU=Summer
input int   USsumEUwin=  -7200;    // US=Summer & EU=Winter

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---
   OffsetBroker.USwinEUwin = USwinEUwin;
   OffsetBroker.USsumEUsum = USsumEUsum;
   OffsetBroker.USsumEUwin = USsumEUwin;
   OffsetBroker.actOffset  = WRONG_VALUE;

   nxtSwitch_USD = nxtSwitch_EUR = nxtSwitch_AUD = 0;
   //--- Just a simple test if not ste or changed
   if(OffsetBroker.USwinEUwin+OffsetBroker.USsumEUsum+OffsetBroker.USsumEUwin==0)
      OffsetBroker.set     = false;
   else
      OffsetBroker.set     = true;
   //...
//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//---
   checkTimeOffset(TimeCurrent());
   tGMT  = TimeCurrent() + OffsetBroker.actOffset;    // GMT
   tNY   = tGMT - (NYShift+DST_USD);                  // time in New York
   tLon  = tGMT - (LondonShift+DST_EUR);              // time in London
   tSyd  = tGMT - (SidneyShift+DST_AUD);              // time in Sidney
   tMosc = tGMT - (MoskwaShift+DST_RUS);              // time in Moscow
   tTok  = tGMT - (TokyoShift);                       // time in Tokyo - no DST

   //...

  }
```

Here I used the offests of Metaquotes. Make sure that you use the offsets of your broker!

In OnTick() firstly the time offsets are calculated and right after that GMT and the local times of New York, London, Sydney, Moscow, and Tokyo to show how simple this is now. And don't forget: mind the parentheses.

### Conclusion

Instead of some final words just the results of (only) the function _setBokerOffset()_ applied to demo accounts of several brokers:

```
EUR: Fr.2020.10.23 23:00: hNY:16  hGMT:20  hTC:23  hDiff:-3   BrokerTime => GMT: 2020.10.23 20:00 => tNY: 2020.10.23 16:00  End-FX after: 143h
USD: Fr.2020.10.30 22:00: hNY:16  hGMT:20  hTC:22  hDiff:-2   BrokerTime => GMT: 2020.10.30 20:00 => tNY: 2020.10.30 16:00  End-FX after: 142h
NXT: Fr.2020.11.06 23:00: hNY:16  hGMT:21  hTC:23  hDiff:-2   BrokerTime => GMT: 2020.11.06 21:00 => tNY: 2020.11.06 16:00  End-FX after: 143h
USD: Fr.2021.03.12 23:00: hNY:16  hGMT:21  hTC:23  hDiff:-2   BrokerTime => GMT: 2021.03.12 21:00 => tNY: 2021.03.12 16:00  End-FX after: 143h
EUR: Fr.2021.03.26 22:00: hNY:16  hGMT:20  hTC:22  hDiff:-2   BrokerTime => GMT: 2021.03.26 20:00 => tNY: 2021.03.26 16:00  End-FX after: 142h
NXT: Fr.2021.04.02 23:00: hNY:16  hGMT:20  hTC:23  hDiff:-3   BrokerTime => GMT: 2021.04.02 20:00 => tNY: 2021.04.02 16:00  End-FX after: 143h

Time Offset of MetaQuotes Software Corp.:
US=Winter & EU=Winter (USwinEUwin) = -7200
US=Summer & EU=Summer (USsumEUsum) = -10800
US=Summer & EU=Winter (USsumEUwin) = -7200

EUR: Fr.2020.10.23 23:00: hNY:16  hGMT:20  hTC:23  hDiff:-3   BrokerTime => GMT: 2020.10.23 20:00 => tNY: 2020.10.23 16:00  End-FX after: 143h
USD: Fr.2020.10.30 22:00: hNY:16  hGMT:20  hTC:22  hDiff:-2   BrokerTime => GMT: 2020.10.30 20:00 => tNY: 2020.10.30 16:00  End-FX after: 142h
NXT: Fr.2020.11.06 23:00: hNY:16  hGMT:21  hTC:23  hDiff:-2   BrokerTime => GMT: 2020.11.06 21:00 => tNY: 2020.11.06 16:00  End-FX after: 143h
USD: Fr.2021.03.12 23:00: hNY:16  hGMT:21  hTC:23  hDiff:-2   BrokerTime => GMT: 2021.03.12 21:00 => tNY: 2021.03.12 16:00  End-FX after: 143h
EUR: Fr.2021.03.26 22:00: hNY:16  hGMT:20  hTC:22  hDiff:-2   BrokerTime => GMT: 2021.03.26 20:00 => tNY: 2021.03.26 16:00  End-FX after: 142h
NXT: Fr.2021.04.02 23:00: hNY:16  hGMT:20  hTC:23  hDiff:-3   BrokerTime => GMT: 2021.04.02 20:00 => tNY: 2021.04.02 16:00  End-FX after: 143h

Time Offset of RoboForex Ltd:
US=Winter & EU=Winter (USwinEUwin) = -7200
US=Summer & EU=Summer (USsumEUsum) = -10800
US=Summer & EU=Winter (USsumEUwin) = -7200

EUR: Fr.2020.10.23 23:00: hNY:16  hGMT:20  hTC:23  hDiff:-3   BrokerTime => GMT: 2020.10.23 20:00 => tNY: 2020.10.23 16:00  End-FX after: 143h
USD: Fr.2020.10.30 22:00: hNY:16  hGMT:20  hTC:22  hDiff:-2   BrokerTime => GMT: 2020.10.30 20:00 => tNY: 2020.10.30 16:00  End-FX after: 142h
NXT: Fr.2020.11.06 23:00: hNY:16  hGMT:21  hTC:23  hDiff:-2   BrokerTime => GMT: 2020.11.06 21:00 => tNY: 2020.11.06 16:00  End-FX after: 143h
USD: Fr.2021.03.12 23:00: hNY:16  hGMT:21  hTC:23  hDiff:-2   BrokerTime => GMT: 2021.03.12 21:00 => tNY: 2021.03.12 16:00  End-FX after: 143h
EUR: Fr.2021.03.26 22:00: hNY:16  hGMT:20  hTC:22  hDiff:-2   BrokerTime => GMT: 2021.03.26 20:00 => tNY: 2021.03.26 16:00  End-FX after: 142h
NXT: Fr.2021.04.02 23:00: hNY:16  hGMT:20  hTC:23  hDiff:-3   BrokerTime => GMT: 2021.04.02 20:00 => tNY: 2021.04.02 16:00  End-FX after: 143h

Time Offset of Alpari International:
US=Winter & EU=Winter (USwinEUwin) = -7200
US=Summer & EU=Summer (USsumEUsum) = -10800
US=Summer & EU=Winter (USsumEUwin) = -7200

EUR: Fr.2020.10.23 23:00: hNY:16  hGMT:20  hTC:23  hDiff:-3   BrokerTime => GMT: 2020.10.23 20:00 => tNY: 2020.10.23 16:00  End-FX after: 143h
USD: Fr.2020.10.30 23:00: hNY:16  hGMT:20  hTC:23  hDiff:-3   BrokerTime => GMT: 2020.10.30 20:00 => tNY: 2020.10.30 16:00  End-FX after: 143h
NXT: Fr.2020.11.06 23:00: hNY:16  hGMT:21  hTC:23  hDiff:-2   BrokerTime => GMT: 2020.11.06 21:00 => tNY: 2020.11.06 16:00  End-FX after: 143h
USD: Fr.2021.03.12 23:00: hNY:16  hGMT:21  hTC:23  hDiff:-2   BrokerTime => GMT: 2021.03.12 21:00 => tNY: 2021.03.12 16:00  End-FX after: 143h
EUR: Fr.2021.03.26 23:00: hNY:16  hGMT:20  hTC:23  hDiff:-3   BrokerTime => GMT: 2021.03.26 20:00 => tNY: 2021.03.26 16:00  End-FX after: 143h
NXT: Fr.2021.04.02 23:00: hNY:16  hGMT:20  hTC:23  hDiff:-3   BrokerTime => GMT: 2021.04.02 20:00 => tNY: 2021.04.02 16:00  End-FX after: 143h

Time Offset of Pepperstone Group Limited:
US=Winter & EU=Winter (USwinEUwin) = -7200
US=Summer & EU=Summer (USsumEUsum) = -10800
US=Summer & EU=Winter (USsumEUwin) = -10800

EUR: Fr.2020.10.23 23:00: hNY:16  hGMT:20  hTC:23  hDiff:-3   BrokerTime => GMT: 2020.10.23 20:00 => tNY: 2020.10.23 16:00  End-FX after: 143h
USD: Fr.2020.10.30 23:00: hNY:16  hGMT:20  hTC:23  hDiff:-3   BrokerTime => GMT: 2020.10.30 20:00 => tNY: 2020.10.30 16:00  End-FX after: 143h
NXT: Fr.2020.11.06 23:00: hNY:16  hGMT:21  hTC:23  hDiff:-2   BrokerTime => GMT: 2020.11.06 21:00 => tNY: 2020.11.06 16:00  End-FX after: 143h
USD: Fr.2021.03.12 23:00: hNY:16  hGMT:21  hTC:23  hDiff:-2   BrokerTime => GMT: 2021.03.12 21:00 => tNY: 2021.03.12 16:00  End-FX after: 143h
EUR: Fr.2021.03.26 23:00: hNY:16  hGMT:20  hTC:23  hDiff:-3   BrokerTime => GMT: 2021.03.26 20:00 => tNY: 2021.03.26 16:00  End-FX after: 143h
NXT: Fr.2021.04.02 23:00: hNY:16  hGMT:20  hTC:23  hDiff:-3   BrokerTime => GMT: 2021.04.02 20:00 => tNY: 2021.04.02 16:00  End-FX after: 143h

Time Offset of Eightcap Pty Ltd:
US=Winter & EU=Winter (USwinEUwin) = -7200
US=Summer & EU=Summer (USsumEUsum) = -10800
US=Summer & EU=Winter (USsumEUwin) = -10800

EUR: Fr.2020.10.23 23:00: hNY:16  hGMT:20  hTC:23  hDiff:-3   BrokerTime => GMT: 2020.10.23 20:00 => tNY: 2020.10.23 16:00  End-FX after: 143h
USD: Fr.2020.10.30 22:00: hNY:16  hGMT:20  hTC:22  hDiff:-2   BrokerTime => GMT: 2020.10.30 20:00 => tNY: 2020.10.30 16:00  End-FX after: 142h
NXT: Fr.2020.11.06 23:00: hNY:16  hGMT:21  hTC:23  hDiff:-2   BrokerTime => GMT: 2020.11.06 21:00 => tNY: 2020.11.06 16:00  End-FX after: 143h
USD: Fr.2021.03.12 23:00: hNY:16  hGMT:21  hTC:23  hDiff:-2   BrokerTime => GMT: 2021.03.12 21:00 => tNY: 2021.03.12 16:00  End-FX after: 143h
EUR: Fr.2021.03.26 22:00: hNY:16  hGMT:20  hTC:22  hDiff:-2   BrokerTime => GMT: 2021.03.26 20:00 => tNY: 2021.03.26 16:00  End-FX after: 142h
NXT: Fr.2021.04.02 23:00: hNY:16  hGMT:20  hTC:23  hDiff:-3   BrokerTime => GMT: 2021.04.02 20:00 => tNY: 2021.04.02 16:00  End-FX after: 143h

Time Offset of InstaForex Companies Group:
US=Winter & EU=Winter (USwinEUwin) = -7200
US=Summer & EU=Summer (USsumEUsum) = -10800
US=Summer & EU=Winter (USsumEUwin) = -7200

EUR: Fr.2020.10.23 21:00: hNY:16  hGMT:20  hTC:21  hDiff:-1   BrokerTime => GMT: 2020.10.23 20:00 => tNY: 2020.10.23 16:00  End-FX after: 141h
USD: Fr.2020.10.30 21:00: hNY:16  hGMT:20  hTC:21  hDiff:-1   BrokerTime => GMT: 2020.10.30 20:00 => tNY: 2020.10.30 16:00  End-FX after: 141h
NXT: Fr.2020.11.06 21:00: hNY:16  hGMT:21  hTC:21  hDiff: 0   BrokerTime => GMT: 2020.11.06 21:00 => tNY: 2020.11.06 16:00  End-FX after: 141h
USD: Fr.2021.03.12 21:00: hNY:16  hGMT:21  hTC:21  hDiff: 0   BrokerTime => GMT: 2021.03.12 21:00 => tNY: 2021.03.12 16:00  End-FX after: 141h
EUR: Fr.2021.03.26 21:00: hNY:16  hGMT:20  hTC:21  hDiff:-1   BrokerTime => GMT: 2021.03.26 20:00 => tNY: 2021.03.26 16:00  End-FX after: 141h
NXT: Fr.2021.04.02 21:00: hNY:16  hGMT:20  hTC:21  hDiff:-1   BrokerTime => GMT: 2021.04.02 20:00 => tNY: 2021.04.02 16:00  End-FX after: 141h

Time Offset of JFD Group Ltd:
US=Winter & EU=Winter (USwinEUwin) = 0
US=Summer & EU=Summer (USsumEUsum) = -3600
US=Summer & EU=Winter (USsumEUwin) = -3600
```

**May trading enrich you. :)**

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/9929.zip "Download all attachments in the single ZIP archive")

[DealingWithTimeScript.mq5](https://www.mql5.com/en/articles/download/9929/dealingwithtimescript.mq5 "Download DealingWithTimeScript.mq5")(2.81 KB)

[DealingWithTime.mqh](https://www.mql5.com/en/articles/download/9929/dealingwithtime.mqh "Download DealingWithTime.mqh")(52.89 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Dealing with Time (Part 1): The Basics](https://www.mql5.com/en/articles/9926)
- [Cluster analysis (Part I): Mastering the slope of indicator lines](https://www.mql5.com/en/articles/9527)
- [Enhancing the StrategyTester to Optimize Indicators Solely on the Example of Flat and Trend Markets](https://www.mql5.com/en/articles/2118)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/379233)**
(37)


![Carl Schreiber](https://c.mql5.com/avatar/2018/2/5A745EEE-EB76.PNG)

**[Carl Schreiber](https://www.mql5.com/en/users/gooly)**
\|
5 Jul 2023 at 19:08

The version of _DealingWithTime.mqh_ v. 1.01 from the article Dealing with Time (Part 2): Functions ( [https://www.mql5.com/en/articles/9929](https://www.mql5.com/en/articles/9929) ) stopped working because MQ changed the behaviour of _CopyTime( )_ some time after this article was published. Now this function no longer returns future time values if they are greater than the _TimeCurrent( )_ specified for the start\_time and/or stop\_time parameters. Instead, it returns the maximum possible opening time value of the last current bar.

Since the end of the currency session was defined in such a way as to determine the broker's time offset, this now results in incorrect values!

This calculation has been changed in version 2.03. This version is now available in CodeBase here: [https://www.mql5.com/en/code/45287](https://www.mql5.com/en/code/45287).

But also the time translation calculation has been completely changed, so that the complex time translation times from Sydney, Australia back to the 70s are now covered.

Also attached is the _DST_ table _1975 - 2030.xlsx_ as a zip file with all the time changes since the 70s so everyone can check the formulas are working correctly, here is an example of the table series:

[![](https://c.mql5.com/18/147/3335251937667__1.png)](https://c.mql5.com/18/147/3335251937667.png "https://c.mql5.com/18/147/3335251937667.png")

1 January 1982. - US Standard Time (DST == 0), and the next change is on 25 April 1982, the last (25th of the month) Sunday in April (4). The table is already sorted by geographic time zone (column A), then by time zone of the year (column L, spr=spring, aut=autumn), and finally by the date of the query (column C). The spreadsheet can be created automatically by the enabled EA(the script cannot be run in debug mode). _Test\_DST 2.mq5_ if you run it in debug mode and copy the log lines in the debugger and paste them into the spreadsheet; The cell separator will be a space.

In addition, there is now a new simple function _SecTillClose()_, which gives you the remaining time in seconds (MQ time currency) until the forex market closes - without _CopyTime()_. This is interesting for those who want to close their positions before the weekend or don't want to open a new position in a certain period before the weekend.

The included indicator _DealingWithTime\_TestIndi.mq5_, as a comment on the chart, shows not only daylight saving time in Europe, USA and Australia (Sydney), but also the current time and time difference of different cities. Here you can find a table with different local times of major cities for comparison: [https://www.timeanddate.com/worldclock/](https://www.mql5.com/go?link=https://www.timeanddate.com/worldclock/ "https://www.timeanddate.com/worldclock/"). This way you can check the values at any time. This indicator also shows how these values are defined and used (what is subtracted or added from what), which makes it easy to use on your own - copy and paste, the fastest form of programming.

The last two lines also show the last second of the current FX session and the time remaining in hours (which is easier to judge) and seconds. In New York, when the FX session closes at 17:00 local time on Friday, there is no valid bar open at 17:00 New York time. Therefore, this function subtracts 1 second to get the last valid open time of the last bar in the broker's time. However, some brokers end their currency session a few minutes early, no longer providing prices or accepting trade orders.

![](https://c.mql5.com/3/412/4814139301946.png)

![Daniel K](https://c.mql5.com/avatar/2022/12/638C7DE5-0DFD.jpg)

**[Daniel K](https://www.mql5.com/en/users/danielkay)**
\|
18 Feb 2024 at 16:37

Hi [@Anil Varma](https://www.mql5.com/en/users/anilvarma)

I read the last post of the author [@Carl Schreiber](https://www.mql5.com/en/users/gooly) about _CopyTime()_ function but since I'm finding more understandable the 1st version I'm still using  _DealingWithTime.mqh_v. 1.01.

In my indicator I want to:

```
Assign NY raw time seconds), NY hour and NY minute to each bar using the following buffers in order to displaying them in the data window:
double      NyRawTimeBuffer[];
double      NyHourBuffer[];
double      NyMinuteBuffer[];
```

```
 void AssignNyTime (const datetime& time[],int rates_total)
   {

      MqlDateTime dT_struc;

      //--- Assign too each candle: NY raw Time (in seconds), NY hour, NY min
      ArraySetAsSeries(time,true);
      for(int z=0;z<rates_total;z++)
         {
            checkTimeOffset(time[z]);                   // check changes of DST
            datetime tC, tGMT, tNY;
            tC    = time[z];
            tGMT  = time[z] + OffsetBroker.actOffset;   // GMT
            tNY   = tGMT - (NYShift+DST_USD);           // time in New York
            int j = int (tNY);                          // casting datetime to int
            NyRawTimeBuffer[z]=j;

            TimeToStruct(tNY,dT_struc);
            NyHourBuffer[z]=dT_struc.hour;
            NyMinuteBuffer[z]=dT_struc.min;

         }

   return;
   }
```

The function works just when the timeframe chart where the terminal starts it's set to H1.

If I close the terminal and the timeframe is set to let's say M5, and then I restart the terminal it gives me the following error:

_2024.02.18 15:33:38.048MyFractals\_V4 (EURUSD,M5)240: CopyTime() FAILED for EURUSD H1: need times from 2024.02.12 02:00:00, but there are only from 1970.01.01 00:00:00 error: 440_ 1

You already suggested me through messaging to use _CheckLoadHistory()_ from this article ( https://www.mql5.com/en/code/1251 ) and placing it before CopyXXXX() function in your library:

```
//--- find the broker offset
    OffsetBroker.set = false;

    CheckLoadHistory("EURUSD",PERIOD_H1,TERMINAL_MAXBARS,true);

    b = CopyTime("EURUSD",PERIOD_H1,BegWk+26*3600,5,arrTme);      // get time last 1h bar before switch in EU
```

But the issue is still there.

In the _checkhistory.mqh_ (row 19) I noted the following comment but I don't understand if it could be an issue. I tried to comment it ad test the program again but did not work.

```
//--- don't ask for load of its own data if it is an indicator
   if(MQL5InfoInteger(MQL5_PROGRAM_TYPE)==PROGRAM_INDICATOR && Period()==period && Symbol()==symbol) return(true);
```

Is there a way to adjust the bug without to switch to the updated library _DealingWithTimeV2.03.mqh_ without to re-write all the code?

![Anil Varma](https://c.mql5.com/avatar/2023/11/6561abd1-95bd.jpg)

**[Anil Varma](https://www.mql5.com/en/users/anilvarma)**
\|
19 Feb 2024 at 05:53

**Nauris Zukas [#](https://www.mql5.com/en/forum/379233#comment_28816162):**

Hello,

As I understand from the article, the function "setBokerOffset ()" should work in the [strategy tester](https://www.mql5.com/en/articles/239 "Article: The Fundamentals of Testing in MetaTrader 5 ") as well, but it doesn't work.

Does the "The Alternative using it via Input Variables" the only way to get correct times in the strategy tester?

Hi

I have tried to modify the code as below and so far it is working for me. Note that I have converted the Class with constructors and all methods are part of class. Class needs to be called in and initialized in your EA/Strategy Class.

CDealWithTime.OnTick() should be placed in EA/Strategy OnTick()

```
//+-----------------------------------------------------------------------------------------------------------------------------+
//| function to determin broker offset for the time given (tB)
//+-----------------------------------------------------------------------------------------------------------------------------+
void CDealWithTime::OnTick(void) {

                string vMethod = __FUNCTION__ + " Line[" + (string)__LINE__ + "] ";

                // Perform Hourly update of TimeH01[]
                if(IsNewBarH01()) {
      // ... BegWk = BoW(tC), Original code
                        datetime weekBegin = BoW(TimeCurrent());
                        // Original code added 5 bars to WeakBegin Time.
                        // Adding 5 days * 3600 secondsInHour to weak begin time, is not equal to five bars, as there may not be bars on
                        // weekend/holiday. Hence WeekBegin+5*3600 may result a time while there is no bar on it.
                        int                      barShift  = iBarShift(mSymbol,mTimeFrame,weekBegin+(5*3600));
                        // Will return first available bar at Time(weekBegin+(5*3600)
                        datetime timeStop        = iTime(mSymbol,mTimeFrame,barShift);          // Last One Hour bar of Friday, before switch in EU.
                        // Result when seconds added (weekBegin)+(5*3600): CDealWithTime::OnTick Line[229] : GetLastError[0] copiedTime EURUSD-PERIOD_H1 for [0] bars weekBegin[2024.01.01 02:00] to timeStop[2023.12.31 21:00] time5th[2023.12.29 23:00]
                        int bars = Bars(mSymbol,mTimeFrame,weekBegin,timeStop);

                        // We need while..loop, as IsNewBarH01() will be checked only once on the Tick, if true then untill next new bar no further check
                        ResetLastError();
                        int attempt = 0;
                        while(CopyTime(mSymbol,mTimeFrame,weekBegin,timeStop,TimeH01) != bars && attempt <= 10) {
                                Sleep(100);
                                attempt++;
                        }
                        if(attempt > 0) {
                                PrintFormat("%s: GetLastError[%i] copiedTime %s-%s for [%i] bars weekBegin[%s] to timeStop[%i][%s]",vMethod,GetLastError(),mSymbol,EnumToString(mTimeFrame),bars,TimeToString(weekBegin),barShift,TimeToString(timeStop));
                        }
                }

                // Perform a weekly check, if there is change in Day Light Saving Times (DST)
                if(IsNewBarW01()) {
                        checkTimeOffset(TimeCurrent());
                        int  attempt  = 0;
                        bool isOffset = false;
                        do{
                                isOffset = setBokerOffset();
                                attempt++;
                        } while(!isOffset && attempt <= 10);
                }

} // End of OnTick()
```

![Anil Varma](https://c.mql5.com/avatar/2023/11/6561abd1-95bd.jpg)

**[Anil Varma](https://www.mql5.com/en/users/anilvarma)**
\|
19 Feb 2024 at 05:58

**amrali [#](https://www.mql5.com/en/forum/379233#comment_44157935):**

This code calculates the DST automatically for European and US brokers:

[https://www.mql5.com/en/code/27860](https://www.mql5.com/en/code/27860)

The above code was used in Forex Market Hours [https://www.mql5.com/en/code/27771](https://www.mql5.com/en/code/27771) to calculate the day-light saving time changes.

Similar functions can be constructed for different areas of the world.

Hi Amrali

Nice and simple code as an alternative to DealingWithTime v2.03 article. Will look into it for more detailed study.

![Anil Varma](https://c.mql5.com/avatar/2023/11/6561abd1-95bd.jpg)

**[Anil Varma](https://www.mql5.com/en/users/anilvarma)**
\|
19 Feb 2024 at 06:08

**Daniel K [#](https://www.mql5.com/en/forum/379233/page2#comment_52324101):**

_DealingWithTime.mqh_v. 1.01.

Hi Daniel

_DealingWithTime.mqh_v. 1.01. This article and its code does not work any more due to changes in MQL calculation methods, as explained by Carl in _DealingWithTime.mqh_ v 2.03 article [https://www.mql5.com/en/code/45287](https://www.mql5.com/en/code/45287)

You should not be using it at all.

![Graphics in DoEasy library (Part 83): Class of the abstract standard graphical object](https://c.mql5.com/2/43/MQL5-avatar-doeasy-library3-2__2.png)[Graphics in DoEasy library (Part 83): Class of the abstract standard graphical object](https://www.mql5.com/en/articles/9902)

In this article, I will create the class of the abstract graphical object. This object is to serve as a basis for creating the class of standard graphical objects. Graphical objects feature multiple properties. Therefore, I will need to do a lot of preparatory work before actually creating the abstract graphical object class. This work includes setting the properties in the library enumerations.

![Combinatorics and probability theory for trading (Part III): The first mathematical model](https://c.mql5.com/2/43/gix1_2.png)[Combinatorics and probability theory for trading (Part III): The first mathematical model](https://www.mql5.com/en/articles/9570)

A logical continuation of the earlier discussed topic would be the development of multifunctional mathematical models for trading tasks. In this article, I will describe the entire process related to the development of the first mathematical model describing fractals, from scratch. This model should become an important building block and be multifunctional and universal. It will build up our theoretical basis for further development of this idea.

![Better Programmer (Part 06): 9 habits that lead to effective coding](https://c.mql5.com/2/43/coding_pro.png)[Better Programmer (Part 06): 9 habits that lead to effective coding](https://www.mql5.com/en/articles/9923)

It's not always all about writing the code that leads to effective coding. There are certain habits that I have found in my experience that lead to effective coding. We are going to discuss some of them in detail in this article. This is a must-read article for every programmer who wants to improve their ability to write complex algorithms with less hassle.

![Dealing with Time (Part 1): The Basics](https://c.mql5.com/2/43/mql5-dealing-with-time.png)[Dealing with Time (Part 1): The Basics](https://www.mql5.com/en/articles/9926)

Functions and code snippets that simplify and clarify the handling of time, broker offset, and the changes to summer or winter time. Accurate timing may be a crucial element in trading. At the current hour, is the stock exchange in London or New York already open or not yet open, when does the trading time for Forex trading start and end? For a trader who trades manually and live, this is not a big problem.

[![](https://www.mql5.com/ff/sh/x8fwvn495ta7y774z2/01.png)Does your broker offer sponsored hosting for trading?Now it's even easier to get MetaTrader VPS for free – contact your broker for details](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=xscnzeyhifcgygpwvysykhqydcmmbgpp&s=f87b748147e376d34c8f0fdb9737b1766f20cc2174769a0e6b9975b5c2e8ddae&uid=&ref=https://www.mql5.com/en/articles/9929&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5082938368888672703)

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