---
title: Trading with the MQL5 Economic Calendar (Part 9): Elevating News Interaction with a Dynamic Scrollbar and Polished Display
url: https://www.mql5.com/en/articles/18135
categories: Trading, Trading Systems, Expert Advisors
relevance_score: -2
scraped_at: 2026-01-24T14:15:20.112830
---

[What's wrong with regular VPS?Here are the 8 most common problems that algorithmic traders may encounterRead![](https://www.mql5.com/ff/sh/hzatb686qjqxwtr4z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/450486&a=drhremihlwuaqyvgpzfddbtmgciejpba&s=c37d25bcceb93ed153b814e6ba4d4839461a9b2d68dd82b95b142be06d310f3f&uid=&ref=https://www.mql5.com/en/articles/18135&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5083441343918775237)

MetaTrader 5 / Trading


### Introduction

In this article, we advance the MQL5 Economic Calendar series by introducing a vertical dynamic [scrollbar](https://en.wikipedia.org/wiki/Scrollbar "https://en.wikipedia.org/wiki/Scrollbar") and polished display to enhance trader interaction with news events, ensuring intuitive navigation and reliable event presentation. Building on [Part 8’s optimized backtesting and smart filtering](https://www.mql5.com/en/articles/17999), we now focus on a responsive [User Interface](https://en.wikipedia.org/wiki/User_interface "https://en.wikipedia.org/wiki/User_interface") (UI) with a scrollbar that visually signals clickable states, streamlining access to economic news for both live and backtesting environments. We structure the article with the following topics:

1. [Crafting a Dynamic Scrollbar for Effortless News Exploration](https://www.mql5.com/en/articles/18135#para1)
2. [Implementation in MQL5](https://www.mql5.com/en/articles/18135#para2)
3. [Testing and Validation](https://www.mql5.com/en/articles/18135#para3)
4. [Conclusion](https://www.mql5.com/en/articles/18135#para4)

Let’s dive into these enhancements!

### Crafting a Dynamic Scrollbar for Effortless News Exploration

To elevate our interaction with the [MQL5 Economic Calendar](https://www.mql5.com/en/book/advanced/calendar), we envision a dynamic scrollbar as the cornerstone of intuitive news navigation. We plan to design a responsive vertical scrollbar that visually signals clickable states, paired with a robust event storage system to ensure seamless access to all filtered news, transforming the dashboard into a fluid, trader-centric tool. We will eliminate the minimum displayable events and list all the filtered events so one can scroll through all the news when necessary, eliminating the restriction to visualization on only a set of prioritized events. Here’s how we will achieve this:

- Dynamic Scrollbar Design: We will implement a scrollbar with icons that shift between black for clickable states and light gray for non-clickable, providing instant visual feedback to guide navigation through extensive event lists.
- Reliable Event Storage: We will develop a system to store all filtered events, guaranteeing that every news item is accessible via the scrollbar, eliminating display limitations and ensuring a comprehensive view.
- Efficient Update Mechanism: We will optimize the dashboard to redraw only when filters change, new events appear, or scrolling occurs, maintaining a smooth and distraction-free experience.
- Polished User Interface (UI) Refinements: We will enhance the interface with precise layout adjustments, ensuring the scrollbar and event display integrate seamlessly for a professional, user-friendly design.

This strategic roadmap will set the stage for a dashboard that empowers us to explore economic news effortlessly. The dynamic scrollbar will be the key to unlocking a refined trading experience. In a nutshell, here is a visualization of what we aim to achieve.

![PLAN VISUAL](https://c.mql5.com/2/142/Screenshot_2025-05-14_224520.png)

### Implementation in MQL5

To make the improvements in MQL5, we first will need to define the extra scrollbar objects using the [#define](https://www.mql5.com/en/docs/basis/preprosessor/constant) directive alongside their layout constants, some variables to track the scroll status, and events to be processed, and then [arrays](https://www.mql5.com/en/book/basis/arrays/arrays_usage) to store the events data that will be processed seamlessly as below.

```
// Scrollbar UI elements
#define SCROLL_UP_REC "SCROLL_UP_REC"
#define SCROLL_UP_LABEL "SCROLL_UP_LABEL"
#define SCROLL_DOWN_REC "SCROLL_DOWN_REC"
#define SCROLL_DOWN_LABEL "SCROLL_DOWN_LABEL"
#define SCROLL_LEADER "SCROLL_LEADER"
#define SCROLL_SLIDER "SCROLL_SLIDER"

//---- Scrollbar layout constants
#define LIST_X 62
#define LIST_Y 162
#define LIST_WIDTH 716
#define LIST_HEIGHT 286
#define VISIBLE_ITEMS 11
#define ITEM_HEIGHT 26
#define SCROLLBAR_X (LIST_X + LIST_WIDTH + 2) // 780
#define SCROLLBAR_Y LIST_Y
#define SCROLLBAR_WIDTH 20
#define SCROLLBAR_HEIGHT LIST_HEIGHT // 286
#define BUTTON_SIZE 15
#define BUTTON_WIDTH (SCROLLBAR_WIDTH - 2)
#define BUTTON_OFFSET_X 1
#define SCROLL_AREA_HEIGHT (SCROLLBAR_HEIGHT - 2 * BUTTON_SIZE)
#define SLIDER_MIN_HEIGHT 20
#define SLIDER_WIDTH 18
#define SLIDER_OFFSET_X 1

//---- Event name tracking
string current_eventNames_data[];
string previous_eventNames_data[];
string last_dashboard_eventNames[];
string previous_displayable_eventNames[];
string current_displayable_eventNames[];
datetime last_dashboard_update = 0;

//---- Filter flags
bool enableCurrencyFilter = true;
bool enableImportanceFilter = true;
bool enableTimeFilter = true;
bool isDashboardUpdate = true;
bool filters_changed = true;

//---- Scrollbar flags and variables
bool scroll_visible = false;
bool moving_state_slider = false;
int scroll_pos = 0;
int prev_scroll_pos = -1; // Track previous scroll position
int mlb_down_x = 0;
int mlb_down_y = 0;
int mlb_down_yd_slider = 0;
int prev_mouse_state = 0;
int slider_height = SLIDER_MIN_HEIGHT;

//---- Event counters
int totalEvents_Considered = 0;
int totalEvents_Filtered = 0;
int totalEvents_Displayable = 0;

//---- Global arrays for events
EconomicEvent allEvents[];
EconomicEvent filteredEvents[];
EconomicEvent displayableEvents[];
```

Here, we lay the groundwork for the MQL5 Economic Calendar’s dynamic scrollbar and polished event display by defining essential UI constants, variables, and arrays to enable intuitive navigation and efficient event handling. We establish scrollbar components using constants like "SCROLL\_UP\_LABEL", "SCROLL\_DOWN\_LABEL", "SCROLL\_UP\_REC", and "SCROLL\_DOWN\_REC", which identify the graphical elements for the scrollbar’s up and down buttons.

Layout constants such as "LIST\_X" (62), "LIST\_Y" (162), "LIST\_WIDTH" (716), and "LIST\_HEIGHT" (286) define the event display area, while "SCROLLBAR\_X" (780), "SCROLLBAR\_Y" (162), "SCROLLBAR\_WIDTH" (20), and "SCROLLBAR\_HEIGHT" (286) position the scrollbar precisely, with "VISIBLE\_ITEMS" (11) and "ITEM\_HEIGHT" (26) ensuring 11 events are shown, each 26 [pixels](https://en.wikipedia.org/wiki/Pixel "https://en.wikipedia.org/wiki/Pixel") tall, and "BUTTON\_SIZE" (15) and "SLIDER\_WIDTH" (18) shaping the compact buttons and slider.

To manage events and interactions, we declare arrays like "current\_displayable\_eventNames" and "previous\_displayable\_eventNames" to track event names for change detection, supporting silent updates, and "last\_dashboard\_update" to timestamp dashboard refreshes. Filter flags "enableCurrencyFilter", "enableImportanceFilter", and "enableTimeFilter" (all true) control event selection, while "isDashboardUpdate" and "filters\_changed" dictate when updates occur. Scrollbar variables, including "scroll\_visible", "scroll\_pos", "prev\_scroll\_pos", and "moving\_state\_slider", track visibility and position, with "mlb\_down\_x", "mlb\_down\_y", and "slider\_height" enabling slider drag functionality.

We use counters "totalEvents\_Considered", "totalEvents\_Filtered", and "totalEvents\_Displayable" to monitor event processing, and arrays "allEvents", "filteredEvents", and "displayableEvents" to store event data, ensuring all filtered news is navigable, setting the stage for a responsive trader interface. We can then first create the scroll bar elements using existing creation functions, and then adjust the main rectangle sizes and positions to contain the scrollbar on the right side.

```
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit() {
   // Enable mouse move events for scrollbar
   ChartSetInteger(0, CHART_EVENT_MOUSE_MOVE, true);

   // Create dashboard UI
   createRecLabel(MAIN_REC,50,50,740+13,410,clrSeaGreen,1);
   createRecLabel(SUB_REC1,50+3,50+30,740-3-3+13,410-30-3,clrWhite,1);
   createRecLabel(SUB_REC2,50+3+5,50+30+50+27,740-3-3-5-5,410-30-3-50-27-10+5,clrGreen,1);
   createLabel(HEADER_LABEL,50+3+5,50+5,"MQL5 Economic Calendar",clrWhite,15);

   //---

}
```

Here, on the [OnInit](https://www.mql5.com/en/docs/event_handlers/oninit) event handler, we use the [ChartSetInteger](https://www.mql5.com/en/docs/chart_operations/chartsetinteger) function to initialize the mouse move event to true, so that we can control the scroll priority of the main chart and our custom objects. This will allow us to scroll the vertical scrollbar and its elements, allowing seamless movement and transition. We then adjust the main rectangle and sub-rectangle 1 width by adding 13 pixels to it, so it can also house the vertical scrollbar. We extend the sub-rectangle 2's height by 5 pixels to accommodate all the 11 events efficiently, eliminating the outflow effect. Upon compilation, we have the following outcome.

![ACCOMODATION CHANGES](https://c.mql5.com/2/142/Screenshot_2025-05-14_185835.png)

From the image, we can see all the accommodation changes we made, labeled from 1 to 3. Change 1 indicates the push of the cancel button to the edge of the panel, then change 2 shows the adjustment of the main main rectangles width to accommodate the cancel and the scrollbar and then change 3 shows the adjustment if the height of the dashboard rectangle to accommodate the overflow of the last element row holder. We can now proceed to defining and creating the scrollbar within the space created. However, since we want to make the slider dynamic and show only when needed, we will need to define some functions to enable that.

```
//+------------------------------------------------------------------+
//| Calculate slider height                                          |
//+------------------------------------------------------------------+
int calculateSliderHeight() {
   if (totalEvents_Filtered <= VISIBLE_ITEMS)
      return SCROLL_AREA_HEIGHT;
   double visible_ratio = (double)VISIBLE_ITEMS / totalEvents_Filtered;
   int height = (int)::floor(SCROLL_AREA_HEIGHT * visible_ratio);
   return MathMax(SLIDER_MIN_HEIGHT, MathMin(height, SCROLL_AREA_HEIGHT));
}

//+------------------------------------------------------------------+
//| Update slider position                                           |
//+------------------------------------------------------------------+
void updateSliderPosition() {
   int max_scroll = MathMax(0, ArraySize(displayableEvents) - VISIBLE_ITEMS);
   if (max_scroll <= 0) return;
   double scroll_ratio = (double)scroll_pos / max_scroll;
   int scroll_area_y_min = SCROLLBAR_Y + BUTTON_SIZE;
   int scroll_area_y_max = scroll_area_y_min + SCROLL_AREA_HEIGHT - slider_height;
   int new_y = scroll_area_y_min + (int)(scroll_ratio * (scroll_area_y_max - scroll_area_y_min));
   ObjectSetInteger(0, SCROLL_SLIDER, OBJPROP_YDISTANCE, new_y);
   if (debugLogging) Print("Slider moved to y=", new_y);
   ChartRedraw(0);
}

//+------------------------------------------------------------------+
//| Update button colors based on scroll position                    |
//+------------------------------------------------------------------+
void updateButtonColors() {
   int max_scroll = MathMax(0, ArraySize(displayableEvents) - VISIBLE_ITEMS);
   if (scroll_pos == 0) {
      ObjectSetInteger(0, SCROLL_UP_LABEL, OBJPROP_COLOR, clrLightGray);
   } else {
      ObjectSetInteger(0, SCROLL_UP_LABEL, OBJPROP_COLOR, clrBlack);
   }
   if (scroll_pos >= max_scroll) {
      ObjectSetInteger(0, SCROLL_DOWN_LABEL, OBJPROP_COLOR, clrLightGray);
   } else {
      ObjectSetInteger(0, SCROLL_DOWN_LABEL, OBJPROP_COLOR, clrBlack);
   }
   ChartRedraw(0);
}
```

We enhance the MQL5 Economic Calendar’s dynamic scrollbar by implementing three key functions—"calculateSliderHeight", "updateSliderPosition", and "updateButtonColors"—to ensure intuitive navigation and clear visual feedback. In the "calculateSliderHeight" function, we determine the height of the scrollbar’s slider to visually represent the proportion of visible events relative to the total filtered events.

We check if "totalEvents\_Filtered" is less than or equal to "VISIBLE\_ITEMS" (11), returning "SCROLL\_AREA\_HEIGHT" (256 pixels) to fill the scroll area if all events fit in one view. Otherwise, we compute a "visible\_ratio" by dividing "VISIBLE\_ITEMS" by "totalEvents\_Filtered", multiply it by "SCROLL\_AREA\_HEIGHT", and use the [floor](https://www.mql5.com/en/docs/math/mathfloor) function to get an integer "height". We then return the maximum of "SLIDER\_MIN\_HEIGHT" (20 pixels) and the minimum of "height" and "SCROLL\_AREA\_HEIGHT", ensuring the slider is appropriately sized for the event list’s scale.

In the "updateSliderPosition" function, we position the scrollbar’s slider to reflect the current scroll position within the event list. We calculate "max\_scroll" as the difference between " [ArraySize](https://www.mql5.com/en/docs/array/arraysize)(displayableEvents)" and "VISIBLE\_ITEMS", using [MathMax](https://www.mql5.com/en/docs/math/mathmax) to ensure it’s non-negative, and exit if "max\_scroll" is zero (no scrolling needed). We compute a "scroll\_ratio" by dividing "scroll\_pos" by "max\_scroll", define the slider’s vertical range with "scroll\_area\_y\_min" ("SCROLLBAR\_Y" + "BUTTON\_SIZE") and "scroll\_area\_y\_max" ("scroll\_area\_y\_min" + "SCROLL\_AREA\_HEIGHT" - "slider\_height"), and calculate a "new\_y" position by interpolating "scroll\_ratio" within this range.

We then use [ObjectSetInteger](https://www.mql5.com/en/docs/objects/ObjectSetInteger) to set "SCROLL\_SLIDER"’s "OBJPROP\_YDISTANCE" to "new\_y", log the movement if "debugLogging" is true, and call [ChartRedraw](https://www.mql5.com/en/docs/chart_operations/ChartRedraw) to update the display.

In the "updateButtonColors" function, we dynamically update the colors of the scrollbar’s up and down button icons to indicate their clickable states, enhancing user feedback. We calculate "max\_scroll" as in "updateSliderPosition" and check "scroll\_pos" to determine the state of "SCROLL\_UP\_LABEL" and "SCROLL\_DOWN\_LABEL". If "scroll\_pos" is 0, we set "SCROLL\_UP\_LABEL" and "OBJPROP\_COLOR" to "clrLightGray" (non-clickable, at top); otherwise, we set it to "clrBlack" (clickable). Similarly, if "scroll\_pos" equals or exceeds "max\_scroll", we set "SCROLL\_DOWN\_LABEL" to "clrLightGray" (non-clickable, at bottom); otherwise, we set it to "clrBlack" (clickable).

We conclude by calling "ChartRedraw" to refresh the chart, ensuring the icons visually guide the navigation. We can now create the scrollbar dynamically when updating the dashboard values, as shown below.

```
// Update TIME_LABEL
string timeText = updateServerTime ? "Server Time: "+TimeToString(TimeCurrent(),TIME_DATE|TIME_SECONDS) : "Server Time: Static";
updateLabel(TIME_LABEL,timeText+"   |||   Total News: "+IntegerToString(totalEvents_Filtered)+"/"+IntegerToString(totalEvents_Considered));

// Update scrollbar visibility
bool new_scroll_visible = totalEvents_Filtered > VISIBLE_ITEMS;
if (new_scroll_visible != scroll_visible || events_changed || filters_changed) {
   scroll_visible = new_scroll_visible;
   if (debugLogging) Print("Scrollbar visibility: ", scroll_visible ? "Visible" : "Hidden");
   if (scroll_visible) {
      if (ObjectFind(0, SCROLL_LEADER) < 0) {
         createRecLabel(SCROLL_LEADER, SCROLLBAR_X, SCROLLBAR_Y, SCROLLBAR_WIDTH, SCROLLBAR_HEIGHT, clrSilver, 1, clrNONE);
         int max_scroll = MathMax(0, ArraySize(displayableEvents) - VISIBLE_ITEMS);
         color up_color = (scroll_pos == 0) ? clrLightGray : clrBlack;
         color down_color = (scroll_pos >= max_scroll) ? clrLightGray : clrBlack;
         createRecLabel(SCROLL_UP_REC, SCROLLBAR_X + BUTTON_OFFSET_X, SCROLLBAR_Y, BUTTON_WIDTH, BUTTON_SIZE, clrDarkGray, 1, clrDarkGray);
         createLabel(SCROLL_UP_LABEL, SCROLLBAR_X + BUTTON_OFFSET_X, SCROLLBAR_Y-5, CharToString(0x35), up_color, 15, "Webdings");
         int down_y = SCROLLBAR_Y + SCROLLBAR_HEIGHT - BUTTON_SIZE;
         createRecLabel(SCROLL_DOWN_REC, SCROLLBAR_X + BUTTON_OFFSET_X, down_y, BUTTON_WIDTH, BUTTON_SIZE, clrDarkGray, 1, clrDarkGray);
         createLabel(SCROLL_DOWN_LABEL, SCROLLBAR_X + BUTTON_OFFSET_X, down_y-5, CharToString(0x36), down_color, 15, "Webdings");
         slider_height = calculateSliderHeight();
         int slider_y = SCROLLBAR_Y + BUTTON_SIZE;
         createButton(SCROLL_SLIDER, SCROLLBAR_X + SLIDER_OFFSET_X, slider_y, SLIDER_WIDTH, slider_height, "", clrWhite, 12, clrLightSlateGray, clrDarkGray, "Arial Bold");
         ObjectSetInteger(0, SCROLL_SLIDER, OBJPROP_WIDTH, 2);
         if (debugLogging) Print("Scrollbar created: totalEvents_Filtered=", totalEvents_Filtered, ", slider_height=", slider_height);
      }
      updateSliderPosition();
      updateButtonColors();
   } else {
      ObjectDelete(0, SCROLL_LEADER);
      ObjectDelete(0, SCROLL_UP_REC);
      ObjectDelete(0, SCROLL_UP_LABEL);
      ObjectDelete(0, SCROLL_DOWN_REC);
      ObjectDelete(0, SCROLL_DOWN_LABEL);
      ObjectDelete(0, SCROLL_SLIDER);
      if (debugLogging) Print("Scrollbar removed: totalEvents_Filtered=", totalEvents_Filtered);
   }
}
```

We advance the user interface by implementing key updates to the dashboard’s time display and dynamic scrollbar, ensuring we can seamlessly interact with and navigate through news events. We update the "TIME\_LABEL" to reflect real-time server time and event statistics, providing clear feedback on the dashboard’s state. Additionally, we manage the scrollbar’s visibility and initialize its components with dynamic icon colors, creating an intuitive navigation system that enhances our experience.

First, we focus on updating the "TIME\_LABEL" to keep us informed about the current time and event processing status. We create a "timeText" string using a conditional expression: if "updateServerTime" is true, we call the [TimeToString](https://www.mql5.com/en/docs/convert/timetostring) function with [TimeCurrent](https://www.mql5.com/en/docs/dateandtime/timecurrent) and the "TIME\_DATE\|TIME\_SECONDS" flags to format the server time; otherwise, we set it to "Server Time: Static". We then use the "updateLabel" function to set "TIME\_LABEL"’s text to "timeText", concatenated with a separator ("   \|\|\|   ") and the event counts formatted via [IntegerToString](https://www.mql5.com/en/docs/convert/integertostring) for "totalEvents\_Filtered" and "totalEvents\_Considered", resulting in a display like "Server Time: 2025.03.01 12:00:00   \|\|\|   Total News: 1711/3000", clearly showing filtered versus total events, eliminating the displayable news as we had with the prior versions.

Then, we implement the scrollbar’s visibility logic and component creation to ensure it appears only when needed and provides visual feedback. We determine "new\_scroll\_visible" by checking if "totalEvents\_Filtered" exceeds "VISIBLE\_ITEMS" (11), indicating more events than can be displayed at once. If "new\_scroll\_visible" differs from "scroll\_visible", or if "events\_changed" or "filters\_changed" is true, we update "scroll\_visible" and log the state with [Print](https://www.mql5.com/en/docs/common/print) if "debugLogging" is enabled. When "scroll\_visible" is true, we use [ObjectFind](https://www.mql5.com/en/docs/objects/ObjectFind) to check if "SCROLL\_LEADER" exists, and if not, we create the scrollbar: we call "createRecLabel" for "SCROLL\_LEADER", "SCROLL\_UP\_REC", and "SCROLL\_DOWN\_REC" at positions defined by "SCROLLBAR\_X", "SCROLLBAR\_Y", "BUTTON\_OFFSET\_X", and "BUTTON\_SIZE", using "clrSilver" and "clrDarkGray".

We compute "max\_scroll" with [MathMax](https://www.mql5.com/en/docs/math/mathmax) and " [ArraySize](https://www.mql5.com/en/docs/array/arraysize)(displayableEvents)" minus "VISIBLE\_ITEMS", set "up\_color" and "down\_color" to "clrLightGray" or "clrBlack" based on "scroll\_pos" and "max\_scroll", and create "SCROLL\_UP\_LABEL" and "SCROLL\_DOWN\_LABEL" with "createLabel", using [CharToString](https://www.mql5.com/en/docs/convert/chartostring) for Web dings arrows, specifically 5 and 6 as shown below.

![WEB FONTS](https://c.mql5.com/2/142/Screenshot_2025-05-15_005124.png)

In case you are wondering why we used "0x35" instead of just 5, that is the hexadecimal binary format in base-16. You will achieve the same results if you used 5, but as a string like "5". If you want to use the [ASCII character code](https://www.mql5.com/go?link=https://www.ascii-code.com/53 "https://www.ascii-code.com/53"), you could still convert the 53 direct into a string like [CharToString(53)](https://www.mql5.com/en/docs/convert/chartostring) and you get the same results. It is a wide range of options you could use, so, your choice. Here is the code's information.

![HEXADECIMAL "0x35" AS 5](https://c.mql5.com/2/142/Screenshot_2025-05-15_010044.png)

We calculate "slider\_height" via "calculateSliderHeight", create "SCROLL\_SLIDER" with "createButton" at "slider\_y", set its "OBJPROP\_WIDTH" to 2, and log details if "debugLogging" is true. We call "updateSliderPosition" and "updateButtonColors" to initialize the slider and icon colors. If "scroll\_visible" is false, we remove scrollbar objects using [ObjectDelete](https://www.mql5.com/en/docs/objects/objectdelete) for "SCROLL\_LEADER", "SCROLL\_UP\_REC", "SCROLL\_DOWN\_REC", "SCROLL\_UP\_LABEL", "SCROLL\_DOWN\_LABEL", and "SCROLL\_SLIDER", logging the removal to maintain a clean interface when scrolling isn’t required. The rest of the function housing this logic, as well as the one responsible for storing the events, is as below.

```
//+------------------------------------------------------------------+
//| Update dashboard values                                          |
//+------------------------------------------------------------------+
void update_dashboard_values(string &curr_filter_array[], ENUM_CALENDAR_EVENT_IMPORTANCE &imp_filter_array[]) {
   totalEvents_Considered = 0;
   totalEvents_Filtered = 0;
   totalEvents_Displayable = 0;
   ArrayFree(current_eventNames_data);
   ArrayFree(current_displayable_eventNames);

   datetime timeRange = PeriodSeconds(range_time);
   datetime timeBefore = TimeTradeServer() - timeRange;
   datetime timeAfter = TimeTradeServer() + timeRange;

   // Populate displayableEvents
   if (MQLInfoInteger(MQL_TESTER)) {
      if (filters_changed) {
         FilterEventsForTester();
         ArrayFree(displayableEvents); // Clear displayableEvents on filter change
      }
      int eventIndex = 0;
      for (int i = 0; i < ArraySize(filteredEvents); i++) {
         totalEvents_Considered++;
         datetime eventDateTime = filteredEvents[i].eventDateTime;
         if (eventDateTime < StartDate || eventDateTime > EndDate) {
            if (debugLogging) Print("Event ", filteredEvents[i].event, " skipped due to date range.");
            continue;
         }

         bool timeMatch = !enableTimeFilter;
         if (enableTimeFilter) {
            if (eventDateTime <= TimeTradeServer() && eventDateTime >= timeBefore) timeMatch = true;
            else if (eventDateTime >= TimeTradeServer() && eventDateTime <= timeAfter) timeMatch = true;
         }
         if (!timeMatch) {
            if (debugLogging) Print("Event ", filteredEvents[i].event, " skipped due to time filter.");
            continue;
         }

         bool currencyMatch = !enableCurrencyFilter;
         if (enableCurrencyFilter) {
            for (int j = 0; j < ArraySize(curr_filter_array); j++) {
               if (filteredEvents[i].currency == curr_filter_array[j]) {
                  currencyMatch = true;
                  break;
               }
            }
         }
         if (!currencyMatch) {
            if (debugLogging) Print("Event ", filteredEvents[i].event, " skipped due to currency filter.");
            continue;
         }

         bool importanceMatch = !enableImportanceFilter;
         if (enableImportanceFilter) {
            string imp_str = filteredEvents[i].importance;
            ENUM_CALENDAR_EVENT_IMPORTANCE event_imp = (imp_str == "None") ? CALENDAR_IMPORTANCE_NONE :
                                                      (imp_str == "Low") ? CALENDAR_IMPORTANCE_LOW :
                                                      (imp_str == "Medium") ? CALENDAR_IMPORTANCE_MODERATE :
                                                      CALENDAR_IMPORTANCE_HIGH;
            for (int k = 0; k < ArraySize(imp_filter_array); k++) {
               if (event_imp == imp_filter_array[k]) {
                  importanceMatch = true;
                  break;
               }
            }
         }
         if (!importanceMatch) {
            if (debugLogging) Print("Event ", filteredEvents[i].event, " skipped due to importance filter.");
            continue;
         }

         ArrayResize(displayableEvents, eventIndex + 1);
         displayableEvents[eventIndex] = filteredEvents[i];
         ArrayResize(current_displayable_eventNames, eventIndex + 1);
         current_displayable_eventNames[eventIndex] = filteredEvents[i].event;
         eventIndex++;
      }
      totalEvents_Filtered = ArraySize(displayableEvents);
      if (debugLogging) Print("Tester mode: Stored ", totalEvents_Filtered, " displayable events.");
   } else {
      MqlCalendarValue values[];
      datetime startTime = TimeTradeServer() - PeriodSeconds(start_time);
      datetime endTime = TimeTradeServer() + PeriodSeconds(end_time);
      int allValues = CalendarValueHistory(values,startTime,endTime,NULL,NULL);
      int eventIndex = 0;
      if (filters_changed) ArrayFree(displayableEvents); // Clear displayableEvents on filter change
      for (int i = 0; i < allValues; i++) {
         MqlCalendarEvent event;
         CalendarEventById(values[i].event_id, event);
         MqlCalendarCountry country;
         CalendarCountryById(event.country_id, country);
         MqlCalendarValue value;
         CalendarValueById(values[i].id, value);
         totalEvents_Considered++;

         bool currencyMatch = false;
         if (enableCurrencyFilter) {
            for (int j = 0; j < ArraySize(curr_filter_array); j++) {
               if (country.currency == curr_filter_array[j]) {
                  currencyMatch = true;
                  break;
               }
            }
            if (!currencyMatch) continue;
         }

         bool importanceMatch = false;
         if (enableImportanceFilter) {
            for (int k = 0; k < ArraySize(imp_filter_array); k++) {
               if (event.importance == imp_filter_array[k]) {
                  importanceMatch = true;
                  break;
               }
            }
            if (!importanceMatch) continue;
         }

         bool timeMatch = false;
         if (enableTimeFilter) {
            datetime eventTime = values[i].time;
            if (eventTime <= TimeTradeServer() && eventTime >= timeBefore) timeMatch = true;
            else if (eventTime >= TimeTradeServer() && eventTime <= timeAfter) timeMatch = true;
            if (!timeMatch) continue;
         }

         ArrayResize(displayableEvents, eventIndex + 1);
         displayableEvents[eventIndex].eventDate = TimeToString(values[i].time,TIME_DATE);
         displayableEvents[eventIndex].eventTime = TimeToString(values[i].time,TIME_MINUTES);
         displayableEvents[eventIndex].currency = country.currency;
         displayableEvents[eventIndex].event = event.name;
         displayableEvents[eventIndex].importance = (event.importance == CALENDAR_IMPORTANCE_NONE) ? "None" :
                                                   (event.importance == CALENDAR_IMPORTANCE_LOW) ? "Low" :
                                                   (event.importance == CALENDAR_IMPORTANCE_MODERATE) ? "Medium" : "High";
         displayableEvents[eventIndex].actual = value.GetActualValue();
         displayableEvents[eventIndex].forecast = value.GetForecastValue();
         displayableEvents[eventIndex].previous = value.GetPreviousValue();
         displayableEvents[eventIndex].eventDateTime = values[i].time;
         ArrayResize(current_displayable_eventNames, eventIndex + 1);
         current_displayable_eventNames[eventIndex] = event.name;
         eventIndex++;
      }
      totalEvents_Filtered = ArraySize(displayableEvents);
      if (debugLogging) Print("Live mode: Stored ", totalEvents_Filtered, " displayable events.");
   }

   // Check for changes in displayable events
   bool events_changed = isChangeInStringArrays(previous_displayable_eventNames, current_displayable_eventNames);
   bool scroll_changed = (scroll_pos != prev_scroll_pos);
   if (events_changed || filters_changed || scroll_changed) {
      if (debugLogging) {
         if (events_changed) Print("Changes detected in displayable events.");
         if (filters_changed) Print("Filter changes detected.");
         if (scroll_changed) Print("Scroll position changed: ", prev_scroll_pos, " -> ", scroll_pos);
      }
      ArrayFree(previous_displayable_eventNames);
      ArrayCopy(previous_displayable_eventNames, current_displayable_eventNames);
      prev_scroll_pos = scroll_pos;

      // Clear and redraw UI
      ObjectsDeleteAll(0, DATA_HOLDERS);
      ObjectsDeleteAll(0, ARRAY_NEWS);

      int startY = LIST_Y;
      int start_idx = scroll_visible ? scroll_pos : 0;
      int end_idx = MathMin(start_idx + VISIBLE_ITEMS, ArraySize(displayableEvents));
      for (int i = start_idx; i < end_idx; i++) {
         totalEvents_Displayable++;
         color holder_color = (totalEvents_Displayable % 2 == 0) ? C'213,227,207' : clrWhite;
         createRecLabel(DATA_HOLDERS+string(totalEvents_Displayable),LIST_X,startY-1,LIST_WIDTH,ITEM_HEIGHT+1,holder_color,1,clrNONE);

         int startX = LIST_X + 3;
         string news_data[ArraySize(array_calendar)];
         news_data[0] = displayableEvents[i].eventDate;
         news_data[1] = displayableEvents[i].eventTime;
         news_data[2] = displayableEvents[i].currency;
         color importance_color = clrBlack;
         if (displayableEvents[i].importance == "Low") importance_color = clrYellow;
         else if (displayableEvents[i].importance == "Medium") importance_color = clrOrange;
         else if (displayableEvents[i].importance == "High") importance_color = clrRed;
         news_data[3] = ShortToString(0x25CF);
         news_data[4] = displayableEvents[i].event;
         news_data[5] = DoubleToString(displayableEvents[i].actual, 3);
         news_data[6] = DoubleToString(displayableEvents[i].forecast, 3);
         news_data[7] = DoubleToString(displayableEvents[i].previous, 3);

         for (int k = 0; k < ArraySize(array_calendar); k++) {
            if (k == 3) {
               createLabel(ARRAY_NEWS+IntegerToString(i)+" "+array_calendar[k],startX,startY-(22-12),news_data[k],importance_color,22,"Calibri");
            } else {
               createLabel(ARRAY_NEWS+IntegerToString(i)+" "+array_calendar[k],startX,startY,news_data[k],clrBlack,12,"Calibri");
            }
            startX += buttons[k]+3;
         }

         ArrayResize(current_eventNames_data, ArraySize(current_eventNames_data)+1);
         current_eventNames_data[ArraySize(current_eventNames_data)-1] = displayableEvents[i].event;
         startY += ITEM_HEIGHT;
      }

      if (debugLogging) Print("Displayed ", totalEvents_Displayable, " events, start_idx=", start_idx, ", end_idx=", end_idx);
   } else {
      if (debugLogging) Print("No changes detected. Skipping redraw.");
   }

   // Update TIME_LABEL
   string timeText = updateServerTime ? "Server Time: "+TimeToString(TimeCurrent(),TIME_DATE|TIME_SECONDS) : "Server Time: Static";
   updateLabel(TIME_LABEL,timeText+"   |||   Total News: "+IntegerToString(totalEvents_Filtered)+"/"+IntegerToString(totalEvents_Considered));

   // Update scrollbar visibility
   bool new_scroll_visible = totalEvents_Filtered > VISIBLE_ITEMS;
   if (new_scroll_visible != scroll_visible || events_changed || filters_changed) {
      scroll_visible = new_scroll_visible;
      if (debugLogging) Print("Scrollbar visibility: ", scroll_visible ? "Visible" : "Hidden");
      if (scroll_visible) {
         if (ObjectFind(0, SCROLL_LEADER) < 0) {
            createRecLabel(SCROLL_LEADER, SCROLLBAR_X, SCROLLBAR_Y, SCROLLBAR_WIDTH, SCROLLBAR_HEIGHT, clrSilver, 1, clrNONE);
            int max_scroll = MathMax(0, ArraySize(displayableEvents) - VISIBLE_ITEMS);
            color up_color = (scroll_pos == 0) ? clrLightGray : clrBlack;
            color down_color = (scroll_pos >= max_scroll) ? clrLightGray : clrBlack;
            createRecLabel(SCROLL_UP_REC, SCROLLBAR_X + BUTTON_OFFSET_X, SCROLLBAR_Y, BUTTON_WIDTH, BUTTON_SIZE, clrDarkGray, 1, clrDarkGray);
            createLabel(SCROLL_UP_LABEL, SCROLLBAR_X + BUTTON_OFFSET_X, SCROLLBAR_Y-5, CharToString(0x35), up_color, 15, "Webdings");
            int down_y = SCROLLBAR_Y + SCROLLBAR_HEIGHT - BUTTON_SIZE;
            createRecLabel(SCROLL_DOWN_REC, SCROLLBAR_X + BUTTON_OFFSET_X, down_y, BUTTON_WIDTH, BUTTON_SIZE, clrDarkGray, 1, clrDarkGray);
            createLabel(SCROLL_DOWN_LABEL, SCROLLBAR_X + BUTTON_OFFSET_X, down_y-5, CharToString(0x36), down_color, 15, "Webdings");
            slider_height = calculateSliderHeight();
            int slider_y = SCROLLBAR_Y + BUTTON_SIZE;
            createButton(SCROLL_SLIDER, SCROLLBAR_X + SLIDER_OFFSET_X, slider_y, SLIDER_WIDTH, slider_height, "", clrWhite, 12, clrLightSlateGray, clrDarkGray, "Arial Bold");
            ObjectSetInteger(0, SCROLL_SLIDER, OBJPROP_WIDTH, 2);
            if (debugLogging) Print("Scrollbar created: totalEvents_Filtered=", totalEvents_Filtered, ", slider_height=", slider_height);
         }
         updateSliderPosition();
         updateButtonColors();
      } else {
         ObjectDelete(0, SCROLL_LEADER);
         ObjectDelete(0, SCROLL_UP_REC);
         ObjectDelete(0, SCROLL_UP_LABEL);
         ObjectDelete(0, SCROLL_DOWN_REC);
         ObjectDelete(0, SCROLL_DOWN_LABEL);
         ObjectDelete(0, SCROLL_SLIDER);
         if (debugLogging) Print("Scrollbar removed: totalEvents_Filtered=", totalEvents_Filtered);
      }
   }

   if (isChangeInStringArrays(previous_eventNames_data, current_eventNames_data)) {
      if (debugLogging) Print("CHANGES IN EVENT NAMES DETECTED.");
      ArrayFree(previous_eventNames_data);
      ArrayCopy(previous_eventNames_data, current_eventNames_data);
   }
}
```

Here, we refine the dashboard in the "update\_dashboard\_values" function, introducing new logic to ensure a polished event display and dynamic scrollbar functionality while prioritizing silent updates for efficiency. We leverage the "displayableEvents" array to store all filtered events, implement change detection to avoid unnecessary redraws, and integrate the scrollbar for intuitive navigation, addressing previous limitations and enhancing trader interaction. We begin by resetting counters "totalEvents\_Considered", "totalEvents\_Filtered", and "totalEvents\_Displayable" to zero and clearing arrays "current\_eventNames\_data" and "current\_displayable\_eventNames" using [ArrayFree](https://www.mql5.com/en/docs/array/ArrayFree) to prepare for updated event data.

While preserving the existing filtering logic for tester and live modes, we introduce the "displayableEvents" array to store all filtered events, ensuring comprehensive access (e.g., 1711 events), unlike the original’s 5-6 event display issue. In tester mode, we call "FilterEventsForTester" if "filters\_changed" is true, clear "displayableEvents" with "ArrayFree", and [loop](https://www.mql5.com/en/docs/basis/operators/for) through "filteredEvents", applying filters ("enableTimeFilter", "enableCurrencyFilter", "enableImportanceFilter") to populate "displayableEvents" and "current\_displayable\_eventNames", setting "totalEvents\_Filtered" to " [ArraySize](https://www.mql5.com/en/docs/array/arraysize)(displayableEvents)".

In live mode, we use [CalendarValueHistory](https://www.mql5.com/en/docs/calendar/calendarvaluehistory) to fetch events, clear "displayableEvents" if "filters\_changed", and store filtered events similarly, logging the count if "debugLogging" is enabled.

We implement silent updates by using "isChangeInStringArrays" to compare "previous\_displayable\_eventNames" with "current\_displayable\_eventNames", setting "events\_changed", and checking if "scroll\_pos" differs from "prev\_scroll\_pos" for "scroll\_changed". If "events\_changed", "filters\_changed", or "scroll\_changed" is true, we log changes via "Print" if "debugLogging" is enabled, update "previous\_displayable\_eventNames" with [ArrayCopy](https://www.mql5.com/en/docs/array/arraycopy), and set "prev\_scroll\_pos". We clear UI elements using [ObjectsDeleteAll](https://www.mql5.com/en/docs/objects/ObjectDeleteAll) for "DATA\_HOLDERS" and "ARRAY\_NEWS", then draw up to "VISIBLE\_ITEMS" (11) events from "displayableEvents", starting at "start\_idx" (based on "scroll\_pos" if "scroll\_visible") to "end\_idx" (capped by "ArraySize(displayableEvents)").

For each event, we call "createRecLabel" for a background with alternating colors ("C'213,227,207'" or "clrWhite"), populate "news\_data" with event details, and use "createLabel" to display fields, adjusting "importance\_color" (e.g., "clrYellow" for Low) and logging display counts if "debugLogging" is true. If no changes occur, we skip redrawing and log this, maintaining performance.

We integrate the scrollbar by setting "new\_scroll\_visible" if "totalEvents\_Filtered" exceeds "VISIBLE\_ITEMS", updating "scroll\_visible" if it changes or if "events\_changed" or "filters\_changed" is true, and creating components ("SCROLL\_LEADER", "SCROLL\_UP\_REC", "SCROLL\_UP\_LABEL", "SCROLL\_DOWN\_REC", "SCROLL\_DOWN\_LABEL", "SCROLL\_SLIDER") with "createRecLabel", "createLabel", and "createButton", using "clrBlack" or "clrLightGray" for icons based on "scroll\_pos" and "max\_scroll". We remove components with [ObjectDelete](https://www.mql5.com/en/docs/objects/objectdelete) when not needed, and update "previous\_eventNames\_data" if "isChangeInStringArrays" detects changes in "current\_eventNames\_data", ensuring a robust, navigable, and efficient dashboard. Since we have created new objects, we need to clear them as part of the main dashboard.

```
//+------------------------------------------------------------------+
//| Destroy dashboard                                                |
//+------------------------------------------------------------------+
void destroy_Dashboard() {
   ObjectDelete(0,"MAIN_REC");
   ObjectDelete(0,"SUB_REC1");
   ObjectDelete(0,"SUB_REC2");
   ObjectDelete(0,"HEADER_LABEL");
   ObjectDelete(0,"TIME_LABEL");
   ObjectDelete(0,"IMPACT_LABEL");
   ObjectsDeleteAll(0,"ARRAY_CALENDAR");
   ObjectsDeleteAll(0,"ARRAY_NEWS");
   ObjectsDeleteAll(0,"DATA_HOLDERS");
   ObjectsDeleteAll(0,"IMPACT_LABEL");
   ObjectDelete(0,"FILTER_LABEL");
   ObjectDelete(0,"FILTER_CURR_BTN");
   ObjectDelete(0,"FILTER_IMP_BTN");
   ObjectDelete(0,"FILTER_TIME_BTN");
   ObjectDelete(0,"CANCEL_BTN");
   ObjectsDeleteAll(0,"CURRENCY_BTNS");
   ObjectDelete(0, SCROLL_LEADER);
   ObjectDelete(0, SCROLL_UP_REC);
   ObjectDelete(0, SCROLL_UP_LABEL);
   ObjectDelete(0, SCROLL_DOWN_REC);
   ObjectDelete(0, SCROLL_DOWN_LABEL);
   ObjectDelete(0, SCROLL_SLIDER);
   ArrayFree(displayableEvents);
   ArrayFree(current_displayable_eventNames);
   ArrayFree(previous_displayable_eventNames);
   ChartRedraw(0);
}
```

To clear the new objects created, we just call the [ObjectDelete](https://www.mql5.com/en/docs/objects/objectdelete) function and pass in their names, respectively, to ensure they are deleted when we delete the dashboard, as they are now part of the dashboard. Upon compilation, we have the following outcome.

Less than or equal to 11 filtered events.

![FITTING EVENTS](https://c.mql5.com/2/142/Screenshot_2025-05-14_194232.png)

More than 11 filtered events.

![NON-FITTING EVENTS](https://c.mql5.com/2/142/Screenshot_2025-05-14_194250.png)

Extensively filtered events range.

![EXTENSIVE EVENTS RANGE](https://c.mql5.com/2/142/Screenshot_2025-05-14_195609.png)

From the images above, we can see that we create the calendar scrollbar dynamically based on available events. What we now need to do is add life to the scrollbar elements, and we can achieve that by using the [OnChartEvent](https://www.mql5.com/en/docs/event_handlers/onchartevent) function as follows.

```
//+------------------------------------------------------------------+
//| Chart event handler                                              |
//+------------------------------------------------------------------+
void OnChartEvent(const int id, const long& lparam, const double& dparam, const string& sparam) {
   int mouse_x = (int)lparam;
   int mouse_y = (int)dparam;
   int mouse_state = (int)sparam;

   if (id == CHARTEVENT_OBJECT_CLICK) {

      // Scrollbar button clicks
      if (scroll_visible && (sparam == SCROLL_UP_REC || sparam == SCROLL_UP_LABEL)) {
         scrollUp();
         updateButtonColors();
         if (debugLogging) Print("Up button clicked (", sparam, "). CurrPos: ", scroll_pos);
         ChartRedraw(0);
      }
      if (scroll_visible && (sparam == SCROLL_DOWN_REC || sparam == SCROLL_DOWN_LABEL)) {
         scrollDown();
         updateButtonColors();
         if (debugLogging) Print("Down button clicked (", sparam, "). CurrPos: ", scroll_pos);
         ChartRedraw(0);
      }
   }
   else if (id == CHARTEVENT_MOUSE_MOVE && scroll_visible) {
      if (prev_mouse_state == 0 && mouse_state == 1) {
         int xd = (int)ObjectGetInteger(0, SCROLL_SLIDER, OBJPROP_XDISTANCE);
         int yd = (int)ObjectGetInteger(0, SCROLL_SLIDER, OBJPROP_YDISTANCE);
         int xs = (int)ObjectGetInteger(0, SCROLL_SLIDER, OBJPROP_XSIZE);
         int ys = (int)ObjectGetInteger(0, SCROLL_SLIDER, OBJPROP_YSIZE);
         if (mouse_x >= xd && mouse_x <= xd + xs && mouse_y >= yd && mouse_y <= yd + ys) {
            moving_state_slider = true;
            mlb_down_x = mouse_x;
            mlb_down_y = mouse_y;
            mlb_down_yd_slider = yd;
            ObjectSetInteger(0, SCROLL_SLIDER, OBJPROP_BGCOLOR, clrDodgerBlue);
            ObjectSetInteger(0, SCROLL_SLIDER, OBJPROP_YSIZE, slider_height + 2);
            ChartSetInteger(0, CHART_MOUSE_SCROLL, false);
            if (debugLogging) Print("Slider drag started at y=", mouse_y);
         }
      }
      if (moving_state_slider && mouse_state == 1) {
         int delta_y = mouse_y - mlb_down_y;
         int new_y = mlb_down_yd_slider + delta_y;
         int scroll_area_y_min = SCROLLBAR_Y + BUTTON_SIZE;
         int scroll_area_y_max = scroll_area_y_min + SCROLL_AREA_HEIGHT - slider_height;
         new_y = MathMax(scroll_area_y_min, MathMin(new_y, scroll_area_y_max));
         ObjectSetInteger(0, SCROLL_SLIDER, OBJPROP_YDISTANCE, new_y);
         int max_scroll = MathMax(0, ArraySize(displayableEvents) - VISIBLE_ITEMS);
         double scroll_ratio = (double)(new_y - scroll_area_y_min) / (scroll_area_y_max - scroll_area_y_min);
         int new_scroll_pos = (int)MathRound(scroll_ratio * max_scroll);
         if (new_scroll_pos != scroll_pos) {
            scroll_pos = new_scroll_pos;
            update_dashboard_values(curr_filter_selected, imp_filter_selected);
            updateButtonColors();
            if (debugLogging) Print("Slider dragged. CurrPos: ", scroll_pos, ", Total steps: ", max_scroll, ", Slider y=", new_y);
         }
         ChartRedraw(0);
      }
      if (mouse_state == 0) {
         if (moving_state_slider) {
            moving_state_slider = false;
            ObjectSetInteger(0, SCROLL_SLIDER, OBJPROP_BGCOLOR, clrLightSlateGray);
            ObjectSetInteger(0, SCROLL_SLIDER, OBJPROP_YSIZE, slider_height);
            ChartSetInteger(0, CHART_MOUSE_SCROLL, true);
            if (debugLogging) Print("Slider drag stopped.");
            ChartRedraw(0);
         }
      }
      prev_mouse_state = mouse_state;
   }
}
```

Here, we enhance the interactivity by implementing the new scrollbar-related logic in the [OnChartEvent](https://www.mql5.com/en/docs/event_handlers/onchartevent) function, enabling us to navigate news events seamlessly through clicks and drags. We focus on handling user interactions with the dynamic scrollbar, specifically processing clicks on the up and down buttons and mouse movements for dragging the slider, ensuring responsive updates to the dashboard’s display. We process [CHARTEVENT\_OBJECT\_CLICK](https://www.mql5.com/en/docs/constants/chartconstants/enum_chartevents) events to handle clicks on the scrollbar’s buttons when "scroll\_visible" is true. If the clicked object ("sparam") is "SCROLL\_UP\_REC" or "SCROLL\_UP\_LABEL", we call the "scrollUp" function to decrement "scroll\_pos", followed by "updateButtonColors" to refresh the icon colors ("clrBlack" or "clrLightGray") based on the new position, log the action with "Print" if "debugLogging" is enabled, and call [ChartRedraw](https://www.mql5.com/en/docs/chart_operations/ChartRedraw) to update the display.

Similarly, for clicks on "SCROLL\_DOWN\_REC" or "SCROLL\_DOWN\_LABEL", we invoke "scrollDown" to increment "scroll\_pos", call "updateButtonColors", log the event, and refresh the chart with "ChartRedraw", ensuring the dashboard reflects the scrolled event list. We have defined the functions and will explain the logic later.

For [CHARTEVENT\_MOUSE\_MOVE](https://www.mql5.com/en/docs/constants/chartconstants/enum_chartevents) events when "scroll\_visible" is true, we manage slider dragging. When "prev\_mouse\_state" is 0 and "mouse\_state" is 1, we use [ObjectGetInteger](https://www.mql5.com/en/docs/objects/objectgetinteger) to retrieve "SCROLL\_SLIDER"’s position ("OBJPROP\_XDISTANCE", "OBJPROP\_YDISTANCE") and size ("OBJPROP\_XSIZE", "OBJPROP\_YSIZE") as "xd", "yd", "xs", and "ys". If the mouse coordinates ("mouse\_x", "mouse\_y") lie within the slider’s bounds, we set "moving\_state\_slider" to true, store "mlb\_down\_x", "mlb\_down\_y", and "mlb\_down\_yd\_slider", change "SCROLL\_SLIDER"’s "OBJPROP\_BGCOLOR" to "clrDodgerBlue" and increase "OBJPROP\_YSIZE" by 2, disable chart scrolling with [ChartSetInteger](https://www.mql5.com/en/docs/chart_operations/chartsetinteger), and log the drag start if "debugLogging" is enabled.

While "moving\_state\_slider" and "mouse\_state" are true, we calculate "delta\_y" as "mouse\_y" minus "mlb\_down\_y", compute "new\_y" within bounds "scroll\_area\_y\_min" ("SCROLLBAR\_Y" + "BUTTON\_SIZE") and "scroll\_area\_y\_max" ("scroll\_area\_y\_min" + "SCROLL\_AREA\_HEIGHT" - "slider\_height") using [MathMax](https://www.mql5.com/en/docs/math/mathmax) and [MathMin](https://www.mql5.com/en/docs/math/mathmin), and set "SCROLL\_SLIDER"’s "OBJPROP\_YDISTANCE" to "new\_y". We derive "new\_scroll\_pos" from the "scroll\_ratio" of "new\_y" within the scroll range, and if it differs from "scroll\_pos", we update "scroll\_pos", call "update\_dashboard\_values" with "curr\_filter\_selected" and "imp\_filter\_selected", invoke "updateButtonColors", log the drag details, and call [ChartRedraw](https://www.mql5.com/en/docs/chart_operations/ChartRedraw).

When "mouse\_state" is 0, we reset "moving\_state\_slider", restore "SCROLL\_SLIDER"’s "OBJPROP\_BGCOLOR" to "clrLightSlateGray" and "OBJPROP\_YSIZE" to "slider\_height", re-enable chart scrolling, log the drag stop, and call "ChartRedraw", ensuring smooth slider interaction. The functions responsible for the scroll logic are as follows.

```
//+------------------------------------------------------------------+
//| Scroll up                                                        |
//+------------------------------------------------------------------+
void scrollUp() {
   if (scroll_pos > 0) {
      scroll_pos--;
      update_dashboard_values(curr_filter_selected, imp_filter_selected);
      updateSliderPosition();
      if (debugLogging) Print("Scrolled up. CurrPos: ", scroll_pos);
   } else {
      if (debugLogging) Print("Cannot scroll up further. Already at top.");
   }
}

//+------------------------------------------------------------------+
//| Scroll down                                                      |
//+------------------------------------------------------------------+
void scrollDown() {
   int max_scroll = MathMax(0, ArraySize(displayableEvents) - VISIBLE_ITEMS);
   if (scroll_pos < max_scroll) {
      scroll_pos++;
      update_dashboard_values(curr_filter_selected, imp_filter_selected);
      updateSliderPosition();
      if (debugLogging) Print("Scrolled down. CurrPos: ", scroll_pos);
   } else {
      if (debugLogging) Print("Cannot scroll down further. Max scroll reached: ", max_scroll);
   }
}
//+------------------------------------------------------------------+
```

Here, in the "scrollUp" function, we facilitate upward navigation through the event list. We check if "scroll\_pos" is greater than 0, indicating that scrolling up is possible. If true, we decrement "scroll\_pos", call "update\_dashboard\_values" with "curr\_filter\_selected" and "imp\_filter\_selected" to refresh the displayed events, invoke "updateSliderPosition" to adjust the "SCROLL\_SLIDER"’s position, and log the new "scroll\_pos" using [Print](https://www.mql5.com/en/docs/common/print) if "debugLogging" is enabled. If "scroll\_pos" is 0, we log a message indicating that the top of the list has been reached, preventing unnecessary updates and maintaining a clean interaction flow.

In the "scrollDown" function, we enable downward navigation through the event list. We calculate "max\_scroll" using [MathMax](https://www.mql5.com/en/docs/math/mathmax) to ensure a non-negative value, derived from " [ArraySize](https://www.mql5.com/en/docs/array/arraysize)(displayableEvents)" minus "VISIBLE\_ITEMS" (11), representing the maximum scrollable position.

If "scroll\_pos" is less than "max\_scroll", we increment "scroll\_pos", call "update\_dashboard\_values" with "curr\_filter\_selected" and "imp\_filter\_selected" to update the displayed events, use "updateSliderPosition" to reposition the "SCROLL\_SLIDER", and log the new "scroll\_pos" with "Print" if "debugLogging" is enabled. If "scroll\_pos" equals or exceeds "max\_scroll", we log a message indicating that the bottom of the list has been reached, avoiding redundant updates and ensuring intuitive navigation. To ensure seamless interaction, we call the functions defined in both the [OnInit](https://www.mql5.com/en/docs/event_handlers/oninit) and [OnTick](https://www.mql5.com/en/docs/event_handlers/ontick) event handlers.

```
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick() {
   UpdateFilterInfo();
   CheckForNewsTrade();
   if (isDashboardUpdate) {
      if (MQLInfoInteger(MQL_TESTER)) {
         datetime currentTime = TimeTradeServer();
         datetime timeRange = PeriodSeconds(range_time);
         datetime timeAfter = currentTime + timeRange;
         if (filters_changed || last_dashboard_update < timeAfter) {
            update_dashboard_values(curr_filter_selected, imp_filter_selected);
            ArrayFree(last_dashboard_eventNames);
            ArrayCopy(last_dashboard_eventNames, current_eventNames_data);
            last_dashboard_update = currentTime;
         }
      } else {
         update_dashboard_values(curr_filter_selected, imp_filter_selected);
      }
   }
}

//+------------------------------------------------------------------+
//| Chart event handler                                              |
//+------------------------------------------------------------------+
void OnChartEvent(const int id, const long& lparam, const double& dparam, const string& sparam) {
   int mouse_x = (int)lparam;
   int mouse_y = (int)dparam;
   int mouse_state = (int)sparam;

   if (id == CHARTEVENT_OBJECT_CLICK) {
      UpdateFilterInfo();
      CheckForNewsTrade();
      if (sparam == CANCEL_BTN) {
         isDashboardUpdate = false;
         destroy_Dashboard();
      }
      if (sparam == FILTER_CURR_BTN) {
         bool btn_state = ObjectGetInteger(0,sparam,OBJPROP_STATE);
         enableCurrencyFilter = btn_state;
         if (debugLogging) Print(sparam+" STATE = "+(string)btn_state+", FLAG = "+(string)enableCurrencyFilter);
         string filter_curr_text = enableCurrencyFilter ? ShortToString(0x2714)+"Currency" : ShortToString(0x274C)+"Currency";
         color filter_curr_txt_color = enableCurrencyFilter ? clrLime : clrRed;
         ObjectSetString(0,FILTER_CURR_BTN,OBJPROP_TEXT,filter_curr_text);
         ObjectSetInteger(0,FILTER_CURR_BTN,OBJPROP_COLOR,filter_curr_txt_color);
         if (MQLInfoInteger(MQL_TESTER)) filters_changed = true;
         update_dashboard_values(curr_filter_selected,imp_filter_selected);
         if (debugLogging) Print("Success. Changes updated! State: "+(string)enableCurrencyFilter);
         ChartRedraw(0);
      }
      if (sparam == FILTER_IMP_BTN) {
         bool btn_state = ObjectGetInteger(0,sparam,OBJPROP_STATE);
         enableImportanceFilter = btn_state;
         if (debugLogging) Print(sparam+" STATE = "+(string)btn_state+", FLAG = "+(string)enableImportanceFilter);
         string filter_imp_text = enableImportanceFilter ? ShortToString(0x2714)+"Importance" : ShortToString(0x274C)+"Importance";
         color filter_imp_txt_color = enableImportanceFilter ? clrLime : clrRed;
         ObjectSetString(0,FILTER_IMP_BTN,OBJPROP_TEXT,filter_imp_text);
         ObjectSetInteger(0,FILTER_IMP_BTN,OBJPROP_COLOR,filter_imp_txt_color);
         if (MQLInfoInteger(MQL_TESTER)) filters_changed = true;
         update_dashboard_values(curr_filter_selected,imp_filter_selected);
         if (debugLogging) Print("Success. Changes updated! State: "+(string)enableImportanceFilter);
         ChartRedraw(0);
      }
      if (sparam == FILTER_TIME_BTN) {
         bool btn_state = ObjectGetInteger(0,sparam,OBJPROP_STATE);
         enableTimeFilter = btn_state;
         if (debugLogging) Print(sparam+" STATE = "+(string)btn_state+", FLAG = "+(string)enableTimeFilter);
         string filter_time_text = enableTimeFilter ? ShortToString(0x2714)+"Time" : ShortToString(0x274C)+"Time";
         color filter_time_txt_color = enableTimeFilter ? clrLime : clrRed;
         ObjectSetString(0,FILTER_TIME_BTN,OBJPROP_TEXT,filter_time_text);
         ObjectSetInteger(0,FILTER_TIME_BTN,OBJPROP_COLOR,filter_time_txt_color);
         if (MQLInfoInteger(MQL_TESTER)) filters_changed = true;
         update_dashboard_values(curr_filter_selected,imp_filter_selected);
         if (debugLogging) Print("Success. Changes updated! State: "+(string)enableTimeFilter);
         ChartRedraw(0);
      }
      if (StringFind(sparam,CURRENCY_BTNS) >= 0) {
         string selected_curr = ObjectGetString(0,sparam,OBJPROP_TEXT);
         if (debugLogging) Print("BTN NAME = ",sparam,", CURRENCY = ",selected_curr);
         bool btn_state = ObjectGetInteger(0,sparam,OBJPROP_STATE);
         if (btn_state == false) {
            if (debugLogging) Print("BUTTON IS IN UN-SELECTED MODE.");
            for (int i = 0; i < ArraySize(curr_filter_selected); i++) {
               if (curr_filter_selected[i] == selected_curr) {
                  for (int j = i; j < ArraySize(curr_filter_selected) - 1; j++) {
                     curr_filter_selected[j] = curr_filter_selected[j + 1];
                  }
                  ArrayResize(curr_filter_selected, ArraySize(curr_filter_selected) - 1);
                  if (debugLogging) Print("Removed from selected filters: ", selected_curr);
                  break;
               }
            }
         } else {
            if (debugLogging) Print("BUTTON IS IN SELECTED MODE. TAKE ACTION");
            bool already_selected = false;
            for (int j = 0; j < ArraySize(curr_filter_selected); j++) {
               if (curr_filter_selected[j] == selected_curr) {
                  already_selected = true;
                  break;
               }
            }
            if (!already_selected) {
               ArrayResize(curr_filter_selected, ArraySize(curr_filter_selected) + 1);
               curr_filter_selected[ArraySize(curr_filter_selected) - 1] = selected_curr;
               if (debugLogging) Print("Added to selected filters: ", selected_curr);
            } else {
               if (debugLogging) Print("Currency already selected: ", selected_curr);
            }
         }
         if (debugLogging) Print("SELECTED ARRAY SIZE = ",ArraySize(curr_filter_selected));
         if (debugLogging) ArrayPrint(curr_filter_selected);
         if (MQLInfoInteger(MQL_TESTER)) filters_changed = true;
         update_dashboard_values(curr_filter_selected,imp_filter_selected);
         if (debugLogging) Print("SUCCESS. DASHBOARD UPDATED");
         ChartRedraw(0);
      }
      if (StringFind(sparam, IMPACT_LABEL) >= 0) {
         string selected_imp = ObjectGetString(0, sparam, OBJPROP_TEXT);
         ENUM_CALENDAR_EVENT_IMPORTANCE selected_importance_lvl = get_importance_level(impact_labels,allowed_importance_levels,selected_imp);
         if (debugLogging) Print("BTN NAME = ", sparam, ", IMPORTANCE LEVEL = ", selected_imp,"(",selected_importance_lvl,")");
         bool btn_state = ObjectGetInteger(0, sparam, OBJPROP_STATE);
         color color_border = btn_state ? clrNONE : clrBlack;
         if (btn_state == false) {
            if (debugLogging) Print("BUTTON IS IN UN-SELECTED MODE.");
            for (int i = 0; i < ArraySize(imp_filter_selected); i++) {
               if (impact_filter_selected[i] == selected_imp) {
                  for (int j = i; j < ArraySize(imp_filter_selected) - 1; j++) {
                     imp_filter_selected[j] = imp_filter_selected[j + 1];
                     impact_filter_selected[j] = impact_filter_selected[j + 1];
                  }
                  ArrayResize(imp_filter_selected, ArraySize(imp_filter_selected) - 1);
                  ArrayResize(impact_filter_selected, ArraySize(impact_filter_selected) - 1);
                  if (debugLogging) Print("Removed from selected importance filters: ", selected_imp,"(",selected_importance_lvl,")");
                  break;
               }
            }
         } else {
            if (debugLogging) Print("BUTTON IS IN SELECTED MODE. TAKE ACTION");
            bool already_selected = false;
            for (int j = 0; j < ArraySize(imp_filter_selected); j++) {
               if (impact_filter_selected[j] == selected_imp) {
                  already_selected = true;
                  break;
               }
            }
            if (!already_selected) {
               ArrayResize(imp_filter_selected, ArraySize(imp_filter_selected) + 1);
               imp_filter_selected[ArraySize(imp_filter_selected) - 1] = selected_importance_lvl;
               ArrayResize(impact_filter_selected, ArraySize(impact_filter_selected) + 1);
               impact_filter_selected[ArraySize(impact_filter_selected) - 1] = selected_imp;
               if (debugLogging) Print("Added to selected importance filters: ", selected_imp,"(",selected_importance_lvl,")");
            } else {
               if (debugLogging) Print("Importance level already selected: ", selected_imp,"(",selected_importance_lvl,")");
            }
         }
         if (debugLogging) Print("SELECTED ARRAY SIZE = ", ArraySize(imp_filter_selected)," >< ",ArraySize(impact_filter_selected));
         if (debugLogging) ArrayPrint(imp_filter_selected);
         if (debugLogging) ArrayPrint(impact_filter_selected);
         if (MQLInfoInteger(MQL_TESTER)) filters_changed = true;
         update_dashboard_values(curr_filter_selected,imp_filter_selected);
         ObjectSetInteger(0,sparam,OBJPROP_BORDER_COLOR,color_border);
         if (debugLogging) Print("SUCCESS. DASHBOARD UPDATED");
         ChartRedraw(0);
      }
      // Scrollbar button clicks
      if (scroll_visible && (sparam == SCROLL_UP_REC || sparam == SCROLL_UP_LABEL)) {
         scrollUp();
         updateButtonColors();
         if (debugLogging) Print("Up button clicked (", sparam, "). CurrPos: ", scroll_pos);
         ChartRedraw(0);
      }
      if (scroll_visible && (sparam == SCROLL_DOWN_REC || sparam == SCROLL_DOWN_LABEL)) {
         scrollDown();
         updateButtonColors();
         if (debugLogging) Print("Down button clicked (", sparam, "). CurrPos: ", scroll_pos);
         ChartRedraw(0);
      }
   }
   else if (id == CHARTEVENT_MOUSE_MOVE && scroll_visible) {
      if (prev_mouse_state == 0 && mouse_state == 1) {
         int xd = (int)ObjectGetInteger(0, SCROLL_SLIDER, OBJPROP_XDISTANCE);
         int yd = (int)ObjectGetInteger(0, SCROLL_SLIDER, OBJPROP_YDISTANCE);
         int xs = (int)ObjectGetInteger(0, SCROLL_SLIDER, OBJPROP_XSIZE);
         int ys = (int)ObjectGetInteger(0, SCROLL_SLIDER, OBJPROP_YSIZE);
         if (mouse_x >= xd && mouse_x <= xd + xs && mouse_y >= yd && mouse_y <= yd + ys) {
            moving_state_slider = true;
            mlb_down_x = mouse_x;
            mlb_down_y = mouse_y;
            mlb_down_yd_slider = yd;
            ObjectSetInteger(0, SCROLL_SLIDER, OBJPROP_BGCOLOR, clrDodgerBlue);
            ObjectSetInteger(0, SCROLL_SLIDER, OBJPROP_YSIZE, slider_height + 2);
            ChartSetInteger(0, CHART_MOUSE_SCROLL, false);
            if (debugLogging) Print("Slider drag started at y=", mouse_y);
         }
      }
      if (moving_state_slider && mouse_state == 1) {
         int delta_y = mouse_y - mlb_down_y;
         int new_y = mlb_down_yd_slider + delta_y;
         int scroll_area_y_min = SCROLLBAR_Y + BUTTON_SIZE;
         int scroll_area_y_max = scroll_area_y_min + SCROLL_AREA_HEIGHT - slider_height;
         new_y = MathMax(scroll_area_y_min, MathMin(new_y, scroll_area_y_max));
         ObjectSetInteger(0, SCROLL_SLIDER, OBJPROP_YDISTANCE, new_y);
         int max_scroll = MathMax(0, ArraySize(displayableEvents) - VISIBLE_ITEMS);
         double scroll_ratio = (double)(new_y - scroll_area_y_min) / (scroll_area_y_max - scroll_area_y_min);
         int new_scroll_pos = (int)MathRound(scroll_ratio * max_scroll);
         if (new_scroll_pos != scroll_pos) {
            scroll_pos = new_scroll_pos;
            update_dashboard_values(curr_filter_selected, imp_filter_selected);
            updateButtonColors();
            if (debugLogging) Print("Slider dragged. CurrPos: ", scroll_pos, ", Total steps: ", max_scroll, ", Slider y=", new_y);
         }
         ChartRedraw(0);
      }
      if (mouse_state == 0) {
         if (moving_state_slider) {
            moving_state_slider = false;
            ObjectSetInteger(0, SCROLL_SLIDER, OBJPROP_BGCOLOR, clrLightSlateGray);
            ObjectSetInteger(0, SCROLL_SLIDER, OBJPROP_YSIZE, slider_height);
            ChartSetInteger(0, CHART_MOUSE_SCROLL, true);
            if (debugLogging) Print("Slider drag stopped.");
            ChartRedraw(0);
         }
      }
      prev_mouse_state = mouse_state;
   }
}
```

Here, we just call the functions and logic implemented on event handler functions for the changes to take effect on any necessary event handler. Upon compilation, we have the following output.

![SCROLLBAR VISUALIZATION](https://c.mql5.com/2/142/Calendar_part_9_GIF.gif)

From the visualization above, we can see that we have added a dynamic scrollbar to the dashboard. What now remains is backtesting the system thoroughly, and that is handled in the next section.

### Testing and Validation

We tested the enhanced dashboard to confirm that the dynamic scrollbar and event display function as intended, providing a seamless experience for navigating news events. Our test focused on the scrollbar’s visual feedback, the display of all filtered events, and the efficiency of silent updates, validated across both live and Strategy Tester modes. We captured these tests in a concise [Graphics Interchange Format](https://en.wikipedia.org/wiki/GIF "https://en.wikipedia.org/wiki/GIF") (GIF) to visually demonstrate the dashboard’s performance as shown below.

![TESTING 1](https://c.mql5.com/2/142/Calendar_part_9_TESTING_GIF.gif)

From the visualization, we can see that the scrollbar works correctly, but then there is an issue in that the scrollbar is not dynamic when we change the filters by clicking on them, though the events change correctly. That is due to a lack of recalculation, which disables the full dynamic feature. To achieve that, we will need to [recalibrate](https://www.mql5.com/go?link=https://dictionary.cambridge.org/dictionary/english/recalibrate "https://dictionary.cambridge.org/dictionary/english/recalibrate") the scrollbar every time we have a hit on the buttons. We could achieve the same by simply updating it when we have changes in the data, but it would then again result in redundant processes when not necessary. Here is the full logic we needed to achieve that.

```
//+------------------------------------------------------------------+
//| Chart event handler                                              |
//+------------------------------------------------------------------+
void OnChartEvent(const int id, const long& lparam, const double& dparam, const string& sparam) {
   int mouse_x = (int)lparam;
   int mouse_y = (int)dparam;
   int mouse_state = (int)sparam;

   if (id == CHARTEVENT_OBJECT_CLICK) {
      UpdateFilterInfo();
      CheckForNewsTrade();
      if (sparam == CANCEL_BTN) {
         isDashboardUpdate = false;
         destroy_Dashboard();
      }
      if (sparam == FILTER_CURR_BTN) {
         bool btn_state = ObjectGetInteger(0,sparam,OBJPROP_STATE);
         enableCurrencyFilter = btn_state;
         if (debugLogging) Print(sparam+" STATE = "+(string)btn_state+", FLAG = "+(string)enableCurrencyFilter);
         string filter_curr_text = enableCurrencyFilter ? ShortToString(0x2714)+"Currency" : ShortToString(0x274C)+"Currency";
         color filter_curr_txt_color = enableCurrencyFilter ? clrLime : clrRed;
         ObjectSetString(0,FILTER_CURR_BTN,OBJPROP_TEXT,filter_curr_text);
         ObjectSetInteger(0,FILTER_CURR_BTN,OBJPROP_COLOR,filter_curr_txt_color);
         if (MQLInfoInteger(MQL_TESTER)) filters_changed = true;
         update_dashboard_values(curr_filter_selected,imp_filter_selected);
         // Recalculate scrollbar
         ObjectDelete(0, SCROLL_LEADER);
         ObjectDelete(0, SCROLL_UP_REC);
         ObjectDelete(0, SCROLL_UP_LABEL);
         ObjectDelete(0, SCROLL_DOWN_REC);
         ObjectDelete(0, SCROLL_DOWN_LABEL);
         ObjectDelete(0, SCROLL_SLIDER);
         scroll_visible = totalEvents_Filtered > VISIBLE_ITEMS;
         if (debugLogging) Print("Scrollbar visibility: ", scroll_visible ? "Visible" : "Hidden");
         if (scroll_visible) {
            createRecLabel(SCROLL_LEADER, SCROLLBAR_X, SCROLLBAR_Y, SCROLLBAR_WIDTH, SCROLLBAR_HEIGHT, clrSilver, 1, clrNONE);
            int max_scroll = MathMax(0, ArraySize(displayableEvents) - VISIBLE_ITEMS);
            color up_color = (scroll_pos == 0) ? clrLightGray : clrBlack;
            color down_color = (scroll_pos >= max_scroll) ? clrLightGray : clrBlack;
            createRecLabel(SCROLL_UP_REC, SCROLLBAR_X + BUTTON_OFFSET_X, SCROLLBAR_Y, BUTTON_WIDTH, BUTTON_SIZE, clrDarkGray, 1, clrDarkGray);
            createLabel(SCROLL_UP_LABEL, SCROLLBAR_X + BUTTON_OFFSET_X, SCROLLBAR_Y-5, CharToString(0x35), up_color, 15, "Webdings");
            int down_y = SCROLLBAR_Y + SCROLLBAR_HEIGHT - BUTTON_SIZE;
            createRecLabel(SCROLL_DOWN_REC, SCROLLBAR_X + BUTTON_OFFSET_X, down_y, BUTTON_WIDTH, BUTTON_SIZE, clrDarkGray, 1, clrDarkGray);
            createLabel(SCROLL_DOWN_LABEL, SCROLLBAR_X + BUTTON_OFFSET_X, down_y-5, CharToString(0x36), down_color, 15, "Webdings");
            slider_height = calculateSliderHeight();
            int slider_y = SCROLLBAR_Y + BUTTON_SIZE;
            createButton(SCROLL_SLIDER, SCROLLBAR_X + SLIDER_OFFSET_X, slider_y, SLIDER_WIDTH, slider_height, "", clrWhite, 12, clrLightSlateGray, clrDarkGray, "Arial Bold");
            ObjectSetInteger(0, SCROLL_SLIDER, OBJPROP_WIDTH, 2);
            if (debugLogging) Print("Scrollbar created: totalEvents_Filtered=", totalEvents_Filtered, ", slider_height=", slider_height);
            updateSliderPosition();
            updateButtonColors();
         }
         if (debugLogging) Print("Success. Changes updated! State: "+(string)enableCurrencyFilter);
         ChartRedraw(0);
      }
      if (sparam == FILTER_IMP_BTN) {
         bool btn_state = ObjectGetInteger(0,sparam,OBJPROP_STATE);
         enableImportanceFilter = btn_state;
         if (debugLogging) Print(sparam+" STATE = "+(string)btn_state+", FLAG = "+(string)enableImportanceFilter);
         string filter_imp_text = enableImportanceFilter ? ShortToString(0x2714)+"Importance" : ShortToString(0x274C)+"Importance";
         color filter_imp_txt_color = enableImportanceFilter ? clrLime : clrRed;
         ObjectSetString(0,FILTER_IMP_BTN,OBJPROP_TEXT,filter_imp_text);
         ObjectSetInteger(0,FILTER_IMP_BTN,OBJPROP_COLOR,filter_imp_txt_color);
         if (MQLInfoInteger(MQL_TESTER)) filters_changed = true;
         update_dashboard_values(curr_filter_selected,imp_filter_selected);
         // Recalculate scrollbar
         ObjectDelete(0, SCROLL_LEADER);
         ObjectDelete(0, SCROLL_UP_REC);
         ObjectDelete(0, SCROLL_UP_LABEL);
         ObjectDelete(0, SCROLL_DOWN_REC);
         ObjectDelete(0, SCROLL_DOWN_LABEL);
         ObjectDelete(0, SCROLL_SLIDER);
         scroll_visible = totalEvents_Filtered > VISIBLE_ITEMS;
         if (debugLogging) Print("Scrollbar visibility: ", scroll_visible ? "Visible" : "Hidden");
         if (scroll_visible) {
            createRecLabel(SCROLL_LEADER, SCROLLBAR_X, SCROLLBAR_Y, SCROLLBAR_WIDTH, SCROLLBAR_HEIGHT, clrSilver, 1, clrNONE);
            int max_scroll = MathMax(0, ArraySize(displayableEvents) - VISIBLE_ITEMS);
            color up_color = (scroll_pos == 0) ? clrLightGray : clrBlack;
            color down_color = (scroll_pos >= max_scroll) ? clrLightGray : clrBlack;
            createRecLabel(SCROLL_UP_REC, SCROLLBAR_X + BUTTON_OFFSET_X, SCROLLBAR_Y, BUTTON_WIDTH, BUTTON_SIZE, clrDarkGray, 1, clrDarkGray);
            createLabel(SCROLL_UP_LABEL, SCROLLBAR_X + BUTTON_OFFSET_X, SCROLLBAR_Y-5, CharToString(0x35), up_color, 15, "Webdings");
            int down_y = SCROLLBAR_Y + SCROLLBAR_HEIGHT - BUTTON_SIZE;
            createRecLabel(SCROLL_DOWN_REC, SCROLLBAR_X + BUTTON_OFFSET_X, down_y, BUTTON_WIDTH, BUTTON_SIZE, clrDarkGray, 1, clrDarkGray);
            createLabel(SCROLL_DOWN_LABEL, SCROLLBAR_X + BUTTON_OFFSET_X, down_y-5, CharToString(0x36), down_color, 15, "Webdings");
            slider_height = calculateSliderHeight();
            int slider_y = SCROLLBAR_Y + BUTTON_SIZE;
            createButton(SCROLL_SLIDER, SCROLLBAR_X + SLIDER_OFFSET_X, slider_y, SLIDER_WIDTH, slider_height, "", clrWhite, 12, clrLightSlateGray, clrDarkGray, "Arial Bold");
            ObjectSetInteger(0, SCROLL_SLIDER, OBJPROP_WIDTH, 2);
            if (debugLogging) Print("Scrollbar created: totalEvents_Filtered=", totalEvents_Filtered, ", slider_height=", slider_height);
            updateSliderPosition();
            updateButtonColors();
         }
         if (debugLogging) Print("Success. Changes updated! State: "+(string)enableImportanceFilter);
         ChartRedraw(0);
      }
      if (sparam == FILTER_TIME_BTN) {
         bool btn_state = ObjectGetInteger(0,sparam,OBJPROP_STATE);
         enableTimeFilter = btn_state;
         if (debugLogging) Print(sparam+" STATE = "+(string)btn_state+", FLAG = "+(string)enableTimeFilter);
         string filter_time_text = enableTimeFilter ? ShortToString(0x2714)+"Time" : ShortToString(0x274C)+"Time";
         color filter_time_txt_color = enableTimeFilter ? clrLime : clrRed;
         ObjectSetString(0,FILTER_TIME_BTN,OBJPROP_TEXT,filter_time_text);
         ObjectSetInteger(0,FILTER_TIME_BTN,OBJPROP_COLOR,filter_time_txt_color);
         if (MQLInfoInteger(MQL_TESTER)) filters_changed = true;
         update_dashboard_values(curr_filter_selected,imp_filter_selected);
         // Recalculate scrollbar
         ObjectDelete(0, SCROLL_LEADER);
         ObjectDelete(0, SCROLL_UP_REC);
         ObjectDelete(0, SCROLL_UP_LABEL);
         ObjectDelete(0, SCROLL_DOWN_REC);
         ObjectDelete(0, SCROLL_DOWN_LABEL);
         ObjectDelete(0, SCROLL_SLIDER);
         scroll_visible = totalEvents_Filtered > VISIBLE_ITEMS;
         if (debugLogging) Print("Scrollbar visibility: ", scroll_visible ? "Visible" : "Hidden");
         if (scroll_visible) {
            createRecLabel(SCROLL_LEADER, SCROLLBAR_X, SCROLLBAR_Y, SCROLLBAR_WIDTH, SCROLLBAR_HEIGHT, clrSilver, 1, clrNONE);
            int max_scroll = MathMax(0, ArraySize(displayableEvents) - VISIBLE_ITEMS);
            color up_color = (scroll_pos == 0) ? clrLightGray : clrBlack;
            color down_color = (scroll_pos >= max_scroll) ? clrLightGray : clrBlack;
            createRecLabel(SCROLL_UP_REC, SCROLLBAR_X + BUTTON_OFFSET_X, SCROLLBAR_Y, BUTTON_WIDTH, BUTTON_SIZE, clrDarkGray, 1, clrDarkGray);
            createLabel(SCROLL_UP_LABEL, SCROLLBAR_X + BUTTON_OFFSET_X, SCROLLBAR_Y-5, CharToString(0x35), up_color, 15, "Webdings");
            int down_y = SCROLLBAR_Y + SCROLLBAR_HEIGHT - BUTTON_SIZE;
            createRecLabel(SCROLL_DOWN_REC, SCROLLBAR_X + BUTTON_OFFSET_X, down_y, BUTTON_WIDTH, BUTTON_SIZE, clrDarkGray, 1, clrDarkGray);
            createLabel(SCROLL_DOWN_LABEL, SCROLLBAR_X + BUTTON_OFFSET_X, down_y-5, CharToString(0x36), down_color, 15, "Webdings");
            slider_height = calculateSliderHeight();
            int slider_y = SCROLLBAR_Y + BUTTON_SIZE;
            createButton(SCROLL_SLIDER, SCROLLBAR_X + SLIDER_OFFSET_X, slider_y, SLIDER_WIDTH, slider_height, "", clrWhite, 12, clrLightSlateGray, clrDarkGray, "Arial Bold");
            ObjectSetInteger(0, SCROLL_SLIDER, OBJPROP_WIDTH, 2);
            if (debugLogging) Print("Scrollbar created: totalEvents_Filtered=", totalEvents_Filtered, ", slider_height=", slider_height);
            updateSliderPosition();
            updateButtonColors();
         }
         if (debugLogging) Print("Success. Changes updated! State: "+(string)enableTimeFilter);
         ChartRedraw(0);
      }
      if (StringFind(sparam,CURRENCY_BTNS) >= 0) {
         string selected_curr = ObjectGetString(0,sparam,OBJPROP_TEXT);
         if (debugLogging) Print("BTN NAME = ",sparam,", CURRENCY = ",selected_curr);
         bool btn_state = ObjectGetInteger(0,sparam,OBJPROP_STATE);
         if (btn_state == false) {
            if (debugLogging) Print("BUTTON IS IN UN-SELECTED MODE.");
            for (int i = 0; i < ArraySize(curr_filter_selected); i++) {
               if (curr_filter_selected[i] == selected_curr) {
                  for (int j = i; j < ArraySize(curr_filter_selected) - 1; j++) {
                     curr_filter_selected[j] = curr_filter_selected[j + 1];
                  }
                  ArrayResize(curr_filter_selected, ArraySize(curr_filter_selected) - 1);
                  if (debugLogging) Print("Removed from selected filters: ", selected_curr);
                  break;
               }
            }
         } else {
            if (debugLogging) Print("BUTTON IS IN SELECTED MODE. TAKE ACTION");
            bool already_selected = false;
            for (int j = 0; j < ArraySize(curr_filter_selected); j++) {
               if (curr_filter_selected[j] == selected_curr) {
                  already_selected = true;
                  break;
               }
            }
            if (!already_selected) {
               ArrayResize(curr_filter_selected, ArraySize(curr_filter_selected) + 1);
               curr_filter_selected[ArraySize(curr_filter_selected) - 1] = selected_curr;
               if (debugLogging) Print("Added to selected filters: ", selected_curr);
            } else {
               if (debugLogging) Print("Currency already selected: ", selected_curr);
            }
         }
         if (debugLogging) Print("SELECTED ARRAY SIZE = ",ArraySize(curr_filter_selected));
         if (debugLogging) ArrayPrint(curr_filter_selected);
         if (MQLInfoInteger(MQL_TESTER)) filters_changed = true;
         update_dashboard_values(curr_filter_selected,imp_filter_selected);
         // Recalculate scrollbar
         ObjectDelete(0, SCROLL_LEADER);
         ObjectDelete(0, SCROLL_UP_REC);
         ObjectDelete(0, SCROLL_UP_LABEL);
         ObjectDelete(0, SCROLL_DOWN_REC);
         ObjectDelete(0, SCROLL_DOWN_LABEL);
         ObjectDelete(0, SCROLL_SLIDER);
         scroll_visible = totalEvents_Filtered > VISIBLE_ITEMS;
         if (debugLogging) Print("Scrollbar visibility: ", scroll_visible ? "Visible" : "Hidden");
         if (scroll_visible) {
            createRecLabel(SCROLL_LEADER, SCROLLBAR_X, SCROLLBAR_Y, SCROLLBAR_WIDTH, SCROLLBAR_HEIGHT, clrSilver, 1, clrNONE);
            int max_scroll = MathMax(0, ArraySize(displayableEvents) - VISIBLE_ITEMS);
            color up_color = (scroll_pos == 0) ? clrLightGray : clrBlack;
            color down_color = (scroll_pos >= max_scroll) ? clrLightGray : clrBlack;
            createRecLabel(SCROLL_UP_REC, SCROLLBAR_X + BUTTON_OFFSET_X, SCROLLBAR_Y, BUTTON_WIDTH, BUTTON_SIZE, clrDarkGray, 1, clrDarkGray);
            createLabel(SCROLL_UP_LABEL, SCROLLBAR_X + BUTTON_OFFSET_X, SCROLLBAR_Y-5, CharToString(0x35), up_color, 15, "Webdings");
            int down_y = SCROLLBAR_Y + SCROLLBAR_HEIGHT - BUTTON_SIZE;
            createRecLabel(SCROLL_DOWN_REC, SCROLLBAR_X + BUTTON_OFFSET_X, down_y, BUTTON_WIDTH, BUTTON_SIZE, clrDarkGray, 1, clrDarkGray);
            createLabel(SCROLL_DOWN_LABEL, SCROLLBAR_X + BUTTON_OFFSET_X, down_y-5, CharToString(0x36), down_color, 15, "Webdings");
            slider_height = calculateSliderHeight();
            int slider_y = SCROLLBAR_Y + BUTTON_SIZE;
            createButton(SCROLL_SLIDER, SCROLLBAR_X + SLIDER_OFFSET_X, slider_y, SLIDER_WIDTH, slider_height, "", clrWhite, 12, clrLightSlateGray, clrDarkGray, "Arial Bold");
            ObjectSetInteger(0, SCROLL_SLIDER, OBJPROP_WIDTH, 2);
            if (debugLogging) Print("Scrollbar created: totalEvents_Filtered=", totalEvents_Filtered, ", slider_height=", slider_height);
            updateSliderPosition();
            updateButtonColors();
         }
         if (debugLogging) Print("SUCCESS. DASHBOARD UPDATED");
         ChartRedraw(0);
      }
      if (StringFind(sparam, IMPACT_LABEL) >= 0) {
         string selected_imp = ObjectGetString(0, sparam, OBJPROP_TEXT);
         ENUM_CALENDAR_EVENT_IMPORTANCE selected_importance_lvl = get_importance_level(impact_labels,allowed_importance_levels,selected_imp);
         if (debugLogging) Print("BTN NAME = ", sparam, ", IMPORTANCE LEVEL = ", selected_imp,"(",selected_importance_lvl,")");
         bool btn_state = ObjectGetInteger(0, sparam, OBJPROP_STATE);
         color color_border = btn_state ? clrNONE : clrBlack;
         if (btn_state == false) {
            if (debugLogging) Print("BUTTON IS IN UN-SELECTED MODE.");
            for (int i = 0; i < ArraySize(imp_filter_selected); i++) {
               if (impact_filter_selected[i] == selected_imp) {
                  for (int j = i; j < ArraySize(imp_filter_selected) - 1; j++) {
                     imp_filter_selected[j] = imp_filter_selected[j + 1];
                     impact_filter_selected[j] = impact_filter_selected[j + 1];
                  }
                  ArrayResize(imp_filter_selected, ArraySize(imp_filter_selected) - 1);
                  ArrayResize(impact_filter_selected, ArraySize(impact_filter_selected) - 1);
                  if (debugLogging) Print("Removed from selected importance filters: ", selected_imp,"(",selected_importance_lvl,")");
                  break;
               }
            }
         } else {
            if (debugLogging) Print("BUTTON IS IN SELECTED MODE. TAKE ACTION");
            bool already_selected = false;
            for (int j = 0; j < ArraySize(imp_filter_selected); j++) {
               if (impact_filter_selected[j] == selected_imp) {
                  already_selected = true;
                  break;
               }
            }
            if (!already_selected) {
               ArrayResize(imp_filter_selected, ArraySize(imp_filter_selected) + 1);
               imp_filter_selected[ArraySize(imp_filter_selected) - 1] = selected_importance_lvl;
               ArrayResize(impact_filter_selected, ArraySize(impact_filter_selected) + 1);
               impact_filter_selected[ArraySize(impact_filter_selected) - 1] = selected_imp;
               if (debugLogging) Print("Added to selected importance filters: ", selected_imp,"(",selected_importance_lvl,")");
            } else {
               if (debugLogging) Print("Importance level already selected: ", selected_imp,"(",selected_importance_lvl,")");
            }
         }
         if (debugLogging) Print("SELECTED ARRAY SIZE = ", ArraySize(imp_filter_selected)," >< ",ArraySize(impact_filter_selected));
         if (debugLogging) ArrayPrint(imp_filter_selected);
         if (debugLogging) ArrayPrint(impact_filter_selected);
         if (MQLInfoInteger(MQL_TESTER)) filters_changed = true;
         update_dashboard_values(curr_filter_selected,imp_filter_selected);
         ObjectSetInteger(0,sparam,OBJPROP_BORDER_COLOR,color_border);
         // Recalculate scrollbar
         ObjectDelete(0, SCROLL_LEADER);
         ObjectDelete(0, SCROLL_UP_REC);
         ObjectDelete(0, SCROLL_UP_LABEL);
         ObjectDelete(0, SCROLL_DOWN_REC);
         ObjectDelete(0, SCROLL_DOWN_LABEL);
         ObjectDelete(0, SCROLL_SLIDER);
         scroll_visible = totalEvents_Filtered > VISIBLE_ITEMS;
         if (debugLogging) Print("Scrollbar visibility: ", scroll_visible ? "Visible" : "Hidden");
         if (scroll_visible) {
            createRecLabel(SCROLL_LEADER, SCROLLBAR_X, SCROLLBAR_Y, SCROLLBAR_WIDTH, SCROLLBAR_HEIGHT, clrSilver, 1, clrNONE);
            int max_scroll = MathMax(0, ArraySize(displayableEvents) - VISIBLE_ITEMS);
            color up_color = (scroll_pos == 0) ? clrLightGray : clrBlack;
            color down_color = (scroll_pos >= max_scroll) ? clrLightGray : clrBlack;
            createRecLabel(SCROLL_UP_REC, SCROLLBAR_X + BUTTON_OFFSET_X, SCROLLBAR_Y, BUTTON_WIDTH, BUTTON_SIZE, clrDarkGray, 1, clrDarkGray);
            createLabel(SCROLL_UP_LABEL, SCROLLBAR_X + BUTTON_OFFSET_X, SCROLLBAR_Y-5, CharToString(0x35), up_color, 15, "Webdings");
            int down_y = SCROLLBAR_Y + SCROLLBAR_HEIGHT - BUTTON_SIZE;
            createRecLabel(SCROLL_DOWN_REC, SCROLLBAR_X + BUTTON_OFFSET_X, down_y, BUTTON_WIDTH, BUTTON_SIZE, clrDarkGray, 1, clrDarkGray);
            createLabel(SCROLL_DOWN_LABEL, SCROLLBAR_X + BUTTON_OFFSET_X, down_y-5, CharToString(0x36), down_color, 15, "Webdings");
            slider_height = calculateSliderHeight();
            int slider_y = SCROLLBAR_Y + BUTTON_SIZE;
            createButton(SCROLL_SLIDER, SCROLLBAR_X + SLIDER_OFFSET_X, slider_y, SLIDER_WIDTH, slider_height, "", clrWhite, 12, clrLightSlateGray, clrDarkGray, "Arial Bold");
            ObjectSetInteger(0, SCROLL_SLIDER, OBJPROP_WIDTH, 2);
            if (debugLogging) Print("Scrollbar created: totalEvents_Filtered=", totalEvents_Filtered, ", slider_height=", slider_height);
            updateSliderPosition();
            updateButtonColors();
         }
         if (debugLogging) Print("SUCCESS. DASHBOARD UPDATED");
         ChartRedraw(0);
      }
      // Scrollbar button clicks
      if (scroll_visible && (sparam == SCROLL_UP_REC || sparam == SCROLL_UP_LABEL)) {
         scrollUp();
         updateButtonColors();
         if (debugLogging) Print("Up button clicked (", sparam, "). CurrPos: ", scroll_pos);
         ChartRedraw(0);
      }
      if (scroll_visible && (sparam == SCROLL_DOWN_REC || sparam == SCROLL_DOWN_LABEL)) {
         scrollDown();
         updateButtonColors();
         if (debugLogging) Print("Down button clicked (", sparam, "). CurrPos: ", scroll_pos);
         ChartRedraw(0);
      }
   }
   else if (id == CHARTEVENT_MOUSE_MOVE && scroll_visible) {
      if (prev_mouse_state == 0 && mouse_state == 1) {
         int xd = (int)ObjectGetInteger(0, SCROLL_SLIDER, OBJPROP_XDISTANCE);
         int yd = (int)ObjectGetInteger(0, SCROLL_SLIDER, OBJPROP_YDISTANCE);
         int xs = (int)ObjectGetInteger(0, SCROLL_SLIDER, OBJPROP_XSIZE);
         int ys = (int)ObjectGetInteger(0, SCROLL_SLIDER, OBJPROP_YSIZE);
         if (mouse_x >= xd && mouse_x <= xd + xs && mouse_y >= yd && mouse_y <= yd + ys) {
            moving_state_slider = true;
            mlb_down_x = mouse_x;
            mlb_down_y = mouse_y;
            mlb_down_yd_slider = yd;
            ObjectSetInteger(0, SCROLL_SLIDER, OBJPROP_BGCOLOR, clrDodgerBlue);
            ObjectSetInteger(0, SCROLL_SLIDER, OBJPROP_YSIZE, slider_height + 2);
            ChartSetInteger(0, CHART_MOUSE_SCROLL, false);
            if (debugLogging) Print("Slider drag started at y=", mouse_y);
         }
      }
      if (moving_state_slider && mouse_state == 1) {
         int delta_y = mouse_y - mlb_down_y;
         int new_y = mlb_down_yd_slider + delta_y;
         int scroll_area_y_min = SCROLLBAR_Y + BUTTON_SIZE;
         int scroll_area_y_max = scroll_area_y_min + SCROLL_AREA_HEIGHT - slider_height;
         new_y = MathMax(scroll_area_y_min, MathMin(new_y, scroll_area_y_max));
         ObjectSetInteger(0, SCROLL_SLIDER, OBJPROP_YDISTANCE, new_y);
         int max_scroll = MathMax(0, ArraySize(displayableEvents) - VISIBLE_ITEMS);
         double scroll_ratio = (double)(new_y - scroll_area_y_min) / (scroll_area_y_max - scroll_area_y_min);
         int new_scroll_pos = (int)MathRound(scroll_ratio * max_scroll);
         if (new_scroll_pos != scroll_pos) {
            scroll_pos = new_scroll_pos;
            update_dashboard_values(curr_filter_selected, imp_filter_selected);
            updateButtonColors();
            if (debugLogging) Print("Slider dragged. CurrPos: ", scroll_pos, ", Total steps: ", max_scroll, ", Slider y=", new_y);
         }
         ChartRedraw(0);
      }
      if (mouse_state == 0) {
         if (moving_state_slider) {
            moving_state_slider = false;
            ObjectSetInteger(0, SCROLL_SLIDER, OBJPROP_BGCOLOR, clrLightSlateGray);
            ObjectSetInteger(0, SCROLL_SLIDER, OBJPROP_YSIZE, slider_height);
            ChartSetInteger(0, CHART_MOUSE_SCROLL, true);
            if (debugLogging) Print("Slider drag stopped.");
            ChartRedraw(0);
         }
      }
      prev_mouse_state = mouse_state;
   }
}
```

Here, we just took the implementation of the logic that we had with the scrollbar element clicks and extended it to the individual filter buttons. We don't need to spend much time on this. Upon compilation, we have the following outcome.

![](https://c.mql5.com/2/142/Calendar_part_9_TESTING_GIF_2.gif)

From the visualization, we can see that everything is now working fine, with dynamic updates and a polished display of the events.

### Conclusion

In conclusion, we’ve advanced the [MQL5 Economic Calendar](https://www.mql5.com/en/economic-calendar) series by introducing a dynamic scrollbar and polished event display, empowering us with intuitive navigation and reliable access to news events, as demonstrated in our comprehensive GIF. These enhancements, built on [Part 8’s optimized backtesting](https://www.mql5.com/en/articles/17999), ensure seamless interaction across live and tester modes, providing a robust platform for news-driven trading strategies. You can leverage this refined dashboard as a foundation, customizing it to suit your unique trading requirements.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/18135.zip "Download all attachments in the single ZIP archive")

[MQL5\_NEWS\_CALENDAR\_PART\_9.mq5](https://www.mql5.com/en/articles/download/18135/mql5_news_calendar_part_9.mq5 "Download MQL5_NEWS_CALENDAR_PART_9.mq5")(166.58 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [MQL5 Trading Tools (Part 12): Enhancing the Correlation Matrix Dashboard with Interactivity](https://www.mql5.com/en/articles/20962)
- [Creating Custom Indicators in MQL5 (Part 5): WaveTrend Crossover Evolution Using Canvas for Fog Gradients, Signal Bubbles, and Risk Management](https://www.mql5.com/en/articles/20815)
- [MQL5 Trading Tools (Part 11): Correlation Matrix Dashboard (Pearson, Spearman, Kendall) with Heatmap and Standard Modes](https://www.mql5.com/en/articles/20945)
- [Creating Custom Indicators in MQL5 (Part 4): Smart WaveTrend Crossover with Dual Oscillators](https://www.mql5.com/en/articles/20811)
- [Building AI-Powered Trading Systems in MQL5 (Part 8): UI Polish with Animations, Timing Metrics, and Response Management Tools](https://www.mql5.com/en/articles/20722)
- [Creating Custom Indicators in MQL5 (Part 3): Multi-Gauge Enhancements with Sector and Round Styles](https://www.mql5.com/en/articles/20719)
- [Creating Custom Indicators in MQL5 (Part 2): Building a Gauge-Style RSI Display with Canvas and Needle Mechanics](https://www.mql5.com/en/articles/20632)

**[Go to discussion](https://www.mql5.com/en/forum/486644)**

![Neural Networks in Trading: Controlled Segmentation](https://c.mql5.com/2/96/Neural_Networks_in_Trading_Controlled_Segmentation___LOGO.png)[Neural Networks in Trading: Controlled Segmentation](https://www.mql5.com/en/articles/16038)

In this article. we will discuss a method of complex multimodal interaction analysis and feature understanding.

![MQL5 Wizard Techniques you should know (Part 65): Using Patterns of FrAMA and the Force Index](https://c.mql5.com/2/143/18144-mql5-wizard-techniques-you-logo.png)[MQL5 Wizard Techniques you should know (Part 65): Using Patterns of FrAMA and the Force Index](https://www.mql5.com/en/articles/18144)

The Fractal Adaptive Moving Average (FrAMA) and the Force Index Oscillator are another pair of indicators that could be used in conjunction within an MQL5 Expert Advisor. These two indicators complement each other a little bit because FrAMA is a trend following indicator while the Force Index is a volume based oscillator. As always, we use the MQL5 wizard to rapidly explore any potential these two may have.

![Overcoming The Limitation of Machine Learning (Part 2): Lack of Reproducibility](https://c.mql5.com/2/143/18133-overcoming-the-limitation-of-logo.png)[Overcoming The Limitation of Machine Learning (Part 2): Lack of Reproducibility](https://www.mql5.com/en/articles/18133)

The article explores why trading results can differ significantly between brokers, even when using the same strategy and financial symbol, due to decentralized pricing and data discrepancies. The piece helps MQL5 developers understand why their products may receive mixed reviews on the MQL5 Marketplace, and urges developers to tailor their approaches to specific brokers to ensure transparent and reproducible outcomes. This could grow to become an important domain-bound best practice that will serve our community well if the practice were to be widely adopted.

![Price Action Analysis Toolkit Development (Part 23): Currency Strength Meter](https://c.mql5.com/2/143/18108-price-action-analysis-toolkit-logo.png)[Price Action Analysis Toolkit Development (Part 23): Currency Strength Meter](https://www.mql5.com/en/articles/18108)

Do you know what really drives a currency pair’s direction? It’s the strength of each individual currency. In this article, we’ll measure a currency’s strength by looping through every pair it appears in. That insight lets us predict how those pairs may move based on their relative strengths. Read on to learn more.

[![](https://www.mql5.com/ff/sh/0hvxp984jjj79943z2/6373d9e5710a718ffa6a7d50a5db9dd1.jpg)\\
Web terminal on your iPhone or Android\\
\\
Full-featured MetaTrader 5 platform for any devices and web browsers\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=uyigsjnbfcdvysiynusmriwvhincciwd&s=c95531ae2fd8a81b0fac3def2e4cf820a67584bbf4b02f76ec75f808942dbbd2&uid=&ref=https://www.mql5.com/en/articles/18135&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5083441343918775237)

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