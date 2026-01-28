---
title: Trading with the MQL5 Economic Calendar (Part 10): Draggable Dashboard and Interactive Hover Effects for Seamless News Navigation
url: https://www.mql5.com/en/articles/18241
categories: Trading, Trading Systems, Expert Advisors
relevance_score: -2
scraped_at: 2026-01-24T14:15:09.767116
---

[![](https://www.mql5.com/ff/sh/5z040u47jcv59943z2/6c76c03a8b37e08b8655a1a085770b7a.jpg)\\
MetaTrader 5 for iOS and Android\\
\\
Fully featured platform for any devices and web browsers\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=ddonqpipxfqlnsvzlwuowsuwlejpyjxk&s=9daba65b69f40afc3c35f95b1f84ef5824d68c47f29ce96a6dc5b164a2727baa&uid=&ref=https://www.mql5.com/en/articles/18241&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5083439467018066875)

MetaTrader 5 / Trading


### Introduction

In this article, we advance the [MQL5 Economic Calendar](https://www.mql5.com/en/book/advanced/calendar) series by introducing a draggable dashboard and interactive hover effects to enhance trader control and navigation of news events, ensuring a flexible and intuitive user experience. Building on [Part 9’s dynamic scrollbar and polished display](https://www.mql5.com/en/articles/18135), we now focus on a responsive [User Interface](https://en.wikipedia.org/wiki/User_interface "https://en.wikipedia.org/wiki/User_interface") (UI) that allows repositioning the calendar on the chart and provides visual feedback for button interactions, optimizing access to economic news in live and backtesting environments. We structure the article with the following topics:

1. [Adding a Draggable Dashboard for Enhanced Chart Flexibility](https://www.mql5.com/en/articles/18241#para1)
2. [Implementation in MQL5](https://www.mql5.com/en/articles/18241#para2)
3. [Testing and Validation](https://www.mql5.com/en/articles/18241#para3)
4. [Conclusion](https://www.mql5.com/en/articles/18241#para4)

Let’s dive into these enhancements!

### Adding a Draggable Dashboard for Enhanced Chart Flexibility

To elevate the MQL5 Economic Calendar’s usability, we will introduce a draggable dashboard that will allow us to reposition the interface on the chart, paired with a dynamically positioned scrollbar to maintain seamless news navigation. We aim to create a trader-centric tool that adapts to diverse chart layouts, eliminating the fixed positioning of the previous versions and ensuring the dashboard, news events, and scrollbar move together effortlessly. Here’s how we will achieve this:

- Draggable Dashboard Design: We will implement a system to detect mouse clicks on the header area, enabling users to drag the entire dashboard. All UI elements will update their positions in real-time to maintain alignment.
- Dynamic Scrollbar Positioning: We will adjust the scrollbar to use relative coordinates tied to the dashboard’s position, ensuring it remains functional and correctly placed during dragging.
- Chart Boundary Constraints: We will enforce limits to keep the dashboard within the chart’s visible area, preventing it from moving off-screen and ensuring accessibility.
- Cohesive Element Movement: We will ensure that news events, filter buttons, and trade labels move in sync with the dashboard, providing a unified and professional interface.

This strategic approach will transform the dashboard into a flexible tool that we can position as needed, enhancing chart visibility and interaction. In a nutshell, this is what we aim to achieve.

![STRATEGY PLAN](https://c.mql5.com/2/144/Screenshot_2025-05-22_185716.png)

### Implementation in MQL5

To make the improvements in MQL5, we first will need to define some extra [functions](https://www.mql5.com/en/docs/basis/function) for enhancing the dynamic interactivity. We will start from the simplest to the most complex, and that is from hover states and drag states. Let us first define a [void](https://www.mql5.com/en/docs/basis/types/void) function, that will determine a hovered button state depending on the cursor position on the chart and update the button states, typically by darkening them, but you can define your color hover states. So, let us first define the function responsible for color darkening.

```
//+------------------------------------------------------------------+
//| Helper function to darken a color for hover effect               |
//+------------------------------------------------------------------+
color ColorToDarken(color clr) {
   int r = (clr & 0xFF);
   int g = ((clr >> 8) & 0xFF);
   int b = ((clr >> 16) & 0xFF);
   r = MathMax(0, r - 50);
   g = MathMax(0, g - 50);
   b = MathMax(0, b - 50);
   return (color)((b << 16) | (g << 8) | r);
}
```

First, we define the "ColorToDarken" function to generate a darker version of an input color, which we will use to enhance the visual feedback for buttons when we hover over them. We start with the input color "clr", a single number that combines red, green, and blue components, each ranging from 0 to [255](https://www.mql5.com/go?link=https://www.ascii-code.com/CP1252/255 "https://www.ascii-code.com/CP1252/255"), to represent a specific color. To separate these components, we employ [bitwise operations](https://www.mql5.com/en/docs/basis/operations/bit), which allow us to manipulate the bits within "clr".

For the red component "r", we use the "&" operator (bitwise AND) with "0xFF", a [hexadecimal](https://en.wikipedia.org/wiki/Hexadecimal "https://en.wikipedia.org/wiki/Hexadecimal") value equal to 255 in decimal, which acts like a filter to extract only the lowest 8 bits of "clr", giving us the red intensity. For the green component "g", we shift "clr" right by 8 bits using the ">>" operator to move the green data into the lowest 8 bits, then apply "&" with "0xFF" to isolate it. Similarly, we extract the blue component "b" by shifting "clr" right by 16 bits and using "&" with "0xFF". The [characters in hexadecimal](https://www.mql5.com/go?link=https://www.ascii-code.com/CP1252/255 "https://www.ascii-code.com/CP1252/255") are as below.

![255 IN HEXADECIMAL 0xFF](https://c.mql5.com/2/144/Screenshot_2025-05-22_200611.png)

To darken the color, we subtract 50 from each component—"r", "g", and "b"—to reduce their intensity, and we use the " [MathMax](https://www.mql5.com/en/docs/math/mathmax)" function to ensure no value goes below 0 since negative color values are invalid. Finally, we combine the darkened components back into a single color value by shifting "b" left by 16 bits with the "<<" operator, shifting "g" left by 8 bits, and merging them with "r" using the "\|" (bitwise OR) operator. We return this new "color" value, which we apply to buttons to create a darker hover effect, making the interface more interactive and intuitive. With this function, we can now use it in formatting the hovered color states dynamically.

```
//+------------------------------------------------------------------+
//| Update hover states for header and buttons                       |
//+------------------------------------------------------------------+
void updateHoverStates(int mouse_x, int mouse_y) {
   // Header hover
   int header_x = (int)ObjectGetInteger(0, HEADER_LABEL, OBJPROP_XDISTANCE);
   int header_y = (int)ObjectGetInteger(0, HEADER_LABEL, OBJPROP_YDISTANCE);
   int header_width = 740;
   int header_height = 30;

   bool is_header_hovered = (mouse_x >= header_x && mouse_x <= header_x + header_width &&
                             mouse_y >= header_y && mouse_y <= header_y + header_height);

   if (is_header_hovered && !header_hovered) {
      ObjectSetInteger(0, MAIN_REC, OBJPROP_BGCOLOR, clrDarkGreen);
      header_hovered = true;
   } else if (!is_header_hovered && header_hovered) {
      ObjectSetInteger(0, MAIN_REC, OBJPROP_BGCOLOR, clrSeaGreen);
      header_hovered = false;
   }
}
```

Here, we implement the "updateHoverStates" function to add a hover effect on the header. We retrieve the "HEADER\_LABEL" position using [ObjectGetInteger](https://www.mql5.com/en/docs/objects/ObjectGetInteger) to get "header\_x" and "header\_y" with [OBJPROP\_XDISTANCE](https://www.mql5.com/en/docs/constants/objectconstants/enum_object_property#enum_object_property_integer) and [OBJPROP\_YDISTANCE](https://www.mql5.com/en/docs/constants/objectconstants/enum_object_property#enum_object_property_integer), defining a 740x30 pixel area. We check if "mouse\_x" and "mouse\_y" are within this area, setting "is\_header\_hovered".

If "is\_header\_hovered" is true and "header\_hovered" is false, we use [ObjectSetInteger](https://www.mql5.com/en/docs/objects/objectsetinteger) to set the "MAIN\_REC" [OBJPROP\_BGCOLOR](https://www.mql5.com/en/docs/constants/objectconstants/enum_object_property#enum_object_property_integer) to "clrDarkGreen" and update "header\_hovered" to true; if false and "header\_hovered" is true, we restore it to [clrSeaGreen](https://www.mql5.com/en/docs/constants/objectconstants/webcolors) and set "header\_hovered" to false, providing visual feedback for hovering. We can now do the same to the other buttons.

```
// FILTER_CURR_BTN hover
int curr_btn_x = (int)ObjectGetInteger(0, FILTER_CURR_BTN, OBJPROP_XDISTANCE);
int curr_btn_y = (int)ObjectGetInteger(0, FILTER_CURR_BTN, OBJPROP_YDISTANCE);
int curr_btn_width = 110;
int curr_btn_height = 26;

bool is_curr_btn_hovered = (mouse_x >= curr_btn_x && mouse_x <= curr_btn_x + curr_btn_width &&
                            mouse_y >= curr_btn_y && mouse_y <= curr_btn_y + curr_btn_height);

if (is_curr_btn_hovered && !filter_curr_hovered) {
   ObjectSetInteger(0, FILTER_CURR_BTN, OBJPROP_BGCOLOR, clrDarkGray);
   filter_curr_hovered = true;
} else if (!is_curr_btn_hovered && filter_curr_hovered) {
   ObjectSetInteger(0, FILTER_CURR_BTN, OBJPROP_BGCOLOR, clrBlack);
   filter_curr_hovered = false;
}

// FILTER_IMP_BTN hover
int imp_btn_x = (int)ObjectGetInteger(0, FILTER_IMP_BTN, OBJPROP_XDISTANCE);
int imp_btn_y = (int)ObjectGetInteger(0, FILTER_IMP_BTN, OBJPROP_YDISTANCE);
int imp_btn_width = 120;
int imp_btn_height = 26;

bool is_imp_btn_hovered = (mouse_x >= imp_btn_x && mouse_x <= imp_btn_x + imp_btn_width &&
                           mouse_y >= imp_btn_y && mouse_y <= imp_btn_y + imp_btn_height);

if (is_imp_btn_hovered && !filter_imp_hovered) {
   ObjectSetInteger(0, FILTER_IMP_BTN, OBJPROP_BGCOLOR, clrDarkGray);
   filter_imp_hovered = true;
} else if (!is_imp_btn_hovered && filter_imp_hovered) {
   ObjectSetInteger(0, FILTER_IMP_BTN, OBJPROP_BGCOLOR, clrBlack);
   filter_imp_hovered = false;
}

// FILTER_TIME_BTN hover
int time_btn_x = (int)ObjectGetInteger(0, FILTER_TIME_BTN, OBJPROP_XDISTANCE);
int time_btn_y = (int)ObjectGetInteger(0, FILTER_TIME_BTN, OBJPROP_YDISTANCE);
int time_btn_width = 70;
int time_btn_height = 26;

bool is_time_btn_hovered = (mouse_x >= time_btn_x && mouse_x <= time_btn_x + time_btn_width &&
                            mouse_y >= time_btn_y && mouse_y <= time_btn_y + time_btn_height);

if (is_time_btn_hovered && !filter_time_hovered) {
   ObjectSetInteger(0, FILTER_TIME_BTN, OBJPROP_BGCOLOR, clrDarkGray);
   filter_time_hovered = true;
} else if (!is_time_btn_hovered && filter_time_hovered) {
   ObjectSetInteger(0, FILTER_TIME_BTN, OBJPROP_BGCOLOR, clrBlack);
   filter_time_hovered = false;
}

// CANCEL_BTN hover
int cancel_btn_x = (int)ObjectGetInteger(0, CANCEL_BTN, OBJPROP_XDISTANCE);
int cancel_btn_y = (int)ObjectGetInteger(0, CANCEL_BTN, OBJPROP_YDISTANCE);
int cancel_btn_width = 50;
int cancel_btn_height = 30;

bool is_cancel_btn_hovered = (mouse_x >= cancel_btn_x && mouse_x <= cancel_btn_x + cancel_btn_width &&
                              mouse_y >= cancel_btn_y && mouse_y <= cancel_btn_y + cancel_btn_height);

if (is_cancel_btn_hovered && !cancel_hovered) {
   ObjectSetInteger(0, CANCEL_BTN, OBJPROP_BGCOLOR, clrDarkRed);
   cancel_hovered = true;
} else if (!is_cancel_btn_hovered && cancel_hovered) {
   ObjectSetInteger(0, CANCEL_BTN, OBJPROP_BGCOLOR, clrRed);
   cancel_hovered = false;
}

// CURRENCY_BTNS hover
int curr_size = 51, button_height = 22, spacing_x = 0, spacing_y = 3, max_columns = 4;
for (int i = 0; i < ArraySize(curr_filter); i++) {
   int row = i / max_columns;
   int col = i % max_columns;
   int x_pos = panel_x + 525 + col * (curr_size + spacing_x);
   int y_pos = panel_y + 33 + row * (button_height + spacing_y);
   bool is_curr_hovered = (mouse_x >= x_pos && mouse_x <= x_pos + curr_size &&
                           mouse_y >= y_pos && mouse_y <= y_pos + button_height);
   string btn_name = CURRENCY_BTNS+IntegerToString(i);
   if (is_curr_hovered && !currency_btns_hovered[i]) {
      ObjectSetInteger(0, btn_name, OBJPROP_BGCOLOR, clrLightGray);
      currency_btns_hovered[i] = true;
   } else if (!is_curr_hovered && currency_btns_hovered[i]) {
      ObjectSetInteger(0, btn_name, OBJPROP_BGCOLOR, clrNONE);
      currency_btns_hovered[i] = false;
   }
}

// IMPACT_LABEL buttons hover
int impact_size = 100;
for (int i = 0; i < ArraySize(impact_labels); i++) {
   int x_pos = panel_x + 90 + impact_size * i;
   int y_pos = panel_y + 55;
   bool is_impact_hovered = (mouse_x >= x_pos && mouse_x <= x_pos + impact_size &&
                             mouse_y >= y_pos && mouse_y <= y_pos + 25);
   string btn_name = IMPACT_LABEL+string(i);
   color normal_color = clrBlack;
   if (impact_labels[i] == "None") normal_color = clrBlack;
   else if (impact_labels[i] == "Low") normal_color = clrYellow;
   else if (impact_labels[i] == "Medium") normal_color = clrOrange;
   else if (impact_labels[i] == "High") normal_color = clrRed;
   color hover_color = normal_color == clrBlack ? clrDarkGray : ColorToDarken(normal_color);
   if (is_impact_hovered && !impact_btns_hovered[i]) {
      ObjectSetInteger(0, btn_name, OBJPROP_BGCOLOR, hover_color);
      impact_btns_hovered[i] = true;
   } else if (!is_impact_hovered && impact_btns_hovered[i]) {
      ObjectSetInteger(0, btn_name, OBJPROP_BGCOLOR, normal_color);
      impact_btns_hovered[i] = false;
   }
}
```

We enhance the "updateHoverStates" function to add hover effects for multiple buttons, providing visual feedback for user interactions. For the "FILTER\_CURR\_BTN", we use the [ObjectGetInteger](https://www.mql5.com/en/docs/objects/ObjectGetInteger) function to retrieve its coordinates "curr\_btn\_x" and "curr\_btn\_y" with [OBJPROP\_XDISTANCE](https://www.mql5.com/en/docs/constants/objectconstants/enum_object_property#enum_object_property_integer) and "OBJPROP\_YDISTANCE", defining a 110x26 pixel area, and check if "mouse\_x" and "mouse\_y" are within it to set "is\_curr\_btn\_hovered".

If "is\_curr\_btn\_hovered" is true and "filter\_curr\_hovered" is false, we set the "FILTER\_CURR\_BTN" and [OBJPROP\_BGCOLOR](https://www.mql5.com/en/docs/constants/objectconstants/enum_object_property#enum_object_property_integer) to [clrDarkGray](https://www.mql5.com/en/docs/constants/objectconstants/webcolors) using [ObjectSetInteger](https://www.mql5.com/en/docs/objects/objectsetinteger) and update "filter\_curr\_hovered" to true; otherwise, we restore it to "clrBlack" and set "filter\_curr\_hovered" to false.

We apply similar logic for "FILTER\_IMP\_BTN" (120x26 pixels) and "FILTER\_TIME\_BTN" (70x26 pixels), updating "filter\_imp\_hovered" and "filter\_time\_hovered" respectively. For "CANCEL\_BTN" (50x30 pixels), we toggle between [clrDarkRed](https://www.mql5.com/en/docs/constants/objectconstants/webcolors) and "clrRed" based on "is\_cancel\_btn\_hovered" and "cancel\_hovered". For "CURRENCY\_BTNS", we loop through each button, calculating positions with "panel\_x" and "panel\_y", checking "is\_curr\_hovered", and toggling "clrLightGray" and "clrNONE" for "currency\_btns\_hovered\[i\]".

For "IMPACT\_LABEL" buttons, we assign "normal\_color" based on impact level (e.g., "clrYellow" for Low), compute "hover\_color" with "ColorToDarken", and toggle colors based on "is\_impact\_hovered" and "impact\_btns\_hovered\[i\]", ensuring dynamic hover effects. For the changes to take effect, we call the function on the [OnChartEvent](https://www.mql5.com/en/docs/event_handlers/onchartevent) event handler.

```
//+------------------------------------------------------------------+
//| Chart event handler                                              |
//+------------------------------------------------------------------+
void OnChartEvent(const int id, const long& lparam, const double& dparam, const string& sparam) {
   int mouse_x = (int)lparam;
   int mouse_y = (int)dparam;
   int mouse_state = (int)sparam;

   // Update hover states
   if (id == CHARTEVENT_MOUSE_MOVE) {
      updateHoverStates(mouse_x, mouse_y);
   }
}
```

In the [OnChartEvent](https://www.mql5.com/en/docs/event_handlers/onchartevent) event handler, we begin by capturing the mouse coordinates "mouse\_x" and "mouse\_y" from the "lparam" and "dparam" parameters, and the mouse state "mouse\_state" from "sparam", which indicate the mouse’s position and whether it’s clicked or released. When the event "id" is [CHARTEVENT\_MOUSE\_MOVE](https://www.mql5.com/en/docs/constants/chartconstants/enum_chartevents), we call the "updateHoverStates" function, passing "mouse\_x" and "mouse\_y", to check if the mouse is over the header or buttons and update their visual appearance accordingly, ensuring responsive hover feedback for the dashboard’s interactive elements. We then need to capture the other element updates on click and determine their movement status.

```
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
         createRecLabel(SCROLL_LEADER, panel_x + SCROLLBAR_X_OFFSET, panel_y + SCROLLBAR_Y_OFFSET, SCROLLBAR_WIDTH, SCROLLBAR_HEIGHT, clrSilver, 1, clrNONE);
         int max_scroll = MathMax(0, ArraySize(displayableEvents) - VISIBLE_ITEMS);
         color up_color = (scroll_pos == 0) ? clrLightGray : clrBlack;
         color down_color = (scroll_pos >= max_scroll) ? clrLightGray : clrBlack;
         createRecLabel(SCROLL_UP_REC, panel_x + SCROLLBAR_X_OFFSET + BUTTON_OFFSET_X, panel_y + SCROLLBAR_Y_OFFSET, BUTTON_WIDTH, BUTTON_SIZE, clrDarkGray, 1, clrDarkGray);
         createLabel(SCROLL_UP_LABEL, panel_x + SCROLLBAR_X_OFFSET + BUTTON_OFFSET_X, panel_y + SCROLLBAR_Y_OFFSET-5, CharToString(0x35), up_color, 15, "Webdings");
         int down_y = panel_y + SCROLLBAR_Y_OFFSET + SCROLLBAR_HEIGHT - BUTTON_SIZE;
         createRecLabel(SCROLL_DOWN_REC, panel_x + SCROLLBAR_X_OFFSET + BUTTON_OFFSET_X, down_y, BUTTON_WIDTH, BUTTON_SIZE, clrDarkGray, 1, clrDarkGray);
         createLabel(SCROLL_DOWN_LABEL, panel_x + SCROLLBAR_X_OFFSET + BUTTON_OFFSET_X, down_y-5, CharToString(0x36), down_color, 15, "Webdings");
         slider_height = calculateSliderHeight();
         int slider_y = panel_y + SCROLLBAR_Y_OFFSET + BUTTON_SIZE;
         createButton(SCROLL_SLIDER, panel_x + SCROLLBAR_X_OFFSET + SLIDER_OFFSET_X, slider_y, SLIDER_WIDTH, slider_height, "", clrWhite, 12, clrLightSlateGray, clrDarkGray, "Arial Bold");
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
         createRecLabel(SCROLL_LEADER, panel_x + SCROLLBAR_X_OFFSET, panel_y + SCROLLBAR_Y_OFFSET, SCROLLBAR_WIDTH, SCROLLBAR_HEIGHT, clrSilver, 1, clrNONE);
         int max_scroll = MathMax(0, ArraySize(displayableEvents) - VISIBLE_ITEMS);
         color up_color = (scroll_pos == 0) ? clrLightGray : clrBlack;
         color down_color = (scroll_pos >= max_scroll) ? clrLightGray : clrBlack;
         createRecLabel(SCROLL_UP_REC, panel_x + SCROLLBAR_X_OFFSET + BUTTON_OFFSET_X, panel_y + SCROLLBAR_Y_OFFSET, BUTTON_WIDTH, BUTTON_SIZE, clrDarkGray, 1, clrDarkGray);
         createLabel(SCROLL_UP_LABEL, panel_x + SCROLLBAR_X_OFFSET + BUTTON_OFFSET_X, panel_y + SCROLLBAR_Y_OFFSET-5, CharToString(0x35), up_color, 15, "Webdings");
         int down_y = panel_y + SCROLLBAR_Y_OFFSET + SCROLLBAR_HEIGHT - BUTTON_SIZE;
         createRecLabel(SCROLL_DOWN_REC, panel_x + SCROLLBAR_X_OFFSET + BUTTON_OFFSET_X, down_y, BUTTON_WIDTH, BUTTON_SIZE, clrDarkGray, 1, clrDarkGray);
         createLabel(SCROLL_DOWN_LABEL, panel_x + SCROLLBAR_X_OFFSET + BUTTON_OFFSET_X, down_y-5, CharToString(0x36), down_color, 15, "Webdings");
         slider_height = calculateSliderHeight();
         int slider_y = panel_y + SCROLLBAR_Y_OFFSET + BUTTON_SIZE;
         createButton(SCROLL_SLIDER, panel_x + SCROLLBAR_X_OFFSET + SLIDER_OFFSET_X, slider_y, SLIDER_WIDTH, slider_height, "", clrWhite, 12, clrLightSlateGray, clrDarkGray, "Arial Bold");
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
         createRecLabel(SCROLL_LEADER, panel_x + SCROLLBAR_X_OFFSET, panel_y + SCROLLBAR_Y_OFFSET, SCROLLBAR_WIDTH, SCROLLBAR_HEIGHT, clrSilver, 1, clrNONE);
         int max_scroll = MathMax(0, ArraySize(displayableEvents) - VISIBLE_ITEMS);
         color up_color = (scroll_pos == 0) ? clrLightGray : clrBlack;
         color down_color = (scroll_pos >= max_scroll) ? clrLightGray : clrBlack;
         createRecLabel(SCROLL_UP_REC, panel_x + SCROLLBAR_X_OFFSET + BUTTON_OFFSET_X, panel_y + SCROLLBAR_Y_OFFSET, BUTTON_WIDTH, BUTTON_SIZE, clrDarkGray, 1, clrDarkGray);
         createLabel(SCROLL_UP_LABEL, panel_x + SCROLLBAR_X_OFFSET + BUTTON_OFFSET_X, panel_y + SCROLLBAR_Y_OFFSET-5, CharToString(0x35), up_color, 15, "Webdings");
         int down_y = panel_y + SCROLLBAR_Y_OFFSET + SCROLLBAR_HEIGHT - BUTTON_SIZE;
         createRecLabel(SCROLL_DOWN_REC, panel_x + SCROLLBAR_X_OFFSET + BUTTON_OFFSET_X, down_y, BUTTON_WIDTH, BUTTON_SIZE, clrDarkGray, 1, clrDarkGray);
         createLabel(SCROLL_DOWN_LABEL, panel_x + SCROLLBAR_X_OFFSET + BUTTON_OFFSET_X, down_y-5, CharToString(0x36), down_color, 15, "Webdings");
         slider_height = calculateSliderHeight();
         int slider_y = panel_y + SCROLLBAR_Y_OFFSET + BUTTON_SIZE;
         createButton(SCROLL_SLIDER, panel_x + SCROLLBAR_X_OFFSET + SLIDER_OFFSET_X, slider_y, SLIDER_WIDTH, slider_height, "", clrWhite, 12, clrLightSlateGray, clrDarkGray, "Arial Bold");
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
         createRecLabel(SCROLL_LEADER, panel_x + SCROLLBAR_X_OFFSET, panel_y + SCROLLBAR_Y_OFFSET, SCROLLBAR_WIDTH, SCROLLBAR_HEIGHT, clrSilver, 1, clrNONE);
         int max_scroll = MathMax(0, ArraySize(displayableEvents) - VISIBLE_ITEMS);
         color up_color = (scroll_pos == 0) ? clrLightGray : clrBlack;
         color down_color = (scroll_pos >= max_scroll) ? clrLightGray : clrBlack;
         createRecLabel(SCROLL_UP_REC, panel_x + SCROLLBAR_X_OFFSET + BUTTON_OFFSET_X, panel_y + SCROLLBAR_Y_OFFSET, BUTTON_WIDTH, BUTTON_SIZE, clrDarkGray, 1, clrDarkGray);
         createLabel(SCROLL_UP_LABEL, panel_x + SCROLLBAR_X_OFFSET + BUTTON_OFFSET_X, panel_y + SCROLLBAR_Y_OFFSET-5, CharToString(0x35), up_color, 15, "Webdings");
         int down_y = panel_y + SCROLLBAR_Y_OFFSET + SCROLLBAR_HEIGHT - BUTTON_SIZE;
         createRecLabel(SCROLL_DOWN_REC, panel_x + SCROLLBAR_X_OFFSET + BUTTON_OFFSET_X, down_y, BUTTON_WIDTH, BUTTON_SIZE, clrDarkGray, 1, clrDarkGray);
         createLabel(SCROLL_DOWN_LABEL, panel_x + SCROLLBAR_X_OFFSET + BUTTON_OFFSET_X, down_y-5, CharToString(0x36), down_color, 15, "Webdings");
         slider_height = calculateSliderHeight();
         int slider_y = panel_y + SCROLLBAR_Y_OFFSET + BUTTON_SIZE;
         createButton(SCROLL_SLIDER, panel_x + SCROLLBAR_X_OFFSET + SLIDER_OFFSET_X, slider_y, SLIDER_WIDTH, slider_height, "", clrWhite, 12, clrLightSlateGray, clrDarkGray, "Arial Bold");
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
            ArrayResize(impact_filter_selected, ArraySize(impact_filter_selected) + 1);
            imp_filter_selected[ArraySize(imp_filter_selected) - 1] = selected_importance_lvl;
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
         createRecLabel(SCROLL_LEADER, panel_x + SCROLLBAR_X_OFFSET, panel_y + SCROLLBAR_Y_OFFSET, SCROLLBAR_WIDTH, SCROLLBAR_HEIGHT, clrSilver, 1, clrNONE);
         int max_scroll = MathMax(0, ArraySize(displayableEvents) - VISIBLE_ITEMS);
         color up_color = (scroll_pos == 0) ? clrLightGray : clrBlack;
         color down_color = (scroll_pos >= max_scroll) ? clrLightGray : clrBlack;
         createRecLabel(SCROLL_UP_REC, panel_x + SCROLLBAR_X_OFFSET + BUTTON_OFFSET_X, panel_y + SCROLLBAR_Y_OFFSET, BUTTON_WIDTH, BUTTON_SIZE, clrDarkGray, 1, clrDarkGray);
         createLabel(SCROLL_UP_LABEL, panel_x + SCROLLBAR_X_OFFSET + BUTTON_OFFSET_X, panel_y + SCROLLBAR_Y_OFFSET-5, CharToString(0x35), up_color, 15, "Webdings");
         int down_y = panel_y + SCROLLBAR_Y_OFFSET + SCROLLBAR_HEIGHT - BUTTON_SIZE;
         createRecLabel(SCROLL_DOWN_REC, panel_x + SCROLLBAR_X_OFFSET + BUTTON_OFFSET_X, down_y, BUTTON_WIDTH, BUTTON_SIZE, clrDarkGray, 1, clrDarkGray);
         createLabel(SCROLL_DOWN_LABEL, panel_x + SCROLLBAR_X_OFFSET + BUTTON_OFFSET_X, down_y-5, CharToString(0x36), down_color, 15, "Webdings");
         slider_height = calculateSliderHeight();
         int slider_y = panel_y + SCROLLBAR_Y_OFFSET + BUTTON_SIZE;
         createButton(SCROLL_SLIDER, panel_x + SCROLLBAR_X_OFFSET + SLIDER_OFFSET_X, slider_y, SLIDER_WIDTH, slider_height, "", clrWhite, 12, clrLightSlateGray, clrDarkGray, "Arial Bold");
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
```

To enhance the dynamic object click events when the event "id" is [CHARTEVENT\_OBJECT\_CLICK](https://www.mql5.com/en/docs/constants/chartconstants/enum_chartevents), we first call "UpdateFilterInfo" and "CheckForNewsTrade" to refresh filter states and check trading conditions. If "sparam" is "CANCEL\_BTN", we set "isDashboardUpdate" to false and call "destroy\_Dashboard" to close the dashboard. For "FILTER\_CURR\_BTN", "FILTER\_IMP\_BTN", or "FILTER\_TIME\_BTN", we use [ObjectGetInteger](https://www.mql5.com/en/docs/objects/ObjectGetInteger) to get "btn\_state" with [OBJPROP\_STATE](https://www.mql5.com/en/docs/constants/objectconstants/enum_object_property#enum_object_property_integer), update "enableCurrencyFilter", "enableImportanceFilter", or "enableTimeFilter", and adjust button text and color using "ObjectSetString" and [ObjectSetInteger](https://www.mql5.com/en/docs/objects/objectsetinteger) with "OBJPROP\_TEXT" and "OBJPROP\_COLOR" (e.g., "clrLime" for enabled, "clrRed" for disabled); we then call "update\_dashboard\_values" and rebuild the scrollbar using "createRecLabel", "createLabel", and "createButton" for elements like "SCROLL\_LEADER" and "SCROLL\_SLIDER", positioned with "panel\_x" and "panel\_y", and update its state with "updateSliderPosition" and "updateButtonColors".

For "CURRENCY\_BTNS", we use [ObjectGetString](https://www.mql5.com/en/docs/objects/objectgetstring) to get "selected\_curr", add or remove it from "curr\_filter\_selected" using [ArrayResize](https://www.mql5.com/en/docs/array/arrayresize), and update the dashboard similarly. For "IMPACT\_LABEL" buttons, we call "get\_importance\_level" to map "selected\_imp" to "selected\_importance\_lvl", manage "imp\_filter\_selected" and "impact\_filter\_selected" arrays, set "color\_border" with "ObjectSetInteger", and refresh the dashboard. If "sparam" is "SCROLL\_UP\_REC" or "SCROLL\_UP\_LABEL", we call "scrollUp", and for "SCROLL\_DOWN\_REC" or "SCROLL\_DOWN\_LABEL", we call "scrollDown", updating colors with "updateButtonColors" and redrawing the chart with [ChartRedraw](https://www.mql5.com/en/docs/chart_operations/ChartRedraw) to reflect changes. When we compile and run the program, we have the following results.

![HOVERABLE BUTTONS](https://c.mql5.com/2/144/PART_10_1.gif)

From the visualization, we can see that we can get visual hover feedback on defined buttons. What we now need to do is move on to the complex part which is the movement of the entire dashboard. We achieve that via the logic below.

```
else if (id == CHARTEVENT_MOUSE_MOVE && isDashboardUpdate) {
   // Handle panel dragging
   int header_x = (int)ObjectGetInteger(0, HEADER_LABEL, OBJPROP_XDISTANCE);
   int header_y = (int)ObjectGetInteger(0, HEADER_LABEL, OBJPROP_YDISTANCE);
   int header_width = 740;
   int header_height = 30;

   if (prev_mouse_state == 0 && mouse_state == 1) { // Mouse button down
      if (mouse_x >= header_x && mouse_x <= header_x + header_width &&
          mouse_y >= header_y && mouse_y <= header_y + header_height) {
         panel_dragging = true;
         panel_drag_x = mouse_x;
         panel_drag_y = mouse_y;
         panel_start_x = panel_x;
         panel_start_y = panel_y;
         ChartSetInteger(0, CHART_MOUSE_SCROLL, false);
         if (debugLogging) Print("Panel drag started at x=", mouse_x, ", y=", mouse_y);
      }
   }

   if (panel_dragging && mouse_state == 1) { // Dragging panel
      int dx = mouse_x - panel_drag_x;
      int dy = mouse_y - panel_drag_y;
      panel_x = panel_start_x + dx;
      panel_y = panel_start_y + dy;

      // Ensure panel stays within chart boundaries
      int chart_width = (int)ChartGetInteger(0, CHART_WIDTH_IN_PIXELS);
      int chart_height = (int)ChartGetInteger(0, CHART_HEIGHT_IN_PIXELS);
      panel_x = MathMax(0, MathMin(panel_x, chart_width - 753)); // 753 = panel width
      panel_y = MathMax(0, MathMin(panel_y, chart_height - 410)); // 410 = panel height

      // Update positions of all panel objects
      ObjectSetInteger(0, MAIN_REC, OBJPROP_XDISTANCE, panel_x);
      ObjectSetInteger(0, MAIN_REC, OBJPROP_YDISTANCE, panel_y);
      ObjectSetInteger(0, SUB_REC1, OBJPROP_XDISTANCE, panel_x + 3);
      ObjectSetInteger(0, SUB_REC1, OBJPROP_YDISTANCE, panel_y + 30);
      ObjectSetInteger(0, SUB_REC2, OBJPROP_XDISTANCE, panel_x + 3 + 5);
      ObjectSetInteger(0, SUB_REC2, OBJPROP_YDISTANCE, panel_y + 30 + 50 + 27);
      ObjectSetInteger(0, HEADER_LABEL, OBJPROP_XDISTANCE, panel_x + 3 + 5);
      ObjectSetInteger(0, HEADER_LABEL, OBJPROP_YDISTANCE, panel_y + 5);
      ObjectSetInteger(0, TIME_LABEL, OBJPROP_XDISTANCE, panel_x + 20);
      ObjectSetInteger(0, TIME_LABEL, OBJPROP_YDISTANCE, panel_y + 35);
      ObjectSetInteger(0, IMPACT_LABEL, OBJPROP_XDISTANCE, panel_x + 20);
      ObjectSetInteger(0, IMPACT_LABEL, OBJPROP_YDISTANCE, panel_y + 55);
      ObjectSetInteger(0, FILTER_LABEL, OBJPROP_XDISTANCE, panel_x + 320);
      ObjectSetInteger(0, FILTER_LABEL, OBJPROP_YDISTANCE, panel_y + 5);
      ObjectSetInteger(0, FILTER_CURR_BTN, OBJPROP_XDISTANCE, panel_x + 380);
      ObjectSetInteger(0, FILTER_CURR_BTN, OBJPROP_YDISTANCE, panel_y + 5);
      ObjectSetInteger(0, FILTER_IMP_BTN, OBJPROP_XDISTANCE, panel_x + 490);
      ObjectSetInteger(0, FILTER_IMP_BTN, OBJPROP_YDISTANCE, panel_y + 5);
      ObjectSetInteger(0, FILTER_TIME_BTN, OBJPROP_XDISTANCE, panel_x + 610);
      ObjectSetInteger(0, FILTER_TIME_BTN, OBJPROP_YDISTANCE, panel_y + 5);
      ObjectSetInteger(0, CANCEL_BTN, OBJPROP_XDISTANCE, panel_x + 692+10);
      ObjectSetInteger(0, CANCEL_BTN, OBJPROP_YDISTANCE, panel_y + 1);

      // Update calendar buttons
      int startX = panel_x + 9;
      for (int i = 0; i < ArraySize(array_calendar); i++) {
         ObjectSetInteger(0, ARRAY_CALENDAR+IntegerToString(i), OBJPROP_XDISTANCE, startX);
         ObjectSetInteger(0, ARRAY_CALENDAR+IntegerToString(i), OBJPROP_YDISTANCE, panel_y + 82);
         startX += buttons[i] + 3;
      }

      // Update impact buttons
      for (int i = 0; i < ArraySize(impact_labels); i++) {
         ObjectSetInteger(0, IMPACT_LABEL+string(i), OBJPROP_XDISTANCE, panel_x + 90 + 100 * i);
         ObjectSetInteger(0, IMPACT_LABEL+string(i), OBJPROP_YDISTANCE, panel_y + 55);
      }

      // Update currency buttons
      int curr_size = 51, button_height = 22, spacing_x = 0, spacing_y = 3, max_columns = 4;
      for (int i = 0; i < ArraySize(curr_filter); i++) {
         int row = i / max_columns;
         int col = i % max_columns;
         int x_pos = panel_x + 525 + col * (curr_size + spacing_x);
         int y_pos = panel_y + 33 + row * (button_height + spacing_y);
         ObjectSetInteger(0, CURRENCY_BTNS+IntegerToString(i), OBJPROP_XDISTANCE, x_pos);
         ObjectSetInteger(0, CURRENCY_BTNS+IntegerToString(i), OBJPROP_YDISTANCE, y_pos);
      }

      // Update scrollbar
      if (scroll_visible) {
         ObjectSetInteger(0, SCROLL_LEADER, OBJPROP_XDISTANCE, panel_x + SCROLLBAR_X_OFFSET);
         ObjectSetInteger(0, SCROLL_LEADER, OBJPROP_YDISTANCE, panel_y + SCROLLBAR_Y_OFFSET);
         ObjectSetInteger(0, SCROLL_UP_REC, OBJPROP_XDISTANCE, panel_x + SCROLLBAR_X_OFFSET + BUTTON_OFFSET_X);
         ObjectSetInteger(0, SCROLL_UP_REC, OBJPROP_YDISTANCE, panel_y + SCROLLBAR_Y_OFFSET);
         ObjectSetInteger(0, SCROLL_UP_LABEL, OBJPROP_XDISTANCE, panel_x + SCROLLBAR_X_OFFSET + BUTTON_OFFSET_X);
         ObjectSetInteger(0, SCROLL_UP_LABEL, OBJPROP_YDISTANCE, panel_y + SCROLLBAR_Y_OFFSET - 5);
         ObjectSetInteger(0, SCROLL_DOWN_REC, OBJPROP_XDISTANCE, panel_x + SCROLLBAR_X_OFFSET + BUTTON_OFFSET_X);
         ObjectSetInteger(0, SCROLL_DOWN_REC, OBJPROP_YDISTANCE, panel_y + SCROLLBAR_Y_OFFSET + SCROLLBAR_HEIGHT - BUTTON_SIZE);
         ObjectSetInteger(0, SCROLL_DOWN_LABEL, OBJPROP_XDISTANCE, panel_x + SCROLLBAR_X_OFFSET + BUTTON_OFFSET_X);
         ObjectSetInteger(0, SCROLL_DOWN_LABEL, OBJPROP_YDISTANCE, panel_y + SCROLLBAR_Y_OFFSET + SCROLLBAR_HEIGHT - BUTTON_SIZE - 5);
         ObjectSetInteger(0, SCROLL_SLIDER, OBJPROP_XDISTANCE, panel_x + SCROLLBAR_X_OFFSET + SLIDER_OFFSET_X);
         // Y-position of slider is managed by updateSliderPosition()
      }

      // Update event display
      update_dashboard_values(curr_filter_selected, imp_filter_selected);

      // Update trade labels if they exist
      if (ObjectFind(0, "NewsCountdown") >= 0) {
         ObjectSetInteger(0, "NewsCountdown", OBJPROP_XDISTANCE, panel_x);
         ObjectSetInteger(0, "NewsCountdown", OBJPROP_YDISTANCE, panel_y - 33);
      }
      if (ObjectFind(0, "NewsTradeInfo") >= 0) {
         ObjectSetInteger(0, "NewsTradeInfo", OBJPROP_XDISTANCE, panel_x + 305);
         ObjectSetInteger(0, "NewsTradeInfo", OBJPROP_YDISTANCE, panel_y - 28);
      }

      ChartRedraw(0);
      if (debugLogging) Print("Panel moved to x=", panel_x, ", y=", panel_y);
   }

   if (mouse_state == 0) { // Mouse button released
      if (panel_dragging) {
         panel_dragging = false;
         ChartSetInteger(0, CHART_MOUSE_SCROLL, true);
         if (debugLogging) Print("Panel drag stopped.");
         ChartRedraw(0);
      }
   }

}
```

Here, we enhance the [OnChartEvent](https://www.mql5.com/en/docs/event_handlers/onchartevent) function to implement panel dragging functionality, allowing us to reposition the interface when "isDashboardUpdate" is true and the event "id" is [CHARTEVENT\_MOUSE\_MOVE](https://www.mql5.com/en/docs/constants/chartconstants/enum_chartevents). We retrieve the header’s position using [ObjectGetInteger](https://www.mql5.com/en/docs/objects/ObjectGetInteger) to get "header\_x" and "header\_y" from "HEADER\_LABEL" with "OBJPROP\_XDISTANCE" and "OBJPROP\_YDISTANCE", defining a 740x30 pixel area. When "prev\_mouse\_state" is 0 and "mouse\_state" is 1 (mouse button pressed), we check if "mouse\_x" and "mouse\_y" are within the header, setting "panel\_dragging" to true, storing "panel\_drag\_x", "panel\_drag\_y", "panel\_start\_x", and "panel\_start\_y", and disabling chart scrolling with [ChartSetInteger](https://www.mql5.com/en/docs/chart_operations/chartsetinteger) and [CHART\_MOUSE\_SCROLL](https://www.mql5.com/en/docs/constants/chartconstants/enum_chart_property#enum_chart_property_integer).

If "panel\_dragging" is true and "mouse\_state" is 1, we calculate the drag offset "dx" and "dy", update "panel\_x" and "panel\_y", and use [MathMax](https://www.mql5.com/en/docs/math/mathmax) and [MathMin](https://www.mql5.com/en/docs/math/mathmin) to constrain them within the chart’s "chart\_width" and "chart\_height" (obtained via ChartGetInteger with "CHART\_WIDTH\_IN\_PIXELS" and [CHART\_HEIGHT\_IN\_PIXELS](https://www.mql5.com/en/docs/constants/chartconstants/enum_chart_property#enum_chart_property_integer)). We then reposition all UI elements using [ObjectSetInteger](https://www.mql5.com/en/docs/objects/objectsetinteger) with "OBJPROP\_XDISTANCE" and "OBJPROP\_YDISTANCE" for objects like "MAIN\_REC", "SUB\_REC1", "SUB\_REC2", "HEADER\_LABEL", "TIME\_LABEL", "IMPACT\_LABEL", "FILTER\_LABEL", "FILTER\_CURR\_BTN", "FILTER\_IMP\_BTN", "FILTER\_TIME\_BTN", "CANCEL\_BTN", "ARRAY\_CALENDAR", "IMPACT\_LABEL", and "CURRENCY\_BTNS", and update scrollbar elements ("SCROLL\_LEADER", "SCROLL\_UP\_REC", "SCROLL\_UP\_LABEL", "SCROLL\_DOWN\_REC", "SCROLL\_DOWN\_LABEL", "SCROLL\_SLIDER") if "scroll\_visible" is true, using "panel\_x" and "panel\_y" with offsets.

We call "update\_dashboard\_values" to refresh news events, reposition trade labels "NewsCountdown" and "NewsTradeInfo" if present using [ObjectFind](https://www.mql5.com/en/docs/objects/ObjectFind), and redraw the chart with [ChartRedraw](https://www.mql5.com/en/docs/chart_operations/ChartRedraw). When "mouse\_state" is 0 (mouse released), we set "panel\_dragging" to false, re-enable scrolling with ChartSetInteger, and redraw the chart, logging drag events with [Print](https://www.mql5.com/en/docs/common/print) if "debugLogging" is enabled. Then, we can handle the scrollbar feature, provided the panel is not in drag mode, to prevent event conflicts.

```
// Scrollbar handling (only if not dragging panel)
if (scroll_visible && !panel_dragging) {
   if (prev_mouse_state == 0 && mouse_state == 1) {
      int xd = (int)ObjectGetInteger(0, SCROLL_SLIDER, OBJPROP_XDISTANCE);
      int yd = (int)ObjectGetInteger(0, SCROLL_SLIDER, OBJPROP_YDISTANCE);
      int xs = (int)ObjectGetInteger(0, SCROLL_SLIDER, OBJPROP_XSIZE);
      int ys = (int)ObjectGetInteger(0, SCROLL_SLIDER, OBJPROP_YSIZE);
      // Skip if mouse is over header to prioritize panel dragging
      if (mouse_x >= (panel_x + 3 + 5) && mouse_x <= (panel_x + 3 + 5 + 740) &&
          mouse_y >= (panel_y + 5) && mouse_y <= (panel_y + 5 + 30)) {
          return;
      }
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
      int scroll_area_y_min = panel_y + SCROLLBAR_Y_OFFSET + BUTTON_SIZE;
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
}
prev_mouse_state = mouse_state;
```

Here, we implement scrollbar dragging functionality within, enabling us to scroll through news events when "scroll\_visible" is true and "panel\_dragging" is false, ensuring no conflict with panel dragging. When "prev\_mouse\_state" is 0 and "mouse\_state" is 1 (mouse button pressed), we apply the scrolling logic that we had already done, as well as the other slider and release states. Upon compilation, we have the following output.

![MOVEMENT INTERACTION](https://c.mql5.com/2/144/PART_10_2.gif)

From the visualization, we can see that we can hover among the buttons and drag the panel within the proximity of the chart extremums. What now remains is backtesting the system thoroughly, and that is handled in the next section.

### Testing and Validation

We tested the enhanced dashboard to confirm that the dynamic movement, hoverable buttons, and scrollbar function as intended, providing a seamless experience for navigating news events. Our test focused on the entire dashboard's button visual feedback on hover, the movement of the entire components of the panel within the vicinity of the chart, and the integrability of the scrollbar in live mode. We captured these tests in a concise [Graphics Interchange Format](https://en.wikipedia.org/wiki/GIF "https://en.wikipedia.org/wiki/GIF") (GIF) to visually demonstrate the dashboard’s performance as shown below.

![BACKTEST GIF](https://c.mql5.com/2/144/PART_10_3.gif)

From the visualization, we can see that the dashboard works seamlessly.

### Conclusion

In conclusion, we’ve elevated the [MQL5 Economic Calendar](https://www.mql5.com/en/economic-calendar) series by introducing a draggable dashboard and interactive hover effects, empowering us with flexible positioning and intuitive navigation of news events. These enhancements, built on [Part 9’s dynamic scrollbar and polished display](https://www.mql5.com/en/articles/18135), ensure seamless interaction in both live and Strategy Tester modes, providing a robust and adaptable platform for news-driven trading strategies. You can now leverage this versatile dashboard as a foundation, customizing it to meet your unique chart and trading needs.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/18241.zip "Download all attachments in the single ZIP archive")

[MQL5\_NEWS\_CALENDAR\_PART\_10.mq5](https://www.mql5.com/en/articles/download/18241/mql5_news_calendar_part_10.mq5 "Download MQL5_NEWS_CALENDAR_PART_10.mq5")(99.83 KB)

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

**[Go to discussion](https://www.mql5.com/en/forum/487492)**

![From Basic to Intermediate: Array (II)](https://c.mql5.com/2/98/Do_btsico_ao_intermediario_Array_II___LOGO3.png)[From Basic to Intermediate: Array (II)](https://www.mql5.com/en/articles/15472)

In this article, we will look at what a dynamic array and a static array are. Is there a difference between using one or the other? Or are they always the same? When should you use one and when the other type? And what about constant arrays? We will try to understand what they are designed for and consider the risks of not initializing all the values in the array.

![Building MQL5-Like Trade Classes in Python for MetaTrader 5](https://c.mql5.com/2/144/18208-building-mql5-like-trade-classes-logo.png)[Building MQL5-Like Trade Classes in Python for MetaTrader 5](https://www.mql5.com/en/articles/18208)

MetaTrader 5 python package provides an easy way to build trading applications for the MetaTrader 5 platform in the Python language, while being a powerful and useful tool, this module isn't as easy as MQL5 programming language when it comes to making an algorithmic trading solution. In this article, we are going to build trade classes similar to the one offered in MQL5 to create a similar syntax and make it easier to make trading robots in Python as in MQL5.

![Developing a Replay System (Part 70): Getting the Time Right (III)](https://c.mql5.com/2/98/Desenvolvendo_um_sistema_de_Replay_Parte_69___LOGO.png)[Developing a Replay System (Part 70): Getting the Time Right (III)](https://www.mql5.com/en/articles/12326)

In this article, we will look at how to use the CustomBookAdd function correctly and effectively. Despite its apparent simplicity, it has many nuances. For example, it allows you to tell the mouse indicator whether a custom symbol is on auction, being traded, or the market is closed. The content presented here is intended solely for educational purposes. Under no circumstances should the application be viewed for any purpose other than to learn and master the concepts presented.

![Neural Networks in Trading: Transformer with Relative Encoding](https://c.mql5.com/2/97/Neural_Networks_in_Trading_Transformer_with_Relative_Encoding_____LOGO.png)[Neural Networks in Trading: Transformer with Relative Encoding](https://www.mql5.com/en/articles/16097)

Self-supervised learning can be an effective way to analyze large amounts of unlabeled data. The efficiency is provided by the adaptation of models to the specific features of financial markets, which helps improve the effectiveness of traditional methods. This article introduces an alternative attention mechanism that takes into account the relative dependencies and relationships between inputs.

[Need a reliable hosting solution for your robots?Contact your broker and find out about available Sponsored MetaTrader VPS offeringsLearn more![](https://www.mql5.com/ff/sh/0pw0dk81s56qy774z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=vljwvezfjkfbvviocwskggexlvgykvob&s=70cf8e354b9a125332533ffb65d7365abe8dde5b5c1ede9caac479a9e9df4f25&uid=&ref=https://www.mql5.com/en/articles/18241&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5083439467018066875)

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