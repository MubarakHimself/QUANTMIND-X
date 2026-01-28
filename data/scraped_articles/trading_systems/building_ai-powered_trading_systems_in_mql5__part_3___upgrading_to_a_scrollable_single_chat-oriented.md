---
title: Building AI-Powered Trading Systems in MQL5 (Part 3): Upgrading to a Scrollable Single Chat-Oriented UI
url: https://www.mql5.com/en/articles/19741
categories: Trading Systems, Integration, Expert Advisors
relevance_score: 9
scraped_at: 2026-01-22T17:32:39.982954
---

[![](https://www.mql5.com/ff/sh/20jc81m23z78s5z9z2/01.png)![](https://www.mql5.com/ff/sh/20jc81m23z78s5z9z2/02.png)Create your own AI for tradingRead our book "Neural Networks in Algo Trading with MQL5"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/neurobook%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.neurobook%26utm_content=visit.page%26utm_campaign=neurobook.promo.04.2024&a=elbyupbppbqpzzvzhxtydvlupfcbmnmb&s=0d2f8feb92df3772a11aca1f195d2996b59d6539e283cdf4a18ccff02e5ad43d&uid=&ref=https://www.mql5.com/en/articles/19741&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5049205979213571845)

MetaTrader 5 / Trading systems


### Introduction

In our [previous article (Part 2)](https://www.mql5.com/en/articles/19567), we built an interactive ChatGPT-powered program with a User Interface (UI) in [MetaQuotes Language 5](https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5 "https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5") (MQL5). The tool enabled us to send prompts to [OpenAI’s](https://www.mql5.com/go?link=https://openai.com/ "https://openai.com/") API and instantly view responses directly on the chart. Now, in Part 3, we're taking it to the next level: get ready for a scrollable, chat-style dashboard, complete with timestamps, smooth dynamic scrolling, and rich conversation history for engaging, multi-turn chats. Here’s what we’ll dive into:

1. [Understanding the Upgraded ChatGPT Program Framework](https://www.mql5.com/en/articles/19741#para1)
2. [Implementation in MQL5](https://www.mql5.com/en/articles/19741#para2)
3. [Testing the ChatGPT Program](https://www.mql5.com/en/articles/19741#para3)
4. [Conclusion](https://www.mql5.com/en/articles/19741#para4)

By the end, you’ll have an enhanced MQL5 program for interactive AI queries, ready for customization—let’s dive in!

### Understanding the Upgraded ChatGPT Program Framework

The upgraded ChatGPT program framework enhances our AI-driven trading interface by incorporating a scrollable, chat-oriented UI that supports multi-turn conversations, timestamps, and dynamic message handling, allowing us to maintain context for queries across sessions. Its role is to provide a seamless conversational experience, improving usability by enabling us to review history and build on previous AI responses, which is vital for refining trading strategies without losing insights from prior interactions. We figured that if we stick to one conversation that the AI can reference, we can always go back and refine prompts and make corrections when needed.

Our approach is to build a single-chat-focused dashboard with scrollable text, hover effects, and message building for [API](https://en.wikipedia.org/wiki/API "https://en.wikipedia.org/wiki/API") requests, ensuring the interface adapts to conversation length and user preferences for scrollbar visibility. We will implement logic to parse history for multi-turn queries, add timestamps for clarity, and enable features like clearing conversations or starting new chats, creating a tool that facilitates ongoing AI assistance for trading decisions with a new upgraded interface. Have a look at what we will be achieving.

![UPGRADED UI ROADMAP](https://c.mql5.com/2/172/Screenshot_2025-09-27_160857.png)

### Implementation in MQL5

To implement the upgraded program in MQL5, we will first alter the [inputs](https://www.mql5.com/en/docs/basis/variables/inputvariables) section to incorporate a new input for the scrollbar mode so that we can either reveal the scrollbar when we need to hover or have it always there, visible. We will add an [enumeration](https://www.mql5.com/en/docs/basis/types/integer/enumeration) for that and also increase the response tokens limit to 3000, since we can now have a rich conversation that feels just sufficient, but you can increase it if needed.

```
//+------------------------------------------------------------------+
//|                                         ChatGPT AI EA Part 3.mq5 |
//|                           Copyright 2025, Allan Munene Mutiiria. |
//|                                   https://t.me/Forex_Algo_Trader |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, Allan Munene Mutiiria."
#property link "https://t.me/Forex_Algo_Trader"
#property version "1.00"
#property strict

//--- Input parameters
enum ENUM_SCROLLBAR_MODE
{
   SCROLL_DYNAMIC_ALWAYS, // Show when needed
   SCROLL_DYNAMIC_HOVER,  // Show on hover when needed
   SCROLL_WHEEL_ONLY      // No scrollbar, wheel scroll only
};
input ENUM_SCROLLBAR_MODE ScrollbarMode = SCROLL_DYNAMIC_HOVER;            // Scrollbar Behavior
//--- Scrollbar object names
#define SCROLL_LEADER "ChatGPT_Scroll_Leader"
#define SCROLL_UP_REC "ChatGPT_Scroll_Up_Rec"
#define SCROLL_UP_LABEL "ChatGPT_Scroll_Up_Label"
#define SCROLL_DOWN_REC "ChatGPT_Scroll_Down_Rec"
#define SCROLL_DOWN_LABEL "ChatGPT_Scroll_Down_Label"
#define SCROLL_SLIDER "ChatGPT_Scroll_Slider"

//--- Input parameters
input string OpenAI_Model = "gpt-3.5-turbo";                                 // OpenAI Model
input string OpenAI_Endpoint = "https://api.openai.com/v1/chat/completions"; // OpenAI API Endpoint
input int MaxResponseLength = 3000;                                          // Max length of ChatGPT response to display
input string LogFileName = "ChatGPT_EA_Log.txt";                             // Log file name
```

We begin the upgrade implementation by defining the configuration parameters and constants to control scrolling behavior and API settings. First, we create the "ENUM\_SCROLLBAR\_MODE" [enum](https://www.mql5.com/en/docs/basis/types/integer/enumeration) with options "SCROLL\_DYNAMIC\_ALWAYS" (show scrollbar when needed), "SCROLL\_DYNAMIC\_HOVER" (show on hover when needed), and "SCROLL\_WHEEL\_ONLY" (no scrollbar, wheel scroll only), setting the input "ScrollbarMode" to "SCROLL\_DYNAMIC\_HOVER" for user preference.

Then, we [define](https://www.mql5.com/en/docs/basis/preprosessor/constant) constants for scrollbar object names like "SCROLL\_LEADER", "SCROLL\_UP\_REC", "SCROLL\_UP\_LABEL", "SCROLL\_DOWN\_REC", "SCROLL\_DOWN\_LABEL", and "SCROLL\_SLIDER" for consistent referencing in the UI. Next, we set the [input](https://www.mql5.com/en/docs/basis/variables/inputvariables) parameter "MaxResponseLength" to 3000 to limit the displayed response text, ensuring we can get longer conversations. The next thing that we need to do is alter the [JSON](https://en.wikipedia.org/wiki/JSON "https://en.wikipedia.org/wiki/JSON") class so that it can handle double values as well, just a minor enhancement.

```
bool DeserializeFromArray(char &jsonCharacterArray[], int arrayLength, int &currentIndex) { //--- Deserialize from array
   string validNumericCharacters = "0123456789+-.eE";     //--- Valid number chars
   int startPosition = currentIndex;                      //--- Start position
   for(; currentIndex < arrayLength; currentIndex++) {    //--- Loop array
      char currentCharacter = jsonCharacterArray[currentIndex]; //--- Current char
      if(currentCharacter == 0) break;                    //--- Break on null
      switch(currentCharacter) {                          //--- Switch on char
         case '\t': case '\r': case '\n': case ' ': startPosition = currentIndex + 1; break; //--- Skip whitespace
         case '[': {                                      //--- Array start\
            startPosition = currentIndex + 1;             //--- Update start\
            if(m_type != JsonUndefined) return false;     //--- Type check\
            m_type = JsonArray;                           //--- Set array\
            currentIndex++;                               //--- Increment\
            JsonValue childValue(GetPointer(this), JsonUndefined); //--- Child value\
            while(childValue.DeserializeFromArray(jsonCharacterArray, arrayLength, currentIndex)) { //--- Loop children\
               if(childValue.m_type != JsonUndefined) AddChild(childValue); //--- Add if defined\
               if(childValue.m_type == JsonInteger || childValue.m_type == JsonDouble || childValue.m_type == JsonArray) currentIndex++; //--- Adjust index\
               childValue.Reset();                        //--- Reset child\
               childValue.m_parent = GetPointer(this);    //--- Set parent\
               if(jsonCharacterArray[currentIndex] == ']') break; //--- End array
               currentIndex++;                            //--- Increment
               if(currentIndex >= arrayLength) return false; //--- Bounds check
            }
            return (jsonCharacterArray[currentIndex] == ']' || jsonCharacterArray[currentIndex] == 0); //--- Valid end
         }                                                //--- End array case
         case ']': return (m_parent && m_parent.m_type == JsonArray); //--- Array end
         case ':': {                                      //--- Key separator
            if(m_temporaryKey == "") return false;        //--- Key check
            JsonValue childValue(GetPointer(this), JsonUndefined); //--- New child
            JsonValue *addedChild = AddChild(childValue); //--- Add
            addedChild.m_key = m_temporaryKey;            //--- Set key
            m_temporaryKey = "";                          //--- Clear temp
            currentIndex++;                               //--- Increment
            if(!addedChild.DeserializeFromArray(jsonCharacterArray, arrayLength, currentIndex)) return false; //--- Recurse
         } break;                                         //--- End key case
         case ',': {                                      //--- Value separator
            startPosition = currentIndex + 1;             //--- Update start
            if(!m_parent && m_type != JsonObject) return false; //--- Check context
            if(m_parent && m_parent.m_type != JsonArray && m_parent.m_type != JsonObject) return false; //--- Parent type
            if(m_parent && m_parent.m_type == JsonArray && m_type == JsonUndefined) return true; //--- Undefined in array
         } break;                                         //--- End separator
         case '{': {                                      //--- Object start
            startPosition = currentIndex + 1;             //--- Update start
            if(m_type != JsonUndefined) return false;     //--- Type check
            m_type = JsonObject;                          //--- Set object
            currentIndex++;                               //--- Increment
            if(!DeserializeFromArray(jsonCharacterArray, arrayLength, currentIndex)) return false; //--- Recurse
            return (jsonCharacterArray[currentIndex] == '}' || jsonCharacterArray[currentIndex] == 0); //--- Valid end
         } break;                                         //--- End object case
         case '}': return (m_type == JsonObject);         //--- Object end
         case 't': case 'T': case 'f': case 'F': {        //--- Boolean start
            if(m_type != JsonUndefined) return false;     //--- Type check
            m_type = JsonBoolean;                         //--- Set boolean
            if(currentIndex + 3 < arrayLength && StringCompare(GetSubstringFromArray(jsonCharacterArray, currentIndex, 4), "true", false) == 0) { //--- True check
               m_booleanValue = true; currentIndex += 3; return true; //--- Set true
            }
            if(currentIndex + 4 < arrayLength && StringCompare(GetSubstringFromArray(jsonCharacterArray, currentIndex, 5), "false", false) == 0) { //--- False check
               m_booleanValue = false; currentIndex += 4; return true; //--- Set false
            }
            return false;                                 //--- Invalid boolean
         } break;                                         //--- End boolean
         case 'n': case 'N': {                            //--- Null start
            if(m_type != JsonUndefined) return false;     //--- Type check
            m_type = JsonNull;                            //--- Set null
            if(currentIndex + 3 < arrayLength && StringCompare(GetSubstringFromArray(jsonCharacterArray, currentIndex, 4), "null", false) == 0) { //--- Null check
               currentIndex += 3; return true;            //--- Valid null
            }
            return false;                                 //--- Invalid null
         } break;                                         //--- End null
         case '0': case '1': case '2': case '3': case '4':
         case '5': case '6': case '7': case '8': case '9':
         case '-': case '+': case '.': {                  //--- Number start
            if(m_type != JsonUndefined) return false;     //--- Type check
            bool isDouble = false;                        //--- Double flag
            int startOfNumber = currentIndex;             //--- Number start
            while(jsonCharacterArray[currentIndex] != 0 && currentIndex < arrayLength) { //--- Parse number
               currentIndex++;                            //--- Increment
               if(StringFind(validNumericCharacters, GetSubstringFromArray(jsonCharacterArray, currentIndex, 1)) < 0) break; //--- Invalid char
               if(!isDouble) isDouble = (jsonCharacterArray[currentIndex] == '.' || jsonCharacterArray[currentIndex] == 'e' || jsonCharacterArray[currentIndex] == 'E'); //--- Set double
            }
            m_stringValue = GetSubstringFromArray(jsonCharacterArray, startOfNumber, currentIndex - startOfNumber); //--- Get string
            if(isDouble) {                                //--- Double handling
               m_type = JsonDouble;                       //--- Set type
               m_doubleValue = StringToDouble(m_stringValue); //--- Convert double
               m_integerValue = (long)m_doubleValue;      //--- Set integer
               m_booleanValue = m_integerValue != 0;      //--- Set boolean
            } else {                                      //--- Integer handling
               m_type = JsonInteger;                      //--- Set type
               m_integerValue = StringToInteger(m_stringValue); //--- Convert integer
               m_doubleValue = (double)m_integerValue;    //--- Set double
               m_booleanValue = m_integerValue != 0;      //--- Set boolean
            }
            currentIndex--;                               //--- Adjust index
            return true;                                  //--- Success
         } break;                                         //--- End number
         case '\"': {                                     //--- String or key start
            if(m_type == JsonObject) {                    //--- Key in object
               currentIndex++;                            //--- Increment
               int startOfString = currentIndex;          //--- String start
               if(!ExtractStringFromArray(jsonCharacterArray, arrayLength, currentIndex)) return false; //--- Extract
               m_temporaryKey = GetSubstringFromArray(jsonCharacterArray, startOfString, currentIndex - startOfString); //--- Set temp key
            } else {                                      //--- Value string
               if(m_type != JsonUndefined) return false;  //--- Type check
               m_type = JsonString;                       //--- Set string
               currentIndex++;                            //--- Increment
               int startOfString = currentIndex;          //--- String start
               if(!ExtractStringFromArray(jsonCharacterArray, arrayLength, currentIndex)) return false; //--- Extract
               SetFromString(JsonString, GetSubstringFromArray(jsonCharacterArray, startOfString, currentIndex - startOfString)); //--- Set value
               return true;                               //--- Success
            }
         } break;                                         //--- End string
      }
   }
   return true;                                           //--- Default success
}
```

In the deserialization function, we just handle the double types that we had not considered in the previous versions. We have highlighted the specific section for clarity. We now need to add new [global variables](https://www.mql5.com/en/docs/basis/variables/global) for the UI layout, scrolling, hover states, and colors to support a more complex dashboard with a header, footer, buttons, and a scrollbar.

```
bool clear_hover = false;
bool new_chat_hover = false;

color clear_original_bg = clrLightCoral;
color clear_darker_bg;
color new_chat_original_bg = clrLightBlue;
color new_chat_darker_bg;
int g_mainX = 10;
int g_mainY = 30;
int g_mainWidth = 550;
int g_mainHeight = 0;
int g_padding = 10;
int g_sidePadding = 6;
int g_textPadding = 10;
int g_headerHeight = 40;
int g_displayHeight = 280;
int g_footerHeight = 50;
int g_lineSpacing = 2;
bool scroll_visible = false;
bool mouse_in_display = false;
int scroll_pos = 0;
int prev_scroll_pos = -1;
int slider_height = 20;
bool movingStateSlider = false;
int mlbDownX_Slider = 0;
int mlbDownY_Slider = 0;
int mlbDown_YD_Slider = 0;
int g_total_height = 0;
int g_visible_height = 0;
```

Here, we initialize some more [global variables](https://www.mql5.com/en/docs/basis/variables/global) and color schemes for the upgraded program to support dynamic hover effects and layout management. We set hover flags "clear\_hover" and "new\_chat\_hover" to false for the clear and new chat buttons, and define original backgrounds "clear\_original\_bg" as "clrLightCoral" and "new\_chat\_original\_bg" as " [clrLightBlue](https://www.mql5.com/en/docs/constants/objectconstants/webcolors)", with darker versions "clear\_darker\_bg" and "new\_chat\_darker\_bg" for hover states. Then, we configure dashboard dimensions: "g\_mainX" at 10, "g\_mainY" at 30, "g\_mainWidth" at 550, "g\_mainHeight" at 0 (to be calculated), padding values like "g\_padding" at 10, "g\_sidePadding" at 6, "g\_textPadding" at 10, heights for header ("g\_headerHeight" at 40), display ("g\_displayHeight" at 280), footer ("g\_footerHeight" at 50), and line spacing ("g\_lineSpacing" at 2).

Finally, we initialize scrolling variables: "scroll\_visible" and "mouse\_in\_display" to false, "scroll\_pos" and "prev\_scroll\_pos" to 0 and -1, "slider\_height" at 20, drag states "movingStateSlider" to false and positions "mlbDownX\_Slider", "mlbDownY\_Slider", "mlbDown\_YD\_Slider" to 0, and height trackers "g\_total\_height" and "g\_visible\_height" to 0. Then, we need to define the scrollbar before updating the display since it will be dynamic. So let us define the scrollbar functions.

```
//+------------------------------------------------------------------+
//| Calculate font size based on screen DPI                          |
//+------------------------------------------------------------------+
int getFontSizeByDPI(int baseFontSize, int baseDPI = 96) {
   int currentDPI = (int)TerminalInfoInteger(TERMINAL_SCREEN_DPI);          //--- Retrieve current screen DPI
   int scaledFontSize = (int)(baseFontSize * (double)baseDPI / currentDPI); //--- Calculate scaled font size
   return MathMax(scaledFontSize, 8);                                       //--- Ensure minimum font size of 8
}

//+------------------------------------------------------------------+
//| Create scrollbar elements                                        |
//+------------------------------------------------------------------+
void CreateScrollbar() {
   int displayX = g_mainX + g_sidePadding;          //--- Calculate display x position
   int displayY = g_mainY + g_headerHeight + g_padding; //--- Calculate display y position
   int displayW = g_mainWidth - 2 * g_sidePadding;  //--- Calculate display width
   int scrollbar_x = displayX + displayW - 16;      //--- Set scrollbar x position
   int scrollbar_y = displayY + 16;                 //--- Set scrollbar y position
   int scrollbar_width = 16;                        //--- Set scrollbar width
   int scrollbar_height = g_displayHeight - 2 * 16; //--- Calculate scrollbar height
   int button_size = 16;                            //--- Set button size
   if (!createRecLabel(SCROLL_LEADER, scrollbar_x, scrollbar_y, scrollbar_width, scrollbar_height, C'220,220,220', 1, clrGainsboro, BORDER_FLAT, STYLE_SOLID, CORNER_LEFT_UPPER)) { //--- Create scrollbar leader
      FileWriteString(logFileHandle, "Failed to create scrollbar leader\n"); //--- Log failure
   }
   if (!createRecLabel(SCROLL_UP_REC, scrollbar_x, displayY, scrollbar_width, button_size, clrGainsboro, 1, clrGainsboro, BORDER_FLAT, STYLE_SOLID, CORNER_LEFT_UPPER)) { //--- Create scroll up button
      FileWriteString(logFileHandle, "Failed to create scrollbar up button\n"); //--- Log failure
   }
   if (!createLabel(SCROLL_UP_LABEL, scrollbar_x + 2, displayY + -2, CharToString(0x35), clrDimGray, getFontSizeByDPI(10), "Webdings", CORNER_LEFT_UPPER)) { //--- Create scroll up label
      FileWriteString(logFileHandle, "Failed to create scrollbar up label\n"); //--- Log failure
   }
   if (!createRecLabel(SCROLL_DOWN_REC, scrollbar_x, displayY + g_displayHeight - button_size, scrollbar_width, button_size, clrGainsboro, 1, clrGainsboro, BORDER_FLAT, STYLE_SOLID, CORNER_LEFT_UPPER)) { //--- Create scroll down button
      FileWriteString(logFileHandle, "Failed to create scrollbar down button\n"); //--- Log failure
   }
   if (!createLabel(SCROLL_DOWN_LABEL, scrollbar_x + 2, displayY + g_displayHeight - button_size + -2, CharToString(0x36), clrDimGray, getFontSizeByDPI(10), "Webdings", CORNER_LEFT_UPPER)) { //--- Create scroll down label
      FileWriteString(logFileHandle, "Failed to create scrollbar down label\n"); //--- Log failure
   }
   slider_height = CalculateSliderHeight();         //--- Calculate slider height
   if (!createRecLabel(SCROLL_SLIDER, scrollbar_x, displayY + g_displayHeight - button_size - slider_height, scrollbar_width, slider_height, clrSilver, 1, clrGainsboro, BORDER_FLAT, STYLE_SOLID, CORNER_LEFT_UPPER)) { //--- Create scrollbar slider
      FileWriteString(logFileHandle, "Failed to create scrollbar slider\n"); //--- Log failure
   }
   FileWriteString(logFileHandle, "Scrollbar created: x=" + IntegerToString(scrollbar_x) + ", y=" + IntegerToString(scrollbar_y) + ", height=" + IntegerToString(scrollbar_height) + ", slider_height=" + IntegerToString(slider_height) + "\n"); //--- Log scrollbar creation
}

//+------------------------------------------------------------------+
//| Delete scrollbar elements                                        |
//+------------------------------------------------------------------+
void DeleteScrollbar() {
   ObjectDelete(0, SCROLL_LEADER);                  //--- Remove scrollbar leader
   ObjectDelete(0, SCROLL_UP_REC);                  //--- Remove scroll up rectangle
   ObjectDelete(0, SCROLL_UP_LABEL);                //--- Remove scroll up label
   ObjectDelete(0, SCROLL_DOWN_REC);                //--- Remove scroll down rectangle
   ObjectDelete(0, SCROLL_DOWN_LABEL);              //--- Remove scroll down label
   ObjectDelete(0, SCROLL_SLIDER);                  //--- Remove scrollbar slider
}

//+------------------------------------------------------------------+
//| Calculate scrollbar slider height                                |
//+------------------------------------------------------------------+
int CalculateSliderHeight() {
   int scroll_area_height = g_displayHeight - 2 * 16;                 //--- Calculate scroll area height
   int slider_min_height = 20;                                        //--- Set minimum slider height
   if (g_total_height <= g_visible_height) return scroll_area_height; //--- Return full height if no scroll
   double visible_ratio = (double)g_visible_height / g_total_height;  //--- Calculate visible ratio
   int height = (int)MathFloor(scroll_area_height * visible_ratio);   //--- Calculate slider height
   return MathMax(slider_min_height, height);                         //--- Return minimum or calculated height
}

//+------------------------------------------------------------------+
//| Update scrollbar slider position                                 |
//+------------------------------------------------------------------+
void UpdateSliderPosition() {
   int displayX = g_mainX + g_sidePadding;                              //--- Calculate display x position
   int displayY = g_mainY + g_headerHeight + g_padding;                 //--- Calculate display y position
   int scrollbar_x = displayX + (g_mainWidth - 2 * g_sidePadding) - 16; //--- Set scrollbar x position
   int scrollbar_y = displayY + 16;                                     //--- Set scrollbar y position
   int scroll_area_height = g_displayHeight - 2 * 16;                   //--- Calculate scroll area height
   int max_scroll = MathMax(0, g_total_height - g_visible_height);      //--- Calculate maximum scroll
   if (max_scroll <= 0) return;                                         //--- Exit if no scroll needed
   double scroll_ratio = (double)scroll_pos / max_scroll;               //--- Calculate scroll ratio
   int scroll_area_y_max = scrollbar_y + scroll_area_height - slider_height; //--- Calculate max slider y
   int scroll_area_y_min = scrollbar_y;                                 //--- Set min slider y
   int new_y = scroll_area_y_min + (int)(scroll_ratio * (scroll_area_y_max - scroll_area_y_min)); //--- Calculate new y position
   new_y = MathMax(scroll_area_y_min, MathMin(new_y, scroll_area_y_max)); //--- Clamp y position
   ObjectSetInteger(0, SCROLL_SLIDER, OBJPROP_YDISTANCE, new_y);        //--- Update slider y position
   FileWriteString(logFileHandle, "Slider position updated: scroll_pos=" + IntegerToString(scroll_pos) + ", max_scroll=" + IntegerToString(max_scroll) + ", new_y=" + IntegerToString(new_y) + "\n"); //--- Log slider update
}

//+------------------------------------------------------------------+
//| Update scrollbar button colors                                   |
//+------------------------------------------------------------------+
void UpdateButtonColors() {
   int max_scroll = MathMax(0, g_total_height - g_visible_height);      //--- Calculate maximum scroll
   if (scroll_pos == 0) {                                               //--- Check if at top
      ObjectSetInteger(0, SCROLL_UP_LABEL, OBJPROP_COLOR, clrSilver);   //--- Set scroll up label to disabled color
   } else {                                                             //--- Not at top
      ObjectSetInteger(0, SCROLL_UP_LABEL, OBJPROP_COLOR, clrDimGray);  //--- Set scroll up label to active color
   }
   if (scroll_pos == max_scroll) {                                      //--- Check if at bottom
      ObjectSetInteger(0, SCROLL_DOWN_LABEL, OBJPROP_COLOR, clrSilver); //--- Set scroll down label to disabled color
   } else {                                                             //--- Not at bottom
      ObjectSetInteger(0, SCROLL_DOWN_LABEL, OBJPROP_COLOR, clrDimGray); //--- Set scroll down label to active color
   }
   FileWriteString(logFileHandle, "Button colors updated: scroll_pos=" + IntegerToString(scroll_pos) + ", max_scroll=" + IntegerToString(max_scroll) + "\n"); //--- Log button color update
}

//+------------------------------------------------------------------+
//| Scroll up (show earlier messages)                                |
//+------------------------------------------------------------------+
void ScrollUp() {
   if (scroll_pos > 0) {                            //--- Check if scroll possible
      scroll_pos = MathMax(0, scroll_pos - 30);     //--- Decrease scroll position
      UpdateResponseDisplay();                      //--- Update response display
      if (scroll_visible) {                         //--- Check if scrollbar visible
         UpdateSliderPosition();                    //--- Update slider position
         UpdateButtonColors();                      //--- Update button colors
      }
      FileWriteString(logFileHandle, "Scrolled up: scroll_pos=" + IntegerToString(scroll_pos) + "\n"); //--- Log scroll up
   }
}

//+------------------------------------------------------------------+
//| Scroll down (show later messages)                                |
//+------------------------------------------------------------------+
void ScrollDown() {
   int max_scroll = MathMax(0, g_total_height - g_visible_height); //--- Calculate maximum scroll
   if (scroll_pos < max_scroll) {                   //--- Check if scroll possible
      scroll_pos = MathMin(max_scroll, scroll_pos + 30); //--- Increase scroll position
      UpdateResponseDisplay();                      //--- Update response display
      if (scroll_visible) {                         //--- Check if scrollbar visible
         UpdateSliderPosition();                    //--- Update slider position
         UpdateButtonColors();                      //--- Update button colors
      }
      FileWriteString(logFileHandle, "Scrolled down: scroll_pos=" + IntegerToString(scroll_pos) + "\n"); //--- Log scroll down
   }
}
```

To ensure a responsive and adaptive chat-oriented interface, in the "getFontSizeByDPI" function, we retrieve the screen [DPI](https://en.wikipedia.org/wiki/Dots_per_inch "https://en.wikipedia.org/wiki/Dots_per_inch") (Dots Per Inch) with [TerminalInfoInteger](https://www.mql5.com/en/docs/check/terminalinfointeger) using [TERMINAL\_SCREEN\_DPI](https://www.mql5.com/en/docs/constants/environment_state/terminalstatus#enum_terminal_info_integer), scale a base font size relative to a standard DPI (96), and enforce a minimum size of 8 with [MathMax](https://www.mql5.com/en/docs/math/mathmax) for consistent text readability across displays. In the "CreateScrollbar" function, we calculate positions ("displayX", "displayY") and dimensions for the scrollbar, creating a leader rectangle ("SCROLL\_LEADER") with a light gray background (C'220,220,220'), up/down buttons ("SCROLL\_UP\_REC", "SCROLL\_DOWN\_REC") with [clrGainsboro](https://www.mql5.com/en/docs/constants/objectconstants/webcolors), and their labels ("SCROLL\_UP\_LABEL", "SCROLL\_DOWN\_LABEL") using Webdings arrows (0x35, 0x36) and DPI-adjusted font sizes via "createLabel" and "createRecLabel", logging failures with [FileWriteString](https://www.mql5.com/en/docs/files/filewritestring) to our file.

The "DeleteScrollbar" function removes all scrollbar objects with [ObjectDelete](https://www.mql5.com/en/docs/objects/objectdelete) for cleanup. In "CalculateSliderHeight", we compute the scrollbar slider height based on the visible text ratio, ensuring a minimum of 20 pixels. This makes sure the slider is not too small to be used when the conversation grows longer. The "UpdateSliderPosition" function adjusts the slider’s y-position using a scroll ratio derived from "scroll\_pos" and "max\_scroll", clamping it within bounds and logging updates. In "UpdateButtonColors", we set scroll button colors to "clrSilver" when disabled (at top/bottom) or "clrDimGray" when active, logging changes. The "ScrollUp" and "ScrollDown" functions adjust "scroll\_pos" by 30 pixels, call "UpdateResponseDisplay", update the scrollbar if visible, and log actions, creating a system for a dynamic, scrollable UI with adaptive text sizing and logging for the chat interface.

Now, since we will be wrapping complex conversations with longer paragraphs, we will need to handle empty lines to create clarity. Here is the logic we use to enhance the "WrapText" function to achieve that.

```
//+------------------------------------------------------------------+
//| Wrap text respecting newlines and max width                      |
//+------------------------------------------------------------------+
void WrapText(const string inputText, const string font, const int fontSize, const int maxWidth, string &wrappedLines[], int offset = 0) {
   const int maxChars = 60;                         //--- Set maximum characters per line
   ArrayResize(wrappedLines, 0);                    //--- Clear wrapped lines array
   TextSetFont(font, fontSize);                     //--- Set font
   string paragraphs[];                             //--- Declare paragraphs array
   int numParagraphs = StringSplit(inputText, '\n', paragraphs); //--- Split text into paragraphs
   for (int p = 0; p < numParagraphs; p++) {        //--- Iterate through paragraphs
      string para = paragraphs[p];                  //--- Get current paragraph
      if (StringLen(para) == 0) {                   //--- Check empty paragraph
         int size = ArraySize(wrappedLines);        //--- Get current size
         ArrayResize(wrappedLines, size + 1);       //--- Resize lines array
         wrappedLines[size] = " ";                  //--- Add empty line
         continue;                                  //--- Skip to next
      }
      string words[];                               //--- Declare words array
      int numWords = StringSplit(para, ' ', words); //--- Split paragraph into words
      string currentLine = "";                      //--- Initialize current line
      for (int w = 0; w < numWords; w++) {          //--- Iterate through words
         string testLine = currentLine + (StringLen(currentLine) > 0 ? " " : "") + words[w]; //--- Build test line
         uint wid, hei;                             //--- Declare width and height
         TextGetSize(testLine, wid, hei);           //--- Get test line size
         int textWidth = (int)wid;                  //--- Get text width
         if (textWidth + offset <= maxWidth && StringLen(testLine) <= maxChars) { //--- Check line fits
            currentLine = testLine;                 //--- Update current line
         } else {                                   //--- Line exceeds limits
            if (StringLen(currentLine) > 0) {       //--- Check non-empty line
               int size = ArraySize(wrappedLines);  //--- Get current size
               ArrayResize(wrappedLines, size + 1); //--- Resize lines array
               wrappedLines[size] = currentLine;    //--- Add line
            }
            currentLine = words[w];                 //--- Start new line
            TextGetSize(currentLine, wid, hei);     //--- Get new line size
            textWidth = (int)wid;                   //--- Update text width
            if (textWidth + offset > maxWidth || StringLen(currentLine) > maxChars) { //--- Check word too long
               string wrappedWord = "";             //--- Initialize wrapped word
               for (int c = 0; c < StringLen(words[w]); c++) { //--- Iterate through characters
                  string testWord = wrappedWord + StringSubstr(words[w], c, 1); //--- Build test word
                  TextGetSize(testWord, wid, hei);  //--- Get test word size
                  int wordWidth = (int)wid;         //--- Get word width
                  if (wordWidth + offset > maxWidth || StringLen(testWord) > maxChars) { //--- Check word fits
                     if (StringLen(wrappedWord) > 0) {       //--- Check non-empty word
                        int size = ArraySize(wrappedLines);  //--- Get current size
                        ArrayResize(wrappedLines, size + 1); //--- Resize lines array
                        wrappedLines[size] = wrappedWord;    //--- Add wrapped word
                     }
                     wrappedWord = StringSubstr(words[w], c, 1); //--- Start new word
                  } else {                          //--- Word fits
                     wrappedWord = testWord;        //--- Update wrapped word
                  }
               }
               currentLine = wrappedWord;           //--- Set current line to wrapped word
            }
         }
      }
      if (StringLen(currentLine) > 0) {             //--- Check remaining line
         int size = ArraySize(wrappedLines);        //--- Get current size
         ArrayResize(wrappedLines, size + 1);       //--- Resize lines array
         wrappedLines[size] = currentLine;          //--- Add line
      }
   }
}
```

It is not the first time we are encountering this function, so we will review it, pointing to the most important upgrade logic. In the function "WrapText", we set a maximum of 60 characters per line ("maxChars") and clear the output array "wrappedLines" with [ArrayResize](https://www.mql5.com/en/docs/array/arrayresize) just like we did with the previous function. We configure the font and size with [TextSetFont](https://www.mql5.com/en/docs/objects/textsetfont) and split the input text into paragraphs using [StringSplit](https://www.mql5.com/en/docs/strings/StringSplit) on newlines. For each paragraph, we handle empty paragraphs by adding a space to "wrappedLines" and skipping to the next.

Non-empty paragraphs are split into words with "StringSplit", and we build lines by adding words if they fit within "maxWidth" (adjusted by "offset") and the character limit, using [TextGetSize](https://www.mql5.com/en/docs/objects/textgetsize) to check width. If a line exceeds limits, we add the current line to "wrappedLines" and start a new line with the current word; for oversized words, we split them character by character, adding segments to new lines when exceeding width or character limits, ensuring each segment is stored in "wrappedLines". Any remaining line is added to the output. During initialization, we will need to set the element colors and delete the new elements when removing the program.

```
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit() {
   button_darker_bg = DarkenColor(button_original_bg);     //--- Set darker button background
   clear_darker_bg = DarkenColor(clear_original_bg);       //--- Set darker clear button background
   new_chat_darker_bg = DarkenColor(new_chat_original_bg); //--- Set darker new chat button background
   logFileHandle = FileOpen(LogFileName, FILE_READ | FILE_WRITE | FILE_TXT); //--- Open log file
   if (logFileHandle == INVALID_HANDLE) {                  //--- Check file open failure
      Print("Failed to open log file: ", GetLastError());  //--- Log failure
      return(INIT_FAILED);                                 //--- Return initialization failure
   }
   FileSeek(logFileHandle, 0, SEEK_END);                   //--- Move to end of log file
   FileWriteString(logFileHandle, "EA Initialized at " + TimeToString(TimeCurrent()) + "\n"); //--- Log initialization
   CreateDashboard();                                      //--- Create dashboard UI
   UpdateResponseDisplay();                                //--- Update response display
   ChartSetInteger(0, CHART_EVENT_MOUSE_MOVE, true);       //--- Enable mouse move events
   ChartSetInteger(0, CHART_EVENT_MOUSE_WHEEL, true);      //--- Enable mouse wheel events
   ChartSetInteger(0, CHART_MOUSE_SCROLL, true);           //--- Enable chart scrolling
   return(INIT_SUCCEEDED);                                 //--- Return initialization success
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason) {
   ObjectsDeleteAll(0, "ChatGPT_");                        //--- Remove all ChatGPT objects
   DeleteScrollbar();                                      //--- Delete scrollbar elements
   if (logFileHandle != INVALID_HANDLE) {                  //--- Check if log file open
      FileClose(logFileHandle);                            //--- Close log file
   }
}
```

With the initialization being done, let us define the new elements and add them to the display so we can see what milestone we hit.

```
//+------------------------------------------------------------------+
//| Create dashboard UI                                              |
//+------------------------------------------------------------------+
void CreateDashboard() {
   g_mainHeight = g_headerHeight + 2 * g_padding + g_displayHeight + g_footerHeight; //--- Calculate main height
   int displayX = g_mainX + g_sidePadding;          //--- Calculate display x
   int displayY = g_mainY + g_headerHeight + g_padding; //--- Calculate display y
   int displayW = g_mainWidth - 2 * g_sidePadding;  //--- Calculate display width
   int footerY = displayY + g_displayHeight + g_padding; //--- Calculate footer y
   int inputWidth = 448;                            //--- Set input field width
   int sendWidth = 80;                              //--- Set send button width
   int gap = 10;                                    //--- Set gap between elements
   int totalW = inputWidth + gap + sendWidth;       //--- Calculate total width
   int centerX = g_mainX + (g_mainWidth - totalW) / 2; //--- Calculate center x
   int inputX = centerX;                            //--- Set input field x
   int sendX = inputX + inputWidth + gap;           //--- Calculate send button x
   int elemHeight = 36;                             //--- Set element height
   int elemY = footerY + 8;                         //--- Calculate element y
   createRecLabel("ChatGPT_MainContainer", g_mainX, g_mainY, g_mainWidth, g_mainHeight, clrWhite, 1, clrLightGray); //--- Create main container
   createRecLabel("ChatGPT_HeaderBg", g_mainX, g_mainY, g_mainWidth, g_headerHeight, clrWhiteSmoke, 0, clrNONE); //--- Create header background
   string title = "ChatGPT AI EA";                  //--- Set title
   string titleFont = "Arial Rounded MT Bold";      //--- Set title font
   int titleSize = 14;                              //--- Set title font size
   TextSetFont(titleFont, titleSize);               //--- Set title font
   uint titleWid, titleHei;                         //--- Declare title dimensions
   TextGetSize(title, titleWid, titleHei);          //--- Get title size
   int titleY = g_mainY + (g_headerHeight - (int)titleHei) / 2 - 4; //--- Calculate title y
   int titleX = g_mainX + g_sidePadding;            //--- Set title x
   createLabel("ChatGPT_TitleLabel", titleX, titleY, title, clrDarkSlateGray, titleSize, titleFont, CORNER_LEFT_UPPER, ANCHOR_LEFT_UPPER); //--- Create title label
   string dateStr = TimeToString(TimeTradeServer(), TIME_DATE|TIME_MINUTES); //--- Get current date
   string dateFont = "Arial";                       //--- Set date font
   int dateSize = 12;                               //--- Set date font size
   TextSetFont(dateFont, dateSize);                 //--- Set date font
   uint dateWid, dateHei;                           //--- Declare date dimensions
   TextGetSize(dateStr, dateWid, dateHei);          //--- Get date size
   int dateX = g_mainX + g_mainWidth / 2 - (int)(dateWid / 2) - 50; //--- Calculate date x
   int dateY = g_mainY + (g_headerHeight - (int)dateHei) / 2 - 4; //--- Calculate date y
   createLabel("ChatGPT_DateLabel", dateX, dateY, dateStr, clrSlateGray, dateSize, dateFont, CORNER_LEFT_UPPER, ANCHOR_LEFT_UPPER); //--- Create date label
   int clearWidth = 100;                            //--- Set clear button width
   int clearX = g_mainX + g_mainWidth - clearWidth - g_sidePadding; //--- Calculate clear button x
   int clearY = g_mainY + 4;                        //--- Set clear button y
   createButton("ChatGPT_ClearButton", clearX, clearY, clearWidth, g_headerHeight - 8, "Clear", clrWhite, 11, clear_original_bg, clrIndianRed); //--- Create clear button
   int new_chat_width = 100;                        //--- Set new chat button width
   int new_chat_x = clearX - new_chat_width - g_sidePadding; //--- Calculate new chat button x
   createButton("ChatGPT_NewChatButton", new_chat_x, clearY, new_chat_width, g_headerHeight - 8, "New Chat", clrWhite, 11, new_chat_original_bg, clrRoyalBlue); //--- Create new chat button
   createRecLabel("ChatGPT_ResponseBg", displayX, displayY, displayW, g_displayHeight, clrWhite, 1, clrGainsboro, BORDER_FLAT, STYLE_SOLID); //--- Create response background
   createRecLabel("ChatGPT_FooterBg", g_mainX, footerY, g_mainWidth, g_footerHeight, clrGainsboro, 0, clrNONE); //--- Create footer background
   createEdit("ChatGPT_InputEdit", inputX, elemY, inputWidth, elemHeight, "", clrBlack, 11, clrWhite, clrSilver); //--- Create input field
   createButton("ChatGPT_SubmitButton", sendX, elemY, sendWidth, elemHeight, "Send", clrWhite, 11, button_original_bg, clrDarkBlue); //--- Create send button
   ChartRedraw();                                   //--- Redraw chart
}
```

In the core dashboard layout function, we calculate the main container height ("g\_mainHeight") as the sum of "g\_headerHeight", "g\_displayHeight", "g\_footerHeight", and twice "g\_padding", and determine positions for the display area ("displayX", "displayY") and footer ("footerY") using padding values since we want our dashboard to be dynamic and not static as we had done in the previous version. We create the main container ("ChatGPT\_MainContainer") and header background ("ChatGPT\_HeaderBg") with "createRecLabel" using white and light gray colors, and add a title label ("ChatGPT\_TitleLabel") with "ChatGPT AI EA" in "Arial Rounded MT Bold" at size 14, positioned with [TextGetSize](https://www.mql5.com/en/docs/objects/textgetsize) for alignment. A date label ("ChatGPT\_DateLabel") is created with the current server time from [TimeTradeServer](https://www.mql5.com/en/docs/dateandtime/timetradeserver) in "Arial" at size 12, centered horizontally.

We add "Clear" ("ChatGPT\_ClearButton") and "New Chat" ("ChatGPT\_NewChatButton") buttons in the header with "createButton", using distinct colors (" [clrLightCoral](https://www.mql5.com/en/docs/constants/objectconstants/webcolors)", "clrLightBlue") and a smaller font size of 11. The response area ("ChatGPT\_ResponseBg") and footer ("ChatGPT\_FooterBg") are created with "createRecLabel" for the chat display and input section. An input field ("ChatGPT\_InputEdit") of width 448 and a "Send" button ("ChatGPT\_SubmitButton") of width 80 are centered in the footer using "createEdit" and "createButton", with a 10-pixel gap. Finally, we redraw the chart with the [ChartRedraw](https://www.mql5.com/en/docs/chart_operations/ChartRedraw) function. Upon compilation, we get the following outcome.

![FINAL UI WITH ELEMENTS](https://c.mql5.com/2/172/Screenshot_2025-09-27_172818.png)

Since we now have the interface with all the elements, we can move on to updating the display with the new conversation history. We will, however, need some utility functions to get the message lines and their heights to ensure the conversation fits within the display area without overflowing, as well as take care of the timestamp lines we will be incorporating.

```
//+------------------------------------------------------------------+
//| Check if string is a timestamp in HH:MM format                   |
//+------------------------------------------------------------------+
bool IsTimestamp(string line) {
   StringTrimLeft(line);                                 //--- Trim left whitespace
   StringTrimRight(line);                                //--- Trim right whitespace
   if (StringLen(line) != 5) return false;               //--- Check length
   if (StringGetCharacter(line, 2) != ':') return false; //--- Check colon
   string hh = StringSubstr(line, 0, 2);                 //--- Extract hours
   string mm = StringSubstr(line, 3, 2);                 //--- Extract minutes
   int h = (int)StringToInteger(hh);                     //--- Convert hours to integer
   int m = (int)StringToInteger(mm);                     //--- Convert minutes to integer
   if (h < 0 || h > 23 || m < 0 || m > 59) return false; //--- Validate time
   return true;                                          //--- Confirm valid timestamp
}

//+------------------------------------------------------------------+
//| Compute lines and height for messages                            |
//+------------------------------------------------------------------+
void ComputeLinesAndHeight(const string &font, const int fontSize, const int timestampFontSize,
                           const int adjustedLineHeight, const int adjustedTimestampHeight,
                           const int messageMargin, const int maxTextWidth,
                           const string &msgRoles[], const string &msgContents[], const string &msgTimestamps[],
                           const int numMessages, int &totalHeight_out, int &totalLines_out,
                           string &allLines_out[], string &lineRoles_out[], int &lineHeights_out[]) {
   ArrayResize(allLines_out, 0);                    //--- Clear lines array
   ArrayResize(lineRoles_out, 0);                   //--- Clear roles array
   ArrayResize(lineHeights_out, 0);                 //--- Clear heights array
   totalLines_out = 0;                              //--- Initialize total lines
   totalHeight_out = 0;                             //--- Initialize total height
   for (int m = 0; m < numMessages; m++) {          //--- Iterate through messages
      string wrappedLines[];                        //--- Declare wrapped lines
      WrapText(msgContents[m], font, fontSize, maxTextWidth, wrappedLines); //--- Wrap message content
      int numLines = ArraySize(wrappedLines);       //--- Get number of lines
      int currSize = ArraySize(allLines_out);       //--- Get current size
      ArrayResize(allLines_out, currSize + numLines + 1); //--- Resize lines array
      ArrayResize(lineRoles_out, currSize + numLines + 1); //--- Resize roles array
      ArrayResize(lineHeights_out, currSize + numLines + 1); //--- Resize heights array
      for (int l = 0; l < numLines; l++) {          //--- Iterate through wrapped lines
         allLines_out[currSize + l] = wrappedLines[l]; //--- Add line
         lineRoles_out[currSize + l] = msgRoles[m]; //--- Add role
         lineHeights_out[currSize + l] = adjustedLineHeight; //--- Set line height
         totalHeight_out += adjustedLineHeight;     //--- Update total height
      }
      allLines_out[currSize + numLines] = msgTimestamps[m]; //--- Add timestamp
      lineRoles_out[currSize + numLines] = msgRoles[m] + "_timestamp"; //--- Add timestamp role
      lineHeights_out[currSize + numLines] = adjustedTimestampHeight; //--- Set timestamp height
      totalHeight_out += adjustedTimestampHeight;   //--- Update total height
      totalLines_out += numLines + 1;               //--- Update total lines
      if (m < numMessages - 1) {                    //--- Check for margin
         totalHeight_out += messageMargin;          //--- Add message margin
      }
   }
}
```

Here, we implement utility functions to validate timestamps and compute message display properties. In the "IsTimestamp" function, we trim whitespace from the input string using [StringTrimLeft](https://www.mql5.com/en/docs/strings/stringtrimleft) and [StringTrimRight](https://www.mql5.com/en/docs/strings/stringtrimright), check if its length is exactly 5 characters, verify a colon at position 2 with [StringGetCharacter](https://www.mql5.com/en/docs/strings/stringgetcharacter), extract hours and minutes with [StringSubstr](https://www.mql5.com/en/docs/strings/stringsubstr), convert them to integers with [StringToInteger](https://www.mql5.com/en/docs/convert/StringToInteger), and return true if the hours (0-23) and minutes (0-59) form a valid time, ensuring accurate timestamp detection in the conversation history. You will need to define your specific rules if you choose a different approach.

In the "ComputeLinesAndHeight" function, we clear output arrays ("allLines\_out", "lineRoles\_out", "lineHeights\_out") with [ArrayResize](https://www.mql5.com/en/docs/array/arrayresize) and initialize "totalLines\_out" and "totalHeight\_out" to zero. For each message, we wrap its content using "WrapText" with the specified font, font size, and maximum width, then resize output arrays to accommodate the wrapped lines plus a timestamp, assigning each line’s text, role ("User" or "AI"), and height ("adjustedLineHeight" for content, "adjustedTimestampHeight" for timestamps) to "allLines\_out", "lineRoles\_out", and "lineHeights\_out", respectively, and update "totalHeight\_out" and "totalLines\_out". We add a message margin ("messageMargin") between messages (except the last) to ensure visual separation, ensuring we validate timestamps and organize message text for display in our scrollable chat interface.

With these functions, we can now update the display function to parse history into roles, contents, and timestamps, align role messages, add margins, handle scrolling and cropping, and show the scrollbar dynamically.

```
//+------------------------------------------------------------------+
//| Update response display with scrolling                           |
//+------------------------------------------------------------------+
void UpdateResponseDisplay() {
   int total = ObjectsTotal(0, 0, -1);              //--- Get total objects
   for (int j = total - 1; j >= 0; j--) {           //--- Iterate through objects
      string name = ObjectName(0, j, 0, -1);        //--- Get object name
      if (StringFind(name, "ChatGPT_ResponseLine_") == 0 ||
          StringFind(name, "ChatGPT_MessageBg_") == 0 ||
          StringFind(name, "ChatGPT_MessageText_") == 0 ||
          StringFind(name, "ChatGPT_Timestamp_") == 0) { //--- Check for message objects
         ObjectDelete(0, name);                     //--- Delete object
      }
   }
   string displayText = conversationHistory;        //--- Get conversation history
   int textX = g_mainX + g_sidePadding + g_textPadding; //--- Calculate text x position
   int textY = g_mainY + g_headerHeight + g_padding + g_textPadding; //--- Calculate text y position
   int fullMaxWidth = g_mainWidth - 2 * g_sidePadding - 2 * g_textPadding; //--- Calculate max text width
   if (displayText == "") {                         //--- Check empty history
      string objName = "ChatGPT_ResponseLine_0";    //--- Set default label name
      createLabel(objName, textX, textY, "Type your message below and click Send to chat with the AI.", clrGray, 10, "Arial", CORNER_LEFT_UPPER, ANCHOR_LEFT_UPPER); //--- Create default label
      g_total_height = 0;                           //--- Reset total height
      g_visible_height = g_displayHeight - 2 * g_textPadding; //--- Set visible height
      if (scroll_visible) {                         //--- Check scrollbar visible
         DeleteScrollbar();                         //--- Delete scrollbar
         scroll_visible = false;                    //--- Reset scrollbar visibility
      }
      ChartRedraw();                                //--- Redraw chart
      return;                                       //--- Exit function
   }
   string parts[];                                  //--- Declare parts array
   int numParts = StringSplit(displayText, '\n', parts); //--- Split history into parts
   string msgRoles[];                               //--- Declare roles array
   string msgContents[];                            //--- Declare contents array
   string msgTimestamps[];                          //--- Declare timestamps array
   string currentRole = "";                         //--- Initialize current role
   string currentContent = "";                      //--- Initialize current content
   string currentTimestamp = "";                    //--- Initialize current timestamp
   for (int p = 0; p < numParts; p++) {             //--- Iterate through parts
      string line = parts[p];                       //--- Get current line
      StringTrimLeft(line);                         //--- Trim left whitespace
      StringTrimRight(line);                        //--- Trim right whitespace
      if (StringLen(line) == 0) {                   //--- Check empty line
         if (currentRole != "") currentContent += "\n"; //--- Append newline
         continue;                                  //--- Skip to next
      }
      if (StringFind(line, "You: ") == 0) {         //--- Check user message
         if (currentRole != "") {                   //--- Check existing message
            int size = ArraySize(msgRoles);         //--- Get current size
            ArrayResize(msgRoles, size + 1);        //--- Resize roles array
            ArrayResize(msgContents, size + 1);     //--- Resize contents array
            ArrayResize(msgTimestamps, size + 1);   //--- Resize timestamps array
            msgRoles[size] = currentRole;           //--- Add role
            msgContents[size] = currentContent;     //--- Add content
            msgTimestamps[size] = currentTimestamp; //--- Add timestamp
         }
         currentRole = "User";                      //--- Set role to User
         currentContent = StringSubstr(line, 5);    //--- Extract user content
         currentTimestamp = "";                     //--- Reset timestamp
         continue;                                  //--- Skip to next
      } else if (StringFind(line, "AI: ") == 0) {   //--- Check AI message
         if (currentRole != "") {                   //--- Check existing message
            int size = ArraySize(msgRoles);         //--- Get current size
            ArrayResize(msgRoles, size + 1);        //--- Resize roles array
            ArrayResize(msgContents, size + 1);     //--- Resize contents array
            ArrayResize(msgTimestamps, size + 1);   //--- Resize timestamps array
            msgRoles[size] = currentRole;           //--- Add role
            msgContents[size] = currentContent;     //--- Add content
            msgTimestamps[size] = currentTimestamp; //--- Add timestamp
         }
         currentRole = "AI";                        //--- Set role to AI
         currentContent = StringSubstr(line, 4);    //--- Extract AI content
         currentTimestamp = "";                     //--- Reset timestamp
         continue;                                  //--- Skip to next
      } else if (IsTimestamp(line)) {               //--- Check timestamp
         if (currentRole != "") {                   //--- Check existing message
            currentTimestamp = line;                //--- Set timestamp
            int size = ArraySize(msgRoles);         //--- Get current size
            ArrayResize(msgRoles, size + 1);        //--- Resize roles array
            ArrayResize(msgContents, size + 1);     //--- Resize contents array
            ArrayResize(msgTimestamps, size + 1);   //--- Resize timestamps array
            msgRoles[size] = currentRole;           //--- Add role
            msgContents[size] = currentContent;     //--- Add content
            msgTimestamps[size] = currentTimestamp; //--- Add timestamp
            currentRole = "";                       //--- Reset role
         }
      } else {                                      //--- Append to content
         if (currentRole != "") {                   //--- Check active message
            currentContent += "\n" + line;          //--- Append line
         }
      }
   }
   if (currentRole != "") {                         //--- Check final message
      int size = ArraySize(msgRoles);               //--- Get current size
      ArrayResize(msgRoles, size + 1);              //--- Resize roles array
      ArrayResize(msgContents, size + 1);           //--- Resize contents array
      ArrayResize(msgTimestamps, size + 1);         //--- Resize timestamps array
      msgRoles[size] = currentRole;                 //--- Add role
      msgContents[size] = currentContent;           //--- Add content
      msgTimestamps[size] = currentTimestamp;       //--- Add timestamp
   }
   int numMessages = ArraySize(msgRoles);           //--- Get number of messages
   if (numMessages == 0) {                          //--- Check no messages
      string objName = "ChatGPT_ResponseLine_0";    //--- Set default label name
      createLabel(objName, textX, textY, "Type your message below and click Send to chat with the AI.", clrGray, 10, "Arial", CORNER_LEFT_UPPER, ANCHOR_LEFT_UPPER); //--- Create default label
      g_total_height = 0;                           //--- Reset total height
      g_visible_height = g_displayHeight - 2 * g_textPadding; //--- Set visible height
      if (scroll_visible) {                         //--- Check scrollbar visible
         DeleteScrollbar();                         //--- Delete scrollbar
         scroll_visible = false;                    //--- Reset scrollbar visibility
      }
      ChartRedraw();                                //--- Redraw chart
      return;                                       //--- Exit function
   }
   string font = "Arial";                           //--- Set font
   int fontSize = 10;                               //--- Set font size
   int timestampFontSize = 8;                       //--- Set timestamp font size
   int lineHeight = TextGetHeight("A", font, fontSize); //--- Get line height
   int timestampHeight = TextGetHeight("A", font, timestampFontSize); //--- Get timestamp height
   int adjustedLineHeight = lineHeight + g_lineSpacing; //--- Calculate adjusted line height
   int adjustedTimestampHeight = timestampHeight + g_lineSpacing; //--- Calculate adjusted timestamp height
   int messageMargin = 12;                          //--- Set message margin
   int visibleHeight = g_displayHeight - 2 * g_textPadding; //--- Calculate visible height
   g_visible_height = visibleHeight;                //--- Set visible height
   string tentativeAllLines[];                      //--- Declare tentative lines
   string tentativeLineRoles[];                     //--- Declare tentative roles
   int tentativeLineHeights[];                      //--- Declare tentative heights
   int tentativeTotalHeight, tentativeTotalLines;   //--- Declare tentative totals
   ComputeLinesAndHeight(font, fontSize, timestampFontSize, adjustedLineHeight, adjustedTimestampHeight,
                         messageMargin, fullMaxWidth, msgRoles, msgContents, msgTimestamps, numMessages,
                         tentativeTotalHeight, tentativeTotalLines, tentativeAllLines, tentativeLineRoles, tentativeLineHeights); //--- Compute tentative lines
   bool need_scroll = tentativeTotalHeight > visibleHeight; //--- Check if scrolling needed
   bool should_show_scrollbar = false;              //--- Initialize scrollbar visibility
   int reserved_width = 0;                          //--- Initialize reserved width
   if (ScrollbarMode != SCROLL_WHEEL_ONLY) {        //--- Check scrollbar mode
      should_show_scrollbar = need_scroll && (ScrollbarMode == SCROLL_DYNAMIC_ALWAYS || (ScrollbarMode == SCROLL_DYNAMIC_HOVER && mouse_in_display)); //--- Determine scrollbar visibility
      if (should_show_scrollbar) {                  //--- Check if scrollbar needed
         reserved_width = 16;                       //--- Reserve scrollbar width
      }
   }
   string allLines[];                               //--- Declare final lines
   string lineRoles[];                              //--- Declare final roles
   int lineHeights[];                               //--- Declare final heights
   int totalHeight, totalLines;                     //--- Declare final totals
   int maxTextWidth = fullMaxWidth - reserved_width; //--- Calculate max text width
   if (reserved_width > 0) {                        //--- Check if scrollbar reserved
      ComputeLinesAndHeight(font, fontSize, timestampFontSize, adjustedLineHeight, adjustedTimestampHeight,
                            messageMargin, maxTextWidth, msgRoles, msgContents, msgTimestamps, numMessages,
                            totalHeight, totalLines, allLines, lineRoles, lineHeights); //--- Compute lines with reduced width
   } else {                                         //--- Use tentative values
      totalHeight = tentativeTotalHeight;           //--- Set total height
      totalLines = tentativeTotalLines;             //--- Set total lines
      ArrayCopy(allLines, tentativeAllLines);       //--- Copy lines
      ArrayCopy(lineRoles, tentativeLineRoles);     //--- Copy roles
      ArrayCopy(lineHeights, tentativeLineHeights); //--- Copy heights
   }
   FileWriteString(logFileHandle, "UpdateResponseDisplay: totalHeight=" + IntegerToString(totalHeight) + ", visibleHeight=" + IntegerToString(visibleHeight) + ", totalLines=" + IntegerToString(totalLines) + ", reserved_width=" + IntegerToString(reserved_width) + "\n"); //--- Log display update
   g_total_height = totalHeight;                    //--- Set total height
   bool prev_scroll_visible = scroll_visible;       //--- Store previous scrollbar state
   scroll_visible = should_show_scrollbar;          //--- Update scrollbar visibility
   if (scroll_visible != prev_scroll_visible) {     //--- Check scrollbar state change
      if (scroll_visible) {                         //--- Show scrollbar
         CreateScrollbar();                         //--- Create scrollbar
      } else {                                      //--- Hide scrollbar
         DeleteScrollbar();                         //--- Delete scrollbar
      }
   }
   int max_scroll = MathMax(0, totalHeight - visibleHeight); //--- Calculate max scroll
   if (scroll_pos > max_scroll) scroll_pos = max_scroll; //--- Clamp scroll position
   if (scroll_pos < 0) scroll_pos = 0;              //--- Ensure non-negative scroll
   if (totalHeight > visibleHeight && scroll_pos == prev_scroll_pos && prev_scroll_pos == -1) { //--- Check initial scroll
      scroll_pos = max_scroll;                      //--- Set to bottom
   }
   if (scroll_visible) {                            //--- Update scrollbar
      slider_height = CalculateSliderHeight();      //--- Calculate slider height
      ObjectSetInteger(0, SCROLL_SLIDER, OBJPROP_YSIZE, slider_height); //--- Set slider height
      UpdateSliderPosition();                       //--- Update slider position
      UpdateButtonColors();                         //--- Update button colors
   }
   int currentY = textY - scroll_pos;               //--- Calculate current y position
   int endY = textY + visibleHeight;                //--- Calculate end y position
   int startLineIndex = 0;                          //--- Initialize start line index
   int currentHeight = 0;                           //--- Initialize current height
   for (int line = 0; line < totalLines; line++) {  //--- Find start line
      if (currentHeight >= scroll_pos) {            //--- Check if at scroll position
         startLineIndex = line;                     //--- Set start line
         currentY = textY + (currentHeight - scroll_pos); //--- Set current y
         break;                                     //--- Exit loop
      }
      currentHeight += lineHeights[line];           //--- Add line height
      if (line < totalLines - 1 && StringFind(lineRoles[line], "_timestamp") >= 0 && StringFind(lineRoles[line + 1], "_timestamp") < 0) { //--- Check message gap
         currentHeight += messageMargin;            //--- Add message margin
      }
   }
   int numVisibleLines = 0;                         //--- Initialize visible lines
   int visibleHeightUsed = 0;                       //--- Initialize used height
   for (int line = startLineIndex; line < totalLines; line++) { //--- Count visible lines
      int lineHeight = lineHeights[line];           //--- Get line height
      if (visibleHeightUsed + lineHeight > visibleHeight) break; //--- Check height limit
      visibleHeightUsed += lineHeight;              //--- Add line height
      numVisibleLines++;                            //--- Increment visible lines
      if (line < totalLines - 1 && StringFind(lineRoles[line], "_timestamp") >= 0 && StringFind(lineRoles[line + 1], "_timestamp") < 0) { //--- Check message gap
         if (visibleHeightUsed + messageMargin > visibleHeight) break; //--- Check margin limit
         visibleHeightUsed += messageMargin;        //--- Add message margin
      }
   }
   FileWriteString(logFileHandle, "Visible lines: startLineIndex=" + IntegerToString(startLineIndex) + ", numVisibleLines=" + IntegerToString(numVisibleLines) + ", scroll_pos=" + IntegerToString(scroll_pos) + ", currentY=" + IntegerToString(currentY) + "\n"); //--- Log visible lines
   int leftX = g_mainX + g_sidePadding + g_textPadding; //--- Set left text x
   int rightX = g_mainX + g_mainWidth - g_sidePadding - g_textPadding - reserved_width; //--- Set right text x
   color userColor = clrGray;                       //--- Set user text color
   color aiColor = clrBlue;                         //--- Set AI text color
   color timestampColor = clrDarkGray;              //--- Set timestamp color
   for (int li = 0; li < numVisibleLines; li++) {   //--- Display visible lines
      int lineIndex = startLineIndex + li;          //--- Calculate line index
      if (lineIndex >= totalLines) break;           //--- Check valid index
      string line = allLines[lineIndex];            //--- Get line text
      string role = lineRoles[lineIndex];           //--- Get line role
      bool isTimestamp = StringFind(role, "_timestamp") >= 0; //--- Check if timestamp
      int currFontSize = isTimestamp ? timestampFontSize : fontSize; //--- Set font size
      color textCol = isTimestamp ? timestampColor : (StringFind(role, "User") >= 0 ? userColor : aiColor); //--- Set text color
      string display_line = line;                   //--- Set display line
      if (line == " ") {                            //--- Check empty line
         display_line = " ";                        //--- Set to space
         textCol = clrWhite;                        //--- Set to white
      }
      int textX_pos = (StringFind(role, "User") >= 0) ? rightX : leftX; //--- Set text x position
      ENUM_ANCHOR_POINT textAnchor = (StringFind(role, "User") >= 0) ? ANCHOR_RIGHT_UPPER : ANCHOR_LEFT_UPPER; //--- Set text anchor
      string lineName = "ChatGPT_MessageText_" + IntegerToString(lineIndex); //--- Generate line name
      if (currentY >= textY && currentY < endY) {   //--- Check if visible
         createLabel(lineName, textX_pos, currentY, display_line, textCol, currFontSize, font, CORNER_LEFT_UPPER, textAnchor); //--- Create label
      }
      currentY += lineHeights[lineIndex];           //--- Increment y position
      if (lineIndex < totalLines - 1 && StringFind(lineRoles[lineIndex], "_timestamp") >= 0 && StringFind(lineRoles[lineIndex + 1], "_timestamp") < 0) { //--- Check message gap
         currentY += messageMargin;                 //--- Add message margin
      }
   }
   ChartRedraw();                                   //--- Redraw chart
}
```

In the "UpdateResponseDisplay" function, we first clear existing message-related objects ("ChatGPT\_ResponseLine\_", "ChatGPT\_MessageBg\_", "ChatGPT\_MessageText\_", "ChatGPT\_Timestamp\_") using [ObjectsTotal](https://www.mql5.com/en/docs/objects/objectstotal) and [ObjectDelete](https://www.mql5.com/en/docs/objects/objectdelete) to refresh the display. If the conversation history ("conversationHistory") is empty, we create a default label with "createLabel" prompting the user to type a message, reset "g\_total\_height" to 0, set "g\_visible\_height" to the display height minus padding, remove the scrollbar with "DeleteScrollbar" if visible, and redraw with the [ChartRedraw](https://www.mql5.com/en/docs/chart_operations/ChartRedraw) function. Otherwise, we split the history into parts using [StringSplit](https://www.mql5.com/en/docs/strings/StringSplit) on newlines, parsing lines into "msgRoles", "msgContents", and "msgTimestamps" by identifying "You: ", "AI: ", and timestamps with "IsTimestamp", accumulating content across lines, and storing completed messages.

We calculate text positioning ("textX", "textY") and maximum width ("fullMaxWidth"), set font sizes (10 for messages, 8 for timestamps), and compute line heights with "TextGetHeight" plus "g\_lineSpacing". Using "ComputeLinesAndHeight", we generate tentative line arrays and heights, check if scrolling is needed, and determine scrollbar visibility based on "ScrollbarMode" and "mouse\_in\_display", reserving 16 pixels for the scrollbar if shown. We recompute lines with adjusted width if necessary, update "g\_total\_height", manage scrollbar visibility with "CreateScrollbar" or "DeleteScrollbar", clamp "scroll\_pos" within "max\_scroll", and set it to the bottom for new messages.

We calculate the starting line and y-position based on "scroll\_pos", determine visible lines within "g\_visible\_height", and render each line with "createLabel", using left-aligned AI messages ("clrBlue") and right-aligned user messages ("clrGray") with timestamps ("clrDarkGray"), applying a 12-pixel message margin. Finally, we log display details with [FileWriteString](https://www.mql5.com/en/docs/files/filewritestring) and redraw. This will ensure the display area is populated with available conversation history. We now need to make sure that when we hit send, we send the prompt. We will expand the existing function into several functions for easier management in future versions.

```
//+------------------------------------------------------------------+
//| Build messages array from history                                |
//+------------------------------------------------------------------+
string BuildMessagesFromHistory(string newPrompt) {
   string messages = "[";                          //--- Start JSON array\
   string temp = conversationHistory;              //--- Copy conversation history\
   while (StringLen(temp) > 0) {                   //--- Process history\
      int you_pos = StringFind(temp, "You: ");     //--- Find user message\
      if (you_pos != 0) break;                     //--- Exit if no user message\
      temp = StringSubstr(temp, 5);                //--- Extract after "You: "\
      int end_user = StringFind(temp, "\n");       //--- Find end of user message\
      string user_content = StringSubstr(temp, 0, end_user); //--- Get user content\
      temp = StringSubstr(temp, end_user + 1);     //--- Move past user message\
      int end_ts1 = StringFind(temp, "\n");        //--- Find end of timestamp\
      temp = StringSubstr(temp, end_ts1 + 1);      //--- Move past timestamp\
      int ai_pos = StringFind(temp, "AI: ");       //--- Find AI message\
      if (ai_pos != 0) break;                      //--- Exit if no AI message\
      temp = StringSubstr(temp, 4);                //--- Extract after "AI: "\
      int end_ai = StringFind(temp, "\n");         //--- Find end of AI message\
      string ai_content = StringSubstr(temp, 0, end_ai); //--- Get AI content\
      temp = StringSubstr(temp, end_ai + 1);       //--- Move past AI message\
      int end_ts2 = StringFind(temp, "\n\n");      //--- Find end of conversation block\
      temp = StringSubstr(temp, end_ts2 + 2);      //--- Move past block\
      messages += "{\"role\":\"user\",\"content\":\"" + JsonEscape(user_content) + "\"},"; //--- Add user message\
      messages += "{\"role\":\"assistant\",\"content\":\"" + JsonEscape(ai_content) + "\"},"; //--- Add AI message\
   }\
   messages += "{\"role\":\"user\",\"content\":\"" + JsonEscape(newPrompt) + "\"}]"; //--- Add new prompt
   return messages;                                //--- Return JSON messages
}

//+------------------------------------------------------------------+
//| Get ChatGPT response via API                                     |
//+------------------------------------------------------------------+
string GetChatGPTResponse(string prompt) {
   string messages = BuildMessagesFromHistory(prompt); //--- Build JSON messages
   string requestData = "{\"model\":\"" + OpenAI_Model + "\",\"messages\":" + messages + ",\"max_tokens\":" + IntegerToString(MaxResponseLength) + "}"; //--- Create request JSON
   FileWriteString(logFileHandle, "Request Data: " + requestData + "\n"); //--- Log request data
   char postData[];                                    //--- Declare post data array
   int dataLen = StringToCharArray(requestData, postData, 0, WHOLE_ARRAY, CP_UTF8); //--- Convert request to char array
   ArrayResize(postData, dataLen - 1);                 //--- Remove null terminator
   FileWriteString(logFileHandle, "Raw Post Data (Hex): " + LogCharArray(postData) + "\n"); //--- Log raw data
   string headers = "Authorization: Bearer " + OpenAI_API_Key + "\r\n" +
                    "Content-Type: application/json; charset=UTF-8\r\n" +
                    "Content-Length: " + IntegerToString(dataLen - 1) + "\r\n\r\n"; //--- Set request headers
   FileWriteString(logFileHandle, "Request Headers: " + headers + "\n"); //--- Log headers
   char result[];                                     //--- Declare result array
   string resultHeaders;                              //--- Declare result headers
   int res = WebRequest("POST", OpenAI_Endpoint, headers, 10000, postData, result, resultHeaders); //--- Send API request
   if (res != 200) {                                  //--- Check request failure
      string response = CharArrayToString(result, 0, WHOLE_ARRAY, CP_UTF8); //--- Convert result to string
      string errMsg = "API request failed: HTTP Code " + IntegerToString(res) + ", Error: " + IntegerToString(GetLastError()) + ", Response: " + response; //--- Create error message
      Print(errMsg);                                  //--- Print error
      FileWriteString(logFileHandle, errMsg + "\n");  //--- Log error
      FileWriteString(logFileHandle, "Raw Response Data (Hex): " + LogCharArray(result) + "\n"); //--- Log raw response
      return errMsg;                                  //--- Return error message
   }
   string response = CharArrayToString(result, 0, WHOLE_ARRAY, CP_UTF8); //--- Convert response to string
   FileWriteString(logFileHandle, "API Response: " + response + "\n"); //--- Log response
   JsonValue jsonObject;                              //--- Declare JSON object
   int index = 0;                                     //--- Initialize parse index
   char charArray[];                                  //--- Declare char array
   int arrayLength = StringToCharArray(response, charArray, 0, WHOLE_ARRAY, CP_UTF8); //--- Convert response to char array
   if (!jsonObject.DeserializeFromArray(charArray, arrayLength, index)) { //--- Parse JSON
      string errMsg = "Error: Failed to parse API response JSON: " + response; //--- Create error message
      Print(errMsg);                                  //--- Print error
      FileWriteString(logFileHandle, errMsg + "\n");  //--- Log error
      return errMsg;                                  //--- Return error message
   }
   JsonValue *error = jsonObject.FindChildByKey("error"); //--- Check for error
   if (error != NULL) {                               //--- Check error exists
      string errMsg = "API Error: " + error["message"].ToString(); //--- Get error message
      Print(errMsg);                                  //--- Print error
      FileWriteString(logFileHandle, errMsg + "\n");  //--- Log error
      return errMsg;                                  //--- Return error message
   }
   string content = jsonObject["choices"][0]["message"]["content"].ToString(); //--- Extract response content
   if (StringLen(content) > 0) {                      //--- Check non-empty content
      StringReplace(content, "\\n", "\n");            //--- Replace escaped newlines
      StringTrimLeft(content);                        //--- Trim left whitespace
      StringTrimRight(content);                       //--- Trim right whitespace
      return content;                                 //--- Return content
   }
   string errMsg = "Error: No content in API response: " + response; //--- Create error message
   Print(errMsg);                                     //--- Print error
   FileWriteString(logFileHandle, errMsg + "\n");     //--- Log error
   return errMsg;                                     //--- Return error message
}

//+------------------------------------------------------------------+
//| Submit user message to ChatGPT                                   |
//+------------------------------------------------------------------+
void SubmitMessage() {
   string prompt = (string)ObjectGetString(0, "ChatGPT_InputEdit", OBJPROP_TEXT); //--- Get user input
   if (StringLen(prompt) > 0) {                      //--- Check non-empty input
      string response = GetChatGPTResponse(prompt);  //--- Get AI response
      Print("User: " + prompt);                      //--- Log user prompt
      Print("AI: " + response);                      //--- Log AI response
      string timestamp = TimeToString(TimeCurrent(), TIME_MINUTES); //--- Get current timestamp
      conversationHistory += "You: " + prompt + "\n" + timestamp + "\nAI: " + response + "\n" + timestamp + "\n\n"; //--- Append to history
      ObjectSetString(0, "ChatGPT_InputEdit", OBJPROP_TEXT, ""); //--- Clear input field
      UpdateResponseDisplay();                       //--- Update display with new content
      scroll_pos = MathMax(0, g_total_height - g_visible_height); //--- Scroll to bottom
      UpdateResponseDisplay();                       //--- Redraw display
      if (scroll_visible) {                          //--- Check scrollbar visible
         UpdateSliderPosition();                     //--- Update slider position
         UpdateButtonColors();                       //--- Update button colors
      }
      FileWriteString(logFileHandle, "Prompt: " + prompt + " | Response: " + response + " | Time: " + timestamp + "\n"); //--- Log interaction
      ChartRedraw();                                 //--- Redraw chart
   }
}
```

Here, we implement the core API interaction and message handling functions. We first define the "BuildMessagesFromHistory" function, where we construct a JSON array for API requests by parsing "conversationHistory", iterating through it to extract user ("You: ") and AI ("AI: ") messages with [StringFind](https://www.mql5.com/en/docs/strings/stringfind) and [StringSubstr](https://www.mql5.com/en/docs/strings/stringsubstr), skipping timestamps and empty lines, and using "JsonEscape" to format content into JSON objects with roles ("user" or "assistant"), appending the new user prompt as the final message, resulting in a properly formatted array for multi-turn conversations.

In the "GetChatGPTResponse" function, we create a JSON request with "BuildMessagesFromHistory", the "OpenAI\_Model", and "MaxResponseLength", convert it to a char array with [StringToCharArray](https://www.mql5.com/en/docs/convert/stringtochararray), set headers with the "OpenAI\_API\_Key", and send a [POST](https://en.wikipedia.org/wiki/POST_(HTTP) "https://en.wikipedia.org/wiki/POST_(HTTP)") request to "OpenAI\_Endpoint" using the [WebRequest](https://www.mql5.com/en/docs/network/webrequest) function. We handle responses by checking for [HTTP](https://en.wikipedia.org/wiki/HTTP "https://en.wikipedia.org/wiki/HTTP") errors (non-200 status), logging raw data and errors with [FileWriteString](https://www.mql5.com/en/docs/files/filewritestring) to "logFileHandle", parsing the JSON response with "JsonValue::DeserializeFromArray", checking for API errors, and extracting the content from "choices\[0\]\[message\]\[content\]", unescaping newlines and trimming whitespace before returning it, just like we did with the previous version.

In the "SubmitMessage" function, we retrieve the user input from "ChatGPT\_InputEdit" with [ObjectGetString](https://www.mql5.com/en/docs/objects/objectgetstring), call "GetChatGPTResponse" if non-empty, log the prompt and response with [Print](https://www.mql5.com/en/docs/common/print), append to "conversationHistory" with timestamps from [TimeCurrent](https://www.mql5.com/en/docs/dateandtime/timecurrent), clear the input field with [ObjectSetString](https://www.mql5.com/en/docs/objects/objectsetstring), update the display with "UpdateResponseDisplay", scroll to the bottom by setting "scroll\_pos", and update scrollbar visuals if needed, logging the interaction. This creates a system for managing AI conversations, API communication, and updating the chat UI dynamically. We can call this function when we click on the send message button as follows.

```
//+------------------------------------------------------------------+
//| Chart event handler for ChatGPT UI                               |
//+------------------------------------------------------------------+
void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam) {
   if (id == CHARTEVENT_OBJECT_CLICK && sparam == "ChatGPT_SubmitButton") { //--- Handle submit button click
      SubmitMessage();                              //--- Submit user message
   }
}
```

To handle chart interactions, we use the [OnChartEvent](https://www.mql5.com/en/docs/event_handlers/onchartevent) event handler as our event listener. When the event is a click on our button, we call our function to send the prompt. Here is a visualization of what we get.

![OUR FIRST PROMPT](https://c.mql5.com/2/172/Screenshot_2025-09-27_180931.png)

From the image, we can see that we have the conversation being longer, and intuitive with the user conversation being added on the right and AI on the left, all with timestamps. What now remains is ensuring an interactive display to enable scrollbar dragging and hover states on the buttons we added. Here is the full logic we used to achieve that.

```
//+------------------------------------------------------------------+
//| Chart event handler for ChatGPT UI                               |
//+------------------------------------------------------------------+
void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam) {
   int displayX = g_mainX + g_sidePadding;          //--- Calculate display x position
   int displayY = g_mainY + g_headerHeight + g_padding; //--- Calculate display y position
   int displayW = g_mainWidth - 2 * g_sidePadding;  //--- Calculate display width
   int displayH = g_displayHeight;                  //--- Set display height
   int clearX = g_mainX + g_mainWidth - 100 - g_sidePadding; //--- Calculate clear button x
   int clearY = g_mainY + 4;                        //--- Set clear button y
   int clearW = 100;                                //--- Set clear button width
   int clearH = g_headerHeight - 8;                 //--- Calculate clear button height
   int new_chat_x = clearX - 100 - g_sidePadding;   //--- Calculate new chat button x
   int new_chat_w = 100;                            //--- Set new chat button width
   int new_chat_h = clearH;                         //--- Set new chat button height
   int sendX = g_mainX + (g_mainWidth - 448 - 10 - 80) / 2 + 448 + 10; //--- Calculate send button x
   int sendY = g_mainY + g_headerHeight + g_padding + g_displayHeight + g_padding; //--- Calculate send button y
   int sendW = 80;                                  //--- Set send button width
   int sendH = g_footerHeight;                      //--- Set send button height
   bool need_scroll = g_total_height > g_visible_height; //--- Check if scrolling needed
   if (id == CHARTEVENT_OBJECT_CLICK && sparam == "ChatGPT_SubmitButton") { //--- Handle submit button click
      SubmitMessage();                              //--- Submit user message
   } else if (id == CHARTEVENT_OBJECT_CLICK && sparam == "ChatGPT_ClearButton") { //--- Handle clear button click
      conversationHistory = "";                     //--- Clear conversation history
      scroll_pos = 0;                               //--- Reset scroll position
      prev_scroll_pos = -1;                         //--- Reset previous scroll position
      UpdateResponseDisplay();                      //--- Update response display
      ObjectSetString(0, "ChatGPT_InputEdit", OBJPROP_TEXT, ""); //--- Clear input field
      ChartRedraw();                                //--- Redraw chart
   } else if (id == CHARTEVENT_OBJECT_CLICK && sparam == "ChatGPT_NewChatButton") { //--- Handle new chat button click
      conversationHistory = "";                     //--- Clear conversation history
      scroll_pos = 0;                               //--- Reset scroll position
      prev_scroll_pos = -1;                         //--- Reset previous scroll position
      UpdateResponseDisplay();                      //--- Update response display
      ObjectSetString(0, "ChatGPT_InputEdit", OBJPROP_TEXT, ""); //--- Clear input field
      ChartRedraw();                                //--- Redraw chart
   } else if (id == CHARTEVENT_OBJECT_CLICK && (sparam == SCROLL_UP_REC || sparam == SCROLL_UP_LABEL)) { //--- Handle scroll up click
      ScrollUp();                                   //--- Scroll up
   } else if (id == CHARTEVENT_OBJECT_CLICK && (sparam == SCROLL_DOWN_REC || sparam == SCROLL_DOWN_LABEL)) { //--- Handle scroll down click
      ScrollDown();                                 //--- Scroll down
   } else if (id == CHARTEVENT_MOUSE_MOVE) {        //--- Handle mouse move events
      int mouseX = (int)lparam;                     //--- Get mouse x coordinate
      int mouseY = (int)dparam;                     //--- Get mouse y coordinate
      bool isOverSend = (mouseX >= sendX && mouseX <= sendX + sendW && mouseY >= sendY && mouseY <= sendY + sendH); //--- Check send button hover
      if (isOverSend && !button_hover) {            //--- Check send button hover start
         ObjectSetInteger(0, "ChatGPT_SubmitButton", OBJPROP_BGCOLOR, button_darker_bg); //--- Set hover background
         button_hover = true;                       //--- Set hover flag
         ChartRedraw();                             //--- Redraw chart
      } else if (!isOverSend && button_hover) {     //--- Check send button hover end
         ObjectSetInteger(0, "ChatGPT_SubmitButton", OBJPROP_BGCOLOR, button_original_bg); //--- Reset background
         button_hover = false;                      //--- Reset hover flag
         ChartRedraw();                             //--- Redraw chart
      }
      bool isOverClear = (mouseX >= clearX && mouseX <= clearX + clearW && mouseY >= clearY && mouseY <= clearY + clearH); //--- Check clear button hover
      if (isOverClear && !clear_hover) {            //--- Check clear button hover start
         ObjectSetInteger(0, "ChatGPT_ClearButton", OBJPROP_BGCOLOR, clear_darker_bg); //--- Set hover background
         clear_hover = true;                        //--- Set hover flag
         ChartRedraw();                             //--- Redraw chart
      } else if (!isOverClear && clear_hover) {     //--- Check clear button hover end
         ObjectSetInteger(0, "ChatGPT_ClearButton", OBJPROP_BGCOLOR, clear_original_bg); //--- Reset background
         clear_hover = false;                       //--- Reset hover flag
         ChartRedraw();                             //--- Redraw chart
      }
      bool isOverNewChat = (mouseX >= new_chat_x && mouseX <= new_chat_x + new_chat_w && mouseY >= clearY && mouseY <= clearY + new_chat_h); //--- Check new chat button hover
      if (isOverNewChat && !new_chat_hover) {       //--- Check new chat button hover start
         ObjectSetInteger(0, "ChatGPT_NewChatButton", OBJPROP_BGCOLOR, new_chat_darker_bg); //--- Set hover background
         new_chat_hover = true;                     //--- Set hover flag
         ChartRedraw();                             //--- Redraw chart
      } else if (!isOverNewChat && new_chat_hover) { //--- Check new chat button hover end
         ObjectSetInteger(0, "ChatGPT_NewChatButton", OBJPROP_BGCOLOR, new_chat_original_bg); //--- Reset background
         new_chat_hover = false;                    //--- Reset hover flag
         ChartRedraw();                             //--- Redraw chart
      }
      bool is_in = (mouseX >= displayX && mouseX <= displayX + displayW &&
                    mouseY >= displayY && mouseY <= displayY + displayH); //--- Check if mouse in display
      if (is_in != mouse_in_display) {              //--- Check display hover change
         mouse_in_display = is_in;                  //--- Update display hover status
         ChartSetInteger(0, CHART_MOUSE_SCROLL, !(mouse_in_display && need_scroll)); //--- Update chart scroll
         if (ScrollbarMode == SCROLL_DYNAMIC_HOVER) { //--- Check dynamic hover mode
            UpdateResponseDisplay();                //--- Update response display
         }
      }
      static int prevMouseState = 0;                //--- Store previous mouse state
      int MouseState = (int)sparam;                 //--- Get current mouse state
      if (prevMouseState == 0 && MouseState == 1 && scroll_visible) { //--- Check slider drag start
         int scrollbar_x = displayX + displayW - 16; //--- Calculate scrollbar x
         int xd_slider = (int)ObjectGetInteger(0, SCROLL_SLIDER, OBJPROP_XDISTANCE); //--- Get slider x
         int yd_slider = (int)ObjectGetInteger(0, SCROLL_SLIDER, OBJPROP_YDISTANCE); //--- Get slider y
         int xs_slider = (int)ObjectGetInteger(0, SCROLL_SLIDER, OBJPROP_XSIZE); //--- Get slider width
         int ys_slider = (int)ObjectGetInteger(0, SCROLL_SLIDER, OBJPROP_YSIZE); //--- Get slider height
         if (mouseX >= xd_slider && mouseX <= xd_slider + xs_slider &&
             mouseY >= yd_slider && mouseY <= yd_slider + ys_slider) { //--- Check slider click
            movingStateSlider = true;              //--- Set drag state
            mlbDownX_Slider = mouseX;              //--- Store mouse x
            mlbDownY_Slider = mouseY;              //--- Store mouse y
            mlbDown_YD_Slider = yd_slider;         //--- Store slider y
            ObjectSetInteger(0, SCROLL_SLIDER, OBJPROP_BGCOLOR, clrDimGray); //--- Set drag color
            ObjectSetInteger(0, SCROLL_SLIDER, OBJPROP_YSIZE, slider_height); //--- Set slider height
            ChartSetInteger(0, CHART_MOUSE_SCROLL, false); //--- Disable chart scroll
            FileWriteString(logFileHandle, "Slider drag started: x=" + IntegerToString(mouseX) + ", y=" + IntegerToString(mouseY) + "\n"); //--- Log drag start
         }
      }
      if (movingStateSlider) {                     //--- Handle slider drag
         int delta_y = mouseY - mlbDownY_Slider;   //--- Calculate y displacement
         int new_y = mlbDown_YD_Slider + delta_y;  //--- Calculate new y position
         int scroll_area_y_min = (g_mainY + g_headerHeight + g_padding) + 16; //--- Set min slider y
         int scroll_area_y_max = (g_mainY + g_headerHeight + g_padding + g_displayHeight - 16 - slider_height); //--- Set max slider y
         new_y = MathMax(scroll_area_y_min, MathMin(new_y, scroll_area_y_max)); //--- Clamp y position
         ObjectSetInteger(0, SCROLL_SLIDER, OBJPROP_YDISTANCE, new_y); //--- Update slider y
         int max_scroll = MathMax(0, g_total_height - g_visible_height); //--- Calculate max scroll
         double scroll_ratio = (double)(new_y - scroll_area_y_min) / (scroll_area_y_max - scroll_area_y_min); //--- Calculate scroll ratio
         int new_scroll_pos = (int)MathRound(scroll_ratio * max_scroll); //--- Calculate new scroll position
         if (new_scroll_pos != scroll_pos) {       //--- Check if scroll changed
            scroll_pos = new_scroll_pos;           //--- Update scroll position
            UpdateResponseDisplay();               //--- Update response display
            if (scroll_visible) {                  //--- Check scrollbar visible
               UpdateSliderPosition();             //--- Update slider position
               UpdateButtonColors();               //--- Update button colors
            }
            FileWriteString(logFileHandle, "Slider dragged: new_scroll_pos=" + IntegerToString(new_scroll_pos) + "\n"); //--- Log drag
         }
         ChartRedraw();                            //--- Redraw chart
      }
      if (MouseState == 0) {                       //--- Handle mouse release
         if (movingStateSlider) {                  //--- Check if dragging
            movingStateSlider = false;             //--- Reset drag state
            if (scroll_visible) {                  //--- Check scrollbar visible
               ObjectSetInteger(0, SCROLL_SLIDER, OBJPROP_BGCOLOR, clrGray); //--- Reset slider color
            }
            ChartSetInteger(0, CHART_MOUSE_SCROLL, !(mouse_in_display && need_scroll)); //--- Restore chart scroll
            FileWriteString(logFileHandle, "Slider drag ended\n"); //--- Log drag end
         }
      }
      prevMouseState = MouseState;                   //--- Update previous mouse state
      static bool prevMouseInsideScrollUp = false;   //--- Track previous scroll up hover
      static bool prevMouseInsideScrollDown = false; //--- Track previous scroll down hover
      static bool prevMouseInsideSlider = false;     //--- Track previous slider hover
      if (scroll_visible) {                          //--- Check scrollbar visible
         int scrollbar_x = displayX + displayW - 16; //--- Calculate scrollbar x
         int button_size = 16;                       //--- Set button size
         int xd_slider = (int)ObjectGetInteger(0, SCROLL_SLIDER, OBJPROP_XDISTANCE); //--- Get slider x
         int yd_slider = (int)ObjectGetInteger(0, SCROLL_SLIDER, OBJPROP_YDISTANCE); //--- Get slider y
         int xs_slider = (int)ObjectGetInteger(0, SCROLL_SLIDER, OBJPROP_XSIZE); //--- Get slider width
         int ys_slider = (int)ObjectGetInteger(0, SCROLL_SLIDER, OBJPROP_YSIZE); //--- Get slider height
         bool isMouseInsideUp = (mouseX >= scrollbar_x && mouseX <= scrollbar_x + 16 &&
                                 mouseY >= displayY &&
                                 mouseY <= displayY + button_size); //--- Check scroll up hover
         bool isMouseInsideDown = (mouseX >= scrollbar_x && mouseX <= scrollbar_x + 16 &&
                                   mouseY >= displayY + g_displayHeight - button_size &&
                                   mouseY <= displayY + g_displayHeight); //--- Check scroll down hover
         bool isMouseInsideSlider = (mouseX >= xd_slider && mouseX <= xd_slider + xs_slider &&
                                     mouseY >= yd_slider && mouseY <= yd_slider + ys_slider); //--- Check slider hover
         if (isMouseInsideUp != prevMouseInsideScrollUp) { //--- Check scroll up hover change
            ObjectSetInteger(0, SCROLL_UP_REC, OBJPROP_BGCOLOR, isMouseInsideUp ? clrSilver : clrGainsboro); //--- Update scroll up color
            prevMouseInsideScrollUp = isMouseInsideUp; //--- Update hover state
            ChartRedraw();                         //--- Redraw chart
         }
         if (isMouseInsideDown != prevMouseInsideScrollDown) { //--- Check scroll down hover change
            ObjectSetInteger(0, SCROLL_DOWN_REC, OBJPROP_BGCOLOR, isMouseInsideDown ? clrSilver : clrGainsboro); //--- Update scroll down color
            prevMouseInsideScrollDown = isMouseInsideDown; //--- Update hover state
            ChartRedraw();                         //--- Redraw chart
         }
         if (isMouseInsideSlider != prevMouseInsideSlider && !movingStateSlider) { //--- Check slider hover change
            ObjectSetInteger(0, SCROLL_SLIDER, OBJPROP_BGCOLOR, isMouseInsideSlider ? clrDarkGray : clrSilver); //--- Update slider color
            prevMouseInsideSlider = isMouseInsideSlider; //--- Update hover state
            ChartRedraw();                         //--- Redraw chart
         }
      }
   } else if (id == CHARTEVENT_MOUSE_WHEEL) {      //--- Handle mouse wheel events
      int mouseX = (int)lparam;                    //--- Get mouse x coordinate
      int mouseY = (int)dparam;                    //--- Get mouse y coordinate
      int delta = (int)sparam;                     //--- Get wheel delta
      bool in_display = (mouseX >= displayX && mouseX <= displayX + displayW &&
                         mouseY >= displayY && mouseY <= displayY + displayH); //--- Check if mouse in display
      if (in_display != mouse_in_display) {        //--- Check display hover change
         mouse_in_display = in_display;            //--- Update display hover
         ChartSetInteger(0, CHART_MOUSE_SCROLL, !(mouse_in_display && need_scroll)); //--- Update chart scroll
         if (ScrollbarMode == SCROLL_DYNAMIC_HOVER) { //--- Check dynamic hover mode
            UpdateResponseDisplay();               //--- Update response display
         }
      }
      if (in_display && need_scroll) {             //--- Check scroll conditions
         int scroll_amount = 30 * (delta > 0 ? -1 : 1); //--- Calculate scroll amount
         scroll_pos = MathMax(0, MathMin(MathMax(0, g_total_height - g_visible_height), scroll_pos + scroll_amount)); //--- Update scroll position
         UpdateResponseDisplay();                  //--- Update response display
         if (scroll_visible) {                     //--- Check scrollbar visible
            UpdateSliderPosition();                //--- Update slider position
            UpdateButtonColors();                  //--- Update button colors
         }
         ChartRedraw();                            //--- Redraw chart
      }
   }
}
```

To achieve full interactivity, in the [OnChartEvent](https://www.mql5.com/en/docs/event_handlers/onchartevent) function, we calculate positions for the display area ("displayX", "displayY", "displayW", "displayH"), clear button ("clearX", "clearY", "clearW", "clearH"), new chat button ("new\_chat\_x", "new\_chat\_w", "new\_chat\_h"), and send button ("sendX", "sendY", "sendW", "sendH") using global layout variables. For click events ( [CHARTEVENT\_OBJECT\_CLICK](https://www.mql5.com/en/docs/constants/chartconstants/enum_chartevents)), we handle "ChatGPT\_ClearButton" and "ChatGPT\_NewChatButton" by clearing "conversationHistory", resetting "scroll\_pos" and "prev\_scroll\_pos", clearing the input field with [ObjectSetString](https://www.mql5.com/en/docs/objects/ObjectSetString), and updating the display with "UpdateResponseDisplay", and scroll buttons ("SCROLL\_UP\_REC", "SCROLL\_UP\_LABEL", "SCROLL\_DOWN\_REC", "SCROLL\_DOWN\_LABEL") by calling the "ScrollUp" or "ScrollDown" functions.

For mouse move events ( [CHARTEVENT\_MOUSE\_MOVE](https://www.mql5.com/en/docs/constants/chartconstants/enum_chartevents)), we detect hover over the send, clear, and new chat buttons, updating their backgrounds ("button\_darker\_bg", "clear\_darker\_bg", "new\_chat\_darker\_bg") with "ObjectSetInteger" when hovered, and check if the mouse is in the display area to toggle "mouse\_in\_display" and update chart scrolling with "ChartSetInteger", refreshing the display in "SCROLL\_DYNAMIC\_HOVER" mode.

We handle slider dragging by detecting clicks on "SCROLL\_SLIDER", setting "movingStateSlider", updating the slider’s y-position with [ObjectSetInteger](https://www.mql5.com/en/docs/objects/objectsetinteger) based on mouse movement, calculating "scroll\_pos" via scroll ratio, and logging with the [FileWriteString](https://www.mql5.com/en/docs/files/filewritestring) function. On mouse release, we reset the drag state and slider color. For mouse wheel events ( [CHARTEVENT\_MOUSE\_WHEEL](https://www.mql5.com/en/docs/constants/chartconstants/enum_chartevents)), we adjust "scroll\_pos" by 30 pixels based on wheel direction, update the display, and refresh scrollbar visuals if visible. We also manage scrollbar hover effects, updating colors for up/down buttons and the slider. Each action triggers [ChartRedraw](https://www.mql5.com/en/docs/chart_operations/ChartRedraw) for visual updates. This makes sure our program supports clicks, hovers, drags, and scrolling. Here is the final outcome.

![UPGRADED VS PREVIOUS VERSION COMPARISON](https://c.mql5.com/2/172/Screenshot_2025-09-27_183224.png)

From the image, we can see that we are able to upgrade the program by adding new elements, displaying a scrollable conversation history, and making the interface interactable, hence achieving our objectives. The thing that remains is backtesting the program, and that is handled in the next section.

### Testing the ChatGPT Program

We did the testing, and below is the compiled visualization in a single [Graphics Interchange Format](https://en.wikipedia.org/wiki/GIF "https://en.wikipedia.org/wiki/GIF") (GIF) bitmap image format.

![BACKTEST GIF](https://c.mql5.com/2/172/CHAT_GPT_PT_3_GIF.gif)

### Conclusion

In conclusion, we’ve enhanced our ChatGPT-integrated program in MQL5, upgrading to a scrollable single chat-oriented UI with dynamic [JSON](https://en.wikipedia.org/wiki/JSON "https://en.wikipedia.org/wiki/JSON") parsing, timestamped conversation history, and interactive controls like submit, clear, and new chat buttons. This system empowers us to engage seamlessly with AI-driven insights for market analysis, maintaining context across multi-turn conversations while optimizing usability with adaptive scrolling and hover effects. In the preceding versions, we will update the display to handle back-and-forth conversations and share live data to get trading insights. Stay tuned.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/19741.zip "Download all attachments in the single ZIP archive")

[ChatGPT\_AI\_EA\_Part\_3.mq5](https://www.mql5.com/en/articles/download/19741/ChatGPT_AI_EA_Part_3.mq5 "Download ChatGPT_AI_EA_Part_3.mq5")(219.2 KB)

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

**[Go to discussion](https://www.mql5.com/en/forum/496699)**

![Price Action Analysis Toolkit Development (Part 43): Candlestick Probability and Breakouts](https://c.mql5.com/2/173/19738-price-action-analysis-toolkit-logo__1.png)[Price Action Analysis Toolkit Development (Part 43): Candlestick Probability and Breakouts](https://www.mql5.com/en/articles/19738)

Enhance your market analysis with the MQL5-native Candlestick Probability EA, a lightweight tool that transforms raw price bars into real-time, instrument-specific probability insights. It classifies Pinbars, Engulfing, and Doji patterns at bar close, uses ATR-aware filtering, and optional breakout confirmation. The EA calculates raw and volume-weighted follow-through percentages, helping you understand each pattern's typical outcome on specific symbols and timeframes. On-chart markers, a compact dashboard, and interactive toggles allow easy validation and focus. Export detailed CSV logs for offline testing. Use it to develop probability profiles, optimize strategies, and turn pattern recognition into a measurable edge.

![Price movement discretization methods in Python](https://c.mql5.com/2/114/Price_Movement_Discretization_Methods_in_Python____LOGO2.png)[Price movement discretization methods in Python](https://www.mql5.com/en/articles/16914)

We will look at price discretization methods using Python + MQL5. In this article, I will share my practical experience developing a Python library that implements a wide range of approaches to bar formation — from classic Volume and Range bars to more exotic methods like Renko and Kagi. We will consider three-line breakout candles and range bars analyzing their statistics and trying to define how else the prices can be represented discretely.

![Developing Advanced ICT Trading Systems: Implementing Signals in the Order Blocks Indicator](https://c.mql5.com/2/100/Final_Logo.png)[Developing Advanced ICT Trading Systems: Implementing Signals in the Order Blocks Indicator](https://www.mql5.com/en/articles/16268)

In this article, you will learn how to develop an Order Blocks indicator based on order book volume (market depth) and optimize it using buffers to improve accuracy. This concludes the current stage of the project and prepares for the next phase, which will include the implementation of a risk management class and a trading bot that uses signals generated by the indicator.

![Building a Professional Trading System with Heikin Ashi (Part 2): Developing an EA](https://c.mql5.com/2/171/18810-building-a-professional-trading-logo.png)[Building a Professional Trading System with Heikin Ashi (Part 2): Developing an EA](https://www.mql5.com/en/articles/18810)

This article explains how to develop a professional Heikin Ashi-based Expert Advisor (EA) in MQL5. You will learn how to set up input parameters, enumerations, indicators, global variables, and implement the core trading logic. You will also be able to run a backtest on gold to validate your work.

[![](https://www.mql5.com/ff/si/0nfwvn6yhmgzf97n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F117%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dorder.expert%26utm_content%3Dorder.freelance%26utm_campaign%3D0622.MQL5.com.Internal&a=tunpwtbhegzufrqocbwiszessdutnobs&s=d9e7484e15300021b4066b1df77a94a1352f9e7c326d5113006bb4f6476bafeb&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=crdwmwemmmitapqbuunwpvyivrequnat&ssn=1769092357206907929&ssn_dr=0&ssn_sr=0&fv_date=1769092357&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F19741&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Building%20AI-Powered%20Trading%20Systems%20in%20MQL5%20(Part%203)%3A%20Upgrading%20to%20a%20Scrollable%20Single%20Chat-Oriented%20UI%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176909235779389752&fz_uniq=5049205979213571845&sv=2552)

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