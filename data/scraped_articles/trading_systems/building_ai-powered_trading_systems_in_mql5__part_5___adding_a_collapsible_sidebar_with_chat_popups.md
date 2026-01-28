---
title: Building AI-Powered Trading Systems in MQL5 (Part 5): Adding a Collapsible Sidebar with Chat Popups
url: https://www.mql5.com/en/articles/20249
categories: Trading Systems, Integration, Expert Advisors
relevance_score: 9
scraped_at: 2026-01-22T17:32:08.317324
---

[![](https://www.mql5.com/ff/sh/wm94j0jmkwd29943z2/ddfa713cb3cdd580c3e81e0e13b5b1b8.jpg)\\
Revised MetaTrader 5 Web Terminal\\
\\
Trade with no restrictions from any mobile device, OS and web browser\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=fkjlpstbxdmrrwpblfatcsdjyrxbizyj&s=f462f051eb7aaec36d6b31792d312d60d3f5a50c83b12d0d66e85d5d61bd941b&uid=&ref=https://www.mql5.com/en/articles/20249&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5049199863180142310)

MetaTrader 5 / Trading systems


### Introduction

In our [previous article (Part 4)](https://www.mql5.com/en/articles/19782), we enhanced the ChatGPT-integrated [MetaQuotes Language 5](https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5 "https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5") (MQL5) program by improving multiline input, adding encrypted chat storage with [AES256](https://en.wikipedia.org/wiki/Advanced_Encryption_Standard "https://en.wikipedia.org/wiki/Advanced_Encryption_Standard") and [ZIP](https://en.wikipedia.org/wiki/ZIP_(file_format) "https://en.wikipedia.org/wiki/ZIP_(file_format)"), and generating trade signals from chart data. In Part 5, we add a collapsible sidebar to the AI-powered trading system. This upgrade introduces a dynamic sidebar that toggles between expanded and contracted states for better screen management, incorporates small and large history [popups](https://en.wikipedia.org/wiki/Popup "https://en.wikipedia.org/wiki/Popup") for quick chat access, and maintains seamless multiline input, encrypted persistence, and signal generation features. We will cover the following topics:

1. [Importance of a Collapsible Sidebar in AI Trading Interfaces](https://www.mql5.com/en/articles/20249#para2)
2. [Implementation in MQL5](https://www.mql5.com/en/articles/20249#para3)
3. [Testing the Collapsible Sidebar](https://www.mql5.com/en/articles/20249#para4)
4. [Conclusion](https://www.mql5.com/en/articles/20249#para5)

By the end, you’ll have an MQL5 AI trading assistant with optimized UI flexibility, ready for customization—let’s dive in!

### Importance of a Collapsible Sidebar in AI Trading Interfaces

A collapsible sidebar in AI trading interfaces, such as our ChatGPT-integrated system, is vital for optimizing screen real estate and enhancing user experience. By allowing the sidebar to [toggle](https://en.wikipedia.org/wiki/Toggle "https://en.wikipedia.org/wiki/Toggle") between an expanded state and a contracted state, we can prioritize chart space when needed or access full navigation details during analysis, improving workflow efficiency without sacrificing functionality. The addition of a small history pop-up on hover in contracted mode and a scrollable, large history pop-up for all chats ensures quick access to persistent conversations, allowing us to reference past AI-driven market insights or signals seamlessly. This supports faster, more informed trading decisions while maintaining a clean, intuitive interface.

Our approach builds on the existing system's strengths, which you know by now, and if not, please refer to the [previous versions](https://www.mql5.com/en/articles/19782) of the series. The collapsible sidebar enhances this by offering flexibility: in expanded mode, it provides clear chat navigation; when contracted, it minimizes intrusion. The small pop-up allows quick history access, while the large pop-up enables comprehensive chat selection, making the transition between modes smooth and intuitive. This design caters to traders needing both compact and detailed views, ensuring the interface adapts to varying needs while preserving the ability to interact with AI-generated insights for market opportunities like trend analysis or reversals. We will also include a larger chat view for dynamic search functions, but for now, keep things simple. Below are visual representations of the intended setups.

Contracted Sidebar with Small History Popup:

![CONTRACTED SIDEBAR WITH SMALL HISTORY POPUP](https://c.mql5.com/2/180/PART_5_1.gif)

Large History Popup:

![LARGE HISTORY POPUP WITH SCROLLBAR](https://c.mql5.com/2/180/PART_5_1_1.gif)

### Implementation in MQL5

To implement the upgrades, we will first declare the objects that we will be adding for the contracted sidebar, the small history pop-up that we want to show on the hovered history button, the collapse button itself, and then the big pop-up showing all the chat histories. The most significant is the scrollbar that we implement for the big popup, so let us start with that, since the rest can be implemented from the prefix we have been using.

```
#define BIG_SCROLL_LEADER "ChatGPT_Big_Scroll_Leader"
#define BIG_SCROLL_UP_REC "ChatGPT_Big_Scroll_Up_Rec"
#define BIG_SCROLL_UP_LABEL "ChatGPT_Big_Scroll_Up_Label"
#define BIG_SCROLL_DOWN_REC "ChatGPT_Big_Scroll_Down_Rec"
#define BIG_SCROLL_DOWN_LABEL "ChatGPT_Big_Scroll_Down_Label"
#define BIG_SCROLL_SLIDER "ChatGPT_Big_Scroll_Slider"
```

Here, we define constants for the large history popup's scrollbar elements using [#define](https://www.mql5.com/en/docs/basis/preprosessor/constant), assigning names like "BIG\_SCROLL\_LEADER" for the track, "BIG\_SCROLL\_UP\_REC" and "BIG\_SCROLL\_UP\_LABEL" for the up button and its arrow, "BIG\_SCROLL\_DOWN\_REC" and "BIG\_SCROLL\_DOWN\_LABEL" for the down button, and "BIG\_SCROLL\_SLIDER" for the draggable thumb, ensuring consistent referencing in object creation and management functions to handle scrolling in the expanded chat history view. The next thing we need to do is add [global variables](https://www.mql5.com/en/docs/basis/variables/global) to control the new elements.

```
color toggle_original_bg = clrLightGray;
color toggle_darker_bg;
bool toggle_hover = false;
color history_original_bg = clrWhite;
color history_darker_bg;
bool history_hover = false;
color see_more_original_bg = clrRoyalBlue;
color see_more_darker_bg;
bool see_more_hover = false;
color big_close_original_bg = clrLightGray;
color big_close_darker_bg;
bool big_close_hover = false;
int expandedSidebarWidth = 150;
int contractedSidebarWidth = 50;
bool sidebarExpanded = true;

bool showing_small_history_popup = false;
int small_popup_x, small_popup_y, small_popup_w, small_popup_h;
string small_popup_objects[];
string small_chat_bgs[];
bool showing_big_history_popup = false;
int big_popup_x, big_popup_y, big_popup_w, big_popup_h;
string big_popup_objects[];
string big_chat_bgs[];
string side_chat_bgs[];
int big_scroll_pos = 0;
bool big_scroll_visible = false;
int big_total_height = 0;
int big_visible_height = 0;
int big_slider_height = 20;
bool big_movingStateSlider = false;
int big_mlbDownX_Slider = 0;
int big_mlbDownY_Slider = 0;
int big_mlbDown_YD_Slider = 0;
bool just_opened_big = false;
bool just_opened_small = false;
```

Going onwards, we define variables for the collapsible sidebar and history popups, starting with colors for the toggle button ("toggle\_original\_bg" as light gray, "toggle\_darker\_bg" for hover state), history button ("history\_original\_bg" as white, "history\_darker\_bg"), "See More" button ("see\_more\_original\_bg" as royal blue, "see\_more\_darker\_bg"), and large popup close button ("big\_close\_original\_bg" as light gray, "big\_close\_darker\_bg"), each paired with hover flags ("toggle\_hover", "history\_hover", "see\_more\_hover", "big\_close\_hover") initialized to false. For sidebar sizing, we set "expandedSidebarWidth" to 150 pixels and "contractedSidebarWidth" to 50 pixels, with "sidebarExpanded" defaulting to true for the expanded state.

We manage two history popups: a small popup with "showing\_small\_history\_popup" flag, coordinates ("small\_popup\_x", "small\_popup\_y"), size ("small\_popup\_w", "small\_popup\_h"), and arrays "small\_popup\_objects" and "small\_chat\_bgs" for objects and chat backgrounds; and a large popup with "showing\_big\_history\_popup" flag, coordinates ("big\_popup\_x", "big\_popup\_y"), size ("big\_popup\_w", "big\_popup\_h"), arrays "big\_popup\_objects" and "big\_chat\_bgs", and scrollbar variables ("big\_scroll\_pos", "big\_scroll\_visible", "big\_total\_height", "big\_visible\_height", "big\_slider\_height" at 20, "big\_movingStateSlider", "big\_mlbDownX\_Slider", "big\_mlbDownY\_Slider", "big\_mlbDown\_YD\_Slider"). The "side\_chat\_bgs" array tracks sidebar chat backgrounds, while "just\_opened\_big" and "just\_opened\_small" flags prevent immediate popup closure on opening, enabling dynamic UI management for collapsible navigation and chat history access. With these variables, we now need to initialize the new interface.

```
int OnInit() {
   //--- initialize the new variables
   toggle_darker_bg = DarkenColor(toggle_original_bg);
   history_darker_bg = DarkenColor(history_original_bg, 0.9);
   see_more_darker_bg = DarkenColor(see_more_original_bg);
   big_close_darker_bg = DarkenColor(big_close_original_bg);
   logFileHandle = FileOpen(LogFileName, FILE_READ | FILE_WRITE | FILE_TXT);
   if (logFileHandle == INVALID_HANDLE) {
      Print("Failed to open log file: ", GetLastError());
      return(INIT_FAILED);
   }
   FileSeek(logFileHandle, 0, SEEK_END);
   uint img_pixels[];
   uint orig_width = 0, orig_height = 0;
   bool image_loaded = ResourceReadImage(resourceImg, img_pixels, orig_width, orig_height);
   if (image_loaded && orig_width > 0 && orig_height > 0) {
      ScaleImage(img_pixels, (int)orig_width, (int)orig_height, 104, 40);
      g_scaled_image_resource = "::ChatGPT_HeaderImageScaled";
      if (ResourceCreate(g_scaled_image_resource, img_pixels, 104, 40, 0, 0, 104, COLOR_FORMAT_ARGB_NORMALIZE)) {
         Print("Scaled image resource created successfully");
      } else {
         Print("Failed to create scaled image resource");
      }
   } else {
      Print("Failed to load original image resource");
   }
   uint img_pixels_logo[];
   uint orig_width_logo = 0, orig_height_logo = 0;
   bool image_loaded_logo = ResourceReadImage(resourceImgLogo, img_pixels_logo, orig_width_logo, orig_height_logo);
   if (image_loaded_logo && orig_width_logo > 0 && orig_height_logo > 0) {
      ScaleImage(img_pixels_logo, (int)orig_width_logo, (int)orig_height_logo, 81, 81);
      g_scaled_sidebar_resource = "::ChatGPT_SidebarImageScaled";
      if (ResourceCreate(g_scaled_sidebar_resource, img_pixels_logo, 81, 81, 0, 0, 81, COLOR_FORMAT_ARGB_NORMALIZE)) {
         Print("Scaled sidebar image resource created successfully");
      } else {
         Print("Failed to create scaled sidebar image resource");
      }
      uint img_pixels_logo_small[];
      ArrayCopy(img_pixels_logo_small, img_pixels_logo);
      ScaleImage(img_pixels_logo_small, 81, 81, 30, 30);
      g_scaled_sidebar_small = "::ChatGPT_SidebarImageSmall";
      if (ResourceCreate(g_scaled_sidebar_small, img_pixels_logo_small, 30, 30, 0, 0, 30, COLOR_FORMAT_ARGB_NORMALIZE)) {
         Print("Scaled small sidebar image resource created successfully");
      } else {
         Print("Failed to create scaled small sidebar image resource");
      }
   } else {
      Print("Failed to load sidebar image resource");
   }
   uint img_pixels_newchat[];
   uint orig_width_newchat = 0, orig_height_newchat = 0;
   bool image_loaded_newchat = ResourceReadImage(resourceNewChat, img_pixels_newchat, orig_width_newchat, orig_height_newchat);
   if (image_loaded_newchat && orig_width_newchat > 0 && orig_height_newchat > 0) {
      ScaleImage(img_pixels_newchat, (int)orig_width_newchat, (int)orig_height_newchat, 30, 30);
      g_scaled_newchat_resource = "::ChatGPT_NewChatIconScaled";
      if (ResourceCreate(g_scaled_newchat_resource, img_pixels_newchat, 30, 30, 0, 0, 30, COLOR_FORMAT_ARGB_NORMALIZE)) {
         Print("Scaled new chat icon resource created successfully");
      } else {
         Print("Failed to create scaled new chat icon resource");
      }
   } else {
      Print("Failed to load new chat icon resource");
   }
   uint img_pixels_clear[];
   uint orig_width_clear = 0, orig_height_clear = 0;
   bool image_loaded_clear = ResourceReadImage(resourceClear, img_pixels_clear, orig_width_clear, orig_height_clear);
   if (image_loaded_clear && orig_width_clear > 0 && orig_height_clear > 0) {
      ScaleImage(img_pixels_clear, (int)orig_width_clear, (int)orig_height_clear, 30, 30);
      g_scaled_clear_resource = "::ChatGPT_ClearIconScaled";
      if (ResourceCreate(g_scaled_clear_resource, img_pixels_clear, 30, 30, 0, 0, 30, COLOR_FORMAT_ARGB_NORMALIZE)) {
         Print("Scaled clear icon resource created successfully");
      } else {
         Print("Failed to create scaled clear icon resource");
      }
   } else {
      Print("Failed to load clear icon resource");
   }
   uint img_pixels_history[];
   uint orig_width_history = 0, orig_height_history = 0;
   bool image_loaded_history = ResourceReadImage(resourceHistory, img_pixels_history, orig_width_history, orig_height_history);
   if (image_loaded_history && orig_width_history > 0 && orig_height_history > 0) {
      ScaleImage(img_pixels_history, (int)orig_width_history, (int)orig_height_history, 30, 30);
      g_scaled_history_resource = "::ChatGPT_HistoryIconScaled";
      if (ResourceCreate(g_scaled_history_resource, img_pixels_history, 30, 30, 0, 0, 30, COLOR_FORMAT_ARGB_NORMALIZE)) {
         Print("Scaled history icon resource created successfully");
      } else {
         Print("Failed to create scaled history icon resource");
      }
   } else {
      Print("Failed to load history icon resource");
   }
   g_sidebarWidth = sidebarExpanded ? expandedSidebarWidth : contractedSidebarWidth;
   g_mainContentX = g_dashboardX + g_sidebarWidth;
   g_dashboardWidth = g_sidebarWidth + g_mainWidth;
   g_mainHeight = g_headerHeight + 2 * g_padding + g_displayHeight + g_footerHeight;
   createRecLabel("ChatGPT_DashboardBg", g_dashboardX, g_mainY, g_dashboardWidth, g_mainHeight, clrWhite, 1, clrLightGray);
   ObjectSetInteger(0, "ChatGPT_DashboardBg", OBJPROP_ZORDER, 0);
   createRecLabel("ChatGPT_SidebarBg", g_dashboardX+2, g_mainY+2, g_sidebarWidth - 2 - 1, g_mainHeight - 2 - 2, clrGainsboro, 1, clrNONE);
   ObjectSetInteger(0, "ChatGPT_SidebarBg", OBJPROP_ZORDER, 0);

   //--- the rest of the functions

   return(INIT_SUCCEEDED);
}
```

In the [OnInit](https://www.mql5.com/en/docs/event_handlers/oninit) function, we initialize the new elements by setting their new colors, scaling the small contracted logos, and initializing the small sidebar setup. We first set darker hover colors for the added buttons, like toggle, history, see more, and large popup close, using "DarkenColor" on their original backgrounds. We then create a resource "g\_scaled\_image\_resource" as "::ChatGPT\_HeaderImageScaled" using [ResourceCreate](https://www.mql5.com/en/docs/common/resourcecreate) in ARGB format, logging success or failure.

Similarly, for the sidebar logo from "resourceImgLogo", we scale to 81x81 for "g\_scaled\_sidebar\_resource" as we had done before, and further to 30x30 for a small version "g\_scaled\_sidebar\_small" which is our interest currently, creating resources and logging. We do the same for the new resources as well, and we will now use this later when needed accordingly. We have added highlights to show the actual changes we have made for clarity. Next, we need to update our dynamic function for the sidebar to now accommodate the logic for contracted states and add the toggle button.

```
void UpdateSidebarDynamic() {
   int total = ObjectsTotal(0, 0, -1);
   for (int j = total - 1; j >= 0; j--) {
      string name = ObjectName(0, j, 0, -1);
      if (StringFind(name, "ChatGPT_NewChatButton") == 0 || StringFind(name, "ChatGPT_ClearButton") == 0 || StringFind(name, "ChatGPT_HistoryButton") == 0 || StringFind(name, "ChatGPT_ChatLabel_") == 0 || StringFind(name, "ChatGPT_ChatBg_") == 0 || StringFind(name, "ChatGPT_SidebarLogo") == 0 || StringFind(name, "ChatGPT_NewChatIcon") == 0 || StringFind(name, "ChatGPT_NewChatLabel") == 0 || StringFind(name, "ChatGPT_ClearIcon") == 0 || StringFind(name, "ChatGPT_ClearLabel") == 0 || StringFind(name, "ChatGPT_HistoryIcon") == 0 || StringFind(name, "ChatGPT_HistoryLabel") == 0 || StringFind(name, "ChatGPT_ToggleButton") == 0) {
         ObjectDelete(0, name);
      }
   }
   ArrayResize(side_chat_bgs, 0);
   int sidebarX = g_dashboardX;
   int itemY = g_mainY + 10;
   string sidebar_logo_resource = sidebarExpanded ? ((StringLen(g_scaled_sidebar_resource) > 0) ? g_scaled_sidebar_resource : resourceImgLogo) : ((StringLen(g_scaled_sidebar_small) > 0) ? g_scaled_sidebar_small : resourceImgLogo);
   int logo_size = sidebarExpanded ? 81 : 30;
   createBitmapLabel("ChatGPT_SidebarLogo", sidebarX + (g_sidebarWidth - logo_size)/2, itemY, logo_size, logo_size, sidebar_logo_resource, clrNONE, CORNER_LEFT_UPPER);
   ObjectSetInteger(0, "ChatGPT_SidebarLogo", OBJPROP_ZORDER, 1);
   itemY += logo_size + 10;
   createButton("ChatGPT_NewChatButton", sidebarX + 5, itemY, g_sidebarWidth - 10, g_buttonHeight, "", clrWhite, 11, new_chat_original_bg, clrRoyalBlue);
   ObjectSetInteger(0, "ChatGPT_NewChatButton", OBJPROP_ZORDER, 1);
   string newchat_icon_resource = (StringLen(g_scaled_newchat_resource) > 0) ? g_scaled_newchat_resource : resourceNewChat;
   int iconX = sidebarExpanded ? sidebarX + 5 + 10 : sidebarX + 5 + (g_sidebarWidth - 10 - 30)/2;
   createBitmapLabel("ChatGPT_NewChatIcon", iconX, itemY + (g_buttonHeight - 30)/2, 30, 30, newchat_icon_resource, clrNONE, CORNER_LEFT_UPPER);
   ObjectSetInteger(0, "ChatGPT_NewChatIcon", OBJPROP_ZORDER, 2);
   ObjectSetInteger(0, "ChatGPT_NewChatIcon", OBJPROP_SELECTABLE, false);
   if (sidebarExpanded) {
      createLabel("ChatGPT_NewChatLabel", sidebarX + 5 + 10 + 30 + 5, itemY + (g_buttonHeight - 20)/2, "New Chat", clrWhite, 11, "Arial", CORNER_LEFT_UPPER);
      ObjectSetInteger(0, "ChatGPT_NewChatLabel", OBJPROP_ZORDER, 2);
      ObjectSetInteger(0, "ChatGPT_NewChatLabel", OBJPROP_SELECTABLE, false);
   }
   itemY += g_buttonHeight + 5;
   createButton("ChatGPT_ClearButton", sidebarX + 5, itemY, g_sidebarWidth - 10, g_buttonHeight, "", clrWhite, 11, clear_original_bg, clrIndianRed);
   ObjectSetInteger(0, "ChatGPT_ClearButton", OBJPROP_ZORDER, 1);
   string clear_icon_resource = (StringLen(g_scaled_clear_resource) > 0) ? g_scaled_clear_resource : resourceClear;
   iconX = sidebarExpanded ? sidebarX + 5 + 10 : sidebarX + 5 + (g_sidebarWidth - 10 - 30)/2;
   createBitmapLabel("ChatGPT_ClearIcon", iconX, itemY + (g_buttonHeight - 30)/2, 30, 30, clear_icon_resource, clrNONE, CORNER_LEFT_UPPER);
   ObjectSetInteger(0, "ChatGPT_ClearIcon", OBJPROP_ZORDER, 2);
   ObjectSetInteger(0, "ChatGPT_ClearIcon", OBJPROP_SELECTABLE, false);
   if (sidebarExpanded) {
      createLabel("ChatGPT_ClearLabel", sidebarX + 5 + 10 + 30 + 5, itemY + (g_buttonHeight - 20)/2, "Clear", clrWhite, 11, "Arial", CORNER_LEFT_UPPER);
      ObjectSetInteger(0, "ChatGPT_ClearLabel", OBJPROP_ZORDER, 2);
      ObjectSetInteger(0, "ChatGPT_ClearLabel", OBJPROP_SELECTABLE, false);
   }
   itemY += g_buttonHeight + 10;
   createButton("ChatGPT_HistoryButton", sidebarX + 5, itemY, g_sidebarWidth - 10, g_buttonHeight, "", clrBlack, 12, history_original_bg, clrGray);
   ObjectSetInteger(0, "ChatGPT_HistoryButton", OBJPROP_ZORDER, 1);
   string history_icon_resource = (StringLen(g_scaled_history_resource) > 0) ? g_scaled_history_resource : resourceHistory;
   iconX = sidebarExpanded ? sidebarX + 5 + 10 : sidebarX + 5 + (g_sidebarWidth - 10 - 30)/2;
   createBitmapLabel("ChatGPT_HistoryIcon", iconX, itemY + (g_buttonHeight - 30)/2, 30, 30, history_icon_resource, clrNONE, CORNER_LEFT_UPPER);
   ObjectSetInteger(0, "ChatGPT_HistoryIcon", OBJPROP_ZORDER, 2);
   ObjectSetInteger(0, "ChatGPT_HistoryIcon", OBJPROP_SELECTABLE, false);
   if (sidebarExpanded) {
      createLabel("ChatGPT_HistoryLabel", sidebarX + 5 + 10 + 30 + 5, itemY + (g_buttonHeight - 20)/2, "History", clrBlack, 12, "Arial", CORNER_LEFT_UPPER);
      ObjectSetInteger(0, "ChatGPT_HistoryLabel", OBJPROP_ZORDER, 2);
      ObjectSetInteger(0, "ChatGPT_HistoryLabel", OBJPROP_SELECTABLE, false);
   }
   itemY += g_buttonHeight + 5;
   if (sidebarExpanded) {
      int numChats = MathMin(ArraySize(chats), 7);
      int chatIndices[7];
      for (int i = 0; i < numChats; i++) {
         chatIndices[i] = ArraySize(chats) - 1 - i;
      }
      for (int i = 0; i < numChats; i++) {
         int chatIdx = chatIndices[i];
         string hashed_id = EncodeID(chats[chatIdx].id);
         string fullText = chats[chatIdx].title + " > " + hashed_id;
         string labelText = fullText;
         if (StringLen(fullText) > 19) {
            labelText = StringSubstr(fullText, 0, 16) + "...";
         }
         string bgName = "ChatGPT_ChatBg_" + hashed_id;
         createRecLabel(bgName, sidebarX + 5 + 10, itemY, g_sidebarWidth - 10 - 10, 25, clrBeige, 1, DarkenColor(clrBeige, 0.9), BORDER_FLAT, STYLE_SOLID);
         ObjectSetInteger(0, bgName, OBJPROP_ZORDER, 1);
         color textColor = (chats[chatIdx].id == current_chat_id) ? clrBlue : clrBlack;
         createLabel("ChatGPT_ChatLabel_" + hashed_id, sidebarX + 10 + 10, itemY + 3, labelText, textColor, 10, "Arial", CORNER_LEFT_UPPER, ANCHOR_LEFT_UPPER);
         ObjectSetInteger(0, "ChatGPT_ChatLabel_" + hashed_id, OBJPROP_ZORDER, 2);
         int size = ArraySize(side_chat_bgs);
         ArrayResize(side_chat_bgs, size + 1);
         side_chat_bgs[size] = bgName;
         itemY += 25 + 5;
      }
   }
   itemY += 10;
   string toggle_text = sidebarExpanded ? "<<" : ">>";
   createButton("ChatGPT_ToggleButton", sidebarX + 5, itemY, g_sidebarWidth - 10, g_buttonHeight, toggle_text, clrBlack, 12, toggle_original_bg, clrGray);
   ObjectSetInteger(0, "ChatGPT_ToggleButton", OBJPROP_ZORDER, 1);
   ChartRedraw();
}
```

In the "UpdateSidebarDynamic" function, we refresh the sidebar's appearance based on its state, either expanded or contracted. We start by clearing existing sidebar objects, identified by prefixes like "ChatGPT\_NewChatButton", and toggle buttons, using [ObjectsTotal](https://www.mql5.com/en/docs/objects/objectstotal) and [ObjectDelete](https://www.mql5.com/en/docs/objects/objectdelete) with "StringFind" checks, and resetting the "side\_chat\_bgs" array. We set the sidebar's x-position to "g\_dashboardX" and initial y-position to "g\_mainY + 10". The logo uses "g\_scaled\_sidebar\_resource" (81x81) when expanded as before or "g\_scaled\_sidebar\_small" (30x30) when contracted, created with "createBitmapLabel" centered via "sidebarX + (g\_sidebarWidth - logo\_size)/2", with z-order 1.

For buttons (New Chat, Clear, History), we create each with "createButton" at width "g\_sidebarWidth - 10", height "g\_buttonHeight", z-order 1, using colors like "new\_chat\_original\_bg", "clear\_original\_bg", and "history\_original\_bg". Their icons are placed with "createBitmapLabel" at "iconX", calculated left-aligned when expanded or centered when contracted, using scaled resources or defaults like "resourceNewChat", set to z-order 2 and non-selectable.

When expanded, we add text labels for each button ("New Chat", "Clear", "History") with "createLabel" in [Arial](https://en.wikipedia.org/wiki/Arial "https://en.wikipedia.org/wiki/Arial") size 11/12, z-order 2, non-selectable, positioned right of icons. If expanded, we display up to seven recent chats from the "chats" array like before. Finally, we add the new toggle button "ChatGPT\_ToggleButton" with "createButton", displaying "<<" when expanded or ">>" when contracted - we'll change to a better view in the next version, in Arial size 12, z-order 1, and redraw the chart with [ChartRedraw](https://www.mql5.com/en/docs/chart_operations/ChartRedraw) to reflect the updated sidebar layout. Now that we have taken care of the toggle button, we need a function to recalculate and set positions/sizes for all the dashboard objects when needed to prevent overlap gaps.

```
void UpdateDashboardPositions() {
   ObjectSetInteger(0, "ChatGPT_DashboardBg", OBJPROP_XSIZE, g_dashboardWidth);
   ObjectSetInteger(0, "ChatGPT_SidebarBg", OBJPROP_XSIZE, g_sidebarWidth - 2 - 1);
   int diff = g_mainContentX - (g_dashboardX + (sidebarExpanded ? contractedSidebarWidth : expandedSidebarWidth));
   for (int i = 0; i < objCount; i++) {
      string obj = dashboardObjects[i];
      if (ObjectFind(0, obj) >= 0) {
         long curX = ObjectGetInteger(0, obj, OBJPROP_XDISTANCE);
         ObjectSetInteger(0, obj, OBJPROP_XDISTANCE, curX + diff);
      }
   }
   uint date_wid, date_hei;
   string dateStr = TimeToString(TimeTradeServer(), TIME_MINUTES);
   TextSetFont("Arial", 12);
   TextGetSize(dateStr, date_wid, date_hei);
   int dateX = g_mainContentX + g_mainWidth / 2 - (int)(date_wid / 2) + 20;
   int dateY = (int)ObjectGetInteger(0, "ChatGPT_DateLabel", OBJPROP_YDISTANCE);
   ObjectSetInteger(0, "ChatGPT_DateLabel", OBJPROP_XDISTANCE, dateX);
   int closeWidth = 100;
   int closeX = g_mainContentX + g_mainWidth - closeWidth - g_sidePadding;
   ObjectSetInteger(0, "ChatGPT_CloseButton", OBJPROP_XDISTANCE, closeX);
   int promptW = g_mainWidth - 2 * g_sidePadding;
   int editX = g_mainContentX + g_sidePadding + g_textPadding;
   g_editW = promptW - 2 * g_textPadding;
   ObjectSetInteger(0, "ChatGPT_PromptEdit", OBJPROP_XDISTANCE, editX);
   ObjectSetInteger(0, "ChatGPT_PromptEdit", OBJPROP_XSIZE, g_editW);
   int buttonW = 140;
   int chartX = g_mainContentX + g_sidePadding;
   int sendX = g_mainContentX + g_mainWidth - g_sidePadding - buttonW;
   ObjectSetInteger(0, "ChatGPT_GetChartButton", OBJPROP_XDISTANCE, chartX);
   ObjectSetInteger(0, "ChatGPT_SendPromptButton", OBJPROP_XDISTANCE, sendX);
   int displayX = g_mainContentX + g_sidePadding;
   int displayW = g_mainWidth - 2 * g_sidePadding;
   ObjectSetInteger(0, "ChatGPT_ResponseBg", OBJPROP_XDISTANCE, displayX);
   ObjectSetInteger(0, "ChatGPT_ResponseBg", OBJPROP_XSIZE, displayW);
   ObjectSetInteger(0, "ChatGPT_PromptBg", OBJPROP_XDISTANCE, displayX);
   ObjectSetInteger(0, "ChatGPT_PromptBg", OBJPROP_XSIZE, displayW);
   int footerY = g_mainY + g_headerHeight + g_padding + g_displayHeight + g_padding;
   ObjectSetInteger(0, "ChatGPT_FooterBg", OBJPROP_XDISTANCE, g_mainContentX);
   ObjectSetInteger(0, "ChatGPT_FooterBg", OBJPROP_XSIZE, g_mainWidth);
   if (ObjectFind(0, "ChatGPT_PromptPlaceholder") >= 0) {
      int labelY = (int)ObjectGetInteger(0, "ChatGPT_PromptPlaceholder", OBJPROP_YDISTANCE);
      ObjectSetInteger(0, "ChatGPT_PromptPlaceholder", OBJPROP_XDISTANCE, editX + 2);
   }
   if (scroll_visible) {
      int displayY = g_mainY + g_headerHeight + g_padding;
      int scrollbar_x = displayX + displayW - 16;
      int button_size = 16;
      ObjectSetInteger(0, SCROLL_LEADER, OBJPROP_XDISTANCE, scrollbar_x);
      ObjectSetInteger(0, SCROLL_UP_REC, OBJPROP_XDISTANCE, scrollbar_x);
      ObjectSetInteger(0, SCROLL_UP_LABEL, OBJPROP_XDISTANCE, scrollbar_x + 2);
      ObjectSetInteger(0, SCROLL_DOWN_REC, OBJPROP_XDISTANCE, scrollbar_x);
      ObjectSetInteger(0, SCROLL_DOWN_LABEL, OBJPROP_XDISTANCE, scrollbar_x + 2);
      ObjectSetInteger(0, SCROLL_SLIDER, OBJPROP_XDISTANCE, scrollbar_x);
      UpdateSliderPosition();
   }
   if (p_scroll_visible) {
      int promptX = g_mainContentX + g_sidePadding;
      int promptY = footerY + g_margin;
      int promptW = g_mainWidth - 2 * g_sidePadding;
      int scrollbar_x = promptX + promptW - 16;
      int button_size = 16;
      ObjectSetInteger(0, P_SCROLL_LEADER, OBJPROP_XDISTANCE, scrollbar_x);
      ObjectSetInteger(0, P_SCROLL_UP_REC, OBJPROP_XDISTANCE, scrollbar_x);
      ObjectSetInteger(0, P_SCROLL_UP_LABEL, OBJPROP_XDISTANCE, scrollbar_x + 2);
      ObjectSetInteger(0, P_SCROLL_DOWN_REC, OBJPROP_XDISTANCE, scrollbar_x);
      ObjectSetInteger(0, P_SCROLL_DOWN_LABEL, OBJPROP_XDISTANCE, scrollbar_x + 2);
      ObjectSetInteger(0, P_SCROLL_SLIDER, OBJPROP_XDISTANCE, scrollbar_x);
      UpdatePromptSliderPosition();
   }
}
```

Here, we create a function to ensure the main content shifts left/right when the sidebar resizes, preventing overlap or gaps. We will call it on toggle for seamless transitions. We call the function "UpdateDashboardPositions". We first update the dashboard background "ChatGPT\_DashboardBg" width to "g\_dashboardWidth" and sidebar background "ChatGPT\_SidebarBg" to "g\_sidebarWidth - 2 - 1" using [ObjectSetInteger](https://www.mql5.com/en/docs/objects/ObjectSetInteger) with [OBJPROP\_XSIZE](https://www.mql5.com/en/docs/constants/objectconstants/enum_object_property#enum_object_property_integer).

We calculate a horizontal shift "diff" as the change in "g\_mainContentX" relative to the dashboard start plus the opposite sidebar width (contracted if expanded, and vice versa), then loop through "objCount" items in "dashboardObjects", checking each with [ObjectFind](https://www.mql5.com/en/docs/objects/objectfind), retrieving its current X via [ObjectGetInteger](https://www.mql5.com/en/docs/objects/objectgetinteger) on "OBJPROP\_XDISTANCE", and offsetting it by "diff" to reposition the main content. It is extra important here to note that we do not delete and recreate the existing main content; we just shift its position, so no flickering is created.

For the date label, we fetch the current server time string with [TimeToString](https://www.mql5.com/en/docs/convert/timetostring), set Arial font size 12, compute its dimensions using [TextGetSize](https://www.mql5.com/en/docs/objects/textgetsize), center it within the main area adjusted by "g\_mainContentX" and "g\_mainWidth", and set the new X position. We recalculate the close button's X based on "g\_mainContentX" and "g\_mainWidth" minus padding and width, setting it accordingly. For the prompt, we update its width "promptW" and edit field X/width "g\_editW", applying to the prompt edit with "ObjectSetInteger" on "OBJPROP\_XSIZE". Buttons like chart and send get new X positions from "g\_mainContentX" and "g\_sidePadding", set similarly.

Backgrounds for response and prompt are repositioned with updated X and sizes matching "displayX" and "displayW". The footer background shifts to "g\_mainContentX" with a width of "g\_mainWidth". If the prompt placeholder exists, its X is updated to match the edit field's. If the main scrollbar is visible, we recalculate its X as "displayX + displayW - 16" and update all components (leader, up/down rectangles/labels, slider) X positions, then call "UpdateSliderPosition". Similarly, for the prompt scrollbar, we compute its X from "promptX + promptW - 16" and adjust all its elements, invoking "UpdatePromptSliderPosition" to maintain alignment during state changes. So far, so good. We now need to create the pop-ups. We will need new functions for that.

```
void ShowSmallHistoryPopup(int button_y) {
   ArrayResize(small_popup_objects, 0);
   ArrayResize(small_chat_bgs, 0);
   long history_x = ObjectGetInteger(0, "ChatGPT_HistoryButton", OBJPROP_XDISTANCE);
   long history_w = ObjectGetInteger(0, "ChatGPT_HistoryButton", OBJPROP_XSIZE);
   int popup_x = (int)(history_x + history_w);
   int popup_y = button_y;
   int popup_w = 200;
   int item_height = 25;
   int num_chats = MathMin(ArraySize(chats), 7);
   int items_h = num_chats * (item_height + 5) - 5;
   int popup_h = items_h + g_buttonHeight + 20;
   string popup_bg = "ChatGPT_SmallHistoryBg";
   createRecLabel(popup_bg, popup_x, popup_y, popup_w, popup_h, clrWhite, 1, clrLightGray);
   int size = ArraySize(small_popup_objects);
   ArrayResize(small_popup_objects, size + 1);
   small_popup_objects[size] = popup_bg;
   int item_y = popup_y + 10;
   for (int i = 0; i < num_chats; i++) {
      int chatIdx = ArraySize(chats) - 1 - i;
      string hashed_id = EncodeID(chats[chatIdx].id);
      string labelText = chats[chatIdx].title;
      if (StringLen(labelText) > 25) labelText = StringSubstr(labelText, 0, 22) + "...";
      string bgName = "ChatGPT_SmallChatBg_" + hashed_id;
      createRecLabel(bgName, popup_x + 10, item_y, popup_w - 20, item_height, clrBeige, 1, DarkenColor(clrBeige, 0.9), BORDER_FLAT, STYLE_SOLID);
      string labelName = "ChatGPT_SmallChatLabel_" + hashed_id;
      createLabel(labelName, popup_x + 20, item_y + 3, labelText, clrBlack, 10, "Arial", CORNER_LEFT_UPPER, ANCHOR_LEFT_UPPER);
      size = ArraySize(small_popup_objects);
      ArrayResize(small_popup_objects, size + 2);
      small_popup_objects[size] = bgName;
      small_popup_objects[size + 1] = labelName;
      int bg_size = ArraySize(small_chat_bgs);
      ArrayResize(small_chat_bgs, bg_size + 1);
      small_chat_bgs[bg_size] = bgName;
      item_y += item_height + 5;
   }
   string see_more = "ChatGPT_SeeMoreButton";
   createButton(see_more, popup_x + 10, item_y, popup_w - 20, g_buttonHeight, "See All", clrWhite, 11, see_more_original_bg, clrDarkBlue);
   size = ArraySize(small_popup_objects);
   ArrayResize(small_popup_objects, size + 1);
   small_popup_objects[size] = see_more;
   small_popup_x = popup_x;
   small_popup_y = popup_y;
   small_popup_w = popup_w;
   small_popup_h = popup_h;
   showing_small_history_popup = true;
   just_opened_small = true;
}

void DeleteSmallHistoryPopup() {
   for (int i = 0; i < ArraySize(small_popup_objects); i++) {
      ObjectDelete(0, small_popup_objects[i]);
   }
   ArrayResize(small_popup_objects, 0);
   ArrayResize(small_chat_bgs, 0);
   showing_small_history_popup = false;
}

void ShowBigHistoryPopup() {
   ArrayResize(big_popup_objects, 0);
   ArrayResize(big_chat_bgs, 0);
   big_scroll_pos = 0;
   big_scroll_visible = false;
   int popup_w = g_mainWidth - 20;
   int item_height = 25;
   int num_chats = ArraySize(chats);
   big_total_height = num_chats * (item_height + 5) - 5;
   int max_h = g_displayHeight - 20;
   int content_h = max_h - 40 - 10;
   big_visible_height = content_h - 35;
   int popup_h = MathMin(big_total_height +40+20 + 10, max_h);
   big_popup_w = popup_w;
   big_popup_h = popup_h;
   big_popup_x = g_mainContentX + 10;
   big_popup_y = g_mainY + g_headerHeight + g_padding + 10;
   string popup_bg = "ChatGPT_BigHistoryBg";
   createRecLabel(popup_bg, big_popup_x, big_popup_y, popup_w, popup_h, C'250,250,250', 1, clrDodgerBlue);
   int size = ArraySize(big_popup_objects);
   ArrayResize(big_popup_objects, size + 1);
   big_popup_objects[size] = popup_bg;
   string close_button = "ChatGPT_BigCloseButton";
   createButton(close_button, big_popup_x + popup_w - 80 - 10 + 40, big_popup_y + 5, 40, 30, "r", clrRed, 13, big_close_original_bg, clrGray,"Webdings");
   size = ArraySize(big_popup_objects);
   ArrayResize(big_popup_objects, size + 1);
   big_popup_objects[size] = close_button;
   int content_y = big_popup_y + 40;
   string content_bg = "ChatGPT_BigContentBg";
   createRecLabel(content_bg, big_popup_x + 10, content_y, popup_w - 20, content_h, clrWhite, 1, clrGainsboro, BORDER_FLAT, STYLE_SOLID);
   size = ArraySize(big_popup_objects);
   ArrayResize(big_popup_objects, size + 1);
   big_popup_objects[size] = content_bg;
   bool need_big_scroll = big_total_height > big_visible_height;
   int reserved_w = need_big_scroll ? 16 : 0;
   int item_w = popup_w - 20 - 20 - reserved_w;
   UpdateBigHistoryDisplay();
   if (need_big_scroll) {
      CreateBigScrollbar();
      big_scroll_visible = true;
      UpdateBigSliderPosition();
      UpdateBigButtonColors();
   }
   showing_big_history_popup = true;
   just_opened_big = true;
   ChartRedraw();
}

void CreateBigScrollbar() {
   int scrollbar_x = big_popup_x + big_popup_w - 10 - 16;
   int scrollbar_y = big_popup_y + 40 + 16;
   int scrollbar_width = 16;
   int scrollbar_height = big_popup_h - 40 - 10 - 2 * 16;
   int button_size = 16;
   createRecLabel(BIG_SCROLL_LEADER, scrollbar_x, scrollbar_y, scrollbar_width, scrollbar_height, C'220,220,220', 1, clrGainsboro, BORDER_FLAT, STYLE_SOLID, CORNER_LEFT_UPPER);
   createRecLabel(BIG_SCROLL_UP_REC, scrollbar_x, big_popup_y + 40, scrollbar_width, button_size, clrGainsboro, 1, clrGainsboro, BORDER_FLAT, STYLE_SOLID, CORNER_LEFT_UPPER);
   createLabel(BIG_SCROLL_UP_LABEL, scrollbar_x + 2, big_popup_y + 40 + -2, CharToString(0x35), clrDimGray, getFontSizeByDPI(10), "Webdings", CORNER_LEFT_UPPER);
   createRecLabel(BIG_SCROLL_DOWN_REC, scrollbar_x, big_popup_y + 40 + (big_popup_h - 40 - 10) - button_size, scrollbar_width, button_size, clrGainsboro, 1, clrGainsboro, BORDER_FLAT, STYLE_SOLID, CORNER_LEFT_UPPER);
   createLabel(BIG_SCROLL_DOWN_LABEL, scrollbar_x + 2, big_popup_y + 40 + (big_popup_h - 40 - 10) - button_size + -2, CharToString(0x36), clrDimGray, getFontSizeByDPI(10), "Webdings", CORNER_LEFT_UPPER);
   big_slider_height = CalculateBigSliderHeight();
   createRecLabel(BIG_SCROLL_SLIDER, scrollbar_x, big_popup_y + 40 + (big_popup_h - 40 - 10) - button_size - big_slider_height, scrollbar_width, big_slider_height, clrSilver, 1, clrGainsboro, BORDER_FLAT, STYLE_SOLID, CORNER_LEFT_UPPER);
   int size = ArraySize(big_popup_objects);
   ArrayResize(big_popup_objects, size + 6);
   big_popup_objects[size] = BIG_SCROLL_LEADER;
   big_popup_objects[size + 1] = BIG_SCROLL_UP_REC;
   big_popup_objects[size + 2] = BIG_SCROLL_UP_LABEL;
   big_popup_objects[size + 3] = BIG_SCROLL_DOWN_REC;
   big_popup_objects[size + 4] = BIG_SCROLL_DOWN_LABEL;
   big_popup_objects[size + 5] = BIG_SCROLL_SLIDER;
}

int CalculateBigSliderHeight() {
   int scroll_area_height = big_popup_h - 40 - 25 - 2 * 16;
   int slider_min_height = 20;
   if (big_total_height <= big_visible_height) return scroll_area_height;
   double visible_ratio = (double)big_visible_height / big_total_height;
   int height = (int)MathFloor(scroll_area_height * visible_ratio);
   return MathMax(slider_min_height, height);
}

void UpdateBigSliderPosition() {
   int scrollbar_x = big_popup_x + big_popup_w - 10 - 16;
   int scrollbar_y = big_popup_y + 40 + 16;
   int scroll_area_height = big_popup_h - 40 - 25 - 2 * 16;
   int max_scroll = MathMax(0, big_total_height - big_visible_height);
   if (max_scroll <= 0) return;
   double scroll_ratio = (double)big_scroll_pos / max_scroll;
   int scroll_area_y_max = scrollbar_y + scroll_area_height - big_slider_height;
   int scroll_area_y_min = scrollbar_y;
   int new_y = scroll_area_y_min + (int)(scroll_ratio * (scroll_area_y_max - scroll_area_y_min));
   new_y = MathMax(scroll_area_y_min, MathMin(new_y, scroll_area_y_max));
   ObjectSetInteger(0, BIG_SCROLL_SLIDER, OBJPROP_YDISTANCE, new_y);
}

void UpdateBigButtonColors() {
   int max_scroll = MathMax(0, big_total_height - big_visible_height);
   if (big_scroll_pos == 0) {
      ObjectSetInteger(0, BIG_SCROLL_UP_LABEL, OBJPROP_COLOR, clrSilver);
   } else {
      ObjectSetInteger(0, BIG_SCROLL_UP_LABEL, OBJPROP_COLOR, clrDimGray);
   }
   if (big_scroll_pos == max_scroll) {
      ObjectSetInteger(0, BIG_SCROLL_DOWN_LABEL, OBJPROP_COLOR, clrSilver);
   } else {
      ObjectSetInteger(0, BIG_SCROLL_DOWN_LABEL, OBJPROP_COLOR, clrDimGray);
   }
}

void BigScrollUp() {
   if (big_scroll_pos > 0) {
      big_scroll_pos = MathMax(0, big_scroll_pos - 30);
      UpdateBigHistoryDisplay();
      if (big_scroll_visible) {
         UpdateBigSliderPosition();
         UpdateBigButtonColors();
      }
   }
}

void BigScrollDown() {
   int max_scroll = MathMax(0, big_total_height - big_visible_height);
   if (big_scroll_pos < max_scroll) {
      big_scroll_pos = MathMin(max_scroll, big_scroll_pos + 30);
      UpdateBigHistoryDisplay();
      if (big_scroll_visible) {
         UpdateBigSliderPosition();
         UpdateBigButtonColors();
      }
   }
}

void UpdateBigHistoryDisplay() {
   int total = ObjectsTotal(0, 0, -1);
   for (int j = total - 1; j >= 0; j--) {
      string name = ObjectName(0, j, 0, -1);
      if (StringFind(name, "ChatGPT_BigChatBg_") == 0 || StringFind(name, "ChatGPT_BigChatLabel_") == 0) {
         ObjectDelete(0, name);
      }
   }
   int content_y = big_popup_y + 40;
   int content_h = big_popup_h - 40 - 10;
   int item_height = 25;
   int num_chats = ArraySize(chats);
   big_total_height = num_chats * (item_height + 5) - 5;
   bool need_big_scroll = big_total_height > big_visible_height;
   int reserved_w = need_big_scroll ? 16 : 0;
   int item_w = big_popup_w - 20 - 20 - reserved_w;
   int item_y = content_y + 10 - big_scroll_pos;
   int end_y = content_y + content_h - 25;
   int start_idx = 0;
   int current_h = 0;
   for (int i = 0; i < num_chats; i++) {
      if (current_h >= big_scroll_pos) {
         start_idx = i;
         item_y = content_y + 10 + (current_h - big_scroll_pos);
         break;
      }
      current_h += item_height + 5;
   }
   int visible_count = 0;
   current_h = 0;
   for (int i = start_idx; i < num_chats; i++) {
      if (current_h + item_height > big_visible_height) break;
      current_h += item_height + 5;
      visible_count++;
   }
   for (int i = 0; i < visible_count; i++) {
      int chatIdx = num_chats - 1 - (start_idx + i);
      string hashed_id = EncodeID(chats[chatIdx].id);
      string fullText = chats[chatIdx].title + " > " + hashed_id;
      string labelText = fullText;
      if (StringLen(fullText) > 35) {
         labelText = StringSubstr(fullText, 0, 32) + "...";
      }
      string bgName = "ChatGPT_BigChatBg_" + hashed_id;
      if (item_y >= content_y + 10 && item_y < end_y) {
         createRecLabel(bgName, big_popup_x + 20, item_y, item_w, item_height, clrBeige, 1, DarkenColor(clrBeige, 0.9), BORDER_FLAT, STYLE_SOLID);
      }
      string labelName = "ChatGPT_BigChatLabel_" + hashed_id;
      if (item_y >= content_y + 10 && item_y < end_y) {
         createLabel(labelName, big_popup_x + 30, item_y + 3, labelText, clrBlack, 10, "Arial", CORNER_LEFT_UPPER, ANCHOR_LEFT_UPPER);
      }
      int size = ArraySize(big_popup_objects);
      ArrayResize(big_popup_objects, size + 2);
      big_popup_objects[size] = bgName;
      big_popup_objects[size + 1] = labelName;
      int bg_size = ArraySize(big_chat_bgs);
      ArrayResize(big_chat_bgs, bg_size + 1);
      big_chat_bgs[bg_size] = bgName;
      item_y += item_height + 5;
   }
   ChartRedraw();
}

void DeleteBigHistoryPopup() {
   for (int i = 0; i < ArraySize(big_popup_objects); i++) {
      ObjectDelete(0, big_popup_objects[i]);
   }
   ArrayResize(big_popup_objects, 0);
   ArrayResize(big_chat_bgs, 0);
   showing_big_history_popup = false;
   big_scroll_visible = false;
}
```

We implement the "ShowSmallHistoryPopup" function to display a compact popup for recent chat histories when hovering over the history button in contracted sidebar mode, resetting "small\_popup\_objects" and "small\_chat\_bgs" arrays. We calculate the pop-up's position right of the history button using [ObjectGetInteger](https://www.mql5.com/en/docs/objects/objectgetinteger) for its X and width, setting width to 200 pixels, item height to 25, and dynamically computing height from up to seven recent chats plus a "See All" button. We create the background "ChatGPT\_SmallHistoryBg" with "createRecLabel" in white with light gray border, then loop through recent chats (latest first), generating backgrounds "ChatGPT\_SmallChatBg\_" and labels "ChatGPT\_SmallChatLabel\_" with "createRecLabel" and "createLabel", truncating titles to 22 characters with ellipsis if needed, storing names in arrays for cleanup, and add a "See More" button at the bottom with "createButton" in royal blue.

The "DeleteSmallHistoryPopup" function clears the small popup by deleting all objects in "small\_popup\_objects" with [ObjectDelete](https://www.mql5.com/en/docs/objects/objectdelete), resetting arrays, and setting "showing\_small\_history\_popup" to false. For "ShowBigHistoryPopup", we display a full scrollable history popup, resetting "big\_popup\_objects" and "big\_chat\_bgs", initializing "big\_scroll\_pos" to 0, and setting visibility to false. We compute the popup width as "g\_mainWidth - 20", height clamped to max from total chats using item height 25 plus padding, positioned inset from main content, creating background "ChatGPT\_BigHistoryBg" in light gray with dodger blue border. We add a close button "ChatGPT\_BigCloseButton" with "createButton" using [Webdings](https://en.wikipedia.org/wiki/Webdings "https://en.wikipedia.org/wiki/Webdings") 'r' in red, a content background "ChatGPT\_BigContentBg" in white with gainsboro border, determine if scrolling is needed based on "big\_total\_height" vs. "big\_visible\_height", call "UpdateBigHistoryDisplay" to populate chats, and if scrolling required, invoke "CreateBigScrollbar", set visibility true, update position and colors. We mark as shown with "showing\_big\_history\_popup" true and "just\_opened\_big" true, redrawing with the [ChartRedraw](https://www.mql5.com/en/docs/chart_operations/ChartRedraw) function.

The "CreateBigScrollbar" function builds the large pop-up's scrollbar similar to others, using "createRecLabel" for leader, up/down rectangles in gainsboro, labels with Webdings arrows in dim gray, and a slider in silver with gainsboro border, calculating positions from "big\_popup\_x/y/w/h", and appends names to "big\_popup\_objects" for management. "CalculateBigSliderHeight" computes the large slider height proportionally from visible to total height, with min 20 pixels, returning the full area if no scroll is needed. "UpdateBigSliderPosition" adjusts the large slider's y based on "big\_scroll\_pos" ratio over max scroll, clamping within the scroll area using "MathMax" and "MathMin", set with the [ObjectSetInteger](https://www.mql5.com/en/docs/objects/ObjectSetInteger) function. "UpdateBigButtonColors" grays out up/down arrows in silver if at top/bottom or dim gray if scrollable. "BigScrollUp" and "BigScrollDown" decrement/increment "big\_scroll\_pos" by 30, clamped to 0/max, then update display with "UpdateBigHistoryDisplay" and, if visible, position/colors with "UpdateBigSliderPosition" and "UpdateBigButtonColors".

In "UpdateBigHistoryDisplay", we clear existing big chat backgrounds and labels with [ObjectsTotal](https://www.mql5.com/en/docs/objects/objectstotal) and "ObjectDelete" using the [StringFind](https://www.mql5.com/en/docs/strings/stringfind) function. We recalculate "big\_total\_height" from chat count, determine scrolling need and reserved width, set item width accordingly, compute start index and y from "big\_scroll\_pos", count visible items within "big\_visible\_height", then loop to create backgrounds "ChatGPT\_BigChatBg\_" and labels "ChatGPT\_BigChatLabel\_" for visible chats (latest first), truncating titles to 32 characters with ellipsis, storing in arrays, positioned only if within view bounds, and redraw with "ChartRedraw". The "DeleteBigHistoryPopup" function removes all objects in "big\_popup\_objects" with [ObjectDelete](https://www.mql5.com/en/docs/objects/objectdelete), resets arrays, sets "showing\_big\_history\_popup" and "big\_scroll\_visible" to false for cleanup. The next thing we need to do is call these functions in the chart event where applicable.

```
void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam) {

//--- most relevant events

   if (id == CHARTEVENT_CLICK) {
      int clickX = (int)lparam;
      int clickY = (int)dparam;
      if (showing_big_history_popup) {
         if (just_opened_big) {
            just_opened_big = false;
         } else if (clickX < big_popup_x || clickX > big_popup_x + big_popup_w || clickY < big_popup_y || clickY > big_popup_y + big_popup_h) {
            DeleteBigHistoryPopup();
            ChartRedraw();
         }
      }
      if (showing_small_history_popup) {
         if (just_opened_small) {
            just_opened_small = false;
         } else if (clickX < small_popup_x || clickX > small_popup_x + small_popup_w || clickY < small_popup_y || clickY > small_popup_y + small_popup_h) {
            DeleteSmallHistoryPopup();
            ChartRedraw();
         }
      }
      return;
   }

   else if (id == CHARTEVENT_OBJECT_CLICK && sparam == "ChatGPT_ToggleButton") {
      sidebarExpanded = !sidebarExpanded;
      g_sidebarWidth = sidebarExpanded ? expandedSidebarWidth : contractedSidebarWidth;
      g_mainContentX = g_dashboardX + g_sidebarWidth;
      g_dashboardWidth = g_sidebarWidth + g_mainWidth;
      if (showing_big_history_popup) DeleteBigHistoryPopup();
      if (showing_small_history_popup) DeleteSmallHistoryPopup();
      UpdateSidebarDynamic();
      UpdateDashboardPositions();
      UpdateResponseDisplay();
      UpdatePromptDisplay();
      ChartRedraw();
   } else if (id == CHARTEVENT_OBJECT_CLICK && sparam == "ChatGPT_HistoryButton" && sidebarExpanded) {
      ShowBigHistoryPopup();
      just_opened_big = true;
   } else if (id == CHARTEVENT_OBJECT_CLICK && sparam == "ChatGPT_SeeMoreButton") {
      DeleteSmallHistoryPopup();
      ShowBigHistoryPopup();
      just_opened_big = true;
   } else if (id == CHARTEVENT_OBJECT_CLICK && sparam == "ChatGPT_BigCloseButton") {
      DeleteBigHistoryPopup();
      ChartRedraw();
   }

   else if (id == CHARTEVENT_OBJECT_CLICK && (sparam == BIG_SCROLL_UP_REC || sparam == BIG_SCROLL_UP_LABEL)) {
      BigScrollUp();
   } else if (id == CHARTEVENT_OBJECT_CLICK && (sparam == BIG_SCROLL_DOWN_REC || sparam == BIG_SCROLL_DOWN_LABEL)) {
      BigScrollDown();
   }

//--- the rest of the hover and mouse move logic implementation for new elements is the same

}
```

We use the [OnChartEvent](https://www.mql5.com/en/docs/event_handlers/onchartevent) event handler to handle user interactions for the collapsible sidebar and history popups, focusing on clicks outside popups and specific object clicks. For general clicks ( [CHARTEVENT\_CLICK](https://www.mql5.com/en/docs/constants/chartconstants/enum_chartevents)), we capture coordinates as "clickX" and "clickY", and if the large popup is shown but not just opened (checking "just\_opened\_big" to prevent accidental immediate closure), verify if the click is outside its bounds and call "DeleteBigHistoryPopup" followed by [ChartRedraw](https://www.mql5.com/en/docs/chart_operations/ChartRedraw) to close it. Similarly, for the small popup, use "just\_opened\_small" and "DeleteSmallHistoryPopup". This is very important because we want that when we click outside the popups and they are open, we close them. Thought it was a genius move to have an 'off-click' event listener, though not necessary.

For object clicks ( [CHARTEVENT\_OBJECT\_CLICK](https://www.mql5.com/en/docs/constants/chartconstants/enum_chartevents)), if on the toggle button "ChatGPT\_ToggleButton", we flip "sidebarExpanded", recalculate "g\_sidebarWidth" to expanded or contracted, update "g\_mainContentX" and "g\_dashboardWidth", close any open popups with "DeleteBigHistoryPopup" and "DeleteSmallHistoryPopup", then refresh the UI by calling the "UpdateSidebarDynamic", "UpdateDashboardPositions", "UpdateResponseDisplay", "UpdatePromptDisplay", and "ChartRedraw" functions. If on the history button "ChatGPT\_HistoryButton" when the sidebar is expanded, we invoke "ShowBigHistoryPopup" and set "just\_opened\_big" to true.

For the "See More" button "ChatGPT\_SeeMoreButton", we delete the small popup with "DeleteSmallHistoryPopup", show the big one, and set "just\_opened\_big". On the big close button "ChatGPT\_BigCloseButton", we call "DeleteBigHistoryPopup" and redraw. For big scrollbar up elements ("BIG\_SCROLL\_UP\_REC" or label), we trigger "BigScrollUp"; similarly for down with "BigScrollDown". The remaining logic for hover effects and mouse moves on new elements follows the same pattern as previous implementations, ensuring consistent interactivity. Upon compilation, we get the following outcome.

![UPDATED SIDEBAR WITH SMALL POPUP](https://c.mql5.com/2/180/Screenshot_2025-11-10_233613.png)

When we have the popups open, we don't want to do updates on the main display. To achieve this, we will need to add a conditional statement in the response update function to prevent unnecessary redraws/conflicts when the popup overlaps main areas, improving stability and reducing flicker during popup use.

```
void UpdateResponseDisplay() {
   if (showing_small_history_popup || showing_big_history_popup) return;

   //--- rest of the function logic

}
```

Here, we just add the conditional statement at the start of the function so that when we have the popups open, we skip the response display updates, just because we create the popups within their area. Upon compilation, we have the following outcome for the big popup.

![BIG CHARTS POPUP WITH HOVER EFFECTS & SCROLLBAR](https://c.mql5.com/2/180/Screenshot_2025-11-10_234636.png)

From the image, we can see that we are able to upgrade the program by adding the new target elements, making the sidebar collapsible, and adding chat popups, hence achieving our objectives. The thing that remains is backtesting the program, and that is handled in the next section.

### Testing the Collapsible Sidebar

We did the testing, and below is the compiled visualization in a single [Graphics Interchange Format](https://en.wikipedia.org/wiki/GIF "https://en.wikipedia.org/wiki/GIF") (GIF) bitmap image format.

![BACKTEST GIF](https://c.mql5.com/2/180/PART_5_TEST.gif)

### Conclusion

In conclusion, we’ve enhanced our AI-powered trading system in [MQL5](https://www.mql5.com/) by introducing a collapsible sidebar that toggles between expanded and contracted states for better screen management, incorporating small hover-based history popups and a large scrollable history view for efficient chat navigation, while preserving multiline input, encrypted persistence, and AI-driven signal generation from chart data. This upgrade optimizes the interface for flexibility, allowing us to maximize chart visibility or access detailed controls as needed, with seamless transitions and intuitive hover effects. In the next parts, we will explore the addition of chat search functionality to search the chats, the addition of delete buttons to delete chats, and advanced signal execution and automation features to further empower AI-assisted trading. Stay tuned.

### Attachments

| S/N | Name | Type | Description |
| --- | --- | --- | --- |
| 1 | AI\_JSON\_FILE.mqh | JSON Class Library | Class for handling JSON serialization and deserialization |
| 2 | AI\_CREATE\_OBJECTS\_FNS.mqh | Object Functions Library | Functions for creating visualization objects like labels and buttons |
| 3 | AI\_ChatGPT\_EA\_Part\_5.mq5 | Expert Advisor | Main Expert Advisor for handling AI integration |

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/20249.zip "Download all attachments in the single ZIP archive")

[AI\_JSON\_FILE.mqh](https://www.mql5.com/en/articles/download/20249/AI_JSON_FILE.mqh "Download AI_JSON_FILE.mqh")(26.62 KB)

[AI\_CREATE\_OBJECTS\_FNS.mqh](https://www.mql5.com/en/articles/download/20249/AI_CREATE_OBJECTS_FNS.mqh "Download AI_CREATE_OBJECTS_FNS.mqh")(11.26 KB)

[AI\_ChatGPT\_EA\_Part\_5.mq5](https://www.mql5.com/en/articles/download/20249/AI_ChatGPT_EA_Part_5.mq5 "Download AI_ChatGPT_EA_Part_5.mq5")(122.12 KB)

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

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/500065)**
(2)


![AMIRREZA REZAZAD](https://c.mql5.com/avatar/2021/10/615E97B1-101B.jpg)

**[AMIRREZA REZAZAD](https://www.mql5.com/en/users/olduz96)**
\|
17 Nov 2025 at 04:33

Thanks for building this AI popup chat. If we can upload chart picture to it(chart with useful data like an price action indicator active on chart) it will enhance AI analysis from market and very helpful if it's possible please tell how can we do that.


![Allan Munene Mutiiria](https://c.mql5.com/avatar/2022/11/637df59b-9551.jpg)

**[Allan Munene Mutiiria](https://www.mql5.com/en/users/29210372)**
\|
17 Nov 2025 at 07:14

**AMIRREZA REZAZAD [#](https://www.mql5.com/en/forum/500065#comment_58527881) :** Thanks for building this AI popup chat. If we can upload chart picture to it(chart with useful data like an price action indicator active on chart) it will enhance AI analysis from market and very helpful if it's possible please tell how can we do that.

Hello. Thanks for the kind feedback.


![From Novice to Expert: Time Filtered Trading](https://c.mql5.com/2/181/20037-from-novice-to-expert-time-logo.png)[From Novice to Expert: Time Filtered Trading](https://www.mql5.com/en/articles/20037)

Just because ticks are constantly flowing in doesn’t mean every moment is an opportunity to trade. Today, we take an in-depth study into the art of timing—focusing on developing a time isolation algorithm to help traders identify and trade within their most favorable market windows. Cultivating this discipline allows retail traders to synchronize more closely with institutional timing, where precision and patience often define success. Join this discussion as we explore the science of timing and selective trading through the analytical capabilities of MQL5.

![Neural Networks in Trading: Memory Augmented Context-Aware Learning for Cryptocurrency Markets (Final Part)](https://c.mql5.com/2/113/Neural_Networks_in_Trading_MacroHFT____LOGO.png)[Neural Networks in Trading: Memory Augmented Context-Aware Learning for Cryptocurrency Markets (Final Part)](https://www.mql5.com/en/articles/16993)

The MacroHFT framework for high-frequency cryptocurrency trading uses context-aware reinforcement learning and memory to adapt to dynamic market conditions. At the end of this article, we will test the implemented approaches on real historical data to assess their effectiveness.

![Blood inheritance optimization (BIO)](https://c.mql5.com/2/120/Blood_inheritance_optimization__LOGO.png)[Blood inheritance optimization (BIO)](https://www.mql5.com/en/articles/17246)

I present to you my new population optimization algorithm - Blood Inheritance Optimization (BIO), inspired by the human blood group inheritance system. In this algorithm, each solution has its own "blood type" that determines the way it evolves. Just as in nature where a child's blood type is inherited according to specific rules, in BIO new solutions acquire their characteristics through a system of inheritance and mutations.

![Analyzing all price movement options on the IBM quantum computer](https://c.mql5.com/2/122/Analysis_of_all_price_movement_options_on_an_IBM_quantum_computer__LOGO.png)[Analyzing all price movement options on the IBM quantum computer](https://www.mql5.com/en/articles/17171)

We will use a quantum computer from IBM to discover all price movement options. Sounds like science fiction? Welcome to the world of quantum computing for trading!

[![](https://www.mql5.com/ff/si/q0vxp9pq0887p07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fvps%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Duse.vps%26utm_content%3Drent.vps%26utm_campaign%3D0622.MQL5.com.Internal&a=rktadgjlwhobyedohbrepzshvpcqrlpo&s=a93cef75a53eb5da24c98e0068b3c2b96015191a0af0d1857f5b4dd22e55e7bf&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=kuwozcblnxgjddhdanltioguevzjpdmi&ssn=1769092326472216549&ssn_dr=0&ssn_sr=0&fv_date=1769092326&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F20249&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Building%20AI-Powered%20Trading%20Systems%20in%20MQL5%20(Part%205)%3A%20Adding%20a%20Collapsible%20Sidebar%20with%20Chat%20Popups%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176909232644388052&fz_uniq=5049199863180142310&sv=2552)

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