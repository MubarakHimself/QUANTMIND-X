---
title: Trading with the MQL5 Economic Calendar (Part 4): Implementing Real-Time News Updates in the Dashboard
url: https://www.mql5.com/en/articles/16386
categories: Trading, Trading Systems, Expert Advisors
relevance_score: 1
scraped_at: 2026-01-23T21:33:18.355949
---

[![](https://www.mql5.com/ff/si/fx5m8s6u6uxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=ysducdhemkrdsdtzzbfkclolrllnhezk&s=33f180a31db6c3b846d77732b0bc78169421a47b8cf9f076ca717f4e4846d1c7&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=hnlnvoufxnorlqsvpxlfujqvwqfsnwmf&ssn=1769193197210003646&ssn_dr=0&ssn_sr=0&fv_date=1769193197&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F16386&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Trading%20with%20the%20MQL5%20Economic%20Calendar%20(Part%204)%3A%20Implementing%20Real-Time%20News%20Updates%20in%20the%20Dashboard%20-%20MQL5%20Articles&scr_res=1920x1080&ac=17691931970778798&fz_uniq=5071922301760385260&sv=2552)

MetaTrader 5 / Trading


### Introduction

In this article, we advance our work on the [MetaQuotes Language 5](https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5 "https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5") (MQL5) [Economic Calendar](https://www.mql5.com/en/book/advanced/calendar) dashboard by adding functionality for live updates, allowing us to maintain a continuously refreshed display of critical economic news events. In the [previous part](https://www.mql5.com/en/articles/16380), we designed and implemented a dashboard panel to filter news based on currency, importance, and time, giving us a tailored view of relevant events. Now, we take it further by enabling real-time updates, ensuring that our calendar displays the latest data for timely decision-making. The topics we will cover include:

1. [Implementing Real-Time News Updates in MQL5](https://www.mql5.com/en/articles/16386#para1)
2. [Conclusion](https://www.mql5.com/en/articles/16386#para2)

This enhancement will transform our dashboard into a dynamic, real-time tool continuously updated with the latest economic events. By implementing live refresh functionality, we will ensure that our calendar remains accurate and relevant, supporting timely trading decisions in [MetaTrader 5](https://www.metatrader5.com/en "https://www.metatrader5.com/en") or the trading terminal.

### Implementing Real-Time News Updates in MQL5

To implement live updates for our dashboard, we need to ensure that news events are stored and compared periodically to detect any changes. This requires maintaining arrays to hold current and previous event data, enabling us to identify updates and reflect them accurately on the dashboard. By doing so, we can ensure that the dashboard dynamically adjusts to display the most recent economic events in real-time. Below, we define the arrays that we will use for this purpose:

```
string current_eventNames_data[];
string previous_eventNames_data[];
```

Here, we define two [string](https://www.mql5.com/en/docs/basis/types/stringconst) arrays, "current\_eventNames\_data" and "previous\_eventNames\_data", which we will use to manage and compare economic event data for live updates to the dashboard. The array "current\_eventNames\_data" will store the latest events retrieved from the economic calendar, while "previous\_eventNames\_data" will hold the data from the last update cycle. By comparing these two arrays, we can identify any changes or new additions to the events, allowing us to update the dashboard dynamically.

Using these arrays, we will need to get the current values on every event selected in the initialization section and store them in the current data holder array, and then later on copy them into the previous holder, which we will use to make the comparison on the next price tick.

```
ArrayResize(current_eventNames_data,ArraySize(current_eventNames_data)+1);
current_eventNames_data[ArraySize(current_eventNames_data)-1] = event.name;
```

Here, we dynamically expand the "current\_eventNames\_data" array and add a new event name to it. We use the function [ArrayResize](https://www.mql5.com/en/docs/array/arrayresize) to increase the size of the array by one, making space for a new entry. After resizing, we assign the event's name to the last index of the array using the expression "current\_eventNames\_data\[ArraySize(current\_eventNames\_data)-1\]". This process ensures that each new event name retrieved from the economic calendar is stored in the array, enabling us to maintain an up-to-date list of events for further processing and comparison.

However, before adding the events to the array, we need to ensure that we start fresh, meaning that we need an empty array.

```
ArrayFree(current_eventNames_data);
```

Here, we use the [ArrayFree](https://www.mql5.com/en/docs/array/arrayfree) function to clear all elements from the "current\_eventNames\_data" array, effectively resetting it to an empty state. This is essential for ensuring that the array does not retain outdated data from previous iterations, thus preparing it to store a fresh set of event names during the processing cycle. After filling the array, we then need to copy it into the previous holder and use it later to make the comparison.

```
Print("CURRENT EVENT NAMES DATA SIZE = ",ArraySize(current_eventNames_data));
ArrayPrint(current_eventNames_data);
Print("PREVIOUS EVENT NAMES DATA SIZE (Before) = ",ArraySize(previous_eventNames_data));
ArrayCopy(previous_eventNames_data,current_eventNames_data);
Print("PREVIOUS EVENT NAMES DATA SIZE (After) = ",ArraySize(previous_eventNames_data));
ArrayPrint(previous_eventNames_data);
```

Here, we log and manage the transition of data between the "current\_eventNames\_data" and "previous\_eventNames\_data" arrays. First, we use the [Print](https://www.mql5.com/en/docs/common/print) function to display the size of the "current\_eventNames\_data" array, providing visibility into the number of event names stored at that moment. We then call the [ArrayPrint](https://www.mql5.com/en/docs/array/arrayprint) function to output the array's contents for further verification. Next, we log the size of the "previous\_eventNames\_data" array before copying, giving us a baseline for comparison.

Using the [ArrayCopy](https://www.mql5.com/en/docs/array/arraycopy) function, we copy the contents of "current\_eventNames\_data" into "previous\_eventNames\_data", effectively transferring the latest event names for future comparisons. We then print the size of the "previous\_eventNames\_data" array after the copy operation to confirm the successful update. Finally, we call the ArrayPrint function to output the updated contents of "previous\_eventNames\_data", ensuring that the data transfer is accurate and complete. Those are the changes that we need on the [OnInit](https://www.mql5.com/en/docs/event_handlers/oninit) event handler to store the initial events. Let us highlight them for clarity.

```
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit(){

   //---

   ArrayFree(current_eventNames_data);

   //--- Loop through each calendar value up to the maximum defined total
   for (int i = 0; i < valuesTotal; i++){

      //---

      //--- Loop through calendar data columns
      for (int k=0; k<ArraySize(array_calendar); k++){
         //---

         //--- Prepare news data array with time, country, and other event details
         string news_data[ArraySize(array_calendar)];
         news_data[0] = TimeToString(values[i].time,TIME_DATE); //--- Event date
         news_data[1] = TimeToString(values[i].time,TIME_MINUTES); //--- Event time
         news_data[2] = country.currency; //--- Event country currency

         //--- Determine importance color based on event impact
         color importance_color = clrBlack;
         if (event.importance == CALENDAR_IMPORTANCE_LOW){importance_color=clrYellow;}
         else if (event.importance == CALENDAR_IMPORTANCE_MODERATE){importance_color=clrOrange;}
         else if (event.importance == CALENDAR_IMPORTANCE_HIGH){importance_color=clrRed;}

         //--- Set importance symbol for the event
         news_data[3] = ShortToString(0x25CF);

         //--- Set event name in the data array
         news_data[4] = event.name;

         //--- Populate actual, forecast, and previous values in the news data array
         news_data[5] = DoubleToString(value.GetActualValue(),3);
         news_data[6] = DoubleToString(value.GetForecastValue(),3);
         news_data[7] = DoubleToString(value.GetPreviousValue(),3);
         //---

      }

      ArrayResize(current_eventNames_data,ArraySize(current_eventNames_data)+1);
      current_eventNames_data[ArraySize(current_eventNames_data)-1] = event.name;

   }
   Print("CURRENT EVENT NAMES DATA SIZE = ",ArraySize(current_eventNames_data));
   ArrayPrint(current_eventNames_data);
   Print("PREVIOUS EVENT NAMES DATA SIZE (Before) = ",ArraySize(previous_eventNames_data));
   ArrayCopy(previous_eventNames_data,current_eventNames_data);
   Print("PREVIOUS EVENT NAMES DATA SIZE (After) = ",ArraySize(previous_eventNames_data));
   ArrayPrint(previous_eventNames_data);

   //Print("Final News = ",news_filter_count);
   updateLabel(TIME_LABEL,"Server Time: "+TimeToString(TimeCurrent(),
              TIME_DATE|TIME_SECONDS)+"   |||   Total News: "+
              IntegerToString(news_filter_count)+"/"+IntegerToString(allValues));
//---
   return(INIT_SUCCEEDED);
}
```

Here, we just have made the changes that will help us get the initial event data. We have highlighted them in yellow color for clarity. Next, we just need to update the dashboard values when there are changes detected during comparison. For this, we will create a custom function that will contain all the logic for the event updates and dashboard recalculation, respectively.

```
//+------------------------------------------------------------------+
//| Function to update dashboard values                              |
//+------------------------------------------------------------------+
void update_dashboard_values(){

//---

}
```

Here, we define the "update\_dashboard\_values" function, which we will use to handle the dynamic updating of the economic calendar dashboard. This function will contain the core logic for comparing stored news data, identifying any changes, and applying the necessary updates to the dashboard interface. By organizing this functionality into this dedicated function, we will ensure a clean and modular code structure, making future modifications or enhancements easier to manage. Next, we will call the function on the [OnTick](https://www.mql5.com/en/docs/event_handlers/ontick) event handler as follows.

```
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick(){
//---

   update_dashboard_values();

}
```

Here, we just call the custom function on every tick to do the updates. Upon compilation and running of the program, we have the following results:

![INITIALIZATION ARRAYS DATA](https://c.mql5.com/2/132/Screenshot_2024-11-15_172545__4.png)

From the image, we can see that we gather all the news events into the "current" array, which we then copy into the fresh "previous" array for storage, which perfectly aligns with what we want. We can now proceed to use these copied data for further analysis. In the function, we just get the current events as follows:

```
//--- Declare variables for tracking news events and status
int totalNews = 0;
bool isNews = false;
MqlCalendarValue values[]; //--- Array to store calendar values

//--- Define start and end time for calendar event retrieval
datetime startTime = TimeTradeServer() - PeriodSeconds(PERIOD_H12);
datetime endTime = TimeTradeServer() + PeriodSeconds(PERIOD_H12);

//--- Set a specific country code filter (e.g., "US" for USD)
string country_code = "US";
string currency_base = SymbolInfoString(_Symbol,SYMBOL_CURRENCY_BASE);

//--- Retrieve historical calendar values within the specified time range
int allValues = CalendarValueHistory(values,startTime,endTime,NULL,NULL);

//--- Print the total number of values retrieved and the array size
//Print("TOTAL VALUES = ",allValues," || Array size = ",ArraySize(values));

//--- Define time range for filtering news events based on daily period
datetime timeRange = PeriodSeconds(PERIOD_D1);
datetime timeBefore = TimeTradeServer() - timeRange;
datetime timeAfter = TimeTradeServer() + timeRange;

//--- Print the furthest time look-back and current server time
//Print("FURTHEST TIME LOOK BACK = ",timeBefore," >>> CURRENT = ",TimeTradeServer());

//--- Limit the total number of values to display
int valuesTotal = (allValues <= 11) ? allValues : 11;

string curr_filter[] = {"AUD","CAD","CHF","EUR","GBP","JPY","NZD","USD"};
int news_filter_count = 0;

ArrayFree(current_eventNames_data);

// Define the levels of importance to filter (low, moderate, high)
ENUM_CALENDAR_EVENT_IMPORTANCE allowed_importance_levels[] = {CALENDAR_IMPORTANCE_LOW, CALENDAR_IMPORTANCE_MODERATE, CALENDAR_IMPORTANCE_HIGH};

//--- Loop through each calendar value up to the maximum defined total
for (int i = 0; i < valuesTotal; i++){

   MqlCalendarEvent event; //--- Declare event structure
   CalendarEventById(values[i].event_id,event); //--- Retrieve event details by ID

   MqlCalendarCountry country; //--- Declare country structure
   CalendarCountryById(event.country_id,country); //--- Retrieve country details by event's country ID

   MqlCalendarValue value; //--- Declare calendar value structure
   CalendarValueById(values[i].id,value); //--- Retrieve actual, forecast, and previous values

   //--- Check if the event’s currency matches any in the filter array (if the filter is enabled)
   bool currencyMatch = false;
   if (enableCurrencyFilter) {
      for (int j = 0; j < ArraySize(curr_filter); j++) {
         if (country.currency == curr_filter[j]) {
            currencyMatch = true;
            break;
         }
      }

      //--- If no match found, skip to the next event
      if (!currencyMatch) {
         continue;
      }
   }

   //--- Check importance level if importance filter is enabled
   bool importanceMatch = false;
   if (enableImportanceFilter) {
      for (int k = 0; k < ArraySize(allowed_importance_levels); k++) {
         if (event.importance == allowed_importance_levels[k]) {
            importanceMatch = true;
            break;
         }
      }

      //--- If importance does not match the filter criteria, skip the event
      if (!importanceMatch) {
         continue;
      }
   }

   //--- Apply time filter and set timeMatch flag (if the filter is enabled)
   bool timeMatch = false;
   if (enableTimeFilter) {
      datetime eventTime = values[i].time;

      if (eventTime <= TimeTradeServer() && eventTime >= timeBefore) {
         timeMatch = true;  //--- Event is already released
      }
      else if (eventTime >= TimeTradeServer() && eventTime <= timeAfter) {
         timeMatch = true;  //--- Event is yet to be released
      }

      //--- Skip if the event doesn't match the time filter
      if (!timeMatch) {
         continue;
      }
   }

   //--- If we reach here, the currency matches the filter
   news_filter_count++; //--- Increment the count of filtered events

   //--- Set alternating colors for each data row holder
   color holder_color = (news_filter_count % 2 == 0) ? C'213,227,207' : clrWhite;

   //--- Loop through calendar data columns
   for (int k=0; k<ArraySize(array_calendar); k++){

      //--- Print event details for debugging
      //Print("Name = ",event.name,", IMP = ",EnumToString(event.importance),", COUNTRY = ",country.name,", TIME = ",values[i].time);

      //--- Skip event if currency does not match the selected country code
      // if (StringFind(_Symbol,country.currency) < 0) continue;

      //--- Prepare news data array with time, country, and other event details
      string news_data[ArraySize(array_calendar)];
      news_data[0] = TimeToString(values[i].time,TIME_DATE); //--- Event date
      news_data[1] = TimeToString(values[i].time,TIME_MINUTES); //--- Event time
      news_data[2] = country.currency; //--- Event country currency

      //--- Determine importance color based on event impact
      color importance_color = clrBlack;
      if (event.importance == CALENDAR_IMPORTANCE_LOW){importance_color=clrYellow;}
      else if (event.importance == CALENDAR_IMPORTANCE_MODERATE){importance_color=clrOrange;}
      else if (event.importance == CALENDAR_IMPORTANCE_HIGH){importance_color=clrRed;}

      //--- Set importance symbol for the event
      news_data[3] = ShortToString(0x25CF);

      //--- Set event name in the data array
      news_data[4] = event.name;

      //--- Populate actual, forecast, and previous values in the news data array
      news_data[5] = DoubleToString(value.GetActualValue(),3);
      news_data[6] = DoubleToString(value.GetForecastValue(),3);
      news_data[7] = DoubleToString(value.GetPreviousValue(),3);
   }

   ArrayResize(current_eventNames_data,ArraySize(current_eventNames_data)+1);
   current_eventNames_data[ArraySize(current_eventNames_data)-1] = event.name;

}
//Print("Final News = ",news_filter_count);
updateLabel(TIME_LABEL,"Server Time: "+TimeToString(TimeCurrent(),
           TIME_DATE|TIME_SECONDS)+"   |||   Total News: "+
           IntegerToString(news_filter_count)+"/"+IntegerToString(allValues));
```

This is just the prior code snippet code that we used during the initialization section, but we need it on every tick to get the updates. We will however go via it briefly. We focus on dynamically updating the dashboard with the latest economic news events on every tick, and to achieve this, we streamline the logic by removing the creation of graphical objects and instead concentrate on managing event data efficiently. We begin by defining variables to track the total news count and filter criteria such as currency, importance, and time ranges. These filters ensure that only relevant events are considered for further processing.

We [loop](https://www.mql5.com/en/docs/basis/operators/for) through the retrieved calendar values, applying the filters to identify events that match the specified conditions. For each matching event, we extract key details like the event name, time, currency, and importance level. These details are stored in the "current\_eventNames\_data" array, which is resized dynamically to accommodate new entries. This array is crucial for tracking the events and allows us to identify changes between ticks by comparing them with previous data. Finally, we update the dashboard label to reflect the total filtered events and current server time, ensuring the dashboard always reflects the latest event data without creating unnecessary objects. This approach efficiently captures and updates the economic news information in real-time.

Next, we need to track if there are changes in the storage arrays using the newly acquired data. To do this, we use a custom function.

```
//+------------------------------------------------------------------+
//|   Function to compare two string arrays and detect changes       |
//+------------------------------------------------------------------+
bool isChangeInStringArrays(string &arr1[], string &arr2[]) {
   bool isChange = false;

   int size1 = ArraySize(arr1);  // Get the size of the first array
   int size2 = ArraySize(arr2);  // Get the size of the second array

   // Check if sizes are different
   if (size1 != size2) {
      Print("Arrays have different sizes. Size of Array 1: ", size1, ", Size of Array 2: ", size2);
      isChange = true;
      return (isChange);
   }

   // Loop through the arrays and compare corresponding elements
   for (int i = 0; i < size1; i++) {
      // Compare the strings at the same index in both arrays
      if (StringCompare(arr1[i], arr2[i]) != 0) {  // If strings are different
         // Action when strings differ at the same index
         Print("Change detected at index ", i, ": '", arr1[i], "' vs '", arr2[i], "'");
         isChange = true;
         return (isChange);
      }
   }

   // If no differences are found, you can also log this as no changes detected
   //Print("No changes detected between arrays.");
   return (isChange);
}
```

Here, we define the [boolean](https://www.mql5.com/en/docs/basis/types/integer/boolconst) function "isChangeInStringArrays", which compares two string arrays, "arr1" and "arr2", to detect any changes between them. We begin by determining the sizes of both arrays using the [ArraySize](https://www.mql5.com/en/docs/array/arraysize) function and store these sizes in "size1" and "size2". If the sizes of the arrays differ, we print the respective sizes, set the "isChange" flag to true, and return true, indicating a change. If the sizes are the same, we proceed to compare the elements of the arrays using a [for loop](https://www.mql5.com/en/docs/array/arraysize). For each index, we use the [StringCompare](https://www.mql5.com/en/docs/strings/stringcompare) function to check if the strings in both arrays are identical. If any strings differ, we print the details of the change and set "isChange" to true, returning true to signal the update. If no differences are found after the loop, the function returns false, indicating that there are no changes. This approach is essential for detecting updates, such as new or updated news events, which we need to reflect on the dashboard.

Armed with the function, we can then use it to update the events.

```
if (isChangeInStringArrays(previous_eventNames_data,current_eventNames_data)){
   Print("CHANGES IN EVENT NAMES DETECTED. UPDATE THE DASHBOARD VALUES");

   ObjectsDeleteAll(0,DATA_HOLDERS);
   ObjectsDeleteAll(0,ARRAY_NEWS);

   ArrayFree(current_eventNames_data);

   //---

}
```

Here, we check if there has been any change between the "previous\_eventNames\_data" and "current\_eventNames\_data" arrays by calling the "isChangeInStringArrays" function. If the function returns true, indicating that changes have been detected, we print a message "CHANGES IN EVENT NAMES DETECTED. UPDATE THE DASHBOARD VALUES". Following this, we delete all objects related to data holders and news arrays on the chart using the [ObjectsDeleteAll](https://www.mql5.com/en/docs/objects/objectdeleteall) function, specifying the identifiers "DATA\_HOLDERS" and "ARRAY\_NEWS" as the object prefixes. We do this to clear up any outdated information before updating the dashboard with the latest event data. Finally, we free the memory used by the "current\_eventNames\_data" array using the [ArrayFree](https://www.mql5.com/en/docs/array/arrayfree) function, ensuring that the array is cleared in preparation for the next update. Following this, we update the events data as usual, but this time round, create the data holders and update the news events on the fresh dashboard. Here is the logic.

```
if (isChangeInStringArrays(previous_eventNames_data,current_eventNames_data)){
   Print("CHANGES IN EVENT NAMES DETECTED. UPDATE THE DASHBOARD VALUES");

   ObjectsDeleteAll(0,DATA_HOLDERS);
   ObjectsDeleteAll(0,ARRAY_NEWS);

   ArrayFree(current_eventNames_data);

   //--- Initialize starting y-coordinate for displaying news data
   int startY = 162;
   //--- Loop through each calendar value up to the maximum defined total
   for (int i = 0; i < valuesTotal; i++){

      MqlCalendarEvent event; //--- Declare event structure
      CalendarEventById(values[i].event_id,event); //--- Retrieve event details by ID

      MqlCalendarCountry country; //--- Declare country structure
      CalendarCountryById(event.country_id,country); //--- Retrieve country details by event's country ID

      MqlCalendarValue value; //--- Declare calendar value structure
      CalendarValueById(values[i].id,value); //--- Retrieve actual, forecast, and previous values

      //--- Check if the event’s currency matches any in the filter array (if the filter is enabled)
      bool currencyMatch = false;
      if (enableCurrencyFilter) {
         for (int j = 0; j < ArraySize(curr_filter); j++) {
            if (country.currency == curr_filter[j]) {
               currencyMatch = true;
               break;
            }
         }

         //--- If no match found, skip to the next event
         if (!currencyMatch) {
            continue;
         }
      }

      //--- Check importance level if importance filter is enabled
      bool importanceMatch = false;
      if (enableImportanceFilter) {
         for (int k = 0; k < ArraySize(allowed_importance_levels); k++) {
            if (event.importance == allowed_importance_levels[k]) {
               importanceMatch = true;
               break;
            }
         }

         //--- If importance does not match the filter criteria, skip the event
         if (!importanceMatch) {
            continue;
         }
      }

      //--- Apply time filter and set timeMatch flag (if the filter is enabled)
      bool timeMatch = false;
      if (enableTimeFilter) {
         datetime eventTime = values[i].time;

         if (eventTime <= TimeTradeServer() && eventTime >= timeBefore) {
            timeMatch = true;  //--- Event is already released
         }
         else if (eventTime >= TimeTradeServer() && eventTime <= timeAfter) {
            timeMatch = true;  //--- Event is yet to be released
         }

         //--- Skip if the event doesn't match the time filter
         if (!timeMatch) {
            continue;
         }
      }

      //--- If we reach here, the currency matches the filter
      news_filter_count++; //--- Increment the count of filtered events

      //--- Set alternating colors for each data row holder
      color holder_color = (news_filter_count % 2 == 0) ? C'213,227,207' : clrWhite;

      //--- Create rectangle label for each data row holder
      createRecLabel(DATA_HOLDERS+string(news_filter_count),62,startY-1,716,26+1,holder_color,1,clrNONE);

      //--- Initialize starting x-coordinate for each data entry
      int startX = 65;

      //--- Loop through calendar data columns
      for (int k=0; k<ArraySize(array_calendar); k++){

         //--- Print event details for debugging
         //Print("Name = ",event.name,", IMP = ",EnumToString(event.importance),", COUNTRY = ",country.name,", TIME = ",values[i].time);

         //--- Skip event if currency does not match the selected country code
         // if (StringFind(_Symbol,country.currency) < 0) continue;

         //--- Prepare news data array with time, country, and other event details
         string news_data[ArraySize(array_calendar)];
         news_data[0] = TimeToString(values[i].time,TIME_DATE); //--- Event date
         news_data[1] = TimeToString(values[i].time,TIME_MINUTES); //--- Event time
         news_data[2] = country.currency; //--- Event country currency

         //--- Determine importance color based on event impact
         color importance_color = clrBlack;
         if (event.importance == CALENDAR_IMPORTANCE_LOW){importance_color=clrYellow;}
         else if (event.importance == CALENDAR_IMPORTANCE_MODERATE){importance_color=clrOrange;}
         else if (event.importance == CALENDAR_IMPORTANCE_HIGH){importance_color=clrRed;}

         //--- Set importance symbol for the event
         news_data[3] = ShortToString(0x25CF);

         //--- Set event name in the data array
         news_data[4] = event.name;

         //--- Populate actual, forecast, and previous values in the news data array
         news_data[5] = DoubleToString(value.GetActualValue(),3);
         news_data[6] = DoubleToString(value.GetForecastValue(),3);
         news_data[7] = DoubleToString(value.GetPreviousValue(),3);

         //--- Create label for each news data item
         if (k == 3){
            createLabel(ARRAY_NEWS+IntegerToString(i)+" "+array_calendar[k],startX,startY-(22-12),news_data[k],importance_color,22,"Calibri");
         }
         else {
            createLabel(ARRAY_NEWS+IntegerToString(i)+" "+array_calendar[k],startX,startY,news_data[k],clrBlack,12,"Calibri");
         }

         //--- Increment x-coordinate for the next column
         startX += buttons[k]+3;
      }

      ArrayResize(current_eventNames_data,ArraySize(current_eventNames_data)+1);
      current_eventNames_data[ArraySize(current_eventNames_data)-1] = event.name;

      //--- Increment y-coordinate for the next row of data
      startY += 25;
      //Print(startY); //--- Print current y-coordinate for debugging
   }

   Print("CURRENT EVENT NAMES DATA SIZE = ",ArraySize(current_eventNames_data));
   ArrayPrint(current_eventNames_data);
   Print("PREVIOUS EVENT NAMES DATA SIZE (Before) = ",ArraySize(previous_eventNames_data));
   ArrayPrint(previous_eventNames_data);
   ArrayFree(previous_eventNames_data);
   ArrayCopy(previous_eventNames_data,current_eventNames_data);
   Print("PREVIOUS EVENT NAMES DATA SIZE (After) = ",ArraySize(previous_eventNames_data));
   ArrayPrint(previous_eventNames_data);

}
```

Here, we update the dashboard based on the newly acquired data to ensure there are live updates taking effect. Using copy logic, we log the data into the "previous" data holder so that we can use the current data on the next check. We have highlighted the logic that takes care of that, but let us have a deeper look at it.

```
Print("CURRENT EVENT NAMES DATA SIZE = ",ArraySize(current_eventNames_data));
ArrayPrint(current_eventNames_data);
Print("PREVIOUS EVENT NAMES DATA SIZE (Before) = ",ArraySize(previous_eventNames_data));
ArrayPrint(previous_eventNames_data);
ArrayFree(previous_eventNames_data);
ArrayCopy(previous_eventNames_data,current_eventNames_data);
Print("PREVIOUS EVENT NAMES DATA SIZE (After) = ",ArraySize(previous_eventNames_data));
ArrayPrint(previous_eventNames_data);
```

Here, we begin by printing the current size of the "current\_eventNames\_data" array using the [ArraySize](https://www.mql5.com/en/docs/array/arraysize) function and displaying its contents with the [ArrayPrint](https://www.mql5.com/en/docs/array/arrayprint) function. This will help us inspect the current set of event names that we are tracking. Then, we print the size of the "previous\_eventNames\_data" array before it is updated, followed by printing its contents.

Next, we free the memory used by "previous\_eventNames\_data" with the [ArrayFree](https://www.mql5.com/en/docs/array/arrayfree) function, ensuring any previous data stored in the array is cleared to avoid memory issues. After freeing the memory, we use the [ArrayCopy](https://www.mql5.com/en/docs/array/arraycopy) function to copy the contents of the "current\_eventNames\_data" array into "previous\_eventNames\_data", effectively updating it with the latest event names.

Finally, we print the updated size of the "previous\_eventNames\_data" array and its contents to confirm that the array now holds the most recent event names. This ensures that the previous event names are correctly updated for future comparison. Upon running the program, we have the following outcome.

Time updates.

![TIME UPDATES GIF](https://c.mql5.com/2/132/TIME_UPDATES_GIF__4.gif)

Events update.

![EVENTS UPDATE](https://c.mql5.com/2/132/Screenshot_2024-11-18_124734__4.png)

Dashboard update.

![DASHBOARD UPDATE](https://c.mql5.com/2/132/Screenshot_2024-11-18_124417__4.png)

From the image, we can see that the newly registered data is accurately updated on the dashboard. To reconfirm this, we can wait again for some time and try to see of we can keep track of this data, and with the one that logs the updated data. Here is the outcome.

Events data update.

![NEW EVENTS DATA UPDATE](https://c.mql5.com/2/132/Screenshot_2024-11-18_124841__4.png)

Dashboard update.

![NEW DASHBOARD UPDATE](https://c.mql5.com/2/132/Screenshot_2024-11-18_124921__4.png)

From the image, we can see that once there are changes from the previously stored data, they are detected and updated according to the newly registered data and stored for further reference. The stored data is the used to update the dashboard interface in real time, displaying the current news events, and thus confirming the success of our objective.

### Conclusion

In conclusion, we implemented a robust system to monitor and detect changes in [MQL5 Economic News](https://www.mql5.com/en/book/advanced/calendar) events by comparing previously stored event data with newly retrieved updates. This comparison mechanism ensures that any differences in event names or details are promptly identified, triggering our dashboard refresh to maintain accuracy and relevance. By filtering data based on currency, importance, and time, we further refined the process to focus on impactful events while dynamically updating the interface.

In the next parts of this series, we will build on this foundation by integrating economic news events into trading strategies, enabling practical applications of the data. Additionally, we aim to enhance the dashboard's functionality by introducing mobility and responsiveness, ensuring a more seamless and interactive experience for traders. Keep tuned.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/16386.zip "Download all attachments in the single ZIP archive")

[MQL5\_NEWS\_CALENDAR\_PART\_4.mq5](https://www.mql5.com/en/articles/download/16386/mql5_news_calendar_part_4.mq5 "Download MQL5_NEWS_CALENDAR_PART_4.mq5")(75.02 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/477556)**
(2)


![Stanislav Korotky](https://c.mql5.com/avatar/2010/10/4CA7CFA0-1F0C.jpg)

**[Stanislav Korotky](https://www.mql5.com/en/users/marketeer)**
\|
3 Dec 2024 at 20:57

Users will probably be interested to know that the same topics have been covered thoroughly in the algotrading book (and the site does not show this somehow in automatic suggestions), including [tracking and saving event changes](https://www.mql5.com/en/book/advanced/calendar/calendar_change_last), [filtering events by multiple conditions](https://www.mql5.com/en/book/advanced/calendar/calendar_filter_custom) (with different logical operators) for extended set of fields, displaying customizable on-the-fly calendar subset in a panel on chart, and embedding for trading into EAs with support of [transferring complete archive of the calendar into the tester.](https://www.mql5.com/en/book/advanced/calendar/calendar_cache_tester)

![Allan Munene Mutiiria](https://c.mql5.com/avatar/2022/11/637df59b-9551.jpg)

**[Allan Munene Mutiiria](https://www.mql5.com/en/users/29210372)**
\|
5 Dec 2024 at 00:08

**Stanislav Korotky [#](https://www.mql5.com/en/forum/477556#comment_55292377):**

Users will probably be interested to know that the same topics have been covered thoroughly in the algotrading book (and the site does not show this somehow in automatic suggestions), including [tracking and saving event changes](https://www.mql5.com/en/book/advanced/calendar/calendar_change_last), [filtering events by multiple conditions](https://www.mql5.com/en/book/advanced/calendar/calendar_filter_custom) (with different logical operators) for extended set of fields, displaying customizable on-the-fly calendar subset in a panel on chart, and embedding for trading into EAs with support of [transferring complete archive of the calendar into the tester.](https://www.mql5.com/en/book/advanced/calendar/calendar_cache_tester)

Okay. Thanks.


![Introduction to MQL5 (Part 10): A Beginner's Guide to Working with Built-in Indicators in MQL5](https://c.mql5.com/2/104/Introduction_to_MQL5_Part_10___LOGO__1.png)[Introduction to MQL5 (Part 10): A Beginner's Guide to Working with Built-in Indicators in MQL5](https://www.mql5.com/en/articles/16514)

This article introduces working with built-in indicators in MQL5, focusing on creating an RSI-based Expert Advisor (EA) using a project-based approach. You'll learn to retrieve and utilize RSI values, handle liquidity sweeps, and enhance trade visualization using chart objects. Additionally, the article emphasizes effective risk management, including setting percentage-based risk, implementing risk-reward ratios, and applying risk modifications to secure profits.

![Generative Adversarial Networks (GANs) for Synthetic Data in Financial Modeling (Part 1): Introduction to GANs and Synthetic Data in Financial Modeling](https://c.mql5.com/2/102/Generative_Adversarial_Networks_pGANso_for_Synthetic_Data_in_Financial_Modeling_Part_1__LOGO.png)[Generative Adversarial Networks (GANs) for Synthetic Data in Financial Modeling (Part 1): Introduction to GANs and Synthetic Data in Financial Modeling](https://www.mql5.com/en/articles/16214)

This article introduces traders to Generative Adversarial Networks (GANs) for generating Synthetic Financial data, addressing data limitations in model training. It covers GAN basics, python and MQL5 code implementations, and practical applications in finance, empowering traders to enhance model accuracy and robustness through synthetic data.

![Chemical reaction optimization (CRO) algorithm (Part II): Assembling and results](https://c.mql5.com/2/81/Algorithm_for_optimization_by_chemical_reactions__LOGO___1.png)[Chemical reaction optimization (CRO) algorithm (Part II): Assembling and results](https://www.mql5.com/en/articles/15080)

In the second part, we will collect chemical operators into a single algorithm and present a detailed analysis of its results. Let's find out how the Chemical reaction optimization (CRO) method copes with solving complex problems on test functions.

![Price Action Analysis Toolkit Development (Part 3): Analytics Master — EA](https://c.mql5.com/2/103/Price_Action_Analysis_Toolkit_Development_Part_3___LOGO.png)[Price Action Analysis Toolkit Development (Part 3): Analytics Master — EA](https://www.mql5.com/en/articles/16434)

Moving from a simple trading script to a fully functioning Expert Advisor (EA) can significantly enhance your trading experience. Imagine having a system that automatically monitors your charts, performs essential calculations in the background, and provides regular updates every two hours. This EA would be equipped to analyze key metrics that are crucial for making informed trading decisions, ensuring that you have access to the most current information to adjust your strategies effectively.

[![](https://www.mql5.com/ff/si/s2n3m9ymjh52n07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F523%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dchoose.signals%26utm_content%3Dsubscribe.signal%26utm_campaign%3D0622.MQL5.com.Internal&a=fyznzyduwsltgnhlftytumasbfgbwlqw&s=91bc0eca8f132d3df7d14cdb1baebac753aef179403d60dc83856af55a4d6769&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=tjdxwfcnpscskmtrpueqaqgpkfkefojn&ssn=1769193197210003646&ssn_dr=0&ssn_sr=0&fv_date=1769193197&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F16386&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Trading%20with%20the%20MQL5%20Economic%20Calendar%20(Part%204)%3A%20Implementing%20Real-Time%20News%20Updates%20in%20the%20Dashboard%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176919319707799354&fz_uniq=5071922301760385260&sv=2552)

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