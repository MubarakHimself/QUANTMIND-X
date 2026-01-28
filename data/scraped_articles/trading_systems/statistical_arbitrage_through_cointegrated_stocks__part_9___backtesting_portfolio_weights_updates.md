---
title: Statistical Arbitrage Through Cointegrated Stocks (Part 9): Backtesting Portfolio Weights Updates
url: https://www.mql5.com/en/articles/20657
categories: Trading Systems
relevance_score: 6
scraped_at: 2026-01-23T11:31:27.353661
---

[![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/01.png)![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/02.png)Powerful analytics for traders of any levelAll the necessary trading reports for beginners and professionals](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/454106&a=muccpajyfystoakuukdobwigjejzmpqn&s=52daad60fa795e635264e6f94898f05493bca3b5124d4cca8eb7e82333c2ef12&uid=&ref=https://www.mql5.com/en/articles/20657&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5062528022873023453)

MetaTrader 5 / Trading systems


### Introduction

The market is in a continuous state of change. This is the mantra that is guiding us in this journey towards building a statistical arbitrage framework for the average retail trader. We stick to it by avoiding the usual notions of bear and bull markets, directional trends, or correlated assets. Instead, we use statistical methods to estimate the probabilities of pairs or groups of assets to preserve some kind of relationship over a foreseeable time horizon. For now, we are dealing with the cointegration relationship for its flexibility and almost universal applicability in financial markets. We can look for cointegration between any assets, including assets from different classes, and even between financial assets and non-financial data, like a stock symbol and the evolution of shipping costs. Once a cointegration is found, it can almost certainly be traded.

The drawback is that, from a statistical point of view, there is no guarantee that the cointegration will remain valid for the next hour, day, or week. It will always have some residual probability that it will break starting from the next tick. There is almost a one-hundred percent probability that the tickers’ prices will change.

We calculate the cointegration vector from the tickers’ prices at the moment. From the cointegration vector, we obtain the relative portfolio weights, that is, the assets volume to be bought or sold in each order. Since the tickers’ prices are changing continuously, so are the portfolio weights. We can say with nearly one hundred percent probability that the optimum order volume of yesterday is not the optimum order volume for today. The relative prices changed, so the portfolio weights must be updated.

In the [last article](https://www.mql5.com/en/articles/20485), we saw how we can run a continuous monitoring of the portfolio weights stability by using the In-Sample/Out-of-Sample ADF (IS/OOS ADF) validation in tandem with the Rolling Windows Eigenvector Comparison (RWEC). This well-known technique is effective in detecting past cointegration breaks and in estimating the probability of asset relationship breaks in the future. These features make it useful both in the context of data analysis for portfolio building and as a risk-management tool in the context of live trading monitoring. While building the portfolio, we can evaluate how the model performs with variations of each method's main parameters, and while monitoring, we can fine-tune these parameters according to the most recent data analysis. Since our RWEC implementation is using the same method we use in our scoring system for ranking the cointegration strength - the Johansen cointegration test - its resulting cointegration vectors can be used to update our portfolio weights.

In live trading, our EA will read the portfolio weights from the ‘strategy’ table, as we discussed when [setting up the database as the single source of truth](https://www.mql5.com/en/articles/19242). But in the backtests, we do not have access to the database. Backtesting is the only practical way to simulate the fine-tuning of those parameters for dozens, potentially hundreds of asset pairs and baskets in a reasonable timespan. By backtesting, we can evaluate our signal stability test methods and also the effectiveness of our rebalancing algorithms. We need to check if the EA logic for rebalancing is working as expected, and to what extent the chosen parameters for rebalancing would improve our results.The most straightforward method to access database data in Metatrader 5 Tester is to export the required data into a file and read it directly from the EA. Here we’ll be exporting the full ‘strategy’ table as a CSV (Comma Separated Values) file and loading the “new” portfolio weights on a dedicated test helper function.

### Filling the database with sample data

Before exporting our database ‘strategy’ table, we need the RWEC data on it. We’ll do this using a simple Python script that runs the RWEC analysis and stores its results in our database. Again, note that this is a simulation for the backtest. In normal circumstances, the RWEC analysis will be part of our daily routine of live trading monitoring, and its results will be consumed by the EA directly from the database.

The script will fetch the required data from the Metatrader 5 terminal. As usual, in our examples, we are using only the symbols available in the Meta Quotes demo account, so you should have an easy time when running these experiments by yourself.

You can find this script attached here as _rwec2db.py_.

![Figure 1 - Screen capture showing a folded overview of rwec2db.py methods/functions.](https://c.mql5.com/2/186/Capture_script_rwec2db_overview_folded.png)

Figure 1. Screen capture showing a folded overview of _rwec2db.py_ methods/functions

By running this script, you should see an output like this:

![Figure 2 - Screen capture showing the rwec2db.py expected output.](https://c.mql5.com/2/186/Capture_script_rwec2db_output_ok.png)

Figure 2. Screen capture showing the _rwec2db.py_ expected output

If no cointegration vector is found, it will inform you.

![Figure 3 - Screen capture showing the rwec2db.py output when no cointegration vector is found.](https://c.mql5.com/2/186/Capture_script_rwec2db_output_no_vectors.png)

Figure 3. Screen capture showing the _rwec2db.py_ output when no cointegration vector is found

Note that the above may occur when testing the same symbols over a relatively short period, let’s say 30 bars. That is because there may not be enough historical data for rolling windows eigenvector comparison. If this happens, you can request more data and/or increase the rolling window length.If everything goes well, your ‘strategy’ table should end with something like this.

![Fig. 4 - Screen capture of MetaEditor integrated SQLite db showing the ‘strategy’ table filled with sample data](https://c.mql5.com/2/186/Capture_mt5_db_strategy_table.png)

Fig. 4. Screen capture of MetaEditor integrated SQLite db showing the ‘strategy’ table filled with sample data

Now we are ready to export the table.

### Exporting the database table

MetaEditor has a built-in export table function. Just right-click on the table name.

![Fig. 5 - Screen capture of MetaEditor database contextual menu](https://c.mql5.com/2/186/Capture_mt5_db_export_menu.png)

Fig. 5. Screen capture of MetaEditor database contextual menu

To have your export fully compatible with the example code attached, choose the following options in the dialog box that follows.

![Fig. 6 - Screen capture of MetaEditor database export options dialog with recommended options highlighted ](https://c.mql5.com/2/186/Capture_mt5_db_export_dialog_recomm_options.png)

Fig. 6. Screen capture of MetaEditor database export options dialog with recommended options highlighted

Choose the tab character as a separator. Thus, actually, we’ll work with a Tab-Separated Values file - TSV. This is meant to ease our portfolio weights parsing, since they are being stored as a JSON array in the database. These arrays already contain commas.

```
[1.0, -0.045459, 0.021855, -0.033486]
```

By choosing the tab character as a separator, we can ease this parsing. Also, chose to preserve the column names and the double-quoted strings.

Remember that for your TSV file to be accessible to the Tester environment, it must be saved on the TERMINAL\_DATA\_PATH (“Folder in which terminal data are stored”) or on the TERMINAL\_COMMONDATA\_PATH (“Common path for all of the terminals installed on a computer”). For this example, we’ll be using the former.

Unless you are saving the TSV file in the common path mentioned above, you’ll need to include the property tester\_file in your main MQL5 file.

```
//+------------------------------------------------------------------+
//|                                                  CointNasdaq.mq5 |
//|                                  Copyright 2025, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| Expert Advisor - Cointegration Statistical Arbitrage             |
//| Assets: Dynamic assets allocation                                |
//| Strategy: Mean-reversion on Johansen cointegration portfolio     |
//+------------------------------------------------------------------+
#property tester_file "StatArb\\strategy_202512041731.csv"
```

"“Properties described in included files are completely ignored. Properties must be specified in the main mq5-file. (...) File name for a tester with the indication of extension, in double quotes (as a constant string). The specified file will be passed to the tester. Input files to be tested, if there are necessary ones, must always be specified.” ( [MQL5 docs](https://www.mql5.com/en/docs/basis/preprosessor/compilation))

Without this property, you’ll not be able to read the file from the Tester environment.

![Fig. 7 - Screen capture of Metatrader 5 journal reporting file opening error on Tester.](https://c.mql5.com/2/186/Capture_journal_err_non-declared-file_terminal_data_path.png)

Fig. 7. Screen capture of Metatrader 5 journal reporting file opening error on Tester

This happens because the Tester environment works as an isolated sandbox for security reasons. This article has a short and clear description of the internal process involved in [working with files in the Tester](https://www.mql5.com/en/articles/2720#z20). There is a lot of information in the Metatrader 5 documentation about reading and writing files. The AlgoBook has a comprehensive [guide about working with files in MQL5](https://www.mql5.com/en/book/common/files). Here we will focus on the requirements for our specific use case.

### Loading the strategy parameters from the file

After these preliminary measures, we are ready to present the LoadStrategyFromFile() function, now included in our sample EA CointNasdaq.mq5, that you can find attached at the bottom of this article. The attached code is intentionally ‘raw’. I did not embellish it by removing debug prints. Instead, I left them alongside all the comments that guided my writing, so you can follow better and investigate simpler, cleaner, or efficient solutions. Most of the debug prints I used are commented out. So, if you have trouble while testing your own modifications, you can simply uncomment the code where needed.

The LoadStrategyFromFile() function may be called right on the EA OnInit() event handler, depending on the running environment.

```
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   ResetLastError();
// Check if all symbols are available
   for(int i = 0; i < ArraySize(symbols); i++)
     {
      if(!SymbolSelect(symbols[i], true))
        {
         Print("Error: Symbol ", symbols[i], " not found!");
         return(INIT_FAILED);
        }
     }
// Initialize spread buffer
   ArrayResize(spreadBuffer, InpLookbackPeriod);
// Set a timer for spread, mean, stdev calculations
// and strategy parameters update (check DB)
   EventSetTimer(InpUpdateFreq * 60); // min one minute
// check if we are backtesting
   if(!MQLInfoInteger(MQL_TESTER))
     {
      // Load strategy parameters from database
      if(!LoadStrategyFromDB(InpDbFilename,
                             InpStrategyName,
                             symbols,
                             weights,
                             timeframe,
                             InpLookbackPeriod))
        {
         // Handle error - maybe use default values
         printf("Error at " + __FUNCTION__ + " %s ",
                getUninitReasonText(GetLastError()));
         return INIT_FAILED;
        }
     }
   else
     {
      // Load strategy parameters from CSV file
      if(!LoadStrategyFromFile(InpTesterStrategyFilename, symbols, weights))
        {
         // Handle error - maybe use default values
         printf("Error at " + __FUNCTION__ + " %s ",
                getUninitReasonText(GetLastError()));
         return INIT_FAILED;
        }
     }
   return(INIT_SUCCEEDED);
  }
```

If the EA is running on the Tester environment, we load the strategy from the file. Otherwise, in normal trading, the strategy will be loaded from the database.

To avoid cluttering our EA main file, the function is implemented in a companion header file, TestHelper.mqh, also attached here.

```
//+------------------------------------------------------------------+
//|  Load the strategy parameters from CSV/TSV file                  |
//+------------------------------------------------------------------+
bool LoadStrategyFromFile(string filename,
                          string &strat_symbols[],
                          double &strat_weights[])
 {
   Print("Running on tester");
// Instantiate the hash map
   CHashMap<ulong, CArrayDouble*> updates;
// Load the weights from the CSV file
   LoadWeights(filename, updates);
   Print("Updates count ", updates.Count());
(...)
```

By looking at the function parameters, you can see that the function deals with the portfolio weights only. That is in contrast with its counterpart, which loads the strategy from the database and includes the strategy name, timeframe, and lookback period. That is because this implementation is a work in progress. Later, when dealing with portfolio rotation, it will include all strategy parameters as well.

The function starts by instantiating a dynamic hash table from the [MQL5 Standard Library/Generic Data Collections](https://www.mql5.com/en/docs/standardlibrary/generic/chashmap), a CHashMap in which the keys will store our portfolio updates timestamps, and the values will be our portfolio weights as arrays of doubles. Its object instance, alongside the filename, is then passed as a reference to a LoadWeights() dedicated function to properly read the CSV file.

```
//+------------------------------------------------------------------+
//|    Load portfolio weights updates from CSV/TSV file              |
//+------------------------------------------------------------------+
void LoadWeights(string filename,
                 CHashMap<ulong, CArrayDouble*> &updates)
  {
   ResetLastError();
   int filehandle = FileOpen(filename,
                             FILE_ANSI | FILE_CSV | FILE_READ, '\\t', CP_ACP);
   if(filehandle != INVALID_HANDLE)
     {
      printf("Data Path: %s Filename: %s", TerminalInfoString(TERMINAL_DATA_PATH), filename);
      // Read and discard the header line
      string first_line = FileReadString(filehandle);
      Print(first_line);
(...)
```

The LoadWeights() function follows the regular approach of reading text files in MQL5. Again, note that we are NOT using the FILE\_COMMON flag while opening the file, meaning we are using the terminal data path, which requires the use of #property tester\_file in our EA, as commented above.We are discarding the first line, the CSV header, and printing it to the journal for checking.

![Fig. 8 - Metatrader 5 journal screen capture showing the first lines of test logging](https://c.mql5.com/2/186/Capture_journal_test_start.png)

Fig. 8. Metatrader 5 journal screen capture showing the first lines of test logging

Then we start iterating over the CSV file lines besides the header to split them by the tab character. The resulting segments are the values we are looking for. So, we store them in the fields\[\] string array.

```
// iterate over lines
      while(!FileIsEnding(filehandle))
        {
         string line = FileReadString(filehandle);
         string fields[]; // fields[0] -> tstamp   fields[4] -> weights
         int count = StringSplit(line, '\t', fields);
         //printf("fields => %s  %s %s %s %s %s %s",
         //       fields[0], fields[1], fields[2],
         //       fields[3], fields[4], fields[5], fields[6]);
         //—
(...)
```

We create a [CArrayDouble object](https://www.mql5.com/en/docs/standardlibrary/datastructures/carraydouble) to receive the weights array for each read line. This is the object required by our CHashMap class.

```
// Create the CArrayDouble object for this timestamp
         CArrayDouble *current_weights_arr = new CArrayDouble();
         ulong tstamp = 0;
(...)
```

Since our portfolio weights arrays are stored as JSON arrays in the database, we need to clean them by removing the brackets before passing them to the CArrayDouble object.

```
// Ensure we have at least the tstamp (0) and weights (4) fields
         if(count > 4)
           {
            // weights string
            string weights_str = fields[4];
            StringReplace(weights_str, "[", "");\
            StringReplace(weights_str, "]", "");
            // weights strings array
            string weights_str_arr[];
            int weights_count = StringSplit(weights_str, ',', weights_str_arr);
            //---
            if(current_weights_arr == NULL)
              {
               Print("Err creating CArrayDouble for timestamp ", fields[0]);
               continue; // Skip to the next line
              }
            // Populate the new CArrayDouble
            for(int i = 0; i < weights_count; i++)
              {
               //printf("weights_str_arr %s", weights_str_arr[i]);
               double weight_value = StringToDouble(weights_str_arr[i]);
               //printf("weight_value %.6f", weight_value);
               tstamp = (ulong)StringToInteger(fields[0]);
               //printf("tstamp %I64u ", tstamp);
               current_weights_arr.Add(weight_value);
               //printf("current_weights_arr Total: %d", current_weights_arr.Total());
              }
(...)
```

Now that our CArrayDouble object is populated, we can add it to our hash map.

```
// 4. Add to the HashMap once per line
            if(updates.Add(tstamp, current_weights_arr))
              {
               printf("Added tstamp %I64u -> %s ", tstamp, TimeToString(tstamp));
              }
            else
              {
               Print("Failed adding record");
              }
           }
        }
     }
   else
     {
      printf("Error opening file %s. Error: %i", filename, GetLastError());
     }
   FileClose(filehandle);
  }
```

![Fig. 9 - Screen capture of the Metatrader 5 journal showing the updates’ timestamps added to the hash map](https://c.mql5.com/2/186/Capture_journal_test_start_1_added_timestamps.png)

Fig. 9. Screen capture of the Metatrader 5 journal showing the updates’ timestamps added to the hash map

Back to our LoadStrategyFromFile() function, we check the number of portfolio updates loaded. This is the number of lines in your CSV file minus one (the header).

```
// Load the weights from the CSV file
   LoadWeights(filename, updates);
   Print("Updates count ", updates.Count());
// copy the values to iterable arrays
   ulong tstamp_keys[];
   CArrayDouble *weights_values[];
   updates.CopyTo(tstamp_keys, weights_values);
// check if everything was copied
   Print("Keys size: ", tstamp_keys.Size());
   Print("Values size: ", weights_values.Size());
(...)
```

![Fig. 10 - Screen capture of the Metatrader 5 journal showing the number of updates to be processed](https://c.mql5.com/2/186/Capture_journal_test_start_2_number_of_updates.png)

Fig. 10. Screen capture of the Metatrader 5 journal showing the number of updates to be processed

We then check for outdated updates on file.

```
// check for outdated updates on file
   ulong first_tstamp_on_file = tstamp_keys[0];
   printf("first_tstamp_on_file %I64u", first_tstamp_on_file);
   ulong update_to_apply = 0;
   if(FileHasOutdatedUpdates(first_tstamp_on_file))
      FileCleanUpdates(tstamp_keys, updates, update_to_apply);
(...)
```

Remember that we are using exported database data to simulate the portfolio weights updates we will be sourcing from the database when live trading. So, the only relevant update is the last one, that with the earlier timestamp, right before the current time. But when exporting the data our database may contain - and probably will contain - older RWEC data, from many days or weeks before our backtest start time. These older data must be removed so we preserve the datetime alignment between our sample data and our backtest settings.

```
//+------------------------------------------------------------------+
//|   check if the CSV file has outdated updates                     |
//+------------------------------------------------------------------+
bool FileHasOutdatedUpdates(ulong updates_start_time)
  {
   datetime test_start_time = TimeCurrent();
   if((datetime)updates_start_time < test_start_time)
     {
      Print("Warning! Updates starts before test start time.");
      printf("Test start time: %s", TimeToString(test_start_time));
      printf("Updates start time: %s", TimeToString(updates_start_time));
      Print("Will REMOVE outdated updates.");
      return true;
     }
   return false;
  }
```

![Fig. 11 - Screen capture of the Metatrader 5 journal showing the warning of outdated updates to be removed](https://c.mql5.com/2/186/Capture_journal_test_start_3_outdated_updates_warn.png)

Fig. 11. Screen capture of the Metatrader 5 journal showing the warning of outdated updates to be removed

The FileCleanUpdates() function will iterate over the timestamps to remove all outdated updates but the last one.

```
//+------------------------------------------------------------------+
//| iterate over keys to remove outdated updates                     |
//+------------------------------------------------------------------+
void FileCleanUpdates(ulong &tstamp_keys[],
                      CHashMap<ulong, CArrayDouble*> &updates,
                      ulong &update_to_apply)
  {
   int outdated_count = 0;
   ulong outdated_keys[];
   for(int i = 0; i < ArraySize(tstamp_keys); i++)
     {
      if((datetime)tstamp_keys[i] < TimeCurrent()) // look for outdated updates
        {
         printf("Outdated updates at: %s", TimeToString(tstamp_keys[i]));
         outdated_keys.Push(tstamp_keys[i]);
         outdated_count++;
        }
      while(outdated_count > 1) // preserve the newest one to be applied
        {
         if(updates.Remove(tstamp_keys[i - 1]))
           {
            outdated_count--;
            printf("Removed outdated update from %s ", TimeToString(tstamp_keys[i - 1]));
           }
        }
     }
   printf("Removed %i outdated updates:", outdated_keys.Size() - 1);
   update_to_apply = outdated_keys[outdated_keys.Size() - 1];
   printf("Update from %s to be applied", TimeToString(update_to_apply));
  }
```

The removed outdated updates and the update to be applied are both reported in the journal.

![Fig. 12 - Screen capture of the Metatrader 5 journal showing the notice of removed updates](https://c.mql5.com/2/186/Capture_journal_test_start_4_outdated_updates_removed_notice.png)

Fig. 12. Screen capture of the Metatrader 5 journal showing the notice of removed updates

Finally, our LoadStrategyFromFile() function can finish its job by setting the EA portfolio weights.

```
//---
   CArrayDouble *new_weights = new CArrayDouble();
   if(updates.TryGetValue(update_to_apply, new_weights))
     {
      ArrayResize(strat_weights, 4);
      //---
      strat_weights[0] = new_weights[0];
      strat_weights[1] = new_weights[1];
      strat_weights[2] = new_weights[2];
      strat_weights[3] = new_weights[3];
      //---
      ArrayResize(strat_symbols, 4);
      //---
      strat_symbols[0] = "INTC";
      strat_symbols[1] = "AMD";
      strat_symbols[2] = "AVGO";
      strat_symbols[3] = "MU";
      //---
      printf("New weights: %s %.6f | %s %.6f | %s %.6f | %s %.6f",
             strat_symbols[0], new_weights[0],
             strat_symbols[1], new_weights[1],
             strat_symbols[2], new_weights[2],
             strat_symbols[3], new_weights[3]
            );
      return true;
     }
   return false;
  }
```

The new portfolio weights are reported in the journal.

![Fig. 13 - Screen capture of the Metatrader 5 journal showing the notice of new portfolio weights ](https://c.mql5.com/2/186/Capture_journal_test_start_5_new_weights_notice.png)

Fig. 13. Screen capture of the Metatrader 5 journal showing the notice of new portfolio weights

All of the above is called once from the EA OnInit() event handler. At the backtest start, our EA loads the strategy parameters from the CSV file, stores them in a hash map object, removes the updates whose timestamps are less than the backtest start time (the "outdated updates”) if any, and applies the most recent of them. Now, as the backtest goes forward in time, we need to apply the following updates, those whose timestamps are greater than the backtest start time, to simulate the portfolio weights updates that will come while the EA is running on live trading. This is done using the OnTimer() event handler.

```
//+------------------------------------------------------------------+
//| Timer function                                                   |
//+------------------------------------------------------------------+
void OnTimer(void)
  {
   ResetLastError();
   if(!MQLInfoInteger(MQL_TESTER))
     {
      // Wrapper around LoadStrategyFromDB: for clarity
      if(!UpdateModelParams(InpDbFilename,
                            InpStrategyName,
                            symbols,
                            weights,
                            timeframe,
                            InpLookbackPeriod))
        {
         printf("%s failed: Error %i", __FUNCTION__, GetLastError());
        }
     }
   else
     {
      if(!UpdateModelParamsFromFile(symbols, weights))
        {
         printf("%s failed: Error %i", __FUNCTION__, GetLastError());
        }
     }
   printf("Actual weights: %s %.6f | %s %.6f | %s %.6f | %s %.6f",
          symbols[0], weights[0],
          symbols[1], weights[1],
          symbols[2], weights[2],
          symbols[3], weights[3]
         );
```

The UpdateModelParamsFromFile() function is very simple. Since we have copied our HashMap keys to a separate array, all that we need here is to get each one of them sequentially at each function call. Each key is an update’s timestamp. If it is less than the current time, we update the basket weights accordingly. Otherwise, if the key/timestamp is greater than the current time, it is an update that doesn’t exist yet. In this case, the function returns true because there was no error, but no update is applied. It will only return false/error if the HashMap<> TryGetValue() method cannot find the respective key in the hash map object.

```
ulong tstamp_keys[];
CHashMap<ulong, CArrayDouble*> *updates = new CHashMap<ulong, CArrayDouble*>();

//+------------------------------------------------------------------+
//|  get the earlier tstamp on the updates hash map;                 |
//|  then update the model params (symbols and weights)              |
//|  with its values                                                 |
//+------------------------------------------------------------------+
bool UpdateModelParamsFromFile(string &curr_symbols[], double &curr_weights[])
  {
// Print("Updating model params from file");

// get the earlier tstamp on the updates hash map
   int static i = 0;
   if((datetime)tstamp_keys[i] < TimeCurrent())
     {
      curr_symbols[0] = "INTC";
      curr_symbols[1] = "AMD";
      curr_symbols[2] = "AVGO";
      curr_symbols[3] = "MU";
//—--
      CArrayDouble *new_weights = new CArrayDouble();
      if(!updates.TryGetValue(tstamp_keys[i], new_weights))
        {
         return false;
        }
      curr_weights[0] = new_weights[0];
      curr_weights[1] = new_weights[1];
      curr_weights[2] = new_weights[2];
      curr_weights[3] = new_weights[3];
//—-- increment the idx
      i++;
      delete new_weights;
     }
   else
     {
      Print("No update to apply");
     }
   return true;
  }
```

When an update occurs in the backtest, it is logged in the journal like this.

![Fig. 14 - Screen capture of the Metatrader 5 journal showing the portfolio weights update in backtest](https://c.mql5.com/2/186/Capture_journal_test_start_6_new_weights_updates.png)

Fig. 14. Screen capture of the Metatrader 5 journal showing the portfolio weights update in backtest

Our backtest starts here, after loading, parsing, and cleaning up the database exported CSV/TSV file.

As stated in this article’s introduction, we need to check if the EA logic for rebalancing is working as expected, and to what extent the chosen RWEC parameters for rebalancing would improve our results.

### The RWEC parameters to be tested

The backtest goal is to get close to the optimum rolling cointegration parameter values. Note that RWEC uses a Johansen cointegration test to assess the existence of cointegration. This test has its own parameters, but we’ll not be considering them here because it is assumed that we are already working with a cointegrated basket. The RWEC's role now is to update the portfolio weights previously calculated by the Johansen test in the screening/scoring steps of our pipeline.The main RWEC parameters we want to backtest are:

1. The time horizon of the requested history data
2. The length of the cointegration test window
3. The length of the overlapping windows

As you can see below, where we describe each of these parameters, there is always a trade-off between timeliness and accuracy when tweaking them. Our goal is to approximate the right balance. To start, we will backtest the RWEC metric for at least three window sizes to see which one provides the best portfolio weights rebalancing for the H4 timeframe. This is the timeframe we’ve been working on right from the start. Our evaluation criterion will be the Relative Drawdown.

WARNING: At this point, we are backtesting to find optimum RWEC parameters. We are NOT focused on the strategy's profitability. We already have an objective criterion to evaluate: the Relative Drawdown. Once we choose the optimum parameters for the basket, we’ll use them on live trading, not in backtests anymore. Understanding that this is the backtest goal is crucial.

### The backtest settings

Let’s start with the usual one-year period, that is, nearly 252 trading days.

![Fig. 15 - Screen capture of the settings for one one-year backtest (approx. 252 trading days)](https://c.mql5.com/2/186/Capture_backtest_settings_RWEC_252_window.png)

Fig. 15. Screen capture of the settings for one one-year backtest (approx. 252 trading days)

The RWEC time horizon (n\_bars) should be at least equal to our backtest start date plus the window length, so we can better simulate live trading by having early updates right at the beginning of the backtest. Since we are in the H4 timeframe, and for stocks, we have 2 H4 bars per day, we need at least 504 bars. We will add 90 bars per our window length.

```
def fetch_data(self, symbols, timeframe=mt5.TIMEFRAME_H4, n_bars=504+90):
  """Fetch OHLC data from MT5"""
```

TIP: I suggest you clear your ‘strategy’ table before each RWEC run to make the export easier. Remember that this table is only a bridge between our data analysis and our EA. Its only purpose is to provide up-to-date strategy parameters to the live trading. It could be any external data source, including text files and web APIs. If you want to preserve this data, you can use a temporary table for backtests as well. The relevant point here is that this data is disposable.

After running the _rwec2db.py_ script with the above n\_bars, window=90, and step=22, our ‘strategy’ table should have this content.

![Fig. 16 - Screen capture of the ‘strategy’ table with RWEC vectors for a one-year backtest](https://c.mql5.com/2/186/Capture_mt5_db_strategy_table_RWEC_504s90.png)

Fig. 16. Screen capture of the ‘strategy’ table with RWEC vectors for a one-year backtest

After exporting this table as detailed in the section above, we should have a TSV file like this in our TERMINAL\_DATA\_PATH.

```
"tstamp"        "test_id"       "name"  "symbols"       "weights"       "timeframe"     "lookback"
1733155200      1       RWEC_CointNasdaq        INTC,AMD,AVGO,MU        [1.0, 0.19818, -0.385289, 0.29447]      H4      90
1734451200      1       RWEC_CointNasdaq        INTC,AMD,AVGO,MU        [1.0, -1.166557, -0.762914, 3.44909]    H4      90
(...)
1764360000      1       RWEC_CointNasdaq        INTC,AMD,AVGO,MU        [1.0, -0.072521, -0.063501, 0.023864]   H4      90
```

Note that our test start date is on December 13, 2024, and the first vector timestamp calculated by RWEC has the timestamp 1733155200, which translates to December 2, 2024. This is nearly two trading weeks before our backtest start date. That means we’ll be starting with a cointegration vector already in place and without requiring the removal of outdated entries.

Also, the second vector has the timestamp 1734451200, which translates to December 17, 2024, right after our backtest start date. That means we’ll be applying portfolio weights updates at a frequency that is a bit larger than our swing trade strategy. This is not ideal. Maybe we can find a better update interval.

Anyway, take note of these frequencies. They will change a bit next, when we start changing the window length and step. When taken relatively to these frequencies, the consequences of those changes will be informative to our analysis.

The time horizon

The time horizon is probably the most critical parameter to be tested and fine-tuned for live trading portfolio rebalancing. That is because it is also the most critical parameter when assessing the portfolio stability with RWEC. The reason is pretty intuitive: the longer the time horizon, the more accurate the evaluation, but with reduced timeliness, that is, the resulting evaluation is less sensitive to the current market structure. On the other hand, while a shorter time horizon gives us an assessment with more weight for the current market structure, it is also more sensitive to noise.

There is no single "optimal" time horizon. It depends on the frequency of our trading strategy and on the mean-reversion half-time of our spread. If our spread mean-reverts in hours or days, a shorter time horizon like 20 to 60 bars might be enough to capture the current relationship dynamics. For spread mean-reversion half-times in the range of weeks or months, a longer window of 120 to 250 bars or more might be necessary to ensure the weights are robustly estimated and not driven by noise.

These are the results we obtained when rebalancing with RWEC 504/90/22 (n\_bars/window/step):

![Fig. 17 - Screen capture of backtest report for portfolio weights rebalancing according to RWEC 504/90/22](https://c.mql5.com/2/186/Capture_backtest_report_RWEC_252_90_22.png)

Fig. 17. Screen capture of backtest report for portfolio weights rebalancing according to RWEC 504/90/22

This is the resulting Balance/Equity graph.

![Fig. 18 - Screen capture of backtest Balance/Equity graph for portfolio weights rebalancing according to RWEC 504/90/22](https://c.mql5.com/2/186/Capture_backtest_graph_RWEC_252_90_22.png)

Fig. 18. Screen capture of backtest Balance/Equity graph for portfolio weights rebalancing according to RWEC 504/90/22

The length of the cointegration test window

Let’s see what we can obtain with the same time horizon and step, but with a 45-day window.

```
def rolling_cointegration(self, data, window=45, step=22):
        """Compute rolling cointegration vectors"""
```

In general terms, the same remarks made above about the trade-offs involved in choosing the optimum time horizon also apply to the test window length. But this parameter is directly involved in the calculation of the eigenvectors, so it directly affects the portfolio weights values. When scoring, it has a direct impact on the calculation of the portfolio weights' stability. But here, when updating on live trading, this impact is reflected in the portfolio turnover. A shorter window leads to higher turnover. It is more adaptive, but also captures more short-term noise. This noise translates into greater fluctuations in the estimated eigenvectors from one window to the next, which is measured by the RWEC (cosine distance). If the RWEC threshold is hit more often, it suggests higher portfolio turnover due to frequent rebalancing.

On the other hand, a longer window leads to lower turnover, increasing the risk of holding a broken pair. A long window leads to slower-changing, smoother eigenvectors, but it also means that the strategy is slow to react when the cointegration relationship breaks down.

The only way to find the optimal length is by backtesting. We should take into account the timeframe and the basket's half-time mean reversion. The objective criterion for the backtest evaluation will guide us in finding the optimal window length. Here we are using the Relative Drawdown as a criterion.These are the results we obtained when rebalancing with RWEC 504/45/22 (n\_bars/window/step):

![Fig. 19 - Screen capture of backtest report for portfolio weights rebalancing according to RWEC 504/45/22](https://c.mql5.com/2/186/Capture_backtest_report_RWEC_252_45_22.png)

Fig. 19. Screen capture of backtest report for portfolio weights rebalancing according to RWEC 504/45/22

This is the resulting Balance/Equity graph when we cut the window parameter in half.

![Fig. 20 - Screen capture of backtest Balance/Equity graph for portfolio weights rebalancing according to RWEC 504/45/22](https://c.mql5.com/2/186/Capture_backtest_graph_RWEC_252_45_22.png)

Fig. 20. Screen capture of backtest Balance/Equity graph for portfolio weights rebalancing according to RWEC 504/45/22

The length of the overlapping windows

Now, we preserve the 90-day window, but reduce the length of the overlapping windows to one trading week.

```
def rolling_cointegration(self, data, window=90, step=5):
        """Compute rolling cointegration vectors"""
```

This parameter represents the number of time periods (trading days) the window moves forward between successive eigenvector calculations. The step size controls the temporal resolution of the signal and the frequency of re-evaluation. This frequency determines how often a new set of portfolio weights (eigenvectors) is calculated and compared against the previous set.

When we choose a small step size like 1 day, we are working with a high-resolution signal. We get a new RWEC comparison value almost every day. This provides a daily check on the stability of the long-term relationship. We can detect a breakdown in the cointegration relationship (a large change in the RWEC) the instant it happens. We can react almost immediately by exiting the trade or accepting the rebalance, which occurs automatically on the backtest. As a drawback, with a very small step size, we need a proportionally high-frequency computation of the eigenvectors. This can be an issue for a large portfolio, but probably will not be one for the average retail trader (our focus here), who usually does not deal with large portfolios.

A larger step size, like one trading month (~22 days), means a relatively low-resolution signal. We’ll be re-evaluating the portfolio weights and the RWEC signal once a month. It reduces computational burden significantly, but the signal loses timeliness. If the cointegration breaks on day 1, we won’t detect the instability or need of rebalance until day 20. This delay can lead to substantial losses in a fast-moving market.

So, the step size has a direct relationship with how often we will be rebalancing the portfolio weights. We can choose to have the portfolio weights always based on the most current market data, with the hedge as close to optimal as possible. In this case, we’ll need to cope with higher transaction costs (commissions, slippage), which can compromise the small margins we typically have on statistical arbitrage. Or we can choose a lower turnover and lower transaction costs. Our basket will remain with stale weights for the duration of the step. In volatile markets, we’ll be increasing our risk considerably.

![Fig. 21 - Screen capture of backtest report for portfolio weights rebalancing according to RWEC 504/90/5](https://c.mql5.com/2/186/Capture_backtest_report_RWEC_252_90_5.png)

Fig. 21. Screen capture of backtest report for portfolio weights rebalancing according to RWEC 504/90/5

![Fig. 22 - Screen capture of backtest Balance/Equity graph for portfolio weights rebalancing according to RWEC 504/90/5](https://c.mql5.com/2/186/Capture_backtest_graph_RWEC_252_90_5.png)

Fig. 22. Screen capture of backtest Balance/Equity graph for portfolio weights rebalancing according to RWEC 504/90/5

Comparative table of relative drawdowns for the same time horizon with different RWEC window length and step.

| n\_bars | window length | overlapping step | resulting data points | relative drawdown |
| --- | --- | --- | --- | --- |
| 504+90 | 90 | 22 | 23 | 18.18% |
| 504+90 | 45 | 22 | 25 | 36.08% |
| 504+90 | 90 | 5 | 101 | 85.11% |

Table 1. Comparison of backtests relative drawdown metric for different RWEC window length and step

The main purpose here is to show how each RWEC’s main parameter affects the rebalancing of weights. As you start experimenting with different baskets, you quickly understand that the only viable way to find the optimum set of parameters is by exhaustive testing.

However, even this simple comparison can show us that when we reduced our window length to half, we also degraded our results by one hundred percent, from 18.08% to 36.08% of relative drawdown, clearly showing us that the first option is more fit to our basket.

When we reduced our step from one trading month (22 days) to one trading week (5 days), we degraded our results dramatically from 18.8% to 85.11%. However, by having nearly five times more data points, from 23 to 101, we also spotted a failure in the cointegration. In this last run, the resulting graph shows what seems to be a structural break that was not detected with the larger rolling windows step.

I hope that these quick evaluations with three RWEC parameter combinations can give you both numerical and visual evidence of the impact that each of them can cause in our backtest, and also can reinforce the role of the overlapping window step in early detection of structural breaks.The detection of this kind of cointegration break is the subject of our next talk.

### Conclusion

In this article, we presented one possible method to backtest the update of portfolio weights of a basket of cointegrated stocks. We showed that by loading CSV/TSV data in a HashMap generic collection and reading them sequentially with backtest-aligned timestamps, we can simulate live trading updates.

The method used to calculate the new weights is the Rolling Windows Eigenvector Comparison (RWEC). We presented a brief description of the relative impact of each of its parameters in the backtest results: the time horizon, or backtest period, the cointegration vector calculation timespan, or the window length, and the overlapping windows period, or the forward step. For each of these parameters, we presented the results of a backtest to demonstrate how their analysis can guide us in choosing the optimal parameters for live trading.

We provide a Python script for running the RWEC and storing its results in a dedicated database table to be exported as CSV/TSV, and a header file with the MQL functions required to read the data in the Tester.

| Filename | Description |
| --- | --- |
| Experts\\StatArb\\CointNasdaq.mq5 | Sample Expert Advisor main MQL5 file |
| Include\\StatArb\\CointNasdaq.mqh | Sample Expert Advisor main MQL5 header file |
| Include\\StatArb\\TestHelper.mqh | Sample Expert Advisor test helper header file |
| Files\\StatArb\\strategy\_\*.csv | Database exported TSV files used in the article |
| rwec2db | Python script to run RWEC and store its results in the integrated SQLite database |
| CointNasdaq.INTC.H4.20241213\_20251213.000 | Backtest settings used in the article |

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/20657.zip "Download all attachments in the single ZIP archive")

[MQL5-article-20657-files.zip](https://www.mql5.com/en/articles/download/20657/MQL5-article-20657-files.zip "Download MQL5-article-20657-files.zip")(18.41 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Statistical Arbitrage Through Cointegrated Stocks (Part 8): Rolling Windows Eigenvector Comparison for Portfolio Rebalancing](https://www.mql5.com/en/articles/20485)
- [Statistical Arbitrage Through Cointegrated Stocks (Part 7): Scoring System 2](https://www.mql5.com/en/articles/20173)
- [Statistical Arbitrage Through Cointegrated Stocks (Part 6): Scoring System](https://www.mql5.com/en/articles/20026)
- [Statistical Arbitrage Through Cointegrated Stocks (Part 5): Screening](https://www.mql5.com/en/articles/19626)
- [Statistical Arbitrage Through Cointegrated Stocks (Part 4): Real-time Model Updating](https://www.mql5.com/en/articles/19428)
- [Statistical Arbitrage Through Cointegrated Stocks (Part 3): Database Setup](https://www.mql5.com/en/articles/19242)

**[Go to discussion](https://www.mql5.com/en/forum/502167)**

![Creating Custom Indicators in MQL5 (Part 2): Building a Gauge-Style RSI Display with Canvas and Needle Mechanics](https://c.mql5.com/2/186/20632-creating-custom-indicators-logo__1.png)[Creating Custom Indicators in MQL5 (Part 2): Building a Gauge-Style RSI Display with Canvas and Needle Mechanics](https://www.mql5.com/en/articles/20632)

In this article, we develop a gauge-style RSI indicator in MQL5 that visualizes Relative Strength Index values on a circular scale with a dynamic needle, color-coded ranges for overbought and oversold levels, and customizable legends. We utilize the Canvas class to draw elements like arcs, ticks, and pies, ensuring smooth updates on new RSI data.

![Pure implementation of RSA encryption in MQL5](https://c.mql5.com/2/185/20273-pure-implementation-of-rsa-logo__1.png)[Pure implementation of RSA encryption in MQL5](https://www.mql5.com/en/articles/20273)

MQL5 lacks built-in asymmetric cryptography, making secure data exchange over insecure channels like HTTP difficult. This article presents a pure MQL5 implementation of RSA using PKCS#1 v1.5 padding, enabling safe transmission of AES session keys and small data blocks without external libraries. This approach provides HTTPS-like security over standard HTTP and even more, it fills an important gap in secure communication for MQL5 applications.

![From Novice to Expert: Navigating Market Irregularities](https://c.mql5.com/2/186/20645-from-novice-to-expert-navigating-logo.png)[From Novice to Expert: Navigating Market Irregularities](https://www.mql5.com/en/articles/20645)

Market rules are continuously evolving, and many once-reliable principles gradually lose their effectiveness. What worked in the past no longer works consistently over time. Today’s discussion focuses on probability ranges and how they can be used to navigate market irregularities. We will leverage MQL5 to develop an algorithm capable of trading effectively even in the choppiest market conditions. Join this discussion to find out more.

![Larry Williams Market Secrets (Part 1): Building a Swing Structure Indicator in MQL5](https://c.mql5.com/2/186/20511-larry-williams-market-secrets-logo.png)[Larry Williams Market Secrets (Part 1): Building a Swing Structure Indicator in MQL5](https://www.mql5.com/en/articles/20511)

A practical guide to building a Larry Williams–style market structure indicator in MQL5, covering buffer setup, swing-point detection, plot configuration, and how traders can apply the indicator in technical market analysis.

[![](https://www.mql5.com/ff/sh/bhdtjfb1zry09943z2/267b575d2182c180804d340af38ce02c.jpg)\\
Trade from your iPhone or Android device\\
\\
You only need an internet connection to use the new powerful MetaTrader 5 Web terminal\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=wtigumvtenarnsocpyfoqnanxrilnbxx&s=ec8c539e52b83881ff2d16eaff6913b25803952eb277cac55f670a102b2edc1f&uid=&ref=https://www.mql5.com/en/articles/20657&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5062528022873023453)

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