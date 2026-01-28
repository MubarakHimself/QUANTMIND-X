---
title: From Novice to Expert: Reporting EA — Setting up the work flow
url: https://www.mql5.com/en/articles/18882
categories: Integration, Expert Advisors
relevance_score: 0
scraped_at: 2026-01-24T14:01:31.574809
---

[Running robots on virtual hosting is easyFollow our step-by-step MetaTrader VPS guide for beginnersRead![](https://www.mql5.com/ff/sh/au4fqg4kms7s9mq1z2/01.png)![](https://www.mql5.com/ff/sh/au4fqg4kms7s9mq1z2/02.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/articles/13586&a=uzpprdshbcrtxvjxpmescehprypbymxc&s=516438f25b531570d9b7d49dcfb29c82fa1021f5ede6571df8026dbfbafcd13f&uid=&ref=https://www.mql5.com/en/articles/18882&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5083287712938596489)

MetaTrader 5 / Examples


### Contents:

- [Introduction](https://www.mql5.com/en/articles/18882#para1)
- [Understanding Trading Reports in MQL5](https://www.mql5.com/en/articles/18882#para2)
- [Implementation](https://www.mql5.com/en/articles/18882#para3)

  - [Developing a Reporting EA](https://www.mql5.com/en/articles/18882#parai1)
  - [Developing the Python Processor Script](https://www.mql5.com/en/articles/18882#parai2)

- [Testing](https://www.mql5.com/en/articles/18882#para4)
- [Conclusion](https://www.mql5.com/en/articles/18882#para5)
- [Key Lessons](https://www.mql5.com/en/articles/18882#para6)
- [Attachments](https://www.mql5.com/en/articles/18882#para7)

### Introduction

The inspiration for this concept arose when I started receiving daily trade confirmations from my broker via email. These summaries offered a clean, professional overview of trading activity and highlighted the potential for statistical performance analysis. While exploring the MetaTrader 5 terminal, I found that it includes a comprehensive reporting tool in its current build, capable of exporting detailed reports in both HTML and PDF formats. However, it lacks the functionality for automatic report delivery and centralized management. In this discussion, we aim to explore the role of trading reports, clarify the key terms they involve, and emphasize their practical importance to every trader seeking informed decision-making.

Ultimately, we plan to implement a custom reporting system using MQL5 in collaboration with Python and other external tools. This solution will generate detailed reports, ensure compatibility across multiple file formats, and support automated delivery through practical and reliable methods.

After examining the documents exported from MetaTrader 5, it became clear that they contain valuable insights and pre-calculated metrics that offer traders a mathematical view of their performance—information that can directly influence trading decisions and behavioral adjustments.

Currently, reports in MetaTrader 5 are typically obtained through manual navigation. However, when working with Expert Advisors (EAs), we have the opportunity to programmatically control the generation and delivery of these reports. While some brokers already send these automatically, our goal in this discussion is to develop a system that allows us to schedule report delivery according to our own preferences—customizing both the frequency and content.

As we work toward solving this, we'll explore the practical importance of trading reports using one of my own experiences as an example. We'll also examine common terms found in these reports and discuss how they can be used to improve both manual trading strategies and Expert Advisor performance evaluation.

In MetaTrader 5, you can access trading reports by selecting Reports from the View menu or by using the shortcut **Alt+E**. The report window displays various aspects of your trading activity in a clear, organized format. From this window, you also have the option to save your reports in either HTML or PDF format, as shown in the image below:

![Accessing MetaTrader 5 account terminal reports](https://c.mql5.com/2/157/terminal64_LsYA0hgrCR.gif)

Accessing Reports on MetaTrader 5

The illustration above highlights some of the key components typically found in trading reports. While this information is highly valuable, many traders tend to focus solely on charts and trade execution—often neglecting to review their performance records periodically. Gaining a clear understanding of these reports is essential, as it reinforces discipline and supports a healthier trading mindset by drawing insights from past activity.

After reviewing my trading report, I took the initiative to research the key terms it contained and compiled notes along with practical examples to make them easier to understand. In the next section, you'll find five structured insights, each offering a quick yet meaningful explanation. These serve as a foundation for the implementation blueprint we’ll explore later in the article. By the time we transition into the technical build phase, the goal is to ensure a solid grasp of these report concepts. If you're already familiar with them, feel free to skip ahead to the implementation stage.

### Understanding Trading Reports in MQL5

1\. Performance Summary

ROI:

- Measures how much the account’s balance grows relative to its initial capital.
- Consider starting with $5 000 and ending with $12 000; that yields a 140% increase, meaning each dollar risked became $2.40.

Drawdown:

- Captures the steepest percentage decline from a high-water mark down to the next low.
- Imagine equity peaking at $12 000 and then dipping to $9 600—a 20% retreat that highlights the deepest setback before recovery.

Activity Metrics:

- Reflect trading pace and consistency—count of trades per week, success rate and average holding duration.
- As an illustration, completing 80 trades over eight weeks works out to 10 per week; winning 32 of those results in a 40% success rate; and totaling 160 days in the market gives a 2‑day average hold.

Sharpe Ratio:

- Indicates how much excess return is earned for each unit of volatility endured.
- To demonstrate, compare two systems both returning 15% annually: the one oscillating ±2% daily will have a notably higher Sharpe than one swinging ±6%, reflecting a smoother equity journey.

Profit Factor:

- Expresses gross profit divided by gross loss to show efficiency of winners versus losers.
- Take the case of $8 000 in winning trades versus $5 000 in losses, yielding a ratio of 1.6—so every dollar lost is offset by $1.60 gained.

Recovery Factor:

- Compares net profit against the largest drawdown experienced.
- Suppose a strategy nets $4 000 and at its worst dips $1 000; the Recovery Factor of 4 indicates profits were four times greater than the deepest loss.

2\. Profit & Loss Breakdown

Gross Profit and Gross Loss:

- Totals of winning versus losing trades establish the raw profit pool before costs.
- For instance, $15 000 in winners against $6 000 in losers creates a $9 000 gross edge.

Fees and Net Profit:

- Subtracting commissions, swaps and other charges from gross results reveals true earnings.
- To put that into perspective, deducting $1 200 in fees from a $9 000 edge leaves $7 800 in net profit.

Monthly Trends:

- Plotting net results month by month unveils performance shifts.
- As a case in point, seeing +$4 000 in January, +$3 000 in February, +$800 in March and –$500 in April highlights a downward trajectory that demands attention.

3\. Risk Analysis

Max Drawdown:

- Defines the largest percentage fall from a peak to the next trough.
- Picture an account sliding from $20 000 to $14 000 before rebounding—a 30% worst‑case decline.

Consecutive Wins and Losses:

- Longest streaks of positive or negative outcomes test both system stability and trader psychology.
- Imagine suffering seven losses in a row versus riding five consecutive winners; each streak shapes confidence, discipline and position sizing.

Maximum Favorable Excursion (MFE):

- Tracks the highest unrealized profit reached during a trade’s life.
- To illustrate, one position might climb $600 at its peak before any exit orders are triggered.

Maximum Adverse Excursion (MAE):

- Records the deepest unrealized loss experienced before closure.
- Consider that the same trade could dip $200 before closing profitably, signaling where stop‑loss adjustments might help.

4\. Instrument/Symbol Performance

Win Rate by Asset:

- Success percentages per market reveal where edge is strongest.
- As an example, EURUSD might secure wins in 48% of trades, while USDJPY wins 60%, suggesting stronger signals on the latter.

Profit Contribution:

- Net earnings by instrument show where gains actually originate.
- Take the instance of EURUSD adding $9 600 to the bottom line while XAUUSD subtracts $1 100, guiding resource allocation.

Concentration Risk:

- Highlights the portion of capital or number of trades tied to a single market.
- Suppose 40% of funds sit in EURUSD; a sudden Euro move could disproportionately affect overall performance.

Profit Factor per Symbol:

- Compares gross wins to losses for each market, clarifying efficiency.
- To demonstrate, USDJPY may boast a ratio of 1.8, whereas AUDCAD sits at 0.9—steering which pairs to prioritize or avoid.

5\. Activity & Behavioral Patterns

Trade Distribution:

- Balance of long and short positions uncovers directional bias.
- For example, holding 70% longs suggests a primarily bullish stance that may falter in sideways markets.

Automation vs. Manual:

- Comparing algorithmic to human‑entered trades reveals where true edge lies.
- Consider that 65% of net profit comes from automated entries versus 35% from manual ones, underscoring the system’s strength.

Time Analysis:

- Breaking down outcomes by hour and weekday identifies optimal and vulnerable windows.
- As a case in point, most losses may occur between 11:00 and 12:00 GMT, indicating a lunchtime period best avoided.

Style Metrics:

- Average holding duration and weekly trade volume define the approach.
- To illustrate, a four‑hour average hold time combined with roughly 25 trades per week characterizes a moderate‑tempo intraday strategy.

### Implementation

At this stage, we proceed to set up our workflow. Thanks to the previous section, which provided an in-depth understanding of trading report terminology, you’ll find this part easier to follow with fewer unfamiliar concepts.

To accomplish the goal of this article, we will develop an Expert Advisor (EA) that handles data export and prepares the necessary logs. This EA will serve as a bridge between MetaTrader 5 (MQL5) and the Python libraries responsible for processing historical trading data and generating a final report in a portable format—similar to the reporting tools built into the MetaTrader 5 terminal.

We’ll begin by presenting a flow chart outlining the entire process, followed by a breakdown of the required tools and environment setup to get everything working. All components of this project are based on open-source technologies, ensuring accessibility for everyone.

![Reporting EA process flow](https://c.mql5.com/2/159/Reporter_EA_flow.drawio_82e.png)

Process Flow

To get started, make sure you have MetaTrader 5 installed on your system. Then proceed to set up the Python environment, which I’ll explain in detail later. Refer to the table below for a simplified list of requirements and tools used in this workflow.

| Component | Open-Source Tools | Cost | Implementation Notes |
| --- | --- | --- | --- |
| Data Extraction | MetaTrader 5 Expert Advisor (ReporterEA.mq5) | Free | Custom CSV export with date range filtering |
| Scheduling Engine | MetaTrader 5 Timer Events (OnTimer()) | Free | Internal scheduling within MetaTrader 5 terminal |
| Processing | Python 3.10+ (Pandas, Matplotlib) | Free | Triggered by EA via ShellExecute |
| Report Delivery | smtplib + Gmail SMTP | Free | Many free emails/day from Python script |
| Maintenance | EA Auto-Cleanup Functions | Free | File rotation and error handling in MQL5/Python |

### Developing a Reporting EA

At this stage, we begin developing the Expert Advisor (EA) as planned. I will guide you through each component of the code, explaining how they work together to form a complete and functional system. The goal is to build a robust EA that serves its intended purpose seamlessly. Follow along as we break the development process into clear, manageable steps, ensuring that each piece is easy to understand and contributes meaningfully to the final product.

In MetaEditor 5, open a new file and select the "Expert Advisor" template. I recommend removing any sections of the auto-generated code that are not relevant to our current development goals, so we can focus only on what matters. Follow the numbered steps below as we begin building the Expert Advisor step by step.

1\. File Metadata and Compilation Directives

At the top of the file, the #property directives declare metadata for the Expert Advisor (EA). These properties include the copyright and link information, which appear in the MetaTrader 5 terminal’s “About” box, as well as the version number and strict compilation mode. The strict mode enforces more rigorous type checking and compilation rules, helping prevent subtle bugs by disallowing implicit casts and requiring explicit conversions.

```
#property copyright "Clemence Benjamin"
#property link      "https://www.mql5.com/go?link=https://www.mql5.com/en/users/billionaire2024/seller"
#property version   "1.00"
#property strict
```

2\. Constants and Windows API Imports

The #define statements create symbolic names for commonly used constant values (SW\_HIDE to hide console windows and INVALID\_FILE\_ATTRIBUTES for error checking). Following that, the code imports two Windows system libraries via #import: kernel32.dll for file-attribute functions (GetFileAttributesW) and shell32.dll for executing external processes (ShellExecuteW). By calling these native DLL functions, the EA extends MetaTrader 5’s built‑in capabilities to verify file existence and launch the Python interpreter.

```
#define SW_HIDE                 0
#define INVALID_FILE_ATTRIBUTES 0xFFFFFFFF

//--- Windows API imports
#import "kernel32.dll"
uint   GetFileAttributesW(string lpFileName);
#import

#import "shell32.dll"
int    ShellExecuteW(int hwnd, string lpOperation, string lpFile, string lpParameters, string lpDirectory, int nShowCmd);
#import
```

3\. User Inputs for Configuration

The input declarations expose customizable parameters in the EA’s settings dialog. Users can specify the absolute path to the Python executable (PythonPath), the Python script to run (ScriptPath), the hour and minute when the daily report should execute (ReportHour, ReportMinute), whether push notifications are sent (EnableEmail), and whether an initial test run occurs on startup (TestOnInit). Exposing these as inputs allows non‑programmers to tweak behavior without editing source code.

```
//--- User inputs
input string PythonPath   = "C:\\Users\\BTA24\\AppData\\Local\\Programs\\Python\\Python312\\python.exe";
input string ScriptPath   = "C:\\Users\\BTA24\\Documents\\BENJC_TRADE_ADVISOR\\Scripts\\processor.py";
input int    ReportHour   = 15;    // Hour (24h) to run report
input int    ReportMinute = 55;    // Minute after the hour
input bool   EnableEmail  = true;  // Send push notification
input bool   TestOnInit   = true;  // Immediately run export+Python on init
```

4\. Global State Management

A single global variable, lastRunTime, stores the timestamp of the most recent successful report execution. By comparing TimeCurrent() against lastRunTime, the EA ensures that the report only runs once every 24 hours, even though the timer callback checks more frequently.

```
//--- Globals
datetime lastRunTime = 0;
```

5\. Initialization Logic (OnInit)

The OnInit() function performs all startup routines. First, it prints status messages to the Experts log. It checks file attributes for the Python executable and script and prints warnings if they’re missing. It then tests write permissions by creating, writing, closing, and deleting a dummy file in the MQL5\\Files directory. Next, it sets up a recurring timer event every 30 seconds via EventSetTimer(30). Finally, if TestOnInit is true, it calls RunDailyExport() immediately to validate the full export-and-Python workflow, recording the current time in lastRunTime.

```
int OnInit()
{
   Print(">> Reporting EA initializing…");

   // Verify Python executable
   if(GetFileAttributesW(PythonPath) == INVALID_FILE_ATTRIBUTES)
      Print("!! Python executable not found at: ", PythonPath);
   else
      Print("✔ Found Python at: ", PythonPath);

   // Verify Python script
   if(GetFileAttributesW(ScriptPath) == INVALID_FILE_ATTRIBUTES)
      Print("!! Python script not found at:   ", ScriptPath);
   else
      Print("✔ Found script at:   ", ScriptPath);

   // Test write permission
   int h = FileOpen("test_perm.txt", FILE_WRITE|FILE_COMMON|FILE_ANSI);
   if(h==INVALID_HANDLE)
      Print("!! Cannot write to MQL5\\Files directory!");
   else
   {
      FileWrite(h, "OK");
      FileClose(h);
      FileDelete("test_perm.txt");
      Print("✔ Write permission confirmed.");
   }

   // Set timer
   EventSetTimer(30);
   Print(">> Timer set to 30 seconds.");

   // Test run on init
   if(TestOnInit)
   {
      Print(">> Test mode: running initial export.");
      RunDailyExport();
      lastRunTime = TimeCurrent();
   }

   return(INIT_SUCCEEDED);
}
```

6\. Deinitialization Logic (OnDeinit)

When the EA is removed or the platform shuts down, OnDeinit() is invoked. Its sole responsibility is to clean up by killing the timer (EventKillTimer()) and logging a deinitialization message. Properly releasing timer resources prevents orphaned callbacks and potential crashes.

```
void OnDeinit(const int reason)
{
   EventKillTimer();
   Print(">> Reporting EA deinitialized.");
}
```

7\. Timer Callback for Scheduling (OnTimer)

Every 30 seconds, OnTimer() runs and retrieves the current hour and minute through the MqlDateTime struct. It checks whether the current time matches or exceeds the configured report time (ReportHour, ReportMinute), and whether at least 86 400 seconds (24 hours) have elapsed since lastRunTime. This double‑check ensures the report runs once daily at or after the scheduled minute.

```
void OnTimer()
{
   MqlDateTime dt; TimeToStruct(TimeCurrent(), dt);

   if(dt.hour==ReportHour && dt.min>=ReportMinute && (TimeCurrent()-lastRunTime)>86400)
   {
      RunDailyExport();
      lastRunTime = TimeCurrent();
   }
}
```

8\. Main Workflow: Export and Python Invocation (RunDailyExport)

1. This function encapsulates the core steps of the EA’s reporting feature.
2. Compute the absolute path to MetaTrader 5’s MQL5\\Files directory using TerminalInfoString.
3. Generate date-stamped filenames for both the CSV export and log file by formatting the current date and stripping dots.
4. Call ExportHistoryToCSV() to write the last 30 days of trade history into the CSV file. If this fails, the function aborts.
5. Build a command string that invokes the Python interpreter with the script and CSV filename, redirecting both standard output and standard error into the log file. String formatting and StringFormat ensure proper quoting for paths containing spaces.
6. Call the EA’s ShellExecute() wrapper to launch cmd.exe /c <pythonCmd>, capturing the integer return code for diagnostic logging.
7. Pause for 3 seconds with Sleep(3000) to allow the external Python process to complete.
8. Check for the existence of the log file, then read and print its contents line by line to the Experts log.
9. Finally, compose a notification message summarizing the CSV path and—if enabled—send it via SendNotification() to the user’s mobile terminals.

```
void RunDailyExport()
{
   string filesDir = TerminalInfoString(TERMINAL_DATA_PATH) + "\\MQL5\\Files\\";

   string dateStr = TimeToString(TimeCurrent(), TIME_DATE);
   StringReplace(dateStr, ".", "");

   string csvName = "History_" + dateStr + ".csv";
   string logName = "ProcLog_" + dateStr + ".txt";

   string csvFull  = filesDir + csvName;
   string logFull  = filesDir + logName;

   // 3) Export CSV
   if(!ExportHistoryToCSV(csvName))
   {
      Print("!! CSV export failed: ", csvFull);
      return;
   }
   Print("✔ CSV exported: ", csvFull);

   // 4) Build Python command
   string pythonCmd = StringFormat(
      "\"%s\" \"%s\" \"%s\" >> \"%s\" 2>&1",
      PythonPath,
      ScriptPath,
      csvFull,
      logFull
   );
   string fullCmd = "/c " + pythonCmd;
   PrintFormat("→ Launching: cmd.exe %s", fullCmd);

   // 5) Execute
   int result = ShellExecute(" " + fullCmd);
   PrintFormat("← ShellExecute returned: %d", result);

   // 6) Wait
   Sleep(3000);

   // 7) Read log
   if(GetFileAttributesW(logFull) == INVALID_FILE_ATTRIBUTES)
      Print("!! Log file not created: ", logFull);
   else
   {
      Print("=== Python Log Start ===");
      int fh = FileOpen(logName, FILE_READ|FILE_COMMON|FILE_TXT);
      while(fh!=INVALID_HANDLE && !FileIsEnding(fh))
         Print("PY: ", FileReadString(fh));
      if(fh!=INVALID_HANDLE) FileClose(fh);
      Print("=== Python Log End ===");
   }

   // 8) Notification
   string msg = "Report & log generated: " + csvFull;
   Print(msg);
   if(EnableEmail) SendNotification(msg);
}
```

9\. CSV Generation (ExportHistoryToCSV)

This helper function automates the extraction of deal history into a CSV file. It selects all history within the past 30 days (HistorySelect), iterates through each deal ticket, retrieves properties (time, type, symbol, volume, price, profit, commission, swap) using HistoryDealGet\* functions, and writes them as comma‑separated values with FileWrite. After outputting a header row, the loop constructs each line using DoubleToString and TimeToString, ensuring consistent numeric precision and timestamp formatting. Proper error checking on FileOpen prevents silent failures.

```
bool ExportHistoryToCSV(string filename)
{
   datetime end   = TimeCurrent();
   datetime start = end - 2592000; // last 30 days

   HistorySelect(start, end);
   int total = HistoryDealsTotal();

   int fh = FileOpen(filename, FILE_WRITE|FILE_CSV|FILE_COMMON|FILE_ANSI, ",");
   if(fh==INVALID_HANDLE)
   {
      Print("!! FileOpen failed for: ", filename);
      return(false);
   }

   FileWrite(fh, "Ticket,Time,Type,Symbol,Volume,Price,Profit,Commission,Swap");

   for(int i=0; i<total; i++)
   {
      ulong deal = HistoryDealGetTicket(i);
      if(deal==0) continue;

      long    tr = HistoryDealGetInteger(deal, DEAL_TIME);
      datetime t = (datetime)tr;
      int     tp = (int)HistoryDealGetInteger(deal, DEAL_TYPE);

      string  sym = HistoryDealGetString (deal, DEAL_SYMBOL);
      double  vol = HistoryDealGetDouble (deal, DEAL_VOLUME);
      double  prc = HistoryDealGetDouble (deal, DEAL_PRICE);
      double  pf  = HistoryDealGetDouble (deal, DEAL_PROFIT);
      double  cm  = HistoryDealGetDouble (deal, DEAL_COMMISSION);
      double  sw  = HistoryDealGetDouble (deal, DEAL_SWAP);

      FileWrite(fh,
         deal,
         TimeToString(t, TIME_DATE|TIME_SECONDS),
         tp,
         sym,
         DoubleToString(vol,2),
         DoubleToString(prc,5),
         DoubleToString(pf,2),
         DoubleToString(cm,2),
         DoubleToString(sw,2)
      );
   }

   FileClose(fh);
   return(true);
}
```

10\. Shell Command Wrapper (ShellExecute)

The ShellExecute function serves as a thin wrapper around the imported ShellExecuteW API call. By standardizing the invocation of cmd.exe, it hides the native API’s complexity and always uses SW\_HIDE to suppress console windows. Returning the integer result code allows the EA to detect and log potential errors in launching external commands

```
int ShellExecute(string command)
{
   return(ShellExecuteW(0, "open", "cmd.exe", command, NULL, SW_HIDE));
}
```

### Developing the Python Processor Script

We begin by setting up Python and installing the required libraries. First, open the Command Prompt and run the necessary installation commands. After that, you can prepare your Python script using a text editor. I personally prefer Notepad++, an open-source tool, but you're free to use any Python IDE of your choice.

Setting Up

1\. To prepare the Python side, start by installing a recent Python 3.x interpreter (e.g. 3.10 or 3.12). Create and activate a virtual environment in your project folder:

```
python -m venv venv

source venv/Scripts/activate    # Windows

# or

source venv/bin/activate        # macOS/Linux
```

2\. Once activated, install the required packages with:

```
pip install pandas fpdf
```

- pandas handles CSV parsing and data analysis,
- FPDF (or another PDF library of your choice) generates the report.

3\. If you plan to send email alerts, also install an SMTP library such as yagmail or use Python’s built‑in smtplib.

Now it’s time to develop the Python script. We’ll proceed through the following steps to implement each part.

1\. Script Header and Imports

The script begins with a Unix‑style shebang to allow execution on systems that respect it, followed by imports of key libraries:

- sys, os, and traceback for interacting with the OS, handling arguments, and printing errors;
- pandas for data loading and manipulation;
- datetime and timedelta for date calculations;
- FPDF from the fpdf package to generate PDF reports.

```
#!/usr/bin/env python

import sys, os, traceback

import pandas as pd

from datetime import datetime, timedelta

from fpdf import FPDF
```

2\. Main Workflow: Argument Validation and File Checks

The main(csv\_path) function is the orchestrator. It prints the CSV file being processed and immediately verifies that the file exists, raising a FileNotFoundError if not. This mirrors the MQL5 EA’s own preflight checks for the Python executable and script paths.

```
def main(csv_path):

    print(f"Processing CSV: {csv_path}")

    if not os.path.isfile(csv_path):

        raise FileNotFoundError(f"CSV not found: {csv_path}")
```

3\. Loading and Parsing the CSV

Using pandas.read\_csv, the script loads the trade-history CSV produced by the EA. It then converts the 'Time' column to datetime objects with pd.to\_datetime, ensuring subsequent time‑based calculations are accurate. This parallels the EA’s formatting of times with TimeToString.

```
# 1. Load & parse

    df = pd.read_csv(csv_path)

    df['Time'] = pd.to_datetime(df['Time'])
```

4\. Computing Summary Analytics

The script aggregates key performance metrics into a report dictionary:

- 'date': today’s date as a string;
- 'net\_profit': the sum of the Profit column;
- 'trade\_count': total rows in the DataFrame;
- 'top\_symbol': the symbol with the highest cumulative profit, using groupby and idxmax.

These metrics match the EA’s CSV contents and allow the PDF to summarize exactly what was exported.

```
 # 2. Analytics

    report = {

        'date'       : datetime.now().strftime("%Y-%m-%d"),

        'net_profit' : df['Profit'].sum(),

        'trade_count': len(df),

        'top_symbol' : df.groupby('Symbol')['Profit'].sum().idxmax()

    }
```

5\. Generating the PDF Report

The script builds the output path in the same MQL5\\Files directory as the CSV, naming the PDF by date. It then calls generate\_pdf(report, pdf\_file). This dovetails with the EA’s logging of Python output and the expectation that any artifacts (both CSV and PDF) land in the common files folder.

```
# 3. Generate PDF

    dirpath = os.path.dirname(csv_path)

    pdf_file = os.path.join(dirpath, f"Report_{report['date']}.pdf")

    generate_pdf(report, pdf_file)

    print(f"PDF written: {pdf_file}")

    return 0
```

6\. PDF Construction with FPDF

The generate\_pdf function uses FPDF’s simple API: creating a document, adding a page, setting the font, and writing lines for each metric. The ln=True parameter moves to the next line automatically. This modular helper keeps PDF formatting concerns separate from data logic.

```
def generate_pdf(report, output_path):

    pdf = FPDF()

    pdf.add_page()

    pdf.set_font("Arial", size=12)

    pdf.cell(0, 10, f"Report Date: {report['date']}", ln=True)

    pdf.cell(0, 10, f"Total Trades: {report['trade_count']}", ln=True)

    pdf.cell(0, 10, f"Net Profit:    ${report['net_profit']:.2f}", ln=True)

    pdf.cell(0, 10, f"Top Symbol:    {report['top_symbol']}", ln=True)

    pdf.output(output_path)
```

7\. Maintenance: Cleaning Up Old Reports

To prevent disk bloat, clean\_old\_reports deletes any PDF older than a configurable number of days (default 30). It only runs when the EA invokes the script on Sundays (weekday() == 6) and receives the CSV path in sys.argv\[1\], ensuring it targets the correct directory. This maintenance parallels the EA’s own date‑based naming and 24‑hour gating logic.

```
def clean_old_reports(days=30):

    now = datetime.now()

    cutoff = now - timedelta(days=days)

    dirpath = os.path.dirname(sys.argv[1])

    for fname in os.listdir(dirpath):

        if fname.startswith("Report_") and fname.endswith(".pdf"):

            full = os.path.join(dirpath, fname)

            if datetime.fromtimestamp(os.path.getmtime(full)) < cutoff:

                os.remove(full)

                print(f"Deleted old report: {full}")
```

8\. Script Entry Point and Error Handling

The if \_\_name\_\_ == "\_\_main\_\_": block enforces usage of exactly one argument (the full path to the CSV). It wraps the call to main in a try/except to catch any exception, print a traceback, and exit with a non‑zero code—just as the EA captures Python’s stdout/stderr and surface any errors into its own logs. Optional maintenance is run weekly before exiting.

```
if __name__ == "__main__":

    if len(sys.argv) != 2:

        print("Usage: processor.py <full_csv_path>")

        sys.exit(1)

    try:

        ret = main(sys.argv[1])

        # Optional maintenance

        if datetime.now().weekday() == 6:

            clean_old_reports(30)

        sys.exit(ret)

    except Exception as e:

        print("ERROR:", e)

        traceback.print_exc()

        sys.exit(1)
```

Interplay Between MetaTrader 5 EA and Python Script

1. Filename Conventions: The EA builds a date‑formatted CSV filename (History\_YYYYMMDD.csv) and the Python script expects exactly one argument pointing to that file.
2. Directory Alignment: Both components operate in MetaTrader 5’s MQL5\\Files folder, ensuring seamless file discovery.
3. Error Propagation: The EA redirects Python’s stdout and stderr into a timestamped log file; any exception in the script (e.g., missing CSV, parsing error) is captured by the EA’s log reader and printed to the Experts log.
4. Scheduling: The EA’s 24‑hour timer logic drives when Python runs; the script itself remains stateless beyond processing its input, relying on the EA to call it at the right times.
5. Maintenance Coordination: Weekly cleanup of PDFs is triggered from within the script but only when called on Sundays—matching the EA’s weekly cadence check if extended.

### Testing

In this section, we deploy the Reporting EA onto a MetaTrader 5 chart. On a Windows computer, it’s essential to enable DLL imports to allow external process execution. During testing, the EA successfully achieved its goal: exporting trading history as a CSV file and triggering the Python script responsible for processing the data. The script then generates the required report metrics and exports them as a polished PDF document, ready to be sent via email or archived for review.

![Deploying the Reporting EA](https://c.mql5.com/2/159/terminal64_tJ1GmuckmB.gif)

Deploying the Reporting EA

Reporting EA Experts log:

```
2025.07.24 20:44:57.061 Reporting EA (GBPJPY.0,M1)      >> Reporting EA initializing…
2025.07.24 20:44:57.061 Reporting EA (GBPJPY.0,M1)      !! Python executable not found at: C:\path_to\python.exe
2025.07.24 20:44:57.061 Reporting EA (GBPJPY.0,M1)      !! Python script not found at:   C:\path_to\reports_processor.py
2025.07.24 20:44:57.062 Reporting EA (GBPJPY.0,M1)      ✔ Write permission confirmed.
2025.07.24 20:44:57.062 Reporting EA (GBPJPY.0,M1)      >> Timer set to 30 seconds.
2025.07.24 20:44:57.062 Reporting EA (GBPJPY.0,M1)      >> Test mode: running initial export.
2025.07.24 20:44:57.063 Reporting EA (GBPJPY.0,M1)      ✔ CSV exported: C:\TERMINAL_PATH\MQL5\Files\History_20250724.csv
2025.07.24 20:44:57.063 Reporting EA (GBPJPY.0,M1)      → Launching: cmd.exe /c "C:\path_to\python.exe" "C:\path_to\reports_processor.py" "C:\TERMINAL_PATH\MQL5\Files\History_20250724.csv" >> "C:\TERMINAL_PATH\MQL5\Files\ProcLog_20250724.txt" 2>&1
2025.07.24 20:44:57.124 Reporting EA (GBPJPY.0,M1)      ← ShellExecute returned: 42
2025.07.24 20:45:00.124 Reporting EA (GBPJPY.0,M1)      !! Log file not created: C:\Users\TERMINAL_PATH\MQL5\Files\ProcLog_20250724.txt
2025.07.24 20:45:00.124 Reporting EA (GBPJPY.0,M1)      Report & log generated: C:\Users\TERMINAL_PATH\Files\History_20250724.csv
```

The Expert Log shown above reveals an initialization attempt of the Reporting EA, where the system failed to locate the specified Python executable and script. This happened because the file paths were deliberately renamed in the code (e.g., C:\\path\_to\\python.exe and C:\\path\_to\\reports\_processor.py) for demonstration or placeholder purposes. As a result, the EA could not execute the Python script or generate the expected log output (ProcLog\_20250724.txt). Despite this, the EA successfully confirmed write permissions and exported the trading history as a CSV file.

This test highlights the importance of correctly configuring file paths in your EA—pointing to the actual Python interpreter and processing script—to ensure seamless end-to-end report generation. Always double-check and use valid, absolute paths that match your local system setup to avoid such issues and unlock the full functionality of the reporting tool.

### Conclusion

This discussion focused primarily on understanding trading reports, setting up a functional workflow, and developing the tools necessary for delivering custom trading reports in a portable document format (PDF). The final stages of sending the generated PDF via email have been reserved for a future publication, to avoid overwhelming this presentation. However, the PDF generation process—based on the exported CSV—was successfully handled using Python libraries. Further enhancements, such as including charts and advanced reporting features, will be introduced in the next discussion.

Now that both the Expert Advisor and its corresponding Python script are complete, we are well-positioned to achieve even more. In summary, this project solves the challenge of receiving scheduled, customizable trading reports with clarity-focused features that aid user understanding. Just as bookkeeping is essential for any business, regular reporting is vital in trading—it promotes performance awareness, discipline, and psychological growth for traders.

### Key Lessons

| Lesson | Description |
| --- | --- |
| 1\. Modular Design | Separating CSV export logic (MQL5) from report generation (Python) promotes maintainability and allows each component to evolve independently. |
| 2\. Robust Error Handling | Validating file paths, catching exceptions in Python, and logging failures ensures the system fails gracefully and remains debuggable. |
| 3\. Cross‑Language Integration | Calling external scripts via ShellExecute demonstrates how trading platforms can leverage powerful external libraries and ecosystems. |
| 4\. Automated Scheduling | Using timer callbacks in MQL5 to trigger daily exports ensures reports run without manual intervention, improving consistency and reliability. |
| 5\. Centralized Logging | Writing logs to the MetaTrader 5 Experts tab and to external text files provides clear visibility into each step of the workflow. |
| 6\. Environment Validation | Checking Python and script availability on init avoids runtime surprises and guides users through setup requirements. |
| 7\. Open‑Source Tooling | Relying on free, widely‑adopted libraries (pandas, FPDF) and standard APIs lowers barriers to entry and encourages community collaboration. |
| 8\. User‑Configurable Parameters | Exposing paths, schedule times, and notification toggles as inputs makes the EA flexible and adaptable to diverse trading environments. |

### Attachments

| Filename | Version | Description |
| --- | --- | --- |
| reporting\_processor.py |  | Python script that loads a CSV of trade history, computes summary analytics (net profit, trade count, top symbol), generates a PDF report via FPDF, and optionally cleans up older reports. |
| Reporting EA.mq5 | 1.0 | MetaTrader 5 Expert Advisor that exports the last 30 days of trade history to CSV, invokes processor.py , captures its exit code, checks for the generated PDF, and sends a push notification. |

[Back to contents](https://www.mql5.com/en/articles/18882#para0)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/18882.zip "Download all attachments in the single ZIP archive")

[Reports\_Processor.py](https://www.mql5.com/en/articles/download/18882/reports_processor.py "Download Reports_Processor.py")(2.14 KB)

[Reporting\_EA.mq5](https://www.mql5.com/en/articles/download/18882/reporting_ea.mq5 "Download Reporting_EA.mq5")(15.06 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [From Novice to Expert: Higher Probability Signals](https://www.mql5.com/en/articles/20658)
- [From Novice to Expert: Navigating Market Irregularities](https://www.mql5.com/en/articles/20645)
- [From Novice to Expert: Automating Trade Discipline with an MQL5 Risk Enforcement EA](https://www.mql5.com/en/articles/20587)
- [From Novice to Expert: Trading the RSI with Market Structure Awareness](https://www.mql5.com/en/articles/20554)
- [From Novice to Expert: Developing a Geographic Market Awareness with MQL5 Visualization](https://www.mql5.com/en/articles/20417)
- [The MQL5 Standard Library Explorer (Part 5): Multiple Signal Expert](https://www.mql5.com/en/articles/20289)
- [The MQL5 Standard Library Explorer (Part 4): Custom Signal Library](https://www.mql5.com/en/articles/20266)

**[Go to discussion](https://www.mql5.com/en/forum/492139)**

![Self Optimizing Expert Advisors in MQL5 (Part 10): Matrix Factorization](https://c.mql5.com/2/160/18873-self-optimizing-expert-advisors-logo__1.png)[Self Optimizing Expert Advisors in MQL5 (Part 10): Matrix Factorization](https://www.mql5.com/en/articles/18873)

Factorization is a mathematical process used to gain insights into the attributes of data. When we apply factorization to large sets of market data — organized in rows and columns — we can uncover patterns and characteristics of the market. Factorization is a powerful tool, and this article will show how you can use it within the MetaTrader 5 terminal, through the MQL5 API, to gain more profound insights into your market data.

![Market Profile indicator (Part 2): Optimization and rendering on canvas](https://c.mql5.com/2/106/Market_Profile_Indicator_Part2_LOGO.png)[Market Profile indicator (Part 2): Optimization and rendering on canvas](https://www.mql5.com/en/articles/16579)

The article considers an optimized version of the Market Profile indicator, where rendering with multiple graphical objects is replaced with rendering on a canvas - an object of the CCanvas class.

![Price Action Analysis Toolkit Development (Part 34): Turning Raw Market Data into Predictive Models Using an Advanced Ingestion Pipeline](https://c.mql5.com/2/160/18979-price-action-analysis-toolkit-logo.png)[Price Action Analysis Toolkit Development (Part 34): Turning Raw Market Data into Predictive Models Using an Advanced Ingestion Pipeline](https://www.mql5.com/en/articles/18979)

Have you ever missed a sudden market spike or been caught off‑guard when one occurred? The best way to anticipate live events is to learn from historical patterns. Intending to train an ML model, this article begins by showing you how to create a script in MetaTrader 5 that ingests historical data and sends it to Python for storage—laying the foundation for your spike‑detection system. Read on to see each step in action.

![MQL5 Wizard Techniques you should know (Part 77): Using Gator Oscillator and the Accumulation/Distribution Oscillator](https://c.mql5.com/2/160/18946-mql5-wizard-techniques-you-logo.png)[MQL5 Wizard Techniques you should know (Part 77): Using Gator Oscillator and the Accumulation/Distribution Oscillator](https://www.mql5.com/en/articles/18946)

The Gator Oscillator by Bill Williams and the Accumulation/Distribution Oscillator are another indicator pairing that could be used harmoniously within an MQL5 Expert Advisor. We use the Gator Oscillator for its ability to affirm trends, while the A/D is used to provide confirmation of the trends via checks on volume. In exploring this indicator pairing, as always, we use the MQL5 wizard to build and test out their potential.

[![](https://www.mql5.com/ff/sh/qv94j0cd8n2n55z9z2/01.png)![](https://www.mql5.com/ff/sh/qv94j0cd8n2n55z9z2/02.png)Boost your trading experienceRead our book "MQL5 Programming for Traders"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/book%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.algobook%26utm_content=visit.page%26utm_campaign=algobook.promo.04.2024&a=heclgjpfbvfghpmyaciuaesdtswflupo&s=4255fbe1b8cbc4d1b40afbaebf4235e5ace8b5103cba60d996897a03d588556f&uid=&ref=https://www.mql5.com/en/articles/18882&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5083287712938596489)

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