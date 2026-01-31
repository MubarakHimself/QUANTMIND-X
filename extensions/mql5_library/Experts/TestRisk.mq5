//+------------------------------------------------------------------+
//|                                                          TestRisk.mq5 |
//|                        Test EA for QuantMind Risk Library         |
//|                                                                  |
//| Validates all code paths in the QuantMind Risk Management        |
//| Library including fast path (GlobalVariable), fallback path      |
//| (JSON file), and REST heartbeat.                                  |
//+------------------------------------------------------------------+
#property copyright "QuantMindX"
#property link      "https://github.com/quantmindx"
#property version   "1.00"
#property strict

#include <QuantMind/QuantMind_Risk.mqh>

//+------------------------------------------------------------------+
//| Input parameters                                                  |
//+------------------------------------------------------------------+
input string TestSymbol = "EURUSD";            // Symbol to test
input int TestMagicNumber = 12345;              // Magic number for testing
input bool TestGlobalVariablePath = true;       // Test fast path (GlobalVariable)
input bool TestFileFallbackPath = true;         // Test fallback path (JSON file)
input bool TestHeartbeat = true;                // Test REST heartbeat
input int TestHeartbeatIntervalSeconds = 60;    // Heartbeat interval for testing

//+------------------------------------------------------------------+
//| Global state                                                      |
//+------------------------------------------------------------------+
datetime g_lastTestTime = 0;
int g_testCounter = 0;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
    Print("========================================");
    Print("QuantMind Risk Library Test EA Started");
    Print("========================================");
    Print("Test Symbol: ", TestSymbol);
    Print("Magic Number: ", TestMagicNumber);

    // Initialize risk library
    RiskInit();

    // Set up test interval (every 30 seconds for testing)
    g_lastTestTime = TimeCurrent();

    // Run initial tests
    RunAllTests();

    return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    Print("========================================");
    Print("QuantMind Risk Library Test EA Stopped");
    Print("Deinit reason: ", reason);
    Print("Total tests run: ", g_testCounter);
    Print("========================================");

    RiskDeinit(reason);
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
    datetime currentTime = TimeCurrent();

    // Run tests every 30 seconds
    if((currentTime - g_lastTestTime) >= 30)
    {
        RunAllTests();
        g_lastTestTime = currentTime;
    }
}

//+------------------------------------------------------------------+
//| Run all library tests                                            |
//+------------------------------------------------------------------+
void RunAllTests()
{
    g_testCounter++;

    Print("========================================");
    Print("Test Cycle #", g_testCounter, " at ", TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS));
    Print("========================================");

    if(TestGlobalVariablePath)
    {
        TestFastPath();
    }

    if(TestFileFallbackPath)
    {
        TestFallbackPath();
    }

    if(TestHeartbeat)
    {
        TestHeartbeatFunction();
    }

    Print("========================================");
    Print("Test Cycle #", g_testCounter, " Complete");
    Print("========================================");
}

//+------------------------------------------------------------------+
//| Test Fast Path (GlobalVariable)                                  |
//+------------------------------------------------------------------+
void TestFastPath()
{
    Print("--- Test Fast Path (GlobalVariable) ---");

    // Set a test GlobalVariable
    double testMultiplier = 1.5;
    GlobalVariableSet(QM_RISK_MULTIPLIER_VAR, testMultiplier);
    Print("Set GlobalVariable \"", QM_RISK_MULTIPLIER_VAR, "\" = ", testMultiplier);

    // Read it back via GetRiskMultiplier
    double retrievedMultiplier = GetRiskMultiplier(TestSymbol);
    Print("Retrieved multiplier for ", TestSymbol, ": ", retrievedMultiplier);

    // Verify
    if(MathAbs(retrievedMultiplier - testMultiplier) < 0.0001)
    {
        Print("PASS: Fast path working correctly");
    }
    else
    {
        Print("FAIL: Fast path mismatch (expected ", testMultiplier, ", got ", retrievedMultiplier, ")");
    }

    // Clean up
    GlobalVariableDel(QM_RISK_MULTIPLIER_VAR);
    Print("Cleaned up GlobalVariable");
}

//+------------------------------------------------------------------+
//| Test Fallback Path (JSON file)                                   |
//+------------------------------------------------------------------+
void TestFallbackPath()
{
    Print("--- Test Fallback Path (JSON File) ---");

    // Create a test JSON file
    string testJson = "{";
    testJson += "\"" + TestSymbol + "\": {";
    testJson += "\"multiplier\": 2.0, ";
    testJson += "\"timestamp\": " + IntegerToString((int)TimeCurrent());
    testJson += "}, ";
    testJson += "\"GBPUSD\": {";
    testJson += "\"multiplier\": 1.25, ";
    testJson += "\"timestamp\": " + IntegerToString((int)TimeCurrent());
    testJson += "}";
    testJson += "}";

    // Write test file
    int handle = FileOpen(QM_RISK_MATRIX_FILE, FILE_WRITE|FILE_TXT|FILE_ANSI, '\0');
    if(handle != INVALID_HANDLE)
    {
        FileWriteString(handle, testJson);
        FileClose(handle);
        Print("Created test ", QM_RISK_MATRIX_FILE, " with content:");
        Print(testJson);

        // Read it back via GetRiskMultiplier
        double retrievedMultiplier = GetRiskMultiplier(TestSymbol);
        Print("Retrieved multiplier for ", TestSymbol, ": ", retrievedMultiplier);

        // Verify (should be 2.0 from file)
        if(MathAbs(retrievedMultiplier - 2.0) < 0.0001)
        {
            Print("PASS: Fallback path working correctly");
        }
        else
        {
            Print("FAIL: Fallback path mismatch (expected 2.0, got ", retrievedMultiplier, ")");
        }
    }
    else
    {
        Print("FAIL: Could not create test file (error: ", GetLastError(), ")");
    }

    // Test with missing symbol
    Print("Testing with non-existent symbol...");
    double defaultMultiplier = GetRiskMultiplier("MISSING_SYMBOL");
    Print("Retrieved multiplier for MISSING_SYMBOL: ", defaultMultiplier);

    if(MathAbs(defaultMultiplier - QM_DEFAULT_MULTIPLIER) < 0.0001)
    {
        Print("PASS: Default multiplier returned for missing symbol");
    }
    else
    {
        Print("FAIL: Default multiplier mismatch (expected ", QM_DEFAULT_MULTIPLIER, ", got ", defaultMultiplier, ")");
    }
}

//+------------------------------------------------------------------+
//| Test Heartbeat Function                                          |
//+------------------------------------------------------------------+
void TestHeartbeatFunction()
{
    Print("--- Test REST Heartbeat ---");

    string testEAName = "TestRisk_EA";
    double testMultiplier = GetRiskMultiplier(TestSymbol);

    Print("Sending heartbeat for EA: ", testEAName);
    Print("Symbol: ", TestSymbol);
    Print("Magic Number: ", TestMagicNumber);
    Print("Risk Multiplier: ", testMultiplier);

    bool result = SendHeartbeat(testEAName, TestSymbol, TestMagicNumber, testMultiplier);

    if(result)
    {
        Print("PASS: Heartbeat function executed successfully");
        Print("Note: Verify Python backend received the heartbeat");
    }
    else
    {
        Print("WARNING: Heartbeat function returned false");
        Print("Note: This is expected if Python backend is not running");
        Print("Ensure localhost:8000/heartbeat endpoint is available");
    }
}

//+------------------------------------------------------------------+
//| Test JSON parsing functions directly                             |
//+------------------------------------------------------------------+
void TestJsonParsing()
{
    Print("--- Test JSON Parsing Functions ---");

    string testJson = "{";
    testJson += "\"EURUSD\": {";
    testJson += "\"multiplier\": 1.75, ";
    testJson += "\"timestamp\": 1704067200";
    testJson += "}, ";
    testJson += "\"GBPUSD\": {";
    testJson += "\"multiplier\": -0.5, ";
    testJson += "\"timestamp\": 1704067300";
    testJson += "}";
    testJson += "}";

    Print("Test JSON: ", testJson);

    // Test FindJsonObject
    string eurJson = FindJsonObject(testJson, "EURUSD");
    Print("Extracted EURUSD object: ", eurJson);

    if(StringLen(eurJson) > 0)
    {
        Print("PASS: FindJsonObject found EURUSD");
    }
    else
    {
        Print("FAIL: FindJsonObject did not find EURUSD");
    }

    // Test ExtractJsonDouble
    double multiplier = ExtractJsonDouble(eurJson, "multiplier");
    Print("Extracted multiplier: ", multiplier);

    if(MathAbs(multiplier - 1.75) < 0.0001)
    {
        Print("PASS: ExtractJsonDouble retrieved correct value");
    }
    else
    {
        Print("FAIL: ExtractJsonDouble retrieved incorrect value");
    }

    // Test negative number parsing
    string gbpJson = FindJsonObject(testJson, "GBPUSD");
    double gbpMultiplier = ExtractJsonDouble(gbpJson, "multiplier");
    Print("Extracted GBPUSD multiplier (negative): ", gbpMultiplier);

    if(MathAbs(gbpMultiplier - (-0.5)) < 0.0001)
    {
        Print("PASS: Negative number parsing works");
    }
    else
    {
        Print("FAIL: Negative number parsing failed");
    }
}

//+------------------------------------------------------------------+
//| Test timestamp validation for stale data                         |
//+------------------------------------------------------------------+
void TestStaleDataValidation()
{
    Print("--- Test Stale Data Validation ---");

    // Create JSON with old timestamp (more than 1 hour ago)
    int oldTimestamp = (int)TimeCurrent() - QM_MAX_DATA_AGE_SECONDS - 100;
    string testJson = "{";
    testJson += "\"" + TestSymbol + "\": {";
    testJson += "\"multiplier\": 3.0, ";
    testJson += "\"timestamp\": " + IntegerToString(oldTimestamp);
    testJson += "}";
    testJson += "}";

    // Write stale data file
    int handle = FileOpen(QM_RISK_MATRIX_FILE, FILE_WRITE|FILE_TXT|FILE_ANSI, '\0');
    if(handle != INVALID_HANDLE)
    {
        FileWriteString(handle, testJson);
        FileClose(handle);
        Print("Created stale data file (timestamp: ", oldTimestamp, ")");

        // Read it back
        double retrievedMultiplier = GetRiskMultiplier(TestSymbol);
        Print("Retrieved multiplier: ", retrievedMultiplier);

        // Should return default (1.0) because data is stale
        if(MathAbs(retrievedMultiplier - QM_DEFAULT_MULTIPLIER) < 0.0001)
        {
            Print("PASS: Stale data rejected, default returned");
        }
        else
        {
            Print("FAIL: Stale data not rejected (got ", retrievedMultiplier, ")");
        }
    }
}

//+------------------------------------------------------------------+
