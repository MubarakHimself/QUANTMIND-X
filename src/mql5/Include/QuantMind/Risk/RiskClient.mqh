//+------------------------------------------------------------------+
//|                                                       RiskClient.mqh |
//|                        QuantMind Standard Library (QSL) Risk        |
//|                                                                  |
//| Risk Client for retrieving risk multipliers with fast path and    |
//| fallback mechanisms. Provides Python-MQL5 synchronization for    |
//| dynamic risk management.                                          |
//+------------------------------------------------------------------+
#property copyright "QuantMindX"
#property link      "https://github.com/quantmindx"
#property version   "7.00"
#property strict

//+------------------------------------------------------------------+
//| Include Guard                                                     |
//+------------------------------------------------------------------+
#ifndef __QSL_RISK_CLIENT_MQH__
#define __QSL_RISK_CLIENT_MQH__

//+------------------------------------------------------------------+
//| Dependencies                                                      |
//+------------------------------------------------------------------+
#include <QuantMind/Utils/JSON.mqh>

//+------------------------------------------------------------------+
//| Constants                                                         |
//+------------------------------------------------------------------+
#define QM_RISK_MULTIPLIER_VAR    "QM_RISK_MULTIPLIER"
#define QM_RISK_MATRIX_FILE       "risk_matrix.json"
#define QM_DEFAULT_MULTIPLIER     1.0
#define QM_MAX_DATA_AGE_SECONDS   3600    // Max age for cached data (1 hour)

//+------------------------------------------------------------------+
//| Risk data structure                                               |
//+------------------------------------------------------------------+
struct QMRiskData
{
    double multiplier;
    int timestamp;

    QMRiskData() { multiplier = QM_DEFAULT_MULTIPLIER; timestamp = 0; }
};

//+------------------------------------------------------------------+
//| Function: GetRiskMultiplier                                      |
//|------------------------------------------------------------------+
//| Get risk multiplier for a symbol with fast path fallback.        |
//|                                                                  |
//| Priority chain:                                                  |
//| 1. Fast path: GlobalVariable (set by Python agent)               |
//| 2. Fallback path: risk_matrix.json file                          |
//| 3. Default: 1.0                                                  |
//|                                                                  |
//| The function validates data freshness when reading from file,    |
//| rejecting data older than 1 hour to ensure risk values are       |
//| current and accurate.                                            |
//|                                                                  |
//| @param symbol Trading symbol (e.g., "EURUSD")                    |
//| @return Risk multiplier value (default 1.0 if not found)         |
//+------------------------------------------------------------------+
double GetRiskMultiplier(string symbol)
{
    // Fast path: Check GlobalVariable set by Python agent
    double globalMultiplier = GlobalVariableCheck(QM_RISK_MULTIPLIER_VAR);
    if(globalMultiplier > 0)
    {
        return globalMultiplier;
    }

    // Fallback path: Read from risk_matrix.json file
    QMRiskData data;
    if(ReadRiskFromFile(symbol, data))
    {
        // Validate data freshness
        int currentTimestamp = (int)TimeCurrent();
        if((currentTimestamp - data.timestamp) < QM_MAX_DATA_AGE_SECONDS)
        {
            return data.multiplier;
        }
        else
        {
            Print("[QuantMind RiskClient] Risk data for ", symbol, " is stale (timestamp: ", data.timestamp, ")");
        }
    }

    // Default fallback
    Print("[QuantMind RiskClient] Using default multiplier for ", symbol);
    return QM_DEFAULT_MULTIPLIER;
}

//+------------------------------------------------------------------+
//| Function: ReadRiskFromFile                                       |
//|------------------------------------------------------------------+
//| Read risk data from risk_matrix.json file.                       |
//|                                                                  |
//| Parses JSON structure:                                           |
//| {                                                                |
//|   "EURUSD": { "multiplier": 1.5, "timestamp": 1234567890 },      |
//|   "GBPUSD": { "multiplier": 1.0, "timestamp": 1234567890 }       |
//| }                                                                |
//|                                                                  |
//| @param symbol Trading symbol                                      |
//| @param data Output risk data structure                           |
//| @return true if data found and valid, false otherwise            |
//+------------------------------------------------------------------+
bool ReadRiskFromFile(string symbol, QMRiskData &data)
{
    string filePath = QM_RISK_MATRIX_FILE;
    int handle = FileOpen(filePath, FILE_READ|FILE_TXT|FILE_ANSI, '\0');

    if(handle == INVALID_HANDLE)
    {
        Print("[QuantMind RiskClient] Could not open ", filePath, " (error: ", GetLastError(), ")");
        return false;
    }

    // Read entire file content
    string content = "";
    while(!FileIsEnding(handle))
    {
        content += FileReadString(handle);
    }
    FileClose(handle);

    // Parse JSON manually using JSON utility functions
    string symbolSection = FindJsonObject(content, symbol);
    if(symbolSection == "")
    {
        Print("[QuantMind RiskClient] Symbol ", symbol, " not found in risk matrix");
        return false;
    }

    // Extract multiplier and timestamp
    data.multiplier = ExtractJsonDouble(symbolSection, "multiplier");
    data.timestamp = (int)ExtractJsonDouble(symbolSection, "timestamp");

    // Normalize invalid multipliers to default
    if(data.multiplier <= 0)
    {
        data.multiplier = QM_DEFAULT_MULTIPLIER;
    }

    return true;
}

//+------------------------------------------------------------------+
//| Function: SendHeartbeat                                          |
//|------------------------------------------------------------------+
//| Send heartbeat to Python backend via WebRequest POST.            |
//|                                                                  |
//| Sends JSON payload with EA status information to the backend     |
//| for monitoring and risk management updates.                      |
//|                                                                  |
//| @param eaName EA identifier                                      |
//| @param symbol Trading symbol                                     |
//| @param magicNumber Magic number                                  |
//| @param riskMultiplier Current risk multiplier                    |
//| @return true if heartbeat sent successfully                      |
//+------------------------------------------------------------------+
bool SendHeartbeat(string eaName, string symbol, int magicNumber, double riskMultiplier)
{
    // Build heartbeat URL
    string url = "http://localhost:8000/heartbeat";
    
    // Build JSON payload
    string payload = "{";
    payload += "\"ea_name\":\"" + eaName + "\",";
    payload += "\"symbol\":\"" + symbol + "\",";
    payload += "\"magic_number\":" + IntegerToString(magicNumber) + ",";
    payload += "\"risk_multiplier\":" + DoubleToString(riskMultiplier, 4) + ",";
    payload += "\"account_balance\":" + DoubleToString(AccountInfoDouble(ACCOUNT_BALANCE), 2) + ",";
    payload += "\"account_equity\":" + DoubleToString(AccountInfoDouble(ACCOUNT_EQUITY), 2) + ",";
    payload += "\"timestamp\":" + IntegerToString((int)TimeCurrent()) + ",";
    payload += "\"status\":\"active\"";
    payload += "}";
    
    // Prepare request
    char post[];
    char result[];
    string headers = "Content-Type: application/json\r\n";
    
    // Convert payload to char array
    StringToCharArray(payload, post, 0, StringLen(payload));
    
    // Send WebRequest
    int timeout = 5000; // 5 second timeout
    int res = WebRequest("POST", url, headers, timeout, post, result, headers);
    
    if(res == -1)
    {
        int error = GetLastError();
        Print("[QuantMind RiskClient] Heartbeat failed: ", error);
        
        // Common error codes:
        // 4060 = URL not allowed (add to Tools -> Options -> Expert Advisors -> Allow WebRequest)
        // 4014 = Function not allowed
        if(error == 4060)
        {
            Print("[QuantMind RiskClient] Add 'http://localhost:8000' to allowed URLs in MT5 settings");
        }
        
        return false;
    }
    
    // Check response code
    if(res == 200)
    {
        Print("[QuantMind RiskClient] Heartbeat sent successfully for ", eaName, " on ", symbol);
        return true;
    }
    else
    {
        Print("[QuantMind RiskClient] Heartbeat returned status code: ", res);
        return false;
    }
}

//+------------------------------------------------------------------+
//| End of Include Guard                                             |
//+------------------------------------------------------------------+
#endif // __QSL_RISK_CLIENT_MQH__
