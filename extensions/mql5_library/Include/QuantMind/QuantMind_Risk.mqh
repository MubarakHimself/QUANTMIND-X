//+------------------------------------------------------------------+
//|                                                    QuantMind_Risk.mqh |
//|                        QuantMind Risk Management Library (MQL5)    |
//|                                                                  |
//| Provides risk multiplier retrieval with fast path (GlobalVariable) |
//| and fallback path (JSON file) for Python-MQL5 synchronization.    |
//| Includes REST heartbeat for EA lifecycle management.              |
//+------------------------------------------------------------------+
#property copyright "QuantMindX"
#property link      "https://github.com/quantmindx"
#property version   "1.00"
#property strict

//+------------------------------------------------------------------+
//| Configuration                                                     |
//+------------------------------------------------------------------+
input int HeartbeatIntervalSeconds = 60;    // Heartbeat interval in seconds
input string HeartbeatURL = "http://localhost:8000/heartbeat";  // Heartbeat endpoint

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
struct RiskData
{
    double multiplier;
    int timestamp;

    RiskData() { multiplier = QM_DEFAULT_MULTIPLIER; timestamp = 0; }
};

//+------------------------------------------------------------------+
//| Global state                                                      |
//+------------------------------------------------------------------+
datetime g_lastHeartbeatTime = 0;

//+------------------------------------------------------------------+
//| Get risk multiplier for a symbol with fast path fallback         |
//|                                                                   |
//| First attempts to read from GlobalVariable (fast path set by      |
//| Python agents). Falls back to reading risk_matrix.json from       |
//| MQL5/Files/ directory.                                            |
//|                                                                   |
//| @param symbol Trading symbol (e.g., "EURUSD")                     |
//| @return Risk multiplier value (default 1.0 if not found)          |
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
    RiskData data;
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
            Print("[QuantMind Risk] Risk data for ", symbol, " is stale (timestamp: ", data.timestamp, ")");
        }
    }

    // Default fallback
    Print("[QuantMind Risk] Using default multiplier for ", symbol);
    return QM_DEFAULT_MULTIPLIER;
}

//+------------------------------------------------------------------+
//| Read risk data from risk_matrix.json file                        |
//|                                                                   |
//| Parses JSON structure:                                            |
//| {                                                                 |
//|   "EURUSD": { "multiplier": 1.5, "timestamp": 1234567890 },       |
//|   "GBPUSD": { "multiplier": 1.0, "timestamp": 1234567890 }        |
//| }                                                                 |
//|                                                                   |
//| @param symbol Trading symbol                                      |
//| @param data Output risk data structure                            |
//| @return true if data found and valid, false otherwise            |
//+------------------------------------------------------------------+
bool ReadRiskFromFile(string symbol, RiskData &data)
{
    string filePath = QM_RISK_MATRIX_FILE;
    int handle = FileOpen(filePath, FILE_READ|FILE_TXT|FILE_ANSI, '\0');

    if(handle == INVALID_HANDLE)
    {
        Print("[QuantMind Risk] Could not open ", filePath, " (error: ", GetLastError(), ")");
        return false;
    }

    // Read entire file content
    string content = "";
    while(!FileIsEnding(handle))
    {
        content += FileReadString(handle);
    }
    FileClose(handle);

    // Parse JSON manually (MQL5 has limited JSON support)
    string symbolSection = FindJsonObject(content, symbol);
    if(symbolSection == "")
    {
        Print("[QuantMind Risk] Symbol ", symbol, " not found in risk matrix");
        return false;
    }

    // Extract multiplier
    data.multiplier = ExtractJsonDouble(symbolSection, "multiplier");
    data.timestamp = (int)ExtractJsonDouble(symbolSection, "timestamp");

    if(data.multiplier <= 0)
    {
        data.multiplier = QM_DEFAULT_MULTIPLIER;
    }

    return true;
}

//+------------------------------------------------------------------+
//| Find JSON object for a given key in JSON string                  |
//|                                                                   |
//| @param jsonContent Full JSON content                              |
//| @param key Key to search for                                      |
//| @return JSON object as string, or empty if not found             |
//+------------------------------------------------------------------+
string FindJsonObject(string jsonContent, string key)
{
    // Search for "KEY": {
    string searchPattern = "\"" + key + "\"";
    int keyPos = StringFind(jsonContent, searchPattern);

    if(keyPos < 0)
    {
        return "";
    }

    // Find opening brace after key
    int start = StringFind(jsonContent, "{", keyPos);
    if(start < 0)
    {
        return "";
    }

    // Find matching closing brace (handle nested objects)
    int depth = 0;
    int end = -1;

    for(int i = start; i < StringLen(jsonContent); i++)
    {
        string char = StringSubstr(jsonContent, i, 1);
        if(char == "{")
            depth++;
        else if(char == "}")
        {
            depth--;
            if(depth == 0)
            {
                end = i;
                break;
            }
        }
    }

    if(end < 0)
    {
        return "";
    }

    return StringSubstr(jsonContent, start, end - start + 1);
}

//+------------------------------------------------------------------+
//| Extract double value from JSON object string                     |
//|                                                                   |
//| @param jsonObject JSON object string                              |
//| @param key Key to extract                                         |
//| @return Double value, or 0 if not found                          |
//+------------------------------------------------------------------+
double ExtractJsonDouble(string jsonObject, string key)
{
    string searchPattern = "\"" + key + "\"";
    int keyPos = StringFind(jsonObject, searchPattern);

    if(keyPos < 0)
    {
        return 0;
    }

    // Find colon after key
    int colonPos = StringFind(jsonObject, ":", keyPos);
    if(colonPos < 0)
    {
        return 0;
    }

    // Extract value (handle number, including negative and decimals)
    string valueStr = "";
    int i = colonPos + 1;

    // Skip whitespace
    while(i < StringLen(jsonObject))
    {
        string char = StringSubstr(jsonObject, i, 1);
        if(char != " " && char != "\t" && char != "\n")
            break;
        i++;
    }

    // Extract number characters
    bool hasDecimal = false;
    while(i < StringLen(jsonObject))
    {
        string char = StringSubstr(jsonObject, i, 1);

        // Handle negative numbers
        if(char == "-" && StringLen(valueStr) == 0)
        {
            valueStr += char;
        }
        // Handle decimal point
        else if(char == "." && !hasDecimal)
        {
            valueStr += char;
            hasDecimal = true;
        }
        // Handle digits
        else if(char >= "0" && char <= "9")
        {
            valueStr += char;
        }
        else
        {
            // End of number
            break;
        }
        i++;
    }

    if(StringLen(valueStr) > 0)
    {
        return StringToDouble(valueStr);
    }

    return 0;
}

//+------------------------------------------------------------------+
//| Send heartbeat to Python backend                                 |
//|                                                                   |
//| POSTs EA lifecycle information to localhost:8000/heartbeat        |
//| for monitoring and health checking.                              |
//|                                                                   |
//| @param eaName Name of the Expert Advisor                         |
//| @param symbol Trading symbol                                      |
//| @param magicNumber EA magic number for identification             |
//| @param riskMultiplier Current risk multiplier                     |
//| @return true if heartbeat sent successfully, false otherwise     |
//+------------------------------------------------------------------+
bool SendHeartbeat(string eaName, string symbol, int magicNumber, double riskMultiplier)
{
    datetime currentTime = TimeCurrent();

    // Check if heartbeat is due based on interval
    if((currentTime - g_lastHeartbeatTime) < HeartbeatIntervalSeconds)
    {
        return true;  // Not time yet, consider as success
    }

    // Prepare heartbeat data
    int timestamp = (int)currentTime;
    string jsonPayload = StringFormat("{\"ea_name\": \"%s\", \"symbol\": \"%s\", \"magic_number\": %d, \"risk_multiplier\": %.2f, \"timestamp\": %d}",
                                       eaName, symbol, magicNumber, riskMultiplier, timestamp);

    // Convert to char array for WebRequest
    char postData[];
    char result[];
    string resultHeaders;

    StringToCharArray(jsonPayload, postData, 0, StringLen(jsonPayload));
    ArrayResize(postData, StringLen(jsonPayload));  // Resize to exact length

    // Prepare headers
    string headers = "Content-Type: application/json\r\n";

    // Send POST request
    int timeout = 5000;  // 5 seconds
    int res = WebRequest("POST", HeartbeatURL, headers, timeout, postData, result, resultHeaders);

    if(res > 0)
    {
        g_lastHeartbeatTime = currentTime;
        Print("[QuantMind Risk] Heartbeat sent successfully (EA: ", eaName, ", Symbol: ", symbol, ")");
        return true;
    }
    else
    {
        Print("[QuantMind Risk] Heartbeat failed (error: ", GetLastError(), ", HTTP code: ", res, ")");
        return false;
    }
}

//+------------------------------------------------------------------+
//| Initialize Risk Management System                                |
//|                                                                   |
//| Call from OnInit() to set up risk management.                    |
//|                                                                   |
//| @return INIT_SUCCEEDED                                           |
//+------------------------------------------------------------------+
int RiskInit()
{
    Print("[QuantMind Risk] Risk Management Library initialized");
    Print("[QuantMind Risk] Heartbeat interval: ", HeartbeatIntervalSeconds, " seconds");
    Print("[QuantMind Risk] Heartbeat URL: ", HeartbeatURL);

    return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Deinitialize Risk Management System                              |
//|                                                                   |
//| Call from OnDeinit() for cleanup.                                |
//+------------------------------------------------------------------+
void RiskDeinit(const const int reason)
{
    Print("[QuantMind Risk] Risk Management Library deinitialized (reason: ", reason, ")");
}

//+------------------------------------------------------------------+
