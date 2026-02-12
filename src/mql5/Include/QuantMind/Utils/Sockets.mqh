//+------------------------------------------------------------------+
//|                                                      Sockets.mqh |
//|                        QuantMind Standard Library (QSL) - Utils  |
//|                        WebSocket Communication Module            |
//|                                                                  |
//| Provides WebSocket and HTTP communication utilities for MQL5.   |
//| Wraps MQL5's WebRequest function with error handling and        |
//| convenience methods for common operations.                       |
//+------------------------------------------------------------------+
#property copyright "QuantMindX"
#property link      "https://github.com/quantmindx"
#property version   "1.00"
#property strict

#ifndef __QSL_SOCKETS_MQH__
#define __QSL_SOCKETS_MQH__

// Include dependencies
#include <QuantMind/Core/Constants.mqh>

//+------------------------------------------------------------------+
//| Socket Error Codes                                               |
//+------------------------------------------------------------------+
#define QM_SOCKET_OK                    0
#define QM_SOCKET_ERROR_URL_NOT_ALLOWED 4060
#define QM_SOCKET_ERROR_FUNCTION_DENIED 4014
#define QM_SOCKET_ERROR_TIMEOUT         -1
#define QM_SOCKET_ERROR_INVALID_PARAMS  -2

//+------------------------------------------------------------------+
//| CSocketClient Class                                              |
//|                                                                  |
//| Provides HTTP/WebSocket communication capabilities for MQL5.    |
//| Wraps WebRequest with error handling and convenience methods.   |
//+------------------------------------------------------------------+
class CSocketClient
{
private:
    string            m_baseUrl;
    int               m_timeout;
    int               m_lastError;
    string            m_lastErrorMessage;
    
public:
    //+------------------------------------------------------------------+
    //| Constructor                                                      |
    //+------------------------------------------------------------------+
    CSocketClient()
    {
        m_baseUrl = "http://localhost:8000";
        m_timeout = QM_BRIDGE_TIMEOUT_MS;
        m_lastError = QM_SOCKET_OK;
        m_lastErrorMessage = "";
    }
    
    //+------------------------------------------------------------------+
    //| Constructor with base URL                                       |
    //|                                                                  |
    //| @param baseUrl Base URL for requests (e.g., "http://localhost:8000") |
    //| @param timeout Timeout in milliseconds                          |
    //+------------------------------------------------------------------+
    CSocketClient(string baseUrl, int timeout = QM_BRIDGE_TIMEOUT_MS)
    {
        m_baseUrl = baseUrl;
        m_timeout = timeout;
        m_lastError = QM_SOCKET_OK;
        m_lastErrorMessage = "";
    }
    
    //+------------------------------------------------------------------+
    //| Destructor                                                       |
    //+------------------------------------------------------------------+
    ~CSocketClient()
    {
    }
    
    //+------------------------------------------------------------------+
    //| Send HTTP POST request                                          |
    //|                                                                  |
    //| @param endpoint API endpoint (e.g., "/heartbeat")               |
    //| @param jsonPayload JSON payload string                          |
    //| @param response Output response string                          |
    //| @return HTTP status code (200 = success, -1 = error)            |
    //+------------------------------------------------------------------+
    int Post(string endpoint, string jsonPayload, string &response)
    {
        // Build full URL
        string url = m_baseUrl + endpoint;
        
        // Prepare request
        char post[];
        char result[];
        string headers = "Content-Type: application/json\r\n";
        
        // Convert payload to char array
        StringToCharArray(jsonPayload, post, 0, StringLen(jsonPayload));
        
        // Send WebRequest
        int statusCode = WebRequest("POST", url, headers, m_timeout, post, result, headers);
        
        // Handle errors
        if(statusCode == -1)
        {
            m_lastError = GetLastError();
            HandleWebRequestError(m_lastError, url);
            response = "";
            return -1;
        }
        
        // Convert response to string
        response = CharArrayToString(result);
        
        m_lastError = QM_SOCKET_OK;
        m_lastErrorMessage = "";
        
        return statusCode;
    }
    
    //+------------------------------------------------------------------+
    //| Send HTTP GET request                                           |
    //|                                                                  |
    //| @param endpoint API endpoint (e.g., "/status")                  |
    //| @param response Output response string                          |
    //| @return HTTP status code (200 = success, -1 = error)            |
    //+------------------------------------------------------------------+
    int Get(string endpoint, string &response)
    {
        // Build full URL
        string url = m_baseUrl + endpoint;
        
        // Prepare request
        char post[];
        char result[];
        string headers = "";
        
        // Send WebRequest
        int statusCode = WebRequest("GET", url, headers, m_timeout, post, result, headers);
        
        // Handle errors
        if(statusCode == -1)
        {
            m_lastError = GetLastError();
            HandleWebRequestError(m_lastError, url);
            response = "";
            return -1;
        }
        
        // Convert response to string
        response = CharArrayToString(result);
        
        m_lastError = QM_SOCKET_OK;
        m_lastErrorMessage = "";
        
        return statusCode;
    }
    
    //+------------------------------------------------------------------+
    //| Send HTTP PUT request                                           |
    //|                                                                  |
    //| @param endpoint API endpoint                                     |
    //| @param jsonPayload JSON payload string                          |
    //| @param response Output response string                          |
    //| @return HTTP status code (200 = success, -1 = error)            |
    //+------------------------------------------------------------------+
    int Put(string endpoint, string jsonPayload, string &response)
    {
        // Build full URL
        string url = m_baseUrl + endpoint;
        
        // Prepare request
        char post[];
        char result[];
        string headers = "Content-Type: application/json\r\n";
        
        // Convert payload to char array
        StringToCharArray(jsonPayload, post, 0, StringLen(jsonPayload));
        
        // Send WebRequest
        int statusCode = WebRequest("PUT", url, headers, m_timeout, post, result, headers);
        
        // Handle errors
        if(statusCode == -1)
        {
            m_lastError = GetLastError();
            HandleWebRequestError(m_lastError, url);
            response = "";
            return -1;
        }
        
        // Convert response to string
        response = CharArrayToString(result);
        
        m_lastError = QM_SOCKET_OK;
        m_lastErrorMessage = "";
        
        return statusCode;
    }
    
    //+------------------------------------------------------------------+
    //| Send HTTP DELETE request                                        |
    //|                                                                  |
    //| @param endpoint API endpoint                                     |
    //| @param response Output response string                          |
    //| @return HTTP status code (200 = success, -1 = error)            |
    //+------------------------------------------------------------------+
    int Delete(string endpoint, string &response)
    {
        // Build full URL
        string url = m_baseUrl + endpoint;
        
        // Prepare request
        char post[];
        char result[];
        string headers = "";
        
        // Send WebRequest
        int statusCode = WebRequest("DELETE", url, headers, m_timeout, post, result, headers);
        
        // Handle errors
        if(statusCode == -1)
        {
            m_lastError = GetLastError();
            HandleWebRequestError(m_lastError, url);
            response = "";
            return -1;
        }
        
        // Convert response to string
        response = CharArrayToString(result);
        
        m_lastError = QM_SOCKET_OK;
        m_lastErrorMessage = "";
        
        return statusCode;
    }
    
    //+------------------------------------------------------------------+
    //| Check if connection is available                                |
    //|                                                                  |
    //| Sends a simple GET request to test connectivity                 |
    //|                                                                  |
    //| @return true if connection successful, false otherwise          |
    //+------------------------------------------------------------------+
    bool TestConnection()
    {
        string response;
        int statusCode = Get("/", response);
        
        return (statusCode >= 200 && statusCode < 300);
    }
    
    //+------------------------------------------------------------------+
    //| Get last error code                                             |
    //|                                                                  |
    //| @return Last error code                                         |
    //+------------------------------------------------------------------+
    int GetLastError()
    {
        return m_lastError;
    }
    
    //+------------------------------------------------------------------+
    //| Get last error message                                          |
    //|                                                                  |
    //| @return Last error message string                               |
    //+------------------------------------------------------------------+
    string GetLastErrorMessage()
    {
        return m_lastErrorMessage;
    }
    
    //+------------------------------------------------------------------+
    //| Set base URL                                                     |
    //|                                                                  |
    //| @param baseUrl New base URL                                     |
    //+------------------------------------------------------------------+
    void SetBaseUrl(string baseUrl)
    {
        m_baseUrl = baseUrl;
    }
    
    //+------------------------------------------------------------------+
    //| Get base URL                                                     |
    //|                                                                  |
    //| @return Current base URL                                        |
    //+------------------------------------------------------------------+
    string GetBaseUrl()
    {
        return m_baseUrl;
    }
    
    //+------------------------------------------------------------------+
    //| Set timeout                                                      |
    //|                                                                  |
    //| @param timeout Timeout in milliseconds                          |
    //+------------------------------------------------------------------+
    void SetTimeout(int timeout)
    {
        m_timeout = timeout;
    }
    
    //+------------------------------------------------------------------+
    //| Get timeout                                                      |
    //|                                                                  |
    //| @return Current timeout in milliseconds                         |
    //+------------------------------------------------------------------+
    int GetTimeout()
    {
        return m_timeout;
    }
    
private:
    //+------------------------------------------------------------------+
    //| Handle WebRequest errors                                        |
    //|                                                                  |
    //| @param errorCode MQL5 error code                                |
    //| @param url URL that failed                                      |
    //+------------------------------------------------------------------+
    void HandleWebRequestError(int errorCode, string url)
    {
        switch(errorCode)
        {
            case QM_SOCKET_ERROR_URL_NOT_ALLOWED:
                m_lastErrorMessage = "URL not allowed. Add '" + m_baseUrl + 
                                   "' to allowed URLs in Tools -> Options -> Expert Advisors";
                Print("[CSocketClient] ERROR: ", m_lastErrorMessage);
                break;
                
            case QM_SOCKET_ERROR_FUNCTION_DENIED:
                m_lastErrorMessage = "WebRequest function not allowed. Enable in Expert Advisor settings.";
                Print("[CSocketClient] ERROR: ", m_lastErrorMessage);
                break;
                
            default:
                m_lastErrorMessage = "WebRequest failed with error code: " + IntegerToString(errorCode);
                Print("[CSocketClient] ERROR: ", m_lastErrorMessage, " for URL: ", url);
                break;
        }
    }
};

//+------------------------------------------------------------------+
//| Helper Functions                                                 |
//+------------------------------------------------------------------+

//+------------------------------------------------------------------+
//| Build JSON payload from key-value pairs                          |
//|                                                                  |
//| @param keys Array of keys                                        |
//| @param values Array of values (as strings)                       |
//| @param count Number of key-value pairs                           |
//| @return JSON string                                              |
//+------------------------------------------------------------------+
string BuildJsonPayload(string &keys[], string &values[], int count)
{
    string json = "{";
    
    for(int i = 0; i < count; i++)
    {
        if(i > 0)
        {
            json += ",";
        }
        
        json += "\"" + keys[i] + "\":\"" + values[i] + "\"";
    }
    
    json += "}";
    
    return json;
}

//+------------------------------------------------------------------+
//| Parse JSON response for a specific key                           |
//|                                                                  |
//| Simple JSON parser for extracting string values                 |
//|                                                                  |
//| @param jsonResponse JSON response string                         |
//| @param key Key to extract                                        |
//| @return Value string, or empty if not found                      |
//+------------------------------------------------------------------+
string ParseJsonResponse(string jsonResponse, string key)
{
    string searchPattern = "\"" + key + "\"";
    int keyPos = StringFind(jsonResponse, searchPattern);
    
    if(keyPos < 0)
    {
        return "";
    }
    
    // Find colon after key
    int colonPos = StringFind(jsonResponse, ":", keyPos);
    if(colonPos < 0)
    {
        return "";
    }
    
    // Find opening quote
    int startQuote = StringFind(jsonResponse, "\"", colonPos);
    if(startQuote < 0)
    {
        return "";
    }
    
    // Find closing quote
    int endQuote = StringFind(jsonResponse, "\"", startQuote + 1);
    if(endQuote < 0)
    {
        return "";
    }
    
    return StringSubstr(jsonResponse, startQuote + 1, endQuote - startQuote - 1);
}

#endif // __QSL_SOCKETS_MQH__
//+------------------------------------------------------------------+


//+------------------------------------------------------------------+
//| V8 HFT Infrastructure: CQuantMindSocket Class                     |
//|                                                                   |
//| Enhanced socket client for sub-5ms latency communication with    |
//| Python ZMQ socket server. Uses optimized JSON formatting and     |
//| persistent connection patterns.                                   |
//+------------------------------------------------------------------+
class CQuantMindSocket : public CSocketClient
{
private:
    string m_serverAddress;
    uint   m_serverPort;
    bool   m_connected;
    int    m_reconnectAttempts;
    int    m_maxReconnectAttempts;
    
public:
    //+------------------------------------------------------------------+
    //| Constructor                                                      |
    //+------------------------------------------------------------------+
    CQuantMindSocket() : CSocketClient()
    {
        m_serverAddress = "localhost";
        m_serverPort = 5555;
        m_connected = false;
        m_reconnectAttempts = 0;
        m_maxReconnectAttempts = 3;
        
        // Set base URL for socket server
        SetBaseUrl("http://" + m_serverAddress + ":" + IntegerToString(m_serverPort));
    }
    
    //+------------------------------------------------------------------+
    //| Constructor with custom server address                          |
    //|                                                                  |
    //| @param serverAddress Server hostname or IP                      |
    //| @param serverPort Server port number                            |
    //+------------------------------------------------------------------+
    CQuantMindSocket(string serverAddress, uint serverPort) : CSocketClient()
    {
        m_serverAddress = serverAddress;
        m_serverPort = serverPort;
        m_connected = false;
        m_reconnectAttempts = 0;
        m_maxReconnectAttempts = 3;
        
        // Set base URL for socket server
        SetBaseUrl("http://" + m_serverAddress + ":" + IntegerToString(m_serverPort));
    }
    
    //+------------------------------------------------------------------+
    //| Connect to socket server                                        |
    //|                                                                  |
    //| Tests connection and marks as connected if successful.          |
    //|                                                                  |
    //| @return true if connected, false otherwise                      |
    //+------------------------------------------------------------------+
    bool ConnectToServer()
    {
        if(m_connected)
        {
            return true;
        }
        
        // Test connection
        if(TestConnection())
        {
            m_connected = true;
            m_reconnectAttempts = 0;
            Print("[CQuantMindSocket] ✓ Connected to socket server at ", GetBaseUrl());
            return true;
        }
        
        m_reconnectAttempts++;
        Print("[CQuantMindSocket] Failed to connect (attempt ", m_reconnectAttempts, "/", m_maxReconnectAttempts, ")");
        
        return false;
    }
    
    //+------------------------------------------------------------------+
    //| Send trade open event                                           |
    //|                                                                  |
    //| @param eaName EA identifier                                     |
    //| @param symbol Trading symbol                                    |
    //| @param volume Position size in lots                             |
    //| @param magic Magic number                                       |
    //| @param riskMultiplier Output risk multiplier from server        |
    //| @return true if successful, false otherwise                     |
    //+------------------------------------------------------------------+
    bool SendTradeOpen(string eaName, string symbol, double volume, int magic, double &riskMultiplier)
    {
        if(!m_connected && !ConnectToServer())
        {
            return false;
        }
        
        // Build JSON message
        string message = StringFormat(
            "{\"type\":\"trade_open\",\"ea_name\":\"%s\",\"symbol\":\"%s\",\"volume\":%.2f,\"magic\":%d,\"timestamp\":%d}",
            eaName, symbol, volume, magic, (int)TimeLocal()
        );
        
        // Send to server
        string response;
        int statusCode = Post("/", message, response);
        
        if(statusCode == 200)
        {
            // Parse risk multiplier from response
            riskMultiplier = ParseRiskMultiplier(response);
            return true;
        }
        
        m_connected = false;
        return false;
    }
    
    //+------------------------------------------------------------------+
    //| Send trade close event                                          |
    //|                                                                  |
    //| @param eaName EA identifier                                     |
    //| @param symbol Trading symbol                                    |
    //| @param ticket Order ticket number                               |
    //| @param profit Profit/loss amount                                |
    //| @return true if successful, false otherwise                     |
    //+------------------------------------------------------------------+
    bool SendTradeClose(string eaName, string symbol, int ticket, double profit)
    {
        if(!m_connected && !ConnectToServer())
        {
            return false;
        }
        
        // Build JSON message
        string message = StringFormat(
            "{\"type\":\"trade_close\",\"ea_name\":\"%s\",\"symbol\":\"%s\",\"ticket\":%d,\"profit\":%.2f,\"timestamp\":%d}",
            eaName, symbol, ticket, profit, (int)TimeLocal()
        );
        
        // Send to server
        string response;
        int statusCode = Post("/", message, response);
        
        if(statusCode == 200)
        {
            return true;
        }
        
        m_connected = false;
        return false;
    }
    
    //+------------------------------------------------------------------+
    //| Send trade modify event                                         |
    //|                                                                  |
    //| @param eaName EA identifier                                     |
    //| @param ticket Order ticket number                               |
    //| @param newSL New stop loss price                                |
    //| @param newTP New take profit price                              |
    //| @return true if successful, false otherwise                     |
    //+------------------------------------------------------------------+
    bool SendTradeModify(string eaName, int ticket, double newSL, double newTP)
    {
        if(!m_connected && !ConnectToServer())
        {
            return false;
        }
        
        // Build JSON message
        string message = StringFormat(
            "{\"type\":\"trade_modify\",\"ea_name\":\"%s\",\"ticket\":%d,\"new_sl\":%.5f,\"new_tp\":%.5f,\"timestamp\":%d}",
            eaName, ticket, newSL, newTP, (int)TimeLocal()
        );
        
        // Send to server
        string response;
        int statusCode = Post("/", message, response);
        
        if(statusCode == 200)
        {
            return true;
        }
        
        m_connected = false;
        return false;
    }
    
    //+------------------------------------------------------------------+
    //| Send heartbeat                                                   |
    //|                                                                  |
    //| @param eaName EA identifier                                     |
    //| @param symbol Trading symbol                                    |
    //| @param magic Magic number                                       |
    //| @param riskMultiplier Output risk multiplier from server        |
    //| @return true if successful, false otherwise                     |
    //+------------------------------------------------------------------+
    bool SendHeartbeat(string eaName, string symbol, int magic, double &riskMultiplier)
    {
        if(!m_connected && !ConnectToServer())
        {
            return false;
        }
        
        // Build JSON message
        string message = StringFormat(
            "{\"type\":\"heartbeat\",\"ea_name\":\"%s\",\"symbol\":\"%s\",\"magic\":%d,\"timestamp\":%d}",
            eaName, symbol, magic, (int)TimeLocal()
        );
        
        // Send to server
        string response;
        int statusCode = Post("/", message, response);
        
        if(statusCode == 200)
        {
            // Parse risk multiplier from response
            riskMultiplier = ParseRiskMultiplier(response);
            return true;
        }
        
        m_connected = false;
        return false;
    }
    
    //+------------------------------------------------------------------+
    //| Send risk update                                                 |
    //|                                                                  |
    //| @param eaName EA identifier                                     |
    //| @param newMultiplier New risk multiplier to set                 |
    //| @return true if successful, false otherwise                     |
    //+------------------------------------------------------------------+
    bool SendRiskUpdate(string eaName, double newMultiplier)
    {
        if(!m_connected && !ConnectToServer())
        {
            return false;
        }
        
        // Build JSON message
        string message = StringFormat(
            "{\"type\":\"risk_update\",\"ea_name\":\"%s\",\"risk_multiplier\":%.2f,\"timestamp\":%d}",
            eaName, newMultiplier, (int)TimeLocal()
        );
        
        // Send to server
        string response;
        int statusCode = Post("/", message, response);
        
        if(statusCode == 200)
        {
            return true;
        }
        
        m_connected = false;
        return false;
    }
    
    //+------------------------------------------------------------------+
    //| Check if connected                                               |
    //|                                                                  |
    //| @return true if connected, false otherwise                      |
    //+------------------------------------------------------------------+
    bool IsConnected()
    {
        return m_connected;
    }
    
    //+------------------------------------------------------------------+
    //| Disconnect from server                                           |
    //+------------------------------------------------------------------+
    void Disconnect()
    {
        m_connected = false;
        Print("[CQuantMindSocket] Disconnected from socket server");
    }
    
private:
    //+------------------------------------------------------------------+
    //| Parse risk multiplier from JSON response                        |
    //|                                                                  |
    //| @param jsonResponse JSON response string                        |
    //| @return Risk multiplier value                                   |
    //+------------------------------------------------------------------+
    double ParseRiskMultiplier(string jsonResponse)
    {
        // Find "risk_multiplier" key
        string searchPattern = "\"risk_multiplier\"";
        int keyPos = StringFind(jsonResponse, searchPattern);
        
        if(keyPos < 0)
        {
            return 1.0; // Default multiplier
        }
        
        // Find colon after key
        int colonPos = StringFind(jsonResponse, ":", keyPos);
        if(colonPos < 0)
        {
            return 1.0;
        }
        
        // Find comma or closing brace after value
        int commaPos = StringFind(jsonResponse, ",", colonPos);
        int bracePos = StringFind(jsonResponse, "}", colonPos);
        
        int endPos = (commaPos > 0 && commaPos < bracePos) ? commaPos : bracePos;
        
        if(endPos < 0)
        {
            return 1.0;
        }
        
        // Extract value string
        string valueStr = StringSubstr(jsonResponse, colonPos + 1, endPos - colonPos - 1);
        
        // Remove whitespace
        StringTrimLeft(valueStr);
        StringTrimRight(valueStr);
        
        // Convert to double
        return StringToDouble(valueStr);
    }
};

//+------------------------------------------------------------------+
//| Global: Create QuantMind Socket instance                         |
//+------------------------------------------------------------------+
CQuantMindSocket g_quantMindSocket;

//+------------------------------------------------------------------+
