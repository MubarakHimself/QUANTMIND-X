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
