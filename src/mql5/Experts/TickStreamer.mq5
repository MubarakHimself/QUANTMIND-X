//+------------------------------------------------------------------+
//|                                                  TickStreamer.mq5 |
//|                                                                  |
//|                    QuantMindX Real-Time Tick Streaming EA        |
//|                                                                  |
//|  This EA publishes real-time tick data via ZMQ PUB socket       |
//|  for ultra-low latency streaming to Python tick handler.         |
//|                                                                  |
//|  Features:                                                       |
//|  - ZMQ PUB socket publishing on port 5555                       |
//|  - JSON format tick data with sequence numbers                   |
//|  - Heartbeat every 5 seconds for connection monitoring          |
//|  - Automatic reconnection on ZMQ errors                         |
//|  - Configurable symbols via input parameters                     |
//+------------------------------------------------------------------+

#property copyright "QuantMindX"
#property version   "1.00"
#property description "Real-time tick data publisher via ZMQ"
#property strict

#include <Zmq/Zmq.mqh>

//--- Input Parameters ---
input string   InpSymbols = "EURUSD,GBPUSD,USDJPY,XAUUSD";  // Symbols to stream (comma-separated)
input int      InpZmqPort = 5555;                            // ZMQ PUB socket port
input int      InpHeartbeatSec = 5;                         // Heartbeat interval in seconds
input bool     InpEnableDebug = false;                      // Enable debug output

//--- Global Variables ---
CContext       m_context;
CZmqSocket*    m_socket;
datetime       m_last_heartbeat;
int            m_sequence_counter = 0;
bool           m_connected = false;
string         m_symbols[];
ulong          m_last_tick_time[];  // Track last tick time per symbol

//+------------------------------------------------------------------+
//| Expert initialization function                                     |
//+------------------------------------------------------------------+
int OnInit()
{
    // Parse symbols from input
    ParseSymbols();
    
    // Initialize array for last tick times
    ArrayResize(m_last_tick_time, ArraySize(m_symbols));
    for(int i = 0; i < ArraySize(m_symbols); i++)
    {
        m_last_tick_time[i] = 0;
    }
    
    // Initialize ZMQ context and socket
    m_context.Init();
    m_socket = new CZmqSocket(m_context, ZMQ_PUB);
    
    // Set socket options
    m_socket.SetSockOpt(ZMQ_LINGER, 0);  // Don't wait on close
    
    // Try to connect
    if(!ConnectSocket())
    {
        Print("WARNING: Could not connect to ZMQ socket. Will retry...");
    }
    
    m_last_heartbeat = TimeCurrent();
    Print("TickStreamer initialized with ", ArraySize(m_symbols), " symbols on port ", InpZmqPort);
    
    return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                  |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    // Send disconnect message before closing
    if(m_connected && m_socket != NULL)
    {
        string disconnect_msg = "{\"type\":\"disconnect\",\"timestamp\":" + IntegerToString(TimeCurrent()) + "}";
        m_socket.Send(disconnect_msg);
    }
    
    // Clean up
    if(m_socket != NULL)
    {
        m_socket.Shutdown();
        delete m_socket;
    }
    m_context.Shutdown();
    
    Print("TickStreamer deinitialized");
}

//+------------------------------------------------------------------+
//| Parse symbols from input string                                   |
//+------------------------------------------------------------------+
void ParseSymbols()
{
    string tmp = InpSymbols;
    ushort separator = StringGetCharacter(",", 0);
    
    int count = 0;
    string result[];
    
    // Split by comma
    StringSplit(tmp, separator, result);
    ArrayResize(m_symbols, ArraySize(result));
    
    for(int i = 0; i < ArraySize(result); i++)
    {
        // Trim whitespace
        string symbol = StringTrimLeft(StringTrimRight(result[i]));
        
        // Check if symbol is valid and selected in Market Watch
        if(symbol != "" && SymbolSelect(symbol, true))
        {
            m_symbols[count] = symbol;
            count++;
        }
    }
    
    ArrayResize(m_symbols, count);
}

//+------------------------------------------------------------------+
//| Connect to ZMQ socket                                             |
//+------------------------------------------------------------------+
bool ConnectSocket()
{
    string endpoint = "tcp://*:" + IntegerToString(InpZmqPort);
    
    // Try to bind
    if(!m_socket.Bind(endpoint))
    {
        if(InpEnableDebug)
            Print("Failed to bind to ZMQ socket: ", endpoint);
        m_connected = false;
        return false;
    }
    
    m_connected = true;
    if(InpEnableDebug)
        Print("ZMQ socket bound to: ", endpoint);
    
    return true;
}

//+------------------------------------------------------------------+
//| Expert tick function                                              |
//+------------------------------------------------------------------+
void OnTick()
{
    datetime current_time = TimeCurrent();
    
    // Check heartbeat
    if(current_time - m_last_heartbeat >= InpHeartbeatSec)
    {
        SendHeartbeat();
        m_last_heartbeat = current_time;
    }
    
    // Process each symbol
    for(int i = 0; i < ArraySize(m_symbols); i++)
    {
        string symbol = m_symbols[i];
        
        // Get latest tick
        MqlTick tick;
        if(!SymbolInfoTick(symbol, tick))
        {
            if(InpEnableDebug)
                Print("Failed to get tick for ", symbol);
            continue;
        }
        
        // Skip if no new tick (same time as last)
        if(tick.time == m_last_tick_time[i])
            continue;
        
        m_last_tick_time[i] = tick.time;
        
        // Send tick data
        SendTickData(symbol, tick);
    }
}

//+------------------------------------------------------------------+
//| Send tick data in JSON format                                     |
//+------------------------------------------------------------------+
void SendTickData(const string symbol, const MqlTick& tick)
{
    if(!m_connected || m_socket == NULL)
        return;
    
    m_sequence_counter++;
    
    // Build JSON message
    // Format: {symbol, bid, ask, last, volume, time, time_msc, flags, sequence}
    string json = "{";
    json += "\"type\":\"tick\",";
    json += "\"symbol\":\"" + symbol + "\",";
    json += "\"bid\":" + DoubleToString(tick.bid, _Digits) + ",";
    json += "\"ask\":" + DoubleToString(tick.ask, _Digits) + ",";
    json += "\"last\":" + DoubleToString(tick.last, _Digits) + ",";
    json += "\"volume\":" + IntegerToString(tick.volume) + ",";
    json += "\"time\":" + IntegerToString(tick.time) + ",";
    json += "\"time_msc\":" + IntegerToString(tick.time_msc) + ",";
    json += "\"flags\":" + IntegerToString(tick.flags) + ",";
    json += "\"sequence\":" + IntegerToString(m_sequence_counter);
    json += "}";
    
    // Send via ZMQ
    if(!m_socket.Send(json))
    {
        if(InpEnableDebug)
            Print("Failed to send tick for ", symbol);
    }
}

//+------------------------------------------------------------------+
//| Send heartbeat message                                            |
//+------------------------------------------------------------------+
void SendHeartbeat()
{
    if(!m_connected || m_socket == NULL)
    {
        // Try to reconnect
        ConnectSocket();
        return;
    }
    
    string json = "{";
    json += "\"type\":\"heartbeat\",";
    json += "\"timestamp\":" + IntegerToString(TimeCurrent()) + ",";
    json += "\"sequence\":" + IntegerToString(m_sequence_counter) + ",";
    json += "\"symbols\":[";
    
    for(int i = 0; i < ArraySize(m_symbols); i++)
    {
        if(i > 0) json += ",";
        json += "\"" + m_symbols[i] + "\"";
    }
    
    json += "]}";
    
    if(!m_socket.Send(json))
    {
        if(InpEnableDebug)
            Print("Failed to send heartbeat");
    }
}

//+------------------------------------------------------------------+
//| Handle trade transaction events                                   |
//+------------------------------------------------------------------+
void OnTradeTransaction(const MqlTradeTransaction& trans,
                        const MqlTradeRequest& request,
                        const MqlTradeResult& result)
{
    // Not used by this EA, but required for compilation
}
