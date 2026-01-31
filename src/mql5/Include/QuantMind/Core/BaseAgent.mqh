//+------------------------------------------------------------------+
//|                                                    BaseAgent.mqh |
//|                        QuantMind Standard Library (QSL) - Core   |
//|                        Base Agent Functionality Module           |
//|                                                                  |
//| Provides base functionality for all QuantMind Expert Advisors.  |
//| Includes initialization, error handling, logging, and common     |
//| utility methods that all EAs can inherit.                        |
//+------------------------------------------------------------------+
#property copyright "QuantMindX"
#property link      "https://github.com/quantmindx"
#property version   "1.00"
#property strict

#ifndef __QSL_BASE_AGENT_MQH__
#define __QSL_BASE_AGENT_MQH__

//+------------------------------------------------------------------+
//| Base Agent Class                                                 |
//|                                                                  |
//| Provides common functionality for all QuantMind Expert Advisors:|
//| - Initialization and validation                                  |
//| - Error handling and logging                                     |
//| - Symbol and timeframe management                                |
//| - Common utility methods                                         |
//+------------------------------------------------------------------+
class CBaseAgent
{
protected:
    string            m_agentName;          // Agent identifier
    string            m_symbol;             // Trading symbol
    ENUM_TIMEFRAMES   m_timeframe;          // Chart timeframe
    int               m_magicNumber;        // Magic number for trades
    bool              m_initialized;        // Initialization status
    datetime          m_lastLogTime;        // Last log timestamp
    int               m_logIntervalSeconds; // Minimum seconds between logs
    
public:
    //+------------------------------------------------------------------+
    //| Constructor                                                      |
    //+------------------------------------------------------------------+
    CBaseAgent()
    {
        m_agentName = "BaseAgent";
        m_symbol = Symbol();
        m_timeframe = Period();
        m_magicNumber = 0;
        m_initialized = false;
        m_lastLogTime = 0;
        m_logIntervalSeconds = 60; // Default: log at most once per minute
    }
    
    //+------------------------------------------------------------------+
    //| Destructor                                                       |
    //+------------------------------------------------------------------+
    ~CBaseAgent()
    {
        Deinitialize();
    }
    
    //+------------------------------------------------------------------+
    //| Initialize the agent                                            |
    //|                                                                  |
    //| @param agentName Name identifier for this agent                 |
    //| @param symbol Trading symbol (default: current chart symbol)    |
    //| @param timeframe Chart timeframe (default: current chart TF)    |
    //| @param magicNumber Magic number for trade identification        |
    //| @return true if initialization successful, false otherwise      |
    //+------------------------------------------------------------------+
    virtual bool Initialize(string agentName, 
                           string symbol = "", 
                           ENUM_TIMEFRAMES timeframe = PERIOD_CURRENT,
                           int magicNumber = 0)
    {
        m_agentName = agentName;
        m_symbol = (symbol == "") ? Symbol() : symbol;
        m_timeframe = (timeframe == PERIOD_CURRENT) ? Period() : timeframe;
        m_magicNumber = magicNumber;
        
        // Validate symbol
        if(!ValidateSymbol())
        {
            LogError("Symbol validation failed: " + m_symbol);
            return false;
        }
        
        // Validate timeframe
        if(!ValidateTimeframe())
        {
            LogError("Timeframe validation failed");
            return false;
        }
        
        m_initialized = true;
        LogInfo("Agent initialized successfully: " + m_agentName);
        
        return true;
    }
    
    //+------------------------------------------------------------------+
    //| Deinitialize the agent                                          |
    //+------------------------------------------------------------------+
    virtual void Deinitialize()
    {
        if(m_initialized)
        {
            LogInfo("Agent deinitialized: " + m_agentName);
            m_initialized = false;
        }
    }
    
    //+------------------------------------------------------------------+
    //| Check if agent is initialized                                   |
    //|                                                                  |
    //| @return true if initialized, false otherwise                    |
    //+------------------------------------------------------------------+
    bool IsInitialized() const
    {
        return m_initialized;
    }
    
    //+------------------------------------------------------------------+
    //| Validate trading symbol                                         |
    //|                                                                  |
    //| @return true if symbol is valid, false otherwise                |
    //+------------------------------------------------------------------+
    bool ValidateSymbol()
    {
        // Check if symbol exists
        if(SymbolSelect(m_symbol, true))
        {
            // Verify symbol is available for trading
            if(SymbolInfoInteger(m_symbol, SYMBOL_TRADE_MODE) == SYMBOL_TRADE_MODE_DISABLED)
            {
                LogError("Symbol trading is disabled: " + m_symbol);
                return false;
            }
            return true;
        }
        
        LogError("Symbol not found: " + m_symbol);
        return false;
    }
    
    //+------------------------------------------------------------------+
    //| Validate timeframe                                              |
    //|                                                                  |
    //| @return true if timeframe is valid, false otherwise             |
    //+------------------------------------------------------------------+
    bool ValidateTimeframe()
    {
        // Check if timeframe is valid
        switch(m_timeframe)
        {
            case PERIOD_M1:
            case PERIOD_M5:
            case PERIOD_M15:
            case PERIOD_M30:
            case PERIOD_H1:
            case PERIOD_H4:
            case PERIOD_D1:
            case PERIOD_W1:
            case PERIOD_MN1:
                return true;
            default:
                LogError("Invalid timeframe specified");
                return false;
        }
    }
    
    //+------------------------------------------------------------------+
    //| Get agent name                                                   |
    //|                                                                  |
    //| @return Agent name string                                        |
    //+------------------------------------------------------------------+
    string GetAgentName() const
    {
        return m_agentName;
    }
    
    //+------------------------------------------------------------------+
    //| Get trading symbol                                              |
    //|                                                                  |
    //| @return Symbol string                                            |
    //+------------------------------------------------------------------+
    string GetSymbol() const
    {
        return m_symbol;
    }
    
    //+------------------------------------------------------------------+
    //| Get timeframe                                                    |
    //|                                                                  |
    //| @return Timeframe enum value                                     |
    //+------------------------------------------------------------------+
    ENUM_TIMEFRAMES GetTimeframe() const
    {
        return m_timeframe;
    }
    
    //+------------------------------------------------------------------+
    //| Get magic number                                                 |
    //|                                                                  |
    //| @return Magic number integer                                     |
    //+------------------------------------------------------------------+
    int GetMagicNumber() const
    {
        return m_magicNumber;
    }
    
    //+------------------------------------------------------------------+
    //| Log information message                                          |
    //|                                                                  |
    //| @param message Message to log                                    |
    //+------------------------------------------------------------------+
    void LogInfo(string message)
    {
        if(ShouldLog())
        {
            Print("[", m_agentName, "] INFO: ", message);
            m_lastLogTime = TimeCurrent();
        }
    }
    
    //+------------------------------------------------------------------+
    //| Log warning message                                              |
    //|                                                                  |
    //| @param message Warning message to log                            |
    //+------------------------------------------------------------------+
    void LogWarning(string message)
    {
        Print("[", m_agentName, "] WARNING: ", message);
    }
    
    //+------------------------------------------------------------------+
    //| Log error message                                                |
    //|                                                                  |
    //| @param message Error message to log                              |
    //+------------------------------------------------------------------+
    void LogError(string message)
    {
        int errorCode = GetLastError();
        Print("[", m_agentName, "] ERROR: ", message, 
              " (Error Code: ", errorCode, ")");
    }
    
    //+------------------------------------------------------------------+
    //| Check if logging should occur (rate limiting)                   |
    //|                                                                  |
    //| @return true if enough time has passed since last log           |
    //+------------------------------------------------------------------+
    bool ShouldLog()
    {
        datetime currentTime = TimeCurrent();
        return (currentTime - m_lastLogTime) >= m_logIntervalSeconds;
    }
    
    //+------------------------------------------------------------------+
    //| Set log interval                                                 |
    //|                                                                  |
    //| @param seconds Minimum seconds between info logs                |
    //+------------------------------------------------------------------+
    void SetLogInterval(int seconds)
    {
        m_logIntervalSeconds = (seconds > 0) ? seconds : 60;
    }
    
    //+------------------------------------------------------------------+
    //| Get current account balance                                      |
    //|                                                                  |
    //| @return Account balance                                          |
    //+------------------------------------------------------------------+
    double GetAccountBalance()
    {
        return AccountInfoDouble(ACCOUNT_BALANCE);
    }
    
    //+------------------------------------------------------------------+
    //| Get current account equity                                       |
    //|                                                                  |
    //| @return Account equity                                           |
    //+------------------------------------------------------------------+
    double GetAccountEquity()
    {
        return AccountInfoDouble(ACCOUNT_EQUITY);
    }
    
    //+------------------------------------------------------------------+
    //| Get account free margin                                          |
    //|                                                                  |
    //| @return Free margin available                                    |
    //+------------------------------------------------------------------+
    double GetAccountFreeMargin()
    {
        return AccountInfoDouble(ACCOUNT_FREEMARGIN);
    }
    
    //+------------------------------------------------------------------+
    //| Check if trading is allowed                                      |
    //|                                                                  |
    //| @return true if trading is allowed, false otherwise             |
    //+------------------------------------------------------------------+
    bool IsTradingAllowed()
    {
        // Check if trading is allowed in terminal
        if(!TerminalInfoInteger(TERMINAL_TRADE_ALLOWED))
        {
            LogWarning("Trading is not allowed in terminal");
            return false;
        }
        
        // Check if expert trading is allowed
        if(!MQLInfoInteger(MQL_TRADE_ALLOWED))
        {
            LogWarning("Expert trading is not allowed");
            return false;
        }
        
        // Check if trading is allowed for the symbol
        if(!SymbolInfoInteger(m_symbol, SYMBOL_TRADE_MODE))
        {
            LogWarning("Trading is not allowed for symbol: " + m_symbol);
            return false;
        }
        
        return true;
    }
    
    //+------------------------------------------------------------------+
    //| Get symbol point value                                           |
    //|                                                                  |
    //| @return Point value for the symbol                               |
    //+------------------------------------------------------------------+
    double GetSymbolPoint()
    {
        return SymbolInfoDouble(m_symbol, SYMBOL_POINT);
    }
    
    //+------------------------------------------------------------------+
    //| Get symbol digits                                                |
    //|                                                                  |
    //| @return Number of digits after decimal point                    |
    //+------------------------------------------------------------------+
    int GetSymbolDigits()
    {
        return (int)SymbolInfoInteger(m_symbol, SYMBOL_DIGITS);
    }
    
    //+------------------------------------------------------------------+
    //| Get symbol tick size                                             |
    //|                                                                  |
    //| @return Minimum price change                                     |
    //+------------------------------------------------------------------+
    double GetSymbolTickSize()
    {
        return SymbolInfoDouble(m_symbol, SYMBOL_TRADE_TICK_SIZE);
    }
    
    //+------------------------------------------------------------------+
    //| Get symbol minimum lot                                           |
    //|                                                                  |
    //| @return Minimum lot size                                         |
    //+------------------------------------------------------------------+
    double GetSymbolMinLot()
    {
        return SymbolInfoDouble(m_symbol, SYMBOL_VOLUME_MIN);
    }
    
    //+------------------------------------------------------------------+
    //| Get symbol maximum lot                                           |
    //|                                                                  |
    //| @return Maximum lot size                                         |
    //+------------------------------------------------------------------+
    double GetSymbolMaxLot()
    {
        return SymbolInfoDouble(m_symbol, SYMBOL_VOLUME_MAX);
    }
    
    //+------------------------------------------------------------------+
    //| Get symbol lot step                                              |
    //|                                                                  |
    //| @return Lot size step                                            |
    //+------------------------------------------------------------------+
    double GetSymbolLotStep()
    {
        return SymbolInfoDouble(m_symbol, SYMBOL_VOLUME_STEP);
    }
    
    //+------------------------------------------------------------------+
    //| Normalize lot size according to symbol requirements             |
    //|                                                                  |
    //| @param lots Desired lot size                                     |
    //| @return Normalized lot size                                      |
    //+------------------------------------------------------------------+
    double NormalizeLot(double lots)
    {
        double minLot = GetSymbolMinLot();
        double maxLot = GetSymbolMaxLot();
        double lotStep = GetSymbolLotStep();
        
        // Ensure lot is within bounds
        if(lots < minLot) lots = minLot;
        if(lots > maxLot) lots = maxLot;
        
        // Round to lot step
        lots = MathRound(lots / lotStep) * lotStep;
        
        return lots;
    }
    
    //+------------------------------------------------------------------+
    //| Get current bid price                                            |
    //|                                                                  |
    //| @return Current bid price                                        |
    //+------------------------------------------------------------------+
    double GetBid()
    {
        return SymbolInfoDouble(m_symbol, SYMBOL_BID);
    }
    
    //+------------------------------------------------------------------+
    //| Get current ask price                                            |
    //|                                                                  |
    //| @return Current ask price                                        |
    //+------------------------------------------------------------------+
    double GetAsk()
    {
        return SymbolInfoDouble(m_symbol, SYMBOL_ASK);
    }
    
    //+------------------------------------------------------------------+
    //| Get current spread in points                                     |
    //|                                                                  |
    //| @return Spread in points                                         |
    //+------------------------------------------------------------------+
    int GetSpread()
    {
        return (int)SymbolInfoInteger(m_symbol, SYMBOL_SPREAD);
    }
};

#endif // __QSL_BASE_AGENT_MQH__
//+------------------------------------------------------------------+
