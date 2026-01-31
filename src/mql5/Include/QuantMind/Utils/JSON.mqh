//+------------------------------------------------------------------+
//|                                                           JSON.mqh |
//|                        QuantMind Standard Library (QSL) Utils     |
//|                                                                  |
//| JSON parsing utilities for MQL5.                                 |
//| Provides manual JSON parsing functions for scenarios where       |
//| native MQL5 JSON support is limited or for compatibility.       |
//+------------------------------------------------------------------+
#property copyright "QuantMindX"
#property link      "https://github.com/quantmindx"
#property version   "7.00"
#property strict

//+------------------------------------------------------------------+
//| Include Guard                                                     |
//+------------------------------------------------------------------+
#ifndef __QSL_JSON_MQH__
#define __QSL_JSON_MQH__

//+------------------------------------------------------------------+
//| Function: FindJsonObject                                         |
//|------------------------------------------------------------------+
//| Find JSON object for a given key in JSON string.                |
//|                                                                  |
//| This function searches for a key in JSON content and returns    |
//| the complete JSON object (including nested braces) associated   |
//| with that key.                                                   |
//|                                                                  |
//| Algorithm:                                                       |
//| 1. Search for "KEY" pattern in content                          |
//| 2. Find opening brace after the key                             |
//| 3. Count brace depth to find matching closing brace             |
//| 4. Return substring from opening to closing brace               |
//|                                                                  |
//| @param jsonContent Full JSON content string                     |
//| @param key          Key to search for (e.g., "EURUSD")          |
//| @return             JSON object as string, or empty if not found|
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
        {
            depth++;
        }
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
//| Function: ExtractJsonDouble                                      |
//|------------------------------------------------------------------+
//| Extract double value from JSON object string.                   |
//|                                                                  |
//| This function parses a JSON object string to extract a numeric  |
//| value associated with a specific key. Supports:                 |
//| - Positive and negative numbers                                 |
//| - Integer and decimal values                                    |
//| - Whitespace handling after colon                               |
//|                                                                  |
//| Note: This is a manual parser that stops at non-numeric chars.  |
//| Scientific notation (e.g., 1.5e10) is NOT fully supported.      |
//|                                                                  |
//| @param jsonObject JSON object string (e.g., '{"multiplier":1.5}')|
//| @param key         Key to extract (e.g., "multiplier")          |
//| @return            Double value, or 0 if not found              |
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
        {
            break;
        }
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
//| End of Include Guard                                             |
//+------------------------------------------------------------------+
#endif // __QSL_JSON_MQH__
