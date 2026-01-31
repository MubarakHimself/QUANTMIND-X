//+------------------------------------------------------------------+
//|                                                   RingBuffer.mqh |
//|                        QuantMind Standard Library (QSL) - Utils  |
//|                        Ring Buffer (Circular Buffer) Module      |
//|                                                                  |
//| Provides O(1) ring buffer implementation for efficient          |
//| indicator calculations and time-series data management.         |
//| CRiBuff class offers constant-time operations for push/pop.     |
//+------------------------------------------------------------------+
#property copyright "QuantMindX"
#property link      "https://github.com/quantmindx"
#property version   "1.00"
#property strict

#ifndef __QSL_RING_BUFFER_MQH__
#define __QSL_RING_BUFFER_MQH__

// Include dependencies
#include <QuantMind/Core/Constants.mqh>

//+------------------------------------------------------------------+
//| CRiBuff Class (Ring Buffer)                                      |
//|                                                                  |
//| Circular buffer implementation with O(1) operations:            |
//| - Push: Add element to buffer (overwrites oldest if full)       |
//| - Get: Retrieve element by index                                |
//| - Size: Get current number of elements                          |
//| - IsFull: Check if buffer is full                               |
//|                                                                  |
//| Use cases:                                                       |
//| - Indicator calculations (moving averages, etc.)                |
//| - Price history tracking                                        |
//| - Time-series data management                                   |
//+------------------------------------------------------------------+
class CRiBuff
{
private:
    double            m_buffer[];       // Internal buffer array
    int               m_capacity;       // Maximum capacity
    int               m_size;           // Current size
    int               m_head;           // Head index (write position)
    int               m_tail;           // Tail index (read position)
    
public:
    //+------------------------------------------------------------------+
    //| Constructor                                                      |
    //|                                                                  |
    //| @param capacity Maximum buffer capacity                         |
    //+------------------------------------------------------------------+
    CRiBuff(int capacity = QM_RING_BUFFER_SIZE_MEDIUM)
    {
        m_capacity = (capacity > 0) ? capacity : QM_RING_BUFFER_SIZE_MEDIUM;
        m_size = 0;
        m_head = 0;
        m_tail = 0;
        
        ArrayResize(m_buffer, m_capacity);
        ArrayInitialize(m_buffer, 0.0);
    }
    
    //+------------------------------------------------------------------+
    //| Destructor                                                       |
    //+------------------------------------------------------------------+
    ~CRiBuff()
    {
        ArrayFree(m_buffer);
    }
    
    //+------------------------------------------------------------------+
    //| Push element to buffer (O(1) operation)                         |
    //|                                                                  |
    //| Adds element to the head of the buffer. If buffer is full,     |
    //| overwrites the oldest element.                                  |
    //|                                                                  |
    //| @param value Value to push                                      |
    //+------------------------------------------------------------------+
    void Push(double value)
    {
        // Write to head position
        m_buffer[m_head] = value;
        
        // Move head forward (circular)
        m_head = (m_head + 1) % m_capacity;
        
        // If buffer is full, move tail forward (overwrite oldest)
        if(m_size == m_capacity)
        {
            m_tail = (m_tail + 1) % m_capacity;
        }
        else
        {
            m_size++;
        }
    }
    
    //+------------------------------------------------------------------+
    //| Get element by index (O(1) operation)                           |
    //|                                                                  |
    //| Retrieves element at specified index. Index 0 is the most      |
    //| recent element, index 1 is the second most recent, etc.        |
    //|                                                                  |
    //| @param index Index (0 = most recent)                            |
    //| @return Value at index, or 0.0 if index out of bounds          |
    //+------------------------------------------------------------------+
    double Get(int index)
    {
        // Check bounds
        if(index < 0 || index >= m_size)
        {
            return 0.0;
        }
        
        // Calculate actual position in circular buffer
        // Most recent is at (head - 1), second most recent at (head - 2), etc.
        int actualIndex = (m_head - 1 - index + m_capacity) % m_capacity;
        
        return m_buffer[actualIndex];
    }
    
    //+------------------------------------------------------------------+
    //| Get oldest element (O(1) operation)                             |
    //|                                                                  |
    //| @return Oldest element in buffer, or 0.0 if empty               |
    //+------------------------------------------------------------------+
    double GetOldest()
    {
        if(m_size == 0)
        {
            return 0.0;
        }
        
        return m_buffer[m_tail];
    }
    
    //+------------------------------------------------------------------+
    //| Get newest element (O(1) operation)                             |
    //|                                                                  |
    //| @return Newest element in buffer, or 0.0 if empty               |
    //+------------------------------------------------------------------+
    double GetNewest()
    {
        if(m_size == 0)
        {
            return 0.0;
        }
        
        int newestIndex = (m_head - 1 + m_capacity) % m_capacity;
        return m_buffer[newestIndex];
    }
    
    //+------------------------------------------------------------------+
    //| Get current size (O(1) operation)                               |
    //|                                                                  |
    //| @return Number of elements currently in buffer                  |
    //+------------------------------------------------------------------+
    int Size()
    {
        return m_size;
    }
    
    //+------------------------------------------------------------------+
    //| Get capacity (O(1) operation)                                   |
    //|                                                                  |
    //| @return Maximum capacity of buffer                              |
    //+------------------------------------------------------------------+
    int Capacity()
    {
        return m_capacity;
    }
    
    //+------------------------------------------------------------------+
    //| Check if buffer is full (O(1) operation)                        |
    //|                                                                  |
    //| @return true if buffer is full, false otherwise                 |
    //+------------------------------------------------------------------+
    bool IsFull()
    {
        return (m_size == m_capacity);
    }
    
    //+------------------------------------------------------------------+
    //| Check if buffer is empty (O(1) operation)                       |
    //|                                                                  |
    //| @return true if buffer is empty, false otherwise                |
    //+------------------------------------------------------------------+
    bool IsEmpty()
    {
        return (m_size == 0);
    }
    
    //+------------------------------------------------------------------+
    //| Clear buffer (O(1) operation)                                   |
    //|                                                                  |
    //| Resets buffer to empty state without deallocating memory        |
    //+------------------------------------------------------------------+
    void Clear()
    {
        m_size = 0;
        m_head = 0;
        m_tail = 0;
        ArrayInitialize(m_buffer, 0.0);
    }
    
    //+------------------------------------------------------------------+
    //| Calculate sum of all elements (O(n) operation)                  |
    //|                                                                  |
    //| @return Sum of all elements in buffer                           |
    //+------------------------------------------------------------------+
    double Sum()
    {
        double sum = 0.0;
        
        for(int i = 0; i < m_size; i++)
        {
            sum += Get(i);
        }
        
        return sum;
    }
    
    //+------------------------------------------------------------------+
    //| Calculate average of all elements (O(n) operation)              |
    //|                                                                  |
    //| @return Average of all elements, or 0.0 if empty                |
    //+------------------------------------------------------------------+
    double Average()
    {
        if(m_size == 0)
        {
            return 0.0;
        }
        
        return Sum() / m_size;
    }
    
    //+------------------------------------------------------------------+
    //| Find maximum value (O(n) operation)                             |
    //|                                                                  |
    //| @return Maximum value in buffer, or 0.0 if empty                |
    //+------------------------------------------------------------------+
    double Max()
    {
        if(m_size == 0)
        {
            return 0.0;
        }
        
        double maxVal = Get(0);
        
        for(int i = 1; i < m_size; i++)
        {
            double val = Get(i);
            if(val > maxVal)
            {
                maxVal = val;
            }
        }
        
        return maxVal;
    }
    
    //+------------------------------------------------------------------+
    //| Find minimum value (O(n) operation)                             |
    //|                                                                  |
    //| @return Minimum value in buffer, or 0.0 if empty                |
    //+------------------------------------------------------------------+
    double Min()
    {
        if(m_size == 0)
        {
            return 0.0;
        }
        
        double minVal = Get(0);
        
        for(int i = 1; i < m_size; i++)
        {
            double val = Get(i);
            if(val < minVal)
            {
                minVal = val;
            }
        }
        
        return minVal;
    }
    
    //+------------------------------------------------------------------+
    //| Copy buffer contents to array (O(n) operation)                  |
    //|                                                                  |
    //| Copies buffer contents to output array in chronological order  |
    //| (oldest to newest).                                             |
    //|                                                                  |
    //| @param output Output array (will be resized)                    |
    //| @return Number of elements copied                               |
    //+------------------------------------------------------------------+
    int ToArray(double &output[])
    {
        ArrayResize(output, m_size);
        
        for(int i = 0; i < m_size; i++)
        {
            // Get from oldest to newest
            output[i] = Get(m_size - 1 - i);
        }
        
        return m_size;
    }
    
    //+------------------------------------------------------------------+
    //| Get buffer status string                                        |
    //|                                                                  |
    //| @return Formatted status string                                 |
    //+------------------------------------------------------------------+
    string GetStatus()
    {
        string status = "[CRiBuff] Status:\n";
        status += "  Capacity: " + IntegerToString(m_capacity) + "\n";
        status += "  Size: " + IntegerToString(m_size) + "\n";
        status += "  Full: " + (IsFull() ? "YES" : "NO") + "\n";
        status += "  Empty: " + (IsEmpty() ? "YES" : "NO") + "\n";
        
        if(m_size > 0)
        {
            status += "  Newest: " + DoubleToString(GetNewest(), 4) + "\n";
            status += "  Oldest: " + DoubleToString(GetOldest(), 4) + "\n";
            status += "  Average: " + DoubleToString(Average(), 4);
        }
        
        return status;
    }
};

//+------------------------------------------------------------------+
//| Helper function: Create ring buffer with specified capacity      |
//|                                                                  |
//| @param capacity Buffer capacity                                  |
//| @return Pointer to new CRiBuff instance                          |
//+------------------------------------------------------------------+
CRiBuff* CreateRingBuffer(int capacity)
{
    return new CRiBuff(capacity);
}

#endif // __QSL_RING_BUFFER_MQH__
//+------------------------------------------------------------------+
