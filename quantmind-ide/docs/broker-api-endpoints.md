# Broker Connection API Endpoints

## Overview

This document describes the backend API endpoints required for broker connection functionality in the QuantMindX IDE.

## Base URL

```
http://localhost:8000/api/trading
```

## Endpoints

### 1. Test Broker Connection

Test broker credentials without saving them.

**Endpoint:** `POST /broker/test`

**Request Body:**
```typescript
{
  broker: 'metaquotes' | 'ctrader' | 'binance' | 'bybit',
  credentials: {
    // MetaQuotes
    account?: string,
    password?: string,
    server?: string,

    // cTrader
    token?: string,

    // Binance/Bybit
    apiKey?: string,
    apiSecret?: string,

    // Common
    type: 'demo' | 'live'
  }
}
```

**Response:**
```typescript
{
  success: boolean,
  message: string,
  account?: {
    id: string,
    balance: number,
    currency: string,
    platform: string
  }
}
```

**Status Codes:**
- `200 OK` - Connection successful
- `400 Bad Request` - Invalid credentials format
- `401 Unauthorized` - Invalid credentials
- `503 Service Unavailable` - Broker API unavailable

---

### 2. Connect Broker

Connect and save broker credentials for persistent use.

**Endpoint:** `POST /broker/connect`

**Request Body:**
```typescript
{
  broker: 'metaquotes' | 'ctrader' | 'binance' | 'bybit',
  credentials: {
    // Same as test endpoint
    type: 'demo' | 'live',
    // ... broker-specific fields
  }
}
```

**Response:**
```typescript
{
  success: boolean,
  connectionId: string,
  broker: string,
  account: {
    id: string,
    balance: number,
    currency: string,
    platform: string
  },
  connectedAt: string // ISO timestamp
}
```

**Status Codes:**
- `200 OK` - Connection established
- `400 Bad Request` - Invalid credentials format
- `401 Unauthorized` - Invalid credentials
- `409 Conflict` - Connection already exists
- `503 Service Unavailable` - Broker API unavailable

---

### 3. List Connected Brokers

Get list of all connected broker accounts.

**Endpoint:** `GET /brokers`

**Response:**
```typescript
{
  brokers: Array<{
    connectionId: string,
    broker: 'metaquotes' | 'ctrader' | 'binance' | 'bybit',
    account: {
      id: string,
      type: 'demo' | 'live',
      balance: number,
      currency: string,
      platform: string
    },
    status: 'connected' | 'disconnected' | 'error',
    connectedAt: string,
    lastActivity: string
  }>
}
```

---

### 4. Disconnect Broker

Remove a broker connection.

**Endpoint:** `DELETE /broker/{connectionId}`

**URL Parameters:**
- `connectionId` (string) - The ID of the connection to remove

**Response:**
```typescript
{
  success: boolean,
  message: string
}
```

**Status Codes:**
- `200 OK` - Connection removed
- `404 Not Found` - Connection ID not found

---

### 5. Get Broker Account Info

Get detailed account information for a specific connection.

**Endpoint:** `GET /broker/{connectionId}/account`

**URL Parameters:**
- `connectionId` (string) - The ID of the connection

**Response:**
```typescript
{
  connectionId: string,
  broker: string,
  account: {
    id: string,
    type: 'demo' | 'live',
    balance: number,
    equity: number,
    margin: number,
    freeMargin: number,
    currency: string,
    leverage: number,
    platform: string
  },
  positions: Array<{
    id: string,
    symbol: string,
    type: 'buy' | 'sell',
    volume: number,
    price: number,
    profit: number
  }>,
  lastUpdate: string
}
```

---

## WebSocket Events

### Connection Status Updates

**Event:** `broker:status`

**Payload:**
```typescript
{
  connectionId: string,
  status: 'connected' | 'disconnected' | 'error',
  timestamp: string
}
```

### Account Data Updates

**Event:** `broker:account`

**Payload:**
```typescript
{
  connectionId: string,
  account: {
    balance: number,
    equity: number,
    margin: number,
    freeMargin: number
  }
}
```

---

## Error Handling

All endpoints follow consistent error response format:

```typescript
{
  error: {
    code: string,
    message: string,
    details?: any
  }
}
```

### Common Error Codes

- `INVALID_CREDENTIALS` - Provided credentials are invalid
- `CONNECTION_FAILED` - Could not establish connection to broker
- `BROKER_UNAVAILABLE` - Broker API is temporarily unavailable
- `ALREADY_CONNECTED` - Broker account already connected
- `CONNECTION_NOT_FOUND` - Requested connection does not exist
- `UNSUPPORTED_BROKER` - Broker type is not supported

---

## Rate Limiting

- Test connection: 10 requests per minute per IP
- Connect broker: 5 requests per minute per IP
- List brokers: 60 requests per minute per IP
- Account info: 30 requests per minute per connection

---

## Security Notes

1. All credentials should be encrypted at rest
2. Credentials should never be logged
3. API keys/secrets should be stored using secure encryption
4. Connections should use TLS/SSL
5. Consider implementing IP whitelisting for production
