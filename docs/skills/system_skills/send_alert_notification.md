---
name: send_alert_notification
category: system_skills
description: Send alert notifications via email, Telegram, or other messaging channels
version: 1.0.0
dependencies: []
tags:
  - alerts
  - notifications
  - communication
---

# Send Alert Notification Skill

## Description

Sends alert notifications through various communication channels including email, Telegram, and webhooks. Supports customizable message templates with trade information, price alerts, and system notifications.

## Input Schema

```json
{
  "type": "object",
  "properties": {
    "channel": {
      "type": "string",
      "enum": ["email", "telegram", "webhook", "console", "log"],
      "description": "Notification channel"
    },
    "subject": {
      "type": "string",
      "description": "Alert subject/title"
    },
    "message": {
      "type": "string",
      "description": "Alert message body"
    },
    "priority": {
      "type": "string",
      "enum": ["low", "normal", "high", "urgent"],
      "description": "Alert priority level (default: normal)"
    },
    "recipient": {
      "type": "string",
      "description": "Recipient (email address, Telegram chat ID, or webhook URL)"
    },
    "data": {
      "type": "object",
      "description": "Additional data to include in notification"
    },
    "template": {
      "type": "string",
      "enum": [
        "trade_entry",
        "trade_exit",
        "price_alert",
        "error",
        "system",
        "custom"
      ],
      "description": "Message template to use"
    }
  },
  "required": ["channel", "subject", "message"]
}
```

## Output Schema

```json
{
  "type": "object",
  "properties": {
    "success": {
      "type": "boolean",
      "description": "Whether notification was sent successfully"
    },
    "channel": {
      "type": "string",
      "description": "Channel used for notification"
    },
    "timestamp": {
      "type": "string",
      "description": "ISO timestamp of notification"
    },
    "message_id": {
      "type": "string",
      "description": "Unique message ID"
    },
    "error": {
      "type": "string",
      "description": "Error message if unsuccessful"
    }
  }
}
```

## Code

```python
import os
import json
import uuid
from typing import Dict, Any, Optional
from datetime import datetime
import requests
from enum import Enum


class AlertPriority(Enum):
    """Alert priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


# Message templates
MESSAGE_TEMPLATES = {
    "trade_entry": """
🔔 TRADE ENTRY ALERT

Strategy: {strategy}
Symbol: {symbol}
Direction: {direction}
Entry Price: {entry_price}
Lot Size: {lots}
Stop Loss: {stop_loss}
Take Profit: {take_profit}

Risk: ${risk_amount}
Risk-Reward: {risk_reward}

Timestamp: {timestamp}
""",

    "trade_exit": """
✅ TRADE EXIT ALERT

Strategy: {strategy}
Symbol: {symbol}
Direction: {direction}
Entry Price: {entry_price}
Exit Price: {exit_price}

Profit: ${profit}
Pips: {pips}

Reason: {reason}

Timestamp: {timestamp}
""",

    "price_alert": """
📊 PRICE ALERT

Symbol: {symbol}
Alert Price: {alert_price}
Current Price: {current_price}
Condition: {condition}

Timestamp: {timestamp}
""",

    "error": """
⚠️ ERROR ALERT

System: {system}
Error: {error}
Severity: {severity}

Details:
{details}

Timestamp: {timestamp}
""",

    "system": """
🔧 SYSTEM ALERT

Message: {message}

Details:
{details}

Timestamp: {timestamp}
"""
}


def send_alert_notification(
    channel: str,
    subject: str,
    message: str,
    priority: str = "normal",
    recipient: Optional[str] = None,
    data: Optional[Dict[str, Any]] = None,
    template: Optional[str] = None
) -> Dict[str, Any]:
    """
    Send alert notification through specified channel.

    Args:
        channel: Notification channel (email, telegram, webhook, console, log)
        subject: Alert subject/title
        message: Alert message body
        priority: Alert priority level
        recipient: Recipient (varies by channel)
        data: Additional data for template rendering
        template: Message template to use

    Returns:
        Dictionary containing send result and message_id
    """
    message_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()

    # Apply template if specified
    if template and template in MESSAGE_TEMPLATES:
        if data is None:
            data = {}
        data["timestamp"] = timestamp
        try:
            message = MESSAGE_TEMPLATES[template].format(**data)
        except KeyError as e:
            message = f"Template error: {e}\n\n{message}"

    # Add priority indicator to subject
    priority_emoji = {
        "low": "💙",
        "normal": "🔔",
        "high": "🔶",
        "urgent": "🚨"
    }

    if channel in ["telegram", "console", "log"]:
        subject = f"{priority_emoji.get(priority, '')} {subject}"

    try:
        if channel == "email":
            result = _send_email(subject, message, recipient, priority)
        elif channel == "telegram":
            result = _send_telegram(subject, message, recipient, priority)
        elif channel == "webhook":
            result = _send_webhook(subject, message, recipient, priority)
        elif channel == "console":
            result = _send_console(subject, message, priority)
        elif channel == "log":
            result = _send_log(subject, message, priority)
        else:
            return {
                "success": False,
                "channel": channel,
                "error": f"Unknown channel: {channel}"
            }

        result["message_id"] = message_id
        result["timestamp"] = timestamp
        result["channel"] = channel

        return result

    except Exception as e:
        return {
            "success": False,
            "channel": channel,
            "message_id": message_id,
            "timestamp": timestamp,
            "error": str(e)
        }


def _send_email(subject: str, message: str, recipient: str, priority: str) -> Dict[str, Any]:
    """Send email notification."""
    try:
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart

        # Get email configuration from environment
        smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        smtp_port = int(os.getenv("SMTP_PORT", "587"))
        smtp_user = os.getenv("SMTP_USER")
        smtp_password = os.getenv("SMTP_PASSWORD")

        if not all([smtp_user, smtp_password]):
            return {
                "success": False,
                "error": "Email credentials not configured (SMTP_USER, SMTP_PASSWORD)"
            }

        # Create message
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"[{priority.upper()}] {subject}"
        msg["From"] = smtp_user
        msg["To"] = recipient

        # Add priority header
        priority_map = {
            "low": "5",
            "normal": "3",
            "high": "2",
            "urgent": "1"
        }
        msg["X-Priority"] = priority_map.get(priority, "3")

        # Attach message body
        msg.attach(MIMEText(message, "plain"))

        # Send email
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.send_message(msg)

        return {"success": True}

    except ImportError:
        return {
            "success": False,
            "error": "Email libraries not available (smtplib required)"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Email send failed: {str(e)}"
        }


def _send_telegram(subject: str, message: str, chat_id: str, priority: str) -> Dict[str, Any]:
    """Send Telegram notification."""
    try:
        # Get Telegram bot token from environment
        bot_token = os.getenv("TELEGRAM_BOT_TOKEN")

        if not bot_token:
            return {
                "success": False,
                "error": "Telegram bot token not configured (TELEGRAM_BOT_TOKEN)"
            }

        if not chat_id:
            chat_id = os.getenv("TELEGRAM_CHAT_ID")
            if not chat_id:
                return {
                    "success": False,
                    "error": "Telegram chat ID not configured"
                }

        # Combine subject and message
        full_message = f"{subject}\n\n{message}"

        # Send message via Telegram API
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        data = {
            "chat_id": chat_id,
            "text": full_message,
            "parse_mode": "HTML"
        }

        response = requests.post(url, json=data, timeout=10)
        response.raise_for_status()

        return {"success": True}

    except Exception as e:
        return {
            "success": False,
            "error": f"Telegram send failed: {str(e)}"
        }


def _send_webhook(subject: str, message: str, url: str, priority: str) -> Dict[str, Any]:
    """Send webhook notification."""
    try:
        payload = {
            "subject": subject,
            "message": message,
            "priority": priority,
            "timestamp": datetime.now().isoformat()
        }

        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()

        return {"success": True}

    except Exception as e:
        return {
            "success": False,
            "error": f"Webhook send failed: {str(e)}"
        }


def _send_console(subject: str, message: str, priority: str) -> Dict[str, Any]:
    """Print notification to console."""
    print(f"\n{'='*60}")
    print(f"ALERT [{priority.upper()}]: {subject}")
    print(f"{'='*60}")
    print(message)
    print(f"{'='*60}\n")

    return {"success": True}


def _send_log(subject: str, message: str, priority: str) -> Dict[str, Any]:
    """Write notification to log file."""
    try:
        log_dir = "/data/logs/alerts"
        os.makedirs(log_dir, exist_ok=True)

        date_str = datetime.now().strftime("%Y-%m-%d")
        log_file = os.path.join(log_dir, f"alerts_{date_str}.log")

        with open(log_file, 'a') as f:
            f.write(f"[{datetime.now().isoformat()}] [{priority.upper()}] {subject}\n")
            f.write(f"{message}\n")
            f.write("="*60 + "\n")

        return {"success": True}

    except Exception as e:
        return {
            "success": False,
            "error": f"Log write failed: {str(e)}"
        }


# Convenience functions for common alerts

def alert_trade_entry(
    symbol: str,
    direction: str,
    entry_price: float,
    lots: float,
    stop_loss: float,
    take_profit: float,
    strategy: str,
    risk_amount: float,
    risk_reward: float,
    channel: str = "telegram"
) -> Dict[str, Any]:
    """Send trade entry alert."""
    return send_alert_notification(
        channel=channel,
        subject=f"Trade Entry: {symbol} {direction.upper()}",
        message="",  # Template will be used
        priority="normal",
        template="trade_entry",
        data={
            "strategy": strategy,
            "symbol": symbol,
            "direction": direction.upper(),
            "entry_price": entry_price,
            "lots": lots,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "risk_amount": risk_amount,
            "risk_reward": risk_reward
        }
    )


def alert_trade_exit(
    symbol: str,
    direction: str,
    entry_price: float,
    exit_price: float,
    profit: float,
    pips: float,
    reason: str,
    strategy: str,
    channel: str = "telegram"
) -> Dict[str, Any]:
    """Send trade exit alert."""
    priority = "high" if profit > 0 else "normal"

    return send_alert_notification(
        channel=channel,
        subject=f"Trade Exit: {symbol} ({'PROFIT' if profit > 0 else 'LOSS'})",
        message="",
        priority=priority,
        template="trade_exit",
        data={
            "strategy": strategy,
            "symbol": symbol,
            "direction": direction.upper(),
            "entry_price": entry_price,
            "exit_price": exit_price,
            "profit": profit,
            "pips": pips,
            "reason": reason
        }
    )


def alert_price_alert(
    symbol: str,
    alert_price: float,
    current_price: float,
    condition: str,
    channel: str = "telegram"
) -> Dict[str, Any]:
    """Send price alert."""
    return send_alert_notification(
        channel=channel,
        subject=f"Price Alert: {symbol}",
        message="",
        priority="normal",
        template="price_alert",
        data={
            "symbol": symbol,
            "alert_price": alert_price,
            "current_price": current_price,
            "condition": condition
        }
    )


def alert_error(system: str, error: str, severity: str = "normal", details: str = ""):
    """Send error alert."""
    return send_alert_notification(
        channel="console",
        subject=f"Error: {system}",
        message="",
        priority="high",
        template="error",
        data={
            "system": system,
            "error": error,
            "severity": severity,
            "details": details
        }
    )
```

## Example Usage

```python
# Example 1: Send trade entry alert
result = alert_trade_entry(
    symbol="EURUSD",
    direction="long",
    entry_price=1.0850,
    lots=0.1,
    stop_loss=1.0820,
    take_profit=1.0910,
    strategy="RSI_Mean_Reversion",
    risk_amount=100.0,
    risk_reward=2.0,
    channel="console"  # Use "telegram" for production
)

print(f"Alert sent: {result['success']}")

# Example 2: Send trade exit alert
result = alert_trade_exit(
    symbol="EURUSD",
    direction="long",
    entry_price=1.0850,
    exit_price=1.0910,
    profit=100.0,
    pips=60,
    reason="Take profit hit",
    strategy="RSI_Mean_Reversion",
    channel="console"
)

# Example 3: Send price alert
result = alert_price_alert(
    symbol="GBPUSD",
    alert_price=1.2700,
    current_price=1.2705,
    condition="Price above 1.2700 resistance",
    channel="console"
)

# Example 4: Send custom alert via webhook
result = send_alert_notification(
    channel="webhook",
    subject="Custom Alert",
    message="This is a custom notification",
    recipient="https://your-webhook-url.com/alerts",
    priority="high"
)

# Example 5: Send error alert
result = alert_error(
    system="MT5 Connection",
    error="Failed to connect to MT5 terminal",
    severity="high",
    details="Check if MT5 terminal is running"
)

# Example 6: Send alert with template
result = send_alert_notification(
    channel="console",
    subject="System Status",
    message="",
    template="system",
    data={
        "message": "Trading system operational",
        "details": "All indicators calculated, positions open"
    }
)

# Example 7: Configure environment variables for email/Telegram
"""
# Set these environment variables for email alerts:
export SMTP_SERVER="smtp.gmail.com"
export SMTP_PORT="587"
export SMTP_USER="your-email@gmail.com"
export SMTP_PASSWORD="your-app-password"

# Set these for Telegram alerts:
export TELEGRAM_BOT_TOKEN="your-bot-token"
export TELEGRAM_CHAT_ID="your-chat-id"

# Then use:
result = alert_trade_entry(..., channel="email")
# or
result = alert_trade_entry(..., channel="telegram")
"""
```

## Notes

- **Console channel**: Prints to stdout (useful for development/testing)
- **Log channel**: Writes to `/data/logs/alerts/` directory
- **Email channel**: Requires SMTP configuration (see environment variables)
- **Telegram channel**: Requires bot token and chat ID (see environment variables)
- **Webhook channel**: Sends POST request to specified URL with JSON payload

## Environment Variables

| Variable | Description | Required For |
|----------|-------------|--------------|
| `SMTP_SERVER` | SMTP server address | Email alerts |
| `SMTP_PORT` | SMTP port (usually 587) | Email alerts |
| `SMTP_USER` | Email username | Email alerts |
| `SMTP_PASSWORD` | Email password/app password | Email alerts |
| `TELEGRAM_BOT_TOKEN` | Telegram bot token from BotFather | Telegram alerts |
| `TELEGRAM_CHAT_ID` | Default chat ID for messages | Telegram alerts |

## Priority Levels

| Priority | Description | Use For |
|----------|-------------|---------|
| `low` | Informational | System status, daily summaries |
| `normal` | Standard alerts | Trade entries, price alerts |
| `high` | Important | Trade exits, errors |
| `urgent` | Critical | Margin calls, system failures |

## Message Templates

Templates are formatted with the provided `data` dictionary:
- `trade_entry`: Trade entry notifications
- `trade_exit`: Trade exit notifications
- `price_alert`: Price level alerts
- `error`: Error notifications
- `system`: System status messages

## Dependencies

- No skill dependencies (standalone notification system)
- `requests` library for Telegram/webhook (can be omitted if not used)
- `smtplib` for email (standard library)

## See Also

- `log_trade_event`: Log events to journal (can trigger alerts)
- `read_mt5_data`: Monitor positions for exit alerts
