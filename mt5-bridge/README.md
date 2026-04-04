# QuantMind MT5 Bridge (Windows VPS)

This is the middleware required to connect your Linux QuantMind IDE to your Windows MetaTrader 5 terminal.

## Setup Instructions (On your Windows VPS)

1.  **Install Python**: Ensure Python 3.10+ is installed on your Windows VPS.
2.  **Install Dependencies**:
    ```powershell
    pip install -r requirements.txt
    ```
3.  **Configure Environment**:
    *   Set `MT5_BRIDGE_TOKEN` to a strong random value.
    *   Optional MT5 startup variables:
        * `MT5_TERMINAL_PATH`
        * `MT5_LOGIN`
        * `MT5_PASSWORD`
        * `MT5_SERVER`
        * `MT5_TIMEOUT_MS`
        * `MT5_PORTABLE`
        * `MT5_BRIDGE_PORT` (defaults to `5005`)
4.  **Run the Server**:
    ```powershell
    python server.py
    ```
    *   It will start on port `5005` unless `MT5_BRIDGE_PORT` is set.
    *   Ensure your VPS Firewall allows traffic on the configured bridge port (TCP).
5.  **Connect from IDE**:
    *   Go to QuantMind IDE -> Settings -> Remote MT5 Bridge.
    *   Enter your VPS IP and bridge port.
    *   Use `Authorization: Bearer <MT5_BRIDGE_TOKEN>` or `X-Token: <MT5_BRIDGE_TOKEN>`.

## Troubleshooting
*   **"MT5 Initialize failed"**: Make sure the MT5 terminal is installed. You may need to open the terminal manually once to set up the account.
*   **"Connection Refused"**: Check your VPS Firewall / AWS Security Groups to allow inbound TCP on the configured bridge port.
