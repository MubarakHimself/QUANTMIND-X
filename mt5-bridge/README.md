# QuantMind MT5 Bridge (Windows VPS)

This is the middleware required to connect your Linux QuantMind IDE to your Windows MetaTrader 5 terminal.

## Setup Instructions (On your Windows VPS)

1.  **Install Python**: Ensure Python 3.10+ is installed on your Windows VPS.
2.  **Install Dependencies**:
    ```powershell
    pip install -r requirements.txt
    ```
3.  **Configure Token**:
    *   Open `server.py`.
    *   Change `API_TOKEN` to a secure password (match this with your IDE Settings).
    *   *Optional*: Set `MT5_BRIDGE_TOKEN` environment variable.
4.  **Run the Server**:
    ```powershell
    python server.py
    ```
    *   It will start on port `5005`.
    *   Ensure your VPS Firewall allows traffic on Port 5005 (TCP).
5.  **Connect from IDE**:
    *   Go to QuantMind IDE -> Settings -> Remote MT5 Bridge.
    *   Enter your VPS IP and Port 5005.

## Troubleshooting
*   **"MT5 Initialize failed"**: Make sure the MT5 terminal is installed. You may need to open the terminal manually once to set up the account.
*   **"Connection Refused"**: Check your VPS Firewall / AWS Security Groups to allow Inbound Custom TCP 5005.
