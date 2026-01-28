"""
QuantMindX Native Bridge (Interface Layer)
Low-latency Socket Integration between Python and MT5.
"""

import sys
import json
import logging
from typing import Dict, Optional

try:
    import zmq
except ImportError:
    zmq = None

logger = logging.getLogger(__name__)

class NativeBridge:
    """
    The Socket Hub for the Router.
    - REQ/REP for Commands (Execute Order).
    - PUB/SUB for Data Stream (Tick Ingestion).
    """
    def __init__(self, command_port: int = 5555, data_port: int = 5556):
        if zmq is None:
            logger.warning("ZeroMQ (pyzmq) not installed. Bridge will operate in SIMULATION mode.")
            self.simulation_mode = True
        else:
            self.simulation_mode = False
            self.ctx = zmq.Context()
            
            # Command Socket (Dispatcher)
            self.cmd_socket = self.ctx.socket(zmq.PUSH)
            self.cmd_socket.bind(f"tcp://*:{command_port}")
            
            # Data Socket (Sentinel Ingestion)
            self.data_socket = self.ctx.socket(zmq.SUB)
            self.data_socket.bind(f"tcp://*:{data_port}")
            self.data_socket.setsockopt_string(zmq.SUBSCRIBE, "")

        logger.info(f"Native Bridge initialized on ports {command_port}/{data_port} (Sim: {self.simulation_mode})")

    def send_command(self, payload: Dict):
        """
        Dispatches a JSON command to MT5.
        """
        if self.simulation_mode:
            logger.info(f"[SIM] Dispatching Command: {payload}")
            return True
            
        try:
            self.cmd_socket.send_json(payload, zmq.NOBLOCK)
            return True
        except Exception as e:
            logger.error(f"Failed to dispatch command via ZMQ: {e}")
            return False

    def get_latest_tick(self) -> Optional[Dict]:
        """
        Non-blocking poll for the latest price data.
        """
        if self.simulation_mode:
            return None
            
        try:
            # Poll for new messages
            if self.data_socket.poll(timeout=1):
                return self.data_socket.recv_json()
            return None
        except Exception as e:
            logger.error(f"Failed to receive tick via ZMQ: {e}")
            return None

    def close(self):
        """Clean up sockets."""
        if not self.simulation_mode:
            self.cmd_socket.close()
            self.data_socket.close()
            self.ctx.term()
