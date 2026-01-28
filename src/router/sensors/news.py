"""
News Sensor (Time Guardian)
Tracks high-impact news events and enforces Kill Zones.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List

@dataclass
class NewsEvent:
    title: str
    impact: str # HIGH, MEDIUM, LOW
    time: datetime

class NewsSensor:
    def __init__(self):
        self.events: List[NewsEvent] = []
        self.kill_zone_minutes_pre = 15
        self.kill_zone_minutes_post = 15

    def update_calendar(self, calendar_data: List[dict]):
        """Load events from Crawler/API."""
        # Parsing logic here
        pass

    def check_state(self) -> str:
        """
        Returns: SAFE, PRE_NEWS, KILL_ZONE, POST_NEWS
        """
        now = datetime.utcnow()
        
        for event in self.events:
            if event.impact != "HIGH":
                continue
                
            time_diff = (event.time - now).total_seconds() / 60
            
            # Inside Kill Zone (-15 to +15)
            if -self.kill_zone_minutes_post <= time_diff <= self.kill_zone_minutes_pre:
                return "KILL_ZONE"
                
        return "SAFE"
