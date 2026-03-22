from dataclasses import dataclass
from datetime import datetime
import time
import random
import uuid

@dataclass
class ViolationAlert:
    camera_id: str
    location_name: str
    vehicle_id: int
    duration_seconds: int
    confidence: float
    timestamp: str
    status: str = "pending"

def format_whatsapp_message(alert: ViolationAlert) -> str:
    m = alert.duration_seconds // 60
    s = alert.duration_seconds % 60
    return (f"🚨 SIMS Alert — Solapur Traffic Police\n"
            f"Violation: Illegal Parking\n"
            f"Location: {alert.location_name} ({alert.camera_id})\n"
            f"Vehicle ID: #{alert.vehicle_id}\n"
            f"Duration: {m} min {s} sec\n"
            f"Confidence: {alert.confidence:.0%}\n"
            f"Timestamp: {alert.timestamp}\n"
            f"Action: Tap to review evidence → [SIMS-EVIDENCE-LINK]")

def simulate_whatsapp_send(alert: ViolationAlert) -> dict:
    time.sleep(0.5)
    hex_id = uuid.uuid4().hex[:16]
    return {
        "status": "sent",
        "message_id": f"wamid.{hex_id}",
        "recipient": "duty_officer_roster",
        "timestamp": datetime.now().isoformat(),
        "template": "sims_violation_alert_v1"
    }

class WhatsAppEnforcementLog:
    def __init__(self):
        self._alerts = []

    def add_alert(self, alert: ViolationAlert):
        self._alerts.append(alert)
        simulate_whatsapp_send(alert)

    def get_pending(self) -> list:
        return [a for a in self._alerts if a.status == "pending"]

    def approve(self, vehicle_id: int):
        for a in self._alerts:
            if a.vehicle_id == vehicle_id and a.status == "pending":
                a.status = "approved"
                a.action_timestamp = datetime.now().isoformat()
                break

    def reject(self, vehicle_id: int):
        for a in self._alerts:
            if a.vehicle_id == vehicle_id and a.status == "pending":
                a.status = "rejected"
                a.action_timestamp = datetime.now().isoformat()
                break

    def get_all(self) -> list:
        return self._alerts
