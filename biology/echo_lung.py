import time


class EchoLung:
    def __init__(self, capacity=5.0):
        self.capacity = float(capacity)
        self.breath = 0.0
        self.last_event_ts = time.time()

    def on_event(self, chaos: float):
        now = time.time()
        dt = now - self.last_event_ts
        self.last_event_ts = now
        load = max(0.0, 1.0 - dt / self.capacity)
        self.breath = max(0.0, min(self.breath * 0.8 + load * chaos, 1.0))
        if self.breath > 0.7:
            return "Echo breath glitch"
        return "Breath steady"

    def get(self):
        return self.breath
