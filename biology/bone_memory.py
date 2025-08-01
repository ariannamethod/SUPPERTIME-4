import random


class BoneMemory:
    def __init__(self, limit=50):
        self.limit = int(limit)
        self.events = []
        self.metabolic_push = 0.0

    def on_event(self, ev_type: str):
        self.events.append(ev_type)
        if len(self.events) > self.limit:
            self.events.pop(0)
        density = self.events.count('click') / max(1, len(self.events))
        self.metabolic_push = density * 0.2 + random.uniform(0.0, 0.05)
        return self.metabolic_push

    def get(self):
        return len(self.events)
