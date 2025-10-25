import asyncio
from asyncio import Queue

class Publisher:
    """In-memory pub/sub using asyncio.Queue. Replaceable with Redis later.
    Topics: 'stock', 'alerts'
    """
    def __init__(self):
        self.q: Queue = Queue()

    async def publish(self, topic: str, payload: dict):
        await self.q.put((topic, payload))

    async def run(self, broadcaster):
        while True:
            topic, payload = await self.q.get()
            await broadcaster(topic, payload)

publisher = Publisher()
