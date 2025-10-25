from typing import Dict, Set
from fastapi import WebSocket
import asyncio, json, time

class Connection:
    def __init__(self, ws: WebSocket, topics: Set[str]):
        self.ws = ws
        self.topics = topics
        self.last_seen = time.time()

class ConnectionManager:
    def __init__(self, heartbeat_sec: int = 30):
        self._clients: Dict[int, Connection] = {}
        self._id_seq = 0
        self.heartbeat_sec = heartbeat_sec

    async def connect(self, websocket: WebSocket, topics: Set[str]) -> int:
        await websocket.accept()
        self._id_seq += 1
        cid = self._id_seq
        self._clients[cid] = Connection(websocket, topics)
        return cid

    def disconnect(self, cid: int):
        self._clients.pop(cid, None)

    async def broadcast(self, topic: str, payload: dict):
        msg = json.dumps({"v":1, **payload})
        dead = []
        for cid, conn in list(self._clients.items()):
            if topic in conn.topics:
                try:
                    await conn.ws.send_text(msg)
                except Exception:
                    dead.append(cid)
        for cid in dead:
            self.disconnect(cid)

    async def heartbeat(self):
        while True:
            await asyncio.sleep(self.heartbeat_sec)
            now = time.time()
            dead = []
            for cid, conn in list(self._clients.items()):
                try:
                    await conn.ws.send_text(json.dumps({"type": "heartbeat", "ts": now}))
                except Exception:
                    dead.append(cid)
            for cid in dead:
                self.disconnect(cid)

manager = ConnectionManager()
