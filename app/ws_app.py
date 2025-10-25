from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
import asyncio
from .realtime import manager
from .publisher import publisher

app = FastAPI(title="Realtime WS", version="0.1.0")

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket, topics: str = Query("stock,alerts")):
    topic_set = {t.strip() for t in topics.split(",") if t.strip()}
    cid = await manager.connect(ws, topic_set)
    try:
        while True:
            await ws.receive_text()  # read to detect disconnect
    except WebSocketDisconnect:
        manager.disconnect(cid)

@app.on_event("startup")
async def _startup():
    asyncio.create_task(manager.heartbeat())
    asyncio.create_task(publisher.run(manager.broadcast))

# test helpers
@app.post("/test/stock")
async def test_stock(sku: str, on_hand: int):
    await publisher.publish("stock", {"type": "stock_update", "sku": sku, "on_hand": on_hand})
    return {"ok": True}

@app.post("/test/alert")
async def test_alert(sku: str, severity: str, message: str):
    await publisher.publish("alerts", {"type": "alert_created", "sku": sku, "severity": severity, "message": message})
    return {"ok": True}
