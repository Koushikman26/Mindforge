# Real-time WebSocket Integration

This folder adds a standalone FastAPI app providing WebSocket support without modifying existing files.

## Run
```bash
pip install fastapi uvicorn
uvicorn app.ws_app:app --reload
# open http://127.0.0.1:8000/app/client_demo.html (serve via any static server)
```

## Endpoints
- `WS /ws?topics=stock,alerts` — subscribe to topics
- `POST /test/stock?sku=SKU-001&on_hand=42` — broadcast a stock update
- `POST /test/alert?sku=SKU-001&severity=warning&message=Low+stock` — broadcast an alert

## Notes
- ConnectionManager manages clients, topics, and heartbeat pings.
- Publisher is an asyncio queue; swap with Redis later without API changes.
- Client demo auto-reconnects with exponential backoff and shows connection status.
