from fastapi import FastAPI

app = FastAPI(title="Hello World App")

@app.get("/")
def root():
    return {"message": "Hello World"}

@app.get("/health")
def health():
    return {"status": "ok"}
