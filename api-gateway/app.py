from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def health_check():
    return {"status": "api-gateway server is running", "status-code": 200}
