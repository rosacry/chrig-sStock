from fastapi import FastAPI
from .routers import users, transactions

app = FastAPI(title="Stock AI Bot API")

app.include_router(users.router, prefix="/users", tags=["Users"])
app.include_router(transactions.router, prefix="/transactions", tags=["Transactions"])

@app.get("/")
async def root():
    return {"message": "Welcome to the Stock AI Bot API!"}