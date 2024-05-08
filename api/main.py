from fastapi import FastAPI
from .routers import users, transactions
from data.web_scraping import scrape_financial_news

app = FastAPI(title="Stock AI Bot API")

# Include routes for users and transactions
app.include_router(users.router, prefix="/users", tags=["Users"])
app.include_router(transactions.router, prefix="/transactions", tags=["Transactions"])

@app.get("/")
async def root():
    return {"message": "Welcome to the Stock AI Bot API! Now featuring enhanced data insights from financial news."}

@app.get("/")
async def root():
    return {"message": "Welcome to the Stock AI Bot API!"}
