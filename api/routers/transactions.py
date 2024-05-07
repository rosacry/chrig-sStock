from fastapi import APIRouter

router = APIRouter()

@router.post("/deposit", tags=["Transactions"])
async def deposit(amount: float):
    # Simulate deposit operation
    return {"status": "success", "amount": amount}

@router.post("/withdraw", tags=["Transactions"])
async def withdraw(amount: float):
    # Simulate withdrawal operation
    return {"status": "success", "amount": amount}