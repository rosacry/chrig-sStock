from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from uuid import uuid4
from .models import Transaction, User
from .dependencies import get_db

router = APIRouter()

@router.post("/deposit", tags=["Transactions"])
async def deposit(amount: float, idempotency_key: str = Depends(uuid4), db: Session = Depends(get_db)):
    if amount <= 0:
        raise HTTPException(status_code=400, detail="Deposit amount must be positive")
    transaction = db.query(Transaction).filter(Transaction.idempotency_key == idempotency_key).first()
    if transaction:
        return {"status": "success", "amount": transaction.amount, "note": "Transaction already processed"}
    new_transaction = Transaction(amount=amount, transaction_type='deposit', user_id=1, idempotency_key=idempotency_key)  # Assuming user_id for simplicity
    db.add(new_transaction)
    db.commit()
    db.refresh(new_transaction)
    return {"status": "success", "amount": amount}

@router.post("/withdraw", tags=["Transactions"])
async def withdraw(amount: float, idempotency_key: str = Depends(uuid4), db: Session = Depends(get_db)):
    if amount <= 0:
        raise HTTPException(status_code=400, detail="Withdrawal amount must be positive")
    user = db.query(User).filter(User.id == 1).first()  # Assuming user_id for simplicity
    if user.balance < amount:
        raise HTTPException(status_code=400, detail="Insufficient funds")
    transaction = db.query(Transaction).filter(Transaction.idempotency_key == idempotency_key).first()
    if transaction:
        return {"status": "success", "amount": transaction.amount, "note": "Transaction already processed"}
    new_transaction = Transaction(amount=-amount, transaction_type='withdraw', user_id=1, idempotency_key=idempotency_key)
    db.add(new_transaction)
    user.balance -= amount
    db.commit()
    return {"status": "success", "amount": amount}

