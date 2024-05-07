from pydantic import BaseModel

class TransactionBase(BaseModel):
    amount: float
    type: str  # 'deposit' or 'withdraw'

class TransactionCreate(TransactionBase):
    pass

class TransactionResponse(TransactionBase):
    status: str  # 'success' or 'failure'