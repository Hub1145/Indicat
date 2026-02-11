from fastapi import Header, HTTPException, Depends
from sqlalchemy.future import select
from lib.database import User, get_db
import os

RAPIDAPI_SECRET = os.getenv("RAPIDAPI_PROXY_SECRET", "dev_secret")

async def get_current_user(
    x_api_key: str = Header(None),
    x_rapidapi_proxy_secret: str = Header(None),
    db = Depends(get_db)
):
    # RapidAPI Proxy verification
    if RAPIDAPI_SECRET != "dev_secret":
        if x_rapidapi_proxy_secret != RAPIDAPI_SECRET:
            raise HTTPException(status_code=403, detail="Unauthorized: Invalid Proxy Secret")

    # If using local API keys
    if x_api_key:
        result = await db.execute(select(User).where(User.api_key == x_api_key))
        user = result.scalars().first()
        if not user:
            raise HTTPException(status_code=401, detail="Invalid API Key")
        return user

    # Default for dev/anonymous (if allowed) or RapidAPI user identification
    return User(id="anonymous", tier="free", api_key="none")

def get_tier_limit(tier: str):
    limits = {
        "free": {"rpm": 10, "rpd": 1000},
        "pro": {"rpm": 100, "rpd": 50000},
        "enterprise": {"rpm": 1000, "rpd": 1000000}
    }
    return limits.get(tier, limits["free"])
