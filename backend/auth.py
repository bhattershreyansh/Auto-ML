import os
import requests
from jose import jwt
from fastapi import HTTPException, Security, Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from dotenv import load_dotenv

# Load env from root if it's there
load_dotenv("../.env") 
load_dotenv()

CLERK_SECRET_KEY = os.getenv("CLERK_SECRET_KEY")
# For testing/dev, if you don't have CLERK_JWKS_URL, we'll try to derive it from the secret
CLERK_JWKS_URL = os.getenv("CLERK_JWKS_URL", "https://api.clerk.com/v1/jwks")

security = HTTPBearer()

def get_jwks():
    """Fetch the JWKS from Clerk using the Secret Key."""
    try:
        headers = {"Authorization": f"Bearer {CLERK_SECRET_KEY}"}
        response = requests.get(CLERK_JWKS_URL, headers=headers)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Auth Error: {e}")
        return None

async def get_current_user(auth: HTTPAuthorizationCredentials = Security(security)):
    """
    FastAPI dependency to verify the Clerk JWT and return the user_id.
    """
    token = auth.credentials
    
    if not CLERK_SECRET_KEY:
        # Fallback for development ONLY
        return "dev_user_123"

    try:
        jwks = get_jwks()
        if not jwks:
             raise HTTPException(status_code=500, detail="Could not retrieve Clerk public keys")
             
        # Decoding without issuer check for now to maximize compatibility
        # unless CLERK_ISSUER is explicitly provided
        payload = jwt.decode(
            token,
            jwks,
            algorithms=["RS256"],
            options={"verify_aud": False}
        )
        return payload.get("sub")
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid authentication: {str(e)}")
