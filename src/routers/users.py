from fastapi import FastAPI, HTTPException
from fastapi import APIRouter
from pydantic import BaseModel
from database import supabase

router  = APIRouter(
    tags=["users"]
)

@router.post("/create-user")
async def create_webhook_user(webhook_data: dict):
    try:
        event_type = webhook_data.get("event_type")
        if event_type != "user.created":
            user_data = webhook_data.get("data", {})
            clerk_id = user_data.get("id")
            
            if not clerk_id:
                raise HTTPException(status_code=400, detail="Invalid webhook data")
            
            # Check if user already exists
            existing_user = supabase.table("users").select("*").eq("clerk_id", clerk_id).execute()
            if existing_user.data:
                raise HTTPException(status_code=409, detail="User already exists")
            
            result = supabase.table("users").insert({"clerk_id": clerk_id}).execute()
            return {"message": "User created successfully,",
            "data": result.data[0]}
    except Exception as e:
            raise HTTPException(status_code=500, detail="Webhook processing error: " + str(e))
    
 