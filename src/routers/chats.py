from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from src.auth import get_current_user
from database import supabase

router = APIRouter(tags=['chats'])
class ChatCreateRequest(BaseModel):
    project_id: str
    title: str

@router.post("/api/chats")
def create_chat(chat_request: ChatCreateRequest, clerk_id: str = Depends(get_current_user)):
    try:
        # Verify that the project belongs to the authenticated user
        project = supabase.table('projects').select('*').eq('id', chat_request.project_id).eq('clerk_id', clerk_id).execute()
        if not project.data or len(project.data) == 0:
            raise HTTPException(status_code=404, detail="Project not found or access denied")
        
        # Create the chat
        result = supabase.table('chats').insert({
            'project_id': chat_request.project_id,
            'title': chat_request.title,
            'clerk_id': clerk_id
        }).execute()
        
        return {
            "success": True,
            "message": "Chat created successfully",
            "data": result.data[0]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create chat: {str(e)}")
    
@router.delete("/api/chats/{chat_id}")
def delete_chat(chat_id: str, clerk_id: str = Depends(get_current_user)):
    try:
        # Verify that the chat belongs to a project of the authenticated user
        chat = supabase.table('chats').select('*, projects(clerk_id)').eq('id', chat_id).eq('clerk_id', clerk_id).execute()
        if not chat.data or len(chat.data) == 0 or chat.data[0]['projects']['clerk_id'] != clerk_id: # type: ignore
            raise HTTPException(status_code=404, detail="Chat not found or access denied")
        
        # Delete the chat
        result = supabase.table('chats').delete().eq('id', chat_id).execute()
        
        return {
            "success": True,
            "message": "Chat deleted successfully",
            "data": result.data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete chat: {str(e)}") 