from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from src.auth import get_current_user
from database import supabase
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
import os
import re
import json
from typing import List, Dict, Tuple, Union, Any
from langchain_core.messages import BaseMessage
from src.services.llm import llm, embeddings_model  # Changed to use Google Gemini models
from src.rag.retrieval.index import (
    retrieve_context,
    prepare_prompt_and_invoke_llm
)
load_dotenv()
router = APIRouter(tags=['chats'])
class ChatCreateRequest(BaseModel):
    project_id: str
    title: str
class SendMessageRequest(BaseModel):
    content: str


@router.post("/api/chats")
async def create_chat(chat_request: ChatCreateRequest, clerk_id: str = Depends(get_current_user)):
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
async def delete_chat(chat_id: str, clerk_id: str = Depends(get_current_user)):
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

@router.get("/api/chats/{chat_id}")
async def get_chat(
    chat_id: str,
    clerk_id: str = Depends(get_current_user)
):
    try:
        # Get the chat and verify it belongs to the user AND has a project_id
        result = supabase.table('chats').select('*').eq('id', chat_id).eq('clerk_id', clerk_id).execute()
        
        if not result.data:
            raise HTTPException(status_code=404, detail="Chat not found or access denied")
        
        chat = result.data[0] 
        
        # Get messages for this chat
        messages_result = supabase.table('messages').select('*').eq('chat_id', chat_id).order('created_at', desc=False).execute()
        
        # Add messages to chat object
        chat['messages'] = messages_result.data or [] # type: ignore
        
        return {
            "message": "Chat retrieved successfully",
            "data": chat
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get chat: {str(e)}")
    


@router.post("/api/projects/{project_id}/chats/{chat_id}/messages")
async def send_message(
    chat_id: str,
    project_id: str,
    request: SendMessageRequest,
    clerk_id: str = Depends(get_current_user)
):
    """
        User message → LLM → AI response
    """
    try:
        message = request.content
        
        print(f"💬 New message: {message[:50]}...")
        
        # 1. Save user message
        print(f"💾 Saving user message...")
        user_message_result = supabase.table('messages').insert({
            "chat_id": chat_id,
            "content": message,
            "role": "user",
            "clerk_id": clerk_id
        }).execute()
        
        user_message = user_message_result.data[0]
        print(f"✅ User message saved: {user_message['id']}") # type: ignore
        
        # 2. Retrieve context using RAG
        print(f"🔍 Retrieving context...")
        texts, images, tables, citations = retrieve_context(message, project_id)
        
        # 3. Build system prompt with injected context and invoke LLM
        print(f"🤖 Preparing context and calling LLM...")
        ai_response = prepare_prompt_and_invoke_llm(
            user_query=message,
            texts=texts,
            images=images,
            tables=tables
        )
        
        # 4. Save AI message with citations to database
        # Store the AI's response along with citations
        
        print(f"💾 Saving AI message...")

        ai_message_result = supabase.table('messages').insert({
            "chat_id": chat_id,
            "content": ai_response,
            "role": "assistant",
            "clerk_id": clerk_id,
            "citations": citations
        }).execute()
        
        ai_message = ai_message_result.data[0]
        print(f"✅ AI message saved: {ai_message['id']}") # type: ignore
        
        # 5. Return data
        return {
            "message": "Messages sent successfully",
            "data": {
                "userMessage": user_message,
                "aiMessage": ai_message
            }
        }
        
    except Exception as e:
        print(f"❌ Error in send_message: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

