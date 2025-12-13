from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from database import supabase
from src.auth import get_current_user

router = APIRouter(
    tags=["projects"]
)

class ProjectCreate(BaseModel):
    name: str
    description: Optional[str] = None

@router.get("/api/projects") 
def get_projects(clerk_id: str = Depends(get_current_user)): 
    """
    Retrieve all projects for the authenticated user
    """
    try:
        result = supabase.table('projects').select('*').eq('clerk_id', clerk_id).execute()
 
        return { 
            "success": True,
            "message": "Projects retrieved successfully",
            "data": result.data
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get projects: {str(e)}")


@router.post("/api/projects")
def create_project(project: ProjectCreate, clerk_id: str = Depends(get_current_user)):
    try:
        new_project = {
            "name": project.name,
            "description": project.description,
            "clerk_id": clerk_id
        }
        # Insert the new project into the database
        result = supabase.table('projects').insert(new_project).execute()
        if not result.data or len(result.data) == 0:
            raise HTTPException(status_code=500, detail="Project creation failed")
        
        # when we create a project, we should also create project settings with default values with project_id
        created_project = result.data[0]
        project_settings = {
            "project_id": created_project['id'], # type: ignore
             "embedding_model": "embeddinggemma:300m",
            "rag_strategy": "basic",
            "agent_type": "agentic",
            "chunks_per_search": 10,
            "final_context_size": 5,
            "similarity_threshold": 0.3,
            "number_of_queries": 5,
            "reranking_enabled": True,
            "reranking_model": "bge-reranker-v2-gemma", # model form cohere
            "vector_weight": 0.7,
            "keyword_weight": 0.3
        }
        settings_result = supabase.table('project_settings').insert(project_settings).execute()
        if not settings_result.data:
            supabase.table('projects').delete().eq('id', created_project['id']).execute() # type: ignore
            raise HTTPException(status_code=500, detail="Failed to create project settings")

        return {
            "success": True,
            "message": "Project and project settings created successfully",
            "data": result.data
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create project: {str(e)}")
    

@router.delete("/api/projects/{project_id}")
def delete_project(project_id: str, clerk_id: str = Depends(get_current_user)):
    try:
        # Verify that the project belongs to the authenticated user
        project = supabase.table('projects').select('*').eq('id', project_id).eq('clerk_id', clerk_id).execute()
        if not project.data or len(project.data) == 0:
            raise HTTPException(status_code=404, detail="Project not found or access denied")
        
        # Delete the project
        
        project_result = supabase.table('projects').delete().eq('id', project_id).execute()
        if not project_result.data:
            raise HTTPException(status_code=500, detail="Failed to delete project")

        return {
            "success": True,
            "message": "Project deleted successfully",
            "data": project_result.data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete project: {str(e)}")
    
@router.get("/api/projects/{project_id}")
def get_project(project_id: str, clerk_id: str = Depends(get_current_user)):
    try:
        project = supabase.table('projects').select('*').eq('id', project_id).eq('clerk_id', clerk_id).execute()
        if not project.data or len(project.data) == 0:
            raise HTTPException(status_code=404, detail="Project not found or access denied")
        
        return {
            "success": True,
            "message": "Project retrieved successfully",
            "data": project.data[0]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get project: {str(e)}")
    
@router.get("/api/projects/{project_id}/chats")
def get_project_chats(project_id: str, clerk_id: str = Depends(get_current_user)):
    try:
        # Verify that the project belongs to the authenticated user
        project = supabase.table('projects').select('*').eq('id', project_id).eq('clerk_id', clerk_id).execute()
        if not project.data or len(project.data) == 0:
            raise HTTPException(status_code=404, detail="Project not found or access denied")
        
        chats = supabase.table('chats').select('*').eq('project_id', project_id).execute()
        
        return {
            "success": True,
            "message": "Chats retrieved successfully",
            "data": chats.data or []
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get chats: {str(e)}")   

@router.get("/api/projects/{project_id}/settings")
def get_project_settings(project_id: str, clerk_id: str = Depends(get_current_user)):
    try:
        # Verify that the project belongs to the authenticated user
        project = supabase.table('projects').select('*').eq('id', project_id).eq('clerk_id', clerk_id).execute()
        if not project.data or len(project.data) == 0:
            raise HTTPException(status_code=404, detail="Project not found or access denied")
        
        settings = supabase.table('project_settings').select('*').eq('project_id', project_id).execute()
        if not settings.data or len(settings.data) == 0:
            raise HTTPException(status_code=404, detail="Project settings not found")
        
        return {
            "success": True,
            "message": "Project settings retrieved successfully",
            "data": settings.data[0]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get project settings: {str(e)}")

@router.get("/api/projects/{project_id}/files")
def get_project_documents(project_id: str, clerk_id: str = Depends(get_current_user)):
    try:
        # Verify that the project belongs to the authenticated user
        project = supabase.table('projects').select('*').eq('id', project_id).eq('clerk_id', clerk_id).execute()
        if not project.data or len(project.data) == 0:
            raise HTTPException(status_code=404, detail="Project not found or access denied")
        
        documents = supabase.table('project_documents').select('*').eq('project_id', project_id).execute()
        
        return {
            "success": True,
            "message": "Project documents retrieved successfully",
            "data": documents.data or []  # return empty list if no documents
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get project documents: {str(e)}")

    
