import uuid
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from src.auth import get_current_user
from database import supabase
from src.services.celery import celery_app, perform_rag_ingestion_task
from src.services.s3 import generate_presigned_upload_url, upload_file
router = APIRouter(tags=['files'])
class FileUploadRequest(BaseModel):
    file_name: str
    file_size: int
    file_type: str

@router.post("/api/projects/{project_id}/files/upload-url")
async def get_file_upload_url(project_id: str, file_request: FileUploadRequest, clerk_id: str = Depends(get_current_user)):
    try:
        # Verify that the project belongs to the authenticated user
        project = supabase.table('projects').select('*').eq('id', project_id).eq('clerk_id', clerk_id).execute()
        if not project.data or len(project.data) == 0:
            raise HTTPException(status_code=404, detail="Project not found or access denied")
        
        # generate unique s3key for the file
        file_extension = file_request.file_name.split('.')[-1] in file_request.file_name and file_request.file_name.split('.')[-1] or ''
        uuid_str = str(uuid.uuid4())
        s3_key = f"projects/{project_id}/documents/{uuid_str}.{file_extension}"
       
        upload_url = generate_presigned_upload_url(
            "put_object",
            Params= {
                "Key": s3_key,
                "ContentType": file_request.file_type
            },
           ExpiresIn=3600
        )
        document_result = supabase.table('project_documents').insert({
            "project_id": project_id,
            "filename": file_request.file_name,
            "s3_key": s3_key,
            "file_type": file_request.file_type,
            "file_size": file_request.file_size,
            "clerk_id": clerk_id,
            "processing_status": "uploading"
        }).execute()

        if not document_result.data or len(document_result.data) == 0:
            raise HTTPException(status_code=500, detail="Failed to create document record")
        return {
            "success": True,
            "message": "Upload URL generated successfully",
            "data": {
                "upload_url": upload_url,
                "s3_key": s3_key,
                "document": document_result.data[0]
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate upload URL: {str(e)}")

@router.post("/api/test-s3-upload")
def test_s3_upload():
    """Test endpoint to verify S3 upload functionality"""
    try:
        # Create test file content
        test_content = b"This is a test file for S3 upload verification."
        test_filename = f"test/test-{uuid.uuid4()}.txt"
        
        # Upload the file
        file_key = upload_file(
            file_content=test_content,
            file_name=test_filename,
            content_type="text/plain"
        )
        
        return {
            "success": True,
            "message": "Test file uploaded successfully",
            "data": {
                "file_key": file_key,
                "test_content": test_content.decode()
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"S3 upload test failed: {str(e)}")
    
@router.post("/api/projects/{project_id}/files/confirm")
async def confirm_file_upload(project_id: str, request: dict, clerk_id: str = Depends(get_current_user)):
    try:
        if 's3_key' not in request:
            raise HTTPException(status_code=400, detail="s3_key is required in the request body")
        s3_key = request['s3_key']

        # Verify that the project belongs to the authenticated user
        project = supabase.table('projects').select('*').eq('id', project_id).eq('clerk_id', clerk_id).execute()
        if not project.data or len(project.data) == 0:
            raise HTTPException(status_code=404, detail="Project not found or access denied")
        
        # Update document status to 'uploaded'
        update_result = supabase.table('project_documents').update({
            "processing_status": "queued"
        }).eq('project_id', project_id).eq('s3_key', s3_key).eq('clerk_id', clerk_id).execute()
        
        if not update_result.data or len(update_result.data) == 0:
            raise HTTPException(status_code=404, detail="Document not found or access denied")
        
        # Type-safe access to document data
        document_data = update_result.data[0]
        if not isinstance(document_data, dict) or 'id' not in document_data:
            raise HTTPException(status_code=500, detail="Invalid document data structure")
        # start background processing task here
        document_id= document_data['id']
        task_id = perform_rag_ingestion_task.delay(document_id, project_id, clerk_id)
        supabase.table('project_documents').update({"task_id": task_id.id}).eq('id', document_id).execute()
        return {
            "success": True,
            "message": "File upload confirmed successfully, processing queued for the document with celery",
            "data": update_result.data[0]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to confirm file upload: {str(e)}")
    
class URLAddRequest(BaseModel):
    url: str

@router.post("/api/projects/{project_id}/urls")
async def add_website_url(project_id: str, request: URLAddRequest, clerk_id: str = Depends(get_current_user)):
    try:
        url = request.url.strip()
        if not url.startswith("http://") and not url.startswith("https://"):
            raise HTTPException(status_code=400, detail="Invalid URL format. URL must start with http:// or https://")
        # Verify that the project belongs to the authenticated user
        project = supabase.table('projects').select('*').eq('id', project_id).eq('clerk_id', clerk_id).execute()
        if not project.data or len(project.data) == 0:
            raise HTTPException(status_code=404, detail="Project not found or access denied")
        
        # Insert the website URL as a document
        insert_result = supabase.table('project_documents').insert({
            "project_id": project_id,
            "filename": url,
            "s3_key": "", # No S3 key for URL type
            "file_type": "text/html",
            "file_size": 0,
            "clerk_id": clerk_id,
            "processing_status": "queued",
            "source_url": url,
            "source_type": "url"
        }).execute()
        
        if not insert_result.data or len(insert_result.data) == 0:
            raise HTTPException(status_code=500, detail="Failed to add website URL as document")
        
        return {
            "success": True,
            "message": "Website URL added successfully as document",
            "data": insert_result.data[0]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add website URL: {str(e)}")
    
@router.delete("/api/projects/{project_id}/files/{document_id}")
async def delete_project_file(project_id: str, document_id: str, clerk_id: str = Depends(get_current_user)):
    try:
        # Verify that the project belongs to the authenticated user
        project = supabase.table('projects').select('*').eq('id', project_id).eq('clerk_id', clerk_id).execute()
        if not project.data or len(project.data) == 0:
            raise HTTPException(status_code=404, detail="Project not found or access denied")
        
        # Verify that the document belongs to the project and user
        document = supabase.table('project_documents').select('*').eq('id', document_id).eq('project_id', project_id).eq('clerk_id', clerk_id).execute()
        if not document.data or len(document.data) == 0:
            raise HTTPException(status_code=404, detail="Document not found or access denied")
        
        # Delete the document record
        delete_result = supabase.table('project_documents').delete().eq('id', document_id).eq('project_id', project_id).eq('clerk_id', clerk_id).execute()
        if not delete_result.data or len(delete_result.data) == 0:
            raise HTTPException(status_code=500, detail="Failed to delete document")
        
        file_record = document.data[0]
        s3_key = file_record['s3_key'] # type: ignore
        if s3_key and isinstance(s3_key, str):
            # Optionally, delete the file from S3 as well
            from src.services.s3 import delete_file
            try:
                delete_file(s3_key)
            except Exception as e:
                print(f"Warning: Failed to delete file from S3: {str(e)}")

        return {
            "success": True,
            "message": "Document deleted successfully",
            "data": delete_result.data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")
    
@router.get("/api/projects/{project_id}/files/{file_id}/chunks")
async def get_document_chunks(
    project_id: str,
    file_id: str,
    clerk_id: str = Depends(get_current_user)
):
    try:
        project_result = supabase.table('projects').select('id').eq('id', project_id).eq('clerk_id', clerk_id).execute()
        
        if not project_result.data:
            raise HTTPException(status_code=404, detail="Project not found or access denied")
        
        doc_result = supabase.table('project_documents').select('id').eq('id', file_id).eq('project_id', project_id).execute()
        
        if not doc_result.data:
            raise HTTPException(status_code=404, detail="Document not found")
        
        chunks_result = supabase.table('document_chunks').select('*').eq('document_id', file_id).order('chunk_index').execute()
        
        return {
            "message": "Document chunks retrieved successfully",
            "data": chunks_result.data or []
        }

    except Exception as e:
        print(f"ERROR getting chunks: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get document chunks: {str(e)}")

        