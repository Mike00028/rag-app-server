import uuid
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from src.auth import get_current_user
from database import supabase
from src.services.s3 import generate_presigned_upload_url, upload_file
router = APIRouter(tags=['files'])
class FileUploadRequest(BaseModel):
    file_name: str
    file_size: int
    file_type: str

@router.post("/api/projects/{project_id}/files/upload-url")
def get_file_upload_url(project_id: str, file_request: FileUploadRequest, clerk_id: str = Depends(get_current_user)):
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
def confirm_file_upload(project_id: str, request: dict, clerk_id: str = Depends(get_current_user)):
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
        
        # start background processing task here
        return {
            "success": True,
            "message": "File upload confirmed successfully, processing queued for the document with celery",
            "data": update_result.data[0]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to confirm file upload: {str(e)}")