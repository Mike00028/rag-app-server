import boto3
from botocore.client import Config
from src.config.index import appConfig
import os

# Initialize S3 client
s3_client = boto3.client(
    's3',
    endpoint_url=appConfig.get("aws_endpoint_url_s3"),
    aws_access_key_id=appConfig.get("aws_access_key_id"),
    aws_secret_access_key=appConfig.get("aws_secret_access_key"),
    region_name=appConfig.get("aws_region"),
    config=Config(
        signature_version='s3v4',
        s3={'addressing_style': 'path'}
    )
)

bucket_name = appConfig.get("s3_bucket_name")

def upload_file(file_content: bytes, file_name: str, content_type: str = "application/octet-stream") -> str:
    """
    Upload a file to S3 bucket
    
    Args:
        file_content: The file content as bytes
        file_name: The name/key for the file in S3
        content_type: The MIME type of the file
    
    Returns:
        The file key/path in S3
    """
    try:
        s3_client.put_object(
            Bucket=bucket_name,
            Key=file_name,
            Body=file_content,
            ContentType=content_type
        )
        return file_name
    except Exception as e:
        raise Exception(f"Failed to upload file to S3: {str(e)}")

def download_file(file_key: str) -> bytes:
    """
    Download a file from S3 bucket
    
    Args:
        file_key: The key/path of the file in S3
    
    Returns:
        The file content as bytes
    """
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
        return response['Body'].read()
    except Exception as e:
        raise Exception(f"Failed to download file from S3: {str(e)}")

def delete_file(file_key: str) -> bool:
    """
    Delete a file from S3 bucket
    
    Args:
        file_key: The key/path of the file in S3
    
    Returns:
        True if successful
    """
    try:
        s3_client.delete_object(Bucket=bucket_name, Key=file_key)
        return True
    except Exception as e:
        raise Exception(f"Failed to delete file from S3: {str(e)}")

def get_file_url(file_key: str, expiration: int = 3600) -> str:
    """
    Generate a presigned URL for downloading a file
    
    Args:
        file_key: The key/path of the file in S3
        expiration: URL expiration time in seconds (default: 1 hour)
    
    Returns:
        Presigned URL for the file
    """
    try:
        url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket_name, 'Key': file_key},
            ExpiresIn=expiration
        )
        return url
    except Exception as e:
        raise Exception(f"Failed to generate presigned URL: {str(e)}")

def generate_presigned_upload_url(client_method: str, Params: dict, ExpiresIn: int = 3600) -> str:
    """
    Generate a presigned URL for uploading a file
    
    Args:
        client_method: The S3 client method (e.g., 'put_object')
        Params: Dictionary containing Key, ContentType, and optionally ContentLength
        ExpiresIn: URL expiration time in seconds (default: 1 hour)
    
    Returns:
        Presigned URL string for uploading
    """
    try:
        # Add bucket to params
        params = {'Bucket': bucket_name, **Params}
        
        url = s3_client.generate_presigned_url(
            ClientMethod=client_method,
            Params=params,
            ExpiresIn=ExpiresIn,
            HttpMethod='PUT'
        )
        return url
    except Exception as e:
        raise Exception(f"Failed to generate presigned upload URL: {str(e)}")

def list_files(prefix: str = "") -> list:
    """
    List files in S3 bucket with optional prefix
    
    Args:
        prefix: Optional prefix to filter files
    
    Returns:
        List of file keys
    """
    try:
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        if 'Contents' in response:
            return [obj['Key'] for obj in response['Contents']]
        return []
    except Exception as e:
        raise Exception(f"Failed to list files from S3: {str(e)}")

def create_bucket_if_not_exists():
    """
    Create the S3 bucket if it doesn't exist
    """
    try:
        s3_client.head_bucket(Bucket=bucket_name)
        print(f"Bucket '{bucket_name}' already exists")
    except:
        try:
            s3_client.create_bucket(Bucket=bucket_name)
            print(f"Bucket '{bucket_name}' created successfully")
        except Exception as e:
            print(f"Failed to create bucket: {str(e)}")
