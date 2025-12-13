import os
import requests
from dotenv import load_dotenv
import boto3
from botocore.client import Config

# Load environment variables
load_dotenv()

# S3 Configuration
endpoint_url = os.getenv("AWS_ENDPOINT_URL_S3")
access_key = os.getenv("AWS_ACCESS_KEY_ID")
secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
region = os.getenv("AWS_REGION")
bucket_name = os.getenv("S3_BUCKET_NAME")

# Initialize S3 client
s3_client = boto3.client(
    's3',
    endpoint_url=endpoint_url,
    aws_access_key_id=access_key,
    aws_secret_access_key=secret_key,
    region_name=region,
    config=Config(
        signature_version='s3v4',
        s3={'addressing_style': 'path'}
    )
)

print("=== Testing Presigned URL Upload ===\n")

# Read the PDF file
print("1. Reading download.pdf...")
try:
    with open("download.pdf", "rb") as f:
        test_content = f.read()
    print(f"✓ File read successfully ({len(test_content)} bytes)\n")
except Exception as e:
    print(f"✗ Failed to read file: {str(e)}")
    exit(1)

# Generate presigned URL
test_key = "test/download.pdf"

print("2. Generating presigned URL...")
presigned_url = s3_client.generate_presigned_url(
    ClientMethod='put_object',
    Params={
        'Bucket': bucket_name,
        'Key': test_key,
        'ContentType': 'application/pdf',
        'ContentLength': len(test_content)
    },
    ExpiresIn=3600,
    HttpMethod='PUT'
)
print(f"✓ Presigned URL generated")
print(f"  URL: {presigned_url[:120]}...\n")

# Try to upload using the presigned URL
print("3. Uploading file using presigned URL...")
try:
    response = requests.put(
        presigned_url,
        data=test_content,
        headers={
            'Content-Type': 'application/pdf',
            'Content-Length': str(len(test_content))
        }
    )
    
    print(f"  Status Code: {response.status_code}")
    print(f"  Response: {response.text[:200] if response.text else 'Empty'}")
    
    if response.status_code == 200 or response.status_code == 204:
        print("✓ Upload successful!")
    else:
        print(f"✗ Upload failed with status {response.status_code}")
        
except Exception as e:
    print(f"✗ Upload failed: {str(e)}")

# Verify file exists
print("\n4. Verifying file exists...")
try:
    s3_client.head_object(Bucket=bucket_name, Key=test_key)
    print(f"✓ File exists in bucket!")
except Exception as e:
    print(f"✗ File not found: {str(e)}")

print("\n=== Test Complete ===")
