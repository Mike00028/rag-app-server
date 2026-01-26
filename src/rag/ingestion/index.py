from pickle import NONE
from time import time
from celery import Celery
from database import supabase
import os
import tempfile
from dotenv import load_dotenv
from src.services import s3
from src.config.index import appConfig
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.docx import partition_docx
from unstructured.partition.html import partition_html
from unstructured.partition.pptx import partition_pptx
from unstructured.partition.text import partition_text
from unstructured.partition.md import partition_md
from unstructured.chunking.title import chunk_by_title
from typing import Optional
from langchain_core.messages import HumanMessage
from src.services.llm import llm, embeddings_model

celery_app = Celery('DocumentProcessor', 
backend=appConfig["celery_result_backend"],
broker=appConfig["celery_broker_url"])
load_dotenv()

def update_document_status(document_id: str, status: str, details: Optional[dict] = None):
    """Update the processing status of the document in the database with optional details"""

    # get current document
    result = supabase.table('project_documents').select("processing_details").eq('id', document_id).execute()
    # start with existing details if any
    current_details = {}
    if result.data and result.data[0]["processing_details"]: # type: ignore
        processing_details = result.data[0]["processing_details"] # type: ignore
        if isinstance(processing_details, dict):
            current_details = processing_details
    
    if details:
        current_details.update(details)
    print(f"Updating document {document_id} status to {status} with details {details}")
    supabase.table('project_documents').update({"processing_status": status, "processing_details": current_details}).eq('id', document_id).execute()

@celery_app.task
def process_document(document_id: str, project_id: str, clerk_id: str):
    try:
        # Step1 download the document from S3 using the document_id or s3_key
        doc_result = supabase.table('project_documents').select('*').eq('id', document_id).eq('project_id', project_id).eq('clerk_id', clerk_id).execute()
        if not doc_result.data or len(doc_result.data) == 0:
            print(f"Document {document_id} not found for project {project_id} and user {clerk_id}")
            return False
        document = doc_result.data[0]
        source_type = document.get("source_type","file") #type: ignore
        update_document_status(document_id, "partitioning")
        elements = download_and_partition_document(document_id, document)
            
        # Step2 chunk elements
        chunks, chunking_metrics = chunk_elements(elements)
        update_document_status(document_id, "summarising", {"chunking": chunking_metrics})
        # Step3 summarise chunks
        #summarise only tables and images
        processed_chunks = summarize_chunks(chunks, document_id, source_type) # pyright: ignore[reportArgumentType]
        # Step4 generate embeddings and store
        update_document_status(document_id, 'vectorization')
        stored_chunk_ids = store_chunks_with_embeddings(document_id, processed_chunks)

        # Mark as completed
        update_document_status(document_id, 'completed')
        print(f"✅ Celery task completed for document: {document_id} with {len(stored_chunk_ids)} chunks")
        

        return {
                "status": "success", 
                "document_id": document_id
            }

    except Exception as e: 
        pass
 

def download_and_partition_document(document_id: str, document):
    """Download document from S3 and partition it into chunks./Crawl url and partition the content"""
    print(f"Downloading and partitioning document {document}")
    source_type = document.get("source_type","file")
    if source_type == "url":
      pass
    else:
        s3_key = document.get("s3_key")
        # download file from s3 using s3_key
        filename = document["filename"];
        print(f"Downloading document {document_id} from S3 key {s3_key} with filename {filename}")
        
        s3_key = document.get("s3_key")
        file_type = filename.split('.')[-1].lower()
        
        # Use system temp directory (works on Windows and Unix)
        temp_dir = tempfile.gettempdir()
        temp_file = os.path.join(temp_dir, f"{document_id}.{file_type}")
        
        BUCKET_NAME = os.getenv("S3_BUCKET_NAME","rag-app-bucket")
        
        # Download the file from s3 - it returns bytes
        file_content = s3.download_file(BUCKET_NAME, s3_key, temp_file) # type: ignore
        
        # Write the downloaded content to temp file
        with open(temp_file, 'wb') as f:
            f.write(file_content)
        
        elements = partion_document(temp_file, file_type, source_type="file")
        elements_summary= analyse_elements(elements)
        update_document_status(document_id, "chunking",{ "partitioning":{
"elements_found": elements_summary
        }})
        os.remove(temp_file)
        
        return elements
    
def partion_document(temp_file: str, file_type: str, source_type: str):
    """Partition document into chunks based on file type and source type"""
    if source_type == "url":
      pass
    if file_type == "pdf":
     return partition_pdf(filename=temp_file,
            strategy="hi_res",
                 infer_table_structure=True,
                 extract_image_block_types=["Image"],
                 extract_image_block_to_payload=True)

    elif file_type == 'docx':
        return partition_docx(
            filename=temp_file,
            strategy="hi_res",
            infer_table_structure=True
        )

    elif file_type == 'pptx':
        return partition_pptx(
            filename=temp_file,
            strategy="hi_res",
            infer_table_structure=True, 
        )

    elif file_type == "txt":
        return partition_text(
            filename=temp_file
        )
    
    elif file_type == "md":
        return partition_md(
            filename=temp_file
        )
    


    
def analyse_elements(elements):
    """ Count different types of elements in the partitioned document"""
    text_count =0
    table_count=0
    image_count=0
    title_count=0
    other_count=0
    for el in elements:
        el_name = type(el).__name__
        if el_name == "Text" or el_name == "NarrativeText":
            text_count +=1
        elif el_name == "Table":
            table_count +=1
        elif el_name == "Image":
            image_count +=1
        elif el_name == "Title":
            title_count +=1
        else:
            other_count +=1
    return {
        "text": text_count,
        "table": table_count,
        "image": image_count,
        "titles": title_count,
        "other": other_count
    }
def chunk_elements(elements):
    """Chunk partitioned elements using title-based chunking strategy and collect metrics"""
    print("Creating smart chunks...")
    chunks = chunk_by_title(elements, max_characters=3000, new_after_n_chars=2400, combine_text_under_n_chars=500)

    total_chunks = len(chunks)
    chunking_metrics ={
        "total_chunks": total_chunks
    }
    print(f"Created {total_chunks} chunks.")
    return chunks, chunking_metrics

def summarize_chunks(chunks, document_id, source_type="file"):
    """Transform chunks into searchable content with AI summaries"""
    print("🧠 Processing chunks with AI Summarisation...")
    
    processed_chunks = []
    total_chunks = len(chunks)
    
    for i, chunk in enumerate(chunks):
        current_chunk = i + 1
        
        # Update progress directly
        update_document_status(document_id, 'summarising', {
            "summarising": {
                "current_chunk": current_chunk,
                "total_chunks": total_chunks
            }
        })
        
        # Extract content from the chunk
        content_data = separate_content_types(chunk, source_type)

        # Debug prints
        print(f"     Types found: {content_data['types']}")
        print(f"     Tables: {len(content_data['tables'])}, Images: {len(content_data['images'])}")
        
        # Decide if we need AI summarisation
        if content_data['tables'] or content_data['images']:
            print(f"     Creating AI summary for mixed content...")
            enhanced_content = create_ai_summary( 
                content_data['text'], 
                content_data['tables'], 
                content_data['images']
            )
        else:
            enhanced_content = content_data['text']
        
        # Build the original_content structure
        original_content = {'text': content_data['text']}
        if content_data['tables']:
            original_content['tables'] = content_data['tables']
        if content_data['images']:
            original_content['images'] = content_data['images']
        
        # Create processed chunk with all data
        processed_chunk = {
            'content': enhanced_content,
            'original_content': original_content, 
            'type': content_data['types'],
            'page_number': get_page_number(chunk, i),
            'char_count': len(enhanced_content)
        }
        
        processed_chunks.append(processed_chunk)
    
    print(f"✅ Processed {len(processed_chunks)} chunks")
    return processed_chunks

def get_page_number(chunk, chunk_index):
    """Get page number from chunk or use fallback"""
    if hasattr(chunk, 'metadata'):
        page_number = getattr(chunk.metadata, 'page_number', None)
        if page_number is not None:
            return page_number
    
    # Fallback: use chunk index as page number
    return chunk_index + 1


def separate_content_types(chunk, source_type="file"):
    """Analyze what types of content are in a chunk"""
    is_url_source = source_type == 'url'
    
    content_data = {
        'text': chunk.text,
        'tables': [],
        'images': [],
        'types': ['text']
    }
    
    # Check for tables and images in original elements
    if hasattr(chunk, 'metadata') and hasattr(chunk.metadata, 'orig_elements'):
        for element in chunk.metadata.orig_elements:
            element_type = type(element).__name__
            
            # Handle tables
            if element_type == 'Table':
                content_data['types'].append('table')
                table_html = getattr(element.metadata, 'text_as_html', element.text)
                content_data['tables'].append(table_html)
            
            # Handle images (skip for URL sources)
            elif element_type == 'Image' and not is_url_source:
                if (hasattr(element, 'metadata') and 
                    hasattr(element.metadata, 'image_base64') and 
                    element.metadata.image_base64 is not None):
                    content_data['types'].append('image')
                    content_data['images'].append(element.metadata.image_base64)
    
    content_data['types'] = list(set(content_data['types']))
    return content_data


def create_ai_summary(text, tables_html, images_base64):
    """Create AI-enhanced summary for mixed content"""
    
    try:
        # Build the text prompt with more efficient instructions
        prompt_text = f"""Create a searchable index for this document content.

CONTENT:
{text}

"""
        
        # Add tables if present
        if tables_html:
            prompt_text += "TABLES:\n"
            for i, table in enumerate(tables_html):
                prompt_text += f"Table {i+1}:\n{table}\n\n"
        
        # More concise but effective prompt
        prompt_text += """
Generate a structured search index (aim for 250-400 words):

QUESTIONS: List 5-7 key questions this content answers (use what/how/why/when/who variations)

KEYWORDS: Include:
- Specific data (numbers, dates, percentages, amounts)
- Core concepts and themes
- Technical terms and casual alternatives
- Industry terminology

VISUALS (if images present):
- Chart/graph types and what they show
- Trends and patterns visible
- Key insights from visualizations

DATA RELATIONSHIPS (if tables present):
- Column headers and their meaning
- Key metrics and relationships
- Notable values or patterns

Focus on terms users would actually search for. Be specific and comprehensive.

SEARCH INDEX:"""
        
        # Build message content starting with the text prompt
        message_content = [{"type": "text", "text": prompt_text}]
        
        # Add images to the message
        for i, image_base64 in enumerate(images_base64):
            message_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
            }) # type: ignore
            print(f"🖼️ Image {i+1} included in summary request")
        
        message = HumanMessage(content=message_content) # type: ignore
        
        response = llm.invoke([message])
        
        return response.content
        
    except Exception as e:
        print(f" AI summary failed: {e}")


def store_chunks_with_embeddings(document_id: str, processed_chunks: list):
    """Generate embeddings and store chunks in one efficient operation"""
    print("Generating embeddings and storing chunks...")
    
    if not processed_chunks:
        print(" No chunks to process")
        return []
    
    # Step 1: Generate embeddings for all chunks
    print(f"Generating embeddings for {len(processed_chunks)} chunks...")
    
    # Extract content for embedding generation
    texts = [chunk_data['content'] for chunk_data in processed_chunks]
    
    # Generate embeddings in batches to avoid API limits
    batch_size = 10
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_embeddings = embeddings_model.embed_documents(batch_texts)
        all_embeddings.extend(batch_embeddings)
        print(f" ✅ Generated embeddings for batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
    
    # Step 2: Store chunks with embeddings
    print("Storing chunks with embeddings in database...")
    stored_chunk_ids = []
    
    for i, (chunk_data, embedding) in enumerate(zip(processed_chunks, all_embeddings)):
        try:
            # Add document_id, chunk_index, and embedding
            chunk_data_with_embedding = {
                **chunk_data,
                'document_id': document_id,
                'chunk_index': i,
                'embedding': embedding
            }
            
            result = supabase.table('document_chunks').insert(chunk_data_with_embedding).execute()
            if result.data and len(result.data) > 0:
                stored_chunk_ids.append(result.data[0]['id']) # type: ignore
            else:
                print(f"⚠️ Failed to insert chunk {i}: No data returned")
        except Exception as e:
            print(f"⚠️ Failed to insert chunk {i}: {e}")
            continue
    
    print(f"Successfully stored {len(processed_chunks)} chunks with embeddings")
    return stored_chunk_ids