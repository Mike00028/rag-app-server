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
load_dotenv()
router = APIRouter(tags=['chats'])
class ChatCreateRequest(BaseModel):
    project_id: str
    title: str
class QueryVariations(BaseModel):
    queries: List[str] = Field(..., description="The variations of the query")


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
    

def load_project_settings(project_id: str) -> dict:
    """Load project settings from database"""
    print(f"⚙️ Fetching project settings...")
    settings_result = supabase.table('project_settings').select('*').eq('project_id', project_id).execute()
    
    if not settings_result.data:
        raise HTTPException(status_code=404, detail="Project settings not found")
    
    settings = settings_result.data[0]
    print(f"✅ Settings retrieved")
    return settings # type: ignore

def get_document_ids(project_id: str) -> list[str]:
    """Get all document IDs for a project"""
    print(f"📄 Fetching project documents...")
    documents_result = supabase.table('project_documents').select('id').eq('project_id', project_id).execute()
    
    document_ids = [doc['id'] for doc in documents_result.data] # type: ignore
    print(f"✅ Found {len(document_ids)} documents")
    return document_ids #   type: ignore

def vector_search(query: str, document_ids: list[str], settings: dict) -> list[dict]:
    """Execute vector search"""
    query_embedding = embeddings_model.embed_query(query)
    
    result = supabase.rpc('vector_search_document_chunks', {
        'query_embedding': query_embedding,
        'filter_document_ids': document_ids,
        'match_threshold': settings['similarity_threshold'],
        'chunks_per_search': settings['chunks_per_search']
    }).execute()
    
    return result.data if result.data else [] # type: ignore
def rrf_rank_and_fuse(search_results_list: List[List[Dict]], weights: List[float] = None, k: int = 60) -> List[Dict]:
    """RRF (Reciprocal Rank Fusion) ranking"""
    if not search_results_list or not any(search_results_list):
        return []
    
    if weights is None:
        weights = [1.0 / len(search_results_list)] * len(search_results_list)
    
    chunk_scores = {}
    all_chunks = {}
    
    for search_idx, results in enumerate(search_results_list):
        weight = weights[search_idx]
        
        for rank, chunk in enumerate(results):
            chunk_id = chunk.get('id')
            if not chunk_id:
                continue
            
            rrf_score = weight * (1.0 / (k + rank + 1))
            
            if chunk_id in chunk_scores:
                chunk_scores[chunk_id] += rrf_score
            else:
                chunk_scores[chunk_id] = rrf_score
                all_chunks[chunk_id] = chunk
    
    sorted_chunk_ids = sorted(chunk_scores.keys(), key=lambda cid: chunk_scores[cid], reverse=True)
    return [all_chunks[chunk_id] for chunk_id in sorted_chunk_ids]

def hybrid_search(query: str, document_ids: List[str], settings: dict) -> List[Dict]:
    """Execute hybrid search by combining vector and keyword results"""
    # Get results from both search methods
    vector_results = vector_search(query, document_ids, settings)
    keyword_results = keyword_search(query, document_ids, settings)

    print(f"📈 Vector search returned: {len(vector_results)} chunks")
    print(f"📈 Keyword search returned: {len(keyword_results)} chunks")
    
    # Combine using RRF with configured weights
    return rrf_rank_and_fuse(
        [vector_results, keyword_results], 
        [settings['vector_weight'], settings['keyword_weight']]
    )


def keyword_search(query: str, document_ids: List[str], settings: dict) -> List[Dict]:
    """Execute keyword search"""
    result = supabase.rpc('keyword_search_document_chunks', {
        'query_text': query,
        'filter_document_ids': document_ids,
        'chunks_per_search': settings['chunks_per_search']
    }).execute()
    
    return result.data if result.data else [] # type: ignore


def generate_query_variations(original_query: str, num_queries: int = 3) -> List[str]:
    """Generate query variations using LLM with structured output"""
    system_prompt = f"""Generate 3 search variations for '{original_query}'. Return ONLY a JSON object matching this schema: {QueryVariations.model_json_schema()}
    """
    try:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Original query: {original_query}"),
        ]
        
        # Use structured output directly
        query_variations = llm.invoke(messages) or {}
        # 1. Clean the response to extract JSON
        json_str = re.sub(r'```json\n?|\n?```', '', query_variations.content).strip() #type: ignore

        # 2. Parse the JSON
        data = json.loads(json_str)

        # 3. Navigate the nested "properties" structure
        """response is in format properties: {queries: [list of queries]} """
        query_variations = data.get("properties", {}).get("queries", [])

        print(query_variations)
        # print(f"✅ Generated {len(query_variations['queries'])} query variations")
        # print(f"Queries: {query_variations['queries']}")

        return [original_query] + query_variations[:num_queries - 1]
    except Exception as e:
        print(f"❌ Query variation generation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return [original_query]
    
def build_context(chunks: List[Dict]) -> Tuple[List[str], List[str], List[str], List[Dict]]:
    """
    Returns:
        Tuple of (texts, images, tables, citations)
    """
    if not chunks:
        return [], [], [], []
    
    texts = []
    images = []
    tables = []
    citations = [] 
    
    # Batch fetch all filenames in ONE query
    doc_ids = [chunk['document_id'] for chunk in chunks if chunk.get('document_id')]
    unique_doc_ids: List[str] = list(set(doc_ids))  # ✅ Fixed syntax
    
    filename_map = {}
    
    if unique_doc_ids:
        result = supabase.table('project_documents')\
            .select('id, filename')\
            .in_('id', unique_doc_ids)\
            .execute()
        filename_map = {doc['id']: doc['filename'] for doc in result.data} # type: ignore
    
    # Process each chunk
    for chunk in chunks:
        original_content = chunk.get('original_content', {})
        
        # Extract content from chunk
        chunk_text = original_content.get('text', '')
        chunk_images = original_content.get('images', [])
        chunk_tables = original_content.get('tables', [])

        # Collect content
        if chunk_text:  # ✅ Add this check back
            texts.append(chunk_text)
        images.extend(chunk_images)
        tables.extend(chunk_tables)
        
        # Add citation for every chunk
        doc_id = chunk.get('document_id')
        if doc_id:
            citations.append({
                "chunk_id": chunk.get('id'),
                "document_id": doc_id,
                "filename": filename_map.get(doc_id, 'Unknown Document'),
                "page": chunk.get('page_number', 'Unknown')
            })
    
    return texts, images, tables, citations

def validate_context(texts: List[str], images: List[str], tables: List[str], citations: List[Dict]) -> None:
    """Validate and print context data in a readable format"""
    print("\n" + "="*80)
    print("📦 CONTEXT VALIDATION")
    print("="*80)
    
    # Texts - SHOW FULL TEXT
    print(f"\n📝 TEXTS: {len(texts)} chunks")
    for i, text in enumerate(texts, 1):
        print(f"\n{'='*80}")
        print(f"CHUNK [{i}] - {len(text)} characters")
        print(f"{'='*80}")
        print(text)  # ✅ Full text, no truncation
        print(f"{'='*80}\n")
    
    # Images
    print(f"\n🖼️  IMAGES: {len(images)}")
    for i, img in enumerate(images, 1):
        img_preview = str(img)[:60] + ('...' if len(str(img)) > 60 else '')
        print(f"  [{i}] {img_preview}")
    
    # Tables
    print(f"\n📊 TABLES: {len(tables)}")
    for i, table in enumerate(tables, 1):
        if isinstance(table, dict):
            rows = len(table.get('rows', []))
            cols = len(table.get('headers', []))
            print(f"  [{i}] {rows} rows × {cols} cols")
        else:
            print(f"  [{i}] Type: {type(table).__name__}")
    
    # Citations
    print(f"\n📚 CITATIONS: {len(citations)}")
    for i, cite in enumerate(citations, 1):
        chunk_id = cite['chunk_id'][:8] if cite.get('chunk_id') else 'N/A'
        print(f"  [{i}] {cite['filename']} (pg.{cite['page']}) | chunk: {chunk_id}...")
    
    # Summary
    total_chars = sum(len(text) for text in texts)
    print(f"\n{'='*80}")
    print(f"✅ Total: {len(texts)} texts ({total_chars:,} chars), {len(images)} images, {len(tables)} tables, {len(citations)} citations")
    print("="*80 + "\n")


def prepare_prompt_and_invoke_llm(
    user_query: str,
    texts: List[str],
    images: List[str],
    tables: List[str]
) -> str:
    """
    Builds system prompt with context and invokes LLM with multi-modal support
    
    Args:
        user_query: The user's question
        texts: List of text chunks from documents
        images: List of base64-encoded images
        tables: List of HTML table strings
    
    Returns:
        AI response string
    """
    # Build system prompt parts
    prompt_parts = []
    
    # Main instruction
    prompt_parts.append(
        "You are a helpful AI assistant that answers questions based solely on the provided context. "
        "Your task is to provide accurate, detailed answers using ONLY the information available in the context below.\n\n"
        "IMPORTANT RULES:\n"
        "- Only answer based on the provided context (texts, tables, and images)\n"
        "- If the answer cannot be found in the context, respond with: 'I don't have enough information in the provided context to answer that question.'\n"
        "- Do not use external knowledge or make assumptions beyond what's explicitly stated\n"
        "- When referencing information, be specific and cite relevant parts of the context\n"
        "- Synthesize information from texts, tables, and images to provide comprehensive answers\n\n"
    )
    
    # Add text contexts
    if texts:
        prompt_parts.append("=" * 80)
        prompt_parts.append("CONTEXT DOCUMENTS")
        prompt_parts.append("=" * 80 + "\n")
        
        for i, text in enumerate(texts, 1):
            prompt_parts.append(f"--- Document Chunk {i} ---")
            prompt_parts.append(text.strip())
            prompt_parts.append("")
    
    # Add tables if present
    if tables:
        prompt_parts.append("\n" + "=" * 80)
        prompt_parts.append("RELATED TABLES")
        prompt_parts.append("=" * 80)
        prompt_parts.append(
            "The following tables contain structured data that may be relevant to your answer. "
            "Analyze the table contents carefully.\n"
        )
        
        for i, table_html in enumerate(tables, 1):
            prompt_parts.append(f"--- Table {i} ---")
            prompt_parts.append(table_html)
            prompt_parts.append("")
    
    # Reference images if present
    if images:
        prompt_parts.append("\n" + "=" * 80)
        prompt_parts.append("RELATED IMAGES")
        prompt_parts.append("=" * 80)
        prompt_parts.append(
            f"{len(images)} image(s) will be provided alongside the user's question. "
            "These images may contain diagrams, charts, figures, formulas, or other visual information. "
            "Carefully analyze the visual content when formulating your response. "
            "The images are part of the retrieved context and should be used to answer the question.\n"
        )
    
    # Final instruction
    prompt_parts.append("=" * 80)
    prompt_parts.append(
        "Based on all the context provided above (documents, tables, and images), "
        "please answer the user's question accurately and comprehensively."
    )
    prompt_parts.append("=" * 80)
    
    system_prompt = "\n".join(prompt_parts)
    # Build messages for LLM
    messages: List[BaseMessage] = [SystemMessage(content=system_prompt)]
    
    # Create human message with user query and images
    if images:
        # Multi-modal message: text + images
        content_parts: List[Union[str, Dict[str, Any]]] = [{"type": "text", "text": user_query}]
        
        # Add each image to the content array
        for img_base64 in images:
            # Clean base64 string if it has data URI prefix
            if img_base64.startswith('data:image'):
                img_base64 = img_base64.split(',', 1)[1]
            
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
            })
        
        messages.append(HumanMessage(content=content_parts))
        messages.append(HumanMessage(content=content_parts))
    else:
        # Text-only message
        messages.append(HumanMessage(content=user_query))
    
    # Invoke LLM and return response
    print(f"Invoking LLM with {len(messages)} messages ({len(texts)} texts, {len(tables)} tables, {len(images)} images)...")
    response = llm.invoke(messages)
    
    # Handle both string and list content types
    if isinstance(response.content, str):
        return response.content
    elif isinstance(response.content, list):
        # If it's a list, join string elements and ignore non-string elements
        return " ".join(str(item) for item in response.content if isinstance(item, (str, dict)))
    else:
        return str(response.content)



class SendMessageRequest(BaseModel):
    content: str

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
        
        # 2. Load project settings
        # We need settings to know: chunk size, similarity threshold, etc.
        settings = load_project_settings(project_id)
        
        # 3. Get document IDs for this project
        # This narrows our search scope to only documents uploaded to this specific project
        document_ids = get_document_ids(project_id)
        #4 get rag_strategy
        rag_strategy = settings.get('rag_strategy', 'basic')
        print(f"🔍 Using RAG strategy: {rag_strategy}")
        chunks  = []
        if rag_strategy == 'basic':
            # 5. Perform basic/vector  search  
            chunks = hybrid_search(message, document_ids, settings)
            print(f"✅ Retrieved {len(chunks)} relevant chunks from vector search")
        elif rag_strategy == 'hybrid':
            chunks = hybrid_search(message, document_ids, settings)
            print(f"✅ Retrieved {len(chunks)} relevant chunks from hybrid search")

        elif rag_strategy == 'multi-query-vector':
            queries = generate_query_variations(
                message, settings["number_of_queries"]
            )
            print(f"Generated {len(queries)} query variations")

            all_chunks = []
            for index, query in enumerate(queries):
                chunks = vector_search(query, document_ids, settings)
                all_chunks.append(chunks)
                print(
                    f"Vector search for query {index+1}/{len(queries)}: {query} resulted in: {len(chunks)} chunks"
                )

            chunks = rrf_rank_and_fuse(all_chunks)
            print(f"RRF Fusion returned {len(chunks)} chunks from multi-query vector search")
         
        elif rag_strategy == 'multi-query-hybrid':
            queries = generate_query_variations(
                message, settings["number_of_queries"]
            )
            print(f"Generated {len(queries)} query variations")

            all_chunks = []
            for index, query in enumerate(queries):
                chunks = hybrid_search(query, document_ids, settings)
                all_chunks.append(chunks)
                print(
                    f"Multi vector Hybrid search for query {index+1}/{len(queries)}: {query} resulted in: {len(chunks)} chunks"
                )

            chunks = rrf_rank_and_fuse(all_chunks)
            print(f"RRF Fusion returned {len(chunks)} chunks from multi-query hybrid search")
            
        # 6. Build context from retrieved chunks
        # Format the retrieved chunks into a structured context with citations
        texts, images, tables, citations = build_context(chunks)
        validate_context(texts, images, tables, citations)
        
        # 7. Build system prompt with injected context
        # Add the retrieved document context to the system prompt so the LLM can answer based on the documents
        print(f"🤖 Preparing context and calling LLM...")
        ai_response = prepare_prompt_and_invoke_llm(
            user_query=message,
            texts=texts,
            images=images,
            tables=tables
        )
        
        # 8. Save AI message with citations to database
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
        
        # 4. Return data
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

