import os
from openai import OpenAI
from langchain_ollama import ChatOllama, OllamaEmbeddings
from dotenv import load_dotenv

load_dotenv()

ENVIRONMENT = os.getenv("ENVIRONMENT", "local")  # default to local

# Create wrapper classes for Ollama models
class OllamaLLMWrapper:
    def __init__(self, model_name=None, temperature=0,):
        model = model_name or os.getenv("OLLAMA_LLM_MODEL", "gemma2:9b")
        try:
            self.model = ChatOllama(model=model, temperature=temperature)
            self.model_name = model
        except Exception as e:
            print(f"Warning: Failed to initialize Ollama LLM ({model}): {str(e)}")
            self.model = None
            self.model_name = model
    
    def invoke(self, messages):
        if self.model is None:
            raise Exception(f"Ollama LLM model ({self.model_name}) is not available")
        return self.model.invoke(messages)

class OllamaEmbeddingsWrapper:
    def __init__(self, model_name=None):
        embedding_model = model_name or os.getenv("OLLAMA_EMBEDDING_MODEL", "snowflake-arctic-embed:335m")
        try:
            self.model = OllamaEmbeddings(model=embedding_model)
            self.model_name = embedding_model
        except Exception as e:
            print(f"Warning: Failed to initialize Ollama Embeddings ({embedding_model}): {str(e)}")
            self.model = None
            self.model_name = embedding_model
    
    def embed_documents(self, texts):
        if self.model is None:
            raise Exception(f"Ollama Embeddings model ({self.model_name}) is not available")
        return self.model.embed_documents(texts)
    
    def embed_query(self, text):
        if self.model is None:
            raise Exception(f"Ollama Embeddings model ({self.model_name}) is not available")
        return self.model.embed_query(text)

# Initialize models based on environment
if ENVIRONMENT == "local":
    # Use Ollama models for local development
    print(f"🔧 Using Ollama models for local development")
    llm = OllamaLLMWrapper()
    vision_model = OllamaLLMWrapper()
    embeddings_model = OllamaEmbeddingsWrapper()
    print(f"✅ Ollama LLM: {llm.model_name}")
    print(f"✅ Ollama Embeddings: {embeddings_model.model_name}")
else:
    # Use Google Gemini for production
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY must be set for production environment")
    
    print(f"🚀 Using Google Gemini models for production")
    
    # Create OpenAI client for Gemini
    gemini_client = OpenAI(
        api_key=google_api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )
    
    # Create wrapper classes for Google Gemini models
    class GeminiLLM:
        def __init__(self, client):
            self.client = client
            self.model = "models/gemini-2.5-flash-lite"  # Low-cost model for efficiency
        
        def invoke(self, messages):
            # Convert LangChain messages to OpenAI format
            if isinstance(messages, list) and len(messages) > 0:
                openai_messages = []
                for msg in messages:
                    if hasattr(msg, 'content') and hasattr(msg, 'type'):
                        # LangChain message format
                        content = msg.content
                        
                        # Handle multimodal content (list with text and images)
                        if isinstance(content, list):
                            openai_messages.append({
                                "role": "user" if msg.type == 'human' else "system" if msg.type == 'system' else "assistant",
                                "content": content  # OpenAI API supports this format directly
                            })
                        else:
                            # Simple text content
                            if msg.type == 'human':
                                openai_messages.append({"role": "user", "content": content})
                            elif msg.type == 'system':
                                openai_messages.append({"role": "system", "content": content})
                            elif msg.type == 'ai':
                                openai_messages.append({"role": "assistant", "content": content})
                    else:
                        # Simple string message
                        openai_messages.append({"role": "user", "content": str(msg)})
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=openai_messages,
                    reasoning_effort="low"
                )
                
                # Create a simple response object that matches expected interface
                class SimpleResponse:
                    def __init__(self, content):
                        self.content = content
                
                return SimpleResponse(response.choices[0].message.content)
            return None
        
    class GeminiEmbeddings:
        def __init__(self, client):
            self.client = client
            self.model = "text-embedding-004"  # Specify the embedding model
        
        def embed_documents(self, texts):
            # Embed multiple documents
            embeddings = []
            for text in texts:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=text
                )
                embeddings.append(response.data[0].embedding)
            return embeddings
        
        def embed_query(self, text):
            # Embed a single query
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            return response.data[0].embedding
    
    # Google Gemini models
    llm = GeminiLLM(gemini_client)
    vision_model = GeminiLLM(gemini_client)  # Same model handles vision
    embeddings_model = GeminiEmbeddings(gemini_client)
    print(f"✅ Google Gemini models initialized")

# Ollama models with wrapper classes

